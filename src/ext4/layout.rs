//! Block-group geometry, the initial root and lost+found directories,
//! and serialization of all metadata held in memory into the image.

use super::{DiskData, Ext4ImageBuilder};
use crate::ext4::format::*;
use anyhow::Result;
use std::{fs::File, io::Write, path::Path};
use zerocopy::IntoBytes;

impl Ext4ImageBuilder {
    /// Inodes per group for an image of `blocks` blocks in `groups` groups.
    /// Like ext2fs_initialize(), the count is rounded up so the inode table
    /// fills its last block completely ("waste not, want not"), then kept a
    /// multiple of 8 for the inode bitmap.
    pub(crate) fn inodes_per_group(blocks: usize, groups: usize) -> usize {
        let target = blocks * BLOCK_SIZE / BYTES_PER_INODE_RATIO / groups;
        let inodes_per_block = BLOCK_SIZE / INODE_SIZE;
        let table_blocks = (target.max(16) * INODE_SIZE).div_ceil(BLOCK_SIZE);
        (table_blocks * inodes_per_block).min(BLOCKS_PER_GROUP) & !7
    }

    /// Number of blocks occupied by each group's inode table.
    pub(crate) fn inode_table_blocks(&self) -> usize {
        (self.superblock.s_inodes_per_group as usize * INODE_SIZE).div_ceil(BLOCK_SIZE)
    }

    /// Returns whether a block group contains a sparse-super backup.
    fn group_has_super(group_idx: usize) -> bool {
        if group_idx == 0 || group_idx == 1 {
            return true;
        }

        // Check if group_idx is a power of 3, 5, or 7
        [3, 5, 7].iter().any(|&base| {
            let mut p = base;
            while p < group_idx {
                p *= base;
            }
            p == group_idx
        })
    }

    /// Initializes per-group metadata placement and allocation maps; returns no value.
    pub(crate) fn init_layout(&mut self) {
        let block_groups_number = self.groups.len();

        // Calculate how many blocks the Group Descriptor Table (GDT) takes
        let gdt_bytes = block_groups_number * std::mem::size_of::<Ext4GroupDesc>();
        let gdt_blocks = gdt_bytes.div_ceil(BLOCK_SIZE);

        let inodes_per_group = self.superblock.s_inodes_per_group as usize;
        let itable_blocks = self.inode_table_blocks();

        for group_idx in 0..block_groups_number {
            // Absolute block offset where this group starts on disk
            let group_start_blk = (group_idx * BLOCKS_PER_GROUP) as u32;

            // Check if this group stores a primary or backup superblock/GDT
            let has_super = Self::group_has_super(group_idx);

            let mut current_local_blk: u32 = 0;
            if has_super {
                // Reserve Superblock (1 block)
                self.mark_block_used(group_idx, current_local_blk);
                current_local_blk += 1;

                // Reserve GDT blocks
                for _ in 0..gdt_blocks {
                    self.mark_block_used(group_idx, current_local_blk);
                    current_local_blk += 1;
                }
            }

            // Assign & Reserve Block Bitmap (1 block)
            let block_bitmap_abs = group_start_blk + current_local_blk;
            self.group_descriptors[group_idx].bg_block_bitmap_lo = block_bitmap_abs;
            self.mark_block_used(group_idx, current_local_blk);
            current_local_blk += 1;

            // Assign & Reserve Inode Bitmap (1 block)
            let inode_bitmap_abs = group_start_blk + current_local_blk;
            self.group_descriptors[group_idx].bg_inode_bitmap_lo = inode_bitmap_abs;
            self.mark_block_used(group_idx, current_local_blk);
            current_local_blk += 1;

            // Assign & Reserve Inode Table Range
            let itable_abs = group_start_blk + current_local_blk;
            self.group_descriptors[group_idx].bg_inode_table_lo = itable_abs;
            for _ in 0..itable_blocks {
                self.mark_block_used(group_idx, current_local_blk);
                current_local_blk += 1;
            }

            self.group_descriptors[group_idx].bg_free_inodes_count_lo = inodes_per_group as u16;

            // A bitmap block covers a full 32,768-block group. However last block could be smaller that that.
            // To avoid the OS from trying to read beyond the filesystem, we mark all blocks as used.
            for local_blk in self.groups[group_idx].blocks_count as usize..BLOCKS_PER_GROUP {
                self.groups[group_idx].block_bitmap[local_blk / 8] |= 1 << (local_blk % 8);
            }

            // Every inode bitmap is a full block too. e2fsck requires all
            // bits after s_inodes_per_group to be set as padding.
            for local_inode in inodes_per_group..BLOCKS_PER_GROUP {
                self.groups[group_idx].inode_bitmap[local_inode / 8] |= 1 << (local_inode % 8);
            }
        }
    }

    /// Serializes a group's superblock and descriptor table into the image; returns no value.
    fn write_superblock_and_gdt(&mut self, group_idx: usize) {
        let group_start = group_idx * BLOCKS_PER_GROUP * BLOCK_SIZE;
        let mut superblock = self.superblock;
        superblock.s_block_group_nr = group_idx as u16;

        let superblock_offset = group_start + 1024;
        self.disk_data
            [superblock_offset..superblock_offset + std::mem::size_of::<Ext4SuperBlock>()]
            .copy_from_slice(superblock.as_bytes());

        let gdt_offset = group_start + BLOCK_SIZE;
        for (i, desc) in self.group_descriptors.iter().enumerate() {
            let desc_bytes = desc.as_bytes();
            let start = gdt_offset + (i * std::mem::size_of::<Ext4GroupDesc>());
            self.disk_data[start..start + desc_bytes.len()].copy_from_slice(desc_bytes);
        }
    }

    /// Serializes all in-memory inode tables into the image buffer; returns no value.
    fn write_inode_table(&mut self) {
        for group_idx in 0..self.groups.len() {
            // Set the INODE_ZEROED flag on the group descriptor so Linux knows
            // the inode table blocks on disk do not contain uninitialized garbage.
            self.group_descriptors[group_idx].bg_flags |= EXT4_BG_INODE_ZEROED;

            let itable_start_block = self.group_descriptors[group_idx].bg_inode_table_lo as usize;
            let mut disk_offset = itable_start_block * BLOCK_SIZE;

            for inode in &self.groups[group_idx].inode_table {
                let inode_bytes = inode.as_bytes();
                self.disk_data[disk_offset..disk_offset + INODE_SIZE].copy_from_slice(inode_bytes);
                disk_offset += INODE_SIZE;
            }
        }
    }

    /// Initializes the ext4 root directory inode and entries; returns `Ok(())` or an allocation error.
    pub fn create_root_dir(&mut self) -> Result<()> {
        let root_data_block = self.allocate_free_blocks(0, 1)?;

        let mut dir_block = vec![0u8; BLOCK_SIZE];

        // Entry "."
        let dot_header = Ext4DirEntry2Header {
            inode: EXT4_ROOT_INO,
            rec_len: 12, // 8 byte header + 1 char '.' + 3 padding
            name_len: 1,
            file_type: EXT4_FT_DIR,
        };
        let dot_bytes = dot_header.as_bytes();
        dir_block[..8].copy_from_slice(dot_bytes);
        dir_block[8] = b'.';

        // Entry ".."
        let dot_dot_header = Ext4DirEntry2Header {
            inode: EXT4_ROOT_INO,
            rec_len: (BLOCK_SIZE - 12) as u16,
            name_len: 2,
            file_type: EXT4_FT_DIR,
        };
        let dot_dot_bytes = dot_dot_header.as_bytes();
        dir_block[12..20].copy_from_slice(dot_dot_bytes);
        dir_block[20..22].copy_from_slice(b"..");

        let disk_offset = (root_data_block as usize) * BLOCK_SIZE;
        self.disk_data[disk_offset..disk_offset + BLOCK_SIZE].copy_from_slice(&dir_block);

        // Create Root Inode (Inode #2, index 1 in group 0's inode_table)
        let mut root_inode = Ext4Inode {
            i_mode: S_IFDIR | 0o755,
            i_uid: 0,
            i_gid: 0,
            i_size_lo: BLOCK_SIZE as u32,
            i_links_count: 2,
            i_blocks_lo: BLOCK_SIZE as u32 / 512,
            i_flags: EXT4_EXTENTS_FL,
            ..Default::default()
        };

        // Extent Header for root inode
        let header = Ext4ExtentHeader {
            eh_magic: EXT4_EH_MAGIC,
            eh_entries: 1,
            eh_max: 4,
            eh_depth: 0,
            eh_generation: 0,
        };

        let extent = Ext4Extent {
            ee_block: 0,
            ee_len: 1,
            ee_start_hi: 0,
            ee_start_lo: root_data_block,
        };

        let header_bytes = header.as_bytes();
        let extent_bytes = extent.as_bytes();
        root_inode.i_block[..12].copy_from_slice(header_bytes);
        root_inode.i_block[12..24].copy_from_slice(extent_bytes);

        // save inode #2 in group 0's inode_table
        self.groups[0].inode_table[1] = root_inode;

        self.mark_inode_used(EXT4_ROOT_INO);

        self.group_descriptors[0].bg_used_dirs_count_lo += 1;

        Ok(())
    }

    /// Creates and links the conventional `lost+found` directory; returns `Ok(())` or an error.
    pub fn create_lost_and_found(&mut self) -> Result<()> {
        let lpf_inode_num = EXT4_FIRST_INO;

        let lpf_data_block = self.allocate_free_blocks(0, 1)?;

        let mut dir_block = vec![0u8; BLOCK_SIZE];

        // Entry "."
        let dot_header = Ext4DirEntry2Header {
            inode: lpf_inode_num,
            rec_len: 12, // 8 byte header + 1 char '.' + 3 padding
            name_len: 1,
            file_type: EXT4_FT_DIR,
        };
        let dot_bytes = dot_header.as_bytes();
        dir_block[..8].copy_from_slice(dot_bytes);
        dir_block[8] = b'.';

        // Entry ".."
        let dot_dot_header = Ext4DirEntry2Header {
            inode: EXT4_ROOT_INO,
            rec_len: (BLOCK_SIZE - 12) as u16,
            name_len: 2,
            file_type: EXT4_FT_DIR,
        };
        let dot_dot_bytes = dot_dot_header.as_bytes();
        dir_block[12..20].copy_from_slice(dot_dot_bytes);
        dir_block[20..22].copy_from_slice(b"..");

        let disk_offset = (lpf_data_block as usize) * BLOCK_SIZE;
        self.disk_data[disk_offset..disk_offset + BLOCK_SIZE].copy_from_slice(&dir_block);

        // Expand lost+found with empty directory blocks (up to 16KB = 4 blocks)
        let mut allocated_blocks = vec![lpf_data_block];
        for _ in 1..4 {
            let new_block = self.allocate_free_blocks(0, 1)?;
            allocated_blocks.push(new_block);

            // An empty expanded directory block has a single dummy header with inode = 0
            let mut empty_dir_block = vec![0u8; BLOCK_SIZE];
            let empty_header = Ext4DirEntry2Header {
                inode: 0, // 0 indicates an unused/deleted directory entry
                rec_len: BLOCK_SIZE as u16,
                name_len: 0,
                file_type: 0,
            };
            empty_dir_block[..8].copy_from_slice(empty_header.as_bytes());

            let disk_offset = (new_block as usize) * BLOCK_SIZE;
            self.disk_data[disk_offset..disk_offset + BLOCK_SIZE].copy_from_slice(&empty_dir_block);
        }

        let mut lpf_inode = Ext4Inode {
            i_mode: S_IFDIR | 0o700, // Restricted perms (drwx------)
            i_uid: 0,
            i_gid: 0,
            i_size_lo: (allocated_blocks.len() * BLOCK_SIZE) as u32, // 16KB total size
            i_links_count: 2,
            i_blocks_lo: (allocated_blocks.len() * 8) as u32, // 8 = BLOCK_SIZE / 512
            i_flags: EXT4_EXTENTS_FL,
            ..Default::default()
        };

        let header = Ext4ExtentHeader {
            eh_magic: EXT4_EH_MAGIC,
            eh_entries: 1,
            eh_max: 4,
            eh_depth: 0,
            eh_generation: 0,
        };

        let extent = Ext4Extent {
            ee_block: 0,
            ee_len: allocated_blocks.len() as u16,
            ee_start_hi: 0,
            ee_start_lo: lpf_data_block,
        };

        let header_bytes = header.as_bytes();
        let extent_bytes = extent.as_bytes();
        lpf_inode.i_block[..12].copy_from_slice(header_bytes);
        lpf_inode.i_block[12..24].copy_from_slice(extent_bytes);

        let local_inode_idx = lpf_inode_num - 1;
        self.groups[0].inode_table[local_inode_idx as usize] = lpf_inode;
        self.mark_inode_used(lpf_inode_num);

        self.add_dir_entry(EXT4_ROOT_INO, "lost+found", lpf_inode_num, EXT4_FT_DIR)
            .expect("lost+found must fit in the root directory");

        // update root inode's link count (due to lost+found/.. directory pointing to root)
        self.groups[0].inode_table[1].i_links_count += 1;
        self.group_descriptors[0].bg_used_dirs_count_lo += 1;

        Ok(())
    }

    /// Reserves ext4's fixed low-numbered inodes; returns no value.
    pub fn reserve_inodes(&mut self) {
        self.mark_inode_used(EXT4_BAD_INO);

        // Skip Root (#2) and lost+found (#11), mark remaining 3..=10
        for inode_num in (EXT4_ROOT_INO + 1)..EXT4_FIRST_INO {
            self.mark_inode_used(inode_num);
        }
    }

    /// Flushes Superblock and Group Descriptors into disk_data, then writes to file
    /// Finalizes metadata and writes the image to `output_path`; returns I/O success or failure.
    pub fn write_to_file(&mut self, output_path: &str) -> std::io::Result<()> {
        if matches!(self.disk_data, DiskData::ReadOnly(_)) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::PermissionDenied,
                "image was opened read-only",
            ));
        }
        self.commit_metadata();
        // Flush file-backed pages, or write the compatibility in-memory image.
        match &mut self.disk_data {
            DiskData::Memory(data) => {
                let mut file = File::create(output_path)?;
                file.write_all(data)?;
            }
            DiskData::File { mapping, path } => {
                if path != Path::new(output_path) {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidInput,
                        format!(
                            "file-backed image targets {}, not {output_path}",
                            path.display()
                        ),
                    ));
                }
                mapping.flush()?;
            }
            DiskData::ReadOnly(_) => unreachable!("rejected above"),
        }
        Ok(())
    }

    /// Serializes the inode tables, superblocks, group descriptors and
    /// bitmaps held in memory into `disk_data`.
    pub(crate) fn commit_metadata(&mut self) {
        // Commit Inode Tables
        self.write_inode_table();
        // Write the primary metadata and all sparse-super backups.  Each
        // backup superblock records the group where it resides.
        for group_idx in 0..self.groups.len() {
            if Self::group_has_super(group_idx) {
                self.write_superblock_and_gdt(group_idx);
            }
        }
        // Commit Block & Inode Bitmaps into disk_data
        for (i, group) in self.groups.iter().enumerate() {
            let bb_offset = (self.group_descriptors[i].bg_block_bitmap_lo as usize) * BLOCK_SIZE;
            self.disk_data[bb_offset..bb_offset + group.block_bitmap.len()]
                .copy_from_slice(&group.block_bitmap);
            let ib_offset = (self.group_descriptors[i].bg_inode_bitmap_lo as usize) * BLOCK_SIZE;
            self.disk_data[ib_offset..ib_offset + group.inode_bitmap.len()]
                .copy_from_slice(&group.inode_bitmap);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inode_table_fills_whole_blocks_like_ext2fs_initialize() {
        // 524 MiB has 134,144 blocks in 5 groups: a target of 26,828 inodes
        // per group, which previously left the last inode-table block half used.
        for size_mb in [16, 100, 524, 1000] {
            let builder = Ext4ImageBuilder::new(size_mb);
            let per_group = builder.superblock.s_inodes_per_group as usize;
            assert_eq!(per_group * INODE_SIZE % BLOCK_SIZE, 0, "{size_mb} MiB");
            assert_eq!(per_group % 8, 0);
            let total = builder.superblock.s_inodes_count as usize;
            assert_eq!(total, per_group * builder.groups.len());
        }
    }
}
