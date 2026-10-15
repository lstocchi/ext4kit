use anyhow::Result;
use std::{fs::File, io::Write};

use zerocopy::IntoBytes;

use crate::ext4::*;

// ============================================================================
// Ext4 Image Writer Engine
// ============================================================================

pub struct Ext4ImageBuilder {
    pub superblock: Ext4SuperBlock,
    pub group_descriptors: Vec<Ext4GroupDesc>,
    pub groups: Vec<Ext4BlockGroup>,
    pub disk_data: Vec<u8>,
}

impl Ext4ImageBuilder {
    pub fn new(size_mb: usize) -> Self {
        if BYTES_PER_INODE_RATIO < BLOCK_SIZE {
            panic!("BYTES_PER_INODE_RATIO must be greater than or equal to BLOCK_SIZE");
        }

        let image_size_bytes = size_mb * 1024 * 1024;
        let blocks_number = image_size_bytes / BLOCK_SIZE;

        let block_groups_number = blocks_number.div_ceil(BLOCKS_PER_GROUP);

        let target_inodes = image_size_bytes / BYTES_PER_INODE_RATIO;
        let mut inodes_per_group = target_inodes / block_groups_number;

        // Align inodes_per_group to a multiple of 8
        if inodes_per_group % 8 != 0 {
            inodes_per_group -= inodes_per_group % 8;
        }

        // Ensure minimum ext4 inode limit per group
        if inodes_per_group < 16 {
            inodes_per_group = 16;
        }

        // Re-calculate actual total inodes based on rounded group count
        let total_inodes = inodes_per_group * block_groups_number;

        let mut superblock = Ext4SuperBlock::default();
        superblock.s_inodes_count = total_inodes as u32;
        superblock.s_blocks_count_lo = blocks_number as u32;
        superblock.s_free_blocks_count_lo = blocks_number as u32; // Adjusted in init_layout
        superblock.s_free_inodes_count = total_inodes as u32; // Adjusted in init_layout
        superblock.s_log_block_size = 2; // 4KB block size
        superblock.s_blocks_per_group = BLOCKS_PER_GROUP as u32;
        superblock.s_clusters_per_group = BLOCKS_PER_GROUP as u32;
        superblock.s_inodes_per_group = inodes_per_group as u32;

        // 4. Build groups and group descriptor arrays
        let mut groups = Vec::with_capacity(block_groups_number);
        let mut group_descriptors = Vec::with_capacity(block_groups_number);

        for group_idx in 0..block_groups_number {
            let group_blocks = if group_idx == block_groups_number - 1 {
                let remainder = blocks_number % BLOCKS_PER_GROUP;
                if remainder != 0 {
                    remainder
                } else {
                    BLOCKS_PER_GROUP
                }
            } else {
                BLOCKS_PER_GROUP
            };

            let block_bitmap_bytes = BLOCKS_PER_GROUP / 8;
            groups.push(Ext4BlockGroup {
                group_id: group_idx as u32,
                blocks_count: group_blocks as u32,
                block_bitmap: vec![0u8; block_bitmap_bytes],
                // The on-disk inode bitmap occupies one complete block.  Keep
                // that entire block so its unused trailing bits can be set.
                inode_bitmap: vec![0u8; BLOCKS_PER_GROUP / 8],
                inode_table: vec![Ext4Inode::default(); inodes_per_group],
                free_blocks_count: group_blocks as u16,
                free_inodes_count: inodes_per_group as u16,
            });

            group_descriptors.push(Ext4GroupDesc {
                bg_block_bitmap_lo: 0,
                bg_inode_bitmap_lo: 0,
                bg_inode_table_lo: 0,
                bg_free_blocks_count_lo: group_blocks as u16,
                bg_free_inodes_count_lo: inodes_per_group as u16,
                bg_used_dirs_count_lo: 0,
                bg_flags: 0,
                bg_exclude_bitmap_lo: 0,
                bg_block_bitmap_csum_lo: 0,
                bg_inode_bitmap_csum_lo: 0,
                bg_itable_unused_lo: 0,
                bg_checksum: 0,
            });
        }

        let mut builder = Self {
            superblock,
            group_descriptors,
            groups,
            disk_data: vec![0u8; image_size_bytes],
        };

        builder.init_layout();
        builder
    }

    fn mark_block_used(&mut self, group_idx: usize, block: u32) {
        let byte_idx = (block / 8) as usize;
        let bit_idx = block % 8;

        self.groups[group_idx].block_bitmap[byte_idx] |= 1 << bit_idx;
        self.groups[group_idx].free_blocks_count -= 1;
        self.group_descriptors[group_idx].bg_free_blocks_count_lo -= 1;
        self.superblock.s_free_blocks_count_lo -= 1;
    }

    pub(crate) fn mark_inode_used(&mut self, inode: u32) {
        let (group, index_in_group) = self.inode_location(inode);
        let byte_idx = (index_in_group / 8) as usize;
        let bit_idx = index_in_group % 8;

        self.groups[group].inode_bitmap[byte_idx] |= 1 << bit_idx;
        self.groups[group].free_inodes_count -= 1;
        self.group_descriptors[group].bg_free_inodes_count_lo -= 1;
        self.superblock.s_free_inodes_count -= 1;
    }

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

    fn init_layout(&mut self) {
        let block_groups_number = self.groups.len();

        // Calculate how many blocks the Group Descriptor Table (GDT) takes
        let gdt_bytes = block_groups_number * std::mem::size_of::<Ext4GroupDesc>();
        let gdt_blocks = gdt_bytes.div_ceil(BLOCK_SIZE);

        let inodes_per_group = self.superblock.s_inodes_per_group as usize;
        let itable_blocks = (inodes_per_group * INODE_SIZE) / BLOCK_SIZE;

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

            // Every inode bitmap is a full block too.  e2fsck requires all
            // bits after s_inodes_per_group to be set as padding.
            for local_inode in inodes_per_group..BLOCKS_PER_GROUP {
                self.groups[group_idx].inode_bitmap[local_inode / 8] |= 1 << (local_inode % 8);
            }
        }
    }

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

    /// Reserve a contiguous extent, preferring the inode's block group and
    /// wrapping to later groups when necessary.
    pub(crate) fn allocate_free_blocks(
        &mut self,
        preferred_group: usize,
        blocks_needed: u32,
    ) -> Result<u32> {
        if blocks_needed == 0 {
            return Ok(0);
        }
        for delta in 0..self.groups.len() {
            let group = (preferred_group + delta) % self.groups.len();
            let limit = self.groups[group].blocks_count as usize;
            let mut candidate_start = 0;
            let mut contiguous_free = 0;
            for bit in 0..limit {
                // if block is free, if contiguous_free is 0, set candidate_start to bit, otherwise increment contiguous_free
                if self.groups[group].block_bitmap[bit / 8] & (1 << (bit % 8)) == 0 {
                    if contiguous_free == 0 {
                        candidate_start = bit;
                    }
                    contiguous_free += 1;
                    if contiguous_free == blocks_needed {
                        for allocated in candidate_start..candidate_start + blocks_needed as usize {
                            self.groups[group].block_bitmap[allocated / 8] |= 1 << (allocated % 8);
                        }
                        self.groups[group].free_blocks_count -= blocks_needed as u16;
                        self.group_descriptors[group].bg_free_blocks_count_lo -=
                            blocks_needed as u16;
                        self.superblock.s_free_blocks_count_lo -= blocks_needed;
                        return Ok((group * BLOCKS_PER_GROUP + candidate_start) as u32);
                    }
                } else {
                    contiguous_free = 0;
                }
            }
        }
        Err(anyhow::anyhow!(
            "no contiguous run of {blocks_needed} free blocks is available"
        ))
    }

    /// Prove that a file extent plus any separately allocated metadata block
    /// can be allocated before changing the block bitmap.
    pub(crate) fn ensure_allocation_capacity(
        &self,
        preferred_group: usize,
        extent_blocks: u32,
        metadata_blocks: u32,
    ) -> Result<()> {
        if self.superblock.s_free_blocks_count_lo < extent_blocks + metadata_blocks {
            return Err(anyhow::anyhow!("not enough free blocks"));
        }
        if extent_blocks == 0 {
            return Ok(());
        }
        // Check for a contiguous run of free blocks for the file extent.
        // Metadata blocks can be allocated anywhere.
        for delta in 0..self.groups.len() {
            let group = (preferred_group + delta) % self.groups.len();
            let mut run = 0u32;
            for bit in 0..self.groups[group].blocks_count as usize {
                // if group is not used, increment run
                if self.groups[group].block_bitmap[bit / 8] & (1 << (bit % 8)) == 0 {
                    run += 1;
                    if run == extent_blocks {
                        return Ok(());
                    }
                } else {
                    run = 0;
                }
            }
        }
        Err(anyhow::anyhow!(
            "no contiguous run of {extent_blocks} free blocks is available"
        ))
    }

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
        let mut root_inode = Ext4Inode::default();
        root_inode.i_mode = S_IFDIR | 0o755;
        root_inode.i_uid = 0;
        root_inode.i_gid = 0;
        root_inode.i_size_lo = BLOCK_SIZE as u32;
        root_inode.i_links_count = 2;
        root_inode.i_blocks_lo = BLOCK_SIZE as u32 / 512;
        root_inode.i_flags = EXT4_EXTENTS_FL;

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

        let mut lpf_inode = Ext4Inode::default();
        lpf_inode.i_mode = S_IFDIR | 0o700; // Restricted perms (drwx------)
        lpf_inode.i_uid = 0;
        lpf_inode.i_gid = 0;
        lpf_inode.i_size_lo = (allocated_blocks.len() * BLOCK_SIZE) as u32; // 16KB total size
        lpf_inode.i_links_count = 2;
        lpf_inode.i_blocks_lo = (allocated_blocks.len() * 8) as u32; // 8 = BLOCK_SIZE / 512
        lpf_inode.i_flags = EXT4_EXTENTS_FL;

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

    pub fn reserve_inodes(&mut self) {
        self.mark_inode_used(EXT4_BAD_INO);

        // Skip Root (#2) and lost+found (#11), mark remaining 3..=10
        for inode_num in (EXT4_ROOT_INO + 1)..EXT4_FIRST_INO {
            self.mark_inode_used(inode_num);
        }
    }

    /// Flushes Superblock and Group Descriptors into disk_data, then writes to file
    pub fn write_to_file(&mut self, output_path: &str) -> std::io::Result<()> {
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
        // Write out to disk
        let mut file = File::create(output_path)?;
        file.write_all(&self.disk_data)?;
        Ok(())
    }
}
