use anyhow::Result;
use std::{
    collections::HashMap,
    fs::{File, OpenOptions},
    io::Write,
    ops::{Deref, DerefMut},
    path::{Path, PathBuf},
};

use memmap2::{MmapMut, MmapOptions};
use zerocopy::IntoBytes;

use crate::ext4::*;

mod allocation;
pub mod directory;
pub mod dirhash;
mod extent;
mod file;
mod htree;
mod path;
mod special;
mod symlink;
pub use directory::PosixMetadata;

// ============================================================================
// Ext4 Image Writer Engine
// ============================================================================

pub struct Ext4ImageBuilder {
    pub superblock: Ext4SuperBlock,
    pub group_descriptors: Vec<Ext4GroupDesc>,
    pub groups: Vec<Ext4BlockGroup>,
    pub disk_data: DiskData,
    /// Next bitmap positions to probe, avoiding a full bitmap rescan for
    /// every imported entry.
    pub(crate) next_block_hints: Vec<usize>,
    pub(crate) next_inode_hints: Vec<usize>,
    /// Cached names and append slots for linear directories only. Indexed
    /// directories deliberately bypass this cache and use htree lookup.
    pub(crate) directory_caches: HashMap<u32, directory::DirectoryCache>,
}

/// Mutable image bytes backed either by memory (primarily tests and library
/// compatibility) or by the destination file used for streamed image builds.
pub enum DiskData {
    Memory(Vec<u8>),
    File { mapping: MmapMut, path: PathBuf },
}

impl Deref for DiskData {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        match self {
            Self::Memory(data) => data,
            Self::File { mapping, .. } => mapping,
        }
    }
}

impl DerefMut for DiskData {
    fn deref_mut(&mut self) -> &mut Self::Target {
        match self {
            Self::Memory(data) => data,
            Self::File { mapping, .. } => mapping,
        }
    }
}

impl Ext4ImageBuilder {
    /// Creates an in-memory ext4 image builder of `size_mb`; returns initialized filesystem state.
    pub fn new(size_mb: usize) -> Self {
        let image_size_bytes = size_mb * 1024 * 1024;
        Self::new_with_storage(
            image_size_bytes,
            DiskData::Memory(vec![0; image_size_bytes]),
        )
    }

    /// Creates an image directly in `output_path`. The mapping is file-backed,
    /// so imported payload pages can be flushed instead of retaining a second
    /// complete image-sized heap allocation.
    pub fn new_file(size_mb: usize, output_path: impl AsRef<Path>) -> std::io::Result<Self> {
        let image_size_bytes = size_mb
            .checked_mul(1024 * 1024)
            .ok_or_else(|| std::io::Error::other("image size overflow"))?;
        let path = output_path.as_ref().to_owned();
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&path)?;
        file.set_len(image_size_bytes as u64)?;
        // the file remains at its fixed length for the lifetime of the
        // mapping, and all access is mediated through the uniquely owned MmapMut.
        let mapping = unsafe { MmapOptions::new().map_mut(&file)? };
        Ok(Self::new_with_storage(
            image_size_bytes,
            DiskData::File { mapping, path },
        ))
    }

    fn new_with_storage(image_size_bytes: usize, disk_data: DiskData) -> Self {
        if BYTES_PER_INODE_RATIO < BLOCK_SIZE {
            panic!("BYTES_PER_INODE_RATIO must be greater than or equal to BLOCK_SIZE");
        }

        let blocks_number = image_size_bytes / BLOCK_SIZE;

        let block_groups_number = blocks_number.div_ceil(BLOCKS_PER_GROUP);

        let inodes_per_group = Self::inodes_per_group(blocks_number, block_groups_number);

        // Re-calculate actual total inodes based on rounded group count
        let total_inodes = inodes_per_group * block_groups_number;

        let mut superblock = Ext4SuperBlock::default();
        // Calculate a random seed for the directory-index hashes
        let mut hash_seed_bytes = [0u8; 16];
        getrandom::fill(&mut hash_seed_bytes)
            .expect("the operating system must provide random bytes for the ext4 hash seed");
        superblock.s_hash_seed = std::array::from_fn(|word| {
            u32::from_le_bytes(hash_seed_bytes[word * 4..word * 4 + 4].try_into().unwrap())
        });
        superblock.s_inodes_count = total_inodes as u32;
        superblock.s_blocks_count_lo = blocks_number as u32;
        superblock.s_free_blocks_count_lo = blocks_number as u32; // Adjusted in init_layout
        superblock.s_free_inodes_count = total_inodes as u32; // Adjusted in init_layout
        superblock.s_log_block_size = 2; // 4KB block size
        superblock.s_blocks_per_group = BLOCKS_PER_GROUP as u32;
        superblock.s_clusters_per_group = BLOCKS_PER_GROUP as u32;
        superblock.s_inodes_per_group = inodes_per_group as u32;

        // Build groups and group descriptor arrays
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

            groups.push(Ext4BlockGroup {
                group_id: group_idx as u32,
                blocks_count: group_blocks as u32,
                // The on-disk inode/block bitmaps occupie one complete block.
                // During init, the unused bits will be padded with 1s
                block_bitmap: vec![0u8; BLOCKS_PER_GROUP / 8],
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
            disk_data,
            next_block_hints: vec![0; block_groups_number],
            next_inode_hints: vec![0; block_groups_number],
            directory_caches: HashMap::new(),
        };

        builder.init_layout();
        builder
    }

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

    /// Marks one physical block allocated and updates accounting; returns no value.
    fn mark_block_used(&mut self, group_idx: usize, block: u32) {
        let byte_idx = (block / 8) as usize;
        let bit_idx = block % 8;

        self.groups[group_idx].block_bitmap[byte_idx] |= 1 << bit_idx;
        self.groups[group_idx].free_blocks_count -= 1;
        self.group_descriptors[group_idx].bg_free_blocks_count_lo -= 1;
        self.superblock.s_free_blocks_count_lo -= 1;
        self.next_block_hints[group_idx] = self.next_block_hints[group_idx].max(block as usize + 1);
    }

    /// Marks one inode allocated and updates accounting; returns no value.
    pub(crate) fn mark_inode_used(&mut self, inode: u32) {
        let (group, index_in_group) = self.inode_location(inode);
        let byte_idx = (index_in_group / 8) as usize;
        let bit_idx = index_in_group % 8;

        self.groups[group].inode_bitmap[byte_idx] |= 1 << bit_idx;
        self.groups[group].free_inodes_count -= 1;
        self.group_descriptors[group].bg_free_inodes_count_lo -= 1;
        self.superblock.s_free_inodes_count -= 1;
        self.next_inode_hints[group] =
            self.next_inode_hints[group].max(index_in_group as usize + 1);
    }

    /// Releases an allocated inode and restores its allocation accounting.
    /// The caller is responsible for releasing any blocks owned by the inode.
    pub(crate) fn release_inode(&mut self, inode: u32) -> Result<()> {
        if !self.is_inode_allocated(inode) {
            return Err(anyhow::anyhow!("inode {inode} is not allocated"));
        }
        let (group, index_in_group) = self.inode_location(inode);
        let byte_idx = index_in_group / 8;
        let bit_idx = index_in_group % 8;

        self.groups[group].inode_bitmap[byte_idx] &= !(1 << bit_idx);
        self.groups[group].free_inodes_count += 1;
        self.group_descriptors[group].bg_free_inodes_count_lo += 1;
        self.superblock.s_free_inodes_count += 1;
        self.next_inode_hints[group] = self.next_inode_hints[group].min(index_in_group);
        self.groups[group].inode_table[index_in_group] = Ext4Inode::default();
        Ok(())
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
    fn init_layout(&mut self) {
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

    /// Reserves a contiguous extent near `preferred_group`, beginning at the
    /// group's allocation cursor and wrapping to reclaimed blocks if needed.
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
            if u32::from(self.groups[group].free_blocks_count) < blocks_needed {
                continue;
            }
            let limit = self.groups[group].blocks_count as usize;
            let hint = self.next_block_hints[group].min(limit);

            // Directories and extent-tree metadata overwhelmingly request one
            // block. Search 64 bitmap bits at a time instead of interpreting
            // every allocated bit individually.
            if blocks_needed == 1 {
                let bitmap = &self.groups[group].block_bitmap;
                if let Some(block) = find_next_free_bit(bitmap, hint, limit)
                    .or_else(|| find_next_free_bit(bitmap, 0, hint))
                {
                    self.groups[group].block_bitmap[block / 8] |= 1 << (block % 8);
                    self.groups[group].free_blocks_count -= 1;
                    self.group_descriptors[group].bg_free_blocks_count_lo -= 1;
                    self.superblock.s_free_blocks_count_lo -= 1;
                    self.next_block_hints[group] = block + 1;
                    return Ok((group * BLOCKS_PER_GROUP + block) as u32);
                }
                continue;
            }

            let mut candidate_start = hint;
            let mut contiguous_free = 0;
            for bit in (hint..limit).chain(0..hint) {
                if bit == 0 {
                    contiguous_free = 0;
                }
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
                        self.next_block_hints[group] = candidate_start + blocks_needed as usize;
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

    /// Verifies that every block in `[start, start + len)` lies inside the
    /// filesystem and is marked allocated.
    pub(crate) fn validate_allocated_run(&self, start: u32, len: u32) -> Result<()> {
        for block in u64::from(start)..u64::from(start) + u64::from(len) {
            let group = block as usize / BLOCKS_PER_GROUP;
            let local = block as usize % BLOCKS_PER_GROUP;
            if group >= self.groups.len() || local >= self.groups[group].blocks_count as usize {
                return Err(anyhow::anyhow!("block {block} is outside the filesystem"));
            }
            if self.groups[group].block_bitmap[local / 8] & (1 << (local % 8)) == 0 {
                return Err(anyhow::anyhow!("block {block} is not allocated"));
            }
        }
        Ok(())
    }

    /// Releases a run of allocated blocks (ext2fs_block_alloc_stats_range),
    /// updating accounting and rewinding each group's allocation cursor.
    pub(crate) fn release_blocks(&mut self, start: u32, len: u32) -> Result<()> {
        self.validate_allocated_run(start, len)?;
        for block in start..start + len {
            let group = block as usize / BLOCKS_PER_GROUP;
            let local = block as usize % BLOCKS_PER_GROUP;
            self.groups[group].block_bitmap[local / 8] &= !(1 << (local % 8));
            self.groups[group].free_blocks_count += 1;
            self.group_descriptors[group].bg_free_blocks_count_lo += 1;
            self.superblock.s_free_blocks_count_lo += 1;
            self.next_block_hints[group] = self.next_block_hints[group].min(local);
        }
        Ok(())
    }

    /// Releases one allocated block.
    pub(crate) fn release_block(&mut self, block: u32) -> Result<()> {
        self.release_blocks(block, 1)
    }

    /// Prove that a file extent plus any separately allocated metadata block
    /// can be allocated before changing the block bitmap.
    /// Verifies enough data and metadata blocks remain; returns `Ok(())` or an exhaustion error.
    pub(crate) fn ensure_allocation_capacity(
        &self,
        extent_blocks: u32,
        metadata_blocks: u32,
    ) -> Result<()> {
        let total_needed = extent_blocks
            .checked_add(metadata_blocks)
            .ok_or_else(|| anyhow::anyhow!("block allocation overflow"))?;

        let free_blocks = self.superblock.s_free_blocks_count_lo;
        if free_blocks < total_needed {
            return Err(anyhow::anyhow!(
                "not enough free blocks: needed {total_needed}, available {}",
                free_blocks
            ));
        }
        Ok(())
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
        }
        Ok(())
    }

    /// Calculate maximum metadata blocks required for an extent tree.
    /// Calculates the maximum external extent-tree blocks for leaf extent count; returns that count.
    pub fn max_extent_tree_blocks(num_extents: usize) -> usize {
        if num_extents <= EXT4_EXTENTS_IN_INODE {
            return 0;
        }

        // External leaf blocks (depth 0)
        let leaf_blocks = num_extents.div_ceil(EXT4_EXTENTS_PER_BLOCK);
        let mut total_metadata_blocks = leaf_blocks;

        // Index blocks (depth 1+)
        let mut current_index_entries = leaf_blocks;
        while current_index_entries > EXT4_EXTENTS_IN_INODE {
            let index_blocks = current_index_entries.div_ceil(EXT4_INDEXES_PER_BLOCK);
            total_metadata_blocks += index_blocks;
            current_index_entries = index_blocks;
        }

        total_metadata_blocks
    }

    /// Serializes an extent tree into the inode and external blocks.
    /// Returns the number of external tree metadata blocks allocated.
    /// Serializes an inode's extent tree; returns the number of external tree blocks allocated.
    pub(crate) fn write_extent_tree(
        &mut self,
        preferred_group: usize,
        inode: &mut Ext4Inode,
        leaf_extents: &[Ext4Extent],
    ) -> Result<u32> {
        // Depth 0: Extents fit entirely within the inode's 60-byte i_block
        if leaf_extents.len() <= EXT4_EXTENTS_IN_INODE {
            let header = Ext4ExtentHeader {
                eh_magic: EXT4_EH_MAGIC,
                eh_entries: leaf_extents.len() as u16,
                eh_max: EXT4_EXTENTS_IN_INODE as u16,
                eh_depth: 0,
                eh_generation: 0,
            };
            inode.i_block[..12].copy_from_slice(header.as_bytes());
            for (i, ext) in leaf_extents.iter().enumerate() {
                let off = 12 + i * 12;
                inode.i_block[off..off + 12].copy_from_slice(ext.as_bytes());
            }
            return Ok(0);
        }

        // Depth >= 1: Build bottom-up
        let mut metadata_blocks_count = 0u32;
        let mut current_indices = Vec::new();

        // Level 0: External leaf blocks containing Ext4Extent
        for chunk in leaf_extents.chunks(EXT4_EXTENTS_PER_BLOCK) {
            let meta_blk = self.allocate_free_blocks(preferred_group, 1)?;
            metadata_blocks_count += 1;

            let mut block_data = vec![0u8; BLOCK_SIZE];
            let header = Ext4ExtentHeader {
                eh_magic: EXT4_EH_MAGIC,
                eh_entries: chunk.len() as u16,
                eh_max: EXT4_EXTENTS_PER_BLOCK as u16,
                eh_depth: 0,
                eh_generation: 0,
            };
            block_data[..12].copy_from_slice(header.as_bytes());
            for (i, ext) in chunk.iter().enumerate() {
                let off = 12 + i * 12;
                block_data[off..off + 12].copy_from_slice(ext.as_bytes());
            }

            let disk_offset = meta_blk as usize * BLOCK_SIZE;
            self.disk_data[disk_offset..disk_offset + BLOCK_SIZE].copy_from_slice(&block_data);

            current_indices.push(Ext4ExtentIdx {
                ei_block: chunk[0].ee_block,
                ei_leaf_lo: meta_blk,
                ei_leaf_hi: 0,
                ei_unused: 0,
            });
        }

        let mut current_depth = 1u16;

        // Level 1+: External index blocks containing Ext4ExtentIdx
        while current_indices.len() > EXT4_EXTENTS_IN_INODE {
            let mut next_level_indices = Vec::new();
            for chunk in current_indices.chunks(EXT4_INDEXES_PER_BLOCK) {
                let meta_blk = self.allocate_free_blocks(preferred_group, 1)?;
                metadata_blocks_count += 1;

                let mut block_data = vec![0u8; BLOCK_SIZE];
                let header = Ext4ExtentHeader {
                    eh_magic: EXT4_EH_MAGIC,
                    eh_entries: chunk.len() as u16,
                    eh_max: EXT4_INDEXES_PER_BLOCK as u16,
                    eh_depth: current_depth,
                    eh_generation: 0,
                };
                block_data[..12].copy_from_slice(header.as_bytes());
                for (i, idx) in chunk.iter().enumerate() {
                    let off = 12 + i * 12;
                    block_data[off..off + 12].copy_from_slice(idx.as_bytes());
                }

                let disk_offset = meta_blk as usize * BLOCK_SIZE;
                self.disk_data[disk_offset..disk_offset + BLOCK_SIZE].copy_from_slice(&block_data);

                next_level_indices.push(Ext4ExtentIdx {
                    ei_block: chunk[0].ei_block,
                    ei_leaf_lo: meta_blk,
                    ei_leaf_hi: 0,
                    ei_unused: 0,
                });
            }
            current_indices = next_level_indices;
            current_depth += 1;
        }

        // Write root index node into inode's i_block
        let header = Ext4ExtentHeader {
            eh_magic: EXT4_EH_MAGIC,
            eh_entries: current_indices.len() as u16,
            eh_max: EXT4_EXTENTS_IN_INODE as u16,
            eh_depth: current_depth,
            eh_generation: 0,
        };
        inode.i_block[..12].copy_from_slice(header.as_bytes());
        for (i, idx) in current_indices.iter().enumerate() {
            let off = 12 + i * 12;
            inode.i_block[off..off + 12].copy_from_slice(idx.as_bytes());
        }

        Ok(metadata_blocks_count)
    }
}

/// Finds the first zero bitmap bit in `[start, end)`, scanning full machine
/// words where possible while preserving ascending-bit allocation order.
fn find_next_free_bit(bitmap: &[u8], start: usize, end: usize) -> Option<usize> {
    let mut bit = start;
    while bit < end && !bit.is_multiple_of(64) {
        if bitmap[bit / 8] & (1 << (bit % 8)) == 0 {
            return Some(bit);
        }
        bit += 1;
    }

    while bit + 64 <= end {
        let byte = bit / 8;
        let word = u64::from_le_bytes(bitmap[byte..byte + 8].try_into().unwrap());
        if word != u64::MAX {
            return Some(bit + (!word).trailing_zeros() as usize);
        }
        bit += 64;
    }

    while bit < end {
        if bitmap[bit / 8] & (1 << (bit % 8)) == 0 {
            return Some(bit);
        }
        bit += 1;
    }
    None
}

#[cfg(test)]
mod bitmap_tests {
    use super::{DiskData, Ext4ImageBuilder, find_next_free_bit};
    use crate::ext4::{BLOCK_SIZE, INODE_SIZE};

    #[test]
    fn free_bit_search_respects_range_and_bit_order() {
        let mut bitmap = [u8::MAX; 24];
        bitmap[1] &= !(1 << 3); // bit 11
        bitmap[9] &= !(1 << 5); // bit 77
        bitmap[20] &= !(1 << 1); // bit 161

        assert_eq!(find_next_free_bit(&bitmap, 0, 192), Some(11));
        assert_eq!(find_next_free_bit(&bitmap, 12, 160), Some(77));
        assert_eq!(find_next_free_bit(&bitmap, 78, 192), Some(161));
        assert_eq!(find_next_free_bit(&bitmap, 12, 77), None);
    }

    #[test]
    fn file_backed_builder_flushes_without_an_image_sized_heap_buffer() -> anyhow::Result<()> {
        let path =
            std::env::temp_dir().join(format!("ext4kit-file-backed-{}.img", std::process::id()));
        let _ = std::fs::remove_file(&path);
        {
            let mut builder = Ext4ImageBuilder::new_file(16, &path)?;
            assert!(matches!(builder.disk_data, DiskData::File { .. }));
            builder.create_root_dir()?;
            builder.create_lost_and_found()?;
            builder.reserve_inodes();
            builder.write_to_file(path.to_str().unwrap())?;
        }
        assert_eq!(std::fs::metadata(&path)?.len(), 16 * 1024 * 1024);
        std::fs::remove_file(path)?;
        Ok(())
    }

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
