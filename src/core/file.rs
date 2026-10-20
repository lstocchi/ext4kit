//! Extracted core filesystem operations.

use super::{Ext4ImageBuilder, directory::PosixMetadata};
use crate::ext4::*;
use crate::xattr::prepare_xattrs;
use anyhow::Result;
use std::io::{Cursor, Read};

impl Ext4ImageBuilder {
    /// Creates a regular-file inode, writes `data`, links it under `parent`, and returns its inode number.
    pub fn add_file(
        &mut self,
        parent: u32,
        name: &str,
        data: &[u8],
        meta: &PosixMetadata,
    ) -> Result<u32> {
        self.add_file_from_reader(parent, name, Cursor::new(data), data.len() as u64, meta)
    }

    /// Creates a regular file while copying exactly `size` bytes from `reader`
    /// directly into its allocated inode number.
    pub fn add_file_from_reader(
        &mut self,
        parent: u32,
        name: &str,
        mut reader: impl Read,
        size: u64,
        meta: &PosixMetadata,
    ) -> Result<u32> {
        Self::validate_name(name)?;

        let file_blocks_needed = u32::try_from(size.div_ceil(BLOCK_SIZE as u64))
            .map_err(|_| anyhow::anyhow!("file is too large for ext4: {size} bytes"))?;
        let xattrs = prepare_xattrs(&meta.xattrs)?;

        // it finds the next free inode starting from the block of the parent directory, but still do not allocate it
        let file_inode = self.next_free_inode(Some(parent))?;
        // it gets the group of the free inode found
        let (group, index_in_group) = self.inode_location(file_inode);

        // it plans the data blocks to be allocated
        let target_block_ranges = self.plan_data_blocks(group, file_blocks_needed)?;

        // Extent count is known exactly from the target block ranges
        let exact_tree_meta_blocks = Self::max_extent_tree_blocks(target_block_ranges.len()) as u32;
        let xattr_blocks = u32::from(!xattrs.is_empty());
        let metadata_blocks_required = exact_tree_meta_blocks
            .checked_add(xattr_blocks)
            .ok_or_else(|| anyhow::anyhow!("block allocation count overflow"))?;

        // it ensures that the group has enough capacity to allocate the new inode
        let dir_growth_blocks = self.estimate_dir_expansion_blocks(parent, name)?;
        self.ensure_allocation_capacity(
            file_blocks_needed as u32 + dir_growth_blocks,
            metadata_blocks_required,
        )?;

        // allocate the inode and commit the data blocks
        self.mark_inode_used(file_inode);
        self.commit_data_blocks(&target_block_ranges);

        // Write payload data into the runs
        let mut data_offset = 0u64;
        for &(phys_start, len) in &target_block_ranges {
            if data_offset >= size {
                break;
            }
            let chunk_size = ((len as u64) * BLOCK_SIZE as u64).min(size - data_offset) as usize;
            let disk_offset = phys_start as usize * BLOCK_SIZE;
            reader
                .read_exact(&mut self.disk_data[disk_offset..disk_offset + chunk_size])
                .map_err(|error| {
                    anyhow::anyhow!("cannot read {size}-byte file payload: {error}")
                })?;
            data_offset += chunk_size as u64;
        }

        let mut leaf_extents = Vec::with_capacity(target_block_ranges.len());
        let mut current_logical = 0u32;
        for &(phys_start, len) in &target_block_ranges {
            leaf_extents.push(Ext4Extent {
                ee_block: current_logical,
                ee_len: len as u16,
                ee_start_hi: 0,
                ee_start_lo: phys_start,
            });
            current_logical += len;
        }

        let mut inode = Ext4Inode::default();
        Self::populate_inode_metadata(&mut inode, meta, S_IFREG, 0o644);
        inode.i_size_lo = size as u32;
        inode.i_size_high = (size >> 32) as u32;
        inode.i_links_count = 1;
        inode.i_flags = EXT4_EXTENTS_FL;

        let tree_meta_blocks = self.write_extent_tree(group, &mut inode, &leaf_extents)?;

        // Total 512-byte sectors: data + tree metadata blocks
        let mut total_sectors =
            (file_blocks_needed as u64 + tree_meta_blocks as u64) * (BLOCK_SIZE as u64 / 512);

        if !xattrs.is_empty() {
            inode.i_file_acl_lo = self.write_prepared_xattr_block(group, &xattrs)?;
            total_sectors += BLOCK_SIZE as u64 / 512;
        }

        inode.i_blocks_lo = total_sectors as u32;
        let high_sectors = (total_sectors >> 32) as u16;
        inode.i_osd2[0..2].copy_from_slice(&high_sectors.to_le_bytes());

        self.groups[group].inode_table[index_in_group] = inode;
        self.add_dir_entry(parent, name, file_inode, EXT4_FT_REG_FILE)?;

        Ok(file_inode)
    }
}
