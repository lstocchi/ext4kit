//! Extracted core filesystem operations.

use super::{Ext4ImageBuilder, directory::PosixMetadata};
use crate::ext4::*;
use crate::xattr::prepare_xattrs;
use anyhow::Result;
use zerocopy::IntoBytes;

impl Ext4ImageBuilder {
    /// Creates and links a symbolic link, returning its inode number. Targets shorter than the inode's 60-byte
    /// `i_block` field are stored inline; longer targets use one data block.
    pub fn add_symlink(
        &mut self,
        parent: u32,
        name: &str,
        target: &str,
        meta: &PosixMetadata,
    ) -> Result<u32> {
        Self::validate_name(name)?;
        let expansion_blocks = self.dir_expansion_blocks_required(parent, name)?;

        let target_bytes = target.as_bytes();
        // Match ext2fs_symlink(): the terminating NUL must fit in one block,
        // so an empty target and targets up to BLOCK_SIZE - 1 bytes are valid.
        if target_bytes.len() >= BLOCK_SIZE {
            return Err(anyhow::anyhow!(
                "symlink target must be no longer than {} bytes",
                BLOCK_SIZE - 1
            ));
        }

        let xattrs = prepare_xattrs(&meta.xattrs)?;

        // e2fsprogs defines fastlink as strictly less than i_block size (60 bytes)
        let is_fastlink = target_bytes.len() < 60;
        let block_needed = if is_fastlink { 0 } else { 1 };

        // it finds the next free inode starting from the block of the parent directory, but still do not allocate it
        let ino = self.next_free_inode(Some(parent))?;
        // it gets the group of the free inode found
        let (group, index_in_group) = self.inode_location(ino);
        // it ensures that the group has enough capacity to allocate the new inode
        self.ensure_allocation_capacity(
            block_needed as u32 + expansion_blocks,
            u32::from(!xattrs.is_empty()),
        )?;

        // allocate the inode
        self.mark_inode_used(ino);

        let mut inode = Ext4Inode::default();
        Self::populate_inode_metadata(&mut inode, meta, S_IFLNK, 0o777);
        inode.i_size_lo = target_bytes.len() as u32;
        inode.i_links_count = 1;

        if is_fastlink {
            // Fast Symlink: Write directly into the 60-byte i_block array
            inode.i_blocks_lo = 0;
            inode.i_block[..target_bytes.len()].copy_from_slice(target_bytes);
        } else {
            // Note: e2fsprogs attempts inline_data here if the feature is enabled.
            // Since this builder currently do not have inline_data support, we route
            // directly to the slow block-allocated fallback.
            let data_block = self.allocate_free_blocks(group, 1)?;
            let offset = data_block as usize * BLOCK_SIZE;
            // The e2fsprogs buffer is zero-initialized before copying the
            // target, which supplies the NUL terminator and a zero-filled tail.
            self.disk_data[offset..offset + BLOCK_SIZE].fill(0);
            self.disk_data[offset..offset + target_bytes.len()].copy_from_slice(target_bytes);
            inode.i_blocks_lo = (BLOCK_SIZE / 512) as u32;
            inode.i_flags = EXT4_EXTENTS_FL;
            Self::init_single_extent(&mut inode, 1, data_block);
        }

        if !xattrs.is_empty() {
            inode.i_file_acl_lo = self.write_prepared_xattr_block(group, &xattrs)?;
            inode.i_blocks_lo += (BLOCK_SIZE / 512) as u32;
        }

        self.groups[group].inode_table[index_in_group] = inode;
        self.add_dir_entry(parent, name, ino, EXT4_FT_SYMLINK)?;

        Ok(ino)
    }

    /// Initializes a depth-0 inline extent header with a single contiguous extent.
    /// Initializes `inode` with one depth-zero extent; returns no value because it mutates the inode in place.
    pub(crate) fn init_single_extent(inode: &mut Ext4Inode, blocks: u16, start: u32) {
        let extent_header = Ext4ExtentHeader {
            eh_magic: EXT4_EH_MAGIC,
            eh_entries: u16::from(blocks != 0),
            eh_max: 4,
            eh_depth: 0,
            eh_generation: 0,
        };
        inode.i_block[..12].copy_from_slice(extent_header.as_bytes());
        if blocks != 0 {
            let extent = Ext4Extent {
                ee_block: 0,
                ee_len: blocks,
                ee_start_hi: 0,
                ee_start_lo: start,
            };
            inode.i_block[12..24].copy_from_slice(extent.as_bytes());
        }
    }
}
