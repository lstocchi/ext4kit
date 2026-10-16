//! Extracted core filesystem operations.

use super::{Ext4ImageBuilder, directory::PosixMetadata};
use crate::ext4::*;
use crate::xattr::prepare_xattrs;
use anyhow::Result;

impl Ext4ImageBuilder {
    /// Creates and links a special inode; returns its inode number.
    // Create special inodes (devices, FIFOs, sockets)
    // which do not occupy file extent blocks.
    fn add_special_file(
        &mut self,
        parent: u32,
        name: &str,
        kind: u16,
        default_perms: u16,
        dir_file_type: u8,
        i_block_payload: [u8; 60],
        meta: &PosixMetadata,
    ) -> Result<u32> {
        Self::validate_name(name)?;
        let expansion_blocks = self.dir_expansion_blocks_required(parent, name)?;

        let xattrs = prepare_xattrs(&meta.xattrs)?;
        // it finds the next free inode starting from the block of the parent directory, but still do not allocate it
        let ino = self.next_free_inode(Some(parent))?;
        // it gets the group of the free inode found
        let (group, index_in_group) = self.inode_location(ino);
        // it ensures that the group has enough capacity to allocate the new inode
        self.ensure_allocation_capacity(expansion_blocks, u32::from(!xattrs.is_empty()))?;
        // allocate the inode
        self.mark_inode_used(ino);

        let mut inode = Ext4Inode::default();
        Self::populate_inode_metadata(&mut inode, meta, kind, default_perms);
        inode.i_size_lo = 0;
        inode.i_links_count = 1;
        inode.i_blocks_lo = 0;
        inode.i_flags = 0; // Special files do not use EXT4_EXTENTS_FL
        inode.i_block = i_block_payload;

        if !xattrs.is_empty() {
            inode.i_file_acl_lo = self.write_prepared_xattr_block(group, &xattrs)?;
            inode.i_blocks_lo += (BLOCK_SIZE / 512) as u32;
        }

        self.groups[group].inode_table[index_in_group] = inode;
        self.add_dir_entry(parent, name, ino, dir_file_type)?;
        Ok(ino)
    }

    /// Creates a character-device node and returns its inode number (e.g. /dev/null).
    pub fn add_char_device(
        &mut self,
        parent: u32,
        name: &str,
        major: u32,
        minor: u32,
        meta: &PosixMetadata,
    ) -> Result<u32> {
        let payload = Self::encode_device_number(major, minor)?;
        self.add_special_file(parent, name, S_IFCHR, 0o660, EXT4_FT_CHRDEV, payload, meta)
    }

    /// Creates a block-device node and returns its inode number (e.g. /dev/sda).
    pub fn add_block_device(
        &mut self,
        parent: u32,
        name: &str,
        major: u32,
        minor: u32,
        meta: &PosixMetadata,
    ) -> Result<u32> {
        let payload = Self::encode_device_number(major, minor)?;
        self.add_special_file(parent, name, S_IFBLK, 0o660, EXT4_FT_BLKDEV, payload, meta)
    }

    /// Creates a FIFO/named pipe and returns its inode number.
    pub fn add_fifo(&mut self, parent: u32, name: &str, meta: &PosixMetadata) -> Result<u32> {
        self.add_special_file(parent, name, S_IFIFO, 0o644, EXT4_FT_FIFO, [0u8; 60], meta)
    }

    /// Creates a Unix-domain socket node and returns its inode number.
    pub fn add_socket(&mut self, parent: u32, name: &str, meta: &PosixMetadata) -> Result<u32> {
        self.add_special_file(parent, name, S_IFSOCK, 0o777, EXT4_FT_SOCK, [0u8; 60], meta)
    }

    /// Encodes device major/minor numbers into the 60-byte `i_block` payload
    /// following Linux kernel / e2fsprogs encoding rules.
    /// Encodes a Linux device number into an inode payload; returns the 60-byte payload or an error for invalid values.
    pub(crate) fn encode_device_number(major: u32, minor: u32) -> Result<[u8; 60]> {
        if major > 0x0fff || minor > 0x000f_ffff {
            return Err(anyhow::anyhow!(
                "device number ({major}, {minor}) exceeds Linux dev_t limits"
            ));
        }
        let mut block = [0u8; 60];
        if major < 256 && minor < 256 {
            let old_dev = ((major << 8) | minor) as u32;
            block[..4].copy_from_slice(&old_dev.to_le_bytes());
        } else {
            let new_dev = (minor & 0xff) | (major << 8) | ((minor & !0xff) << 12);
            block[4..8].copy_from_slice(&new_dev.to_le_bytes());
        }
        Ok(block)
    }
}
