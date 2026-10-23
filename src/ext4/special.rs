//! Special inodes: character and block devices, FIFOs and sockets.

use super::{
    Ext4ImageBuilder,
    directory::dir_entry_file_type,
    inode::{PosixMetadata, default_permissions},
};
use crate::ext4::format::xattr::prepare_xattrs;
use crate::ext4::format::*;
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
        i_block_payload: [u8; 60],
        meta: &PosixMetadata,
    ) -> Result<u32> {
        Self::validate_name(name)?;
        let (Some(default_perms), Some(dir_file_type)) =
            (default_permissions(kind), dir_entry_file_type(kind))
        else {
            anyhow::bail!("unsupported special file type {kind:#o}");
        };
        let expansion_blocks = self.estimate_dir_expansion_blocks(parent, name)?;

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
            inode.i_file_acl_lo = match self.write_prepared_xattr_block(group, &xattrs) {
                Ok(block) => block,
                Err(error) => return Err(self.rollback_created_runs(ino, &[], error)),
            };
            inode.i_blocks_lo += (BLOCK_SIZE / 512) as u32;
        }

        self.groups[group].inode_table[index_in_group] = inode;
        if let Err(error) = self.add_dir_entry(parent, name, ino, dir_file_type) {
            return Err(self.rollback_created_inode(ino, error));
        }
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
        self.add_special_file(parent, name, S_IFCHR, payload, meta)
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
        self.add_special_file(parent, name, S_IFBLK, payload, meta)
    }

    /// Creates a FIFO/named pipe and returns its inode number.
    pub fn add_fifo(&mut self, parent: u32, name: &str, meta: &PosixMetadata) -> Result<u32> {
        self.add_special_file(parent, name, S_IFIFO, [0u8; 60], meta)
    }

    /// Creates a Unix-domain socket node and returns its inode number.
    pub fn add_socket(&mut self, parent: u32, name: &str, meta: &PosixMetadata) -> Result<u32> {
        self.add_special_file(parent, name, S_IFSOCK, [0u8; 60], meta)
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
            let old_dev = (major << 8) | minor;
            block[..4].copy_from_slice(&old_dev.to_le_bytes());
        } else {
            let new_dev = (minor & 0xff) | (major << 8) | ((minor & !0xff) << 12);
            block[4..8].copy_from_slice(&new_dev.to_le_bytes());
        }
        Ok(block)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ext4::test_support::fresh_builder;

    #[test]
    fn char_and_block_device_nodes_encode_rdev() {
        let mut builder = fresh_builder().expect("failed to create builder");
        let metadata = PosixMetadata {
            mode: Some(0o666),
            ..Default::default()
        };

        // Character device with standard minor/major (< 256): /dev/null (1, 3)
        let chr_ino = builder
            .add_char_device(EXT4_ROOT_INO, "null", 1, 3, &metadata)
            .expect("failed to create char device");
        let (group, idx) = builder.inode_location(chr_ino);
        let chr = &builder.groups[group].inode_table[idx];

        let chr_mode = chr.i_mode;
        let chr_size = chr.i_size_lo;
        let chr_blocks = chr.i_blocks_lo;
        let chr_flags = chr.i_flags;
        let chr_block = chr.i_block;
        assert_eq!(chr_mode, S_IFCHR | 0o666);
        assert_eq!(chr_size, 0);
        assert_eq!(chr_blocks, 0);
        assert_eq!(chr_flags, 0); // Must NOT have EXT4_EXTENTS_FL
        let dev_lo = u32::from_le_bytes(chr_block[..4].try_into().unwrap());
        assert_eq!(dev_lo, (1 << 8) | 3);

        // Block device with high major/minor (>= 256): (259, 1)
        let blk_ino = builder
            .add_block_device(EXT4_ROOT_INO, "nvme0n1", 259, 1, &metadata)
            .expect("failed to create block device");
        let (b_group, b_idx) = builder.inode_location(blk_ino);
        let blk = &builder.groups[b_group].inode_table[b_idx];

        let blk_mode = blk.i_mode;
        let blk_size = blk.i_size_lo;
        let blk_blocks = blk.i_blocks_lo;
        let blk_flags = blk.i_flags;
        assert_eq!(blk_mode, S_IFBLK | 0o666);
        assert_eq!(blk_size, 0);
        assert_eq!(blk_blocks, 0);
        assert_eq!(blk_flags, 0);
        // High device numbers leave i_block[0] as 0 and set i_block[1]
        let dev_old = u32::from_le_bytes(blk.i_block[..4].try_into().unwrap());
        let dev_new = u32::from_le_bytes(blk.i_block[4..8].try_into().unwrap());
        assert_eq!(dev_old, 0);
        let expected_new = (1 & 0xff) | (259 << 8);
        assert_eq!(dev_new, expected_new);
    }

    #[test]
    fn fifo_and_socket_nodes() {
        let mut builder = fresh_builder().expect("failed to create builder");
        let metadata = PosixMetadata::default();

        let fifo_ino = builder
            .add_fifo(EXT4_ROOT_INO, "test_pipe", &metadata)
            .expect("failed to create fifo");
        let (f_grp, f_idx) = builder.inode_location(fifo_ino);
        let fifo = &builder.groups[f_grp].inode_table[f_idx];

        let fifo_mode = fifo.i_mode;
        let fifo_size = fifo.i_size_lo;
        let fifo_blocks = fifo.i_blocks_lo;
        let fifo_flags = fifo.i_flags;
        let fifo_block = fifo.i_block;
        assert_eq!(fifo_mode, S_IFIFO | 0o644);
        assert_eq!(fifo_size, 0);
        assert_eq!(fifo_blocks, 0);
        assert_eq!(fifo_flags, 0);
        assert_eq!(fifo_block, [0u8; 60]);

        let sock_ino = builder
            .add_socket(EXT4_ROOT_INO, "test.sock", &metadata)
            .expect("failed to create socket");
        let (s_grp, s_idx) = builder.inode_location(sock_ino);
        let sock = &builder.groups[s_grp].inode_table[s_idx];

        let sock_mode = sock.i_mode;
        let sock_size = sock.i_size_lo;
        let sock_blocks = sock.i_blocks_lo;
        let sock_flags = sock.i_flags;
        let sock_block = sock.i_block;
        assert_eq!(sock_mode, S_IFSOCK | 0o777);
        assert_eq!(sock_size, 0);
        assert_eq!(sock_blocks, 0);
        assert_eq!(sock_flags, 0);
        assert_eq!(sock_block, [0u8; 60]);
    }

    #[test]
    fn device_numbers_must_fit_linux_dev_t() {
        assert!(Ext4ImageBuilder::encode_device_number(0x1000, 0).is_err());
        assert!(Ext4ImageBuilder::encode_device_number(0, 0x10_0000).is_err());
    }
}
