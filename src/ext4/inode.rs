//! Inode table access, inode metadata (mode, ownership, timestamps) and
//! external xattr blocks.

use super::Ext4ImageBuilder;
use crate::ext4::format::xattr::{PreparedXattr, encode_xattr_block, prepare_xattrs};
use crate::ext4::format::*;
use anyhow::Result;
use std::collections::{HashMap, HashSet};

/// Metadata collected from an OCI layer or a host filesystem.
#[derive(Clone, Debug, Default)]
pub struct PosixMetadata {
    /// Permission bits from the source; `None` selects the inode-type
    /// default.  `Some(0)` is valid POSIX metadata and is kept as is.
    pub mode: Option<u16>,
    pub uid: u32,
    pub gid: u32,
    pub atime: u32,
    pub ctime: u32,
    pub mtime: u32,
    pub xattrs: HashMap<String, Vec<u8>>,
}

/// Returns the permissions applied when an inode of `kind` has no source mode.
pub(crate) fn default_permissions(kind: u16) -> Option<u16> {
    Some(match kind {
        S_IFDIR => 0o755,
        S_IFREG | S_IFIFO => 0o644,
        S_IFLNK | S_IFSOCK => 0o777,
        S_IFCHR | S_IFBLK => 0o660,
        _ => return None,
    })
}

impl Ext4ImageBuilder {
    /// Maps an inode number to its block-group and zero-based table index; returns both indices.
    pub fn inode_location(&self, ino: u32) -> (usize, usize) {
        let inodes_per_group = self.superblock.s_inodes_per_group;
        (
            ((ino - 1) / inodes_per_group) as usize,
            ((ino - 1) % inodes_per_group) as usize,
        )
    }

    /// Returns the in-memory inode table record for `ino`.
    pub fn inode(&self, ino: u32) -> &Ext4Inode {
        let (group, index) = self.inode_location(ino);
        &self.groups[group].inode_table[index]
    }

    /// Returns a mutable inode table record for `ino`.
    pub(crate) fn inode_mut(&mut self, ino: u32) -> &mut Ext4Inode {
        let (group, index) = self.inode_location(ino);
        &mut self.groups[group].inode_table[index]
    }

    /// Returns whether `ino` is an allocated directory inode.
    pub fn is_directory(&self, ino: u32) -> bool {
        self.is_inode_allocated(ino) && self.inode(ino).is_dir()
    }

    /// Replaces an inode's ownership, permissions, timestamps, and xattrs.
    pub fn set_inode_metadata(&mut self, ino: u32, meta: &PosixMetadata) -> Result<()> {
        if !self.is_inode_allocated(ino) {
            return Err(anyhow::anyhow!("inode {ino} does not exist"));
        }
        let kind = self.inode(ino).file_type();
        let default_perms = default_permissions(kind)
            .ok_or_else(|| anyhow::anyhow!("inode {ino} has an unsupported type"))?;
        let xattrs = prepare_xattrs(&meta.xattrs)?;
        let (group, _) = self.inode_location(ino);
        // Write the replacement before freeing the old block so a failed
        // write leaves the inode's existing xattrs intact.
        let new_xattr = if xattrs.is_empty() {
            0
        } else {
            self.ensure_allocation_capacity(0, 1)?;
            self.write_prepared_xattr_block(group, &xattrs)?
        };
        let old_xattr = self.inode(ino).i_file_acl_lo;
        if old_xattr != 0 {
            self.release_block(old_xattr)?;
        }
        let sectors = (BLOCK_SIZE / 512) as u32;
        let inode = self.inode_mut(ino);
        inode.i_blocks_lo = inode.i_blocks_lo - sectors * u32::from(old_xattr != 0)
            + sectors * u32::from(new_xattr != 0);
        inode.i_file_acl_lo = new_xattr;
        Self::populate_inode_metadata(inode, meta, kind, default_perms);
        Ok(())
    }

    /// Replaces only an inode's ownership, like chown(2) by root.
    pub fn set_inode_owner(&mut self, ino: u32, uid: u32, gid: u32) -> Result<()> {
        if !self.is_inode_allocated(ino) {
            return Err(anyhow::anyhow!("inode {ino} does not exist"));
        }
        Self::populate_inode_owner(self.inode_mut(ino), uid, gid);
        Ok(())
    }

    /// Replaces only an inode's permission bits (including setuid, setgid
    /// and sticky), like chmod(2).  The file type, ownership, timestamps and
    /// xattrs are kept.
    pub fn set_inode_mode(&mut self, ino: u32, permissions: u16) -> Result<()> {
        if permissions & !0o7777 != 0 {
            return Err(anyhow::anyhow!(
                "mode {permissions:o} has bits outside the permission mask 7777"
            ));
        }
        if !self.is_inode_allocated(ino) {
            return Err(anyhow::anyhow!("inode {ino} does not exist"));
        }
        let inode = self.inode_mut(ino);
        inode.i_mode = (inode.i_mode & S_IFMT) | permissions;
        Ok(())
    }

    /// Sets the ownership of `root` and everything below it, like
    /// `chown -R -P`: symlinks are changed themselves and never followed, so
    /// the walk cannot leave the subtree.  Returns the number of inodes
    /// changed.  The subtree is validated before any owner changes.
    pub fn set_owner_recursive(&mut self, root: u32, uid: u32, gid: u32) -> Result<usize> {
        if !self.is_inode_allocated(root) {
            return Err(anyhow::anyhow!("inode {root} does not exist"));
        }
        // An explicit stack keeps deep trees off the call stack; the set
        // visits each hard-linked inode once.
        let mut pending = vec![root];
        let mut seen = HashSet::from([root]);
        while let Some(inode) = pending.pop() {
            if !self.inode(inode).is_dir() {
                continue;
            }
            for (_, child) in self.directory_entries(inode)? {
                if !self.is_inode_allocated(child) {
                    return Err(anyhow::anyhow!(
                        "directory inode {inode} references free inode {child}"
                    ));
                }
                if seen.insert(child) {
                    pending.push(child);
                }
            }
        }
        for &inode in &seen {
            Self::populate_inode_owner(self.inode_mut(inode), uid, gid);
        }
        Ok(seen.len())
    }

    /// Splits a 32-bit UID and GID across the inode's low and Linux high fields.
    fn populate_inode_owner(inode: &mut Ext4Inode, uid: u32, gid: u32) {
        inode.i_uid = uid as u16;
        inode.i_gid = gid as u16;
        inode.i_osd2[4..6].copy_from_slice(&((uid >> 16) as u16).to_le_bytes());
        inode.i_osd2[6..8].copy_from_slice(&((gid >> 16) as u16).to_le_bytes());
    }

    /// Applies POSIX metadata and file-type defaults to an inode; returns no value because it mutates in place.
    pub(crate) fn populate_inode_metadata(
        inode: &mut Ext4Inode,
        m: &PosixMetadata,
        kind: u16,
        default_perms: u16,
    ) {
        let mode = m.mode.map_or(default_perms, |mode| mode & 0o7777);
        inode.i_mode = kind | mode;
        Self::populate_inode_owner(inode, m.uid, m.gid);
        inode.i_atime = m.atime;
        inode.i_ctime = m.ctime;
        inode.i_mtime = m.mtime;
        inode.i_extra_isize = 32;
    }

    /// Serializes an unshared ext4 xattr block; returns its allocated physical block number.
    pub fn write_xattr_block(
        &mut self,
        group: usize,
        xattrs: &HashMap<String, Vec<u8>>,
    ) -> Result<u32> {
        let xattrs = prepare_xattrs(xattrs)?;
        self.write_prepared_xattr_block(group, &xattrs)
    }

    /// Writes already validated xattrs into one block; returns that block's physical number.
    pub(crate) fn write_prepared_xattr_block(
        &mut self,
        group: usize,
        xattrs: &[PreparedXattr],
    ) -> Result<u32> {
        let data = encode_xattr_block(xattrs)?;
        let number = self.allocate_free_blocks(group, 1)?;
        let offset = number as usize * BLOCK_SIZE;
        self.disk_data[offset..offset + BLOCK_SIZE].copy_from_slice(&data);
        Ok(number)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ext4::test_support::fresh_builder;
    use zerocopy::IntoBytes;

    /// Returns `(uid, gid)` including the high 16 bits kept in `i_osd2`.
    fn owner(builder: &Ext4ImageBuilder, ino: u32) -> (u32, u32) {
        let inode = builder.inode(ino);
        let high =
            |at: usize| u32::from(u16::from_le_bytes([inode.i_osd2[at], inode.i_osd2[at + 1]]));
        (
            u32::from(inode.i_uid) | high(4) << 16,
            u32::from(inode.i_gid) | high(6) << 16,
        )
    }

    #[test]
    fn set_inode_owner_changes_only_ownership() -> Result<()> {
        let mut builder = fresh_builder()?;
        let metadata = PosixMetadata {
            mode: Some(0o700),
            mtime: 1_700_000_000,
            ..Default::default()
        };
        let dir = builder.mkdir(EXT4_ROOT_INO, "dir", Some(&metadata))?;
        builder.set_inode_owner(dir, 70_000, 80_000)?;

        let inode = builder.inode(dir);
        assert_eq!({ inode.i_uid }, 70_000u32 as u16);
        assert_eq!({ inode.i_gid }, 80_000u32 as u16);
        assert_eq!(&inode.i_osd2[4..8], &[1, 0, 1, 0]);
        assert_eq!(inode.i_mode & 0o7777, 0o700);
        assert_eq!({ inode.i_mtime }, 1_700_000_000);
        assert!(builder.set_inode_owner(9999, 0, 0).is_err());
        Ok(())
    }

    #[test]
    fn set_inode_mode_changes_only_permission_bits() -> Result<()> {
        let mut builder = fresh_builder()?;
        let meta = PosixMetadata {
            mode: Some(0o600),
            uid: 7,
            gid: 8,
            atime: 1,
            ctime: 2,
            mtime: 3,
            xattrs: HashMap::from([("user.tag".to_owned(), b"v".to_vec())]),
        };
        let file = builder.add_file(EXT4_ROOT_INO, "tool", b"#!/bin/sh\n", &meta)?;
        let before = *builder.inode(file);

        builder.set_inode_mode(file, 0o4755)?;

        let mut expected = before;
        expected.i_mode = S_IFREG | 0o4755;
        assert_eq!(builder.inode(file).as_bytes(), expected.as_bytes());
        Ok(())
    }

    #[test]
    fn failed_set_inode_mode_leaves_the_inode_unchanged() -> Result<()> {
        let mut builder = fresh_builder()?;
        let before = *builder.inode(EXT4_ROOT_INO);
        assert!(
            builder
                .set_inode_mode(EXT4_ROOT_INO, S_IFREG | 0o755)
                .is_err()
        );
        assert!(builder.set_inode_mode(9999, 0o755).is_err());
        assert_eq!(builder.inode(EXT4_ROOT_INO).as_bytes(), before.as_bytes());
        Ok(())
    }

    #[test]
    fn set_owner_recursive_changes_symlinks_without_following_them() -> Result<()> {
        let mut builder = fresh_builder()?;
        let meta = PosixMetadata::default();
        let etc = builder.mkdir(EXT4_ROOT_INO, "etc", None)?;
        let passwd = builder.add_file(etc, "passwd", b"root:x:0:0::/root:/bin/sh\n", &meta)?;
        let sandbox = builder.mkdir(EXT4_ROOT_INO, "sandbox", None)?;
        let nested = builder.mkdir(sandbox, "nested", None)?;
        let file = builder.add_file(nested, "file", b"data", &meta)?;
        let link = builder.add_symlink(sandbox, "etc-link", "/etc", &meta)?;
        let escape = builder.add_symlink(sandbox, "passwd-link", "../etc/passwd", &meta)?;
        // A hard link inside the subtree shares its inode with the outside.
        builder.add_hard_link(sandbox, "hard", file)?;

        let changed = builder.set_owner_recursive(sandbox, 100_000, 1000)?;

        assert_eq!(changed, 5);
        for inode in [sandbox, nested, file, link, escape] {
            assert_eq!(owner(&builder, inode), (100_000, 1000));
        }
        for inode in [EXT4_ROOT_INO, etc, passwd] {
            assert_eq!(owner(&builder, inode), (0, 0));
        }
        Ok(())
    }

    #[test]
    fn set_owner_recursive_on_a_symlink_changes_only_the_link() -> Result<()> {
        let mut builder = fresh_builder()?;
        let meta = PosixMetadata::default();
        let etc = builder.mkdir(EXT4_ROOT_INO, "etc", None)?;
        let link = builder.add_symlink(EXT4_ROOT_INO, "sandbox", "/etc", &meta)?;

        assert_eq!(builder.set_owner_recursive(link, 1000, 1000)?, 1);
        assert_eq!(owner(&builder, link), (1000, 1000));
        assert_eq!(owner(&builder, etc), (0, 0));
        Ok(())
    }

    #[test]
    fn failed_set_owner_recursive_preflight_changes_nothing() -> Result<()> {
        let mut builder = fresh_builder()?;
        let meta = PosixMetadata::default();
        let sandbox = builder.mkdir(EXT4_ROOT_INO, "sandbox", None)?;
        let first = builder.add_file(sandbox, "a", b"a", &meta)?;
        let broken = builder.mkdir(sandbox, "broken", None)?;
        // A directory without extents cannot be listed, so the walk fails
        // after it has already seen some entries.
        builder.inode_mut(broken).i_flags &= !EXT4_EXTENTS_FL;

        assert!(builder.set_owner_recursive(sandbox, 1000, 1000).is_err());
        for inode in [sandbox, first, broken] {
            assert_eq!(owner(&builder, inode), (0, 0));
        }
        assert!(builder.set_owner_recursive(9999, 1000, 1000).is_err());
        Ok(())
    }
}
