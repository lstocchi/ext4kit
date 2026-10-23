//! Path-level edits of an existing image with the semantics of the matching
//! coreutils commands, as used by `ext4kit edit`.
//!
//! Paths are absolute image paths split on `/` only.  Parent directories
//! are resolved through symlinks, confined to the image root like a chroot.
//! Each operation either completes or leaves the image unchanged, except for
//! `make_directory` with `parents`, which keeps the parents it created
//! before a failure, like `mkdir -p`.

use crate::{
    ext4::format::{EXT4_ROOT_INO, S_IFDIR, S_IFLNK, S_IFREG},
    ext4::{Ext4ImageBuilder, PosixMetadata},
};
use anyhow::{Result, anyhow, bail};
use std::io::{Read, Write};

/// Mode and owner for an entry created or changed by an edit.  `None`
/// keeps an existing entry's value or selects the default for a new one.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EntryAttributes {
    pub mode: Option<u16>,
    pub owner: Option<(u32, u32)>,
}

impl EntryAttributes {
    /// Returns metadata for a new entry.  Timestamps are left at zero, like
    /// the destination directories that `create` makes, which keeps edited
    /// images reproducible.
    fn metadata(&self) -> PosixMetadata {
        let (uid, gid) = self.owner.unwrap_or((0, 0));
        PosixMetadata {
            mode: self.mode,
            uid,
            gid,
            ..Default::default()
        }
    }
}

/// A `test(1)` file predicate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PathTest {
    /// `-e`: the path exists, following symlinks.
    Exists,
    /// `-f`: the path is a regular file, following symlinks.
    File,
    /// `-d`: the path is a directory, following symlinks.
    Directory,
    /// `-x`: the path exists and has an execute bit, following symlinks.
    /// As for root, any of the user, group or other bits counts.
    Executable,
    /// `-L`: the path itself is a symlink.
    Symlink,
}

/// Fails unless `path` is an absolute image path.  A Windows path most
/// likely comes from Git Bash rewriting a POSIX-looking argument, so the
/// error says how to stop it.
fn require_absolute(path: &str) -> Result<()> {
    if path.starts_with('/') {
        return Ok(());
    }
    let bytes = path.as_bytes();
    if bytes.len() >= 3 && bytes[0].is_ascii_alphabetic() && bytes[1] == b':' {
        bail!(
            "image path must be absolute, got the Windows path {path}; \
             Git Bash rewrites arguments such as /etc/passwd, set MSYS_NO_PATHCONV=1 to stop it"
        );
    }
    bail!("image path must be absolute: {path}");
}

/// Splits an absolute image path into its parent path and final name.
fn split_path(path: &str) -> Result<(&str, &str)> {
    require_absolute(path)?;
    let trimmed = path.trim_end_matches('/');
    let (parent, name) = trimmed.rsplit_once('/').unwrap_or(("", trimmed));
    if name.is_empty() || name == "." || name == ".." {
        bail!("image path does not name an entry: {path}");
    }
    Ok((if parent.is_empty() { "/" } else { parent }, name))
}

impl Ext4ImageBuilder {
    /// Resolves the parent directory of `path` and returns it with the final
    /// name, failing when the parent does not exist.
    fn parent_of<'a>(&self, path: &'a str) -> Result<(u32, &'a str)> {
        let (parent, name) = split_path(path)?;
        let directory = self
            .resolve_directory(parent)?
            .ok_or_else(|| anyhow!("no such directory: {parent}"))?;
        Ok((directory, name))
    }

    /// Resolves `path` following symlinks, failing when it does not exist.
    fn existing(&self, path: &str) -> Result<u32> {
        require_absolute(path)?;
        self.resolve_path(path)?
            .ok_or_else(|| anyhow!("no such file or directory: {path}"))
    }

    /// `rm [-r]`: removes the entry at `path` without following a final
    /// symlink.  A directory needs `recursive`.  Returns `false` when the
    /// entry or its parent does not exist.
    pub fn remove_path(&mut self, path: &str, recursive: bool) -> Result<bool> {
        let (parent, name) = split_path(path)?;
        let Some(parent) = self.resolve_directory(parent)? else {
            return Ok(false);
        };
        let Some(inode) = self.lookup_dir_entry(parent, name)? else {
            return Ok(false);
        };
        if self.inode(inode).is_dir() && !recursive {
            bail!("cannot remove {path}: is a directory");
        }
        self.remove_dir_entry(parent, name)
    }

    /// `mkdir [-p]`: creates the directory `path` with `attributes` and
    /// returns its inode.  Without `parents` the parent must exist and
    /// `path` must not.  With `parents`, missing parents are created with
    /// the directory defaults, and an existing directory is returned
    /// unchanged.
    pub fn make_directory(
        &mut self,
        path: &str,
        parents: bool,
        attributes: EntryAttributes,
    ) -> Result<u32> {
        if parents {
            if let Some(existing) = self.resolve_directory(path)? {
                return Ok(existing);
            }
            self.ensure_directory(split_path(path)?.0, false)?;
        }
        let (parent, name) = self.parent_of(path)?;
        if self.lookup_dir_entry(parent, name)?.is_some() {
            bail!("cannot create directory {path}: file exists");
        }
        self.mkdir(parent, name, Some(&attributes.metadata()))
    }

    /// `chmod`: sets the permission bits of `path`, following symlinks.
    pub fn chmod_path(&mut self, path: &str, mode: u16) -> Result<()> {
        let inode = self.existing(path)?;
        self.set_inode_mode(inode, mode)
    }

    /// `chown [-R]`: sets the owner of `path`.  Without `recursive` a final
    /// symlink is followed, like `chown`; with it the walk follows no
    /// symlink at all, like `chown -R -P`.  Returns the number of inodes
    /// changed.
    pub fn chown_path(&mut self, path: &str, uid: u32, gid: u32, recursive: bool) -> Result<usize> {
        if !recursive {
            let inode = self.existing(path)?;
            self.set_inode_owner(inode, uid, gid)?;
            return Ok(1);
        }
        require_absolute(path)?;
        let inode = self
            .resolve_entry(path)?
            .ok_or_else(|| anyhow!("no such file or directory: {path}"))?;
        self.set_owner_recursive(inode, uid, gid)
    }

    /// Writes exactly `size` bytes from `reader` to the regular file at
    /// `path`, following symlinks.  An existing file keeps its inode, links,
    /// timestamps and xattrs, and its mode and owner unless `attributes`
    /// sets them.  A new file defaults to mode 0644 owned by root.
    pub fn write_path(
        &mut self,
        path: &str,
        reader: impl Read,
        size: u64,
        attributes: EntryAttributes,
    ) -> Result<u32> {
        if attributes.mode.is_some_and(|mode| mode & !0o7777 != 0) {
            bail!("mode must be at most 7777");
        }
        let (parent, name) = self.parent_of(path)?;
        let Some(inode) = self.resolve_path(path)? else {
            if self.lookup_dir_entry(parent, name)?.is_some() {
                bail!("cannot write {path}: it is a dangling symlink");
            }
            return self.add_file_from_reader(parent, name, reader, size, &attributes.metadata());
        };
        if self.inode(inode).file_type() != S_IFREG {
            bail!("cannot write {path}: not a regular file");
        }
        self.replace_file_contents(inode, reader, size)?;
        if let Some(mode) = attributes.mode {
            self.set_inode_mode(inode, mode)?;
        }
        if let Some((uid, gid)) = attributes.owner {
            self.set_inode_owner(inode, uid, gid)?;
        }
        Ok(inode)
    }

    /// `ln -s`: creates a symlink at `path` pointing to `target`, which is
    /// stored as given and need not exist.
    pub fn symlink_path(
        &mut self,
        target: &str,
        path: &str,
        attributes: EntryAttributes,
    ) -> Result<u32> {
        let (parent, name) = self.parent_of(path)?;
        self.add_symlink(parent, name, target, &attributes.metadata())
    }

    /// `ln`: adds `path` as another name for the existing non-directory at
    /// `existing`, without following a final symlink.  Returns its inode.
    pub fn hard_link_path(&mut self, existing: &str, path: &str) -> Result<u32> {
        require_absolute(existing)?;
        let inode = self
            .resolve_entry(existing)?
            .ok_or_else(|| anyhow!("no such file: {existing}"))?;
        if self.inode(inode).is_dir() {
            bail!("cannot hard link directory {existing}");
        }
        let (parent, name) = self.parent_of(path)?;
        self.add_hard_link(parent, name, inode)?;
        Ok(inode)
    }

    /// `test`: evaluates `predicate` on `path`.  A missing path or a missing
    /// parent is `false`, not an error.
    pub fn test_path(&self, path: &str, predicate: PathTest) -> Result<bool> {
        require_absolute(path)?;
        let resolved = if predicate == PathTest::Symlink {
            self.resolve_entry(path)?
        } else {
            self.resolve_path(path)?
        };
        let Some(inode) = resolved else {
            return Ok(false);
        };
        let inode = self.inode(inode);
        Ok(match predicate {
            PathTest::Exists => true,
            PathTest::File => inode.file_type() == S_IFREG,
            PathTest::Directory => inode.file_type() == S_IFDIR,
            PathTest::Executable => inode.i_mode & 0o111 != 0,
            PathTest::Symlink => inode.file_type() == S_IFLNK,
        })
    }

    /// `cat`: copies the regular file at `path`, following symlinks, to
    /// `writer`; returns the number of bytes written.
    pub fn cat_path(&self, path: &str, writer: &mut impl Write) -> Result<u64> {
        let inode = self.existing(path)?;
        if inode == EXT4_ROOT_INO || self.inode(inode).file_type() != S_IFREG {
            bail!("cannot read {path}: not a regular file");
        }
        self.read_file(inode, writer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn builder() -> Result<Ext4ImageBuilder> {
        let mut builder = Ext4ImageBuilder::new(16);
        builder.create_root_dir()?;
        builder.create_lost_and_found()?;
        builder.reserve_inodes();
        builder.make_directory("/usr/bin", true, EntryAttributes::default())?;
        builder.symlink_path("usr/bin", "/bin", EntryAttributes::default())?;
        Ok(builder)
    }

    fn write(builder: &mut Ext4ImageBuilder, path: &str, data: &str) -> Result<u32> {
        builder.write_path(
            path,
            data.as_bytes(),
            data.len() as u64,
            EntryAttributes::default(),
        )
    }

    fn cat(builder: &Ext4ImageBuilder, path: &str) -> String {
        let mut data = Vec::new();
        builder.cat_path(path, &mut data).unwrap();
        String::from_utf8(data).unwrap()
    }

    fn mode(builder: &Ext4ImageBuilder, path: &str) -> u16 {
        builder
            .inode(builder.resolve_path(path).unwrap().unwrap())
            .i_mode
            & 0o7777
    }

    #[test]
    fn split_path_requires_an_absolute_path_naming_an_entry() {
        assert_eq!(split_path("/etc/passwd").unwrap(), ("/etc", "passwd"));
        assert_eq!(split_path("/sandbox/").unwrap(), ("/", "sandbox"));
        for bad in ["etc/passwd", "/", "/etc/..", "/etc/."] {
            assert!(split_path(bad).is_err(), "{bad}");
        }
        let mangled = split_path("C:/Program Files/Git/etc/passwd").unwrap_err();
        assert!(mangled.to_string().contains("MSYS_NO_PATHCONV"));
    }

    #[test]
    fn remove_then_recreate_in_sequence() -> Result<()> {
        let mut builder = builder()?;
        write(&mut builder, "/bin/tool", "v1")?;
        assert!(builder.remove_path("/bin/tool", false)?);
        assert!(!builder.test_path("/usr/bin/tool", PathTest::Exists)?);
        write(&mut builder, "/bin/tool", "v2")?;
        assert_eq!(cat(&builder, "/usr/bin/tool"), "v2");
        Ok(())
    }

    #[test]
    fn remove_needs_recursive_for_directories_and_reports_missing_entries() -> Result<()> {
        let mut builder = builder()?;
        write(&mut builder, "/usr/bin/tool", "x")?;
        assert!(builder.remove_path("/usr", false).is_err());
        assert!(builder.test_path("/usr/bin/tool", PathTest::File)?);
        assert!(!builder.remove_path("/missing", false)?);
        // A final symlink is removed itself, not its target.
        assert!(builder.remove_path("/bin", false)?);
        assert!(builder.test_path("/usr/bin/tool", PathTest::File)?);
        assert!(builder.remove_path("/usr", true)?);
        assert!(!builder.test_path("/usr", PathTest::Exists)?);
        assert!(!builder.remove_path("/missing/child", false)?);
        Ok(())
    }

    #[test]
    fn make_directory_follows_mkdir_semantics() -> Result<()> {
        let mut builder = builder()?;
        let attributes = EntryAttributes {
            mode: Some(0o700),
            owner: Some((1000, 1000)),
        };
        assert!(builder.make_directory("/a/b", false, attributes).is_err());
        let b = builder.make_directory("/a/b", true, attributes)?;
        assert_eq!(mode(&builder, "/a"), 0o755);
        assert_eq!(mode(&builder, "/a/b"), 0o700);
        assert_eq!({ builder.inode(b).i_uid }, 1000);
        // -p leaves an existing directory unchanged; without -p it fails.
        assert_eq!(
            builder.make_directory("/a/b", true, EntryAttributes::default())?,
            b
        );
        assert_eq!(mode(&builder, "/a/b"), 0o700);
        assert!(builder.make_directory("/a/b", false, attributes).is_err());
        // Parents resolve through symlinks.
        let sbin = builder.make_directory("/bin/sub", false, EntryAttributes::default())?;
        assert_eq!(builder.resolve_directory("/usr/bin/sub")?, Some(sbin));
        write(&mut builder, "/file", "x")?;
        assert!(builder.make_directory("/file", true, attributes).is_err());
        Ok(())
    }

    #[test]
    fn chmod_and_chown_follow_symlinks_unless_recursive() -> Result<()> {
        let mut builder = builder()?;
        write(&mut builder, "/usr/bin/bash", "elf")?;
        builder.symlink_path("bash", "/usr/bin/sh", EntryAttributes::default())?;
        builder.chmod_path("/bin/sh", 0o755)?;
        assert_eq!(mode(&builder, "/usr/bin/bash"), 0o755);
        assert!(builder.chmod_path("/missing", 0o755).is_err());

        builder.chown_path("/bin/sh", 5, 6, false)?;
        let bash = builder.resolve_path("/usr/bin/bash")?.unwrap();
        assert_eq!(
            ({ builder.inode(bash).i_uid }, { builder.inode(bash).i_gid }),
            (5, 6)
        );

        // -R on a symlink changes only the link.
        let link = builder.resolve_entry("/bin")?.unwrap();
        assert_eq!(builder.chown_path("/bin", 7, 7, true)?, 1);
        assert_eq!({ builder.inode(link).i_uid }, 7);
        assert_eq!({ builder.inode(bash).i_uid }, 5);
        Ok(())
    }

    #[test]
    fn write_replaces_existing_files_and_creates_new_ones() -> Result<()> {
        let mut builder = builder()?;
        let attributes = EntryAttributes {
            mode: Some(0o600),
            owner: Some((0, 42)),
        };
        let created = builder.write_path("/shadow", &b"a"[..], 1, attributes)?;
        assert_eq!(mode(&builder, "/shadow"), 0o600);
        let replaced = write(&mut builder, "/shadow", "root:*::\n")?;
        assert_eq!(created, replaced);
        assert_eq!(cat(&builder, "/shadow"), "root:*::\n");
        // Existing mode and owner survive a write without attributes.
        assert_eq!(mode(&builder, "/shadow"), 0o600);
        assert_eq!({ builder.inode(created).i_gid }, 42);

        assert!(write(&mut builder, "/bin", "x").is_err());
        builder.symlink_path("/nowhere", "/dangling", EntryAttributes::default())?;
        assert!(write(&mut builder, "/dangling", "x").is_err());
        assert!(write(&mut builder, "/missing/file", "x").is_err());
        let bad_mode = EntryAttributes {
            mode: Some(0o10000),
            owner: None,
        };
        assert!(
            builder
                .write_path("/shadow", &b"zz"[..], 2, bad_mode)
                .is_err()
        );
        assert!(builder.write_path("/new", &b"zz"[..], 2, bad_mode).is_err());
        assert!(!builder.test_path("/new", PathTest::Exists)?);
        assert_eq!(cat(&builder, "/shadow"), "root:*::\n");
        Ok(())
    }

    #[test]
    fn links_and_tests() -> Result<()> {
        let mut builder = builder()?;
        write(&mut builder, "/usr/bin/mount", "elf")?;
        builder.chmod_path("/usr/bin/mount", 0o755)?;
        builder.hard_link_path("/usr/bin/mount", "/usr/bin/mount2")?;
        let mount = builder.resolve_path("/usr/bin/mount")?.unwrap();
        assert_eq!({ builder.inode(mount).i_links_count }, 2);
        assert!(builder.hard_link_path("/usr", "/usr2").is_err());

        assert!(builder.test_path("/bin/mount", PathTest::Executable)?);
        assert!(builder.test_path("/bin/mount2", PathTest::File)?);
        assert!(builder.test_path("/bin", PathTest::Directory)?);
        assert!(builder.test_path("/bin", PathTest::Symlink)?);
        assert!(!builder.test_path("/usr/bin", PathTest::Symlink)?);
        assert!(!builder.test_path("/bin/sed", PathTest::Exists)?);
        assert!(!builder.test_path("/missing/sed", PathTest::Exists)?);
        write(&mut builder, "/plain", "x")?;
        assert!(!builder.test_path("/plain", PathTest::Executable)?);
        assert!(builder.cat_path("/usr", &mut Vec::new()).is_err());
        Ok(())
    }
}
