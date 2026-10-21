//! Import of a host-directory tree into an ext4 image.

use super::{Ext4ImageBuilder, directory::PosixMetadata};
use crate::ext4::{BLOCK_SIZE, BLOCKS_PER_GROUP, EXT4_ROOT_INO, INODE_SIZE};
use anyhow::{Context, Result};
use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

impl Ext4ImageBuilder {
    /// Returns a filesystem size, in MiB, suitable for importing `source`.
    ///
    /// The estimate accounts for block-rounded file sizes, directory entries,
    /// inode tables, ext4 bookkeeping, and ten percent free space. It is
    /// rounded up to 16 MiB so callers do not need to guess an image size.
    pub fn recommended_image_size_mib(source: impl AsRef<Path>) -> Result<usize> {
        Self::recommended_image_size_mib_for_paths([source.as_ref()], 0)
    }

    /// Returns a filesystem size for several imported paths and additional
    /// destination directories created by the caller.
    pub fn recommended_image_size_mib_for_paths<'a>(
        sources: impl IntoIterator<Item = &'a Path>,
        additional_directories: usize,
    ) -> Result<usize> {
        let mut usage = HostUsage::default();
        for source in sources {
            let tree = HostTree::scan(source)?;
            usage.data_blocks += tree.usage.data_blocks;
            usage.inodes += tree.usage.inodes;
        }
        Self::recommended_image_size_mib_for_usage(usage, additional_directories)
    }

    /// Returns a filesystem size for already-scanned host trees.
    pub fn recommended_image_size_mib_for_trees(
        trees: &[HostTree],
        additional_directories: usize,
    ) -> Result<usize> {
        let usage = trees.iter().fold(HostUsage::default(), |mut usage, tree| {
            usage.data_blocks += tree.usage.data_blocks;
            usage.inodes += tree.usage.inodes;
            usage
        });
        Self::recommended_image_size_mib_for_usage(usage, additional_directories)
    }

    fn recommended_image_size_mib_for_usage(
        mut usage: HostUsage,
        additional_directories: usize,
    ) -> Result<usize> {
        usage.inodes += additional_directories as u64;
        usage.data_blocks += additional_directories as u64;
        usage.data_blocks += 5; // Root plus lost+found.
        usage.inodes += 11; // Root, lost+found, and ext4's reserved inodes.

        // Start close to the final answer: payload blocks need ten percent
        // free space, and ext4 needs inode tables, bitmaps, and superblock
        // copies. The exact geometry checks below remain the authority.
        let raw_data_bytes = usage
            .data_blocks
            .checked_mul(BLOCK_SIZE as u64)
            .ok_or_else(|| anyhow::anyhow!("data block size overflow"))?;
        let payload_bytes = raw_data_bytes
            .checked_mul(110)
            .ok_or_else(|| anyhow::anyhow!("data block size overflow"))?
            .div_ceil(100);
        let estimated_bytes = payload_bytes
            .checked_add(payload_bytes / 20)
            .and_then(|value| value.checked_add(1024 * 1024))
            .ok_or_else(|| anyhow::anyhow!("image size overflow"))?;
        let raw_mib = estimated_bytes.div_ceil(1024 * 1024);
        let raw_mib = usize::try_from(raw_mib)
            .map_err(|_| anyhow::anyhow!("image size does not fit this platform"))?;
        let mut size_mib = raw_mib.max(16).next_multiple_of(16);

        loop {
            let image_blocks = size_mib
                .checked_mul(1024 * 1024 / BLOCK_SIZE)
                .ok_or_else(|| anyhow::anyhow!("image size overflow"))?;
            let groups = image_blocks.div_ceil(BLOCKS_PER_GROUP);
            let inodes_per_group = Self::inodes_per_group(image_blocks, groups);
            let inode_table_blocks = groups
                .checked_mul((inodes_per_group * INODE_SIZE).div_ceil(BLOCK_SIZE))
                .ok_or_else(|| anyhow::anyhow!("inode table size overflow"))?;
            // One block bitmap and one inode bitmap per group, plus primary
            // and sparse-super backup metadata. This is conservative.
            let ext4_metadata_blocks = inode_table_blocks + groups * 4;
            let allocated_blocks = usage
                .data_blocks
                .checked_add(ext4_metadata_blocks as u64)
                .ok_or_else(|| anyhow::anyhow!("allocation estimate overflow"))?;
            let required_with_free_space = allocated_blocks
                .checked_mul(110)
                .ok_or_else(|| anyhow::anyhow!("allocation estimate overflow"))?
                .div_ceil(100);

            if usage.inodes <= groups as u64 * inodes_per_group as u64
                && required_with_free_space <= image_blocks as u64
            {
                return Ok(size_mib);
            }
            size_mib = size_mib
                .checked_add(16)
                .ok_or_else(|| anyhow::anyhow!("image size overflow"))?;
        }
    }

    /// Creates a directory named after `source` below `parent`, imports the
    /// source's contents into it, and returns the new directory's inode.
    ///
    /// For example, importing `C:\\tree` into the filesystem root creates
    /// `/tree`, matching the layout produced by cloning a repository there.
    /// Host paths must have UTF-8 names because the public builder API accepts
    /// UTF-8 ext4 directory components. Directories, regular files, and
    /// symbolic links are supported. Other host file types are rejected rather
    /// than silently creating an incorrect image.
    pub fn import_host_directory(&mut self, parent: u32, source: impl AsRef<Path>) -> Result<u32> {
        let source = source.as_ref();
        let metadata = fs::symlink_metadata(source)
            .with_context(|| format!("cannot inspect host directory {}", source.display()))?;
        if !metadata.file_type().is_dir() {
            return Err(anyhow::anyhow!(
                "host import source is not a directory: {}",
                source.display()
            ));
        }
        self.import_host_path(parent, source)
    }

    /// Imports a regular file, directory, or symbolic link below `parent`.
    /// The imported entry retains its host basename and returns its inode.
    pub fn import_host_path(&mut self, parent: u32, source: impl AsRef<Path>) -> Result<u32> {
        self.import_host_path_impl(parent, source.as_ref())
    }

    /// Imports a previously scanned host tree without re-reading directory
    /// entries or host metadata.
    pub fn import_host_tree(&mut self, parent: u32, tree: &HostTree) -> Result<u32> {
        self.import_host_tree_impl(parent, tree)
    }

    /// Imports a host path in one bounded dua-core traversal without first
    /// retaining a `HostTree`. Parents are emitted before descendants, so only
    /// the directory-ID-to-inode map remains live during import.
    pub fn import_host_path_streaming(
        &mut self,
        parent: u32,
        source: impl AsRef<Path>,
    ) -> Result<u32> {
        self.import_host_path_streaming_impl(parent, source.as_ref())
    }

    /// Merges a host directory into an existing ext4 directory.  Host entries
    /// replace like-named non-directories; like-named directories are merged.
    /// The existing directory keeps its own metadata (mode, owner, xattrs),
    /// as when copying into a directory that already exists.
    pub fn overlay_host_directory(&mut self, parent: u32, source: impl AsRef<Path>) -> Result<()> {
        let source = source.as_ref();
        let metadata = fs::symlink_metadata(source)
            .with_context(|| format!("cannot inspect host directory {}", source.display()))?;
        if !metadata.file_type().is_dir() {
            return Err(anyhow::anyhow!(
                "host overlay source is not a directory: {}",
                source.display()
            ));
        }
        self.merge_host_directory(parent, source)
    }

    fn merge_host_directory(&mut self, parent: u32, source: &Path) -> Result<()> {
        for path in Self::sorted_host_entries(source)? {
            let name = path
                .file_name()
                .and_then(|name| name.to_str())
                .ok_or_else(|| {
                    anyhow::anyhow!("host path name is not valid UTF-8: {}", path.display())
                })?;
            if let Some(existing) = self.lookup_dir_entry(parent, name)? {
                let host_is_dir = fs::symlink_metadata(&path)
                    .with_context(|| format!("cannot inspect host path {}", path.display()))?
                    .file_type()
                    .is_dir();
                if host_is_dir && self.is_directory(existing) {
                    self.merge_host_directory(existing, &path)?;
                    continue;
                }
                self.remove_dir_entry(parent, name)?;
            }
            self.import_host_path_impl(parent, &path)?;
        }
        Ok(())
    }

    fn import_host_path_streaming_impl(&mut self, parent: u32, source: &Path) -> Result<u32> {
        let threads = std::thread::available_parallelism().map_or(1, usize::from);
        let walker = dua_core::walk(
            source,
            threads,
            dua_core::Order::ParentFirst,
            dua_core::Options::default(),
            |_| true,
        );
        let mut directory_inodes = HashMap::new();
        let mut root_inode = None;

        for walked in walker {
            let walked =
                walked.with_context(|| format!("cannot walk host tree {}", source.display()))?;
            let path = walked.path();
            let parent_inode = walked
                .parent_directory_id
                .map_or(parent, |id| directory_inodes[&id]);
            let name = path
                .file_name()
                .and_then(|name| name.to_str())
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "host path name is not valid UTF-8 or is missing: {}",
                        path.display()
                    )
                })?;
            Self::validate_name(name)?;

            let std_metadata = fs::symlink_metadata(&path)
                .with_context(|| format!("cannot inspect host path {}", path.display()))?;
            let file_type = if walked.file_type.is_dir() {
                HostFileType::Directory
            } else if walked.file_type.is_file() {
                HostFileType::File
            } else if walked.file_type.is_symlink() {
                HostFileType::Symlink
            } else {
                return Err(anyhow::anyhow!(
                    "unsupported host file type at {} (only files, directories, and symlinks can be imported)",
                    path.display()
                ));
            };
            let metadata =
                HostMetadata::from_std(&std_metadata, file_type == HostFileType::Directory);
            let host_meta = metadata.as_posix();

            let inode = match file_type {
                HostFileType::Directory => {
                    let inode = self.mkdir(parent_inode, name, Some(&host_meta))?;
                    directory_inodes.insert(
                        walked.directory_id.expect("directory entry has an ID"),
                        inode,
                    );
                    inode
                }
                HostFileType::File => {
                    let file = fs::File::open(&path)
                        .with_context(|| format!("cannot open host file {}", path.display()))?;
                    let inode = self.add_file_from_reader(
                        parent_inode,
                        name,
                        file,
                        metadata.size,
                        &host_meta,
                    )?;
                    inode
                }
                HostFileType::Symlink => {
                    let target = fs::read_link(&path)
                        .with_context(|| format!("cannot read symbolic link {}", path.display()))?;
                    let target = target.to_str().ok_or_else(|| {
                        anyhow::anyhow!(
                            "symbolic-link target is not valid UTF-8: {}",
                            path.display()
                        )
                    })?;
                    self.add_symlink(parent_inode, name, target, &host_meta)?
                }
            };
            root_inode.get_or_insert(inode);
        }

        root_inode.ok_or_else(|| anyhow::anyhow!("host traversal produced no entries"))
    }

    fn import_host_path_impl(&mut self, parent: u32, source: &Path) -> Result<u32> {
        let metadata = fs::symlink_metadata(source)
            .with_context(|| format!("cannot inspect host path {}", source.display()))?;
        let name = source
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "host path name is not valid UTF-8 or is missing: {}",
                    source.display()
                )
            })?;
        Self::validate_name(name)?;
        let file_type = metadata.file_type();
        let host_meta = Self::host_metadata(&metadata, file_type.is_dir());
        if file_type.is_dir() {
            let directory = self.mkdir(parent, name, Some(&host_meta))?;
            self.import_host_directory_contents(directory, source)?;
            Ok(directory)
        } else if file_type.is_file() {
            let file = fs::File::open(source)
                .with_context(|| format!("cannot open host file {}", source.display()))?;
            self.add_file_from_reader(parent, name, file, metadata.len(), &host_meta)
        } else if file_type.is_symlink() {
            let target = fs::read_link(source)
                .with_context(|| format!("cannot read symbolic link {}", source.display()))?;
            let target = target.to_str().ok_or_else(|| {
                anyhow::anyhow!(
                    "symbolic-link target is not valid UTF-8: {}",
                    source.display()
                )
            })?;
            self.add_symlink(parent, name, target, &host_meta)
        } else {
            Err(anyhow::anyhow!(
                "unsupported host file type at {} (only files, directories, and symlinks can be imported)",
                source.display()
            ))
        }
    }

    fn import_host_tree_impl(&mut self, parent: u32, tree: &HostTree) -> Result<u32> {
        let mut ext4_inodes = Vec::with_capacity(tree.entries.len());
        for entry in &tree.entries {
            let parent_inode = entry
                .parent_index
                .map_or(parent, |index| ext4_inodes[index]);
            let name = entry
                .path
                .file_name()
                .and_then(|name| name.to_str())
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "host path name is not valid UTF-8 or is missing: {}",
                        entry.path.display()
                    )
                })?;
            let host_meta = entry.metadata.as_posix();
            let inode = if entry.file_type == HostFileType::Directory {
                let inode = self.mkdir(parent_inode, name, Some(&host_meta))?;
                inode
            } else if entry.file_type == HostFileType::File {
                let file = fs::File::open(&entry.path)
                    .with_context(|| format!("cannot open host file {}", entry.path.display()))?;
                let inode = self.add_file_from_reader(
                    parent_inode,
                    name,
                    file,
                    entry.metadata.size,
                    &host_meta,
                )?;
                inode
            } else {
                self.add_symlink(
                    parent_inode,
                    name,
                    entry
                        .symlink_target
                        .as_deref()
                        .expect("scanned symlink has a target"),
                    &host_meta,
                )?
            };
            ext4_inodes.push(inode);
        }
        Ok(ext4_inodes[0])
    }

    /// Creates a directory for a host tree at the ext4 filesystem root.
    pub fn import_host_root(&mut self, source: impl AsRef<Path>) -> Result<u32> {
        self.import_host_path(EXT4_ROOT_INO, source)
    }

    /// Creates an absolute directory path in a newly created image and returns
    /// its final inode. Existing paths are not looked up by this builder.
    pub fn create_directory_path(&mut self, path: &str) -> Result<u32> {
        if !path.starts_with('/') {
            return Err(anyhow::anyhow!("destination path must be absolute: {path}"));
        }
        let mut parent = EXT4_ROOT_INO;
        for component in path.split('/').filter(|component| !component.is_empty()) {
            Self::validate_name(component)?;
            parent = self.mkdir(parent, component, None)?;
        }
        Ok(parent)
    }

    /// Returns a host directory's children in a stable order, which keeps
    /// generated images reproducible for an unchanged tree.
    fn sorted_host_entries(source: &Path) -> Result<Vec<PathBuf>> {
        let mut entries: Vec<PathBuf> = fs::read_dir(source)
            .with_context(|| format!("cannot read host directory {}", source.display()))?
            .map(|entry| entry.map(|entry| entry.path()))
            .collect::<std::io::Result<_>>()
            .with_context(|| format!("cannot enumerate host directory {}", source.display()))?;
        entries.sort();
        Ok(entries)
    }

    /// Imports sorted children of a host directory below an existing inode.
    fn import_host_directory_contents(&mut self, parent: u32, source: &Path) -> Result<()> {
        for path in Self::sorted_host_entries(source)? {
            self.import_host_path_impl(parent, &path)?;
        }
        Ok(())
    }

    /// Converts portable host metadata into the ext4 metadata we can preserve.
    fn host_metadata(metadata: &fs::Metadata, is_dir: bool) -> PosixMetadata {
        let default_mode = if is_dir { 0o755 } else { 0o644 };
        let mode = if metadata.permissions().readonly() {
            default_mode & !0o222
        } else {
            default_mode
        };
        let timestamp = |time: std::io::Result<SystemTime>| {
            time.ok()
                .and_then(|value| value.duration_since(UNIX_EPOCH).ok())
                .map_or(0, |value| value.as_secs().min(u32::MAX as u64) as u32)
        };
        PosixMetadata {
            mode: Some(mode),
            atime: timestamp(metadata.accessed()),
            ctime: timestamp(metadata.created()),
            mtime: timestamp(metadata.modified()),
            ..Default::default()
        }
    }
}

#[derive(Clone, Copy, Default)]
struct HostUsage {
    data_blocks: u64,
    inodes: u64,
}

/// A pre-order host-tree snapshot used to size and import an image without a
/// second metadata or directory-enumeration pass. Entries live in a flat arena
/// so scanning and importing arbitrarily deep directory trees never recurse.
pub struct HostTree {
    entries: Vec<HostEntry>,
    usage: HostUsage,
}

struct HostEntry {
    path: PathBuf,
    metadata: HostMetadata,
    file_type: HostFileType,
    symlink_target: Option<String>,
    parent_index: Option<usize>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum HostFileType {
    Directory,
    File,
    Symlink,
}

/// Compact, platform-neutral metadata retained between the scan and import
/// passes. Unlike `fs::Metadata`, this stores only values written to ext4.
#[derive(Clone, Copy, Debug, Default)]
struct HostMetadata {
    size: u64,
    uid: u32,
    gid: u32,
    mode: u16,
    atime: u32,
    ctime: u32,
    mtime: u32,
}

impl HostMetadata {
    fn from_std(metadata: &fs::Metadata, is_dir: bool) -> Self {
        let default_mode = if is_dir { 0o755 } else { 0o644 };
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            Self {
                size: metadata.len(),
                uid: metadata.uid(),
                gid: metadata.gid(),
                mode: (metadata.mode() & 0o7777) as u16,
                atime: clamp_unix_timestamp(metadata.atime()),
                ctime: clamp_unix_timestamp(metadata.ctime()),
                mtime: clamp_unix_timestamp(metadata.mtime()),
            }
        }
        #[cfg(not(unix))]
        {
            let mode = if metadata.permissions().readonly() {
                default_mode & !0o222
            } else {
                default_mode
            };
            Self {
                size: metadata.len(),
                mode,
                atime: system_time_seconds(metadata.accessed()),
                ctime: system_time_seconds(metadata.created()),
                mtime: system_time_seconds(metadata.modified()),
                ..Self::default()
            }
        }
    }

    fn as_posix(self) -> PosixMetadata {
        PosixMetadata {
            mode: Some(self.mode),
            uid: self.uid,
            gid: self.gid,
            atime: self.atime,
            ctime: self.ctime,
            mtime: self.mtime,
            ..Default::default()
        }
    }
}

#[cfg(unix)]
fn clamp_unix_timestamp(seconds: i64) -> u32 {
    seconds.clamp(0, i64::from(u32::MAX)) as u32
}

fn system_time_seconds(time: std::io::Result<SystemTime>) -> u32 {
    time.ok()
        .and_then(|value| value.duration_since(UNIX_EPOCH).ok())
        .map_or(0, |value| value.as_secs().min(u64::from(u32::MAX)) as u32)
}

struct PendingHostEntry {
    entry: HostEntry,
    directory_id: Option<dua_core::DirectoryId>,
    parent_directory_id: Option<dua_core::DirectoryId>,
}

impl HostTree {
    /// Streams `source` through dua-core's bounded parallel walker and retains
    /// compact metadata in a stable pre-order arena for sizing and import.
    pub fn scan(source: impl AsRef<Path>) -> Result<Self> {
        let mut usage = HostUsage::default();
        let mut pending = Vec::with_capacity(8192);
        let mut directory_bytes = HashMap::new();
        let threads = std::thread::available_parallelism().map_or(1, usize::from);
        let walker = dua_core::walk(
            source.as_ref(),
            threads,
            dua_core::Order::ParentFirst,
            dua_core::Options::default(),
            |_| true,
        );

        for walked in walker {
            let walked = walked
                .with_context(|| format!("cannot walk host tree {}", source.as_ref().display()))?;
            let path = walked.path();
            let std_metadata = fs::symlink_metadata(&path)
                .with_context(|| format!("cannot inspect host path {}", path.display()))?;
            let file_type = if walked.file_type.is_dir() {
                HostFileType::Directory
            } else if walked.file_type.is_file() {
                HostFileType::File
            } else if walked.file_type.is_symlink() {
                HostFileType::Symlink
            } else {
                return Err(anyhow::anyhow!(
                    "unsupported host file type at {} (only files, directories, and symlinks can be imported)",
                    path.display()
                ));
            };
            let metadata =
                HostMetadata::from_std(&std_metadata, file_type == HostFileType::Directory);
            let mut symlink_target = None;

            usage.inodes += 1;
            if walked.depth > 0 {
                let name = walked.file_name.to_str().ok_or_else(|| {
                    anyhow::anyhow!("host path is not valid UTF-8: {}", path.display())
                })?;
                Ext4ImageBuilder::validate_name(name)?;
                *directory_bytes
                    .entry(
                        walked
                            .parent_directory_id
                            .expect("non-root entry has a parent"),
                    )
                    .or_insert(24u64) += (8 + name.len()).next_multiple_of(4) as u64;
            }

            if file_type == HostFileType::Directory {
                directory_bytes
                    .entry(walked.directory_id.expect("directory has an id"))
                    .or_insert(24);
            } else if file_type == HostFileType::File {
                usage.data_blocks += metadata.size.div_ceil(BLOCK_SIZE as u64);
            } else {
                let target = fs::read_link(&path)
                    .with_context(|| format!("cannot read symbolic link {}", path.display()))?;
                let target = target.to_str().ok_or_else(|| {
                    anyhow::anyhow!(
                        "symbolic-link target is not valid UTF-8: {}",
                        path.display()
                    )
                })?;
                if target.len() >= BLOCK_SIZE {
                    return Err(anyhow::anyhow!(
                        "symlink target must be no longer than {} bytes: {}",
                        BLOCK_SIZE - 1,
                        path.display()
                    ));
                }
                usage.data_blocks += u64::from(target.len() >= 60);
                symlink_target = Some(target.to_owned());
            }

            pending.push(PendingHostEntry {
                entry: HostEntry {
                    path,
                    metadata,
                    file_type,
                    symlink_target,
                    parent_index: None,
                },
                directory_id: walked.directory_id,
                parent_directory_id: walked.parent_directory_id,
            });
        }

        // Parallel completion does not define sibling order. Sorting full paths
        // restores deterministic pre-order while directory IDs preserve parents.
        pending.sort_unstable_by(|left, right| left.entry.path.cmp(&right.entry.path));
        let directory_indices: HashMap<_, _> = pending
            .iter()
            .enumerate()
            .filter_map(|(index, entry)| entry.directory_id.map(|id| (id, index)))
            .collect();
        let entries = pending
            .into_iter()
            .map(|mut pending| {
                pending.entry.parent_index =
                    pending.parent_directory_id.map(|id| directory_indices[&id]);
                pending.entry
            })
            .collect();
        usage.data_blocks += directory_bytes
            .values()
            .map(|bytes| bytes.div_ceil(BLOCK_SIZE as u64))
            .sum::<u64>();

        Ok(Self { entries, usage })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retained_host_metadata_is_compact() {
        assert_eq!(std::mem::size_of::<HostMetadata>(), 32);
        assert!(std::mem::size_of::<HostMetadata>() < std::mem::size_of::<fs::Metadata>());
    }

    #[test]
    fn imports_nested_host_tree() -> Result<()> {
        let base = std::env::temp_dir().join(format!("ext4kit-import-{}", std::process::id()));
        let _ = fs::remove_dir_all(&base);
        fs::create_dir_all(base.join("nested"))?;
        fs::write(base.join("top.txt"), b"top")?;
        fs::write(base.join("nested").join("child.txt"), b"child")?;

        let size_mib = Ext4ImageBuilder::recommended_image_size_mib(&base)?;
        assert!(size_mib >= 16);

        let mut builder = Ext4ImageBuilder::new(size_mib);
        builder.create_root_dir()?;
        builder.create_lost_and_found()?;
        builder.reserve_inodes();
        let imported = builder.import_host_root(&base)?;
        let (group, index) = builder.inode_location(imported);
        assert_eq!(
            builder.groups[group].inode_table[index].i_mode & 0xf000,
            crate::ext4::S_IFDIR
        );

        fs::remove_dir_all(base)?;
        Ok(())
    }

    #[test]
    fn imports_pre_scanned_host_tree() -> Result<()> {
        let base = std::env::temp_dir().join(format!("ext4kit-manifest-{}", std::process::id()));
        let _ = fs::remove_dir_all(&base);
        fs::create_dir_all(base.join("nested"))?;
        fs::write(base.join("nested").join("child.txt"), b"child")?;

        let trees = [HostTree::scan(&base)?];
        let size_mib = Ext4ImageBuilder::recommended_image_size_mib_for_trees(&trees, 0)?;
        let mut builder = Ext4ImageBuilder::new(size_mib);
        builder.create_root_dir()?;
        builder.create_lost_and_found()?;
        builder.reserve_inodes();
        builder.import_host_tree(EXT4_ROOT_INO, &trees[0])?;

        fs::remove_dir_all(base)?;
        Ok(())
    }

    #[test]
    fn imports_host_tree_in_one_streaming_pass() -> Result<()> {
        let base = std::env::temp_dir().join(format!("ext4kit-stream-{}", std::process::id()));
        let _ = fs::remove_dir_all(&base);
        fs::create_dir_all(base.join("nested"))?;
        fs::write(base.join("nested").join("child.txt"), b"child")?;

        let mut builder = Ext4ImageBuilder::new(16);
        builder.create_root_dir()?;
        builder.create_lost_and_found()?;
        builder.reserve_inodes();
        let inode = builder.import_host_path_streaming(EXT4_ROOT_INO, &base)?;
        let (group, index) = builder.inode_location(inode);
        assert_eq!(
            builder.groups[group].inode_table[index].i_mode & 0xf000,
            crate::ext4::S_IFDIR
        );

        fs::remove_dir_all(base)?;
        Ok(())
    }

    #[test]
    fn overlay_replacing_a_nested_directory_releases_links_and_caches() -> Result<()> {
        let base =
            std::env::temp_dir().join(format!("ext4kit-overlay-remove-{}", std::process::id()));
        let incoming = base.join("incoming");
        let _ = fs::remove_dir_all(&base);
        fs::create_dir_all(&incoming)?;
        fs::write(incoming.join("replacement"), b"new file")?;

        let mut builder = Ext4ImageBuilder::new(16);
        builder.create_root_dir()?;
        builder.create_lost_and_found()?;
        builder.reserve_inodes();

        let overlay = builder.mkdir(EXT4_ROOT_INO, "overlay", None)?;
        let replacement_dir = builder.mkdir(overlay, "replacement", None)?;
        let nested_dir = builder.mkdir(replacement_dir, "nested", None)?;
        builder.add_file(
            nested_dir,
            "old.txt",
            b"old file",
            &PosixMetadata::default(),
        )?;
        builder
            .directory_caches
            .insert(replacement_dir, Default::default());
        builder
            .directory_caches
            .insert(nested_dir, Default::default());

        let free_inodes = builder.superblock.s_free_inodes_count;
        builder.overlay_host_directory(overlay, &incoming)?;

        let replacement = builder
            .lookup_dir_entry(overlay, "replacement")?
            .expect("overlay should import the replacement file");
        assert_eq!(builder.inode(replacement).file_type(), crate::ext4::S_IFREG);
        assert_eq!({ builder.inode(EXT4_ROOT_INO).i_links_count }, 4);
        assert_eq!({ builder.inode(overlay).i_links_count }, 2);
        assert_eq!({ builder.group_descriptors[0].bg_used_dirs_count_lo }, 3);
        // Two directories and one file were released; the replacement file
        // may reuse one of their inode numbers.
        assert_eq!(
            { builder.superblock.s_free_inodes_count },
            free_inodes + 3 - 1
        );
        assert!(!builder.directory_caches.contains_key(&replacement_dir));
        assert!(!builder.directory_caches.contains_key(&nested_dir));

        fs::remove_dir_all(base)?;
        Ok(())
    }

    #[test]
    fn scanned_tree_is_a_preorder_flat_arena() -> Result<()> {
        let base = std::env::temp_dir().join(format!("ext4kit-arena-{}", std::process::id()));
        let _ = fs::remove_dir_all(&base);
        fs::create_dir_all(base.join("a").join("nested"))?;
        fs::write(base.join("a").join("nested").join("child"), b"child")?;
        fs::write(base.join("z"), b"z")?;

        let tree = HostTree::scan(&base)?;
        assert_eq!(tree.entries[0].parent_index, None);
        for (index, entry) in tree.entries.iter().enumerate().skip(1) {
            assert!(entry.parent_index.is_some_and(|parent| parent < index));
        }
        let names: Vec<_> = tree
            .entries
            .iter()
            .map(|entry| entry.path.file_name().unwrap().to_str().unwrap())
            .collect();
        assert_eq!(
            names,
            [
                base.file_name().unwrap().to_str().unwrap(),
                "a",
                "nested",
                "child",
                "z"
            ]
        );

        fs::remove_dir_all(base)?;
        Ok(())
    }

    #[test]
    fn imports_a_file_below_a_created_destination_path() -> Result<()> {
        let base = std::env::temp_dir().join(format!("ext4kit-file-import-{}", std::process::id()));
        let _ = fs::remove_dir_all(&base);
        fs::create_dir_all(&base)?;
        let source = base.join("hosts");
        fs::write(&source, b"127.0.0.1 localhost\n")?;

        let mut builder =
            Ext4ImageBuilder::new(Ext4ImageBuilder::recommended_image_size_mib(&source)?);
        builder.create_root_dir()?;
        builder.create_lost_and_found()?;
        builder.reserve_inodes();
        let etc = builder.create_directory_path("/etc")?;
        let imported = builder.import_host_path(etc, &source)?;
        let (group, index) = builder.inode_location(imported);
        assert_eq!(
            builder.groups[group].inode_table[index].i_mode & 0xf000,
            crate::ext4::S_IFREG
        );

        fs::remove_dir_all(base)?;
        Ok(())
    }
}
