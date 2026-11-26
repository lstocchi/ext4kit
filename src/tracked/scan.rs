//! The image-side walk that feeds a commit: an iterative DFS over one
//! mapping target, collecting names, kinds, inode numbers, symlink targets
//! and the stability fields, but no content — content streams into
//! fast-import later, from the inode numbers recorded here.
//!
//! Everything read from the image is hostile input. Names are classified
//! by [`super::names`]; structures the reader cannot validate become
//! per-path corruption notes instead of aborting the run; directories the
//! walk must not descend into (`.git` aliases, host-incompatible names,
//! nested-repository boundaries, directories it cannot list) are skipped
//! with the reason recorded on the directory's own entry, so the commit
//! layer can carry the parent tree's content forward instead of deleting
//! it.

use super::names::{self, NameVerdict};
use crate::ext4::format::*;
use crate::ext4::reader::ImageReader;
use std::collections::HashSet;

/// The longest symlink target accepted, `PATH_MAX - 1`.
const MAX_SYMLINK_TARGET: u64 = 4095;

/// What one walked path is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EntryKind {
    File,
    Directory,
    Symlink,
    /// Character or block device: git cannot represent it.
    Device,
    Fifo,
    Socket,
    /// A crafted `i_mode` type the kernel would not produce.
    Unknown,
}

impl EntryKind {
    fn of(inode: &Ext4Inode) -> Self {
        match inode.file_type() {
            S_IFREG => Self::File,
            S_IFDIR => Self::Directory,
            S_IFLNK => Self::Symlink,
            S_IFCHR | S_IFBLK => Self::Device,
            S_IFIFO => Self::Fifo,
            S_IFSOCK => Self::Socket,
            _ => Self::Unknown,
        }
    }
}

/// The inode fields the publication-gate stability check re-reads.
/// `ctime` and `mtime` pair the seconds field with the extra word, so
/// nanosecond changes are visible.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stability {
    pub ctime: (u32, u32),
    pub mtime: (u32, u32),
    pub size: u64,
    pub links: u16,
}

impl Stability {
    fn of(inode: &Ext4Inode, size: u64) -> Self {
        Self {
            ctime: (inode.i_ctime, inode.i_ctime_extra),
            mtime: (inode.i_mtime, inode.i_mtime_extra),
            size,
            links: inode.i_links_count,
        }
    }
}

/// One walked path under a mapping target.
#[derive(Clone, Debug)]
pub struct ScannedEntry {
    /// The path relative to the mapping target, `/`-separated, no leading
    /// or trailing separator. For a file mapping the single entry's path
    /// is empty.
    pub rel_path: String,
    pub kind: EntryKind,
    pub verdict: NameVerdict,
    /// A directory below the mapping root that contains a `.git` entry:
    /// a nested-repository boundary. The walk did not descend into it.
    pub contains_git: bool,
    pub mode: u16,
    pub ino: u32,
    pub size: u64,
    /// The target bytes, for symlinks.
    pub symlink_target: Option<Vec<u8>>,
    pub stability: Stability,
    /// Corruption found while walking this path; the entry stays in the
    /// list so the commit layer can carry its parent-tree content forward.
    pub corrupted: Option<String>,
}

/// The walk of one mapping target.
#[derive(Clone, Debug, Default)]
pub struct MappingScan {
    /// The resolved target inode; `None` when the target (or an ancestor)
    /// is missing, a symlink, or the wrong kind, in which case `replaced`
    /// names the offending component.
    pub root_ino: Option<u32>,
    pub replaced: Option<String>,
    /// The mapping root itself could not be listed, so the walk stopped
    /// before producing any entry. Like `replaced`, this skips the whole
    /// mapping — no entry, no deletion — but the root did resolve, so the
    /// commit layer can tell the two cases apart in its notes.
    pub root_unreadable: Option<String>,
    /// DFS preorder: directories before their children.
    pub entries: Vec<ScannedEntry>,
}

/// A directory waiting to be listed, with the index of its own entry (the
/// mapping root itself is not an entry).
struct DirFrame {
    ino: u32,
    /// Path prefix for children, with a trailing separator unless empty.
    prefix: String,
    entry_index: Option<usize>,
    /// Depth below the mapping root; the root's children are at depth 1.
    depth: u32,
}

/// Walks one mapping target in `reader`.
///
/// The target path is resolved component by component from the filesystem
/// root, following **no symlinks at all**: an agent that replaces `/opt`
/// with a symlink would otherwise redirect `/opt/src` to other image
/// content, so any symlink on the way counts as a replaced root. A
/// replaced root skips the whole mapping — no entry, no deletion — and the
/// report names the offending component.
pub fn scan_mapping(reader: &ImageReader, target: &str, is_file: bool) -> MappingScan {
    let mut scan = MappingScan::default();
    let root = match resolve_root(reader, target, is_file) {
        Ok(root) => root,
        Err(replaced) => {
            scan.replaced = Some(replaced);
            return scan;
        }
    };
    scan.root_ino = Some(root);
    if is_file {
        // The mapping target is the single file; its "relative path" is
        // empty, so the commit layer maps it to the mapping's own
        // repo-relative path.
        let name = target.rsplit('/').next().unwrap_or_default();
        match entry_for(reader, root, "", name.as_bytes()) {
            Ok(entry) => scan.entries.push(entry),
            Err(reason) => scan
                .entries
                .push(corrupted_entry("", root, reason, name.as_bytes())),
        }
        return scan;
    }

    let mut seen_dirs = HashSet::from([root]);
    let mut stack = vec![DirFrame {
        ino: root,
        prefix: String::new(),
        entry_index: None,
        depth: 0,
    }];
    while let Some(frame) = stack.pop() {
        let listing = match reader.directory_entries_raw(frame.ino) {
            Ok(listing) => listing,
            Err(error) => {
                match frame.entry_index {
                    Some(index) => {
                        scan.entries[index].corrupted =
                            Some(format!("the directory cannot be listed: {error}"))
                    }
                    // The mapping root itself is unlistable: the walk has
                    // nothing to descend into, and the mapping must be
                    // skipped rather than turned into a deletion of
                    // everything the parent tree holds under it.
                    None => {
                        scan.root_unreadable =
                            Some(format!("the mapping root cannot be listed: {error}"))
                    }
                }
                continue;
            }
        };
        // A `.git` entry below the mapping root marks a nested repository:
        // git never adds its files to the outer repo, so the walk does not
        // descend and none of its children become entries (the mapping
        // root itself is the repo and always walks).
        let contains_git =
            frame.depth > 0 && listing.iter().any(|(name, _)| names::is_dotgit(name));
        if let Some(index) = frame.entry_index {
            scan.entries[index].contains_git = contains_git;
        }
        if contains_git {
            continue;
        }
        let mut children = Vec::new();
        for (name, ino) in listing {
            let rel_path = format!("{}{}", frame.prefix, String::from_utf8_lossy(&name));
            let mut entry = match entry_for(reader, ino, &rel_path, &name) {
                Ok(entry) => entry,
                Err(reason) => corrupted_entry(&rel_path, ino, reason, &name),
            };
            // entry_for classifies the name once the inode kind is known:
            // the `.gitmodules`-as-symlink rule needs it.
            let walkable = matches!(
                entry.verdict.class,
                names::NameClass::Keep | names::NameClass::PortabilityWarning
            ) && entry.corrupted.is_none()
                && entry.kind == EntryKind::Directory;
            if walkable && !seen_dirs.insert(ino) {
                // Only crafted metadata can list one directory inode twice;
                // walking it again would be walking a cycle.
                entry.corrupted = Some("the directory inode appears more than once".to_owned());
                scan.entries.push(entry);
            } else if walkable {
                let entry_index = scan.entries.len();
                scan.entries.push(entry);
                children.push(DirFrame {
                    ino,
                    prefix: format!("{rel_path}/"),
                    entry_index: Some(entry_index),
                    depth: frame.depth + 1,
                });
            } else {
                // Names that keep the path out of the tree, kinds git
                // cannot represent and unreadable inodes are reported on
                // their own entry; the subtree below an excluded name is
                // not descended.
                scan.entries.push(entry);
            }
        }
        // Push reversed so the DFS preorder follows on-disk order.
        for child in children.into_iter().rev() {
            stack.push(child);
        }
    }
    scan
}

/// Resolves `target` from the filesystem root without following any
/// symlink, and checks the final kind against the mapping.
fn resolve_root(reader: &ImageReader, target: &str, is_file: bool) -> Result<u32, String> {
    let kind = if is_file { "file" } else { "directory" };
    let mut current = EXT4_ROOT_INO;
    let components: Vec<&str> = target.split('/').filter(|c| !c.is_empty()).collect();
    for (index, component) in components.iter().enumerate() {
        let so_far = format!("/{}", components[..=index].join("/"));
        let listing = reader
            .directory_entries_raw(current)
            .map_err(|error| format!("cannot list the parent of {so_far}: {error}"))?;
        let Some(&(_, ino)) = listing
            .iter()
            .find(|(name, _)| name == component.as_bytes())
        else {
            return Err(format!("{so_far} does not exist"));
        };
        let inode = reader
            .read_inode(ino)
            .map_err(|error| format!("cannot read {so_far}: {error}"))?;
        let is_last = index == components.len() - 1;
        match inode.file_type() {
            S_IFLNK => return Err(format!("{so_far} is a symlink, not a real {kind}")),
            S_IFDIR if is_last && is_file => return Err(format!("{so_far} is not a {kind}")),
            S_IFDIR if is_last => return Ok(ino),
            S_IFDIR => current = ino,
            S_IFREG if is_last && is_file => return Ok(ino),
            _ => return Err(format!("{so_far} is not a {kind}")),
        }
    }
    Err(format!(
        "{target} is the filesystem root; a mapping target is always a named entry"
    ))
}

/// Builds the entry for one listed child, or the reason it is corrupt.
/// The name is classified here, after the inode kind is known, because
/// the `.gitmodules`-as-symlink rule of git's `verify_path` depends on
/// the kind.
fn entry_for(
    reader: &ImageReader,
    ino: u32,
    rel_path: &str,
    name: &[u8],
) -> Result<ScannedEntry, String> {
    let inode = reader
        .read_inode(ino)
        .map_err(|error| format!("the inode cannot be read: {error}"))?;
    let kind = EntryKind::of(&inode);
    let verdict = names::validate_component(name, kind == EntryKind::Symlink);
    let size = match kind {
        EntryKind::File => u64::from(inode.i_size_lo) | (u64::from(inode.i_size_high) << 32),
        _ => u64::from(inode.i_size_lo),
    };
    let mut entry = ScannedEntry {
        rel_path: rel_path.to_owned(),
        kind,
        verdict,
        contains_git: false,
        mode: inode.i_mode,
        ino,
        size,
        symlink_target: None,
        stability: Stability::of(&inode, size),
        corrupted: None,
    };
    match kind {
        EntryKind::Symlink => {
            if size > MAX_SYMLINK_TARGET {
                return Err(format!(
                    "the symlink target is {size} bytes, longer than {MAX_SYMLINK_TARGET}"
                ));
            }
            entry.symlink_target = Some(
                reader
                    .symlink_target_bytes(&inode)
                    .map_err(|error| format!("the symlink target cannot be read: {error}"))?,
            );
        }
        EntryKind::Unknown => {
            let mode = inode.i_mode;
            return Err(format!(
                "the inode mode {mode:#07o} has an unknown file type"
            ));
        }
        _ => {}
    }
    Ok(entry)
}

/// A placeholder entry for a path whose inode could not be read at all.
/// The name is still classified — an unreadable inode must not lose its
/// `.git` handling — with the symlink flag unknown, so only the
/// kind-dependent rules cannot fire; the entry is excluded as corrupt
/// regardless.
fn corrupted_entry(rel_path: &str, ino: u32, reason: String, name: &[u8]) -> ScannedEntry {
    ScannedEntry {
        rel_path: rel_path.to_owned(),
        kind: EntryKind::Unknown,
        verdict: names::validate_component(name, false),
        contains_git: false,
        mode: 0,
        ino,
        size: 0,
        symlink_target: None,
        stability: Stability {
            ctime: (0, 0),
            mtime: (0, 0),
            size: 0,
            links: 0,
        },
        corrupted: Some(reason),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ext4::test_support::fresh_builder;
    use crate::ext4::{Ext4ImageBuilder, PosixMetadata};
    use names::NameClass;

    /// A builder with a mapping-shaped tree under /opt/src.
    fn builder() -> Ext4ImageBuilder {
        let mut builder = fresh_builder().unwrap();
        let meta = PosixMetadata::default();
        let opt = builder.mkdir(EXT4_ROOT_INO, "opt", None).unwrap();
        let src = builder.mkdir(opt, "src", None).unwrap();
        builder.add_file(src, "a.txt", b"hello\n", &meta).unwrap();
        let sub = builder.mkdir(src, "sub", None).unwrap();
        builder.add_file(sub, "b.txt", b"world\n", &meta).unwrap();
        builder.add_symlink(src, "link", "a.txt", &meta).unwrap();
        builder
            .add_file(src, "run.sh", b"#!/bin/sh\n", &meta)
            .unwrap();
        let git_dir = builder.mkdir(src, ".git", None).unwrap();
        builder
            .add_file(git_dir, "config", b"[core]\n", &meta)
            .unwrap();
        let vendor = builder.mkdir(src, "vendor", None).unwrap();
        let lib = builder.mkdir(vendor, "lib", None).unwrap();
        let nested_git = builder.mkdir(lib, ".git", None).unwrap();
        builder
            .add_file(nested_git, "HEAD", b"ref\n", &meta)
            .unwrap();
        builder.add_file(lib, "Makefile", b"all:\n", &meta).unwrap();
        builder.commit_metadata();
        builder
    }

    fn reader(builder: &Ext4ImageBuilder) -> ImageReader {
        ImageReader::from_memory(builder.disk_data.to_vec())
            .unwrap()
            .0
    }

    fn scan(builder: &Ext4ImageBuilder, target: &str) -> MappingScan {
        scan_mapping(&reader(builder), target, false)
    }

    #[test]
    fn walks_the_whole_mapping_with_directories_before_children() {
        let builder = builder();
        let scan = scan(&builder, "/opt/src");
        assert_eq!(
            scan.root_ino,
            Some(builder.resolve_directory("/opt/src").unwrap().unwrap())
        );
        let mut paths: Vec<&str> = scan.entries.iter().map(|e| e.rel_path.as_str()).collect();
        paths.sort_unstable();
        assert_eq!(
            paths,
            vec![
                ".git",
                "a.txt",
                "link",
                "run.sh",
                "sub",
                "sub/b.txt",
                "vendor",
                "vendor/lib"
            ]
        );
        // DFS preorder: every directory entry precedes its children.
        let position = |path: &str| {
            scan.entries
                .iter()
                .position(|e| e.rel_path == path)
                .unwrap_or(usize::MAX)
        };
        assert!(position("sub") < position("sub/b.txt"));
    }

    #[test]
    fn dotgit_entries_are_never_in_the_tree_and_not_walked() {
        let scan = scan(&builder(), "/opt/src");
        let git = scan.entries.iter().find(|e| e.rel_path == ".git").unwrap();
        assert_eq!(git.verdict.class, NameClass::NeverInTree);
        // The .git directory's own children do not appear.
        assert!(!scan.entries.iter().any(|e| e.rel_path == ".git/config"));
    }

    #[test]
    fn nested_repositories_are_boundaries() {
        let scan = scan(&builder(), "/opt/src");
        let boundary = scan
            .entries
            .iter()
            .find(|e| e.rel_path == "vendor/lib")
            .unwrap();
        assert!(boundary.contains_git);
        // The boundary's children were never listed; only the boundary
        // itself appears under vendor/.
        let under_vendor: Vec<&str> = scan
            .entries
            .iter()
            .map(|e| e.rel_path.as_str())
            .filter(|path| path.starts_with("vendor/"))
            .collect();
        assert_eq!(under_vendor, vec!["vendor/lib"]);
    }

    #[test]
    fn kinds_modes_and_targets_are_collected() {
        let scan = scan(&builder(), "/opt/src");
        let link = scan.entries.iter().find(|e| e.rel_path == "link").unwrap();
        assert_eq!(link.kind, EntryKind::Symlink);
        assert_eq!(link.symlink_target.as_deref(), Some(b"a.txt".as_slice()));
        let run = scan
            .entries
            .iter()
            .find(|e| e.rel_path == "run.sh")
            .unwrap();
        assert_eq!(run.kind, EntryKind::File);
        assert_eq!(run.size, 10);
        let sub = scan.entries.iter().find(|e| e.rel_path == "sub").unwrap();
        assert_eq!(sub.kind, EntryKind::Directory);
        // '.', '..', and one per subdirectory: b.txt is a file and does not
        // count, so a directory holding only files has two links.
        assert_eq!(sub.stability.links, 2);
    }

    #[test]
    fn a_replaced_target_is_reported() {
        let fixture = builder();
        let missing = scan(&fixture, "/opt/missing");
        assert!(missing.root_ino.is_none());
        assert!(missing.entries.is_empty());
        assert!(
            missing
                .replaced
                .as_deref()
                .unwrap()
                .contains("does not exist"),
            "{:?}",
            missing.replaced
        );

        let mut redirected = builder();
        let meta = PosixMetadata::default();
        redirected
            .add_symlink(EXT4_ROOT_INO, "elsewhere", "/opt/src", &meta)
            .unwrap();
        redirected.commit_metadata();
        let replaced = scan(&redirected, "/elsewhere/src");
        assert!(replaced.replaced.as_deref().unwrap().contains("symlink"));
    }

    #[test]
    fn file_mappings_scan_the_single_file() {
        let builder = builder();
        let reader = reader(&builder);
        let scan = scan_mapping(&reader, "/opt/src/a.txt", true);
        assert!(scan.root_ino.is_some());
        assert_eq!(scan.entries.len(), 1);
        assert_eq!(scan.entries[0].rel_path, "");
        assert_eq!(scan.entries[0].kind, EntryKind::File);
        assert_eq!(scan.entries[0].size, 6);

        // A file mapping whose target became a directory is replaced.
        let scan = scan_mapping(&reader, "/opt/src", true);
        assert!(scan.replaced.as_deref().unwrap().contains("not a file"));
    }

    #[test]
    fn gitmodules_is_never_in_the_tree_only_as_a_symlink() {
        // The name rule depends on the inode kind, so classification runs
        // after the inode is read: a `.gitmodules` symlink is refused, a
        // regular `.gitmodules` file is kept.
        let meta = PosixMetadata::default();
        let mut with_symlink = builder();
        let src = with_symlink.resolve_directory("/opt/src").unwrap().unwrap();
        with_symlink
            .add_symlink(src, ".gitmodules", "https://evil.example/x", &meta)
            .unwrap();
        with_symlink.commit_metadata();
        let scanned = scan(&with_symlink, "/opt/src");
        let symlink = scanned
            .entries
            .iter()
            .find(|e| e.rel_path == ".gitmodules")
            .unwrap();
        assert_eq!(symlink.verdict.class, NameClass::NeverInTree);
        assert_eq!(symlink.kind, EntryKind::Symlink);

        // A regular file of the same name stays: git accepts it.
        let mut with_file = builder();
        let src = with_file.resolve_directory("/opt/src").unwrap().unwrap();
        with_file
            .add_file(src, ".gitmodules", b"[submodule]\n", &meta)
            .unwrap();
        with_file.commit_metadata();
        let scanned = scan(&with_file, "/opt/src");
        let file = scanned
            .entries
            .iter()
            .find(|e| e.rel_path == ".gitmodules")
            .unwrap();
        assert_eq!(file.verdict.class, NameClass::Keep);
    }

    #[test]
    fn an_unreadable_inode_keeps_its_name_classification() {
        let mut builder = builder();
        builder.commit_metadata();
        let src = builder.resolve_directory("/opt/src").unwrap().unwrap();
        // Corrupt the `.git` record's inode number, so read_inode fails:
        // the `.git` alias must still classify as NeverInTree. The name
        // starts eight bytes into the record (inode, rec_len, lengths).
        let blocks = builder.get_dir_blocks(src).unwrap();
        let block = blocks[0] as usize * BLOCK_SIZE;
        let mut bytes = builder.disk_data.to_vec();
        let name_at = (block + 24..block + BLOCK_SIZE - 4)
            .find(|&offset| &bytes[offset..offset + 4] == b".git")
            .expect("the .git record is in the first directory block");
        bytes[name_at - 8..name_at - 4].copy_from_slice(&u32::MAX.to_le_bytes());
        let reader = ImageReader::from_memory(bytes).unwrap().0;
        let scan = scan_mapping(&reader, "/opt/src", false);
        let git = scan.entries.iter().find(|e| e.rel_path == ".git").unwrap();
        assert!(git.corrupted.is_some(), "the inode must be unreadable");
        assert_eq!(git.verdict.class, NameClass::NeverInTree);
    }

    #[test]
    fn an_unlistable_directory_corrupts_only_its_own_entry() {
        let mut builder = builder();
        // Serialize first: commit_metadata would overwrite a direct disk
        // edit, because the builder owns the inode tables in memory.
        builder.commit_metadata();
        let sub = builder.resolve_directory("/opt/src/sub").unwrap().unwrap();
        // Zero the directory's i_blocks so the reader's ownership bound
        // rejects listing it.
        let (group, index) = builder.inode_location(sub);
        let offset = builder.group_descriptors[group].bg_inode_table_lo as usize * BLOCK_SIZE
            + index * INODE_SIZE
            + 28;
        builder.disk_data[offset..offset + 4].copy_from_slice(&0u32.to_le_bytes());
        let scan = scan(&builder, "/opt/src");
        let sub = scan.entries.iter().find(|e| e.rel_path == "sub").unwrap();
        assert!(
            sub.corrupted
                .as_deref()
                .unwrap()
                .contains("cannot be listed"),
            "{:?}",
            sub.corrupted
        );
        assert!(!scan.entries.iter().any(|e| e.rel_path == "sub/b.txt"));
        assert!(scan.entries.iter().any(|e| e.rel_path == "a.txt"));
    }

    #[test]
    fn an_unlistable_mapping_root_skips_the_mapping() {
        let mut builder = builder();
        // Serialize first: commit_metadata would overwrite a direct disk
        // edit, because the builder owns the inode tables in memory.
        builder.commit_metadata();
        let src = builder.resolve_directory("/opt/src").unwrap().unwrap();
        // Zero the mapping root's i_blocks so the reader's ownership bound
        // rejects listing it: the walk must skip the mapping entirely, not
        // report it as a deletable empty tree.
        let (group, index) = builder.inode_location(src);
        let offset = builder.group_descriptors[group].bg_inode_table_lo as usize * BLOCK_SIZE
            + index * INODE_SIZE
            + 28;
        builder.disk_data[offset..offset + 4].copy_from_slice(&0u32.to_le_bytes());
        let scan = scan(&builder, "/opt/src");
        assert!(
            scan.root_unreadable
                .as_deref()
                .unwrap()
                .contains("cannot be listed"),
            "{:?}",
            scan.root_unreadable
        );
        assert!(scan.entries.is_empty());
        assert!(scan.replaced.is_none());
    }

    #[test]
    fn a_hostile_name_is_classified_not_fatal() {
        let mut builder = builder();
        let src = builder.resolve_directory("/opt/src").unwrap().unwrap();
        let meta = PosixMetadata::default();
        builder.add_file(src, "bad:name", b"x", &meta).unwrap();
        builder.commit_metadata();
        let scan = scan(&builder, "/opt/src");
        let bad = scan
            .entries
            .iter()
            .find(|e| e.rel_path == "bad:name")
            .unwrap();
        if cfg!(windows) {
            assert_eq!(bad.verdict.class, NameClass::IncompatibleHost);
        } else {
            assert_eq!(bad.verdict.class, NameClass::PortabilityWarning);
        }
    }
}
