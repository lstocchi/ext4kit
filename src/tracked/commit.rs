//! Turning a scan of the image into git commits. The same code produces
//! the **base** commit `create` writes (parented on HEAD-at-create) and
//! every **head** commit of a snapshot; the only differences are the
//! parent and the target ref.
//!
//! Content never passes through memory unbounded: regular files stream
//! from the image reader through the [`crate::host::git::ExactLen`] framing
//! writer straight into fast-import (raw), into a bounded spool (converting),
//! or into `git hash-object` (the filter route). Unreadable, oversized or
//! host-incompatible paths copy their parent-tree entry forward instead
//! of becoming deletions, so a cherry-pick never removes something the
//! walk simply could not see.

use super::{
    Mapping, MappingKind,
    names::{self, NameClass},
    scan::{EntryKind, MappingScan},
};
use crate::host::git::{CheckIgnoreSession, IgnoreAnswer, ParentTree};
use anyhow::Result;
use std::collections::{HashMap, HashSet};

/// Options for the base-commit run.
pub struct BaseOptions {
    pub label: String,
    pub include_ignored: bool,
    /// Per-blob limit; a larger file is reported and carried forward.
    pub max_blob_bytes: u64,
    /// Per-run budget across every blob; exceeding it aborts before
    /// `done`, so nothing is published.
    pub max_total_bytes: u64,
    pub unix_time: u64,
}

/// The per-repo result of a base run.
pub struct RepoBase {
    pub root: String,
    pub base_commit: String,
}

/// Everything `create` needs after the base refs are written.
pub struct BaseOutcome {
    pub repos: Vec<RepoBase>,
    /// Human-readable notes: incompatible names, nested repositories,
    /// carried-forward paths, high-risk changes.
    pub notes: Vec<String>,
}

/// One entry bound for the commit, after ignore filtering.
pub struct PlannedEntry {
    pub repo_path: String,
    pub kind: EntryKind,
    pub mode: u16,
    pub ino: u32,
    pub size: u64,
    pub symlink_target: Option<Vec<u8>>,
}

/// One mapping's contribution to the commit.
pub struct PlannedMapping {
    /// The mapping root resolved and walked; `false` means the mapping is
    /// skipped entirely — no `D`, no `M`, the parent content stays.
    pub emit_delete: bool,
    pub included: Vec<PlannedEntry>,
    /// Repo paths whose parent-tree entry is copied forward.
    pub carry_forward: Vec<String>,
}

/// Per-path notes collapsed into list-level counts, so a hostile or huge
/// tree cannot flood the output; the tree itself is unaffected.
#[derive(Default)]
pub struct NoteCounters {
    ignored: usize,
    never_in_tree: Vec<String>,
    portability: Vec<String>,
}

/// Classifies one mapping's scan into included entries, carry-forward
/// paths and notes, applying the ignore rules with the parent tree's
/// tracked-prefix guard and exact-path exemption. The context the walk
/// needs is inherently the whole plan's state, so the argument count is
/// the clearest shape.
#[allow(clippy::too_many_arguments)]
pub fn plan_mapping(
    scan: &MappingScan,
    mapping: &Mapping,
    parent_tree: &ParentTree,
    gitlinks: &[(String, String)],
    ignore: &mut CheckIgnoreSession,
    options: &BaseOptions,
    counters: &mut NoteCounters,
    notes: &mut Vec<String>,
) -> Result<PlannedMapping> {
    if let Some(reason) = &scan.replaced {
        notes.push(format!(
            "mapping root {} replaced: {reason} (mapping skipped)",
            mapping.image
        ));
        return Ok(PlannedMapping {
            emit_delete: false,
            included: Vec::new(),
            carry_forward: Vec::new(),
        });
    }
    // An unlistable root is the same decision as a replaced one: the walk
    // saw nothing, and a `D` would delete the parent tree's content
    // without any evidence the image actually lost it.
    if let Some(reason) = &scan.root_unreadable {
        notes.push(format!(
            "mapping root {} is unreadable: {reason} (mapping skipped)",
            mapping.image
        ));
        return Ok(PlannedMapping {
            emit_delete: false,
            included: Vec::new(),
            carry_forward: Vec::new(),
        });
    }
    let mut planned = PlannedMapping {
        emit_delete: true,
        included: Vec::new(),
        carry_forward: Vec::new(),
    };

    // A file mapping is one entry at the mapping's own repo path.
    if mapping.kind == MappingKind::File {
        if let Some(entry) = scan.entries.first() {
            // Same rule as query_children: only names plan_entry will
            // actually ask about reach the session.
            let answers = if queryable(entry) {
                ignore.check(&[(
                    mapping.repo_relative.clone(),
                    entry.kind == EntryKind::Directory,
                )])?
            } else {
                Vec::new()
            };
            plan_entry(
                entry,
                &mapping.repo_path(&entry.rel_path),
                parent_tree,
                gitlinks,
                answers.first(),
                options,
                counters,
                notes,
                &mut planned,
            );
        }
        return Ok(planned);
    }

    // Group the walk's entries by their parent directory, preserving the
    // DFS order; the root's children form the first batch.
    let mut children: HashMap<&str, Vec<usize>> = HashMap::new();
    for (index, entry) in scan.entries.iter().enumerate() {
        children
            .entry(parent_dir(&entry.rel_path))
            .or_default()
            .push(index);
    }
    let mut answers: HashMap<usize, IgnoreAnswer> = HashMap::new();
    query_children("", &children, scan, mapping, ignore, &mut answers)?;

    // Prefix sets for the ancestor test below: the walk is parent-first,
    // so an entry is skipped exactly when it or one of its ancestors is a
    // pruned directory or a re-emitted gitlink.
    let gitlink_paths: HashSet<String> = gitlinks
        .iter()
        .map(|(gitlink, _)| gitlink.clone())
        .collect();
    let mut pruned: HashSet<String> = HashSet::new();
    for (index, entry) in scan.entries.iter().enumerate() {
        let repo_path = mapping.repo_path(&entry.rel_path);
        // A pruned or re-emitted-gitlink subtree is not walked further.
        if ancestor_in(&repo_path, &pruned) || ancestor_in(&repo_path, &gitlink_paths) {
            continue;
        }
        match plan_entry(
            entry,
            &repo_path,
            parent_tree,
            gitlinks,
            answers.get(&index),
            options,
            counters,
            notes,
            &mut planned,
        ) {
            EntryPlan::WalkDir => {
                // Not pruned: query this directory's children as one
                // batch, ready for when the preorder reaches them.
                query_children(
                    &entry.rel_path,
                    &children,
                    scan,
                    mapping,
                    ignore,
                    &mut answers,
                )?;
            }
            EntryPlan::Skip if entry.kind == EntryKind::Directory => {
                pruned.insert(repo_path);
            }
            _ => {}
        }
    }
    Ok(planned)
}

/// What the walk does after one entry is classified.
pub enum EntryPlan {
    /// The subtree is not walked further (pruned, boundary, excluded).
    Skip,
    /// A directory the walk descends into; its children are queried.
    WalkDir,
    /// A leaf entry; nothing to descend.
    Leaf,
}

/// Classifies one entry against the name rules, the parent tree and the
/// ignore answer, recording it as included, carried forward or excluded.
// The walk's context is the same whole-plan state as plan_mapping's; the
// argument count is the clearest shape here too.
#[allow(clippy::too_many_arguments)]
pub fn plan_entry(
    entry: &super::scan::ScannedEntry,
    repo_path: &str,
    parent_tree: &ParentTree,
    gitlinks: &[(String, String)],
    answer: Option<&IgnoreAnswer>,
    options: &BaseOptions,
    counters: &mut NoteCounters,
    notes: &mut Vec<String>,
    planned: &mut PlannedMapping,
) -> EntryPlan {
    // Names first: some classes cannot become git paths at all, so no
    // parent entry can exist for them either. Every printed path goes
    // through escape_display: these are exactly the classes that exist
    // to catch hostile names, and a control byte must not become a
    // terminal escape sequence.
    match entry.verdict.class {
        NameClass::Corrupted => {
            notes.push(format!(
                "corrupted name {} ({})",
                names::escape_display(repo_path),
                entry.verdict.reason.as_deref().unwrap_or("invalid")
            ));
            return EntryPlan::Skip;
        }
        NameClass::NeverInTree => {
            counters
                .never_in_tree
                .push(names::escape_display(repo_path));
            return EntryPlan::Skip;
        }
        NameClass::IncompatibleHost => {
            // A HEAD entry at the same lossy path is re-emitted with its
            // raw bytes by the unrepresentable list, so the exclusion
            // costs nothing there; without one the path is simply gone.
            let disposition = if parent_tree.entries.contains_key(repo_path) {
                "kept from HEAD"
            } else {
                "excluded"
            };
            notes.push(format!(
                "name {} cannot exist on this host: {} ({disposition})",
                names::escape_display(repo_path),
                entry.verdict.reason.as_deref().unwrap_or("incompatible")
            ));
            return EntryPlan::Skip;
        }
        NameClass::PortabilityWarning => {
            counters.portability.push(names::escape_display(repo_path))
        }
        NameClass::Keep => {}
    }
    // Structural corruption and failed ignore queries carry the parent
    // entry forward instead of becoming deletions.
    let corrupted = entry.corrupted.as_deref().or_else(|| match answer {
        Some(Err(reason)) => Some(reason),
        _ => None,
    });
    if let Some(reason) = corrupted {
        notes.push(format!(
            "{} is unreadable ({}); carried forward",
            names::escape_display(repo_path),
            reason
        ));
        planned.carry_forward.push(repo_path.to_owned());
        return EntryPlan::Skip;
    }
    let ignored = matches!(answer, Some(Ok(true)));
    // An exact blob, symlink or gitlink in the parent tree is never
    // treated as ignored, whatever the index says.
    let exempt = parent_tree
        .entries
        .get(repo_path)
        .is_some_and(|entry| !entry.is_tree());
    match entry.kind {
        EntryKind::Directory => {
            // The ignore decision comes first: an ignored nested
            // repository is excluded like any other ignored directory (a
            // HEAD entry at the path, if any, is carried forward by the
            // caller), not reported as a walked boundary. A tracked
            // descendant still keeps the directory, exactly as for a
            // non-repository ignored directory.
            if !options.include_ignored
                && ignored
                && !exempt
                && !parent_tree.has_tracked_descendants(repo_path)
            {
                counters.ignored += 1;
                return EntryPlan::Skip;
            }
            // A nested repository is a boundary: either its gitlink is
            // re-emitted (handled by the caller's gitlink list) or it is
            // skipped with its parent content carried forward.
            if entry.contains_git {
                if !gitlinks.iter().any(|(gitlink, _)| gitlink == repo_path) {
                    notes.push(format!(
                        "nested repository {} skipped",
                        names::escape_display(repo_path)
                    ));
                    planned.carry_forward.push(repo_path.to_owned());
                }
                return EntryPlan::Skip;
            }
            EntryPlan::WalkDir
        }
        EntryKind::File | EntryKind::Symlink => {
            if !options.include_ignored && ignored && !exempt {
                counters.ignored += 1;
                return EntryPlan::Leaf;
            }
            planned.included.push(PlannedEntry {
                repo_path: repo_path.to_owned(),
                kind: entry.kind,
                mode: entry.mode,
                ino: entry.ino,
                size: entry.size,
                symlink_target: entry.symlink_target.clone(),
            });
            EntryPlan::Leaf
        }
        kind => {
            notes.push(format!(
                "{} is a {} and cannot be represented in git; excluded",
                names::escape_display(repo_path),
                kind_name(kind)
            ));
            EntryPlan::Skip
        }
    }
}

/// Whether plan_entry will read an entry's ignore answer: every other
/// class returns before the answer matters, and its name may be exactly
/// the hostile bytes (a NUL, pathspec magic) that must never reach the
/// check-ignore session — each one kills the process and costs a
/// respawn, and four abort the base.
pub fn queryable(entry: &super::scan::ScannedEntry) -> bool {
    matches!(
        entry.verdict.class,
        NameClass::Keep | NameClass::PortabilityWarning
    ) && entry.corrupted.is_none()
}

/// Sends one directory's children to the ignore session as a batch and
/// stores their answers. Only [`queryable`] children are sent.
pub fn query_children(
    dir: &str,
    children: &HashMap<&str, Vec<usize>>,
    scan: &MappingScan,
    mapping: &Mapping,
    ignore: &mut CheckIgnoreSession,
    answers: &mut HashMap<usize, IgnoreAnswer>,
) -> Result<()> {
    let Some(group) = children.get(dir) else {
        return Ok(());
    };
    let queried: Vec<usize> = group
        .iter()
        .copied()
        .filter(|&index| queryable(&scan.entries[index]))
        .collect();
    if queried.is_empty() {
        return Ok(());
    }
    let queries: Vec<(String, bool)> = queried
        .iter()
        .map(|&index| {
            let entry = &scan.entries[index];
            (
                mapping.repo_path(&entry.rel_path),
                entry.kind == EntryKind::Directory,
            )
        })
        .collect();
    for (index, answer) in queried.into_iter().zip(ignore.check(&queries)?) {
        answers.insert(index, answer);
    }
    Ok(())
}

/// The parent directory of a mapping-relative path; `""` for depth one.
fn parent_dir(rel_path: &str) -> &str {
    rel_path.rsplit_once('/').map_or("", |(dir, _)| dir)
}

/// Whether `path`, or one of its ancestor directories, is in `dirs`.
///
/// This is the same test as "`path` is `dir` or lies under it", but walks
/// `path`'s own prefixes instead of scanning every directory: the walk is
/// parent-first, and a per-entry scan of both sets (with a `format!` per
/// comparison) is quadratic on a tree with many pruned or gitlink
/// boundaries.
fn ancestor_in(path: &str, dirs: &HashSet<String>) -> bool {
    let mut candidate = path;
    loop {
        if dirs.contains(candidate) {
            return true;
        }
        match candidate.rsplit_once('/') {
            Some((parent, _)) => candidate = parent,
            None => return false,
        }
    }
}

fn kind_name(kind: EntryKind) -> &'static str {
    match kind {
        EntryKind::Device => "device node",
        EntryKind::Fifo => "FIFO",
        EntryKind::Socket => "socket",
        _ => "special file",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::host::git::Repo;

    /// A minimal committed repository holding `files`, or `None` (with a
    /// note) when git is missing, unless `EXT4KIT_REQUIRE_GIT` is set —
    /// the same skip pattern as the integration tests. `core.autocrlf` is
    /// pinned so blob contents are byte-deterministic.
    fn repo_fixture(files: &[(&str, &str)]) -> Option<(tempfile::TempDir, Repo)> {
        if !crate::host::git::git_available() {
            if std::env::var_os("EXT4KIT_REQUIRE_GIT").is_some() {
                panic!("EXT4KIT_REQUIRE_GIT is set but git was not found");
            }
            eprintln!("note: git not found; skipping git-backed unit tests");
            return None;
        }
        let directory = tempfile::tempdir().expect("a fresh temp directory");
        let repo = Repo::at(directory.path());
        let succeeds = |args: &[&str]| {
            std::process::Command::new("git")
                .arg("-C")
                .arg(&repo.root)
                .args(args)
                .output()
                .expect("git runs")
                .status
                .success()
        };
        assert!(succeeds(&["init", "--quiet"]));
        assert!(succeeds(&["config", "user.email", "t@example.com"]));
        assert!(succeeds(&["config", "user.name", "T"]));
        assert!(succeeds(&["config", "commit.gpgsign", "false"]));
        assert!(succeeds(&["config", "core.autocrlf", "false"]));
        for (name, content) in files {
            std::fs::write(directory.path().join(name), content).unwrap();
        }
        assert!(succeeds(&["add", "-A"]));
        assert!(succeeds(&["commit", "--quiet", "-m", "init"]));
        Some((directory, repo))
    }

    #[test]
    fn an_unreadable_mapping_root_skips_the_mapping_in_the_plan() {
        let Some((directory, repo)) = repo_fixture(&[("a.txt", "a\n")]) else {
            return;
        };
        let root = directory.path().to_string_lossy().into_owned();
        let mapping = Mapping {
            host: root.clone(),
            image: "/opt/src".to_owned(),
            kind: MappingKind::Dir,
            repo: Some(root),
            repo_relative: String::new(),
        };
        let scan = MappingScan {
            root_ino: Some(2),
            replaced: None,
            root_unreadable: Some(
                "the mapping root cannot be listed: the inode owns no blocks".to_owned(),
            ),
            entries: Vec::new(),
        };
        let mut ignore = CheckIgnoreSession::new(&repo.root).unwrap();
        let options = BaseOptions {
            label: "out".to_owned(),
            include_ignored: false,
            max_blob_bytes: 1024 * 1024,
            max_total_bytes: 1024 * 1024,
            unix_time: 0,
        };
        let mut counters = NoteCounters::default();
        let mut notes = Vec::new();
        let planned = plan_mapping(
            &scan,
            &mapping,
            &ParentTree::default(),
            &[],
            &mut ignore,
            &options,
            &mut counters,
            &mut notes,
        )
        .unwrap();
        // No `D` for the mapped subtree: the parent tree's content under it
        // survives the base commit.
        assert!(!planned.emit_delete);
        assert!(planned.included.is_empty());
        assert_eq!(planned.carry_forward, Vec::<String>::new());
        assert_eq!(notes.len(), 1, "{notes:?}");
        assert!(notes[0].contains("/opt/src is unreadable"), "{notes:?}");
        assert!(notes[0].contains("mapping skipped"), "{notes:?}");
    }
}
