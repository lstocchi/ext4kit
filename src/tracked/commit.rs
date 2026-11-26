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
use crate::ext4::format::{EXT4_EXTENTS_FL, Ext4Inode, S_IFREG};
use crate::ext4::reader::ImageReader;
use crate::host::git::{
    AttrValue, CatFileBatch, CheckIgnoreSession, ConversionRoute, EolRule, ExactLen, FastImport,
    HashObject, IgnoreAnswer, ParentTree, PathAttrs, Repo, Spool, TextStat,
};
use anyhow::{Result, bail};
use std::{
    collections::{HashMap, HashSet},
    io::Write,
};

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

/// How many per-path names one collapsed note lists before the rest is
/// replaced by "..."; display only, the tree is unaffected.
const NOTE_CAP: usize = 20;

fn summarize(paths: &[String]) -> String {
    let mut shown: Vec<&str> = paths.iter().take(NOTE_CAP).map(String::as_str).collect();
    if paths.len() > NOTE_CAP {
        shown.push("...");
    }
    shown.join(", ")
}

/// The per-run streaming budget, so sparse or crafted files cannot
/// amplify a run into terabytes of zeros (holes count at full length,
/// because they become real bytes in the blob).
pub struct Budget {
    remaining: u64,
    largest: Vec<(String, u64)>,
}

/// Keeps the five largest contributors, largest first.
fn insert_largest(largest: &mut Vec<(String, u64)>, path: &str, size: u64) {
    largest.push((path.to_owned(), size));
    largest.sort_by_key(|(_, size)| std::cmp::Reverse(*size));
    largest.truncate(5);
}

/// The run's streaming budget is exhausted. Always terminal: a caller that
/// carries per-path failures forward must stop the run on this error
/// instead of recording it as one more failed path.
#[derive(Debug)]
pub struct BudgetExceeded(String);

impl std::fmt::Display for BudgetExceeded {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for BudgetExceeded {}

impl Budget {
    pub fn new(total: u64) -> Self {
        Self {
            remaining: total,
            largest: Vec::new(),
        }
    }

    /// Reserves `size` bytes for `path`, or fails with [`BudgetExceeded`]
    /// naming the largest contributors. A failed reservation charges
    /// nothing and is not recorded as a contributor; the error names it
    /// alongside the recorded ones because it is what tipped the run over.
    fn reserve(&mut self, path: &str, size: u64) -> Result<()> {
        if size > self.remaining {
            let mut largest = self.largest.clone();
            insert_largest(&mut largest, path, size);
            let largest = largest
                .iter()
                .map(|(path, size)| format!("{} ({size} bytes)", names::escape_display(path)))
                .collect::<Vec<_>>()
                .join(", ");
            return Err(BudgetExceeded(format!(
                "the run would stream more than its byte budget: {largest}"
            ))
            .into());
        }
        insert_largest(&mut self.largest, path, size);
        self.remaining -= size;
        Ok(())
    }

    /// Returns `size` for `path` when the path failed before any byte
    /// reached git or fast-import, and drops it from the largest-
    /// contributors list so a later over-budget error cannot name it.
    fn refund(&mut self, path: &str, size: u64) {
        self.remaining = self.remaining.saturating_add(size);
        if let Some(index) = self
            .largest
            .iter()
            .position(|(candidate, candidate_size)| candidate == path && *candidate_size == size)
        {
            self.largest.remove(index);
        }
    }
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

/// Streams one regular file into fast-import as a blob and returns its
/// mark. The framing invariant holds throughout: the size comes from one
/// fresh inode read, and exactly that many bytes are emitted. A returned
/// error is terminal for the stream: the caller aborts the run and drops
/// the fast-import process. A failed Normalize preparation releases the
/// budget reservation, since nothing reached fast-import; any other
/// failure keeps the charge, which no longer matters once the run stops.
// Streaming needs the reader, both git sinks, the entry, the route, the
// reference parent, and the shared budget and notes; the argument count is
// the clearest shape.
#[allow(clippy::too_many_arguments)]
pub fn stream_file(
    reader: &ImageReader,
    fast_import: &mut FastImport,
    cat_file: &mut CatFileBatch,
    entry: &PlannedEntry,
    route: ConversionRoute,
    parent: Option<&str>,
    budget: &mut Budget,
    notes: &mut Vec<String>,
) -> Result<u32> {
    // The filter route has its own function; reaching the match below
    // would panic after the blob header is already on the wire.
    if matches!(route, ConversionRoute::Filter) {
        bail!("the filter route does not stream a mark");
    }
    let inode = reader.read_inode_fresh(entry.ino)?;
    // Validate before writing anything: once `data <n>` is on the wire, a
    // file that is no longer a regular extents file would need the whole
    // frame padded just to keep the stream parseable.
    if inode.file_type() != S_IFREG || inode.i_flags & EXT4_EXTENTS_FL == 0 {
        bail!(
            "{} is no longer a regular file with extents",
            names::escape_display(&entry.repo_path)
        );
    }
    // Frame from the inode just read, not from the scan: a file the agent
    // changed between the walk and this stream is captured at its current
    // length, so the blob is neither truncated nor zero-padded, and the
    // budget charges the bytes actually emitted.
    let size = inode.file_size();
    budget.reserve(&entry.repo_path, size)?;
    match route {
        ConversionRoute::Raw => {
            let mark = fast_import.begin_blob()?;
            fast_import.begin_data(size)?;
            let mut framed = ExactLen::new(&mut *fast_import, size);
            let read = reader.read_file_from(&inode, &mut framed);
            // Finish even when the read failed: `finish` zero-pads a short
            // read and flushes, so the `data <n>` block stays framed and a
            // caller that survives the error cannot misalign the stream.
            let finish = framed.finish();
            let mismatch = framed.mismatch();
            read?;
            finish?;
            if mismatch {
                notes.push(format!(
                    "length mismatch while reading {} from the image",
                    names::escape_display(&entry.repo_path)
                ));
            }
            Ok(mark)
        }
        ConversionRoute::Normalize(rule) => {
            // Every fallible preparation step runs before the blob header:
            // once `blob\nmark :N` is on the wire, the only way out is a
            // complete `data <n>` block. When preparation fails, nothing
            // reached fast-import, so the reservation goes back.
            let Prepared {
                spool,
                stats,
                convert,
                read_mismatch,
            } = match prepare_normalize(reader, &inode, size, rule, parent, cat_file, entry) {
                Ok(prepared) => prepared,
                Err(error) => {
                    budget.refund(&entry.repo_path, size);
                    return Err(error);
                }
            };
            // Noted only now: a preparation that failed produced no blob, so
            // a mismatch note for it would describe nothing.
            if read_mismatch {
                notes.push(format!(
                    "length mismatch while reading {} from the image",
                    names::escape_display(&entry.repo_path)
                ));
            }
            let mark = fast_import.begin_blob()?;
            // The emitted bytes go through ExactLen as well: whatever a
            // conversion bug does to the length, the framing holds — the
            // framing-integrity invariant must not depend on the
            // normalizer being exact.
            if convert {
                let len = spool.len() - stats.crlf_pairs();
                fast_import.begin_data(len)?;
                let mut framed = ExactLen::new(&mut *fast_import, len);
                let written = spool.stream_crlf_to_git(|chunk| framed.write_all(chunk));
                let finish = framed.finish();
                let mismatch = framed.mismatch();
                written.map_err(|error| anyhow::anyhow!("cannot stream the spool: {error}"))?;
                finish?;
                if mismatch {
                    notes.push(format!(
                        "length mismatch while normalizing {}",
                        names::escape_display(&entry.repo_path)
                    ));
                }
            } else {
                fast_import.begin_data(spool.len())?;
                let mut framed = ExactLen::new(&mut *fast_import, spool.len());
                let written = spool.for_each_chunk(|chunk| framed.write_all(chunk));
                let finish = framed.finish();
                let mismatch = framed.mismatch();
                written.map_err(|error| anyhow::anyhow!("cannot stream the spool: {error}"))?;
                finish?;
                if mismatch {
                    notes.push(format!(
                        "length mismatch while normalizing {}",
                        names::escape_display(&entry.repo_path)
                    ));
                }
            }
            Ok(mark)
        }
        ConversionRoute::Filter => unreachable!("the route was rejected above"),
    }
}

/// A converting file spooled and classified, ready to stream.
struct Prepared {
    spool: Spool,
    stats: TextStat,
    /// Whether the CRLF-to-LF conversion applies.
    convert: bool,
    /// Whether the image read did not match the framed size.
    read_mismatch: bool,
}

/// Spools a converting file at `size`, collects its text statistics and
/// decides whether to convert it. Called before the blob header is
/// written, so every error here leaves the fast-import stream untouched.
fn prepare_normalize(
    reader: &ImageReader,
    inode: &Ext4Inode,
    size: u64,
    rule: EolRule,
    parent: Option<&str>,
    cat_file: &mut CatFileBatch,
    entry: &PlannedEntry,
) -> Result<Prepared> {
    let mut spool = Spool::create()?;
    let read_mismatch = {
        let mut framed = ExactLen::new(&mut spool, size);
        reader.read_file_from(inode, &mut framed)?;
        framed.finish()?;
        framed.mismatch()
    };
    let mut stats = TextStat::default();
    spool
        .for_each_chunk(|chunk| {
            stats.feed(chunk);
            Ok(())
        })
        .map_err(|error| anyhow::anyhow!("cannot reread the spool: {error}"))?;
    let convert = match rule {
        EolRule::Explicit => true,
        EolRule::Auto => {
            if stats.is_binary() || !stats.has_crlf() {
                false
            } else {
                // The reference-blob exception, queried lazily against the
                // fixed parent tree: no reference blob (an unborn
                // repository, or a new path) means convert, like git's
                // new-file case. The stats stream, so a huge reference blob
                // is never materialized.
                match parent {
                    Some(parent) => {
                        let reference =
                            cat_file.blob_stats(&format!("{parent}:{}", entry.repo_path))?;
                        !reference.is_some_and(|stats| stats.reference_has_crlf())
                    }
                    None => true,
                }
            }
        }
    };
    Ok(Prepared {
        spool,
        stats,
        convert,
        read_mismatch,
    })
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

/// Why one path's filter route failed. Both kinds are per-path: the
/// caller carries the path forward and keeps the run going.
#[derive(Debug)]
pub enum FilterRouteFailure {
    /// git or the clean filter failed (a missing `git-lfs`, a crashing
    /// driver); grouped per driver by [`group_filter_failures`].
    Filter(anyhow::Error),
    /// The image could not supply the content (the inode changed type, a
    /// corrupt extent tree): the path is corrupted, and blaming the filter
    /// would point the user at the wrong cause.
    Image(anyhow::Error),
}

/// Records whether a write into the inner writer failed, so a failed image
/// read can be told apart from a broken pipe into git.
struct WatchWrites<W> {
    inner: W,
    failed: bool,
}

impl<W: Write> Write for WatchWrites<W> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let written = self.inner.write(buf);
        self.failed |= written.is_err();
        written
    }

    fn flush(&mut self) -> std::io::Result<()> {
        let flushed = self.inner.flush();
        self.failed |= flushed.is_err();
        flushed
    }
}

/// Streams one file through `git hash-object -w --stdin --path`, the
/// filter route, and returns the object name plus whether the content
/// matched the framed length. A per-path failure comes back as the inner
/// [`FilterRouteFailure`]; the outer error is only [`BudgetExceeded`],
/// which is terminal for the run.
pub fn hash_object_route(
    reader: &ImageReader,
    repo: &Repo,
    entry: &PlannedEntry,
    budget: &mut Budget,
) -> Result<std::result::Result<(String, bool), FilterRouteFailure>> {
    let inode = match reader.read_inode_fresh(entry.ino) {
        Ok(inode) => inode,
        Err(error) => return Ok(Err(FilterRouteFailure::Image(error))),
    };
    if inode.file_type() != S_IFREG || inode.i_flags & EXT4_EXTENTS_FL == 0 {
        return Ok(Err(FilterRouteFailure::Image(anyhow::anyhow!(
            "{} is no longer a regular file with extents",
            names::escape_display(&entry.repo_path)
        ))));
    }
    // Charge the same fresh size the frame uses, so a sparse or grown file
    // cannot stream past the run's budget.
    let size = inode.file_size();
    budget.reserve(&entry.repo_path, size)?;
    let mut hash = match HashObject::spawn(&repo.root, &entry.repo_path, size) {
        Ok(hash) => hash,
        Err(error) => {
            // Nothing reached git, so the reservation goes back.
            budget.refund(&entry.repo_path, size);
            return Ok(Err(FilterRouteFailure::Filter(error)));
        }
    };
    // Once the writer exists, git may have received the whole zero-padded
    // frame, so the charge stays even when the path fails: refunding it
    // would let repeated failures stream past the budget.
    let mut sink = WatchWrites {
        inner: hash.writer(),
        failed: false,
    };
    let read = reader.read_file_from(&inode, &mut sink);
    let pipe_broke = sink.failed;
    // Finish even after a failed read: a clean filter that exits early
    // breaks the pipe, and git's stderr (e.g. a missing git-lfs) is the
    // real cause, not the write error.
    let finished = hash.finish();
    Ok(match (read, finished) {
        (Ok(_), Ok(result)) => Ok(result),
        // The read failed on the image side, not on the pipe.
        (Err(read), finished) if !pipe_broke => Err(FilterRouteFailure::Image(match finished {
            Ok(_) => read,
            Err(filter) => {
                anyhow::anyhow!("{read:#} (the clean filter also failed: {filter:#})")
            }
        })),
        (_, Err(filter)) => Err(FilterRouteFailure::Filter(filter)),
        // git exited cleanly without reading all of its input.
        (Err(write), Ok(_)) => Err(FilterRouteFailure::Filter(write)),
    })
}

/// One path whose filter route failed.
pub struct FilterFailure {
    /// What selected the filter route, e.g. `filter=lfs` or `ident`.
    driver: String,
    path: String,
    error: String,
}

/// Names what sent a path down the filter route, the grouping key for
/// failures: the error text itself always names the path (both ours and
/// git's), so grouping by it would never merge two failures.
pub fn filter_driver(attrs: Option<&PathAttrs>) -> String {
    let Some(attrs) = attrs else {
        return "unknown attribute".to_owned();
    };
    let named = |name: &str, value: &AttrValue| match value {
        AttrValue::Value(value) => Some(format!("{name}={value}")),
        AttrValue::True => Some(name.to_owned()),
        _ => None,
    };
    named("filter", &attrs.filter)
        .or_else(|| named("ident", &attrs.ident))
        .or_else(|| named("working-tree-encoding", &attrs.working_tree_encoding))
        .unwrap_or_else(|| "unknown attribute".to_owned())
}

/// Groups clean-filter failures into one note per driver, so a missing
/// `git-lfs` shows up as one line with a count, a few example paths and
/// the first error, instead of a thousand lines.
pub fn group_filter_failures(failures: &[FilterFailure]) -> Vec<String> {
    let mut groups: Vec<(&str, Vec<&FilterFailure>)> = Vec::new();
    for failure in failures {
        match groups
            .iter_mut()
            .find(|(driver, _)| *driver == failure.driver)
        {
            Some((_, members)) => members.push(failure),
            None => groups.push((&failure.driver, vec![failure])),
        }
    }
    groups
        .into_iter()
        .map(|(driver, members)| {
            let count = members.len();
            let noun = if count == 1 { "path" } else { "paths" };
            let paths: Vec<String> = members
                .iter()
                .map(|failure| names::escape_display(&failure.path))
                .collect();
            format!(
                "{count} {noun} failed their clean filter ({}) and were carried forward: {} (first error: {})",
                names::escape_display(driver),
                summarize(&paths),
                names::escape_display(&members[0].error)
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ext4::PosixMetadata;
    use crate::ext4::format::EXT4_ROOT_INO;
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

    /// Writes `data` as `a.txt` under `/src` in a fresh image and returns
    /// the image path plus the file's inode number.
    fn image_with_file(directory: &std::path::Path, data: &[u8]) -> (std::path::PathBuf, u32) {
        let mut builder = crate::ext4::test_support::fresh_builder().unwrap();
        let meta = PosixMetadata::default();
        let src = builder.mkdir(EXT4_ROOT_INO, "src", None).unwrap();
        let ino = builder.add_file(src, "a.txt", data, &meta).unwrap();
        let image = directory.join("stream.img");
        builder
            .write_to_file(image.to_str().expect("a UTF-8 temp path"))
            .unwrap();
        (image, ino)
    }

    #[test]
    fn streaming_a_raw_blob_preserves_the_bytes() {
        let Some((directory, repo)) = repo_fixture(&[("keep.txt", "k\n")]) else {
            return;
        };
        let data = b"raw\r\nbytes\x00";
        let (image, ino) = image_with_file(directory.path(), data);
        let entry = PlannedEntry {
            repo_path: "a.txt".to_owned(),
            kind: EntryKind::File,
            mode: 0o100644,
            ino,
            size: data.len() as u64,
            symlink_target: None,
        };
        let (reader, _state) = ImageReader::open(&image).unwrap();
        let mut budget = Budget::new(1 << 20);
        let mut notes = Vec::new();
        let mut fast_import = FastImport::spawn(&repo.root).unwrap();
        let mut cat_file = CatFileBatch::new(&repo.root).unwrap();
        let mark = stream_file(
            &reader,
            &mut fast_import,
            &mut cat_file,
            &entry,
            ConversionRoute::Raw,
            None,
            &mut budget,
            &mut notes,
        )
        .unwrap();
        fast_import
            .begin_commit("refs/heads/stream-test", None, "test", 0)
            .unwrap();
        fast_import
            .modify("100644", &format!(":{mark}"), "a.txt")
            .unwrap();
        fast_import.finish().unwrap();

        let mut cat = CatFileBatch::new(&repo.root).unwrap();
        assert_eq!(
            cat.blob("refs/heads/stream-test:a.txt").unwrap().as_deref(),
            Some(data.as_slice())
        );
        assert!(notes.is_empty(), "{notes:?}");
    }

    #[test]
    fn an_explicit_conversion_strips_crlf() {
        let Some((directory, repo)) = repo_fixture(&[("keep.txt", "k\n")]) else {
            return;
        };
        let data = b"a\r\nb\r\n";
        let (image, ino) = image_with_file(directory.path(), data);
        let entry = PlannedEntry {
            repo_path: "a.txt".to_owned(),
            kind: EntryKind::File,
            mode: 0o100644,
            ino,
            size: data.len() as u64,
            symlink_target: None,
        };
        let (reader, _state) = ImageReader::open(&image).unwrap();
        let mut budget = Budget::new(1 << 20);
        let mut notes = Vec::new();
        let mut fast_import = FastImport::spawn(&repo.root).unwrap();
        let mut cat_file = CatFileBatch::new(&repo.root).unwrap();
        let mark = stream_file(
            &reader,
            &mut fast_import,
            &mut cat_file,
            &entry,
            ConversionRoute::Normalize(EolRule::Explicit),
            None,
            &mut budget,
            &mut notes,
        )
        .unwrap();
        fast_import
            .begin_commit("refs/heads/stream-test", None, "test", 0)
            .unwrap();
        fast_import
            .modify("100644", &format!(":{mark}"), "a.txt")
            .unwrap();
        fast_import.finish().unwrap();

        let mut cat = CatFileBatch::new(&repo.root).unwrap();
        assert_eq!(
            cat.blob("refs/heads/stream-test:a.txt").unwrap().as_deref(),
            Some(b"a\nb\n".as_slice())
        );
        assert!(notes.is_empty(), "{notes:?}");
    }

    #[test]
    fn clean_filter_failures_are_grouped_by_driver() {
        let failures = vec![
            FilterFailure {
                driver: "filter=upper".to_owned(),
                path: "a.dat".to_owned(),
                error: "boom".to_owned(),
            },
            FilterFailure {
                driver: "filter=upper".to_owned(),
                path: "b.dat".to_owned(),
                error: "boom".to_owned(),
            },
            FilterFailure {
                driver: "ident".to_owned(),
                path: "c.txt".to_owned(),
                error: "nope".to_owned(),
            },
        ];
        let notes = group_filter_failures(&failures);
        assert_eq!(notes.len(), 2, "{notes:?}");
        assert!(
            notes[0].contains("2 paths failed their clean filter (filter=upper)"),
            "{notes:?}"
        );
        assert!(
            notes[0].contains("a.dat") && notes[0].contains("b.dat"),
            "{notes:?}"
        );
        assert!(
            notes[1].contains("1 path failed their clean filter (ident)"),
            "{notes:?}"
        );
    }

    #[test]
    fn a_refund_restores_the_budget_and_drops_the_contributor() {
        let mut budget = Budget::new(100);
        budget.reserve("a.bin", 60).unwrap();
        assert!(budget.reserve("b.bin", 50).is_err());
        budget.refund("a.bin", 60);
        assert!(budget.reserve("b.bin", 50).is_ok());
        assert!(
            !budget.largest.iter().any(|(path, _)| path == "a.bin"),
            "{:?}",
            budget.largest
        );
    }
    #[test]
    fn a_failed_reservation_is_typed_and_not_recorded() {
        let mut budget = Budget::new(100);
        budget.reserve("a.bin", 60).unwrap();
        let error = budget.reserve("huge.bin", 90).unwrap_err();
        assert!(
            error.downcast_ref::<BudgetExceeded>().is_some(),
            "{error:#}"
        );
        // The path that tipped the run over is named, but charges nothing
        // and does not linger as a contributor.
        assert!(error.to_string().contains("huge.bin"), "{error:#}");
        assert!(
            !budget.largest.iter().any(|(path, _)| path == "huge.bin"),
            "{:?}",
            budget.largest
        );
        let error = budget.reserve("b.bin", 50).unwrap_err();
        assert!(!error.to_string().contains("huge.bin"), "{error:#}");
        assert!(budget.reserve("c.bin", 40).is_ok());
    }

    /// A file-route entry for `a.txt` at `ino`.
    fn filter_entry(ino: u32, size: u64) -> PlannedEntry {
        PlannedEntry {
            repo_path: "a.txt".to_owned(),
            kind: EntryKind::File,
            mode: 0o100644,
            ino,
            size,
            symlink_target: None,
        }
    }

    #[test]
    fn filter_route_failures_name_their_cause() {
        let Some((directory, repo)) = repo_fixture(&[(
            ".gitattributes",
            "a.txt filter=missing
",
        )]) else {
            return;
        };
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
        assert!(succeeds(&[
            "config",
            "filter.missing.clean",
            "ext4kit-no-such-filter-command"
        ]));
        assert!(succeeds(&["config", "filter.missing.required", "true"]));
        let data = b"content
";
        let (image, ino) = image_with_file(directory.path(), data);
        let (reader, _state) = ImageReader::open(&image).unwrap();

        // A required clean filter that cannot run is the filter's fault.
        let mut budget = Budget::new(1 << 20);
        let outcome = hash_object_route(
            &reader,
            &repo,
            &filter_entry(ino, data.len() as u64),
            &mut budget,
        )
        .unwrap();
        assert!(
            matches!(outcome, Err(FilterRouteFailure::Filter(_))),
            "{outcome:?}"
        );

        // An inode that is no longer a regular file is the image's.
        let outcome =
            hash_object_route(&reader, &repo, &filter_entry(EXT4_ROOT_INO, 0), &mut budget)
                .unwrap();
        assert!(
            matches!(outcome, Err(FilterRouteFailure::Image(_))),
            "{outcome:?}"
        );

        // An exhausted budget is not a per-path failure: it is the outer,
        // terminal error.
        let mut budget = Budget::new(1);
        let error = hash_object_route(
            &reader,
            &repo,
            &filter_entry(ino, data.len() as u64),
            &mut budget,
        )
        .unwrap_err();
        assert!(
            error.downcast_ref::<BudgetExceeded>().is_some(),
            "{error:#}"
        );
    }
}
