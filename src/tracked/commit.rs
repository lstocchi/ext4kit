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
    names::{self, NameClass, NameVerdict},
    scan::{self, EntryKind, MappingScan},
};
use crate::ext4::format::{EXT4_EXTENTS_FL, Ext4Inode, S_IFREG};
use crate::ext4::reader::ImageReader;
use crate::host::git::{
    AttrValue, CatFileBatch, CheckIgnoreSession, ConversionRoute, EolRule, ExactLen, FastImport,
    HashObject, IgnoreAnswer, ParentTree, PathAttrs, Repo, Spool, TextStat, TreeEntry, check_attr,
    conversion_route,
};
use anyhow::{Result, bail};
use std::{
    collections::{HashMap, HashSet},
    io::Write,
    path::Path,
};

/// Names whose changes get attention before apply at any depth: git reads
/// `.gitattributes` and `.gitmodules` from every directory they govern,
/// and direnv loads an `.envrc` for the tree below it, so a nested one
/// changes behavior just as much as a root one.
const HIGH_RISK_NAMES: &[&str] = &[".gitattributes", ".gitmodules", ".envrc"];

/// Root-only build entry points; a nested one is the project's business.
const HIGH_RISK_ROOT_PATHS: &[&str] = &["Makefile", "package.json"];

/// Paths whose changes get attention before apply: attribute and
/// submodule configuration, CI hooks, build entry points.
fn is_high_risk(path: &str) -> bool {
    path.rsplit('/')
        .next()
        .is_some_and(|name| HIGH_RISK_NAMES.contains(&name))
        || HIGH_RISK_ROOT_PATHS.contains(&path)
        || path.starts_with(".github/workflows/")
        || path.starts_with(".husky/")
        || path.starts_with(".vscode/")
}

/// The base ref of `label`, built in one place so the pre-check and the
/// write can never disagree.
fn base_ref(label: &str) -> String {
    format!("refs/ext4kit/{label}/base")
}

/// Options for the base-commit run.
pub struct BaseOptions {
    pub label: String,
    pub include_ignored: bool,
    /// Per-blob limit; a larger file is reported and left out (HEAD's
    /// version is kept where present).
    pub max_blob_bytes: u64,
    /// Per-run budget across every blob; exceeding it aborts before
    /// `done`, so nothing is published.
    pub max_total_bytes: u64,
    pub unix_time: u64,
}

/// The per-repo result of a base run.
#[derive(Debug)]
pub struct RepoBase {
    pub root: String,
    pub base_commit: String,
}

/// Everything `create` needs after the base refs are written.
#[derive(Debug)]
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

/// Per-path notes collapsed into list-level counts with a bounded path
/// list, so a hostile or huge tree can neither flood the output nor make
/// the run's memory grow with it; the tree itself is unaffected.
#[derive(Default)]
pub struct NoteCounters {
    ignored: usize,
    /// One collapsed note per class, in first-seen order.
    classes: Vec<PathNote>,
}

/// One class of per-path notes: how many paths it holds, and at most
/// [`NOTE_CAP`] of them kept for display.
#[derive(Default)]
struct PathNote {
    /// The text after the count, e.g. `corrupted names are excluded`.
    legend: String,
    /// Per-class context folded into the note, such as the first read
    /// error; only the first one is kept.
    extra: Option<String>,
    /// The first [`NOTE_CAP`] paths; the rest live only in `total`.
    shown: Vec<String>,
    total: usize,
}

impl PathNote {
    fn add(&mut self, path: String) {
        self.total += 1;
        if self.shown.len() < NOTE_CAP {
            self.shown.push(path);
        }
    }

    fn render(&self) -> String {
        let mut text = format!("{} {}", self.total, self.legend);
        if let Some(extra) = &self.extra {
            text.push_str(&format!(" ({extra})"));
        }
        text.push_str(": ");
        text.push_str(&self.shown.join(", "));
        if self.total > self.shown.len() {
            text.push_str(", ...");
        }
        text
    }
}

impl NoteCounters {
    /// Records one path under `legend`, creating the class on first use.
    fn note(&mut self, legend: &str, path: String) {
        self.class_for(legend).add(path);
    }

    /// Records one path under `legend`, keeping `extra` (the first one
    /// only) as per-class context such as the first error message.
    fn note_with_extra(&mut self, legend: &str, extra: String, path: String) {
        let class = self.class_for(legend);
        if class.extra.is_none() {
            class.extra = Some(extra);
        }
        class.add(path);
    }

    fn class_for(&mut self, legend: &str) -> &mut PathNote {
        let index = match self.classes.iter().position(|class| class.legend == legend) {
            Some(index) => index,
            None => {
                self.classes.push(PathNote {
                    legend: legend.to_owned(),
                    ..PathNote::default()
                });
                self.classes.len() - 1
            }
        };
        &mut self.classes[index]
    }

    fn flush(&mut self, notes: &mut Vec<String>) {
        if self.ignored > 0 {
            notes.push(format!("{} ignored paths excluded", self.ignored));
            self.ignored = 0;
        }
        notes.extend(self.classes.iter().map(PathNote::render));
        self.classes.clear();
    }
}

/// How many per-path names one collapsed note keeps before the rest is
/// dropped from the text; the count still covers every path, and the tree
/// is unaffected.
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

/// Writes `refs/ext4kit/<label>/base` in every repository the mappings
/// cover, parented on HEAD (no parent in an unborn repository), with the
/// imported content normalized and ignore-filtered exactly the way every
/// later snapshot will be.
pub fn write_base_refs(
    image: &Path,
    mappings: &[Mapping],
    options: &BaseOptions,
) -> Result<BaseOutcome> {
    let (reader, state) = ImageReader::open(image)?;
    if state.mounted {
        bail!(
            "{} is mounted or was not cleanly unmounted; the base is read \
             from the image ext4kit just wrote",
            image.display()
        );
    }

    let mut notes = Vec::new();
    let mut budget = Budget::new(options.max_total_bytes);
    // Repositories in first-seen order, with the mappings covering each.
    let mut repos: Vec<(String, Vec<usize>)> = Vec::new();
    for (index, mapping) in mappings.iter().enumerate() {
        let Some(root) = &mapping.repo else {
            // create already reported untracked mappings; this covers
            // library callers.
            notes.push(format!(
                "mapping {} is not inside a git repository; it is not tracked",
                mapping.host
            ));
            continue;
        };
        match repos.iter_mut().find(|(repo, _)| repo == root) {
            Some((_, group)) => group.push(index),
            None => repos.push((root.clone(), vec![index])),
        }
    }

    let mut bases = Vec::new();
    let reference = base_ref(&options.label);
    // Refs this run created, with the SHA it wrote: cleanup deletes only
    // an exact match, so it can never remove a ref that moved.
    let mut started: Vec<(String, String)> = Vec::new();
    for (root, indexes) in repos {
        let repo = Repo::at(root.as_str());
        // The label may be taken by a ref this run did not create — any
        // case/head/base variant, since the loose-ref backend folds case.
        match repo.label_in_use(&options.label) {
            Ok(false) => {}
            Ok(true) => {
                return Err(rollback(
                    anyhow::anyhow!(
                        "label {} is in use in {}; pass --label to choose another",
                        options.label,
                        repo.root.display()
                    ),
                    &reference,
                    &started,
                ));
            }
            Err(error) => return Err(rollback(error, &reference, &started)),
        }
        let repo_mappings: Vec<&Mapping> = indexes.iter().map(|&i| &mappings[i]).collect();
        match write_repo_base(
            &reader,
            &repo,
            &repo_mappings,
            options,
            &mut budget,
            &mut notes,
        ) {
            Ok(base) => {
                started.push((root.clone(), base.clone()));
                bases.push(RepoBase {
                    root,
                    base_commit: base,
                });
            }
            Err(error) => {
                // A write that failed after fast-import updated the ref
                // (for example the SHA query) still leaves a ref this run
                // owns; the pre-check proved it was absent, so record it
                // for exact cleanup.
                if let Ok(Some(sha)) = repo.rev_parse_optional(&reference)
                    && !started.iter().any(|(created, _)| created == &root)
                {
                    started.push((root.clone(), sha));
                }
                return Err(rollback(error, &reference, &started));
            }
        }
    }
    Ok(BaseOutcome {
        repos: bases,
        notes,
    })
}

/// Removes the base refs this run created, by SHA, and returns `error`
/// with any cleanup failure attached, so a later "label in use" has a
/// cause instead of a silent leftover.
fn rollback(
    mut error: anyhow::Error,
    reference: &str,
    started: &[(String, String)],
) -> anyhow::Error {
    let mut failures = Vec::new();
    for (root, sha) in started {
        let repo = Repo::at(root.as_str());
        let owned = [(reference.to_owned(), sha.clone())];
        if let Err(cleanup) = repo.delete_refs_exact(&owned) {
            failures.push(format!("{}: {cleanup:#}", repo.root.display()));
        }
    }
    if !failures.is_empty() {
        error = error.context(format!(
            "the base refs this run created could not all be removed ({})",
            failures.join("; ")
        ));
    }
    error
}

/// One repository's base commit. Returns the commit SHA.
fn write_repo_base(
    reader: &ImageReader,
    repo: &Repo,
    mappings: &[&Mapping],
    options: &BaseOptions,
    budget: &mut Budget,
    notes: &mut Vec<String>,
) -> Result<String> {
    let parent = repo.head_commit()?;
    let parent_tree = match &parent {
        Some(parent) => ParentTree::from_entries(repo.ls_tree(parent)?),
        None => ParentTree::default(),
    };

    let mut counters = NoteCounters::default();

    // Gitlinks under the mapped subtrees are re-emitted unchanged: the
    // D / deleteall would otherwise delete them, and the scan never adds
    // the image's copy of a submodule's files to the outer repository.
    // This vector is for planning and the submodule notes, which work on
    // lossy names; the emission iterates the listing directly so distinct
    // raw paths sharing one lossy rendering are not collapsed.
    let mut seen_gitlinks: HashSet<String> = HashSet::new();
    let gitlinks: Vec<(String, String)> = parent_tree
        .listing
        .iter()
        .filter(|entry| entry.is_gitlink() && covered_by(mappings, &entry.path))
        .filter(|entry| seen_gitlinks.insert(entry.path.clone()))
        .map(|entry| (entry.path.clone(), entry.sha.clone()))
        .collect();

    // Parent-tree paths no walk on this host could ever re-add — some
    // component of the name cannot exist here — are re-emitted with their
    // raw bytes: the D / deleteall must not delete them, and a lossy path
    // (U+FFFD) would corrupt the name. Gitlinks with such a name take this
    // route too, instead of the plain gitlink re-emit. Only the top-most
    // such entry is re-emitted; a tree entry carries its descendants.
    let mut incompatible: Vec<&TreeEntry> = parent_tree
        .listing
        .iter()
        .filter(|entry| covered_by(mappings, &entry.path) && has_incompatible_component(entry))
        .collect();
    incompatible.sort_by(|a, b| a.raw_path.cmp(&b.raw_path));
    // Keyed by raw bytes: two distinct raw paths can share one lossy name,
    // and the compatible twin must still be re-emitted by name.
    let incompatible_paths: HashSet<&[u8]> = incompatible
        .iter()
        .map(|entry| entry.raw_path.as_slice())
        .collect();
    // The top-most incompatible entries: a descendant of another entry is
    // carried by that ancestor's raw re-emit. The set of accepted raw
    // paths answers "under one already?" in O(depth) instead of scanning
    // every entry, and the sorted order guarantees an ancestor is seen
    // before its descendants.
    let mut accepted_ancestors: HashSet<&[u8]> = HashSet::new();
    let mut unrepresentable: Vec<&TreeEntry> = Vec::new();
    for entry in &incompatible {
        let entry = *entry;
        let mut end = entry.raw_path.len();
        let mut under = false;
        while let Some(position) = entry.raw_path[..end].iter().rposition(|&byte| byte == b'/') {
            if accepted_ancestors.contains(&entry.raw_path[..position]) {
                under = true;
                break;
            }
            end = position;
        }
        if !under {
            accepted_ancestors.insert(entry.raw_path.as_slice());
            unrepresentable.push(entry);
        }
    }
    // Git itself may refuse to re-create some of these paths on this host
    // (Windows rejects control bytes and reserved names a repository
    // cloned from elsewhere can still carry in HEAD); re-emitting one
    // would abort the whole base, so only the accepted subset is
    // re-emitted and the rest is reported, never dropped silently.
    // An index holds no tree entries (`update-index` ignores a 040000
    // record), so a directory is probed through its non-tree
    // descendants: they share the incompatible component, and git
    // accepting any of them means it accepts the directory's name too.
    // The non-tree entries are sorted once, so each probe is a binary
    // search plus its matches, not a scan of the whole parent tree.
    let mut non_trees: Vec<&TreeEntry> = parent_tree
        .listing
        .iter()
        .filter(|entry| !entry.is_tree())
        .collect();
    non_trees.sort_by(|a, b| a.raw_path.cmp(&b.raw_path));
    let descendants_of = |tree: &TreeEntry| -> Vec<&TreeEntry> {
        let mut prefix = tree.raw_path.clone();
        prefix.push(b'/');
        let start = non_trees.partition_point(|entry| entry.raw_path < prefix);
        non_trees[start..]
            .iter()
            .copied()
            .take_while(|entry| entry.raw_path.starts_with(&prefix))
            .collect()
    };
    let probe: Vec<(String, String, Vec<u8>)> = unrepresentable
        .iter()
        .flat_map(|entry| {
            if entry.is_tree() {
                descendants_of(entry)
            } else {
                vec![*entry]
            }
        })
        .map(|entry| {
            (
                entry.mode.clone(),
                entry.sha.clone(),
                entry.raw_path.clone(),
            )
        })
        .collect();
    let accepted = repo.accepted_tree_paths(&probe)?;
    // The top-most paths git refuses: nothing under one may be re-emitted
    // either, or a subtree reference would restore what the note says is
    // deleted.
    let mut rejected: HashSet<Vec<u8>> = HashSet::new();
    let unrepresentable: Vec<&TreeEntry> = unrepresentable
        .into_iter()
        .filter(|entry| {
            let is_accepted = if entry.is_tree() {
                descendants_of(entry)
                    .iter()
                    .any(|descendant| accepted.contains(&descendant.raw_path))
            } else {
                accepted.contains(&entry.raw_path)
            };
            if !is_accepted {
                rejected.insert(entry.raw_path.clone());
                counters.note(
                    "paths cannot be re-created by git on this host; they and their tracked \
                     descendants are deleted from the baseline",
                    names::escape_display(&entry.path),
                );
            }
            is_accepted
        })
        .collect();

    let mut ignore = CheckIgnoreSession::new(&repo.root)?;
    let mut planned: Vec<PlannedMapping> = Vec::new();
    let mut scans: Vec<MappingScan> = Vec::new();
    for mapping in mappings {
        let scan = scan::scan_mapping(reader, &mapping.image, mapping.kind == MappingKind::File);
        planned.push(plan_mapping(
            &scan,
            mapping,
            &parent_tree,
            &gitlinks,
            &mut ignore,
            options,
            &mut counters,
            notes,
        )?);
        scans.push(scan);
    }
    for (path, _) in &gitlinks {
        // A mapping whose root was replaced or unreadable was never
        // walked, so it says nothing about whether the submodule is still
        // there; only a walked mapping covering the path can report it
        // deleted.
        let walked = |scan: &MappingScan| scan.replaced.is_none() && scan.root_unreadable.is_none();
        let covered = mappings.iter().zip(&scans).any(|(mapping, scan)| {
            walked(scan) && relative_to(path, &mapping.repo_relative).is_some()
        });
        // The scan's boundary entry (a submodule checkout holds `.git`)
        // proves the image still has the directory.
        let present = mappings.iter().zip(&scans).any(|(mapping, scan)| {
            walked(scan)
                && relative_to(path, &mapping.repo_relative).is_some_and(|rel| {
                    rel.is_empty() || scan.entries.iter().any(|entry| entry.rel_path == rel)
                })
        });
        if covered && !present {
            notes.push(format!(
                "submodule {} deleted in the image (pointer preserved)",
                names::escape_display(path)
            ));
        }
    }

    // Attributes decide the conversion route; one batch for every
    // included regular file.
    let file_paths: Vec<String> = planned
        .iter()
        .flat_map(|plan| plan.included.iter())
        .filter(|entry| entry.kind == EntryKind::File)
        .map(|entry| entry.repo_path.clone())
        .collect();
    let attributes = check_attr(&repo.root, &file_paths)?;
    let autocrlf = repo.autocrlf();

    // Phase A: stream every blob. Blob commands precede the commit, and
    // M lines inside it reference their marks.
    let mut fast_import = FastImport::spawn(&repo.root)?;
    let mut rows: Vec<(String, String, String)> = Vec::new();
    let mut late_carry: Vec<String> = Vec::new();
    // Filter-route failures, grouped by driver at the end: a missing
    // git-lfs must be one line, not a thousand.
    let mut filter_failures: Vec<FilterFailure> = Vec::new();
    let mut cat_file = CatFileBatch::new(&repo.root)?;
    // Hard links share a blob per conversion class, except the two routes
    // whose result depends on the path: the filter route (a clean command
    // may use `%f`) and the auto eol rule (its decision reads each path's
    // reference blob, so two links to one inode can legitimately become
    // different blobs — the tracked twin keeps its CRLF, the new one is
    // normalized).
    let mut shared_marks: HashMap<(u32, ConversionRoute), String> = HashMap::new();
    let mut case_seen: HashMap<String, String> = HashMap::new();
    for plan in &planned {
        for entry in &plan.included {
            let path = &entry.repo_path;
            if is_high_risk(path) && !parent_tree.entries.contains_key(path) {
                counters.note("high-risk paths added", names::escape_display(path));
            }
            if case_folding() {
                // Lower-casing is closer to the host tables than
                // upper-casing (which folds ß to SS); the note names both
                // paths so the collision can be resolved.
                if let Some(other) = case_seen.insert(path.to_lowercase(), path.clone())
                    && other != *path
                {
                    counters.note(
                        "case collisions cannot be checked out together",
                        format!(
                            "{} and {}",
                            names::escape_display(&other),
                            names::escape_display(path)
                        ),
                    );
                }
            }
            match entry.kind {
                EntryKind::Symlink => {
                    let target = entry.symlink_target.as_deref().unwrap_or_default();
                    let mark = fast_import.blob(target)?;
                    rows.push(("120000".to_owned(), format!(":{mark}"), path.clone()));
                }
                EntryKind::File => {
                    if entry.size > options.max_blob_bytes {
                        note_oversized(
                            path,
                            options.max_blob_bytes,
                            &mut counters,
                            &mut late_carry,
                        );
                        continue;
                    }
                    let route = attributes.get(path).map_or(ConversionRoute::Raw, |attrs| {
                        conversion_route(attrs, autocrlf)
                    });
                    let source = match route {
                        ConversionRoute::Filter => {
                            // `hash_object_route` charges the budget with
                            // the inode it reads, so the driver must not
                            // reserve for this route again.
                            match hash_object_route(
                                reader,
                                repo,
                                entry,
                                budget,
                                options.max_blob_bytes,
                            )? {
                                Ok((sha, mismatch)) => {
                                    if mismatch {
                                        counters.note(
                                            "files changed size while being read from the image",
                                            names::escape_display(path),
                                        );
                                    }
                                    sha
                                }
                                Err(FilterRouteFailure::Filter(error)) => {
                                    // A clean filter that is missing or
                                    // crashing must not cost the run its
                                    // tracking: the path is carried
                                    // forward and reported with its
                                    // siblings that failed the same way.
                                    filter_failures.push(FilterFailure {
                                        driver: filter_driver(attributes.get(path)),
                                        path: path.clone(),
                                        error: format!("{error:#}"),
                                    });
                                    late_carry.push(path.clone());
                                    continue;
                                }
                                Err(FilterRouteFailure::Image(error)) => {
                                    // The image could not supply the
                                    // content: the path is corrupt, and
                                    // blaming the filter would point at
                                    // the wrong cause.
                                    note_file_read(path, &error, &mut counters, &mut late_carry);
                                    continue;
                                }
                                Err(FilterRouteFailure::TooLarge) => {
                                    // The file grew past the per-blob limit
                                    // between the walk and the stream, so
                                    // nothing was read or charged.
                                    note_oversized(
                                        path,
                                        options.max_blob_bytes,
                                        &mut counters,
                                        &mut late_carry,
                                    );
                                    continue;
                                }
                            }
                        }
                        shared @ (ConversionRoute::Raw
                        | ConversionRoute::Normalize(EolRule::Explicit)) => {
                            match shared_marks.get(&(entry.ino, shared)) {
                                Some(mark) => mark.clone(),
                                None => match stream_file(
                                    reader,
                                    &mut fast_import,
                                    &mut cat_file,
                                    entry,
                                    shared,
                                    parent.as_deref(),
                                    budget,
                                    options.max_blob_bytes,
                                    &mut counters,
                                ) {
                                    Ok(mark) => {
                                        let mark = format!(":{mark}");
                                        shared_marks.insert((entry.ino, shared), mark.clone());
                                        mark
                                    }
                                    Err(error) => {
                                        handle_stream_error(
                                            path,
                                            error,
                                            options.max_blob_bytes,
                                            &mut counters,
                                            &mut late_carry,
                                        )?;
                                        continue;
                                    }
                                },
                            }
                        }
                        ConversionRoute::Normalize(EolRule::Auto) => {
                            // Never shared between hard links: the convert
                            // decision reads this path's reference blob.
                            match stream_file(
                                reader,
                                &mut fast_import,
                                &mut cat_file,
                                entry,
                                ConversionRoute::Normalize(EolRule::Auto),
                                parent.as_deref(),
                                budget,
                                options.max_blob_bytes,
                                &mut counters,
                            ) {
                                Ok(mark) => format!(":{mark}"),
                                Err(error) => {
                                    handle_stream_error(
                                        path,
                                        error,
                                        options.max_blob_bytes,
                                        &mut counters,
                                        &mut late_carry,
                                    )?;
                                    continue;
                                }
                            }
                        }
                    };
                    let mode = if entry.mode & 0o100 != 0 {
                        "100755"
                    } else {
                        "100644"
                    };
                    rows.push((mode.to_owned(), source, path.clone()));
                }
                _ => unreachable!("planning keeps only files and symlinks"),
            }
        }
    }
    drop(cat_file);
    notes.extend(group_filter_failures(&filter_failures));

    // Phase B: the commit — clear each mapped subtree, re-emit gitlinks
    // and carried-forward entries, then add the streamed rows.
    let targets = mappings
        .iter()
        .map(|mapping| mapping.image.as_str())
        .collect::<Vec<_>>()
        .join(", ");
    let reference = base_ref(&options.label);
    let message = format!("base of {} ({targets})", options.label);
    fast_import.begin_commit(&reference, parent.as_deref(), &message, options.unix_time)?;
    let mut deleted_all = false;
    for (mapping, plan) in mappings.iter().zip(&planned) {
        if !plan.emit_delete {
            continue;
        }
        if mapping.repo_relative.is_empty() {
            fast_import.delete_all()?;
            deleted_all = true;
        } else {
            fast_import.delete_path(&mapping.repo_relative)?;
        }
    }
    for entry in parent_tree
        .listing
        .iter()
        .filter(|entry| entry.is_gitlink() && covered_by(mappings, &entry.path))
    {
        if incompatible_paths.contains(entry.raw_path.as_slice()) {
            // An incompatible gitlink goes through the raw-byte path
            // below (or was reported as unrepresentable): its lossy
            // name would rename the submodule pointer.
            continue;
        }
        fast_import.modify("160000", &entry.sha, &entry.path)?;
    }
    for entry in &unrepresentable {
        fast_import.modify_raw(&entry.mode, &entry.sha, &entry.raw_path)?;
    }
    if deleted_all {
        // A mapping that was skipped (a replaced or unreadable root) emits
        // no `D` and no carry-forward, so a sibling root mapping's
        // `deleteall` must not take its HEAD content with it. Its leaves
        // are re-emitted individually: a tree reference would also restore
        // descendants the host probe rejected.
        for (mapping, plan) in mappings.iter().zip(&planned) {
            if plan.emit_delete || mapping.repo_relative.is_empty() {
                continue;
            }
            reemit_subtree(
                &mut fast_import,
                &non_trees,
                mapping.repo_relative.as_bytes(),
                &rejected,
            )?;
        }
    }
    let mut carried: Vec<&String> = planned
        .iter()
        .flat_map(|plan| plan.carry_forward.iter())
        .collect();
    carried.extend(late_carry.iter());
    for path in carried {
        let Some(entry) = parent_tree.entries.get(path.as_str()) else {
            continue;
        };
        if has_incompatible_component(entry) {
            // The map is keyed by the lossy path, so this can be a
            // different raw name; the real entry was either re-emitted
            // with its raw bytes above or reported as unrepresentable.
            continue;
        }
        if entry.is_tree() {
            // Re-emit the subtree leaf by leaf: `M 040000 <sha>` would
            // also restore descendants the host probe rejected, which the
            // run reported as deleted.
            reemit_subtree(&mut fast_import, &non_trees, &entry.raw_path, &rejected)?;
        } else {
            fast_import.modify(&entry.mode, &entry.sha, path)?;
        }
    }
    // Rows are image-side names that passed the host name rules, so they
    // are valid UTF-8 and can never be the same raw name as a HEAD path
    // re-emitted above; there is no lossy-path skip here.
    for (mode, source, path) in rows {
        fast_import.modify(&mode, &source, &path)?;
    }
    fast_import.finish()?;
    let base = repo.rev_parse(&reference)?;
    // A high-risk file that was already in HEAD and changed needs a note
    // too, not only a newly added one. Additions were recorded per entry
    // above; the diff catches modifications. Advisory only: a diff failure
    // (a broken external diff, an unreadable object) must not cost the run
    // the base ref it just wrote.
    if let Some(parent) = &parent {
        match repo.changed_paths(parent, &base) {
            Ok(changed) => {
                for changed in changed {
                    if is_high_risk(&changed) && parent_tree.entries.contains_key(&changed) {
                        counters.note("high-risk paths changed", names::escape_display(&changed));
                    }
                }
            }
            Err(error) => notes.push(format!(
                "the paths the base changed could not be listed ({error:#}); high-risk changes \
                 may be missing from these notes"
            )),
        }
    }
    counters.flush(notes);
    Ok(base)
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
            let repo_path = mapping.repo_path(&entry.rel_path);
            // The scan classified the *image target's* basename; the git
            // path is the mapping's repo-relative one, which a library
            // caller can name independently, so the verdict is recomputed
            // for the components that actually become the path.
            let mut entry = entry.clone();
            entry.verdict = repo_path_verdict(&repo_path, entry.kind == EntryKind::Symlink);
            // Same rule as query_children: only names plan_entry will
            // actually ask about reach the session.
            let answers = if queryable(&entry) {
                ignore.check(&[(
                    mapping.repo_relative.clone(),
                    entry.kind == EntryKind::Directory,
                )])?
            } else {
                Vec::new()
            };
            plan_entry(
                &entry,
                &repo_path,
                parent_tree,
                gitlinks,
                answers.first(),
                options,
                counters,
                &mut planned,
            );
            if planned.included.is_empty() {
                // The single entry could not be included (its path is not
                // representable, it is corrupt, or it is no longer a file):
                // skipping the mapping leaves HEAD's version in place, so
                // no `D` may be emitted for it.
                planned.emit_delete = false;
            }
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
    planned: &mut PlannedMapping,
) -> EntryPlan {
    // Names first: some classes cannot become git paths at all, so no
    // parent entry can exist for them either. Every printed path goes
    // through escape_display: these are exactly the classes that exist
    // to catch hostile names, and a control byte must not become a
    // terminal escape sequence.
    match entry.verdict.class {
        NameClass::Corrupted => {
            counters.note(
                "corrupted names are excluded",
                format!(
                    "{} ({})",
                    names::escape_display(repo_path),
                    entry.verdict.reason.as_deref().unwrap_or("invalid")
                ),
            );
            return EntryPlan::Skip;
        }
        NameClass::NeverInTree => {
            counters.note(
                "paths never enter the tree (.git aliases)",
                names::escape_display(repo_path),
            );
            return EntryPlan::Skip;
        }
        NameClass::IncompatibleHost => {
            // A HEAD entry at the same lossy path is re-emitted with its
            // raw bytes by the unrepresentable list, so the exclusion
            // costs nothing there; without one the path is simply gone.
            let (legend, disposition) = if parent_tree.entries.contains_key(repo_path) {
                (
                    "names cannot exist on this host; their HEAD version is kept",
                    "kept from HEAD",
                )
            } else {
                (
                    "names cannot exist on this host and are excluded",
                    "excluded",
                )
            };
            counters.note(
                legend,
                format!(
                    "{} ({}; {disposition})",
                    names::escape_display(repo_path),
                    entry.verdict.reason.as_deref().unwrap_or("incompatible")
                ),
            );
            return EntryPlan::Skip;
        }
        NameClass::PortabilityWarning => counters.note(
            "names are not portable to Windows",
            names::escape_display(repo_path),
        ),
        NameClass::Keep => {}
    }
    // Structural corruption and failed ignore queries carry the parent
    // entry forward instead of becoming deletions.
    let corrupted = entry.corrupted.as_deref().or_else(|| match answer {
        Some(Err(reason)) => Some(reason),
        _ => None,
    });
    if let Some(reason) = corrupted {
        counters.note_with_extra(
            "paths are unreadable; the baseline keeps HEAD's version where present",
            reason.to_owned(),
            names::escape_display(repo_path),
        );
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
                    counters.note(
                        "nested repositories are skipped",
                        names::escape_display(repo_path),
                    );
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
            counters.note(
                "paths have a kind git cannot represent and are excluded",
                format!("{} ({})", names::escape_display(repo_path), kind_name(kind)),
            );
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

/// A per-file failure: the image could not supply this file, so the
/// driver carries the path forward and keeps the run going.
#[derive(Debug)]
pub struct FileRead(anyhow::Error);

impl std::fmt::Display for FileRead {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:#}", self.0)
    }
}

impl std::error::Error for FileRead {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.0.source()
    }
}

/// Wraps an image-side read failure so the driver can tell it apart from
/// a terminal budget or fast-import error.
fn file_read(error: anyhow::Error) -> anyhow::Error {
    FileRead(error).into()
}

/// Records a per-file image failure: the path is left out of the streamed
/// rows, and the base keeps HEAD's version where it has one.
fn note_file_read(
    path: &str,
    error: &anyhow::Error,
    counters: &mut NoteCounters,
    late_carry: &mut Vec<String>,
) {
    counters.note_with_extra(
        "paths are unreadable; the baseline keeps HEAD's version where present",
        error.to_string(),
        names::escape_display(path),
    );
    late_carry.push(path.to_owned());
}

/// Records a per-path file over the per-blob limit: the path is left out of
/// the streamed rows, and the base keeps HEAD's version where it has one.
fn note_oversized(
    path: &str,
    max_blob_bytes: u64,
    counters: &mut NoteCounters,
    late_carry: &mut Vec<String>,
) {
    counters.note(
        &format!(
            "paths are over the {max_blob_bytes}-byte blob limit; the baseline keeps HEAD's \
             version where present"
        ),
        names::escape_display(path),
    );
    late_carry.push(path.to_owned());
}

/// Handles one [`stream_file`] failure. A per-path failure ([`FileRead`],
/// [`BlobTooLarge`]) is carried forward and the run continues; anything
/// else is terminal and returned to the caller.
fn handle_stream_error(
    path: &str,
    error: anyhow::Error,
    max_blob_bytes: u64,
    counters: &mut NoteCounters,
    late_carry: &mut Vec<String>,
) -> Result<()> {
    if error.downcast_ref::<BlobTooLarge>().is_some() {
        note_oversized(path, max_blob_bytes, counters, late_carry);
        return Ok(());
    }
    if error.downcast_ref::<FileRead>().is_some() {
        note_file_read(path, &error, counters, late_carry);
        return Ok(());
    }
    Err(error)
}

/// A file grew past the per-blob limit between the walk and the stream, so
/// nothing was emitted and the driver carries HEAD's version forward.
#[derive(Debug)]
pub struct BlobTooLarge;

impl std::fmt::Display for BlobTooLarge {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("the file is over the per-blob limit")
    }
}

impl std::error::Error for BlobTooLarge {}

/// Streams one regular file into fast-import as a blob and returns its
/// mark. The framing invariant holds throughout: the size comes from one
/// fresh inode read, and exactly that many bytes are emitted.
///
/// Errors are per-path or terminal. [`FileRead`] (the image could not
/// supply the content) and [`BlobTooLarge`] (the fresh size is over
/// `max_blob_bytes`) are caught by the driver, which carries the path
/// forward and keeps the run going. Any other error is terminal: the
/// caller aborts and drops the fast-import process. A failed Normalize
/// preparation releases the budget reservation, since nothing reached
/// fast-import; a failure after the blob header keeps the charge, because
/// the frame may have been partly emitted and the path is not retried.
// Streaming needs the reader, both git sinks, the entry, the route, the
// reference parent, the shared budget, the per-blob limit and the note
// counters; the argument count is the clearest shape.
#[allow(clippy::too_many_arguments)]
pub fn stream_file(
    reader: &ImageReader,
    fast_import: &mut FastImport,
    cat_file: &mut CatFileBatch,
    entry: &PlannedEntry,
    route: ConversionRoute,
    parent: Option<&str>,
    budget: &mut Budget,
    max_blob_bytes: u64,
    counters: &mut NoteCounters,
) -> Result<u32> {
    // The filter route has its own function; reaching the match below
    // would panic after the blob header is already on the wire.
    if matches!(route, ConversionRoute::Filter) {
        bail!("the filter route does not stream a mark");
    }
    let inode = reader.read_inode_fresh(entry.ino).map_err(file_read)?;
    // Validate before writing anything: once `data <n>` is on the wire, a
    // file that is no longer a regular extents file would need the whole
    // frame padded just to keep the stream parseable.
    if inode.file_type() != S_IFREG || inode.i_flags & EXT4_EXTENTS_FL == 0 {
        return Err(file_read(anyhow::anyhow!(
            "{} is no longer a regular file with extents",
            names::escape_display(&entry.repo_path)
        )));
    }
    // Frame from the inode just read, not from the scan: a file the agent
    // changed between the walk and this stream is captured at its current
    // length, so the blob is neither truncated nor zero-padded, and the
    // budget charges the bytes actually emitted.
    let size = inode.file_size();
    // The scan checked its own size; a file that grew since must not slip
    // past the per-blob policy. Nothing has been written, so no charge.
    if size > max_blob_bytes {
        return Err(BlobTooLarge.into());
    }
    budget.reserve(&entry.repo_path, size)?;
    match route {
        ConversionRoute::Raw => {
            let mark = fast_import.begin_blob()?;
            fast_import.begin_data(size)?;
            // The watcher tells an image read failure (carry the path
            // forward) apart from a broken pipe into fast-import (stop
            // the run).
            let mut sink = WatchWrites {
                inner: &mut *fast_import,
                failed: false,
            };
            let mut framed = ExactLen::new(&mut sink, size);
            let read = reader.read_file_from(&inode, &mut framed);
            // Finish even when the read failed: `finish` zero-pads a short
            // read and flushes, so the `data <n>` block stays framed and
            // the next command is never read as blob content.
            let finish = framed.finish();
            let mismatch = framed.mismatch();
            let pipe_broke = sink.failed;
            if let Err(error) = read {
                if pipe_broke {
                    finish?;
                    return Err(error);
                }
                return Err(file_read(error));
            }
            finish?;
            if mismatch {
                counters.note(
                    "files changed size while being read from the image",
                    names::escape_display(&entry.repo_path),
                );
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
                counters.note(
                    "files changed size while being read from the image",
                    names::escape_display(&entry.repo_path),
                );
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
                    counters.note(
                        "files did not normalize to the length the image reported",
                        names::escape_display(&entry.repo_path),
                    );
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
                    counters.note(
                        "files did not normalize to the length the image reported",
                        names::escape_display(&entry.repo_path),
                    );
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
        // The watcher tells an image read failure (per-path, carried
        // forward) apart from a failed spool write — a full temp directory
        // is the host's problem and must abort the run instead of being
        // reported as one more corrupt file.
        let mut sink = WatchWrites {
            inner: &mut spool,
            failed: false,
        };
        let mut framed = ExactLen::new(&mut sink, size);
        let read = reader.read_file_from(inode, &mut framed);
        // Finish even when the read failed: `finish` zero-pads a short
        // read, so the spool holds exactly `size` bytes.
        let finish = framed.finish();
        let mismatch = framed.mismatch();
        let spool_failed = sink.failed;
        if let Err(error) = read {
            return if spool_failed {
                Err(error)
            } else {
                Err(file_read(error))
            };
        }
        finish?;
        mismatch
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

/// Re-emits the leaves at or under `prefix` (raw repo-relative bytes),
/// skipping every path the host probe rejected: a subtree must survive a
/// `deleteall` without restoring names git cannot re-create here. `non_trees`
/// is sorted by raw path, so the subtree is one contiguous run.
fn reemit_subtree(
    fast_import: &mut FastImport,
    non_trees: &[&TreeEntry],
    prefix: &[u8],
    rejected: &HashSet<Vec<u8>>,
) -> Result<()> {
    let mut with_slash = prefix.to_vec();
    with_slash.push(b'/');
    let start = non_trees.partition_point(|entry| entry.raw_path.as_slice() < prefix);
    for entry in &non_trees[start..] {
        let raw = entry.raw_path.as_slice();
        if raw != prefix && !raw.starts_with(&with_slash) {
            break;
        }
        if !under_any(raw, rejected) {
            fast_import.modify_raw(&entry.mode, &entry.sha, raw)?;
        }
    }
    Ok(())
}

/// Whether `path` is one of `prefixes` or lies under one.
fn under_any(path: &[u8], prefixes: &HashSet<Vec<u8>>) -> bool {
    if prefixes.is_empty() {
        return false;
    }
    let mut end = path.len();
    loop {
        if prefixes.contains(&path[..end]) {
            return true;
        }
        match path[..end].iter().rposition(|&byte| byte == b'/') {
            Some(position) => end = position,
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

/// `path` relative to `prefix` (a mapping's repo-relative root), without
/// a leading separator; `None` when `path` is outside.
fn relative_to<'a>(path: &'a str, prefix: &str) -> Option<&'a str> {
    if prefix.is_empty() {
        return Some(path);
    }
    let rest = path.strip_prefix(prefix)?;
    if rest.is_empty() {
        return Some("");
    }
    rest.strip_prefix('/')
}

/// Whether `path` lies under one of the mappings (or is a mapping's own
/// repo-relative root).
fn covered_by(mappings: &[&Mapping], path: &str) -> bool {
    mappings
        .iter()
        .any(|mapping| relative_to(path, &mapping.repo_relative).is_some())
}

/// Whether some component of a raw parent-tree path classifies as
/// [`NameClass::IncompatibleHost`], so no walk on this host can produce
/// the path — the scan's own name would be lossy where it is not valid
/// UTF-8, and the host rules exclude the rest.
fn has_incompatible_component(entry: &TreeEntry) -> bool {
    let components: Vec<&[u8]> = entry.raw_path.split(|&byte| byte == b'/').collect();
    let is_symlink = entry.mode == "120000";
    components.iter().enumerate().any(|(index, component)| {
        // Only the final component can be the symlink the
        // `.gitmodules` rule asks about.
        let last = index == components.len() - 1;
        names::validate_component(component, is_symlink && last).class
            == NameClass::IncompatibleHost
    })
}

/// Classifies the components of a repo-relative git path, the verdict a
/// file mapping's single entry gets: the scan classified the *image
/// target's* basename, which a library caller can name independently.
fn repo_path_verdict(repo_path: &str, is_symlink: bool) -> NameVerdict {
    let mut components = repo_path.split('/').peekable();
    while let Some(component) = components.next() {
        let last = components.peek().is_none();
        let verdict = names::validate_component(component.as_bytes(), is_symlink && last);
        if verdict.class != NameClass::Keep {
            return verdict;
        }
    }
    NameVerdict {
        class: NameClass::Keep,
        reason: None,
    }
}

/// Whether paths are compared case-insensitively on this host.
fn case_folding() -> bool {
    cfg!(windows) || cfg!(target_os = "macos")
}

/// Why one path's filter route failed. Every kind is per-path: the caller
/// carries the path forward and keeps the run going.
#[derive(Debug)]
pub enum FilterRouteFailure {
    /// git or the clean filter failed (a missing `git-lfs`, a crashing
    /// driver); grouped per driver by [`group_filter_failures`].
    Filter(anyhow::Error),
    /// The image could not supply the content (the inode changed type, a
    /// corrupt extent tree): the path is corrupted, and blaming the filter
    /// would point the user at the wrong cause.
    Image(anyhow::Error),
    /// The fresh inode size is over the per-blob limit; nothing was read
    /// or charged.
    TooLarge,
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
    max_blob_bytes: u64,
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
    // cannot stream past the run's budget. The scan checked its own size;
    // a file that grew since must not slip past the per-blob policy.
    let size = inode.file_size();
    if size > max_blob_bytes {
        return Ok(Err(FilterRouteFailure::TooLarge));
    }
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
                "{count} {noun} failed their clean filter ({}); the baseline keeps HEAD's version \
                 where present: {} (first error: {})",
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
    use crate::ext4::format::{BLOCK_SIZE, EXT4_ROOT_INO};
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

    /// The notes a counters set would emit now.
    fn flushed(counters: &mut NoteCounters) -> Vec<String> {
        let mut notes = Vec::new();
        counters.flush(&mut notes);
        notes
    }

    /// Commits every current change, for fixtures that grow a repository
    /// after `repo_fixture`.
    fn commit_all(repo: &Repo, message: &str) {
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
        assert!(succeeds(&["add", "-A"]));
        assert!(succeeds(&["commit", "--quiet", "-m", message]));
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
        let mut counters = NoteCounters::default();
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
            1 << 20,
            &mut counters,
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
        assert!(flushed(&mut counters).is_empty());
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
        let mut counters = NoteCounters::default();
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
            1 << 20,
            &mut counters,
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
        assert!(flushed(&mut counters).is_empty());
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
            1 << 20,
        )
        .unwrap();
        assert!(
            matches!(outcome, Err(FilterRouteFailure::Filter(_))),
            "{outcome:?}"
        );

        // An inode that is no longer a regular file is the image's.
        let outcome = hash_object_route(
            &reader,
            &repo,
            &filter_entry(EXT4_ROOT_INO, 0),
            &mut budget,
            1 << 20,
        )
        .unwrap();
        assert!(
            matches!(outcome, Err(FilterRouteFailure::Image(_))),
            "{outcome:?}"
        );

        // A file over the per-blob limit is refused before anything is
        // read, so the budget stays untouched.
        let outcome =
            hash_object_route(&reader, &repo, &filter_entry(ino, 1), &mut budget, 1).unwrap();
        assert!(
            matches!(outcome, Err(FilterRouteFailure::TooLarge)),
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
            1 << 20,
        )
        .unwrap_err();
        assert!(
            error.downcast_ref::<BudgetExceeded>().is_some(),
            "{error:#}"
        );
    }

    #[test]
    fn a_case_variant_ref_blocks_the_label() {
        let Some((directory, repo)) = repo_fixture(&[("a.txt", "a\n")]) else {
            return;
        };
        // A ref this run did not create, differing only by case: the
        // loose-ref backend folds case on Windows, so writing `out`
        // would overwrite it.
        let update = std::process::Command::new("git")
            .arg("-C")
            .arg(&repo.root)
            .args(["update-ref", "refs/ext4kit/Out/base", "HEAD"])
            .output()
            .expect("git runs");
        assert!(update.status.success(), "{update:?}");

        let (image, _ino) = image_with_file(directory.path(), b"a\n");
        let root = directory.path().to_string_lossy().into_owned();
        let mapping = Mapping {
            host: root.clone(),
            image: "/src".to_owned(),
            kind: MappingKind::Dir,
            repo: Some(root),
            repo_relative: String::new(),
        };
        let options = BaseOptions {
            label: "out".to_owned(),
            include_ignored: false,
            max_blob_bytes: 1024 * 1024,
            max_total_bytes: 1024 * 1024,
            unix_time: 0,
        };
        let error = write_base_refs(&image, std::slice::from_ref(&mapping), &options).unwrap_err();
        assert!(
            format!("{error:#}").contains("label out is in use"),
            "{error:#}"
        );
        // The foreign ref is untouched.
        assert!(repo.rev_parse("refs/ext4kit/Out/base").unwrap().len() >= 40);
    }

    #[test]
    fn auto_normalized_hard_links_do_not_share_blobs() {
        // HEAD tracks a.txt with CRLF content, committed before the
        // `text=auto` attribute existed; the image hard-links a second
        // name onto the same inode. The reference blob suppresses a.txt's
        // conversion but not b.txt's, so the two links must become
        // different blobs — sharing one mark would give both the first
        // link's conversion.
        let Some((directory, repo)) = repo_fixture(&[("a.txt", "a\r\nb\r\n")]) else {
            return;
        };
        std::fs::write(directory.path().join(".gitattributes"), "* text=auto\n").unwrap();
        commit_all(&repo, "attributes");

        let mut builder = crate::ext4::test_support::fresh_builder().unwrap();
        let meta = PosixMetadata::default();
        let opt = builder.mkdir(EXT4_ROOT_INO, "opt", None).unwrap();
        let src = builder.mkdir(opt, "src", None).unwrap();
        let a = builder
            .add_file(src, "a.txt", b"a\r\nb\r\n", &meta)
            .unwrap();
        builder.add_hard_link(src, "b.txt", a).unwrap();
        let image = directory.path().join("links.img");
        builder
            .write_to_file(image.to_str().expect("a UTF-8 temp path"))
            .unwrap();

        let root = directory.path().to_string_lossy().into_owned();
        let mapping = Mapping {
            host: root.clone(),
            image: "/opt/src".to_owned(),
            kind: MappingKind::Dir,
            repo: Some(root),
            repo_relative: String::new(),
        };
        let options = BaseOptions {
            label: "out".to_owned(),
            include_ignored: false,
            max_blob_bytes: 1024 * 1024,
            max_total_bytes: 1024 * 1024,
            unix_time: 0,
        };
        let outcome = write_base_refs(&image, std::slice::from_ref(&mapping), &options).unwrap();
        assert_eq!(outcome.repos.len(), 1);
        let base = &outcome.repos[0].base_commit;
        let mut cat_file = CatFileBatch::new(&repo.root).unwrap();
        // The tracked twin keeps its CRLF (the reference blob carries it);
        // the new name has no reference and is normalized.
        assert_eq!(
            cat_file.blob(&format!("{base}:a.txt")).unwrap().as_deref(),
            Some(b"a\r\nb\r\n".as_slice())
        );
        assert_eq!(
            cat_file.blob(&format!("{base}:b.txt")).unwrap().as_deref(),
            Some(b"a\nb\n".as_slice())
        );
    }

    #[test]
    fn crafted_nul_names_do_not_kill_the_base() {
        // Five NUL names cost five check-ignore respawns before the fix —
        // more than MAX_IGNORE_RESPAWNS — aborting the whole base.
        let Some((directory, repo)) = repo_fixture(&[("keep.txt", "k\n")]) else {
            return;
        };
        let mut builder = crate::ext4::test_support::fresh_builder().unwrap();
        let meta = PosixMetadata::default();
        let opt = builder.mkdir(EXT4_ROOT_INO, "opt", None).unwrap();
        let src = builder.mkdir(opt, "src", None).unwrap();
        builder.add_file(src, "keep.txt", b"k\n", &meta).unwrap();
        for index in 0..5 {
            builder
                .add_file(src, &format!("bad{index}.txt"), b"x\n", &meta)
                .unwrap();
        }
        // Serialize first: commit_metadata would overwrite a direct disk
        // edit, because the builder owns the inode tables in memory.
        builder.commit_metadata();
        let blocks = builder.get_dir_blocks(src).unwrap();
        let block = blocks[0] as usize * BLOCK_SIZE;
        let mut bytes = builder.disk_data.to_vec();
        for index in 0..5 {
            let name = format!("bad{index}.txt");
            let name_at = (block..block + BLOCK_SIZE - name.len())
                .find(|&offset| &bytes[offset..offset + name.len()] == name.as_bytes())
                .unwrap_or_else(|| panic!("the {name} record is in the first directory block"));
            // Overwrite the digit with a NUL: the name stays a record of
            // the same length, but classifies as corrupted.
            bytes[name_at + 3] = 0;
        }
        let image = directory.path().join("nul.img");
        std::fs::write(&image, &bytes).unwrap();

        let root = directory.path().to_string_lossy().into_owned();
        let mapping = Mapping {
            host: root.clone(),
            image: "/opt/src".to_owned(),
            kind: MappingKind::Dir,
            repo: Some(root),
            repo_relative: String::new(),
        };
        let options = BaseOptions {
            label: "out".to_owned(),
            include_ignored: false,
            max_blob_bytes: 1024 * 1024,
            max_total_bytes: 1024 * 1024,
            unix_time: 0,
        };
        let outcome = write_base_refs(&image, std::slice::from_ref(&mapping), &options)
            .expect("five NUL names must not abort the base");
        // The clean name is baselined and every NUL name is reported.
        assert_eq!(outcome.repos.len(), 1);
        let base = &outcome.repos[0].base_commit;
        let mut cat_file = CatFileBatch::new(&repo.root).unwrap();
        assert_eq!(
            cat_file
                .blob(&format!("{base}:keep.txt"))
                .unwrap()
                .as_deref(),
            Some(b"k\n".as_slice())
        );
        assert!(
            outcome
                .notes
                .iter()
                .any(|note| note.contains("corrupted name")),
            "{:?}",
            outcome.notes
        );
    }

    /// A directory mapping from `root` onto `image` at `repo_relative`.
    fn mapping_at(root: &str, image: &str, repo_relative: &str) -> Mapping {
        Mapping {
            host: root.to_owned(),
            image: image.to_owned(),
            kind: MappingKind::Dir,
            repo: Some(root.to_owned()),
            repo_relative: repo_relative.to_owned(),
        }
    }

    fn options_for(label: &str) -> BaseOptions {
        BaseOptions {
            label: label.to_owned(),
            include_ignored: false,
            max_blob_bytes: 1024 * 1024,
            max_total_bytes: 1024 * 1024,
            unix_time: 0,
        }
    }

    #[test]
    fn a_file_mapping_is_judged_by_its_repo_path() {
        // The image target's basename is a control-byte name, which cannot
        // exist on any host; the git path is an ordinary one. The base must
        // take the content and must not delete HEAD's version.
        let Some((directory, repo)) = repo_fixture(&[("lib.rs", "old\n")]) else {
            return;
        };
        let mut builder = crate::ext4::test_support::fresh_builder().unwrap();
        let meta = PosixMetadata::default();
        let opt = builder.mkdir(EXT4_ROOT_INO, "opt", None).unwrap();
        let src = builder.mkdir(opt, "src", None).unwrap();
        builder.add_file(src, "a\nb.txt", b"new\n", &meta).unwrap();
        let image = directory.path().join("file.img");
        builder
            .write_to_file(image.to_str().expect("a UTF-8 temp path"))
            .unwrap();

        let root = directory.path().to_string_lossy().into_owned();
        let mapping = Mapping {
            host: format!("{root}/lib.rs"),
            image: "/opt/src/a\nb.txt".to_owned(),
            kind: MappingKind::File,
            repo: Some(root),
            repo_relative: "lib.rs".to_owned(),
        };
        let options = options_for("out");
        let outcome = write_base_refs(&image, std::slice::from_ref(&mapping), &options).unwrap();
        let base = &outcome.repos[0].base_commit;
        let mut cat_file = CatFileBatch::new(&repo.root).unwrap();
        assert_eq!(
            cat_file.blob(&format!("{base}:lib.rs")).unwrap().as_deref(),
            Some(b"new\n".as_slice())
        );
        assert!(
            !outcome
                .notes
                .iter()
                .any(|note| note.contains("cannot exist on this host")),
            "{:?}",
            outcome.notes
        );
    }

    #[test]
    fn twin_raw_paths_sharing_a_lossy_name_are_both_kept() {
        // Two HEAD paths differing only in an invalid byte render to the
        // same lossy string. Both must be re-emitted: the map is keyed by
        // the lossy path, the listing is not.
        let Some((directory, repo)) = repo_fixture(&[("keep.txt", "k\n")]) else {
            return;
        };
        let parent = repo.head_commit().unwrap().unwrap();
        let first: &[u8] = b"caf\xe9";
        let second: &[u8] = b"caf\xea";
        {
            let mut fast_import = FastImport::spawn(&repo.root).unwrap();
            let blob = fast_import.blob(b"x").unwrap();
            fast_import
                .begin_commit("refs/ext4kit/test/head", Some(&parent), "raw", 1)
                .unwrap();
            fast_import
                .modify_raw("100644", &format!(":{blob}"), first)
                .unwrap();
            fast_import
                .modify_raw("100644", &format!(":{blob}"), second)
                .unwrap();
            fast_import.finish().unwrap();
        }
        let head = repo.rev_parse("refs/ext4kit/test/head").unwrap();
        let update = std::process::Command::new("git")
            .arg("-C")
            .arg(&repo.root)
            .args(["update-ref", "HEAD", &head])
            .output()
            .expect("git runs");
        assert!(update.status.success(), "{update:?}");

        let (image, _ino) = image_with_file(directory.path(), b"a\n");
        let root = directory.path().to_string_lossy().into_owned();
        let mapping = mapping_at(&root, "/src", "");
        let options = options_for("out");
        let outcome = write_base_refs(&image, std::slice::from_ref(&mapping), &options).unwrap();
        let base = &outcome.repos[0].base_commit;
        let raw_paths: Vec<Vec<u8>> = repo
            .ls_tree(base)
            .unwrap()
            .into_iter()
            .map(|entry| entry.raw_path)
            .collect();
        assert!(raw_paths.iter().any(|path| path == first), "{raw_paths:?}");
        assert!(raw_paths.iter().any(|path| path == second), "{raw_paths:?}");
    }

    #[test]
    fn a_deleteall_keeps_a_skipped_mappings_heads_content() {
        // The root mapping's `deleteall` must not take the content of a
        // sibling mapping that was skipped (its image root is missing): a
        // skipped mapping promises "no D, the parent content stays".
        let Some((directory, repo)) = repo_fixture(&[("keep.txt", "k\n")]) else {
            return;
        };
        std::fs::create_dir(directory.path().join("sub")).unwrap();
        std::fs::write(directory.path().join("sub/keep.txt"), "k\n").unwrap();
        commit_all(&repo, "sub");
        let (image, _ino) = image_with_file(directory.path(), b"a\n");
        let root = directory.path().to_string_lossy().into_owned();
        let mappings = [
            mapping_at(&root, "/src", ""),
            mapping_at(&root, "/opt/missing", "sub"),
        ];
        let options = options_for("out");
        let outcome = write_base_refs(&image, &mappings, &options).unwrap();
        assert!(
            outcome
                .notes
                .iter()
                .any(|note| note.contains("mapping skipped")),
            "{:?}",
            outcome.notes
        );
        let base = &outcome.repos[0].base_commit;
        let mut cat_file = CatFileBatch::new(&repo.root).unwrap();
        assert_eq!(
            cat_file
                .blob(&format!("{base}:sub/keep.txt"))
                .unwrap()
                .as_deref(),
            Some(b"k\n".as_slice())
        );
    }

    #[test]
    fn a_carried_forward_directory_keeps_its_tracked_leaves() {
        // A directory the walk cannot list is carried forward. Re-emitting
        // it as one tree would be fine here; the leaves are re-emitted
        // individually so a rejected descendant is never restored, and the
        // tracked content must survive either way.
        use crate::ext4::format::INODE_SIZE;
        let Some((directory, repo)) = repo_fixture(&[("keep.txt", "k\n")]) else {
            return;
        };
        std::fs::create_dir(directory.path().join("dir")).unwrap();
        std::fs::write(directory.path().join("dir/keep.txt"), "leaf\n").unwrap();
        commit_all(&repo, "dir");

        let mut builder = crate::ext4::test_support::fresh_builder().unwrap();
        let meta = PosixMetadata::default();
        let opt = builder.mkdir(EXT4_ROOT_INO, "opt", None).unwrap();
        let src = builder.mkdir(opt, "src", None).unwrap();
        let dir = builder.mkdir(src, "dir", None).unwrap();
        builder.add_file(dir, "a.txt", b"a\n", &meta).unwrap();
        // Serialize first: commit_metadata would overwrite a direct disk
        // edit, because the builder owns the inode tables in memory.
        builder.commit_metadata();
        // Zero the directory's i_blocks so the reader's ownership bound
        // rejects listing it.
        let (group, index) = builder.inode_location(dir);
        let offset = builder.group_descriptors[group].bg_inode_table_lo as usize * BLOCK_SIZE
            + index * INODE_SIZE
            + 28;
        builder.disk_data[offset..offset + 4].copy_from_slice(&0u32.to_le_bytes());
        let image = directory.path().join("carry.img");
        std::fs::write(&image, builder.disk_data.to_vec()).unwrap();

        let root = directory.path().to_string_lossy().into_owned();
        let mapping = mapping_at(&root, "/opt/src", "");
        let options = options_for("out");
        let outcome = write_base_refs(&image, std::slice::from_ref(&mapping), &options).unwrap();
        assert!(
            outcome
                .notes
                .iter()
                .any(|note| note.contains("unreadable") && note.contains("dir")),
            "{:?}",
            outcome.notes
        );
        let base = &outcome.repos[0].base_commit;
        let mut cat_file = CatFileBatch::new(&repo.root).unwrap();
        assert_eq!(
            cat_file
                .blob(&format!("{base}:dir/keep.txt"))
                .unwrap()
                .as_deref(),
            Some(b"leaf\n".as_slice())
        );
    }

    #[test]
    fn nested_attribute_files_are_high_risk() {
        assert!(is_high_risk(".gitattributes"));
        assert!(is_high_risk("src/.gitattributes"));
        assert!(is_high_risk("a/b/.gitmodules"));
        assert!(is_high_risk("sub/.envrc"));
        assert!(is_high_risk("Makefile"));
        assert!(!is_high_risk("src/Makefile"));
        assert!(!is_high_risk("src/notes.txt"));
    }

    #[test]
    fn collapsed_notes_keep_only_the_cap() {
        let mut counters = NoteCounters::default();
        for index in 0..NOTE_CAP + 5 {
            counters.note("names are odd", format!("p{index}"));
        }
        let mut notes = Vec::new();
        counters.flush(&mut notes);
        assert_eq!(notes.len(), 1, "{notes:?}");
        assert!(
            notes[0].starts_with(&format!("{} names are odd: p0,", NOTE_CAP + 5)),
            "{notes:?}"
        );
        assert!(notes[0].ends_with(", ..."), "{notes:?}");
        assert!(
            !notes[0].contains(&format!("p{}", NOTE_CAP + 4)),
            "{notes:?}"
        );
    }

    #[test]
    fn a_failing_advisory_diff_does_not_cost_the_base_ref() {
        let Some((directory, repo)) = repo_fixture(&[("a.txt", "a\n")]) else {
            return;
        };
        // Any `git diff` in this repository now fails, while every command
        // the base actually needs still works.
        let configured = std::process::Command::new("git")
            .arg("-C")
            .arg(&repo.root)
            .args(["config", "diff.algorithm", "definitely-not-an-algorithm"])
            .output()
            .expect("git runs");
        assert!(configured.status.success(), "{configured:?}");

        let (image, _ino) = image_with_file(directory.path(), b"a\n");
        let root = directory.path().to_string_lossy().into_owned();
        let mapping = mapping_at(&root, "/src", "");
        let options = options_for("out");
        let outcome = write_base_refs(&image, std::slice::from_ref(&mapping), &options)
            .expect("the advisory diff must not fail the base");
        assert!(repo.rev_parse("refs/ext4kit/out/base").is_ok());
        assert!(
            outcome
                .notes
                .iter()
                .any(|note| note.contains("could not be listed")),
            "{:?}",
            outcome.notes
        );
    }

    #[test]
    fn a_stream_over_the_blob_limit_is_a_per_path_failure() {
        let Some((directory, repo)) = repo_fixture(&[("keep.txt", "k\n")]) else {
            return;
        };
        let data = b"1234567890";
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
        let mut counters = NoteCounters::default();
        let mut fast_import = FastImport::spawn(&repo.root).unwrap();
        let mut cat_file = CatFileBatch::new(&repo.root).unwrap();
        let error = stream_file(
            &reader,
            &mut fast_import,
            &mut cat_file,
            &entry,
            ConversionRoute::Raw,
            None,
            &mut budget,
            5,
            &mut counters,
        )
        .unwrap_err();
        assert!(error.downcast_ref::<BlobTooLarge>().is_some(), "{error:#}");
        // Nothing was emitted, so nothing was charged.
        assert_eq!(budget.remaining, 1 << 20);
    }
}
