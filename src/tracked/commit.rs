//! Turning a scan of the image into git commits. The same code produces
//! the **base** commit `create` writes (parented on HEAD-at-create) and,
//! from step 4 on, every **head** commit of a snapshot; the only
//! differences are the parent and the target ref.
//!
//! Content never passes through memory unbounded: regular files stream
//! from the image reader through the [`ExactLen`] framing writer straight
//! into fast-import (raw), into a bounded spool (converting), or into
//! `git hash-object` (the filter route). Unreadable, oversized or
//! host-incompatible paths copy their parent-tree entry forward instead
//! of becoming deletions, so a cherry-pick never removes something the
//! walk simply could not see.

use super::{
    Mapping, MappingKind,
    names::{self, NameClass},
    scan::{self, EntryKind, MappingScan},
};
use crate::ext4::reader::ImageReader;
use crate::host::git::{
    AttrValue, CatFileBatch, CheckIgnoreSession, ConversionRoute, EolRule, ExactLen, FastImport,
    HashObject, IgnoreAnswer, ParentTree, PathAttrs, Repo, Spool, TextStat, TreeEntry, check_attr,
    conversion_route, reference_has_crlf,
};
use anyhow::{Result, bail};
use std::{
    collections::{HashMap, HashSet},
    io::Write as _,
    path::Path,
};

/// Paths whose changes get attention before apply: attribute and
/// submodule configuration, CI hooks, build entry points.
const HIGH_RISK_PATHS: &[&str] = &[
    ".gitattributes",
    ".gitmodules",
    ".envrc",
    "Makefile",
    "package.json",
];

/// How many per-path notes of one kind are recorded before the rest is
/// collapsed into a count; display only, the tree is unaffected.
const NOTE_CAP: usize = 20;

fn is_high_risk(path: &str) -> bool {
    HIGH_RISK_PATHS.contains(&path)
        || path.starts_with(".github/workflows/")
        || path.starts_with(".husky/")
        || path.starts_with(".vscode/")
}

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

/// The per-run streaming budget, so sparse or crafted files cannot
/// amplify a run into terabytes of zeros (holes count at full length,
/// because they become real bytes in the blob).
struct Budget {
    remaining: u64,
    largest: Vec<(String, u64)>,
}

impl Budget {
    fn new(total: u64) -> Self {
        Self {
            remaining: total,
            largest: Vec::new(),
        }
    }

    /// Reserves `size` bytes for `path`, or fails naming the largest
    /// contributors.
    fn reserve(&mut self, path: &str, size: u64) -> Result<()> {
        self.largest.push((path.to_owned(), size));
        self.largest
            .sort_by_key(|(_, size)| std::cmp::Reverse(*size));
        self.largest.truncate(5);
        if size > self.remaining {
            let largest = self
                .largest
                .iter()
                .map(|(path, size)| format!("{} ({size} bytes)", names::escape_display(path)))
                .collect::<Vec<_>>()
                .join(", ");
            bail!("the run would stream more than its byte budget: {largest}");
        }
        self.remaining -= size;
        Ok(())
    }
}

/// One entry bound for the commit, after ignore filtering.
struct PlannedEntry {
    repo_path: String,
    kind: EntryKind,
    mode: u16,
    ino: u32,
    size: u64,
    symlink_target: Option<Vec<u8>>,
}

/// One mapping's contribution to the commit.
struct PlannedMapping {
    /// The mapping root resolved and walked; `false` means the mapping is
    /// skipped entirely — no `D`, no `M`, the parent content stays.
    emit_delete: bool,
    included: Vec<PlannedEntry>,
    /// Repo paths whose parent-tree entry is copied forward.
    carry_forward: Vec<String>,
}

/// Per-path notes collapsed into list-level counts, so a hostile or huge
/// tree cannot flood the output; the tree itself is unaffected.
#[derive(Default)]
struct NoteCounters {
    ignored: usize,
    never_in_tree: Vec<String>,
    portability: Vec<String>,
}

impl NoteCounters {
    fn flush(&mut self, notes: &mut Vec<String>) {
        if self.ignored > 0 {
            notes.push(format!("{} ignored paths excluded", self.ignored));
        }
        if !self.never_in_tree.is_empty() {
            notes.push(format!(
                "{} paths never enter the tree (.git aliases): {}",
                self.never_in_tree.len(),
                summarize(&self.never_in_tree)
            ));
        }
        if !self.portability.is_empty() {
            notes.push(format!(
                "{} names are not portable to Windows: {}",
                self.portability.len(),
                summarize(&self.portability)
            ));
        }
        self.ignored = 0;
        self.never_in_tree.clear();
        self.portability.clear();
    }
}

fn summarize(paths: &[String]) -> String {
    let mut shown: Vec<&str> = paths.iter().take(NOTE_CAP).map(String::as_str).collect();
    if paths.len() > NOTE_CAP {
        shown.push("...");
    }
    shown.join(", ")
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
    let reference = format!("refs/ext4kit/{}/base", options.label);
    // Repositories whose base ref this run may have created: the per-repo
    // pre-check below proves the ref was absent before the run started,
    // so anything present in one of these afterwards is ours to remove.
    let mut started: Vec<String> = Vec::new();
    for (root, indexes) in repos {
        let repo = Repo::at(&root);
        if repo.rev_parse_optional(&reference)?.is_some() {
            // A foreign ref this run did not create: remove only the
            // started ones, never this one.
            for started_root in &started {
                let _ = Repo::at(started_root).delete_refs(std::slice::from_ref(&reference));
            }
            bail!(
                "label {} is in use in {}; pass --label to choose another",
                options.label,
                repo.root.display()
            );
        }
        started.push(root.clone());
        let repo_mappings: Vec<&Mapping> = indexes.iter().map(|&i| &mappings[i]).collect();
        match write_repo_base(
            &reader,
            &repo,
            &repo_mappings,
            options,
            &mut budget,
            &mut notes,
        ) {
            Ok(base) => bases.push(RepoBase {
                root,
                base_commit: base,
            }),
            Err(error) => {
                for started_root in &started {
                    let _ = Repo::at(started_root).delete_refs(std::slice::from_ref(&reference));
                }
                return Err(error);
            }
        }
    }
    Ok(BaseOutcome {
        repos: bases,
        notes,
    })
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

    // Gitlinks under the mapped subtrees are re-emitted unchanged: the
    // D / deleteall would otherwise delete them, and the scan never adds
    // the image's copy of a submodule's files to the outer repository.
    let gitlinks: Vec<(String, String)> = parent_tree
        .entries
        .values()
        .filter(|entry| entry.is_gitlink() && covered_by(mappings, &entry.path))
        .map(|entry| (entry.path.clone(), entry.sha.clone()))
        .collect();

    // Parent-tree paths no walk on this host could ever re-add — some
    // component of the name cannot exist here — are re-emitted with their
    // raw bytes: the D / deleteall must not delete them, and a lossy path
    // (U+FFFD) would corrupt the name. Gitlinks with such a name take this
    // route too, instead of the plain gitlink re-emit. Only the top-most
    // such entry is re-emitted; a tree entry carries its descendants.
    let mut incompatible: Vec<&TreeEntry> = parent_tree
        .entries
        .values()
        .filter(|entry| covered_by(mappings, &entry.path) && has_incompatible_component(entry))
        .collect();
    incompatible.sort_by(|a, b| a.raw_path.cmp(&b.raw_path));
    let incompatible_paths: HashSet<String> = incompatible
        .iter()
        .map(|entry| entry.path.clone())
        .collect();
    let unrepresentable: Vec<&TreeEntry> = incompatible
        .iter()
        .filter(|entry| {
            !incompatible.iter().any(|other| {
                other.raw_path.len() < entry.raw_path.len()
                    && entry.raw_path.starts_with(&other.raw_path)
                    && entry.raw_path[other.raw_path.len()] == b'/'
            })
        })
        .copied()
        .collect();
    // Git itself may refuse to re-create some of these paths on this host
    // (Windows rejects control bytes and reserved names a repository
    // cloned from elsewhere can still carry in HEAD); re-emitting one
    // would abort the whole base, so only the accepted subset is
    // re-emitted and the rest is reported, never dropped silently.
    // An index holds no tree entries (`update-index` ignores a 040000
    // record), so a directory is probed through its non-tree
    // descendants: they share the incompatible component, and git
    // accepting any of them means it accepts the directory's name too.
    let descendants_of = |tree: &TreeEntry| -> Vec<&TreeEntry> {
        let mut prefix = tree.raw_path.clone();
        prefix.push(b'/');
        parent_tree
            .entries
            .values()
            .filter(|entry| !entry.is_tree() && entry.raw_path.starts_with(&prefix))
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
            if is_accepted {
                true
            } else {
                notes.push(format!(
                    "path {} cannot be re-created by git on this host; it and its tracked \
                     descendants are deleted from the baseline",
                    names::escape_display(&entry.path)
                ));
                false
            }
        })
        .collect();
    let unrepresentable_paths: HashSet<&str> = unrepresentable
        .iter()
        .map(|entry| entry.path.as_str())
        .collect();

    let mut ignore = CheckIgnoreSession::new(&repo.root)?;
    let mut counters = NoteCounters::default();
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
    counters.flush(notes);

    for (path, _) in &gitlinks {
        // The scan's boundary entry (a submodule checkout holds `.git`)
        // proves the image still has the directory.
        let present = mappings.iter().zip(&scans).any(|(mapping, scan)| {
            relative_to(path, &mapping.repo_relative).is_some_and(|rel| {
                rel.is_empty() || scan.entries.iter().any(|entry| entry.rel_path == rel)
            })
        });
        if !present {
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
                notes.push(format!(
                    "high-risk path added: {}",
                    names::escape_display(path)
                ));
            }
            if case_folding()
                && case_seen
                    .insert(path.to_uppercase(), path.clone())
                    .is_some_and(|other| other != *path)
            {
                notes.push(format!(
                    "case collision: {} cannot be checked out beside its case twin",
                    names::escape_display(path)
                ));
            }
            match entry.kind {
                EntryKind::Symlink => {
                    let target = entry.symlink_target.as_deref().unwrap_or_default();
                    let mark = fast_import.blob(target)?;
                    rows.push(("120000".to_owned(), format!(":{mark}"), path.clone()));
                }
                EntryKind::File => {
                    if entry.size > options.max_blob_bytes {
                        notes.push(format!(
                            "{} is {} bytes, over the {}-byte blob limit; carried forward",
                            names::escape_display(path),
                            entry.size,
                            options.max_blob_bytes
                        ));
                        late_carry.push(path.clone());
                        continue;
                    }
                    let route = attributes.get(path).map_or(ConversionRoute::Raw, |attrs| {
                        conversion_route(attrs, autocrlf)
                    });
                    let source = match route {
                        ConversionRoute::Filter => {
                            // The filter route streams too, so it pays
                            // into the same byte budget.
                            budget.reserve(path, entry.size)?;
                            match hash_object_route(reader, repo, entry) {
                                Ok((sha, mismatch)) => {
                                    if mismatch {
                                        notes.push(format!(
                                            "length mismatch while reading {} from the image",
                                            names::escape_display(path)
                                        ));
                                    }
                                    sha
                                }
                                Err(error) => {
                                    // A clean filter that is missing or
                                    // crashing must not cost the run its
                                    // tracking: the path is corrupted,
                                    // carried forward, and reported with
                                    // its siblings that failed the same
                                    // way.
                                    filter_failures.push(FilterFailure {
                                        driver: filter_driver(attributes.get(path)),
                                        path: path.clone(),
                                        error: format!("{error:#}"),
                                    });
                                    late_carry.push(path.clone());
                                    continue;
                                }
                            }
                        }
                        shared @ (ConversionRoute::Raw
                        | ConversionRoute::Normalize(EolRule::Explicit)) => {
                            match shared_marks.get(&(entry.ino, shared)) {
                                Some(mark) => mark.clone(),
                                None => {
                                    let mark = stream_file(
                                        reader,
                                        &mut fast_import,
                                        &mut cat_file,
                                        entry,
                                        shared,
                                        parent.as_deref(),
                                        budget,
                                        notes,
                                    )?;
                                    let mark = format!(":{mark}");
                                    shared_marks.insert((entry.ino, shared), mark.clone());
                                    mark
                                }
                            }
                        }
                        ConversionRoute::Normalize(EolRule::Auto) => {
                            // Never shared between hard links: the convert
                            // decision reads this path's reference blob.
                            let mark = stream_file(
                                reader,
                                &mut fast_import,
                                &mut cat_file,
                                entry,
                                ConversionRoute::Normalize(EolRule::Auto),
                                parent.as_deref(),
                                budget,
                                notes,
                            )?;
                            format!(":{mark}")
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
    let reference = format!("refs/ext4kit/{}/base", options.label);
    let message = format!("base of {} ({targets})", options.label);
    fast_import.begin_commit(&reference, parent.as_deref(), &message, options.unix_time)?;
    for (mapping, plan) in mappings.iter().zip(&planned) {
        if !plan.emit_delete {
            continue;
        }
        if mapping.repo_relative.is_empty() {
            fast_import.delete_all()?;
        } else {
            fast_import.delete_path(&mapping.repo_relative)?;
        }
    }
    for (path, sha) in &gitlinks {
        if incompatible_paths.contains(path.as_str()) {
            // An incompatible gitlink goes through the raw-byte path
            // below (or was reported as unrepresentable): its lossy
            // name would rename the submodule pointer.
            continue;
        }
        fast_import.modify("160000", sha, path)?;
    }
    for entry in &unrepresentable {
        fast_import.modify_raw(&entry.mode, &entry.sha, &entry.raw_path)?;
    }
    let mut carried: Vec<&String> = planned
        .iter()
        .flat_map(|plan| plan.carry_forward.iter())
        .collect();
    carried.extend(late_carry.iter());
    for path in carried {
        if unrepresentable_paths.contains(path.as_str()) {
            // Re-emitted with its raw bytes above.
            continue;
        }
        let Some(entry) = parent_tree.entries.get(path.as_str()) else {
            continue;
        };
        fast_import.modify(&entry.mode, &entry.sha, path)?;
    }
    for (mode, source, path) in rows {
        if unrepresentable_paths.contains(path.as_str()) {
            // The walk's lossy path names a HEAD entry that was re-emitted
            // with its raw bytes above.
            continue;
        }
        fast_import.modify(&mode, &source, &path)?;
    }
    fast_import.finish()?;

    repo.rev_parse(&reference)
}

/// Classifies one mapping's scan into included entries, carry-forward
/// paths and notes, applying the ignore rules with the parent tree's
/// tracked-prefix guard and exact-path exemption. The context the walk
/// needs is inherently the whole plan's state, so the argument count is
/// the clearest shape.
#[allow(clippy::too_many_arguments)]
fn plan_mapping(
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

    let mut pruned: HashSet<String> = HashSet::new();
    for (index, entry) in scan.entries.iter().enumerate() {
        let repo_path = mapping.repo_path(&entry.rel_path);
        // A pruned or re-emitted-gitlink subtree is not walked further.
        if pruned.iter().any(|dir| under(&repo_path, dir))
            || gitlinks
                .iter()
                .any(|(gitlink, _)| repo_path == *gitlink || under(&repo_path, gitlink))
        {
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
enum EntryPlan {
    /// The subtree is not walked further (pruned, boundary, excluded).
    Skip,
    /// A directory the walk descends into; its children are queried.
    WalkDir,
    /// A leaf entry; nothing to descend.
    Leaf,
}

/// Classifies one entry against the name rules, the parent tree and the
/// ignore answer, recording it as included, carried forward or excluded.
#[allow(clippy::too_many_arguments)]
fn plan_entry(
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
                names::escape_display(&entry.rel_path),
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
                names::escape_display(&entry.rel_path),
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
    // A nested repository is a boundary: either its gitlink is re-emitted
    // (handled by the caller's gitlink list) or it is skipped with its
    // parent content carried forward.
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
    // An exact blob, symlink or gitlink in the parent tree is never
    // treated as ignored, whatever the index says.
    let exempt = parent_tree
        .entries
        .get(repo_path)
        .is_some_and(|entry| !entry.is_tree());
    match entry.kind {
        EntryKind::Directory => {
            if !options.include_ignored
                && ignored
                && !exempt
                && !parent_tree.has_tracked_descendants(repo_path)
            {
                counters.ignored += 1;
                EntryPlan::Skip
            } else {
                EntryPlan::WalkDir
            }
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
fn queryable(entry: &super::scan::ScannedEntry) -> bool {
    matches!(
        entry.verdict.class,
        NameClass::Keep | NameClass::PortabilityWarning
    ) && entry.corrupted.is_none()
}

/// Sends one directory's children to the ignore session as a batch and
/// stores their answers. Only [`queryable`] children are sent.
fn query_children(
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
/// fresh inode read, and exactly that many bytes are emitted.
#[allow(clippy::too_many_arguments)]
fn stream_file(
    reader: &ImageReader,
    fast_import: &mut FastImport,
    cat_file: &mut CatFileBatch,
    entry: &PlannedEntry,
    route: ConversionRoute,
    parent: Option<&str>,
    budget: &mut Budget,
    notes: &mut Vec<String>,
) -> Result<u32> {
    budget.reserve(&entry.repo_path, entry.size)?;
    let inode = reader.read_inode_fresh(entry.ino)?;
    let mark = fast_import.begin_blob()?;
    match route {
        ConversionRoute::Raw => {
            fast_import.begin_data(entry.size)?;
            let mut framed = ExactLen::new(&mut *fast_import, entry.size);
            reader.read_file_from(&inode, &mut framed)?;
            framed.finish()?;
            if framed.mismatch() {
                notes.push(format!(
                    "length mismatch while reading {} from the image",
                    names::escape_display(&entry.repo_path)
                ));
            }
        }
        ConversionRoute::Normalize(rule) => {
            // A converting file is spooled first — `data <n>` needs the
            // exact normalized length up front — in memory below 8 MiB,
            // in a temp file above it.
            let mut spool = Spool::create()?;
            {
                let mut framed = ExactLen::new(&mut spool, entry.size);
                reader.read_file_from(&inode, &mut framed)?;
                framed.finish()?;
                if framed.mismatch() {
                    notes.push(format!(
                        "length mismatch while reading {} from the image",
                        entry.repo_path
                    ));
                }
            }
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
                        // The reference-blob exception, queried lazily
                        // against the fixed parent tree: no reference
                        // blob (an unborn repository, or a new path)
                        // means convert, like git's new-file case.
                        match parent {
                            Some(parent) => {
                                let reference =
                                    cat_file.blob(&format!("{parent}:{}", entry.repo_path))?;
                                !reference.as_deref().is_some_and(reference_has_crlf)
                            }
                            None => true,
                        }
                    }
                }
            };
            // The emitted bytes go through ExactLen as well: whatever a
            // conversion bug does to the length, the framing holds — the
            // framing-integrity invariant must not depend on the
            // normalizer being exact.
            if convert {
                let len = spool.len() - stats.crlf_pairs();
                fast_import.begin_data(len)?;
                let mut framed = ExactLen::new(&mut *fast_import, len);
                spool
                    .stream_crlf_to_git(|chunk| framed.write_all(chunk))
                    .map_err(|error| anyhow::anyhow!("cannot stream the spool: {error}"))?;
                framed.finish()?;
                if framed.mismatch() {
                    notes.push(format!(
                        "length mismatch while normalizing {}",
                        names::escape_display(&entry.repo_path)
                    ));
                }
            } else {
                fast_import.begin_data(spool.len())?;
                let mut framed = ExactLen::new(&mut *fast_import, spool.len());
                spool
                    .for_each_chunk(|chunk| framed.write_all(chunk))
                    .map_err(|error| anyhow::anyhow!("cannot stream the spool: {error}"))?;
                framed.finish()?;
                if framed.mismatch() {
                    notes.push(format!(
                        "length mismatch while normalizing {}",
                        names::escape_display(&entry.repo_path)
                    ));
                }
            }
        }
        ConversionRoute::Filter => unreachable!("the filter route does not stream a mark"),
    }
    Ok(mark)
}

/// The parent directory of a mapping-relative path; `""` for depth one.
fn parent_dir(rel_path: &str) -> &str {
    rel_path.rsplit_once('/').map_or("", |(dir, _)| dir)
}

/// Streams one file through `git hash-object -w --stdin --path`, the
/// filter route, and returns the object name plus whether the content
/// matched the framed length. Any failure — a clean filter that is not
/// installed or crashes, an unreadable inode — is an error for this path
/// only; the caller marks it corrupted and carries it forward.
fn hash_object_route(
    reader: &ImageReader,
    repo: &Repo,
    entry: &PlannedEntry,
) -> Result<(String, bool)> {
    let inode = reader.read_inode_fresh(entry.ino)?;
    let mut hash = HashObject::spawn(&repo.root, &entry.repo_path, entry.size)?;
    reader.read_file_from(&inode, hash.writer())?;
    hash.finish()
}

/// One path whose filter route failed.
struct FilterFailure {
    /// What selected the filter route, e.g. `filter=lfs` or `ident`.
    driver: String,
    path: String,
    error: String,
}

/// Names what sent a path down the filter route, the grouping key for
/// failures: the error text itself always names the path (both ours and
/// git's), so grouping by it would never merge two failures.
fn filter_driver(attrs: Option<&PathAttrs>) -> String {
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
fn group_filter_failures(failures: &[FilterFailure]) -> Vec<String> {
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

/// Whether `path` is `dir` or lies under it.
fn under(path: &str, dir: &str) -> bool {
    path == dir || path.starts_with(&format!("{dir}/"))
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

/// Whether paths are compared case-insensitively on this host.
fn case_folding() -> bool {
    cfg!(windows) || cfg!(target_os = "macos")
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
    use crate::ext4::format::{BLOCK_SIZE, EXT4_ROOT_INO};

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
        let meta = crate::ext4::PosixMetadata::default();
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
        let meta = crate::ext4::PosixMetadata::default();
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
}
