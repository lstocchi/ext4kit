//! The create-time orchestration of folder tracking: the checks that run
//! before the output is truncated (old-image gates, overlap, repository
//! discovery, the dirty check, the label), the retirement of an old
//! image's tracking, and the base refs and sidecar written once the
//! image exists. The CLI parses flags and calls in; notes and warnings
//! keep going to stdout/stderr from here, exactly as before the move.

use super::{
    Mapping, MappingKind,
    commit::{self, BaseOptions},
    normalize_host_path,
    sidecar::{self, RepoRecord, Sidecar},
};
use crate::ext4::reader::ImageReader;
use crate::host::git::{self, Repo};
use anyhow::{Result, bail};
use std::{
    fs,
    path::{Path, PathBuf},
    time::{Instant, UNIX_EPOCH},
};

/// One `--add` source whose imported content may be tracked, already
/// classified by the caller: the host source, the image-side target it
/// lands at, and whether it is a directory. A library type, so the
/// library never sees the CLI's own addition parsing.
pub struct TrackedSource {
    pub source: PathBuf,
    pub target: String,
    pub is_dir: bool,
}

/// The create-time tracking switches, bundled to keep the create flow
/// callable.
#[derive(Debug, Default)]
pub struct TrackingFlags {
    pub label: Option<String>,
    pub include_ignored: bool,
    pub replace_mounted: bool,
    pub discard_unsnapshotted: bool,
    pub verbose: bool,
}

/// The tracking state `create` carries between its pure checks and the
/// base refs written after the image exists.
pub struct TrackingPrepared {
    label: String,
    include_ignored: bool,
    /// Every `--add` mapping; untracked ones carry `repo: None`.
    mappings: Vec<Mapping>,
    /// The repositories to write a base ref into, in mapping order.
    repos: Vec<PendingRepo>,
}

struct PendingRepo {
    root: String,
    dirty_at_create: bool,
}

/// The output's existing sidecar, with whether it actually describes
/// this output's image: a foreign sidecar owns no refs here.
pub struct OldTracking {
    sidecar: Sidecar,
    foreign: bool,
}

/// The create-time tracking checks that run before the output is
/// truncated: the two-sided overlap check, repo discovery, the dirty
/// check with the artifact advisory, and the label. A returned `None`
/// means this image is not tracked (no `--add`, no git, overlapping
/// targets, or every mapping outside a repository). The old sidecar is
/// handed back so the caller can retire it after the network work —
/// a registry pull failure must leave the old image exactly as it was,
/// still tracked.
pub fn prepare_tracking(
    sources: &[TrackedSource],
    output: &Path,
    flags: &TrackingFlags,
) -> Result<(Option<TrackingPrepared>, Option<OldTracking>)> {
    let sidecar_path = Sidecar::path_for(output);
    let old_sidecar = if sidecar_path.exists() {
        Some(Sidecar::load(output)?)
    } else {
        None
    };
    // The old image's refusals gate every path from here on, including
    // the ones that cannot track: without this, a missing git would let a
    // live tracked image be truncated with no mounted check at all.
    let ownership = check_old_image_state(output, old_sidecar.as_ref(), flags)?;
    let old_tracking = old_sidecar.map(|sidecar| OldTracking {
        sidecar,
        foreign: matches!(ownership, SidecarOwnership::Foreign { .. }),
    });
    if let SidecarOwnership::Foreign { found, expected } = ownership {
        eprintln!(
            "warning: the sidecar of {} records image {expected} but the image is \
             {found}; its refs are left alone",
            output.display()
        );
    }

    if sources.is_empty() {
        // No --add content: nothing to track, but an existing sidecar
        // still has to be retired with its refs.
        return Ok((None, old_tracking));
    }
    if !git::git_available() {
        eprintln!("warning: git is not on PATH; folder tracking is disabled for this image");
        return Ok((None, old_tracking));
    }

    // Step 1: overlap — image side and host side. Overlaps disable
    // tracking but do not fail the create.
    let image_overlap = sources.iter().any(|a| {
        sources
            .iter()
            .any(|b| a.target != b.target && under(&a.target, &b.target))
    });
    if image_overlap {
        eprintln!("warning: overlapping --add targets: folder tracking disabled for this image");
        return Ok((None, old_tracking));
    }

    // Step 2: repository discovery per mapping.
    let mut mappings: Vec<Mapping> = Vec::new();
    let mut untracked = Vec::new();
    for source in sources {
        let repo = match git::discover_repo(&source.source)? {
            Some(repo) => Some(repo),
            None => {
                untracked.push(source.source.display().to_string());
                None
            }
        };
        let (repo, repo_relative) = match &repo {
            Some(repo) => (
                Some(normalize_host_path(&repo.root)),
                repo_relative_path(&source.source, &repo.root),
            ),
            None => (None, String::new()),
        };
        mappings.push(Mapping {
            host: normalize_host_path(&source.source),
            image: source.target.clone(),
            kind: if source.is_dir {
                MappingKind::Dir
            } else {
                MappingKind::File
            },
            repo,
            repo_relative,
        });
    }
    for source in untracked {
        eprintln!("note: {source} is not inside a git repository; this mapping is not tracked");
    }
    let host_overlap = mappings.iter().enumerate().any(|(a, first)| {
        mappings.iter().enumerate().any(|(b, second)| {
            a < b
                && first.repo.is_some()
                && first.repo == second.repo
                && (first.repo_relative == second.repo_relative
                    || under(&second.repo_relative, &first.repo_relative)
                    || under(&first.repo_relative, &second.repo_relative))
        })
    });
    if host_overlap {
        eprintln!(
            "warning: overlapping --add host paths in one repository: \
             folder tracking disabled for this image"
        );
        return Ok((None, old_tracking));
    }
    if !mappings.iter().any(|mapping| mapping.repo.is_some()) {
        // Every mapping is outside a repository: nothing to track, and
        // the notes above already said so.
        return Ok((None, old_tracking));
    }

    // Step 3: the dirty check (never writing the index) and the
    // artifact-directory advisory, per repository.
    let mut repos: Vec<PendingRepo> = Vec::new();
    for mapping in &mappings {
        let Some(root) = &mapping.repo else {
            continue;
        };
        if repos.iter().any(|repo| &repo.root == root) {
            continue;
        }
        let repo = Repo::at(root);
        let subtrees: Vec<String> = mappings
            .iter()
            .filter(|m| m.repo.as_deref() == Some(root.as_str()))
            .map(|m| m.repo_relative.clone())
            .filter(|path| !path.is_empty())
            .collect();
        let dirty = !repo.dirty_paths(&subtrees)?.is_empty();
        if dirty {
            eprintln!(
                "warning: uncommitted changes in {root} are part of the baseline. \
                 Commit or stash them before applying a snapshot: cherry-pick \
                 refuses to overwrite local changes, and agent edits to files that \
                 are untracked now need `git apply`. Commit first for the smoothest \
                 review."
            );
        }
        advise_artifact_directories(&repo, &mappings, root);
        repos.push(PendingRepo {
            root: root.clone(),
            dirty_at_create: dirty,
        });
    }

    // Step 4: the label. An explicit --label is validated — the user
    // chose it — while the default is derived by transformation, so an
    // image named "My Repo.img" or "con.img" never fails a create over
    // its own file name. Refs the old sidecar owns are moved away by the
    // retirement below, so they do not collide — but only in the
    // repositories that sidecar actually tracked, and never when the
    // sidecar is foreign: in any other case a same-named ref belongs to a
    // different image and must collide.
    let label = match flags.label.as_deref() {
        Some(label) => sidecar::sanitize_label(label)?,
        None => sidecar::derived_label(&default_label(output)),
    };
    for repo in &repos {
        let handle = Repo::at(&repo.root);
        // A foreign sidecar owns no refs, so it exempts nothing: a
        // same-named ref then belongs to a different image and collides.
        let owned_by_old_sidecar = old_tracking
            .as_ref()
            .filter(|old| !old.foreign)
            .is_some_and(|old| {
                old.sidecar.label == label
                    && old
                        .sidecar
                        .repos
                        .iter()
                        .any(|record| record.root == repo.root)
            });
        if handle.label_in_use(&label)? && !owned_by_old_sidecar {
            bail!(
                "label {label} is in use in {}; pass --label to choose another",
                handle.root.display()
            );
        }
    }

    Ok((
        Some(TrackingPrepared {
            label,
            include_ignored: flags.include_ignored,
            mappings,
            repos,
        }),
        old_tracking,
    ))
}

/// The label an image gets when `--label` is absent: its file name
/// without the extension, so `out.img` tracks as `out`.
fn default_label(output: &Path) -> String {
    output.file_stem().map_or_else(
        || output.to_string_lossy().into_owned(),
        |stem| stem.to_string_lossy().into_owned(),
    )
}

/// Whether `path` is `dir` or lies under it, for slash-separated paths.
fn under(path: &str, dir: &str) -> bool {
    dir.is_empty() || path == dir || path.starts_with(&format!("{dir}/"))
}

/// The host source's path relative to its repository root, normalized;
/// empty when the source is the root. `None` (which disables tracking
/// for the mapping) cannot happen after discovery, but a canonicalize
/// failure can.
fn repo_relative_path(source: &Path, repo_root: &Path) -> String {
    let source = fs::canonicalize(source)
        .map(|path| normalize_host_path(&path))
        .unwrap_or_else(|_| normalize_host_path(source));
    let root = normalize_host_path(repo_root);
    source
        .strip_prefix(&root)
        .unwrap_or(&source)
        .trim_start_matches('/')
        .to_owned()
}

/// Well-known artifact directories the advisory looks for: display only,
/// because the repository's declared policy is the only policy.
const ARTIFACT_DIRECTORIES: [&str; 6] = [
    "node_modules",
    "target",
    "dist",
    "build",
    "__pycache__",
    ".venv",
];

fn advise_artifact_directories(repo: &Repo, mappings: &[Mapping], root: &str) {
    for mapping in mappings.iter().filter(|m| m.repo.as_deref() == Some(root)) {
        if mapping.kind != MappingKind::Dir {
            continue;
        }
        for name in ARTIFACT_DIRECTORIES {
            let host_dir = Path::new(&mapping.host).join(name);
            if !host_dir.is_dir() {
                continue;
            }
            let repo_relative = if mapping.repo_relative.is_empty() {
                name.to_owned()
            } else {
                format!("{}/{}", mapping.repo_relative, name)
            };
            let ignored = repo.is_ignored(&repo_relative).unwrap_or(false);
            let tracked = repo.tracked_in_head(&repo_relative).unwrap_or(false);
            if !ignored && !tracked {
                eprintln!(
                    "note: {name}/ is neither ignored nor tracked; it will appear \
                     in every snapshot diff — consider adding it to .gitignore"
                );
            }
        }
    }
}

/// Whether the output's existing sidecar describes this output's image.
#[derive(Debug)]
enum SidecarOwnership {
    /// No sidecar, no image to compare against, or a matching UUID: the
    /// refs it lists belong to this image.
    Owned,
    /// The sidecar records a different image's UUID — a copied sidecar or
    /// a wrapper mix-up. Its refs are not this image's and are never
    /// moved or discarded.
    Foreign { found: String, expected: String },
}

/// The refusals that guard the output's existing tracked state: the
/// mounted check, the UUID ownership check and the unsnapshotted-work
/// refusal. These run before every path through `prepare_tracking` —
/// including the ones that cannot track (no `--add`, no git) — because
/// truncating a mounted image or discarding unsnapshotted work must
/// never happen silently. Needs no git: only the image and the sidecar
/// are read.
fn check_old_image_state(
    output: &Path,
    old_sidecar: Option<&Sidecar>,
    flags: &TrackingFlags,
) -> Result<SidecarOwnership> {
    let Some(sidecar) = old_sidecar else {
        return Ok(SidecarOwnership::Owned);
    };
    // The state check runs before anything is published: taking a final
    // snapshot of a live filesystem first would be reading torn state.
    if output.exists() {
        let (reader, state) = ImageReader::open(output)?;
        if state.mounted && !flags.replace_mounted {
            bail!(
                "the old image {} is mounted or was not cleanly unmounted; stop the \
                 VM before re-creating (if the VM is known to be dead, run e2fsck \
                 or pass --replace-mounted)",
                output.display()
            );
        }
        // A sidecar whose UUID does not match the image describes another
        // image's tracking: refusing to re-create over it would block on
        // work this image never held, while retiring its refs would
        // destroy another image's history. It is reported and ignored.
        let found = sidecar::uuid_hex(&reader.fs_uuid());
        if found != sidecar.image_uuid {
            return Ok(SidecarOwnership::Foreign {
                found,
                expected: sidecar.image_uuid.clone(),
            });
        }
    }
    // The final snapshot of the old image lands with the `snapshot`
    // command, so until then any unsnapshotted work can only be
    // discarded, never silently.
    if !flags.discard_unsnapshotted {
        bail!(
            "{} has tracked folders; re-creating it would discard any changes \
             that were never snapshotted. Pass --discard-unsnapshotted to discard \
             them, or run `ext4kit snapshot` first once it exists",
            output.display()
        );
    }
    Ok(SidecarOwnership::Owned)
}

/// Retires the output's existing tracked state before the image is
/// truncated: the old refs move to `previous/<unix-ts>/` (when git is
/// available; without it they are left in place with a warning, since
/// nothing else can move them) and the old sidecar is removed. A foreign
/// sidecar describes another image's tracking, so only its file goes —
/// its refs are never moved.
pub fn retire_old_tracking(output: &Path, old: Option<&OldTracking>) -> Result<()> {
    let Some(old) = old else {
        return Ok(());
    };
    if !old.foreign {
        let sidecar = &old.sidecar;
        eprintln!(
            "note: unsnapshotted changes in {} are being discarded",
            output.display()
        );
        if git::git_available() {
            let timestamp = std::time::SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|duration| duration.as_secs())
                .unwrap_or(0);
            for repo in &sidecar.repos {
                let repo = Repo::at(&repo.root);
                let moved = repo.move_label_refs_to_previous(&sidecar.label, timestamp)?;
                for (reference, sha) in moved {
                    println!(
                        "previous snapshots of {} kept at {reference} ({sha})",
                        sidecar.label
                    );
                }
            }
        } else {
            eprintln!(
                "warning: git is not on PATH; the previous tracking refs of {} were \
                 left in place under refs/ext4kit/{}/",
                output.display(),
                sidecar.label
            );
        }
    }
    fs::remove_file(Sidecar::path_for(output)).map_err(|error| {
        anyhow::anyhow!(
            "cannot remove the old sidecar {}: {error}",
            Sidecar::path_for(output).display()
        )
    })?;
    Ok(())
}

/// Steps 6 and 7: the base refs (one commit per repository, parented on
/// HEAD) and the sidecar. A failure here leaves the image valid, so it
/// is reported, partial refs are removed, and the create still succeeds.
pub fn finish_tracking(
    output: &Path,
    uuid: [u8; 16],
    prepared: &TrackingPrepared,
    image_size_mib: usize,
    flags: &TrackingFlags,
) {
    let started = Instant::now();
    let options = BaseOptions {
        label: prepared.label.clone(),
        include_ignored: prepared.include_ignored,
        max_blob_bytes: 100 * 1024 * 1024,
        max_total_bytes: image_size_mib as u64 * 1024 * 1024,
        unix_time: std::time::SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|duration| duration.as_secs())
            .unwrap_or(0),
    };
    let result = commit::write_base_refs(output, &prepared.mappings, &options);
    let outcome = match result {
        Ok(outcome) => outcome,
        Err(error) => {
            // write_base_refs removes the refs it created before failing,
            // so there is nothing left to clean up here.
            eprintln!(
                "warning: folder tracking is unavailable for {}: {error:#}",
                output.display()
            );
            return;
        }
    };
    if flags.verbose {
        eprintln!(
            "tracking baseline written in {:.1}s",
            started.elapsed().as_secs_f32()
        );
    }
    for note in &outcome.notes {
        eprintln!("note: {note}");
    }
    let mut records: Vec<RepoRecord> = Vec::new();
    let mut by_root: Vec<(String, String)> = Vec::new();
    for base in &outcome.repos {
        println!(
            "tracking baseline: refs/ext4kit/{}/base -> {} in {}",
            prepared.label, base.base_commit, base.root
        );
        by_root.push((base.root.clone(), base.base_commit.clone()));
    }
    for repo in &prepared.repos {
        let base_commit = by_root
            .iter()
            .find(|(root, _)| root == &repo.root)
            .map(|(_, sha)| sha.clone())
            .unwrap_or_default();
        records.push(RepoRecord {
            root: repo.root.clone(),
            base_commit,
            dirty_at_create: repo.dirty_at_create,
        });
    }
    let sidecar = Sidecar {
        tool: format!("ext4kit {}", env!("CARGO_PKG_VERSION")),
        created_unix: options.unix_time,
        image_uuid: sidecar::uuid_hex(&uuid),
        label: prepared.label.clone(),
        include_ignored: prepared.include_ignored,
        mappings: prepared.mappings.clone(),
        repos: records,
    };
    if let Err(error) = sidecar.write(output) {
        eprintln!(
            "warning: folder tracking is unavailable for {}: {error:#}",
            output.display()
        );
        // Remove the refs this run created, by SHA: a foreign ref that
        // happens to carry the same label must survive our failure. Each
        // repository gets only its own pair: update-ref transactions are
        // atomic, and one that names the same ref twice, or another
        // repository's SHA, fails as a whole and deletes nothing.
        let reference = format!("refs/ext4kit/{}/base", prepared.label);
        for base in &outcome.repos {
            let owned = [(reference.clone(), base.base_commit.clone())];
            if let Err(error) = Repo::at(&base.root).delete_refs_exact(&owned) {
                eprintln!(
                    "warning: cannot remove {reference} in {}: {error:#}",
                    base.root
                );
            }
        }
    }
}
