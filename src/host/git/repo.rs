use super::{Autocrlf, git_command, run, run_env};
use anyhow::{Result, anyhow, bail};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::atomic::{AtomicU64, Ordering};

/// One git repository, addressed by its worktree root.
#[derive(Clone, Debug)]
pub struct Repo {
    pub root: PathBuf,
}

impl Repo {
    pub fn at(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// Reads one configuration value; `None` when unset.
    pub fn config(&self, name: &str) -> Result<Option<String>> {
        let output = run(&self.root, &["config", name])?;
        if !output.status.success() {
            return Ok(None);
        }
        let value = String::from_utf8_lossy(&output.stdout).trim().to_owned();
        Ok((!value.is_empty()).then_some(value))
    }

    /// `core.autocrlf`, as far as the eol rules care.
    pub fn autocrlf(&self) -> Autocrlf {
        match self.config("core.autocrlf").ok().flatten().as_deref() {
            Some("true") | Some("1") | Some("yes") => Autocrlf::True,
            Some("input") => Autocrlf::Input,
            _ => Autocrlf::False,
        }
    }

    /// The commit `HEAD` points at, or `None` in an unborn repository.
    /// Detected through `--verify --quiet`'s exit code, not a message:
    /// exit 1 is an unborn HEAD, anything else is a real failure. Git for
    /// Windows ships translations, so stderr text is not a stable
    /// contract even with `LC_ALL=C` forced in [`git_command`].
    pub fn head_commit(&self) -> Result<Option<String>> {
        let output = run(&self.root, &["rev-parse", "--verify", "--quiet", "HEAD"])?;
        if output.status.success() {
            return Ok(Some(
                String::from_utf8_lossy(&output.stdout).trim().to_owned(),
            ));
        }
        if output.status.code() == Some(1) {
            return Ok(None);
        }
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!("cannot resolve HEAD in {}: {stderr}", self.root.display())
    }

    /// Resolves `rev`, failing when it does not exist.
    pub fn rev_parse(&self, rev: &str) -> Result<String> {
        let output = ensure_success(
            run(&self.root, &["rev-parse", "--verify", rev])?,
            &format!("cannot resolve {rev}"),
            &self.root,
        )?;
        Ok(String::from_utf8_lossy(&output.stdout).trim().to_owned())
    }

    /// Resolves `rev`, or `None` when it does not exist.
    pub fn rev_parse_optional(&self, rev: &str) -> Result<Option<String>> {
        let output = run(&self.root, &["rev-parse", "--verify", "--quiet", rev])?;
        if !output.status.success() {
            return Ok(None);
        }
        Ok(Some(
            String::from_utf8_lossy(&output.stdout).trim().to_owned(),
        ))
    }

    /// The `ls-tree -r -t` listing of `treeish`: every blob, symlink,
    /// gitlink and tree entry, the input for the ignore pruning guard and
    /// the carry-forward rule. `None` for a treeish that does not exist
    /// (an unborn repository's HEAD).
    pub fn ls_tree(&self, treeish: &str) -> Result<Vec<TreeEntry>> {
        let output = ensure_success(
            run(&self.root, &["ls-tree", "-r", "-t", "-z", treeish])?,
            &format!("cannot list the tree of {treeish}"),
            &self.root,
        )?;
        let mut entries = Vec::new();
        for record in output.stdout.split(|&byte| byte == 0) {
            if record.is_empty() {
                continue;
            }
            // The path is raw bytes: a HEAD tree can hold names no host
            // filesystem can represent, and those must survive a round
            // trip for the carry-forward re-emission.
            let Some(tab) = record.iter().position(|&byte| byte == b'\t') else {
                continue;
            };
            let header = String::from_utf8_lossy(&record[..tab]);
            let raw_path = record[tab + 1..].to_vec();
            let mut fields = header.split_whitespace();
            let (Some(mode), Some(kind), Some(sha)) = (fields.next(), fields.next(), fields.next())
            else {
                continue;
            };
            entries.push(TreeEntry {
                mode: mode.to_owned(),
                kind: kind.to_owned(),
                sha: sha.to_owned(),
                path: String::from_utf8_lossy(&raw_path).into_owned(),
                raw_path,
            });
        }
        Ok(entries)
    }

    /// The paths `git status --porcelain -z` reports as changed under
    /// `subtrees` (repo-relative, no pathspec when empty). The check never
    /// writes the index or takes `index.lock` (`GIT_OPTIONAL_LOCKS=0`).
    /// The subtrees sit behind `--` with `GIT_LITERAL_PATHSPECS=1`, so a
    /// repo-relative path like `-dist` or `:(top)x` is a path, never an
    /// option or pathspec magic (status honors the variable; only
    /// check-ignore rejects it).
    pub fn dirty_paths(&self, subtrees: &[String]) -> Result<Vec<String>> {
        let mut args = vec![
            "status".to_owned(),
            "--porcelain".to_owned(),
            "-z".to_owned(),
            "--".to_owned(),
        ];
        args.extend(subtrees.iter().cloned());
        let args: Vec<&str> = args.iter().map(String::as_str).collect();
        let output = ensure_success(
            run_env(&self.root, &args, &[("GIT_LITERAL_PATHSPECS", "1")])?,
            "git status failed",
            &self.root,
        )?;
        // With -z, records are NUL-terminated; a rename or copy status is
        // followed by an extra NUL-terminated record holding the original
        // path, which is not another entry.
        let mut paths = Vec::new();
        let mut records = output.stdout.split(|&byte| byte == 0);
        while let Some(record) = records.next() {
            if record.is_empty() {
                continue;
            }
            let status = String::from_utf8_lossy(&record[..2.min(record.len())]).into_owned();
            paths.push(String::from_utf8_lossy(&record[3.min(record.len())..]).into_owned());
            if status.starts_with('R') || status.starts_with('C') {
                records.next();
            }
        }
        Ok(paths)
    }

    /// Whether any `refs/ext4kit/<label>/{base,head}` exists, compared
    /// case-insensitively: the loose-ref backend inherits NTFS and APFS
    /// case insensitivity, so `Out` and `out` collide even though labels
    /// are lower-cased when written.
    pub fn label_in_use(&self, label: &str) -> Result<bool> {
        let output = ensure_success(
            run(
                &self.root,
                &["for-each-ref", "--format=%(refname)", "refs/ext4kit/"],
            )?,
            "cannot list refs/ext4kit",
            &self.root,
        )?;
        let wanted_base = format!("refs/ext4kit/{label}/base");
        let wanted_head = format!("refs/ext4kit/{label}/head");
        Ok(String::from_utf8_lossy(&output.stdout)
            .lines()
            .map(str::trim)
            .any(|name| {
                name.eq_ignore_ascii_case(&wanted_base) || name.eq_ignore_ascii_case(&wanted_head)
            }))
    }

    /// Moves the label's `base` and `head` refs to
    /// `refs/ext4kit/<label>/previous/<timestamp>/` in one transaction and
    /// returns the moved `(ref, sha)` pairs. These refs have no reflog
    /// (nothing outside heads, remotes and notes does), so the move is the
    /// only recovery path for unapplied agent work — two re-creates within
    /// one second therefore land in distinct generations: the first free
    /// timestamp at or after the given one is used, and the refs are
    /// `create`d so a race that slips through fails loudly instead of
    /// overwriting a generation.
    pub fn move_label_refs_to_previous(
        &self,
        label: &str,
        timestamp: u64,
    ) -> Result<Vec<(String, String)>> {
        let mut timestamp = timestamp;
        while ["base", "head"].iter().any(|name| {
            self.rev_parse_optional(&format!("refs/ext4kit/{label}/previous/{timestamp}/{name}"))
                .is_ok_and(|sha| sha.is_some())
        }) {
            timestamp += 1;
        }
        let mut moved = Vec::new();
        let mut commands = String::new();
        for name in ["base", "head"] {
            let from = format!("refs/ext4kit/{label}/{name}");
            let Some(sha) = self.rev_parse_optional(&from)? else {
                continue;
            };
            let to = format!("refs/ext4kit/{label}/previous/{timestamp}/{name}");
            commands.push_str(&format!("create {to} {sha}\n"));
            commands.push_str(&format!("delete {from}\n"));
            moved.push((to, sha));
        }
        if commands.is_empty() {
            return Ok(moved);
        }
        self.update_ref_stdin(&commands)?;
        Ok(moved)
    }

    /// Applies one `git update-ref --stdin` transaction.
    pub fn update_ref_stdin(&self, commands: &str) -> Result<()> {
        let mut child = git_command(&self.root, &["update-ref", "--stdin"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|error| anyhow!("cannot run git update-ref: {error}"))?;
        {
            let mut stdin = child.stdin.take().expect("stdin was piped");
            stdin.write_all(commands.as_bytes())?;
            stdin.flush()?;
        }
        let output = child.wait_with_output()?;
        ensure_success(output, "git update-ref failed", &self.root)?;
        Ok(())
    }

    /// Deletes refs that exist; missing ones are not an error.
    pub fn delete_refs(&self, refs: &[String]) -> Result<()> {
        let existing: Vec<&String> = refs
            .iter()
            .filter(|reference| {
                self.rev_parse_optional(reference)
                    .is_ok_and(|sha| sha.is_some())
            })
            .collect();
        if existing.is_empty() {
            return Ok(());
        }
        let commands: String = existing
            .iter()
            .map(|reference| format!("delete {reference}\n"))
            .collect();
        self.update_ref_stdin(&commands)
    }

    /// Deletes refs **by SHA**: `delete <ref> <sha>` fails when the ref
    /// moved, so a run cleaning up after itself can never remove a ref it
    /// did not create.
    pub fn delete_refs_exact(&self, refs: &[(String, String)]) -> Result<()> {
        if refs.is_empty() {
            return Ok(());
        }
        let commands: String = refs
            .iter()
            .map(|(reference, sha)| format!("delete {reference} {sha}\n"))
            .collect();
        self.update_ref_stdin(&commands)
    }

    /// One one-shot `check-ignore` for the create-time artifact advisory.
    pub fn is_ignored(&self, path: &str) -> Result<bool> {
        let output = run(&self.root, &["check-ignore", "-q", "--", path])?;
        match output.status.code() {
            Some(0) => Ok(true),
            Some(1) => Ok(false),
            _ => bail!(
                "git check-ignore failed for {path} in {}: {}",
                self.root.display(),
                String::from_utf8_lossy(&output.stderr)
            ),
        }
    }

    /// Whether `path` has an entry in HEAD's tree (an unborn repository
    /// has none at all).
    pub fn tracked_in_head(&self, path: &str) -> Result<bool> {
        let output = run(&self.root, &["ls-tree", "--name-only", "HEAD", "--", path])?;
        Ok(output.status.success() && !String::from_utf8_lossy(&output.stdout).trim().is_empty())
    }

    /// Paths that differ between two commits, as lossy UTF-8 (`git diff
    /// --name-only -z`). Advisory only: a path that is not valid UTF-8 is
    /// lossily rendered and never matches the ASCII high-risk names.
    pub fn changed_paths(&self, from: &str, to: &str) -> Result<Vec<String>> {
        let output = ensure_success(
            run(&self.root, &["diff", "--name-only", "-z", from, to])?,
            "git diff failed",
            &self.root,
        )?;
        Ok(output
            .stdout
            .split(|&byte| byte == 0)
            .filter(|path| !path.is_empty())
            .map(|path| String::from_utf8_lossy(path).into_owned())
            .collect())
    }

    /// Which of `entries` — `(mode, sha, raw path)` triples — git itself
    /// accepts as index paths on this host. `fast-import` validates every
    /// path with git's `verify_path`, which on some hosts (Windows)
    /// rejects names that cannot exist there even though a repository
    /// cloned from elsewhere can carry them in HEAD; re-emitting such a
    /// path would abort the whole run, so the caller re-emits only what
    /// this probe accepts and reports the rest. The probe feeds
    /// `update-index --index-info` — the same validation — against a
    /// throwaway index file, so the repository's own index is never
    /// touched.
    pub fn accepted_tree_paths(
        &self,
        entries: &[(String, String, Vec<u8>)],
    ) -> Result<HashSet<Vec<u8>>> {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let mut accepted: HashSet<Vec<u8>> = HashSet::new();
        if entries.is_empty() {
            return Ok(accepted);
        }
        let index = std::env::temp_dir().join(format!(
            "ext4kit-path-probe-{}-{}.idx",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        let mut records = Vec::new();
        for (mode, sha, path) in entries {
            records.extend_from_slice(format!("{mode} {sha}\t").as_bytes());
            records.extend_from_slice(path);
            records.push(0);
        }
        let probe = (|| -> Result<()> {
            let mut feed = git_command(&self.root, &["update-index", "-z", "--index-info"]);
            feed.env("GIT_INDEX_FILE", &index)
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped());
            let mut child = feed
                .spawn()
                .map_err(|error| anyhow!("cannot run git update-index: {error}"))?;
            {
                let mut stdin = child.stdin.take().expect("stdin was piped");
                stdin.write_all(&records)?;
                stdin.flush()?;
            }
            let output = child.wait_with_output()?;
            ensure_success(output, "git update-index failed", &self.root)?;
            let mut list = git_command(&self.root, &["ls-files", "-z"]);
            list.env("GIT_INDEX_FILE", &index)
                .stdout(Stdio::piped())
                .stderr(Stdio::piped());
            let output = ensure_success(list.output()?, "git ls-files failed", &self.root)?;
            for path in output.stdout.split(|&byte| byte == 0) {
                if !path.is_empty() {
                    accepted.insert(path.to_vec());
                }
            }
            Ok(())
        })();
        let _ = fs::remove_file(&index);
        probe.map(|()| accepted)
    }
}

/// One `ls-tree -r -t` record.
#[derive(Clone, Debug)]
pub struct TreeEntry {
    /// The full mode string: `100644`, `100755`, `120000`, `160000`,
    /// `040000`.
    pub mode: String,
    pub kind: String,
    pub sha: String,
    /// The path as a lossy UTF-8 string, for lookups and display.
    pub path: String,
    /// The path's raw bytes, exactly as git holds them: a name that is
    /// not valid UTF-8 must be re-emittable byte for byte.
    pub raw_path: Vec<u8>,
}

impl TreeEntry {
    pub fn is_tree(&self) -> bool {
        self.mode == "040000"
    }

    pub fn is_gitlink(&self) -> bool {
        self.mode == "160000"
    }
}

/// A summary of an `ls-tree -r -t` listing: the input for the ignore
/// pruning guard, the exact-path exemption, the gitlink re-emission and
/// the carry-forward rule.
#[derive(Clone, Debug, Default)]
pub struct ParentTree {
    /// One entry per lossy path, for lookups and display. Two distinct raw
    /// paths can render to the same string when a name is not valid UTF-8,
    /// so a lookup answers "a HEAD entry exists here", not "this exact
    /// entry does"; use [`Self::listing`] when every entry must be visited.
    pub entries: HashMap<String, TreeEntry>,
    /// Every record of the listing in `ls-tree` order, including the
    /// distinct raw paths `entries` collapses onto one lossy key.
    pub listing: Vec<TreeEntry>,
    /// Directory prefixes that contain tracked content.
    pub tracked_prefixes: Vec<String>,
}

impl ParentTree {
    pub fn from_entries(entries: Vec<TreeEntry>) -> Self {
        let tracked_prefixes: Vec<String> = entries
            .iter()
            .filter(|entry| entry.is_tree())
            .map(|entry| entry.path.clone())
            .collect();
        let by_path = entries
            .iter()
            .map(|entry| (entry.path.clone(), entry.clone()))
            .collect();
        Self {
            entries: by_path,
            listing: entries,
            tracked_prefixes,
        }
    }

    /// Whether a directory holds tracked descendants, so it must not be
    /// pruned even when ignored (`build/` with a force-added
    /// `build/keep.txt`).
    pub fn has_tracked_descendants(&self, dir: &str) -> bool {
        let prefix = format!("{dir}/");
        self.tracked_prefixes
            .iter()
            .any(|path| path == dir || path.starts_with(&prefix))
    }
}

/// Finds the repository containing `path` (for a file, its parent).
/// `Ok(None)` when the path is not inside a repository; a `safe.directory`
/// refusal is an error with the config hint, because a wrapper running as
/// another user hits it on every call.
pub fn discover_repo(path: &Path) -> Result<Option<Repo>> {
    let start = if path.is_dir() {
        path.to_owned()
    } else {
        path.parent()
            .ok_or_else(|| anyhow!("{} has no parent directory", path.display()))?
            .to_owned()
    };
    let output = run(&start, &["rev-parse", "--show-toplevel"])?;
    if output.status.success() {
        let root = PathBuf::from(String::from_utf8_lossy(&output.stdout).trim());
        return Ok(Some(Repo::at(root)));
    }
    let stderr = String::from_utf8_lossy(&output.stderr);
    if stderr.contains("dubious ownership") {
        bail!(
            "git refused {} for ownership reasons; run `git config --global \
             --add safe.directory <path>` as this user",
            path.display()
        )
    }
    Ok(None)
}

/// Fails with git's stderr when `output` exited non-zero. `what` is the
/// failure phrase up to the repository — `"git status failed"`,
/// `"cannot resolve HEAD"` — so the message reads
/// `<what> in <repo>: <stderr>` exactly as each caller spelled it before
/// the helper existed.
fn ensure_success(
    output: std::process::Output,
    what: &str,
    repo: &Path,
) -> Result<std::process::Output> {
    if !output.status.success() {
        bail!(
            "{what} in {}: {}",
            repo.display(),
            String::from_utf8_lossy(&output.stderr)
        );
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::host::git::FastImport;
    use crate::host::git::test_support::repo_fixture;
    use std::fs;

    #[test]
    fn accepted_tree_paths_follow_the_host_git_rules() {
        let Some((_directory, repo)) = repo_fixture(&[("a.txt", "a\n")]) else {
            return;
        };
        let blob = repo.rev_parse("HEAD:a.txt").unwrap();
        let entries = vec![
            (
                "100644".to_owned(),
                blob.clone(),
                b"high\xffbyte.txt".to_vec(),
            ),
            ("100644".to_owned(), blob, b"ctrl\x01name.txt".to_vec()),
        ];
        let accepted = repo.accepted_tree_paths(&entries).unwrap();
        // A non-UTF-8 byte is a legal git path everywhere.
        assert!(accepted.contains(b"high\xffbyte.txt".as_slice()));
        // A control byte is one only where the host allows it; Windows git
        // refuses it, so the probe must agree with the platform.
        assert_eq!(
            accepted.contains(b"ctrl\x01name.txt".as_slice()),
            !cfg!(windows)
        );
    }

    #[test]
    fn parent_tree_tracks_descendants_for_the_pruning_guard() {
        let entries = vec![
            TreeEntry {
                mode: "040000".into(),
                kind: "tree".into(),
                sha: "1".into(),
                path: "build".into(),
                raw_path: b"build".to_vec(),
            },
            TreeEntry {
                mode: "100644".into(),
                kind: "blob".into(),
                sha: "2".into(),
                path: "build/keep.txt".into(),
                raw_path: b"build/keep.txt".to_vec(),
            },
        ];
        let tree = ParentTree::from_entries(entries);
        assert!(tree.has_tracked_descendants("build"));
        assert!(!tree.has_tracked_descendants("node_modules"));
    }

    #[test]
    fn the_listing_keeps_raw_paths_the_lossy_map_collapses() {
        // Two names that are not valid UTF-8 render to the same lossy
        // path. The map keeps one (for lookups); the listing keeps both,
        // so neither HEAD entry is lost when the tree is re-emitted.
        let entry = |raw: &[u8]| TreeEntry {
            mode: "100644".into(),
            kind: "blob".into(),
            sha: "1".into(),
            path: String::from_utf8_lossy(raw).into_owned(),
            raw_path: raw.to_vec(),
        };
        let tree = ParentTree::from_entries(vec![entry(b"caf\xe9"), entry(b"caf\xea")]);
        assert_eq!(tree.entries.len(), 1);
        assert_eq!(tree.listing.len(), 2);
    }

    #[test]
    fn dirty_paths_reports_changes_without_touching_the_index() {
        let Some((directory, repo)) = repo_fixture(&[("a.txt", "a\n")]) else {
            return;
        };
        assert!(repo.dirty_paths(&[]).unwrap().is_empty());
        let index_before = fs::read(directory.path().join(".git/index")).unwrap();
        fs::write(directory.path().join("a.txt"), "changed\n").unwrap();
        fs::write(directory.path().join("new.txt"), "new\n").unwrap();
        let dirty = repo.dirty_paths(&[]).unwrap();
        assert_eq!(dirty, vec!["a.txt".to_owned(), "new.txt".to_owned()]);
        let index_after = fs::read(directory.path().join(".git/index")).unwrap();
        assert_eq!(
            index_before, index_after,
            "the dirty check must not rewrite the index"
        );
        assert!(!directory.path().join(".git/index.lock").exists());
    }

    #[test]
    fn dirty_paths_treats_dash_leading_subtrees_literally() {
        // Without `--` and literal pathspecs, `-dist` reads as an option
        // and the whole dirty check fails.
        let Some((directory, repo)) = repo_fixture(&[("-dist/a.txt", "a\n")]) else {
            return;
        };
        fs::write(directory.path().join("-dist/a.txt"), "changed\n").unwrap();
        let dirty = repo.dirty_paths(&["-dist".to_owned()]).unwrap();
        assert_eq!(dirty, vec!["-dist/a.txt".to_owned()]);
    }

    #[test]
    fn labels_are_checked_case_insensitively_and_refs_move_to_previous() {
        let Some((_directory, repo)) = repo_fixture(&[("a.txt", "a\n")]) else {
            return;
        };
        let parent = repo.head_commit().unwrap().unwrap();
        let mut fast_import = FastImport::spawn(&repo.root).unwrap();
        fast_import
            .begin_commit("refs/ext4kit/out/base", Some(&parent), "base", 1)
            .unwrap();
        fast_import.finish().unwrap();
        assert!(repo.label_in_use("out").unwrap());
        assert!(repo.label_in_use("OUT").unwrap());
        assert!(!repo.label_in_use("other").unwrap());

        let moved = repo
            .move_label_refs_to_previous("out", 1_700_000_000)
            .unwrap();
        assert_eq!(moved.len(), 1);
        assert_eq!(moved[0].0, "refs/ext4kit/out/previous/1700000000/base");
        assert!(!repo.label_in_use("out").unwrap());
        assert_eq!(
            repo.rev_parse_optional("refs/ext4kit/out/previous/1700000000/base")
                .unwrap()
                .as_deref(),
            Some(moved[0].1.as_str())
        );

        // A second generation inside the same second must not overwrite
        // the first: it lands in the next timestamp instead.
        let mut fast_import = FastImport::spawn(&repo.root).unwrap();
        fast_import
            .begin_commit("refs/ext4kit/out/base", Some(&parent), "again", 1)
            .unwrap();
        fast_import.finish().unwrap();
        let again = repo
            .move_label_refs_to_previous("out", 1_700_000_000)
            .unwrap();
        assert_eq!(again.len(), 1);
        assert_eq!(again[0].0, "refs/ext4kit/out/previous/1700000001/base");
        assert!(
            repo.rev_parse_optional("refs/ext4kit/out/previous/1700000000/base")
                .unwrap()
                .is_some(),
            "the first generation survives"
        );
    }

    #[test]
    fn discover_repo_finds_the_toplevel_and_rejects_outside() {
        let Some((directory, repo)) = repo_fixture(&[("sub/a.txt", "a\n")]) else {
            return;
        };
        let canonical = |path: &Path| fs::canonicalize(path).unwrap();
        assert_eq!(
            discover_repo(&directory.path().join("sub"))
                .unwrap()
                .map(|repo| canonical(&repo.root)),
            Some(canonical(&repo.root))
        );
        assert_eq!(
            discover_repo(&directory.path().join("sub/a.txt"))
                .unwrap()
                .map(|repo| canonical(&repo.root)),
            Some(canonical(&repo.root))
        );
        let outside = tempfile::tempdir().unwrap();
        assert_eq!(
            discover_repo(outside.path()).unwrap().map(|repo| repo.root),
            None
        );
    }
}
