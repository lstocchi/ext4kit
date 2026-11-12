//! The git side of folder tracking: repo discovery and command hygiene,
//! the persistent `check-ignore` session, the `check-attr` batch, lazy
//! `cat-file` reference queries, the `hash-object` filter route, the
//! [`ExactLen`] framing writer, the eol normalizer that mirrors git's
//! `crlf_to_git`, and the fast-import stream writer.
//!
//! Every git process runs `git -C <repo>` with `GIT_DIR`, `GIT_WORK_TREE`,
//! `GIT_INDEX_FILE` and `GIT_OBJECT_DIRECTORY` removed from the inherited
//! environment (a wrapper must not silently point ext4kit at the wrong
//! repository) and with `GIT_OPTIONAL_LOCKS=0`, so commands such as
//! `git status` never refresh the index or take `index.lock`.
//!
//! Two pipe facts drive the concurrency here, both verified on git 2.55:
//! `--stdin` subprocesses emit output before EOF once their stdio buffer
//! fills, so any batch whose output can exceed the pipe buffer needs a
//! writer thread; and `GIT_FLUSH=1` makes `check-ignore` flush after every
//! record, which is what makes a persistent interactive session possible.

use anyhow::{Context, Result, anyhow, bail};
use std::{
    collections::{HashMap, HashSet},
    fs,
    io::{self, BufRead, BufReader, BufWriter, Read, Write},
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::{
        OnceLock,
        atomic::{AtomicU64, Ordering},
        mpsc,
    },
    thread,
    time::Duration,
};

/// How long the check-ignore reader waits for one record before declaring
/// the session dead. With `GIT_FLUSH=1` a record arrives immediately, so
/// this only fires when the flush contract breaks (or git hangs).
const IGNORE_RECORD_TIMEOUT: Duration = Duration::from_secs(30);

/// After this many check-ignore respawns in one run the snapshot aborts:
/// a systematic failure must not be silently turned into thousands of
/// carried-forward paths.
const MAX_IGNORE_RESPAWNS: u32 = 3;

/// Whether a `git` binary is available at all, cached for the process.
pub fn git_available() -> bool {
    static AVAILABLE: OnceLock<bool> = OnceLock::new();
    *AVAILABLE.get_or_init(|| {
        Command::new("git")
            .arg("--version")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .is_ok_and(|status| status.success())
    })
}

/// Builds a `git` command for `repo` with the shared environment hygiene.
fn git_command(repo: &Path, args: &[&str]) -> Command {
    let mut command = Command::new("git");
    command.arg("-C").arg(repo).args(args);
    for variable in [
        "GIT_DIR",
        "GIT_WORK_TREE",
        "GIT_INDEX_FILE",
        "GIT_OBJECT_DIRECTORY",
    ] {
        command.env_remove(variable);
    }
    // Read-only commands must never refresh or lock the index; this runs
    // alongside the user's own git commands.
    command.env("GIT_OPTIONAL_LOCKS", "0");
    // Git for Windows ships translations; every message this code matches
    // on ("dubious ownership", "missing") must be English regardless.
    command.env("LC_ALL", "C");
    command
}

/// Runs a git command and returns its output, mapping spawn failures (git
/// missing from PATH) to a clear error.
fn run(repo: &Path, args: &[&str]) -> Result<std::process::Output> {
    git_command(repo, args)
        .output()
        .map_err(|error| anyhow!("cannot run git (is it on PATH?): {error}"))
        .with_context(|| format!("git -C {} {}", repo.display(), args.join(" ")))
}

/// Runs a git command with one extra environment variable, like [`run`].
fn run_with_env(
    repo: &Path,
    args: &[&str],
    variable: &'static str,
    value: &str,
) -> Result<std::process::Output> {
    let mut command = git_command(repo, args);
    command.env(variable, value);
    command
        .output()
        .map_err(|error| anyhow!("cannot run git (is it on PATH?): {error}"))
        .with_context(|| format!("git -C {} {}", repo.display(), args.join(" ")))
}

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
        let output = run(&self.root, &["rev-parse", "--verify", rev])?;
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            bail!("cannot resolve {rev} in {}: {stderr}", self.root.display())
        }
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
        let output = run(&self.root, &["ls-tree", "-r", "-t", "-z", treeish])?;
        if !output.status.success() {
            bail!(
                "cannot list the tree of {treeish} in {}: {}",
                self.root.display(),
                String::from_utf8_lossy(&output.stderr)
            )
        }
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
        let output = run_with_env(&self.root, &args, "GIT_LITERAL_PATHSPECS", "1")?;
        if !output.status.success() {
            bail!(
                "git status failed in {}: {}",
                self.root.display(),
                String::from_utf8_lossy(&output.stderr)
            )
        }
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
        let output = run(
            &self.root,
            &["for-each-ref", "--format=%(refname)", "refs/ext4kit/"],
        )?;
        if !output.status.success() {
            bail!(
                "cannot list refs/ext4kit in {}: {}",
                self.root.display(),
                String::from_utf8_lossy(&output.stderr)
            )
        }
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
        if !output.status.success() {
            bail!(
                "git update-ref failed in {}: {}",
                self.root.display(),
                String::from_utf8_lossy(&output.stderr)
            )
        }
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
            if !output.status.success() {
                bail!(
                    "git update-index failed in {}: {}",
                    self.root.display(),
                    String::from_utf8_lossy(&output.stderr)
                );
            }
            let mut list = git_command(&self.root, &["ls-files", "-z"]);
            list.env("GIT_INDEX_FILE", &index)
                .stdout(Stdio::piped())
                .stderr(Stdio::piped());
            let output = list.output()?;
            if !output.status.success() {
                bail!(
                    "git ls-files failed in {}: {}",
                    self.root.display(),
                    String::from_utf8_lossy(&output.stderr)
                );
            }
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
    /// Every entry, keyed by path.
    pub entries: HashMap<String, TreeEntry>,
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
        let entries = entries
            .into_iter()
            .map(|entry| (entry.path.clone(), entry))
            .collect();
        Self {
            entries,
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

/// A writer that emits exactly `len` bytes and swallows anything beyond:
/// the framing integrity invariant. The length comes from one read of the
/// inode, and exactly that many bytes reach fast-import's `data <n>`
/// block — a file growing mid-stream must never let agent bytes become
/// fast-import commands. A short input is zero-padded; both directions
/// are reported as a mismatch by [`ExactLen::mismatch`].
pub struct ExactLen<W: Write> {
    inner: W,
    remaining: u64,
    overflowed: bool,
    padded: bool,
}

impl<W: Write> ExactLen<W> {
    pub fn new(inner: W, len: u64) -> Self {
        Self {
            inner,
            remaining: len,
            overflowed: false,
            padded: false,
        }
    }

    /// Zero-pads a short input to the framed length and flushes.
    pub fn finish(&mut self) -> io::Result<()> {
        let zeros = [0u8; 8192];
        while self.remaining > 0 {
            let take = (self.remaining.min(zeros.len() as u64)) as usize;
            self.inner.write_all(&zeros[..take])?;
            self.remaining -= take as u64;
            self.padded = true;
        }
        self.inner.flush()
    }

    /// Whether the input did not match the framed length.
    pub fn mismatch(&self) -> bool {
        self.overflowed || self.padded || self.remaining > 0
    }

    pub fn into_inner(self) -> W {
        self.inner
    }
}

impl<W: Write> Write for ExactLen<W> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        if buf.is_empty() {
            return Ok(0);
        }
        if self.remaining == 0 {
            // Excess input is dropped, not passed on.
            self.overflowed = true;
            return Ok(buf.len());
        }
        let take = (buf.len() as u64).min(self.remaining) as usize;
        self.inner.write_all(&buf[..take])?;
        self.remaining -= take as u64;
        if take < buf.len() {
            self.overflowed = true;
        }
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}

/// `core.autocrlf`, the part of it the conversion rules need.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Autocrlf {
    True,
    Input,
    False,
}

/// One `check-attr` answer.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum AttrValue {
    #[default]
    Unspecified,
    True,
    False,
    Value(String),
}

impl AttrValue {
    fn parse(raw: &[u8]) -> Self {
        match raw {
            b"set" => Self::True,
            b"unset" => Self::False,
            b"unspecified" => Self::Unspecified,
            other => Self::Value(String::from_utf8_lossy(other).into_owned()),
        }
    }
}

/// The attributes that decide how one path's content is converted.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PathAttrs {
    pub text: AttrValue,
    pub eol: AttrValue,
    pub filter: AttrValue,
    pub ident: AttrValue,
    pub working_tree_encoding: AttrValue,
}

/// How one file's content reaches the object database. The route is also
/// the hard-link sharing class: two links to one inode share a blob only
/// when their routes are equal, because the route is a function of the
/// path's attributes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ConversionRoute {
    /// Raw bytes, no conversion.
    Raw,
    /// CRLF to LF, with git's rules for the mode.
    Normalize(EolRule),
    /// Through `git hash-object -w --stdin --path`, which runs the user's
    /// clean filter on agent-controlled bytes at snapshot time.
    Filter,
}

/// Which CRLF-to-LF rules apply, matching git's `crlf_action`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum EolRule {
    /// `text`, `eol=`: always convert, binary or not.
    Explicit,
    /// `text=auto`, `autocrlf true/input`: convert unless the content is
    /// binary, has no CRLF, or the reference blob already carries CRLF.
    Auto,
}

/// Decides the conversion route for one path from its attributes and
/// `core.autocrlf`, mirroring git's `convert_attrs` / `crlf_to_git`
/// gating. Filter, `ident` and `working-tree-encoding` paths always take
/// the hash-object route, because a clean command may depend on the path
/// (`%f`), so hard links never share a blob there.
pub fn conversion_route(attrs: &PathAttrs, autocrlf: Autocrlf) -> ConversionRoute {
    // Only attributes that positively select a driver or expansion take
    // the hash-object route: an explicit `-filter` or `-ident` means
    // "none", and paying a process spawn per file for it is pure cost.
    let enabled = |value: &AttrValue| matches!(value, AttrValue::True | AttrValue::Value(_));
    if enabled(&attrs.filter) || enabled(&attrs.ident) || enabled(&attrs.working_tree_encoding) {
        return ConversionRoute::Filter;
    }
    // `eol=lf` / `eol=crlf`; a bare or negated `eol` selects nothing.
    let eol_set = matches!(&attrs.eol, AttrValue::Value(_));
    match &attrs.text {
        AttrValue::True => ConversionRoute::Normalize(EolRule::Explicit),
        AttrValue::False => ConversionRoute::Raw,
        AttrValue::Value(value) if value == "auto" => ConversionRoute::Normalize(EolRule::Auto),
        // Unknown text values behave as unspecified, like git's
        // git_path_check_crlf.
        AttrValue::Value(_) | AttrValue::Unspecified => {
            if eol_set {
                ConversionRoute::Normalize(EolRule::Explicit)
            } else {
                match autocrlf {
                    Autocrlf::True | Autocrlf::Input => ConversionRoute::Normalize(EolRule::Auto),
                    Autocrlf::False => ConversionRoute::Raw,
                }
            }
        }
    }
}

/// The line-ending statistics git's `gather_stats` collects, and its
/// binary heuristic: any NUL, a lone CR, or too many non-printables
/// (non-printables exceeding printables over 128). Feeding chunk by chunk
/// keeps the CR carry across chunk boundaries exact.
#[derive(Default)]
pub struct TextStat {
    cr: u64,
    crlf: u64,
    nul: bool,
    printable: u64,
    nonprintable: u64,
    carry_cr: bool,
}

impl TextStat {
    pub fn gather(content: &[u8]) -> Self {
        let mut stats = Self::default();
        stats.feed(content);
        stats
    }

    pub fn feed(&mut self, chunk: &[u8]) {
        for &byte in chunk {
            match byte {
                b'\r' => self.cr += 1,
                b'\n' => {
                    if self.carry_cr {
                        self.crlf += 1;
                    }
                }
                0x7f => self.nonprintable += 1,
                // \b, \t, ESC and FF count as printable, like git.
                0x08 | 0x09 | 0x1b | 0x0c => self.printable += 1,
                byte if byte < 32 => {
                    if byte == 0 {
                        self.nul = true;
                    }
                    self.nonprintable += 1;
                }
                _ => self.printable += 1,
            }
            self.carry_cr = byte == b'\r';
        }
    }

    pub fn is_binary(&self) -> bool {
        let lone_cr = self.cr - self.crlf;
        self.nul || lone_cr > 0 || (self.printable >> 7) < self.nonprintable
    }

    pub fn has_crlf(&self) -> bool {
        self.crlf > 0
    }

    /// The number of CRLF pairs, which is also how many bytes a CRLF-to-LF
    /// conversion removes.
    pub fn crlf_pairs(&self) -> u64 {
        self.crlf
    }
}

/// git's binary heuristic over content.
pub fn is_binary(content: &[u8]) -> bool {
    TextStat::gather(content).is_binary()
}

/// Whether a reference blob suppresses an auto conversion, git's
/// `has_crlf_in_index`: not binary and carrying CRLF (the `i/crlf` and
/// `i/mixed` cases).
pub fn reference_has_crlf(content: &[u8]) -> bool {
    let stats = TextStat::gather(content);
    !stats.is_binary() && stats.has_crlf()
}

/// Strips the CR of every CRLF pair, git's `crlf_to_git` output step.
pub fn crlf_to_git(content: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(content.len());
    let mut previous_cr = false;
    for &byte in content {
        if byte == b'\n' && previous_cr {
            out.pop();
        }
        out.push(byte);
        previous_cr = byte == b'\r';
    }
    out
}

/// A bounded spool for normalized content: `data <n>` needs the exact
/// length up front, so a converting file is materialized — in memory up
/// to [`SPOOL_MEMORY_LIMIT`] bytes, in a temp file above it, preserving
/// the bounded-memory invariant.
pub enum Spool {
    Memory(Vec<u8>),
    Temp {
        file: fs::File,
        path: PathBuf,
        len: u64,
    },
}

/// In-memory spool threshold; per-file opens dominate on Windows
/// (~180 microseconds each), so small files stay in memory.
pub const SPOOL_MEMORY_LIMIT: u64 = 8 * 1024 * 1024;

static SPOOL_COUNTER: AtomicU64 = AtomicU64::new(0);

impl Spool {
    pub fn create() -> io::Result<Self> {
        Ok(Self::Memory(Vec::new()))
    }

    pub fn write(&mut self, buf: &[u8]) -> io::Result<()> {
        match self {
            Self::Memory(data) => {
                if data.len() as u64 + buf.len() as u64 > SPOOL_MEMORY_LIMIT {
                    // Move the memory out first: the new value replaces
                    // the whole enum, so it cannot borrow from itself.
                    let drained = std::mem::take(data);
                    let mut temp = Self::make_temp_file()?;
                    temp.write(&drained)?;
                    temp.write(buf)?;
                    *self = temp;
                } else {
                    data.extend_from_slice(buf);
                }
                Ok(())
            }
            Self::Temp { file, len, .. } => {
                file.write_all(buf)?;
                *len += buf.len() as u64;
                Ok(())
            }
        }
    }

    fn make_temp_file() -> io::Result<Self> {
        let index = SPOOL_COUNTER.fetch_add(1, Ordering::Relaxed);
        let path =
            std::env::temp_dir().join(format!("ext4kit-spool-{}-{index}.tmp", std::process::id()));
        let file = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&path)?;
        Ok(Self::Temp { file, path, len: 0 })
    }

    /// The spooled byte count.
    pub fn len(&self) -> u64 {
        match self {
            Self::Memory(data) => data.len() as u64,
            Self::Temp { len, .. } => *len,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Streams the spooled content in chunks to `sink`. The temp file is
    /// reopened for reading; Rust's open shares read access on Windows.
    pub fn for_each_chunk(&self, mut sink: impl FnMut(&[u8]) -> io::Result<()>) -> io::Result<()> {
        const CHUNK: usize = 512 * 1024;
        match self {
            Self::Memory(data) => {
                for chunk in data.chunks(CHUNK) {
                    sink(chunk)?;
                }
                Ok(())
            }
            Self::Temp { path, len, .. } => {
                let mut reader = BufReader::with_capacity(CHUNK, fs::File::open(path)?);
                let mut remaining = *len;
                let mut buffer = vec![0u8; CHUNK];
                while remaining > 0 {
                    let take = (remaining as usize).min(CHUNK);
                    reader.read_exact(&mut buffer[..take])?;
                    sink(&buffer[..take])?;
                    remaining -= take as u64;
                }
                Ok(())
            }
        }
    }

    /// Streams the spooled content through a CRLF-to-LF conversion to
    /// `sink`. The byte count is the spool length minus the CRLF pair
    /// count the caller already computed for the `data <n>` header, so
    /// the output must be exactly that long: a CRLF pair straddling a
    /// chunk boundary is handled by withholding the trailing CR until
    /// the next chunk's first byte is known — emitting it up front would
    /// push one byte past the framing and let image bytes reach
    /// fast-import's command parser.
    pub fn stream_crlf_to_git(
        &self,
        mut sink: impl FnMut(&[u8]) -> io::Result<()>,
    ) -> io::Result<()> {
        // A CR withheld at the end of the previous chunk, not yet emitted.
        let mut withheld_cr = false;
        self.for_each_chunk(|chunk| {
            let mut start = 0;
            if withheld_cr {
                if chunk.first() == Some(&b'\n') {
                    // The withheld CR pairs with this LF: drop it.
                    sink(b"\n")?;
                    start = 1;
                } else {
                    // A lone CR: it stays in the output.
                    sink(b"\r")?;
                }
                withheld_cr = false;
            }
            let mut previous_cr = false;
            for (index, &byte) in chunk.iter().enumerate().skip(start) {
                if byte == b'\n' && previous_cr {
                    // Drop the CR at index - 1, which this chunk still holds.
                    sink(&chunk[start..index - 1])?;
                    sink(b"\n")?;
                    start = index + 1;
                }
                previous_cr = byte == b'\r';
            }
            let mut end = chunk.len();
            if previous_cr && end > start {
                // The chunk's last byte is a CR that may pair with the
                // next chunk's LF: withhold it rather than emit it now.
                withheld_cr = true;
                end -= 1;
            }
            sink(&chunk[start..end])?;
            Ok(())
        })?;
        // A CR at the very end of the content is a lone CR and stays.
        if withheld_cr {
            sink(b"\r")?;
        }
        Ok(())
    }
}

impl Write for Spool {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        Spool::write(self, buf)?;
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        match self {
            Self::Memory(_) => Ok(()),
            Self::Temp { file, .. } => file.flush(),
        }
    }
}

impl Drop for Spool {
    fn drop(&mut self) {
        if let Self::Temp { path, .. } = self {
            let _ = fs::remove_file(path);
        }
    }
}

/// The answer to one ignore query: ignored or not, or the reason the
/// query failed (the path is then treated like any other unreadable path
/// and carried forward).
pub type IgnoreAnswer = Result<bool, String>;

/// A persistent `git check-ignore --no-index --stdin -z -v -n` session,
/// queried once per directory while the walk runs.
///
/// The protocol, verified on git 2.55:
/// - with `-v -n` every query yields exactly one NUL-separated record
///   `<source>\0<line>\0<pattern>\0<path>\0`, with empty fields for a
///   non-match — without them an interactive reader cannot tell "pending"
///   from "not ignored";
/// - a pattern starting with `!` is a negation: not ignored;
/// - each stdin line is parsed as a **pathspec**, so an agent-chosen name
///   such as `:(foo)x` kills the process and `:(top)` is interpreted
///   rather than matched. `GIT_LITERAL_PATHSPECS` is rejected by
///   check-ignore; the `./` prefix makes the query literal and git echoes
///   it back, which the reader strips;
/// - directory queries carry a trailing `/`, because check-ignore stats
///   the *host worktree* to resolve a path's type, and a directory that
///   exists only in the image does not match a `dir/` pattern without it;
/// - `GIT_FLUSH=1` makes git flush after every record, so records arrive
///   while stdin is still open; with `-z` the input paths are
///   NUL-terminated, not newline-terminated.
///
/// A dead or desynchronized session (process exit, record mismatch,
/// timeout) is recoverable: the entries are re-queried one at a time on a
/// freshly spawned session, isolating the offending path, which fails
/// with git's stderr. After [`MAX_IGNORE_RESPAWNS`] respawns the run
/// aborts.
pub struct CheckIgnoreSession {
    repo: PathBuf,
    child: Child,
    stderr: Option<std::process::ChildStderr>,
    queries: Option<mpsc::Sender<Vec<String>>>,
    records: mpsc::Receiver<Vec<Vec<u8>>>,
    respawns: u32,
}

impl CheckIgnoreSession {
    pub fn new(repo: &Path) -> Result<Self> {
        let (child, stderr, queries, records) = spawn_ignore_session(repo)?;
        Ok(Self {
            repo: repo.to_owned(),
            child,
            stderr,
            queries: Some(queries),
            records,
            respawns: 0,
        })
    }

    /// Queries `entries` — `(repo-relative path, is-directory)` — in one
    /// round trip and returns one answer per entry, in order. A path
    /// containing a NUL byte cannot cross the `-z` protocol (it would
    /// split the record and desynchronize the session), so those entries
    /// answer `Err` locally without a respawn — a guard behind the
    /// planner's own name filter.
    pub fn check(&mut self, entries: &[(String, bool)]) -> Result<Vec<IgnoreAnswer>> {
        if entries.iter().all(|(path, _)| !path.contains('\0')) {
            return self.check_batch(entries);
        }
        let mut answers: Vec<IgnoreAnswer> = Vec::with_capacity(entries.len());
        let mut others: Vec<(String, bool)> = Vec::new();
        let mut positions: Vec<usize> = Vec::new();
        for (index, (path, is_dir)) in entries.iter().enumerate() {
            if path.contains('\0') {
                answers.push(Err("the path contains a NUL byte".to_owned()));
            } else {
                // A placeholder, replaced with the real answer below.
                answers.push(Ok(false));
                positions.push(index);
                others.push((path.clone(), *is_dir));
            }
        }
        for (position, answer) in positions.into_iter().zip(self.check_batch(&others)?) {
            answers[position] = answer;
        }
        Ok(answers)
    }

    /// The batch round trip without the NUL guard.
    fn check_batch(&mut self, entries: &[(String, bool)]) -> Result<Vec<IgnoreAnswer>> {
        let lines: Vec<String> = entries
            .iter()
            .map(|(path, is_dir)| format!("./{path}{}", if *is_dir { "/" } else { "" }))
            .collect();
        match self.try_batch(&lines) {
            Ok(answers) => Ok(answers.into_iter().map(Ok).collect()),
            Err(_) => {
                // The session died or desynchronized mid-batch. Respawn
                // and ask every entry alone: only the entry that kills
                // the session again fails, with git's stderr as the
                // reason; everything else answers normally.
                self.respawn()?;
                let mut answers = Vec::with_capacity(entries.len());
                for (index, line) in lines.iter().enumerate() {
                    match self.try_batch(std::slice::from_ref(line)) {
                        Ok(mut one) => answers.push(Ok(one.remove(0))),
                        Err(_) => {
                            // Make sure stderr reaches EOF even when git
                            // merely hung, then read it for the message.
                            let _ = self.child.kill();
                            let _ = self.child.wait();
                            let stderr = self.take_stderr();
                            answers.push(Err(format!(
                                "ignore query failed for {}: {stderr}",
                                entries[index].0
                            )));
                            self.respawn()?;
                        }
                    }
                }
                Ok(answers)
            }
        }
    }

    /// One round trip on the live session; `Err(index)` names the entry
    /// whose answer did not arrive or did not match.
    fn try_batch(&mut self, lines: &[String]) -> Result<Vec<bool>, usize> {
        if self.child.try_wait().ok().flatten().is_some() {
            return Err(0);
        }
        let sender = match &self.queries {
            Some(sender) => sender.clone(),
            None => return Err(0),
        };
        if sender.send(lines.to_vec()).is_err() {
            return Err(0);
        }
        let mut answers = Vec::with_capacity(lines.len());
        for (index, line) in lines.iter().enumerate() {
            let record = match self.records.recv_timeout(IGNORE_RECORD_TIMEOUT) {
                Ok(record) => record,
                Err(_) => return Err(index),
            };
            if record.len() != 4 || String::from_utf8_lossy(&record[3]) != *line {
                return Err(index);
            }
            let pattern = &record[2];
            let ignored = !pattern.is_empty() && !pattern.starts_with(b"!");
            answers.push(ignored);
        }
        Ok(answers)
    }

    fn respawn(&mut self) -> Result<()> {
        self.respawns += 1;
        if self.respawns > MAX_IGNORE_RESPAWNS {
            bail!(
                "check-ignore needed more than {MAX_IGNORE_RESPAWNS} respawns in {} \
                 ({}); aborting",
                self.repo.display(),
                git_version_or_empty()
            );
        }
        let _ = self.child.kill();
        let _ = self.child.wait();
        // Dropping the sender ends the writer thread, which closes stdin.
        self.queries = None;
        let (child, stderr, queries, records) = spawn_ignore_session(&self.repo)?;
        self.child = child;
        self.stderr = stderr;
        self.queries = Some(queries);
        self.records = records;
        Ok(())
    }

    /// Drains git's stderr after a death, for the failing path's message.
    fn take_stderr(&mut self) -> String {
        let mut message = String::new();
        if let Some(stderr) = self.stderr.as_mut() {
            let _ = stderr.read_to_string(&mut message);
        }
        message.trim().to_owned()
    }
}

fn git_version_or_empty() -> String {
    Command::new("git")
        .arg("--version")
        .output()
        .ok()
        .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_owned())
        .unwrap_or_default()
}

type IgnoreSessionParts = (
    Child,
    Option<std::process::ChildStderr>,
    mpsc::Sender<Vec<String>>,
    mpsc::Receiver<Vec<Vec<u8>>>,
);

/// Spawns one check-ignore process with its writer and reader threads.
/// The writer thread owns stdin, so the pipe stays open between batches;
/// git emits `--stdin` output before EOF once its stdio buffer fills
/// (verified: 3000 paths produce 24 KiB with stdin still open), so a
/// sequential write-then-read would deadlock on large directories.
fn spawn_ignore_session(repo: &Path) -> Result<IgnoreSessionParts> {
    let mut command = git_command(
        repo,
        &["check-ignore", "--no-index", "--stdin", "-z", "-v", "-n"],
    );
    command
        .env("GIT_FLUSH", "1")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = command
        .spawn()
        .map_err(|error| anyhow!("cannot run git check-ignore: {error}"))?;
    let mut stdin = child.stdin.take().expect("stdin was piped");
    let stdout = child.stdout.take().expect("stdout was piped");
    let stderr = child.stderr.take();
    let (query_tx, query_rx) = mpsc::channel::<Vec<String>>();
    thread::spawn(move || {
        // Holding stdin open across batches is the point of the thread;
        // it closes only when the channel ends or a write fails. With -z
        // the input paths are NUL-terminated, not newline-terminated
        // (verified on git 2.55: a newline becomes part of the path).
        for batch in query_rx {
            let mut failed = false;
            for line in &batch {
                if stdin.write_all(line.as_bytes()).is_err() || stdin.write_all(&[0]).is_err() {
                    failed = true;
                    break;
                }
            }
            if failed || stdin.flush().is_err() {
                break;
            }
        }
    });
    let (record_tx, record_rx) = mpsc::channel::<Vec<Vec<u8>>>();
    thread::spawn(move || {
        let mut reader = BufReader::new(stdout);
        loop {
            let mut fields: Vec<Vec<u8>> = Vec::with_capacity(4);
            let mut closed = false;
            for _ in 0..4 {
                let mut field = Vec::new();
                match reader.read_until(0, &mut field) {
                    Ok(0) => {
                        closed = true;
                        break;
                    }
                    Ok(_) => {
                        field.pop();
                        fields.push(field);
                    }
                    Err(_) => {
                        closed = true;
                        break;
                    }
                }
            }
            if closed || record_tx.send(fields).is_err() {
                break;
            }
        }
    });
    Ok((child, stderr, query_tx, record_rx))
}

/// The `git check-attr --stdin -z` attributes for `paths`, one batch with
/// a writer thread (write all, close stdin, read concurrently): check-attr
/// prints five records per path, so real repositories always exceed the
/// pipe buffer and a sequential write-close-read would deadlock.
pub fn check_attr(repo: &Path, paths: &[String]) -> Result<HashMap<String, PathAttrs>> {
    const ATTRIBUTES: [&str; 5] = ["text", "eol", "filter", "ident", "working-tree-encoding"];
    let mut command = git_command(repo, &["check-attr", "--stdin", "-z"]);
    command.args(ATTRIBUTES);
    command.stdin(Stdio::piped()).stdout(Stdio::piped());
    let mut child = command
        .spawn()
        .map_err(|error| anyhow!("cannot run git check-attr: {error}"))?;
    let mut stdin = child.stdin.take().expect("stdin was piped");
    let paths_owned = paths.to_vec();
    thread::spawn(move || {
        // With -z the input paths are NUL-terminated, like check-ignore.
        for path in &paths_owned {
            if stdin.write_all(path.as_bytes()).is_err() || stdin.write_all(&[0]).is_err() {
                break;
            }
        }
        // Dropping stdin closes the pipe and ends the batch.
    });
    let mut stdout = child.stdout.take().expect("stdout was piped");
    let mut raw = Vec::new();
    stdout.read_to_end(&mut raw)?;
    let status = child.wait()?;
    if !status.success() {
        bail!("git check-attr failed in {}", repo.display());
    }
    // -z output: <path>\0<attr>\0<value>\0 repeated, five triples per
    // path in command-line order.
    let mut fields: Vec<&[u8]> = raw.split(|&byte| byte == 0).collect();
    fields.pop(); // The trailing NUL leaves one empty field.
    let mut attributes: HashMap<String, PathAttrs> = HashMap::new();
    for triple in fields.chunks(3) {
        if triple.len() < 3 {
            continue;
        }
        let path = String::from_utf8_lossy(triple[0]).into_owned();
        let entry = attributes.entry(path).or_default();
        let value = AttrValue::parse(triple[2]);
        match String::from_utf8_lossy(triple[1]).as_ref() {
            "text" => entry.text = value,
            "eol" => entry.eol = value,
            "filter" => entry.filter = value,
            "ident" => entry.ident = value,
            "working-tree-encoding" => entry.working_tree_encoding = value,
            _ => {}
        }
    }
    Ok(attributes)
}

/// A lazy `git cat-file --batch` session for reference-blob queries. It
/// flushes after each object (verified), so a write-one-read-one loop
/// never deadlocks. Used against the fixed tree `base^` (HEAD at create)
/// so the answers do not drift with the current index.
pub struct CatFileBatch {
    child: Child,
    stdin: std::process::ChildStdin,
    stdout: BufReader<std::process::ChildStdout>,
}

impl CatFileBatch {
    pub fn new(repo: &Path) -> Result<Self> {
        let mut command = git_command(repo, &["cat-file", "--batch"]);
        command.stdin(Stdio::piped()).stdout(Stdio::piped());
        let mut child = command
            .spawn()
            .map_err(|error| anyhow!("cannot run git cat-file: {error}"))?;
        let stdin = child.stdin.take().expect("stdin was piped");
        let stdout = child.stdout.take().expect("stdout was piped");
        Ok(Self {
            child,
            stdin,
            stdout: BufReader::new(stdout),
        })
    }

    /// The content of the object named `name` (any extended SHA-1 syntax,
    /// e.g. `HEAD~1:src/a.rs`), or `None` when it does not exist. A path
    /// containing `:` cannot be expressed in that syntax and reads as
    /// missing, which only costs the eol exception for that path.
    pub fn blob(&mut self, name: &str) -> Result<Option<Vec<u8>>> {
        if writeln!(self.stdin, "{name}")
            .and_then(|()| self.stdin.flush())
            .is_err()
        {
            bail!("cannot query git cat-file: the process stopped responding");
        }
        let mut header = String::new();
        self.stdout
            .read_line(&mut header)
            .map_err(|error| anyhow!("git cat-file stopped responding: {error}"))?;
        let header = header.trim().to_owned();
        if header.is_empty() || header.ends_with(" missing") {
            return Ok(None);
        }
        let mut parts = header.split_whitespace();
        let (Some(_oid), Some(kind), Some(size)) = (parts.next(), parts.next(), parts.next())
        else {
            bail!("git cat-file answered unexpectedly: {header:?}");
        };
        let size: usize = size
            .parse()
            .with_context(|| format!("git cat-file reported a bad size: {header:?}"))?;
        let mut content = vec![0u8; size];
        self.stdout.read_exact(&mut content)?;
        let mut newline = [0u8; 1];
        self.stdout.read_exact(&mut newline)?;
        if kind != "blob" {
            // A tree or gitlink where the image now has a file: there is
            // no reference blob, exactly like a missing path, and the
            // auto eol rule falls back to converting. The content was
            // consumed above to keep the session in sync.
            return Ok(None);
        }
        Ok(Some(content))
    }
}

impl Drop for CatFileBatch {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// One `git hash-object -w --stdin --path=<path>` run: the filter route.
/// Image content streams through the framed writer into git's stdin, so
/// the clean filter (a user-configured command, running on
/// agent-controlled bytes at snapshot time — a stated property of the
/// design) sees exactly `len` bytes.
pub struct HashObject {
    child: Child,
    framed: Option<ExactLen<std::process::ChildStdin>>,
    path: String,
}

impl HashObject {
    pub fn spawn(repo: &Path, path: &str, len: u64) -> Result<Self> {
        let mut command = git_command(repo, &["hash-object", "-w", "--stdin"]);
        command.arg(format!("--path={path}"));
        command
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        let mut child = command
            .spawn()
            .map_err(|error| anyhow!("cannot run git hash-object: {error}"))?;
        let stdin = child.stdin.take().expect("stdin was piped");
        Ok(Self {
            child,
            framed: Some(ExactLen::new(stdin, len)),
            path: path.to_owned(),
        })
    }

    /// The framed writer receiving the file's content.
    pub fn writer(&mut self) -> &mut ExactLen<std::process::ChildStdin> {
        self.framed.as_mut().expect("finish takes the writer once")
    }

    /// Finishes the framing, closes stdin and returns the object name,
    /// plus whether the content did not match the framed length. A
    /// non-zero exit (a missing or crashing `filter.<driver>.required`)
    /// is an error naming git's stderr.
    pub fn finish(mut self) -> Result<(String, bool)> {
        let mut framed = self.framed.take().expect("finish takes the writer once");
        framed.finish()?;
        let mismatch = framed.mismatch();
        drop(framed);
        let output = self.child.wait_with_output()?;
        if !output.status.success() {
            bail!(
                "clean filter failed for {}: {}",
                self.path,
                String::from_utf8_lossy(&output.stderr).trim()
            )
        }
        let sha = String::from_utf8_lossy(&output.stdout).trim().to_owned();
        if sha.is_empty() {
            bail!("git hash-object printed no object name for {}", self.path);
        }
        Ok((sha, mismatch))
    }
}

/// One `git fast-import --quiet --done` run. Blobs stream in through the
/// [`Write`] impl (wrapped in [`ExactLen`] by the caller), the commit is
/// assembled from `deleteall`/`D` and `M` commands, and [`FastImport::finish`]
/// closes the stream with `done` — so a run killed anywhere before that
/// fails instead of committing a partial tree. Without `--force`,
/// fast-import only fast-forwards refs.
///
/// Agent-controlled bytes reach only `data <n>` blocks; every path in a
/// command goes through [`fi_path`], which quotes anything that could
/// confuse the parser.
pub struct FastImport {
    child: Option<Child>,
    stdin: Option<BufWriter<std::process::ChildStdin>>,
    next_mark: u32,
}

impl FastImport {
    pub fn spawn(repo: &Path) -> Result<Self> {
        let mut command = git_command(repo, &["fast-import", "--quiet", "--done"]);
        command
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        let mut child = command
            .spawn()
            .map_err(|error| anyhow!("cannot run git fast-import: {error}"))?;
        let stdin = child.stdin.take().expect("stdin was piped");
        Ok(Self {
            child: Some(child),
            stdin: Some(BufWriter::with_capacity(256 * 1024, stdin)),
            next_mark: 0,
        })
    }

    /// The stream writer; `finish` consumes it.
    fn stream(&mut self) -> &mut BufWriter<std::process::ChildStdin> {
        self.stdin.as_mut().expect("finish consumed the writer")
    }

    fn command(&mut self, line: &str) -> Result<()> {
        writeln!(self.stream(), "{line}").map_err(|error| anyhow!("fast-import stream: {error}"))
    }

    /// Writes a blob from memory and returns its mark.
    pub fn blob(&mut self, data: &[u8]) -> Result<u32> {
        let mark = self.reserve_mark();
        self.command(&format!("blob\nmark :{mark}\ndata {}", data.len()))?;
        self.stream()
            .write_all(data)
            .map_err(|error| anyhow!("fast-import stream: {error}"))?;
        Ok(mark)
    }

    /// Starts a `data <len>` block; the caller writes exactly `len` bytes
    /// through [`FastImport`]'s [`Write`] impl (usually inside
    /// [`ExactLen`]). Returns the mark for streamed blobs.
    pub fn begin_blob(&mut self) -> Result<u32> {
        let mark = self.reserve_mark();
        self.command(&format!("blob\nmark :{mark}"))?;
        Ok(mark)
    }

    pub fn begin_data(&mut self, len: u64) -> Result<()> {
        self.command(&format!("data {len}"))
    }

    fn reserve_mark(&mut self) -> u32 {
        self.next_mark += 1;
        self.next_mark
    }

    /// Starts a commit on `reference`. `parent` is a resolved SHA, never
    /// a symbolic name.
    pub fn begin_commit(
        &mut self,
        reference: &str,
        parent: Option<&str>,
        message: &str,
        unix_time: u64,
    ) -> Result<()> {
        self.command(&format!("commit {reference}"))?;
        let who = format!("ext4kit <ext4kit@localhost> {unix_time} +0000");
        self.command(&format!("author {who}"))?;
        self.command(&format!("committer {who}"))?;
        self.command(&format!("data {}", message.len()))?;
        self.stream()
            .write_all(message.as_bytes())
            .map_err(|error| anyhow!("fast-import stream: {error}"))?;
        if let Some(parent) = parent {
            self.command(&format!("from {parent}"))?;
        }
        Ok(())
    }

    /// Clears the whole tree; the only spelling for a mapping that covers
    /// the repository root, because fast-import rejects `D` with an empty
    /// path. It also clears gitlinks, which is why they are re-emitted.
    pub fn delete_all(&mut self) -> Result<()> {
        self.command("deleteall")
    }

    /// Deletes one subtree (or file).
    pub fn delete_path(&mut self, path: &str) -> Result<()> {
        self.command(&format!("D {}", fi_path(path)))
    }

    /// One tree entry: `mode` is `100644`, `100755`, `120000`, `160000`
    /// or `040000`; `source` is either a mark (`:7`) or an object name.
    pub fn modify(&mut self, mode: &str, source: &str, path: &str) -> Result<()> {
        self.command(&format!("M {mode} {} {}", source, fi_path(path)))
    }

    /// One tree entry whose path is raw bytes that need not be UTF-8,
    /// for the carry-forward of host-impossible HEAD paths.
    pub fn modify_raw(&mut self, mode: &str, source: &str, path: &[u8]) -> Result<()> {
        self.command(&format!("M {mode} {} {}", source, fi_path_bytes(path)))
    }

    /// Ends the stream with `done`, waits for fast-import and fails on a
    /// non-zero exit, including git's stderr in the message.
    pub fn finish(mut self) -> Result<()> {
        let result = (|| -> Result<std::process::Output> {
            self.command("done")?;
            self.stream()
                .flush()
                .map_err(|error| anyhow!("fast-import stream: {error}"))?;
            // Drop the writer to close stdin, then reap the child; the
            // Option fields let Drop coexist with taking them here.
            drop(self.stdin.take());
            let child = self.child.take().expect("the child lives until finish");
            child
                .wait_with_output()
                .map_err(|error| anyhow!("git fast-import failed: {error}"))
        })();
        match result {
            Ok(output) if output.status.success() => Ok(()),
            Ok(output) => bail!(
                "git fast-import failed: {}",
                String::from_utf8_lossy(&output.stderr).trim()
            ),
            Err(error) => Err(error),
        }
    }
}

impl Write for FastImport {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.stream().write(buf)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.stream().flush()
    }
}

impl Drop for FastImport {
    fn drop(&mut self) {
        // An abandoned run (an error mid-stream) must not leave a
        // fast-import process alive; without `done` it would fail anyway
        // once the pipe closes. After `finish` both fields are gone.
        if let Some(child) = self.child.as_mut() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

/// Formats a path for a fast-import command. Unquoted paths may not
/// start with `"`; a path containing one — legal on Unix — switches to
/// the quoted form, which escapes `"` and `\` and octal-escapes
/// non-printable bytes. Names with control bytes never reach the tree
/// (the name rules exclude them), so this is belt and braces.
pub fn fi_path(path: &str) -> String {
    fi_path_bytes(path.as_bytes())
}

/// The raw-byte form of [`fi_path`]: C-style quoting with `\ooo` octal
/// escapes for every byte that is not printable ASCII, the same escaping
/// git itself uses, so a HEAD tree entry with a host-impossible name
/// survives the carry-forward re-emission byte for byte.
pub fn fi_path_bytes(path: &[u8]) -> String {
    let needs_quoting = path.first() == Some(&b'"')
        || path
            .iter()
            .any(|&byte| byte == b'"' || byte == b'\\' || !byte.is_ascii_graphic() && byte != b' ');
    if !needs_quoting {
        // Only printable ASCII and spaces reach here, so the lossy
        // conversion is exact.
        return String::from_utf8_lossy(path).into_owned();
    }
    let mut quoted = String::from("\"");
    for &byte in path {
        match byte {
            b'"' => quoted.push_str("\\\""),
            b'\\' => quoted.push_str("\\\\"),
            0x20..=0x7e => quoted.push(byte as char),
            _ => quoted.push_str(&format!("\\{byte:03o}")),
        }
    }
    quoted.push('"');
    quoted
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    /// A git fixture: an initialized repository with identity config, an
    /// initial commit, and a `.gitignore`. Returns `None` (with a note)
    /// when git is missing, unless `EXT4KIT_REQUIRE_GIT` is set.
    fn repo_fixture(files: &[(&str, &str)]) -> Option<(tempfile::TempDir, Repo)> {
        if !git_available() {
            if std::env::var_os("EXT4KIT_REQUIRE_GIT").is_some() {
                panic!("EXT4KIT_REQUIRE_GIT is set but git was not found");
            }
            eprintln!("note: git not found; skipping git-backed unit tests");
            return None;
        }
        let directory = tempfile::tempdir().expect("a fresh temp directory");
        let repo = Repo::at(directory.path());
        for (name, content) in files {
            let path = directory.path().join(name);
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent).unwrap();
            }
            fs::write(path, content).unwrap();
        }
        let succeeds = |args: &[&str]| run(directory.path(), args).unwrap().status.success();
        assert!(succeeds(&["init", "--quiet"]));
        assert!(succeeds(&["config", "user.email", "t@example.com"]));
        assert!(succeeds(&["config", "user.name", "T"]));
        // The host may sign commits by default (gpg or ssh); the fixture
        // must not prompt for a passphrase. autocrlf is pinned so blob
        // contents are byte-deterministic regardless of the host config.
        assert!(succeeds(&["config", "commit.gpgsign", "false"]));
        assert!(succeeds(&["config", "core.autocrlf", "false"]));
        assert!(succeeds(&["add", "-A"]));
        assert!(succeeds(&["commit", "--quiet", "-m", "init"]));
        Some((directory, repo))
    }

    #[test]
    fn exact_len_passes_exact_content() {
        let mut out = Vec::new();
        let mismatch = {
            let mut framed = ExactLen::new(&mut out, 5);
            framed.write_all(b"hello").unwrap();
            framed.finish().unwrap();
            framed.mismatch()
        };
        assert_eq!(out, b"hello");
        assert!(!mismatch);
    }

    #[test]
    fn exact_len_drops_excess_bytes() {
        let mut out = Vec::new();
        let mismatch = {
            let mut framed = ExactLen::new(&mut out, 5);
            // The excess is a valid fast-import command sequence: it must
            // never reach the stream.
            framed.write_all(b"hell").unwrap();
            framed.write_all(b"o\nreset refs/heads/main\n").unwrap();
            framed.finish().unwrap();
            framed.mismatch()
        };
        assert_eq!(out, b"hello");
        assert!(mismatch);
    }

    #[test]
    fn exact_len_zero_pads_short_input() {
        let mut out = Vec::new();
        let mismatch = {
            let mut framed = ExactLen::new(&mut out, 8);
            framed.write_all(b"hi").unwrap();
            framed.finish().unwrap();
            framed.mismatch()
        };
        assert_eq!(out, b"hi\0\0\0\0\0\0");
        assert!(mismatch);
    }

    #[test]
    fn binary_detection_follows_gits_heuristic() {
        assert!(is_binary(b"a\0b"));
        assert!(is_binary(b"lone\rcr"));
        assert!(!is_binary(b"line\r\nline\r\n"));
        // One non-printable per 128 printables stays text.
        let mut mostly_text = vec![b'a'; 1280];
        mostly_text[500] = 0x01;
        assert!(!is_binary(&mostly_text));
        let mut too_dirty = vec![b'a'; 128];
        too_dirty[50] = 0x01;
        assert!(is_binary(&too_dirty));
    }

    #[test]
    fn reference_crlf_detection() {
        assert!(reference_has_crlf(b"a\r\nb\r\nc"));
        // Mixed line endings count as carrying CRLF.
        assert!(reference_has_crlf(b"a\r\nb\nc"));
        assert!(!reference_has_crlf(b"a\nb\nc"));
        // A binary reference does not suppress the conversion.
        assert!(!reference_has_crlf(b"a\r\n\0b\r\n"));
    }

    #[test]
    fn crlf_conversion_matches_the_streaming_form() {
        for content in [
            &b""[..],
            b"a",
            b"a\r\nb",
            b"a\rb\r\n",
            b"\r\n\r\n",
            b"ends with cr\r",
            b"cr at chunk end\r\nand more",
        ] {
            assert_eq!(crlf_to_git(content).len() as u64, {
                let stat = TextStat::gather(content);
                content.len() as u64 - stat.crlf_pairs()
            });
            let spool = Spool::Memory(content.to_vec());
            let mut streamed = Vec::new();
            spool
                .stream_crlf_to_git(|chunk| {
                    streamed.extend_from_slice(chunk);
                    Ok(())
                })
                .unwrap();
            assert_eq!(streamed, crlf_to_git(content), "{content:?}");
        }
    }

    #[test]
    fn crlf_pairs_straddling_chunk_boundaries_stay_inside_the_framing() {
        // A CR as the last byte of one 512 KiB chunk with its LF opening
        // the next: without withholding the CR, one byte too many would
        // leave the data <n> block and become fast-import commands. The
        // trailing lone CR exercises the end-of-content flush.
        const CHUNK: usize = 512 * 1024;
        let mut content = vec![b'a'; CHUNK - 1];
        content.push(b'\r');
        content.push(b'\n');
        content.extend_from_slice(b"mid\rdle\r\n");
        content.push(b'\r'); // a lone CR to withhold at the very end
        let expected = crlf_to_git(&content);
        let stat = TextStat::gather(&content);
        assert_eq!(
            expected.len() as u64,
            content.len() as u64 - stat.crlf_pairs()
        );
        let spool = Spool::Memory(content.clone());
        let mut streamed = Vec::new();
        spool
            .stream_crlf_to_git(|chunk| {
                streamed.extend_from_slice(chunk);
                Ok(())
            })
            .unwrap();
        assert_eq!(streamed.len(), expected.len());
        assert_eq!(streamed, expected);
    }

    #[test]
    fn conversion_routes_mirror_gits_gating() {
        let attrs = |text: AttrValue, eol: AttrValue, filter: AttrValue| PathAttrs {
            text,
            eol,
            filter,
            ..PathAttrs::default()
        };
        assert_eq!(
            conversion_route(
                &attrs(
                    AttrValue::True,
                    AttrValue::Unspecified,
                    AttrValue::Unspecified
                ),
                Autocrlf::False
            ),
            ConversionRoute::Normalize(EolRule::Explicit)
        );
        assert_eq!(
            conversion_route(
                &attrs(
                    AttrValue::Value("auto".into()),
                    AttrValue::Unspecified,
                    AttrValue::Unspecified
                ),
                Autocrlf::False
            ),
            ConversionRoute::Normalize(EolRule::Auto)
        );
        assert_eq!(
            conversion_route(
                &attrs(
                    AttrValue::Unspecified,
                    AttrValue::Unspecified,
                    AttrValue::Unspecified
                ),
                Autocrlf::True
            ),
            ConversionRoute::Normalize(EolRule::Auto)
        );
        assert_eq!(
            conversion_route(
                &attrs(
                    AttrValue::False,
                    AttrValue::Unspecified,
                    AttrValue::Unspecified
                ),
                Autocrlf::True
            ),
            ConversionRoute::Raw
        );
        assert_eq!(
            conversion_route(
                &attrs(
                    AttrValue::Unspecified,
                    AttrValue::Value("lf".into()),
                    AttrValue::Unspecified
                ),
                Autocrlf::False
            ),
            ConversionRoute::Normalize(EolRule::Explicit)
        );
        assert_eq!(
            conversion_route(
                &attrs(
                    AttrValue::Unspecified,
                    AttrValue::Unspecified,
                    AttrValue::Value("lfs".into())
                ),
                Autocrlf::False
            ),
            ConversionRoute::Filter
        );
        // An explicitly unset filter or ident selects nothing: paying a
        // hash-object spawn per file for `-filter` is pure cost.
        assert_eq!(
            conversion_route(
                &attrs(
                    AttrValue::Unspecified,
                    AttrValue::Unspecified,
                    AttrValue::False
                ),
                Autocrlf::False
            ),
            ConversionRoute::Raw
        );
        let mut with_ident = attrs(
            AttrValue::Unspecified,
            AttrValue::Unspecified,
            AttrValue::Unspecified,
        );
        with_ident.ident = AttrValue::False;
        assert_eq!(
            conversion_route(&with_ident, Autocrlf::False),
            ConversionRoute::Raw
        );
        with_ident.ident = AttrValue::True;
        assert_eq!(
            conversion_route(&with_ident, Autocrlf::False),
            ConversionRoute::Filter
        );
    }

    #[test]
    fn fast_import_paths_are_quoted_only_when_needed() {
        assert_eq!(fi_path("src/a.rs"), "src/a.rs");
        assert_eq!(fi_path("we \"ird"), "\"we \\\"ird\"");
        assert_eq!(fi_path("back\\slash"), "\"back\\\\slash\"");
        // The raw-byte form is the same function over bytes.
        assert_eq!(fi_path_bytes(b"src/a.rs"), "src/a.rs");
        assert_eq!(fi_path_bytes(b"we \"ird"), "\"we \\\"ird\"");
        assert_eq!(fi_path("pláin"), fi_path_bytes("pláin".as_bytes()));
    }

    #[test]
    fn raw_paths_octal_escape_every_non_printable_byte() {
        assert_eq!(fi_path_bytes(b"con\ntrol"), "\"con\\012trol\"");
        assert_eq!(fi_path_bytes(b"del\x7f"), "\"del\\177\"");
        assert_eq!(fi_path_bytes(&[0xff, b'a']), "\"\\377a\"");
        assert_eq!(fi_path_bytes(b"sp ace"), "sp ace");
    }

    #[test]
    fn modify_raw_round_trips_non_utf8_paths() {
        let Some((_directory, repo)) = repo_fixture(&[("a.txt", "a\n")]) else {
            return;
        };
        let parent = repo.head_commit().unwrap().unwrap();
        let mut fast_import = FastImport::spawn(&repo.root).unwrap();
        let blob = fast_import.blob(b"raw content\n").unwrap();
        fast_import
            .begin_commit(
                "refs/ext4kit/test/base",
                Some(&parent),
                "raw path",
                1_700_000_000,
            )
            .unwrap();
        // A non-UTF-8 byte: incompatible with every host filesystem, yet a
        // legal git path on every host (a control byte is not — git itself
        // refuses it on Windows).
        let path: &[u8] = b"bad\xffname.txt";
        fast_import
            .modify_raw("100644", &format!(":{blob}"), path)
            .unwrap();
        fast_import.finish().unwrap();
        // `ls-tree -z` hands the path back as raw bytes: the octal escapes
        // decode to exactly what was sent.
        let base = repo.rev_parse("refs/ext4kit/test/base").unwrap();
        let output = run(&repo.root, &["ls-tree", "-r", "-z", &base]).unwrap();
        assert!(
            output
                .stdout
                .windows(path.len())
                .any(|window| window == path),
            "the raw path must survive the round trip"
        );
    }

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
    fn the_spool_promotes_to_a_temp_file_above_the_limit() {
        let mut spool = Spool::create().unwrap();
        spool.write(&vec![0u8; 1024]).unwrap();
        assert!(matches!(spool, Spool::Memory(_)));
        assert_eq!(spool.len(), 1024);
        spool
            .write(&vec![0u8; SPOOL_MEMORY_LIMIT as usize])
            .unwrap();
        assert!(matches!(spool, Spool::Temp { .. }));
        // The temp file reads back exactly what was written, in chunks.
        let mut read_back = 0u64;
        spool
            .for_each_chunk(|chunk| {
                read_back += chunk.len() as u64;
                Ok(())
            })
            .unwrap();
        assert_eq!(read_back, SPOOL_MEMORY_LIMIT + 1024);
    }

    #[test]
    fn check_ignore_answers_one_record_per_query() {
        let Some((_directory, repo)) =
            repo_fixture(&[(".gitignore", "*.log\n!keep.log\nnode_modules/\n")])
        else {
            return;
        };
        let mut session = CheckIgnoreSession::new(&repo.root).unwrap();
        let queries = [
            ("src/a.log".to_owned(), false),
            ("src/keep.log".to_owned(), false),
            ("src/a.txt".to_owned(), false),
            ("node_modules".to_owned(), true),
            ("src".to_owned(), true),
        ];
        let answers = session.check(&queries).unwrap();
        let ignored: Vec<bool> = answers.into_iter().map(|answer| answer.unwrap()).collect();
        assert_eq!(ignored, vec![true, false, false, true, false]);
    }

    #[test]
    fn check_ignore_matches_pathspec_magic_names_literally() {
        let Some((_directory, repo)) = repo_fixture(&[(".gitignore", "*.log\n")]) else {
            return;
        };
        let mut session = CheckIgnoreSession::new(&repo.root).unwrap();
        // Without the ./ prefix these queries kill the process (invalid
        // pathspec magic) or are interpreted (:(top)); with it they are
        // matched as names.
        let queries = [
            (":(foo)x".to_owned(), false),
            (":(top)a.log".to_owned(), false),
        ];
        let answers = session.check(&queries).unwrap();
        let ignored: Vec<bool> = answers.into_iter().map(|answer| answer.unwrap()).collect();
        assert_eq!(ignored, vec![false, true]);
    }

    #[test]
    fn check_ignore_answers_nul_paths_locally_without_respawning() {
        let Some((_directory, repo)) = repo_fixture(&[(".gitignore", "*.log\n")]) else {
            return;
        };
        let mut session = CheckIgnoreSession::new(&repo.root).unwrap();
        let queries = vec![
            ("a.log".to_owned(), false),
            ("b\0c".to_owned(), false),
            ("d".to_owned(), false),
        ];
        let answers = session.check(&queries).unwrap();
        assert_eq!(answers.len(), 3);
        assert_eq!(answers[0], Ok(true));
        assert!(answers[1].is_err(), "{answers:?}");
        assert_eq!(answers[2], Ok(false));
        assert_eq!(session.respawns, 0, "a NUL path must not cost a respawn");
        // The session stays alive and synchronized around the local
        // answer.
        let answers = session.check(&[("e.log".to_owned(), false)]).unwrap();
        assert_eq!(answers[0], Ok(true));
    }

    #[test]
    fn check_ignore_survives_the_death_of_the_process() {
        let Some((_directory, repo)) = repo_fixture(&[(".gitignore", "*.log\n")]) else {
            return;
        };
        let mut session = CheckIgnoreSession::new(&repo.root).unwrap();
        let _ = session.child.kill();
        let _ = session.child.wait();
        let answers = session
            .check(&[("a.log".to_owned(), false), ("b.txt".to_owned(), false)])
            .unwrap();
        let ignored: Vec<bool> = answers.into_iter().map(|answer| answer.unwrap()).collect();
        assert_eq!(ignored, vec![true, false]);
        // And the respawned session keeps answering.
        let answers = session.check(&[("c.log".to_owned(), false)]).unwrap();
        let ignored: Vec<bool> = answers.into_iter().map(|answer| answer.unwrap()).collect();
        assert_eq!(ignored, vec![true]);
    }

    #[test]
    fn check_ignore_completes_batches_beyond_the_pipe_buffer() {
        // A .gitignore matching everything makes every answer maximal, so
        // 10 000 queries produce far more output than a pipe holds; the
        // writer thread keeps the session alive through it.
        let Some((_directory, repo)) = repo_fixture(&[(".gitignore", "generated/\n")]) else {
            return;
        };
        let mut session = CheckIgnoreSession::new(&repo.root).unwrap();
        let queries: Vec<(String, bool)> = (0..10_000)
            .map(|index| (format!("generated/file-{index}.bin"), false))
            .collect();
        let answers = session.check(&queries).unwrap();
        assert_eq!(answers.len(), 10_000);
        assert!(answers.iter().all(|answer| matches!(answer, Ok(true))));
    }

    #[test]
    fn check_attr_reports_the_five_attributes() {
        let Some((_directory, repo)) = repo_fixture(&[(
            ".gitattributes",
            "*.txt text\n*.dat filter=lfs\n*.sh eol=lf\n",
        )]) else {
            return;
        };
        let attributes = check_attr(
            &repo.root,
            &[
                "a.txt".to_owned(),
                "b.dat".to_owned(),
                "c.sh".to_owned(),
                "d.bin".to_owned(),
            ],
        )
        .unwrap();
        assert_eq!(attributes["a.txt"].text, AttrValue::True);
        assert_eq!(
            attributes["b.dat"].filter,
            AttrValue::Value("lfs".to_owned())
        );
        assert_eq!(attributes["c.sh"].eol, AttrValue::Value("lf".to_owned()));
        assert_eq!(attributes["d.bin"].text, AttrValue::Unspecified);
    }

    #[test]
    fn cat_file_answers_blobs_and_missing_objects() {
        let Some((_directory, repo)) = repo_fixture(&[("sub/a.txt", "content\r\n")]) else {
            return;
        };
        let mut batch = CatFileBatch::new(&repo.root).unwrap();
        let head = repo.head_commit().unwrap().unwrap();
        assert_eq!(
            batch.blob(&format!("{head}:sub/a.txt")).unwrap().as_deref(),
            Some(b"content\r\n".as_slice())
        );
        assert_eq!(batch.blob(&format!("{head}:missing.txt")).unwrap(), None);
        // A tree where the image may now have a CRLF file reads as "no
        // reference blob", not an abort; the session stays in sync.
        assert_eq!(batch.blob(&format!("{head}:sub")).unwrap(), None);
        assert_eq!(
            batch.blob(&format!("{head}:sub/a.txt")).unwrap().as_deref(),
            Some(b"content\r\n".as_slice())
        );
    }

    #[test]
    fn hash_object_runs_the_configured_clean_filter() {
        let Some((_directory, repo)) = repo_fixture(&[
            (".gitattributes", "*.dat filter=upper\n"),
            ("a.dat", "ignored at add time\n"),
        ]) else {
            return;
        };
        let succeeds = |args: &[&str]| run(&repo.root, args).unwrap().status.success();
        assert!(succeeds(&["config", "filter.upper.clean", "tr a-z A-Z"]));
        assert!(succeeds(&["config", "filter.upper.smudge", "cat"]));
        assert!(succeeds(&["config", "filter.upper.required", "true"]));
        let mut hash = HashObject::spawn(&repo.root, "a.dat", 8).unwrap();
        hash.writer().write_all(b"clean me").unwrap();
        let (sha, mismatch) = hash.finish().unwrap();
        assert!(!mismatch);
        let mut batch = CatFileBatch::new(&repo.root).unwrap();
        assert_eq!(
            batch.blob(&sha).unwrap().as_deref(),
            Some(b"CLEAN ME".as_slice())
        );
    }

    #[test]
    fn fast_import_writes_blobs_commits_and_deletions() {
        let Some((_directory, repo)) = repo_fixture(&[("old.txt", "old\n")]) else {
            return;
        };
        let parent = repo.head_commit().unwrap().unwrap();
        let mut fast_import = FastImport::spawn(&repo.root).unwrap();
        let blob = fast_import.blob(b"new content\n").unwrap();
        let symlink = fast_import.blob(b"target").unwrap();
        fast_import
            .begin_commit(
                "refs/ext4kit/test/base",
                Some(&parent),
                "test base",
                1_700_000_000,
            )
            .unwrap();
        fast_import.delete_path("old.txt").unwrap();
        fast_import
            .modify("100644", &format!(":{blob}"), "dir/new.txt")
            .unwrap();
        fast_import
            .modify("120000", &format!(":{symlink}"), "dir/link")
            .unwrap();
        fast_import.finish().unwrap();

        let base = repo.rev_parse("refs/ext4kit/test/base").unwrap();
        let entries = repo.ls_tree(&base).unwrap();
        // -r -t lists the tree entry as well as its contents.
        let paths: Vec<&str> = entries.iter().map(|e| e.path.as_str()).collect();
        assert_eq!(paths, vec!["dir", "dir/link", "dir/new.txt"]);
        assert_eq!(entries[0].mode, "040000");
        assert_eq!(entries[1].mode, "120000");
        assert_eq!(entries[2].mode, "100644");
        let mut batch = CatFileBatch::new(&repo.root).unwrap();
        assert_eq!(
            batch
                .blob(&format!("{base}:dir/new.txt"))
                .unwrap()
                .as_deref(),
            Some(b"new content\n".as_slice())
        );
    }

    #[test]
    fn fast_import_delete_all_clears_gitlinks_and_everything_else() {
        let Some((_directory, repo)) = repo_fixture(&[("kept.txt", "kept\n")]) else {
            return;
        };
        let parent = repo.head_commit().unwrap().unwrap();
        let mut fast_import = FastImport::spawn(&repo.root).unwrap();
        let blob = fast_import.blob(b"x").unwrap();
        fast_import
            .begin_commit("refs/ext4kit/test/base", Some(&parent), "root", 1)
            .unwrap();
        fast_import.delete_all().unwrap();
        fast_import
            .modify("100644", &format!(":{blob}"), "only.txt")
            .unwrap();
        fast_import.finish().unwrap();
        let base = repo.rev_parse("refs/ext4kit/test/base").unwrap();
        let entries = repo.ls_tree(&base).unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].path, "only.txt");
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
        assert_eq!(
            discover_repo(&directory.path().join("sub"))
                .unwrap()
                .map(|r| r.root),
            Some(repo.root.clone())
        );
        assert_eq!(
            discover_repo(&directory.path().join("sub/a.txt"))
                .unwrap()
                .map(|r| r.root),
            Some(repo.root.clone())
        );
        let outside = tempfile::tempdir().unwrap();
        assert_eq!(
            discover_repo(outside.path()).unwrap().map(|repo| repo.root),
            None
        );
    }
}
