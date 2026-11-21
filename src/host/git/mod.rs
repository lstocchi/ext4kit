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

mod attrs;
mod check_ignore;
mod exact_len;
mod fast_import;
mod objects;
mod repo;
mod text;

pub use attrs::{
    AttrValue, Autocrlf, ConversionRoute, EolRule, PathAttrs, check_attr, conversion_route,
};
pub use check_ignore::{CheckIgnoreSession, IgnoreAnswer};
pub use exact_len::ExactLen;
pub use fast_import::{FastImport, fi_path, fi_path_bytes};
pub use objects::{CatFileBatch, HashObject};
pub use repo::{ParentTree, Repo, TreeEntry, discover_repo};
pub use text::{SPOOL_MEMORY_LIMIT, Spool, TextStat, crlf_to_git, is_binary, reference_has_crlf};

use anyhow::{Context, Result, anyhow};
use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::OnceLock;

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
pub(super) fn git_command(repo: &Path, args: &[&str]) -> Command {
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
pub(super) fn run(repo: &Path, args: &[&str]) -> Result<std::process::Output> {
    run_env(repo, args, &[])
}

/// Runs a git command with extra environment variables, like [`run`].
pub(super) fn run_env(
    repo: &Path,
    args: &[&str],
    env: &[(&str, &str)],
) -> Result<std::process::Output> {
    let mut command = git_command(repo, args);
    command.envs(env.iter().copied());
    command
        .output()
        .map_err(|error| anyhow!("cannot run git (is it on PATH?): {error}"))
        .with_context(|| format!("git -C {} {}", repo.display(), args.join(" ")))
}

#[cfg(test)]
mod test_support {
    use super::{Repo, git_available, run};
    use std::fs;

    /// A git fixture: an initialized repository with identity config, an
    /// initial commit, and a `.gitignore`. Returns `None` (with a note)
    /// when git is missing, unless `EXT4KIT_REQUIRE_GIT` is set.
    pub(super) fn repo_fixture(files: &[(&str, &str)]) -> Option<(tempfile::TempDir, Repo)> {
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
}
