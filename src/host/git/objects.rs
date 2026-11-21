use super::{ExactLen, git_command};
use anyhow::{Context, Result, anyhow, bail};
use std::io::{BufRead, BufReader, Read, Write};
use std::path::Path;
use std::process::{Child, Stdio};

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::host::git::run;
    use crate::host::git::test_support::repo_fixture;

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
}
