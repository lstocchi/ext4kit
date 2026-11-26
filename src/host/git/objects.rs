use super::{ExactLen, TextStat, git_command};
use anyhow::{Context, Result, anyhow, bail};
use std::io::{BufRead, BufReader, Read, Write};
use std::path::Path;
use std::process::{Child, Stdio};
use std::thread::{self, JoinHandle};

/// A lazy `git cat-file --batch` session for reference-blob queries. It
/// flushes after each object (verified), so a write-one-read-one loop
/// never deadlocks. Used against the fixed tree `base^` (HEAD at create)
/// so the answers do not drift with the current index.
pub struct CatFileBatch {
    child: Child,
    stdin: std::process::ChildStdin,
    stdout: BufReader<std::process::ChildStdout>,
    /// Set when a read fails after the header: the remaining body bytes
    /// would be parsed as the next header, so every later query must fail
    /// instead of answering garbage.
    broken: bool,
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
            broken: false,
        })
    }

    /// The object header for `name`: its kind and size, or `None` when the
    /// object is missing. Any failure means the session is out of step and
    /// is marked broken.
    fn header(&mut self, name: &str) -> Result<Option<(String, u64)>> {
        if self.broken {
            bail!("the git cat-file session is out of sync after an earlier read error");
        }
        let result = self.read_header(name);
        if result.is_err() {
            self.broken = true;
        }
        result
    }

    fn read_header(&mut self, name: &str) -> Result<Option<(String, u64)>> {
        if writeln!(self.stdin, "{name}")
            .and_then(|()| self.stdin.flush())
            .is_err()
        {
            bail!("cannot query git cat-file: the process stopped responding");
        }
        let mut header = String::new();
        let read = self
            .stdout
            .read_line(&mut header)
            .map_err(|error| anyhow!("git cat-file stopped responding: {error}"))?;
        if read == 0 {
            // EOF without an answer means the process died. Reporting it
            // as a missing object would silently convert every later file.
            bail!("git cat-file closed unexpectedly");
        }
        let header = header.trim().to_owned();
        if header.is_empty() {
            bail!("git cat-file answered unexpectedly: an empty line");
        }
        // `missing` and `ambiguous` are one-line answers that consume no
        // body, so the session stays in step and the path simply has no
        // reference blob.
        if header.ends_with(" missing") || header.ends_with(" ambiguous") {
            return Ok(None);
        }
        let mut parts = header.split_whitespace();
        let (Some(_oid), Some(kind), Some(size)) = (parts.next(), parts.next(), parts.next())
        else {
            bail!("git cat-file answered unexpectedly: {header:?}");
        };
        let size: u64 = size
            .parse()
            .with_context(|| format!("git cat-file reported a bad size: {header:?}"))?;
        Ok(Some((kind.to_owned(), size)))
    }

    /// Reads object-body bytes; a failure here leaves unconsumed body bytes
    /// behind, so the session is marked broken.
    fn read_body(&mut self, buf: &mut [u8]) -> Result<()> {
        if let Err(error) = self.stdout.read_exact(buf) {
            self.broken = true;
            return Err(error.into());
        }
        Ok(())
    }

    /// Consumes the newline git terminates every batch object with, so the
    /// next query starts at a header.
    fn consume_newline(&mut self) -> Result<()> {
        self.read_body(&mut [0u8; 1])
    }

    /// The content of the object named `name` (any extended SHA-1 syntax,
    /// e.g. `HEAD~1:src/a.rs`), or `None` when it does not exist. A path
    /// containing `:` cannot be expressed in that syntax and reads as
    /// missing, which only costs the eol exception for that path.
    pub fn blob(&mut self, name: &str) -> Result<Option<Vec<u8>>> {
        let Some((kind, size)) = self.header(name)? else {
            return Ok(None);
        };
        let size = match usize::try_from(size) {
            Ok(size) => size,
            Err(_) => {
                // The body stays unread, so the session cannot be reused.
                self.broken = true;
                bail!("git cat-file blob is too large: {name}");
            }
        };
        let mut content = vec![0u8; size];
        self.read_body(&mut content)?;
        self.consume_newline()?;
        if kind != "blob" {
            // A tree or gitlink where the image now has a file: there is
            // no reference blob, exactly like a missing path, and the
            // auto eol rule falls back to converting. The content was
            // consumed above to keep the session in sync.
            return Ok(None);
        }
        Ok(Some(content))
    }

    /// The text statistics of the object named `name`, or `None` when it is
    /// missing or not a blob. The content streams through [`TextStat`] in
    /// chunks, so a multi-gigabyte reference blob is never materialized
    /// just to answer the eol question.
    pub fn blob_stats(&mut self, name: &str) -> Result<Option<TextStat>> {
        let Some((kind, size)) = self.header(name)? else {
            return Ok(None);
        };
        let mut stats = TextStat::default();
        let mut buffer = [0u8; 8192];
        let mut remaining = size;
        while remaining > 0 {
            let take = (remaining.min(buffer.len() as u64)) as usize;
            self.read_body(&mut buffer[..take])?;
            stats.feed(&buffer[..take]);
            remaining -= take as u64;
        }
        self.consume_newline()?;
        if kind != "blob" {
            // Consumed above to keep the session in sync; a tree or
            // gitlink is no reference blob for the eol rule.
            return Ok(None);
        }
        Ok(Some(stats))
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
    /// stdout and stderr are drained on their own threads while stdin is
    /// written: a clean filter that prints more than a pipe buffer would
    /// otherwise block on its output while we block on its input.
    stdout: Option<JoinHandle<Vec<u8>>>,
    stderr: Option<JoinHandle<Vec<u8>>>,
    path: String,
}

/// How much of git's stdout or stderr one `hash-object` run keeps. The
/// clean filter is a user command running on agent-controlled bytes, so its
/// output is bounded; the rest is still read, so the pipe never fills.
const CAPTURE_LIMIT: usize = 64 * 1024;

/// Reads `pipe` to its end on a thread, keeping the first
/// [`CAPTURE_LIMIT`] bytes.
fn drain(mut pipe: impl Read + Send + 'static) -> JoinHandle<Vec<u8>> {
    thread::spawn(move || {
        let mut kept = Vec::new();
        let mut buffer = [0u8; 8192];
        loop {
            match pipe.read(&mut buffer) {
                Ok(0) | Err(_) => break,
                Ok(read) => {
                    let room = CAPTURE_LIMIT.saturating_sub(kept.len());
                    kept.extend_from_slice(&buffer[..read.min(room)]);
                }
            }
        }
        kept
    })
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
        let stdout = child.stdout.take().expect("stdout was piped");
        let stderr = child.stderr.take().expect("stderr was piped");
        Ok(Self {
            child,
            framed: Some(ExactLen::new(stdin, len)),
            stdout: Some(drain(stdout)),
            stderr: Some(drain(stderr)),
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
        let framing = framed.finish();
        let mismatch = framed.mismatch();
        drop(framed);
        let status = self.child.wait()?;
        let collect = |handle: Option<JoinHandle<Vec<u8>>>| {
            handle
                .expect("finish joins the drains once")
                .join()
                .unwrap_or_default()
        };
        let stdout = collect(self.stdout.take());
        let stderr = collect(self.stderr.take());
        if !status.success() {
            // Prefer git's stderr: when a required clean filter is missing
            // or crashes, the write side only sees a broken pipe, and the
            // real cause is here.
            bail!(
                "clean filter failed for {}: {}",
                self.path,
                String::from_utf8_lossy(&stderr).trim()
            )
        }
        framing?;
        let sha = String::from_utf8_lossy(&stdout).trim().to_owned();
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
        // The streaming stats answer the same eol question without
        // materializing the blob, and a missing path or a tree is `None`.
        let stats = batch
            .blob_stats(&format!("{head}:sub/a.txt"))
            .unwrap()
            .expect("stats for a blob");
        assert!(!stats.is_binary() && stats.has_crlf());
        assert!(
            batch
                .blob_stats(&format!("{head}:missing.txt"))
                .unwrap()
                .is_none()
        );
        assert!(batch.blob_stats(&format!("{head}:sub")).unwrap().is_none());
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
    fn a_chatty_clean_filter_does_not_deadlock() {
        let Some((_directory, repo)) = repo_fixture(&[(
            ".gitattributes",
            "*.dat filter=chatty
",
        )]) else {
            return;
        };
        // The filter fills its stderr pipe before reading stdin, while we
        // write more than a pipe buffer to its stdin: without draining,
        // each side blocks on the other.
        let succeeds = |args: &[&str]| run(&repo.root, args).unwrap().status.success();
        assert!(succeeds(&[
            "config",
            "filter.chatty.clean",
            "head -c 1000000 /dev/zero >&2; cat"
        ]));
        assert!(succeeds(&["config", "filter.chatty.required", "true"]));
        let root = repo.root.clone();
        let (done_tx, done_rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let content = vec![b'x'; 1 << 20];
            let mut hash = HashObject::spawn(&root, "big.dat", content.len() as u64).unwrap();
            hash.writer().write_all(&content).unwrap();
            let _ = done_tx.send(hash.finish().map(|(_, mismatch)| mismatch));
        });
        let outcome = done_rx
            .recv_timeout(std::time::Duration::from_secs(60))
            .expect("hash-object deadlocked on a chatty clean filter");
        assert!(!outcome.unwrap());
    }
}
