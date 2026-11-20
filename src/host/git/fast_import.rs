use super::git_command;
use anyhow::{Result, anyhow, bail};
use std::io::{self, BufWriter, Write};
use std::path::Path;
use std::process::{Child, Stdio};

/// One `git fast-import --quiet --done` run. Blobs stream in through the
/// [`Write`] impl (wrapped in [`ExactLen`](super::ExactLen) by the
/// caller), the commit is assembled from `deleteall`/`D` and `M`
/// commands, and [`FastImport::finish`] closes the stream with `done` —
/// so a run killed anywhere before that fails instead of committing a
/// partial tree. Without `--force`, fast-import only fast-forwards refs.
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
    /// [`ExactLen`](super::ExactLen)). Returns the mark for streamed blobs.
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
    use crate::host::git::CatFileBatch;
    use crate::host::git::run;
    use crate::host::git::test_support::repo_fixture;

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
}
