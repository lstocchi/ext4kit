use super::git_command;
use anyhow::{Result, anyhow, bail};
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

/// How long the check-ignore reader waits for one record before declaring
/// the session dead. With `GIT_FLUSH=1` a record arrives immediately, so
/// this only fires when the flush contract breaks (or git hangs).
const IGNORE_RECORD_TIMEOUT: Duration = Duration::from_secs(30);

/// After this many check-ignore respawns in one run the snapshot aborts:
/// a systematic failure must not be silently turned into thousands of
/// carried-forward paths.
const MAX_IGNORE_RESPAWNS: u32 = 3;

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::host::git::test_support::repo_fixture;

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
}
