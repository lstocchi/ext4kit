//! End-to-end tests for folder tracking at create time: the sidecar, the
//! create-time checks and the base refs. The `snapshot` command lands in
//! the next step, so re-create semantics are tested through its step-3
//! shape: unsnapshotted work can only be discarded, never silently.
//!
//! The tests follow the e2fsprogs skip pattern: skipped with a note when
//! git is absent, failing instead under `EXT4KIT_REQUIRE_GIT=1` in CI.

mod common;

use common::{Image, arg, ext4kit};
use serde_json::Value;
use std::{
    fs,
    io::Write,
    path::{Path, PathBuf},
    process::{Command, Output, Stdio},
    sync::OnceLock,
};

/// Whether git is available, once per test binary.
fn git_ready() -> bool {
    static READY: OnceLock<bool> = OnceLock::new();
    *READY.get_or_init(|| {
        let found = Command::new("git")
            .arg("--version")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .is_ok_and(|status| status.success());
        if !found {
            if std::env::var_os("EXT4KIT_REQUIRE_GIT").is_some() {
                panic!("EXT4KIT_REQUIRE_GIT is set but git was not found");
            }
            eprintln!("note: git not found; skipping tracked-folder tests");
        }
        found
    })
}

fn git(repo: &Path, args: &[&str]) -> Output {
    let mut command = Command::new("git");
    command.arg("-C").arg(repo).args(args);
    command.output().expect("git runs")
}

fn git_ok(repo: &Path, args: &[&str]) {
    let output = git(repo, args);
    assert!(
        output.status.success(),
        "git {:?} failed: {}\n{}",
        args,
        output.status,
        String::from_utf8_lossy(&output.stderr),
    );
}

fn git_stdout(repo: &Path, args: &[&str]) -> String {
    let output = git(repo, args);
    assert!(
        output.status.success(),
        "git {:?} failed: {}\n{}",
        args,
        output.status,
        String::from_utf8_lossy(&output.stderr),
    );
    String::from_utf8(output.stdout)
        .expect("git prints UTF-8")
        .trim()
        .to_owned()
}

/// A committed repository in a temp directory, plus room for images
/// beside it.
struct Fixture {
    directory: tempfile::TempDir,
    repo: PathBuf,
}

impl Fixture {
    fn image(&self, name: &str) -> PathBuf {
        self.directory.path().join(name)
    }
}

fn fixture(files: &[(&str, &str)]) -> Fixture {
    let directory = tempfile::tempdir().expect("a fresh temp directory");
    let repo = directory.path().join("repo");
    fs::create_dir_all(&repo).expect("the repo directory");
    for (name, content) in files {
        let path = repo.join(name);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).expect("the parent directory");
        }
        fs::write(path, content).expect("the fixture file");
    }
    // The host's global config may sign commits or normalize line
    // endings; pin both so fixtures are byte-deterministic.
    git_ok(&repo, &["init", "--quiet"]);
    git_ok(&repo, &["config", "user.email", "t@example.com"]);
    git_ok(&repo, &["config", "user.name", "T"]);
    git_ok(&repo, &["config", "commit.gpgsign", "false"]);
    git_ok(&repo, &["config", "core.autocrlf", "false"]);
    git_ok(&repo, &["add", "-A"]);
    git_ok(&repo, &["commit", "--quiet", "-m", "init"]);
    Fixture { directory, repo }
}

/// Runs `ext4kit create EXTRA... --add REPO:/opt IMAGE`.
fn create_tracked(repo: &Path, extra: &[&str], image: &Path) -> Output {
    let mut args: Vec<String> = vec!["create".to_owned(), "--track".to_owned()];
    args.extend(extra.iter().map(|flag| (*flag).to_owned()));
    args.push("--add".to_owned());
    args.push(format!("{}:/opt", arg(repo)));
    args.push(arg(image));
    ext4kit(&args.iter().map(String::as_str).collect::<Vec<_>>())
}

fn create_ok(repo: &Path, extra: &[&str], image: &Path) {
    let output = create_tracked(repo, extra, image);
    assert!(
        output.status.success(),
        "create failed: {}\nstdout:\n{}\nstderr:\n{}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
}

/// The sidecar path of `image`.
fn sidecar_path(image: &Path) -> PathBuf {
    let mut name = image.as_os_str().to_os_string();
    name.push(".ext4kit-tracked");
    PathBuf::from(name)
}

/// The parsed sidecar of `image`, which must exist.
fn sidecar(image: &Path) -> Value {
    let text = fs::read_to_string(sidecar_path(image)).expect("the sidecar exists");
    serde_json::from_str(&text).expect("the sidecar is valid JSON")
}

fn sidecar_exists(image: &Path) -> bool {
    sidecar_path(image).exists()
}

/// The tracked subtree of a label's base commit, as `mode path sha` lines.
fn base_entries(repo: &Path, label: &str) -> Vec<String> {
    git_stdout(
        repo,
        &["ls-tree", "-r", &format!("refs/ext4kit/{label}/base")],
    )
    .lines()
    .map(str::to_owned)
    .collect()
}

fn base_sha(repo: &Path, label: &str) -> String {
    git_stdout(repo, &["rev-parse", &format!("refs/ext4kit/{label}/base")])
}

fn diff_names(repo: &Path, from: &str, to: &str) -> Vec<String> {
    git_stdout(repo, &["diff", "--name-status", from, to])
        .lines()
        .map(str::to_owned)
        .collect()
}

fn file_hash(path: &Path) -> u64 {
    use std::hash::{DefaultHasher, Hasher};
    let mut hasher = DefaultHasher::new();
    hasher.write(&fs::read(path).expect("the image file"));
    hasher.finish()
}

#[test]
fn create_records_mappings_and_writes_a_matching_base() {
    if !git_ready() {
        return;
    }
    let fixture = fixture(&[("a.txt", "a\n"), ("sub/b.txt", "b\n")]);
    let image = fixture.image("out.img");
    create_ok(&fixture.repo, &[], &image);

    let record = sidecar(&image);
    assert_eq!(record["label"], "out");
    assert_eq!(record["include_ignored"], false);
    assert_eq!(record["mappings"][0]["image"], "/opt/repo");
    assert_eq!(record["mappings"][0]["kind"], "dir");
    assert_eq!(record["mappings"][0]["repo"], arg(&fixture.repo));
    assert_eq!(
        record["repos"][0]["base_commit"],
        base_sha(&fixture.repo, "out")
    );
    assert_eq!(record["repos"][0]["dirty_at_create"], false);

    // The mapping covers the repository root, so the base's tree is the
    // imported content of a clean worktree: identical to HEAD.
    let base = base_sha(&fixture.repo, "out");
    assert_eq!(
        diff_names(&fixture.repo, "HEAD", &base),
        Vec::<String>::new()
    );
    let entries = base_entries(&fixture.repo, "out");
    assert_eq!(entries.len(), 2); // a.txt and sub/b.txt (no tree entries)
}

#[test]
fn file_additions_are_tracked_as_single_files() {
    if !git_ready() {
        return;
    }
    let fixture = fixture(&[("docs/notes.txt", "notes\n"), ("other.txt", "x\n")]);
    let image = fixture.image("out.img");
    // The destination is the parent directory, like every --add: the
    // file's basename lands inside it.
    let notes = format!("{}:/etc", arg(&fixture.repo.join("docs/notes.txt")));
    ext4kit(&[
        "create",
        "--track",
        "--add",
        notes.as_str(),
        arg(&image).as_str(),
    ])
    .status
    .success()
    .then_some(())
    .expect("create succeeds");
    let record = sidecar(&image);
    assert_eq!(record["mappings"][0]["kind"], "file");
    assert_eq!(record["mappings"][0]["image"], "/etc/notes.txt");
    assert_eq!(record["mappings"][0]["repo_relative"], "docs/notes.txt");
    // The file mapping replaces exactly its own path in the base.
    let base = base_sha(&fixture.repo, "out");
    assert_eq!(
        diff_names(&fixture.repo, "HEAD", &base),
        Vec::<String>::new()
    );
}

#[test]
fn the_base_captures_uncommitted_work_and_warns() {
    if !git_ready() {
        return;
    }
    let fixture = fixture(&[("a.txt", "a\n")]);
    fs::write(fixture.repo.join("a.txt"), "changed\n").unwrap();
    fs::write(fixture.repo.join("new.txt"), "new\n").unwrap();
    let image = fixture.image("out.img");
    let output = create_tracked(&fixture.repo, &[], &image);
    assert!(output.status.success(), "{output:?}");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("uncommitted changes"), "{stderr}");

    let record = sidecar(&image);
    assert_eq!(record["repos"][0]["dirty_at_create"], true);
    // The dirty tree is the baseline: both changes are in the base.
    let base = base_sha(&fixture.repo, "out");
    let mut changed = diff_names(&fixture.repo, "HEAD", &base);
    changed.sort();
    assert_eq!(
        changed,
        vec!["A\tnew.txt".to_owned(), "M\ta.txt".to_owned()]
    );
}

#[test]
fn overlapping_targets_disable_tracking_but_not_the_create() {
    if !git_ready() {
        return;
    }
    let fixture = fixture(&[("app/keep.txt", "k\n")]);
    let elsewhere = fixture.directory.path().join("elsewhere");
    fs::create_dir_all(&elsewhere).unwrap();
    fs::write(elsewhere.join("x.txt"), "x\n").unwrap();

    // Image side: /opt/keep.txt's parent /opt and /opt/app nest through
    // the second mapping's target /opt/app/elsewhere.
    let image = fixture.image("nested.img");
    let first = format!("{}:/opt", arg(&fixture.repo.join("app")));
    let second = format!("{}:/opt/app", arg(&elsewhere));
    let output = ext4kit(&[
        "create",
        "--track",
        "--add",
        first.as_str(),
        "--add",
        second.as_str(),
        arg(&image).as_str(),
    ]);
    assert!(output.status.success(), "{output:?}");
    assert!(String::from_utf8_lossy(&output.stderr).contains("overlapping"));
    assert!(!sidecar_exists(&image));

    // Host side: one repository mapped twice at nested repo paths.
    let image = fixture.image("hostside.img");
    let output = create_tracked_against(&fixture.repo, &fixture.repo.join("app"), &image);
    assert!(output.status.success(), "{output:?}");
    assert!(String::from_utf8_lossy(&output.stderr).contains("overlapping"));
    assert!(!sidecar_exists(&image));
}

/// `create --add REPO:/opt --add OTHER:/srv` — the host-side overlap
/// shape (two sources from one repository).
fn create_tracked_against(repo: &Path, other: &Path, image: &Path) -> Output {
    let first = format!("{}:/opt", arg(repo));
    let second = format!("{}:/srv", arg(other));
    ext4kit(&[
        "create",
        "--track",
        "--add",
        first.as_str(),
        "--add",
        second.as_str(),
        arg(image).as_str(),
    ])
}

#[test]
fn mappings_outside_repositories_are_reported_not_tracked() {
    if !git_ready() {
        return;
    }
    let fixture = fixture(&[("a.txt", "a\n")]);
    let untracked = fixture.directory.path().join("plain");
    fs::create_dir_all(&untracked).unwrap();
    fs::write(untracked.join("x.txt"), "x\n").unwrap();
    let image = fixture.image("out.img");
    let add = format!("{}:/opt", arg(&untracked));
    let output = ext4kit(&[
        "create",
        "--track",
        "--add",
        add.as_str(),
        arg(&image).as_str(),
    ]);
    assert!(output.status.success(), "{output:?}");
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("not inside a git repository"),
        "{:?}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(!sidecar_exists(&image));
}

#[test]
fn label_collisions_fail_before_anything_is_written() {
    if !git_ready() {
        return;
    }
    let fixture = fixture(&[("a.txt", "a\n")]);
    let first = fixture.image("first.img");
    create_ok(&fixture.repo, &["--label", "out"], &first);
    let first_base = base_sha(&fixture.repo, "out");

    for label in ["out", "OUT"] {
        let second = fixture.image("second.img");
        let output = create_tracked(&fixture.repo, &["--label", label], &second);
        assert!(!output.status.success(), "{label}: {output:?}");
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(stderr.contains("label out is in use"), "{label}: {stderr}");
        // The refusal happens before the output is created.
        assert!(!second.exists());
    }
    // The existing image and its refs are untouched.
    assert_eq!(base_sha(&fixture.repo, "out"), first_base);
    assert!(sidecar_exists(&first));
}

#[test]
fn re_creating_requires_discarding_unsnapshotted_work() {
    if !git_ready() {
        return;
    }
    let fixture = fixture(&[("a.txt", "a\n")]);
    let image = fixture.image("out.img");
    create_ok(&fixture.repo, &[], &image);
    let old_base = base_sha(&fixture.repo, "out");
    let image_hash = file_hash(&image);

    // Without the override, re-creating refuses and changes nothing.
    let output = create_tracked(&fixture.repo, &[], &image);
    assert!(!output.status.success(), "{output:?}");
    assert!(String::from_utf8_lossy(&output.stderr).contains("--discard-unsnapshotted"));
    assert_eq!(file_hash(&image), image_hash);
    assert_eq!(base_sha(&fixture.repo, "out"), old_base);
    assert!(sidecar_exists(&image));

    // With it, the old refs move to previous/ and the new base is fresh.
    let output = create_tracked(&fixture.repo, &["--discard-unsnapshotted"], &image);
    assert!(output.status.success(), "{output:?}");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("previous snapshots of out kept at"),
        "{stdout}"
    );
    assert_ne!(base_sha(&fixture.repo, "out"), old_base);
    // The old base is still resolvable under previous/<ts>/.
    let previous = git_stdout(
        &fixture.repo,
        &[
            "for-each-ref",
            "--format=%(refname)",
            "refs/ext4kit/out/previous",
        ],
    );
    assert_eq!(previous.lines().count(), 1);
    let previous_base = previous.lines().next().unwrap().to_owned();
    assert!(previous_base.starts_with("refs/ext4kit/out/previous/"));
    assert!(previous_base.ends_with("/base"));

    // A second re-create keeps both generations.
    create_ok(&fixture.repo, &["--discard-unsnapshotted"], &image);
    let previous = git_stdout(
        &fixture.repo,
        &[
            "for-each-ref",
            "--format=%(refname)",
            "refs/ext4kit/out/previous",
        ],
    );
    assert_eq!(previous.lines().count(), 2);
}

#[test]
fn mounted_old_images_are_refused_until_replaced() {
    if !git_ready() {
        return;
    }
    let fixture = fixture(&[("a.txt", "a\n")]);
    let image = fixture.image("out.img");
    create_ok(&fixture.repo, &[], &image);
    let old_base = base_sha(&fixture.repo, "out");

    // Simulate the kernel's mount state: INCOMPAT_RECOVER set in the
    // primary superblock (1024 + 0x60), the state a journaled image
    // carries while the VM runs unfrozen. The refusal must leave the
    // image byte-identical from here on.
    {
        let mut bytes = fs::read(&image).unwrap();
        let offset = 1024 + 0x60;
        let incompat = u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap()) | 0x4;
        bytes[offset..offset + 4].copy_from_slice(&incompat.to_le_bytes());
        fs::write(&image, bytes).unwrap();
    }
    let image_hash = file_hash(&image);

    // Discarding alone does not get past the mounted refusal.
    let output = create_tracked(&fixture.repo, &["--discard-unsnapshotted"], &image);
    assert!(!output.status.success(), "{output:?}");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("mounted or was not cleanly unmounted"),
        "{stderr}"
    );
    assert_eq!(file_hash(&image), image_hash);
    assert_eq!(base_sha(&fixture.repo, "out"), old_base);

    // Neither does --replace-mounted alone (it never implies discarding).
    let output = create_tracked(&fixture.repo, &["--replace-mounted"], &image);
    assert!(!output.status.success(), "{output:?}");
    assert!(String::from_utf8_lossy(&output.stderr).contains("--discard-unsnapshotted"));

    // Both overrides together proceed.
    let output = create_tracked(
        &fixture.repo,
        &["--replace-mounted", "--discard-unsnapshotted"],
        &image,
    );
    assert!(output.status.success(), "{output:?}");
    assert_ne!(base_sha(&fixture.repo, "out"), old_base);
}

#[test]
fn artifact_directories_are_advised_but_baselined() {
    if !git_ready() {
        return;
    }
    let fixture = fixture(&[("a.txt", "a\n")]);
    fs::create_dir_all(fixture.repo.join("node_modules/pkg")).unwrap();
    fs::write(fixture.repo.join("node_modules/pkg/index.js"), "x\n").unwrap();
    let image = fixture.image("out.img");
    let output = create_tracked(&fixture.repo, &[], &image);
    assert!(output.status.success(), "{output:?}");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("node_modules/ is neither ignored nor tracked"),
        "{stderr}"
    );
    // Display only: the directory is imported and baselined like any
    // other content, because the repo's declared policy is the only one.
    let entries = base_entries(&fixture.repo, "out");
    assert!(
        entries
            .iter()
            .any(|entry| entry.ends_with("node_modules/pkg/index.js"))
    );
}

#[test]
fn ignored_files_follow_the_label_policy() {
    if !git_ready() {
        return;
    }
    let fixture = fixture(&[("a.txt", "a\n"), (".gitignore", "ignored.log\n")]);
    fs::write(fixture.repo.join("ignored.log"), "secret\n").unwrap();

    let image = fixture.image("plain.img");
    create_ok(&fixture.repo, &[], &image);
    assert!(
        !base_entries(&fixture.repo, "plain")
            .iter()
            .any(|entry| entry.ends_with("ignored.log")),
        "ignored files are excluded from the base"
    );
    assert_eq!(sidecar(&image)["include_ignored"], false);

    let image = fixture.image("included.img");
    create_ok(&fixture.repo, &["--include-ignored"], &image);
    assert_eq!(sidecar(&image)["label"], "included");
    assert_eq!(sidecar(&image)["include_ignored"], true);
    assert!(
        base_entries(&fixture.repo, "included")
            .iter()
            .any(|entry| entry.ends_with("ignored.log")),
        "--include-ignored puts ignored paths in the base"
    );
}

#[test]
fn force_added_files_under_ignored_directories_survive() {
    if !git_ready() {
        return;
    }
    let fixture = fixture(&[("a.txt", "a\n"), (".gitignore", "build/\n")]);
    fs::create_dir_all(fixture.repo.join("build")).unwrap();
    fs::write(fixture.repo.join("build/keep.txt"), "keep\n").unwrap();
    git_ok(&fixture.repo, &["add", "-f", "build/keep.txt"]);
    git_ok(&fixture.repo, &["commit", "--quiet", "-m", "force add"]);
    fs::write(fixture.repo.join("build/other.txt"), "other\n").unwrap();

    let image = fixture.image("out.img");
    create_ok(&fixture.repo, &[], &image);
    let entries = base_entries(&fixture.repo, "out");
    // The tracked descendant keeps the ignored directory from being
    // pruned; the untracked file under it stays excluded.
    assert!(
        entries
            .iter()
            .any(|entry| entry.ends_with("build/keep.txt"))
    );
    assert!(
        !entries
            .iter()
            .any(|entry| entry.ends_with("build/other.txt"))
    );
    let base = base_sha(&fixture.repo, "out");
    assert_eq!(
        diff_names(&fixture.repo, "HEAD", &base),
        Vec::<String>::new()
    );
}

#[test]
fn nested_repositories_are_boundaries_in_the_base() {
    if !git_ready() {
        return;
    }
    let fixture = fixture(&[("a.txt", "a\n")]);
    // A nested clone, not a submodule of the outer repository.
    let nested = fixture.repo.join("vendor/lib");
    fs::create_dir_all(&nested).unwrap();
    git_ok(&nested, &["init", "--quiet"]);
    fs::write(nested.join("secret.txt"), "s\n").unwrap();
    for args in [
        &["config", "user.email", "t@example.com"][..],
        &["config", "user.name", "T"][..],
        &["config", "commit.gpgsign", "false"][..],
        &["add", "-A"][..],
        &["commit", "--quiet", "-m", "nested"][..],
    ] {
        git_ok(&nested, args);
    }

    let image = fixture.image("out.img");
    let output = create_tracked(&fixture.repo, &[], &image);
    assert!(output.status.success(), "{output:?}");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("nested repository"), "{stderr}");
    let entries = base_entries(&fixture.repo, "out");
    assert!(
        !entries.iter().any(|entry| entry.contains("vendor/lib")),
        "nested repository files must not enter the outer commit: {entries:?}"
    );
}

#[test]
fn submodule_pointers_survive_a_root_mapping() {
    if !git_ready() {
        return;
    }
    let fixture = fixture(&[("a.txt", "a\n")]);
    // A real submodule, added from a sibling repository.
    let sub_repo = fixture.directory.path().join("lib");
    fs::create_dir_all(&sub_repo).unwrap();
    git_ok(&sub_repo, &["init", "--quiet"]);
    fs::write(sub_repo.join("lib.txt"), "l\n").unwrap();
    for args in [
        &["config", "user.email", "t@example.com"][..],
        &["config", "user.name", "T"][..],
        &["config", "commit.gpgsign", "false"][..],
        &["add", "-A"][..],
        &["commit", "--quiet", "-m", "lib"][..],
    ] {
        git_ok(&sub_repo, args);
    }
    git_ok(
        &fixture.repo,
        &[
            "-c",
            "protocol.file.allow=always",
            "submodule",
            "add",
            "../lib",
            "lib",
        ],
    );
    git_ok(&fixture.repo, &["commit", "--quiet", "-m", "submodule"]);
    let expected_gitlink = git_stdout(&fixture.repo, &["rev-parse", "HEAD:lib"]);

    let image = fixture.image("out.img");
    create_ok(&fixture.repo, &[], &image);
    // The root mapping uses deleteall, which clears gitlinks: the
    // pointer must be re-emitted unchanged.
    let entries = base_entries(&fixture.repo, "out");
    let gitlink = entries
        .iter()
        .find(|entry| entry.ends_with("\tlib"))
        .expect("the submodule entry");
    // `160000 commit <sha>\tlib`: the pointer is unchanged.
    assert!(gitlink.starts_with("160000"), "{gitlink}");
    let sha = gitlink.split_whitespace().nth(2).unwrap();
    assert_eq!(sha, expected_gitlink, "{gitlink}");
    assert!(entries.iter().any(|entry| entry.ends_with(".gitmodules")));
    // Nothing from inside the submodule leaked.
    assert!(!entries.iter().any(|entry| entry.ends_with("lib/lib.txt")));
}

#[test]
fn images_without_tracked_content_get_no_sidecar() {
    if !git_ready() {
        return;
    }
    let fixture = fixture(&[("a.txt", "a\n")]);
    let image = fixture.image("out.img");
    let plain = fixture.directory.path().join("plain");
    fs::create_dir_all(&plain).unwrap();
    fs::write(plain.join("x.txt"), "x\n").unwrap();
    let add = format!("{}:/opt", arg(&plain));
    ext4kit(&[
        "create",
        "--track",
        "--add",
        add.as_str(),
        arg(&image).as_str(),
    ])
    .status
    .success()
    .then_some(())
    .expect("create succeeds");
    assert!(!sidecar_exists(&image));
}

#[test]
fn a_foreign_ref_with_the_same_label_survives_a_re_create() {
    if !git_ready() {
        return;
    }
    // repoA is tracked by first.img under label "shared"; repoB holds a
    // "shared" base belonging to a different image. Re-creating first.img
    // while also tracking repoB must collide on the foreign ref — the
    // old-sidecar exemption covers only the repositories that sidecar
    // lists — and must leave every ref exactly where it was.
    let a = fixture(&[("a.txt", "a\n")]);
    let b = fixture(&[("b.txt", "b\n")]);
    let first = a.image("first.img");
    create_ok(&a.repo, &["--label", "shared"], &first);
    let other = b.image("other.img");
    create_ok(&b.repo, &["--label", "shared"], &other);
    let foreign_base = base_sha(&b.repo, "shared");
    let owned_base = base_sha(&a.repo, "shared");

    let first_add = format!("{}:/opt", arg(&a.repo));
    let second_add = format!("{}:/other", arg(&b.repo));
    let output = ext4kit(&[
        "create",
        "--track",
        "--label",
        "shared",
        "--discard-unsnapshotted",
        "--add",
        first_add.as_str(),
        "--add",
        second_add.as_str(),
        arg(&first).as_str(),
    ]);
    assert!(!output.status.success(), "{output:?}");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("label shared is in use"), "{stderr}");
    // Both refs are untouched: the refusal runs before anything moves.
    assert_eq!(base_sha(&b.repo, "shared"), foreign_base);
    assert_eq!(base_sha(&a.repo, "shared"), owned_base);
    let moved = git_stdout(
        &a.repo,
        &[
            "for-each-ref",
            "--format=%(refname)",
            "refs/ext4kit/shared/previous",
        ],
    );
    assert_eq!(moved, "", "nothing may move before the refusal");
    assert!(sidecar_exists(&first));
}

/// Runs `ext4kit` with an empty `PATH`, so no git can be found: the
/// old-image gates must hold even in a git-less environment.
fn ext4kit_without_git(args: &[String]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_ext4kit"))
        .args(args.iter().map(String::as_str))
        .env("PATH", "")
        .output()
        .expect("ext4kit runs without a PATH")
}

#[test]
fn a_missing_git_still_gates_the_old_image() {
    if !git_ready() {
        return;
    }
    let fixture = fixture(&[("a.txt", "a\n")]);
    let image = fixture.image("out.img");
    create_ok(&fixture.repo, &[], &image);
    let old_base = base_sha(&fixture.repo, "out");

    // The kernel's mount state: INCOMPAT_RECOVER set in the primary
    // superblock. Without git on PATH the mounted refusal must still
    // fire, not be skipped on the way to truncating the image.
    {
        let mut bytes = fs::read(&image).unwrap();
        let offset = 1024 + 0x60;
        let incompat = u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap()) | 0x4;
        bytes[offset..offset + 4].copy_from_slice(&incompat.to_le_bytes());
        fs::write(&image, bytes).unwrap();
    }
    let image_hash = file_hash(&image);

    let add = format!("{}:/opt", arg(&fixture.repo));
    let args: Vec<String> = vec![
        "create".to_owned(),
        "--track".to_owned(),
        "--discard-unsnapshotted".to_owned(),
        "--add".to_owned(),
        add,
        arg(&image),
    ];
    let output = ext4kit_without_git(&args);
    assert!(!output.status.success(), "{output:?}");
    let stderr = String::from_utf8_lossy(&output.stderr);
    // The refusal fires before the missing-git warning is even printed:
    // the image gates do not depend on git being present.
    assert!(
        stderr.contains("mounted or was not cleanly unmounted"),
        "{stderr}"
    );
    assert_eq!(file_hash(&image), image_hash);
    assert_eq!(base_sha(&fixture.repo, "out"), old_base);
}

#[test]
fn retiring_without_git_leaves_the_refs_in_place() {
    if !git_ready() {
        return;
    }
    let fixture = fixture(&[("a.txt", "a\n")]);
    let image = fixture.image("out.img");
    create_ok(&fixture.repo, &[], &image);
    let old_base = base_sha(&fixture.repo, "out");

    let add = format!("{}:/opt", arg(&fixture.repo));
    let args: Vec<String> = vec![
        "create".to_owned(),
        "--track".to_owned(),
        "--discard-unsnapshotted".to_owned(),
        "--add".to_owned(),
        add,
        arg(&image),
    ];
    let output = ext4kit_without_git(&args);
    assert!(output.status.success(), "{output:?}");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("the previous tracking refs"),
        "the leftover refs get a warning: {stderr}"
    );
    // The old sidecar is gone (it would describe a destroyed image), the
    // refs survive where they are, and the new image is untracked.
    assert!(!sidecar_exists(&image));
    assert_eq!(base_sha(&fixture.repo, "out"), old_base);
    let moved = git_stdout(
        &fixture.repo,
        &[
            "for-each-ref",
            "--format=%(refname)",
            "refs/ext4kit/out/previous",
        ],
    );
    assert_eq!(moved, "");
}

#[test]
fn a_from_failure_leaves_the_old_image_tracked() {
    if !git_ready() {
        return;
    }
    let fixture = fixture(&[("a.txt", "a\n")]);
    let image = fixture.image("out.img");
    create_ok(&fixture.repo, &[], &image);
    let old_base = base_sha(&fixture.repo, "out");

    // The registry reference fails to parse after every tracking check
    // has passed but before anything retires the old state.
    let add = format!("{}:/opt", arg(&fixture.repo));
    let output = ext4kit(&[
        "create",
        "--track",
        "--discard-unsnapshotted",
        "--add",
        add.as_str(),
        "--from",
        "docker://!!!",
        arg(&image).as_str(),
    ]);
    assert!(!output.status.success(), "{output:?}");
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("invalid OCI image reference"),
        "{:?}",
        String::from_utf8_lossy(&output.stderr)
    );
    // Retirement never ran: the old image stays tracked, refs unmoved.
    assert!(sidecar_exists(&image));
    assert_eq!(base_sha(&fixture.repo, "out"), old_base);
    let moved = git_stdout(
        &fixture.repo,
        &[
            "for-each-ref",
            "--format=%(refname)",
            "refs/ext4kit/out/previous",
        ],
    );
    assert_eq!(moved, "");
}

#[test]
fn failing_clean_filters_carry_forward_instead_of_aborting() {
    if !git_ready() {
        return;
    }
    // A filter attribute whose clean command is missing or crashes must
    // not cost the run its tracking: the paths are carried forward and
    // the failure is one grouped note, like a missing git-lfs.
    let fixture = fixture(&[
        (".gitattributes", "*.dat filter=upper\n"),
        ("keep.txt", "keep\n"),
        ("data.dat", "clean me\n"),
        ("more.dat", "and me\n"),
        ("sub/deep.dat", "me too\n"),
    ]);
    git_ok(&fixture.repo, &["config", "filter.upper.clean", "false"]);
    git_ok(&fixture.repo, &["config", "filter.upper.required", "true"]);

    let image = fixture.image("out.img");
    let output = create_tracked(&fixture.repo, &[], &image);
    assert!(output.status.success(), "{output:?}");
    let stderr = String::from_utf8_lossy(&output.stderr);
    // Every git error names its path, so grouping must key on the driver:
    // three failures are one line, not three.
    let notes: Vec<&str> = stderr
        .lines()
        .filter(|line| line.contains("failed their clean filter"))
        .collect();
    assert_eq!(notes.len(), 1, "{stderr}");
    assert!(
        notes[0].contains("3 paths failed their clean filter (filter=upper)"),
        "{stderr}"
    );
    for path in ["data.dat", "more.dat", "sub/deep.dat"] {
        assert!(notes[0].contains(path), "{path}: {stderr}");
    }
    // The base keeps the parent's data.dat (carried forward) and every
    // other path: identical to HEAD.
    let base = base_sha(&fixture.repo, "out");
    assert_eq!(
        diff_names(&fixture.repo, "HEAD", &base),
        Vec::<String>::new()
    );
}

#[test]
fn awkward_image_names_derive_a_label_instead_of_failing() {
    if !git_ready() {
        return;
    }
    let fixture = fixture(&[("a.txt", "a\n")]);

    // A space in the image name is transformed, not rejected: the user
    // asked for an image, not a label.
    let image = fixture.image("My Repo.img");
    create_ok(&fixture.repo, &[], &image);
    assert_eq!(sidecar(&image)["label"], "my-repo");
    assert!(!base_sha(&fixture.repo, "my-repo").is_empty());

    let image = fixture.image(".hidden.img");
    create_ok(&fixture.repo, &[], &image);
    assert_eq!(sidecar(&image)["label"], "hidden");

    // A device name with an extension derives a usable label: the dash
    // breaks the reserved part before the first dot.
    let image = fixture.image("con.tar.img");
    create_ok(&fixture.repo, &[], &image);
    assert_eq!(sidecar(&image)["label"], "con-.tar");
    assert!(!base_sha(&fixture.repo, "con-.tar").is_empty());

    // An explicitly chosen label is still validated, reserved device
    // names included.
    let output = create_tracked(&fixture.repo, &["--label", "con"], &fixture.image("x.img"));
    assert!(!output.status.success(), "{output:?}");
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("not usable as a git ref component"),
        "{:?}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn dash_leading_subtrees_still_report_dirty_state() {
    if !git_ready() {
        return;
    }
    // A repo-relative path starting with a dash: the dirty check must
    // treat it as a path, not an option, or the whole create fails.
    let fixture = fixture(&[("-dist/a.txt", "a\n")]);
    fs::write(fixture.repo.join("-dist/a.txt"), "changed\n").unwrap();
    let image = fixture.image("out.img");
    let add = format!("{}:/opt/d", arg(&fixture.repo.join("-dist")));
    let output = ext4kit(&[
        "create",
        "--track",
        "--add",
        add.as_str(),
        arg(&image).as_str(),
    ]);
    assert!(output.status.success(), "{output:?}");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("uncommitted changes"), "{stderr}");
    assert_eq!(sidecar(&image)["repos"][0]["dirty_at_create"], true);
}

#[test]
fn tracking_is_opt_in() {
    if !git_ready() {
        return;
    }
    // A repo-sourced create without --track writes an image and nothing
    // else: no sidecar, no refs.
    let fixture = fixture(&[("a.txt", "a\n")]);
    let image = fixture.image("out.img");
    let add = format!("{}:/opt", arg(&fixture.repo));
    let output = ext4kit(&["create", "--add", add.as_str(), arg(&image).as_str()]);
    assert!(output.status.success(), "{output:?}");
    assert!(!sidecar_exists(&image));
    let refs = git_stdout(&fixture.repo, &["for-each-ref", "refs/ext4kit/"]);
    assert_eq!(refs, "");

    // --label and --include-ignored are tracking switches: without
    // --track they are usage errors.
    for flags in [&["--label", "x"][..], &["--include-ignored"][..]] {
        let mut args: Vec<String> = vec!["create".to_owned()];
        args.extend(flags.iter().map(|flag| (*flag).to_owned()));
        args.push("--add".to_owned());
        args.push(add.clone());
        args.push(arg(&image));
        let output = ext4kit(&args.iter().map(String::as_str).collect::<Vec<_>>());
        assert!(!output.status.success(), "{flags:?}: {output:?}");
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(stderr.contains("--track"), "{flags:?}: {stderr}");
    }
}

#[test]
fn re_creating_without_track_still_gates_and_retires() {
    if !git_ready() {
        return;
    }
    // The old-image gates and the retirement do not need --track: the
    // refusal fires without it, and --discard-unsnapshotted retires the
    // old refs while the new image stays untracked.
    let fixture = fixture(&[("a.txt", "a\n")]);
    let image = fixture.image("out.img");
    create_ok(&fixture.repo, &[], &image);
    let old_base = base_sha(&fixture.repo, "out");

    let add = format!("{}:/opt", arg(&fixture.repo));
    let output = ext4kit(&["create", "--add", add.as_str(), arg(&image).as_str()]);
    assert!(!output.status.success(), "{output:?}");
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("--discard-unsnapshotted"),
        "{output:?}"
    );
    assert_eq!(base_sha(&fixture.repo, "out"), old_base);

    let output = ext4kit(&[
        "create",
        "--discard-unsnapshotted",
        "--add",
        add.as_str(),
        arg(&image).as_str(),
    ]);
    assert!(output.status.success(), "{output:?}");
    let moved = git_stdout(
        &fixture.repo,
        &[
            "for-each-ref",
            "--format=%(refname)",
            "refs/ext4kit/out/previous",
        ],
    );
    assert_eq!(moved.lines().count(), 1, "the old refs retired");
    assert!(!sidecar_exists(&image), "the new image is untracked");
}

#[test]
fn a_foreign_sidecar_is_reported_and_owns_nothing() {
    if !git_ready() {
        return;
    }
    let fixture = fixture(&[("a.txt", "a\n")]);
    let first = fixture.image("first.img");
    create_ok(&fixture.repo, &["--label", "out"], &first);
    let owned_base = base_sha(&fixture.repo, "out");

    // An untracked second image, so no label of its own exists anywhere.
    let plain = fixture.directory.path().join("plain");
    fs::create_dir_all(&plain).unwrap();
    fs::write(plain.join("x.txt"), "x\n").unwrap();
    let second = fixture.image("second.img");
    let add = format!("{}:/opt", arg(&plain));
    ext4kit(&["create", "--add", add.as_str(), arg(&second).as_str()])
        .status
        .success()
        .then_some(())
        .expect("create succeeds");
    assert!(!sidecar_exists(&second));

    // The first image's sidecar lands at the second image's sidecar path:
    // a wrapper mix-up. Re-creating under the foreign label collides — a
    // foreign sidecar exempts nothing — and refuses before writing.
    fs::copy(sidecar_path(&first), sidecar_path(&second)).unwrap();
    let output = create_tracked(&fixture.repo, &["--label", "out"], &second);
    assert!(!output.status.success(), "{output:?}");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("label out is in use"), "{stderr}");
    assert!(stderr.contains("its refs are left alone"), "{stderr}");
    assert_eq!(base_sha(&fixture.repo, "out"), owned_base);
    assert!(sidecar_exists(&second));

    // Under its own label the re-create needs no --discard-unsnapshotted
    // (the foreign sidecar's work is not this image's), removes the stale
    // sidecar without touching the first image's refs, and tracks afresh.
    let output = create_tracked(&fixture.repo, &[], &second);
    assert!(output.status.success(), "{output:?}");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("its refs are left alone"), "{stderr}");
    assert_eq!(base_sha(&fixture.repo, "out"), owned_base);
    let moved = git_stdout(
        &fixture.repo,
        &[
            "for-each-ref",
            "--format=%(refname)",
            "refs/ext4kit/out/previous",
        ],
    );
    assert_eq!(moved, "", "a foreign sidecar's refs never move");
    assert_eq!(sidecar(&second)["label"], "second");
    assert!(!base_sha(&fixture.repo, "second").is_empty());
}

/// Writes `content` as a blob in `repo` and returns its SHA.
fn hash_blob(repo: &Path, content: &[u8]) -> String {
    let mut child = Command::new("git")
        .arg("-C")
        .arg(repo)
        .args(["hash-object", "-w", "--stdin"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("hash-object runs");
    child.stdin.take().unwrap().write_all(content).unwrap();
    let output = child.wait_with_output().unwrap();
    assert!(output.status.success());
    String::from_utf8(output.stdout).unwrap().trim().to_owned()
}

/// Stages `raw` (any bytes) in `repo`'s index with `mode` and `sha`. Raw
/// bytes cannot cross a Windows command line, so the record goes through
/// `update-index --index-info` on stdin.
fn index_raw_path(repo: &Path, mode: &str, sha: &str, raw: &[u8]) {
    let mut record = format!("{mode} {sha}\t").into_bytes();
    record.extend_from_slice(raw);
    record.push(0);
    let mut child = Command::new("git")
        .arg("-C")
        .arg(repo)
        .args(["update-index", "-z", "--index-info"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("update-index runs");
    child.stdin.take().unwrap().write_all(&record).unwrap();
    let output = child.wait_with_output().unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn host_incompatible_head_paths_are_kept_in_the_base() {
    if !git_ready() {
        return;
    }
    let fixture = fixture(&[("a.txt", "a\n")]);
    // A name with a non-UTF-8 byte is incompatible with every host
    // filesystem, so the file can exist only in git. Before the fix the
    // base's D deleted it silently.
    let blob = hash_blob(&fixture.repo, b"weird content\n");
    let weird: &[u8] = b"bad\xffname.txt";
    index_raw_path(&fixture.repo, "100644", &blob, weird);
    git_ok(&fixture.repo, &["commit", "--quiet", "-m", "weird"]);

    let image = fixture.image("out.img");
    create_ok(&fixture.repo, &[], &image);
    // The base keeps the path with its blob unchanged: identical to HEAD.
    let base = base_sha(&fixture.repo, "out");
    assert_eq!(
        diff_names(&fixture.repo, "HEAD", &base),
        Vec::<String>::new()
    );
    // The raw bytes are in the base tree, not a lossy U+FFFD name.
    let listed = git(&fixture.repo, &["ls-tree", "-r", "-z", &base]);
    assert!(
        listed
            .stdout
            .windows(weird.len())
            .any(|window| window == weird),
        "the raw path must survive: {:?}",
        listed.stdout
    );
    assert!(
        String::from_utf8_lossy(&listed.stdout).contains(&blob),
        "the blob must be unchanged: {:?}",
        listed.stdout
    );
}

#[test]
fn host_incompatible_head_directories_are_kept_in_the_base() {
    if !git_ready() {
        return;
    }
    let fixture = fixture(&[("a.txt", "a\n")]);
    // The incompatible component is a directory: the probe that asks git
    // whether it can re-create the path must not reject it just because
    // an index never holds tree entries.
    let blob = hash_blob(&fixture.repo, b"inside\n");
    index_raw_path(&fixture.repo, "100644", &blob, b"bad\xffdir/one.txt");
    index_raw_path(&fixture.repo, "100644", &blob, b"bad\xffdir/sub/two.txt");
    git_ok(&fixture.repo, &["commit", "--quiet", "-m", "weird dir"]);

    let image = fixture.image("out.img");
    let output = create_tracked(&fixture.repo, &[], &image);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let printed = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        !printed.contains("cannot be re-created"),
        "the directory must not be reported as deleted: {printed}"
    );
    let base = base_sha(&fixture.repo, "out");
    assert_eq!(
        diff_names(&fixture.repo, "HEAD", &base),
        Vec::<String>::new()
    );
    let listed = git(
        &fixture.repo,
        &["ls-tree", "-r", "-z", "--name-only", &base],
    );
    for raw in [&b"bad\xffdir/one.txt"[..], b"bad\xffdir/sub/two.txt"] {
        assert!(
            listed.stdout.split(|&b| b == 0).any(|path| path == raw),
            "{raw:?} must survive: {:?}",
            listed.stdout
        );
    }
}

#[test]
fn host_incompatible_gitlinks_keep_their_raw_name() {
    if !git_ready() {
        return;
    }
    let fixture = fixture(&[("a.txt", "a\n")]);
    // A gitlink's target commit need not exist in the outer repository.
    let pointer = "1111111111111111111111111111111111111111";
    let raw: &[u8] = b"bad\xffsub";
    index_raw_path(&fixture.repo, "160000", pointer, raw);
    git_ok(
        &fixture.repo,
        &["commit", "--quiet", "-m", "weird submodule"],
    );

    let image = fixture.image("out.img");
    create_ok(&fixture.repo, &[], &image);
    let base = base_sha(&fixture.repo, "out");
    assert_eq!(
        diff_names(&fixture.repo, "HEAD", &base),
        Vec::<String>::new()
    );
    let listed = git(&fixture.repo, &["ls-tree", "-r", "-z", &base]);
    let expected = [format!("160000 commit {pointer}\t").as_bytes(), raw].concat();
    assert!(
        listed
            .stdout
            .split(|&b| b == 0)
            .any(|line| line == expected),
        "the gitlink must keep its raw name: {:?}",
        listed.stdout
    );
}

#[test]
fn created_tracked_images_still_pass_e2fsck() {
    if !git_ready() {
        return;
    }
    let fixture = fixture(&[("a.txt", "a\n"), ("sub/b.bin", "b\n")]);
    let image = fixture.image("out.img");
    create_ok(&fixture.repo, &[], &image);
    // The tracking steps must not change what e2fsck sees.
    Image::check(&image);
}

#[test]
fn a_failed_sidecar_write_removes_the_base_refs_in_every_repository() {
    if !git_ready() {
        return;
    }
    // Two repositories, so the cleanup has to address each one with its
    // own ref and SHA: one update-ref transaction naming the same ref
    // twice (or the other repository's SHA) fails as a whole.
    let first = fixture(&[("a.txt", "a\n")]);
    let second = fixture(&[("b.txt", "b\n")]);
    let image = first.image("out.img");
    // A directory where the sidecar's temporary file goes makes the
    // atomic write fail after both base refs exist.
    let mut blocker = image.as_os_str().to_os_string();
    blocker.push(".ext4kit-tracked.tmp");
    fs::create_dir_all(PathBuf::from(blocker)).unwrap();

    let output = create_tracked_against(&first.repo, &second.repo, &image);
    assert!(output.status.success(), "{output:?}");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("folder tracking is unavailable"),
        "{stderr}"
    );
    assert!(!sidecar_exists(&image));
    for repo in [&first.repo, &second.repo] {
        let refs = git_stdout(repo, &["for-each-ref", "refs/ext4kit/"]);
        assert!(refs.is_empty(), "{}: {refs}", repo.display());
    }
}
