//! The tracking sidecar: `<image>.ext4kit-tracked`, the JSON file that
//! ties an image to the git refs holding its snapshots.
//!
//! The sidecar lives next to the image, deliberately: an in-image baseline
//! would be agent-writable. It is written only by `create`, atomically
//! (a temporary file, `sync_all`, then a rename), and its `image_uuid`
//! must match the filesystem `s_uuid`, which guards against wrapper
//! mix-ups — not against the agent, who can rewrite `s_uuid`.

use super::{Mapping, MappingKind};
use anyhow::{Result, anyhow, bail};
use serde_json::{Value, json};
use std::{
    fs,
    io::Write,
    path::{Path, PathBuf},
};

/// The only sidecar layout this version reads and writes.
const VERSION: u64 = 1;

/// The tracking record of one image.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Sidecar {
    pub tool: String,
    pub created_unix: u64,
    /// The filesystem `s_uuid` at create time, lowercase hex.
    pub image_uuid: String,
    pub label: String,
    pub include_ignored: bool,
    pub mappings: Vec<Mapping>,
    /// One record per repository, in mapping order.
    pub repos: Vec<RepoRecord>,
}

/// Per-repository facts the snapshot loop needs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RepoRecord {
    /// The repository root, normalized like the mapping host paths.
    pub root: String,
    /// The base commit written at create; lets snapshot restore
    /// `refs/ext4kit/<label>/base` if the user deleted the ref while the
    /// object still exists.
    pub base_commit: String,
    /// The working tree was dirty at create, which blocks `cherry-pick`
    /// until the user commits or stashes.
    pub dirty_at_create: bool,
}

impl Sidecar {
    /// The sidecar path for `image`.
    pub fn path_for(image: &Path) -> PathBuf {
        let mut name = image.as_os_str().to_os_string();
        name.push(".ext4kit-tracked");
        PathBuf::from(name)
    }

    /// Reads the sidecar for `image`. A missing file is a specific error
    /// naming the cases that produce it; a `version` above this tool's is
    /// a hard error; unknown fields are ignored.
    pub fn load(image: &Path) -> Result<Self> {
        let path = Self::path_for(image);
        let text = match fs::read_to_string(&path) {
            Ok(text) => text,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => bail!(
                "no tracked folders recorded for {}: images created without --add, \
                 by an older ext4kit, or with overlapping --add targets cannot be \
                 snapshotted",
                image.display()
            ),
            Err(error) => bail!("cannot read {path:?}: {error}"),
        };
        let value: Value = serde_json::from_str(&text)
            .map_err(|error| anyhow!("{path:?} is not a valid sidecar: {error}"))?;
        let version = value
            .get("version")
            .and_then(Value::as_u64)
            .ok_or_else(|| anyhow!("{path:?} has no version"))?;
        if version > VERSION {
            bail!(
                "{path:?} was written by a newer ext4kit (sidecar version {version}); \
                 upgrade ext4kit to snapshot it"
            );
        }
        Ok(Self::from_value(&value))
    }

    fn from_value(value: &Value) -> Self {
        let string = |key: &str| {
            value
                .get(key)
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned()
        };
        let mappings = value
            .get("mappings")
            .and_then(Value::as_array)
            .map(|entries| {
                entries
                    .iter()
                    .map(|entry| Mapping {
                        host: entry
                            .get("host")
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_owned(),
                        image: entry
                            .get("image")
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_owned(),
                        kind: match entry.get("kind").and_then(Value::as_str) {
                            Some("file") => MappingKind::File,
                            _ => MappingKind::Dir,
                        },
                        repo: entry.get("repo").and_then(Value::as_str).map(str::to_owned),
                        repo_relative: entry
                            .get("repo_relative")
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_owned(),
                    })
                    .collect()
            })
            .unwrap_or_default();
        let repos = value
            .get("repos")
            .and_then(Value::as_array)
            .map(|entries| {
                entries
                    .iter()
                    .map(|entry| RepoRecord {
                        root: entry
                            .get("root")
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_owned(),
                        base_commit: entry
                            .get("base_commit")
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_owned(),
                        dirty_at_create: entry
                            .get("dirty_at_create")
                            .and_then(Value::as_bool)
                            .unwrap_or(false),
                    })
                    .collect()
            })
            .unwrap_or_default();
        Self {
            tool: string("tool"),
            created_unix: value
                .get("created_unix")
                .and_then(Value::as_u64)
                .unwrap_or(0),
            image_uuid: string("image_uuid"),
            label: string("label"),
            include_ignored: value
                .get("include_ignored")
                .and_then(Value::as_bool)
                .unwrap_or(false),
            mappings,
            repos,
        }
    }

    fn to_value(&self) -> Value {
        json!({
            "version": VERSION,
            "tool": self.tool,
            "created_unix": self.created_unix,
            "image_uuid": self.image_uuid,
            "label": self.label,
            "include_ignored": self.include_ignored,
            "mappings": self
                .mappings
                .iter()
                .map(|mapping| json!({
                    "host": mapping.host,
                    "image": mapping.image,
                    "kind": match mapping.kind {
                        MappingKind::Dir => "dir",
                        MappingKind::File => "file",
                    },
                    "repo": mapping.repo,
                    "repo_relative": mapping.repo_relative,
                }))
                .collect::<Vec<_>>(),
            "repos": self
                .repos
                .iter()
                .map(|repo| json!({
                    "root": repo.root,
                    "base_commit": repo.base_commit,
                    "dirty_at_create": repo.dirty_at_create,
                }))
                .collect::<Vec<_>>(),
        })
    }

    /// Writes the sidecar atomically: a temporary file next to the image,
    /// flushed and synced, then renamed over the destination.
    pub fn write(&self, image: &Path) -> Result<()> {
        let path = Self::path_for(image);
        let mut temp = path.clone().into_os_string();
        temp.push(".tmp");
        let temp = PathBuf::from(temp);
        let body = serde_json::to_string_pretty(&self.to_value())?;
        {
            let mut file = fs::File::create(&temp)?;
            file.write_all(body.as_bytes())?;
            file.write_all(b"\n")?;
            file.sync_all()?;
        }
        fs::rename(&temp, &path)?;
        Ok(())
    }
}

/// Formats a filesystem UUID as lowercase hex, the sidecar's `image_uuid`.
pub fn uuid_hex(uuid: &[u8; 16]) -> String {
    uuid.iter().map(|byte| format!("{byte:02x}")).collect()
}

/// Sanitizes a tracking label to a single ref-safe component, lower-cased:
/// the loose-ref backend inherits NTFS and APFS case insensitivity, so
/// `Out` and `out` would collide. A Windows reserved device name is also
/// refused: it cannot become a loose-ref file there.
pub fn sanitize_label(label: &str) -> Result<String> {
    let lowered = label.to_lowercase();
    let bad =
        |c: char| c.is_control() || matches!(c, ' ' | '~' | '^' | ':' | '?' | '*' | '[' | '\\');
    if lowered.is_empty()
        || lowered.starts_with('.')
        || lowered.starts_with('-')
        || lowered.ends_with('.')
        || lowered.ends_with(".lock")
        || lowered == "@"
        || lowered.contains('/')
        || lowered.contains("..")
        || lowered.contains("@{")
        || lowered.chars().any(bad)
        || crate::tracked::names::is_windows_device_name(&lowered)
    {
        bail!(
            "label {label:?} is not usable as a git ref component: use letters, \
             digits, dots, dashes and underscores, and avoid reserved device \
             names like con or aux"
        );
    }
    Ok(lowered)
}

/// Derives the label of an image whose user never chose one: unlike an
/// explicit `--label`, a default is transformed rather than rejected —
/// the user asked for an image, not a label. Every character a ref
/// component cannot hold becomes a dash, the shapes git refuses are
/// repaired, and a reserved device name gains a trailing dash.
pub fn derived_label(raw: &str) -> String {
    let mut label: String = raw
        .to_lowercase()
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-') {
                c
            } else {
                '-'
            }
        })
        .collect();
    // Each repair can expose a shape another one fixes (`a.lock.lock`
    // loses one suffix at a time, `x.lock.` only ends in `.lock` once the
    // dot is gone), so they repeat until the label stops changing.
    loop {
        let before = label.len();
        while label.starts_with(['.', '-']) {
            label.remove(0);
        }
        while label.ends_with('.') {
            label.pop();
        }
        while label.contains("..") {
            label = label.replace("..", "-");
        }
        while label.ends_with(".lock") {
            label.truncate(label.len() - ".lock".len());
        }
        if label.len() == before {
            break;
        }
    }
    if crate::tracked::names::is_windows_device_name(&label) {
        // The device name is the part before the first dot, so the dash
        // must break exactly that part: `con.tar` becomes `con-.tar`,
        // and a bare `con` becomes `con-`. Appending instead would leave
        // the reserved name intact (`con.tar-`) and fail
        // `sanitize_label`.
        match label.find('.') {
            Some(dot) => label.insert(dot, '-'),
            None => label.push('-'),
        }
    }
    let label = if label.is_empty() {
        "image".to_owned()
    } else {
        label
    };
    debug_assert!(
        sanitize_label(&label).is_ok(),
        "a derived label must always be usable: {label}"
    );
    label
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Sidecar {
        Sidecar {
            tool: "ext4kit 0.1.0".to_owned(),
            created_unix: 1_700_000_000,
            image_uuid: "6f1c".to_owned(),
            label: "out".to_owned(),
            include_ignored: false,
            mappings: vec![Mapping {
                host: "C:/src".to_owned(),
                image: "/opt/src".to_owned(),
                kind: MappingKind::Dir,
                repo: Some("C:/src".to_owned()),
                repo_relative: String::new(),
            }],
            repos: vec![RepoRecord {
                root: "C:/src".to_owned(),
                base_commit: "a1b2".to_owned(),
                dirty_at_create: true,
            }],
        }
    }

    #[test]
    fn round_trips_through_json() {
        let sidecar = sample();
        let value = sidecar.to_value();
        assert_eq!(Sidecar::from_value(&value), sidecar);
    }

    #[test]
    fn unknown_fields_are_ignored() {
        let mut value = sample().to_value();
        value
            .as_object_mut()
            .unwrap()
            .insert("from_the_future".to_owned(), json!(true));
        assert_eq!(Sidecar::from_value(&value), sample());
    }

    #[test]
    fn writes_and_loads_atomically() {
        let directory =
            std::env::temp_dir().join(format!("ext4kit-sidecar-{}", std::process::id()));
        fs::create_dir_all(&directory).unwrap();
        let image = directory.join("out.img");
        let sidecar = sample();
        sidecar.write(&image).unwrap();
        assert_eq!(Sidecar::load(&image).unwrap(), sidecar);
        // The temporary file is renamed away, not left behind.
        let entries: Vec<_> = fs::read_dir(&directory)
            .unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        assert_eq!(entries, vec!["out.img.ext4kit-tracked".to_owned()]);
        fs::remove_dir_all(&directory).unwrap();
    }

    #[test]
    fn a_missing_sidecar_is_reported() {
        let error = Sidecar::load(Path::new("nowhere.img"))
            .unwrap_err()
            .to_string();
        assert!(error.contains("no tracked folders recorded"), "{error}");
    }

    #[test]
    fn labels_are_lower_cased_and_checked() {
        assert_eq!(sanitize_label("Out").unwrap(), "out");
        assert_eq!(sanitize_label("sandbox-2").unwrap(), "sandbox-2");
        for bad in [
            "",
            ".hidden",
            "-dash",
            "ends.",
            "x.lock",
            "@",
            "a/b",
            "a..b",
            "a@{b",
            "sp ace",
            "tilde~",
            "caret^",
            "colon:",
            "star*",
            "bracket[",
            "back\\slash",
            "con",
            "aux",
            "com1",
            "lpt9.txt",
        ] {
            assert!(sanitize_label(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn default_labels_are_transformed_not_rejected() {
        for (raw, expected) in [
            ("out", "out"),
            ("My Repo", "my-repo"),
            (".hidden", "hidden"),
            ("-dash", "dash"),
            ("x.lock", "x"),
            // Repairs that expose another forbidden shape repeat.
            ("a.lock.lock", "a"),
            ("x.lock.", "x"),
            (".lock", "lock"),
            ("x.lock.lock.", "x"),
            ("ends.", "ends"),
            ("a..b", "a-b"),
            ("con", "con-"),
            ("aux", "aux-"),
            ("COM1", "com1-"),
            // A device name with an extension: the dash breaks the part
            // before the first dot, where the reserved name lives.
            ("con.tar", "con-.tar"),
            ("aux.tar.gz", "aux-.tar.gz"),
            ("nul.x", "nul-.x"),
            ("COM1.img.bak", "com1-.img.bak"),
            ("lpt9.", "lpt9-"),
            ("..", "image"),
            ("", "image"),
            ("we!rd@stuff", "we-rd-stuff"),
        ] {
            let derived = derived_label(raw);
            assert_eq!(derived, expected, "{raw}");
            // Whatever the transformation produces must itself be a label
            // sanitize_label accepts: the default never fails a create.
            assert_eq!(sanitize_label(&derived).unwrap(), derived, "{raw}");
        }
    }

    #[test]
    fn uuids_format_as_lowercase_hex() {
        assert_eq!(
            uuid_hex(&[0x6f, 0x1c, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xff]),
            "6f1c00000000000000000000000000ff"
        );
    }
}
