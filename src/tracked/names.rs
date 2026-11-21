//! Classification of directory-entry names read from the image.
//!
//! Every name is agent-controlled bytes, so nothing becomes a git path — and
//! nothing is printed — before it passes through [`validate_component`].
//! The classes mirror the snapshot plan:
//!
//! 1. [`NameClass::Corrupted`]: names the kernel cannot produce (empty, `/`,
//!    NUL, `.`, `..`).
//! 2. [`NameClass::NeverInTree`]: names git's `verify_path` refuses, matched
//!    the way `core.protectNTFS` / `core.protectHFS` match them
//!    case-insensitively, ignoring trailing dots and spaces (NTFS) and HFS
//!    ignorable code points: `.git` and its aliases, `git~1`, and
//!    `.gitmodules` as a symlink.
//! 3. [`NameClass::IncompatibleHost`]: names that cannot exist on the host
//!    OS — not valid UTF-8, control characters, or (on Windows) reserved
//!    device names, illegal characters, trailing dots and spaces. Excluded
//!    and reported, never auto-renamed.
//! 4. [`NameClass::PortabilityWarning`]: on non-Windows hosts, names that
//!    would fall under rule 3 on Windows. Kept in the tree, warned about.

/// How a directory-entry name is treated.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NameClass {
    /// The name becomes a git path as-is.
    Keep,
    /// The name becomes a git path but cannot be checked out on Windows.
    PortabilityWarning,
    /// Never becomes a git path: `.git` and its aliases.
    NeverInTree,
    /// The kernel cannot produce the name; the image is corrupt.
    Corrupted,
    /// Excluded and reported: the name cannot exist on the host OS.
    IncompatibleHost,
}

/// The result of classifying one name.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NameVerdict {
    pub class: NameClass,
    /// Why the name was not kept, escaped for display; `None` for
    /// [`NameClass::Keep`] and [`NameClass::PortabilityWarning`].
    pub reason: Option<String>,
}

impl NameVerdict {
    fn kept(reason: Option<&str>) -> Self {
        Self {
            class: NameClass::Keep,
            reason: reason.map(str::to_owned),
        }
    }

    fn new(class: NameClass, reason: String) -> Self {
        Self {
            class,
            reason: Some(reason),
        }
    }
}

/// Classifies one directory-entry name. `is_symlink` selects the extra
/// `.gitmodules` rule from git's `verify_path`, which refuses a symlink of
/// that name (an attacker could otherwise redirect the submodule
/// configuration the host will read).
pub fn validate_component(name: &[u8], is_symlink: bool) -> NameVerdict {
    // Rule 1: the kernel cannot produce these.
    if name.is_empty() {
        return NameVerdict::new(NameClass::Corrupted, "an empty name".to_owned());
    }
    if name.contains(&b'/') {
        return NameVerdict::new(
            NameClass::Corrupted,
            format!("a slash in the name ({})", escape_name(name)),
        );
    }
    if name.contains(&0) {
        return NameVerdict::new(
            NameClass::Corrupted,
            format!("a NUL in the name ({})", escape_name(name)),
        );
    }
    if name == b"." || name == b".." {
        return NameVerdict::new(
            NameClass::Corrupted,
            format!("a dot entry ({})", escape_name(name)),
        );
    }

    // Rule 2: git's verify_path refusals.
    if is_dotgit(name) {
        return NameVerdict::new(
            NameClass::NeverInTree,
            format!("a .git alias ({})", escape_name(name)),
        );
    }
    if is_symlink && is_dotgitmodules(name) {
        return NameVerdict::new(
            NameClass::NeverInTree,
            format!(".gitmodules as a symlink ({})", escape_name(name)),
        );
    }

    // Rule 3a: valid UTF-8 without control characters, on every host. C0
    // controls are the bytes below 0x20 plus DEL; git's verify_path checks
    // the same set with iscntrl().
    let Ok(text) = std::str::from_utf8(name) else {
        return NameVerdict::new(
            NameClass::IncompatibleHost,
            format!("a name that is not valid UTF-8 ({})", escape_name(name)),
        );
    };
    if text.bytes().any(|byte| byte < 0x20 || byte == 0x7f) {
        return NameVerdict::new(
            NameClass::IncompatibleHost,
            format!("a control character in the name ({})", escape_name(name)),
        );
    }

    // Rules 3b and 4: host-OS legality, portability warning elsewhere.
    match windows_incompatible(text) {
        Some(reason) if cfg!(windows) => {
            NameVerdict::new(NameClass::IncompatibleHost, reason.to_owned())
        }
        Some(_) => NameVerdict {
            class: NameClass::PortabilityWarning,
            reason: None,
        },
        None => NameVerdict::kept(None),
    }
}

/// Formats a raw name for display: printable ASCII as-is, everything else
/// as `\xNN` escapes, so hostile bytes never reach a terminal unescaped.
pub fn escape_name(name: &[u8]) -> String {
    let mut out = String::new();
    for &byte in name {
        if (0x20..0x7f).contains(&byte) {
            out.push(byte as char);
        } else {
            out.push_str(&format!("\\x{byte:02x}"));
        }
    }
    out
}

/// Escapes the control characters of a walked path for display, so a
/// hostile name cannot inject terminal escape sequences into notes and
/// reports. Printable characters — including Unicode — are kept as-is.
pub fn escape_display(path: &str) -> String {
    path.chars()
        .map(|c| {
            if c.is_control() {
                format!("\\x{:02x}", c as u32)
            } else {
                c.to_string()
            }
        })
        .collect()
}

/// Whether `name` is one of git's aliases for `.git`: the NTFS rules
/// (`.git` case-insensitively with trailing dots and spaces, the `git~1`
/// short-name form) and the HFS rule (ignorable code points stripped).
/// Used both by [`validate_component`] and by the walk, which must treat a
/// directory holding such an entry as a nested-repository boundary.
pub fn is_dotgit(name: &[u8]) -> bool {
    // NTFS rules operate on bytes and only need ASCII case folding.
    let folded: Vec<u8> = name.iter().map(|&b| b.to_ascii_lowercase()).collect();
    let tail_all_dots_and_spaces = |tail: &[u8]| tail.iter().all(|&b| b == b'.' || b == b' ');
    if folded.starts_with(b".git") && tail_all_dots_and_spaces(&folded[4..]) {
        return true;
    }
    // The 8.3 short name an NTFS volume can synthesize for ".git": git~
    // followed by one digit and only dots and spaces.
    if folded.starts_with(b"git~")
        && folded.get(4).is_some_and(|b| b.is_ascii_digit())
        && tail_all_dots_and_spaces(&folded[5..])
    {
        return true;
    }
    // HFS: ignorable code points stripped anywhere around and inside .git.
    is_hfs_dotgit(name)
}

/// git's `.gitmodules` refusal for symlinks: the name case-insensitively,
/// with NTFS trailing dots and spaces ignored.
fn is_dotgitmodules(name: &[u8]) -> bool {
    let folded: Vec<u8> = name.iter().map(|&b| b.to_ascii_lowercase()).collect();
    folded.starts_with(b".gitmodules") && folded[11..].iter().all(|&b| b == b'.' || b == b' ')
}

/// git's `is_hfs_dotgit`: match `.git` after removing the Unicode code
/// points macOS considers ignorable in filenames.
fn is_hfs_dotgit(name: &[u8]) -> bool {
    let is_ignorable =
        |c: char| matches!(c as u32, 0x200c..=0x200f | 0x202a..=0x202e | 0x206a..=0x206f | 0xfeff);
    // A name that is not valid UTF-8 cannot carry these multi-byte code
    // points; the plain NTFS comparison above already handled the ASCII
    // aliases, so there is nothing more to match here.
    let Ok(text) = std::str::from_utf8(name) else {
        return false;
    };
    let mut chars = text.chars().filter(|&c| !is_ignorable(c));
    if chars.next() != Some('.') {
        return false;
    }
    for expected in ['g', 'i', 't'] {
        if !chars
            .next()
            .is_some_and(|c| c.eq_ignore_ascii_case(&expected))
        {
            return false;
        }
    }
    chars.next().is_none()
}

/// Why a name cannot exist on Windows, when it cannot. Checked on every
/// host: the answer is an exclusion on Windows and a portability warning
/// everywhere else.
fn windows_incompatible(name: &str) -> Option<&'static str> {
    for byte in name.bytes() {
        if matches!(byte, b'<' | b'>' | b':' | b'"' | b'|' | b'?' | b'*' | b'\\') {
            return Some("an illegal Windows character in the name");
        }
    }
    if name.ends_with('.') || name.ends_with(' ') {
        return Some("a trailing dot or space in the name");
    }
    if is_windows_device_name(name) {
        return Some("a reserved Windows device name");
    }
    None
}

/// Windows reserved device names, with or without an extension and
/// case-insensitively: CON, PRN, AUX, NUL, COM0-9 and LPT0-9. Windows 11
/// also reserves the superscript-digit forms (COM¹, COM², COM³), so they
/// are included. The device name is the part before the first dot.
/// Labels reuse the check: such a name cannot be a loose-ref file on
/// Windows.
pub fn is_windows_device_name(name: &str) -> bool {
    let base = name.split('.').next().unwrap_or(name);
    let base = base.trim_end_matches(' ');
    let plain = |expected: &str| base.eq_ignore_ascii_case(expected);
    if plain("CON") || plain("PRN") || plain("AUX") || plain("NUL") {
        return true;
    }
    let Some(rest) = base.get(..3) else {
        return false;
    };
    let digit = base.get(3..).filter(|d| d.chars().count() == 1);
    let superscript = matches!(digit, Some("¹") | Some("²") | Some("³"));
    let digit = digit.is_some_and(|d| d.chars().next().is_some_and(|c| c.is_ascii_digit()));
    (rest.eq_ignore_ascii_case("COM") || rest.eq_ignore_ascii_case("LPT")) && (digit || superscript)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn class(name: &[u8], is_symlink: bool) -> NameClass {
        validate_component(name, is_symlink).class
    }

    #[test]
    fn kernel_impossible_names_are_corrupted() {
        assert_eq!(class(b"", false), NameClass::Corrupted);
        assert_eq!(class(b"a/b", false), NameClass::Corrupted);
        assert_eq!(class(b"a\0b", false), NameClass::Corrupted);
        assert_eq!(class(b".", false), NameClass::Corrupted);
        assert_eq!(class(b"..", false), NameClass::Corrupted);
    }

    #[test]
    fn dotgit_aliases_are_never_in_the_tree() {
        for name in [".git", ".GIT", ".Git", ".git.", ".git ", ".git.. "] {
            assert_eq!(
                class(name.as_bytes(), false),
                NameClass::NeverInTree,
                "{name}"
            );
        }
        for name in ["git~1", "GIT~1", "git~1 ", "git~1.", "git~9.."] {
            assert_eq!(
                class(name.as_bytes(), false),
                NameClass::NeverInTree,
                "{name}"
            );
        }
        // Not aliases.
        for name in ["git", ".gitignore", ".github", "git~", "git~x", "a.git"] {
            assert_eq!(class(name.as_bytes(), false), NameClass::Keep, "{name}");
        }
    }

    #[test]
    fn hfs_ignorable_code_points_are_dotgit_aliases() {
        // U+200B is not in git's ignorable set; the zero-width joiners are.
        let ignorable = "\u{200d}";
        assert!(is_hfs_dotgit(format!(".git{ignorable}").as_bytes()));
        assert!(is_hfs_dotgit(".\u{feff}git".as_bytes()));
        assert!(!is_hfs_dotgit(".git\u{200b}x".as_bytes()));
        assert!(!is_hfs_dotgit(".gitx".as_bytes()));
    }

    #[test]
    fn gitmodules_symlinks_are_refused_only_as_symlinks() {
        assert_eq!(class(b".gitmodules", true), NameClass::NeverInTree);
        assert_eq!(class(b".Gitmodules.", true), NameClass::NeverInTree);
        assert_eq!(class(b".gitmodules", false), NameClass::Keep);
    }

    #[test]
    fn non_utf8_and_control_names_are_incompatible_everywhere() {
        assert_eq!(
            NameClass::IncompatibleHost,
            class(&[0x61, 0xff, 0x62], false)
        );
        assert_eq!(NameClass::IncompatibleHost, class(b"a\nb", false));
        assert_eq!(NameClass::IncompatibleHost, class(b"a\rb", false));
        assert_eq!(NameClass::IncompatibleHost, class(b"a\x01b", false));
        assert_eq!(NameClass::IncompatibleHost, class(b"a\x7fb", false));
        // A tab is whitespace but still a control byte below 0x20.
        assert_eq!(NameClass::IncompatibleHost, class(b"a\tb", false));
    }

    #[test]
    fn windows_rules_exclude_or_warn_depending_on_the_host() {
        for name in [
            "bad:name", "aux.c", "x\\y", "name.", "name ", "a<b", "com1.txt", "lpt9",
        ] {
            let verdict = validate_component(name.as_bytes(), false);
            if cfg!(windows) {
                assert_eq!(verdict.class, NameClass::IncompatibleHost, "{name}");
            } else {
                assert_eq!(verdict.class, NameClass::PortabilityWarning, "{name}");
                assert_eq!(verdict.reason, None, "{name}");
            }
        }
    }

    #[test]
    fn plain_names_are_kept() {
        for name in ["src", "a.txt", "Notes.md", "x+y", "semi;colon"] {
            assert_eq!(class(name.as_bytes(), false), NameClass::Keep, "{name}");
        }
        assert_eq!(
            validate_component(b"a\xffb", false).reason.as_deref(),
            Some("a name that is not valid UTF-8 (a\\xffb)")
        );
    }

    #[test]
    fn device_names_with_superscript_digits_are_reserved() {
        assert!(is_windows_device_name("COM¹"));
        assert!(is_windows_device_name("com².txt"));
        assert!(!is_windows_device_name("COM¹x"));
        assert!(!is_windows_device_name("COMA"));
    }

    #[test]
    fn display_paths_lose_their_control_characters() {
        // A terminal-escape sequence in a walked name must not survive
        // into a note; printable Unicode stays readable.
        assert_eq!(escape_display("a\u{1b}[2Jb"), "a\\x1b[2Jb");
        assert_eq!(escape_display("pl\u{00e1}in.txt"), "pl\u{00e1}in.txt");
        assert_eq!(escape_display("line\u{85}break"), "line\\x85break");
    }
}
