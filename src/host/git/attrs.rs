use super::git_command;
use anyhow::{Result, anyhow, bail};
use std::collections::HashMap;
use std::io::{Read, Write};
use std::path::Path;
use std::process::Stdio;
use std::thread;

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::host::git::test_support::repo_fixture;

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
}
