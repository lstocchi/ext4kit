use clap::{Parser, Subcommand};
use ext4kit::core::{Ext4ImageBuilder, HostTree};
use std::{
    collections::{HashMap, HashSet},
    fs,
    path::PathBuf,
};

#[derive(Debug, Parser)]
#[command(name = "ext4kit", about = "Create ext4 images from host directories")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Create an ext4 image containing the included file or directory.
    Create {
        /// Host file or directory, optionally followed by :/image/destination.
        /// Example: C:\\work\\hosts:/etc
        #[arg(long, required = true, value_name = "SOURCE[:/DESTINATION]")]
        add: Vec<String>,

        /// Destination ext4 image file.
        #[arg(value_name = "OUTPUT.IMG")]
        output: PathBuf,

        /// Output image size in MiB. Skips the source-tree sizing pass.
        #[arg(long, value_name = "MIB")]
        size: Option<usize>,
    },
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let Command::Create { add, output, size } = cli.command;
    let additions: Vec<Addition> = add
        .iter()
        .map(|add| parse_add(add))
        .collect::<Result<_, _>>()?;
    let destination_directories = destination_directories(&additions);
    reject_target_collisions(&additions, &destination_directories)?;
    if size.is_some_and(|size| size < 16) {
        anyhow::bail!("--size must be at least 16 MiB");
    }
    let (image_size_mb, host_trees) = match size {
        Some(size) => (size, None),
        None => {
            let trees: Vec<HostTree> = additions
                .iter()
                .map(|addition| HostTree::scan(&addition.source))
                .collect::<Result<_, _>>()?;
            let size = Ext4ImageBuilder::recommended_image_size_mib_for_trees(
                &trees,
                destination_directories.len(),
            )?;
            (size, Some(trees))
        }
    };
    println!(
        "Creating {image_size_mb}MiB ext4 filesystem image from {} addition(s)...",
        additions.len()
    );
    let mut builder = Ext4ImageBuilder::new_file(image_size_mb, &output)?;

    builder.create_root_dir()?;
    builder.create_lost_and_found()?;
    builder.reserve_inodes();

    let mut destination_inodes = HashMap::from([("/".to_owned(), ext4kit::ext4::EXT4_ROOT_INO)]);
    // Import a directory before anything that needs it as a destination.  This
    // lets `--add b:/a` provide `/a/b` for a later `--add c:/a/b`.
    let mut import_order: Vec<_> = (0..additions.len()).collect();
    import_order.sort_by_key(|&index| additions[index].target().matches('/').count());
    for index in import_order {
        let addition = &additions[index];
        let destination_parent =
            ensure_destination_path(&mut builder, &addition.destination, &mut destination_inodes)?;
        let inode = if let Some(trees) = &host_trees {
            builder.import_host_tree(destination_parent, &trees[index])?
        } else {
            builder.import_host_path_streaming(destination_parent, &addition.source)?
        };
        if addition.is_directory()? {
            destination_inodes.insert(addition.target(), inode);
        }
    }

    builder.write_to_file(output.to_str().ok_or_else(|| {
        anyhow::anyhow!("output path is not valid UTF-8: {}", output.display())
    })?)?;

    println!("Ext4 image successfully generated at: {}", output.display());
    Ok(())
}

/// One source entry and the absolute directory where its basename is placed.
#[derive(Debug)]
struct Addition {
    source: PathBuf,
    destination: String,
}

impl Addition {
    fn target(&self) -> String {
        let name = self
            .source
            .file_name()
            .and_then(|name| name.to_str())
            .expect("additions validate their source name while parsing");
        if self.destination == "/" {
            format!("/{name}")
        } else {
            format!("{}/{name}", self.destination)
        }
    }

    fn is_directory(&self) -> anyhow::Result<bool> {
        Ok(fs::symlink_metadata(&self.source)
            .map_err(|error| {
                anyhow::anyhow!(
                    "cannot inspect add source {}: {error}",
                    self.source.display()
                )
            })?
            .file_type()
            .is_dir())
    }
}

/// Parses `host-path` or `host-path:/absolute/image/path` into one addition.
/// A bare host path targets the image root; a `C:/` Windows drive prefix is
/// not treated as a destination separator.
fn parse_add(add: &str) -> anyhow::Result<Addition> {
    let destination_separator = add
        .rfind(":/")
        .filter(|&index| !is_windows_drive_separator(add, index));
    match destination_separator {
        Some(separator) => {
            let source = &add[..separator];
            let destination = normalize_destination(&add[separator + 1..])?;
            if source.is_empty() {
                anyhow::bail!("add source cannot be empty");
            }
            let source = PathBuf::from(source);
            validate_add_source(&source)?;
            Ok(Addition {
                source,
                destination,
            })
        }
        None => {
            let source = PathBuf::from(add);
            validate_add_source(&source)?;
            Ok(Addition {
                source,
                destination: "/".to_owned(),
            })
        }
    }
}

/// Rejects filesystem roots, which have no name to place in the image.
fn validate_add_source(source: &PathBuf) -> anyhow::Result<()> {
    match source.file_name().and_then(|name| name.to_str()) {
        Some(_) => Ok(()),
        None if source.parent().is_none() => anyhow::bail!(
            "add source is a filesystem root and cannot be imported as a named image entry: {}",
            source.display()
        ),
        None => anyhow::bail!(
            "add source has no valid UTF-8 file name: {}",
            source.display()
        ),
    }
}

/// Returns whether `index` is the `:/` portion of a Windows `C:/` prefix.
fn is_windows_drive_separator(value: &str, index: usize) -> bool {
    let Some(&drive) = value.as_bytes().get(index.saturating_sub(1)) else {
        return false;
    };
    if !drive.is_ascii_alphabetic() {
        return false;
    }
    let prefix = &value[..index - 1];
    prefix.is_empty() || prefix.ends_with("\\\\?\\") || prefix.ends_with("//?/")
}

/// Canonicalizes a CLI destination to an absolute slash-separated image path.
fn normalize_destination(destination: &str) -> anyhow::Result<String> {
    if !destination.starts_with('/') {
        anyhow::bail!("destination must be an absolute image path: {destination}");
    }
    let components: Vec<_> = destination
        .split('/')
        .filter(|component| !component.is_empty())
        .collect();
    if components
        .iter()
        .any(|component| *component == "." || *component == "..")
    {
        anyhow::bail!("destination cannot contain '.' or '..': {destination}");
    }
    Ok(if components.is_empty() {
        "/".to_owned()
    } else {
        format!("/{}", components.join("/"))
    })
}

/// Returns every distinct directory that must be created for `additions`.
fn destination_directories(additions: &[Addition]) -> HashSet<String> {
    additions
        .iter()
        .flat_map(|addition| {
            let mut path = String::new();
            addition
                .destination
                .split('/')
                .filter(|component| !component.is_empty())
                .map(move |component| {
                    path.push('/');
                    path.push_str(component);
                    path.clone()
                })
        })
        .collect()
}

/// Creates `destination` once, sharing directory inodes among all additions.
fn ensure_destination_path(
    builder: &mut Ext4ImageBuilder,
    destination: &str,
    directories: &mut HashMap<String, u32>,
) -> anyhow::Result<u32> {
    let mut path = String::new();
    let mut parent = ext4kit::ext4::EXT4_ROOT_INO;
    for component in destination
        .split('/')
        .filter(|component| !component.is_empty())
    {
        path.push('/');
        path.push_str(component);
        parent = match directories.get(&path) {
            Some(&inode) => inode,
            None => {
                let inode = builder.mkdir(parent, component, None)?;
                directories.insert(path.clone(), inode);
                inode
            }
        };
    }
    Ok(parent)
}

/// Rejects additions whose basenames would map to the same image path.
fn reject_target_collisions(
    additions: &[Addition],
    destination_directories: &HashSet<String>,
) -> anyhow::Result<()> {
    let mut targets = HashSet::new();
    for addition in additions {
        let target = addition.target();
        if !targets.insert(target.clone()) {
            anyhow::bail!("multiple additions target {target}");
        }
        if destination_directories.contains(&target) && !addition.is_directory()? {
            anyhow::bail!(
                "addition targets {target}, which is also needed as another addition's destination directory"
            );
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        Addition, Cli, Command, destination_directories, parse_add, reject_target_collisions,
    };
    use clap::Parser;
    use std::{fs, path::PathBuf};

    #[test]
    fn accepts_explicit_image_size() {
        let cli = Cli::try_parse_from([
            "ext4kit",
            "create",
            "--add",
            "source",
            "--size",
            "3200",
            "output.img",
        ])
        .unwrap();
        let Command::Create { size, .. } = cli.command;
        assert_eq!(size, Some(3200));
    }

    #[test]
    fn add_destination_preserves_windows_drive_prefix() {
        let addition = parse_add(r"C:\\work\\hosts:/etc").unwrap();
        assert_eq!(
            addition.source,
            std::path::PathBuf::from(r"C:\\work\\hosts")
        );
        assert_eq!(addition.destination, "/etc");
    }

    #[cfg(windows)]
    #[test]
    fn add_without_destination_preserves_forward_slash_drive_path() {
        let addition = parse_add("C:/work/hosts").unwrap();
        assert_eq!(addition.source, std::path::PathBuf::from("C:/work/hosts"));
        assert_eq!(addition.destination, "/");
    }

    #[test]
    fn extended_windows_drive_path_is_not_a_destination_separator() {
        let addition = parse_add(r"\\?\C:/work/hosts").unwrap();
        assert_eq!(addition.source, PathBuf::from(r"\\?\C:/work/hosts"));
        assert_eq!(addition.destination, "/");
    }

    #[test]
    fn root_add_source_has_a_specific_error() {
        let error = parse_add("/").unwrap_err();
        assert!(error.to_string().contains("filesystem root"));
    }

    #[test]
    fn directory_target_can_supply_another_destination_path() {
        let base = std::env::temp_dir().join(format!("ext4kit-main-test-{}", std::process::id()));
        let directory = base.join("b");
        fs::create_dir_all(&directory).unwrap();
        let additions = vec![
            Addition {
                source: directory,
                destination: "/a".to_owned(),
            },
            Addition {
                source: PathBuf::from("c"),
                destination: "/a/b".to_owned(),
            },
        ];
        let directories = destination_directories(&additions);
        assert!(directories.contains("/a/b"));
        assert!(reject_target_collisions(&additions, &directories).is_ok());
        fs::remove_dir_all(base).unwrap();
    }
}
