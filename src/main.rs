use clap::{Parser, Subcommand};
use ext4kit::{
    core::{Ext4ImageBuilder, HostTree},
    oci::{
        image_size_mib,
        local::{Engine, LocalImage},
        remote::RemoteImage,
    },
};
use std::{
    collections::HashSet,
    fs,
    path::{Path, PathBuf},
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
        #[arg(
            long,
            value_name = "SOURCE[:/DESTINATION]",
            required_unless_present = "from"
        )]
        add: Vec<String>,

        /// Container image from a local engine when available, otherwise an OCI registry.
        #[arg(long, value_name = "IMAGE")]
        from: Option<String>,

        /// Destination ext4 image file.
        #[arg(value_name = "OUTPUT.IMG")]
        output: PathBuf,

        /// Output image size in MiB. Skips the source-tree sizing pass and
        /// overrides the base-image estimate.
        #[arg(long, value_name = "MIB")]
        size: Option<usize>,
    },
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let Command::Create {
        add,
        from,
        output,
        size,
    } = cli.command;
    let additions: Vec<Addition> = add
        .iter()
        .map(|add| parse_add(add))
        .collect::<Result<_, _>>()?;
    let destination_directories = destination_directories(&additions);
    reject_target_collisions(&additions, &destination_directories)?;
    if size.is_some_and(|size| size < 16) {
        anyhow::bail!("--size must be at least 16 MiB");
    }
    // The runtime only drives engine and registry I/O; image import itself
    // runs on this thread and blocks on it.
    let base = match from.as_deref() {
        Some(image) => {
            let runtime = tokio::runtime::Runtime::new()?;
            let source = runtime.block_on(resolve_image(image))?;
            Some((runtime, source))
        }
        None => None,
    };
    let base_size = base
        .as_ref()
        .map(|(_, source)| source.size_mib())
        .transpose()?;
    let (image_size_mb, host_trees) = match size {
        Some(size) => {
            if let Some(estimate) = base_size.filter(|&estimate| estimate > size) {
                eprintln!(
                    "warning: --size {size} is below the base image estimate of {estimate} MiB"
                );
            }
            (size, None)
        }
        None => {
            let trees: Vec<HostTree> = additions
                .iter()
                .map(|addition| HostTree::scan(&addition.source))
                .collect::<Result<_, _>>()?;
            let host_size = Ext4ImageBuilder::recommended_image_size_mib_for_trees(
                &trees,
                destination_directories.len(),
            )?;
            let size = output_image_size_mib(host_size, base_size, !additions.is_empty())?;
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

    if let Some((runtime, source)) = &base {
        match source {
            ImageSource::Local(image) => {
                println!("Importing local {} image...", image.engine());
                image.import(&mut builder, runtime.handle())?;
            }
            ImageSource::Remote(image) => {
                println!("Importing OCI image...");
                image.import(&mut builder, runtime.handle())?;
            }
        }
    }

    // Import a directory before anything that needs it as a destination.  This
    // lets `--add b:/a` provide `/a/b` for a later `--add c:/a/b`.
    let mut import_order: Vec<_> = (0..additions.len()).collect();
    import_order.sort_by_key(|&index| additions[index].target().matches('/').count());
    for index in import_order {
        let addition = &additions[index];
        let destination_parent = builder.ensure_directory(&addition.destination, false)?;
        let name = addition.name();
        let existing = builder.lookup_dir_entry(destination_parent, name)?;
        match existing {
            Some(inode) if addition.is_directory()? && builder.is_directory(inode) => {
                builder.overlay_host_directory(inode, &addition.source)?;
            }
            _ => {
                builder.remove_dir_entry(destination_parent, name)?;
                if let Some(trees) = &host_trees {
                    builder.import_host_tree(destination_parent, &trees[index])?;
                } else {
                    builder.import_host_path_streaming(destination_parent, &addition.source)?;
                }
            }
        }
    }

    builder.write_to_file(output.to_str().ok_or_else(|| {
        anyhow::anyhow!("output path is not valid UTF-8: {}", output.display())
    })?)?;

    println!("Ext4 image successfully generated at: {}", output.display());
    Ok(())
}

enum ImageSource {
    Local(LocalImage),
    Remote(Box<RemoteImage>),
}

impl ImageSource {
    fn size_mib(&self) -> anyhow::Result<usize> {
        match self {
            Self::Local(image) => local_image_size_mib(image.size()),
            Self::Remote(image) => image.size_mib(),
        }
    }
}

/// Resolves explicit transports first; a plain name favors a locally present
/// image and otherwise uses the registry without ever pulling through an engine.
async fn resolve_image(image: &str) -> anyhow::Result<ImageSource> {
    match requested_transport(image)? {
        RequestedTransport::Local(engine, name) => {
            let image = LocalImage::find(name, &[engine], true)
                .await?
                .expect("strict lookup reports a missing image as an error");
            Ok(ImageSource::Local(image))
        }
        RequestedTransport::Remote => Ok(ImageSource::Remote(Box::new(
            RemoteImage::resolve(image).await?,
        ))),
        RequestedTransport::Auto => {
            match LocalImage::find(image, &[Engine::Podman, Engine::Docker], false).await? {
                Some(local) => Ok(ImageSource::Local(local)),
                None => Ok(ImageSource::Remote(Box::new(
                    RemoteImage::resolve(image).await?,
                ))),
            }
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
enum RequestedTransport<'a> {
    Local(Engine, &'a str),
    Remote,
    Auto,
}

fn requested_transport(image: &str) -> anyhow::Result<RequestedTransport<'_>> {
    Ok(if let Some(name) = image.strip_prefix("docker-daemon:") {
        RequestedTransport::Local(Engine::Docker, name)
    } else if let Some(name) = image.strip_prefix("containers-storage:") {
        RequestedTransport::Local(Engine::Podman, name)
    } else if image.starts_with("docker://") {
        RequestedTransport::Remote
    } else if image.starts_with("oci:") {
        // In containers tooling `oci:` names a local OCI layout directory.
        anyhow::bail!("OCI layout directories are not supported; use docker://REFERENCE")
    } else {
        RequestedTransport::Auto
    })
}

fn local_image_size_mib(size: u64) -> anyhow::Result<usize> {
    // Keep the computation integral: floor(size * 1.2) plus the buffer, then
    // round the resulting filesystem allocation up to a MiB.
    let payload = size
        .checked_mul(6)
        .ok_or_else(|| anyhow::anyhow!("local image size calculation overflow"))?
        / 5;
    image_size_mib(payload, 1)
}

/// Combines the independent base-image and host-tree sizing estimates.
fn output_image_size_mib(
    host_size: usize,
    image_size: Option<usize>,
    has_host_additions: bool,
) -> anyhow::Result<usize> {
    match (image_size, has_host_additions) {
        (Some(image_size), true) => host_size
            .checked_add(image_size)
            .ok_or_else(|| anyhow::anyhow!("combined image size overflows usize")),
        (Some(image_size), false) => Ok(image_size),
        (None, _) => Ok(host_size),
    }
}

/// One source entry and the absolute directory where its basename is placed.
#[derive(Debug)]
struct Addition {
    source: PathBuf,
    destination: String,
}

impl Addition {
    /// The source basename, validated while parsing.
    fn name(&self) -> &str {
        self.source
            .file_name()
            .and_then(|name| name.to_str())
            .expect("additions validate their source name while parsing")
    }

    fn target(&self) -> String {
        if self.destination == "/" {
            format!("/{}", self.name())
        } else {
            format!("{}/{}", self.destination, self.name())
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
fn validate_add_source(source: &Path) -> anyhow::Result<()> {
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
        Addition, Cli, Command, RequestedTransport, destination_directories, local_image_size_mib,
        output_image_size_mib, parse_add, reject_target_collisions, requested_transport,
    };
    use clap::Parser;
    use ext4kit::oci::local::Engine;
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
    fn accepts_oci_image_without_host_additions() {
        let cli = Cli::try_parse_from([
            "ext4kit",
            "create",
            "--from",
            "docker.io/library/alpine:latest",
            "--size",
            "64",
            "output.img",
        ])
        .unwrap();
        let Command::Create { from, add, .. } = cli.command;
        assert_eq!(from.as_deref(), Some("docker.io/library/alpine:latest"));
        assert!(add.is_empty());
    }

    #[test]
    fn accepts_local_transport_without_host_additions() {
        let cli = Cli::try_parse_from([
            "ext4kit",
            "create",
            "--from",
            "docker-daemon:alpine:latest",
            "--size",
            "64",
            "output.img",
        ])
        .unwrap();
        let Command::Create { from, add, .. } = cli.command;
        assert_eq!(from.as_deref(), Some("docker-daemon:alpine:latest"));
        assert!(add.is_empty());
    }

    #[test]
    fn accepts_from_and_add_without_an_explicit_size() {
        let cli = Cli::try_parse_from([
            "ext4kit",
            "create",
            "--from",
            "docker://example/image",
            "--add",
            "host:/opt",
            "output.img",
        ])
        .unwrap();
        let Command::Create {
            from, add, size, ..
        } = cli.command;
        assert_eq!(from.as_deref(), Some("docker://example/image"));
        assert_eq!(add, vec![String::from("host:/opt")]);
        assert_eq!(size, None);
    }

    #[test]
    fn routes_explicit_transports() {
        assert_eq!(
            requested_transport("docker://example/image").unwrap(),
            RequestedTransport::Remote
        );
        assert_eq!(
            requested_transport("docker-daemon:alpine:latest").unwrap(),
            RequestedTransport::Local(Engine::Docker, "alpine:latest")
        );
        assert_eq!(
            requested_transport("containers-storage:alpine").unwrap(),
            RequestedTransport::Local(Engine::Podman, "alpine")
        );
        assert_eq!(
            requested_transport("alpine:latest").unwrap(),
            RequestedTransport::Auto
        );
        assert!(requested_transport("oci:/tmp/layout").is_err());
    }

    #[test]
    fn local_size_adds_twenty_percent_and_a_512_mib_buffer() {
        assert_eq!(local_image_size_mib(100 * 1024 * 1024).unwrap(), 632);
    }

    #[test]
    fn sizing_combines_from_image_and_host_additions_without_explicit_size() {
        assert_eq!(output_image_size_mib(128, Some(640), true).unwrap(), 768);
    }

    #[test]
    fn sizing_uses_only_the_from_image_when_there_are_no_host_additions() {
        assert_eq!(output_image_size_mib(128, Some(640), false).unwrap(), 640);
    }

    #[test]
    fn sizing_uses_the_host_tree_estimate_without_a_from_image() {
        assert_eq!(output_image_size_mib(128, None, true).unwrap(), 128);
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
