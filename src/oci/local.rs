//! Import of a flattened root filesystem from a local Podman or Docker engine.

use crate::{
    ext4::Ext4ImageBuilder,
    ext4::format::EXT4_ROOT_INO,
    oci::{LayerCompression, apply_compressed_layer, verify_platform},
};
use anyhow::{Context, Result, bail};
use bollard::{
    API_DEFAULT_VERSION, Docker, models::ContainerCreateBody,
    query_parameters::RemoveContainerOptionsBuilder,
};
use futures_util::TryStreamExt;
use std::{fmt, io};
use tokio::runtime::Handle;
use tokio_util::io::{StreamReader, SyncIoBridge};

/// A local container engine.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Engine {
    Podman,
    Docker,
}

impl fmt::Display for Engine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Podman => "Podman",
            Self::Docker => "Docker",
        })
    }
}

/// An image found in a local engine, inspected once.
pub struct LocalImage {
    docker: Docker,
    engine: Engine,
    name: String,
    size: u64,
}

impl LocalImage {
    /// Looks for `name` in `engines`, in order, without pulling anything.
    ///
    /// With `strict` unset (an unqualified reference), an unreachable engine,
    /// a missing image or a non-linux/amd64 image yields `Ok(None)` so the
    /// caller can fall back to the registry.  With `strict` set (an explicit
    /// `docker-daemon:` or `containers-storage:` transport), each of those is
    /// reported as an error.
    pub async fn find(name: &str, engines: &[Engine], strict: bool) -> Result<Option<Self>> {
        let mut failures = Vec::new();
        let mut reached = false;
        for &engine in engines {
            let docker = match connect(engine).await {
                Ok(docker) => docker,
                Err(error) => {
                    failures.push(format!("{error:#}"));
                    continue;
                }
            };
            reached = true;
            let inspect = match docker.inspect_image(name).await {
                Ok(inspect) => inspect,
                Err(bollard::errors::Error::DockerResponseServerError {
                    status_code: 404, ..
                }) => continue,
                Err(error) => {
                    return Err(error)
                        .with_context(|| format!("cannot inspect local {engine} image {name:?}"));
                }
            };
            if let Err(error) =
                verify_platform(inspect.os.as_deref(), inspect.architecture.as_deref(), name)
            {
                if strict {
                    return Err(error);
                }
                continue;
            }
            let size = inspect
                .size
                .and_then(|size| u64::try_from(size).ok())
                .ok_or_else(|| anyhow::anyhow!("local image {name:?} does not report its size"))?;
            return Ok(Some(Self {
                docker,
                engine,
                name: name.to_owned(),
                size,
            }));
        }
        if !strict {
            Ok(None)
        } else if reached {
            bail!("local container image {name:?} was not found")
        } else {
            bail!(
                "cannot connect to a local container engine: {}",
                failures.join("; ")
            )
        }
    }

    /// The engine's uncompressed image size in bytes.
    pub fn size(&self) -> u64 {
        self.size
    }

    /// The engine that holds the image.
    pub fn engine(&self) -> Engine {
        self.engine
    }

    /// Exports the image's root filesystem through a temporary, never-started
    /// container and applies it.  The container is removed afterwards, also
    /// when extraction fails.  Blocks on `runtime`, so it must not be called
    /// from an async context.
    pub fn import(&self, builder: &mut Ext4ImageBuilder, runtime: &Handle) -> Result<()> {
        let container = runtime
            .block_on(self.docker.create_container(
                None::<bollard::query_parameters::CreateContainerOptions>,
                ContainerCreateBody {
                    image: Some(self.name.clone()),
                    // Engines refuse to create a container without a command;
                    // data-only images have none.  It is never started.
                    cmd: Some(vec!["/".to_owned()]),
                    ..Default::default()
                },
            ))
            .with_context(|| format!("cannot create temporary container for {:?}", self.name))?;
        let id = container.id;

        let extraction = self.export_and_apply(builder, runtime, &id);
        let removal = runtime
            .block_on(self.docker.remove_container(
                &id,
                Some(RemoveContainerOptionsBuilder::new().force(true).build()),
            ))
            .with_context(|| format!("cannot remove temporary {} container {id}", self.engine));

        match (extraction, removal) {
            (Ok(()), removal) => removal,
            (Err(extraction), Ok(())) => Err(extraction),
            (Err(extraction), Err(removal)) => Err(extraction.context(format!(
                "temporary container cleanup also failed: {removal:#}"
            ))),
        }
    }

    fn export_and_apply(
        &self,
        builder: &mut Ext4ImageBuilder,
        runtime: &Handle,
        id: &str,
    ) -> Result<()> {
        let export = self.docker.export_container(id).map_err(io::Error::other);
        let reader = SyncIoBridge::new_with_handle(StreamReader::new(export), runtime.clone());
        apply_compressed_layer(builder, reader, LayerCompression::Raw)
            .context("cannot extract exported container rootfs")?;
        // Docker's init layer adds this marker to every container rootfs.
        if self.engine == Engine::Docker {
            builder.remove_dir_entry(EXT4_ROOT_INO, ".dockerenv")?;
        }
        Ok(())
    }
}

/// Connects to the first responsive endpoint of `engine`.
async fn connect(engine: Engine) -> Result<Docker> {
    let mut failures = Vec::new();
    for endpoint in endpoints(engine) {
        match connect_endpoint(&endpoint) {
            Ok(docker) => match docker.version().await {
                Ok(_) => return Ok(docker),
                Err(error) => failures.push(format!("{engine} at {endpoint}: {error}")),
            },
            Err(error) => failures.push(format!("{engine} at {endpoint}: {error}")),
        }
    }
    bail!("{}", failures.join("; "))
}

/// Candidate endpoints, honouring CONTAINER_HOST (Podman) and DOCKER_HOST.
fn endpoints(engine: Engine) -> Vec<String> {
    let variable = match engine {
        Engine::Podman => "CONTAINER_HOST",
        Engine::Docker => "DOCKER_HOST",
    };
    let mut endpoints: Vec<String> = std::env::var(variable).into_iter().collect();
    endpoints.extend(default_endpoints(engine));
    endpoints
}

#[cfg(unix)]
fn default_endpoints(engine: Engine) -> Vec<String> {
    let runtime_dir = std::env::var("XDG_RUNTIME_DIR").ok();
    let mut endpoints = Vec::new();
    match engine {
        Engine::Podman => {
            endpoints.extend(runtime_dir.map(|dir| format!("{dir}/podman/podman.sock")));
            endpoints.push("/run/podman/podman.sock".into());
        }
        Engine::Docker => {
            endpoints.push("/var/run/docker.sock".into());
            endpoints.extend(runtime_dir.map(|dir| format!("{dir}/docker.sock")));
            endpoints.extend(
                std::env::var("HOME")
                    .ok()
                    .map(|home| format!("{home}/.docker/run/docker.sock")),
            );
        }
    }
    endpoints
}

#[cfg(windows)]
fn default_endpoints(engine: Engine) -> Vec<String> {
    vec![match engine {
        Engine::Podman => "//./pipe/podman-machine-default".into(),
        Engine::Docker => "//./pipe/docker_engine".into(),
    }]
}

#[cfg(unix)]
fn connect_endpoint(endpoint: &str) -> Result<Docker> {
    let path = endpoint.strip_prefix("unix://").unwrap_or(endpoint);
    if path.contains("://") {
        bail!("unsupported endpoint scheme");
    }
    Docker::connect_with_unix(path, 120, API_DEFAULT_VERSION).map_err(Into::into)
}

#[cfg(windows)]
fn connect_endpoint(endpoint: &str) -> Result<Docker> {
    let path = endpoint.strip_prefix("npipe://").unwrap_or(endpoint);
    if path.contains("://") {
        bail!("unsupported endpoint scheme");
    }
    Docker::connect_with_named_pipe(path, 120, API_DEFAULT_VERSION).map_err(Into::into)
}
