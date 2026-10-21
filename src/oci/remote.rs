//! Import of filesystem layers from an OCI registry.

use crate::{
    core::Ext4ImageBuilder,
    oci::{
        OCI_SIZE_MULTIPLIER, apply_compressed_layer, image_size_mib, layer_compression,
        verify_platform,
    },
};
use anyhow::{Context, Result};
use oci_distribution::{
    Client, Reference,
    client::{ClientConfig, linux_amd64_resolver},
    manifest::OciImageManifest,
    secrets::RegistryAuth,
};
use sha2::{Digest, Sha256};
use std::io::{self, Read};
use tokio::runtime::Handle;
use tokio_util::io::{StreamReader, SyncIoBridge};

/// A registry image whose manifest and config were fetched once and whose
/// reference is pinned to the manifest digest, so layers cannot drift from
/// the manifest used for sizing.
pub struct RemoteImage {
    client: Client,
    reference: Reference,
    manifest: OciImageManifest,
}

impl RemoteImage {
    /// Resolves `image` (optionally prefixed with `docker://`) to a
    /// linux/amd64 manifest.  Only anonymous registry access is supported.
    pub async fn resolve(image: &str) -> Result<Self> {
        let reference: Reference = image
            .strip_prefix("docker://")
            .unwrap_or(image)
            .parse()
            .with_context(|| format!("invalid OCI image reference {image:?}"))?;
        // The resulting filesystem is Linux even when ext4kit itself runs on
        // Windows, so never let the host OS select a Windows manifest.
        let client = Client::new(ClientConfig {
            platform_resolver: Some(Box::new(linux_amd64_resolver)),
            ..Default::default()
        });
        let (manifest, digest, config) = client
            .pull_manifest_and_config(&reference, &RegistryAuth::Anonymous)
            .await
            .with_context(|| format!("cannot pull manifest for {image}"))?;
        // A tag that names a single-platform manifest bypasses the resolver.
        let config: serde_json::Value =
            serde_json::from_str(&config).context("invalid image config")?;
        verify_platform(
            config["os"].as_str(),
            config["architecture"].as_str(),
            image,
        )?;
        for layer in &manifest.layers {
            layer_compression(&layer.media_type)?;
        }
        let reference = Reference::with_digest(
            reference.registry().to_owned(),
            reference.repository().to_owned(),
            digest,
        );
        Ok(Self {
            client,
            reference,
            manifest,
        })
    }

    /// Estimates the unpacked filesystem capacity from compressed layer sizes.
    pub fn size_mib(&self) -> Result<usize> {
        manifest_size_mib(self.manifest.layers.iter().map(|layer| layer.size))
    }

    /// Downloads and applies the layers from oldest to newest, verifying each
    /// blob digest.  Blocks on `runtime`, so it must not be called from an
    /// async context.
    pub fn import(&self, builder: &mut Ext4ImageBuilder, runtime: &Handle) -> Result<()> {
        for layer in &self.manifest.layers {
            let stream = runtime
                .block_on(self.client.pull_blob_stream(&self.reference, layer))
                .with_context(|| format!("cannot download layer {}", layer.digest))?;
            let reader = SyncIoBridge::new_with_handle(StreamReader::new(stream), runtime.clone());
            let mut verified = DigestReader::new(reader, &layer.digest)?;
            apply_compressed_layer(
                builder,
                &mut verified,
                layer_compression(&layer.media_type)?,
            )
            .and_then(|()| verified.finish())
            .with_context(|| format!("cannot import layer {}", layer.digest))?;
        }
        Ok(())
    }
}

/// Hashes everything read through it and compares it with a sha256 digest.
struct DigestReader<R> {
    inner: R,
    hasher: Sha256,
    expected: String,
}

impl<R: Read> DigestReader<R> {
    fn new(inner: R, digest: &str) -> Result<Self> {
        let expected = digest
            .strip_prefix("sha256:")
            .ok_or_else(|| anyhow::anyhow!("unsupported layer digest {digest:?}"))?
            .to_ascii_lowercase();
        Ok(Self {
            inner,
            hasher: Sha256::new(),
            expected,
        })
    }

    /// Reads any remaining bytes and verifies the digest.
    fn finish(mut self) -> Result<()> {
        io::copy(&mut self, &mut io::sink()).context("cannot read layer")?;
        let actual = format!("{:x}", self.hasher.finalize());
        if actual != self.expected {
            anyhow::bail!(
                "layer digest mismatch: expected sha256:{}, got sha256:{actual}",
                self.expected
            );
        }
        Ok(())
    }
}

impl<R: Read> Read for DigestReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let read = self.inner.read(buf)?;
        self.hasher.update(&buf[..read]);
        Ok(read)
    }
}

fn manifest_size_mib(layer_sizes: impl IntoIterator<Item = i64>) -> Result<usize> {
    let compressed_bytes = layer_sizes.into_iter().try_fold(0u64, |total, size| {
        let size = u64::try_from(size)
            .map_err(|_| anyhow::anyhow!("manifest contains a negative layer size"))?;
        total
            .checked_add(size)
            .ok_or_else(|| anyhow::anyhow!("manifest layer sizes overflow"))
    })?;
    image_size_mib(compressed_bytes, OCI_SIZE_MULTIPLIER)
}

#[cfg(test)]
mod tests {
    use super::manifest_size_mib;
    use crate::oci::{IMAGE_SIZE_BUFFER_BYTES, OCI_SIZE_MULTIPLIER, image_size_mib};

    #[test]
    fn registry_size_uses_compressed_layer_estimate_and_buffer() {
        let compressed_layers = 10 * 1024 * 1024;
        assert_eq!(
            manifest_size_mib([3 * 1024 * 1024, 7 * 1024 * 1024]).unwrap(),
            542
        );
        assert_eq!(
            image_size_mib(compressed_layers, OCI_SIZE_MULTIPLIER).unwrap(),
            542
        );
        assert_eq!(IMAGE_SIZE_BUFFER_BYTES / (1024 * 1024), 512);
    }
}
