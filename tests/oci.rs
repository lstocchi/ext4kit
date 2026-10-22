//! End-to-end tests for `ext4kit create --from`. They need a registry or a
//! container engine, so they are ignored by default. Run them with
//! `cargo test --test oci -- --ignored`; the docker-daemon test expects
//! `docker pull alpine:3.20` to have run first.

mod common;

use common::{Image, arg, create};
use std::fs;
use tempfile::TempDir;

const ALPINE: &str = "docker.io/library/alpine:3.20";

/// Builds an image from `from` with a host `etc` directory overlaid on the
/// base image's `/etc`, then checks both survived.
fn assert_overlaid_image(from: &str) {
    let tmp = TempDir::new().unwrap();
    let etc = tmp.path().join("etc");
    fs::create_dir(&etc).unwrap();
    fs::write(etc.join("ext4kit.conf"), "overlay=yes\n").unwrap();
    let out = tmp.path().join("out.img");
    create(
        &[
            "--from",
            from,
            "--add",
            &format!("{}:/", arg(&etc)),
            "--root-owner",
            "1000",
            "--chown",
            "1000",
        ],
        &out,
    );
    let Some(image) = Image::check(&out) else {
        return;
    };
    assert!(image.cat("/etc/os-release").contains("Alpine Linux"));
    assert_eq!(image.cat("/etc/ext4kit.conf"), "overlay=yes\n");
    // The overlaid directory keeps the base image's ownership and --chown
    // applies only to the added file.
    assert_eq!(image.owner("/etc"), (0, 0));
    assert_eq!(image.owner("/etc/ext4kit.conf"), (1000, 1000));
    assert_eq!(image.owner("/"), (1000, 1000));
    assert_eq!(image.owner("/bin/busybox"), (0, 0));
    let sh = image.stat("/bin/sh");
    assert!(sh.contains("Type: symlink"), "{sh}");
}

#[test]
#[ignore = "pulls from Docker Hub"]
fn registry_image_with_host_overlay_is_clean() {
    assert_overlaid_image(&format!("docker://{ALPINE}"));
}

#[test]
#[ignore = "needs a Docker engine with alpine:3.20 pulled"]
fn docker_daemon_image_with_host_overlay_is_clean() {
    assert_overlaid_image(&format!("docker-daemon:{ALPINE}"));
}
