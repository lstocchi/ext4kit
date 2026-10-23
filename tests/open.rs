//! End-to-end tests for reopening an image written by `ext4kit create` and
//! changing it in place through the library.

mod common;

use common::{Image, arg, create};
use ext4kit::{Ext4ImageBuilder, core::PosixMetadata, ext4::EXT4_ROOT_INO};
use std::fs;
use tempfile::TempDir;

#[test]
fn reopened_image_stays_clean_after_edits() {
    let tmp = TempDir::new().unwrap();
    let src = tmp.path().join("src");
    // Enough entries to make `many` a hashed directory, so that inserting
    // into it after reopening depends on the hash seed read back from disk.
    fs::create_dir_all(src.join("many")).unwrap();
    for index in 0..2000 {
        fs::write(src.join("many").join(format!("file-{index:04}")), "x").unwrap();
    }
    fs::create_dir_all(src.join("etc")).unwrap();
    fs::write(src.join("etc/hosts"), "127.0.0.1 localhost\n").unwrap();
    let out = tmp.path().join("out.img");
    create(&["--add", &format!("{}:/", arg(&src))], &out);

    {
        let mut builder = Ext4ImageBuilder::open(&out).unwrap();
        let many = builder.resolve_directory("/src/many").unwrap().unwrap();
        builder
            .add_file(many, "added-later", b"later\n", &PosixMetadata::default())
            .unwrap();
        assert!(builder.remove_dir_entry(many, "file-0007").unwrap());
        let etc = builder.resolve_directory("/src/etc").unwrap().unwrap();
        assert!(builder.remove_dir_entry(etc, "hosts").unwrap());
        builder
            .add_file(
                etc,
                "hosts",
                b"10.0.0.1 replaced\n",
                &PosixMetadata::default(),
            )
            .unwrap();
        builder.mkdir(EXT4_ROOT_INO, "sandbox", None).unwrap();
        builder.write_to_file(out.to_str().unwrap()).unwrap();
    }

    let Some(image) = Image::check(&out) else {
        return;
    };
    // debugfs warns about checksum space on every dx directory; see AGENTS.md.
    let (htree, _) = image.debugfs_with_stderr("htree /src/many");
    assert!(htree.contains("Root node dump"), "{htree}");
    assert_eq!(image.cat("/src/many/added-later"), "later\n");
    assert_eq!(image.cat("/src/etc/hosts"), "10.0.0.1 replaced\n");
    let many = image.list("/src/many");
    assert_eq!(many.len(), 2000);
    assert!(!many.iter().any(|name| name == "file-0007"));
    assert!(image.list("/").iter().any(|name| name == "sandbox"));
}

#[test]
fn reopening_rejects_a_file_that_is_not_an_ext4kit_image() {
    let tmp = TempDir::new().unwrap();
    let out = tmp.path().join("not-ext4.img");
    fs::write(&out, vec![0u8; 64 * 1024]).unwrap();
    let error = Ext4ImageBuilder::open(&out).err().unwrap().to_string();
    assert!(error.contains("bad superblock magic"), "{error}");
}
