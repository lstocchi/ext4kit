use ext4kit::{
    core::{Ext4ImageBuilder, PosixMetadata},
    ext4,
};

/// This is used just to build the demonstration ext4 image
fn main() -> anyhow::Result<()> {
    const IMAGE_SIZE_MB: usize = 400;
    let output_file = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "output.img".to_owned());

    println!("Creating {IMAGE_SIZE_MB}MB ext4 filesystem image...");
    let mut builder = Ext4ImageBuilder::new(IMAGE_SIZE_MB);

    builder.create_root_dir()?;
    builder.create_lost_and_found()?;
    builder.reserve_inodes();
    let mut sample_metadata = PosixMetadata {
        mode: 0o644,
        ..Default::default()
    };
    sample_metadata
        .xattrs
        .insert("user.creator".to_owned(), b"ext4kit".to_vec());
    let dir = builder.mkdir(ext4::EXT4_ROOT_INO, "test_dir", None)?;
    builder.add_file(dir, "hello.txt", b"Created by ext4kit\n", &sample_metadata)?;

    builder.write_to_file(&output_file)?;

    println!("Ext4 image successfully generated at: {}", output_file);
    Ok(())
}
