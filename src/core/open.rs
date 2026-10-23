//! Loading an image written by ext4kit back into a builder, so that it can
//! be edited in place.

use super::{DiskData, Ext4ImageBuilder};
use crate::ext4::*;
use anyhow::{Result, anyhow, bail};
use memmap2::MmapOptions;
use std::{collections::HashMap, fs::OpenOptions, path::Path};
use zerocopy::FromBytes;

/// The builder rewrites all metadata from its own model, so it only accepts
/// the feature set it writes itself.  Anything else (a journal, flex_bg,
/// 64bit, resize_inode, checksums, ...) carries metadata that would be
/// silently dropped or corrupted.
const EXPECTED_COMPAT: u32 = EXT4_FEATURE_COMPAT_EXT_ATTR | EXT4_FEATURE_COMPAT_DIR_INDEX;
const EXPECTED_INCOMPAT: u32 = EXT4_FEATURE_INCOMPAT_FILETYPE | EXT4_FEATURE_INCOMPAT_EXTENTS;
const EXPECTED_RO_COMPAT: u32 =
    EXT4_FEATURE_RO_COMPAT_SPARSE_SUPER | EXT4_FEATURE_RO_COMPAT_DIR_NLINK;
/// The kernel sets large_file the first time a file grows past 2 GiB.  It
/// changes nothing in the layout, so it is accepted and preserved.
const OPTIONAL_RO_COMPAT: u32 = EXT4_FEATURE_RO_COMPAT_LARGE_FILE;

impl Ext4ImageBuilder {
    /// Opens an image file written by ext4kit for in-place editing and
    /// returns a builder holding its metadata.
    ///
    /// Data and directory blocks are changed directly in the mapped file.
    /// The bitmaps, inode tables and superblocks are only written by
    /// [`Ext4ImageBuilder::write_to_file`] with the same path, which is safe
    /// after any successful or failed operation because every builder
    /// operation validates before it commits.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_owned();
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .map_err(|error| anyhow!("cannot open {}: {error}", path.display()))?;
        let length = file.metadata()?.len();
        if length < 2 * 1024 {
            bail!("{} is too small to be an ext4 image", path.display());
        }
        // The file keeps its length for the lifetime of the mapping, and all
        // access is mediated through the uniquely owned MmapMut.
        let mapping = unsafe { MmapOptions::new().map_mut(&file)? };
        Self::load(DiskData::File { mapping, path })
    }

    /// Opens an image file written by ext4kit for inspection only.  The
    /// file needs only read access; the returned builder may still run
    /// mutating operations, but their effects stay in private memory and
    /// [`Ext4ImageBuilder::write_to_file`] refuses to save them.
    pub fn open_read_only(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let file = std::fs::File::open(path)
            .map_err(|error| anyhow!("cannot open {}: {error}", path.display()))?;
        if file.metadata()?.len() < 2 * 1024 {
            bail!("{} is too small to be an ext4 image", path.display());
        }
        // A private mapping: pages written by the builder are copied, so the
        // file itself is never modified.
        let mapping = unsafe { MmapOptions::new().map_copy(&file)? };
        Self::load(DiskData::ReadOnly(mapping))
    }

    /// Loads an image written by ext4kit from memory; see
    /// [`Ext4ImageBuilder::open`].
    pub fn from_image(data: Vec<u8>) -> Result<Self> {
        Self::load(DiskData::Memory(data))
    }

    /// Reads and cross-checks the superblock, group descriptors, bitmaps and
    /// inode tables of `disk_data`.
    fn load(disk_data: DiskData) -> Result<Self> {
        let superblock = read_superblock(&disk_data)?;
        let blocks = superblock.s_blocks_count_lo as usize;
        let image_bytes = blocks
            .checked_mul(BLOCK_SIZE)
            .ok_or_else(|| anyhow!("superblock block count overflows"))?;
        if disk_data.len() < image_bytes {
            bail!(
                "image is truncated: the superblock describes {image_bytes} bytes, but only {} are present",
                disk_data.len()
            );
        }
        let groups_number = blocks.div_ceil(BLOCKS_PER_GROUP);
        let inodes_per_group = superblock.s_inodes_per_group as usize;
        if groups_number == 0
            || inodes_per_group == 0
            || !inodes_per_group.is_multiple_of(8)
            || inodes_per_group > BLOCKS_PER_GROUP
            || superblock.s_inodes_count as usize != inodes_per_group * groups_number
        {
            bail!("superblock has an inconsistent block or inode geometry");
        }

        let descriptor_size = std::mem::size_of::<Ext4GroupDesc>();
        let group_descriptors: Vec<Ext4GroupDesc> = disk_data
            .get(BLOCK_SIZE..BLOCK_SIZE + groups_number * descriptor_size)
            .ok_or_else(|| anyhow!("group descriptor table lies outside the image"))?
            .chunks_exact(descriptor_size)
            .map(|bytes| Ext4GroupDesc::read_from_bytes(bytes).expect("chunks match the size"))
            .collect();

        let inode_table_blocks = (inodes_per_group * INODE_SIZE).div_ceil(BLOCK_SIZE);
        let mut groups = Vec::with_capacity(groups_number);
        let (mut free_blocks_total, mut free_inodes_total) = (0usize, 0usize);
        for (index, descriptor) in group_descriptors.iter().enumerate() {
            if descriptor.bg_flags & (EXT4_BG_INODE_UNINIT | EXT4_BG_BLOCK_UNINIT) != 0 {
                bail!("block group {index} has uninitialized metadata");
            }
            let metadata = |start: u32, count: usize, what: &str| -> Result<&[u8]> {
                let start = start as usize;
                if start == 0 || start + count > blocks {
                    bail!("block group {index} {what} lies outside the filesystem");
                }
                Ok(&disk_data[start * BLOCK_SIZE..(start + count) * BLOCK_SIZE])
            };
            let block_bitmap = metadata(descriptor.bg_block_bitmap_lo, 1, "block bitmap")?;
            let inode_bitmap = metadata(descriptor.bg_inode_bitmap_lo, 1, "inode bitmap")?;
            let inode_table: Vec<Ext4Inode> = metadata(
                descriptor.bg_inode_table_lo,
                inode_table_blocks,
                "inode table",
            )?[..inodes_per_group * INODE_SIZE]
                .as_chunks::<INODE_SIZE>()
                .0
                .iter()
                .map(|bytes| Ext4Inode::read_from_bytes(bytes).expect("chunks match the size"))
                .collect();

            let blocks_count = (blocks - index * BLOCKS_PER_GROUP).min(BLOCKS_PER_GROUP);
            let is_set = |bitmap: &[u8], bit: usize| bitmap[bit / 8] & (1 << (bit % 8)) != 0;
            let free_blocks = (0..blocks_count)
                .filter(|&bit| !is_set(block_bitmap, bit))
                .count();
            let free_inodes = (0..inodes_per_group)
                .filter(|&bit| !is_set(inode_bitmap, bit))
                .count();
            let used_dirs = inode_table
                .iter()
                .enumerate()
                .filter(|&(bit, inode)| is_set(inode_bitmap, bit) && inode.is_dir())
                .count();
            // Removal and allocation adjust these counters incrementally, so
            // they must start out exactly as e2fsck would recompute them.
            if free_blocks != usize::from(descriptor.bg_free_blocks_count_lo)
                || free_inodes != usize::from(descriptor.bg_free_inodes_count_lo)
                || used_dirs != usize::from(descriptor.bg_used_dirs_count_lo)
            {
                bail!("block group {index} counters do not match its bitmaps; run e2fsck -f");
            }
            free_blocks_total += free_blocks;
            free_inodes_total += free_inodes;

            groups.push(Ext4BlockGroup {
                group_id: index as u32,
                blocks_count: blocks_count as u32,
                block_bitmap: block_bitmap.to_vec(),
                inode_bitmap: inode_bitmap.to_vec(),
                inode_table,
                free_blocks_count: free_blocks as u16,
                free_inodes_count: free_inodes as u16,
            });
        }
        if free_blocks_total != superblock.s_free_blocks_count_lo as usize
            || free_inodes_total != superblock.s_free_inodes_count as usize
        {
            bail!("superblock free counts do not match the block groups; run e2fsck -f");
        }

        Ok(Self {
            superblock,
            group_descriptors,
            groups,
            disk_data,
            next_block_hints: vec![0; groups_number],
            next_inode_hints: vec![0; groups_number],
            directory_caches: HashMap::new(),
            host_owner: None,
        })
    }
}

/// Reads the primary superblock and rejects anything outside ext4kit's fixed
/// format choices.
fn read_superblock(data: &[u8]) -> Result<Ext4SuperBlock> {
    let bytes = data
        .get(1024..1024 + std::mem::size_of::<Ext4SuperBlock>())
        .ok_or_else(|| anyhow!("image is too small to hold an ext4 superblock"))?;
    let superblock = Ext4SuperBlock::read_from_bytes(bytes).expect("slice matches the size");
    if superblock.s_magic != EXT4_SUPER_MAGIC {
        bail!("not an ext4 image: bad superblock magic");
    }
    if superblock.s_rev_level != 1
        || superblock.s_first_data_block != 0
        || superblock.s_log_block_size != 2
        || superblock.s_log_cluster_size != 2
        || superblock.s_blocks_per_group as usize != BLOCKS_PER_GROUP
        || superblock.s_clusters_per_group as usize != BLOCKS_PER_GROUP
        || usize::from(superblock.s_inode_size) != INODE_SIZE
        || superblock.s_first_ino != EXT4_FIRST_INO
        || !matches!(superblock._s_desc_size, 0 | 32)
        || superblock._s_reserved_gdt_blocks != 0
    {
        bail!(
            "unsupported ext4 layout: only images with 4 KiB blocks, 256-byte inodes, \
             32-byte group descriptors and no reserved GDT blocks can be edited"
        );
    }
    let (compat, incompat, ro_compat) = (
        superblock.s_feature_compat,
        superblock.s_feature_incompat,
        superblock.s_feature_ro_compat,
    );
    if compat != EXPECTED_COMPAT
        || incompat != EXPECTED_INCOMPAT
        || ro_compat & !OPTIONAL_RO_COMPAT != EXPECTED_RO_COMPAT
    {
        bail!(
            "unsupported ext4 features (compat {compat:#x}, incompat {incompat:#x}, \
             ro_compat {ro_compat:#x}): only images created by ext4kit can be edited"
        );
    }
    let state = superblock.s_state;
    if state & EXT2_VALID_FS == 0 || state & EXT2_ERROR_FS != 0 || superblock._s_last_orphan != 0 {
        bail!("filesystem was not cleanly unmounted or has errors; run e2fsck -f");
    }
    // The dx hash is computed with signed chars; an image flagged for
    // unsigned hashing would get index entries the kernel cannot find.
    let flags_offset = 1024 + EXT4_SUPERBLOCK_FLAGS_OFFSET;
    let flags = u32::from_le_bytes(data[flags_offset..flags_offset + 4].try_into().unwrap());
    if flags & EXT2_FLAGS_UNSIGNED_HASH != 0 {
        bail!("unsupported ext4 image: directory hashes use unsigned characters");
    }
    Ok(superblock)
}

#[cfg(test)]
mod tests {
    use crate::core::{Ext4ImageBuilder, PosixMetadata};
    use crate::ext4::*;
    use zerocopy::{FromBytes, IntoBytes};

    /// Builds a small image with a directory and a file and returns its
    /// finalized bytes together with the builder that wrote them.
    fn written_image() -> (Ext4ImageBuilder, Vec<u8>) {
        let mut builder = Ext4ImageBuilder::new(16);
        builder.create_root_dir().unwrap();
        builder.create_lost_and_found().unwrap();
        builder.reserve_inodes();
        let etc = builder.mkdir(EXT4_ROOT_INO, "etc", None).unwrap();
        builder
            .add_file(
                etc,
                "hosts",
                b"127.0.0.1 localhost\n",
                &PosixMetadata::default(),
            )
            .unwrap();
        builder.commit_metadata();
        let bytes = builder.disk_data.to_vec();
        (builder, bytes)
    }

    fn with_superblock(bytes: &mut [u8], change: impl FnOnce(&mut Ext4SuperBlock)) {
        let range = 1024..1024 + std::mem::size_of::<Ext4SuperBlock>();
        let mut superblock = Ext4SuperBlock::read_from_bytes(&bytes[range.clone()]).unwrap();
        change(&mut superblock);
        bytes[range].copy_from_slice(superblock.as_bytes());
    }

    fn load_error(bytes: Vec<u8>) -> String {
        match Ext4ImageBuilder::from_image(bytes) {
            Ok(_) => panic!("the image was accepted"),
            Err(error) => error.to_string(),
        }
    }

    #[test]
    fn loading_a_written_image_restores_the_builder_state() {
        let (original, bytes) = written_image();
        let loaded = Ext4ImageBuilder::from_image(bytes).unwrap();

        assert_eq!(loaded.superblock.as_bytes(), original.superblock.as_bytes());
        assert_eq!(
            loaded.group_descriptors.as_bytes(),
            original.group_descriptors.as_bytes()
        );
        for (loaded, original) in loaded.groups.iter().zip(&original.groups) {
            assert_eq!(loaded.block_bitmap, original.block_bitmap);
            assert_eq!(loaded.inode_bitmap, original.inode_bitmap);
            assert_eq!(
                loaded.inode_table.as_bytes(),
                original.inode_table.as_bytes()
            );
            assert_eq!(loaded.free_blocks_count, original.free_blocks_count);
            assert_eq!(loaded.free_inodes_count, original.free_inodes_count);
        }
        let etc = loaded
            .lookup_dir_entry(EXT4_ROOT_INO, "etc")
            .unwrap()
            .unwrap();
        assert!(loaded.lookup_dir_entry(etc, "hosts").unwrap().is_some());
    }

    #[test]
    fn a_loaded_image_can_be_changed_and_loaded_again() {
        let (_, bytes) = written_image();
        let mut builder = Ext4ImageBuilder::from_image(bytes).unwrap();
        let free_blocks = builder.superblock.s_free_blocks_count_lo;
        let free_inodes = builder.superblock.s_free_inodes_count;

        let etc = builder.resolve_directory("/etc").unwrap().unwrap();
        assert!(builder.remove_dir_entry(etc, "hosts").unwrap());
        builder
            .add_file(
                etc,
                "passwd",
                b"root:x:0:0::/root:/bin/sh\n",
                &PosixMetadata::default(),
            )
            .unwrap();
        builder.mkdir(EXT4_ROOT_INO, "sandbox", None).unwrap();
        builder.commit_metadata();

        let reloaded = Ext4ImageBuilder::from_image(builder.disk_data.to_vec()).unwrap();
        assert_eq!({ reloaded.superblock.s_free_inodes_count }, free_inodes - 1);
        // The removed and the added file each use one block; the new
        // directory adds one more.
        assert_eq!(
            { reloaded.superblock.s_free_blocks_count_lo },
            free_blocks - 1
        );
        assert!(reloaded.resolve_entry("/etc/hosts").unwrap().is_none());
        assert!(reloaded.resolve_entry("/etc/passwd").unwrap().is_some());
        assert!(reloaded.resolve_directory("/sandbox").unwrap().is_some());
    }

    #[test]
    fn loading_keeps_the_directory_hash_seed() {
        let (original, bytes) = written_image();
        let loaded = Ext4ImageBuilder::from_image(bytes).unwrap();
        let (loaded_seed, original_seed) = (
            loaded.superblock.s_hash_seed,
            original.superblock.s_hash_seed,
        );
        assert_eq!(loaded_seed, original_seed);
    }

    #[test]
    fn rejects_images_that_are_not_ext4() {
        let (_, mut bytes) = written_image();
        with_superblock(&mut bytes, |superblock| superblock.s_magic = 0);
        assert!(load_error(bytes).contains("bad superblock magic"));
        assert!(load_error(vec![0; 1024]).contains("too small"));
    }

    #[test]
    fn rejects_features_the_builder_does_not_write() {
        const HAS_JOURNAL: u32 = 0x0004;
        const FLEX_BG: u32 = 0x0200;
        const METADATA_CSUM: u32 = EXT4_FEATURE_RO_COMPAT_METADATA_CSUM;
        for change in [
            (|sb: &mut Ext4SuperBlock| sb.s_feature_compat |= HAS_JOURNAL) as fn(&mut _),
            |sb| sb.s_feature_incompat |= FLEX_BG,
            |sb| sb.s_feature_ro_compat |= METADATA_CSUM,
            |sb| sb.s_feature_compat &= !EXT4_FEATURE_COMPAT_DIR_INDEX,
        ] {
            let (_, mut bytes) = written_image();
            with_superblock(&mut bytes, change);
            assert!(load_error(bytes).contains("unsupported ext4 features"));
        }
    }

    #[test]
    fn accepts_the_large_file_feature() {
        let (_, mut bytes) = written_image();
        with_superblock(&mut bytes, |superblock| {
            superblock.s_feature_ro_compat |= EXT4_FEATURE_RO_COMPAT_LARGE_FILE;
        });
        let loaded = Ext4ImageBuilder::from_image(bytes).unwrap();
        let ro_compat = loaded.superblock.s_feature_ro_compat;
        assert_ne!(ro_compat & EXT4_FEATURE_RO_COMPAT_LARGE_FILE, 0);
    }

    #[test]
    fn rejects_unclean_filesystems() {
        for change in [
            (|sb: &mut Ext4SuperBlock| sb.s_state = 0) as fn(&mut _),
            |sb| sb.s_state |= EXT2_ERROR_FS,
            |sb| sb._s_last_orphan = 12,
        ] {
            let (_, mut bytes) = written_image();
            with_superblock(&mut bytes, change);
            assert!(load_error(bytes).contains("not cleanly unmounted"));
        }
    }

    #[test]
    fn rejects_unsigned_directory_hashes() {
        let (_, mut bytes) = written_image();
        let offset = 1024 + EXT4_SUPERBLOCK_FLAGS_OFFSET;
        bytes[offset..offset + 4].copy_from_slice(&EXT2_FLAGS_UNSIGNED_HASH.to_le_bytes());
        assert!(load_error(bytes).contains("unsigned"));
    }

    #[test]
    fn rejects_counters_that_disagree_with_the_bitmaps() {
        // bg_free_blocks_count_lo of group 0 sits at byte 12 of its descriptor.
        let (_, mut bytes) = written_image();
        bytes[BLOCK_SIZE + 12] = bytes[BLOCK_SIZE + 12].wrapping_add(1);
        assert!(load_error(bytes).contains("counters do not match"));

        let (_, mut bytes) = written_image();
        with_superblock(&mut bytes, |superblock| superblock.s_free_inodes_count -= 1);
        assert!(load_error(bytes).contains("free counts do not match"));
    }

    #[test]
    fn rejects_truncated_images() {
        let (_, mut bytes) = written_image();
        bytes.truncate(bytes.len() - BLOCK_SIZE);
        assert!(load_error(bytes).contains("truncated"));
    }
}
