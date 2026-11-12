//! The jbd2 journal ext4kit writes into every new image, so the guest
//! kernel recovers a crashed VM by replaying it on mount instead of needing
//! e2fsprogs in the guest.
//!
//! The journal is written once, here, and never touched again: the builder
//! does not replay or checkpoint — the kernel is the only writer after
//! create. The on-disk form mirrors `mke2fs` (inode 8, extent-mapped,
//! contiguous blocks, a v2 journal superblock without checksum
//! features), except placement: mke2fs puts the journal mid-filesystem,
//! ext4kit at the first free data block. All jbd2 fields are
//! **big-endian**, unlike every ext4 field.

use super::Ext4ImageBuilder;
use super::extent::extents_for_runs;
use crate::ext4::format::*;
use anyhow::{Result, bail};

/// The jbd2 journal superblock: the header words plus the fields mke2fs
/// writes into the first block of the journal.
pub(crate) struct JournalSuperblockBytes {
    /// Length of the journal in blocks, including the superblock block.
    maxlen: u32,
    /// The filesystem UUID, stored verbatim (not endian-swapped).
    uuid: [u8; 16],
}

impl JournalSuperblockBytes {
    /// Serializes the jsb. Offsets per e2fsprogs `kernel-jbd.h`
    /// https://github.com/tytso/e2fsprogs/blob/master/lib/ext2fs/kernel-jbd.h#L183
    fn to_bytes(&self) -> [u8; BLOCK_SIZE] {
        let mut jsb = [0u8; BLOCK_SIZE];
        let put = |jsb: &mut [u8; BLOCK_SIZE], offset: usize, value: u32| {
            jsb[offset..offset + 4].copy_from_slice(&value.to_be_bytes());
        };
        put(&mut jsb, 0x00, crate::ext4::reader::JBD2_MAGIC);
        // JBD2_SUPERBLOCK_V2: the kernel and e2fsck expect a v2 jsb.
        put(&mut jsb, 0x04, crate::ext4::reader::JBD2_SUPERBLOCK_V2);
        put(&mut jsb, 0x08, 0); // h_sequence of the superblock itself
        put(&mut jsb, 0x0C, BLOCK_SIZE as u32); // s_blocksize
        put(&mut jsb, 0x10, self.maxlen); // s_maxlen
        put(&mut jsb, 0x14, 1); // s_first: first log block after the jsb
        put(&mut jsb, 0x18, 1); // s_sequence: first transaction to use
        put(&mut jsb, 0x1C, 0); // s_start: the log is empty
        put(&mut jsb, 0x20, 0); // s_errno
        put(&mut jsb, 0x24, 0); // s_feature_compat: no checksums
        put(&mut jsb, 0x28, 0); // s_feature_incompat
        put(&mut jsb, 0x2C, 0); // s_feature_ro_compat
        jsb[0x30..0x40].copy_from_slice(&self.uuid);
        put(&mut jsb, 0x40, 1); // s_nr_users: this filesystem only
        put(&mut jsb, 0x44, 0); // s_dynsuper: external journals only
        put(&mut jsb, 0x48, 0); // s_max_transaction: 0 = no limit
        put(&mut jsb, 0x4C, 0); // s_max_trans_data: 0 = no limit
        put(&mut jsb, 0x50, 0); // s_checksum_type + padding2: JBD2_CHECKSUM_NONE
        put(&mut jsb, 0x54, 0); // s_num_fc_blks: no fast commit
        put(&mut jsb, 0xFC, 0); // s_checksum: ignored without checksum features
        // s_users (0x100..0x200): only scanned when mounting an external
        // journal, so an internal one leaves it zeroed like mke2fs does.
        jsb
    }
}

/// The journal size `ext2fs_default_journal_size` picks for an image of
/// `blocks` blocks: `None` below 8 MiB (mke2fs writes no journal there),
/// otherwise 4 MiB under 128 MiB and a growing ladder above. An honest
/// journal never exceeds its filesystem.
pub(crate) fn default_journal_size(blocks: u64) -> Option<u32> {
    if blocks < 2048 {
        return None;
    }
    let journal: u64 = if blocks < 32768 {
        1024
    } else if blocks < 256 * 1024 {
        4096
    } else if blocks < 512 * 1024 {
        8192
    } else if blocks < 4096 * 1024 {
        16384
    } else if blocks < 8192 * 1024 {
        32768
    } else if blocks < 16384 * 1024 {
        65536
    } else if blocks < 32768 * 1024 {
        131072
    } else {
        262144
    };
    (journal < blocks).then_some(journal as u32)
}

impl Ext4ImageBuilder {
    /// Creates the journal: a contiguous run of `journal_blocks` data
    /// blocks owned by extent-mapped inode 8, a fresh v2 journal
    /// superblock in the first block, and the superblock feature bits and
    /// backup fields mke2fs writes. Only called by the constructors; the
    /// blocks are taken from the fresh data area, so the allocation cannot
    /// fragment the journal.
    pub(crate) fn create_journal(&mut self) -> Result<()> {
        let fs_blocks = u64::from(self.superblock.s_blocks_count_lo);
        let Some(journal_blocks) = default_journal_size(fs_blocks) else {
            return Ok(());
        };
        let free_blocks = self.superblock.s_free_blocks_count_lo;
        if u64::from(free_blocks) < u64::from(journal_blocks) {
            bail!("no room for a {journal_blocks}-block journal: {free_blocks} free blocks");
        }

        // Plan the run before touching any bitmap (the preflight rule), then
        // commit it; on a fresh image this is one contiguous extent.
        let runs = self.plan_data_blocks(0, journal_blocks)?;
        if runs.is_empty() {
            bail!("the journal needs {journal_blocks} blocks");
        }
        self.commit_data_blocks(&runs);

        let first_block = runs[0].0;

        // Inode 8: mode 0600, root-owned, extent-mapped, its block list in
        // the superblock backup. The extent tree is written with the same
        // machinery as file payloads, so a journal too large for one extent
        // (over 32768 blocks) still lays out correctly.
        //
        // This breaks "preflight, then commit" on purpose: a journal with
        // more than 4 runs needs an external tree block, and
        // write_extent_tree allocates it after the data blocks are
        // committed. That is acceptable only because the constructors are
        // the sole callers and turn any failure into a panic, so no
        // half-built builder ever escapes.
        let size = u64::from(journal_blocks) * BLOCK_SIZE as u64;
        let mut inode = Ext4Inode {
            i_mode: S_IFREG | 0o600,
            i_links_count: 1,
            i_flags: EXT4_EXTENTS_FL,
            i_size_lo: size as u32,
            i_size_high: (size >> 32) as u32,
            ..Ext4Inode::default()
        };
        let tree_blocks = self.write_extent_tree(0, &mut inode, &extents_for_runs(&runs))?;
        let data_blocks: u32 = runs.iter().map(|&(_, len)| len).sum();
        inode.i_blocks_lo = (data_blocks + tree_blocks) * (BLOCK_SIZE as u32 / 512);
        let local_index = (EXT4_JOURNAL_INO - 1) as usize;
        self.groups[0].inode_table[local_index] = inode;
        self.mark_inode_used(EXT4_JOURNAL_INO);

        // A fresh jsb in the journal's first block; the rest of the journal
        // stays zeroed like the surrounding image.
        let jsb = JournalSuperblockBytes {
            maxlen: journal_blocks,
            uuid: self.superblock.s_uuid,
        }
        .to_bytes();
        let offset = first_block as usize * BLOCK_SIZE;
        self.disk_data[offset..offset + BLOCK_SIZE].copy_from_slice(&jsb);

        // Superblock feature bit, journal location and backup, as mke2fs
        // writes them.
        self.superblock.s_feature_compat |= EXT4_FEATURE_COMPAT_HAS_JOURNAL;
        self.superblock._s_journal_inum = EXT4_JOURNAL_INO;
        self.superblock._s_jnl_backup_type = EXT3_JNL_BACKUP_BLOCKS;
        // The backup (mkjournal.c): the journal inode's extent root as 15
        // u32s, then i_size_high, then i_size. e2fsck rebuilds inode 8
        // from this backup when the inode is damaged, and a writable
        // e2fsck rewrites it when it doesn't match, so the format must
        // be exact.
        let mut backup = [0u32; 17];
        let (words, _) = inode.i_block.as_chunks::<4>();
        for (word, chunk) in backup.iter_mut().zip(words) {
            *word = u32::from_le_bytes(*chunk);
        }
        backup[15] = inode.i_size_high;
        backup[16] = inode.i_size_lo;
        self.superblock.s_jnl_blocks = backup;
        Ok(())
    }

    /// The journal size the constructors would pick for this image's block
    /// count, in blocks. Used by the sizing estimate so a recommended image
    /// has room for tree plus journal.
    pub fn journal_blocks_for_size(size_mib: usize) -> Option<u32> {
        default_journal_size((size_mib as u64) * 1024 * 1024 / BLOCK_SIZE as u64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ext4::PosixMetadata;
    use crate::ext4::extent::extent_len;
    use crate::ext4::test_support::fresh_builder;

    #[test]
    fn the_journal_ladder_matches_mke2fs() {
        // 8 MiB is mke2fs's minimum for a journal; below it, none.
        assert_eq!(default_journal_size(2047), None);
        // Under 128 MiB: 4 MiB, the size the WSL experiments showed.
        assert_eq!(default_journal_size(2048), Some(1024));
        assert_eq!(default_journal_size(4096), Some(1024));
        assert_eq!(default_journal_size(32767), Some(1024));
        // The growing ladder above. Every tier was measured against
        // mke2fs 1.47.2 in WSL (the size depends only on the block count): 256 MiB
        // → 16 MiB, 512 MiB → 16 MiB, 1 GiB → 32 MiB, 2–8 GiB → 64 MiB,
        // 20 GiB → 128 MiB, 32 GiB → 256 MiB, 64 GiB → 512 MiB.
        assert_eq!(default_journal_size(32768), Some(4096));
        assert_eq!(default_journal_size(65536), Some(4096));
        assert_eq!(default_journal_size(256 * 1024), Some(8192));
        // Each tier starts exactly at its bound (the ladder compares with
        // `<`), so check both sides of every boundary.
        assert_eq!(default_journal_size(512 * 1024 - 1), Some(8192));
        assert_eq!(default_journal_size(512 * 1024), Some(16384));
        assert_eq!(default_journal_size(4096 * 1024 - 1), Some(16384));
        assert_eq!(default_journal_size(4096 * 1024), Some(32768));
        assert_eq!(default_journal_size(8192 * 1024 - 1), Some(32768));
        assert_eq!(default_journal_size(8192 * 1024), Some(65536));
        assert_eq!(default_journal_size(16384 * 1024 - 1), Some(65536));
        assert_eq!(default_journal_size(16384 * 1024), Some(131072));
        assert_eq!(default_journal_size(32768 * 1024 - 1), Some(131072));
        // The 1 GiB cap: mke2fs never goes larger.
        assert_eq!(default_journal_size(32768 * 1024), Some(262144));
        assert_eq!(default_journal_size(u64::from(u32::MAX)), Some(262144));
    }

    #[test]
    fn a_new_image_carries_an_empty_extent_mapped_journal() -> Result<()> {
        let builder = fresh_builder()?;
        let compat = builder.superblock.s_feature_compat;
        let journal_inum = builder.superblock._s_journal_inum;
        assert_ne!(compat & EXT4_FEATURE_COMPAT_HAS_JOURNAL, 0);
        assert_eq!(journal_inum, EXT4_JOURNAL_INO);

        // The reader (which validates the jsb big-endian) must see an
        // empty journal; serialize first, like write_to_file does.
        let mut builder = builder;
        builder.commit_metadata();
        let (_, state) = crate::ext4::reader::ImageReader::from_memory(builder.disk_data.to_vec())?;
        assert!(state.has_journal);
        assert!(state.journal_empty);
        assert!(!state.mounted);

        let inode = builder.inode(EXT4_JOURNAL_INO);
        let (mode, flags, size_lo, blocks_lo) = (
            inode.i_mode,
            inode.i_flags,
            inode.i_size_lo,
            inode.i_blocks_lo,
        );
        assert_eq!(mode & S_IFMT, S_IFREG);
        assert_eq!(mode & 0o777, 0o600);
        assert_eq!(flags & EXT4_EXTENTS_FL, EXT4_EXTENTS_FL);
        let journal_blocks = default_journal_size(4096).unwrap();
        assert_eq!(
            size_lo,
            journal_blocks * BLOCK_SIZE as u32,
            "a 16 MiB image carries the 4 MiB journal"
        );
        let extents = builder.get_inode_extents(inode)?;
        let mapped: u32 = extents.iter().map(extent_len).sum();
        assert_eq!(mapped, journal_blocks);
        assert_eq!(blocks_lo, journal_blocks * (BLOCK_SIZE as u32 / 512));

        // The backup: the journal inode's extent root as 15 u32s, then
        // i_size_high, then i_size (mkjournal.c format, verified against
        // a real mke2fs image's superblock).
        let backup = builder.superblock.s_jnl_blocks;
        let mut i_block_words = [0u32; 15];
        let (words, _) = inode.i_block.as_chunks::<4>();
        for (word, chunk) in i_block_words.iter_mut().zip(words) {
            *word = u32::from_le_bytes(*chunk);
        }
        assert_eq!(&backup[..15], &i_block_words);
        assert_eq!(backup[15], { inode.i_size_high });
        assert_eq!(backup[16], { inode.i_size_lo });

        // Free counts must reflect the journal: reopen recomputes them.
        let reloaded = Ext4ImageBuilder::from_image(builder.disk_data.to_vec())?;
        let (free, expected) = (
            reloaded.superblock.s_free_blocks_count_lo,
            builder.superblock.s_free_blocks_count_lo,
        );
        assert_eq!(free, expected);
        Ok(())
    }

    #[test]
    fn a_journal_opt_out_image_has_no_journal() -> Result<()> {
        let mut builder = Ext4ImageBuilder::new_without_journal(16);
        builder.create_root_dir()?;
        builder.create_lost_and_found()?;
        builder.reserve_inodes();
        let compat = builder.superblock.s_feature_compat;
        assert_eq!(compat & EXT4_FEATURE_COMPAT_HAS_JOURNAL, 0);
        // Inode 8 is still reserved by reserve_inodes, but holds no journal.
        let journal_inode = builder.inode(EXT4_JOURNAL_INO);
        let (flags, size_lo) = (journal_inode.i_flags, journal_inode.i_size_lo);
        assert_eq!(flags & EXT4_EXTENTS_FL, 0);
        assert_eq!(size_lo, 0);
        builder.commit_metadata();
        let reloaded = Ext4ImageBuilder::from_image(builder.disk_data.to_vec())?;
        let reloaded_compat = reloaded.superblock.s_feature_compat;
        assert_eq!(reloaded_compat & EXT4_FEATURE_COMPAT_HAS_JOURNAL, 0);
        Ok(())
    }

    #[test]
    fn images_below_the_mke2fs_threshold_get_no_journal() -> Result<()> {
        // 4 MiB = 1024 blocks, below mke2fs's 2048-block journal minimum.
        let mut builder = Ext4ImageBuilder::new(4);
        builder.create_root_dir()?;
        builder.create_lost_and_found()?;
        builder.reserve_inodes();
        let compat = builder.superblock.s_feature_compat;
        assert_eq!(compat & EXT4_FEATURE_COMPAT_HAS_JOURNAL, 0);
        Ok(())
    }

    #[test]
    fn the_journal_survives_edits_and_a_reopen() -> Result<()> {
        let mut builder = fresh_builder()?;
        let meta = PosixMetadata::default();
        builder.add_file(EXT4_ROOT_INO, "f.txt", b"hi", &meta)?;
        builder.commit_metadata();
        let reloaded = Ext4ImageBuilder::from_image(builder.disk_data.to_vec())?;
        let reloaded_compat = reloaded.superblock.s_feature_compat;
        assert_ne!(reloaded_compat & EXT4_FEATURE_COMPAT_HAS_JOURNAL, 0);
        // The journal blocks stay allocated; the reader still sees it empty.
        let (_, state) =
            crate::ext4::reader::ImageReader::from_memory(reloaded.disk_data.to_vec())?;
        assert!(state.journal_empty);
        Ok(())
    }
}
