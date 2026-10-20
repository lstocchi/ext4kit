use super::Ext4ImageBuilder;
use crate::ext4::*;
pub use crate::xattr::parse_xattr_name;
#[cfg(test)]
use crate::xattr::posix_acl_to_ext4_disk;
use crate::xattr::prepare_xattrs;
use anyhow::Result;
use std::collections::{HashMap, HashSet};
use zerocopy::{FromBytes, IntoBytes};

/// Metadata collected from an OCI layer or a host filesystem.
#[derive(Clone, Debug, Default)]
pub struct PosixMetadata {
    pub mode: u16,
    pub uid: u32,
    pub gid: u32,
    pub atime: u32,
    pub ctime: u32,
    pub mtime: u32,
    pub xattrs: HashMap<String, Vec<u8>>,
}

#[derive(Clone, Debug)]
pub(crate) struct PreparedXattr {
    pub(crate) index: u8,
    pub(crate) suffix: String,
    pub(crate) value: Vec<u8>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DirSlot {
    /// An active directory entry with slack space that can be split.
    SplitActiveRecord {
        offset: usize,
        actual: usize,
        rec_len: usize,
    },
    /// An unused/empty entry (inode == 0) that can be reused directly.
    EmptyRecord { offset: usize, rec_len: usize },
}

/// Import-time cache for a linear directory. Htree directories must never use
/// it as their leaf selection and duplicate detection are hash based.
#[derive(Default)]
pub(crate) struct DirectoryCache {
    existing_names: HashSet<String>,
    /// Caches an available slot found in the directory's last block.
    /// Subsequent file additions can reuse this cached slot to skip
    /// re-scanning earlier blocks.
    next_append_slot: Option<DirSlot>,
}

impl Ext4ImageBuilder {
    /// Discards linear-directory import state after callers directly alter its
    /// on-disk data or metadata. Indexed directories do not populate it.
    pub fn invalidate_directory_cache(&mut self, directory: u32) {
        self.directory_caches.remove(&directory);
    }

    /// Validates one ext4 directory-entry component; returns `Ok(())` or a descriptive error.
    pub(crate) fn validate_name(name: &str) -> Result<()> {
        if name.is_empty() || name == "." || name == ".." || name.len() > 255 || name.contains('/')
        {
            Err(anyhow::anyhow!(
                "name must be a non-empty component no longer than 255 bytes"
            ))
        } else {
            Ok(())
        }
    }

    /// Applies POSIX metadata and file-type defaults to an inode; returns no value because it mutates in place.
    pub(crate) fn populate_inode_metadata(
        inode: &mut Ext4Inode,
        m: &PosixMetadata,
        kind: u16,
        default_perms: u16,
    ) {
        let mode = if m.mode == 0 {
            default_perms
        } else {
            m.mode & 0o7777
        };
        inode.i_mode = kind | mode;
        inode.i_uid = m.uid as u16;
        inode.i_gid = m.gid as u16;
        inode.i_osd2[4..6].copy_from_slice(&((m.uid >> 16) as u16).to_le_bytes());
        inode.i_osd2[6..8].copy_from_slice(&((m.gid >> 16) as u16).to_le_bytes());
        inode.i_atime = m.atime;
        inode.i_ctime = m.ctime;
        inode.i_mtime = m.mtime;
        inode.i_extra_isize = 32;
    }

    /// Creates and links a directory below `parent`; returns its allocated inode number.
    pub fn mkdir(&mut self, parent: u32, name: &str, meta: Option<&PosixMetadata>) -> Result<u32> {
        Self::validate_name(name)?;

        let expansion_blocks = self.estimate_dir_expansion_blocks(parent, name)?;

        let default_meta = PosixMetadata::default();
        let meta = meta.unwrap_or(&default_meta);
        let xattrs = prepare_xattrs(&meta.xattrs)?;
        // it finds the next free inode starting from the block of the parent directory, but still do not allocate it
        let ino = self.next_free_inode(Some(parent))?;
        // it gets the group of the free inode found
        let (group, index_in_group) = self.inode_location(ino);
        // it ensures that the group has enough capacity to allocate the new inode
        self.ensure_allocation_capacity(1 + expansion_blocks, u32::from(!xattrs.is_empty()))?;
        // allocate the inode
        self.mark_inode_used(ino);

        let data_block = match self.allocate_free_blocks(group, 1) {
            Ok(block) => block,
            Err(error) => {
                self.release_inode(ino)?;
                return Err(error);
            }
        };
        let mut block = vec![0; BLOCK_SIZE];

        let dot_header = Ext4DirEntry2Header {
            inode: ino,
            rec_len: 12,
            name_len: 1,
            file_type: EXT4_FT_DIR,
        };
        let dot_entry = dot_header.as_bytes();
        block[..8].copy_from_slice(dot_entry);
        block[8] = b'.';

        let dot_dot_header = Ext4DirEntry2Header {
            inode: parent,
            rec_len: (BLOCK_SIZE - 12) as u16,
            name_len: 2,
            file_type: EXT4_FT_DIR,
        };
        let dot_dot_entry = dot_dot_header.as_bytes();
        block[12..20].copy_from_slice(dot_dot_entry);
        block[20..22].copy_from_slice(b"..");

        self.disk_data[data_block as usize * BLOCK_SIZE..(data_block as usize + 1) * BLOCK_SIZE]
            .copy_from_slice(&block);

        let mut inode = Ext4Inode::default();
        inode.i_size_lo = BLOCK_SIZE as u32;
        inode.i_links_count = 2;
        inode.i_blocks_lo = (BLOCK_SIZE / 512) as u32;
        inode.i_flags = EXT4_EXTENTS_FL;

        Self::populate_inode_metadata(&mut inode, meta, S_IFDIR, 0o755);
        Self::init_single_extent(&mut inode, 1, data_block);
        if !xattrs.is_empty() {
            inode.i_file_acl_lo = match self.write_prepared_xattr_block(group, &xattrs) {
                Ok(block) => block,
                Err(error) => {
                    self.release_block(data_block)?;
                    self.release_inode(ino)?;
                    return Err(error);
                }
            };
            inode.i_blocks_lo += (BLOCK_SIZE / 512) as u32;
        }

        self.groups[group].inode_table[index_in_group] = inode;
        self.group_descriptors[group].bg_used_dirs_count_lo += 1;
        // add the new directory entry to the parent
        if let Err(error) = self.add_dir_entry(parent, name, ino, EXT4_FT_DIR) {
            if inode.i_file_acl_lo != 0 {
                self.release_block(inode.i_file_acl_lo)?;
            }
            self.release_block(data_block)?;
            self.release_inode(ino)?;
            self.group_descriptors[group].bg_used_dirs_count_lo -= 1;
            return Err(error);
        }
        let (pgroup, pindex_in_group) = self.inode_location(parent);
        self.groups[pgroup].inode_table[pindex_in_group].i_links_count += 1;
        Ok(ino)
    }

    /// Plans uncommitted physical runs for `blocks_needed`, starting from each
    /// group's allocation cursor before wrapping to reclaimed earlier blocks.
    /// Runs are capped at 32,768 blocks for ext4 initialized extents.
    pub(crate) fn plan_data_blocks(
        &self,
        preferred_group: usize,
        mut blocks_needed: u32,
    ) -> Result<Vec<(u32, u32)>> {
        if blocks_needed == 0 {
            return Ok(Vec::new());
        }

        let free_blocks = self.superblock.s_free_blocks_count_lo;
        if free_blocks < blocks_needed {
            return Err(anyhow::anyhow!(
                "not enough free blocks: needed {blocks_needed}, available {}",
                free_blocks
            ));
        }

        let mut planned_runs = Vec::new();
        let total_groups = self.groups.len();

        for delta in 0..total_groups {
            if blocks_needed == 0 {
                break;
            }
            let group = (preferred_group + delta) % total_groups;
            if self.groups[group].free_blocks_count == 0 {
                continue;
            }

            let limit = self.groups[group].blocks_count as usize;
            let hint = self.next_block_hints[group].min(limit);
            for (range_start, range_end) in [(hint, limit), (0, hint)] {
                let mut bit = range_start;
                while bit < range_end && blocks_needed > 0 {
                    // If block is free in bitmap
                    if self.groups[group].block_bitmap[bit / 8] & (1 << (bit % 8)) == 0 {
                        let run_start = bit;
                        let max_run =
                            (blocks_needed.min(32768) as usize).min(range_end - run_start);
                        let mut run_len = 0;

                        while run_len < max_run
                            && (self.groups[group].block_bitmap[(run_start + run_len) / 8]
                                & (1 << ((run_start + run_len) % 8)))
                                == 0
                        {
                            run_len += 1;
                        }

                        let phys_block = (group * BLOCKS_PER_GROUP + run_start) as u32;
                        planned_runs.push((phys_block, run_len as u32));
                        blocks_needed -= run_len as u32;
                        bit = run_start + run_len;
                    } else {
                        bit += 1;
                    }
                }
            }
        }

        if blocks_needed > 0 {
            return Err(anyhow::anyhow!(
                "insufficient contiguous blocks to fulfill allocation"
            ));
        }

        Ok(planned_runs)
    }

    /// Commits planned runs and advances each affected group's allocation cursor.
    pub(crate) fn commit_data_blocks(&mut self, planned_runs: &[(u32, u32)]) {
        for &(phys_start, len) in planned_runs {
            let group = (phys_start as usize) / BLOCKS_PER_GROUP;
            let local_start = (phys_start as usize) % BLOCKS_PER_GROUP;

            for idx in local_start..local_start + len as usize {
                self.groups[group].block_bitmap[idx / 8] |= 1 << (idx % 8);
            }

            self.groups[group].free_blocks_count -= len as u16;
            self.group_descriptors[group].bg_free_blocks_count_lo -= len as u16;
            self.superblock.s_free_blocks_count_lo -= len;
            self.next_block_hints[group] = local_start + len as usize;
        }
    }

    /// Scans a directory leaf to verify `name` is unique, finding an available slot.
    /// Finds reusable directory-entry space while rejecting duplicate names; returns a slot or `None` when full.
    fn find_dir_slot(&mut self, parent: u32, name: &str) -> Result<Option<DirSlot>> {
        if self.dir_is_indexed(parent)? {
            return self.find_htree_slot(parent, name);
        }

        let name_bytes = name.as_bytes();
        let needed = align4(8 + name_bytes.len());

        if let Some(cache) = self.directory_caches.get(&parent) {
            if cache.existing_names.contains(name) {
                return Err(anyhow::anyhow!("directory entry {name:?} already exists"));
            }
            if let Some(slot) = cache.next_append_slot.filter(|slot| match slot {
                DirSlot::SplitActiveRecord {
                    actual, rec_len, ..
                } => *rec_len >= *actual + needed,
                DirSlot::EmptyRecord { rec_len, .. } => *rec_len >= needed,
            }) {
                return Ok(Some(slot));
            }
            // An exhausted next_append_slot cannot exclude the possibility of an earlier reusable record.
            self.directory_caches.remove(&parent);
        }

        let blocks = self.get_dir_blocks(parent)?;
        let mut first_available_slot = None;
        let mut is_in_last_dir_block = false;
        let mut names = HashSet::new();

        for (block_index, &block) in blocks.iter().enumerate() {
            let base = block as usize * BLOCK_SIZE;
            let mut off = 0;
            while off < BLOCK_SIZE {
                let e = Ext4DirEntry2Header::read_from_bytes(
                    &self.disk_data[base + off..base + off + 8],
                )
                .map_err(|_| anyhow::anyhow!("invalid directory entry header"))?;
                let len = e.rec_len as usize;
                if len < 8
                    || len % 4 != 0
                    || off + len > BLOCK_SIZE
                    || (e.inode != 0 && (e.name_len as usize + 8 > len))
                {
                    return Err(anyhow::anyhow!(
                        "directory inode {parent} has an invalid record"
                    ));
                }

                if e.inode != 0 {
                    let entry_name = std::str::from_utf8(
                        &self.disk_data[base + off + 8..base + off + 8 + e.name_len as usize],
                    )
                    .map_err(|_| {
                        anyhow::anyhow!("directory inode {parent} contains a non-UTF-8 name")
                    })?;
                    names.insert(entry_name.to_owned());
                    if e.name_len as usize == name_bytes.len()
                        && &self.disk_data[base + off + 8..base + off + 8 + e.name_len as usize]
                            == name_bytes
                    {
                        return Err(anyhow::anyhow!("directory entry {name:?} already exists"));
                    }
                    let actual = align4(8 + e.name_len as usize);
                    if first_available_slot.is_none() && len >= actual + needed {
                        first_available_slot = Some(DirSlot::SplitActiveRecord {
                            offset: base + off,
                            actual,
                            rec_len: len,
                        });
                        is_in_last_dir_block = block_index + 1 == blocks.len();
                    }
                } else if first_available_slot.is_none() && len >= needed {
                    first_available_slot = Some(DirSlot::EmptyRecord {
                        offset: base + off,
                        rec_len: len,
                    });
                    is_in_last_dir_block = block_index + 1 == blocks.len();
                }
                off += len;
            }
            if off != BLOCK_SIZE {
                return Err(anyhow::anyhow!(
                    "directory inode {parent} block did not end at block boundary"
                ));
            }
        }
        self.directory_caches.insert(
            parent,
            DirectoryCache {
                existing_names: names,
                next_append_slot: first_available_slot.filter(|_| is_in_last_dir_block),
            },
        );
        Ok(first_available_slot)
    }

    /// Adds one initialized data block to a directory; returns the new physical block number.
    pub(crate) fn expand_dir(&mut self, dir_ino: u32) -> Result<u32> {
        let (group, index_in_group) = self.inode_location(dir_ino);
        let inode = &self.groups[group].inode_table[index_in_group];
        if inode.i_mode & S_IFDIR != S_IFDIR {
            return Err(anyhow::anyhow!("inode {dir_ino} is not a directory"));
        }

        let mut leaf_extents = self.get_inode_extents(inode)?;

        // When write_extent_tree runs later, it serializes a completely new replacement tree.
        // We gather the old tree metadata block numbers so they can be safely reclaimed.
        let mut old_tree_blocks = Vec::new();
        self.collect_extent_tree_blocks(&inode.i_block, &mut old_tree_blocks)?;

        // Calculate the number of extents and tree metadata blocks needed in the worst scenario
        // and check if we have enough free blocks to allocate them.
        let existing_extents = leaf_extents.len();
        let potential_extents = existing_extents + 1;
        let potential_tree_meta = Self::max_extent_tree_blocks(potential_extents) as u32;
        self.ensure_allocation_capacity(1, potential_tree_meta)?;

        // Allocate a new data block and initialize it with an empty entry covering the full block
        let new_blk = self.allocate_free_blocks(group, 1)?;
        let mut empty_dir_block = vec![0u8; BLOCK_SIZE];
        let empty_header = Ext4DirEntry2Header {
            inode: 0,
            rec_len: BLOCK_SIZE as u16,
            name_len: 0,
            file_type: 0,
        };
        empty_dir_block[..8].copy_from_slice(empty_header.as_bytes());
        let disk_offset = new_blk as usize * BLOCK_SIZE;
        self.disk_data[disk_offset..disk_offset + BLOCK_SIZE].copy_from_slice(&empty_dir_block);

        // If the last extent is contiguous with the new block, merge them
        let mut merged = false;
        if let Some(last) = leaf_extents.last_mut() {
            if last.ee_start_lo + (last.ee_len as u32) == new_blk && (last.ee_len as usize) < 32768
            {
                last.ee_len += 1;
                merged = true;
            }
        }
        if !merged {
            let current_size = self.groups[group].inode_table[index_in_group].i_size_lo;
            let logical_block = current_size / BLOCK_SIZE as u32;
            leaf_extents.push(Ext4Extent {
                ee_block: logical_block,
                ee_len: 1,
                ee_start_hi: 0,
                ee_start_lo: new_blk,
            });
        }

        let mut dir_inode = self.groups[group].inode_table[index_in_group];
        let tree_meta_blocks = self.write_extent_tree(group, &mut dir_inode, &leaf_extents)?;

        // The replacement tree is now reachable from the inode; reclaim the
        // old external tree blocks so repeated growth does not leak space.
        for block in old_tree_blocks {
            self.release_block(block)?;
        }

        dir_inode.i_size_lo += BLOCK_SIZE as u32;
        let total_data_blocks = dir_inode.i_size_lo / BLOCK_SIZE as u32;
        let xattr_blocks = u32::from(dir_inode.i_file_acl_lo != 0);
        let total_sectors =
            (total_data_blocks as u64 + tree_meta_blocks as u64 + xattr_blocks as u64)
                * (BLOCK_SIZE as u64 / 512);
        dir_inode.i_blocks_lo = total_sectors as u32;
        let high_sectors = (total_sectors >> 32) as u16;
        dir_inode.i_osd2[0..2].copy_from_slice(&high_sectors.to_le_bytes());

        self.groups[group].inode_table[index_in_group] = dir_inode;
        if !self.dir_is_indexed(dir_ino)? {
            self.directory_caches
                .entry(dir_ino)
                .or_default()
                .next_append_slot = Some(DirSlot::EmptyRecord {
                offset: disk_offset,
                rec_len: BLOCK_SIZE,
            });
        }
        Ok(new_blk)
    }

    /// Adds a directory entry to `parent`, using the linear cache only before
    /// the directory is promoted to an htree.
    pub fn add_dir_entry(
        &mut self,
        parent: u32,
        name: &str,
        child: u32,
        file_type: u8,
    ) -> Result<()> {
        Self::validate_name(name)?;
        let slot = match self.find_dir_slot(parent, name)? {
            Some(slot) => slot,
            None => {
                if !self.dir_is_indexed(parent)? {
                    // The directory is permanently switching lookup models.
                    self.directory_caches.remove(&parent);
                    return self.make_indexed_and_insert(parent, name, child, file_type);
                }
                return self.split_htree_leaf_and_insert(parent, name, child, file_type);
            }
        };
        let is_next_slot_cached = self
            .directory_caches
            .get(&parent)
            .is_some_and(|cache| cache.next_append_slot == Some(slot));
        let bytes = name.as_bytes();
        match slot {
            DirSlot::SplitActiveRecord {
                offset,
                actual,
                rec_len,
            } => {
                self.disk_data[offset + 4..offset + 6]
                    .copy_from_slice(&(actual as u16).to_le_bytes());
                let new_offset = offset + actual;
                let new_rec_len = (rec_len - actual) as u16;
                self.disk_data[new_offset..new_offset + 8].copy_from_slice(
                    Ext4DirEntry2Header {
                        inode: child,
                        rec_len: new_rec_len,
                        name_len: bytes.len() as u8,
                        file_type,
                    }
                    .as_bytes(),
                );
                self.disk_data[new_offset + 8..new_offset + 8 + bytes.len()].copy_from_slice(bytes);
                if let Some(cache) = self.directory_caches.get_mut(&parent) {
                    cache.next_append_slot =
                        is_next_slot_cached.then_some(DirSlot::SplitActiveRecord {
                            offset: new_offset,
                            actual: align4(8 + bytes.len()),
                            rec_len: new_rec_len as usize,
                        });
                }
            }
            DirSlot::EmptyRecord { offset, rec_len } => {
                self.disk_data[offset..offset + 8].copy_from_slice(
                    Ext4DirEntry2Header {
                        inode: child,
                        rec_len: rec_len as u16,
                        name_len: bytes.len() as u8,
                        file_type,
                    }
                    .as_bytes(),
                );
                self.disk_data[offset + 8..offset + 8 + bytes.len()].copy_from_slice(bytes);
                if let Some(cache) = self.directory_caches.get_mut(&parent) {
                    cache.next_append_slot = None;
                }
            }
        }
        if let Some(cache) = self.directory_caches.get_mut(&parent) {
            cache.existing_names.insert(name.to_owned());
        }
        Ok(())
    }

    /// Calculates blocks required to add an entry, including possible growth
    /// and replacement extent-tree metadata.
    pub(crate) fn estimate_dir_expansion_blocks(&mut self, parent: u32, name: &str) -> Result<u32> {
        match self.find_dir_slot(parent, name)? {
            Some(_) => Ok(0),
            None => {
                let (group, index) = self.inode_location(parent);
                let extents = self.get_inode_extents(&self.groups[group].inode_table[index])?;
                let data_blocks: u32 = if self.dir_is_indexed(parent)? {
                    self.htree_growth_blocks(parent, name)?
                } else {
                    // When a directory gets full and needs to grow, it is converted into an htree.
                    // The original block becomes the index root, and its
                    // entries are split across 2 newly allocated leaf blocks.
                    2
                };
                let tree_blocks =
                    Self::max_extent_tree_blocks(extents.len() + data_blocks as usize) as u32;
                data_blocks
                    .checked_add(tree_blocks)
                    .ok_or_else(|| anyhow::anyhow!("directory expansion allocation overflow"))
            }
        }
    }
}

/// Rounds a directory-record size up to four-byte alignment; returns the aligned size.
pub(crate) fn align4(v: usize) -> usize {
    (v + 3) & !3
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fresh_builder() -> Result<Ext4ImageBuilder> {
        let mut builder = Ext4ImageBuilder::new(16);
        builder.create_root_dir()?;
        builder.create_lost_and_found()?;
        builder.reserve_inodes();
        Ok(builder)
    }

    #[test]
    fn file_is_linked_and_uses_contiguous_extent_and_xattrs() {
        let mut builder = fresh_builder().expect("failed to create builder");
        let mut metadata = PosixMetadata {
            mode: 0o640,
            uid: 70_000,
            gid: 80_000,
            mtime: 42,
            ..Default::default()
        };
        metadata
            .xattrs
            .insert("user.note".into(), b"metadata".to_vec());
        let ino = builder
            .add_file(EXT4_ROOT_INO, "payload", &[7; BLOCK_SIZE + 10], &metadata)
            .expect("failed to add file");
        let (group, index) = builder.inode_location(ino);
        let inode = &builder.groups[group].inode_table[index];
        let mode = inode.i_mode;
        let size = inode.i_size_lo;
        let sectors = inode.i_blocks_lo;
        let xattr_block = inode.i_file_acl_lo;
        assert_eq!(mode, S_IFREG | 0o640);
        assert_eq!(size, (BLOCK_SIZE + 10) as u32);
        assert_eq!(sectors, 24); // two payload blocks plus one xattr block, in sectors
        assert_ne!(xattr_block, 0);
        let extent = Ext4Extent::read_from_bytes(&inode.i_block[12..24]).unwrap();
        let extent_len = extent.ee_len;
        let extent_start = extent.ee_start_lo;
        assert_eq!(extent_len, 2);
        assert_eq!(
            &builder.disk_data[extent_start as usize * BLOCK_SIZE
                ..extent_start as usize * BLOCK_SIZE + BLOCK_SIZE + 10],
            &[7; BLOCK_SIZE + 10]
        );
    }

    #[test]
    fn metadata_directory_has_expected_type_and_parent_link() {
        let mut builder = fresh_builder().expect("failed to create builder");
        let metadata = PosixMetadata {
            mode: 0o750,
            uid: 1,
            gid: 2,
            ..Default::default()
        };
        let ino = builder
            .mkdir(EXT4_ROOT_INO, "layers", Some(&metadata))
            .expect("failed to create directory");
        let (group, index) = builder.inode_location(ino);
        let mode = builder.groups[group].inode_table[index].i_mode;
        let links = builder.groups[0].inode_table[1].i_links_count;
        assert_eq!(mode, S_IFDIR | 0o750);
        assert_eq!(links, 4);
    }

    #[test]
    fn failed_preflight_does_not_allocate_inode_or_blocks() {
        let mut builder = fresh_builder().expect("failed to create builder");
        let free_inodes = builder.superblock.s_free_inodes_count;
        let free_blocks = builder.superblock.s_free_blocks_count_lo;
        let mut metadata = PosixMetadata::default();
        metadata.xattrs.insert("invalid.namespace".into(), vec![1]);
        assert!(
            builder
                .add_file(EXT4_ROOT_INO, "bad", b"data", &metadata)
                .is_err()
        );
        let current_free_inodes = builder.superblock.s_free_inodes_count;
        let current_free_blocks = builder.superblock.s_free_blocks_count_lo;
        assert_eq!(current_free_inodes, free_inodes);
        assert_eq!(current_free_blocks, free_blocks);
        assert!(
            builder
                .add_file(0, "bad-parent", b"", &PosixMetadata::default())
                .is_err()
        );
    }

    #[test]
    fn posix_acl_is_converted_to_compact_ext4_encoding() {
        let mut host_acl = Vec::new();
        host_acl.extend_from_slice(&2u32.to_le_bytes());
        for (tag, perm, id) in [(0x01u16, 7u16, 0u32), (0x04, 5, 0), (0x20, 5, 0)] {
            host_acl.extend_from_slice(&tag.to_le_bytes());
            host_acl.extend_from_slice(&perm.to_le_bytes());
            host_acl.extend_from_slice(&id.to_le_bytes());
        }
        let disk_acl = posix_acl_to_ext4_disk(&host_acl).unwrap();
        assert_eq!(
            disk_acl,
            vec![1, 0, 0, 0, 1, 0, 7, 0, 4, 0, 5, 0, 32, 0, 5, 0]
        );
    }

    #[test]
    fn symlinks_use_fast_and_slow_storage_at_e2fsprogs_boundary() {
        let mut builder = fresh_builder().expect("failed to create builder");
        let metadata = PosixMetadata::default();

        let fast_ino = builder
            .add_symlink(EXT4_ROOT_INO, "fast", &"f".repeat(59), &metadata)
            .expect("failed to create fast symlink");
        let (fast_group, fast_index) = builder.inode_location(fast_ino);
        let fast = &builder.groups[fast_group].inode_table[fast_index];
        let (fast_mode, fast_size, fast_blocks) = (fast.i_mode, fast.i_size_lo, fast.i_blocks_lo);
        assert_eq!(fast_mode, S_IFLNK | 0o777);
        assert_eq!(fast_size, 59);
        assert_eq!(fast_blocks, 0);
        assert_eq!(
            &fast.i_block[..59],
            b"fffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
        );

        let target = "s".repeat(60);
        let slow_ino = builder
            .add_symlink(EXT4_ROOT_INO, "slow", &target, &metadata)
            .expect("failed to create slow symlink");
        let (slow_group, slow_index) = builder.inode_location(slow_ino);
        let slow = &builder.groups[slow_group].inode_table[slow_index];
        let extent = Ext4Extent::read_from_bytes(&slow.i_block[12..24]).unwrap();
        let block_offset = extent.ee_start_lo as usize * BLOCK_SIZE;
        let (slow_size, slow_blocks, extent_len) =
            (slow.i_size_lo, slow.i_blocks_lo, extent.ee_len);
        assert_eq!(slow_size, 60);
        assert_eq!(slow_blocks, (BLOCK_SIZE / 512) as u32);
        assert_eq!(extent_len, 1);
        assert_eq!(
            &builder.disk_data[block_offset..block_offset + 60],
            target.as_bytes()
        );
        assert!(
            builder.disk_data[block_offset + 60..block_offset + BLOCK_SIZE]
                .iter()
                .all(|byte| *byte == 0)
        );
    }

    #[test]
    fn symlink_target_may_be_empty_but_must_leave_room_for_nul() {
        let mut builder = fresh_builder().expect("failed to create builder");
        let metadata = PosixMetadata::default();
        let free_inodes = builder.superblock.s_free_inodes_count;

        let ino = builder
            .add_symlink(EXT4_ROOT_INO, "empty", "", &metadata)
            .expect("e2fsprogs permits an empty symlink target");
        let (group, index) = builder.inode_location(ino);
        let empty_size = builder.groups[group].inode_table[index].i_size_lo;
        assert_eq!(empty_size, 0);

        assert!(
            builder
                .add_symlink(
                    EXT4_ROOT_INO,
                    "too-long",
                    &"x".repeat(BLOCK_SIZE),
                    &metadata,
                )
                .is_err()
        );
        let remaining_inodes = builder.superblock.s_free_inodes_count;
        assert_eq!(remaining_inodes, free_inodes - 1);
    }

    #[test]
    fn char_and_block_device_nodes_encode_rdev() {
        let mut builder = fresh_builder().expect("failed to create builder");
        let metadata = PosixMetadata {
            mode: 0o666,
            ..Default::default()
        };

        // Character device with standard minor/major (< 256): /dev/null (1, 3)
        let chr_ino = builder
            .add_char_device(EXT4_ROOT_INO, "null", 1, 3, &metadata)
            .expect("failed to create char device");
        let (group, idx) = builder.inode_location(chr_ino);
        let chr = &builder.groups[group].inode_table[idx];

        let chr_mode = chr.i_mode;
        let chr_size = chr.i_size_lo;
        let chr_blocks = chr.i_blocks_lo;
        let chr_flags = chr.i_flags;
        let chr_block = chr.i_block;
        assert_eq!(chr_mode, S_IFCHR | 0o666);
        assert_eq!(chr_size, 0);
        assert_eq!(chr_blocks, 0);
        assert_eq!(chr_flags, 0); // Must NOT have EXT4_EXTENTS_FL
        let dev_lo = u32::from_le_bytes(chr_block[..4].try_into().unwrap());
        assert_eq!(dev_lo, (1 << 8) | 3);

        // Block device with high major/minor (>= 256): (259, 1)
        let blk_ino = builder
            .add_block_device(EXT4_ROOT_INO, "nvme0n1", 259, 1, &metadata)
            .expect("failed to create block device");
        let (b_group, b_idx) = builder.inode_location(blk_ino);
        let blk = &builder.groups[b_group].inode_table[b_idx];

        let blk_mode = blk.i_mode;
        let blk_size = blk.i_size_lo;
        let blk_blocks = blk.i_blocks_lo;
        let blk_flags = blk.i_flags;
        assert_eq!(blk_mode, S_IFBLK | 0o666);
        assert_eq!(blk_size, 0);
        assert_eq!(blk_blocks, 0);
        assert_eq!(blk_flags, 0);
        // High device numbers leave i_block[0] as 0 and set i_block[1]
        let dev_old = u32::from_le_bytes(blk.i_block[..4].try_into().unwrap());
        let dev_new = u32::from_le_bytes(blk.i_block[4..8].try_into().unwrap());
        assert_eq!(dev_old, 0);
        let expected_new = (1 & 0xff) | (259 << 8) | ((1 & !0xff) << 12);
        assert_eq!(dev_new, expected_new);
    }

    #[test]
    fn fifo_and_socket_nodes() {
        let mut builder = fresh_builder().expect("failed to create builder");
        let metadata = PosixMetadata::default();

        let fifo_ino = builder
            .add_fifo(EXT4_ROOT_INO, "test_pipe", &metadata)
            .expect("failed to create fifo");
        let (f_grp, f_idx) = builder.inode_location(fifo_ino);
        let fifo = &builder.groups[f_grp].inode_table[f_idx];

        let fifo_mode = fifo.i_mode;
        let fifo_size = fifo.i_size_lo;
        let fifo_blocks = fifo.i_blocks_lo;
        let fifo_flags = fifo.i_flags;
        let fifo_block = fifo.i_block;
        assert_eq!(fifo_mode, S_IFIFO | 0o644);
        assert_eq!(fifo_size, 0);
        assert_eq!(fifo_blocks, 0);
        assert_eq!(fifo_flags, 0);
        assert_eq!(fifo_block, [0u8; 60]);

        let sock_ino = builder
            .add_socket(EXT4_ROOT_INO, "test.sock", &metadata)
            .expect("failed to create socket");
        let (s_grp, s_idx) = builder.inode_location(sock_ino);
        let sock = &builder.groups[s_grp].inode_table[s_idx];

        let sock_mode = sock.i_mode;
        let sock_size = sock.i_size_lo;
        let sock_blocks = sock.i_blocks_lo;
        let sock_flags = sock.i_flags;
        let sock_block = sock.i_block;
        assert_eq!(sock_mode, S_IFSOCK | 0o777);
        assert_eq!(sock_size, 0);
        assert_eq!(sock_blocks, 0);
        assert_eq!(sock_flags, 0);
        assert_eq!(sock_block, [0u8; 60]);
    }

    #[test]
    fn device_numbers_must_fit_linux_dev_t() {
        assert!(Ext4ImageBuilder::encode_device_number(0x1000, 0).is_err());
        assert!(Ext4ImageBuilder::encode_device_number(0, 0x10_0000).is_err());
    }

    #[test]
    fn multi_extent_tree_depth_1_verification() {
        let mut builder = fresh_builder().expect("failed to create builder");
        let meta = PosixMetadata::default();

        // Artificially fragment free blocks to force >= 5 extents
        // Group 0 has some free blocks. We allocate single blocks with gaps.
        let mut fragmented_blocks = Vec::new();
        for _ in 0..5 {
            let blk = builder.allocate_free_blocks(0, 1).unwrap();
            fragmented_blocks.push(blk);
            let _gap = builder.allocate_free_blocks(0, 1).unwrap();
        }

        // Free the 5 blocks we want to use as fragmented targets
        for &blk in &fragmented_blocks {
            builder.release_block(blk).unwrap();
        }

        // Allocate a 5-block file that will consume the 5 isolated gaps
        let file_data = vec![0xABu8; 5 * BLOCK_SIZE];
        let ino = builder
            .add_file(EXT4_ROOT_INO, "fragmented.bin", &file_data, &meta)
            .expect("should create fragmented file");

        let (grp, idx) = builder.inode_location(ino);
        let inode = &builder.groups[grp].inode_table[idx];

        // Root inode must now be at depth 1 with 1 index entry
        let root_hdr = Ext4ExtentHeader::read_from_bytes(&inode.i_block[..12]).unwrap();

        let root_magic = root_hdr.eh_magic;
        let root_depth = root_hdr.eh_depth;
        let root_entries = root_hdr.eh_entries;
        let root_max = root_hdr.eh_max;
        assert_eq!(root_magic, EXT4_EH_MAGIC);
        assert_eq!(root_depth, 1);
        assert_eq!(root_entries, 1);
        assert_eq!(root_max, 4);

        let root_idx = Ext4ExtentIdx::read_from_bytes(&inode.i_block[12..24]).unwrap();
        let root_index = root_idx.ei_block;
        assert_eq!(root_index, 0);

        // Inspect the external leaf block pointed to by ei_leaf_lo
        let leaf_blk_offset = root_idx.ei_leaf_lo as usize * BLOCK_SIZE;
        let leaf_hdr = Ext4ExtentHeader::read_from_bytes(
            &builder.disk_data[leaf_blk_offset..leaf_blk_offset + 12],
        )
        .unwrap();
        let leaf_magic = leaf_hdr.eh_magic;
        let leaf_depth = leaf_hdr.eh_depth;
        let leaf_entries = leaf_hdr.eh_entries;
        let leaf_max = leaf_hdr.eh_max;
        assert_eq!(leaf_magic, EXT4_EH_MAGIC);
        assert_eq!(leaf_depth, 0);
        assert_eq!(leaf_entries, 5);
        assert_eq!(leaf_max, 340);

        // Accounted sectors: 5 data blocks + 1 extent tree block = 6 blocks * 8 sectors = 48
        let inode_blocks_lo = inode.i_blocks_lo;
        assert_eq!(inode_blocks_lo, 48);
    }

    #[test]
    fn directory_expands_when_first_block_fills_up() {
        let mut builder = fresh_builder().expect("failed to create builder");
        let test_dir = builder
            .mkdir(EXT4_ROOT_INO, "overflow_dir", None)
            .expect("failed to create directory");

        let (g, i) = builder.inode_location(test_dir);
        let dir_inode = &builder.groups[g].inode_table[i];
        let size_lo = u32::from_le_bytes(dir_inode.i_size_lo.to_le_bytes());
        assert_eq!(size_lo, BLOCK_SIZE as u32);
        let blocks_lo = u32::from_le_bytes(dir_inode.i_blocks_lo.to_le_bytes());
        assert_eq!(blocks_lo, 8);

        // With 12-char names (8-byte header + 12 name bytes = 20 bytes aligned),
        // 4096 - 24 bytes for "." and ".." leaves 4072 bytes.
        // 4072 / 20 = 203 files. File #204 exceeds block 0 and triggers expansion.
        let meta = PosixMetadata::default();
        for file_idx in 0..220 {
            let fname = format!("file_{:07}", file_idx);
            builder
                .add_file(test_dir, &fname, b"content", &meta)
                .unwrap_or_else(|e| panic!("failed to add {fname}: {e}"));
        }

        let dir_inode = &builder.groups[g].inode_table[i];
        let size_lo = u32::from_le_bytes(dir_inode.i_size_lo.to_le_bytes());
        // A full linear block is converted into one dx root and two hashed
        // leaves, following e2fsprogs' initial htree split layout.
        assert_eq!(size_lo, (3 * BLOCK_SIZE) as u32);
        let blocks_lo = u32::from_le_bytes(dir_inode.i_blocks_lo.to_le_bytes());
        assert_eq!(blocks_lo, 24);
        assert_ne!(dir_inode.i_flags & EXT4_INDEX_FL, 0);
        let entries = builder.read_dx_entries(test_dir).unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0], (0, 1));

        // Duplicate detection must work across both block 0 and block 1
        assert!(
            builder
                .add_file(test_dir, "file_0000000", b"x", &meta)
                .is_err()
        );
        assert!(
            builder
                .add_file(test_dir, "file_0000219", b"x", &meta)
                .is_err()
        );
    }

    #[test]
    fn fragmented_directory_growth_reclaims_replaced_extent_trees() {
        let mut builder = fresh_builder().expect("failed to create builder");
        let directory = builder
            .mkdir(EXT4_ROOT_INO, "fragmented_dir", None)
            .expect("failed to create directory");
        let free_before = builder.superblock.s_free_blocks_count_lo;
        let meta = PosixMetadata::default();

        // File data interleaves directory expansion, producing one directory
        // extent per block. Six blocks force both creation and replacement of
        // an external extent tree.
        let entry_count = 203 + 4 * 204 + 1;
        for file_idx in 0..entry_count {
            builder
                .add_file(directory, &format!("file_{file_idx:07}"), b"content", &meta)
                .unwrap_or_else(|e| panic!("failed to add file {file_idx}: {e}"));
        }

        let (group, index) = builder.inode_location(directory);
        let inode = &builder.groups[group].inode_table[index];
        let inode_size = inode.i_size_lo;
        assert_eq!(inode_size as usize % BLOCK_SIZE, 0);
        assert_ne!(inode.i_flags & EXT4_INDEX_FL, 0);
        let data_blocks = inode_size as usize / BLOCK_SIZE;
        assert!(data_blocks >= 3);
        // Directory data blocks plus the live extent-tree metadata block.
        let inode_blocks = inode.i_blocks_lo;
        let tree_blocks = Ext4ImageBuilder::max_extent_tree_blocks(
            builder.get_inode_extents(inode).unwrap().len(),
        );
        assert_eq!(
            inode_blocks,
            ((data_blocks + tree_blocks) * (BLOCK_SIZE / 512)) as u32
        );
        // Every file uses one data block; directory data and its live extent
        // metadata account for the rest (no replacement-tree leaks).
        assert_eq!(
            free_before - builder.superblock.s_free_blocks_count_lo,
            entry_count as u32 + (data_blocks - 1) as u32 + tree_blocks as u32
        );
    }

    #[test]
    fn directory_growth_preflight_reserves_extent_tree_metadata() {
        let mut builder = fresh_builder().expect("failed to create builder");
        let directory = builder
            .mkdir(EXT4_ROOT_INO, "preflight_dir", None)
            .expect("failed to create directory");
        let meta = PosixMetadata::default();

        // Fill four fragmented blocks. The next insertion needs a fifth data
        // block and the first external extent-tree block.
        for file_idx in 0..(203 + 3 * 204) {
            builder
                .add_file(directory, &format!("file_{file_idx:07}"), b"content", &meta)
                .unwrap();
        }
        let (group, index) = builder.inode_location(directory);
        let size_before = builder.groups[group].inode_table[index].i_size_lo;
        let free_inodes_before = builder.superblock.s_free_inodes_count;

        // Model an almost-full filesystem: insufficient for the child data,
        // directory block, and tree metadata. The failure must occur before
        // the child inode or either directory allocation is committed.
        builder.superblock.s_free_blocks_count_lo = 0;
        assert!(
            builder
                .add_file(directory, "requires_tree", b"content", &meta)
                .is_err()
        );
        let free_inodes_after = builder.superblock.s_free_inodes_count;
        let free_blocks_after = builder.superblock.s_free_blocks_count_lo;
        assert_eq!(free_inodes_after, free_inodes_before);
        assert_eq!(free_blocks_after, 0);
        let size_after = builder.groups[group].inode_table[index].i_size_lo;
        assert_eq!(size_after, size_before);
    }

    #[test]
    fn htree_lookup_follows_collision_continuation_leaf() {
        let mut builder = fresh_builder().unwrap();
        let directory = builder.mkdir(EXT4_ROOT_INO, "collisions", None).unwrap();
        let meta = PosixMetadata::default();
        for file_idx in 0..220 {
            builder
                .add_file(directory, &format!("file_{file_idx:07}"), b"", &meta)
                .unwrap();
        }

        let blocks = builder.get_dir_blocks(directory).unwrap();
        let name = "collision-target";
        let hash = builder.dir_hash(name.as_bytes());
        builder
            .write_dir_leaf(
                blocks[1],
                &[super::super::htree::HtreeEntry {
                    hash: 0,
                    inode: 100,
                    file_type: EXT4_FT_REG_FILE,
                    name: b"other".to_vec(),
                }],
            )
            .unwrap();
        builder
            .write_dir_leaf(
                blocks[2],
                &[super::super::htree::HtreeEntry {
                    hash,
                    inode: 101,
                    file_type: EXT4_FT_REG_FILE,
                    name: name.as_bytes().to_vec(),
                }],
            )
            .unwrap();
        builder
            .write_dx_root(directory, EXT4_ROOT_INO, 0, &[(0, 1), (hash | 1, 2)])
            .unwrap();

        assert!(builder.find_htree_slot(directory, name).is_err());
    }

    #[test]
    fn full_dx_root_is_promoted_to_indirect_node() {
        let mut builder = fresh_builder().unwrap();
        let directory = builder.mkdir(EXT4_ROOT_INO, "promotion", None).unwrap();
        let meta = PosixMetadata::default();
        for file_idx in 0..220 {
            builder
                .add_file(directory, &format!("file_{file_idx:07}"), b"", &meta)
                .unwrap();
        }

        let blocks = builder.get_dir_blocks(directory).unwrap();
        let full_leaf: Vec<_> = (0..204)
            .map(|i| {
                let name = format!("e{i:011}").into_bytes();
                super::super::htree::HtreeEntry {
                    hash: builder.dir_hash(&name),
                    inode: 100 + i,
                    file_type: EXT4_FT_REG_FILE,
                    name,
                }
            })
            .collect();
        builder.write_dir_leaf(blocks[1], &full_leaf).unwrap();
        let root_entries = vec![(0, 1); (BLOCK_SIZE - 32) / 8];
        builder
            .write_dx_root(directory, EXT4_ROOT_INO, 0, &root_entries)
            .unwrap();

        builder
            .split_htree_leaf_and_insert(directory, "new_entry", 999, EXT4_FT_REG_FILE)
            .unwrap();

        let root_block = builder.get_dir_blocks(directory).unwrap()[0] as usize * BLOCK_SIZE;
        assert_eq!(builder.disk_data[root_block + 30], 1);
        let root = builder.read_dx_entries(directory).unwrap();
        assert_eq!(root, vec![(0, 4)]);
    }
}
