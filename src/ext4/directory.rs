//! Directories: entry records, lookup, insertion and removal, growth and
//! the linear-directory import cache.

use super::Ext4ImageBuilder;
use super::extent::ExtentTree;
use super::inode::PosixMetadata;
use crate::ext4::format::xattr::prepare_xattrs;
use crate::ext4::format::*;
use anyhow::Result;
use std::collections::{HashMap, HashSet};
use zerocopy::{FromBytes, IntoBytes};

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

/// One directory record validated by [`Ext4ImageBuilder::dir_block_records`].
#[derive(Clone, Copy, Debug)]
pub(crate) struct DirRecord {
    /// Absolute byte offset of the record in the image.
    pub(crate) offset: usize,
    pub(crate) inode: u32,
    pub(crate) rec_len: usize,
    pub(crate) name_len: usize,
    pub(crate) file_type: u8,
}

/// Inodes and blocks released by a recursive removal, validated up front so
/// that committing it cannot fail halfway.
#[derive(Default)]
struct RemovalPlan {
    /// Surviving non-directory inodes and how many of their links are removed.
    unlinked: Vec<(u32, u16)>,
    /// Inodes whose final link is removed, in postorder.
    released: Vec<u32>,
    /// Physical block runs owned by the released inodes.
    runs: Vec<(u32, u32)>,
}

/// Returns the directory-entry file type for an inode type.
pub(crate) fn dir_entry_file_type(kind: u16) -> Option<u8> {
    Some(match kind {
        S_IFREG => EXT4_FT_REG_FILE,
        S_IFDIR => EXT4_FT_DIR,
        S_IFLNK => EXT4_FT_SYMLINK,
        S_IFCHR => EXT4_FT_CHRDEV,
        S_IFBLK => EXT4_FT_BLKDEV,
        S_IFIFO => EXT4_FT_FIFO,
        S_IFSOCK => EXT4_FT_SOCK,
        _ => return None,
    })
}

impl Ext4ImageBuilder {
    /// Parses one directory block, rejecting records that
    /// ext2fs_process_dir_block() treats as EXT2_ET_DIR_CORRUPTED.
    pub(crate) fn dir_block_records(&self, block: u32) -> Result<Vec<DirRecord>> {
        let base = block as usize * BLOCK_SIZE;
        let mut records = Vec::new();
        let mut off = 0;
        while off < BLOCK_SIZE {
            let header = self
                .disk_data
                .get(base + off..base + off + 8)
                .filter(|_| off + 8 <= BLOCK_SIZE)
                .and_then(|bytes| Ext4DirEntry2Header::read_from_bytes(bytes).ok())
                .ok_or_else(|| anyhow::anyhow!("directory block {block} is truncated"))?;
            let rec_len = header.rec_len as usize;
            let name_len = header.name_len as usize;
            if rec_len < 8
                || !rec_len.is_multiple_of(4)
                || off + rec_len > BLOCK_SIZE
                || name_len + 8 > rec_len
            {
                return Err(anyhow::anyhow!(
                    "directory block {block} has an invalid record"
                ));
            }
            records.push(DirRecord {
                offset: base + off,
                inode: header.inode,
                rec_len,
                name_len,
                file_type: header.file_type,
            });
            off += rec_len;
        }
        Ok(records)
    }

    /// Returns the name bytes of a parsed directory record.
    pub(crate) fn record_name(&self, record: &DirRecord) -> &[u8] {
        &self.disk_data[record.offset + 8..record.offset + 8 + record.name_len]
    }

    /// Finds the live record for `name`, returning its block and record.
    /// Indexed directories only scan the hash-selected leaf chain; dx root and
    /// node blocks never hold ordinary names.
    fn find_dir_record(&self, parent: u32, name: &str) -> Result<Option<(u32, DirRecord)>> {
        Self::validate_name(name)?;
        let blocks = if self.dir_is_indexed(parent)? {
            self.htree_candidate_blocks(parent, self.dir_hash(name.as_bytes()))?
        } else {
            self.get_dir_blocks(parent)?
        };
        for block in blocks {
            for record in self.dir_block_records(block)? {
                if record.inode != 0 && self.record_name(&record) == name.as_bytes() {
                    return Ok(Some((block, record)));
                }
            }
        }
        Ok(None)
    }

    /// Looks up a child inode by name without changing the directory.
    pub fn lookup_dir_entry(&self, parent: u32, name: &str) -> Result<Option<u32>> {
        Ok(self
            .find_dir_record(parent, name)?
            .map(|(_, record)| record.inode))
    }

    /// Lists ordinary child entries as `(name, inode)`, excluding `.` and `..`.
    /// Every block is scanned: dx root and node blocks parse as ordinary
    /// directory blocks whose only live records are `.` and `..`.
    pub fn directory_entries(&self, directory: u32) -> Result<Vec<(String, u32)>> {
        let mut children = Vec::new();
        for block in self.get_dir_blocks(directory)? {
            for record in self.dir_block_records(block)? {
                if record.inode == 0 {
                    continue;
                }
                let name = std::str::from_utf8(self.record_name(&record)).map_err(|_| {
                    anyhow::anyhow!("directory inode {directory} contains a non-UTF-8 name")
                })?;
                if name != "." && name != ".." {
                    children.push((name.to_owned(), record.inode));
                }
            }
        }
        Ok(children)
    }

    /// Adds another name for an existing non-directory inode.
    pub fn add_hard_link(&mut self, parent: u32, name: &str, inode_number: u32) -> Result<()> {
        Self::validate_name(name)?;
        if !self.is_inode_allocated(inode_number) {
            return Err(anyhow::anyhow!("inode {inode_number} does not exist"));
        }
        let inode = self.inode(inode_number);
        if inode.is_dir() {
            return Err(anyhow::anyhow!(
                "hard links to directories are not supported"
            ));
        }
        let file_type = dir_entry_file_type(inode.file_type())
            .ok_or_else(|| anyhow::anyhow!("inode {inode_number} has an unsupported type"))?;
        if inode.i_links_count >= EXT4_LINK_MAX {
            return Err(anyhow::anyhow!("inode {inode_number} has too many links"));
        }
        // Like ext2fs_link() callers, link first and only then account for
        // the new name, so a failed insertion leaves the count untouched.
        let expansion_blocks = self.estimate_dir_expansion_blocks(parent, name)?;
        self.ensure_allocation_capacity(expansion_blocks, 0)?;
        self.add_dir_entry(parent, name, inode_number, file_type)?;
        self.inode_mut(inode_number).i_links_count += 1;
        Ok(())
    }

    /// ext2fs_inc_nlink(): a directory with more than EXT4_LINK_MAX
    /// subdirectories is pinned at one link (dir_nlink).
    pub(crate) fn inc_dir_nlink(&mut self, directory: u32) {
        let inode = self.inode_mut(directory);
        if inode.i_links_count == 1 || inode.i_links_count >= EXT4_LINK_MAX {
            inode.i_links_count = 1;
        } else {
            inode.i_links_count += 1;
        }
    }

    /// ext2fs_dec_nlink(): a directory pinned at one link stays pinned.
    fn dec_dir_nlink(&mut self, directory: u32) {
        let inode = self.inode_mut(directory);
        if inode.i_links_count > 2 {
            inode.i_links_count -= 1;
        }
    }

    /// Removes a named entry and recursively releases the inode and its blocks.
    /// Returns `false` when no such entry exists.
    pub fn remove_dir_entry(&mut self, parent: u32, name: &str) -> Result<bool> {
        let Some((block, record)) = self.find_dir_record(parent, name)? else {
            return Ok(false);
        };
        // Validate the whole subtree before mutating anything, so malformed
        // deep children leave the tree untouched when removal fails.
        let plan = self.plan_removal(record.inode)?;
        let removes_directory = self.inode(record.inode).is_dir();

        // Like debugfs kill_file_by_inode(): unlink the top-level name only;
        // records inside released directories vanish with their blocks.
        self.unlink_record(parent, block, record.offset)?;
        if removes_directory {
            self.dec_dir_nlink(parent);
        }
        for (inode, count) in plan.unlinked {
            self.inode_mut(inode).i_links_count -= count;
        }
        for (start, len) in plan.runs {
            self.release_blocks(start, len)
                .expect("removal plan validated every released block");
        }
        for inode in plan.released {
            if self.inode(inode).is_dir() {
                let (group, _) = self.inode_location(inode);
                self.group_descriptors[group].bg_used_dirs_count_lo -= 1;
                self.invalidate_directory_cache(inode);
            }
            self.release_inode(inode)
                .expect("removal plan validated every released inode");
        }
        Ok(true)
    }

    /// Collects the inodes, link counts and block runs released by removing
    /// one link to `root`, without changing the filesystem.
    fn plan_removal(&self, root: u32) -> Result<RemovalPlan> {
        let mut non_dir_link_counts = HashMap::<u32, usize>::new();
        let mut all_inodes_postorder = Vec::new();
        let mut visited_directories = HashSet::new();
        self.collect_removal_subtree(
            root,
            &mut non_dir_link_counts,
            &mut all_inodes_postorder,
            &mut visited_directories,
        )?;

        let mut plan = RemovalPlan::default();
        let mut removed_dirs = vec![0usize; self.groups.len()];
        for inode_number in all_inodes_postorder {
            let inode = self.inode(inode_number);
            if inode.is_dir() {
                removed_dirs[self.inode_location(inode_number).0] += 1;
            } else {
                let removed = non_dir_link_counts[&inode_number];
                let links = usize::from(inode.i_links_count);
                if removed > links {
                    return Err(anyhow::anyhow!(
                        "inode {inode_number} has fewer links than entries being removed"
                    ));
                }
                if removed < links {
                    plan.unlinked.push((inode_number, removed as u16));
                    continue;
                }
            }
            plan.runs.extend(self.inode_storage_runs(inode)?);
            plan.released.push(inode_number);
        }
        for (group, &count) in removed_dirs.iter().enumerate() {
            if count > usize::from(self.group_descriptors[group].bg_used_dirs_count_lo) {
                return Err(anyhow::anyhow!(
                    "block group {group} has an inconsistent directory count"
                ));
            }
        }

        plan.runs.sort_unstable();
        for (index, &(start, len)) in plan.runs.iter().enumerate() {
            self.validate_allocated_run(start, len)?;
            if let Some(&(next, _)) = plan.runs.get(index + 1)
                && u64::from(start) + u64::from(len) > u64::from(next)
            {
                return Err(anyhow::anyhow!(
                    "multiple inodes reference block {next} being released"
                ));
            }
        }
        Ok(plan)
    }

    /// Postorder walk of the subtree below `inode_number`, counting removed
    /// links to non-directories and recording every inode once.
    fn collect_removal_subtree(
        &self,
        inode_number: u32,
        link_removals: &mut HashMap<u32, usize>,
        order: &mut Vec<u32>,
        visited: &mut HashSet<u32>,
    ) -> Result<()> {
        if !self.is_inode_allocated(inode_number) {
            return Err(anyhow::anyhow!("inode {inode_number} does not exist"));
        }
        if !self.inode(inode_number).is_dir() {
            let count = link_removals.entry(inode_number).or_default();
            if *count == 0 {
                order.push(inode_number);
            }
            *count += 1;
            return Ok(());
        }
        if !visited.insert(inode_number) {
            return Err(anyhow::anyhow!(
                "directory inode {inode_number} appears more than once in the removal tree"
            ));
        }
        for (_, child) in self.directory_entries(inode_number)? {
            self.collect_removal_subtree(child, link_removals, order, visited)?;
        }
        order.push(inode_number);
        Ok(())
    }

    /// Removes the record at absolute `offset` in directory `block` exactly as
    /// e2fsprogs unlink_proc() does: merge it into the immediately preceding
    /// record, or clear its inode when it is the first record of the block.
    fn unlink_record(&mut self, parent: u32, block: u32, offset: usize) -> Result<()> {
        let records = self.dir_block_records(block)?;
        let index = records
            .iter()
            .position(|record| record.offset == offset)
            .ok_or_else(|| anyhow::anyhow!("directory record disappeared before unlink"))?;
        if index == 0 {
            self.disk_data[offset..offset + 4].fill(0);
        } else {
            let previous = records[index - 1];
            let merged = (previous.rec_len + records[index].rec_len) as u16;
            self.disk_data[previous.offset + 4..previous.offset + 6]
                .copy_from_slice(&merged.to_le_bytes());
        }
        self.invalidate_directory_cache(parent);
        Ok(())
    }

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

        let mut inode = Ext4Inode {
            i_size_lo: BLOCK_SIZE as u32,
            i_links_count: 2,
            i_blocks_lo: (BLOCK_SIZE / 512) as u32,
            i_flags: EXT4_EXTENTS_FL,
            ..Default::default()
        };

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
        self.inc_dir_nlink(parent);
        Ok(ino)
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
            for record in self.dir_block_records(block)? {
                let entry_name = self.record_name(&record);
                if record.inode != 0 {
                    if entry_name == name_bytes {
                        return Err(anyhow::anyhow!("directory entry {name:?} already exists"));
                    }
                    let entry_name = std::str::from_utf8(entry_name).map_err(|_| {
                        anyhow::anyhow!("directory inode {parent} contains a non-UTF-8 name")
                    })?;
                    names.insert(entry_name.to_owned());
                    let actual = align4(8 + record.name_len);
                    if first_available_slot.is_none() && record.rec_len >= actual + needed {
                        first_available_slot = Some(DirSlot::SplitActiveRecord {
                            offset: record.offset,
                            actual,
                            rec_len: record.rec_len,
                        });
                        is_in_last_dir_block = block_index + 1 == blocks.len();
                    }
                } else if first_available_slot.is_none() && record.rec_len >= needed {
                    first_available_slot = Some(DirSlot::EmptyRecord {
                        offset: record.offset,
                        rec_len: record.rec_len,
                    });
                    is_in_last_dir_block = block_index + 1 == blocks.len();
                }
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
        if !inode.is_dir() {
            return Err(anyhow::anyhow!("inode {dir_ino} is not a directory"));
        }

        // When write_extent_tree runs later, it serializes a completely new replacement tree.
        // We gather the old tree metadata block numbers so they can be safely reclaimed.
        let ExtentTree {
            leaves: mut leaf_extents,
            tree_blocks: old_tree_blocks,
        } = self.read_extent_tree(inode)?;

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
        if let Some(last) = leaf_extents.last_mut()
            && last.ee_start_lo + (last.ee_len as u32) == new_blk
            && (last.ee_len as usize) < 32768
        {
            last.ee_len += 1;
            merged = true;
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
                    // When a single-block linear directory gets full, it is
                    // converted into an htree: the original block becomes the
                    // index root, and its entries are split across 2 newly
                    // allocated leaf blocks.  A multi-block linear directory
                    // (only `lost+found` gets that layout) cannot grow, so
                    // fail here, before anything is committed.
                    if self.get_dir_blocks(parent)?.len() != 1 {
                        anyhow::bail!(
                            "directory inode {parent} is full and cannot grow: \
                             cannot convert a multi-block linear directory to htree"
                        );
                    }
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
    use crate::ext4::test_support::fresh_builder;

    #[test]
    fn failed_deep_preflight_leaves_recursive_removal_unchanged() -> Result<()> {
        let mut builder = fresh_builder()?;
        let outer = builder.mkdir(EXT4_ROOT_INO, "outer", None)?;
        let early_file = builder.add_file(
            outer,
            "a-file",
            b"keep until removal commits",
            &PosixMetadata::default(),
        )?;
        let deep = builder.mkdir(outer, "z-deep", None)?;
        builder.add_file(
            deep,
            "nested-file",
            b"invalid directory record follows",
            &PosixMetadata::default(),
        )?;
        builder
            .directory_caches
            .insert(outer, DirectoryCache::default());
        builder
            .directory_caches
            .insert(deep, DirectoryCache::default());

        let free_blocks = builder.superblock.s_free_blocks_count_lo;
        let free_inodes = builder.superblock.s_free_inodes_count;
        let root_links = builder.groups[0].inode_table[1].i_links_count;
        let deep_block = builder.get_dir_blocks(deep)?[0] as usize;
        let deep_offset = deep_block * BLOCK_SIZE;
        builder.disk_data[deep_offset + 4..deep_offset + 6].copy_from_slice(&0u16.to_le_bytes());

        assert!(builder.remove_dir_entry(EXT4_ROOT_INO, "outer").is_err());
        assert_eq!(
            builder.lookup_dir_entry(EXT4_ROOT_INO, "outer")?,
            Some(outer)
        );
        assert!(builder.is_inode_allocated(outer));
        assert!(builder.is_inode_allocated(early_file));
        assert!(builder.is_inode_allocated(deep));
        assert_eq!({ builder.superblock.s_free_blocks_count_lo }, free_blocks);
        assert_eq!({ builder.superblock.s_free_inodes_count }, free_inodes);
        assert_eq!({ builder.inode(EXT4_ROOT_INO).i_links_count }, root_links);
        assert!(builder.directory_caches.contains_key(&outer));
        assert!(builder.directory_caches.contains_key(&deep));
        Ok(())
    }

    #[test]
    fn metadata_directory_has_expected_type_and_parent_link() {
        let mut builder = fresh_builder().expect("failed to create builder");
        let metadata = PosixMetadata {
            mode: Some(0o750),
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
    fn filling_lost_and_found_fails_before_anything_is_committed() -> Result<()> {
        let mut builder = fresh_builder()?;
        let lost = builder
            .lookup_dir_entry(EXT4_ROOT_INO, "lost+found")?
            .expect("lost+found exists");
        // lost+found is the only multi-block linear directory ext4kit
        // creates; about a thousand short records fill its four blocks.
        let meta = PosixMetadata::default();
        let mut added = 0;
        while builder
            .add_file(lost, &format!("f{added:04}"), b"", &meta)
            .is_ok()
        {
            added += 1;
        }
        assert!(added > 1000, "lost+found held only {added} entries");

        let free_blocks = builder.superblock.s_free_blocks_count_lo;
        let free_inodes = builder.superblock.s_free_inodes_count;
        let error = builder
            .add_file(lost, "overflow", b"payload", &meta)
            .unwrap_err();
        assert!(
            error.to_string().contains("multi-block linear directory"),
            "{error}"
        );
        assert_eq!({ builder.superblock.s_free_blocks_count_lo }, free_blocks);
        assert_eq!({ builder.superblock.s_free_inodes_count }, free_inodes);
        assert_eq!(builder.lookup_dir_entry(lost, "overflow")?, None);
        Ok(())
    }

    #[test]
    fn rollback_created_inode_restores_the_pre_creation_state() -> Result<()> {
        let mut builder = fresh_builder()?;
        let free_blocks = builder.superblock.s_free_blocks_count_lo;
        let free_inodes = builder.superblock.s_free_inodes_count;
        let mut meta = PosixMetadata::default();
        meta.xattrs.insert("user.note".into(), b"note".to_vec());
        let ino = builder.add_file(EXT4_ROOT_INO, "file", &[1; BLOCK_SIZE * 2], &meta)?;

        // Leave the committed inode in the state a failed add_dir_entry
        // would: the record gone, but the inode and its blocks still held.
        let (block, record) = builder
            .find_dir_record(EXT4_ROOT_INO, "file")?
            .expect("the file is linked");
        builder.unlink_record(EXT4_ROOT_INO, block, record.offset)?;
        let error = builder.rollback_created_inode(ino, anyhow::anyhow!("link failed"));

        assert_eq!(format!("{error:#}"), "link failed");
        assert_eq!({ builder.superblock.s_free_blocks_count_lo }, free_blocks);
        assert_eq!({ builder.superblock.s_free_inodes_count }, free_inodes);
        assert!(!builder.is_inode_allocated(ino));
        Ok(())
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
}
