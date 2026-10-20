//! Ext4 dx (htree) directory-index support.

use super::Ext4ImageBuilder;
use super::directory::{DirSlot, align4};
use crate::core::dirhash;
use crate::ext4::*;
use anyhow::Result;
use zerocopy::{FromBytes, IntoBytes};

#[derive(Clone, Debug)]
pub(crate) struct HtreeEntry {
    pub(crate) hash: u32,
    pub(crate) inode: u32,
    pub(crate) file_type: u8,
    pub(crate) name: Vec<u8>,
}

const DX_ROOT_ENTRIES_OFFSET: usize = 32;
const DX_NODE_ENTRIES_OFFSET: usize = 8;

fn dx_root_limit() -> usize {
    (BLOCK_SIZE - DX_ROOT_ENTRIES_OFFSET) / 8
}

fn dx_node_limit() -> usize {
    (BLOCK_SIZE - DX_NODE_ENTRIES_OFFSET) / 8
}

fn split_boundary(entries: &[HtreeEntry], split: usize) -> u32 {
    let hash = entries[split].hash;
    hash | u32::from(entries[split - 1].hash == hash)
}

fn dx_position(entries: &[(u32, u32)], hash: u32) -> Result<usize> {
    entries
        .iter()
        .rposition(|(boundary, _)| *boundary <= hash)
        .ok_or_else(|| anyhow::anyhow!("dx index has no zero entry"))
}

impl Ext4ImageBuilder {
    /// Converts a full linear directory into a level-zero htree and inserts
    /// `new_entry`.  This is the same root/leaf transition performed by
    /// e2fsprogs: logical block zero becomes the dx root and the old records
    /// move into hashed leaf blocks one and two.
    pub(crate) fn make_indexed_and_insert(
        &mut self,
        parent: u32,
        name: &str,
        child: u32,
        file_type: u8,
    ) -> Result<()> {
        let blocks = self.get_dir_blocks(parent)?;
        if blocks.len() != 1 {
            return Err(anyhow::anyhow!(
                "cannot convert a multi-block linear directory to htree"
            ));
        }
        let root = blocks[0];
        let base = root as usize * BLOCK_SIZE;
        let dotdot = Ext4DirEntry2Header::read_from_bytes(&self.disk_data[base + 12..base + 20])
            .map_err(|_| anyhow::anyhow!("invalid dotdot entry"))?;
        let dotdot_inode = dotdot.inode;
        let mut entries = self.read_dir_leaf(root)?;
        // '.' and '..' never enter leaf blocks.
        entries.retain(|entry| entry.name.as_slice() != b"." && entry.name.as_slice() != b"..");
        if entries.iter().any(|entry| entry.name == name.as_bytes()) {
            return Err(anyhow::anyhow!("directory entry {name:?} already exists"));
        }
        entries.push(HtreeEntry {
            hash: self.dir_hash(name.as_bytes()),
            inode: child,
            file_type,
            name: name.as_bytes().to_vec(),
        });
        entries.sort_by(|a, b| a.hash.cmp(&b.hash).then_with(|| a.name.cmp(&b.name)));

        let total: usize = entries
            .iter()
            .map(|entry| align4(8 + entry.name.len()))
            .sum();
        if total > BLOCK_SIZE * 2 {
            return Err(anyhow::anyhow!(
                "initial htree leaves are unexpectedly full"
            ));
        }
        let mut split = 1;
        let mut left_size = align4(8 + entries[0].name.len());
        while split < entries.len() && left_size < total / 2 {
            left_size += align4(8 + entries[split].name.len());
            split += 1;
        }
        let first_leaf = self.expand_dir(parent)?;
        let second_leaf = self.expand_dir(parent)?;
        self.write_dir_leaf(first_leaf, &entries[..split])?;
        self.write_dir_leaf(second_leaf, &entries[split..])?;
        let boundary = split_boundary(&entries, split);
        self.write_dx_root(parent, dotdot_inode, 0, &[(0, 1), (boundary, 2)])?;
        let (group, index) = self.inode_location(parent);
        self.groups[group].inode_table[index].i_flags |= EXT4_INDEX_FL;
        Ok(())
    }

    /// Splits the htree-selected full leaf and adds its new boundary to the
    /// level-zero dx root.  A dx root's entries are sorted by major hash;
    /// equal hashes remain adjacent, matching e2fsprogs' collision ordering.
    pub(crate) fn split_htree_leaf_and_insert(
        &mut self,
        parent: u32,
        name: &str,
        child: u32,
        file_type: u8,
    ) -> Result<()> {
        let hash = self.dir_hash(name.as_bytes());
        let (levels, mut root_index) = self.read_dx_root(parent)?;
        let root_pos = dx_position(&root_index, hash)?;
        let (node_logical, mut index, leaf_pos) = if levels == 0 {
            let leaf_pos = dx_position(&root_index, hash)?;
            (None, root_index.clone(), leaf_pos)
        } else {
            let node_logical = root_index[root_pos].1;
            let index = self.read_dx_node(parent, node_logical)?;
            let leaf_pos = dx_position(&index, hash)?;
            (Some(node_logical), index, leaf_pos)
        };
        if levels == 1 && index.len() == dx_node_limit() && root_index.len() == dx_root_limit() {
            return Err(anyhow::anyhow!(
                "maximum htree depth and root capacity reached"
            ));
        }
        let blocks = self.get_dir_blocks(parent)?;
        let logical = index[leaf_pos].1 as usize;
        let old_leaf = *blocks
            .get(logical)
            .ok_or_else(|| anyhow::anyhow!("dx entry points outside directory"))?;
        let mut entries = self.read_dir_leaf(old_leaf)?;
        if entries.iter().any(|entry| entry.name == name.as_bytes()) {
            return Err(anyhow::anyhow!("directory entry {name:?} already exists"));
        }
        entries.push(HtreeEntry {
            hash,
            inode: child,
            file_type,
            name: name.as_bytes().to_vec(),
        });
        entries.sort_by(|a, b| a.hash.cmp(&b.hash).then_with(|| a.name.cmp(&b.name)));
        let total: usize = entries
            .iter()
            .map(|entry| align4(8 + entry.name.len()))
            .sum();
        let mut split = 1;
        let mut left_size = align4(8 + entries[0].name.len());
        while split < entries.len() - 1 && left_size < total / 2 {
            left_size += align4(8 + entries[split].name.len());
            split += 1;
        }
        let new_leaf = self.expand_dir(parent)?;
        let new_logical = blocks.len() as u32;
        self.write_dir_leaf(old_leaf, &entries[..split])?;
        self.write_dir_leaf(new_leaf, &entries[split..])?;
        index.insert(leaf_pos + 1, (split_boundary(&entries, split), new_logical));
        let root = blocks[0] as usize * BLOCK_SIZE;
        let dotdot_inode =
            u32::from_le_bytes(self.disk_data[root + 12..root + 16].try_into().unwrap());
        match node_logical {
            None if index.len() <= dx_root_limit() => {
                self.write_dx_root(parent, dotdot_inode, 0, &index)
            }
            None => {
                let node_block = self.expand_dir(parent)?;
                let node_logical = new_logical + 1;
                self.write_dx_node(node_block, &index)?;
                self.write_dx_root(parent, dotdot_inode, 1, &[(0, node_logical)])
            }
            Some(node_logical) if index.len() <= dx_node_limit() => {
                let node_block = blocks[node_logical as usize];
                self.write_dx_node(node_block, &index)
            }
            Some(node_logical) => {
                if root_index.len() >= dx_root_limit() {
                    return Err(anyhow::anyhow!(
                        "maximum htree depth and root capacity reached"
                    ));
                }
                let split = index.len() / 2;
                let boundary = index[split].0;
                let new_node_block = self.expand_dir(parent)?;
                let new_node_logical = new_logical + 1;
                let old_node_block = blocks[node_logical as usize];
                self.write_dx_node(old_node_block, &index[..split])?;
                self.write_dx_node(new_node_block, &index[split..])?;
                root_index.insert(root_pos + 1, (boundary, new_node_logical));
                self.write_dx_root(parent, dotdot_inode, 1, &root_index)
            }
        }
    }
}

impl Ext4ImageBuilder {
    /// Returns whether `ino` contains an ext4 dx (htree) directory root.
    pub(crate) fn dir_is_indexed(&self, ino: u32) -> Result<bool> {
        if ino == 0 {
            return Err(anyhow::anyhow!("inode 0 does not exist"));
        }
        if !self.is_inode_allocated(ino) {
            return Err(anyhow::anyhow!("inode {ino} does not exist"));
        }
        let (group, index) = self.inode_location(ino);
        let inode = &self.groups[group].inode_table[index];
        if inode.i_mode & S_IFDIR != S_IFDIR {
            return Err(anyhow::anyhow!("inode {ino} is not a directory"));
        }
        Ok(inode.i_flags & EXT4_INDEX_FL != 0)
    }

    /// Computes the on-disk htree hash using the filesystem's configured seed.
    pub(crate) fn dir_hash(&self, name: &[u8]) -> u32 {
        // Only HALF_MD4 is emitted by this formatter.  Keeping this check here
        // makes an unsupported superblock setting fail loudly instead of making
        // a silently corrupt index.
        debug_assert_eq!(
            self.superblock.s_def_hash_version,
            dirhash::DX_HASH_HALF_MD4
        );
        let seed = self.superblock.s_hash_seed;
        dirhash::ext4_dirhash_half_md4(name, Some(&seed)).0
    }

    /// Number of directory blocks needed to split the selected full leaf.
    pub(crate) fn htree_growth_blocks(&self, parent: u32, name: &str) -> Result<u32> {
        let (levels, root) = self.read_dx_root(parent)?;
        if levels == 0 {
            return Ok(if root.len() == dx_root_limit() { 2 } else { 1 });
        }
        let hash = self.dir_hash(name.as_bytes());
        let node_pos = dx_position(&root, hash)?;
        let node = self.read_dx_node(parent, root[node_pos].1)?;
        if node.len() == dx_node_limit() && root.len() == dx_root_limit() {
            return Err(anyhow::anyhow!(
                "maximum htree depth and root capacity reached"
            ));
        }
        Ok(1 + u32::from(node.len() == dx_node_limit()))
    }

    /// Reads active entries from one ordinary directory leaf.
    pub(crate) fn read_dir_leaf(&self, block: u32) -> Result<Vec<HtreeEntry>> {
        let base = block as usize * BLOCK_SIZE;
        let mut entries = Vec::new();
        let mut off = 0;
        while off < BLOCK_SIZE {
            let e =
                Ext4DirEntry2Header::read_from_bytes(&self.disk_data[base + off..base + off + 8])
                    .map_err(|_| anyhow::anyhow!("invalid directory entry header"))?;
            let len = e.rec_len as usize;
            if len < 8
                || len % 4 != 0
                || off + len > BLOCK_SIZE
                || (e.inode != 0 && e.name_len as usize + 8 > len)
            {
                return Err(anyhow::anyhow!(
                    "directory block {block} has an invalid record"
                ));
            }
            if e.inode != 0 {
                let name =
                    self.disk_data[base + off + 8..base + off + 8 + e.name_len as usize].to_vec();
                entries.push(HtreeEntry {
                    hash: self.dir_hash(&name),
                    inode: e.inode,
                    file_type: e.file_type,
                    name,
                });
            }
            off += len;
        }
        Ok(entries)
    }

    /// Serializes active entries as one ordinary directory leaf.
    pub(crate) fn write_dir_leaf(&mut self, block: u32, entries: &[HtreeEntry]) -> Result<()> {
        let used: usize = entries
            .iter()
            .map(|entry| align4(8 + entry.name.len()))
            .sum();
        if used > BLOCK_SIZE {
            return Err(anyhow::anyhow!("htree leaf does not fit in one block"));
        }
        let base = block as usize * BLOCK_SIZE;
        self.disk_data[base..base + BLOCK_SIZE].fill(0);
        let mut off = 0;
        for (i, entry) in entries.iter().enumerate() {
            let rec_len = if i + 1 == entries.len() {
                BLOCK_SIZE - off
            } else {
                align4(8 + entry.name.len())
            };
            self.disk_data[base + off..base + off + 8].copy_from_slice(
                Ext4DirEntry2Header {
                    inode: entry.inode,
                    rec_len: rec_len as u16,
                    name_len: entry.name.len() as u8,
                    file_type: entry.file_type,
                }
                .as_bytes(),
            );
            self.disk_data[base + off + 8..base + off + 8 + entry.name.len()]
                .copy_from_slice(&entry.name);
            off += rec_len;
        }
        if entries.is_empty() {
            self.disk_data[base..base + 8].copy_from_slice(
                Ext4DirEntry2Header {
                    inode: 0,
                    rec_len: BLOCK_SIZE as u16,
                    name_len: 0,
                    file_type: 0,
                }
                .as_bytes(),
            );
        }
        Ok(())
    }

    /// Reads the single-level dx root.  The first hash is implicit zero: its
    /// on-disk hash field is overlaid by the count/limit pair, as in e2fsprogs.
    #[cfg(test)]
    pub(crate) fn read_dx_entries(&self, parent: u32) -> Result<Vec<(u32, u32)>> {
        Ok(self.read_dx_root(parent)?.1)
    }

    fn read_dx_root(&self, parent: u32) -> Result<(u8, Vec<(u32, u32)>)> {
        let blocks = self.get_dir_blocks(parent)?;
        let root = *blocks
            .first()
            .ok_or_else(|| anyhow::anyhow!("directory has no root block"))?;
        let base = root as usize * BLOCK_SIZE;
        let info = Ext4DxRootInfo::read_from_bytes(&self.disk_data[base + 24..base + 32])
            .map_err(|_| anyhow::anyhow!("invalid dx root info"))?;
        if info.hash_version != dirhash::DX_HASH_HALF_MD4
            || info.info_length != 8
            || info.indirect_levels > 1
        {
            return Err(anyhow::anyhow!("unsupported dx root in directory {parent}"));
        }
        let counts = Ext4DxCountlimit::read_from_bytes(&self.disk_data[base + 32..base + 36])
            .map_err(|_| anyhow::anyhow!("invalid dx count/limit"))?;
        let entries = self.read_dx_entry_array(
            base + DX_ROOT_ENTRIES_OFFSET,
            counts,
            dx_root_limit(),
            parent,
        )?;
        Ok((info.indirect_levels, entries))
    }

    fn read_dx_node(&self, parent: u32, logical: u32) -> Result<Vec<(u32, u32)>> {
        let blocks = self.get_dir_blocks(parent)?;
        let block = *blocks
            .get(logical as usize)
            .ok_or_else(|| anyhow::anyhow!("dx node points outside directory"))?;
        let base = block as usize * BLOCK_SIZE;
        let counts = Ext4DxCountlimit::read_from_bytes(
            &self.disk_data[base + DX_NODE_ENTRIES_OFFSET..base + DX_NODE_ENTRIES_OFFSET + 4],
        )
        .map_err(|_| anyhow::anyhow!("invalid dx node count/limit"))?;
        self.read_dx_entry_array(
            base + DX_NODE_ENTRIES_OFFSET,
            counts,
            dx_node_limit(),
            parent,
        )
    }

    fn read_dx_entry_array(
        &self,
        entries_base: usize,
        counts: Ext4DxCountlimit,
        expected_limit: usize,
        parent: u32,
    ) -> Result<Vec<(u32, u32)>> {
        let limit = counts.limit as usize;
        let count = counts.count as usize;
        if count == 0 || count > limit || limit != expected_limit {
            return Err(anyhow::anyhow!(
                "invalid dx entry count in directory {parent}"
            ));
        }
        let mut entries = Vec::with_capacity(count);
        for i in 0..count {
            let off = entries_base + i * 8;
            let hash = if i == 0 {
                0
            } else {
                u32::from_le_bytes(self.disk_data[off..off + 4].try_into().unwrap())
            };
            let block = u32::from_le_bytes(self.disk_data[off + 4..off + 8].try_into().unwrap());
            entries.push((hash, block));
        }
        Ok(entries)
    }

    pub(crate) fn write_dx_root(
        &mut self,
        parent: u32,
        dotdot_inode: u32,
        indirect_levels: u8,
        entries: &[(u32, u32)],
    ) -> Result<()> {
        let blocks = self.get_dir_blocks(parent)?;
        let root = *blocks
            .first()
            .ok_or_else(|| anyhow::anyhow!("directory has no root block"))?;
        let limit = dx_root_limit();
        if entries.is_empty() || entries.len() > limit {
            return Err(anyhow::anyhow!("dx root entry limit exceeded"));
        }
        let base = root as usize * BLOCK_SIZE;
        self.disk_data[base..base + BLOCK_SIZE].fill(0);
        self.disk_data[base..base + 8].copy_from_slice(
            Ext4DirEntry2Header {
                inode: parent,
                rec_len: 12,
                name_len: 1,
                file_type: EXT4_FT_DIR,
            }
            .as_bytes(),
        );
        self.disk_data[base + 8] = b'.';
        self.disk_data[base + 12..base + 20].copy_from_slice(
            Ext4DirEntry2Header {
                inode: dotdot_inode,
                // The dx payload occupies the tail of the '..' record.  This
                // is how e2fsprogs keeps a dx root readable as a conventional
                // directory block by older ext2 readers.
                rec_len: (BLOCK_SIZE - 12) as u16,
                name_len: 2,
                file_type: EXT4_FT_DIR,
            }
            .as_bytes(),
        );
        self.disk_data[base + 20..base + 22].copy_from_slice(b"..");
        self.disk_data[base + 24..base + 32].copy_from_slice(
            Ext4DxRootInfo {
                reserved_zero: 0,
                hash_version: dirhash::DX_HASH_HALF_MD4,
                info_length: 8,
                indirect_levels,
                unused_flags: 0,
            }
            .as_bytes(),
        );
        self.disk_data[base + 32..base + 36].copy_from_slice(
            Ext4DxCountlimit {
                limit: limit as u16,
                count: entries.len() as u16,
            }
            .as_bytes(),
        );
        for (i, &(hash, logical_block)) in entries.iter().enumerate() {
            let off = base + 32 + i * 8;
            if i != 0 {
                self.disk_data[off..off + 4].copy_from_slice(&hash.to_le_bytes());
            }
            self.disk_data[off + 4..off + 8].copy_from_slice(&logical_block.to_le_bytes());
        }
        Ok(())
    }

    fn write_dx_node(&mut self, block: u32, entries: &[(u32, u32)]) -> Result<()> {
        if entries.is_empty() || entries.len() > dx_node_limit() {
            return Err(anyhow::anyhow!("dx node entry limit exceeded"));
        }
        let base = block as usize * BLOCK_SIZE;
        self.disk_data[base..base + BLOCK_SIZE].fill(0);
        self.disk_data[base..base + 8].copy_from_slice(
            Ext4DirEntry2Header {
                inode: 0,
                rec_len: BLOCK_SIZE as u16,
                name_len: 0,
                file_type: 0,
            }
            .as_bytes(),
        );
        self.disk_data[base + 8..base + 12].copy_from_slice(
            Ext4DxCountlimit {
                limit: dx_node_limit() as u16,
                count: entries.len() as u16,
            }
            .as_bytes(),
        );
        for (i, &(hash, logical)) in entries.iter().enumerate() {
            let off = base + DX_NODE_ENTRIES_OFFSET + i * 8;
            if i != 0 {
                self.disk_data[off..off + 4].copy_from_slice(&hash.to_le_bytes());
            }
            self.disk_data[off + 4..off + 8].copy_from_slice(&logical.to_le_bytes());
        }
        Ok(())
    }

    /// Looks up only the htree-selected leaf, rather than linearly walking the directory.
    pub(crate) fn find_htree_slot(&self, parent: u32, name: &str) -> Result<Option<DirSlot>> {
        let hash = self.dir_hash(name.as_bytes());
        let (levels, root) = self.read_dx_root(parent)?;
        let root_pos = dx_position(&root, hash)?;
        let blocks = self.get_dir_blocks(parent)?;
        let needed = align4(8 + name.len());
        let mut found = None;
        let mut node_pos = root_pos;
        let mut index = if levels == 0 {
            root.clone()
        } else {
            self.read_dx_node(parent, root[node_pos].1)?
        };
        let mut leaf_pos = dx_position(&index, hash)?;
        'collision_chain: loop {
            let logical = index[leaf_pos].1;
            let block = *blocks
                .get(logical as usize)
                .ok_or_else(|| anyhow::anyhow!("dx entry points outside directory"))?;
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
                    || (e.inode != 0 && e.name_len as usize + 8 > len)
                {
                    return Err(anyhow::anyhow!(
                        "directory inode {parent} has an invalid record"
                    ));
                }
                if e.inode != 0 {
                    if e.name_len as usize == name.len()
                        && &self.disk_data[base + off + 8..base + off + 8 + e.name_len as usize]
                            == name.as_bytes()
                    {
                        return Err(anyhow::anyhow!("directory entry {name:?} already exists"));
                    }
                    let actual = align4(8 + e.name_len as usize);
                    if found.is_none() && len >= actual + needed {
                        found = Some(DirSlot::SplitActiveRecord {
                            offset: base + off,
                            actual,
                            rec_len: len,
                        });
                    }
                } else if found.is_none() && len >= needed {
                    found = Some(DirSlot::EmptyRecord {
                        offset: base + off,
                        rec_len: len,
                    });
                }
                off += len;
            }
            leaf_pos += 1;
            if leaf_pos < index.len() {
                if index[leaf_pos].0 & 1 != 0 && index[leaf_pos].0 & !1 == hash {
                    continue;
                }
                break;
            }
            if levels == 1 && node_pos + 1 < root.len() {
                let boundary = root[node_pos + 1].0;
                if boundary & 1 != 0 && boundary & !1 == hash {
                    node_pos += 1;
                    index = self.read_dx_node(parent, root[node_pos].1)?;
                    leaf_pos = 0;
                    continue 'collision_chain;
                }
            }
            break;
        }
        Ok(found)
    }
}
