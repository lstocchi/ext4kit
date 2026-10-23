//! Extracted core filesystem operations.

use super::Ext4ImageBuilder;
use crate::ext4::format::*;
use anyhow::Result;
use zerocopy::FromBytes;

/// Maximum extent tree depth accepted by e2fsprogs and the kernel.
const EXT4_MAX_EXTENT_DEPTH: u16 = 5;

/// Initialized extents are at most 32,768 blocks long; larger `ee_len`
/// values encode an uninitialized extent of `ee_len - 32768` blocks.
pub(super) const EXT_INIT_MAX_LEN: u16 = 1 << 15;

/// Leaf extents and external index/leaf blocks reachable from an inode.
#[derive(Default)]
pub(crate) struct ExtentTree {
    pub(crate) leaves: Vec<Ext4Extent>,
    pub(crate) tree_blocks: Vec<u32>,
}

/// Builds initialized leaf extents from physical runs, assigning consecutive logical blocks.
pub(super) fn extents_for_runs(runs: &[(u32, u32)]) -> Vec<Ext4Extent> {
    let mut logical = 0u32;
    runs.iter()
        .map(|&(start, len)| {
            let extent = Ext4Extent {
                ee_block: logical,
                ee_len: len as u16,
                ee_start_hi: 0,
                ee_start_lo: start,
            };
            logical += len;
            extent
        })
        .collect()
}

/// Returns the number of blocks mapped by `extent`, initialized or not.
pub(crate) fn extent_len(extent: &Ext4Extent) -> u32 {
    let len = extent.ee_len;
    u32::from(if len > EXT_INIT_MAX_LEN {
        len - EXT_INIT_MAX_LEN
    } else {
        len
    })
}

/// Mirrors ext2fs_extent_header_verify(): validates magic, entry count and
/// `eh_max` against the node size (allowing room for a checksum tail).
fn verify_extent_header(node: &[u8]) -> Result<Ext4ExtentHeader> {
    let header = node
        .get(..12)
        .and_then(|bytes| Ext4ExtentHeader::read_from_bytes(bytes).ok())
        .ok_or_else(|| anyhow::anyhow!("truncated extent header"))?;
    if header.eh_magic != EXT4_EH_MAGIC {
        return Err(anyhow::anyhow!("invalid extent magic"));
    }
    let (entries, max) = (header.eh_entries as usize, header.eh_max as usize);
    let capacity = (node.len() - 12) / 12;
    if entries > max || max > capacity || max + 2 < capacity {
        return Err(anyhow::anyhow!("invalid extent header entry count"));
    }
    Ok(header)
}

impl Ext4ImageBuilder {
    /// Walks an extent tree node, requiring each child to be exactly one
    /// level below its parent.  Strictly decreasing depth bounds the walk, so
    /// a corrupt tree cannot cycle.
    fn walk_extent_node(
        &self,
        node: &[u8],
        expected_depth: Option<u16>,
        tree: &mut ExtentTree,
    ) -> Result<()> {
        let header = verify_extent_header(node)?;
        let depth = header.eh_depth;
        if depth > EXT4_MAX_EXTENT_DEPTH || expected_depth.is_some_and(|d| d != depth) {
            return Err(anyhow::anyhow!("invalid extent tree depth {depth}"));
        }
        for i in 0..header.eh_entries as usize {
            let entry = &node[12 + i * 12..24 + i * 12];
            if depth == 0 {
                tree.leaves.push(
                    Ext4Extent::read_from_bytes(entry)
                        .map_err(|_| anyhow::anyhow!("invalid extent entry"))?,
                );
                continue;
            }
            let idx = Ext4ExtentIdx::read_from_bytes(entry)
                .map_err(|_| anyhow::anyhow!("invalid extent index"))?;
            let block = idx.ei_leaf_lo;
            let offset = block as usize * BLOCK_SIZE;
            let child = self
                .disk_data
                .get(offset..offset + BLOCK_SIZE)
                .ok_or_else(|| anyhow::anyhow!("extent block {block} is outside the image"))?;
            tree.tree_blocks.push(block);
            self.walk_extent_node(child, Some(depth - 1), tree)?;
        }
        Ok(())
    }

    /// Decodes the complete extent tree rooted in `inode.i_block`.
    pub(crate) fn read_extent_tree(&self, inode: &Ext4Inode) -> Result<ExtentTree> {
        let mut tree = ExtentTree::default();
        self.walk_extent_node(&inode.i_block, None, &mut tree)?;
        Ok(tree)
    }

    /// Decodes every leaf extent referenced by `inode`; returns them in tree traversal order.
    pub(crate) fn get_inode_extents(&self, inode: &Ext4Inode) -> Result<Vec<Ext4Extent>> {
        Ok(self.read_extent_tree(inode)?.leaves)
    }

    /// Returns a directory's physical data blocks in logical order, or an error if it is invalid.
    pub(crate) fn get_dir_blocks(&self, ino: u32) -> Result<Vec<u32>> {
        if ino == 0 {
            return Err(anyhow::anyhow!("inode 0 does not exist"));
        }
        if !self.is_inode_allocated(ino) {
            return Err(anyhow::anyhow!("inode {ino} does not exist"));
        }
        let inode = self.inode(ino);
        if !inode.is_dir() {
            return Err(anyhow::anyhow!("inode {ino} is not a directory"));
        }
        if inode.i_flags & EXT4_EXTENTS_FL == 0 {
            return Err(anyhow::anyhow!(
                "directory inode {ino} does not use extents"
            ));
        }
        let extents = self.get_inode_extents(inode)?;
        let mut blocks = Vec::new();
        for ext in extents {
            blocks.extend((0..extent_len(&ext)).map(|b| ext.ee_start_lo + b));
        }
        Ok(blocks)
    }
}
