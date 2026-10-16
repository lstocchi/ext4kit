//! Extracted core filesystem operations.

use super::Ext4ImageBuilder;
use crate::ext4::*;
use anyhow::Result;
use zerocopy::FromBytes;

impl Ext4ImageBuilder {
    /// Recursively appends leaf extents reachable from `block_data`; returns an error for malformed extent data.
    fn collect_leaf_extents(&self, block_data: &[u8], extents: &mut Vec<Ext4Extent>) -> Result<()> {
        let header = Ext4ExtentHeader::read_from_bytes(&block_data[..12])
            .map_err(|_| anyhow::anyhow!("invalid extent header"))?;
        if header.eh_magic != EXT4_EH_MAGIC {
            return Err(anyhow::anyhow!("invalid extent magic"));
        }
        let entries = header.eh_entries as usize;
        if header.eh_depth == 0 {
            for i in 0..entries {
                let off = 12 + i * 12;
                let ext = Ext4Extent::read_from_bytes(&block_data[off..off + 12])
                    .map_err(|_| anyhow::anyhow!("invalid extent record"))?;
                extents.push(ext);
            }
        } else {
            for i in 0..entries {
                let off = 12 + i * 12;
                let idx = Ext4ExtentIdx::read_from_bytes(&block_data[off..off + 12])
                    .map_err(|_| anyhow::anyhow!("invalid extent index"))?;
                let blk = idx.ei_leaf_lo as usize;
                let offset = blk * BLOCK_SIZE;
                self.collect_leaf_extents(&self.disk_data[offset..offset + BLOCK_SIZE], extents)?;
            }
        }
        Ok(())
    }

    /// Collects external extent-tree block numbers reachable from a root; returns an error for malformed data.
    pub(crate) fn collect_extent_tree_blocks(
        &self,
        block_data: &[u8],
        blocks: &mut Vec<u32>,
    ) -> Result<()> {
        let header = Ext4ExtentHeader::read_from_bytes(&block_data[..12])
            .map_err(|_| anyhow::anyhow!("invalid extent header"))?;
        if header.eh_magic != EXT4_EH_MAGIC {
            return Err(anyhow::anyhow!("invalid extent magic"));
        }
        if header.eh_depth == 0 {
            return Ok(());
        }
        for i in 0..header.eh_entries as usize {
            let off = 12 + i * 12;
            let idx = Ext4ExtentIdx::read_from_bytes(&block_data[off..off + 12])
                .map_err(|_| anyhow::anyhow!("invalid extent index"))?;
            let block = idx.ei_leaf_lo;
            blocks.push(block);
            let data_offset = block as usize * BLOCK_SIZE;
            self.collect_extent_tree_blocks(
                &self.disk_data[data_offset..data_offset + BLOCK_SIZE],
                blocks,
            )?;
        }
        Ok(())
    }

    /// Decodes every leaf extent referenced by `inode`; returns them in tree traversal order.
    pub(crate) fn get_inode_extents(&self, inode: &Ext4Inode) -> Result<Vec<Ext4Extent>> {
        let mut extents = Vec::new();
        self.collect_leaf_extents(&inode.i_block, &mut extents)?;
        Ok(extents)
    }

    /// Returns a directory's physical data blocks in logical order, or an error if it is invalid.
    pub(crate) fn get_dir_blocks(&self, ino: u32) -> Result<Vec<u32>> {
        if !self.is_inode_allocated(ino) {
            return Err(anyhow::anyhow!("inode {ino} does not exist"));
        }
        let (g, i) = self.inode_location(ino);
        let inode = &self.groups[g].inode_table[i];
        if inode.i_mode & S_IFDIR != S_IFDIR {
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
            for b in 0..ext.ee_len {
                blocks.push(ext.ee_start_lo + b as u32);
            }
        }
        Ok(blocks)
    }
}
