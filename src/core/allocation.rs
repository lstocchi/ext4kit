use super::Ext4ImageBuilder;
use anyhow::Result;

impl Ext4ImageBuilder {
    /// Maps an inode number to its block-group and zero-based table index; returns both indices.
    pub fn inode_location(&self, ino: u32) -> (usize, usize) {
        let inodes_per_group = self.superblock.s_inodes_per_group;
        (
            ((ino - 1) / inodes_per_group) as usize,
            ((ino - 1) % inodes_per_group) as usize,
        )
    }

    /// Returns whether `ino` is valid and marked allocated in its inode bitmap.
    pub fn is_inode_allocated(&self, ino: u32) -> bool {
        if ino == 0 || ino > self.superblock.s_inodes_count {
            return false;
        }
        let (group, index_in_group) = self.inode_location(ino);
        let byte_idx = (index_in_group / 8) as usize;
        let bit_idx = index_in_group % 8;

        self.groups[group as usize].inode_bitmap[byte_idx] & (1 << bit_idx) != 0
    }

    /// Finds the next free inode, preferring the parent's block group; returns its inode number.
    pub(crate) fn next_free_inode(&self, parent_dir_ino: Option<u32>) -> Result<u32> {
        let total_inodes = self.superblock.s_inodes_count;
        let inodes_per_group = self.superblock.s_inodes_per_group;
        let start_ino = match parent_dir_ino {
            Some(dir) if dir > 0 && dir <= total_inodes => {
                ((dir - 1) / inodes_per_group) * inodes_per_group + 1
            }
            _ => 1,
        };

        (start_ino..=total_inodes)
            .chain(1..start_ino)
            .find(|&ino| !self.is_inode_allocated(ino))
            .ok_or_else(|| anyhow::anyhow!("No free inodes available"))
    }
}
