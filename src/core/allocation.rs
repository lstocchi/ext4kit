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

    /// Finds a free inode near `parent_dir_ino`, probing from the group's
    /// allocation cursor and wrapping only when reclaimed space requires it.
    pub(crate) fn next_free_inode(&mut self, parent_dir_ino: Option<u32>) -> Result<u32> {
        let total_inodes = self.superblock.s_inodes_count;
        let inodes_per_group = self.superblock.s_inodes_per_group;
        let preferred_group = parent_dir_ino
            .filter(|&dir| dir > 0 && dir <= total_inodes)
            .map(|dir| ((dir - 1) / inodes_per_group) as usize)
            .unwrap_or(0);

        for delta in 0..self.groups.len() {
            let group = (preferred_group + delta) % self.groups.len();
            if self.groups[group].free_inodes_count == 0 {
                continue;
            }
            let limit = inodes_per_group as usize;
            let start = self.next_inode_hints[group].min(limit);
            for index in (start..limit).chain(0..start) {
                if self.groups[group].inode_bitmap[index / 8] & (1 << (index % 8)) == 0 {
                    self.next_inode_hints[group] = index + 1;
                    return Ok((group * limit + index + 1) as u32);
                }
            }
        }
        Err(anyhow::anyhow!("No free inodes available"))
    }
}
