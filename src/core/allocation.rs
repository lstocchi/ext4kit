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
        if inodes_per_group == 0 || self.groups.is_empty() {
            return Err(anyhow::anyhow!("filesystem has no inode groups"));
        }
        let preferred_group = parent_dir_ino
            .filter(|&dir| dir > 0 && dir <= total_inodes)
            .map(|dir| ((dir - 1) / inodes_per_group) as usize)
            .unwrap_or(0);

        for delta in 0..self.groups.len() {
            let group = (preferred_group + delta) % self.groups.len();
            if self.groups[group].free_inodes_count == 0 {
                continue;
            }

            // A corrupt or truncated bitmap must not permit indexing past its
            // end.  The final group may also contain fewer logical inodes
            // than s_inodes_per_group, so never allocate its padding bits.
            let group_start = (group as u64) * u64::from(inodes_per_group);
            let logical_limit = u64::from(total_inodes)
                .saturating_sub(group_start)
                .min(u64::from(inodes_per_group)) as usize;
            let limit = logical_limit.min(self.groups[group].inode_bitmap.len().saturating_mul(8));
            if limit == 0 {
                continue;
            }

            let start = self.next_inode_hints[group] % limit;
            let bitmap = &self.groups[group].inode_bitmap;
            if let Some(index) = (start..limit)
                .chain(0..start)
                .find(|&index| bitmap[index / 8] & (1 << (index % 8)) == 0)
            {
                self.next_inode_hints[group] = (index + 1) % limit;
                return Ok((group_start + index as u64 + 1) as u32);
            }
        }
        Err(anyhow::anyhow!("No free inodes available"))
    }
}

#[cfg(test)]
mod tests {
    use super::Ext4ImageBuilder;

    #[test]
    fn inode_search_does_not_index_past_a_truncated_bitmap() {
        let mut builder = Ext4ImageBuilder::new(16);
        builder.groups[0].inode_bitmap = vec![u8::MAX];
        builder.groups[0].free_inodes_count = 1;

        assert!(builder.next_free_inode(None).is_err());
    }

    #[test]
    fn inode_search_wraps_a_cursor_at_the_group_limit() {
        let mut builder = Ext4ImageBuilder::new(16);
        let limit = builder.superblock.s_inodes_per_group as usize;
        builder.groups[0].inode_bitmap[..limit / 8].fill(u8::MAX);
        builder.groups[0].inode_bitmap[(limit - 1) / 8] &= !(1 << ((limit - 1) % 8));
        builder.groups[0].free_inodes_count = 1;
        builder.next_inode_hints[0] = limit;

        assert_eq!(builder.next_free_inode(None).unwrap(), limit as u32);
        assert_eq!(builder.next_inode_hints[0], 0);
    }
}
