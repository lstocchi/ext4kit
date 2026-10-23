//! Block and inode bitmaps: planning, committing and releasing blocks and
//! inodes, and the free counts that go with them.

use super::Ext4ImageBuilder;
use crate::ext4::format::*;
use anyhow::Result;

impl Ext4ImageBuilder {
    /// Returns whether `ino` is valid and marked allocated in its inode bitmap.
    pub fn is_inode_allocated(&self, ino: u32) -> bool {
        if ino == 0 || ino > self.superblock.s_inodes_count {
            return false;
        }
        let (group, index_in_group) = self.inode_location(ino);
        let byte_idx = index_in_group / 8;
        let bit_idx = index_in_group % 8;

        self.groups[group].inode_bitmap[byte_idx] & (1 << bit_idx) != 0
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

    /// Marks one physical block allocated and updates accounting; returns no value.
    pub(crate) fn mark_block_used(&mut self, group_idx: usize, block: u32) {
        let byte_idx = (block / 8) as usize;
        let bit_idx = block % 8;

        self.groups[group_idx].block_bitmap[byte_idx] |= 1 << bit_idx;
        self.groups[group_idx].free_blocks_count -= 1;
        self.group_descriptors[group_idx].bg_free_blocks_count_lo -= 1;
        self.superblock.s_free_blocks_count_lo -= 1;
        self.next_block_hints[group_idx] = self.next_block_hints[group_idx].max(block as usize + 1);
    }

    /// Marks one inode allocated and updates accounting; returns no value.
    pub(crate) fn mark_inode_used(&mut self, inode: u32) {
        let (group, index_in_group) = self.inode_location(inode);
        let byte_idx = index_in_group / 8;
        let bit_idx = index_in_group % 8;

        self.groups[group].inode_bitmap[byte_idx] |= 1 << bit_idx;
        self.groups[group].free_inodes_count -= 1;
        self.group_descriptors[group].bg_free_inodes_count_lo -= 1;
        self.superblock.s_free_inodes_count -= 1;
        self.next_inode_hints[group] = self.next_inode_hints[group].max(index_in_group + 1);
    }

    /// Releases an allocated inode and restores its allocation accounting.
    /// The caller is responsible for releasing any blocks owned by the inode.
    pub(crate) fn release_inode(&mut self, inode: u32) -> Result<()> {
        if !self.is_inode_allocated(inode) {
            return Err(anyhow::anyhow!("inode {inode} is not allocated"));
        }
        let (group, index_in_group) = self.inode_location(inode);
        let byte_idx = index_in_group / 8;
        let bit_idx = index_in_group % 8;

        self.groups[group].inode_bitmap[byte_idx] &= !(1 << bit_idx);
        self.groups[group].free_inodes_count += 1;
        self.group_descriptors[group].bg_free_inodes_count_lo += 1;
        self.superblock.s_free_inodes_count += 1;
        self.next_inode_hints[group] = self.next_inode_hints[group].min(index_in_group);
        self.groups[group].inode_table[index_in_group] = Ext4Inode::default();
        Ok(())
    }

    /// Reserves a contiguous extent near `preferred_group`, beginning at the
    /// group's allocation cursor and wrapping to reclaimed blocks if needed.
    pub(crate) fn allocate_free_blocks(
        &mut self,
        preferred_group: usize,
        blocks_needed: u32,
    ) -> Result<u32> {
        if blocks_needed == 0 {
            return Ok(0);
        }
        for delta in 0..self.groups.len() {
            let group = (preferred_group + delta) % self.groups.len();
            if u32::from(self.groups[group].free_blocks_count) < blocks_needed {
                continue;
            }
            let limit = self.groups[group].blocks_count as usize;
            let hint = self.next_block_hints[group].min(limit);

            // Directories and extent-tree metadata overwhelmingly request one
            // block. Search 64 bitmap bits at a time instead of interpreting
            // every allocated bit individually.
            if blocks_needed == 1 {
                let bitmap = &self.groups[group].block_bitmap;
                if let Some(block) = find_next_free_bit(bitmap, hint, limit)
                    .or_else(|| find_next_free_bit(bitmap, 0, hint))
                {
                    self.groups[group].block_bitmap[block / 8] |= 1 << (block % 8);
                    self.groups[group].free_blocks_count -= 1;
                    self.group_descriptors[group].bg_free_blocks_count_lo -= 1;
                    self.superblock.s_free_blocks_count_lo -= 1;
                    self.next_block_hints[group] = block + 1;
                    return Ok((group * BLOCKS_PER_GROUP + block) as u32);
                }
                continue;
            }

            let mut candidate_start = hint;
            let mut contiguous_free = 0;
            for bit in (hint..limit).chain(0..hint) {
                if bit == 0 {
                    contiguous_free = 0;
                }
                // if block is free, if contiguous_free is 0, set candidate_start to bit, otherwise increment contiguous_free
                if self.groups[group].block_bitmap[bit / 8] & (1 << (bit % 8)) == 0 {
                    if contiguous_free == 0 {
                        candidate_start = bit;
                    }
                    contiguous_free += 1;
                    if contiguous_free == blocks_needed {
                        for allocated in candidate_start..candidate_start + blocks_needed as usize {
                            self.groups[group].block_bitmap[allocated / 8] |= 1 << (allocated % 8);
                        }
                        self.groups[group].free_blocks_count -= blocks_needed as u16;
                        self.group_descriptors[group].bg_free_blocks_count_lo -=
                            blocks_needed as u16;
                        self.superblock.s_free_blocks_count_lo -= blocks_needed;
                        self.next_block_hints[group] = candidate_start + blocks_needed as usize;
                        return Ok((group * BLOCKS_PER_GROUP + candidate_start) as u32);
                    }
                } else {
                    contiguous_free = 0;
                }
            }
        }
        Err(anyhow::anyhow!(
            "no contiguous run of {blocks_needed} free blocks is available"
        ))
    }

    /// Verifies that every block in `[start, start + len)` lies inside the
    /// filesystem and is marked allocated.
    pub(crate) fn validate_allocated_run(&self, start: u32, len: u32) -> Result<()> {
        for block in u64::from(start)..u64::from(start) + u64::from(len) {
            let group = block as usize / BLOCKS_PER_GROUP;
            let local = block as usize % BLOCKS_PER_GROUP;
            if group >= self.groups.len() || local >= self.groups[group].blocks_count as usize {
                return Err(anyhow::anyhow!("block {block} is outside the filesystem"));
            }
            if self.groups[group].block_bitmap[local / 8] & (1 << (local % 8)) == 0 {
                return Err(anyhow::anyhow!("block {block} is not allocated"));
            }
        }
        Ok(())
    }

    /// Releases a run of allocated blocks (ext2fs_block_alloc_stats_range),
    /// updating accounting and rewinding each group's allocation cursor.
    pub(crate) fn release_blocks(&mut self, start: u32, len: u32) -> Result<()> {
        self.validate_allocated_run(start, len)?;
        for block in start..start + len {
            let group = block as usize / BLOCKS_PER_GROUP;
            let local = block as usize % BLOCKS_PER_GROUP;
            self.groups[group].block_bitmap[local / 8] &= !(1 << (local % 8));
            self.groups[group].free_blocks_count += 1;
            self.group_descriptors[group].bg_free_blocks_count_lo += 1;
            self.superblock.s_free_blocks_count_lo += 1;
            self.next_block_hints[group] = self.next_block_hints[group].min(local);
        }
        Ok(())
    }

    /// Releases one allocated block.
    pub(crate) fn release_block(&mut self, block: u32) -> Result<()> {
        self.release_blocks(block, 1)
    }

    /// Prove that a file extent plus any separately allocated metadata block
    /// can be allocated before changing the block bitmap.
    /// Verifies enough data and metadata blocks remain; returns `Ok(())` or an exhaustion error.
    pub(crate) fn ensure_allocation_capacity(
        &self,
        extent_blocks: u32,
        metadata_blocks: u32,
    ) -> Result<()> {
        let total_needed = extent_blocks
            .checked_add(metadata_blocks)
            .ok_or_else(|| anyhow::anyhow!("block allocation overflow"))?;

        let free_blocks = self.superblock.s_free_blocks_count_lo;
        if free_blocks < total_needed {
            return Err(anyhow::anyhow!(
                "not enough free blocks: needed {total_needed}, available {}",
                free_blocks
            ));
        }
        Ok(())
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
}

/// Finds the first zero bitmap bit in `[start, end)`, scanning full machine
/// words where possible while preserving ascending-bit allocation order.
fn find_next_free_bit(bitmap: &[u8], start: usize, end: usize) -> Option<usize> {
    let mut bit = start;
    while bit < end && !bit.is_multiple_of(64) {
        if bitmap[bit / 8] & (1 << (bit % 8)) == 0 {
            return Some(bit);
        }
        bit += 1;
    }

    while bit + 64 <= end {
        let byte = bit / 8;
        let word = u64::from_le_bytes(bitmap[byte..byte + 8].try_into().unwrap());
        if word != u64::MAX {
            return Some(bit + (!word).trailing_zeros() as usize);
        }
        bit += 64;
    }

    while bit < end {
        if bitmap[bit / 8] & (1 << (bit % 8)) == 0 {
            return Some(bit);
        }
        bit += 1;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::{Ext4ImageBuilder, find_next_free_bit};

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

    #[test]
    fn free_bit_search_respects_range_and_bit_order() {
        let mut bitmap = [u8::MAX; 24];
        bitmap[1] &= !(1 << 3); // bit 11
        bitmap[9] &= !(1 << 5); // bit 77
        bitmap[20] &= !(1 << 1); // bit 161

        assert_eq!(find_next_free_bit(&bitmap, 0, 192), Some(11));
        assert_eq!(find_next_free_bit(&bitmap, 12, 160), Some(77));
        assert_eq!(find_next_free_bit(&bitmap, 78, 192), Some(161));
        assert_eq!(find_next_free_bit(&bitmap, 12, 77), None);
    }
}
