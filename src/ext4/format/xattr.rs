//! Ext4 extended-attribute block encoding and POSIX ACL conversion.

use crate::{
    ext4::format::*,
    ext4::{Ext4ImageBuilder, directory::PreparedXattr},
};
use anyhow::Result;
use std::collections::HashMap;
use zerocopy::IntoBytes;

impl Ext4ImageBuilder {
    /// Serializes an unshared ext4 xattr block; returns its allocated physical block number.
    pub fn write_xattr_block(
        &mut self,
        group: usize,
        xattrs: &HashMap<String, Vec<u8>>,
    ) -> Result<u32> {
        let xattrs = prepare_xattrs(xattrs)?;
        self.write_prepared_xattr_block(group, &xattrs)
    }

    /// Writes already validated xattrs into one block; returns that block's physical number.
    pub(crate) fn write_prepared_xattr_block(
        &mut self,
        group: usize,
        xattrs: &[PreparedXattr],
    ) -> Result<u32> {
        let mut data = vec![0; BLOCK_SIZE];
        let header = Ext4XattrHeader {
            h_magic: EXT4_XATTR_MAGIC,
            h_refcount: 1,
            h_blocks: 1,
            ..Default::default()
        };
        data[..32].copy_from_slice(header.as_bytes());

        let (mut entry, mut value) = (32, BLOCK_SIZE);
        let mut block_hash = 0u32;
        for xattr in xattrs {
            let aligned_value_len = align4(xattr.value.len());
            value = value
                .checked_sub(aligned_value_len)
                .ok_or(anyhow::anyhow!("xattrs exceed one filesystem block"))?;
            let entry_len = 16 + align4(xattr.suffix.len());
            if entry + entry_len + 4 > value {
                return Err(anyhow::anyhow!("xattrs exceed one filesystem block"));
            }
            let entry_hash = xattr_entry_hash(xattr.suffix.as_bytes(), &xattr.value);
            let xattr_entry = Ext4XattrEntry {
                e_name_len: xattr.suffix.len() as u8,
                e_name_index: xattr.index,
                e_value_offs: value as u16,
                e_value_block: 0,
                e_value_size: xattr.value.len() as u32,
                e_hash: entry_hash,
            };
            data[entry..entry + 16].copy_from_slice(xattr_entry.as_bytes());
            block_hash = block_hash.rotate_left(16) ^ entry_hash;
            data[entry + 16..entry + 16 + xattr.suffix.len()]
                .copy_from_slice(xattr.suffix.as_bytes());
            // Values occupy an aligned extent on disk. Clear it explicitly so that
            // alignment padding remains deterministic if this buffer is ever reused.
            data[value..value + aligned_value_len].fill(0);
            data[value..value + xattr.value.len()].copy_from_slice(&xattr.value);
            entry += entry_len;
        }
        data[12..16].copy_from_slice(&block_hash.to_le_bytes());
        let number = self.allocate_free_blocks(group, 1)?;
        let offset = number as usize * BLOCK_SIZE;
        self.disk_data[offset..offset + BLOCK_SIZE].copy_from_slice(&data);
        Ok(number)
    }
}

/// Parses a Linux xattr name; returns its ext4 namespace index and unprefixed suffix.
pub fn parse_xattr_name(name: &str) -> Result<(u8, &str)> {
    for (prefix, index) in [
        ("system.posix_acl_access", EXT4_XATTR_INDEX_POSIX_ACL_ACCESS),
        (
            "system.posix_acl_default",
            EXT4_XATTR_INDEX_POSIX_ACL_DEFAULT,
        ),
        ("user.", EXT4_XATTR_INDEX_USER),
        ("trusted.", EXT4_XATTR_INDEX_TRUSTED),
        ("security.", EXT4_XATTR_INDEX_SECURITY),
        ("system.", EXT4_XATTR_INDEX_SYSTEM),
    ] {
        if let Some(suffix) = name.strip_prefix(prefix) {
            return Ok((index, suffix));
        }
    }
    Err(anyhow::anyhow!(
        "unsupported ext4 xattr namespace in {name:?}"
    ))
}

/// Validates, sorts, and converts xattrs for on-disk writing; returns prepared entries.
pub(crate) fn prepare_xattrs(xattrs: &HashMap<String, Vec<u8>>) -> Result<Vec<PreparedXattr>> {
    let mut prepared = Vec::with_capacity(xattrs.len());
    for (name, value) in xattrs {
        let (index, suffix) = parse_xattr_name(name)?;
        if suffix.len() > u8::MAX as usize
            || (suffix.is_empty()
                && !matches!(
                    index,
                    EXT4_XATTR_INDEX_POSIX_ACL_ACCESS | EXT4_XATTR_INDEX_POSIX_ACL_DEFAULT
                ))
        {
            return Err(anyhow::anyhow!("invalid xattr name {name:?}"));
        }
        prepared.push(PreparedXattr {
            index,
            suffix: suffix.to_owned(),
            value: if matches!(
                index,
                EXT4_XATTR_INDEX_POSIX_ACL_ACCESS | EXT4_XATTR_INDEX_POSIX_ACL_DEFAULT
            ) {
                posix_acl_to_ext4_disk(value)?
            } else {
                value.clone()
            },
        });
    }
    prepared.sort_unstable_by(|left, right| {
        (left.index, &left.suffix).cmp(&(right.index, &right.suffix))
    });
    Ok(prepared)
}

/// Rounds a byte count up to an ext4 four-byte boundary; returns the aligned count.
fn align4(value: usize) -> usize {
    (value + 3) & !3
}

/// Computes the ext2fs-compatible hash for an xattr entry; returns its 32-bit hash.
fn xattr_entry_hash(name: &[u8], value: &[u8]) -> u32 {
    let mut hash = 0u32;
    for byte in name {
        hash = hash.rotate_left(5) ^ u32::from(*byte);
    }
    for bytes in value.chunks(4) {
        let mut word = [0u8; 4];
        word[..bytes.len()].copy_from_slice(bytes);
        hash = hash.rotate_left(16) ^ u32::from_le_bytes(word);
    }
    hash
}

/// Converts a Linux POSIX ACL xattr to ext4's compact disk encoding; returns encoded bytes.
pub(crate) fn posix_acl_to_ext4_disk(value: &[u8]) -> Result<Vec<u8>> {
    if value.len() < 12
        || !(value.len() - 4).is_multiple_of(8)
        || u32::from_le_bytes(value[..4].try_into().unwrap()) != 2
    {
        return Err(anyhow::anyhow!("invalid POSIX ACL xattr payload"));
    }
    let mut disk = Vec::with_capacity(value.len());
    disk.extend_from_slice(&1u32.to_le_bytes());
    for entry in value[4..].as_chunks::<8>().0 {
        let tag = u16::from_le_bytes(entry[..2].try_into().unwrap());
        let perm = u16::from_le_bytes(entry[2..4].try_into().unwrap());
        if perm > 0o7 {
            return Err(anyhow::anyhow!("POSIX ACL permissions exceed 0o7"));
        }
        disk.extend_from_slice(&tag.to_le_bytes());
        disk.extend_from_slice(&perm.to_le_bytes());
        match tag {
            0x02 | 0x08 => disk.extend_from_slice(&entry[4..8]),
            0x01 | 0x04 | 0x10 | 0x20 => {}
            _ => return Err(anyhow::anyhow!("unsupported POSIX ACL tag {tag:#x}")),
        }
    }
    Ok(disk)
}
