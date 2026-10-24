//! Ext4 extended-attribute block encoding and POSIX ACL conversion.

use super::*;
use anyhow::Result;
use std::collections::HashMap;
use zerocopy::IntoBytes;

/// On-disk sizes of the external xattr block header, of one entry before
/// its name, and of the zero end marker that follows the last entry.
const XATTR_HEADER_LEN: usize = size_of::<Ext4XattrHeader>();
const XATTR_ENTRY_LEN: usize = size_of::<Ext4XattrEntry>();
const XATTR_END_MARKER_LEN: usize = 4;

#[derive(Clone, Debug)]
pub(crate) struct PreparedXattr {
    pub(crate) index: u8,
    pub(crate) suffix: String,
    pub(crate) value: Vec<u8>,
}

/// Encodes validated xattrs as one unshared external xattr block with its
/// entry and block hashes; returns the block's bytes.
pub(crate) fn encode_xattr_block(xattrs: &[PreparedXattr]) -> Result<Vec<u8>> {
    let mut data = vec![0; BLOCK_SIZE];
    let header = Ext4XattrHeader {
        h_magic: EXT4_XATTR_MAGIC,
        h_refcount: 1,
        h_blocks: 1,
        ..Default::default()
    };
    data[..XATTR_HEADER_LEN].copy_from_slice(header.as_bytes());

    let (mut entry, mut value) = (XATTR_HEADER_LEN, BLOCK_SIZE);
    let mut block_hash = 0u32;
    for xattr in xattrs {
        let aligned_value_len = align4(xattr.value.len());
        value = value
            .checked_sub(aligned_value_len)
            .ok_or(anyhow::anyhow!("xattrs exceed one filesystem block"))?;
        let entry_len = XATTR_ENTRY_LEN + align4(xattr.suffix.len());
        if entry + entry_len + XATTR_END_MARKER_LEN > value {
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
        data[entry..entry + XATTR_ENTRY_LEN].copy_from_slice(xattr_entry.as_bytes());
        block_hash = block_hash.rotate_left(16) ^ entry_hash;
        let name = entry + XATTR_ENTRY_LEN;
        data[name..name + xattr.suffix.len()].copy_from_slice(xattr.suffix.as_bytes());
        // Values occupy an aligned extent on disk. Clear it explicitly so that
        // alignment padding remains deterministic if this buffer is ever reused.
        data[value..value + aligned_value_len].fill(0);
        data[value..value + xattr.value.len()].copy_from_slice(&xattr.value);
        entry += entry_len;
    }
    data[12..16].copy_from_slice(&block_hash.to_le_bytes());
    Ok(data)
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
    // Checked here rather than only when encoding, so callers can reject an
    // oversized set before they commit an inode or any blocks.
    ensure_fits_one_block(&prepared)?;
    Ok(prepared)
}

/// Verifies that `xattrs` fit one external xattr block: the header and
/// entries grow up from the start, the 4-byte end marker follows them, and
/// the 4-byte-aligned values grow down from the end of the block.  This is
/// the final, tightest check [`encode_xattr_block`] makes while it writes.
fn ensure_fits_one_block(xattrs: &[PreparedXattr]) -> Result<()> {
    let entries: usize = xattrs
        .iter()
        .map(|xattr| XATTR_ENTRY_LEN + align4(xattr.suffix.len()))
        .sum();
    let values: usize = xattrs.iter().map(|xattr| align4(xattr.value.len())).sum();
    if XATTR_HEADER_LEN + entries + XATTR_END_MARKER_LEN + values > BLOCK_SIZE {
        return Err(anyhow::anyhow!("xattrs exceed one filesystem block"));
    }
    Ok(())
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

#[cfg(test)]
mod tests {
    use super::*;

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
    fn prepare_rejects_sets_that_do_not_fit_one_block_like_the_encoder() {
        // One "user.v" entry: 32-byte header, 20-byte entry (16 + "v"
        // aligned to 4), 4-byte end marker, so 4040 value bytes fill the block.
        let fill = BLOCK_SIZE - 32 - 20 - 4;
        let set = |len: usize| HashMap::from([("user.v".to_owned(), vec![7u8; len])]);

        let fits = prepare_xattrs(&set(fill)).unwrap();
        assert!(encode_xattr_block(&fits).is_ok());

        let error = prepare_xattrs(&set(fill + 1)).unwrap_err();
        assert!(error.to_string().contains("exceed one filesystem block"));
        // The encoder agrees, so the two checks cannot drift apart unnoticed.
        let oversized = vec![PreparedXattr {
            index: EXT4_XATTR_INDEX_USER,
            suffix: "v".to_owned(),
            value: vec![7; fill + 1],
        }];
        assert!(encode_xattr_block(&oversized).is_err());
    }
}
