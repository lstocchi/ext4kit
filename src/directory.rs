use crate::{builder::Ext4ImageBuilder, ext4::*};
use anyhow::Result;
use std::collections::HashMap;
use zerocopy::{FromBytes, IntoBytes};

/// Metadata collected from an OCI layer or a host filesystem.
#[derive(Clone, Debug, Default)]
pub struct PosixMetadata {
    pub mode: u16,
    pub uid: u32,
    pub gid: u32,
    pub atime: u32,
    pub ctime: u32,
    pub mtime: u32,
    pub xattrs: HashMap<String, Vec<u8>>,
}

#[derive(Clone, Debug)]
struct PreparedXattr {
    index: u8,
    suffix: String,
    value: Vec<u8>,
}

impl Ext4ImageBuilder {
    fn validate_name(name: &str) -> Result<()> {
        if name.is_empty() || name == "." || name == ".." || name.len() > 255 || name.contains('/')
        {
            Err(anyhow::anyhow!(
                "name must be a non-empty component no longer than 255 bytes"
            ))
        } else {
            Ok(())
        }
    }

    fn read_dir_block_index(&self, ino: u32) -> Result<u32> {
        if !self.is_inode_allocated(ino) {
            return Err(anyhow::anyhow!("inode {ino} does not exist"));
        }
        let (g, i) = self.inode_location(ino);
        let inode = &self.groups[g].inode_table[i];
        if inode.i_mode & S_IFDIR != S_IFDIR {
            return Err(anyhow::anyhow!("inode {ino} is not a directory"));
        }
        let h = Ext4ExtentHeader::read_from_bytes(&inode.i_block[..12])
            .map_err(|_| anyhow::anyhow!("invalid extent header"))?;
        if h.eh_magic != EXT4_EH_MAGIC || h.eh_depth != 0 || h.eh_entries != 1 {
            return Err(anyhow::anyhow!(
                "inode {ino} has an unsupported directory extent"
            ));
        }
        Ok(Ext4Extent::read_from_bytes(&inode.i_block[12..24])
            .map_err(|_| anyhow::anyhow!("invalid directory extent"))?
            .ee_start_lo)
    }

    pub fn add_dir_entry(
        &mut self,
        parent: u32,
        name: &str,
        child: u32,
        file_type: u8,
    ) -> Result<()> {
        Self::validate_name(name)?;
        let (base, off, actual) = self.find_dir_entry_slot(parent, name)?;
        let bytes = name.as_bytes();
        // it copy the actual length of the last old entry to its rec_len field
        self.disk_data[base + off + 4..base + off + 6]
            .copy_from_slice(&(actual as u16).to_le_bytes());

        let new_offset = off + actual;
        self.disk_data[base + new_offset..base + new_offset + 8].copy_from_slice(
            Ext4DirEntry2Header {
                inode: child,
                rec_len: (BLOCK_SIZE - new_offset) as u16,
                name_len: bytes.len() as u8,
                file_type,
            }
            .as_bytes(),
        );
        self.disk_data[base + new_offset + 8..base + new_offset + 8 + bytes.len()]
            .copy_from_slice(bytes);
        Ok(())
    }

    /// Preflight check: verifies that the name is valid, does not already exist
    /// in the parent directory, and that the directory block has enough space.
    fn ensure_dir_entry_creatable(&self, parent: u32, name: &str) -> Result<()> {
        self.find_dir_entry_slot(parent, name).map(|_| ())
    }

    /// Finds and returns the precise byte offsets `(base, off, actual)` for inserting
    /// a new entry into the parent directory block.
    fn find_dir_entry_slot(&self, parent: u32, name: &str) -> Result<(usize, usize, usize)> {
        //Self::validate_name(name)?;
        Self::validate_name(name)?;
        let base = self.read_dir_block_index(parent)? as usize * BLOCK_SIZE;
        let bytes = name.as_bytes();
        let needed = align4(8 + bytes.len());
        let mut off = 0;

        while off < BLOCK_SIZE {
            let e =
                Ext4DirEntry2Header::read_from_bytes(&self.disk_data[base + off..base + off + 8])
                    .map_err(|_| anyhow::anyhow!("invalid directory entry"))?;
            let len = e.rec_len as usize;
            // TODO: to verify .... is it really useful? we're checking an already added entry. So i assume the check has been done at insertion
            if len < 8 || len % 4 != 0 || off + len > BLOCK_SIZE || e.name_len as usize + 8 > len {
                return Err(anyhow::anyhow!(
                    "directory inode {parent} has an invalid record"
                ));
            }
            if e.inode != 0
                && &self.disk_data[base + off + 8..base + off + 8 + e.name_len as usize] == bytes
            {
                return Err(anyhow::anyhow!("directory entry {name:?} already exists"));
            }
            if off + len == BLOCK_SIZE {
                let actual = align4(8 + e.name_len as usize);
                if len - actual < needed {
                    return Err(anyhow::anyhow!(
                        "directory inode {parent} has no space for {name:?}"
                    ));
                }
                return Ok((base, off, actual));
            }
            off += len;
        }

        Err(anyhow::anyhow!(
            "directory inode {parent} has no terminal entry"
        ))
    }

    /// Populates an in-memory Ext4Inode with POSIX metadata fields.
    fn populate_inode_metadata(
        inode: &mut Ext4Inode,
        m: &PosixMetadata,
        kind: u16,
        default_perms: u16,
    ) {
        let mode = if m.mode == 0 {
            default_perms
        } else {
            m.mode & 0o7777
        };
        inode.i_mode = kind | mode;
        inode.i_uid = m.uid as u16;
        inode.i_gid = m.gid as u16;
        inode.i_osd2[4..6].copy_from_slice(&((m.uid >> 16) as u16).to_le_bytes());
        inode.i_osd2[6..8].copy_from_slice(&((m.gid >> 16) as u16).to_le_bytes());
        inode.i_atime = m.atime;
        inode.i_ctime = m.ctime;
        inode.i_mtime = m.mtime;
        inode.i_extra_isize = 32;
    }

    pub fn mkdir(&mut self, parent: u32, name: &str, meta: Option<&PosixMetadata>) -> Result<u32> {
        Self::validate_name(name)?;
        self.ensure_dir_entry_creatable(parent, name)?;

        let default_meta = PosixMetadata::default();
        let meta = meta.unwrap_or(&default_meta);
        let xattrs = prepare_xattrs(&meta.xattrs)?;
        // it finds the next free inode starting from the block of the parent directory, but still do not allocate it
        let ino = self.next_free_inode(Some(parent))?;
        // it gets the group of the free inode found
        let (group, index_in_group) = self.inode_location(ino);
        // it ensures that the group has enough capacity to allocate the new inode
        self.ensure_allocation_capacity(group, 1, u32::from(!xattrs.is_empty()))?;
        // allocate the inode
        self.mark_inode_used(ino);

        let data_block = self.allocate_free_blocks(group, 1)?;
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

        let mut inode = Ext4Inode::default();
        inode.i_size_lo = BLOCK_SIZE as u32;
        inode.i_links_count = 2;
        inode.i_blocks_lo = (BLOCK_SIZE / 512) as u32;
        inode.i_flags = EXT4_EXTENTS_FL;

        Self::populate_inode_metadata(&mut inode, meta, S_IFDIR, 0o755);
        init_single_extent(&mut inode, 1, data_block);
        if !xattrs.is_empty() {
            inode.i_file_acl_lo = self.write_prepared_xattr_block(group, &xattrs)?;
            inode.i_blocks_lo += (BLOCK_SIZE / 512) as u32;
        }

        self.groups[group].inode_table[index_in_group] = inode;
        self.group_descriptors[group].bg_used_dirs_count_lo += 1;
        self.add_dir_entry(parent, name, ino, EXT4_FT_DIR)?;
        let (pgroup, pindex_in_group) = self.inode_location(parent);
        self.groups[pgroup].inode_table[pindex_in_group].i_links_count += 1;
        Ok(ino)
    }

    /// Create a regular file with one contiguous, depth-zero extent.
    pub fn add_file(
        &mut self,
        parent: u32,
        name: &str,
        data: &[u8],
        meta: &PosixMetadata,
    ) -> Result<u32> {
        Self::validate_name(name)?;
        self.ensure_dir_entry_creatable(parent, name)?;

        let blocks_needed = data.len().div_ceil(BLOCK_SIZE);
        if blocks_needed > BLOCKS_PER_GROUP {
            return Err(anyhow::anyhow!("file is too large for a single extent"));
        }

        let xattrs = prepare_xattrs(&meta.xattrs)?;

        // it finds the next free inode starting from the block of the parent directory, but still do not allocate it
        let ino = self.next_free_inode(Some(parent))?;
        // it gets the group of the free inode found
        let (group, index_in_group) = self.inode_location(ino);
        // it ensures that the group has enough capacity to allocate the new inode
        self.ensure_allocation_capacity(
            group,
            blocks_needed as u32,
            u32::from(!xattrs.is_empty()),
        )?;
        // allocate the inode
        self.mark_inode_used(ino);
        let data_extent_start_block = if blocks_needed == 0 {
            0
        } else {
            self.allocate_free_blocks(group, blocks_needed as u32)?
        };

        if !data.is_empty() {
            let offset = data_extent_start_block as usize * BLOCK_SIZE;
            self.disk_data[offset..offset + data.len()].copy_from_slice(data);
        }

        let mut inode = Ext4Inode::default();
        Self::populate_inode_metadata(&mut inode, meta, S_IFREG, 0o644);
        inode.i_size_lo =
            u32::try_from(data.len()).map_err(|_| anyhow::anyhow!("file is larger than 4 GiB"))?;
        inode.i_links_count = 1;
        inode.i_blocks_lo = blocks_needed as u32 * (BLOCK_SIZE / 512) as u32;
        inode.i_flags = EXT4_EXTENTS_FL;

        init_single_extent(&mut inode, blocks_needed as u16, data_extent_start_block);
        if !xattrs.is_empty() {
            inode.i_file_acl_lo = self.write_prepared_xattr_block(group, &xattrs)?;
            inode.i_blocks_lo += (BLOCK_SIZE / 512) as u32;
        }

        self.groups[group].inode_table[index_in_group] = inode;
        self.add_dir_entry(parent, name, ino, EXT4_FT_REG_FILE)?;

        Ok(ino)
    }

    /// External, unshared xattr block. Its layout matches ext2fs' ext_attr format.
    pub fn write_xattr_block(
        &mut self,
        group: usize,
        xattrs: &HashMap<String, Vec<u8>>,
    ) -> Result<u32> {
        let xattrs = prepare_xattrs(xattrs)?;
        self.write_prepared_xattr_block(group, &xattrs)
    }

    fn write_prepared_xattr_block(
        &mut self,
        group: usize,
        xattrs: &[PreparedXattr],
    ) -> Result<u32> {
        let mut data = vec![0; BLOCK_SIZE];

        let xattr_header = Ext4XattrHeader {
            h_magic: EXT4_XATTR_MAGIC,
            h_refcount: 1,
            h_blocks: 1,
            ..Default::default()
        };
        data[..32].copy_from_slice(xattr_header.as_bytes());

        let (mut entry, mut value) = (32, BLOCK_SIZE);
        let mut block_hash = 0u32;
        for xattr in xattrs {
            let aligned_contents_len = align4(xattr.value.len());
            value = value
                .checked_sub(aligned_contents_len)
                .ok_or(anyhow::anyhow!("xattrs exceed one filesystem block"))?;
            let entry_len = 16 + align4(xattr.suffix.len());
            if entry + entry_len + 4 > value {
                return Err(anyhow::anyhow!("xattrs exceed one filesystem block"));
            }

            let xattr_entry = Ext4XattrEntry {
                e_name_len: xattr.suffix.len() as u8,
                e_name_index: xattr.index,
                e_value_offs: value as u16,
                e_value_block: 0,
                e_value_size: xattr.value.len() as u32,
                e_hash: xattr_entry_hash(xattr.suffix.as_bytes(), &xattr.value),
            };
            data[entry..entry + 16].copy_from_slice(xattr_entry.as_bytes());

            let entry_hash = xattr_entry_hash(xattr.suffix.as_bytes(), &xattr.value);
            block_hash = block_hash.rotate_left(16) ^ entry_hash;
            data[entry + 16..entry + 16 + xattr.suffix.len()]
                .copy_from_slice(xattr.suffix.as_bytes());
            data[value..value + xattr.value.len()].copy_from_slice(&xattr.value);
            entry += entry_len;
        }

        // e2fsprogs uses this hash to identify an external attribute block.
        // A nonzero hash is required even when we deliberately do not deduplicate blocks.
        data[12..16].copy_from_slice(&block_hash.to_le_bytes());
        // All serialization can fail before this allocation; after it, this
        // function has no fallible path and cannot leak a reserved block.
        let number = self.allocate_free_blocks(group, 1)?;
        let offset = number as usize * BLOCK_SIZE;
        self.disk_data[offset..offset + BLOCK_SIZE].copy_from_slice(&data);

        Ok(number)
    }

    /// Create a symbolic link. Targets shorter than the inode's 60-byte
    /// `i_block` field are stored inline; longer targets use one data block.
    pub fn add_symlink(
        &mut self,
        parent: u32,
        name: &str,
        target: &str,
        meta: &PosixMetadata,
    ) -> Result<u32> {
        Self::validate_name(name)?;
        self.ensure_dir_entry_creatable(parent, name)?;

        let target_bytes = target.as_bytes();
        // Match ext2fs_symlink(): the terminating NUL must fit in one block,
        // so an empty target and targets up to BLOCK_SIZE - 1 bytes are valid.
        if target_bytes.len() >= BLOCK_SIZE {
            return Err(anyhow::anyhow!(
                "symlink target must be no longer than {} bytes",
                BLOCK_SIZE - 1
            ));
        }

        let xattrs = prepare_xattrs(&meta.xattrs)?;

        // e2fsprogs defines fastlink as strictly less than i_block size (60 bytes)
        let is_fastlink = target_bytes.len() < 60;
        let block_needed = if is_fastlink { 0 } else { 1 };

        // it finds the next free inode starting from the block of the parent directory, but still do not allocate it
        let ino = self.next_free_inode(Some(parent))?;
        // it gets the group of the free inode found
        let (group, index_in_group) = self.inode_location(ino);
        // it ensures that the group has enough capacity to allocate the new inode
        self.ensure_allocation_capacity(group, block_needed as u32, u32::from(!xattrs.is_empty()))?;

        // allocate the inode
        self.mark_inode_used(ino);

        let mut inode = Ext4Inode::default();
        Self::populate_inode_metadata(&mut inode, meta, S_IFLNK, 0o777);
        inode.i_size_lo = target_bytes.len() as u32;
        inode.i_links_count = 1;

        if is_fastlink {
            // Fast Symlink: Write directly into the 60-byte i_block array
            inode.i_blocks_lo = 0;
            inode.i_block[..target_bytes.len()].copy_from_slice(target_bytes);
        } else {
            // Note: e2fsprogs attempts inline_data here if the feature is enabled.
            // Since this builder currently do not have inline_data support, we route
            // directly to the slow block-allocated fallback.
            let data_block = self.allocate_free_blocks(group, 1)?;
            let offset = data_block as usize * BLOCK_SIZE;
            // The e2fsprogs buffer is zero-initialized before copying the
            // target, which supplies the NUL terminator and a zero-filled tail.
            self.disk_data[offset..offset + BLOCK_SIZE].fill(0);
            self.disk_data[offset..offset + target_bytes.len()].copy_from_slice(target_bytes);
            inode.i_blocks_lo = (BLOCK_SIZE / 512) as u32;
            inode.i_flags = EXT4_EXTENTS_FL;
            init_single_extent(&mut inode, 1, data_block);
        }

        if !xattrs.is_empty() {
            inode.i_file_acl_lo = self.write_prepared_xattr_block(group, &xattrs)?;
            inode.i_blocks_lo += (BLOCK_SIZE / 512) as u32;
        }

        self.groups[group].inode_table[index_in_group] = inode;
        self.add_dir_entry(parent, name, ino, EXT4_FT_SYMLINK)?;

        Ok(ino)
    }

    // Create special inodes (devices, FIFOs, sockets)
    // which do not occupy file extent blocks.
    fn add_special_file(
        &mut self,
        parent: u32,
        name: &str,
        kind: u16,
        default_perms: u16,
        dir_file_type: u8,
        i_block_payload: [u8; 60],
        meta: &PosixMetadata,
    ) -> Result<u32> {
        Self::validate_name(name)?;
        self.ensure_dir_entry_creatable(parent, name)?;
        let xattrs = prepare_xattrs(&meta.xattrs)?;
        // it finds the next free inode starting from the block of the parent directory, but still do not allocate it
        let ino = self.next_free_inode(Some(parent))?;
        // it gets the group of the free inode found
        let (group, index_in_group) = self.inode_location(ino);
        // it ensures that the group has enough capacity to allocate the new inode
        self.ensure_allocation_capacity(group, 0, u32::from(!xattrs.is_empty()))?;
        // allocate the inode
        self.mark_inode_used(ino);

        let mut inode = Ext4Inode::default();
        Self::populate_inode_metadata(&mut inode, meta, kind, default_perms);
        inode.i_size_lo = 0;
        inode.i_links_count = 1;
        inode.i_blocks_lo = 0;
        inode.i_flags = 0; // Special files do not use EXT4_EXTENTS_FL
        inode.i_block = i_block_payload;

        if !xattrs.is_empty() {
            inode.i_file_acl_lo = self.write_prepared_xattr_block(group, &xattrs)?;
            inode.i_blocks_lo += (BLOCK_SIZE / 512) as u32;
        }

        self.groups[group].inode_table[index_in_group] = inode;
        self.add_dir_entry(parent, name, ino, dir_file_type)?;
        Ok(ino)
    }

    /// Creates a character device node (e.g. /dev/null, /dev/console).
    pub fn add_char_device(
        &mut self,
        parent: u32,
        name: &str,
        major: u32,
        minor: u32,
        meta: &PosixMetadata,
    ) -> Result<u32> {
        let payload = encode_device_number(major, minor)?;
        self.add_special_file(parent, name, S_IFCHR, 0o660, EXT4_FT_CHRDEV, payload, meta)
    }

    /// Creates a block device node (e.g. /dev/sda, /dev/vda1).
    pub fn add_block_device(
        &mut self,
        parent: u32,
        name: &str,
        major: u32,
        minor: u32,
        meta: &PosixMetadata,
    ) -> Result<u32> {
        let payload = encode_device_number(major, minor)?;
        self.add_special_file(parent, name, S_IFBLK, 0o660, EXT4_FT_BLKDEV, payload, meta)
    }

    /// Creates a FIFO / named pipe.
    pub fn add_fifo(&mut self, parent: u32, name: &str, meta: &PosixMetadata) -> Result<u32> {
        self.add_special_file(parent, name, S_IFIFO, 0o644, EXT4_FT_FIFO, [0u8; 60], meta)
    }

    /// Creates a Unix domain socket.
    pub fn add_socket(&mut self, parent: u32, name: &str, meta: &PosixMetadata) -> Result<u32> {
        self.add_special_file(parent, name, S_IFSOCK, 0o777, EXT4_FT_SOCK, [0u8; 60], meta)
    }
}

fn align4(v: usize) -> usize {
    (v + 3) & !3
}

/// ext2fs' name/value rolling hashes (NAME_HASH_SHIFT=5, VALUE_HASH_SHIFT=16).
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

/// Initializes a depth-0 inline extent header with a single contiguous extent.
fn init_single_extent(inode: &mut Ext4Inode, blocks: u16, start: u32) {
    let extent_header = Ext4ExtentHeader {
        eh_magic: EXT4_EH_MAGIC,
        eh_entries: u16::from(blocks != 0),
        eh_max: 4,
        eh_depth: 0,
        eh_generation: 0,
    };
    inode.i_block[..12].copy_from_slice(extent_header.as_bytes());
    if blocks != 0 {
        let extent = Ext4Extent {
            ee_block: 0,
            ee_len: blocks,
            ee_start_hi: 0,
            ee_start_lo: start,
        };
        inode.i_block[12..24].copy_from_slice(extent.as_bytes());
    }
}

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

fn prepare_xattrs(xattrs: &HashMap<String, Vec<u8>>) -> Result<Vec<PreparedXattr>> {
    let mut prepared = Vec::with_capacity(xattrs.len());
    for (name, value) in xattrs {
        let (index, suffix) = parse_xattr_name(name)?;
        // POSIX ACL attributes have no suffix. Other ext4 namespaces require one.
        if suffix.len() > u8::MAX as usize
            || (suffix.is_empty()
                && !matches!(
                    index,
                    EXT4_XATTR_INDEX_POSIX_ACL_ACCESS | EXT4_XATTR_INDEX_POSIX_ACL_DEFAULT
                ))
        {
            return Err(anyhow::anyhow!("invalid xattr name {name:?}"));
        }
        let value = if matches!(
            index,
            EXT4_XATTR_INDEX_POSIX_ACL_ACCESS | EXT4_XATTR_INDEX_POSIX_ACL_DEFAULT
        ) {
            posix_acl_to_ext4_disk(value)?
        } else {
            value.clone()
        };
        prepared.push(PreparedXattr {
            index,
            suffix: suffix.to_owned(),
            value,
        });
    }
    prepared.sort_unstable_by(|left, right| {
        (left.index, &left.suffix).cmp(&(right.index, &right.suffix))
    });
    Ok(prepared)
}

// Linux's getxattr ABI uses 8-byte entries; ext4 stores object/group/mask/
// other entries in four bytes, exactly as e2fsprogs' convert_posix_acl_to_disk_buffer.
fn posix_acl_to_ext4_disk(value: &[u8]) -> Result<Vec<u8>> {
    const POSIX_ACL_XATTR_VERSION: u32 = 0x0002;
    const EXT4_ACL_VERSION: u32 = 0x0001;
    const ACL_USER_OBJ: u16 = 0x01;
    const ACL_USER: u16 = 0x02;
    const ACL_GROUP_OBJ: u16 = 0x04;
    const ACL_GROUP: u16 = 0x08;
    const ACL_MASK: u16 = 0x10;
    const ACL_OTHER: u16 = 0x20;

    if value.len() < 12
        || (value.len() - 4) % 8 != 0
        || u32::from_le_bytes(value[..4].try_into().unwrap()) != POSIX_ACL_XATTR_VERSION
    {
        return Err(anyhow::anyhow!("invalid POSIX ACL xattr payload"));
    }
    let mut disk = Vec::with_capacity(value.len());
    disk.extend_from_slice(&EXT4_ACL_VERSION.to_le_bytes());
    for entry in value[4..].chunks_exact(8) {
        let tag = u16::from_le_bytes(entry[..2].try_into().unwrap());
        let perm = u16::from_le_bytes(entry[2..4].try_into().unwrap());
        if perm > 0o7 {
            return Err(anyhow::anyhow!("POSIX ACL permissions exceed 0o7"));
        }
        disk.extend_from_slice(&tag.to_le_bytes());
        disk.extend_from_slice(&perm.to_le_bytes());
        match tag {
            ACL_USER | ACL_GROUP => disk.extend_from_slice(&entry[4..8]),
            ACL_USER_OBJ | ACL_GROUP_OBJ | ACL_MASK | ACL_OTHER => {}
            _ => return Err(anyhow::anyhow!("unsupported POSIX ACL tag {tag:#x}")),
        }
    }
    Ok(disk)
}

/// Encodes device major/minor numbers into the 60-byte `i_block` payload
/// following Linux kernel / e2fsprogs encoding rules.
fn encode_device_number(major: u32, minor: u32) -> Result<[u8; 60]> {
    if major > 0x0fff || minor > 0x000f_ffff {
        return Err(anyhow::anyhow!(
            "device number ({major}, {minor}) exceeds Linux dev_t limits"
        ));
    }
    let mut block = [0u8; 60];
    if major < 256 && minor < 256 {
        let old_dev = ((major << 8) | minor) as u32;
        block[..4].copy_from_slice(&old_dev.to_le_bytes());
    } else {
        let new_dev = (minor & 0xff) | (major << 8) | ((minor & !0xff) << 12);
        block[4..8].copy_from_slice(&new_dev.to_le_bytes());
    }
    Ok(block)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fresh_builder() -> Result<Ext4ImageBuilder> {
        let mut builder = Ext4ImageBuilder::new(16);
        builder.create_root_dir()?;
        builder.create_lost_and_found()?;
        builder.reserve_inodes();
        Ok(builder)
    }

    #[test]
    fn file_is_linked_and_uses_contiguous_extent_and_xattrs() {
        let mut builder = fresh_builder().expect("failed to create builder");
        let mut metadata = PosixMetadata {
            mode: 0o640,
            uid: 70_000,
            gid: 80_000,
            mtime: 42,
            ..Default::default()
        };
        metadata
            .xattrs
            .insert("user.note".into(), b"metadata".to_vec());
        let ino = builder
            .add_file(EXT4_ROOT_INO, "payload", &[7; BLOCK_SIZE + 10], &metadata)
            .expect("failed to add file");
        let (group, index) = builder.inode_location(ino);
        let inode = &builder.groups[group].inode_table[index];
        let mode = inode.i_mode;
        let size = inode.i_size_lo;
        let sectors = inode.i_blocks_lo;
        let xattr_block = inode.i_file_acl_lo;
        assert_eq!(mode, S_IFREG | 0o640);
        assert_eq!(size, (BLOCK_SIZE + 10) as u32);
        assert_eq!(sectors, 24); // two payload blocks plus one xattr block, in sectors
        assert_ne!(xattr_block, 0);
        let extent = Ext4Extent::read_from_bytes(&inode.i_block[12..24]).unwrap();
        let extent_len = extent.ee_len;
        let extent_start = extent.ee_start_lo;
        assert_eq!(extent_len, 2);
        assert_eq!(
            &builder.disk_data[extent_start as usize * BLOCK_SIZE
                ..extent_start as usize * BLOCK_SIZE + BLOCK_SIZE + 10],
            &[7; BLOCK_SIZE + 10]
        );
    }

    #[test]
    fn metadata_directory_has_expected_type_and_parent_link() {
        let mut builder = fresh_builder().expect("failed to create builder");
        let metadata = PosixMetadata {
            mode: 0o750,
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
    fn symlinks_use_fast_and_slow_storage_at_e2fsprogs_boundary() {
        let mut builder = fresh_builder().expect("failed to create builder");
        let metadata = PosixMetadata::default();

        let fast_ino = builder
            .add_symlink(EXT4_ROOT_INO, "fast", &"f".repeat(59), &metadata)
            .expect("failed to create fast symlink");
        let (fast_group, fast_index) = builder.inode_location(fast_ino);
        let fast = &builder.groups[fast_group].inode_table[fast_index];
        let (fast_mode, fast_size, fast_blocks) = (fast.i_mode, fast.i_size_lo, fast.i_blocks_lo);
        assert_eq!(fast_mode, S_IFLNK | 0o777);
        assert_eq!(fast_size, 59);
        assert_eq!(fast_blocks, 0);
        assert_eq!(
            &fast.i_block[..59],
            b"fffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
        );

        let target = "s".repeat(60);
        let slow_ino = builder
            .add_symlink(EXT4_ROOT_INO, "slow", &target, &metadata)
            .expect("failed to create slow symlink");
        let (slow_group, slow_index) = builder.inode_location(slow_ino);
        let slow = &builder.groups[slow_group].inode_table[slow_index];
        let extent = Ext4Extent::read_from_bytes(&slow.i_block[12..24]).unwrap();
        let block_offset = extent.ee_start_lo as usize * BLOCK_SIZE;
        let (slow_size, slow_blocks, extent_len) =
            (slow.i_size_lo, slow.i_blocks_lo, extent.ee_len);
        assert_eq!(slow_size, 60);
        assert_eq!(slow_blocks, (BLOCK_SIZE / 512) as u32);
        assert_eq!(extent_len, 1);
        assert_eq!(
            &builder.disk_data[block_offset..block_offset + 60],
            target.as_bytes()
        );
        assert!(
            builder.disk_data[block_offset + 60..block_offset + BLOCK_SIZE]
                .iter()
                .all(|byte| *byte == 0)
        );
    }

    #[test]
    fn symlink_target_may_be_empty_but_must_leave_room_for_nul() {
        let mut builder = fresh_builder().expect("failed to create builder");
        let metadata = PosixMetadata::default();
        let free_inodes = builder.superblock.s_free_inodes_count;

        let ino = builder
            .add_symlink(EXT4_ROOT_INO, "empty", "", &metadata)
            .expect("e2fsprogs permits an empty symlink target");
        let (group, index) = builder.inode_location(ino);
        let empty_size = builder.groups[group].inode_table[index].i_size_lo;
        assert_eq!(empty_size, 0);

        assert!(
            builder
                .add_symlink(
                    EXT4_ROOT_INO,
                    "too-long",
                    &"x".repeat(BLOCK_SIZE),
                    &metadata,
                )
                .is_err()
        );
        let remaining_inodes = builder.superblock.s_free_inodes_count;
        assert_eq!(remaining_inodes, free_inodes - 1);
    }

    #[test]
    fn char_and_block_device_nodes_encode_rdev() {
        let mut builder = fresh_builder().expect("failed to create builder");
        let metadata = PosixMetadata {
            mode: 0o666,
            ..Default::default()
        };

        // Character device with standard minor/major (< 256): /dev/null (1, 3)
        let chr_ino = builder
            .add_char_device(EXT4_ROOT_INO, "null", 1, 3, &metadata)
            .expect("failed to create char device");
        let (group, idx) = builder.inode_location(chr_ino);
        let chr = &builder.groups[group].inode_table[idx];

        let chr_mode = chr.i_mode;
        let chr_size = chr.i_size_lo;
        let chr_blocks = chr.i_blocks_lo;
        let chr_flags = chr.i_flags;
        let chr_block = chr.i_block;
        assert_eq!(chr_mode, S_IFCHR | 0o666);
        assert_eq!(chr_size, 0);
        assert_eq!(chr_blocks, 0);
        assert_eq!(chr_flags, 0); // Must NOT have EXT4_EXTENTS_FL
        let dev_lo = u32::from_le_bytes(chr_block[..4].try_into().unwrap());
        assert_eq!(dev_lo, (1 << 8) | 3);

        // Block device with high major/minor (>= 256): (259, 1)
        let blk_ino = builder
            .add_block_device(EXT4_ROOT_INO, "nvme0n1", 259, 1, &metadata)
            .expect("failed to create block device");
        let (b_group, b_idx) = builder.inode_location(blk_ino);
        let blk = &builder.groups[b_group].inode_table[b_idx];

        let blk_mode = blk.i_mode;
        let blk_size = blk.i_size_lo;
        let blk_blocks = blk.i_blocks_lo;
        let blk_flags = blk.i_flags;
        assert_eq!(blk_mode, S_IFBLK | 0o666);
        assert_eq!(blk_size, 0);
        assert_eq!(blk_blocks, 0);
        assert_eq!(blk_flags, 0);
        // High device numbers leave i_block[0] as 0 and set i_block[1]
        let dev_old = u32::from_le_bytes(blk.i_block[..4].try_into().unwrap());
        let dev_new = u32::from_le_bytes(blk.i_block[4..8].try_into().unwrap());
        assert_eq!(dev_old, 0);
        let expected_new = (1 & 0xff) | (259 << 8) | ((1 & !0xff) << 12);
        assert_eq!(dev_new, expected_new);
    }

    #[test]
    fn fifo_and_socket_nodes() {
        let mut builder = fresh_builder().expect("failed to create builder");
        let metadata = PosixMetadata::default();

        let fifo_ino = builder
            .add_fifo(EXT4_ROOT_INO, "test_pipe", &metadata)
            .expect("failed to create fifo");
        let (f_grp, f_idx) = builder.inode_location(fifo_ino);
        let fifo = &builder.groups[f_grp].inode_table[f_idx];

        let fifo_mode = fifo.i_mode;
        let fifo_size = fifo.i_size_lo;
        let fifo_blocks = fifo.i_blocks_lo;
        let fifo_flags = fifo.i_flags;
        let fifo_block = fifo.i_block;
        assert_eq!(fifo_mode, S_IFIFO | 0o644);
        assert_eq!(fifo_size, 0);
        assert_eq!(fifo_blocks, 0);
        assert_eq!(fifo_flags, 0);
        assert_eq!(fifo_block, [0u8; 60]);

        let sock_ino = builder
            .add_socket(EXT4_ROOT_INO, "test.sock", &metadata)
            .expect("failed to create socket");
        let (s_grp, s_idx) = builder.inode_location(sock_ino);
        let sock = &builder.groups[s_grp].inode_table[s_idx];

        let sock_mode = sock.i_mode;
        let sock_size = sock.i_size_lo;
        let sock_blocks = sock.i_blocks_lo;
        let sock_flags = sock.i_flags;
        let sock_block = sock.i_block;
        assert_eq!(sock_mode, S_IFSOCK | 0o777);
        assert_eq!(sock_size, 0);
        assert_eq!(sock_blocks, 0);
        assert_eq!(sock_flags, 0);
        assert_eq!(sock_block, [0u8; 60]);
    }

    #[test]
    fn device_numbers_must_fit_linux_dev_t() {
        assert!(encode_device_number(0x1000, 0).is_err());
        assert!(encode_device_number(0, 0x10_0000).is_err());
    }
}
