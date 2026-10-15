use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

// ============================================================================
// On-Disk Ext4 Structures (Matching Linux Kernel Specs)
// ============================================================================

pub const EXT4_SUPER_MAGIC: u16 = 0xEF53;
pub const BLOCK_SIZE: usize = 4096;
pub const INODE_SIZE: usize = 256;
pub const BYTES_PER_INODE_RATIO: usize = 4096;
pub const BLOCKS_PER_GROUP: usize = 32768;

// Inode Numbers
pub const EXT4_BAD_INO: u32 = 1;
pub const EXT4_ROOT_INO: u32 = 2;
pub const EXT4_FIRST_INO: u32 = 11;

// File Types
pub const EXT4_FT_UNKNOWN: u8 = 0;
pub const EXT4_FT_REG_FILE: u8 = 1;
pub const EXT4_FT_DIR: u8 = 2;
pub const EXT4_FT_CHRDEV: u8 = 3;
pub const EXT4_FT_BLKDEV: u8 = 4;
pub const EXT4_FT_FIFO: u8 = 5;
pub const EXT4_FT_SOCK: u8 = 6;
pub const EXT4_FT_SYMLINK: u8 = 7;

pub const EXT4_EXTENTS_FL: u32 = 0x80000;

pub const EXT4_EH_MAGIC: u16 = 0xF30A;
pub const EXT4_XATTR_MAGIC: u32 = 0xEA02_0000;
pub const EXT4_XATTR_INDEX_USER: u8 = 1;
pub const EXT4_XATTR_INDEX_POSIX_ACL_ACCESS: u8 = 2;
pub const EXT4_XATTR_INDEX_POSIX_ACL_DEFAULT: u8 = 3;
pub const EXT4_XATTR_INDEX_TRUSTED: u8 = 4;
pub const EXT4_XATTR_INDEX_SECURITY: u8 = 6;
pub const EXT4_XATTR_INDEX_SYSTEM: u8 = 7;

// File Modes (inode.i_mode type masks)
pub const S_IFIFO: u16 = 0x1000; // Named pipe (FIFO)
pub const S_IFCHR: u16 = 0x2000; // Character device
pub const S_IFDIR: u16 = 0x4000; // Directory
pub const S_IFBLK: u16 = 0x6000; // Block device
pub const S_IFREG: u16 = 0x8000; // Regular file
pub const S_IFLNK: u16 = 0xA000; // Symbolic link
pub const S_IFSOCK: u16 = 0xC000; // Socket

// Feature Flags
// These are separate flags: INODE_UNINIT is 0x0001, while INODE_ZEROED is
// used for an initialized (and zero-filled) inode table.
pub const EXT4_BG_INODE_ZEROED: u16 = 0x0004;
pub const EXT4_FEATURE_INCOMPAT_FILETYPE: u32 = 0x0002;
pub const EXT4_FEATURE_INCOMPAT_EXTENTS: u32 = 0x0040;
pub const EXT4_FEATURE_COMPAT_EXT_ATTR: u32 = 0x0008;
pub const EXT4_FEATURE_RO_COMPAT_SPARSE_SUPER: u32 = 0x0001;

#[repr(C, packed)]
#[derive(Clone, Copy, Debug, Immutable, IntoBytes, KnownLayout)]
pub struct Ext4SuperBlock {
    pub s_inodes_count: u32,
    pub s_blocks_count_lo: u32,
    pub s_r_blocks_count_lo: u32,
    pub s_free_blocks_count_lo: u32,
    pub s_free_inodes_count: u32,
    pub s_first_data_block: u32, // 0 for 4KB blocks
    pub s_log_block_size: u32,   // 2 for 4KB (1024 << 2)
    pub s_log_cluster_size: u32, // 2
    pub s_blocks_per_group: u32,
    pub s_clusters_per_group: u32,
    pub s_inodes_per_group: u32,
    pub s_mtime: u32,
    pub s_wtime: u32,
    pub s_mnt_count: u16,
    pub s_max_mnt_count: u16,
    pub s_magic: u16, // 0xEF53
    pub s_state: u16,
    pub s_errors: u16,
    pub s_minor_rev_level: u16,
    pub s_lastcheck: u32,
    pub s_checkinterval: u32,
    pub s_creator_os: u32, // 0 = Linux
    pub s_rev_level: u32,  // 1 = Dynamic revision
    pub s_def_resuid: u16,
    pub s_def_resgid: u16,
    // Revision 1 fields
    pub s_first_ino: u32,  // 11
    pub s_inode_size: u16, // 256
    pub s_block_group_nr: u16,
    pub s_feature_compat: u32,
    pub s_feature_incompat: u32,
    pub s_feature_ro_compat: u32,
    pub s_uuid: [u8; 16],
    pub s_volume_name: [u8; 16],
    pub s_last_mounted: [u8; 64],
    pub s_algorithm_usage_bitmap: u32,
    pub _padding: [u8; 820],
}

impl Default for Ext4SuperBlock {
    fn default() -> Self {
        Self {
            s_inodes_count: 0,
            s_blocks_count_lo: 0,
            s_r_blocks_count_lo: 0,
            s_free_blocks_count_lo: 0,
            s_free_inodes_count: 0,
            s_first_data_block: 0, // 0 for 4KB blocks (1 for 1KB blocks)
            s_log_block_size: 2,   // 2 = 4096 bytes (1024 << 2)
            s_log_cluster_size: 2, // Must match s_log_block_size (unless bigalloc)
            s_blocks_per_group: BLOCKS_PER_GROUP as u32, // Standard max blocks per group for 4KB blocks
            s_clusters_per_group: BLOCKS_PER_GROUP as u32,
            s_inodes_per_group: 0, // Dynamically calculated during geometry phase
            s_mtime: 0,
            s_wtime: 0,
            s_mnt_count: 0,
            s_max_mnt_count: 0xFFFF,   // -1 (disabled mount count check)
            s_magic: EXT4_SUPER_MAGIC, // EXT4 Magic Number
            s_state: 1,                // EXT2_VALID_FS (Clean)
            s_errors: 1,               // EXT2_ERRORS_CONTINUE
            s_minor_rev_level: 0,
            s_lastcheck: 0,
            s_checkinterval: 0,
            s_creator_os: 0, // 0 = Linux
            s_rev_level: 1,  // 1 = Dynamic revision
            s_def_resuid: 0,
            s_def_resgid: 0,
            s_first_ino: 11,   // EXT4_FIRST_INO
            s_inode_size: 256, // Standard ext4 inode size
            s_block_group_nr: 0,
            // External xattr blocks are referenced through i_file_acl_lo.
            s_feature_compat: EXT4_FEATURE_COMPAT_EXT_ATTR,
            s_feature_incompat: EXT4_FEATURE_INCOMPAT_FILETYPE | EXT4_FEATURE_INCOMPAT_EXTENTS,
            // The allocator below places backups only in sparse-super groups.
            s_feature_ro_compat: EXT4_FEATURE_RO_COMPAT_SPARSE_SUPER,
            s_uuid: [0; 16],
            s_volume_name: [0; 16],
            s_last_mounted: [0; 64],
            s_algorithm_usage_bitmap: 0,
            _padding: [0; 820],
        }
    }
}

#[repr(C, packed)]
#[derive(Clone, Copy, Debug, Default, Immutable, IntoBytes, KnownLayout)]
pub struct Ext4GroupDesc {
    pub bg_block_bitmap_lo: u32,
    pub bg_inode_bitmap_lo: u32,
    pub bg_inode_table_lo: u32,
    pub bg_free_blocks_count_lo: u16,
    pub bg_free_inodes_count_lo: u16,
    pub bg_used_dirs_count_lo: u16,
    pub bg_flags: u16,
    pub bg_exclude_bitmap_lo: u32,
    pub bg_block_bitmap_csum_lo: u16,
    pub bg_inode_bitmap_csum_lo: u16,
    pub bg_itable_unused_lo: u16,
    pub bg_checksum: u16,
}

#[repr(C, packed)]
#[derive(Clone, Copy, Debug, FromBytes, Immutable, IntoBytes, KnownLayout)]
// https://github.com/torvalds/linux/blob/master/fs/ext4/ext4.h#L794
pub struct Ext4Inode {
    pub i_mode: u16,
    pub i_uid: u16,
    pub i_size_lo: u32,
    pub i_atime: u32,
    pub i_ctime: u32,
    pub i_mtime: u32,
    pub i_dtime: u32,
    pub i_gid: u16,
    pub i_links_count: u16,
    pub i_blocks_lo: u32, // 512-byte sector count
    pub i_flags: u32,     // 0x80000 = EXT4_EXTENTS_FL
    pub i_osd1: u32,
    pub i_block: [u8; 60], // Extent tree root or block pointers
    pub i_generation: u32,
    pub i_file_acl_lo: u32,
    pub i_size_high: u32,
    pub i_obso_faddr: u32,
    pub i_osd2: [u8; 12], // Completes standard 128-byte base inode
    // Extended Inode Fields (for 256-byte inode size)
    pub i_extra_isize: u16, // Typically set to 32 (size of extra fields)
    pub i_checksum_hi: u16,
    pub i_ctime_extra: u32,
    pub i_mtime_extra: u32,
    pub i_atime_extra: u32,
    pub i_crtime: u32, // File creation time
    pub i_crtime_extra: u32,
    pub i_version_hi: u32,
    pub i_projid: u32,
    pub _padding: [u8; 96], // 128 + 32 + 96 = 256 bytes total
}

impl Default for Ext4Inode {
    fn default() -> Self {
        unsafe { std::mem::zeroed() }
    }
}

#[repr(C, packed)]
#[derive(Clone, Copy, Debug, FromBytes, Immutable, IntoBytes, KnownLayout)]
pub struct Ext4ExtentHeader {
    pub eh_magic: u16,   // 0xF30A
    pub eh_entries: u16, // Number of valid entries
    pub eh_max: u16,     // Max capacity
    pub eh_depth: u16,   // 0 for leaf nodes
    pub eh_generation: u32,
}

#[repr(C, packed)]
#[derive(Clone, Copy, Debug, FromBytes, Immutable, IntoBytes, KnownLayout)]
pub struct Ext4Extent {
    pub ee_block: u32,    // First logical block
    pub ee_len: u16,      // Number of blocks
    pub ee_start_hi: u16, // High 16 bits of physical block (0 for <16TB)
    pub ee_start_lo: u32, // Low 32 bits of physical block
}

#[repr(C, packed)]
#[derive(Clone, Copy, Debug, FromBytes, Immutable, IntoBytes, KnownLayout)]
pub struct Ext4DirEntry2Header {
    pub inode: u32,
    pub rec_len: u16,
    pub name_len: u8,
    pub file_type: u8, // 1 = file, 2 = directory
}

/// `struct ext4_xattr_header`, used at the start of an external xattr block.
#[repr(C, packed)]
#[derive(Clone, Copy, Debug, Default, Immutable, IntoBytes, KnownLayout)]
pub struct Ext4XattrHeader {
    pub h_magic: u32,
    pub h_refcount: u32,
    pub h_blocks: u32,
    pub h_hash: u32,
    pub h_checksum: u32,
    pub h_reserved: [u32; 3],
}

/// `struct ext4_xattr_entry`; the un-terminated name follows this header.
#[repr(C, packed)]
#[derive(Clone, Copy, Debug, Default, Immutable, IntoBytes, KnownLayout)]
pub struct Ext4XattrEntry {
    pub e_name_len: u8,
    pub e_name_index: u8,
    pub e_value_offs: u16,
    pub e_value_block: u32,
    pub e_value_size: u32,
    pub e_hash: u32,
}

// Represents a single 128MB Section
pub struct Ext4BlockGroup {
    pub group_id: u32,
    pub blocks_count: u32,
    pub block_bitmap: Vec<u8>,
    pub inode_bitmap: Vec<u8>,
    pub inode_table: Vec<Ext4Inode>,
    pub free_blocks_count: u16,
    pub free_inodes_count: u16,
}
