# Supported features

## Current status

The formatter currently creates valid ext4 images with a 4 KiB block size and
256-byte inodes. It writes sparse-superblock layout (including backups and
group descriptors), block and inode bitmaps, inode tables, the root directory,
and `lost+found`. Every image carries a random filesystem UUID (`s_uuid`),
as `mke2fs` writes one, alongside the random directory-hash seed, and every
image of at least 8 MiB a jbd2 journal.

It supports regular files across fragmented block runs, building multi-level
extent trees when needed. Directories have `.` and `..` entries, update parent
link counts, and grow across multiple blocks, including fragmented
extent-backed directories. Symlinks use inline storage below 60 bytes and one
data-block extent for longer targets.

Character and block devices, FIFOs, and Unix-domain sockets use
Linux-compatible device-number encoding. Files and directories support POSIX
permissions, 32-bit UID/GID, access/change/modification timestamps, external
xattr blocks with deterministic e2fsprogs-compatible hashes, and conversion
of `system.posix_acl_access` / `system.posix_acl_default` into compact ext4
ACL encoding.

The demonstration image contains:

```text
/
├── lost+found/
└── test_dir/
    └── hello.txt
```

`hello.txt` contains `Created by ext4kit` and has `user.creator=ext4kit`.

## Tracked folders

`create` records folder tracking when every `--add` source is inside a git
repository and no two additions overlap (image targets or repo-relative host
paths): a sidecar `<image>.ext4kit-tracked` ties the image's UUID to a label
and its mappings, and one baseline commit per repository lands on
`refs/ext4kit/<label>/base`, parented on HEAD. The baseline holds exactly the
imported content — normalized with git's clean-conversion rules (eol
attributes, `core.autocrlf`, `filter=`/`ident` routes through
`git hash-object`) and filtered with the repository's ignore rules, which a
per-label `--include-ignored` at create turns off. Nested repositories and
submodule pointers are boundaries like they are for git itself: a gitlink is
re-emitted unchanged, a nested clone's files never enter the outer commit. A
mapping root the walk cannot resolve or list (replaced by a symlink,
unlistable) skips the mapping — content the walk could not see is never
deleted from the baseline. HEAD paths whose names cannot exist on the host
(a control byte, a reserved Windows device name) are kept from HEAD with
their raw bytes instead of being deleted, and names the walk classifies as
host-incompatible are reported, never auto-renamed.
The `ext4kit snapshot` command that turns image state into commits on
`refs/ext4kit/<label>/head` follows in the next release.

Re-creating over a tracked image keeps the previous baseline refs under
`refs/ext4kit/<label>/previous/<timestamp>/` and refuses to discard
unsnapshotted work (or a still-mounted image) without the explicit
`--discard-unsnapshotted` / `--replace-mounted` overrides. A sidecar whose
recorded image UUID does not match the image is foreign: it is reported and
removed, its refs are never moved, and a label it happens to use still
collides.

## Metadata API

Creation methods accept `PosixMetadata`:

```rust
let metadata = PosixMetadata {
    mode: 0o640,
    uid: 1000,
    gid: 1000,
    mtime: 1_700_000_000,
    ..Default::default()
};

let directory = builder.mkdir(EXT4_ROOT_INO, "app", Some(&metadata))?;
builder.add_file(directory, "config", b"enabled=true\n", &metadata)?;
builder.add_symlink(directory, "current", "config", &metadata)?;
builder.add_char_device(directory, "null", 1, 3, &metadata)?;
builder.add_fifo(directory, "events", &metadata)?;
```

Xattr keys use fully qualified Linux names such as `user.comment`,
`security.capability`, `trusted.overlay.opaque`, and
`system.posix_acl_access`. Each inode currently gets one unshared external
4 KiB xattr block; oversized sets fail before allocation state changes.

Regular files and directories may span fragmented runs. Extent-tree metadata
is added automatically when extents do not fit in the inode. Directory growth
reserves its data and required extent-tree blocks before creating a child, and
reclaims replaced extent-tree blocks. Directories that outgrow their first
block are converted to hashed (dx) indexes. `lost+found` is created as a
multi-block linear directory, as mke2fs writes it, and is not converted: once
its four blocks are full, adding to it fails with an error. How many entries
fit depends on name length, from about 1,000 short names down to about 60
names of 255 bytes.

Symlink targets hold from 1 to `BLOCK_SIZE - 1` bytes (4,095 bytes with
4 KiB blocks). An empty target is rejected, as `symlink(2)` does: e2fsck
reports a zero-length symlink as invalid, and recent Linux kernels refuse to
load one. Targets under 60 bytes are inline; longer targets occupy one data 
block. Inline-data symlinks are not implemented.

Device methods accept a Linux major number up to 4,095 and minor number up to
1,048,575. Special nodes do not allocate data extents, though they may use an
external xattr block.

## Editing existing images

`Ext4ImageBuilder::open` loads an image that ext4kit wrote back into the
builder, and `ext4kit edit` exposes path-level operations on it: `rm`,
`mkdir`, `chmod`, `chown` (recursive without following symlinks), `add`,
`write` (in-place content replacement that keeps the inode, links, timestamps
and xattrs), `cat`, `test` and `ln`. Images in any other format, including
those made by `mke2fs`, are rejected. The image size is fixed. See
[Editing images](editing.md).

## Design constraints and limitations

- Image creation runs on Windows; Linux tools are only optional verification.
- Layout is modeled on `e2fsprogs` and written directly using ext4 structures.
- The implementation uses 32-bit block addresses and 32-byte group descriptors.
- Every image of at least 8 MiB carries a jbd2 journal (v2 superblock,
  big-endian, no checksum features; mke2fs-compatible sizes), so the guest
  kernel recovers a crashed VM by replaying it on mount. Smaller images get
  none, mirroring mke2fs. The snapshot reader never replays a journal; a
  non-empty one is refused.
- Sparse files, inline/shared xattrs, and SELinux validation are not
  implemented yet.
- Host import preserves the portable subset of metadata supported by the host
  OS. OCI import applies layer metadata, whiteouts (including opaque
  directories), and hard links; only linux/amd64 images and anonymous registry
  access are supported.
