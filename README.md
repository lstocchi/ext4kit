# ext4kit

`ext4kit` is a Windows-native Rust toolkit for creating ext4 filesystem images
without requiring Linux, WSL, or `mke2fs` at image-creation time.

Its purpose is to make application files available to a Linux VM through a
real ext4 disk image instead of a shared-folder filesystem. The VM can mount
the image directly, retaining normal Linux filesystem semantics and avoiding
the performance overhead often associated with host/guest file sharing.

## Project goal

The intended end-to-end workflows are:

1. Extract a root filesystem from an OCI image and write its files into a
   valid ext4 image.
2. Copy or inject an application directory that lives on the Windows host
   into a valid ext4 image.
3. Attach that image to a Linux VM, mount it as ext4, and run the application
   from the VM at native filesystem speed.

For example, the eventual CLI should support a workflow conceptually like:

```text
ext4kit from-oci ghcr.io/example/service:latest service.ext4
ext4kit inject C:\work\my-app app-data.ext4 --destination /opt/my-app
```

The generated image can then be attached as a virtual disk and mounted by the
guest:

```sh
mount -t ext4 /dev/vdb /mnt/app
```

## Current status

The project currently creates a valid ext4 image containing:

- a 4 KiB block size and 256-byte inodes;
- extents and directory-entry file types;
- sparse-superblock layout, including backup superblocks and group descriptor
  tables;
- initialized block and inode bitmaps, inode tables, the root directory, and
  `lost+found`.
- regular-file creation across fragmented block runs, with multi-level ext4
  extent trees when required;
- symbolic-link creation, using fast symlinks for targets shorter than 60 bytes
  and a single data-block extent for longer targets;
- directory creation and automatic growth across multiple blocks, including
  `.` / `..` entries, parent link-count updates, and fragmented
  extent-backed directories;
- character-device and block-device nodes, FIFOs, and Unix-domain socket
  nodes, with Linux-compatible device-number encoding;
- POSIX metadata for created files and directories: permissions, 32-bit UID
  and GID, and access/change/modification timestamps;
- external ext4 xattr blocks, including deterministic entry ordering and
  e2fsprogs-compatible entry and block hashes; and
- conversion of Linux `system.posix_acl_access` and
  `system.posix_acl_default` xattr values into ext4's compact ACL encoding.

The demonstration image currently contains:

```text
/
├── lost+found/
└── test_dir/
    └── hello.txt
```

`hello.txt` contains `Created by ext4kit` and has the xattr
`user.creator=ext4kit`.

The current prototype does **not yet** import OCI layers, traverse host
directories, or expose the planned `from-oci` / `inject` commands. It is the
filesystem formatter and metadata foundation those features will build on.

## Build and create an image

Install a current Rust toolchain, then run:

```powershell
cargo run -- output.img
```

This currently creates a 400 MiB ext4 image named `output.img` in the current
directory. An output path can be supplied as the first argument. The image
size and sample tree are currently defined in `src/main.rs`; command-line OCI
or host-directory import has not been implemented yet.

## Metadata API

Library code supplies `PosixMetadata` to creation methods:

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

Xattr keys use their fully qualified Linux names, such as `user.comment`,
`security.capability`, `trusted.overlay.opaque`, and
`system.posix_acl_access`. Xattrs currently use one unshared external 4 KiB
block per inode; oversized xattr sets are rejected before filesystem
allocation state is changed. Regular files and directories may span fragmented
runs; ext4 extent-tree metadata is created automatically when their extents no
longer fit in the inode. Directory growth reserves both the new directory
block and any required extent-tree metadata before creating the child entry,
and reclaims replaced extent-tree blocks. Symlink targets may be empty and are
limited to `BLOCK_SIZE - 1`
bytes (4,095 bytes with the current 4 KiB block size). Targets shorter than 60
bytes are stored directly in the inode; longer targets are stored in a single
data block. Inline-data symlinks are not implemented.

`add_char_device` and `add_block_device` take Linux major and minor device
numbers. The supported range is a major number up to 4,095 and a minor number
up to 1,048,575. Special-file nodes do not allocate data extents, though they
may have an external xattr block.

## Verify the image in Linux or WSL

Validation should always include a read-only filesystem check:

```sh
e2fsck -fn output.img
debugfs -R 'ls -l /test_dir' output.img
debugfs -R 'ea_list /test_dir/hello.txt' output.img
```

`e2fsck -fn` must complete without proposing repairs. `debugfs` should list
`hello.txt` and display its `user.creator` xattr. Images containing symlinks
should show them with type `120777` in `debugfs -R 'ls -l …'` output.
Special files should appear as their corresponding Linux types, for example
`20660` for a character device and `10644` for a FIFO. For fragmented files,
`debugfs -R 'stat /path/to/file' output.img` can be used to inspect the extent
tree. Use `debugfs -R 'stat /path/to/directory' output.img` to confirm a
grown directory's size, block count, and directory extents.

## Design constraints

- Image creation must run on Windows; Linux tooling is used only for optional
  verification.
- The filesystem layout is modeled on `e2fsprogs` and written directly to the
  image buffer using on-disk ext4 structures.
- The current implementation uses 32-bit block addresses and 32-byte group
  descriptors.
- Images are deliberately created without a journal at this stage.
- The formatter supports regular files and directories (including fragmented
  multi-extent trees), symlinks, device nodes, FIFOs, and sockets. It does not
  yet create hard links or sparse files.
- Xattr block sharing/deduplication, inline xattrs, SELinux policy validation,
  and OCI whiteout semantics are not implemented.
- UUID generation and OCI/host metadata extraction are still planned. The
  caller is responsible for supplying validated `PosixMetadata` values.

## Development

```powershell
cargo fmt --check
cargo test
```

When changing allocation or metadata code, regenerate an image and run
`e2fsck -fn` after every change. A clean Rust build is not enough to establish
on-disk ext4 correctness.
