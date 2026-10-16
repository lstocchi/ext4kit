# ext4kit

`ext4kit` is a Rust toolkit for creating real ext4 filesystem images without requiring Linux or `mke2fs` at image-creation time.

It is intended to make application files available to Linux VMs through ext4 disk images rather than shared folders.

## Status

The formatter currently supports files, directories, symlinks, special nodes, extents, POSIX metadata, xattrs, and POSIX ACLs. OCI import, host-directory traversal, and a complete CLI are planned.

## Quick start

```powershell
cargo run -- output.img
```

This creates a 400 MiB demonstration image. The sample tree and image size currently live in `src/main.rs`.

# Development and verification

Run:

```powershell
cargo fmt --check
cargo test
```

Validate generated images with:

```sh
e2fsck -fn output.img
debugfs -R 'ls -l /test_dir' output.img
debugfs -R 'ea_list /test_dir/hello.txt' output.img
```

`e2fsck -fn` must not propose repairs. Regenerate and inspect an image whenever allocation or metadata code changes.

`debugfs` should list `hello.txt` and display its `user.creator` xattr. Images
containing symlinks should show them with type `120777` in `debugfs -R 'ls -l
…'` output. Special files should appear as their corresponding Linux types,
such as `20660` for a character device and `10644` for a FIFO. For fragmented
files, run `debugfs -R 'stat /path/to/file' output.img` to inspect the extent
tree. Run `debugfs -R 'stat /path/to/directory' output.img` to confirm a grown
directory's size, block count, and directory extents.

## Documentation

- [Architecture](docs/architecture.md)
- [Supported features](docs/supported-features.md)
- [Roadmap and intended workflows](docs/roadmap.md)
