# ext4kit

`ext4kit` is a Rust toolkit for creating real ext4 filesystem images without requiring Linux or `mke2fs` at image-creation time.

It is intended to make application files available to Linux VMs through ext4 disk images rather than shared folders.

## Status

The formatter supports files, directories, symlinks, special nodes, extents, POSIX metadata, xattrs, POSIX ACLs, and host-directory traversal. OCI import and a complete CLI are planned.

## Quick start

```powershell
cargo run -- create --add C:\path\to\source:/etc output.img
```

By default, ext4kit scans the additions first and calculates an image size. To
choose the image size yourself and import in a single streaming traversal, pass
the size in MiB:

```powershell
cargo run -- create --size 3200 --add C:\path\to\source:/etc output.img
```

Both commands create `/etc/source` and import the contents of
`C:\path\to\source` there. Add `:/destination` to place a source directory below
an absolute path in the image. Omit it to create `/source` in the filesystem
root. Without `--size`, the image size is calculated automatically
from the source tree, including ext4 metadata and free space. Directory
discovery is streamed and file payloads are copied directly
into a ext4 file-backed image.

Repeat `--add` to compose an image from several sources:

```powershell
cargo run -- create --add C:\work\hosts:/etc --add C:\work\app:/opt output.img
```

In library code, initialize the root directory and then call
`builder.import_host_root(source)`, or use `import_host_directory(parent, source)`
to create an appropriately named child directory beneath an existing ext4
directory. Both methods return the imported directory inode. Regular files,
directories, and symbolic links are imported without following links.

Use `Ext4ImageBuilder::new_file(size_mib, output)` for file-backed construction.
`Ext4ImageBuilder::new(size_mib)` remains available when an in-memory image is
useful, such as small tests.


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
