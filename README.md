# ext4kit

`ext4kit` is a Rust toolkit for creating real ext4 filesystem images without requiring Linux or `mke2fs` at image-creation time.

It is intended to make application files available to Linux VMs through ext4 disk images rather than shared folders.

## Status

The formatter supports files, directories, symlinks, special nodes, extents, POSIX metadata, xattrs, POSIX ACLs, host-directory traversal, and streaming OCI image import.

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

Use an image as the base filesystem with `--from`. An unqualified reference is
first looked up in a local Podman or Docker engine, then falls back to its OCI
registry when absent (only linux/amd64 images and anonymous registry access are
supported). Prefix a remote reference with `docker://` to skip the local
lookup. Layers are downloaded, digest-verified and extracted without touching
the host filesystem; later `--add` arguments can replace or augment paths from
the base image, merging into existing directories. The filesystem is sized
automatically from the image manifest (or the local engine's uncompressed image
size); an explicit `--size` overrides that estimate:

```powershell
cargo run -- create --from docker.io/library/alpine:latest `
  --add C:\work\app:/opt output.img
```

To require a local Docker or Podman image, use the `docker-daemon:` or
`containers-storage:` transport respectively. Local resolution honours
`DOCKER_HOST` / `CONTAINER_HOST` and otherwise tries the default Unix sockets
or Windows named pipes:

```powershell
cargo run -- create --from docker-daemon:alpine:latest output.img
```

Files added from Windows have no POSIX owner and are created as `root:root`
(on Unix hosts the host UID/GID is kept). Use `--chown` to give everything
imported with `--add` to the guest user, like a Dockerfile `COPY --chown`, so
it can be edited and built without `sudo chown` inside the VM. Destination
directories that ext4kit creates for an addition get the same owner, while
directories that already exist, from `--from` or an earlier `--add`, keep
theirs. `--from` image entries are never re-owned, since a base system relies
on its own ownership (`/etc/shadow`, setuid binaries, service accounts).

`--root-owner` sets the owner of the filesystem root directory, like
`mke2fs -E root_owner`, so the user can also write at the top of the mounted
disk. It works with or without `--add` and `--from`:

```powershell
cargo run -- create --chown 1000:1000 --root-owner 1000:1000 `
  --add C:\work\app:/home/test output.img
```

Here `/home`, `/home/test` and everything under `/home/test/app` belong to
`1000:1000`. With `--from`, an existing `/home` would stay `root:root`. Both
flags take numeric IDs only, because the user that matters is the one in the
VM the disk is attached to, not one in an image's `/etc/passwd`. A missing GID
defaults to the UID.

### Editing an image

`ext4kit edit` changes an image that `create` wrote, one coreutils-style
action per call, so fix-ups that would otherwise need a Linux VM run as a
plain script:

```powershell
cargo run -- edit output.img rm -f /opt/app/.initialized
cargo run -- edit output.img mkdir -p /sandbox
cargo run -- edit output.img chown -R 1000:1000 /sandbox
cargo run -- edit output.img cat /etc/passwd > passwd
cargo run -- edit output.img write passwd /etc/passwd
cargo run -- edit output.img test -x /bin/sh
```

The actions are `rm`, `mkdir`, `chmod`, `chown`, `add`, `write`, `cat`,
`test` and `ln`. They work on entries from `--from` as well as `--add`, and
only on images that ext4kit created. See [Editing images](docs/editing.md)
for their exact behaviour, failure handling and an OpenShell rootfs recipe.

In library code, initialize the root directory and then call
`builder.import_host_root(source)`, or use `import_host_directory(parent, source)`
to create an appropriately named child directory beneath an existing ext4
directory. Both methods return the imported directory inode. Regular files,
directories, and symbolic links are imported without following links.
Call `builder.set_host_owner(Some((uid, gid)))` before importing to get the
`--chown` behaviour (use `ensure_host_destination(path)` to create destination
directories with that owner), and `set_inode_owner(EXT4_ROOT_INO, uid, gid)`
for `--root-owner`.

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

## Testing

```sh
cargo test                           # unit tests plus end-to-end image tests
cargo test --test oci -- --ignored   # --from tests (Docker Hub and a Docker engine)
```

The end-to-end tests in `tests/` run the CLI and check every image with
`e2fsck` and `debugfs`. They find e2fsprogs on Linux, or through WSL on
Windows, and skip those checks with a note when it is missing. CI
(`.github/workflows/ci.yml`) runs the tests on Linux and Windows, and runs the
`--from` tests on Linux.

## Documentation

- [Architecture](docs/architecture.md)
- [Supported features](docs/supported-features.md)
- [Editing images](docs/editing.md)
- [Roadmap and intended workflows](docs/roadmap.md)
