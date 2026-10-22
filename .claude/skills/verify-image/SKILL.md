---
name: verify-image
description: Build an ext4 image with ext4kit and validate it with e2fsck/debugfs (natively or via WSL). Use after any change to on-disk structures, allocation, directories, extents, xattrs, host or OCI import, or when asked to check that a generated image is valid.
---

# Verify an ext4kit image end to end

Unit tests read images back through the builder, so they share its
assumptions. Only e2fsprogs can confirm that the image is really valid ext4.
Follow these steps whenever a change affects what is written to disk.

## 1. Pick inputs that exercise the change

Create the inputs in a fresh temp directory, never in the repo. Match the
fixture to the code you touched:

| Change area | Fixture |
|-------------|---------|
| Files and extents | One file larger than 512 MiB. An extent covers at most 128 MiB and the inode holds only 4, so this forces an extent-tree block. |
| Directory growth and htree | One directory with 5,000+ entries, plus names that collide or are long (up to 255 bytes) |
| Symlinks | Targets of 59, 60 and 4,095 bytes, and an empty target |
| Ownership / metadata | `--chown 1000:1000 --root-owner 1000:1000` |
| OCI import | `--from docker://docker.io/library/alpine:latest` (needs network) or `--from docker-daemon:IMAGE` (needs a local engine), combined with an `--add` that overlays an existing directory |
| Sizing | Run once without `--size` (sizing pass) and once with `--size` (streaming path) |

```sh
tmp=$(mktemp -d); command -v cygpath >/dev/null && tmp=$(cygpath -m "$tmp")  # Git Bash: C:/... paths
mkdir -p "$tmp/src/big"
for i in $(seq 1 5000); do : > "$tmp/src/big/file-$i"; done
head -c $((600*1024*1024)) /dev/urandom > "$tmp/src/large.bin"  # only for extent changes
```

## 2. Build the image

```sh
cargo run --release -q -- create --add "$tmp/src:/opt" "$tmp/out.img"
```

Use `--release` for large fixtures, because debug builds are much slower.

## 3. Check and inspect

```sh
bash scripts/verify-image.sh "$tmp/out.img" \
  'stats' 'htree /opt/src/big' 'stat /opt/src/large.bin' 'ls -l /opt/src'
```

- An exit code of 0 with no `e2fsck` complaints is the bar. Any message in
  passes 1–5 (wrong free counts, bad extent, dir entry, link count, htree
  problem) is a bug in ext4kit, even when the kernel would still mount the
  image.
- Use the `debugfs` output to confirm the *intended* structure as well: depth
  of the extent tree, whether the directory is indexed, mode, uid/gid and
  xattrs (`ea_list PATH`).
- On Windows the script runs through WSL and picks a distro that has
  e2fsprogs; set `EXT4KIT_WSL_DISTRO` to force one. If no tool is available,
  say so in your report rather than skipping verification silently.

## 4. Clean up and report

Run `rm -rf "$tmp"`. Report the exact command, the `e2fsck` result and any
`debugfs` facts that show the feature works. If you find a bug, add a unit
test in the relevant `#[cfg(test)] mod tests` that reproduces it before you
fix it.
