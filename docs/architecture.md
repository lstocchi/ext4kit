# Architecture

`ext4kit` is organized in layers; each uses only the layers before it.

* `ext4/format/` holds the on-disk wire structures and constants, the
  directory hash, and xattr/ACL encoding. It is plain data with no builder
  state.
* `ext4/` is the filesystem engine. `Ext4ImageBuilder` owns image layout,
  loading, allocation, inode operations, directory records, extent trees and
  node creation. It works on inode numbers.
* `vfs/` gives name-based access: it resolves paths through directories and
  symlinks, confined to the image root like a chroot, and implements the
  coreutils-style operations behind `ext4kit edit`.
* `host/` imports host files, directories, and symlinks (`tree.rs` holds the pre-scanned `HostTree` arena used for sizing).
* `oci/` streams OCI image layers from a registry or a local container engine.

CLI image creation uses a destination-file-backed mapping. Ext4 blocks are
written into that mapping as they are allocated, and regular-file payloads are
copied from readers directly into their extents. This avoids both an
image-sized heap buffer and a second full-image copy at finalization. The
in-memory builder remains available for small library uses and unit tests.

Host import has two passes. `dua-core` streams a bounded, parallel filesystem
walk into a deterministic flat arena containing paths, parent indices, link
targets, and compact platform-neutral metadata. The import pass walks that
arena sequentially; parents therefore exist before children and neither pass
uses recursion. This structure is also intended to accept OCI-layer entries
without changing the ext4-writing path.

When the caller supplies the image size, the sizing arena is unnecessary.
`dua-core`'s parent-first stream is imported directly while retaining only a
map from live directory IDs to ext4 inode numbers. The CLI selects this path
with `create --size MIB`.

`Ext4ImageBuilder::open` maps an existing image and loads its superblock,
group descriptors, bitmaps and inode tables back into the builder, so the
same operations can edit an image in place. Because `write_to_file` rewrites
all metadata from the builder's model, `open` accepts only images in
ext4kit's own format: the exact feature set it writes, 4 KiB blocks, 256-byte
inodes, no reserved GDT blocks, a clean state and counters that match the
bitmaps. Images made by `mke2fs` (journal, `flex_bg`, `64bit`, checksums) are
rejected rather than silently stripped of metadata.

An agent harness will become a sibling component to `ext4` when concrete functionality exists.
