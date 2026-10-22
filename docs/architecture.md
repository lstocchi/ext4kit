# Architecture

`ext4kit` is organized around a `core` boundary.

* `core/` owns filesystem-image construction, allocation, inode operations, directory records, extent trees, and node creation.
* `host/` imports host files, directories, and symlinks (`tree.rs` holds the pre-scanned `HostTree` arena used for sizing).
* `oci/` streams OCI image layers from a registry or a local container engine.
* `ext4.rs` holds the on-disk wire structures and constants.
* `xattr.rs` handles inode metadata shared by files, directories, links, and special nodes.

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

An agent harness will become a sibling component to `core` when concrete functionality exists.
