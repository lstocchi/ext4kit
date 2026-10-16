# Architecture

`ext4kit` is organized around a `core` boundary. 
* `core/` owns filesystem-image construction, allocation, inode operations, directory records, extent trees, and node creation. 
* `ext4.rs` holds the on-disk wire structures and constants. 
* `xattr.rs` handles inode metadata shared by files, directories, links, and special nodes.

OCI import and an agent harness will become sibling components to `core` when concrete functionality exists.
