//! Name-based access to an image: resolving paths through directories and
//! symlinks, and coreutils-style edits on top of the inode-level operations
//! in [`crate::ext4`].

pub mod edit;
mod resolve;
