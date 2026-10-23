//! Library interface for constructing and editing ext4 filesystem images.
//!
//! The binary in `main.rs` is deliberately kept as a small usage example;
//! callers can use this library to build their own images.
//!
//! The modules are layered, each using only the ones before it:
//!
//! - [`ext4`]: the filesystem itself. [`ext4::format`] holds the on-disk
//!   structures and encodings; the builder lays out, loads and changes an
//!   image by inode number.
//! - [`vfs`]: name-based access, resolving paths through directories and
//!   symlinks, and coreutils-style edits.
//! - [`host`] and [`oci`]: importers for host directories and OCI images.

pub mod ext4;
pub mod host;
pub mod oci;
pub mod vfs;

pub use ext4::{Ext4ImageBuilder, PosixMetadata};
