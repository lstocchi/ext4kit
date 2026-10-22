//! Library interface for constructing ext4 filesystem images.
//!
//! The binary in `main.rs` is deliberately kept as a small usage example;
//! callers can use this library to build their own images.

pub mod core;
pub mod ext4;
pub mod host;
pub mod oci;
pub mod xattr;

pub use core::{Ext4ImageBuilder, directory};
