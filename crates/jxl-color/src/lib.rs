//! This crate provides a set of functions related to color encodings defined by the JPEG XL
//! specification.
//!
//! # Color transformation
//! jxl-color can handle the transformation between "well-known" color encodings without any
//! external color management system, since it is required by the specification. Such
//! transformations can be done by creating a [`ColorTransform`].
//!
//! # Modules
//! - [`consts`] defines constants used by the various colorspaces.
//! - [`icc`] provides functions related to ICC profiles.
#![cfg_attr(not(feature = "std"), no_std)]
#![allow(unsafe_op_in_unsafe_fn)]

extern crate alloc;

#[cfg(all(not(feature = "std"), not(feature = "libm")))]
compile_error!("enable the libm feature when building jxl-color without std");

// Keep host dispatch under std and use only guaranteed CPU features without std.
#[cfg(all(target_arch = "x86_64", feature = "std"))]
macro_rules! is_x86_feature_detected {
    ($feature:tt) => {
        std::arch::is_x86_feature_detected!($feature)
    };
}

#[cfg(all(target_arch = "x86_64", not(feature = "std")))]
macro_rules! is_x86_feature_detected {
    ($feature:tt) => {
        cfg!(target_feature = $feature)
    };
}

#[cfg(all(target_arch = "aarch64", feature = "std"))]
macro_rules! is_aarch64_feature_detected {
    ($feature:tt) => {
        std::arch::is_aarch64_feature_detected!($feature)
    };
}

#[cfg(all(target_arch = "aarch64", not(feature = "std")))]
macro_rules! is_aarch64_feature_detected {
    ($feature:tt) => {
        cfg!(target_feature = $feature)
    };
}

mod ciexyz;
mod cms;
pub mod consts;
mod convert;
mod error;
mod fastmath;
mod gamut;
pub mod icc;
mod math;
mod tf;
mod xyb;
mod ycbcr;

pub use jxl_image::color::*;

pub use ciexyz::{AsIlluminant, AsPrimaries};
pub use cms::*;
pub use convert::*;
pub use error::*;
pub use ycbcr::ycbcr_to_rgb;
