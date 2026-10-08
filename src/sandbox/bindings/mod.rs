//! Sandbox binding implementations to expose QSym² to other languages.

#![cfg_attr(coverage_nightly, feature(coverage_attribute))]

#[cfg_attr(coverage_nightly, coverage(off))]
#[cfg(feature = "python")]
pub mod python;
