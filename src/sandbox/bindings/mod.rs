//! Sandbox binding implementations to expose QSym² to other languages.

#[cfg(not(tarpaulin_include))]
#[cfg(feature = "python")]
pub mod python;
