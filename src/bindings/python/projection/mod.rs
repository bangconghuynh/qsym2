//! Python bindings for QSym² symmetry projection via representation theory.
//!
//! See [`crate::drivers::projection`] for more information.

use pyo3::FromPyObject;

pub mod density;
pub mod slater_determinant;

/// Python-exposed enumerated type to handle the union type `str | int`.
#[derive(FromPyObject)]
pub enum PyProjectionTarget {
    Symbolic(String),
    Numeric(usize),
}
