use num_complex::Complex;
use numpy::{PyArray2, PyArray4};
use pyo3::prelude::*;

pub mod slater_determinant;
pub mod vibrational_coordinate;

type C128 = Complex<f64>;

/// A Python-exposed enumerated type to handle the union type of numpy float 2d-arrays and numpy
/// complex 2d-arrays in Python.
#[derive(FromPyObject)]
pub enum PyArray2RC<'a> {
    Real(&'a PyArray2<f64>),
    Complex(&'a PyArray2<C128>),
}

/// A Python-exposed enumerated type to handle the union type of numpy float 4d-arrays and numpy
/// complex 4d-arrays in Python.
#[derive(FromPyObject)]
pub enum PyArray4RC<'a> {
    Real(&'a PyArray4<f64>),
    Complex(&'a PyArray4<C128>),
}
