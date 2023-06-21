use std::fmt;

use approx;
use derive_builder::Builder;
use ndarray::Array1;
use ndarray_linalg::types::Lapack;
use num_complex::{Complex, ComplexFloat};
use num_traits::float::{Float, FloatConst};

use crate::aux::molecule::Molecule;

#[cfg(test)]
mod vibration_tests;

pub mod vibration_analysis;
mod vibration_transformation;

// ==================
// Struct definitions
// ==================

/// A structure to manage vibrational coordinates.
#[derive(Builder, Clone)]
#[builder(build_fn(validate = "Self::validate"))]
pub struct VibrationalCoordinate<'a, T>
where
    T: ComplexFloat + Lapack,
{
    /// The associated molecule.
    mol: &'a Molecule,

    /// The coefficients describing this vibrational coordinate in terms of the local Cartesian
    /// coordinates.
    coefficients: Array1<T>,

    /// The threshold for comparing vibrational coordinates.
    threshold: <T as ComplexFloat>::Real,
}

impl<'a, T> VibrationalCoordinateBuilder<'a, T>
where
    T: ComplexFloat + Lapack,
{
    fn validate(&self) -> Result<(), String> {
        let coefficients = self
            .coefficients
            .as_ref()
            .ok_or("No coefficients found.".to_string())?;
        let mol = self.mol.ok_or("No molecule found.".to_string())?;
        let natoms = 3 * mol.atoms.len() == coefficients.len();
        if !natoms {
            log::error!("The number of coefficients for this vibrational coordinate does not match the number of atoms in the molecule.");
        }

        if natoms {
            Ok(())
        } else {
            Err("Vibrational coordinate validation failed.".to_string())
        }
    }
}

impl<'a, T> VibrationalCoordinate<'a, T>
where
    T: ComplexFloat + Clone + Lapack,
{
    /// Returns a builder to construct a new [`VibrationalCoordinate`].
    pub fn builder() -> VibrationalCoordinateBuilder<'a, T> {
        VibrationalCoordinateBuilder::default()
    }

    /// Returns a shared reference to the coefficient array.
    pub fn coefficients(&self) -> &Array1<T> {
        &self.coefficients
    }
}

// =====================
// Trait implementations
// =====================

// ----
// From
// ----
impl<'a, T> From<VibrationalCoordinate<'a, T>> for VibrationalCoordinate<'a, Complex<T>>
where
    T: Float + FloatConst + Lapack,
    Complex<T>: Lapack,
{
    fn from(value: VibrationalCoordinate<'a, T>) -> Self {
        VibrationalCoordinate::<'a, Complex<T>>::builder()
            .coefficients(value.coefficients.map(Complex::from))
            .mol(value.mol)
            .threshold(value.threshold)
            .build()
            .expect("Unable to construct a complex vibrational coordinate.")
    }
}

// ---------
// PartialEq
// ---------
impl<'a, T> PartialEq for VibrationalCoordinate<'a, T>
where
    T: ComplexFloat<Real = f64> + Lapack,
{
    fn eq(&self, other: &Self) -> bool {
        let thresh = (self.threshold * other.threshold).sqrt();
        let coefficients_eq = approx::relative_eq!(
            (&self.coefficients - &other.coefficients)
                .map(|x| ComplexFloat::abs(*x).powi(2))
                .sum()
                .sqrt(),
            0.0,
            epsilon = thresh,
            max_relative = thresh,
        );
        self.mol == other.mol && coefficients_eq
    }
}

// --
// Eq
// --
impl<'a, T> Eq for VibrationalCoordinate<'a, T> where T: ComplexFloat<Real = f64> + Lapack {}

// -----
// Debug
// -----
impl<'a, T> fmt::Debug for VibrationalCoordinate<'a, T>
where
    T: fmt::Debug + ComplexFloat + Lapack,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "VibrationalCoordinate[coefficient array of length {}]",
            self.coefficients.len()
        )?;
        Ok(())
    }
}

// -------
// Display
// -------
impl<'a, T> fmt::Display for VibrationalCoordinate<'a, T>
where
    T: fmt::Display + ComplexFloat + Lapack,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "VibrationalCoordinate[coefficient array of length {}]",
            self.coefficients.len()
        )?;
        Ok(())
    }
}
