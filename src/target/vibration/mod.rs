//! Vibrational coordinates for normal modes.

use std::fmt;

use approx;
use derive_builder::Builder;
use ndarray::{Array1, Array2};
use ndarray_linalg::types::Lapack;
use num_complex::{Complex, ComplexFloat};
use num_traits::float::{Float, FloatConst};

use crate::auxiliary::molecule::Molecule;

#[cfg(test)]
mod vibration_tests;

pub mod vibration_analysis;
mod vibration_transformation;

// ==================
// Struct definitions
// ==================

// ---------------------
// VibrationalCoordinate
// ---------------------

/// Structure to manage vibrational coordinates.
#[derive(Builder, Clone)]
#[builder(build_fn(validate = "Self::validate"))]
pub struct VibrationalCoordinate<'a, T>
where
    T: ComplexFloat + Lapack,
{
    /// The associated molecule.
    mol: &'a Molecule,

    /// The frequency of this vibration in $`\mathrm{cm}^{-1}`$.
    frequency: T,

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

    /// Returns the frequency of the vibration.
    pub fn frequency(&self) -> &T {
        &self.frequency
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
            .frequency(Complex::from(value.frequency))
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
        let frequency_eq = approx::relative_eq!(
            ComplexFloat::abs(self.frequency - other.frequency),
            0.0,
            epsilon = thresh,
            max_relative = thresh,
        );
        self.mol == other.mol && coefficients_eq && frequency_eq
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
            "VibrationalCoordinate[frequency = {:?}, coefficient array of length {}]",
            self.frequency,
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
            "VibrationalCoordinate[frequency = {}, coefficient array of length {}]",
            self.frequency,
            self.coefficients.len()
        )?;
        Ok(())
    }
}

// -------------------------------
// VibrationalCoordinateCollection
// -------------------------------

/// Structure to manage multiple vibrational coordinates of a single molecule.
#[derive(Builder, Clone)]
#[builder(build_fn(validate = "Self::validate"))]
pub struct VibrationalCoordinateCollection<'a, T>
where
    T: ComplexFloat + Lapack,
{
    /// The associated molecule.
    mol: &'a Molecule,

    /// The frequencies of the vibrations in $`\mathrm{cm}^{-1}`$.
    frequencies: Array1<T>,

    /// The coefficients describing this vibrational coordinates in terms of the local Cartesian
    /// coordinates. Each column corresponds to one vibrational coordinate.
    coefficients: Array2<T>,

    /// The threshold for comparing vibrational coordinates.
    threshold: <T as ComplexFloat>::Real,
}

impl<'a, T> VibrationalCoordinateCollectionBuilder<'a, T>
where
    T: ComplexFloat + Lapack,
{
    fn validate(&self) -> Result<(), String> {
        let coefficients = self
            .coefficients
            .as_ref()
            .ok_or("No coefficients found.".to_string())?;
        let mol = self.mol.ok_or("No molecule found.".to_string())?;
        let natoms = 3 * mol.atoms.len() == coefficients.nrows();
        if !natoms {
            log::error!("The number of coefficient components does not match the number of atoms in the molecule.");
        }

        let frequencies = self
            .frequencies
            .as_ref()
            .ok_or("No frequencies found.".to_string())?;
        let nfreqs = frequencies.len() == coefficients.ncols();

        if natoms && nfreqs {
            Ok(())
        } else {
            Err("Vibrational coordinate collection validation failed.".to_string())
        }
    }
}

impl<'a, T> VibrationalCoordinateCollection<'a, T>
where
    T: ComplexFloat + Clone + Lapack,
{
    /// Returns a builder to construct a new [`VibrationalCoordinateCollection`].
    pub fn builder() -> VibrationalCoordinateCollectionBuilder<'a, T> {
        VibrationalCoordinateCollectionBuilder::default()
    }

    /// Returns the number of vibrational modes in this collection.
    pub fn n_modes(&self) -> usize {
        self.frequencies.len()
    }

    /// Returns the frequencies of the vibrations.
    pub fn frequencies(&self) -> &Array1<T> {
        &self.frequencies
    }

    /// Returns a shared reference to the coefficient array.
    pub fn coefficients(&self) -> &Array2<T> {
        &self.coefficients
    }

    /// Returns a vector of separate vibrational coordinates from this collection.
    pub fn to_vibrational_coordinates(&self) -> Vec<VibrationalCoordinate<'a, T>> {
        self.frequencies
            .iter()
            .zip(self.coefficients.columns())
            .map(|(freq, vib_coord)| {
                VibrationalCoordinate::builder()
                    .coefficients(vib_coord.to_owned())
                    .frequency(*freq)
                    .mol(self.mol)
                    .threshold(self.threshold)
                    .build()
                    .expect("Unable to construct a vibrational coordinate.")
            })
            .collect::<Vec<_>>()
    }
}

// =====================
// Trait implementations
// =====================

// ----
// From
// ----
impl<'a, T> From<VibrationalCoordinateCollection<'a, T>>
    for VibrationalCoordinateCollection<'a, Complex<T>>
where
    T: Float + FloatConst + Lapack,
    Complex<T>: Lapack,
{
    fn from(value: VibrationalCoordinateCollection<'a, T>) -> Self {
        VibrationalCoordinateCollection::<'a, Complex<T>>::builder()
            .coefficients(value.coefficients.map(Complex::from))
            .mol(value.mol)
            .frequencies(value.frequencies.map(Complex::from))
            .threshold(value.threshold)
            .build()
            .expect("Unable to construct a complex vibrational coordinate collection.")
    }
}

// -----
// Debug
// -----
impl<'a, T> fmt::Debug for VibrationalCoordinateCollection<'a, T>
where
    T: fmt::Debug + ComplexFloat + Lapack,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let n = self.frequencies.len();
        write!(
            f,
            "VibrationalCoordinateCollection[{} {}, coefficient array of shape {} × {}]",
            n,
            if n == 1 { "mode" } else { "modes" },
            self.coefficients.nrows(),
            self.coefficients.ncols(),
        )?;
        Ok(())
    }
}

// -------
// Display
// -------
impl<'a, T> fmt::Display for VibrationalCoordinateCollection<'a, T>
where
    T: fmt::Display + ComplexFloat + Lapack,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let n = self.frequencies.len();
        write!(
            f,
            "VibrationalCoordinateCollection[{} {}, coefficient array of shape {} × {}]",
            n,
            if n == 1 { "mode" } else { "modes" },
            self.coefficients.nrows(),
            self.coefficients.ncols(),
        )?;
        Ok(())
    }
}
