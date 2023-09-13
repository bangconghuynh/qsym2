use std::fmt;

use approx;
use derive_builder::Builder;
use nalgebra::Vector3;
use ndarray_linalg::types::Lapack;
use num_complex::{Complex, ComplexFloat};
use num_traits::float::{Float, FloatConst};

#[cfg(test)]
#[path = "axialvector_tests.rs"]
mod axialvector_tests;

#[path = "axialvector_analysis.rs"]
pub mod axialvector_analysis;
#[path = "axialvector_transformation.rs"]
mod axialvector_transformation;

/// an enumerated type to handle the two possible time parities.
#[derive(Debug, PartialEq, Eq, Clone)]
pub enum TimeParity {
    /// Variant for time-even.
    Even,

    /// Variant for time-odd.
    Odd,
}

// -------
// Display
// -------
impl fmt::Display for TimeParity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TimeParity::Even => write!(f, "time-even"),
            TimeParity::Odd => write!(f, "time-odd"),
        }
    }
}

// ==================
// Struct definitions
// ==================

/// A structure to manage axial vectors in three dimensions.
#[derive(Builder, Clone)]
pub struct AxialVector3<T>
where
    T: ComplexFloat + Lapack,
{
    /// The $`(x, y, z)`$ components of this axial vector.
    components: Vector3<T>,

    /// The time parity of this axial vector.
    time_parity: TimeParity,

    /// The threshold for comparing determinants.
    threshold: <T as ComplexFloat>::Real,
}

impl<T> AxialVector3<T>
where
    T: ComplexFloat + Clone + Lapack,
{
    /// Returns a builder to construct a new [`AxialVector3`].
    pub fn builder() -> AxialVector3Builder<T> {
        AxialVector3Builder::default()
    }

    /// Returns a shared reference to the components of the axial vector.
    pub fn components(&self) -> &Vector3<T> {
        &self.components
    }

    /// Returns a shared reference to the time parity.
    pub fn time_parity(&self) -> &TimeParity {
        &self.time_parity
    }

    /// Returns the threshold with which axial vectors are compared.
    pub fn threshold(&self) -> <T as ComplexFloat>::Real {
        self.threshold
    }
}

// =====================
// Trait implementations
// =====================

// ----
// From
// ----
impl<T> From<AxialVector3<T>> for AxialVector3<Complex<T>>
where
    T: Float + FloatConst + Lapack,
    Complex<T>: Lapack,
{
    fn from(value: AxialVector3<T>) -> Self {
        AxialVector3::<Complex<T>>::builder()
            .components(value.components.map(Complex::from))
            .time_parity(value.time_parity.clone())
            .threshold(value.threshold)
            .build()
            .expect("Unable to construct a complex axial vector in three dimensions.")
    }
}

// ---------
// PartialEq
// ---------
impl<T> PartialEq for AxialVector3<T>
where
    T: ComplexFloat<Real = f64> + Lapack,
{
    fn eq(&self, other: &Self) -> bool {
        let thresh = (self.threshold * other.threshold).sqrt();
        let components_eq = approx::relative_eq!(
            (&self.components - &other.components)
                .map(|x| ComplexFloat::abs(x).powi(2))
                .sum()
                .sqrt(),
            0.0,
            epsilon = thresh,
            max_relative = thresh,
        );
        self.time_parity == other.time_parity && components_eq
    }
}

// --
// Eq
// --
impl<T> Eq for AxialVector3<T> where T: ComplexFloat<Real = f64> + Lapack {}

// -----
// Debug
// -----
impl<T> fmt::Debug for AxialVector3<T>
where
    T: fmt::Debug + ComplexFloat + Lapack,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "AxialVector3({})({:+.7}, {:+.7}, {:+.7})",
            self.time_parity, self.components[0], self.components[1], self.components[2],
        )
    }
}

// -------
// Display
// -------
impl<T> fmt::Display for AxialVector3<T>
where
    T: fmt::Display + ComplexFloat + Lapack,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "AxialVector3({})({:+.3}, {:+.3}, {:+.3})",
            self.time_parity, self.components[0], self.components[1], self.components[2],
        )
    }
}
