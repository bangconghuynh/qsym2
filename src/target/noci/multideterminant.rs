//! Multi-determinant wavefunctions for non-orthogonal configuration interaction.

use std::collections::HashSet;
use std::fmt;
use std::hash::Hash;
use std::marker::PhantomData;

use derive_builder::Builder;
use log;
use ndarray::Array1;
use ndarray_linalg::types::Lapack;
use num_complex::ComplexFloat;

use crate::angmom::spinor_rotation_3d::StructureConstraint;
use crate::target::determinant::SlaterDeterminant;

use super::basis::Basis;

#[path = "multideterminant_transformation.rs"]
pub(crate) mod multideterminant_transformation;

#[path = "multideterminant_analysis.rs"]
pub(crate) mod multideterminant_analysis;

#[cfg(test)]
#[path = "multideterminant_tests.rs"]
mod multideterminant_tests;

// ------------------
// Struct definitions
// ------------------

/// Structure to manage multi-determinantal wavefunctions.
#[derive(Builder, Clone)]
#[builder(build_fn(validate = "Self::validate"))]
pub struct MultiDeterminant<'a, T, B, SC>
where
    T: ComplexFloat + Lapack,
    SC: StructureConstraint + Hash + Eq + fmt::Display,
    B: Basis<SlaterDeterminant<'a, T, SC>> + Clone,
{
    #[builder(setter(skip), default = "PhantomData")]
    _lifetime: PhantomData<&'a ()>,

    #[builder(setter(skip), default = "PhantomData")]
    _structure_constraint: PhantomData<SC>,

    /// A boolean indicating if inner products involving this wavefunction should be the
    /// complex-symmetric bilinear form, rather than the conventional Hermitian sesquilinear form.
    #[builder(setter(skip), default = "self.complex_symmetric_from_basis()?")]
    complex_symmetric: bool,

    /// A boolean indicating if the wavefunction has been acted on by an antiunitary operation. This
    /// is so that the correct metric can be used during overlap evaluation.
    #[builder(default = "false")]
    complex_conjugated: bool,

    /// The basis of Slater determinants in which this multi-determinantal wavefunction is defined.
    basis: B,

    /// The linear combination coefficients of the elements in the multi-orbit to give this
    /// multi-determinant wavefunction.
    coefficients: Array1<T>,

    /// The energy of this multi-determinantal wavefunction.
    #[builder(
        default = "Err(\"Multi-determinantal wavefunction energy not yet set.\".to_string())"
    )]
    energy: Result<T, String>,

    /// The threshold for comparing wavefunctions.
    threshold: <T as ComplexFloat>::Real,
}

// ----------------------
// Struct implementations
// ----------------------

impl<'a, T, B, SC> MultiDeterminantBuilder<'a, T, B, SC>
where
    T: ComplexFloat + Lapack,
    SC: StructureConstraint + Hash + Eq + Clone + fmt::Display,
    B: Basis<SlaterDeterminant<'a, T, SC>> + Clone,
{
    fn validate(&self) -> Result<(), String> {
        let basis = self.basis.as_ref().ok_or("No basis found.".to_string())?;
        let coefficients = self
            .coefficients
            .as_ref()
            .ok_or("No coefficients found.".to_string())?;
        let nbasis = basis.n_items() == coefficients.len();
        if !nbasis {
            log::error!(
                "The number of coefficients does not match the number of basis determinants."
            );
        }

        let complex_symmetric = basis
            .iter()
            .map(|det_res| det_res.map(|det| det.complex_symmetric()))
            .collect::<Result<HashSet<_>, _>>()
            .map_err(|err| err.to_string())?
            .len()
            == 1;
        if !complex_symmetric {
            log::error!("Inconsistent complex-symmetric flag across basis determinants.");
        }

        let structcons_check = basis
            .iter()
            .map(|det_res| det_res.map(|det| det.structure_constraint().clone()))
            .collect::<Result<HashSet<_>, _>>()
            .map_err(|err| err.to_string())?
            .len()
            == 1;
        if !structcons_check {
            log::error!("Inconsistent spin constraints across basis determinants.");
        }

        if nbasis && structcons_check && complex_symmetric {
            Ok(())
        } else {
            Err("Multi-determinant wavefunction validation failed.".to_string())
        }
    }

    /// Retrieves the consistent complex-symmetric flag from the basis determinants.
    fn complex_symmetric_from_basis(&self) -> Result<bool, String> {
        let basis = self.basis.as_ref().ok_or("No basis found.".to_string())?;
        let complex_symmetric_set = basis
            .iter()
            .map(|det_res| det_res.map(|det| det.complex_symmetric()))
            .collect::<Result<HashSet<_>, _>>()
            .map_err(|err| err.to_string())?;
        if complex_symmetric_set.len() == 1 {
            complex_symmetric_set
                .into_iter()
                .next()
                .ok_or("Unable to retrieve the complex-symmetric flag from the basis.".to_string())
        } else {
            Err("Inconsistent complex-symmetric flag across basis determinants.".to_string())
        }
    }
}

impl<'a, T, B, SC> MultiDeterminant<'a, T, B, SC>
where
    T: ComplexFloat + Lapack,
    SC: StructureConstraint + Hash + Eq + Clone + fmt::Display,
    B: Basis<SlaterDeterminant<'a, T, SC>> + Clone,
{
    /// Returns a builder to construct a new [`MultiDeterminant`].
    pub fn builder() -> MultiDeterminantBuilder<'a, T, B, SC> {
        MultiDeterminantBuilder::default()
    }

    /// Returns the structure constraint of the multi-determinantal wavefunction.
    pub fn structure_constraint(&self) -> SC {
        self.basis
            .iter()
            .next()
            .expect("No basis determinant found.")
            .expect("No basis determinant found.")
            .structure_constraint()
            .clone()
    }
}

impl<'a, T, B, SC> MultiDeterminant<'a, T, B, SC>
where
    T: ComplexFloat + Lapack,
    SC: StructureConstraint + Hash + Eq + fmt::Display,
    B: Basis<SlaterDeterminant<'a, T, SC>> + Clone,
{
    /// Returns the complex-conjugated flag of the multi-determinantal wavefunction.
    pub fn complex_conjugated(&self) -> bool {
        self.complex_conjugated
    }

    /// Returns the basis of determinants in which this multi-determinantal wavefunction is
    /// defined.
    pub fn basis(&self) -> &B {
        &self.basis
    }

    /// Returns the coefficients of the basis determinants constituting this multi-determinantal
    /// wavefunction.
    pub fn coefficients(&self) -> &Array1<T> {
        &self.coefficients
    }

    /// Returns the energy of the multi-determinantal wavefunction.
    pub fn energy(&self) -> Result<&T, &String> {
        self.energy.as_ref()
    }

    /// Returns the threshold with which multi-determinantal wavefunctions are compared.
    pub fn threshold(&self) -> <T as ComplexFloat>::Real {
        self.threshold
    }
}

// -----
// Debug
// -----
impl<'a, T, B, SC> fmt::Debug for MultiDeterminant<'a, T, B, SC>
where
    T: ComplexFloat + Lapack,
    SC: StructureConstraint + Hash + Eq + fmt::Display,
    B: Basis<SlaterDeterminant<'a, T, SC>> + Clone,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "MultiDeterminant over {} basis Slater determinants",
            self.coefficients.len(),
        )?;
        Ok(())
    }
}

// -------
// Display
// -------
impl<'a, T, B, SC> fmt::Display for MultiDeterminant<'a, T, B, SC>
where
    T: ComplexFloat + Lapack,
    SC: StructureConstraint + Hash + Eq + fmt::Display,
    B: Basis<SlaterDeterminant<'a, T, SC>> + Clone,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "MultiDeterminant over {} basis Slater determinants",
            self.coefficients.len(),
        )?;
        Ok(())
    }
}
