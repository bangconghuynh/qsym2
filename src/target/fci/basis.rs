//! Basis for full configuration interaction of Slater determinants.

use anyhow::format_err;
use derive_builder::Builder;
use ndarray::Array1;
use ndarray_linalg::types::Lapack;
use num_complex::ComplexFloat;
use std::fmt;

use crate::{
    angmom::spinor_rotation_3d::StructureConstraint,
    target::{determinant::SlaterDeterminant, noci::basis::Basis},
};

#[path = "basis_transformation.rs"]
mod basis_transformation;

// =====
// Basis
// =====

// --------------------------------------
// Struct definitions and implementations
// --------------------------------------

// ~~~~~~~~~
// FCI basis
// ~~~~~~~~~

#[derive(Builder, Clone)]
pub struct FCIBasis<'a, T, SC>
where
    T: ComplexFloat + Lapack,
    SC: StructureConstraint + fmt::Display,
{
    /// The reference determinant of the full-configuration-interaction basis. This determinant
    /// carries both occupied and virtual molecular orbitals.
    reference: SlaterDeterminant<'a, T, SC>,

    /// Vector of full-configuration-interaction occupation patterns.
    occupation_patterns: Vec<Vec<Array1<<T as ComplexFloat>::Real>>>,
}

impl<'a, T, SC> FCIBasis<'a, T, SC>
where
    T: ComplexFloat + Lapack,
    SC: StructureConstraint + fmt::Display + Clone,
{
    pub fn builder() -> FCIBasisBuilder<'a, T, SC> {
        FCIBasisBuilder::<T, SC>::default()
    }

    /// The reference determinant of the full-configuration-interaction basis.
    pub fn reference(&self) -> &SlaterDeterminant<'a, T, SC> {
        &self.reference
    }
}

impl<'a, T, SC> Basis<SlaterDeterminant<'a, T, SC>> for FCIBasis<'a, T, SC>
where
    T: ComplexFloat + Lapack,
    SC: 'a + StructureConstraint + fmt::Display + Clone,
{
    type BasisIter<'b>
        = FCIBasisIterator<'b, 'a, T, SC>
    where
        Self: 'b;

    fn n_items(&self) -> usize {
        self.occupation_patterns.len()
    }

    /// Iterates over the elements of the [`FCIBasis`]. Each element is a Slater determinant
    /// obtained by replacing the occupation pattern in the reference with the corresponding one
    /// from the [`Self::occupation_patterns`] list.
    fn iter(&self) -> Self::BasisIter<'_> {
        FCIBasisIterator::new(&self)
    }

    fn first(&self) -> Option<SlaterDeterminant<'a, T, SC>> {
        Some(self.reference.clone())
    }
}

// ~~~~~~~~~~~~~~~~~~
// FCIBasis interator
// ~~~~~~~~~~~~~~~~~~

/// Lazy iterator for full-configuration-interaction basis.
#[derive(Builder, Clone)]
pub struct FCIBasisIterator<'b, 'a, T, SC>
where
    T: ComplexFloat + Lapack,
    SC: StructureConstraint + fmt::Display,
{
    /// The associated full-configuration-interaction basis.
    fci_basis: &'b FCIBasis<'a, T, SC>,

    /// The current index of the occupation patterns.
    current_occupation_index: usize,
}

impl<'b, 'a, T, SC> FCIBasisIterator<'b, 'a, T, SC>
where
    T: ComplexFloat + Lapack,
    SC: StructureConstraint + fmt::Display,
{
    /// Creates and returns a new full-configuration-interaction basis iterator.
    ///
    /// # Arguments
    ///
    /// * `fci_basis` - The associated full-configuration-interaction basis.
    ///
    /// # Returns
    ///
    /// An full-configuration-interaction basis iterator.
    fn new(fci_basis: &'b FCIBasis<'a, T, SC>) -> Self {
        Self {
            fci_basis,
            current_occupation_index: 0,
        }
    }
}

impl<'b, 'a, T, SC> Iterator for FCIBasisIterator<'b, 'a, T, SC>
where
    T: ComplexFloat + Lapack,
    SC: StructureConstraint + fmt::Display + Clone,
{
    type Item = Result<SlaterDeterminant<'a, T, SC>, anyhow::Error>;

    fn next(&mut self) -> Option<Self::Item> {
        let reference = &self.fci_basis.reference;
        let index = self.current_occupation_index;
        self.current_occupation_index += 1;
        if index > self.fci_basis.occupation_patterns.len() {
            None
        } else {
            let det = SlaterDeterminant::builder()
                .structure_constraint(reference.structure_constraint().clone())
                .baos(reference.baos().clone())
                .complex_symmetric(reference.complex_symmetric())
                .complex_conjugated(reference.complex_conjugated())
                .mol(reference.mol())
                .coefficients(reference.coefficients())
                .occupations(&self.fci_basis.occupation_patterns[index])
                .mo_energies(reference.mo_energies().cloned())
                .threshold(reference.threshold())
                .build()
                .map_err(|err| format_err!(err));
            Some(det)
        }
    }
}
