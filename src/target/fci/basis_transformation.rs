//! Implementation of symmetry transformations for bases for full-configuration-interaction of
//! Slater determinants.

use std::fmt;

use ndarray::Array2;
use ndarray_linalg::types::Lapack;
use num::Complex;
use num_complex::ComplexFloat;

use crate::angmom::spinor_rotation_3d::{SpinOrbitCoupled, StructureConstraint};
use crate::permutation::Permutation;
use crate::symmetry::symmetry_element::SymmetryOperation;
use crate::symmetry::symmetry_transformation::{
    ComplexConjugationTransformable, DefaultTimeReversalTransformable, SpatialUnitaryTransformable,
    SpinUnitaryTransformable, SymmetryTransformable, TimeReversalTransformable,
    TransformationError,
};
use crate::target::determinant::SlaterDeterminant;
use crate::target::fci::basis::FCIBasis;

// ~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~
// Lazy basis from orbits
// ~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~

// ---------------------------
// SpatialUnitaryTransformable
// ---------------------------
impl<'a, T, SC> SpatialUnitaryTransformable for FCIBasis<'a, T, SC>
where
    T: ComplexFloat + Lapack,
    SC: StructureConstraint + fmt::Display + Clone,
    SlaterDeterminant<'a, T, SC>: SpatialUnitaryTransformable,
{
    fn transform_spatial_mut(
        &mut self,
        rmat: &Array2<f64>,
        perm: Option<&Permutation<usize>>,
    ) -> Result<&mut Self, TransformationError> {
        self.reference.transform_spatial_mut(rmat, perm)?;
        Ok(self)
    }
}

// ------------------------
// SpinUnitaryTransformable
// ------------------------
impl<'a, T, SC> SpinUnitaryTransformable for FCIBasis<'a, T, SC>
where
    T: ComplexFloat + Lapack,
    SC: StructureConstraint + fmt::Display + Clone,
    SlaterDeterminant<'a, T, SC>: SpinUnitaryTransformable,
{
    fn transform_spin_mut(
        &mut self,
        rmat: &Array2<Complex<f64>>,
    ) -> Result<&mut Self, TransformationError> {
        self.reference.transform_spin_mut(rmat)?;
        Ok(self)
    }
}

// -------------------------------
// ComplexConjugationTransformable
// -------------------------------

impl<'a, T, SC> ComplexConjugationTransformable for FCIBasis<'a, T, SC>
where
    T: ComplexFloat + Lapack,
    SC: StructureConstraint + fmt::Display + Clone,
    SlaterDeterminant<'a, T, SC>: ComplexConjugationTransformable,
{
    /// Performs a complex conjugation in-place.
    fn transform_cc_mut(&mut self) -> Result<&mut Self, TransformationError> {
        self.reference.transform_cc_mut()?;
        Ok(self)
    }
}

// --------------------------------
// DefaultTimeReversalTransformable
// --------------------------------
impl<'a, T, SC> DefaultTimeReversalTransformable for FCIBasis<'a, T, SC>
where
    T: ComplexFloat + Lapack,
    SC: StructureConstraint + fmt::Display + Clone,
    SlaterDeterminant<'a, T, SC>: DefaultTimeReversalTransformable,
{
}

// ---------------------------------------
// TimeReversalTransformable (non-default)
// ---------------------------------------
// `SlaterDeterminant<_, _, SpinOrbitCoupled>` does not implement `DefaultTimeReversalTransformable`
// and so the surrounding `FCIBasis` does not get a blanket implementation of
// `TimeReversalTransformable`.
impl<'a> TimeReversalTransformable for FCIBasis<'a, Complex<f64>, SpinOrbitCoupled> {
    fn transform_timerev_mut(&mut self) -> Result<&mut Self, TransformationError> {
        self.reference.transform_timerev_mut()?;
        Ok(self)
    }
}

// ---------------------
// SymmetryTransformable
// ---------------------
impl<'a, T, SC> SymmetryTransformable for FCIBasis<'a, T, SC>
where
    T: ComplexFloat + Lapack,
    SC: StructureConstraint + fmt::Display + Clone,
    SlaterDeterminant<'a, T, SC>: SymmetryTransformable,
    FCIBasis<'a, T, SC>: TimeReversalTransformable,
{
    fn sym_permute_sites_spatial(
        &self,
        symop: &SymmetryOperation,
    ) -> Result<Permutation<usize>, TransformationError> {
        self.reference.sym_permute_sites_spatial(symop)
    }
}
