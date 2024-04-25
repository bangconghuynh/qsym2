//! Implementation of symmetry transformations for multi-determinantal wavefunctions.

use ndarray::Array2;
use ndarray_linalg::types::Lapack;
use num_complex::{Complex, ComplexFloat};

use crate::permutation::Permutation;
use crate::symmetry::symmetry_element::SymmetryOperation;
use crate::symmetry::symmetry_transformation::{
    ComplexConjugationTransformable, DefaultTimeReversalTransformable, SpatialUnitaryTransformable,
    SpinUnitaryTransformable, SymmetryTransformable, TimeReversalTransformable,
    TransformationError,
};
use crate::target::determinant::SlaterDeterminant;
use crate::target::noci::basis::Basis;
use crate::target::noci::multideterminant::MultiDeterminant;

// ---------------------------
// SpatialUnitaryTransformable
// ---------------------------

impl<'a, T, B> SpatialUnitaryTransformable for MultiDeterminant<'a, T, B>
where
    T: ComplexFloat + Lapack,
    B: Basis<SlaterDeterminant<'a, T>> + SpatialUnitaryTransformable + Clone,
{
    fn transform_spatial_mut(
        &mut self,
        rmat: &Array2<f64>,
        perm: Option<&Permutation<usize>>,
    ) -> Result<&mut Self, anyhow::Error> {
        self.basis.transform_spatial_mut(rmat, perm)?;
        Ok(self)
    }
}

// ------------------------
// SpinUnitaryTransformable
// ------------------------

impl<'a, T, B> SpinUnitaryTransformable for MultiDeterminant<'a, T, B>
where
    T: ComplexFloat + Lapack,
    B: Basis<SlaterDeterminant<'a, T>> + SpinUnitaryTransformable + Clone,
{
    fn transform_spin_mut(
        &mut self,
        dmat: &Array2<Complex<f64>>,
    ) -> Result<&mut Self, TransformationError> {
        self.basis.transform_spin_mut(dmat)?;
        Ok(self)
    }
}

// -------------------------------
// ComplexConjugationTransformable
// -------------------------------

impl<'a, T, B> ComplexConjugationTransformable for MultiDeterminant<'a, T, B>
where
    T: ComplexFloat + Lapack,
    B: Basis<SlaterDeterminant<'a, T>> + ComplexConjugationTransformable + Clone,
{
    /// Performs a complex conjugation in-place.
    fn transform_cc_mut(&mut self) -> &mut Self {
        self.basis.transform_cc_mut();
        self.coefficients.iter_mut().for_each(|v| *v = v.conj());
        self.complex_conjugated = !self.complex_conjugated;
        self
    }
}

// --------------------------------
// DefaultTimeReversalTransformable
// --------------------------------

impl<'a, T, B> DefaultTimeReversalTransformable for MultiDeterminant<'a, T, B>
where
    T: ComplexFloat + Lapack,
    B: Basis<SlaterDeterminant<'a, T>> + DefaultTimeReversalTransformable + Clone,
{
}

// ---------------------
// SymmetryTransformable
// ---------------------

impl<'a, T, B> SymmetryTransformable for MultiDeterminant<'a, T, B>
where
    T: ComplexFloat + Lapack,
    B: Basis<SlaterDeterminant<'a, T>> + SymmetryTransformable + Clone,
    MultiDeterminant<'a, T, B>: TimeReversalTransformable,
{
    fn sym_permute_sites_spatial(
        &self,
        symop: &SymmetryOperation,
    ) -> Result<Permutation<usize>, TransformationError> {
        self.basis.sym_permute_sites_spatial(symop)
    }
}
