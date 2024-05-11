//! Implementation of symmetry transformations for multi-determinantal wavefunctions.

use ndarray::Array2;
use ndarray_linalg::types::Lapack;
use num_complex::{Complex, ComplexFloat};

use crate::group::GroupProperties;
use crate::permutation::Permutation;
use crate::symmetry::symmetry_element::{SpecialSymmetryTransformation, SymmetryOperation};
use crate::symmetry::symmetry_transformation::{
    ComplexConjugationTransformable, DefaultTimeReversalTransformable, SpatialUnitaryTransformable,
    SpinUnitaryTransformable, SymmetryTransformable, TransformationError,
};
use crate::target::determinant::SlaterDeterminant;
use crate::target::noci::basis::{Basis, EagerBasis, OrbitBasis};
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
    ) -> Result<&mut Self, TransformationError> {
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
    fn transform_cc_mut(&mut self) -> Result<&mut Self, TransformationError> {
        self.basis.transform_cc_mut()?;
        self.coefficients.iter_mut().for_each(|v| *v = v.conj());
        self.complex_conjugated = !self.complex_conjugated;
        Ok(self)
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

impl<'a, 'go, G, T> SymmetryTransformable
    for MultiDeterminant<'a, T, OrbitBasis<'go, G, SlaterDeterminant<'a, T>>>
where
    T: ComplexFloat + Lapack,
    G: GroupProperties<GroupElement = SymmetryOperation> + Clone,
    SlaterDeterminant<'a, T>: SymmetryTransformable,
{
    fn sym_permute_sites_spatial(
        &self,
        symop: &SymmetryOperation,
    ) -> Result<Permutation<usize>, TransformationError> {
        self.basis.sym_permute_sites_spatial(symop)
    }

    // --------------------------
    // Overriden provided methods
    // --------------------------
    fn sym_transform_spatial_mut(
        &mut self,
        symop: &SymmetryOperation,
    ) -> Result<&mut Self, TransformationError> {
        self.basis.sym_transform_spatial_mut(symop)?;
        if symop.contains_time_reversal() {
            self.coefficients.iter_mut().for_each(|v| *v = v.conj());
            self.complex_conjugated = !self.complex_conjugated;
        }
        Ok(self)
    }

    fn sym_transform_spatial_with_spintimerev_mut(
        &mut self,
        symop: &SymmetryOperation,
    ) -> Result<&mut Self, TransformationError> {
        self.basis
            .sym_transform_spatial_with_spintimerev_mut(symop)?;
        if symop.contains_time_reversal() {
            self.coefficients.iter_mut().for_each(|v| *v = v.conj());
            self.complex_conjugated = !self.complex_conjugated;
        }
        Ok(self)
    }

    fn sym_transform_spin_mut(
        &mut self,
        symop: &SymmetryOperation,
    ) -> Result<&mut Self, TransformationError> {
        self.basis.sym_transform_spin_mut(symop)?;
        if symop.contains_time_reversal() {
            self.coefficients.iter_mut().for_each(|v| *v = v.conj());
            self.complex_conjugated = !self.complex_conjugated;
        }
        Ok(self)
    }

    fn sym_transform_spin_spatial_mut(
        &mut self,
        symop: &SymmetryOperation,
    ) -> Result<&mut Self, TransformationError> {
        self.basis.sym_transform_spin_spatial_mut(symop)?;
        if symop.contains_time_reversal() {
            self.coefficients.iter_mut().for_each(|v| *v = v.conj());
            self.complex_conjugated = !self.complex_conjugated;
        }
        Ok(self)
    }
}

impl<'a, T> SymmetryTransformable for MultiDeterminant<'a, T, EagerBasis<SlaterDeterminant<'a, T>>>
where
    T: ComplexFloat + Lapack,
    SlaterDeterminant<'a, T>: SymmetryTransformable,
{
    fn sym_permute_sites_spatial(
        &self,
        symop: &SymmetryOperation,
    ) -> Result<Permutation<usize>, TransformationError> {
        self.basis.sym_permute_sites_spatial(symop)
    }
}
