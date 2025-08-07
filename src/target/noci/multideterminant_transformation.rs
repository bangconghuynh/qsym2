//! Implementation of symmetry transformations for multi-determinantal wavefunctions.

use std::fmt;
use std::hash::Hash;

use ndarray::Array2;
use ndarray_linalg::types::Lapack;
use num_complex::{Complex, ComplexFloat};

use crate::angmom::spinor_rotation_3d::{SpinOrbitCoupled, StructureConstraint};
use crate::group::GroupProperties;
use crate::permutation::Permutation;
use crate::symmetry::symmetry_element::{SpecialSymmetryTransformation, SymmetryOperation};
use crate::symmetry::symmetry_transformation::{
    ComplexConjugationTransformable, DefaultTimeReversalTransformable, SpatialUnitaryTransformable,
    SpinUnitaryTransformable, SymmetryTransformable, TimeReversalTransformable,
    TransformationError,
};
use crate::target::determinant::SlaterDeterminant;
use crate::target::noci::basis::{Basis, EagerBasis, OrbitBasis};
use crate::target::noci::multideterminant::MultiDeterminant;

// ---------------------------
// SpatialUnitaryTransformable
// ---------------------------

impl<'a, T, B, SC> SpatialUnitaryTransformable for MultiDeterminant<'a, T, B, SC>
where
    T: ComplexFloat + Lapack,
    SC: StructureConstraint + Hash + Eq + Clone + fmt::Display,
    B: Basis<SlaterDeterminant<'a, T, SC>> + SpatialUnitaryTransformable + Clone,
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

impl<'a, T, B, SC> SpinUnitaryTransformable for MultiDeterminant<'a, T, B, SC>
where
    T: ComplexFloat + Lapack,
    SC: StructureConstraint + Hash + Eq + Clone + fmt::Display,
    B: Basis<SlaterDeterminant<'a, T, SC>> + SpinUnitaryTransformable + Clone,
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

impl<'a, T, B, SC> ComplexConjugationTransformable for MultiDeterminant<'a, T, B, SC>
where
    T: ComplexFloat + Lapack,
    SC: StructureConstraint + Hash + Eq + Clone + fmt::Display,
    B: Basis<SlaterDeterminant<'a, T, SC>> + ComplexConjugationTransformable + Clone,
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

impl<'a, T, B, SC> DefaultTimeReversalTransformable for MultiDeterminant<'a, T, B, SC>
where
    T: ComplexFloat + Lapack,
    SC: StructureConstraint + Hash + Eq + Clone + fmt::Display,
    B: Basis<SlaterDeterminant<'a, T, SC>> + DefaultTimeReversalTransformable + Clone,
{
}

// ---------------------------------------
// TimeReversalTransformable (non-default)
// ---------------------------------------
// `SlaterDeterminant<_, _, SpinOrbitCoupled>` does not implement
// `DefaultTimeReversalTransformable` and so the surrounding `OrbitBasis` does not get a blanket
// implementation of `TimeReversalTransformable`, and neither does the surrounding
// `MultiDeterminant`.
impl<'a, 'g, G> TimeReversalTransformable
    for MultiDeterminant<
        'a,
        Complex<f64>,
        OrbitBasis<'g, G, SlaterDeterminant<'a, Complex<f64>, SpinOrbitCoupled>>,
        SpinOrbitCoupled,
    >
where
    G: GroupProperties<GroupElement = SymmetryOperation> + Clone,
{
    fn transform_timerev_mut(&mut self) -> Result<&mut Self, TransformationError> {
        self.basis.transform_timerev_mut()?;
        self.coefficients.mapv_inplace(|v| v.conj());
        self.complex_conjugated = !self.complex_conjugated;
        Ok(self)
    }
}

// `SlaterDeterminant<_, _, SpinOrbitCoupled>` does not implement
// `DefaultTimeReversalTransformable` and so the surrounding `EagerBasis` does not get a blanket
// implementation of `TimeReversalTransformable`, and neither does the surrounding
// `MultiDeterminant`.
impl<'a> TimeReversalTransformable
    for MultiDeterminant<
        'a,
        Complex<f64>,
        EagerBasis<SlaterDeterminant<'a, Complex<f64>, SpinOrbitCoupled>>,
        SpinOrbitCoupled,
    >
{
    fn transform_timerev_mut(&mut self) -> Result<&mut Self, TransformationError> {
        self.basis.transform_timerev_mut()?;
        self.coefficients.mapv_inplace(|v| v.conj());
        self.complex_conjugated = !self.complex_conjugated;
        Ok(self)
    }
}

// ---------------------
// SymmetryTransformable
// ---------------------

impl<'a, 'go, G, T, SC> SymmetryTransformable
    for MultiDeterminant<'a, T, OrbitBasis<'go, G, SlaterDeterminant<'a, T, SC>>, SC>
where
    T: ComplexFloat + Lapack,
    G: GroupProperties<GroupElement = SymmetryOperation> + Clone,
    SC: StructureConstraint + Hash + Eq + Clone + fmt::Display,
    SlaterDeterminant<'a, T, SC>: SymmetryTransformable,
    Self: TimeReversalTransformable,
    OrbitBasis<'go, G, SlaterDeterminant<'a, T, SC>>: TimeReversalTransformable,
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

impl<'a, T, SC> SymmetryTransformable
    for MultiDeterminant<'a, T, EagerBasis<SlaterDeterminant<'a, T, SC>>, SC>
where
    T: ComplexFloat + Lapack,
    SC: StructureConstraint + Hash + Eq + Clone + fmt::Display,
    SlaterDeterminant<'a, T, SC>: SymmetryTransformable,
    Self: TimeReversalTransformable,
    EagerBasis<SlaterDeterminant<'a, T, SC>>: TimeReversalTransformable,
{
    fn sym_permute_sites_spatial(
        &self,
        symop: &SymmetryOperation,
    ) -> Result<Permutation<usize>, TransformationError> {
        self.basis.sym_permute_sites_spatial(symop)
    }
}
