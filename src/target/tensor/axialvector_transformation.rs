//! Implementation of symmetry transformation for axial vectors.

use nalgebra::Vector3;
use ndarray::{Axis, Array1, Array2, LinalgScalar, ScalarOperand};
use ndarray_linalg::solve::Determinant;
use ndarray_linalg::types::Lapack;
use num_complex::{Complex, ComplexFloat};

use crate::permutation::Permutation;
use crate::symmetry::symmetry_element::SymmetryOperation;
use crate::symmetry::symmetry_transformation::{
    ComplexConjugationTransformable, SpatialUnitaryTransformable, SpinUnitaryTransformable,
    SymmetryTransformable, TimeReversalTransformable, TransformationError,
};
use crate::target::tensor::axialvector::{AxialVector3, TimeParity};

// ---------------------------
// SpatialUnitaryTransformable
// ---------------------------
impl<T> SpatialUnitaryTransformable for AxialVector3<T>
where
    T: ComplexFloat + LinalgScalar + ScalarOperand + Copy + Lapack,
    f64: Into<T>,
{
    fn transform_spatial_mut(
        &mut self,
        rmat: &Array2<f64>,
        _perm: Option<&Permutation<usize>>,
    ) -> &mut Self {
        // rmat is in (y, z, x) order. We must first reorder into (x, y, z) order.
        let rmat_xyz = rmat
            .select(Axis(0), &[2, 0, 1])
            .select(Axis(1), &[2, 0, 1]);
        let det = rmat_xyz
            .det()
            .expect("Unable to obtain the determinant of the transformation matrix.");
        let old_components = Array1::from_iter(self.components.iter().cloned());
        let new_components = rmat_xyz.mapv(|x| (det * x).into()).dot(&old_components);
        self.components = Vector3::from_iterator(new_components.into_iter());
        self
    }
}

// ------------------------
// SpinUnitaryTransformable
// ------------------------

impl<T> SpinUnitaryTransformable for AxialVector3<T>
where
    T: ComplexFloat + Lapack,
{
    /// Axial vectors are spatial quantities, therefore spin transformations have no effects on
    /// them.
    fn transform_spin_mut(
        &mut self,
        _dmat: &Array2<Complex<f64>>,
    ) -> Result<&mut Self, TransformationError> {
        Ok(self)
    }
}

// -------------------------------
// ComplexConjugationTransformable
// -------------------------------

impl<T> ComplexConjugationTransformable for AxialVector3<T>
where
    T: ComplexFloat + Lapack,
{
    fn transform_cc_mut(&mut self) -> &mut Self {
        self.components = self.components.map(|x| x.conj());
        self
    }
}

// -------------------------
// TimeReversalTransformable
// -------------------------

impl<T> TimeReversalTransformable for AxialVector3<T>
where
    T: ComplexFloat + Lapack,
{
    /// Provides a custom implementation of time reversal where the axial vector is kept invariant
    /// or inverted based on its time-parity.
    fn transform_timerev_mut(&mut self) -> Result<&mut Self, TransformationError> {
        match self.time_parity {
            TimeParity::Even => {}
            TimeParity::Odd => self.components = -self.components,
        }
        Ok(self)
    }
}

// ---------------------
// SymmetryTransformable
// ---------------------
impl<T> SymmetryTransformable for AxialVector3<T>
where
    T: ComplexFloat + Lapack,
    AxialVector3<T>: SpatialUnitaryTransformable + TimeReversalTransformable,
{
    fn sym_permute_sites_spatial(
        &self,
        _symop: &SymmetryOperation,
    ) -> Result<Permutation<usize>, TransformationError> {
        Ok(Permutation::from_image(vec![0]))
    }
}
