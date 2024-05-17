//! Implementation of symmetry transformations for real-space functions.

use itertools::Itertools;
use nalgebra::Point3;
use ndarray::{Array2, Axis, ShapeBuilder};
use ndarray_linalg::types::Lapack;
use num_complex::{Complex, ComplexFloat};

use crate::permutation::Permutation;
use crate::sandbox::target::real_space_function::RealSpaceFunction;
use crate::symmetry::symmetry_element::SymmetryOperation;
use crate::symmetry::symmetry_transformation::{
    ComplexConjugationTransformable, DefaultTimeReversalTransformable, SpatialUnitaryTransformable,
    SpinUnitaryTransformable, SymmetryTransformable, TimeReversalTransformable,
    TransformationError,
};

// ---------------------------
// SpatialUnitaryTransformable
// ---------------------------
impl<T, F> SpatialUnitaryTransformable for RealSpaceFunction<T, F>
where
    T: ComplexFloat + Lapack,
    F: Clone + Fn(&Point3<f64>) -> T,
{
    fn transform_spatial_mut(
        &mut self,
        rmat: &Array2<f64>,
        _: Option<&Permutation<usize>>,
    ) -> Result<&mut Self, TransformationError> {
        let rmat = rmat.select(Axis(0), &[2, 0, 1]).select(Axis(1), &[2, 0, 1]);
        let rmatinv = rmat.t();
        let grid_array = Array2::from_shape_vec(
            (3, self.grid_points.len()).f(),
            self.grid_points
                .iter()
                .flat_map(|pt| pt.iter().cloned())
                .collect_vec(),
        )
        .map_err(|err| TransformationError(err.to_string()))?;
        let rmatinv_grid_array = rmatinv.dot(&grid_array);
        let rmatinv_grid_points = rmatinv_grid_array
            .columns()
            .into_iter()
            .map(|col| Point3::new(col[0], col[1], col[2]))
            .collect_vec();
        self.grid_points = rmatinv_grid_points;
        Ok(self)
    }
}

// ------------------------
// SpinUnitaryTransformable
// ------------------------

impl<T, F> SpinUnitaryTransformable for RealSpaceFunction<T, F>
where
    T: ComplexFloat + Lapack,
    F: Clone + Fn(&Point3<f64>) -> T,
{
    /// Performs a spin transformation in-place.
    ///
    /// Since real-space functions are entirely spatial, spin transformations have no effect on
    /// them. This thus simply returns `self` without modification.
    fn transform_spin_mut(
        &mut self,
        _: &Array2<Complex<f64>>,
    ) -> Result<&mut Self, TransformationError> {
        Ok(self)
    }
}

// -------------------------------
// ComplexConjugationTransformable
// -------------------------------

impl<T, F> ComplexConjugationTransformable for RealSpaceFunction<T, F>
where
    T: ComplexFloat + Lapack,
    F: Clone + Fn(&Point3<f64>) -> T,
{
    fn transform_cc_mut(&mut self) -> Result<&mut Self, TransformationError> {
        self.complex_conjugated = !self.complex_conjugated;
        Ok(self)
    }
}

// --------------------------------
// DefaultTimeReversalTransformable
// --------------------------------
impl<T, F> DefaultTimeReversalTransformable for RealSpaceFunction<T, F>
where
    T: ComplexFloat + Lapack,
    F: Clone + Fn(&Point3<f64>) -> T,
{
}

// ---------------------
// SymmetryTransformable
// ---------------------
impl<T, F> SymmetryTransformable for RealSpaceFunction<T, F>
where
    T: ComplexFloat + Lapack,
    F: Clone + Fn(&Point3<f64>) -> T,
    RealSpaceFunction<T, F>: SpatialUnitaryTransformable + TimeReversalTransformable,
{
    /// Real-space functions have no local sites for permutation. This method therefore simply
    /// returns the identity permutation on one object.
    fn sym_permute_sites_spatial(
        &self,
        _: &SymmetryOperation,
    ) -> Result<Permutation<usize>, TransformationError> {
        Permutation::from_image(vec![0]).map_err(|err| TransformationError(err.to_string()))
    }
}
