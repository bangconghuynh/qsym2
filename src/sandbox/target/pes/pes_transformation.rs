//! Implementation of symmetry transformations for PESes.

use itertools::Itertools;
use nalgebra::Point3;
use ndarray::{Array2, Axis, ShapeBuilder};
use ndarray_linalg::types::Lapack;
use ndarray_linalg::Inverse;
use num_complex::{Complex, ComplexFloat};

use crate::permutation::Permutation;
use crate::sandbox::target::pes::PES;
use crate::symmetry::symmetry_element::SymmetryOperation;
use crate::symmetry::symmetry_transformation::{
    ComplexConjugationTransformable, DefaultTimeReversalTransformable, SpatialUnitaryTransformable,
    SpinUnitaryTransformable, SymmetryTransformable, TimeReversalTransformable,
    TransformationError,
};

// ---------------------------
// SpatialUnitaryTransformable
// ---------------------------
impl<'a, T> SpatialUnitaryTransformable for PES<'a, T>
where
    T: ComplexFloat + Lapack,
{
    fn transform_spatial_mut(
        &mut self,
        rmat: &Array2<f64>,
        _: Option<&Permutation<usize>>,
    ) -> Result<&mut Self, anyhow::Error> {
        let rmat = rmat.select(Axis(0), &[2, 0, 1]).select(Axis(1), &[2, 0, 1]);
        let rmatinv = rmat.inv()?;
        let grid_array: Array2<f64> =
            Array2::<f64>::from_iter(self.grid_points.iter().flat_map(|pt| pt.iter()))
                .reshape((3, self.grid_points.len()).f());
        let rmatinv_grid_array = rmatinv.dot(&grid_array);
        let rmatinv_grid_points = rmatinv_grid_array
            .columns()
            .map(|col| Point3::new(col[0], col[1], col[2]))
            .collect_vec();
        self.grid_points = rmatinv_grid_points;
        Ok(self)
    }
}

// ------------------------
// SpinUnitaryTransformable
// ------------------------

impl<'a, T> SpinUnitaryTransformable for PES<'a, T>
where
    T: ComplexFloat + Lapack,
{
    /// Performs a spin transformation in-place.
    ///
    /// Since PESes are entirely spatial, spin transformations have no effect on them. This
    /// thus simply returns `self` without modification.
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

impl<'a, T> ComplexConjugationTransformable for PES<'a, T>
where
    T: ComplexFloat + Lapack,
{
    fn transform_cc_mut(&mut self) -> &mut Self {
        self.function = |pt: &Point3<f64>| self.function()(pt).conj();
        self
    }
}

// --------------------------------
// DefaultTimeReversalTransformable
// --------------------------------
impl<'a, T> DefaultTimeReversalTransformable for PES<'a, T>
where
    T: ComplexFloat + Lapack,
{
}

// ---------------------
// SymmetryTransformable
// ---------------------
impl<'a, T> SymmetryTransformable for PES<'a, T>
where
    T: ComplexFloat + Lapack,
    PES<'a, T>: SpatialUnitaryTransformable + TimeReversalTransformable,
{
    /// PESes have no local sites for permutation. This method therefore simply returns the
    /// identity permutation on one object.
    fn sym_permute_sites_spatial(
        &self,
        _: &SymmetryOperation,
    ) -> Result<Permutation<usize>, TransformationError> {
        Permutation::from_image(vec![0]).map_err(|err| TransformationError(err.to_string()))
    }
}
