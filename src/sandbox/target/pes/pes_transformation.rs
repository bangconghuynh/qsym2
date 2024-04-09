//! Implementation of symmetry transformations for PESes.

use anyhow::format_err;
use itertools::Itertools;
use ndarray::{s, Array2, Axis};
use ndarray_linalg::norm::Norm;
use ndarray_linalg::types::Lapack;
use num_complex::{Complex, ComplexFloat};

use crate::permutation::Permutation;
use crate::sandbox::target::pes::PES;
use crate::symmetry::symmetry_element::SymmetryOperation;
use crate::symmetry::symmetry_group::SymmetryGroupProperties;
use crate::symmetry::symmetry_transformation::{
    ComplexConjugationTransformable, DefaultTimeReversalTransformable, SpatialUnitaryTransformable,
    SpinUnitaryTransformable, SymmetryTransformable, TimeReversalTransformable,
    TransformationError,
};

// ---------------------------
// SpatialUnitaryTransformable
// ---------------------------
impl<'a, T, G> SpatialUnitaryTransformable for PES<'a, T, G>
where
    T: ComplexFloat + Lapack,
    G: SymmetryGroupProperties + Clone,
{
    fn transform_spatial_mut(
        &mut self,
        rmat: &Array2<f64>,
        _: Option<&Permutation<usize>>,
    ) -> Result<&mut Self, anyhow::Error> {
        let gi = self.group().elements().clone().into_iter().position(|g| {
            (rmat - g.get_3d_spatial_matrix()).norm_l2() < 1e-7
        }).ok_or(format_err!("Unable to find the symmetry operation in the PES specification that matches the required transformation matrix."))?;
        let transformed_values = self
            .group()
            .cayley_table()
            .ok_or(format_err!("Cayley table not found in the symmetry group associated with the PES specification."))
            .and_then(|ctb| {
                ctb
                    .slice(s![.., gi])
                    .iter()
                    .position(|&x| x == 0)
                    .ok_or(format_err!(
                        "Unable to find the inverse of group element `{gi}`."
                    )).map(|giinv| {
                        let giinv_perm = ctb.slice(s![giinv, ..]).iter().cloned().collect_vec();
                        self.values().select(Axis(0), &giinv_perm)
                    })
            })?;
        self.values = transformed_values;
        Ok(self)
    }
}

// ------------------------
// SpinUnitaryTransformable
// ------------------------

impl<'a, T, G> SpinUnitaryTransformable for PES<'a, T, G>
where
    T: ComplexFloat + Lapack,
    G: SymmetryGroupProperties + Clone,
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

impl<'a, T, G> ComplexConjugationTransformable for PES<'a, T, G>
where
    T: ComplexFloat + Lapack,
    G: SymmetryGroupProperties + Clone,
{
    fn transform_cc_mut(&mut self) -> &mut Self {
        self.values.mapv_inplace(|x| x.conj());
        self
    }
}

// --------------------------------
// DefaultTimeReversalTransformable
// --------------------------------
impl<'a, T, G> DefaultTimeReversalTransformable for PES<'a, T, G>
where
    T: ComplexFloat + Lapack,
    G: SymmetryGroupProperties + Clone,
{
}

// ---------------------
// SymmetryTransformable
// ---------------------
impl<'a, T, G> SymmetryTransformable for PES<'a, T, G>
where
    T: ComplexFloat + Lapack,
    G: SymmetryGroupProperties + Clone,
    PES<'a, T, G>: SpatialUnitaryTransformable + TimeReversalTransformable,
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
