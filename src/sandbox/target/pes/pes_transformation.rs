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
use crate::symmetry::symmetry_element::symmetry_operation::SpecialSymmetryTransformation;
use crate::symmetry::symmetry_group::SymmetryGroupProperties;
use crate::symmetry::symmetry_transformation::{
    ComplexConjugationTransformable, DefaultTimeReversalTransformable, SpatialUnitaryTransformable,
    SpinUnitaryTransformable, SymmetryTransformable, TimeReversalTransformable,
    TransformationError,
};

// ---------------------------
// SpatialUnitaryTransformable
// ---------------------------
impl<'a, G, T> SpatialUnitaryTransformable for PES<'a, G, T>
where
    T: ComplexFloat + Lapack,
    G: SymmetryGroupProperties + Clone,
{
    /// Note that this is potentially slow since `rmat` instead of the actual group element is given.
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

impl<'a, G, T> SpinUnitaryTransformable for PES<'a, G, T>
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

impl<'a, G, T> ComplexConjugationTransformable for PES<'a, G, T>
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
impl<'a, G, T> DefaultTimeReversalTransformable for PES<'a, G, T>
where
    T: ComplexFloat + Lapack,
    G: SymmetryGroupProperties + Clone,
{
}

// ---------------------
// SymmetryTransformable
// ---------------------
impl<'a, G, T> SymmetryTransformable for PES<'a, G, T>
where
    T: ComplexFloat + Lapack,
    G: SymmetryGroupProperties + Clone,
    PES<'a, G, T>: SpatialUnitaryTransformable + TimeReversalTransformable,
{
    /// PESes have no local sites for permutation. This method therefore simply returns the
    /// identity permutation on one object.
    fn sym_permute_sites_spatial(
        &self,
        _: &SymmetryOperation,
    ) -> Result<Permutation<usize>, TransformationError> {
        Permutation::from_image(vec![0]).map_err(|err| TransformationError(err.to_string()))
    }

    /// Performs a spatial transformation according to a specified symmetry operation in-place.
    ///
    /// Note that both $`\mathsf{SO}(3)`$ and $`\mathsf{SU}(2)`$ rotations effect the same spatial
    /// transformation. Also note that, if the transformation contains time reversal, it will be
    /// accompanied by a complex conjugation.
    ///
    /// This is a specialised implementation for [`PES`] since the default implementation of this
    /// method for the [`SymmetryTransformable`] trait makes use of [`Self::transform_spatial_mut`]
    /// which is inefficient for [`PES`].
    ///
    /// # Arguments
    ///
    /// * `symop` - A symmetry operation.
    fn sym_transform_spatial_mut(
        &mut self,
        symop: &SymmetryOperation,
    ) -> Result<&mut Self, TransformationError> {
        let gi = self
            .group()
            .get_index_of(symop)
            .ok_or(TransformationError("Unable to find the symmetry operation in the PES specification that matches the required transformation matrix.".to_string()))?;
        let transformed_values = self
            .group()
            .cayley_table()
            .ok_or(TransformationError("Cayley table not found in the symmetry group associated with the PES specification.".to_string()))
            .and_then(|ctb| {
                ctb
                    .slice(s![.., gi])
                    .iter()
                    .position(|&x| x == 0)
                    .ok_or(TransformationError(
                        format!("Unable to find the inverse of group element `{gi}`.")
                    )).map(|giinv| {
                        let giinv_perm = ctb.slice(s![giinv, ..]).iter().cloned().collect_vec();
                        self.values().select(Axis(0), &giinv_perm)
                    })
            })?;
        self.values = transformed_values;
        if symop.contains_time_reversal() {
            self.transform_cc_mut();
        }
        Ok(self)
    }

    /// Performs a coupled spin-spatial transformation according to a specified symmetry operation
    /// in-place.
    ///
    /// Note that only $`\mathsf{SU}(2)`$ rotations can effect spin transformations.
    ///
    /// This is a specialised implementation for [`PES`] since the default implementation of this
    /// method for the [`SymmetryTransformable`] trait makes use of [`Self::transform_spatial_mut`]
    /// which is inefficient for [`PES`].
    ///
    /// # Arguments
    ///
    /// * `symop` - A symmetry operation.
    fn sym_transform_spin_spatial_mut(
        &mut self,
        symop: &SymmetryOperation,
    ) -> Result<&mut Self, TransformationError> {
        // PESes are entirely spatial, so spin transformations have no effects.
        self.sym_transform_spatial_mut(symop)
    }
}
