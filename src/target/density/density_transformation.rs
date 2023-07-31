use ndarray::{concatenate, s, Array2, Axis, LinalgScalar, ScalarOperand};
use ndarray_linalg::types::Lapack;
use num_complex::{Complex, ComplexFloat};

use crate::permutation::{IntoPermutation, PermutableCollection, Permutation};
use crate::symmetry::symmetry_element::SymmetryOperation;
use crate::symmetry::symmetry_transformation::{
    assemble_sh_rotation_3d_matrices, permute_array_by_atoms, ComplexConjugationTransformable,
    SpatialUnitaryTransformable, SpinUnitaryTransformable, SymmetryTransformable,
    TimeReversalTransformable, TransformationError,
};
use crate::target::density::Density;

// ---------------------------
// SpatialUnitaryTransformable
// ---------------------------
impl<'a, T> SpatialUnitaryTransformable for Density<'a, T>
where
    T: ComplexFloat + LinalgScalar + ScalarOperand + Copy + Lapack,
    f64: Into<T>,
{
    fn transform_spatial_mut(
        &mut self,
        rmat: &Array2<f64>,
        perm: Option<&Permutation<usize>>,
    ) -> &mut Self {
        let tmats: Vec<Array2<T>> = assemble_sh_rotation_3d_matrices(self.bao, rmat, perm)
            .iter()
            .map(|tmat| tmat.map(|&x| x.into()))
            .collect();
        let pbao = if let Some(p) = perm {
            self.bao.permute(p)
        } else {
            self.bao.clone()
        };
        let old_denmat = &self.density_matrix;
        let p_coeff = if let Some(p) = perm {
            permute_array_by_atoms(old_denmat, p, &[Axis(0), Axis(1)], self.bao)
        } else {
            old_denmat.clone()
        };
        let trow_p_blocks = pbao
            .shell_boundary_indices()
            .into_iter()
            .zip(tmats.iter())
            .map(|((shl_start, shl_end), tmat)| {
                tmat.dot(&p_coeff.slice(s![shl_start..shl_end, ..]))
            })
            .collect::<Vec<_>>();
        let trow_p_coeff = concatenate(
            Axis(0),
            &trow_p_blocks
                .iter()
                .map(|trow_p_block| trow_p_block.view())
                .collect::<Vec<_>>(),
        )
        .expect("Unable to concatenate the transformed rows for the various shells.");

        let tcol_trow_p_blocks = pbao
            .shell_boundary_indices()
            .into_iter()
            .zip(tmats.iter())
            .map(|((shl_start, shl_end), tmat)| {
                // tmat is real-valued, so there is no need for tmat.t().map(|x| x.conj()).
                trow_p_coeff
                    .slice(s![.., shl_start..shl_end])
                    .dot(&tmat.t())
            })
            .collect::<Vec<_>>();
        let new_denmat = concatenate(
            Axis(1),
            &tcol_trow_p_blocks
                .iter()
                .map(|tcol_trow_p_block| tcol_trow_p_block.view())
                .collect::<Vec<_>>(),
        )
        .expect("Unable to concatenate the transformed columns for the various shells.");
        self.density_matrix = new_denmat;
        self
    }
}

// ------------------------
// SpinUnitaryTransformable
// ------------------------

impl<'a, T> SpinUnitaryTransformable for Density<'a, T>
where
    T: ComplexFloat + Lapack,
{
    /// Performs a spin transformation in-place.
    ///
    /// Since densities are entirely spatial, spin transformations have no effect on them. This
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

impl<'a, T> ComplexConjugationTransformable for Density<'a, T>
where
    T: ComplexFloat + Lapack,
{
    /// Performs a complex conjugation in-place.
    fn transform_cc_mut(&mut self) -> &mut Self {
        self.density_matrix.mapv_inplace(|x| x.conj());
        self
    }
}

// ---------------------
// SymmetryTransformable
// ---------------------
impl<'a, T> SymmetryTransformable for Density<'a, T>
where
    T: ComplexFloat + Lapack,
    Density<'a, T>: SpatialUnitaryTransformable + TimeReversalTransformable,
{
    fn sym_permute_sites_spatial(
        &self,
        symop: &SymmetryOperation,
    ) -> Result<Permutation<usize>, TransformationError> {
        if (symop.generating_element.threshold().log10() - self.mol.threshold.log10()).abs() >= 3.0
        {
            log::warn!(
                "Symmetry operation threshold ({:.3e}) and molecule threshold ({:.3e}) \
                differ by more than three orders of magnitudes.",
                symop.generating_element.threshold(),
                self.mol.threshold
            )
        }
        symop
            .act_permute(&self.mol.molecule_ordinary_atoms())
            .ok_or(TransformationError(format!(
            "Unable to determine the atom permutation corresponding to the operation `{symop}`.",
        )))
    }
}
