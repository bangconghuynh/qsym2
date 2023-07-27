use std::ops::Mul;

use approx;
use ndarray::{concatenate, s, Array2, Axis, LinalgScalar, ScalarOperand};
use ndarray_linalg::types::Lapack;
use num_complex::{Complex, ComplexFloat};

use crate::angmom::spinor_rotation_3d::SpinConstraint;
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
        let new_denmat = match self.spin_constraint {
            SpinConstraint::Restricted(_) | SpinConstraint::Unrestricted(_, _) => {
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
                concatenate(
                    Axis(1),
                    &tcol_trow_p_blocks
                        .iter()
                        .map(|tcol_trow_p_block| tcol_trow_p_block.view())
                        .collect::<Vec<_>>(),
                )
                .expect("Unable to concatenate the transformed columns for the various shells.")
            }
            SpinConstraint::Generalised(nspins, _) => {
                let nspatial = self.bao.n_funcs();
                let jspin_blocks = (0..nspins)
                    .map(|ispin| {
                        let spin_start_row = usize::from(ispin) * nspatial;
                        let spin_end_row = (usize::from(ispin) + 1) * nspatial;
                        let ispin_blocks = (0..nspins).map(|jspin| {
                            // Extract spin block (ispin, jspin).
                            let spin_start_col = usize::from(jspin) * nspatial;
                            let spin_end_col = (usize::from(jspin) + 1) * nspatial;
                            let spin_block = old_denmat.slice(
                                s![spin_start_row..spin_end_row, spin_start_col..spin_end_col]
                            ).to_owned();

                            // Permute within spin block (ispin, jspin).
                            let p_spin_block = if let Some(p) = perm {
                                permute_array_by_atoms(&spin_block, p, &[Axis(0), Axis(1)], self.bao)
                            } else {
                                spin_block
                            };

                            // Transform rows within spin block (ispin, jspin).
                            let trow_p_blocks = pbao
                                .shell_boundary_indices()
                                .into_iter()
                                .zip(tmats.iter())
                                .map(|((shl_start, shl_end), tmat)| {
                                    tmat.dot(&p_spin_block.slice(s![shl_start..shl_end, ..]))
                                })
                                .collect::<Vec<_>>();

                            // Concatenate blocks row-wise for various shells within spin block (ispin, jspin).
                            let trow_p_coeff = concatenate(
                                Axis(0),
                                &trow_p_blocks.iter().map(|trow_p_block| trow_p_block.view()).collect::<Vec<_>>(),
                            )
                            .expect("Unable to concatenate the transformed rows for the various shells.");

                            // Transform columns within spin block (ispin, jspin).
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

                            // Concatenate blocks column-wise for various shells within spin block (ispin, jspin).
                            concatenate(
                                Axis(1),
                                &tcol_trow_p_blocks
                                    .iter()
                                    .map(|tcol_trow_p_block| tcol_trow_p_block.view())
                                    .collect::<Vec<_>>(),
                            )
                            .expect("Unable to concatenate the transformed columns for the various shells.")
                        })
                        .collect::<Vec<_>>();

                        // Concatenate spin blocks at fixed ispin, across all jspin.
                        concatenate(
                            Axis(1),
                            &ispin_blocks
                                .iter()
                                .map(|jspin_block| jspin_block.view())
                                .collect::<Vec<_>>(),
                        )
                        .expect("Unable to concatenate the transformed spin blocks horizontally.")
                    })
                    .collect::<Vec<_>>();

                // Concatenate spin blocks across all ispin.
                concatenate(
                    Axis(0),
                    &jspin_blocks
                        .iter()
                        .map(|ispin_block| ispin_block.view())
                        .collect::<Vec<_>>(),
                )
                .expect("Unable to concatenate the transformed spin blocks vertically.")
            }
        };
        self.density_matrix = new_denmat;
        self
    }
}

// ------------------------
// SpinUnitaryTransformable
// ------------------------

// ~~~~~~~~~~~~~~~~~~
// For real densities
// ~~~~~~~~~~~~~~~~~~

impl<'a> SpinUnitaryTransformable for Density<'a, f64> {
    /// Performs a spin transformation in-place.
    ///
    /// # Arguments
    ///
    /// * `dmat` - The two-dimensional representation matrix of the transformation in the basis of
    /// the $`\{ \alpha, \beta \}`$ spinors (*i.e.* decreasing $`m`$ order).
    fn transform_spin_mut(
        &mut self,
        dmat: &Array2<Complex<f64>>,
    ) -> Result<&mut Self, TransformationError> {
        let cdmat = dmat.view().split_complex();
        if approx::relative_ne!(
            cdmat.im.map(|x| x.powi(2)).sum().sqrt(),
            0.0,
            epsilon = 1e-14,
            max_relative = 1e-14,
        ) {
            log::error!("Spin transformation matrix is complex-valued:\n{dmat}");
            Err(TransformationError(
                "Complex spin transformations cannot be performed with real densities.".to_string(),
            ))
        } else {
            let rdmat = cdmat.re.to_owned();
            match self.spin_constraint {
                SpinConstraint::Restricted(_) | SpinConstraint::Unrestricted(_, _) => {
                    if approx::relative_eq!(
                        (&rdmat - Array2::<f64>::eye(2))
                            .map(|x| x.abs().powi(2))
                            .sum()
                            .sqrt(),
                        0.0,
                        epsilon = 1e-14,
                        max_relative = 1e-14,
                    ) || approx::relative_eq!(
                        (&rdmat + Array2::<f64>::eye(2))
                            .map(|x| x.abs().powi(2))
                            .sum()
                            .sqrt(),
                        0.0,
                        epsilon = 1e-14,
                        max_relative = 1e-14,
                    ) {
                        // Identity or minus-identity spin rotation
                        Ok(self)
                    } else {
                        log::error!("Unsupported spin transformation matrix:\n{}", &rdmat);
                        Err(TransformationError(
                            "Only the identity or negative identity spin transformations are supported with restricted or unrestricted spin constraint."
                                .to_string(),
                        ))
                    }
                }
                SpinConstraint::Generalised(nspins, increasingm) => {
                    if nspins != 2 {
                        return Err(TransformationError(
                            "Only two-component spinor transformations are supported for now."
                                .to_string(),
                        ));
                    }

                    let nspatial = self.bao.n_funcs();
                    let old_denmat = &self.density_matrix;
                    let new_denmat = if !increasingm {
                        let aa = old_denmat.slice(s![0..nspatial, 0..nspatial]).to_owned();
                        let ab = old_denmat
                            .slice(s![0..nspatial, nspatial..2 * nspatial])
                            .to_owned();
                        let ba = old_denmat
                            .slice(s![nspatial..2 * nspatial, 0..nspatial])
                            .to_owned();
                        let bb = old_denmat
                            .slice(s![nspatial..2 * nspatial, nspatial..2 * nspatial])
                            .to_owned();
                        let m00aa = &aa * rdmat[[0, 0]];
                        let m01ba = &ba * rdmat[[0, 1]];
                        let m00ab = &ab * rdmat[[0, 0]];
                        let m01bb = &bb * rdmat[[0, 1]];
                        let m10aa = &aa * rdmat[[1, 0]];
                        let m11ba = &ba * rdmat[[1, 1]];
                        let m10ab = &ab * rdmat[[1, 0]];
                        let m11bb = &bb * rdmat[[1, 1]];
                        let m00aa01ba = m00aa + m01ba;
                        let m00ab01bb = m00ab + m01bb;
                        let m10aa11ba = m10aa + m11ba;
                        let m10ab11bb = m10ab + m11bb;
                        let t_aa = &m00aa01ba * rdmat[[0, 0]] + &m00ab01bb * rdmat[[0, 1]];
                        let t_ab = &m00aa01ba * rdmat[[1, 0]] + &m00ab01bb * rdmat[[1, 1]];
                        let t_ba = &m10aa11ba * rdmat[[0, 0]] + &m10ab11bb * rdmat[[0, 1]];
                        let t_bb = &m10aa11ba * rdmat[[1, 0]] + &m10ab11bb * rdmat[[1, 1]];
                        let t_a = concatenate(Axis(1), &[t_aa.view(), t_ab.view()]).expect(
                            "Unable to concatenate the transformed columns for the various shells.",
                        );
                        let t_b = concatenate(Axis(1), &[t_ba.view(), t_bb.view()]).expect(
                            "Unable to concatenate the transformed columns for the various shells.",
                        );
                        concatenate(Axis(0), &[t_a.view(), t_b.view()]).expect(
                            "Unable to concatenate the transformed rows for the various shells.",
                        )
                    } else {
                        let bb = old_denmat.slice(s![0..nspatial, 0..nspatial]).to_owned();
                        let ba = old_denmat
                            .slice(s![0..nspatial, nspatial..2 * nspatial])
                            .to_owned();
                        let ab = old_denmat
                            .slice(s![nspatial..2 * nspatial, 0..nspatial])
                            .to_owned();
                        let aa = old_denmat
                            .slice(s![nspatial..2 * nspatial, nspatial..2 * nspatial])
                            .to_owned();
                        let m00bb = &bb * rdmat[[0, 0]];
                        let m01ab = &ab * rdmat[[0, 1]];
                        let m00ba = &ba * rdmat[[0, 0]];
                        let m01aa = &aa * rdmat[[0, 1]];
                        let m10bb = &bb * rdmat[[1, 0]];
                        let m11ab = &ab * rdmat[[1, 1]];
                        let m10ba = &ba * rdmat[[1, 0]];
                        let m11aa = &aa * rdmat[[1, 1]];
                        let m00bb01ab = m00bb + m01ab;
                        let m00ba01aa = m00ba + m01aa;
                        let m10bb11ab = m10bb + m11ab;
                        let m10ba11aa = m10ba + m11aa;
                        let t_bb = &m00bb01ab * rdmat[[0, 0]] + &m00ba01aa * rdmat[[0, 1]];
                        let t_ba = &m00bb01ab * rdmat[[1, 0]] + &m00ba01aa * rdmat[[1, 1]];
                        let t_ab = &m10bb11ab * rdmat[[0, 0]] + &m10ba11aa * rdmat[[0, 1]];
                        let t_aa = &m10bb11ab * rdmat[[1, 0]] + &m10ba11aa * rdmat[[1, 1]];
                        let t_b = concatenate(Axis(1), &[t_bb.view(), t_ba.view()]).expect(
                            "Unable to concatenate the transformed columns for the various shells.",
                        );
                        let t_a = concatenate(Axis(1), &[t_ab.view(), t_aa.view()]).expect(
                            "Unable to concatenate the transformed columns for the various shells.",
                        );
                        concatenate(Axis(0), &[t_b.view(), t_a.view()]).expect(
                            "Unable to concatenate the transformed rows for the various shells.",
                        )
                    };
                    self.density_matrix = new_denmat;
                    Ok(self)
                }
            }
        }
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~
// For complex determinants
// ~~~~~~~~~~~~~~~~~~~~~~~~

impl<'a, T> SpinUnitaryTransformable for Density<'a, Complex<T>>
where
    T: Clone,
    Complex<T>: ComplexFloat<Real = T>
        + LinalgScalar
        + ScalarOperand
        + Lapack
        + Mul<Complex<T>, Output = Complex<T>>
        + Mul<Complex<f64>, Output = Complex<T>>,
{
    /// Performs a spin transformation in-place.
    ///
    /// # Arguments
    ///
    /// * `dmat` - The two-dimensional representation matrix of the transformation in the basis of
    /// the $`\{ \alpha, \beta \}`$ spinors (*i.e.* decreasing $`m`$ order).
    ///
    /// # Panics
    ///
    /// Panics if the spin constraint is not generalised. Spin transformations can only be
    /// performed with generalised spin constraint.
    fn transform_spin_mut(
        &mut self,
        dmat: &Array2<Complex<f64>>,
    ) -> Result<&mut Self, TransformationError> {
        match self.spin_constraint {
            SpinConstraint::Restricted(_) | SpinConstraint::Unrestricted(_, _) => {
                if approx::relative_eq!(
                    (dmat - Array2::<Complex<f64>>::eye(2))
                        .map(|x| x.abs().powi(2))
                        .sum()
                        .sqrt(),
                    0.0,
                    epsilon = 1e-14,
                    max_relative = 1e-14,
                ) || approx::relative_eq!(
                    (dmat + Array2::<Complex<f64>>::eye(2))
                        .map(|x| x.abs().powi(2))
                        .sum()
                        .sqrt(),
                    0.0,
                    epsilon = 1e-14,
                    max_relative = 1e-14,
                ) {
                    // Identity or minus identity spin rotation
                    Ok(self)
                } else {
                    log::error!("Unsupported spin transformation matrix:\n{}", dmat);
                    Err(TransformationError(
                        "Only the identity or negative identity spin transformations are possible with restricted spin constraint."
                            .to_string(),
                    ))
                }
            }
            SpinConstraint::Generalised(nspins, increasingm) => {
                if nspins != 2 {
                    panic!("Only two-component spinor transformations are supported for now.");
                }

                let nspatial = self.bao.n_funcs();
                let old_denmat = &self.density_matrix;

                let new_denmat = if !increasingm {
                    let aa = old_denmat.slice(s![0..nspatial, 0..nspatial]).to_owned();
                    let ab = old_denmat
                        .slice(s![0..nspatial, nspatial..2 * nspatial])
                        .to_owned();
                    let ba = old_denmat
                        .slice(s![nspatial..2 * nspatial, 0..nspatial])
                        .to_owned();
                    let bb = old_denmat
                        .slice(s![nspatial..2 * nspatial, nspatial..2 * nspatial])
                        .to_owned();
                    let m00aa = &aa * dmat[[0, 0]];
                    let m01ba = &ba * dmat[[0, 1]];
                    let m00ab = &ab * dmat[[0, 0]];
                    let m01bb = &bb * dmat[[0, 1]];
                    let m10aa = &aa * dmat[[1, 0]];
                    let m11ba = &ba * dmat[[1, 1]];
                    let m10ab = &ab * dmat[[1, 0]];
                    let m11bb = &bb * dmat[[1, 1]];
                    let m00aa01ba = m00aa + m01ba;
                    let m00ab01bb = m00ab + m01bb;
                    let m10aa11ba = m10aa + m11ba;
                    let m10ab11bb = m10ab + m11bb;
                    let t_aa = &m00aa01ba * dmat[[0, 0]].conj() + &m00ab01bb * dmat[[0, 1]].conj();
                    let t_ab = &m00aa01ba * dmat[[1, 0]].conj() + &m00ab01bb * dmat[[1, 1]].conj();
                    let t_ba = &m10aa11ba * dmat[[0, 0]].conj() + &m10ab11bb * dmat[[0, 1]].conj();
                    let t_bb = &m10aa11ba * dmat[[1, 0]].conj() + &m10ab11bb * dmat[[1, 1]].conj();
                    let t_a = concatenate(Axis(1), &[t_aa.view(), t_ab.view()]).expect(
                        "Unable to concatenate the transformed columns for the various shells.",
                    );
                    let t_b = concatenate(Axis(1), &[t_ba.view(), t_bb.view()]).expect(
                        "Unable to concatenate the transformed columns for the various shells.",
                    );
                    concatenate(Axis(0), &[t_a.view(), t_b.view()]).expect(
                        "Unable to concatenate the transformed rows for the various shells.",
                    )
                } else {
                    let bb = old_denmat.slice(s![0..nspatial, 0..nspatial]).to_owned();
                    let ba = old_denmat
                        .slice(s![0..nspatial, nspatial..2 * nspatial])
                        .to_owned();
                    let ab = old_denmat
                        .slice(s![nspatial..2 * nspatial, 0..nspatial])
                        .to_owned();
                    let aa = old_denmat
                        .slice(s![nspatial..2 * nspatial, nspatial..2 * nspatial])
                        .to_owned();
                    let m00bb = &bb * dmat[[0, 0]];
                    let m01ab = &ab * dmat[[0, 1]];
                    let m00ba = &ba * dmat[[0, 0]];
                    let m01aa = &aa * dmat[[0, 1]];
                    let m10bb = &bb * dmat[[1, 0]];
                    let m11ab = &ab * dmat[[1, 1]];
                    let m10ba = &ba * dmat[[1, 0]];
                    let m11aa = &aa * dmat[[1, 1]];
                    let m00bb01ab = m00bb + m01ab;
                    let m00ba01aa = m00ba + m01aa;
                    let m10bb11ab = m10bb + m11ab;
                    let m10ba11aa = m10ba + m11aa;
                    let t_bb = &m00bb01ab * dmat[[0, 0]].conj() + &m00ba01aa * dmat[[0, 1]].conj();
                    let t_ba = &m00bb01ab * dmat[[1, 0]].conj() + &m00ba01aa * dmat[[1, 1]].conj();
                    let t_ab = &m10bb11ab * dmat[[0, 0]].conj() + &m10ba11aa * dmat[[0, 1]].conj();
                    let t_aa = &m10bb11ab * dmat[[1, 0]].conj() + &m10ba11aa * dmat[[1, 1]].conj();
                    let t_b = concatenate(Axis(1), &[t_bb.view(), t_ba.view()]).expect(
                        "Unable to concatenate the transformed columns for the various shells.",
                    );
                    let t_a = concatenate(Axis(1), &[t_ab.view(), t_aa.view()]).expect(
                        "Unable to concatenate the transformed columns for the various shells.",
                    );
                    concatenate(Axis(0), &[t_b.view(), t_a.view()]).expect(
                        "Unable to concatenate the transformed rows for the various shells.",
                    )
                };
                self.density_matrix = new_denmat;
                Ok(self)
            }
        }
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
