//! Implementation of symmetry transformations for Slater determinants.

use std::ops::Mul;

use approx;
use ndarray::{array, concatenate, s, Array2, Axis, LinalgScalar, ScalarOperand};
use ndarray_linalg::types::Lapack;
use num_complex::{Complex, ComplexFloat};

use crate::angmom::spinor_rotation_3d::SpinConstraint;
use crate::permutation::{IntoPermutation, PermutableCollection, Permutation};
use crate::symmetry::symmetry_element::SymmetryOperation;
use crate::symmetry::symmetry_transformation::{
    assemble_sh_rotation_3d_matrices, permute_array_by_atoms, ComplexConjugationTransformable,
    DefaultTimeReversalTransformable, SpatialUnitaryTransformable, SpinUnitaryTransformable,
    SymmetryTransformable, TimeReversalTransformable, TransformationError,
};
use crate::target::determinant::SlaterDeterminant;

// ---------------------------
// SpatialUnitaryTransformable
// ---------------------------
impl<'a, T> SpatialUnitaryTransformable for SlaterDeterminant<'a, T>
where
    T: ComplexFloat + LinalgScalar + ScalarOperand + Copy + Lapack,
    f64: Into<T>,
{
    fn transform_spatial_mut(
        &mut self,
        rmat: &Array2<f64>,
        perm: Option<&Permutation<usize>>,
    ) -> Result<&mut Self, anyhow::Error> {
        let tmats: Vec<Array2<T>> = assemble_sh_rotation_3d_matrices(self.bao, rmat, perm)?
            .iter()
            .map(|tmat| tmat.map(|&x| x.into()))
            .collect();
        let pbao = if let Some(p) = perm {
            self.bao.permute(p)?
        } else {
            self.bao.clone()
        };
        let new_coefficients = self
            .coefficients
            .iter()
            .map(|old_coeff| match self.spin_constraint {
                SpinConstraint::Restricted(_) | SpinConstraint::Unrestricted(_, _) => {
                    let p_coeff = if let Some(p) = perm {
                        permute_array_by_atoms(old_coeff, p, &[Axis(0)], self.bao)
                    } else {
                        old_coeff.clone()
                    };
                    let t_p_blocks = pbao
                        .shell_boundary_indices()
                        .into_iter()
                        .zip(tmats.iter())
                        .map(|((shl_start, shl_end), tmat)| {
                            tmat.dot(&p_coeff.slice(s![shl_start..shl_end, ..]))
                        })
                        .collect::<Vec<_>>();
                    concatenate(
                        Axis(0),
                        &t_p_blocks.iter().map(|t_p_block| t_p_block.view()).collect::<Vec<_>>(),
                    )
                    .expect("Unable to concatenate the transformed rows for the various shells.")
                }
                SpinConstraint::Generalised(nspins, _) => {
                    let nspatial = self.bao.n_funcs();
                    let t_p_spin_blocks = (0..nspins).map(|ispin| {
                        // Extract spin block ispin.
                        let spin_start = usize::from(ispin) * nspatial;
                        let spin_end = (usize::from(ispin) + 1) * nspatial;
                        let spin_block = old_coeff.slice(s![spin_start..spin_end, ..]).to_owned();

                        // Permute within spin block ispin.
                        let p_spin_block = if let Some(p) = perm {
                            permute_array_by_atoms(&spin_block, p, &[Axis(0)], self.bao)
                        } else {
                            spin_block
                        };

                        // Transform within spin block ispin.
                        let t_p_blocks = pbao
                            .shell_boundary_indices()
                            .into_iter()
                            .zip(tmats.iter())
                            .map(|((shl_start, shl_end), tmat)| {
                                tmat.dot(&p_spin_block.slice(s![shl_start..shl_end, ..]))
                            })
                            .collect::<Vec<_>>();

                        // Concatenate blocks for various shells within spin block ispin.
                        concatenate(
                            Axis(0),
                            &t_p_blocks.iter().map(|t_p_block| t_p_block.view()).collect::<Vec<_>>(),
                        )
                        .expect("Unable to concatenate the transformed rows for the various shells.")
                    }).collect::<Vec<_>>();

                    // Concatenate spin blocks.
                    concatenate(
                        Axis(0),
                        &t_p_spin_blocks
                            .iter()
                            .map(|t_p_spin_block| t_p_spin_block.view())
                            .collect::<Vec<_>>(),
                    )
                    .expect("Unable to concatenate the transformed spin blocks.")
                }
            })
            .collect::<Vec<Array2<T>>>();
        self.coefficients = new_coefficients;
        Ok(self)
    }
}

// ------------------------
// SpinUnitaryTransformable
// ------------------------

// ~~~~~~~~~~~~~~~~~~~~~
// For real determinants
// ~~~~~~~~~~~~~~~~~~~~~

impl<'a> SpinUnitaryTransformable for SlaterDeterminant<'a, f64> {
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
                "Complex spin transformations cannot be performed with real coefficients."
                    .to_string(),
            ))
        } else {
            let rdmat = cdmat.re.to_owned();
            match self.spin_constraint {
                SpinConstraint::Restricted(_) => {
                    if approx::relative_eq!(
                        (&rdmat - Array2::<f64>::eye(2))
                            .map(|x| x.abs().powi(2))
                            .sum()
                            .sqrt(),
                        0.0,
                        epsilon = 1e-14,
                        max_relative = 1e-14,
                    ) {
                        // Identity spin rotation
                        Ok(self)
                    } else if approx::relative_eq!(
                        (&rdmat + Array2::<f64>::eye(2))
                            .map(|x| x.abs().powi(2))
                            .sum()
                            .sqrt(),
                        0.0,
                        epsilon = 1e-14,
                        max_relative = 1e-14,
                    ) {
                        // Negative identity spin rotation
                        self.coefficients
                            .iter_mut()
                            .for_each(|coeff| *coeff *= -1.0);
                        Ok(self)
                    } else {
                        log::error!("Unsupported spin transformation matrix:\n{}", &rdmat);
                        Err(TransformationError(
                            "Only the identity or negative identity spin transformations are possible with restricted spin constraint."
                                .to_string(),
                        ))
                    }
                }
                SpinConstraint::Unrestricted(nspins, increasingm) => {
                    // Only spin flip possible, so the order of the basis in which `dmat` is
                    // expressed and the order of the spin blocks do not need to match.
                    if nspins != 2 {
                        return Err(TransformationError(
                            "Only two-component spinor transformations are supported for now."
                                .to_string(),
                        ));
                    }
                    let dmat_y = array![[0.0, -1.0], [1.0, 0.0]];
                    if approx::relative_eq!(
                        (&rdmat - Array2::<f64>::eye(2))
                            .map(|x| x.abs().powi(2))
                            .sum()
                            .sqrt(),
                        0.0,
                        epsilon = 1e-14,
                        max_relative = 1e-14,
                    ) {
                        // Identity spin rotation
                        Ok(self)
                    } else if approx::relative_eq!(
                        (&rdmat + Array2::<f64>::eye(2))
                            .map(|x| x.abs().powi(2))
                            .sum()
                            .sqrt(),
                        0.0,
                        epsilon = 1e-14,
                        max_relative = 1e-14,
                    ) {
                        // Negative identity spin rotation
                        self.coefficients
                            .iter_mut()
                            .for_each(|coeff| *coeff = -coeff.clone());
                        Ok(self)
                    } else if approx::relative_eq!(
                        (&rdmat - &dmat_y).map(|x| x.abs().powi(2)).sum().sqrt(),
                        0.0,
                        epsilon = 1e-14,
                        max_relative = 1e-14,
                    ) {
                        // π-rotation about y-axis, effectively spin-flip
                        let new_coefficients = if increasingm {
                            vec![self.coefficients[1].clone(), -self.coefficients[0].clone()]
                        } else {
                            vec![-self.coefficients[1].clone(), self.coefficients[0].clone()]
                        };
                        let new_occupations =
                            vec![self.occupations[1].clone(), self.occupations[0].clone()];
                        self.coefficients = new_coefficients;
                        self.occupations = new_occupations;
                        Ok(self)
                    } else if approx::relative_eq!(
                        (&rdmat + &dmat_y).map(|x| x.abs().powi(2)).sum().sqrt(),
                        0.0,
                        epsilon = 1e-14,
                        max_relative = 1e-14,
                    ) {
                        // 3π-rotation about y-axis, effectively negative spin-flip
                        let new_coefficients = if increasingm {
                            vec![-self.coefficients[1].clone(), self.coefficients[0].clone()]
                        } else {
                            vec![self.coefficients[1].clone(), -self.coefficients[0].clone()]
                        };
                        let new_occupations =
                            vec![self.occupations[1].clone(), self.occupations[0].clone()];
                        self.coefficients = new_coefficients;
                        self.occupations = new_occupations;
                        Ok(self)
                    } else {
                        log::error!("Unsupported spin transformation matrix:\n{rdmat}");
                        Err(TransformationError(
                            "Only the identity or πy spin transformations are possible with unrestricted spin constraint."
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
                    let new_coefficients = self
                        .coefficients
                        .iter()
                        .map(|old_coeff| {
                            if !increasingm {
                                let a_coeff = old_coeff.slice(s![0..nspatial, ..]).to_owned();
                                let b_coeff = old_coeff.slice(s![nspatial..2 * nspatial, ..]).to_owned();
                                let t_a_coeff = &a_coeff * rdmat[[0, 0]] + &b_coeff * rdmat[[0, 1]];
                                let t_b_coeff = &a_coeff * rdmat[[1, 0]] + &b_coeff * rdmat[[1, 1]];
                                concatenate(Axis(0), &[t_a_coeff.view(), t_b_coeff.view()]).expect(
                                    "Unable to concatenate the transformed rows for the various shells.",
                                )
                            } else {
                                let b_coeff = old_coeff.slice(s![0..nspatial, ..]).to_owned();
                                let a_coeff = old_coeff.slice(s![nspatial..2 * nspatial, ..]).to_owned();
                                let t_a_coeff = &a_coeff * rdmat[[0, 0]] + &b_coeff * rdmat[[0, 1]];
                                let t_b_coeff = &a_coeff * rdmat[[1, 0]] + &b_coeff * rdmat[[1, 1]];
                                concatenate(Axis(0), &[t_b_coeff.view(), t_a_coeff.view()]).expect(
                                    "Unable to concatenate the transformed rows for the various shells.",
                                )
                            }
                        })
                        .collect::<Vec<Array2<f64>>>();
                    self.coefficients = new_coefficients;
                    Ok(self)
                }
            }
        }
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~
// For complex determinants
// ~~~~~~~~~~~~~~~~~~~~~~~~

impl<'a, T> SpinUnitaryTransformable for SlaterDeterminant<'a, Complex<T>>
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
            SpinConstraint::Restricted(_) => {
                if approx::relative_eq!(
                    (dmat - Array2::<Complex<f64>>::eye(2))
                        .map(|x| x.abs().powi(2))
                        .sum()
                        .sqrt(),
                    0.0,
                    epsilon = 1e-14,
                    max_relative = 1e-14,
                ) {
                    // Identity spin rotation
                    Ok(self)
                } else if approx::relative_eq!(
                    (dmat + Array2::<Complex<f64>>::eye(2))
                        .map(|x| x.abs().powi(2))
                        .sum()
                        .sqrt(),
                    0.0,
                    epsilon = 1e-14,
                    max_relative = 1e-14,
                ) {
                    // Negative identity spin rotation
                    self.coefficients
                        .iter_mut()
                        .for_each(|coeff| *coeff = -coeff.clone());
                    Ok(self)
                } else {
                    log::error!("Unsupported spin transformation matrix:\n{}", dmat);
                    Err(TransformationError(
                        "Only the identity or negative identity spin transformations are possible with restricted spin constraint."
                            .to_string(),
                    ))
                }
            }
            SpinConstraint::Unrestricted(nspins, increasingm) => {
                // Only spin flip possible, so the order of the basis in which `dmat` is
                // expressed and the order of the spin blocks do not need to match.
                if nspins != 2 {
                    return Err(TransformationError(
                        "Only two-component spinor transformations are supported for now."
                            .to_string(),
                    ));
                }
                let dmat_y = array![
                    [Complex::from(0.0), Complex::from(-1.0)],
                    [Complex::from(1.0), Complex::from(0.0)],
                ];
                if approx::relative_eq!(
                    (dmat - Array2::<Complex<f64>>::eye(2))
                        .map(|x| x.abs().powi(2))
                        .sum()
                        .sqrt(),
                    0.0,
                    epsilon = 1e-14,
                    max_relative = 1e-14,
                ) {
                    // Identity spin rotation
                    Ok(self)
                } else if approx::relative_eq!(
                    (dmat + Array2::<Complex<f64>>::eye(2))
                        .map(|x| x.abs().powi(2))
                        .sum()
                        .sqrt(),
                    0.0,
                    epsilon = 1e-14,
                    max_relative = 1e-14,
                ) {
                    // Negative identity spin rotation
                    self.coefficients
                        .iter_mut()
                        .for_each(|coeff| *coeff = -coeff.clone());
                    Ok(self)
                } else if approx::relative_eq!(
                    (dmat - &dmat_y).map(|x| x.abs().powi(2)).sum().sqrt(),
                    0.0,
                    epsilon = 1e-14,
                    max_relative = 1e-14,
                ) {
                    // π-rotation about y-axis, effectively spin-flip
                    let new_coefficients = if increasingm {
                        vec![self.coefficients[1].clone(), -self.coefficients[0].clone()]
                    } else {
                        vec![-self.coefficients[1].clone(), self.coefficients[0].clone()]
                    };
                    let new_occupations =
                        vec![self.occupations[1].clone(), self.occupations[0].clone()];
                    self.coefficients = new_coefficients;
                    self.occupations = new_occupations;
                    Ok(self)
                } else if approx::relative_eq!(
                    (dmat + &dmat_y).map(|x| x.abs().powi(2)).sum().sqrt(),
                    0.0,
                    epsilon = 1e-14,
                    max_relative = 1e-14,
                ) {
                    // 3π-rotation about y-axis, effectively negative spin-flip
                    let new_coefficients = if increasingm {
                        vec![-self.coefficients[1].clone(), self.coefficients[0].clone()]
                    } else {
                        vec![self.coefficients[1].clone(), -self.coefficients[0].clone()]
                    };
                    let new_occupations =
                        vec![self.occupations[1].clone(), self.occupations[0].clone()];
                    self.coefficients = new_coefficients;
                    self.occupations = new_occupations;
                    Ok(self)
                } else {
                    log::error!("Unsupported spin transformation matrix:\n{dmat}");
                    Err(TransformationError(
                        "Only the identity or πy spin transformations are possible with unrestricted spin constraint."
                            .to_string(),
                    ))
                }
            }
            SpinConstraint::Generalised(nspins, increasingm) => {
                if nspins != 2 {
                    panic!("Only two-component spinor transformations are supported for now.");
                }

                let nspatial = self.bao.n_funcs();

                let new_coefficients = self
                    .coefficients
                    .iter()
                    .map(|old_coeff| {
                        if increasingm {
                            let b_coeff = old_coeff.slice(s![0..nspatial, ..]).to_owned();
                            let a_coeff = old_coeff.slice(s![nspatial..2 * nspatial, ..]).to_owned();
                            let t_a_coeff = &a_coeff * dmat[[0, 0]] + &b_coeff * dmat[[0, 1]];
                            let t_b_coeff = &a_coeff * dmat[[1, 0]] + &b_coeff * dmat[[1, 1]];
                            concatenate(Axis(0), &[t_b_coeff.view(), t_a_coeff.view()]).expect(
                                "Unable to concatenate the transformed rows for the various shells.",
                            )
                        } else {
                            let a_coeff = old_coeff.slice(s![0..nspatial, ..]).to_owned();
                            let b_coeff = old_coeff.slice(s![nspatial..2 * nspatial, ..]).to_owned();
                            let t_a_coeff = &a_coeff * dmat[[0, 0]] + &b_coeff * dmat[[0, 1]];
                            let t_b_coeff = &a_coeff * dmat[[1, 0]] + &b_coeff * dmat[[1, 1]];
                            concatenate(Axis(0), &[t_a_coeff.view(), t_b_coeff.view()]).expect(
                                "Unable to concatenate the transformed rows for the various shells.",
                            )
                        }
                    })
                    .collect::<Vec<Array2<Complex<T>>>>();
                self.coefficients = new_coefficients;
                Ok(self)
            }
        }
    }
}

// -------------------------------
// ComplexConjugationTransformable
// -------------------------------

impl<'a, T> ComplexConjugationTransformable for SlaterDeterminant<'a, T>
where
    T: ComplexFloat + Lapack,
{
    /// Performs a complex conjugation in-place.
    fn transform_cc_mut(&mut self) -> &mut Self {
        self.coefficients
            .iter_mut()
            .for_each(|coeff| coeff.mapv_inplace(|x| x.conj()));
        self.complex_conjugated = !self.complex_conjugated;
        self
    }
}

// --------------------------------
// DefaultTimeReversalTransformable
// --------------------------------
impl<'a, T> DefaultTimeReversalTransformable for SlaterDeterminant<'a, T> where
    T: ComplexFloat + Lapack
{
}

// ---------------------
// SymmetryTransformable
// ---------------------
impl<'a, T> SymmetryTransformable for SlaterDeterminant<'a, T>
where
    T: ComplexFloat + Lapack,
    SlaterDeterminant<'a, T>: SpatialUnitaryTransformable + TimeReversalTransformable,
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
