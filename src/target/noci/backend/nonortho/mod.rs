use std::fmt::LowerExp;
use std::iter::Product;

use anyhow::{self, ensure, format_err};
use derive_builder::Builder;
use duplicate::duplicate_item;
use indexmap::IndexSet;
use itertools::Itertools;
use ndarray::{
    stack, Array1, Array2, ArrayView1, ArrayView2, ArrayView4, Axis, Ix0, Ix2, ScalarOperand,
};
use ndarray_einsum::einsum;
use ndarray_linalg::types::Lapack;
use ndarray_linalg::{Determinant, Eig, Eigh, Norm, Scalar, SVD, UPLO};
use num::{Complex, Float};
use num_complex::ComplexFloat;

use crate::angmom::spinor_rotation_3d::StructureConstraint;

use super::denmat::{calc_unweighted_codensity_matrix, calc_weighted_codensity_matrix};

#[cfg(test)]
#[path = "nonortho_tests.rs"]
mod nonortho_tests;

/// Structure containing Löwdin-paired coefficients, the corresponding Löwdin overlaps, and the
/// indices of the zero overlaps.
///
/// The Löwdin-paired coefficients satisfy
/// ```math
///     ^{wx}\mathbf{\Lambda}
///         = \mathrm{diag}(^{wx}\lambda_i)
///         = ^{w}\!\tilde{\mathbf{C}}^{\dagger\lozenge}
///           \ \mathbf{S}_{\mathrm{AO}}
///           \ ^{x}\tilde{\mathbf{C}}.
/// ```
#[derive(Builder)]
#[builder(build_fn(validate = "Self::validate"))]
pub struct LowdinPairedCoefficients<T: ComplexFloat> {
    /// The $`^{w}\!\tilde{\mathbf{C}}`$ coefficient matrix.
    paired_cw: Array2<T>,

    /// The $`^{x}\!\tilde{\mathbf{C}}`$ coefficient matrix.
    paired_cx: Array2<T>,

    /// The Löwdin overlaps $`\{^{wx}\lambda_i\}`$.
    lowdin_overlaps: Vec<T>,

    /// The indices of the zero overlaps with respect to [`Self::thresh_zeroov`]. If these are not
    /// provided, the specified [`Self::thresh_zeroov`] will be used to deduce them from the
    /// specified `[Self::lowdin_overlaps]`.
    #[builder(default = "self.default_zero_indices()?")]
    zero_indices: IndexSet<usize>,

    /// Threshold for determining zero overlaps.
    thresh_zeroov: T::Real,

    /// Boolean indicating whether the coefficients have been Löwdin-paired with respect to the
    /// complex-symmetric inner product.
    complex_symmetric: bool,
}

impl<T: ComplexFloat> LowdinPairedCoefficientsBuilder<T> {
    fn validate(&self) -> Result<(), String> {
        let paired_cw = self
            .paired_cw
            .as_ref()
            .ok_or("Löwdin-paired coefficients `paired_cw` not set.".to_string())?;
        let paired_cx = self
            .paired_cx
            .as_ref()
            .ok_or("Löwdin-paired coefficients `paired_cx` not set.".to_string())?;
        let lowdin_overlaps = self
            .lowdin_overlaps
            .as_ref()
            .ok_or("Löwdin overlaps not set.".to_string())?;
        let zero_indices = self
            .zero_indices
            .as_ref()
            .ok_or("Indices of zero Löwdin overlaps not set.".to_string())?;

        if paired_cw.shape() == paired_cx.shape() {
            let lowdin_dim = paired_cw.shape()[1];
            if lowdin_dim == lowdin_overlaps.len() {
                if zero_indices.iter().all(|i| *i < lowdin_dim) {
                    Ok(())
                } else {
                    Err("Some indices of zero Löwdin overlaps are out-of-bound!".to_string())
                }
            } else {
                Err(
                    "Inconsistent number of Löwdin-paired orbitals and Löwdin overlaps."
                        .to_string(),
                )
            }
        } else {
            Err(format!(
                "Inconsistent shapes between `paired_cw` ({:?}) and `paired_cx` ({:?}).",
                paired_cw.shape(),
                paired_cx.shape()
            ))
        }
    }

    fn default_zero_indices(&self) -> Result<IndexSet<usize>, String> {
        let lowdin_overlaps = self
            .lowdin_overlaps
            .as_ref()
            .ok_or("Löwdin overlaps not set.".to_string())?;
        let thresh_zeroov = self
            .thresh_zeroov
            .as_ref()
            .ok_or("threshold for zero Löwdin overlaps not set.".to_string())?;
        let zero_indices = lowdin_overlaps
            .iter()
            .enumerate()
            .filter(|(_, ov)| ComplexFloat::abs(**ov) < *thresh_zeroov)
            .map(|(i, _)| i)
            .collect::<IndexSet<_>>();
        Ok(zero_indices)
    }
}

impl<T: ComplexFloat> LowdinPairedCoefficients<T> {
    pub fn builder() -> LowdinPairedCoefficientsBuilder<T> {
        LowdinPairedCoefficientsBuilder::<T>::default()
    }

    /// Returns the Löwdin-paired coefficients.
    pub fn paired_coefficients(&self) -> (&Array2<T>, &Array2<T>) {
        (&self.paired_cw, &self.paired_cx)
    }

    /// Returns the number of atomic-orbital basis functions in which the coefficient matrices are
    /// expressed.
    pub fn nbasis(&self) -> usize {
        self.paired_cw.nrows()
    }

    /// Returns the number of molecular orbitals being Löwdin-paired.
    pub fn lowdin_dim(&self) -> usize {
        self.lowdin_overlaps.len()
    }

    /// Returns the number of zero Löwdin overlaps.
    pub fn n_lowdin_zeros(&self) -> usize {
        self.zero_indices.len()
    }

    /// Returns the Löwdin overlaps.
    pub fn lowdin_overlaps(&self) -> &Vec<T> {
        &self.lowdin_overlaps
    }

    /// Returns the indices of the zero Löwdin overlaps.
    pub fn zero_indices(&self) -> &IndexSet<usize> {
        &self.zero_indices
    }

    /// Returns the indices of the non-zero Löwdin overlaps.
    pub fn nonzero_indices(&self) -> IndexSet<usize> {
        (0..self.lowdin_dim())
            .filter(|i| !self.zero_indices.contains(i))
            .collect::<IndexSet<_>>()
    }

    /// Returns the boolean indicating whether the Löwdin pairing is with respect to the
    /// complex-symmetric inner product.
    pub fn complex_symmetric(&self) -> bool {
        self.complex_symmetric
    }
}

impl<T: ComplexFloat + Product> LowdinPairedCoefficients<T> {
    /// The reduced overlap between the two determinants.
    pub fn reduced_overlap(&self) -> T {
        self.nonzero_indices()
            .iter()
            .map(|i| self.lowdin_overlaps[*i])
            .product()
    }
}

/// Performs Löwdin pairing on two coefficient matrices $`^{w}\mathbf{C}`$ and
/// $`^{x}\mathbf{C}`$.
///
/// Löwdin pairing ensures that
/// ```math
///     ^{wx}\mathbf{\Lambda}
///         = \mathrm{diag}(^{wx}\lambda_i)
///         = ^{w}\!\tilde{\mathbf{C}}^{\dagger\lozenge}
///           \ \mathbf{S}_{\mathrm{AO}}
///           \ ^{x}\tilde{\mathbf{C}},
/// ```
/// where the Löwdin-paired coefficient matrices are given by
/// ```math
///     \begin{align*}
///         ^{w}\!\tilde{\mathbf{C}}
///             &=\ ^{w}\mathbf{C}\ ^{wx}\mathbf{U}^{\lozenge} \\
///         ^{x}\!\tilde{\mathbf{C}}
///             &=\ ^{x}\mathbf{C}\ ^{wx}\mathbf{V}
///     \end{align*}
/// ```
/// with $`^{wx}\mathbf{U}`$ and $`^{wx}\mathbf{V}`$ being SVD factorisation matrices:
/// ```math
///     ^{w}\mathbf{C}^{\dagger\lozenge}
///     \ \mathbf{S}_{\mathrm{AO}}
///     \ ^{x}\mathbf{C}
///     =
///     \ ^{wx}\mathbf{U}
///     \ ^{wx}\mathbf{\Lambda}
///     \ ^{wx}\mathbf{V}^{\dagger}.
/// ```
///
/// We note that the first columns of $`^{w}\!\tilde{\mathbf{C}}`$ and $`^{x}\!\tilde{\mathbf{C}}`$
/// are also adjusted by the determinants of $`^{wx}\mathbf{U}`$ and $`^{wx}\mathbf{V}`$ as
/// appropriate to ensure that the Slater determinants corresponding to them remain invariant with
/// respect to the unitary transformations brought about by $`^{wx}\mathbf{U}`$ and $`^{wx}\mathbf{V}`$.
///
/// # Arguments
///
/// * `cw` - Coefficient matrix $`^{w}\mathbf{C}`$.
/// * `cx` - Coefficient matrix $`^{x}\mathbf{C}`$.
/// * `sao` - The overlap matrix $`\mathbf{S}_{\mathrm{AO}}`$ of the underlying atomic basis
/// functions.
/// * `complex_symmetric` - If `true`, $`\lozenge = \star`$. If `false`, $`\lozenge = \hat{e}`$.
/// * `thresh_offdiag` - Threshold to check if the off-diagonal elements in the original orbital
/// overlap matrix and in the Löwdin-paired orbital overlap matrix $`^{wx}\mathbf{\Lambda}`$ are zero.
/// * `thresh_zeroov` - Threshold to identify which Löwdin overlaps $`^{wx}\lambda_i`$ are zero.
///
/// # Returns
///
/// A [`LowdinPairedCoefficients`] structure containing the result of the Löwdin pairing.
pub fn calc_lowdin_pairing<T>(
    cw: &ArrayView2<T>,
    cx: &ArrayView2<T>,
    sao: &ArrayView2<T>,
    complex_symmetric: bool,
    thresh_offdiag: <T as ComplexFloat>::Real,
    thresh_zeroov: <T as ComplexFloat>::Real,
) -> Result<LowdinPairedCoefficients<T>, anyhow::Error>
where
    T: ComplexFloat + Lapack,
    <T as ComplexFloat>::Real: PartialOrd + LowerExp,
{
    if cw.shape() != cx.shape() {
        Err(format_err!(
            "Coefficient dimensions mismatched: cw ({:?}) !~ cx ({:?}).",
            cw.shape(),
            cx.shape()
        ))
    } else {
        let init_orb_ovmat = if complex_symmetric {
            einsum("ji,jk,kl->il", &[cw, sao, cx])
        } else {
            einsum("ji,jk,kl->il", &[&cw.map(|x| x.conj()), sao, cx])
        }
        .map_err(|err| format_err!(err))?
        .into_dimensionality::<Ix2>()?;

        let max_offdiag = (&init_orb_ovmat - &Array2::from_diag(&init_orb_ovmat.diag().to_owned()))
            .iter()
            .map(|x| ComplexFloat::abs(*x))
            .max_by(|x, y| {
                x.partial_cmp(y)
                    .expect("Unable to compare two `abs` values.")
            })
            .ok_or_else(|| format_err!("Unable to determine the maximum off-diagonal element."))?;

        if max_offdiag <= thresh_offdiag {
            let lowdin_overlaps = init_orb_ovmat.into_diag().to_vec();
            let zero_indices = lowdin_overlaps
                .iter()
                .enumerate()
                .filter(|(_, ov)| ComplexFloat::abs(**ov) < thresh_zeroov)
                .map(|(i, _)| i)
                .collect::<IndexSet<_>>();
            LowdinPairedCoefficients::builder()
                .paired_cw(cw.to_owned())
                .paired_cx(cx.to_owned())
                .lowdin_overlaps(lowdin_overlaps)
                .zero_indices(zero_indices)
                .thresh_zeroov(thresh_zeroov)
                .complex_symmetric(complex_symmetric)
                .build()
                .map_err(|err| format_err!(err))
        } else {
            let (u_opt, _, vh_opt) = init_orb_ovmat.svd(true, true)?;
            let u = u_opt.ok_or_else(|| format_err!("Unable to compute the U matrix from SVD."))?;
            let vh =
                vh_opt.ok_or_else(|| format_err!("Unable to compute the V matrix from SVD."))?;
            let v = vh.t().map(|x| x.conj());
            let det_v_c = v.det()?.conj();

            let paired_cw = if complex_symmetric {
                let uc = u.map(|x| x.conj());
                let mut cwt = cw.dot(&uc);
                let det_uc_c = uc.det()?.conj();
                cwt.column_mut(0)
                    .iter_mut()
                    .for_each(|x| *x = *x * det_uc_c);
                cwt
            } else {
                let mut cwt = cw.dot(&u);
                let det_u_c = u.det()?.conj();
                cwt.column_mut(0).iter_mut().for_each(|x| *x = *x * det_u_c);
                cwt
            };

            let paired_cx = {
                let mut cxt = cx.dot(&v);
                cxt.column_mut(0).iter_mut().for_each(|x| *x = *x * det_v_c);
                cxt
            };

            let lowdin_orb_ovmat = if complex_symmetric {
                einsum("ji,jk,kl->il", &[&paired_cw, sao, &paired_cx])
            } else {
                einsum(
                    "ji,jk,kl->il",
                    &[&paired_cw.map(|x| x.conj()), sao, &paired_cx],
                )
            }
            .map_err(|err| format_err!(err))?
            .into_dimensionality::<Ix2>()?;

            let max_offdiag_lowdin = (&lowdin_orb_ovmat - &Array2::from_diag(&lowdin_orb_ovmat.diag().to_owned()))
                .iter()
                .map(|x| ComplexFloat::abs(*x))
                .max_by(|x, y| {
                    x.partial_cmp(y)
                        .expect("Unable to compare two `abs` values.")
                })
                .ok_or_else(|| format_err!("Unable to determine the maximum off-diagonal element of the Lowdin-paired overlap matrix."))?;
            if max_offdiag_lowdin <= thresh_offdiag {
                let lowdin_overlaps = lowdin_orb_ovmat.into_diag().to_vec();
                let zero_indices = lowdin_overlaps
                    .iter()
                    .enumerate()
                    .filter(|(_, ov)| ComplexFloat::abs(**ov) < thresh_zeroov)
                    .map(|(i, _)| i)
                    .collect::<IndexSet<_>>();
                LowdinPairedCoefficients::builder()
                    .paired_cw(paired_cw.clone())
                    .paired_cx(paired_cx.clone())
                    .lowdin_overlaps(lowdin_overlaps)
                    .zero_indices(zero_indices)
                    .thresh_zeroov(thresh_zeroov)
                    .complex_symmetric(complex_symmetric)
                    .build()
                    .map_err(|err| format_err!(err))
            } else {
                Err(format_err!(
                    "Löwdin overlap matrix deviates from diagonality. Maximum off-diagonal overlap has magnitude {max_offdiag_lowdin:.3e} > threshold of {thresh_offdiag:.3e}. Löwdin pairing has failed."
                ))
            }
        }
    }
}

/// Calculates the matrix element of a zero-particle operator between two Löwdin-paired
/// determinants.
///
/// # Arguments
///
/// `lowdin_paired_coefficientss` - A sequence of pairs of Löwdin-paired coefficients, one for each
/// subspace determined by the specified structure constraint.
/// `o0` - The zero-particle operator.
/// `structure_constraint` - The structure constraint governing the coefficients.
///
/// # Returns
///
/// The zero-particle matrix element.
pub fn calc_o0_matrix_element<T, SC>(
    lowdin_paired_coefficientss: &[LowdinPairedCoefficients<T>],
    o0: T,
    structure_constraint: &SC,
) -> Result<T, anyhow::Error>
where
    T: ComplexFloat + ScalarOperand + Product,
    SC: StructureConstraint,
{
    let nzeros_explicit: usize = lowdin_paired_coefficientss
        .iter()
        .map(|lpc| lpc.n_lowdin_zeros())
        .sum();
    let nzeros = nzeros_explicit * structure_constraint.implicit_factor()?;
    if nzeros > 0 {
        Ok(T::zero())
    } else {
        let reduced_ov_explicit: T = lowdin_paired_coefficientss
            .iter()
            .map(|lpc| lpc.reduced_overlap())
            .product();
        let reduced_ov = (0..structure_constraint.implicit_factor()?)
            .fold(T::one(), |acc, _| acc * reduced_ov_explicit);
        Ok(reduced_ov * o0)
    }
}

/// Calculates the matrix element of a one-particle operator between two Löwdin-paired
/// determinants.
///
/// # Arguments
///
/// `lowdin_paired_coefficientss` - A sequence of pairs of Löwdin-paired coefficients, one for each
/// subspace determined by the specified structure constraint.
/// `o1` - The one-particle operator in the atomic-orbital basis.
/// `structure_constraint` - The structure constraint governing the coefficients.
///
/// # Returns
///
/// The one-particle matrix element.
pub fn calc_o1_matrix_element<T, SC>(
    lowdin_paired_coefficientss: &[LowdinPairedCoefficients<T>],
    o1: &ArrayView2<T>,
    structure_constraint: &SC,
) -> Result<T, anyhow::Error>
where
    T: ComplexFloat + ScalarOperand + Product,
    SC: StructureConstraint,
{
    let nzeros_explicit: usize = lowdin_paired_coefficientss
        .iter()
        .map(|lpc| lpc.n_lowdin_zeros())
        .sum();
    let nzeros = nzeros_explicit * structure_constraint.implicit_factor()?;
    if nzeros > 1 {
        Ok(T::zero())
    } else {
        let reduced_ov_explicit: T = lowdin_paired_coefficientss
            .iter()
            .map(|lpc| lpc.reduced_overlap())
            .product();
        let reduced_ov = (0..structure_constraint.implicit_factor()?)
            .fold(T::one(), |acc, _| acc * reduced_ov_explicit);

        if nzeros == 0 {
            let nbasis = lowdin_paired_coefficientss[0].nbasis();
            let w = (0..structure_constraint.implicit_factor()?)
                .cartesian_product(lowdin_paired_coefficientss.iter())
                .fold(
                    Ok(Array2::<T>::zeros((nbasis, nbasis))),
                    |acc_res, (_, lpc)| {
                        calc_weighted_codensity_matrix(lpc).and_then(|w| acc_res.map(|acc| acc + w))
                    },
                )?;
            // i = μ, j = μ'
            einsum("ij,ji->", &[o1, &w.view()])
                .map_err(|err| format_err!(err))?
                .into_dimensionality::<Ix0>()?
                .into_iter()
                .next()
                .ok_or_else(|| {
                    format_err!("Unable to extract the result of the einsum contraction.")
                })
                .map(|v| v * reduced_ov)
        } else {
            ensure!(
                nzeros == 1,
                "Unexpected number of zero Löwdin overlaps: {nzeros} != 1."
            );
            let ps = (0..structure_constraint.implicit_factor()?)
                .flat_map(|_| {
                    lowdin_paired_coefficientss.iter().flat_map(|lpc| {
                        lpc.zero_indices()
                            .iter()
                            .map(|mbar| calc_unweighted_codensity_matrix(lpc, *mbar))
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            ensure!(
                ps.len() == 1,
                "Unexpected number of unweighted codensity matrices ({}) for one zero overlap.",
                ps.len()
            );
            let p_mbar = ps.first().ok_or_else(|| {
                format_err!("Unable to retrieve the computed unweighted codensity matrix.")
            })?;

            // i = μ, j = μ'
            einsum("ij,ji->", &[o1, &p_mbar.view()])
                .map_err(|err| format_err!(err))?
                .into_dimensionality::<Ix0>()?
                .into_iter()
                .next()
                .ok_or_else(|| {
                    format_err!("Unable to extract the result of the einsum contraction.")
                })
                .map(|v| v * reduced_ov)
        }
    }
}

/// Calculates the matrix element of a two-particle operator between two Löwdin-paired
/// determinants.
///
/// # Arguments
///
/// `lowdin_paired_coefficientss` - A sequence of pairs of Löwdin-paired coefficients, one for each
/// subspace determined by the specified structure constraint.
/// `o1` - The one-particle operator in the atomic-orbital basis.
/// `structure_constraint` - The structure constraint governing the coefficients.
///
/// # Returns
///
/// The two-particle matrix element.
pub fn calc_o2_matrix_element<T, SC>(
    lowdin_paired_coefficientss: &[LowdinPairedCoefficients<T>],
    o2: &ArrayView4<T>,
    structure_constraint: &SC,
) -> Result<T, anyhow::Error>
where
    T: ComplexFloat + ScalarOperand + Product + std::fmt::Display,
    SC: StructureConstraint,
{
    let nzeros_explicit: usize = lowdin_paired_coefficientss
        .iter()
        .map(|lpc| lpc.n_lowdin_zeros())
        .sum();
    let nzeros = nzeros_explicit * structure_constraint.implicit_factor()?;
    if nzeros > 2 {
        Ok(T::zero())
    } else {
        let reduced_ov_explicit: T = lowdin_paired_coefficientss
            .iter()
            .map(|lpc| lpc.reduced_overlap())
            .product();
        let reduced_ov = (0..structure_constraint.implicit_factor()?)
            .fold(T::one(), |acc, _| acc * reduced_ov_explicit);

        if nzeros == 0 {
            let nbasis = lowdin_paired_coefficientss[0].nbasis();
            let w_sigmas = (0..structure_constraint.implicit_factor()?)
                .cartesian_product(lowdin_paired_coefficientss.iter())
                .map(|(_, lpc)| calc_weighted_codensity_matrix(lpc))
                .collect::<Result<Vec<_>, _>>()?;
            let w = w_sigmas.iter().fold(
                Ok::<_, anyhow::Error>(Array2::<T>::zeros((nbasis, nbasis))),
                |acc_res, w_sigma| acc_res.map(|acc| acc + w_sigma),
            )?;

            // i = μ, j = μ', k = ν, l = ν'
            let j_term = einsum("ikjl,ji,lk->", &[o2, &w.view(), &w.view()])
                .map_err(|err| format_err!(err))?
                .into_dimensionality::<Ix0>()?
                .into_iter()
                .next()
                .ok_or_else(|| {
                    format_err!("Unable to extract the result of the einsum contraction.")
                })
                .map(|v| v * reduced_ov / (T::one() + T::one()))?;
            let k_term = w_sigmas
                .iter()
                .fold(Ok(T::zero()), |acc_res, w_sigma| {
                    einsum("ikjl,li,jk->", &[o2, &w_sigma.view(), &w_sigma.view()])
                        .map_err(|err| format_err!(err))?
                        .into_dimensionality::<Ix0>()?
                        .into_iter()
                        .next()
                        .ok_or_else(|| {
                            format_err!("Unable to extract the result of the einsum contraction.")
                        })
                        .and_then(|v| acc_res.map(|acc| acc + v))
                })
                .map(|v| v * reduced_ov / (T::one() + T::one()))?;
            Ok(j_term - k_term)
        } else if nzeros == 1 {
            ensure!(
                nzeros_explicit == 1,
                "Unexpected number of explicit zero Löwdin overlaps: {nzeros_explicit} != 1."
            );

            let nbasis = lowdin_paired_coefficientss[0].nbasis();
            let w = (0..structure_constraint.implicit_factor()?)
                .cartesian_product(lowdin_paired_coefficientss.iter())
                .fold(
                    Ok::<_, anyhow::Error>(Array2::<T>::zeros((nbasis, nbasis))),
                    |acc_res, (_, lpc)| {
                        calc_weighted_codensity_matrix(lpc)
                            .and_then(|w_sigma| acc_res.map(|acc| acc + w_sigma))
                    },
                )?;

            lowdin_paired_coefficientss
                .iter()
                .filter_map(|lpc| {
                    if lpc.n_lowdin_zeros() == 1 {
                        let w_sigma_res = calc_weighted_codensity_matrix(lpc);
                        let mbar = lpc.zero_indices()[0];
                        let p_mbar_sigma_res = calc_unweighted_codensity_matrix(lpc, mbar);
                        Some((w_sigma_res, p_mbar_sigma_res))
                    } else {
                        None
                    }
                })
                .fold(Ok(T::zero()), |acc_res, (w_sigma_res, p_mbar_sigma_res)| {
                    w_sigma_res.and_then(|w_sigma| {
                        p_mbar_sigma_res.and_then(|p_mbar_sigma| {
                            // i = μ, j = μ', k = ν, l = ν'
                            let j_term_1 =
                                einsum("ikjl,ji,lk->", &[o2, &w.view(), &p_mbar_sigma.view()])
                                    .map_err(|err| format_err!(err))?
                                    .into_dimensionality::<Ix0>()?
                                    .into_iter()
                                    .next()
                                    .ok_or_else(|| {
                                        format_err!(
                                    "Unable to extract the result of the einsum contraction."
                                )
                                    })?;
                            let j_term_2 =
                                einsum("ikjl,ji,lk->", &[o2, &p_mbar_sigma.view(), &w.view()])
                                    .map_err(|err| format_err!(err))?
                                    .into_dimensionality::<Ix0>()?
                                    .into_iter()
                                    .next()
                                    .ok_or_else(|| {
                                        format_err!(
                                    "Unable to extract the result of the einsum contraction."
                                )
                                    })?;
                            let k_term_1 = einsum(
                                "ikjl,li,jk->",
                                &[o2, &w_sigma.view(), &p_mbar_sigma.view()],
                            )
                            .map_err(|err| format_err!(err))?
                            .into_dimensionality::<Ix0>()?
                            .into_iter()
                            .next()
                            .ok_or_else(|| {
                                format_err!(
                                    "Unable to extract the result of the einsum contraction."
                                )
                            })?;
                            let k_term_2 = einsum(
                                "ikjl,li,jk->",
                                &[o2, &p_mbar_sigma.view(), &w_sigma.view()],
                            )
                            .map_err(|err| format_err!(err))?
                            .into_dimensionality::<Ix0>()?
                            .into_iter()
                            .next()
                            .ok_or_else(|| {
                                format_err!(
                                    "Unable to extract the result of the einsum contraction."
                                )
                            })?;
                            acc_res.map(|acc| acc + j_term_1 + j_term_2 - k_term_1 - k_term_2)
                        })
                    })
                })
                .map(|v| v * reduced_ov / (T::one() + T::one()))
        } else {
            ensure!(
                nzeros == 2,
                "Unexpected number of zero Löwdin overlaps: {nzeros} != 2."
            );

            let ps = (0..structure_constraint.implicit_factor()?)
                .flat_map(|_| {
                    lowdin_paired_coefficientss.iter().flat_map(|lpc| {
                        lpc.zero_indices()
                            .iter()
                            .map(|mbar| calc_unweighted_codensity_matrix(lpc, *mbar))
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            ensure!(
                ps.len() == 2,
                "Unexpected number of unweighted codensity matrices ({}) for two zero overlaps.",
                ps.len()
            );
            let p_mbar = ps.first().ok_or_else(|| {
                format_err!("Unable to retrieve the first computed unweighted codensity matrix.")
            })?;
            let p_nbar = ps.last().ok_or_else(|| {
                format_err!("Unable to retrieve the second computed unweighted codensity matrix.")
            })?;

            // i = μ, j = μ', k = ν, l = ν'
            let j_term_1 = einsum("ikjl,ji,lk->", &[o2, &p_mbar.view(), &p_nbar.view()])
                .map_err(|err| format_err!(err))?
                .into_dimensionality::<Ix0>()?
                .into_iter()
                .next()
                .ok_or_else(|| {
                    format_err!("Unable to extract the result of the einsum contraction.")
                })?;
            let j_term_2 = einsum("ikjl,ji,lk->", &[o2, &p_nbar.view(), &p_mbar.view()])
                .map_err(|err| format_err!(err))?
                .into_dimensionality::<Ix0>()?
                .into_iter()
                .next()
                .ok_or_else(|| {
                    format_err!("Unable to extract the result of the einsum contraction.")
                })?;

            let (k_term_1, k_term_2) = if lowdin_paired_coefficientss
                .iter()
                .any(|lpc| lpc.n_lowdin_zeros() == 2)
            {
                let k_term_1 = einsum("ikjl,li,jk->", &[o2, &p_mbar.view(), &p_nbar.view()])
                    .map_err(|err| format_err!(err))?
                    .into_dimensionality::<Ix0>()?
                    .into_iter()
                    .next()
                    .ok_or_else(|| {
                        format_err!("Unable to extract the result of the einsum contraction.")
                    })?;
                let k_term_2 = einsum("ikjl,li,jk->", &[o2, &p_nbar.view(), &p_mbar.view()])
                    .map_err(|err| format_err!(err))?
                    .into_dimensionality::<Ix0>()?
                    .into_iter()
                    .next()
                    .ok_or_else(|| {
                        format_err!("Unable to extract the result of the einsum contraction.")
                    })?;
                (k_term_1, k_term_2)
            } else {
                (T::zero(), T::zero())
            };
            Ok(reduced_ov * (j_term_1 - k_term_1 + j_term_2 - k_term_2) / (T::one() + T::one()))
        }
    }
}

/// Performs modified Gram--Schmidt orthonormalisation on a set of column vectors in a matrix with
/// respect to the complex-symmetric or Hermitian dot product.
///
/// # Arguments
///
/// * `vmat` - Matrix containing column vectors forming a basis for a subspace.
/// * `complex_symmetric` - A boolean indicating if the vector dot product is complex-symmetric. If
/// `false`, the conventional Hermitian dot product is used.
/// * `thresh` - A threshold for determining self-orthogonal vectors.
///
/// # Returns
///
/// The orthonormal vectors forming a basis for the same subspace collected as column vectors in a
/// matrix.
///
/// # Errors
///
/// Errors when the orthonormalisation procedure fails, which occurs when there is linear dependency
/// between the basis vectors and/or when self-orthogonal vectors are encountered.
pub fn complex_modified_gram_schmidt<T>(
    vmat: &ArrayView2<T>,
    complex_symmetric: bool,
    thresh: <T as ComplexFloat>::Real,
) -> Result<Array2<T>, anyhow::Error>
where
    T: ComplexFloat + std::fmt::Display + 'static,
{
    let mut us: Vec<Array1<T>> = Vec::with_capacity(vmat.shape()[1]);
    let mut us_sq_norm: Vec<T> = Vec::with_capacity(vmat.shape()[1]);
    for (i, vi) in vmat.columns().into_iter().enumerate() {
        // u[i] now initialised with v[i]
        us.push(vi.to_owned());

        // Project ui onto all uj (0 <= j < i)
        // This is the 'modified' part of Gram--Schmidt. We project the current (and being updated)
        // ui onto uj, rather than projecting vi onto uj. This enhances numerical stability.
        for j in 0..i {
            let p_uj_ui = if complex_symmetric {
                us[j].t().dot(&us[i]) / us_sq_norm[j]
            } else {
                us[j].t().map(|x| x.conj()).dot(&us[i]) / us_sq_norm[j]
            };
            us[i] = &us[i] - us[j].map(|&x| x * p_uj_ui);
        }

        // Evaluate the squared norm of ui which will no longer be changed after this iteration.
        // us_sq_norm[i] now available.
        let us_sq_norm_i = if complex_symmetric {
            us[i].t().dot(&us[i])
        } else {
            us[i].t().map(|x| x.conj()).dot(&us[i])
        };
        if us_sq_norm_i.abs() < thresh {
            return Err(format_err!("A zero-norm vector found: {}", us[i]));
        }
        us_sq_norm.push(us_sq_norm_i);
    }

    // Normalise ui
    for i in 0..us.len() {
        us[i].mapv_inplace(|x| x / us_sq_norm[i].sqrt());
    }

    let ortho_check = us.iter().enumerate().all(|(i, ui)| {
        us.iter().enumerate().all(|(j, uj)| {
            let ov_ij = if complex_symmetric {
                ui.dot(uj)
            } else {
                ui.map(|x| x.conj()).dot(uj)
            };
            i == j || ov_ij.abs() < thresh
        })
    });

    if ortho_check {
        stack(Axis(1), &us.iter().map(|u| u.view()).collect_vec()).map_err(|err| format_err!(err))
    } else {
        Err(format_err!(
            "Post-Gram--Schmidt orthogonality check failed."
        ))
    }
}

/// Trait for Löwdin canonical orthogonalisation of a square matrix.
pub trait CanonicalOrthogonalisable {
    /// Numerical type of the matrix elements.
    type NumType;

    /// Type of real threshold values.
    type RealType;

    /// Calculates the Löwdin canonical orthogonalisation matrix $`\mathbf{X}`$ for a square
    /// matrix.
    ///
    /// # Arguments
    ///
    /// * `complex_symmetric` - Boolean indicating if the orthogonalisation is with respect to the
    /// complex-symmetric inner product.
    /// * `preserves_full_rank` - Boolean indicating if a full-rank square matrix should be left
    /// unchanged, thus forcing $`\mathbf{X} = \mathbf{I}`$.
    /// * `thresh_offdiag` - Threshold for verifying that the orthogonalised matrix is indeed
    /// orthogonal.
    /// * `thresh_zeroov` - Threshold for determining zero eigenvalues of the input square matrix.
    ///
    /// # Returns
    ///
    /// The canonical orthogonalisation result.
    fn calc_canonical_orthogonal_matrix(
        &self,
        complex_symmetric: bool,
        preserves_full_rank: bool,
        thresh_offdiag: Self::RealType,
        thresh_zeroov: Self::RealType,
    ) -> Result<CanonicalOrthogonalisationResult<Self::NumType>, anyhow::Error>;
}

/// Structure containing the results of the Löwdin canonical orthogonalisation.
pub struct CanonicalOrthogonalisationResult<T> {
    /// The eigenvalues of the input matrix.
    eigenvalues: Array1<T>,

    /// The Löwdin canonical orthogonalisation matrix $`\mathbf{X}`$.
    xmat: Array2<T>,

    /// The conjugate of the Löwdin canonical orthogonalisation matrix,
    /// $`\mathbf{X}^{\dagger\lozenge}`$, where $`\lozenge = \star`$ for complex-symmetric matrices
    /// and $`\lozenge = \hat{e}`$ otherwise.
    xmat_d: Array2<T>,
}

impl<T> CanonicalOrthogonalisationResult<T> {
    /// Returns the eigenvalues of the input matrix.
    pub fn eigenvalues(&self) -> ArrayView1<T> {
        self.eigenvalues.view()
    }

    /// Returns the Löwdin canonical orthogonalisation matrix $`\mathbf{X}`$.
    pub fn xmat(&self) -> ArrayView2<T> {
        self.xmat.view()
    }

    /// Returns the conjugate of the Löwdin canonical orthogonalisation matrix,
    /// $`\mathbf{X}^{\dagger\lozenge}`$, where $`\lozenge = \star`$ for complex-symmetric matrices
    /// and $`\lozenge = \hat{e}`$ otherwise.
    pub fn xmat_d(&self) -> ArrayView2<T> {
        self.xmat_d.view()
    }
}

#[duplicate_item(
    [
        dtype_ [ f64 ]
    ]
    [
        dtype_ [ f32 ]
    ]
)]
impl CanonicalOrthogonalisable for ArrayView2<'_, dtype_> {
    type NumType = dtype_;

    type RealType = dtype_;

    fn calc_canonical_orthogonal_matrix(
        &self,
        _: bool,
        preserves_full_rank: bool,
        thresh_offdiag: dtype_,
        thresh_zeroov: dtype_,
    ) -> Result<CanonicalOrthogonalisationResult<Self::NumType>, anyhow::Error> {
        let smat = self;

        // Real, symmetric S
        ensure!(
            (smat.to_owned() - smat.t()).norm_l2() <= thresh_offdiag,
            "Overlap matrix is not real-symmetric."
        );

        // S is real-symmetric, so U is orthogonal, i.e. U^T = U^(-1).
        let (s_eig, umat) = smat.eigh(UPLO::Lower).map_err(|err| format_err!(err))?;
        // Real eigenvalues, so both comparison modes are the same.
        let nonzero_s_indices = s_eig
            .iter()
            .positions(|x| x.abs() > thresh_zeroov)
            .collect_vec();
        let nonzero_s_eig = s_eig.select(Axis(0), &nonzero_s_indices);
        if nonzero_s_eig.iter().any(|v| *v < 0.0) {
            return Err(format_err!(
                "The matrix has negative eigenvalues and therefore cannot be orthogonalised over the reals."
            ));
        }
        let nonzero_umat = umat.select(Axis(1), &nonzero_s_indices);
        let nullity = smat.shape()[0] - nonzero_s_indices.len();
        let (xmat, xmat_d) = if nullity == 0 && preserves_full_rank {
            (Array2::eye(smat.shape()[0]), Array2::eye(smat.shape()[0]))
        } else {
            let s_s = Array2::<dtype_>::from_diag(&nonzero_s_eig.mapv(|x| 1.0 / x.sqrt()));
            (nonzero_umat.dot(&s_s), s_s.dot(&nonzero_umat.t()))
        };
        let res = CanonicalOrthogonalisationResult {
            eigenvalues: s_eig,
            xmat,
            xmat_d,
        };
        Ok(res)
    }
}

impl<T> CanonicalOrthogonalisable for ArrayView2<'_, Complex<T>>
where
    T: Float + Scalar<Complex = Complex<T>>,
    Complex<T>: ComplexFloat<Real = T> + Scalar<Real = T, Complex = Complex<T>> + Lapack,
{
    type NumType = Complex<T>;

    type RealType = T;

    fn calc_canonical_orthogonal_matrix(
        &self,
        complex_symmetric: bool,
        preserves_full_rank: bool,
        thresh_offdiag: T,
        thresh_zeroov: T,
    ) -> Result<CanonicalOrthogonalisationResult<Self::NumType>, anyhow::Error> {
        let smat = self;

        if complex_symmetric {
            // Complex-symmetric S
            ensure!(
                (smat.to_owned() - smat.t()).norm_l2() <= thresh_offdiag,
                "Overlap matrix is not complex-symmetric."
            );
        } else {
            // Complex-Hermitian S
            ensure!(
                (smat.to_owned() - smat.map(|v| v.conj()).t()).norm_l2() <= thresh_offdiag,
                "Overlap matrix is not complex-Hermitian."
            );
        }

        let (s_eig, umat_nonortho) = smat.eig().map_err(|err| format_err!(err))?;

        let nonzero_s_indices = s_eig
            .iter()
            .positions(|x| ComplexFloat::abs(*x) > thresh_zeroov)
            .collect_vec();
        let nonzero_s_eig = s_eig.select(Axis(0), &nonzero_s_indices);
        let nonzero_umat_nonortho = umat_nonortho.select(Axis(1), &nonzero_s_indices);

        // `eig` does not guarantee orthogonality of `nonzero_umat_nonortho`.
        // Gram--Schmidt is therefore required.
        let nonzero_umat = complex_modified_gram_schmidt(
            &nonzero_umat_nonortho.view(),
            complex_symmetric,
            thresh_zeroov,
        )
        .map_err(
            |_| format_err!("Unable to orthonormalise the linearly-independent eigenvectors of the overlap matrix.")
        )?;

        let nullity = smat.shape()[0] - nonzero_s_indices.len();
        let (xmat, xmat_d) = if nullity == 0 && preserves_full_rank {
            (
                Array2::<Complex<T>>::eye(smat.shape()[0]),
                Array2::<Complex<T>>::eye(smat.shape()[0]),
            )
        } else {
            let s_s = Array2::<Complex<T>>::from_diag(
                &nonzero_s_eig.mapv(|x| Complex::<T>::from(T::one()) / x.sqrt()),
            );
            if complex_symmetric {
                (nonzero_umat.dot(&s_s), s_s.dot(&nonzero_umat.t()))
            } else {
                let xmat = nonzero_umat.dot(&s_s);
                let xmat_d = xmat.map(|v| v.conj()).t().to_owned();
                (xmat, xmat_d)
            }
        };
        let res = CanonicalOrthogonalisationResult {
            eigenvalues: s_eig,
            xmat,
            xmat_d,
        };
        Ok(res)
    }
}
