use std::fmt::{Display, LowerExp};
use std::marker::PhantomData;

use anyhow::{self, ensure, format_err};
use derive_builder::Builder;
use itertools::Itertools;
use ndarray::{Array2, ArrayView2, ArrayView4, Axis, ScalarOperand};
use ndarray_linalg::types::Lapack;
use num::FromPrimitive;
use num_complex::ComplexFloat;

use crate::angmom::spinor_rotation_3d::StructureConstraint;
use crate::symmetry::symmetry_transformation::SymmetryTransformable;
use crate::target::determinant::SlaterDeterminant;
use crate::target::noci::backend::nonortho::{
    calc_lowdin_pairing, calc_o0_matrix_element, calc_o1_matrix_element, calc_o2_matrix_element,
};

use super::OrbitMatrix;

#[cfg(test)]
#[path = "hamiltonian_tests.rs"]
mod hamiltonian_tests;

/// Structure for managing the electronic Hamiltonian integrals in an atomic-orbital basis.
#[derive(Builder)]
pub struct HamiltonianAO<'a, T, SC, F>
where
    T: ComplexFloat + Lapack,
    SC: StructureConstraint + Clone,
    F: Fn(&Array2<T>) -> Result<(Array2<T>, Array2<T>), anyhow::Error> + Clone,
{
    /// The nuclear repulsion energy.
    enuc: T,

    /// The one-electron integrals in an atomic-orbital basis (with respect to the specified
    /// structure constraint).
    onee: ArrayView2<'a, T>,

    /// The two-electron integrals in an atomic-orbital basis (with respect to the specified
    /// structure constraint).
    #[builder(default = "None")]
    twoe: Option<ArrayView4<'a, T>>,

    #[builder(default = "None")]
    get_jk: Option<F>,

    /// The structure constraint for the wavefunctions described by this Hamiltonian.
    #[builder(setter(skip), default = "PhantomData")]
    structure_constraint: PhantomData<SC>,
}

impl<'a, T, SC, F> HamiltonianAO<'a, T, SC, F>
where
    T: ComplexFloat + Lapack,
    SC: StructureConstraint + Clone,
    F: Fn(&Array2<T>) -> Result<(Array2<T>, Array2<T>), anyhow::Error> + Clone,
{
    /// Returns a builder for [`HamiltonianAO`].
    pub fn builder() -> HamiltonianAOBuilder<'a, T, SC, F> {
        HamiltonianAOBuilder::<'a, T, SC, F>::default()
    }

    /// Returns the nuclear repulsion energy.
    pub fn enuc(&self) -> T {
        self.enuc
    }

    /// Returns the one-electron integrals in an atomic-orbital basis (with respect to the specified
    /// structure constraint).
    pub fn onee(&'a self) -> &'a ArrayView2<'a, T> {
        &self.onee
    }

    /// Returns the two-electron integrals in an atomic-orbital basis (with respect to the specified
    /// structure constraint).
    pub fn twoe(&self) -> Option<&ArrayView4<'a, T>> {
        self.twoe.as_ref()
    }

    pub fn get_jk(&self) -> Option<&F> {
        self.get_jk.as_ref()
    }
}

impl<'a, T, SC, F> HamiltonianAO<'a, T, SC, F>
where
    T: ComplexFloat + Lapack + ScalarOperand + FromPrimitive,
    <T as ComplexFloat>::Real: LowerExp,
    SC: StructureConstraint + Display + PartialEq + Clone,
    F: Fn(&Array2<T>) -> Result<(Array2<T>, Array2<T>), anyhow::Error> + Clone,
{
    /// Calculates the zero-, one-, and two-electron contributions to the matrix element of the
    /// electronic Hamiltonian between two possibly non-orthogonal Slater determinants.
    ///
    /// The matrix element is given by
    /// ```math
    ///     \braket{\hat{\iota} ^{w}\Psi | \hat{\mathscr{H}} | ^{x}\Psi}
    /// ```
    /// where $`\hat{\iota}`$ is an involutory operator that is either the identity or the
    /// complex-conjugation operator, which depends on whether the specified determinants have been
    /// defined with the [`SlaterDeterminant::complex_symmetric`] boolean set to `false` or `true`,
    /// respectively.
    ///
    /// # Arguments
    ///
    /// * `det_w` - The determinant $`^{w}\Psi`$.
    /// * `det_x` - The determinant $`^{x}\Psi`$.
    /// * `sao` - The atomic-orbital overlap matrix.
    /// * `thresh_offdiag` - Threshold for determining non-zero off-diagonal elements in the
    /// orbital overlap matrix between $`^{w}\Psi`$ and $`^{x}\Psi`$ during Löwdin pairing.
    /// * `thresh_zeroov` - threshold for identifying zero Löwdin overlaps.
    ///
    /// # Returns
    ///
    /// A tuple containing the zero-, one-, and two-electron contributions to the matrix element.
    pub fn calc_hamiltonian_matrix_element_contributions(
        &self,
        det_w: &SlaterDeterminant<T, SC>,
        det_x: &SlaterDeterminant<T, SC>,
        sao: &ArrayView2<T>,
        thresh_offdiag: <T as ComplexFloat>::Real,
        thresh_zeroov: <T as ComplexFloat>::Real,
    ) -> Result<(T, T, T), anyhow::Error> {
        ensure!(
            det_w.structure_constraint() == det_x.structure_constraint(),
            "Inconsistent spin constraints: {} != {}.",
            det_w.structure_constraint(),
            det_x.structure_constraint(),
        );
        let sc = det_w.structure_constraint();

        if det_w.complex_symmetric() != det_x.complex_symmetric() {
            return Err(format_err!(
                "The `complex_symmetric` booleans of the specified determinants do not match: `det_w` (`{}`) != `det_x` (`{}`).",
                det_w.complex_symmetric(),
                det_x.complex_symmetric(),
            ));
        }
        let complex_symmetric = det_w.complex_symmetric();

        let lowdin_paired_coefficientss = det_w
            .coefficients()
            .iter()
            .zip(det_w.occupations().iter())
            .zip(det_x.coefficients().iter().zip(det_x.occupations().iter()))
            .map(|((cw, occw), (cx, occx))| {
                let occw_indices = occw
                    .iter()
                    .enumerate()
                    .filter_map(|(i, occ_i)| {
                        if occ_i.abs() >= det_w.threshold() {
                            Some(i)
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>();
                let cw_occ = cw.select(Axis(1), &occw_indices);
                let occx_indices = occx
                    .iter()
                    .enumerate()
                    .filter_map(|(i, occ_i)| {
                        if occ_i.abs() >= det_x.threshold() {
                            Some(i)
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>();
                let cx_occ = cx.select(Axis(1), &occx_indices);
                calc_lowdin_pairing(
                    &cw_occ.view(),
                    &cx_occ.view(),
                    sao,
                    complex_symmetric,
                    thresh_offdiag,
                    thresh_zeroov,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;

        let zeroe_h_wx = calc_o0_matrix_element(&lowdin_paired_coefficientss, self.enuc, sc)?;
        let onee_h_wx = calc_o1_matrix_element(&lowdin_paired_coefficientss, &self.onee, sc)?;
        let twoe_h_wx =
            calc_o2_matrix_element(&lowdin_paired_coefficientss, self.twoe(), self.get_jk(), sc)?;
        Ok((zeroe_h_wx, onee_h_wx, twoe_h_wx))
    }

    /// Calculates the Hamiltonian matrix in the basis of non-orthogonal Slater determinants.
    ///
    /// Each matrix element is given by
    /// ```math
    ///     \braket{\hat{\iota} ^{w}\Psi | \hat{\mathscr{H}} | ^{x}\Psi}
    /// ```
    /// where $`\hat{\iota}`$ is an involutory operator that is either the identity or the
    /// complex-conjugation operator, which depends on whether the specified determinants have been
    /// defined with the [`SlaterDeterminant::complex_symmetric`] boolean set to `false` or `true`,
    /// respectively.
    ///
    /// # Arguments
    ///
    /// * `dets` - A sequence of Slater determinants to be used as the basis for the Hamiltonian
    /// matrix.
    /// * `sao` - The atomic-orbital overlap matrix.
    /// * `thresh_offdiag` - Threshold for determining non-zero off-diagonal elements in the
    /// orbital overlap matrix between $`^{w}\Psi`$ and $`^{x}\Psi`$ during Löwdin pairing.
    /// * `thresh_zeroov` - threshold for identifying zero Löwdin overlaps.
    ///
    /// # Returns
    ///
    /// The Hamiltonian matrix.
    pub fn calc_hamiltonian_matrix(
        &self,
        dets: &[&SlaterDeterminant<T, SC>],
        sao: &ArrayView2<T>,
        thresh_offdiag: <T as ComplexFloat>::Real,
        thresh_zeroov: <T as ComplexFloat>::Real,
    ) -> Result<Array2<T>, anyhow::Error> {
        let dim = dets.len();
        let mut hmat = Array2::<T>::zeros((dim, dim));
        for pair in dets.iter().enumerate().combinations_with_replacement(2) {
            let (w, det_w) = &pair[0];
            let (x, det_x) = &pair[1];
            let (zeroe_wx, onee_wx, twoe_wx) = self.calc_hamiltonian_matrix_element_contributions(
                det_w,
                det_x,
                sao,
                thresh_offdiag,
                thresh_zeroov,
            )?;
            hmat[(*w, *x)] = zeroe_wx + onee_wx + twoe_wx;
            if *w != *x {
                let (zeroe_xw, onee_xw, twoe_xw) = self
                    .calc_hamiltonian_matrix_element_contributions(
                        det_x,
                        det_w,
                        sao,
                        thresh_offdiag,
                        thresh_zeroov,
                    )?;
                hmat[(*x, *w)] = zeroe_xw + onee_xw + twoe_xw;
            }
        }
        Ok(hmat)
    }
}

impl<'a, T, SC, F> OrbitMatrix<'a, T, SC> for &HamiltonianAO<'a, T, SC, F>
where
    T: ComplexFloat + Lapack + ScalarOperand + FromPrimitive,
    <T as ComplexFloat>::Real: LowerExp,
    SC: StructureConstraint + Clone + Display + PartialEq,
    SlaterDeterminant<'a, T, SC>: SymmetryTransformable,
    F: Fn(&Array2<T>) -> Result<(Array2<T>, Array2<T>), anyhow::Error> + Clone,
{
    fn calc_matrix_element(
        &self,
        det_w: &SlaterDeterminant<T, SC>,
        det_x: &SlaterDeterminant<T, SC>,
        sao: &ArrayView2<T>,
        thresh_offdiag: <T as ComplexFloat>::Real,
        thresh_zeroov: <T as ComplexFloat>::Real,
    ) -> Result<T, anyhow::Error> {
        let (zeroe, onee, twoe) = self.calc_hamiltonian_matrix_element_contributions(
            det_w,
            det_x,
            sao,
            thresh_offdiag,
            thresh_zeroov,
        )?;
        Ok(zeroe + onee + twoe)
    }
}

impl<'a, T, SC, F> OrbitMatrix<'a, T, SC> for HamiltonianAO<'a, T, SC, F>
where
    T: ComplexFloat + Lapack + ScalarOperand + FromPrimitive,
    <T as ComplexFloat>::Real: LowerExp,
    SC: StructureConstraint + Clone + Display + PartialEq,
    SlaterDeterminant<'a, T, SC>: SymmetryTransformable,
    F: Fn(&Array2<T>) -> Result<(Array2<T>, Array2<T>), anyhow::Error> + Clone,
{
    fn calc_matrix_element(
        &self,
        det_w: &SlaterDeterminant<T, SC>,
        det_x: &SlaterDeterminant<T, SC>,
        sao: &ArrayView2<T>,
        thresh_offdiag: <T as ComplexFloat>::Real,
        thresh_zeroov: <T as ComplexFloat>::Real,
    ) -> Result<T, anyhow::Error> {
        (&self).calc_matrix_element(det_w, det_x, sao, thresh_offdiag, thresh_zeroov)
    }
}
