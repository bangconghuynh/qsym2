//! Individual multi-determinant wavefunctions for configuration interaction.

use std::collections::HashSet;
use std::fmt::{self, LowerExp};
use std::hash::Hash;
use std::marker::PhantomData;

use anyhow::{ensure, format_err};
use derive_builder::Builder;
use itertools::Itertools;
use log;
use ndarray::{Array1, Array2, ArrayView2, ArrayView4, Axis, Ix0, Ix2, ScalarOperand};
use ndarray_einsum::*;
use ndarray_linalg::types::Lapack;
use num_complex::ComplexFloat;
use rayon::prelude::*;

use crate::analysis::Overlap;
use crate::angmom::spinor_rotation_3d::StructureConstraint;
use crate::group::GroupProperties;
use crate::symmetry::symmetry_group::SymmetryGroupProperties;
use crate::symmetry::symmetry_transformation::SymmetryTransformable;
use crate::target::determinant::SlaterDeterminant;
use crate::target::noci::backend::matelem::OrbitMatrix;
use crate::target::noci::backend::matelem::hamiltonian::HamiltonianAO;
use crate::target::noci::backend::matelem::overlap::OverlapAO;
use crate::target::noci::backend::nonortho::{calc_lowdin_pairing, calc_transition_density_matrix};
use crate::target::noci::basis::{EagerBasis, OrbitBasis};

use super::basis::Basis;

#[path = "multideterminant_transformation.rs"]
pub(crate) mod multideterminant_transformation;

#[path = "multideterminant_analysis.rs"]
pub(crate) mod multideterminant_analysis;

#[cfg(test)]
#[path = "multideterminant_tests.rs"]
mod multideterminant_tests;

// ------------------
// Struct definitions
// ------------------

/// Structure to manage individual multi-determinantal wavefunctions.
#[derive(Builder, Clone)]
#[builder(build_fn(validate = "Self::validate"))]
pub struct MultiDeterminant<'a, T, B, SC>
where
    T: ComplexFloat + Lapack,
    SC: StructureConstraint + Hash + Eq + fmt::Display,
    B: Basis<Item = SlaterDeterminant<'a, T, SC>> + Clone,
{
    #[builder(setter(skip), default = "PhantomData")]
    _lifetime: PhantomData<&'a ()>,

    #[builder(setter(skip), default = "PhantomData")]
    _structure_constraint: PhantomData<SC>,

    /// A boolean indicating if inner products involving this wavefunction should be the
    /// complex-symmetric bilinear form, rather than the conventional Hermitian sesquilinear form.
    #[builder(setter(skip), default = "self.complex_symmetric_from_basis()?")]
    complex_symmetric: bool,

    /// A boolean indicating if the wavefunction has been acted on by an antiunitary operation. This
    /// is so that the correct metric can be used during overlap evaluation.
    #[builder(default = "false")]
    complex_conjugated: bool,

    /// The basis of Slater determinants in which this multi-determinantal wavefunction is defined.
    basis: B,

    /// The linear combination coefficients of the elements in the multi-orbit to give this
    /// multi-determinant wavefunction.
    coefficients: Array1<T>,

    /// The energy of this multi-determinantal wavefunction.
    #[builder(
        default = "Err(\"Multi-determinantal wavefunction energy not yet set.\".to_string())"
    )]
    energy: Result<T, String>,

    /// The threshold for comparing wavefunctions.
    threshold: <T as ComplexFloat>::Real,
}

// ----------------------
// Struct implementations
// ----------------------

impl<'a, T, B, SC> MultiDeterminantBuilder<'a, T, B, SC>
where
    T: ComplexFloat + Lapack,
    SC: StructureConstraint + Hash + Eq + Clone + fmt::Display,
    B: Basis<Item = SlaterDeterminant<'a, T, SC>> + Clone,
{
    fn validate(&self) -> Result<(), String> {
        let basis = self.basis.as_ref().ok_or("No basis found.".to_string())?;
        let coefficients = self
            .coefficients
            .as_ref()
            .ok_or("No coefficients found.".to_string())?;
        let nbasis = basis.n_items() == coefficients.len();
        if !nbasis {
            log::error!(
                "The number of coefficients does not match the number of basis determinants."
            );
        }

        let complex_symmetric = basis
            .iter()
            .map(|det_res| det_res.map(|det| det.complex_symmetric()))
            .collect::<Result<HashSet<_>, _>>()
            .map_err(|err| err.to_string())?
            .len()
            == 1;
        if !complex_symmetric {
            log::error!("Inconsistent complex-symmetric flag across basis determinants.");
        }

        let structcons_check = basis
            .iter()
            .map(|det_res| det_res.map(|det| det.structure_constraint().clone()))
            .collect::<Result<HashSet<_>, _>>()
            .map_err(|err| err.to_string())?
            .len()
            == 1;
        if !structcons_check {
            log::error!("Inconsistent spin constraints across basis determinants.");
        }

        if nbasis && structcons_check && complex_symmetric {
            Ok(())
        } else {
            Err("Multi-determinant wavefunction validation failed.".to_string())
        }
    }

    /// Retrieves the consistent complex-symmetric flag from the basis determinants.
    fn complex_symmetric_from_basis(&self) -> Result<bool, String> {
        let basis = self.basis.as_ref().ok_or("No basis found.".to_string())?;
        let complex_symmetric_set = basis
            .iter()
            .map(|det_res| det_res.map(|det| det.complex_symmetric()))
            .collect::<Result<HashSet<_>, _>>()
            .map_err(|err| err.to_string())?;
        if complex_symmetric_set.len() == 1 {
            complex_symmetric_set
                .into_iter()
                .next()
                .ok_or("Unable to retrieve the complex-symmetric flag from the basis.".to_string())
        } else {
            Err("Inconsistent complex-symmetric flag across basis determinants.".to_string())
        }
    }
}

impl<'a, T, B, SC> MultiDeterminant<'a, T, B, SC>
where
    T: ComplexFloat + Lapack,
    SC: StructureConstraint + Hash + Eq + Clone + fmt::Display,
    B: Basis<Item = SlaterDeterminant<'a, T, SC>> + Clone,
{
    /// Returns a builder to construct a new [`MultiDeterminant`].
    pub fn builder() -> MultiDeterminantBuilder<'a, T, B, SC> {
        MultiDeterminantBuilder::default()
    }

    /// Returns the structure constraint of the multi-determinantal wavefunction.
    pub fn structure_constraint(&self) -> SC {
        self.basis
            .iter()
            .next()
            .expect("No basis determinant found.")
            .expect("No basis determinant found.")
            .structure_constraint()
            .clone()
    }
}

impl<'a, T, B, SC> MultiDeterminant<'a, T, B, SC>
where
    T: ComplexFloat + Lapack,
    SC: StructureConstraint + Hash + Eq + fmt::Display,
    B: Basis<Item = SlaterDeterminant<'a, T, SC>> + Clone,
{
    /// Returns the complex-conjugated flag of the multi-determinantal wavefunction.
    pub fn complex_conjugated(&self) -> bool {
        self.complex_conjugated
    }

    /// Returns the complex-symmetric flag of the multi-determinantal wavefunction.
    pub fn complex_symmetric(&self) -> bool {
        self.complex_symmetric
    }

    /// Returns the basis of determinants in which this multi-determinantal wavefunction is
    /// defined.
    pub fn basis(&self) -> &B {
        &self.basis
    }

    /// Returns the coefficients of the basis determinants constituting this multi-determinantal
    /// wavefunction.
    pub fn coefficients(&self) -> &Array1<T> {
        &self.coefficients
    }

    /// Returns the energy of the multi-determinantal wavefunction.
    pub fn energy(&self) -> Result<&T, &String> {
        self.energy.as_ref()
    }

    /// Returns the threshold with which multi-determinantal wavefunctions are compared.
    pub fn threshold(&self) -> <T as ComplexFloat>::Real {
        self.threshold
    }

    /// Sets the energy of this multi-determinantal wavefunction.
    pub(crate) fn set_energy(&mut self, energy: Result<T, String>) {
        self.energy = energy;
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Specific implementations for OrbitBasis
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

impl<'a, T, G, SC> MultiDeterminant<'a, T, OrbitBasis<G, SlaterDeterminant<'a, T, SC>>, SC>
where
    T: ComplexFloat + Lapack,
    G: GroupProperties + Clone,
    SC: StructureConstraint + Hash + Eq + fmt::Display + Clone,
{
    /// Converts this multi-determinant with an orbit basis into a multi-determinant with the
    /// equivalent eager basis.
    #[allow(clippy::type_complexity)]
    pub fn to_eager_basis(
        &self,
    ) -> Result<MultiDeterminant<'a, T, EagerBasis<SlaterDeterminant<'a, T, SC>>, SC>, anyhow::Error>
    {
        MultiDeterminant::<T, EagerBasis<SlaterDeterminant<'a, T, SC>>, SC>::builder()
            .complex_conjugated(self.complex_conjugated)
            .basis(self.basis.to_eager()?)
            .coefficients(self.coefficients().clone())
            .energy(self.energy.clone())
            .threshold(self.threshold)
            .build()
            .map_err(|err| format_err!(err))
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Generic implementation for all Basis
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

impl<'a, T, B, SC> MultiDeterminant<'a, T, B, SC>
where
    T: ComplexFloat + Lapack + ScalarOperand + Send + Sync,
    <T as ComplexFloat>::Real: LowerExp + fmt::Display + Sync,
    SC: StructureConstraint + Hash + Eq + Clone + fmt::Display + Sync,
    B: Basis<Item = SlaterDeterminant<'a, T, SC>> + Clone + Sync,
    SlaterDeterminant<'a, T, SC>: Send + Sync,
{
    /// Calculates the (contravariant) density matrix $`\mathbf{P}(\hat{\iota})`$ of the
    /// multi-determinantal wavefunction in the AO basis.
    ///
    /// Note that the contravariant density matrix $`\mathbf{P}(\hat{\iota})`$ needs to be converted
    /// to the mixed form $`\tilde{\mathbf{P}}(\hat{\iota})`$ given by
    /// ```math
    ///     \tilde{\mathbf{P}}(\hat{\iota}) = \mathbf{P}(\hat{\iota}) \mathbf{S}_{\mathrm{AO}}
    /// ```
    /// before being diagonalised to obtain natural orbitals and their occupation numbers.
    ///
    /// # Arguments
    ///
    /// * `sao` - The atomic-orbital overlap matrix.
    /// * `thresh_offdiag` - Threshold for determining non-zero off-diagonal elements in the
    ///   orbital overlap matrix two Slater determinants during Löwdin pairing.
    /// * `thresh_zeroov` - Threshold for identifying zero Löwdin overlaps.
    /// * `normalised_wavefunction` - Boolean indicating whether the computed density matrix needs
    ///   to be normalised by the squared norm of the wavefunction.
    pub fn density_matrix(
        &self,
        sao: &ArrayView2<T>,
        thresh_offdiag: <T as ComplexFloat>::Real,
        thresh_zeroov: <T as ComplexFloat>::Real,
        normalised_wavefunction: bool,
    ) -> Result<Array2<T>, anyhow::Error> {
        let nao = sao.nrows();
        let dets = self.basis().iter().collect::<Result<Vec<_>, _>>()?;
        let sqnorm_denmat_res = dets.iter()
            .zip(self.coefficients().iter())
            .cartesian_product(dets.iter().zip(self.coefficients().iter()))
            .par_bridge()
            .fold(
                || Ok((T::zero(), Array2::<T>::zeros((nao, nao)))),
                |acc_res, ((det_w, c_w), (det_x, c_x))| {
                    ensure!(
                        det_w.structure_constraint() == det_x.structure_constraint(),
                        "Inconsistent spin constraints: {} != {}.",
                        det_w.structure_constraint(),
                        det_x.structure_constraint(),
                    );

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
                            let ne_w = occw_indices.len();
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
                            let ne_x = occx_indices.len();
                            ensure!(ne_w == ne_x, "Inconsistent number of electrons: {ne_w} != {ne_x}.");
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
                    let den_wx = calc_transition_density_matrix(&lowdin_paired_coefficientss, &self.structure_constraint())?;
                    let ov_wx = lowdin_paired_coefficientss
                        .iter()
                        .flat_map(|lpc| lpc.lowdin_overlaps().iter())
                        .fold(T::one(), |acc, ov| acc * *ov);

                    let c_w = if self.complex_conjugated() {
                        if complex_symmetric { c_w.conj() } else { *c_w }
                    } else if complex_symmetric { *c_w } else { c_w.conj() };
                    let c_x = if self.complex_conjugated() {
                        c_x.conj()
                    } else {
                        *c_x
                    };
                    let den_wx = if self.complex_conjugated() {
                        den_wx.mapv(|v| v.conj())
                    } else {
                        den_wx
                    };
                    acc_res.map(|(sqnorm_acc, denmat_acc)| (sqnorm_acc + ov_wx * c_w * c_x, denmat_acc + den_wx * c_w * c_x))
                },
            )
            .reduce(
                || Ok((T::zero(), Array2::<T>::zeros((nao, nao)))),
                |sqnorm_denmat_res_a: Result<(T, Array2<T>), anyhow::Error>, sqnorm_denmat_res_b: Result<(T, Array2<T>), anyhow::Error>| {
                    sqnorm_denmat_res_a.and_then(|(sqnorm_acc, denmat_acc)| sqnorm_denmat_res_b.map(|(sqnorm, denmat)| {
                        (
                            sqnorm_acc + sqnorm,
                            denmat_acc + denmat
                        )
                    }))
                }
            );
        sqnorm_denmat_res.map(|(sqnorm, denmat)| {
            if normalised_wavefunction {
                denmat / sqnorm
            } else {
                denmat
            }
        })
    }
}

impl<'a, T, G, SC> MultiDeterminant<'a, T, OrbitBasis<G, SlaterDeterminant<'a, T, SC>>, SC>
where
    G: SymmetryGroupProperties + Clone,
    T: ComplexFloat + Lapack + ScalarOperand + Send + Sync,
    <T as ComplexFloat>::Real: LowerExp + fmt::Display + Sync,
    SC: StructureConstraint + Hash + Eq + Clone + fmt::Display + Sync,
    SlaterDeterminant<'a, T, SC>: SymmetryTransformable + Send + Sync,
    for<'b> SlaterDeterminant<'b, T, SC>: Overlap<T, Ix2>,
{
    /// Calculates the energy of the multi-determinantal wavefunction in the orbit basis.
    ///
    /// # Arguments
    ///
    /// * `sao` - The atomic-orbital overlap matrix.
    /// * `enuc` - The nuclear repulsion energy.
    /// * `onee` - The one-electron integral matrix.
    /// * `twoe` - The optional two-electron integral tensor in physicists' ordering. Either this or
    ///   `get_jk` is required for the energy computation.
    /// * `get_jk` - The optional function callable on a density matrix $`\mathbf{D}`$ to calculate
    ///   the corresponding $`\mathbf{J}`$ and $`\mathbf{K}`$ matrices by contracting with
    ///   appropriate two-electron integrals calculated on-the-fly. Either this or `twoe` is
    ///   required for the energy computation.
    /// * `thresh_offdiag` - Threshold for determining non-zero off-diagonal elements in the
    ///   orbital overlap matrix two Slater determinants during Löwdin pairing.
    /// * `thresh_zeroov` - Threshold for identifying zero Löwdin overlaps.
    /// * `use_cayley_table` - Boolean indicating whether group closure is used in the computation
    ///   the Hamiltonian and overlap matrices between the elements in the orbit basis.
    ///
    /// # Returns
    ///
    /// The computed energy.
    #[allow(clippy::too_many_arguments)]
    pub fn compute_energy<F>(
        &self,
        sao: ArrayView2<'a, T>,
        enuc: T,
        onee: ArrayView2<'a, T>,
        twoe: Option<ArrayView4<'a, T>>,
        get_jk: Option<F>,
        thresh_offdiag: <T as ComplexFloat>::Real,
        thresh_zeroov: <T as ComplexFloat>::Real,
        use_cayley_table: bool,
    ) -> Result<T, anyhow::Error>
    where
        F: Fn(&Array2<T>) -> Result<(Array2<T>, Array2<T>), anyhow::Error> + Clone + Sync,
    {
        let basis = self.basis();
        let overlap_ao = OverlapAO::<T, SC>::builder()
            .sao(sao)
            .build()
            .map_err(|err| format_err!(err))?;
        let hamiltonian_ao = HamiltonianAO::<T, SC, _>::builder()
            .enuc(enuc)
            .onee(onee)
            .twoe(twoe)
            .get_jk(get_jk)
            .build()
            .map_err(|err| format_err!(err))?;
        let smat = overlap_ao.calc_orbit_matrix(
            basis,
            use_cayley_table,
            &sao,
            thresh_offdiag,
            thresh_zeroov,
        )?;
        let hmat = hamiltonian_ao.calc_orbit_matrix(
            basis,
            use_cayley_table,
            &sao,
            thresh_offdiag,
            thresh_zeroov,
        )?;
        let sq_norm = einsum(
            "ij,i,j->",
            &[
                &smat.view(),
                &self.coefficients.mapv(ComplexFloat::conj).view(),
                &self.coefficients.view(),
            ],
        )
        .map_err(|err| format_err!(err))?
        .into_dimensionality::<Ix0>()
        .map_err(|err| format_err!(err))?
        .into_scalar();
        let psi_h_psi = einsum(
            "ij,i,j->",
            &[
                &hmat.view(),
                &self.coefficients.mapv(ComplexFloat::conj).view(),
                &self.coefficients.view(),
            ],
        )
        .map_err(|err| format_err!(err))?
        .into_dimensionality::<Ix0>()
        .map_err(|err| format_err!(err))?
        .into_scalar();
        if ComplexFloat::abs(sq_norm) >= thresh_zeroov {
            Ok(psi_h_psi / sq_norm)
        } else {
            Err(format_err!(
                "The multi-determinantal wavefunction has near-zero squared norm: |{sq_norm:+.3e}| < {thresh_zeroov:.3e}."
            ))
        }
    }
}

// -----
// Debug
// -----
impl<'a, T, B, SC> fmt::Debug for MultiDeterminant<'a, T, B, SC>
where
    T: ComplexFloat + Lapack,
    SC: StructureConstraint + Hash + Eq + fmt::Display,
    B: Basis<Item = SlaterDeterminant<'a, T, SC>> + Clone,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "MultiDeterminant over {} basis Slater determinants",
            self.coefficients.len(),
        )?;
        Ok(())
    }
}

// -------
// Display
// -------
impl<'a, T, B, SC> fmt::Display for MultiDeterminant<'a, T, B, SC>
where
    T: ComplexFloat + Lapack,
    SC: StructureConstraint + Hash + Eq + fmt::Display,
    B: Basis<Item = SlaterDeterminant<'a, T, SC>> + Clone,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "MultiDeterminant over {} basis Slater determinants",
            self.coefficients.len(),
        )?;
        Ok(())
    }
}
