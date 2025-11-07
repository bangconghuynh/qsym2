//! Collections of multi-determinant wavefunctions for non-orthogonal configuration interaction.

use std::collections::HashSet;
use std::fmt::{self, LowerExp};
use std::hash::Hash;
use std::marker::PhantomData;

use anyhow::{ensure, format_err};
use derive_builder::Builder;
use itertools::Itertools;
use log;
use ndarray::{Array1, Array2, Array3, ArrayView2, Axis, Ix1, Ix3, ScalarOperand, ShapeBuilder};
use ndarray_einsum::einsum;
use ndarray_linalg::types::Lapack;
use num_complex::ComplexFloat;

use crate::angmom::spinor_rotation_3d::StructureConstraint;
use crate::group::GroupProperties;
use crate::target::determinant::SlaterDeterminant;
use crate::target::noci::backend::nonortho::{calc_lowdin_pairing, calc_transition_density_matrix};
use crate::target::noci::basis::{EagerBasis, OrbitBasis};
use crate::target::noci::multideterminant::MultiDeterminant;

use super::basis::Basis;

#[cfg(test)]
#[path = "multideterminants_tests.rs"]
mod multideterminants_tests;

// ------------------
// Struct definitions
// ------------------

/// Structure to manage collections of multi-determinantal wavefunctions that share the same basis
/// but have different linear combination coefficients.
#[derive(Builder, Clone)]
#[builder(build_fn(validate = "Self::validate"))]
pub struct MultiDeterminants<'a, T, B, SC>
where
    T: ComplexFloat + Lapack,
    SC: StructureConstraint + Hash + Eq + fmt::Display,
    B: Basis<SlaterDeterminant<'a, T, SC>> + Clone,
{
    #[builder(setter(skip), default = "PhantomData")]
    _lifetime: PhantomData<&'a ()>,

    #[builder(setter(skip), default = "PhantomData")]
    _structure_constraint: PhantomData<SC>,

    /// A boolean indicating if inner products involving the wavefunctions in this collection should
    /// be the complex-symmetric bilinear form, rather than the conventional Hermitian sesquilinear
    /// form.
    #[builder(setter(skip), default = "self.complex_symmetric_from_basis()?")]
    complex_symmetric: bool,

    /// A boolean indicating if the wavefunctions in this collection have been acted on by an
    /// antiunitary operation. This is so that the correct metric can be used during overlap
    /// evaluation.
    #[builder(default = "false")]
    complex_conjugated: bool,

    /// The basis of Slater determinants in which the multi-determinantal wavefunctions in this
    /// collection are defined.
    basis: B,

    /// The linear combination coefficients of the elements in the multi-orbit to give the
    /// multi-determinantal wavefunctions in this collection. Each column corresponds to one
    /// multi-determinantal wavefunction.
    coefficients: Array2<T>,

    /// The energies of the multi-determinantal wavefunctions in this collection.
    #[builder(
        default = "Err(\"Multi-determinantal wavefunction energies not yet set.\".to_string())"
    )]
    energies: Result<Array1<T>, String>,

    /// The threshold for comparing wavefunctions.
    threshold: <T as ComplexFloat>::Real,
}

// ----------------------
// Struct implementations
// ----------------------

impl<'a, T, B, SC> MultiDeterminantsBuilder<'a, T, B, SC>
where
    T: ComplexFloat + Lapack,
    SC: StructureConstraint + Hash + Eq + Clone + fmt::Display,
    B: Basis<SlaterDeterminant<'a, T, SC>> + Clone,
{
    fn validate(&self) -> Result<(), String> {
        let basis = self.basis.as_ref().ok_or("No basis found.".to_string())?;
        let coefficients = self
            .coefficients
            .as_ref()
            .ok_or("No coefficients found.".to_string())?;
        let nbasis = basis.n_items() == coefficients.nrows();
        if !nbasis {
            log::error!(
                "The number of coefficient rows does not match the number of basis determinants."
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
            Err("Multi-determinantal wavefunction collection validation failed.".to_string())
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

impl<'a, T, B, SC> MultiDeterminants<'a, T, B, SC>
where
    T: ComplexFloat + Lapack,
    SC: StructureConstraint + Hash + Eq + Clone + fmt::Display,
    B: Basis<SlaterDeterminant<'a, T, SC>> + Clone,
{
    /// Returns a builder to construct a new [`MultiDeterminants`].
    pub fn builder() -> MultiDeterminantsBuilder<'a, T, B, SC> {
        MultiDeterminantsBuilder::default()
    }

    /// Constructs a collection of multi-determinantal wavefunctions from a sequence of individual
    /// multi-determinantal wavefunctions.
    ///
    /// No checks are performed to ensure that the single-determinantal bases are consistent across
    /// all supplied multi-determinantal wavefunctions. Only the basis of the first
    /// multi-determinantal wavefunction will be used.
    ///
    /// # Arguments
    ///
    /// * `mtds` - A sequence of individual multi-determinantal wavefunctions.
    pub fn from_multideterminant_vec(
        mtds: &[&MultiDeterminant<'a, T, B, SC>],
    ) -> Result<MultiDeterminants<'a, T, B, SC>, anyhow::Error> {
        log::warn!(
            "Using basis from the first multi-determinantal wavefunction as the common basis for the collection of multi-determinantal wavefunctions..."
        );
        let nmultidets = mtds.len();
        let dims_set = mtds
            .iter()
            .map(|mtd| mtd.basis().n_items())
            .collect::<HashSet<_>>();
        let dim = if dims_set.len() == 1 {
            dims_set
                .into_iter()
                .next()
                .ok_or_else(|| format_err!("Unable to obtain the unique basis size."))
        } else {
            Err(format_err!(
                "Inconsistent basis sizes across the supplied multi-determinantal wavefunctions."
            ))
        }?;
        let coefficients = Array2::from_shape_vec(
            (dim, nmultidets).f(),
            mtds.iter()
                .flat_map(|mtd| mtd.coefficients())
                .cloned()
                .collect::<Vec<_>>(),
        )
        .map_err(|err| format_err!(err))?;

        let (basis, threshold) = mtds
            .first()
            .map(|mtd| (mtd.basis().clone(), mtd.threshold()))
            .ok_or_else(|| {
                format_err!("Unable to access the first multi-determinantal wavefunction.")
            })?;

        MultiDeterminants::builder()
            .basis(basis)
            .coefficients(coefficients)
            .threshold(threshold)
            .build()
            .map_err(|err| format_err!(err))
    }

    /// Returns the structure constraint of the multi-determinantal wavefunctions in the collection.
    pub fn structure_constraint(&self) -> SC {
        self.basis
            .iter()
            .next()
            .expect("No basis determinant found.")
            .expect("No basis determinant found.")
            .structure_constraint()
            .clone()
    }

    /// Returns an iterator over the multi-determinantal wavefunctions in this collection.
    pub fn iter(&self) -> impl Iterator {
        let energies = self
            .energies
            .as_ref()
            .map(|energies| energies.mapv(|v| Ok(v)))
            .unwrap_or(Array1::from_elem(
                self.coefficients.ncols(),
                Err("Multi-determinantal energy not available.".to_string()),
            ));
        self.coefficients
            .columns()
            .into_iter()
            .zip(energies.into_iter())
            .map(|(c, e)| {
                MultiDeterminant::builder()
                    .complex_conjugated(self.complex_conjugated)
                    .basis(self.basis().clone())
                    .coefficients(c.to_owned())
                    .energy(e)
                    .threshold(self.threshold)
                    .build()
                    .map_err(|err| format_err!(err))
            })
    }
}

impl<'a, T, B, SC> MultiDeterminants<'a, T, B, SC>
where
    T: ComplexFloat + Lapack,
    SC: StructureConstraint + Hash + Eq + fmt::Display,
    B: Basis<SlaterDeterminant<'a, T, SC>> + Clone,
{
    /// Returns the complex-conjugated flag of the multi-determinantal wavefunctions in the
    /// collection.
    pub fn complex_conjugated(&self) -> bool {
        self.complex_conjugated
    }

    /// Returns the complex-symmetric flag of the multi-determinantal wavefunctions in the
    /// collection.
    pub fn complex_symmetric(&self) -> bool {
        self.complex_symmetric
    }

    /// Returns the basis of determinants in which the multi-determinantal wavefunctions in this
    /// collection are defined.
    pub fn basis(&self) -> &B {
        &self.basis
    }

    /// Returns the coefficients of the basis determinants constituting the multi-determinantal
    /// wavefunctions in this collection.
    pub fn coefficients(&self) -> &Array2<T> {
        &self.coefficients
    }

    /// Returns the energies of the multi-determinantal wavefunctions in this collection.
    pub fn energies(&self) -> Result<&Array1<T>, &String> {
        self.energies.as_ref()
    }

    /// Returns the threshold with which multi-determinantal wavefunctions are compared.
    pub fn threshold(&self) -> <T as ComplexFloat>::Real {
        self.threshold
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Specific implementations for OrbitBasis
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

impl<'a, T, G, SC> MultiDeterminants<'a, T, OrbitBasis<'a, G, SlaterDeterminant<'a, T, SC>>, SC>
where
    T: ComplexFloat + Lapack,
    G: GroupProperties + Clone,
    SC: StructureConstraint + Hash + Eq + fmt::Display + Clone,
{
    /// Converts this multi-determinantal wavefunction collection with an orbit basis into one with
    /// the equivalent eager basis.
    pub fn to_eager_basis(
        &self,
    ) -> Result<MultiDeterminants<'a, T, EagerBasis<SlaterDeterminant<'a, T, SC>>, SC>, anyhow::Error>
    {
        MultiDeterminants::<T, EagerBasis<SlaterDeterminant<'a, T, SC>>, SC>::builder()
            .complex_conjugated(self.complex_conjugated)
            .basis(self.basis.to_eager()?)
            .coefficients(self.coefficients().clone())
            .energies(self.energies.clone())
            .threshold(self.threshold)
            .build()
            .map_err(|err| format_err!(err))
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Generic implementation for all Basis
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

impl<'a, T, B, SC> MultiDeterminants<'a, T, B, SC>
where
    T: ComplexFloat + Lapack + ScalarOperand,
    <T as ComplexFloat>::Real: LowerExp + fmt::Display,
    SC: StructureConstraint + Hash + Eq + Clone + fmt::Display,
    B: Basis<SlaterDeterminant<'a, T, SC>> + Clone,
{
    /// Calculates the (contravariant) density matrices $`\mathbf{P}_m(\hat{\iota})`$ of all
    /// multi-determinantal wavefunctions in this collection in the AO basis.
    ///
    /// Note that each contravariant density matrix $`\mathbf{P}_m(\hat{\iota})`$ needs to be
    /// converted to the mixed form $`\tilde{\mathbf{P}}(\hat{\iota})`$ given by
    /// ```math
    ///     \tilde{\mathbf{P}}(\hat{\iota}) = \mathbf{P}(\hat{\iota}) \mathbf{S}_{\mathrm{AO}}
    /// ```
    /// before being diagonalised to obtain natural orbitals and their occupation numbers.
    ///
    /// # Arguments
    ///
    /// * `sao` - The atomic-orbital overlap matrix.
    /// * `thresh_offdiag` - Threshold for determining non-zero off-diagonal elements in the
    /// orbital overlap matrix two Slater determinants during Löwdin pairing.
    /// * `thresh_zeroov` - Threshold for identifying zero Löwdin overlaps.
    ///
    /// # Returns
    ///
    /// Returns a three-dimensional array $`P`$ containing the density matrices of the
    /// multi-determinantal wavefunctions in this collection. The array is indexed $`P_{mij}`$
    /// where $`m`$ is the index for the multi-determinantal wavefunctions in this collection.
    pub fn density_matrices(
        &self,
        sao: &ArrayView2<T>,
        thresh_offdiag: <T as ComplexFloat>::Real,
        thresh_zeroov: <T as ComplexFloat>::Real,
        normalised_wavefunctions: bool,
    ) -> Result<Array3<T>, anyhow::Error> {
        let nao = sao.nrows();
        let dets = self.basis().iter().collect::<Result<Vec<_>, _>>()?;
        let nmultidets = self.coefficients.ncols();
        let sqnorms_denmats_res = dets.iter()
            .zip(self.coefficients().rows())
            .cartesian_product(dets.iter().zip(self.coefficients().rows()))
            .fold(
                Ok((Array1::<T>::zeros(nmultidets), Array3::<T>::zeros((nmultidets, nao, nao)))),
                |acc_res, ((det_w, c_wm), (det_x, c_xm))| {
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
                        .map(|lpc| lpc.lowdin_overlaps().iter())
                        .flatten()
                        .fold(T::one(), |acc, ov| acc * *ov);

                    let c_wm = if self.complex_conjugated() {
                        if complex_symmetric { c_wm.map(|v| v.conj()) } else { c_wm.to_owned() }
                    } else {
                        if complex_symmetric { c_wm.to_owned() } else { c_wm.map(|v| v.conj()) }
                    };
                    let c_xm = if self.complex_conjugated() {
                        c_xm.map(|v| v.conj())
                    } else {
                        c_xm.to_owned()
                    };
                    let den_wx = if self.complex_conjugated() {
                        den_wx.mapv(|v| v.conj())
                    } else {
                        den_wx
                    };
                    acc_res.and_then(|(sqnorm_acc, denmat_acc)| {
                        let ov_wx_m = einsum("m,m->m", &[&c_wm.view(), &c_xm.view()])
                            .map_err(|err| format_err!(err))?
                            .into_dimensionality::<Ix1>()
                            .map_err(|err| format_err!(err))?
                            .mapv(|v| v * ov_wx);
                        let denmat_wx_mij = einsum("ij,m,m->mij", &[&den_wx.view(), &c_wm.view(), &c_xm.view()])
                            .map_err(|err| format_err!(err))?
                            .into_dimensionality::<Ix3>()
                            .map_err(|err| format_err!(err))?;
                        Ok((sqnorm_acc + ov_wx_m, denmat_acc + denmat_wx_mij))
                    })
                },
            );
        sqnorms_denmats_res.and_then(|(sqnorms, denmats)| {
            if normalised_wavefunctions {
                let sqnorms_inv = sqnorms.mapv(|v| T::one() / v);
                einsum("m,mij->mij", &[&sqnorms_inv.view(), &denmats.view()])
                    .map_err(|err| format_err!(err))?
                    .into_dimensionality::<Ix3>()
                    .map_err(|err| format_err!(err))
            } else {
                Ok(denmats)
            }
        })
    }
}

// -----
// Debug
// -----
impl<'a, T, B, SC> fmt::Debug for MultiDeterminants<'a, T, B, SC>
where
    T: ComplexFloat + Lapack,
    SC: StructureConstraint + Hash + Eq + fmt::Display,
    B: Basis<SlaterDeterminant<'a, T, SC>> + Clone,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "MultiDeterminant collection over {} basis Slater determinants",
            self.coefficients.len(),
        )?;
        Ok(())
    }
}

// -------
// Display
// -------
impl<'a, T, B, SC> fmt::Display for MultiDeterminants<'a, T, B, SC>
where
    T: ComplexFloat + Lapack,
    SC: StructureConstraint + Hash + Eq + fmt::Display,
    B: Basis<SlaterDeterminant<'a, T, SC>> + Clone,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "MultiDeterminant collection over {} basis Slater determinants",
            self.coefficients.len(),
        )?;
        Ok(())
    }
}
