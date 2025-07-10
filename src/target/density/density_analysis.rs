//! Implementation of symmetry analysis for electron densities.

use std::fmt;
use std::ops::Mul;

use anyhow::{self, ensure, format_err, Context};
use approx;
use derive_builder::Builder;
use itertools::Itertools;
use log;
use ndarray::{Array1, Array2, Array4, Axis, Ix4};
use ndarray_einsum::*;
use ndarray_linalg::{
    eig::Eig,
    eigh::Eigh,
    norm::Norm,
    types::{Lapack, Scalar},
    UPLO,
};
use num_complex::{Complex, ComplexFloat};
use num_traits::{Float, ToPrimitive, Zero};

use crate::analysis::{
    fn_calc_xmat_complex, fn_calc_xmat_real, EigenvalueComparisonMode, Orbit, OrbitIterator,
    Overlap, RepAnalysis,
};
use crate::auxiliary::misc::complex_modified_gram_schmidt;
use crate::chartab::chartab_group::CharacterProperties;
use crate::chartab::{DecompositionError, SubspaceDecomposable};
use crate::symmetry::symmetry_element::symmetry_operation::SpecialSymmetryTransformation;
use crate::symmetry::symmetry_group::SymmetryGroupProperties;
use crate::symmetry::symmetry_transformation::{SymmetryTransformable, SymmetryTransformationKind};
use crate::target::density::Density;

// =======
// Overlap
// =======

impl<'a, T> Overlap<T, Ix4> for Density<'a, T>
where
    T: Lapack
        + ComplexFloat<Real = <T as Scalar>::Real>
        + fmt::Debug
        + Mul<<T as ComplexFloat>::Real, Output = T>,
    <T as ComplexFloat>::Real: fmt::Debug
        + approx::RelativeEq<<T as ComplexFloat>::Real>
        + approx::AbsDiffEq<Epsilon = <T as Scalar>::Real>,
{
    fn complex_symmetric(&self) -> bool {
        self.complex_symmetric
    }

    /// Computes the overlap between two densities.
    ///
    /// When one or both of the densities have been acted on by an antiunitary operation, the
    /// correct Hermitian or complex-symmetric metric will be chosen in the evalulation of the
    /// overlap.
    ///
    /// # Panics
    ///
    /// Panics if `self` and `other` have mismatched spin constraints.
    fn overlap(
        &self,
        other: &Self,
        metric: Option<&Array4<T>>,
        metric_h: Option<&Array4<T>>,
    ) -> Result<T, anyhow::Error> {
        ensure!(
            self.density_matrix.shape() == other.density_matrix.shape(),
            "Inconsistent shapes of density matrices between `self` and `other`."
        );
        ensure!(
            self.bao == other.bao,
            "Inconsistent basis angular order between `self` and `other`."
        );
        let dennorm = self.density_matrix().norm_l2();
        ensure!(
            approx::abs_diff_ne!(
                dennorm,
                <T as ndarray_linalg::Scalar>::Real::zero(),
                epsilon = self.threshold()
            ),
            "Zero density (density matrix has Frobenius norm {dennorm:.3e})"
        );

        let sao_4c = metric
            .ok_or_else(|| format_err!("No atomic-orbital four-centre overlap tensor found for density overlap calculation."))?;
        let sao_4c_h = metric_h.unwrap_or(sao_4c);

        if self.complex_symmetric() {
            match (self.complex_conjugated, other.complex_conjugated) {
                (false, false) => einsum(
                    "ijkl,ji,lk->",
                    &[
                        &sao_4c_h.view(),
                        &self.density_matrix().view(),
                        &other.density_matrix().view(),
                    ],
                )
                .map_err(|err| format_err!(err))?
                .into_iter()
                .next()
                .ok_or(format_err!("Unable to extract the density overlap scalar.")),
                (true, false) => einsum(
                    "ijkl,ji,lk->",
                    &[
                        &sao_4c.view(),
                        &self.density_matrix().view(),
                        &other.density_matrix().view(),
                    ],
                )
                .map_err(|err| format_err!(err))?
                .into_iter()
                .next()
                .ok_or(format_err!("Unable to extract the density overlap scalar.")),
                (false, true) => einsum(
                    "ijkl,ji,lk->",
                    &[
                        &sao_4c.view(),
                        &other.density_matrix().view(),
                        &self.density_matrix().view(),
                    ],
                )
                .map_err(|err| format_err!(err))?
                .into_iter()
                .next()
                .ok_or(format_err!("Unable to extract the density overlap scalar.")),
                (true, true) => einsum(
                    "klij,ji,lk->",
                    &[
                        &sao_4c_h.view(),
                        &self.density_matrix().view(),
                        &other.density_matrix().view(),
                    ],
                )
                .map_err(|err| format_err!(err))?
                .into_iter()
                .next()
                .ok_or(format_err!("Unable to extract the density overlap scalar.")),
            }
        } else {
            match (self.complex_conjugated, other.complex_conjugated) {
                (false, false) => einsum(
                    "ijkl,ji,lk->",
                    &[
                        &sao_4c.view(),
                        &self.density_matrix().mapv(|x| x.conj()).view(),
                        &other.density_matrix().view(),
                    ],
                )
                .map_err(|err| format_err!(err))?
                .into_iter()
                .next()
                .ok_or(format_err!("Unable to extract the density overlap scalar.")),
                (true, false) => einsum(
                    "ijkl,ji,lk->",
                    &[
                        &sao_4c_h.view(),
                        &self.density_matrix().mapv(|x| x.conj()).view(),
                        &other.density_matrix().view(),
                    ],
                )
                .map_err(|err| format_err!(err))?
                .into_iter()
                .next()
                .ok_or(format_err!("Unable to extract the density overlap scalar.")),
                (false, true) => einsum(
                    "ijkl,ji,lk->",
                    &[
                        &sao_4c_h.view(),
                        &other.density_matrix().mapv(|x| x.conj()).view(),
                        &self.density_matrix().view(),
                    ],
                )
                .map_err(|err| format_err!(err))?
                .into_iter()
                .next()
                .ok_or(format_err!("Unable to extract the density overlap scalar."))
                .map(|x| x.conj()),
                (true, true) => einsum(
                    "klij,ji,lk->",
                    &[
                        &sao_4c.view(),
                        &self.density_matrix().mapv(|x| x.conj()).view(),
                        &other.density_matrix().view(),
                    ],
                )
                .map_err(|err| format_err!(err))?
                .into_iter()
                .next()
                .ok_or(format_err!("Unable to extract the density overlap scalar.")),
            }
        }
    }

    /// Returns the mathematical definition of the overlap between two densities.
    fn overlap_definition(&self) -> String {
        let k = if self.complex_symmetric() { "κ " } else { "" };
        format!("⟨{k}ρ_1|ρ_2⟩ = ∫ [{k}ρ_1(r)]* ρ_2(r) dr")
    }
}

// ====================
// DensitySymmetryOrbit
// ====================

// -----------------
// Struct definition
// -----------------

/// Structure to manage symmetry orbits (*i.e.* orbits generated by symmetry groups) of
/// densities.
#[derive(Builder, Clone)]
pub struct DensitySymmetryOrbit<'a, G, T>
where
    G: SymmetryGroupProperties,
    T: ComplexFloat + fmt::Debug + Lapack,
    Density<'a, T>: SymmetryTransformable,
{
    /// The generating symmetry group.
    group: &'a G,

    /// The origin density of the orbit.
    origin: &'a Density<'a, T>,

    /// The threshold for determining zero eigenvalues in the orbit overlap matrix.
    pub(crate) linear_independence_threshold: <T as ComplexFloat>::Real,

    /// The threshold for determining if calculated multiplicities in representation analysis are
    /// integral.
    integrality_threshold: <T as ComplexFloat>::Real,

    /// The kind of transformation determining the way the symmetry operations in `group` act on
    /// [`Self::origin`].
    symmetry_transformation_kind: SymmetryTransformationKind,

    /// The overlap matrix between the symmetry-equivalent densities in the orbit.
    #[builder(setter(skip), default = "None")]
    smat: Option<Array2<T>>,

    /// The eigenvalues of the overlap matrix between the symmetry-equivalent densities in
    /// the orbit.
    #[builder(setter(skip), default = "None")]
    pub(crate) smat_eigvals: Option<Array1<T>>,

    /// The $`\mathbf{X}`$ matrix for the overlap matrix between the symmetry-equivalent densities
    /// in the orbit.
    ///
    /// See [`RepAnalysis::xmat`] for further information.
    #[builder(setter(skip), default = "None")]
    xmat: Option<Array2<T>>,

    /// An enumerated type specifying the comparison mode for filtering out orbit overlap
    /// eigenvalues.
    eigenvalue_comparison_mode: EigenvalueComparisonMode,
}

// ----------------------------
// Struct method implementation
// ----------------------------

impl<'a, G, T> DensitySymmetryOrbit<'a, G, T>
where
    G: SymmetryGroupProperties + Clone,
    T: ComplexFloat + fmt::Debug + Lapack,
    Density<'a, T>: SymmetryTransformable,
{
    /// Returns a builder for constructing a new density symmetry orbit.
    pub fn builder() -> DensitySymmetryOrbitBuilder<'a, G, T> {
        DensitySymmetryOrbitBuilder::default()
    }
}

impl<'a, G> DensitySymmetryOrbit<'a, G, f64>
where
    G: SymmetryGroupProperties,
{
    fn_calc_xmat_real!(
        /// Calculates the $`\mathbf{X}`$ matrix for real and symmetric overlap matrix
        /// $`\mathbf{S}`$ between the symmetry-equivalent densities in the orbit.
        ///
        /// The resulting $`\mathbf{X}`$ is stored in the orbit.
        ///
        /// # Arguments
        ///
        /// * `preserves_full_rank` - If `true`, when $`\mathbf{S}`$ is already of full rank, then
        /// $`\mathbf{X}`$ is set to be the identity matrix to avoid mixing the orbit densities. If
        /// `false`, $`\mathbf{X}`$ also orthogonalises $`\mathbf{S}`$ even when it is already of
        /// full rank.
        pub calc_xmat
    );
}

impl<'a, G, T> DensitySymmetryOrbit<'a, G, Complex<T>>
where
    G: SymmetryGroupProperties,
    T: Float + Scalar<Complex = Complex<T>>,
    Complex<T>: ComplexFloat<Real = T> + Scalar<Real = T, Complex = Complex<T>> + Lapack,
    Density<'a, Complex<T>>: SymmetryTransformable + Overlap<Complex<T>, Ix4>,
{
    fn_calc_xmat_complex!(
        /// Calculates the $`\mathbf{X}`$ matrix for complex and symmetric or Hermitian overlap
        /// matrix $`\mathbf{S}`$ between the symmetry-equivalent densities in the orbit.
        ///
        /// The resulting $`\mathbf{X}`$ is stored in the orbit.
        ///
        /// # Arguments
        ///
        /// * `preserves_full_rank` - If `true`, when $`\mathbf{S}`$ is already of full rank, then
        /// $`\mathbf{X}`$ is set to be the identity matrix to avoid mixing the orbit densities. If
        /// `false`, $`\mathbf{X}`$ also orthogonalises $`\mathbf{S}`$ even when it is already of
        /// full rank.
        pub calc_xmat
    );
}

// ---------------------
// Trait implementations
// ---------------------

// ~~~~~
// Orbit
// ~~~~~

impl<'a, G, T> Orbit<G, Density<'a, T>> for DensitySymmetryOrbit<'a, G, T>
where
    G: SymmetryGroupProperties,
    T: ComplexFloat + fmt::Debug + Lapack,
    Density<'a, T>: SymmetryTransformable,
{
    type OrbitIter = OrbitIterator<'a, G, Density<'a, T>>;

    fn group(&self) -> &G {
        self.group
    }

    fn origin(&self) -> &Density<'a, T> {
        self.origin
    }

    fn iter(&self) -> Self::OrbitIter {
        OrbitIterator::new(
            self.group,
            self.origin,
            match self.symmetry_transformation_kind {
                SymmetryTransformationKind::Spatial => |op, det| {
                    det.sym_transform_spatial(op).with_context(|| {
                        format!("Unable to apply `{op}` spatially on the origin density")
                    })
                },
                SymmetryTransformationKind::SpatialWithSpinTimeReversal => |op, det| {
                    det.sym_transform_spatial_with_spintimerev(op).with_context(|| {
                        format!("Unable to apply `{op}` spatially (with spin-including time-reversal) on the origin density")
                    })
                },
                SymmetryTransformationKind::Spin => |op, det| {
                    det.sym_transform_spin(op).with_context(|| {
                        format!("Unable to apply `{op}` spin-wise on the origin density")
                    })
                },
                SymmetryTransformationKind::SpinSpatial => |op, det| {
                    det.sym_transform_spin_spatial(op).with_context(|| {
                        format!("Unable to apply `{op}` spin-spatially on the origin density")
                    })
                },
            },
        )
    }
}

// ~~~~~~~~~~~
// RepAnalysis
// ~~~~~~~~~~~

impl<'a, G, T> RepAnalysis<G, Density<'a, T>, T, Ix4> for DensitySymmetryOrbit<'a, G, T>
where
    G: SymmetryGroupProperties,
    G::CharTab: SubspaceDecomposable<T>,
    T: Lapack
        + ComplexFloat<Real = <T as Scalar>::Real>
        + fmt::Debug
        + Mul<<T as ComplexFloat>::Real, Output = T>,
    <T as ComplexFloat>::Real: fmt::Debug
        + Zero
        + From<u16>
        + ToPrimitive
        + approx::RelativeEq<<T as ComplexFloat>::Real>
        + approx::AbsDiffEq<Epsilon = <T as Scalar>::Real>,
    Density<'a, T>: SymmetryTransformable,
{
    fn set_smat(&mut self, smat: Array2<T>) {
        self.smat = Some(smat)
    }

    fn smat(&self) -> Option<&Array2<T>> {
        self.smat.as_ref()
    }

    fn xmat(&self) -> &Array2<T> {
        self.xmat
            .as_ref()
            .expect("Orbit overlap orthogonalisation matrix not found.")
    }

    fn norm_preserving_scalar_map(&self, i: usize) -> Result<fn(T) -> T, anyhow::Error> {
        if self.origin.complex_symmetric {
            Err(format_err!("`norm_preserving_scalar_map` is currently not implemented for complex-symmetric overlaps. This thus precludes the use of the Cayley table to speed up the computation of the orbit overlap matrix."))
        } else {
            if self
                .group
                .get_index(i)
                .unwrap_or_else(|| panic!("Group operation index `{i}` not found."))
                .contains_time_reversal()
            {
                Ok(ComplexFloat::conj)
            } else {
                Ok(|x| x)
            }
        }
    }

    fn integrality_threshold(&self) -> <T as ComplexFloat>::Real {
        self.integrality_threshold
    }

    fn eigenvalue_comparison_mode(&self) -> &EigenvalueComparisonMode {
        &self.eigenvalue_comparison_mode
    }

    /// Reduces the representation or corepresentation spanned by the densities in the orbit to
    /// a direct sum of the irreducible representations or corepresentations of the generating
    /// symmetry group.
    ///
    /// # Returns
    ///
    /// The decomposed result.
    ///
    /// # Errors
    ///
    /// Errors if the decomposition fails, *e.g.* because one or more calculated multiplicities
    /// are non-integral.
    fn analyse_rep(
        &self,
    ) -> Result<
        <<G as CharacterProperties>::CharTab as SubspaceDecomposable<T>>::Decomposition,
        DecompositionError,
    > {
        log::debug!("Analysing representation symmetry for a density...");
        let chis = self
            .calc_characters()
            .map_err(|err| DecompositionError(err.to_string()))?;
        log::debug!("Characters calculated.");
        let res = self.group().character_table().reduce_characters(
            &chis.iter().map(|(cc, chi)| (cc, *chi)).collect::<Vec<_>>(),
            self.integrality_threshold(),
        );
        log::debug!("Characters reduced.");
        log::debug!("Analysing representation symmetry for a density... Done.");
        res
    }
}
