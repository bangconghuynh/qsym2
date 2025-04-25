//! Implementation of symmetry analysis for Slater determinants.

use std::fmt;
use std::ops::Mul;

use anyhow::{self, ensure, format_err, Context};
use approx;
use derive_builder::Builder;
use itertools::{izip, Itertools};
use log;
use ndarray::{Array1, Array2, Axis, Ix2};
use ndarray_linalg::{
    eig::Eig,
    eigh::Eigh,
    solve::Determinant,
    types::{Lapack, Scalar},
    UPLO,
};
use num_complex::{Complex, ComplexFloat};
use num_traits::{Float, ToPrimitive, Zero};

use crate::analysis::{
    fn_calc_xmat_complex, fn_calc_xmat_real, EigenvalueComparisonMode, Orbit, OrbitIterator,
    Overlap, RepAnalysis,
};
use crate::angmom::spinor_rotation_3d::StructureConstraint;
use crate::auxiliary::misc::complex_modified_gram_schmidt;
use crate::chartab::chartab_group::CharacterProperties;
use crate::chartab::{DecompositionError, SubspaceDecomposable};
use crate::group::GroupType;
use crate::io::format::{log_subtitle, qsym2_output, QSym2Output};
use crate::symmetry::symmetry_element::symmetry_operation::SpecialSymmetryTransformation;
use crate::symmetry::symmetry_group::SymmetryGroupProperties;
use crate::symmetry::symmetry_transformation::{SymmetryTransformable, SymmetryTransformationKind};
use crate::target::determinant::SlaterDeterminant;

// =======
// Overlap
// =======

impl<'a, T, SC> Overlap<T, Ix2> for SlaterDeterminant<'a, T, SC>
where
    T: Lapack
        + ComplexFloat<Real = <T as Scalar>::Real>
        + fmt::Debug
        + Mul<<T as ComplexFloat>::Real, Output = T>,
    <T as ComplexFloat>::Real: fmt::Debug
        + approx::RelativeEq<<T as ComplexFloat>::Real>
        + approx::AbsDiffEq<Epsilon = <T as Scalar>::Real>,
    SC: StructureConstraint + Eq + fmt::Display,
{
    fn complex_symmetric(&self) -> bool {
        self.complex_symmetric
    }

    /// Computes the overlap between two Slater determinants.
    ///
    /// Determinants with fractional electrons are currently not supported.
    ///
    /// When one or both of the Slater determinants have been acted on by an antiunitary operation,
    /// the correct Hermitian or complex-symmetric metric will be chosen in the evalulation of the
    /// overlap.
    ///
    /// # Arguments
    ///
    /// * `metric` - The atomic-orbital overlap matrix with respect to the conventional sesquilinear
    /// inner product.
    /// * `metric_h` - The atomic-orbital overlap matrix with respect to the bilinear inner product.
    ///
    /// # Panics
    ///
    /// Panics if `self` and `other` have mismatched spin constraints or numbers of coefficient
    /// matrices, or if fractional occupation numbers are detected.
    fn overlap(
        &self,
        other: &Self,
        metric: Option<&Array2<T>>,
        metric_h: Option<&Array2<T>>,
    ) -> Result<T, anyhow::Error> {
        ensure!(
            self.structure_constraint == other.structure_constraint,
            "Inconsistent structure constraints between `self` and `other`."
        );
        ensure!(
            self.coefficients.len() == other.coefficients.len(),
            "Inconsistent numbers of coefficient matrices between `self` and `other`."
        );
        ensure!(
            self.bao == other.bao,
            "Inconsistent basis angular order between `self` and `other`."
        );

        let thresh = Float::sqrt(self.threshold * other.threshold);
        ensure!(self
            .occupations
            .iter()
            .chain(other.occupations.iter())
            .all(|occs| occs.iter().all(|&occ| approx::relative_eq!(
                occ,
                occ.round(),
                epsilon = thresh,
                max_relative = thresh
            ))),
            "Overlaps between determinants with fractional occupation numbers are currently not supported."
        );

        let sao = metric.ok_or_else(|| format_err!("No atomic-orbital metric found."))?;
        let sao_h = metric_h.unwrap_or(sao);

        let ov = izip!(
            &self.coefficients,
            &self.occupations,
            &other.coefficients,
            &other.occupations
        )
        .map(|(cw, occw, cx, occx)| {
            let nonzero_occ_w = occw.iter().positions(|&occ| occ > thresh).collect_vec();
            let cw_o = cw.select(Axis(1), &nonzero_occ_w);
            let nonzero_occ_x = occx.iter().positions(|&occ| occ > thresh).collect_vec();
            let cx_o = cx.select(Axis(1), &nonzero_occ_x);

            let mo_ov_mat = if self.complex_symmetric() {
                match (self.complex_conjugated, other.complex_conjugated) {
                    (false, false) => cw_o.t().dot(sao_h).dot(&cx_o),
                    (true, false) => cw_o.t().dot(sao).dot(&cx_o),
                    (false, true) => cx_o.t().dot(sao).dot(&cw_o),
                    (true, true) => cw_o.t().dot(&sao_h.t()).dot(&cx_o),
                }
            } else {
                match (self.complex_conjugated, other.complex_conjugated) {
                    (false, false) => cw_o.t().mapv(|x| x.conj()).dot(sao).dot(&cx_o),
                    (true, false) => cw_o.t().mapv(|x| x.conj()).dot(sao_h).dot(&cx_o),
                    (false, true) => cx_o
                        .t()
                        .mapv(|x| x.conj())
                        .dot(sao_h)
                        .dot(&cw_o)
                        .mapv(|x| x.conj()),
                    (true, true) => cw_o.t().mapv(|x| x.conj()).dot(&sao.t()).dot(&cx_o),
                }
            };
            mo_ov_mat
                .det()
                .expect("The determinant of the MO overlap matrix could not be found.")
        })
        .fold(T::one(), |acc, x| acc * x);

        let implicit_factor = self.structure_constraint.implicit_factor()?;
        if implicit_factor > 1 {
            let p_i32 = i32::try_from(implicit_factor)?;
            Ok(ComplexFloat::powi(ov, p_i32))
        } else {
            Ok(ov)
        }
    }

    /// Returns the mathematical definition of the overlap between two Slater determinants.
    fn overlap_definition(&self) -> String {
        let k = if self.complex_symmetric() { "κ " } else { "" };
        format!("⟨{k}Ψ_1|Ψ_2⟩ = ∫ [{k}Ψ_1(x^Ne)]* Ψ_2(x^Ne) dx^Ne")
    }
}

// ==============================
// SlaterDeterminantSymmetryOrbit
// ==============================

// -----------------
// Struct definition
// -----------------

/// Structure to manage symmetry orbits (*i.e.* orbits generated by symmetry groups) of Slater
/// determinants.
#[derive(Builder, Clone)]
pub struct SlaterDeterminantSymmetryOrbit<'a, G, T, SC>
where
    G: SymmetryGroupProperties,
    T: ComplexFloat + fmt::Debug + Lapack,
    SC: StructureConstraint + fmt::Display,
    SlaterDeterminant<'a, T, SC>: SymmetryTransformable,
{
    /// The generating symmetry group.
    group: &'a G,

    /// The origin Slater determinant of the orbit.
    origin: &'a SlaterDeterminant<'a, T, SC>,

    /// The threshold for determining zero eigenvalues in the orbit overlap matrix.
    linear_independence_threshold: <T as ComplexFloat>::Real,

    /// The threshold for determining if calculated multiplicities in representation analysis are
    /// integral.
    integrality_threshold: <T as ComplexFloat>::Real,

    /// The kind of transformation determining the way the symmetry operations in `group` act on
    /// [`Self::origin`].
    symmetry_transformation_kind: SymmetryTransformationKind,

    /// The overlap matrix between the symmetry-equivalent Slater determinants in the orbit.
    #[builder(setter(skip), default = "None")]
    smat: Option<Array2<T>>,

    /// The eigenvalues of the overlap matrix between the symmetry-equivalent Slater determinants in
    /// the orbit.
    #[builder(setter(skip), default = "None")]
    pub(crate) smat_eigvals: Option<Array1<T>>,

    /// The $`\mathbf{X}`$ matrix for the overlap matrix between the symmetry-equivalent Slater
    /// determinants in the orbit.
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

impl<'a, G, T, SC> SlaterDeterminantSymmetryOrbit<'a, G, T, SC>
where
    G: SymmetryGroupProperties + Clone,
    T: ComplexFloat + fmt::Debug + Lapack,
    SC: StructureConstraint + Clone + fmt::Display,
    SlaterDeterminant<'a, T, SC>: SymmetryTransformable,
{
    /// Returns a builder for constructing a new Slater determinant symmetry orbit.
    pub fn builder() -> SlaterDeterminantSymmetryOrbitBuilder<'a, G, T, SC> {
        SlaterDeterminantSymmetryOrbitBuilder::default()
    }
}

impl<'a, G, SC> SlaterDeterminantSymmetryOrbit<'a, G, f64, SC>
where
    G: SymmetryGroupProperties,
    SC: StructureConstraint + fmt::Display,
    SlaterDeterminant<'a, f64, SC>: SymmetryTransformable,
{
    fn_calc_xmat_real!(
        /// Calculates the $`\mathbf{X}`$ matrix for real and symmetric overlap matrix
        /// $`\mathbf{S}`$ between the symmetry-equivalent Slater determinants in the orbit.
        ///
        /// The resulting $`\mathbf{X}`$ is stored in the orbit.
        ///
        /// # Arguments
        ///
        /// * `preserves_full_rank` - If `true`, when $`\mathbf{S}`$ is already of full rank, then
        /// $`\mathbf{X}`$ is set to be the identity matrix to avoid mixing the orbit determinants.
        /// If `false`, $`\mathbf{X}`$ also orthogonalises $`\mathbf{S}`$ even when it is already of
        /// full rank.
        pub calc_xmat
    );
}

impl<'a, G, T, SC> SlaterDeterminantSymmetryOrbit<'a, G, Complex<T>, SC>
where
    G: SymmetryGroupProperties,
    T: Float + Scalar<Complex = Complex<T>>,
    Complex<T>: ComplexFloat<Real = T> + Scalar<Real = T, Complex = Complex<T>> + Lapack,
    SC: StructureConstraint + fmt::Display,
    SlaterDeterminant<'a, Complex<T>, SC>: SymmetryTransformable + Overlap<Complex<T>, Ix2>,
{
    fn_calc_xmat_complex!(
        /// Calculates the $`\mathbf{X}`$ matrix for complex and symmetric or Hermitian overlap
        /// matrix $`\mathbf{S}`$ between the symmetry-equivalent Slater determinants in the orbit.
        ///
        /// The resulting $`\mathbf{X}`$ is stored in the orbit.
        ///
        /// # Arguments
        ///
        /// * `preserves_full_rank` - If `true`, when $`\mathbf{S}`$ is already of full rank, then
        /// $`\mathbf{X}`$ is set to be the identity matrix to avoid mixing the orbit determinants.
        /// If `false`, $`\mathbf{X}`$ also orthogonalises $`\mathbf{S}`$ even when it is already of
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

impl<'a, G, T, SC> Orbit<G, SlaterDeterminant<'a, T, SC>>
    for SlaterDeterminantSymmetryOrbit<'a, G, T, SC>
where
    G: SymmetryGroupProperties,
    T: ComplexFloat + fmt::Debug + Lapack,
    SC: StructureConstraint + fmt::Display,
    SlaterDeterminant<'a, T, SC>: SymmetryTransformable,
{
    type OrbitIter = OrbitIterator<'a, G, SlaterDeterminant<'a, T, SC>>;

    fn group(&self) -> &G {
        self.group
    }

    fn origin(&self) -> &SlaterDeterminant<'a, T, SC> {
        self.origin
    }

    fn iter(&self) -> Self::OrbitIter {
        OrbitIterator::new(
            self.group,
            self.origin,
            match self.symmetry_transformation_kind {
                SymmetryTransformationKind::Spatial => |op, det| {
                    det.sym_transform_spatial(op).with_context(|| {
                        format!("Unable to apply `{op}` spatially on the origin determinant")
                    })
                },
                SymmetryTransformationKind::SpatialWithSpinTimeReversal => |op, det| {
                    det.sym_transform_spatial_with_spintimerev(op).with_context(|| {
                        format!("Unable to apply `{op}` spatially (with spin-including time reversal) on the origin determinant")
                    })
                },
                SymmetryTransformationKind::Spin => |op, det| {
                    det.sym_transform_spin(op).with_context(|| {
                        format!("Unable to apply `{op}` spin-wise on the origin determinant")
                    })
                },
                SymmetryTransformationKind::SpinSpatial => |op, det| {
                    det.sym_transform_spin_spatial(op).with_context(|| {
                        format!("Unable to apply `{op}` spin-spatially on the origin determinant")
                    })
                },
            },
        )
    }
}

// ~~~~~~~~~~~
// RepAnalysis
// ~~~~~~~~~~~

impl<'a, G, T, SC> RepAnalysis<G, SlaterDeterminant<'a, T, SC>, T, Ix2>
    for SlaterDeterminantSymmetryOrbit<'a, G, T, SC>
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
    SC: StructureConstraint + Eq + fmt::Display,
    SlaterDeterminant<'a, T, SC>: SymmetryTransformable,
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
            Err(format_err!("`norm_preserving_scalar_map` is currently not implemented for complex symmetric overlaps."))
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

    /// Reduces the representation or corepresentation spanned by the determinants in the orbit to
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
    /// are non-integral, or also because the combination of group type, transformation type, and
    /// oddity of the number of electrons would not give sensible symmetry results. In particular,
    /// spin or spin-spatial symmetry analysis of odd-electron systems in unitary-represented
    /// magnetic groups is not valid.
    fn analyse_rep(
        &self,
    ) -> Result<
        <<G as CharacterProperties>::CharTab as SubspaceDecomposable<T>>::Decomposition,
        DecompositionError,
    > {
        log::debug!("Analysing representation symmetry for a Slater determinant...");
        let nelectrons_float = self.origin().nelectrons();
        if approx::relative_eq!(
            nelectrons_float.round(),
            nelectrons_float,
            epsilon = self.integrality_threshold,
            max_relative = self.integrality_threshold
        ) {
            let nelectrons_usize = nelectrons_float.round().to_usize().unwrap_or_else(|| {
                panic!(
                    "Unable to convert the number of electrons `{nelectrons_float:.7}` to `usize`."
                );
            });
            let (valid_symmetry, err_str) = if nelectrons_usize.rem_euclid(2) == 0 {
                // Even number of electrons; always valid
                (true, String::new())
            } else {
                // Odd number of electrons; validity depends on group and orbit type
                match self.symmetry_transformation_kind {
                    SymmetryTransformationKind::Spatial => (true, String::new()),
                    SymmetryTransformationKind::SpatialWithSpinTimeReversal
                        | SymmetryTransformationKind::Spin
                        | SymmetryTransformationKind::SpinSpatial => {
                        match self.group().group_type() {
                            GroupType::Ordinary(_) => (true, String::new()),
                            GroupType::MagneticGrey(_) | GroupType::MagneticBlackWhite(_) => {
                                (!self.group().unitary_represented(),
                                "Unitary-represented magnetic groups cannot be used for symmetry analysis of odd-electron systems where spin is treated explicitly.".to_string())
                            }
                        }
                    }
                }
            };

            if valid_symmetry {
                let chis = self
                    .calc_characters()
                    .map_err(|err| DecompositionError(err.to_string()))?;
                log::debug!("Characters calculated.");

                log_subtitle("Determinant orbit characters");
                qsym2_output!("");
                self.characters_to_string(&chis, self.integrality_threshold)
                    .log_output_display();
                qsym2_output!("");

                let res = self.group().character_table().reduce_characters(
                    &chis.iter().map(|(cc, chi)| (cc, *chi)).collect::<Vec<_>>(),
                    self.integrality_threshold(),
                );
                log::debug!("Characters reduced.");
                log::debug!("Analysing representation symmetry for a Slater determinant... Done.");
                res
            } else {
                Err(DecompositionError(err_str))
            }
        } else {
            Err(DecompositionError(format!(
                "Symmetry analysis for determinant with non-integer number of electrons `{nelectrons_float:.7}` (threshold = {:.3e}) not supported.",
                self.integrality_threshold
            )))
        }
    }
}
