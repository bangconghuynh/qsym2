//! Implementation of symmetry analysis for orbitals.

use std::fmt;
use std::ops::Mul;

use anyhow::{self, ensure, format_err, Context};
use approx;
use derive_builder::Builder;
use itertools::{izip, Itertools};
use log;
use ndarray::{s, Array1, Array2, Axis, Ix2};
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
use crate::auxiliary::misc::{complex_modified_gram_schmidt, ProductRepeat};
use crate::chartab::chartab_group::CharacterProperties;
use crate::chartab::{DecompositionError, SubspaceDecomposable};
use crate::group::GroupType;
use crate::symmetry::symmetry_element::symmetry_operation::SpecialSymmetryTransformation;
use crate::symmetry::symmetry_group::SymmetryGroupProperties;
use crate::symmetry::symmetry_transformation::{SymmetryTransformable, SymmetryTransformationKind};
use crate::target::determinant::determinant_analysis::SlaterDeterminantSymmetryOrbit;
use crate::target::determinant::SlaterDeterminant;
use crate::target::orbital::MolecularOrbital;

// -------
// Overlap
// -------

impl<'a, T, SC> Overlap<T, Ix2> for MolecularOrbital<'a, T, SC>
where
    T: Lapack
        + ComplexFloat<Real = <T as Scalar>::Real>
        + fmt::Debug
        + Mul<<T as ComplexFloat>::Real, Output = T>,
    <T as ComplexFloat>::Real: fmt::Debug
        + approx::RelativeEq<<T as ComplexFloat>::Real>
        + approx::AbsDiffEq<Epsilon = <T as Scalar>::Real>,
    SC: StructureConstraint + Eq,
{
    fn complex_symmetric(&self) -> bool {
        self.complex_symmetric
    }

    /// Computes the overlap between two molecular orbitals.
    ///
    /// When one or both of the orbitals have been acted on by an antiunitary operation, the correct
    /// Hermitian or complex-symmetric metric will be chosen in the evalulation of the overlap.
    ///
    /// # Panics
    ///
    /// Panics if `self` and `other` have mismatched spin constraints or coefficient array lengths.
    fn overlap(
        &self,
        other: &Self,
        metric: Option<&Array2<T>>,
        metric_h: Option<&Array2<T>>,
    ) -> Result<T, anyhow::Error> {
        ensure!(
            self.structure_constraint == other.structure_constraint,
            "Inconsistent spin constraints between `self` and `other`."
        );
        ensure!(
            self.coefficients.len() == other.coefficients.len(),
            "Inconsistent numbers of coefficient matrices between `self` and `other`."
        );

        let sao = metric.ok_or_else(|| format_err!("No atomic-orbital metric found."))?;
        let sao_h = metric_h.unwrap_or(sao);
        let ov = if self.component_index != other.component_index {
            T::zero()
        } else if self.complex_symmetric() {
            match (self.complex_conjugated, other.complex_conjugated) {
                (false, false) => self.coefficients.t().dot(sao_h).dot(&other.coefficients),
                (true, false) => self.coefficients.t().dot(sao).dot(&other.coefficients),
                (false, true) => other.coefficients.t().dot(sao).dot(&self.coefficients),
                (true, true) => self
                    .coefficients
                    .t()
                    .dot(&sao_h.t())
                    .dot(&other.coefficients),
            }
        } else {
            match (self.complex_conjugated, other.complex_conjugated) {
                (false, false) => self
                    .coefficients
                    .t()
                    .mapv(|x| x.conj()) // This conjugation is still needed as it comes from the sesquilinear form.
                    .dot(sao)
                    .dot(&other.coefficients),
                (true, false) => self
                    .coefficients
                    .t()
                    .mapv(|x| x.conj()) // This conjugation is still needed as it comes from the sesquilinear form.
                    .dot(sao_h)
                    .dot(&other.coefficients),
                (false, true) => other
                    .coefficients
                    .t()
                    .mapv(|x| x.conj()) // This conjugation is still needed as it comes from the sesquilinear form.
                    .dot(sao_h)
                    .dot(&self.coefficients)
                    .conj(),
                (true, true) => self
                    .coefficients
                    .t()
                    .mapv(|x| x.conj()) // This conjugation is still needed as it comes from the sesquilinear form.
                    .dot(&sao.t())
                    .dot(&other.coefficients),
            }
        };
        Ok(ov)
    }

    /// Returns the mathematical definition of the overlap between two orbitals.
    fn overlap_definition(&self) -> String {
        let k = if self.complex_symmetric() { "κ " } else { "" };
        format!("⟨{k}ψ_1|ψ_2⟩ = ∫ [{k}ψ_1(x)]* ψ_2(x) dx")
    }
}

// =============================
// MolecularOrbitalSymmetryOrbit
// =============================

// -----------------
// Struct definition
// -----------------

/// Structure to manage symmetry orbits (*i.e.* orbits generated by symmetry groups) of molecular
/// orbitals.
#[derive(Builder, Clone)]
pub struct MolecularOrbitalSymmetryOrbit<'a, G, T, SC>
where
    G: SymmetryGroupProperties,
    T: ComplexFloat + fmt::Debug + Lapack,
    SC: StructureConstraint,
    MolecularOrbital<'a, T, SC>: SymmetryTransformable,
{
    /// The generating symmetry group.
    group: &'a G,

    /// The origin molecular orbital of the orbit.
    origin: &'a MolecularOrbital<'a, T, SC>,

    /// The threshold for determining if calculated multiplicities in representation analysis are
    /// integral.
    integrality_threshold: <T as ComplexFloat>::Real,

    /// The threshold for determining zero eigenvalues in the orbit overlap matrix.
    pub(crate) linear_independence_threshold: <T as ComplexFloat>::Real,

    /// The kind of transformation determining the way the symmetry operations in `group` act on
    /// [`Self::origin`].
    symmetry_transformation_kind: SymmetryTransformationKind,

    /// The overlap matrix between the symmetry-equivalent molecular orbitals in the orbit.
    #[builder(setter(skip), default = "None")]
    smat: Option<Array2<T>>,

    /// The eigenvalues of the overlap matrix between the symmetry-equivalent molecular orbitals in
    /// the orbit.
    #[builder(setter(skip), default = "None")]
    pub(crate) smat_eigvals: Option<Array1<T>>,

    /// The $`\mathbf{X}`$ matrix for the overlap matrix between the symmetry-equivalent molecular
    /// orbitals in the orbit.
    ///
    /// See [`RepAnalysis::xmat`] for further information.
    #[builder(setter(skip), default = "None")]
    xmat: Option<Array2<T>>,

    /// An enumerated type specifying the comparison mode for filtering out orbit overlap
    /// eigenvalues.
    pub(crate) eigenvalue_comparison_mode: EigenvalueComparisonMode,
}

// ----------------------------
// Struct method implementation
// ----------------------------

impl<'a, G, T, SC> MolecularOrbitalSymmetryOrbit<'a, G, T, SC>
where
    G: SymmetryGroupProperties + Clone,
    T: ComplexFloat + fmt::Debug + Lapack,
    SC: StructureConstraint + Clone,
    MolecularOrbital<'a, T, SC>: SymmetryTransformable,
{
    /// Returns a builder to construct a new [`MolecularOrbitalSymmetryOrbit`] structure.
    pub fn builder() -> MolecularOrbitalSymmetryOrbitBuilder<'a, G, T, SC> {
        MolecularOrbitalSymmetryOrbitBuilder::default()
    }

    /// Constructs multiple molecular orbital orbits, each from one of the supplied orbitals.
    ///
    /// # Arguments
    ///
    /// * `group` - The orbit-generating group.
    /// * `orbitals` - The origin orbitals, each of which generates its own orbit.
    /// * `sym_kind` - The symmetry transformation kind.
    /// * `integrality_thresh` - The threshold of integrality check of multiplicity coefficients in
    /// each orbit.
    /// * `linear_independence_thresh` - The threshold of linear independence for each orbit.
    ///
    /// # Returns
    ///
    /// A vector of molecular orbital orbits.
    pub fn from_orbitals(
        group: &'a G,
        orbitals: &'a [Vec<MolecularOrbital<'a, T, SC>>],
        sym_kind: SymmetryTransformationKind,
        eig_comp_mode: EigenvalueComparisonMode,
        integrality_thresh: <T as ComplexFloat>::Real,
        linear_independence_thresh: <T as ComplexFloat>::Real,
    ) -> Vec<Vec<Self>> {
        orbitals
            .iter()
            .map(|orbs_spin| {
                orbs_spin
                    .iter()
                    .map(|orb| {
                        MolecularOrbitalSymmetryOrbit::builder()
                            .group(group)
                            .origin(orb)
                            .integrality_threshold(integrality_thresh)
                            .linear_independence_threshold(linear_independence_thresh)
                            .symmetry_transformation_kind(sym_kind.clone())
                            .eigenvalue_comparison_mode(eig_comp_mode.clone())
                            .build()
                            .expect("Unable to construct a molecular orbital symmetry orbit.")
                    })
                    .collect_vec()
            })
            .collect_vec()
    }
}

impl<'a, G, SC> MolecularOrbitalSymmetryOrbit<'a, G, f64, SC>
where
    G: SymmetryGroupProperties,
    SC: StructureConstraint,
    MolecularOrbital<'a, f64, SC>: SymmetryTransformable,
{
    fn_calc_xmat_real!(
        /// Calculates the $`\mathbf{X}`$ matrix for real and symmetric overlap matrix
        /// $`\mathbf{S}`$ between the symmetry-equivalent molecular orbitals in the orbit.
        ///
        /// The resulting $`\mathbf{X}`$ is stored in the orbit.
        ///
        /// # Arguments
        ///
        /// * `preserves_full_rank` - If `true`, when $`\mathbf{S}`$ is already of full rank, then
        /// $`\mathbf{X}`$ is set to be the identity matrix to avoid mixing the orbit molecular
        /// orbitals. If `false`, $`\mathbf{X}`$ also orthogonalises $`\mathbf{S}`$ even when it is
        /// already of full rank.
        pub calc_xmat
    );
}

impl<'a, G, T, SC> MolecularOrbitalSymmetryOrbit<'a, G, Complex<T>, SC>
where
    G: SymmetryGroupProperties,
    T: Float + Scalar<Complex = Complex<T>>,
    Complex<T>: ComplexFloat<Real = T> + Scalar<Real = T, Complex = Complex<T>> + Lapack,
    SC: StructureConstraint,
    MolecularOrbital<'a, Complex<T>, SC>: SymmetryTransformable + Overlap<Complex<T>, Ix2>,
{
    fn_calc_xmat_complex!(
        /// Calculates the $`\mathbf{X}`$ matrix for complex and symmetric or Hermitian overlap
        /// matrix $`\mathbf{S}`$ between the symmetry-equivalent molecular orbitals in the orbit.
        ///
        /// The resulting $`\mathbf{X}`$ is stored in the orbit.
        ///
        /// # Arguments
        ///
        /// * `preserves_full_rank` - If `true`, when $`\mathbf{S}`$ is already of full rank, then
        /// $`\mathbf{X}`$ is set to be the identity matrix to avoid mixing the orbit molecular
        /// orbitals. If `false`, $`\mathbf{X}`$ also orthogonalises $`\mathbf{S}`$ even when it is
        /// already of full rank.
        pub calc_xmat
    );
}

// ---------------------
// Trait implementations
// ---------------------

// ~~~~~
// Orbit
// ~~~~~

impl<'a, G, T, SC> Orbit<G, MolecularOrbital<'a, T, SC>>
    for MolecularOrbitalSymmetryOrbit<'a, G, T, SC>
where
    G: SymmetryGroupProperties,
    T: ComplexFloat + fmt::Debug + Lapack,
    SC: StructureConstraint,
    MolecularOrbital<'a, T, SC>: SymmetryTransformable,
{
    type OrbitIter = OrbitIterator<'a, G, MolecularOrbital<'a, T, SC>>;

    fn group(&self) -> &G {
        self.group
    }

    fn origin(&self) -> &MolecularOrbital<'a, T, SC> {
        self.origin
    }

    fn iter(&self) -> Self::OrbitIter {
        OrbitIterator::new(
            self.group,
            self.origin,
            match self.symmetry_transformation_kind {
                SymmetryTransformationKind::Spatial => |op, orb| {
                    orb.sym_transform_spatial(op).with_context(|| {
                        format!("Unable to apply `{op}` spatially on the origin orbital")
                    })
                },
                SymmetryTransformationKind::SpatialWithSpinTimeReversal => |op, orb| {
                    orb.sym_transform_spatial_with_spintimerev(op).with_context(|| {
                        format!("Unable to apply `{op}` spatially (with spin-including time reversal) on the origin orbital")
                    })
                },
                SymmetryTransformationKind::Spin => |op, orb| {
                    orb.sym_transform_spin(op).with_context(|| {
                        format!("Unable to apply `{op}` spin-wise on the origin orbital")
                    })
                },
                SymmetryTransformationKind::SpinSpatial => |op, orb| {
                    orb.sym_transform_spin_spatial(op).with_context(|| {
                        format!("Unable to apply `{op}` spin-spatially on the origin orbital",)
                    })
                },
            },
        )
    }
}

// ~~~~~~~~~~~
// RepAnalysis
// ~~~~~~~~~~~

impl<'a, G, T, SC> RepAnalysis<G, MolecularOrbital<'a, T, SC>, T, Ix2>
    for MolecularOrbitalSymmetryOrbit<'a, G, T, SC>
where
    G: SymmetryGroupProperties,
    G::CharTab: SubspaceDecomposable<T>,
    T: Lapack
        + ComplexFloat<Real = <T as Scalar>::Real>
        + fmt::Debug
        + Mul<<T as ComplexFloat>::Real, Output = T>,
    <T as ComplexFloat>::Real: fmt::Debug
        + Zero
        + approx::RelativeEq<<T as ComplexFloat>::Real>
        + approx::AbsDiffEq<Epsilon = <T as Scalar>::Real>,
    SC: StructureConstraint + Eq,
    MolecularOrbital<'a, T, SC>: SymmetryTransformable,
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

    /// Reduces the representation or corepresentation spanned by the molecular orbitals in the
    /// orbit to a direct sum of the irreducible representations or corepresentations of the
    /// generating symmetry group.
    ///
    /// # Returns
    ///
    /// The decomposed result.
    ///
    /// # Errors
    ///
    /// Errors if the decomposition fails, *e.g.* because one or more calculated multiplicities
    /// are non-integral, or also because the combination of group type and transformation type
    /// would not give sensible symmetry results for a single-electron orbital. In particular, spin
    /// or spin-spatial symmetry analysis in unitary-represented magnetic groups is not valid for
    /// one-electron orbitals.
    fn analyse_rep(
        &self,
    ) -> Result<
        <<G as CharacterProperties>::CharTab as SubspaceDecomposable<T>>::Decomposition,
        DecompositionError,
    > {
        // A single electron; validity depends on group and orbit type
        let (valid_symmetry, err_str) = match self.symmetry_transformation_kind {
                SymmetryTransformationKind::Spatial => (true, String::new()),
                SymmetryTransformationKind::SpatialWithSpinTimeReversal
                    | SymmetryTransformationKind::Spin
                    | SymmetryTransformationKind::SpinSpatial => {
                    match self.group().group_type() {
                        GroupType::Ordinary(_) => (true, String::new()),
                        GroupType::MagneticGrey(_) | GroupType::MagneticBlackWhite(_) => {
                            (!self.group().unitary_represented(),
                            "Unitary-represented magnetic groups cannot be used for symmetry analysis of a one-electron molecular orbital where spin is treated explicitly.".to_string())
                        }
                    }
                }
        };
        if valid_symmetry {
            log::debug!("Analysing representation symmetry for an MO...");
            let chis = self
                .calc_characters()
                .map_err(|err| DecompositionError(err.to_string()))?;
            let res = self.group().character_table().reduce_characters(
                &chis.iter().map(|(cc, chi)| (cc, *chi)).collect::<Vec<_>>(),
                self.integrality_threshold(),
            );
            log::debug!("Analysing representation symmetry for an MO... Done.");
            res
        } else {
            Err(DecompositionError(err_str))
        }
    }
}

// ---------
// Functions
// ---------

/// Given an origin determinant, generates the determinant orbit and all molecular-orbital orbits
/// in tandem while populating their $`\mathbf{S}`$ matrices at the same time.
///
/// The evaluation of the $`\mathbf{S}`$ matrix for the determinant orbit passes through
/// intermediate values that can be used to populate the $`\mathbf{S}`$ matrices of the
/// molecular-orbital orbits, so it saves time significantly to construct all orbits together.
///
/// # Arguments
///
/// * `det` - An origin determinant.
/// * `mos` - The molecular orbitals of `det`.
/// * `group` - An orbit-generating symmetry group.
/// * `metric` - The metric of the basis in which the coefficients of `det` and `mos` are written.
/// * `integrality_threshold` - The threshold of integrality check of multiplicity coefficients in
/// each orbit.
/// * `linear_independence_threshold` - The threshold of linear independence for each orbit.
/// * `symmetry_transformation_kind` - The kind of symmetry transformation to be applied to the
/// * `eigenvalue_comparison_mode` - The mode of comparing the overlap eigenvalues to the specified
/// `linear_independence_threshold`.
/// * `use_cayley_table` - A boolean indicating if the Cayley table of the group, if available,
/// should be used to speed up the computation of the orbit overlap matrix.
///
/// # Returns
///
/// A tuple consisting of:
/// - the determinant orbit, and
/// - a vector of vectors of molecular-orbital orbits, where each element of the outer vector is
/// for one spin space, and each element of an inner vector is for one molecular orbital.
pub fn generate_det_mo_orbits<'a, G, T, SC>(
    det: &'a SlaterDeterminant<'a, T, SC>,
    mos: &'a [Vec<MolecularOrbital<'a, T, SC>>],
    group: &'a G,
    metric: &Array2<T>,
    metric_h: Option<&Array2<T>>,
    integrality_threshold: <T as ComplexFloat>::Real,
    linear_independence_threshold: <T as ComplexFloat>::Real,
    symmetry_transformation_kind: SymmetryTransformationKind,
    eigenvalue_comparison_mode: EigenvalueComparisonMode,
    use_cayley_table: bool,
) -> Result<
    (
        SlaterDeterminantSymmetryOrbit<'a, G, T, SC>,
        Vec<Vec<MolecularOrbitalSymmetryOrbit<'a, G, T, SC>>>,
    ),
    anyhow::Error,
>
where
    G: SymmetryGroupProperties + Clone,
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
    SC: StructureConstraint + Clone + Eq,
    SlaterDeterminant<'a, T, SC>: SymmetryTransformable,
    MolecularOrbital<'a, T, SC>: SymmetryTransformable,
{
    log::debug!("Constructing determinant and MO orbits in tandem...");
    let order = group.order();
    let mut det_orbit = SlaterDeterminantSymmetryOrbit::<G, T, SC>::builder()
        .group(group)
        .origin(det)
        .integrality_threshold(integrality_threshold)
        .linear_independence_threshold(linear_independence_threshold)
        .symmetry_transformation_kind(symmetry_transformation_kind.clone())
        .eigenvalue_comparison_mode(eigenvalue_comparison_mode.clone())
        .build()
        .map_err(|err| format_err!(err))?;
    let mut mo_orbitss = MolecularOrbitalSymmetryOrbit::from_orbitals(
        group,
        mos,
        symmetry_transformation_kind,
        eigenvalue_comparison_mode,
        integrality_threshold,
        linear_independence_threshold,
    );

    let mut det_smat = Array2::<T>::zeros((order, order));
    let mut mo_smatss = mo_orbitss
        .iter()
        .map(|mo_orbits| {
            mo_orbits
                .iter()
                .map(|_| Array2::<T>::zeros((order, order)))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    let thresh = det.threshold();

    let sao = metric;
    let sao_h = metric_h.unwrap_or(sao);

    if let (Some(ctb), true) = (group.cayley_table(), use_cayley_table) {
        log::debug!("Cayley table available. Group closure will be used to speed up overlap matrix computation.");
        let mut det_smatw0 = Array1::<T>::zeros(order);
        let mut mo_smatw0ss = mo_orbitss
            .iter()
            .map(|mo_orbits| {
                mo_orbits
                    .iter()
                    .map(|_| Array1::<T>::zeros(order))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let det_0 = det_orbit.origin();
        for (w, det_w_res) in det_orbit.iter().enumerate() {
            let det_w = det_w_res?;
            let w0_ov = izip!(
                det_w.coefficients(),
                det_w.occupations(),
                det_0.coefficients(),
                det_0.occupations(),
            )
            .enumerate()
            .map(|(ispin, (cw, occw, c0, occ0))| {
                let nonzero_occ_w = occw.iter().positions(|&occ| occ > thresh).collect_vec();
                let nonzero_occ_0 = occ0.iter().positions(|&occ| occ > thresh).collect_vec();

                let all_mo_w0_ov_mat = if det.complex_symmetric() {
                    match (det_w.complex_conjugated(), det_0.complex_conjugated()) {
                        (false, false) => cw.t().dot(sao_h).dot(c0),
                        (true, false) => cw.t().dot(sao).dot(c0),
                        (false, true) => c0.t().dot(sao).dot(cw),
                        (true, true) => cw.t().dot(&sao_h.t()).dot(c0),
                    }
                } else {
                    match (det_w.complex_conjugated(), det_0.complex_conjugated()) {
                        (false, false) => cw.t().mapv(|x| x.conj()).dot(sao).dot(c0),
                        (true, false) => cw.t().mapv(|x| x.conj()).dot(sao_h).dot(c0),
                        (false, true) => c0
                            .t()
                            .mapv(|x| x.conj())
                            .dot(sao_h)
                            .dot(cw)
                            .mapv(|x| x.conj()),
                        (true, true) => cw.t().mapv(|x| x.conj()).dot(&sao.t()).dot(c0),
                    }
                };

                all_mo_w0_ov_mat
                    .diag()
                    .iter()
                    .enumerate()
                    .for_each(|(imo, mo_w0_ov)| {
                        mo_smatw0ss[ispin][imo][w] = *mo_w0_ov;
                    });

                let occ_mo_w0_ov_mat = all_mo_w0_ov_mat
                    .select(Axis(0), &nonzero_occ_w)
                    .select(Axis(1), &nonzero_occ_0);

                occ_mo_w0_ov_mat
                    .det()
                    .expect("The determinant of the MO overlap matrix `w0` could not be found.")
            })
            .fold(T::one(), |acc, x| acc * x);

            let implicit_factor = det.structure_constraint().implicit_factor()?;
            let w0_ov = if implicit_factor > 1 {
                let p_i32 = i32::try_from(implicit_factor)?;
                ComplexFloat::powi(w0_ov, p_i32)
            } else {
                w0_ov
            };
            det_smatw0[w] = w0_ov;
        }

        for (i, j) in (0..order).cartesian_product(0..order) {
            let jinv = ctb
                .slice(s![.., j])
                .iter()
                .position(|&x| x == 0)
                .ok_or(format_err!(
                    "Unable to find the inverse of group element `{j}`."
                ))?;
            let jinv_i = ctb[(jinv, i)];

            for (ispin, mo_smatw0s) in mo_smatw0ss.iter().enumerate() {
                for (imo, mo_smat_w0) in mo_smatw0s.iter().enumerate() {
                    mo_smatss[ispin][imo][(i, j)] =
                        mo_orbitss[ispin][imo].norm_preserving_scalar_map(jinv)?(mo_smat_w0[jinv_i])
                }
            }

            det_smat[(i, j)] = det_orbit.norm_preserving_scalar_map(jinv)?(det_smatw0[jinv_i]);
        }
    } else {
        log::debug!("Cayley table not available or the use of Cayley table not requested. Overlap matrix will be constructed without group-closure speed-up.");
        let indexed_dets = det_orbit
            .iter()
            .map(|det_res| det_res.map_err(|err| err.to_string()))
            .enumerate()
            .collect::<Vec<_>>();
        for det_pair in indexed_dets.iter().product_repeat(2) {
            let (w, det_w_res) = &det_pair[0];
            let (x, det_x_res) = &det_pair[1];
            let det_w = det_w_res
                .as_ref()
                .map_err(|err| format_err!(err.to_owned()))
                .with_context(|| "One of the determinants in the orbit is not available")?;
            let det_x = det_x_res
                .as_ref()
                .map_err(|err| format_err!(err.to_owned()))
                .with_context(|| "One of the determinants in the orbit is not available")?;

            let wx_ov = izip!(
                det_w.coefficients(),
                det_w.occupations(),
                det_x.coefficients(),
                det_x.occupations(),
            )
            .enumerate()
            .map(|(ispin, (cw, occw, cx, occx))| {
                let nonzero_occ_w = occw.iter().positions(|&occ| occ > thresh).collect_vec();
                let nonzero_occ_x = occx.iter().positions(|&occ| occ > thresh).collect_vec();

                // let all_mo_wx_ov_mat = if det.complex_symmetric() {
                //     cw.t().dot(metric).dot(cx)
                // } else {
                //     cw.t().mapv(|x| x.conj()).dot(metric).dot(cx)
                // };
                let all_mo_wx_ov_mat = if det.complex_symmetric() {
                    match (det_w.complex_conjugated(), det_x.complex_conjugated()) {
                        (false, false) => cw.t().dot(sao_h).dot(cx),
                        (true, false) => cw.t().dot(sao).dot(cx),
                        (false, true) => cx.t().dot(sao).dot(cw),
                        (true, true) => cw.t().dot(&sao_h.t()).dot(cx),
                    }
                } else {
                    match (det_w.complex_conjugated(), det_x.complex_conjugated()) {
                        (false, false) => cw.t().mapv(|x| x.conj()).dot(sao).dot(cx),
                        (true, false) => cw.t().mapv(|x| x.conj()).dot(sao_h).dot(cx),
                        (false, true) => cx
                            .t()
                            .mapv(|x| x.conj())
                            .dot(sao_h)
                            .dot(cw)
                            .mapv(|x| x.conj()),
                        (true, true) => cw.t().mapv(|x| x.conj()).dot(&sao.t()).dot(cx),
                    }
                };

                all_mo_wx_ov_mat
                    .diag()
                    .iter()
                    .enumerate()
                    .for_each(|(imo, mo_wx_ov)| {
                        mo_smatss[ispin][imo][(*w, *x)] = *mo_wx_ov;
                    });

                let occ_mo_wx_ov_mat = all_mo_wx_ov_mat
                    .select(Axis(0), &nonzero_occ_w)
                    .select(Axis(1), &nonzero_occ_x);

                occ_mo_wx_ov_mat
                    .det()
                    .expect("The determinant of the MO overlap matrix could not be found.")
            })
            .fold(T::one(), |acc, x| acc * x);

            let implicit_factor = det.structure_constraint().implicit_factor()?;
            let wx_ov = if implicit_factor > 1 {
                let p_i32 = i32::try_from(implicit_factor)?;
                ComplexFloat::powi(wx_ov, p_i32)
            } else {
                wx_ov
            };
            det_smat[(*w, *x)] = wx_ov;
        }
    }

    if det_orbit.origin().complex_symmetric() {
        det_orbit.set_smat(
            (det_smat.clone() + det_smat.t().to_owned()).mapv(|x| x / (T::one() + T::one())),
        );
    } else {
        det_orbit.set_smat(
            (det_smat.clone() + det_smat.t().to_owned().mapv(|x| x.conj()))
                .mapv(|x| x / (T::one() + T::one())),
        );
    };

    mo_orbitss
        .iter_mut()
        .enumerate()
        .for_each(|(ispin, mo_orbits)| {
            mo_orbits
                .iter_mut()
                .enumerate()
                .for_each(|(imo, mo_orbit)| {
                    if mo_orbit.origin().complex_symmetric() {
                        mo_orbit.set_smat(
                            (mo_smatss[ispin][imo].clone() + mo_smatss[ispin][imo].t().to_owned())
                                .mapv(|x| x / (T::one() + T::one())),
                        )
                    } else {
                        mo_orbit.set_smat(
                            (mo_smatss[ispin][imo].clone()
                                + mo_smatss[ispin][imo].t().to_owned().mapv(|x| x.conj()))
                            .mapv(|x| x / (T::one() + T::one())),
                        )
                    }
                })
        });

    log::debug!("Constructing determinant and MO orbits in tandem... Done.");
    Ok((det_orbit, mo_orbitss))
}
