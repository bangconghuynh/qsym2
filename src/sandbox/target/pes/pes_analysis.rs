//! Implementation of symmetry analysis for electron densities.

use std::fmt;
use std::ops::Mul;

use anyhow::{self, ensure, format_err, Context};
use approx;
use derive_builder::Builder;
use itertools::Itertools;
use log;
use nalgebra::Point3;
use ndarray::{Array1, Array2, Axis, Ix1};
use ndarray_linalg::{
    eig::Eig,
    eigh::Eigh,
    types::{Lapack, Scalar},
    UPLO,
};
use num_complex::{Complex, ComplexFloat};
use num_traits::{Float, ToPrimitive, Zero};
use rayon::prelude::*;

use crate::analysis::{
    fn_calc_xmat_complex, fn_calc_xmat_real, EigenvalueComparisonMode, Orbit, OrbitIterator,
    Overlap, RepAnalysis,
};
use crate::auxiliary::misc::complex_modified_gram_schmidt;
use crate::chartab::chartab_group::CharacterProperties;
use crate::chartab::{DecompositionError, SubspaceDecomposable};
use crate::io::format::{log_subtitle, qsym2_output, QSym2Output};
use crate::sandbox::target::pes::PES;
use crate::symmetry::symmetry_element::symmetry_operation::SpecialSymmetryTransformation;
use crate::symmetry::symmetry_group::SymmetryGroupProperties;
use crate::symmetry::symmetry_transformation::{SymmetryTransformable, SymmetryTransformationKind};

// =======
// Overlap
// =======

impl<T, F> Overlap<T, Ix1> for PES<T, F>
where
    T: ComplexFloat + Lapack + Send + Sync,
    F: Clone + Sync + Send + Fn(&Point3<f64>) -> T,
{
    fn complex_symmetric(&self) -> bool {
        false
    }

    /// Computes the overlap between two PESes.
    ///
    /// No attempts are made to check that the grid points between the two PESes are 'compatible'.
    /// The overlap is simply evaluated as
    ///
    /// ```math
    ///     \sum_i V_1^*(\mathbf{r}_{1i}) V_2(\mathbf{r}_{2i}) w_i.
    /// ```
    ///
    ///
    /// # Errors
    ///
    /// Errors if `self` and `other` have mismatched numbers of grid points or if the number of
    /// weight values does not match the number of grid points.
    fn overlap(
        &self,
        other: &Self,
        metric: Option<&Array1<T>>,
        _: Option<&Array1<T>>,
    ) -> Result<T, anyhow::Error> {
        let weight =
            metric.ok_or_else(|| format_err!("No weights found for PES overlap calculation."))?;

        ensure!(
            self.grid_points.len() == other.grid_points.len(),
            "Inconsistent number of grid points between `self` and `other`."
        );
        ensure!(
            self.grid_points.len() == weight.len(),
            "Inconsistent number of weight values and grid points."
        );

        let overlap = (0..weight.len())
            .into_par_iter()
            .map(|i| {
                let s_pt = self.grid_points[i];
                let o_pt = other.grid_points[i];
                let w = weight[i];
                match (self.complex_conjugated, other.complex_conjugated) {
                    (false, false) => self.function()(&s_pt).conj() * other.function()(&o_pt) * w,
                    (false, true) => {
                        self.function()(&s_pt).conj() * other.function()(&o_pt).conj() * w
                    }
                    (true, false) => self.function()(&s_pt) * other.function()(&o_pt) * w,
                    (true, true) => self.function()(&s_pt) * other.function()(&o_pt).conj() * w,
                }
            })
            .sum();
        Ok(overlap)
    }
}

// ================
// PESSymmetryOrbit
// ================

// -----------------
// Struct definition
// -----------------

/// Structure to manage symmetry orbits (*i.e.* orbits generated by symmetry groups) of
/// PESes.
#[derive(Builder, Clone)]
pub struct PESSymmetryOrbit<'a, G, T, F>
where
    G: SymmetryGroupProperties,
    T: ComplexFloat + fmt::Debug + Lapack,
    F: Fn(&Point3<f64>) -> T,
    PES<T, F>: SymmetryTransformable,
{
    /// The generating symmetry group.
    group: &'a G,

    /// The origin PES of the orbit.
    origin: &'a PES<T, F>,

    /// The threshold for determining zero eigenvalues in the orbit overlap matrix.
    pub(crate) linear_independence_threshold: <T as ComplexFloat>::Real,

    /// The threshold for determining if calculated multiplicities in representation analysis are
    /// integral.
    integrality_threshold: <T as ComplexFloat>::Real,

    /// The kind of transformation determining the way the symmetry operations in the generating
    /// group act on [`Self::origin`].
    symmetry_transformation_kind: SymmetryTransformationKind,

    /// The overlap matrix between the symmetry-equivalent PESes in the orbit.
    #[builder(setter(skip), default = "None")]
    smat: Option<Array2<T>>,

    /// The eigenvalues of the overlap matrix between the symmetry-equivalent PESes in the orbit.
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

impl<'a, G, T, F> PESSymmetryOrbit<'a, G, T, F>
where
    G: SymmetryGroupProperties + Clone,
    T: ComplexFloat + fmt::Debug + Lapack,
    F: Clone + Fn(&Point3<f64>) -> T,
    PES<T, F>: SymmetryTransformable,
{
    /// Returns a builder for constructing a new density symmetry orbit.
    pub fn builder() -> PESSymmetryOrbitBuilder<'a, G, T, F> {
        PESSymmetryOrbitBuilder::default()
    }
}

impl<'a, G, F> PESSymmetryOrbit<'a, G, f64, F>
where
    G: SymmetryGroupProperties + Clone,
    F: Clone + Fn(&Point3<f64>) -> f64,
{
    fn_calc_xmat_real!(
        /// Calculates the $`\mathbf{X}`$ matrix for real and symmetric overlap matrix
        /// $`\mathbf{S}`$ between the symmetry-equivalent PESes in the orbit.
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

impl<'a, G, T, F> PESSymmetryOrbit<'a, G, Complex<T>, F>
where
    G: SymmetryGroupProperties + Clone,
    T: Float + Scalar<Complex = Complex<T>>,
    Complex<T>: ComplexFloat<Real = T> + Scalar<Real = T, Complex = Complex<T>> + Lapack,
    F: Clone + Fn(&Point3<f64>) -> Complex<T>,
    PES<Complex<T>, F>: SymmetryTransformable + Overlap<Complex<T>, Ix1>,
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

impl<'a, G, T, F> Orbit<G, PES<T, F>> for PESSymmetryOrbit<'a, G, T, F>
where
    G: SymmetryGroupProperties + Clone,
    T: ComplexFloat + fmt::Debug + Lapack,
    F: Fn(&Point3<f64>) -> T,
    PES<T, F>: SymmetryTransformable,
{
    type OrbitIter = OrbitIterator<'a, G, PES<T, F>>;

    fn group(&self) -> &'a G {
        self.group
    }

    fn origin(&self) -> &PES<T, F> {
        self.origin
    }

    fn iter(&self) -> Self::OrbitIter {
        OrbitIterator::new(
            self.group,
            self.origin,
            match self.symmetry_transformation_kind {
                SymmetryTransformationKind::Spatial => |op, pes| {
                    pes.sym_transform_spatial(op).with_context(|| {
                        format!("Unable to apply `{op}` spatially on the origin PES")
                    })
                },
                SymmetryTransformationKind::SpatialWithSpinTimeReversal => |op, pes| {
                    pes.sym_transform_spatial_with_spintimerev(op).with_context(|| {
                        format!("Unable to apply `{op}` spatially (with spin-including time-reversal) on the origin PES")
                    })
                },
                SymmetryTransformationKind::Spin => |op, pes| {
                    pes.sym_transform_spin(op).with_context(|| {
                        format!("Unable to apply `{op}` spin-wise on the origin PES")
                    })
                },
                SymmetryTransformationKind::SpinSpatial => |op, pes| {
                    pes.sym_transform_spin_spatial(op).with_context(|| {
                        format!("Unable to apply `{op}` spin-spatially on the origin PES")
                    })
                },
            },
        )
    }
}

// ~~~~~~~~~~~
// RepAnalysis
// ~~~~~~~~~~~

impl<'a, G, T, F> RepAnalysis<G, PES<T, F>, T, Ix1> for PESSymmetryOrbit<'a, G, T, F>
where
    G: SymmetryGroupProperties + Clone,
    G::CharTab: SubspaceDecomposable<T>,
    T: Lapack
        + ComplexFloat<Real = <T as Scalar>::Real>
        + fmt::Debug
        + Send
        + Sync
        + Mul<<T as ComplexFloat>::Real, Output = T>,
    <T as ComplexFloat>::Real: fmt::Debug
        + Zero
        + From<u16>
        + ToPrimitive
        + approx::RelativeEq<<T as ComplexFloat>::Real>
        + approx::AbsDiffEq<Epsilon = <T as Scalar>::Real>,
    F: Clone + Sync + Send + Fn(&Point3<f64>) -> T,
    PES<T, F>: SymmetryTransformable,
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

    fn norm_preserving_scalar_map(&self, i: usize) -> fn(T) -> T {
        if self
            .group()
            .get_index(i)
            .unwrap_or_else(|| panic!("Group operation index `{i}` not found."))
            .contains_time_reversal()
        {
            ComplexFloat::conj
        } else {
            |x| x
        }
    }

    fn integrality_threshold(&self) -> <T as ComplexFloat>::Real {
        self.integrality_threshold
    }

    fn eigenvalue_comparison_mode(&self) -> &EigenvalueComparisonMode {
        &self.eigenvalue_comparison_mode
    }

    /// Reduces the representation or corepresentation spanned by the PESes in the orbit to
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
        log::debug!("Analysing representation symmetry for a PES...");
        let chis = self
            .calc_characters()
            .map_err(|err| DecompositionError(err.to_string()))?;
        log::debug!("Characters calculated.");
        log_subtitle("PES orbit characters");
        qsym2_output!("");
        self.characters_to_string(&chis, self.integrality_threshold)
            .log_output_display();
        qsym2_output!("");

        let res = self.group().character_table().reduce_characters(
            &chis.iter().map(|(cc, chi)| (cc, *chi)).collect::<Vec<_>>(),
            self.integrality_threshold(),
        );
        log::debug!("Characters reduced.");
        log::debug!("Analysing representation symmetry for a PES... Done.");
        res
    }
}
