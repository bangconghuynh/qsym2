//! Implementation of symmetry analysis for vibrational coordinates.

use std::fmt;
use std::ops::Mul;

use anyhow::{self, ensure, format_err, Context};
use approx;
use derive_builder::Builder;
use itertools::Itertools;
use ndarray::{Array1, Array2, Axis, Ix2};
use ndarray_linalg::{
    eig::Eig,
    eigh::Eigh,
    types::{Lapack, Scalar},
    UPLO,
};
use num_complex::{Complex, ComplexFloat};
use num_traits::{Float, Zero};

use crate::analysis::{
    fn_calc_xmat_complex, fn_calc_xmat_real, EigenvalueComparisonMode, Orbit, OrbitIterator,
    Overlap, RepAnalysis,
};
use crate::auxiliary::misc::complex_modified_gram_schmidt;
use crate::chartab::SubspaceDecomposable;
use crate::symmetry::symmetry_element::symmetry_operation::SpecialSymmetryTransformation;
use crate::symmetry::symmetry_group::SymmetryGroupProperties;
use crate::symmetry::symmetry_transformation::{SymmetryTransformable, SymmetryTransformationKind};
use crate::target::vibration::VibrationalCoordinate;

// -------
// Overlap
// -------

impl<'a, T> Overlap<T, Ix2> for VibrationalCoordinate<'a, T>
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
        false
    }

    /// Computes the overlap between two vibrational coordinates.
    ///
    /// Typically, `metric` is `None`, which means that the local orthogonal Cartesian coordinates
    /// are non-overlapping.
    ///
    /// # Panics
    ///
    /// Panics if `self` and `other` have mismatched coefficient array lengths.
    fn overlap(
        &self,
        other: &Self,
        metric: Option<&Array2<T>>,
        _: Option<&Array2<T>>,
    ) -> Result<T, anyhow::Error> {
        ensure!(
            self.coefficients.len() == other.coefficients.len(),
            "Inconsistent numbers of coefficient matrices between `self` and `other`."
        );

        let ov = if let Some(s) = metric {
            self.coefficients
                .t()
                .mapv(|x| x.conj())
                .dot(s)
                .dot(&other.coefficients)
        } else {
            self.coefficients
                .t()
                .mapv(|x| x.conj())
                .dot(&other.coefficients)
        };
        Ok(ov)
    }

    /// Returns the mathematical definition of the overlap between two vibrational coordinates.
    fn overlap_definition(&self) -> String {
        let k = if self.complex_symmetric() { "κ " } else { "" };
        format!("⟨{k}v_1|v_2⟩ = [{k}v_1]† g v_2    where g is an optional metric")
    }
}

// ==================================
// VibrationalCoordinateSymmetryOrbit
// ==================================

// -----------------
// Struct definition
// -----------------

/// Structure to manage symmetry orbits (*i.e.* orbits generated by symmetry groups) of
/// vibrational coordinates.
#[derive(Builder, Clone)]
pub struct VibrationalCoordinateSymmetryOrbit<'a, G, T>
where
    G: SymmetryGroupProperties,
    T: ComplexFloat + fmt::Debug + Lapack,
    VibrationalCoordinate<'a, T>: SymmetryTransformable,
{
    /// The generating symmetry group.
    group: &'a G,

    /// The origin vibrational coordinate of the orbit.
    origin: &'a VibrationalCoordinate<'a, T>,

    /// The threshold for determining if calculated multiplicities in representation analysis are
    /// integral.
    integrality_threshold: <T as ComplexFloat>::Real,

    /// The threshold for determining zero eigenvalues in the orbit overlap matrix.
    pub(crate) linear_independence_threshold: <T as ComplexFloat>::Real,

    /// The kind of transformation determining the way the symmetry operations in `group` act on
    /// [`Self::origin`].
    symmetry_transformation_kind: SymmetryTransformationKind,

    /// The overlap matrix between the symmetry-equivalent vibrational coordinates in the orbit.
    #[builder(setter(skip), default = "None")]
    smat: Option<Array2<T>>,

    /// The eigenvalues of the overlap matrix between the symmetry-equivalent vibrational
    /// coordinates in the orbit.
    #[builder(setter(skip), default = "None")]
    pub(crate) smat_eigvals: Option<Array1<T>>,

    /// The $`\mathbf{X}`$ matrix for the overlap matrix between the symmetry-equivalent
    /// vibrational coordinates in the orbit.
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

impl<'a, G, T> VibrationalCoordinateSymmetryOrbit<'a, G, T>
where
    G: SymmetryGroupProperties + Clone,
    T: ComplexFloat + fmt::Debug + Lapack,
    VibrationalCoordinate<'a, T>: SymmetryTransformable,
{
    /// Returns a builder to construct a new [`VibrationalCoordinateSymmetryOrbit`].
    pub fn builder() -> VibrationalCoordinateSymmetryOrbitBuilder<'a, G, T> {
        VibrationalCoordinateSymmetryOrbitBuilder::default()
    }
}

impl<'a, G> VibrationalCoordinateSymmetryOrbit<'a, G, f64>
where
    G: SymmetryGroupProperties,
{
    fn_calc_xmat_real!(
        /// Calculates the $`\mathbf{X}`$ matrix for real and symmetric overlap matrix
        /// $`\mathbf{S}`$ between the symmetry-equivalent vibrational coordinates in the orbit.
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

impl<'a, G, T> VibrationalCoordinateSymmetryOrbit<'a, G, Complex<T>>
where
    G: SymmetryGroupProperties,
    T: Float + Scalar<Complex = Complex<T>>,
    Complex<T>: ComplexFloat<Real = T> + Scalar<Real = T, Complex = Complex<T>> + Lapack,
    VibrationalCoordinate<'a, Complex<T>>: SymmetryTransformable + Overlap<Complex<T>, Ix2>,
{
    fn_calc_xmat_complex!(
        /// Calculates the $`\mathbf{X}`$ matrix for complex and symmetric or Hermitian overlap
        /// matrix $`\mathbf{S}`$ between the symmetry-equivalent vibrational coordinates in the
        /// orbit.
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

impl<'a, G, T> Orbit<G, VibrationalCoordinate<'a, T>>
    for VibrationalCoordinateSymmetryOrbit<'a, G, T>
where
    G: SymmetryGroupProperties,
    T: ComplexFloat + fmt::Debug + Lapack,
    VibrationalCoordinate<'a, T>: SymmetryTransformable,
{
    type OrbitIter = OrbitIterator<'a, G, VibrationalCoordinate<'a, T>>;

    fn group(&self) -> &G {
        self.group
    }

    fn origin(&self) -> &VibrationalCoordinate<'a, T> {
        self.origin
    }

    fn iter(&self) -> Self::OrbitIter {
        OrbitIterator::new(
            self.group,
            self.origin,
            match self.symmetry_transformation_kind {
                SymmetryTransformationKind::Spatial
                | SymmetryTransformationKind::SpatialWithSpinTimeReversal => |op, vib| {
                    // Vibrational coordinates are time-even, so both `sym_transform_spatial` and
                    // `sym_transform_spatial_with_spintimerev` would give the same thing.
                    vib.sym_transform_spatial(op).with_context(|| {
                        format_err!(
                            "Unable to apply `{op}` spatially on the origin vibrational coordinate"
                        )
                    })
                },
                SymmetryTransformationKind::Spin => |op, vib| {
                    vib.sym_transform_spin(op).with_context(|| {
                        format_err!(
                            "Unable to apply `{op}` spin-wise on the origin vibrational coordinate"
                        )
                    })
                },
                SymmetryTransformationKind::SpinSpatial => |op, vib| {
                    vib.sym_transform_spin_spatial(op).with_context(|| {
                        format_err!(
                            "Unable to apply `{op}` spin-spatially on the origin vibrational coordinate"
                        )
                    })
                },
            },
        )
    }
}

// ~~~~~~~~~~~
// RepAnalysis
// ~~~~~~~~~~~

impl<'a, G, T> RepAnalysis<G, VibrationalCoordinate<'a, T>, T, Ix2>
    for VibrationalCoordinateSymmetryOrbit<'a, G, T>
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
    VibrationalCoordinate<'a, T>: SymmetryTransformable,
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
        if self.origin.complex_symmetric() {
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
}
