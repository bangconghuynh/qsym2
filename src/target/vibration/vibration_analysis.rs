use log;
use std::fmt;
use std::ops::Mul;

use approx;
use derive_builder::Builder;
use itertools::Itertools;
use ndarray::{Array2, Axis, Ix2};
use ndarray_linalg::{
    assert_close_l2,
    eig::Eig,
    eigh::Eigh,
    types::{Lapack, Scalar},
    UPLO,
};
use num_complex::{Complex, ComplexFloat};
use num_traits::{Float, Zero};

use crate::analysis::{Orbit, OrbitIterator, Overlap, RepAnalysis, RepAnalysisError};
use crate::aux::misc::complex_modified_gram_schmidt;
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
    fn overlap(&self, other: &Self, metric: Option<&Array2<T>>) -> Result<T, RepAnalysisError> {
        assert_eq!(
            self.coefficients.len(),
            other.coefficients.len(),
            "Inconsistent numbers of coefficient matrices between `self` and `other`."
        );

        let ov = if let Some(s) = metric {
            self
                .coefficients
                .t()
                .mapv(|x| x.conj())
                .dot(s)
                .dot(&other.coefficients)
        } else {
            self
                .coefficients
                .t()
                .mapv(|x| x.conj())
                .dot(&other.coefficients)
        };
        Ok(ov)
    }
}

// ==================================
// VibrationalCoordinateSymmetryOrbit
// ==================================

// -----------------
// Struct definition
// -----------------

#[derive(Builder, Clone)]
pub struct VibrationalCoordinateSymmetryOrbit<'a, G, T>
where
    G: SymmetryGroupProperties,
    T: ComplexFloat + fmt::Debug + Lapack,
    VibrationalCoordinate<'a, T>: SymmetryTransformable,
{
    group: &'a G,

    origin: &'a VibrationalCoordinate<'a, T>,

    symmetry_transformation_kind: SymmetryTransformationKind,

    #[builder(setter(skip), default = "None")]
    pub smat: Option<Array2<T>>,

    #[builder(setter(skip), default = "None")]
    pub xmat: Option<Array2<T>>,
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
    pub fn builder() -> VibrationalCoordinateSymmetryOrbitBuilder<'a, G, T> {
        VibrationalCoordinateSymmetryOrbitBuilder::default()
    }
}

impl<'a, G> VibrationalCoordinateSymmetryOrbit<'a, G, f64>
where
    G: SymmetryGroupProperties,
{
    pub fn calc_xmat(&mut self, preserves_full_rank: bool) -> &mut Self {
        // Real, symmetric S
        let thresh = self.origin.threshold;
        let smat = self
            .smat
            .as_ref()
            .expect("No overlap matrix found for this orbit.");
        assert_close_l2!(&smat, &smat.t(), thresh);
        let (s_eig, umat) = smat.eigh(UPLO::Lower).unwrap();
        let nonzero_s_indices = s_eig.iter().positions(|x| x.abs() > thresh).collect_vec();
        let nonzero_s_eig = s_eig.select(Axis(0), &nonzero_s_indices);
        let nonzero_umat = umat.select(Axis(1), &nonzero_s_indices);
        let nullity = smat.shape()[0] - nonzero_s_indices.len();
        let xmat = if nullity == 0 && preserves_full_rank {
            Array2::eye(smat.shape()[0])
        } else {
            let s_s = Array2::<f64>::from_diag(&nonzero_s_eig.mapv(|x| 1.0 / x.sqrt()));
            nonzero_umat.dot(&s_s)
        };
        self.xmat = Some(xmat);
        self
    }
}

impl<'a, G, T> VibrationalCoordinateSymmetryOrbit<'a, G, Complex<T>>
where
    G: SymmetryGroupProperties,
    T: Float + Scalar<Complex = Complex<T>>,
    Complex<T>: ComplexFloat<Real = T> + Scalar<Real = T, Complex = Complex<T>> + Lapack,
    VibrationalCoordinate<'a, Complex<T>>: SymmetryTransformable + Overlap<Complex<T>, Ix2>,
{
    pub fn calc_xmat(&mut self, preserves_full_rank: bool) {
        // Complex S, symmetric or Hermitian
        let thresh = self.origin.threshold;
        let smat = self
            .smat
            .as_ref()
            .expect("No overlap matrix found for this orbit.");
        let (s_eig, umat_nonortho) = smat.eig().unwrap();

        let nonzero_s_indices = s_eig
            .iter()
            .positions(|x| ComplexFloat::abs(*x) > thresh)
            .collect_vec();
        let nonzero_s_eig = s_eig.select(Axis(0), &nonzero_s_indices);
        let nonzero_umat_nonortho = umat_nonortho.select(Axis(1), &nonzero_s_indices);

        // `eig` does not guarantee orthogonality of `nonzero_umat_nonortho`.
        // Gram--Schmidt is therefore required.
        let nonzero_umat = complex_modified_gram_schmidt(
            &nonzero_umat_nonortho,
            self.origin.complex_symmetric(),
            thresh,
        )
        .expect(
            "Unable to orthonormalise the linearly-independent eigenvectors of the overlap matrix.",
        );

        let nullity = smat.shape()[0] - nonzero_s_indices.len();
        let xmat = if nullity == 0 && preserves_full_rank {
            Array2::<Complex<T>>::eye(smat.shape()[0])
        } else {
            let s_s = Array2::<Complex<T>>::from_diag(
                &nonzero_s_eig.mapv(|x| Complex::<T>::from(T::one()) / x.sqrt()),
            );
            nonzero_umat.dot(&s_s)
        };
        self.xmat = Some(xmat);
    }
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
                SymmetryTransformationKind::Spatial => |op, orb| {
                    orb.sym_transform_spatial(&op).unwrap_or_else(|err| {
                        log::error!("{err}");
                        panic!("Unable to apply `{op}` spatially on the origin orbital.")
                    })
                },
                SymmetryTransformationKind::Spin => |op, orb| {
                    orb.sym_transform_spin(&op).unwrap_or_else(|err| {
                        log::error!("{err}");
                        panic!("Unable to apply `{op}` spin-wise on the origin orbital.")
                    })
                },
                SymmetryTransformationKind::SpinSpatial => |op, orb| {
                    orb.sym_transform_spin_spatial(&op).unwrap_or_else(|err| {
                        log::error!("{err}");
                        panic!("Unable to apply `{op}` spin-spatially on the origin orbital.",)
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

    fn smat(&self) -> &Array2<T> {
        self.smat.as_ref().expect("Orbit overlap matrix not found.")
    }

    fn xmat(&self) -> &Array2<T> {
        self.xmat
            .as_ref()
            .expect("Orbit overlap orthogonalisation matrix not found.")
    }

    fn norm_preserving_scalar_map(&self, i: usize) -> fn(T) -> T {
        if self.group.get_index(i).unwrap().is_antiunitary() {
            ComplexFloat::conj
        } else {
            |x| x
        }
    }

    fn integrality_threshold(&self) -> <T as ComplexFloat>::Real {
        self.origin.threshold
    }
}
