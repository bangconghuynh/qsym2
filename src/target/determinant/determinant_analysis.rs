use log;
use std::fmt;
use std::ops::Mul;

use approx;
use derive_builder::Builder;
use itertools::{izip, Itertools};
use ndarray::{Array2, Axis};
use ndarray_linalg::{
    assert_close_l2,
    eig::Eig,
    eigh::Eigh,
    solve::Determinant,
    types::{Lapack, Scalar},
    UPLO,
};
use num_complex::{Complex, ComplexFloat};
use num_traits::{Float, ToPrimitive, Zero};

use crate::analysis::{Orbit, OrbitIterator, Overlap, RepAnalysis, RepAnalysisError};
use crate::angmom::spinor_rotation_3d::SpinConstraint;
use crate::aux::misc::complex_modified_gram_schmidt;
use crate::chartab::chartab_group::CharacterProperties;
use crate::chartab::{DecompositionError, SubspaceDecomposable};
use crate::group::GroupType;
use crate::symmetry::symmetry_element::symmetry_operation::SpecialSymmetryTransformation;
use crate::symmetry::symmetry_group::SymmetryGroupProperties;
use crate::symmetry::symmetry_transformation::{SymmetryTransformable, SymmetryTransformationKind};
use crate::target::determinant::SlaterDeterminant;

// -------
// Overlap
// -------

impl<'a, T> Overlap<T> for SlaterDeterminant<'a, T>
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

    fn overlap(&self, other: &Self, metric: &Array2<T>) -> Result<T, RepAnalysisError> {
        assert_eq!(
            self.spin_constraint, other.spin_constraint,
            "Inconsistent spin constraints between `self` and `other`."
        );
        assert_eq!(
            self.coefficients.len(),
            other.coefficients.len(),
            "Inconsistent numbers of coefficient matrices between `self` and `other`."
        );

        let thresh = Float::sqrt(self.threshold * other.threshold);
        assert!(self
            .occupations
            .iter()
            .all(|occs| occs.iter().all(|&occ| approx::relative_eq!(
                occ,
                occ.round(),
                epsilon = thresh,
                max_relative = thresh
            ))),
            "`self` contains fractional occupation numbers. Overlaps between determinants with fractional occupation numbers are currently not supported."
        );
        assert!(other
            .occupations
            .iter()
            .all(|occs| occs.iter().all(|&occ| approx::relative_eq!(
                occ,
                occ.round(),
                epsilon = thresh,
                max_relative = thresh
            ))),
            "`other` contains fractional occupation numbers. Overlaps between determinants with fractional occupation numbers are currently not supported."
        );

        let sao = metric;

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
                cw_o.t().dot(sao).dot(&cx_o)
            } else {
                cw_o.t().mapv(|x| x.conj()).dot(sao).dot(&cx_o)
            };
            mo_ov_mat
                .det()
                .expect("The determinant of the MO overlap matrix could not be found.")
        })
        .fold(T::one(), |acc, x| acc * x);

        match self.spin_constraint {
            SpinConstraint::Restricted(n_spin_spaces) => {
                Ok(ComplexFloat::powi(ov, n_spin_spaces.into()))
            }
            _ => Ok(ov),
        }
    }
}

// =====================================
// SlaterDeterminantSymmetryOrbit
// =====================================

// -----------------
// Struct definition
// -----------------

#[derive(Builder, Clone)]
pub struct SlaterDeterminantSymmetryOrbit<'a, G, T>
where
    G: SymmetryGroupProperties,
    T: ComplexFloat + fmt::Debug + Lapack,
    SlaterDeterminant<'a, T>: SymmetryTransformable,
{
    group: &'a G,

    origin: &'a SlaterDeterminant<'a, T>,

    symmetry_transformation_kind: SymmetryTransformationKind,

    #[builder(setter(skip), default = "None")]
    pub smat: Option<Array2<T>>,

    #[builder(setter(skip), default = "None")]
    pub xmat: Option<Array2<T>>,
}

// ----------------------------
// Struct method implementation
// ----------------------------

impl<'a, G, T> SlaterDeterminantSymmetryOrbit<'a, G, T>
where
    G: SymmetryGroupProperties + Clone,
    T: ComplexFloat + fmt::Debug + Lapack,
    SlaterDeterminant<'a, T>: SymmetryTransformable,
{
    pub fn builder() -> SlaterDeterminantSymmetryOrbitBuilder<'a, G, T> {
        SlaterDeterminantSymmetryOrbitBuilder::default()
    }
}

impl<'a, G> SlaterDeterminantSymmetryOrbit<'a, G, f64>
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

impl<'a, G, T> SlaterDeterminantSymmetryOrbit<'a, G, Complex<T>>
where
    G: SymmetryGroupProperties,
    T: Float + Scalar<Complex = Complex<T>>,
    Complex<T>: ComplexFloat<Real = T> + Scalar<Real = T, Complex = Complex<T>> + Lapack,
    SlaterDeterminant<'a, Complex<T>>: SymmetryTransformable + Overlap<Complex<T>>,
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

impl<'a, G, T> Orbit<G, SlaterDeterminant<'a, T>> for SlaterDeterminantSymmetryOrbit<'a, G, T>
where
    G: SymmetryGroupProperties,
    T: ComplexFloat + fmt::Debug + Lapack,
    SlaterDeterminant<'a, T>: SymmetryTransformable,
{
    type OrbitIter = OrbitIterator<'a, G, SlaterDeterminant<'a, T>>;

    fn group(&self) -> &G {
        self.group
    }

    fn origin(&self) -> &SlaterDeterminant<'a, T> {
        self.origin
    }

    fn iter(&self) -> Self::OrbitIter {
        OrbitIterator::new(
            self.group,
            self.origin,
            match self.symmetry_transformation_kind {
                SymmetryTransformationKind::Spatial => |op, det| {
                    det.sym_transform_spatial(op).unwrap_or_else(|err| {
                        log::error!("{err}");
                        panic!("Unable to apply `{op}` spatially on the origin determinant.")
                    })
                },
                SymmetryTransformationKind::Spin => |op, det| {
                    det.sym_transform_spin(op).unwrap_or_else(|err| {
                        log::error!("{err}");
                        panic!("Unable to apply `{op}` spin-wise on the origin determinant.")
                    })
                },
                SymmetryTransformationKind::SpinSpatial => |op, det| {
                    det.sym_transform_spin_spatial(op).unwrap_or_else(|err| {
                        log::error!("{err}");
                        panic!("Unable to apply `{op}` spin-spatially on the origin determinant.")
                    })
                },
            },
        )
    }
}

// ~~~~~~~~~~~
// RepAnalysis
// ~~~~~~~~~~~

impl<'a, G, T> RepAnalysis<G, SlaterDeterminant<'a, T>, T>
    for SlaterDeterminantSymmetryOrbit<'a, G, T>
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
    SlaterDeterminant<'a, T>: SymmetryTransformable,
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

    #[must_use]
    fn analyse_rep(
        &self,
    ) -> Result<
        <<G as CharacterProperties>::CharTab as SubspaceDecomposable<T>>::Decomposition,
        DecompositionError,
    > {
        let nelectrons_float = self.origin().nelectrons();
        if approx::relative_eq!(
            nelectrons_float.round(),
            nelectrons_float,
            epsilon = self.origin().threshold,
            max_relative = self.origin().threshold
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
                    SymmetryTransformationKind::Spin | SymmetryTransformationKind::SpinSpatial => {
                        match self.group().group_type() {
                            GroupType::Ordinary(_) => (true, String::new()),
                            GroupType::MagneticGrey(_) | GroupType::MagneticBlackWhite(_) => {
                                (!self.group().unitary_represented(),
                                format!("Unitary-represented magnetic groups cannot be used for symmetry analysis of odd-electron systems."))
                            }
                        }
                    }
                }
            };

            if valid_symmetry {
                let chis = self.calc_characters();
                let res = self.group().character_table().reduce_characters(
                    &chis.iter().map(|(cc, chi)| (cc, *chi)).collect::<Vec<_>>(),
                    self.integrality_threshold(),
                );
                res
            } else {
                Err(DecompositionError(err_str))
            }
        } else {
            Err(DecompositionError(format!(
                "Symmetry analysis for determinant with non-integer number of electrons `{:.7}` not supported.",
                nelectrons_float
            )))
        }
    }
}
