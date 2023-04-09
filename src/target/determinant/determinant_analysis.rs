use log;
use std::fmt;
use std::ops::Mul;

use approx;
use derive_builder::Builder;
use itertools::{izip, Itertools};
use ndarray::{s, Array2, Axis};
use ndarray_linalg::{
    assert_close_l2,
    eig::Eig,
    eigh::Eigh,
    solve::Determinant,
    types::{Lapack, Scalar},
    UPLO,
};
use num_complex::{Complex, ComplexFloat};
use num_traits::{Float, Zero};

use crate::analysis::{Orbit, Overlap, RepAnalysis, RepAnalysisError};
use crate::angmom::spinor_rotation_3d::SpinConstraint;
use crate::aux::misc::complex_gram_schmidt_orthonormalisation;
use crate::chartab::SubspaceDecomposable;
use crate::symmetry::symmetry_group::SymmetryGroupProperties;
use crate::symmetry::symmetry_transformation::SymmetryTransformable;
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
    <T as ComplexFloat>::Real: fmt::Debug + approx::RelativeEq<<T as ComplexFloat>::Real>,
{
    fn complex_symmetric(&self) -> bool {
        self.complex_symmetric
    }

    fn overlap(&self, other: &Self, metric: &Array2<T>) -> Result<T, RepAnalysisError> {
        assert_eq!(self.spin_constraint, other.spin_constraint);
        assert_eq!(self.coefficients.len(), other.coefficients.len());
        let sao = metric;
        let thresh = Float::sqrt(self.threshold * other.threshold);

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
            // let cw_o = cw.dot(&Array2::from_diag(
            //     &occw.mapv(|x| T::from_real(Float::sqrt(x))),
            // ));
            // let cx_o = cx.dot(&Array2::from_diag(
            //     &occx.mapv(|x| T::from_real(Float::sqrt(x))),
            // ));
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

#[derive(Builder, Clone)]
pub struct SlaterDeterminantSpatialSymmetryOrbit<'a, G, T>
where
    G: SymmetryGroupProperties,
    T: ComplexFloat + fmt::Debug + Lapack,
    SlaterDeterminant<'a, T>: SymmetryTransformable,
{
    group: &'a G,

    origin: &'a SlaterDeterminant<'a, T>,

    #[builder(setter(skip), default = "None")]
    pub smat: Option<Array2<T>>,

    #[builder(setter(skip), default = "None")]
    pub xmat: Option<Array2<T>>,
}

impl<'a, G, T> SlaterDeterminantSpatialSymmetryOrbit<'a, G, T>
where
    G: SymmetryGroupProperties + Clone,
    T: ComplexFloat + fmt::Debug + Lapack,
    SlaterDeterminant<'a, T>: SymmetryTransformable,
{
    pub fn builder() -> SlaterDeterminantSpatialSymmetryOrbitBuilder<'a, G, T> {
        SlaterDeterminantSpatialSymmetryOrbitBuilder::default()
    }
}

impl<'a, G, T> SlaterDeterminantSpatialSymmetryOrbit<'a, G, T>
where
    G: SymmetryGroupProperties,
    T: ComplexFloat + Lapack + fmt::Debug,
    <T as ComplexFloat>::Real: fmt::Debug + approx::RelativeEq<<T as ComplexFloat>::Real>,
    SlaterDeterminant<'a, T>: SymmetryTransformable + Overlap<T>,
{
    pub fn calc_smat(&mut self, metric: &Array2<T>) -> &mut Self {
        let order = self.group.order();
        let mut smat = Array2::<T>::zeros((order, order));
        self.orbit()
            .iter()
            .enumerate()
            .combinations_with_replacement(2)
            .for_each(|pair| {
                let (w, det_w) = pair[0];
                let (x, det_x) = pair[1];
                smat[(w, x)] = det_w.overlap(&det_x, metric).unwrap_or_else(|err| {
                    log::error!("{err}");
                    panic!("Unable to calculate the overlap between determinants `{w}` and `{x}`.");
                });
                if w != x {
                    smat[(x, w)] = det_x.overlap(&det_w, metric).unwrap_or_else(|err| {
                        log::error!("{err}");
                        panic!(
                            "Unable to calculate the overlap between determinants `{x}` and `{w}`."
                        );
                    });
                }
            });
        self.smat = Some(smat);
        self
    }
}

impl<'a, G> SlaterDeterminantSpatialSymmetryOrbit<'a, G, f64>
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

impl<'a, G, T> SlaterDeterminantSpatialSymmetryOrbit<'a, G, Complex<T>>
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
        let nonzero_umat = complex_gram_schmidt_orthonormalisation(
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

impl<'a, G, T> Orbit<G, SlaterDeterminant<'a, T>>
    for SlaterDeterminantSpatialSymmetryOrbit<'a, G, T>
where
    G: SymmetryGroupProperties,
    T: ComplexFloat + fmt::Debug + Lapack,
    SlaterDeterminant<'a, T>: SymmetryTransformable,
{
    type OrbitIntoIter = Vec<SlaterDeterminant<'a, T>>;

    fn group(&self) -> &G {
        self.group
    }

    fn origin(&self) -> &SlaterDeterminant<'a, T> {
        self.origin
    }

    fn orbit(&self) -> Self::OrbitIntoIter {
        self.group
            .elements()
            .clone()
            .into_iter()
            .map(|op| self.origin.sym_transform_spatial(&op).unwrap())
            .collect::<Vec<_>>()
    }
}

impl<'a, G, T> RepAnalysis<G, SlaterDeterminant<'a, T>, T>
    for SlaterDeterminantSpatialSymmetryOrbit<'a, G, T>
where
    G: SymmetryGroupProperties,
    G::CharTab: SubspaceDecomposable<T>,
    T: Lapack
        + ComplexFloat<Real = <T as Scalar>::Real>
        + fmt::Debug
        + Mul<<T as ComplexFloat>::Real, Output = T>,
    <T as ComplexFloat>::Real: fmt::Debug + Zero + approx::RelativeEq<<T as ComplexFloat>::Real>,
    SlaterDeterminant<'a, T>: SymmetryTransformable,
{
    fn smat(&self) -> &Array2<T> {
        self.smat.as_ref().expect("Orbit overlap matrix not found.")
    }

    fn xmat(&self) -> &Array2<T> {
        self.xmat
            .as_ref()
            .expect("Orbit overlap orthogonalisation matrix not found.")
    }

    fn tmat(&self, op: &G::GroupElement) -> Array2<T> {
        let ctb = self
            .group
            .cayley_table()
            .expect("The Cayley table for the group cannot be found.");
        let i = self.group.get_index_of(op).unwrap_or_else(|| {
            panic!("Unable to retrieve the index of element `{op}` in the group.")
        });
        let order = self.group.order();
        let mut twx = Array2::<T>::zeros((order, order));
        for x in 0..order {
            let ix = ctb[(i, x)];
            let ixinv = ctb
                .slice(s![.., ix])
                .iter()
                .position(|&z| z == 0)
                .unwrap_or_else(|| panic!("The inverse of element index `{ix}` cannot be found."));

            for w in 0..order {
                let ixinv_w = ctb[(ixinv, w)];
                twx[(w, x)] = self.smat()[(ixinv_w, 0)];
            }
        }
        twx
    }
}
