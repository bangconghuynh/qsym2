use log;
use std::fmt;

use approx;
use derive_builder::Builder;
use itertools::Itertools;
use ndarray::{s, Array2};
use ndarray_linalg::{solve::Determinant, types::Lapack};
use num_complex::ComplexFloat;
use num_traits::Zero;

use crate::analysis::{Orbit, Overlap, RepAnalysis, RepAnalysisError};
use crate::angmom::spinor_rotation_3d::SpinConstraint;
use crate::symmetry::symmetry_group::SymmetryGroupProperties;
use crate::symmetry::symmetry_transformation::SymmetryTransformable;
use crate::target::determinant::SlaterDeterminant;

// -------
// Overlap
// -------

impl<'a, T> Overlap<T> for SlaterDeterminant<'a, T>
where
    T: Lapack + ComplexFloat + fmt::Debug,
    <T as ComplexFloat>::Real: fmt::Debug + approx::RelativeEq<<T as ComplexFloat>::Real>,
{
    fn complex_symmetric(&self) -> bool {
        self.complex_symmetric
    }

    fn overlap(&self, other: &Self, metric: &Array2<T>) -> Result<T, RepAnalysisError> {
        assert_eq!(self.spin_constraint, other.spin_constraint);
        assert_eq!(self.coefficients.len(), other.coefficients.len());
        let sao = metric;

        let ov = self
            .coefficients
            .iter()
            .zip(other.coefficients.iter())
            .map(|(cw, cx)| {
                let mo_ov_mat = if self.complex_symmetric() {
                    cw.t().dot(sao).dot(cx)
                } else {
                    cw.t().mapv(|x| x.conj()).dot(sao).dot(cx)
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
struct SlaterDeterminantSpatialSymmetryOrbit<'a, G, T>
where
    G: SymmetryGroupProperties,
    T: ComplexFloat + fmt::Debug + Lapack,
    SlaterDeterminant<'a, T>: SymmetryTransformable,
{
    group: &'a G,

    origin: &'a SlaterDeterminant<'a, T>,

    #[builder(setter(skip), default = "None")]
    smat: Option<Array2<T>>,

    #[builder(setter(skip), default = "None")]
    xmat: Option<Array2<T>>,
}

impl<'a, G, T> SlaterDeterminantSpatialSymmetryOrbit<'a, G, T>
where
    G: SymmetryGroupProperties,
    T: ComplexFloat + Lapack + fmt::Debug,
    <T as ComplexFloat>::Real: fmt::Debug + approx::RelativeEq<<T as ComplexFloat>::Real>,
    SlaterDeterminant<'a, T>: SymmetryTransformable,
{
    fn calc_smat(&mut self, metric: &Array2<T>) {
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
            });
        self.smat = Some(smat)
    }
}

// impl<'a, G, T> SlaterDeterminantSpatialSymmetryOrbit<'a, G, T>
// where
//     fn calc_xmat(&mut self, preserves_full_rank: bool) {
//         let thresh = self.origin.threshold;
//         let smat = self.smat.as_ref().expect("No overlap matrix found for this orbit.");
//         let (eigvals, eigvecs) = smat.eig().unwrap();
//         // let smat_na = DMatrix::from_row_iterator(
//         //     smat.nrows(),
//         //     smat.ncols(),
//         //     smat.into_iter(),
//         // );

//     }
// }

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
    T: ComplexFloat + Lapack + fmt::Debug,
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
