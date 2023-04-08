use std::error::Error;
use std::fmt;

use ndarray::{Array2, Ix0, Ix2};
use ndarray_linalg::{types::Lapack, solve::Inverse};
use ndarray_einsum_beta::*;
use num_complex::ComplexFloat;

use crate::chartab::chartab_symbols::ReducibleLinearSpaceSymbol;
use crate::group::GroupProperties;

#[derive(Debug, Clone)]
pub struct RepAnalysisError(pub String);

impl fmt::Display for RepAnalysisError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Representation analysis error: {}.", self.0)
    }
}

impl Error for RepAnalysisError {}

pub trait Overlap<T> where
    T: ComplexFloat + fmt::Debug + Lapack,
{
    fn complex_symmetric(&self) -> bool;

    fn overlap(&self, other: &Self, metric: &Array2<T>) -> Result<T, RepAnalysisError>;
}


pub trait Orbit<G, I> where
    G: GroupProperties,
{
    type OrbitIntoIter: IntoIterator<Item = I>;

    fn group(&self) -> &G;

    fn origin(&self) -> &I;

    fn orbit(&self) -> Self::OrbitIntoIter;
}

pub(crate) trait RepAnalysis<G, I, T>: Orbit<G, I>
where
    T: ComplexFloat + Lapack + fmt::Debug,
    G: GroupProperties,
    I: Overlap<T>,
    Self::OrbitIntoIter: IntoIterator<Item = I>,
{
    fn smat(&self) -> &Array2<T>;

    fn xmat(&self) -> &Array2<T>;

    fn tmat(&self, op: &G::GroupElement) -> Array2<T>;

    // ----------------
    // Provided methods
    // ----------------

    #[must_use]
    fn calc_dmat(&self, op: &G::GroupElement) -> Array2<T> {
        let complex_symmetric = self.origin().complex_symmetric();
        let xmath = if complex_symmetric {
            self.xmat().t().to_owned()
        } else {
            self.xmat().t().mapv(|x| x.conj())
        };
        let smattilde = xmath.dot(self.smat()).dot(self.xmat());
        let smattilde_inv = smattilde.inv().expect("The inverse of S~ could not be found.");
        let dmat = einsum(
            "ij,jk,kl,lm->im",
            &[&smattilde_inv, &xmath, &self.tmat(op), self.xmat()],
        )
        .expect("Unable to compute the matrix product (S~)^(-1) X† T X.")
        .into_dimensionality::<Ix2>()
        .expect("Unable to convert the matrix product (S~)^(-1) X† T X to two dimensions.");
        dmat
    }

    #[must_use]
    fn calc_character(&self, op: &G::GroupElement) -> T {
        let complex_symmetric = self.origin().complex_symmetric();
        let xmath = if complex_symmetric {
            self.xmat().t().to_owned()
        } else {
            self.xmat().t().mapv(|x| x.conj())
        };
        let smattilde = xmath.dot(self.smat()).dot(self.xmat());
        let smattilde_inv = smattilde.inv().expect("The inverse of S~ could not be found.");
        let dmat = einsum(
            "ij,jk,kl,li",
            &[&smattilde_inv, &xmath, &self.tmat(op), self.xmat()],
        )
        .expect("Unable to compute the trace of the matrix product (S~)^(-1) X† T X.")
        .into_dimensionality::<Ix0>()
        .expect("Unable to convert the trace of the matrix product (S~)^(-1) X† T X to zero dimensions.");
        *dmat
            .iter()
            .next()
            .expect("Unable to extract the character from the representation matrix.")
    }
}

pub trait RepAnalysisResult<G, R>
where
    G: GroupProperties,
    R: ReducibleLinearSpaceSymbol,
{
    fn actual_group(&self) -> &G;

    fn finite_group(&self) -> &G;

    fn representation(&self) -> &R;
}
