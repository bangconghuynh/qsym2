use std::error::Error;
use std::fmt;

use ndarray::{s, Array2, Ix0, Ix2};
use ndarray_einsum_beta::*;
use ndarray_linalg::{solve::Inverse, types::Lapack};
use num_complex::ComplexFloat;

use crate::chartab::chartab_group::CharacterProperties;
use crate::chartab::chartab_symbols::ReducibleLinearSpaceSymbol;
use crate::chartab::{DecompositionError, SubspaceDecomposable};
use crate::group::{class::ClassProperties, GroupProperties};

#[derive(Debug, Clone)]
pub struct RepAnalysisError(pub String);

impl fmt::Display for RepAnalysisError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Representation analysis error: {}.", self.0)
    }
}

impl Error for RepAnalysisError {}

pub trait Overlap<T>
where
    T: ComplexFloat + fmt::Debug + Lapack,
{
    fn complex_symmetric(&self) -> bool;

    fn overlap(&self, other: &Self, metric: &Array2<T>) -> Result<T, RepAnalysisError>;
}

pub trait Orbit<G, I>
where
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
    G: GroupProperties + ClassProperties + CharacterProperties,
    G::GroupElement: fmt::Display,
    G::CharTab: SubspaceDecomposable<T>,
    I: Overlap<T>,
    Self::OrbitIntoIter: IntoIterator<Item = I>,
{
    // ----------------
    // Required methods
    // ----------------

    fn smat(&self) -> &Array2<T>;

    fn xmat(&self) -> &Array2<T>;

    // ----------------
    // Provided methods
    // ----------------

    fn tmat(&self, op: &G::GroupElement) -> Array2<T> {
        let ctb = self
            .group()
            .cayley_table()
            .expect("The Cayley table for the group cannot be found.");
        let i = self.group().get_index_of(op).unwrap_or_else(|| {
            panic!("Unable to retrieve the index of element `{op}` in the group.")
        });
        let order = self.group().order();
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

    #[must_use]
    fn calc_dmat(&self, op: &G::GroupElement) -> Array2<T> {
        let complex_symmetric = self.origin().complex_symmetric();
        let xmath = if complex_symmetric {
            self.xmat().t().to_owned()
        } else {
            self.xmat().t().mapv(|x| x.conj())
        };
        let smattilde = xmath.dot(self.smat()).dot(self.xmat());
        let smattilde_inv = smattilde
            .inv()
            .expect("The inverse of S~ could not be found.");
        let dmat = einsum(
            "ij,jk,kl,lm->im",
            &[&smattilde_inv, &xmath, &self.tmat(op), self.xmat()],
        )
        .expect("Unable to compute the matrix product [(S~)^(-1) X† T X].")
        .into_dimensionality::<Ix2>()
        .expect("Unable to convert the matrix product [(S~)^(-1) X† T X] to two dimensions.");
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
        let smattilde_inv = smattilde
            .inv()
            .expect("The inverse of S~ could not be found.");
        let chi = einsum(
            "ij,jk,kl,li",
            &[&smattilde_inv, &xmath, &self.tmat(op), self.xmat()],
        )
        .expect("Unable to compute the trace of the matrix product [(S~)^(-1) X† T X].")
        .into_dimensionality::<Ix0>()
        .expect("Unable to convert the trace of the matrix product [(S~)^(-1) X† T X] to zero dimensions.");
        *chi.iter()
            .next()
            .expect("Unable to extract the character from the representation matrix.")
    }

    #[must_use]
    fn calc_characters(&self) -> Vec<(<G as ClassProperties>::ClassSymbol, T)> {
        let complex_symmetric = self.origin().complex_symmetric();
        let xmath = if complex_symmetric {
            self.xmat().t().to_owned()
        } else {
            self.xmat().t().mapv(|x| x.conj())
        };
        let smattilde = xmath.dot(self.smat()).dot(self.xmat());
        let smattilde_inv = smattilde
            .inv()
            .expect("The inverse of S~ could not be found.");
        let chis = (0..self.group().class_number()).map(|cc_i| {
            let cc = self.group().get_cc_symbol_of_index(cc_i).unwrap();
            let op = self.group().get_cc_transversal(cc_i).unwrap();
            let chi = einsum(
                "ij,jk,kl,li",
                &[&smattilde_inv, &xmath, &self.tmat(&op), self.xmat()],
            )
            .expect("Unable to compute the trace of the matrix product (S~)^(-1) X† T X.")
            .into_dimensionality::<Ix0>()
            .expect("Unable to convert the trace of the matrix product (S~)^(-1) X† T X to zero dimensions.");
            (cc, *chi
                .iter()
                .next()
                .expect("Unable to extract the character from the representation matrix."))
        }).collect::<Vec<_>>();
        chis
    }

    fn analyse_rep(
        &self,
    ) -> Result<
        <<G as CharacterProperties>::CharTab as SubspaceDecomposable<T>>::Decomposition,
        DecompositionError,
    > {
        let chis = self.calc_characters();
        let res = self.group().character_table().reduce_characters(
            &chis.iter().map(|(cc, chi)| (cc, *chi)).collect::<Vec<_>>(),
            1e-7,
        );
        res
    }
}
