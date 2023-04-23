use std::fmt;

use derive_builder::Builder;
use itertools::Itertools;
use nalgebra::ComplexField;
use ndarray::{s, Array2};
use num_complex::ComplexFloat;

use crate::analysis::{Orbit, RepAnalysis};
use crate::symmetry::symmetry_group::SymmetryGroupProperties;
use crate::symmetry::symmetry_transformation::SymmetryTransformable;

#[derive(Builder, Clone)]
struct DeterminantSpatialSymmetryOrbit<'a, G, T>
where
    G: SymmetryGroupProperties,
    T: ComplexField + ComplexFloat + fmt::Debug,
    Determinant<'a, T>: SymmetryTransformable,
{
    group: &'a G,

    origin: &'a Determinant<'a, T>,

    #[builder(setter(skip), default = "None")]
    smat: Option<Array2<T>>,

    #[builder(setter(skip), default = "None")]
    xmat: Option<Array2<T>>,
}

impl<'a, G, T> DeterminantSpatialSymmetryOrbit<'a, G, T>
where
    G: SymmetryGroupProperties,
    T: ComplexField + ComplexFloat + fmt::Debug,
    Determinant<'a, T>: SymmetryTransformable,
{
    fn calc_smat(&mut self) {
        let order = self.group.order();
        let mut smat = Array2::<T>::zeros((order, order));
        self.orbit()
            .iter()
            .enumerate()
            .combinations_with_replacement(2)
            .for_each(|pair| {
                let (w, det_w) = pair[0];
                let (x, det_x) = pair[1];
            });
    }
}

impl<'a, G, T> Orbit<G, Determinant<'a, T>> for DeterminantSpatialSymmetryOrbit<'a, G, T>
where
    G: SymmetryGroupProperties,
    T: ComplexField + ComplexFloat + fmt::Debug,
    Determinant<'a, T>: SymmetryTransformable,
{
    type OrbitIntoIter = Vec<Determinant<'a, T>>;

    fn group(&self) -> &G {
        self.group
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

impl<'a, G, T> RepAnalysis<G, Determinant<'a, T>, T> for DeterminantSpatialSymmetryOrbit<'a, G, T>
where
    G: SymmetryGroupProperties,
    T: ComplexField + ComplexFloat + fmt::Debug,
    Determinant<'a, T>: SymmetryTransformable,
{
    fn smat(&self) -> &Array2<T> {
        self.smat.as_ref().expect("Orbit overlap matrix not found.")
    }

    fn xmat(&self) -> &Array2<T> {
        self.xmat.as_ref().expect("Orbit overlap orthogonalisation matrix not found.")
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
