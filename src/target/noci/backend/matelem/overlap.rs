use std::marker::PhantomData;

use anyhow::{self, format_err};
use derive_builder::Builder;
use itertools::Itertools;
use ndarray::{Array2, ArrayView2, Ix2};
use ndarray_linalg::Lapack;
use num_complex::ComplexFloat;

use crate::analysis::Overlap;
use crate::angmom::spinor_rotation_3d::StructureConstraint;
use crate::symmetry::symmetry_transformation::SymmetryTransformable;
use crate::target::determinant::SlaterDeterminant;

use super::OrbitMatrix;

/// Structure for managing the overlap integrals in an atomic-orbital basis.
#[derive(Builder)]
pub struct OverlapAO<'a, T, SC>
where
    T: ComplexFloat + Lapack,
    SC: StructureConstraint + Clone,
{
    /// The overlap integral in an atomic-orbital basis.
    sao: ArrayView2<'a, T>,

    /// The structure constraint for the wavefunctions on the Hilbert space with this overlap
    /// metric.
    #[builder(setter(skip), default = "PhantomData")]
    structure_constraint: PhantomData<SC>,
}

impl<'a, T, SC> OverlapAO<'a, T, SC>
where
    T: ComplexFloat + Lapack,
    SC: StructureConstraint + Clone,
{
    /// Returns a builder for [`OverlapAO`].
    pub fn builder() -> OverlapAOBuilder<'a, T, SC> {
        OverlapAOBuilder::<T, SC>::default()
    }

    /// Returns the overlap integrals in an atomic-orbital basis.
    pub fn sao(&'a self) -> &'a ArrayView2<'a, T> {
        &self.sao
    }
}

impl<'a, T, SC> OverlapAO<'a, T, SC>
where
    T: ComplexFloat + Lapack,
    SC: StructureConstraint + Clone + std::fmt::Display,
    for<'b> SlaterDeterminant<'b, T, SC>: Overlap<T, Ix2>,
{
    /// Calculates the overlap matrix element between two determinants.
    pub fn calc_overlap_matrix_element(
        &self,
        det_w: &SlaterDeterminant<T, SC>,
        det_x: &SlaterDeterminant<T, SC>,
    ) -> Result<T, anyhow::Error> {
        if det_w.complex_symmetric() != det_x.complex_symmetric() {
            return Err(format_err!(
                "The `complex_symmetric` booleans of the specified determinants do not match: `det_w` (`{}`) != `det_x` (`{}`).",
                det_w.complex_symmetric(),
                det_x.complex_symmetric(),
            ));
        }
        det_w.overlap(det_x, Some(&self.sao.to_owned()), None)
    }

    pub fn calc_overlap_matrix(
        &self,
        dets: &[&SlaterDeterminant<T, SC>],
    ) -> Result<Array2<T>, anyhow::Error> {
        let dim = dets.len();
        let mut smat = Array2::<T>::zeros((dim, dim));
        for pair in dets.iter().enumerate().combinations_with_replacement(2) {
            let (w, det_w) = &pair[0];
            let (x, det_x) = &pair[1];
            let ov_wx = self.calc_overlap_matrix_element(det_w, det_x)?;
            smat[(*w, *x)] = ov_wx;
            if *w != *x {
                let ov_xw = self.calc_overlap_matrix_element(det_x, det_w)?;
                smat[(*x, *w)] = ov_xw;
            }
        }
        Ok(smat)
    }
}

impl<'a, T, SC> OrbitMatrix<'a, T, SC> for &OverlapAO<'a, T, SC>
where
    T: ComplexFloat + Lapack,
    SC: StructureConstraint + Clone + std::fmt::Display,
    for<'b> SlaterDeterminant<'b, T, SC>: Overlap<T, Ix2>,
    SlaterDeterminant<'a, T, SC>: SymmetryTransformable,
{
    type MatrixElement = T;

    fn calc_matrix_element(
        &self,
        det_w: &SlaterDeterminant<T, SC>,
        det_x: &SlaterDeterminant<T, SC>,
        _sao: &ArrayView2<T>,
        _thresh_offdiag: <T as ComplexFloat>::Real,
        _thresh_zeroov: <T as ComplexFloat>::Real,
    ) -> Result<T, anyhow::Error> {
        self.calc_overlap_matrix_element(det_w, det_x)
    }

    fn t(x: &T) -> T {
        *x
    }

    fn conj(x: &T) -> T {
        <T as ComplexFloat>::conj(*x)
    }

    fn zero(&self) -> T {
        T::zero()
    }
}

impl<'a, T, SC> OrbitMatrix<'a, T, SC> for OverlapAO<'a, T, SC>
where
    T: ComplexFloat + Lapack,
    SC: StructureConstraint + Clone + std::fmt::Display,
    for<'b> SlaterDeterminant<'b, T, SC>: Overlap<T, Ix2>,
    SlaterDeterminant<'a, T, SC>: SymmetryTransformable,
{
    type MatrixElement = T;

    fn calc_matrix_element(
        &self,
        det_w: &SlaterDeterminant<T, SC>,
        det_x: &SlaterDeterminant<T, SC>,
        _sao: &ArrayView2<T>,
        _thresh_offdiag: <T as ComplexFloat>::Real,
        _thresh_zeroov: <T as ComplexFloat>::Real,
    ) -> Result<T, anyhow::Error> {
        (&self).calc_matrix_element(det_w, det_x, _sao, _thresh_offdiag, _thresh_zeroov)
    }

    fn t(x: &T) -> T {
        *x
    }

    fn conj(x: &T) -> T {
        <T as ComplexFloat>::conj(*x)
    }

    fn zero(&self) -> T {
        T::zero()
    }
}
