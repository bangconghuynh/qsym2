//! Trait for defining a metric in the bases for configuration interaction of Slater determinants.

use std::fmt;

use anyhow::{self, ensure, format_err};
use itertools::{Itertools, iproduct, izip};
use ndarray::{Array, Array2, Array3, Array5, Axis, Dimension, Ix5, s};
use ndarray_linalg::solve::Determinant;
use ndarray_linalg::types::Lapack;
use num_complex::ComplexFloat;
use num_traits::Float;
use numpy::Ix2;
use rayon::prelude::*;

use crate::analysis::Overlap;
use crate::angmom::spinor_rotation_3d::StructureConstraint;
use crate::group::GroupProperties;
use crate::symmetry::symmetry_element::SpecialSymmetryTransformation;
use crate::target::noci::basis::{Basis, EagerBasis, FCIBasis, OrbitBasis};

// =====
// Basis
// =====

// -----------------
// Trait definitions
// -----------------

/// Trait defining behaviours of a basis consisting of linear-space items.
pub trait BasisMetric<T, D>: Basis
where
    D: Dimension,
{
    /// Computes the metric between this basis and another.
    fn basis_metric(
        &self,
        other: Option<&Self>,
        metric: Option<&Array2<T>>,
        metric_h: Option<&Array2<T>>,
    ) -> Result<Array<T, D>, anyhow::Error>;
}

// ---------------------
// Trait implementations
// ---------------------

// ~~~~~~~~~~
// OrbitBasis
// ~~~~~~~~~~

impl<'g, G, I, T> BasisMetric<T, Ix5> for OrbitBasis<'g, G, I>
where
    G: GroupProperties + Clone,
    G::GroupElement: SpecialSymmetryTransformation,
    I: Overlap<T, Ix2> + Clone,
    T: ComplexFloat + fmt::Debug + Lapack + Sync + Send,
{
    /// Computes the metric (*i.e.* the overlap matrix between the basis elements) between this
    /// eager basis and another.
    ///
    /// # Arguments
    ///
    /// * `other` - Another orbit basis.
    /// * `metric` - The atomic-orbital overlap matrix with respect to the conventional sesquilinear
    ///   inner product.
    /// * `metric_h` - The atomic-orbital overlap matrix with respect to the bilinear inner product.
    ///
    /// # Returns
    ///
    /// The overmap matrix between the basis elements.
    fn basis_metric(
        &self,
        _other: Option<&OrbitBasis<'g, G, I>>,
        metric: Option<&Array2<T>>,
        metric_h: Option<&Array2<T>>,
    ) -> Result<Array5<T>, anyhow::Error> {
        let order = self.group.order();
        let origins = self.origins();
        let n_origins = origins.len();

        if let Some(ctb) = self.group.cayley_table() {
            log::debug!(
                "Cayley table available. Group closure will be used to speed up `OrbitBasis` metric computation."
            );
            let detov_kpwx_vec = self
                .iter()
                .collect::<Result<Vec<_>, _>>()?
                .into_iter()
                .cartesian_product(origins.iter())
                .map(|(w, x)| w.overlap(x, metric, metric_h))
                .collect::<Result<Vec<_>, _>>()?;
            let detov_kpwx = Array3::from_shape_vec((order, n_origins, n_origins), detov_kpwx_vec)?;

            let norm_preserving_scalar_map = |i, v: T| {
                if origins.iter().any(|origin| origin.complex_symmetric()) {
                    Err(format_err!(
                        "`norm_preserving_scalar_map` is currently not implemented for complex-symmetric overlaps. This thus precludes the use of the Cayley table to speed up the computation of the orbit overlap matrix."
                    ))
                } else if self
                    .group
                    .get_index(i)
                    .unwrap_or_else(|| panic!("Group operation index `{i}` not found."))
                    .contains_time_reversal()
                {
                    Ok(v.conj())
                } else {
                    Ok(v)
                }
            };
            let ukipwjpx_vec = [
                (0..order),
                (0..order),
                (0..n_origins),
                (0..order),
                (0..n_origins),
            ]
            .into_iter()
            .multi_cartesian_product()
            .map(|v| {
                let k = v[0];
                let ip = v[1];
                let w = v[2];
                let jp = v[3];
                let x = v[4];

                let jpinv =
                    ctb.slice(s![.., jp])
                        .iter()
                        .position(|&x| x == 0)
                        .ok_or(format_err!(
                            "Unable to find the inverse of group element `{jp}`."
                        ))?;
                let kp = ctb[(jpinv, ctb[(k, ip)])];
                norm_preserving_scalar_map(jp, detov_kpwx[(kp, w, x)])
            })
            .collect::<Result<Vec<_>, _>>()?;
            Ok(Array5::from_shape_vec(
                (order, order, n_origins, order, n_origins),
                ukipwjpx_vec,
            )?)
        } else {
            let ukipwjpx_vec = iproduct!(
                self.group.elements().clone().into_iter(),
                self.iter().collect::<Result<Vec<_>, _>>()?.into_iter(),
                self.iter().collect::<Result<Vec<_>, _>>()?.into_iter()
            )
            .map(|(gk, gipww, gjpwx)| {
                let gkgipww = (self.action)(&gk, &gipww)?;
                gkgipww.overlap(&gjpwx, metric, metric_h)
            })
            .collect::<Result<Vec<_>, _>>()?;
            Ok(Array5::from_shape_vec(
                (order, order, n_origins, order, n_origins),
                ukipwjpx_vec,
            )?)
        }
    }
}

// ~~~~~~~~~~
// EagerBasis
// ~~~~~~~~~~

impl<I, T> BasisMetric<T, Ix2> for EagerBasis<I>
where
    I: Clone + Overlap<T, Ix2> + Sync + Send,
    T: ComplexFloat + fmt::Debug + Lapack + Sync + Send,
{
    /// Computes the metric (*i.e.* the overlap matrix between the basis elements) between this
    /// eager basis and another.
    ///
    /// # Arguments
    ///
    /// * `other` - Another FCI basis.
    /// * `metric` - The atomic-orbital overlap matrix with respect to the conventional sesquilinear
    ///   inner product.
    /// * `metric_h` - The atomic-orbital overlap matrix with respect to the bilinear inner product.
    ///
    /// # Returns
    ///
    /// The overmap matrix between the basis elements.
    fn basis_metric(
        &self,
        other: Option<&EagerBasis<I>>,
        metric: Option<&Array2<T>>,
        metric_h: Option<&Array2<T>>,
    ) -> Result<Array2<T>, anyhow::Error> {
        let other = if let Some(o) = other { o } else { self };
        let mut ovs_vec = self
            .elements
            .iter()
            .cartesian_product(other.elements.iter())
            .enumerate()
            .par_bridge()
            .map(|(i, (s_item, o_item))| {
                let ov = s_item.overlap(o_item, metric, metric_h)?;
                Ok((i, ov))
            })
            .collect::<Result<Vec<_>, anyhow::Error>>()?;
        ovs_vec.sort_by_key(|p| p.0);
        let ovs_vec = ovs_vec.into_iter().map(|(_, ov)| ov).collect_vec();

        Ok(Array2::from_shape_vec(
            (self.n_items(), other.n_items()),
            ovs_vec,
        )?)
    }
}

// ~~~~~~~~
// FCIBasis
// ~~~~~~~~

impl<'a, T, SC> BasisMetric<T, Ix2> for FCIBasis<'a, T, SC>
where
    T: ComplexFloat + Lapack + Sync + Send,
    <T as ComplexFloat>::Real: Sync + Send,
    SC: 'a + StructureConstraint + fmt::Display + Clone + PartialEq + Sync + Send,
{
    /// Computes the FCI metric (*i.e.* the overlap matrix between the basis elements) between this
    /// FCI basis and another.
    ///
    /// # Arguments
    ///
    /// * `other` - Another FCI basis.
    /// * `metric` - The atomic-orbital overlap matrix with respect to the conventional sesquilinear
    ///   inner product.
    /// * `metric_h` - The atomic-orbital overlap matrix with respect to the bilinear inner product.
    ///
    /// # Returns
    ///
    /// The overmap matrix between the basis elements.
    fn basis_metric(
        &self,
        other: Option<&FCIBasis<'a, T, SC>>,
        metric: Option<&Array2<T>>,
        metric_h: Option<&Array2<T>>,
    ) -> Result<Array2<T>, anyhow::Error> {
        let other = if let Some(o) = other { o } else { self };
        let sao = metric.ok_or_else(|| format_err!("No atomic-orbital metric found."))?;
        let sao_h = metric_h.unwrap_or(sao);

        let s_ref = self.reference();
        let o_ref = other.reference();
        let thresh = Float::sqrt(s_ref.threshold() * o_ref.threshold());

        ensure!(
            s_ref.structure_constraint() == o_ref.structure_constraint(),
            "Inconsistent structure constraints between the two FCI bases."
        );
        ensure!(
            s_ref.coefficients().len() == o_ref.coefficients().len(),
            "Inconsistent numbers of coefficient matrices between the references of the two FCI bases."
        );
        ensure!(
            s_ref.baos() == o_ref.baos(),
            "Inconsistent basis angular order between the two FCI bases."
        );
        ensure!(
            s_ref.complex_symmetric() == s_ref.complex_symmetric(),
            "Inconsistent `complex_symmetric` between the two FCI bases."
        );

        let mo_ov_mats = s_ref
            .coefficients()
            .iter()
            .zip(o_ref.coefficients().iter())
            .map(|(cw, cx)| {
                if s_ref.complex_symmetric() {
                    match (s_ref.complex_conjugated(), o_ref.complex_conjugated()) {
                        (false, false) => cw.t().dot(sao_h).dot(cx),
                        (true, false) => cw.t().dot(sao).dot(cx),
                        (false, true) => cx.t().dot(sao).dot(cw),
                        (true, true) => cw.t().dot(&sao_h.t()).dot(cx),
                    }
                } else {
                    match (s_ref.complex_conjugated(), o_ref.complex_conjugated()) {
                        (false, false) => cw.t().mapv(|x| x.conj()).dot(sao).dot(cx),
                        (true, false) => cw.t().mapv(|x| x.conj()).dot(sao_h).dot(cx),
                        (false, true) => cx
                            .t()
                            .mapv(|x| x.conj())
                            .dot(sao_h)
                            .dot(cw)
                            .mapv(|x| x.conj()),
                        (true, true) => cw.t().mapv(|x| x.conj()).dot(&sao.t()).dot(cx),
                    }
                }
            })
            .collect_vec();

        let mut ovs_vec = self
            .occupation_patterns
            .iter()
            .cartesian_product(other.occupation_patterns.iter())
            .enumerate()
            .par_bridge()
            .map(|(i, (s_occs, o_occs))| {
                let ov = izip!(s_occs, o_occs, &mo_ov_mats)
                    .map(|(s_occ, o_occ, mo_ov_mat)| {
                        let nonzero_s_occ =
                            s_occ.iter().positions(|&occ| occ > thresh).collect_vec();
                        let nonzero_o_occ =
                            o_occ.iter().positions(|&occ| occ > thresh).collect_vec();
                        let mo_ov_mat_occ = mo_ov_mat
                            .select(Axis(0), &nonzero_s_occ)
                            .select(Axis(1), &nonzero_o_occ);
                        mo_ov_mat_occ
                            .det()
                            .expect("The determinant of the MO overlap matrix could not be found.")
                    })
                    .fold(T::one(), |acc, x| acc * x);

                let implicit_factor = s_ref.structure_constraint().implicit_factor()?;
                if implicit_factor > 1 {
                    let p_i32 = i32::try_from(implicit_factor)?;
                    Ok((i, ComplexFloat::powi(ov, p_i32)))
                } else {
                    Ok((i, ov))
                }
            })
            .collect::<Result<Vec<_>, anyhow::Error>>()?;
        ovs_vec.sort_by_key(|p| p.0);
        let ovs_vec = ovs_vec.into_iter().map(|(_, ov)| ov).collect_vec();

        Ok(Array2::from_shape_vec(
            (self.n_items(), other.n_items()),
            ovs_vec,
        )?)
    }
}
