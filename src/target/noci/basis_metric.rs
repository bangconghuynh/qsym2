//! Traits for defining metrics in bases.

use std::fmt;

use anyhow::{self, format_err};
use itertools::{Itertools, iproduct, izip};
use ndarray::{Array2, Array3, Axis, Ix2, s};
use ndarray_linalg::solve::Determinant;
use ndarray_linalg::types::Lapack;
use num_complex::ComplexFloat;
use rayon::prelude::*;

use crate::analysis::{Orbit, Overlap};
use crate::angmom::spinor_rotation_3d::StructureConstraint;
use crate::symmetry::symmetry_element::SpecialSymmetryTransformation;
use crate::symmetry::symmetry_group::SymmetryGroupProperties;
use crate::symmetry::symmetry_transformation::{SymmetryTransformable, SymmetryTransformationKind};
use crate::target::noci::basis::basis_orbit::BasisSymmetryOrbit;
use crate::target::noci::basis::{Basis, EagerBasis, FCIBasis, OrbitBasis};

// ========================
// BasisSymmetryOrbitMetric
// ========================

// -----------------
// Trait definitions
// -----------------

/// Trait specifying the symmetry-orbit metric for bases of Slater determinants.
///
/// For a group $`\mathcal{G}`$, this metric is defined as
///
/// ```math
/// \braket{\hat{g} \mathbf{w} | \mathbf{x}}
/// ```
///
/// where $`g \in \mathcal{G}`$, and $`\mathbf{w}`$ and $`\mathbf{x}`$ are members of the basis.
pub trait BasisSymmetryOrbitMetric<G, T>: Basis
where
    G: SymmetryGroupProperties,
    Self: SymmetryTransformable,
{
    /// Computes the basis symmetry-orbit metric for this basis.
    ///
    /// For a group $`\mathcal{G}`$, this metric is defined as
    ///
    /// ```math
    /// \braket{\hat{g} \mathbf{w} | \mathbf{x}}
    /// ```
    ///
    /// where $`g \in \mathcal{G}`$, and $`\mathbf{w}`$ and $`\mathbf{x}`$ are members of the basis.
    ///
    /// # Arguments
    ///
    /// * `group` - The group $`\mathcal{G}`$.
    /// * `symmetry_transformation_kind` - Enum specifying how the elements of $`\mathcal{G}`$ act
    ///   on the elements of the basis.
    /// * `metric` - The atomic-orbital overlap matrix with respect to the conventional sesquilinear
    ///   inner product.
    /// * `metric_h` - The atomic-orbital overlap matrix with respect to the bilinear inner product.
    ///
    /// # Returns
    ///
    /// The basis symmetry-orbit metric as a rank-3 array indexed by $`g`$, $`\mathbf{w}`$, and
    /// $`\mathbf{x}`$.
    fn basis_symmetry_orbit_metric(
        &self,
        group: &G,
        symmetry_transformation_kind: &SymmetryTransformationKind,
        metric: Option<&Array2<T>>,
        metric_h: Option<&Array2<T>>,
    ) -> Result<Array3<T>, anyhow::Error>;
}

// ---------------------
// Trait implementations
// ---------------------

// ~~~~~~~~~~
// OrbitBasis
// ~~~~~~~~~~

impl<'g, G, I, T> BasisSymmetryOrbitMetric<G, T> for OrbitBasis<'g, G, I>
where
    G: SymmetryGroupProperties + Clone,
    I: Overlap<T, Ix2> + Clone,
    T: ComplexFloat + fmt::Debug + Lapack,
    OrbitBasis<'g, G, I>: SymmetryTransformable,
{
    /// Computes the basis symmetry-orbit metric for this orbit basis.
    ///
    /// Since this orbit basis already contains information about its group and group action,
    /// `_group` and `_symmetry_transformation_kind` will be ignored.
    ///
    /// Given that the associated group of the orbit basis is $`\mathcal{G}`$ and that its origins
    /// are $`\{ \mathbf{w}_{I} \}`$, the metric is defined as
    ///
    /// ```math
    /// \braket{\hat{g}_k \hat{g}_i \mathbf{w}_I | \hat{g}_j \mathbf{w}_J},
    /// ```
    ///
    /// arranged in a rank-3 array indexed by $`k`$, $`iI`$, and $`jJ`$.
    ///
    /// # Arguments
    ///
    /// * `metric` - The atomic-orbital overlap matrix with respect to the conventional sesquilinear
    ///   inner product.
    /// * `metric_h` - The atomic-orbital overlap matrix with respect to the bilinear inner product.
    ///
    /// # Returns
    ///
    /// The basis symmetry-orbit metric as a rank-3 array indexed as described above.
    fn basis_symmetry_orbit_metric(
        &self,
        _group: &G,
        _symmetry_transformation_kind: &SymmetryTransformationKind,
        metric: Option<&Array2<T>>,
        metric_h: Option<&Array2<T>>,
    ) -> Result<Array3<T>, anyhow::Error> {
        log::debug!("Computing basis symmetry-orbit metric for `OrbitBasis`...");
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
            Ok(Array3::from_shape_vec(
                (order, order * n_origins, order * n_origins),
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
            Ok(Array3::from_shape_vec(
                (order, order * n_origins, order * n_origins),
                ukipwjpx_vec,
            )?)
        }
    }
}

// ~~~~~~~~~~
// EagerBasis
// ~~~~~~~~~~

impl<G, I, T> BasisSymmetryOrbitMetric<G, T> for EagerBasis<I>
where
    G: SymmetryGroupProperties + Clone,
    I: Clone + Overlap<T, Ix2> + Sync + Send,
    T: ComplexFloat + fmt::Debug + Lapack + Sync + Send,
    <T as ComplexFloat>::Real: Sync + Send,
    EagerBasis<I>: SymmetryTransformable,
{
    fn basis_symmetry_orbit_metric(
        &self,
        group: &G,
        symmetry_tranformation_kind: &SymmetryTransformationKind,
        metric: Option<&Array2<T>>,
        metric_h: Option<&Array2<T>>,
    ) -> Result<Array3<T>, anyhow::Error> {
        log::debug!("Computing basis symmetry-orbit metric for `EagerBasis`...");
        let g_self_orbit = BasisSymmetryOrbit::builder()
            .group(group)
            .origin(self)
            .symmetry_transformation_kind(symmetry_tranformation_kind.clone())
            .build()?;
        let ovss_vec = g_self_orbit
            .iter()
            .map(|g_basis_res| {
                g_basis_res.and_then(|g_basis| {
                    let mut ovs_vec = g_basis
                        .elements
                        .iter()
                        .cartesian_product(self.elements.iter())
                        .enumerate()
                        .par_bridge()
                        .map(|(i, (g_item, item))| {
                            let ov = g_item.overlap(item, metric, metric_h)?;
                            Ok((i, ov))
                        })
                        .collect::<Result<Vec<_>, anyhow::Error>>()?;
                    ovs_vec.sort_by_key(|v| v.0);
                    Ok(ovs_vec.into_iter().map(|v| v.1).collect_vec())
                })
            })
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .flatten()
            .collect_vec();
        Ok(Array3::from_shape_vec(
            (group.order(), self.n_items(), self.n_items()),
            ovss_vec,
        )?)
    }
}

// ~~~~~~~~
// FCIBasis
// ~~~~~~~~

impl<'a, G, T, SC> BasisSymmetryOrbitMetric<G, T> for FCIBasis<'a, T, SC>
where
    G: SymmetryGroupProperties + Clone,
    T: ComplexFloat + Lapack + Sync + Send,
    <T as ComplexFloat>::Real: Sync + Send,
    SC: 'a + StructureConstraint + fmt::Display + Clone + PartialEq + Sync + Send,
    FCIBasis<'a, T, SC>: SymmetryTransformable,
{
    /// Computes the basis symmetry-orbit metric for this configuration-interaction basis.
    ///
    /// For a group $`\mathcal{G}`$ and a Slater-determinant reference $`\Psi_0`$, this metric is
    /// defined as
    ///
    /// ```math
    /// \braket{\hat{g} \Psi_I | \Psi_J}
    /// ```
    ///
    /// where $`g \in \mathcal{G}`$, and $`\Psi_I`$ and $`\Psi_J`$ are replacement Slater
    /// determinants constructed from $`\Psi_0`$.
    /// These determinants are specified by the occupation patterns with respect to the set of
    /// molecular orbitals defined by $`\Psi_0`$.
    ///
    /// # Arguments
    ///
    /// * `group` - The group $`\mathcal{G}`$.
    /// * `symmetry_transformation_kind` - Enum specifying how the elements of $`\mathcal{G}`$ act
    ///   on the elements of the basis.
    /// * `metric` - The atomic-orbital overlap matrix with respect to the conventional sesquilinear
    ///   inner product.
    /// * `metric_h` - The atomic-orbital overlap matrix with respect to the bilinear inner product.
    ///
    /// # Returns
    ///
    /// The basis symmetry-orbit metric as a rank-3 array indexed by $`g`$, $`I`$, and $`J`$.
    fn basis_symmetry_orbit_metric(
        &self,
        group: &G,
        symmetry_tranformation_kind: &SymmetryTransformationKind,
        metric: Option<&Array2<T>>,
        metric_h: Option<&Array2<T>>,
    ) -> Result<Array3<T>, anyhow::Error> {
        log::debug!("Computing basis symmetry-orbit metric for `FCIBasis`...");
        let g_self_orbit = BasisSymmetryOrbit::builder()
            .group(group)
            .origin(self)
            .symmetry_transformation_kind(symmetry_tranformation_kind.clone())
            .build()?;
        let sao = metric.ok_or_else(|| format_err!("No atomic-orbital metric found."))?;
        let sao_h = metric_h.unwrap_or(sao);

        let s_ref = self.reference();
        let thresh = s_ref.threshold();

        let ovss_vec = g_self_orbit
            .iter()
            .map(|g_self_res| {
                g_self_res.and_then(|g_self| {
                    let g_s_ref = g_self.reference();
                    let mo_ov_mats = g_s_ref
                        .coefficients()
                        .iter()
                        .zip(s_ref.coefficients().iter())
                        .map(|(cw, cx)| {
                            if s_ref.complex_symmetric() {
                                match (g_s_ref.complex_conjugated(), s_ref.complex_conjugated()) {
                                    (false, false) => cw.t().dot(sao_h).dot(cx),
                                    (true, false) => cw.t().dot(sao).dot(cx),
                                    (false, true) => cx.t().dot(sao).dot(cw),
                                    (true, true) => cw.t().dot(&sao_h.t()).dot(cx),
                                }
                            } else {
                                match (g_s_ref.complex_conjugated(), s_ref.complex_conjugated()) {
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

                    let mut ovs_vec = g_self
                        .occupation_patterns
                        .iter()
                        .cartesian_product(self.occupation_patterns.iter())
                        .enumerate()
                        .par_bridge()
                        .map(|(i, (g_s_occs, s_occs))| {
                            let ov = izip!(g_s_occs, s_occs, &mo_ov_mats)
                                .map(|(g_s_occ, s_occ, mo_ov_mat)| {
                                    let nonzero_g_s_occ =
                                        g_s_occ.iter().positions(|&occ| occ > thresh).collect_vec();
                                    let nonzero_s_occ =
                                        s_occ.iter().positions(|&occ| occ > thresh).collect_vec();
                                    let mo_ov_mat_occ = mo_ov_mat
                                        .select(Axis(0), &nonzero_g_s_occ)
                                        .select(Axis(1), &nonzero_s_occ);
                                    mo_ov_mat_occ.det().expect(
                                    "The determinant of the MO overlap matrix could not be found.",
                                )
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
                    ovs_vec.sort_by_key(|v| v.0);
                    let ovs_vec = ovs_vec.into_iter().map(|v| v.1).collect_vec();
                    Ok(ovs_vec)
                })
            })
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .flatten()
            .collect_vec();

        Ok(Array3::from_shape_vec(
            (group.order(), self.n_items(), self.n_items()),
            ovss_vec,
        )?)
    }
}
