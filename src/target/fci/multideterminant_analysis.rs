//! Implementation of symmetry analysis for multi-determinantal wavefunctions.

use std::fmt;
use std::hash::Hash;
use std::ops::Mul;

use anyhow::{self, ensure, format_err};
use approx;
use itertools::Itertools;
use log;
use ndarray::{Array2, Array3, s};
use ndarray_linalg::types::{Lapack, Scalar};
use num_complex::ComplexFloat;
use num_traits::{ToPrimitive, Zero};

use crate::analysis::{Orbit, Overlap, RepAnalysis};
use crate::angmom::spinor_rotation_3d::StructureConstraint;
use crate::chartab::SubspaceDecomposable;
use crate::symmetry::symmetry_group::SymmetryGroupProperties;
use crate::symmetry::symmetry_transformation::SymmetryTransformable;
use crate::target::determinant::SlaterDeterminant;
use crate::target::fci::basis::FCIBasis;
use crate::target::noci::backend::solver::check_complex_matrix_symmetry;
use crate::target::noci::basis::Basis;
use crate::target::noci::multideterminant::MultiDeterminant;
use crate::target::noci::multideterminant::multideterminant_analysis::MultiDeterminantSymmetryOrbit;

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Optimised implementation for full-configuration-interaction multi-determinantal wavefunctions
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
impl<'a, 'go, 'g, G, T, SC> MultiDeterminantSymmetryOrbit<'a, 'go, G, T, FCIBasis<'a, T, SC>, SC>
where
    G: SymmetryGroupProperties + Clone,
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
    SC: StructureConstraint + Hash + Eq + Clone + fmt::Display,
    MultiDeterminant<'a, T, FCIBasis<'a, T, SC>, SC>: SymmetryTransformable,
{
    /// Calculates and stores the overlap matrix between multi-determinantal wavefunctions in the
    /// orbit, with respect to a metric of the basis in which the constituting Slater determinants
    /// are expressed.
    ///
    /// This function is particularly optimised for multi-determinantal wavefunctions constructed
    /// from orbits of origin Slater determinants such that the multi-determinantal wavefunctions
    /// are never explicitly transformed by group operations.
    ///
    /// # Arguments
    ///
    /// * `metric` - The metric of the basis in which the orbit items are expressed.
    /// * `metric_h` - The complex-symmetric metric of the basis in which the orbit items are
    ///   expressed. This is required if antiunitary operations are involved.
    /// * `use_cayley_table` - A boolean indicating if the Cayley table of the group should be used
    ///   to speed up the computation of the overlap matrix. If `false`, this will revert back to the
    ///   non-optimised overlap matrix calculation.
    pub(crate) fn calc_smat_optimised(
        &mut self,
        metric: Option<&Array2<T>>,
        metric_h: Option<&Array2<T>>,
        use_cayley_table: bool,
    ) -> Result<&mut Self, anyhow::Error> {
        ensure!(
            self.group.name() == self.origin.basis.group().name(),
            "Multi-determinantal wavefunction orbit-generating group does not match symmetry analysis group."
        );

        if let (Some(ctb), true) = (self.group().cayley_table(), use_cayley_table) {
            log::debug!(
                "Cayley table available. Group closure will be used to speed up overlap matrix computation."
            );

            let order = self.group.order();
            let mut smat = Array2::<T>::zeros((order, order));
            let multidet_0 = self.origin();
            let det_origins = multidet_0.basis.origins();
            let n_det_origins = det_origins.len();

            let detov_kpwx_vec = multidet_0
                .basis
                .iter()
                .collect::<Result<Vec<_>, _>>()?
                .into_iter()
                .cartesian_product(det_origins.iter())
                .map(|(w, x)| w.overlap(x, metric, metric_h))
                .collect::<Result<Vec<_>, _>>()?;
            let detov_kpwx =
                Array3::from_shape_vec((order, n_det_origins, n_det_origins), detov_kpwx_vec)?;
            let ovs = (0..order)
                .map(|k| {
                    [
                        (0..order),
                        (0..order),
                        (0..n_det_origins),
                        (0..n_det_origins),
                    ]
                    .into_iter()
                    .multi_cartesian_product()
                    .try_fold(T::zero(), |acc, v| {
                        let ip = v[0];
                        let jp = v[1];
                        let w = v[2];
                        let x = v[3];
                        let aipw = multidet_0.coefficients[ip * n_det_origins + w];
                        let ajpx = multidet_0.coefficients[jp * n_det_origins + x];

                        let jpinv = ctb.slice(s![.., jp]).iter().position(|&x| x == 0).ok_or(
                            format_err!("Unable to find the inverse of group element `{jp}`."),
                        )?;
                        let kp = ctb[(jpinv, ctb[(k, ip)])];
                        let ukipwjpx = self.norm_preserving_scalar_map(jp)?(detov_kpwx[(kp, w, x)]);

                        Ok::<_, anyhow::Error>(
                            acc + self.norm_preserving_scalar_map(k)?(aipw.conj())
                                * ukipwjpx
                                * ajpx,
                        )
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            for (i, j) in (0..order).cartesian_product(0..order) {
                let jinv = ctb
                    .slice(s![.., j])
                    .iter()
                    .position(|&x| x == 0)
                    .ok_or(format_err!(
                        "Unable to find the inverse of group element `{j}`."
                    ))?;
                let jinv_i = ctb[(jinv, i)];
                smat[(i, j)] = self.norm_preserving_scalar_map(jinv)?(ovs[jinv_i]);
            }
            if self.origin().complex_symmetric() {
                let _ = check_complex_matrix_symmetry(
                    &smat.view(),
                    true,
                    self.linear_independence_threshold,
                    "Orbit overlap",
                    "S_orbit",
                );
                self.set_smat(
                    (smat.clone() + smat.t().to_owned()).mapv(|x| x / (T::one() + T::one())),
                )
            } else {
                let _ = check_complex_matrix_symmetry(
                    &smat.view(),
                    false,
                    self.linear_independence_threshold,
                    "Orbit overlap",
                    "S_orbit",
                );
                self.set_smat(
                    (smat.clone() + smat.t().to_owned().mapv(|x| x.conj()))
                        .mapv(|x| x / (T::one() + T::one())),
                )
            }
            Ok(self)
        } else {
            self.calc_smat(metric, metric_h, use_cayley_table)
        }
    }
}
