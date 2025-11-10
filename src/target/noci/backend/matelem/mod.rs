use std::collections::HashSet;
use std::fmt::{self, LowerExp};

use anyhow::{self, ensure, format_err};
use itertools::Itertools;
use log;
use ndarray::{Array2, Array3, ArrayView2, s};
use ndarray_linalg::types::Lapack;
use num_complex::ComplexFloat;
use rayon::iter::{ParallelBridge, ParallelIterator};

use crate::angmom::spinor_rotation_3d::StructureConstraint;
use crate::symmetry::symmetry_element::SpecialSymmetryTransformation;
use crate::symmetry::symmetry_group::SymmetryGroupProperties;
use crate::symmetry::symmetry_transformation::SymmetryTransformable;
use crate::target::determinant::SlaterDeterminant;
use crate::target::noci::basis::{Basis, OrbitBasis};

pub mod hamiltonian;
pub mod overlap;

pub trait OrbitMatrix<'a, T, SC>
where
    T: Lapack + ComplexFloat,
    SC: StructureConstraint + Clone + fmt::Display,
    SlaterDeterminant<'a, T, SC>: SymmetryTransformable,
{
    /// The type of the matrix elements.
    type MatrixElement;

    // ----------------
    // Required methods
    // ----------------
    /// Calculates the matrix element between two Slater determinants.
    ///
    /// # Arguments
    ///
    /// * `det_w` - The determinant $`^{w}\Psi`$.
    /// * `det_x` - The determinant $`^{x}\Psi`$.
    /// * `sao` - The atomic-orbital overlap matrix.
    /// * `thresh_offdiag` - Threshold for determining non-zero off-diagonal elements in the
    ///   orbital overlap matrix between $`^{w}\Psi`$ and $`^{x}\Psi`$ during Löwdin pairing.
    /// * `thresh_zeroov` - Threshold for identifying zero Löwdin overlaps.
    ///
    /// # Returns
    ///
    /// The resulting matrix element.
    fn calc_matrix_element(
        &self,
        det_w: &SlaterDeterminant<T, SC>,
        det_x: &SlaterDeterminant<T, SC>,
        sao: &ArrayView2<T>,
        thresh_offdiag: <T as ComplexFloat>::Real,
        thresh_zeroov: <T as ComplexFloat>::Real,
    ) -> Result<Self::MatrixElement, anyhow::Error>;

    /// Computes the transpose of a matrix element.
    fn t(x: &Self::MatrixElement) -> Self::MatrixElement;

    /// Computes the complex conjugation of a matrix element.
    fn conj(x: &Self::MatrixElement) -> Self::MatrixElement;

    /// Returns the zero matrix element.
    fn zero(&self) -> Self::MatrixElement;

    // ----------------
    // Provided methods
    // ----------------

    /// Returns the norm-presearving scalar map connecting diagonally-symmetric elements in the
    /// matrix.
    #[allow(clippy::type_complexity)]
    fn norm_preserving_scalar_map<'b, G>(
        &self,
        i: usize,
        orbit_basis: &'b OrbitBasis<'b, G, SlaterDeterminant<'a, T, SC>>,
    ) -> Result<fn(&Self::MatrixElement) -> Self::MatrixElement, anyhow::Error>
    where
        G: SymmetryGroupProperties + Clone,
        'a: 'b,
    {
        let group = orbit_basis.group();
        let complex_symmetric_set = orbit_basis
            .origins()
            .iter()
            .map(|det| det.complex_symmetric())
            .collect::<HashSet<_>>();
        ensure!(
            complex_symmetric_set.len() == 1,
            "Inconsistent complex-symmetric flags across origin determinants."
        );
        let complex_symmetric = *complex_symmetric_set
            .iter()
            .next()
            .ok_or(format_err!("Unable to obtain the complex-symmetric flag."))?;
        if complex_symmetric {
            Err(format_err!(
                "`norm_preserving_scalar_map` is currently not implemented for complex-symmetric inner products. This thus precludes the use of the Cayley table to speed up the computation of orbit matrices."
            ))
        } else if group
            .get_index(i)
            .unwrap_or_else(|| panic!("Group operation index `{i}` not found."))
            .contains_time_reversal()
        {
            Ok(Self::conj)
        } else {
            Ok(Self::t)
        }
    }

    /// Computes the entire matrix of matrix elements in an orbit basis, making use of group
    /// closure for optimisation.
    ///
    /// # Arguments
    ///
    /// * `orbit_basis` - The orbit basis in which the matrix elements are to be computed.
    /// * `use_cayley_table` - Boolean indicating whether group closure should be used to speed up
    ///   the computation.
    /// * `sao` - The atomic-orbital overlap matrix.
    /// * `thresh_offdiag` - Threshold for determining non-zero off-diagonal elements in the
    ///   orbital overlap matrix between two Slater determinants during Löwdin pairing.
    /// * `thresh_zeroov` - Threshold for identifying zero Löwdin overlaps.
    fn calc_orbit_matrix<'g, G>(
        &self,
        orbit_basis: &'g OrbitBasis<'g, G, SlaterDeterminant<'a, T, SC>>,
        use_cayley_table: bool,
        sao: &ArrayView2<T>,
        thresh_offdiag: <T as ComplexFloat>::Real,
        thresh_zeroov: <T as ComplexFloat>::Real,
    ) -> Result<Array2<Self::MatrixElement>, anyhow::Error>
    where
        G: SymmetryGroupProperties + Clone,
        T: Sync + Send,
        <T as ComplexFloat>::Real: Sync,
        SlaterDeterminant<'a, T, SC>: Sync,
        Self: Sync,
        Self::MatrixElement: Send + LowerExp,
        'a: 'g,
        Self::MatrixElement: Clone,
    {
        let group = orbit_basis.group();
        let order = group.order();
        let det_origins = orbit_basis.origins();
        let n_det_origins = det_origins.len();
        let mut mat = Array2::<Self::MatrixElement>::from_elem(
            (n_det_origins * order, n_det_origins * order),
            self.zero(),
        );

        if let (Some(ctb), true) = (group.cayley_table(), use_cayley_table) {
            log::debug!(
                "Cayley table available and its use requested. Group closure will be used to speed up orbit matrix computation."
            );
            // Compute unique matrix elements
            let mut ov_elems = orbit_basis
                .iter()
                .collect::<Result<Vec<_>, _>>()?
                .iter()
                .enumerate()
                .cartesian_product(orbit_basis.origins().iter().enumerate())
                // .par_bridge()
                .map(|((k_ii, k_ii_det), (jj, jj_det))| {
                    let k = k_ii.div_euclid(n_det_origins);
                    let ii = k_ii.rem_euclid(n_det_origins);
                    (
                        ii,
                        jj,
                        k,
                        self.calc_matrix_element(
                            k_ii_det,
                            jj_det,
                            sao,
                            thresh_offdiag,
                            thresh_zeroov,
                        ),
                    )
                })
                .collect::<Vec<_>>();
            ov_elems.sort_by_key(|v| (v.0, v.1, v.2));
            let mut ov_ii_jj_k =
                Array3::from_elem((n_det_origins, n_det_origins, order), self.zero());
            for (ii, jj, k, elem_res) in ov_elems {
                log::debug!(
                    "⟨g_{k} Ψ_{ii} | Ψ_{jj}⟩ = ⟨{} Ψ_{ii} | Ψ_{jj}⟩ = {}",
                    group
                        .get_index(k)
                        .map(|g| g.to_string())
                        .unwrap_or_else(|| format!("g_{k}")),
                    elem_res
                        .as_ref()
                        .map(|v| format!("{v:+.8e}"))
                        .unwrap_or_else(|err| err.to_string())
                );
                ov_ii_jj_k[(ii, jj, k)] = elem_res?;
            }

            // Populate all matrix elements
            for v in [
                (0..order),
                (0..n_det_origins),
                (0..order),
                (0..n_det_origins),
            ]
            .into_iter()
            .multi_cartesian_product()
            {
                let i = v[0];
                let ii = v[1];
                let j = v[2];
                let jj = v[3];

                let jinv = ctb
                    .slice(s![.., j])
                    .iter()
                    .position(|&x| x == 0)
                    .ok_or(format_err!(
                        "Unable to find the inverse of group element `{j}`."
                    ))?;
                let k = ctb[(jinv, i)];
                log::debug!(
                    "{}^(-1) = {} ⇒ ⟨g_{i} Ψ_{ii} | g_{j} Ψ_{jj}⟩ = ⟨{} Ψ_{ii} | {} Ψ_{jj}⟩ = ⟨{} Ψ_{ii} | Ψ_{jj}⟩ = {:+8e}",
                    group
                        .get_index(j)
                        .map(|g| g.to_string())
                        .unwrap_or_else(|| format!("g_{j}")),
                    group
                        .get_index(jinv)
                        .map(|g| g.to_string())
                        .unwrap_or_else(|| format!("g_{jinv}")),
                    group
                        .get_index(i)
                        .map(|g| g.to_string())
                        .unwrap_or_else(|| format!("g_{i}")),
                    group
                        .get_index(j)
                        .map(|g| g.to_string())
                        .unwrap_or_else(|| format!("g_{j}")),
                    group
                        .get_index(k)
                        .map(|g| g.to_string())
                        .unwrap_or_else(|| format!("g_{k}")),
                    ov_ii_jj_k[(ii, jj, k)],
                );
                mat[(i + ii * order, j + jj * order)] =
                    self.norm_preserving_scalar_map(jinv, orbit_basis)?(&ov_ii_jj_k[(ii, jj, k)]);
            }
        } else {
            log::debug!(
                "Cayley table not available or its use not requested. Group closure will not be used for orbit matrix computation."
            );
            let orbit_basis_vec = orbit_basis.iter().collect::<Result<Vec<_>, _>>()?;
            let mut elems = orbit_basis_vec
                .iter()
                .enumerate()
                .cartesian_product(orbit_basis_vec.iter().enumerate())
                .map(|((i_ii, i_ii_det), (j_jj, j_jj_det))| {
                    let i = i_ii.div_euclid(n_det_origins);
                    let ii = i_ii.rem_euclid(n_det_origins);
                    let j = j_jj.div_euclid(n_det_origins);
                    let jj = j_jj.rem_euclid(n_det_origins);
                    let elem_res = self.calc_matrix_element(
                        i_ii_det,
                        j_jj_det,
                        sao,
                        thresh_offdiag,
                        thresh_zeroov,
                    );
                    (i, ii, j, jj, elem_res)
                })
                .collect::<Vec<_>>();
            elems.sort_by_key(|v| (v.1, v.0, v.3, v.2));
            for (i, ii, j, jj, elem_res) in elems {
                log::debug!(
                    "⟨g_{i} Ψ_{ii} | g_{j} Ψ_{jj}⟩ = ⟨{} Ψ_{ii} | {} Ψ_{jj}⟩ = {}",
                    group
                        .get_index(i)
                        .map(|g| g.to_string())
                        .unwrap_or_else(|| format!("g_{i}")),
                    group
                        .get_index(j)
                        .map(|g| g.to_string())
                        .unwrap_or_else(|| format!("g_{j}")),
                    elem_res
                        .as_ref()
                        .map(|v| format!("{v:+.8e}"))
                        .unwrap_or_else(|err| err.to_string())
                );
                mat[(i + ii * order, j + jj * order)] = elem_res?;
            }
        }
        Ok(mat)
    }
}
