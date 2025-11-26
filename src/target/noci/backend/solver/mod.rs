use std::fmt::LowerExp;

use anyhow::{self, ensure, format_err};
use duplicate::duplicate_item;
use itertools::Itertools;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis, Ix2, LinalgScalar, stack};
use ndarray_einsum::einsum;
use ndarray_linalg::{Eig, EigGeneralized, Eigh, GeneralizedEigenvalue, Lapack, Scalar, UPLO};
use num::traits::FloatConst;
use num::{Float, One};
use num_complex::{Complex, ComplexFloat};
use num_traits::float::TotalOrder;

use crate::analysis::EigenvalueComparisonMode;

use crate::io::format::qsym2_warn;
use crate::target::noci::backend::nonortho::CanonicalOrthogonalisable;

pub mod noci;

#[cfg(test)]
#[path = "solver_tests.rs"]
mod solver_tests;

// -----------------------------
// GeneralisedEigenvalueSolvable
// -----------------------------

/// Trait to solve the generalised eigenvalue equation for a pair of square matrices $`\mathbf{A}`$
/// and $`\mathbf{B}`$:
/// ```math
///     \mathbf{A} \mathbf{v} = \lambda \mathbf{B} \mathbf{v},
/// ```
/// where $`\mathbf{A}`$ and $`\mathbf{B}`$ are in general non-Hermitian and non-positive-definite.
pub trait GeneralisedEigenvalueSolvable {
    /// Numerical type of the matrix elements constituting the generalised eigenvalue problem.
    type NumType;

    /// Numerical type of the various thresholds for comparison.
    type RealType;

    /// Solves the *auxiliary* generalised eigenvalue problem
    /// ```math
    ///     \tilde{\mathbf{A}} \tilde{\mathbf{v}} = \tilde{\lambda} \tilde{\mathbf{B}} \tilde{\mathbf{v}},
    /// ```
    /// where $`\tilde{\mathbf{B}}`$ is the canonical-orthogonalised version of $`\mathbf{B}`$. If
    /// $`\mathbf{B}`$ is not of full rank, then the two eigenvalue problems are different.
    ///
    /// # Arguments
    ///
    /// * `complex_symmetric` - Boolean indicating whether the provided pair of matrices are
    ///   complex-symmetric.
    /// * `thresh_offdiag` - Threshold for checking if any off-diagonal elements are non-zero when
    ///   verifying orthogonality.
    /// * `thresh_zeroov` - Threshold for determining zero eigenvalues of $`\mathbf{B}`$.
    /// * `eigenvalue_comparison_mode` - Comparison mode for sorting eigenvalues and their
    ///   corresponding eigenvectors.
    ///
    /// # Returns
    ///
    /// The generalised eigenvalue result.
    fn solve_generalised_eigenvalue_problem_with_canonical_orthogonalisation(
        &self,
        complex_symmetric: bool,
        thresh_offdiag: Self::RealType,
        thresh_zeroov: Self::RealType,
        eigenvalue_comparison_mode: EigenvalueComparisonMode,
    ) -> Result<GeneralisedEigenvalueResult<Self::NumType>, anyhow::Error>;

    /// Solves the generalised eigenvalue problem using LAPACK's `?ggev` generalised eigensolver.
    ///
    /// Note that this can be numerically unstable if $`\mathbf{B}`$ is not of full rank.
    ///
    /// # Arguments
    ///
    /// * `complex_symmetric` - Boolean indicating whether the provided pair of matrices are
    ///   complex-symmetric.
    /// * `thresh_offdiag` - Threshold for checking if any off-diagonal elements are non-zero when
    ///   verifying orthogonality.
    /// * `thresh_zeroov` - Threshold for determining zero eigenvalues of $`\mathbf{B}`$.
    /// * `eigenvalue_comparison_mode` - Comparison mode for sorting eigenvalues and their
    ///   corresponding eigenvectors.
    ///
    /// # Returns
    ///
    /// The generalised eigenvalue result.
    fn solve_generalised_eigenvalue_problem_with_ggev(
        &self,
        complex_symmetric: bool,
        thresh_offdiag: Self::RealType,
        thresh_zeroov: Self::RealType,
        eigenvalue_comparison_mode: EigenvalueComparisonMode,
    ) -> Result<GeneralisedEigenvalueResult<Self::NumType>, anyhow::Error>;
}

/// Structure containing the eigenvalues and eigenvectors of a generalised
/// eigenvalue problem.
pub struct GeneralisedEigenvalueResult<T> {
    /// The resulting eigenvalues.
    eigenvalues: Array1<T>,

    /// The corresponding eigenvectors.
    eigenvectors: Array2<T>,
}

impl<T> GeneralisedEigenvalueResult<T> {
    /// Returns the eigenvalues.
    pub fn eigenvalues(&'_ self) -> ArrayView1<'_, T> {
        self.eigenvalues.view()
    }

    /// Returns the eigenvectors.
    pub fn eigenvectors(&'_ self) -> ArrayView2<'_, T> {
        self.eigenvectors.view()
    }
}

#[duplicate_item(
    [
        dtype_ [ f64 ]
    ]
    [
        dtype_ [ f32 ]
    ]
)]
impl GeneralisedEigenvalueSolvable for (&ArrayView2<'_, dtype_>, &ArrayView2<'_, dtype_>) {
    type NumType = dtype_;
    type RealType = dtype_;

    fn solve_generalised_eigenvalue_problem_with_canonical_orthogonalisation(
        &self,
        _: bool,
        thresh_offdiag: dtype_,
        thresh_zeroov: dtype_,
        eigenvalue_comparison_mode: EigenvalueComparisonMode,
    ) -> Result<GeneralisedEigenvalueResult<Self::NumType>, anyhow::Error> {
        let (hmat, smat) = (self.0.to_owned(), self.1.to_owned());

        // Symmetrise `hmat` and `smat` to improve numerical stability
        let (hmat, smat): (Array2<dtype_>, Array2<dtype_>) = {
            // Real, symmetric S and H
            check_real_matrix_symmetry(&hmat.view(), thresh_offdiag, "Hamiltonian", "H")?;
            check_real_matrix_symmetry(&smat.view(), thresh_offdiag, "Overlap", "S")?;

            (
                (hmat.to_owned() + hmat.t().to_owned()).map(|v| v / (2.0)),
                (smat.to_owned() + smat.t().to_owned()).map(|v| v / (2.0)),
            )
        };

        // CanonicalOrthogonalisationResult::calc_canonical_orthogonal_matrix checks for
        // real-symmetry of S.
        // This will fail over the reals if smat contains negative eigenvalues.
        let xmat_res = smat.view().calc_canonical_orthogonal_matrix(
            true,
            false,
            thresh_offdiag,
            thresh_zeroov,
        )?;

        let xmat = xmat_res.xmat();
        let xmat_d = xmat_res.xmat_d();

        let hmat_t = xmat_d.dot(&hmat).dot(&xmat);
        let smat_t = xmat_d.dot(&smat).dot(&xmat);

        log::debug!("Canonical-orthogonalised NOCI Hamiltonian matrix H~:\n  {hmat_t:+.8e}");
        log::debug!("Canonical-orthogonalised NOCI overlap matrix S~:\n  {smat_t:+.8e}");

        // Over the reals, canonical orthogonalisation cannot handle `smat` with negative
        // eigenvalues. This means that `smat_t` can only be the identity.
        let (pos, max_diff) = (&smat_t - &Array2::<dtype_>::eye(smat_t.nrows()))
            .iter()
            .map(|x| ComplexFloat::abs(*x))
            .enumerate()
            .max_by(|(_, x), (_, y)| {
                x.partial_cmp(y)
                    .expect("Unable to compare two `abs` values.")
            })
            .ok_or_else(|| {
                format_err!("Unable to determine the maximum element of the |S - I| matrix.")
            })?;
        let (pos_i, pos_j) = (pos.div_euclid(hmat.ncols()), pos.rem_euclid(hmat.ncols()));
        ensure!(
            max_diff <= thresh_offdiag,
            "The orthogonalised overlap matrix is not the identity matrix: the maximum absolute deviation is {max_diff:.3e} > {thresh_offdiag:.3e} at ({pos_i}, {pos_j})."
        );

        let (eigvals_t, eigvecs_t) = hmat_t.eigh(UPLO::Lower)?;

        // Sort the eigenvalues and eigenvectors
        let (eigvals_t_sorted, eigvecs_t_sorted) = sort_eigenvalues_eigenvectors(
            &eigvals_t.view(),
            &eigvecs_t.view(),
            &eigenvalue_comparison_mode,
        );
        let eigvecs_sorted = xmat.dot(&eigvecs_t_sorted);

        // Normalise the eigenvectors
        let eigvecs_sorted_normalised =
            normalise_eigenvectors_real(&eigvecs_sorted.view(), &smat.view(), thresh_offdiag)?;

        // Regularise the eigenvectors
        let eigvecs_sorted_normalised_regularised =
            regularise_eigenvectors(&eigvecs_sorted_normalised.view(), thresh_offdiag);

        Ok(GeneralisedEigenvalueResult {
            eigenvalues: eigvals_t_sorted,
            eigenvectors: eigvecs_sorted_normalised_regularised,
        })
    }

    fn solve_generalised_eigenvalue_problem_with_ggev(
        &self,
        _: bool,
        thresh_offdiag: dtype_,
        thresh_zeroov: dtype_,
        eigenvalue_comparison_mode: EigenvalueComparisonMode,
    ) -> Result<GeneralisedEigenvalueResult<Self::NumType>, anyhow::Error> {
        let (hmat, smat) = (self.0.to_owned(), self.1.to_owned());

        // Real, symmetric S and H
        check_real_matrix_symmetry(&hmat.view(), thresh_offdiag, "Hamiltonian", "H")?;
        check_real_matrix_symmetry(&smat.view(), thresh_offdiag, "Overlap", "S")?;

        let (geneigvals, eigvecs) =
            (hmat.clone(), smat.clone()).eig_generalized(Some(thresh_zeroov))?;

        for gv in geneigvals.iter() {
            if let GeneralizedEigenvalue::Finite(v, _) = gv {
                ensure!(
                    v.im().abs() <= thresh_offdiag,
                    "Unexpected complex eigenvalue {v} for real, symmetric S and H."
                );
            }
        }

        // Filter and sort the eigenvalues and eigenvectors
        let mut indices = (0..geneigvals.len())
            .filter(|i| matches!(geneigvals[*i], GeneralizedEigenvalue::Finite(_, _)))
            .collect_vec();

        match eigenvalue_comparison_mode {
            EigenvalueComparisonMode::Modulus => {
                indices.sort_by(|i, j| {
                    if let (
                        GeneralizedEigenvalue::Finite(e_i, _),
                        GeneralizedEigenvalue::Finite(e_j, _),
                    ) = (&geneigvals[*i], &geneigvals[*j])
                    {
                        ComplexFloat::abs(*e_i)
                            .partial_cmp(&ComplexFloat::abs(*e_j))
                            .unwrap()
                    } else {
                        panic!("Unable to compare some eigenvalues.")
                    }
                });
            }
            EigenvalueComparisonMode::Real => {
                indices.sort_by(|i, j| {
                    if let (
                        GeneralizedEigenvalue::Finite(e_i, _),
                        GeneralizedEigenvalue::Finite(e_j, _),
                    ) = (&geneigvals[*i], &geneigvals[*j])
                    {
                        e_i.re().partial_cmp(&e_j.re()).unwrap()
                    } else {
                        panic!("Unable to compare some eigenvalues.")
                    }
                });
            }
        }

        let eigvals_re_sorted = geneigvals.select(Axis(0), &indices).map(|gv| {
            if let GeneralizedEigenvalue::Finite(v, _) = gv {
                v.re()
            } else {
                panic!("Unexpected indeterminate eigenvalue.")
            }
        });
        let eigvecs_sorted = eigvecs.select(Axis(1), &indices);
        ensure!(
            eigvecs_sorted.iter().all(|v| v.im().abs() < thresh_offdiag),
            "Unexpected complex eigenvectors."
        );
        let eigvecs_re_sorted = eigvecs_sorted.map(|v| v.re());

        // Normalise the eigenvectors
        let eigvecs_re_sorted_normalised =
            normalise_eigenvectors_real(&eigvecs_re_sorted.view(), &smat.view(), thresh_offdiag)?;

        // Regularise the eigenvectors
        let eigvecs_re_sorted_normalised_regularised =
            regularise_eigenvectors(&eigvecs_re_sorted_normalised.view(), thresh_offdiag);

        Ok(GeneralisedEigenvalueResult {
            eigenvalues: eigvals_re_sorted,
            eigenvectors: eigvecs_re_sorted_normalised_regularised,
        })
    }
}

impl<T> GeneralisedEigenvalueSolvable for (&ArrayView2<'_, Complex<T>>, &ArrayView2<'_, Complex<T>>)
where
    T: Float + FloatConst + Scalar<Complex = Complex<T>>,
    Complex<T>: ComplexFloat<Real = T> + Scalar<Real = T, Complex = Complex<T>> + Lapack,
    for<'a> ArrayView2<'a, Complex<T>>:
        CanonicalOrthogonalisable<NumType = Complex<T>, RealType = T>,
{
    type NumType = Complex<T>;

    type RealType = T;

    fn solve_generalised_eigenvalue_problem_with_canonical_orthogonalisation(
        &self,
        complex_symmetric: bool,
        thresh_offdiag: T,
        thresh_zeroov: T,
        eigenvalue_comparison_mode: EigenvalueComparisonMode,
    ) -> Result<GeneralisedEigenvalueResult<Complex<T>>, anyhow::Error> {
        let (hmat, smat) = (self.0.to_owned(), self.1.to_owned());

        // Symmetrise `hmat` and `smat` to improve numerical stability
        let (hmat, smat): (Array2<Complex<T>>, Array2<Complex<T>>) = if complex_symmetric {
            // Complex-symmetric
            check_complex_matrix_symmetry(&hmat.view(), complex_symmetric, thresh_offdiag, "Hamiltonian", "H")?;
            check_complex_matrix_symmetry(&smat.view(), complex_symmetric, thresh_offdiag, "Overlap", "S")?;
            (
                (hmat.to_owned() + hmat.t().to_owned())
                    .map(|v| v / (Complex::<T>::one() + Complex::<T>::one())),
                (smat.to_owned() + smat.t().to_owned())
                    .map(|v| v / (Complex::<T>::one() + Complex::<T>::one())),
            )
        } else {
            // Complex-Hermitian
            check_complex_matrix_symmetry(&hmat.view(), complex_symmetric, thresh_offdiag, "Hamiltonian", "H")?;
            check_complex_matrix_symmetry(&smat.view(), complex_symmetric, thresh_offdiag, "Overlap", "S")?;
            (
                (hmat.to_owned() + hmat.map(|v| v.conj()).t().to_owned())
                    .map(|v| v / (Complex::<T>::one() + Complex::<T>::one())),
                (smat.to_owned() + smat.map(|v| v.conj()).t().to_owned())
                    .map(|v| v / (Complex::<T>::one() + Complex::<T>::one())),
            )
        };

        // CanonicalOrthogonalisationResult::calc_canonical_orthogonal_matrix checks for
        // complex-symmetry or complex-Hermiticity of S.
        let xmat_res = smat.view().calc_canonical_orthogonal_matrix(
            complex_symmetric,
            false,
            thresh_offdiag,
            thresh_zeroov,
        )?;

        let xmat = xmat_res.xmat();
        let xmat_d = xmat_res.xmat_d();

        let hmat_t = xmat_d.dot(&hmat).dot(&xmat);
        let smat_t = xmat_d.dot(&smat).dot(&xmat);

        // Symmetrise `hmat_t` and `smat_t` to improve numerical stability
        let (hmat_t_sym, smat_t_sym): (Array2<Complex<T>>, Array2<Complex<T>>) =
            if complex_symmetric {
                // Complex-symmetric
                check_complex_matrix_symmetry(&hmat_t.view(), complex_symmetric, thresh_offdiag, "Transformed Hamiltonian", "H~")?;
                check_complex_matrix_symmetry(&smat_t.view(), complex_symmetric, thresh_offdiag, "Transformed Overlap", "S~")?;
                let hmat_t_s = (hmat_t.to_owned() + hmat_t.t().to_owned())
                    .map(|v| v / (Complex::<T>::one() + Complex::<T>::one()));
                let smat_t_s = (smat_t.to_owned() + smat_t.t().to_owned())
                    .map(|v| v / (Complex::<T>::one() + Complex::<T>::one()));
                (hmat_t_s, smat_t_s)
            } else {
                // Complex-Hermitian
                check_complex_matrix_symmetry(&hmat_t.view(), complex_symmetric, thresh_offdiag, "Transformed Hamiltonian", "H~")?;
                check_complex_matrix_symmetry(&smat_t.view(), complex_symmetric, thresh_offdiag, "Transformed Overlap", "S~")?;
                let hmat_t_s = (hmat_t.to_owned() + hmat_t.map(|v| v.conj()).t().to_owned())
                    .map(|v| v / (Complex::<T>::one() + Complex::<T>::one()));
                let smat_t_s = (smat_t.to_owned() + smat_t.map(|v| v.conj()).t().to_owned())
                    .map(|v| v / (Complex::<T>::one() + Complex::<T>::one()));
                (hmat_t_s, smat_t_s)
            };
        let smat_t_sym_d = smat_t_sym.map(|v| v.conj()).t().to_owned();
        log::debug!("Complex-symmetric? {complex_symmetric}");
        log::debug!("Canonical orthogonalisation X matrix:\n  {xmat:+.8e}");
        log::debug!("Canonical-orthogonalised NOCI Hamiltonian matrix H~:\n  {hmat_t_sym:+.8e}");
        log::debug!("Canonical-orthogonalised NOCI overlap matrix S~:\n  {smat_t_sym:+.8e}");

        // smat_t_sym is not necessarily the identity, but is guaranteed to be Hermitian.
        let max_diff = (&smat_t_sym_d.dot(&smat_t_sym) - &Array2::<T>::eye(smat_t_sym.nrows()))
            .iter()
            .map(|x| ComplexFloat::abs(*x))
            .max_by(|x, y| {
                x.partial_cmp(y)
                    .expect("Unable to compare two `abs` values.")
            })
            .ok_or_else(|| {
                format_err!("Unable to determine the maximum element of the |S^†.S - I| matrix.")
            })?;
        ensure!(
            max_diff <= thresh_offdiag,
            "The S^†.S matrix is not the identity matrix. S is therefore not Hermitian."
        );
        let smat_t_sym_d_hmat_t_sym = smat_t_sym_d.dot(&hmat_t_sym);
        log::debug!(
            "Hamiltonian matrix for diagonalisation (S~)^†.(H~):\n  {smat_t_sym_d_hmat_t_sym:+.8e}"
        );

        let (eigvals_t, eigvecs_t) = smat_t_sym_d_hmat_t_sym.eig()?;

        // Sort the eigenvalues and eigenvectors
        let (eigvals_t_sorted, eigvecs_t_sorted) = sort_eigenvalues_eigenvectors(
            &eigvals_t.view(),
            &eigvecs_t.view(),
            &eigenvalue_comparison_mode,
        );
        log::debug!("Sorted eigenvalues of (S~)^†.(H~):");
        for (i, eigval) in eigvals_t_sorted.iter().enumerate() {
            log::debug!("  {i}: {eigval:+.8e}");
        }
        log::debug!("");
        log::debug!("Sorted eigenvectors of (S~)^†.(H~):\n  {eigvecs_t_sorted:+.8e}");
        log::debug!("");

        // Check orthogonality
        // let _ = normalise_eigenvectors_complex(
        //     &eigvecs_t.view(),
        //     &smat_t.view(),
        //     complex_symmetric,
        //     Some(thresh_offdiag),
        // )?;

        let eigvecs_sorted = xmat.dot(&eigvecs_t_sorted);

        // Normalise the eigenvectors
        let eigvecs_sorted_normalised = normalise_eigenvectors_complex(
            &eigvecs_sorted.view(),
            &smat.view(),
            complex_symmetric,
            None,
        )?;

        // Regularise the eigenvectors
        let eigvecs_sorted_normalised_regularised =
            regularise_eigenvectors(&eigvecs_sorted_normalised.view(), thresh_offdiag);

        Ok(GeneralisedEigenvalueResult {
            eigenvalues: eigvals_t_sorted,
            eigenvectors: eigvecs_sorted_normalised_regularised,
        })
    }

    fn solve_generalised_eigenvalue_problem_with_ggev(
        &self,
        complex_symmetric: bool,
        thresh_offdiag: T,
        thresh_zeroov: T,
        eigenvalue_comparison_mode: EigenvalueComparisonMode,
    ) -> Result<GeneralisedEigenvalueResult<Self::NumType>, anyhow::Error> {
        let (hmat, smat) = (self.0.to_owned(), self.1.to_owned());

        check_complex_matrix_symmetry(&hmat.view(), complex_symmetric, thresh_offdiag, "Hamiltonian", "H")?;
        check_complex_matrix_symmetry(&smat.view(), complex_symmetric, thresh_offdiag, "Overlap", "S")?;

        let (geneigvals, eigvecs) =
            (hmat.clone(), smat.clone()).eig_generalized(Some(thresh_zeroov))?;

        // Filter and sort the eigenvalues and eigenvectors
        let mut indices = (0..geneigvals.len())
            .filter(|i| matches!(geneigvals[*i], GeneralizedEigenvalue::Finite(_, _)))
            .collect_vec();

        match eigenvalue_comparison_mode {
            EigenvalueComparisonMode::Modulus => {
                indices.sort_by(|i, j| {
                    if let (
                        GeneralizedEigenvalue::Finite(e_i, _),
                        GeneralizedEigenvalue::Finite(e_j, _),
                    ) = (&geneigvals[*i], &geneigvals[*j])
                    {
                        ComplexFloat::abs(*e_i)
                            .partial_cmp(&ComplexFloat::abs(*e_j))
                            .unwrap()
                    } else {
                        panic!("Unable to compare some eigenvalues.")
                    }
                });
            }
            EigenvalueComparisonMode::Real => {
                indices.sort_by(|i, j| {
                    if let (
                        GeneralizedEigenvalue::Finite(e_i, _),
                        GeneralizedEigenvalue::Finite(e_j, _),
                    ) = (&geneigvals[*i], &geneigvals[*j])
                    {
                        e_i.re().partial_cmp(&e_j.re()).unwrap()
                    } else {
                        panic!("Unable to compare some eigenvalues.")
                    }
                });
            }
        }

        let eigvals_sorted = geneigvals.select(Axis(0), &indices).map(|gv| {
            if let GeneralizedEigenvalue::Finite(v, _) = gv {
                *v
            } else {
                panic!("Unexpected indeterminate eigenvalue.")
            }
        });
        let eigvecs_sorted = eigvecs.select(Axis(1), &indices);

        // Normalise the eigenvectors
        let eigvecs_sorted_normalised = normalise_eigenvectors_complex(
            &eigvecs_sorted.view(),
            &smat.view(),
            complex_symmetric,
            Some(thresh_offdiag),
        )?;

        // Regularise the eigenvectors
        let eigvecs_sorted_normalised_regularised =
            regularise_eigenvectors(&eigvecs_sorted_normalised.view(), thresh_offdiag);

        Ok(GeneralisedEigenvalueResult {
            eigenvalues: eigvals_sorted,
            eigenvectors: eigvecs_sorted_normalised_regularised,
        })
    }
}

// -------------------
// Auxiliary functions
// -------------------

/// Sorts the eigenvalues and the corresponding eigenvectors.
///
/// # Arguments
///
/// * `eigvals` - The eigenvalues.
/// * `eigvecs` - The corresponding eigenvectors.
/// * `eigenvalue_comparison_mode` - Eigenvalue comparison mode.
///
/// # Returns
///
/// A tuple containing thw sorted eigenvalues and eigenvectors.
fn sort_eigenvalues_eigenvectors<T: ComplexFloat>(
    eigvals: &ArrayView1<T>,
    eigvecs: &ArrayView2<T>,
    eigenvalue_comparison_mode: &EigenvalueComparisonMode,
) -> (Array1<T>, Array2<T>) {
    let mut indices = (0..eigvals.len()).collect_vec();
    match eigenvalue_comparison_mode {
        EigenvalueComparisonMode::Modulus => {
            indices.sort_by(|i, j| {
                ComplexFloat::abs(eigvals[*i])
                    .partial_cmp(&ComplexFloat::abs(eigvals[*j]))
                    .unwrap()
            });
        }
        EigenvalueComparisonMode::Real => {
            indices.sort_by(|i, j| eigvals[*i].re().partial_cmp(&eigvals[*j].re()).unwrap());
        }
    }
    let eigvals_sorted = eigvals.select(Axis(0), &indices);
    let eigvecs_sorted = eigvecs.select(Axis(1), &indices);
    (eigvals_sorted, eigvecs_sorted)
}

/// Regularises the eigenvectors such that the first entry of each of them has a positive real
/// part, or a positive imaginary part if the real part is zero.
///
/// # Arguments
///
/// * `eigvecs` - The eigenvectors to be regularised.
/// * `thresh` - Threshold for determining if a real number is zero.
///
/// # Returns
///
/// The regularised eigenvectors.
fn regularise_eigenvectors<T>(eigvecs: &ArrayView2<T>, thresh: T::Real) -> Array2<T>
where
    T: ComplexFloat + One,
    T::Real: Float,
{
    let eigvecs_sgn = stack!(
        Axis(0),
        eigvecs
            .row(0)
            .map(|v| {
                if Float::abs(ComplexFloat::re(*v)) > thresh {
                    T::from(v.re().signum()).expect("Unable to convert a signum to the right type.")
                } else if Float::abs(ComplexFloat::im(*v)) > thresh {
                    T::from(v.im().signum()).expect("Unable to convert a signum to the right type.")
                } else {
                    T::one()
                }
            })
            .view()
    );
    eigvecs * eigvecs_sgn
}

/// Normalises the real eigenvectors with respect to a metric.
///
/// # Arguments
///
/// * `eigvecs` - The eigenvectors to be normalised.
/// * `smat` - The metric.
/// * `thresh` - Threshold for verifying the orthogonality of the eigenvectors.
///
/// # Returns
///
/// The normalised eigenvectors.
fn normalise_eigenvectors_real<T>(
    eigvecs: &ArrayView2<T>,
    smat: &ArrayView2<T>,
    thresh: T,
) -> Result<Array2<T>, anyhow::Error>
where
    T: LinalgScalar + Float + std::fmt::LowerExp,
{
    let sq_norm = einsum("ji,jk,kl->il", &[eigvecs, smat, eigvecs])
        .map_err(|err| format_err!(err))?
        .into_dimensionality::<Ix2>()
        .map_err(|err| format_err!(err))?;
    let max_diff = (&sq_norm - &Array2::from_diag(&sq_norm.diag()))
        .iter()
        .map(|x| x.abs())
        .max_by(|x, y| {
            x.partial_cmp(y)
                .expect("Unable to compare two `abs` values.")
        })
        .ok_or_else(|| {
            format_err!(
                "Unable to determine the maximum off-diagonal element of the C^T.S.C matrix."
            )
        })?;

    ensure!(
        max_diff <= thresh,
        "The C^T.S.C matrix is not a diagonal matrix: the maximum absolute value of the off-diagonal elements is {max_diff:.3e} > {thresh:.3e}."
    );
    ensure!(
        sq_norm.diag().iter().all(|v| *v > T::zero()),
        "Some eigenvectors have negative squared norms and cannot be normalised over the reals."
    );
    let eigvecs_normalised = eigvecs / sq_norm.diag().map(|v| v.sqrt());
    Ok(eigvecs_normalised)
}

/// Normalises the complex eigenvectors with respect to a metric.
///
/// # Arguments
///
/// * `eigvecs` - The eigenvectors to be normalised.
/// * `smat` - The metric.
/// * `complex_symmetric` - Boolean indicating if the inner product is complex-symmetric or not.
/// * `thresh` - Optioanl threshold for verifying the orthogonality of the eigenvectors. If `None`,
///   orthogonality will not be verified.
///
/// # Returns
///
/// The normalised eigenvectors.
fn normalise_eigenvectors_complex<T>(
    eigvecs: &ArrayView2<T>,
    smat: &ArrayView2<T>,
    complex_symmetric: bool,
    thresh: Option<T::Real>,
) -> Result<Array2<T>, anyhow::Error>
where
    T: LinalgScalar + ComplexFloat + std::fmt::Display + std::fmt::LowerExp,
    T::Real: Float + std::fmt::LowerExp,
{
    let sq_norm = if complex_symmetric {
        einsum("ji,jk,kl->il", &[eigvecs, smat, eigvecs])
            .map_err(|err| format_err!(err))?
            .into_dimensionality::<Ix2>()
            .map_err(|err| format_err!(err))?
    } else {
        einsum(
            "ji,jk,kl->il",
            &[&eigvecs.map(|v| v.conj()).view(), smat, eigvecs],
        )
        .map_err(|err| format_err!(err))?
        .into_dimensionality::<Ix2>()
        .map_err(|err| format_err!(err))?
    };

    if let Some(thr) = thresh {
        let max_diff = (&sq_norm - &Array2::from_diag(&sq_norm.diag()))
            .iter()
            .map(|x| ComplexFloat::abs(*x))
            .max_by(|x, y| {
                x.partial_cmp(y)
                    .expect("Unable to compare two `abs` values.")
            })
            .ok_or_else(|| {
                if complex_symmetric {
                    format_err!(
                        "Unable to determine the maximum off-diagonal element of the C^T.S.C matrix."
                    )
                } else {
                    format_err!(
                        "Unable to determine the maximum off-diagonal element of the C^†.S.C matrix."
                    )
                }
            })?;

        if complex_symmetric {
            log::debug!("C^T.S.C:\n  {sq_norm:+.8e}");
            ensure!(
                max_diff <= thr,
                "The C^T.S.C matrix is not a diagonal matrix: the maximum absolute value of the off-diagonal elements is {max_diff:.3e} > {thr:.3e}."
            )
        } else {
            log::debug!("C^†.S.C:\n  {sq_norm:+.8e}");
            ensure!(
                max_diff <= thr,
                "The C^†.S.C matrix is not a diagonal matrix: the maximum absolute value of the off-diagonal elements is {max_diff:.3e} > {thr:.3e}."
            )
        };
    }
    let eigvecs_normalised = eigvecs / sq_norm.diag().map(|v| v.sqrt());
    Ok(eigvecs_normalised)
}

/// Checks for complex-symmetric or complex-Hermitian symmetry of a complex square matrix.
///
/// # Arguments
///
/// * `mat` - The complex square matrix to be checked.
/// * `complex_symmetric` - Boolean indicating if complex-symmetry is to be checked instead of
///   complex-Hermiticity.
/// * `thresh_offdiag` - Threshold for checking.
/// * `matname` - Name of the matrix.
/// * `matsymbol` - Symbol of the matrix.
fn check_complex_matrix_symmetry<T>(
    mat: &ArrayView2<T>,
    complex_symmetric: bool,
    thresh_offdiag: <T as ComplexFloat>::Real,
    matname: &str,
    matsymbol: &str,
) -> Result<(), anyhow::Error>
where
    T: LinalgScalar + ComplexFloat + std::fmt::Display + std::fmt::LowerExp,
    <T as ComplexFloat>::Real: Float + std::fmt::LowerExp + std::fmt::Display,
{
    if complex_symmetric {
        let deviation = mat.to_owned() - mat.t();
        let (pos, &max_offdiag) = deviation
            .mapv(ComplexFloat::abs)
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or_else(|| panic!("Unable to compare {a} and {b}.")))
            .ok_or_else(|| format_err!("Unable to find the maximum absolute value of the {matname} complex-symmetric deviation matrix."))?;
        let (pos_i, pos_j) = (pos.div_euclid(mat.ncols()), pos.rem_euclid(mat.ncols()));
        log::debug!("{matname} matrix:\n  {mat:+.3e}");
        log::debug!("{matname} matrix complex-symmetric deviation:\n  {deviation:+.3e}",);
        qsym2_warn!("{matname} matrix complex-symmetric deviation:\n  {deviation:+.3e}",);
        ensure!(
            max_offdiag <= thresh_offdiag,
            "The {matname} matrix is not complex-symmetric: ||{matsymbol} - ({matsymbol})^T||_∞ = {max_offdiag:.3e} > {thresh_offdiag:.3e} at ({pos_i}, {pos_j})."
        );
    } else {
        let deviation = mat.to_owned() - mat.map(|v| v.conj()).t();
        let (pos, &max_offdiag) = deviation
                .mapv(ComplexFloat::abs)
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or_else(|| panic!("Unable to compare {a} and {b}.")))
                .ok_or_else(|| format_err!("Unable to find the maximum absolute value of the {matname} complex-Hermitian deviation matrix."))?;
        let (pos_i, pos_j) = (pos.div_euclid(mat.ncols()), pos.rem_euclid(mat.ncols()));
        log::debug!("{matname} matrix:\n  {mat:+.3e}");
        log::debug!("{matname} matrix complex-Hermitian deviation:\n  {deviation:+.3e}",);
        qsym2_warn!("{matname} matrix complex-Hermitian deviation:\n  {deviation:+.3e}",);
        ensure!(
            max_offdiag <= thresh_offdiag,
            "The {matname} matrix is not complex-Hermitian: ||{matsymbol} - ({matsymbol})^†||_∞ = {max_offdiag:.3e} > {thresh_offdiag:.3e} at ({pos_i}, {pos_j})."
        );
    }
    Ok(())
}

/// Checks for real-symmetric symmetry of a complex square matrix.
///
/// # Arguments
///
/// * `mat` - The real square matrix to be checked.
/// * `thresh_offdiag` - Threshold for checking.
/// * `matname` - Name of the matrix.
/// * `matsymbol` - Symbol of the matrix.
fn check_real_matrix_symmetry<T>(
    mat: &ArrayView2<T>,
    thresh_offdiag: T,
    matname: &str,
    matsymbol: &str,
) -> Result<(), anyhow::Error>
where
    T: LowerExp + Clone + LinalgScalar + Float + TotalOrder,
{
    let deviation = mat.to_owned() - mat.t();
    let (pos, &max_offdiag) = deviation
        .map(|v| v.abs())
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .ok_or_else(|| format_err!("Unable to find the maximum absolute value of the {matname} real-symmetric deviation matrix."))?;
    log::debug!("{matname} matrix:\n  {mat:+.3e}");
    log::debug!("{matname} matrix real-symmetric deviation:\n  {deviation:+.3e}",);
    qsym2_warn!("{matname} matrix real-symmetric deviation:\n  {deviation:+.3e}",);
    let (pos_i, pos_j) = (pos.div_euclid(mat.ncols()), pos.rem_euclid(mat.ncols()));
    ensure!(
        max_offdiag <= thresh_offdiag,
        "{matname} matrix is not real-symmetric: ||{matsymbol} - ({matsymbol})^T||_∞ = {max_offdiag:.3e} > {thresh_offdiag:.3e} at ({pos_i}, {pos_j})."
    );
    Ok(())
}
