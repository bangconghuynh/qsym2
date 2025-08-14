use anyhow::{self, ensure, format_err};
use duplicate::duplicate_item;
use itertools::Itertools;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis, Ix2, LinalgScalar, stack};
use ndarray_einsum::einsum;
use ndarray_linalg::{
    Eig, EigGeneralized, Eigh, GeneralizedEigenvalue, Lapack, Norm, Scalar, UPLO,
};
use num::traits::FloatConst;
use num::{Float, One};
use num_complex::{Complex, ComplexFloat};

use crate::analysis::EigenvalueComparisonMode;

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
    /// complex-symmetric.
    /// * `thresh_offdiag` - Threshold for checking if any off-diagonal elements are non-zero when
    /// verifying orthogonality.
    /// * `thresh_zeroov` - Threshold for determining zero eigenvalues of $`\mathbf{B}`$.
    /// * `eigenvalue_comparison_mode` - Comparison mode for sorting eigenvalues and their
    /// corresponding eigenvectors.
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
    /// complex-symmetric.
    /// * `thresh_offdiag` - Threshold for checking if any off-diagonal elements are non-zero when
    /// verifying orthogonality.
    /// * `thresh_zeroov` - Threshold for determining zero eigenvalues of $`\mathbf{B}`$.
    /// * `eigenvalue_comparison_mode` - Comparison mode for sorting eigenvalues and their
    /// corresponding eigenvectors.
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
    pub fn eigenvalues(&self) -> ArrayView1<T> {
        self.eigenvalues.view()
    }

    /// Returns the eigenvectors.
    pub fn eigenvectors(&self) -> ArrayView2<T> {
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

        // Real, symmetric S and H
        let deviation_h = (hmat.to_owned() - hmat.t()).norm_l2();
        ensure!(
            deviation_h <= thresh_offdiag,
            "Hamiltonian matrix is not real-symmetric: ||H - H^T|| = {deviation_h:.3e} > {thresh_offdiag:.3e}."
        );

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

        // Over the reals, canonical orthogonalisation cannot handle `smat` with negative
        // eigenvalues. This means that `smat_t` can only be the identity.
        let max_diff = (&smat_t - &Array2::<dtype_>::eye(smat_t.nrows()))
            .iter()
            .map(|x| ComplexFloat::abs(*x))
            .max_by(|x, y| {
                x.partial_cmp(y)
                    .expect("Unable to compare two `abs` values.")
            })
            .ok_or_else(|| {
                format_err!("Unable to determine the maximum element of the |S - I| matrix.")
            })?;
        ensure!(
            max_diff <= thresh_offdiag,
            "The orthogonalised overlap matrix is not the identity matrix: the maximum absolute deviation is {max_diff:.3e} > {thresh_offdiag:.3e}."
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
        let deviation_h = (hmat.to_owned() - hmat.t()).norm_l2();
        ensure!(
            deviation_h <= thresh_offdiag,
            "Hamiltonian matrix is not real-symmetric: ||H - H^T|| = {deviation_h:.3e} > {thresh_offdiag:.3e}."
        );
        let deviation_s = (smat.to_owned() - smat.t()).norm_l2();
        ensure!(
            deviation_s <= thresh_offdiag,
            "Overlap matrix is not real-symmetric: ||S - S^T|| = {deviation_s:.3e} > {thresh_offdiag:.3e}."
        );

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

        if complex_symmetric {
            // Complex-symmetric H
            let deviation = (hmat.to_owned() - hmat.t()).norm_l2();
            ensure!(
                deviation <= thresh_offdiag,
                "Hamiltonian matrix is not complex-symmetric: ||H - H^T|| = {deviation:.3e} > {thresh_offdiag:.3e}."
            );
        } else {
            // Complex-Hermitian H
            let deviation = (hmat.to_owned() - hmat.map(|v| v.conj()).t()).norm_l2();
            ensure!(
                deviation <= thresh_offdiag,
                "Hamiltonian matrix is not complex-Hermitian: ||H - H^†|| = {deviation:.3e} > {thresh_offdiag:.3e}."
            );
        }

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
        let smat_t_d = smat_t.map(|v| v.conj()).t().to_owned();

        // smat_t is not necessarily the identity, but is guaranteed to be Hermitian.
        let max_diff = (&smat_t_d.dot(&smat_t) - &Array2::<T>::eye(smat_t.nrows()))
            .iter()
            .map(|x| ComplexFloat::abs(*x))
            .max_by(|x, y| {
                x.partial_cmp(y)
                    .expect("Unable to compare two `abs` values.")
            })
            .ok_or_else(|| {
                format_err!("Unable to determine the maximum element of the |S - I| matrix.")
            })?;
        ensure!(
            max_diff <= thresh_offdiag,
            "The orthogonalised overlap matrix is not the identity matrix."
        );
        let smat_t_d_hmat_t = smat_t_d.dot(&hmat_t);

        let (eigvals_t, eigvecs_t) = smat_t_d_hmat_t.eig()?;

        // Sort the eigenvalues and eigenvectors
        let (eigvals_t_sorted, eigvecs_t_sorted) = sort_eigenvalues_eigenvectors(
            &eigvals_t.view(),
            &eigvecs_t.view(),
            &eigenvalue_comparison_mode,
        );
        let eigvecs_sorted = xmat.dot(&eigvecs_t_sorted);

        // Normalise the eigenvectors
        let eigvecs_sorted_normalised = normalise_eigenvectors_complex(
            &eigvecs_sorted.view(),
            &smat.view(),
            complex_symmetric,
            thresh_offdiag,
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

        if complex_symmetric {
            // Complex-symmetric H and S
            let deviation_h = (hmat.to_owned() - hmat.t()).norm_l2();
            ensure!(
                deviation_h <= thresh_offdiag,
                "Hamiltonian matrix is not complex-symmetric: ||H - H^T|| = {deviation_h:.3e} > {thresh_offdiag:.3e}."
            );
            let deviation_s = (smat.to_owned() - smat.t()).norm_l2();
            ensure!(
                deviation_s <= thresh_offdiag,
                "Overlap matrix is not complex-symmetric: ||S - S^T|| = {deviation_s:.3e} > {thresh_offdiag:.3e}."
            );
        } else {
            // Complex-Hermitian H and S
            let deviation_h = (hmat.to_owned() - hmat.map(|v| v.conj()).t()).norm_l2();
            ensure!(
                deviation_h <= thresh_offdiag,
                "Hamiltonian matrix is not complex-Hermitian: ||H - H^†|| = {deviation_h:.3e} > {thresh_offdiag:.3e}."
            );
            let deviation_s = (smat.to_owned() - smat.map(|v| v.conj()).t()).norm_l2();
            ensure!(
                deviation_s <= thresh_offdiag,
                "Overlap matrix is not complex-Hermitian: ||S - S^†|| = {deviation_s:.3e} > {thresh_offdiag:.3e}."
            );
        }

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
            thresh_offdiag,
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
    let eigvecs_regularised = eigvecs * eigvecs_sgn;
    eigvecs_regularised
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
/// * `thresh` - Threshold for verifying the orthogonality of the eigenvectors.
///
/// # Returns
///
/// The normalised eigenvectors.
fn normalise_eigenvectors_complex<T>(
    eigvecs: &ArrayView2<T>,
    smat: &ArrayView2<T>,
    complex_symmetric: bool,
    thresh: T::Real,
) -> Result<Array2<T>, anyhow::Error>
where
    T: LinalgScalar + ComplexFloat + std::fmt::Display,
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
                    "Unable to determine the maximum off-diagonal element of the C^†.S.C matrix."
                )
            } else {
                format_err!(
                    "Unable to determine the maximum off-diagonal element of the C^†.S.C matrix."
                )
            }
        })?;

    if complex_symmetric {
        ensure!(
            max_diff <= thresh,
            "The C^T.S.C matrix is not a diagonal matrix: the maximum absolute value of the off-diagonal elements is {max_diff:.3e} > {thresh:.3e}."
        )
    } else {
        ensure!(
            max_diff <= thresh,
            "The C^†.S.C matrix is not a diagonal matrix: the maximum absolute value of the off-diagonal elements is {max_diff:.3e} > {thresh:.3e}."
        )
    };
    let eigvecs_normalised = eigvecs / sq_norm.diag().map(|v| v.sqrt());
    Ok(eigvecs_normalised)
}
