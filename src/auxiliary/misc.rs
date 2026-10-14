//! Miscellaneous items.

use std::collections::hash_map::DefaultHasher;
use std::error::Error;
use std::fmt::{self, LowerExp};
use std::hash::{Hash, Hasher};

use anyhow::{ensure, format_err};
use itertools::{Itertools, MultiProduct};
use log;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis, Ix2, LinalgScalar, stack};
use ndarray_einsum::einsum;
use num::{Float, One};
use num_complex::ComplexFloat;
use num_traits::float::TotalOrder;

use crate::analysis::EigenvalueComparisonMode;
use crate::io::format::qsym2_warn;

/// Trait to enable floating point numbers to be hashed.
pub trait HashableFloat {
    /// Returns a float rounded after being multiplied by a factor.
    ///
    /// Let $`x`$ be a float, $`k`$ a factor, and $`[\cdot]`$ denote the
    /// rounding-to-integer operation. This function yields $`[x \times k] / k`$.
    ///
    /// Arguments
    ///
    /// * `threshold` - The inverse $`k^{-1}`$ of the factor $`k`$ used in the
    ///   rounding of the float.
    ///
    /// Returns
    ///
    /// The rounded float.
    #[must_use]
    fn round_factor(self, threshold: Self) -> Self;

    /// Returns the mantissa-exponent-sign triplet for a float.
    ///
    /// Reference: <https://stackoverflow.com/questions/39638363/how-can-i-use-a-hashmap-with-f64-as-key-in-rust>
    ///
    /// # Arguments
    ///
    /// * `val` - A floating point number.
    ///
    /// # Returns
    ///
    /// The corresponding mantissa-exponent-sign triplet.
    fn integer_decode(self) -> (u64, i16, i8);
}

impl HashableFloat for f64 {
    fn round_factor(self, factor: f64) -> Self {
        (self / factor).round() * factor + 0.0
    }

    fn integer_decode(self) -> (u64, i16, i8) {
        let bits: u64 = self.to_bits();
        let sign: i8 = if bits >> 63 == 0 { 1 } else { -1 };
        let mut exponent: i16 = ((bits >> 52) & 0x7ff) as i16;
        let mantissa = if exponent == 0 {
            (bits & 0x000f_ffff_ffff_ffff) << 1
        } else {
            (bits & 0x000f_ffff_ffff_ffff) | 0x0010_0000_0000_0000
        };

        exponent -= 1023 + 52;
        (mantissa, exponent, sign)
    }
}

/// Returns the hash value of a hashable struct.
///
/// Arguments
///
/// * `t` - A struct of a hashable type.
///
/// Returns
///
/// The hash value.
pub fn calculate_hash<T: Hash>(t: &T) -> u64 {
    let mut s = DefaultHasher::new();
    t.hash(&mut s);
    s.finish()
}

/// Trait for performing repeated products of iterators.
pub trait ProductRepeat: Iterator + Clone
where
    Self::Item: Clone,
{
    /// Rust implementation of Python's `itertools.product()` with repetition.
    ///
    /// From <https://stackoverflow.com/a/68231315>.
    ///
    /// # Arguments
    ///
    /// * `repeat` - Number of repetitions of the given iterator.
    ///
    /// # Returns
    ///
    /// A [`MultiProduct`] iterator.
    fn product_repeat(self, repeat: usize) -> MultiProduct<Self> {
        std::iter::repeat_n(self, repeat).multi_cartesian_product()
    }
}

impl<T: Iterator + Clone> ProductRepeat for T where T::Item: Clone {}

// =============
// Gram--Schmidt
// =============

/// Error during Gram--Schmidt orthogonalisation.
#[derive(Debug, Clone)]
pub struct GramSchmidtError<'a, T> {
    pub mat: Option<&'a Array2<T>>,
    pub vecs: Option<&'a [Array1<T>]>,
}

impl<'a, T: fmt::Display + fmt::Debug> fmt::Display for GramSchmidtError<'a, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "Unable to perform Gram--Schmidt orthogonalisation on:",)?;
        if let Some(mat) = self.mat {
            writeln!(f, "{mat}")?;
        } else if let Some(vecs) = self.vecs {
            for vec in vecs {
                writeln!(f, "{vec}")?;
            }
        } else {
            writeln!(f, "Unspecified basis vectors for Gram--Schmidt.")?;
        }
        Ok(())
    }
}

impl<'a, T: fmt::Display + fmt::Debug> Error for GramSchmidtError<'a, T> {}

/// Performs modified Gram--Schmidt orthonormalisation on a set of column vectors in a matrix with
/// respect to the complex-symmetric or Hermitian dot product.
///
/// # Arguments
///
/// * `vmat` - Matrix containing column vectors forming a basis for a subspace.
/// * `complex_symmetric` - A boolean indicating if the vector dot product is complex-symmetric. If
///   `false`, the conventional Hermitian dot product is used.
/// * `thresh` - A threshold for determining self-orthogonal vectors.
///
/// # Returns
///
/// The orthonormal vectors forming a basis for the same subspace collected as column vectors in a
/// matrix.
///
/// # Errors
///
/// Errors when the orthonormalisation procedure fails, which occurs when there is linear dependency
/// between the basis vectors and/or when self-orthogonal vectors are encountered.
pub fn complex_modified_gram_schmidt<T>(
    vmat: &'_ Array2<T>,
    complex_symmetric: bool,
    thresh: T::Real,
) -> Result<Array2<T>, GramSchmidtError<'_, T>>
where
    T: ComplexFloat + fmt::Display + 'static,
{
    let mut us: Vec<Array1<T>> = Vec::with_capacity(vmat.shape()[1]);
    let mut us_sq_norm: Vec<T> = Vec::with_capacity(vmat.shape()[1]);
    for (i, vi) in vmat.columns().into_iter().enumerate() {
        // u[i] now initialised with v[i]
        us.push(vi.to_owned());

        // Project ui onto all uj (0 <= j < i)
        // This is the 'modified' part of Gram--Schmidt. We project the current (and being updated)
        // ui onto uj, rather than projecting vi onto uj. This enhances numerical stability.
        for j in 0..i {
            let p_uj_ui = if complex_symmetric {
                us[j].t().dot(&us[i]) / us_sq_norm[j]
            } else {
                us[j].t().map(|x| x.conj()).dot(&us[i]) / us_sq_norm[j]
            };
            us[i] = &us[i] - us[j].map(|&x| x * p_uj_ui);
        }

        // Evaluate the squared norm of ui which will no longer be changed after this iteration.
        // us_sq_norm[i] now available.
        let us_sq_norm_i = if complex_symmetric {
            us[i].t().dot(&us[i])
        } else {
            us[i].t().map(|x| x.conj()).dot(&us[i])
        };
        if us_sq_norm_i.abs() < thresh {
            log::error!("A zero-norm vector found: {}", us[i]);
            return Err(GramSchmidtError {
                mat: Some(vmat),
                vecs: None,
            });
        }
        us_sq_norm.push(us_sq_norm_i);
    }

    // Normalise ui
    for i in 0..us.len() {
        us[i].mapv_inplace(|x| x / us_sq_norm[i].sqrt());
    }

    let ortho_check = us.iter().enumerate().all(|(i, ui)| {
        us.iter().enumerate().all(|(j, uj)| {
            let ov_ij = if complex_symmetric {
                ui.dot(uj)
            } else {
                ui.map(|x| x.conj()).dot(uj)
            };
            i == j || ov_ij.abs() < thresh
        })
    });

    if ortho_check {
        stack(Axis(1), &us.iter().map(|u| u.view()).collect_vec()).map_err(|err| {
            log::error!("{}", err);
            GramSchmidtError {
                mat: Some(vmat),
                vecs: None,
            }
        })
    } else {
        Err(GramSchmidtError {
            mat: Some(vmat),
            vecs: None,
        })
    }
}

// ====================================
// Linear-algebraic auxiliary functions
// ====================================

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
pub(crate) fn sort_eigenvalues_eigenvectors<T: ComplexFloat>(
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
pub(crate) fn regularise_eigenvectors<T>(eigvecs: &ArrayView2<T>, thresh: T::Real) -> Array2<T>
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
pub(crate) fn normalise_eigenvectors_real<T>(
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
pub(crate) fn normalise_eigenvectors_complex<T>(
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
pub(crate) fn check_complex_matrix_symmetry<T>(
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
pub(crate) fn check_real_matrix_symmetry<T>(
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
