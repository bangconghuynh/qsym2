use std::collections::{HashMap, HashSet};
use std::fmt::{self, Debug, Display};
use std::hash::Hash;
use std::ops::Div;
use std::panic;

use itertools::Itertools;
use log;
use ndarray::{s, Array1, Array2, ArrayView1, Axis, LinalgScalar, ShapeBuilder, Zip};
use num_modular::ModularInteger;
use num_traits::Zero;

#[cfg(test)]
#[path = "modular_linalg_tests.rs"]
mod modular_linalg_tests;

/// Calculates the determinant of a square matrix over a finite integer field
/// using the Bereiss algorithm.
///
/// For more information, see
/// <https://stackoverflow.com/questions/66192894/precise-determinant-of-integer-nxn-matrix>.
///
/// # Arguments
///
/// * `mat` - A square matrix.
///
/// # Returns
///
/// The determinant of `mat` in the same field.
///
/// # Panics
///
/// Panics if `mat` is not a square matrix.
pub fn modular_determinant<T>(mat: &Array2<T>) -> T
where
    T: Clone + LinalgScalar + ModularInteger<Base = u32> + Div<Output = T>,
{
    let mut mat = mat.clone();
    let rep = mat
        .first()
        .expect("Unable to obtain the first element of `mat`.");
    assert_eq!(mat.ncols(), mat.nrows(), "A square matrix is expected.");
    let dim = mat.ncols();
    let mut sign = rep.convert(1u32);
    let mut prev = rep.convert(1u32);
    let zero = rep.convert(0u32);

    for i in 0..(dim - 1) {
        if mat[(i, i)] == zero {
            // Swap with another row having non-zero i-th element.
            let rel_swapto = mat.slice(s![(i + 1).., i]).iter().position(|x| *x != zero);
            if let Some(rel_index) = rel_swapto {
                let (mut mat_above, mut mat_below) = mat.view_mut().split_at(Axis(0), i + 1);
                let row_from = mat_above.slice_mut(s![i, ..]);
                let row_to = mat_below.slice_mut(s![rel_index, ..]);
                Zip::from(row_from).and(row_to).for_each(std::mem::swap);
                sign = -sign;
            } else {
                // All mat[.., i] are zero => zero determinant.
                return zero;
            }
        }
        for (j, k) in ((i + 1)..dim).cartesian_product((i + 1)..dim) {
            let numerator = mat[(j, k)] * mat[(i, i)] - mat[(j, i)] * mat[(i, k)];
            mat[(j, k)] = numerator / prev;
        }
        prev = mat[(i, i)];
    }
    sign * *mat
        .last()
        .expect("Unable to obtain the last element of `mat`.")
}

/// Converts an array into its unique reduced row echelon form using Gaussian
/// elimination over a finite integer field.
///
/// # Arguments
///
/// * `mat` - A rectangular matrix.
///
/// # Returns
///
/// * The reduced row echelon form of `mat`.
/// * The nullity of `mat`.
///
/// # Panics
///
/// Panics when the pivoting values are not unity.
pub fn modular_rref<T>(mat: &Array2<T>) -> (Array2<T>, usize)
where
    T: Clone + Copy + Debug + ModularInteger<Base = u32> + Div<Output = T>,
{
    let mut mat = mat.clone();
    let nrows = mat.nrows();
    let ncols = mat.ncols();
    let rep = mat
        .first()
        .expect("Unable to obtain the first element in `mat`.");
    let zero = rep.convert(0);
    let one = rep.convert(1);
    let mut rank = 0usize;

    let mut pivot_row = 0usize;
    let mut pivot_col = 0usize;

    while pivot_row < nrows && pivot_col < ncols {
        // Find the pivot in column pivot_col
        let rel_i_nonzero_option = mat
            .slice(s![pivot_row.., pivot_col])
            .iter()
            .position(|x| *x != zero);
        if let Some(rel_i_nonzero) = rel_i_nonzero_option {
            if rel_i_nonzero > 0 {
                // Possible pivot in this column at row (pivot_row + rel_i_nonzero)
                // Swap row pivot_row with row (pivot_row + rel_i_nonzero)
                let (mut mat_above, mut mat_below) =
                    mat.view_mut().split_at(Axis(0), pivot_row + 1);
                let row_from = mat_above.slice_mut(s![pivot_row, ..]);
                let row_to = mat_below.slice_mut(s![rel_i_nonzero - 1, ..]);
                Zip::from(row_from).and(row_to).for_each(std::mem::swap);
            }

            // Scale all elements in pivot row to make the pivot element equal to one
            let pivot_val = mat[(pivot_row, pivot_col)];
            for j in (pivot_col)..ncols {
                mat[(pivot_row, j)] = mat[(pivot_row, j)] / pivot_val;
            }

            // Eliminate below the pivot
            for i in (pivot_row + 1)..nrows {
                assert_eq!(mat[(pivot_row, pivot_col)], one);
                let f = mat[(i, pivot_col)];
                // row_i -= f * pivot_row
                // Fill with zeros the lower part of pivot column
                // This is essentially a subtraction but has been optimised away.
                mat[(i, pivot_col)] = zero;
                // Subtract all remaining elements in current row
                for j in (pivot_col + 1)..ncols {
                    let a = mat[(pivot_row, j)];
                    mat[(i, j)] = mat[(i, j)] - a * f;
                }
            }

            // Eliminate above the pivot
            for i in (0..pivot_row).rev() {
                assert_eq!(mat[(pivot_row, pivot_col)], one);
                let f = mat[(i, pivot_col)];
                // row_i -= f * pivot_row
                // Fill with zeros the upper part of pivot column
                mat[(i, pivot_col)] = zero;
                // Subtract all remaining elements in current row
                for j in (pivot_col + 1)..ncols {
                    let a = mat[(pivot_row, j)];
                    mat[(i, j)] = mat[(i, j)] - a * f;
                }
            }

            // Increase pivot row and column for the next while iteration
            pivot_row += 1;
            pivot_col += 1;

            // Pivot column increases rank.
            rank += 1;
        } else {
            // No pivot in this column; pass to next column.
            pivot_col += 1;
        }
    }
    (mat, ncols - rank)
}

/// Determines a set of basis vectors for the kernel of a matrix via Gaussian
/// elimination over a finite integer field.
///
/// The kernel of an `$m \times n$` matrix `$\mathbf{M}$` is the space of
/// the solutions to the equation
///
/// ```math
///     \mathbf{M} \mathbf{x} = \mathbf{0},
/// ```
///
/// where `$\mathbf{x}$` is an `$n \times 1$` column vector.
///
/// # Arguments
///
/// * mat - A rectangular matrix.
///
/// # Returns
///
/// A vector of basis vectors for the kernel of `mat`.
fn modular_kernel<T>(mat: &Array2<T>) -> Vec<Array1<T>>
where
    T: Clone + Copy + Debug + ModularInteger<Base = u32> + Div<Output = T>,
{
    let (mat_rref, nullity) = modular_rref(mat);
    let ncols = mat.ncols();
    let rep = mat
        .first()
        .expect("Unable to obtain the first element in `mat`.");
    let zero = rep.convert(0);
    let one = rep.convert(1);
    let pivot_cols: Vec<usize> = mat_rref
        .axis_iter(Axis(0))
        .filter_map(|row| row.iter().position(|&x| x != zero))
        .collect();
    let rank = ncols - nullity;
    assert_eq!(rank, pivot_cols.len());
    log::debug!("Rank: {}", rank);
    log::debug!("Kernel dim: {}", nullity);

    let pivot_cols_set: HashSet<usize> = pivot_cols.iter().copied().collect::<HashSet<_>>();
    let non_pivot_cols = (0..ncols).collect::<HashSet<_>>();
    let non_pivot_cols = non_pivot_cols.difference(&pivot_cols_set);
    non_pivot_cols
        .map(|&non_pivot_col| {
            let mut kernel_basis_vec = Array1::from_elem((ncols,), zero);
            kernel_basis_vec[non_pivot_col] = one;

            for (i, &pivot_col) in pivot_cols.iter().enumerate() {
                kernel_basis_vec[pivot_col] = -mat_rref[(i, non_pivot_col)];
            }
            let first_nonzero_pos = kernel_basis_vec
                .iter()
                .position(|&x| x != zero)
                .expect("Kernel basis vector cannot be zero.");
            let first_nonzero = kernel_basis_vec[first_nonzero_pos];
            kernel_basis_vec
                .iter_mut()
                .for_each(|x| *x = *x / first_nonzero);
            kernel_basis_vec
        })
        .collect()
}

/// Determines the eigenvalues and eigenvector of a square matrix over a finite
/// integer field.
///
/// # Arguments
///
/// * mat - A square matrix.
///
/// # Returns
///
/// A hashmap containing the eigenvalues and the associated eigenvectors.
/// One eigenvalue can be associated with multiple eigenvectors in cases of
/// degeneracy.
///
/// # Panics
///
/// Panics when inconsistent ring moduli between matrix elements are encountered.
#[must_use]
pub fn modular_eig<T>(mat: &Array2<T>) -> HashMap<T, Vec<Array1<T>>>
where
    T: Clone
        + LinalgScalar
        + Display
        + Debug
        + ModularInteger<Base = u32>
        + Eq
        + Hash
        + panic::UnwindSafe
        + panic::RefUnwindSafe,
{
    assert!(mat.is_square(), "Only square matrices are supported.");
    let dim = mat.nrows();
    let modulus_set: HashSet<u32> = mat
        .iter()
        .filter_map(|x| panic::catch_unwind(|| x.modulus()).ok())
        .collect();
    assert_eq!(
        modulus_set.len(),
        1,
        "Inconsistent ring moduli between matrix elements."
    );
    let modulus = *modulus_set
        .iter()
        .next()
        .expect("Unexpected empty `modulus_set`.");
    let rep = mat
        .iter()
        .find(|x| panic::catch_unwind(|| x.modulus()).is_ok())
        .expect("At least one modular integer with a known modulus should have been found.");
    let zero = T::zero();
    log::debug!("Diagonalising in GF({})...", modulus);

    let results: HashMap<T, Vec<Array1<T>>> = (0..modulus)
        .filter_map(|lam| {
            let lamb = rep.convert(lam);
            let char_mat = mat - Array2::from_diag_elem(dim, lamb);
            let det = modular_determinant(&char_mat);
            if det == zero {
                let vecs = modular_kernel(&char_mat);
                log::debug!(
                    "{} is an eigenvalue with multiplicity {}.",
                    lamb,
                    vecs.len()
                );
                Some((lamb, vecs))
            } else {
                None
            }
        })
        .collect();
    let eigen_dim = results.values().fold(0usize, |acc, vecs| acc + vecs.len());
    assert_eq!(
        eigen_dim,
        dim,
        "Found {} / {} eigenvector{}. The matrix is not diagonalisable in GF({}).",
        eigen_dim,
        dim,
        if dim > 1 { "s" } else { "" },
        modulus
    );
    log::debug!(
        "Found {} / {} eigenvector{}. Eigensolver done in GF({}).",
        eigen_dim,
        dim,
        if dim > 1 { "s" } else { "" },
        modulus
    );

    results
}

/// Calculates the weighted Hermitian inner product between two vectors defined
/// as:
///
/// ```math
/// \langle \mathbf{u}, \mathbf{w} \rangle
/// = \lvert G \rvert^{-1} \sum_i
///     \frac{u_i \bar{w}_i}{\lvert K_i \rvert},
/// ```
///
/// where `$K_i$` is the i-th conjugacy class of the group, and
/// `$\bar{w_i}$` the character in `$\mathbf{w}$` corresponding to the
/// inverse conjugacy class of `$K_i$`.
///
/// Note that, in `$\mathbb{C}$`, `$\bar{w}_i = w_i^*$`, but this is not true
/// in `$\mathrm{GF}(p)$`.
///
/// # Arguments
///
/// * `vec_pair` - A pair of vectors for which the Hermitian inner product is to be
/// calculated.
/// * `class_sizes` - The sizes of the conjugacy classes.
/// * `perm_for_conj` - The permutation indices to take a vector into its conjugate.
///
/// # Returns
/// The weighted Hermitian inner product.
///
/// # Panics
///
/// Panics when inconsistent ring moduli between vector elements are encountered.
#[must_use]
pub fn weighted_hermitian_inprod<T>(
    vec_pair: (&Array1<T>, &Array1<T>),
    class_sizes: &[usize],
    perm_for_conj: Option<&Vec<usize>>,
) -> T
where
    T: Display
        + Debug
        + LinalgScalar
        + ModularInteger<Base = u32>
        + panic::UnwindSafe
        + panic::RefUnwindSafe,
{
    let (vec_u, vec_w) = vec_pair;
    assert_eq!(vec_u.len(), vec_w.len());
    assert_eq!(vec_u.len(), class_sizes.len());

    let modulus_set: HashSet<u32> = vec_u
        .iter()
        .chain(vec_w.iter())
        .filter_map(|x| panic::catch_unwind(|| x.modulus()).ok())
        .collect();
    assert_eq!(
        modulus_set.len(),
        1,
        "Inconsistent ring moduli between vector elements."
    );

    let rep = vec_u
        .iter()
        .chain(vec_w.iter())
        .find(|x| panic::catch_unwind(|| x.modulus()).is_ok())
        .expect("No known modulus found.");

    let vec_w_conj = if let Some(indices) = perm_for_conj {
        vec_w.select(Axis(0), indices)
    } else {
        vec_w.clone()
    };

    Zip::from(vec_u)
        .and(&vec_w_conj)
        .and(class_sizes)
        .fold(T::zero(), |acc, &u, &w_conj, &k| {
            acc + (u * w_conj)
                / rep.convert(
                    u32::try_from(k)
                        .unwrap_or_else(|_| panic!("Unable to convert `{k}` to `u32`.")),
                )
        })
        / rep.convert(
            u32::try_from(class_sizes.iter().sum::<usize>())
                .expect("Unable to convert the group order to `u32`."),
        )
}

/// Performs Gram--Schmidt orthogonalisation (but not normalisation) on a set of vectors.
///
/// # Arguments
///
/// * `vecs` - Vectors forming a basis for a subspace.
/// * `class_sizes` - Sizes for the conjugacy classes.
/// * `perm_for_conj` - The permutation indices to take a vector into its conjugate.
///
/// # Returns
/// The orthogonal vectors forming a basis for the same subspace.
fn gram_schmidt<T>(
    vecs: &[Array1<T>],
    class_sizes: &[usize],
    perm_for_conj: Option<&Vec<usize>>,
) -> Vec<Array1<T>>
where
    T: Display
        + Debug
        + LinalgScalar
        + ModularInteger<Base = u32>
        + panic::UnwindSafe
        + panic::RefUnwindSafe,
{
    let mut ortho_vecs: Vec<Array1<T>> = vec![];
    for (j, vec_j) in vecs.iter().enumerate() {
        ortho_vecs.push(vec_j.to_owned());
        for i in 0..j {
            let rij =
                weighted_hermitian_inprod((vec_j, &ortho_vecs[i]), class_sizes, perm_for_conj);
            ortho_vecs[j] = &ortho_vecs[j] - vecs[i].map(|&x| x * rij);
        }
    }
    ortho_vecs
}

#[derive(Debug, Clone)]
pub struct SplitSpaceError<'a, T> {
    mat: &'a Array2<T>,
    vecs: &'a [Array1<T>],
}

impl<'a, T: Display + Debug> fmt::Display for SplitSpaceError<'a, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "Unable to split the degenerate subspace spanned by {:#?} with {}.",
            self.vecs, self.mat
        )
    }
}

/// Splits a space into smaller subspaces under the action of a matrix.
///
/// # Arguments
///
/// * `mat` - A matrix to act on the specified space.
/// * `vecs` - The basis vectors specifying the space.
/// * `class_sizes` - Sizes for the conjugacy classes.
/// * `perm_for_conj` - The permutation indices to take a vector into its conjugate.
///
/// # Returns
///
/// A vector of vectors of vectors, where each inner vector contains the basis
/// vectors for an `$n$`-dimensional subspace, `$n \ge 1$`.
///
/// # Panics
///
/// Panics when inconsistent ring moduli between vector and matrix elements are found.
///
/// # Errors
///
/// Errors when the degeneracy subspace cannot be split, which occurs when any of the
/// orthogonalised vectors spanning the subspace is a null vector.
#[allow(clippy::too_many_lines)]
pub fn split_space<'a, T>(
    mat: &'a Array2<T>,
    vecs: &'a [Array1<T>],
    class_sizes: &[usize],
    perm_for_conj: Option<&Vec<usize>>,
) -> Result<Vec<Vec<Array1<T>>>, SplitSpaceError<'a, T>>
where
    T: Display
        + LinalgScalar
        + Debug
        + ModularInteger<Base = u32>
        + Eq
        + Hash
        + Zero
        + panic::UnwindSafe
        + panic::RefUnwindSafe,
{
    let modulus_set: HashSet<u32> = vecs
        .iter()
        .flatten()
        .chain(mat.iter())
        .filter_map(|x| panic::catch_unwind(|| x.modulus()).ok())
        .collect();
    assert_eq!(
        modulus_set.len(),
        1,
        "Inconsistent ring moduli between vector and matrix elements."
    );

    let rep = vecs
        .iter()
        .flatten()
        .chain(mat.iter())
        .find(|x| panic::catch_unwind(|| x.modulus()).is_ok())
        .expect("No known modulus found.");

    let dim = vecs.len();
    log::debug!("Dimensionality of space to be split: {}", dim);
    let split_subspaces = if dim <= 1 {
        log::debug!("Nothing to do.");
        vec![Vec::from(vecs)]
    } else {
        // Orthogonalise the subspace basis
        let ortho_vecs = gram_schmidt(vecs, class_sizes, perm_for_conj);
        let ortho_vecs_mat = Array2::from_shape_vec(
            (class_sizes.len(), dim).f(),
            ortho_vecs.iter().flatten().copied().collect::<Vec<_>>(),
        )
        .expect("Unable to construct a two-dimensional matrix of the orthogonal vectors.");

        // Find the representation matrix of the action of `mat` on the basis vectors
        let ortho_vecs_mag = Array2::from_shape_vec(
            (dim, 1),
            ortho_vecs
                .iter()
                .map(|col_i| weighted_hermitian_inprod((col_i, col_i), class_sizes, perm_for_conj))
                .collect(),
        )
        .expect(
            "Unable to construct a column vector of the magnitudes of the orthogonalised vectors.",
        );
        if ortho_vecs_mag.iter().any(|x| Zero::is_zero(x)) {
            return Err(SplitSpaceError { mat, vecs });
        }

        let group_order = class_sizes.iter().sum::<usize>();
        let ortho_vecs_conj_mat = Array2::from_shape_vec(
            (class_sizes.len(), dim).f(),
            ortho_vecs
                .iter()
                .flat_map(|col_i| {
                    let col_i_conj = if let Some(indices) = perm_for_conj {
                        col_i.select(Axis(0), indices)
                    } else {
                        col_i.clone()
                    };
                    Zip::from(col_i_conj.view())
                        .and(ArrayView1::from(class_sizes))
                        .map_collect(|&eij, &kj| {
                            eij / rep.convert(u32::try_from(kj * group_order).unwrap_or_else(
                                |_| panic!("Unable to convert `{}` to `u32`.", kj * group_order),
                            ))
                        })
                })
                .collect::<Vec<_>>(),
        )
        .expect(
            "Unable to construct a two-dimensional matrix of the conjugated orthogonal vectors.",
        );
        let rep_mat = ortho_vecs_conj_mat.t().dot(mat).dot(&ortho_vecs_mat) / ortho_vecs_mag;

        // Diagonalise the representation matrix
        // Then use the eigenvectors to form linear combinations of the original
        // basis vectors and split the subspace
        let eigs = modular_eig(&rep_mat);
        let n_subspaces = eigs.len();
        if n_subspaces == dim {
            log::debug!(
                "{}-dimensional space is completely split into {} one-dimensional subspaces.",
                dim,
                n_subspaces
            );
        } else {
            log::debug!(
                "{}-dimensional space is incompletely split into {} subspace{}.",
                dim,
                n_subspaces,
                if n_subspaces == 1 { "" } else { "s" }
            );
        }

        // Each eigenvalue of the representation matrix corresponds to one sub-subspace.
        eigs.iter().fold(vec![], |mut acc, (eigval, eigvecs)| {
            log::debug!(
                "Handling eigenvalue {} of the representation matrix...",
                eigval
            );
            acc.push(
                eigvecs
                    .iter()
                    .map(|vec| {
                        // Form linear combinations of the original basis vectors
                        let transformed_vec = ortho_vecs_mat.dot(vec);

                        // Normalise so that the first non-zero element is one
                        let first_non_zero = transformed_vec
                            .iter()
                            .find(|&x| !Zero::is_zero(x))
                            .expect("Unexpected zero eigenvector.");
                        Array1::from_vec(
                            transformed_vec
                                .iter()
                                .map(|x| *x / *first_non_zero)
                                .collect(),
                        )
                    })
                    .collect::<Vec<_>>(),
            );
            acc
        })
    };
    Ok(split_subspaces)
}
