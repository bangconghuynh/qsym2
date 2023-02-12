use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::fmt::{self, Debug, Display};
use std::hash::Hash;
use std::ops::Div;
use std::panic;

use itertools::Itertools;
use log;
use ndarray::{s, Array1, Array2, ArrayView1, Axis, LinalgScalar, ShapeBuilder, Zip};
use num::Complex;
use num_modular::ModularInteger;
use num_traits::{Inv, Pow, ToPrimitive, Zero};

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
fn modular_determinant<T>(mat: &Array2<T>) -> T
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
fn modular_rref<T>(mat: &Array2<T>) -> (Array2<T>, usize)
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

#[derive(Debug, Clone)]
pub struct ModularEigError<'a, T> {
    mat: &'a Array2<T>,
}

impl<'a, T: Display + Debug> fmt::Display for ModularEigError<'a, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Unable to diagonalise {}.", self.mat)
    }
}

impl<'a, T: Display + Debug> Error for ModularEigError<'a, T> {}

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
pub fn modular_eig<'a, T>(
    mat: &'a Array2<T>,
) -> Result<HashMap<T, Vec<Array1<T>>>, ModularEigError<'a, T>>
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
    if eigen_dim != dim {
        log::warn!(
            "Found {} / {} eigenvector{}. The matrix is not diagonalisable in GF({}).",
            eigen_dim,
            dim,
            if dim > 1 { "s" } else { "" },
            modulus
        );
        Err(ModularEigError { mat })
    } else {
        // assert_eq!(
        //     eigen_dim,
        //     dim,
        //     "Found {} / {} eigenvector{}. The matrix is not diagonalisable in GF({}).",
        //     eigen_dim,
        //     dim,
        //     if dim > 1 { "s" } else { "" },
        //     modulus
        // );
        log::debug!(
            "Found {} / {} eigenvector{}. Eigensolver done in GF({}).",
            eigen_dim,
            dim,
            if dim > 1 { "s" } else { "" },
            modulus
        );

        Ok(results)
    }
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

#[derive(Debug, Clone)]
pub struct GramSchmidtError<'a, T> {
    vecs: &'a [Array1<T>],
}

impl<'a, T: Display + Debug> fmt::Display for GramSchmidtError<'a, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "Unable to perform Gram--Schmidt orthogonalisation of the vectors {:#?}.",
            self.vecs
        )
    }
}

impl<'a, T: Display + Debug> Error for GramSchmidtError<'a, T> {}

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
fn gram_schmidt<'a, T>(
    vecs: &'a [Array1<T>],
    class_sizes: &[usize],
    perm_for_conj: Option<&Vec<usize>>,
) -> Result<Vec<Array1<T>>, GramSchmidtError<'a, T>>
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
            let norm_sq_i = weighted_hermitian_inprod(
                (&ortho_vecs[i], &ortho_vecs[i]),
                class_sizes,
                perm_for_conj,
            );
            if Zero::is_zero(&norm_sq_i) {
                return Err(GramSchmidtError { vecs });
            }
            let rij =
                weighted_hermitian_inprod((vec_j, &ortho_vecs[i]), class_sizes, perm_for_conj)
                    / norm_sq_i;
            ortho_vecs[j] = &ortho_vecs[j] - vecs[i].map(|&x| x * rij);
        }
    }
    Ok(ortho_vecs)
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

impl<'a, T: Display + Debug> Error for SplitSpaceError<'a, T> {}

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
        + Inv
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
        let ortho_vecs = gram_schmidt(vecs, class_sizes, perm_for_conj).map_err(|err| {
            log::warn!("{err}");
            SplitSpaceError { mat, vecs }
        })?;
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
                            eij / (rep.convert(u32::try_from(kj).unwrap_or_else(|_| {
                                panic!("Unable to convert `{}` to `u32`.", kj)
                            })) * rep.convert(u32::try_from(group_order).unwrap_or_else(
                                |_| panic!("Unable to convert `{}` to `u32`.", group_order),
                            )))
                        })
                })
                .collect::<Vec<_>>(),
        )
        .expect(
            "Unable to construct a two-dimensional matrix of the conjugated orthogonal vectors.",
        );

        // The division below is correct: `ortho_vecs_mag` (dim × 1) is broadcast to (dim × dim),
        // hence every row of the dividend is divided by the corresponding element of
        // `ortho_vecs_mag`.
        let rep_mat = ortho_vecs_conj_mat.t().dot(mat).dot(&ortho_vecs_mat) / ortho_vecs_mag;

        // Diagonalise the representation matrix
        // Then use the eigenvectors to form linear combinations of the original
        // basis vectors and split the subspace
        let eigs = modular_eig(&rep_mat).map_err(|_| SplitSpaceError { mat, vecs })?;
        let n_subspaces = eigs.len();
        if n_subspaces == dim {
            log::debug!(
                "{dim}-dimensional space is completely split into {n_subspaces} one-dimensional subspaces.",
            );
        } else {
            log::debug!(
                "{dim}-dimensional space is incompletely split into {n_subspaces} subspace{}.",
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

#[derive(Debug, Clone)]
pub struct Split2dSpaceError<'a, T> {
    vecs: &'a [Array1<T>],
}

impl<'a, T: Display + Debug> fmt::Display for Split2dSpaceError<'a, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(
            f,
            "Unable to greedily split the two-dimensional degenerate subspace spanned by",
        )?;
        for vec in self.vecs {
            writeln!(f, "  {vec}")?;
        }
        fmt::Result::Ok(())
    }
}

impl<'a, T: Display + Debug> Error for Split2dSpaceError<'a, T> {}

pub fn split_2d_space<'a, T>(
    vecs: &'a [Array1<T>],
    class_sizes: &[usize],
    sq_indices: &[usize],
    perm_for_conj: Option<&Vec<usize>>,
) -> Result<Vec<Vec<Array1<T>>>, Split2dSpaceError<'a, T>>
where
    T: Display
        + LinalgScalar
        + Debug
        + ModularInteger<Base = u32>
        + Eq
        + Hash
        + Zero
        + Inv
        + Pow<u32, Output = T>
        + panic::UnwindSafe
        + panic::RefUnwindSafe,
{
    assert_eq!(vecs.len(), 2, "Only two-dimensional spaces are allowed.");
    let rep = vecs
        .iter()
        .flatten()
        .find(|x| panic::catch_unwind(|| x.modulus()).is_ok())
        .expect("No known modulus found.");

    // Echelonise the basis so that v0 has first entry of 1, while v1 has first entry of 0.
    let v_flat: Vec<T> = vecs.iter().flatten().cloned().collect();
    let shape = (vecs.len(), vecs[0].dim());
    let (v_mat, _) = modular_rref(&Array2::from_shape_vec(shape, v_flat).unwrap());
    let vs = v_mat.rows().into_iter().map(|v| v.to_owned()).collect::<Vec<_>>();
    let v0 = vs[0].clone();
    let v1 = vs[1].clone();

    let v00 = weighted_hermitian_inprod((&v0, &v0), class_sizes, perm_for_conj);
    let v11 = weighted_hermitian_inprod((&v1, &v1), class_sizes, perm_for_conj);
    let v01 = weighted_hermitian_inprod((&v0, &v1), class_sizes, perm_for_conj);
    let v10 = weighted_hermitian_inprod((&v1, &v0), class_sizes, perm_for_conj);
    let group_order = class_sizes.iter().sum::<usize>();
    let group_order_u32 = u32::try_from(group_order).unwrap_or_else(|_| {
                panic!("Unable to convert the group order {group_order} to `u32`.")
    });
    let sqrt_group_order = group_order
        .to_f64()
        .expect("Unable to convert the group order to `f64`.")
        .sqrt()
        .floor()
        .to_u32()
        .expect("Unable to convert the square root of the group order to `u32`.");
    let one = rep.convert(1);
    let p = rep.modulus();
    let results = (1..=sqrt_group_order)
        .filter_map(|d0_u32| {
            if group_order.rem_euclid(usize::try_from(d0_u32).unwrap_or_else(|_| {
                panic!("Unable to convert the trial dimension {d0_u32} to `usize`.")
            })) != 0 {
                None
            } else {
                let res = (0..p).filter_map(|a0_u32| {
                    let a0 = rep.convert(a0_u32);
                    if Zero::is_zero(
                        &(a0 * (a0 * v11 + v01 + v10) + v00
                            - one / rep.convert(d0_u32).square()),
                    ) {
                        let denom = a0 * v11 + v10;
                        if Zero::is_zero(&denom) {
                            None
                        } else {
                            let a1 = -(v00 + a0 * v01) / denom;
                            let d1p2 = one / (a1 * (a1 * v11 + v01 + v10) + v00);
                            let res2 = (1..=sqrt_group_order).filter_map(|d1_u32| {
                                if group_order.rem_euclid(usize::try_from(d1_u32).unwrap_or_else(|_| {
                                    panic!("Unable to convert the trial dimension {d1_u32} to `usize`.")
                                })) == 0 && rep.convert(d1_u32).square() == d1p2 {

                                    let v0_split = Array1::from_vec(
                                        v0.iter()
                                            .zip(v1.iter())
                                            .map(|(&v0_x, &v1_x)| v0_x + a0 * v1_x)
                                            .collect_vec(),
                                    );
                                    let v1_split = Array1::from_vec(
                                        v0.iter()
                                            .zip(v1.iter())
                                            .map(|(&v0_x, &v1_x)| v0_x + a1 * v1_x)
                                            .collect_vec(),
                                    );

                                    let d0 = rep.convert(d0_u32);
                                    let d1 = rep.convert(d1_u32);
                                    let char0 = v0_split.iter().zip(class_sizes.iter()).map(|(&x, &k)| d0 * x / rep.convert(k as u32)).collect::<Vec<_>>();
                                    let char1 = v1_split.iter().zip(class_sizes.iter()).map(|(&x, &k)| d1 * x / rep.convert(k as u32)).collect::<Vec<_>>();

                                    let fs0 = sq_indices
                                        .iter()
                                        .zip(class_sizes.iter())
                                        .fold(T::zero(), |acc, (&sq_idx, &k)| {
                                            let k_u32 = u32::try_from(k).unwrap_or_else(|_| {
                                                panic!("Unable to convert the class size {k} to `u32`.");
                                            });
                                            acc + rep.convert(k_u32) * char0[sq_idx]
                                        }) / rep.convert(group_order_u32);
                                    let fs0_good = fs0.is_one() || Zero::is_zero(&fs0) || fs0 == rep.convert(p - 1);
                                    let fs1 = sq_indices
                                        .iter()
                                        .zip(class_sizes.iter())
                                        .fold(T::zero(), |acc, (&sq_idx, &k)| {
                                            let k_u32 = u32::try_from(k).unwrap_or_else(|_| {
                                                panic!("Unable to convert the class size {k} to `u32`.");
                                            });
                                            acc + rep.convert(k_u32) * char1[sq_idx]
                                        }) / rep.convert(group_order_u32);
                                    let fs1_good = fs1.is_one() || Zero::is_zero(&fs1) || fs1 == rep.convert(p - 1);

                                    if fs0_good && fs1_good && d0_u32 <= d1_u32 {
                                        Some((
                                            (v0_split, v1_split),
                                            (d0_u32, d1_u32),
                                        ))
                                    } else {
                                        None
                                    }
                                } else {
                                    None
                                }
                            }).collect_vec();
                            Some(res2)
                        }
                    } else {
                        None
                    }
                })
                .flatten()
                .collect_vec();
                Some(res)
            }
        })
        .flatten()
        .collect_vec();

    if results.len() == 1 {
        // Unique solution found.
        log::debug!("Greedy splitting algorithm for 2-D subspace found a unique solution.");
        let (v0_split, v1_split) = results[0].0.clone();
        Ok(vec![vec![v0_split], vec![v1_split]])
    } else {
        // Multiple solutions found. We will error out for now.
        log::debug!(
            "Greedy splitting algorithm for 2-D subspace found {} solutions.",
            results.len()
        );
        for (i, (_, (d0_u32, d1_u32))) in results.iter().enumerate() {
            log::debug!("Irrep dimensionalities of solution {i}: ({d0_u32}, {d1_u32})");
        }
        Err(Split2dSpaceError { vecs })
    }
}
