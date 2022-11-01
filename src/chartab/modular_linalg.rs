use std::collections::HashSet;
use std::fmt::Debug;
use std::ops::Div;

use itertools::Itertools;
use log;
use ndarray::{s, Array1, Array2, Axis, Zip, LinalgScalar};
use num_modular::ModularInteger;

// use crate::aux::ndarray_shuffle;

#[cfg(test)]
#[path = "modular_linalg_tests.rs"]
mod modular_linalg_tests;

/// Calculates the determinant of a square matrix over a finite integer field
/// using the Bereiss algorithm.
/// For more information, see
/// <https://stackoverflow.com/questions/66192894/precise-determinant-of-integer-nxn-matrix>.
///
/// # Arguments
///
/// * mat - A square matrix.
///
/// # Returns
///
/// The determinant of `mat` in the same field.
fn modular_determinant<T>(mat: &Array2<T>) -> T
where
    T: Clone + LinalgScalar + ModularInteger<Base = u64> + Div<Output = T>,
{
    let mut mat = mat.clone();
    let rep = mat.first().unwrap();
    assert_eq!(mat.ncols(), mat.nrows(), "A square matrix is expected.");
    let dim = mat.ncols();
    let mut sign = rep.convert(1u64);
    let mut prev = rep.convert(1u64);
    let zero = rep.convert(0u64);

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
    sign * *mat.last().unwrap()
}

/// Converts an array into its unique reduced row echelon form using Gaussian
/// elimination over a finite integer field.
///
/// # Arguments
///
/// * mat - A rectangular matrix.
///
/// # Returns
///
/// * The reduced row echelon form of `mat`.
/// * The nullity of `mat`.
fn modular_rref<T>(mat: &Array2<T>) -> (Array2<T>, usize)
where
    T: Clone + Copy + Debug + ModularInteger<Base = u64> + Div<Output = T>,
{
    let mut mat = mat.clone();
    let nrows = mat.nrows();
    let ncols = mat.ncols();
    let rep = mat.first().unwrap();
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
/// The kernel of an `$m \times n$` matrix `$\boldsymbol{M}$` is the space of
/// the solutions to the equation
///
/// ```math
///     \boldsymbol{M} \boldsymbbol{x} = \boldsymbol{0},
/// ```
///
/// where `$\boldsymbol{x}$` is an `$n \times 1$` column vector.
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
    T: Clone + Copy + Debug + ModularInteger<Base = u64> + Div<Output = T>,
{
    let (mat_rref, nullity) = modular_rref(mat);
    let ncols = mat.ncols();
    let rep = mat.first().unwrap();
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

    let pivot_cols_set: HashSet<usize> = HashSet::from_iter(pivot_cols.iter().cloned());
    let non_pivot_cols = HashSet::from_iter(0..ncols);
    let non_pivot_cols = non_pivot_cols.difference(&pivot_cols_set);
    let kernel_basis_vecs = non_pivot_cols.map(|&non_pivot_col| {
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
    }).collect();
    kernel_basis_vecs
}
