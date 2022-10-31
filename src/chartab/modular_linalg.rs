use itertools::Itertools;
use ndarray::{s, Array1, Array2, Axis, DataOwned, RawData, ScalarOperand, ViewRepr, Zip};
use num_modular::ModularInteger;
use std::ops::{Div, DivAssign, Mul, SubAssign};

// use crate::aux::ndarray_shuffle;

#[cfg(test)]
#[path = "reducedint_tests.rs"]
mod reducedint_tests;

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
    T: Clone + Copy + ModularInteger<Base = u64> + Div<Output = T>,
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
    T: Clone
        + Copy
        + ModularInteger<Base = u64>
        + Div<Output = T>
        + ScalarOperand
        + DivAssign
        + SubAssign,
{
    let mut mat = mat.clone();
    let nrows = mat.nrows();
    let ncols = mat.ncols();
    let mut rank = nrows;
    let mut kernel_dim = ncols.abs_diff(nrows);
    let rep = mat.first().unwrap();
    let zero = rep.convert(0);
    let mut pivot_cols: Vec<usize> = vec![];

    // Make sure the top left element is not zero.
    if mat[(0, 0)] == zero {
        let rel_swapto_index = mat
            .slice(s![1usize.., 0usize])
            .iter()
            .position(|x| *x != zero)
            .expect("All elements in the first column are zero.");
        let (mut mat_above, mut mat_below) = mat.view_mut().split_at(Axis(0), 1);
        let row_from = mat_above.slice_mut(s![0, ..]);
        let row_to = mat_below.slice_mut(s![rel_swapto_index, ..]);
        Zip::from(row_from).and(row_to).for_each(std::mem::swap);
    }

    // Make row echelon form
    for row in 0..nrows {
        let pivot_col_search = mat.slice(s![row, ..]).iter().position(|x| *x != zero);
        if let Some(pivot_col) = pivot_col_search {
            // Scale row by the pivot value
            pivot_cols.push(pivot_col);
            let pivot_val = mat[(row, pivot_col)];
            let mut cur_row = mat.row_mut(row);
            cur_row /= pivot_val;

            // Make everything below the pivot value zero
            // Note that mat[(row, pivot_col)] is now one.
            for below_pivot_row in (row + 1)..nrows {
                if mat[(below_pivot_row, pivot_col)] != zero {
                    let mut cur_below_pivot_row = mat.row_mut(below_pivot_row);
                    Zip::from(cur_below_pivot_row)
                        .and(mat.row(row))
                        .for_each(|a, &b| {
                            *a -= b * mat[(below_pivot_row, pivot_col)];
                        });
                }
            }
        } else {
            // All zero.
            pivot_cols.push(ncols);
            rank -= 1;
            kernel_dim += 1;
        }
    }

    // Make reduced row echelon form
    for row in (nrows - 1)..=0 {
        let pivot_col = pivot_cols[row];
        if pivot_col < ncols {
            for above_pivot_row in 0..row {
                mat.row_mut(above_pivot_row) -= mat[(above_pivot_row, pivot_col)] * mat.row(row);
            }
        }
    }

    // Reorder rows so that the pivots are in a "staircase" arrangement
}
