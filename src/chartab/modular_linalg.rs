use std::ops::Div;
use ndarray::{s, Array2, Axis, Zip};
use num_modular::ModularInteger;
use num_traits::Inv;
use itertools::Itertools;

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
fn find_modular_determinant<T>(mat: &Array2<T>) -> T
where
    T: Clone + Copy + ModularInteger<Base = u64> + Div<Output = T>,
{
    let mut mat = mat.clone();
    let rep = &mat[(0, 0)];
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

// fn find_kernel(&self) -> Self;
