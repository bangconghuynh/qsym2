use ndarray::{Array, Array2, Axis, RemoveAxis};

use crate::aux::ao_basis::{BasisAngularOrder, CartOrder, ShellOrder};
use crate::permutation::{PermutableCollection, Permutation};

#[cfg(test)]
#[path = "transformation_tests.rs"]
mod transformation_tests;

// =====================
// Trait implementations
// =====================

// =========
// Functions
// =========

/// Permutes the generalised rows of an array along one or more dimensions.
///
/// Each generalised row corresponds to a basis function, and consecutive generalised rows
/// corresponding to basis functions localised on a single atom are grouped together and then
/// permuted according to the permutation of the atoms.
///
/// # Arguments
///
/// * `arr` - A coefficient array of any dimensions.
/// * `atom_perm` - A permutation for the atoms.
/// * `axes` - The dimensions along which the generalised rows are to be permuted. The number of
/// generalised rows along each of these dimensions *must* be equal to the number of functions in
/// the basis.
/// * `bao` - A structure specifying the angular order of the underlying basis.
///
/// # Returns
///
/// The permuted array.
///
/// # Panics
///
/// Panics if the number of generalised rows along any of the dimensions in `axes` does not match
/// the number of functions in the basis, or if the permutation rank does not match the number of
/// atoms in the basis.
fn permute_array_by_atoms<D>(
    arr: &Array<f64, D>,
    atom_perm: &Permutation<usize>,
    axes: &[Axis],
    bao: &BasisAngularOrder,
) -> Array<f64, D>
where
    D: RemoveAxis,
{
    assert_eq!(
        atom_perm.rank(),
        bao.n_atoms(),
        "The rank of permutation does not match the number of atoms in the basis."
    );
    let atom_boundary_indices = bao.atom_boundary_indices();
    let permuted_shell_indices: Vec<usize> = atom_perm
        .image()
        .iter()
        .flat_map(|&i| {
            let (shell_min, shell_max) = atom_boundary_indices[i];
            shell_min..shell_max
        })
        .collect();

    let mut r = arr.clone();
    for axis in axes {
        assert_eq!(
            arr.shape()[axis.0],
            bao.n_funcs(),
            "The number of generalised rows along {axis:?} in the given array does not match the number of basis functions, {}.", bao.n_funcs()
        );
        r = r.select(*axis, &permuted_shell_indices);
    }
    r
}

fn digest_sh_rotation_3d_matrices(
    bao: &BasisAngularOrder,
    rls: &[&Array2<f64>],
    cart2rss: &[&[&Array2<f64>]],
    r2cartss: &[&[&Array2<f64>]],
) -> Vec<Array2<f64>> {
    bao.basis_shells()
        .map(|shl| {
            let l = usize::try_from(shl.l).unwrap_or_else(|_| {
                panic!(
                    "Unable to concert the angular momentum order `{}` to `usize`.",
                    shl.l
                );
            });
            match &shl.shell_order {
                ShellOrder::Pure(increasingm) => {
                    // Spherical functions.
                    let mut rl = rls[l].clone();
                    if !increasingm {
                        // `rl` is in increasing-m order by default.
                        rl.invert_axis(Axis(0));
                        rl.invert_axis(Axis(1));
                    }
                    rl
                }
                ShellOrder::Cart(cart_order) => {
                    // Cartesian functions. Convert them to real solid harmonics first, then
                    // applying the transformation, then convert back.
                    let cart2rs = cart2rss[l];
                    let r2carts = r2cartss[l];
                    let rl = cart2rs.iter().zip(r2carts.iter()).enumerate().fold(
                        Array2::zeros((cart_order.ncomps(), cart_order.ncomps())),
                        |acc, (i, (&xmat, &wmat))| {
                            let lpure = l - 2 * i;
                            acc + wmat.dot(rls[lpure]).dot(xmat)
                        },
                    );
                    let lex_cart_order = CartOrder::lex(shl.l);
                    if *cart_order != lex_cart_order {
                        // `rl` is in lexicographic order (because of `wmat` and `xmat`) by default.
                        // Consider a transformation R and its representation matrix D in a
                        // lexicographically-ordered Cartesian basis b collected in a row vector.
                        // Then,
                        //      R b = b D.
                        // If we now permute the basis functions in b by a permutation π, then the
                        // representation matrix for R changes:
                        //      R πb = πb D(π).
                        // To relate D(π) to D, we first note the representation matrix for π, P:
                        //      πb = π b = b P,
                        // which, when acts on a left row vector, permutes its entry normally, but
                        // when acts on a right column vector, permutes its entry inversely.
                        // Then,
                        //      R πb = R b P = b P D(π) => R b = b PD(π)P^(-1).
                        // Thus,
                        //      D(π) = P^(-1)DP,
                        // i.e., to obtain D(π), we permute the rows and columns of D normally
                        // according to π.
                        let perm = lex_cart_order
                            .get_perm_of(cart_order)
                            .expect("Unable to find a permutation to ");
                        rl.select(Axis(0), &perm.image())
                            .select(Axis(1), &perm.image())
                    } else {
                        rl
                    }
                }
            }
        })
        .collect::<Vec<Array2<f64>>>()
}
