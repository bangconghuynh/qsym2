use std::collections::HashSet;
use std::fmt::{Debug, Display};
use std::hash::Hash;
use std::panic;

use log;
use ndarray::{Array1, Array2, ArrayView1, Axis, LinalgScalar, ShapeBuilder, Zip};
use num_modular::ModularInteger;
use num_traits::Zero;

use crate::chartab::modular_linalg::modular_eig;

mod character;
mod modular_linalg;
mod reducedint;
mod unityroot;

/// Calculates the weighted Hermitian inner product between two vectors defined
/// as:
///
/// ```math
///     \langle \boldsymbol{u}, \boldsymbol{w} \rangle
///     = \lvert G \rvert^{-1} \sum_i
///         \frac{u_i \bar{w}_i}{\lvert K_i \rvert},
/// ```
///
/// where `$K_i$` is the i-th conjugacy class of the group, and
/// `$\bar{w_i}$` the character in `$\boldsymbol{w}$` corresponding to the
/// inverse conjugacy class of `$K_i$`.
///
/// Note that, in `$\mathbb{C}$`, `$\bar{w}_i = w_i^*$`, but this is not true
/// in `$\mathrm{GF}(p)$`.
///
/// # Arguments
///
/// * vec_pair - A pair of vectors for which the Hermitian inner product is to be
/// calculated.
/// * class_sizes - The sizes of the conjugacy classes.
/// * perm_for_conj - The permutation indices to take a vector into its conjugate.
///
/// # Returns
/// The weighted Hermitian inner product.
fn weighted_hermitian_inprod<T>(
    vec_pair: (&Array1<T>, &Array1<T>),
    class_sizes: &[usize],
    perm_for_conj: Option<&Vec<usize>>,
) -> T
where
    T: Display
        + Debug
        + LinalgScalar
        + ModularInteger<Base = u64>
        + panic::UnwindSafe
        + panic::RefUnwindSafe,
{
    let (vec_u, vec_w) = vec_pair;
    assert_eq!(vec_u.len(), vec_w.len());
    assert_eq!(vec_u.len(), class_sizes.len());

    let modulus_set: HashSet<u64> = vec_u
        .iter()
        .chain(vec_w.iter())
        .filter_map(|x| panic::catch_unwind(|| x.modulus()).ok())
        .collect();
    assert_eq!(
        modulus_set.len(),
        1,
        "Inconsistent moduli between vector elements."
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
            acc + (u * w_conj) / rep.convert(u64::try_from(k).unwrap())
        })
        / rep.convert(u64::try_from(class_sizes.iter().sum::<usize>()).unwrap())
}

/// Performs Gram--Schmidt orthogonalisation (but not normalisation) on a set of vectors.
///
/// # Arguments
///
/// * vecs - Vectors forming a basis for a subspace.
/// * class_sizes - Sizes for the conjugacy classes.
/// * perm_for_conj - The permutation indices to take a vector into its conjugate.
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
        + ModularInteger<Base = u64>
        + panic::UnwindSafe
        + panic::RefUnwindSafe,
{
    vecs.iter().fold(vec![], |mut ortho_vecs, vec_j| {
        ortho_vecs.push(
            ortho_vecs
                .iter()
                .fold(vec_j.to_owned(), |ortho_vec_j, vec_i| {
                    let rij = weighted_hermitian_inprod(
                        (vec_j, vec_i),
                        class_sizes,
                        perm_for_conj,
                    );
                    ortho_vec_j - vec_i.map(|&x| x * rij)
                }),
        );
        ortho_vecs
    })
}

/// Splits a space into smaller subspaces under the action of a matrix.
///
/// # Arguments
///
/// * mat - A matrix to act on the specified space.
/// * vecs - The basis vectors specifying the space.
/// * class_sizes - Sizes for the conjugacy classes.
/// * perm_for_conj - The permutation indices to take a vector into its conjugate.
///
/// # Returns
/// A vector of vectors of vectors, where each inner vector contains the basis
/// vectors for an `$n$`-dimensional subspace, `$n \ge 1$`.
fn split_space<T>(
    mat: &Array2<T>,
    vecs: &[Array1<T>],
    class_sizes: &[usize],
    perm_for_conj: Option<&Vec<usize>>,
) -> Vec<Vec<Array1<T>>>
where
    T: Display
        + LinalgScalar
        + Display
        + Debug
        + ModularInteger<Base = u64>
        + Eq
        + Hash
        + Zero
        + panic::UnwindSafe
        + panic::RefUnwindSafe,
{
    let modulus_set: HashSet<u64> = vecs
        .iter()
        .flatten()
        .chain(mat.iter())
        .filter_map(|x| panic::catch_unwind(|| x.modulus()).ok())
        .collect();
    assert_eq!(
        modulus_set.len(),
        1,
        "Inconsistent moduli between vector and matrix elements."
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
        let ortho_vecs = gram_schmidt(
            vecs,
            class_sizes,
            perm_for_conj,
        );
        let ortho_vecs_mat = Array2::from_shape_vec(
            (class_sizes.len(), dim).f(),
            ortho_vecs.iter().flatten().cloned().collect::<Vec<_>>(),
        )
        .unwrap();

        // Find the representation matrix of the action of `mat` on the basis vectors
        let ortho_vecs_mag = Array2::from_shape_vec(
            (dim, 1),
            ortho_vecs
                .iter()
                .map(|col_i| {
                    weighted_hermitian_inprod(
                        (&col_i, &col_i),
                        class_sizes,
                        perm_for_conj,
                    )
                })
                .collect(),
        )
        .unwrap();

        let group_order = class_sizes.iter().sum::<usize>();
        let ortho_vecs_conj: Vec<_> = ortho_vecs
            .iter()
            .map(|col_i| {
                Zip::from(col_i.view())
                    .and(ArrayView1::from(class_sizes))
                    .map_collect(|&eij, &kj| {
                        eij / rep.convert(u64::try_from(kj * group_order).unwrap())
                    })
            })
            .collect();
        let ortho_vecs_conj_mat = Array2::from_shape_vec(
            (class_sizes.len(), dim).f(),
            ortho_vecs_conj.into_iter().flatten().collect::<Vec<_>>(),
        )
        .unwrap();
        let rep_mat = ortho_vecs_conj_mat.t().dot(mat).dot(&ortho_vecs_mat) / ortho_vecs_mag;

        // Diagonalise the representation matrix
        // Then use the eigenvectors to form linear combinations of the original
        // basis vectors and split the subspace
        let eigs = modular_eig(&rep_mat.view());
        let n_subspaces = eigs.len();
        if n_subspaces == dim {
            log::debug!(
                "{}-dimensional space is completely split into {} one-dimensional subspaces.",
                dim,
                n_subspaces
            );
        } else {
            log::debug!(
                "{}-dimensional space is incompletely split into {} subspaces.",
                dim,
                n_subspaces
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
                        let first_non_zero =
                            transformed_vec.iter().find(|&x| !Zero::is_zero(x)).unwrap();
                        Array1::from_vec(
                            transformed_vec
                                .iter()
                                .map(|x| *x / *first_non_zero)
                                .collect()
                        )
                    })
                    .collect::<Vec<_>>(),
            );
            acc
        })
    };
    split_subspaces
}
