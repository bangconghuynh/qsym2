use std::collections::HashSet;
use std::fmt::{Debug, Display};
use std::hash::Hash;
use std::panic;

use ndarray::{Array1, LinalgScalar, Zip};
use num_modular::ModularInteger;

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
    T: LinalgScalar
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
        indices.iter().map(|&i| vec_w[i]).collect()
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
    T: LinalgScalar
        + ModularInteger<Base = u64>
        + panic::UnwindSafe
        + panic::RefUnwindSafe,
{
    vecs.iter().fold(vec![], |mut ortho_vecs, vec_j| {
        ortho_vecs.push(ortho_vecs.iter().fold(vec_j.clone(), |ortho_vec_j, vec_i| {
            let rij = weighted_hermitian_inprod((vec_j, vec_i), class_sizes, perm_for_conj);
            ortho_vec_j - vec_i.map(|&x| x * rij)
        }));
        ortho_vecs
    })
}

// fn split_space
