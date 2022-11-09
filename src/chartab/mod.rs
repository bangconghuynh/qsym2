use std::fmt::{Debug, Display};
use std::hash::Hash;
use std::panic;

use ndarray::{Array1, LinalgScalar, Zip};
use num_modular::ModularInteger;

mod character;
mod modular_linalg;
mod reducedint;
mod unityroot;

///// Calculates the weighted Hermitian inner product between two vectors defined
///// as:
/////
///// ```math
/////     \langle \boldsymbol{u}, \boldsymbol{w} \rangle
/////     = \lvert G \rvert^{-1} \sum_i
/////         \frac{u_i \bar{w}_i}{\lvert K_i \rvert},
///// ```
/////
///// where `$K_i$` is the i-th conjugacy class of the group, and
///// `$\bar{w_i}$` the character in `$\boldsymbol{w}$` corresponding to the
///// inverse conjugacy class of `$K_i$`.
/////
///// Note that, in `$\mathbb{C}$`, `$\bar{w}_i = w_i^*$`, but this is not true
///// in `$\mathrm{GF}(p)$`.
//fn weighted_hermitian_inprod<T>(
//    vec_pair: (Array1<T>, Array1<T>),
//    class_sizes: Vec<usize>,
//    perm_for_conj: Option<Vec<usize>>,
//) -> T
//where
//    T: LinalgScalar + Display + Debug,
//{
//    let (vec_u, vec_w) = vec_pair;
//    assert_eq!(vec_u.len(), vec_w.len());
//    let vec_w_conj = if let Some(indices) = perm_for_conj {
//        indices.into_iter().map(|i| vec_w[i]).collect()
//    } else {
//        vec_w
//    };

//    Zip::from(&vec_u)
//        .and(&vec_w_conj)
//        .and(&class_sizes)
//        .fold(T::zero(), |acc, &u, &w_conj, &k| acc + (u * w_conj) / );
//}
