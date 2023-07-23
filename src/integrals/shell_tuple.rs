macro_rules! define_shell_tuple {
    ( <$($shell_name:ident),+> ) => {
        use std::marker::PhantomData;

        use duplicate::duplicate_item;
        use factorial::DoubleFactorial;
        use log;
        use indexmap::{IndexMap, IndexSet};
        use itertools::{izip, Itertools};
        use nalgebra::{Point3, Vector3};
        use ndarray::{Array, Array1, Array2, Dim, Dimension, Zip};
        use ndarray_einsum_beta::*;
        use num_complex::Complex;
        use num_traits::ToPrimitive;

        use crate::basis::ao::{CartOrder, ShellOrder};
        use crate::basis::ao_integrals::BasisShellContraction;

        const RANK: usize = count_exprs!($($shell_name),+);

        type C128 = Complex<f64>;

        /// A structure to handle pre-computed properties of a tuple of shells consisting of
        /// non-integration primitives.
        struct ShellTuple<'a, D: Dimension, T> {
            dim: PhantomData<D>,

            typ: PhantomData<T>,

            /// The non-integration shells in this shell tuple. Each shell has an associated
            /// boolean indicating if it is to be complex-conjugated in the integral
            /// evalulation.
            shells: [(&'a BasisShellContraction<f64, f64>, bool); RANK],

            /// A fixed-size array indicating the shape of this shell tuple.
            ///
            /// Each element in the array gives the number of functions of the corresponding shell.
            function_shell_shape: [usize; RANK],

            /// A fixed-size array indicating the shape of this shell tuple.
            ///
            /// Each element in the array gives the number of Gaussian primitives of the
            /// corresponding shell.
            primitive_shell_shape: [usize; RANK],

            /// A fixed-size array containing the boundaries of the shells in the shell tuple.
            ///
            /// Each element in the array is a tuple containing the starting (inclusive) and
            /// ending (exclusive) function indices of the shell in the entire basis set.
            shell_boundaries: [(usize, usize); RANK],

            // -----------------------------------------------
            // Quantities common to all primitive combinations
            // -----------------------------------------------
            /// A fixed-size array containing the $`\mathbf{k}`$ vectors of the shells in this
            /// shell tuple. Each $`\mathbf{k}`$ vector is appropriately signed to take into
            /// account the complex conjugation pattern of the shell tuple.
            ks: [Option<Vector3<f64>>; RANK],

            k: Vector3<f64>,

            /// A fixed-size array containing the Cartesian origins of the shells in this shell
            /// tuple.
            rs: [&'a Point3<f64>; RANK],

            /// A fixed-size array containing the angular momentum orders of the shells in this
            /// shell tuple.
            ns: [usize; RANK],

            rl2carts: [Option<Array2<f64>>; RANK],

            // ------------------------------------------------
            // Quantities unique for each primitive combination
            // ------------------------------------------------
            /// A fixed-size array of arrays of non-integration primitive exponents.
            ///
            /// This quantity is $`\zeta_g^{(k)}`$ appearing in Equations 81 and 83 of Honda, M.,
            /// Sato, K. & Obara, S. Formulation of molecular integrals over Gaussian functions
            /// treatable by both the Laplace and Fourier transforms of spatial operators by
            /// using derivative of Fourier-kernel multiplied Gaussians. *The Journal of
            /// Chemical Physics* **94**, 3790–3804 (1991),
            /// [DOI](https://doi.org/10.1063/1.459751).
            ///
            /// The i-th array in the vector is for the i-th shell. The j-th element in that
            /// array then gives the exponent of the j-th primitive exponent of that shell.
            zs: [Array1<&'a f64>; RANK],

            /// An array containing the sums of all possible combinations of non-integration
            /// primitive exponents across all shells.
            ///
            /// This quantity is $`\zeta_G^{(k)}`$ defined in Equation 81 of Honda, M.,
            /// Sato, K. & Obara, S. Formulation of molecular integrals over Gaussian functions
            /// treatable by both the Laplace and Fourier transforms of spatial operators by
            /// using derivative of Fourier-kernel multiplied Gaussians. *The Journal of
            /// Chemical Physics* **94**, 3790–3804 (1991),
            /// [DOI](https://doi.org/10.1063/1.459751).
            ///
            /// This is a [`Self::rank`]-dimensional array. The element `zg[i, j, k, ...]`
            /// gives the sum of the i-th primitive exponent on the first shell, the j-th
            /// primitive exponent on the second shell, the k-th primitive exponent on the
            /// third shell, and so on.
            zg: Array<f64, Dim<[usize; RANK]>>,

            /// An array containing the products of all possible combinations of
            /// non-integration primitive exponents across all shells.
            ///
            /// This is a [`Self::rank`]-dimensional array. The element `sd[i, j, k, ...]`
            /// gives the product of the i-th primitive exponent on the first shell, the j-th
            /// primitive exponent on the second shell, the k-th primitive exponent on the
            /// third shell, and so on.
            zd: Array<f64, Dim<[usize; RANK]>>,

            /// A fixed-size array of arrays of contraction coefficients of non-integration
            /// primitives.
            ///
            /// The i-th array in the vector is for the i-th shell. The j-th element in that
            /// array then gives the contraction coefficient of the j-th primitive exponent of
            /// that shell.
            ds: [Array1<&'a f64>; RANK],

            /// An array containing the product of all possible combinations of non-integration
            /// primitive coefficients across all shells.
            ///
            /// This is a [`Self::rank`]-dimensional array. The element `dd[i, j, k, ...]` gives
            /// the product of the i-th primitive's coefficient on the first shell, the j-th
            /// primitive's coefficient on the second shell, the k-th primitive's coefficient
            /// on the third shell, and so on.
            dd: Array<f64, Dim<[usize; RANK]>>,

            /// An array containing the exponent-weighted Cartesian centres of all possible
            /// combinations of primitives across all shells.
            ///
            /// This is a [`Self::rank`]-dimensional array. The element `rg[i, j, k, ...]` gives
            /// the exponent-weighted Cartesian centre of the i-th primitive on the first shell,
            /// the j-th primitive on the second shell, the k-th primitive on the third shell,
            /// and so on.
            rg: Array<Point3<f64>, Dim<[usize; RANK]>>,

            /// A fixed-size array of arrays giving the optional quantity $`\mathbf{Q}_j`$ for
            /// the j-th shell.
            ///
            /// This quantity is defined in Equation 122 of Honda, M., Sato, K. & Obara, S.
            /// Formulation of molecular integrals over Gaussian functions treatable by both
            /// the Laplace and Fourier transforms of spatial operators by using derivative of
            /// Fourier-kernel multiplied Gaussians. *The Journal of Chemical Physics* **94**,
            /// 3790–3804 (1991), [DOI](https://doi.org/10.1063/1.459751). Since there are no
            /// integration exponents in the current implementation of [`ShellTuple`], the
            /// summation over $`v`$ in Equation 122 is not included. See also Equation 171 in
            /// the reference.
            ///
            /// The j-th array is for the j-th shell. Each array is a [`Self::rank`]-dimensional
            /// array. The element `qs[j][i, m, k, ...]` gives the :math:`\mathbf{Q}_j` vector
            /// defined using the i-th primitive exponent on the first shell, the m-th
            /// primitive exponent on the second shell, the k-th primitive exponent on the
            /// third shell, and so on. The exponent-combination dependence comes from the
            /// $`\mathbf{R}_G`$ term.
            ///
            /// If the j-th shell does not have a $`\mathbf{k}_j`$ plane-wave vector, then the
            /// corresponding $`\mathbf{Q}_j`$ is set to `None`.
            qs: [Option<Array<Vector3<f64>, Dim<[usize; RANK]>>>; RANK]
        }

        impl<'a, D: Dimension, T> ShellTuple<'a, D, T> {
            /// The number of shells in this tuple.
            fn rank(&self) -> usize {
                self.shells.len()
            }

            /// The maximum angular momentum across all shells.
            fn lmax(&self) -> u32 {
                self.shells
                    .iter()
                    .map(|(bsc, _)| bsc.basis_shell().l)
                    .max()
                    .expect("The maximum angular momentum across all shells cannot be found.")
            }
        }

        impl_shell_tuple_overlap!(<$($shell_name),+>);

        // #[duplicate_item(
        //     [
        //         dtype [ f64 ]
        //         arr_map_closure [ |x| x ]
        //         exp_kqs_func [
        //             if self.ks.iter().any(|k| k.is_some()) {
        //                 panic!("Real-valued overlaps cannot handle plane-wave vectors.")
        //             } else {
        //                 (
        //                     None::<Vec<Array<f64, Dim<[usize; RANK]>>>>,
        //                     None::<Vec<Array<f64, Dim<[usize; RANK]>>>>,
        //                 )
        //             }
        //         ]
        //         exp_ks_kqs_to_int_func [
        //             (0..3).for_each(|i| {
        //                 match (exp_ks_opt.as_ref(), exp_kqs_opt.as_ref()) {
        //                     (Some(_), _) | (_, Some(_)) => {
        //                         panic!("Real-valued overlaps cannot handle plane-wave vectors.")
        //                     }
        //                     (None, None) => {
        //                         ints_r[i][l_tuple][n_tuple] = Some(
        //                             &pre_zg * &exp_zgs[i]
        //                         );
        //                     }
        //                 }
        //             })
        //         ]
        //         n_recur_k_term_func [
        //             panic!("Real-valued overlaps cannot handle plane-wave vectors.")
        //         ]
        //         l_recur_k_term_func [
        //             panic!("Real-valued overlaps cannot handle plane-wave vectors.")
        //         ]
        //     ]
        //     [
        //         dtype [ C128 ]
        //         arr_map_closure [ |x| C128::from(x) ]
        //         exp_kqs_func [
        //             if self.ks.iter().any(|k| k.is_some()) {
        //                 // exp_ks = exp(-|k|^2 / 4 zg)
        //                 // exp_ks[i] is the contribution from the ith Cartesian component.
        //                 // zg is primitive-combination-specific.
        //                 let exp_ks = (0..3).map(|i| {
        //                     self.zg.mapv(|zg| {
        //                         (-self.k[i].abs().powi(2) / (4.0 * zg)).exp()
        //                     })
        //                 }).collect::<Vec<_>>();

        //                 // exp_kqs = exp(ii * sum(j) k_j · q_j)
        //                 // exp_kqs[i] is the contribution from the ith Cartesian component.
        //                 // q_j is primitive-combination-specific.
        //                 let exp_kqs = (0..3).map(|i| {
        //                     (0..RANK).filter_map(|j| {
        //                         match (self.ks[j], self.qs[j].as_ref()) {
        //                             (Some(kj), Some(qj)) => {
        //                                 Some(
        //                                     qj.mapv(|qjj| kj[i] * qjj[i])
        //                                 )
        //                             }
        //                             _ => None
        //                         }
        //                     })
        //                     .fold(
        //                         Array::<C128, Dim<[usize; RANK]>>::zeros(self.zg.raw_dim()),
        //                         |acc, arr| acc + arr
        //                     )
        //                     .mapv(|x| (x * C128::i()).exp())
        //                 })
        //                 .collect::<Vec<_>>();
        //                 (Some(exp_ks), Some(exp_kqs))
        //             } else {
        //                 (None, None)
        //             }
        //         ]
        //         exp_ks_kqs_to_int_func [
        //             (0..3).for_each(|i| {
        //                 match (exp_ks_opt.as_ref(), exp_kqs_opt.as_ref()) {
        //                     // Element-wise multiplication. Each element is for a specific
        //                     // primitive combination.
        //                     (Some(exp_ks), Some(exp_kqs)) => {
        //                         ints_r[i][l_tuple][n_tuple] = Some(
        //                             (&pre_zg * &exp_zgs[i] * &exp_ks[i]).mapv(C128::from)
        //                                 * &exp_kqs[i]
        //                         );
        //                     }
        //                     _ => {
        //                         ints_r[i][l_tuple][n_tuple] = Some(
        //                             (&pre_zg * &exp_zgs[i]).mapv(C128::from)
        //                         );
        //                     }
        //                 }
        //             })
        //         ]
        //         n_recur_k_term_func [
        //             // 1 / (2 * zg) * sum(i) ii * k_iα * [[:|:]]
        //             // zg is primitive-combination-specific.
        //             (0..3).for_each(|i| {
        //                 let add_term = self.zg.mapv(|zg| {
        //                     C128::i() * kk[i] / (2.0 * zg)
        //                 }) * ints_r[i][l_tuple][n_tuple].as_ref().unwrap_or_else(|| {
        //                     panic!("({l_tuple:?}, {n_tuple:?}) => ({l_tuple:?}, {next_n_tuple:?}) failed.")
        //                 });
        //                 if let Some(arr) = ints_r[i][l_tuple][next_n_tuple].as_mut() {
        //                     Zip::from(arr).and(&add_term).for_each(|a, &t| *a += t);
        //                 } else {
        //                     ints_r[i][l_tuple][next_n_tuple] = Some(add_term);
        //                 }
        //             });
        //         ]
        //         l_recur_k_term_func [
        //             // -ii * k_gα * [[:|:]]
        //             (0..3).for_each(|i| {
        //                 let add_term = C128::i()
        //                 * kr[i]
        //                 * ints_r[i][l_tuple][n_tuple].as_ref().unwrap_or_else(|| {
        //                     panic!("({l_tuple:?}, {n_tuple:?}) => ({next_l_tuple:?}, {n_tuple:?}) failed.")
        //                 });
        //                 if let Some(arr) = ints_r[i][next_l_tuple][n_tuple].as_mut() {
        //                     Zip::from(arr).and(&add_term).for_each(|a, &t| *a -= t);
        //                 } else {
        //                     ints_r[i][next_l_tuple][n_tuple] = Some(-add_term);
        //                 }
        //             });
        //         ]
        //     ]
        // )]
        // impl<'a, D: Dimension> ShellTuple<'a, D, dtype> {
        //     fn overlap(&self, ls: [usize; RANK]) -> Vec<Array<dtype, Dim<[usize; RANK]>>> {
        //         // ~~~~~~~~~~~~~~~~~~~
        //         // Preparation begins.
        //         // ~~~~~~~~~~~~~~~~~~~

        //         // We require extra Cartesian degrees to calculate derivatives, because each
        //         // derivative order increases a corresponding Cartesian rank by one.
        //         let ns: [usize; RANK] = if ls.iter().any(|l| *l > 0) {
        //             let mut ns = self.ns.clone();
        //             ns.iter_mut().for_each(|n| *n += 1);
        //             ns
        //         } else {
        //             self.ns.clone()
        //         };

        //         // Generate all terms in recurrence series
        //         // First index: Cartesian component
        //         // Next stc.rank indices: l-recursion indices
        //         // Next stc.rank indices: n-recursion indices
        //         // Last stc.rank indices: primitive indices
        //         // E.g.: rank 3,
        //         //   ints_r[1][(0, 0, 1)][(0, 1, 2)][(3, 8, 7)]: y-component integral value with
        //         //     0th y-derivative of 0th Cartesian y-power of 3rd primitive on first shell,
        //         //     0th y-derivative of 1st Cartesian y-power of 8th primitive on second shell, and
        //         //     1st y-derivative of 2nd Cartesian y-power of 7th primitive on third shell
        //         let lrecursion_shape = {
        //             let mut ls_mut = ls.clone();
        //             ls_mut.iter_mut().for_each(|l| *l += 1);
        //             ls_mut
        //         };
        //         let nrecursion_shape = {
        //             let mut ns_mut = ns.clone();
        //             ns_mut.iter_mut().for_each(|n| *n += 1);
        //             ns_mut
        //         };
        //         let arr = Array::<_, Dim<[usize; RANK]>>::from_elem(
        //             lrecursion_shape, Array::<_, Dim<[usize; RANK]>>::from_elem(
        //                 nrecursion_shape, None::<Array::<dtype, Dim<[usize; RANK]>>>
        //             )
        //         );
        //         let mut ints_r = [arr.clone(), arr.clone(), arr];

        //         let default_tuple = [$(replace_expr!(($shell_name) 0)),+];
        //         let l_tuples = ls
        //             .iter()
        //             .map(|l| 0..=*l)
        //             .multi_cartesian_product()
        //             .map(|ltuple| {
        //                 let mut ltuple_arr = default_tuple.clone();
        //                 ltuple_arr.iter_mut().enumerate().for_each(|(i, l)| *l = ltuple[i]);
        //                 ltuple_arr
        //             })
        //             .collect::<Vec<_>>();
        //         let n_tuples = ns
        //             .iter()
        //             .map(|n| 0..=*n)
        //             .multi_cartesian_product()
        //             .map(|ntuple| {
        //                 let mut ntuple_arr = default_tuple.clone();
        //                 ntuple_arr.iter_mut().enumerate().for_each(|(i, n)| *n = ntuple[i]);
        //                 ntuple_arr
        //             })
        //             .collect::<Vec<_>>();
        //         let n_tuples_noextra = ns
        //             .iter()
        //             .map(|n| 0..*n)
        //             .multi_cartesian_product()
        //             .map(|ntuple| {
        //                 let mut ntuple_arr = default_tuple.clone();
        //                 ntuple_arr.iter_mut().enumerate().for_each(|(i, n)| *n = ntuple[i]);
        //                 ntuple_arr
        //             })
        //             .collect::<Vec<_>>();

        //         let all_tuples = l_tuples.iter().cloned().cartesian_product(
        //             n_tuples.iter().cloned()
        //         ).into_iter().collect::<IndexSet<_>>();
        //         let mut remaining_tuples = all_tuples.clone();

        //         let remaining_tuples_noextra = l_tuples.iter().cloned().cartesian_product(
        //             n_tuples_noextra.iter().cloned()
        //         ).into_iter().collect::<IndexSet<_>>();

        //         let extra_tuples = all_tuples
        //             .difference(&remaining_tuples_noextra)
        //             .cloned()
        //             .collect::<IndexSet<_>>();
        //         // ~~~~~~~~~~~~~~~~~
        //         // Preparation ends.
        //         // ~~~~~~~~~~~~~~~~~

        //         // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        //         // Loop over all tuples begins.
        //         // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        //         for (tuple_index, (l_tuple, n_tuple)) in all_tuples.into_iter().enumerate() {
        //             // ~~~~~~~~~~~~~~~~~~~~
        //             // Initial term begins.
        //             // ~~~~~~~~~~~~~~~~~~~~
        //             if tuple_index == 0 {
        //                 assert!(remaining_tuples.remove(&(l_tuple, n_tuple)));

        //                 // pre_zg = sqrt(pi / zg)
        //                 // zg is primitive-combination-specific.
        //                 let pre_zg = self.zg.mapv(|zg| (std::f64::consts::PI / zg).sqrt());

        //                 // exp_zgs = sum(g < h) [ -(z_g * z_h) / zg * |r_g - r_h|^2 ]
        //                 // exp_zgs[i] is the contribution from the ith Cartesian component.
        //                 // z_g, z_h, and zg are primitive-combination-specific.
        //                 let exp_zgs = (0..3).map(|i| {
        //                     let mut exp_zg_i = self.zg.clone();
        //                     exp_zg_i.indexed_iter_mut().for_each(|(indices, zg)| {
        //                         let ($($shell_name),+) = indices;
        //                         let indices = [$($shell_name),+];
        //                         *zg = (
        //                             -1.0
        //                             / *zg
        //                             * (0..RANK).flat_map(|g| ((g + 1)..RANK).map(move |h| {
        //                                 self.zs[g][indices[g]]
        //                                     * self.zs[h][indices[h]]
        //                                     * (self.rs[g][i] - self.rs[h][i]).powi(2)
        //                             })).sum::<f64>()
        //                         ).exp();
        //                     });
        //                     exp_zg_i
        //                 }).collect::<Vec<_>>();

        //                 // let (exp_ks_opt, exp_kqs_opt) = if self.ks.iter().any(|k| k.is_some()) {
        //                 //     // exp_ks = exp(-|k|^2 / 4 zg)
        //                 //     // exp_ks[i] is the contribution from the ith Cartesian component.
        //                 //     // zg is primitive-combination-specific.
        //                 //     let exp_ks = (0..3).map(|i| {
        //                 //         self.zg.mapv(|zg| {
        //                 //             (-self.k[i].abs().powi(2) / (4.0 * zg)).exp()
        //                 //         })
        //                 //     }).collect::<Vec<_>>();

        //                 //     // exp_kqs = exp(ii * sum(j) k_j · q_j)
        //                 //     // exp_kqs[i] is the contribution from the ith Cartesian component.
        //                 //     // q_j is primitive-combination-specific.
        //                 //     let exp_kqs = (0..3).map(|i| {
        //                 //         (0..RANK).filter_map(|j| {
        //                 //             match (self.ks[j], self.qs[j].as_ref()) {
        //                 //                 (Some(kj), Some(qj)) => {
        //                 //                     Some(
        //                 //                         qj.mapv(|qjj| kj[i] * qjj[i])
        //                 //                     )
        //                 //                 }
        //                 //                 _ => None
        //                 //             }
        //                 //         })
        //                 //         .fold(
        //                 //             Array::<C128, Dim<[usize; RANK]>>::zeros(self.zg.raw_dim()),
        //                 //             |acc, arr| acc + arr
        //                 //         )
        //                 //         .mapv(|x| (x * C128::i()).exp())
        //                 //     })
        //                 //     .collect::<Vec<_>>();
        //                 //     (Some(exp_ks), Some(exp_kqs))
        //                 // } else {
        //                 //     (None, None)
        //                 // };
        //                 let (exp_ks_opt, exp_kqs_opt) = exp_kqs_func;

        //                 // (0..3).for_each(|i| {
        //                 //     match (exp_ks_opt.as_ref(), exp_kqs_opt.as_ref()) {
        //                 //         // Element-wise multiplication. Each element is for a specific
        //                 //         // primitive combination.
        //                 //         (Some(exp_ks), Some(exp_kqs)) => {
        //                 //             ints_r[i][l_tuple][n_tuple] = Some(
        //                 //                 (&pre_zg * &exp_zgs[i] * &exp_ks[i]).mapv(C128::from)
        //                 //                     * &exp_kqs[i]
        //                 //             );
        //                 //         }
        //                 //         _ => {
        //                 //             ints_r[i][l_tuple][n_tuple] = Some(
        //                 //                 (&pre_zg * &exp_zgs[i]).mapv(C128::from)
        //                 //             );
        //                 //         }
        //                 //     }
        //                 // });
        //                 exp_ks_kqs_to_int_func
        //             }
        //             // ~~~~~~~~~~~~~~~~~~
        //             // Initial term ends.
        //             // ~~~~~~~~~~~~~~~~~~

        //             // ~~~~~~~~~~~~~~~~~~~~~~~~
        //             // n-recurrent terms begin.
        //             // ~~~~~~~~~~~~~~~~~~~~~~~~
        //             for r_index in 0..RANK {
        //                 // r_index: recursion index (j in handwritten note)
        //                 let next_n_tuple = {
        //                     let mut new_n_tuple = n_tuple.clone();
        //                     new_n_tuple.iter_mut().enumerate().for_each(|(t, n)| {
        //                         if t == r_index { *n += 1 }
        //                     });
        //                     new_n_tuple
        //                 };
        //                 if !remaining_tuples.remove(&(l_tuple, next_n_tuple)) {
        //                     continue
        //                 }

        //                 (0..3).for_each(|i| {
        //                     // (rg - r_j) * [[:|:]]
        //                     // rg is primitive-combination-specific.
        //                     ints_r[i][l_tuple][next_n_tuple] = Some(
        //                         self.rg.map(|r| dtype::from(r[i] - self.rs[r_index][i]))
        //                         * ints_r[i][l_tuple][n_tuple].as_ref().unwrap_or_else(|| {
        //                             panic!("({l_tuple:?}, {n_tuple:?}) => ({l_tuple:?}, {next_n_tuple:?}) failed.")
        //                         })
        //                     );
        //                 });

        //                 (0..RANK).for_each(|k| {
        //                     // if let Some(kk) = self.ks[k].as_ref() {
        //                     //     // 1 / (2 * zg) * sum(i) ii * k_iα * [[:|:]]
        //                     //     // zg is primitive-combination-specific.
        //                     //     (0..3).for_each(|i| {
        //                     //         let add_term = self.zg.mapv(|zg| {
        //                     //             C128::i() * kk[i] / (2.0 * zg)
        //                     //         }) * ints_r[i][l_tuple][n_tuple].as_ref().unwrap_or_else(|| {
        //                     //             panic!("({l_tuple:?}, {n_tuple:?}) => ({l_tuple:?}, {next_n_tuple:?}) failed.")
        //                     //         });
        //                     //         if let Some(arr) = ints_r[i][l_tuple][next_n_tuple].as_mut() {
        //                     //             Zip::from(arr).and(&add_term).for_each(|a, &t| *a += t);
        //                     //         } else {
        //                     //             ints_r[i][l_tuple][next_n_tuple] = Some(add_term);
        //                     //         }
        //                     //     });
        //                     // };
        //                     if let Some(kk) = self.ks[k].as_ref() {
        //                         n_recur_k_term_func
        //                     }

        //                     if n_tuple[k] > 0 {
        //                         let mut prev_n_tuple_k = n_tuple.clone();
        //                         prev_n_tuple_k.iter_mut().enumerate().for_each(|(t, n)| {
        //                             if t == k { *n -= 1 }
        //                         });
        //                         assert!(!remaining_tuples.contains(&(l_tuple, prev_n_tuple_k)));
        //                         // 1 / (2 * zg) * sum(i) Nα(n_i) * [[n_i - 1_α:|:]]
        //                         (0..3).for_each(|i| {
        //                             let add_term = self.zg.mapv(|zg| {
        //                                 dtype::from(1.0)
        //                                 / (2.0 * zg)
        //                                 * n_tuple[k]
        //                                     .to_f64()
        //                                     .unwrap_or_else(|| panic!("Unable to convert `n_tuple[k]` = {} to `f64`.", n_tuple[k]))
        //                             }) * ints_r[i][l_tuple][prev_n_tuple_k].as_ref().unwrap_or_else(|| {
        //                                 panic!("({l_tuple:?}, {prev_n_tuple_k:?}) => ({l_tuple:?}, {next_n_tuple:?}) failed.")
        //                             });
        //                             if let Some(arr) = ints_r[i][l_tuple][next_n_tuple].as_mut() {
        //                                 Zip::from(arr).and(&add_term).for_each(|a, &t| *a += t);
        //                             } else {
        //                                 ints_r[i][l_tuple][next_n_tuple] = Some(add_term);
        //                             }
        //                         });
        //                     };
        //                 });

        //                 if l_tuple[r_index] > 0 {
        //                     let mut prev_l_tuple = l_tuple.clone();
        //                     prev_l_tuple.iter_mut().enumerate().for_each(|(t, l)| {
        //                         if t == r_index { *l -= 1 }
        //                     });
        //                     assert!(!remaining_tuples.contains(&(prev_l_tuple, n_tuple)));
        //                     // -Nα(l_j) * [[:l_j - 1_α|:]]
        //                     // Note that Nα(l_j) = (l_j)_α.
        //                     (0..3).for_each(|i| {
        //                         let add_term = dtype::from(l_tuple[r_index]
        //                             .to_f64()
        //                             .unwrap_or_else(|| panic!("Unable to convert `l_tuple[r_index]` = {} to `f64`.", l_tuple[r_index]))
        //                         )
        //                         * ints_r[i][prev_l_tuple][n_tuple]
        //                             .as_ref()
        //                             .unwrap_or_else(|| {
        //                                 panic!("({prev_l_tuple:?}, {n_tuple:?}) => ({l_tuple:?}, {next_n_tuple:?}) failed.")
        //                             });
        //                         if let Some(arr) = ints_r[i][l_tuple][next_n_tuple].as_mut() {
        //                             Zip::from(arr).and(&add_term).for_each(|a, &t| *a -= t);
        //                         } else {
        //                             ints_r[i][l_tuple][next_n_tuple] = Some(-add_term);
        //                         }
        //                     });
        //                 }

        //                 (0..RANK).for_each(|k| {
        //                     if l_tuple[k] > 0 {
        //                         let mut prev_l_tuple_k = l_tuple.clone();
        //                         prev_l_tuple_k.iter_mut().enumerate().for_each(|(t, l)| {
        //                             if t == k { *l -= 1 }
        //                         });
        //                         assert!(!remaining_tuples.contains(&(prev_l_tuple_k, n_tuple)));
        //                         // (1 / zg) * sum(g) z_g * Nα(l_g) * [[:l_g - 1_α|:]]
        //                         (0..3).for_each(|i| {
        //                             // let mut zk_zg_i = self.zg.clone();
        //                             // zk_zg_i.indexed_iter_mut().for_each(|(indices, zg)| {
        //                             //     let ($($shell_name),+) = indices;
        //                             //     let indices = [$($shell_name),+];
        //                             //     *zg = self.zs[k][indices[k]] / *zg;
        //                             // });
        //                             let add_term = dtype::from(
        //                                 l_tuple[k]
        //                                     .to_f64()
        //                                     .unwrap_or_else(|| panic!("Unable to convert `l_tuple[k]` = {} to `f64`.", l_tuple[k])))
        //                             // * zk_zg_i.mapv(arr_map_closure)
        //                             / self.zg.mapv(arr_map_closure)
        //                             * &self.zs[k] // broadcasting zs[k] to the shape of zg.
        //                             * ints_r[i][prev_l_tuple_k][n_tuple].as_ref().unwrap_or_else(|| {
        //                                 panic!("({prev_l_tuple_k:?}, {n_tuple:?}) => ({l_tuple:?}, {next_n_tuple:?}) failed.")
        //                             });
        //                             if let Some(arr) = ints_r[i][l_tuple][next_n_tuple].as_mut() {
        //                                 Zip::from(arr).and(&add_term).for_each(|a, &t| *a += t);
        //                             } else {
        //                                 ints_r[i][l_tuple][next_n_tuple] = Some(add_term);
        //                             }
        //                         });
        //                     }
        //                 })
        //             }
        //             // ~~~~~~~~~~~~~~~~~~~~~~
        //             // n-recurrent terms end.
        //             // ~~~~~~~~~~~~~~~~~~~~~~

        //             // ~~~~~~~~~~~~~~~~~~~~~~~~
        //             // l-recurrent terms begin.
        //             // ~~~~~~~~~~~~~~~~~~~~~~~~
        //             if extra_tuples.contains(&(l_tuple, n_tuple)) {
        //                 continue
        //             }
        //             for r_index in 0..RANK {
        //                 // r_index: recursion index (g in handwritten note)
        //                 let next_l_tuple = {
        //                     let mut new_l_tuple = l_tuple.clone();
        //                     new_l_tuple.iter_mut().enumerate().for_each(|(t, l)| {
        //                         if t == r_index { *l += 1 }
        //                     });
        //                     new_l_tuple
        //                 };
        //                 if !remaining_tuples.remove(&(next_l_tuple, n_tuple)) {
        //                     continue
        //                 }

        //                 if let Some(kr) = self.ks[r_index].as_ref() {
        //                     // // -ii * k_gα * [[:|:]]
        //                     // (0..3).for_each(|i| {
        //                     //     let add_term = C128::i()
        //                     //     * kr[i]
        //                     //     * ints_r[i][l_tuple][n_tuple].as_ref().unwrap_or_else(|| {
        //                     //         panic!("({l_tuple:?}, {n_tuple:?}) => ({next_l_tuple:?}, {n_tuple:?}) failed.")
        //                     //     });
        //                     //     if let Some(arr) = ints_r[i][next_l_tuple][n_tuple].as_mut() {
        //                     //         Zip::from(arr).and(&add_term).for_each(|a, &t| *a -= t);
        //                     //     } else {
        //                     //         ints_r[i][next_l_tuple][n_tuple] = Some(-add_term);
        //                     //     }
        //                     // });
        //                     l_recur_k_term_func
        //                 }

        //                 let next_n_tuple = {
        //                     let mut new_n_tuple = n_tuple.clone();
        //                     new_n_tuple.iter_mut().enumerate().for_each(|(t, n)| {
        //                         if t == r_index { *n += 1 }
        //                     });
        //                     new_n_tuple
        //                 };
        //                 assert!(next_n_tuple.iter().enumerate().all(|(t, n)| *n <= ns[t]));
        //                 assert!(!remaining_tuples.contains(&(l_tuple, next_n_tuple)));

        //                 // 2 * z_g * [[n_g + 1_α:|:]]
        //                 (0..3).for_each(|i| {
        //                     let add_term = dtype::from(2.0)
        //                     * ints_r[i][l_tuple][next_n_tuple].as_ref().unwrap_or_else(|| {
        //                         panic!("({l_tuple:?}, {next_n_tuple:?}) => ({next_l_tuple:?}, {n_tuple:?}) failed.")
        //                     })
        //                     * &self.zs[r_index]; // broadcasting zs[r_index] to the shape of
        //                                          // ints_r[i][l_tuple][next_n_tuple].
        //                     if let Some(arr) = ints_r[i][next_l_tuple][n_tuple].as_mut() {
        //                         Zip::from(arr).and(&add_term).for_each(|a, &t| *a += t);
        //                     } else {
        //                         ints_r[i][next_l_tuple][n_tuple] = Some(add_term);
        //                     }
        //                 });

        //                 if n_tuple[r_index] > 0 {
        //                     let mut prev_n_tuple = n_tuple.clone();
        //                     prev_n_tuple.iter_mut().enumerate().for_each(|(t, n)| {
        //                         if t == r_index { *n -= 1 }
        //                     });
        //                     assert!(!remaining_tuples.contains(&(l_tuple, prev_n_tuple)));

        //                     // -Nα(n_g) * [[n_g - 1_α:|:]]
        //                     (0..3).for_each(|i| {
        //                         let add_term = dtype::from(
        //                             n_tuple[r_index]
        //                                 .to_f64()
        //                                 .unwrap_or_else(|| panic!("Unable to convert `n_tuple[r_index]` = {} to `f64`.", n_tuple[r_index]))
        //                         )
        //                         * ints_r[i][l_tuple][prev_n_tuple].as_ref().unwrap_or_else(|| {
        //                             panic!("({l_tuple:?}, {prev_n_tuple:?}) => ({next_l_tuple:?}, {n_tuple:?}) failed.")
        //                         });
        //                         if let Some(arr) = ints_r[i][next_l_tuple][n_tuple].as_mut() {
        //                             Zip::from(arr).and(&add_term).for_each(|a, &t| *a -= t);
        //                         } else {
        //                             ints_r[i][next_l_tuple][n_tuple] = Some(-add_term);
        //                         }
        //                     });
        //                 }
        //             }
        //             // ~~~~~~~~~~~~~~~~~~~~~~
        //             // l-recurrent terms end.
        //             // ~~~~~~~~~~~~~~~~~~~~~~
        //         }
        //         // ~~~~~~~~~~~~~~~~~~~~~~~~~~
        //         // Loop over all tuples ends.
        //         // ~~~~~~~~~~~~~~~~~~~~~~~~~~

        //         // ~~~~~~~~~~~~~~~~~~~~~
        //         // Normalisation begins.
        //         // ~~~~~~~~~~~~~~~~~~~~~
        //         for n_tuple in n_tuples.iter() {
        //             let rank_i32 = RANK
        //                 .to_i32()
        //                 .expect("Unable to convert the tuple rank to `i32`.");
        //             let norm_arr =
        //                 (2.0 / std::f64::consts::PI).sqrt().sqrt().powi(rank_i32)
        //                 * n_tuple.iter().map(|n| {
        //                     let doufac = if *n == 0 {
        //                         1
        //                     } else {
        //                         ((2 * n) - 1)
        //                             .checked_double_factorial()
        //                             .unwrap_or_else(|| panic!("Unable to obtain the double factorial of `{}`.", 2 * n - 1))
        //                     }
        //                     .to_f64()
        //                     .unwrap_or_else(|| panic!("Unable to convert the double factorial of `{}` to `f64.", 2 * n - 1));
        //                     1.0 / doufac.sqrt()
        //                 }).product::<f64>()
        //                 * self.zd.map(|zd| zd.sqrt().sqrt());
        //             let norm_arr = self
        //                 .zs
        //                 .iter()
        //                 .zip(n_tuple.iter())
        //                 .enumerate()
        //                 .fold(norm_arr, |acc, (j, (z, n))| {
        //                     let mut shape = [$(replace_expr!(($shell_name) 1)),+];
        //                     shape[j] = z.len();
        //                     let z_transformed = z.mapv(|z_val| {
        //                         if n.rem_euclid(2) == 0 {
        //                             (4.0 * z_val).powi(
        //                                 (n.div_euclid(2))
        //                                     .to_i32()
        //                                     .expect("Unable to convert `n` to `i32`.")
        //                             )
        //                         } else {
        //                             (4.0 * z_val).powf(
        //                                 n.to_f64().expect("Unable to convert `n` to `f64`.")
        //                                 / 2.0
        //                             )
        //                         }
        //                     })
        //                     .into_shape(shape)
        //                     .expect("Unable to convert transformed `z` to {RANK} dimensions.");
        //                     acc * z_transformed
        //                 });

        //             for l_tuple in l_tuples.iter() {
        //                 (0..3).for_each(|i| {
        //                     if let Some(arr) = ints_r[i][*l_tuple][*n_tuple].as_mut() {
        //                         Zip::from(arr)
        //                             .and(&norm_arr)
        //                             .for_each(|a, &n| *a *= dtype::from(n));
        //                     }
        //                 });
        //             }
        //         }
        //         // ~~~~~~~~~~~~~~~~~~~
        //         // Normalisation ends.
        //         // ~~~~~~~~~~~~~~~~~~~

        //         // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        //         // Population of Cartesian integrals for each derivative component begins.
        //         // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        //         let lex_cart_orders = (0..=*ls.iter().max().expect("Unable to determine the maximum derivative order."))
        //             .map(|l| CartOrder::lex(u32::try_from(l).expect("Unable to convert a derivative order to `u32`.")))
        //             .collect::<Vec<_>>();
        //         let cart_shell_shape = {
        //             let mut cart_shell_shape_iter = self
        //                 .ns
        //                 .iter()
        //                 .map(|n| ((n + 1) * (n + 2)).div_euclid(2));
        //             $(
        //                 let $shell_name = cart_shell_shape_iter
        //                     .next()
        //                     .expect("cart_shell_shape out of range.");
        //             )+
        //             [$($shell_name),+]
        //         };
        //         let cart_shell_blocks = ls
        //             .iter()
        //             .map(|l| 0..((l + 1) * (l + 2).div_euclid(2)))
        //             .multi_cartesian_product()
        //             .map(|l_indices| {
        //                 // ls = [m, n, p, ...]
        //                 // l_indices = [a, b, c, ...]
        //                 //   - a-th component (lexicographic order) of the m-th derivative of the first shell,
        //                 //   - b-th component (lexicographic order) of the n-th derivative of the second shell,
        //                 //   - etc.
        //                 // The derivative components are arranged in lexicographic Cartersian order.
        //                 // If ls = [0, 1, 2], then the a particular l_indices could take the value
        //                 // [0, 2, 3] which represents
        //                 //   - 0th derivative of the first shell
        //                 //   - d/dz of the second shell (x, y, z)
        //                 //   - d2/dyy of the third shell (xx, xy, xz, yy, yz, zz)
        //                 assert_eq!(l_indices.len(), RANK);
        //                 let mut l_indices_iter = l_indices.into_iter();
        //                 $(
        //                     let $shell_name = l_indices_iter
        //                         .next()
        //                         .expect("l index out of range.");
        //                 )+
        //                 let l_indices = [$($shell_name),+];

        //                 // l_powers translates l_indices into tuples of component derivative orders
        //                 // for each shell.
        //                 // For example, with ls = [0, 1, 2] and l_indices = [0, 2, 3],
        //                 // l_powers is given by [(0, 0, 0), (0, 0, 1), (0, 2, 0)].
        //                 let l_powers = {
        //                     let mut l_powers_mut = [$(
        //                         replace_expr!(($shell_name) (0, 0, 0))
        //                     ),+];
        //                     l_powers_mut.iter_mut().enumerate().for_each(|(shell_index, l_power)| {
        //                         *l_power = lex_cart_orders[ls[shell_index]].cart_tuples[l_indices[shell_index]].clone();
        //                     });
        //                     l_powers_mut
        //                 };

        //                 // l_tuples_xyz gives l_tuple for each Cartesian component.
        //                 // With l_powers [(0, 0, 0), (0, 0, 1), (0, 2, 0)],
        //                 // l_tuples_xyz is given by
        //                 // [(0, 0, 0), (0, 0, 2), (0, 1, 0)]
        //                 //  ----x----  ----y----  ----z----
        //                 // which means: take the product of the (0, 0, 0) x-derivative,
        //                 // (0, 0, 2) y-derivative, and (0, 1, 0) z-derivative to give int_xyz.
        //                 // Essentially, l_tuples_xyz is transposed l_powers.
        //                 // l_tuples_xyz will be cloned inside the for loop below because it
        //                 // is consumed after every iteration.
        //                 let outer_l_tuples_xyz = {
        //                     let mut l_tuples_xyz_mut = [[$(replace_expr!(($shell_name) 0usize)),+]; 3];
        //                     l_tuples_xyz_mut[0].iter_mut().enumerate().for_each(|(shell_index, l)| {
        //                         *l = usize::try_from(l_powers[shell_index].0)
        //                             .expect("Unable to convert `l` to `usize`.");
        //                     });
        //                     l_tuples_xyz_mut[1].iter_mut().enumerate().for_each(|(shell_index, l)| {
        //                         *l = usize::try_from(l_powers[shell_index].1)
        //                             .expect("Unable to convert `l` to `usize`.");
        //                     });
        //                     l_tuples_xyz_mut[2].iter_mut().enumerate().for_each(|(shell_index, l)| {
        //                         *l = usize::try_from(l_powers[shell_index].2)
        //                             .expect("Unable to convert `l` to `usize`.");
        //                     });
        //                     l_tuples_xyz_mut
        //                 };

        //                 let mut cart_shell_block = Array::<dtype, Dim<[usize; RANK]>>::zeros(
        //                     cart_shell_shape
        //                 );
        //                 for cart_indices in cart_shell_shape.iter().map(|d| 0..*d).multi_cartesian_product() {
        //                     // cart_indices = [i, j, k, l, ...]
        //                     //   - i-th Cartesian component (shell's specified order) of the first shell,
        //                     //   - j-th Cartesian component (shell's specified order) of the second shell,
        //                     //   - etc.
        //                     // If a shell has pure ordering, a lexicographic Cartesian order will
        //                     // be used. Integrals involving this shell will be converted back to
        //                     // pure form later.
        //                     // If shell_tuple.ns = [0, 2, 3, 1], then the a particular cart_indices could
        //                     // take the value [0, 2, 10, 1] which represents
        //                     //   - s function on the first shell
        //                     //   - dxz function on the second shell
        //                     //   - fzzz function on the third shell
        //                     //   - py function on the fourth shell
        //                     let mut cart_indices_iter = cart_indices.into_iter();
        //                     $(
        //                         let $shell_name = cart_indices_iter
        //                             .next()
        //                             .expect("cart_index out of range.");
        //                     )+
        //                     let cart_indices = [$($shell_name),+];

        //                     // cart_powers translates cart_indices into tuples of Cartesian powers
        //                     // for each shell.
        //                     // For example, with shell_tuple.ns = (0, 2, 3, 1) and
        //                     // cart_indices = (0, 2, 10, 1), cart_powers is given by
        //                     // [(0, 0, 0), (1, 0, 1), (0, 0, 3), (0, 1, 0)] (assuming
        //                     // lexicographic ordering).
        //                     let cart_powers = {
        //                         let mut cart_powers_mut = [$(
        //                             replace_expr!(($shell_name) (0, 0, 0))
        //                         ),+];
        //                         cart_powers_mut.iter_mut().enumerate().for_each(|(shell_index, cart_power)| {
        //                             let cart_order = match &self
        //                                 .shells[shell_index].0
        //                                 .basis_shell()
        //                                 .shell_order {
        //                                     ShellOrder::Pure(po) => CartOrder::lex(po.lpure),
        //                                     ShellOrder::Cart(co) => co.clone()
        //                                 };
        //                             *cart_power = cart_order
        //                                 .cart_tuples[cart_indices[shell_index]]
        //                                 .clone();
        //                         });
        //                         cart_powers_mut
        //                     };

        //                     // n_tuples_xyz gives n_tuple for each Cartesian component.
        //                     // With cart_powers = [(0, 0, 0), (1, 0, 1), (0, 0, 3), (0, 1, 0)],
        //                     // n_tuples_xyz is given by
        //                     // [(0, 1, 0, 0), (0, 0, 0, 1), (0, 1, 3, 0)]
        //                     //  -----x------  -----y------  -----z------
        //                     // which means: take the product of the (0, 1, 0, 0) x-integral,
        //                     // (0, 0, 0, 1) y-integral, and (0, 1, 3, 0) z-integral to give int_xyz.
        //                     // Essentially, n_tuples_xyz is transposed cart_powers.
        //                     let l_tuples_xyz = outer_l_tuples_xyz.clone();
        //                     let n_tuples_xyz = {
        //                         let mut n_tuples_xyz_mut = [[$(replace_expr!(($shell_name) 0usize)),+]; 3];
        //                         n_tuples_xyz_mut[0].iter_mut().enumerate().for_each(|(shell_index, n)| {
        //                             *n = usize::try_from(cart_powers[shell_index].0)
        //                                 .expect("Unable to convert `n` to `usize`.");
        //                         });
        //                         n_tuples_xyz_mut[1].iter_mut().enumerate().for_each(|(shell_index, n)| {
        //                             *n = usize::try_from(cart_powers[shell_index].1)
        //                                 .expect("Unable to convert `n` to `usize`.");
        //                         });
        //                         n_tuples_xyz_mut[2].iter_mut().enumerate().for_each(|(shell_index, n)| {
        //                             *n = usize::try_from(cart_powers[shell_index].2)
        //                                 .expect("Unable to convert `n` to `usize`.");
        //                         });
        //                         n_tuples_xyz_mut
        //                     };
        //                     let int_xyz = izip!(l_tuples_xyz.iter(), n_tuples_xyz.iter())
        //                         .enumerate()
        //                         .map(|(i, (l_tuple, n_tuple))| {
        //                             ints_r[i][*l_tuple][*n_tuple].as_ref()
        //                         })
        //                         .collect::<Option<Vec<_>>>()
        //                         .map(|arrs| arrs.into_iter().fold(
        //                             Array::<dtype, Dim<[usize; RANK]>>::ones(
        //                                 self.primitive_shell_shape
        //                             ),
        //                             |acc, arr| acc * arr
        //                         ))
        //                         .unwrap_or_else(
        //                             || Array::<dtype, Dim<[usize; RANK]>>::zeros(
        //                                 self.primitive_shell_shape
        //                             )
        //                         );

        //                     let contraction_str = (0..RANK)
        //                         .map(|i| (i.to_u8().expect("Unable to convert a shell index to `u8`.") + 97) as char)
        //                         .collect::<String>();
        //                     cart_shell_block[cart_indices] = einsum(
        //                         &format!("{contraction_str},{contraction_str}->"),
        //                         &[&int_xyz, &self.dd.mapv(arr_map_closure)]
        //                     )
        //                         .expect("Unable to contract `int_xyz` with `dd`.")
        //                         .into_iter()
        //                         .next()
        //                         .expect("Unable to retrieve the result of the contraction between `int_xyz` and `dd`.");
        //                 }
        //                 cart_shell_block
        //             }).collect::<Vec<_>>();
        //         // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        //         // Population of Cartesian integrals for each derivative component ends.
        //         // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        //         cart_shell_blocks
        //     }
        // }

        /// A structure to handle all possible shell tuples for a particular type of integral.
        struct ShellTupleCollection<'a, D: Dimension, T> {
            shell_tuples: Array<ShellTuple<'a, D, T>, Dim<[usize; RANK]>>,

            lmax: u32,

            ccs: [bool; RANK],

            n_shells: [usize; RANK],
        }

        impl<'a, D: Dimension, T> ShellTupleCollection<'a, D, T> {
            /// The maximum angular momentum across all shell tuples.
            fn lmax(&self) -> u32 {
                self.lmax
            }

            /// The number of shells in each tuple in the collection.
            fn rank(&self) -> usize {
                RANK
            }

            /// Returns an iterator of the shell tuples unique with respect to permutations of the
            /// constituent shells, taking into account complex conjugation, symmetry
            /// transformation, and derivative patterns.
            ///
            /// # Arguments
            ///
            /// * `ls` - The derivative pattern.
            ///
            /// # Returns
            ///
            /// A vector of the unique shell tuples.
            fn unique_shell_tuples_iter<'it>(
                &'it self, ls: [usize; RANK]
            ) -> UniqueShellTupleIterator<'it, 'a, D, T>
                where 'a: 'it
            {
                // Example:
                //     ccs = [true, true, false, true, false]
                //     ls  = [   1,    1,     0,    2,     0]
                //     sym = [   1,    0,     0,    0,     0] -- not considered for now
                //     nsh = [   2,    1,     2,    2,     2]
                //           i.e. Each shell position has three possible shells.

                // Each shell type is a tuple of its complex-conjugationness, its derivative
                // order, and its length. The vector `shell_types` collects these tuples together.
                // Example:
                // shell_types = [
                //     (true, 1, 3), (true, 1, 4), (false, 0, 3), (true, 2, 3), (false, 0, 3)
                // ].
                // We see that there are four types here, and only shell positions that
                // have the same type have permutation equivalence, i.e. [0, 1], [2, 4], [3].
                let shell_types: Vec<(bool, usize, usize)>
                    = izip!(self.ccs, ls, self.n_shells).collect::<Vec<_>>();

                // The map `shell_types_classified` keeps track of the unique shell types in this
                // shell tuple and the associated shell positions as tuples.
                // Example:
                // shell_types_classified = {
                //     (true , 1, 2): {0},
                //     (true , 1, 1): {1},
                //     (false, 0, 2): {2, 4},
                //     (true , 2, 2): {3}
                // }.
                let mut shell_types_classified: IndexMap<(bool, usize, usize), IndexSet<usize>>
                    = IndexMap::new();
                shell_types.into_iter().enumerate().for_each(|(shell_index, shell_type)| {
                    shell_types_classified.entry(shell_type).or_default().insert(shell_index);
                });

                // The map `shell_indices_unique_combinations` maps, for each shell type, the
                // corresponding indices of shells of that type to the unique index combinations.
                // Example:
                // shell_type_unique_combinations = {
                //     [0]   : [[0], [1]],
                //     [1]   : [[0]],
                //     [2, 4]: [[0, 0], [0, 1], [1, 1]],
                //     [3]   : [[0], [1]],
                // }
                let shell_indices_unique_combinations = shell_types_classified
                    .iter()
                    .map(|(shell_type, shell_indices)| {
                        (
                            shell_indices.iter().collect::<Vec<_>>(),
                            (0..shell_type.2)
                                .combinations_with_replacement(shell_indices.len())
                                .collect::<Vec<_>>()
                        )
                    })
                    .collect::<IndexMap<_, _>>();

                // Example:
                // sis = [0, 1, 2, 4, 3]
                // gg = [
                //     [[0], [1]],
                //     [[0]],
                //     [[0, 0], [0, 1], [1, 1]],
                //     [[0], [1]]
                // ]
                // order = [0, 1, 2, 4, 3]
                let sis = shell_indices_unique_combinations.keys().flatten().collect::<Vec<_>>();
                let mut order = (0..sis.len()).collect::<Vec<_>>();
                order.sort_by_key(|&i| &sis[i]);
                let gg = shell_indices_unique_combinations.into_values().collect::<Vec<_>>();

                // `unordered_recombined_shell_indices` forms all possible combinations of
                // shell indices across all different shell types.
                // Example:
                // unordered_recombined_shell_indices = [
                //     [[0], [0], [0, 0], [0]],
                //     [[0], [0], [0, 0], [1]],
                //     [[0], [0], [0, 1], [0]],
                //     [[0], [0], [0, 1], [1]],
                //     [[0], [0], [1, 1], [0]],
                //     [[0], [0], [1, 1], [1]],
                //     ...
                // ] (12 = 2 * 1 * 3 * 2 terms in total)
                let unordered_recombined_shell_indices = gg
                    .into_iter()
                    .multi_cartesian_product()
                    .into_iter()
                    .collect::<Vec<_>>();

                UniqueShellTupleIterator::<'it, 'a, D, T> {
                    index: 0,
                    shell_order: order,
                    unordered_recombined_shell_indices,
                    shell_tuples: &self.shell_tuples
                }
            }
        }

        struct UniqueShellTupleIterator<'it, 'a: 'it, D: Dimension, T> {
            index: usize,
            shell_order: Vec<usize>,
            unordered_recombined_shell_indices: Vec<Vec<Vec<usize>>>,
            shell_tuples: &'it Array<ShellTuple<'a, D, T>, Dim<[usize; RANK]>>,
        }

        impl<'it, 'a: 'it, D: Dimension, T> Iterator for UniqueShellTupleIterator<'it, 'a, D, T> {
            type Item = (&'it ShellTuple<'a, D, T>, Vec<Vec<usize>>);

            fn next(&mut self) -> Option<Self::Item> {
                let unordered_shell_index = self.unordered_recombined_shell_indices.get(self.index)?;

                // Now, for each term in `unordered_recombined_shell_indices`, we need to
                // flatten and then reorder to put the shell indices at the correct positions. This
                // gives `ordered_shell_index`.
                let flattened_unordered_shell_index = unordered_shell_index
                    .clone()
                    .into_iter()
                    .flatten()
                    .collect::<Vec<_>>();
                let ordered_shell_index = self.shell_order.iter().map(|i| {
                    flattened_unordered_shell_index[*i]
                }).collect::<Vec<_>>();
                assert_eq!(ordered_shell_index.len(), RANK);
                let mut ordered_shell_index_iter = ordered_shell_index.into_iter();
                $(
                    let $shell_name = ordered_shell_index_iter
                        .next()
                        .expect("Shell index out of range.");
                )+

                // For each term in `unordered_recombined_shell_indices`, all unique
                // permutations of each sub-vector gives an equivalent permutation.
                // Example: consider [[0], [0], [0, 1], [1]]. This gives the following
                // equivalent permutations:
                //   [[0], [0], [0, 1], [1]]
                //   [[0], [0], [1, 0], [1]]
                // There are two of them (1 * 1 * 2 * 1).
                // Each equivalent permutation undergoes the same 'flattening' and
                // 'reordering' process as for the unique term.
                let equiv_terms = unordered_shell_index
                    .iter()
                    .map(|y| y.into_iter().permutations(y.len()).into_iter().unique())
                    .multi_cartesian_product()
                    .into_iter()
                    .map(|x| x.into_iter().flatten().cloned().collect::<Vec<_>>())
                    .collect::<Vec<_>>();

                self.index += 1;
                Some((&self.shell_tuples[[$($shell_name),+]], equiv_terms))
            }
        }
    }
}

macro_rules! build_shell_tuple {
    ( $($shell:expr),+; $ty:ty ) => {
        {
            use itertools::Itertools;

            use crate::angmom::sh_conversion::sh_rl2cart_mat;
            use crate::basis::ao::CartOrder;

            let zg = {
                let arr_vec = [$(
                    $shell.0.contraction.primitives.iter().map(|(e, _)| e).collect::<Vec<_>>()
                ),+].into_iter()
                    .multi_cartesian_product()
                    .map(|s| s.into_iter().sum()).collect::<Vec<_>>();
                let arr = Array::<f64, Dim<[usize; RANK]>>::from_shape_vec(
                    ($($shell.0.contraction.primitives.len()),+), arr_vec
                ).unwrap_or_else(|err| {
                    log::error!("{err}");
                    panic!("Unable to construct the {RANK}-dimensional array of exponent sums.")
                });
                arr
            };

            let rg = {
                let arr_vec = [$(
                    $shell
                        .0
                        .contraction.primitives
                        .iter()
                        .map(|(e, _)| *e * $shell.0.cart_origin).collect::<Vec<_>>()
                ),+].into_iter()
                    .multi_cartesian_product()
                    .map(|s| {
                        s.into_iter()
                            .fold(Point3::origin(), |acc, r| acc + r.coords)
                    })
                    .collect::<Vec<_>>();
                let arr = Array::<Point3<f64>, Dim<[usize; RANK]>>::from_shape_vec(
                    ($($shell.0.contraction.primitives.len()),+), arr_vec
                ).unwrap_or_else(|err| {
                    log::error!("{err}");
                    panic!("Unable to construct the {RANK}-dimensional array of exponent-weighted centres.")
                }) / &zg;
                arr
            };

            let qs = [$(
                $shell.0.k().map(|_| rg.map(|r| (r - $shell.0.cart_origin().coords).coords))
            ),+];

            ShellTuple::<Dim<[usize; RANK]>, $ty> {
                dim: PhantomData,

                typ: PhantomData,

                shells: [$($shell),+],

                function_shell_shape: [$($shell.0.basis_shell().n_funcs()),+],

                primitive_shell_shape: [$($shell.0.contraction_length()),+],

                shell_boundaries: [$(
                    ($shell.0.start_index, $shell.0.start_index + $shell.0.basis_shell().n_funcs())
                ),+],

                rs: [$($shell.0.cart_origin()),+],

                ks: [$(
                        if $shell.1 {
                            $shell.0.k().copied().map(|k| -k)
                        } else {
                            $shell.0.k().copied()
                        }
                    ),+],

                k: [$(
                        if $shell.1 {
                            $shell.0.k().copied().map(|k| -k)
                        } else {
                            $shell.0.k().copied()
                        }
                    ),+]
                    .into_iter()
                    .filter_map(|k| k)
                    .fold(Vector3::zeros(), |acc, k| acc + k),

                ns: [$(
                        usize::try_from($shell.0.basis_shell().l)
                            .expect("Unable to convert an angular momentum `l` value to `usize`.")
                    ),+],

                rl2carts: [$(
                    match &$shell.0.basis_shell().shell_order {
                        ShellOrder::Cart(_) => None,
                        ShellOrder::Pure(po) => Some(sh_rl2cart_mat(
                            po.lpure,
                            po.lpure,
                            &CartOrder::lex(po.lpure),
                            true,
                            &po
                        )),
                    }
                ),+],

                zs: [$(Array1::from_iter($shell.0.contraction.primitives.iter().map(|(e, _)| e))),+],

                zg,

                zd: {
                    let arr_vec = [$(
                        $shell.0.contraction.primitives.iter().map(|(e, _)| e).collect::<Vec<_>>()
                    ),+].into_iter()
                        .multi_cartesian_product()
                        .map(|s| s.into_iter().fold(1.0, |acc, e| acc * e)).collect::<Vec<_>>();
                    let arr = Array::<f64, Dim<[usize; RANK]>>::from_shape_vec(
                        ($($shell.0.contraction.primitives.len()),+), arr_vec
                    ).unwrap_or_else(|err| {
                        log::error!("{err}");
                        panic!("Unable to construct the {RANK}-dimensional array of exponent products.")
                    });
                    arr
                },

                ds: [$(Array1::from_iter($shell.0.contraction.primitives.iter().map(|(_, c)| c))),+],

                dd: {
                    let arr_vec = [$(
                        $shell.0.contraction.primitives.iter().map(|(_, c)| c).collect::<Vec<_>>()
                    ),+].into_iter()
                        .multi_cartesian_product()
                        .map(|s| s.into_iter().fold(1.0, |acc, c| acc * c)).collect::<Vec<_>>();
                    let arr = Array::<f64, Dim<[usize; RANK]>>::from_shape_vec(
                        ($($shell.0.contraction.primitives.len()),+), arr_vec
                    ).unwrap_or_else(|err| {
                        log::error!("{err}");
                        panic!("Unable to construct the {RANK}-dimensional array of coefficient products.")
                    });
                    arr
                },

                rg,

                qs,
            }
        }
    }
}

macro_rules! build_shell_tuple_collection {
    ( <$($shell_name:ident),+>; $($shell_cc:expr),+; $($shells:expr),+; $ty:ty ) => {
        {
            use itertools::iproduct;

            define_shell_tuple![<$($shell_name),+>];

            let shell_tuples = {
                let arr_vec = iproduct!($($shells.iter()),+)
                    .map(|shell_tuple| {
                        let ($($shell_name),+) = shell_tuple;
                        build_shell_tuple!($((*$shell_name, $shell_cc)),+; $ty)
                    })
                    .collect::<Vec<_>>();
                let arr = Array::<ShellTuple<Dim<[usize; RANK]>, $ty>, Dim<[usize; RANK]>>::from_shape_vec(
                    ($($shells.len()),+), arr_vec
                ).unwrap_or_else(|err| {
                    log::error!("{err}");
                    panic!("Unable to construct the {RANK}-dimensional array of shell tuples.")
                });
                arr
            };

            let lmax: u32 = *[$(
                $shells.iter().map(|shell| shell.basis_shell.l).collect::<Vec<_>>()
            ),+].iter()
                .flatten()
                .max()
                .expect("Unable to determine the maximum angular momentum across all shells.");

            log::debug!("Rank-{RANK} shell tuple collection construction:");
            log::debug!(
                "  Total number of tuples: {}",
                shell_tuples.shape().iter().fold(1, |acc, s| acc * s)
            );
            ShellTupleCollection::<Dim<[usize; RANK]>, $ty> {
                shell_tuples,
                lmax,
                ccs: [$($shell_cc),+],
                n_shells: [$($shells.len()),+]
            }
        }
    }
}

#[cfg(test)]
#[path = "shell_tuple_tests.rs"]
mod shell_tuple_tests;
