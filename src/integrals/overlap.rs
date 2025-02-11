//! Atomic-orbital $`n`$-centre overlap evaluations.

/// Implements the `overlap` method for [`crate::integrals::shell_tuple::ShellTuple`] and
/// [`crate::integrals::shell_tuple::ShellTupleCollection`] of a specified pattern.
macro_rules! impl_shell_tuple_overlap {
    ( $RANK:ident, <$($shell_name:ident),+> ) => {
        #[duplicate_item(
            [
                dtype [ f64 ]
                zg_type [ &self.zg ]
                dd_type [ &self.dd ]
                rl2cart_type [
                    self.rl2carts[i]
                        .as_ref()
                        .unwrap_or_else(|| panic!("Transformation matrix to convert shell {i} to spherical order not found."))
                ]
                exp_kqs_func [
                    if self.ks.iter().any(|k| k.is_some()) {
                        panic!("Real-valued overlaps cannot handle plane-wave vectors.")
                    } else {
                        (
                            None::<Vec<Array<f64, Dim<[usize; $RANK]>>>>,
                            None::<Vec<Array<f64, Dim<[usize; $RANK]>>>>,
                        )
                    }
                ]
                exp_ks_kqs_to_int_func [
                    (0..3).for_each(|i| {
                        match (exp_ks_opt.as_ref(), exp_kqs_opt.as_ref()) {
                            (Some(_), _) | (_, Some(_)) => {
                                panic!("Real-valued overlaps cannot handle plane-wave vectors.")
                            }
                            (None, None) => {
                                ints_r[i][l_tuple][n_tuple] = Some(
                                    &pre_zg * &exp_zgs[i]
                                );
                            }
                        }
                    })
                ]
                n_recur_k_term_kk [ _ ]
                n_recur_k_term_func [
                    panic!("Real-valued overlaps cannot handle plane-wave vectors.")
                ]
                l_recur_k_term_kr [ _ ]
                l_recur_k_term_func [
                    panic!("Real-valued overlaps cannot handle plane-wave vectors.")
                ]
            ]
            [
                dtype [ C128 ]
                zg_type [ self.zg.mapv(C128::from) ]
                dd_type [ &self.dd.mapv(C128::from) ]
                rl2cart_type [
                    &self.rl2carts[i]
                        .as_ref()
                        .unwrap_or_else(|| panic!("Transformation matrix to convert shell {i} to spherical order not found."))
                        .mapv(C128::from)
                ]
                exp_kqs_func [
                    if self.ks.iter().any(|k| k.is_some()) {
                        // exp_ks = exp(-|k|^2 / 4 zg)
                        // exp_ks[i] is the contribution from the ith Cartesian component.
                        // zg is primitive-combination-specific.
                        let exp_ks = (0..3).map(|i| {
                            self.zg.mapv(|zg| {
                                (-self.k[i].abs().powi(2) / (4.0 * zg)).exp()
                            })
                        }).collect::<Vec<_>>();

                        // exp_kqs = exp(ii * sum(j) k_j · q_j)
                        // exp_kqs[i] is the contribution from the ith Cartesian component.
                        // q_j is primitive-combination-specific.
                        let exp_kqs = (0..3).map(|i| {
                            (0..$RANK).filter_map(|j| {
                                match (self.ks[j], self.qs[j].as_ref()) {
                                    (Some(kj), Some(qj)) => {
                                        Some(
                                            qj.mapv(|qjj| kj[i] * qjj[i])
                                        )
                                    }
                                    _ => None
                                }
                            })
                            .fold(
                                Array::<C128, Dim<[usize; $RANK]>>::zeros(self.zg.raw_dim()),
                                |acc, arr| acc + arr
                            )
                            .mapv(|x| (x * C128::i()).exp())
                        })
                        .collect::<Vec<_>>();
                        (Some(exp_ks), Some(exp_kqs))
                    } else {
                        (None, None)
                    }
                ]
                exp_ks_kqs_to_int_func [
                    (0..3).for_each(|i| {
                        match (exp_ks_opt.as_ref(), exp_kqs_opt.as_ref()) {
                            // Element-wise multiplication. Each element is for a specific
                            // primitive combination.
                            (Some(exp_ks), Some(exp_kqs)) => {
                                ints_r[i][l_tuple][n_tuple] = Some(
                                    (&pre_zg * &exp_zgs[i] * &exp_ks[i]).mapv(C128::from)
                                        * &exp_kqs[i]
                                );
                            }
                            _ => {
                                ints_r[i][l_tuple][n_tuple] = Some(
                                    (&pre_zg * &exp_zgs[i]).mapv(C128::from)
                                );
                            }
                        }
                    })
                ]
                n_recur_k_term_kk [ kk ]
                n_recur_k_term_func [
                    // 1 / (2 * zg) * sum(i) ii * k_iα * [[:|:]]
                    // zg is primitive-combination-specific.
                    (0..3).for_each(|i| {
                        let add_term = self.zg.mapv(|zg| {
                            C128::i() * kk[i] / (2.0 * zg)
                        }) * ints_r[i][l_tuple][n_tuple].as_ref().unwrap_or_else(|| {
                            panic!("({l_tuple:?}, {n_tuple:?}) => ({l_tuple:?}, {next_n_tuple:?}) failed.")
                        });
                        if let Some(arr) = ints_r[i][l_tuple][next_n_tuple].as_mut() {
                            Zip::from(arr).and(&add_term).for_each(|a, &t| *a += t);
                        } else {
                            ints_r[i][l_tuple][next_n_tuple] = Some(add_term);
                        }
                    });
                ]
                l_recur_k_term_kr [ kr ]
                l_recur_k_term_func [
                    // -ii * k_gα * [[:|:]]
                    (0..3).for_each(|i| {
                        let add_term = C128::i()
                        * kr[i]
                        * ints_r[i][l_tuple][n_tuple].as_ref().unwrap_or_else(|| {
                            panic!("({l_tuple:?}, {n_tuple:?}) => ({next_l_tuple:?}, {n_tuple:?}) failed.")
                        });
                        if let Some(arr) = ints_r[i][next_l_tuple][n_tuple].as_mut() {
                            Zip::from(arr).and(&add_term).for_each(|a, &t| *a -= t);
                        } else {
                            ints_r[i][next_l_tuple][n_tuple] = Some(-add_term);
                        }
                    });
                ]
            ]
        )]
        impl<'a> ShellTuple<'a, $RANK, dtype> {
            /// Calculates the overlap arrays for this shell tuple.
            ///
            /// # Arguments
            ///
            /// * `ls` - The derivative pattern.
            ///
            /// # Returns
            ///
            /// A vector of overlap arrays, each of which is for one derivative component.
            pub(crate) fn overlap(
                &self, ls: [usize; $RANK]
            ) -> Vec<Array<dtype, Dim<[usize; $RANK]>>> {
                // ~~~~~~~~~~~~~~~~~~~
                // Preparation begins.
                // ~~~~~~~~~~~~~~~~~~~

                // We require extra Cartesian degrees to calculate derivatives, because each
                // derivative order increases a corresponding Cartesian rank by one.
                let ns: [usize; $RANK] = if ls.iter().any(|l| *l > 0) {
                    let mut ns = self.ns.clone();
                    ns.iter_mut().for_each(|n| *n += 1);
                    ns
                } else {
                    self.ns.clone()
                };

                // Generate all terms in recurrence series
                // First index: Cartesian component
                // Next stc.rank indices: l-recursion indices
                // Next stc.rank indices: n-recursion indices
                // Last stc.rank indices: primitive indices
                // E.g.: rank 3,
                //   ints_r[1][(0, 0, 1)][(0, 1, 2)][(3, 8, 7)]: y-component integral value with
                //     0th y-derivative of 0th Cartesian y-power of 3rd primitive on first shell,
                //     0th y-derivative of 1st Cartesian y-power of 8th primitive on second shell, and
                //     1st y-derivative of 2nd Cartesian y-power of 7th primitive on third shell
                let lrecursion_shape = {
                    let mut ls_mut = ls.clone();
                    ls_mut.iter_mut().for_each(|l| *l += 1);
                    ls_mut
                };
                let nrecursion_shape = {
                    let mut ns_mut = ns.clone();
                    ns_mut.iter_mut().for_each(|n| *n += 1);
                    ns_mut
                };
                let arr = Array::<_, Dim<[usize; $RANK]>>::from_elem(
                    lrecursion_shape, Array::<_, Dim<[usize; $RANK]>>::from_elem(
                        nrecursion_shape, None::<Array::<dtype, Dim<[usize; $RANK]>>>
                    )
                );
                let mut ints_r = [arr.clone(), arr.clone(), arr];

                let default_tuple = [$(replace_expr!(($shell_name) 0)),+];
                let l_tuples = ls
                    .iter()
                    .map(|l| 0..=*l)
                    .multi_cartesian_product()
                    .map(|ltuple| {
                        let mut ltuple_arr = default_tuple.clone();
                        ltuple_arr.iter_mut().enumerate().for_each(|(i, l)| *l = ltuple[i]);
                        ltuple_arr
                    })
                    .collect::<Vec<_>>();
                let n_tuples = ns
                    .iter()
                    .map(|n| 0..=*n)
                    .multi_cartesian_product()
                    .map(|ntuple| {
                        let mut ntuple_arr = default_tuple.clone();
                        ntuple_arr.iter_mut().enumerate().for_each(|(i, n)| *n = ntuple[i]);
                        ntuple_arr
                    })
                    .collect::<Vec<_>>();
                let n_tuples_noextra = ns
                    .iter()
                    .map(|n| 0..*n)
                    .multi_cartesian_product()
                    .map(|ntuple| {
                        let mut ntuple_arr = default_tuple.clone();
                        ntuple_arr.iter_mut().enumerate().for_each(|(i, n)| *n = ntuple[i]);
                        ntuple_arr
                    })
                    .collect::<Vec<_>>();

                let all_tuples = l_tuples.iter().cloned().cartesian_product(
                    n_tuples.iter().cloned()
                ).into_iter().collect::<IndexSet<_>>();
                let mut remaining_tuples = all_tuples.clone();

                let remaining_tuples_noextra = l_tuples.iter().cloned().cartesian_product(
                    n_tuples_noextra.iter().cloned()
                ).into_iter().collect::<IndexSet<_>>();

                let extra_tuples = all_tuples
                    .difference(&remaining_tuples_noextra)
                    .cloned()
                    .collect::<IndexSet<_>>();
                // ~~~~~~~~~~~~~~~~~
                // Preparation ends.
                // ~~~~~~~~~~~~~~~~~

                // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                // Loop over all tuples begins.
                // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                for (tuple_index, (l_tuple, n_tuple)) in all_tuples.into_iter().enumerate() {
                    // ~~~~~~~~~~~~~~~~~~~~
                    // Initial term begins.
                    // ~~~~~~~~~~~~~~~~~~~~
                    if tuple_index == 0 {
                        debug_assert!(remaining_tuples.shift_remove(&(l_tuple, n_tuple)));

                        // pre_zg = sqrt(pi / zg)
                        // zg is primitive-combination-specific.
                        let pre_zg = self.zg.mapv(|zg| (std::f64::consts::PI / zg).sqrt());

                        // exp_zgs = sum(g < h) [ -(z_g * z_h) / zg * |r_g - r_h|^2 ]
                        // exp_zgs[i] is the contribution from the ith Cartesian component.
                        // z_g, z_h, and zg are primitive-combination-specific.
                        let exp_zgs = (0..3).map(|i| {
                            let mut exp_zg_i = self.zg.clone();
                            exp_zg_i.indexed_iter_mut().for_each(|(indices, zg)| {
                                let ($($shell_name),+) = indices;
                                let indices = [$($shell_name),+];
                                *zg = (
                                    -1.0
                                    / *zg
                                    * (0..$RANK).flat_map(|g| ((g + 1)..$RANK).map(move |h| {
                                        self.zs[g][indices[g]]
                                            * self.zs[h][indices[h]]
                                            * (self.rs[g][i] - self.rs[h][i]).powi(2)
                                    })).sum::<f64>()
                                ).exp();
                            });
                            exp_zg_i
                        }).collect::<Vec<_>>();

                        // let (exp_ks_opt, exp_kqs_opt) = if self.ks.iter().any(|k| k.is_some()) {
                        //     // exp_ks = exp(-|k|^2 / 4 zg)
                        //     // exp_ks[i] is the contribution from the ith Cartesian component.
                        //     // zg is primitive-combination-specific.
                        //     let exp_ks = (0..3).map(|i| {
                        //         self.zg.mapv(|zg| {
                        //             (-self.k[i].abs().powi(2) / (4.0 * zg)).exp()
                        //         })
                        //     }).collect::<Vec<_>>();

                        //     // exp_kqs = exp(ii * sum(j) k_j · q_j)
                        //     // exp_kqs[i] is the contribution from the ith Cartesian component.
                        //     // q_j is primitive-combination-specific.
                        //     let exp_kqs = (0..3).map(|i| {
                        //         (0..$RANK).filter_map(|j| {
                        //             match (self.ks[j], self.qs[j].as_ref()) {
                        //                 (Some(kj), Some(qj)) => {
                        //                     Some(
                        //                         qj.mapv(|qjj| kj[i] * qjj[i])
                        //                     )
                        //                 }
                        //                 _ => None
                        //             }
                        //         })
                        //         .fold(
                        //             Array::<C128, Dim<[usize; $RANK]>>::zeros(self.zg.raw_dim()),
                        //             |acc, arr| acc + arr
                        //         )
                        //         .mapv(|x| (x * C128::i()).exp())
                        //     })
                        //     .collect::<Vec<_>>();
                        //     (Some(exp_ks), Some(exp_kqs))
                        // } else {
                        //     (None, None)
                        // };
                        let (exp_ks_opt, exp_kqs_opt) = exp_kqs_func;

                        // (0..3).for_each(|i| {
                        //     match (exp_ks_opt.as_ref(), exp_kqs_opt.as_ref()) {
                        //         // Element-wise multiplication. Each element is for a specific
                        //         // primitive combination.
                        //         (Some(exp_ks), Some(exp_kqs)) => {
                        //             ints_r[i][l_tuple][n_tuple] = Some(
                        //                 (&pre_zg * &exp_zgs[i] * &exp_ks[i]).mapv(C128::from)
                        //                     * &exp_kqs[i]
                        //             );
                        //         }
                        //         _ => {
                        //             ints_r[i][l_tuple][n_tuple] = Some(
                        //                 (&pre_zg * &exp_zgs[i]).mapv(C128::from)
                        //             );
                        //         }
                        //     }
                        // });
                        exp_ks_kqs_to_int_func
                    }
                    // ~~~~~~~~~~~~~~~~~~
                    // Initial term ends.
                    // ~~~~~~~~~~~~~~~~~~

                    // ~~~~~~~~~~~~~~~~~~~~~~~~
                    // n-recurrent terms begin.
                    // ~~~~~~~~~~~~~~~~~~~~~~~~
                    for r_index in 0..$RANK {
                        // r_index: recursion index (j in handwritten note)
                        let next_n_tuple = {
                            let mut new_n_tuple = n_tuple.clone();
                            new_n_tuple.iter_mut().enumerate().for_each(|(t, n)| {
                                if t == r_index { *n += 1 }
                            });
                            new_n_tuple
                        };
                        if !remaining_tuples.shift_remove(&(l_tuple, next_n_tuple)) {
                            continue
                        }

                        (0..3).for_each(|i| {
                            // (rg - r_j) * [[:|:]]
                            // rg is primitive-combination-specific.
                            ints_r[i][l_tuple][next_n_tuple] = Some(
                                self.rg.map(|r| dtype::from(r[i] - self.rs[r_index][i]))
                                * ints_r[i][l_tuple][n_tuple].as_ref().unwrap_or_else(|| {
                                    panic!("({l_tuple:?}, {n_tuple:?}) => ({l_tuple:?}, {next_n_tuple:?}) failed.")
                                })
                            );
                        });

                        (0..$RANK).for_each(|k| {
                            // if let Some(kk) = self.ks[k].as_ref() {
                            //     // 1 / (2 * zg) * sum(i) ii * k_iα * [[:|:]]
                            //     // zg is primitive-combination-specific.
                            //     (0..3).for_each(|i| {
                            //         let add_term = self.zg.mapv(|zg| {
                            //             C128::i() * kk[i] / (2.0 * zg)
                            //         }) * ints_r[i][l_tuple][n_tuple].as_ref().unwrap_or_else(|| {
                            //             panic!("({l_tuple:?}, {n_tuple:?}) => ({l_tuple:?}, {next_n_tuple:?}) failed.")
                            //         });
                            //         if let Some(arr) = ints_r[i][l_tuple][next_n_tuple].as_mut() {
                            //             Zip::from(arr).and(&add_term).for_each(|a, &t| *a += t);
                            //         } else {
                            //             ints_r[i][l_tuple][next_n_tuple] = Some(add_term);
                            //         }
                            //     });
                            // };
                            if let Some(n_recur_k_term_kk) = self.ks[k].as_ref() {
                                n_recur_k_term_func
                            }

                            if n_tuple[k] > 0 {
                                let mut prev_n_tuple_k = n_tuple.clone();
                                prev_n_tuple_k.iter_mut().enumerate().for_each(|(t, n)| {
                                    if t == k { *n -= 1 }
                                });
                                debug_assert!(!remaining_tuples.contains(&(l_tuple, prev_n_tuple_k)));
                                // 1 / (2 * zg) * sum(i) Nα(n_i) * [[n_i - 1_α:|:]]
                                (0..3).for_each(|i| {
                                    let add_term = self.zg.mapv(|zg| {
                                        dtype::from(1.0)
                                        / (2.0 * zg)
                                        * n_tuple[k]
                                            .to_f64()
                                            .unwrap_or_else(|| panic!("Unable to convert `n_tuple[k]` = {} to `f64`.", n_tuple[k]))
                                    }) * ints_r[i][l_tuple][prev_n_tuple_k].as_ref().unwrap_or_else(|| {
                                        panic!("({l_tuple:?}, {prev_n_tuple_k:?}) => ({l_tuple:?}, {next_n_tuple:?}) failed.")
                                    });
                                    if let Some(arr) = ints_r[i][l_tuple][next_n_tuple].as_mut() {
                                        Zip::from(arr).and(&add_term).for_each(|a, &t| *a += t);
                                    } else {
                                        ints_r[i][l_tuple][next_n_tuple] = Some(add_term);
                                    }
                                });
                            };
                        });

                        if l_tuple[r_index] > 0 {
                            let mut prev_l_tuple = l_tuple.clone();
                            prev_l_tuple.iter_mut().enumerate().for_each(|(t, l)| {
                                if t == r_index { *l -= 1 }
                            });
                            debug_assert!(!remaining_tuples.contains(&(prev_l_tuple, n_tuple)));
                            // -Nα(l_j) * [[:l_j - 1_α|:]]
                            // Note that Nα(l_j) = (l_j)_α.
                            (0..3).for_each(|i| {
                                let add_term = dtype::from(l_tuple[r_index]
                                    .to_f64()
                                    .unwrap_or_else(|| panic!("Unable to convert `l_tuple[r_index]` = {} to `f64`.", l_tuple[r_index]))
                                )
                                * ints_r[i][prev_l_tuple][n_tuple]
                                    .as_ref()
                                    .unwrap_or_else(|| {
                                        panic!("({prev_l_tuple:?}, {n_tuple:?}) => ({l_tuple:?}, {next_n_tuple:?}) failed.")
                                    });
                                if let Some(arr) = ints_r[i][l_tuple][next_n_tuple].as_mut() {
                                    Zip::from(arr).and(&add_term).for_each(|a, &t| *a -= t);
                                } else {
                                    ints_r[i][l_tuple][next_n_tuple] = Some(-add_term);
                                }
                            });
                        }

                        (0..$RANK).for_each(|k| {
                            if l_tuple[k] > 0 {
                                let mut prev_l_tuple_k = l_tuple.clone();
                                prev_l_tuple_k.iter_mut().enumerate().for_each(|(t, l)| {
                                    if t == k { *l -= 1 }
                                });
                                debug_assert!(!remaining_tuples.contains(&(prev_l_tuple_k, n_tuple)));
                                // (1 / zg) * sum(g) z_g * Nα(l_g) * [[:l_g - 1_α|:]]
                                (0..3).for_each(|i| {
                                    let add_term = dtype::from(
                                        l_tuple[k]
                                            .to_f64()
                                            .unwrap_or_else(|| panic!("Unable to convert `l_tuple[k]` = {} to `f64`.", l_tuple[k])))
                                    / zg_type
                                    * &self.zs[k] // broadcasting zs[k] to the shape of zg.
                                    * ints_r[i][prev_l_tuple_k][n_tuple].as_ref().unwrap_or_else(|| {
                                        panic!("({prev_l_tuple_k:?}, {n_tuple:?}) => ({l_tuple:?}, {next_n_tuple:?}) failed.")
                                    });
                                    if let Some(arr) = ints_r[i][l_tuple][next_n_tuple].as_mut() {
                                        Zip::from(arr).and(&add_term).for_each(|a, &t| *a += t);
                                    } else {
                                        ints_r[i][l_tuple][next_n_tuple] = Some(add_term);
                                    }
                                });
                            }
                        })
                    }
                    // ~~~~~~~~~~~~~~~~~~~~~~
                    // n-recurrent terms end.
                    // ~~~~~~~~~~~~~~~~~~~~~~

                    // ~~~~~~~~~~~~~~~~~~~~~~~~
                    // l-recurrent terms begin.
                    // ~~~~~~~~~~~~~~~~~~~~~~~~
                    if extra_tuples.contains(&(l_tuple, n_tuple)) {
                        continue
                    }
                    for r_index in 0..$RANK {
                        // r_index: recursion index (g in handwritten note)
                        let next_l_tuple = {
                            let mut new_l_tuple = l_tuple.clone();
                            new_l_tuple.iter_mut().enumerate().for_each(|(t, l)| {
                                if t == r_index { *l += 1 }
                            });
                            new_l_tuple
                        };
                        if !remaining_tuples.shift_remove(&(next_l_tuple, n_tuple)) {
                            continue
                        }

                        if let Some(l_recur_k_term_kr) = self.ks[r_index].as_ref() {
                            // // -ii * k_gα * [[:|:]]
                            // (0..3).for_each(|i| {
                            //     let add_term = C128::i()
                            //     * kr[i]
                            //     * ints_r[i][l_tuple][n_tuple].as_ref().unwrap_or_else(|| {
                            //         panic!("({l_tuple:?}, {n_tuple:?}) => ({next_l_tuple:?}, {n_tuple:?}) failed.")
                            //     });
                            //     if let Some(arr) = ints_r[i][next_l_tuple][n_tuple].as_mut() {
                            //         Zip::from(arr).and(&add_term).for_each(|a, &t| *a -= t);
                            //     } else {
                            //         ints_r[i][next_l_tuple][n_tuple] = Some(-add_term);
                            //     }
                            // });
                            l_recur_k_term_func
                        }

                        let next_n_tuple = {
                            let mut new_n_tuple = n_tuple.clone();
                            new_n_tuple.iter_mut().enumerate().for_each(|(t, n)| {
                                if t == r_index { *n += 1 }
                            });
                            new_n_tuple
                        };
                        debug_assert!(next_n_tuple.iter().enumerate().all(|(t, n)| *n <= ns[t]));
                        debug_assert!(!remaining_tuples.contains(&(l_tuple, next_n_tuple)));

                        // 2 * z_g * [[n_g + 1_α:|:]]
                        (0..3).for_each(|i| {
                            let add_term = dtype::from(2.0)
                            * ints_r[i][l_tuple][next_n_tuple].as_ref().unwrap_or_else(|| {
                                panic!("({l_tuple:?}, {next_n_tuple:?}) => ({next_l_tuple:?}, {n_tuple:?}) failed.")
                            })
                            * &self.zs[r_index]; // broadcasting zs[r_index] to the shape of
                                                 // ints_r[i][l_tuple][next_n_tuple].
                            if let Some(arr) = ints_r[i][next_l_tuple][n_tuple].as_mut() {
                                Zip::from(arr).and(&add_term).for_each(|a, &t| *a += t);
                            } else {
                                ints_r[i][next_l_tuple][n_tuple] = Some(add_term);
                            }
                        });

                        if n_tuple[r_index] > 0 {
                            let mut prev_n_tuple = n_tuple.clone();
                            prev_n_tuple.iter_mut().enumerate().for_each(|(t, n)| {
                                if t == r_index { *n -= 1 }
                            });
                            debug_assert!(!remaining_tuples.contains(&(l_tuple, prev_n_tuple)));

                            // -Nα(n_g) * [[n_g - 1_α:|:]]
                            (0..3).for_each(|i| {
                                let add_term = dtype::from(
                                    n_tuple[r_index]
                                        .to_f64()
                                        .unwrap_or_else(|| panic!("Unable to convert `n_tuple[r_index]` = {} to `f64`.", n_tuple[r_index]))
                                )
                                * ints_r[i][l_tuple][prev_n_tuple].as_ref().unwrap_or_else(|| {
                                    panic!("({l_tuple:?}, {prev_n_tuple:?}) => ({next_l_tuple:?}, {n_tuple:?}) failed.")
                                });
                                if let Some(arr) = ints_r[i][next_l_tuple][n_tuple].as_mut() {
                                    Zip::from(arr).and(&add_term).for_each(|a, &t| *a -= t);
                                } else {
                                    ints_r[i][next_l_tuple][n_tuple] = Some(-add_term);
                                }
                            });
                        }
                    }
                    // ~~~~~~~~~~~~~~~~~~~~~~
                    // l-recurrent terms end.
                    // ~~~~~~~~~~~~~~~~~~~~~~
                }
                // ~~~~~~~~~~~~~~~~~~~~~~~~~~
                // Loop over all tuples ends.
                // ~~~~~~~~~~~~~~~~~~~~~~~~~~

                // ~~~~~~~~~~~~~~~~~~~~~
                // Normalisation begins.
                // ~~~~~~~~~~~~~~~~~~~~~
                // Each element of `norm_arr` gives the scaling factor to be multiplied by an
                // integral involving a specific combination of primitives (i.e. in the integral,
                // each centre comprises a single Gaussian function) over one of the three
                // Cartesian directions, such that each primitive is normalised and that the
                // two-centre integrals〈g0|g0〉is precisely one.
                // The elements of `norm_arr` are only dependent on the exponents, not the
                // contraction coefficients, because `g0` is taken to have a coefficient of 1.
                // Contraction coefficients are introduced later.
                for n_tuple in n_tuples.iter() {
                    let rank_i32 = $RANK
                        .to_i32()
                        .expect("Unable to convert the tuple rank to `i32`.");
                    let norm_arr =
                        (2.0 / std::f64::consts::PI).sqrt().sqrt().powi(rank_i32)
                        * n_tuple.iter().map(|n| {
                            let doufac = if *n == 0 {
                                1
                            } else {
                                ((2 * n) - 1)
                                    .checked_double_factorial()
                                    .unwrap_or_else(|| panic!("Unable to obtain `{}!!`.", 2 * n - 1))
                            }
                            .to_f64()
                            .unwrap_or_else(|| panic!("Unable to convert `{}!!` to `f64`.", 2 * n - 1));
                            1.0 / doufac.sqrt()
                        }).product::<f64>()
                        * self.zd.map(|zd| zd.sqrt().sqrt());
                    let norm_arr = self
                        .zs
                        .iter()
                        .zip(n_tuple.iter())
                        .enumerate()
                        .fold(norm_arr, |acc, (j, (z, n))| {
                            let mut shape = [$(replace_expr!(($shell_name) 1)),+];
                            shape[j] = z.len();
                            let z_transformed = z.mapv(|z_val| {
                                if n.rem_euclid(2) == 0 {
                                    (4.0 * z_val).powi(
                                        (n.div_euclid(2))
                                            .to_i32()
                                            .expect("Unable to convert `n` to `i32`.")
                                    )
                                } else {
                                    (4.0 * z_val).powf(
                                        n.to_f64().expect("Unable to convert `n` to `f64`.")
                                        / 2.0
                                    )
                                }
                            })
                            .into_shape_with_order(shape)
                            .expect("Unable to convert transformed `z` to {$RANK} dimensions.");
                            acc * z_transformed
                        });

                    for l_tuple in l_tuples.iter() {
                        (0..3).for_each(|i| {
                            if let Some(arr) = ints_r[i][*l_tuple][*n_tuple].as_mut() {
                                Zip::from(arr)
                                    .and(&norm_arr)
                                    .for_each(|a, &n| *a *= dtype::from(n));
                            }
                        });
                    }
                }
                // ~~~~~~~~~~~~~~~~~~~
                // Normalisation ends.
                // ~~~~~~~~~~~~~~~~~~~

                // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                // Population of Cartesian integrals for each derivative component begins.
                // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                let lex_cart_orders = (0..=*ls.iter().max().expect("Unable to determine the maximum derivative order."))
                    .map(|l| CartOrder::lex(u32::try_from(l).expect("Unable to convert a derivative order to `u32`.")))
                    .collect::<Vec<_>>();
                let cart_shell_shape = {
                    let mut cart_shell_shape_iter = self
                        .ns
                        .iter()
                        .map(|n| ((n + 1) * (n + 2)).div_euclid(2));
                    $(
                        let $shell_name = cart_shell_shape_iter
                            .next()
                            .expect("cart_shell_shape out of range.");
                    )+
                    [$($shell_name),+]
                };
                let all_shells_contraction_str = (0..$RANK)
                    .map(|i| (i.to_u8().expect("Unable to convert a shell index to `u8`.") + 97) as char)
                    .collect::<String>();
                let shell_blocks = ls
                    .iter()
                    .map(|l| 0..((l + 1) * (l + 2)).div_euclid(2))
                    .multi_cartesian_product()
                    .map(|l_indices| {
                        // ls = [m, n, p, ...]
                        // l_indices = [a, b, c, ...]
                        //   - a-th component (lexicographic order) of the m-th derivative of the first shell,
                        //   - b-th component (lexicographic order) of the n-th derivative of the second shell,
                        //   - etc.
                        // The derivative components are arranged in lexicographic Cartersian order.
                        // If ls = [0, 1, 2], then the a particular l_indices could take the value
                        // [0, 2, 3] which represents
                        //   - 0th derivative of the first shell
                        //   - d/dz of the second shell (x, y, z)
                        //   - d2/dyy of the third shell (xx, xy, xz, yy, yz, zz)
                        debug_assert_eq!(l_indices.len(), $RANK);
                        let mut l_indices_iter = l_indices.into_iter();
                        $(
                            let $shell_name = l_indices_iter
                                .next()
                                .expect("l index out of range.");
                        )+
                        let l_indices = [$($shell_name),+];

                        // l_powers translates l_indices into tuples of component derivative orders
                        // for each shell.
                        // For example, with ls = [0, 1, 2] and l_indices = [0, 2, 3],
                        // l_powers is given by [(0, 0, 0), (0, 0, 1), (0, 2, 0)].
                        let l_powers = {
                            let mut l_powers_mut = [$(
                                replace_expr!(($shell_name) (0, 0, 0))
                            ),+];
                            l_powers_mut.iter_mut().enumerate().for_each(|(shell_index, l_power)| {
                                *l_power = lex_cart_orders[ls[shell_index]].cart_tuples[l_indices[shell_index]].clone();
                            });
                            l_powers_mut
                        };

                        // l_tuples_xyz gives l_tuple for each Cartesian component.
                        // With l_powers [(0, 0, 0), (0, 0, 1), (0, 2, 0)],
                        // l_tuples_xyz is given by
                        // [(0, 0, 0), (0, 0, 2), (0, 1, 0)]
                        //  ----x----  ----y----  ----z----
                        // which means: take the product of the (0, 0, 0) x-derivative,
                        // (0, 0, 2) y-derivative, and (0, 1, 0) z-derivative to give int_xyz.
                        // Essentially, l_tuples_xyz is transposed l_powers.
                        // l_tuples_xyz will be cloned inside the for loop below because it
                        // is consumed after every iteration.
                        let outer_l_tuples_xyz = {
                            let mut l_tuples_xyz_mut = [
                                [$(replace_expr!(($shell_name) 0usize)),+]; 3
                            ];
                            l_tuples_xyz_mut[0].iter_mut().enumerate().for_each(|(shell_index, l)| {
                                *l = usize::try_from(l_powers[shell_index].0)
                                    .expect("Unable to convert `l` to `usize`.");
                            });
                            l_tuples_xyz_mut[1].iter_mut().enumerate().for_each(|(shell_index, l)| {
                                *l = usize::try_from(l_powers[shell_index].1)
                                    .expect("Unable to convert `l` to `usize`.");
                            });
                            l_tuples_xyz_mut[2].iter_mut().enumerate().for_each(|(shell_index, l)| {
                                *l = usize::try_from(l_powers[shell_index].2)
                                    .expect("Unable to convert `l` to `usize`.");
                            });
                            l_tuples_xyz_mut
                        };

                        let mut cart_shell_block = Array::<dtype, Dim<[usize; $RANK]>>::zeros(
                            cart_shell_shape
                        );
                        for cart_indices in cart_shell_shape.iter().map(|d| 0..*d).multi_cartesian_product() {
                            // cart_indices = [i, j, k, l, ...]
                            //   - i-th Cartesian component (shell's specified order) of the first shell,
                            //   - j-th Cartesian component (shell's specified order) of the second shell,
                            //   - etc.
                            // If a shell has pure ordering, a lexicographic Cartesian order will
                            // be used. Integrals involving this shell will be converted back to
                            // pure form later.
                            // If shell_tuple.ns = [0, 2, 3, 1], then the a particular cart_indices could
                            // take the value [0, 2, 10, 1] which represents
                            //   - s function on the first shell
                            //   - dxz function on the second shell
                            //   - fzzz function on the third shell
                            //   - py function on the fourth shell
                            let mut cart_indices_iter = cart_indices.into_iter();
                            $(
                                let $shell_name = cart_indices_iter
                                    .next()
                                    .expect("cart_index out of range.");
                            )+
                            let cart_indices = [$($shell_name),+];

                            // cart_powers translates cart_indices into tuples of Cartesian powers
                            // for each shell.
                            // For example, with shell_tuple.ns = (0, 2, 3, 1) and
                            // cart_indices = (0, 2, 10, 1), cart_powers is given by
                            // [(0, 0, 0), (1, 0, 1), (0, 0, 3), (0, 1, 0)] (assuming
                            // lexicographic ordering).
                            let cart_powers = {
                                let mut cart_powers_mut = [$(
                                    replace_expr!(($shell_name) (0, 0, 0))
                                ),+];
                                cart_powers_mut.iter_mut().enumerate().for_each(|(shell_index, cart_power)| {
                                    let cart_order = match &self
                                        .shells[shell_index].0
                                        .basis_shell()
                                        .shell_order {
                                            ShellOrder::Pure(po) => CartOrder::lex(po.lpure),
                                            ShellOrder::Cart(co) => co.clone()
                                        };
                                    *cart_power = cart_order
                                        .cart_tuples[cart_indices[shell_index]]
                                        .clone();
                                });
                                cart_powers_mut
                            };

                            // n_tuples_xyz gives n_tuple for each Cartesian component.
                            // With cart_powers = [(0, 0, 0), (1, 0, 1), (0, 0, 3), (0, 1, 0)],
                            // n_tuples_xyz is given by
                            // [(0, 1, 0, 0), (0, 0, 0, 1), (0, 1, 3, 0)]
                            //  -----x------  -----y------  -----z------
                            // which means: take the product of the (0, 1, 0, 0) x-integral,
                            // (0, 0, 0, 1) y-integral, and (0, 1, 3, 0) z-integral to give int_xyz.
                            // Essentially, n_tuples_xyz is transposed cart_powers.
                            let l_tuples_xyz = outer_l_tuples_xyz.clone();
                            let n_tuples_xyz = {
                                let mut n_tuples_xyz_mut = [[$(replace_expr!(($shell_name) 0usize)),+]; 3];
                                n_tuples_xyz_mut[0].iter_mut().enumerate().for_each(|(shell_index, n)| {
                                    *n = usize::try_from(cart_powers[shell_index].0)
                                        .expect("Unable to convert `n` to `usize`.");
                                });
                                n_tuples_xyz_mut[1].iter_mut().enumerate().for_each(|(shell_index, n)| {
                                    *n = usize::try_from(cart_powers[shell_index].1)
                                        .expect("Unable to convert `n` to `usize`.");
                                });
                                n_tuples_xyz_mut[2].iter_mut().enumerate().for_each(|(shell_index, n)| {
                                    *n = usize::try_from(cart_powers[shell_index].2)
                                        .expect("Unable to convert `n` to `usize`.");
                                });
                                n_tuples_xyz_mut
                            };
                            let int_xyz = izip!(l_tuples_xyz.iter(), n_tuples_xyz.iter())
                                .enumerate()
                                .map(|(i, (l_tuple, n_tuple))| {
                                    ints_r[i][*l_tuple][*n_tuple].as_ref()
                                })
                                .collect::<Option<Vec<_>>>()
                                .map(|arrs| arrs.into_iter().fold(
                                    Array::<dtype, Dim<[usize; $RANK]>>::ones(
                                        self.primitive_shell_shape
                                    ),
                                    |acc, arr| acc * arr
                                ))
                                .unwrap_or_else(
                                    || Array::<dtype, Dim<[usize; $RANK]>>::zeros(
                                        self.primitive_shell_shape
                                    )
                                );

                            // Contraction coefficients are involved here.
                            cart_shell_block[cart_indices] = einsum(
                                &format!("{all_shells_contraction_str},{all_shells_contraction_str}->"),
                                &[&int_xyz, dd_type]
                            )
                                .expect("Unable to contract `int_xyz` with `dd`.")
                                .into_iter()
                                .next()
                                .expect("Unable to retrieve the result of the contraction between `int_xyz` and `dd`.");
                        }

                        // Transform some shells to spherical if necessary
                        if (0..$RANK).any(|i| matches!(self.shells[i].0.basis_shell().shell_order, ShellOrder::Pure(_))) {
                            // We need an extra letter for the contraction axis.
                            assert!($RANK < 26);
                            let rank_u8 = $RANK.to_u8().expect("Unable to convert the shell tuple rank to `u8`.");
                            let transformed_shell_block = (0..$RANK)
                                .fold(cart_shell_block, |acc, i| {
                                    if let ShellOrder::Pure(_) = self.shells[i].0.basis_shell().shell_order {
                                        let i_u8 = i.to_u8().expect("Unable to convert a shell index to `u8`.");
                                        let rl2cart = rl2cart_type;
                                        let cart_to_pure_contraction_str = format!(
                                            "{}{}",
                                            (i_u8 + 97) as char,
                                            (i_u8 + 97 + rank_u8) as char,
                                        );
                                        let result_str = (0..$RANK).map(|j| {
                                            if j == i {
                                                (i_u8 + 97 + rank_u8) as char
                                            } else {
                                                let j_u8 = j.to_u8().expect("Unable to convert a shell index to `u8`.");
                                                (j_u8 + 97) as char
                                            }
                                        }).collect::<String>();
                                        einsum(
                                            &format!(
                                                "{all_shells_contraction_str},\
                                                {cart_to_pure_contraction_str}->\
                                                {result_str}"
                                            ),
                                            &[&acc, rl2cart]
                                        )
                                        .unwrap_or_else(|_| panic!("Unable to convert shell {i} to spherical order."))
                                        .into_dimensionality::<Dim<[usize; $RANK]>>()
                                        .unwrap_or_else(|_| panic!("Unable to convert the transformed shell block into the correct shape."))
                                    } else {
                                        acc
                                    }
                                });
                            transformed_shell_block
                        } else {
                            cart_shell_block
                        }
                    }).collect::<Vec<_>>();
                // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                // Population of Cartesian integrals for each derivative component ends.
                // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

                shell_blocks
            }
        }

        #[duplicate_item(
            [
                dtype [ f64 ]
            ]
            [
                dtype [ C128 ]
            ]
        )]
        impl<'a> ShellTupleCollection<'a, $RANK, dtype> {
            /// Calculates the overlap arrays for this shell tuple collection.
            ///
            /// # Arguments
            ///
            /// * `ls` - The derivative pattern.
            ///
            /// # Returns
            ///
            /// A vector of overlap arrays, each of which is for one derivative component.
            pub(crate) fn overlap(
                &self, ls: [usize; $RANK]
            ) -> Vec<Array<dtype, Dim<[usize; $RANK]>>> {
                let lex_cart_orders = (0..=*ls.iter().max().expect("Unable to determine the maximum derivative order."))
                    .map(|l| CartOrder::lex(u32::try_from(l).expect("Unable to convert a derivative order to `u32`.")))
                    .collect::<Vec<_>>();

                // shell_blockss = [shell_blocks0, shell_blocks1, ...]
                //   shell_blocks0: all derivative components for shell tuple 0
                //   shell_blocks0 = [shell_block00, shell_block11, ...]
                //     shell_block01: first derivative component for shell tuple 0
                let shell_blockss = self
                    .unique_shell_tuples_iter(ls)
                    .par_bridge()
                    .map(|(shell_tuple, unique_perm, equiv_perms)| {
                        log::debug!("Working on unique permutation: {unique_perm:?}");
                        (shell_tuple.overlap(ls), unique_perm, equiv_perms)
                    })
                    .collect::<Vec<_>>();

                // Pack integrals
                let intss = ls
                    .iter()
                    .map(|l| 0..(((l + 1) * (l + 2)).div_euclid(2)))
                    .multi_cartesian_product()
                    .enumerate()
                    .map(|(l_component_index, l_indices)| {
                        let mut l_indices_iter = l_indices.into_iter();
                        $(
                            let $shell_name = l_indices_iter
                                .next()
                                .expect("l index out of range.");
                        )+
                        let l_indices = [$($shell_name),+];

                        // l_powers translates l_indices into tuples of component derivative orders
                        // for each shell.
                        // For example, with ls = [0, 1, 2] and l_indices = [0, 2, 3],
                        // l_powers is given by [(0, 0, 0), (0, 0, 1), (0, 2, 0)].
                        let l_powers = {
                            let mut l_powers_mut = [$(
                                replace_expr!(($shell_name) (0, 0, 0))
                            ),+];
                            l_powers_mut.iter_mut().enumerate().for_each(|(shell_index, l_power)| {
                                *l_power = lex_cart_orders[ls[shell_index]].cart_tuples[l_indices[shell_index]].clone();
                            });
                            l_powers_mut
                        };
                        log::debug!("Component {l_component_index} is for derivative {l_powers:?}.");
                        let mut ints = Array::<dtype, Dim<[usize; $RANK]>>::zeros(
                            self.angular_all_shell_shape
                        );

                        shell_blockss.iter().for_each(|(shell_blocks, unique_perm, equiv_perms)| {
                            equiv_perms.iter().for_each(|&equiv_perm| {
                                let mut unique_perm_vec = unique_perm
                                    .iter()
                                    .map(|i| Some(*i))
                                    .collect::<Vec<_>>();
                                let mut transpose_indices_iter = equiv_perm.iter().map(|i| {
                                    let index = unique_perm_vec
                                        .iter()
                                        .position(|&j| j == Some(*i))
                                        .unwrap_or_else(|| {
                                            panic!("Unable to find a permutation that maps {unique_perm:?} to {equiv_perm:?}.");
                                        });
                                    unique_perm_vec[index] = None;
                                    index
                                });
                                $(
                                    let $shell_name = transpose_indices_iter
                                        .next()
                                        .expect("Shell index out of range.");
                                )+
                                let transpose_indices = [$($shell_name),+];

                                let mut shell_boundaries_iter = equiv_perm
                                    .iter()
                                    .enumerate()
                                    .map(|(shell_index, &i)| {
                                        self.basis_sets[shell_index].shell_boundaries()[i]
                                    });
                                $(
                                    let $shell_name = shell_boundaries_iter
                                        .next()
                                        .expect("Shell index out of range.");
                                )+
                                let shell_slices = s![$(
                                    $shell_name.0..$shell_name.1
                                ),+];
                                ints.slice_mut(shell_slices).assign(
                                    &shell_blocks[l_component_index]
                                        .clone()
                                        .permuted_axes(transpose_indices)
                                );
                            })
                        });
                        ints
                    }).collect::<Vec<_>>();

                intss
            }
        }
    }
}

pub(crate) use impl_shell_tuple_overlap;
