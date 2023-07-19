#[cfg(test)]
#[path = "shell_tuple_tests.rs"]
mod shell_tuple_tests;

macro_rules! define_shell_tuple {
    ( <$($shell_name:ident),+> ) => {
        use std::marker::PhantomData;
        use std::collections::{HashSet, HashMap};

        use log;
        use indexmap::IndexSet;
        use itertools::{izip, Itertools};
        use nalgebra::{Point3, Vector3};
        use ndarray::{Array, Array1, Dim, Dimension};

        use crate::basis::ao_integrals::BasisShellContraction;
        use crate::integrals::{count_exprs, replace_expr};

        const RANK: usize = count_exprs!($($shell_name),+);

        /// A structure to handle pre-computed properties of a tuple of shells consisting of
        /// non-integration primitives.
        #[derive(Debug)]
        struct ShellTuple<'a, D: Dimension> {
            dim: PhantomData<D>,

            /// The non-integration shells in this shell tuple. Each shell has an associated
            /// boolean indicating if it is to be complex-conjugated in the integral
            /// evalulation.
            shells: [(&'a BasisShellContraction<f64, f64>, bool); RANK],

            /// A fixed-size array indicating the shape of this shell tuple.
            ///
            /// Each element in the array gives the length of the corresponding shell.
            shell_shape: [usize; RANK],

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

        impl<'a, D: Dimension> ShellTuple<'a, D> {
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

        /// A structure to handle all possible shell tuples for a particular type of integral.
        #[derive(Debug)]
        struct ShellTupleCollection<'a, D: Dimension> {
            dim: PhantomData<D>,

            shell_tuples: Array<ShellTuple<'a, D>, Dim<[usize; RANK]>>,

            lmax: u32,

            ccs: [bool; RANK],

            n_shells: [usize; RANK],
        }

        impl<'a, D: Dimension> ShellTupleCollection<'a, D> {
            /// The maximum angular momentum across all shell tuples.
            fn lmax(&self) -> u32 {
                self.lmax
            }

            /// The number of shells in each tuple in the collection.
            fn rank(&self) -> usize {
                self.shell_tuples.ndim()
            }

            /// Returns a vector of the shell tuples unique with respect to permutations of the
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
            fn get_unique_shell_tuples(&self, ls: [usize; RANK]) {
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
                //     (true , 1, 2): (0,),
                //     (true , 1, 1): (1,),
                //     (false, 0, 2): (2, 4),
                //     (true , 2, 2): (3,)
                // }.
                let mut shell_types_classified: HashMap<(bool, usize, usize), IndexSet<usize>>
                    = HashMap::new();
                shell_types.into_iter().enumerate().for_each(|(shell_index, shell_type)| {
                    shell_types_classified.entry(shell_type).or_default().insert(shell_index);
                });

                // The map `masks` is essentially equivalent to `shell_types_classified`.
                // Example:
                // masks = {
                //     (true , 1, 2): [true , false, false, false, false],
                //     (true , 1, 1): [false, true , false, false, false],
                //     (false, 0, 2): [false, false, true , false, true ],
                //     (true , 2, 2): [false, false, false, true , false]
                // }.
                let default_mask = [$(replace_expr!(($shell_name) false)),+];
                let masks = shell_types_classified.iter().map(|(shell_type, shell_indices)| {
                    let mut mask = default_mask.clone();
                    (0..RANK).for_each(|i| mask[i] = shell_indices.contains(&i));
                    (shell_type, mask)
                }).collect::<HashMap<_, _>>();

                // The map `shell_type_unique_combinations` stores the unique index combinations
                // for each shell type.
                // Example:
                // shell_type_unique_combinations = {
                //   (true , 1, 2): [(0,), (1,)],
                //   (true , 1, 1): [(0,)],
                //   (false, 0, 2): [(0, 0), (0, 1), (1, 1)],
                //   (true , 2, 2): [(0,), (1,)],
                // }
                let shell_type_unique_combinations = shell_types_classified
                    .iter()
                    .map(|(shell_type, shell_indices)| {
                        (
                            shell_type,
                            (0..shell_type.2)
                                .combinations_with_replacement(shell_indices.len())
                                .collect::<Vec<_>>()
                        )
                    }).collect::<HashMap<_, _>>();

                println!("{:?}", shell_type_unique_combinations);
            }
        }

        // struct UniqueShellTupleIterator<'a, D: Dimension> {
        //     dim: PhantomData<D>,
        // }
    }
}

macro_rules! build_shell_tuple {
    ( $($shell:expr),+ ) => {
        {
            use itertools::Itertools;

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

            ShellTuple::<Dim<[usize; RANK]>> {
                dim: PhantomData,

                shells: [$($shell),+],

                shell_shape: [$($shell.0.basis_shell().n_funcs()),+],

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
    ( <$($shell_name:ident),+>; $($shell_cc:expr),+; $($shells:expr),+ ) => {
        {
            use itertools::iproduct;

            define_shell_tuple![<$($shell_name),+>];

            let shell_tuples = {
                let arr_vec = iproduct!($($shells.iter()),+)
                    .map(|shell_tuple| {
                        let ($($shell_name),+) = shell_tuple;
                        build_shell_tuple!($((*$shell_name, $shell_cc)),+)
                    })
                    .collect::<Vec<_>>();
                let arr = Array::<ShellTuple<Dim<[usize; RANK]>>, Dim<[usize; RANK]>>::from_shape_vec(
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
            ShellTupleCollection::<Dim<[usize; RANK]>> {
                dim: PhantomData,
                shell_tuples,
                lmax,
                ccs: [$($shell_cc),+],
                n_shells: [$($shells.len()),+]
            }
        }
    }
}

use {build_shell_tuple, build_shell_tuple_collection, define_shell_tuple};
