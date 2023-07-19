use crate::integrals::count_exprs;

macro_rules! define_shell_tuple {
    ( $($shell:expr),+ ) => {
        use std::marker::PhantomData;

        use log;
        use itertools::Itertools;
        use nalgebra::{Point3, Vector3};
        use ndarray::{Array, Array1, Dim, Dimension};

        use crate::basis::ao_integrals::BasisShellContraction;

        const RANK: usize = count_exprs!($($shell),+);

        /// A structure to handle pre-computed properties of a tuple of shells consisting of
        /// non-integration primitives.
        #[derive(Clone, Debug)]
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
            /// This is a [`Self::rank`]-dimensional array. The element `sg[i, j, k, ...]`
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

            fn lmax(&self) -> u32 {
                self.shells
                    .iter()
                    .map(|(bsc, _)| bsc.basis_shell().l)
                    .max()
                    .expect("The maximum angular momentum across all shells cannot be found.")
            }
        }
    }
}
macro_rules! build_shell_tuple {
    ( $($shell:expr),+ ) => {
        {
            use std::marker::PhantomData;

            use log;
            use itertools::Itertools;
            use nalgebra::{Point3, Vector3};
            use ndarray::{Array, Array1, Dim, Dimension};

            use crate::basis::ao_integrals::BasisShellContraction;

            const RANK: usize = count_exprs!($($shell),+);

            /// A structure to handle pre-computed properties of a tuple of shells consisting of
            /// non-integration primitives.
            #[derive(Clone, Debug)]
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
                /// This is a [`Self::rank`]-dimensional array. The element `sg[i, j, k, ...]`
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

                fn lmax(&self) -> u32 {
                    self.shells
                        .iter()
                        .map(|(bsc, _)| bsc.basis_shell().l)
                        .max()
                        .expect("The maximum angular momentum across all shells cannot be found.")
                }
            }

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

                k: [$($shell.0.k()),+]
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
    ( $($shells:expr),+ ) => {
        {
            use std::marker::PhantomData;

            use log;
            use itertools::{iproduct, Itertools};
            use ndarray::{Array, Dim, Dimension};

            use crate::integrals::shell_tuple::ShellTuple;

            const RANK: usize = count_exprs!($($shells),+);

            /// A structure to handle pre-computed properties of a tuple of shells consisting of
            /// non-integration primitives.
            #[derive(Clone, Debug)]
            struct ShellTupleCollection<'a, D: Dimension> {
                dim: PhantomData<D>,

                shell_tuples: Array<ShellTuple<'a, D>, Dim<[usize; RANK]>>,
            }

            ShellTupleCollection::<Dim<[usize; RANK]>> {
                dim: PhantomData,

                shell_tuples: {
                    let arr_vec = iproduct!($($shells.iter()),+)
                        .map(|shell_tuple| build_shell_tuple!(shell_tuple))
                        .collect::<Vec<_>>();
                    let arr = Array::<ShellTuple<'a, D>, Dim<[usize; RANK]>>::from_shape_vec(
                        ($($shells.len()),+), arr_vec
                    ).unwrap_or_else(|err| {
                        log::error!("{err}");
                        panic!("Unable to construct the {RANK}-dimensional array of coefficient products.")
                    });
                    arr
                    // [$(0..$shells.len()),+]
                    //     .into_iter()
                    //     .multi_cartesian_product()
                },
            }
        }
    }
}

#[test]
fn test_integrals_shell_tuple() {
    use crate::basis::ao::*;
    use crate::basis::ao_integrals::*;
    use nalgebra::{Point3, Vector3};

    let bs0 = BasisShell::new(1, ShellOrder::Cart(CartOrder::lex(1)));
    let gc0 = GaussianContraction::<f64, f64> {
        primitives: vec![(0.1, 0.1), (0.2, 0.2)],
    };
    let bsc0 = BasisShellContraction::<f64, f64> {
        basis_shell: bs0,
        start_index: 0,
        contraction: gc0,
        cart_origin: Point3::new(1.0, 0.0, 0.0),
        k: None,
    };
    let bs1 = BasisShell::new(2, ShellOrder::Cart(CartOrder::lex(2)));
    let gc1 = GaussianContraction::<f64, f64> {
        primitives: vec![(0.3, 0.3), (0.4, 0.4), (0.5, 0.5)],
    };
    let bsc1 = BasisShellContraction::<f64, f64> {
        basis_shell: bs1,
        start_index: 3,
        contraction: gc1,
        cart_origin: Point3::new(2.0, 1.0, 1.0),
        k: Some(Vector3::z()),
    };
    let st = build_shell_tuple![(&bsc0, true), (&bsc1, false), (&bsc1, true)];
    println!("{:#?}", st.shell_boundaries);
    println!("{:#?}", st.ks);

    // let st2 = build_shell_tuple![(&bsc, true), (&bsc, false), (&bsc, false), (&bsc, true)];
    // println!("{}", st2.rank());
}

#[test]
fn test_integrals_shell_tuple_collection() {
    use crate::basis::ao::*;
    use crate::basis::ao_integrals::*;
    use nalgebra::{Point3, Vector3};

    let bs0 = BasisShell::new(1, ShellOrder::Cart(CartOrder::lex(1)));
    let gc0 = GaussianContraction::<f64, f64> {
        primitives: vec![(0.1, 0.1), (0.2, 0.2)],
    };
    let bsc0 = BasisShellContraction::<f64, f64> {
        basis_shell: bs0,
        start_index: 0,
        contraction: gc0,
        cart_origin: Point3::new(1.0, 0.0, 0.0),
        k: None,
    };
    let bs1 = BasisShell::new(2, ShellOrder::Cart(CartOrder::lex(2)));
    let gc1 = GaussianContraction::<f64, f64> {
        primitives: vec![(0.3, 0.3), (0.4, 0.4), (0.5, 0.5)],
    };
    let bsc1 = BasisShellContraction::<f64, f64> {
        basis_shell: bs1,
        start_index: 3,
        contraction: gc1,
        cart_origin: Point3::new(2.0, 1.0, 1.0),
        k: Some(Vector3::z()),
    };
    let bscs = [bsc0, bsc1];
    let stc = build_shell_tuple_collection![&bscs, &bscs];
    println!("{:#?}", stc.shell_tuples);

    // let st2 = build_shell_tuple![(&bsc, true), (&bsc, false), (&bsc, false), (&bsc, true)];
    // println!("{}", st2.rank());
}
