macro_rules! count_exprs {
    () => (0);
    ($head:expr) => (1);
    ($head:expr, $($tail:expr),*) => (1 + count_exprs!($($tail),*));
}

macro_rules! build_shell_tuple {
    ( $($shell:expr),+ ) => {
        {
            use log;
            use itertools::Itertools;
            use nalgebra::{Point3, Vector3};
            use ndarray::{Array, Array1, Dim};

            use crate::basis::ao_integrals::BasisShellContraction;

            const RANK: usize = count_exprs!($($shell),+);

            /// A structure to handle pre-computed properties of a tuple of shells.
            #[derive(Clone, Debug)]
            struct ShellTuple<'a> {
                /// The shells in this shell tuple. Each shell has an associated boolean indicating if it is to
                /// be complex-conjugated in the integral evalulation.
                shells: [(&'a BasisShellContraction<f64, f64>, bool); RANK],

                // -----------------------------------------------
                // Quantities common to all primitive combinations
                // -----------------------------------------------
                ks: [Option<&'a Vector3<f64>>; RANK],

                k: Vector3<f64>,

                rs: [&'a Point3<f64>; RANK],

                ns: [usize; RANK],

                /// An array indicating the shape of this shell tuple.
                ///
                /// Each element in the array gives the length of the corresponding shell.
                shell_shape: [usize; RANK],

                //shell_slices: [(usize, usize); RANK],

                // ------------------------------------------------
                // Quantities unique for each primitive combination
                // ------------------------------------------------
                /// A vector of arrays of primitive exponents.
                ///
                /// The i-th array in the vector is for the i-th shell. The j-th element in that array then
                /// gives the exponent of the j-th primitive exponent of that shell.
                ss: [Array1<&'a f64>; RANK],

                /// An array containing the sum of all possible combinations of primitive exponents across all
                /// shells.
                ///
                /// This is a [`Self::rank`]-dimensional array. The element sg[i, j, k, ...]
                /// gives the sum of the i-th primitive exponent on the first shell, the
                /// j-th primitive exponent on the second shell, the k-th primitive
                /// exponent on the third shell, and so on.
                sg: Array<f64, Dim<[usize; RANK]>>,
            }

            impl<'a> ShellTuple<'a> {
                fn shells(&self) -> &[(&'a BasisShellContraction<f64, f64>, bool); RANK] {
                    &self.shells
                }

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

                fn ks(&self) -> &[Option<&Vector3<f64>>; RANK] {
                    &self.ks
                }

                fn k(&self) -> &Vector3<f64> {
                    &self.k
                }
            }

            ShellTuple {
                shells: [$($shell),+],
                rs: [$($shell.0.cart_origin()),+],
                ks: [$($shell.0.k()),+],
                k: [$($shell.0.k()),+]
                    .into_iter()
                    .filter_map(|k| k)
                    .fold(Vector3::zeros(), |acc, k| acc + k),
                ns: [$(
                        usize::try_from($shell.0.basis_shell().l)
                            .expect("Unable to convert an angular momentum `l` value to `usize`.")
                    ),+],
                shell_shape: [$($shell.0.basis_shell().n_funcs()),+],
                ss: [$(Array1::from_iter($shell.0.contraction.primitives.iter().map(|(e, _)| e))),+],
                sg: {
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
                },
            }
        }
    }
}

#[test]
fn test_macro() {
    use crate::basis::ao::*;
    use crate::basis::ao_integrals::*;
    use nalgebra::Point3;

    let bs0 = BasisShell::new(1, ShellOrder::Cart(CartOrder::lex(1)));
    let gc0 = GaussianContraction::<f64, f64> {
        primitives: vec![(0.1, 0.1), (0.2, 0.2)],
    };
    let bsc0 = BasisShellContraction::<f64, f64> {
        basis_shell: bs0,
        contraction: gc0,
        cart_origin: Point3::<f64>::origin(),
        k: None,
    };
    let bs1 = BasisShell::new(2, ShellOrder::Cart(CartOrder::lex(2)));
    let gc1 = GaussianContraction::<f64, f64> {
        primitives: vec![(0.3, 0.3), (0.4, 0.4), (0.5, 0.5)],
    };
    let bsc1 = BasisShellContraction::<f64, f64> {
        basis_shell: bs1,
        contraction: gc1,
        cart_origin: Point3::<f64>::origin(),
        k: None,
    };
    let st = build_shell_tuple![(&bsc0, true), (&bsc1, false)];
    println!("{}", st.sg);

    // let st2 = build_shell_tuple![(&bsc, true), (&bsc, false), (&bsc, false), (&bsc, true)];
    // println!("{}", st2.rank());
}
