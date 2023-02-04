use std::collections::HashSet;
use std::fmt;
use std::hash::Hash;
use std::ops::Mul;

use approx;
use log;
use ndarray::{array, s, Array1, Array2, Zip};
use num::{integer::lcm, Complex};
use num_modular::{ModularInteger, MontgomeryInt};
use num_ord::NumOrd;
use num_traits::{Inv, Pow, ToPrimitive, Zero};
use primes::is_prime;
use rayon::prelude::*;

use crate::chartab::character::Character;
use crate::chartab::chartab_symbols::{
    CollectionSymbol, LinearSpaceSymbol, ReducibleLinearSpaceSymbol,
};
use crate::chartab::modular_linalg::{modular_eig, split_space, weighted_hermitian_inprod};
use crate::chartab::reducedint::{IntoLinAlgReducedInt, LinAlgMontgomeryInt};
use crate::chartab::unityroot::UnityRoot;
use crate::chartab::{CharacterTable, CorepCharacterTable, RepCharacterTable};
use crate::group::class::ClassProperties;
use crate::group::{GroupProperties, MagneticRepresentedGroup, UnitaryRepresentedGroup};
use crate::symmetry::symmetry_element::symmetry_operation::FiniteOrder;
// use crate::symmetry::symmetry_symbols::{
//     deduce_mulliken_irrep_symbols, deduce_principal_classes, sort_irreps, ClassSymbol,
//     MathematicalSymbol, MullikenIrcorepSymbol, MullikenIrrepSymbol, FORCED_PRINCIPAL_GROUPS,
// };

// #[cfg(test)]
// #[path = "chartab_construction_tests.rs"]
// mod chartab_construction_tests;

pub trait CharacterProperties: ClassProperties
where
    Self::RowSymbol: LinearSpaceSymbol,
    Self::CharTab: CharacterTable<RowSymbol = Self::RowSymbol, ColSymbol = Self::ClassSymbol>,
{
    /// Type of the character table of this group.
    type RowSymbol;
    type CharTab;

    /// Constructs the character table for this group.
    fn construct_character_table(&mut self);

    /// Returns a shared reference to the character table of this group.
    fn character_table(&self) -> &Self::CharTab;
}

impl<T, RowSymbol, ColSymbol> CharacterProperties
    for UnitaryRepresentedGroup<T, RowSymbol, ColSymbol>
where
    RowSymbol: LinearSpaceSymbol + Sync,
    ColSymbol: CollectionSymbol<CollectionElement = T> + Sync,
    T: Mul<Output = T>
        + Hash
        + Eq
        + Clone
        + Sync
        + fmt::Debug
        + FiniteOrder<Int = u32>
        + Pow<i32, Output = T>,
    for<'a, 'b> &'b T: Mul<&'a T, Output = T>,
{
    type RowSymbol = RowSymbol;
    type CharTab = RepCharacterTable<RowSymbol, ColSymbol>;

    fn character_table(&self) -> &Self::CharTab {
        self.irrep_character_table
            .as_ref()
            .expect("Irrep character table not found for this group.")
    }

    /// Constructs the irrep character table for this group using the Burnside--Dixon algorithm.
    ///
    /// # References
    ///
    /// * J. D. Dixon, Numer. Math., 1967, 10, 446–450.
    /// * L. C. Grove, Groups and Characters, John Wiley & Sons, Inc., 1997.
    ///
    /// # Panics
    ///
    /// Panics if the Frobenius--Schur indicator takes on unexpected values.
    #[allow(clippy::too_many_lines)]
    fn construct_character_table(&mut self) {
        // Variable definitions
        // --------------------
        // m: LCM of the orders of the elements in the group (i.e. the group
        //    exponent)
        // p: A prime greater than 2*sqrt(|G|) and m | (p - 1), which is
        //    guaranteed to exist by Dirichlet's theorem. p is guaranteed to be
        //    odd.
        // z: An integer having multiplicative order m when viewed as an
        //    element of Z*p, i.e. z^m ≡ 1 (mod p) but z^n !≡ 1 (mod p) for all
        //    0 <= n < m.

        log::debug!("=============================================");
        log::debug!("Construction of irrep character table begins.");
        log::debug!("     *** Burnside -- Dixon algorithm ***     ");
        log::debug!("=============================================");

        // Identify a suitable finite field
        let m = self
            .elements()
            .keys()
            .map(FiniteOrder::order)
            .reduce(lcm)
            .expect("Unable to find the LCM for the orders of the elements in this group.");
        let zeta = UnityRoot::new(1, m);
        log::debug!("Found group exponent m = {}.", m);
        log::debug!("Chosen primitive unity root ζ = {}.", zeta);

        let rf64 = (2.0
            * self
                .order()
                .to_f64()
                .unwrap_or_else(|| panic!("Unable to convert `{}` to `f64`.", self.order()))
                .sqrt()
            / (f64::from(m)))
        .round();
        assert!(rf64.is_sign_positive());
        assert!(rf64 <= f64::from(u32::MAX));
        #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
        let mut r = rf64 as u32;
        if r == 0 {
            r = 1;
        };
        let mut p = r * m + 1;
        while !is_prime(u64::from(p)) {
            log::debug!("Trying {}: not prime.", p);
            r += 1;
            p = r * m + 1;
        }
        log::debug!("Found r = {}.", r);
        log::debug!("Found prime number p = r * m + 1 = {}.", p);
        log::debug!("All arithmetic will now be carried out in GF({}).", p);

        let modp = MontgomeryInt::<u32>::new(1, &p).linalg();
        // p is prime, so there is guaranteed a z < p such that z^m ≡ 1 (mod p).
        let mut i = 1u32;
        while modp.convert(i).multiplicative_order().unwrap_or_else(|| {
            panic!(
                "Unable to find multiplicative order for `{}`",
                modp.convert(i)
            )
        }) != m
            && i < p
        {
            i += 1;
        }
        let z = modp.convert(i);
        assert_eq!(
            z.multiplicative_order()
                .unwrap_or_else(|| panic!("Unable to find multiplicative order for `{z}`.")),
            m
        );
        log::debug!("Found integer z = {} with multiplicative order {}.", z, m);

        // Diagonalise class matrices
        let class_sizes: Vec<_> = self.conjugacy_classes().iter().map(HashSet::len).collect();
        let inverse_conjugacy_classes = Some(self.inverse_conjugacy_classes());
        let mut eigvecs_1d: Vec<Array1<LinAlgMontgomeryInt<u32>>> = vec![];

        if self.class_number() == 1 {
            eigvecs_1d.push(array![modp.convert(1)]);
        } else {
            let mut degenerate_subspaces: Vec<Vec<Array1<LinAlgMontgomeryInt<u32>>>> = vec![];
            let nmat = self.class_matrix().map(|&i| {
                modp.convert(
                    u32::try_from(i)
                        .unwrap_or_else(|_| panic!("Unable to convert `{i}` to `u32`.")),
                )
            });
            log::debug!("Considering class matrix N1...");
            let nmat_1 = nmat.slice(s![1, .., ..]).to_owned();
            let eigs_1 = modular_eig(&nmat_1);
            eigvecs_1d.extend(eigs_1.iter().filter_map(|(_, eigvecs)| {
                if eigvecs.len() == 1 {
                    Some(eigvecs[0].clone())
                } else {
                    None
                }
            }));
            degenerate_subspaces.extend(
                eigs_1
                    .iter()
                    .filter_map(|(_, eigvecs)| {
                        if eigvecs.len() > 1 {
                            Some(eigvecs)
                        } else {
                            None
                        }
                    })
                    .cloned(),
            );

            let mut r = 1;
            while !degenerate_subspaces.is_empty() {
                assert!(
                    r < (self.class_number() - 1),
                    "Class matrices exhausted before degenerate subspaces are fully resolved."
                );

                r += 1;
                log::debug!(
                    "Number of 1-D eigenvectors found: {} / {}.",
                    eigvecs_1d.len(),
                    self.class_number()
                );
                log::debug!(
                    "Number of degenerate subspaces found: {}.",
                    degenerate_subspaces.len(),
                );

                log::debug!("Considering class matrix N{}...", r);
                let nmat_r = nmat.slice(s![r, .., ..]).to_owned();

                let mut remaining_degenerate_subspaces: Vec<Vec<Array1<LinAlgMontgomeryInt<u32>>>> =
                    vec![];
                while !degenerate_subspaces.is_empty() {
                    let subspace = degenerate_subspaces
                        .pop()
                        .expect("Unexpected empty `degenerate_subspaces`.");
                    if let Ok(subsubspaces) =
                        split_space(&nmat_r, &subspace, &class_sizes, inverse_conjugacy_classes)
                    {
                        eigvecs_1d.extend(subsubspaces.iter().filter_map(|subsubspace| {
                            if subsubspace.len() == 1 {
                                Some(subsubspace[0].clone())
                            } else {
                                None
                            }
                        }));
                        remaining_degenerate_subspaces.extend(
                            subsubspaces
                                .iter()
                                .filter(|subsubspace| subsubspace.len() > 1)
                                .cloned(),
                        );
                    } else {
                        log::debug!(
                            "Class matrix N{} failed to split degenerate subspace {}.",
                            r,
                            degenerate_subspaces.len()
                        );
                        log::debug!("Stashing this subspace for the next class matrices...");
                        remaining_degenerate_subspaces.push(subspace);
                    }
                }
                degenerate_subspaces = remaining_degenerate_subspaces;
            }
        }

        assert_eq!(eigvecs_1d.len(), self.class_number());
        log::debug!(
            "Successfully found {} / {} one-dimensional eigenvectors for the class matrices.",
            eigvecs_1d.len(),
            self.class_number()
        );
        for (i, vec) in eigvecs_1d.iter().enumerate() {
            log::debug!("Eigenvector {}: {}", i, vec);
        }

        // Lift characters back to the complex field
        log::debug!(
            "Lifting characters from GF({}) back to the complex field...",
            p
        );
        let class_transversal = self.conjugacy_class_transversal();

        let chars: Vec<_> = eigvecs_1d
            .par_iter()
            .flat_map(|vec_i| {
                let mut dim2_mod_p = weighted_hermitian_inprod(
                    (vec_i, vec_i),
                    &class_sizes,
                    inverse_conjugacy_classes,
                )
                .inv()
                .residue();
                while !approx::relative_eq!(
                    f64::from(dim2_mod_p).sqrt().round(),
                    f64::from(dim2_mod_p).sqrt()
                ) {
                    dim2_mod_p += p;
                }

                let dim_if64 = f64::from(dim2_mod_p).sqrt().round();
                assert!(dim_if64.is_sign_positive());
                assert!(dim_if64 <= f64::from(u32::MAX));
                #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
                let dim_i = dim_if64 as u32;

                let tchar_i =
                    Zip::from(vec_i)
                        .and(class_sizes.as_slice())
                        .par_map_collect(|&v, &k| {
                            v * dim_i
                                / modp.convert(u32::try_from(k).unwrap_or_else(|_| {
                                    panic!("Unable to convert `{k}` to `u32`.")
                                }))
                        });
                let char_i: Vec<_> = class_transversal
                    .par_iter()
                    .map(|x_idx| {
                        let x = self
                            .elements()
                            .get_index(*x_idx)
                            .unwrap_or_else(|| {
                                panic!("Element with index {x_idx} cannot be retrieved.")
                            })
                            .0;
                        let k = x.order();
                        let xi = zeta.clone().pow(
                            i32::try_from(m.checked_div_euclid(k).unwrap_or_else(|| {
                                panic!("`{m}` cannot be Euclid-divided by `{k}`.")
                            }))
                            .unwrap_or_else(|_| {
                                panic!(
                                    "The Euclid division `{m} / {k}` cannot be converted to `i32`."
                                )
                            }),
                        );
                        let char_ij_terms: Vec<_> = (0..k)
                            .into_par_iter()
                            .map(|s| {
                                let mu_s = (0..k).fold(modp.convert(0), |acc, l| {
                                    let x_l =
                                        x.clone().pow(i32::try_from(l).unwrap_or_else(|_| {
                                            panic!("Unable to convert `{l}` to `i32`.")
                                        }));
                                    let x_l_idx = *self
                                        .elements()
                                        .get(&x_l)
                                        .unwrap_or_else(|| panic!("Element {x_l:?} not found."));
                                    let x_l_class_idx =
                                        self.element_to_conjugacy_classes()[x_l_idx].unwrap_or_else(|| {
                                            panic!("Element `{x_l_idx}` does not have a conjugacy class.")
                                        });
                                    let tchar_i_x_l = tchar_i[x_l_class_idx];
                                    acc + tchar_i_x_l
                                        * z.pow(
                                            s * l
                                                * m.checked_div_euclid(k).unwrap_or_else(|| {
                                                    panic!(
                                                        "`{m}` cannot be Euclid-divided by `{k}`."
                                                    )
                                                }),
                                        )
                                        .inv()
                                }) / k;
                                (
                                    xi.pow(i32::try_from(s).unwrap_or_else(|_| {
                                        panic!("Unable to convert `{s}` to `i32`.")
                                    })),
                                    usize::try_from(mu_s.residue()).unwrap_or_else(|_| {
                                        panic!("Unable to convert `{}` to `usize`.", mu_s.residue())
                                    }),
                                )
                            })
                            .collect();
                        // We do not wish to simplify the character here, even if it can be
                        // simplified (e.g. (E8)^2 + (E8)^6 can be simplified to 0). This is so
                        // that the full unity-root-term-structure can be used in the ordering of
                        // irreps.
                        Character::new(&char_ij_terms)
                    })
                    .collect();
                char_i
            })
            .collect();

        let char_arr = Array2::from_shape_vec((self.class_number(), self.class_number()), chars)
            .expect("Unable to construct the two-dimensional table of characters.");
        log::debug!(
            "Lifting characters from GF({}) back to the complex field... Done.",
            p
        );

        let class_symbols = self.conjugacy_class_symbols();

        let default_irrep_symbols = char_arr
            .rows()
            .into_iter()
            .enumerate()
            .map(|(irrep_i, _)| {
                RowSymbol::from_str(&format!("||Λ|_({})|", irrep_i))
                    .ok()
                    .expect("Unable to construct default irrep symbols.")
            })
            .collect::<Vec<_>>();
        let default_principal_classes = vec![self
            .conjugacy_class_symbols()
            .first()
            .expect("No conjugacy class symbols found.")
            .0
            .clone()];
        let frobenius_schur_indicators: Vec<_> = char_arr
            .rows()
            .into_iter()
            .enumerate()
            .map(|(irrep_i, _)| {
                let indicator: Complex<f64> =
                    self.elements()
                        .keys()
                        .fold(Complex::new(0.0f64, 0.0f64), |acc, ele| {
                            let ele_2_idx =
                                self.elements().get(&ele.clone().pow(2)).unwrap_or_else(|| {
                                    panic!("Element {:?} not found.", &ele.clone().pow(2))
                                });
                            let class_2_j = self.element_to_conjugacy_classes()[*ele_2_idx]
                                .unwrap_or_else(|| {
                                    panic!("Element `{ele:?}` does not have a conjugacy class.")
                                });
                            acc + char_arr[[irrep_i, class_2_j]].complex_value()
                        })
                        / self.order().to_f64().unwrap_or_else(|| {
                            panic!("Unable to convert `{}` to `f64`.", self.order())
                        });
                approx::assert_relative_eq!(
                    indicator.im,
                    0.0,
                    epsilon = 1e-14,
                    max_relative = 1e-14
                );
                approx::assert_relative_eq!(
                    indicator.re,
                    indicator.re.round(),
                    epsilon = 1e-14,
                    max_relative = 1e-14
                );
                assert!(
                    approx::relative_eq!(indicator.re, 1.0, epsilon = 1e-14, max_relative = 1e-14)
                        || approx::relative_eq!(
                            indicator.re,
                            0.0,
                            epsilon = 1e-14,
                            max_relative = 1e-14
                        )
                        || approx::relative_eq!(
                            indicator.re,
                            -1.0,
                            epsilon = 1e-14,
                            max_relative = 1e-14
                        )
                );
                #[allow(clippy::cast_possible_truncation)]
                let indicator_i8 = indicator.re.round() as i8;
                indicator_i8
            })
            .collect();

        let chartab_name = if let Some(finite_name) = self.finite_subgroup_name().as_ref() {
            format!("{} > {finite_name}", self.name())
        } else {
            self.name().to_string()
        };
        self.irrep_character_table = Some(RepCharacterTable::new(
            chartab_name.as_str(),
            &default_irrep_symbols,
            &class_symbols.keys().cloned().collect::<Vec<_>>(),
            &default_principal_classes,
            char_arr,
            &frobenius_schur_indicators,
        ));

        log::debug!("===========================================");
        log::debug!("    *** Burnside -- Dixon algorithm ***    ");
        log::debug!("Construction of irrep character table ends.");
        log::debug!("===========================================");
    }
}

// impl<T, UG> CharacterProperties<MullikenIrcorepSymbol, ClassSymbol<T>>
//     for MagneticRepresentedGroup<T, UG, UG::CharTab>
// where
//     T: Mul<Output = T>
//         + Hash
//         + Eq
//         + Clone
//         + Sync
//         + fmt::Debug
//         + FiniteOrder<Int = u32>
//         + Pow<i32, Output = T>,
//     for<'a, 'b> &'b T: Mul<&'a T, Output = T>,
//     UG: Clone
//         + GroupProperties<GroupElement = T>
//         + ClassProperties<GroupElement = T>
//         + CharacterProperties<MullikenIrrepSymbol, ClassSymbol<T>>,
//     UG::CharTab: CharacterTable<MullikenIrrepSymbol, ClassSymbol<T>>,
// {
//     type CharTab = CorepCharacterTable<T, UG::CharTab>;
impl<T, RowSymbol, UG> CharacterProperties for MagneticRepresentedGroup<T, UG, RowSymbol>
where
    RowSymbol: ReducibleLinearSpaceSymbol<Subspace = UG::RowSymbol>,
    T: Mul<Output = T>
        + Hash
        + Eq
        + Clone
        + Sync
        + fmt::Debug
        + FiniteOrder<Int = u32>
        + Pow<i32, Output = T>,
    for<'a, 'b> &'b T: Mul<&'a T, Output = T>,
    UG: Clone
        + GroupProperties<GroupElement = T>
        + ClassProperties<GroupElement = T>
        + CharacterProperties,
{
    type RowSymbol = RowSymbol;
    type CharTab = CorepCharacterTable<Self::RowSymbol, UG::CharTab>;

    fn character_table(&self) -> &Self::CharTab {
        self.ircorep_character_table
            .as_ref()
            .expect("Ircorep character table not found for this group.")
    }

    /// Constructs the ircorep character table for this group.
    ///
    /// For each irrep in the unitary subgroup, the type of the ircorep it induces is determined
    /// using the Dimmock--Wheeler character test, then the ircorep's characters in the
    /// unitary-represented part of the full group are determined to give a square character table.
    fn construct_character_table(&mut self) {
        log::debug!("===============================================");
        log::debug!("Construction of ircorep character table begins.");
        log::debug!("===============================================");

        if self.unitary_subgroup().order() == self.order() {
            log::debug!(
                "The unitary subgroup order and the full group order are both {}. This is not a magnetic group.", self.order()
            );
            // This is not a magnetic group. There is nothing to do.
            return;
        }

        debug_assert_eq!(self.order() % 2, 0);
        debug_assert_eq!(self.order().div_euclid(2), self.unitary_subgroup().order());
        let unitary_order: i32 = self
            .order()
            .div_euclid(2)
            .try_into()
            .expect("Unable to convert the unitary group order to `i32`.");
        let unitary_chartab = self.unitary_subgroup().character_table();

        let mag_ctb = self.cayley_table();
        let uni_e2c = self.unitary_subgroup().element_to_conjugacy_classes();
        let mag_ccsyms = self.conjugacy_class_symbols();
        let uni_ccsyms = self.unitary_subgroup().conjugacy_class_symbols();
        let (_, a0_mag_idx) = self
            .elements()
            .iter()
            .find(|(op, _)| self.check_elem_antiunitary(op))
            .expect("No antiunitary elements found in the magnetic group.");

        let mut remaining_irreps = unitary_chartab.get_all_rows().clone();
        remaining_irreps.reverse();

        let mut ircoreps_ins: Vec<(RowSymbol, u8)> = Vec::new();
        while let Some(irrep) = remaining_irreps.pop() {
            log::debug!("Considering irrep {irrep} of the unitary subgroup...");
            let char_sum = self
                .elements()
                .iter()
                .filter(|(op, _)| self.check_elem_antiunitary(op))
                .fold(Character::zero(), |acc, (_, a_mag_idx)| {
                    let a2_mag_idx = mag_ctb[(*a_mag_idx, *a_mag_idx)];
                    let (a2, _) = self.elements().get_index(a2_mag_idx).unwrap_or_else(|| {
                        panic!("Element index `{a2_mag_idx}` not found in the magnetic group.")
                    });
                    let a2_uni_idx = *self.unitary_subgroup().elements().get(a2).unwrap_or_else(|| {
                        panic!("Element `{a2:?}` not found in the unitary subgroup.")
                    });
                    let (a2_uni_class, _) = uni_ccsyms.get_index(
                        uni_e2c[a2_uni_idx].unwrap_or_else(|| {
                            panic!("Conjugacy class for `{a2:?}` not found in the unitary subgroup.")
                        })
                    ).unwrap_or_else(|| {
                            panic!("Conjugacy class symbol for `{a2:?}` not found in the unitary subgroup.")
                    });
                    acc + unitary_chartab.get_character(&irrep, a2_uni_class)
                })
                .simplify();
            log::debug!("  Dimmock--Wheeler indicator for {irrep}: {char_sum}");
            let char_sum_c128 = char_sum.complex_value();
            approx::assert_relative_eq!(
                char_sum_c128.im,
                0.0,
                max_relative = char_sum.threshold
                    * unitary_order
                        .to_f64()
                        .expect("Unable to convert the unitary order to `f64`.")
                        .sqrt(),
                epsilon = char_sum.threshold
                    * unitary_order
                        .to_f64()
                        .expect("Unable to convert the unitary order to `f64`.")
                        .sqrt(),
            );
            approx::assert_relative_eq!(
                char_sum_c128.re,
                char_sum_c128.re.round(),
                max_relative = char_sum.threshold
                    * unitary_order
                        .to_f64()
                        .expect("Unable to convert the unitary order to `f64`.")
                        .sqrt(),
                epsilon = char_sum.threshold
                    * unitary_order
                        .to_f64()
                        .expect("Unable to convert the unitary order to `f64`.")
                        .sqrt(),
            );
            let char_sum = char_sum_c128.re.round();

            let (intertwining_number, ircorep) = if NumOrd(char_sum) == NumOrd(unitary_order) {
                // Irreducible corepresentation type a
                // Δ(u) is equivalent to Δ*[a^(-1)ua].
                // Δ(u) is contained once in the induced irreducible corepresentation.
                log::debug!(
                    "  Ircorep induced by {irrep} is of type (a) with intertwining number 1."
                );
                (1u8, RowSymbol::from_subspaces(&[(irrep, 1)]))
            } else if NumOrd(char_sum) == NumOrd(-unitary_order) {
                // Irreducible corepresentation type b
                // Δ(u) is equivalent to Δ*[a^(-1)ua].
                // Δ(u) is contained twice in the induced irreducible corepresentation.
                log::debug!(
                    "  Ircorep induced by {irrep} is of type (b) with intertwining number 4."
                );
                (4u8, RowSymbol::from_subspaces(&[(irrep, 2)]))
            } else if NumOrd(char_sum) == NumOrd(0i8) {
                // Irreducible corepresentation type c
                // Δ(u) is inequivalent to Δ*[a^(-1)ua].
                // Δ(u) and Δ*[a^(-1)ua] are contained the induced irreducible corepresentation.
                let irrep_conj_chars: Vec<Character> = unitary_chartab.get_all_cols().iter().enumerate().map(|(cc_idx, cc)| {
                    let u_unitary_idx = self.unitary_subgroup().conjugacy_classes()[cc_idx]
                        .iter()
                        .next()
                        .unwrap_or_else(|| panic!("No unitary elements found for conjugacy class `{cc}`."));
                    let (u, _) = self.unitary_subgroup()
                        .elements()
                        .get_index(*u_unitary_idx)
                        .unwrap_or_else(|| panic!("Unitary element with index `{u_unitary_idx}` cannot be retrieved."));
                    let u_mag_idx = self
                        .elements()
                        .get(u)
                        .unwrap_or_else(|| panic!("Unable to retrieve the index of unitary element `{u:?}` in the magnetic group."));
                    let ua0_mag_idx = mag_ctb[(*u_mag_idx, *a0_mag_idx)];
                    let mag_ctb_a0x = mag_ctb.slice(s![*a0_mag_idx, ..]);
                    let a0invua0_mag_idx = mag_ctb_a0x.iter().position(|&x| x == ua0_mag_idx).unwrap_or_else(|| {
                        panic!("No element `{ua0_mag_idx}` can be found in row `{a0_mag_idx}` of the magnetic group Cayley table.")
                    });
                    let (a0invua0, _) = self
                        .elements()
                        .get_index(a0invua0_mag_idx)
                        .unwrap_or_else(|| {
                            panic!("Unable to retrieve element with index `{a0invua0_mag_idx}` in the magnetic group.")
                        });
                    let a0invua0_unitary_idx = self.unitary_subgroup().elements()
                        .get(a0invua0)
                        .unwrap_or_else(|| {
                            panic!("Unable to retrieve the index of element `{a0invua0:?}` in the unitary subgroup.")
                        });
                    let (a0invua0_unitary_class, _) = uni_ccsyms
                        .get_index(uni_e2c[*a0invua0_unitary_idx].unwrap_or_else(|| {
                            panic!("Unable to retrieve the class for `{a0invua0:?}` in the unitary subgroup.")
                        }))
                        .unwrap_or_else(|| panic!("Unable to retrieve the class for `{a0invua0:?}` in the unitary subgroup."));
                    unitary_chartab.get_character(&irrep, a0invua0_unitary_class).complex_conjugate()
                }).collect();
                let all_irreps = unitary_chartab.get_all_rows();
                let (_, conj_irrep) = all_irreps
                    .iter()
                    .enumerate()
                    .find(|(irrep_idx, _)| {
                        unitary_chartab.array().row(*irrep_idx).to_vec() == irrep_conj_chars
                    })
                    .unwrap_or_else(|| panic!("Conjugate irrep for {irrep} not found."));
                assert!(remaining_irreps.remove(conj_irrep));

                log::debug!("  The Wigner-conjugate irrep of {irrep} is {conj_irrep}.");
                log::debug!("  Ircorep induced by {irrep} and {conj_irrep} is of type (c) with intertwining number 2.");
                (
                    2u8,
                    RowSymbol::from_subspaces(&[(irrep, 1), (conj_irrep.to_owned(), 1)]),
                )
            } else {
                log::error!(
                    "Unexpected `char_sum`: {char_sum}. This can only be ±{unitary_order} or 0."
                );
                panic!("Unexpected `char_sum`: {char_sum}. This can only be ±{unitary_order} or 0.")
            };
            ircoreps_ins.push((ircorep, intertwining_number));
        }

        let mut char_arr: Array2<Character> =
            Array2::zeros((ircoreps_ins.len(), self.class_number()));
        for (i, (ircorep, intertwining_number)) in ircoreps_ins.iter().enumerate() {
            for (mag_cc, &cc_idx) in mag_ccsyms {
                let mag_cc_rep = mag_cc.representative().unwrap_or_else(|| {
                    panic!(
                        "No representative element found for magnetic conjugacy class {mag_cc}."
                    );
                });
                let mag_cc_uni_idx = *self
                    .unitary_subgroup()
                    .elements()
                    .get(&mag_cc_rep)
                    .unwrap_or_else(|| {
                        panic!(
                            "Index for element {mag_cc_rep:?} not found in the unitary subgroup."
                        );
                    });
                let (uni_cc, _) = uni_ccsyms.get_index(
                    uni_e2c[mag_cc_uni_idx].unwrap_or_else(|| {
                        panic!("Unable to find the conjugacy class of element {mag_cc_rep:?} in the unitary subgroup.");
                    })
                ).unwrap_or_else(|| {
                    panic!("Unable to find the conjugacy class symbol of element {mag_cc_rep:?} in the unitary subgroup.");
                });

                char_arr[(i, cc_idx)] = ircorep
                    .subspaces()
                    .iter()
                    .fold(Character::zero(), |acc, (irrep, _)| {
                        acc + unitary_chartab.get_character(irrep, uni_cc)
                    });
                if *intertwining_number == 4 {
                    // Irreducible corepresentation type b
                    // The inducing irrep appears twice.
                    char_arr[(i, cc_idx)] *= 2;
                }
            }
        }

        let principal_classes = mag_ccsyms
            .iter()
            .filter_map(|(mag_cc, _)| {
                let mag_cc_rep = mag_cc.representative().unwrap_or_else(|| {
                    panic!("No representative element found for magnetic conjugacy class {mag_cc}.");
                });
                let mag_cc_uni_idx = *self.unitary_subgroup().elements().get(&mag_cc_rep).unwrap_or_else(|| {
                    panic!("Index for element {mag_cc_rep:?} not found in the unitary subgroup.");
                });
                let (uni_cc, _) = uni_ccsyms.get_index(
                    uni_e2c[mag_cc_uni_idx].unwrap_or_else(|| {
                        panic!("Unable to find the conjugacy class of element {mag_cc_rep:?} in the unitary subgroup.");
                    })
                ).unwrap_or_else(|| {
                    panic!("Unable to find the conjugacy class symbol of element {mag_cc_rep:?} in the unitary subgroup.");
                });
                if unitary_chartab.principal_classes().contains(uni_cc) {
                    Some(mag_cc.clone())
                } else {
                    None
                }
            }).collect::<Vec<_>>();

        let (ircoreps, ins): (Vec<RowSymbol>, Vec<u8>) = ircoreps_ins.into_iter().unzip();

        let chartab_name = if let Some(finite_name) = self.finite_subgroup_name().as_ref() {
            format!("{} > {finite_name}", self.name())
        } else {
            self.name().to_string()
        };
        self.ircorep_character_table = Some(CorepCharacterTable::new(
            chartab_name.as_str(),
            unitary_chartab.clone(),
            &ircoreps,
            &mag_ccsyms.keys().cloned().collect::<Vec<_>>(),
            &principal_classes,
            char_arr,
            &ins,
        ));

        log::debug!("=============================================");
        log::debug!("Construction of ircorep character table ends.");
        log::debug!("=============================================");
    }
}
