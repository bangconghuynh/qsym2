//! Traits enabling groups to construct and possess character tables.

use std::fmt;
use std::hash::Hash;
use std::ops::Mul;
use std::str::FromStr;

use approx;
use log;
use ndarray::{Array1, Array2, Zip, array, s};
use num::integer::lcm;
use num_modular::{ModularInteger, MontgomeryInt};
use num_ord::NumOrd;
use num_traits::{Inv, One, Pow, ToPrimitive, Zero};
use primes::is_prime;
use rayon::prelude::*;
use serde::{Serialize, de::DeserializeOwned};

use crate::chartab::character::Character;
use crate::chartab::chartab_symbols::{
    CollectionSymbol, LinearSpaceSymbol, ReducibleLinearSpaceSymbol,
};
use crate::chartab::modular_linalg::{
    modular_eig, split_2d_space, split_space, weighted_hermitian_inprod,
};
use crate::chartab::reducedint::{IntoLinAlgReducedInt, LinAlgMontgomeryInt};
use crate::chartab::unityroot::UnityRoot;
use crate::chartab::{CharacterTable, CorepCharacterTable, RepCharacterTable};
use crate::group::class::ClassProperties;
use crate::group::{
    FiniteOrder, GroupProperties, HasUnitarySubgroup, MagneticRepresentedGroup,
    UnitaryRepresentedGroup,
};

// =================
// Trait definitions
// =================

/// Trait to indicate the presence of character properties in a group and enable access to the
/// character table of the group.
pub trait CharacterProperties: ClassProperties
where
    Self::RowSymbol: LinearSpaceSymbol,
    Self::CharTab: CharacterTable<RowSymbol = Self::RowSymbol, ColSymbol = Self::ClassSymbol>,
{
    /// Type of the row-labelling symbols in the associated character table.
    type RowSymbol;

    /// Type of the associated character table whose row-labelling symbol type is constrained to be
    /// the same as [`Self::RowSymbol`].
    type CharTab;

    /// Returns a shared reference to the character table of this group.
    fn character_table(&self) -> &Self::CharTab;

    /// Returns a boolean indicating if this character table contains irreducible representations
    /// of a unitary-represented group.
    ///
    /// If `false`, then some elements in the group are not unitary-represented and one has
    /// corepresentations instead of representations.
    fn unitary_represented(&self) -> bool;
}

/// Trait for the ability to construct an irrep character table for the group.
///
/// This trait comes with a default implementation of character table calculation based on the
/// Burnside--Dixon algorithm with Schneider optimisation.
pub trait IrrepCharTabConstruction:
    CharacterProperties<
    CharTab = RepCharacterTable<
        <Self as CharacterProperties>::RowSymbol,
        <Self as ClassProperties>::ClassSymbol,
    >,
>
where
    Self: Sync,
    Self::GroupElement: Mul<Output = Self::GroupElement>
        + Hash
        + Eq
        + Clone
        + Sync
        + fmt::Debug
        + FiniteOrder<Int = u32>
        + Pow<i32, Output = Self::GroupElement>,
{
    /// Sets the irrep character table internally.
    fn set_irrep_character_table(&mut self, chartab: Self::CharTab);

    /// Constructs the irrep character table for this group using the Burnside--Dixon algorithm
    /// with Schneider optimisation.
    ///
    /// # References
    ///
    /// * Dixon, J. D. High speed computation of group characters. *Numerische Mathematik* **10**, 446–450 (1967).
    /// * Schneider, G. J. A. Dixon’s character table algorithm revisited. *Journal of Symbolic Computation* **9**, 601–606 (1990).
    /// * Grove, L. C. Groups and Characters. (John Wiley & Sons, Inc., 1997).
    ///
    /// # Panics
    ///
    /// Panics if the Frobenius--Schur indicator for any resulted irrep takes on unexpected values
    /// and thus implies that the computed irrep is invalid.
    #[allow(clippy::too_many_lines)]
    fn construct_irrep_character_table(&mut self) {
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
        log::debug!("      *** Burnside--Dixon algorithm ***      ");
        log::debug!("      ** with Schneider optimisation **      ");
        log::debug!("=============================================");

        // Identify a suitable finite field
        let m = (0..self.class_number())
            .map(|i| {
                self.get_cc_transversal(i)
                    .unwrap_or_else(|| panic!("No representative of class index `{i}` found."))
                    .order()
            })
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
        .ceil();
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
        let class_sizes: Vec<_> = (0..self.class_number())
            .map(|i| {
                self.class_size(i)
                    .expect("Not all class sizes can be found.")
            })
            .collect();
        log::debug!("Class sizes: {:?}", class_sizes);
        let sq_indices: Vec<usize> = (0..self.class_number())
            .map(|i| {
                let ele = self
                    .get_cc_transversal(i)
                    .unwrap_or_else(|| panic!("No representative of class index `{i}` found."));
                let elep2 = ele.pow(2);
                let elep2_i = self
                    .get_index_of(&elep2)
                    .unwrap_or_else(|| panic!("Element {elep2:?} not found."));
                self.get_cc_of_element_index(elep2_i)
                    .unwrap_or_else(|| panic!("Conjugacy class for element {elep2:?} not found."))
            })
            .collect();
        let inverse_conjugacy_classes = Some(
            (0..self.class_number())
                .map(|i| {
                    self.get_inverse_cc(i)
                        .expect("Not all class inverses could be obtained.")
                })
                .collect::<Vec<_>>(),
        );
        log::debug!(
            "Inverse conjugacy classes: {:?}",
            inverse_conjugacy_classes.as_ref().unwrap()
        );

        let mut eigvecs_1d: Vec<Array1<LinAlgMontgomeryInt<u32>>> = vec![];
        let mut success = false;
        let mut rotate_count = 0;
        let rs_original = (1..self.class_number()).collect::<Vec<_>>();

        while !success && rotate_count < self.class_number() {
            let mut rs = rs_original.clone();
            rs.rotate_left(rotate_count);
            log::debug!("Class matrix consideration order: {rs:?}");
            rotate_count += 1;
            let mut r_iter = rs.into_iter();
            eigvecs_1d = vec![];

            if self.class_number() == 1 {
                eigvecs_1d.push(array![modp.convert(1)]);
                success = true;
            } else {
                let mut degenerate_subspaces: Vec<Vec<Array1<LinAlgMontgomeryInt<u32>>>> = vec![];
                let ctb = self.cayley_table();
                let r = r_iter
                    .next()
                    .expect("Unable to obtain any `r` indices for class matrices.");
                log::debug!("Considering class matrix N{r}...");
                let nmat_1 = self.class_matrix(ctb, r).map(|&i| {
                    modp.convert(
                        u32::try_from(i)
                            .unwrap_or_else(|_| panic!("Unable to convert `{i}` to `u32`.")),
                    )
                });
                let eigs_1 = modular_eig(&nmat_1).unwrap_or_else(|err| {
                    log::error!("{err}");
                    panic!("{err}");
                });
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

                while !degenerate_subspaces.is_empty() {
                    log::debug!(
                        "Number of 1-D eigenvectors found: {} / {}.",
                        eigvecs_1d.len(),
                        self.class_number()
                    );
                    log::debug!(
                        "Number of degenerate subspaces found: {}.",
                        degenerate_subspaces.len(),
                    );

                    let mut degenerate_2d_subspaces = degenerate_subspaces
                        .iter()
                        .filter(|subspace| subspace.len() == 2)
                        .cloned()
                        .collect::<Vec<_>>();
                    if !degenerate_2d_subspaces.is_empty() {
                        degenerate_subspaces.retain(|subspace| subspace.len() > 2);
                        log::debug!(
                            "Number of 2-D degenerate subspaces found: {}",
                            degenerate_2d_subspaces.len()
                        );
                        log::debug!(
                            "Schneider's greedy algorithm for splitting 2-D spaces will be attempted."
                        );
                        while let Some(subspace) = degenerate_2d_subspaces.pop() {
                            if let Ok(subsubspaces) = split_2d_space(
                                &subspace,
                                &class_sizes,
                                &sq_indices,
                                inverse_conjugacy_classes.as_ref(),
                            ) {
                                eigvecs_1d.extend(subsubspaces.iter().filter_map(|subsubspace| {
                                    if subsubspace.len() == 1 {
                                        Some(subsubspace[0].clone())
                                    } else {
                                        None
                                    }
                                }));
                                log::debug!(
                                    "2-D subspace index {} successfully split.",
                                    degenerate_2d_subspaces.len()
                                );
                            } else {
                                log::debug!(
                                    "2-D subspace index {} cannot be split greedily.",
                                    degenerate_2d_subspaces.len()
                                );
                                log::debug!(
                                    "Stashing this 2-D subspace for splitting with class matrices..."
                                );
                                degenerate_subspaces.push(subspace);
                            }
                        }
                    }

                    if !degenerate_subspaces.is_empty() {
                        if let Some(r) = r_iter.next() {
                            log::debug!("Considering class matrix N{}...", r);
                            let nmat_r = self.class_matrix(ctb, r).map(|&i| {
                                modp.convert(u32::try_from(i).unwrap_or_else(|_| {
                                    panic!("Unable to convert `{i}` to `u32`.")
                                }))
                            });

                            let mut remaining_degenerate_subspaces: Vec<
                                Vec<Array1<LinAlgMontgomeryInt<u32>>>,
                            > = vec![];
                            while let Some(subspace) = degenerate_subspaces.pop() {
                                if let Ok(subsubspaces) = split_space(
                                    &nmat_r,
                                    &subspace,
                                    &class_sizes,
                                    inverse_conjugacy_classes.as_ref(),
                                ) {
                                    eigvecs_1d.extend(subsubspaces.iter().filter_map(
                                        |subsubspace| {
                                            if subsubspace.len() == 1 {
                                                Some(subsubspace[0].clone())
                                            } else {
                                                None
                                            }
                                        },
                                    ));
                                    remaining_degenerate_subspaces.extend(
                                        subsubspaces
                                            .iter()
                                            .filter(|subsubspace| subsubspace.len() > 1)
                                            .cloned(),
                                    );
                                } else {
                                    log::warn!(
                                        "Class matrix N{} failed to split degenerate subspace {}.",
                                        r,
                                        degenerate_subspaces.len()
                                    );
                                    log::warn!(
                                        "Stashing this subspace for the next class matrices..."
                                    );
                                    remaining_degenerate_subspaces.push(subspace);
                                }
                            }
                            degenerate_subspaces = remaining_degenerate_subspaces;
                        } else {
                            log::warn!(
                                "Class matrices exhausted before degenerate subspaces are fully resolved."
                            );
                            log::warn!(
                                "Restarting the solver using a different class matrix order..."
                            );
                            break;
                        }
                    }
                }
            }
            if eigvecs_1d.len() == self.class_number() {
                success = true;
            }
        }

        if eigvecs_1d.len() != self.class_number() {
            log::error!(
                "Degenerate subspaces failed to be fully resolved for all cyclic permutations of class matrices N1, ..., N{}.",
                self.class_number()
            );
        }
        assert_eq!(
            eigvecs_1d.len(),
            self.class_number(),
            "Degenerate subspaces failed to be fully resolved."
        );
        log::debug!(
            "Successfully found {} / {} one-dimensional eigenvectors for the class matrices.",
            eigvecs_1d.len(),
            self.class_number()
        );
        for (i, vec) in eigvecs_1d.iter().enumerate() {
            log::debug!("Eigenvector {}: {}", i, vec);
        }

        // Lift characters back to the complex field
        log::debug!("Lifting characters from GF({p}) back to the complex field...",);

        let chars: Vec<_> = eigvecs_1d
            .par_iter()
            .flat_map(|vec_i| {
                let vec_i_inprod = weighted_hermitian_inprod(
                    (vec_i, vec_i),
                    &class_sizes,
                    inverse_conjugacy_classes.as_ref(),
                );
                let dim_i = (1..=p.div_euclid(2))
                    .map(|d| vec_i_inprod.convert(d))
                    .find(|d_modp| {
                        Some(vec_i_inprod) == d_modp.square().inv()
                    })
                    .unwrap_or_else(|| {
                        log::error!("Unable to deduce the irrep dimensionality from ⟨θvi, θvi⟩ = {vec_i_inprod} where vi = {vec_i}.");
                        panic!("Unable to deduce the irrep dimensionality from ⟨θvi, θvi⟩ = {vec_i_inprod} where vi = {vec_i}.");
                    });
                log::debug!("⟨θvi, θvi⟩ = {vec_i_inprod} yields irrep dimensionality {}.", dim_i.residue());

                let tchar_i =
                    Zip::from(vec_i)
                        .and(class_sizes.as_slice())
                        .par_map_collect(|&v, &k| {
                            v * dim_i
                                / modp.convert(u32::try_from(k).unwrap_or_else(|_| {
                                    panic!("Unable to convert `{k}` to `u32`.")
                                }))
                        });
                let char_i: Vec<_> = (0..self.class_number())
                    .into_par_iter()
                    .map(|cc_i| {
                        let x = self
                            .get_cc_transversal(cc_i)
                            .unwrap_or_else(|| {
                                panic!("Representative of class index `{cc_i}` not found.")
                            });
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
                                    let x_l_idx = self
                                        .get_index_of(&x_l)
                                        .unwrap_or_else(|| panic!("Element {x_l:?} not found."));
                                    let x_l_class_idx =
                                        self.get_cc_of_element_index(x_l_idx).unwrap_or_else(|| {
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
                                        .expect("Unable to find the modular inverse of z^(slm//k).")
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
            .expect("Unable to construct the two-dimensional array of characters.");
        log::debug!("Lifting characters from GF({p}) back to the complex field... Done.",);

        let default_irrep_symbols = char_arr
            .rows()
            .into_iter()
            .enumerate()
            .map(|(irrep_i, _)| {
                Self::RowSymbol::from_str(&format!("||Λ|_({irrep_i})|"))
                    .ok()
                    .expect("Unable to construct default irrep symbols.")
            })
            .collect::<Vec<_>>();
        let default_principal_classes = vec![
            self.get_cc_symbol_of_index(0)
                .expect("No conjugacy class symbols found."),
        ];

        log::debug!("Computing the Frobenius--Schur indicators in GF({p})...");
        let group_order = class_sizes.iter().sum::<usize>();
        let group_order_u32 = u32::try_from(group_order).unwrap_or_else(|_| {
            panic!("Unable to convert the group order {group_order} to `u32`.")
        });
        log::debug!("Indices of squared conjugacy classes: {sq_indices:?}");
        let frobenius_schur_indicators: Vec<i8> = eigvecs_1d
            .par_iter()
            .map(|vec_i| {
                let vec_i_inprod = weighted_hermitian_inprod(
                    (vec_i, vec_i),
                    &class_sizes,
                    inverse_conjugacy_classes.as_ref(),
                );
                let dim_i = (1..=p.div_euclid(2))
                    .map(|d| vec_i_inprod.convert(d))
                    .find(|d_modp| {
                        Some(vec_i_inprod) == d_modp.square().inv()
                    })
                    .unwrap_or_else(|| {
                        log::error!("Unable to deduce the irrep dimensionality from ⟨θvi, θvi⟩ = {vec_i_inprod} where vi = {vec_i}.");
                        panic!("Unable to deduce the irrep dimensionality from ⟨θvi, θvi⟩ = {vec_i_inprod} where vi = {vec_i}.");
                    });

                let tchar_i =
                    Zip::from(vec_i)
                        .and(class_sizes.as_slice())
                        .par_map_collect(|&v, &k| {
                            v * dim_i
                                / modp.convert(u32::try_from(k).unwrap_or_else(|_| {
                                    panic!("Unable to convert `{k}` to `u32`.")
                                }))
                        });
                let fs_i = sq_indices
                    .iter()
                    .zip(class_sizes.iter())
                    .fold(modp.convert(0), |acc, (&sq_idx, &k)| {
                        let k_u32 = u32::try_from(k).unwrap_or_else(|_| {
                            panic!("Unable to convert the class size {k} to `u32`.");
                        });
                        acc + modp.convert(k_u32) * tchar_i[sq_idx]
                    }) / modp.convert(group_order_u32);

                if fs_i.is_one() {
                    1i8
                } else if Zero::is_zero(&fs_i) {
                    0i8
                } else if fs_i == modp.convert(modp.modulus() - 1) {
                    -1i8
                } else {
                    log::error!("Invalid Frobenius -- Schur indicator: `{fs_i}`.");
                    panic!("Invalid Frobenius -- Schur indicator: `{fs_i}`.");
                }
            }).collect();
        log::debug!("Computing the Frobenius--Schur indicators in GF({p})... Done.");

        let chartab_name = if let Some(finite_name) = self.finite_subgroup_name().as_ref() {
            format!("{} > {finite_name}", self.name())
        } else {
            self.name()
        };
        let ccsyms = (0..self.class_number())
            .map(|i| {
                self.get_cc_symbol_of_index(i)
                    .unwrap_or_else(|| {
                        let rep = self
                            .get_cc_transversal(i)
                            .unwrap_or_else(||
                                panic!("Unable to obtain a representative for conjugacy class `{i}`.")
                            );
                        panic!("Class symbol for conjugacy class `{i}` with representative element `{rep:?}` cannot be found.")
                    })
            })
            .collect::<Vec<_>>();
        self.set_irrep_character_table(RepCharacterTable::new(
            chartab_name.as_str(),
            &default_irrep_symbols,
            &ccsyms,
            &default_principal_classes,
            char_arr,
            &frobenius_schur_indicators,
        ));

        log::debug!("===========================================");
        log::debug!("     *** Burnside--Dixon algorithm ***     ");
        log::debug!("     ** with Schneider optimisation **     ");
        log::debug!("Construction of irrep character table ends.");
        log::debug!("===========================================");
    }
}

/// Trait for the ability to construct an ircorep character table for the group.
///
/// This trait comes with a default implementation of ircorep character table calculation based on
/// the irreps of the unitary subgroup.
pub trait IrcorepCharTabConstruction: HasUnitarySubgroup + CharacterProperties<
    CharTab = CorepCharacterTable<
        <Self as CharacterProperties>::RowSymbol,
        <<Self as HasUnitarySubgroup>::UnitarySubgroup as CharacterProperties>::CharTab,
    >
>
where
    Self: ClassProperties<ClassSymbol = <<Self as HasUnitarySubgroup>::UnitarySubgroup as ClassProperties>::ClassSymbol>,
    Self::RowSymbol: ReducibleLinearSpaceSymbol<
        Subspace = <
            <Self as HasUnitarySubgroup>::UnitarySubgroup as CharacterProperties
        >::RowSymbol
    >,
    <<Self as HasUnitarySubgroup>::UnitarySubgroup as ClassProperties>::ClassSymbol: Serialize + DeserializeOwned,
{
    /// Sets the ircorep character table internally.
    fn set_ircorep_character_table(&mut self, chartab: Self::CharTab);

    /// Constructs the ircorep character table for this group.
    ///
    /// For each irrep in the unitary subgroup, the type of the ircorep it induces is determined
    /// using the Dimmock--Wheeler character test, then the ircorep's characters in the
    /// unitary-represented part of the full group are determined to give a square character table.
    ///
    /// # References
    ///
    /// * Bradley, C. J. & Davies, B. L. Magnetic Groups and Their Corepresentations. *Rev. Mod. Phys.* **40**, 359–379 (1968).
    /// * Newmarch, J. D. & Golding, R. M. The character table for the corepresentations of magnetic groups. *J. Math. Phys.* **23**, 695–704 (1982).
    /// * Newmarch, J. D. Some character theory for groups of linear and antilinear operators. *J. Math. Phys.* **24**, 742–756 (1983).
    ///
    /// # Panics
    ///
    /// Panics if any calculated ircoreps are found to be invalid.
    fn construct_ircorep_character_table(&mut self) {
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

        let mag_ctb = self.cayley_table().expect("Cayley table not found for the magnetic group.");

        let mag = &self;
        let uni = self.unitary_subgroup();
        let mag_ccsyms = (0..mag.class_number()) .map(|i| {
            mag.get_cc_symbol_of_index(i).expect("Unable to retrieve all class symbols of the magnetic group.")
        })
        .collect::<Vec<_>>();
        let uni_ccsyms = (0..uni.class_number()).map(|i| {
            uni.get_cc_symbol_of_index(i).expect("Unable to retrieve all class symbols of the unitary subgroup.")
        }).collect::<Vec<_>>();
        let (a0_mag_idx, _) = self
            .elements()
            .clone()
            .into_iter()
            .enumerate()
            .find(|(_, op)| self.check_elem_antiunitary(op))
            .expect("No antiunitary elements found in the magnetic group.");

        let mut remaining_irreps = unitary_chartab.get_all_rows();
        remaining_irreps.reverse();

        let mut ircoreps_ins: Vec<(Self::RowSymbol, u8)> = Vec::new();
        while let Some(irrep) = remaining_irreps.pop() {
            log::debug!("Considering irrep {irrep} of the unitary subgroup...");
            let char_sum = self
                .elements()
                .clone()
                .into_iter()
                .enumerate()
                .filter(|(_, op)| self.check_elem_antiunitary(op))
                .fold(Character::zero(), |acc, (a_mag_idx, _)| {
                    let a2_mag_idx = mag_ctb[(a_mag_idx, a_mag_idx)];
                    let a2 = self.get_index(a2_mag_idx).unwrap_or_else(|| {
                        panic!("Element index `{a2_mag_idx}` not found in the magnetic group.")
                    });
                    let a2_uni_idx = self.unitary_subgroup().get_index_of(&a2).unwrap_or_else(|| {
                        panic!("Element `{a2:?}` not found in the unitary subgroup.")
                    });
                    let a2_uni_class = &uni_ccsyms[
                        uni.get_cc_of_element_index(a2_uni_idx).unwrap_or_else(|| {
                            panic!("Conjugacy class for `{a2:?}` not found in the unitary subgroup.")
                        })
                    ];
                    acc + unitary_chartab.get_character(&irrep, a2_uni_class)
                })
                .simplify();
            log::debug!("  Dimmock--Wheeler indicator for {irrep}: {char_sum}");
            let char_sum_c128 = char_sum.complex_value();
            approx::assert_relative_eq!(
                char_sum_c128.im,
                0.0,
                max_relative = char_sum.threshold()
                    * unitary_order
                        .to_f64()
                        .expect("Unable to convert the unitary order to `f64`.")
                        .sqrt(),
                epsilon = char_sum.threshold()
                    * unitary_order
                        .to_f64()
                        .expect("Unable to convert the unitary order to `f64`.")
                        .sqrt(),
            );
            approx::assert_relative_eq!(
                char_sum_c128.re,
                char_sum_c128.re.round(),
                max_relative = char_sum.threshold()
                    * unitary_order
                        .to_f64()
                        .expect("Unable to convert the unitary order to `f64`.")
                        .sqrt(),
                epsilon = char_sum.threshold()
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
                (1u8, Self::RowSymbol::from_subspaces(&[(irrep, 1)]))
            } else if NumOrd(char_sum) == NumOrd(-unitary_order) {
                // Irreducible corepresentation type b
                // Δ(u) is equivalent to Δ*[a^(-1)ua].
                // Δ(u) is contained twice in the induced irreducible corepresentation.
                log::debug!(
                    "  Ircorep induced by {irrep} is of type (b) with intertwining number 4."
                );
                (4u8, Self::RowSymbol::from_subspaces(&[(irrep, 2)]))
            } else if NumOrd(char_sum) == NumOrd(0i8) {
                // Irreducible corepresentation type c
                // Δ(u) is inequivalent to Δ*[a^(-1)ua].
                // Δ(u) and Δ*[a^(-1)ua] are contained the induced irreducible corepresentation.
                let irrep_conj_chars: Vec<Character> = unitary_chartab
                    .get_all_cols()
                    .iter()
                    .enumerate()
                    .map(|(cc_idx, cc)| {
                        let u_cc = self
                            .unitary_subgroup()
                            .get_cc_index(cc_idx)
                            .unwrap_or_else(|| panic!("Conjugacy class `{cc}` not found."));
                        let u_unitary_idx = u_cc
                            .iter()
                            .next()
                            .unwrap_or_else(|| panic!("No unitary elements found for conjugacy class `{cc}`."));
                        let u = self.unitary_subgroup()
                            .get_index(*u_unitary_idx)
                            .unwrap_or_else(|| panic!("Unitary element with index `{u_unitary_idx}` cannot be retrieved."));
                        let u_mag_idx = self
                            .get_index_of(&u)
                            .unwrap_or_else(|| panic!("Unable to retrieve the index of unitary element `{u:?}` in the magnetic group."));
                        let ua0_mag_idx = mag_ctb[(u_mag_idx, a0_mag_idx)];
                        let mag_ctb_a0x = mag_ctb.slice(s![a0_mag_idx, ..]);
                        let a0invua0_mag_idx = mag_ctb_a0x.iter().position(|&x| x == ua0_mag_idx).unwrap_or_else(|| {
                            panic!("No element `{ua0_mag_idx}` can be found in row `{a0_mag_idx}` of the magnetic group Cayley table.")
                        });
                        let a0invua0 = self
                            .get_index(a0invua0_mag_idx)
                            .unwrap_or_else(|| {
                                panic!("Unable to retrieve element with index `{a0invua0_mag_idx}` in the magnetic group.")
                            });
                        let a0invua0_unitary_idx = self.unitary_subgroup()
                            .get_index_of(&a0invua0)
                            .unwrap_or_else(|| {
                                panic!("Unable to retrieve the index of element `{a0invua0:?}` in the unitary subgroup.")
                            });
                        let a0invua0_unitary_class = &uni_ccsyms[
                            uni.get_cc_of_element_index(a0invua0_unitary_idx).unwrap_or_else(|| {
                                panic!("Unable to retrieve the class for `{a0invua0:?}` in the unitary subgroup.")
                            })
                        ];
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
                assert!(remaining_irreps.shift_remove(conj_irrep));

                log::debug!("  The Wigner-conjugate irrep of {irrep} is {conj_irrep}.");
                log::debug!("  Ircorep induced by {irrep} and {conj_irrep} is of type (c) with intertwining number 2.");
                (
                    2u8,
                    Self::RowSymbol::from_subspaces(&[(irrep, 1), (conj_irrep.to_owned(), 1)]),
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
            for cc_idx in 0..mag.class_number() {
                let mag_cc = &mag_ccsyms[cc_idx];
                let mag_cc_rep = mag_cc.representative().unwrap_or_else(|| {
                    panic!(
                        "No representative element found for magnetic conjugacy class {mag_cc}."
                    );
                });
                let mag_cc_uni_idx = self
                    .unitary_subgroup()
                    .get_index_of(mag_cc_rep)
                    .unwrap_or_else(|| {
                        panic!(
                            "Index for element {mag_cc_rep:?} not found in the unitary subgroup."
                        );
                    });
                let uni_cc = &uni_ccsyms[
                    uni.get_cc_of_element_index(mag_cc_uni_idx).unwrap_or_else(|| {
                        panic!("Unable to find the conjugacy class of element {mag_cc_rep:?} in the unitary subgroup.");
                    })
                ];

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

        let principal_classes = (0..mag.class_number())
            .filter_map(|i| {
                let mag_cc = &mag_ccsyms[i];
                let mag_cc_rep = mag_cc.representative().unwrap_or_else(|| {
                    panic!("No representative element found for magnetic conjugacy class {mag_cc}.");
                });
                let mag_cc_uni_idx = self.unitary_subgroup().get_index_of(mag_cc_rep).unwrap_or_else(|| {
                    panic!("Index for element {mag_cc_rep:?} not found in the unitary subgroup.");
                });
                let uni_cc = uni.get_cc_symbol_of_index(
                    uni.get_cc_of_element_index(mag_cc_uni_idx).unwrap_or_else(|| {
                        panic!("Unable to find the conjugacy class of element {mag_cc_rep:?} in the unitary subgroup.");
                    })
                ).unwrap_or_else(|| {
                    panic!("Unable to find the conjugacy class symbol of element {mag_cc_rep:?} in the unitary subgroup.");
                });
                if unitary_chartab.get_principal_cols().contains(&uni_cc) {
                    Some(mag_cc.clone())
                } else {
                    None
                }
            }).collect::<Vec<_>>();

        let (ircoreps, ins): (Vec<Self::RowSymbol>, Vec<u8>) = ircoreps_ins.into_iter().unzip();

        let chartab_name = if let Some(finite_name) = self.finite_subgroup_name().as_ref() {
            format!("{} > {finite_name}", self.name())
        } else {
            self.name()
        };
        self.set_ircorep_character_table(Self::CharTab::new(
            chartab_name.as_str(),
            unitary_chartab.clone(),
            &ircoreps,
            &mag_ccsyms,
            &principal_classes,
            char_arr,
            &ins,
        ));

        log::debug!("=============================================");
        log::debug!("Construction of ircorep character table ends.");
        log::debug!("=============================================");
    }
}

// =====================
// Trait implementations
// =====================

// ---------------------------------------------
// UnitaryRepresentedGroup trait implementations
// ---------------------------------------------

impl<T, RowSymbol, ColSymbol> CharacterProperties
    for UnitaryRepresentedGroup<T, RowSymbol, ColSymbol>
where
    RowSymbol: LinearSpaceSymbol + Sync,
    ColSymbol: CollectionSymbol<CollectionElement = T> + Sync,
    T: Mul<Output = T>
        + Inv<Output = T>
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

    fn unitary_represented(&self) -> bool {
        true
    }
}

impl<T, RowSymbol, ColSymbol> IrrepCharTabConstruction
    for UnitaryRepresentedGroup<T, RowSymbol, ColSymbol>
where
    RowSymbol: LinearSpaceSymbol + Sync,
    ColSymbol: CollectionSymbol<CollectionElement = T> + Sync,
    T: Mul<Output = T>
        + Inv<Output = T>
        + Hash
        + Eq
        + Clone
        + Sync
        + fmt::Debug
        + FiniteOrder<Int = u32>
        + Pow<i32, Output = T>,
    for<'a, 'b> &'b T: Mul<&'a T, Output = T>,
{
    fn set_irrep_character_table(&mut self, chartab: Self::CharTab) {
        self.irrep_character_table = Some(chartab)
    }
}

// ----------------------------------------------
// MagneticRepresentedGroup trait implementations
// ----------------------------------------------

impl<T, RowSymbol, UG> CharacterProperties for MagneticRepresentedGroup<T, UG, RowSymbol>
where
    RowSymbol: ReducibleLinearSpaceSymbol<Subspace = UG::RowSymbol> + Serialize + DeserializeOwned,
    T: Mul<Output = T>
        + Inv<Output = T>
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
    <UG as ClassProperties>::ClassSymbol: Serialize + DeserializeOwned,
    <UG as CharacterProperties>::CharTab: Serialize + DeserializeOwned,
    CorepCharacterTable<RowSymbol, <UG as CharacterProperties>::CharTab>:
        Serialize + DeserializeOwned,
{
    type RowSymbol = RowSymbol;
    type CharTab = CorepCharacterTable<Self::RowSymbol, UG::CharTab>;

    fn character_table(&self) -> &Self::CharTab {
        self.ircorep_character_table
            .as_ref()
            .expect("Ircorep character table not found for this group.")
    }

    fn unitary_represented(&self) -> bool {
        false
    }
}

impl<T, RowSymbol, UG> IrcorepCharTabConstruction for MagneticRepresentedGroup<T, UG, RowSymbol>
where
    RowSymbol: ReducibleLinearSpaceSymbol<Subspace = UG::RowSymbol> + Serialize + DeserializeOwned,
    T: Mul<Output = T>
        + Inv<Output = T>
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
    <UG as ClassProperties>::ClassSymbol: Serialize + DeserializeOwned,
    <UG as CharacterProperties>::CharTab: Serialize + DeserializeOwned,
    CorepCharacterTable<RowSymbol, <UG as CharacterProperties>::CharTab>:
        Serialize + DeserializeOwned,
{
    fn set_ircorep_character_table(&mut self, chartab: Self::CharTab) {
        self.ircorep_character_table = Some(chartab);
    }
}
