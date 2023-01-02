use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::hash::Hash;
use std::ops::Mul;

use approx;
use log;

use derive_builder::Builder;
use indexmap::IndexMap;
use itertools::Itertools;
use ndarray::{array, s, Array1, Array2, Array3, Axis, Zip};
use num::{integer::lcm, Complex};
use num_modular::{ModularInteger, MontgomeryInt};
use num_traits::{Inv, Pow};
use ordered_float::OrderedFloat;
use primes::is_prime;
use rayon::iter::ParallelBridge;
use rayon::prelude::*;

use crate::chartab::character::Character;
use crate::chartab::modular_linalg::{modular_eig, split_space, weighted_hermitian_inprod};
use crate::chartab::reducedint::{IntoLinAlgReducedInt, LinAlgMontgomeryInt};
use crate::chartab::unityroot::UnityRoot;
use crate::chartab::CharacterTable;
use crate::symmetry::symmetry_core::Symmetry;
use crate::symmetry::symmetry_element::symmetry_operation::{
    FiniteOrder, SpecialSymmetryTransformation,
};
use crate::symmetry::symmetry_element::{SymmetryElement, SymmetryOperation, SIG};
use crate::symmetry::symmetry_element_order::{ElementOrder, ORDER_1};
use crate::symmetry::symmetry_symbols::{
    deduce_mulliken_irrep_symbols, deduce_principal_classes, sort_irreps, ClassSymbol,
};

#[cfg(test)]
mod group_tests;

#[cfg(test)]
mod chartab_construction_tests;

/// A struct for managing abstract groups.
#[derive(Builder)]
struct Group<T: Hash + Eq + Clone + Sync + Debug + FiniteOrder> {
    /// A name for the group.
    name: String,

    /// An ordered hash table containing the elements of the group.
    #[builder(setter(custom))]
    elements: IndexMap<T, usize>,

    /// The order of the group.
    #[builder(
        setter(skip),
        default = "self.elements.as_ref().expect(\"No group elements found.\").len()"
    )]
    order: usize,

    /// An optional name if this group is actually a finite subgroup of [`Self::name`].
    #[builder(default = "None", setter(custom))]
    finite_subgroup_name: Option<String>,

    /// The Cayley table for this group w.r.t. the elements in [`Self::elements`].
    ///
    /// Elements are multiplied left-to-right, with functions written on the right.
    /// Row elements are on the left, column elements on the right.  So column
    /// elements act on the function first, followed by row elements.
    ///
    /// Each element in this array contains the index of the resultant operation
    /// from the composition, w.r.t. the array [`Self::elements`].
    #[builder(setter(skip), default = "None")]
    cayley_table: Option<Array2<usize>>,

    /// A vector of conjugacy classes for this group.
    ///
    /// Each element in the vector is a hashset containing the indices of the
    /// elements in [`Self::elements`] for a particular conjugacy class. This
    /// thus defines a multi-valued map from each conjugacy class index to one
    /// or more element indices.
    #[builder(setter(skip), default = "None")]
    conjugacy_classes: Option<Vec<HashSet<usize>>>,

    /// The conjugacy class representatives of the group.
    ///
    /// Each element in the vector is an index for a representative element of the corresponding
    /// conjugacy class.
    #[builder(setter(skip), default = "None")]
    conjugacy_class_transversal: Option<Vec<usize>>,

    /// An index map of symbols for the conjugacy classes in this group.
    ///
    /// Each key in the index map is a class symbol, and the associated value is the index of
    /// the corresponding conjugacy class in [`Self::conjugacy_classes`].
    #[builder(setter(skip), default = "None")]
    conjugacy_class_symbols: Option<IndexMap<ClassSymbol<T>, usize>>,

    /// A vector containing the indices of inverse conjugacy classes.
    ///
    /// Each index gives the inverse conjugacy class for the corresponding
    /// conjugacy class.
    #[builder(setter(skip), default = "None")]
    inverse_conjugacy_classes: Option<Vec<usize>>,

    /// The conjugacy class index of the elements in [`Self::elements`].
    ///
    /// This is the so-called inverse of [`Self::conjugacy_classes`]. This maps
    /// each element index to its corresponding conjugacy class index.
    #[builder(setter(skip), default = "None")]
    element_to_conjugacy_classes: Option<Vec<usize>>,

    /// The number of conjugacy classes of this group.
    ///
    /// This is also the number of distinct irreducible representations of the
    /// group.
    #[builder(setter(skip), default = "None")]
    class_number: Option<usize>,

    /// The class matrices $`\mathbf{N}`$ for the conjugacy classes in the group.
    ///
    /// Let $`K_i`$ be the $`i^{\textrm{th}}`conjugacy class of the group. The
    /// elements of the class matrix $`\mathbf{N}`$ are given by
    ///
    /// ```math
    ///     N_{rst} = \lvert \{ (x, y) \in K_r \times K_s : xy = z \in K_t \} \rvert,
    /// ```
    ///
    /// independent of any $`z \in K_t`$.
    #[builder(setter(skip), default = "None")]
    class_matrix: Option<Array3<usize>>,

    /// The character table for this group.
    #[builder(setter(skip), default = "None")]
    pub character_table: Option<CharacterTable<T>>,
}

impl<T: Hash + Eq + Clone + Sync + Debug + FiniteOrder> GroupBuilder<T> {
    fn elements(&mut self, elems: Vec<T>) -> &mut Self {
        self.elements = Some(
            elems
                .into_iter()
                .enumerate()
                .map(|(i, element)| (element, i))
                .collect(),
        );
        self
    }

    fn finite_subgroup_name(&mut self, name_opt: Option<String>) -> &mut Self {
        if name_opt.is_some() {
            if self.name.as_ref().expect("Group name not found.").clone() == *"O(3)"
                || self
                    .name
                    .as_ref()
                    .expect("Group name not found.")
                    .contains('∞')
            {
                self.finite_subgroup_name = Some(name_opt);
            } else {
                panic!(
                    "Setting a finite subgroup name for a non-infinite group is not supported yet."
                )
            }
        }
        self
    }
}

impl<T> Group<T>
where
    T: Hash + Eq + Clone + Sync + Send + Debug + Pow<i32, Output = T> + FiniteOrder<Int = u32>,
    for<'a, 'b> &'b T: Mul<&'a T, Output = T>,
{
    /// Returns a builder to construct a new group.
    ///
    /// # Returns
    ///
    /// A builder to construct a new group.
    pub fn builder() -> GroupBuilder<T> {
        GroupBuilder::default()
    }

    /// Constructs a group from its elements.
    ///
    /// # Arguments
    ///
    /// * name - A name to be given to the group.
    /// * elements - A vector of *all* group elements.
    ///
    /// # Returns
    ///
    /// A group with its Cayley table constructed and conjugacy classes determined.
    fn new(name: &str, elements: Vec<T>) -> Self {
        let mut grp = Self::builder()
            .name(name.to_string())
            .elements(elements)
            .build()
            .expect("Unable to construct a group.");
        grp.construct_cayley_table();
        grp.find_conjugacy_classes();
        grp.assign_class_symbols();
        grp.calc_class_matrix();
        grp
    }

    /// Checks if this group is Abelian.
    ///
    /// This method requires the Cayley table to have been constructed.
    ///
    /// # Returns
    ///
    /// A flag indicating if this group is Abelian.
    fn is_abelian(&self) -> bool {
        let ctb = self.cayley_table.as_ref().expect("Cayley table not found.");
        ctb == ctb.t()
    }

    /// Constructs the Cayley table for the group.
    ///
    /// This method sets the [`Self::cayley_table`] field.
    fn construct_cayley_table(&mut self) {
        log::debug!("Constructing Cayley table in parallel...");
        let mut ctb = Array2::<usize>::zeros((self.order, self.order));
        Zip::indexed(&mut ctb).par_for_each(|(i, j), k| {
            let (op_i_ref, _) = self.elements
                .get_index(i)
                .unwrap_or_else(|| panic!("Element with index {i} cannot be retrieved."));
            let (op_j_ref, _) = self.elements
                .get_index(j)
                .unwrap_or_else(|| panic!("Element with index {j} cannot be retrieved."));
            let op_k = op_i_ref * op_j_ref;
            *k = *self
                .elements
                .get(&op_k)
                .unwrap_or_else(|| panic!("Group closure not fulfilled. The composition {:?} * {:?} = {:?} is not contained in the group.",
                        op_i_ref,
                        op_j_ref,
                        &op_k));
        });
        self.cayley_table = Some(ctb);
        log::debug!("Constructing Cayley table in parallel... Done.");
    }

    /// Find the conjugacy classes and their inverses for the group.
    ///
    /// This method sets the [`Self::conjugacy_classes`], [`Self::inverse_conjugacy_classes`],
    /// [`Self::conjugacy_class_transversal`], [`Self::element_to_conjugacy_classes`], and
    /// [`Self::class_number`] fields.
    #[allow(clippy::too_many_lines)]
    fn find_conjugacy_classes(&mut self) {
        // Find conjugacy classes
        log::debug!("Finding conjugacy classes...");
        if self.is_abelian() {
            log::debug!("Abelian group found.");
            // Abelian group; each element is in its own conjugacy class.
            self.conjugacy_classes =
                Some((0usize..self.order).map(|i| HashSet::from([i])).collect());
            self.element_to_conjugacy_classes = Some((0usize..self.order).collect());
        } else {
            // Non-Abelian group.
            log::debug!("Non-Abelian group found.");
            let mut ccs: Vec<HashSet<usize>> = vec![HashSet::from([0usize])];
            let mut e2ccs = vec![0usize; self.order];
            let mut remaining_elements: HashSet<usize> = (1usize..self.order).collect();
            let ctb = self.cayley_table.as_ref().expect("Cayley table not found.");

            while !remaining_elements.is_empty() {
                // For a fixed g, find all h such that sg = hs for all s in the group.
                let g = *remaining_elements
                    .iter()
                    .next()
                    .expect("Unexpected empty `remaining_elements`.");
                let mut cur_cc = HashSet::from([g]);
                for s in 0usize..self.order {
                    let sg = ctb[[s, g]];
                    let ctb_xs = ctb.slice(s![.., s]);
                    let h = ctb_xs.iter().position(|&x| x == sg).unwrap_or_else(|| {
                        panic!("No element `{sg}` can be found in column `{s}` of Cayley table.")
                    });
                    if remaining_elements.contains(&h) {
                        remaining_elements.remove(&h);
                        cur_cc.insert(h);
                    }
                }
                ccs.push(cur_cc);
            }
            ccs.sort_by_key(|cc| {
                *cc.iter()
                    .min()
                    .expect("Unable to find the minimum element index in one conjugacy class.")
            });
            ccs.iter().enumerate().for_each(|(i, cc)| {
                cc.iter().for_each(|&j| e2ccs[j] = i);
            });
            self.conjugacy_classes = Some(ccs);
            assert!(e2ccs.iter().skip(1).all(|&x| x > 0));
            self.element_to_conjugacy_classes = Some(e2ccs);
        }
        self.class_number = Some(
            self.conjugacy_classes
                .as_ref()
                .expect("Conjugacy classes not found.")
                .len(),
        );
        log::debug!("Finding conjugacy classes... Done.");

        // Set conjugacy class transversal
        self.conjugacy_class_transversal = Some(
            self.conjugacy_classes
                .as_ref()
                .expect("Conjugacy classes not found.")
                .iter()
                .map(|cc| {
                    *cc.iter()
                        .next()
                        .expect("No conjugacy classes can be empty.")
                })
                .collect(),
        );

        // Set default class symbols
        // self.conjugacy_class_symbols = Some(IndexMap::from_iter(class_symbols_iter));

        // Find inverse conjugacy classes
        log::debug!("Finding inverse conjugacy classes...");
        let mut iccs: Vec<_> = self
            .conjugacy_classes
            .as_ref()
            .expect("Conjugacy classes not found.")
            .iter()
            .map(|_| 0usize)
            .collect();
        let mut remaining_classes: HashSet<_> =
            (1..self.class_number.expect("Class number not found.")).collect();
        let ctb = self.cayley_table.as_ref().expect("Cayley table not found.");
        while !remaining_classes.is_empty() {
            let class_index = *remaining_classes
                .iter()
                .next()
                .expect("Unexpected empty `remaining_classes`.");
            remaining_classes.remove(&class_index);
            let g = *self
                .conjugacy_classes
                .as_ref()
                .expect("Conjugacy classes not found.")[class_index]
                .iter()
                .next()
                .expect("No conjugacy classes can be empty.");
            let g_inv = ctb
                .slice(s![g, ..])
                .iter()
                .position(|&x| x == 0)
                .unwrap_or_else(|| {
                    panic!("No identity element can be found in row `{g}` of Cayley table.")
                });
            let inv_class_index = self
                .element_to_conjugacy_classes
                .as_ref()
                .expect("No element-to-conjugacy-class mappings found.")[g_inv];
            iccs[class_index] = inv_class_index;
            if remaining_classes.contains(&inv_class_index) {
                remaining_classes.remove(&inv_class_index);
                iccs[inv_class_index] = class_index;
            }
        }
        assert!(iccs.iter().skip(1).all(|&x| x > 0));
        self.inverse_conjugacy_classes = Some(iccs);
        log::debug!("Finding inverse conjugacy classes... Done.");
    }

    /// Assigns generic class symbols to the conjugacy classes.
    ///
    /// This method sets the [`Self::conjugacy_class_symbols`] field.
    fn assign_class_symbols(&mut self) {
        log::debug!("Assigning generic class symbols...");
        let class_sizes: Vec<_> = self
            .conjugacy_classes
            .as_ref()
            .expect("Conjugacy classes not found.")
            .iter()
            .map(HashSet::len)
            .collect();
        let class_symbols_iter = self
            .conjugacy_class_transversal
            .as_ref()
            .expect("Conjugacy class transversals not found.")
            .iter()
            .enumerate()
            .map(|(i, &rep_ele_index)| {
                let (rep_ele, _) = self.elements.get_index(rep_ele_index).unwrap_or_else(|| {
                    panic!("Element with index {rep_ele_index} cannot be retrieved.")
                });
                (
                    ClassSymbol::new(
                        format!("{}||K{i}||", class_sizes[i]).as_str(),
                        Some(rep_ele.clone()),
                    )
                    .unwrap_or_else(|_| {
                        panic!(
                            "Unable to construct a class symbol from `{}||K{i}||`.",
                            class_sizes[i]
                        )
                    }),
                    i,
                )
            });
        self.conjugacy_class_symbols = Some(class_symbols_iter.collect::<IndexMap<_, _>>());
        log::debug!("Assigning generic class symbols... Done.");
    }

    /// Calculates the class matrix $`\mathbf{N}`$ for the conjugacy classes in
    /// the group.
    ///
    /// Let $`K_i`$ be the $`i^{\textrm{th}}`conjugacy class of the group. The
    /// elements of the class matrix $`\mathbf{N}`$ are given by
    ///
    /// ```math
    ///     N_{rst} = \lvert \{ (x, y) \in K_r \times K_s : xy = z \in K_t \} \rvert,
    /// ```
    ///
    /// independent of any $`z \in K_t`$.
    ///
    /// This method sets the [`Self::class_matrix`] field.
    fn calc_class_matrix(&mut self) {
        let mut nmat = Array3::<usize>::zeros((
            self.class_number.expect("Class number not found."),
            self.class_number.expect("Class number not found."),
            self.class_number.expect("Class number not found."),
        ));
        for (r, class_r) in self
            .conjugacy_classes
            .as_ref()
            .expect("Conjugacy classes not found.")
            .iter()
            .enumerate()
        {
            let idx_r = class_r.iter().copied().collect::<Vec<_>>();
            for (s, class_s) in self
                .conjugacy_classes
                .as_ref()
                .expect("Conjugacy classes not found.")
                .iter()
                .enumerate()
            {
                let idx_s = class_s.iter().copied().collect::<Vec<_>>();
                let cayley_block_rs = self
                    .cayley_table
                    .as_ref()
                    .expect("Cayley table not found.")
                    .select(Axis(0), &idx_r)
                    .select(Axis(1), &idx_s)
                    .iter()
                    .copied()
                    .counts();

                for (t, class_t) in self
                    .conjugacy_classes
                    .as_ref()
                    .expect("Conjugacy classes not found.")
                    .iter()
                    .enumerate()
                {
                    nmat[[r, s, t]] = *cayley_block_rs
                        .get(
                            class_t
                                .iter()
                                .next()
                                .expect("No conjugacy classes can be empty."),
                        )
                        .unwrap_or(&0);
                }
            }
        }
        self.class_matrix = Some(nmat);
    }
}

impl<T> Group<T>
where
    T: Hash
        + Eq
        + Clone
        + Sync
        + Send
        + Debug
        + Pow<i32, Output = T>
        + SpecialSymmetryTransformation
        + FiniteOrder<Int = u32>,
    for<'a, 'b> &'b T: Mul<&'a T, Output = T>,
{
    /// Constructs the character table for this group using the Burnside--Dixon--Schneider
    /// algorithm.
    ///
    /// This method sets the [`Self::class_matrix`] field.
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

        // Identify a suitable finite field
        let m = self
            .elements
            .keys()
            .map(FiniteOrder::order)
            .reduce(lcm)
            .expect("Unable to find the LCM for the orders of the elements in this group.");
        let zeta = UnityRoot::new(1, m);
        log::debug!("Found group exponent m = {}.", m);
        log::debug!("Chosen primitive unity root ζ = {}.", zeta);

        let rf64 = (2.0 * (self.order as f64).sqrt() / (m as f64)).round();
        assert!(rf64.is_sign_positive());
        assert!(rf64 <= u32::MAX as f64);
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
        let class_sizes: Vec<_> = self
            .conjugacy_classes
            .as_ref()
            .expect("Conjugacy classes not found.")
            .iter()
            .map(HashSet::len)
            .collect();
        let inverse_conjugacy_classes = self.inverse_conjugacy_classes.as_ref();
        let mut eigvecs_1d: Vec<Array1<LinAlgMontgomeryInt<u32>>> = vec![];

        if self.class_number.expect("Class number not found.") == 1 {
            eigvecs_1d.push(array![modp.convert(1)]);
        } else {
            let mut degenerate_subspaces: Vec<Vec<Array1<LinAlgMontgomeryInt<u32>>>> = vec![];
            let nmat = self
                .class_matrix
                .as_ref()
                .expect("Class matrix not found.")
                .map(|&i| {
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
                    r < (self.class_number.expect("Class number not found.") - 1),
                    "Class matrices exhausted before degenerate subspaces are fully resolved."
                );

                r += 1;
                log::debug!(
                    "Number of 1-D eigenvectors found: {} / {}.",
                    eigvecs_1d.len(),
                    self.class_number.expect("Class number not found.")
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

        assert_eq!(
            eigvecs_1d.len(),
            self.class_number.expect("Class number not found.")
        );
        log::debug!(
            "Successfully found {} / {} one-dimensional eigenvectors for the class matrices.",
            eigvecs_1d.len(),
            self.class_number.expect("Class number not found.")
        );
        for (i, vec) in eigvecs_1d.iter().enumerate() {
            log::debug!("Eigenvector {}: {}", i, vec);
        }

        // Lift characters back to the complex field
        log::debug!(
            "Lifting characters from GF({}) back to the complex field...",
            p
        );
        let class_transversal = self
            .conjugacy_class_transversal
            .as_ref()
            .expect("Conjugacy class transversals not found.");

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
                    (dim2_mod_p as f64).sqrt().round(),
                    (dim2_mod_p as f64).sqrt()
                ) {
                    dim2_mod_p += p;
                }

                let dim_if64 = (dim2_mod_p as f64).sqrt().round();
                assert!(dim_if64.is_sign_positive());
                assert!(dim_if64 <= u32::MAX as f64);
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
                            .elements
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
                                        .elements
                                        .get(&x_l)
                                        .unwrap_or_else(|| panic!("Element {x_l:?} not found."));
                                    let x_l_class_idx =
                                        self.element_to_conjugacy_classes.as_ref().expect(
                                            "No element-to-conjugacy-class mappings found.",
                                        )[x_l_idx];
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
                        Character::new(&char_ij_terms)
                    })
                    .collect();
                char_i
            })
            .collect();

        let char_arr = Array2::from_shape_vec(
            (
                self.class_number.expect("Class number not found."),
                self.class_number.expect("Class number not found."),
            ),
            chars,
        )
        .expect("Unable to construct the two-dimensional table of characters.");
        log::debug!(
            "Lifting characters from GF({}) back to the complex field... Done.",
            p
        );

        let class_symbols = self
            .conjugacy_class_symbols
            .as_ref()
            .expect("No conjugacy class symbols found.");

        let i_cc = ClassSymbol::new("1||i||", None)
            .expect("Unable to construct a class symbol from `1||i||`.");
        let s_cc = ClassSymbol::new("1||σh||", None)
            .expect("Unable to construct a class symbol from `1||σh||`.");
        let mut force_proper_principal = if class_symbols.contains_key(&i_cc) {
            log::debug!(
                "Inversion centre exists. Principal-axis classes will be forced to be proper."
            );
            true
        } else if class_symbols.contains_key(&s_cc) {
            log::debug!(
                "Horizontal mirror plane exists. Principal-axis classes will be forced to be proper."
            );
            true
        } else {
            false
        };

        let force_principal = if matches!(self.name.as_str(), "O" | "Oh" | "Td")
            || matches!(
                self.finite_subgroup_name
                    .as_ref()
                    .unwrap_or(&String::new())
                    .as_str(),
                "O" | "Oh" | "Td"
            ) {
            force_proper_principal = false;
            let c3_cc: ClassSymbol<T> = ClassSymbol::new("8||C3||", None)
                .expect("Unable to construct a class symbol from `8||C3||`.");
            log::debug!(
                "Group is {}. Principal-axis classes will be forced to be {}. This is to obtain non-standard Mulliken symbols that are in line with conventions in the literature.",
                self.name,
                c3_cc
            );
            Some(c3_cc)
        } else {
            None
        };

        let principal_classes =
            deduce_principal_classes(class_symbols, force_proper_principal, force_principal);

        let char_arr = sort_irreps(&char_arr.view(), class_symbols, &principal_classes);

        let ordered_irreps =
            deduce_mulliken_irrep_symbols(&char_arr.view(), class_symbols, &principal_classes);

        let frobenius_schur_indicators: Vec<_> = ordered_irreps
            .iter()
            .enumerate()
            .map(|(irrep_i, _)| {
                let indicator: Complex<f64> =
                    self.elements
                        .keys()
                        .fold(Complex::new(0.0f64, 0.0f64), |acc, ele| {
                            let ele_2_idx =
                                self.elements.get(&ele.clone().pow(2)).unwrap_or_else(|| {
                                    panic!("Element {:?} not found.", &ele.clone().pow(2))
                                });
                            let class_2_j = self
                                .element_to_conjugacy_classes
                                .as_ref()
                                .expect("Conjugacy classes not found.")[*ele_2_idx];
                            acc + char_arr[[irrep_i, class_2_j]].complex_value()
                        })
                        / (self.order as f64);
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
                    approx::relative_eq!(
                        indicator.re,
                        1.0,
                        epsilon = 1e-14,
                        max_relative = 1e-14
                    ) || approx::relative_eq!(
                        indicator.re,
                        0.0,
                        epsilon = 1e-14,
                        max_relative = 1e-14
                    ) || approx::relative_eq!(
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

        let chartab_name = if let Some(finite_name) = self.finite_subgroup_name.as_ref() {
            format!("{} > {finite_name}", self.name)
        } else {
            self.name.clone()
        };
        self.character_table = Some(CharacterTable::new(
            chartab_name.as_str(),
            &ordered_irreps,
            &class_symbols.keys().cloned().collect::<Vec<_>>(),
            &principal_classes,
            char_arr,
            &frobenius_schur_indicators,
        ));
    }
}

impl Group<SymmetryOperation> {
    /// Assigns class symbols to the conjugacy classes.
    ///
    /// This method sets the [`Self::conjugacy_class_symbols`] field.
    #[allow(clippy::too_many_lines)]
    fn assign_class_symbols_from_symmetry(&mut self) {
        // Assign class symbols
        log::debug!("Assigning class symbols from symmetry operations...");
        let mut proper_class_orders: HashMap<(ElementOrder, Option<u32>, i32, String), usize> =
            HashMap::new();
        let mut improper_class_orders: HashMap<(ElementOrder, Option<u32>, i32, String), usize> =
            HashMap::new();
        let class_symbols_iter = self
            .conjugacy_classes
            .as_ref()
            .expect("Conjugacy classes not found.")
            .iter()
            .enumerate()
            .map(|(i, class_element_indices)| {
                let rep_ele_index = *class_element_indices
                    .iter()
                    .min_by_key(|&&j| {
                        let op = self
                            .elements
                            .get_index(j)
                            .unwrap_or_else(|| {
                                panic!("Element with index {j} cannot be retrieved.")
                            })
                            .0;
                        (op.power, op.generating_element.proper_power)
                    })
                    .expect("Unable to obtain a representative element index.");
                let (rep_ele, _) = self.elements.get_index(rep_ele_index).unwrap_or_else(|| {
                    panic!("Unable to retrieve group element with index `{rep_ele_index}`.")
                });
                if rep_ele.is_identity() {
                    (
                        ClassSymbol::new("1||E||", Some(rep_ele.clone()))
                            .expect("Unable to construct a class symbol from `1||E||`."),
                        i,
                    )
                } else if rep_ele.is_inversion() {
                    (
                        ClassSymbol::new("1||i||", Some(rep_ele.clone()))
                            .expect("Unable to construct a class symbol from `1||i||`."),
                        i,
                    )
                } else if rep_ele.is_time_reversal() {
                    (
                        ClassSymbol::new("1||θ||", Some(rep_ele.clone()))
                            .expect("Unable to construct a class symbol from `1||θ||`."),
                        i,
                    )
                } else {
                    let rep_proper_order = rep_ele.generating_element.proper_order;
                    let rep_proper_power = rep_ele.generating_element.proper_power;
                    let rep_power = rep_ele.power;
                    let rep_sub = rep_ele.generating_element.additional_subscript.clone();
                    let dash = if rep_ele.is_proper() {
                        if let Some(v) = proper_class_orders.get_mut(&(
                            rep_proper_order,
                            rep_proper_power,
                            rep_power,
                            rep_sub.clone(),
                        )) {
                            *v += 1;
                            "'".repeat(*v)
                        } else {
                            proper_class_orders.insert(
                                (rep_proper_order, rep_proper_power, rep_power, rep_sub),
                                0,
                            );
                            String::new()
                        }
                    } else if let Some(v) = improper_class_orders.get_mut(&(
                        rep_proper_order,
                        rep_proper_power,
                        rep_power,
                        rep_sub.clone(),
                    )) {
                        *v += 1;
                        "'".repeat(*v)
                    } else {
                        improper_class_orders
                            .insert((rep_proper_order, rep_proper_power, rep_power, rep_sub), 0);
                        String::new()
                    };
                    let size = class_element_indices.len();
                    (
                        ClassSymbol::new(
                            format!(
                                "{}||{}|^({})|",
                                size,
                                rep_ele.get_abbreviated_symbol(),
                                dash
                            )
                            .as_str(),
                            Some(rep_ele.clone()),
                        )
                        .unwrap_or_else(|_| {
                            panic!(
                                "Unable to construct a class symbol from `{size}||{}|^({dash})|`",
                                rep_ele.get_abbreviated_symbol()
                            )
                        }),
                        i,
                    )
                }
            });
        self.conjugacy_class_symbols = Some(class_symbols_iter.collect::<IndexMap<_, _>>());
        log::debug!("Assigning class symbols from symmetry operations... Done.");
    }
}

/// Constructs a group from molecular symmetry *elements* (not operations).
///
/// # Arguments
///
/// * `sym` - A molecular symmetry struct.
/// * `infinite_order_to_finite` - Interpret infinite-order generating
/// elements as finite-order generating elements to create a finite subgroup
/// of an otherwise infinite group.
///
/// # Returns
///
/// A finite abstract group struct.
#[allow(clippy::too_many_lines)]
fn group_from_molecular_symmetry(
    sym: &Symmetry,
    infinite_order_to_finite: Option<u32>,
) -> Group<SymmetryOperation> {
    let group_name = sym
        .point_group
        .as_ref()
        .expect("No point groups found.")
        .clone();

    let handles_infinite_group = if sym.is_infinite() {
        assert_ne!(infinite_order_to_finite, None);
        infinite_order_to_finite
    } else {
        None
    };

    if let Some(finite_order) = handles_infinite_group {
        if group_name == "O(3)" {
            assert!(
                matches!(finite_order, 2 | 4),
                "Finite order of {} is not yet supported for O(3).",
                finite_order
            );
        }
    }

    let id_element = sym
        .proper_elements
        .get(&ORDER_1)
        .expect("No identity elements found.")
        .iter()
        .next()
        .expect("No identity elements found.")
        .clone();

    let id_operation = SymmetryOperation::builder()
        .generating_element(id_element)
        .power(1)
        .build()
        .expect("Unable to construct an identity operation.");

    // Finite proper operations
    let mut proper_orders = sym.proper_elements.keys().collect::<Vec<_>>();
    proper_orders.sort_by(|a, b| {
        a.partial_cmp(b)
            .unwrap_or_else(|| panic!("`{a}` and `{b}` cannot be compared."))
    });
    let proper_operations =
        proper_orders
            .iter()
            .fold(vec![id_operation], |mut acc, proper_order| {
                sym.proper_elements
                    .get(proper_order)
                    .unwrap_or_else(|| panic!("Proper elements C{proper_order} not found."))
                    .iter()
                    .for_each(|proper_element| {
                        if let ElementOrder::Int(io) = proper_order {
                            acc.extend((1..*io).map(|power| {
                                SymmetryOperation::builder()
                                    .generating_element(proper_element.clone())
                                    .power(power.try_into().unwrap_or_else(|_| {
                                        panic!("Unable to convert `{power}` to `i32`.")
                                    }))
                                    .build()
                                    .expect("Unable to construct a symmetry operation.")
                            }));
                        }
                    });
                acc
            });

    // Finite proper operations from generators
    let proper_operations_from_generators = if let Some(fin_ord) = handles_infinite_group {
        sym.proper_generators
            .par_iter()
            .fold(std::vec::Vec::new, |mut acc, (order, proper_generators)| {
                for proper_generator in proper_generators.iter() {
                    let finite_order = match order {
                        ElementOrder::Int(io) => *io,
                        ElementOrder::Inf => fin_ord,
                    };
                    let finite_proper_element = SymmetryElement::builder()
                        .threshold(proper_generator.threshold)
                        .proper_order(ElementOrder::Int(finite_order))
                        .proper_power(1)
                        .axis(proper_generator.axis)
                        .kind(proper_generator.kind.clone())
                        .additional_superscript(proper_generator.additional_superscript.clone())
                        .additional_subscript(proper_generator.additional_subscript.clone())
                        .build()
                        .expect("Unable to construct a symmetry element.");
                    acc.extend((1..finite_order).map(|power| {
                        SymmetryOperation::builder()
                            .generating_element(finite_proper_element.clone())
                            .power(power.try_into().unwrap_or_else(|_| {
                                panic!("Unable to convert `{power}` to `i32`.")
                            }))
                            .build()
                            .expect("Unable to construct a symmetry operation.")
                    }));
                }
                acc
            })
            .reduce(std::vec::Vec::new, |mut acc, vec| {
                acc.extend(vec);
                acc
            })
    } else {
        vec![]
    };

    // Finite improper operations
    let mut improper_orders = sym.improper_elements.keys().collect::<Vec<_>>();
    improper_orders.sort_by(|a, b| {
        a.partial_cmp(b)
            .unwrap_or_else(|| panic!("`{a}` and `{b}` cannot be compared."))
    });
    let improper_operations = improper_orders
        .iter()
        .fold(vec![], |mut acc, improper_order| {
            sym.improper_elements
                .get(improper_order)
                .unwrap_or_else(|| panic!("Improper elements S{improper_order} not found."))
                .iter()
                .for_each(|improper_element| {
                    if let ElementOrder::Int(io) = improper_order {
                        acc.extend((1..(2 * *io)).step_by(2).map(|power| {
                            SymmetryOperation::builder()
                                .generating_element(improper_element.clone())
                                .power(power.try_into().unwrap_or_else(|_| {
                                    panic!("Unable to convert `{power}` to `i32`.")
                                }))
                                .build()
                                .expect("Unable to construct a symmetry operation.")
                        }));
                    }
                });
            acc
        });

    // Finite improper operations from generators
    let improper_operations_from_generators = if let Some(fin_ord) = handles_infinite_group {
        sym.improper_generators
            .par_iter()
            .fold(
                std::vec::Vec::new,
                |mut acc, (order, improper_generators)| {
                    for improper_generator in improper_generators.iter() {
                        let finite_order = match order {
                            ElementOrder::Int(io) => *io,
                            ElementOrder::Inf => fin_ord,
                        };
                        let finite_improper_element = SymmetryElement::builder()
                            .threshold(improper_generator.threshold)
                            .proper_order(ElementOrder::Int(finite_order))
                            .proper_power(1)
                            .axis(improper_generator.axis)
                            .kind(improper_generator.kind.clone())
                            .additional_superscript(
                                improper_generator.additional_superscript.clone(),
                            )
                            .additional_subscript(improper_generator.additional_subscript.clone())
                            .build()
                            .expect("Unable to construct a symmetry element.");
                        acc.extend((1..(2 * finite_order)).step_by(2).map(|power| {
                            SymmetryOperation::builder()
                                .generating_element(finite_improper_element.clone())
                                .power(power.try_into().unwrap_or_else(|_| {
                                    panic!("Unable to convert `{power}` to `i32`.")
                                }))
                                .build()
                                .expect("Unable to construct a symmetry operation.")
                        }));
                    }
                    acc
                },
            )
            .reduce(std::vec::Vec::new, |mut acc, vec| {
                acc.extend(vec);
                acc
            })
    } else {
        vec![]
    };

    let operations: HashSet<_> = if handles_infinite_group.is_none() {
        proper_operations
            .into_iter()
            .chain(proper_operations_from_generators)
            .chain(improper_operations)
            .chain(improper_operations_from_generators)
            .collect()
    } else {
        // Fulfil group closure
        log::debug!("Fulfilling closure for a finite subgroup of an infinite group...");
        let mut existing_operations: HashSet<_> = proper_operations
            .into_iter()
            .chain(proper_operations_from_generators)
            .chain(improper_operations)
            .chain(improper_operations_from_generators)
            .collect();
        let mut extra_operations = HashSet::<SymmetryOperation>::new();
        let mut npasses = 0;
        let mut nstable = 0;

        while nstable < 2 || npasses == 0 {
            let n_extra_operations = extra_operations.len();
            existing_operations.extend(extra_operations);

            npasses += 1;
            log::debug!(
                "Generating all group elements: {} pass{}, {} element{} (of which {} {} new)",
                npasses,
                {
                    if npasses > 1 {
                        "es"
                    } else {
                        ""
                    }
                }
                .to_string(),
                existing_operations.len(),
                {
                    if existing_operations.len() > 1 {
                        "s"
                    } else {
                        ""
                    }
                }
                .to_string(),
                n_extra_operations,
                {
                    if n_extra_operations > 1 {
                        "are"
                    } else {
                        "is"
                    }
                }
                .to_string(),
            );

            extra_operations = existing_operations
                .iter()
                .combinations_with_replacement(2)
                .par_bridge()
                .filter_map(|op_pairs| {
                    let op_i_ref = op_pairs[0];
                    let op_j_ref = op_pairs[1];
                    let op_k = op_i_ref * op_j_ref;
                    if existing_operations.contains(&op_k) {
                        None
                    } else if op_k.is_proper() {
                        Some(op_k)
                    } else {
                        Some(op_k.convert_to_improper_kind(&SIG))
                    }
                })
                .collect();
            if extra_operations.is_empty() {
                nstable += 1;
            } else {
                nstable = 0;
            }
        }

        assert_eq!(extra_operations.len(), 0);
        log::debug!(
            "Group closure reached with {} elements.",
            existing_operations.len()
        );
        existing_operations
    };

    let mut sorted_operations: Vec<SymmetryOperation> = operations.into_iter().collect();
    sorted_operations.sort_by_key(|op| {
        let (axis_closeness, closest_axis) = op.generating_element.closeness_to_cartesian_axes();
        (
            !op.is_proper(),
            !(op.is_identity() || op.is_inversion()),
            op.is_binary_rotation() || op.is_reflection(),
            -(i64::try_from(
                *op.total_proper_fraction
                    .expect("No total proper fractions found.")
                    .denom()
                    .expect("The denominator of the total proper fraction cannot be extracted."),
            )
            .unwrap_or_else(|_| {
                panic!(
                    "Unable to convert the denominator of `{:?}` to `i64`.",
                    op.total_proper_fraction
                )
            })),
            op.power,
            OrderedFloat(axis_closeness),
            closest_axis,
        )
    });

    let mut group = Group::<SymmetryOperation>::new(group_name.as_str(), sorted_operations);
    if handles_infinite_group.is_some() {
        let finite_group = if group.name.contains('∞') {
            // # C∞, C∞h, C∞v, S∞, D∞, D∞h, D∞d
            if group.name.as_bytes()[0] == b'D' {
                if matches!(
                    group
                        .name
                        .as_bytes()
                        .iter()
                        .last()
                        .expect("The last character in the group name cannot be retrieved."),
                    b'h' | b'd'
                ) {
                    assert_eq!(group.order % 4, 0);
                    group
                        .name
                        .replace('∞', format!("{}", group.order / 4).as_str())
                } else {
                    assert_eq!(group.order % 2, 0);
                    group
                        .name
                        .replace('∞', format!("{}", group.order / 2).as_str())
                }
            } else {
                assert!(matches!(group.name.as_bytes()[0], b'C' | b'S'));
                if matches!(
                    group
                        .name
                        .as_bytes()
                        .iter()
                        .last()
                        .expect("The last character in the group name cannot be retrieved."),
                    b'h' | b'v'
                ) {
                    assert_eq!(group.order % 2, 0);
                    if group.order > 2 {
                        group
                            .name
                            .replace('∞', format!("{}", group.order / 2).as_str())
                    } else {
                        assert_eq!(group.name.as_bytes()[0], b'C');
                        "Cs".to_string()
                    }
                } else {
                    group.name.replace('∞', format!("{}", group.order).as_str())
                }
            }
        } else {
            // O(3)
            match group.order {
                8 => "D2h".to_string(),
                48 => "Oh".to_string(),
                _ => panic!("Unsupported number of group elements."),
            }
        };
        group.finite_subgroup_name = Some(finite_group);
    }
    group.assign_class_symbols_from_symmetry();
    group.construct_character_table();
    group
}
