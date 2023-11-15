//! Conjugacy class structures.

use std::collections::HashSet;
use std::fmt;
use std::hash::Hash;
use std::ops::Mul;

use anyhow::{self, ensure, format_err};
use derive_builder::Builder;
use indexmap::IndexMap;
use itertools::Itertools;
use ndarray::{s, Array2};
use num_traits::Inv;
use serde::{de::DeserializeOwned, Deserialize, Serialize};

use crate::chartab::chartab_group::CharacterProperties;
use crate::chartab::chartab_symbols::{
    CollectionSymbol, LinearSpaceSymbol, ReducibleLinearSpaceSymbol,
};
use crate::group::{
    FiniteOrder, GroupProperties, HasUnitarySubgroup, MagneticRepresentedGroup,
    UnitaryRepresentedGroup,
};

// =================
// Trait definitions
// =================

/// Trait for conjugacy class properties of a finite group.
pub trait ClassProperties: GroupProperties
where
    Self::ClassSymbol: CollectionSymbol<CollectionElement = Self::GroupElement>,
    <Self as GroupProperties>::GroupElement: Inv<Output = <Self as GroupProperties>::GroupElement>,
{
    /// The type of class symbols.
    type ClassSymbol;

    // ----------------
    // Required methods
    // ----------------

    /// Computes the class structure of the group and store the result.
    fn compute_class_structure(&mut self) -> Result<(), anyhow::Error>;

    /// Given a class index, returns an optional shared reference to the set containing the indices
    /// of all elements in that class.
    ///
    /// # Arguments
    ///
    /// * `cc_idx` - A class index.
    ///
    /// # Returns
    ///
    /// Returns a shared reference to the set containing the indices of all elements in that class, or
    /// `None` if `cc_idx` is not a valid class index of the group.
    #[must_use]
    fn get_cc_index(&self, cc_idx: usize) -> Option<&HashSet<usize>>;

    /// Given an element index, returns an optional index of the conjugacy class to which the
    /// element belongs.
    ///
    /// # Arguments
    ///
    /// * `e_idx` - An element index.
    ///
    /// # Returns
    ///
    /// Returns an index of the conjugacy class to which the element belongs, or `None` if either
    /// the element does not have a conjugacy class, or the index is out of range.
    #[must_use]
    fn get_cc_of_element_index(&self, e_idx: usize) -> Option<usize>;

    /// Given a class index, returns an optional representative element of that conjugacy class.
    ///
    /// # Arguments
    ///
    /// * `cc_idx` - A class index.
    ///
    /// # Returns
    ///
    /// Returns a representative element of the class, or `None` if the class index is out of
    /// range.
    #[must_use]
    fn get_cc_transversal(&self, cc_idx: usize) -> Option<Self::GroupElement>;

    /// Given a conjugacy class symbol, returns the index of the corresponding conjugacy class.
    ///
    /// # Arguments
    ///
    /// * `cc_sym` - A conjugacy class symbol.
    ///
    /// # Returns
    ///
    /// Returns an index corresponding to the conjugacy class of `cc_sym`, or `None` if `cc_sym`
    /// does not exist in the group.
    #[must_use]
    fn get_index_of_cc_symbol(&self, cc_sym: &Self::ClassSymbol) -> Option<usize>;

    /// Given a class index, returns its conjugacy class symbol, if any.
    ///
    /// # Arguments
    ///
    /// * `cc_idx` - A class index.
    ///
    /// # Returns
    ///
    /// Returns a conjugacy class symbol, or `None` if such a symbol does not exist for the class,
    /// or if the class index is out of range.
    #[must_use]
    fn get_cc_symbol_of_index(&self, cc_idx: usize) -> Option<Self::ClassSymbol>;

    /// Given a predicate, returns conjugacy class symbols satisfying it.
    ///
    /// # Arguments
    ///
    /// * `predicate` - A predicate to filter conjugacy class symbols.
    ///
    /// # Returns
    ///
    /// Returns conjugacy class symbols satisfying `predicate`, or `None` if such a symbol does not
    /// exist for the class.
    #[must_use]
    fn filter_cc_symbols<P: FnMut(&Self::ClassSymbol) -> bool>(
        &self,
        predicate: P,
    ) -> Vec<Self::ClassSymbol>;

    /// Sets the conjugacy class symbols for this group.
    ///
    /// # Arguments
    ///
    /// `cc_symbols` - A sliced of owned conjugacy class symbols.
    fn set_class_symbols(&mut self, cc_symbols: &[Self::ClassSymbol]);

    /// Given a class index, returns an index for its inverse.
    ///
    /// The inverse of a class contains the inverses of its elements.
    ///
    /// # Arguments
    ///
    /// `cc_idx` - A class index.
    ///
    /// # Returns
    ///
    /// The index of the inverse of `cc_idx`, or `None` if the class index is out of range.
    #[must_use]
    fn get_inverse_cc(&self, cc_idx: usize) -> Option<usize>;

    /// Returns the number of conjugacy classes in the group.
    #[must_use]
    fn class_number(&self) -> usize;

    /// Given a class index, returns its size.
    ///
    /// # Arguments
    ///
    /// `cc_idx` - A class index.
    ///
    /// # Returns
    ///
    /// The size of the class with index `cc_idx`, or `None` if the class index is out of range.
    #[must_use]
    fn class_size(&self, cc_idx: usize) -> Option<usize>;

    // ----------------
    // Provided methods
    // ----------------

    /// The class matrix $`\mathbf{N}_r`$ for the conjugacy classes in the group.
    ///
    /// Let $`K_i`$ be the $`i^{\textrm{th}}`$ conjugacy class of the group. The
    /// elements of the class matrix $`\mathbf{N}_r`$ are given by
    ///
    /// ```math
    ///     N_{r, st} = \lvert \{ (x, y) \in K_r \times K_s : xy = z \in K_t \} \rvert,
    /// ```
    ///
    /// independent of any $`z \in K_t`$.
    ///
    /// # Arguments
    ///
    /// * `ctb_opt` - An optional Cayley table.
    /// * `r` - The index $`r`$.
    ///
    /// # Returns
    ///
    /// The class matrix $`\mathbf{N}_r`$.
    #[must_use]
    fn class_matrix(&self, ctb_opt: Option<&Array2<usize>>, r: usize) -> Array2<usize> {
        let class_number = self.class_number();
        let mut nmat_r = Array2::<usize>::zeros((class_number, class_number));
        let class_r = &self
            .get_cc_index(r)
            .unwrap_or_else(|| panic!("Conjugacy class index `{r}` not found."));

        if let Some(ctb) = ctb_opt {
            log::debug!("Computing class matrix N{r} using the Cayley table...");
            (0..class_number).for_each(|t| {
                let class_t = self
                    .get_cc_index(t)
                    .unwrap_or_else(|| panic!("Conjugacy class index `{t}` not found."));
                let rep_z_idx = *class_t
                    .iter()
                    .next()
                    .expect("No conjugacy classes can be empty.");
                for &x_idx in class_r.iter() {
                    let x_inv_idx = ctb
                        .slice(s![.., x_idx])
                        .iter()
                        .position(|&x| x == 0)
                        .unwrap_or_else(|| {
                            panic!("The inverse of element index `{x_idx}` cannot be found.")
                        });
                    let y_idx = ctb[[x_inv_idx, rep_z_idx]];
                    let s = self.get_cc_of_element_index(y_idx).unwrap_or_else(|| {
                        panic!("Conjugacy class of element index `{y_idx}` not found.")
                    });
                    nmat_r[[s, t]] += 1;
                }
            });
        } else {
            log::debug!("Computing class matrix N{r} without the Cayley table...");
            (0..class_number).for_each(|t| {
                let class_t = self
                    .get_cc_index(t)
                    .unwrap_or_else(|| panic!("Conjugacy class index `{t}` not found."));
                let rep_z_idx = *class_t
                    .iter()
                    .next()
                    .expect("No conjugacy classes can be empty.");
                let z = self
                    .get_index(rep_z_idx)
                    .unwrap_or_else(|| panic!("No element with index `{rep_z_idx}` found."));
                for &x_idx in class_r.iter() {
                    let x = self
                        .get_index(x_idx)
                        .unwrap_or_else(|| panic!("No element with index `{x_idx}` found."));
                    let y = x.clone().inv() * z.clone();
                    let y_idx = self
                        .get_index_of(&y)
                        .unwrap_or_else(|| panic!("Element `{y:?}` not found in this group."));
                    let s = self
                        .get_cc_of_element_index(y_idx)
                        .unwrap_or_else(|| panic!("Conjugacy class of element `{y:?}` not found."));
                    nmat_r[[s, t]] += 1;
                }
            });
        };

        log::debug!("Computing class matrix N{r}... Done.");
        nmat_r
    }
}

/// Trait for outputting summaries of conjugacy class properties.
pub trait ClassPropertiesSummary: ClassProperties
where
    <Self as GroupProperties>::GroupElement: fmt::Display,
{
    /// Outputs a class transversal as a nicely formatted table.
    fn class_transversal_to_string(&self) -> String {
        let cc_transversal = (0..self.class_number())
            .filter_map(|i| {
                let cc_opt = self.get_cc_symbol_of_index(i);
                let op_opt = self.get_cc_transversal(i);
                match (cc_opt, op_opt) {
                    (Some(cc), Some(op)) => Some((cc.to_string(), op.to_string())),
                    _ => None,
                }
            })
            .collect::<Vec<_>>();
        let cc_width = cc_transversal
            .iter()
            .map(|(cc, _)| cc.chars().count())
            .max()
            .unwrap_or(5)
            .max(5);
        let op_width = cc_transversal
            .iter()
            .map(|(_, op)| op.chars().count())
            .max()
            .unwrap_or(14)
            .max(14);

        let divider = "┈".repeat(cc_width + op_width + 4);
        let header = format!(" {:<cc_width$}  {:<}", "Class", "Representative");
        let body = Itertools::intersperse(
            cc_transversal
                .iter()
                .map(|(cc, op)| format!(" {:<cc_width$}  {:<}", cc, op)),
            "\n".to_string(),
        )
        .collect::<String>();

        Itertools::intersperse(
            [divider.clone(), header, divider.clone(), body, divider].into_iter(),
            "\n".to_string(),
        )
        .collect::<String>()
    }
}

// Blanket implementation
impl<G> ClassPropertiesSummary for G
where
    G: ClassProperties,
    G::GroupElement: fmt::Display,
{
}

// ======================================
// Struct definitions and implementations
// ======================================

/// Structure for managing class structures eagerly, *i.e.* all elements and their class maps are
/// stored.
#[derive(Builder, Clone, Serialize, Deserialize)]
pub(super) struct EagerClassStructure<T, ClassSymbol>
where
    T: Mul<Output = T> + Inv<Output = T> + Hash + Eq + Clone + Sync + fmt::Debug + FiniteOrder,
    ClassSymbol: CollectionSymbol<CollectionElement = T>,
{
    /// A vector of conjugacy classes.
    ///
    /// Each element in the vector is a hashset containing the indices of the
    /// elements in a certain ordered collection of group elements for a particular conjugacy
    /// class. This thus defines a multi-valued map from each conjugacy class index to one
    /// or more element indices in said collection.
    conjugacy_classes: Vec<HashSet<usize>>,

    /// The conjugacy class index of the elements in a certain ordered collection of group
    /// elements.
    ///
    /// This is the so-called inverse map of [`Self::conjugacy_classes`]. This maps
    /// each element index in said collection to its corresponding conjugacy class index.
    element_to_conjugacy_classes: Vec<Option<usize>>,

    /// The conjugacy class representatives of the group.
    ///
    /// Each element in the vector is an index for a representative element of the corresponding
    /// conjugacy class in a certain ordered collection of group elements.
    #[builder(setter(custom))]
    conjugacy_class_transversal: Vec<usize>,

    /// An index map of symbols for the conjugacy classes in this group.
    ///
    /// Each key in the index map is a class symbol, and the associated value is the index of
    /// the corresponding conjugacy class in [`Self::conjugacy_classes`].
    #[builder(setter(custom))]
    conjugacy_class_symbols: IndexMap<ClassSymbol, usize>,

    /// A vector containing the indices of inverse conjugacy classes.
    ///
    /// Each index gives the inverse conjugacy class for the corresponding
    /// conjugacy class.
    #[builder(setter(custom))]
    inverse_conjugacy_classes: Vec<usize>,
}

impl<T, ClassSymbol> EagerClassStructureBuilder<T, ClassSymbol>
where
    T: Mul<Output = T> + Inv<Output = T> + Hash + Eq + Clone + Sync + fmt::Debug + FiniteOrder,
    ClassSymbol: CollectionSymbol<CollectionElement = T>,
{
    fn conjugacy_class_transversal(&mut self) -> &mut Self {
        self.conjugacy_class_transversal = Some(
            self.conjugacy_classes
                .as_ref()
                .expect("Conjugacy classes have not been found.")
                .iter()
                .map(|cc| *cc.iter().min().expect("No conjugacy classes can be empty."))
                .collect::<Vec<usize>>(),
        );
        self
    }

    fn conjugacy_class_symbols(
        &mut self,
        group: &impl GroupProperties<GroupElement = T>,
    ) -> &mut Self {
        log::debug!("Assigning generic class symbols...");
        let class_sizes: Vec<_> = self
            .conjugacy_classes
            .as_ref()
            .expect("Conjugacy classes have not been found.")
            .iter()
            .map(HashSet::len)
            .collect();
        let class_symbols_iter = self
            .conjugacy_class_transversal
            .as_ref()
            .expect("A conjugacy class transversal has not been found.")
            .iter()
            .enumerate()
            .map(|(i, &rep_ele_index)| {
                let rep_ele = group.get_index(rep_ele_index).unwrap_or_else(|| {
                    panic!("Element with index {rep_ele_index} cannot be retrieved.")
                });
                (
                    ClassSymbol::from_reps(
                        format!("{}||K{i}||", class_sizes[i]).as_str(),
                        Some(vec![rep_ele]),
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
        self
    }

    fn inverse_conjugacy_classes(&mut self, ctb: &Array2<usize>) -> &mut Self {
        log::debug!("Finding inverse conjugacy classes...");
        let mut iccs: Vec<_> = self
            .conjugacy_classes
            .as_ref()
            .expect("Conjugacy classes have not been found.")
            .iter()
            .map(|_| 0usize)
            .collect();
        let class_number = self
            .conjugacy_classes
            .as_ref()
            .expect("Conjugacy classes have not been found.")
            .len();
        let mut remaining_classes: HashSet<_> = (1..class_number).collect();
        while !remaining_classes.is_empty() {
            let class_index = *remaining_classes
                .iter()
                .next()
                .expect("Unexpected empty `remaining_classes`.");
            remaining_classes.remove(&class_index);
            let g = *self
                .conjugacy_classes
                .as_ref()
                .expect("Conjugacy classes have not been found.")[class_index]
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
                .expect("Map from element to conjugacy class has not been found.")[g_inv]
                .unwrap_or_else(|| {
                    panic!("Element index `{g_inv}` does not have a conjugacy class.",)
                });
            iccs[class_index] = inv_class_index;
            if remaining_classes.contains(&inv_class_index) {
                remaining_classes.remove(&inv_class_index);
                iccs[inv_class_index] = class_index;
            }
        }
        assert!(iccs.iter().skip(1).all(|&x| x > 0));
        self.inverse_conjugacy_classes = Some(iccs);
        log::debug!("Finding inverse conjugacy classes... Done.");
        self
    }

    fn custom_inverse_conjugacy_classes(&mut self, iccs: Vec<usize>) -> &mut Self {
        log::debug!("Setting custom inverse conjugacy classes...");
        assert_eq!(
            iccs.len(),
            self.conjugacy_classes
                .as_ref()
                .expect("Conjugacy classes have not been set.")
                .len(),
            "The provided inverse conjugacy class structure does not have the correct class number."
        );
        self.inverse_conjugacy_classes = Some(iccs);
        log::debug!("Setting custom inverse conjugacy classes... Done.");
        self
    }
}

impl<T, ClassSymbol> EagerClassStructure<T, ClassSymbol>
where
    T: Mul<Output = T> + Inv<Output = T> + Hash + Eq + Clone + Sync + fmt::Debug + FiniteOrder,
    ClassSymbol: CollectionSymbol<CollectionElement = T>,
{
    /// Returns a builder to construct a new class structure.
    fn builder() -> EagerClassStructureBuilder<T, ClassSymbol> {
        EagerClassStructureBuilder::<T, ClassSymbol>::default()
    }

    /// Constructs a new eager class structure.
    ///
    /// # Arguments
    ///
    /// * `group` - A group with its Cayley table computed.
    /// * `conjugacy_classes` - A vector of hashsets, each of which contains the indices of the
    /// elements in `group` that are in the same conjugacy class.
    /// * `element_to_conjugacy_classes` - A vector containing the conjugacy class indices for the
    /// elements in `group`. An element might not have a conjugacy class (*e.g.*
    /// antiunitary-represented elements in magnetic-represented groups).
    ///
    /// # Returns
    ///
    /// A new class structure.
    fn new(
        group: &impl GroupProperties<GroupElement = T>,
        conjugacy_classes: Vec<HashSet<usize>>,
        element_to_conjugacy_classes: Vec<Option<usize>>,
    ) -> Self {
        let ctb_opt = group.cayley_table();
        let ctb = ctb_opt
            .as_ref()
            .expect("Cayley table not found for this group.");
        Self::builder()
            .conjugacy_classes(conjugacy_classes)
            .element_to_conjugacy_classes(element_to_conjugacy_classes)
            .conjugacy_class_transversal()
            .conjugacy_class_symbols(group)
            .inverse_conjugacy_classes(ctb)
            .build()
            .expect("Unable to construct a `EagerClassStructure`.")
    }

    /// Constructs a new eager class structure without using any information from any Cayley table.
    ///
    /// # Arguments
    ///
    /// * `conjugacy_classes` - A vector of hashsets, each of which contains the indices of the
    /// elements in `group` that are in the same conjugacy class.
    /// * `element_to_conjugacy_classes` - A vector containing the conjugacy class indices for the
    /// elements in `group`. An element might not have a conjugacy class (*e.g.*
    /// antiunitary-represented elements in magnetic-represented groups).
    ///
    /// # Returns
    ///
    /// A new class structure.
    fn new_no_ctb(
        group: &impl GroupProperties<GroupElement = T>,
        conjugacy_classes: Vec<HashSet<usize>>,
        element_to_conjugacy_classes: Vec<Option<usize>>,
        inverse_conjugacy_classes: Vec<usize>,
    ) -> Self {
        Self::builder()
            .conjugacy_classes(conjugacy_classes)
            .element_to_conjugacy_classes(element_to_conjugacy_classes)
            .conjugacy_class_transversal()
            .conjugacy_class_symbols(group)
            .custom_inverse_conjugacy_classes(inverse_conjugacy_classes)
            .build()
            .expect("Unable to construct a `EagerClassStructure`.")
    }

    /// Returns the number of conjugacy classes in the class structure.
    #[must_use]
    fn class_number(&self) -> usize {
        self.conjugacy_classes.len()
    }

    /// Sets the symbols of the conjugacy classes.
    ///
    /// # Arguments
    ///
    /// `csyms` - A slice of class symbols.
    ///
    /// # Panics
    ///
    /// Panics if the length of `csyms` does not match that of [`Self::conjugacy_classes`].
    fn set_class_symbols(&mut self, csyms: &[ClassSymbol]) {
        assert_eq!(csyms.len(), self.conjugacy_classes.len());
        self.conjugacy_class_symbols = csyms
            .iter()
            .enumerate()
            .map(|(i, cc)| (cc.clone(), i))
            .collect::<IndexMap<_, _>>();
    }
}

// =====================
// Trait implementations
// =====================

// ---------------------------------------------
// UnitaryRepresentedGroup trait implementations
// ---------------------------------------------

impl<T, RowSymbol, ColSymbol> ClassProperties for UnitaryRepresentedGroup<T, RowSymbol, ColSymbol>
where
    T: Mul<Output = T> + Inv<Output = T> + Hash + Eq + Clone + Sync + fmt::Debug + FiniteOrder,
    for<'a, 'b> &'b T: Mul<&'a T, Output = T>,
    <Self as GroupProperties>::GroupElement: Inv,
    RowSymbol: LinearSpaceSymbol,
    ColSymbol: CollectionSymbol<CollectionElement = T>,
{
    type ClassSymbol = ColSymbol;

    /// Compute the class structure of this unitary-represented group that is induced by the
    /// following equivalence relation:
    ///
    /// ```math
    ///     g \sim h \Leftrightarrow \exists u : h = u g u^{-1}.
    /// ```
    fn compute_class_structure(&mut self) -> Result<(), anyhow::Error> {
        log::debug!("Finding unitary conjugacy classes...");
        let order = self.abstract_group.order();
        let (ccs, e2ccs) = if self.abstract_group.is_abelian() {
            log::debug!("Abelian group found.");
            // Abelian group; each element is in its own conjugacy class.
            (
                (0usize..order)
                    .map(|i| HashSet::from([i]))
                    .collect::<Vec<_>>(),
                (0usize..order).map(Some).collect::<Vec<_>>(),
            )
        } else {
            // Non-Abelian group.
            log::debug!("Non-Abelian group found.");
            let mut ccs: Vec<HashSet<usize>> = vec![HashSet::from([0usize])];
            let mut e2ccs = vec![0usize; order];
            let mut remaining_elements: HashSet<usize> = (1usize..order).collect();
            let ctb = self.abstract_group.cayley_table.as_ref().expect(
                "Cayley table required for computing unitary class structure, but not found.",
            );

            while !remaining_elements.is_empty() {
                // For a fixed g, find all h such that sg = hs for all s in the group.
                let g = *remaining_elements
                    .iter()
                    .next()
                    .expect("Unexpected empty `remaining_elements`.");
                let mut cur_cc = HashSet::from([g]);
                for s in 0usize..order {
                    let sg = ctb[[s, g]];
                    let ctb_xs = ctb.slice(s![.., s]);
                    let h = ctb_xs.iter().position(|&x| x == sg).ok_or_else(|| {
                        format_err!(
                            "No element `{sg}` can be found in column `{s}` of Cayley table."
                        )
                    })?;
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
            assert!(e2ccs.iter().skip(1).all(|&x| x > 0));
            (ccs, e2ccs.iter().map(|&i| Some(i)).collect::<Vec<_>>())
        };

        let class_structure =
            EagerClassStructure::<T, Self::ClassSymbol>::new(&self.abstract_group, ccs, e2ccs);
        self.class_structure = Some(class_structure);
        log::debug!("Finding unitary conjugacy classes... Done.");
        Ok(())
    }

    #[must_use]
    fn get_cc_index(&self, cc_idx: usize) -> Option<&HashSet<usize>> {
        self.class_structure
            .as_ref()
            .expect("No class structure found.")
            .conjugacy_classes
            .get(cc_idx)
    }

    #[must_use]
    fn get_cc_of_element_index(&self, e_idx: usize) -> Option<usize> {
        self.class_structure
            .as_ref()
            .expect("No class structure found.")
            .element_to_conjugacy_classes[e_idx]
    }

    #[must_use]
    fn get_cc_transversal(&self, cc_idx: usize) -> Option<Self::GroupElement> {
        self.class_structure
            .as_ref()
            .expect("No class structure found.")
            .conjugacy_class_transversal
            .get(cc_idx)
            .and_then(|&i| self.get_index(i))
    }

    #[must_use]
    fn get_index_of_cc_symbol(&self, cc_sym: &Self::ClassSymbol) -> Option<usize> {
        self.class_structure
            .as_ref()
            .expect("No class structure found.")
            .conjugacy_class_symbols
            .get_index_of(cc_sym)
    }

    #[must_use]
    fn get_cc_symbol_of_index(&self, cc_idx: usize) -> Option<Self::ClassSymbol> {
        self.class_structure
            .as_ref()
            .expect("No class structure found.")
            .conjugacy_class_symbols
            .get_index(cc_idx)
            .map(|(cc_sym, _)| cc_sym.clone())
    }

    #[must_use]
    fn filter_cc_symbols<P: FnMut(&Self::ClassSymbol) -> bool>(
        &self,
        predicate: P,
    ) -> Vec<Self::ClassSymbol> {
        self.class_structure
            .as_ref()
            .expect("No class structure found.")
            .conjugacy_class_symbols
            .keys()
            .cloned()
            .filter(predicate)
            .collect::<Vec<_>>()
    }

    fn set_class_symbols(&mut self, cc_symbols: &[Self::ClassSymbol]) {
        self.class_structure
            .as_mut()
            .unwrap()
            .set_class_symbols(cc_symbols);
    }

    #[must_use]
    fn get_inverse_cc(&self, cc_idx: usize) -> Option<usize> {
        self.class_structure
            .as_ref()
            .expect("No class structure found.")
            .inverse_conjugacy_classes
            .get(cc_idx)
            .cloned()
    }

    #[must_use]
    fn class_number(&self) -> usize {
        self.class_structure
            .as_ref()
            .expect("No class structure found.")
            .class_number()
    }

    #[must_use]
    fn class_size(&self, cc_idx: usize) -> Option<usize> {
        self.class_structure
            .as_ref()
            .expect("No class structure found.")
            .conjugacy_classes
            .get(cc_idx)
            .map(|cc| cc.len())
    }
}

// ----------------------------------------------
// MagneticRepresentedGroup trait implementations
// ----------------------------------------------

impl<T, UG, RowSymbol> ClassProperties for MagneticRepresentedGroup<T, UG, RowSymbol>
where
    T: Mul<Output = T> + Inv<Output = T> + Hash + Eq + Clone + Sync + fmt::Debug + FiniteOrder,
    for<'a, 'b> &'b T: Mul<&'a T, Output = T>,
    <Self as GroupProperties>::GroupElement: Inv,
    UG: Clone + GroupProperties<GroupElement = T> + CharacterProperties,
    RowSymbol: ReducibleLinearSpaceSymbol<Subspace = UG::RowSymbol> + Serialize + DeserializeOwned,
    <UG as ClassProperties>::ClassSymbol: Serialize + DeserializeOwned,
    <UG as CharacterProperties>::CharTab: Serialize + DeserializeOwned,
{
    type ClassSymbol = UG::ClassSymbol;

    /// Compute the class structure of this magnetic-represented group that is induced by the
    /// following equivalence relation:
    ///
    /// ```math
    ///     g \sim h \Leftrightarrow
    ///     \exists u : h = u g u^{-1}
    ///     \quad \textrm{or} \quad
    ///     \exists a : h = a g^{-1} a^{-1},
    /// ```
    ///
    /// where $`u`$ is unitary-represented and $`a`$ is antiunitary-represented in the group.
    fn compute_class_structure(&mut self) -> Result<(), anyhow::Error> {
        log::debug!("Finding magnetic conjugacy classes...");
        let order = self.abstract_group.order();
        let mut ccs: Vec<HashSet<usize>> = vec![HashSet::from([0usize])];
        let mut e2ccs = vec![None; order];
        let mut remaining_unitary_elements = self
            .elements()
            .iter()
            .enumerate()
            .skip(1)
            .filter_map(|(i, op)| {
                if self.check_elem_antiunitary(op) {
                    None
                } else {
                    Some(i)
                }
            })
            .collect::<HashSet<usize>>();
        let ctb =
            self.abstract_group.cayley_table.as_ref().expect(
                "Cayley table required for computing magnetic class structure, but not found.",
            );

        while !remaining_unitary_elements.is_empty() {
            // For a fixed unitary g, find all unitary h such that ug = hu for all unitary u
            // in the group, and all unitary h such that ag^(-1) = ha for all antiunitary a in
            // the group.
            let g = *remaining_unitary_elements
                .iter()
                .next()
                .expect("Unexpected empty `remaining_elements`.");
            let ctb_xg = ctb.slice(s![.., g]);
            let ginv = ctb_xg
                .iter()
                .position(|&x| x == 0)
                .unwrap_or_else(|| panic!("The inverse of `{g}` cannot be found."));
            let mut cur_cc = HashSet::from([g]);
            for (s, op) in self.elements().iter().enumerate() {
                let h = if self.check_elem_antiunitary(op) {
                    // s denotes a.
                    let sginv = ctb[[s, ginv]];
                    let ctb_xs = ctb.slice(s![.., s]);
                    ctb_xs.iter().position(|&x| x == sginv).ok_or_else(|| {
                        format_err!("No element `{sginv}` can be found in column `{s}` of Cayley table.")
                    })?
                } else {
                    // s denotes u.
                    let sg = ctb[[s, g]];
                    let ctb_xs = ctb.slice(s![.., s]);
                    ctb_xs.iter().position(|&x| x == sg).ok_or_else(|| {
                        format_err!("No element `{sg}` can be found in column `{s}` of Cayley table.")
                    })?
                };
                if remaining_unitary_elements.contains(&h) {
                    remaining_unitary_elements.remove(&h);
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
            cc.iter().for_each(|&j| e2ccs[j] = Some(i));
        });
        ensure!(e2ccs
            .iter()
            .skip(1)
            .all(|x_opt| if let Some(x) = x_opt { *x > 0 } else { true }));

        let class_structure =
            EagerClassStructure::<T, Self::ClassSymbol>::new(&self.abstract_group, ccs, e2ccs);
        self.class_structure = Some(class_structure);
        log::debug!("Finding magnetic conjugacy classes... Done.");
        Ok(())
    }

    #[must_use]
    fn get_cc_index(&self, cc_idx: usize) -> Option<&HashSet<usize>> {
        self.class_structure
            .as_ref()
            .expect("No class structure found.")
            .conjugacy_classes
            .get(cc_idx)
    }

    #[must_use]
    fn get_cc_of_element_index(&self, e_idx: usize) -> Option<usize> {
        self.class_structure
            .as_ref()
            .expect("No class structure found.")
            .element_to_conjugacy_classes[e_idx]
    }

    #[must_use]
    fn get_cc_transversal(&self, cc_idx: usize) -> Option<Self::GroupElement> {
        self.class_structure
            .as_ref()
            .expect("No class structure found.")
            .conjugacy_class_transversal
            .get(cc_idx)
            .and_then(|&i| self.get_index(i))
    }

    #[must_use]
    fn get_index_of_cc_symbol(&self, cc_sym: &Self::ClassSymbol) -> Option<usize> {
        self.class_structure
            .as_ref()
            .expect("No class structure found.")
            .conjugacy_class_symbols
            .get_index_of(cc_sym)
    }

    #[must_use]
    fn get_cc_symbol_of_index(&self, cc_idx: usize) -> Option<Self::ClassSymbol> {
        self.class_structure
            .as_ref()
            .expect("No class structure found.")
            .conjugacy_class_symbols
            .get_index(cc_idx)
            .map(|(cc_sym, _)| cc_sym.clone())
    }

    #[must_use]
    fn filter_cc_symbols<P: FnMut(&Self::ClassSymbol) -> bool>(
        &self,
        predicate: P,
    ) -> Vec<Self::ClassSymbol> {
        self.class_structure
            .as_ref()
            .expect("No class structure found.")
            .conjugacy_class_symbols
            .keys()
            .cloned()
            .filter(predicate)
            .collect::<Vec<_>>()
    }

    fn set_class_symbols(&mut self, cc_symbols: &[Self::ClassSymbol]) {
        self.class_structure
            .as_mut()
            .unwrap()
            .set_class_symbols(cc_symbols);
    }

    #[must_use]
    fn get_inverse_cc(&self, cc_idx: usize) -> Option<usize> {
        self.class_structure
            .as_ref()
            .expect("No class structure found.")
            .inverse_conjugacy_classes
            .get(cc_idx)
            .cloned()
    }

    #[must_use]
    fn class_number(&self) -> usize {
        self.class_structure
            .as_ref()
            .expect("No class structure found.")
            .class_number()
    }

    #[must_use]
    fn class_size(&self, cc_idx: usize) -> Option<usize> {
        self.class_structure
            .as_ref()
            .expect("No class structure found.")
            .conjugacy_classes
            .get(cc_idx)
            .map(|cc| cc.len())
    }
}
