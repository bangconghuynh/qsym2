use std::collections::HashSet;
use std::fmt;
use std::hash::Hash;
use std::ops::Mul;

use derive_builder::Builder;
use indexmap::IndexMap;
use itertools::Itertools;
use log;
use ndarray::{s, Array2, Array3, Axis, Zip};
use num_traits::Pow;

use crate::chartab::{CorepCharacterTable, RepCharacterTable};
use crate::symmetry::symmetry_element::symmetry_operation::{
    FiniteOrder, SpecialSymmetryTransformation,
};
use crate::symmetry::symmetry_symbols::ClassSymbol;

#[cfg(test)]
mod group_tests;

#[cfg(test)]
mod chartab_construction_tests;

/// An enum to contain information about the type of a group.
#[derive(Clone, Hash, PartialEq, Eq, Debug)]
enum GroupType {
    /// Variant for an ordinary group which contains no time-reversed operations.
    ///
    /// The associated boolean indicates whether this group is a double group or not.
    Ordinary(bool),

    /// Variant for a magnetic grey group which contains the time-reversal operation.
    ///
    /// The associated boolean indicates whether this group is a double group or not.
    MagneticGrey(bool),

    /// Variant for a magnetic black and white group which contains time-reversed operations, but
    /// not the time-reversal operation itself.
    ///
    /// The associated boolean indicates whether this group is a double group or not.
    MagneticBlackWhite(bool),
}

impl fmt::Display for GroupType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Ordinary(true) => write!(f, "Double ordinary group"),
            Self::Ordinary(false) => write!(f, "Ordinary group"),
            Self::MagneticGrey(true) => write!(f, "Double magnetic grey group"),
            Self::MagneticGrey(false) => write!(f, "Magnetic grey group"),
            Self::MagneticBlackWhite(true) => write!(f, "Double magnetic black-and-white group"),
            Self::MagneticBlackWhite(false) => write!(f, "Magnetic black-and-white group"),
        }
    }
}

const ORGRP: GroupType = GroupType::Ordinary(false);
const BWGRP: GroupType = GroupType::MagneticBlackWhite(false);
const GRGRP: GroupType = GroupType::MagneticGrey(false);

/// A struct for managing abstract groups.
#[derive(Builder, Clone)]
struct Group<T: Hash + Eq + Clone + Sync + fmt::Debug + FiniteOrder> {
    /// An ordered hash table containing the elements of the group. Each key is a group element,
    /// and the associated value is its index.
    #[builder(setter(custom))]
    elements: IndexMap<T, usize>,

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
}

impl<T: Hash + Eq + Clone + Sync + fmt::Debug + FiniteOrder> GroupBuilder<T> {
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
}

impl<T> Group<T>
where
    T: Hash + Eq + Clone + Sync + fmt::Debug + FiniteOrder,
    for<'a, 'b> &'b T: Mul<&'a T, Output = T>,
{
    /// Returns a builder to construct a new group.
    ///
    /// # Returns
    ///
    /// A builder to construct a new group.
    fn builder() -> GroupBuilder<T> {
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
        Self::builder()
            .elements(elements)
            .build()
            .expect("Unable to construct a group.")
    }

    /// Checks if this group is Abelian.
    ///
    /// This method requires the Cayley table to have been constructed.
    ///
    /// # Returns
    ///
    /// A flag indicating if this group is Abelian.
    fn is_abelian(&self) -> bool {
        let ctb = self
            .cayley_table
            .as_ref()
            .expect("Cayley table not found for this group.");
        ctb == ctb.t()
    }

    /// Determines the order of the group.
    #[must_use]
    fn order(&self) -> usize {
        self.elements.len()
    }

    /// Constructs the Cayley table for the group.
    #[must_use]
    fn compute_cayley_table(&self) -> Array2<usize> {
        log::debug!("Constructing Cayley table in parallel...");
        let mut ctb = Array2::<usize>::zeros((self.order(), self.order()));
        Zip::indexed(&mut ctb).par_for_each(|(i, j), k| {
            let (op_i_ref, _) = self.elements
                .get_index(i)
                .unwrap_or_else(|| panic!("Element with index {i} cannot be retrieved."));
            let (op_j_ref, _) = self.elements
                .get_index(j)
                .unwrap_or_else(|| panic!("Element with index {j} cannot be retrieved."));
            let op_k = op_i_ref * op_j_ref;
            *k = *self.elements
                .get(&op_k)
                .unwrap_or_else(|| panic!("Group closure not fulfilled. The composition {:?} * {:?} = {:?} is not contained in the group. Try changing thresholds.",
                        op_i_ref,
                        op_j_ref,
                        &op_k));
        });
        log::debug!("Constructing Cayley table in parallel... Done.");
        ctb
    }
}

#[derive(Builder, Clone)]
pub struct ClassStructure<T: Clone + Hash> {
    /// A vector of conjugacy classes for this group.
    ///
    /// Each element in the vector is a hashset containing the indices of the
    /// elements in [`Self::elements`] for a particular conjugacy class. This
    /// thus defines a multi-valued map from each conjugacy class index to one
    /// or more element indices.
    conjugacy_classes: Vec<HashSet<usize>>,

    /// The conjugacy class index of the elements in [`Self::elements`].
    ///
    /// This is the so-called inverse of [`Self::conjugacy_classes`]. This maps
    /// each element index to its corresponding conjugacy class index.
    element_to_conjugacy_classes: Vec<Option<usize>>,

    /// The conjugacy class representatives of the group.
    ///
    /// Each element in the vector is an index for a representative element of the corresponding
    /// conjugacy class.
    #[builder(setter(custom))]
    conjugacy_class_transversal: Vec<usize>,

    /// An index map of symbols for the conjugacy classes in this group.
    ///
    /// Each key in the index map is a class symbol, and the associated value is the index of
    /// the corresponding conjugacy class in [`Self::conjugacy_classes`].
    #[builder(setter(custom))]
    conjugacy_class_symbols: IndexMap<ClassSymbol<T>, usize>,

    /// A vector containing the indices of inverse conjugacy classes.
    ///
    /// Each index gives the inverse conjugacy class for the corresponding
    /// conjugacy class.
    #[builder(setter(custom))]
    inverse_conjugacy_classes: Vec<usize>,

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
    #[builder(setter(custom))]
    class_matrix: Array3<usize>,
}

impl<T> ClassStructureBuilder<T>
where
    T: Clone + Hash,
{
    fn conjugacy_class_transversal(&mut self) -> &mut Self {
        self.conjugacy_class_transversal = Some(
            self.conjugacy_classes
                .as_ref()
                .expect("Conjugacy classes have not been found.")
                .iter()
                .map(|cc| {
                    *cc.iter()
                        .next()
                        .expect("No conjugacy classes can be empty.")
                })
                .collect::<Vec<usize>>(),
        );
        self
    }

    fn conjugacy_class_symbols(&mut self, elements: &IndexMap<T, usize>) -> &mut Self {
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
                let (rep_ele, _) = elements.get_index(rep_ele_index).unwrap_or_else(|| {
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
    fn class_matrix(&mut self, ctb: &Array2<usize>) -> &mut Self {
        let class_number = self
            .conjugacy_classes
            .as_ref()
            .expect("Conjugacy classes have not been found.")
            .len();
        let mut nmat = Array3::<usize>::zeros((class_number, class_number, class_number));
        for (r, class_r) in self
            .conjugacy_classes
            .as_ref()
            .expect("Conjugacy classes have not been found.")
            .iter()
            .enumerate()
        {
            let idx_r = class_r.iter().copied().collect::<Vec<_>>();
            for (s, class_s) in self
                .conjugacy_classes
                .as_ref()
                .expect("Conjugacy classes have not been found.")
                .iter()
                .enumerate()
            {
                let idx_s = class_s.iter().copied().collect::<Vec<_>>();
                let cayley_block_rs = ctb
                    .select(Axis(0), &idx_r)
                    .select(Axis(1), &idx_s)
                    .iter()
                    .copied()
                    .counts();

                for (t, class_t) in self
                    .conjugacy_classes
                    .as_ref()
                    .expect("Conjugacy classes have not been found.")
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
        self
    }
}

impl<T> ClassStructure<T>
where
    T: Hash + Eq + Clone + Sync + fmt::Debug + FiniteOrder,
{
    /// Returns a builder to construct a new class structure.
    ///
    /// # Returns
    ///
    /// A builder to construct a new class structure.
    fn builder() -> ClassStructureBuilder<T> {
        ClassStructureBuilder::default()
    }

    fn new(
        group: &Group<T>,
        conjugacy_classes: Vec<HashSet<usize>>,
        element_to_conjugacy_classes: Vec<Option<usize>>,
    ) -> Self {
        let ctb = group
            .cayley_table
            .as_ref()
            .expect("Cayley table not found for this group.");
        Self::builder()
            .conjugacy_classes(conjugacy_classes)
            .element_to_conjugacy_classes(element_to_conjugacy_classes)
            .conjugacy_class_transversal()
            .conjugacy_class_symbols(&group.elements)
            .inverse_conjugacy_classes(ctb)
            .class_matrix(ctb)
            .build()
            .expect("Unable to construct a `ClassStructure`.")
    }
}

impl<T> ClassStructure<T>
where
    T: Clone + Hash,
{
    #[must_use]
    fn class_number(&self) -> usize {
        self.conjugacy_classes.len()
    }
}

pub trait ClassAnalysed
where
    Self::Element: Clone + Hash,
{
    type Element;

    #[must_use]
    fn class_structure(&self) -> &ClassStructure<Self::Element>;

    fn compute_class_structure(&mut self);

    #[must_use]
    fn conjugacy_classes(&self) -> &Vec<HashSet<usize>> {
        &self.class_structure().conjugacy_classes
    }

    #[must_use]
    fn element_to_conjugacy_classes(&self) -> &Vec<Option<usize>> {
        &self.class_structure().element_to_conjugacy_classes
    }

    #[must_use]
    fn conjugacy_class_transversal(&self) -> &Vec<usize> {
        &self.class_structure().conjugacy_class_transversal
    }

    #[must_use]
    fn conjugacy_class_symbols(&self) -> &IndexMap<ClassSymbol<Self::Element>, usize> {
        &self.class_structure().conjugacy_class_symbols
    }

    #[must_use]
    fn inverse_conjugacy_classes(&self) -> &Vec<usize> {
        &self.class_structure().inverse_conjugacy_classes
    }

    #[must_use]
    fn class_matrix(&self) -> &Array3<usize> {
        &self.class_structure().class_matrix
    }

    #[must_use]
    fn class_number(&self) -> usize {
        self.class_structure().class_number()
    }
}

#[derive(Clone, Builder)]
struct UnitaryGroup<T: Hash + Eq + Clone + Sync + fmt::Debug + FiniteOrder> {
    /// A name for the group.
    name: String,

    /// An optional name if this group is actually a finite subgroup of [`Self::name`].
    #[builder(setter(custom), default = "None")]
    finite_subgroup_name: Option<String>,

    abstract_group: Group<T>,

    #[builder(setter(skip), default = "None")]
    class_structure: Option<ClassStructure<T>>,

    /// The character table for the irreducible representations of this group.
    #[builder(setter(skip), default = "None")]
    pub irrep_character_table: Option<RepCharacterTable<T>>,
}

impl<T> UnitaryGroupBuilder<T>
where
    T: Hash + Eq + Clone + Sync + fmt::Debug + FiniteOrder,
{
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

impl<T> UnitaryGroup<T>
where
    T: Hash + Eq + Clone + Sync + Send + fmt::Debug + Pow<i32, Output = T> + FiniteOrder<Int = u32>,
    for<'a, 'b> &'b T: Mul<&'a T, Output = T>,
{
    /// Returns a builder to construct a new unitary group.
    ///
    /// # Returns
    ///
    /// A builder to construct a new group.
    fn builder() -> UnitaryGroupBuilder<T> {
        UnitaryGroupBuilder::default()
    }

    /// Constructs a unitary group from its elements.
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
        let abstract_group = Group::<T>::builder()
            .elements(elements)
            .build()
            .expect("Unable to construct a group.");
        let mut unitary_group = UnitaryGroup::<T>::builder()
            .name(name.to_string())
            .abstract_group(abstract_group)
            .build()
            .expect("Unable to construct a unitary group.");
        unitary_group.compute_class_structure();
        unitary_group
    }
}

impl<T> ClassAnalysed for UnitaryGroup<T>
where
    T: Hash + Eq + Clone + Sync + fmt::Debug + FiniteOrder,
    for<'a, 'b> &'b T: Mul<&'a T, Output = T>,
{
    type Element = T;

    fn class_structure(&self) -> &ClassStructure<Self::Element> {
        self.class_structure
            .as_ref()
            .expect("Class structure not found for this group.")
    }

    fn compute_class_structure(&mut self) {
        log::debug!("Finding unitary conjugacy classes...");
        let order = self.abstract_group.order();
        let (ccs, e2ccs) = if self.abstract_group.is_abelian() {
            log::debug!("Abelian group found.");
            // Abelian group; each element is in its own conjugacy class.
            (
                (0usize..order)
                    .map(|i| HashSet::from([i]))
                    .collect::<Vec<_>>(),
                (0usize..order).map(|i| Some(i)).collect::<Vec<_>>(),
            )
        } else {
            // Non-Abelian group.
            log::debug!("Non-Abelian group found.");
            let mut ccs: Vec<HashSet<usize>> = vec![HashSet::from([0usize])];
            let mut e2ccs = vec![0usize; order];
            let mut remaining_elements: HashSet<usize> = (1usize..order).collect();
            let ctb = self
                .abstract_group
                .cayley_table
                .as_ref()
                .expect("Cayley table not found.");

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
            assert!(e2ccs.iter().skip(1).all(|&x| x > 0));
            (ccs, e2ccs.iter().map(|&i| Some(i)).collect::<Vec<_>>())
        };
        log::debug!("Finding unitary conjugacy classes... Done.");

        let class_structure = ClassStructure::new(&self.abstract_group, ccs, e2ccs);
        self.class_structure = Some(class_structure);
    }
}

// mod construct_chartab;
mod symmetry_group;
