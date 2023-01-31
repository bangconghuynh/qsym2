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

// use class::GroupStructure;

#[cfg(test)]
mod group_tests;

#[cfg(test)]
mod chartab_construction_tests;

mod class;

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
        let ctb = self.cayley_table.expect("Cayley table not found for this group.");
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
        let elements = self.elements;
        Zip::indexed(&mut ctb).par_for_each(|(i, j), k| {
            let (op_i_ref, _) = elements
                .get_index(i)
                .unwrap_or_else(|| panic!("Element with index {i} cannot be retrieved."));
            let (op_j_ref, _) = elements
                .get_index(j)
                .unwrap_or_else(|| panic!("Element with index {j} cannot be retrieved."));
            let op_k = op_i_ref * op_j_ref;
            *k = *elements
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




#[derive(Clone, Builder)]
struct UnitaryGroup<T: Hash + Eq + Clone + Sync + fmt::Debug + FiniteOrder> {
    /// A name for the group.
    name: String,

    /// An optional name if this group is actually a finite subgroup of [`Self::name`].
    #[builder(default = "None", setter(custom))]
    finite_subgroup_name: Option<String>,

    abstract_group: Group<T>,

    /// The character table for the irreducible representations of this group.
    #[builder(setter(skip), default = "None")]
    pub character_table: Option<RepCharacterTable<T>>,
}

impl<T: Hash + Eq + Clone + Sync + fmt::Debug + FiniteOrder> UnitaryGroupBuilder<T> {
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
    pub fn builder() -> UnitaryGroupBuilder<T> {
        UnitaryGroupBuilder::default()
    }

    ///// Constructs a unitary group from its elements.
    /////
    ///// # Arguments
    /////
    ///// * name - A name to be given to the group.
    ///// * elements - A vector of *all* group elements.
    /////
    ///// # Returns
    /////
    ///// A group with its Cayley table constructed and conjugacy classes determined.
    //fn new(name: &str, elements: Vec<T>) -> Self {
    //    let mut unitary_group = Self::builder()
    //        .name(name.to_string())
    //        .abstract_group(
    //            Group::builder()
    //                .elements(elements)
    //                .build()
    //                .expect("Unable to construct a group."),
    //        )
    //        .build()
    //        .expect("Unable to construct a unitary group.");
    //    unitary_group.abstract_group.cayley_table = Some(unitary_group.compute_cayley_table());
    //    let (ccs, e2ccs) = unitary_group.compute_conjugacy_classes();
    //    unitary_group.abstract_group.conjugacy_classes = Some(ccs);
    //    unitary_group.abstract_group.element_to_conjugacy_classes = Some(e2ccs);
    //    unitary_group.abstract_group.inverse_conjugacy_classes =
    //        Some(unitary_group.compute_inverse_conjugacy_classes());
    //    unitary_group.abstract_group.conjugacy_class_symbols = Some(unitary_group.compute_class_symbols());
    //    unitary_group.abstract_group.class_matrix = Some(unitary_group.compute_class_matrix());
    //    unitary_group
    //}
}

// impl<T> GroupStructure for UnitaryGroup<T>
// where
//     T: Hash + Eq + Clone + Sync + Send + fmt::Debug + Pow<i32, Output = T> + FiniteOrder<Int = u32>,
//     for<'a, 'b> &'b T: Mul<&'a T, Output = T>,
// {
//     type Element = T;

//     fn name(&self) -> &str {
//         &self.name
//     }

//     fn elements(&self) -> &IndexMap<T, usize> {
//         &self.abstract_group.elements
//     }

//     fn cayley_table(&self) -> &Array2<usize> {
//         self.abstract_group
//             .cayley_table
//             .as_ref()
//             .expect("Cayley table for this group not found.")
//     }

//     fn conjugacy_classes(&self) -> &Vec<HashSet<usize>> {
//         self.abstract_group
//             .conjugacy_classes
//             .as_ref()
//             .expect("Conjugacy class structure for this group not found.")
//     }

//     fn conjugacy_class_transversal(&self) -> &Vec<usize> {
//         self.abstract_group
//             .conjugacy_class_transversal
//             .as_ref()
//             .expect("Conjugacy class transversal for this group not found.")
//     }

//     fn conjugacy_class_symbols(&self) -> &IndexMap<ClassSymbol<T>, usize> {
//         self.abstract_group
//             .conjugacy_class_symbols
//             .as_ref()
//             .expect("Conjugacy class symbols not yet assigned for this group.")
//     }

//     fn inverse_conjugacy_classes(&self) -> &Vec<usize> {
//         self.abstract_group
//             .inverse_conjugacy_classes
//             .as_ref()
//             .expect("Conjugacy class inverses for this group not found.")
//     }

//     fn element_to_conjugacy_classes(&self) -> &Vec<usize> {
//         self.abstract_group
//             .element_to_conjugacy_classes
//             .as_ref()
//             .expect("Map from element to conjugacy class not found.")
//     }

//     fn class_matrix(&self) -> &Array3<usize> {
//         self.abstract_group
//             .class_matrix
//             .as_ref()
//             .expect("Class matrices for this group not found.")
//     }

//     fn compute_conjugacy_classes(&self) -> (Vec<HashSet<usize>>, Vec<usize>) {
//         // Find conjugacy classes
//         log::debug!("Finding conjugacy classes...");
//         let order = self.order();
//         if self.is_abelian() {
//             log::debug!("Abelian group found.");
//             log::debug!("Finding conjugacy classes... Done.");
//             // Abelian group; each element is in its own conjugacy class.
//             (
//                 (0usize..order).map(|i| HashSet::from([i])).collect(),
//                 (0usize..order).collect(),
//             )
//         } else {
//             // Non-Abelian group.
//             log::debug!("Non-Abelian group found.");
//             let mut ccs: Vec<HashSet<usize>> = vec![HashSet::from([0usize])];
//             let mut e2ccs = vec![0usize; order];
//             let mut remaining_elements: HashSet<usize> = (1usize..order).collect();
//             let ctb = self.cayley_table();

//             while !remaining_elements.is_empty() {
//                 // For a fixed g, find all h such that sg = hs for all s in the group.
//                 let g = *remaining_elements
//                     .iter()
//                     .next()
//                     .expect("Unexpected empty `remaining_elements`.");
//                 let mut cur_cc = HashSet::from([g]);
//                 for s in 0usize..order {
//                     let sg = ctb[[s, g]];
//                     let ctb_xs = ctb.slice(s![.., s]);
//                     let h = ctb_xs.iter().position(|&x| x == sg).unwrap_or_else(|| {
//                         panic!("No element `{sg}` can be found in column `{s}` of Cayley table.")
//                     });
//                     if remaining_elements.contains(&h) {
//                         remaining_elements.remove(&h);
//                         cur_cc.insert(h);
//                     }
//                 }
//                 ccs.push(cur_cc);
//             }
//             ccs.sort_by_key(|cc| {
//                 *cc.iter()
//                     .min()
//                     .expect("Unable to find the minimum element index in one conjugacy class.")
//             });
//             ccs.iter().enumerate().for_each(|(i, cc)| {
//                 cc.iter().for_each(|&j| e2ccs[j] = i);
//             });
//             assert!(e2ccs.iter().skip(1).all(|&x| x > 0));
//             log::debug!("Finding conjugacy classes... Done.");
//             (ccs, e2ccs)
//         }
//     }
// }

// mod construct_chartab;
// mod symmetry_group;
