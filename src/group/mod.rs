use std::fmt;
use std::hash::Hash;
use std::ops::Mul;

use derive_builder::Builder;
use indexmap::IndexMap;
use log;
use ndarray::{Array2, Zip};
use num_traits::Pow;

use crate::chartab::{CorepCharacterTable, RepCharacterTable};
use crate::symmetry::symmetry_element::symmetry_operation::{
    FiniteOrder, SpecialSymmetryTransformation,
};
use crate::group::class::{ClassStructure, ClassProperties};

#[cfg(test)]
mod group_tests;

#[cfg(test)]
mod chartab_construction_tests;

mod class;

/// An enum to contain information about the type of a group.
#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub enum GroupType {
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
pub struct Group<T: Hash + Eq + Clone + Sync + fmt::Debug + FiniteOrder> {
    name: String,

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
        let mut group = Self::builder()
            .name(name.to_string())
            .elements(elements)
            .build()
            .expect("Unable to construct a group.");
        group.compute_cayley_table();
        group
    }

    /// Constructs the Cayley table for the group.
    fn compute_cayley_table(&mut self) {
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
        self.cayley_table = Some(ctb);
        log::debug!("Constructing Cayley table in parallel... Done.");
    }
}

pub trait GroupProperties
where Self::GroupElement: Hash + Eq + Clone + Sync + fmt::Debug + FiniteOrder,
{
    type GroupElement;

    fn abstract_group(&self) -> &Group<Self::GroupElement>;

    fn name(&self) -> &str;

    fn elements(&self) -> &IndexMap<Self::GroupElement, usize> {
        &self.abstract_group().elements
    }

    fn is_abelian(&self) -> bool {
        let ctb = self
            .abstract_group()
            .cayley_table
            .as_ref()
            .expect("Cayley table not found for this group.");
        ctb == ctb.t()
    }

    fn order(&self) -> usize {
        self.abstract_group().elements.len()
    }

    fn cayley_table(&self) -> &Array2<usize> {
        self.abstract_group().cayley_table.as_ref().expect("Cayley table not found for this group.")
    }
}

impl<T> GroupProperties for Group<T>
where T: Hash + Eq + Clone + Sync + fmt::Debug + FiniteOrder,
{
    type GroupElement = T;

    fn name(&self) -> &str {
        &self.name
    }

    fn abstract_group(&self) -> &Self {
        self
    }
}

#[derive(Clone, Builder)]
struct UnitaryRepresentedGroup<T: Hash + Eq + Clone + Sync + fmt::Debug + FiniteOrder> {
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

impl<T> UnitaryRepresentedGroupBuilder<T>
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

impl<T> UnitaryRepresentedGroup<T>
where
    T: Hash + Eq + Clone + Sync + Send + fmt::Debug + Pow<i32, Output = T> + FiniteOrder<Int = u32>,
    for<'a, 'b> &'b T: Mul<&'a T, Output = T>,
{
    /// Returns a builder to construct a new unitary group.
    ///
    /// # Returns
    ///
    /// A builder to construct a new group.
    fn builder() -> UnitaryRepresentedGroupBuilder<T> {
        UnitaryRepresentedGroupBuilder::default()
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
        let abstract_group = Group::<T>::new(name, elements);
        let mut unitary_group = UnitaryRepresentedGroup::<T>::builder()
            .name(name.to_string())
            .abstract_group(abstract_group)
            .build()
            .expect("Unable to construct a unitary group.");
        unitary_group.compute_class_structure();
        unitary_group
    }
}

impl<T> GroupProperties for UnitaryRepresentedGroup<T>
where T: Hash + Eq + Clone + Sync + fmt::Debug + FiniteOrder,
{
    type GroupElement = T;

    fn name(&self) -> &str {
        &self.name
    }

    fn abstract_group(&self) -> &Group<Self::GroupElement> {
        &self.abstract_group
    }
}

#[derive(Clone, Builder)]
struct MagneticRepresentedGroup<T: Hash + Eq + Clone + Sync + fmt::Debug + FiniteOrder + SpecialSymmetryTransformation> {
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
    pub ircorep_character_table: Option<CorepCharacterTable<T>>,
}

impl<T> MagneticRepresentedGroupBuilder<T>
where
    T: Hash + Eq + Clone + Sync + fmt::Debug + FiniteOrder + SpecialSymmetryTransformation,
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

impl<T> MagneticRepresentedGroup<T>
where
    T: Hash + Eq + Clone + Sync + Send + fmt::Debug + Pow<i32, Output = T> + FiniteOrder<Int = u32> + SpecialSymmetryTransformation,
    for<'a, 'b> &'b T: Mul<&'a T, Output = T>,
{
    /// Returns a builder to construct a new unitary group.
    ///
    /// # Returns
    ///
    /// A builder to construct a new group.
    fn builder() -> MagneticRepresentedGroupBuilder<T> {
        MagneticRepresentedGroupBuilder::default()
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
        let abstract_group = Group::<T>::new(name, elements);
        let mut magnetic_group = MagneticRepresentedGroup::<T>::builder()
            .name(name.to_string())
            .abstract_group(abstract_group)
            .build()
            .expect("Unable to construct a magnetic group.");
        magnetic_group.compute_class_structure();
        magnetic_group
    }
}

impl<T> GroupProperties for MagneticRepresentedGroup<T>
where T: Hash + Eq + Clone + Sync + fmt::Debug + FiniteOrder + SpecialSymmetryTransformation,
{
    type GroupElement = T;

    fn name(&self) -> &str {
        &self.name
    }

    fn abstract_group(&self) -> &Group<Self::GroupElement> {
        &self.abstract_group
    }
}

mod symmetry_group;
mod construct_chartab;
