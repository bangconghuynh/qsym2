use std::fmt;
use std::hash::Hash;
use std::ops::Mul;

use derive_builder::Builder;
use indexmap::{IndexSet, IndexMap};
use log;
use ndarray::{Array2, Zip};
use num::Integer;
use num_traits::Inv;

use crate::chartab::chartab_group::CharacterProperties;
use crate::chartab::chartab_symbols::{
    CollectionSymbol, LinearSpaceSymbol, ReducibleLinearSpaceSymbol,
};
use crate::chartab::{CharacterTable, CorepCharacterTable, RepCharacterTable};
use crate::group::class::{ClassProperties, ClassStructure};

pub mod class;

// =================
// Trait definitions
// =================

/// A trait for order finiteness.
pub trait FiniteOrder {
    /// The integer type for the order of the element.
    type Int: Integer;

    /// Calculates the finite order.
    fn order(&self) -> Self::Int;
}

/// A trait for purely group-theoretic properties.
pub trait GroupProperties
where
    Self::GroupElement: Mul<Output = Self::GroupElement>
        + Inv<Output = Self::GroupElement>
        + Hash
        + Eq
        + Clone
        + Sync
        + fmt::Debug
        + FiniteOrder,
{
    /// The type of the elements in the group.
    type GroupElement;

    /// The underlying abstract group of the possibly concrete group.
    fn abstract_group(&self) -> &Group<Self::GroupElement>;

    /// The name of the group.
    fn name(&self) -> String;

    /// The finite subgroup name of this group, if any.
    fn finite_subgroup_name(&self) -> Option<&String>;

    /// The elements in the group.
    fn elements(&self) -> &IndexSet<Self::GroupElement> {
        &self.abstract_group().elements
    }

    /// Checks if this group is abelian.
    fn is_abelian(&self) -> bool {
        let ctb = self
            .abstract_group()
            .cayley_table
            .as_ref()
            .expect("Cayley table not found for this group.");
        ctb == ctb.t()
    }

    /// The order of the group.
    fn order(&self) -> usize {
        self.abstract_group().elements.len()
    }

    /// The Cayley table of the group.
    fn cayley_table(&self) -> Option<&Array2<usize>> {
        self.abstract_group().cayley_table.as_ref()
    }
}

/// A trait for indicating that a group can be partitioned into a unitary halving subgroup and and
/// antiunitary coset.
pub trait HasUnitarySubgroup: GroupProperties
where
    Self::UnitarySubgroup: GroupProperties<GroupElement = Self::GroupElement> + CharacterProperties,
{
    /// The type of the unitary halving subgroup.
    type UnitarySubgroup;

    /// Returns a shared reference to the unitary subgroup associated with this group.
    fn unitary_subgroup(&self) -> &Self::UnitarySubgroup;

    /// Checks if an element in the group belongs to the antiunitary coset.
    ///
    /// # Arguments
    ///
    /// * `element` - A group element.
    ///
    /// # Returns
    ///
    /// Returns `true` if `element` is in the antiunitary coset of the group.
    ///
    /// # Panics
    ///
    /// Panics if `element` is not a member of the group.
    fn check_elem_antiunitary(&self, element: &Self::GroupElement) -> bool;
}

// ====================================
// Enum definitions and implementations
// ====================================

/// An enumerated type to contain information about the type of a group.
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

pub const ORGRP: GroupType = GroupType::Ordinary(false);
pub const BWGRP: GroupType = GroupType::MagneticBlackWhite(false);
pub const GRGRP: GroupType = GroupType::MagneticGrey(false);

// ======================================
// Struct definitions and implementations
// ======================================

// --------------
// Abstract group
// --------------

/// A structure for managing abstract groups.
#[derive(Builder, Clone)]
pub struct Group<T>
where
    T: Mul<Output = T> + Inv<Output = T> + Hash + Eq + Clone + Sync + fmt::Debug + FiniteOrder,
{
    /// A name for the abstract group.
    name: String,

    /// An ordered hash table containing the elements of the group. Each key is a group element,
    /// and the associated value is its index.
    #[builder(setter(custom))]
    elements: IndexSet<T>,

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

impl<T> GroupBuilder<T>
where
    T: Mul<Output = T> + Inv<Output = T> + Hash + Eq + Clone + Sync + fmt::Debug + FiniteOrder,
    for<'a, 'b> &'b T: Mul<&'a T, Output = T>,
{
    fn elements(&mut self, elems: Vec<T>) -> &mut Self {
        self.elements = Some(
            elems
                .into_iter()
                .collect(),
        );
        self
    }
}

impl<T> Group<T>
where
    T: Mul<Output = T> + Inv<Output = T> + Hash + Eq + Clone + Sync + fmt::Debug + FiniteOrder,
    for<'a, 'b> &'b T: Mul<&'a T, Output = T>,
{
    /// Returns a builder to construct a new abstract group.
    ///
    /// # Returns
    ///
    /// A builder to construct a new abstract group.
    fn builder() -> GroupBuilder<T> {
        GroupBuilder::<T>::default()
    }

    /// Constructs an abstract group from its elements and calculate its Cayley table.
    ///
    /// # Arguments
    ///
    /// * `name` - A name to be given to the abstract group.
    /// * `elements` - A vector of *all* group elements.
    ///
    /// # Returns
    ///
    /// An abstract group with its Cayley table constructed.
    #[must_use]
    pub fn new(name: &str, elements: Vec<T>) -> Self {
        let mut group = Self::builder()
            .name(name.to_string())
            .elements(elements)
            .build()
            .expect("Unable to construct a group.");
        group.compute_cayley_table();
        group
    }

    /// Constructs an abstract group from its elements but without calculating its Cayley table.
    ///
    /// # Arguments
    ///
    /// * `name` - A name to be given to the abstract group.
    /// * `elements` - A vector of *all* group elements.
    ///
    /// # Returns
    ///
    /// An abstract group without its Cayley table constructed.
    pub fn new_no_ctb(name: &str, elements: Vec<T>) -> Self {
        Self::builder()
            .name(name.to_string())
            .elements(elements)
            .build()
            .expect("Unable to construct a group.")
    }

    /// Constructs the Cayley table for the abstract group.
    fn compute_cayley_table(&mut self) {
        log::debug!("Constructing Cayley table in parallel...");
        let mut ctb = Array2::<usize>::zeros((self.order(), self.order()));
        Zip::indexed(&mut ctb).par_for_each(|(i, j), k| {
            let op_i_ref = self.elements
                .get_index(i)
                .unwrap_or_else(|| panic!("Element with index {i} cannot be retrieved."));
            let op_j_ref = self.elements
                .get_index(j)
                .unwrap_or_else(|| panic!("Element with index {j} cannot be retrieved."));
            let op_k = op_i_ref * op_j_ref;
            *k = self.elements
                .get_index_of(&op_k)
                .unwrap_or_else(||
                    panic!("Group closure not fulfilled. The composition {:?} * {:?} = {:?} is not contained in the group. Try changing thresholds.",
                        op_i_ref,
                        op_j_ref,
                        &op_k)
                    );
        });
        self.cayley_table = Some(ctb);
        log::debug!("Constructing Cayley table in parallel... Done.");
    }
}

impl<T> GroupProperties for Group<T>
where
    T: Mul<Output = T> + Inv<Output = T> + Hash + Eq + Clone + Sync + fmt::Debug + FiniteOrder,
    for<'a, 'b> &'b T: Mul<&'a T, Output = T>,
{
    type GroupElement = T;

    fn name(&self) -> String {
        self.name.to_string()
    }

    fn finite_subgroup_name(&self) -> Option<&String> {
        None
    }

    fn abstract_group(&self) -> &Self {
        self
    }
}

// -------------------------
// Unitary-represented group
// -------------------------

/// A structure for managing groups with unitary representations.
#[derive(Clone, Builder)]
pub struct UnitaryRepresentedGroup<T, RowSymbol, ColSymbol>
where
    T: Mul<Output = T> + Inv<Output = T> + Hash + Eq + Clone + Sync + fmt::Debug + FiniteOrder,
    RowSymbol: LinearSpaceSymbol,
    ColSymbol: CollectionSymbol<CollectionElement = T>,
{
    /// A name for the unitary-represented group.
    name: String,

    /// An optional name if this unitary group is actually a finite subgroup of an infinite
    /// group [`Self::name`].
    #[builder(setter(custom), default = "None")]
    finite_subgroup_name: Option<String>,

    /// The underlying abstract group of this unitary-represented group.
    abstract_group: Group<T>,

    /// The class structure of this unitary-represented group that is induced by the following
    /// equivalence relation:
    ///
    /// ```math
    ///     g \sim h \Leftrightarrow \exists u : h = u g u ^{-1}.
    /// ```
    #[builder(setter(skip), default = "None")]
    class_structure: Option<ClassStructure<T, ColSymbol>>,

    /// The character table for the irreducible representations of this group.
    #[builder(setter(skip), default = "None")]
    pub irrep_character_table: Option<RepCharacterTable<RowSymbol, ColSymbol>>,
}

impl<T, RowSymbol, ColSymbol> UnitaryRepresentedGroupBuilder<T, RowSymbol, ColSymbol>
where
    T: Mul<Output = T> + Inv<Output = T> + Hash + Eq + Clone + Sync + fmt::Debug + FiniteOrder,
    for<'a, 'b> &'b T: Mul<&'a T, Output = T>,
    RowSymbol: LinearSpaceSymbol,
    ColSymbol: CollectionSymbol<CollectionElement = T>,
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

impl<T, RowSymbol, ColSymbol> UnitaryRepresentedGroup<T, RowSymbol, ColSymbol>
where
    T: Mul<Output = T> + Inv<Output = T> + Hash + Eq + Clone + Sync + fmt::Debug + FiniteOrder,
    RowSymbol: LinearSpaceSymbol,
    ColSymbol: CollectionSymbol<CollectionElement = T>,
{
    /// Sets the finite subgroup name of this group.
    ///
    /// # Arguments
    ///
    /// * `name` - A name to be set as the finite subgroup name of this group.
    pub fn set_finite_subgroup_name(&mut self, name: Option<String>) {
        self.finite_subgroup_name = name;
    }
}

impl<T, RowSymbol, ColSymbol> UnitaryRepresentedGroup<T, RowSymbol, ColSymbol>
where
    T: Mul<Output = T> + Inv<Output = T> + Hash + Eq + Clone + Sync + fmt::Debug + FiniteOrder,
    for<'a, 'b> &'b T: Mul<&'a T, Output = T>,
    RowSymbol: LinearSpaceSymbol,
    ColSymbol: CollectionSymbol<CollectionElement = T>,
{
    /// Returns a builder to construct a new unitary-represented group.
    ///
    /// # Returns
    ///
    /// A builder to construct a new unitary-represented group.
    fn builder() -> UnitaryRepresentedGroupBuilder<T, RowSymbol, ColSymbol> {
        UnitaryRepresentedGroupBuilder::<T, RowSymbol, ColSymbol>::default()
    }

    /// Constructs a unitary-represented group from its elements.
    ///
    /// # Arguments
    ///
    /// * `name` - A name to be given to the unitary-represented group.
    /// * `elements` - A vector of *all* group elements.
    ///
    /// # Returns
    ///
    /// A unitary-represented group with its Cayley table constructed and conjugacy classes
    /// determined.
    #[must_use]
    pub fn new(name: &str, elements: Vec<T>) -> Self {
        let abstract_group = Group::<T>::new(name, elements);
        let mut unitary_group = UnitaryRepresentedGroup::<T, RowSymbol, ColSymbol>::builder()
            .name(name.to_string())
            .abstract_group(abstract_group)
            .build()
            .expect("Unable to construct a unitary group.");
        unitary_group.compute_class_structure();
        unitary_group
    }
}

impl<T, RowSymbol, ColSymbol> GroupProperties for UnitaryRepresentedGroup<T, RowSymbol, ColSymbol>
where
    T: Mul<Output = T> + Inv<Output = T> + Hash + Eq + Clone + Sync + fmt::Debug + FiniteOrder,
    for<'a, 'b> &'b T: Mul<&'a T, Output = T>,
    RowSymbol: LinearSpaceSymbol,
    ColSymbol: CollectionSymbol<CollectionElement = T>,
{
    type GroupElement = T;

    fn name(&self) -> String {
        self.name.to_string()
    }

    fn finite_subgroup_name(&self) -> Option<&String> {
        self.finite_subgroup_name.as_ref()
    }

    fn abstract_group(&self) -> &Group<Self::GroupElement> {
        &self.abstract_group
    }
}

// --------------------------
// Magnetic-represented group
// --------------------------

/// A structure for managing groups with magnetic corepresentations. Such a group consists of two
/// types of elements in equal numbers: those that are unitary represented and those that are
/// antiunitary represented. This division of elements affects the class structure of the group via
/// an equivalence relation defined in Newmarch, J. D. & Golding, R. M. The character table for the
/// corepresentations of magnetic groups. *Journal of Mathematical Physics* **23**, 695–704 (1982).
#[derive(Clone, Builder)]
pub struct MagneticRepresentedGroup<T, UG, RowSymbol>
where
    T: Mul<Output = T> + Inv<Output = T> + Hash + Eq + Clone + Sync + fmt::Debug + FiniteOrder,
    RowSymbol: ReducibleLinearSpaceSymbol<Subspace = UG::RowSymbol>,
    UG: Clone + GroupProperties<GroupElement = T> + CharacterProperties,
{
    /// A name for the magnetic-represented group.
    name: String,

    /// An optional name if this unitary group is actually a finite subgroup of an infinite
    /// group [`Self::name`].
    #[builder(setter(custom), default = "None")]
    finite_subgroup_name: Option<String>,

    /// The underlying abstract group of this magnetic-represented group.
    abstract_group: Group<T>,

    /// The subgroup consisting of the unitary-represented elements in the full group.
    #[builder(setter(custom))]
    unitary_subgroup: UG,

    /// The class structure of this magnetic-represented group that is induced by the following
    /// equivalence relation:
    ///
    /// ```math
    ///     g \sim h \Leftrightarrow \exists u : h = u g u^{-1} \quad \textrm{or} \quad \exists a : h = a
    ///     g^{-1} a^{-1},
    /// ```
    ///
    /// where $`u`$ is unitary-represented (*i.e.* $`u`$ is in [`Self::unitary_subgroup`]) and
    /// $`a`$ is antiunitary-represented (*i.e.* $`a`$ is not in [`Self::unitary_subgroup`]).
    #[builder(setter(skip), default = "None")]
    class_structure: Option<
        ClassStructure<T, <<UG as CharacterProperties>::CharTab as CharacterTable>::ColSymbol>,
    >,

    /// The character table for the irreducible corepresentations of this group.
    #[builder(setter(skip), default = "None")]
    pub ircorep_character_table: Option<CorepCharacterTable<RowSymbol, UG::CharTab>>,
}

impl<T, UG, RowSymbol> MagneticRepresentedGroupBuilder<T, UG, RowSymbol>
where
    T: Mul<Output = T> + Inv<Output = T> + Hash + Eq + Clone + Sync + fmt::Debug + FiniteOrder,
    for<'a, 'b> &'b T: Mul<&'a T, Output = T>,
    RowSymbol: ReducibleLinearSpaceSymbol<Subspace = UG::RowSymbol>,
    UG: Clone + GroupProperties<GroupElement = T> + CharacterProperties,
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

    fn unitary_subgroup(&mut self, uni_subgrp: UG) -> &mut Self {
        assert!(uni_subgrp.elements().iter().all(|op| self
            .abstract_group
            .as_ref()
            .expect("Abstract group not yet set for this magnetic-represented group.")
            .elements()
            .contains(op)));
        self.unitary_subgroup = Some(uni_subgrp);
        self
    }
}

impl<T, UG, RowSymbol> MagneticRepresentedGroup<T, UG, RowSymbol>
where
    T: Mul<Output = T> + Inv<Output = T> + Hash + Eq + Clone + Sync + fmt::Debug + FiniteOrder,
    RowSymbol: ReducibleLinearSpaceSymbol<Subspace = UG::RowSymbol>,
    UG: Clone + GroupProperties<GroupElement = T> + CharacterProperties,
{
    /// Sets the finite subgroup name of this group.
    ///
    /// # Arguments
    ///
    /// * `name` - A name to be set as the finite subgroup name of this group.
    pub fn set_finite_subgroup_name(&mut self, name: Option<String>) {
        self.finite_subgroup_name = name;
    }

    /// Returns a shared reference to the unitary subgroup of this group.
    pub fn unitary_subgroup(&self) -> &UG {
        &self.unitary_subgroup
    }
}

impl<T, UG, RowSymbol> MagneticRepresentedGroup<T, UG, RowSymbol>
where
    T: Mul<Output = T> + Inv<Output = T> + Hash + Eq + Clone + Sync + fmt::Debug + FiniteOrder,
    for<'a, 'b> &'b T: Mul<&'a T, Output = T>,
    RowSymbol: ReducibleLinearSpaceSymbol<Subspace = UG::RowSymbol>,
    UG: Clone + GroupProperties<GroupElement = T> + CharacterProperties,
{
    /// Returns a builder to construct a new magnetic-represented group.
    ///
    /// # Returns
    ///
    /// A builder to construct a new magnetic-represented group.
    fn builder() -> MagneticRepresentedGroupBuilder<T, UG, RowSymbol> {
        MagneticRepresentedGroupBuilder::<T, UG, RowSymbol>::default()
    }

    /// Constructs a magnetic-represented group from its elements and the unitary subgroup.
    ///
    /// # Arguments
    ///
    /// * `name` - A name to be given to the magnetic-reprented group.
    /// * `elements` - A vector of *all* group elements.
    /// * `unitary_subgroup` - The unitary subgroup of the magnetic-represented group. All elements
    /// of this must be present in `elements`.
    ///
    /// # Returns
    ///
    /// A magnetic-represented group with its Cayley table constructed and conjugacy classes
    /// determined.
    pub fn new(name: &str, elements: Vec<T>, unitary_subgroup: UG) -> Self {
        let abstract_group = Group::<T>::new(name, elements);
        let mut magnetic_group = MagneticRepresentedGroup::<T, UG, RowSymbol>::builder()
            .name(name.to_string())
            .abstract_group(abstract_group)
            .unitary_subgroup(unitary_subgroup)
            .build()
            .expect("Unable to construct a magnetic group.");
        magnetic_group.compute_class_structure();
        magnetic_group
    }
}

impl<T, UG, RowSymbol> GroupProperties for MagneticRepresentedGroup<T, UG, RowSymbol>
where
    T: Mul<Output = T> + Inv<Output = T> + Hash + Eq + Clone + Sync + fmt::Debug + FiniteOrder,
    for<'a, 'b> &'b T: Mul<&'a T, Output = T>,
    RowSymbol: ReducibleLinearSpaceSymbol<Subspace = UG::RowSymbol>,
    UG: Clone + GroupProperties<GroupElement = T> + CharacterProperties,
{
    type GroupElement = T;

    fn name(&self) -> String {
        self.name.to_string()
    }

    fn finite_subgroup_name(&self) -> Option<&String> {
        self.finite_subgroup_name.as_ref()
    }

    fn abstract_group(&self) -> &Group<Self::GroupElement> {
        &self.abstract_group
    }
}

impl<T, UG, RowSymbol> HasUnitarySubgroup for MagneticRepresentedGroup<T, UG, RowSymbol>
where
    T: Mul<Output = T> + Inv<Output = T> + Hash + Eq + Clone + Sync + fmt::Debug + FiniteOrder,
    for<'a, 'b> &'b T: Mul<&'a T, Output = T>,
    RowSymbol: ReducibleLinearSpaceSymbol<Subspace = UG::RowSymbol>,
    UG: Clone + GroupProperties<GroupElement = T> + CharacterProperties,
{
    type UnitarySubgroup = UG;

    fn unitary_subgroup(&self) -> &Self::UnitarySubgroup {
        &self.unitary_subgroup
    }

    /// Checks if a given element is antiunitary-represented in this group.
    ///
    /// # Arguments
    ///
    /// `element` - A reference to an element to be checked.
    ///
    /// # Returns
    ///
    /// Returns `true` if `element` is antiunitary-represented in this group.
    ///
    /// # Panics
    ///
    /// Panics if `element` is not in the group.
    fn check_elem_antiunitary(&self, element: &T) -> bool {
        if self.abstract_group.elements().contains(element) {
            !self.unitary_subgroup.elements().contains(element)
        } else {
            panic!("`{element:?}` is not an element of the group.")
        }
    }
}
