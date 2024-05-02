//! Basis for non-orthogonal configuration interaction of Slater determinants.

use anyhow;
use derive_builder::Builder;
use itertools::structs::Product;
use itertools::Itertools;

use crate::group::GroupProperties;

#[path = "basis_transformation.rs"]
mod basis_transformation;

// =====
// Basis
// =====

// -----------------
// Trait definitions
// -----------------

/// Trait defining behaviours of a basis consisting of linear-space items.
pub trait Basis<I> {
    /// Type of the iterator over items in the basis.
    type BasisIter: Iterator<Item = Result<I, anyhow::Error>>;

    /// Returns the number of items in the basis.
    fn n_items(&self) -> usize;

    /// An iterator over items in the basis.
    fn iter(&self) -> Self::BasisIter;

    /// Shared reference to the first item in the basis.
    fn first(&self) -> Option<&I>;
}

// --------------------------------------
// Struct definitions and implementations
// --------------------------------------

// ~~~~~~~~~~~~~~~~~~~~~~
// Lazy basis from orbits
// ~~~~~~~~~~~~~~~~~~~~~~

#[derive(Builder, Clone)]
pub(crate) struct OrbitBasis<'g, G, I>
where
    G: GroupProperties,
{
    /// The origins from which orbits are generated.
    origins: Vec<I>,

    /// The group acting on the origins to generate orbits, the concatenation of which forms the
    /// basis.
    group: &'g G,

    /// A function defining the action of each group element on the origin.
    action: fn(&G::GroupElement, &I) -> Result<I, anyhow::Error>,
}

impl<'g, G, I> OrbitBasis<'g, G, I>
where
    G: GroupProperties + Clone,
    I: Clone,
{
    pub(crate) fn builder() -> OrbitBasisBuilder<'g, G, I> {
        OrbitBasisBuilder::<G, I>::default()
    }

    /// The origins from which orbits are generated.
    pub fn origins(&self) -> &Vec<I> {
        &self.origins
    }

    /// The group acting on the origins to generate orbits, the concatenation of which forms the
    /// basis.
    pub fn group(&self) -> &G {
        self.group
    }
}

impl<'g, G, I> Basis<I> for OrbitBasis<'g, G, I>
where
    G: GroupProperties,
    I: Clone,
{
    type BasisIter = OrbitBasisIterator<G, I>;

    fn n_items(&self) -> usize {
        self.origins.len() * self.group.order()
    }

    fn iter(&self) -> Self::BasisIter {
        OrbitBasisIterator::new(self.group, self.origins.clone(), self.action)
    }

    fn first(&self) -> Option<&I> {
        self.origins.get(0)
    }
}

/// Lazy iterator for basis constructed from the concatenation of orbits generated from multiple
/// origins.
pub struct OrbitBasisIterator<G, I>
where
    G: GroupProperties,
{
    /// A mutable iterator over the Cartesian product between the group elements and the origins.
    group_origin_iter: Product<
        <<G as GroupProperties>::ElementCollection as IntoIterator>::IntoIter,
        std::vec::IntoIter<I>,
    >,

    /// A function defining the action of each group element on the origin.
    action: fn(&G::GroupElement, &I) -> Result<I, anyhow::Error>,
}

impl<G, I> OrbitBasisIterator<G, I>
where
    G: GroupProperties,
    I: Clone,
{
    /// Creates and returns a new orbit basis iterator.
    ///
    /// # Arguments
    ///
    /// * `group` - A group.
    /// * `origins` - A slice of origins.
    /// * `action` - A function or closure defining the action of each group element on the origins.
    ///
    /// # Returns
    ///
    /// An orbit basis iterator.
    fn new(
        group: &G,
        origins: Vec<I>,
        action: fn(&G::GroupElement, &I) -> Result<I, anyhow::Error>,
    ) -> Self {
        Self {
            group_origin_iter: group
                .elements()
                .clone()
                .into_iter()
                .cartesian_product(origins.into_iter()),
            action,
        }
    }
}

impl<G, I> Iterator for OrbitBasisIterator<G, I>
where
    G: GroupProperties,
    I: Clone,
{
    type Item = Result<I, anyhow::Error>;

    fn next(&mut self) -> Option<Self::Item> {
        self.group_origin_iter
            .next()
            .map(|(op, origin)| (self.action)(&op, &origin))
    }
}

// ~~~~~~~~~~~
// Eager basis
// ~~~~~~~~~~~

#[derive(Builder, Clone)]
pub(crate) struct EagerBasis<I: Clone> {
    /// The elements in the basis.
    elements: Vec<I>,
}

impl<I: Clone> EagerBasis<I> {
    fn builder() -> EagerBasisBuilder<I> {
        EagerBasisBuilder::default()
    }
}

impl<I: Clone> Basis<I> for EagerBasis<I> {
    type BasisIter = std::vec::IntoIter<Result<I, anyhow::Error>>;

    fn n_items(&self) -> usize {
        self.elements.len()
    }

    fn iter(&self) -> Self::BasisIter {
        self.elements
            .iter()
            .cloned()
            .map(Ok)
            .collect_vec()
            .into_iter()
    }

    fn first(&self) -> Option<&I> {
        self.elements.get(0)
    }
}
