//! Basis for non-orthogonal configuration interaction of Slater determinants.

use anyhow;
use derive_builder::Builder;
use itertools::structs::Product;
use itertools::Itertools;

use crate::group::GroupProperties;

// =====
// Basis
// =====

// -----------------
// Trait definitions
// -----------------

/// Trait defining behaviours of a basis consisting of linear-space items.
pub(crate) trait Basis<I> {
    /// Type of the iterator over items in the basis.
    type BasisIter: Iterator<Item = Result<I, anyhow::Error>>;

    /// Returns the number of items in the basis.
    fn n_items(&self) -> usize;

    /// An iterator over items in the basis.
    fn iter(&self) -> Self::BasisIter;
}

// --------------------------------------
// Struct definitions and implementations
// --------------------------------------

// ~~~~~~~~~~~~~~~~~~~~~~
// Lazy basis from orbits
// ~~~~~~~~~~~~~~~~~~~~~~

#[derive(Builder)]
struct OrbitBasis<'g, 'i, G, I>
where
    G: GroupProperties,
{
    /// The origins from which orbits are generated.
    origins: Vec<&'i I>,

    /// The group acting on the origins to generate orbits, the concatenation of which forms the
    /// basis.
    group: &'g G,

    /// A function defining the action of each group element on the origin.
    action: fn(&G::GroupElement, &I) -> Result<I, anyhow::Error>,
}

impl<'g, 'i, G, I> OrbitBasis<'g, 'i, G, I>
where
    G: GroupProperties + Clone,
    I: Clone,
{
    fn builder() -> OrbitBasisBuilder<'g, 'i, G, I> {
        OrbitBasisBuilder::<G, I>::default()
    }
}

impl<'g, 'i, G, I> Basis<I> for OrbitBasis<'g, 'i, G, I>
where
    G: GroupProperties,
{
    type BasisIter = OrbitBasisIterator<'i, G, I>;

    fn n_items(&self) -> usize {
        self.origins.len() * self.group.order()
    }

    fn iter(&self) -> Self::BasisIter {
        OrbitBasisIterator::new(self.group, self.origins.clone(), self.action)
    }
}

/// Lazy iterator for basis constructed from the concatenation of orbits generated from multiple
/// origins.
struct OrbitBasisIterator<'i, G, I>
where
    G: GroupProperties,
{
    /// A mutable iterator over the Cartesian product between the group elements and the origins.
    group_origin_iter: Product<
        <<G as GroupProperties>::ElementCollection as IntoIterator>::IntoIter,
        std::vec::IntoIter<&'i I>,
    >,

    /// A function defining the action of each group element on the origin.
    action: fn(&G::GroupElement, &I) -> Result<I, anyhow::Error>,
}

impl<'i, G, I> OrbitBasisIterator<'i, G, I>
where
    G: GroupProperties,
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
        origins: Vec<&'i I>,
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

impl<'i, G, I> Iterator for OrbitBasisIterator<'i, G, I>
where
    G: GroupProperties,
{
    type Item = Result<I, anyhow::Error>;

    fn next(&mut self) -> Option<Self::Item> {
        self.group_origin_iter
            .next()
            .map(|(op, origin)| (self.action)(&op, origin))
    }
}

// ~~~~~~~~~~~
// Eager basis
// ~~~~~~~~~~~

#[derive(Builder)]
struct EagerBasis<I: Clone> {
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
}
