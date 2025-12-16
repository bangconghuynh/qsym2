//! Basis for non-orthogonal configuration interaction of Slater determinants.

use std::collections::VecDeque;

use anyhow::{self, format_err};
use derive_builder::Builder;
use itertools::Itertools;
use itertools::structs::Product;

use crate::group::GroupProperties;

#[path = "basis_transformation.rs"]
mod basis_transformation;

#[cfg(test)]
#[path = "basis_tests.rs"]
mod basis_tests;

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
    fn first(&self) -> Option<I>;
}

// --------------------------------------
// Struct definitions and implementations
// --------------------------------------

// ~~~~~~~~~~~~~~~~~~~~~~
// Lazy basis from orbits
// ~~~~~~~~~~~~~~~~~~~~~~

#[derive(Builder, Clone)]
pub struct OrbitBasis<'g, G, I>
where
    G: GroupProperties,
{
    /// The origins from which orbits are generated.
    origins: Vec<I>,

    /// The group acting on the origins to generate orbits, the concatenation of which forms the
    /// basis.
    group: &'g G,

    /// Additional operators acting on the entire orbit basis (right-most operator acts first). Each
    /// operator has an associated action that defines how it operatres on the elements in the
    /// orbit basis.
    #[builder(default = "None")]
    #[allow(clippy::type_complexity)]
    prefactors: Option<
        VecDeque<(
            G::GroupElement,
            fn(&G::GroupElement, &I) -> Result<I, anyhow::Error>,
        )>,
    >,

    /// A function defining the action of each group element on the origin.
    action: fn(&G::GroupElement, &I) -> Result<I, anyhow::Error>,
}

impl<'g, G, I> OrbitBasis<'g, G, I>
where
    G: GroupProperties + Clone,
    I: Clone,
{
    pub fn builder() -> OrbitBasisBuilder<'g, G, I> {
        OrbitBasisBuilder::<'g, G, I>::default()
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

    /// Additional operators acting on the entire orbit basis (right-most operator acts first). Each
    /// operator has an associated action that defines how it operatres on the elements in the
    /// orbit basis.
    #[allow(clippy::type_complexity)]
    pub fn prefactors(
        &self,
    ) -> Option<
        &VecDeque<(
            G::GroupElement,
            fn(&G::GroupElement, &I) -> Result<I, anyhow::Error>,
        )>,
    > {
        self.prefactors.as_ref()
    }
}

impl<'g, G, I> OrbitBasis<'g, G, I>
where
    G: GroupProperties,
    I: Clone,
{
    /// Converts this orbit basis into the equivalent eager basis.
    pub fn to_eager(&self) -> Result<EagerBasis<I>, anyhow::Error> {
        let elements = self.iter().collect::<Result<Vec<_>, _>>()?;
        EagerBasis::builder()
            .elements(elements)
            .build()
            .map_err(|err| format_err!(err))
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

    /// Iterates over the elements of the [`OrbitBasis`]. Each element is indexed by `iI` where `i`
    /// enumerates the group elements and `I` enumerates the origins. `I` is the fast index and `i`
    /// the slow one.
    fn iter(&self) -> Self::BasisIter {
        OrbitBasisIterator::new(
            self.prefactors.clone(),
            self.group,
            self.origins.clone(),
            self.action,
        )
    }

    fn first(&self) -> Option<I> {
        if let Some(prefactors) = self.prefactors.as_ref() {
            prefactors
                .iter()
                .rev()
                .try_fold(self.origins.first()?.clone(), |acc, (symop, action)| {
                    (action)(symop, &acc).ok()
                })
        } else {
            self.origins.first().cloned()
        }
    }
}

/// Lazy iterator for basis constructed from the concatenation of orbits generated from multiple
/// origins.
pub struct OrbitBasisIterator<G, I>
where
    G: GroupProperties,
{
    #[allow(clippy::type_complexity)]
    prefactors: Option<
        VecDeque<(
            G::GroupElement,
            fn(&G::GroupElement, &I) -> Result<I, anyhow::Error>,
        )>,
    >,

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
    /// Each element returned by the iterator is indexed by `iI` where `i` enumerates the group
    /// elements and `I` enumerates the origins. `I` is the fast index and `i` the slow one.
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
    #[allow(clippy::type_complexity)]
    fn new(
        prefactors: Option<
            VecDeque<(
                G::GroupElement,
                fn(&G::GroupElement, &I) -> Result<I, anyhow::Error>,
            )>,
        >,
        group: &G,
        origins: Vec<I>,
        action: fn(&G::GroupElement, &I) -> Result<I, anyhow::Error>,
    ) -> Self {
        Self {
            prefactors,
            group_origin_iter: group
                .elements()
                .clone()
                .into_iter()
                .cartesian_product(origins),
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
        if let Some(prefactors) = self.prefactors.as_ref() {
            // let group_action_result = self
            //     .group_origin_iter
            //     .next()
            //     .map(|(op, origin)| (self.action)(&op, &origin))?;
            // Some((self.action)(prefactor, group_action_result.as_ref().ok()?))
            self.group_origin_iter.next().and_then(|(op, origin)| {
                prefactors
                    .iter()
                    .rev()
                    .try_fold((self.action)(&op, &origin), |acc_res, (symop, action)| {
                        acc_res.map(|acc| (action)(symop, &acc))
                    })
                    .ok()
            })
        } else {
            self.group_origin_iter
                .next()
                .map(|(op, origin)| (self.action)(&op, &origin))
        }
    }
}

// ~~~~~~~~~~~
// Eager basis
// ~~~~~~~~~~~

#[derive(Builder, Clone)]
pub struct EagerBasis<I: Clone> {
    /// The elements in the basis.
    elements: Vec<I>,
}

impl<I: Clone> EagerBasis<I> {
    pub fn builder() -> EagerBasisBuilder<I> {
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

    fn first(&self) -> Option<I> {
        self.elements.first().cloned()
    }
}
