use std::collections::HashSet;
use std::fmt;
use std::ops::Mul;

use derive_builder::Builder;
use indexmap::IndexSet;
use log;
use num::integer::lcm;
use num_traits::{Inv, Pow};

use crate::group::FiniteOrder;

mod permutation_group;
mod permutation_symbols;

#[cfg(test)]
mod permutation_tests;

// =================
// Trait definitions
// =================

/// A trait defining a permutable collection consisting of discrete and distinguishable items that
/// can be permuted.
pub trait PermutableCollection {
    /// Type of the items in the collection being permuted.
    type Item;

    /// Determines the permutation, if any, that maps `self` to `other`.
    fn perm(&self, other: &Self) -> Option<Permutation>;
}

/// A trait defining an action on a permutable collection that can be converted into an equivalent
/// permutation acting on that collection.
pub trait IntoPermutation<C: PermutableCollection> {

    /// Determines the permutation of `rhs` considered as a collection induced by the action of
    /// `self` on `rhs` considered as an element in its domain.
    fn act_permute(&self, rhs: &C) -> Permutation;
}

// ==================
// Struct definitions
// ==================

/// A structure to manage permutation actions of a finite set.
#[derive(Builder, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Permutation {
    /// The rank of the permutation, *i.e.* the number of elements in the finite set on which the
    /// permutation acts.
    rank: usize,

    /// If the permutation is to act on an ordered sequence of $`n`$ integers, $`0, 1, \ldots, n`$
    /// where $`n`$ is [`Self::rank`], then this gives the result of the action.
    #[builder(setter(custom))]
    image: Vec<usize>,

    /// The disjoint cycles of this permutation.
    #[builder(setter(skip), default = "self.calc_cycles()")]
    cycles: Vec<Vec<usize>>,
}

impl PermutationBuilder {
    fn image(&mut self, perm: &[usize]) -> &mut Self {
        assert_eq!(
            self.rank
                .expect("The rank for this permutation has not been set."),
            perm.len(),
            "The permutation image `{perm:?}` does not contain the expected number of elements",
        );
        let mut uniq = HashSet::<usize>::new();
        assert!(
            perm.into_iter().all(move |x| uniq.insert(*x)),
            "The permutation image `{perm:?}` contains repeated elements."
        );
        self.image = Some(perm.to_vec());
        self
    }

    fn calc_cycles(&self) -> Vec<Vec<usize>> {
        let rank = self.rank.expect("Permutation rank has not been set.");
        let image = self
            .image
            .as_ref()
            .expect("Permutation image has not been set.");
        let mut remaining_indices = (0..rank).rev().collect::<IndexSet<usize>>();
        let mut cycles: Vec<Vec<usize>> = Vec::with_capacity(rank);
        while !remaining_indices.is_empty() {
            let start = remaining_indices
                .pop()
                .expect("`remaining_indices` should not be empty.");
            let mut cycle: Vec<usize> = Vec::with_capacity(remaining_indices.len());
            cycle.push(start);
            let mut idx = start;
            while image[idx] != start {
                idx = image[idx];
                assert!(remaining_indices.shift_remove(&idx));
                cycle.push(idx);
            }
            cycles.push(cycle);
        }
        cycles.sort_by_key(|cycle| (!cycle.len(), cycle.clone()));
        cycles
    }
}

impl Permutation {
    /// Returns a builder to construct a new permutation.
    #[must_use]
    fn builder() -> PermutationBuilder {
        PermutationBuilder::default()
    }

    /// Constructs a permutation from the image of its action on an ordered sequence of integers,
    /// $`0, 1, \ldots`$.
    ///
    /// # Arguments
    ///
    /// * `image` - The image of the permutation when acting on an ordered sequence of integers,
    /// $`0, 1, \ldots`$.
    ///
    /// # Returns
    ///
    /// The corresponding permutation.
    ///
    /// # Panics
    ///
    /// Panics if `image` contains repeated elements.
    pub fn from_image(image: &[usize]) -> Self {
        Self::builder()
            .rank(image.len())
            .image(image)
            .build()
            .unwrap_or_else(|err| {
                log::error!("{err}");
                panic!("Unable to construct a `Permutation` from `{image:?}`.")
            })
    }

    /// Constructs a permutation from its disjoint cycles.
    ///
    /// # Arguments
    ///
    /// * `cycles` - The disjoint cycles defining the permutation.
    ///
    /// # Returns
    ///
    /// The corresponding permutation.
    ///
    /// # Panics
    ///
    /// Panics if the cycles in `cycles` contain repeated elements.
    pub fn from_cycles(cycles: &[Vec<usize>]) -> Self {
        let mut uniq = HashSet::<usize>::new();
        assert!(
            cycles.iter().flatten().all(move |x| uniq.insert(*x)),
            "The permutation cycles `{cycles:?}` contains repeated elements."
        );
        let mut image_map = cycles
            .iter()
            .flat_map(|cycle| {
                let start = *cycle.first().expect("Empty cycles are not permitted.");
                let end = *cycle.last().expect("Empty cycles are not permitted.");
                cycle
                    .windows(2)
                    .map(|pair| {
                        let idx = pair[0];
                        let img = pair[1];
                        (idx, img)
                    })
                    .chain([(end, start)])
            })
            .collect::<Vec<(usize, usize)>>();
        image_map.sort();
        let image = image_map
            .into_iter()
            .map(|(_, img)| img)
            .collect::<Vec<usize>>();
        Self::from_image(&image)
    }

    /// If the permutation is to act on an ordered sequence of integers, $`0, 1, \ldots`$, then
    /// this gives the result of the action.
    pub fn image(&self) -> &Vec<usize> {
        &self.image
    }

    /// Obtains the cycle representation of the permutation.
    pub fn cycles(&self) -> &Vec<Vec<usize>> {
        &self.cycles
    }

    /// Obtains the pattern of the cycle representation of the permutation.
    pub fn cycle_pattern(&self) -> Vec<usize> {
        self.cycles
            .iter()
            .map(|cycle| cycle.len())
            .collect::<Vec<usize>>()
    }

    /// Returns `true` if this permutation is the identity permutation for this rank.
    pub fn is_identity(&self) -> bool {
        self.image == (0..self.rank).collect::<Vec<usize>>()
    }
}

// =====================
// Trait implementations
// =====================

// -------
// Display
// -------
impl fmt::Display for Permutation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "π({})",
            self.image
                .iter()
                .map(|i| i.to_string())
                .collect::<Vec<_>>()
                .join(".")
        )
    }
}

// ---
// Mul
// ---
impl Mul<&'_ Permutation> for &Permutation {
    type Output = Permutation;

    fn mul(self, rhs: &Permutation) -> Self::Output {
        assert_eq!(
            self.rank, rhs.rank,
            "The ranks of two multiplying permutations do not match."
        );
        Self::Output::builder()
            .rank(self.rank)
            .image(
                &rhs.image
                    .iter()
                    .map(|&ri| self.image[ri])
                    .collect::<Vec<usize>>(),
            )
            .build()
            .expect("Unable to construct a product `Permutation`.")
    }
}

impl Mul<&'_ Permutation> for Permutation {
    type Output = Permutation;

    fn mul(self, rhs: &Permutation) -> Self::Output {
        &self * rhs
    }
}

impl Mul<Permutation> for Permutation {
    type Output = Permutation;

    fn mul(self, rhs: Permutation) -> Self::Output {
        &self * &rhs
    }
}

impl Mul<Permutation> for &Permutation {
    type Output = Permutation;

    fn mul(self, rhs: Permutation) -> Self::Output {
        self * &rhs
    }
}

// ---
// Inv
// ---
impl Inv for &Permutation {
    type Output = Permutation;

    fn inv(self) -> Self::Output {
        let mut image_inv = (0..self.rank).collect::<Vec<_>>();
        image_inv.sort_by_key(|&i| self.image[i]);
        Self::Output::builder()
            .rank(self.rank)
            .image(&image_inv)
            .build()
            .expect("Unable to construct an inverse `Permutation`.")
    }
}

impl Inv for Permutation {
    type Output = Permutation;

    fn inv(self) -> Self::Output {
        (&self).inv()
    }
}

// ---
// Pow
// ---
impl Pow<i32> for &Permutation {
    type Output = Permutation;

    fn pow(self, rhs: i32) -> Self::Output {
        if rhs == 0 {
            Self::Output::builder()
                .rank(self.rank)
                .image(&(0..self.rank).collect::<Vec<_>>())
                .build()
                .expect("Unable to construct an identity `Permutation`.")
        } else if rhs > 0 {
            self * self.pow(rhs - 1)
        } else {
            let rhs_p = -rhs;
            (self * self.pow(rhs_p - 1)).inv()
        }
    }
}

impl Pow<i32> for Permutation {
    type Output = Permutation;

    fn pow(self, rhs: i32) -> Self::Output {
        (&self).pow(rhs)
    }
}

// -----------
// FiniteOrder
// -----------
impl FiniteOrder for Permutation {
    type Int = u32;

    /// Calculates the order of this permutation. This is the lowest common multiplier of the
    /// lengths of the disjoint cycles constituting this permutation.
    fn order(&self) -> Self::Int {
        u32::try_from(
            self.cycle_pattern()
                .iter()
                .cloned()
                .reduce(lcm)
                .unwrap_or_else(|| {
                    panic!("Unable to determine the permutation order of `{self}`.")
                }),
        )
        .expect("Unable to convert the permutation order to `u32`.")
    }
}
