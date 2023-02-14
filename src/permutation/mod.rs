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

// ==================
// Struct definitions
// ==================

/// A structure to manage permutation actions of a finite set.
#[derive(Builder, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Permutation {
    /// If the permutation is to act on an ordered sequence of $`n`$ integers, $`0, 1, \ldots, n`$
    /// where $`n`$ is [`Self::rank`], then this gives the result of the action.
    #[builder(setter(custom))]
    image: Vec<u8>,
}

impl PermutationBuilder {
    fn image(&mut self, perm: Vec<u8>) -> &mut Self {
        let mut uniq = HashSet::<u8>::new();
        assert!(
            perm.iter().all(move |x| uniq.insert(*x)),
            "The permutation image `{perm:?}` contains repeated elements."
        );
        self.image = Some(perm);
        self
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
    pub fn from_image(image: Vec<u8>) -> Self {
        Self::builder().image(image.clone()).build().unwrap_or_else(|err| {
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
    pub fn from_cycles(cycles: &[Vec<u8>]) -> Self {
        let mut uniq = HashSet::<u8>::new();
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
            .collect::<Vec<(u8, u8)>>();
        image_map.sort();
        let image = image_map
            .into_iter()
            .map(|(_, img)| img)
            .collect::<Vec<u8>>();
        Self::from_image(image)
    }

    pub fn rank(&self) -> u8 {
        let rank = u8::try_from(self.image.len()).unwrap_or_else(|_| {
            panic!(
                "The rank of `{:?}` is too large to be represented as `u8`.",
                self.image
            )
        });
        rank
    }

    /// If the permutation is to act on an ordered sequence of integers, $`0, 1, \ldots`$, then
    /// this gives the result of the action.
    pub fn image(&self) -> &Vec<u8> {
        &self.image
    }

    /// Obtains the cycle representation of the permutation.
    pub fn cycles(&self) -> Vec<Vec<u8>> {
        let image = &self.image;
        let rank = self.rank();
        let mut remaining_indices = (0..rank).rev().collect::<IndexSet<u8>>();
        let mut cycles: Vec<Vec<u8>> = Vec::with_capacity(usize::from(rank));
        while !remaining_indices.is_empty() {
            let start = remaining_indices
                .pop()
                .expect("`remaining_indices` should not be empty.");
            let mut cycle: Vec<u8> = Vec::with_capacity(remaining_indices.len());
            cycle.push(start);
            let mut idx = start;
            while image[usize::from(idx)] != start {
                idx = image[usize::from(idx)];
                assert!(remaining_indices.shift_remove(&idx));
                cycle.push(idx);
            }
            cycle.shrink_to_fit();
            cycles.push(cycle);
        }
        cycles.shrink_to_fit();
        cycles.sort_by_key(|cycle| (!cycle.len(), cycle.clone()));
        cycles
    }

    /// Obtains the pattern of the cycle representation of the permutation.
    pub fn cycle_pattern(&self) -> Vec<u8> {
        self.cycles()
            .iter()
            .map(|cycle| u8::try_from(cycle.len()).expect("Some cycle lengths are too long for `u8`."))
            .collect::<Vec<u8>>()
    }

    /// Returns `true` if this permutation is the identity permutation for this rank.
    pub fn is_identity(&self) -> bool {
        self.image == (0..self.rank()).collect::<Vec<u8>>()
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
            self.rank(),
            rhs.rank(),
            "The ranks of two multiplying permutations do not match."
        );
        Self::Output::builder()
            .image(
                rhs.image
                    .iter()
                    .map(|&ri| self.image[usize::from(ri)])
                    .collect::<Vec<u8>>(),
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
        let mut image_inv = (0..self.rank()).collect::<Vec<_>>();
        image_inv.sort_by_key(|&i| self.image[usize::from(i)]);
        Self::Output::builder()
            .image(image_inv)
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
                .image((0..self.rank()).collect::<Vec<_>>())
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
