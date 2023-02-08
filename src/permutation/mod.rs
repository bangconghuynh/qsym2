use std::collections::HashSet;
use std::ops::Mul;

use derive_builder::Builder;
use indexmap::IndexSet;
use log;
use num_traits::{Inv, Pow};

#[cfg(test)]
mod permutation_tests;

/// A structure to manage permutation actions of a finite set.
#[derive(Builder, Clone, Debug, PartialEq, Eq, Hash)]
struct Permutation {
    /// The rank of the permutation, *i.e.* the number of elements in the finite set on which the
    /// permutation acts.
    rank: usize,

    /// If the permutation is to act on an ordered sequence of $`n`$ integers, $`0, 1, \ldots, n`$
    /// where $`n`$ is [`Self::rank`], then this gives the result of the action.
    #[builder(setter(custom))]
    image: Vec<usize>,

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

    pub fn from_image(image: &[usize]) -> Self {
        assert_eq!(
            image.len(),
            image.iter().cloned().collect::<HashSet<usize>>().len()
        );
        Self::builder()
            .rank(image.len())
            .image(image)
            .build()
            .unwrap_or_else(|err| {
                log::error!("{err}");
                panic!("Unable to construct a `Permutation` from `{image:?}`.")
            })
    }

    pub fn from_cycles(cycles: &[Vec<usize>]) -> Self {
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
