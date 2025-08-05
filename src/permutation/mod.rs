//! Permutations as elements in symmetric groups.

use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;
use std::hash::Hash;
use std::ops::Mul;

use anyhow::{self, format_err};
use bitvec::prelude::*;
use derive_builder::Builder;
use factorial::Factorial;
use indexmap::IndexSet;
use num::{integer::lcm, Integer, Unsigned};
use num_traits::{Inv, Pow, PrimInt};
use serde::{Deserialize, Serialize};

use crate::group::FiniteOrder;

mod permutation_group;
mod permutation_symbols;

#[cfg(test)]
mod permutation_tests;

// =================
// Trait definitions
// =================

/// Trait defining a permutable collection consisting of discrete and distinguishable items that
/// can be permuted.
pub trait PermutableCollection
where
    Self: Sized,
    Self::Rank: PermutationRank,
{
    /// The type of the rank of the permutation.
    type Rank;

    /// Determines the permutation, if any, that maps `self` to `other`.
    fn get_perm_of(&self, other: &Self) -> Option<Permutation<Self::Rank>>;

    /// Permutes the items in the current collection by `perm` and returns a new collection with
    /// the permuted items.
    fn permute(&self, perm: &Permutation<Self::Rank>) -> Result<Self, anyhow::Error>;

    /// Permutes in-place the items in the current collection by `perm`.
    fn permute_mut(&mut self, perm: &Permutation<Self::Rank>) -> Result<(), anyhow::Error>;
}

/// Trait defining an action on a permutable collection that can be converted into an equivalent
/// permutation acting on that collection.
pub trait IntoPermutation<C: PermutableCollection> {
    /// Determines the permutation of `rhs` considered as a collection induced by the action of
    /// `self` on `rhs` considered as an element in its domain. If no such permutation could be
    /// found, `None` is returned.
    fn act_permute(&self, rhs: &C) -> Option<Permutation<C::Rank>>;
}

/// Trait for generic permutation rank types.
pub trait PermutationRank:
    Integer + Unsigned + BitStore + PrimInt + Hash + TryFrom<usize> + Into<usize> + Serialize
{
}

/// Blanket implementation of `PermutationRank`.
impl<T> PermutationRank for T
where
    T: Integer + Unsigned + BitStore + PrimInt + Hash + TryFrom<usize> + Into<usize> + Serialize,
    <T as TryFrom<usize>>::Error: fmt::Debug,
    std::ops::Range<T>: Iterator + DoubleEndedIterator,
    Vec<T>: FromIterator<<std::ops::Range<T> as Iterator>::Item>,
    VecDeque<T>: FromIterator<<std::ops::Range<T> as Iterator>::Item>,
    IndexSet<T>: FromIterator<<std::ops::Range<T> as Iterator>::Item>,
{
}

// ==================
// Struct definitions
// ==================

/// Structure to manage permutation actions of a finite set.
#[derive(Builder, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Permutation<T: PermutationRank> {
    /// If the permutation is to act on an ordered sequence of $`n`$ integers, $`0, 1, \ldots, n`$,
    /// then this gives the result of the action.
    #[builder(setter(custom))]
    image: Vec<T>,
}

impl<T: PermutationRank> PermutationBuilder<T> {
    /// If the permutation is to act on an ordered sequence of $`n`$ integers, $`0, 1, \ldots, n`$,
    /// then this gives the result of the action.
    pub fn image(&mut self, perm: Vec<T>) -> &mut Self {
        let mut uniq = HashSet::<T>::new();
        // assert!(
        //     perm.iter().all(move |x| uniq.insert(*x)),
        //     "The permutation image `{perm:?}` contains repeated elements."
        // );
        if perm.iter().all(move |x| uniq.insert(*x)) {
            // The permutation contains all distinct elements.
            self.image = Some(perm);
        }
        self
    }
}

impl<T: PermutationRank> Permutation<T> {
    /// Returns a builder to construct a new permutation.
    #[must_use]
    fn builder() -> PermutationBuilder<T> {
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
    pub fn from_image(image: Vec<T>) -> Result<Self, anyhow::Error> {
        Self::builder()
            .image(image.clone())
            .build()
            .map_err(|err| format_err!(err))
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
    pub fn from_cycles(cycles: &[Vec<T>]) -> Result<Self, anyhow::Error> {
        let mut uniq = HashSet::<T>::new();
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
            .collect::<Vec<(T, T)>>();
        image_map.sort();
        let image = image_map
            .into_iter()
            .map(|(_, img)| img)
            .collect::<Vec<T>>();
        Self::from_image(image)
    }

    /// Constructs a permutation from its Lehmer encoding.
    ///
    /// See [here](https://en.wikipedia.org/wiki/Lehmer_code) for additional information.
    pub fn from_lehmer(lehmer: Vec<T>) -> Result<Self, anyhow::Error>
    where
        <T as TryFrom<usize>>::Error: fmt::Debug,
        std::ops::Range<T>: Iterator,
        VecDeque<T>: FromIterator<<std::ops::Range<T> as Iterator>::Item>,
    {
        let n = T::try_from(lehmer.len()).expect("Unable to convert the `lehmer` length to `u8`.");
        let mut remaining = (T::zero()..n).collect::<VecDeque<T>>();
        let image = lehmer
            .iter()
            .map(|&k| {
                remaining.remove(k.into()).unwrap_or_else(|| {
                    panic!("Unable to retrieve element index `{k:?}` from `{remaining:?}`.")
                })
            })
            .collect::<Vec<_>>();
        Self::from_image(image)
    }

    /// Constructs a permutation from the integer index obtained from a Lehmer encoding.
    ///
    /// See [here](https://en.wikipedia.org/wiki/Lehmer_code) and
    /// Korf, R. E. Linear-time disk-based implicit graph search. *J. ACM* **55**, 1–40 (2008).
    /// for additional information.
    ///
    /// # Arguments
    ///
    /// * `index` - An integer index.
    /// * `rank` - A rank for the permutation to be constructed.
    ///
    /// # Returns
    ///
    /// Returns the corresponding permutation.
    ///
    /// # Errors
    ///
    /// If `index` is not valid for a permutation of rank `rank`.
    pub fn from_lehmer_index(index: usize, rank: T) -> Result<Self, anyhow::Error>
    where
        <T as TryFrom<usize>>::Error: fmt::Debug,
        std::ops::Range<T>: Iterator,
        VecDeque<T>: FromIterator<<std::ops::Range<T> as Iterator>::Item>,
    {
        let mut quotient = index;
        let mut lehmer: VecDeque<T> = VecDeque::new();
        let mut i = 1usize;
        while quotient != 0 {
            if i == 1 {
                lehmer.push_front(T::zero());
            } else {
                lehmer.push_front(T::try_from(quotient.rem_euclid(i)).unwrap());
                quotient = quotient.div_euclid(i);
            }
            i += 1;
        }
        if lehmer.len() > rank.into() {
            Err(format_err!(
                "The Lehmer encode length is larger than the rank of the permutation."
            ))
        } else {
            while lehmer.len() < rank.into() {
                lehmer.push_front(T::zero());
            }
            Self::from_lehmer(lehmer.into_iter().collect::<Vec<_>>())
        }
    }

    /// The rank of the permutation.
    pub fn rank(&self) -> T {
        T::try_from(self.image.len()).unwrap_or_else(|_| {
            panic!(
                "The rank of `{:?}` is too large to be represented as `u8`.",
                self.image
            )
        })
    }

    /// If the permutation is to act on an ordered sequence of integers, $`0, 1, \ldots`$, then
    /// this gives the result of the action.
    pub fn image(&self) -> &Vec<T> {
        &self.image
    }

    /// Obtains the cycle representation of the permutation.
    pub fn cycles(&self) -> Vec<Vec<T>>
    where
        std::ops::Range<T>: Iterator + DoubleEndedIterator,
        Vec<T>: FromIterator<<std::ops::Range<T> as Iterator>::Item>,
        IndexSet<T>: FromIterator<<std::ops::Range<T> as Iterator>::Item>,
    {
        let image = &self.image;
        let rank = self.rank();
        let mut remaining_indices = (T::zero()..rank).rev().collect::<IndexSet<T>>();
        let mut cycles: Vec<Vec<T>> = Vec::with_capacity(rank.into());
        while !remaining_indices.is_empty() {
            let start = remaining_indices
                .pop()
                .expect("`remaining_indices` should not be empty.");
            let mut cycle: Vec<T> = Vec::with_capacity(remaining_indices.len());
            cycle.push(start);
            let mut idx = start;
            while image[idx.into()] != start {
                idx = image[idx.into()];
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

    /// Returns the pattern of the cycle representation of the permutation.
    pub fn cycle_pattern(&self) -> Vec<T>
    where
        <T as TryFrom<usize>>::Error: fmt::Debug,
        std::ops::Range<T>: Iterator + DoubleEndedIterator,
        Vec<T>: FromIterator<<std::ops::Range<T> as Iterator>::Item>,
        IndexSet<T>: FromIterator<<std::ops::Range<T> as Iterator>::Item>,
    {
        self.cycles()
            .iter()
            .map(|cycle| {
                T::try_from(cycle.len()).expect("Some cycle lengths are too long for `u8`.")
            })
            .collect::<Vec<T>>()
    }

    /// Returns `true` if this permutation is the identity permutation for this rank.
    pub fn is_identity(&self) -> bool
    where
        <T as TryFrom<usize>>::Error: fmt::Debug,
    {
        self.lehmer_index(None) == 0
    }

    /// Returns the Lehmer encoding of the permutation.
    ///
    /// # Arguments
    ///
    /// * `count_ones_opt` - An optional hashmap containing the number of ones in each of the
    /// possible bit vectors of length [`Self::rank`].
    ///
    /// # Returns
    ///
    /// The Lehmer encoding of this permutation. See
    /// [here](https://en.wikipedia.org/wiki/Lehmer_code) for additional information.
    pub fn lehmer(&self, count_ones_opt: Option<&HashMap<BitVec<T, Lsb0>, T>>) -> Vec<T>
    where
        <T as TryFrom<usize>>::Error: fmt::Debug,
    {
        let mut bv: BitVec<T, Lsb0> = bitvec![T, Lsb0; 0; self.rank().into()];
        let n = self.rank();
        self.image
            .iter()
            .enumerate()
            .map(|(i, &k)| {
                let flipped_bv_k = !bv[k.into()];
                bv.set(k.into(), flipped_bv_k);
                if i == 0 {
                    k
                } else if i == (n - T::one()).into() {
                    T::zero()
                } else {
                    let mut bv_k = bv.clone();
                    bv_k.shift_right((n - k).into());
                    k - if let Some(count_ones) = count_ones_opt {
                        *(count_ones.get(&bv_k).unwrap_or_else(|| {
                            panic!("Unable to count the number of ones in `{bv}`.")
                        }))
                    } else {
                        T::try_from(bv_k.count_ones())
                            .expect("Unable to convert the number of ones to `u8`.")
                    }
                }
            })
            .collect::<Vec<T>>()
    }

    /// Returns the integer corresponding to the Lehmer encoding of this permutation.
    ///
    /// See [here](https://en.wikipedia.org/wiki/Lehmer_code) and
    /// Korf, R. E. Linear-time disk-based implicit graph search. *J. ACM* **55**, 1–40 (2008).
    /// for additional information.
    ///
    /// # Arguments
    ///
    /// * `count_ones_opt` - An optional hashmap containing the number of ones in each of the
    /// possible bit vectors of length [`Self::rank`].
    ///
    /// # Returns
    ///
    /// Returns the integer corresponding to the Lehmer encoding of this permutation.
    pub fn lehmer_index(&self, count_ones_opt: Option<&HashMap<BitVec<T, Lsb0>, T>>) -> usize
    where
        <T as TryFrom<usize>>::Error: fmt::Debug,
    {
        let lehmer = self.lehmer(count_ones_opt);
        let n = self.rank().into() - 1;
        if n == 0 {
            0
        } else {
            lehmer
                .into_iter()
                .enumerate()
                .map(|(i, l)| {
                    l.into()
                        * (n - i).checked_factorial().unwrap_or_else(|| {
                            panic!("The factorial of `{}` cannot be correctly computed.", n - i)
                        })
                })
                .sum()
        }
    }
}

// =====================
// Trait implementations
// =====================

// -------
// Display
// -------
impl<T: PermutationRank + fmt::Display> fmt::Display for Permutation<T> {
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
impl<T: PermutationRank> Mul<&'_ Permutation<T>> for &Permutation<T>
where
    std::ops::Range<T>: Iterator + DoubleEndedIterator,
    Vec<T>: FromIterator<<std::ops::Range<T> as Iterator>::Item>,
    VecDeque<T>: FromIterator<<std::ops::Range<T> as Iterator>::Item>,
    <T as TryFrom<usize>>::Error: fmt::Debug,
    IndexSet<T>: FromIterator<<std::ops::Range<T> as Iterator>::Item>,
{
    type Output = Permutation<T>;

    fn mul(self, rhs: &Permutation<T>) -> Self::Output {
        assert_eq!(
            self.rank(),
            rhs.rank(),
            "The ranks of two multiplying permutations do not match."
        );
        Self::Output::builder()
            .image(
                rhs.image
                    .iter()
                    .map(|&ri| self.image[ri.into()])
                    .collect::<Vec<T>>(),
            )
            .build()
            .expect("Unable to construct a product `Permutation`.")
    }
}

impl<T: PermutationRank> Mul<&'_ Permutation<T>> for Permutation<T>
where
    std::ops::Range<T>: Iterator + DoubleEndedIterator,
    Vec<T>: FromIterator<<std::ops::Range<T> as Iterator>::Item>,
    VecDeque<T>: FromIterator<<std::ops::Range<T> as Iterator>::Item>,
    <T as TryFrom<usize>>::Error: fmt::Debug,
    IndexSet<T>: FromIterator<<std::ops::Range<T> as Iterator>::Item>,
{
    type Output = Permutation<T>;

    fn mul(self, rhs: &Permutation<T>) -> Self::Output {
        &self * rhs
    }
}

impl<T: PermutationRank> Mul<Permutation<T>> for Permutation<T>
where
    std::ops::Range<T>: Iterator + DoubleEndedIterator,
    Vec<T>: FromIterator<<std::ops::Range<T> as Iterator>::Item>,
    VecDeque<T>: FromIterator<<std::ops::Range<T> as Iterator>::Item>,
    <T as TryFrom<usize>>::Error: fmt::Debug,
    IndexSet<T>: FromIterator<<std::ops::Range<T> as Iterator>::Item>,
{
    type Output = Permutation<T>;

    fn mul(self, rhs: Permutation<T>) -> Self::Output {
        &self * &rhs
    }
}

impl<T: PermutationRank> Mul<Permutation<T>> for &Permutation<T>
where
    std::ops::Range<T>: Iterator + DoubleEndedIterator,
    Vec<T>: FromIterator<<std::ops::Range<T> as Iterator>::Item>,
    VecDeque<T>: FromIterator<<std::ops::Range<T> as Iterator>::Item>,
    <T as TryFrom<usize>>::Error: fmt::Debug,
    IndexSet<T>: FromIterator<<std::ops::Range<T> as Iterator>::Item>,
{
    type Output = Permutation<T>;

    fn mul(self, rhs: Permutation<T>) -> Self::Output {
        self * &rhs
    }
}

// ---
// Inv
// ---
impl<T: PermutationRank> Inv for &Permutation<T>
where
    std::ops::Range<T>: Iterator + DoubleEndedIterator,
    Vec<T>: FromIterator<<std::ops::Range<T> as Iterator>::Item>,
    VecDeque<T>: FromIterator<<std::ops::Range<T> as Iterator>::Item>,
    <T as TryFrom<usize>>::Error: fmt::Debug,
    IndexSet<T>: FromIterator<<std::ops::Range<T> as Iterator>::Item>,
{
    type Output = Permutation<T>;

    fn inv(self) -> Self::Output {
        let mut image_inv = (T::zero()..self.rank()).collect::<Vec<T>>();
        image_inv.sort_by_key(|&i| self.image[i.into()]);
        Self::Output::builder()
            .image(image_inv)
            .build()
            .expect("Unable to construct an inverse `Permutation`.")
    }
}

impl<T: PermutationRank> Inv for Permutation<T>
where
    std::ops::Range<T>: Iterator + DoubleEndedIterator,
    Vec<T>: FromIterator<<std::ops::Range<T> as Iterator>::Item>,
    VecDeque<T>: FromIterator<<std::ops::Range<T> as Iterator>::Item>,
    <T as TryFrom<usize>>::Error: fmt::Debug,
    IndexSet<T>: FromIterator<<std::ops::Range<T> as Iterator>::Item>,
{
    type Output = Permutation<T>;

    fn inv(self) -> Self::Output {
        (&self).inv()
    }
}

// ---
// Pow
// ---
impl<T: PermutationRank> Pow<i32> for &Permutation<T>
where
    std::ops::Range<T>: Iterator + DoubleEndedIterator,
    Vec<T>: FromIterator<<std::ops::Range<T> as Iterator>::Item>,
    VecDeque<T>: FromIterator<<std::ops::Range<T> as Iterator>::Item>,
    <T as TryFrom<usize>>::Error: fmt::Debug,
    IndexSet<T>: FromIterator<<std::ops::Range<T> as Iterator>::Item>,
{
    type Output = Permutation<T>;

    fn pow(self, rhs: i32) -> Self::Output {
        if rhs == 0 {
            Self::Output::builder()
                .image((T::zero()..self.rank()).collect::<Vec<T>>())
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

impl<T: PermutationRank> Pow<i32> for Permutation<T>
where
    std::ops::Range<T>: Iterator + DoubleEndedIterator,
    Vec<T>: FromIterator<<std::ops::Range<T> as Iterator>::Item>,
    VecDeque<T>: FromIterator<<std::ops::Range<T> as Iterator>::Item>,
    <T as TryFrom<usize>>::Error: fmt::Debug,
    IndexSet<T>: FromIterator<<std::ops::Range<T> as Iterator>::Item>,
{
    type Output = Permutation<T>;

    fn pow(self, rhs: i32) -> Self::Output {
        (&self).pow(rhs)
    }
}

// -----------
// FiniteOrder
// -----------
impl<T: PermutationRank + fmt::Display> FiniteOrder for Permutation<T>
where
    u32: From<T>,
    std::ops::Range<T>: Iterator + DoubleEndedIterator,
    Vec<T>: FromIterator<<std::ops::Range<T> as Iterator>::Item>,
    VecDeque<T>: FromIterator<<std::ops::Range<T> as Iterator>::Item>,
    <T as TryFrom<usize>>::Error: fmt::Debug,
    IndexSet<T>: FromIterator<<std::ops::Range<T> as Iterator>::Item>,
{
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

// =========
// Functions
// =========
/// Permutes the items in a vector in-place.
///
/// # Arguments
///
/// * `vec` - An exclusive reference to a vector of items to be permuted.
/// * `perm` - A shared reference to a permutation determining how the items are to be permuted.
pub(crate) fn permute_inplace<T>(vec: &mut Vec<T>, perm: &Permutation<usize>) {
    assert_eq!(
        perm.rank(),
        vec.len(),
        "The permutation rank does not match the number of items in the vector."
    );
    let mut image = perm.image().clone();
    for idx in 0..vec.len() {
        if image[idx] != idx {
            let mut current_idx = idx;
            loop {
                let target_idx = image[current_idx];
                image[current_idx] = current_idx;
                if image[target_idx] == target_idx {
                    break;
                }
                vec.swap(current_idx, target_idx);
                current_idx = target_idx;
            }
        }
    }
}
