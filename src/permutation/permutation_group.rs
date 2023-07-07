use std::collections::{HashSet, VecDeque};
use std::fmt;
use std::ops::Range;

use anyhow;
use derive_builder::Builder;
use factorial::Factorial;
use indexmap::IndexSet;
use itertools::Itertools;
use ndarray::Array2;
use rayon::prelude::*;

use crate::chartab::chartab_group::{CharacterProperties, IrrepCharTabConstruction};
use crate::chartab::{CharacterTable, RepCharacterTable};
use crate::group::class::ClassProperties;
use crate::group::{GroupProperties, UnitaryRepresentedGroup};
use crate::permutation::permutation_symbols::{
    deduce_permutation_irrep_symbols, sort_perm_irreps, PermutationClassSymbol,
    PermutationIrrepSymbol,
};
use crate::permutation::Permutation;
use crate::permutation::PermutationRank;

#[cfg(test)]
#[path = "permutation_group_tests.rs"]
mod permutation_group_tests;

// ==================
// Struct definitions
// ==================

/// A lazy iterator for permutations of a particular rank enumerated by their Lehmer encoding
/// integers.
#[derive(Clone)]
pub struct PermutationIterator<T: PermutationRank> {
    rank: T,
    raw_perm_indices: Range<usize>,
}

impl<T: PermutationRank> Iterator for PermutationIterator<T>
where
    std::ops::Range<T>: Iterator + DoubleEndedIterator,
    Vec<T>: FromIterator<<std::ops::Range<T> as Iterator>::Item>,
    VecDeque<T>: FromIterator<<std::ops::Range<T> as Iterator>::Item>,
    <T as TryFrom<usize>>::Error: fmt::Debug,
    IndexSet<T>: FromIterator<<std::ops::Range<T> as Iterator>::Item>,
{
    type Item = Permutation<T>;

    fn next(&mut self) -> Option<Self::Item> {
        self.raw_perm_indices
            .next()
            .and_then(|index| Permutation::<T>::from_lehmer_index(index, self.rank))
    }
}

/// A dedicated structure for managing permutation groups efficiently. Only permutation groups of
/// ranks up to 20 are supported, as higher-rank permutation groups have too large orders not
/// representable with `usize` which is `u64` on most modern machines.
#[derive(Clone, Builder)]
pub struct PermutationGroup {
    /// The rank of the permutation group.
    rank: u8,

    /// The lazy iterator yielding all permutations in this group in their Lehmer-encoding-integer
    /// order.
    perms_iter: PermutationIterator<u8>,

    /// All possible cycle patterns of this group. These are the possible partitions of
    /// [`Self::rank`].
    #[builder(setter(skip), default = "None")]
    cycle_patterns: Option<IndexSet<Vec<u8>>>,

    /// A vector of hashsets, each of which contains the indices of the elements having the
    /// corresponding cycle pattern in [`Self::cycle_patterns`]. These elements are also in the
    /// same conjugacy class.
    #[builder(setter(skip), default = "None")]
    conjugacy_classes: Option<Vec<HashSet<usize>>>,

    /// The symbols for the conjugacy classes.
    #[builder(setter(skip), default = "None")]
    conjugacy_class_symbols: Option<IndexSet<PermutationClassSymbol<u8>>>,

    /// The character table for the irreducible representations of this permutation group.
    #[builder(setter(skip), default = "None")]
    irrep_character_table:
        Option<RepCharacterTable<PermutationIrrepSymbol, PermutationClassSymbol<u8>>>,
}

impl PermutationGroup {
    fn builder() -> PermutationGroupBuilder {
        PermutationGroupBuilder::default()
    }
}

// =================
// Trait definitions
// =================

/// A trait for permutation groups. Only permutation groups of ranks up to 20 are supported, as
/// higher-rank permutation groups have too large orders not representable with `usize` which is
/// `u64` on most modern machines.
pub trait PermutationGroupProperties:
    ClassProperties<GroupElement = Permutation<u8>, ClassSymbol = PermutationClassSymbol<u8>>
    + CharacterProperties
    + Sized
{
    /// Constructs a permutation group $`Sym(n)`$ from a given rank $`n`$ (*i.e.* the number of
    /// elements in the set to be permuted).
    ///
    /// # Arguments
    ///
    /// * `rank` - The permutation rank.
    ///
    /// # Returns
    ///
    /// A finite group of permutations.
    fn from_rank(rank: u8) -> Result<Self, anyhow::Error>;

    /// Sets class symbols from cycle patterns.
    ///
    /// Classes in permutation groups are determined by the cycle patterns of their elements. The
    /// number of classes for $`Sym(n)`$ is the number of integer partitions of $`n`$.
    fn set_class_symbols_from_cycle_patterns(&mut self) {
        log::debug!("Assigning class symbols from cycle patterns...");
        let class_symbols = (0..self.class_number())
            .map(|cc_i| {
                let rep_ele = self.get_cc_transversal(cc_i).unwrap_or_else(|| {
                    panic!("No representative element found for conjugacy class index `{cc_i}`.")
                });
                let cycle_pattern = rep_ele.cycle_pattern();
                let mut cycle_pattern_count: Vec<(u8, u8)> =
                    Vec::with_capacity(cycle_pattern.len());
                let mut i = 0u8;
                while i < u8::try_from(cycle_pattern.len()).unwrap() {
                    let mut j = i + 1;
                    while j < u8::try_from(cycle_pattern.len()).unwrap()
                        && cycle_pattern[usize::from(j)] == cycle_pattern[usize::from(i)]
                    {
                        j += 1;
                    }
                    cycle_pattern_count.push((cycle_pattern[usize::from(i)], j - i));
                    i = j;
                }
                let cycle_pattern_str = cycle_pattern_count
                    .iter()
                    .map(|(length, count)| {
                        if *count > 1 {
                            format!("{length}^{count}")
                        } else {
                            length.to_string()
                        }
                    })
                    .collect_vec()
                    .join("·");
                let size = self
                    .class_size(cc_i)
                    .unwrap_or_else(|| panic!("Unknown size for conjugacy class index `{i}`."));
                PermutationClassSymbol::new(
                    format!("{size}||{cycle_pattern_str}||").as_str(),
                    Some(vec![rep_ele]),
                )
                .unwrap_or_else(|_| {
                    panic!(
                        "Unable to construct a class symbol from `{size}||{cycle_pattern_str}||`"
                    )
                })
            })
            .collect_vec();
        self.set_class_symbols(&class_symbols);
        log::debug!("Assigning class symbols from cycle patterns... Done.");
    }

    /// Reorders and relabels the rows and columns of the constructed character table using
    /// permutation-specific rules and conventions.
    fn canonicalise_character_table(&mut self);
}

// =====================
// Trait implementations
// =====================

// -----------------------
// UnitaryRepresentedGroup
// -----------------------

impl PermutationGroupProperties
    for UnitaryRepresentedGroup<Permutation<u8>, PermutationIrrepSymbol, PermutationClassSymbol<u8>>
{
    fn from_rank(rank: u8) -> Result<Self, anyhow::Error> {
        assert!(rank > 0, "A permutation rank must be a positive integer.");
        assert!(
            rank <= 20,
            "Permutations of rank more than 20 will not be representable."
        );
        log::debug!("Generating all permutations of rank {rank}...");
        let perms = (0..rank)
            .permutations(usize::from(rank))
            .map(Permutation::from_image)
            .collect_vec();
        log::debug!("Generating all permutations of rank {rank}... Done.");
        log::debug!("Collecting all permutations into a unitary-represented group...");
        let mut group = UnitaryRepresentedGroup::<
            Permutation<u8>,
            PermutationIrrepSymbol,
            PermutationClassSymbol<u8>,
        >::new(format!("Sym({rank})").as_str(), perms)?;
        log::debug!("Collecting all permutations into a unitary-represented group... Done.");
        group.set_class_symbols_from_cycle_patterns();
        group.construct_irrep_character_table();
        group.canonicalise_character_table();
        Ok(group)
    }

    fn canonicalise_character_table(&mut self) {
        let old_chartab = self.character_table();
        let class_symbols = (0..self.class_number())
            .map(|i| {
                self.get_cc_symbol_of_index(i)
                    .expect("Unable to retrieve all class symbols.")
            })
            .collect_vec();
        let (char_arr, sorted_fs) = sort_perm_irreps(
            &old_chartab.array().view(),
            &old_chartab.frobenius_schurs.values().copied().collect_vec(),
        );
        let ordered_irreps = deduce_permutation_irrep_symbols(&char_arr.view());
        self.set_irrep_character_table(RepCharacterTable::new(
            &old_chartab.name,
            &ordered_irreps,
            &class_symbols,
            &[],
            char_arr,
            &sorted_fs,
        ));
    }
}

// ----------------
// PermutationGroup
// ----------------

impl GroupProperties for PermutationGroup {
    type GroupElement = Permutation<u8>;
    type ElementCollection = PermutationIterator<u8>;

    fn name(&self) -> String {
        format!("Sym({})", self.rank)
    }

    fn finite_subgroup_name(&self) -> Option<&String> {
        None
    }

    fn get_index(&self, index: usize) -> Option<Self::GroupElement> {
        Permutation::from_lehmer_index(index, self.rank)
    }

    fn get_index_of(&self, g: &Self::GroupElement) -> Option<usize> {
        if g.rank() != self.rank {
            None
        } else {
            Some(g.lehmer_index(None))
        }
    }

    fn contains(&self, g: &Self::GroupElement) -> bool {
        g.rank() == self.rank
    }

    fn elements(&self) -> &Self::ElementCollection {
        &self.perms_iter
    }

    fn is_abelian(&self) -> bool {
        self.perms_iter.clone().enumerate().all(|(i, gi)| {
            (0..i).all(|j| {
                let gj = self
                    .get_index(j)
                    .unwrap_or_else(|| panic!("Element with index `{j}` not found."));
                (&gi) * (&gj) == (&gj) * (&gi)
            })
        })
    }

    fn order(&self) -> usize {
        usize::from(self.rank).checked_factorial().unwrap()
    }

    fn cayley_table(&self) -> Option<&Array2<usize>> {
        None
    }
}

/// Returns all partitions of an integer.
///
/// The partitions are generated in lexicographic order, then, within each partition, the
/// sub-partitions are arranged in decreasing order.
///
/// See [here](https://jeromekelleher.net/generating-integer-partitions.html) for the algorithm.
///
/// # Arguments
///
/// * `n` - An integer.
///
/// # Returns
///
/// An indexset containing the partition vectors.
fn partitions(n: u8) -> IndexSet<Vec<u8>> {
    if n == 0 {
        IndexSet::from([vec![0]])
    } else {
        let mut res: IndexSet<Vec<u8>> = IndexSet::new();
        let mut a = vec![0; usize::from(n) + 1];
        let mut k = 1;
        let mut y = n - 1;
        while k != 0 {
            let mut x = a[k - 1] + 1;
            k -= 1;
            while 2 * x <= y {
                a[k] = x;
                y -= x;
                k += 1;
            }
            let l = k + 1;
            while x <= y {
                a[k] = x;
                a[l] = y;
                let mut cycle = a[0..k + 2].to_vec();
                cycle.reverse();
                res.insert(cycle);
                x += 1;
                y -= 1;
            }
            a[k] = x + y;
            y = x + y - 1;
            let mut cycle = a[0..k + 1].to_vec();
            cycle.reverse();
            res.insert(cycle);
        }
        res
    }
}

impl ClassProperties for PermutationGroup {
    type ClassSymbol = PermutationClassSymbol<u8>;

    /// Computes the class structure of this permutation group based on cycle patterns.
    fn compute_class_structure(&mut self) {
        log::debug!("Finding all partitions of {}...", self.rank);
        self.cycle_patterns = Some(partitions(self.rank));
        log::debug!("Finding all partitions of {}... Done.", self.rank);

        log::debug!("Finding conjugacy classes based on cycle patterns in parallel...");
        let mut conjugacy_classes = vec![HashSet::<usize>::new(); self.class_number()];

        // For ranks up to 20, the number of classes is at most 627. `u16` is then plenty to store
        // each class index. This helps reduce memory cost a little bit.
        let mut e2ccs: Vec<(usize, u16)> = Vec::with_capacity(self.order());
        (0..self.order())
            .into_par_iter()
            .map(|i| {
                let p_i = Permutation::from_lehmer_index(i, self.rank).unwrap_or_else(|| {
                    panic!(
                        "Unable to construct a permutation of rank {} with Lehmer index {i}.",
                        self.rank
                    );
                });
                let cycle_pattern = p_i.cycle_pattern();
                let c_i = u16::try_from(
                    self.cycle_patterns
                        .as_ref()
                        .expect("Cycle patterns not found.")
                        .get_index_of(&cycle_pattern)
                        .unwrap_or_else(|| {
                            panic!("Cycle pattern {cycle_pattern:?} is not valid in this group.");
                        }),
                )
                .expect("A class index cannot fit within a `u16`.");
                (i, c_i)
            })
            .collect_into_vec(&mut e2ccs);
        e2ccs.into_iter().for_each(|(i, c_i)| {
            conjugacy_classes[usize::from(c_i)].insert(i);
        });
        self.conjugacy_classes = Some(conjugacy_classes);
        log::debug!("Finding conjugacy classes based on cycle patterns in parallel... Done.");
    }

    fn get_cc_index(&self, cc_idx: usize) -> Option<&HashSet<usize>> {
        self.conjugacy_classes
            .as_ref()
            .map(|conjugacy_classes| &conjugacy_classes[cc_idx])
    }

    fn get_cc_of_element_index(&self, e_idx: usize) -> Option<usize> {
        let perm = Permutation::from_lehmer_index(e_idx, self.rank)?;
        self.cycle_patterns
            .as_ref()
            .expect("Cycle patterns not found.")
            .get_index_of(&perm.cycle_pattern())
    }

    fn get_cc_transversal(&self, cc_idx: usize) -> Option<Self::GroupElement> {
        let cycle_pattern = self
            .cycle_patterns
            .as_ref()
            .expect("Cycle patterns not found.")
            .get_index(cc_idx)?;
        let cycles = cycle_pattern
            .iter()
            .scan(0u8, |start, &l| {
                let cycle = (*start..*start + l).collect::<Vec<u8>>();
                *start += l;
                Some(cycle)
            })
            .collect_vec();
        Some(Permutation::from_cycles(&cycles))
    }

    fn get_index_of_cc_symbol(&self, cc_sym: &Self::ClassSymbol) -> Option<usize> {
        self.conjugacy_class_symbols
            .as_ref()
            .expect("Conjugacy class symbols not found.")
            .get_index_of(cc_sym)
    }

    fn get_cc_symbol_of_index(&self, cc_idx: usize) -> Option<Self::ClassSymbol> {
        self.conjugacy_class_symbols
            .as_ref()
            .expect("Conjugacy class symbols not found.")
            .get_index(cc_idx)
            .cloned()
    }

    fn set_class_symbols(&mut self, cc_symbols: &[Self::ClassSymbol]) {
        self.conjugacy_class_symbols = Some(cc_symbols.iter().cloned().collect());
    }

    fn get_inverse_cc(&self, cc_idx: usize) -> Option<usize> {
        Some(cc_idx)
    }

    fn class_number(&self) -> usize {
        self.cycle_patterns
            .as_ref()
            .expect("Cycle patterns not found.")
            .len()
    }

    /// Computes the class size given a class index.
    ///
    /// The closed form for a conjugacy class of a particular cycle pattern is given
    /// [here](https://math.stackexchange.com/questions/140311/number-of-permutations-for-a-cycle-type).
    fn class_size(&self, cc_idx: usize) -> Option<usize> {
        let cycle_pattern = self
            .cycle_patterns
            .as_ref()
            .expect("Cycle patterns not found.")
            .get_index(cc_idx)?;
        let mut cycle_pattern_count: Vec<(u8, u8)> = Vec::with_capacity(cycle_pattern.len());
        let mut i = 0u8;
        while i < u8::try_from(cycle_pattern.len()).unwrap() {
            let mut j = i + 1;
            while j < u8::try_from(cycle_pattern.len()).unwrap()
                && cycle_pattern[usize::from(j)] == cycle_pattern[usize::from(i)]
            {
                j += 1;
            }
            cycle_pattern_count.push((cycle_pattern[usize::from(i)], j - i));
            i = j;
        }
        let denom = cycle_pattern_count
            .into_iter()
            .map(|(l, m)| {
                usize::from(l).pow(u32::from(m)) * usize::from(m).checked_factorial().unwrap()
            })
            .product();
        Some(
            usize::from(self.rank)
                .checked_factorial()
                .unwrap()
                .div_euclid(denom),
        )
    }
}

impl CharacterProperties for PermutationGroup {
    type RowSymbol = PermutationIrrepSymbol;
    type CharTab = RepCharacterTable<PermutationIrrepSymbol, PermutationClassSymbol<u8>>;

    fn character_table(&self) -> &Self::CharTab {
        self.irrep_character_table
            .as_ref()
            .expect("Irrep character table not found for this group.")
    }

    fn unitary_represented(&self) -> bool {
        true
    }
}

impl IrrepCharTabConstruction for PermutationGroup {
    fn set_irrep_character_table(&mut self, chartab: Self::CharTab) {
        self.irrep_character_table = Some(chartab)
    }
}

impl PermutationGroupProperties for PermutationGroup {
    fn from_rank(rank: u8) -> Result<Self, anyhow::Error> {
        assert!(rank > 0, "A permutation rank must be a positive integer.");
        assert!(
            rank <= 20,
            "Permutations of rank more than 20 will not be representable."
        );
        log::debug!("Initialising lazy iterator for permutations of rank {rank}...");
        let perms_iter = PermutationIterator {
            rank,
            raw_perm_indices: (0..usize::from(rank).checked_factorial().unwrap()),
        };
        let mut group = PermutationGroup::builder()
            .rank(rank)
            .perms_iter(perms_iter)
            .build()
            .expect("Unable to construct a `PermutationGroup`.");
        log::debug!("Initialising lazy iterator for permutations of rank {rank}... Done.");
        group.compute_class_structure();
        group.set_class_symbols_from_cycle_patterns();
        group.construct_irrep_character_table();
        group.canonicalise_character_table();
        Ok(group)
    }

    fn canonicalise_character_table(&mut self) {
        let old_chartab = self.character_table();
        let class_symbols = (0..self.class_number())
            .map(|i| {
                self.get_cc_symbol_of_index(i)
                    .expect("Unable to retrieve all class symbols.")
            })
            .collect_vec();
        let (char_arr, sorted_fs) = sort_perm_irreps(
            &old_chartab.array().view(),
            &old_chartab.frobenius_schurs.values().copied().collect_vec(),
        );
        let ordered_irreps = deduce_permutation_irrep_symbols(&char_arr.view());
        self.set_irrep_character_table(RepCharacterTable::new(
            &old_chartab.name,
            &ordered_irreps,
            &class_symbols,
            &[],
            char_arr,
            &sorted_fs,
        ));
    }
}
