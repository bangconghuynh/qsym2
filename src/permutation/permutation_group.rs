use std::collections::HashSet;
use std::ops::Range;

use derive_builder::Builder;
use indexmap::map::Entry::Vacant;
use indexmap::IndexMap;
use itertools::Itertools;
use factorial::Factorial;
use ndarray::Array2;

use crate::chartab::chartab_group::{CharacterProperties, IrrepCharTabConstruction};
use crate::chartab::chartab_symbols::CollectionSymbol;
use crate::chartab::{CharacterTable, RepCharacterTable};
use crate::group::class::{ClassProperties, ClassStructure};
use crate::group::{GroupProperties, UnitaryRepresentedGroup};
use crate::permutation::permutation_symbols::{
    deduce_permutation_irrep_symbols, sort_perm_irreps, PermutationClassSymbol,
    PermutationIrrepSymbol,
};
use crate::permutation::Permutation;

#[cfg(test)]
#[path = "permutation_group_tests.rs"]
mod permutation_group_tests;

// ==================
// Struct definitions
// ==================

#[derive(Clone)]
pub struct PermutationIterator {
    rank: u8,
    raw_perm_indices: Range<usize>,
}

impl Iterator for PermutationIterator {
    type Item = Permutation;

    fn next(&mut self) -> Option<Self::Item> {
        self.raw_perm_indices
            .next()
            .map(|index| Permutation::from_lehmer_index(index, self.rank))
    }
}

/// A dedicated structure for managing permutation groups efficiently.
#[derive(Clone, Builder)]
pub struct PermutationGroup {
    /// The rank of the permutation group.
    rank: u8,

    perms_iter: PermutationIterator,

    /// The class structure of this permutation group that is induced by the following equivalence
    /// relation:
    ///
    /// ```math
    ///     g \sim h \Leftrightarrow \exists u : h = u g u^{-1}.
    /// ```
    ///
    /// This means that all permutations having the same cycle pattern are in the same conjugacy
    /// class.
    #[builder(setter(skip), default = "None")]
    class_structure: Option<ClassStructure<Permutation, PermutationClassSymbol>>,

    /// The character table for the irreducible representations of this permutation group.
    #[builder(setter(skip), default = "None")]
    irrep_character_table:
        Option<RepCharacterTable<PermutationIrrepSymbol, PermutationClassSymbol>>,
}

impl PermutationGroup {
    fn builder() -> PermutationGroupBuilder {
        PermutationGroupBuilder::default()
    }
}

// =================
// Trait definitions
// =================

/// Trait for permutation groups.
pub trait PermutationGroupProperties:
    ClassProperties<GroupElement = Permutation, ClassSymbol = PermutationClassSymbol>
    + CharacterProperties
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
    fn from_rank(rank: u8) -> Self;

    /// Sets class symbols from cycle patterns.
    ///
    /// Classes in permutation groups are determined by the cycle patterns of their elements. The
    /// number of classes for $`Sym(n)`$ is the number of integer partitions of $`n`$.
    fn set_class_symbols_from_cycle_patterns(&mut self) {
        log::debug!("Assigning class symbols from cycle patterns...");
        let class_symbols = self
            .conjugacy_class_symbols()
            .iter()
            .map(|(old_symbol, _)| {
                let rep_ele = old_symbol.representative().unwrap_or_else(|| {
                    panic!("No representative element found for conjugacy class `{old_symbol}`.")
                });
                let cycle_pattern = rep_ele.cycle_pattern().clone();
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
                let size = old_symbol.size();
                PermutationClassSymbol::new(
                    format!("{size}||{cycle_pattern_str}||").as_str(),
                    Some(rep_ele.clone()),
                )
                .unwrap_or_else(|_| {
                    panic!(
                        "Unable to construct a class symbol from `{size}||{cycle_pattern_str}||`"
                    )
                })
            })
            .collect_vec();
        self.class_structure_mut().set_class_symbols(&class_symbols);
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
    for UnitaryRepresentedGroup<Permutation, PermutationIrrepSymbol, PermutationClassSymbol>
{
    fn from_rank(rank: u8) -> Self {
        log::debug!("Generating all permutations of rank {rank}...");
        let perms = (0..rank)
            .permutations(usize::from(rank))
            .map(|image| Permutation::from_image(image))
            .collect_vec();
        log::debug!("Generating all permutations of rank {rank}... Done.");
        log::debug!("Collecting all permutations into a unitary-represented group...");
        let mut group = UnitaryRepresentedGroup::<
            Permutation,
            PermutationIrrepSymbol,
            PermutationClassSymbol,
        >::new(format!("Sym({rank})").as_str(), perms);
        log::debug!("Collecting all permutations into a unitary-represented group... Done.");
        group.set_class_symbols_from_cycle_patterns();
        group.construct_irrep_character_table();
        group.canonicalise_character_table();
        group
    }

    fn canonicalise_character_table(&mut self) {
        let old_chartab = self.character_table();
        let class_symbols = self.conjugacy_class_symbols();
        let (char_arr, sorted_fs) = sort_perm_irreps(
            &old_chartab.array().view(),
            &old_chartab.frobenius_schurs.values().copied().collect_vec(),
        );
        let ordered_irreps = deduce_permutation_irrep_symbols(&char_arr.view());
        self.set_irrep_character_table(RepCharacterTable::new(
            &old_chartab.name,
            &ordered_irreps,
            &class_symbols.keys().cloned().collect::<Vec<_>>(),
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
    type GroupElement = Permutation;
    type ElementCollection = PermutationIterator;

    fn name(&self) -> String {
        format!("Sym({})", self.rank)
    }

    fn finite_subgroup_name(&self) -> Option<&String> {
        None
    }

    fn get_index(&self, index: usize) -> Option<Self::GroupElement> {
        let perm = Permutation::from_lehmer_index(index, self.rank);
        if perm.rank() != self.rank {
            None
        } else {
            Some(perm)
        }
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
        self.perms_iter
            .clone()
            .into_iter()
            .enumerate()
            .all(|(i, gi)| {
                (0..i).all(|j| {
                    let gj = self
                        .get_index(j)
                        .unwrap_or_else(|| panic!("Element with index `{j}` not found."));
                    (&gi) * (&gj) == (&gj) * (&gi)
                })
            })
    }

    fn order(&self) -> usize {
        self.perms_iter.clone().into_iter().count()
    }

    fn cayley_table(&self) -> Option<&Array2<usize>> {
        None
    }
}

impl ClassProperties for PermutationGroup {
    type ClassSymbol = PermutationClassSymbol;

    /// Computes the class structure of this permutation group based on cycle patterns.
    fn compute_class_structure(&mut self) {
        log::debug!("Finding unitary conjugacy classes using permutation cycle patterns...");

        let order = self.order();
        let mut cycle_patterns: IndexMap<Vec<u8>, HashSet<usize>> = IndexMap::new();
        let mut e2ccs = vec![Some(0usize); order];
        for (i, element) in self.elements().clone().into_iter().enumerate() {
            let cycle_pattern = element.cycle_pattern();
            if let Vacant(class) = cycle_patterns.entry(cycle_pattern.clone()) {
                class.insert(HashSet::from([i]));
            } else {
                cycle_patterns.get_mut(&cycle_pattern).unwrap().insert(i);
            }
            let cc_idx = cycle_patterns.get_index_of(&cycle_pattern).unwrap();
            e2ccs[i] = Some(cc_idx);
        }
        let ccs = cycle_patterns.into_values().collect_vec();

        let class_number = ccs.len();
        let class_structure = ClassStructure::<Permutation, PermutationClassSymbol>::new_no_ctb(
            self,
            ccs,
            e2ccs,
            (0..class_number).into_iter().collect_vec(),
        );
        self.class_structure = Some(class_structure);

        log::debug!("Finding unitary conjugacy classes using permutation cycle patterns... Done.");
    }

    fn class_structure(&self) -> &ClassStructure<Permutation, PermutationClassSymbol> {
        self.class_structure
            .as_ref()
            .expect("Class structure not found for this group.")
    }

    fn class_structure_mut(&mut self) -> &mut ClassStructure<Permutation, PermutationClassSymbol> {
        self.class_structure
            .as_mut()
            .expect("Class structure not found for this group.")
    }
}

impl CharacterProperties for PermutationGroup {
    type RowSymbol = PermutationIrrepSymbol;
    type CharTab = RepCharacterTable<PermutationIrrepSymbol, PermutationClassSymbol>;

    fn character_table(&self) -> &Self::CharTab {
        self.irrep_character_table
            .as_ref()
            .expect("Irrep character table not found for this group.")
    }
}

impl IrrepCharTabConstruction for PermutationGroup {
    fn set_irrep_character_table(&mut self, chartab: Self::CharTab) {
        self.irrep_character_table = Some(chartab)
    }
}

impl PermutationGroupProperties for PermutationGroup {
    fn from_rank(rank: u8) -> Self {
        assert!(rank > 0, "A permutation rank must be a positive integer.");
        log::debug!("Generating all permutations of rank {rank}...");
        let perms_iter = PermutationIterator {
            rank,
            raw_perm_indices: (0..usize::from(rank).checked_factorial().unwrap()),
        };
        log::debug!("Generating all permutations of rank {rank}... Done.");
        let mut group = PermutationGroup::builder()
            .rank(rank)
            .perms_iter(perms_iter)
            .build()
            .expect("Unable to construct a `PermutationGroup`.");
        log::debug!("Collecting all permutations into a permutation group... Done.");
        group.compute_class_structure();
        group.set_class_symbols_from_cycle_patterns();
        group.construct_irrep_character_table();
        group.canonicalise_character_table();
        group
    }

    fn canonicalise_character_table(&mut self) {
        let old_chartab = self.character_table();
        let class_symbols = self.conjugacy_class_symbols();
        let (char_arr, sorted_fs) = sort_perm_irreps(
            &old_chartab.array().view(),
            &old_chartab.frobenius_schurs.values().copied().collect_vec(),
        );
        let ordered_irreps = deduce_permutation_irrep_symbols(&char_arr.view());
        self.set_irrep_character_table(RepCharacterTable::new(
            &old_chartab.name,
            &ordered_irreps,
            &class_symbols.keys().cloned().collect::<Vec<_>>(),
            &[],
            char_arr,
            &sorted_fs,
        ));
    }
}
