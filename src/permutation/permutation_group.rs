use itertools::Itertools;

use crate::chartab::chartab_group::{CharacterProperties, IrrepCharTabConstruction};
use crate::chartab::chartab_symbols::CollectionSymbol;
use crate::chartab::{CharacterTable, RepCharacterTable};
use crate::group::class::ClassProperties;
use crate::group::UnitaryRepresentedGroup;
use crate::permutation::permutation_symbols::{
    deduce_permutation_irrep_symbols, sort_irreps, PermutationClassSymbol, PermutationIrrepSymbol,
};
use crate::permutation::Permutation;

#[cfg(test)]
#[path = "permutation_group_tests.rs"]
mod permutation_group_tests;

pub trait PermutationGroupProperties:
    ClassProperties<GroupElement = Permutation, ClassSymbol = PermutationClassSymbol>
    + CharacterProperties
{
    /// Constructs a group from molecular symmetry *elements* (not operations).
    ///
    /// # Arguments
    ///
    /// * `sym` - A molecular symmetry struct.
    /// * `infinite_order_to_finite` - Interpret infinite-order generating
    /// elements as finite-order generating elements to create a finite subgroup
    /// of an otherwise infinite group.
    ///
    /// # Returns
    ///
    /// A finite group of symmetry operations.
    fn from_rank(rank: usize) -> Self;

    fn set_class_symbols_from_cycle_patterns(&mut self);

    /// Reorders and relabels the rows and columns of the constructed character table using
    /// symmetry-specific rules and conventions.
    fn canonicalise_character_table(&mut self);
}

impl PermutationGroupProperties
    for UnitaryRepresentedGroup<Permutation, PermutationIrrepSymbol, PermutationClassSymbol>
{
    fn from_rank(rank: usize) -> Self {
        let perms = (0..rank)
            .permutations(rank)
            .map(|image| Permutation::from_image(&image))
            .collect_vec();
        let mut group = UnitaryRepresentedGroup::<Permutation, PermutationIrrepSymbol, PermutationClassSymbol>::new(
            format!("Sym({rank})").as_str(),
            perms,
        );
        group.set_class_symbols_from_cycle_patterns();
        group.construct_irrep_character_table();
        group.canonicalise_character_table();
        group
    }

    fn set_class_symbols_from_cycle_patterns(&mut self) {
        log::debug!("Assigning class symbols from cycle patterns...");
        let class_symbols = self
            .conjugacy_class_symbols()
            .iter()
            .map(|(old_symbol, _)| {
                let rep_ele = old_symbol
                    .representative()
                    .unwrap_or_else(|| {
                        panic!(
                            "No representative element found for conjugacy class `{old_symbol}`."
                        )
                    });
                let cycle_pattern = rep_ele
                    .cycle_pattern()
                    .clone();
                let mut cycle_pattern_count: Vec<(usize, usize)> = Vec::with_capacity(cycle_pattern.len());
                let mut i = 0usize;
                while i < cycle_pattern.len() {
                    let mut j = i + 1;
                    while j < cycle_pattern.len() && cycle_pattern[j] == cycle_pattern[i] {
                        j += 1;
                    }
                    cycle_pattern_count.push((cycle_pattern[i], j - i));
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
                    panic!("Unable to construct a class symbol from `{size}||{cycle_pattern_str}||`")
                })
            })
            .collect_vec();
        self.class_structure_mut().set_class_symbols(&class_symbols);
        log::debug!("Assigning class symbols from cycle patterns... Done.");
    }

    fn canonicalise_character_table(&mut self) {
        let old_chartab = self.character_table();
        let class_symbols = self.conjugacy_class_symbols();
        let (char_arr, sorted_fs) = sort_irreps(
            &old_chartab.array().view(),
            &old_chartab.frobenius_schurs.values().copied().collect_vec(),
        );
        let ordered_irreps = deduce_permutation_irrep_symbols(&char_arr.view());
        self.irrep_character_table = Some(RepCharacterTable::new(
            &old_chartab.name,
            &ordered_irreps,
            &class_symbols.keys().cloned().collect::<Vec<_>>(),
            &[],
            char_arr,
            &sorted_fs,
        ));
    }
}
