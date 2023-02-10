use std::collections::HashSet;

use derive_builder::Builder;
use indexmap::map::Entry::Vacant;
use indexmap::IndexMap;
use itertools::Itertools;
use ndarray::Array2;
use rayon::prelude::*;

use crate::chartab::chartab_group::{CharacterProperties, IrrepCharTabConstruction};
use crate::chartab::chartab_symbols::CollectionSymbol;
use crate::chartab::{CharacterTable, RepCharacterTable};
use crate::group::class::{ClassProperties, ClassStructure};
use crate::group::{Group, GroupProperties, UnitaryRepresentedGroup};
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
                let mut cycle_pattern_count: Vec<(usize, usize)> =
                    Vec::with_capacity(cycle_pattern.len());
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
        let mut group = UnitaryRepresentedGroup::<
            Permutation,
            PermutationIrrepSymbol,
            PermutationClassSymbol,
        >::new(format!("Sym({rank})").as_str(), perms);
        group.set_class_symbols_from_cycle_patterns();
        group.construct_irrep_character_table();
        group.canonicalise_character_table();
        group
    }

    fn canonicalise_character_table(&mut self) {
        let old_chartab = self.character_table();
        let class_symbols = self.conjugacy_class_symbols();
        let (char_arr, sorted_fs) = sort_irreps(
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

/// A structure for managing permutation groups efficiently.
#[derive(Clone, Builder)]
pub struct PermutationGroup {
    /// The rank of the permutation group.
    rank: usize,

    /// The underlying abstract group of this permutation group.
    abstract_group: Group<Permutation>,

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

impl GroupProperties for PermutationGroup {
    type GroupElement = Permutation;

    fn abstract_group(&self) -> &Group<Self::GroupElement> {
        &self.abstract_group
    }

    fn name(&self) -> String {
        format!("Sym({})", self.rank)
    }

    fn finite_subgroup_name(&self) -> Option<&String> {
        None
    }
}

impl ClassProperties for PermutationGroup {
    type ClassSymbol = PermutationClassSymbol;

    /// Computes the class structure of this permutation group based on cycle patterns.
    fn compute_class_structure(&mut self) {
        log::debug!("Finding unitary conjugacy classes using permutation cycle patterns...");

        let order = self.abstract_group.order();
        let mut cycle_patterns: IndexMap<Vec<usize>, HashSet<usize>> = IndexMap::new();
        let mut e2ccs = vec![Some(0usize); order];
        for (element, &i) in self.abstract_group.elements().iter() {
            let cycle_pattern = element.cycle_pattern();
            if let Vacant(class) = cycle_patterns.entry(cycle_pattern.clone()) {
                class.insert(HashSet::from([i]));
            } else {
                cycle_patterns.get_mut(&cycle_pattern).unwrap().insert(i);
            }
            let cc_idx = cycle_patterns.get_index_of(&cycle_pattern).unwrap();
            e2ccs[i] = Some(cc_idx);
        }
        let ccs = cycle_patterns.values().cloned().collect_vec();

        let class_number = ccs.len();
        let class_structure = ClassStructure::<Permutation, PermutationClassSymbol>::new_no_ctb(
            &self.abstract_group,
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

    /// Obtains the class matrix $`\mathbf{N}_r`$ for the conjugacy classes in the group without
    /// using the Cayley table.
    fn class_matrix(&self, ctb_opt: Option<&Array2<usize>>, r: usize) -> Array2<usize> {
        log::debug!("Constructing N{r}...");
        assert!(ctb_opt.is_none());
        let class_number = self.conjugacy_classes().len();
        let mut nmat_r = Array2::<usize>::zeros((class_number, class_number));
        let class_r = &self.conjugacy_classes()[r];
        for (s, class_s) in self.conjugacy_classes().iter().enumerate() {
            for (t, class_t) in self.conjugacy_classes().iter().enumerate() {
                let rep_t_idx = *class_t
                    .iter()
                    .next()
                    .expect("No conjugacy classes can be empty.");
                nmat_r[[s, t]] = class_r
                    .iter()
                    .cartesian_product(class_s.iter())
                    .par_bridge()
                    .filter(|(&r_idx, &s_idx)| {
                        let r = self
                            .abstract_group
                            .elements()
                            .get_index(r_idx)
                            .unwrap_or_else(|| panic!("No element with index `{r_idx}` found."))
                            .0;
                        let s = self
                            .abstract_group()
                            .elements()
                            .get_index(s_idx)
                            .unwrap_or_else(|| panic!("No element with index `{s_idx}` found."))
                            .0;
                        let t = r * s;
                        let t_idx = self.abstract_group.elements().get_index_of(&t).unwrap();
                        t_idx == rep_t_idx
                    })
                    .count();
            }
        }
        log::debug!("Constructing N{r}... Done.");
        nmat_r
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
    fn from_rank(rank: usize) -> Self {
        let perms = (0..rank)
            .permutations(rank)
            .map(|image| Permutation::from_image(&image))
            .collect_vec();
        let abstract_group =
            Group::<Permutation>::new_no_ctb(format!("Sym({rank})").as_str(), perms);
        let mut group = PermutationGroup::builder()
            .rank(rank)
            .abstract_group(abstract_group)
            .build()
            .expect("Unable to construct a `PermutationGroup`.");
        group.compute_class_structure();
        group.set_class_symbols_from_cycle_patterns();
        group.construct_irrep_character_table();
        group.canonicalise_character_table();
        group
    }

    fn canonicalise_character_table(&mut self) {
        let old_chartab = self.character_table();
        let class_symbols = self.conjugacy_class_symbols();
        let (char_arr, sorted_fs) = sort_irreps(
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
