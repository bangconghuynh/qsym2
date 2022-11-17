use derive_builder::Builder;
use indexmap::IndexMap;

use crate::chartab::character::Character;
use crate::symmetry::symmetry_element::symmetry_operation::SymmetryOperation;
use crate::symmetry::symmetry_symbols::{ClassSymbol, MathematicalSymbol, MullikenIrrepSymbol};

mod character;
mod modular_linalg;
mod reducedint;
mod unityroot;

type RowMap<C> = IndexMap<C, Character>;
type ColMap<R> = IndexMap<R, Character>;

/// A struct to manage character tables.
#[derive(Builder)]
struct CharacterTable<R, C>
where
    R: MathematicalSymbol,
    C: MathematicalSymbol,
{
    /// The name given to the character table.
    name: String,

    /// The characters of the irreducible representations in this group.
    #[builder(setter(custom))]
    chartab: IndexMap<R, RowMap<C>>,
}

impl<R, C> CharacterTable<R, C>
where
    R: MathematicalSymbol,
    C: MathematicalSymbol,
{
    fn get_character(&self, row: &R, col: &C) -> &Character {
        self.chartab.get(row).unwrap().get(col).unwrap()
    }

    fn get_row(&self, row: &R) -> RowMap<C> {
        self.chartab.get(row).unwrap().clone()
    }

    fn get_col(&self, col: &C) -> ColMap<R> {
        self.chartab
            .keys()
            .map(|row| {
                (
                    row.clone(),
                    self.chartab.get(row).unwrap().get(col).unwrap().clone(),
                )
            })
            .collect::<ColMap<R>>()
    }
}
