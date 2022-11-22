use derive_builder::Builder;
use indexmap::IndexMap;
use std::fmt::Display;

use crate::chartab::character::Character;
use crate::symmetry::symmetry_element::symmetry_operation::SymmetryOperation;
use crate::symmetry::symmetry_symbols::{ClassSymbol, MathematicalSymbol, MullikenIrrepSymbol};

mod character;
mod modular_linalg;
mod reducedint;
mod unityroot;

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
    chartab: IndexMap<(R, C), Character>,
}

impl<R, C> CharacterTable<R, C>
where
    R: MathematicalSymbol + Display,
    C: MathematicalSymbol + Display,
{
    fn get_character(&self, row: &R, col: &C) -> &Character {
        self.chartab
            .iter()
            .find_map(|((r, c), character)| {
                if r == row && c == col {
                    Some(character)
                } else {
                    None
                }
            })
            .expect(format!("No character can be found at ({}, {}).", row, col).as_str())
    }

    fn get_row(&self, row: &R) -> IndexMap<&C, &Character> {
        self.chartab
            .iter()
            .filter_map(
                |((r, c), character)| {
                    if r == row {
                        Some((c, character))
                    } else {
                        None
                    }
                },
            )
            .collect()
    }

    fn get_col(&self, col: &C) -> IndexMap<&R, &Character> {
        self.chartab
            .iter()
            .filter_map(
                |((r, c), character)| {
                    if c == col {
                        Some((r, character))
                    } else {
                        None
                    }
                },
            )
            .collect()
    }
}
