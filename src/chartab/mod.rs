use derive_builder::Builder;
use indexmap::IndexMap;

use crate::chartab::character::Character;
use crate::symmetry::symmetry_element::symmetry_operation::SymmetryOperation;
use crate::symmetry::symmetry_symbols::{ClassSymbol, MullikenIrrepSymbol};

mod character;
mod modular_linalg;
mod reducedint;
mod unityroot;

type IrrepMap = IndexMap<ClassSymbol<SymmetryOperation>, Character>;

/// A struct to manage character tables.
#[derive(Builder)]
struct CharacterTable {
    /// The name given to the character table.
    name: String,

    /// The characters of the irreducible representations in this group.
    chartab: IndexMap<MullikenIrrepSymbol, IrrepMap>,
}

impl CharacterTable {
    
}
