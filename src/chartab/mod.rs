use indexmap::{IndexMap, IndexSet};
use ndarray::{Array2, ArrayView1};

use crate::chartab::character::Character;
use crate::symmetry::symmetry_element::symmetry_operation::SymmetryOperation;
use crate::symmetry::symmetry_symbols::{ClassSymbol, MullikenIrrepSymbol};

mod character;
mod modular_linalg;
mod reducedint;
mod unityroot;

/// A struct to manage character tables.
struct CharacterTable {
    /// The name given to the character table.
    name: String,

    /// The irreducible representations of the group and their row indices in the character
    /// table.
    irreps: IndexMap<MullikenIrrepSymbol, usize>,

    /// The conjugacy classes of the group and their column indices in the character table.
    classes: IndexMap<ClassSymbol<SymmetryOperation>, usize>,

    /// The characters of the irreducible representations in this group, laid out in row-major
    /// ordering.
    characters: Array2<Character>,
}

impl CharacterTable {
    /// Constructs a new character table.
    ///
    /// # Arguments
    ///
    /// * name - A name given to the character table.
    /// * irreps - A slice of Mulliken irreducible representation symbols in the right order.
    /// * classes - A slice of conjugacy class symbols in the right order.
    /// * char_arr - A two-dimensional array of characters,
    ///
    /// # Returns
    ///
    /// The required character.
    fn new(
        name: &str,
        irreps: &[MullikenIrrepSymbol],
        classes: &[ClassSymbol<SymmetryOperation>],
        char_arr: Array2<Character>,
    ) -> Self {
        assert_eq!(irreps.len(), char_arr.dim().0);
        assert_eq!(classes.len(), char_arr.dim().1);
        assert_eq!(char_arr.dim().0, char_arr.dim().1);

        let irreps_indexmap = irreps
            .iter()
            .cloned()
            .enumerate()
            .map(|(i, irrep)| (irrep, i))
            .collect::<IndexMap<_, _>>();

        let classes_indexmap = classes
            .iter()
            .cloned()
            .enumerate()
            .map(|(i, class)| (class, i))
            .collect::<IndexMap<_, _>>();

        Self {
            name: name.to_string(),
            irreps: irreps_indexmap,
            classes: classes_indexmap,
            characters: char_arr,
        }
    }

    /// Retrieves the character of a particular irreducible representation in a particular
    /// conjugacy class.
    ///
    /// # Arguments
    ///
    /// * irrep - A Mulliken irreducible representation symbol.
    /// * class - A conjugacy class symbol.
    ///
    /// # Returns
    ///
    /// The required character.
    fn get_character(
        &self,
        irrep: &MullikenIrrepSymbol,
        class: &ClassSymbol<SymmetryOperation>,
    ) -> &Character {
        let row = self.irreps.get(irrep).unwrap();
        let col = self.classes.get(class).unwrap();
        &self.characters[(*row, *col)]
    }

    /// Retrieves the characters of all conjugacy classes in a particular irreducible
    /// representation.
    ///
    /// # Arguments
    ///
    /// * irrep - A Mulliken irreducible representation symbol.
    ///
    /// # Returns
    ///
    /// The required characters.
    fn get_irrep(&self, irrep: &MullikenIrrepSymbol) -> ArrayView1<Character> {
        let row = self.irreps.get(irrep).unwrap();
        self.characters.row(*row)
    }

    /// Retrieves the characters of all irreducible representations in a particular conjugacy
    /// class.
    ///
    /// # Arguments
    ///
    /// * class - A conjugacy class symbol.
    ///
    /// # Returns
    ///
    /// The required characters.
    fn get_class(&self, class: &ClassSymbol<SymmetryOperation>) -> ArrayView1<Character> {
        let col = self.classes.get(class).unwrap();
        self.characters.column(*col)
    }
}
