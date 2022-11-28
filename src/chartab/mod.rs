use std::cmp::max;
use std::iter::zip;

use derive_builder::Builder;
use indexmap::IndexMap;
use ndarray::{Array2, ArrayView1};

use crate::chartab::character::Character;
use crate::symmetry::symmetry_symbols::{ClassSymbol, MathematicalSymbol, MullikenIrrepSymbol};

pub mod character;
pub mod modular_linalg;
pub mod reducedint;
pub mod unityroot;

/// A struct to manage character tables.
#[derive(Builder, Debug, Clone)]
pub struct CharacterTable<T: Clone> {
    /// The name given to the character table.
    name: String,

    /// The irreducible representations of the group and their row indices in the character
    /// table.
    irreps: IndexMap<MullikenIrrepSymbol, usize>,

    /// The conjugacy classes of the group and their column indices in the character table.
    classes: IndexMap<ClassSymbol<T>, usize>,

    /// The characters of the irreducible representations in this group.
    characters: Array2<Character>,

    /// The Frobenius--Schur indicators for the irreducible representations in this group.
    frobenius_schurs: IndexMap<MullikenIrrepSymbol, i8>,

    /// The order of the group.
    #[builder(setter(skip), default = "self.order()")]
    order: usize,
}

impl<T: Clone> CharacterTableBuilder<T> {
    fn order(&self) -> usize {
        self.classes
            .as_ref()
            .unwrap()
            .keys()
            .map(|cc| cc.multiplicity().unwrap())
            .sum()
    }
}

impl<T: Clone> CharacterTable<T> {
    fn builder() -> CharacterTableBuilder<T> {
        CharacterTableBuilder::default()
    }

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
    pub fn new(
        name: &str,
        irreps: &[MullikenIrrepSymbol],
        classes: &[ClassSymbol<T>],
        char_arr: Array2<Character>,
        frobenius_schurs: &[i8],
    ) -> Self {
        assert_eq!(irreps.len(), char_arr.dim().0);
        assert_eq!(frobenius_schurs.len(), char_arr.dim().0);
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

        let frobenius_schurs_indexmap = zip(irreps, frobenius_schurs)
            .map(|(irrep, &fsi)| (irrep.clone(), fsi))
            .collect::<IndexMap<_, _>>();

        Self::builder()
            .name(name.to_string())
            .irreps(irreps_indexmap)
            .classes(classes_indexmap)
            .characters(char_arr)
            .frobenius_schurs(frobenius_schurs_indexmap)
            .build()
            .unwrap()
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
    fn get_character(&self, irrep: &MullikenIrrepSymbol, class: &ClassSymbol<T>) -> &Character {
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
    fn get_class(&self, class: &ClassSymbol<T>) -> ArrayView1<Character> {
        let col = self.classes.get(class).unwrap();
        self.characters.column(*col)
    }

    /// Prints the character as a nicely formatted string.
    ///
    /// # Arguments
    ///
    /// * numerical - An option containing a non-negative integer specifying the number of decimal
    /// places for the numerical forms of the characters. If `None`, the characters will be shown
    /// as exact algebraic forms.
    ///
    /// # Returns
    ///
    /// A formatted string containing the character table in a printable form.
    fn print_nice_table(&self, numerical: Option<u8>) -> String {
        // let first_width = max(
        //     self.irreps.keys().map(|irrep| irrep.to_string().len()).max(),
        //     format!("{} ({})", self.name, self)
        // )
        todo!()
    }
}
