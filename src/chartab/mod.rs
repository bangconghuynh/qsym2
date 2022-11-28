use std::cmp::max;
use std::fmt;
use std::iter;

use derive_builder::Builder;
use indexmap::IndexMap;
use ndarray::{Array2, ArrayView1};

use crate::chartab::character::Character;
use crate::symmetry::symmetry_symbols::{
    ClassSymbol, MathematicalSymbol, MullikenIrrepSymbol, FROBENIUS_SCHUR_SYMBOLS,
};

pub mod character;
pub mod modular_linalg;
pub mod reducedint;
pub mod unityroot;

/// A struct to manage character tables.
#[derive(Builder, Debug, Clone)]
pub struct CharacterTable<R: Clone> {
    /// The name given to the character table.
    name: String,

    /// The irreducible representations of the group and their row indices in the character
    /// table.
    pub irreps: IndexMap<MullikenIrrepSymbol, usize>,

    /// The conjugacy classes of the group and their column indices in the character table.
    pub classes: IndexMap<ClassSymbol<R>, usize>,

    /// The characters of the irreducible representations in this group.
    pub characters: Array2<Character>,

    /// The Frobenius--Schur indicators for the irreducible representations in this group.
    frobenius_schurs: IndexMap<MullikenIrrepSymbol, i8>,

    /// The order of the group.
    #[builder(setter(skip), default = "self.order()")]
    order: usize,
}

impl<R: Clone> CharacterTableBuilder<R> {
    fn order(&self) -> usize {
        self.classes
            .as_ref()
            .unwrap()
            .keys()
            .map(|cc| cc.multiplicity().unwrap())
            .sum()
    }
}

impl<R: Clone> CharacterTable<R> {
    fn builder() -> CharacterTableBuilder<R> {
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
        classes: &[ClassSymbol<R>],
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

        let frobenius_schurs_indexmap = iter::zip(irreps, frobenius_schurs)
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
    fn get_character(&self, irrep: &MullikenIrrepSymbol, class: &ClassSymbol<R>) -> &Character {
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
    fn get_class(&self, class: &ClassSymbol<R>) -> ArrayView1<Character> {
        let col = self.classes.get(class).unwrap();
        self.characters.column(*col)
    }

    /// Prints the character as a nicely formatted string.
    ///
    /// # Arguments
    ///
    /// * compact - Flag indicating if the columns are compact with unequal widths or expanded with
    /// all equal widths.
    /// * numerical - An option containing a non-negative integer specifying the number of decimal
    /// places for the numerical forms of the characters. If `None`, the characters will be shown
    /// as exact algebraic forms.
    ///
    /// # Returns
    ///
    /// A formatted string containing the character table in a printable form.
    pub fn print_nice_table(&self, compact: bool, numerical: Option<usize>) -> String {
        let group_order: usize = self
            .classes
            .keys()
            .map(|cc| cc.multiplicity().unwrap())
            .sum();

        let name = format!("{} ({})", self.name, group_order);
        let chars_str = self.characters.map(|character| {
            if let Some(precision) = numerical {
                let real_only = self.characters.iter().all(|character| {
                    approx::relative_eq!(
                        character.complex_value().im,
                        0.0,
                        epsilon = character.threshold,
                        max_relative = character.threshold
                    )
                });
                character.get_numerical(real_only, precision)
            } else {
                character.to_string()
            }
        });
        let irreps_str: Vec<_> = self.irreps.keys().map(|irrep| irrep.to_string()).collect();
        let ccs_str: Vec<_> = self.classes.keys().map(|cc| cc.to_string()).collect();

        let first_width = max(
            irreps_str
                .iter()
                .map(|irrep_str| irrep_str.chars().count())
                .max()
                .unwrap(),
            name.chars().count(),
        ) + 1;

        let digit_widths: Vec<_> = if compact {
            iter::zip(chars_str.columns(), &ccs_str)
                .map(|(chars_col_str, cc_str)| {
                    let char_width = chars_col_str
                        .iter()
                        .map(|c| c.chars().count())
                        .max()
                        .unwrap();
                    let cc_width = cc_str.chars().count();
                    max(char_width, cc_width) + 1
                })
                .collect()
        } else {
            let char_width = chars_str.iter().map(|c| c.chars().count()).max().unwrap();
            let cc_width = ccs_str.iter().map(|cc| cc.chars().count()).max().unwrap();
            let fixed_width = max(char_width, cc_width) + 1;
            iter::repeat(fixed_width).take(ccs_str.len()).collect()
        };

        // Table heading
        let mut stg = format!(" {:^first_width$} ┆ FS ║", name);
        ccs_str.iter().enumerate().for_each(|(i, cc)| {
            stg.push_str(&format!("{:>width$} │", cc, width = digit_widths[i]));
        });
        stg.pop();
        let tab_width = stg.chars().count();
        stg = format!(
            "{}\n{}\n{}\n",
            iter::repeat("━").take(tab_width).collect::<String>(),
            stg,
            iter::repeat("┈").take(tab_width).collect::<String>(),
        );

        // Table body
        let rows =
            iter::zip(self.irreps.keys(), irreps_str)
                .enumerate()
                .map(|(i, (irrep, irrep_str))| {
                    let fs = FROBENIUS_SCHUR_SYMBOLS
                        .get(self.frobenius_schurs.get(irrep).unwrap())
                        .unwrap();
                    let mut line = format!(" {:<first_width$} ┆ {:>2} ║", irrep_str, fs);

                    let line_chars: String = itertools::Itertools::intersperse(
                        ccs_str.iter().enumerate().map(|(j, _)| {
                            format!("{:>width$}", chars_str[[i, j]], width = digit_widths[j])
                        }),
                        " │".to_string(),
                    )
                    .collect();

                    line.push_str(&line_chars);
                    line
                });

        stg.push_str(
            &itertools::Itertools::intersperse(rows, "\n".to_string()).collect::<String>(),
        );

        // Table bottom
        stg.push_str("\n");
        stg.push_str(&iter::repeat("━").take(tab_width).collect::<String>());
        stg.push_str("\n");
        stg
    }
}

// -------
// Display
// -------
impl<R: Clone> fmt::Display for CharacterTable<R> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.print_nice_table(true, None))
    }
}
