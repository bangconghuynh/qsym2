use std::cmp::max;
use std::fmt;
use std::iter;

use derive_builder::Builder;
use indexmap::{IndexMap, IndexSet};
use ndarray::{Array2, ArrayView1};

use crate::chartab::character::Character;
use crate::chartab::chartab_symbols::{
    CollectionSymbol, LinearSpaceSymbol, MathematicalSymbol, ReducibleLinearSpaceSymbol,
    FROBENIUS_SCHUR_SYMBOLS,
};

pub mod character;
pub mod chartab_group;
pub mod chartab_symbols;
pub mod modular_linalg;
pub mod reducedint;
pub mod unityroot;

/// A trait to contain essential methods for a character table.
pub trait CharacterTable: Clone
where
    Self::RowSymbol: LinearSpaceSymbol,
    Self::ColSymbol: CollectionSymbol,
{
    /// The type for the row-labelling symbols.
    type RowSymbol;

    /// The type for the column-labelling symbols.
    type ColSymbol;

    /// Retrieves the character of a particular irreducible representation in a particular
    /// conjugacy class.
    ///
    /// # Arguments
    ///
    /// * `irrep` - A Mulliken irreducible representation symbol.
    /// * `class` - A conjugacy class symbol.
    ///
    /// # Returns
    ///
    /// The required character.
    ///
    /// # Panics
    ///
    /// Panics if the specified `irrep` or `class` cannot be found.
    fn get_character(&self, irrep: &Self::RowSymbol, class: &Self::ColSymbol) -> &Character;

    /// Retrieves the characters of all columns in a particular row.
    ///
    /// # Arguments
    ///
    /// * `row` - A row-labelling symbol.
    ///
    /// # Returns
    ///
    /// The required characters.
    fn get_row(&self, row: &Self::RowSymbol) -> ArrayView1<Character>;

    /// Retrieves the characters of all rows in a particular column.
    ///
    /// # Arguments
    ///
    /// * `col` - A column-labelling symbol.
    ///
    /// # Returns
    ///
    /// The required characters.
    fn get_col(&self, col: &Self::ColSymbol) -> ArrayView1<Character>;

    /// Retrieves the symbols of all rows in the character table.
    fn get_all_rows(&self) -> IndexSet<Self::RowSymbol>;

    /// Retrieves the symbols of all columns in the character table.
    fn get_all_cols(&self) -> IndexSet<Self::ColSymbol>;

    /// Returns a shared reference to the underlying array of the character table.
    fn array(&self) -> &Array2<Character>;

    /// Retrieves the order of the group.
    fn get_order(&self) -> usize;

    /// Returns the principal columns of the character table.
    fn get_principal_cols(&self) -> &IndexSet<Self::ColSymbol>;

    /// Prints a nicely formatted character table.
    ///
    /// # Arguments
    ///
    /// * `compact` - Flag indicating if the columns are compact with unequal widths or expanded
    /// with all equal widths.
    /// * `numerical` - An option containing a non-negative integer specifying the number of decimal
    /// places for the numerical forms of the characters. If `None`, the characters will be shown
    /// as exact algebraic forms.
    ///
    /// # Returns
    ///
    /// A formatted string containing the character table in a printable form.
    ///
    /// # Errors
    ///
    /// Returns an error when encountering any issue formatting the character table.
    fn write_nice_table(
        &self,
        f: &mut fmt::Formatter,
        compact: bool,
        numerical: Option<usize>,
    ) -> fmt::Result;
}

// =================
// RepCharacterTable
// =================

/// A struct to manage character tables of irreducible representations.
#[derive(Builder, Clone)]
pub struct RepCharacterTable<RowSymbol, ColSymbol>
where
    RowSymbol: LinearSpaceSymbol,
    ColSymbol: CollectionSymbol,
{
    /// The name given to the character table.
    pub name: String,

    /// The irreducible representations of the group and their row indices in the character
    /// table.
    pub irreps: IndexMap<RowSymbol, usize>,

    /// The conjugacy classes of the group and their column indices in the character table.
    pub classes: IndexMap<ColSymbol, usize>,

    /// The principal conjugacy classes of the group.
    principal_classes: IndexSet<ColSymbol>,

    /// The characters of the irreducible representations in this group.
    pub characters: Array2<Character>,

    /// The Frobenius--Schur indicators for the irreducible representations in this group.
    pub frobenius_schurs: IndexMap<RowSymbol, i8>,
}

impl<RowSymbol, ColSymbol> RepCharacterTable<RowSymbol, ColSymbol>
where
    RowSymbol: LinearSpaceSymbol,
    ColSymbol: CollectionSymbol,
{
    fn builder() -> RepCharacterTableBuilder<RowSymbol, ColSymbol> {
        RepCharacterTableBuilder::default()
    }

    /// Constructs a new character table of irreducible representations.
    ///
    /// # Arguments
    ///
    /// * `name` - A name given to the character table.
    /// * `irreps` - A slice of Mulliken irreducible representation symbols in the right order.
    /// * `classes` - A slice of conjugacy class symbols in the right order.
    /// * `char_arr` - A two-dimensional array of characters,
    ///
    /// # Returns
    ///
    /// A character table.
    ///
    /// # Panics
    ///
    /// Panics if the character table cannot be constructed.
    pub fn new(
        name: &str,
        irreps: &[RowSymbol],
        classes: &[ColSymbol],
        principal_classes: &[ColSymbol],
        char_arr: Array2<Character>,
        frobenius_schurs: &[i8],
    ) -> Self {
        assert_eq!(irreps.len(), char_arr.dim().0);
        assert_eq!(frobenius_schurs.len(), char_arr.dim().0);
        assert_eq!(classes.len(), char_arr.dim().1);
        assert_eq!(char_arr.dim().0, char_arr.dim().1);

        let irreps_indexmap: IndexMap<RowSymbol, usize> = irreps
            .iter()
            .cloned()
            .enumerate()
            .map(|(i, irrep)| (irrep, i))
            .collect();

        let classes_indexmap: IndexMap<ColSymbol, usize> = classes
            .iter()
            .cloned()
            .enumerate()
            .map(|(i, class)| (class, i))
            .collect();

        let principal_classes_indexset: IndexSet<ColSymbol> =
            principal_classes.iter().cloned().collect();

        let frobenius_schurs_indexmap = iter::zip(irreps, frobenius_schurs)
            .map(|(irrep, &fsi)| (irrep.clone(), fsi))
            .collect::<IndexMap<_, _>>();

        Self::builder()
            .name(name.to_string())
            .irreps(irreps_indexmap)
            .classes(classes_indexmap)
            .principal_classes(principal_classes_indexset)
            .characters(char_arr)
            .frobenius_schurs(frobenius_schurs_indexmap)
            .build()
            .expect("Unable to construct a character table.")
    }
}

impl<RowSymbol, ColSymbol> CharacterTable for RepCharacterTable<RowSymbol, ColSymbol>
where
    RowSymbol: LinearSpaceSymbol,
    ColSymbol: CollectionSymbol,
{
    type RowSymbol = RowSymbol;
    type ColSymbol = ColSymbol;

    /// Retrieves the character of a particular irreducible representation in a particular
    /// conjugacy class.
    ///
    /// # Arguments
    ///
    /// * `irrep` - A Mulliken irreducible representation symbol.
    /// * `class` - A conjugacy class symbol.
    ///
    /// # Returns
    ///
    /// The required character.
    ///
    /// # Panics
    ///
    /// Panics if the specified `irrep` or `class` cannot be found.
    fn get_character(&self, irrep: &Self::RowSymbol, class: &Self::ColSymbol) -> &Character {
        let row = self
            .irreps
            .get(irrep)
            .unwrap_or_else(|| panic!("Irrep {irrep} not found."));
        let col = self
            .classes
            .get(class)
            .unwrap_or_else(|| panic!("Conjugacy class {class} not found."));
        &self.characters[(*row, *col)]
    }

    /// Retrieves the characters of all conjugacy classes in a particular irreducible
    /// representation.
    ///
    /// # Arguments
    ///
    /// * `irrep` - A Mulliken irreducible representation symbol.
    ///
    /// # Returns
    ///
    /// The required characters.
    fn get_row(&self, row: &Self::RowSymbol) -> ArrayView1<Character> {
        let row = self
            .irreps
            .get(row)
            .unwrap_or_else(|| panic!("Irrep {row} not found."));
        self.characters.row(*row)
    }

    /// Retrieves the characters of all irreducible representations in a particular conjugacy
    /// class.
    ///
    /// # Arguments
    ///
    /// * `class` - A conjugacy class symbol.
    ///
    /// # Returns
    ///
    /// The required characters.
    fn get_col(&self, col: &Self::ColSymbol) -> ArrayView1<Character> {
        let col = self
            .classes
            .get(col)
            .unwrap_or_else(|| panic!("Conjugacy class {col} not found."));
        self.characters.column(*col)
    }

    /// Retrieves the Mulliken symbols of all irreducible representations of the group.
    fn get_all_rows(&self) -> IndexSet<Self::RowSymbol> {
        self.irreps.keys().cloned().collect::<IndexSet<_>>()
    }

    /// Retrieves the symbols of all conjugacy classes of the group.
    fn get_all_cols(&self) -> IndexSet<Self::ColSymbol> {
        self.classes.keys().cloned().collect::<IndexSet<_>>()
    }

    /// Returns a shared reference to the underlying array of the character table.
    fn array(&self) -> &Array2<Character> {
        &self.characters
    }

    /// Retrieves the order of the group.
    fn get_order(&self) -> usize {
        self.classes
            .keys()
            .map(|cc| {
                cc.multiplicity().unwrap_or_else(|| {
                    panic!("Unable to find the multiplicity for conjugacy class `{cc}`.")
                })
            })
            .sum()
    }

    /// Prints a nicely formatted character table.
    ///
    /// # Arguments
    ///
    /// * `compact` - Flag indicating if the columns are compact with unequal widths or expanded
    /// with all equal widths.
    /// * `numerical` - An option containing a non-negative integer specifying the number of decimal
    /// places for the numerical forms of the characters. If `None`, the characters will be shown
    /// as exact algebraic forms.
    ///
    /// # Returns
    ///
    /// A formatted string containing the character table in a printable form.
    ///
    /// # Panics
    ///
    /// Panics upon encountering any missing information required for a complete print-out of the
    /// character table.
    ///
    /// # Errors
    ///
    /// Errors upon encountering any issue formatting the character table.
    #[allow(clippy::too_many_lines)]
    fn write_nice_table(
        &self,
        f: &mut fmt::Formatter,
        compact: bool,
        numerical: Option<usize>,
    ) -> fmt::Result {
        let group_order: usize = self
            .classes
            .keys()
            .map(|cc| {
                cc.multiplicity().unwrap_or_else(|| {
                    panic!("Unable to find the multiplicity for conjugacy class `{cc}`.")
                })
            })
            .sum();

        let name = format!("u {} ({group_order})", self.name);
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
        let irreps_str: Vec<_> = self
            .irreps
            .keys()
            .map(std::string::ToString::to_string)
            .collect();
        let ccs_str: Vec<_> = self
            .classes
            .keys()
            .map(|cc| {
                if self.principal_classes.contains(cc) {
                    format!("◈{cc}")
                } else {
                    cc.to_string()
                }
            })
            .collect();

        let first_width = max(
            irreps_str
                .iter()
                .map(|irrep_str| irrep_str.chars().count())
                .max()
                .expect("Unable to find the maximum length for the irrep symbols."),
            name.chars().count(),
        ) + 1;

        let digit_widths: Vec<_> = if compact {
            iter::zip(chars_str.columns(), &ccs_str)
                .map(|(chars_col_str, cc_str)| {
                    let char_width = chars_col_str
                        .iter()
                        .map(|c| c.chars().count())
                        .max()
                        .expect("Unable to find the maximum length for the characters.");
                    let cc_width = cc_str.chars().count();
                    max(char_width, cc_width) + 1
                })
                .collect()
        } else {
            let char_width = chars_str
                .iter()
                .map(|c| c.chars().count())
                .max()
                .expect("Unable to find the maximum length for the characters.");
            let cc_width = ccs_str
                .iter()
                .map(|cc| cc.chars().count())
                .max()
                .expect("Unable to find the maximum length for the conjugacy class symbols.");
            let fixed_width = max(char_width, cc_width) + 1;
            iter::repeat(fixed_width).take(ccs_str.len()).collect()
        };

        // Table heading
        let mut heading = format!(" {name:^first_width$} ┆ FS ║");
        ccs_str.iter().enumerate().for_each(|(i, cc)| {
            heading.push_str(&format!("{cc:>width$} │", width = digit_widths[i]));
        });
        heading.pop();
        let tab_width = heading.chars().count();
        heading = format!(
            "{}\n{}\n{}\n",
            "━".repeat(tab_width),
            heading,
            "┈".repeat(tab_width),
        );
        write!(f, "{heading}")?;

        // Table body
        let rows =
            iter::zip(self.irreps.keys(), irreps_str)
                .enumerate()
                .map(|(i, (irrep, irrep_str))| {
                    let fs = FROBENIUS_SCHUR_SYMBOLS
                        .get(self.frobenius_schurs.get(irrep).unwrap_or_else(|| {
                            panic!(
                            "Unable to obtain the Frobenius--Schur indicator for irrep `{irrep}`."
                        )
                        }))
                        .expect("Unknown Frobenius--Schur symbol.");
                    let mut line = format!(" {irrep_str:<first_width$} ┆ {fs:>2} ║");

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

        write!(
            f,
            "{}",
            &itertools::Itertools::intersperse(rows, "\n".to_string()).collect::<String>(),
        )?;

        // Table bottom
        write!(f, "\n{}\n", &"━".repeat(tab_width))
    }

    fn get_principal_cols(&self) -> &IndexSet<Self::ColSymbol> {
        &self.principal_classes
    }
}

// -------
// Display
// -------
impl<RowSymbol, ColSymbol> fmt::Display for RepCharacterTable<RowSymbol, ColSymbol>
where
    RowSymbol: LinearSpaceSymbol,
    ColSymbol: CollectionSymbol,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.write_nice_table(f, true, Some(3))
    }
}

// -----
// Debug
// -----
impl<RowSymbol, ColSymbol> fmt::Debug for RepCharacterTable<RowSymbol, ColSymbol>
where
    RowSymbol: LinearSpaceSymbol,
    ColSymbol: CollectionSymbol,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.write_nice_table(f, true, None)
    }
}

// ===================
// CorepCharacterTable
// ===================

/// A structure to manage character tables of irreducible corepresentations of magnetic groups.
#[derive(Builder, Clone)]
pub struct CorepCharacterTable<RowSymbol, UC>
where
    RowSymbol: ReducibleLinearSpaceSymbol,
    UC: CharacterTable,
{
    /// The name given to the character table.
    pub name: String,

    /// The character table of the irreducible representations of the halving unitary subgroup that
    /// induce the irreducible corepresentations of the current magnetic group.
    pub unitary_character_table: UC,

    /// The irreducible corepresentations of the group and their row indices in the character
    /// table.
    pub ircoreps: IndexMap<RowSymbol, usize>,

    /// The conjugacy classes of the group and their column indices in the character table.
    pub classes: IndexMap<UC::ColSymbol, usize>,

    /// The principal conjugacy classes of the group.
    principal_classes: IndexSet<UC::ColSymbol>,

    /// The characters of the irreducible corepresentations in this group.
    pub characters: Array2<Character>,

    /// The intertwining numbers of the irreducible corepresentations.
    pub intertwining_numbers: IndexMap<RowSymbol, u8>,
}

impl<RowSymbol, UC> CorepCharacterTable<RowSymbol, UC>
where
    RowSymbol: ReducibleLinearSpaceSymbol,
    UC: CharacterTable,
{
    fn builder() -> CorepCharacterTableBuilder<RowSymbol, UC> {
        CorepCharacterTableBuilder::default()
    }

    /// Constructs a new character table of irreducible corepresentations.
    ///
    /// # Arguments
    ///
    /// * `name` - A name given to the character table.
    /// * `unitary_chartab` - The character table of irreducible representations of the unitary
    /// halving subgroup, which will be owned by this [`CorepCharacterTable`].
    /// * `ircoreps` - A slice of Mulliken irreducible corepresentation symbols in the right order.
    /// * `char_arr` - A two-dimensional array of characters,
    /// * `intertwining_numbers` - A slice of the intertwining numbers of the irreducible
    /// corepresentations in the right order.
    ///
    /// # Returns
    ///
    /// A character table.
    ///
    /// # Panics
    ///
    /// Panics if the character table cannot be constructed.
    pub fn new(
        name: &str,
        unitary_chartab: UC,
        ircoreps: &[RowSymbol],
        classes: &[UC::ColSymbol],
        principal_classes: &[UC::ColSymbol],
        char_arr: Array2<Character>,
        intertwining_numbers: &[u8],
    ) -> Self {
        assert_eq!(ircoreps.len(), char_arr.dim().0);
        assert_eq!(intertwining_numbers.len(), char_arr.dim().0);

        let ircoreps_indexmap: IndexMap<RowSymbol, usize> = ircoreps
            .iter()
            .cloned()
            .enumerate()
            .map(|(i, ircorep)| (ircorep, i))
            .collect();

        let classes_indexmap: IndexMap<UC::ColSymbol, usize> = classes
            .iter()
            .cloned()
            .enumerate()
            .map(|(i, class)| (class, i))
            .collect();

        let principal_classes_indexset: IndexSet<UC::ColSymbol> =
            principal_classes.iter().cloned().collect();

        let intertwining_numbers_indexmap = iter::zip(ircoreps, intertwining_numbers)
            .map(|(ircorep, &ini)| (ircorep.clone(), ini))
            .collect::<IndexMap<_, _>>();

        Self::builder()
            .name(name.to_string())
            .unitary_character_table(unitary_chartab)
            .ircoreps(ircoreps_indexmap)
            .classes(classes_indexmap)
            .principal_classes(principal_classes_indexset)
            .characters(char_arr)
            .intertwining_numbers(intertwining_numbers_indexmap)
            .build()
            .expect("Unable to construct a character table.")
    }
}

impl<RowSymbol, UC> CharacterTable for CorepCharacterTable<RowSymbol, UC>
where
    RowSymbol: ReducibleLinearSpaceSymbol,
    UC: CharacterTable,
{
    type RowSymbol = RowSymbol;
    type ColSymbol = UC::ColSymbol;

    /// Retrieves the character of a particular irreducible corepresentation in a particular
    /// unitary conjugacy class.
    ///
    /// # Arguments
    ///
    /// * `ircorep` - A Mulliken irreducible representation symbol.
    /// * `class` - A unitary conjugacy class symbol.
    ///
    /// # Returns
    ///
    /// The required character.
    ///
    /// # Panics
    ///
    /// Panics if the specified `ircorep` or `class` cannot be found.
    fn get_character(&self, ircorep: &Self::RowSymbol, class: &Self::ColSymbol) -> &Character {
        let row = self
            .ircoreps
            .get(ircorep)
            .unwrap_or_else(|| panic!("Ircorep {ircorep} not found."));
        let col = self
            .classes
            .get(class)
            .unwrap_or_else(|| panic!("Conjugacy class {class} not found."));
        &self.characters[(*row, *col)]
    }

    /// Retrieves the characters of all conjugacy classes in a particular irreducible
    /// corepresentation.
    ///
    /// # Arguments
    ///
    /// * `ircorep` - A Mulliken irreducible corepresentation symbol.
    ///
    /// # Returns
    ///
    /// The required characters.
    fn get_row(&self, row: &Self::RowSymbol) -> ArrayView1<Character> {
        let row = self
            .ircoreps
            .get(row)
            .unwrap_or_else(|| panic!("Ircorep {row} not found."));
        self.characters.row(*row)
    }

    /// Retrieves the characters of all irreducible corepresentations in a particular conjugacy
    /// class.
    ///
    /// # Arguments
    ///
    /// * `class` - A conjugacy class symbol.
    ///
    /// # Returns
    ///
    /// The required characters.
    fn get_col(&self, col: &Self::ColSymbol) -> ArrayView1<Character> {
        let col = self
            .classes
            .get(col)
            .unwrap_or_else(|| panic!("Conjugacy class {col} not found."));
        self.characters.column(*col)
    }

    /// Retrieves the Mulliken symbols of all irreducible corepresentations of the group.
    fn get_all_rows(&self) -> IndexSet<Self::RowSymbol> {
        self.ircoreps.keys().cloned().collect::<IndexSet<_>>()
    }

    /// Retrieves the symbols of all conjugacy classes of the group.
    fn get_all_cols(&self) -> IndexSet<Self::ColSymbol> {
        self.classes.keys().cloned().collect::<IndexSet<_>>()
    }

    /// Returns a shared reference to the underlying array of the character table.
    fn array(&self) -> &Array2<Character> {
        &self.characters
    }

    /// Retrieves the order of the group.
    fn get_order(&self) -> usize {
        2 * self.unitary_character_table.get_order()
    }

    /// Prints a nicely formatted character table.
    ///
    /// # Arguments
    ///
    /// * `compact` - Flag indicating if the columns are compact with unequal widths or expanded
    /// with all equal widths.
    /// * `numerical` - An option containing a non-negative integer specifying the number of decimal
    /// places for the numerical forms of the characters. If `None`, the characters will be shown
    /// as exact algebraic forms.
    ///
    /// # Returns
    ///
    /// A formatted string containing the character table in a printable form.
    ///
    /// # Panics
    ///
    /// Panics upon encountering any missing information required for a complete print-out of the
    /// character table.
    ///
    /// # Errors
    ///
    /// Errors upon encountering any issue formatting the character table.
    #[allow(clippy::too_many_lines)]
    fn write_nice_table(
        &self,
        f: &mut fmt::Formatter,
        compact: bool,
        numerical: Option<usize>,
    ) -> fmt::Result {
        let unitary_group_order: usize = self
            .classes
            .keys()
            .map(|cc| {
                cc.multiplicity().unwrap_or_else(|| {
                    panic!("Unable to find the multiplicity for conjugacy class `{cc}`.")
                })
            })
            .sum();

        let name = format!("m {} ({})", self.name, 2 * unitary_group_order);
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
        let ircoreps_str: Vec<_> = self
            .ircoreps
            .keys()
            .map(std::string::ToString::to_string)
            .collect();
        let ccs_str: Vec<_> = self
            .classes
            .keys()
            .map(|cc| {
                if self.principal_classes.contains(cc) {
                    format!("◈{cc}")
                } else {
                    cc.to_string()
                }
            })
            .collect();

        let first_width = max(
            ircoreps_str
                .iter()
                .map(|ircorep_str| ircorep_str.chars().count())
                .max()
                .expect("Unable to find the maximum length for the ircorep symbols."),
            name.chars().count(),
        ) + 1;

        let digit_widths: Vec<_> = if compact {
            iter::zip(chars_str.columns(), &ccs_str)
                .map(|(chars_col_str, cc_str)| {
                    let char_width = chars_col_str
                        .iter()
                        .map(|c| c.chars().count())
                        .max()
                        .expect("Unable to find the maximum length for the characters.");
                    let cc_width = cc_str.chars().count();
                    max(char_width, cc_width) + 1
                })
                .collect()
        } else {
            let char_width = chars_str
                .iter()
                .map(|c| c.chars().count())
                .max()
                .expect("Unable to find the maximum length for the characters.");
            let cc_width = ccs_str
                .iter()
                .map(|cc| cc.chars().count())
                .max()
                .expect("Unable to find the maximum length for the conjugacy class symbols.");
            let fixed_width = max(char_width, cc_width) + 1;
            iter::repeat(fixed_width).take(ccs_str.len()).collect()
        };

        // Table heading
        let mut heading = format!(" {name:^first_width$} ┆ IN ║");
        ccs_str.iter().enumerate().for_each(|(i, cc)| {
            heading.push_str(&format!("{cc:>width$} │", width = digit_widths[i]));
        });
        heading.pop();
        let tab_width = heading.chars().count();
        heading = format!(
            "{}\n{}\n{}\n",
            "━".repeat(tab_width),
            heading,
            "┈".repeat(tab_width),
        );
        write!(f, "{heading}")?;

        // Table body
        let rows = iter::zip(self.ircoreps.keys(), ircoreps_str)
            .enumerate()
            .map(|(i, (ircorep, ircorep_str))| {
                let intertwining_number =
                    self.intertwining_numbers.get(ircorep).unwrap_or_else(|| {
                        panic!("Unable to obtain the intertwining_number for ircorep `{ircorep}`.")
                    });
                let mut line = format!(" {ircorep_str:<first_width$} ┆ {intertwining_number:>2} ║");

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

        write!(
            f,
            "{}",
            &itertools::Itertools::intersperse(rows, "\n".to_string()).collect::<String>(),
        )?;

        // Table bottom
        write!(f, "\n{}\n", &"━".repeat(tab_width))
    }

    fn get_principal_cols(&self) -> &IndexSet<Self::ColSymbol> {
        &self.principal_classes
    }
}

// -------
// Display
// -------
impl<RowSymbol, UC> fmt::Display for CorepCharacterTable<RowSymbol, UC>
where
    RowSymbol: ReducibleLinearSpaceSymbol,
    UC: CharacterTable,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.write_nice_table(f, true, Some(3))
    }
}

// -----
// Debug
// -----
impl<RowSymbol, UC> fmt::Debug for CorepCharacterTable<RowSymbol, UC>
where
    RowSymbol: ReducibleLinearSpaceSymbol,
    UC: CharacterTable,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.write_nice_table(f, true, None)
    }
}
