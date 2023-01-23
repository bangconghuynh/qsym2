use std::cmp::max;
use std::fmt;
use std::iter;

use derive_builder::Builder;
use indexmap::{IndexMap, IndexSet};
use ndarray::{Array2, ArrayView1};

use crate::chartab::character::Character;
use crate::symmetry::symmetry_symbols::{
    ClassSymbol, MathematicalSymbol, MullikenIrcorepSymbol, MullikenIrrepSymbol,
    FROBENIUS_SCHUR_SYMBOLS,
};

pub mod character;
pub mod modular_linalg;
pub mod reducedint;
pub mod unityroot;

/// A trait to contain essential methods for a character table.
pub trait CharacterTable<RowSymbol, ColSymbol>
where
    RowSymbol: MathematicalSymbol,
    ColSymbol: MathematicalSymbol,
{
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
    fn get_character(&self, irrep: &RowSymbol, class: &ColSymbol) -> &Character;

    /// Retrieves the characters of all columns in a particular row.
    ///
    /// # Arguments
    ///
    /// * `irrep` - A Mulliken irreducible representation symbol.
    ///
    /// # Returns
    ///
    /// The required characters.
    fn get_row(&self, row: &RowSymbol) -> ArrayView1<Character>;

    /// Retrieves the characters of all rows in a particular column.
    ///
    /// # Arguments
    ///
    /// * `class` - A conjugacy class symbol.
    ///
    /// # Returns
    ///
    /// The required characters.
    fn get_col(&self, col: &ColSymbol) -> ArrayView1<Character>;

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
pub struct RepCharacterTable<R: Clone> {
    /// The name given to the character table.
    name: String,

    /// The irreducible representations of the group and their row indices in the character
    /// table.
    pub irreps: IndexMap<MullikenIrrepSymbol, usize>,

    /// The conjugacy classes of the group and their column indices in the character table.
    pub classes: IndexMap<ClassSymbol<R>, usize>,

    /// The principal conjugacy classes of the group.
    principal_classes: IndexSet<ClassSymbol<R>>,

    /// The characters of the irreducible representations in this group.
    pub characters: Array2<Character>,

    /// The Frobenius--Schur indicators for the irreducible representations in this group.
    frobenius_schurs: IndexMap<MullikenIrrepSymbol, i8>,

    /// The order of the group.
    #[builder(setter(skip), default = "self.order()")]
    order: usize,
}

impl<R: Clone> RepCharacterTableBuilder<R> {
    fn order(&self) -> usize {
        self.classes
            .as_ref()
            .expect("Conjugacy classes not found.")
            .keys()
            .map(|cc| {
                cc.multiplicity().unwrap_or_else(|| {
                    panic!("Unable to find the multiplicity for conjugacy class `{cc}`.")
                })
            })
            .sum()
    }
}

impl<R: Clone> RepCharacterTable<R> {
    fn builder() -> RepCharacterTableBuilder<R> {
        RepCharacterTableBuilder::default()
    }

    /// Constructs a new character table.
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
        irreps: &[MullikenIrrepSymbol],
        classes: &[ClassSymbol<R>],
        principal_classes: &[ClassSymbol<R>],
        char_arr: Array2<Character>,
        frobenius_schurs: &[i8],
    ) -> Self {
        assert_eq!(irreps.len(), char_arr.dim().0);
        assert_eq!(frobenius_schurs.len(), char_arr.dim().0);
        assert_eq!(classes.len(), char_arr.dim().1);
        assert_eq!(char_arr.dim().0, char_arr.dim().1);

        let irreps_indexmap: IndexMap<MullikenIrrepSymbol, usize> = irreps
            .iter()
            .cloned()
            .enumerate()
            .map(|(i, irrep)| (irrep, i))
            .collect();

        let classes_indexmap: IndexMap<ClassSymbol<R>, usize> = classes
            .iter()
            .cloned()
            .enumerate()
            .map(|(i, class)| (class, i))
            .collect();

        let principal_classes_indexset: IndexSet<ClassSymbol<R>> =
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

    /// Retrieves the characters of all conjugacy classes in a particular irreducible
    /// representation.
    ///
    /// This is an alias for [`Self::get_row`].
    ///
    /// # Arguments
    ///
    /// * `irrep` - A Mulliken irreducible representation symbol.
    ///
    /// # Returns
    ///
    /// The required characters.
    fn get_irrep(&self, irrep: &MullikenIrrepSymbol) -> ArrayView1<Character> {
        self.get_row(irrep)
    }

    /// Retrieves the characters of all irreducible representations in a particular conjugacy
    /// class.
    ///
    /// This is an alias for [`Self::get_col`].
    ///
    /// # Arguments
    ///
    /// * `class` - A conjugacy class symbol.
    ///
    /// # Returns
    ///
    /// The required characters.
    fn get_class(&self, class: &ClassSymbol<R>) -> ArrayView1<Character> {
        self.get_col(class)
    }
}

impl<R: Clone> CharacterTable<MullikenIrrepSymbol, ClassSymbol<R>> for RepCharacterTable<R> {
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
    fn get_character(&self, irrep: &MullikenIrrepSymbol, class: &ClassSymbol<R>) -> &Character {
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
    fn get_row(&self, row: &MullikenIrrepSymbol) -> ArrayView1<Character> {
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
    fn get_col(&self, col: &ClassSymbol<R>) -> ArrayView1<Character> {
        let col = self
            .classes
            .get(col)
            .unwrap_or_else(|| panic!("Conjugacy class {col} not found."));
        self.characters.column(*col)
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

        let name = format!("{} ({group_order})", self.name);
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
}

// -------
// Display
// -------
impl<R: Clone> fmt::Display for RepCharacterTable<R> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.write_nice_table(f, true, Some(3))
    }
}

// -----
// Debug
// -----
impl<R: Clone> fmt::Debug for RepCharacterTable<R> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.write_nice_table(f, true, None)
    }
}

// ===================
// CorepCharacterTable
// ===================

/// A struct to manage character tables of irreducible corepresentations of magnetic groups.
#[derive(Builder, Clone)]
pub struct CorepCharacterTable<R: Clone> {
    /// The name given to the character table.
    name: String,

    /// The character table of the irreducible representations of the halving unitary subgroup that
    /// induce the irreducible corepresentations of the current magnetic group.
    unitary_character_table: RepCharacterTable<R>,

    /// The intertwining numbers of the irreducible corepresentations.
    intertwining_numbers: IndexMap<MullikenIrrepSymbol, i8>,
}

impl<R: Clone> CharacterTable<MullikenIrcorepSymbol, ClassSymbol<R>> for CorepCharacterTable<R> {
    /// Retrieves the character of a particular irreducible corepresentation in a particular
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
    fn get_character(&self, irrep: &MullikenIrcorepSymbol, class: &ClassSymbol<R>) -> &Character {
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
    fn get_row(&self, row: &MullikenIrrepSymbol) -> ArrayView1<Character> {
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
    fn get_col(&self, col: &ClassSymbol<R>) -> ArrayView1<Character> {
        let col = self
            .classes
            .get(col)
            .unwrap_or_else(|| panic!("Conjugacy class {col} not found."));
        self.characters.column(*col)
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

        let name = format!("{} ({group_order})", self.name);
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
}
