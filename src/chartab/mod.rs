use std::cmp::max;
use std::error::Error;
use std::fmt;
use std::iter;
use std::ops::Mul;

use derive_builder::Builder;
use indexmap::{IndexMap, IndexSet};
use ndarray::{Array2, ArrayView1};
use num_complex::{Complex, ComplexFloat};
use num_traits::{ToPrimitive, Zero};
use rayon::prelude::*;
use serde::{de::DeserializeOwned, Deserialize, Serialize};

use crate::chartab::character::Character;
use crate::chartab::chartab_symbols::{
    CollectionSymbol, DecomposedSymbol, LinearSpaceSymbol, ReducibleLinearSpaceSymbol,
    FROBENIUS_SCHUR_SYMBOLS,
};

pub mod character;
pub mod chartab_group;
pub mod chartab_symbols;
pub mod modular_linalg;
pub mod reducedint;
pub mod unityroot;

// =================
// Trait definitions
// =================

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

/// A trait for character tables that support decomposing a space into its irreducible subspaces
/// using characters.
pub trait SubspaceDecomposable<T>: CharacterTable
where
    T: ComplexFloat,
    <T as ComplexFloat>::Real: ToPrimitive,
    Self::Decomposition: ReducibleLinearSpaceSymbol<Subspace = Self::RowSymbol>,
{
    /// The type for the decomposed result.
    type Decomposition;

    /// Reduces a space into subspaces using its characters under the conjugacy classes of the
    /// character table.
    ///
    /// # Arguments
    ///
    /// * `characters` - A hashmap of characters for conjugacy classes.
    ///
    /// # Returns
    ///
    /// The decomposition result.
    fn reduce_characters(
        &self,
        characters: &[(&Self::ColSymbol, T)],
        thresh: T::Real,
    ) -> Result<Self::Decomposition, DecompositionError>;
}

#[derive(Debug, Clone)]
pub struct DecompositionError(pub String);

impl fmt::Display for DecompositionError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Subspace decomposition error: {}", self.0)
    }
}

impl Error for DecompositionError {}

// ======================================
// Struct definitions and implementations
// ======================================

// -----------------
// RepCharacterTable
// -----------------

/// A structure to manage character tables of irreducible representations.
#[derive(Builder, Clone, Serialize, Deserialize)]
pub struct RepCharacterTable<RowSymbol, ColSymbol>
where
    RowSymbol: LinearSpaceSymbol,
    ColSymbol: CollectionSymbol,
{
    /// The name given to the character table.
    pub name: String,

    /// The irreducible representations of the group and their row indices in the character
    /// table.
    pub(crate) irreps: IndexMap<RowSymbol, usize>,

    /// The conjugacy classes of the group and their column indices in the character table.
    pub(crate) classes: IndexMap<ColSymbol, usize>,

    /// The principal conjugacy classes of the group.
    principal_classes: IndexSet<ColSymbol>,

    /// The characters of the irreducible representations in this group.
    pub(crate) characters: Array2<Character>,

    /// The Frobenius--Schur indicators for the irreducible representations in this group.
    pub(crate) frobenius_schurs: IndexMap<RowSymbol, i8>,
}

impl<RowSymbol, ColSymbol> RepCharacterTable<RowSymbol, ColSymbol>
where
    RowSymbol: LinearSpaceSymbol,
    ColSymbol: CollectionSymbol,
{
    /// Returns a builder to construct a `RepCharacterTable`.
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
    /// * `principal_classes` - A slice of the principal classes used in determining the irrep
    /// symbols.
    /// * `char_arr` - A two-dimensional array of characters.
    /// * `frobenius_schurs` - A slice of Frobenius--Schur indicators for the irreps.
    ///
    /// # Returns
    ///
    /// A character table.
    ///
    /// # Panics
    ///
    /// Panics if the character table cannot be constructed.
    pub(crate) fn new(
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
            .unwrap_or_else(|| panic!("Irrep `{irrep}` not found."));
        let col = self
            .classes
            .get(class)
            .unwrap_or_else(|| panic!("Conjugacy class `{class}` not found."));
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
    fn get_row(&self, irrep: &Self::RowSymbol) -> ArrayView1<Character> {
        let row = self
            .irreps
            .get(irrep)
            .unwrap_or_else(|| panic!("Irrep `{irrep}` not found."));
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
    fn get_col(&self, class: &Self::ColSymbol) -> ArrayView1<Character> {
        let col = self
            .classes
            .get(class)
            .unwrap_or_else(|| panic!("Conjugacy class `{class}` not found."));
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
        self.classes.keys().map(|cc| cc.size()).sum()
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
        let group_order = self.get_order();

        let name = format!("u {} ({group_order})", self.name);
        let chars_str = self.characters.map(|character| {
            if let Some(precision) = numerical {
                let real_only = self.characters.iter().all(|character| {
                    approx::relative_eq!(
                        character.complex_value().im,
                        0.0,
                        epsilon = character.threshold(),
                        max_relative = character.threshold()
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
                    let ind = self.frobenius_schurs.get(irrep).unwrap_or_else(|| {
                        panic!(
                            "Unable to obtain the Frobenius--Schur indicator for irrep `{irrep}`."
                        )
                    });
                    let fs = FROBENIUS_SCHUR_SYMBOLS.get(ind).unwrap_or_else(|| {
                        panic!("Unknown Frobenius--Schur symbol for indicator {ind}.")
                    });
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

impl<RowSymbol, ColSymbol, T> SubspaceDecomposable<T> for RepCharacterTable<RowSymbol, ColSymbol>
where
    RowSymbol: LinearSpaceSymbol + PartialOrd + Sync + Send,
    ColSymbol: CollectionSymbol + Sync + Send,
    T: ComplexFloat + Sync + Send,
    <T as ComplexFloat>::Real: ToPrimitive + Sync + Send,
    for<'a> Complex<f64>: Mul<&'a T, Output = Complex<f64>>,
{
    type Decomposition = DecomposedSymbol<RowSymbol>;

    /// Reduces a representation into irreducible representations using its characters under the
    /// conjugacy classes of the character table.
    ///
    /// # Arguments
    ///
    /// * `characters` - A hashmap of characters for conjugacy classes.
    ///
    /// # Returns
    ///
    /// The representation as a direct sum of irreducible representations.
    fn reduce_characters(
        &self,
        characters: &[(&Self::ColSymbol, T)],
        thresh: T::Real,
    ) -> Result<Self::Decomposition, DecompositionError> {
        assert_eq!(characters.len(), self.classes.len());
        let rep_syms: Result<Vec<Option<(RowSymbol, usize)>>, _> = self
            .irreps
            .par_iter()
            .map(|(irrep_symbol, &i)| {
                let c = characters
                    .par_iter()
                    .try_fold(|| Complex::<f64>::zero(), |acc, (cc_symbol, character)| {
                        let j = self.classes.get_index_of(*cc_symbol).ok_or(DecompositionError(
                            format!(
                                "The conjugacy class `{cc_symbol}` cannot be found in this group."
                            )
                        ))?;
                        Ok(
                            acc + cc_symbol.size().to_f64().ok_or(DecompositionError(
                                format!(
                                    "The size of conjugacy class `{cc_symbol}` cannot be converted to `f64`."
                                )
                            ))?
                                * self.characters[(i, j)].complex_conjugate().complex_value()
                                * character
                        )
                    })
                    .try_reduce(|| Complex::<f64>::zero(), |a, s| Ok(a + s))? / self.get_order().to_f64().ok_or(
                        DecompositionError("The group order cannot be converted to `f64`.".to_string())
                    )?;

                let thresh_f64 = thresh.to_f64().expect("Unable to convert the threshold to `f64`.");
                if approx::relative_ne!(c.im, 0.0, epsilon = thresh_f64, max_relative = thresh_f64) {
                    Err(
                        DecompositionError(
                            format!(
                                "Non-negligible imaginary part for irrep multiplicity: {:.3e}",
                                c.im
                            )
                        )
                    )
                } else if c.re < -thresh_f64 {
                    Err(
                        DecompositionError(
                            format!(
                                "Negative irrep multiplicity: {:.3e}",
                                c.re
                            )
                        )
                    )
                } else if approx::relative_ne!(
                    c.re, c.re.round(), epsilon = thresh_f64, max_relative = thresh_f64
                ) {
                    Err(
                        DecompositionError(
                            format!(
                                "Non-integer coefficient: {:.3e}",
                                c.re
                            )
                        )
                    )
                } else {
                    let mult = c.re.round().to_usize().ok_or(DecompositionError(
                        format!(
                            "Unable to convert the rounded coefficient `{}` to `usize`.",
                            c.re.round()
                        )
                    ))?;
                    if mult != 0 {
                        Ok(Some((irrep_symbol.clone(), mult)))
                    } else {
                        Ok(None)
                    }
                }
        })
        .collect();

        rep_syms.map(|syms| {
            DecomposedSymbol::<RowSymbol>::from_subspaces(
                &syms.into_iter().flatten().collect::<Vec<_>>(),
            )
        })
    }
}

impl<RowSymbol, ColSymbol> fmt::Display for RepCharacterTable<RowSymbol, ColSymbol>
where
    RowSymbol: LinearSpaceSymbol,
    ColSymbol: CollectionSymbol,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.write_nice_table(f, true, Some(3))
    }
}

impl<RowSymbol, ColSymbol> fmt::Debug for RepCharacterTable<RowSymbol, ColSymbol>
where
    RowSymbol: LinearSpaceSymbol,
    ColSymbol: CollectionSymbol,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.write_nice_table(f, true, None)
    }
}

// -------------------
// CorepCharacterTable
// -------------------

/// A structure to manage character tables of irreducible corepresentations of magnetic groups.
#[derive(Builder, Clone, Serialize, Deserialize)]
pub struct CorepCharacterTable<RowSymbol, UC>
where
    <UC as CharacterTable>::ColSymbol: Serialize + DeserializeOwned,
    RowSymbol: ReducibleLinearSpaceSymbol,
    UC: CharacterTable,
{
    /// The name given to the character table.
    pub name: String,

    /// The character table of the irreducible representations of the halving unitary subgroup that
    /// induce the irreducible corepresentations of the current magnetic group.
    pub(crate) unitary_character_table: UC,

    /// The irreducible corepresentations of the group and their row indices in the character
    /// table.
    pub(crate) ircoreps: IndexMap<RowSymbol, usize>,

    /// The conjugacy classes of the group and their column indices in the character table.
    classes: IndexMap<UC::ColSymbol, usize>,

    /// The principal conjugacy classes of the group.
    principal_classes: IndexSet<UC::ColSymbol>,

    /// The characters of the irreducible corepresentations in this group.
    characters: Array2<Character>,

    /// The intertwining numbers of the irreducible corepresentations.
    pub(crate) intertwining_numbers: IndexMap<RowSymbol, u8>,
}

impl<RowSymbol, UC> CorepCharacterTable<RowSymbol, UC>
where
    <UC as CharacterTable>::ColSymbol: Serialize + DeserializeOwned,
    RowSymbol: ReducibleLinearSpaceSymbol,
    UC: CharacterTable,
{
    /// Returns a builder to construct a new [`CorepCharacterTable`].
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
    /// * `classes` - A slice of conjugacy class symbols in the right order. These symbols must be
    /// of the same type as those of the unitary subgroup.
    /// * `principal_classes` - A slice of the principal classes of the group.
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
    pub(crate) fn new(
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
    <UC as CharacterTable>::ColSymbol: Serialize + DeserializeOwned,
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
            .unwrap_or_else(|| panic!("Ircorep `{ircorep}` not found."));
        let col = self
            .classes
            .get(class)
            .unwrap_or_else(|| panic!("Conjugacy class `{class}` not found."));
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
    fn get_row(&self, ircorep: &Self::RowSymbol) -> ArrayView1<Character> {
        let row = self
            .ircoreps
            .get(ircorep)
            .unwrap_or_else(|| panic!("Ircorep `{ircorep}` not found."));
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
    fn get_col(&self, class: &Self::ColSymbol) -> ArrayView1<Character> {
        let col = self
            .classes
            .get(class)
            .unwrap_or_else(|| panic!("Conjugacy class `{class}` not found."));
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
        let group_order = self.get_order();

        let name = format!("m {} ({})", self.name, group_order);
        let chars_str = self.characters.map(|character| {
            if let Some(precision) = numerical {
                let real_only = self.characters.iter().all(|character| {
                    approx::relative_eq!(
                        character.complex_value().im,
                        0.0,
                        epsilon = character.threshold(),
                        max_relative = character.threshold()
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

impl<RowSymbol, UC, T> SubspaceDecomposable<T> for CorepCharacterTable<RowSymbol, UC>
where
    RowSymbol: ReducibleLinearSpaceSymbol + PartialOrd + Sync + Send,
    UC: CharacterTable + Sync + Send,
    <UC as CharacterTable>::ColSymbol: Serialize + DeserializeOwned + Sync + Send,
    T: ComplexFloat + Sync + Send,
    <T as ComplexFloat>::Real: ToPrimitive + Sync + Send,
    for<'a> Complex<f64>: Mul<&'a T, Output = Complex<f64>>,
{
    type Decomposition = DecomposedSymbol<RowSymbol>;

    /// Reduces a corepresentation into irreducible corepresentations using its characters under the
    /// conjugacy classes of the character table.
    ///
    /// # Arguments
    ///
    /// * `characters` - A hashmap of characters for conjugacy classes.
    ///
    /// # Returns
    ///
    /// The corepresentation as a direct sum of irreducible corepresentations.
    fn reduce_characters(
        &self,
        characters: &[(&Self::ColSymbol, T)],
        thresh: T::Real,
    ) -> Result<Self::Decomposition, DecompositionError> {
        assert_eq!(characters.len(), self.classes.len());
        let rep_syms: Result<Vec<Option<(RowSymbol, usize)>>, _> = self
            .ircoreps
            .par_iter()
            .map(|(ircorep_symbol, &i)| {
                let c = characters
                    .par_iter()
                    .try_fold(|| Complex::<f64>::zero(), |acc, (cc_symbol, character)| {
                        let j = self.classes.get_index_of(*cc_symbol).ok_or(DecompositionError(
                            format!(
                                "The conjugacy class `{cc_symbol}` cannot be found in this group."
                            )
                        ))?;
                        Ok(
                            acc + cc_symbol.size().to_f64().ok_or(DecompositionError(
                                format!(
                                    "The size of conjugacy class `{cc_symbol}` cannot be converted to `f64`."
                                )
                            ))?
                                * self.characters[(i, j)].complex_conjugate().complex_value()
                                * character
                        )
                    })
                    .try_reduce(|| Complex::<f64>::zero(), |a, s| Ok(a + s))? / (self.unitary_character_table.get_order().to_f64().ok_or(
                        DecompositionError("The unitary subgroup order cannot be converted to `f64`.".to_string())
                    )? * self.intertwining_numbers.get(ircorep_symbol).and_then(|x| x.to_f64()).ok_or(
                        DecompositionError(
                            format!(
                                "The intertwining number of `{ircorep_symbol}` cannot be retrieved and/or converted to `f64`."
                            )
                        )
                    )?);

                let thresh_f64 = thresh.to_f64().expect("Unable to convert the threshold to `f64`.");
                if approx::relative_ne!(c.im, 0.0, epsilon = thresh_f64, max_relative = thresh_f64) {
                    Err(
                        DecompositionError(
                            format!(
                                "Non-negligible imaginary part for ircorep multiplicity: {:.3e}",
                                c.im
                            )
                        )
                    )
                } else if c.re < -thresh_f64 {
                    Err(
                        DecompositionError(
                            format!(
                                "Negative ircorep multiplicity: {:.3e}",
                                c.re
                            )
                        )
                    )
                } else if approx::relative_ne!(
                    c.re, c.re.round(), epsilon = thresh_f64, max_relative = thresh_f64
                ) {
                    Err(
                        DecompositionError(
                            format!(
                                "Non-integer coefficient: {:.3e}",
                                c.re
                            )
                        )
                    )
                } else {
                    let mult = c.re.round().to_usize().ok_or(DecompositionError(
                        format!(
                            "Unable to convert the rounded coefficient `{}` to `usize`.",
                            c.re.round()
                        )
                    ))?;
                    if mult != 0 {
                        Ok(Some((ircorep_symbol.clone(), mult)))
                    } else {
                        Ok(None)
                    }
                }
        })
        .collect();

        rep_syms.map(|syms| {
            DecomposedSymbol::<RowSymbol>::from_subspaces(
                &syms.into_iter().flatten().collect::<Vec<_>>(),
            )
        })
    }
}

impl<RowSymbol, UC> fmt::Display for CorepCharacterTable<RowSymbol, UC>
where
    RowSymbol: ReducibleLinearSpaceSymbol,
    UC: CharacterTable,
    <UC as CharacterTable>::ColSymbol: Serialize + DeserializeOwned,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.write_nice_table(f, true, Some(3))
    }
}

impl<RowSymbol, UC> fmt::Debug for CorepCharacterTable<RowSymbol, UC>
where
    RowSymbol: ReducibleLinearSpaceSymbol,
    UC: CharacterTable,
    <UC as CharacterTable>::ColSymbol: Serialize + DeserializeOwned,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.write_nice_table(f, true, None)
    }
}
