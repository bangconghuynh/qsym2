use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::hash::{Hash, Hasher};
use std::str::FromStr;

use log;

use derive_builder::Builder;
use indexmap::{IndexMap, IndexSet};
use itertools::Itertools;
use nalgebra::Vector3;
use ndarray::{Array2, ArrayView2, Axis};
use phf::{phf_map, phf_set};
use regex::Regex;

use crate::chartab::character::Character;
use crate::chartab::chartab_symbols::{
    disambiguate_linspace_symbols, CollectionSymbol, GenericSymbol, GenericSymbolParsingError,
    LinearSpaceSymbol, MathematicalSymbol, ReducibleLinearSpaceSymbol,
};
use crate::chartab::unityroot::UnityRoot;
use crate::group::FiniteOrder;
use crate::symmetry::symmetry_element::symmetry_operation::SpecialSymmetryTransformation;
use crate::symmetry::symmetry_element::SymmetryElement;
use crate::symmetry::symmetry_element_order::ORDER_1;

#[cfg(test)]
#[path = "symmetry_symbols_tests.rs"]
mod symmetry_symbols_tests;

// =========
// Constants
// =========

static MULLIKEN_IRREP_DEGENERACIES: phf::Map<&'static str, u64> = phf_map! {
    "A" => 1u64,
    "B" => 1u64,
    "Σ" => 1u64,
    "Γ" => 1u64,
    "E" => 2u64,
    "Π" => 2u64,
    "Δ" => 2u64,
    "Φ" => 2u64,
    "T" => 3u64,
    "G" => 4u64,
    "H" => 5u64,
    "I" => 6u64,
    "J" => 7u64,
    "K" => 8u64,
    "L" => 9u64,
    "M" => 10u64,
};

static INV_MULLIKEN_IRREP_DEGENERACIES: phf::Map<u64, &'static str> = phf_map! {
     2u64 => "E",
     3u64 => "T",
     4u64 => "G",
     5u64 => "H",
     6u64 => "I",
     7u64 => "J",
     8u64 => "K",
     9u64 => "L",
     10u64 => "M",
};

pub static FORCED_PRINCIPAL_GROUPS: phf::Set<&'static str> = phf_set! {
    "O",
    "Oh",
    "Td",
    "O + θ·O",
    "Oh + θ·Oh",
    "Td + θ·Td",
};

// ==================
// Struct definitions
// ==================

// -------------------
// MullikenIrrepSymbol
// -------------------

/// A struct to handle Mulliken irreducible representation symbols.
#[derive(Builder, Debug, Clone, PartialEq, Eq, Hash, PartialOrd)]
pub struct MullikenIrrepSymbol {
    /// The generic part of the symbol.
    generic_symbol: GenericSymbol,
}

impl MullikenIrrepSymbol {
    fn builder() -> MullikenIrrepSymbolBuilder {
        MullikenIrrepSymbolBuilder::default()
    }

    /// Parses a string representing a Mulliken irrep symbol.
    ///
    /// Some permissible Mulliken irrep symbols:
    ///
    /// ```text
    /// "T"
    /// "||T|_(2g)|"
    /// "|^(3)|T|_(2g)|"
    /// ```
    ///
    /// # Arguments
    ///
    /// * `symstr` - A string to be parsed to give a Mulliken symbol.
    ///
    /// # Errors
    ///
    /// Errors when the string cannot be parsed as a generic symbol.
    pub fn new(symstr: &str) -> Result<Self, MullikenIrrepSymbolBuilderError> {
        Self::from_str(symstr)
    }
}

// ---------------------
// MullikenIrcorepSymbol
// ---------------------

/// A struct to handle Mulliken irreducible corepresentation symbols.
#[derive(Builder, Debug, Clone, Eq)]
pub struct MullikenIrcorepSymbol {
    inducing_irreps: HashMap<MullikenIrrepSymbol, usize>,
}

impl MullikenIrcorepSymbol {
    fn builder() -> MullikenIrcorepSymbolBuilder {
        MullikenIrcorepSymbolBuilder::default()
    }

    /// Parses a string representing a Mulliken irrep symbol.
    ///
    /// Some permissible Mulliken irrep symbols:
    ///
    /// ```text
    /// "T"
    /// "||T|_(2g)|"
    /// "|^(3)|T|_(2g)|"
    /// ```
    ///
    /// # Arguments
    ///
    /// * `symstr` - A string to be parsed to give a Mulliken symbol.
    ///
    /// # Errors
    ///
    /// Errors when the string cannot be parsed as a generic symbol.
    pub fn new(symstr: &str) -> Result<Self, MullikenIrcorepSymbolBuilderError> {
        Self::from_str(symstr)
    }

    /// Returns an iterator containing sorted references to the symbols of the inducing irreps.
    ///
    /// # Panics
    ///
    /// Panics if the inducing irrep symbols cannot be ordered.
    #[must_use]
    pub fn sorted_inducing_irreps(&self) -> std::vec::IntoIter<(&MullikenIrrepSymbol, &usize)> {
        self.inducing_irreps.iter().sorted_by(|(a, _), (b, _)| {
            a.partial_cmp(b)
                .unwrap_or_else(|| panic!("`{a}` and `{b}` cannot be compared."))
        })
    }
}

// -------------------
// SymmetryClassSymbol
// -------------------

/// A struct to handle conjugacy class symbols.
#[derive(Builder, Debug, Clone)]
pub struct SymmetryClassSymbol<R: Clone> {
    /// The generic part of the symbol.
    generic_symbol: GenericSymbol,

    /// A representative element in the class.
    representative: Option<R>,
}

impl<R: Clone> SymmetryClassSymbol<R> {
    fn builder() -> SymmetryClassSymbolBuilder<R> {
        SymmetryClassSymbolBuilder::default()
    }

    pub fn representative(&self) -> Option<R> {
        self.representative.clone()
    }

    /// Creates a class symbol from a string and a representative element.
    ///
    /// Some permissible conjugacy class symbols:
    ///
    /// ```text
    /// "1||C3||"
    /// "1||C3|^(2)|"
    /// "12||C2|^(5)|"
    /// "2||S|^(z)|(α)"
    /// ```
    ///
    /// Note that the prefactor is required.
    ///
    /// # Arguments
    ///
    /// * `symstr` - A string to be parsed to give a class symbol.
    /// * `rep` - An optional representative element for this class.
    ///
    /// # Returns
    ///
    /// A [`Result`] wrapping the constructed class symbol.
    ///
    /// # Panics
    ///
    /// Panics when unable to construct a class symbol from the specified string.
    ///
    /// # Errors
    ///
    /// Errors when the string contains no parsable class size prefactor, or when the string cannot
    /// be parsed as a generic symbol.
    pub fn new(symstr: &str, rep: Option<R>) -> Result<Self, GenericSymbolParsingError> {
        let generic_symbol = GenericSymbol::from_str(symstr)?;
        if generic_symbol.multiplicity().is_none() {
            Err(GenericSymbolParsingError(format!(
                "{symstr} contains no class size prefactor."
            )))
        } else {
            Ok(Self::builder()
                .generic_symbol(generic_symbol)
                .representative(rep)
                .build()
                .unwrap_or_else(|_| panic!("Unable to construct a class symbol from `{symstr}`.")))
        }
    }
}

// =====================
// Trait implementations
// =====================

// -------------------
// MullikenIrrepSymbol
// -------------------

impl MathematicalSymbol for MullikenIrrepSymbol {
    /// The main part of the symbol, which primarily denotes the dimensionality of the irrep space.
    fn main(&self) -> String {
        self.generic_symbol.main()
    }

    /// The pre-superscript part of the symbol, which can be used to denote antiunitary symmetries
    /// or spin multiplicities.
    fn presuper(&self) -> String {
        self.generic_symbol.presuper()
    }

    fn presub(&self) -> String {
        self.generic_symbol.presub()
    }

    /// The post-superscript part of the symbol, which denotes reflection parity.
    fn postsuper(&self) -> String {
        self.generic_symbol.postsuper()
    }

    /// The post-subscript part of the symbol, which denotes inversion parity when available and
    /// which disambiguates similar irreps.
    fn postsub(&self) -> String {
        self.generic_symbol.postsub()
    }

    /// The prefactor part of the symbol, which is always `"1"` implicitly because of irreducibility.
    fn prefactor(&self) -> String {
        String::new()
    }

    /// The postfactor part of the symbol, which is always empty.
    fn postfactor(&self) -> String {
        String::new()
    }

    /// The dimensionality of the irreducible representation.
    fn multiplicity(&self) -> Option<usize> {
        if let Some(&mult) = MULLIKEN_IRREP_DEGENERACIES.get(&self.main()) {
            Some(
                mult.try_into()
                    .unwrap_or_else(|_| panic!("Unable to convert {mult} to `usize`.")),
            )
        } else {
            None
        }
    }
}

impl FromStr for MullikenIrrepSymbol {
    type Err = MullikenIrrepSymbolBuilderError;

    /// Parses a string representing a Mulliken irrep symbol.
    ///
    /// Some permissible Mulliken irrep symbols:
    ///
    /// ```text
    /// "T"
    /// "||T|_(2g)|"
    /// "|^(3)|T|_(2g)|"
    /// ```
    /// # Arguments
    ///
    /// * `symstr` - A string to be parsed to give a Mulliken symbol.
    ///
    /// # Returns
    ///
    /// A [`Result`] wrapping the constructed Mulliken symbol.
    ///
    /// # Panics
    ///
    /// Panics when unable to construct a Mulliken symbol from the specified string.
    ///
    /// # Errors
    ///
    /// Errors when the string cannot be parsed as a generic symbol.
    fn from_str(symstr: &str) -> Result<Self, Self::Err> {
        let generic_symbol = GenericSymbol::from_str(symstr)
            .unwrap_or_else(|_| panic!("Unable to parse {symstr} as a generic symbol."));
        Self::builder().generic_symbol(generic_symbol).build()
    }
}

impl LinearSpaceSymbol for MullikenIrrepSymbol {
    fn dimensionality(&self) -> usize {
        usize::try_from(
            *MULLIKEN_IRREP_DEGENERACIES
                .get(&self.main())
                .unwrap_or_else(|| {
                    panic!(
                        "Unknown dimensionality for Mulliken symbol {}.",
                        self.main()
                    )
                }),
        )
        .expect("Unable to convert the dimensionality of this irrep to `usize`.")
    }

    fn set_dimensionality(&mut self, dim: usize) -> bool {
        let dim_u64 = u64::try_from(dim).unwrap_or_else(|err| {
            log::error!("{err}");
            panic!("Unable to convert `{dim}` to `u64`.")
        });
        let main_opt = INV_MULLIKEN_IRREP_DEGENERACIES.get(&dim_u64);
        if let Some(main) = main_opt {
            self.generic_symbol.set_main(main);
            true
        } else {
            log::warn!("Unable to retrieve an unambiguous Mulliken symbol for dimensionality `{dim_u64}`. Main symbol of {self} will be kept unchanged.");
            false
        }
    }
}

impl fmt::Display for MullikenIrrepSymbol {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.generic_symbol)
    }
}

// ---------------------
// MullikenIrcorepSymbol
// ---------------------

impl MathematicalSymbol for MullikenIrcorepSymbol {
    /// The main part of the symbol.
    fn main(&self) -> String {
        format!(
            "D[{}]",
            self.sorted_inducing_irreps()
                .map(|(irrep, mult)| format!(
                    "{}{irrep}",
                    if *mult > 1 {
                        mult.to_string()
                    } else {
                        String::new()
                    }
                ))
                .join(" ⊕ ")
        )
    }

    /// The pre-superscript part of the symbol, which is always empty.
    fn presuper(&self) -> String {
        String::new()
    }

    fn presub(&self) -> String {
        String::new()
    }

    /// The post-superscript part of the symbol, which is always empty.
    fn postsuper(&self) -> String {
        String::new()
    }

    /// The post-subscript part of the symbol, which is always empty.
    fn postsub(&self) -> String {
        String::new()
    }

    /// The prefactor part of the symbol, which is always `"1"` implicitly because of irreducibility.
    fn prefactor(&self) -> String {
        "1".to_string()
    }

    /// The postfactor part of the symbol, which is always empty.
    fn postfactor(&self) -> String {
        String::new()
    }

    /// The dimensionality of the irreducible corepresentation.
    fn multiplicity(&self) -> Option<usize> {
        Some(
            self.inducing_irreps
            .iter()
            .map(|(irrep, mult)| {
                irrep
                    .multiplicity()
                    .expect("One of the inducing irreducible representations has an undefined multiplicity.")
                * mult
            })
            .sum()
        )
    }
}

impl FromStr for MullikenIrcorepSymbol {
    type Err = MullikenIrcorepSymbolBuilderError;

    /// Parses a string representing a Mulliken ircorep symbol. A valid string representing a
    /// Mulliken ircorep symbol is one consisting of one or more Mulliken irrep symbol strings,
    /// separated by a `+` character.
    ///
    /// # Arguments
    ///
    /// * `symstr` - A string to be parsed to give a Mulliken ircorep symbol.
    ///
    /// # Returns
    ///
    /// A [`Result`] wrapping the constructed Mulliken ircorep symbol.
    ///
    /// # Panics
    ///
    /// Panics when unable to construct a Mulliken ircorep symbol from the specified string.
    ///
    /// # Errors
    ///
    /// Errors when the string cannot be parsed.
    fn from_str(symstr: &str) -> Result<Self, Self::Err> {
        let re = Regex::new(r"(\d?)(.*)").expect("Regex pattern invalid.");
        let irreps = symstr
            .split('+')
            .map(|irrep_str| {
                let cap = re
                    .captures(irrep_str.trim())
                    .unwrap_or_else(|| panic!("{irrep_str} does not fit the expected pattern."));
                let mult_str = cap
                    .get(1)
                    .expect("Unable to parse the multiplicity of the irrep.")
                    .as_str();
                let mult = if mult_str.is_empty() {
                    1
                } else {
                    str::parse::<usize>(mult_str)
                        .unwrap_or_else(|_| panic!("`{mult_str}` is not a positive integer."))
                };
                let irrep = cap.get(2).expect("Unable to parse the irrep.").as_str();
                (
                    MullikenIrrepSymbol::from_str(irrep).unwrap_or_else(|_| {
                        panic!("Unable to parse {irrep} as a Mulliken irrep symbol.")
                    }),
                    mult,
                )
            })
            .collect::<HashMap<_, _>>();
        MullikenIrcorepSymbol::builder()
            .inducing_irreps(irreps)
            .build()
    }
}

impl LinearSpaceSymbol for MullikenIrcorepSymbol {
    fn dimensionality(&self) -> usize {
        self.inducing_irreps
            .iter()
            .map(|(irrep, mult)| {
                irrep
                    .multiplicity()
                    .expect("One of the inducing irreducible representations has an undefined multiplicity.")
                * mult
            }).sum()
    }

    fn set_dimensionality(&mut self, _: usize) -> bool {
        log::error!("The dimensionality of `{self}` cannot be set.");
        false
    }
}

impl ReducibleLinearSpaceSymbol for MullikenIrcorepSymbol {
    type Subspace = MullikenIrrepSymbol;

    fn from_subspaces(irreps: &[(Self::Subspace, usize)]) -> Self {
        Self::builder()
            .inducing_irreps(irreps.iter().cloned().collect::<HashMap<_, _>>())
            .build()
            .expect("Unable to construct a Mulliken ircorep symbol from a slice of irrep symbols.")
    }

    fn subspaces(&self) -> Vec<(&Self::Subspace, &usize)> {
        self.sorted_inducing_irreps().collect_vec()
    }
}

impl fmt::Display for MullikenIrcorepSymbol {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.main())
    }
}

impl PartialEq for MullikenIrcorepSymbol {
    fn eq(&self, other: &Self) -> bool {
        let self_irreps = self.sorted_inducing_irreps().collect_vec();
        let other_irreps = other.sorted_inducing_irreps().collect_vec();
        self_irreps == other_irreps
    }
}

impl Hash for MullikenIrcorepSymbol {
    fn hash<H: Hasher>(&self, state: &mut H) {
        for irrep in self.sorted_inducing_irreps() {
            irrep.hash(state);
        }
    }
}

// -------------------
// SymmetryClassSymbol
// -------------------

impl<R: Clone> PartialEq for SymmetryClassSymbol<R> {
    fn eq(&self, other: &Self) -> bool {
        self.generic_symbol == other.generic_symbol
    }
}

impl<R: Clone> Eq for SymmetryClassSymbol<R> {}

impl<R: Clone> Hash for SymmetryClassSymbol<R> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.generic_symbol.hash(state);
    }
}

impl<R: Clone> MathematicalSymbol for SymmetryClassSymbol<R> {
    /// The main part of the symbol, which denotes the representative symmetry operation.
    fn main(&self) -> String {
        self.generic_symbol.main()
    }

    /// The pre-superscript part of the symbol, which is empty.
    fn presuper(&self) -> String {
        String::new()
    }

    /// The pre-subscript part of the symbol, which is empty.
    fn presub(&self) -> String {
        String::new()
    }

    /// The post-superscript part of the symbol, which is empty.
    fn postsuper(&self) -> String {
        String::new()
    }

    /// The post-subscript part of the symbol, which is empty.
    fn postsub(&self) -> String {
        String::new()
    }

    /// The prefactor part of the symbol, which denotes the size of the class.
    fn prefactor(&self) -> String {
        self.generic_symbol.prefactor()
    }

    /// The postfactor part of the symbol, which is empty.
    fn postfactor(&self) -> String {
        String::new()
    }

    /// This is a synonym for the size of the conjugacy class.
    fn multiplicity(&self) -> Option<usize> {
        self.generic_symbol.multiplicity()
    }
}

impl<R: Clone> CollectionSymbol for SymmetryClassSymbol<R> {
    type CollectionElement = R;

    fn size(&self) -> usize {
        self.multiplicity().unwrap_or_else(|| {
            panic!(
                "Unable to deduce the size of the class from the prefactor {}.",
                self.prefactor()
            )
        })
    }

    fn from_rep(
        symstr: &str,
        rep: Option<Self::CollectionElement>,
    ) -> Result<Self, GenericSymbolParsingError> {
        Self::new(symstr, rep)
    }

    fn representative(&self) -> Option<Self::CollectionElement> {
        self.representative.clone()
    }
}

impl<R: SpecialSymmetryTransformation + Clone> SpecialSymmetryTransformation
    for SymmetryClassSymbol<R>
{
    // ============
    // Spatial part
    // ============

    /// Checks if this class is proper.
    ///
    /// # Returns
    ///
    /// A flag indicating if this class is proper.
    fn is_proper(&self) -> bool {
        self.representative
            .as_ref()
            .expect("No representative element found for this class.")
            .is_proper()
    }

    fn is_spatial_identity(&self) -> bool {
        self.representative
            .as_ref()
            .expect("No representative element found for this class.")
            .is_spatial_identity()
    }

    fn is_spatial_binary_rotation(&self) -> bool {
        self.representative
            .as_ref()
            .expect("No representative element found for this class.")
            .is_spatial_binary_rotation()
    }

    fn is_spatial_inversion(&self) -> bool {
        self.representative
            .as_ref()
            .expect("No representative element found for this class.")
            .is_spatial_inversion()
    }

    fn is_spatial_reflection(&self) -> bool {
        self.representative
            .as_ref()
            .expect("No representative element found for this class.")
            .is_spatial_reflection()
    }

    // ==================
    // Time-reversal part
    // ==================

    /// Checks if this class is antiunitary.
    ///
    /// # Returns
    ///
    /// A flag indicating if this class is antiunitary.
    fn is_antiunitary(&self) -> bool {
        self.representative
            .as_ref()
            .expect("No representative element found for this class.")
            .is_antiunitary()
    }

    // ==================
    // Spin rotation part
    // ==================

    /// Checks if this class contains an active associated spin rotation (normal or inverse).
    ///
    /// # Returns
    ///
    /// A flag indicating if this class contains an active associated spin rotation.
    fn is_su2(&self) -> bool {
        self.representative
            .as_ref()
            .expect("No representative element found for this class.")
            .is_su2()
    }

    /// Checks if this class contains an active and inverse associated spin rotation.
    ///
    /// # Returns
    ///
    /// A flag indicating if this class contains an active and inverse associated spin rotation.
    fn is_su2_class_1(&self) -> bool {
        self.representative
            .as_ref()
            .expect("No representative element found for this class.")
            .is_su2_class_1()
    }
}

impl<R: FiniteOrder + Clone> FiniteOrder for SymmetryClassSymbol<R> {
    type Int = R::Int;

    fn order(&self) -> Self::Int {
        self.representative
            .as_ref()
            .expect("No representative element found for this class.")
            .order()
    }
}

impl<R: Clone> fmt::Display for SymmetryClassSymbol<R> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.generic_symbol)
    }
}

// =======
// Methods
// =======

/// Reorder the rows so that the characters are in increasing order as we
/// go down the table, with the first column being used as the primary sort
/// order, then the second column, and so on, with the exceptions of some
/// special columns, if available. These are:
///
/// * time-reversal class, $`\theta`$,
/// * inversion class, $`i`$, or horizonal mirror plane $`\sigma_h`$ if $`i`$ not available.
///
/// # Arguments
///
/// * `char_arr` - A view of a two-dimensional square array containing the characters where
/// each column is for one conjugacy class and each row one irrep.
/// * `class_symbols` - An index map containing the conjugacy class symbols for the columns of
/// `char_arr`. The keys are the symbols and the values are the column indices.
///
/// # Returns
///
/// A new array with the rows correctly sorted.
///
/// # Panics
///
/// Panics when expected classes cannot be found.
pub fn sort_irreps<R: Clone>(
    char_arr: &ArrayView2<Character>,
    frobenius_schur_indicators: &[i8],
    class_symbols: &IndexMap<SymmetryClassSymbol<R>, usize>,
    principal_classes: &[SymmetryClassSymbol<R>],
) -> (Array2<Character>, Vec<i8>) {
    log::debug!("Sorting irreducible representations...");
    let class_e = class_symbols
        .first()
        .expect("No class symbols found.")
        .0
        .clone();
    let class_i = SymmetryClassSymbol::new("1||i||", None)
        .expect("Unable to construct class symbol `1||i||`.");
    let class_ti = SymmetryClassSymbol::new("1||θ·i||", None)
        .expect("Unable to construct class symbol `1||θ·i||`.");
    let class_s = SymmetryClassSymbol::new("1||σh||", None)
        .expect("Unable to construct class symbol `1||σh||`.");
    let class_ts = SymmetryClassSymbol::new("1||θ·σh||", None)
        .expect("Unable to construct class symbol `1||θ·σh||`.");
    let class_t = SymmetryClassSymbol::new("1||θ||", None)
        .expect("Unable to construct class symbol `1||θ||`.");

    let mut leading_classes: IndexSet<SymmetryClassSymbol<R>> = IndexSet::new();

    // Highest priority: time-reversal
    if class_symbols.contains_key(&class_t) {
        leading_classes.insert(class_t);
    }

    // Second highest priority: inversion, or horizontal mirror plane if inversion not available,
    // or time-reversed horizontal mirror plane if non-time-reversed version not available.
    if class_symbols.contains_key(&class_i) {
        leading_classes.insert(class_i);
    } else if class_symbols.contains_key(&class_ti) {
        leading_classes.insert(class_ti);
    } else if class_symbols.contains_key(&class_s) {
        leading_classes.insert(class_s);
    } else if class_symbols.contains_key(&class_ts) {
        leading_classes.insert(class_ts);
    };

    // Third highest priority: identity
    leading_classes.insert(class_e);

    // Fourth highest priority: principal classes, if not yet encountered
    leading_classes.extend(principal_classes.iter().cloned());

    log::debug!("Irreducible representation sort order:");
    for leading_cc in leading_classes.iter() {
        log::debug!("  {}", leading_cc);
    }

    let leading_idxs: IndexSet<usize> = leading_classes
        .iter()
        .map(|cc| {
            *class_symbols
                .get(cc)
                .unwrap_or_else(|| panic!("Class `{cc}` cannot be found."))
        })
        .collect();

    let n_rows = char_arr.nrows();
    let mut col_idxs: Vec<usize> = Vec::with_capacity(n_rows);
    col_idxs.extend(leading_idxs.iter());
    col_idxs.extend((1..n_rows).filter(|i| !leading_idxs.contains(i)));
    let sort_arr = char_arr.select(Axis(1), &col_idxs);

    let sort_row_indices: Vec<_> = (0..n_rows)
        .sorted_by(|&i, &j| {
            let keys_i = sort_arr.row(i).iter().cloned().collect_vec();
            let keys_j = sort_arr.row(j).iter().cloned().collect_vec();
            keys_i
                .partial_cmp(&keys_j)
                .unwrap_or_else(|| panic!("`{keys_i:?}` and `{keys_j:?}` cannot be compared."))
        })
        .collect();
    let char_arr = char_arr.select(Axis(0), &sort_row_indices);
    let old_fs = frobenius_schur_indicators.iter().collect::<Vec<_>>();
    let sorted_fs = sort_row_indices.iter().map(|&i| *old_fs[i]).collect_vec();
    log::debug!("Sorting irreducible representations... Done.");
    (char_arr, sorted_fs)
}

/// Determines the principal classes given a list of class symbols and any forcing conditions.
///
/// By default, the principal classes are those with the highest order, regardless of whether they
/// are proper or improper, unitary or antiunitary.
///
/// # Arguments
///
/// * `class_symbols` - An indexmap of class symbols and their corresponding indices.
/// * `force_proper_principal` - A flag indicating if the principal classes are forced to be
/// proper.
/// * `force_principal` - An option containing specific classes that are forced to be principal.
///
/// # Returns
///
/// A vector of symbols of principal classes.
///
/// # Panics
///
/// Panics when:
///
/// * both `force_proper_principal` and `force_principal` are specified;
/// * classes specified in `force_principal` cannot be found in `class_symbols`;
/// * no principal classes can be found.
pub fn deduce_principal_classes<R, P>(
    class_symbols: &IndexMap<SymmetryClassSymbol<R>, usize>,
    force_principal_predicate: Option<P>,
    force_principal: Option<SymmetryClassSymbol<R>>,
) -> Vec<SymmetryClassSymbol<R>>
where
    R: fmt::Debug + SpecialSymmetryTransformation + FiniteOrder + Clone,
    P: Copy + Fn(&SymmetryClassSymbol<R>) -> bool,
{
    log::debug!("Determining principal classes...");
    let principal_classes = force_principal.map_or_else(|| {
        // sorted_class_symbols contains class symbols sorted in the order:
        // - decreasing operation order
        // - unitary operations, then antiunitary operations
        // - proper operations, then improper operations in each operation order
        let mut sorted_class_symbols = class_symbols
            .clone()
            .sorted_by(|cc1, _, cc2, _| {
                PartialOrd::partial_cmp(
                    &(cc1.order(), !cc1.is_antiunitary(), cc1.is_proper()),
                    &(cc2.order(), !cc2.is_antiunitary(), cc2.is_proper()),
                )
                .expect("Unable to sort class symbols.")
            })
            .rev();
        let (principal_rep, _) = if let Some(predicate) = force_principal_predicate {
            // Larger order always prioritised, regardless of proper or improper,
            // unless force_proper_principal is set, in which case
            sorted_class_symbols
                .find(|(cc, _)| predicate(cc))
                .expect("`Unable to find classes fulfilling the specified predicate.`")
        } else {
            // No force_proper_principal, so proper prioritised, which has been ensured by the sort
            // in the construction of `sorted_class_symbols`.
            sorted_class_symbols
                .next()
                .expect("Unexpected empty `sorted_class_symbols`.")
        };

        // Now find all principal classes with the same order, parity, and unitarity as
        // `principal_rep`.
        class_symbols
            .keys()
            .filter(|cc| {
                cc.is_antiunitary() == principal_rep.is_antiunitary()
                && cc.is_proper() == principal_rep.is_proper()
                && cc.order() == principal_rep.order()
            })
            .cloned()
            .collect::<Vec<_>>()
    }, |principal_cc| {
        assert!(
            force_principal_predicate.is_none(),
            "`force_principal_predicate` and `force_principal` cannot be both provided."
        );
        assert!(
            class_symbols.contains_key(&principal_cc),
            "Forcing principal-axis class to be `{principal_cc}`, but `{principal_cc}` is not a valid class in the group."
        );

        log::warn!(
            "Principal-axis class forced to be `{principal_cc}`. Auto-detection of principal-axis classes will be skipped.",
        );
        vec![principal_cc]
    });

    assert!(!principal_classes.is_empty());
    if principal_classes.len() == 1 {
        log::debug!("Principal-axis class found: {}", principal_classes[0]);
    } else {
        log::debug!("Principal-axis classes found:");
        for princc in &principal_classes {
            log::debug!("  {}", princc);
        }
    }
    log::debug!("Determining principal classes... Done.");
    principal_classes
}

/// Deduces irreducible representation symbols based on Mulliken's convention.
///
/// # Arguments
///
/// * `char_arr` - A view of a two-dimensional square array containing the characters where
/// each column is for one conjugacy class and each row one irrep.
/// * `class_symbols` - An index map containing the conjugacy class symbols for the columns of
/// `char_arr`. The keys are the symbols and the values are the column indices.
/// * `principal_classes` - The principal classes to be used for irrep symbol deduction.
///
/// # Returns
///
/// A vector of Mulliken symbols corresponding to the rows of `char_arr`.
///
/// # Panics
///
/// Panics when expected classes cannot be found in `class_symbols`.
#[allow(clippy::too_many_lines)]
pub(super) fn deduce_mulliken_irrep_symbols<R>(
    char_arr: &ArrayView2<Character>,
    class_symbols: &IndexMap<SymmetryClassSymbol<R>, usize>,
    principal_classes: &[SymmetryClassSymbol<R>],
) -> Vec<MullikenIrrepSymbol>
where
    R: fmt::Debug + SpecialSymmetryTransformation + FiniteOrder + Clone,
{
    log::debug!("Generating Mulliken irreducible representation symbols...");

    let e_cc = class_symbols
        .first()
        .expect("No class symbols found.")
        .0
        .clone();
    let i_cc: SymmetryClassSymbol<R> = SymmetryClassSymbol::new("1||i||", None)
        .expect("Unable to construct class symbol `1||i||`.");
    let ti_cc: SymmetryClassSymbol<R> = SymmetryClassSymbol::new("1||θ·i||", None)
        .expect("Unable to construct class symbol `1||θ·i||`.");
    let s_cc: SymmetryClassSymbol<R> = SymmetryClassSymbol::new("1||σh||", None)
        .expect("Unable to construct class symbol `1||σh||`.");
    let ts_cc: SymmetryClassSymbol<R> = SymmetryClassSymbol::new("1||θ·σh||", None)
        .expect("Unable to construct class symbol `1||θ·σh||`.");
    let t_cc: SymmetryClassSymbol<R> = SymmetryClassSymbol::new("1||θ||", None)
        .expect("Unable to construct class symbol `1||θ||`.");

    // Inversion parity?
    let i_parity = class_symbols.contains_key(&i_cc);

    // Time-reversed inversion parity?
    let ti_parity = class_symbols.contains_key(&ti_cc);

    // Reflection parity?
    let s_parity = class_symbols.contains_key(&s_cc);

    // Time-reversed reflection parity?
    let ts_parity = class_symbols.contains_key(&ts_cc);

    // i_parity takes priority.
    if i_parity {
        log::debug!("Inversion centre found. This will be used for g/u ordering.");
    } else if ti_parity {
        log::debug!(
            "Time-reversed inversion centre found (but no inversion centre). This will be used for g/u ordering."
        );
    } else if s_parity {
        log::debug!(
            "Horizontal mirror plane found (but no inversion centre). This will be used for '/'' ordering."
        );
    } else if ts_parity {
        log::debug!(
            "Time-reversed horizontal mirror plane found (but no inversion centre). This will be used for '/'' ordering."
        );
    }

    // Time-reversal?
    let t_parity = class_symbols.contains_key(&t_cc);
    if t_parity {
        log::debug!("Time reversal found. This will be used for magnetic ordering.");
    }

    let e2p1 = UnityRoot::new(1u32, 2u32);
    let e2p2 = UnityRoot::new(2u32, 2u32);
    let char_p1 = Character::new(&[(e2p2, 1usize)]);
    let char_m1 = Character::new(&[(e2p1, 1usize)]);

    // First pass: assign irrep symbols based on Mulliken's convention as much as possible.
    log::debug!("First pass: assign symbols from rules");

    let raw_irrep_symbols = char_arr.rows().into_iter().map(|irrep| {
        // Determine the main symmetry
        let dim = irrep[
            *class_symbols.get(&e_cc).unwrap_or_else(|| panic!("Class `{}` not found.", &e_cc))
        ].complex_value();
        assert!(
            approx::relative_eq!(dim.im, 0.0)
                && approx::relative_eq!(dim.re.round(), dim.re)
                && dim.re.round() > 0.0
        );

        assert!(dim.re.round() > 0.0);
        #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
        let dim = dim.re.round() as u64;
        assert!(dim > 0);
        let main = if dim >= 2 {
            INV_MULLIKEN_IRREP_DEGENERACIES.get(&dim).map_or_else(|| {
                log::warn!("{} cannot be assigned a standard dimensionality symbol. A generic 'Λ' will be used instead.", dim);
                "Λ"
            }, |sym| sym)
        } else {
            let char_rots: HashSet<_> = principal_classes
                .iter()
                .map(|cc| {
                    irrep[
                        *class_symbols.get(cc).unwrap_or_else(|| panic!("Class `{cc}` not found."))
                    ].clone()
                })
                .collect();
            if char_rots.len() == 1 && *char_rots.iter().next().expect("No rotation classes found.") == char_p1 {
                "A"
            } else if char_rots
                .iter()
                .all(|char_rot| char_rot.clone() == char_p1 || char_rot.clone() == char_m1)
            {
                "B"
            } else {
                // There are principal rotations but with non-(±1) characters. These must be
                // complex.
                "Γ"
            }
        };

        let (inv, mir) = if i_parity || ti_parity {
            // Determine inversion symmetry
            // Inversion symmetry trumps reflection symmetry.
            let char_inv = irrep[
                *class_symbols
                    .get(&i_cc)
                    .unwrap_or_else(|| {
                        class_symbols
                            .get(&ti_cc)
                            .unwrap_or_else(|| {
                                panic!("Neither `{}` nor `{}` found.", &i_cc, &ti_cc)
                            })
                    })
            ].clone();
            let char_inv_c = char_inv.complex_value();
            assert!(
                approx::relative_eq!(
                    char_inv_c.im,
                    0.0,
                    epsilon = char_inv.threshold,
                    max_relative = char_inv.threshold
                ) && approx::relative_eq!(
                    char_inv_c.re.round(),
                    char_inv_c.re,
                    epsilon = char_inv.threshold,
                    max_relative = char_inv.threshold
                ),
            );

            #[allow(clippy::cast_possible_truncation)]
            let char_inv_c = char_inv_c.re.round() as i32;
            match char_inv_c.cmp(&0) {
                Ordering::Greater => ("g", ""),
                Ordering::Less => ("u", ""),
                Ordering::Equal => panic!("Inversion character must not be zero."),
            }
        } else if s_parity || ts_parity {
            // Determine reflection symmetry
            let char_ref = irrep[
                *class_symbols
                    .get(&s_cc)
                    .unwrap_or_else(|| {
                        class_symbols
                            .get(&ts_cc)
                            .unwrap_or_else(|| {
                                panic!("Neither `{}` nor `{}` found.", &s_cc, &ts_cc)
                                })
                    })
            ].clone();
            let char_ref_c = char_ref.complex_value();
            assert!(
                approx::relative_eq!(
                    char_ref_c.im,
                    0.0,
                    epsilon = char_ref.threshold,
                    max_relative = char_ref.threshold
                ) && approx::relative_eq!(
                    char_ref_c.re.round(),
                    char_ref_c.re,
                    epsilon = char_ref.threshold,
                    max_relative = char_ref.threshold
                ),
            );

            #[allow(clippy::cast_possible_truncation)]
            let char_ref_c = char_ref_c.re.round() as i32;
            match char_ref_c.cmp(&0) {
                Ordering::Greater => ("", "'"),
                Ordering::Less => ("", "''"),
                Ordering::Equal => panic!("Reflection character must not be zero."),
            }
        } else {
            ("", "")
        };


        let trev = if t_parity {
            // Determine time-reversal symmetry
            let char_trev = irrep[
                *class_symbols.get(&t_cc).unwrap_or_else(|| {
                    panic!("Class `{}` not found.", &t_cc)
                })
            ].clone();
            let char_trev_c = char_trev.complex_value();
            assert!(
                approx::relative_eq!(
                    char_trev_c.im,
                    0.0,
                    epsilon = char_trev.threshold,
                    max_relative = char_trev.threshold
                ) && approx::relative_eq!(
                    char_trev_c.re.round(),
                    char_trev_c.re,
                    epsilon = char_trev.threshold,
                    max_relative = char_trev.threshold
                ),
            );

            #[allow(clippy::cast_possible_truncation)]
            let char_trev_c = char_trev_c.re.round() as i32;
            match char_trev_c.cmp(&0) {
                Ordering::Greater => "",
                Ordering::Less => "m",
                Ordering::Equal => panic!("Time-reversal character must not be zero."),
            }
        } else {
            ""
        };

        MullikenIrrepSymbol::new(format!("|^({trev})|{main}|^({mir})_({inv})|").as_str())
            .unwrap_or_else(|_| {
                panic!(
                    "Unable to construct symmetry symbol `|^({trev})|{main}|^({mir})_({inv})|`."
                )
            })
    });

    log::debug!("Second pass: disambiguate identical cases not distinguishable by rules");
    let irrep_symbols = disambiguate_linspace_symbols(raw_irrep_symbols);
    log::debug!("Generating Mulliken irreducible representation symbols... Done.");
    irrep_symbols
}

/// Determines the mirror-plane symbol given a principal axis.
///
/// # Arguments
///
/// * `sigma_axis` - The normalised normal vector of a mirror plane.
/// * `principal_axis` - The normalised principal rotation axis.
/// * `thresh` - Threshold for comparisons.
/// * `force_d` - Flag indicating if vertical mirror planes should be given the $`d`$ symbol
/// instead of $`v`$.
///
/// # Returns
///
/// The mirror-plane symbol.
#[must_use]
pub(super) fn deduce_sigma_symbol(
    sigma_axis: &Vector3<f64>,
    principal_element: &SymmetryElement,
    thresh: f64,
    force_d: bool,
) -> Option<String> {
    if approx::relative_eq!(
        principal_element.raw_axis.dot(sigma_axis).abs(),
        0.0,
        epsilon = thresh,
        max_relative = thresh
    ) && principal_element.proper_order != ORDER_1
    {
        // Vertical plane containing principal axis
        if force_d {
            Some("d".to_owned())
        } else {
            Some("v".to_owned())
        }
    } else if approx::relative_eq!(
        principal_element.raw_axis.cross(sigma_axis).norm(),
        0.0,
        epsilon = thresh,
        max_relative = thresh
    ) && principal_element.proper_order != ORDER_1
    {
        // Horizontal plane perpendicular to principal axis
        Some("h".to_owned())
    } else {
        None
    }
}
