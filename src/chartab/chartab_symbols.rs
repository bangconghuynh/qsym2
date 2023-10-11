//! Symbols enumerating rows and columns of character tables.

use std::cmp::Ordering;
use std::collections::{HashMap, VecDeque};
use std::error::Error;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::str::FromStr;

use counter::Counter;
use derive_builder::Builder;
use indexmap::IndexMap;
use itertools::Itertools;
use phf::phf_map;
use regex::Regex;
use serde::{Deserialize, Serialize};

/// Symbols for Frobenius--Schur classifications of irreducible representations.
pub static FROBENIUS_SCHUR_SYMBOLS: phf::Map<i8, &'static str> = phf_map! {
    1i8 => "r",
    0i8 => "c",
    -1i8 => "q",
};

// =================
// Trait definitions
// =================

// ------------------
// MathematicalSymbol
// ------------------

/// A trait for general mathematical symbols. See [`GenericSymbol`] for the definitions of the
/// parts.
pub trait MathematicalSymbol: Clone + Hash + Eq + fmt::Display {
    /// The main part of the symbol.
    fn main(&self) -> String;

    /// The pre-superscript part of the symbol.
    fn presuper(&self) -> String;

    /// The pre-subscript part of the symbol.
    fn presub(&self) -> String;

    /// The post-superscript part of the symbol.
    fn postsuper(&self) -> String;

    /// The post-subscript part of the symbol.
    fn postsub(&self) -> String;

    /// The prefactor part of the symbol.
    fn prefactor(&self) -> String;

    /// The postfactor part of the symbol.
    fn postfactor(&self) -> String;

    /// The multiplicity of the symbol which can have different meanings depending on the exact
    /// nature of the mathematical symbol.
    fn multiplicity(&self) -> Option<usize>;
}

// -----------------
// LinearSpaceSymbol
// -----------------

/// A trait for symbols describing linear spaces.
pub trait LinearSpaceSymbol: MathematicalSymbol + FromStr {
    /// The dimensionality of the linear space.
    fn dimensionality(&self) -> usize;

    /// Sets the dimensionality of the linear space for the symbol.
    ///
    /// # Arguments
    ///
    /// * `dim` - The dimensionality to be set.
    ///
    /// # Returns
    ///
    /// Returns `true` if the dimensionality has been successfully set.
    fn set_dimensionality(&mut self, dim: usize) -> bool;
}

// --------------------------
// ReducibleLinearSpaceSymbol
// --------------------------

// ~~~~~~~~~~~~~~~~
// Trait definition
// ~~~~~~~~~~~~~~~~

/// A trait for symbols describing reducible linear spaces.
pub trait ReducibleLinearSpaceSymbol: LinearSpaceSymbol
where
    Self::Subspace: LinearSpaceSymbol + PartialOrd,
{
    /// The type of the subspace symbols.
    type Subspace;

    /// Constructs [`Self`] from constituting subspace symbols and their multiplicities.
    fn from_subspaces(subspaces: &[(Self::Subspace, usize)]) -> Self;

    /// Returns the constituting subspace symbols and their multiplicities.
    fn subspaces(&self) -> Vec<(&Self::Subspace, &usize)>;

    // ----------------
    // Provided methods
    // ----------------

    /// Returns an iterator containing sorted references to the constituting symbols.
    ///
    /// # Panics
    ///
    /// Panics if the constituting symbols cannot be ordered.
    #[must_use]
    fn sorted_subspaces(&self) -> Vec<(&Self::Subspace, &usize)> {
        self.subspaces()
            .iter()
            .sorted_by(|(a, _), (b, _)| {
                a.partial_cmp(b)
                    .unwrap_or_else(|| panic!("`{a}` and `{b}` cannot be compared."))
            })
            .cloned()
            .collect::<Vec<_>>()
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~
// Blanket implementation
// ~~~~~~~~~~~~~~~~~~~~~~

impl<R> MathematicalSymbol for R
where
    R: ReducibleLinearSpaceSymbol,
{
    /// The main part of the symbol.
    fn main(&self) -> String {
        self.subspaces()
            .iter()
            .map(|(irrep, &mult)| {
                format!(
                    "{}{irrep}",
                    if mult != 1 {
                        mult.to_string()
                    } else {
                        String::new()
                    }
                )
            })
            .join(" ⊕ ")
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

    /// The prefactor part of the symbol, which is always `"1"`.
    fn prefactor(&self) -> String {
        "1".to_string()
    }

    /// The postfactor part of the symbol, which is always empty.
    fn postfactor(&self) -> String {
        String::new()
    }

    /// The multiplicity of the symbol, which is always `"1"`.
    fn multiplicity(&self) -> Option<usize> {
        Some(1)
    }
}

impl<R> LinearSpaceSymbol for R
where
    R: ReducibleLinearSpaceSymbol,
{
    fn dimensionality(&self) -> usize {
        self.subspaces()
            .iter()
            .map(|(symbol, &mult)| symbol.dimensionality() * mult)
            .sum()
    }

    fn set_dimensionality(&mut self, _: usize) -> bool {
        log::error!("The dimensionality of `{self}` cannot be set.");
        false
    }
}

// ----------------
// CollectionSymbol
// ----------------

/// A trait for symbols describing collections of objects such as conjugacy classes.
pub trait CollectionSymbol: MathematicalSymbol {
    /// The type of the elements in the collection.
    type CollectionElement;

    /// Constructs a collection symbol from a string and one or more representative collection
    /// elements.
    ///
    /// # Arguments
    ///
    /// * `symstr` - A string to be parsed. See [`GenericSymbol::from_str`] for more information.
    /// * `reps` - An optional vector of one or more representative collection elements.
    ///
    /// # Errors
    ///
    /// Returns an error if `symstr` cannot be parsed.
    fn from_reps(
        symstr: &str,
        reps: Option<Vec<Self::CollectionElement>>,
    ) -> Result<Self, GenericSymbolParsingError>;

    /// The first representative element of the collection.
    fn representative(&self) -> Option<&Self::CollectionElement>;

    /// All representative elements of the collection.
    fn representatives(&self) -> Option<&Vec<Self::CollectionElement>>;

    /// The size of the collection which is given by the number of representative elements
    /// multiplied by the multiplicity of the symbol. If no representative elements exist, then the
    /// size is taken to be the multiplicity of the symbol itself.
    fn size(&self) -> usize {
        self.multiplicity().unwrap_or_else(|| {
            panic!(
                "Unable to deduce the multiplicity of the class from the prefactor {}.",
                self.prefactor()
            )
        }) * self
            .representatives()
            .map(|reps| reps.len())
            .expect("No representatives found.")
    }
}

// =======
// Structs
// =======

// -------------
// GenericSymbol
// -------------

// ~~~~~~~~~~~~~~~~~
// Struct definition
// ~~~~~~~~~~~~~~~~~

/// A structure to handle generic mathematical symbols.
///
/// Each generic symbol has the format
///
/// ```math
/// \textrm{prefactor}
/// \ ^{\textrm{presuper}}_{\textrm{presub}}
/// \ \textrm{main}
/// \ ^{\textrm{postsuper}}_{\textrm{postsub}}
/// \ \textrm{postfactor}.
/// ```
#[derive(Builder, Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Serialize, Deserialize)]
pub struct GenericSymbol {
    /// The main part of the symbol.
    main: String,

    /// The pre-superscript part of the symbol.
    #[builder(default = "String::new()")]
    presuper: String,

    /// The pre-subscript part of the symbol.
    #[builder(default = "String::new()")]
    presub: String,

    /// The post-superscript part of the symbol.
    #[builder(default = "String::new()")]
    postsuper: String,

    /// The post-subscript part of the symbol.
    #[builder(default = "String::new()")]
    postsub: String,

    /// The prefactor part of the symbol.
    #[builder(default = "String::new()")]
    prefactor: String,

    /// The postfactor part of the symbol.
    #[builder(default = "String::new()")]
    postfactor: String,
}

impl GenericSymbol {
    fn builder() -> GenericSymbolBuilder {
        GenericSymbolBuilder::default()
    }

    /// Sets the main part of the symbol.
    pub(crate) fn set_main(&mut self, main: &str) {
        self.main = main.to_string();
    }

    /// Sets the pre-subscript part of the symbol.
    pub(crate) fn set_presub(&mut self, presub: &str) {
        self.presub = presub.to_string();
    }

    /// Sets the post-subscript part of the symbol.
    pub(crate) fn set_postsub(&mut self, postsub: &str) {
        self.postsub = postsub.to_string();
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Trait implementation for GenericSymbol
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

impl MathematicalSymbol for GenericSymbol {
    fn main(&self) -> String {
        self.main.clone()
    }

    fn presuper(&self) -> String {
        self.presuper.clone()
    }

    fn presub(&self) -> String {
        self.presub.clone()
    }

    fn postsuper(&self) -> String {
        self.postsuper.clone()
    }

    fn postsub(&self) -> String {
        self.postsub.clone()
    }

    fn prefactor(&self) -> String {
        self.prefactor.clone()
    }

    fn postfactor(&self) -> String {
        self.postfactor.clone()
    }

    fn multiplicity(&self) -> Option<usize> {
        str::parse::<usize>(&self.prefactor).ok()
    }
}

impl FromStr for GenericSymbol {
    type Err = GenericSymbolParsingError;

    /// Parses a string representing a generic symbol.
    ///
    /// Some permissible generic symbols:
    ///
    /// ```text
    /// "T"
    /// "||T|_(2g)|"
    /// "|^(3)|T|_(2g)|"
    /// "12||C|^(2)_(5)|"
    /// "2||S|^(z)|(α)"
    /// ```
    fn from_str(symstr: &str) -> Result<Self, Self::Err> {
        let strs: Vec<&str> = symstr.split('|').collect();
        if strs.len() == 1 {
            Ok(Self::builder()
                .main(strs[0].to_string())
                .build()
                .unwrap_or_else(|_| {
                    panic!("Unable to construct a generic symbol from `{symstr}`.")
                }))
        } else if strs.len() == 5 {
            let prefacstr = strs[0];
            let prestr = strs[1];
            let mainstr = strs[2];
            let poststr = strs[3];
            let postfacstr = strs[4];

            let presuper_re = Regex::new(r"\^\((.*?)\)").expect("Regex pattern invalid.");
            let presuperstr = if let Some(cap) = presuper_re.captures(prestr) {
                cap.get(1)
                    .expect("Expected regex group cannot be captured.")
                    .as_str()
            } else {
                ""
            };

            let presub_re = Regex::new(r"_\((.*?)\)").expect("Regex pattern invalid.");
            let presubstr = if let Some(cap) = presub_re.captures(prestr) {
                cap.get(1)
                    .expect("Expected regex group cannot be captured.")
                    .as_str()
            } else {
                ""
            };

            let postsuper_re = Regex::new(r"\^\((.*?)\)").expect("Regex pattern invalid.");
            let postsuperstr = if let Some(cap) = postsuper_re.captures(poststr) {
                cap.get(1)
                    .expect("Expected regex group cannot be captured.")
                    .as_str()
            } else {
                ""
            };

            let postsub_re = Regex::new(r"_\((.*?)\)").expect("Regex pattern invalid.");
            let postsubstr = if let Some(cap) = postsub_re.captures(poststr) {
                cap.get(1)
                    .expect("Expected regex group cannot be captured.")
                    .as_str()
            } else {
                ""
            };

            Ok(Self::builder()
                .main(mainstr.to_string())
                .presuper(presuperstr.to_string())
                .presub(presubstr.to_string())
                .postsuper(postsuperstr.to_string())
                .postsub(postsubstr.to_string())
                .prefactor(prefacstr.to_string())
                .postfactor(postfacstr.to_string())
                .build()
                .unwrap_or_else(|_| {
                    panic!("Unable to construct a generic symbol from `{symstr}`.")
                }))
        } else {
            Err(GenericSymbolParsingError(format!(
                "`{symstr}` is not parsable."
            )))
        }
    }
}

impl fmt::Display for GenericSymbol {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let prefac_str = if self.prefactor() == "1" {
            String::new()
        } else {
            self.prefactor()
        };
        let presuper_str = if self.presuper().is_empty() {
            String::new()
        } else {
            format!("^({})", self.presuper())
        };
        let presub_str = if self.presub().is_empty() {
            String::new()
        } else {
            format!("_({})", self.presub())
        };
        let main_str = format!("|{}|", self.main());
        let postsuper_str = if self.postsuper().is_empty() {
            String::new()
        } else {
            format!("^({})", self.postsuper())
        };
        let postsub_str = if self.postsub().is_empty() {
            String::new()
        } else {
            format!("_({})", self.postsub())
        };
        let postfac_str = if self.postfactor() == "1" {
            String::new()
        } else {
            self.postfactor()
        };
        write!(
            f,
            "{prefac_str}{presuper_str}{presub_str}{main_str}{postsuper_str}{postsub_str}{postfac_str}",
        )
    }
}

#[derive(Debug, Clone)]
pub struct GenericSymbolParsingError(pub String);

impl fmt::Display for GenericSymbolParsingError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Generic symbol parsing error: {}.", self.0)
    }
}

impl Error for GenericSymbolParsingError {}

// ----------------
// DecomposedSymbol
// ----------------

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Trait implementation for DecomposedSymbol
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// A struct to handle symbols consisting of multiple sub-symbols.
#[derive(Builder, Debug, Clone, Eq, Serialize, Deserialize)]
pub struct DecomposedSymbol<S>
where
    S: LinearSpaceSymbol + PartialOrd,
{
    symbols: IndexMap<S, usize>,
}

impl<S> DecomposedSymbol<S>
where
    S: LinearSpaceSymbol + PartialOrd,
{
    fn builder() -> DecomposedSymbolBuilder<S> {
        DecomposedSymbolBuilder::<S>::default()
    }

    /// Parses a string representing a decomposed symbol. See [`Self::from_str`] for more
    /// information.
    ///
    /// # Arguments
    ///
    /// * `symstr` - A string to be parsed to give a decomposed symbol.
    ///
    /// # Errors
    ///
    /// Errors when the string cannot be parsed as a decomposed symbol.
    pub fn new(symstr: &str) -> Result<Self, DecomposedSymbolBuilderError> {
        Self::from_str(symstr)
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Trait implementation for DecomposedSymbol
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

impl<S> ReducibleLinearSpaceSymbol for DecomposedSymbol<S>
where
    S: LinearSpaceSymbol + PartialOrd,
{
    type Subspace = S;

    fn from_subspaces(irreps: &[(Self::Subspace, usize)]) -> Self {
        Self::builder()
            .symbols(
                irreps
                    .iter()
                    .filter(|(_, mult)| *mult != 0)
                    .cloned()
                    .collect::<IndexMap<_, _>>(),
            )
            .build()
            .expect("Unable to construct a decomposed symbol from a slice of symbols.")
    }

    fn subspaces(&self) -> Vec<(&Self::Subspace, &usize)> {
        self.symbols.iter().collect::<Vec<_>>()
    }
}

impl<S> FromStr for DecomposedSymbol<S>
where
    S: LinearSpaceSymbol + PartialOrd + FromStr,
{
    type Err = DecomposedSymbolBuilderError;

    /// Parses a string representing a decomposed symbol. A valid string representing a
    /// decomposed symbol is one consisting of one or more valid symbol strings, separated by a `+`
    /// character.
    ///
    /// # Arguments
    ///
    /// * `symstr` - A string to be parsed to give a decomposed symbol.
    ///
    /// # Returns
    ///
    /// A [`Result`] wrapping the constructed decomposed symbol.
    ///
    /// # Panics
    ///
    /// Panics when unable to construct a decomposed symbol from the specified string.
    ///
    /// # Errors
    ///
    /// Errors when the string cannot be parsed.
    fn from_str(symstr: &str) -> Result<Self, Self::Err> {
        let re = Regex::new(r"(\d?)(.*)").expect("Regex pattern invalid.");
        let symbols = symstr
            .split('⊕')
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
                    S::from_str(irrep)
                        .unwrap_or_else(|_| panic!("Unable to parse {irrep} as a valid symbol.")),
                    mult,
                )
            })
            .collect::<IndexMap<_, _>>();
        Self::builder().symbols(symbols).build()
    }
}

impl<S> Hash for DecomposedSymbol<S>
where
    S: LinearSpaceSymbol + PartialOrd,
{
    fn hash<H: Hasher>(&self, state: &mut H) {
        for symbol in self.sorted_subspaces() {
            symbol.hash(state);
        }
    }
}

impl<S> PartialOrd for DecomposedSymbol<S>
where
    S: LinearSpaceSymbol + PartialOrd,
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        let self_subspaces = self.sorted_subspaces();
        let other_subspaces = other.sorted_subspaces();
        self_subspaces.partial_cmp(&other_subspaces)
    }
}

impl<S> fmt::Display for DecomposedSymbol<S>
where
    S: LinearSpaceSymbol + PartialOrd,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.main())
    }
}

impl<S> PartialEq for DecomposedSymbol<S>
where
    S: LinearSpaceSymbol + PartialOrd,
{
    fn eq(&self, other: &Self) -> bool {
        let self_subspaces = self.sorted_subspaces();
        let other_subspaces = other.sorted_subspaces();
        self_subspaces == other_subspaces
    }
}

// ================
// Helper functions
// ================

/// Disambiguates linear-space labelling symbols that cannot be otherwise distinguished from rules.
///
/// This essentially appends appropriate roman subscripts to otherwise identical symbols.
///
/// # Arguments
///
/// * `raw_symbols` - An iterator of raw symbols, some of which might be identical.
///
/// # Returns
///
/// A vector of disambiguated symbols.
pub(crate) fn disambiguate_linspace_symbols<S>(
    raw_symbols: impl Iterator<Item = S> + Clone,
) -> Vec<S>
where
    S: LinearSpaceSymbol,
{
    let raw_symbol_count = raw_symbols.clone().collect::<Counter<S>>();
    let mut raw_symbols_to_full_symbols: HashMap<S, VecDeque<S>> = raw_symbol_count
        .iter()
        .map(|(raw_symbol, &duplicate_count)| {
            if duplicate_count == 1 {
                let mut symbols: VecDeque<S> = VecDeque::new();
                symbols.push_back(raw_symbol.clone());
                (raw_symbol.clone(), symbols)
            } else {
                let symbols: VecDeque<S> = (0..duplicate_count)
                    .map(|i| {
                        let mut new_symbol = S::from_str(&format!(
                            "|^({})_({})|{}|^({})_({}{})|",
                            raw_symbol.presuper(),
                            raw_symbol.presub(),
                            raw_symbol.main(),
                            raw_symbol.postsuper(),
                            i + 1,
                            raw_symbol.postsub(),
                        ))
                        .unwrap_or_else(|_| {
                            panic!(
                                "Unable to construct symmetry symbol `|^({})|{}|^({})_({}{})|`.",
                                raw_symbol.presuper(),
                                raw_symbol.main(),
                                raw_symbol.postsuper(),
                                i + 1,
                                raw_symbol.postsub(),
                            )
                        });
                        new_symbol.set_dimensionality(raw_symbol.dimensionality());
                        new_symbol
                    })
                    .collect();
                (raw_symbol.clone(), symbols)
            }
        })
        .collect();

    let symbols: Vec<S> = raw_symbols
        .map(|raw_symbol| {
            raw_symbols_to_full_symbols
                .get_mut(&raw_symbol)
                .unwrap_or_else(|| {
                    panic!(
                        "Unknown conversion of raw symbol `{}` to full symbol.",
                        &raw_symbol
                    )
                })
                .pop_front()
                .unwrap_or_else(|| {
                    panic!(
                        "No conversion to full symbol possible for `{}`",
                        &raw_symbol
                    )
                })
        })
        .collect();

    symbols
}
