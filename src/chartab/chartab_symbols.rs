use std::fmt;
use std::hash::Hash;
use std::str::FromStr;
use std::collections::{HashMap, VecDeque};

use counter::Counter;
use derive_builder::Builder;
use phf::phf_map;
use regex::Regex;

pub static FROBENIUS_SCHUR_SYMBOLS: phf::Map<i8, &'static str> = phf_map! {
    1i8 => "r",
    0i8 => "c",
    -1i8 => "q",
};

// ======
// Traits
// ======

/// A trait for general mathematical symbols.
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

    /// The multiplicity of the symbol.
    fn multiplicity(&self) -> Option<usize>;
}

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

/// A trait for symbols describing reducible linear spaces.
pub trait ReducibleLinearSpaceSymbol: LinearSpaceSymbol
where
    Self::Subspace: LinearSpaceSymbol,
{
    /// The type of the subspace symbols.
    type Subspace;

    /// Constructs [`Self`] from constituting subspace symbols and their multiplicities.
    fn from_subspaces(subspaces: &[(Self::Subspace, usize)]) -> Self;

    /// Returns the constituting subspace symbols and their multiplicities.
    fn subspaces(&self) -> Vec<(&Self::Subspace, &usize)>;
}

/// A trait for symbols describing collections of objects.
pub trait CollectionSymbol: MathematicalSymbol {
    type CollectionElement;

    /// Constructs a collection symbol from a string and a representative collection element.
    ///
    /// # Arguments
    ///
    /// * `symstr` - A string to be parsed.
    /// * `rep` - A representative collection element.
    ///
    /// # Errors
    ///
    /// Returns an error if `symstr` cannot be parsed.
    fn from_rep(
        symstr: &str,
        rep: Option<Self::CollectionElement>,
    ) -> Result<Self, GenericSymbolParsingError>;

    /// The size of the collection.
    fn size(&self) -> usize;

    /// The representative element of the collection.
    fn representative(&self) -> Option<Self::CollectionElement>;
}

// =======
// Structs
// =======

// -------------
// GenericSymbol
// -------------

/// A struct to handle generic mathematical symbols.
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
#[derive(Builder, Debug, Clone, PartialEq, Eq, Hash, PartialOrd)]
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
    pub fn set_main(&mut self, main: &str) {
        self.main = main.to_string();
    }
}

// ------------------
// MathematicalSymbol
// ------------------

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

// -------
// FromStr
// -------

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

// -------
// Display
// -------
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

pub fn disambiguate_irrep_symbols<S>(raw_irrep_symbols: impl Iterator<Item = S> + Clone) -> Vec<S>
where
    S: LinearSpaceSymbol
{
    let raw_symbol_count = raw_irrep_symbols.clone().collect::<Counter<S>>();
    let mut raw_symbols_to_full_symbols: HashMap<S, VecDeque<S>> = raw_symbol_count
        .iter()
        .map(|(raw_irrep, &duplicate_count)| {
            if duplicate_count == 1 {
                let mut irreps: VecDeque<S> = VecDeque::new();
                irreps.push_back(raw_irrep.clone());
                (raw_irrep.clone(), irreps)
            } else {
                let irreps: VecDeque<S> = (0..duplicate_count)
                    .map(|i| {
                        let mut new_irrep = S::from_str(
                            &format!(
                                "|^({})|{}|^({})_({}{})|",
                                raw_irrep.presuper(),
                                raw_irrep.main(),
                                raw_irrep.postsuper(),
                                i + 1,
                                raw_irrep.postsub(),
                            )
                        )
                        .unwrap_or_else(|_| {
                            panic!(
                                "Unable to construct symmetry symbol `|^({})|{}|^({})_({}{})|`.",
                                raw_irrep.presuper(),
                                raw_irrep.main(),
                                raw_irrep.postsuper(),
                                i + 1,
                                raw_irrep.postsub(),
                            )
                        });
                        new_irrep.set_dimensionality(raw_irrep.dimensionality());
                        new_irrep
                    })
                    .collect();
                (raw_irrep.clone(), irreps)
            }
        })
        .collect();

    let irrep_symbols: Vec<S> = raw_irrep_symbols
        .map(|raw_irrep| {
            raw_symbols_to_full_symbols
                .get_mut(&raw_irrep)
                .unwrap_or_else(|| {
                    panic!(
                        "Unknown conversion of raw symbol `{}` to full symbol.",
                        &raw_irrep
                    )
                })
                .pop_front()
                .unwrap_or_else(|| {
                    panic!("No conversion to full symbol possible for `{}`", &raw_irrep)
                })
        })
        .collect();

    irrep_symbols
}
