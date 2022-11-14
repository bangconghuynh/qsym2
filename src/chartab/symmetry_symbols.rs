use std::fmt;
use std::hash::{Hash, Hasher};
use std::str::FromStr;

use derive_builder::Builder;
use phf::phf_map;
use regex::Regex;

use crate::symmetry::symmetry_element::symmetry_operation::SpecialSymmetryTransformation;

#[cfg(test)]
#[path = "symmetry_symbols_tests.rs"]
mod symmetry_symbols_tests;

// ======
// Traits
// ======

/// A trait for general mathematical symbols.
trait MathematicalSymbol: Hash {
    /// The main part of the symbol.
    fn main(&self) -> &str;

    /// The pre-superscript part of the symbol.
    fn presuper(&self) -> &str;

    /// The pre-subscript part of the symbol.
    fn presub(&self) -> &str;

    /// The post-superscript part of the symbol.
    fn postsuper(&self) -> &str;

    /// The post-subscript part of the symbol.
    fn postsub(&self) -> &str;

    /// The prefactor part of the symbol.
    fn prefactor(&self) -> &str;

    /// The postfactor part of the symbol.
    fn postfactor(&self) -> &str;
}

/// A trait for symbols describing linear spaces.
trait LinearSpaceSymbol: MathematicalSymbol {
    /// The dimensionality of the linear space.
    fn dimensionality(&self) -> u64;
}

/// A trait for symbols describing collections of objects.
trait CollectionSymbol: MathematicalSymbol {
    /// The size of the collection.
    fn size(&self) -> u64;
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
#[derive(Builder, Debug, Clone, PartialEq, Eq, Hash)]
struct GenericSymbol {
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
}

// ------------------
// MathematicalSymbol
// ------------------

impl MathematicalSymbol for GenericSymbol {
    fn main(&self) -> &str {
        &self.main
    }

    fn presuper(&self) -> &str {
        &self.presuper
    }

    fn presub(&self) -> &str {
        &self.presub
    }

    fn postsuper(&self) -> &str {
        &self.postsuper
    }

    fn postsub(&self) -> &str {
        &self.postsub
    }

    fn prefactor(&self) -> &str {
        &self.prefactor
    }

    fn postfactor(&self) -> &str {
        &self.postfactor
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
    /// ```
    /// "T"
    /// "||T|_(2g)|"
    /// "|^(3)|T|_(2g)|"
    /// "12||C|^(2)_(5)|"
    /// "2||S|^(z)|(α)"
    /// ```
    fn from_str(symstr: &str) -> Result<Self, Self::Err> {
        let strs: Vec<&str> = symstr.split('|').collect();
        if strs.len() == 1 {
            Ok(Self::builder().main(strs[0].to_string()).build().unwrap())
        } else if strs.len() == 5 {
            let prefacstr = strs[0];
            let prestr = strs[1];
            let mainstr = strs[2];
            let poststr = strs[3];
            let postfacstr = strs[4];

            let presuper_re = Regex::new(r"\^\((.*?)\)").unwrap();
            let presuperstr = if let Some(cap) = presuper_re.captures(prestr) {
                cap.get(1).unwrap().as_str()
            } else {
                ""
            };

            let presub_re = Regex::new(r"_\((.*?)\)").unwrap();
            let presubstr = if let Some(cap) = presub_re.captures(prestr) {
                cap.get(1).unwrap().as_str()
            } else {
                ""
            };

            let postsuper_re = Regex::new(r"\^\((.*?)\)").unwrap();
            let postsuperstr = if let Some(cap) = postsuper_re.captures(poststr) {
                cap.get(1).unwrap().as_str()
            } else {
                ""
            };

            let postsub_re = Regex::new(r"_\((.*?)\)").unwrap();
            let postsubstr = if let Some(cap) = postsub_re.captures(poststr) {
                cap.get(1).unwrap().as_str()
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
                .unwrap())
        } else {
            Err(GenericSymbolParsingError(symstr.to_string()))
        }
    }
}

// -------
// Display
// -------
impl fmt::Display for GenericSymbol {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let presuper_str = if !self.presuper().is_empty() {
            format!("^({})", self.presuper())
        } else {
            "".to_string()
        };
        let presub_str = if !self.presub().is_empty() {
            format!("_({})", self.presub())
        } else {
            "".to_string()
        };
        let main_str = format!("|{}|", self.main());
        let postsuper_str = if !self.postsuper().is_empty() {
            format!("^({})", self.postsuper())
        } else {
            "".to_string()
        };
        let postsub_str = if !self.postsub().is_empty() {
            format!("_({})", self.postsub())
        } else {
            "".to_string()
        };
        write!(
            f,
            "{}{}{}{}{}{}{}",
            self.prefactor(),
            presuper_str,
            presub_str,
            main_str,
            postsuper_str,
            postsub_str,
            self.postfactor()
        )
    }
}

#[derive(Debug, Clone)]
struct GenericSymbolParsingError(String);

impl fmt::Display for GenericSymbolParsingError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Unable to parse generic symbol {}.", self.0)
    }
}

// -------------------
// MullikenIrrepSymbol
// -------------------

/// A struct to handle Mulliken irreducible representation symbols.
#[derive(Builder, Debug, PartialEq, Eq, Hash)]
struct MullikenIrrepSymbol {
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
    /// ```
    /// "T"
    /// "||T|_(2g)|"
    /// "|^(3)|T|_(2g)|"
    /// ```
    fn new(symstr: &str) -> Result<Self, GenericSymbolParsingError> {
        Self::from_str(symstr)
    }
}

impl MathematicalSymbol for MullikenIrrepSymbol {
    /// The main part of the symbol, which primarily denotes the dimensionality of the irrep space.
    fn main(&self) -> &str {
        self.generic_symbol.main()
    }

    /// The pre-superscript part of the symbol, which can be used to denote antiunitary symmetries
    /// or spin multiplicities.
    fn presuper(&self) -> &str {
        self.generic_symbol.presuper()
    }

    fn presub(&self) -> &str {
        self.generic_symbol.presub()
    }

    /// The post-superscript part of the symbol, which denotes reflection parity.
    fn postsuper(&self) -> &str {
        self.generic_symbol.postsuper()
    }

    /// The post-subscript part of the symbol, which denotes inversion parity when available and
    /// which disambiguates similar irreps.
    fn postsub(&self) -> &str {
        self.generic_symbol.postsub()
    }

    /// The prefactor part of the symbol, which is always `"1"` implicitly because of irreducibility.
    fn prefactor(&self) -> &str {
        ""
    }

    /// The postfactor part of the symbol, which is always empty.
    fn postfactor(&self) -> &str {
        ""
    }
}

impl FromStr for MullikenIrrepSymbol {
    type Err = GenericSymbolParsingError;

    /// Parses a string representing a Mulliken irrep symbol.
    ///
    /// Some permissible Mulliken irrep symbols:
    ///
    /// ```
    /// "T"
    /// "||T|_(2g)|"
    /// "|^(3)|T|_(2g)|"
    /// ```
    fn from_str(symstr: &str) -> Result<Self, Self::Err> {
        let generic_symbol = GenericSymbol::from_str(symstr)?;
        Ok(Self::builder()
            .generic_symbol(generic_symbol)
            .build()
            .unwrap())
    }
}

static MULLIKEN_IRREP_DEGENERACIES: phf::Map<&'static str, u64> = phf_map! {
    "A" => 1u64,
    "B" => 1u64,
    "Σ" => 1u64,
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

impl LinearSpaceSymbol for MullikenIrrepSymbol {
    fn dimensionality(&self) -> u64 {
        *MULLIKEN_IRREP_DEGENERACIES
            .get(self.main())
            .unwrap_or_else(|| {
                panic!(
                    "Unknown dimensionality for Mulliken symbol {}.",
                    self.main()
                )
            })
    }
}

// -------
// Display
// -------
impl fmt::Display for MullikenIrrepSymbol {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.generic_symbol)
    }
}

// -----------
// ClassSymbol
// -----------

/// A struct to handle conjugacy class symbols.
#[derive(Builder, Debug)]
struct ClassSymbol<T: SpecialSymmetryTransformation + Clone> {
    /// The generic part of the symbol.
    generic_symbol: GenericSymbol,

    /// A representative element in the class.
    representative: T,
}

impl<T: SpecialSymmetryTransformation + Clone> PartialEq for ClassSymbol<T> {
    fn eq(&self, other: &Self) -> bool {
        self.generic_symbol == other.generic_symbol
    }
}

impl<T: SpecialSymmetryTransformation + Clone> Eq for ClassSymbol<T> {}

impl<T: SpecialSymmetryTransformation + Clone> Hash for ClassSymbol<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.generic_symbol.hash(state)
    }
}

impl<T: SpecialSymmetryTransformation + Clone> ClassSymbol<T> {
    fn builder() -> ClassSymbolBuilder<T> {
        ClassSymbolBuilder::default()
    }
}

impl<T: SpecialSymmetryTransformation + Clone> MathematicalSymbol for ClassSymbol<T> {
    /// The main part of the symbol, which denotes the representative symmetry operation.
    fn main(&self) -> &str {
        self.generic_symbol.main()
    }

    /// The pre-superscript part of the symbol, which is empty.
    fn presuper(&self) -> &str {
        ""
    }

    /// The pre-subscript part of the symbol, which is empty.
    fn presub(&self) -> &str {
        ""
    }

    /// The post-superscript part of the symbol, which is empty.
    fn postsuper(&self) -> &str {
        ""
    }

    /// The post-subscript part of the symbol, which is empty.
    fn postsub(&self) -> &str {
        ""
    }

    /// The prefactor part of the symbol, which denotes the size of the class.
    fn prefactor(&self) -> &str {
        self.generic_symbol.prefactor()
    }

    /// The postfactor part of the symbol, which is empty.
    fn postfactor(&self) -> &str {
        ""
    }
}

impl<T: SpecialSymmetryTransformation + Clone> CollectionSymbol for ClassSymbol<T> {
    fn size(&self) -> u64 {
        self.prefactor().parse::<u64>().unwrap_or_else(|_| {
            panic!(
                "Unable to deduce the size of the class from the prefactor {}.",
                self.prefactor()
            )
        })
    }
}

impl<T: SpecialSymmetryTransformation + Clone> ClassSymbol<T> {

    /// Creates a class symbol from a string and a representative element.
    ///
    /// Some permissible conjugacy class symbols:
    ///
    /// ```
    /// "12||C|^(2)_(5)|"
    /// "2||S|^(z)|(α)"
    /// ```
    fn new(symstr: &str, rep: T) -> Result<Self, GenericSymbolParsingError> {
        let generic_symbol = GenericSymbol::from_str(symstr)?;
        Ok(Self::builder()
            .generic_symbol(generic_symbol)
            .representative(rep)
            .build()
            .unwrap())
    }
}

impl<T: SpecialSymmetryTransformation + Clone> SpecialSymmetryTransformation for ClassSymbol<T> {
    /// Checks if this class is proper.
    ///
    /// # Returns
    ///
    /// A flag indicating if this class is proper.
    fn is_proper(&self) -> bool {
        self.representative.is_proper()
    }

    /// Checks if this class is antiunitary.
    ///
    /// # Returns
    ///
    /// A flag indicating if this class is antiunitary.
    fn is_antiunitary(&self) -> bool {
        self.representative.is_antiunitary()
    }

    /// Checks if this class is the identity class.
    ///
    /// # Returns
    ///
    /// A flag indicating if this class is the identity class.
    fn is_identity(&self) -> bool {
        self.representative.is_identity()
    }

    /// Checks if this class is the inversion class.
    ///
    /// # Returns
    ///
    /// A flag indicating if this class is the inversion class.
    fn is_inversion(&self) -> bool {
        self.representative.is_inversion()
    }

    /// Checks if this class is a binary rotation class.
    ///
    /// # Returns
    ///
    /// A flag indicating if this class is a binary rotation class.
    fn is_binary_rotation(&self) -> bool {
        self.representative.is_binary_rotation()
    }

    /// Checks if this class is a reflection class.
    ///
    /// # Returns
    ///
    /// A flag indicating if this class is a reflection class.
    fn is_reflection(&self) -> bool {
        self.representative.is_reflection()
    }

    /// Checks if this class is a pure time-reversal class.
    ///
    /// # Returns
    ///
    /// A flag indicating if this class is a pure time-reversal class.
    fn is_time_reversal(&self) -> bool {
        self.representative.is_time_reversal()
    }
}

// -------
// Display
// -------
impl<T: SpecialSymmetryTransformation + Clone> fmt::Display for ClassSymbol<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.generic_symbol)
    }
}
