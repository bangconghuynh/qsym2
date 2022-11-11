use std::fmt;
use std::hash::Hash;
use std::str::FromStr;

use derive_builder::Builder;
use phf::phf_map;
use regex::Regex;

// ======
// Traits
// ======

/// A trait for general mathematical symbols.
trait MathematicalSymbol: Hash + FromStr {
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
#[derive(Builder, Clone, PartialEq, Eq, Hash)]
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
                cap.get(0).unwrap().as_str()
            } else {
                ""
            };

            let presub_re = Regex::new(r"_\((.*?)\)").unwrap();
            let presubstr = if let Some(cap) = presub_re.captures(prestr) {
                cap.get(0).unwrap().as_str()
            } else {
                ""
            };

            let postsuper_re = Regex::new(r"\^\((.*?)\)").unwrap();
            let postsuperstr = if let Some(cap) = postsuper_re.captures(poststr) {
                cap.get(0).unwrap().as_str()
            } else {
                ""
            };

            let postsub_re = Regex::new(r"_\((.*?)\)").unwrap();
            let postsubstr = if let Some(cap) = postsub_re.captures(poststr) {
                cap.get(0).unwrap().as_str()
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
#[derive(Builder, PartialEq, Eq, Hash)]
struct MullikenIrrepSymbol {
    /// The generic part of the symbol.
    generic_symbol: GenericSymbol,
}

impl MullikenIrrepSymbol {
    fn builder() -> MullikenIrrepSymbolBuilder {
        MullikenIrrepSymbolBuilder::default()
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
        Ok(Self::builder().generic_symbol(generic_symbol).build().unwrap())
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
        *MULLIKEN_IRREP_DEGENERACIES.get(self.main()).unwrap_or_else(||
            panic!(
                "Unknown dimensionality for Mulliken symbol {}.",
                self.main()
            )
        )
    }
}

// -----------
// ClassSymbol
// -----------

/// A struct to handle conjugacy class symbols.
#[derive(Builder, PartialEq, Eq, Hash)]
struct ClassSymbol {
    /// The generic part of the symbol.
    generic_symbol: GenericSymbol,
}

impl ClassSymbol {
    fn builder() -> ClassSymbolBuilder {
        ClassSymbolBuilder::default()
    }
}

impl MathematicalSymbol for ClassSymbol {
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

impl CollectionSymbol for ClassSymbol {
    fn size(&self) -> u64 {
        self.prefactor().parse::<u64>().unwrap_or_else(|_|
            panic!(
                "Unable to deduce the size of the class from the prefactor {}.",
                self.prefactor()
            )
        )
    }
}

impl FromStr for ClassSymbol {
    type Err = GenericSymbolParsingError;

    /// Parses a string representing a Mulliken irrep symbol.
    ///
    /// Some permissible Mulliken irrep symbols:
    ///
    /// ```
    /// "12||C|^(2)_(5)|"
    /// "2||S|^(z)|(α)"
    /// ```
    fn from_str(symstr: &str) -> Result<Self, Self::Err> {
        let generic_symbol = GenericSymbol::from_str(symstr)?;
        Ok(Self::builder().generic_symbol(generic_symbol).build().unwrap())
    }
}

//// -------
//// Display
//// -------
//// impl fmt::Display for SymmetrySymbol {
////     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
////         match self {
////             Self::Irrep(0) => write!(f, "0"),
////         }
////     }
//// }
