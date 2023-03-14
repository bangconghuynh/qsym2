use std::fmt;
use std::hash::{Hash, Hasher};
use std::str::FromStr;

use derive_builder::Builder;
use itertools::Itertools;
use ndarray::{Array2, ArrayView2, Axis};
use num_traits::ToPrimitive;

use crate::chartab::character::Character;
use crate::chartab::chartab_symbols::{
    disambiguate_linspace_symbols, CollectionSymbol, GenericSymbol, GenericSymbolParsingError,
    LinearSpaceSymbol, MathematicalSymbol,
};
use crate::chartab::unityroot::UnityRoot;
use crate::permutation::{Permutation, PermutationRank};

// ==================
// Struct definitions
// ==================

// ----------------------
// PermutationClassSymbol
// ----------------------

/// A struct to handle conjugacy class symbols.
#[derive(Builder, Debug, Clone)]
pub struct PermutationClassSymbol<T: PermutationRank> {
    /// The generic part of the symbol.
    generic_symbol: GenericSymbol,

    /// A representative element in the class.
    representatives: Option<Vec<Permutation<T>>>,
}

impl<T: PermutationRank> PartialEq for PermutationClassSymbol<T> {
    fn eq(&self, other: &Self) -> bool {
        self.generic_symbol == other.generic_symbol
    }
}

impl<T: PermutationRank> Eq for PermutationClassSymbol<T> {}

impl<T: PermutationRank> Hash for PermutationClassSymbol<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.generic_symbol.hash(state);
    }
}

impl<T: PermutationRank> PermutationClassSymbol<T> {
    fn builder() -> PermutationClassSymbolBuilder<T> {
        PermutationClassSymbolBuilder::default()
    }

    /// Creates a class symbol from a string and a representative element.
    ///
    /// Some possible conjugacy class symbols:
    ///
    /// ```text
    /// "1||(5)(2)(1)||"
    /// "1||(4)(1)|^(2)|"
    /// "12||(2)(2)(1)|^(5)|"
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
    pub fn new(
        symstr: &str,
        reps: Option<Vec<Permutation<T>>>,
    ) -> Result<Self, GenericSymbolParsingError> {
        let generic_symbol = GenericSymbol::from_str(symstr)?;
        if generic_symbol.multiplicity().is_none() {
            Err(GenericSymbolParsingError(format!(
                "{symstr} contains no class size prefactor."
            )))
        } else {
            Ok(Self::builder()
                .generic_symbol(generic_symbol)
                .representatives(reps)
                .build()
                .unwrap_or_else(|_| panic!("Unable to construct a class symbol from `{symstr}`.")))
        }
    }
}

impl<T: PermutationRank> MathematicalSymbol for PermutationClassSymbol<T> {
    /// The main part of the symbol, which denotes the cycle pattern of the class.
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

    /// The number of times the representative elements are 'duplicated' to give the size of the
    /// class.
    fn multiplicity(&self) -> Option<usize> {
        self.generic_symbol.multiplicity()
    }
}

impl<T: PermutationRank> CollectionSymbol for PermutationClassSymbol<T> {
    type CollectionElement = Permutation<T>;

    fn from_reps(
        symstr: &str,
        reps: Option<Vec<Self::CollectionElement>>,
    ) -> Result<Self, GenericSymbolParsingError> {
        Self::new(symstr, reps)
    }

    fn representative(&self) -> Option<&Self::CollectionElement> {
        self.representatives.as_ref().map(|reps| &reps[0])
    }

    fn representatives(&self) -> Option<&Vec<Self::CollectionElement>> {
        self.representatives.as_ref()
    }
}

impl<T: PermutationRank> fmt::Display for PermutationClassSymbol<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.generic_symbol)
    }
}

// ----------------------
// PermutationIrrepSymbol
// ----------------------

/// A struct to handle permutation irreducible representation symbols. This will be converted to a
/// suitable representation of Young tableaux symbols in the future.
#[derive(Builder, Debug, Clone, PartialEq, Eq, Hash, PartialOrd)]
pub struct PermutationIrrepSymbol {
    /// The generic part of the symbol.
    #[builder(setter(custom))]
    generic_symbol: GenericSymbol,

    /// The dimensionality of the irreducible representation.
    #[builder(setter(custom), default = "None")]
    dim: Option<usize>,
}

impl PermutationIrrepSymbolBuilder {
    fn generic_symbol(&mut self, sym: GenericSymbol) -> &mut Self {
        assert!(
            sym.main() == "Sym" || sym.main() == "Alt" || sym.main() == "Λ",
            "The main part of a permutation irrep symbol can only be `Sym`, `Alt`, or `Λ`.",
        );
        self.generic_symbol = Some(sym);
        self
    }

    fn dim(&mut self, dim: usize) -> &mut Self {
        let main = self
            .generic_symbol
            .as_ref()
            .expect("The generic symbol has not been set for this permutation symbol.")
            .main();
        if main == "Sym" || main == "Alt" {
            assert_eq!(
                dim, 1,
                "A `{main}` permutation irrep must be one-dimensional."
            );
        }
        self.dim = Some(Some(dim));
        self
    }
}

impl PermutationIrrepSymbol {
    fn builder() -> PermutationIrrepSymbolBuilder {
        PermutationIrrepSymbolBuilder::default()
    }

    /// Construct a permutation irrep symbol from a string and its dimensionality.
    ///
    /// Some permissible permutation irrep symbols:
    ///
    /// ```text
    /// "||Sym||"
    /// "||Alt||"
    /// "||Λ|_(1)|"
    /// ```
    ///
    /// # Arguments
    ///
    /// * `symstr` - A string to be parsed to give a permutation symbol.
    ///
    /// # Errors
    ///
    /// Errors when the string cannot be parsed as a generic symbol.
    pub fn new(symstr: &str, dim: usize) -> Result<Self, PermutationIrrepSymbolBuilderError> {
        let generic_symbol = GenericSymbol::from_str(symstr)
            .unwrap_or_else(|_| panic!("Unable to parse {symstr} as a generic symbol."));
        Self::builder()
            .generic_symbol(generic_symbol)
            .dim(dim)
            .build()
    }
}

impl MathematicalSymbol for PermutationIrrepSymbol {
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
        self.dim
    }
}

impl FromStr for PermutationIrrepSymbol {
    type Err = PermutationIrrepSymbolBuilderError;

    /// Parses a string representing a permutation irrep symbol.
    ///
    /// Some permissible permutation irrep symbols:
    ///
    /// ```text
    /// "||Sym||"
    /// "||Alt||"
    /// "||Λ|_(1)|"
    /// ```
    ///
    /// # Arguments
    ///
    /// * `symstr` - A string to be parsed to give a permutation symbol.
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

impl LinearSpaceSymbol for PermutationIrrepSymbol {
    fn dimensionality(&self) -> usize {
        self.dim
            .unwrap_or_else(|| panic!("Unknown dimensionality for permutation irrep `{self}`."))
    }

    fn set_dimensionality(&mut self, dim: usize) -> bool {
        if dim == 1 {
            if self.main() == "Sym" || self.main() == "Alt" {
                self.dim = Some(dim);
                true
            } else {
                false
            }
        } else {
            if self.main() != "Sym" && self.main() != "Alt" {
                self.dim = Some(dim);
                true
            } else {
                false
            }
        }
    }
}

impl fmt::Display for PermutationIrrepSymbol {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "|{}{}|{}",
            self.main(),
            if self.main() == "Λ" {
                self.dim
                    .map(|dim| dim.to_string())
                    .unwrap_or_else(|| "?".to_string())
            } else {
                String::new()
            },
            if self.postsub().is_empty() {
                String::new()
            } else {
                format!("_({})", self.postsub())
            }
        )
    }
}

// =======
// Methods
// =======

/// Sorts permutation irreps based on their dimensionalities.
///
/// # Arguments
///
/// * `char_arr` - A view of the array of characters for which the irreps are to be sorted.
/// * `frobenius_schur_indicators` - The associated Frobenius--Schur indicators with the irreps.
///
/// # Returns
///
/// An array of characters where the irreps have been sorted, and a vector of the associated
/// Frobenius--Schur indicators that have also been similarly sorted.
pub(super) fn sort_perm_irreps(
    char_arr: &ArrayView2<Character>,
    frobenius_schur_indicators: &[i8],
) -> (Array2<Character>, Vec<i8>) {
    log::debug!("Sorting permutation irreducible representations...");
    let n_rows = char_arr.nrows();
    let col_idxs = (0..n_rows).collect_vec();
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
    log::debug!("Sorting permutation irreducible representations... Done.");
    (char_arr, sorted_fs)
}

/// Deduces the permutation irrep symbols based on the characters.
///
/// This classifies each irrep into either `Sym` for the totally symmetric irrep, `Alt` for the
/// alternating one-dimensional irrep, or `Λ` for all higher-dimensional irreps. No attempt is made
/// to assign Young diagrams to the irreps.
///
/// # Arguments
///
/// * `char_arr` - An array of characters.
///
/// # Returns
///
/// A vector of permutation irrep symbols.
pub(super) fn deduce_permutation_irrep_symbols(
    char_arr: &ArrayView2<Character>,
) -> Vec<PermutationIrrepSymbol> {
    log::debug!("Generating permutation irreducible representation symbols...");

    // First pass: assign irrep symbols from rules as much as possible.
    log::debug!("First pass: assign symbols from rules");

    let one = Character::new(&[(UnityRoot::new(0u32, 1u32), 1)]);
    let raw_irrep_symbols = char_arr.rows().into_iter().map(|irrep| {
        let dim = irrep[0].clone();
        if dim == one {
            if irrep.iter().all(|chr| chr.clone() == one) {
                PermutationIrrepSymbol::new("||Sym||", 1).unwrap_or_else(|_| {
                    panic!("Unable to construct permutation irrep symbol `||Sym||`")
                })
            } else {
                PermutationIrrepSymbol::new("||Alt||", 1).unwrap_or_else(|_| {
                    panic!("Unable to construct permutation irrep symbol `||Alt||`")
                })
            }
        } else {
            let dim_c = dim.complex_value();
            assert!(
                approx::relative_eq!(dim_c.im, 0.0)
                    && approx::relative_eq!(dim_c.re.round(), dim_c.re)
                    && dim_c.re.round() > 0.0
            );
            let dim_u = dim_c.re
                .round()
                .to_usize()
                .expect("Unable to convert the dimensionality of an irrep to `usize`.");
            PermutationIrrepSymbol::new("||Λ||", dim_u).unwrap_or_else(|_| {
                    panic!("Unable to construct permutation irrep symbol `||Λ||` with dimensionality `{dim_u}`.")
                })
        }
    });

    // Second pass: disambiguate identical cases not distinguishable by rules
    log::debug!("Second pass: disambiguate identical cases not distinguishable by rules");
    let irrep_symbols = disambiguate_linspace_symbols(raw_irrep_symbols);

    log::debug!("Generating permutation irreducible representation symbols... Done.");
    irrep_symbols
}
