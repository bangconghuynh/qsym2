use std::cmp::Ordering;

use approx;
use derive_builder::Builder;
use indexmap::IndexMap;
use num::Complex;

use crate::chartab::unityroot::UnityRoot;

#[cfg(test)]
#[path = "character_tests.rs"]
mod character_tests;

/// A struct to represent algebraic group characters.
///
/// Partial orders between characters are based on their complex moduli and
/// phases in the interval `$[0, 2\pi)$` with `$0$` being the smallest.
#[derive(Builder, Debug)]
struct Character {
    /// The unity roots and tinto_heir multiplicities constituting this character.
    #[builder(setter(custom))]
    terms: IndexMap<UnityRoot, usize>,

    /// A threshold for approximate partial ordering comparisons.
    #[builder(setter(custom), default = "1e-14")]
    pub threshold: f64,
}

impl CharacterBuilder {
    fn terms(&mut self, ts: &[(UnityRoot, usize)]) -> &mut Self {
        self.terms = Some(ts.iter().cloned().collect());
        self
    }

    pub fn threshold(&mut self, thresh: f64) -> &mut Self {
        if thresh >= 0.0 {
            self.threshold = Some(thresh);
        } else {
            log::error!(
                "Threshold value {} is invalid. Threshold must be non-negative.",
                thresh
            );
            self.threshold = None;
        }
        self
    }
}

impl Character {
    /// Returns a builder to construct a new character.
    ///
    /// # Returns
    ///
    /// A builder to construct a new character.
    fn builder() -> CharacterBuilder {
        CharacterBuilder::default()
    }

    /// Constructs a character from an array of unity roots and multiplicities.
    ///
    /// # Returns
    ///
    /// A character.
    pub fn new(ts: &[(UnityRoot, usize)]) -> Self {
        Self::builder().terms(ts).build().unwrap()
    }

    /// The complex representation of this character.
    ///
    /// # Returns
    ///
    /// The complex value corresponding to this character.
    fn complex_value(&self) -> Complex<f64> {
        self.terms
            .iter()
            .fold(Complex::<f64>::new(0.0, 0.0), |acc, (uroot, &mult)| {
                acc + uroot.complex_value() * Complex::<f64>::new(mult as f64, 0.0)
            })
    }
}

impl PartialEq for Character {
    fn eq(&self, other: &Self) -> bool {
        self.terms == other.terms
    }
}

impl Eq for Character {}

impl PartialOrd for Character {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        let (self_norm, self_arg) = self.complex_value().to_polar();
        let (other_norm, other_arg) = other.complex_value().to_polar();
        let thresh = (self.threshold * other.threshold).sqrt();

        if approx::relative_eq!(
            self_norm,
            other_norm,
            epsilon = thresh,
            max_relative = thresh
        ) {
            let positive_self_arg = if self_arg >= -thresh {
                self_arg
            } else {
                2.0 * std::f64::consts::PI - self_arg
            };
            let positive_other_arg = if other_arg >= -thresh {
                other_arg
            } else {
                2.0 * std::f64::consts::PI - other_arg
            };
            positive_self_arg.partial_cmp(&positive_other_arg)
        } else {
            self_norm.partial_cmp(&other_norm)
        }
    }
}
