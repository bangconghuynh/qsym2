use std::cmp::Ordering;
use std::fmt;

use approx;
use derive_builder::Builder;
use indexmap::IndexMap;
use num::Complex;

use crate::aux::misc::HashableFloat;
use crate::chartab::unityroot::UnityRoot;

#[cfg(test)]
#[path = "character_tests.rs"]
mod character_tests;

/// A struct to represent algebraic group characters.
///
/// Partial orders between characters are based on their complex moduli and
/// phases in the interval `$[0, 2\pi)$` with `$0$` being the smallest.
#[derive(Builder)]
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
        (self.terms == other.terms) || {
            let self_complex = self.complex_value();
            let other_complex = other.complex_value();
            let thresh = (self.threshold * other.threshold).sqrt();
            approx::relative_eq!(
                self_complex.re,
                other_complex.re,
                epsilon = thresh,
                max_relative = thresh
            ) && approx::relative_eq!(
                self_complex.im,
                other_complex.im,
                epsilon = thresh,
                max_relative = thresh
            )
        }
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

impl fmt::Debug for Character {
    /// Prints the full form for this character showing all contributing unity
    /// roots and their multiplicities.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let one = UnityRoot::new(0u64, 2u64);
        let str_terms: Vec<String> = self
            .terms
            .clone()
            .sorted_by(|k1, _, k2, _| k1.partial_cmp(k2).unwrap())
            .into_iter()
            .filter_map(|(root, mult)| {
                if mult == 1 {
                    Some(format!("{}", root))
                } else if mult == 0 {
                    None
                } else if root == one {
                    Some(format!("{}", mult))
                } else {
                    Some(format!("{}*{}", mult, root))
                }
            })
            .collect();
        if str_terms.len() > 0 {
            write!(f, "{}", str_terms.join(" + "))
        } else {
            write!(f, "0")
        }
    }
}

impl fmt::Display for Character {
    /// Prints the short form for this character that shows either an integer
    /// or an imaginary integer or a compact complex number at 3 d.p.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let complex_value = self.complex_value();
        if approx::relative_eq!(
            complex_value.im,
            0.0,
            epsilon = self.threshold,
            max_relative = self.threshold
        ) {
            // Zero imaginary
            // Zero or non-zero real
            let rounded_re = complex_value.re.round_factor(self.threshold);
            if approx::relative_eq!(
                rounded_re,
                rounded_re.round(),
                epsilon = self.threshold,
                max_relative = self.threshold
            ) {
                // Integer real
                if approx::relative_eq!(
                    rounded_re,
                    0.0,
                    epsilon = self.threshold,
                    max_relative = self.threshold
                ) {
                    write!(f, "0")
                } else {
                    write!(f, "{:+.0}", complex_value.re)
                }
            } else {
                // Non-integer real
                write!(f, "{:+.3}", complex_value.re)
            }
        } else if approx::relative_eq!(
            complex_value.re,
            0.0,
            epsilon = self.threshold,
            max_relative = self.threshold
        ) {
            // Non-zero imaginary
            // Zero real
            let rounded_im = complex_value.im.round_factor(self.threshold);
            if approx::relative_eq!(
                rounded_im,
                rounded_im.round(),
                epsilon = self.threshold,
                max_relative = self.threshold
            ) {
                // Integer imaginary
                if approx::relative_eq!(
                    rounded_im.abs(),
                    1.0,
                    epsilon = self.threshold,
                    max_relative = self.threshold
                ) {
                    // i or -i
                    let imag = if rounded_im > 0.0 { "+i" } else { "-i" };
                    write!(f, "{}", imag)
                } else {
                    // ki
                    write!(f, "{:+.0}i", complex_value.im)
                }
            } else {
                // Non-integer imaginary
                write!(f, "{:+.3}i", complex_value.im)
            }
        } else {
            // Non-zero imaginary
            // Non-zero real
            write!(
                f,
                "{:+.3} {} {:.3}i",
                complex_value.re,
                {
                    if complex_value.im > 0.0 {
                        "+"
                    } else {
                        "-"
                    }
                },
                complex_value.im.abs()
            )
        }
    }
}
