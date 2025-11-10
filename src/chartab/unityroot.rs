//! Symbolic representation of roots of unity for characters.

use std::fmt;
use std::ops::Mul;

use derive_builder::Builder;
use fraction::{self, ToPrimitive};
use num::Complex;
use num_traits::Pow;
use serde::{Deserialize, Serialize};

type F = fraction::GenericFraction<u32>;

#[cfg(test)]
#[path = "unityroot_tests.rs"]
mod unityroot_tests;

/// Structure to represent roots of unity symbolically.
///
/// Partial orders between roots of unity are based on their angular positions
/// on the unit circle in the Argand diagram, with unity being the smallest.
#[derive(Builder, Clone, PartialOrd, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct UnityRoot {
    /// The fraction $`k/n \in [0, 1)`$ of the unity root, represented exactly
    /// for hashing and comparison purposes.
    #[builder(setter(custom))]
    pub(crate) fraction: F,
}

impl UnityRootBuilder {
    fn fraction(&mut self, frac: F) -> &mut Self {
        self.fraction = if F::from(0) <= frac && frac < F::from(1) {
            Some(frac)
        } else {
            let numer = frac
                .numer()
                .unwrap_or_else(|| panic!("The numerator of {frac} cannot be extracted."));
            let denom = frac
                .denom()
                .unwrap_or_else(|| panic!("The denominator of {frac} cannot be extracted."));
            Some(F::new(numer.rem_euclid(*denom), *denom))
        };
        self
    }
}

impl UnityRoot {
    /// Returns a builder to construct a new unity root.
    ///
    /// # Returns
    ///
    /// A builder to construct a new unity root.
    fn builder() -> UnityRootBuilder {
        UnityRootBuilder::default()
    }

    /// Constructs a unity root $`z`$ from a non-negative index $`k`$ and order $`n`$, where
    ///
    /// ```math
    ///   z = e^{\frac{2k\pi i}{n}}.
    /// ```
    ///
    /// # Returns
    ///
    /// The required unity root.
    #[must_use]
    pub fn new(index: u32, order: u32) -> Self {
        Self::builder()
            .fraction(F::new(index, order))
            .build()
            .expect("Unable to construct a unity root.")
    }

    /// The order $`n`$ of the root $`z`$, *i.e.* $`z^n = 1`$.
    ///
    /// # Returns
    ///
    /// The order $`n`$.
    fn order(&self) -> &u32 {
        self.fraction
            .denom()
            .expect("Unable to obtain the order of the root.")
    }

    /// The index $`k`$ of the root $`z`$, *i.e.* $`z = e^{\frac{2k\pi i}{n}}`$
    /// where $`k \in \mathbb{Z}/n\mathbb{Z}`$.
    ///
    /// # Returns
    ///
    /// The index $`k`$.
    fn index(&self) -> &u32 {
        self.fraction
            .numer()
            .expect("Unable to obtain the index of the root.")
    }

    /// The complex representation of this root.
    ///
    /// # Returns
    ///
    /// The complex value corresponding to this root.
    #[must_use]
    pub fn complex_value(&self) -> Complex<f64> {
        let theta = self
            .fraction
            .to_f64()
            .expect("Unable to convert a fraction to `f64`.")
            * std::f64::consts::PI
            * 2.0;
        Complex::<f64>::from_polar(1.0, theta)
    }

    /// The complex conjugate of this root.
    ///
    /// # Returns
    ///
    /// The complex conjugate of this root.
    ///
    /// # Panics
    ///
    /// Panics when the complex conjugate cannot be found.
    #[must_use]
    pub fn complex_conjugate(&self) -> Self {
        Self::new(
            self.order().checked_sub(*self.index()).unwrap_or_else(|| {
                panic!(
                    "Unable to perform the subtraction `{} - {}` correctly.",
                    self.order(),
                    self.index()
                )
            }),
            *self.order(),
        )
    }
}

impl<'a> Mul<&'a UnityRoot> for &UnityRoot {
    type Output = UnityRoot;

    fn mul(self, rhs: &'a UnityRoot) -> Self::Output {
        #[allow(clippy::suspicious_arithmetic_impl)]
        let fract_sum = self.fraction + rhs.fraction;
        Self::Output::builder()
            .fraction(fract_sum)
            .build()
            .unwrap_or_else(|_| {
                panic!("Unable to construct a unity root with fraction {fract_sum}.")
            })
    }
}

impl Pow<i32> for &UnityRoot {
    type Output = UnityRoot;

    fn pow(self, rhs: i32) -> Self::Output {
        Self::Output::new(
            u32::try_from(
                (i32::try_from(*self.index())
                    .unwrap_or_else(|_| panic!("Unable to convert `{}` to `i32`.", self.index()))
                    * rhs)
                    .rem_euclid(i32::try_from(*self.order()).unwrap_or_else(|_| {
                        panic!("Unable to convert `{}` to `i32`.", self.order())
                    })),
            )
            .expect("Unexpected negative remainder."),
            *self.order(),
        )
    }
}

impl fmt::Display for UnityRoot {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.fraction == F::new(0u32, 4u32) {
            write!(f, "1")
        } else if self.fraction == F::new(1u32, 4u32) {
            write!(f, "i")
        } else if self.fraction == F::new(2u32, 4u32) {
            write!(f, "-1")
        } else if self.fraction == F::new(3u32, 4u32) {
            write!(f, "-i")
        } else if *self.index() == 1u32 {
            write!(f, "E{}", self.order())
        } else {
            write!(f, "(E{})^{}", self.order(), self.index())
        }
    }
}

impl fmt::Debug for UnityRoot {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.fraction == F::new(0u32, 4u32) {
            write!(f, "1")
        } else if *self.index() == 1u32 {
            write!(f, "E{}", self.order())
        } else {
            write!(f, "(E{})^{}", self.order(), self.index())
        }
    }
}
