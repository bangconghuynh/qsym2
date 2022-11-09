use std::fmt;
use std::ops::Mul;

use derive_builder::Builder;
use fraction::{self, ToPrimitive};
use num::Complex;
use num_traits::Pow;

type F = fraction::Fraction;

#[cfg(test)]
#[path = "unityroot_tests.rs"]
mod unityroot_tests;

/// A struct to represent roots of unity symbolically.
///
/// Partial orders between roots of unity are based on their angular positions
/// on the unit circle in the Argand diagram, with unity being the smallest.
#[derive(Builder, Clone, PartialOrd, PartialEq, Eq, Hash)]
pub struct UnityRoot {
    /// The fraction $`k/n \in [0, 1)`$ of the unity root, represented exactly
    /// for hashing and comparison purposes.
    #[builder(setter(custom))]
    fraction: F,
}

impl UnityRootBuilder {
    fn fraction(&mut self, frac: F) -> &mut Self {
        self.fraction = if F::from(0) <= frac && frac < F::from(1) {
            Some(frac)
        } else {
            let numer = frac.numer().unwrap();
            let denom = frac.denom().unwrap();
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

    /// Constructs a unity root from a non-negative index and order.
    ///
    /// # Returns
    ///
    /// A unity root.
    pub fn new(index: u64, order: u64) -> Self {
        Self::builder()
            .fraction(F::new(index, order))
            .build()
            .unwrap()
    }

    /// The order $`n`$ of the root $`z`$, *i.e.* $`z^n = 1`$.
    ///
    /// # Returns
    ///
    /// The order $`n`$.
    fn order(&self) -> &u64 {
        self.fraction.denom().unwrap()
    }

    /// The index $`k`$ of the root $`z`$, *i.e.* $`z = e^{\frac{2k\pi i}{n}}`$
    /// where $`k \in \mathbb{Z}/n\mathbb{Z}`$
    ///
    /// # Returns
    ///
    /// The index $`k`$.
    fn index(&self) -> &u64 {
        self.fraction.numer().unwrap()
    }

    /// The complex representation of this root.
    ///
    /// # Returns
    ///
    /// The complex value corresponding to this root.
    pub fn complex_value(&self) -> Complex<f64> {
        let theta = self.fraction.to_f64().unwrap() * std::f64::consts::PI * 2.0;
        Complex::<f64>::from_polar(1.0, theta)
    }

    /// The complex conjugate of this root.
    ///
    /// # Returns
    ///
    /// The complex conjugate of this root.
    pub fn complex_conjugate(&self) -> Self {
        Self::new(
            self.order().checked_sub(*self.index()).unwrap(),
            *self.order(),
        )
    }
}

impl<'a, 'b> Mul<&'a UnityRoot> for &'b UnityRoot {
    type Output = UnityRoot;

    fn mul(self, rhs: &'a UnityRoot) -> Self::Output {
        #[allow(clippy::suspicious_arithmetic_impl)]
        let fract_sum = self.fraction + rhs.fraction;
        Self::Output::builder().fraction(fract_sum).build().unwrap()
    }
}

impl Pow<i32> for &UnityRoot {
    type Output = UnityRoot;

    fn pow(self, rhs: i32) -> Self::Output {
        Self::Output::new(
            u64::try_from((*self.index() as i32 * rhs).rem_euclid(*self.order() as i32))
                .expect("Unexpected negative remainder."),
            *self.order(),
        )
    }
}

impl fmt::Display for UnityRoot {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.fraction == F::new(0u64, 4u64) {
            write!(f, "1")
        } else if self.fraction == F::new(1u64, 4u64) {
            write!(f, "i")
        } else if self.fraction == F::new(2u64, 4u64) {
            write!(f, "-1")
        } else if self.fraction == F::new(3u64, 4u64) {
            write!(f, "-i")
        } else if *self.index() == 1u64 {
            write!(f, "E{}", self.order())
        } else {
            write!(f, "(E{})^{}", self.order(), self.index())
        }
    }
}

impl fmt::Debug for UnityRoot {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.fraction == F::new(0u64, 4u64) {
            write!(f, "1")
        } else if *self.index() == 1u64 {
            write!(f, "E{}", self.order())
        } else {
            write!(f, "(E{})^{}", self.order(), self.index())
        }
    }
}
