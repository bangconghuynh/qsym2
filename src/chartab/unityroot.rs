use std::fmt;
use std::ops::Mul;
use std::hash::Hash;

use derive_builder::Builder;
use fraction::{self, GenericFraction, generic::GenericInteger, ToPrimitive};
use num::Complex;
use num_traits::{Zero, One, Pow};

#[cfg(test)]
#[path = "unityroot_tests.rs"]
mod unityroot_tests;

/// A struct to represent roots of unity symbolically.
///
/// Partial orders between roots of unity are based on their angular positions
/// on the unit circle in the Argand diagram, with unity being the smallest.
#[derive(Builder, Clone, PartialOrd, PartialEq, Eq, Hash)]
pub struct UnityRoot<I>
where
    I: Clone + GenericInteger + Hash,
{
    /// The fraction $`k/n \in [0, 1)`$ of the unity root, represented exactly
    /// for hashing and comparison purposes.
    #[builder(setter(custom))]
    pub fraction: GenericFraction<I>,
}

impl<I> UnityRootBuilder<I>
where
    I: Clone + GenericInteger + Hash + fmt::Display,
{
    fn fraction(&mut self, frac: GenericFraction<I>) -> &mut Self {
        self.fraction =
            if GenericFraction::<I>::zero() <= frac && frac < GenericFraction::<I>::one() {
                Some(frac)
            } else {
                let numer = frac
                    .numer()
                    .unwrap_or_else(|| panic!("The numerator of {frac} cannot be extracted."));
                let denom = frac
                    .denom()
                    .unwrap_or_else(|| panic!("The denominator of {frac} cannot be extracted."));
                Some(GenericFraction::<I>::new(numer.rem(*denom), *denom))
            };
        self
    }
}

impl<I> UnityRoot<I>
where
    I: Clone + GenericInteger + Hash + fmt::Display,
{
    /// Returns a builder to construct a new unity root.
    ///
    /// # Returns
    ///
    /// A builder to construct a new unity root.
    fn builder() -> UnityRootBuilder<I> {
        UnityRootBuilder::<I>::default()
    }

    /// Constructs a unity root from a non-negative index and order.
    ///
    /// # Returns
    ///
    /// A unity root.
    #[must_use]
    pub fn new(index: I, order: I) -> Self {
        Self::builder()
            .fraction(GenericFraction::<I>::new(index, order))
            .build()
            .expect("Unable to construct a unity root.")
    }

    /// The order $`n`$ of the root $`z`$, *i.e.* $`z^n = 1`$.
    ///
    /// # Returns
    ///
    /// The order $`n`$.
    fn order(&self) -> &I {
        self.fraction
            .denom()
            .expect("Unable to obtain the order of the root.")
    }

    /// The index $`k`$ of the root $`z`$, *i.e.* $`z = e^{\frac{2k\pi i}{n}}`$
    /// where $`k \in \mathbb{Z}/n\mathbb{Z}`$
    ///
    /// # Returns
    ///
    /// The index $`k`$.
    fn index(&self) -> &I {
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
            self.order().checked_sub(self.index()).unwrap_or_else(|| {
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

impl<'a, 'b, I> Mul<&'a UnityRoot<I>> for &'b UnityRoot<I>
where
    I: Clone + GenericInteger + Hash + fmt::Display,
{
    type Output = UnityRoot<I>;

    fn mul(self, rhs: &'a UnityRoot<I>) -> Self::Output {
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

// impl<I> Pow<i32> for &UnityRoot<I>
// where
//     I: Clone + GenericInteger + Hash + fmt::Display,
// {
//     type Output = UnityRoot<I>;

//     fn pow(self, rhs: i32) -> Self::Output {
//         let rhs_u = rhs.unsigned_abs();
//         let index = *self.index() * I::from(rhs_u);
//         Self::Output::new(
//             // u32::try_from(
//             //     (i32::try_from(*self.index())
//             //         .unwrap_or_else(|_| panic!("Unable to convert `{}` to `i32`.", self.index()))
//             //         * rhs)
//             //         .rem_euclid(i32::try_from(*self.order()).unwrap_or_else(|_| {
//             //             panic!("Unable to convert `{}` to `i32`.", self.order())
//             //         })),
//             // )
//             // .expect("Unexpected negative remainder."),
//             *self.order(),
//         )
//     }
// }

impl<I> fmt::Display for UnityRoot<I>
where
    I: Clone + GenericInteger + Hash + fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let zero = I::zero();
        let one = I::one();
        let two = one + one;
        let three = one + two;
        let four = two * two;
        if self.fraction == GenericFraction::<I>::new(zero, four) {
            write!(f, "1")
        } else if self.fraction == GenericFraction::<I>::new(one, four) {
            write!(f, "i")
        } else if self.fraction == GenericFraction::<I>::new(two, four) {
            write!(f, "-1")
        } else if self.fraction == GenericFraction::<I>::new(three, four) {
            write!(f, "-i")
        } else if *self.index() == one {
            write!(f, "E{}", self.order())
        } else {
            write!(f, "(E{})^{}", self.order(), self.index())
        }
    }
}

impl<I> fmt::Debug for UnityRoot<I>
where
    I: Clone + GenericInteger + Hash + fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let zero = I::zero();
        let one = I::one();
        let two = one + one;
        let four = two * two;
        if self.fraction == GenericFraction::<I>::new(zero, four) {
            write!(f, "1")
        } else if *self.index() == one {
            write!(f, "E{}", self.order())
        } else {
            write!(f, "(E{})^{}", self.order(), self.index())
        }
    }
}
