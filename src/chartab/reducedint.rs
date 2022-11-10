use std::hash::{Hash, Hasher};
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::fmt;

use num_modular::{ModularInteger, ReducedInt, Reducer, Montgomery};
use num_traits::{Inv, One, Pow, Zero};

use crate::aux::misc;

#[cfg(test)]
#[path = "reducedint_tests.rs"]
mod reducedint_tests;

/// A wrapper enum to represent an integer in a modulo ring, with added additive
/// and multiplicative identities to support linear algebra operations.
#[derive(Clone, Copy, Debug)]
pub enum LinAlgReducedInt<T, R: Reducer<T>> {
    /// Variant to represent an integer in a modulo ring with known
    /// characteristic.
    KnownChar(ReducedInt<T, R>),

    /// Variant to represent the additive identity, irrespective of ring
    /// characteristics.
    Zero,

    /// Variant to represent the multiplicative identity, irrespective of ring
    /// characteristics.
    One,
}

pub type LinAlgMontgomeryInt<T> = LinAlgReducedInt<T, Montgomery<T, T>>;

// ---
// Add
// ---
impl<T, R> Add<&'_ LinAlgReducedInt<T, R>> for &LinAlgReducedInt<T, R>
where
    T: Zero + One + PartialEq + Clone + Hash,
    R: Reducer<T> + Clone,
{
    type Output = LinAlgReducedInt<T, R>;

    fn add(self, rhs: &LinAlgReducedInt<T, R>) -> Self::Output {
        match (self, rhs) {
            (LinAlgReducedInt::Zero, _) => rhs.clone(),
            (_, LinAlgReducedInt::Zero) => self.clone(),
            (LinAlgReducedInt::One, LinAlgReducedInt::One) => {
                panic!("The ring characteristic is not known.")
            }
            (LinAlgReducedInt::One, LinAlgReducedInt::KnownChar(rint))
            | (LinAlgReducedInt::KnownChar(rint), LinAlgReducedInt::One) => {
                let rint_sum = rint + rint.convert(T::one());
                LinAlgReducedInt::KnownChar(rint_sum)
            }
            (LinAlgReducedInt::KnownChar(rint_l), LinAlgReducedInt::KnownChar(rint_r)) => {
                let rint_sum = rint_l + rint_r;
                LinAlgReducedInt::KnownChar(rint_sum)
            }
        }
    }
}

impl<T, R> Add<&'_ LinAlgReducedInt<T, R>> for LinAlgReducedInt<T, R>
where
    T: Zero + One + PartialEq + Clone + Hash,
    R: Reducer<T> + Clone,
{
    type Output = LinAlgReducedInt<T, R>;

    fn add(self, rhs: &Self) -> Self::Output {
        &self + rhs
    }
}

impl<T, R> Add<LinAlgReducedInt<T, R>> for LinAlgReducedInt<T, R>
where
    T: Zero + One + PartialEq + Clone + Hash,
    R: Reducer<T> + Clone,
{
    type Output = LinAlgReducedInt<T, R>;

    fn add(self, rhs: Self) -> Self::Output {
        &self + &rhs
    }
}

impl<T, R> Add<LinAlgReducedInt<T, R>> for &LinAlgReducedInt<T, R>
where
    T: Zero + One + PartialEq + Clone + Hash,
    R: Reducer<T> + Clone,
{
    type Output = LinAlgReducedInt<T, R>;

    fn add(self, rhs: LinAlgReducedInt<T, R>) -> Self::Output {
        self + &rhs
    }
}

impl<T, R> Add<T> for LinAlgReducedInt<T, R>
where
    T: Zero + One + PartialEq + Clone + Hash,
    R: Reducer<T> + Clone,
{
    type Output = LinAlgReducedInt<T, R>;

    fn add(self, rhs: T) -> Self::Output {
        &self + &self.convert(rhs)
    }
}

// ---
// Mul
// ---
impl<T, R> Mul<&'_ LinAlgReducedInt<T, R>> for &LinAlgReducedInt<T, R>
where
    T: Zero + One + PartialEq + Clone + Hash,
    R: Reducer<T> + Clone,
{
    type Output = LinAlgReducedInt<T, R>;

    fn mul(self, rhs: &LinAlgReducedInt<T, R>) -> Self::Output {
        match (self, rhs) {
            (LinAlgReducedInt::Zero, _) | (_, LinAlgReducedInt::Zero) => LinAlgReducedInt::Zero,
            (LinAlgReducedInt::One, _) => rhs.clone(),
            (_, LinAlgReducedInt::One) => self.clone(),
            (LinAlgReducedInt::KnownChar(rint_l), LinAlgReducedInt::KnownChar(rint_r)) => {
                let rint_prod = rint_l * rint_r;
                LinAlgReducedInt::KnownChar(rint_prod)
            }
        }
    }
}

impl<T, R> Mul<&'_ LinAlgReducedInt<T, R>> for LinAlgReducedInt<T, R>
where
    T: Zero + One + PartialEq + Clone + Hash,
    R: Reducer<T> + Clone,
{
    type Output = LinAlgReducedInt<T, R>;

    fn mul(self, rhs: &Self) -> Self::Output {
        &self * rhs
    }
}

impl<T, R> Mul<LinAlgReducedInt<T, R>> for LinAlgReducedInt<T, R>
where
    T: Zero + One + PartialEq + Clone + Hash,
    R: Reducer<T> + Clone,
{
    type Output = LinAlgReducedInt<T, R>;

    fn mul(self, rhs: Self) -> Self::Output {
        &self * &rhs
    }
}

impl<T, R> Mul<LinAlgReducedInt<T, R>> for &LinAlgReducedInt<T, R>
where
    T: Zero + One + PartialEq + Clone + Hash,
    R: Reducer<T> + Clone,
{
    type Output = LinAlgReducedInt<T, R>;

    fn mul(self, rhs: LinAlgReducedInt<T, R>) -> Self::Output {
        self * &rhs
    }
}

impl<T, R> Mul<T> for LinAlgReducedInt<T, R>
where
    T: Zero + One + PartialEq + Clone + Hash,
    R: Reducer<T> + Clone,
{
    type Output = LinAlgReducedInt<T, R>;

    fn mul(self, rhs: T) -> Self::Output {
        &self * &self.convert(rhs)
    }
}

// ---
// Sub
// ---
impl<T, R> Sub<&'_ LinAlgReducedInt<T, R>> for &LinAlgReducedInt<T, R>
where
    T: Zero + One + PartialEq + Clone + Hash,
    R: Reducer<T> + Clone,
{
    type Output = LinAlgReducedInt<T, R>;

    fn sub(self, rhs: &LinAlgReducedInt<T, R>) -> Self::Output {
        match (self, rhs) {
            (_, LinAlgReducedInt::Zero) => self.clone(),
            (LinAlgReducedInt::Zero, LinAlgReducedInt::One) => {
                panic!("The ring characteristic is not known.")
            }
            (LinAlgReducedInt::One, LinAlgReducedInt::One) => LinAlgReducedInt::Zero,
            (LinAlgReducedInt::KnownChar(rint), LinAlgReducedInt::One) => {
                let rint_diff = rint - rint.convert(T::one());
                LinAlgReducedInt::KnownChar(rint_diff)
            }
            (LinAlgReducedInt::Zero, LinAlgReducedInt::KnownChar(rint)) => {
                LinAlgReducedInt::KnownChar(-rint)
            }
            (LinAlgReducedInt::One, LinAlgReducedInt::KnownChar(rint)) => {
                let rint_diff = rint.convert(T::one()) - rint;
                LinAlgReducedInt::KnownChar(rint_diff)
            }
            (LinAlgReducedInt::KnownChar(rint_l), LinAlgReducedInt::KnownChar(rint_r)) => {
                let rint_diff = rint_l - rint_r;
                LinAlgReducedInt::KnownChar(rint_diff)
            }
        }
    }
}

impl<T, R> Sub<&'_ LinAlgReducedInt<T, R>> for LinAlgReducedInt<T, R>
where
    T: Zero + One + PartialEq + Clone + Hash,
    R: Reducer<T> + Clone,
{
    type Output = LinAlgReducedInt<T, R>;

    fn sub(self, rhs: &Self) -> Self::Output {
        &self - rhs
    }
}

impl<T, R> Sub<LinAlgReducedInt<T, R>> for LinAlgReducedInt<T, R>
where
    T: Zero + One + PartialEq + Clone + Hash,
    R: Reducer<T> + Clone,
{
    type Output = LinAlgReducedInt<T, R>;

    fn sub(self, rhs: Self) -> Self::Output {
        &self - &rhs
    }
}

impl<T, R> Sub<LinAlgReducedInt<T, R>> for &LinAlgReducedInt<T, R>
where
    T: Zero + One + PartialEq + Clone + Hash,
    R: Reducer<T> + Clone,
{
    type Output = LinAlgReducedInt<T, R>;

    fn sub(self, rhs: LinAlgReducedInt<T, R>) -> Self::Output {
        self - &rhs
    }
}

impl<T, R> Sub<T> for LinAlgReducedInt<T, R>
where
    T: Zero + One + PartialEq + Clone + Hash,
    R: Reducer<T> + Clone,
{
    type Output = LinAlgReducedInt<T, R>;

    fn sub(self, rhs: T) -> Self::Output {
        &self - &self.convert(rhs)
    }
}

// ---
// Div
// ---
impl<T: Zero + One + PartialEq + Clone + Hash, R: Reducer<T> + Clone> Div<&'_ LinAlgReducedInt<T, R>>
    for &LinAlgReducedInt<T, R>
{
    type Output = LinAlgReducedInt<T, R>;

    fn div(self, rhs: &LinAlgReducedInt<T, R>) -> Self::Output {
        match (self, rhs) {
            (_, LinAlgReducedInt::Zero) => panic!("Division by zero encountered."),
            (LinAlgReducedInt::Zero, _) => LinAlgReducedInt::Zero,
            (_, LinAlgReducedInt::One) => self.clone(),
            (LinAlgReducedInt::One, LinAlgReducedInt::KnownChar(rint)) => {
                let rint_div = rint.convert(T::one()) / rint;
                LinAlgReducedInt::KnownChar(rint_div)
            }
            (LinAlgReducedInt::KnownChar(rint_l), LinAlgReducedInt::KnownChar(rint_r)) => {
                let rint_div = rint_l / rint_r;
                LinAlgReducedInt::KnownChar(rint_div)
            }
        }
    }
}

impl<T, R> Div<&'_ LinAlgReducedInt<T, R>> for LinAlgReducedInt<T, R>
where
    T: Zero + One + PartialEq + Clone + Hash,
    R: Reducer<T> + Clone,
{
    type Output = LinAlgReducedInt<T, R>;

    fn div(self, rhs: &Self) -> Self::Output {
        &self / rhs
    }
}

impl<T, R> Div<LinAlgReducedInt<T, R>> for LinAlgReducedInt<T, R>
where
    T: Zero + One + PartialEq + Clone + Hash,
    R: Reducer<T> + Clone,
{
    type Output = LinAlgReducedInt<T, R>;

    fn div(self, rhs: Self) -> Self::Output {
        &self / &rhs
    }
}

impl<T, R> Div<LinAlgReducedInt<T, R>> for &LinAlgReducedInt<T, R>
where
    T: Zero + One + PartialEq + Clone + Hash,
    R: Reducer<T> + Clone,
{
    type Output = LinAlgReducedInt<T, R>;

    fn div(self, rhs: LinAlgReducedInt<T, R>) -> Self::Output {
        self / &rhs
    }
}

impl<T, R> Div<T> for LinAlgReducedInt<T, R>
where
    T: Zero + One + PartialEq + Clone + Hash,
    R: Reducer<T> + Clone,
{
    type Output = LinAlgReducedInt<T, R>;

    fn div(self, rhs: T) -> Self::Output {
        &self / &self.convert(rhs)
    }
}

// ---
// Inv
// ---
impl<T, R> Inv for &LinAlgReducedInt<T, R>
where
    T: Zero + One + PartialEq + Clone + Hash,
    R: Reducer<T> + Clone,
{
    type Output = LinAlgReducedInt<T, R>;

    fn inv(self) -> Self::Output {
        match self {
            LinAlgReducedInt::Zero => panic!("Inverse of zero encountered."),
            LinAlgReducedInt::One => LinAlgReducedInt::One,
            LinAlgReducedInt::KnownChar(rint) => LinAlgReducedInt::KnownChar(rint.inv()),
        }
    }
}

impl<T, R> Inv for LinAlgReducedInt<T, R>
where
    T: Zero + One + PartialEq + Clone + Hash,
    R: Reducer<T> + Clone,
{
    type Output = LinAlgReducedInt<T, R>;

    fn inv(self) -> Self::Output {
        (&self).inv()
    }
}

// ---
// Neg
// ---
impl<T, R> Neg for &LinAlgReducedInt<T, R>
where
    T: Zero + One + PartialEq + Clone + Hash,
    R: Reducer<T> + Clone,
{
    type Output = LinAlgReducedInt<T, R>;

    fn neg(self) -> Self::Output {
        match self {
            LinAlgReducedInt::Zero => LinAlgReducedInt::Zero,
            LinAlgReducedInt::One => panic!("The ring characteristic is not known."),
            LinAlgReducedInt::KnownChar(rint) => LinAlgReducedInt::KnownChar(-rint),
        }
    }
}

impl<T, R> Neg for LinAlgReducedInt<T, R>
where
    T: Zero + One + PartialEq + Clone + Hash,
    R: Reducer<T> + Clone,
{
    type Output = LinAlgReducedInt<T, R>;

    fn neg(self) -> Self::Output {
        -&self
    }
}

// ---
// Pow
// ---
impl<T, R> Pow<T> for &LinAlgReducedInt<T, R>
where
    T: Zero + One + PartialEq + Clone + Hash,
    R: Reducer<T> + Clone,
{
    type Output = LinAlgReducedInt<T, R>;

    fn pow(self, rhs: T) -> Self::Output {
        match self {
            LinAlgReducedInt::Zero => {
                if rhs == T::zero() {
                    LinAlgReducedInt::One
                } else {
                    LinAlgReducedInt::Zero
                }
            }
            LinAlgReducedInt::One => LinAlgReducedInt::One,
            LinAlgReducedInt::KnownChar(rint) => LinAlgReducedInt::KnownChar(rint.pow(rhs)),
        }
    }
}

impl<T, R> Pow<T> for LinAlgReducedInt<T, R>
where
    T: Zero + One + PartialEq + Clone + Hash,
    R: Reducer<T> + Clone,
{
    type Output = LinAlgReducedInt<T, R>;

    fn pow(self, rhs: T) -> Self::Output {
        (&self).pow(rhs)
    }
}

// ---------
// PartialEq
// ---------
impl<T, R> PartialEq<LinAlgReducedInt<T, R>> for LinAlgReducedInt<T, R>
where
    T: Zero + One + PartialEq + Clone + Hash,
    R: Reducer<T> + Clone,
{
    fn eq(&self, other: &Self) -> bool {
        let result = match (self, other) {
            (LinAlgReducedInt::Zero, LinAlgReducedInt::Zero) => true,
            (LinAlgReducedInt::One, LinAlgReducedInt::One) => true,
            (LinAlgReducedInt::Zero, LinAlgReducedInt::One)
            | (LinAlgReducedInt::One, LinAlgReducedInt::Zero) => false,
            (LinAlgReducedInt::Zero, LinAlgReducedInt::KnownChar(rint))
            | (LinAlgReducedInt::KnownChar(rint), LinAlgReducedInt::Zero) => rint.is_zero(),
            (LinAlgReducedInt::One, LinAlgReducedInt::KnownChar(rint))
            | (LinAlgReducedInt::KnownChar(rint), LinAlgReducedInt::One) => {
                rint.residue() == T::one()
            }
            (LinAlgReducedInt::KnownChar(rint_l), LinAlgReducedInt::KnownChar(rint_r)) => {
                rint_l == rint_r
            }
        };
        if result {
            assert_eq!(misc::calculate_hash(self), misc::calculate_hash(other));
        }
        result
    }
}

// --
// Eq
// --
impl<T, R> Eq for LinAlgReducedInt<T, R>
where
    T: Zero + One + PartialEq + Eq + Clone + Hash,
    R: Reducer<T> + Clone,
{ }

// --------------
// ReducedInteger
// --------------
impl<T, R> ModularInteger for LinAlgReducedInt<T, R>
where
    T: Zero + One + PartialEq + Clone + Hash,
    R: Reducer<T> + Clone,
{
    type Base = T;

    fn modulus(&self) -> T {
        match self {
            Self::Zero | Self::One => {
                panic!("The ring modulus is not known.")
            }
            Self::KnownChar(rint) => rint.modulus(),
        }
    }

    fn residue(&self) -> T {
        match self {
            Self::Zero => T::zero(),
            Self::One => T::one(),
            Self::KnownChar(rint) => rint.residue(),
        }
    }

    fn is_zero(&self) -> bool {
        Zero::is_zero(self)
    }

    fn convert(&self, n: T) -> Self {
        match self {
            Self::Zero | Self::One => {
                panic!("The ring modulus is not known.")
            }
            Self::KnownChar(rint) => Self::KnownChar(rint.convert(n)),
        }
    }

    fn double(self) -> Self {
        match self {
            Self::Zero => Self::Zero,
            Self::One => {
                panic!("The ring modulus is not known.")
            }
            Self::KnownChar(rint) => Self::KnownChar(rint.double()),
        }
    }

    fn square(self) -> Self {
        match self {
            Self::Zero => Self::Zero,
            Self::One => {
                panic!("The ring modulus is not known.")
            }
            Self::KnownChar(rint) => Self::KnownChar(rint.square()),
        }
    }
}

// ----
// Zero
// ----
impl<T, R> Zero for LinAlgReducedInt<T, R>
where
    T: Zero + One + PartialEq + Clone + Hash,
    R: Reducer<T> + Clone,
{
    fn zero() -> Self {
        Self::Zero
    }

    fn is_zero(&self) -> bool {
        match self {
            Self::KnownChar(res) => res.residue() == T::zero(),
            Self::Zero => true,
            Self::One => false,
        }
    }
}

// ---
// One
// ---
impl<T, R> One for LinAlgReducedInt<T, R>
where
    T: Zero + One + PartialEq + Clone + Hash,
    R: Reducer<T> + Clone,
{
    fn one() -> Self {
        Self::One
    }

    fn is_one(&self) -> bool {
        match self {
            Self::KnownChar(res) => res.residue() == T::one(),
            Self::Zero => false,
            Self::One => true,
        }
    }
}

// ----
// Hash
// ----
impl<T, R> Hash for LinAlgReducedInt<T, R>
where
    T: Zero + One + PartialEq + Clone + Hash,
    R: Reducer<T> + Clone,
{
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self {
            LinAlgReducedInt::Zero => T::zero().hash(state),
            LinAlgReducedInt::One => T::one().hash(state),
            LinAlgReducedInt::KnownChar(rint) => {
                if Zero::is_zero(self) {
                    T::zero().hash(state)
                } else if self.is_one() {
                    T::one().hash(state)
                } else {
                    rint.residue().hash(state);
                    rint.modulus().hash(state);
                }
            }
        }
    }
}

// -------
// Display
// -------
impl<T, R> fmt::Display for LinAlgReducedInt<T, R>
where
    T: Zero + One + PartialEq + Clone + Hash + fmt::Display,
    R: Reducer<T> + Clone,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Zero => write!(f, "0"),
            Self::One => write!(f, "1"),
            _ => write!(f, "{} (mod {})", self.residue(), self.modulus())
        }
    }
}

// ----
// Wrap
// ----
pub trait IntoLinAlgReducedInt {
    type InnerT;
    type InnerR: Reducer<Self::InnerT>;
    fn linalg(self) -> LinAlgReducedInt<Self::InnerT, Self::InnerR>;
}

impl<T, R: Reducer<T>> IntoLinAlgReducedInt for ReducedInt<T, R> {
    type InnerT = T;
    type InnerR = R;
    fn linalg(self) -> LinAlgReducedInt<Self::InnerT, Self::InnerR> {
        LinAlgReducedInt::KnownChar(self)
    }
}
