use std::ops::{Add, Div, Mul, Neg, Sub};

use log;
use num_modular::{ModularInteger, ReducedInt, Reducer};
use num_traits::{Inv, One, Pow, Zero};

/// A wrapper struct to represent an integer in a modulo ring.
#[derive(Clone, Copy, Debug)]
pub struct LinAlgReducedInt<T, R: Reducer<T>>(ReducedInt<T, R>);

// ---
// Add
// ---
impl<T: Zero + One + PartialEq + Clone, R: Reducer<T> + Clone> Add<&'_ LinAlgReducedInt<T, R>>
    for &LinAlgReducedInt<T, R>
{
    type Output = LinAlgReducedInt<T, R>;

    fn add(self, rhs: &LinAlgReducedInt<T, R>) -> Self::Output {
        match (Zero::is_zero(self), Zero::is_zero(rhs)) {
            (true, true) => LinAlgReducedInt::zero(),
            (false, true) => LinAlgReducedInt(self.0.clone()),
            (true, false) => LinAlgReducedInt(rhs.0.clone()),
            (false, false) => match (self.is_one(), rhs.is_one()) {
                (true, true) => panic!("The ring modulus is not known."),
                (false, true) => LinAlgReducedInt(&self.0 + &self.convert(T::one()).0),
                (true, false) => LinAlgReducedInt(&rhs.convert(T::one()).0 + &rhs.0),
                (false, false) => LinAlgReducedInt(&self.0 + &rhs.0),
            }
        }
    }
}

impl<T: Zero + One + PartialEq + Clone, R: Reducer<T> + Clone> Add<&'_ LinAlgReducedInt<T, R>>
    for LinAlgReducedInt<T, R>
{
    type Output = LinAlgReducedInt<T, R>;

    fn add(self, rhs: &Self) -> Self::Output {
        &self + rhs
    }
}

impl<T: Zero + One + PartialEq + Clone, R: Reducer<T> + Clone> Add<LinAlgReducedInt<T, R>> for LinAlgReducedInt<T, R> {
    type Output = LinAlgReducedInt<T, R>;

    fn add(self, rhs: Self) -> Self::Output {
        &self + &rhs
    }
}

impl<T: Zero + One + PartialEq + Clone, R: Reducer<T> + Clone> Add<LinAlgReducedInt<T, R>> for &LinAlgReducedInt<T, R> {
    type Output = LinAlgReducedInt<T, R>;

    fn add(self, rhs: LinAlgReducedInt<T, R>) -> Self::Output {
        self + &rhs
    }
}

impl<T: Zero + One + PartialEq + Clone, R: Reducer<T> + Clone> Add<T> for LinAlgReducedInt<T, R> {
    type Output = LinAlgReducedInt<T, R>;

    fn add(self, rhs: T) -> Self::Output {
        &self + &self.convert(rhs)
    }
}

// ---
// Div
// ---
impl<T: Zero + One + PartialEq + Clone, R: Reducer<T> + Clone> Div<&'_ LinAlgReducedInt<T, R>>
    for &LinAlgReducedInt<T, R>
{
    type Output = LinAlgReducedInt<T, R>;

    fn div(self, rhs: &LinAlgReducedInt<T, R>) -> Self::Output {
        match (Zero::is_zero(self), Zero::is_zero(rhs)) {
            (_, true) => panic!("Zero denominator encountered in division."),
            (true, false) => LinAlgReducedInt::zero(),
            (false, false) => LinAlgReducedInt(&self.0 / &rhs.0),
        }
    }
}

impl<T: Zero + One + PartialEq + Clone, R: Reducer<T> + Clone> Div<&'_ LinAlgReducedInt<T, R>>
    for LinAlgReducedInt<T, R>
{
    type Output = LinAlgReducedInt<T, R>;

    fn div(self, rhs: &Self) -> Self::Output {
        &self / rhs
    }
}

impl<T: Zero + One + PartialEq + Clone, R: Reducer<T> + Clone> Div<LinAlgReducedInt<T, R>> for LinAlgReducedInt<T, R> {
    type Output = LinAlgReducedInt<T, R>;

    fn div(self, rhs: Self) -> Self::Output {
        &self / &rhs
    }
}

impl<T: Zero + One + PartialEq + Clone, R: Reducer<T> + Clone> Div<LinAlgReducedInt<T, R>>
    for &LinAlgReducedInt<T, R>
{
    type Output = LinAlgReducedInt<T, R>;

    fn div(self, rhs: LinAlgReducedInt<T, R>) -> Self::Output {
        self / &rhs
    }
}

// ---
// Inv
// ---
impl<T: Zero + One + PartialEq + Clone, R: Reducer<T> + Clone> Inv for &LinAlgReducedInt<T, R> {
    type Output = LinAlgReducedInt<T, R>;

    fn inv(self) -> Self::Output {
        if Zero::is_zero(self) {
            panic!("Inverse of zero encountered.")
        } else if self.is_one() {
            self.clone()
        } else {
            LinAlgReducedInt(self.0.clone().inv())
        }
    }
}

impl<T: Zero + One + PartialEq + Clone, R: Reducer<T> + Clone> Inv for LinAlgReducedInt<T, R> {
    type Output = LinAlgReducedInt<T, R>;

    fn inv(self) -> Self::Output {
        (&self).inv()
    }
}

// ---
// Mul
// ---
impl<T: Zero + One + PartialEq + Clone, R: Reducer<T> + Clone> Mul<&'_ LinAlgReducedInt<T, R>>
    for &LinAlgReducedInt<T, R>
{
    type Output = LinAlgReducedInt<T, R>;

    fn mul(self, rhs: &LinAlgReducedInt<T, R>) -> Self::Output {
        match (Zero::is_zero(self), Zero::is_zero(rhs)) {
            (true, _) | (_, true) => LinAlgReducedInt::zero(),
            (false, false) => match (self.is_one(), rhs.is_one()) {
                (true, true) => LinAlgReducedInt::one(),
                (false, true) => rhs.clone(),
                (true, false) => self.clone(),
                (false, false) => LinAlgReducedInt(&self.0 * &rhs.0),
            }
        }
    }
}

impl<T: Zero + One + PartialEq + Clone, R: Reducer<T> + Clone> Mul<&'_ LinAlgReducedInt<T, R>>
    for LinAlgReducedInt<T, R>
{
    type Output = LinAlgReducedInt<T, R>;

    fn mul(self, rhs: &Self) -> Self::Output {
        &self * rhs
    }
}

impl<T: Zero + One + PartialEq + Clone, R: Reducer<T> + Clone> Mul<LinAlgReducedInt<T, R>> for LinAlgReducedInt<T, R> {
    type Output = LinAlgReducedInt<T, R>;

    fn mul(self, rhs: Self) -> Self::Output {
        &self * &rhs
    }
}

impl<T: Zero + One + PartialEq + Clone, R: Reducer<T> + Clone> Mul<LinAlgReducedInt<T, R>> for &LinAlgReducedInt<T, R> {
    type Output = LinAlgReducedInt<T, R>;

    fn mul(self, rhs: LinAlgReducedInt<T, R>) -> Self::Output {
        self * &rhs
    }
}

impl<T: Zero + One + PartialEq + Clone, R: Reducer<T> + Clone> Mul<T> for LinAlgReducedInt<T, R> {
    type Output = LinAlgReducedInt<T, R>;

    fn mul(self, rhs: T) -> Self::Output {
        &self * &self.convert(rhs)
    }
}

// ---
// Neg
// ---
impl<T: Zero + One + PartialEq + Clone, R: Reducer<T> + Clone> Neg for &LinAlgReducedInt<T, R> {
    type Output = LinAlgReducedInt<T, R>;

    fn neg(self) -> Self::Output {
        if self.is_one() {
            panic!("The ring modulus is not known.")
        } else {
            LinAlgReducedInt(-&self.0)
        }
    }
}

impl<T: Zero + One + PartialEq + Clone, R: Reducer<T> + Clone> Neg for LinAlgReducedInt<T, R> {
    type Output = LinAlgReducedInt<T, R>;

    fn neg(self) -> Self::Output {
        -&self.clone()
    }
}

// ---
// Pow
// ---
impl<T: Zero + One + PartialEq + Clone, R: Reducer<T> + Clone> Pow<T> for &LinAlgReducedInt<T, R> {
    type Output = LinAlgReducedInt<T, R>;

    fn pow(self, rhs: T) -> Self::Output {
        if Zero::is_zero(self) || self.is_one() {
            self.clone()
        } else {
            LinAlgReducedInt(self.0.clone().pow(rhs))
        }
    }
}

impl<T: Zero + One + PartialEq + Clone, R: Reducer<T> + Clone> Pow<T> for LinAlgReducedInt<T, R> {
    type Output = LinAlgReducedInt<T, R>;

    fn pow(self, rhs: T) -> Self::Output {
        (&self).pow(rhs)
    }
}

// ---
// Sub
// ---
impl<T: PartialEq + Clone, R: Reducer<T>> Sub<&'_ LinAlgReducedInt<T, R>>
    for LinAlgReducedInt<T, R>
{
    type Output = LinAlgReducedInt<T, R>;

    fn sub(self, rhs: &Self) -> Self::Output {
        LinAlgReducedInt(self.0 - &rhs.0)
    }
}

impl<T: PartialEq + Clone, R: Reducer<T> + Clone> Sub<&'_ LinAlgReducedInt<T, R>>
    for &LinAlgReducedInt<T, R>
{
    type Output = LinAlgReducedInt<T, R>;

    fn sub(self, rhs: &LinAlgReducedInt<T, R>) -> Self::Output {
        LinAlgReducedInt(&self.0 - &rhs.0)
    }
}

impl<T: PartialEq, R: Reducer<T>> Sub<LinAlgReducedInt<T, R>> for LinAlgReducedInt<T, R> {
    type Output = LinAlgReducedInt<T, R>;

    fn sub(self, rhs: Self) -> Self::Output {
        LinAlgReducedInt(self.0 - rhs.0)
    }
}

impl<T: PartialEq + Clone, R: Reducer<T>> Sub<LinAlgReducedInt<T, R>> for &LinAlgReducedInt<T, R> {
    type Output = LinAlgReducedInt<T, R>;

    fn sub(self, rhs: LinAlgReducedInt<T, R>) -> Self::Output {
        LinAlgReducedInt(&self.0 - rhs.0)
    }
}

impl<T: PartialEq, R: Reducer<T>> Sub<T> for LinAlgReducedInt<T, R> {
    type Output = LinAlgReducedInt<T, R>;

    fn sub(self, rhs: T) -> Self::Output {
        LinAlgReducedInt(self.0 - rhs)
    }
}

// ---------
// PartialEq
// ---------
impl<T: PartialEq + Clone + Zero + One, R: Reducer<T> + Clone> PartialEq<LinAlgReducedInt<T, R>> for LinAlgReducedInt<T, R> {
    fn eq(&self, other: &Self) -> bool {
        if self.modulus() == other.modulus() {
            self.0.eq(&other.0)
        } else {
            Zero::is_zero(self) && Zero::is_zero(other)
        }
    }
}

// --------------
// ReducedInteger
// --------------
impl<T: PartialEq + Clone + Zero + One, R: Reducer<T> + Clone> ModularInteger for LinAlgReducedInt<T, R> {
    type Base = T;

    fn modulus(&self) -> T {
        self.0.modulus()
    }

    fn residue(&self) -> T {
        self.0.residue()
    }

    fn is_zero(&self) -> bool {
        self.0.is_zero()
    }

    fn convert(&self, n: T) -> Self {
        Self(self.0.convert(n))
    }

    fn double(self) -> Self {
        Self(self.0.double())
    }

    fn square(self) -> Self {
        Self(self.0.square())
    }
}

// ----
// Zero
// ----
impl<T: PartialEq + Clone + Zero + One, R: Reducer<T> + Clone> Zero for LinAlgReducedInt<T, R> {
    fn zero() -> Self {
        log::warn!("The ring modulo is not known.");
        Self(ReducedInt::new(T::zero(), &T::one()))
    }

    fn is_zero(&self) -> bool {
        self.0.residue() == T::zero()
    }
}

// ---
// One
// ---
impl<T: PartialEq + Clone + Zero + One, R: Reducer<T> + Clone> One for LinAlgReducedInt<T, R> {
    fn one() -> Self {
        log::warn!("The ring modulo is not known.");
        Self(ReducedInt::new(T::one(), &T::one()))
    }

    fn is_one(&self) -> bool {
        self.0.residue() == T::one()
    }
}

// ----
// Wrap
// ----
pub trait ReducedIntToLinAlgReducedInt {
    type InnerT;
    type InnerR: Reducer<Self::InnerT>;
    fn linalg(self) -> LinAlgReducedInt<Self::InnerT, Self::InnerR>;
}

impl<T, R: Reducer<T>> ReducedIntToLinAlgReducedInt for ReducedInt<T, R> {
    type InnerT = T;
    type InnerR = R;
    fn linalg(self) -> LinAlgReducedInt<Self::InnerT, Self::InnerR> {
        LinAlgReducedInt(self)
    }
}
