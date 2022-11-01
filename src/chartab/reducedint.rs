use std::ops::{Add, Div, Mul, Neg, Sub};

use log;
use num_modular::{ModularInteger, ReducedInt, Reducer};
use num_traits::{Inv, One, Pow, Zero};

#[cfg(test)]
#[path = "reducedint_tests.rs"]
mod reducedint_tests;

/// A wrapper struct to represent an integer in a modulo ring.
#[derive(Clone, Copy, Debug)]
pub enum LinAlgReducedInt<T, R: Reducer<T>> {
    KnownChar(ReducedInt<T, R>),
    Zero,
    One,
}

// ---
// Add
// ---
impl<T, R> Add<&'_ LinAlgReducedInt<T, R>> for &LinAlgReducedInt<T, R>
where
    T: Zero + One + PartialEq + Clone,
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
    T: Zero + One + PartialEq + Clone,
    R: Reducer<T> + Clone,
{
    type Output = LinAlgReducedInt<T, R>;

    fn add(self, rhs: &Self) -> Self::Output {
        &self + rhs
    }
}

impl<T, R> Add<LinAlgReducedInt<T, R>> for LinAlgReducedInt<T, R>
where
    T: Zero + One + PartialEq + Clone,
    R: Reducer<T> + Clone,
{
    type Output = LinAlgReducedInt<T, R>;

    fn add(self, rhs: Self) -> Self::Output {
        &self + &rhs
    }
}

impl<T, R> Add<LinAlgReducedInt<T, R>> for &LinAlgReducedInt<T, R>
where
    T: Zero + One + PartialEq + Clone,
    R: Reducer<T> + Clone,
{
    type Output = LinAlgReducedInt<T, R>;

    fn add(self, rhs: LinAlgReducedInt<T, R>) -> Self::Output {
        self + &rhs
    }
}

impl<T, R> Add<T> for LinAlgReducedInt<T, R>
where
    T: Zero + One + PartialEq + Clone,
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
    T: Zero + One + PartialEq + Clone,
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
    T: Zero + One + PartialEq + Clone,
    R: Reducer<T> + Clone,
{
    type Output = LinAlgReducedInt<T, R>;

    fn mul(self, rhs: &Self) -> Self::Output {
        &self * rhs
    }
}

impl<T, R> Mul<LinAlgReducedInt<T, R>> for LinAlgReducedInt<T, R>
where
    T: Zero + One + PartialEq + Clone,
    R: Reducer<T> + Clone,
{
    type Output = LinAlgReducedInt<T, R>;

    fn mul(self, rhs: Self) -> Self::Output {
        &self * &rhs
    }
}

impl<T, R> Mul<LinAlgReducedInt<T, R>> for &LinAlgReducedInt<T, R>
where
    T: Zero + One + PartialEq + Clone,
    R: Reducer<T> + Clone,
{
    type Output = LinAlgReducedInt<T, R>;

    fn mul(self, rhs: LinAlgReducedInt<T, R>) -> Self::Output {
        self * &rhs
    }
}

impl<T, R> Mul<T> for LinAlgReducedInt<T, R>
where
    T: Zero + One + PartialEq + Clone,
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
    T: Zero + One + PartialEq + Clone,
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
    T: Zero + One + PartialEq + Clone,
    R: Reducer<T> + Clone
{
    type Output = LinAlgReducedInt<T, R>;

    fn sub(self, rhs: &Self) -> Self::Output {
        &self - rhs
    }
}

impl<T, R> Sub<LinAlgReducedInt<T, R>> for LinAlgReducedInt<T, R>
where
    T: Zero + One + PartialEq + Clone,
    R: Reducer<T> + Clone
{
    type Output = LinAlgReducedInt<T, R>;

    fn sub(self, rhs: Self) -> Self::Output {
        &self - &rhs
    }
}

impl<T, R> Sub<LinAlgReducedInt<T, R>> for &LinAlgReducedInt<T, R>
where
    T: Zero + One + PartialEq + Clone,
    R: Reducer<T> + Clone
{
    type Output = LinAlgReducedInt<T, R>;

    fn sub(self, rhs: LinAlgReducedInt<T, R>) -> Self::Output {
        self - &rhs
    }
}

impl<T, R> Sub<T> for LinAlgReducedInt<T, R>
where
    T: Zero + One + PartialEq + Clone,
    R: Reducer<T> + Clone
{
    type Output = LinAlgReducedInt<T, R>;

    fn sub(self, rhs: T) -> Self::Output {
        &self - &self.convert(rhs)
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
    T: Zero + One + PartialEq + Clone,
    R: Reducer<T> + Clone
{
    type Output = LinAlgReducedInt<T, R>;

    fn div(self, rhs: &Self) -> Self::Output {
        &self / rhs
    }
}

impl<T, R> Div<LinAlgReducedInt<T, R>> for LinAlgReducedInt<T, R>
where
    T: Zero + One + PartialEq + Clone,
    R: Reducer<T> + Clone
{
    type Output = LinAlgReducedInt<T, R>;

    fn div(self, rhs: Self) -> Self::Output {
        &self / &rhs
    }
}

impl<T, R> Div<LinAlgReducedInt<T, R>> for &LinAlgReducedInt<T, R>
where
    T: Zero + One + PartialEq + Clone,
    R: Reducer<T> + Clone
{
    type Output = LinAlgReducedInt<T, R>;

    fn div(self, rhs: LinAlgReducedInt<T, R>) -> Self::Output {
        self / &rhs
    }
}

impl<T, R> Div<T> for LinAlgReducedInt<T, R>
where
    T: Zero + One + PartialEq + Clone,
    R: Reducer<T> + Clone
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
    T: Zero + One + PartialEq + Clone,
    R: Reducer<T> + Clone
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
    T: Zero + One + PartialEq + Clone,
    R: Reducer<T> + Clone
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
    T: Zero + One + PartialEq + Clone,
    R: Reducer<T> + Clone
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
    T: Zero + One + PartialEq + Clone,
    R: Reducer<T> + Clone
{
    type Output = LinAlgReducedInt<T, R>;

    fn neg(self) -> Self::Output {
        -&self.clone()
    }
}

// // ---
// // Pow
// // ---
// impl<T: Zero + One + PartialEq + Clone, R: Reducer<T> + Clone> Pow<T> for &LinAlgReducedInt<T, R> {
//     type Output = LinAlgReducedInt<T, R>;

//     fn pow(self, rhs: T) -> Self::Output {
//         if Zero::is_zero(self) || self.is_one() {
//             self.clone()
//         } else {
//             LinAlgReducedInt(self.reduced_int.clone().pow(rhs))
//         }
//     }
// }

// impl<T: Zero + One + PartialEq + Clone, R: Reducer<T> + Clone> Pow<T> for LinAlgReducedInt<T, R> {
//     type Output = LinAlgReducedInt<T, R>;

//     fn pow(self, rhs: T) -> Self::Output {
//         (&self).pow(rhs)
//     }
// }

// // ---------
// // PartialEq
// // ---------
// impl<T: PartialEq + Clone + Zero + One, R: Reducer<T> + Clone> PartialEq<LinAlgReducedInt<T, R>>
//     for LinAlgReducedInt<T, R>
// {
//     fn eq(&self, other: &Self) -> bool {
//         if self.modulus() == other.modulus() {
//             self.reduced_int.eq(&other.reduced_int)
//         } else {
//             Zero::is_zero(self) && Zero::is_zero(other)
//         }
//     }
// }

// // --------------
// // ReducedInteger
// // --------------
// impl<T: PartialEq + Clone + Zero + One, R: Reducer<T> + Clone> ModularInteger
//     for LinAlgReducedInt<T, R>
// {
//     type Base = T;

//     fn modulus(&self) -> T {
//         self.reduced_int.modulus()
//     }

//     fn residue(&self) -> T {
//         if self.zero {
//             debug_assert!(!self.one);
//             T::zero()
//         } else if self.one {
//             T::one()
//         } else {
//             let res = self.reduced_int.residue();
//             debug_assert!(res != T::zero() && res != T::one());
//             res
//         }
//     }

//     fn is_zero(&self) -> bool {
//         Zero::is_zero(self)
//     }

//     fn convert(&self, n: T) -> Self {
//         Self(self.reduced_int.convert(n))
//     }

//     fn double(self) -> Self {
//         Self(self.reduced_int.double())
//     }

//     fn square(self) -> Self {
//         Self(self.reduced_int.square())
//     }
// }

// ----
// Zero
// ----
impl<T: PartialEq + Clone + Zero + One, R: Reducer<T> + Clone> Zero for LinAlgReducedInt<T, R> {
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

// // ---
// // One
// // ---
// impl<T: PartialEq + Clone + Zero + One, R: Reducer<T> + Clone> One for LinAlgReducedInt<T, R> {
//     fn one() -> Self {
//         Self::One
//     }

//     fn is_one(&self) -> bool {
//         match self {
//             Self::KnownChar(res) => res.residue() == T::one(),
//             Self::Zero => false,
//             Self::One => true,
//         }
//     }
// }

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
