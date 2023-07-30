use std::fmt;
use std::iter::Sum;
use std::ops::{Add, Sub};

use approx;
use derive_builder::Builder;
use log;
use ndarray::Array2;
use ndarray_linalg::types::Lapack;
use num_complex::{Complex, ComplexFloat};
use num_traits::float::{Float, FloatConst};

use crate::angmom::spinor_rotation_3d::SpinConstraint;
use crate::auxiliary::molecule::Molecule;
use crate::basis::ao::BasisAngularOrder;

#[cfg(test)]
mod density_tests;

pub mod density_analysis;
mod density_transformation;

// ==================
// Struct definitions
// ==================

/// A structure to manage particle densities.
#[derive(Builder, Clone)]
#[builder(build_fn(validate = "Self::validate"))]
pub struct Density<'a, T>
where
    T: ComplexFloat + Lapack,
{
    /// The spin constraint associated with the density matrix describing this density.
    spin_constraint: SpinConstraint,

    /// The angular order of the basis functions with respect to which the density matrix is
    /// expressed.
    bao: &'a BasisAngularOrder<'a>,

    /// A boolean indicating if inner products involving this density should be the
    /// complex-symmetric bilinear form, rather than the conventional Hermitian sesquilinear form.
    complex_symmetric: bool,

    /// The associated molecule.
    mol: &'a Molecule,

    /// The density matrix describing this density.
    density_matrix: Array2<T>,

    /// The threshold for comparing densities.
    threshold: <T as ComplexFloat>::Real,
}

impl<'a, T> DensityBuilder<'a, T>
where
    T: ComplexFloat + Lapack,
{
    fn validate(&self) -> Result<(), String> {
        let bao = self
            .bao
            .ok_or("No `BasisAngularOrder` found.".to_string())?;
        let nbas = bao.n_funcs();
        let density_matrix = self
            .density_matrix
            .as_ref()
            .ok_or("No density matrices found.".to_string())?;
        let spincons = match self
            .spin_constraint
            .as_ref()
            .ok_or("No spin constraint found.".to_string())?
        {
            SpinConstraint::Restricted(_) | SpinConstraint::Unrestricted(_, _) => {
                density_matrix.shape() == [nbas, nbas]
            }
            SpinConstraint::Generalised(nspins, _) => {
                (0..2).all(|i| {
                    density_matrix.shape()[i].rem_euclid(nbas) == 0
                        && density_matrix.shape()[i].div_euclid(nbas) == usize::from(*nspins)
                }) && density_matrix.shape()[0] == density_matrix.shape()[1]
            }
        };
        if !spincons {
            log::error!("The density matrix fails to satisfy the specified spin constraint.");
        }

        let mol = self.mol.ok_or("No molecule found.".to_string())?;
        let natoms = mol.atoms.len() == bao.n_atoms();
        if !natoms {
            log::error!("The number of atoms in the molecule does not match the number of local sites in the basis.");
        }
        if spincons && natoms {
            Ok(())
        } else {
            Err("Density validation failed.".to_string())
        }
    }
}

impl<'a, T> Density<'a, T>
where
    T: ComplexFloat + Clone + Lapack,
{
    /// Returns a builder to construct a new [`Density`].
    pub fn builder() -> DensityBuilder<'a, T> {
        DensityBuilder::default()
    }

    /// Returns the complex-symmetric flag of the density.
    pub fn complex_symmetric(&self) -> bool {
        self.complex_symmetric
    }

    /// Returns the spin constraint imposed on the density matrix.
    pub fn spin_constraint(&self) -> &SpinConstraint {
        &self.spin_constraint
    }

    /// Returns the basis angular order information of the basis set in which the density matrix is
    /// expressed.
    pub fn bao(&self) -> &BasisAngularOrder {
        self.bao
    }

    /// Returns a shared reference to the density matrix.
    pub fn density_matrix(&self) -> &Array2<T> {
        &self.density_matrix
    }

    /// Returns the threshold with which densities are compared.
    pub fn threshold(&self) -> <T as ComplexFloat>::Real {
        self.threshold
    }
}

// =====================
// Trait implementations
// =====================

// ----
// From
// ----
impl<'a, T> From<Density<'a, T>> for Density<'a, Complex<T>>
where
    T: Float + FloatConst + Lapack,
    Complex<T>: Lapack,
{
    fn from(value: Density<'a, T>) -> Self {
        Density::<'a, Complex<T>>::builder()
            .density_matrix(
                value.density_matrix.map(Complex::from)
            )
            .bao(value.bao)
            .mol(value.mol)
            .spin_constraint(value.spin_constraint)
            .complex_symmetric(value.complex_symmetric)
            .threshold(value.threshold)
            .build()
            .expect("Unable to complexify a `Density`.")
    }
}

// ---------
// PartialEq
// ---------
impl<'a, T> PartialEq for Density<'a, T>
where
    T: ComplexFloat<Real = f64> + Lapack,
{
    fn eq(&self, other: &Self) -> bool {
        let thresh = (self.threshold * other.threshold).sqrt();
        let density_matrix_eq = approx::relative_eq!(
            (&self.density_matrix - &other.density_matrix)
                .map(|x| ComplexFloat::abs(*x).powi(2))
                .sum()
                .sqrt(),
            0.0,
            epsilon = thresh,
            max_relative = thresh,
        );
        self.spin_constraint == other.spin_constraint
            && self.bao == other.bao
            && self.mol == other.mol
            && density_matrix_eq
    }
}

// --
// Eq
// --
impl<'a, T> Eq for Density<'a, T> where T: ComplexFloat<Real = f64> + Lapack {}

// -----
// Debug
// -----
impl<'a, T> fmt::Debug for Density<'a, T>
where
    T: fmt::Debug + ComplexFloat + Lapack,
    <T as ComplexFloat>::Real: Sum + From<u16> + fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Density[{:?}: density matrix of dimensions {}]",
            self.spin_constraint,
            self.density_matrix.shape()
                .iter()
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join("×")
        )?;
        Ok(())
    }
}

// -------
// Display
// -------
impl<'a, T> fmt::Display for Density<'a, T>
where
    T: fmt::Display + ComplexFloat + Lapack,
    <T as ComplexFloat>::Real: Sum + From<u16> + fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Density[{:?}: density matrix of dimensions {}]",
            self.spin_constraint,
            self.density_matrix.shape()
                .iter()
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join("×")
        )?;
        Ok(())
    }
}

// ---
// Add
// ---
impl<'a, T> Add<&'_ Density<'a, T>> for &Density<'a, T>
where
    T: ComplexFloat + Lapack,
{
    type Output = Density<'a, T>;

    fn add(self, rhs: &Density<'a, T>) -> Self::Output {
        assert_eq!(
            self.spin_constraint, rhs.spin_constraint,
            "Inconsistent spin constraints between `self` and `rhs`."
        );
        assert_eq!(
            self.density_matrix.shape(), rhs.density_matrix.shape(),
            "Inconsistent shapes of density matrices between `self` and `rhs`."
        );
        assert_eq!(
            self.bao, rhs.bao,
            "Inconsistent basis angular order between `self` and `rhs`."
        );
        Density::<T>::builder()
            .density_matrix(&self.density_matrix + &rhs.density_matrix)
            .bao(self.bao)
            .mol(self.mol)
            .spin_constraint(self.spin_constraint.clone())
            .complex_symmetric(self.complex_symmetric)
            .threshold(self.threshold)
            .build()
            .expect("Unable to add two densities together.")
    }
}

impl<'a, T> Add<&'_ Density<'a, T>> for Density<'a, T>
where
    T: ComplexFloat + Lapack,
{
    type Output = Density<'a, T>;

    fn add(self, rhs: &Self) -> Self::Output {
        &self + rhs
    }
}

impl<'a, T> Add<Density<'a, T>> for Density<'a, T>
where
    T: ComplexFloat + Lapack,
{
    type Output = Density<'a, T>;

    fn add(self, rhs: Self) -> Self::Output {
        &self + &rhs
    }
}

impl<'a, T> Add<Density<'a, T>> for &Density<'a, T>
where
    T: ComplexFloat + Lapack,
{
    type Output = Density<'a, T>;

    fn add(self, rhs: Density<'a, T>) -> Self::Output {
        self + &rhs
    }
}

// ---
// Sub
// ---
impl<'a, T> Sub<&'_ Density<'a, T>> for &Density<'a, T>
where
    T: ComplexFloat + Lapack,
{
    type Output = Density<'a, T>;

    fn sub(self, rhs: &Density<'a, T>) -> Self::Output {
        assert_eq!(
            self.spin_constraint, rhs.spin_constraint,
            "Inconsistent spin constraints between `self` and `rhs`."
        );
        assert_eq!(
            self.density_matrix.shape(), rhs.density_matrix.shape(),
            "Inconsistent shapes of density matrices between `self` and `rhs`."
        );
        assert_eq!(
            self.bao, rhs.bao,
            "Inconsistent basis angular order between `self` and `rhs`."
        );
        Density::<T>::builder()
            .density_matrix(&self.density_matrix - &rhs.density_matrix)
            .bao(self.bao)
            .mol(self.mol)
            .spin_constraint(self.spin_constraint.clone())
            .complex_symmetric(self.complex_symmetric)
            .threshold(self.threshold)
            .build()
            .expect("Unable to subtract two densities.")
    }
}

impl<'a, T> Sub<&'_ Density<'a, T>> for Density<'a, T>
where
    T: ComplexFloat + Lapack,
{
    type Output = Density<'a, T>;

    fn sub(self, rhs: &Self) -> Self::Output {
        &self - rhs
    }
}

impl<'a, T> Sub<Density<'a, T>> for Density<'a, T>
where
    T: ComplexFloat + Lapack,
{
    type Output = Density<'a, T>;

    fn sub(self, rhs: Self) -> Self::Output {
        &self - &rhs
    }
}

impl<'a, T> Sub<Density<'a, T>> for &Density<'a, T>
where
    T: ComplexFloat + Lapack,
{
    type Output = Density<'a, T>;

    fn sub(self, rhs: Density<'a, T>) -> Self::Output {
        self - &rhs
    }
}
