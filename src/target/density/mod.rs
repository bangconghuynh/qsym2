//! Electron densities.

use std::fmt;
use std::iter::Sum;
use std::ops::{Add, Index, Sub};

use approx;
use derive_builder::Builder;
use itertools::Itertools;
use log;
use ndarray::Array2;
use ndarray_linalg::types::Lapack;
use num_complex::{Complex, ComplexFloat};
use num_traits::float::{Float, FloatConst};

use crate::angmom::spinor_rotation_3d::StructureConstraint;
use crate::auxiliary::molecule::Molecule;
use crate::basis::ao::BasisAngularOrder;

#[cfg(test)]
mod density_tests;

pub mod density_analysis;
mod density_transformation;

// ==================
// Struct definitions
// ==================

/// Wrapper structure to manage references to multiple densities of a single state.
#[derive(Builder, Clone)]
#[builder(build_fn(validate = "Self::validate"))]
pub struct Densities<'a, T, SC>
where
    T: ComplexFloat + Lapack,
    SC: StructureConstraint + fmt::Display,
{
    /// The structure constraint associated with the multiple densities.
    structure_constraint: SC,

    /// A vector containing references to the multiple densities, one for each spin space.
    densities: Vec<&'a Density<'a, T>>,
}

impl<'a, T, SC> Densities<'a, T, SC>
where
    T: ComplexFloat + Lapack,
    SC: StructureConstraint + Clone + fmt::Display,
{
    /// Returns a builder to construct a new `Densities`.
    pub fn builder() -> DensitiesBuilder<'a, T, SC> {
        DensitiesBuilder::default()
    }

    pub fn iter(&self) -> impl Iterator<Item = &Density<'a, T>> {
        self.densities.iter().cloned()
    }
}

impl<'a, T, SC> DensitiesBuilder<'a, T, SC>
where
    T: ComplexFloat + Lapack,
    SC: StructureConstraint + fmt::Display,
{
    fn validate(&self) -> Result<(), String> {
        let densities = self
            .densities
            .as_ref()
            .ok_or("No `densities` found.".to_string())?;
        let structure_constraint = self
            .structure_constraint
            .as_ref()
            .ok_or("No structure constraint found.".to_string())?;
        let num_dens = structure_constraint.n_coefficient_matrices()
            * structure_constraint.n_explicit_comps_per_coefficient_matrix();
        if densities.len() != num_dens {
            Err(format!(
                "{} {} expected in structure constraint {}, but {} found.",
                num_dens,
                structure_constraint,
                if num_dens == 1 {
                    "density"
                } else {
                    "densities"
                },
                densities.len()
            ))
        } else {
            Ok(())
        }
    }
}

impl<'a, T, SC> Index<usize> for Densities<'a, T, SC>
where
    T: ComplexFloat + Lapack,
    SC: StructureConstraint + fmt::Display,
{
    type Output = Density<'a, T>;

    fn index(&self, index: usize) -> &Self::Output {
        self.densities[index]
    }
}

/// Wrapper structure to manage multiple owned densities of a single state.
#[derive(Builder, Clone)]
#[builder(build_fn(validate = "Self::validate"))]
pub struct DensitiesOwned<'a, T, SC>
where
    T: ComplexFloat + Lapack,
    SC: StructureConstraint + fmt::Display,
{
    /// The structure constraint associated with the multiple densities.
    structure_constraint: SC,

    /// A vector containing the multiple densities, one for each spin space.
    densities: Vec<Density<'a, T>>,
}

impl<'a, T, SC> DensitiesOwned<'a, T, SC>
where
    T: ComplexFloat + Lapack,
    SC: StructureConstraint + Clone + fmt::Display,
{
    /// Returns a builder to construct a new `DensitiesOwned`.
    pub fn builder() -> DensitiesOwnedBuilder<'a, T, SC> {
        DensitiesOwnedBuilder::default()
    }

    pub fn iter(&self) -> impl Iterator<Item = &Density<'a, T>> {
        self.densities.iter()
    }
}

impl<'b, 'a: 'b, T, SC> DensitiesOwned<'a, T, SC>
where
    T: ComplexFloat + Lapack,
    SC: StructureConstraint + Clone + fmt::Display,
{
    pub fn as_ref(&'a self) -> Densities<'b, T, SC> {
        Densities::builder()
            .structure_constraint(self.structure_constraint.clone())
            .densities(self.iter().collect_vec())
            .build()
            .expect("Unable to convert `DensitiesOwned` to `Densities`.")
    }
}

impl<'a, T, SC> DensitiesOwnedBuilder<'a, T, SC>
where
    T: ComplexFloat + Lapack,
    SC: StructureConstraint + Clone + fmt::Display,
{
    fn validate(&self) -> Result<(), String> {
        let densities = self
            .densities
            .as_ref()
            .ok_or("No `densities` found.".to_string())?;
        let structure_constraint = self
            .structure_constraint
            .as_ref()
            .ok_or("No spin constraint found.".to_string())?;
        let num_dens = structure_constraint.n_coefficient_matrices()
            * structure_constraint.n_explicit_comps_per_coefficient_matrix();
        if densities.len() != num_dens {
            Err(format!(
                "{} {} expected in structure constraint {}, but {} found.",
                num_dens,
                structure_constraint,
                if num_dens == 1 {
                    "density"
                } else {
                    "densities"
                },
                densities.len()
            ))
        } else {
            Ok(())
        }
    }
}

impl<'a, T, SC> Index<usize> for DensitiesOwned<'a, T, SC>
where
    T: ComplexFloat + Lapack,
    SC: StructureConstraint + fmt::Display,
{
    type Output = Density<'a, T>;

    fn index(&self, index: usize) -> &Self::Output {
        &self.densities[index]
    }
}

/// Structure to manage particle densities in the simplest basis specified by a basis angular order
/// structure.
#[derive(Builder, Clone)]
#[builder(build_fn(validate = "Self::validate"))]
pub struct Density<'a, T>
where
    T: ComplexFloat + Lapack,
{
    /// The angular order of the basis functions with respect to which the density matrix is
    /// expressed.
    bao: &'a BasisAngularOrder<'a>,

    /// A boolean indicating if inner products involving this density should be the
    /// complex-symmetric bilinear form, rather than the conventional Hermitian sesquilinear form.
    complex_symmetric: bool,

    /// A boolean indicating if the density has been acted on by an antiunitary operation. This is
    /// so that the correct metric can be used during overlap evaluation.
    #[builder(default = "false")]
    complex_conjugated: bool,

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

        let denmat_shape = density_matrix.shape() == [nbas, nbas];
        if !denmat_shape {
            log::error!(
                "The density matrix dimensions ({:?}) are incompatible with the basis ({nbas} {}).",
                density_matrix.shape(),
                if nbas != 1 { "functions" } else { "function" }
            );
        }

        let mol = self.mol.ok_or("No molecule found.".to_string())?;
        let natoms = mol.atoms.len() == bao.n_atoms();
        if !natoms {
            log::error!("The number of atoms in the molecule does not match the number of local sites in the basis.");
        }
        if denmat_shape && natoms {
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

    /// Returns the complex-symmetric boolean of the density.
    pub fn complex_symmetric(&self) -> bool {
        self.complex_symmetric
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
            .density_matrix(value.density_matrix.map(Complex::from))
            .bao(value.bao)
            .mol(value.mol)
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
        self.bao == other.bao && self.mol == other.mol && density_matrix_eq
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
            "Density[density matrix of dimensions {}]",
            self.density_matrix
                .shape()
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
            "Density[density matrix of dimensions {}]",
            self.density_matrix
                .shape()
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
            self.density_matrix.shape(),
            rhs.density_matrix.shape(),
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
            self.density_matrix.shape(),
            rhs.density_matrix.shape(),
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
