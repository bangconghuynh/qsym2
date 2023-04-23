use std::fmt;

use approx;
use derive_builder::Builder;
use ndarray::{s, Array1};
use ndarray_linalg::types::Lapack;
use num_complex::{Complex, ComplexFloat};
use num_traits::float::{Float, FloatConst};

use crate::angmom::spinor_rotation_3d::SpinConstraint;
use crate::aux::ao_basis::BasisAngularOrder;
use crate::aux::molecule::Molecule;

#[cfg(test)]
mod orbital_tests;

pub mod orbital_analysis;
pub mod orbital_transformation;

// ==================
// Struct definitions
// ==================

/// A structure to manage molecular orbitals. Each molecular orbital is essentially a one-electron
/// Slater determinant.
#[derive(Builder, Clone)]
#[builder(build_fn(validate = "Self::validate"))]
pub struct MolecularOrbital<'a, T>
where
    T: ComplexFloat + Lapack,
{
    /// The spin constraint associated with the coefficients describing this molecular orbital.
    spin_constraint: SpinConstraint,

    spin_index: usize,

    /// The angular order of the basis functions with respect to which the coefficients are
    /// expressed.
    bao: &'a BasisAngularOrder<'a>,

    /// A boolean indicating if inner products involving this molecular orbital should be the
    /// complex-symmetric bilinear form, rather than the conventional Hermitian sesquilinear form.
    complex_symmetric: bool,

    /// The associated molecule.
    mol: &'a Molecule,

    /// The coefficients describing this molecular orbital.
    coefficients: Array1<T>,

    /// The threshold for comparing determinants.
    threshold: <T as ComplexFloat>::Real,
}

impl<'a, T> MolecularOrbitalBuilder<'a, T>
where
    T: ComplexFloat + Lapack,
{
    fn validate(&self) -> Result<(), String> {
        let bao = self
            .bao
            .ok_or("No `BasisAngularOrder` found.".to_string())?;
        let nbas = bao.n_funcs();
        let coefficients = self
            .coefficients
            .as_ref()
            .ok_or("No coefficients found.".to_string())?;
        let spin_index = self
            .spin_index
            .ok_or("No `spin_index` found.".to_string())?;

        let spincons = match self
            .spin_constraint
            .as_ref()
            .ok_or("No spin constraint found.".to_string())?
        {
            SpinConstraint::Restricted(n) | SpinConstraint::Unrestricted(n, _) => {
                spin_index < usize::from(*n) && coefficients.shape()[0] == nbas
            }
            SpinConstraint::Generalised(nspins, _) => {
                spin_index == 0
                    && coefficients.shape()[0].rem_euclid(nbas) == 0
                    && coefficients.shape()[0].div_euclid(nbas) == usize::from(*nspins)
            }
        };
        if !spincons {
            log::error!("The coefficient vector fails to satisfy the specified spin constraint.");
        }

        let mol = self.mol.ok_or("No molecule found.".to_string())?;
        let natoms = mol.atoms.len() == bao.n_atoms();
        if !natoms {
            log::error!("The number of atoms in the molecule does not match the number of local sites in the basis.");
        }

        if spincons && natoms {
            Ok(())
        } else {
            Err("Molecular orbital validation failed.".to_string())
        }
    }
}

impl<'a, T> MolecularOrbital<'a, T>
where
    T: ComplexFloat + Clone + Lapack,
{
    /// Returns a builder to construct a new [`MolecularOrbital`].
    pub fn builder() -> MolecularOrbitalBuilder<'a, T> {
        MolecularOrbitalBuilder::default()
    }

    /// Augments the encoding of coefficients in this molecular orbital to that in the
    /// corresponding generalised spin constraint.
    ///
    /// # Returns
    ///
    /// The equivalent molecular orbital with the coefficients encoded in the generalised spin
    /// constraint.
    pub fn to_generalised(&self) -> Self {
        match self.spin_constraint {
            SpinConstraint::Restricted(n) => {
                let nbas = self.bao.n_funcs();

                let cr = &self.coefficients;
                let mut cg = Array1::<T>::zeros(nbas * usize::from(n));
                let start = nbas * self.spin_index;
                let end = nbas * (self.spin_index + 1);
                cg.slice_mut(s![start..end]).assign(cr);
                Self::builder()
                    .coefficients(cg)
                    .bao(self.bao)
                    .mol(self.mol)
                    .spin_constraint(SpinConstraint::Generalised(n, false))
                    .spin_index(0)
                    .complex_symmetric(self.complex_symmetric)
                    .threshold(self.threshold)
                    .build()
                    .expect("Unable to construct a generalised molecular orbital.")
            }
            SpinConstraint::Unrestricted(n, increasingm) => {
                let nbas = self.bao.n_funcs();

                let cr = &self.coefficients;
                let mut cg = Array1::<T>::zeros(nbas * usize::from(n));
                let start = nbas * self.spin_index;
                let end = nbas * (self.spin_index + 1);
                cg.slice_mut(s![start..end]).assign(cr);
                Self::builder()
                    .coefficients(cg)
                    .bao(self.bao)
                    .mol(self.mol)
                    .spin_constraint(SpinConstraint::Generalised(n, increasingm))
                    .spin_index(0)
                    .complex_symmetric(self.complex_symmetric)
                    .threshold(self.threshold)
                    .build()
                    .expect("Unable to construct a generalised molecular orbital.")
            }
            SpinConstraint::Generalised(_, _) => self.clone(),
        }
    }

    /// Returns a shared reference to the coefficient array.
    pub fn coefficients(&self) -> &Array1<T> {
        &self.coefficients
    }

    /// Returns a shared reference to the spin constraint.
    pub fn spin_constraint(&self) -> &SpinConstraint {
        &self.spin_constraint
    }

    /// Returns a shared reference to the [`BasisAngularOrder`].
    pub fn bao(&self) -> &BasisAngularOrder {
        &self.bao
    }
}

// =====================
// Trait implementations
// =====================

// ----
// From
// ----
impl<'a, T> From<MolecularOrbital<'a, T>> for MolecularOrbital<'a, Complex<T>>
where
    T: Float + FloatConst + Lapack,
    Complex<T>: Lapack,
{
    fn from(value: MolecularOrbital<'a, T>) -> Self {
        MolecularOrbital::<'a, Complex<T>>::builder()
            .coefficients(value.coefficients.map(|x| Complex::from(x)))
            .bao(value.bao)
            .mol(value.mol)
            .spin_constraint(value.spin_constraint)
            .spin_index(value.spin_index)
            .complex_symmetric(value.complex_symmetric)
            .threshold(value.threshold)
            .build()
            .expect("Unable to construct a complex molecular orbital.")
    }
}

// ---------
// PartialEq
// ---------
impl<'a, T> PartialEq for MolecularOrbital<'a, T>
where
    T: ComplexFloat<Real = f64> + Lapack,
{
    fn eq(&self, other: &Self) -> bool {
        let thresh = (self.threshold * other.threshold).sqrt();
        let coefficients_eq = approx::relative_eq!(
            (&self.coefficients - &other.coefficients)
                .map(|x| ComplexFloat::abs(*x).powi(2))
                .sum()
                .sqrt(),
            0.0,
            epsilon = thresh,
            max_relative = thresh,
        );
        self.spin_constraint == other.spin_constraint
            && self.spin_index == other.spin_index
            && self.bao == other.bao
            && self.mol == other.mol
            && coefficients_eq
    }
}

// --
// Eq
// --
impl<'a, T> Eq for MolecularOrbital<'a, T> where T: ComplexFloat<Real = f64> + Lapack {}

// -----
// Debug
// -----
impl<'a, T> fmt::Debug for MolecularOrbital<'a, T>
where
    T: fmt::Debug + ComplexFloat + Lapack,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "MolecularOrbital[{:?} (spin index {}): coefficient array of length {}]",
            self.spin_constraint,
            self.spin_index,
            self.coefficients.len()
        )?;
        Ok(())
    }
}

// -------
// Display
// -------
impl<'a, T> fmt::Display for MolecularOrbital<'a, T>
where
    T: fmt::Display + ComplexFloat + Lapack,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "MolecularOrbital[{:?} (spin index {}): coefficient array of length {}]",
            self.spin_constraint,
            self.spin_index,
            self.coefficients.len()
        )?;
        Ok(())
    }
}
