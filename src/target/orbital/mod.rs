use std::fmt;
use std::iter::Sum;

use approx;
use derive_builder::Builder;
use ndarray::{s, Array1, Array2};
use ndarray_linalg::types::Lapack;
use num_complex::{Complex, ComplexFloat};
use num_traits::float::{Float, FloatConst};

use crate::angmom::spinor_rotation_3d::SpinConstraint;
use crate::aux::ao_basis::BasisAngularOrder;
use crate::aux::molecule::Molecule;

// #[cfg(test)]
// mod determinant_tests;

// mod determinant_analysis;
// mod determinant_transformation;

// ==================
// Struct definitions
// ==================

/// A structure to manage molecular orbitals. Each molecular orbital is essentially a one-electron
/// Slater determinant.
#[derive(Builder, Clone)]
pub struct MolecularOrbital<'a, T>
where
    T: ComplexFloat + Lapack,
{
    /// The spin constraint associated with the coefficients describing this molecular orbital.
    spin_constraint: SpinConstraint,

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

impl<'a, T> MolecularOrbital<'a, T>
where
    T: ComplexFloat + Clone + Lapack,
{
    /// Returns a builder to construct a new [`SlaterDeterminant`].
    fn builder() -> MolecularOrbitalBuilder<'a, T> {
        MolecularOrbitalBuilder::default()
    }

    /// Constructs a new [`MolecularOrbital`] from its coefficients and associated molecular
    /// information.
    ///
    /// # Arguments
    ///
    /// * `c` - Coefficient array.
    /// * `bao` - A shared reference to a [`BasisAngularOrder`] structure which encapsulates
    /// angular-momentum information about the shells in the basis set.
    /// * `mol` - A shared reference to a [`Molecule`] structure which encapsulates information
    /// about the molecular structure.
    /// * `spincons` - The spin constraint in which the coefficient arrays are defined.
    /// * `thresh` - The threshold for numerical comparisons of determinants.
    pub fn new(
        c: Array1<T>,
        bao: &'a BasisAngularOrder<'a>,
        mol: &'a Molecule,
        spincons: SpinConstraint,
        complex_symmetric: bool,
        thresh: <T as ComplexFloat>::Real,
    ) -> Self {
        let mo = Self::builder()
            .coefficients(c)
            .bao(bao)
            .mol(mol)
            .spin_constraint(spincons)
            .complex_symmetric(complex_symmetric)
            .threshold(thresh)
            .build()
            .expect("Unable to construct a molecular orbital structure.");
        assert!(mo.verify(), "Invalid molecular orbital requested.");
        mo
    }

    pub fn to_generalised(&self, spin_index: usize) -> Self {
        match self.spin_constraint {
            SpinConstraint::Restricted(n) => {
                assert!(spin_index < usize::from(n));
                let nbas = self.bao.n_funcs();

                let cr = &self.coefficients;
                let mut cg = Array1::<T>::zeros(nbas * usize::from(n));
                let start = nbas * spin_index;
                let end = nbas * (spin_index + 1);
                cg.slice_mut(s![start..end]).assign(cr);
                Self::new(
                    cg,
                    self.bao,
                    self.mol,
                    SpinConstraint::Generalised(n, false),
                    self.complex_symmetric,
                    self.threshold,
                )
            }
            SpinConstraint::Unrestricted(n, increasingm) => {
                assert!(spin_index < usize::from(n));
                let nbas = self.bao.n_funcs();

                let cr = &self.coefficients;
                let mut cg = Array1::<T>::zeros(nbas * usize::from(n));
                let start = nbas * spin_index;
                let end = nbas * (spin_index + 1);
                cg.slice_mut(s![start..end]).assign(cr);
                Self::new(
                    cg,
                    self.bao,
                    self.mol,
                    SpinConstraint::Generalised(n, increasingm),
                    self.complex_symmetric,
                    self.threshold,
                )
            }
            SpinConstraint::Generalised(_, _) => self.clone(),
        }
    }

    /// Returns a shared reference to a vector of coefficient arrays.
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

    /// Verifies the validity of the molecular orbital, *i.e.* checks for consistency between
    /// coefficients, basis set shell structure, and spin constraint.
    fn verify(&self) -> bool {
        let nbas = self.bao.n_funcs();
        let spincons = match self.spin_constraint {
            SpinConstraint::Restricted(_) | SpinConstraint::Unrestricted(_, _) => {
                self.coefficients.shape()[0] == nbas
            }
            SpinConstraint::Generalised(nspins, _) => {
                self.coefficients.shape()[0].rem_euclid(nbas) == 0
                    && self.coefficients.shape()[0].div_euclid(nbas) == usize::from(nspins)
            }
        };
        if !spincons {
            log::error!("The coefficient vector fails to satisfy the specified spin constraint.");
        }
        let natoms = self.mol.atoms.len() == self.bao.n_atoms();
        if !natoms {
            log::error!("The number of atoms in the molecule does not match the number of local sites in the basis.");
        }
        spincons && natoms
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
        MolecularOrbital::<'a, Complex<T>>::new(
            value
                .coefficients
                .map(|x| Complex::from(x)),
            value.bao,
            value.mol,
            value.spin_constraint,
            value.complex_symmetric,
            value.threshold,
        )
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
    <T as ComplexFloat>::Real: Sum + From<u16> + fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "MolecularOrbital[{:?}: coefficient array of length {}]",
            self.spin_constraint,
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
    <T as ComplexFloat>::Real: Sum + From<u16> + fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "MolecularOrbital[{:?}: coefficient array of length {}]",
            self.spin_constraint,
            self.coefficients.len()
        )?;
        Ok(())
    }
}
