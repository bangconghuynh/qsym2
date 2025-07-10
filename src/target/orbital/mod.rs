//! Orbitals.

use std::fmt;

use anyhow::{self, format_err};
use approx;
use derive_builder::Builder;
use ndarray::{s, Array1, Array2, Ix2};
use ndarray_einsum::*;
use ndarray_linalg::types::Lapack;
use num_complex::{Complex, ComplexFloat};
use num_traits::float::{Float, FloatConst};

use crate::angmom::spinor_rotation_3d::{SpinConstraint, SpinOrbitCoupled, StructureConstraint};
use crate::auxiliary::molecule::Molecule;
use crate::basis::ao::BasisAngularOrder;
use crate::target::density::Density;

#[cfg(test)]
mod orbital_tests;

pub mod orbital_analysis;
mod orbital_transformation;

// ==================
// Struct definitions
// ==================

/// Structure to manage molecular orbitals. Each molecular orbital is essentially a one-electron
/// Slater determinant.
#[derive(Builder, Clone)]
#[builder(build_fn(validate = "Self::validate"))]
pub struct MolecularOrbital<'a, T, SC>
where
    T: ComplexFloat + Lapack,
    SC: StructureConstraint,
{
    /// The structure constraint associated with the coefficients describing this molecular orbital.
    structure_constraint: SC,

    /// If the structure constraint allows for multiple components, this gives the component index of
    /// this molecular orbital.
    component_index: usize,

    /// A boolean indicating if the orbital has been acted on by an antiunitary operation. This is
    /// so that the correct metric can be used during overlap evaluation.
    #[builder(default = "false")]
    complex_conjugated: bool,

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

    /// The energy of this molecular orbital.
    #[builder(default = "None")]
    energy: Option<T>,

    /// The threshold for comparing determinants.
    threshold: <T as ComplexFloat>::Real,
}

impl<'a, T, SC> MolecularOrbitalBuilder<'a, T, SC>
where
    T: ComplexFloat + Lapack,
    SC: StructureConstraint,
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
        let component_index = self
            .component_index
            .ok_or("No `component_index` found.".to_string())?;

        let structcons = self
            .structure_constraint
            .as_ref()
            .ok_or("No structure constraints found.".to_string())?;
        let coefficients_shape_check = {
            let nrows = nbas * structcons.n_explicit_comps_per_coefficient_matrix();
            if !coefficients.shape()[0] == nrows {
                log::error!(
                    "Unexpected shapes of coefficient vector: {} {} expected, but {} found.",
                    nrows,
                    if nrows == 1 { "row" } else { "rows" },
                    coefficients.shape()[0],
                );
                false
            } else {
                true
            }
        };

        let mol = self.mol.ok_or("No molecule found.".to_string())?;
        let natoms_check = mol.atoms.len() == bao.n_atoms();
        if !natoms_check {
            log::error!("The number of atoms in the molecule does not match the number of local sites in the basis.");
        }

        if coefficients_shape_check && natoms_check {
            Ok(())
        } else {
            Err("Molecular orbital validation failed.".to_string())
        }
    }
}

impl<'a, T, SC> MolecularOrbital<'a, T, SC>
where
    T: ComplexFloat + Clone + Lapack,
    SC: StructureConstraint + Clone,
{
    /// Returns a builder to construct a new [`MolecularOrbital`].
    pub fn builder() -> MolecularOrbitalBuilder<'a, T, SC> {
        MolecularOrbitalBuilder::default()
    }
}

impl<'a, T, SC> MolecularOrbital<'a, T, SC>
where
    T: ComplexFloat + Clone + Lapack,
    SC: StructureConstraint,
{
    /// Returns a shared reference to the coefficient array.
    pub fn coefficients(&self) -> &Array1<T> {
        &self.coefficients
    }

    /// Returns a shared reference to the structure constraint.
    pub fn structure_constraint(&self) -> &SC {
        &self.structure_constraint
    }

    /// Returns a shared reference to the [`BasisAngularOrder`] description of the basis in which
    /// the orbital coefficients are written.
    pub fn bao(&self) -> &BasisAngularOrder {
        self.bao
    }

    /// Returns the molecule associated with this molecular orbital.
    pub fn mol(&self) -> &Molecule {
        self.mol
    }

    /// Returns the complex-symmetric flag of the molecular orbital.
    pub fn complex_symmetric(&self) -> bool {
        self.complex_symmetric
    }

    /// Returns the threshold with which molecular orbitals are compared.
    pub fn threshold(&self) -> <T as ComplexFloat>::Real {
        self.threshold
    }
}

impl<'a, T> MolecularOrbital<'a, T, SpinConstraint>
where
    T: ComplexFloat + Clone + Lapack,
{
    /// Augments the encoding of coefficients in this molecular orbital to that in the
    /// corresponding generalised spin constraint.
    ///
    /// # Returns
    ///
    /// The equivalent molecular orbital with the coefficients encoded in the generalised spin
    /// constraint.
    pub fn to_generalised(&self) -> Self {
        match self.structure_constraint {
            SpinConstraint::Restricted(n) => {
                let nbas = self.bao.n_funcs();

                let cr = &self.coefficients;
                let mut cg = Array1::<T>::zeros(nbas * usize::from(n));
                let start = nbas * self.component_index;
                let end = nbas * (self.component_index + 1);
                cg.slice_mut(s![start..end]).assign(cr);
                Self::builder()
                    .coefficients(cg)
                    .bao(self.bao)
                    .mol(self.mol)
                    .structure_constraint(SpinConstraint::Generalised(n, false))
                    .component_index(0)
                    .complex_symmetric(self.complex_symmetric)
                    .threshold(self.threshold)
                    .build()
                    .expect("Unable to construct a generalised molecular orbital.")
            }
            SpinConstraint::Unrestricted(n, increasingm) => {
                let nbas = self.bao.n_funcs();

                let cr = &self.coefficients;
                let mut cg = Array1::<T>::zeros(nbas * usize::from(n));
                let start = nbas * self.component_index;
                let end = nbas * (self.component_index + 1);
                cg.slice_mut(s![start..end]).assign(cr);
                Self::builder()
                    .coefficients(cg)
                    .bao(self.bao)
                    .mol(self.mol)
                    .structure_constraint(SpinConstraint::Generalised(n, increasingm))
                    .component_index(0)
                    .complex_symmetric(self.complex_symmetric)
                    .threshold(self.threshold)
                    .build()
                    .expect("Unable to construct a generalised molecular orbital.")
            }
            SpinConstraint::Generalised(_, _) => self.clone(),
        }
    }
}

impl<'a> MolecularOrbital<'a, f64, SpinConstraint> {
    /// Constructs the total density of the molecular orbital.
    pub fn to_total_density(&'a self) -> Result<Density<'a, f64>, anyhow::Error> {
        match self.structure_constraint {
            SpinConstraint::Restricted(nspins) => {
                let denmat = f64::from(nspins)
                    * einsum(
                        "m,n->mn",
                        &[&self.coefficients.view(), &self.coefficients.view()],
                    )
                    .expect("Unable to construct a density matrix from the coefficient matrix.")
                    .into_dimensionality::<Ix2>()
                    .expect("Unable to convert the resultant density matrix to two dimensions.");
                Density::<f64>::builder()
                    .density_matrix(denmat)
                    .bao(self.bao())
                    .mol(self.mol())
                    .complex_symmetric(self.complex_symmetric())
                    .threshold(self.threshold())
                    .build()
                    .map_err(|err| format_err!(err))
            }
            SpinConstraint::Unrestricted(_, _) => {
                let denmat = einsum(
                    "m,n->mn",
                    &[&self.coefficients.view(), &self.coefficients.view()],
                )
                .expect("Unable to construct a density matrix from the coefficient matrix.")
                .into_dimensionality::<Ix2>()
                .expect("Unable to convert the resultant density matrix to two dimensions.");
                Density::<f64>::builder()
                    .density_matrix(denmat)
                    .bao(self.bao())
                    .mol(self.mol())
                    .complex_symmetric(self.complex_symmetric())
                    .threshold(self.threshold())
                    .build()
                    .map_err(|err| format_err!(err))
            }
            SpinConstraint::Generalised(nspins, _) => {
                let full_denmat = einsum(
                    "m,n->mn",
                    &[&self.coefficients.view(), &self.coefficients.view()],
                )
                .expect("Unable to construct a density matrix from the coefficient matrix.")
                .into_dimensionality::<Ix2>()
                .expect("Unable to convert the resultant density matrix to two dimensions.");
                let nspatial = self.bao().n_funcs();
                let denmat = (0..usize::from(nspins)).fold(
                    Array2::<f64>::zeros((nspatial, nspatial)),
                    |acc, ispin| {
                        acc + full_denmat.slice(s![
                            ispin * nspatial..(ispin + 1) * nspatial,
                            ispin * nspatial..(ispin + 1) * nspatial
                        ])
                    },
                );
                Density::<f64>::builder()
                    .density_matrix(denmat)
                    .bao(self.bao())
                    .mol(self.mol())
                    .complex_symmetric(self.complex_symmetric())
                    .threshold(self.threshold())
                    .build()
                    .map_err(|err| format_err!(err))
            }
        }
    }
}

impl<'a, T> MolecularOrbital<'a, Complex<T>, SpinConstraint>
where
    T: Float + FloatConst + Lapack + From<u16>,
    Complex<T>: Lapack,
{
    /// Constructs the total density of the molecular orbital.
    pub fn to_total_density(&'a self) -> Result<Density<'a, Complex<T>>, anyhow::Error> {
        match self.structure_constraint {
            SpinConstraint::Restricted(nspins) => {
                let nspins_t = Complex::<T>::from(<T as From<u16>>::from(nspins));
                let denmat = einsum(
                    "m,n->mn",
                    &[
                        &self.coefficients.view(),
                        &self.coefficients.map(Complex::conj).view(),
                    ],
                )
                .expect("Unable to construct a density matrix from the coefficient matrix.")
                .into_dimensionality::<Ix2>()
                .expect("Unable to convert the resultant density matrix to two dimensions.")
                .map(|x| x * nspins_t);
                Density::<Complex<T>>::builder()
                    .density_matrix(denmat)
                    .bao(self.bao())
                    .mol(self.mol())
                    .complex_symmetric(self.complex_symmetric())
                    .threshold(self.threshold())
                    .build()
                    .map_err(|err| format_err!(err))
            }
            SpinConstraint::Unrestricted(_, _) => {
                let denmat = einsum(
                    "m,n->mn",
                    &[
                        &self.coefficients.view(),
                        &self.coefficients.map(Complex::conj).view(),
                    ],
                )
                .expect("Unable to construct a density matrix from the coefficient matrix.")
                .into_dimensionality::<Ix2>()
                .expect("Unable to convert the resultant density matrix to two dimensions.");
                Density::<Complex<T>>::builder()
                    .density_matrix(denmat)
                    .bao(self.bao())
                    .mol(self.mol())
                    .complex_symmetric(self.complex_symmetric())
                    .threshold(self.threshold())
                    .build()
                    .map_err(|err| format_err!(err))
            }
            SpinConstraint::Generalised(nspins, _) => {
                let full_denmat = einsum(
                    "m,n->mn",
                    &[
                        &self.coefficients.view(),
                        &self.coefficients.map(Complex::conj).view(),
                    ],
                )
                .expect("Unable to construct a density matrix from the coefficient matrix.")
                .into_dimensionality::<Ix2>()
                .expect("Unable to convert the resultant density matrix to two dimensions.");
                let nspatial = self.bao().n_funcs();
                let denmat = (0..usize::from(nspins)).fold(
                    Array2::<Complex<T>>::zeros((nspatial, nspatial)),
                    |acc, ispin| {
                        acc + full_denmat.slice(s![
                            ispin * nspatial..(ispin + 1) * nspatial,
                            ispin * nspatial..(ispin + 1) * nspatial
                        ])
                    },
                );
                Density::<Complex<T>>::builder()
                    .density_matrix(denmat)
                    .bao(self.bao())
                    .mol(self.mol())
                    .complex_symmetric(self.complex_symmetric())
                    .threshold(self.threshold())
                    .build()
                    .map_err(|err| format_err!(err))
            }
        }
    }
}

impl<'a, T> MolecularOrbital<'a, Complex<T>, SpinOrbitCoupled>
where
    T: Float + FloatConst + Lapack + From<u16>,
    Complex<T>: Lapack,
{
    /// Constructs the total density of the molecular orbital.
    pub fn to_total_density(&'a self) -> Result<Density<'a, Complex<T>>, anyhow::Error> {
        match self.structure_constraint {
            SpinOrbitCoupled::JAdapted(ncomps) => {
                let full_denmat = einsum(
                    "m,n->mn",
                    &[
                        &self.coefficients.view(),
                        &self.coefficients.map(Complex::conj).view(),
                    ],
                )
                .expect("Unable to construct a density matrix from the coefficient matrix.")
                .into_dimensionality::<Ix2>()
                .expect("Unable to convert the resultant density matrix to two dimensions.");
                let nspatial = self.bao().n_funcs();
                let denmat = (0..usize::from(ncomps)).fold(
                    Array2::<Complex<T>>::zeros((nspatial, nspatial)),
                    |acc, icomp| {
                        acc + full_denmat.slice(s![
                            icomp * nspatial..(icomp + 1) * nspatial,
                            icomp * nspatial..(icomp + 1) * nspatial
                        ])
                    },
                );
                Density::<Complex<T>>::builder()
                    .density_matrix(denmat)
                    .bao(self.bao())
                    .mol(self.mol())
                    .complex_symmetric(self.complex_symmetric())
                    .threshold(self.threshold())
                    .build()
                    .map_err(|err| format_err!(err))
            }
        }
    }
}

// =====================
// Trait implementations
// =====================

// ----
// From
// ----
impl<'a, T, SC> From<MolecularOrbital<'a, T, SC>> for MolecularOrbital<'a, Complex<T>, SC>
where
    T: Float + FloatConst + Lapack,
    Complex<T>: Lapack,
    SC: StructureConstraint + Clone,
{
    fn from(value: MolecularOrbital<'a, T, SC>) -> Self {
        MolecularOrbital::<'a, Complex<T>, SC>::builder()
            .coefficients(value.coefficients.map(Complex::from))
            .bao(value.bao)
            .mol(value.mol)
            .structure_constraint(value.structure_constraint)
            .component_index(value.component_index)
            .complex_symmetric(value.complex_symmetric)
            .threshold(value.threshold)
            .build()
            .expect("Unable to construct a complex molecular orbital.")
    }
}

// ---------
// PartialEq
// ---------
impl<'a, T, SC> PartialEq for MolecularOrbital<'a, T, SC>
where
    T: ComplexFloat<Real = f64> + Lapack,
    SC: StructureConstraint + PartialEq,
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
        self.structure_constraint == other.structure_constraint
            && self.component_index == other.component_index
            && self.bao == other.bao
            && self.mol == other.mol
            && coefficients_eq
    }
}

// --
// Eq
// --
impl<'a, T, SC> Eq for MolecularOrbital<'a, T, SC>
where
    T: ComplexFloat<Real = f64> + Lapack,
    SC: StructureConstraint + Eq,
{
}

// -----
// Debug
// -----
impl<'a, T, SC> fmt::Debug for MolecularOrbital<'a, T, SC>
where
    T: fmt::Debug + ComplexFloat + Lapack,
    SC: StructureConstraint + fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "MolecularOrbital[{:?} (spin index {}): coefficient array of length {}]",
            self.structure_constraint,
            self.component_index,
            self.coefficients.len()
        )?;
        Ok(())
    }
}

// -------
// Display
// -------
impl<'a, T, SC> fmt::Display for MolecularOrbital<'a, T, SC>
where
    T: fmt::Display + ComplexFloat + Lapack,
    SC: StructureConstraint + fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "MolecularOrbital[{} (spin index {}): coefficient array of length {}]",
            self.structure_constraint,
            self.component_index,
            self.coefficients.len()
        )?;
        Ok(())
    }
}
