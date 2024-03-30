//! Non-orthogonal configuration interaction of Slater determinants.

use std::collections::HashSet;
use std::fmt;
use std::iter::Sum;

use anyhow::{self, format_err};
use approx;
use derive_builder::Builder;
use hdf5::Group;
use itertools::structs::Product;
use itertools::Itertools;
use log;
use ndarray::{s, Array1, Array2, Ix2};
use ndarray_einsum_beta::*;
use ndarray_linalg::types::Lapack;
use num_complex::{Complex, ComplexFloat};
use num_traits::float::{Float, FloatConst};

use crate::analysis::Orbit;
use crate::angmom::spinor_rotation_3d::SpinConstraint;
use crate::auxiliary::molecule::Molecule;
use crate::basis::ao::BasisAngularOrder;
use crate::group::GroupProperties;
use crate::symmetry::symmetry_group::SymmetryGroupProperties;
use crate::target::density::{DensitiesOwned, Density};
use crate::target::orbital::MolecularOrbital;

use super::determinant::SlaterDeterminant;

// #[cfg(test)]
// mod determinant_tests;
//
// pub mod determinant_analysis;
// mod determinant_transformation;

// =====
// Basis
// =====

// -----------------
// Trait definitions
// -----------------

/// Trait defining behaviours of a basis consisting of linear-space items.
trait Basis<I> {
    /// Type of the iterator over items in the basis.
    type BasisIter: Iterator<Item = Result<I, anyhow::Error>>;

    /// Returns the number of items in the basis.
    fn n_items(&self) -> usize;

    /// An iterator over items in the basis.
    fn iter(&self) -> Self::BasisIter;
}

// --------------------------------------
// Struct definitions and implementations
// --------------------------------------

// ~~~~~~~~~~~~~~~~~~~~~~
// Lazy basis from orbits
// ~~~~~~~~~~~~~~~~~~~~~~

struct OrbitBasis<'g, 'i, G, I>
where
    G: GroupProperties,
{
    /// The origins from which orbits are generated.
    origins: Vec<&'i I>,

    /// The group acting on the origins to generate orbits, the concatenation of which forms the
    /// basis.
    group: &'g G,

    /// A function defining the action of each group element on the origin.
    action: fn(&G::GroupElement, &I) -> Result<I, anyhow::Error>,
}

impl<'g, 'i, G, I> Basis<I> for OrbitBasis<'g, 'i, G, I>
where
    G: GroupProperties,
{
    type BasisIter = OrbitBasisIterator<'i, G, I>;

    fn n_items(&self) -> usize {
        self.origins.len() * self.group.order()
    }

    fn iter(&self) -> Self::BasisIter {
        OrbitBasisIterator::new(self.group, self.origins.clone(), self.action)
    }
}

/// Lazy iterator for basis constructed from the concatenation of orbits generated from multiple
/// origins.
struct OrbitBasisIterator<'i, G, I>
where
    G: GroupProperties,
{
    /// A mutable iterator over the Cartesian product between the group elements and the origins.
    group_origin_iter: Product<
        <<G as GroupProperties>::ElementCollection as IntoIterator>::IntoIter,
        std::vec::IntoIter<&'i I>,
    >,

    /// A function defining the action of each group element on the origin.
    action: fn(&G::GroupElement, &I) -> Result<I, anyhow::Error>,
}

impl<'i, G, I> OrbitBasisIterator<'i, G, I>
where
    G: GroupProperties,
{
    /// Creates and returns a new orbit basis iterator.
    ///
    /// # Arguments
    ///
    /// * `group` - A group.
    /// * `origins` - A slice of origins.
    /// * `action` - A function or closure defining the action of each group element on the origins.
    ///
    /// # Returns
    ///
    /// An orbit basis iterator.
    fn new(
        group: &G,
        origins: Vec<&'i I>,
        action: fn(&G::GroupElement, &I) -> Result<I, anyhow::Error>,
    ) -> Self {
        Self {
            group_origin_iter: group
                .elements()
                .clone()
                .into_iter()
                .cartesian_product(origins.into_iter()),
            action,
        }
    }
}

impl<'i, G, I> Iterator for OrbitBasisIterator<'i, G, I>
where
    G: GroupProperties,
{
    type Item = Result<I, anyhow::Error>;

    fn next(&mut self) -> Option<Self::Item> {
        self.group_origin_iter
            .next()
            .map(|(op, origin)| (self.action)(&op, origin))
    }
}

// struct BasisIteratorFromOrbit<I> {
// }

// ------------------
// Struct definitions
// ------------------

/// Structure to manage multi-determinantal wavefunctions constructed from orbits of multiple origin
/// determinants.
#[derive(Builder, Clone)]
#[builder(build_fn(validate = "Self::validate"))]
pub struct MultiDeterminant<'a, T, B>
where
    T: ComplexFloat + Lapack,
    B: Basis<SlaterDeterminant<'a, T>>,
{
    /// A boolean indicating if the determinant has been acted on by an antiunitary operation. This
    /// is so that the correct metric can be used during overlap evaluation.
    #[builder(default = "false")]
    complex_conjugated: bool,

    /// The basis of Slater determinants in which this multi-determinantal wavefunction is defined.
    basis: &'a B,

    /// The linear combination coefficients of the elements in the multi-orbit to give this
    /// multi-determinant wavefunction.
    coefficients: Array1<T>,

    /// The energy of this multi-determinantal wavefunction.
    #[builder(
        default = "Err(\"Multi-determinantal wavefunction energy not yet set.\".to_string())"
    )]
    energy: Result<T, String>,

    /// The threshold for comparing wavefunctions.
    threshold: <T as ComplexFloat>::Real,
}

impl<'a, T, B> MultiDeterminantBuilder<'a, T, B>
where
    T: ComplexFloat + Lapack,
    B: Basis<SlaterDeterminant<'a, T>>,
{
    fn validate(&self) -> Result<(), String> {
        let basis = self.basis.ok_or("No basis found.".to_string())?;
        let coefficients = self
            .coefficients
            .ok_or("No coefficients found.".to_string())?;
        let nbasis = basis.n_items() == coefficients.len();
        if !nbasis {
            log::error!(
                "The number of coefficients does not match the number of basis determinants."
            );
        }

        if nbasis {
            Ok(())
        } else {
            Err("Multi-determinant wavefunction validation failed.".to_string())
        }
    }
}

// impl<'a, T> SlaterDeterminant<'a, T>
// where
//     T: ComplexFloat + Clone + Lapack,
// {
//     /// Returns a builder to construct a new [`SlaterDeterminant`].
//     pub fn builder() -> SlaterDeterminantBuilder<'a, T> {
//         SlaterDeterminantBuilder::default()
//     }
//
//     /// Returns the complex-symmetric flag of the determinant.
//     pub fn complex_symmetric(&self) -> bool {
//         self.complex_symmetric
//     }
//
//     /// Returns the complex-conjugated flag of the determinant.
//     pub(crate) fn complex_conjugated(&self) -> bool {
//         self.complex_conjugated
//     }
//
//     /// Returns the spin constraint imposed on the coefficients.
//     pub fn spin_constraint(&self) -> &SpinConstraint {
//         &self.spin_constraint
//     }
//
//     /// Returns the basis angular order information of the basis set in which the coefficients are
//     /// expressed.
//     pub fn bao(&self) -> &BasisAngularOrder {
//         self.bao
//     }
//
//     /// Returns the molecule associated with this Slater determinant.
//     pub fn mol(&self) -> &Molecule {
//         self.mol
//     }
//
//     /// Returns the determinantal energy.
//     pub fn energy(&self) -> Result<&T, &String> {
//         self.energy.as_ref()
//     }
//
//     /// Returns the molecular-orbital energies.
//     pub fn mo_energies(&self) -> Option<&Vec<Array1<T>>> {
//         self.mo_energies.as_ref()
//     }
//
//     /// Returns the occupation patterns of the molecular orbitals.
//     pub fn occupations(&self) -> &Vec<Array1<<T as ComplexFloat>::Real>> {
//         &self.occupations
//     }
//
//     /// Returns a shared reference to a vector of coefficient arrays.
//     pub fn coefficients(&self) -> &Vec<Array2<T>> {
//         &self.coefficients
//     }
//
//     /// Returns the threshold with which determinants are compared.
//     pub fn threshold(&self) -> <T as ComplexFloat>::Real {
//         self.threshold
//     }
//
//     /// Returns the total number of electrons in the determinant.
//     pub fn nelectrons(&self) -> <T as ComplexFloat>::Real
//     where
//         <T as ComplexFloat>::Real: Sum + From<u16>,
//     {
//         match self.spin_constraint {
//             SpinConstraint::Restricted(nspins) => {
//                 <T as ComplexFloat>::Real::from(nspins)
//                     * self
//                         .occupations
//                         .iter()
//                         .map(|occ| occ.iter().copied().sum())
//                         .sum()
//             }
//             SpinConstraint::Unrestricted(_, _) | SpinConstraint::Generalised(_, _) => self
//                 .occupations
//                 .iter()
//                 .map(|occ| occ.iter().copied().sum())
//                 .sum(),
//         }
//     }
//
//     /// Augments the encoding of coefficients in this Slater determinant to that in the
//     /// corresponding generalised spin constraint.
//     ///
//     /// # Returns
//     ///
//     /// The equivalent Slater determinant with the coefficients encoded in the generalised spin
//     /// constraint.
//     pub fn to_generalised(&self) -> Self {
//         match self.spin_constraint {
//             SpinConstraint::Restricted(n) => {
//                 log::debug!(
//                     "Restricted Slater determinant will be augmented to generalised Slater determinant."
//                 );
//                 let nbas = self.bao.n_funcs();
//
//                 let cr = &self.coefficients[0];
//                 let occr = &self.occupations[0];
//                 let norb = cr.ncols();
//                 let mut cg = Array2::<T>::zeros((nbas * usize::from(n), norb * usize::from(n)));
//                 let mut occg = Array1::<<T as ComplexFloat>::Real>::zeros((norb * usize::from(n),));
//                 (0..usize::from(n)).for_each(|i| {
//                     let row_start = nbas * i;
//                     let row_end = nbas * (i + 1);
//                     let col_start = norb * i;
//                     let col_end = norb * (i + 1);
//                     cg.slice_mut(s![row_start..row_end, col_start..col_end])
//                         .assign(cr);
//                     occg.slice_mut(s![col_start..col_end]).assign(occr);
//                 });
//                 let moeg_opt = self.mo_energies.as_ref().map(|moer| {
//                     let mut moeg = Array1::<T>::zeros((norb * usize::from(n),));
//                     (0..usize::from(n)).for_each(|i| {
//                         let col_start = norb * i;
//                         let col_end = norb * (i + 1);
//                         moeg.slice_mut(s![col_start..col_end]).assign(&moer[0]);
//                     });
//                     vec![moeg]
//                 });
//                 Self::builder()
//                     .coefficients(&[cg])
//                     .occupations(&[occg])
//                     .mo_energies(moeg_opt)
//                     .energy(self.energy.clone())
//                     .bao(self.bao)
//                     .mol(self.mol)
//                     .spin_constraint(SpinConstraint::Generalised(n, false))
//                     .complex_symmetric(self.complex_symmetric)
//                     .threshold(self.threshold)
//                     .build()
//                     .expect("Unable to spin-generalise a `SlaterDeterminant`.")
//             }
//             SpinConstraint::Unrestricted(n, increasingm) => {
//                 log::debug!(
//                     "Unrestricted Slater determinant will be augmented to generalised Slater determinant."
//                 );
//                 let nbas = self.bao.n_funcs();
//                 let norb_tot = self.coefficients.iter().map(|c| c.ncols()).sum();
//                 let mut cg = Array2::<T>::zeros((nbas * usize::from(n), norb_tot));
//                 let mut occg = Array1::<<T as ComplexFloat>::Real>::zeros((norb_tot,));
//
//                 let col_boundary_indices = (0..usize::from(n))
//                     .scan(0, |acc, ispin| {
//                         let start_index = *acc;
//                         *acc += self.coefficients[ispin].shape()[1];
//                         Some((start_index, *acc))
//                     })
//                     .collect::<Vec<_>>();
//                 (0..usize::from(n)).for_each(|i| {
//                     let row_start = nbas * i;
//                     let row_end = nbas * (i + 1);
//                     let (col_start, col_end) = col_boundary_indices[i];
//                     cg.slice_mut(s![row_start..row_end, col_start..col_end])
//                         .assign(&self.coefficients[i]);
//                     occg.slice_mut(s![col_start..col_end])
//                         .assign(&self.occupations[i]);
//                 });
//
//                 let moeg_opt = self.mo_energies.as_ref().map(|moer| {
//                     let mut moeg = Array1::<T>::zeros((norb_tot,));
//                     (0..usize::from(n)).for_each(|i| {
//                         let (col_start, col_end) = col_boundary_indices[i];
//                         moeg.slice_mut(s![col_start..col_end]).assign(&moer[i]);
//                     });
//                     vec![moeg]
//                 });
//
//                 Self::builder()
//                     .coefficients(&[cg])
//                     .occupations(&[occg])
//                     .mo_energies(moeg_opt)
//                     .energy(self.energy.clone())
//                     .bao(self.bao)
//                     .mol(self.mol)
//                     .spin_constraint(SpinConstraint::Generalised(n, increasingm))
//                     .complex_symmetric(self.complex_symmetric)
//                     .threshold(self.threshold)
//                     .build()
//                     .expect("Unable to spin-generalise a `SlaterDeterminant`.")
//             }
//             SpinConstraint::Generalised(_, _) => self.clone(),
//         }
//     }
//
//     /// Extracts the molecular orbitals in this Slater determinant.
//     ///
//     /// # Returns
//     ///
//     /// A vector of the molecular orbitals constituting this Slater determinant. In the restricted
//     /// spin constraint, the identical molecular orbitals across different spin spaces are only
//     /// given once. Each molecular orbital does contain an index of the spin space it is in.
//     pub fn to_orbitals(&self) -> Vec<Vec<MolecularOrbital<'a, T>>> {
//         self.coefficients
//             .iter()
//             .enumerate()
//             .map(|(spini, cs_spini)| {
//                 cs_spini
//                     .columns()
//                     .into_iter()
//                     .enumerate()
//                     .map(move |(i, c)| {
//                         MolecularOrbital::builder()
//                             .coefficients(c.to_owned())
//                             .energy(self.mo_energies.as_ref().map(|moes| moes[spini][i]))
//                             .bao(self.bao)
//                             .mol(self.mol)
//                             .spin_constraint(self.spin_constraint.clone())
//                             .spin_index(spini)
//                             .complex_symmetric(self.complex_symmetric)
//                             .threshold(self.threshold)
//                             .build()
//                             .expect("Unable to construct a molecular orbital.")
//                     })
//                     .collect::<Vec<_>>()
//             })
//             .collect::<Vec<_>>()
//     }
// }
//
// impl<'a> SlaterDeterminant<'a, f64> {
//     /// Constructs a vector of real densities, one for each spin space in a Slater determinant.
//     ///
//     /// For restricted and unrestricted spin constraints, spin spaces are well-defined. For
//     /// generalised spin constraints, each spin-space density is constructed from the corresponding
//     /// diagonal block of the overall density matrix.
//     ///
//     /// Occupation numbers are also incorporated in the formation of density matrices.
//     ///
//     /// # Returns
//     ///
//     /// A vector of real densities, one for each spin space.
//     pub fn to_densities(&'a self) -> Result<DensitiesOwned<'a, f64>, anyhow::Error> {
//         let densities = match self.spin_constraint {
//             SpinConstraint::Restricted(_) | SpinConstraint::Unrestricted(_, _) => {
//                 self.coefficients().iter().zip(self.occupations().iter()).map(|(c, o)| {
//                     let denmat = einsum(
//                         "i,mi,ni->mn",
//                         &[&o.view(), &c.view(), &c.view()]
//                     )
//                     .expect("Unable to construct a density matrix from a determinant coefficient matrix.")
//                     .into_dimensionality::<Ix2>()
//                     .expect("Unable to convert the resultant density matrix to two dimensions.");
//                     Density::<f64>::builder()
//                         .density_matrix(denmat)
//                         .bao(self.bao())
//                         .mol(self.mol())
//                         .complex_symmetric(self.complex_symmetric())
//                         .threshold(self.threshold())
//                         .build()
//                     })
//                     .collect::<Result<Vec<_>, _>>()?
//             }
//             SpinConstraint::Generalised(nspins, _) => {
//                 let denmat = einsum(
//                     "i,mi,ni->mn",
//                     &[&self.occupations[0].view(), &self.coefficients[0].view(), &self.coefficients[0].view()]
//                 )
//                 .expect("Unable to construct a density matrix from a determinant coefficient matrix.")
//                 .into_dimensionality::<Ix2>()
//                 .expect("Unable to convert the resultant density matrix to two dimensions.");
//                 let nspatial = self.bao.n_funcs();
//                 (0..usize::from(nspins)).map(|ispin| {
//                     let ispin_denmat = denmat.slice(
//                         s![ispin*nspatial..(ispin + 1)*nspatial, ispin*nspatial..(ispin + 1)*nspatial]
//                     ).to_owned();
//                     Density::<f64>::builder()
//                         .density_matrix(ispin_denmat)
//                         .bao(self.bao())
//                         .mol(self.mol())
//                         .complex_symmetric(self.complex_symmetric())
//                         .threshold(self.threshold())
//                         .build()
//                 }).collect::<Result<Vec<_>, _>>()?
//             }
//         };
//         DensitiesOwned::builder()
//             .spin_constraint(self.spin_constraint.clone())
//             .densities(densities)
//             .build()
//             .map_err(|err| format_err!(err))
//     }
// }
//
// impl<'a, T> SlaterDeterminant<'a, Complex<T>>
// where
//     T: Float + FloatConst + Lapack,
//     Complex<T>: Lapack,
// {
//     /// Constructs a vector of complex densities, one from each coefficient matrix in a Slater
//     /// determinant.
//     ///
//     /// For restricted and unrestricted spin constraints, spin spaces are well-defined. For
//     /// generalised spin constraints, each spin-space density is constructed from the corresponding
//     /// diagonal block of the overall density matrix.
//     ///
//     /// Occupation numbers are also incorporated in the formation of density matrices.
//     ///
//     /// # Arguments
//     ///
//     /// * `sd` - A Slater determinant.
//     ///
//     /// # Returns
//     ///
//     /// A vector of complex densities, one for each spin space.
//     pub fn to_densities(&'a self) -> Result<DensitiesOwned<'a, Complex<T>>, anyhow::Error> {
//         let densities = match self.spin_constraint {
//             SpinConstraint::Restricted(_) | SpinConstraint::Unrestricted(_, _) => {
//                 self.coefficients().iter().zip(self.occupations().iter()).map(|(c, o)| {
//                     let denmat = einsum(
//                         "i,mi,ni->mn",
//                         &[&o.map(Complex::<T>::from).view(), &c.view(), &c.map(Complex::conj).view()]
//                     )
//                     .expect("Unable to construct a density matrix from a determinant coefficient matrix.")
//                     .into_dimensionality::<Ix2>()
//                     .expect("Unable to convert the resultant density matrix to two dimensions.");
//                     Density::<Complex<T>>::builder()
//                         .density_matrix(denmat)
//                         .bao(self.bao())
//                         .mol(self.mol())
//                         .complex_symmetric(self.complex_symmetric())
//                         .threshold(self.threshold())
//                         .build()
//                     })
//                     .collect::<Result<Vec<_>, _>>()?
//             }
//             SpinConstraint::Generalised(nspins, _) => {
//                 let denmat = einsum(
//                     "i,mi,ni->mn",
//                     &[
//                         &self.occupations[0].map(Complex::<T>::from).view(),
//                         &self.coefficients[0].view(),
//                         &self.coefficients[0].map(Complex::conj).view()
//                     ]
//                 )
//                 .expect("Unable to construct a density matrix from a determinant coefficient matrix.")
//                 .into_dimensionality::<Ix2>()
//                 .expect("Unable to convert the resultant density matrix to two dimensions.");
//                 let nspatial = self.bao.n_funcs();
//                 (0..usize::from(nspins)).map(|ispin| {
//                     let ispin_denmat = denmat.slice(
//                         s![ispin*nspatial..(ispin + 1)*nspatial, ispin*nspatial..(ispin + 1)*nspatial]
//                     ).to_owned();
//                     Density::<Complex<T>>::builder()
//                         .density_matrix(ispin_denmat)
//                         .bao(self.bao())
//                         .mol(self.mol())
//                         .complex_symmetric(self.complex_symmetric())
//                         .threshold(self.threshold())
//                         .build()
//                 }).collect::<Result<Vec<_>, _>>()?
//             }
//         };
//         DensitiesOwned::builder()
//             .spin_constraint(self.spin_constraint.clone())
//             .densities(densities)
//             .build()
//             .map_err(|err| format_err!(err))
//     }
// }
//
// // =====================
// // Trait implementations
// // =====================
//
// // ----
// // From
// // ----
// impl<'a, T> From<SlaterDeterminant<'a, T>> for SlaterDeterminant<'a, Complex<T>>
// where
//     T: Float + FloatConst + Lapack,
//     Complex<T>: Lapack,
// {
//     fn from(value: SlaterDeterminant<'a, T>) -> Self {
//         SlaterDeterminant::<'a, Complex<T>>::builder()
//             .coefficients(
//                 &value
//                     .coefficients
//                     .into_iter()
//                     .map(|coeffs| coeffs.map(Complex::from))
//                     .collect::<Vec<_>>(),
//             )
//             .occupations(&value.occupations)
//             .mo_energies(value.mo_energies.map(|moes| {
//                 moes.iter()
//                     .map(|moe| moe.map(Complex::from))
//                     .collect::<Vec<_>>()
//             }))
//             .bao(value.bao)
//             .mol(value.mol)
//             .spin_constraint(value.spin_constraint)
//             .complex_symmetric(value.complex_symmetric)
//             .threshold(value.threshold)
//             .build()
//             .expect("Unable to complexify a `SlaterDeterminant`.")
//     }
// }
//
// // ---------
// // PartialEq
// // ---------
// impl<'a, T> PartialEq for SlaterDeterminant<'a, T>
// where
//     T: ComplexFloat<Real = f64> + Lapack,
// {
//     fn eq(&self, other: &Self) -> bool {
//         let thresh = (self.threshold * other.threshold).sqrt();
//         let coefficients_eq =
//             self.coefficients.len() == other.coefficients.len()
//                 && self.coefficients.iter().zip(other.coefficients.iter()).all(
//                     |(scoeffs, ocoeffs)| {
//                         approx::relative_eq!(
//                             (scoeffs - ocoeffs)
//                                 .map(|x| ComplexFloat::abs(*x).powi(2))
//                                 .sum()
//                                 .sqrt(),
//                             0.0,
//                             epsilon = thresh,
//                             max_relative = thresh,
//                         )
//                     },
//                 );
//         let occs_eq = self.occupations.len() == other.occupations.len()
//             && self
//                 .occupations
//                 .iter()
//                 .zip(other.occupations.iter())
//                 .all(|(soccs, ooccs)| {
//                     approx::relative_eq!(
//                         (soccs - ooccs).map(|x| x.abs().powi(2)).sum().sqrt(),
//                         0.0,
//                         epsilon = thresh,
//                         max_relative = thresh,
//                     )
//                 });
//         self.spin_constraint == other.spin_constraint
//             && self.bao == other.bao
//             && self.mol == other.mol
//             && coefficients_eq
//             && occs_eq
//     }
// }
//
// // --
// // Eq
// // --
// impl<'a, T> Eq for SlaterDeterminant<'a, T> where T: ComplexFloat<Real = f64> + Lapack {}
//
// // -----
// // Debug
// // -----
// impl<'a, T> fmt::Debug for SlaterDeterminant<'a, T>
// where
//     T: fmt::Debug + ComplexFloat + Lapack,
//     <T as ComplexFloat>::Real: Sum + From<u16> + fmt::Debug,
// {
//     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
//         write!(
//             f,
//             "SlaterDeterminant[{:?}: {:?} electrons, {} coefficient {} of dimensions {}]",
//             self.spin_constraint,
//             self.nelectrons(),
//             self.coefficients.len(),
//             if self.coefficients.len() != 1 {
//                 "matrices"
//             } else {
//                 "matrix"
//             },
//             self.coefficients
//                 .iter()
//                 .map(|coeff| coeff
//                     .shape()
//                     .iter()
//                     .map(|x| x.to_string())
//                     .collect::<Vec<_>>()
//                     .join("×"))
//                 .collect::<Vec<_>>()
//                 .join(", ")
//         )?;
//         Ok(())
//     }
// }
//
// // -------
// // Display
// // -------
// impl<'a, T> fmt::Display for SlaterDeterminant<'a, T>
// where
//     T: fmt::Display + ComplexFloat + Lapack,
//     <T as ComplexFloat>::Real: Sum + From<u16> + fmt::Display,
// {
//     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
//         write!(
//             f,
//             "SlaterDeterminant[{:?}: {} electrons, {} coefficient {} of dimensions {}]",
//             self.spin_constraint,
//             self.nelectrons(),
//             self.coefficients.len(),
//             if self.coefficients.len() != 1 {
//                 "matrices"
//             } else {
//                 "matrix"
//             },
//             self.coefficients
//                 .iter()
//                 .map(|coeff| coeff
//                     .shape()
//                     .iter()
//                     .map(|x| x.to_string())
//                     .collect::<Vec<_>>()
//                     .join("×"))
//                 .collect::<Vec<_>>()
//                 .join(", ")
//         )?;
//         Ok(())
//     }
// }
