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
use crate::target::orbital::MolecularOrbital;

#[cfg(test)]
mod determinant_tests;

mod determinant_analysis;
mod determinant_transformation;

// ==================
// Struct definitions
// ==================

/// A structure to manage single-determinantal wavefunctions.
#[derive(Builder, Clone)]
#[builder(build_fn(validate = "Self::validate"))]
pub struct SlaterDeterminant<'a, T>
where
    T: ComplexFloat + Lapack,
{
    /// The spin constraint associated with the coefficients describing this determinant.
    spin_constraint: SpinConstraint,

    /// The angular order of the basis functions with respect to which the coefficients are
    /// expressed.
    bao: &'a BasisAngularOrder<'a>,

    /// A boolean indicating if inner products involving this determinant should be the
    /// complex-symmetric bilinear form, rather than the conventional Hermitian sesquilinear form.
    complex_symmetric: bool,

    /// The associated molecule.
    mol: &'a Molecule,

    /// The coefficients describing this determinant.
    #[builder(setter(custom))]
    coefficients: Vec<Array2<T>>,

    /// The occupation patterns of the molecular orbitals in [`Self::coefficients`].
    #[builder(setter(custom))]
    occupations: Vec<Array1<<T as ComplexFloat>::Real>>,

    /// The threshold for comparing determinants.
    threshold: <T as ComplexFloat>::Real,
}

impl<'a, T> SlaterDeterminantBuilder<'a, T>
where
    T: ComplexFloat + Lapack,
{
    fn coefficients(&mut self, cs: &[Array2<T>]) -> &mut Self {
        self.coefficients = Some(cs.to_vec());
        self
    }

    fn occupations(&mut self, occs: &[Array1<<T as ComplexFloat>::Real>]) -> &mut Self {
        self.occupations = Some(occs.to_vec());
        self
    }

    fn validate(&self) -> Result<(), String> {
        let bao = self
            .bao
            .ok_or("No `BasisAngularOrder` found.".to_string())?;
        let nbas = bao.n_funcs();
        let coefficients = self
            .coefficients
            .as_ref()
            .ok_or("No coefficients found.".to_string())?;
        let spincons = match self
            .spin_constraint
            .as_ref()
            .ok_or("No spin constraint found.".to_string())?
        {
            SpinConstraint::Restricted(_) => {
                coefficients.len() == 1 && coefficients[0].shape()[0] == nbas
            }
            SpinConstraint::Unrestricted(nspins, _) => {
                coefficients.len() == usize::from(*nspins)
                    && coefficients.iter().all(|c| c.shape()[0] == nbas)
            }
            SpinConstraint::Generalised(nspins, _) => {
                coefficients.len() == 1
                    && coefficients[0].shape()[0].rem_euclid(nbas) == 0
                    && coefficients[0].shape()[0].div_euclid(nbas) == usize::from(*nspins)
            }
        };
        if !spincons {
            log::error!("The coefficient matrices fail to satisfy the specified spin constraint.");
        }

        let occupations = self
            .occupations
            .as_ref()
            .ok_or("No occupations found.".to_string())?;
        let occs = match self
            .spin_constraint
            .as_ref()
            .ok_or("No spin constraint found.".to_string())?
        {
            SpinConstraint::Restricted(_) => {
                occupations.len() == 1 && occupations[0].shape()[0] == coefficients[0].shape()[1]
            }
            SpinConstraint::Unrestricted(nspins, _) => {
                occupations.len() == usize::from(*nspins)
                    && occupations
                        .iter()
                        .zip(coefficients.iter())
                        .all(|(occs, coeffs)| occs.shape()[0] == coeffs.shape()[1])
            }
            SpinConstraint::Generalised(_, _) => {
                occupations.len() == 1 && occupations[0].shape()[0] == coefficients[0].shape()[1]
            }
        };
        if !occs {
            log::error!("The occupation patterns do not match the coefficient patterns.");
        }

        let mol = self.mol.ok_or("No molecule found.".to_string())?;
        let natoms = mol.atoms.len() == bao.n_atoms();
        if !natoms {
            log::error!("The number of atoms in the molecule does not match the number of local sites in the basis.");
        }
        if spincons && occs && natoms {
            Ok(())
        } else {
            Err("Slater determinant validation failed.".to_string())
        }
    }
}

impl<'a, T> SlaterDeterminant<'a, T>
where
    T: ComplexFloat + Clone + Lapack,
{
    /// Returns a builder to construct a new [`SlaterDeterminant`].
    pub fn builder() -> SlaterDeterminantBuilder<'a, T> {
        SlaterDeterminantBuilder::default()
    }

    pub fn to_generalised(&self) -> Self {
        match self.spin_constraint {
            SpinConstraint::Restricted(n) => {
                let nbas = self.bao.n_funcs();

                let cr = &self.coefficients[0];
                let occr = &self.occupations[0];
                let norb = cr.ncols();
                let mut cg = Array2::<T>::zeros((nbas * usize::from(n), norb * usize::from(n)));
                let mut occg = Array1::<<T as ComplexFloat>::Real>::zeros((norb * usize::from(n),));
                (0..usize::from(n)).for_each(|i| {
                    let row_start = nbas * i;
                    let row_end = nbas * (i + 1);
                    let col_start = norb * i;
                    let col_end = norb * (i + 1);
                    cg.slice_mut(s![row_start..row_end, col_start..col_end])
                        .assign(cr);
                    occg.slice_mut(s![col_start..col_end]).assign(occr);
                });
                Self::builder()
                    .coefficients(&[cg])
                    .occupations(&[occg])
                    .bao(self.bao)
                    .mol(self.mol)
                    .spin_constraint(SpinConstraint::Generalised(n, false))
                    .complex_symmetric(self.complex_symmetric)
                    .threshold(self.threshold)
                    .build()
                    .expect("Unable to spin-generalise a `SlaterDeterminant`.")
            }
            SpinConstraint::Unrestricted(n, increasingm) => {
                let nbas = self.bao.n_funcs();
                let norb_tot = self.coefficients.iter().map(|c| c.ncols()).sum();
                let mut cg = Array2::<T>::zeros((nbas * usize::from(n), norb_tot));
                let mut occg = Array1::<<T as ComplexFloat>::Real>::zeros((norb_tot,));

                let col_boundary_indices = (0..usize::from(n))
                    .scan(0, |acc, ispin| {
                        let start_index = *acc;
                        *acc += self.coefficients[ispin].shape()[1];
                        Some((start_index, *acc))
                    })
                    .collect::<Vec<_>>();
                (0..usize::from(n)).for_each(|i| {
                    let row_start = nbas * i;
                    let row_end = nbas * (i + 1);
                    let (col_start, col_end) = col_boundary_indices[i];
                    cg.slice_mut(s![row_start..row_end, col_start..col_end])
                        .assign(&self.coefficients[i]);
                    occg.slice_mut(s![col_start..col_end])
                        .assign(&self.occupations[i]);
                });
                Self::builder()
                    .coefficients(&[cg])
                    .occupations(&[occg])
                    .bao(self.bao)
                    .mol(self.mol)
                    .spin_constraint(SpinConstraint::Generalised(n, increasingm))
                    .complex_symmetric(self.complex_symmetric)
                    .threshold(self.threshold)
                    .build()
                    .expect("Unable to spin-generalise a `SlaterDeterminant`.")
            }
            SpinConstraint::Generalised(_, _) => self.clone(),
        }
    }

    pub fn to_orbitals(&self) -> Vec<MolecularOrbital<'a, T>> {
        match self.spin_constraint {
            SpinConstraint::Restricted(_) | SpinConstraint::Generalised(_, _) => {
                assert_eq!(self.coefficients.len(), 1);
                self.coefficients[0]
                    .columns()
                    .into_iter()
                    .map(|c| {
                        MolecularOrbital::new(
                            c.to_owned(),
                            self.bao,
                            self.mol,
                            self.spin_constraint.clone(),
                            0,
                            self.complex_symmetric,
                            self.threshold,
                        )
                    })
                    .collect::<Vec<_>>()
            }
            SpinConstraint::Unrestricted(_, _) => self
                .coefficients
                .iter()
                .enumerate()
                .flat_map(|(spini, cs_spini)| {
                    cs_spini.columns().into_iter().map(move |c| {
                        MolecularOrbital::new(
                            c.to_owned(),
                            self.bao,
                            self.mol,
                            self.spin_constraint.clone(),
                            spini,
                            self.complex_symmetric,
                            self.threshold,
                        )
                    })
                })
                .collect::<Vec<_>>(),
        }
    }

    /// Returns a shared reference to a vector of coefficient arrays.
    pub fn coefficients(&self) -> &Vec<Array2<T>> {
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

    /// Returns the total number of electrons in the determinant.
    pub fn nelectrons(&self) -> <T as ComplexFloat>::Real
    where
        <T as ComplexFloat>::Real: Sum + From<u16>,
    {
        match self.spin_constraint {
            SpinConstraint::Restricted(nspins) => {
                <T as ComplexFloat>::Real::from(nspins)
                    * self
                        .occupations
                        .iter()
                        .map(|occ| occ.iter().copied().sum())
                        .sum()
            }
            SpinConstraint::Unrestricted(_, _) | SpinConstraint::Generalised(_, _) => self
                .occupations
                .iter()
                .map(|occ| occ.iter().copied().sum())
                .sum(),
        }
    }
}

// =====================
// Trait implementations
// =====================

// ----
// From
// ----
impl<'a, T> From<SlaterDeterminant<'a, T>> for SlaterDeterminant<'a, Complex<T>>
where
    T: Float + FloatConst + Lapack,
    Complex<T>: Lapack,
{
    fn from(value: SlaterDeterminant<'a, T>) -> Self {
        SlaterDeterminant::<'a, Complex<T>>::builder()
            .coefficients(
                &value
                    .coefficients
                    .into_iter()
                    .map(|coeffs| coeffs.map(|x| Complex::from(x)))
                    .collect::<Vec<_>>(),
            )
            .occupations(&value.occupations)
            .bao(value.bao)
            .mol(value.mol)
            .spin_constraint(value.spin_constraint)
            .complex_symmetric(value.complex_symmetric)
            .threshold(value.threshold)
            .build()
            .expect("Unable to complexify a `SlaterDeterminant`.")
    }
}

// ---------
// PartialEq
// ---------
impl<'a, T> PartialEq for SlaterDeterminant<'a, T>
where
    T: ComplexFloat<Real = f64> + Lapack,
{
    fn eq(&self, other: &Self) -> bool {
        let thresh = (self.threshold * other.threshold).sqrt();
        let coefficients_eq =
            self.coefficients.len() == other.coefficients.len()
                && self.coefficients.iter().zip(other.coefficients.iter()).all(
                    |(scoeffs, ocoeffs)| {
                        approx::relative_eq!(
                            (scoeffs - ocoeffs)
                                .map(|x| ComplexFloat::abs(*x).powi(2))
                                .sum()
                                .sqrt(),
                            0.0,
                            epsilon = thresh,
                            max_relative = thresh,
                        )
                    },
                );
        let occs_eq = self.occupations.len() == other.occupations.len()
            && self
                .occupations
                .iter()
                .zip(other.occupations.iter())
                .all(|(soccs, ooccs)| {
                    approx::relative_eq!(
                        (soccs - ooccs).map(|x| x.abs().powi(2)).sum().sqrt(),
                        0.0,
                        epsilon = thresh,
                        max_relative = thresh,
                    )
                });
        self.spin_constraint == other.spin_constraint
            && self.bao == other.bao
            && self.mol == other.mol
            && coefficients_eq
            && occs_eq
    }
}

// --
// Eq
// --
impl<'a, T> Eq for SlaterDeterminant<'a, T> where T: ComplexFloat<Real = f64> + Lapack {}

// -----
// Debug
// -----
impl<'a, T> fmt::Debug for SlaterDeterminant<'a, T>
where
    T: fmt::Debug + ComplexFloat + Lapack,
    <T as ComplexFloat>::Real: Sum + From<u16> + fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SlaterDeterminant[{:?}: {:?} electrons, {} coefficient {} of dimensions {}]",
            self.spin_constraint,
            self.nelectrons(),
            self.coefficients.len(),
            if self.coefficients.len() != 1 {
                "matrices"
            } else {
                "matrix"
            },
            self.coefficients
                .iter()
                .map(|coeff| coeff
                    .shape()
                    .iter()
                    .map(|x| x.to_string())
                    .collect::<Vec<_>>()
                    .join("×"))
                .collect::<Vec<_>>()
                .join(", ")
        )?;
        Ok(())
    }
}

// -------
// Display
// -------
impl<'a, T> fmt::Display for SlaterDeterminant<'a, T>
where
    T: fmt::Display + ComplexFloat + Lapack,
    <T as ComplexFloat>::Real: Sum + From<u16> + fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SlaterDeterminant[{:?}: {} electrons, {} coefficient {} of dimensions {}]",
            self.spin_constraint,
            self.nelectrons(),
            self.coefficients.len(),
            if self.coefficients.len() != 1 {
                "matrices"
            } else {
                "matrix"
            },
            self.coefficients
                .iter()
                .map(|coeff| coeff
                    .shape()
                    .iter()
                    .map(|x| x.to_string())
                    .collect::<Vec<_>>()
                    .join("×"))
                .collect::<Vec<_>>()
                .join(", ")
        )?;
        Ok(())
    }
}
