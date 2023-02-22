use std::fmt;
use std::iter::Sum;
use std::ops::Mul;

use approx;
use derive_builder::Builder;
use ndarray::{array, concatenate, s, Array1, Array2, Axis, LinalgScalar, ScalarOperand};
use num_complex::{Complex, ComplexFloat};
use num_traits::float::{Float, FloatConst};

use crate::angmom::spinor_rotation_3d::SpinConstraint;
use crate::aux::ao_basis::BasisAngularOrder;
use crate::aux::molecule::Molecule;
use crate::permutation::{IntoPermutation, PermutableCollection, Permutation};
use crate::symmetry::symmetry_element::{SpecialSymmetryTransformation, SymmetryOperation};
use crate::symmetry::symmetry_transformation::{
    assemble_sh_rotation_3d_matrices, permute_array_by_atoms, ComplexConjugationTransformable,
    SpatialUnitaryTransformable, SpinUnitaryTransformable, SymmetryTransformable,
    TimeReversalTransformable, TransformationError,
};

#[cfg(test)]
mod determinant_tests;

// ==================
// Struct definitions
// ==================

#[derive(Builder, Clone)]
struct Determinant<'a, T>
where
    T: ComplexFloat,
{
    /// The spin constraint associated with the coefficients describing this determinant.
    spin_constraint: SpinConstraint,

    /// The angular order of the basis functions with respect to which the coefficients are
    /// expressed.
    bao: &'a BasisAngularOrder<'a>,

    /// The associated molecule.
    mol: &'a Molecule,

    /// The coefficients describing this determinant.
    coefficients: Vec<Array2<T>>,

    /// The occupation patterns of the molecular orbitals in [`Self::coefficients`].
    occupations: Vec<Array1<T::Real>>,

    /// The threshold for comparing determinants.
    threshold: T::Real,
}

impl<'a, T> Determinant<'a, T>
where
    T: ComplexFloat + Clone,
{
    fn builder() -> DeterminantBuilder<'a, T> {
        DeterminantBuilder::default()
    }

    pub fn new(
        cs: &[Array2<T>],
        occs: &[Array1<T::Real>],
        bao: &'a BasisAngularOrder<'a>,
        mol: &'a Molecule,
        spincons: SpinConstraint,
        thresh: T::Real,
    ) -> Self {
        let det = Self::builder()
            .coefficients(cs.to_vec())
            .occupations(occs.to_vec())
            .bao(bao)
            .mol(mol)
            .spin_constraint(spincons)
            .threshold(thresh)
            .build()
            .expect("Unable to construct a single determinant structure.");
        assert!(det.verify());
        det
    }

    pub fn coefficients(&self) -> &Vec<Array2<T>> {
        &self.coefficients
    }

    pub fn spin_constraint(&self) -> &SpinConstraint {
        &self.spin_constraint
    }

    pub fn bao(&self) -> &BasisAngularOrder {
        &self.bao
    }

    pub fn nelectrons(&self) -> T::Real
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
            SpinConstraint::Unrestricted(_) | SpinConstraint::Generalised(_) => self
                .occupations
                .iter()
                .map(|occ| occ.iter().copied().sum())
                .sum(),
        }
    }

    fn verify(&self) -> bool {
        let nbas = self.bao.n_funcs();
        let spincons = match self.spin_constraint {
            SpinConstraint::Restricted(_) => {
                self.coefficients.len() == 1 && self.coefficients[0].shape()[0] == nbas
            }
            SpinConstraint::Unrestricted(nspins) => {
                self.coefficients.len() == usize::from(nspins)
                    && self.coefficients[0].shape()[0] == nbas
            }
            SpinConstraint::Generalised(nspins) => {
                self.coefficients.len() == 1
                    && self.coefficients[0].shape()[0].rem_euclid(nbas) == 0
                    && self.coefficients[0].shape()[0].div_euclid(nbas) == usize::from(nspins)
            }
        };
        if !spincons {
            log::error!("The coefficient matrices fail to satisfy the specified spin constraint.");
        }
        let occs = match self.spin_constraint {
            SpinConstraint::Restricted(nspins) => {
                self.occupations.len() == 1
                    && self.occupations[0].shape()[0] == self.coefficients[0].shape()[1]
            }
            SpinConstraint::Unrestricted(nspins) => {
                self.occupations.len() == usize::from(nspins)
                    && self
                        .occupations
                        .iter()
                        .zip(self.coefficients.iter())
                        .all(|(occs, coeffs)| occs.shape()[0] == coeffs.shape()[1])
            }
            SpinConstraint::Generalised(nspins) => {
                self.occupations.len() == 1
                    && self.occupations[0].shape()[0] == self.coefficients[0].shape()[1]
            }
        };
        if !occs {
            log::error!("The occupation patterns fail to satisfy the specified spin constraint.");
        }
        let natoms = self.mol.atoms.len() == self.bao.n_atoms();
        if !natoms {
            log::error!("The number of atoms in the molecule does not match the number of local sites in the basis.");
        }
        spincons && occs && natoms
    }
}

// =====================
// Trait implementations
// =====================

// ----
// From
// ----
impl<'a, T> From<Determinant<'a, T>> for Determinant<'a, Complex<T>>
where
    T: Float + FloatConst,
{
    fn from(value: Determinant<'a, T>) -> Self {
        Determinant::<'a, Complex<T>>::new(
            &value
                .coefficients
                .into_iter()
                .map(|coeffs| coeffs.map(|x| Complex::from(x)))
                .collect::<Vec<_>>(),
            &value.occupations,
            value.bao,
            value.mol,
            value.spin_constraint,
            value.threshold,
        )
    }
}

// ---------------------------
// SpatialUnitaryTransformable
// ---------------------------
impl<'a, T> SpatialUnitaryTransformable for Determinant<'a, T>
where
    T: ComplexFloat + LinalgScalar + ScalarOperand + Copy,
    f64: Into<T>,
{
    fn transform_spatial_mut(
        &mut self,
        rmat: &Array2<f64>,
        perm: Option<&Permutation<usize>>,
    ) -> &mut Self {
        let tmats: Vec<Array2<T>> = assemble_sh_rotation_3d_matrices(&self.bao, rmat, perm)
            .iter()
            .map(|tmat| tmat.map(|&x| x.into()))
            .collect();
        let pbao = if let Some(p) = perm {
            self.bao.permute(p)
        } else {
            self.bao.clone()
        };
        let new_coefficients = self
            .coefficients
            .iter()
            .map(|old_coeff| {
                let p_coeff = if let Some(p) = perm {
                    permute_array_by_atoms(old_coeff, p, &[Axis(0)], &self.bao)
                } else {
                    old_coeff.clone()
                };
                let blocks = pbao
                    .shell_boundary_indices()
                    .into_iter()
                    .zip(tmats.iter())
                    .map(|((shl_start, shl_end), tmat)| {
                        tmat.dot(&p_coeff.slice(s![shl_start..shl_end, ..]))
                    })
                    .collect::<Vec<_>>();
                concatenate(
                    Axis(0),
                    &blocks.iter().map(|block| block.view()).collect::<Vec<_>>(),
                )
                .expect("Unable to concatenate the transformed rows for the various shells.")
            })
            .collect::<Vec<Array2<T>>>();
        self.coefficients = new_coefficients;
        self
    }
}

// ------------------------
// SpinUnitaryTransformable
// ------------------------
impl<'a> SpinUnitaryTransformable for Determinant<'a, f64>
{
    /// Performs a spin transformation in-place.
    ///
    /// # Arguments
    ///
    /// * `dmat` - The two-dimensional representation matrix of the transformation in the basis of
    /// the $`\{ \alpha, \beta \}`$ spinors (*i.e.* decreasing $`m`$ order).
    fn transform_spin_mut(
        &mut self,
        dmat: &Array2<Complex<f64>>,
    ) -> Result<&mut Self, TransformationError> {
        let cdmat = dmat.view().split_complex();
        if approx::relative_ne!(
            cdmat.im.map(|x| x.powi(2)).sum().sqrt(),
            0.0,
            epsilon = 1e-14,
            max_relative = 1e-14,
        ) {
            log::error!("Spin transformation matrix is complex-valued:\n{dmat}");
            Err(TransformationError(
                "Complex spin transformations cannot be performed with real coefficients."
                    .to_string(),
            ))
        } else {
            let rdmat = cdmat.re.to_owned();
            match self.spin_constraint {
                SpinConstraint::Restricted(_) => {
                    if approx::relative_eq!(
                        (&rdmat - Array2::<f64>::eye(2))
                            .map(|x| x.abs().powi(2))
                            .sum()
                            .sqrt(),
                        0.0,
                        epsilon = 1e-14,
                        max_relative = 1e-14,
                    ) {
                        // Identity spin rotation
                        Ok(self)
                    } else if approx::relative_eq!(
                        (&rdmat + Array2::<f64>::eye(2))
                            .map(|x| x.abs().powi(2))
                            .sum()
                            .sqrt(),
                        0.0,
                        epsilon = 1e-14,
                        max_relative = 1e-14,
                    ) {
                        // Negative identity spin rotation
                        self.coefficients
                            .iter_mut()
                            .for_each(|coeff| *coeff *= -1.0);
                        Ok(self)
                    } else {
                        log::error!("Unsupported spin transformation matrix:\n{}", &rdmat);
                        Err(TransformationError(
                            "Only the identity or negative identity spin transformations are possible with restricted spin constraint."
                                .to_string(),
                        ))
                    }
                }
                SpinConstraint::Unrestricted(nspins) => {
                    if nspins != 2 {
                        return Err(TransformationError(
                            "Only two-component spinor transformations are supported for now."
                                .to_string(),
                        ));
                    }
                    let dmat_y = array![[0.0, -1.0], [1.0, 0.0],];
                    if approx::relative_eq!(
                        (&rdmat - Array2::<f64>::eye(2))
                            .map(|x| x.abs().powi(2))
                            .sum()
                            .sqrt(),
                        0.0,
                        epsilon = 1e-14,
                        max_relative = 1e-14,
                    ) {
                        // Identity spin rotation
                        Ok(self)
                    } else if approx::relative_eq!(
                        (&rdmat + Array2::<f64>::eye(2))
                            .map(|x| x.abs().powi(2))
                            .sum()
                            .sqrt(),
                        0.0,
                        epsilon = 1e-14,
                        max_relative = 1e-14,
                    ) {
                        // Negative identity spin rotation
                        self.coefficients
                            .iter_mut()
                            .for_each(|coeff| *coeff = -coeff.clone());
                        Ok(self)
                    } else if approx::relative_eq!(
                        (&rdmat - &dmat_y).map(|x| x.abs().powi(2)).sum().sqrt(),
                        0.0,
                        epsilon = 1e-14,
                        max_relative = 1e-14,
                    ) {
                        // π-rotation about y-axis, effectively spin-flip
                        let new_coefficients =
                            vec![-self.coefficients[1].clone(), self.coefficients[0].clone()];
                        let new_occupations =
                            vec![self.occupations[1].clone(), self.occupations[0].clone()];
                        self.coefficients = new_coefficients;
                        self.occupations = new_occupations;
                        Ok(self)
                    } else if approx::relative_eq!(
                        (&rdmat + &dmat_y).map(|x| x.abs().powi(2)).sum().sqrt(),
                        0.0,
                        epsilon = 1e-14,
                        max_relative = 1e-14,
                    ) {
                        // 3π-rotation about y-axis, effectively negative spin-flip
                        let new_coefficients =
                            vec![self.coefficients[1].clone(), -self.coefficients[0].clone()];
                        let new_occupations =
                            vec![self.occupations[1].clone(), self.occupations[0].clone()];
                        self.coefficients = new_coefficients;
                        self.occupations = new_occupations;
                        Ok(self)
                    } else {
                        log::error!("Unsupported spin transformation matrix:\n{rdmat}");
                        Err(TransformationError(
                            "Only the identity or πy spin transformations are possible with unrestricted spin constraint."
                                .to_string(),
                        ))
                    }
                }
                SpinConstraint::Generalised(nspins) => {
                    if nspins != 2 {
                        return Err(TransformationError(
                            "Only two-component spinor transformations are supported for now."
                                .to_string(),
                        ));
                    }

                    let nspatial = self.bao.n_funcs();
                    let new_coefficients = self
                        .coefficients
                        .iter()
                        .map(|old_coeff| {
                            let a_coeff = old_coeff.slice(s![0..nspatial, ..]).to_owned();
                            let b_coeff = old_coeff.slice(s![nspatial..2 * nspatial, ..]).to_owned();
                            let t_a_coeff = &a_coeff * rdmat[[0, 0]] + &b_coeff * rdmat[[0, 1]];
                            let t_b_coeff = &a_coeff * rdmat[[1, 0]] + &b_coeff * rdmat[[1, 1]];
                            concatenate(Axis(0), &[t_a_coeff.view(), t_b_coeff.view()]).expect(
                                "Unable to concatenate the transformed rows for the various shells.",
                            )
                        })
                        .collect::<Vec<Array2<f64>>>();
                    self.coefficients = new_coefficients;
                    Ok(self)
                }
            }
        }
    }
}

impl<'a, T> SpinUnitaryTransformable for Determinant<'a, Complex<T>>
where
    T: Clone,
    Complex<T>: ComplexFloat<Real = T>
        + LinalgScalar
        + ScalarOperand
        + Mul<Complex<T>, Output = Complex<T>>
        + Mul<Complex<f64>, Output = Complex<T>>,
{
    /// Performs a spin transformation in-place.
    ///
    /// # Arguments
    ///
    /// * `dmat` - The two-dimensional representation matrix of the transformation in the basis of
    /// the $`\{ \alpha, \beta \}`$ spinors (*i.e.* decreasing $`m`$ order).
    ///
    /// # Panics
    ///
    /// Panics if the spin constraint is not generalised. Spin transformations can only be
    /// performed with generalised spin constraint.
    fn transform_spin_mut(
        &mut self,
        dmat: &Array2<Complex<f64>>,
    ) -> Result<&mut Self, TransformationError> {
        match self.spin_constraint {
            SpinConstraint::Restricted(_) => {
                if approx::relative_eq!(
                    (dmat - Array2::<Complex<f64>>::eye(2))
                        .map(|x| x.abs().powi(2))
                        .sum()
                        .sqrt(),
                    0.0,
                    epsilon = 1e-14,
                    max_relative = 1e-14,
                ) {
                    // Identity spin rotation
                    Ok(self)
                } else if approx::relative_eq!(
                    (dmat + Array2::<Complex<f64>>::eye(2))
                        .map(|x| x.abs().powi(2))
                        .sum()
                        .sqrt(),
                    0.0,
                    epsilon = 1e-14,
                    max_relative = 1e-14,
                ) {
                    // Negative identity spin rotation
                    self.coefficients
                        .iter_mut()
                        .for_each(|coeff| *coeff = -coeff.clone());
                    Ok(self)
                } else {
                    log::error!("Unsupported spin transformation matrix:\n{}", dmat);
                    Err(TransformationError(
                        "Only the identity or negative identity spin transformations are possible with restricted spin constraint."
                            .to_string(),
                    ))
                }
            }
            SpinConstraint::Unrestricted(nspins) => {
                if nspins != 2 {
                    return Err(TransformationError(
                        "Only two-component spinor transformations are supported for now."
                            .to_string(),
                    ));
                }
                let dmat_y = array![
                    [Complex::from(0.0), Complex::from(-1.0)],
                    [Complex::from(1.0), Complex::from(0.0)],
                ];
                if approx::relative_eq!(
                    (dmat - Array2::<Complex<f64>>::eye(2))
                        .map(|x| x.abs().powi(2))
                        .sum()
                        .sqrt(),
                    0.0,
                    epsilon = 1e-14,
                    max_relative = 1e-14,
                ) {
                    // Identity spin rotation
                    Ok(self)
                } else if approx::relative_eq!(
                    (dmat + Array2::<Complex<f64>>::eye(2))
                        .map(|x| x.abs().powi(2))
                        .sum()
                        .sqrt(),
                    0.0,
                    epsilon = 1e-14,
                    max_relative = 1e-14,
                ) {
                    // Negative identity spin rotation
                    self.coefficients
                        .iter_mut()
                        .for_each(|coeff| *coeff = -coeff.clone());
                    Ok(self)
                } else if approx::relative_eq!(
                    (dmat - &dmat_y).map(|x| x.abs().powi(2)).sum().sqrt(),
                    0.0,
                    epsilon = 1e-14,
                    max_relative = 1e-14,
                ) {
                    // π-rotation about y-axis, effectively spin-flip
                    let new_coefficients =
                        vec![-self.coefficients[1].clone(), self.coefficients[0].clone()];
                    let new_occupations =
                        vec![self.occupations[1].clone(), self.occupations[0].clone()];
                    self.coefficients = new_coefficients;
                    self.occupations = new_occupations;
                    Ok(self)
                } else if approx::relative_eq!(
                    (dmat + &dmat_y).map(|x| x.abs().powi(2)).sum().sqrt(),
                    0.0,
                    epsilon = 1e-14,
                    max_relative = 1e-14,
                ) {
                    // 3π-rotation about y-axis, effectively negative spin-flip
                    let new_coefficients =
                        vec![self.coefficients[1].clone(), -self.coefficients[0].clone()];
                    let new_occupations =
                        vec![self.occupations[1].clone(), self.occupations[0].clone()];
                    self.coefficients = new_coefficients;
                    self.occupations = new_occupations;
                    Ok(self)
                } else {
                    log::error!("Unsupported spin transformation matrix:\n{dmat}");
                    Err(TransformationError(
                        "Only the identity or πy spin transformations are possible with unrestricted spin constraint."
                            .to_string(),
                    ))
                }
            }
            SpinConstraint::Generalised(nspins) => {
                if nspins != 2 {
                    panic!("Only two-component spinor transformations are supported for now.");
                }

                let nspatial = self.bao.n_funcs();

                let new_coefficients = self
                    .coefficients
                    .iter()
                    .map(|old_coeff| {
                        let a_coeff = old_coeff.slice(s![0..nspatial, ..]).to_owned();
                        let b_coeff = old_coeff.slice(s![nspatial..2 * nspatial, ..]).to_owned();
                        let t_a_coeff = &a_coeff * dmat[[0, 0]] + &b_coeff * dmat[[0, 1]];
                        let t_b_coeff = &a_coeff * dmat[[1, 0]] + &b_coeff * dmat[[1, 1]];
                        concatenate(Axis(0), &[t_a_coeff.view(), t_b_coeff.view()]).expect(
                            "Unable to concatenate the transformed rows for the various shells.",
                        )
                    })
                    .collect::<Vec<Array2<Complex<T>>>>();
                self.coefficients = new_coefficients;
                Ok(self)
            }
        }
    }
}

// -------------------------------
// ComplexConjugationTransformable
// -------------------------------

impl<'a, T> ComplexConjugationTransformable for Determinant<'a, T>
where
    T: ComplexFloat,
{
    /// Performs a complex conjugation in-place.
    fn transform_cc_mut(&mut self) -> &mut Self {
        self.coefficients
            .iter_mut()
            .for_each(|coeff| coeff.mapv_inplace(|x| x.conj()));
        self
    }
}

// ---------------------
// SymmetryTransformable
// ---------------------
impl<'a, T> SymmetryTransformable for Determinant<'a, T>
where
    T: ComplexFloat,
    Determinant<'a, T>: SpatialUnitaryTransformable + TimeReversalTransformable,
{
    fn permute_sites(
        &self,
        symop: &SymmetryOperation,
    ) -> Result<Permutation<usize>, TransformationError> {
        symop
            .act_permute(self.mol)
            .ok_or(TransformationError(format!(
            "Unable to determine the atom permutation corresponding to the operation `{symop}`."
        )))
    }

    fn transform_mut(
        &mut self,
        symop: &SymmetryOperation,
    ) -> Result<&mut Self, TransformationError> {
        let rmat = symop.get_3d_matrix();
        let perm = self.permute_sites(symop)?;
        self.transform_spatial_mut(&rmat, Some(&perm));
        if symop.is_antiunitary() {
            self.transform_timerev_mut()?;
        }
        Ok(self)
    }
}

// ---------
// PartialEq
// ---------
impl<'a, T> PartialEq for Determinant<'a, T>
where
    T: ComplexFloat<Real = f64>,
{
    fn eq(&self, other: &Self) -> bool {
        let thresh = (self.threshold * other.threshold).sqrt();
        let coefficients_eq =
            self.coefficients.len() == other.coefficients.len()
                && self.coefficients.iter().zip(other.coefficients.iter()).all(
                    |(scoeffs, ocoeffs)| {
                        approx::relative_eq!(
                            (scoeffs - ocoeffs).map(|x| x.abs().powi(2)).sum().sqrt(),
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
impl<'a, T> Eq for Determinant<'a, T> where T: ComplexFloat<Real = f64> {}

// -----
// Debug
// -----
impl<'a, T> fmt::Debug for Determinant<'a, T>
where
    T: fmt::Debug + ComplexFloat,
    <T as ComplexFloat>::Real: Sum + From<u16> + fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Determinant[{:?}: {:?} electrons, {} coefficient {} of dimensions {}]",
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
impl<'a, T> fmt::Display for Determinant<'a, T>
where
    T: fmt::Display + ComplexFloat,
    <T as ComplexFloat>::Real: Sum + From<u16> + fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Determinant[{:?}: {} electrons, {} coefficient {} of dimensions {}]",
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
