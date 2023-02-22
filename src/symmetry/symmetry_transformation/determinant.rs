use std::ops::Mul;

use approx;
use derive_builder::Builder;
use ndarray::{concatenate, s, Array2, Axis, LinalgScalar, ScalarOperand};
use num_complex::Complex;

use crate::angmom::spinor_rotation_3d::SpinConstraint;
use crate::aux::ao_basis::BasisAngularOrder;
use crate::aux::molecule::Molecule;
use crate::permutation::{IntoPermutation, PermutableCollection, Permutation};
use crate::symmetry::symmetry_element::SymmetryOperation;
use crate::symmetry::symmetry_transformation::{
    assemble_sh_rotation_3d_matrices, SpatialUnitaryTransformable, SpinUnitaryTransformable,
    TransformationError,
};

// ==================
// Struct definitions
// ==================

#[derive(Builder, Clone)]
struct Determinant<'a, T> {
    /// The spin constraint associated with the coefficients describing this determinant.
    spin_constraint: SpinConstraint,

    /// The angular order of the basis functions with respect to which the coefficients are
    /// expressed.
    bao: BasisAngularOrder<'a>,

    /// The associated molecule.
    mol: &'a Molecule,

    /// The coefficients describing this determinant.
    coefficients: Vec<Array2<T>>,
}

impl<'a, T> Determinant<'a, T>
where
    T: Clone,
{
    fn builder() -> DeterminantBuilder<'a, T> {
        DeterminantBuilder::default()
    }

    pub fn new(
        cs: &[Array2<T>],
        bao: BasisAngularOrder<'a>,
        mol: &'a Molecule,
        spincons: SpinConstraint,
    ) -> Self {
        let det = Self::builder()
            .coefficients(cs.to_vec())
            .bao(bao)
            .mol(mol)
            .spin_constraint(spincons)
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

    fn verify(&self) -> bool {
        let nbas = self.bao.n_funcs();
        let spincons = match self.spin_constraint {
            SpinConstraint::Restricted(nspins) => {
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
        let natoms = self.mol.atoms.len() == self.bao.n_atoms();
        spincons && natoms
    }
}

impl<'a> Into<Determinant<'a, Complex<f64>>> for Determinant<'a, f64> {
    fn into(self) -> Determinant<'a, Complex<f64>> {
        Determinant::<'a, Complex<f64>>::new(
            &self
                .coefficients
                .into_iter()
                .map(|coeffs| coeffs.map(|x| Complex::from(x)))
                .collect::<Vec<_>>(),
            self.bao,
            self.mol,
            self.spin_constraint,
        )
    }
}

impl<'a, T> SpatialUnitaryTransformable for Determinant<'a, T>
where
    T: LinalgScalar + ScalarOperand + From<f64>,
{
    // fn permute_sites(&self, symop: &SymmetryOperation) -> Option<Permutation<usize>> {
    //     symop.act_permute(self.mol)
    // }

    fn transform_spatial_mut(
        &mut self,
        rmat: &Array2<f64>,
        perm: Option<&Permutation<usize>>,
    ) -> &mut Self {
        let tmats: Vec<Array2<T>> = assemble_sh_rotation_3d_matrices(&self.bao, rmat, perm)
            .iter()
            .map(|tmat| tmat.map(|&x| <T as From<f64>>::from(x)))
            .collect();
        let pbao = if let Some(p) = perm {
            self.bao.permute(p)
        } else {
            self.bao
        };
        let new_coefficients = self
            .coefficients
            .iter()
            .map(|old_coeff| {
                concatenate(
                    Axis(0),
                    &pbao
                        .basis_shells()
                        .zip(pbao.shell_boundary_indices().into_iter())
                        .zip(tmats.iter())
                        .map(|((shl, (shl_start, shl_end)), tmat)| {
                            tmat.dot(&old_coeff.slice(s![shl_start..shl_end, ..]))
                                .view()
                        })
                        .collect::<Vec<_>>(),
                )
                .expect("Unable to concatenate the transformed rows for the various shells.")
            })
            .collect::<Vec<Array2<T>>>();
        self.coefficients = new_coefficients;
        self
    }
}

impl<'a> SpinUnitaryTransformable for Determinant<'a, f64> {
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
        match self.spin_constraint {
            SpinConstraint::Restricted(_) | SpinConstraint::Unrestricted(_) => {
                Err(TransformationError(
                    "Spin transformations can only be performed with generalised spin constraint."
                        .to_string(),
                ))
            }
            SpinConstraint::Generalised(nspins) => {
                let cdmat = dmat.view().split_complex();
                if approx::relative_eq!(
                    cdmat.im.fold(0.0, |acc, x| acc + x.powi(2)).sqrt(),
                    0.0,
                    epsilon = 1e-14,
                    max_relative = 1e-14,
                ) {
                    if nspins != 2 {
                        panic!("Only two-component spinor transformations are supported for now.");
                    }

                    let nspatial = self.bao.n_funcs();

                    let rdmat = cdmat.re;
                    let new_coefficients = self
                        .coefficients
                        .iter()
                        .map(|old_coeff| {
                            let a_coeff = old_coeff.slice(s![0..nspatial, ..]).to_owned();
                            let b_coeff = old_coeff.slice(s![nspatial..2 * nspatial, ..]).to_owned();
                            let t_a_coeff = a_coeff * rdmat[[0, 0]] + b_coeff * rdmat[[0, 1]];
                            let t_b_coeff = a_coeff * rdmat[[1, 0]] + b_coeff * rdmat[[1, 1]];
                            concatenate(Axis(0), &[t_a_coeff.view(), t_b_coeff.view()]).expect(
                                "Unable to concatenate the transformed rows for the various shells.",
                            )
                        })
                        .collect::<Vec<Array2<f64>>>();
                    self.coefficients = new_coefficients;
                    Ok(self)
                } else {
                    Err(TransformationError(
                        "Complex spin transformations can only be performed with complex coefficients."
                            .to_string(),
                    ))
                }
            }
        }
    }
}

impl<'a, T> SpinUnitaryTransformable for Determinant<'a, Complex<T>>
where
    Complex<T>: LinalgScalar + ScalarOperand + Mul<Complex<f64>, Output = Complex<T>>,
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
            SpinConstraint::Restricted(_) | SpinConstraint::Unrestricted(_) => {
                Err(TransformationError(
                    "Spin transformations can only be performed with generalised spin constraint."
                        .to_string(),
                ))
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
                        let t_a_coeff = a_coeff * dmat[[0, 0]] + b_coeff * dmat[[0, 1]];
                        let t_b_coeff = a_coeff * dmat[[1, 0]] + b_coeff * dmat[[1, 1]];
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

///// Performs a complex conjugation in-place.
//fn transform_cc_mut(&mut self) {
//    self.coefficients = self
//        .coefficients
//        .iter()
//        .map(|coeff| coeff.map(|x| x.conj()))
//        .collect();
//}
