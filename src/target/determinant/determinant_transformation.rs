//! Implementation of symmetry transformations for Slater determinants.

use std::ops::Mul;

use approx;
use nalgebra::Vector3;
use ndarray::{array, concatenate, s, Array2, Axis, LinalgScalar, ScalarOperand};
use ndarray_linalg::types::Lapack;
use num_complex::{Complex, ComplexFloat};

use crate::angmom::spinor_rotation_3d::{SpinConstraint, SpinOrbitCoupled, StructureConstraint};
use crate::permutation::{IntoPermutation, PermutableCollection, Permutation};
use crate::symmetry::symmetry_element::symmetry_operation::SpecialSymmetryTransformation;
use crate::symmetry::symmetry_element::{RotationGroup, SymmetryElement, SymmetryOperation, TRROT};
use crate::symmetry::symmetry_element_order::ElementOrder;
use crate::symmetry::symmetry_transformation::{
    assemble_sh_rotation_3d_matrices, assemble_spinor_rotation_matrices, permute_array_by_atoms,
    ComplexConjugationTransformable, DefaultTimeReversalTransformable, SpatialUnitaryTransformable,
    SpinUnitaryTransformable, SymmetryTransformable, TimeReversalTransformable,
    TransformationError,
};
use crate::target::determinant::SlaterDeterminant;

// ======================================
// Uncoupled spin and spatial coordinates
// ======================================

// ---------------------------
// SpatialUnitaryTransformable
// ---------------------------
impl<'a, T> SpatialUnitaryTransformable for SlaterDeterminant<'a, T, SpinConstraint>
where
    T: ComplexFloat + LinalgScalar + ScalarOperand + Copy + Lapack,
    f64: Into<T>,
{
    fn transform_spatial_mut(
        &mut self,
        rmat: &Array2<f64>,
        perm: Option<&Permutation<usize>>,
    ) -> Result<&mut Self, TransformationError> {
        let tmats: Vec<Array2<T>> = assemble_sh_rotation_3d_matrices(self.bao, rmat, perm)
            .map_err(|err| TransformationError(err.to_string()))?
            .iter()
            .map(|tmat| tmat.map(|&x| x.into()))
            .collect();
        let pbao = if let Some(p) = perm {
            self.bao
                .permute(p)
                .map_err(|err| TransformationError(err.to_string()))?
        } else {
            self.bao.clone()
        };
        let new_coefficients = self
            .coefficients
            .iter()
            .map(|old_coeff| match self.structure_constraint {
                SpinConstraint::Restricted(_) | SpinConstraint::Unrestricted(_, _) => {
                    let p_coeff = if let Some(p) = perm {
                        permute_array_by_atoms(old_coeff, p, &[Axis(0)], self.bao)
                    } else {
                        old_coeff.clone()
                    };
                    let t_p_blocks = pbao
                        .shell_boundary_indices()
                        .into_iter()
                        .zip(tmats.iter())
                        .map(|((shl_start, shl_end), tmat)| {
                            tmat.dot(&p_coeff.slice(s![shl_start..shl_end, ..]))
                        })
                        .collect::<Vec<_>>();
                    concatenate(
                        Axis(0),
                        &t_p_blocks.iter().map(|t_p_block| t_p_block.view()).collect::<Vec<_>>(),
                    )
                    .expect("Unable to concatenate the transformed rows for the various shells.")
                }
                SpinConstraint::Generalised(nspins, _) => {
                    let nspatial = self.bao.n_funcs();
                    let t_p_spin_blocks = (0..nspins).map(|ispin| {
                        // Extract spin block ispin.
                        let spin_start = usize::from(ispin) * nspatial;
                        let spin_end = (usize::from(ispin) + 1) * nspatial;
                        let spin_block = old_coeff.slice(s![spin_start..spin_end, ..]).to_owned();

                        // Permute within spin block ispin.
                        let p_spin_block = if let Some(p) = perm {
                            permute_array_by_atoms(&spin_block, p, &[Axis(0)], self.bao)
                        } else {
                            spin_block
                        };

                        // Transform within spin block ispin.
                        let t_p_blocks = pbao
                            .shell_boundary_indices()
                            .into_iter()
                            .zip(tmats.iter())
                            .map(|((shl_start, shl_end), tmat)| {
                                tmat.dot(&p_spin_block.slice(s![shl_start..shl_end, ..]))
                            })
                            .collect::<Vec<_>>();

                        // Concatenate blocks for various shells within spin block ispin.
                        concatenate(
                            Axis(0),
                            &t_p_blocks.iter().map(|t_p_block| t_p_block.view()).collect::<Vec<_>>(),
                        )
                        .expect("Unable to concatenate the transformed rows for the various shells.")
                    }).collect::<Vec<_>>();

                    // Concatenate spin blocks.
                    concatenate(
                        Axis(0),
                        &t_p_spin_blocks
                            .iter()
                            .map(|t_p_spin_block| t_p_spin_block.view())
                            .collect::<Vec<_>>(),
                    )
                    .expect("Unable to concatenate the transformed spin blocks.")
                }
            })
            .collect::<Vec<Array2<T>>>();
        self.coefficients = new_coefficients;
        Ok(self)
    }
}

// ------------------------
// SpinUnitaryTransformable
// ------------------------

// ~~~~~~~~~~~~~~~~~~~~~
// For real determinants
// ~~~~~~~~~~~~~~~~~~~~~

impl<'a> SpinUnitaryTransformable for SlaterDeterminant<'a, f64, SpinConstraint> {
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
            match self.structure_constraint {
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
                SpinConstraint::Unrestricted(nspins, increasingm) => {
                    // Only spin flip possible, so the order of the basis in which `dmat` is
                    // expressed and the order of the spin blocks do not need to match.
                    if nspins != 2 {
                        return Err(TransformationError(
                            "Only two-component spinor transformations are supported for now."
                                .to_string(),
                        ));
                    }
                    let dmat_y = array![[0.0, -1.0], [1.0, 0.0]];
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
                        let new_coefficients = if increasingm {
                            vec![self.coefficients[1].clone(), -self.coefficients[0].clone()]
                        } else {
                            vec![-self.coefficients[1].clone(), self.coefficients[0].clone()]
                        };
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
                        let new_coefficients = if increasingm {
                            vec![-self.coefficients[1].clone(), self.coefficients[0].clone()]
                        } else {
                            vec![self.coefficients[1].clone(), -self.coefficients[0].clone()]
                        };
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
                SpinConstraint::Generalised(nspins, increasingm) => {
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
                            if !increasingm {
                                let a_coeff = old_coeff.slice(s![0..nspatial, ..]).to_owned();
                                let b_coeff = old_coeff.slice(s![nspatial..2 * nspatial, ..]).to_owned();
                                let t_a_coeff = &a_coeff * rdmat[[0, 0]] + &b_coeff * rdmat[[0, 1]];
                                let t_b_coeff = &a_coeff * rdmat[[1, 0]] + &b_coeff * rdmat[[1, 1]];
                                concatenate(Axis(0), &[t_a_coeff.view(), t_b_coeff.view()]).expect(
                                    "Unable to concatenate the transformed rows for the various shells.",
                                )
                            } else {
                                let b_coeff = old_coeff.slice(s![0..nspatial, ..]).to_owned();
                                let a_coeff = old_coeff.slice(s![nspatial..2 * nspatial, ..]).to_owned();
                                let t_a_coeff = &a_coeff * rdmat[[0, 0]] + &b_coeff * rdmat[[0, 1]];
                                let t_b_coeff = &a_coeff * rdmat[[1, 0]] + &b_coeff * rdmat[[1, 1]];
                                concatenate(Axis(0), &[t_b_coeff.view(), t_a_coeff.view()]).expect(
                                    "Unable to concatenate the transformed rows for the various shells.",
                                )
                            }
                        })
                        .collect::<Vec<Array2<f64>>>();
                    self.coefficients = new_coefficients;
                    Ok(self)
                }
            }
        }
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~
// For complex determinants
// ~~~~~~~~~~~~~~~~~~~~~~~~

impl<'a, T> SpinUnitaryTransformable for SlaterDeterminant<'a, Complex<T>, SpinConstraint>
where
    T: Clone,
    Complex<T>: ComplexFloat<Real = T>
        + LinalgScalar
        + ScalarOperand
        + Lapack
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
        match self.structure_constraint {
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
            SpinConstraint::Unrestricted(nspins, increasingm) => {
                // Only spin flip possible, so the order of the basis in which `dmat` is
                // expressed and the order of the spin blocks do not need to match.
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
                    let new_coefficients = if increasingm {
                        vec![self.coefficients[1].clone(), -self.coefficients[0].clone()]
                    } else {
                        vec![-self.coefficients[1].clone(), self.coefficients[0].clone()]
                    };
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
                    let new_coefficients = if increasingm {
                        vec![-self.coefficients[1].clone(), self.coefficients[0].clone()]
                    } else {
                        vec![self.coefficients[1].clone(), -self.coefficients[0].clone()]
                    };
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
            SpinConstraint::Generalised(nspins, increasingm) => {
                if nspins != 2 {
                    panic!("Only two-component spinor transformations are supported for now.");
                }

                let nspatial = self.bao.n_funcs();

                let new_coefficients = self
                    .coefficients
                    .iter()
                    .map(|old_coeff| {
                        if increasingm {
                            let b_coeff = old_coeff.slice(s![0..nspatial, ..]).to_owned();
                            let a_coeff = old_coeff.slice(s![nspatial..2 * nspatial, ..]).to_owned();
                            let t_a_coeff = &a_coeff * dmat[[0, 0]] + &b_coeff * dmat[[0, 1]];
                            let t_b_coeff = &a_coeff * dmat[[1, 0]] + &b_coeff * dmat[[1, 1]];
                            concatenate(Axis(0), &[t_b_coeff.view(), t_a_coeff.view()]).expect(
                                "Unable to concatenate the transformed rows for the various shells.",
                            )
                        } else {
                            let a_coeff = old_coeff.slice(s![0..nspatial, ..]).to_owned();
                            let b_coeff = old_coeff.slice(s![nspatial..2 * nspatial, ..]).to_owned();
                            let t_a_coeff = &a_coeff * dmat[[0, 0]] + &b_coeff * dmat[[0, 1]];
                            let t_b_coeff = &a_coeff * dmat[[1, 0]] + &b_coeff * dmat[[1, 1]];
                            concatenate(Axis(0), &[t_a_coeff.view(), t_b_coeff.view()]).expect(
                                "Unable to concatenate the transformed rows for the various shells.",
                            )
                        }
                    })
                    .collect::<Vec<Array2<Complex<T>>>>();
                self.coefficients = new_coefficients;
                Ok(self)
            }
        }
    }
}

// --------------------------------
// DefaultTimeReversalTransformable
// --------------------------------
impl<'a, T> DefaultTimeReversalTransformable for SlaterDeterminant<'a, T, SpinConstraint> where
    T: ComplexFloat + Lapack
{
}

// ====================================
// Coupled spin and spatial coordinates
// ====================================

// ---------------------------
// SpatialUnitaryTransformable
// ---------------------------
impl<'a, T> SpatialUnitaryTransformable for SlaterDeterminant<'a, T, SpinOrbitCoupled>
where
    T: ComplexFloat + Lapack,
{
    fn transform_spatial_mut(
        &mut self,
        _rmat: &Array2<f64>,
        _perm: Option<&Permutation<usize>>,
    ) -> Result<&mut Self, TransformationError> {
        Err(TransformationError(
            "Unable to apply only spatial transformations to a spin--orbit-coupled determinant."
                .to_string(),
        ))
    }
}

// ------------------------
// SpinUnitaryTransformable
// ------------------------
impl<'a, T> SpinUnitaryTransformable for SlaterDeterminant<'a, T, SpinOrbitCoupled>
where
    T: ComplexFloat + Lapack,
{
    fn transform_spin_mut(
        &mut self,
        _dmat: &Array2<Complex<f64>>,
    ) -> Result<&mut Self, TransformationError> {
        Err(TransformationError(
            "Unable to apply only spin transformations to a spin--orbit-coupled determinant."
                .to_string(),
        ))
    }
}

// -------------------------
// TimeReversalTransformable
// -------------------------
impl<'a> TimeReversalTransformable for SlaterDeterminant<'a, Complex<f64>, SpinOrbitCoupled> {
    fn transform_timerev_mut(&mut self) -> Result<&mut Self, TransformationError> {
        let tc2y_element = SymmetryElement::builder()
            .threshold(1e-12)
            .proper_order(ElementOrder::Int(2))
            .proper_power(1)
            .raw_axis(Vector3::new(0.0, 1.0, 0.0))
            .kind(TRROT)
            .rotation_group(RotationGroup::SU2(true))
            .build()
            .unwrap();
        let tc2y = SymmetryOperation::builder()
            .generating_element(tc2y_element)
            .power(1)
            .build()
            .unwrap();
        let tmats: Vec<Array2<Complex<f64>>> =
            assemble_spinor_rotation_matrices(self.bao, &tc2y, None)
                .map_err(|err| TransformationError(err.to_string()))?
                .iter()
                .map(|tmat| tmat.map(|&x| x.into()))
                .collect();

        let new_coefficients = self
            .coefficients
            .iter()
            .map(|old_coeff| match self.structure_constraint {
                SpinOrbitCoupled::JAdapted(ncomps, _) => {
                    let nfuncs_per_comp = self.bao.n_funcs();
                    let t_comp_blocks = (0..ncomps).map(|icomp| {
                        // Extract component block icomp.
                        let comp_start = usize::from(icomp) * nfuncs_per_comp;
                        let comp_end = (usize::from(icomp) + 1) * nfuncs_per_comp;
                        let comp_block = old_coeff.slice(s![comp_start..comp_end, ..]).to_owned();

                        // Transform within spin block ispin.
                        let t_blocks = self.bao
                            .shell_boundary_indices()
                            .iter()
                            .zip(tmats.iter())
                            .map(|((shl_start, shl_end), tmat)| {
                                tmat.dot(&comp_block.slice(s![*shl_start..*shl_end, ..]))
                            })
                            .collect::<Vec<_>>();

                        // Concatenate blocks for various shells within spin block ispin.
                        concatenate(
                            Axis(0),
                            &t_blocks.iter().map(|t_block| t_block.view()).collect::<Vec<_>>(),
                        )
                        .expect("Unable to concatenate the transformed rows for the various shells.")
                    }).collect::<Vec<_>>();

                    // Concatenate component blocks.
                    concatenate(
                        Axis(0),
                        &t_comp_blocks
                            .iter()
                            .map(|t_spin_block| t_spin_block.view())
                            .collect::<Vec<_>>(),
                    )
                    .expect("Unable to concatenate the transformed component blocks.")
                }
            })
            .collect::<Vec<Array2<Complex<f64>>>>();
        self.coefficients = new_coefficients;
        Ok(self)
    }
}

// ---------------------
// SymmetryTransformable
// ---------------------
impl<'a> SymmetryTransformable for SlaterDeterminant<'a, Complex<f64>, SpinOrbitCoupled> {
    // ----------------
    // Required methods
    // ----------------
    fn sym_permute_sites_spatial(
        &self,
        symop: &SymmetryOperation,
    ) -> Result<Permutation<usize>, TransformationError> {
        if (symop.generating_element.threshold().log10() - self.mol.threshold.log10()).abs() >= 3.0
        {
            log::warn!(
                "Symmetry operation threshold ({:.3e}) and molecule threshold ({:.3e}) \
                differ by more than three orders of magnitudes.",
                symop.generating_element.threshold(),
                self.mol.threshold
            )
        }
        symop
            .act_permute(&self.mol.molecule_ordinary_atoms())
            .ok_or(TransformationError(format!(
            "Unable to determine the atom permutation corresponding to the operation `{symop}`.",
        )))
    }

    // ----------------------------
    // Overwritten provided methods
    // ----------------------------
    fn sym_transform_spin_spatial_mut(
        &mut self,
        symop: &SymmetryOperation,
    ) -> Result<&mut Self, TransformationError> {
        let perm = self.sym_permute_sites_spatial(symop)?;
        let pbao = self
            .bao
            .permute(&perm)
            .map_err(|err| TransformationError(err.to_string()))?;

        let tmats: Vec<Array2<Complex<f64>>> =
            assemble_spinor_rotation_matrices(&pbao, symop, Some(&perm))
                .map_err(|err| TransformationError(err.to_string()))?
                .iter()
                .map(|tmat| tmat.map(|&x| x.into()))
                .collect();

        let new_coefficients = self
            .coefficients
            .iter()
            .map(|old_coeff| match self.structure_constraint {
                SpinOrbitCoupled::JAdapted(ncomps, _) => {
                    let nfuncs_per_comp = self.bao.n_funcs();
                    let t_p_comp_blocks = (0..ncomps).map(|icomp| {
                        // Extract component block icomp.
                        let comp_start = usize::from(icomp) * nfuncs_per_comp;
                        let comp_end = (usize::from(icomp) + 1) * nfuncs_per_comp;
                        let comp_block = old_coeff.slice(s![comp_start..comp_end, ..]).to_owned();

                        // Permute within comp block icomp.
                        let p_comp_block = permute_array_by_atoms(&comp_block, &perm, &[Axis(0)], self.bao);

                        // Transform within comp block icomp.
                        let t_p_blocks = pbao
                            .shell_boundary_indices()
                            .into_iter()
                            .zip(tmats.iter())
                            .map(|((shl_start, shl_end), tmat)| {
                                tmat.dot(&p_comp_block.slice(s![shl_start..shl_end, ..]))
                            })
                            .collect::<Vec<_>>();

                        // Concatenate blocks for various shells within comp block icomp.
                        concatenate(
                            Axis(0),
                            &t_p_blocks.iter().map(|t_p_block| t_p_block.view()).collect::<Vec<_>>(),
                        )
                        .expect("Unable to concatenate the transformed rows for the various shells.")
                    }).collect::<Vec<_>>();

                    // Concatenate comp blocks.
                    concatenate(
                        Axis(0),
                        &t_p_comp_blocks
                            .iter()
                            .map(|t_p_comp_block| t_p_comp_block.view())
                            .collect::<Vec<_>>(),
                    )
                    .expect("Unable to concatenate the transformed comp blocks.")
                }
            })
            .collect::<Vec<Array2<Complex<f64>>>>();
        self.coefficients = new_coefficients;

        // Time reversal, if any.
        if symop.contains_time_reversal() {
            self.transform_timerev_mut()?;
        }
        Ok(self)
    }
}

// =========================
// All structure constraints
// =========================

// -------------------------------
// ComplexConjugationTransformable
// -------------------------------

impl<'a, T, SC> ComplexConjugationTransformable for SlaterDeterminant<'a, T, SC>
where
    T: ComplexFloat + Lapack,
    SC: StructureConstraint + Clone,
{
    /// Performs a complex conjugation in-place.
    fn transform_cc_mut(&mut self) -> Result<&mut Self, TransformationError> {
        self.coefficients
            .iter_mut()
            .for_each(|coeff| coeff.mapv_inplace(|x| x.conj()));
        self.complex_conjugated = !self.complex_conjugated;
        Ok(self)
    }
}
