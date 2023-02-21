use derive_builder::Builder;
use ndarray::{concatenate, Array2, Axis};
use num::Complex;

use crate::angmom::spinor_rotation_3d::SpinConstraint;
use crate::aux::ao_basis::BasisAngularOrder;
use crate::aux::molecule::Molecule;
use crate::permutation::{IntoPermutation, PermutableCollection, Permutation};
use crate::symmetry::symmetry_element::SymmetryOperation;
use crate::symmetry::symmetry_transformation::assemble_sh_rotation_3d_matrices;
use crate::symmetry::symmetry_transformation::SymmetryTransformable;

// ==================
// Struct definitions
// ==================

#[derive(Builder, Clone)]
struct Determinant<'a> {
    /// The spin constraint associated with the coefficients describing this determinant.
    spin_constraint: SpinConstraint,

    /// The angular order of the basis functions with respect to which the coefficients are
    /// expressed.
    bao: BasisAngularOrder<'a>,

    /// The associated molecule.
    mol: &'a Molecule,

    /// The coefficients describing this determinant.
    coefficients: Vec<Array2<f64>>,
}

impl<'a> Determinant<'a> {
    fn builder() -> DeterminantBuilder<'a> {
        DeterminantBuilder::default()
    }

    pub fn new(cs: &[Array2<f64>], bao: BasisAngularOrder<'a>, spincons: SpinConstraint) -> Self {
        Self::builder()
            .coefficients(cs.to_vec())
            .bao(bao)
            .spin_constraint(spincons)
            .build()
            .expect("Unable to construct a single determinant structure.")
    }

    pub fn coefficients(&self) -> &Vec<Array2<f64>> {
        &self.coefficients
    }

    pub fn spin_constraint(&self) -> &SpinConstraint {
        &self.spin_constraint
    }

    pub fn bao(&self) -> &BasisAngularOrder {
        &self.bao
    }
}

impl<'a> SymmetryTransformable for Determinant<'a> {
    fn permute_sites(&self, symop: &SymmetryOperation) -> Option<Permutation<usize>> {
        symop.act_permute(self.mol)
    }

    fn transform_spatial_mut(&mut self, rmat: &Array2<f64>, perm: Option<&Permutation<usize>>) {
        let tmats = assemble_sh_rotation_3d_matrices(&self.bao, rmat, perm);
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
                            let indices = (shl_start..shl_end).collect::<Vec<_>>();
                            tmat.dot(&old_coeff.select(Axis(0), &indices)).view()
                        })
                        .collect::<Vec<_>>(),
                )
                .expect("Unable to concatenate the transformed rows for the various shells.")
            })
            .collect::<Vec<Array2<f64>>>();
        self.coefficients = new_coefficients
    }

    /// Performs a spin transformation in-place.
    ///
    /// # Arguments
    ///
    /// * `dmat` - The two-dimensional representation matrix of the transformation in the basis of
    /// the $`\{ \alpha, \beta \}`$ spinors (*i.e.* decreasing $`m`$ order).
    fn transform_spin_mut(&mut self, dmat: &Array2<Complex<f64>>) {}

    /// Performs a complex conjugation in-place.
    fn transform_cc_mut(&mut self) {}
}
