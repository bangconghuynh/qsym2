use ndarray::{concatenate, s, Array2, Axis, LinalgScalar, ScalarOperand};
use ndarray_linalg::types::Lapack;
use num_complex::{Complex, ComplexFloat};

use crate::aux::ao_basis::{BasisAngularOrder, BasisAtom, BasisShell, CartOrder, ShellOrder};
use crate::aux::molecule::Molecule;
use crate::permutation::{IntoPermutation, PermutableCollection, Permutation};
use crate::symmetry::symmetry_element::SymmetryOperation;
use crate::symmetry::symmetry_transformation::{
    assemble_sh_rotation_3d_matrices, permute_array_by_atoms, ComplexConjugationTransformable,
    SpatialUnitaryTransformable, SpinUnitaryTransformable, SymmetryTransformable,
    TimeReversalTransformable, TransformationError,
};
use crate::target::vibration::VibrationalCoordinate;

// ---------------------------
// SpatialUnitaryTransformable
// ---------------------------
impl<'a, T> SpatialUnitaryTransformable for VibrationalCoordinate<'a, T>
where
    T: ComplexFloat + LinalgScalar + ScalarOperand + Copy + Lapack,
    f64: Into<T>,
{
    fn transform_spatial_mut(
        &mut self,
        rmat: &Array2<f64>,
        perm: Option<&Permutation<usize>>,
    ) -> &mut Self {
        let vib_bao = construct_vibration_bao(self.mol);
        let tmats: Vec<Array2<T>> = assemble_sh_rotation_3d_matrices(&vib_bao, rmat, perm)
            .iter()
            .map(|tmat| tmat.map(|&x| x.into()))
            .collect();
        let pbao = if let Some(p) = perm {
            vib_bao.permute(p)
        } else {
            vib_bao.clone()
        };
        let old_coeff = &self.coefficients;
        let p_coeff = if let Some(p) = perm {
            permute_array_by_atoms(old_coeff, p, &[Axis(0)], &vib_bao)
        } else {
            old_coeff.clone()
        };
        let t_p_blocks = pbao
            .shell_boundary_indices()
            .into_iter()
            .zip(tmats.iter())
            .map(|((shl_start, shl_end), tmat)| tmat.dot(&p_coeff.slice(s![shl_start..shl_end])))
            .collect::<Vec<_>>();
        let new_coefficients = concatenate(
            Axis(0),
            &t_p_blocks
                .iter()
                .map(|t_p_block| t_p_block.view())
                .collect::<Vec<_>>(),
        )
        .expect("Unable to concatenate the transformed rows for the various atoms.");
        self.coefficients = new_coefficients;
        self
    }
}

// ------------------------
// SpinUnitaryTransformable
// ------------------------

impl<'a, T> SpinUnitaryTransformable for VibrationalCoordinate<'a, T>
where
    T: ComplexFloat + Lapack,
{
    /// Performs a spin transformation in-place.
    ///
    /// This has no effects on the vibrational coordinate as vibrational coordinates are entirely
    /// spatial.
    fn transform_spin_mut(
        &mut self,
        _dmat: &Array2<Complex<f64>>,
    ) -> Result<&mut Self, TransformationError> {
        Ok(self)
    }
}

// -------------------------------
// ComplexConjugationTransformable
// -------------------------------

impl<'a, T> ComplexConjugationTransformable for VibrationalCoordinate<'a, T>
where
    T: ComplexFloat + Lapack,
{
    /// Performs a complex conjugation in-place.
    fn transform_cc_mut(&mut self) -> &mut Self {
        self.coefficients.mapv_inplace(|x| x.conj());
        self
    }
}

// ---------------------
// SymmetryTransformable
// ---------------------
impl<'a, T> SymmetryTransformable for VibrationalCoordinate<'a, T>
where
    T: ComplexFloat + Lapack,
    VibrationalCoordinate<'a, T>: SpatialUnitaryTransformable + TimeReversalTransformable,
{
    fn sym_permute_sites_spatial(
        &self,
        symop: &SymmetryOperation,
    ) -> Result<Permutation<usize>, TransformationError> {
        symop
            .act_permute(&self.mol.molecule_ordinary_atoms())
            .ok_or(TransformationError(format!(
            "Unable to determine the atom permutation corresponding to the operation `{symop}`."
        )))
    }
}

// ---------
// Functions
// ---------

fn construct_vibration_bao(mol: &Molecule) -> BasisAngularOrder {
    let bsp_c = BasisShell::new(1, ShellOrder::Cart(CartOrder::lex(1)));
    let batms = mol
        .atoms
        .iter()
        .map(|atom| BasisAtom::new(atom, &[bsp_c.clone()]))
        .collect::<Vec<_>>();
    BasisAngularOrder::new(&batms)
}
