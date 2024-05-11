//! Implementation of symmetry transformations for bases for non-orthogonal configuration
//! interaction of Slater determinants.

use std::collections::HashSet;

use ndarray::Array2;
use num_complex::Complex;

use crate::group::GroupProperties;
use crate::permutation::Permutation;
use crate::symmetry::symmetry_element::SymmetryOperation;
use crate::symmetry::symmetry_transformation::{
    ComplexConjugationTransformable, DefaultTimeReversalTransformable, SpatialUnitaryTransformable,
    SpinUnitaryTransformable, SymmetryTransformable, TimeReversalTransformable,
    TransformationError,
};
use crate::target::noci::basis::{EagerBasis, OrbitBasis};

// ~~~~~~~~~~~~~~~~~~~~~~
// Lazy basis from orbits
// ~~~~~~~~~~~~~~~~~~~~~~

// ---------------------------
// SpatialUnitaryTransformable
// ---------------------------
impl<'g, G, I> SpatialUnitaryTransformable for OrbitBasis<'g, G, I>
where
    G: GroupProperties + Clone,
    I: SpatialUnitaryTransformable,
{
    fn transform_spatial_mut(
        &mut self,
        _: &Array2<f64>,
        _: Option<&Permutation<usize>>,
    ) -> Result<&mut Self, TransformationError> {
        Err(TransformationError("Transforming an orbit basis by an arbitrary spatial transformation matrix is not supported.".to_string()))
    }
}

// ------------------------
// SpinUnitaryTransformable
// ------------------------
impl<'g, G, I> SpinUnitaryTransformable for OrbitBasis<'g, G, I>
where
    G: GroupProperties + Clone,
    I: SpinUnitaryTransformable,
{
    fn transform_spin_mut(
        &mut self,
        _: &Array2<Complex<f64>>,
    ) -> Result<&mut Self, TransformationError> {
        Err(TransformationError("Transforming an orbit basis by an arbitrary spin transformation matrix is not supported.".to_string()))
        // self.origins
        //     .iter_mut()
        //     .map(|origin| origin.transform_spin_mut(dmat))
        //     .collect::<Result<Vec<_>, _>>()?;
        // Ok(self)
    }
}

// -------------------------------
// ComplexConjugationTransformable
// -------------------------------

impl<'g, G, I> ComplexConjugationTransformable for OrbitBasis<'g, G, I>
where
    G: GroupProperties + Clone,
    I: ComplexConjugationTransformable,
{
    /// Performs a complex conjugation in-place.
    fn transform_cc_mut(&mut self) -> Result<&mut Self, TransformationError> {
        Err(TransformationError(
            "Transforming an orbit basis by an explicit complex conjugation is not supported."
                .to_string(),
        ))
    }
}

// --------------------------------
// DefaultTimeReversalTransformable
// --------------------------------
impl<'g, G, I> DefaultTimeReversalTransformable for OrbitBasis<'g, G, I>
where
    G: GroupProperties + Clone,
    I: DefaultTimeReversalTransformable,
{
}

// ---------------------
// SymmetryTransformable
// ---------------------
impl<'g, G, I> SymmetryTransformable for OrbitBasis<'g, G, I>
where
    G: GroupProperties<GroupElement = SymmetryOperation> + Clone,
    I: SymmetryTransformable,
    OrbitBasis<'g, G, I>: TimeReversalTransformable,
{
    fn sym_permute_sites_spatial(
        &self,
        symop: &SymmetryOperation,
    ) -> Result<Permutation<usize>, TransformationError> {
        let mut perms = self
            .origins
            .iter()
            .map(|origin| origin.sym_permute_sites_spatial(symop))
            .collect::<Result<HashSet<_>, _>>()?;
        if perms.len() == 1 {
            perms.drain().next().ok_or(TransformationError(
                "Unable to retrieve the site permutation.".to_string(),
            ))
        } else {
            Err(TransformationError(
                "Mismatched site permutations across the origins.".to_string(),
            ))
        }
    }

    // --------------------------
    // Overriden provided methods
    // --------------------------
    fn sym_transform_spatial_mut(
        &mut self,
        symop: &SymmetryOperation,
    ) -> Result<&mut Self, TransformationError> {
        if let Some(prefactor) = self.prefactor.as_mut() {
            *prefactor = symop * prefactor.clone();
        } else {
            self.prefactor = Some(symop.clone());
        }
        Ok(self)
    }

    fn sym_transform_spatial_with_spintimerev_mut(
        &mut self,
        symop: &SymmetryOperation,
    ) -> Result<&mut Self, TransformationError> {
        if let Some(prefactor) = self.prefactor.as_mut() {
            *prefactor = symop * prefactor.clone();
        } else {
            self.prefactor = Some(symop.clone());
        }
        Ok(self)
    }

    fn sym_transform_spin_mut(
        &mut self,
        symop: &SymmetryOperation,
    ) -> Result<&mut Self, TransformationError> {
        if let Some(prefactor) = self.prefactor.as_mut() {
            *prefactor = symop * prefactor.clone();
        } else {
            self.prefactor = Some(symop.clone());
        }
        Ok(self)
    }

    fn sym_transform_spin_spatial_mut(
        &mut self,
        symop: &SymmetryOperation,
    ) -> Result<&mut Self, TransformationError> {
        if let Some(prefactor) = self.prefactor.as_mut() {
            *prefactor = symop * prefactor.clone();
        } else {
            self.prefactor = Some(symop.clone());
        }
        Ok(self)
    }
}

// ~~~~~~~~~~~
// Eager basis
// ~~~~~~~~~~~

// ---------------------------
// SpatialUnitaryTransformable
// ---------------------------
impl<I> SpatialUnitaryTransformable for EagerBasis<I>
where
    I: SpatialUnitaryTransformable,
{
    fn transform_spatial_mut(
        &mut self,
        rmat: &Array2<f64>,
        perm: Option<&Permutation<usize>>,
    ) -> Result<&mut Self, TransformationError> {
        self.elements
            .iter_mut()
            .map(|origin| origin.transform_spatial_mut(rmat, perm))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(self)
    }
}

// ------------------------
// SpinUnitaryTransformable
// ------------------------
impl<I> SpinUnitaryTransformable for EagerBasis<I>
where
    I: SpinUnitaryTransformable,
{
    fn transform_spin_mut(
        &mut self,
        dmat: &Array2<Complex<f64>>,
    ) -> Result<&mut Self, TransformationError> {
        self.elements
            .iter_mut()
            .map(|origin| origin.transform_spin_mut(dmat))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(self)
    }
}

// -------------------------------
// ComplexConjugationTransformable
// -------------------------------

impl<I> ComplexConjugationTransformable for EagerBasis<I>
where
    I: ComplexConjugationTransformable,
{
    /// Performs a complex conjugation in-place.
    fn transform_cc_mut(&mut self) -> Result<&mut Self, TransformationError> {
        for origin in self.elements.iter_mut() {
            origin.transform_cc_mut()?;
        }
        Ok(self)
    }
}

// --------------------------------
// DefaultTimeReversalTransformable
// --------------------------------
impl<I> DefaultTimeReversalTransformable for EagerBasis<I> where
    I: DefaultTimeReversalTransformable + Clone
{
}

// ---------------------
// SymmetryTransformable
// ---------------------
impl<I> SymmetryTransformable for EagerBasis<I>
where
    I: SymmetryTransformable,
    EagerBasis<I>: TimeReversalTransformable,
{
    fn sym_permute_sites_spatial(
        &self,
        symop: &SymmetryOperation,
    ) -> Result<Permutation<usize>, TransformationError> {
        let mut perms = self
            .elements
            .iter()
            .map(|origin| origin.sym_permute_sites_spatial(symop))
            .collect::<Result<HashSet<_>, _>>()?;
        if perms.len() == 1 {
            perms.drain().next().ok_or(TransformationError(
                "Unable to retrieve the site permutation.".to_string(),
            ))
        } else {
            Err(TransformationError(
                "Mismatched site permutations across the elements in the basis.".to_string(),
            ))
        }
    }
}
