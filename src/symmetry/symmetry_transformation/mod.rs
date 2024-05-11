//! Transformations under symmetry operations.

use std::error::Error;
use std::fmt;

use nalgebra::Vector3;
use ndarray::{Array, Array2, Axis, RemoveAxis};
use num_complex::Complex;
#[cfg(feature = "python")]
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

use crate::angmom::sh_conversion::{sh_cart2r, sh_r2cart};
use crate::angmom::sh_rotation_3d::rlmat;
use crate::angmom::spinor_rotation_3d::dmat_angleaxis;
use crate::basis::ao::{BasisAngularOrder, CartOrder, PureOrder, ShellOrder};
use crate::permutation::{PermutableCollection, Permutation};
use crate::symmetry::symmetry_element::symmetry_operation::{
    SpecialSymmetryTransformation, SymmetryOperation,
};

#[cfg(test)]
#[path = "symmetry_transformation_tests.rs"]
mod symmetry_transformation_tests;

// ================
// Enum definitions
// ================

/// Enumerated type for managing the kind of symmetry transformation on an object.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "python", pyclass)]
pub enum SymmetryTransformationKind {
    /// Spatial-only transformation.
    Spatial,

    /// Spatial-only transformation but with spin-including time reversal.
    SpatialWithSpinTimeReversal,

    /// Spin-only transformation.
    Spin,

    /// Spin-spatial coupled transformation.
    SpinSpatial,
}

impl Default for SymmetryTransformationKind {
    fn default() -> Self {
        SymmetryTransformationKind::Spatial
    }
}

impl fmt::Display for SymmetryTransformationKind {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Spatial => write!(f, "Spatial-only transformation"),
            Self::SpatialWithSpinTimeReversal => write!(
                f,
                "Spatial-only transformation but with spin-including time reversal"
            ),
            Self::Spin => write!(f, "Spin-only transformation"),
            Self::SpinSpatial => write!(f, "Spin-spatial coupled transformation"),
        }
    }
}

// =================
// Trait definitions
// =================

#[derive(Debug, Clone)]
pub struct TransformationError(pub String);

impl fmt::Display for TransformationError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Transformation error: {}", self.0)
    }
}

impl Error for TransformationError {}

/// Trait for spatial unitary transformation. A spatial unitary transformation also permutes
/// off-origin sites.
pub trait SpatialUnitaryTransformable: Clone {
    // ----------------
    // Required methods
    // ----------------
    /// Performs a spatial transformation in-place.
    ///
    /// # Arguments
    ///
    /// * `rmat` - The three-dimensional representation matrix of the transformation in the basis
    /// of coordinate *functions* $`(y, z, x)`$.
    /// * `perm` - An optional permutation describing how any off-origin sites are permuted amongst
    /// each other under the transformation.
    fn transform_spatial_mut(
        &mut self,
        rmat: &Array2<f64>,
        perm: Option<&Permutation<usize>>,
    ) -> Result<&mut Self, TransformationError>;

    // ----------------
    // Provided methods
    // ----------------
    /// Performs a spatial transformation and returns the transformed result.
    ///
    /// # Arguments
    ///
    /// * `rmat` - The three-dimensional representation matrix of the transformation in the basis
    /// of coordinate *functions* $`(y, z, x)`$.
    ///
    /// # Returns
    ///
    /// The transformed result.
    fn transform_spatial(
        &self,
        rmat: &Array2<f64>,
        perm: Option<&Permutation<usize>>,
    ) -> Result<Self, TransformationError> {
        let mut tself = self.clone();
        tself.transform_spatial_mut(rmat, perm)?;
        Ok(tself)
    }
}

/// Trait for spin unitary transformations. A spin unitary transformation has no spatial effects.
pub trait SpinUnitaryTransformable: Clone {
    // ----------------
    // Required methods
    // ----------------
    /// Performs a spin transformation in-place.
    ///
    /// # Arguments
    ///
    /// * `dmat` - The two-dimensional representation matrix of the transformation in the basis of
    /// the $`\{ \alpha, \beta \}`$ spinors (*i.e.* decreasing $`m`$ order).
    fn transform_spin_mut(
        &mut self,
        dmat: &Array2<Complex<f64>>,
    ) -> Result<&mut Self, TransformationError>;

    // ----------------
    // Provided methods
    // ----------------
    /// Performs a spin transformation and returns the transformed result.
    ///
    /// # Arguments
    ///
    /// * `dmat` - The two-dimensional representation matrix of the transformation in the basis of
    /// the $`\{ \alpha, \beta \}`$ spinors (*i.e.* decreasing $`m`$ order).
    ///
    /// # Returns
    ///
    /// The transformed result.
    fn transform_spin(&self, dmat: &Array2<Complex<f64>>) -> Result<Self, TransformationError> {
        let mut tself = self.clone();
        tself.transform_spin_mut(dmat)?;
        Ok(tself)
    }
}

/// Trait for complex-conjugation transformations.
pub trait ComplexConjugationTransformable: Clone {
    // ----------------
    // Required methods
    // ----------------
    /// Performs a complex conjugation in-place.
    fn transform_cc_mut(&mut self) -> Result<&mut Self, TransformationError>;

    // ----------------
    // Provided methods
    // ----------------
    /// Performs a complex conjugation and returns the complex-conjugated result.
    ///
    /// # Returns
    ///
    /// The complex-conjugated result.
    fn transform_cc(&self) -> Result<Self, TransformationError> {
        let mut tself = self.clone();
        tself.transform_cc_mut()?;
        Ok(tself)
    }
}

/// Trait for time-reversal transformations.
///
/// This trait has a blanket implementation for any implementor of the [`SpinUnitaryTransformable`]
/// trait and the [`ComplexConjugationTransformable`] trait together with the
/// [`DefaultTimeReversalTransformable`] marker trait.
pub trait TimeReversalTransformable: ComplexConjugationTransformable {
    // ----------------
    // Required methods
    // ----------------
    /// Performs a time-reversal transformation in-place.
    fn transform_timerev_mut(&mut self) -> Result<&mut Self, TransformationError>;

    // ----------------
    // Provided methods
    // ----------------
    /// Performs a time-reversal transformation and returns the time-reversed result.
    ///
    /// # Returns
    ///
    /// The time-reversed result.
    fn transform_timerev(&self) -> Result<Self, TransformationError> {
        let mut tself = self.clone();
        tself.transform_timerev_mut()?;
        Ok(tself)
    }
}

// ----------------------
// Blanket implementation
// ----------------------

/// Marker trait indicating that the implementing type should get the blanket implementation for
/// [`TimeReversalTransformable`].
pub trait DefaultTimeReversalTransformable {}

impl<T> TimeReversalTransformable for T
where
    T: DefaultTimeReversalTransformable
        + SpinUnitaryTransformable
        + ComplexConjugationTransformable,
{
    /// Performs a time-reversal transformation in-place.
    ///
    /// The default implementation of the time-reversal transformation for any type that implements
    /// [`SpinUnitaryTransformable`] and [`ComplexConjugationTransformable`] is a spin rotation by
    /// $`\pi`$ about the space-fixed $`y`$-axis followed by a complex conjugation.
    fn transform_timerev_mut(&mut self) -> Result<&mut Self, TransformationError> {
        let dmat_y = dmat_angleaxis(std::f64::consts::PI, Vector3::y(), false);
        self.transform_spin_mut(&dmat_y)?.transform_cc_mut()
    }
}

/// Trait for transformations using [`SymmetryOperation`].
pub trait SymmetryTransformable:
    SpatialUnitaryTransformable + SpinUnitaryTransformable + TimeReversalTransformable
{
    // ----------------
    // Required methods
    // ----------------
    /// Determines the permutation of sites (*e.g.* atoms in molecules) due to the action of a
    /// symmetry operation.
    ///
    /// # Arguments
    ///
    /// * `symop` - A symmetry operation.
    ///
    /// # Returns
    ///
    /// The resultant site permutation under the action of `symop`, or an error if no such
    /// permutation can be found.
    fn sym_permute_sites_spatial(
        &self,
        symop: &SymmetryOperation,
    ) -> Result<Permutation<usize>, TransformationError>;

    // ----------------
    // Provided methods
    // ----------------
    /// Performs a spatial transformation according to a specified symmetry operation in-place.
    ///
    /// Note that both $`\mathsf{SO}(3)`$ and $`\mathsf{SU}(2)`$ rotations effect the same spatial
    /// transformation. Also note that, if the transformation contains time reversal, it will be
    /// accompanied by a complex conjugation.
    ///
    /// # Arguments
    ///
    /// * `symop` - A symmetry operation.
    fn sym_transform_spatial_mut(
        &mut self,
        symop: &SymmetryOperation,
    ) -> Result<&mut Self, TransformationError> {
        let rmat = symop.get_3d_spatial_matrix();
        let perm = self.sym_permute_sites_spatial(symop)?;
        self.transform_spatial_mut(&rmat, Some(&perm))
            .map_err(|err| TransformationError(err.to_string()))?;
        if symop.contains_time_reversal() {
            self.transform_cc_mut()
        } else {
            Ok(self)
        }
    }

    /// Performs a spatial transformation according to a specified symmetry operation and returns
    /// the transformed result.
    ///
    /// Note that both $`\mathsf{SO}(3)`$ and $`\mathsf{SU}(2)`$ rotations effect the same spatial
    /// transformation. Also note that, if the transformation contains time reversal, it will be
    /// accompanied by a complex conjugation.
    ///
    /// # Arguments
    ///
    /// * `symop` - A symmetry operation.
    ///
    /// # Returns
    ///
    /// The transformed result.
    fn sym_transform_spatial(
        &self,
        symop: &SymmetryOperation,
    ) -> Result<Self, TransformationError> {
        let mut tself = self.clone();
        tself.sym_transform_spatial_mut(symop)?;
        Ok(tself)
    }

    /// Performs a spatial transformation according to a specified symmetry operation in-place, but
    /// with spin-including time reversal.
    ///
    /// Note that both $`\mathsf{SO}(3)`$ and $`\mathsf{SU}(2)`$ rotations effect the same spatial
    /// transformation. Also note that, if the transformation contains time reversal, it will be
    /// accompanied by a rotation by $`\pi`$ about the space-fixed $`y`$-axis followed by a complex
    /// conjugation.
    ///
    /// # Arguments
    ///
    /// * `symop` - A symmetry operation.
    fn sym_transform_spatial_with_spintimerev_mut(
        &mut self,
        symop: &SymmetryOperation,
    ) -> Result<&mut Self, TransformationError> {
        let rmat = symop.get_3d_spatial_matrix();
        let perm = self.sym_permute_sites_spatial(symop)?;
        self.transform_spatial_mut(&rmat, Some(&perm))
            .map_err(|err| TransformationError(err.to_string()))?;
        if symop.contains_time_reversal() {
            self.transform_timerev_mut()?;
        }
        Ok(self)
    }

    /// Performs a spatial transformation according to a specified symmetry operation but with
    /// spin-including time reversal and returns the transformed result.
    ///
    /// Note that both $`\mathsf{SO}(3)`$ and $`\mathsf{SU}(2)`$ rotations effect the same spatial
    /// transformation. Also note that, if the transformation contains time reversal, it will be
    /// accompanied by a rotation by $`\pi`$ about the space-fixed $`y`$-axis followed by a complex
    /// conjugation.
    ///
    /// # Arguments
    ///
    /// * `symop` - A symmetry operation.
    ///
    /// # Returns
    ///
    /// The transformed result.
    fn sym_transform_spatial_with_spintimerev(
        &self,
        symop: &SymmetryOperation,
    ) -> Result<Self, TransformationError> {
        let mut tself = self.clone();
        tself.sym_transform_spatial_with_spintimerev_mut(symop)?;
        Ok(tself)
    }

    /// Performs a spin transformation according to a specified symmetry operation in-place.
    ///
    /// Note that only $`\mathsf{SU}(2)`$ rotations can effect spin transformations. Also note
    /// that, if the transformation contains a time reversal, the corresponding explicit time
    /// reveral action will also be carried out.
    ///
    /// # Arguments
    ///
    /// * `symop` - A symmetry operation.
    fn sym_transform_spin_mut(
        &mut self,
        symop: &SymmetryOperation,
    ) -> Result<&mut Self, TransformationError> {
        if symop.is_su2() {
            let angle = symop.calc_pole_angle();
            let axis = symop.calc_pole().coords;
            let dmat = if symop.is_su2_class_1() {
                -dmat_angleaxis(angle, axis, false)
            } else {
                dmat_angleaxis(angle, axis, false)
            };
            self.transform_spin_mut(&dmat)?;
        }
        if symop.contains_time_reversal() {
            self.transform_timerev_mut()?;
        }
        Ok(self)
    }

    /// Performs a spin transformation according to a specified symmetry operation and returns the
    /// transformed result.
    ///
    /// Note that only $`\mathsf{SU}(2)`$ rotations can effect spin transformations. Also note
    /// that, if the transformation is antiunitary, it will be accompanied by a time reversal.
    ///
    /// # Arguments
    ///
    /// * `symop` - A symmetry operation.
    ///
    /// # Returns
    ///
    /// The transformed result.
    fn sym_transform_spin(&self, symop: &SymmetryOperation) -> Result<Self, TransformationError> {
        let mut tself = self.clone();
        tself.sym_transform_spin_mut(symop)?;
        Ok(tself)
    }

    /// Performs a coupled spin-spatial transformation according to a specified symmetry operation
    /// in-place.
    ///
    /// Note that only $`\mathsf{SU}(2)`$ rotations can effect spin transformations.
    ///
    /// # Arguments
    ///
    /// * `symop` - A symmetry operation.
    fn sym_transform_spin_spatial_mut(
        &mut self,
        symop: &SymmetryOperation,
    ) -> Result<&mut Self, TransformationError> {
        // We cannot do the following, because each of the two methods carries out its own
        // antiunitary action, so we'd be double-acting the antiunitary action.
        // self.sym_transform_spatial_mut(symop)?
        //     .sym_transform_spin_mut(symop)

        // Spatial
        let rmat = symop.get_3d_spatial_matrix();
        let perm = self.sym_permute_sites_spatial(symop)?;
        self.transform_spatial_mut(&rmat, Some(&perm))
            .map_err(|err| TransformationError(err.to_string()))?;

        // Spin -- only SU(2) rotations can effect spin transformations.
        if symop.is_su2() {
            let angle = symop.calc_pole_angle();
            let axis = symop.calc_pole().coords;
            let dmat = if symop.is_su2_class_1() {
                -dmat_angleaxis(angle, axis, false)
            } else {
                dmat_angleaxis(angle, axis, false)
            };
            self.transform_spin_mut(&dmat)?;
        }

        // Time reversal, if any.
        if symop.contains_time_reversal() {
            self.transform_timerev_mut()?;
        }
        Ok(self)
    }

    /// Performs a coupled spin-spatial transformation according to a specified symmetry operation
    /// and returns the transformed result.
    ///
    /// Note that only $`\mathsf{SU}(2)`$ rotations can effect spin transformations.
    ///
    /// # Arguments
    ///
    /// * `symop` - A symmetry operation.
    ///
    /// # Returns
    ///
    /// The transformed result.
    fn sym_transform_spin_spatial(
        &self,
        symop: &SymmetryOperation,
    ) -> Result<Self, TransformationError> {
        let mut tself = self.clone();
        tself.sym_transform_spin_spatial_mut(symop)?;
        Ok(tself)
    }
}

// =========
// Functions
// =========

/// Permutes the generalised rows of an array along one or more dimensions.
///
/// Each generalised row corresponds to a basis function, and consecutive generalised rows
/// corresponding to basis functions localised on a single atom are grouped together and then
/// permuted according to the permutation of the atoms.
///
/// # Arguments
///
/// * `arr` - A coefficient array of any dimensions.
/// * `atom_perm` - A permutation for the atoms.
/// * `axes` - The dimensions along which the generalised rows are to be permuted. The number of
/// generalised rows along each of these dimensions *must* be equal to the number of functions in
/// the basis.
/// * `bao` - A structure specifying the angular order of the underlying basis.
///
/// # Returns
///
/// The permuted array.
///
/// # Panics
///
/// Panics if the number of generalised rows along any of the dimensions in `axes` does not match
/// the number of functions in the basis, or if the permutation rank does not match the number of
/// atoms in the basis.
pub(crate) fn permute_array_by_atoms<T, D>(
    arr: &Array<T, D>,
    atom_perm: &Permutation<usize>,
    axes: &[Axis],
    bao: &BasisAngularOrder,
) -> Array<T, D>
where
    D: RemoveAxis,
    T: Clone,
{
    assert_eq!(
        atom_perm.rank(),
        bao.n_atoms(),
        "The rank of permutation does not match the number of atoms in the basis."
    );
    let atom_boundary_indices = bao.atom_boundary_indices();
    let permuted_shell_indices: Vec<usize> = atom_perm
        .image()
        .iter()
        .flat_map(|&i| {
            let (shell_min, shell_max) = atom_boundary_indices[i];
            shell_min..shell_max
        })
        .collect();

    let mut r = arr.clone();
    for axis in axes {
        assert_eq!(
            arr.shape()[axis.0],
            bao.n_funcs(),
            "The number of generalised rows along {axis:?} in the given array does not match the number of basis functions, {}.", bao.n_funcs()
        );
        r = r.select(*axis, &permuted_shell_indices);
    }
    r
}

/// Assembles spherical-harmonic rotation matrices for all shells.
///
/// # Arguments
///
/// * `bao` - A structure specifying the angular order of the underlying basis.
/// * `rmat` - The three-dimensional representation matrix of the transformation in the basis
/// of coordinate *functions* $`(y, z, x)`$.
/// * `perm` - An optional permutation describing how any off-origin sites are permuted amongst
/// each other under the transformation.
///
/// # Returns
///
/// A vector of spherical-harmonic rotation matrices, one for each shells in `bao`. Non-standard
/// orderings of functions in shells are taken into account.
pub(crate) fn assemble_sh_rotation_3d_matrices(
    bao: &BasisAngularOrder,
    rmat: &Array2<f64>,
    perm: Option<&Permutation<usize>>,
) -> Result<Vec<Array2<f64>>, anyhow::Error> {
    let pbao = if let Some(p) = perm {
        bao.permute(p)?
    } else {
        bao.clone()
    };
    let mut rls = vec![Array2::<f64>::eye(1), rmat.clone()];
    let lmax = pbao
        .basis_shells()
        .map(|shl| shl.l)
        .max()
        .expect("The maximum angular momentum cannot be found.");
    for l in 2..=lmax {
        let rl = rlmat(
            l,
            rmat,
            rls.last()
                .expect("The representation matrix for the last angular momentum cannot be found."),
        );
        rls.push(rl);
    }

    // All matrices in `rls` are in increasing-m order by default. See the function `rlmat` for
    // the origin of this order. Hence, conversion matrices must also honour this.
    let cart2rss_lex: Vec<Vec<Array2<f64>>> = (0..=lmax)
        .map(|lcart| sh_cart2r(lcart, &CartOrder::lex(lcart), true, PureOrder::increasingm))
        .collect();
    let r2cartss_lex: Vec<Vec<Array2<f64>>> = (0..=lmax)
        .map(|lcart| sh_r2cart(lcart, &CartOrder::lex(lcart), true, PureOrder::increasingm))
        .collect();

    let rmats = pbao.basis_shells()
        .map(|shl| {
            let l = usize::try_from(shl.l).unwrap_or_else(|_| {
                panic!(
                    "Unable to convert the angular momentum order `{}` to `usize`.",
                    shl.l
                );
            });
            let po_il = PureOrder::increasingm(shl.l);
            match &shl.shell_order {
                ShellOrder::Pure(pureorder) => {
                    // Spherical functions.
                    let rl = rls[l].clone();
                    if *pureorder != po_il {
                        // `rl` is in increasing-m order by default. See the function `rlmat` for
                        // the origin of this order.
                        let perm = pureorder
                            .get_perm_of(&po_il)
                            .expect("Unable to obtain the permutation that maps `pureorder` to the increasing order.");
                        rl.select(Axis(0), &perm.image()).select(Axis(1), &perm.image())
                    } else {
                        rl
                    }
                }
                ShellOrder::Cart(cart_order) => {
                    // Cartesian functions. Convert them to real solid harmonics first, then
                    // applying the transformation, then convert back.
                    // The actual Cartesian order will be taken into account.

                    // Perform the conversion using lexicographic order first. This allows for the
                    // conversion matrices to be computed only once in the lexicographic order.
                    let cart2rs = &cart2rss_lex[l];
                    let r2carts = &r2cartss_lex[l];
                    let rl = cart2rs.iter().zip(r2carts.iter()).enumerate().fold(
                        Array2::zeros((cart_order.ncomps(), cart_order.ncomps())),
                        |acc, (i, (xmat, wmat))| {
                            let lpure = l - 2 * i;
                            acc + wmat.dot(&rls[lpure]).dot(xmat)
                        },
                    );
                    let lex_cart_order = CartOrder::lex(shl.l);

                    // Now deal with the actual Cartesian order by permutations.
                    if *cart_order != lex_cart_order {
                        // `rl` is in lexicographic order (because of `wmat` and `xmat`) by default.
                        // Consider a transformation R and its representation matrix D in a
                        // lexicographically-ordered Cartesian basis b collected in a row vector.
                        // Then,
                        //      R b = b D.
                        // If we now permute the basis functions in b by a permutation π, then the
                        // representation matrix for R changes:
                        //      R πb = πb D(π).
                        // To relate D(π) to D, we first note the representation matrix for π, P:
                        //      πb = π b = b P,
                        // which, when acts on a left row vector, permutes its entries normally, but
                        // when acts on a right column vector, permutes its entries inversely.
                        // Then,
                        //      R πb = R b P = b P D(π) => R b = b PD(π)P^(-1).
                        // Thus,
                        //      D(π) = P^(-1)DP,
                        // i.e., to obtain D(π), we permute the rows and columns of D normally
                        // according to π.
                        let perm = lex_cart_order
                            .get_perm_of(cart_order)
                            .unwrap_or_else(
                                || panic!("Unable to find a permutation to map `{lex_cart_order}` to `{cart_order}`.")
                            );
                        rl.select(Axis(0), perm.image())
                            .select(Axis(1), perm.image())
                    } else {
                        rl
                    }
                }
            }
        })
        .collect::<Vec<Array2<f64>>>();
    Ok(rmats)
}
