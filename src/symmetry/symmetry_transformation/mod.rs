use std::error::Error;
use std::fmt;

use nalgebra::Vector3;
use ndarray::{Array, Array2, Axis, RemoveAxis};
use num_complex::Complex;

use crate::angmom::sh_conversion::{sh_cart2r, sh_r2cart};
use crate::angmom::sh_rotation_3d::rlmat;
use crate::angmom::spinor_rotation_3d::dmat_angleaxis;
use crate::aux::ao_basis::{BasisAngularOrder, CartOrder, ShellOrder};
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

/// An enumerated type for managing the kind of symmetry transformation on an object.
#[derive(Debug, Clone)]
pub enum SymmetryTransformationKind {
    /// Spatial-only transformation.
    Spatial,

    /// Spin-only transformation.
    Spin,

    /// Spin-spatial coupled transformation.
    SpinSpatial,
}

impl fmt::Display for SymmetryTransformationKind {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Spatial => write!(f, "Spatial-only transformation"),
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
        write!(f, "Transformation error: {}.", self.0)
    }
}

impl Error for TransformationError {}

/// A trait for spatial unitary transformation. A spatial unitary transformation also permutes
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
    ) -> &mut Self;

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
    fn transform_spatial(&self, rmat: &Array2<f64>, perm: Option<&Permutation<usize>>) -> Self {
        let mut tself = self.clone();
        tself.transform_spatial_mut(rmat, perm);
        tself
    }
}

/// A trait for spin unitary transformations. A spin unitary transformation has no spatial effects.
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

/// A trait for complex-conjugation transformations.
pub trait ComplexConjugationTransformable: Clone {
    // ----------------
    // Required methods
    // ----------------
    /// Performs a complex conjugation in-place.
    fn transform_cc_mut(&mut self) -> &mut Self;

    // ----------------
    // Provided methods
    // ----------------
    /// Performs a complex conjugation and returns the complex-conjugated result.
    ///
    /// # Returns
    ///
    /// The complex-conjugated result.
    fn transform_cc(&self) -> Self {
        let mut tself = self.clone();
        tself.transform_cc_mut();
        tself
    }
}

/// A trait for time-reversal transformations.
///
/// This trait has a blanket implementation for any implementor of the [`SpinUnitaryTransformable`]
/// trait and the [`ComplexConjugationTransformable`] trait.
pub trait TimeReversalTransformable:
    SpinUnitaryTransformable + ComplexConjugationTransformable
{
    // ----------------
    // Provided methods
    // ----------------
    /// Performs a time-reversal transformation in-place.
    ///
    /// The time-reversal transformation is a spin rotation by $`\pi`$ about the space-fixed
    /// $`y`$-axis followed by a complex conjugation.
    fn transform_timerev_mut(&mut self) -> Result<&mut Self, TransformationError> {
        let dmat_y = dmat_angleaxis(std::f64::consts::PI, Vector3::y(), false);
        self.transform_spin_mut(&dmat_y)?.transform_cc_mut();
        Ok(self)
    }

    /// Performs a time-reversal transformation and returns the time-reversed result.
    ///
    /// The time-reversal transformation is a spin rotation by $`\pi`$ about the space-fixed
    /// $`y`$-axis followed by a complex conjugation.
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

impl<T> TimeReversalTransformable for T where
    T: SpinUnitaryTransformable + ComplexConjugationTransformable
{
}

/// A trait for transformations using [`SymmetryOperation`].
pub trait SymmetryTransformable: SpatialUnitaryTransformable + TimeReversalTransformable {
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
    /// The resultant site permutation under the action of `symop`, or `None` if no such
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
    /// transformation.
    ///
    /// # Arguments
    ///
    /// * `op` - A symmetry operation.
    fn sym_transform_spatial_mut(
        &mut self,
        symop: &SymmetryOperation,
    ) -> Result<&mut Self, TransformationError> {
        let rmat = symop.get_3d_spatial_matrix();
        let perm = self.sym_permute_sites_spatial(symop)?;
        self.transform_spatial_mut(&rmat, Some(&perm));
        Ok(self)
    }

    /// Performs a spatial transformation according to a specified symmetry operation and returns
    /// the transformed result.
    ///
    /// Note that both $`\mathsf{SO}(3)`$ and $`\mathsf{SU}(2)`$ rotations effect the same spatial
    /// transformation.
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

    /// Performs a spin transformation according to a specified symmetry operation in-place.
    ///
    /// Note that only $`\mathsf{SU}(2)`$ rotations can effect spin transformations.
    ///
    /// # Arguments
    ///
    /// * `op` - A symmetry operation.
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
        if symop.is_antiunitary() {
            self.transform_timerev_mut()?;
        }
        Ok(self)
    }

    /// Performs a spin transformation according to a specified symmetry operation and returns the
    /// transformed result.
    ///
    /// Note that only $`\mathsf{SU}(2)`$ rotations can effect spin transformations.
    ///
    /// # Arguments
    ///
    /// * `op` - A symmetry operation.
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
    /// * `op` - A symmetry operation.
    fn sym_transform_spin_spatial_mut(
        &mut self,
        symop: &SymmetryOperation,
    ) -> Result<&mut Self, TransformationError> {
        self.sym_transform_spatial_mut(symop)?
            .sym_transform_spin_mut(symop)
    }

    /// Performs a coupled spin-spatial transformation according to a specified symmetry operation
    /// and returns the transformed result.
    ///
    /// Note that only $`\mathsf{SU}(2)`$ rotations can effect spin transformations.
    ///
    /// # Arguments
    ///
    /// * `op` - A symmetry operation.
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
) -> Vec<Array2<f64>> {
    let pbao = if let Some(p) = perm {
        bao.permute(p)
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

    let cart2rss_lex: Vec<Vec<Array2<f64>>> = (0..=lmax)
        .map(|lcart| sh_cart2r(lcart, &CartOrder::lex(lcart), true, true))
        .collect();
    let r2cartss_lex: Vec<Vec<Array2<f64>>> = (0..=lmax)
        .map(|lcart| sh_r2cart(lcart, &CartOrder::lex(lcart), true, true))
        .collect();

    pbao.basis_shells()
        .map(|shl| {
            let l = usize::try_from(shl.l).unwrap_or_else(|_| {
                panic!(
                    "Unable to concert the angular momentum order `{}` to `usize`.",
                    shl.l
                );
            });
            match &shl.shell_order {
                ShellOrder::Pure(increasingm) => {
                    // Spherical functions.
                    let mut rl = rls[l].clone();
                    if !increasingm {
                        // `rl` is in increasing-m order by default.
                        rl.invert_axis(Axis(0));
                        rl.invert_axis(Axis(1));
                    }
                    rl
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
                        rl.select(Axis(0), &perm.image())
                            .select(Axis(1), &perm.image())
                    } else {
                        rl
                    }
                }
            }
        })
        .collect::<Vec<Array2<f64>>>()
}
