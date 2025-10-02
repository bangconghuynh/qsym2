use std::{
    collections::HashSet,
    fmt::{Display, LowerExp},
    hash::Hash,
};

use anyhow::{self, ensure, format_err};
use log;
use ndarray::{Array2, ArrayView2, Ix2, ScalarOperand};
use ndarray_linalg::Lapack;
use num::FromPrimitive;
use num_complex::ComplexFloat;

use crate::{
    analysis::{EigenvalueComparisonMode, Overlap},
    angmom::spinor_rotation_3d::StructureConstraint,
    symmetry::{
        symmetry_group::SymmetryGroupProperties,
        symmetry_transformation::{SymmetryTransformable, SymmetryTransformationKind},
    },
    target::{
        determinant::SlaterDeterminant,
        noci::{
            backend::{
                matelem::{OrbitMatrix, hamiltonian::HamiltonianAO, overlap::OverlapAO},
                solver::GeneralisedEigenvalueSolvable,
            },
            basis::OrbitBasis,
            multideterminant::MultiDeterminant,
        },
    },
};

#[cfg(test)]
#[path = "noci_tests.rs"]
mod noci_tests;

/// Trait for solving the NOCI problem using symmetry orbits.
pub trait SymmetryOrbitNOCISolvable<'a, G>
where
    G: SymmetryGroupProperties + Clone,
    Self::NumType: ComplexFloat + Lapack,
    Self::StructureConstraintType: StructureConstraint + Clone + Eq + Display + Hash,
{
    /// Numerical type of the elements of the Slater determinant coefficient matrices.
    type NumType;

    /// Numerical type of the various thresholds for comparison.
    type RealType;

    /// Structure constraint type.
    type StructureConstraintType;

    /// Constructs and solves the NOCI problem using symmetry for a set of origin Slater
    /// determinants.
    ///
    /// Symmetry is used to construct the full NOCI basis from the origin Slater determinants.
    /// Optionally, symmetry is also used to speed up the computation of the NOCI Hamiltonian and
    /// overlap matrices via Cayley tables and group closure.
    ///
    /// # Arguments
    ///
    /// * `origins` - A list of Slater determinants to be used as origins for symmetry orbits, the
    /// concatenation of which constitutes the basis for the NOCI problem.
    /// * `group` - The symmetry group acting on the origins to generate symmetry orbits.
    /// * `symmetry_transform_kind` - The transformation kind dictating how `group` acts on the
    /// origin Slater determinants.
    /// * `use_cayley_table` - Boolean indicating if group closure is to be utilised to speed up
    /// the construction of the orbit matrices.
    /// * `thresh_offdiag` - Threshold for verifying zero off-diagonal elements in matrices that
    /// are expected to be diagonal.
    /// * `thresh_zeroov` - Threshold for determining zero Löwdin overlaps in Löwdin pairing.
    ///
    /// # Returns
    ///
    /// A vector of multi-determinants, each of which is for one NOCI state.
    fn solve_symmetry_orbit_noci(
        &'a self,
        origins: &[&SlaterDeterminant<'a, Self::NumType, Self::StructureConstraintType>],
        group: &'a G,
        symmetry_transform_kind: SymmetryTransformationKind,
        use_cayley_table: bool,
        thresh_offdiag: Self::RealType,
        thresh_zeroov: Self::RealType,
    ) -> Result<
        Vec<
            MultiDeterminant<
                'a,
                Self::NumType,
                OrbitBasis<
                    'a,
                    G,
                    SlaterDeterminant<'a, Self::NumType, Self::StructureConstraintType>,
                >,
                Self::StructureConstraintType,
            >,
        >,
        anyhow::Error,
    >;
}

impl<'a, G, T, SC, F> SymmetryOrbitNOCISolvable<'a, G>
    for (&'a HamiltonianAO<'a, T, SC, F>, &'a OverlapAO<'a, T, SC>)
where
    G: SymmetryGroupProperties + Clone,
    T: ComplexFloat + Lapack + ScalarOperand + FromPrimitive + Sync + Send,
    <T as ComplexFloat>::Real: LowerExp + Sync,
    SC: StructureConstraint + Clone + Eq + Display + Hash + Sync,
    SlaterDeterminant<'a, T, SC>: SymmetryTransformable + Sync,
    for<'c> SlaterDeterminant<'c, T, SC>: Overlap<T, Ix2>,
    for<'c> (&'c ArrayView2<'c, T>, &'c ArrayView2<'c, T>):
        GeneralisedEigenvalueSolvable<NumType = T, RealType = <T as ComplexFloat>::Real>,
    F: Fn(&Array2<T>) -> Result<(Array2<T>, Array2<T>), anyhow::Error> + Clone + Sync,
{
    type NumType = T;

    type RealType = <T as ComplexFloat>::Real;

    type StructureConstraintType = SC;

    fn solve_symmetry_orbit_noci(
        &'a self,
        origins: &[&SlaterDeterminant<'a, Self::NumType, Self::StructureConstraintType>],
        group: &'a G,
        symmetry_transform_kind: SymmetryTransformationKind,
        use_cayley_table: bool,
        thresh_offdiag: Self::RealType,
        thresh_zeroov: Self::RealType,
    ) -> Result<
        Vec<
            MultiDeterminant<
                'a,
                Self::NumType,
                OrbitBasis<
                    'a,
                    G,
                    SlaterDeterminant<'a, Self::NumType, Self::StructureConstraintType>,
                >,
                Self::StructureConstraintType,
            >,
        >,
        anyhow::Error,
    > {
        let complex_symmetric_set = origins
            .iter()
            .map(|det| det.complex_symmetric())
            .collect::<HashSet<bool>>();
        ensure!(
            complex_symmetric_set.len() == 1,
            "Inconsistent complex-symmetric flags across origin determinants."
        );
        let complex_symmetric = complex_symmetric_set.iter().next().ok_or(format_err!(
            "Unable to extract the complex-symmetric flag from origin determinants."
        ))?;

        let (hamiltonian_ao, overlap_ao) = (self.0, self.1);
        let origins_cloned = origins.iter().map(|det| (*det).clone()).collect::<Vec<_>>();
        let orbit_basis = match symmetry_transform_kind {
            SymmetryTransformationKind::Spatial => OrbitBasis::builder()
                .origins(origins_cloned)
                .group(group)
                .action(|g, det| det.sym_transform_spatial(g).map_err(|err| format_err!(err)))
                .build()?,
            SymmetryTransformationKind::SpatialWithSpinTimeReversal => OrbitBasis::builder()
                .origins(origins_cloned)
                .group(group)
                .action(|g, det| {
                    det.sym_transform_spatial_with_spintimerev(g)
                        .map_err(|err| format_err!(err))
                })
                .build()?,
            SymmetryTransformationKind::Spin => OrbitBasis::builder()
                .origins(origins_cloned)
                .group(group)
                .action(|g, det| det.sym_transform_spin(g).map_err(|err| format_err!(err)))
                .build()?,
            SymmetryTransformationKind::SpinSpatial => OrbitBasis::builder()
                .origins(origins_cloned)
                .group(group)
                .action(|g, det| {
                    det.sym_transform_spin_spatial(g)
                        .map_err(|err| format_err!(err))
                })
                .build()?,
        };
        let hmat = hamiltonian_ao.calc_orbit_matrix(
            &orbit_basis,
            use_cayley_table,
            overlap_ao.sao(),
            thresh_offdiag,
            thresh_zeroov,
        )?;
        log::debug!("NOCI Hamiltonian matrix:\n  {hmat:+.8e}");
        let smat = overlap_ao.calc_orbit_matrix(
            &orbit_basis,
            use_cayley_table,
            overlap_ao.sao(),
            thresh_offdiag,
            thresh_zeroov,
        )?;
        log::debug!("NOCI overlap matrix:\n  {smat:+.8e}");
        let noci_res = (&hmat.view(), &smat.view())
            .solve_generalised_eigenvalue_problem_with_canonical_orthogonalisation(
                *complex_symmetric,
                thresh_offdiag,
                thresh_zeroov,
                EigenvalueComparisonMode::Real,
            )?;
        noci_res
            .eigenvalues()
            .iter()
            .zip(noci_res.eigenvectors().columns())
            .map(|(e, c)| {
                MultiDeterminant::builder()
                    .basis(orbit_basis.clone())
                    .coefficients(c.to_owned())
                    .energy(Ok(*e))
                    .threshold(thresh_offdiag)
                    .build()
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(move |err| format_err!(err))
    }
}
