use std::fmt;

use anyhow::format_err;
use derive_builder::Builder;
use nalgebra::Point3;
use ndarray::{Array2, Axis};
use num_traits::ToPrimitive;

use crate::aux::format::nice_bool;
use crate::aux::geometry::Transform;
use crate::aux::molecule::Molecule;
use crate::drivers::{QSym2Driver, QSym2Output};
use crate::drivers::symmetry_group_detection::SymmetryGroupDetectionResult;
use crate::group::{GroupProperties, UnitaryRepresentedGroup};
use crate::permutation::{IntoPermutation, Permutation};
use crate::symmetry::symmetry_core::{PreSymmetry, Symmetry};
use crate::symmetry::symmetry_group::SymmetryGroupProperties;

// ==================
// Struct definitions
// ==================

// ----------
// Parameters
// ----------

/// A structure containing control parameters for symmetry-group detection.
#[derive(Clone, Builder, Debug)]
pub struct MoleculeSymmetrisationParams {
    /// Boolean indicating if a summetry of the located symmetry elements is to be written to the
    /// output file.
    use_magnetic_symmetry: bool,

    target_moi_threshold: f64,

    target_distance_threshold: f64,

    /// The maximum number of symmetrisation iterations.
    #[builder(default = "5")]
    max_iterations: usize,

    #[builder(default = "None")]
    infinite_order_to_finite: Option<u32>,
}

impl MoleculeSymmetrisationParams {
    /// Returns a builder to construct a [`MoleculeSymmetrisationParams`] structure.
    pub fn builder() -> MoleculeSymmetrisationParamsBuilder {
        MoleculeSymmetrisationParamsBuilder::default()
    }
}

impl fmt::Display for MoleculeSymmetrisationParams {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "Use magnetic symmetry: {}",
            nice_bool(self.use_magnetic_symmetry)
        )?;
        writeln!(
            f,
            "Maximum symmetrisation iterations: {}",
            self.max_iterations
        )?;
        writeln!(f, "")?;

        Ok(())
    }
}

// ------
// Result
// ------

/// A structure to contain symmetry-group detection results.
#[derive(Clone, Builder, Debug)]
pub struct MoleculeSymmetrisationResult<'a> {
    /// The control parameters used to obtain this set of result.
    parameters: &'a MoleculeSymmetrisationParams,

    symmetrised_group_detection_result: SymmetryGroupDetectionResult<'a>,
}

impl<'a> MoleculeSymmetrisationResult<'a> {
    fn builder() -> MoleculeSymmetrisationResultBuilder<'a> {
        MoleculeSymmetrisationResultBuilder::default()
    }
}

// ------
// Driver
// ------

/// A driver for symmetry-group detection.
#[derive(Clone, Builder)]
#[builder(build_fn(validate = "Self::validate"))]
pub struct MoleculeSymmetrisationDriver<'a> {
    parameters: &'a MoleculeSymmetrisationParams,

    molecule: &'a Molecule,

    target_symmetry_result: &'a SymmetryGroupDetectionResult<'a>,

    #[builder(default = "None")]
    result: Option<MoleculeSymmetrisationResult<'a>>,
}

impl<'a> MoleculeSymmetrisationDriverBuilder<'a> {
    fn validate(&self) -> Result<(), String> {
        let params = self
            .parameters
            .ok_or("No molecule symmetrisation parameters found.".to_string())?;
        let sym_res = self
            .target_symmetry_result
            .ok_or("No target symmetry group result for symmetrisation found.".to_string())?;
        let sym = if params.use_magnetic_symmetry {
            sym_res
                .magnetic_symmetry
                .as_ref()
                .ok_or("Magnetic symmetry requested as symmetrisation target, but no magnetic symmetry found.")?
        } else {
            &sym_res.unitary_symmetry
        };
        if sym.is_infinite() && params.infinite_order_to_finite.is_none() {
            Err(
                format!(
                    "Molecule symmetrisation cannot be performed using the entirety of the infinite group `{}`. Consider setting the parameter `infinite_order_to_finite` to restrict to a finite subgroup instead.",
                    sym.group_name.as_ref().expect("No target group name found.")
                )
            )
        } else {
            Ok(())
        }
    }
}

impl<'a> MoleculeSymmetrisationDriver<'a> {
    /// Returns a builder to construct a [`MoleculeSymmetrisationDriver`] structure.
    pub fn builder() -> MoleculeSymmetrisationDriverBuilder<'a> {
        MoleculeSymmetrisationDriverBuilder::default()
    }

    /// Executes symmetry-group detection.
    fn symmetrise_molecule(&mut self) -> Result<(), anyhow::Error> {
        let mut trial_mol = Molecule::from_atoms(
            &self
                .molecule
                .get_all_atoms()
                .into_iter()
                .cloned()
                .collect::<Vec<_>>(),
            self.parameters.target_distance_threshold,
        );
        trial_mol.recentre_mut();

        let target_presym = &self.target_symmetry_result.pre_symmetry;
        let target_sym = if self.parameters.use_magnetic_symmetry {
            self.target_symmetry_result
                .magnetic_symmetry
                .as_ref()
                .ok_or_else(|| format_err!("Magnetic symmetry requested as symmetrisation target, but no magnetic symmetry found."))?
        } else {
            &self.target_symmetry_result.unitary_symmetry
        };
        let target_group = UnitaryRepresentedGroup::from_molecular_symmetry(
            target_sym,
            self.parameters.infinite_order_to_finite,
        );
        let ts = target_group
            .elements()
            .clone()
            .into_iter()
            .flat_map(|op| {
                let tmat = op
                    .get_3d_spatial_matrix()
                    .select(Axis(0), &[2, 0, 1])
                    .select(Axis(1), &[2, 0, 1]);

                // Determine the atom permutation corresponding to `op`. This uses the molecule in
                // `target_presym`, which should have the right threshold to be compatible with the
                // symmetry operations generated from `target_sym`.
                let perm = op.act_permute(&target_presym.molecule).ok_or_else(|| {
                    format_err!("Unable to determine the permutation corresponding to `{op}`.")
                })?;
                Ok::<_, anyhow::Error>((tmat, perm))
            })
            .collect::<Vec<(Array2<f64>, Permutation<usize>)>>();
        let order_f64 = target_group
            .order()
            .to_f64()
            .ok_or_else(|| format_err!("Unable to convert the group order to `f64`."))?;

        let mut trial_presym = PreSymmetry::builder()
            .moi_threshold(self.parameters.target_moi_threshold)
            .molecule(&trial_mol, true)
            .build()
            .map_err(|_| format_err!("Cannot construct a pre-symmetry structure."))?;
        let mut trial_sym = Symmetry::new();
        trial_sym.analyse(&trial_presym, self.parameters.use_magnetic_symmetry)?;

        let mut symmetrisation_count = 0;
        while symmetrisation_count == 0
            || (trial_sym.group_name != target_sym.group_name
                && symmetrisation_count <= self.parameters.max_iterations)
        {
            symmetrisation_count += 1;
            let trial_coords = Array2::from_shape_vec(
                (trial_mol.atoms.len(), 3),
                trial_mol
                    .atoms
                    .iter()
                    .flat_map(|atom| atom.coordinates.coords.iter().cloned())
                    .collect::<Vec<_>>(),
            )?;
            let ave_coords = ts.iter().fold(
                Array2::<f64>::zeros(trial_coords.raw_dim()),
                |acc, (tmat, perm)| {
                    // coords.dot(tmat) gives the atom positions transformed in R^3 by tmat.
                    // .select(Axis(0), perm.image()) then permutes the rows so that the atom positions
                    // go back to approximately where they were originally.
                    acc + trial_coords.dot(tmat).select(Axis(0), perm.image())
                },
            ) / order_f64;

            trial_mol
                .atoms
                .iter_mut()
                .enumerate()
                .for_each(|(i, atom)| {
                    atom.coordinates = Point3::<f64>::from_slice(
                        ave_coords
                            .row(i)
                            .as_slice()
                            .expect("Unable to convert a row of averaged coordinates to a slice."),
                    )
                });
            trial_presym = PreSymmetry::builder()
                .moi_threshold(self.parameters.target_moi_threshold)
                .molecule(&trial_mol, true)
                .build()
                .map_err(|_| format_err!("Cannot construct a pre-symmetry structure."))?;
            trial_sym = Symmetry::new();
            trial_sym.analyse(&trial_presym, self.parameters.use_magnetic_symmetry)?;
        }
        if trial_sym.group_name != target_sym.group_name {
            Err(format_err!("Molecule symmetrisation has failed after {symmetrisation_count} iterations."))
        } else {
            log::info!(target: "output", "Symmetrised molecule");
            trial_mol.log_output_display();
            Ok(())
        }
    }
}

impl<'a> QSym2Driver for MoleculeSymmetrisationDriver<'a> {
    type Outcome = MoleculeSymmetrisationResult<'a>;

    fn result(&self) -> Result<&Self::Outcome, anyhow::Error> {
        self.result
            .as_ref()
            .ok_or_else(|| format_err!("No molecule symmetrisation results found."))
    }

    fn run(&mut self) -> Result<(), anyhow::Error> {
        self.symmetrise_molecule()
    }
}
