use std::fmt;

use anyhow::{ensure, format_err};
use derive_builder::Builder;
use nalgebra::Point3;
use ndarray::{Array2, Axis};
use num_traits::ToPrimitive;

use crate::aux::format::{log_subtitle, nice_bool, write_title};
use crate::aux::molecule::Molecule;
use crate::drivers::symmetry_group_detection::{
    SymmetryGroupDetectionDriver, SymmetryGroupDetectionResult,
};
use crate::drivers::{QSym2Driver, QSym2Output};
use crate::group::{GroupProperties, UnitaryRepresentedGroup};
use crate::permutation::{IntoPermutation, Permutation};
use crate::symmetry::symmetry_core::{PreSymmetry, Symmetry};
use crate::symmetry::symmetry_group::SymmetryGroupProperties;

#[cfg(test)]
#[path = "molecule_symmetrisation_tests.rs"]
mod molecule_symmetrisation_tests;

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

    #[builder(default = "0")]
    verbose: u8,
}

impl MoleculeSymmetrisationParams {
    /// Returns a builder to construct a [`MoleculeSymmetrisationParams`] structure.
    pub fn builder() -> MoleculeSymmetrisationParamsBuilder {
        MoleculeSymmetrisationParamsBuilder::default()
    }
}

impl fmt::Display for MoleculeSymmetrisationParams {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write_title(f, "Molecule Symmetrisation")?;
        writeln!(f, "")?;
        writeln!(f, "Target MoI threshold: {:.3e}", self.target_moi_threshold)?;
        writeln!(
            f,
            "Target geo threshold: {:.3e}",
            self.target_distance_threshold
        )?;
        writeln!(f, "")?;
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
        writeln!(f, "Output level: {}", self.verbose)?;
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
        let params = self.parameters;
        params.log_output_display();

        let loose_moi_threshold = self.target_symmetry_result.pre_symmetry.moi_threshold;
        let loose_dist_threshold = self.target_symmetry_result.pre_symmetry.dist_threshold;
        let tight_moi_threshold = params.target_moi_threshold;
        let tight_dist_threshold = params.target_distance_threshold;

        let mut trial_mol = Molecule::from_atoms(
            &self
                .target_symmetry_result
                .pre_symmetry
                .recentred_molecule
                .get_all_atoms()
                .into_iter()
                .cloned()
                .collect::<Vec<_>>(),
            tight_dist_threshold,
        );
        let target_uni_sym = &self.target_symmetry_result.unitary_symmetry;
        let target_mag_sym = &self.target_symmetry_result.magnetic_symmetry.as_ref();
        // let target_sym = if self.parameters.use_magnetic_symmetry {
        //     self.target_symmetry_result
        //         .magnetic_symmetry
        //         .as_ref()
        //         .ok_or_else(|| format_err!("Magnetic symmetry requested as symmetrisation target, but no magnetic symmetry found."))?
        // } else {
        //     &self.target_symmetry_result.unitary_symmetry
        // };

        log::info!(
            target: "output",
            "Target symmetry group: {}",
            target_sym.group_name.as_ref().unwrap_or(&"--".to_string()),
        );
        log::info!(
            target: "output",
            "Target-symmetry-guaranteed MoI threshold: {:.3e}",
            loose_moi_threshold,
        );
        log::info!(
            target: "output",
            "Target-symmetry-guaranteed geo threshold: {:.3e}",
            loose_dist_threshold,
        );
        log::info!(
            target: "output",
            "",
        );

        if params.verbose >= 1 {
            log::info!(target: "output", "Unsymmetrised molecule:");
            trial_mol.log_output_display();
            log::info!(target: "output", "");
        }

        let mut trial_presym = PreSymmetry::builder()
            .moi_threshold(tight_moi_threshold)
            .molecule(&trial_mol)
            .build()
            .map_err(|_| format_err!("Cannot construct a pre-symmetry structure."))?;
        let mut trial_sym = Symmetry::new();
        trial_sym.analyse(&trial_presym, self.parameters.use_magnetic_symmetry)?;

        log_subtitle("Iterative molecule symmetrisation");
        log::info!(target: "output", "");
        let count_length = usize::try_from(params.max_iterations.ilog10() + 2).map_err(|_| {
            format_err!(
                "Unable to convert `{}` to `usize`.",
                params.max_iterations.ilog10() + 2
            )
        })?;
        log::info!(target: "output", "{}", "┈".repeat(count_length + 42));
        log::info!(
            target: "output",
            "{:>count_length$} {:>12} {:>12} {:>14}",
            "#",
            "MoI thresh",
            "Geo thresh",
            "Sym. group",
        );
        log::info!(target: "output", "{}", "┈".repeat(count_length + 42));

        let mut symmetrisation_count = 0;
        while symmetrisation_count == 0
            || (trial_sym.group_name != target_sym.group_name
                && symmetrisation_count < self.parameters.max_iterations)
        {
            symmetrisation_count += 1;

            // Re-locate the symmetry elements of the target symmetry group using the trial
            // molecule since the symmeteisation process might move the symmetry elements slightly.
            let high_trial_mol = trial_mol.adjust_threshold(loose_dist_threshold);
            let high_presym = PreSymmetry::builder()
                .moi_threshold(loose_moi_threshold)
                .molecule(&high_trial_mol)
                .build()
                .map_err(|_| format_err!("Cannot construct a pre-symmetry structure."))?;
            let mut high_sym = Symmetry::new();
            let _high_res = high_sym.analyse(&high_presym, self.parameters.use_magnetic_symmetry);
            ensure!(
                high_sym.group_name == target_sym.group_name,
                "Inconsistent target symmetry group -- the target symmetry group is {}, but the symmetry group of the trial molecule at the same thresholds is {}.",
                target_sym.group_name.as_ref().expect("Target symmetry group name not found."),
                high_sym.group_name.as_ref().expect("Trial symmetry group name not found.")
            );

            let high_group = UnitaryRepresentedGroup::from_molecular_symmetry(
                &high_sym,
                self.parameters.infinite_order_to_finite,
            );
            let order_f64 = high_group
                .order()
                .to_f64()
                .ok_or_else(|| format_err!("Unable to convert the group order to `f64`."))?;

            // Generate transformation matrix and ordinary-atom permutation for each operation
            let ts = high_group
                .elements()
                .clone()
                .into_iter()
                .flat_map(|op| {
                    let tmat = op
                        .get_3d_spatial_matrix()
                        .select(Axis(0), &[2, 0, 1])
                        .select(Axis(1), &[2, 0, 1])
                        .reversed_axes();

                    let perm = op.act_permute(&high_trial_mol).ok_or_else(|| {
                        format_err!("Unable to determine the permutation corresponding to `{op}`.")
                    })?;
                    Ok::<_, anyhow::Error>((tmat, perm))
                })
                .collect::<Vec<(Array2<f64>, Permutation<usize>)>>();

            // Apply the totally-symmetric projection operator
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

            // Re-analyse symmetry of the projected molecule
            trial_presym = PreSymmetry::builder()
                .moi_threshold(self.parameters.target_moi_threshold)
                .molecule(&trial_mol)
                .build()
                .map_err(|_| format_err!("Cannot construct a pre-symmetry structure."))?;
            trial_sym = Symmetry::new();
            let _res = trial_sym.analyse(&trial_presym, self.parameters.use_magnetic_symmetry);
            log::info!(
                target: "output",
                "{:>count_length$} {:>12.3e} {:>12.3e} {:>14}",
                symmetrisation_count,
                tight_moi_threshold,
                tight_dist_threshold,
                trial_sym.group_name.as_ref().unwrap_or(&"--".to_string()),
            );
        }
        log::info!(target: "output", "{}", "┈".repeat(count_length + 42));
        log::info!(target: "output", "");

        if trial_sym.group_name != target_sym.group_name {
            log::error!(
                "Molecule symmetrisation has failed after {symmetrisation_count} iterations.",
            );
            log::info!(target: "output", "");
            if params.verbose >= 1 {
                log::info!(target: "output", "Molecule after iteration {symmetrisation_count}:");
                trial_mol.log_output_display();
                log::info!(target: "output", "");
            }
            Err(format_err!(
                "Molecule symmetrisation has failed after {symmetrisation_count} iterations."
            ))
        } else {
            log::info!(
                target: "output",
                "Molecule symmetrisation has completed after {symmetrisation_count} iterations.",
            );
            log::info!(target: "output", "");
            log::info!(target: "output", "Symmetrised molecule");
            trial_mol.log_output_display();
            log::info!(target: "output", "");
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
