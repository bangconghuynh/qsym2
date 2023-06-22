use std::fmt;

use anyhow::{ensure, format_err};
use derive_builder::Builder;
use nalgebra::Point3;
use ndarray::{Array2, Axis};
use num_traits::ToPrimitive;

use crate::aux::format::{log_subtitle, log_title, nice_bool, QSym2Output};
use crate::aux::molecule::Molecule;
use crate::drivers::symmetry_group_detection::{
    SymmetryGroupDetectionDriver, SymmetryGroupDetectionParams, SymmetryGroupDetectionResult,
};
use crate::drivers::QSym2Driver;
use crate::io::QSym2FileType;
use crate::permutation::IntoPermutation;
use crate::symmetry::symmetry_core::{PreSymmetry, Symmetry};

#[cfg(test)]
#[path = "molecule_symmetrisation_tests.rs"]
mod molecule_symmetrisation_tests;

// ==================
// Struct definitions
// ==================

// ----------
// Parameters
// ----------

/// A structure containing control parameters for molecule symmetrisation.
#[derive(Clone, Builder, Debug)]
pub struct MoleculeSymmetrisationParams {
    /// Boolean indicating if any available magnetic group should be used for symmetrisation
    /// instead of the unitary group.
    pub use_magnetic_group: bool,

    /// The target moment-of-inertia threshold for the symmetrisation, *i.e.* the symmetrised
    /// molecule will have the target symmetry group at this target moment-of-inertia threshold.
    pub target_moi_threshold: f64,

    /// The target distance threshold for the symmetrisation, *i.e.* the symmetrised molecule will
    /// have the target symmetry group at this target distance threshold.
    pub target_distance_threshold: f64,

    /// Boolean indicating if the symmetrised molecule is also reoriented to align its principal
    /// axes with the space-fixed Cartesian axes.
    ///
    /// See [`Molecule::reorientate`] for more information.
    pub reorientate_molecule: bool,

    /// The maximum number of symmetrisation iterations.
    #[builder(default = "5")]
    pub max_iterations: usize,

    /// The finite order to which any infinite-order symmetry element is reduced, so that a finite
    /// subgroup of an infinite group can be used for the symmetrisation.
    #[builder(default = "None")]
    pub infinite_order_to_finite: Option<u32>,

    /// The output verbosity level.
    #[builder(default = "0")]
    pub verbose: u8,

    /// Optional name for saving the symmetry-group detection result of the symmetrised system as a
    /// binary file of type [`QSym2FileType::Sym`]. If `None`, the result will not be saved.
    #[builder(default = "None")]
    pub symmetrised_result_save_name: Option<String>,
}

impl MoleculeSymmetrisationParams {
    /// Returns a builder to construct a [`MoleculeSymmetrisationParams`] structure.
    pub fn builder() -> MoleculeSymmetrisationParamsBuilder {
        MoleculeSymmetrisationParamsBuilder::default()
    }
}

impl fmt::Display for MoleculeSymmetrisationParams {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Target MoI threshold: {:.3e}", self.target_moi_threshold)?;
        writeln!(
            f,
            "Target geo threshold: {:.3e}",
            self.target_distance_threshold
        )?;
        writeln!(f)?;
        writeln!(
            f,
            "Group used for symmetrisation: {}",
            if self.use_magnetic_group {
                "magnetic group"
            } else {
                "unitary group"
            }
        )?;
        if let Some(finite_order) = self.infinite_order_to_finite {
            writeln!(f, "Infinite order to finite: {finite_order}")?;
        }
        writeln!(
            f,
            "Maximum symmetrisation iterations: {}",
            self.max_iterations
        )?;
        writeln!(f, "Output level: {}", self.verbose)?;
        writeln!(
            f,
            "Save symmetry-group detection results of symmetrised system to file: {}",
            if let Some(name) = self.symmetrised_result_save_name.as_ref() {
                format!("{name}{}", QSym2FileType::Sym.ext())
            } else {
                nice_bool(false)
            }
        )?;
        writeln!(f)?;

        Ok(())
    }
}

// ------
// Result
// ------

/// A structure to contain molecule symmetrisation results.
#[derive(Clone, Builder, Debug)]
pub struct MoleculeSymmetrisationResult<'a> {
    /// The control parameters used to obtain this set of molecule symmetrisation results.
    parameters: &'a MoleculeSymmetrisationParams,

    /// The symmetrised molecule.
    pub symmetrised_molecule: Molecule,
}

impl<'a> MoleculeSymmetrisationResult<'a> {
    fn builder() -> MoleculeSymmetrisationResultBuilder<'a> {
        MoleculeSymmetrisationResultBuilder::default()
    }
}

// ------
// Driver
// ------

/// A driver for iterative molecule symmetrisation.
///
/// Each symmetrisation iteration involves applying the totally symmetric projection operator of
/// the target group to the molecule. This process is repeated until the molecule attains the
/// desired symmetry group at the desired thresholding level.
#[derive(Clone, Builder)]
#[builder(build_fn(validate = "Self::validate"))]
pub struct MoleculeSymmetrisationDriver<'a> {
    /// The control parameters for molecule symmetrisation.
    parameters: &'a MoleculeSymmetrisationParams,

    /// The target symmetry for symmetrisation. This is the result of a symmetry-group detection
    /// calculation where the symmetry of the molecule has been detected at a certain thresholding
    /// level, and now the molecule is to be symmetrised to attain the same symmetry but at a
    /// tighter thresholding level.
    target_symmetry_result: &'a SymmetryGroupDetectionResult,

    /// The result of the symmetrisation.
    #[builder(setter(skip), default = "None")]
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
        let sym = if params.use_magnetic_group {
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
                    "Molecule symmetrisation cannot be performed using the entirety of the infinite group `{}`. \
                    Consider setting the parameter `infinite_order_to_finite` to restrict to a finite subgroup instead.",
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

    /// Executes molecule symmetrisation.
    fn symmetrise_molecule(&mut self) -> Result<(), anyhow::Error> {
        log_title("Molecule Symmetrisation");
        log::info!(target: "qsym2-output", "");
        let params = self.parameters;
        params.log_output_display();

        let loose_moi_threshold = self.target_symmetry_result.pre_symmetry.moi_threshold;
        let loose_dist_threshold = self.target_symmetry_result.pre_symmetry.dist_threshold;
        let tight_moi_threshold = params.target_moi_threshold;
        let tight_dist_threshold = params.target_distance_threshold;

        let mut trial_mol = self
            .target_symmetry_result
            .pre_symmetry
            .recentred_molecule
            .adjust_threshold(tight_dist_threshold);
        let target_unisym = &self.target_symmetry_result.unitary_symmetry;
        let target_magsym = &self.target_symmetry_result.magnetic_symmetry.as_ref();

        log::info!(
            target: "qsym2-output",
            "Target mag. group: {}",
            target_magsym
                .as_ref()
                .map(|magsym| magsym.group_name.as_ref())
                .unwrap_or(None)
                .unwrap_or(&"--".to_string()),
        );
        log::info!(
            target: "qsym2-output",
            "Target uni. group: {}",
            target_unisym.group_name.as_ref().unwrap_or(&"--".to_string()),
        );
        log::info!(
            target: "qsym2-output",
            "Target-symmetry-guaranteed MoI threshold: {:.3e}",
            loose_moi_threshold,
        );
        log::info!(
            target: "qsym2-output",
            "Target-symmetry-guaranteed geo threshold: {:.3e}",
            loose_dist_threshold,
        );
        log::info!(
            target: "qsym2-output",
            "",
        );

        if params.verbose >= 1 {
            let orig_mol = self
                .target_symmetry_result
                .pre_symmetry
                .original_molecule
                .adjust_threshold(tight_dist_threshold);
            log::info!(target: "qsym2-output", "Unsymmetrised original molecule:");
            orig_mol.log_output_display();
            log::info!(target: "qsym2-output", "");

            log::info!(target: "qsym2-output", "Unsymmetrised recentred molecule:");
            trial_mol.log_output_display();
            log::info!(target: "qsym2-output", "");
        }

        log_subtitle("Iterative molecule symmetrisation");
        log::info!(target: "qsym2-output", "");
        let count_length = usize::try_from(params.max_iterations.ilog10() + 2).map_err(|_| {
            format_err!(
                "Unable to convert `{}` to `usize`.",
                params.max_iterations.ilog10() + 2
            )
        })?;
        log::info!(target: "qsym2-output", "{}", "┈".repeat(count_length + 55));
        log::info!(
            target: "qsym2-output",
            "{:>count_length$} {:>12} {:>12} {:>14} {:>12}",
            "#",
            "MoI thresh",
            "Geo thresh",
            "Mag. group",
            "Uni. group",
        );
        log::info!(target: "qsym2-output", "{}", "┈".repeat(count_length + 55));

        let mut trial_presym = PreSymmetry::builder()
            .moi_threshold(tight_moi_threshold)
            .molecule(&trial_mol)
            .build()
            .map_err(|_| format_err!("Cannot construct a pre-symmetry structure."))?;
        let mut trial_magsym = target_magsym.map(|_| Symmetry::new());
        let mut trial_unisym = Symmetry::new();

        // This may fail since the unsymmetrised molecule is being symmetry-analysed at the tight
        // threshold, but that is okay.
        let _ = trial_unisym.analyse(&trial_presym, false);

        let mut symmetrisation_count = 0;
        let mut unisym_check = trial_unisym.group_name == target_unisym.group_name;
        let mut magsym_check = match (trial_magsym.as_ref(), target_magsym) {
            (Some(tri_magsym), Some(tar_magsym)) => tri_magsym.group_name == tar_magsym.group_name,
            _ => true,
        };

        while symmetrisation_count == 0
            || (!(unisym_check && magsym_check)
                && symmetrisation_count < self.parameters.max_iterations)
        {
            symmetrisation_count += 1;

            // Re-locate the symmetry elements of the target symmetry group using the trial
            // molecule since the symmetrisation process might move the symmetry elements slightly.
            let high_trial_mol = trial_mol.adjust_threshold(loose_dist_threshold);
            let high_presym = PreSymmetry::builder()
                .moi_threshold(loose_moi_threshold)
                .molecule(&high_trial_mol)
                .build()
                .map_err(|_| format_err!("Cannot construct a pre-symmetry structure."))?;
            let mut high_sym = Symmetry::new();
            let _high_res = high_sym.analyse(&high_presym, self.parameters.use_magnetic_group);
            if self.parameters.use_magnetic_group {
                ensure!(
                    high_sym.group_name.as_ref() == target_magsym.map(|magsym| magsym.group_name.as_ref()).unwrap_or(None),
                    "Inconsistent target magnetic group -- the target magnetic group is {}, but the magnetic group of the trial molecule at the same thresholds is {}.",
                    target_magsym.map(|magsym| magsym.group_name.as_ref()).unwrap_or(None).unwrap_or(&"--".to_string()),
                    high_sym.group_name.as_ref().unwrap_or(&"--".to_string())
                );
            } else {
                ensure!(
                    high_sym.group_name == target_unisym.group_name,
                    "Inconsistent target unitary group -- the target unitary group is {}, but the unitary group of the trial molecule at the same thresholds is {}.",
                    target_unisym.group_name.as_ref().unwrap_or(&"--".to_string()),
                    high_sym.group_name.as_ref().unwrap_or(&"--".to_string())
                );
            }

            // Only the operations are needed for the symmetrisation. We avoid constructing the
            // full abstract group here, as group closure might not be fulfilled due to the low
            // thresholds.
            let high_ops =
                high_sym.generate_all_operations(self.parameters.infinite_order_to_finite);
            let order_f64 = high_ops
                .len()
                .to_f64()
                .ok_or_else(|| format_err!("Unable to convert the group order to `f64`."))?;

            // Generate transformation matrix and atom permutations for each operation
            let ts = high_ops
                .into_iter()
                .flat_map(|op| {
                    let tmat = op
                        .get_3d_spatial_matrix()
                        .select(Axis(0), &[2, 0, 1])
                        .select(Axis(1), &[2, 0, 1])
                        .reversed_axes();

                    let ord_perm = op
                        .act_permute(&high_trial_mol.molecule_ordinary_atoms())
                        .ok_or_else(|| {
                            format_err!(
                                "Unable to determine the ordinary-atom permutation corresponding to `{op}`."
                            )
                        })?;
                    let mag_perm_opt = high_trial_mol
                        .molecule_magnetic_atoms()
                        .as_ref()
                        .and_then(|high_trial_mag_mol| op.act_permute(high_trial_mag_mol));
                    let elec_perm_opt = high_trial_mol
                        .molecule_electric_atoms()
                        .as_ref()
                        .and_then(|high_trial_elec_mol| op.act_permute(high_trial_elec_mol));
                    Ok::<_, anyhow::Error>((tmat, ord_perm, mag_perm_opt, elec_perm_opt))
                })
                .collect::<Vec<_>>();

            // Apply the totally-symmetric projection operator to the ordinary atoms
            let trial_ord_coords = Array2::from_shape_vec(
                (trial_mol.atoms.len(), 3),
                trial_mol
                    .atoms
                    .iter()
                    .flat_map(|atom| atom.coordinates.coords.iter().cloned())
                    .collect::<Vec<_>>(),
            )?;
            let ave_ord_coords = ts.iter().fold(
                Array2::<f64>::zeros(trial_ord_coords.raw_dim()),
                |acc, (tmat, ord_perm, _, _)| {
                    // coords.dot(tmat) gives the atom positions transformed in R^3 by tmat.
                    // .select(Axis(0), perm.image()) then permutes the rows so that the atom positions
                    // go back to approximately where they were originally.
                    acc + trial_ord_coords.dot(tmat).select(Axis(0), ord_perm.image())
                },
            ) / order_f64;
            trial_mol
                .atoms
                .iter_mut()
                .enumerate()
                .for_each(|(i, atom)| {
                    atom.coordinates = Point3::<f64>::from_slice(
                        ave_ord_coords
                            .row(i)
                            .as_slice()
                            .expect("Unable to convert a row of averaged coordinates to a slice."),
                    )
                });

            // Apply the totally-symmetric projection operator to the magnetic atoms, if any
            if let Some(mag_atoms) = trial_mol.magnetic_atoms.as_mut() {
                let trial_mag_coords = Array2::from_shape_vec(
                    (mag_atoms.len(), 3),
                    mag_atoms
                        .iter()
                        .flat_map(|atom| atom.coordinates.coords.iter().cloned())
                        .collect::<Vec<_>>(),
                )?;
                let ave_mag_coords = ts.iter().fold(
                    Ok(Array2::<f64>::zeros(trial_mag_coords.raw_dim())),
                    |acc: Result<Array2<f64>, anyhow::Error>, (tmat, _, mag_perm_opt, _)| {
                        // coords.dot(tmat) gives the atom positions transformed in R^3 by tmat.
                        // .select(Axis(0), perm.image()) then permutes the rows so that the atom positions
                        // go back to approximately where they were originally.
                        Ok(acc?
                            + trial_mag_coords.dot(tmat).select(
                                Axis(0),
                                mag_perm_opt
                                    .as_ref()
                                    .ok_or_else(|| {
                                        format_err!("Expected magnetic atom permutation not found.")
                                    })?
                                    .image(),
                            ))
                    },
                )? / order_f64;
                mag_atoms.iter_mut().enumerate().for_each(|(i, atom)| {
                    atom.coordinates = Point3::<f64>::from_slice(
                        ave_mag_coords
                            .row(i)
                            .as_slice()
                            .expect("Unable to convert a row of averaged coordinates to a slice."),
                    )
                });
            }

            // Apply the totally-symmetric projection operator to the electric atoms, if any
            if let Some(elec_atoms) = trial_mol.electric_atoms.as_mut() {
                let trial_elec_coords = Array2::from_shape_vec(
                    (elec_atoms.len(), 3),
                    elec_atoms
                        .iter()
                        .flat_map(|atom| atom.coordinates.coords.iter().cloned())
                        .collect::<Vec<_>>(),
                )?;
                let ave_elec_coords = ts.iter().fold(
                    Ok(Array2::<f64>::zeros(trial_elec_coords.raw_dim())),
                    |acc: Result<Array2<f64>, anyhow::Error>, (tmat, _, _, elec_perm_opt)| {
                        // coords.dot(tmat) gives the atom positions transformed in R^3 by tmat.
                        // .select(Axis(0), perm.image()) then permutes the rows so that the atom positions
                        // go back to approximately where they were originally.
                        Ok(acc?
                            + trial_elec_coords.dot(tmat).select(
                                Axis(0),
                                elec_perm_opt
                                    .as_ref()
                                    .ok_or_else(|| {
                                        format_err!("Expected electric atom permutation not found.")
                                    })?
                                    .image(),
                            ))
                    },
                )? / order_f64;
                elec_atoms.iter_mut().enumerate().for_each(|(i, atom)| {
                    atom.coordinates = Point3::<f64>::from_slice(
                        ave_elec_coords
                            .row(i)
                            .as_slice()
                            .expect("Unable to convert a row of averaged coordinates to a slice."),
                    )
                });
            }

            // Re-analyse symmetry of the symmetrised molecule
            trial_presym = PreSymmetry::builder()
                .moi_threshold(self.parameters.target_moi_threshold)
                .molecule(&trial_mol)
                .build()
                .map_err(|_| format_err!("Cannot construct a pre-symmetry structure."))?;
            trial_unisym = Symmetry::new();
            let _unires = trial_unisym.analyse(&trial_presym, false);
            trial_magsym.as_mut().map(|tri_magsym| {
                *tri_magsym = Symmetry::new();
                let _magres = tri_magsym.analyse(&trial_presym, true);
                tri_magsym
            });

            unisym_check = trial_unisym.group_name == target_unisym.group_name;
            magsym_check = match (trial_magsym.as_ref(), target_magsym) {
                (Some(tri_magsym), Some(tar_magsym)) => {
                    tri_magsym.group_name == tar_magsym.group_name
                }
                _ => true,
            };

            log::info!(
                target: "qsym2-output",
                "{:>count_length$} {:>12.3e} {:>12.3e} {:>14} {:>12}",
                symmetrisation_count,
                tight_moi_threshold,
                tight_dist_threshold,
                trial_magsym.as_ref().map(|magsym| magsym.group_name.as_ref()).unwrap_or(None).unwrap_or(&"--".to_string()),
                trial_unisym.group_name.as_ref().unwrap_or(&"--".to_string()),
            );
        }
        log::info!(target: "qsym2-output", "{}", "┈".repeat(count_length + 55));
        log::info!(target: "qsym2-output", "");

        if unisym_check && magsym_check {
            log::info!(
                target: "qsym2-output",
                "Molecule symmetrisation has completed after {symmetrisation_count} {}.",
                if symmetrisation_count != 1 { "iterations" } else { "iteration" }
            );
            log::info!(target: "qsym2-output", "");
            log::info!(target: "qsym2-output", "Symmetrised recentred molecule:");
            trial_mol.log_output_display();
            log::info!(target: "qsym2-output", "");
            if params.reorientate_molecule {
                trial_mol.reorientate_mut(tight_moi_threshold);
                log::info!(target: "qsym2-output", "Symmetrised recentred and reoriented molecule:");
                trial_mol.log_output_display();
                log::info!(target: "qsym2-output", "");
            }
            self.result = Some(
                MoleculeSymmetrisationResult::builder()
                    .parameters(self.parameters)
                    .symmetrised_molecule(trial_mol)
                    .build()?,
            );

            // Verify the symmetrisation result
            log::info!(target: "qsym2-output", "Verifying symmetrisation results...");
            log::info!(target: "qsym2-output", "");
            let verifying_pd_params = SymmetryGroupDetectionParams::builder()
                .moi_thresholds(&[params.target_moi_threshold])
                .distance_thresholds(&[params.target_distance_threshold])
                .time_reversal(self.target_symmetry_result.parameters.time_reversal)
                .write_symmetry_elements(
                    self.target_symmetry_result
                        .parameters
                        .write_symmetry_elements,
                )
                .result_save_name(params.symmetrised_result_save_name.clone())
                .build()?;
            let mut verifying_pd_driver = SymmetryGroupDetectionDriver::builder()
                .parameters(&verifying_pd_params)
                .molecule(self.result().ok().map(|res| &res.symmetrised_molecule))
                .build()?;
            verifying_pd_driver.run()?;
            let verifying_pd_res = verifying_pd_driver.result()?;
            ensure!(
                verifying_pd_res.unitary_symmetry.group_name
                    == self.target_symmetry_result.unitary_symmetry.group_name,
                "Mismatched unitary symmetry: target is {:?}, but symmetrised system has {:?}.",
                self.target_symmetry_result.unitary_symmetry.group_name,
                verifying_pd_res.unitary_symmetry.group_name
            );
            ensure!(
                verifying_pd_res
                    .magnetic_symmetry
                    .as_ref()
                    .map(|magsym| magsym.group_name.as_ref())
                    == self
                        .target_symmetry_result
                        .magnetic_symmetry
                        .as_ref()
                        .map(|magsym| magsym.group_name.as_ref()),
                "Mismatched magnetic symmetry: target is {:?}, but symmetrised system has {:?}.",
                self.target_symmetry_result
                    .magnetic_symmetry
                    .as_ref()
                    .map(|magsym| magsym.group_name.as_ref()),
                verifying_pd_res
                    .magnetic_symmetry
                    .as_ref()
                    .map(|magsym| magsym.group_name.as_ref())
            );
            log::info!(target: "qsym2-output", "Verifying symmetrisation results... Done.");
            log::info!(target: "qsym2-output", "");

            Ok(())
        } else {
            log::error!(
                "Molecule symmetrisation has failed after {symmetrisation_count} {}.",
                if symmetrisation_count != 1 {
                    "iterations"
                } else {
                    "iteration"
                }
            );
            log::info!(target: "qsym2-output", "");
            if params.verbose >= 1 {
                log::info!(target: "qsym2-output", "Molecule after iteration {symmetrisation_count}:");
                trial_mol.log_output_display();
                log::info!(target: "qsym2-output", "");
            }
            Err(format_err!(
                "Molecule symmetrisation has failed after {symmetrisation_count} {}.",
                if symmetrisation_count != 1 {
                    "iterations"
                } else {
                    "iteration"
                }
            ))
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
