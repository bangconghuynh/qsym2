//! Driver for molecule symmetrisation by bootstrapping in QSym².
//!
//! This algorithm symmetrises a molecule iteratively by defining two threshold levels: a `loose`
//! level and a `target` level.
//!
//! In every iteration, the following steps are performed:
//!
//! 1. The molecule is symmetry-analysed at the `target` level; any symmetry elements found are stashed and the symmetry group name, if any, is registered.
//! 2. The molecule is symmetry-analysed at the `loose` level; any symmetry elements found are added to the stash and the symmetry group name, if any, is registered.
//! 3. The convergence criteria (see below) are checked.
//!     - If convergence has been reached, the symmetrisation procedure is terminated.
//!     - If convergence has not been reached, the following steps are carried out.
//! 4. All symmetry elements found in the stash are used to generate all possible symmetry operations which are then used to symmetrise the molecule: each symmetry operation is applied on the original molecule to produce a symmetry-equivalent copy, then all symmetry-equivalent copies are averaged to give the symmetrised molecule.
//! 5. Repeat steps 1 to 4 above until convergence is reached.
//!
//! There are two convergence criteria for the symmetrisation procedure:
//! - **either** when the loose-threshold symmetry agrees with the target-threshold symmetry,
//! - **or** when the target-threshold symmetry contains more elements than the loose-threshold symmetry and has been consistently identified for a pre-specified number of consecutive iterations.
//!
//! At least one criterion must be satisfied in order for convergence to be reached.

use std::fmt;
use std::path::PathBuf;

use anyhow::{ensure, format_err};
use derive_builder::Builder;
use itertools::Itertools;
use nalgebra::Point3;
use ndarray::{Array2, Axis};
use num_traits::ToPrimitive;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::auxiliary::geometry::Transform;
use crate::auxiliary::molecule::Molecule;
use crate::drivers::QSym2Driver;
use crate::drivers::symmetry_group_detection::{
    SymmetryGroupDetectionDriver, SymmetryGroupDetectionParams,
};
use crate::io::QSym2FileType;
use crate::io::format::{QSym2Output, log_subtitle, log_title, nice_bool, qsym2_output};
use crate::permutation::{IntoPermutation, Permutation};
use crate::symmetry::symmetry_core::{PreSymmetry, Symmetry};

#[cfg(test)]
#[path = "molecule_symmetrisation_bootstrap_tests.rs"]
mod molecule_symmetrisation_bootstrap_tests;

// ==================
// Struct definitions
// ==================

// ----------
// Parameters
// ----------

fn default_true() -> bool {
    true
}
fn default_max_iterations() -> usize {
    50
}
fn default_consistent_iterations() -> usize {
    10
}
fn default_loose_threshold() -> f64 {
    1e-2
}
fn default_tight_threshold() -> f64 {
    1e-7
}

/// Structure containing control parameters for molecule symmetrisation by bootstrapping.
#[derive(Clone, Builder, Debug, Serialize, Deserialize)]
pub struct MoleculeSymmetrisationBootstrapParams {
    /// Boolean indicating if the molecule is also reoriented to align its principal axes with the
    /// space-fixed Cartesian axes at every iteration.
    ///
    /// See [`Molecule::reorientate`] for more information.
    #[builder(default = "true")]
    #[serde(default = "default_true")]
    pub reorientate_molecule: bool,

    /// The `loose` moment-of-inertia threshold for the symmetrisation. The symmetry elements found
    /// at this threshold level will be used to bootstrap the symmetry of the molecule.
    #[builder(default = "1e-2")]
    #[serde(default = "default_loose_threshold")]
    pub loose_moi_threshold: f64,

    /// The `loose` distance threshold for the symmetrisation. The symmetry elements found at this
    /// threshold level will be used to bootstrap the symmetry of the molecule.
    #[builder(default = "1e-2")]
    #[serde(default = "default_loose_threshold")]
    pub loose_distance_threshold: f64,

    /// The `target` moment-of-inertia threshold for the symmetrisation.
    #[builder(default = "1e-7")]
    #[serde(default = "default_tight_threshold")]
    pub target_moi_threshold: f64,

    /// The `target` distance threshold for the symmetrisation.
    #[builder(default = "1e-7")]
    #[serde(default = "default_tight_threshold")]
    pub target_distance_threshold: f64,

    /// The maximum number of symmetrisation iterations.
    #[builder(default = "50")]
    #[serde(default = "default_max_iterations")]
    pub max_iterations: usize,

    /// The number of consecutive iterations during which the symmetry group at the `target` level
    /// of threshold must be consistently found for convergence to be reached, *if this group
    /// cannot become identical to the symmetry group at the `loose` level of threshold*.
    #[builder(default = "10")]
    #[serde(default = "default_consistent_iterations")]
    pub consistent_target_symmetry_iterations: usize,

    /// The finite order to which any infinite-order symmetry element is reduced, so that a finite
    /// number of symmetry operations can be used for the symmetrisation.
    #[builder(default = "None")]
    #[serde(default)]
    pub infinite_order_to_finite: Option<u32>,

    /// Boolean indicating if any available magnetic group should be used for symmetrisation instead
    /// of the unitary group.
    #[builder(default = "false")]
    #[serde(default)]
    pub use_magnetic_group: bool,

    /// The output verbosity level.
    #[builder(default = "0")]
    #[serde(default)]
    pub verbose: u8,

    /// Optional name (without the `.xyz` extension) for writing the symmetrised molecule to an XYZ
    /// file. If `None`, no XYZ files will be written.
    #[builder(default = "None")]
    #[serde(default)]
    pub symmetrised_result_xyz: Option<PathBuf>,

    /// Optional name for saving the symmetry-group detection verification result of the symmetrised
    /// system as a binary file of type [`QSym2FileType::Sym`]. If `None`, the result will not be
    /// saved.
    #[builder(default = "None")]
    #[serde(default)]
    pub symmetrised_result_save_name: Option<PathBuf>,
}

impl MoleculeSymmetrisationBootstrapParams {
    /// Returns a builder to construct a [`MoleculeSymmetrisationBootstrapParams`] structure.
    pub fn builder() -> MoleculeSymmetrisationBootstrapParamsBuilder {
        MoleculeSymmetrisationBootstrapParamsBuilder::default()
    }
}

impl Default for MoleculeSymmetrisationBootstrapParams {
    fn default() -> Self {
        Self::builder()
            .build()
            .expect("Unable to construct a default `MoleculeSymmetrisationBootstrapParams`.")
    }
}

impl fmt::Display for MoleculeSymmetrisationBootstrapParams {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Loose MoI threshold: {:.3e}", self.loose_moi_threshold)?;
        writeln!(
            f,
            "Loose geo threshold: {:.3e}",
            self.loose_distance_threshold
        )?;
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
        writeln!(
            f,
            "Target symmetry consistent iterations: {}",
            self.consistent_target_symmetry_iterations
        )?;
        writeln!(f, "Output level: {}", self.verbose)?;
        writeln!(
            f,
            "Save symmetrised molecule to XYZ file: {}",
            if let Some(name) = self.symmetrised_result_xyz.as_ref() {
                let mut path = name.clone();
                path.set_extension("xyz");
                path.display().to_string()
            } else {
                nice_bool(false)
            }
        )?;
        writeln!(
            f,
            "Save symmetry-group detection results of symmetrised system to file: {}",
            if let Some(name) = self.symmetrised_result_save_name.as_ref() {
                let mut path = name.clone();
                path.set_extension(QSym2FileType::Sym.ext());
                path.display().to_string()
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

/// Structure to contain molecule symmetrisation by bootstrapping results.
#[derive(Clone, Builder, Debug)]
pub struct MoleculeSymmetrisationBootstrapResult<'a> {
    /// The control parameters used to obtain this set of molecule symmetrisation results.
    parameters: &'a MoleculeSymmetrisationBootstrapParams,

    /// The symmetrised molecule.
    pub symmetrised_molecule: Molecule,
}

impl<'a> MoleculeSymmetrisationBootstrapResult<'a> {
    fn builder() -> MoleculeSymmetrisationBootstrapResultBuilder<'a> {
        MoleculeSymmetrisationBootstrapResultBuilder::default()
    }
}

// ------
// Driver
// ------

/// Driver for molecule symmetrisation by bootstrapping in QSym².
#[derive(Clone, Builder)]
#[builder(build_fn(validate = "Self::validate"))]
pub struct MoleculeSymmetrisationBootstrapDriver<'a> {
    /// The control parameters for molecule symmetrisation by bootstrapping.
    parameters: &'a MoleculeSymmetrisationBootstrapParams,

    /// The molecule to be symmetrised.
    molecule: &'a Molecule,

    /// The result of the symmetrisation.
    #[builder(setter(skip), default = "None")]
    result: Option<MoleculeSymmetrisationBootstrapResult<'a>>,
}

impl<'a> MoleculeSymmetrisationBootstrapDriverBuilder<'a> {
    fn validate(&self) -> Result<(), String> {
        let params = self
            .parameters
            .ok_or("No molecule symmetrisation parameters found.".to_string())?;
        if params.consistent_target_symmetry_iterations > params.max_iterations {
            return Err(format!(
                "The number of consistent target-symmetry iterations, `{}`, cannot exceed the \
                    maximum number of iterations, `{}`.",
                params.consistent_target_symmetry_iterations, params.max_iterations,
            ));
        }
        if params.target_moi_threshold < 0.0
            || params.loose_moi_threshold < 0.0
            || params.target_distance_threshold < 0.0
            || params.loose_distance_threshold < 0.0
        {
            return Err("The thresholds cannot be negative.".to_string());
        }
        if params.target_moi_threshold > params.loose_moi_threshold {
            return Err(format!(
                "The target MoI threshold, `{:.3e}`, cannot be larger than the \
                    loose MoI threshold, `{:.3e}`.",
                params.target_moi_threshold, params.loose_moi_threshold
            ));
        }
        if params.target_distance_threshold > params.loose_distance_threshold {
            return Err(format!(
                "The target distance threshold, `{:.3e}`, cannot be larger than the \
                    loose distance threshold, `{:.3e}`.",
                params.target_distance_threshold, params.loose_distance_threshold
            ));
        }
        Ok(())
    }
}

impl<'a> MoleculeSymmetrisationBootstrapDriver<'a> {
    /// Returns a builder to construct a [`MoleculeSymmetrisationBootstrapDriver`] structure.
    pub fn builder() -> MoleculeSymmetrisationBootstrapDriverBuilder<'a> {
        MoleculeSymmetrisationBootstrapDriverBuilder::default()
    }

    /// Executes molecule symmetrisation by bootstrapping.
    fn symmetrise_molecule(&mut self) -> Result<(), anyhow::Error> {
        log_title("Molecule Symmetrisation by Bootstrapping");
        qsym2_output!("");
        let params = self.parameters;
        params.log_output_display();

        let mut trial_mol = self.molecule.recentre();

        if params.verbose >= 1 {
            let orig_mol = self
                .molecule
                .adjust_threshold(params.target_distance_threshold);
            qsym2_output!("Unsymmetrised original molecule:");
            orig_mol.log_output_display();
            qsym2_output!("");

            qsym2_output!("Unsymmetrised recentred molecule:");
            trial_mol.log_output_display();
            qsym2_output!("");
        }

        if params.reorientate_molecule {
            // If reorientation is requested, the trial molecule is reoriented prior to
            // symmetrisation, so that the symmetrisation procedure acts on the reoriented molecule
            // itself. The molecule might become disoriented during the symmetrisation process, but
            // any such disorientation is likely to be fairly small, and post-symmetrisation
            // corrections on small disorientation are better than on large disorientation.
            trial_mol.reorientate_mut(params.target_moi_threshold);
            qsym2_output!("Unsymmetrised recentred and reoriented molecule:");
            trial_mol.log_output_display();
            qsym2_output!("");
        };

        log_subtitle("Iterative molecule symmetry bootstrapping");
        qsym2_output!("");
        qsym2_output!("Thresholds:");
        qsym2_output!(
            "  Loose : {:.3e} (MoI) - {:.3e} (distance)",
            params.loose_distance_threshold,
            params.loose_moi_threshold,
        );
        qsym2_output!(
            "  Target: {:.3e} (MoI) - {:.3e} (distance)",
            params.target_moi_threshold,
            params.target_distance_threshold
        );
        qsym2_output!("");
        qsym2_output!("Convergence criteria:");
        qsym2_output!(
            "  either: (1) when the loose-threshold symmetry agrees with the target-threshold symmetry,",
        );
        qsym2_output!(
            "  or    : (2) when the target-threshold symmetry contains more elements than the loose-threshold symmetry and has been consistently identified for {} consecutive iteration{}.",
            params.consistent_target_symmetry_iterations,
            if params.consistent_target_symmetry_iterations == 1 {
                ""
            } else {
                "s"
            }
        );
        qsym2_output!("");

        let count_length = usize::try_from(params.max_iterations.ilog10() + 1).map_err(|_| {
            format_err!(
                "Unable to convert `{}` to `usize`.",
                params.max_iterations.ilog10() + 1
            )
        })?;
        qsym2_output!("{}", "┈".repeat(count_length + 101));
        qsym2_output!(
            " {:>count_length$} {:>22} {:>19}  {:>22} {:>19}  {:>10}",
            "#",
            "Rot. sym. (loose)",
            "Group (loose)",
            "Rot. sym. (target)",
            "Group (target)",
            "Converged?",
        );
        qsym2_output!("{}", "┈".repeat(count_length + 101));

        let mut symmetrisation_count = 0;
        let mut consistent_target_sym_count = 0;
        let mut loose_ops = vec![];
        let mut prev_target_sym_group_name: Option<String> = None;
        let mut converged = false;
        while symmetrisation_count == 0
            || (!converged && symmetrisation_count < params.max_iterations)
        {
            symmetrisation_count += 1;

            // -------------------------------
            // Loose threshold symmetry search
            // -------------------------------
            let mut loose_mol =
                trial_mol.adjust_threshold(self.parameters.loose_distance_threshold);
            let loose_presym = PreSymmetry::builder()
                .moi_threshold(self.parameters.loose_moi_threshold)
                .molecule(&loose_mol)
                .build()
                .map_err(|_| {
                    format_err!("Cannot construct a loose-threshold pre-symmetry structure.")
                })?;

            let mut loose_sym = Symmetry::new();

            // This might fail, but that's fine. We are bootstrapping.
            let _loose_res = loose_sym.analyse(&loose_presym, self.parameters.use_magnetic_group);

            // Only the operations are needed for the symmetrisation. We avoid constructing the
            // full abstract group here, as group closure might not be fulfilled due to the low
            // thresholds.

            loose_ops.extend_from_slice(
                &loose_sym.generate_all_operations(self.parameters.infinite_order_to_finite),
            );
            let n_ops_f64 = loose_ops.len().to_f64().ok_or_else(|| {
                format_err!("Unable to convert the number of operations to `f64`.")
            })?;

            // Generate transformation matrix and atom permutations for each operation
            let ts = loose_ops
                .into_par_iter()
                .flat_map(|op| {
                    let tmat = op
                        .get_3d_spatial_matrix()
                        .select(Axis(0), &[2, 0, 1])
                        .select(Axis(1), &[2, 0, 1])
                        .reversed_axes();

                    let ord_perm = op
                        .act_permute(&loose_mol.molecule_ordinary_atoms())
                        .ok_or_else(|| {
                            format_err!(
                                "Unable to determine the ordinary-atom permutation corresponding to `{op}`."
                            )
                        })?;
                    let mag_perm_opt = loose_mol
                        .molecule_magnetic_atoms()
                        .as_ref()
                        .and_then(|loose_mag_mol| op.act_permute(loose_mag_mol));
                    let elec_perm_opt = loose_mol
                        .molecule_electric_atoms()
                        .as_ref()
                        .and_then(|loose_elec_mol| op.act_permute(loose_elec_mol));
                    Ok::<_, anyhow::Error>((tmat, ord_perm, mag_perm_opt, elec_perm_opt))
                })
                .collect::<Vec<_>>();

            // Apply symmetry operations to the ordinary atoms
            let loose_ord_coords = Array2::from_shape_vec(
                (loose_mol.atoms.len(), 3),
                loose_mol
                    .atoms
                    .iter()
                    .flat_map(|atom| atom.coordinates.coords.iter().cloned())
                    .collect::<Vec<_>>(),
            )?;
            let ave_ord_coords = ts.iter().fold(
                // Parallelisation here does not improve performance, and even causes more
                // numerical instability.
                Array2::<f64>::zeros(loose_ord_coords.raw_dim()),
                |acc, (tmat, ord_perm, _, _)| {
                    // coords.dot(tmat) gives the atom positions transformed in R^3 by tmat.
                    // .select(Axis(0), perm.image()) then permutes the rows so that the atom positions
                    // go back to approximately where they were originally.
                    acc + loose_ord_coords.dot(tmat).select(Axis(0), ord_perm.image())
                },
            ) / n_ops_f64;
            loose_mol
                .atoms
                .par_iter_mut()
                .enumerate()
                .for_each(|(i, atom)| {
                    atom.coordinates = Point3::<f64>::from_slice(
                        ave_ord_coords
                            .row(i)
                            .as_slice()
                            .expect("Unable to convert a row of averaged coordinates to a slice."),
                    )
                });

            // Apply symmetry operations to the magnetic atoms, if any
            if let Some(mag_atoms) = loose_mol.magnetic_atoms.as_mut() {
                let loose_mag_coords = Array2::from_shape_vec(
                    (mag_atoms.len(), 3),
                    mag_atoms
                        .iter()
                        .flat_map(|atom| atom.coordinates.coords.iter().cloned())
                        .collect::<Vec<_>>(),
                )?;
                let ave_mag_coords =
                    ts.iter().try_fold(
                        Array2::<f64>::zeros(loose_mag_coords.raw_dim()),
                        |acc: Array2<f64>,
                         (tmat, _, mag_perm_opt, _): &(
                            Array2<_>,
                            _,
                            Option<Permutation<usize>>,
                            _,
                        )| {
                            // coords.dot(tmat) gives the atom positions transformed in R^3 by tmat.
                            // .select(Axis(0), perm.image()) then permutes the rows so that the atom positions
                            // go back to approximately where they were originally.
                            Ok::<_, anyhow::Error>(
                                acc + loose_mag_coords.dot(tmat).select(
                                    Axis(0),
                                    mag_perm_opt
                                        .as_ref()
                                        .ok_or_else(|| {
                                            format_err!(
                                                "Expected magnetic atom permutation not found."
                                            )
                                        })?
                                        .image(),
                                ),
                            )
                        },
                    )? / n_ops_f64;
                mag_atoms.iter_mut().enumerate().for_each(|(i, atom)| {
                    atom.coordinates = Point3::<f64>::from_slice(
                        ave_mag_coords
                            .row(i)
                            .as_slice()
                            .expect("Unable to convert a row of averaged coordinates to a slice."),
                    )
                });
            }

            // Apply symmetry operations to the electric atoms, if any
            if let Some(elec_atoms) = loose_mol.electric_atoms.as_mut() {
                let loose_elec_coords = Array2::from_shape_vec(
                    (elec_atoms.len(), 3),
                    elec_atoms
                        .iter()
                        .flat_map(|atom| atom.coordinates.coords.iter().cloned())
                        .collect::<Vec<_>>(),
                )?;
                let ave_elec_coords = ts.iter().try_fold(
                    Array2::<f64>::zeros(loose_elec_coords.raw_dim()),
                    |acc: Array2<f64>,
                     (tmat, _, _, elec_perm_opt): &(
                        Array2<_>,
                        _,
                        Option<Permutation<usize>>,
                        _,
                    )| {
                        // coords.dot(tmat) gives the atom positions transformed in R^3 by tmat.
                        // .select(Axis(0), perm.image()) then permutes the rows so that the atom positions
                        // go back to approximately where they were originally.
                        Ok::<_, anyhow::Error>(
                            acc + loose_elec_coords.dot(tmat).select(
                                Axis(0),
                                elec_perm_opt
                                    .as_ref()
                                    .ok_or_else(|| {
                                        format_err!("Expected electric atom permutation not found.")
                                    })?
                                    .image(),
                            ),
                        )
                    },
                )? / n_ops_f64;
                elec_atoms.iter_mut().enumerate().for_each(|(i, atom)| {
                    atom.coordinates = Point3::<f64>::from_slice(
                        ave_elec_coords
                            .row(i)
                            .as_slice()
                            .expect("Unable to convert a row of averaged coordinates to a slice."),
                    )
                });
            }

            trial_mol = loose_mol;

            // Recentre and reorientate after symmetrisation
            trial_mol.recentre_mut();
            if params.reorientate_molecule {
                // If reorientation is requested, the trial molecule is reoriented prior to
                // symmetrisation, so that the symmetrisation procedure acts on the reoriented molecule
                // itself. The molecule might become disoriented during the symmetrisation process, but
                // any such disorientation is likely to be fairly small, and post-symmetrisation
                // corrections on small disorientation are better than on large disorientation.
                trial_mol.reorientate_mut(params.target_moi_threshold);
            };

            // -------------------------------
            // Target threshold symmetry check
            // -------------------------------
            let target_mol = trial_mol.adjust_threshold(self.parameters.target_distance_threshold);
            let target_presym = PreSymmetry::builder()
                .moi_threshold(self.parameters.target_moi_threshold)
                .molecule(&target_mol)
                .build()
                .map_err(|_| {
                    format_err!("Cannot construct a target-threshold pre-symmetry structure.")
                })?;

            let mut target_sym = Symmetry::new();

            let _ = target_sym.analyse(&target_presym, params.use_magnetic_group);

            let target_loose_consistent = target_sym.n_elements() == loose_sym.n_elements()
                && target_sym.group_name.is_some()
                && target_sym.group_name == loose_sym.group_name;

            if target_sym.group_name == prev_target_sym_group_name
                && target_sym.n_elements() >= loose_sym.n_elements()
            {
                consistent_target_sym_count += 1;
            } else {
                consistent_target_sym_count = 0;
            }
            prev_target_sym_group_name = target_sym.group_name.clone();
            let target_consistent =
                consistent_target_sym_count >= params.consistent_target_symmetry_iterations;

            converged = target_loose_consistent || target_consistent;
            let converged_reason = [target_loose_consistent, target_consistent]
                .iter()
                .enumerate()
                .filter_map(|(i, c)| {
                    if *c {
                        Some(format!("({})", i + 1))
                    } else {
                        None
                    }
                })
                .join("");

            qsym2_output!(
                " {:>count_length$} {:>22} {:>19}  {:>22} {:>19}  {:>10}",
                symmetrisation_count,
                loose_presym.rotational_symmetry.to_string(),
                format!(
                    "{} ({})",
                    loose_sym.group_name.as_ref().unwrap_or(&"--".to_string()),
                    loose_sym.n_elements()
                ),
                target_presym.rotational_symmetry.to_string(),
                format!(
                    "{} ({})",
                    target_sym.group_name.as_ref().unwrap_or(&"--".to_string()),
                    target_sym.n_elements()
                ),
                if converged {
                    "yes ".to_string() + &converged_reason
                } else {
                    "no".to_string()
                },
            );

            loose_ops =
                target_sym.generate_all_operations(self.parameters.infinite_order_to_finite);
        }
        qsym2_output!("{}", "┈".repeat(count_length + 101));
        qsym2_output!("");

        // --------------------------------
        // Verify the symmetrisation result
        // --------------------------------
        qsym2_output!("Verifying symmetrisation results...");
        qsym2_output!("");
        let verifying_pd_params = SymmetryGroupDetectionParams::builder()
            .moi_thresholds(&[params.target_moi_threshold])
            .distance_thresholds(&[params.target_distance_threshold])
            .time_reversal(params.use_magnetic_group)
            .write_symmetry_elements(true)
            .result_save_name(params.symmetrised_result_save_name.clone())
            .build()?;
        let mut verifying_pd_driver = SymmetryGroupDetectionDriver::builder()
            .parameters(&verifying_pd_params)
            .molecule(Some(&trial_mol))
            .build()?;
        verifying_pd_driver.run()?;
        let verifying_pd_res = verifying_pd_driver.result()?;
        let verifying_group_name = if params.use_magnetic_group {
            verifying_pd_res
                .magnetic_symmetry
                .as_ref()
                .and_then(|magsym| magsym.group_name.as_ref())
        } else {
            verifying_pd_res.unitary_symmetry.group_name.as_ref()
        };
        ensure!(
            prev_target_sym_group_name.as_ref() == verifying_group_name,
            "Mismatched symmetry: iterative symmetry bootstrapping found {}, but verification found {}.",
            prev_target_sym_group_name
                .as_ref()
                .unwrap_or(&"--".to_string()),
            verifying_group_name.unwrap_or(&"--".to_string()),
        );
        qsym2_output!("Verifying symmetrisation results... Done.");
        qsym2_output!("");

        // --------------
        // Saving results
        // --------------
        self.result = Some(
            MoleculeSymmetrisationBootstrapResult::builder()
                .parameters(self.parameters)
                .symmetrised_molecule(trial_mol.clone())
                .build()?,
        );

        if let Some(xyz_name) = params.symmetrised_result_xyz.as_ref() {
            let mut path = xyz_name.clone();
            path.set_extension("xyz");
            verifying_pd_res
                .pre_symmetry
                .recentred_molecule
                .to_xyz(&path)?;
            qsym2_output!("Symmetrised molecule written to: {}", path.display());
            qsym2_output!("");
        }

        Ok(())
    }
}

impl<'a> QSym2Driver for MoleculeSymmetrisationBootstrapDriver<'a> {
    type Params = MoleculeSymmetrisationBootstrapParams;

    type Outcome = MoleculeSymmetrisationBootstrapResult<'a>;

    fn result(&self) -> Result<&Self::Outcome, anyhow::Error> {
        self.result
            .as_ref()
            .ok_or_else(|| format_err!("No molecule sprucing results found."))
    }

    fn run(&mut self) -> Result<(), anyhow::Error> {
        self.symmetrise_molecule()
    }
}
