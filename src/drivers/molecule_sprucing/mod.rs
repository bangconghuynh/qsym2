//! Driver for molecule symmetrisation in QSym².

use std::collections::HashMap;
use std::fmt;
use std::path::PathBuf;

use anyhow::{ensure, format_err};
use argmin::core::{CostFunction, Executor, Gradient, Solver, State, TerminationStatus, TerminationReason};
use argmin::solver::{
    linesearch::condition::ArmijoCondition, linesearch::BacktrackingLineSearch, quasinewton::BFGS,
};
use argmin_math::ArgminL2Norm;
use derive_builder::Builder;
use itertools::Itertools;
use nalgebra::{Point3, Vector3};
use ndarray::{Array1, Array2, Axis, ShapeBuilder};
use num_traits::ToPrimitive;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::auxiliary::geometry::Transform;
use crate::auxiliary::misc::HashableFloat;
use crate::auxiliary::molecule::Molecule;
use crate::drivers::symmetry_group_detection::{
    SymmetryGroupDetectionDriver, SymmetryGroupDetectionParams, SymmetryGroupDetectionResult,
};
use crate::drivers::QSym2Driver;
use crate::io::format::{
    log_subtitle, log_title, nice_bool, qsym2_output, qsym2_warn, QSym2Output,
};
use crate::io::QSym2FileType;
use crate::permutation::IntoPermutation;
use crate::symmetry::symmetry_core::{PreSymmetry, Symmetry};

#[cfg(test)]
#[path = "molecule_sprucing_tests.rs"]
mod molecule_sprucing_tests;

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
    5
}
fn default_loose_threshold() -> f64 {
    1e-3
}
fn default_tight_threshold() -> f64 {
    1e-7
}

/// Structure containing control parameters for molecule sprucing.
#[derive(Clone, Builder, Debug, Serialize, Deserialize)]
pub struct MoleculeSprucingParams {
    /// The tolerance for comparing approximate values in the unspruced structure.
    #[builder(default = "1e-3")]
    #[serde(default = "default_loose_threshold")]
    pub sprucing_tolerance: f64,

    /// Boolean indicating if the spruced-up molecule is also reoriented to align its principal
    /// axes with the space-fixed Cartesian axes.
    ///
    /// See [`Molecule::reorientate`] for more information.
    #[builder(default = "true")]
    #[serde(default = "default_true")]
    pub reorientate_molecule: bool,

    /// The gradient threshold for the optimisation of atomic positions that satisfy the spruced-up
    /// distance matrix.
    #[builder(default = "1e-7")]
    #[serde(default = "default_tight_threshold")]
    pub optimisation_gradient_threshold: f64,

    /// The step size for the line search in the optimisation of atomic positions that satisfy the
    /// spruced-up distance matrix.
    #[builder(default = "1e-7")]
    #[serde(default = "default_tight_threshold")]
    pub optimisation_line_search_step_size: f64,

    /// The maximum number of atomic position optimisation iterations.
    #[builder(default = "5")]
    #[serde(default = "default_max_iterations")]
    pub optimisation_max_iterations: usize,

    /// The output verbosity level.
    #[builder(default = "0")]
    #[serde(default)]
    pub verbose: u8,

    /// Optional name (without the `.xyz` extension) for writing the spruced-up molecule to an XYZ
    /// file. If `None`, no XYZ files will be written.
    #[builder(default = "None")]
    #[serde(default)]
    pub spruced_result_xyz: Option<PathBuf>,
    // /// Optional name for saving the symmetry-group detection result of the symmetrised system as a
    // /// binary file of type [`QSym2FileType::Sym`]. If `None`, the result will not be saved.
    // #[builder(default = "None")]
    // #[serde(default)]
    // pub symmetrised_result_save_name: Option<PathBuf>,
}

impl MoleculeSprucingParams {
    /// Returns a builder to construct a [`MoleculeSprucingParams`] structure.
    pub fn builder() -> MoleculeSprucingParamsBuilder {
        MoleculeSprucingParamsBuilder::default()
    }
}

impl Default for MoleculeSprucingParams {
    fn default() -> Self {
        Self::builder()
            .build()
            .expect("Unable to construct a default `MoleculeSprucingParams`.")
    }
}

impl fmt::Display for MoleculeSprucingParams {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Sprucing tolerance: {:.3e}", self.sprucing_tolerance)?;
        writeln!(
            f,
            "Atomic position optimisation gradient threshold: {:.3e}",
            self.optimisation_gradient_threshold
        )?;
        writeln!(
            f,
            "Maximum optimisation iterations: {}",
            self.optimisation_max_iterations
        )?;
        writeln!(
            f,
            "Optimisation line search step size: {:.3e}",
            self.optimisation_line_search_step_size
        )?;
        writeln!(f, "")?;

        writeln!(f, "Output level: {}", self.verbose)?;
        writeln!(
            f,
            "Save spruced-up molecule to XYZ file: {}",
            if let Some(name) = self.spruced_result_xyz.as_ref() {
                let mut path = name.clone();
                path.set_extension("xyz");
                path.display().to_string()
            } else {
                nice_bool(false)
            }
        )?;
        // writeln!(
        //     f,
        //     "Save symmetry-group detection results of symmetrised system to file: {}",
        //     if let Some(name) = self.symmetrised_result_save_name.as_ref() {
        //         let mut path = name.clone();
        //         path.set_extension(QSym2FileType::Sym.ext());
        //         path.display().to_string()
        //     } else {
        //         nice_bool(false)
        //     }
        // )?;
        writeln!(f)?;

        Ok(())
    }
}

// ------
// Result
// ------

/// Structure to contain molecule symmetrisation results.
#[derive(Clone, Builder, Debug)]
pub struct MoleculeSprucingResult<'a> {
    /// The control parameters used to obtain this set of molecule symmetrisation results.
    parameters: &'a MoleculeSprucingParams,

    /// The symmetrised molecule.
    pub symmetrised_molecule: Molecule,
}

impl<'a> MoleculeSprucingResult<'a> {
    fn builder() -> MoleculeSprucingResultBuilder<'a> {
        MoleculeSprucingResultBuilder::default()
    }
}

// ------
// Driver
// ------

/// Driver for iterative molecule symmetrisation by distance matrix.
///
/// Each symmetrisation iteration involves applying the totally symmetric projection operator of
/// the target group to the molecule. This process is repeated until the molecule attains the
/// desired symmetry group at the desired thresholding level.
#[derive(Clone, Builder)]
pub struct MoleculeSprucingDriver<'a> {
    /// The control parameters for molecule symmetrisation by distance matrix.
    parameters: &'a MoleculeSprucingParams,

    /// The molecule to be symmetrised.
    molecule: &'a Molecule,

    /// The result of the symmetrisation.
    #[builder(setter(skip), default = "None")]
    result: Option<MoleculeSprucingResult<'a>>,
}

// impl<'a> MoleculeSprucingDriverBuilder<'a> {
//     fn validate(&self) -> Result<(), String> {
//         let params = self
//             .parameters
//             .ok_or("No molecule symmetrisation parameters found.".to_string())?;
//         let sym_res = self
//             .target_symmetry_result
//             .ok_or("No target symmetry group result for symmetrisation found.".to_string())?;
//         } else {
//             &sym_res.unitary_symmetry
//         };
//         if sym.is_infinite() && params.infinite_order_to_finite.is_none() {
//             Err(
//                 format!(
//                     "Molecule symmetrisation cannot be performed using the entirety of the infinite group `{}`. \
//                     Consider setting the parameter `infinite_order_to_finite` to restrict to a finite subgroup instead.",
//                     sym.group_name.as_ref().expect("No target group name found.")
//                 )
//             )
//         } else {
//             Ok(())
//         }
//     }
// }

impl<'a> MoleculeSprucingDriver<'a> {
    /// Returns a builder to construct a [`MoleculeSprucingDriver`] structure.
    pub fn builder() -> MoleculeSprucingDriverBuilder<'a> {
        MoleculeSprucingDriverBuilder::default()
    }

    /// Executes molecule symmetrisation.
    fn symmetrise_molecule(&mut self) -> Result<(), anyhow::Error> {
        log_title("Molecule Sprucing");
        qsym2_output!("");
        qsym2_output!("Ref.: Beruski, O. & Vidal, L. N. *Journal of Computational Chemistry* **35**, 290–299 (2014).");
        qsym2_output!("");
        let params = self.parameters;
        params.log_output_display();

        let loose_tolerance = params.sprucing_tolerance;
        let loose_precision = loose_tolerance
            .log10()
            .abs()
            .round()
            .to_usize()
            .unwrap_or(7)
            + 1;
        let loose_length = (loose_precision + loose_precision.div_euclid(2)).max(6);

        let tight_precision = params
            .optimisation_gradient_threshold
            .log10()
            .abs()
            .round()
            .to_usize()
            .unwrap_or(7)
            + 1;
        let tight_length = (tight_precision + tight_precision.div_euclid(2)).max(6);

        let mut unspruced_mol = self.molecule.adjust_threshold(loose_tolerance);

        qsym2_output!("Unspruced molecule:");
        unspruced_mol.log_output_display();
        qsym2_output!("");
        if params.reorientate_molecule {
            // If reorientation is requested, the trial molecule is reoriented prior to
            // symmetrisation, so that the symmetrisation procedure acts on the reoriented molecule
            // itself. The molecule might become disoriented during the symmetrisation process, but
            // any such disorientation is likely to be fairly small, and post-symmetrisation
            // corrections on small disorientation are better than on large disorientation.
            unspruced_mol.reorientate_mut(loose_tolerance);
            qsym2_output!("Reoriented unspruced molecule:");
            unspruced_mol.log_output_display();
            qsym2_output!("");
        };

        log_subtitle("Interatomic distance matrix sprucing");
        qsym2_output!("");

        let original_atoms = unspruced_mol.get_all_atoms();
        let n_atoms = original_atoms.len();
        let (unspruced_distmat, equiv_col_indicess) =
            unspruced_mol.calc_interatomic_distance_matrix();
        qsym2_output!("Symmetry-equivalent (unspruced) atoms at the loose tolerance of {loose_tolerance:+.3e}:");
        for (i, equiv_col_indices) in equiv_col_indicess.iter().enumerate() {
            qsym2_output!("  Group {i}:");
            for atom_index in equiv_col_indices.iter() {
                qsym2_output!("  {}", original_atoms[*atom_index]);
            }
        }
        qsym2_output!("");

        if params.verbose >= 1 {
            qsym2_output!("Unspruced interatomic distance matrix:");
            qsym2_output!("{unspruced_distmat:loose_length$.loose_precision$}");
            qsym2_output!("");
        }

        // Symmetrise distance matrix
        let mut spruced_distmat = Array2::<f64>::zeros((n_atoms, n_atoms));
        equiv_col_indicess.iter().for_each(|equiv_col_indices| {
            let unspruced_distmat_sea_j = unspruced_distmat.select(Axis(1), equiv_col_indices);
            equiv_col_indicess.iter().for_each(|equiv_row_indices| {
                let unspruced_distmat_sea_ij =
                    unspruced_distmat_sea_j.select(Axis(0), equiv_row_indices);
                let (sum_sorted_cols, sort_indicess) =
                    unspruced_distmat_sea_ij.columns().into_iter().fold(
                        (Array1::<f64>::zeros(equiv_row_indices.len()), vec![]),
                        |mut acc, col| {
                            let col = col.into_iter().collect_vec();
                            let mut col_argsort = (0..col.len()).collect_vec();
                            col_argsort.sort_by(|&i, &j| {
                                col[i].partial_cmp(&col[j]).unwrap_or_else(|| {
                                    panic!(
                                        "Interatomic distances {} and {} cannot be compared.",
                                        col[i], col[j]
                                    )
                                })
                            });
                            let sorted_col = Array1::from_iter(col_argsort.iter().map(|i| col[*i]));
                            acc.0 = acc.0 + sorted_col;
                            acc.1.push(col_argsort);
                            acc
                        },
                    );
                let scaled_sum_sorted_cols = sum_sorted_cols.clone()
                    / equiv_col_indices
                        .len()
                        .to_f64()
                        .expect("Unable to convert the number of SEAs in this set to `f64`.");
                // println!("For equiv cols {equiv_col_indices:?} and equiv rows {equiv_row_indices:?}, scaled_sum_sorted_cols is {scaled_sum_sorted_cols:?}.");
                let sub_equiv_row_indicess = scaled_sum_sorted_cols
                    .indexed_iter()
                    .tuple_windows()
                    .fold(vec![vec![0]], |mut acc, ((_, disti), (j, distj))| {
                        if approx::abs_diff_eq!(disti, distj, epsilon = loose_tolerance,) {
                            println!("{disti} == {distj}");
                            let n = acc.len();
                            acc[n - 1].push(j);
                        } else {
                            acc.push(vec![j]);
                        }
                        acc
                    });
                // println!("For equiv cols {equiv_col_indices:?} and equiv rows {equiv_row_indices:?}, sub_equiv_row_indicess are {sub_equiv_row_indicess:?}.");
                sub_equiv_row_indicess
                    .iter()
                    .for_each(|sub_equiv_row_indices| {
                        let ave_dist = sum_sorted_cols
                            .select(Axis(0), sub_equiv_row_indices)
                            .iter()
                            .sum::<f64>()
                            / (sub_equiv_row_indices.len() * equiv_col_indices.len())
                                .to_f64()
                                .expect(
                                    "Unable to convert the number averaging distances to `f64`.",
                                );
                        sub_equiv_row_indices.iter().for_each(|i| {
                            equiv_col_indices.iter().zip(sort_indicess.iter()).for_each(
                                |(j, sort_indices_j)| {
                                    // println!("For equiv cols {equiv_col_indices:?} and equiv rows {equiv_row_indices:?}, setting {}, {j} to {ave_dist}.", equiv_row_indices[sort_indices_j[*i]]);
                                    spruced_distmat[(equiv_row_indices[sort_indices_j[*i]], *j)] =
                                        ave_dist;
                                },
                            )
                        });
                    });
            });
        });

        if params.verbose >= 1 {
            qsym2_output!("Spruced-up interatomic distance matrix:");
            qsym2_output!("{spruced_distmat:tight_length$.tight_precision$}");
            qsym2_output!("");
        }

        log_subtitle("Gradient-based atomic position optimisation");
        qsym2_output!("");
        // BFGS search for atomic positions satisfying the symmetrised distance matrix
        let r0 = Array1::from_vec(
            original_atoms
                .iter()
                .flat_map(|atom| atom.coordinates.iter())
                .cloned()
                .collect_vec(),
        );

        let problem = MoleculeSprucingProblem {
            spruced_distmat,
            unspruced_mol,
        };

        let linesearch = BacktrackingLineSearch::<Array1<f64>, Array1<f64>, _, f64>::new(
            ArmijoCondition::new(params.optimisation_line_search_step_size)?,
        );
        let solver: BFGS<_, f64> =
            BFGS::new(linesearch)
            // .with_tolerance_cost(params.optimisation_gradient_threshold)?
            .with_tolerance_grad(params.optimisation_gradient_threshold)?;

        qsym2_output!("Solver: BFGS with backtracking line search");
        qsym2_output!(
            "  BFGS gradient tolerance: {:.3e}",
            params.optimisation_gradient_threshold
        );
        qsym2_output!(
            "  BFGS maximum iterations: {}",
            params.optimisation_max_iterations
        );
        qsym2_output!(
            "  Line search conditions : Armijo with step size {:.3e}",
            params.optimisation_line_search_step_size,
        );
        qsym2_output!("");

        let res = Executor::new(problem.clone(), solver)
            .configure(|state| {
                state
                    .param(r0)
                    .inv_hessian(Array2::<f64>::eye(3 * n_atoms))
                    .target_cost(0.0)
                    .max_iters(params.optimisation_max_iterations.to_u64().unwrap_or_else(|| {
                        qsym2_warn!(
                            "Unable to convert the specified maximum number of iterations, {}, to `u64`. The value {} will be used instead.",
                            params.optimisation_max_iterations,
                            u64::MAX
                        );
                        u64::MAX
                    }))
            })
            .run()?;

        let final_state = res.state();
        let termination_status = final_state.get_termination_status();
        qsym2_output!("BFGS optimisation result:");
        qsym2_output!("  Termination status: {}", termination_status);
        qsym2_output!("  Final iteration: {}", final_state.iter);
        qsym2_output!("  Last best iteration: {}", final_state.last_best_iter);
        qsym2_output!(
            "  Best cost function: {:.3e} (target: {:.3e})",
            final_state.get_best_cost(),
            final_state.get_target_cost()
        );
        qsym2_output!(
            "  Final gradient norm: {:.3e}",
            final_state
                .get_gradient()
                .ok_or(format_err!("Unable to retrieve the final gradient."))?
                .l2_norm(),
        );
        qsym2_output!("");

        if let TerminationStatus::Terminated(TerminationReason::SolverConverged) = termination_status {
            let spruced_r = res.state().get_best_param().ok_or(format_err!(
                "Unable to retrieved the converged atomic positions."
            ))?;

            let spruced_mol = if params.reorientate_molecule {
                let mut spruced_mol = problem
                    .create_molecule_from_param(spruced_r)
                    .adjust_threshold(params.optimisation_gradient_threshold);
                spruced_mol.reorientate_mut(params.optimisation_gradient_threshold);
                qsym2_output!("Reoriented spruced-up molecule:");
                spruced_mol.log_output_display();
                qsym2_output!("");
                spruced_mol
            } else {
                let spruced_mol = problem
                    .create_molecule_from_param(spruced_r)
                    .adjust_threshold(params.optimisation_gradient_threshold);
                qsym2_output!("Spruced-up molecule:");
                spruced_mol.log_output_display();
                qsym2_output!("");
                spruced_mol
            };

            if let Some(xyz_name) = params.spruced_result_xyz.as_ref() {
                let mut path = xyz_name.clone();
                path.set_extension("xyz");
                spruced_mol.to_xyz(&path)?;
                qsym2_output!("Spruced-up molecule written to: {}", path.display());
                qsym2_output!("");
            }

            self.result = Some(
                MoleculeSprucingResult::builder()
                    .parameters(self.parameters)
                    .symmetrised_molecule(spruced_mol)
                    .build()?,
            );

            Ok(())
        } else {
            Err(format_err!("Molecule sprucing has failed with status: {termination_status}."))
        }

    }
}

impl<'a> QSym2Driver for MoleculeSprucingDriver<'a> {
    type Params = MoleculeSprucingParams;

    type Outcome = MoleculeSprucingResult<'a>;

    fn result(&self) -> Result<&Self::Outcome, anyhow::Error> {
        self.result
            .as_ref()
            .ok_or_else(|| format_err!("No molecule sprucing results found."))
    }

    fn run(&mut self) -> Result<(), anyhow::Error> {
        self.symmetrise_molecule()
    }
}

#[derive(Clone)]
struct MoleculeSprucingProblem {
    spruced_distmat: Array2<f64>,
    unspruced_mol: Molecule,
}

impl MoleculeSprucingProblem {
    /// Adjusts the coordinates of [`Self::unspruced_mol`] based on an input coordinate
    /// parameter and generates a new [`Molecule`].
    ///
    /// # Arguments
    ///
    /// * `r` - An input coordinate parameter.
    ///
    /// # Returns
    ///
    /// A new molecule with atomic coordinates specified by `r`.
    fn create_molecule_from_param(&self, r: &Array1<f64>) -> Molecule {
        assert!({
            let original_atoms = self.unspruced_mol.get_all_atoms();
            let n_atoms = original_atoms.len();
            r.len() == 3 * n_atoms
        });

        let mut mol = self.unspruced_mol.clone();
        mol.get_all_atoms_mut()
            .iter_mut()
            .zip(r.iter().chunks(3).into_iter())
            .for_each(|(atom, trial_coords)| {
                let trial_coords = trial_coords.into_iter().collect_vec();
                atom.coordinates =
                    Point3::new(*trial_coords[0], *trial_coords[1], *trial_coords[2]);
            });
        mol
    }
}

impl CostFunction for MoleculeSprucingProblem {
    type Param = Array1<f64>;
    type Output = f64;

    fn cost(&self, r: &Self::Param) -> Result<Self::Output, argmin::core::Error> {
        let original_atoms = self.unspruced_mol.get_all_atoms();
        let trial_mol = self.create_molecule_from_param(r);
        let (trial_distmat, _) = trial_mol.calc_interatomic_distance_matrix();

        let distmat_diff = (trial_distmat - &self.spruced_distmat)
            .iter()
            .map(|v| v.powi(2))
            .sum::<f64>()
            / 2.0;
        let translation_diff = trial_mol
            .get_all_atoms()
            .iter()
            .zip(original_atoms.iter())
            .fold([0.0, 0.0, 0.0], |acc, (t_atom, u_atom)| {
                assert_eq!(t_atom.atomic_number, u_atom.atomic_number);
                let delta_coordinates =
                    (t_atom.coordinates - u_atom.coordinates) * t_atom.atomic_mass;
                [
                    acc[0] + delta_coordinates[0],
                    acc[1] + delta_coordinates[1],
                    acc[2] + delta_coordinates[2],
                ]
            })
            .iter()
            .map(|v| v.powi(2))
            .sum::<f64>();
        let rotation_diff = trial_mol
            .get_all_atoms()
            .iter()
            .zip(original_atoms.iter())
            .fold([0.0, 0.0, 0.0], |acc, (t_atom, u_atom)| {
                assert_eq!(t_atom.atomic_number, u_atom.atomic_number);
                let original_coordinates =
                    Vector3::from_iterator(u_atom.coordinates.iter().cloned());
                let delta_coordinates = Vector3::from_iterator(
                    ((t_atom.coordinates - u_atom.coordinates) * t_atom.atomic_mass)
                        .iter()
                        .cloned(),
                );
                let rot_coordinates = original_coordinates.cross(&delta_coordinates);
                [
                    acc[0] + rot_coordinates[0],
                    acc[1] + rot_coordinates[1],
                    acc[2] + rot_coordinates[2],
                ]
            })
            .iter()
            .map(|v| v.powi(2))
            .sum::<f64>();
        println!("S: {distmat_diff} {translation_diff} {rotation_diff}");
        Ok(distmat_diff + translation_diff + rotation_diff)
    }
}

impl Gradient for MoleculeSprucingProblem {
    type Param = Array1<f64>;
    type Gradient = Array1<f64>;

    fn gradient(&self, r: &Self::Param) -> Result<Self::Gradient, argmin::core::Error> {
        let original_atoms = self.unspruced_mol.get_all_atoms();
        let n_atoms = original_atoms.len();
        let trial_mol = self.create_molecule_from_param(r);
        let trial_atoms = trial_mol.get_all_atoms();

        let (trial_distmat, _) = trial_mol.calc_interatomic_distance_matrix();

        let delta_coordinates = trial_mol
            .get_all_atoms()
            .iter()
            .zip(original_atoms.iter())
            .map(|(t_atom, u_atom)| {
                assert_eq!(t_atom.atomic_number, u_atom.atomic_number);
                (t_atom.coordinates - u_atom.coordinates) * t_atom.atomic_mass
            })
            .collect_vec();

        let rot_coordinates = trial_mol
            .get_all_atoms()
            .iter()
            .zip(original_atoms.iter())
            .map(|(t_atom, u_atom)| {
                assert_eq!(t_atom.atomic_number, u_atom.atomic_number);
                let original_coordinates =
                    Vector3::from_iterator(u_atom.coordinates.iter().cloned());
                let delta_coordinates = Vector3::from_iterator(
                    ((t_atom.coordinates - u_atom.coordinates) * t_atom.atomic_mass)
                        .iter()
                        .cloned(),
                );
                original_coordinates.cross(&delta_coordinates)
            })
            .collect_vec();

        let dfdr = (0..n_atoms)
            .cartesian_product(0..3)
            .map(|(k, a)| {
                let distmat_diff_grad = 2.0
                    * (0..n_atoms)
                        .map(|j| {
                            if j == k {
                                0.0
                            } else {
                                (trial_distmat[(k, j)] - self.spruced_distmat[(k, j)])
                                    * (r[3 * k + a] - r[3 * j + a])
                                    / trial_distmat[(k, j)]
                            }
                        })
                        .sum::<f64>();

                let translation_diff_grad = 2.0
                    * trial_atoms[k].atomic_mass
                    * delta_coordinates
                        .iter()
                        .map(|delta_coords_i| delta_coords_i[a])
                        .sum::<f64>();

                let b = (a + 1).rem_euclid(3);
                let c = (a + 2).rem_euclid(3);
                let rotation_diff_grad = -2.0
                    * original_atoms[k].coordinates[b]
                    * trial_atoms[k].atomic_mass
                    * rot_coordinates
                        .iter()
                        .map(|rot_coords_i| rot_coords_i[c])
                        .sum::<f64>()
                    + 2.0
                        * original_atoms[k].coordinates[c]
                        * trial_atoms[k].atomic_mass
                        * rot_coordinates
                            .iter()
                            .map(|rot_coords_i| rot_coords_i[b])
                            .sum::<f64>();
                println!("Atom {k}, coord {a}: {} {} {}", distmat_diff_grad, translation_diff_grad, rotation_diff_grad);
                distmat_diff_grad + translation_diff_grad + rotation_diff_grad
            })
            .collect_vec();
        println!("dS: {}", dfdr.l2_norm());
        Ok(Array1::from_vec(dfdr))
    }
}
