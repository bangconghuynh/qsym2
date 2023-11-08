//! Driver for molecule symmetrisation in QSym².

use std::collections::HashMap;
use std::fmt;
use std::path::PathBuf;

use anyhow::{ensure, format_err};
use argmin::core::{CostFunction, Executor, Gradient};
use argmin::solver::gradientdescent::SteepestDescent;
use argmin::solver::{
    linesearch::condition::ArmijoCondition, linesearch::BacktrackingLineSearch, quasinewton::BFGS,
};
use derive_builder::Builder;
use itertools::Itertools;
use nalgebra::{Point3, Vector3};
use ndarray::{Array1, Array2, Axis, ShapeBuilder};
use num_traits::ToPrimitive;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::auxiliary::misc::HashableFloat;
use crate::auxiliary::molecule::Molecule;
use crate::drivers::symmetry_group_detection::{
    SymmetryGroupDetectionDriver, SymmetryGroupDetectionParams, SymmetryGroupDetectionResult,
};
use crate::drivers::QSym2Driver;
use crate::io::format::{log_subtitle, log_title, nice_bool, qsym2_output, QSym2Output};
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

/// Structure containing control parameters for molecule symmetrisation by distance matrix.
#[derive(Clone, Builder, Debug, Serialize, Deserialize)]
pub struct MoleculeSymmetrisationDistMatParams {
    /// The loose distance threshold for comparing values in non-symmetric distance matrices.
    #[builder(default = "1e-3")]
    #[serde(default = "default_loose_threshold")]
    pub loose_distance_threshold: f64,

    /// The target moment-of-inertia threshold for the symmetrisation, *i.e.* the symmetrised
    /// molecule will have the target symmetry group at this target moment-of-inertia threshold.
    #[builder(default = "1e-7")]
    #[serde(default = "default_tight_threshold")]
    pub target_moi_threshold: f64,

    /// The target distance threshold for the symmetrisation, *i.e.* the symmetrised molecule will
    /// have the target symmetry group at this target distance threshold.
    #[builder(default = "1e-7")]
    #[serde(default = "default_tight_threshold")]
    pub target_distance_threshold: f64,

    /// Boolean indicating if the symmetrised molecule is also reoriented to align its principal
    /// axes with the space-fixed Cartesian axes.
    ///
    /// See [`Molecule::reorientate`] for more information.
    #[builder(default = "true")]
    #[serde(default = "default_true")]
    pub reorientate_molecule: bool,

    /// The maximum number of symmetrisation iterations.
    #[builder(default = "5")]
    #[serde(default = "default_max_iterations")]
    pub max_iterations: usize,

    /// The output verbosity level.
    #[builder(default = "0")]
    #[serde(default)]
    pub verbose: u8,

    /// Optional name (without the `.xyz` extension) for writing the symmetrised molecule to an XYZ
    /// file. If `None`, no XYZ files will be written.
    #[builder(default = "None")]
    #[serde(default)]
    pub symmetrised_result_xyz: Option<PathBuf>,

    /// Optional name for saving the symmetry-group detection result of the symmetrised system as a
    /// binary file of type [`QSym2FileType::Sym`]. If `None`, the result will not be saved.
    #[builder(default = "None")]
    #[serde(default)]
    pub symmetrised_result_save_name: Option<PathBuf>,
}

impl MoleculeSymmetrisationDistMatParams {
    /// Returns a builder to construct a [`MoleculeSymmetrisationDistMatParams`] structure.
    pub fn builder() -> MoleculeSymmetrisationDistMatParamsBuilder {
        MoleculeSymmetrisationDistMatParamsBuilder::default()
    }
}

impl Default for MoleculeSymmetrisationDistMatParams {
    fn default() -> Self {
        Self::builder()
            .build()
            .expect("Unable to construct a default `MoleculeSymmetrisationDistMatParams`.")
    }
}

impl fmt::Display for MoleculeSymmetrisationDistMatParams {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "Loose distance threshold: {:.3e}",
            self.loose_distance_threshold
        )?;
        writeln!(f)?;
        writeln!(
            f,
            "Maximum symmetrisation iterations: {}",
            self.max_iterations
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

/// Structure to contain molecule symmetrisation results.
#[derive(Clone, Builder, Debug)]
pub struct MoleculeSymmetrisationDistMatResult<'a> {
    /// The control parameters used to obtain this set of molecule symmetrisation results.
    parameters: &'a MoleculeSymmetrisationDistMatParams,

    /// The symmetrised molecule.
    pub symmetrised_molecule: Molecule,
}

impl<'a> MoleculeSymmetrisationDistMatResult<'a> {
    fn builder() -> MoleculeSymmetrisationDistMatResultBuilder<'a> {
        MoleculeSymmetrisationDistMatResultBuilder::default()
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
pub struct MoleculeSymmetrisationDistMatDriver<'a> {
    /// The control parameters for molecule symmetrisation by distance matrix.
    parameters: &'a MoleculeSymmetrisationDistMatParams,

    /// The molecule to be symmetrised.
    molecule: &'a Molecule,

    /// The result of the symmetrisation.
    #[builder(setter(skip), default = "None")]
    result: Option<MoleculeSymmetrisationDistMatResult<'a>>,
}

// impl<'a> MoleculeSymmetrisationDistMatDriverBuilder<'a> {
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

impl<'a> MoleculeSymmetrisationDistMatDriver<'a> {
    /// Returns a builder to construct a [`MoleculeSymmetrisationDistMatDriver`] structure.
    pub fn builder() -> MoleculeSymmetrisationDistMatDriverBuilder<'a> {
        MoleculeSymmetrisationDistMatDriverBuilder::default()
    }

    /// Executes molecule symmetrisation.
    fn symmetrise_molecule(&mut self) -> Result<(), anyhow::Error> {
        log_title("Molecule Symmetrisation by Distance Matrix");
        qsym2_output!("");
        let params = self.parameters;
        params.log_output_display();

        let loose_dist_threshold = params.loose_distance_threshold;

        let unsymmetrised_mol = self.molecule.adjust_threshold(loose_dist_threshold);

        qsym2_output!("Unsymmetrised molecule:");
        unsymmetrised_mol.log_output_display();
        qsym2_output!("");
        // if params.reorientate_molecule {
        //     // If reorientation is requested, the trial molecule is reoriented prior to
        //     // symmetrisation, so that the symmetrisation procedure acts on the reoriented molecule
        //     // itself. The molecule might become disoriented during the symmetrisation process, but
        //     // any such disorientation is likely to be fairly small, and post-symmetrisation
        //     // corrections on small disorientation are better than on large disorientation.
        //     trial_mol.reorientate_mut(tight_moi_threshold);
        //     qsym2_output!("Unsymmetrised recentred and reoriented molecule:");
        //     trial_mol.log_output_display();
        //     qsym2_output!("");
        // };

        // if params.verbose >= 1 {
        //     let orig_mol = self
        //         .target_symmetry_result
        //         .pre_symmetry
        //         .original_molecule
        //         .adjust_threshold(tight_dist_threshold);
        //     qsym2_output!("Unsymmetrised original molecule:");
        //     orig_mol.log_output_display();
        //     qsym2_output!("");
        //
        //     qsym2_output!("Unsymmetrised recentred molecule:");
        //     trial_mol.log_output_display();
        //     qsym2_output!("");
        // }

        let original_atoms = unsymmetrised_mol.get_all_atoms();
        let n_atoms = original_atoms.len();

        let (unsymmetrised_sorted_distmat, sort_indicess, equiv_col_indicess) =
            unsymmetrised_mol.calc_interatomic_distance_matrix();
        println!("n seas: {}", equiv_col_indicess.len());

        let unsorted_indicess = sort_indicess
            .iter()
            .map(|sorted_indices| {
                let mut unsorted_indices = (0..sorted_indices.len()).collect_vec();
                unsorted_indices.sort_by_key(|&i| sorted_indices[i]);
                unsorted_indices
            })
            .collect_vec();

        let unsymmetrised_distmat = Array2::from_shape_vec(
            (n_atoms, n_atoms).f(),
            unsymmetrised_sorted_distmat
                .columns()
                .into_iter()
                .enumerate()
                .flat_map(|(j, col)| col.select(Axis(0), &unsorted_indicess[j]))
                .collect_vec(),
        )?;
        println!("Sorted:\n {unsymmetrised_sorted_distmat}");
        // println!("Unsorted:\n {unsymmetrised_distmat}");

        // Symmetrise `distmat` based on `self.loose_distance_threshold`.
        let mut symmetrised_sorted_distmat = unsymmetrised_sorted_distmat.clone();
        equiv_col_indicess.iter().for_each(|equiv_col_indices| {
            // Consider each SEA group
            // let hashable_dist_col = unsymmetrised_sorted_distmat
            //     .column(equiv_col_indices[0])
            //     .map(|dist| dist.round_factor(loose_dist_threshold).integer_decode());
            // let mut equiv_row_indicess: HashMap<&(u64, i16, i8), Vec<usize>> = HashMap::new();
            // hashable_dist_col.indexed_iter().for_each(|(index, dist)| {
            //     equiv_row_indicess
            //         .entry(dist)
            //         .and_modify(|indices| indices.push(index))
            //         .or_insert(vec![index]);
            // });
            let equiv_row_indicess = unsymmetrised_sorted_distmat
                .column(equiv_col_indices[0])
                .indexed_iter()
                .tuple_windows()
                .fold(vec![vec![0]], |mut acc, ((_, disti), (j, distj))| {
                    if approx::relative_eq!(
                        disti,
                        distj,
                        epsilon = loose_dist_threshold,
                        max_relative = loose_dist_threshold
                    ) {
                        let n = acc.len();
                        acc[n - 1].push(j);
                    } else {
                        acc.push(vec![j]);
                    }
                    acc
                });

            equiv_row_indicess.iter().for_each(|equiv_row_indices| {
                let n_elements = (equiv_row_indices.len() * equiv_col_indices.len())
                    .to_f64()
                    .expect("Unable to convert the number of averaging distances to `f64`.");
                let ave_dist = unsymmetrised_sorted_distmat
                    .select(Axis(0), equiv_row_indices)
                    .select(Axis(1), equiv_col_indices)
                    .iter()
                    .sum::<f64>()
                    / n_elements;
                equiv_row_indices
                    .iter()
                    .cartesian_product(equiv_col_indices.iter())
                    .for_each(|(row_index, col_index)| {
                        symmetrised_sorted_distmat
                            .get_mut((*row_index, *col_index))
                            .map(|v| *v = ave_dist);
                    });
            });
        });
        println!("Symmetrised sorted:\n {symmetrised_sorted_distmat}");
        let symmetrised_distmat = Array2::from_shape_vec(
            (n_atoms, n_atoms).f(),
            symmetrised_sorted_distmat
                .columns()
                .into_iter()
                .enumerate()
                .flat_map(|(j, col)| col.select(Axis(0), &unsorted_indicess[j]))
                .collect_vec(),
        )?;
        // println!("Symmetrised unsorted:\n {symmetrised_distmat}");

        let r0 = Array1::from_vec(
            original_atoms
                .iter()
                .flat_map(|atom| atom.coordinates.iter())
                .cloned()
                .collect_vec(),
        );

        let problem = MoleculeSymmetrisationDistMatProblem {
            symmetrised_distmat,
            symmetrised_sorted_distmat,
            unsymmetrised_mol,
        };

        let linesearch = BacktrackingLineSearch::<Array1<f64>, Array1<f64>, _, f64>::new(
            ArmijoCondition::new(1e-6).unwrap(),
        );
        let solver: BFGS<_, f64> = BFGS::new(linesearch);
        let res = Executor::new(problem, solver)
            .configure(|state| {
                state
                    .param(r0)
                    .inv_hessian(Array2::<f64>::eye(3 * n_atoms))
                    .target_cost(1e-9)
                    .max_iters(100)
            })
            .run()?;
        println!("{}", res);

        // let r = bfgs(r0, f, g)?;
        // println!("{r}");

        Ok(())
    }
}

impl<'a> QSym2Driver for MoleculeSymmetrisationDistMatDriver<'a> {
    type Params = MoleculeSymmetrisationDistMatParams;

    type Outcome = MoleculeSymmetrisationDistMatResult<'a>;

    fn result(&self) -> Result<&Self::Outcome, anyhow::Error> {
        self.result
            .as_ref()
            .ok_or_else(|| format_err!("No molecule symmetrisation results found."))
    }

    fn run(&mut self) -> Result<(), anyhow::Error> {
        self.symmetrise_molecule()
    }
}

struct MoleculeSymmetrisationDistMatProblem {
    symmetrised_distmat: Array2<f64>,
    symmetrised_sorted_distmat: Array2<f64>,
    unsymmetrised_mol: Molecule,
}

impl CostFunction for MoleculeSymmetrisationDistMatProblem {
    type Param = Array1<f64>;
    type Output = f64;

    fn cost(&self, r: &Self::Param) -> Result<Self::Output, argmin::core::Error> {
        let original_atoms = self.unsymmetrised_mol.get_all_atoms();
        let n_atoms = original_atoms.len();

        let mut trial_mol = self.unsymmetrised_mol.clone();
        debug_assert_eq!(r.len(), 3 * n_atoms);
        trial_mol
            .get_all_atoms_mut()
            .iter_mut()
            .zip(r.iter().chunks(3).into_iter())
            .for_each(|(atom, trial_coords)| {
                let trial_coords = trial_coords.into_iter().collect_vec();
                atom.coordinates =
                    Point3::new(*trial_coords[0], *trial_coords[1], *trial_coords[2]);
            });
        let (trial_sorted_distmat, _, _) = trial_mol.calc_interatomic_distance_matrix();

        let distmat_diff = (trial_sorted_distmat - &self.symmetrised_sorted_distmat)
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
        Ok(distmat_diff + translation_diff + rotation_diff)
    }
}

impl Gradient for MoleculeSymmetrisationDistMatProblem {
    type Param = Array1<f64>;
    type Gradient = Array1<f64>;

    fn gradient(&self, r: &Self::Param) -> Result<Self::Gradient, argmin::core::Error> {
        let original_atoms = self.unsymmetrised_mol.get_all_atoms();
        let mut trial_mol = self.unsymmetrised_mol.clone();
        let n_atoms = trial_mol.get_all_atoms().len();
        debug_assert_eq!(r.len(), 3 * n_atoms);
        trial_mol
            .get_all_atoms_mut()
            .iter_mut()
            .zip(r.iter().chunks(3).into_iter())
            .for_each(|(atom, trial_coords)| {
                let trial_coords = trial_coords.into_iter().collect_vec();
                atom.coordinates =
                    Point3::new(*trial_coords[0], *trial_coords[1], *trial_coords[2]);
            });

        let trial_atoms = trial_mol.get_all_atoms();

        let (trial_sorted_distmat, trial_sort_indicess, _) =
            trial_mol.calc_interatomic_distance_matrix();
        let trial_unsorted_indicess = trial_sort_indicess
            .iter()
            .map(|sorted_indices| {
                let mut unsorted_indices = (0..sorted_indices.len()).collect_vec();
                unsorted_indices.sort_by_key(|&i| sorted_indices[i]);
                unsorted_indices
            })
            .collect_vec();

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

        let trial_distmat = Array2::from_shape_vec(
            (n_atoms, n_atoms).f(),
            trial_sorted_distmat
                .columns()
                .into_iter()
                .enumerate()
                .flat_map(|(j, col)| col.select(Axis(0), &trial_unsorted_indicess[j]))
                .collect_vec(),
        )
        .expect("Unable to construct the unsorted trial distance matrix.");
        let dfdr = (0..n_atoms)
            .cartesian_product(0..3)
            .map(|(k, a)| {
                let distmat_diff_grad = 2.0
                    * (0..n_atoms)
                        .map(|j| {
                            if j == k {
                                0.0
                            } else {
                                (trial_distmat[(k, j)] - self.symmetrised_distmat[(k, j)])
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
                let pb = if b.rem_euclid(2) == 0 { 1.0 } else { -1.0 };
                let c = (a + 2).rem_euclid(3);
                let pc = if c.rem_euclid(2) == 0 { 1.0 } else { -1.0 };
                let rotation_diff_grad = -2.0
                    * original_atoms[k].coordinates[b]
                    * trial_atoms[k].atomic_mass
                    * rot_coordinates
                        .iter()
                        .map(|rot_coords_i| rot_coords_i[c])
                        .sum::<f64>()
                    * pc
                    - 2.0
                        * original_atoms[k].coordinates[c]
                        * trial_atoms[k].atomic_mass
                        * rot_coordinates
                            .iter()
                            .map(|rot_coords_i| rot_coords_i[b])
                            .sum::<f64>()
                        * pb;
                distmat_diff_grad + translation_diff_grad + rotation_diff_grad
            })
            .collect_vec();
        Ok(Array1::from_vec(dfdr))
    }
}
