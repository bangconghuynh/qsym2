//! Python bindings for QSym² molecule symmetrisation by bootstrapping.
//!
//! See [`crate::drivers::molecule_symmetrisation_bootstrap`] for more information.

use std::path::PathBuf;

use anyhow::format_err;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use crate::auxiliary::atom::AtomKind;
use crate::auxiliary::molecule::Molecule;
use crate::bindings::python::symmetry_group_detection::PyMolecule;
use crate::drivers::QSym2Driver;
use crate::drivers::molecule_symmetrisation_bootstrap::{
    MoleculeSymmetrisationBootstrapDriver, MoleculeSymmetrisationBootstrapParams,
};
use crate::io::QSym2FileType;

/// Python-exposed function to perform molecule symmetrisation by bootstrapping and log the result
/// via the `qsym2-output` logger at the `INFO` level.
///
/// See [`crate::drivers::molecule_symmetrisation_bootstrap`] for more information.
///
/// # Arguments
///
/// * `inp_xyz` - An optional string providing the path to an XYZ file containing the molecule to
/// be symmetrised. Only one of `inp_xyz` or `inp_mol` can be specified.
/// * `inp_mol` - An optional `PyMolecule` structure containing the molecule to be symmetrised. Only
/// one of `inp_xyz` or `inp_mol` can be specified.
/// * `out_target_sym` - An optional path for a [`QSym2FileType::Sym`] file to be saved that
/// contains the symmetry-group detection results of the symmetrised molecule at the target
/// thresholds.
/// * `loose_moi_threshold` - The loose MoI threshold.
/// * `loose_distance_threshold` - The loose distance threshold.
/// * `target_moi_threshold` - The target (tight) MoI threshold.
/// * `target_distance_threshold` - The target (tight) distance threshold.
/// * `use_magnetic_group` - A boolean indicating if the magnetic group (*i.e.* the group including
/// time-reversed operations) is to be used for the symmetrisation.
/// * `reorientate_molecule` - A boolean indicating if the molecule is also reoriented to align its
/// principal axes with the Cartesian axes.
/// * `max_iterations` - The maximum number of iterations for the symmetrisation process.
/// * `consistent_target_symmetry_iterations` - The number of consecutive iterations during which
/// the symmetry group at the target level of threshold must be consistently found for convergence
/// to be reached, if this group cannot become identical to the symmetry group at the loose level
/// of threshold.
/// * `verbose` - The print-out level.
/// * `infinite_order_to_finite` - The finite order with which infinite-order generators are to be
/// interpreted to form a finite subgroup of the prevailing infinite group. This finite subgroup
/// will be used for the symmetrisation.
///
/// # Returns
///
/// The symmetrised molecule.
///
/// # Errors
///
/// Errors if any intermediate step in the symmetrisation procedure fails.
#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (
    inp_xyz,
    inp_mol,
    out_target_sym=None,
    loose_moi_threshold=1e-2,
    loose_distance_threshold=1e-2,
    target_moi_threshold=1e-7,
    target_distance_threshold=1e-7,
    use_magnetic_group=false,
    reorientate_molecule=true,
    max_iterations=50,
    consistent_target_symmetry_iterations=10,
    verbose=0,
    infinite_order_to_finite=None
))]
pub fn symmetrise_molecule(
    py: Python<'_>,
    inp_xyz: Option<PathBuf>,
    inp_mol: Option<PyMolecule>,
    out_target_sym: Option<PathBuf>,
    loose_moi_threshold: f64,
    loose_distance_threshold: f64,
    target_moi_threshold: f64,
    target_distance_threshold: f64,
    use_magnetic_group: bool,
    reorientate_molecule: bool,
    max_iterations: usize,
    consistent_target_symmetry_iterations: usize,
    verbose: u8,
    infinite_order_to_finite: Option<u32>,
) -> PyResult<PyMolecule> {
    py.detach(|| {
        let mol = match (inp_xyz, inp_mol) {
            (Some(xyz_path), None) => Molecule::from_xyz(xyz_path, 1e-7),
            (None, Some(pymol)) => Molecule::from(pymol),
            _ => {
                return Err(PyRuntimeError::new_err(
                    "One and only one of `inp_xyz` or `inp_mol` must be specified.",
                ));
            }
        };

        let msb_params = MoleculeSymmetrisationBootstrapParams::builder()
            .reorientate_molecule(reorientate_molecule)
            .use_magnetic_group(use_magnetic_group)
            .loose_moi_threshold(loose_moi_threshold)
            .loose_distance_threshold(loose_distance_threshold)
            .target_moi_threshold(target_moi_threshold)
            .target_distance_threshold(target_distance_threshold)
            .infinite_order_to_finite(infinite_order_to_finite)
            .max_iterations(max_iterations)
            .consistent_target_symmetry_iterations(consistent_target_symmetry_iterations)
            .verbose(verbose)
            .symmetrised_result_save_name(out_target_sym)
            .build()
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;

        let mut msb_driver = MoleculeSymmetrisationBootstrapDriver::builder()
            .parameters(&msb_params)
            .molecule(&mol)
            .build()
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
        msb_driver
            .run()
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;

        let symmol = &msb_driver
            .result()
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))?
            .symmetrised_molecule;

        // Note that the magnetic field here will not have the original magnitude as it is back-deduded
        // from the magnetic atoms which have been added using the normalised version of the original
        // magnetic field.
        let magnetic_field = symmol
            .magnetic_atoms
            .as_ref()
            .map(|mag_atoms| {
                if mag_atoms.len() != 2 {
                    Err(format_err!("Only a uniform magnetic field is supported."))
                } else {
                    match (&mag_atoms[0].kind, &mag_atoms[1].kind) {
                        (AtomKind::Magnetic(true), AtomKind::Magnetic(false)) => {
                            let bvec = mag_atoms[0].coordinates - mag_atoms[1].coordinates;
                            Ok([bvec[0], bvec[1], bvec[2]])
                        }
                        (AtomKind::Magnetic(false), AtomKind::Magnetic(true)) => {
                            let bvec = mag_atoms[1].coordinates - mag_atoms[0].coordinates;
                            Ok([bvec[0], bvec[1], bvec[2]])
                        }
                        _ => Err(format_err!("Invalid fictitious magnetic atoms detected.")),
                    }
                }
            })
            .transpose()
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;

        // Note that the electric field here will not have the original magnitude as it is back-deduded
        // from the electric atoms which have been added using the normalised version of the original
        // electric field.
        let electric_field = symmol
            .electric_atoms
            .as_ref()
            .map(|elec_atoms| {
                if elec_atoms.len() != 1 {
                    Err(format_err!("Only a uniform electric field is supported."))
                } else {
                    match &elec_atoms[0].kind {
                        AtomKind::Electric(pos) => {
                            let evec = if *pos {
                                elec_atoms[0].coordinates
                            } else {
                                -elec_atoms[0].coordinates
                            };
                            Ok([evec[0], evec[1], evec[2]])
                        }
                        _ => Err(format_err!("Invalid fictitious electric atoms detected.")),
                    }
                }
            })
            .transpose()
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;

        let pymol = PyMolecule::new(
            symmol
                .atoms
                .iter()
                .map(|atom| {
                    (
                        atom.atomic_symbol.clone(),
                        [
                            atom.coordinates[0],
                            atom.coordinates[1],
                            atom.coordinates[2],
                        ],
                    )
                })
                .collect::<Vec<_>>(),
            symmol.threshold,
            magnetic_field,
            electric_field,
        );
        Ok(pymol)
    })
}
