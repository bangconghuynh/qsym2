use anyhow::format_err;
use pyo3::exceptions::{PyIOError, PyRuntimeError};
use pyo3::prelude::*;

use crate::aux::atom::AtomKind;
use crate::bindings::python::symmetry_group_detection::PyMolecule;
use crate::drivers::molecule_symmetrisation::{
    MoleculeSymmetrisationDriver, MoleculeSymmetrisationParams,
};
use crate::drivers::symmetry_group_detection::SymmetryGroupDetectionResult;
use crate::drivers::QSym2Driver;
use crate::io::{read_qsym2, QSym2FileType};

/// A Python-exposed function to perform molecule symmetrisation.
///
/// # Arguments
///
/// * `inp_loose_sym` - A path to the [`QSym2FileType::Sym`] file containing the target
/// symmetry-group detection results of a molecule at loose thresholds. The symmetrisation process
/// will attempt to symmetrise this molecule to obtain the target symmetry group at tighter
/// thresholds.
/// * `out_tight_sym` - An optional path for a [`QSym2FileType::Sym`] file to be saved that
/// contains the symmetry-group detection results of the symmetrised molecule at the target tight
/// thresholds.
/// * `target_moi_threshold` - The target (tight) MoI threshold.
/// * `target_distance_threshold` - The target (tight) distance threshold.
/// * `use_magnetic_group` - A boolean indicating if the magnetic group, if present, is to be used
/// for the symmetrisation.
/// * `reorientate_molecule` - A boolean indicating if the molecule is also reoriented to align its
/// principal axes with the Cartesian axes.
/// * `max_iterations` - The maximum number of iterations for the symmetrisation process.
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
#[pyfunction]
#[pyo3(signature = (inp_loose_sym, out_tight_sym, target_moi_threshold, target_distance_threshold, use_magnetic_group, reorientate_molecule=true, max_iterations=10, verbose=0, infinite_order_to_finite=None))]
pub(super) fn symmetrise_molecule(
    inp_loose_sym: String,
    out_tight_sym: Option<String>,
    target_moi_threshold: f64,
    target_distance_threshold: f64,
    use_magnetic_group: bool,
    reorientate_molecule: bool,
    max_iterations: usize,
    verbose: u8,
    infinite_order_to_finite: Option<u32>,
) -> PyResult<PyMolecule> {
    let loose_pd_res: SymmetryGroupDetectionResult = read_qsym2(&inp_loose_sym, QSym2FileType::Sym)
        .map_err(|err| PyIOError::new_err(err.to_string()))?;

    let ms_params = MoleculeSymmetrisationParams::builder()
        .use_magnetic_group(use_magnetic_group)
        .target_moi_threshold(target_moi_threshold)
        .target_distance_threshold(target_distance_threshold)
        .reorientate_molecule(reorientate_molecule)
        .max_iterations(max_iterations)
        .verbose(verbose)
        .infinite_order_to_finite(infinite_order_to_finite)
        .symmetrised_result_save_name(out_tight_sym)
        .build()
        .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;

    let mut ms_driver = MoleculeSymmetrisationDriver::builder()
        .parameters(&ms_params)
        .target_symmetry_result(&loose_pd_res)
        .build()
        .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
    ms_driver
        .run()
        .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;

    let symmol = &ms_driver
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
                Err(format_err!(
                    "Only a uniform magnetic field is supported."
                ))
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
                    _ => Err(format_err!(
                        "Invalid fictitious magnetic atoms detected."
                    )),
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
            if elec_atoms.len() != 2 {
                Err(format_err!(
                    "Only a uniform electric field is supported."
                ))
            } else {
                match (&elec_atoms[0].kind, &elec_atoms[1].kind) {
                    (AtomKind::Magnetic(true), AtomKind::Magnetic(false)) => {
                        let evec = elec_atoms[0].coordinates - elec_atoms[1].coordinates;
                        Ok([evec[0], evec[1], evec[2]])
                    }
                    (AtomKind::Magnetic(false), AtomKind::Magnetic(true)) => {
                        let evec = elec_atoms[1].coordinates - elec_atoms[0].coordinates;
                        Ok([evec[0], evec[1], evec[2]])
                    }
                    _ => Err(format_err!(
                        "Invalid fictitious electric atoms detected."
                    )),
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
}
