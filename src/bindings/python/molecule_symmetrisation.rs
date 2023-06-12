use std::fs::File;
use std::io::BufReader;

use bincode;
use pyo3::exceptions::{PyIOError, PyRuntimeError};
use pyo3::prelude::*;

use crate::drivers::molecule_symmetrisation::{
    MoleculeSymmetrisationDriver, MoleculeSymmetrisationParams,
};
use crate::drivers::symmetry_group_detection::SymmetryGroupDetectionResult;
use crate::drivers::QSym2Driver;

/// A Python-exposed function to perform molecule symmetrisation.
#[pyfunction]
#[pyo3(signature = (inp_loose_sym, out_tight_sym, use_magnetic_group, target_moi_threshold, target_distance_threshold, reorientate_molecule, max_iterations, verbose, infinite_order_to_finite=None))]
pub(super) fn symmetrise_molecule(
    inp_loose_sym: String,
    out_tight_sym: Option<String>,
    use_magnetic_group: bool,
    target_moi_threshold: f64,
    target_distance_threshold: f64,
    reorientate_molecule: bool,
    max_iterations: usize,
    verbose: u8,
    infinite_order_to_finite: Option<u32>,
) -> PyResult<()> {
    let loose_pd_res: SymmetryGroupDetectionResult = {
        let mut reader = BufReader::new(
            File::open(format!("{inp_loose_sym}.qsym2.sym")).map_err(PyIOError::new_err)?,
        );
        bincode::deserialize_from(&mut reader).map_err(|err| PyIOError::new_err(err.to_string()))?
    };

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
    Ok(())
}
