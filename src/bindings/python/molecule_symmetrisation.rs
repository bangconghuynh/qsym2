use std::fs::File;
use std::io::{BufReader, BufWriter};

use bincode;
use pyo3::exceptions::{PyIOError, PyRuntimeError};
use pyo3::prelude::*;

use crate::drivers::molecule_symmetrisation::{
    MoleculeSymmetrisationDriver, MoleculeSymmetrisationParams,
};
use crate::drivers::symmetry_group_detection::{
    SymmetryGroupDetectionDriver, SymmetryGroupDetectionParams, SymmetryGroupDetectionResult,
};
use crate::drivers::QSym2Driver;

/// A Python-exposed function to perform molecule symmetrisation.
#[pyfunction]
pub(super) fn symmetrise_molecule(
    inp_loose_sym: String,
    out_tight_sym: String,
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

    let verifying_pd_params = SymmetryGroupDetectionParams::builder()
        .moi_thresholds(&[ms_params.target_moi_threshold])
        .distance_thresholds(&[ms_params.target_distance_threshold])
        .time_reversal(loose_pd_res.parameters.time_reversal)
        .write_symmetry_elements(loose_pd_res.parameters.write_symmetry_elements)
        .build()
        .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
    let mut verifying_pd_driver = SymmetryGroupDetectionDriver::builder()
        .parameters(&verifying_pd_params)
        .molecule(ms_driver.result().ok().map(|res| &res.symmetrised_molecule))
        .build()
        .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
    verifying_pd_driver
        .run()
        .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
    let verifying_pd_res = verifying_pd_driver
        .result()
        .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;

    {
        let mut writer = BufWriter::new(
            File::create(format!("{out_tight_sym}.qsym2.sym")).map_err(PyIOError::new_err)?,
        );
        bincode::serialize_into(&mut writer, &verifying_pd_res)
            .map_err(|err| PyIOError::new_err(err.to_string()))?;
    }

    // {
    //     let mut reader = BufReader::new(File::open("test.sym").unwrap());
    //     let pd_res_2: SymmetryGroupDetectionResult = bincode::deserialize_from(&mut reader).unwrap();
    //     println!("{}", pd_res_2.unitary_symmetry.group_name.unwrap());
    //     println!("{}", pd_res_2.magnetic_symmetry.unwrap().group_name.unwrap());
    // }

    // let py_pd_res = PySymmetryGroupDetectionResult {
    //     rotational_symmetry: pd_res.pre_symmetry.rotational_symmetry.to_string(),
    //     unitary_group_name: pd_res
    //         .unitary_symmetry
    //         .group_name
    //         .as_ref()
    //         .expect("No unitary symmetry found.")
    //         .clone(),
    //     magnetic_group_name: pd_res
    //         .magnetic_symmetry
    //         .as_ref()
    //         .and_then(|magsym| magsym.group_name.as_ref())
    //         .cloned(),
    // };
    Ok(())
}
