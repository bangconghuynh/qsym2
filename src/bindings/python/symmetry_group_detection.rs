use nalgebra::{Point3, Vector3};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use crate::drivers::symmetry_group_detection::{
    SymmetryGroupDetectionDriver, SymmetryGroupDetectionParams,
};
use crate::drivers::QSym2Driver;

/// A Python-exposed function to perform symmetry-group detection.
#[pyfunction]
#[pyo3(signature = (inp_xyz, out_sym, moi_thresholds, distance_thresholds, time_reversal, write_symmetry_elements, magnetic_field, electric_field))]
pub(super) fn detect_symmetry_group(
    inp_xyz: String,
    out_sym: Option<String>,
    moi_thresholds: Vec<f64>,
    distance_thresholds: Vec<f64>,
    time_reversal: bool,
    write_symmetry_elements: bool,
    magnetic_field: Option<[f64; 3]>,
    electric_field: Option<[f64; 3]>,
) -> PyResult<()> {
    let params = SymmetryGroupDetectionParams::builder()
        .distance_thresholds(&distance_thresholds)
        .moi_thresholds(&moi_thresholds)
        .time_reversal(time_reversal)
        .magnetic_fields(
            magnetic_field
                .map(|bs| vec![(Point3::<f64>::origin(), Vector3::new(bs[0], bs[1], bs[2]))]),
        )
        .electric_fields(
            electric_field
                .map(|es| vec![(Point3::<f64>::origin(), Vector3::new(es[0], es[1], es[2]))]),
        )
        .field_origin_com(true)
        .write_symmetry_elements(write_symmetry_elements)
        .result_save_name(out_sym)
        .build()
        .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
    let mut pd_driver = SymmetryGroupDetectionDriver::builder()
        .parameters(&params)
        .xyz(Some(inp_xyz))
        .build()
        .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
    pd_driver
        .run()
        .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
    Ok(())
}
