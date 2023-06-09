use nalgebra::{Point3, Vector3};
use pyo3::prelude::*;

use crate::drivers::symmetry_group_detection::{
    SymmetryGroupDetectionDriver, SymmetryGroupDetectionParams,
};
use crate::drivers::QSym2Driver;

#[pyclass]
pub(super) struct PySymmetryGroupDetectionResult {
    #[pyo3(get)]
    rotational_symmetry: String,

    #[pyo3(get)]
    unitary_group_name: String,

    #[pyo3(get)]
    magnetic_group_name: Option<String>,
}

#[pyfunction]
pub(super) fn detect_symmetry_group(
    xyz_path: String,
    moi_thresholds: Vec<f64>,
    distance_thresholds: Vec<f64>,
    time_reversal: bool,
    write_symmetry_elements: bool,
    fictitious_magnetic_field: Option<[f64; 3]>,
    fictitious_electric_field: Option<[f64; 3]>,
) -> Option<PySymmetryGroupDetectionResult> {
    let params = SymmetryGroupDetectionParams::builder()
        .distance_thresholds(&distance_thresholds)
        .moi_thresholds(&moi_thresholds)
        .time_reversal(time_reversal)
        .fictitious_magnetic_fields(
            fictitious_magnetic_field
                .map(|bs| vec![(Point3::<f64>::origin(), Vector3::new(bs[0], bs[1], bs[2]))]),
        )
        .fictitious_electric_fields(
            fictitious_electric_field
                .map(|es| vec![(Point3::<f64>::origin(), Vector3::new(es[0], es[1], es[2]))]),
        )
        .fictitious_origin_com(true)
        .write_symmetry_elements(write_symmetry_elements)
        .build()
        .ok()?;
    let mut pd_driver = SymmetryGroupDetectionDriver::builder()
        .parameters(&params)
        .xyz(Some(xyz_path))
        .build()
        .ok()?;
    pd_driver.run().ok()?;
    let pd_res = pd_driver.result().ok()?;
    let py_pd_res = PySymmetryGroupDetectionResult {
        rotational_symmetry: pd_res.pre_symmetry.rotational_symmetry.to_string(),
        unitary_group_name: pd_res
            .unitary_symmetry
            .group_name
            .as_ref()
            .expect("No unitary symmetry found.")
            .clone(),
        magnetic_group_name: pd_res
            .magnetic_symmetry
            .as_ref()
            .and_then(|magsym| magsym.group_name.as_ref())
            .cloned(),
    };
    Some(py_pd_res)
}
