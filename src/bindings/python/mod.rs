use pyo3::prelude::*;

mod symmetry_group_detection;

/// A Python module for `QSym2` implemented in Rust.
#[pymodule]
fn qsym2(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    pyo3_log::init();
    m.add_function(wrap_pyfunction!(
        symmetry_group_detection::detect_symmetry_group,
        m
    )?)?;
    m.add_class::<symmetry_group_detection::PySymmetryGroupDetectionResult>()?;
    Ok(())
}
