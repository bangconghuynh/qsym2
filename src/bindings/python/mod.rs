use pyo3::prelude::*;

mod molecule_symmetrisation;
mod symmetry_group_detection;
mod representation_analysis;

/// A Python module for `QSym2` implemented in Rust.
#[pymodule]
fn qsym2(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    pyo3_log::init();
    m.add_function(wrap_pyfunction!(
        symmetry_group_detection::detect_symmetry_group,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        molecule_symmetrisation::symmetrise_molecule,
        m
    )?)?;
    Ok(())
}
