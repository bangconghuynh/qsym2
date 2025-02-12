//! Sandbox Python bindings for QSym².

use pyo3::prelude::*;

pub mod representation_analysis;

pub(crate) fn register_sandbox_module(
    py: Python<'_>,
    parent_module: Bound<'_, PyModule>,
) -> PyResult<()> {
    let sandbox_module = PyModule::new(py, "sandbox")?;

    // ---------
    // Functions
    // ---------
    sandbox_module.add_function(wrap_pyfunction!(
        representation_analysis::real_space_function::rep_analyse_real_space_function_real,
        &sandbox_module
    )?)?;
    sandbox_module.add_function(wrap_pyfunction!(
        representation_analysis::real_space_function::rep_analyse_real_space_function_complex,
        &sandbox_module
    )?)?;

    // ------------
    // Registration
    // ------------
    parent_module.add_submodule(&sandbox_module)?;
    Ok(())
}
