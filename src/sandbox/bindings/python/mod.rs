//! Sandbox Python bindings for QSym².

use pyo3::prelude::*;

mod representation_analysis;

pub(crate) fn register_sandbox_module(py: Python<'_>, parent_module: &PyModule) -> PyResult<()> {
    let sandbox_module = PyModule::new(py, "sandbox")?;

    // ---------
    // Functions
    // ---------
    sandbox_module.add_function(wrap_pyfunction!(
        representation_analysis::pes::rep_analyse_pes_real,
        sandbox_module
    )?)?;
    sandbox_module.add_function(wrap_pyfunction!(
        representation_analysis::pes::rep_analyse_pes_complex,
        sandbox_module
    )?)?;

    // ------------
    // Registration
    // ------------
    parent_module.add_submodule(sandbox_module)?;
    Ok(())
}
