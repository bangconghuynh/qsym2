
use numpy::{IntoPyArray, PyArray2};
use pyo3::prelude::*;
use anyhow::Error as AnyhowError;
use pyo3::exceptions::PyRuntimeError;

use crate::symmetry::symmetry_element::symmetry_operation::{
    wigner_matrix_time_reversal,
    wigner_matrix_ry_minus_pi_su2_true,
};

#[pyfunction] // TODO anderer umgang mit errors
pub fn py_wigner_matrix_time_reversal(
    py: Python<'_>,
    two_j: u32,
) -> PyResult<Bound<'_, PyArray2<f64>>> {
    let m = wigner_matrix_time_reversal(two_j)
        .map_err(|e: AnyhowError| PyRuntimeError::new_err(e.to_string()))?;

    Ok(m.into_pyarray(py))
}

#[pyfunction]
pub fn py_wigner_matrix_ry_minus_pi_su2_true(
    py: Python<'_>,
    two_j: u32,
) -> PyResult<Bound<'_, PyArray2<f64>>> {
    let m = wigner_matrix_ry_minus_pi_su2_true(two_j)
        .map_err(|e: AnyhowError| PyRuntimeError::new_err(e.to_string()))?;

    Ok(m.into_pyarray(py))
}