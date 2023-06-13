use pyo3::prelude::*;

mod molecule_symmetrisation;
mod symmetry_group_detection;
mod representation_analysis;

use crate::symmetry::symmetry_transformation::SymmetryTransformationKind;

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
    m.add_function(wrap_pyfunction!(
        representation_analysis::rep_analyse_slater_determinant,
        m
    )?)?;
    m.add_class::<symmetry_group_detection::PyMolecule>()?;
    m.add_class::<representation_analysis::PyBasisAngularOrder>()?;
    m.add_class::<representation_analysis::PySpinConstraint>()?;
    m.add_class::<representation_analysis::PySlaterDeterminantReal>()?;
    m.add_class::<representation_analysis::PySlaterDeterminantComplex>()?;
    m.add_class::<SymmetryTransformationKind>()?;
    Ok(())
}
