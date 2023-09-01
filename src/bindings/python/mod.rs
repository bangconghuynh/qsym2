use pyo3::prelude::*;

pub mod integrals;
pub mod molecule_symmetrisation;
pub mod representation_analysis;
pub mod symmetry_group_detection;

use crate::analysis::EigenvalueComparisonMode;
use crate::drivers::representation_analysis::MagneticSymmetryAnalysisKind;
use crate::interfaces::cli::{qsym2_output_heading, qsym2_output_contributors};
use crate::symmetry::symmetry_transformation::SymmetryTransformationKind;

/// A Python module for `QSym2` implemented in Rust.
#[pymodule]
pub fn qsym2(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    pyo3_log::init();
    // -------
    // Version
    // -------
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    // ---------
    // Functions
    // ---------
    m.add_function(wrap_pyfunction!(
        qsym2_output_heading,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        qsym2_output_contributors,
        m
    )?)?;
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
    #[cfg(feature = "integrals")]
    m.add_function(wrap_pyfunction!(integrals::calc_overlap_4c_real, m)?)?;
    #[cfg(feature = "integrals")]
    m.add_function(wrap_pyfunction!(integrals::calc_overlap_4c_complex, m)?)?;

    // -------
    // Classes
    // -------
    m.add_class::<integrals::PyBasisAngularOrder>()?;
    m.add_class::<integrals::PySpinConstraint>()?;
    #[cfg(feature = "integrals")]
    m.add_class::<integrals::PyBasisShellContraction>()?;
    m.add_class::<symmetry_group_detection::PyMolecule>()?;
    m.add_class::<symmetry_group_detection::PySymmetry>()?;
    m.add_class::<symmetry_group_detection::PySymmetryElementKind>()?;
    m.add_class::<representation_analysis::PySlaterDeterminantReal>()?;
    m.add_class::<representation_analysis::PySlaterDeterminantComplex>()?;
    m.add_class::<EigenvalueComparisonMode>()?;
    m.add_class::<MagneticSymmetryAnalysisKind>()?;
    m.add_class::<SymmetryTransformationKind>()?;
    Ok(())
}
