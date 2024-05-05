//! Python bindings for QSym².

use pyo3::prelude::*;

pub mod integrals;
pub mod molecule_symmetrisation;
pub mod representation_analysis;
pub mod symmetry_group_detection;

use crate::analysis::EigenvalueComparisonMode;
use crate::drivers::representation_analysis::MagneticSymmetryAnalysisKind;
use crate::interfaces::cli::{qsym2_output_contributors, qsym2_output_heading};
use crate::symmetry::symmetry_transformation::SymmetryTransformationKind;

#[cfg(feature = "sandbox")]
use crate::sandbox::bindings::python::register_sandbox_module;

/// Python module for QSym² implemented in Rust.
#[pymodule]
pub fn qsym2(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    // --------------
    // Python logging
    // --------------
    pyo3_log::init();

    // -------
    // Version
    // -------
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    // ---------
    // Functions
    // ---------
    m.add_function(wrap_pyfunction!(qsym2_output_heading, m)?)?;
    m.add_function(wrap_pyfunction!(qsym2_output_contributors, m)?)?;
    m.add_function(wrap_pyfunction!(
        symmetry_group_detection::detect_symmetry_group,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        molecule_symmetrisation::symmetrise_molecule,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        representation_analysis::density::rep_analyse_densities,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        representation_analysis::slater_determinant::rep_analyse_slater_determinant,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        representation_analysis::multideterminant::rep_analyse_multideterminants_orbit_basis,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        representation_analysis::vibrational_coordinate::rep_analyse_vibrational_coordinate_collection,
        m
    )?)?;
    #[cfg(feature = "integrals")]
    m.add_function(wrap_pyfunction!(integrals::calc_overlap_2c_real, m)?)?;
    #[cfg(feature = "integrals")]
    m.add_function(wrap_pyfunction!(integrals::calc_overlap_2c_complex, m)?)?;
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
    m.add_class::<representation_analysis::density::PyDensityReal>()?;
    m.add_class::<representation_analysis::density::PyDensityComplex>()?;
    m.add_class::<representation_analysis::slater_determinant::PySlaterDeterminantReal>()?;
    m.add_class::<representation_analysis::slater_determinant::PySlaterDeterminantComplex>()?;
    m.add_class::<representation_analysis::vibrational_coordinate::PyVibrationalCoordinateCollectionReal>()?;
    m.add_class::<representation_analysis::vibrational_coordinate::PyVibrationalCoordinateCollectionComplex>()?;
    m.add_class::<EigenvalueComparisonMode>()?;
    m.add_class::<MagneticSymmetryAnalysisKind>()?;
    m.add_class::<SymmetryTransformationKind>()?;

    // ----------
    // Submodules
    // ----------
    #[cfg(feature = "sandbox")]
    register_sandbox_module(_py, m)?;

    Ok(())
}
