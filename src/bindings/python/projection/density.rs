//! Python bindings for QSym² symmetry projection of electron densities.

use std::path::PathBuf;

use indexmap::IndexMap;
use itertools::Itertools;
use num_complex::Complex;
use pyo3::exceptions::{PyIOError, PyRuntimeError};
use pyo3::{IntoPyObjectExt, prelude::*};

use crate::bindings::python::integrals::PyBasisAngularOrder;
use crate::bindings::python::projection::PyProjectionTarget;
use crate::bindings::python::representation_analysis::density::{
    PyDensity, PyDensityComplex, PyDensityReal,
};
use crate::drivers::QSym2Driver;
use crate::drivers::projection::density::{DensityProjectionDriver, DensityProjectionParams};
use crate::drivers::representation_analysis::{
    CharacterTableDisplay, MagneticSymmetryAnalysisKind,
};
use crate::drivers::symmetry_group_detection::SymmetryGroupDetectionResult;
use crate::io::format::qsym2_output;
use crate::io::{QSym2FileType, read_qsym2_binary};
use crate::symmetry::symmetry_group::UnitaryRepresentedSymmetryGroup;
use crate::symmetry::symmetry_transformation::SymmetryTransformationKind;
use crate::target::density::Density;

type C128 = Complex<f64>;

// =====================
// Functions definitions
// =====================

/// Python-exposed function to perform symmetry projection for real and complex electron densities.
///
/// # Arguments
///
/// * `inp_sym` - A path to the [`QSym2FileType::Sym`] file containing the symmetry-group detection
/// result for the system. This will be used to construct abstract groups and character tables for
/// symmetry projection. Python type: `str`.
/// * `pydens` - A sequence of Python-exposed electron densities whose density matrices are of type
/// `float64` or `complex128`. Each density is accompanied by a description string.
/// Python type: `list[tuple[str, PyDensityReal | PyDensityComplex]]`.
/// * `projection_targets` - A sequence of subspace labels for projection. Each label is either a
/// symbolic string or a numerical index for the subspace in the character table of the prevailing
/// group. Python type: `list[str | int]`.
/// * `pybao` - Python-exposed structure containing basis angular order information for the density
/// matrices. Python type: `PyBasisAngularOrder`.
/// * `use_magnetic_group` - An option indicating if the magnetic group is to be used for symmetry
/// analysis, and if so, whether unitary representations or unitary-antiunitary corepresentations
/// should be used. Python type: `None | MagneticSymmetryAnalysisKind`.
/// * `use_double_group` - A boolean indicating if the double group of the prevailing symmetry
/// group is to be used for representation analysis instead. Python type: `bool`.
/// * `symmetry_transformation_kind` - An enumerated type indicating the type of symmetry
/// transformations to be performed on the origin electron density to generate the orbit. Python
/// type: `SymmetryTransformationKind`.
/// * `write_character_table` - A boolean indicating if the character table of the prevailing
/// symmetry group is to be printed out. Python type: `bool`.
/// * `infinite_order_to_finite` - The finite order with which infinite-order generators are to be
/// interpreted to form a finite subgroup of the prevailing infinite group. This finite subgroup
/// will be used for symmetry analysis. Python type: `None | int`.
///
/// # Returns
///
/// The result will be returned as a list of tuples, each of which contains the name/description of
/// an original density and a dictionary in which the keys are the subspace labels and the values
/// are the corresponding projected density.
/// Python type: `list[tuple[str, dict[str, PyDensityReal | PyDensityComplex]]]`
#[pyfunction]
#[pyo3(signature = (
    inp_sym,
    pydens,
    projection_targets,
    pybao,
    use_magnetic_group,
    use_double_group,
    symmetry_transformation_kind,
    write_character_table=true,
    infinite_order_to_finite=None,
))]
pub fn project_densities(
    py: Python<'_>,
    inp_sym: PathBuf,
    pydens: Vec<(String, PyDensity)>,
    projection_targets: Vec<PyProjectionTarget>,
    pybao: &PyBasisAngularOrder,
    use_magnetic_group: Option<MagneticSymmetryAnalysisKind>,
    use_double_group: bool,
    symmetry_transformation_kind: SymmetryTransformationKind,
    write_character_table: bool,
    infinite_order_to_finite: Option<u32>,
) -> PyResult<Vec<(String, IndexMap<String, Py<PyAny>>)>> {
    let pd_res: SymmetryGroupDetectionResult =
        read_qsym2_binary(inp_sym.clone(), QSym2FileType::Sym)
            .map_err(|err| PyIOError::new_err(err.to_string()))?;

    let mut file_name = inp_sym.to_path_buf();
    file_name.set_extension(QSym2FileType::Sym.ext());
    qsym2_output!(
        "Symmetry-group detection results read in from {}.",
        file_name.display(),
    );
    qsym2_output!("");

    let mol = &pd_res.pre_symmetry.recentred_molecule;
    let bao = pybao
        .to_qsym2(mol)
        .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;

    let symbolic_projection_targets = projection_targets
        .iter()
        .filter_map(|pt| match pt {
            PyProjectionTarget::Symbolic(symbolic_pt) => Some(symbolic_pt.clone()),
            PyProjectionTarget::Numeric(_) => None,
        })
        .collect_vec();
    let numeric_projection_targets = projection_targets
        .iter()
        .filter_map(|pt| match pt {
            PyProjectionTarget::Symbolic(_) => None,
            PyProjectionTarget::Numeric(numeric_pt) => Some(*numeric_pt),
        })
        .collect_vec();

    let dp_params = DensityProjectionParams::builder()
        .symbolic_projection_targets(Some(symbolic_projection_targets))
        .numeric_projection_targets(Some(numeric_projection_targets))
        .use_magnetic_group(use_magnetic_group.clone())
        .use_double_group(use_double_group)
        .symmetry_transformation_kind(symmetry_transformation_kind)
        .write_character_table(if write_character_table {
            Some(CharacterTableDisplay::Symbolic)
        } else {
            None
        })
        .infinite_order_to_finite(infinite_order_to_finite)
        .build()
        .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;

    let any_complex = pydens
        .iter()
        .any(|(_, pyden)| matches!(pyden, PyDensity::Complex(_)));

    let projected_densities = if !any_complex {
        // All density matrices are real.
        let dens = pydens
            .iter()
            .map(|(_, pyden)| match pyden {
                PyDensity::Real(pyden_r) => pyden_r.to_qsym2(&bao, mol),
                PyDensity::Complex(_) => panic!("Unexpected complex density."),
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
        let dens_ref = dens
            .iter()
            .zip(pydens.iter())
            .map(|(den, (desc, _))| (desc.clone(), den))
            .collect::<Vec<_>>();

        let projected_densities = match &use_magnetic_group {
            Some(MagneticSymmetryAnalysisKind::Corepresentation) => {
                return Err(PyRuntimeError::new_err(
                    "Projection using corepresentations is not yet supported.",
                ));
            }
            Some(MagneticSymmetryAnalysisKind::Representation) | None => {
                let mut dp_driver =
                    DensityProjectionDriver::<UnitaryRepresentedSymmetryGroup, f64>::builder()
                        .parameters(&dp_params)
                        .densities(dens_ref)
                        .symmetry_group(&pd_res)
                        .build()
                        .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
                py.allow_threads(|| {
                    dp_driver
                        .run()
                        .map_err(|err| PyRuntimeError::new_err(err.to_string()))
                })?;
                dp_driver
                    .result()
                    .map_err(|err| PyRuntimeError::new_err(err.to_string()))?
                    .projected_densities()
                    .clone()
            }
        };
        projected_densities
            .into_iter()
            .map(|(name, projected_dens)| {
                (
                    name,
                    projected_dens
                        .into_iter()
                        .map(|(row, den_res)| {
                            (
                                row.to_string(),
                                den_res
                                    .and_then(|den| {
                                        let pyden = PyDensityReal {
                                            complex_symmetric: den.complex_symmetric(),
                                            density_matrix: den.density_matrix().clone(),
                                            threshold: den.threshold(),
                                        };
                                        pyden.into_py_any(py).map_err(|err| err.to_string())
                                    })
                                    .expect("Unable to convert a projected density into a Python object."),
                            )
                        })
                        .collect::<IndexMap<_, _>>(),
                )
            })
            .collect::<Vec<_>>()
    } else {
        // At least one of coefficients or sao_4c are not real.
        let dens: Vec<Density<C128>> = pydens
            .iter()
            .map(|(_, pyden)| match pyden {
                PyDensity::Real(pyden_r) => pyden_r.to_qsym2(&bao, mol).map(|den_r| den_r.into()),
                PyDensity::Complex(pyden_c) => pyden_c.to_qsym2(&bao, mol),
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
        let dens_ref = dens
            .iter()
            .zip(pydens.iter())
            .map(|(den, (desc, _))| (desc.clone(), den))
            .collect::<Vec<_>>();

        let projected_densities = match &use_magnetic_group {
            Some(MagneticSymmetryAnalysisKind::Corepresentation) => {
                return Err(PyRuntimeError::new_err(
                    "Projection using corepresentations is not yet supported.",
                ));
            }
            Some(MagneticSymmetryAnalysisKind::Representation) | None => {
                let mut dp_driver =
                    DensityProjectionDriver::<UnitaryRepresentedSymmetryGroup, C128>::builder()
                        .parameters(&dp_params)
                        .densities(dens_ref)
                        .symmetry_group(&pd_res)
                        .build()
                        .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
                py.allow_threads(|| {
                    dp_driver
                        .run()
                        .map_err(|err| PyRuntimeError::new_err(err.to_string()))
                })?;
                dp_driver
                    .result()
                    .map_err(|err| PyRuntimeError::new_err(err.to_string()))?
                    .projected_densities()
                    .clone()
            }
        };
        projected_densities
            .into_iter()
            .map(|(name, projected_dens)| {
                (
                    name,
                    projected_dens
                        .into_iter()
                        .map(|(row, den_res)| {
                            (
                                row.to_string(),
                                den_res
                                    .and_then(|den| {
                                        let pyden = PyDensityComplex {
                                            complex_symmetric: den.complex_symmetric(),
                                            density_matrix: den.density_matrix().clone(),
                                            threshold: den.threshold(),
                                        };
                                        pyden.into_py_any(py).map_err(|err| err.to_string())
                                    })
                                    .expect("Unable to convert a projected density into a Python object."),
                            )
                        })
                        .collect::<IndexMap<_, _>>(),
                )
            })
            .collect::<Vec<_>>()
    };
    Ok(projected_densities)
}
