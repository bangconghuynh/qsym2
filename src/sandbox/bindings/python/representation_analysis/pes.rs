//! Sandbox Python bindings for QSym² symmetry analysis of PESes.

use std::path::PathBuf;

use anyhow::format_err;
use nalgebra::Point3;
use ndarray::{Array1, Array2};
use num_complex::Complex;
use numpy::{PyArray1, PyArray2};
use pyo3::exceptions::{PyIOError, PyRuntimeError};
use pyo3::prelude::*;
use pyo3::types::PyFunction;

use crate::analysis::EigenvalueComparisonMode;
use crate::auxiliary::molecule::Molecule;
use crate::basis::ao::BasisAngularOrder;
use crate::bindings::python::integrals::{PyBasisAngularOrder, PySpinConstraint};
use crate::bindings::python::representation_analysis::{PyArray2RC, PyArray4RC};
use crate::drivers::representation_analysis::angular_function::AngularFunctionRepAnalysisParams;
use crate::drivers::representation_analysis::{
    CharacterTableDisplay, MagneticSymmetryAnalysisKind,
};
use crate::drivers::symmetry_group_detection::SymmetryGroupDetectionResult;
use crate::drivers::QSym2Driver;
use crate::io::format::qsym2_output;
use crate::io::{read_qsym2_binary, QSym2FileType};
use crate::sandbox::drivers::representation_analysis::pes::{
    PESRepAnalysisDriver, PESRepAnalysisParams,
};
use crate::sandbox::target::pes::PES;
use crate::symmetry::symmetry_group::{
    MagneticRepresentedSymmetryGroup, UnitaryRepresentedSymmetryGroup,
};
use crate::symmetry::symmetry_transformation::SymmetryTransformationKind;

// =====================
// Functions definitions
// =====================

/// Python-exposed function to perform representation symmetry analysis for potential energy
/// surfaces (PESes) and log the result via the `qsym2-output` logger at the `INFO` level.
///
/// # Arguments
///
/// * `inp_sym` - A path to the [`QSym2FileType::Sym`] file containing the symmetry-group detection
/// result for the system. This will be used to construct abstract groups and character tables for
/// representation analysis. Python type: `str`.
/// * `pes_function` - A Python function callable on three Cartesian coordinates to give a potential
/// energy value. Python type: `Callable[[float, float, float], float]`.
/// * `integrality_threshold` - The threshold for verifying if subspace multiplicities are integral.
/// Python type: `float`.
/// * `linear_independence_threshold` - The threshold for determining the linear independence
/// subspace via the non-zero eigenvalues of the orbit overlap matrix. Python type: `float`.
/// * `use_magnetic_group` - An option indicating if the magnetic group is to be used for symmetry
/// analysis, and if so, whether unitary representations or unitary-antiunitary corepresentations
/// should be used. Python type: `None | MagneticSymmetryAnalysisKind`.
/// * `use_double_group` - A boolean indicating if the double group of the prevailing symmetry
/// group is to be used for representation analysis instead. Python type: `bool`.
/// * `use_cayley_table` - A boolean indicating if the Cayley table for the group, if available,
/// should be used to speed up the calculation of orbit overlap matrices. Python type: `bool`.
/// * `symmetry_transformation_kind` - An enumerated type indicating the type of symmetry
/// transformations to be performed on the origin PES to generate the orbit. If this
/// contains spin transformation, the determinant will be augmented to generalised spin constraint
/// automatically. Python type: `SymmetryTransformationKind`.
/// * `eigenvalue_comparison_mode` - An enumerated type indicating the mode of comparison of orbit
/// overlap eigenvalues with the specified `linear_independence_threshold`.
/// Python type: `EigenvalueComparisonMode`.
/// * `grid_points` - The grid points at which the PES is evaluated specified as a $`3 \times N`$
/// array where $`N`$ is the number of points. Python type: `numpy.2darray[float]`.
/// * `weight` - The weight to be used in the computation of overlaps between PESes specified as a
/// one-dimensional array. The number of weight values must match the number of grid points. Python
/// type: `numpy.1darray[float]`.
/// * `write_overlap_eigenvalues` - A boolean indicating if the eigenvalues of the determinant
/// orbit overlap matrix are to be written to the output. Python type: `bool`.
/// * `write_character_table` - A boolean indicating if the character table of the prevailing
/// symmetry group is to be printed out. Python type: `bool`.
/// * `infinite_order_to_finite` - The finite order with which infinite-order generators are to be
/// interpreted to form a finite subgroup of the prevailing infinite group. This finite subgroup
/// will be used for symmetry analysis. Python type: `Optional[int]`.
/// * `angular_function_integrality_threshold` - The threshold for verifying if subspace
/// multiplicities are integral for the symmetry analysis of angular functions. Python type:
/// `float`.
/// * `angular_function_linear_independence_threshold` - The threshold for determining the linear
/// independence subspace via the non-zero eigenvalues of the orbit overlap matrix for the symmetry
/// analysis of angular functions. Python type: `float`.
/// * `angular_function_max_angular_momentum` - The maximum angular momentum order to be used in
/// angular function symmetry analysis. Python type: `int`.
#[pyfunction]
#[pyo3(signature = (
    inp_sym,
    pes_function,
    integrality_threshold,
    linear_independence_threshold,
    use_magnetic_group,
    use_double_group,
    use_cayley_table,
    symmetry_transformation_kind,
    eigenvalue_comparison_mode,
    grid_points,
    weight,
    write_overlap_eigenvalues=true,
    write_character_table=true,
    infinite_order_to_finite=None,
    angular_function_integrality_threshold=1e-7,
    angular_function_linear_independence_threshold=1e-7,
    angular_function_max_angular_momentum=2
))]
pub fn rep_analyse_pes(
    py: Python<'_>,
    inp_sym: PathBuf,
    pes_function: Py<PyFunction>,
    integrality_threshold: f64,
    linear_independence_threshold: f64,
    use_magnetic_group: Option<MagneticSymmetryAnalysisKind>,
    use_double_group: bool,
    use_cayley_table: bool,
    symmetry_transformation_kind: SymmetryTransformationKind,
    eigenvalue_comparison_mode: EigenvalueComparisonMode,
    grid_points: &PyArray2<f64>,
    weight: &PyArray1<f64>,
    write_overlap_eigenvalues: bool,
    write_character_table: bool,
    infinite_order_to_finite: Option<u32>,
    angular_function_integrality_threshold: f64,
    angular_function_linear_independence_threshold: f64,
    angular_function_max_angular_momentum: u32,
) -> PyResult<()> {
    // if !pes_function.is_callable() {
    //     return Err(PyRuntimeError::new_err("`pes_function` is not callable."));
    // }

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

    let afa_params = AngularFunctionRepAnalysisParams::builder()
        .integrality_threshold(angular_function_integrality_threshold)
        .linear_independence_threshold(angular_function_linear_independence_threshold)
        .max_angular_momentum(angular_function_max_angular_momentum)
        .build()
        .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
    let pes_params = PESRepAnalysisParams::<f64>::builder()
        .integrality_threshold(integrality_threshold)
        .linear_independence_threshold(linear_independence_threshold)
        .use_magnetic_group(use_magnetic_group.clone())
        .use_double_group(use_double_group)
        .use_cayley_table(use_cayley_table)
        .symmetry_transformation_kind(symmetry_transformation_kind)
        .eigenvalue_comparison_mode(eigenvalue_comparison_mode)
        .write_overlap_eigenvalues(write_overlap_eigenvalues)
        .write_character_table(if write_character_table {
            Some(CharacterTableDisplay::Symbolic)
        } else {
            None
        })
        .infinite_order_to_finite(infinite_order_to_finite)
        .build()
        .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;

    let weight = weight.to_owned_array();
    let grid_array = grid_points.to_owned_array();
    if grid_array.shape()[0] != 3 {
        return Err(PyRuntimeError::new_err(
            "The grid point array does not have the expected dimensions of 3 × N.",
        ));
    }
    let grid_points = grid_array
        .columns()
        .into_iter()
        .map(|col| Point3::new(col[0], col[1], col[2]))
        .collect::<Vec<_>>();

    let pes = PES::<f64, _>::builder()
        .function(|pt| {
            Python::with_gil(|py_inner| {
                let res = pes_function
                    .call1(py_inner, (pt.x, pt.y, pt.z))
                    .expect("Unable to apply the PES function.");
                res.extract::<f64>(py_inner)
                    .expect("Unable to extract the `f64` result from the PES function call.")
            })
        })
        .grid_points(grid_points)
        .build()
        .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;

    match &use_magnetic_group {
        Some(MagneticSymmetryAnalysisKind::Corepresentation) => {
            let mut pes_driver =
                PESRepAnalysisDriver::<MagneticRepresentedSymmetryGroup, f64, _>::builder()
                    .parameters(&pes_params)
                    .angular_function_parameters(&afa_params)
                    .pes(&pes)
                    .weight(&weight)
                    .symmetry_group(&pd_res)
                    .build()
                    .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
            py.allow_threads(|| {
                pes_driver
                    .run()
                    .map_err(|err| PyRuntimeError::new_err(err.to_string()))
            })?
        }
        Some(MagneticSymmetryAnalysisKind::Representation) | None => {
            let mut pes_driver =
                PESRepAnalysisDriver::<UnitaryRepresentedSymmetryGroup, f64, _>::builder()
                    .parameters(&pes_params)
                    .angular_function_parameters(&afa_params)
                    .pes(&pes)
                    .weight(&weight)
                    .symmetry_group(&pd_res)
                    .build()
                    .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
            py.allow_threads(|| {
                pes_driver
                    .run()
                    .map_err(|err| PyRuntimeError::new_err(err.to_string()))
            })?
        }
    };

    Ok(())
}