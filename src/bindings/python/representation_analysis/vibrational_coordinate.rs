//! Python bindings for QSym² symmetry analysis of vibrational coordinates.

use std::path::PathBuf;

use anyhow::format_err;
use ndarray::{Array1, Array2};
use num_complex::Complex;
use numpy::{PyArray1, PyArray2};
use pyo3::exceptions::{PyIOError, PyRuntimeError};
use pyo3::prelude::*;

use crate::analysis::EigenvalueComparisonMode;
use crate::auxiliary::molecule::Molecule;
use crate::drivers::representation_analysis::angular_function::AngularFunctionRepAnalysisParams;
use crate::drivers::representation_analysis::vibrational_coordinate::{
    VibrationalCoordinateRepAnalysisDriver, VibrationalCoordinateRepAnalysisParams,
};
use crate::drivers::representation_analysis::{
    CharacterTableDisplay, MagneticSymmetryAnalysisKind,
};
use crate::drivers::symmetry_group_detection::SymmetryGroupDetectionResult;
use crate::drivers::QSym2Driver;
use crate::io::format::qsym2_output;
use crate::io::{read_qsym2_binary, QSym2FileType};
use crate::symmetry::symmetry_group::{
    MagneticRepresentedSymmetryGroup, UnitaryRepresentedSymmetryGroup,
};
use crate::symmetry::symmetry_transformation::SymmetryTransformationKind;
use crate::target::vibration::VibrationalCoordinateCollection;

type C128 = Complex<f64>;

// ==================
// Struct definitions
// ==================

/// Python-exposed structure to marshall real vibrational coordinate collections between Rust and
/// Python.
///
/// # Constructor arguments
///
/// * `coefficients` - The real coefficients for the vibrational coordinates of this collection.
/// Python type: `list[numpy.2darray[float]]`.
/// * `frequencies` - The real vibrational frequencies. Python type: `numpy.1darray[float]`.
/// * `threshold` - The threshold for comparisons. Python type: `float`.
#[pyclass]
#[derive(Clone)]
pub struct PyVibrationalCoordinateCollectionReal {
    /// The real coefficients for the vibrational coordinates of this collection.
    ///
    /// Python type: `list[numpy.2darray[float]]`.
    coefficients: Array2<f64>,

    /// The real vibrational frequencies.
    ///
    /// Python type: `numpy.1darray[float]`.
    frequencies: Array1<f64>,

    /// The threshold for comparisons.
    ///
    /// Python type: `float`.
    threshold: f64,
}

#[pymethods]
impl PyVibrationalCoordinateCollectionReal {
    /// Constructs a real vibrational coordinate collection.
    ///
    /// # Arguments
    ///
    /// * `coefficients` - The real coefficients for the vibrational coordinates of this collection.
    /// Python type: `list[numpy.2darray[float]]`.
    /// * `frequencies` - The real vibrational frequencies. Python type: `numpy.1darray[float]`.
    /// * `threshold` - The threshold for comparisons. Python type: `float`.
    #[new]
    fn new(coefficients: &PyArray2<f64>, frequencies: &PyArray1<f64>, threshold: f64) -> Self {
        let vibs = Self {
            coefficients: coefficients.to_owned_array(),
            frequencies: frequencies.to_owned_array(),
            threshold,
        };
        vibs
    }
}

impl PyVibrationalCoordinateCollectionReal {
    /// Extracts the information in the [`PyVibrationalCoordinateCollectionReal`] structure into
    /// `QSym2`'s native [`VibrationalCoordinateCollection`] structure.
    ///
    /// # Arguments
    ///
    /// * `mol` - The molecule with which the vibrational coordinates are associated.
    ///
    /// # Returns
    ///
    /// The [`VibrationalCoordinateCollection`] structure with the same information.
    ///
    /// # Errors
    ///
    /// Errors if the [`VibrationalCoordinateCollection`] fails to build.
    fn to_qsym2<'b, 'a: 'b>(
        &'b self,
        mol: &'a Molecule,
    ) -> Result<VibrationalCoordinateCollection<f64>, anyhow::Error> {
        let vibs = VibrationalCoordinateCollection::<f64>::builder()
            .mol(mol)
            .coefficients(self.coefficients.clone())
            .frequencies(self.frequencies.clone())
            .threshold(self.threshold)
            .build()
            .map_err(|err| format_err!(err));
        vibs
    }
}

/// Python-exposed structure to marshall complex vibrational coordinate collections between Rust
/// and Python.
///
/// # Constructor arguments
///
/// * `coefficients` - The complex coefficients for the vibrational coordinates of this collection.
/// Python type: `list[numpy.2darray[complex]]`.
/// * `frequencies` - The complex vibrational frequencies. Python type: `numpy.1darray[complex]`.
/// * `threshold` - The threshold for comparisons. Python type: `float`.
#[pyclass]
#[derive(Clone)]
pub struct PyVibrationalCoordinateCollectionComplex {
    /// The complex coefficients for the vibrational coordinates of this collection.
    ///
    /// Python type: `list[numpy.2darray[complex]]`.
    coefficients: Array2<C128>,

    /// The complex vibrational frequencies.
    ///
    /// Python type: `numpy.1darray[complex]`.
    frequencies: Array1<C128>,

    /// The threshold for comparisons.
    ///
    /// Python type: `float`.
    threshold: f64,
}

#[pymethods]
impl PyVibrationalCoordinateCollectionComplex {
    /// Constructs a complex vibrational coordinate collection.
    ///
    /// # Arguments
    ///
    /// * `coefficients` - The complex coefficients for the vibrational coordinates of this
    /// collection.
    /// Python type: `list[numpy.2darray[complex]]`.
    /// * `frequencies` - The complex vibrational frequencies. Python type: `numpy.1darray[complex]`.
    /// * `threshold` - The threshold for comparisons. Python type: `float`.
    #[new]
    fn new(coefficients: &PyArray2<C128>, frequencies: &PyArray1<C128>, threshold: f64) -> Self {
        let vibs = Self {
            coefficients: coefficients.to_owned_array(),
            frequencies: frequencies.to_owned_array(),
            threshold,
        };
        vibs
    }
}

impl PyVibrationalCoordinateCollectionComplex {
    /// Extracts the information in the [`PyVibrationalCoordinateCollectionComplex`] structure into
    /// `QSym2`'s native [`VibrationalCoordinateCollection`] structure.
    ///
    /// # Arguments
    ///
    /// * `mol` - The molecule with which the vibrational coordinates are associated.
    ///
    /// # Returns
    ///
    /// The [`VibrationalCoordinateCollection`] structure with the same information.
    ///
    /// # Errors
    ///
    /// Errors if the [`VibrationalCoordinateCollection`] fails to build.
    fn to_qsym2<'b, 'a: 'b>(
        &'b self,
        mol: &'a Molecule,
    ) -> Result<VibrationalCoordinateCollection<C128>, anyhow::Error> {
        let vibs = VibrationalCoordinateCollection::<C128>::builder()
            .mol(mol)
            .coefficients(self.coefficients.clone())
            .frequencies(self.frequencies.clone())
            .threshold(self.threshold)
            .build()
            .map_err(|err| format_err!(err));
        vibs
    }
}

// ================
// Enum definitions
// ================

/// Python-exposed enumerated type to handle the union type
/// `PyVibrationalCoordinateCollectionReal | PyVibrationalCoordinateCollectionComplex` in Python.
#[derive(FromPyObject)]
pub enum PyVibrationalCoordinateCollection {
    /// Variant for complex Python-exposed vibrational coordinate collection.
    Real(PyVibrationalCoordinateCollectionReal),

    /// Variant for complex Python-exposed vibrational coordinate collection.
    Complex(PyVibrationalCoordinateCollectionComplex),
}

// =====================
// Functions definitions
// =====================

/// Python-exposed function to perform representation symmetry analysis for real and complex
/// vibrational coordinate collections and log the result via the `qsym2-output` logger at the
/// `INFO` level.
///
/// # Arguments
///
/// * `inp_sym` - A path to the [`QSym2FileType::Sym`] file containing the symmetry-group detection
/// result for the system. This will be used to construct abstract groups and character tables for
/// representation analysis. Python type: `str`.
/// * `pyvibs` - A Python-exposed vibrational coordinate collection whose coefficients are of type `
/// float64` or `complex128`.
/// Python type: `PyVibrationalCoordinateCollectionReal | PyVibrationalCoordinateCollectionComplex`
/// * `integrality_threshold` - The threshold for verifying if subspace multiplicities are
/// integral. Python type: `float`.
/// * `linear_independence_threshold` - The threshold for determining the linear independence
/// subspace via the non-zero eigenvalues of the orbit overlap matrix. Python type: `float`.
/// * `use_magnetic_group` - A boolean indicating if any magnetic group present should be used for
/// representation analysis. Otherwise, the unitary group will be used. Python type: `bool`.
/// * `use_double_group` - A boolean indicating if the double group of the prevailing symmetry
/// group is to be used for representation analysis instead. Python type: `bool`.
/// * `use_corepresentation` - A boolean indicating if corepresentations of magnetic groups are to
/// be used for representation analysis instead of unitary representations. Python type: `bool`.
/// * `symmetry_transformation_kind` - An enumerated type indicating the type of symmetry
/// transformations to be performed on the origin determinant to generate the orbit. If this
/// contains spin transformation, the determinant will be augmented to generalised spin constraint
/// automatically. Python type: `SymmetryTransformationKind`.
/// * `eigenvalue_comparison_mode` - An enumerated type indicating the mode of comparison of orbit
/// overlap eigenvalues with the specified `linear_independence_threshold`.
/// Python type: `EigenvalueComparisonMode`.
/// * `write_character_table` - A boolean indicating if the character table of the prevailing
/// symmetry group is to be printed out. Python type: `bool`.
/// * `infinite_order_to_finite` - The finite order with which infinite-order generators are to be
/// interpreted to form a finite subgroup of the prevailing infinite group. This finite subgroup
/// will be used for the symmetrisation. Python type: `Optional[int]`.
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
    pyvibs,
    integrality_threshold,
    linear_independence_threshold,
    use_magnetic_group,
    use_double_group,
    symmetry_transformation_kind,
    eigenvalue_comparison_mode,
    write_character_table=true,
    infinite_order_to_finite=None,
    angular_function_integrality_threshold=1e-7,
    angular_function_linear_independence_threshold=1e-7,
    angular_function_max_angular_momentum=2
))]
pub fn rep_analyse_vibrational_coordinate_collection(
    py: Python<'_>,
    inp_sym: PathBuf,
    pyvibs: PyVibrationalCoordinateCollection,
    integrality_threshold: f64,
    linear_independence_threshold: f64,
    use_magnetic_group: Option<MagneticSymmetryAnalysisKind>,
    use_double_group: bool,
    symmetry_transformation_kind: SymmetryTransformationKind,
    eigenvalue_comparison_mode: EigenvalueComparisonMode,
    write_character_table: bool,
    infinite_order_to_finite: Option<u32>,
    angular_function_integrality_threshold: f64,
    angular_function_linear_independence_threshold: f64,
    angular_function_max_angular_momentum: u32,
) -> PyResult<()> {
    py.allow_threads(|| {
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

        let afa_params = AngularFunctionRepAnalysisParams::builder()
            .integrality_threshold(angular_function_integrality_threshold)
            .linear_independence_threshold(angular_function_linear_independence_threshold)
            .max_angular_momentum(angular_function_max_angular_momentum)
            .build()
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
        match &pyvibs {
            PyVibrationalCoordinateCollection::Real(pyvibs_r) => {
                let vibs_r = pyvibs_r
                    .to_qsym2(mol)
                    .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
                let vca_params = VibrationalCoordinateRepAnalysisParams::<f64>::builder()
                    .integrality_threshold(integrality_threshold)
                    .linear_independence_threshold(linear_independence_threshold)
                    .use_magnetic_group(use_magnetic_group.clone())
                    .use_double_group(use_double_group)
                    .symmetry_transformation_kind(symmetry_transformation_kind)
                    .eigenvalue_comparison_mode(eigenvalue_comparison_mode)
                    .write_character_table(if write_character_table {
                        Some(CharacterTableDisplay::Symbolic)
                    } else {
                        None
                    })
                    .infinite_order_to_finite(infinite_order_to_finite)
                    .build()
                    .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
                match &use_magnetic_group {
                    Some(MagneticSymmetryAnalysisKind::Corepresentation) => {
                        let mut vca_driver = VibrationalCoordinateRepAnalysisDriver::<
                            MagneticRepresentedSymmetryGroup,
                            f64,
                        >::builder()
                        .parameters(&vca_params)
                        .angular_function_parameters(&afa_params)
                        .vibrational_coordinate_collection(&vibs_r)
                        .symmetry_group(&pd_res)
                        .build()
                        .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
                        vca_driver
                            .run()
                            .map_err(|err| PyRuntimeError::new_err(err.to_string()))?
                    }
                    Some(MagneticSymmetryAnalysisKind::Representation) | None => {
                        let mut vca_driver = VibrationalCoordinateRepAnalysisDriver::<
                            UnitaryRepresentedSymmetryGroup,
                            f64,
                        >::builder()
                        .parameters(&vca_params)
                        .angular_function_parameters(&afa_params)
                        .vibrational_coordinate_collection(&vibs_r)
                        .symmetry_group(&pd_res)
                        .build()
                        .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
                        vca_driver
                            .run()
                            .map_err(|err| PyRuntimeError::new_err(err.to_string()))?
                    }
                };
            }
            PyVibrationalCoordinateCollection::Complex(pyvibs_c) => {
                let vibs_c = pyvibs_c
                    .to_qsym2(mol)
                    .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
                let vca_params = VibrationalCoordinateRepAnalysisParams::<f64>::builder()
                    .integrality_threshold(integrality_threshold)
                    .linear_independence_threshold(linear_independence_threshold)
                    .use_magnetic_group(use_magnetic_group.clone())
                    .use_double_group(use_double_group)
                    .symmetry_transformation_kind(symmetry_transformation_kind)
                    .eigenvalue_comparison_mode(eigenvalue_comparison_mode)
                    .write_character_table(if write_character_table {
                        Some(CharacterTableDisplay::Symbolic)
                    } else {
                        None
                    })
                    .infinite_order_to_finite(infinite_order_to_finite)
                    .build()
                    .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
                match &use_magnetic_group {
                    Some(MagneticSymmetryAnalysisKind::Corepresentation) => {
                        let mut vca_driver = VibrationalCoordinateRepAnalysisDriver::<
                            MagneticRepresentedSymmetryGroup,
                            C128,
                        >::builder()
                        .parameters(&vca_params)
                        .angular_function_parameters(&afa_params)
                        .vibrational_coordinate_collection(&vibs_c)
                        .symmetry_group(&pd_res)
                        .build()
                        .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
                        vca_driver
                            .run()
                            .map_err(|err| PyRuntimeError::new_err(err.to_string()))?
                    }
                    Some(MagneticSymmetryAnalysisKind::Representation) | None => {
                        let mut vca_driver = VibrationalCoordinateRepAnalysisDriver::<
                            UnitaryRepresentedSymmetryGroup,
                            C128,
                        >::builder()
                        .parameters(&vca_params)
                        .angular_function_parameters(&afa_params)
                        .vibrational_coordinate_collection(&vibs_c)
                        .symmetry_group(&pd_res)
                        .build()
                        .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
                        vca_driver
                            .run()
                            .map_err(|err| PyRuntimeError::new_err(err.to_string()))?
                    }
                };
            }
        }
        Ok(())
    })
}
