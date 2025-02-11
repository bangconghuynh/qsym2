//! Python bindings for QSym² symmetry analysis of Slater determinants.

use std::path::PathBuf;

use anyhow::format_err;
use ndarray::{Array1, Array2};
use num_complex::Complex;
use numpy::{PyArray1, PyArray2, PyArrayMethods, ToPyArray};
use pyo3::exceptions::{PyIOError, PyRuntimeError};
use pyo3::prelude::*;

use crate::analysis::EigenvalueComparisonMode;
use crate::angmom::spinor_rotation_3d::SpinConstraint;
use crate::auxiliary::molecule::Molecule;
use crate::basis::ao::BasisAngularOrder;
use crate::bindings::python::integrals::{PyBasisAngularOrder, PySpinConstraint};
use crate::bindings::python::representation_analysis::{PyArray2RC, PyArray4RC};
use crate::drivers::representation_analysis::angular_function::AngularFunctionRepAnalysisParams;
use crate::drivers::representation_analysis::slater_determinant::{
    SlaterDeterminantRepAnalysisDriver, SlaterDeterminantRepAnalysisParams,
};
use crate::drivers::representation_analysis::{
    CharacterTableDisplay, MagneticSymmetryAnalysisKind,
};
use crate::drivers::symmetry_group_detection::SymmetryGroupDetectionResult;
use crate::drivers::QSym2Driver;
use crate::group::GroupProperties;
use crate::io::format::qsym2_output;
use crate::io::{read_qsym2_binary, QSym2FileType};
use crate::symmetry::symmetry_group::{
    MagneticRepresentedSymmetryGroup, UnitaryRepresentedSymmetryGroup,
};
use crate::symmetry::symmetry_transformation::SymmetryTransformationKind;
use crate::target::determinant::SlaterDeterminant;

type C128 = Complex<f64>;

// ==================
// Struct definitions
// ==================

// ------------------
// Slater determinant
// ------------------

// ~~~~~~~~~~~~
// Real, no SOC
// ~~~~~~~~~~~~

/// Python-exposed structure to marshall real Slater determinant information between Rust and
/// Python.
///
/// # Constructor arguments
///
/// * `spin_constraint` - The spin constraint applied to the coefficients of the determinant.
/// Python type: `PySpinConstraint`.
/// * `complex_symmetric` - A boolean indicating if inner products involving this determinant
/// are complex-symmetric. Python type: `bool`.
/// * `coefficients` - The real coefficients for the molecular orbitals of this determinant.
/// Python type: `list[numpy.2darray[float]]`.
/// * `occupations` - The occupation patterns for the molecular orbitals. Python type:
/// `list[numpy.1darray[float]]`.
/// * `threshold` - The threshold for comparisons. Python type: `float`.
/// * `mo_energies` - The optional real molecular orbital energies. Python type:
/// `Optional[list[numpy.1darray[float]]]`.
/// * `energy` - The optional real determinantal energy. Python type: `Optional[float]`.
#[pyclass]
#[derive(Clone)]
pub struct PySlaterDeterminantReal {
    /// The spin constraint applied to the coefficients of the determinant.
    ///
    /// Python type: `PySpinConstraint`.
    spin_constraint: PySpinConstraint,

    /// A boolean indicating if inner products involving this determinant are complex-symmetric.
    ///
    /// Python type: `bool`.
    #[pyo3(get)]
    complex_symmetric: bool,

    /// The real coefficients for the molecular orbitals of this determinant.
    ///
    /// Python type: `list[numpy.2darray[float]]`.
    coefficients: Vec<Array2<f64>>,

    /// The occupation patterns for the molecular orbitals.
    ///
    /// Python type: `list[numpy.1darray[float]]`.
    occupations: Vec<Array1<f64>>,

    /// The threshold for comparisons.
    ///
    /// Python type: `float`.
    #[pyo3(get)]
    threshold: f64,

    /// The optional real molecular orbital energies.
    ///
    /// Python type: `Optional[list[numpy.1darray[float]]]`.
    mo_energies: Option<Vec<Array1<f64>>>,

    /// The optional real determinantal energy.
    ///
    /// Python type: `Optional[float]`.
    #[pyo3(get)]
    energy: Option<f64>,
}

#[pymethods]
impl PySlaterDeterminantReal {
    /// Constructs a real Python-exposed Slater determinant.
    ///
    /// # Arguments
    ///
    /// * `spin_constraint` - The spin constraint applied to the coefficients of the determinant.
    /// Python type: `PySpinConstraint`.
    /// * `complex_symmetric` - A boolean indicating if inner products involving this determinant
    /// are complex-symmetric. Python type: `bool`.
    /// * `coefficients` - The real coefficients for the molecular orbitals of this determinant.
    /// Python type: `list[numpy.2darray[float]]`.
    /// * `occupations` - The occupation patterns for the molecular orbitals. Python type:
    /// `list[numpy.1darray[float]]`.
    /// * `threshold` - The threshold for comparisons. Python type: `float`.
    /// * `mo_energies` - The optional real molecular orbital energies. Python type:
    /// `Optional[list[numpy.1darray[float]]]`.
    /// * `energy` - The optional real determinantal energy. Python type: `Optional[float]`.
    #[new]
    #[pyo3(signature = (spin_constraint, complex_symmetric, coefficients, occupations, threshold, mo_energies=None, energy=None))]
    pub(crate) fn new(
        spin_constraint: PySpinConstraint,
        complex_symmetric: bool,
        coefficients: Vec<Bound<'_, PyArray2<f64>>>,
        occupations: Vec<Bound<'_, PyArray1<f64>>>,
        threshold: f64,
        mo_energies: Option<Vec<Bound<'_, PyArray1<f64>>>>,
        energy: Option<f64>,
    ) -> Self {
        let det = Self {
            spin_constraint,
            complex_symmetric,
            coefficients: coefficients
                .iter()
                .map(|pyarr| pyarr.to_owned_array())
                .collect::<Vec<_>>(),
            occupations: occupations
                .iter()
                .map(|pyarr| pyarr.to_owned_array())
                .collect::<Vec<_>>(),
            threshold,
            mo_energies: mo_energies.map(|energies| {
                energies
                    .iter()
                    .map(|pyarr| pyarr.to_owned_array())
                    .collect::<Vec<_>>()
            }),
            energy,
        };
        det
    }

    #[getter]
    fn occupations<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyArray1<f64>>>> {
        Ok(self
            .occupations
            .iter()
            .map(|occ| occ.to_pyarray(py))
            .collect::<Vec<_>>())
    }

    #[getter]
    fn coefficients<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyArray2<f64>>>> {
        Ok(self
            .coefficients
            .iter()
            .map(|occ| occ.to_pyarray(py))
            .collect::<Vec<_>>())
    }
}

impl PySlaterDeterminantReal {
    /// Extracts the information in the [`PySlaterDeterminantReal`] structure into `QSym2`'s native
    /// [`SlaterDeterminant`] structure.
    ///
    /// # Arguments
    ///
    /// * `bao` - The [`BasisAngularOrder`] for the basis set in which the Slater determinant is
    /// given.
    /// * `mol` - The molecule with which the Slater determinant is associated.
    ///
    /// # Returns
    ///
    /// The [`SlaterDeterminant`] structure with the same information.
    ///
    /// # Errors
    ///
    /// Errors if the [`SlaterDeterminant`] fails to build.
    pub(crate) fn to_qsym2<'b, 'a: 'b>(
        &'b self,
        bao: &'a BasisAngularOrder,
        mol: &'a Molecule,
    ) -> Result<SlaterDeterminant<'b, f64, SpinConstraint>, anyhow::Error> {
        let det = SlaterDeterminant::<f64, SpinConstraint>::builder()
            .structure_constraint(self.spin_constraint.clone().into())
            .bao(bao)
            .complex_symmetric(self.complex_symmetric)
            .mol(mol)
            .coefficients(&self.coefficients)
            .occupations(&self.occupations)
            .mo_energies(self.mo_energies.clone())
            .energy(
                self.energy
                    .ok_or_else(|| "No determinantal energy set.".to_string()),
            )
            .threshold(self.threshold)
            .build()
            .map_err(|err| format_err!(err));
        det
    }
}

/// Python-exposed structure to marshall complex Slater determinant information between Rust and
/// Python.
///
/// # Constructor arguments
///
/// * `spin_constraint` - The spin constraint applied to the coefficients of the determinant.
/// Python type: `PySpinConstraint`.
/// * `complex_symmetric` - A boolean indicating if inner products involving this determinant
/// are complex-symmetric. Python type: `bool`.
/// * `coefficients` - The complex coefficients for the molecular orbitals of this determinant.
/// Python type: `list[numpy.2darray[float]]`.
/// * `occupations` - The occupation patterns for the molecular orbitals. Python type:
/// `list[numpy.1darray[float]]`.
/// * `threshold` - The threshold for comparisons. Python type: `float`.
/// * `mo_energies` - The optional complex molecular orbital energies. Python type:
/// `Optional[list[numpy.1darray[complex]]]`.
/// * `energy` - The optional complex determinantal energy. Python type: `Optional[complex]`.
#[pyclass]
#[derive(Clone)]
pub struct PySlaterDeterminantComplex {
    /// The spin constraint applied to the coefficients of the determinant.
    ///
    /// Python type: `PySpinConstraint`.
    #[pyo3(get)]
    spin_constraint: PySpinConstraint,

    /// A boolean indicating if inner products involving this determinant are complex-symmetric.
    ///
    /// Python type: `bool`.
    #[pyo3(get)]
    complex_symmetric: bool,

    /// The complex coefficients for the molecular orbitals of this determinant.
    ///
    /// Python type: `list[numpy.2darray[complex]]`.
    coefficients: Vec<Array2<C128>>,

    /// The occupation patterns for the molecular orbitals.
    ///
    /// Python type: `list[numpy.1darray[float]]`.
    occupations: Vec<Array1<f64>>,

    /// The threshold for comparisons.
    ///
    /// Python type: `float`.
    #[pyo3(get)]
    threshold: f64,

    /// The optional complex molecular orbital energies.
    ///
    /// Python type: `Optional[list[numpy.1darray[complex]]]`.
    mo_energies: Option<Vec<Array1<C128>>>,

    /// The optional complex determinantal energy.
    ///
    /// Python type: `Optional[complex]`.
    #[pyo3(get)]
    energy: Option<C128>,
}

#[pymethods]
impl PySlaterDeterminantComplex {
    /// Constructs a complex Python-exposed Slater determinant.
    ///
    /// # Arguments
    ///
    /// * `spin_constraint` - The spin constraint applied to the coefficients of the determinant.
    /// Python type: `PySpinConstraint`.
    /// * `complex_symmetric` - A boolean indicating if inner products involving this determinant
    /// are complex-symmetric. Python type: `bool`.
    /// * `coefficients` - The complex coefficients for the molecular orbitals of this determinant.
    /// Python type: `list[numpy.2darray[float]]`.
    /// * `occupations` - The occupation patterns for the molecular orbitals. Python type:
    /// `list[numpy.1darray[float]]`.
    /// * `threshold` - The threshold for comparisons. Python type: `float`.
    /// * `mo_energies` - The optional complex molecular orbital energies. Python type:
    /// `Optional[list[numpy.1darray[complex]]]`.
    /// * `energy` - The optional complex determinantal energy. Python type: `Optional[complex]`.
    #[new]
    #[pyo3(signature = (spin_constraint, complex_symmetric, coefficients, occupations, threshold, mo_energies=None, energy=None))]
    pub(crate) fn new(
        spin_constraint: PySpinConstraint,
        complex_symmetric: bool,
        coefficients: Vec<Bound<'_, PyArray2<C128>>>,
        occupations: Vec<Bound<'_, PyArray1<f64>>>,
        threshold: f64,
        mo_energies: Option<Vec<Bound<'_, PyArray1<C128>>>>,
        energy: Option<C128>,
    ) -> Self {
        let det = Self {
            spin_constraint,
            complex_symmetric,
            coefficients: coefficients
                .iter()
                .map(|pyarr| pyarr.to_owned_array())
                .collect::<Vec<_>>(),
            occupations: occupations
                .iter()
                .map(|pyarr| pyarr.to_owned_array())
                .collect::<Vec<_>>(),
            threshold,
            mo_energies: mo_energies.map(|energies| {
                energies
                    .iter()
                    .map(|pyarr| pyarr.to_owned_array())
                    .collect::<Vec<_>>()
            }),
            energy,
        };
        det
    }

    #[getter]
    fn occupations<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyArray1<f64>>>> {
        Ok(self
            .occupations
            .iter()
            .map(|occ| occ.to_pyarray(py))
            .collect::<Vec<_>>())
    }

    #[getter]
    fn coefficients<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyArray2<C128>>>> {
        Ok(self
            .coefficients
            .iter()
            .map(|occ| occ.to_pyarray(py))
            .collect::<Vec<_>>())
    }
}

impl PySlaterDeterminantComplex {
    /// Extracts the information in the [`PySlaterDeterminantComplex`] structure into `QSym2`'s native
    /// [`SlaterDeterminant`] structure.
    ///
    /// # Arguments
    ///
    /// * `bao` - The [`BasisAngularOrder`] for the basis set in which the Slater determinant is
    /// given.
    /// * `mol` - The molecule with which the Slater determinant is associated.
    ///
    /// # Returns
    ///
    /// The [`SlaterDeterminant`] structure with the same information.
    ///
    /// # Errors
    ///
    /// Errors if the [`SlaterDeterminant`] fails to build.
    pub(crate) fn to_qsym2<'b, 'a: 'b>(
        &'b self,
        bao: &'a BasisAngularOrder,
        mol: &'a Molecule,
    ) -> Result<SlaterDeterminant<'b, C128, SpinConstraint>, anyhow::Error> {
        let det = SlaterDeterminant::<C128, SpinConstraint>::builder()
            .structure_constraint(self.spin_constraint.clone().into())
            .bao(bao)
            .complex_symmetric(self.complex_symmetric)
            .mol(mol)
            .coefficients(&self.coefficients)
            .occupations(&self.occupations)
            .mo_energies(self.mo_energies.clone())
            .energy(
                self.energy
                    .ok_or_else(|| "No determinantal energy set.".to_string()),
            )
            .threshold(self.threshold)
            .build()
            .map_err(|err| format_err!(err));
        det
    }
}

// --------------------------------------------
// Slater determinant symmetry analysis results
// --------------------------------------------

/// Python-exposed structure storing the results of Slater determinant representation analysis.
#[pyclass]
#[derive(Clone)]
pub struct PySlaterDeterminantRepAnalysisResult {
    /// The group used for the representation analysis.
    #[pyo3(get)]
    group: String,

    /// The deduced overall symmetry of the determinant.
    #[pyo3(get)]
    determinant_symmetry: Option<String>,

    /// The deduced symmetries of the molecular orbitals constituting the determinant, if required.
    #[pyo3(get)]
    mo_symmetries: Option<Vec<Vec<Option<String>>>>,

    /// The deduced symmetries of the various densities constructible from the determinant, if
    /// required. In each tuple, the first element gives a description of the density corresponding
    /// to the symmetry result.
    #[pyo3(get)]
    determinant_density_symmetries: Option<Vec<(String, Option<String>)>>,

    /// The deduced symmetries of the total densities of the molecular orbitals constituting the
    /// determinant, if required.
    #[pyo3(get)]
    mo_density_symmetries: Option<Vec<Vec<Option<String>>>>,
}

// ================
// Enum definitions
// ================

/// Python-exposed enumerated type to handle the union type
/// `PySlaterDeterminantReal | PySlaterDeterminantComplex` in Python.
#[derive(FromPyObject)]
pub enum PySlaterDeterminant {
    /// Variant for real Python-exposed Slater determinant.
    Real(PySlaterDeterminantReal),

    /// Variant for complex Python-exposed Slater determinant.
    Complex(PySlaterDeterminantComplex),
}

// =====================
// Functions definitions
// =====================

/// Python-exposed function to perform representation symmetry analysis for real and complex
/// Slater determinants and log the result via the `qsym2-output` logger at the `INFO` level.
///
/// If `symmetry_transformation_kind` includes spin transformation, the provided determinant will
/// be augmented to generalised spin constraint automatically.
///
/// # Arguments
///
/// * `inp_sym` - A path to the [`QSym2FileType::Sym`] file containing the symmetry-group detection
/// result for the system. This will be used to construct abstract groups and character tables for
/// representation analysis. Python type: `str`.
/// * `pydet` - A Python-exposed Slater determinant whose coefficients are of type `float64` or
/// `complex128`. Python type: `PySlaterDeterminantReal | PySlaterDeterminantComplex`.
/// * `pybao` - A Python-exposed Python-exposed structure containing basis angular order information.
/// Python type: `PyBasisAngularOrder`.
/// * `integrality_threshold` - The threshold for verifying if subspace multiplicities are
/// integral. Python type: `float`.
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
/// transformations to be performed on the origin determinant to generate the orbit. If this
/// contains spin transformation, the determinant will be augmented to generalised spin constraint
/// automatically. Python type: `SymmetryTransformationKind`.
/// * `eigenvalue_comparison_mode` - An enumerated type indicating the mode of comparison of orbit
/// overlap eigenvalues with the specified `linear_independence_threshold`.
/// Python type: `EigenvalueComparisonMode`.
/// * `sao_spatial` - The atomic-orbital overlap matrix whose elements are of type `float64` or
/// `complex128`. Python type: `numpy.2darray[float] | numpy.2darray[complex]`.
/// * `sao_spatial_h` - The optional complex-symmetric atomic-orbital overlap matrix whose elements
/// are of type `float64` or `complex128`. This is required if antiunitary symmetry operations are
/// involved. Python type: `None | numpy.2darray[float] | numpy.2darray[complex]`.
/// * `sao_spatial_4c` - The optional atomic-orbital four-centre overlap matrix whose elements are
/// of type `float64` or `complex128`.
/// Python type: `numpy.2darray[float] | numpy.2darray[complex] | None`.
/// * `sao_spatial_4c_h` - The optional complex-symmetric atomic-orbital four-centre overlap matrix
/// whose elements are of type `float64` or `complex128`. This is required if antiunitary symmetry
/// operations are involved. Python type: `numpy.2darray[float] | numpy.2darray[complex] | None`.
/// * `analyse_mo_symmetries` - A boolean indicating if the symmetries of individual molecular
/// orbitals are to be analysed. Python type: `bool`.
/// * `analyse_mo_mirror_parities` - A boolean indicating if the mirror parities of individual
/// molecular orbitals are to be printed. Python type: `bool`.
/// * `analyse_density_symmetries` - A boolean indicating if the symmetries of densities are to be
/// analysed. Python type: `bool`.
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
///
/// # Returns
///
/// A Python-exposed [`PySlaterDeterminantRepAnalysisResult`] structure containing the results of the
/// representation analysis. Python type: `PySlaterDeterminantRepAnalysisResult`.
#[pyfunction]
#[pyo3(signature = (
    inp_sym,
    pydet,
    pybao,
    integrality_threshold,
    linear_independence_threshold,
    use_magnetic_group,
    use_double_group,
    use_cayley_table,
    symmetry_transformation_kind,
    eigenvalue_comparison_mode,
    sao_spatial,
    sao_spatial_h=None,
    sao_spatial_4c=None,
    sao_spatial_4c_h=None,
    analyse_mo_symmetries=true,
    analyse_mo_mirror_parities=false,
    analyse_density_symmetries=false,
    write_overlap_eigenvalues=true,
    write_character_table=true,
    infinite_order_to_finite=None,
    angular_function_integrality_threshold=1e-7,
    angular_function_linear_independence_threshold=1e-7,
    angular_function_max_angular_momentum=2
))]
pub fn rep_analyse_slater_determinant(
    py: Python<'_>,
    inp_sym: PathBuf,
    pydet: PySlaterDeterminant,
    pybao: &PyBasisAngularOrder,
    integrality_threshold: f64,
    linear_independence_threshold: f64,
    use_magnetic_group: Option<MagneticSymmetryAnalysisKind>,
    use_double_group: bool,
    use_cayley_table: bool,
    symmetry_transformation_kind: SymmetryTransformationKind,
    eigenvalue_comparison_mode: EigenvalueComparisonMode,
    sao_spatial: PyArray2RC,
    sao_spatial_h: Option<PyArray2RC>,
    sao_spatial_4c: Option<PyArray4RC>,
    sao_spatial_4c_h: Option<PyArray4RC>,
    analyse_mo_symmetries: bool,
    analyse_mo_mirror_parities: bool,
    analyse_density_symmetries: bool,
    write_overlap_eigenvalues: bool,
    write_character_table: bool,
    infinite_order_to_finite: Option<u32>,
    angular_function_integrality_threshold: f64,
    angular_function_linear_independence_threshold: f64,
    angular_function_max_angular_momentum: u32,
) -> PyResult<PySlaterDeterminantRepAnalysisResult> {
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
    let augment_to_generalised = match symmetry_transformation_kind {
        SymmetryTransformationKind::SpatialWithSpinTimeReversal
        | SymmetryTransformationKind::Spin
        | SymmetryTransformationKind::SpinSpatial => true,
        SymmetryTransformationKind::Spatial => false,
    };
    let afa_params = AngularFunctionRepAnalysisParams::builder()
        .integrality_threshold(angular_function_integrality_threshold)
        .linear_independence_threshold(angular_function_linear_independence_threshold)
        .max_angular_momentum(angular_function_max_angular_momentum)
        .build()
        .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
    let sda_params = SlaterDeterminantRepAnalysisParams::<f64>::builder()
        .integrality_threshold(integrality_threshold)
        .linear_independence_threshold(linear_independence_threshold)
        .use_magnetic_group(use_magnetic_group.clone())
        .use_double_group(use_double_group)
        .use_cayley_table(use_cayley_table)
        .symmetry_transformation_kind(symmetry_transformation_kind)
        .eigenvalue_comparison_mode(eigenvalue_comparison_mode)
        .analyse_mo_symmetries(analyse_mo_symmetries)
        .analyse_mo_mirror_parities(analyse_mo_mirror_parities)
        .analyse_density_symmetries(analyse_density_symmetries)
        .write_overlap_eigenvalues(write_overlap_eigenvalues)
        .write_character_table(if write_character_table {
            Some(CharacterTableDisplay::Symbolic)
        } else {
            None
        })
        .infinite_order_to_finite(infinite_order_to_finite)
        .build()
        .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
    let pysda_res: PySlaterDeterminantRepAnalysisResult = match (&pydet, &sao_spatial) {
        (PySlaterDeterminant::Real(pydet_r), PyArray2RC::Real(pysao_r)) => {
            let sao_spatial = pysao_r.to_owned_array();
            let sao_spatial_4c = sao_spatial_4c.and_then(|pysao4c| match pysao4c {
                // sao_spatial_4c must have the same reality as sao_spatial.
                PyArray4RC::Real(pysao4c_r) => Some(pysao4c_r.to_owned_array()),
                PyArray4RC::Complex(_) => None,
            });
            let det_r = if augment_to_generalised {
                pydet_r
                    .to_qsym2(&bao, mol)
                    .map_err(|err| PyRuntimeError::new_err(err.to_string()))?
                    .to_generalised()
            } else {
                pydet_r
                    .to_qsym2(&bao, mol)
                    .map_err(|err| PyRuntimeError::new_err(err.to_string()))?
            };
            match &use_magnetic_group {
                Some(MagneticSymmetryAnalysisKind::Corepresentation) => {
                    let mut sda_driver = SlaterDeterminantRepAnalysisDriver::<
                        MagneticRepresentedSymmetryGroup,
                        f64,
                        SpinConstraint,
                    >::builder()
                    .parameters(&sda_params)
                    .angular_function_parameters(&afa_params)
                    .determinant(&det_r)
                    .sao_spatial(&sao_spatial)
                    .sao_spatial_h(None) // Real SAO.
                    .sao_spatial_4c(sao_spatial_4c.as_ref())
                    .sao_spatial_4c_h(None) // Real SAO.
                    .symmetry_group(&pd_res)
                    .build()
                    .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
                    py.allow_threads(|| {
                        sda_driver
                            .run()
                            .map_err(|err| PyRuntimeError::new_err(err.to_string()))
                    })?;
                    let sda_res = sda_driver
                        .result()
                        .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
                    PySlaterDeterminantRepAnalysisResult {
                        group: sda_res.group().name().clone(),
                        determinant_symmetry: sda_res
                            .determinant_symmetry()
                            .as_ref()
                            .ok()
                            .map(|sym| sym.to_string()),
                        mo_symmetries: sda_res.mo_symmetries().as_ref().map(|mo_symss| {
                            mo_symss
                                .iter()
                                .map(|mo_syms| {
                                    mo_syms
                                        .iter()
                                        .map(|mo_sym_opt| {
                                            mo_sym_opt.as_ref().map(|mo_sym| mo_sym.to_string())
                                        })
                                        .collect::<Vec<_>>()
                                })
                                .collect::<Vec<_>>()
                        }),
                        determinant_density_symmetries: sda_res
                            .determinant_density_symmetries()
                            .as_ref()
                            .map(|den_syms| {
                                den_syms
                                    .iter()
                                    .map(|(den_name, den_sym_res)| {
                                        (
                                            den_name.clone(),
                                            den_sym_res
                                                .as_ref()
                                                .ok()
                                                .map(|den_sym| den_sym.to_string()),
                                        )
                                    })
                                    .collect::<Vec<_>>()
                            }),
                        mo_density_symmetries: sda_res.mo_density_symmetries().as_ref().map(
                            |mo_den_symss| {
                                mo_den_symss
                                    .iter()
                                    .map(|mo_den_syms| {
                                        mo_den_syms
                                            .iter()
                                            .map(|mo_den_sym_opt| {
                                                mo_den_sym_opt
                                                    .as_ref()
                                                    .map(|mo_den_sym| mo_den_sym.to_string())
                                            })
                                            .collect::<Vec<_>>()
                                    })
                                    .collect::<Vec<_>>()
                            },
                        ),
                    }
                }
                Some(MagneticSymmetryAnalysisKind::Representation) | None => {
                    let mut sda_driver = SlaterDeterminantRepAnalysisDriver::<
                        UnitaryRepresentedSymmetryGroup,
                        f64,
                        SpinConstraint,
                    >::builder()
                    .parameters(&sda_params)
                    .angular_function_parameters(&afa_params)
                    .determinant(&det_r)
                    .sao_spatial(&sao_spatial)
                    .sao_spatial_h(None) // Real SAO.
                    .sao_spatial_4c(sao_spatial_4c.as_ref())
                    .sao_spatial_4c_h(None) // Real SAO.
                    .symmetry_group(&pd_res)
                    .build()
                    .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
                    py.allow_threads(|| {
                        sda_driver
                            .run()
                            .map_err(|err| PyRuntimeError::new_err(err.to_string()))
                    })?;
                    let sda_res = sda_driver
                        .result()
                        .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
                    PySlaterDeterminantRepAnalysisResult {
                        group: sda_res.group().name().clone(),
                        determinant_symmetry: sda_res
                            .determinant_symmetry()
                            .as_ref()
                            .ok()
                            .map(|sym| sym.to_string()),
                        mo_symmetries: sda_res.mo_symmetries().as_ref().map(|mo_symss| {
                            mo_symss
                                .iter()
                                .map(|mo_syms| {
                                    mo_syms
                                        .iter()
                                        .map(|mo_sym_opt| {
                                            mo_sym_opt.as_ref().map(|mo_sym| mo_sym.to_string())
                                        })
                                        .collect::<Vec<_>>()
                                })
                                .collect::<Vec<_>>()
                        }),
                        determinant_density_symmetries: sda_res
                            .determinant_density_symmetries()
                            .as_ref()
                            .map(|den_syms| {
                                den_syms
                                    .iter()
                                    .map(|(den_name, den_sym_res)| {
                                        (
                                            den_name.clone(),
                                            den_sym_res
                                                .as_ref()
                                                .ok()
                                                .map(|den_sym| den_sym.to_string()),
                                        )
                                    })
                                    .collect::<Vec<_>>()
                            }),
                        mo_density_symmetries: sda_res.mo_density_symmetries().as_ref().map(
                            |mo_den_symss| {
                                mo_den_symss
                                    .iter()
                                    .map(|mo_den_syms| {
                                        mo_den_syms
                                            .iter()
                                            .map(|mo_den_sym_opt| {
                                                mo_den_sym_opt
                                                    .as_ref()
                                                    .map(|mo_den_sym| mo_den_sym.to_string())
                                            })
                                            .collect::<Vec<_>>()
                                    })
                                    .collect::<Vec<_>>()
                            },
                        ),
                    }
                }
            }
        }
        (PySlaterDeterminant::Real(pydet_r), PyArray2RC::Complex(pysao_c)) => {
            let det_r = if augment_to_generalised {
                pydet_r
                    .to_qsym2(&bao, mol)
                    .map_err(|err| PyRuntimeError::new_err(err.to_string()))?
                    .to_generalised()
            } else {
                pydet_r
                    .to_qsym2(&bao, mol)
                    .map_err(|err| PyRuntimeError::new_err(err.to_string()))?
            };
            let det_c: SlaterDeterminant<C128, SpinConstraint> = det_r.into();
            let sao_spatial_c = pysao_c.to_owned_array();
            let sao_spatial_h_c = sao_spatial_h.and_then(|pysao_h| match pysao_h {
                // sao_spatial_h must have the same reality as sao_spatial.
                PyArray2RC::Real(_) => None,
                PyArray2RC::Complex(pysao_h_c) => Some(pysao_h_c.to_owned_array()),
            });
            let sao_spatial_4c_c = sao_spatial_4c.and_then(|pysao4c| match pysao4c {
                // sao_spatial_4c must have the same reality as sao_spatial.
                PyArray4RC::Real(_) => None,
                PyArray4RC::Complex(pysao4c_c) => Some(pysao4c_c.to_owned_array()),
            });
            let sao_spatial_4c_h_c = sao_spatial_4c_h.and_then(|pysao4c_h| match pysao4c_h {
                // sao_spatial_4c_h must have the same reality as sao_spatial.
                PyArray4RC::Real(_) => None,
                PyArray4RC::Complex(pysao4c_h_c) => Some(pysao4c_h_c.to_owned_array()),
            });
            match &use_magnetic_group {
                Some(MagneticSymmetryAnalysisKind::Corepresentation) => {
                    let mut sda_driver = SlaterDeterminantRepAnalysisDriver::<
                        MagneticRepresentedSymmetryGroup,
                        C128,
                        SpinConstraint,
                    >::builder()
                    .parameters(&sda_params)
                    .angular_function_parameters(&afa_params)
                    .determinant(&det_c)
                    .sao_spatial(&sao_spatial_c)
                    .sao_spatial_h(sao_spatial_h_c.as_ref())
                    .sao_spatial_4c(sao_spatial_4c_c.as_ref())
                    .sao_spatial_4c_h(sao_spatial_4c_h_c.as_ref())
                    .symmetry_group(&pd_res)
                    .build()
                    .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
                    py.allow_threads(|| {
                        sda_driver
                            .run()
                            .map_err(|err| PyRuntimeError::new_err(err.to_string()))
                    })?;
                    let sda_res = sda_driver
                        .result()
                        .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
                    PySlaterDeterminantRepAnalysisResult {
                        group: sda_res.group().name().clone(),
                        determinant_symmetry: sda_res
                            .determinant_symmetry()
                            .as_ref()
                            .ok()
                            .map(|sym| sym.to_string()),
                        mo_symmetries: sda_res.mo_symmetries().as_ref().map(|mo_symss| {
                            mo_symss
                                .iter()
                                .map(|mo_syms| {
                                    mo_syms
                                        .iter()
                                        .map(|mo_sym_opt| {
                                            mo_sym_opt.as_ref().map(|mo_sym| mo_sym.to_string())
                                        })
                                        .collect::<Vec<_>>()
                                })
                                .collect::<Vec<_>>()
                        }),
                        determinant_density_symmetries: sda_res
                            .determinant_density_symmetries()
                            .as_ref()
                            .map(|den_syms| {
                                den_syms
                                    .iter()
                                    .map(|(den_name, den_sym_res)| {
                                        (
                                            den_name.clone(),
                                            den_sym_res
                                                .as_ref()
                                                .ok()
                                                .map(|den_sym| den_sym.to_string()),
                                        )
                                    })
                                    .collect::<Vec<_>>()
                            }),
                        mo_density_symmetries: sda_res.mo_density_symmetries().as_ref().map(
                            |mo_den_symss| {
                                mo_den_symss
                                    .iter()
                                    .map(|mo_den_syms| {
                                        mo_den_syms
                                            .iter()
                                            .map(|mo_den_sym_opt| {
                                                mo_den_sym_opt
                                                    .as_ref()
                                                    .map(|mo_den_sym| mo_den_sym.to_string())
                                            })
                                            .collect::<Vec<_>>()
                                    })
                                    .collect::<Vec<_>>()
                            },
                        ),
                    }
                }
                Some(MagneticSymmetryAnalysisKind::Representation) | None => {
                    let mut sda_driver = SlaterDeterminantRepAnalysisDriver::<
                        UnitaryRepresentedSymmetryGroup,
                        C128,
                        SpinConstraint,
                    >::builder()
                    .parameters(&sda_params)
                    .angular_function_parameters(&afa_params)
                    .determinant(&det_c)
                    .sao_spatial(&sao_spatial_c)
                    .sao_spatial_h(sao_spatial_h_c.as_ref())
                    .sao_spatial_4c(sao_spatial_4c_c.as_ref())
                    .sao_spatial_4c_h(sao_spatial_4c_h_c.as_ref())
                    .symmetry_group(&pd_res)
                    .build()
                    .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
                    py.allow_threads(|| {
                        sda_driver
                            .run()
                            .map_err(|err| PyRuntimeError::new_err(err.to_string()))
                    })?;
                    let sda_res = sda_driver
                        .result()
                        .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
                    PySlaterDeterminantRepAnalysisResult {
                        group: sda_res.group().name().clone(),
                        determinant_symmetry: sda_res
                            .determinant_symmetry()
                            .as_ref()
                            .ok()
                            .map(|sym| sym.to_string()),
                        mo_symmetries: sda_res.mo_symmetries().as_ref().map(|mo_symss| {
                            mo_symss
                                .iter()
                                .map(|mo_syms| {
                                    mo_syms
                                        .iter()
                                        .map(|mo_sym_opt| {
                                            mo_sym_opt.as_ref().map(|mo_sym| mo_sym.to_string())
                                        })
                                        .collect::<Vec<_>>()
                                })
                                .collect::<Vec<_>>()
                        }),
                        determinant_density_symmetries: sda_res
                            .determinant_density_symmetries()
                            .as_ref()
                            .map(|den_syms| {
                                den_syms
                                    .iter()
                                    .map(|(den_name, den_sym_res)| {
                                        (
                                            den_name.clone(),
                                            den_sym_res
                                                .as_ref()
                                                .ok()
                                                .map(|den_sym| den_sym.to_string()),
                                        )
                                    })
                                    .collect::<Vec<_>>()
                            }),
                        mo_density_symmetries: sda_res.mo_density_symmetries().as_ref().map(
                            |mo_den_symss| {
                                mo_den_symss
                                    .iter()
                                    .map(|mo_den_syms| {
                                        mo_den_syms
                                            .iter()
                                            .map(|mo_den_sym_opt| {
                                                mo_den_sym_opt
                                                    .as_ref()
                                                    .map(|mo_den_sym| mo_den_sym.to_string())
                                            })
                                            .collect::<Vec<_>>()
                                    })
                                    .collect::<Vec<_>>()
                            },
                        ),
                    }
                }
            }
        }
        (PySlaterDeterminant::Complex(pydet_c), _) => {
            let det_c = if augment_to_generalised {
                pydet_c
                    .to_qsym2(&bao, mol)
                    .map_err(|err| PyRuntimeError::new_err(err.to_string()))?
                    .to_generalised()
            } else {
                pydet_c
                    .to_qsym2(&bao, mol)
                    .map_err(|err| PyRuntimeError::new_err(err.to_string()))?
            };
            let sao_spatial_c = match sao_spatial {
                PyArray2RC::Real(pysao_r) => pysao_r.to_owned_array().mapv(Complex::from),
                PyArray2RC::Complex(pysao_c) => pysao_c.to_owned_array(),
            };
            let sao_spatial_h_c = sao_spatial_h.and_then(|pysao_h| match pysao_h {
                // sao_spatial_h must have the same reality as sao_spatial.
                PyArray2RC::Real(pysao_h_r) => Some(pysao_h_r.to_owned_array().mapv(Complex::from)),
                PyArray2RC::Complex(pysao_h_c) => Some(pysao_h_c.to_owned_array()),
            });
            let sao_spatial_4c_c = sao_spatial_4c.and_then(|pysao4c| match pysao4c {
                // sao_spatial_4c must have the same reality as sao_spatial.
                PyArray4RC::Real(pysao4c_r) => Some(pysao4c_r.to_owned_array().mapv(Complex::from)),
                PyArray4RC::Complex(pysao4c_c) => Some(pysao4c_c.to_owned_array()),
            });
            let sao_spatial_4c_h_c = sao_spatial_4c_h.and_then(|pysao4c_h| match pysao4c_h {
                // sao_spatial_4c_h must have the same reality as sao_spatial.
                PyArray4RC::Real(pysao4c_h_r) => {
                    Some(pysao4c_h_r.to_owned_array().mapv(Complex::from))
                }
                PyArray4RC::Complex(pysao4c_h_c) => Some(pysao4c_h_c.to_owned_array()),
            });
            match &use_magnetic_group {
                Some(MagneticSymmetryAnalysisKind::Corepresentation) => {
                    let mut sda_driver = SlaterDeterminantRepAnalysisDriver::<
                        MagneticRepresentedSymmetryGroup,
                        C128,
                        SpinConstraint,
                    >::builder()
                    .parameters(&sda_params)
                    .angular_function_parameters(&afa_params)
                    .determinant(&det_c)
                    .sao_spatial(&sao_spatial_c)
                    .sao_spatial_h(sao_spatial_h_c.as_ref())
                    .sao_spatial_4c(sao_spatial_4c_c.as_ref())
                    .sao_spatial_4c_h(sao_spatial_4c_h_c.as_ref())
                    .symmetry_group(&pd_res)
                    .build()
                    .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
                    py.allow_threads(|| {
                        sda_driver
                            .run()
                            .map_err(|err| PyRuntimeError::new_err(err.to_string()))
                    })?;
                    let sda_res = sda_driver
                        .result()
                        .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
                    PySlaterDeterminantRepAnalysisResult {
                        group: sda_res.group().name().clone(),
                        determinant_symmetry: sda_res
                            .determinant_symmetry()
                            .as_ref()
                            .ok()
                            .map(|sym| sym.to_string()),
                        mo_symmetries: sda_res.mo_symmetries().as_ref().map(|mo_symss| {
                            mo_symss
                                .iter()
                                .map(|mo_syms| {
                                    mo_syms
                                        .iter()
                                        .map(|mo_sym_opt| {
                                            mo_sym_opt.as_ref().map(|mo_sym| mo_sym.to_string())
                                        })
                                        .collect::<Vec<_>>()
                                })
                                .collect::<Vec<_>>()
                        }),
                        determinant_density_symmetries: sda_res
                            .determinant_density_symmetries()
                            .as_ref()
                            .map(|den_syms| {
                                den_syms
                                    .iter()
                                    .map(|(den_name, den_sym_res)| {
                                        (
                                            den_name.clone(),
                                            den_sym_res
                                                .as_ref()
                                                .ok()
                                                .map(|den_sym| den_sym.to_string()),
                                        )
                                    })
                                    .collect::<Vec<_>>()
                            }),
                        mo_density_symmetries: sda_res.mo_density_symmetries().as_ref().map(
                            |mo_den_symss| {
                                mo_den_symss
                                    .iter()
                                    .map(|mo_den_syms| {
                                        mo_den_syms
                                            .iter()
                                            .map(|mo_den_sym_opt| {
                                                mo_den_sym_opt
                                                    .as_ref()
                                                    .map(|mo_den_sym| mo_den_sym.to_string())
                                            })
                                            .collect::<Vec<_>>()
                                    })
                                    .collect::<Vec<_>>()
                            },
                        ),
                    }
                }
                Some(MagneticSymmetryAnalysisKind::Representation) | None => {
                    let mut sda_driver = SlaterDeterminantRepAnalysisDriver::<
                        UnitaryRepresentedSymmetryGroup,
                        C128,
                        SpinConstraint,
                    >::builder()
                    .parameters(&sda_params)
                    .angular_function_parameters(&afa_params)
                    .determinant(&det_c)
                    .sao_spatial(&sao_spatial_c)
                    .sao_spatial_h(sao_spatial_h_c.as_ref())
                    .sao_spatial_4c(sao_spatial_4c_c.as_ref())
                    .sao_spatial_4c_h(sao_spatial_4c_h_c.as_ref())
                    .symmetry_group(&pd_res)
                    .build()
                    .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
                    py.allow_threads(|| {
                        sda_driver
                            .run()
                            .map_err(|err| PyRuntimeError::new_err(err.to_string()))
                    })?;
                    let sda_res = sda_driver
                        .result()
                        .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
                    PySlaterDeterminantRepAnalysisResult {
                        group: sda_res.group().name().clone(),
                        determinant_symmetry: sda_res
                            .determinant_symmetry()
                            .as_ref()
                            .ok()
                            .map(|sym| sym.to_string()),
                        mo_symmetries: sda_res.mo_symmetries().as_ref().map(|mo_symss| {
                            mo_symss
                                .iter()
                                .map(|mo_syms| {
                                    mo_syms
                                        .iter()
                                        .map(|mo_sym_opt| {
                                            mo_sym_opt.as_ref().map(|mo_sym| mo_sym.to_string())
                                        })
                                        .collect::<Vec<_>>()
                                })
                                .collect::<Vec<_>>()
                        }),
                        determinant_density_symmetries: sda_res
                            .determinant_density_symmetries()
                            .as_ref()
                            .map(|den_syms| {
                                den_syms
                                    .iter()
                                    .map(|(den_name, den_sym_res)| {
                                        (
                                            den_name.clone(),
                                            den_sym_res
                                                .as_ref()
                                                .ok()
                                                .map(|den_sym| den_sym.to_string()),
                                        )
                                    })
                                    .collect::<Vec<_>>()
                            }),
                        mo_density_symmetries: sda_res.mo_density_symmetries().as_ref().map(
                            |mo_den_symss| {
                                mo_den_symss
                                    .iter()
                                    .map(|mo_den_syms| {
                                        mo_den_syms
                                            .iter()
                                            .map(|mo_den_sym_opt| {
                                                mo_den_sym_opt
                                                    .as_ref()
                                                    .map(|mo_den_sym| mo_den_sym.to_string())
                                            })
                                            .collect::<Vec<_>>()
                                    })
                                    .collect::<Vec<_>>()
                            },
                        ),
                    }
                }
            }
        }
    };
    Ok(pysda_res)
}
