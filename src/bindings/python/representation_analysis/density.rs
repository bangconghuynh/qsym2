//! Python bindings for QSym² symmetry analysis of electron densities.

use std::path::PathBuf;

use anyhow::format_err;
use ndarray::{Array2, Array4};
use num_complex::Complex;
use numpy::{PyArray2, PyArrayMethods};
use pyo3::exceptions::{PyIOError, PyRuntimeError};
use pyo3::prelude::*;

use crate::analysis::EigenvalueComparisonMode;
use crate::auxiliary::molecule::Molecule;
use crate::basis::ao::BasisAngularOrder;
use crate::bindings::python::integrals::{PyBasisAngularOrder, PySpinorBalanceSymmetryAux};
use crate::bindings::python::representation_analysis::PyArray4RC;
use crate::drivers::QSym2Driver;
use crate::drivers::representation_analysis::angular_function::AngularFunctionRepAnalysisParams;
use crate::drivers::representation_analysis::density::{
    DensityRepAnalysisDriver, DensityRepAnalysisParams,
};
use crate::drivers::representation_analysis::{
    CharacterTableDisplay, MagneticSymmetryAnalysisKind,
};
use crate::drivers::symmetry_group_detection::SymmetryGroupDetectionResult;
use crate::io::format::qsym2_output;
use crate::io::{QSym2FileType, read_qsym2_binary};
use crate::symmetry::symmetry_group::{
    MagneticRepresentedSymmetryGroup, UnitaryRepresentedSymmetryGroup,
};
use crate::symmetry::symmetry_transformation::SymmetryTransformationKind;
use crate::target::density::Density;

type C128 = Complex<f64>;

// ==================
// Struct definitions
// ==================

/// Python-exposed structure to marshall real electron density information between Rust and
/// Python.
///
/// # Constructor arguments
///
/// * `complex_symmetric` - A boolean indicating if inner products involving this density
/// are complex-symmetric. Python type: `bool`.
/// * `density_matrix` - The real density matrix describing this density.
/// Python type: `numpy.2darray[float]`.
/// * `threshold` - The threshold for comparisons. Python type: `float`.
#[pyclass]
#[derive(Clone)]
pub struct PyDensityReal {
    /// A boolean indicating if inner products involving this density should be the
    /// complex-symmetric bilinear form, rather than the conventional Hermitian sesquilinear form.
    ///
    /// Python type: `bool`.
    complex_symmetric: bool,

    /// The real density matrix describing this density.
    ///
    /// Python type: `numpy.2darray[float]`.
    density_matrix: Array2<f64>,

    /// The threshold for comparing densities.
    ///
    /// Python type: `float`.
    threshold: f64,
}

#[pymethods]
impl PyDensityReal {
    /// Constructs a real Python-exposed electron density.
    ///
    /// # Arguments
    ///
    /// * `complex_symmetric` - A boolean indicating if inner products involving this density
    /// are complex-symmetric. Python type: `bool`.
    /// * `density_matrix` - The real density matrix describing this density.
    /// Python type: `numpy.2darray[float]`.
    /// * `threshold` - The threshold for comparisons. Python type: `float`.
    #[new]
    fn new(
        complex_symmetric: bool,
        density_matrix: Bound<'_, PyArray2<f64>>,
        threshold: f64,
    ) -> Self {
        let det = Self {
            complex_symmetric,
            density_matrix: density_matrix.to_owned_array(),
            threshold,
        };
        det
    }
}

impl PyDensityReal {
    /// Extracts the information in the [`PyDensityReal`] structure into `QSym2`'s native
    /// [`Density`] structure.
    ///
    /// # Arguments
    ///
    /// * `bao` - The [`BasisAngularOrder`] for the basis set in which the density is given.
    /// * `mol` - The molecule with which the density is associated.
    ///
    /// # Returns
    ///
    /// The [`Density`] structure with the same information.
    ///
    /// # Errors
    ///
    /// Errors if the [`Density`] fails to build.
    fn to_qsym2<'b, 'a: 'b>(
        &'b self,
        bao: &'a BasisAngularOrder,
        mol: &'a Molecule,
    ) -> Result<Density<'b, f64>, anyhow::Error> {
        let den = Density::<f64>::builder()
            .bao(bao)
            .complex_symmetric(self.complex_symmetric)
            .mol(mol)
            .density_matrix(self.density_matrix.clone())
            .threshold(self.threshold)
            .build()
            .map_err(|err| format_err!(err));
        den
    }
}

/// Python-exposed structure to marshall complex electron density information between Rust and
/// Python.
///
/// # Constructor arguments
///
/// * `complex_symmetric` - A boolean indicating if inner products involving this density
/// are complex-symmetric. Python type: `bool`.
/// * `density_matrix` - The complex density matrix describing this density.
/// Python type: `numpy.2darray[complex]`.
/// * `threshold` - The threshold for comparisons. Python type: `float`.
#[pyclass]
#[derive(Clone)]
pub struct PyDensityComplex {
    /// A boolean indicating if inner products involving this density should be the
    /// complex-symmetric bilinear form, rather than the conventional Hermitian sesquilinear form.
    ///
    /// Python type: `bool`.
    complex_symmetric: bool,

    /// The complex density matrix describing this density.
    ///
    /// Python type: `numpy.2darray[complex]`.
    density_matrix: Array2<C128>,

    /// The threshold for comparing densities.
    ///
    /// Python type: `float`.
    threshold: f64,
}

#[pymethods]
impl PyDensityComplex {
    /// Constructs a complex Python-exposed electron density.
    ///
    /// # Arguments
    ///
    /// * `complex_symmetric` - A boolean indicating if inner products involving this density
    /// are complex-symmetric. Python type: `bool`.
    /// * `density_matrix` - The complex density matrix describing this density.
    /// Python type: `numpy.2darray[complex]`.
    /// * `threshold` - The threshold for comparisons. Python type: `float`.
    #[new]
    fn new(
        complex_symmetric: bool,
        density_matrix: Bound<'_, PyArray2<C128>>,
        threshold: f64,
    ) -> Self {
        let det = Self {
            complex_symmetric,
            density_matrix: density_matrix.to_owned_array(),
            threshold,
        };
        det
    }
}

impl PyDensityComplex {
    /// Extracts the information in the [`PyDensityComplex`] structure into `QSym2`'s native
    /// [`Density`] structure.
    ///
    /// # Arguments
    ///
    /// * `bao` - The [`BasisAngularOrder`] for the basis set in which the density is given.
    /// * `mol` - The molecule with which the density is associated.
    ///
    /// # Returns
    ///
    /// The [`Density`] structure with the same information.
    ///
    /// # Errors
    ///
    /// Errors if the [`Density`] fails to build.
    fn to_qsym2<'b, 'a: 'b>(
        &'b self,
        bao: &'a BasisAngularOrder,
        mol: &'a Molecule,
    ) -> Result<Density<'b, C128>, anyhow::Error> {
        let den = Density::<C128>::builder()
            .bao(bao)
            .complex_symmetric(self.complex_symmetric)
            .mol(mol)
            .density_matrix(self.density_matrix.clone())
            .threshold(self.threshold)
            .build()
            .map_err(|err| format_err!(err));
        den
    }
}

// ================
// Enum definitions
// ================

/// Python-exposed enumerated type to handle the union type `PyDensityReal | PyDensityComplex` in
/// Python.
#[derive(FromPyObject)]
pub enum PyDensity {
    /// Variant for real Python-exposed electron density.
    Real(PyDensityReal),

    /// Variant for complex Python-exposed electron density.
    Complex(PyDensityComplex),
}

// =====================
// Functions definitions
// =====================

/// Python-exposed function to perform representation symmetry analysis for real and complex
/// electron densities and log the result via the `qsym2-output` logger at the `INFO` level.
///
/// # Arguments
///
/// * `inp_sym` - A path to the [`QSym2FileType::Sym`] file containing the symmetry-group detection
/// result for the system. This will be used to construct abstract groups and character tables for
/// representation analysis. Python type: `str`.
/// * `pydens` - A sequence of Python-exposed electron densities whose density matrices are of type
/// `float64` or `complex128`. Each density is accompanied by a description string.
/// Python type: `list[tuple[str, PyDensityReal | PyDensityComplex]]`.
/// * `pybao` - Python-exposed structure containing basis angular order information for the density
/// matrices. Python type: `PyBasisAngularOrder`.
/// * `pybalance_symmetry_auxs` - The optional balance symmetry auxiliary information object.
/// Python type: `numpy.3darray[complex] | None`
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
/// transformations to be performed on the origin electron density to generate the orbit. Python
/// type: `SymmetryTransformationKind`.
/// * `eigenvalue_comparison_mode` - An enumerated type indicating the mode of comparison of orbit
/// overlap eigenvalues with the specified `linear_independence_threshold`.
/// Python type: `EigenvalueComparisonMode`.
/// * `sao_spatial_4c` - The atomic-orbital four-centre overlap matrix whose elements are of type
/// `float64` or `complex128`. Python type: `numpy.4darray[float] | numpy.4darray[complex]`.
/// * `sao_spatial_4c_h` - The optional complex-symmetric atomic-orbital four-centre overlap matrix
/// whose elements are of type `float64` or `complex128`. This is required if antiunitary symmetry
/// operations are involved. Python type: `numpy.2darray[float] | numpy.2darray[complex] | None`.
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
    pydens,
    pybao,
    pybalance_symmetry_aux,
    integrality_threshold,
    linear_independence_threshold,
    use_magnetic_group,
    use_double_group,
    use_cayley_table,
    symmetry_transformation_kind,
    eigenvalue_comparison_mode,
    sao_spatial_4c,
    sao_spatial_4c_h=None,
    write_character_table=true,
    infinite_order_to_finite=None,
    angular_function_integrality_threshold=1e-7,
    angular_function_linear_independence_threshold=1e-7,
    angular_function_max_angular_momentum=2
))]
pub fn rep_analyse_densities(
    py: Python<'_>,
    inp_sym: PathBuf,
    pydens: Vec<(String, PyDensity)>,
    pybao: &PyBasisAngularOrder,
    pybalance_symmetry_aux: Option<PySpinorBalanceSymmetryAux>,
    integrality_threshold: f64,
    linear_independence_threshold: f64,
    use_magnetic_group: Option<MagneticSymmetryAnalysisKind>,
    use_double_group: bool,
    use_cayley_table: bool,
    symmetry_transformation_kind: SymmetryTransformationKind,
    eigenvalue_comparison_mode: EigenvalueComparisonMode,
    sao_spatial_4c: PyArray4RC,
    sao_spatial_4c_h: Option<PyArray4RC>,
    write_character_table: bool,
    infinite_order_to_finite: Option<u32>,
    angular_function_integrality_threshold: f64,
    angular_function_linear_independence_threshold: f64,
    angular_function_max_angular_momentum: u32,
) -> PyResult<()> {
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
    let bsa_opt = pybalance_symmetry_aux
        .as_ref()
        .map(|pybsa| pybsa.to_qsym2());
    let bao = pybao
        .to_qsym2(mol, bsa_opt)
        .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
    let afa_params = AngularFunctionRepAnalysisParams::builder()
        .integrality_threshold(angular_function_integrality_threshold)
        .linear_independence_threshold(angular_function_linear_independence_threshold)
        .max_angular_momentum(angular_function_max_angular_momentum)
        .build()
        .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
    let sda_params = DensityRepAnalysisParams::<f64>::builder()
        .integrality_threshold(integrality_threshold)
        .linear_independence_threshold(linear_independence_threshold)
        .use_magnetic_group(use_magnetic_group.clone())
        .use_double_group(use_double_group)
        .use_cayley_table(use_cayley_table)
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

    let any_complex = pydens
        .iter()
        .any(|(_, pyden)| matches!(pyden, PyDensity::Complex(_)));

    match (any_complex, &sao_spatial_4c) {
        (false, PyArray4RC::Real(pysao4c_r)) => {
            // Both coefficients and sao_4c are real.
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

            let sao_spatial_4c = pysao4c_r.to_owned_array();

            match &use_magnetic_group {
                Some(MagneticSymmetryAnalysisKind::Corepresentation) => {
                    let mut da_driver =
                        DensityRepAnalysisDriver::<MagneticRepresentedSymmetryGroup, f64>::builder(
                        )
                        .parameters(&sda_params)
                        .angular_function_parameters(&afa_params)
                        .densities(dens_ref)
                        .sao_spatial_4c(&sao_spatial_4c)
                        .sao_spatial_4c_h(None)
                        .symmetry_group(&pd_res)
                        .build()
                        .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
                    py.allow_threads(|| {
                        da_driver
                            .run()
                            .map_err(|err| PyRuntimeError::new_err(err.to_string()))
                    })?
                }
                Some(MagneticSymmetryAnalysisKind::Representation) | None => {
                    let mut da_driver =
                        DensityRepAnalysisDriver::<UnitaryRepresentedSymmetryGroup, f64>::builder()
                            .parameters(&sda_params)
                            .angular_function_parameters(&afa_params)
                            .densities(dens_ref)
                            .sao_spatial_4c(&sao_spatial_4c)
                            .sao_spatial_4c_h(None)
                            .symmetry_group(&pd_res)
                            .build()
                            .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
                    py.allow_threads(|| {
                        da_driver
                            .run()
                            .map_err(|err| PyRuntimeError::new_err(err.to_string()))
                    })?
                }
            };
        }
        (_, _) => {
            // At least one of coefficients or sao_4c are not real.
            let dens: Vec<Density<C128>> = pydens
                .iter()
                .map(|(_, pyden)| match pyden {
                    PyDensity::Real(pyden_r) => {
                        pyden_r.to_qsym2(&bao, mol).map(|den_r| den_r.into())
                    }
                    PyDensity::Complex(pyden_c) => pyden_c.to_qsym2(&bao, mol),
                })
                .collect::<Result<Vec<_>, _>>()
                .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
            let dens_ref = dens
                .iter()
                .zip(pydens.iter())
                .map(|(den, (desc, _))| (desc.clone(), den))
                .collect::<Vec<_>>();

            let (sao_spatial_4c_c, sao_spatial_4c_h_c): (Array4<C128>, Option<Array4<C128>>) =
                match sao_spatial_4c {
                    PyArray4RC::Real(pysao4c_r) => {
                        (pysao4c_r.to_owned_array().mapv(Complex::from), None)
                    }
                    PyArray4RC::Complex(pysao4c_c) => (
                        pysao4c_c.to_owned_array(),
                        sao_spatial_4c_h.map(|v| match v {
                            PyArray4RC::Real(pysao4c_h_r) => {
                                pysao4c_h_r.to_owned_array().mapv(Complex::from)
                            }
                            PyArray4RC::Complex(pysao4c_h_c) => pysao4c_h_c.to_owned_array(),
                        }),
                    ),
                };

            match &use_magnetic_group {
                Some(MagneticSymmetryAnalysisKind::Corepresentation) => {
                    let mut da_driver = DensityRepAnalysisDriver::<
                        MagneticRepresentedSymmetryGroup,
                        C128,
                    >::builder()
                    .parameters(&sda_params)
                    .angular_function_parameters(&afa_params)
                    .densities(dens_ref)
                    .sao_spatial_4c(&sao_spatial_4c_c)
                    .sao_spatial_4c_h(sao_spatial_4c_h_c.as_ref())
                    .symmetry_group(&pd_res)
                    .build()
                    .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
                    py.allow_threads(|| {
                        da_driver
                            .run()
                            .map_err(|err| PyRuntimeError::new_err(err.to_string()))
                    })?
                }
                Some(MagneticSymmetryAnalysisKind::Representation) | None => {
                    let mut da_driver =
                        DensityRepAnalysisDriver::<UnitaryRepresentedSymmetryGroup, C128>::builder(
                        )
                        .parameters(&sda_params)
                        .angular_function_parameters(&afa_params)
                        .densities(dens_ref)
                        .sao_spatial_4c(&sao_spatial_4c_c)
                        .sao_spatial_4c_h(sao_spatial_4c_h_c.as_ref())
                        .symmetry_group(&pd_res)
                        .build()
                        .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
                    py.allow_threads(|| {
                        da_driver
                            .run()
                            .map_err(|err| PyRuntimeError::new_err(err.to_string()))
                    })?
                }
            };
        }
    }
    Ok(())
}
