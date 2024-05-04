//! Python bindings for QSym² symmetry analysis of Slater determinants.

use std::path::PathBuf;

use anyhow::{bail, format_err, Context};
use ndarray::{Array1, Array2};
use num_complex::Complex;
use numpy::{PyArray1, PyArray2};
use pyo3::exceptions::{PyIOError, PyRuntimeError};
use pyo3::prelude::*;

use crate::analysis::EigenvalueComparisonMode;
use crate::auxiliary::molecule::Molecule;
use crate::basis::ao::BasisAngularOrder;
use crate::bindings::python::integrals::{PyBasisAngularOrder, PySpinConstraint};
use crate::bindings::python::representation_analysis::slater_determinant::PySlaterDeterminant;
use crate::bindings::python::representation_analysis::{PyArray2RC, PyArray4RC};
use crate::drivers::representation_analysis::angular_function::AngularFunctionRepAnalysisParams;
use crate::drivers::representation_analysis::multideterminant::{
    MultiDeterminantRepAnalysisDriver, MultiDeterminantRepAnalysisParams,
};
use crate::drivers::representation_analysis::{
    CharacterTableDisplay, MagneticSymmetryAnalysisKind,
};
use crate::drivers::symmetry_group_detection::SymmetryGroupDetectionResult;
use crate::drivers::QSym2Driver;
use crate::group::MagneticRepresentedGroup;
use crate::io::format::qsym2_output;
use crate::io::{read_qsym2_binary, QSym2FileType};
use crate::symmetry::symmetry_group::{
    MagneticRepresentedSymmetryGroup, SymmetryGroupProperties, UnitaryRepresentedSymmetryGroup,
};
use crate::symmetry::symmetry_transformation::{
    self, SymmetryTransformable, SymmetryTransformationKind,
};
use crate::target::determinant::SlaterDeterminant;
use crate::target::noci::basis::OrbitBasis;

type C128 = Complex<f64>;

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
#[pyfunction]
#[pyo3(signature = (
    inp_sym,
    pyorigins,
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
    write_overlap_eigenvalues=true,
    write_character_table=true,
    infinite_order_to_finite=None,
    angular_function_integrality_threshold=1e-7,
    angular_function_linear_independence_threshold=1e-7,
    angular_function_max_angular_momentum=2
))]
pub fn rep_analyse_multideterminants_orbit_basis(
    py: Python<'_>,
    inp_sym: PathBuf,
    pyorigins: Vec<PySlaterDeterminant>,
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
    write_overlap_eigenvalues: bool,
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
    let mda_params = MultiDeterminantRepAnalysisParams::<f64>::builder()
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

    let all_real = pyorigins
        .iter()
        .all(|pyorigin| matches!(pyorigin, PySlaterDeterminant::Real(_)));

    match (all_real, &sao_spatial) {
        (true, PyArray2RC::Real(pysao_r)) => {
            let sao_spatial = pysao_r.to_owned_array();
            let origins_r = if augment_to_generalised {
                pyorigins
                    .iter()
                    .map(|pydet| {
                        if let PySlaterDeterminant::Real(pydet_r) = pydet {
                            pydet_r
                                .to_qsym2(&bao, mol)
                                .map(|det_r| det_r.to_generalised())
                        } else {
                            bail!("Unexpected complex type for an origin Slater determinant.")
                        }
                    })
                    .collect::<Result<Vec<_>, _>>()
            } else {
                pyorigins
                    .iter()
                    .map(|pydet| {
                        if let PySlaterDeterminant::Real(pydet_r) = pydet {
                            pydet_r.to_qsym2(&bao, mol)
                        } else {
                            bail!("Unexpected complex type for an origin Slater determinant.")
                        }
                    })
                    .collect::<Result<Vec<_>, _>>()
            }
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
            match &use_magnetic_group {
                Some(MagneticSymmetryAnalysisKind::Corepresentation) => {
                    let group = if use_double_group {
                        MagneticRepresentedGroup::from_molecular_symmetry(
                            &pd_res.unitary_symmetry,
                            infinite_order_to_finite,
                        )
                        .and_then(|grp| grp.to_double_group())
                    } else {
                        MagneticRepresentedGroup::from_molecular_symmetry(
                            &pd_res.unitary_symmetry,
                            infinite_order_to_finite,
                        )
                    }
                    .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;

                    let orbit_basis = match symmetry_transformation_kind {
                        SymmetryTransformationKind::Spatial => {
                            OrbitBasis::builder()
                            .group(&group)
                            .origins(origins_r)
                            .action(|op, det| {
                                det.sym_transform_spatial(op).with_context(|| {
                                    format!("Unable to apply `{op}` spatially on the origin multi-determinantal wavefunction")
                                })
                            })
                            .build()
                        }
                        SymmetryTransformationKind::SpatialWithSpinTimeReversal => {
                            OrbitBasis::builder()
                            .group(&group)
                            .origins(origins_r)
                            .action(|op, det| {
                                 det.sym_transform_spatial_with_spintimerev(op).with_context(|| {
                                    format!("Unable to apply `{op}` spatially (with spin-including time reversal) on the origin multi-determinantal wavefunction")
                                })
                            })
                            .build()
                        }
                        SymmetryTransformationKind::Spin => {
                            OrbitBasis::builder()
                            .group(&group)
                            .origins(origins_r)
                            .action(|op, det| {
                                 det.sym_transform_spin(op).with_context(|| {
                                    format!("Unable to apply `{op}` spin-wise on the origin multi-determinantal wavefunction")
                                })
                            })
                            .build()
                        }
                        SymmetryTransformationKind::SpinSpatial => {
                            OrbitBasis::builder()
                            .group(&group)
                            .origins(origins_r)
                            .action(|op, det| {
                                 det.sym_transform_spin_spatial(op).with_context(|| {
                                    format!("Unable to apply `{op}` spin-spatially on the origin multi-determinantal wavefunction")
                                })
                            })
                            .build()
                        }
                    }.map_err(|err| PyRuntimeError::new_err(err.to_string()))?;

                    let mut mda_driver = MultiDeterminantRepAnalysisDriver::<
                        MagneticRepresentedSymmetryGroup,
                        f64,
                        _,
                    >::builder()
                    .parameters(&mda_params)
                    .angular_function_parameters(&afa_params)
                    .multidets(&det_r)
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
                    })?
                }
                Some(MagneticSymmetryAnalysisKind::Representation) | None => {
                    let mut sda_driver = SlaterDeterminantRepAnalysisDriver::<
                        UnitaryRepresentedSymmetryGroup,
                        f64,
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
                    })?
                }
            };
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
            let det_c: SlaterDeterminant<C128> = det_r.into();
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
                    })?
                }
                Some(MagneticSymmetryAnalysisKind::Representation) | None => {
                    let mut sda_driver = SlaterDeterminantRepAnalysisDriver::<
                        UnitaryRepresentedSymmetryGroup,
                        C128,
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
                    })?
                }
            };
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
                    })?
                }
                Some(MagneticSymmetryAnalysisKind::Representation) | None => {
                    let mut sda_driver = SlaterDeterminantRepAnalysisDriver::<
                        UnitaryRepresentedSymmetryGroup,
                        C128,
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
                    })?
                }
            };
        }
    }
    Ok(())
}
