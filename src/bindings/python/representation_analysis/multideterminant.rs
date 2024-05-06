//! Python bindings for QSym² symmetry analysis of Slater determinants.

use std::collections::HashSet;
use std::path::PathBuf;

use anyhow::{bail, format_err, Context};
use duplicate::duplicate_item;
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
use crate::bindings::python::representation_analysis::slater_determinant::{
    PySlaterDeterminant, PySlaterDeterminantComplex, PySlaterDeterminantReal,
};
use crate::bindings::python::representation_analysis::{PyArray1RC, PyArray2RC};
use crate::drivers::representation_analysis::angular_function::AngularFunctionRepAnalysisParams;
use crate::drivers::representation_analysis::multideterminant::{
    MultiDeterminantRepAnalysisDriver, MultiDeterminantRepAnalysisParams,
};
use crate::drivers::representation_analysis::{
    CharacterTableDisplay, MagneticSymmetryAnalysisKind,
};
use crate::drivers::symmetry_group_detection::SymmetryGroupDetectionResult;
use crate::drivers::QSym2Driver;
use crate::io::format::qsym2_output;
use crate::io::{read_qsym2_binary, QSym2FileType};
use crate::symmetry::symmetry_group::{
    MagneticRepresentedSymmetryGroup, SymmetryGroupProperties, UnitaryRepresentedSymmetryGroup,
};
use crate::symmetry::symmetry_transformation::{SymmetryTransformable, SymmetryTransformationKind};
use crate::target::determinant::SlaterDeterminant;
use crate::target::noci::basis::{Basis, EagerBasis, OrbitBasis};
use crate::target::noci::multideterminant::MultiDeterminant;

type C128 = Complex<f64>;

// =====================
// Functions definitions
// =====================

/// Python-exposed function to perform representation symmetry analysis for real and complex
/// multi-determinantal wavefunctions constructed from group-generated orbits and log the result via
/// the `qsym2-output` logger at the `INFO` level.
///
/// If `symmetry_transformation_kind` includes spin transformation, the provided
/// multi-determinantal wavefunctions will be augmented to generalised spin constraint
/// automatically.
///
/// # Arguments
///
/// * `inp_sym` - A path to the [`QSym2FileType::Sym`] file containing the symmetry-group detection
/// result for the system. This will be used to construct abstract groups and character tables for
/// representation analysis. Python type: `str`.
/// * `pyorigins` - A list of Python-exposed Slater determinants whose coefficients are of type
/// `float64` or `complex128`. These determinants serve as origins for group-generated orbits which
/// serve as basis states for non-orthogonal configuration interaction to yield multi-determinantal
/// wavefunctions, the symmetry of which will be analysed by this function.
/// Python type: `list[PySlaterDeterminantReal | PySlaterDeterminantComplex]`.
/// * `py_noci_solver` - A Python function callable on a sequence of Slater determinants to perform
/// non-orthogonal configuration interaction (NOCI) and return a list of NOCI energies and a
/// corresponding list of lists of linear combination coefficients, where each inner list is for one
/// multi-determinantal wavefunction resulting from the NOCI calculation.
/// Python type: `Callable[
///     [list[PySlaterDeterminantReal | PySlaterDeterminantComplex]],
///     tuple[list[float], list[list[float]]] | tuple[list[complex], list[list[complex]]]
/// ]`.
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
    py_noci_solver,
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
    py_noci_solver: Py<PyFunction>,
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
    // Read in point-group detection results
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

    // Set up basic parameters
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
        .symmetry_transformation_kind(symmetry_transformation_kind.clone())
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

    // Set up NOCI function
    let noci_solver_r = |multidets: &Vec<SlaterDeterminant<f64>>| {
        Python::with_gil(|py_inner| {
            let pymultidets = multidets
                .iter()
                .map(|det| {
                    let pysc = det.spin_constraint().clone().try_into()?;
                    Ok(PySlaterDeterminantReal::new(
                        pysc,
                        det.complex_symmetric(),
                        det.coefficients()
                            .iter()
                            .map(|arr| PyArray2::from_array(py_inner, arr))
                            .collect::<Vec<_>>(),
                        det.occupations()
                            .iter()
                            .map(|arr| PyArray1::from_array(py_inner, arr))
                            .collect::<Vec<_>>(),
                        det.threshold(),
                        det.mo_energies().map(|mo_energies| {
                            mo_energies
                                .iter()
                                .map(|arr| PyArray1::from_array(py_inner, arr))
                                .collect::<Vec<_>>()
                        }),
                        det.energy().ok().cloned(),
                    ))
                })
                .collect::<Result<Vec<_>, anyhow::Error>>()
                .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
            py_noci_solver
                .call1(py_inner, (pymultidets,))
                .and_then(|res| res.extract::<(Vec<f64>, Vec<Vec<f64>>)>(py_inner))
        })
    };
    let noci_solver_c = |multidets: &Vec<SlaterDeterminant<C128>>| {
        Python::with_gil(|py_inner| {
            let pymultidets = multidets
                .iter()
                .map(|det| {
                    let pysc = det.spin_constraint().clone().try_into()?;
                    Ok(PySlaterDeterminantComplex::new(
                        pysc,
                        det.complex_symmetric(),
                        det.coefficients()
                            .iter()
                            .map(|arr| PyArray2::from_array(py_inner, arr))
                            .collect::<Vec<_>>(),
                        det.occupations()
                            .iter()
                            .map(|arr| PyArray1::from_array(py_inner, arr))
                            .collect::<Vec<_>>(),
                        det.threshold(),
                        det.mo_energies().map(|mo_energies| {
                            mo_energies
                                .iter()
                                .map(|arr| PyArray1::from_array(py_inner, arr))
                                .collect::<Vec<_>>()
                        }),
                        det.energy().ok().cloned(),
                    ))
                })
                .collect::<Result<Vec<_>, anyhow::Error>>()
                .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
            py_noci_solver
                .call1(py_inner, (pymultidets,))
                .and_then(|res| res.extract::<(Vec<C128>, Vec<Vec<C128>>)>(py_inner))
        })
    };

    let all_real = pyorigins
        .iter()
        .all(|pyorigin| matches!(pyorigin, PySlaterDeterminant::Real(_)));

    match (all_real, &sao_spatial) {
        (true, PyArray2RC::Real(pysao_r)) => {
            // Real numeric data type

            // Preparation
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
                    // Magnetic groups with corepresentations
                    let group = py
                        .allow_threads(|| {
                            let magsym = pd_res
                                .magnetic_symmetry
                                .as_ref()
                                .ok_or(format_err!("Magnetic group required for orbit construction, but no magnetic symmetry found."))?;
                            if use_double_group {
                                MagneticRepresentedSymmetryGroup::from_molecular_symmetry(
                                    magsym,
                                    infinite_order_to_finite,
                                )
                                .and_then(|grp| grp.to_double_group())
                            } else {
                                MagneticRepresentedSymmetryGroup::from_molecular_symmetry(
                                    magsym,
                                    infinite_order_to_finite,
                                )
                            }
                        })
                        .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;

                    // Construct the orbit basis
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

                    // Run non-orthogonal configuration interaction
                    let dets = orbit_basis
                        .iter()
                        .collect::<Result<Vec<_>, _>>()
                        .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;

                    let (noci_energies_vec, noci_coeffs_vec) = noci_solver_r(&dets)
                        .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
                    if noci_energies_vec.len() != noci_coeffs_vec.len()
                        || noci_coeffs_vec
                            .iter()
                            .map(|coeffs| coeffs.len())
                            .collect::<HashSet<_>>()
                            .len()
                            != 1
                    {
                        return Err(PyRuntimeError::new_err(
                            "Inconsistent dimensions encountered in NOCI results.",
                        ));
                    }
                    let multidets = noci_energies_vec
                        .into_iter()
                        .zip(noci_coeffs_vec.into_iter())
                        .map(|(energy, coeffs)| {
                            MultiDeterminant::builder()
                                .basis(orbit_basis.clone())
                                .coefficients(Array1::from_vec(coeffs))
                                .threshold(1e-7)
                                .energy(Ok(energy))
                                .build()
                        })
                        .collect::<Result<Vec<_>, _>>()
                        .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;

                    let mut mda_driver = MultiDeterminantRepAnalysisDriver::<
                        MagneticRepresentedSymmetryGroup,
                        f64,
                        _,
                    >::builder()
                    .parameters(&mda_params)
                    .angular_function_parameters(&afa_params)
                    .multidets(multidets.iter().collect::<Vec<_>>())
                    .sao_spatial(&sao_spatial)
                    .sao_spatial_h(None) // Real SAO.
                    .symmetry_group(&pd_res)
                    .build()
                    .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
                    py.allow_threads(|| {
                        mda_driver
                            .run()
                            .map_err(|err| PyRuntimeError::new_err(err.to_string()))
                    })?
                }
                Some(MagneticSymmetryAnalysisKind::Representation) | None => {
                    // Unitary groups or magnetic groups with representations
                    let group = py
                        .allow_threads(|| {
                            let sym = if use_magnetic_group.is_some() {
                                pd_res
                                    .magnetic_symmetry
                                    .as_ref()
                                    .ok_or(format_err!("Magnetic group required for orbit construction, but no magnetic symmetry found."))?
                            } else {
                                &pd_res.unitary_symmetry
                            };
                            if use_double_group {
                                UnitaryRepresentedSymmetryGroup::from_molecular_symmetry(
                                    sym,
                                    infinite_order_to_finite,
                                )
                                .and_then(|grp| grp.to_double_group())
                            } else {
                                UnitaryRepresentedSymmetryGroup::from_molecular_symmetry(
                                    sym,
                                    infinite_order_to_finite,
                                )
                            }
                        })
                        .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;

                    // Construct the orbit basis
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

                    // Run non-orthogonal configuration interaction
                    let dets = orbit_basis
                        .iter()
                        .collect::<Result<Vec<_>, _>>()
                        .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;

                    let (noci_energies_vec, noci_coeffs_vec) = noci_solver_r(&dets)
                        .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
                    if noci_energies_vec.len() != noci_coeffs_vec.len()
                        || noci_coeffs_vec
                            .iter()
                            .map(|coeffs| coeffs.len())
                            .collect::<HashSet<_>>()
                            .len()
                            != 1
                    {
                        return Err(PyRuntimeError::new_err(
                            "Inconsistent dimensions encountered in NOCI results.",
                        ));
                    }
                    let multidets = noci_energies_vec
                        .into_iter()
                        .zip(noci_coeffs_vec.into_iter())
                        .map(|(energy, coeffs)| {
                            MultiDeterminant::builder()
                                .basis(orbit_basis.clone())
                                .coefficients(Array1::from_vec(coeffs))
                                .threshold(1e-7)
                                .energy(Ok(energy))
                                .build()
                        })
                        .collect::<Result<Vec<_>, _>>()
                        .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;

                    let mut mda_driver = MultiDeterminantRepAnalysisDriver::<
                        UnitaryRepresentedSymmetryGroup,
                        f64,
                        _,
                    >::builder()
                    .parameters(&mda_params)
                    .angular_function_parameters(&afa_params)
                    .multidets(multidets.iter().collect::<Vec<_>>())
                    .sao_spatial(&sao_spatial)
                    .sao_spatial_h(None) // Real SAO.
                    .symmetry_group(&pd_res)
                    .build()
                    .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
                    py.allow_threads(|| {
                        mda_driver
                            .run()
                            .map_err(|err| PyRuntimeError::new_err(err.to_string()))
                    })?
                }
            };
        }
        (_, _) => {
            // Complex numeric data type

            // Preparation
            let sao_spatial_c = match sao_spatial {
                PyArray2RC::Real(pysao_r) => pysao_r.to_owned_array().mapv(Complex::from),
                PyArray2RC::Complex(pysao_c) => pysao_c.to_owned_array(),
            };
            let sao_spatial_h_c = sao_spatial_h.and_then(|pysao_h| match pysao_h {
                // sao_spatial_h must have the same reality as sao_spatial.
                PyArray2RC::Real(pysao_h_r) => Some(pysao_h_r.to_owned_array().mapv(Complex::from)),
                PyArray2RC::Complex(pysao_h_c) => Some(pysao_h_c.to_owned_array()),
            });
            let origins_c = if augment_to_generalised {
                pyorigins
                    .iter()
                    .map(|pydet| match pydet {
                        PySlaterDeterminant::Real(pydet_r) => pydet_r
                            .to_qsym2(&bao, mol)
                            .map(|det_r| SlaterDeterminant::<C128>::from(det_r).to_generalised()),
                        PySlaterDeterminant::Complex(pydet_c) => pydet_c
                            .to_qsym2(&bao, mol)
                            .map(|det_c| det_c.to_generalised()),
                    })
                    .collect::<Result<Vec<_>, _>>()
            } else {
                pyorigins
                    .iter()
                    .map(|pydet| match pydet {
                        PySlaterDeterminant::Real(pydet_r) => pydet_r
                            .to_qsym2(&bao, mol)
                            .map(|det_r| SlaterDeterminant::<C128>::from(det_r)),
                        PySlaterDeterminant::Complex(pydet_c) => pydet_c.to_qsym2(&bao, mol),
                    })
                    .collect::<Result<Vec<_>, _>>()
            }
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;

            match &use_magnetic_group {
                Some(MagneticSymmetryAnalysisKind::Corepresentation) => {
                    // Magnetic groups with corepresentations
                    let group = py
                        .allow_threads(|| {
                            let magsym = pd_res
                                .magnetic_symmetry
                                .as_ref()
                                .ok_or(format_err!("Magnetic group required for orbit construction, but no magnetic symmetry found."))?;
                            if use_double_group {
                                MagneticRepresentedSymmetryGroup::from_molecular_symmetry(
                                    magsym,
                                    infinite_order_to_finite,
                                )
                                .and_then(|grp| grp.to_double_group())
                            } else {
                                MagneticRepresentedSymmetryGroup::from_molecular_symmetry(
                                    magsym,
                                    infinite_order_to_finite,
                                )
                            }
                        })
                        .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;

                    // Construct the orbit basis
                    let orbit_basis = match symmetry_transformation_kind {
                        SymmetryTransformationKind::Spatial => {
                            OrbitBasis::builder()
                            .group(&group)
                            .origins(origins_c)
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
                            .origins(origins_c)
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
                            .origins(origins_c)
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
                            .origins(origins_c)
                            .action(|op, det| {
                                 det.sym_transform_spin_spatial(op).with_context(|| {
                                    format!("Unable to apply `{op}` spin-spatially on the origin multi-determinantal wavefunction")
                                })
                            })
                            .build()
                        }
                    }.map_err(|err| PyRuntimeError::new_err(err.to_string()))?;

                    // Run non-orthogonal configuration interaction
                    let dets = orbit_basis
                        .iter()
                        .collect::<Result<Vec<_>, _>>()
                        .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;

                    let (noci_energies_vec, noci_coeffs_vec) = noci_solver_c(&dets)
                        .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
                    if noci_energies_vec.len() != noci_coeffs_vec.len()
                        || noci_coeffs_vec
                            .iter()
                            .map(|coeffs| coeffs.len())
                            .collect::<HashSet<_>>()
                            .len()
                            != 1
                    {
                        return Err(PyRuntimeError::new_err(
                            "Inconsistent dimensions encountered in NOCI results.",
                        ));
                    }
                    let multidets = noci_energies_vec
                        .into_iter()
                        .zip(noci_coeffs_vec.into_iter())
                        .map(|(energy, coeffs)| {
                            MultiDeterminant::builder()
                                .basis(orbit_basis.clone())
                                .coefficients(Array1::from_vec(coeffs))
                                .threshold(1e-7)
                                .energy(Ok(energy))
                                .build()
                        })
                        .collect::<Result<Vec<_>, _>>()
                        .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;

                    let mut mda_driver = MultiDeterminantRepAnalysisDriver::<
                        MagneticRepresentedSymmetryGroup,
                        C128,
                        _,
                    >::builder()
                    .parameters(&mda_params)
                    .angular_function_parameters(&afa_params)
                    .multidets(multidets.iter().collect::<Vec<_>>())
                    .sao_spatial(&sao_spatial_c)
                    .sao_spatial_h(sao_spatial_h_c.as_ref())
                    .symmetry_group(&pd_res)
                    .build()
                    .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
                    py.allow_threads(|| {
                        mda_driver
                            .run()
                            .map_err(|err| PyRuntimeError::new_err(err.to_string()))
                    })?
                }
                Some(MagneticSymmetryAnalysisKind::Representation) | None => {
                    // Unitary groups or magnetic groups with representations
                    let group = py
                        .allow_threads(|| {
                            let sym = if use_magnetic_group.is_some() {
                                pd_res
                                    .magnetic_symmetry
                                    .as_ref()
                                    .ok_or(format_err!("Magnetic group required for orbit construction, but no magnetic symmetry found."))?
                            } else {
                                &pd_res.unitary_symmetry
                            };
                            if use_double_group {
                                UnitaryRepresentedSymmetryGroup::from_molecular_symmetry(
                                    sym,
                                    infinite_order_to_finite,
                                )
                                .and_then(|grp| grp.to_double_group())
                            } else {
                                UnitaryRepresentedSymmetryGroup::from_molecular_symmetry(
                                    sym,
                                    infinite_order_to_finite,
                                )
                            }
                        })
                        .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;

                    // Construct the orbit basis
                    let orbit_basis = match symmetry_transformation_kind {
                        SymmetryTransformationKind::Spatial => {
                            OrbitBasis::builder()
                            .group(&group)
                            .origins(origins_c)
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
                            .origins(origins_c)
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
                            .origins(origins_c)
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
                            .origins(origins_c)
                            .action(|op, det| {
                                 det.sym_transform_spin_spatial(op).with_context(|| {
                                    format!("Unable to apply `{op}` spin-spatially on the origin multi-determinantal wavefunction")
                                })
                            })
                            .build()
                        }
                    }.map_err(|err| PyRuntimeError::new_err(err.to_string()))?;

                    // Run non-orthogonal configuration interaction
                    let dets = orbit_basis
                        .iter()
                        .collect::<Result<Vec<_>, _>>()
                        .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;

                    let (noci_energies_vec, noci_coeffs_vec) = noci_solver_c(&dets)
                        .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
                    if noci_energies_vec.len() != noci_coeffs_vec.len()
                        || noci_coeffs_vec
                            .iter()
                            .map(|coeffs| coeffs.len())
                            .collect::<HashSet<_>>()
                            .len()
                            != 1
                    {
                        return Err(PyRuntimeError::new_err(
                            "Inconsistent dimensions encountered in NOCI results.",
                        ));
                    }
                    let multidets = noci_energies_vec
                        .into_iter()
                        .zip(noci_coeffs_vec.into_iter())
                        .map(|(energy, coeffs)| {
                            MultiDeterminant::builder()
                                .basis(orbit_basis.clone())
                                .coefficients(Array1::from_vec(coeffs))
                                .threshold(1e-7)
                                .energy(Ok(energy))
                                .build()
                        })
                        .collect::<Result<Vec<_>, _>>()
                        .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;

                    let mut mda_driver = MultiDeterminantRepAnalysisDriver::<
                        UnitaryRepresentedSymmetryGroup,
                        C128,
                        _,
                    >::builder()
                    .parameters(&mda_params)
                    .angular_function_parameters(&afa_params)
                    .multidets(multidets.iter().collect::<Vec<_>>())
                    .sao_spatial(&sao_spatial_c)
                    .sao_spatial_h(sao_spatial_h_c.as_ref())
                    .symmetry_group(&pd_res)
                    .build()
                    .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
                    py.allow_threads(|| {
                        mda_driver
                            .run()
                            .map_err(|err| PyRuntimeError::new_err(err.to_string()))
                    })?
                }
            };
        }
    }
    Ok(())
}

/// Python-exposed function to perform representation symmetry analysis for real and complex
/// multi-determinantal wavefunctions constructed from group-generated orbits and log the result via
/// the `qsym2-output` logger at the `INFO` level.
///
/// If `symmetry_transformation_kind` includes spin transformation, the provided
/// multi-determinantal wavefunctions will be augmented to generalised spin constraint
/// automatically.
///
/// # Arguments
///
/// * `inp_sym` - A path to the [`QSym2FileType::Sym`] file containing the symmetry-group detection
/// result for the system. This will be used to construct abstract groups and character tables for
/// representation analysis. Python type: `str`.
/// * `pyorigins` - A list of Python-exposed Slater determinants whose coefficients are of type
/// `float64` or `complex128`. These determinants serve as origins for group-generated orbits which
/// serve as basis states for non-orthogonal configuration interaction to yield multi-determinantal
/// wavefunctions, the symmetry of which will be analysed by this function.
/// Python type: `list[PySlaterDeterminantReal | PySlaterDeterminantComplex]`.
/// * `py_noci_solver` - A Python function callable on a sequence of Slater determinants to perform
/// non-orthogonal configuration interaction (NOCI) and return a list of NOCI energies and a
/// corresponding list of lists of linear combination coefficients, where each inner list is for one
/// multi-determinantal wavefunction resulting from the NOCI calculation.
/// Python type: `Callable[
///     [list[PySlaterDeterminantReal | PySlaterDeterminantComplex]],
///     tuple[list[float], list[list[float]]] | tuple[list[complex], list[list[complex]]]
/// ]`.
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
    pydets,
    coefficients,
    energies,
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
pub fn rep_analyse_multideterminants_eager_basis(
    py: Python<'_>,
    inp_sym: PathBuf,
    pydets: Vec<PySlaterDeterminant>,
    coefficients: PyArray2RC,
    energies: PyArray1RC,
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
    // Read in point-group detection results
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

    // Set up basic parameters
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
        .symmetry_transformation_kind(symmetry_transformation_kind.clone())
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

    let all_real = pydets
        .iter()
        .all(|pydet| matches!(pydet, PySlaterDeterminant::Real(_)));

    match (all_real, &coefficients, &energies, &sao_spatial) {
        (
            true,
            PyArray2RC::Real(pycoefficients_r),
            PyArray1RC::Real(pyenergies_r),
            PyArray2RC::Real(pysao_r),
        ) => {
            // Real numeric data type

            // Preparation
            let sao_spatial = pysao_r.to_owned_array();
            let coefficients_r = pycoefficients_r.to_owned_array();
            let energies_r = pyenergies_r.to_owned_array();
            let dets_r = if augment_to_generalised {
                pydets
                    .iter()
                    .map(|pydet| {
                        if let PySlaterDeterminant::Real(pydet_r) = pydet {
                            pydet_r
                                .to_qsym2(&bao, mol)
                                .map(|det_r| det_r.to_generalised())
                        } else {
                            bail!("Unexpected complex type for a Slater determinant.")
                        }
                    })
                    .collect::<Result<Vec<_>, _>>()
            } else {
                pydets
                    .iter()
                    .map(|pydet| {
                        if let PySlaterDeterminant::Real(pydet_r) = pydet {
                            pydet_r.to_qsym2(&bao, mol)
                        } else {
                            bail!("Unexpected complex type for a Slater determinant.")
                        }
                    })
                    .collect::<Result<Vec<_>, _>>()
            }
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;

            let n_energies = energies_r.len();
            if coefficients_r.shape()[1] != n_energies {
                return Err(PyRuntimeError::new_err(
                    "Mismatched number of NOCI energies and number of NOCI states.",
                ));
            }

            // Construct the eager basis
            let eager_basis = EagerBasis::builder()
                .elements(dets_r)
                .build()
                .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;

            // Construct the multi-determinantal wavefunctions
            let multidets = energies_r
                .iter()
                .zip(coefficients_r.columns())
                .map(|(energy, coeffs)| {
                    MultiDeterminant::builder()
                        .basis(eager_basis.clone())
                        .coefficients(coeffs.to_owned())
                        .threshold(1e-7)
                        .energy(Ok(*energy))
                        .build()
                })
                .collect::<Result<Vec<_>, _>>()
                .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;

            // Construct the driver
            let mut mda_driver = MultiDeterminantRepAnalysisDriver::<
                MagneticRepresentedSymmetryGroup,
                f64,
                _,
            >::builder()
            .parameters(&mda_params)
            .angular_function_parameters(&afa_params)
            .multidets(multidets.iter().collect::<Vec<_>>())
            .sao_spatial(&sao_spatial)
            .sao_spatial_h(None) // Real SAO.
            .symmetry_group(&pd_res)
            .build()
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;

            // Run the driver
            py.allow_threads(|| {
                mda_driver
                    .run()
                    .map_err(|err| PyRuntimeError::new_err(err.to_string()))
            })?
        }
        (_, _, _, _) => {
            // Complex numeric data type

            // Preparation
            let sao_spatial_c = match sao_spatial {
                PyArray2RC::Real(pysao_r) => pysao_r.to_owned_array().mapv(Complex::from),
                PyArray2RC::Complex(pysao_c) => pysao_c.to_owned_array(),
            };
            let sao_spatial_h_c = sao_spatial_h.and_then(|pysao_h| match pysao_h {
                // sao_spatial_h must have the same reality as sao_spatial.
                PyArray2RC::Real(pysao_h_r) => Some(pysao_h_r.to_owned_array().mapv(Complex::from)),
                PyArray2RC::Complex(pysao_h_c) => Some(pysao_h_c.to_owned_array()),
            });
            let coefficients_c = match coefficients {
                PyArray2RC::Real(pycoefficients_r) => {
                    pycoefficients_r.to_owned_array().mapv(Complex::from)
                }
                PyArray2RC::Complex(pycoefficients_c) => pycoefficients_c.to_owned_array(),
            };
            let energies_c = match energies {
                PyArray1RC::Real(pyenergies_r) => pyenergies_r.to_owned_array().mapv(Complex::from),
                PyArray1RC::Complex(pyenergies_c) => pyenergies_c.to_owned_array(),
            };
            let dets_c = if augment_to_generalised {
                pydets
                    .iter()
                    .map(|pydet| match pydet {
                        PySlaterDeterminant::Real(pydet_r) => pydet_r
                            .to_qsym2(&bao, mol)
                            .map(|det_r| SlaterDeterminant::<C128>::from(det_r).to_generalised()),
                        PySlaterDeterminant::Complex(pydet_c) => pydet_c
                            .to_qsym2(&bao, mol)
                            .map(|det_c| det_c.to_generalised()),
                    })
                    .collect::<Result<Vec<_>, _>>()
            } else {
                pydets
                    .iter()
                    .map(|pydet| match pydet {
                        PySlaterDeterminant::Real(pydet_r) => pydet_r
                            .to_qsym2(&bao, mol)
                            .map(|det_r| SlaterDeterminant::<C128>::from(det_r)),
                        PySlaterDeterminant::Complex(pydet_c) => pydet_c.to_qsym2(&bao, mol),
                    })
                    .collect::<Result<Vec<_>, _>>()
            }
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;

            let n_energies = energies_c.len();
            if coefficients_c.shape()[1] != n_energies {
                return Err(PyRuntimeError::new_err(
                    "Mismatched number of NOCI energies and number of NOCI states.",
                ));
            }

            // Construct the eager basis
            let eager_basis = EagerBasis::builder()
                .elements(dets_c)
                .build()
                .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;

            // Construct the multi-determinantal wavefunctions
            let multidets = energies_c
                .iter()
                .zip(coefficients_c.columns())
                .map(|(energy, coeffs)| {
                    MultiDeterminant::builder()
                        .basis(eager_basis.clone())
                        .coefficients(coeffs.to_owned())
                        .threshold(1e-7)
                        .energy(Ok(*energy))
                        .build()
                })
                .collect::<Result<Vec<_>, _>>()
                .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;

            // Construct the driver
            let mut mda_driver = MultiDeterminantRepAnalysisDriver::<
                MagneticRepresentedSymmetryGroup,
                C128,
                _,
            >::builder()
            .parameters(&mda_params)
            .angular_function_parameters(&afa_params)
            .multidets(multidets.iter().collect::<Vec<_>>())
            .sao_spatial(&sao_spatial_c)
            .sao_spatial_h(sao_spatial_h_c.as_ref())
            .symmetry_group(&pd_res)
            .build()
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;

            // Run the driver
            py.allow_threads(|| {
                mda_driver
                    .run()
                    .map_err(|err| PyRuntimeError::new_err(err.to_string()))
            })?
        }
    }
    Ok(())
}
