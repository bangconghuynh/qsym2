//! Python bindings for QSym² symmetry analysis of multi-determinants with eager bases.

use std::collections::HashSet;
use std::path::PathBuf;

use anyhow::bail;
use itertools::Itertools;
use num_complex::Complex;
use numpy::PyArrayMethods;
use pyo3::exceptions::{PyIOError, PyRuntimeError};
use pyo3::prelude::*;

use crate::analysis::EigenvalueComparisonMode;
use crate::angmom::spinor_rotation_3d::{SpinConstraint, SpinOrbitCoupled};
use crate::bindings::python::integrals::{
    PyBasisAngularOrder, PySpinorBalanceSymmetryAux, PyStructureConstraint,
};
use crate::bindings::python::representation_analysis::slater_determinant::PySlaterDeterminant;
use crate::bindings::python::representation_analysis::{PyArray1RC, PyArray2RC};
use crate::drivers::QSym2Driver;
use crate::drivers::representation_analysis::angular_function::AngularFunctionRepAnalysisParams;
use crate::drivers::representation_analysis::multideterminant::{
    MultiDeterminantRepAnalysisDriver, MultiDeterminantRepAnalysisParams,
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
use crate::target::determinant::SlaterDeterminant;
use crate::target::noci::basis::EagerBasis;
use crate::target::noci::multideterminant::MultiDeterminant;

type C128 = Complex<f64>;

/// Python-exposed function to perform representation symmetry analysis for real and complex
/// multi-determinantal wavefunctions constructed from an eager basis of Slater determinants and log
/// the result via the `qsym2-output` logger at the `INFO` level.
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
/// * `pydets` - A list of Python-exposed Slater determinants whose coefficients are of type
/// `float64` or `complex128`. These determinants serve as basis states for non-orthogonal
/// configuration interaction to yield multi-determinantal wavefunctions, the symmetry of which will
/// be analysed by this function.
/// Python type: `list[PySlaterDeterminantReal | PySlaterDeterminantComplex]`.
/// * `coefficients` - The coefficient matrix where each column gives the linear combination
/// coefficients for one multi-determinantal wavefunction. The number of rows must match the number
/// of determinants specified in `pydets`. The elements are of type `float64` or `complex128`.
/// Python type: `numpy.2darray[float] | numpy.2darray[complex]`.
/// * `energies` - The `float64` or `complex128` energies of the multi-determinantal wavefunctions.
/// The number of terms must match the number of columns of `coefficients`.
/// Python type: `numpy.1darray[float] | numpy.1darray[complex]`.
/// * `pybaos` - Python-exposed structures containing basis angular order information, one for each
/// explicit component per coefficient matrix. Python type: `list[PyBasisAngularOrder]`.
/// * `pybalance_symmetry_auxs` - Optional balance symmetry auxiliary information objects, each
/// corresponding to a `pybao` structure specified.
/// Python type: `list[numpy.3darray[complex] | None]`
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
/// * `sao` - The atomic-orbital overlap matrix whose elements are of type `float64` or
/// `complex128`. Python type: `numpy.2darray[float] | numpy.2darray[complex]`.
/// * `sao_h` - The optional complex-symmetric atomic-orbital overlap matrix whose elements
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
    pybaos,
    pybalance_symmetry_auxs,
    integrality_threshold,
    linear_independence_threshold,
    use_magnetic_group,
    use_double_group,
    use_cayley_table,
    symmetry_transformation_kind,
    eigenvalue_comparison_mode,
    sao,
    sao_h=None,
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
    pybaos: Vec<PyBasisAngularOrder>,
    pybalance_symmetry_auxs: Vec<Option<PySpinorBalanceSymmetryAux>>,
    integrality_threshold: f64,
    linear_independence_threshold: f64,
    use_magnetic_group: Option<MagneticSymmetryAnalysisKind>,
    use_double_group: bool,
    use_cayley_table: bool,
    symmetry_transformation_kind: SymmetryTransformationKind,
    eigenvalue_comparison_mode: EigenvalueComparisonMode,
    sao: PyArray2RC,
    sao_h: Option<PyArray2RC>,
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

    assert_eq!(
        pybaos.len(),
        pybalance_symmetry_auxs.len(),
        "Mismatched numbers of `pybaos` and `pybalance_symmetry_auxs` items."
    );
    let baos = pybaos
        .iter()
        .zip(pybalance_symmetry_auxs.iter())
        .map(|(bao, pybsa_opt)| {
            let bsa_opt = pybsa_opt.as_ref().map(|pybsa| pybsa.to_qsym2());
            bao.to_qsym2(mol, bsa_opt)
                .map_err(|err| PyRuntimeError::new_err(err.to_string()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let baos_ref = baos.iter().collect::<Vec<_>>();
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

    let structure_constraints_set = pydets
        .iter()
        .map(|pydet| match pydet {
            PySlaterDeterminant::Real(pydet) => pydet.structure_constraint().clone(),
            PySlaterDeterminant::Complex(pydet) => pydet.structure_constraint().clone(),
        })
        .collect::<HashSet<_>>();
    if structure_constraints_set.len() != 1 {
        return Err(PyRuntimeError::new_err(
            "Inconsistent structure constraints across origin determinants.`",
        ));
    };
    let structure_constraint = structure_constraints_set
        .iter()
        .next()
        .ok_or_else(|| PyRuntimeError::new_err("Unable to retrieve the structure constraint."))?;

    // Decision tree:
    // - all real numerical data?
    //   + yes:
    //     - structure_constraint:
    //       + SpinConstraint:
    //         - use_magnetic_group:
    //           + Some(Corepresentation)
    //           + Some(Representation) | None
    //       + SpinOrbitCoupled: not supported
    //   - no:
    //     - structure_constraint:
    //       + SpinConstraint:
    //         - use_magnetic_group:
    //           + Some(Corepresentation)
    //           + Some(Representation) | None
    //       + SpinOrbitCoupled:
    //         - use_magnetic_group:
    //           + Some(Corepresentation)
    //           + Some(Representation) | None
    match (all_real, &coefficients, &energies, &sao) {
        (
            true,
            PyArray2RC::Real(pycoefficients_r),
            PyArray1RC::Real(pyenergies_r),
            PyArray2RC::Real(pysao_r),
        ) => {
            // Real numeric data type

            if matches!(
                structure_constraint,
                PyStructureConstraint::SpinOrbitCoupled(_)
            ) {
                return Err(PyRuntimeError::new_err(
                    "Real determinants cannot support spin--orbit-coupled structure constraint.",
                ));
            }

            // Preparation
            let sao = pysao_r.to_owned_array();
            let coefficients_r = pycoefficients_r.to_owned_array();
            let energies_r = pyenergies_r.to_owned_array();
            let dets_r = if augment_to_generalised {
                pydets
                    .iter()
                    .map(|pydet| {
                        if let PySlaterDeterminant::Real(pydet_r) = pydet {
                            pydet_r
                                .to_qsym2(&baos_ref, mol)
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
                            pydet_r.to_qsym2(&baos_ref, mol)
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
            match &use_magnetic_group {
                Some(MagneticSymmetryAnalysisKind::Corepresentation) => {
                    let mut mda_driver = MultiDeterminantRepAnalysisDriver::<
                        MagneticRepresentedSymmetryGroup,
                        f64,
                        _,
                        SpinConstraint,
                    >::builder()
                    .parameters(&mda_params)
                    .angular_function_parameters(&afa_params)
                    .multidets(multidets.iter().collect::<Vec<_>>())
                    .sao(&sao)
                    .sao_h(None) // Real SAO.
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
                Some(MagneticSymmetryAnalysisKind::Representation) | None => {
                    let mut mda_driver = MultiDeterminantRepAnalysisDriver::<
                        UnitaryRepresentedSymmetryGroup,
                        f64,
                        _,
                        SpinConstraint,
                    >::builder()
                    .parameters(&mda_params)
                    .angular_function_parameters(&afa_params)
                    .multidets(multidets.iter().collect::<Vec<_>>())
                    .sao(&sao)
                    .sao_h(None) // Real SAO.
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
        }
        (_, _, _, _) => {
            // Complex numeric data type

            // Preparation
            let sao_c = match sao {
                PyArray2RC::Real(pysao_r) => pysao_r.to_owned_array().mapv(Complex::from),
                PyArray2RC::Complex(pysao_c) => pysao_c.to_owned_array(),
            };
            let sao_h_c = sao_h.and_then(|pysao_h| match pysao_h {
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

            match structure_constraint {
                PyStructureConstraint::SpinConstraint(_) => {
                    let dets_c = if augment_to_generalised {
                        pydets
                            .iter()
                            .map(|pydet| match pydet {
                                PySlaterDeterminant::Real(pydet_r) => pydet_r
                                    .to_qsym2::<SpinConstraint>(&baos_ref, mol)
                                    .map(|det_r| {
                                        SlaterDeterminant::<C128, SpinConstraint>::from(det_r)
                                            .to_generalised()
                                    }),
                                PySlaterDeterminant::Complex(pydet_c) => pydet_c
                                    .to_qsym2::<SpinConstraint>(&baos_ref, mol)
                                    .map(|det_c| det_c.to_generalised()),
                            })
                            .collect::<Result<Vec<_>, _>>()
                    } else {
                        pydets
                            .iter()
                            .map(|pydet| match pydet {
                                PySlaterDeterminant::Real(pydet_r) => pydet_r
                                    .to_qsym2::<SpinConstraint>(&baos_ref, mol)
                                    .map(|det_r| {
                                        SlaterDeterminant::<C128, SpinConstraint>::from(det_r)
                                    }),
                                PySlaterDeterminant::Complex(pydet_c) => {
                                    pydet_c.to_qsym2::<SpinConstraint>(&baos_ref, mol)
                                }
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
                    match &use_magnetic_group {
                        Some(MagneticSymmetryAnalysisKind::Corepresentation) => {
                            let mut mda_driver = MultiDeterminantRepAnalysisDriver::<
                                MagneticRepresentedSymmetryGroup,
                                C128,
                                _,
                                SpinConstraint,
                            >::builder()
                            .parameters(&mda_params)
                            .angular_function_parameters(&afa_params)
                            .multidets(multidets.iter().collect::<Vec<_>>())
                            .sao(&sao_c)
                            .sao_h(sao_h_c.as_ref())
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
                        Some(MagneticSymmetryAnalysisKind::Representation) | None => {
                            let mut mda_driver = MultiDeterminantRepAnalysisDriver::<
                                UnitaryRepresentedSymmetryGroup,
                                C128,
                                _,
                                SpinConstraint,
                            >::builder()
                            .parameters(&mda_params)
                            .angular_function_parameters(&afa_params)
                            .multidets(multidets.iter().collect::<Vec<_>>())
                            .sao(&sao_c)
                            .sao_h(sao_h_c.as_ref())
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
                }
                PyStructureConstraint::SpinOrbitCoupled(_) => {
                    let dets_c = pydets
                        .iter()
                        .map(|pydet| match pydet {
                            PySlaterDeterminant::Real(pydet_r) => pydet_r
                                .to_qsym2::<SpinOrbitCoupled>(&baos_ref, mol)
                                .map(|det_r| {
                                    SlaterDeterminant::<C128, SpinOrbitCoupled>::from(det_r)
                                }),
                            PySlaterDeterminant::Complex(pydet_c) => {
                                pydet_c.to_qsym2::<SpinOrbitCoupled>(&baos_ref, mol)
                            }
                        })
                        .collect::<Result<Vec<_>, _>>()
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
                    match &use_magnetic_group {
                        Some(MagneticSymmetryAnalysisKind::Corepresentation) => {
                            let mut mda_driver = MultiDeterminantRepAnalysisDriver::<
                                MagneticRepresentedSymmetryGroup,
                                C128,
                                _,
                                SpinOrbitCoupled,
                            >::builder()
                            .parameters(&mda_params)
                            .angular_function_parameters(&afa_params)
                            .multidets(multidets.iter().collect::<Vec<_>>())
                            .sao(&sao_c)
                            .sao_h(sao_h_c.as_ref())
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
                        Some(MagneticSymmetryAnalysisKind::Representation) | None => {
                            let mut mda_driver = MultiDeterminantRepAnalysisDriver::<
                                UnitaryRepresentedSymmetryGroup,
                                C128,
                                _,
                                SpinOrbitCoupled,
                            >::builder()
                            .parameters(&mda_params)
                            .angular_function_parameters(&afa_params)
                            .multidets(multidets.iter().collect::<Vec<_>>())
                            .sao(&sao_c)
                            .sao_h(sao_h_c.as_ref())
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
                }
            }
        }
    }
    Ok(())
}
