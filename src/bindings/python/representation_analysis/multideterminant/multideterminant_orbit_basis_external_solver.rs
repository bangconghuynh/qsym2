//! Python bindings for QSym² symmetry analysis of multi-determinants with orbit bases using
//! external NOCI solvers.

use std::collections::HashSet;
use std::path::PathBuf;

use anyhow::{Context, bail, format_err};
use itertools::Itertools;
use ndarray::{Array1, Array2, Axis, ShapeBuilder};
use num_complex::Complex;
use numpy::{PyArray1, PyArray2, PyArrayMethods, ToPyArray};
use pyo3::exceptions::{PyIOError, PyRuntimeError};
use pyo3::types::PyFunction;
use pyo3::{IntoPyObjectExt, prelude::*};

use crate::analysis::EigenvalueComparisonMode;
use crate::angmom::spinor_rotation_3d::{SpinConstraint, SpinOrbitCoupled};
use crate::bindings::python::integrals::{PyBasisAngularOrder, PyStructureConstraint};
use crate::bindings::python::representation_analysis::PyArray2RC;
use crate::bindings::python::representation_analysis::multideterminant::{
    PyMultiDeterminantsComplex, PyMultiDeterminantsReal,
};
use crate::bindings::python::representation_analysis::slater_determinant::{
    PySlaterDeterminant, PySlaterDeterminantComplex, PySlaterDeterminantReal,
};
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
    MagneticRepresentedSymmetryGroup, SymmetryGroupProperties, UnitaryRepresentedSymmetryGroup,
};
use crate::symmetry::symmetry_transformation::{SymmetryTransformable, SymmetryTransformationKind};
use crate::target::determinant::SlaterDeterminant;
use crate::target::noci::basis::{Basis, OrbitBasis};
use crate::target::noci::multideterminant::MultiDeterminant;
use crate::target::noci::multideterminants::MultiDeterminants;

type C128 = Complex<f64>;

// ~~~~~~~~~~~~~
// Macro helpers
// ~~~~~~~~~~~~~
macro_rules! generate_noci_solver {
    ($noci_solver_name:ident, $py_solver_func:ident, $pysd:ty, $t:ty, $sc:ty) => {
        let $noci_solver_name = |multidets: &Vec<SlaterDeterminant<$t, $sc>>| {
            Python::attach(|py_inner| {
                let pymultidets = multidets
                    .iter()
                    .map(|det| {
                        let pysc = det.structure_constraint().clone().try_into()?;
                        Ok(<$pysd>::new(
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
                $py_solver_func
                    .call1(py_inner, (pymultidets,))
                    .and_then(|res| res.extract::<(Vec<$t>, Vec<Vec<$t>>)>(py_inner))
            })
        };
    };
}

// ~~~~~~~~~
// Functions
// ~~~~~~~~~

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
/// representation analysis.
/// * `pyorigins` - A list of Python-exposed Slater determinants whose coefficients are of type
/// `float64` or `complex128`. These determinants serve as origins for group-generated orbits which
/// serve as basis states for non-orthogonal configuration interaction to yield multi-determinantal
/// wavefunctions, the symmetry of which will be analysed by this function.
/// * `py_noci_solver` - A Python function callable on a sequence of Slater determinants to perform
/// non-orthogonal configuration interaction (NOCI) and return a list of NOCI energies and a
/// corresponding list of lists of linear combination coefficients, where each inner list is for one
/// multi-determinantal wavefunction resulting from the NOCI calculation.
/// Python type: `Callable[[list[PySlaterDeterminantReal | PySlaterDeterminantComplex]], tuple[list[float], list[list[float]]] | tuple[list[complex], list[list[complex]]]]`.
/// * `pybaos` - Python-exposed structures containing basis angular order information, one for each
/// explicit component per coefficient matrix.
/// * `density_matrix_calculation_thresholds` - An optional pair of thresholds for Löwdin pairing,
/// one for checking zero off-diagonal values, one for checking zero overlaps, when computing
/// multi-determinantal density matrices. If `None`, no density matrices for the resulting
/// multi-determinants will be computed.
/// * `integrality_threshold` - The threshold for verifying if subspace multiplicities are integral.
/// * `linear_independence_threshold` - The threshold for determining the linear independence
/// subspace via the non-zero eigenvalues of the orbit overlap matrix.
/// * `use_magnetic_group` - An option indicating if the magnetic group is to be used for symmetry
/// analysis, and if so, whether unitary representations or unitary-antiunitary corepresentations
/// should be used.
/// * `use_double_group` - A boolean indicating if the double group of the prevailing symmetry
/// group is to be used for representation analysis instead.
/// * `use_cayley_table` - A boolean indicating if the Cayley table for the group, if available,
/// should be used to speed up the calculation of orbit overlap matrices.
/// * `symmetry_transformation_kind` - An enumerated type indicating the type of symmetry
/// transformations to be performed on the origin determinant to generate the orbit. If this
/// contains spin transformation, the multi-determinant will be augmented to generalised spin
/// constraint automatically.
/// * `eigenvalue_comparison_mode` - An enumerated type indicating the mode of comparison of orbit
/// overlap eigenvalues with the specified `linear_independence_threshold`.
/// * `sao` - The atomic-orbital overlap matrix whose elements are of type `float64` or
/// `complex128`.
/// * `sao_h` - The optional complex-symmetric atomic-orbital overlap matrix whose elements
/// are of type `float64` or `complex128`. This is required if antiunitary symmetry operations are
/// involved.
/// * `write_overlap_eigenvalues` - A boolean indicating if the eigenvalues of the determinant
/// orbit overlap matrix are to be written to the output.
/// * `write_character_table` - A boolean indicating if the character table of the prevailing
/// symmetry group is to be printed out.
/// * `infinite_order_to_finite` - The finite order with which infinite-order generators are to be
/// interpreted to form a finite subgroup of the prevailing infinite group. This finite subgroup
/// will be used for symmetry analysis.
/// * `angular_function_integrality_threshold` - The threshold for verifying if subspace
/// multiplicities are integral for the symmetry analysis of angular functions.
/// * `angular_function_linear_independence_threshold` - The threshold for determining the linear
/// independence subspace via the non-zero eigenvalues of the orbit overlap matrix for the symmetry
/// analysis of angular functions.
/// * `angular_function_max_angular_momentum` - The maximum angular momentum order to be used in
/// angular function symmetry analysis.
#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (
    inp_sym,
    pyorigins,
    py_noci_solver,
    pybaos,
    density_matrix_calculation_thresholds,
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
pub fn rep_analyse_multideterminants_orbit_basis_external_solver(
    py: Python<'_>,
    inp_sym: PathBuf,
    pyorigins: Vec<PySlaterDeterminant>,
    py_noci_solver: Py<PyFunction>,
    pybaos: Vec<PyBasisAngularOrder>,
    density_matrix_calculation_thresholds: Option<(f64, f64)>,
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
) -> PyResult<Py<PyAny>> {
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

    let baos = pybaos
        .iter()
        .map(|bao| {
            bao.to_qsym2(mol)
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

    // Set up NOCI function
    generate_noci_solver!(
        noci_solver_r,
        py_noci_solver,
        PySlaterDeterminantReal,
        f64,
        SpinConstraint
    );
    generate_noci_solver!(
        noci_solver_c_sc,
        py_noci_solver,
        PySlaterDeterminantComplex,
        C128,
        SpinConstraint
    );
    generate_noci_solver!(
        noci_solver_c_soc,
        py_noci_solver,
        PySlaterDeterminantComplex,
        C128,
        SpinOrbitCoupled
    );

    let all_real = pyorigins
        .iter()
        .all(|pyorigin| matches!(pyorigin, PySlaterDeterminant::Real(_)));

    let structure_constraints_set = pyorigins
        .iter()
        .map(|pyorigin| match pyorigin {
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
    // - all_real and real SAO?
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
    match (all_real, &sao) {
        (true, PyArray2RC::Real(pysao_r)) => {
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
            let sao_r = pysao_r.to_owned_array();
            let origins_r = if augment_to_generalised {
                pyorigins
                    .iter()
                    .map(|pydet| {
                        if let PySlaterDeterminant::Real(pydet_r) = pydet {
                            pydet_r
                                .to_qsym2(&baos_ref, mol)
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
                            pydet_r.to_qsym2(&baos_ref, mol)
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
                        .detach(|| {
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

                    // Run NOCI using the real-valued external solver
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
                        SpinConstraint,
                    >::builder()
                    .parameters(&mda_params)
                    .angular_function_parameters(&afa_params)
                    .multidets(multidets.iter().collect::<Vec<_>>())
                    .sao(&sao_r)
                    .sao_h(None) // Real SAO.
                    .symmetry_group(&pd_res)
                    .build()
                    .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
                    py.detach(|| {
                        mda_driver
                            .run()
                            .map_err(|err| PyRuntimeError::new_err(err.to_string()))
                    })?;

                    // Collect real multi-determinantal wavefunctions for returning
                    let basis = multidets
                        .first()
                        .and_then(|multidet| {
                            multidet
                                .basis()
                                .iter()
                                .map(|det_res| det_res.and_then(|det| det.to_python(py)))
                                .collect::<Result<Vec<_>, _>>()
                                .ok()
                        })
                        .ok_or_else(|| {
                            PyRuntimeError::new_err(
                                "Unable to obtain the basis of Slater determinants.".to_string(),
                            )
                        })?;
                    let (coefficientss, energies): (Vec<_>, Vec<_>) = multidets
                        .iter()
                        .map(|multidet| {
                            let coefficients =
                                multidet.coefficients().iter().cloned().collect_vec();
                            let energy = *multidet.energy().unwrap_or(&f64::NAN);
                            Ok::<_, PyErr>((coefficients, energy))
                        })
                        .collect::<Result<Vec<_>, _>>()?
                        .into_iter()
                        .unzip();
                    let coefficientss_arr = Array2::from_shape_vec(
                        (basis.len(), coefficientss.len()).f(),
                        coefficientss.into_iter().flatten().collect_vec(),
                    )
                    .map_err(|err| PyRuntimeError::new_err(err.to_string()))?
                    .to_pyarray(py);
                    let energies_arr = Array1::from_vec(energies).to_pyarray(py);
                    let density_matrices = density_matrix_calculation_thresholds.and_then(
                        |(thresh_offdiag, thresh_zeroov)| {
                            log::debug!("Calculating density matrices...");
                            let multidets_collection =
                                MultiDeterminants::from_multideterminant_vec(
                                    &multidets.iter().collect_vec(),
                                )
                                .ok()?;
                            let denmats_opt = multidets_collection
                                .density_matrices(
                                    &sao_r.view(),
                                    thresh_offdiag,
                                    thresh_zeroov,
                                    true,
                                )
                                .map(|denmats| {
                                    denmats
                                        .axis_iter(Axis(0))
                                        .map(|denmat| denmat.to_pyarray(py))
                                        .collect_vec()
                                })
                                .ok();
                            log::debug!("Calculating density matrices... Done.");
                            denmats_opt
                        },
                    );
                    let pymultidet = PyMultiDeterminantsReal::new(
                        basis,
                        coefficientss_arr,
                        energies_arr,
                        density_matrices,
                        multidets[0].threshold(),
                    )
                    .into_py_any(py)?;
                    Ok(pymultidet)
                }
                Some(MagneticSymmetryAnalysisKind::Representation) | None => {
                    // Unitary groups or magnetic groups with representations
                    let group = py
                        .detach(|| {
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

                    // Run NOCI using the real-valued external solver
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
                        SpinConstraint,
                    >::builder()
                    .parameters(&mda_params)
                    .angular_function_parameters(&afa_params)
                    .multidets(multidets.iter().collect::<Vec<_>>())
                    .sao(&sao_r)
                    .sao_h(None) // Real SAO.
                    .symmetry_group(&pd_res)
                    .build()
                    .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
                    py.detach(|| {
                        mda_driver
                            .run()
                            .map_err(|err| PyRuntimeError::new_err(err.to_string()))
                    })?;

                    // Collect real multi-determinantal wavefunctions for returning
                    let basis = multidets
                        .first()
                        .and_then(|multidet| {
                            multidet
                                .basis()
                                .iter()
                                .map(|det_res| det_res.and_then(|det| det.to_python(py)))
                                .collect::<Result<Vec<_>, _>>()
                                .ok()
                        })
                        .ok_or_else(|| {
                            PyRuntimeError::new_err(
                                "Unable to obtain the basis of Slater determinants.".to_string(),
                            )
                        })?;
                    let (coefficientss, energies): (Vec<_>, Vec<_>) = multidets
                        .iter()
                        .map(|multidet| {
                            let coefficients =
                                multidet.coefficients().iter().cloned().collect_vec();
                            let energy = *multidet.energy().unwrap_or(&f64::NAN);
                            Ok::<_, PyErr>((coefficients, energy))
                        })
                        .collect::<Result<Vec<_>, _>>()?
                        .into_iter()
                        .unzip();
                    let coefficientss_arr = Array2::from_shape_vec(
                        (basis.len(), coefficientss.len()).f(),
                        coefficientss.into_iter().flatten().collect_vec(),
                    )
                    .map_err(|err| PyRuntimeError::new_err(err.to_string()))?
                    .to_pyarray(py);
                    let energies_arr = Array1::from_vec(energies).to_pyarray(py);
                    let density_matrices = density_matrix_calculation_thresholds.and_then(
                        |(thresh_offdiag, thresh_zeroov)| {
                            log::debug!("Calculating density matrices...");
                            let multidets_collection =
                                MultiDeterminants::from_multideterminant_vec(
                                    &multidets.iter().collect_vec(),
                                )
                                .ok()?;
                            let denmats_opt = multidets_collection
                                .density_matrices(
                                    &sao_r.view(),
                                    thresh_offdiag,
                                    thresh_zeroov,
                                    true,
                                )
                                .map(|denmats| {
                                    denmats
                                        .axis_iter(Axis(0))
                                        .map(|denmat| denmat.to_pyarray(py))
                                        .collect_vec()
                                })
                                .ok();
                            log::debug!("Calculating density matrices... Done.");
                            denmats_opt
                        },
                    );
                    let pymultidet = PyMultiDeterminantsReal::new(
                        basis,
                        coefficientss_arr,
                        energies_arr,
                        density_matrices,
                        multidets[0].threshold(),
                    )
                    .into_py_any(py)?;
                    Ok(pymultidet)
                }
            }
        }
        (_, _) => {
            // Complex numeric data type

            // Preparation
            let sao_c = match sao {
                PyArray2RC::Real(pysao_r) => pysao_r.to_owned_array().mapv(Complex::from),
                PyArray2RC::Complex(pysao_c) => pysao_c.to_owned_array(),
            };
            let sao_h_c = sao_h.map(|pysao_h| match pysao_h {
                // sao_h must have the same reality as sao.
                PyArray2RC::Real(pysao_h_r) => pysao_h_r.to_owned_array().mapv(Complex::from),
                PyArray2RC::Complex(pysao_h_c) => pysao_h_c.to_owned_array(),
            });

            match structure_constraint {
                PyStructureConstraint::SpinConstraint(_) => {
                    let origins_c = if augment_to_generalised {
                        pyorigins
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
                        pyorigins
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

                    match &use_magnetic_group {
                        Some(MagneticSymmetryAnalysisKind::Corepresentation) => {
                            // Magnetic groups with corepresentations
                            let group = py
                                .detach(|| {
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

                            let (noci_energies_vec, noci_coeffs_vec) = noci_solver_c_sc(&dets)
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
                            py.detach(|| {
                                mda_driver
                                    .run()
                                    .map_err(|err| PyRuntimeError::new_err(err.to_string()))
                            })?;

                            // Collect complex multi-determinantal wavefunctions for returning
                            let basis = multidets
                                .first()
                                .and_then(|multidet| {
                                    multidet
                                        .basis()
                                        .iter()
                                        .map(|det_res| det_res.and_then(|det| det.to_python(py)))
                                        .collect::<Result<Vec<_>, _>>()
                                        .ok()
                                })
                                .ok_or_else(|| {
                                    PyRuntimeError::new_err(
                                        "Unable to obtain the basis of Slater determinants."
                                            .to_string(),
                                    )
                                })?;
                            let (coefficientss, energies): (Vec<_>, Vec<_>) = multidets
                                .iter()
                                .map(|multidet| {
                                    let coefficients =
                                        multidet.coefficients().iter().cloned().collect_vec();
                                    let energy =
                                        *multidet.energy().unwrap_or(&Complex::from(f64::NAN));
                                    Ok::<_, PyErr>((coefficients, energy))
                                })
                                .collect::<Result<Vec<_>, _>>()?
                                .into_iter()
                                .unzip();
                            let coefficientss_arr = Array2::from_shape_vec(
                                (basis.len(), coefficientss.len()).f(),
                                coefficientss.into_iter().flatten().collect_vec(),
                            )
                            .map_err(|err| PyRuntimeError::new_err(err.to_string()))?
                            .to_pyarray(py);
                            let energies_arr = Array1::from_vec(energies).to_pyarray(py);
                            let density_matrices = density_matrix_calculation_thresholds.and_then(
                                |(thresh_offdiag, thresh_zeroov)| {
                                    log::debug!("Calculating density matrices...");
                                    let multidets_collection =
                                        MultiDeterminants::from_multideterminant_vec(
                                            &multidets.iter().collect_vec(),
                                        )
                                        .ok()?;
                                    let denmats_opt = multidets_collection
                                        .density_matrices(
                                            &sao_c.view(),
                                            thresh_offdiag,
                                            thresh_zeroov,
                                            true,
                                        )
                                        .map(|denmats| {
                                            denmats
                                                .axis_iter(Axis(0))
                                                .map(|denmat| denmat.to_pyarray(py))
                                                .collect_vec()
                                        })
                                        .ok();
                                    log::debug!("Calculating density matrices... Done.");
                                    denmats_opt
                                },
                            );
                            let pymultidet = PyMultiDeterminantsComplex::new(
                                basis,
                                coefficientss_arr,
                                energies_arr,
                                density_matrices,
                                multidets[0].threshold(),
                            )
                            .into_py_any(py)?;
                            Ok(pymultidet)
                        }
                        Some(MagneticSymmetryAnalysisKind::Representation) | None => {
                            // Unitary groups or magnetic groups with representations
                            let group = py
                                .detach(|| {
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

                            let (noci_energies_vec, noci_coeffs_vec) = noci_solver_c_sc(&dets)
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
                            py.detach(|| {
                                mda_driver
                                    .run()
                                    .map_err(|err| PyRuntimeError::new_err(err.to_string()))
                            })?;

                            // Collect complex multi-determinantal wavefunctions for returning
                            let basis = multidets
                                .first()
                                .and_then(|multidet| {
                                    multidet
                                        .basis()
                                        .iter()
                                        .map(|det_res| det_res.and_then(|det| det.to_python(py)))
                                        .collect::<Result<Vec<_>, _>>()
                                        .ok()
                                })
                                .ok_or_else(|| {
                                    PyRuntimeError::new_err(
                                        "Unable to obtain the basis of Slater determinants."
                                            .to_string(),
                                    )
                                })?;
                            let (coefficientss, energies): (Vec<_>, Vec<_>) = multidets
                                .iter()
                                .map(|multidet| {
                                    let coefficients =
                                        multidet.coefficients().iter().cloned().collect_vec();
                                    let energy =
                                        *multidet.energy().unwrap_or(&Complex::from(f64::NAN));
                                    Ok::<_, PyErr>((coefficients, energy))
                                })
                                .collect::<Result<Vec<_>, _>>()?
                                .into_iter()
                                .unzip();
                            let coefficientss_arr = Array2::from_shape_vec(
                                (basis.len(), coefficientss.len()).f(),
                                coefficientss.into_iter().flatten().collect_vec(),
                            )
                            .map_err(|err| PyRuntimeError::new_err(err.to_string()))?
                            .to_pyarray(py);
                            let energies_arr = Array1::from_vec(energies).to_pyarray(py);
                            let density_matrices = density_matrix_calculation_thresholds.and_then(
                                |(thresh_offdiag, thresh_zeroov)| {
                                    log::debug!("Calculating density matrices...");
                                    let multidets_collection =
                                        MultiDeterminants::from_multideterminant_vec(
                                            &multidets.iter().collect_vec(),
                                        )
                                        .ok()?;
                                    let denmats_opt = multidets_collection
                                        .density_matrices(
                                            &sao_c.view(),
                                            thresh_offdiag,
                                            thresh_zeroov,
                                            true,
                                        )
                                        .map(|denmats| {
                                            denmats
                                                .axis_iter(Axis(0))
                                                .map(|denmat| denmat.to_pyarray(py))
                                                .collect_vec()
                                        })
                                        .ok();
                                    log::debug!("Calculating density matrices... Done.");
                                    denmats_opt
                                },
                            );
                            let pymultidet = PyMultiDeterminantsComplex::new(
                                basis,
                                coefficientss_arr,
                                energies_arr,
                                density_matrices,
                                multidets[0].threshold(),
                            )
                            .into_py_any(py)?;
                            Ok(pymultidet)
                        }
                    }
                }
                PyStructureConstraint::SpinOrbitCoupled(_) => {
                    let origins_c = pyorigins
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

                    match &use_magnetic_group {
                        Some(MagneticSymmetryAnalysisKind::Corepresentation) => {
                            // Magnetic groups with corepresentations
                            let group = py
                                .detach(|| {
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

                            let (noci_energies_vec, noci_coeffs_vec) = noci_solver_c_soc(&dets)
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
                            py.detach(|| {
                                mda_driver
                                    .run()
                                    .map_err(|err| PyRuntimeError::new_err(err.to_string()))
                            })?;

                            // Collect complex multi-determinantal wavefunctions for returning
                            let basis = multidets
                                .first()
                                .and_then(|multidet| {
                                    multidet
                                        .basis()
                                        .iter()
                                        .map(|det_res| det_res.and_then(|det| det.to_python(py)))
                                        .collect::<Result<Vec<_>, _>>()
                                        .ok()
                                })
                                .ok_or_else(|| {
                                    PyRuntimeError::new_err(
                                        "Unable to obtain the basis of Slater determinants."
                                            .to_string(),
                                    )
                                })?;
                            let (coefficientss, energies): (Vec<_>, Vec<_>) = multidets
                                .iter()
                                .map(|multidet| {
                                    let coefficients =
                                        multidet.coefficients().iter().cloned().collect_vec();
                                    let energy =
                                        *multidet.energy().unwrap_or(&Complex::from(f64::NAN));
                                    Ok::<_, PyErr>((coefficients, energy))
                                })
                                .collect::<Result<Vec<_>, _>>()?
                                .into_iter()
                                .unzip();
                            let coefficientss_arr = Array2::from_shape_vec(
                                (basis.len(), coefficientss.len()).f(),
                                coefficientss.into_iter().flatten().collect_vec(),
                            )
                            .map_err(|err| PyRuntimeError::new_err(err.to_string()))?
                            .to_pyarray(py);
                            let energies_arr = Array1::from_vec(energies).to_pyarray(py);
                            let density_matrices = density_matrix_calculation_thresholds.and_then(
                                |(thresh_offdiag, thresh_zeroov)| {
                                    log::debug!("Calculating density matrices...");
                                    let multidets_collection =
                                        MultiDeterminants::from_multideterminant_vec(
                                            &multidets.iter().collect_vec(),
                                        )
                                        .ok()?;
                                    let denmats_opt = multidets_collection
                                        .density_matrices(
                                            &sao_c.view(),
                                            thresh_offdiag,
                                            thresh_zeroov,
                                            true,
                                        )
                                        .map(|denmats| {
                                            denmats
                                                .axis_iter(Axis(0))
                                                .map(|denmat| denmat.to_pyarray(py))
                                                .collect_vec()
                                        })
                                        .ok();
                                    log::debug!("Calculating density matrices... Done.");
                                    denmats_opt
                                },
                            );
                            let pymultidet = PyMultiDeterminantsComplex::new(
                                basis,
                                coefficientss_arr,
                                energies_arr,
                                density_matrices,
                                multidets[0].threshold(),
                            )
                            .into_py_any(py)?;
                            Ok(pymultidet)
                        }
                        Some(MagneticSymmetryAnalysisKind::Representation) | None => {
                            // Unitary groups or magnetic groups with representations
                            let group = py
                                .detach(|| {
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

                            let (noci_energies_vec, noci_coeffs_vec) = noci_solver_c_soc(&dets)
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
                            py.detach(|| {
                                mda_driver
                                    .run()
                                    .map_err(|err| PyRuntimeError::new_err(err.to_string()))
                            })?;

                            // Collect complex multi-determinantal wavefunctions for returning
                            let basis = multidets
                                .first()
                                .and_then(|multidet| {
                                    multidet
                                        .basis()
                                        .iter()
                                        .map(|det_res| det_res.and_then(|det| det.to_python(py)))
                                        .collect::<Result<Vec<_>, _>>()
                                        .ok()
                                })
                                .ok_or_else(|| {
                                    PyRuntimeError::new_err(
                                        "Unable to obtain the basis of Slater determinants."
                                            .to_string(),
                                    )
                                })?;
                            let (coefficientss, energies): (Vec<_>, Vec<_>) = multidets
                                .iter()
                                .map(|multidet| {
                                    let coefficients =
                                        multidet.coefficients().iter().cloned().collect_vec();
                                    let energy =
                                        *multidet.energy().unwrap_or(&Complex::from(f64::NAN));
                                    Ok::<_, PyErr>((coefficients, energy))
                                })
                                .collect::<Result<Vec<_>, _>>()?
                                .into_iter()
                                .unzip();
                            let coefficientss_arr = Array2::from_shape_vec(
                                (basis.len(), coefficientss.len()).f(),
                                coefficientss.into_iter().flatten().collect_vec(),
                            )
                            .map_err(|err| PyRuntimeError::new_err(err.to_string()))?
                            .to_pyarray(py);
                            let energies_arr = Array1::from_vec(energies).to_pyarray(py);
                            let density_matrices = density_matrix_calculation_thresholds.and_then(
                                |(thresh_offdiag, thresh_zeroov)| {
                                    log::debug!("Calculating density matrices...");
                                    let multidets_collection =
                                        MultiDeterminants::from_multideterminant_vec(
                                            &multidets.iter().collect_vec(),
                                        )
                                        .ok()?;
                                    let denmats_opt = multidets_collection
                                        .density_matrices(
                                            &sao_c.view(),
                                            thresh_offdiag,
                                            thresh_zeroov,
                                            true,
                                        )
                                        .map(|denmats| {
                                            denmats
                                                .axis_iter(Axis(0))
                                                .map(|denmat| denmat.to_pyarray(py))
                                                .collect_vec()
                                        })
                                        .ok();
                                    log::debug!("Calculating density matrices... Done.");
                                    denmats_opt
                                },
                            );
                            let pymultidet = PyMultiDeterminantsComplex::new(
                                basis,
                                coefficientss_arr,
                                energies_arr,
                                density_matrices,
                                multidets[0].threshold(),
                            )
                            .into_py_any(py)?;
                            Ok(pymultidet)
                        }
                    }
                }
            }
        }
    }
}
