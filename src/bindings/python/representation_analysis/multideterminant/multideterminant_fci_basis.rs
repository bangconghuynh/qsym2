//! Python bindings for QSym² symmetry analysis of multi-determinants with FCI bases.

use std::path::PathBuf;

use itertools::Itertools;
use num_complex::Complex;
use numpy::{PyArray1, PyArrayMethods};
use pyo3::exceptions::{PyIOError, PyRuntimeError};
use pyo3::prelude::*;

use crate::analysis::EigenvalueComparisonMode;
use crate::angmom::spinor_rotation_3d::{SpinConstraint, SpinOrbitCoupled};
use crate::bindings::python::integrals::{PyBasisAngularOrder, PyStructureConstraint};
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
use crate::target::noci::basis::FCIBasis;
use crate::target::noci::multideterminant::MultiDeterminant;
use crate::target::noci::multideterminants::MultiDeterminants;

type C128 = Complex<f64>;

/// Python-exposed function to perform representation symmetry analysis for real and complex
/// multi-determinantal wavefunctions constructed from a FCI basis based on a reference Slater
/// determinant and log the result via the `qsym2-output` logger at the `INFO` level.
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
/// * `pydet` - The Python-exposed reference Slater determinants whose coefficients are of type
/// `float64` or `complex128`. This determinant serves as the reference for the
/// full-configuration-interaction basis and should contain the full set of occupied and virtual
/// molecular orbitals.
/// * `coefficients` - The coefficient matrix where each column gives the linear combination
/// coefficients for one multi-determinantal wavefunction. The number of rows must match the number
/// of elements in the full-configuration-interaction basis, which is also the number of elements in
/// `occupation_patterns`. The elements are of type `float64` or `complex128`.
/// * `energies` - The `float64` or `complex128` energies of the multi-determinantal wavefunctions.
/// The number of terms must match the number of columns of `coefficients`.
/// * `occupation_patterns` - The occupation patterns of the determinants in the
/// full-configuration-interaction basis. Each element specifies one determinant in the basis.
/// * `pybaos` - Python-exposed structures containing basis angular order information, one for each
/// explicit component per coefficient matrix.
/// * `integrality_threshold` - The threshold for verifying if subspace multiplicities are
/// integral.
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
    pydet,
    coefficients,
    energies,
    occupation_patterns,
    pybaos,
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
pub fn rep_analyse_multideterminants_fci_basis(
    py: Python<'_>,
    inp_sym: PathBuf,
    pydet: PySlaterDeterminant,
    coefficients: PyArray2RC,
    energies: PyArray1RC,
    occupation_patterns: Vec<Vec<Bound<'_, PyArray1<f64>>>>,
    pybaos: Vec<PyBasisAngularOrder>,
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

    let real = matches!(pydet, PySlaterDeterminant::Real(_));

    let structure_constraint = match pydet {
        PySlaterDeterminant::Real(ref pd) => pd.structure_constraint().clone(),
        PySlaterDeterminant::Complex(ref pd) => pd.structure_constraint().clone(),
    };

    let occupation_patterns = occupation_patterns
        .into_iter()
        .map(|occs| {
            occs.into_iter()
                .map(|occ| occ.to_owned_array())
                .collect_vec()
        })
        .collect_vec();

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
    match (real, &coefficients, &energies, &sao) {
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
            let sao_r = pysao_r.to_owned_array();
            let coefficients_r = pycoefficients_r.to_owned_array();
            let energies_r = pyenergies_r.to_owned_array();
            let det_r = if augment_to_generalised {
                if let PySlaterDeterminant::Real(ref pydet_r) = pydet {
                    pydet_r
                        .to_qsym2(&baos_ref, mol)
                        .map(|det_r_inner| det_r_inner.to_generalised())
                        .map_err(|err| PyRuntimeError::new_err(err.to_string()))
                } else {
                    Err(PyRuntimeError::new_err(
                        "Unexpected complex type for a Slater determinant.".to_string(),
                    ))
                }
            } else {
                if let PySlaterDeterminant::Real(ref pydet_r) = pydet {
                    pydet_r
                        .to_qsym2(&baos_ref, mol)
                        .map_err(|err| PyRuntimeError::new_err(err.to_string()))
                } else {
                    Err(PyRuntimeError::new_err(
                        "Unexpected complex type for a Slater determinant.".to_string(),
                    ))
                }
            }?;

            let n_energies = energies_r.len();
            if coefficients_r.shape()[1] != n_energies {
                return Err(PyRuntimeError::new_err(
                    "Mismatched number of FCI energies and number of FCI states.",
                ));
            }

            // Construct the FCI basis
            let fci_basis = FCIBasis::builder()
                .reference(det_r)
                .occupation_patterns(occupation_patterns)
                .build()
                .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;

            // Construct the multi-determinantal wavefunctions
            let multidets_vec = energies_r
                .iter()
                .zip(coefficients_r.columns())
                .map(|(energy, coeffs)| {
                    MultiDeterminant::builder()
                        .basis(fci_basis.clone())
                        .coefficients(coeffs.to_owned())
                        .threshold(1e-7)
                        .energy(Ok(*energy))
                        .build()
                })
                .collect::<Result<Vec<_>, _>>()
                .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
            let multidets = MultiDeterminants::from_multideterminant_vec(
                &multidets_vec.iter().collect::<Vec<_>>()
            )
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
                    .multidets(&multidets)
                    .sao(&sao_r)
                    .sao_h(None) // Real SAO.
                    .symmetry_group(&pd_res)
                    .build()
                    .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;

                    // Run the driver
                    py.detach(|| {
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
                    .multidets(&multidets)
                    .sao(&sao_r)
                    .sao_h(None) // Real SAO.
                    .symmetry_group(&pd_res)
                    .build()
                    .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;

                    // Run the driver
                    py.detach(|| {
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
            let sao_h_c = sao_h.map(|pysao_h| match pysao_h {
                // sao_spatial_h must have the same reality as sao_spatial.
                PyArray2RC::Real(pysao_h_r) => pysao_h_r.to_owned_array().mapv(Complex::from),
                PyArray2RC::Complex(pysao_h_c) => pysao_h_c.to_owned_array(),
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
                    let det_c = if augment_to_generalised {
                        match pydet {
                            PySlaterDeterminant::Real(ref pydet_r) => pydet_r
                                .to_qsym2::<SpinConstraint>(&baos_ref, mol)
                                .map(|det_r| {
                                    SlaterDeterminant::<C128, SpinConstraint>::from(det_r)
                                        .to_generalised()
                                }),
                            PySlaterDeterminant::Complex(ref pydet_c) => pydet_c
                                .to_qsym2::<SpinConstraint>(&baos_ref, mol)
                                .map(|det_c| det_c.to_generalised()),
                        }
                    } else {
                        match pydet {
                            PySlaterDeterminant::Real(ref pydet_r) => pydet_r
                                .to_qsym2::<SpinConstraint>(&baos_ref, mol)
                                .map(|det_r| {
                                    SlaterDeterminant::<C128, SpinConstraint>::from(det_r)
                                }),
                            PySlaterDeterminant::Complex(ref pydet_c) => {
                                pydet_c.to_qsym2::<SpinConstraint>(&baos_ref, mol)
                            }
                        }
                    }
                    .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;

                    let n_energies = energies_c.len();
                    if coefficients_c.shape()[1] != n_energies {
                        return Err(PyRuntimeError::new_err(
                            "Mismatched number of FCI energies and number of FCI states.",
                        ));
                    }

                    // Construct the FCI basis
                    let fci_basis = FCIBasis::builder()
                        .reference(det_c)
                        .occupation_patterns(occupation_patterns)
                        .build()
                        .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;

                    // Construct the multi-determinantal wavefunctions
                    let multidets_vec = energies_c
                        .iter()
                        .zip(coefficients_c.columns())
                        .map(|(energy, coeffs)| {
                            MultiDeterminant::builder()
                                .basis(fci_basis.clone())
                                .coefficients(coeffs.to_owned())
                                .threshold(1e-7)
                                .energy(Ok(*energy))
                                .build()
                        })
                        .collect::<Result<Vec<_>, _>>()
                        .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
                    let multidets = MultiDeterminants::from_multideterminant_vec(
                        &multidets_vec.iter().collect::<Vec<_>>()
                    )
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
                            .multidets(&multidets)
                            .sao(&sao_c)
                            .sao_h(sao_h_c.as_ref())
                            .symmetry_group(&pd_res)
                            .build()
                            .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;

                            // Run the driver
                            py.detach(|| {
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
                            .multidets(&multidets)
                            .sao(&sao_c)
                            .sao_h(sao_h_c.as_ref())
                            .symmetry_group(&pd_res)
                            .build()
                            .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;

                            // Run the driver
                            py.detach(|| {
                                mda_driver
                                    .run()
                                    .map_err(|err| PyRuntimeError::new_err(err.to_string()))
                            })?
                        }
                    }
                }
                PyStructureConstraint::SpinOrbitCoupled(_) => {
                    let det_c = match pydet {
                        PySlaterDeterminant::Real(ref pydet_r) => pydet_r
                            .to_qsym2::<SpinOrbitCoupled>(&baos_ref, mol)
                            .map(SlaterDeterminant::<C128, SpinOrbitCoupled>::from),
                        PySlaterDeterminant::Complex(ref pydet_c) => {
                            pydet_c.to_qsym2::<SpinOrbitCoupled>(&baos_ref, mol)
                        }
                    }
                    .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;

                    let n_energies = energies_c.len();
                    if coefficients_c.shape()[1] != n_energies {
                        return Err(PyRuntimeError::new_err(
                            "Mismatched number of FCI energies and number of FCI states.",
                        ));
                    }

                    // Construct the FCI basis
                    let fci_basis = FCIBasis::builder()
                        .reference(det_c)
                        .occupation_patterns(occupation_patterns)
                        .build()
                        .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;

                    // Construct the multi-determinantal wavefunctions
                    let multidets_vec = energies_c
                        .iter()
                        .zip(coefficients_c.columns())
                        .map(|(energy, coeffs)| {
                            MultiDeterminant::builder()
                                .basis(fci_basis.clone())
                                .coefficients(coeffs.to_owned())
                                .threshold(1e-7)
                                .energy(Ok(*energy))
                                .build()
                        })
                        .collect::<Result<Vec<_>, _>>()
                        .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
                    let multidets = MultiDeterminants::from_multideterminant_vec(
                        &multidets_vec.iter().collect::<Vec<_>>()
                    )
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
                            .multidets(&multidets)
                            .sao(&sao_c)
                            .sao_h(sao_h_c.as_ref())
                            .symmetry_group(&pd_res)
                            .build()
                            .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;

                            // Run the driver
                            py.detach(|| {
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
                            .multidets(&multidets)
                            .sao(&sao_c)
                            .sao_h(sao_h_c.as_ref())
                            .symmetry_group(&pd_res)
                            .build()
                            .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;

                            // Run the driver
                            py.detach(|| {
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
