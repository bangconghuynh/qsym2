//! Python bindings for QSym² symmetry analysis of multi-determinants with orbit bases using
//! the internal NOCI solver.

use std::collections::HashSet;
use std::path::PathBuf;

use anyhow::{bail, format_err};
use itertools::Itertools;
use log;
use num_complex::Complex;
use numpy::PyArrayMethods;
use pyo3::exceptions::{PyIOError, PyRuntimeError, PyTypeError};
use pyo3::prelude::*;

use crate::analysis::EigenvalueComparisonMode;
use crate::angmom::spinor_rotation_3d::{SpinConstraint, SpinOrbitCoupled};
use crate::bindings::python::integrals::{
    PyBasisAngularOrder, PyStructureConstraint,
};
use crate::bindings::python::representation_analysis::slater_determinant::PySlaterDeterminant;
use crate::bindings::python::representation_analysis::{PyArray2RC, PyArray4RC, PyScalarRC};
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
use crate::symmetry::symmetry_transformation::SymmetryTransformationKind;
use crate::target::noci::backend::matelem::hamiltonian::HamiltonianAO;
use crate::target::noci::backend::matelem::overlap::OverlapAO;
use crate::target::noci::backend::solver::noci::NOCISolvable;

type C128 = Complex<f64>;

/// Python-exposed function to run non-orthogonal configuration interaction using the internal
/// solver and then perform representation symmetry analysis on the resulting real and complex
/// multi-determinantal wavefunctions constructed from group-generated orbits and log the result via
/// the `qsym2-output` logger at the `INFO` level.
///
/// If `symmetry_transformation_kind` includes spin transformation, the provided
/// multi-determinantal wavefunctions with spin constraint structure will be augmented to
/// the generalised spin constraint automatically.
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
/// * `pybaos` - Python-exposed structures containing basis angular order information, one for each
/// explicit component per coefficient matrix. Python type: `list[PyBasisAngularOrder]`.
/// * `sao` - The atomic-orbital overlap matrix whose elements are of type `float64` or
/// `complex128`. Python type: `numpy.2darray[float] | numpy.2darray[complex]`.
/// * `enuc` - The nuclear repulsion energy. Python type: `float | complex`.
/// * `onee` - The one-electron integral matrix whose elements are of type `float64` or
/// `complex128`. Python type: `numpy.2darray[float] | numpy.2darray[complex]`.
/// * `twoe` - The two-electron integral tensor whose elements are of type `float64` or
/// `complex128`. Python type: `numpy.4darray[float] | numpy.4darray[complex]`.
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
    pyorigins,
    pybaos,
    sao,
    enuc,
    onee,
    twoe,
    thresh_offdiag,
    thresh_zeroov,
    integrality_threshold,
    linear_independence_threshold,
    use_magnetic_group,
    use_double_group,
    use_cayley_table,
    symmetry_transformation_kind,
    eigenvalue_comparison_mode,
    sao_h=None,
    write_overlap_eigenvalues=true,
    write_character_table=true,
    infinite_order_to_finite=None,
    angular_function_integrality_threshold=1e-7,
    angular_function_linear_independence_threshold=1e-7,
    angular_function_max_angular_momentum=2
))]
pub fn rep_analyse_multideterminants_orbit_basis_internal_solver(
    py: Python<'_>,
    inp_sym: PathBuf,
    pyorigins: Vec<PySlaterDeterminant>,
    pybaos: Vec<PyBasisAngularOrder>,
    sao: PyArray2RC,
    enuc: PyScalarRC,
    onee: PyArray2RC,
    twoe: PyArray4RC,
    thresh_offdiag: f64,
    thresh_zeroov: f64,
    integrality_threshold: f64,
    linear_independence_threshold: f64,
    use_magnetic_group: Option<MagneticSymmetryAnalysisKind>,
    use_double_group: bool,
    use_cayley_table: bool,
    symmetry_transformation_kind: SymmetryTransformationKind,
    eigenvalue_comparison_mode: EigenvalueComparisonMode,
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

    let all_real_origins = pyorigins
        .iter()
        .all(|pyorigin| matches!(pyorigin, PySlaterDeterminant::Real(_)));
    let all_real_integrals = matches!(sao, PyArray2RC::Real(_))
        && matches!(enuc, PyScalarRC::Real(_))
        && matches!(onee, PyArray2RC::Real(_))
        && matches!(twoe, PyArray4RC::Real(_));
    let all_real = all_real_origins && all_real_integrals;

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
    // - all_real?
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
    if all_real {
        // Real numeric data type
        // Only SpinConstraint is supported for real numeric data type.

        if matches!(
            structure_constraint,
            PyStructureConstraint::SpinOrbitCoupled(_)
        ) {
            return Err(PyRuntimeError::new_err(
                "Real determinants cannot support spin--orbit-coupled structure constraint.",
            ));
        }

        // Preparation
        let sao_r = match sao {
            PyArray2RC::Real(pysao_r) => Ok(pysao_r.to_owned_array()),
            PyArray2RC::Complex(_) => Err(PyTypeError::new_err(
                "Unexpected complex type for the SAO matrix.",
            )),
        }?;
        let enuc_r = match enuc {
            PyScalarRC::Real(enuc_r) => Ok(enuc_r),
            PyScalarRC::Complex(_) => Err(PyTypeError::new_err(
                "Unexpected complex type for the nuclear repulsion energy.",
            )),
        }?;
        let onee_r = match onee {
            PyArray2RC::Real(pyonee_r) => Ok(pyonee_r.to_owned_array()),
            PyArray2RC::Complex(_) => Err(PyTypeError::new_err(
                "Unexpected complex type for the one-electron integral matrix.",
            )),
        }?;
        let twoe_r = match twoe {
            PyArray4RC::Real(pytwoe_r) => Ok(pytwoe_r.to_owned_array()),
            PyArray4RC::Complex(_) => Err(PyTypeError::new_err(
                "Unexpected complex type for the two-electron integral tensor.",
            )),
        }?;
        let overlap_ao = OverlapAO::<f64, SpinConstraint>::builder()
            .sao(sao_r.view())
            .build()
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
        let hamiltonian_ao = HamiltonianAO::<f64, SpinConstraint>::builder()
            .enuc(enuc_r)
            .onee(onee_r.view())
            .twoe(twoe_r.view())
            .build()
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;

        let origins_r = if augment_to_generalised {
            pyorigins
                .iter()
                .map(|pydet| {
                    if let PySlaterDeterminant::Real(pydet_r) = pydet {
                        pydet_r
                            .to_qsym2::<SpinConstraint>(&baos_ref, mol)
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
                        pydet_r.to_qsym2::<SpinConstraint>(&baos_ref, mol)
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

                // Run NOCI
                let system = (&hamiltonian_ao, &overlap_ao);
                let multidets = system
                    .solve_symmetry_noci(
                        &origins_r.iter().collect_vec(),
                        &group,
                        symmetry_transformation_kind,
                        use_cayley_table,
                        thresh_offdiag,
                        thresh_zeroov,
                    )
                    .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;

                // Symmetry analysis for NOCI states
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

                // Run NOCI
                let system = (&hamiltonian_ao, &overlap_ao);
                let multidets = system
                    .solve_symmetry_noci(
                        &origins_r.iter().collect_vec(),
                        &group,
                        symmetry_transformation_kind,
                        use_cayley_table,
                        thresh_offdiag,
                        thresh_zeroov,
                    )
                    .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;

                // Symmetry analysis for NOCI states
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
                py.allow_threads(|| {
                    mda_driver
                        .run()
                        .map_err(|err| PyRuntimeError::new_err(err.to_string()))
                })?
            }
        };
    } else {
        // Some complex numeric data type

        // Preparation
        let sao_c = match sao {
            PyArray2RC::Real(pysao_r) => pysao_r.to_owned_array().mapv(Complex::from),
            PyArray2RC::Complex(pysao_c) => pysao_c.to_owned_array(),
        };
        let sao_h_c = match sao_h {
            Some(PyArray2RC::Real(pysao_r)) => Some(pysao_r.to_owned_array().mapv(Complex::from)),
            Some(PyArray2RC::Complex(pysao_c)) => Some(pysao_c.to_owned_array()),
            None => None,
        };
        let enuc_c = match enuc {
            PyScalarRC::Real(enuc_r) => Complex::from(enuc_r),
            PyScalarRC::Complex(enuc_c) => enuc_c,
        };
        let onee_c = match onee {
            PyArray2RC::Real(pyonee_r) => pyonee_r.to_owned_array().mapv(Complex::from),
            PyArray2RC::Complex(pyonee_c) => pyonee_c.to_owned_array(),
        };
        let twoe_c = match twoe {
            PyArray4RC::Real(pytwoe_r) => pytwoe_r.to_owned_array().mapv(Complex::from),
            PyArray4RC::Complex(pytwoe_c) => pytwoe_c.to_owned_array(),
        };

        match structure_constraint {
            PyStructureConstraint::SpinConstraint(_) => {
                let overlap_ao = OverlapAO::<Complex<f64>, SpinConstraint>::builder()
                    .sao(sao_c.view())
                    .build()
                    .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
                let hamiltonian_ao = HamiltonianAO::<Complex<f64>, SpinConstraint>::builder()
                    .enuc(enuc_c)
                    .onee(onee_c.view())
                    .twoe(twoe_c.view())
                    .build()
                    .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;

                let origins_c = if augment_to_generalised {
                    pyorigins
                        .iter()
                        .map(|pydet| match pydet {
                            PySlaterDeterminant::Real(pydet_r) => pydet_r
                                .to_qsym2::<SpinConstraint>(&baos_ref, mol)
                                .map(|det_r| det_r.to_generalised().into()),
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
                                .map(|det_r| det_r.into()),
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

                        // Run NOCI
                        let system = (&hamiltonian_ao, &overlap_ao);
                        let multidets = system
                            .solve_symmetry_noci(
                                &origins_c.iter().collect_vec(),
                                &group,
                                symmetry_transformation_kind,
                                use_cayley_table,
                                thresh_offdiag,
                                thresh_zeroov,
                            )
                            .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;

                        // Symmetry analysis for NOCI states
                        let mut mda_driver = MultiDeterminantRepAnalysisDriver::<
                            MagneticRepresentedSymmetryGroup,
                            Complex<f64>,
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

                        // Run NOCI
                        let system = (&hamiltonian_ao, &overlap_ao);
                        let multidets = system
                            .solve_symmetry_noci(
                                &origins_c.iter().collect_vec(),
                                &group,
                                symmetry_transformation_kind,
                                use_cayley_table,
                                thresh_offdiag,
                                thresh_zeroov,
                            )
                            .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;

                        // Symmetry analysis for NOCI states
                        let mut mda_driver = MultiDeterminantRepAnalysisDriver::<
                            UnitaryRepresentedSymmetryGroup,
                            Complex<f64>,
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
                        py.allow_threads(|| {
                            mda_driver
                                .run()
                                .map_err(|err| PyRuntimeError::new_err(err.to_string()))
                        })?
                    }
                };
            }
            PyStructureConstraint::SpinOrbitCoupled(_) => {
                let overlap_ao = OverlapAO::<Complex<f64>, SpinOrbitCoupled>::builder()
                    .sao(sao_c.view())
                    .build()
                    .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
                let hamiltonian_ao = HamiltonianAO::<Complex<f64>, SpinOrbitCoupled>::builder()
                    .enuc(enuc_c)
                    .onee(onee_c.view())
                    .twoe(twoe_c.view())
                    .build()
                    .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;

                let origins_c = pyorigins
                    .iter()
                    .map(|pydet| match pydet {
                        PySlaterDeterminant::Real(pydet_r) => pydet_r
                            .to_qsym2::<SpinOrbitCoupled>(&baos_ref, mol)
                            .map(|det_r| det_r.into()),
                        PySlaterDeterminant::Complex(pydet_c) => {
                            pydet_c.to_qsym2::<SpinOrbitCoupled>(&baos_ref, mol)
                        }
                    })
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;

                // DEBUG
                let (zeroe, onee, twoe) = hamiltonian_ao
                    .calc_hamiltonian_matrix_element_contributions(
                        &origins_c[0],
                        &origins_c[0],
                        overlap_ao.sao(),
                        1e-10,
                        1e-7,
                    )
                    .unwrap();
                log::debug!("Origin energies:\n  {zeroe}\n  {onee}\n  {twoe}");
                // END DEBUG

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

                        // Run NOCI
                        let system = (&hamiltonian_ao, &overlap_ao);
                        let multidets = system
                            .solve_symmetry_noci(
                                &origins_c.iter().collect_vec(),
                                &group,
                                symmetry_transformation_kind,
                                use_cayley_table,
                                thresh_offdiag,
                                thresh_zeroov,
                            )
                            .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;

                        // Symmetry analysis for NOCI states
                        let mut mda_driver = MultiDeterminantRepAnalysisDriver::<
                            MagneticRepresentedSymmetryGroup,
                            Complex<f64>,
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

                        // Run NOCI
                        let system = (&hamiltonian_ao, &overlap_ao);
                        let multidets = system
                            .solve_symmetry_noci(
                                &origins_c.iter().collect_vec(),
                                &group,
                                symmetry_transformation_kind,
                                use_cayley_table,
                                thresh_offdiag,
                                thresh_zeroov,
                            )
                            .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;

                        // Symmetry analysis for NOCI states
                        let mut mda_driver = MultiDeterminantRepAnalysisDriver::<
                            UnitaryRepresentedSymmetryGroup,
                            Complex<f64>,
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
                        py.allow_threads(|| {
                            mda_driver
                                .run()
                                .map_err(|err| PyRuntimeError::new_err(err.to_string()))
                        })?
                    }
                };
            }
        }
    }

    Ok(())
}
