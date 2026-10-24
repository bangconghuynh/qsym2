//! Python bindings for QSym² symmetry projection of Slater determinants.

use std::collections::HashSet;
use std::path::PathBuf;

use anyhow::format_err;
use itertools::Itertools;
use ndarray::{Array2, Axis, ShapeBuilder};
use num_complex::Complex;
use numpy::{PyArrayMethods, ToPyArray};
use pyo3::exceptions::{PyIOError, PyRuntimeError};
use pyo3::types::PyFunction;
use pyo3::{IntoPyObjectExt, prelude::*};

use crate::angmom::spinor_rotation_3d::{SpinConstraint, SpinOrbitCoupled};
use crate::bindings::python::integrals::{PyBasisAngularOrder, PyStructureConstraint};
use crate::bindings::python::projection::PyProjectionTarget;
use crate::bindings::python::representation_analysis::multideterminant::{
    PyMultiDeterminantsComplex, PyMultiDeterminantsReal,
};
use crate::bindings::python::representation_analysis::slater_determinant::PySlaterDeterminant;
use crate::bindings::python::representation_analysis::{PyArray2RC, PyArray4RC, PyScalarRC};
use crate::drivers::QSym2Driver;
use crate::drivers::projection::slater_determinant::{
    SlaterDeterminantProjectionDriver, SlaterDeterminantProjectionParams,
};
use crate::drivers::representation_analysis::{
    CharacterTableDisplay, MagneticSymmetryAnalysisKind,
};
use crate::drivers::symmetry_group_detection::SymmetryGroupDetectionResult;
use crate::io::format::qsym2_output;
use crate::io::{QSym2FileType, read_qsym2_binary};
use crate::symmetry::symmetry_group::UnitaryRepresentedSymmetryGroup;
use crate::symmetry::symmetry_transformation::SymmetryTransformationKind;
use crate::target::noci::basis::Basis;
use crate::target::noci::multideterminants::MultiDeterminants;

type C128 = Complex<f64>;

// =====================
// Functions definitions
// =====================

/// Python-exposed function to perform symmetry projection for real and complex Slater determinants.
///
/// # Arguments
///
/// * `inp_sym` - A path to the [`QSym2FileType::Sym`] file containing the symmetry-group detection
/// result for the system. This will be used to construct abstract groups and character tables for
/// symmetry projection.
/// * `pydet` - A Slater determinant whose coefficient matrices are of type `float64` or `complex128`.
/// * `projection_targets` - A sequence of subspace labels for projection. Each label is either a
/// symbolic string or a numerical index for the subspace in the character table of the prevailing
/// group.
/// * `calculate_energies` - A boolean indicating if the energies of the projected wavefunctions
/// (which are multi-determinantal wavefunctions specified in an orbit basis) are to be calculated.
/// If `true`, then `thresh_offdiag`, `thresh_zeroov`, `enuc`, `onee`, and either `twoe` or
/// `py_get_jk` must also be specified for Löwdin pairing and Hamiltonian matrix element computation.
/// * `calculate_density_matrices` - A boolean indicating if the density matrices for the projected
/// wavefunctions (which are multi-determinantal wavefunctions) are to be calculated. If `true`,
/// then `thresh_offdiag` and `thresh_zeroov` must also be specified for Löwdin pairing.
/// * `pybaos` - Python-exposed structures containing basis angular order information, one for each
/// explicit component per coefficient matrix.
/// * `use_magnetic_group` - An option indicating if the magnetic group is to be used for symmetry
/// analysis, and if so, whether unitary representations or unitary-antiunitary corepresentations
/// should be used.
/// * `use_double_group` - A boolean indicating if the double group of the prevailing symmetry
/// group is to be used for representation analysis instead.
/// * `symmetry_transformation_kind` - An enumerated type indicating the type of symmetry
/// transformations to be performed on the origin Slater determinant to generate the orbit.
/// * `write_character_table` - A boolean indicating if the character table of the prevailing
/// symmetry group is to be printed out.
/// * `infinite_order_to_finite` - The finite order with which infinite-order generators are to be
/// interpreted to form a finite subgroup of the prevailing infinite group. This finite subgroup
/// will be used for symmetry analysis.
/// * `sao` - The optional atomic-orbital overlap matrix whose elements are of type `float64` or
/// `complex128`. If this is not present, no squared norms, energies, or density matrices of the
/// resulting multi-determinants will be computed.
/// * `sao_h` - The optional complex-symmetric atomic-orbital overlap matrix whose elements
/// are of type `float64` or `complex128`. This is required if antiunitary symmetry operations are
/// involved.
/// * `enuc` - The optional nuclear repulsion energy, which is only required for `calculate_energies`.
/// * `onee` - The optional one-electron integral matrix whose elements are of type `float64` or
/// `complex128`, which is only required for `calculate_energies`.
/// * `twoe` - The optional two-electron integral tensor whose elements are of type `float64` or
/// `complex128`, which is only required for `calculate_energies`. Either this or `py_get_jk`
/// must be specified, but not both.
/// * `py_get_jk` - An optional Python function callable on a density matrix $`\mathbf{D}`$ to
/// calculate the corresponding $`\mathbf{J}`$ and $`\mathbf{K}`$ matrices by contracting with
/// appropriate two-electron integrals calculated on-the-fly, which is only required for
/// `calculate_energies`. Either this or `twoe` must be specified, but not both.
/// Python type: `Callable[[numpy.2darray[float] | numpy.2darray[complex]], tuple[numpy.2darray[float], numpy.2darray[float]] | tuple[numpy.2darray[complex], numpy.2darray[complex]]] | None`.
/// * `thresh_offdiag` - The optional threshold for identifying non-zero off-diagonal elements in
/// Löwdin pairing, which is required for `calculate_energies` and `calculate_density_matrices`.
/// * `thresh_zeroov` - The optional threshold for identifying non-zero overlaps in Löwdin pairing,
/// which is required for `calculate_energies` and `calculate_density_matrices`.
///
/// # Returns
///
/// The result will be returned as a tuple where the first item is a list of the labels of the
/// subspaces used for projection, and the second item is an object containing
///   * the Slater determinant basis,
///   * the linear combination coefficients as a two-dimensional array with each column corresponding
///   to one projected state,
///   * the energies of the projected states if requested, or `NaN` if not, and
///   * the density matrices of the projected states if requested, or `None` if not.
#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (
    inp_sym,
    pydet,
    projection_targets,
    calculate_energies,
    calculate_density_matrices,
    pybaos,
    use_magnetic_group,
    use_double_group,
    symmetry_transformation_kind,
    write_character_table=true,
    infinite_order_to_finite=None,
    sao=None,
    sao_h=None,
    enuc=None,
    onee=None,
    twoe=None,
    py_get_jk=None,
    thresh_offdiag=None,
    thresh_zeroov=None,
))]
pub fn project_slater_determinant(
    py: Python<'_>,
    inp_sym: PathBuf,
    pydet: PySlaterDeterminant,
    projection_targets: Vec<PyProjectionTarget>,
    calculate_energies: bool,
    calculate_density_matrices: bool,
    pybaos: Vec<PyBasisAngularOrder>,
    use_magnetic_group: Option<MagneticSymmetryAnalysisKind>,
    use_double_group: bool,
    symmetry_transformation_kind: SymmetryTransformationKind,
    write_character_table: bool,
    infinite_order_to_finite: Option<u32>,
    sao: Option<PyArray2RC>,
    sao_h: Option<PyArray2RC>,
    enuc: Option<PyScalarRC>,
    onee: Option<PyArray2RC>,
    twoe: Option<PyArray4RC>,
    py_get_jk: Option<Py<PyFunction>>,
    thresh_offdiag: Option<f64>,
    thresh_zeroov: Option<f64>,
) -> PyResult<(Vec<String>, Py<PyAny>)> {
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

    let symbolic_projection_targets = projection_targets
        .iter()
        .filter_map(|pt| match pt {
            PyProjectionTarget::Symbolic(symbolic_pt) => Some(symbolic_pt.clone()),
            PyProjectionTarget::Numeric(_) => None,
        })
        .collect_vec();
    let numeric_projection_targets = projection_targets
        .iter()
        .filter_map(|pt| match pt {
            PyProjectionTarget::Symbolic(_) => None,
            PyProjectionTarget::Numeric(numeric_pt) => Some(*numeric_pt),
        })
        .collect_vec();

    let sdp_params = SlaterDeterminantProjectionParams::builder()
        .symbolic_projection_targets(Some(symbolic_projection_targets))
        .numeric_projection_targets(Some(numeric_projection_targets))
        .use_magnetic_group(use_magnetic_group.clone())
        .use_double_group(use_double_group)
        .symmetry_transformation_kind(symmetry_transformation_kind)
        .write_character_table(if write_character_table {
            Some(CharacterTableDisplay::Symbolic)
        } else {
            None
        })
        .infinite_order_to_finite(infinite_order_to_finite)
        .build()
        .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;

    // Decision tree:
    // - real determinant?
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
    match pydet {
        PySlaterDeterminant::Real(pydet_r) => {
            if matches!(
                pydet_r.structure_constraint,
                PyStructureConstraint::SpinOrbitCoupled(_)
            ) {
                return Err(PyRuntimeError::new_err("Real determinants are not compatible with spin--orbit-coupled structure constraint.".to_string()));
            }

            let sao_opt = sao.and_then(|pysao| match pysao {
                PyArray2RC::Real(pysao_r) => Some(pysao_r.to_owned_array()),
                PyArray2RC::Complex(_) => None,
            });
            let sao_h_opt = sao_h.and_then(|pysao_h| match pysao_h {
                PyArray2RC::Real(pysao_h_r) => Some(pysao_h_r.to_owned_array()),
                PyArray2RC::Complex(_) => None,
            });
            let enuc_opt = enuc.and_then(|pyenuc| match pyenuc {
                PyScalarRC::Real(pyenuc_r) => Some(pyenuc_r),
                PyScalarRC::Complex(_) => None,
            });
            let onee_opt = onee.and_then(|pyonee| match pyonee {
                PyArray2RC::Real(pyonee_r) => Some(pyonee_r.to_owned_array()),
                PyArray2RC::Complex(_) => None,
            });
            let twoe_opt = twoe.and_then(|pytwoe| match pytwoe {
                PyArray4RC::Real(pytwoe_r) => Some(pytwoe_r.to_owned_array()),
                PyArray4RC::Complex(_) => None,
            });
            let get_jk_opt = py_get_jk.map(|py_get_jk_inner| {
                crate::generate_get_jk!(get_jk_r, py_get_jk_inner, f64);
                get_jk_r
            });

            let det_r = if augment_to_generalised {
                pydet_r
                    .to_qsym2::<SpinConstraint>(&baos_ref, mol)
                    .map_err(|err| PyRuntimeError::new_err(err.to_string()))?
                    .to_generalised()
            } else {
                pydet_r
                    .to_qsym2::<SpinConstraint>(&baos_ref, mol)
                    .map_err(|err| PyRuntimeError::new_err(err.to_string()))?
            };

            let projected_sds = match &use_magnetic_group {
                Some(MagneticSymmetryAnalysisKind::Corepresentation) => {
                    return Err(PyRuntimeError::new_err(
                        "Projection using corepresentations is not yet supported.",
                    ));
                }
                Some(MagneticSymmetryAnalysisKind::Representation) | None => {
                    let mut sdp_driver = if calculate_energies {
                        SlaterDeterminantProjectionDriver::<
                            UnitaryRepresentedSymmetryGroup,
                            f64,
                            SpinConstraint,
                            _,
                        >::builder()
                        .parameters(&sdp_params)
                        .determinant(&det_r)
                        .symmetry_group(&pd_res)
                        .sao(sao_opt.as_ref())
                        .sao_h(sao_h_opt.as_ref())
                        .enuc(enuc_opt)
                        .onee(onee_opt.as_ref())
                        .twoe(twoe_opt.as_ref())
                        .get_jk(get_jk_opt)
                        .thresh_offdiag(thresh_offdiag)
                        .thresh_zeroov(thresh_zeroov)
                        .build()
                        .map_err(|err| PyRuntimeError::new_err(err.to_string()))?
                    } else {
                        SlaterDeterminantProjectionDriver::<
                            UnitaryRepresentedSymmetryGroup,
                            f64,
                            SpinConstraint,
                            _,
                        >::builder()
                        .parameters(&sdp_params)
                        .determinant(&det_r)
                        .symmetry_group(&pd_res)
                        .sao(sao_opt.as_ref())
                        .sao_h(sao_h_opt.as_ref())
                        .build()
                        .map_err(|err| PyRuntimeError::new_err(err.to_string()))?
                    };
                    py.detach(|| {
                        sdp_driver
                            .run()
                            .map_err(|err| PyRuntimeError::new_err(err.to_string()))
                    })?;
                    sdp_driver
                        .result()
                        .map_err(|err| PyRuntimeError::new_err(err.to_string()))?
                        .projected_determinants()
                        .clone()
                }
            };
            let multidets_vec = projected_sds
                .values()
                .map(|v| v.as_ref())
                .collect::<Result<Vec<_>, _>>()
                .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
            let multidets = MultiDeterminants::from_multideterminant_vec(
                &multidets_vec.into_iter().collect::<Vec<_>>(),
            )
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;

            let energies = multidets.energies();

            let (rows, coefficientss): (Vec<_>, Vec<_>) = projected_sds
                .iter()
                .map(|(row, multidet_res)| {
                    let coefficients = multidet_res
                        .as_ref()
                        .map(|multidet| multidet.coefficients().iter().cloned().collect_vec())
                        .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
                    Ok::<_, PyErr>((row.to_string(), coefficients))
                })
                .collect::<Result<Vec<_>, _>>()?
                .into_iter()
                .unzip();
            let basis = projected_sds
                .iter()
                .next()
                .and_then(|(_, multidet_res)| {
                    multidet_res.as_ref().ok().and_then(|multidet| {
                        multidet
                            .basis()
                            .iter()
                            .map(|det_res| det_res.and_then(|det| det.to_python(py)))
                            .collect::<Result<Vec<_>, _>>()
                            .ok()
                    })
                })
                .ok_or_else(|| {
                    PyRuntimeError::new_err(
                        "Unable to obtain the basis of Slater determinants.".to_string(),
                    )
                })?;
            let coefficientss_arr = Array2::from_shape_vec(
                (basis.len(), coefficientss.len()).f(),
                coefficientss.into_iter().flatten().collect_vec(),
            )
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))?
            .to_pyarray(py);
            let energies_arr = energies.mapv(|v| v.unwrap_or(f64::NAN)).to_pyarray(py);

            let density_matrices = match (
                calculate_density_matrices,
                thresh_offdiag,
                thresh_zeroov,
                sao_opt.as_ref(),
            ) {
                (true, Some(thresh_offdiag), Some(thresh_zeroov), Some(sao)) => {
                    log::debug!("Calculating density matrices...");
                    let denmats_opt = multidets
                        .density_matrices(&sao.view(), thresh_offdiag, thresh_zeroov, true)
                        .map(|denmats| {
                            denmats
                                .axis_iter(Axis(0))
                                .map(|denmat| denmat.to_pyarray(py))
                                .collect_vec()
                        })
                        .ok();
                    log::debug!("Calculating density matrices... Done.");
                    denmats_opt
                }
                _ => None,
            };

            let pymultidet = PyMultiDeterminantsReal::new(
                basis,
                coefficientss_arr,
                energies_arr,
                density_matrices,
                pydet_r.threshold,
            )
            .into_py_any(py)?;
            Ok((rows, pymultidet))
        }
        PySlaterDeterminant::Complex(pydet_c) => {
            let sao_opt = sao.map(|pysao| match pysao {
                PyArray2RC::Real(pysao_r) => pysao_r.to_owned_array().mapv(Complex::from),
                PyArray2RC::Complex(pysao_c) => pysao_c.to_owned_array(),
            });
            let sao_h_opt = sao_h.map(|pysao_h| match pysao_h {
                PyArray2RC::Real(pysao_h_r) => pysao_h_r.to_owned_array().mapv(Complex::from),
                PyArray2RC::Complex(pysao_h_c) => pysao_h_c.to_owned_array(),
            });
            let enuc_opt = enuc.map(|pyenuc| match pyenuc {
                PyScalarRC::Real(pyenuc_r) => Complex::from(pyenuc_r),
                PyScalarRC::Complex(pyenuc_c) => pyenuc_c,
            });
            let onee_opt = onee.map(|pyonee| match pyonee {
                PyArray2RC::Real(pyonee_r) => pyonee_r.to_owned_array().mapv(Complex::from),
                PyArray2RC::Complex(pyonee_c) => pyonee_c.to_owned_array(),
            });
            let twoe_opt = twoe.map(|pytwoe| match pytwoe {
                PyArray4RC::Real(pytwoe_r) => pytwoe_r.to_owned_array().mapv(Complex::from),
                PyArray4RC::Complex(pytwoe_c) => pytwoe_c.to_owned_array(),
            });
            let get_jk_opt = py_get_jk.map(|py_get_jk_inner| {
                crate::generate_get_jk!(get_jk_c, py_get_jk_inner, Complex<f64>);
                get_jk_c
            });

            match pydet_c.structure_constraint {
                PyStructureConstraint::SpinConstraint(_) => {
                    let det_c = if augment_to_generalised {
                        pydet_c
                            .to_qsym2::<SpinConstraint>(&baos_ref, mol)
                            .map_err(|err| PyRuntimeError::new_err(err.to_string()))?
                            .to_generalised()
                    } else {
                        pydet_c
                            .to_qsym2::<SpinConstraint>(&baos_ref, mol)
                            .map_err(|err| PyRuntimeError::new_err(err.to_string()))?
                    };

                    let projected_sds = match &use_magnetic_group {
                        Some(MagneticSymmetryAnalysisKind::Corepresentation) => {
                            return Err(PyRuntimeError::new_err(
                                "Projection using corepresentations is not yet supported.",
                            ));
                        }
                        Some(MagneticSymmetryAnalysisKind::Representation) | None => {
                            let mut sdp_driver = if calculate_energies {
                                SlaterDeterminantProjectionDriver::<
                                    UnitaryRepresentedSymmetryGroup,
                                    C128,
                                    SpinConstraint,
                                    _,
                                >::builder()
                                .parameters(&sdp_params)
                                .determinant(&det_c)
                                .symmetry_group(&pd_res)
                                .sao(sao_opt.as_ref())
                                .sao_h(sao_h_opt.as_ref())
                                .enuc(enuc_opt)
                                .onee(onee_opt.as_ref())
                                .twoe(twoe_opt.as_ref())
                                .get_jk(get_jk_opt)
                                .thresh_offdiag(thresh_offdiag)
                                .thresh_zeroov(thresh_zeroov)
                                .build()
                                .map_err(|err| PyRuntimeError::new_err(err.to_string()))?
                            } else {
                                SlaterDeterminantProjectionDriver::<
                                    UnitaryRepresentedSymmetryGroup,
                                    C128,
                                    SpinConstraint,
                                    _,
                                >::builder()
                                .parameters(&sdp_params)
                                .determinant(&det_c)
                                .symmetry_group(&pd_res)
                                .sao(sao_opt.as_ref())
                                .sao_h(sao_h_opt.as_ref())
                                .build()
                                .map_err(|err| PyRuntimeError::new_err(err.to_string()))?
                            };
                            py.detach(|| {
                                sdp_driver
                                    .run()
                                    .map_err(|err| PyRuntimeError::new_err(err.to_string()))
                            })?;
                            sdp_driver
                                .result()
                                .map_err(|err| PyRuntimeError::new_err(err.to_string()))?
                                .projected_determinants()
                                .clone()
                        }
                    };
                    let multidets_vec = projected_sds
                        .values()
                        .map(|v| v.as_ref())
                        .collect::<Result<Vec<_>, _>>()
                        .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
                    let multidets = MultiDeterminants::from_multideterminant_vec(
                        &multidets_vec.into_iter().collect::<Vec<_>>(),
                    )
                    .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;

                    let energies = multidets.energies();

                    let (rows, coefficientss): (Vec<_>, Vec<_>) = projected_sds
                        .iter()
                        .map(|(row, multidet_res)| {
                            let coefficients = multidet_res
                                .as_ref()
                                .map(|multidet| {
                                    multidet.coefficients().iter().cloned().collect_vec()
                                })
                                .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
                            Ok::<_, PyErr>((row.to_string(), coefficients))
                        })
                        .collect::<Result<Vec<_>, _>>()?
                        .into_iter()
                        .unzip();
                    let basis = projected_sds
                        .iter()
                        .next()
                        .and_then(|(_, multidet_res)| {
                            multidet_res.as_ref().ok().and_then(|multidet| {
                                multidet
                                    .basis()
                                    .iter()
                                    .map(|det_res| det_res.and_then(|det| det.to_python(py)))
                                    .collect::<Result<Vec<_>, _>>()
                                    .ok()
                            })
                        })
                        .ok_or_else(|| {
                            PyRuntimeError::new_err(
                                "Unable to obtain the basis of Slater determinants.".to_string(),
                            )
                        })?;
                    let coefficientss_arr = Array2::from_shape_vec(
                        (basis.len(), coefficientss.len()).f(),
                        coefficientss.into_iter().flatten().collect_vec(),
                    )
                    .map_err(|err| PyRuntimeError::new_err(err.to_string()))?
                    .to_pyarray(py);
                    let energies_arr = energies
                        .mapv(|v| v.unwrap_or(Complex::from(f64::NAN)))
                        .to_pyarray(py);

                    let density_matrices = match (
                        calculate_density_matrices,
                        thresh_offdiag,
                        thresh_zeroov,
                        sao_opt.as_ref(),
                    ) {
                        (true, Some(thresh_offdiag), Some(thresh_zeroov), Some(sao)) => {
                            log::debug!("Calculating density matrices...");
                            let denmats_opt = multidets
                                .density_matrices(&sao.view(), thresh_offdiag, thresh_zeroov, true)
                                .map(|denmats| {
                                    denmats
                                        .axis_iter(Axis(0))
                                        .map(|denmat| denmat.to_pyarray(py))
                                        .collect_vec()
                                })
                                .ok();
                            log::debug!("Calculating density matrices... Done.");
                            denmats_opt
                        }
                        _ => None,
                    };

                    let pymultidet = PyMultiDeterminantsComplex::new(
                        basis,
                        coefficientss_arr,
                        energies_arr,
                        density_matrices,
                        pydet_c.threshold,
                    )
                    .into_py_any(py)?;
                    Ok((rows, pymultidet))
                }
                PyStructureConstraint::SpinOrbitCoupled(_) => {
                    let det_c = pydet_c
                        .to_qsym2::<SpinOrbitCoupled>(&baos_ref, mol)
                        .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;

                    let projected_sds = match &use_magnetic_group {
                        Some(MagneticSymmetryAnalysisKind::Corepresentation) => {
                            return Err(PyRuntimeError::new_err(
                                "Projection using corepresentations is not yet supported.",
                            ));
                        }
                        Some(MagneticSymmetryAnalysisKind::Representation) | None => {
                            let mut sdp_driver = if calculate_energies {
                                SlaterDeterminantProjectionDriver::<
                                    UnitaryRepresentedSymmetryGroup,
                                    C128,
                                    SpinOrbitCoupled,
                                    _,
                                >::builder()
                                .parameters(&sdp_params)
                                .determinant(&det_c)
                                .symmetry_group(&pd_res)
                                .sao(sao_opt.as_ref())
                                .sao_h(sao_h_opt.as_ref())
                                .enuc(enuc_opt)
                                .onee(onee_opt.as_ref())
                                .twoe(twoe_opt.as_ref())
                                .get_jk(get_jk_opt)
                                .thresh_offdiag(thresh_offdiag)
                                .thresh_zeroov(thresh_zeroov)
                                .build()
                                .map_err(|err| PyRuntimeError::new_err(err.to_string()))?
                            } else {
                                SlaterDeterminantProjectionDriver::<
                                    UnitaryRepresentedSymmetryGroup,
                                    C128,
                                    SpinOrbitCoupled,
                                    _,
                                >::builder()
                                .parameters(&sdp_params)
                                .determinant(&det_c)
                                .symmetry_group(&pd_res)
                                .sao(sao_opt.as_ref())
                                .sao_h(sao_h_opt.as_ref())
                                .build()
                                .map_err(|err| PyRuntimeError::new_err(err.to_string()))?
                            };
                            py.detach(|| {
                                sdp_driver
                                    .run()
                                    .map_err(|err| PyRuntimeError::new_err(err.to_string()))
                            })?;
                            sdp_driver
                                .result()
                                .map_err(|err| PyRuntimeError::new_err(err.to_string()))?
                                .projected_determinants()
                                .clone()
                        }
                    };
                    let multidets_vec = projected_sds
                        .values()
                        .map(|v| v.as_ref())
                        .collect::<Result<Vec<_>, _>>()
                        .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
                    let multidets = MultiDeterminants::from_multideterminant_vec(
                        &multidets_vec.into_iter().collect::<Vec<_>>(),
                    )
                    .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;

                    let energies = multidets.energies();

                    let (rows, coefficientss): (Vec<_>, Vec<_>) = projected_sds
                        .iter()
                        .map(|(row, multidet_res)| {
                            let coefficients = multidet_res
                                .as_ref()
                                .map(|multidet| {
                                    multidet.coefficients().iter().cloned().collect_vec()
                                })
                                .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
                            Ok::<_, PyErr>((row.to_string(), coefficients))
                        })
                        .collect::<Result<Vec<_>, _>>()?
                        .into_iter()
                        .unzip();
                    let basis = projected_sds
                        .iter()
                        .next()
                        .and_then(|(_, multidet_res)| {
                            multidet_res.as_ref().ok().and_then(|multidet| {
                                multidet
                                    .basis()
                                    .iter()
                                    .map(|det_res| det_res.and_then(|det| det.to_python(py)))
                                    .collect::<Result<Vec<_>, _>>()
                                    .ok()
                            })
                        })
                        .ok_or_else(|| {
                            PyRuntimeError::new_err(
                                "Unable to obtain the basis of Slater determinants.".to_string(),
                            )
                        })?;
                    let coefficientss_arr = Array2::from_shape_vec(
                        (basis.len(), coefficientss.len()).f(),
                        coefficientss.into_iter().flatten().collect_vec(),
                    )
                    .map_err(|err| PyRuntimeError::new_err(err.to_string()))?
                    .to_pyarray(py);
                    let energies_arr = energies
                        .mapv(|v| v.unwrap_or(Complex::from(f64::NAN)))
                        .to_pyarray(py);

                    let density_matrices = match (
                        calculate_density_matrices,
                        thresh_offdiag,
                        thresh_zeroov,
                        sao_opt.as_ref(),
                    ) {
                        (true, Some(thresh_offdiag), Some(thresh_zeroov), Some(sao)) => {
                            log::debug!("Calculating density matrices...");
                            let denmats_opt = multidets
                                .density_matrices(&sao.view(), thresh_offdiag, thresh_zeroov, true)
                                .map(|denmats| {
                                    denmats
                                        .axis_iter(Axis(0))
                                        .map(|denmat| denmat.to_pyarray(py))
                                        .collect_vec()
                                })
                                .ok();
                            log::debug!("Calculating density matrices... Done.");
                            denmats_opt
                        }
                        _ => None,
                    };

                    let pymultidet = PyMultiDeterminantsComplex::new(
                        basis,
                        coefficientss_arr,
                        energies_arr,
                        density_matrices,
                        pydet_c.threshold,
                    )
                    .into_py_any(py)?;
                    Ok((rows, pymultidet))
                }
            }
        }
    }
}
