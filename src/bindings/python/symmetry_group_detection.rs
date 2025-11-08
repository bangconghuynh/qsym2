//! Python bindings for QSym² symmetry-group detection.
//!
//! See [`crate::drivers::symmetry_group_detection`] for more information.

use std::collections::HashMap;
use std::fmt;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{self, format_err};
use derive_builder::Builder;
use nalgebra::{Point3, Vector3};
use numpy::{PyArray1, ToPyArray};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use crate::auxiliary::atom::{Atom, ElementMap};
use crate::auxiliary::molecule::Molecule;
use crate::drivers::QSym2Driver;
use crate::drivers::symmetry_group_detection::{
    SymmetryGroupDetectionDriver, SymmetryGroupDetectionParams,
};
#[allow(unused_imports)]
use crate::io::QSym2FileType;
use crate::symmetry::symmetry_core::Symmetry;
use crate::symmetry::symmetry_element::{AntiunitaryKind, SymmetryElementKind};
use crate::symmetry::symmetry_element_order::ElementOrder;

// ===========================
// Struct and enum definitions
// ===========================

// ----------
// PyMolecule
// ----------

/// Python-exposed structure to marshall molecular structure information between Rust and Python.
#[pyclass]
#[derive(Clone)]
pub struct PyMolecule {
    /// The ordinary atoms in the molecule.
    ///
    /// Python type: `list[tuple[str, tuple[float, float, float]]]`
    #[pyo3(get)]
    pub atoms: Vec<(String, [f64; 3])>,

    /// An optional uniform external magnetic field.
    ///
    /// Python type: `Optional[tuple[float, float, float]]`
    #[pyo3(get)]
    pub magnetic_field: Option<[f64; 3]>,

    /// An optional uniform external electric field.
    ///
    /// Python type: `Optional[tuple[float, float, float]]`
    #[pyo3(get)]
    pub electric_field: Option<[f64; 3]>,

    /// Threshold for comparing molecules.
    ///
    /// Python type: `float`
    #[pyo3(get)]
    pub threshold: f64,
}

#[pymethods]
impl PyMolecule {
    /// Creates a new `PyMolecule` structure.
    ///
    /// # Arguments
    ///
    /// * `atoms` - The ordinary atoms in the molecule.
    /// * `threshold` - Threshold for comparing molecules.
    /// * `magnetic_field` - An optional uniform external magnetic field.
    /// * `electric_field` - An optional uniform external electric field.
    #[new]
    #[pyo3(signature = (atoms, threshold, magnetic_field=None, electric_field=None))]
    pub fn new(
        atoms: Vec<(String, [f64; 3])>,
        threshold: f64,
        magnetic_field: Option<[f64; 3]>,
        electric_field: Option<[f64; 3]>,
    ) -> Self {
        Self {
            atoms,
            threshold,
            magnetic_field,
            electric_field,
        }
    }
}

impl From<PyMolecule> for Molecule {
    fn from(pymol: PyMolecule) -> Self {
        let emap = ElementMap::new();
        let mut mol = Self::from_atoms(
            &pymol
                .atoms
                .iter()
                .map(|(ele, r)| {
                    Atom::new_ordinary(ele, Point3::new(r[0], r[1], r[2]), &emap, pymol.threshold)
                })
                .collect::<Vec<_>>(),
            pymol.threshold,
        );
        mol.set_magnetic_field(pymol.magnetic_field.map(Vector3::from_iterator));
        mol.set_electric_field(pymol.electric_field.map(Vector3::from_iterator));
        mol
    }
}

// ---------------------
// PySymmetryElementKind
// ---------------------

/// Python-exposed enumerated type to marshall symmetry element kind information one-way from Rust to
/// Python.
#[pyclass(eq, eq_int)]
#[derive(Clone, Hash, PartialEq, Eq)]
pub enum PySymmetryElementKind {
    /// Variant denoting proper symmetry elements.
    Proper,

    /// Variant denoting time-reversed proper symmetry elements.
    ProperTR,

    /// Variant denoting improper symmetry elements (mirror-plane convention).
    ImproperMirrorPlane,

    /// Variant denoting time-reversed improper symmetry elements (mirror-plane convention).
    ImproperMirrorPlaneTR,
}

impl fmt::Display for PySymmetryElementKind {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            PySymmetryElementKind::Proper => write!(f, "Proper"),
            PySymmetryElementKind::ProperTR => write!(f, "Time-reversed proper"),
            PySymmetryElementKind::ImproperMirrorPlane => {
                write!(f, "Improper (mirror-plane convention)")
            }
            PySymmetryElementKind::ImproperMirrorPlaneTR => {
                write!(f, "Time-reversed improper (mirror-plane convention)")
            }
        }
    }
}

impl TryFrom<&SymmetryElementKind> for PySymmetryElementKind {
    type Error = anyhow::Error;

    fn try_from(symkind: &SymmetryElementKind) -> Result<Self, Self::Error> {
        match symkind {
            SymmetryElementKind::Proper(None) => Ok(Self::Proper),
            SymmetryElementKind::Proper(Some(AntiunitaryKind::TimeReversal)) => Ok(Self::ProperTR),
            SymmetryElementKind::ImproperMirrorPlane(None) => Ok(Self::ImproperMirrorPlane),
            SymmetryElementKind::ImproperMirrorPlane(Some(AntiunitaryKind::TimeReversal)) => {
                Ok(Self::ImproperMirrorPlaneTR)
            }
            _ => Err(format_err!(
                "Symmetry element kind `{symkind}` is not yet supported in Python."
            )),
        }
    }
}

// ----------
// PySymmetry
// ----------

/// Python-exposed structure to marshall symmetry information one-way from Rust to Python.
#[pyclass]
#[derive(Clone, Builder)]
pub struct PySymmetry {
    /// The name of the symmetry group.
    #[pyo3(get)]
    group_name: String,

    /// The symmetry elements.
    ///
    /// Python type: `dict[PySymmetryElementKind, dict[int, list[numpy.1darray[float]]]]`
    #[allow(clippy::type_complexity)]
    elements: HashMap<PySymmetryElementKind, HashMap<i32, Vec<Arc<Py<PyArray1<f64>>>>>>,

    /// The symmetry generators.
    ///
    /// Python type: `dict[PySymmetryElementKind, dict[int, list[numpy.1darray[float]]]]`
    #[allow(clippy::type_complexity)]
    generators: HashMap<PySymmetryElementKind, HashMap<i32, Vec<Arc<Py<PyArray1<f64>>>>>>,
}

impl PySymmetry {
    fn builder() -> PySymmetryBuilder {
        PySymmetryBuilder::default()
    }
}

#[pymethods]
impl PySymmetry {
    /// Returns a boolean indicating if the group is infinite.
    pub fn is_infinite(&self) -> bool {
        self.elements
            .values()
            .any(|kind_elements| kind_elements.contains_key(&-1))
            || self
                .generators
                .values()
                .any(|kind_generators| kind_generators.contains_key(&-1))
    }

    /// Returns symmetry elements of all *finite* orders of a given kind.
    ///
    /// # Arguments
    ///
    /// * `kind` - The symmetry element kind.
    ///
    /// # Returns
    ///
    /// A hashmap where the keys are integers indicating the orders of the elements and the values
    /// are vectors of one-dimensional arrays, each of which gives the axis of a symmetry element.
    /// If the order value is `-1`, then the associated elements have infinite order.
    pub fn get_elements_of_kind(
        &self,
        kind: &PySymmetryElementKind,
    ) -> PyResult<HashMap<i32, Vec<Py<PyArray1<f64>>>>> {
        self.elements
            .get(kind)
            .cloned()
            .and_then(|elements| {
                elements
                    .iter()
                    .map(|(order, axes_arc)| {
                        let axes_opt = axes_arc
                            .iter()
                            .map(|axis_arc| Arc::into_inner(axis_arc.clone()))
                            .collect::<Option<Vec<_>>>();
                        axes_opt.map(|axes| (*order, axes))
                    })
                    .collect::<Option<HashMap<_, _>>>()
            })
            .ok_or(PyRuntimeError::new_err(format!(
                "Elements of kind `{kind}` not found."
            )))
    }

    /// Returns symmetry generators of *finite*  and *infinite* orders of a given kind.
    ///
    /// # Arguments
    ///
    /// * `kind` - The symmetry generator kind.
    ///
    /// # Returns
    ///
    /// A hashmap where the keys are integers indicating the orders of the generators and the values
    /// are vectors of one-dimensional arrays, each of which gives the axis of a symmetry generator.
    /// If the order value is `-1`, then the associated generators have infinite order.
    pub fn get_generators_of_kind(
        &self,
        kind: &PySymmetryElementKind,
    ) -> PyResult<HashMap<i32, Vec<Py<PyArray1<f64>>>>> {
        self.generators
            .get(kind)
            .cloned()
            .and_then(|generators| {
                generators
                    .iter()
                    .map(|(order, axes_arc)| {
                        let axes_opt = axes_arc
                            .iter()
                            .map(|axis_arc| Arc::into_inner(axis_arc.clone()))
                            .collect::<Option<Vec<_>>>();
                        axes_opt.map(|axes| (*order, axes))
                    })
                    .collect::<Option<HashMap<_, _>>>()
            })
            .ok_or(PyRuntimeError::new_err(format!(
                "Elements of kind `{kind}` not found."
            )))
    }
}

impl TryFrom<&Symmetry> for PySymmetry {
    type Error = anyhow::Error;

    fn try_from(sym: &Symmetry) -> Result<Self, Self::Error> {
        let group_name = sym
            .group_name
            .clone()
            .ok_or(format_err!("Symmetry group name not found."))?;
        let elements = sym
            .elements
            .iter()
            .map(|(symkind, kind_elements)| {
                let pysymkind = PySymmetryElementKind::try_from(symkind)?;
                let pykind_elements = kind_elements
                    .iter()
                    .map(|(order, order_elements)| {
                        let order_i32 = match order {
                            ElementOrder::Int(ord) => i32::try_from(*ord)?,
                            ElementOrder::Inf => -1,
                        };
                        let pyorder_elements = order_elements
                            .iter()
                            .map(|ele| {
                                Arc::new(Python::attach(|py| {
                                    ele.raw_axis()
                                        .iter()
                                        .cloned()
                                        .collect::<Vec<_>>()
                                        .to_pyarray(py)
                                        .unbind()
                                }))
                            })
                            .collect::<Vec<_>>();
                        Ok::<_, Self::Error>((order_i32, pyorder_elements))
                    })
                    .collect::<Result<HashMap<i32, Vec<_>>, _>>()?;
                Ok::<_, Self::Error>((pysymkind, pykind_elements))
            })
            .collect::<Result<HashMap<_, _>, _>>()?;

        let generators = sym
            .generators
            .iter()
            .map(|(symkind, kind_generators)| {
                let pysymkind = PySymmetryElementKind::try_from(symkind)?;
                let pykind_generators = kind_generators
                    .iter()
                    .map(|(order, order_generators)| {
                        let order_i32 = match order {
                            ElementOrder::Int(ord) => i32::try_from(*ord)?,
                            ElementOrder::Inf => -1,
                        };
                        let pyorder_generators = order_generators
                            .iter()
                            .map(|ele| {
                                Arc::new(Python::attach(|py| {
                                    ele.raw_axis()
                                        .iter()
                                        .cloned()
                                        .collect::<Vec<_>>()
                                        .to_pyarray(py)
                                        .unbind()
                                }))
                            })
                            .collect::<Vec<_>>();
                        Ok::<_, Self::Error>((order_i32, pyorder_generators))
                    })
                    .collect::<Result<HashMap<i32, Vec<_>>, _>>()?;
                Ok::<_, Self::Error>((pysymkind, pykind_generators))
            })
            .collect::<Result<HashMap<_, _>, _>>()?;

        PySymmetry::builder()
            .group_name(group_name)
            .elements(elements)
            .generators(generators)
            .build()
            .map_err(|err| format_err!(err))
    }
}

// =========
// Functions
// =========

/// Python-exposed function to perform symmetry-group detection and log the result via the
/// `qsym2-output` logger at the `INFO` level.
///
/// See [`crate::drivers::symmetry_group_detection`] for more information.
///
/// # Arguments
///
/// * `inp_xyz` - An optional string providing the path to an XYZ file containing the molecule to
/// be analysed. Only one of `inp_xyz` or `inp_mol` can be specified.
/// * `inp_mol` - An optional `PyMolecule` structure containing the molecule to be analysed. Only
/// one of `inp_xyz` or `inp_mol` can be specified.
/// * `out_sym` - An optional name for the [`QSym2FileType::Sym`] file to be saved that contains
/// the serialised results of the symmetry-group detection.
/// * `moi_thresholds` - Thresholds for comparing moments of inertia.
/// * `distance_thresholds` - Thresholds for comparing distances.
/// * `time_reversal` - A boolean indicating whether elements involving time reversal should also
/// be considered.
/// * `write_symmetry_elements` - A boolean indicating if detected symmetry elements should be
/// printed in the output.
/// * `fictitious_magnetic_field` - An optional fictitious uniform external magnetic field.
/// * `fictitious_electric_field` - An optional fictitious uniform external electric field.
///
/// # Returns
///
/// Returns a tuple of a [`PySymmetry`] for the unitary group and another optional [`PySymmetry`]
/// for the magnetic group if requested.
///
/// # Errors
///
/// Returns an error if any intermediate step in the symmetry-group detection procedure fails.
#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (
    inp_xyz,
    inp_mol,
    out_sym,
    moi_thresholds,
    distance_thresholds,
    time_reversal,
    write_symmetry_elements=true,
    fictitious_magnetic_field=None,
    fictitious_electric_field=None,
))]
pub fn detect_symmetry_group(
    py: Python<'_>,
    inp_xyz: Option<PathBuf>,
    inp_mol: Option<PyMolecule>,
    out_sym: Option<PathBuf>,
    moi_thresholds: Vec<f64>,
    distance_thresholds: Vec<f64>,
    time_reversal: bool,
    write_symmetry_elements: bool,
    fictitious_magnetic_field: Option<[f64; 3]>,
    fictitious_electric_field: Option<[f64; 3]>,
) -> PyResult<(PySymmetry, Option<PySymmetry>)> {
    py.detach(|| {
        let params = SymmetryGroupDetectionParams::builder()
            .distance_thresholds(&distance_thresholds)
            .moi_thresholds(&moi_thresholds)
            .time_reversal(time_reversal)
            .fictitious_magnetic_fields(
                fictitious_magnetic_field
                    .map(|bs| vec![(Point3::<f64>::origin(), Vector3::new(bs[0], bs[1], bs[2]))]),
            )
            .fictitious_electric_fields(
                fictitious_electric_field
                    .map(|es| vec![(Point3::<f64>::origin(), Vector3::new(es[0], es[1], es[2]))]),
            )
            .field_origin_com(true)
            .write_symmetry_elements(write_symmetry_elements)
            .result_save_name(out_sym)
            .build()
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
        let inp_mol = inp_mol.map(Molecule::from);
        let mut pd_driver = SymmetryGroupDetectionDriver::builder()
            .parameters(&params)
            .xyz(inp_xyz)
            .molecule(inp_mol.as_ref())
            .build()
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
        pd_driver
            .run()
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
        let pyunitary_symmetry: PySymmetry = (&pd_driver
            .result()
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))?
            .unitary_symmetry)
            .try_into()
            .map_err(|err: anyhow::Error| PyRuntimeError::new_err(err.to_string()))?;
        let pymagnetic_symmetry: Option<PySymmetry> = pd_driver
            .result()
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))?
            .magnetic_symmetry
            .as_ref()
            .map(|magsym| {
                magsym
                    .try_into()
                    .map_err(|err: anyhow::Error| PyRuntimeError::new_err(err.to_string()))
            })
            .transpose()?;
        Ok((pyunitary_symmetry, pymagnetic_symmetry))
    })
}
