use nalgebra::{Point3, Vector3};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use crate::aux::atom::{Atom, ElementMap};
use crate::aux::molecule::Molecule;

use crate::drivers::symmetry_group_detection::{
    SymmetryGroupDetectionDriver, SymmetryGroupDetectionParams,
};
use crate::drivers::QSym2Driver;

/// A Python-exposed class to marshall molecular structure information between Rust and Python.
#[pyclass]
#[derive(Clone)]
pub struct PyMolecule {
    /// The ordinary atoms in the molecule.
    #[pyo3(get)]
    atoms: Vec<(String, [f64; 3])>,

    /// An optional uniform external magnetic field.
    #[pyo3(get)]
    magnetic_field: Option<[f64; 3]>,

    /// An optional uniform external electric field.
    #[pyo3(get)]
    electric_field: Option<[f64; 3]>,

    /// Threshold for comparing molecules.
    #[pyo3(get)]
    threshold: f64,
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
        mol.set_magnetic_field(
            pymol
                .magnetic_field
                .map(Vector3::from_iterator),
        );
        mol.set_electric_field(
            pymol
                .electric_field
                .map(Vector3::from_iterator),
        );
        mol
    }
}

/// A Python-exposed function to perform symmetry-group detection.
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
/// # Errors
///
/// Returns an error if any intermediate step in the symmetry-group detection procedure fails.
#[pyfunction]
#[pyo3(signature = (inp_xyz, inp_mol, out_sym, moi_thresholds, distance_thresholds, time_reversal, write_symmetry_elements=true, fictitious_magnetic_field=None, fictitious_electric_field=None))]
pub(super) fn detect_symmetry_group(
    inp_xyz: Option<String>,
    inp_mol: Option<PyMolecule>,
    out_sym: Option<String>,
    moi_thresholds: Vec<f64>,
    distance_thresholds: Vec<f64>,
    time_reversal: bool,
    write_symmetry_elements: bool,
    fictitious_magnetic_field: Option<[f64; 3]>,
    fictitious_electric_field: Option<[f64; 3]>,
) -> PyResult<()> {
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
    Ok(())
}
