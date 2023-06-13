use nalgebra::{Point3, Vector3};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use crate::aux::atom::{Atom, AtomKind, ElementMap};
use crate::aux::molecule::Molecule;
use crate::drivers::symmetry_group_detection::{
    SymmetryGroupDetectionDriver, SymmetryGroupDetectionParams,
};
use crate::drivers::QSym2Driver;

#[pyclass]
#[derive(Clone)]
pub struct PyMolecule {
    atoms: Vec<(String, [f64; 3])>,
    magnetic_atoms: Option<Vec<(bool, [f64; 3])>>,
    electric_atoms: Option<Vec<(bool, [f64; 3])>>,
    threshold: f64,
}

#[pymethods]
impl PyMolecule {
    #[new]
    fn new(
        atoms: Vec<(String, [f64; 3])>,
        threshold: f64,
        magnetic_atoms: Option<Vec<(bool, [f64; 3])>>,
        electric_atoms: Option<Vec<(bool, [f64; 3])>>,
    ) -> Self {
        Self {
            atoms,
            threshold,
            magnetic_atoms,
            electric_atoms,
        }
    }
}

impl From<PyMolecule> for Molecule {
    fn from(pymol: PyMolecule) -> Self {
        let emap = ElementMap::new();
        Self::from_atoms(
            &pymol
                .atoms
                .iter()
                .map(|(ele, r)| {
                    Atom::new_ordinary(ele, Point3::new(r[0], r[1], r[2]), &emap, pymol.threshold)
                })
                .chain(pymol.magnetic_atoms.iter().flatten().flat_map(|(pos, r)| {
                    Atom::new_special(
                        AtomKind::Magnetic(*pos),
                        Point3::new(r[0], r[1], r[2]),
                        pymol.threshold,
                    )
                }))
                .chain(pymol.electric_atoms.iter().flatten().flat_map(|(pos, r)| {
                    Atom::new_special(
                        AtomKind::Electric(*pos),
                        Point3::new(r[0], r[1], r[2]),
                        pymol.threshold,
                    )
                }))
                .collect::<Vec<_>>(),
            pymol.threshold,
        )
    }
}

/// A Python-exposed function to perform symmetry-group detection.
#[pyfunction]
#[pyo3(signature = (inp_xyz, inp_mol, out_sym, moi_thresholds, distance_thresholds, time_reversal, write_symmetry_elements, magnetic_field, electric_field))]
pub(super) fn detect_symmetry_group(
    inp_xyz: Option<String>,
    inp_mol: Option<PyMolecule>,
    out_sym: Option<String>,
    moi_thresholds: Vec<f64>,
    distance_thresholds: Vec<f64>,
    time_reversal: bool,
    write_symmetry_elements: bool,
    magnetic_field: Option<[f64; 3]>,
    electric_field: Option<[f64; 3]>,
) -> PyResult<()> {
    let params = SymmetryGroupDetectionParams::builder()
        .distance_thresholds(&distance_thresholds)
        .moi_thresholds(&moi_thresholds)
        .time_reversal(time_reversal)
        .magnetic_fields(
            magnetic_field
                .map(|bs| vec![(Point3::<f64>::origin(), Vector3::new(bs[0], bs[1], bs[2]))]),
        )
        .electric_fields(
            electric_field
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
