//! Python bindings for QSym² symmetry analysis of multi-determinants.

use std::collections::HashSet;
use std::fmt;
use std::hash::Hash;
use std::path::PathBuf;

use anyhow::format_err;
use itertools::Itertools;
use ndarray::{Array1, Array2};
use num_complex::Complex;
use numpy::{PyArray1, PyArray2, PyArrayMethods, ToPyArray};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

use crate::angmom::spinor_rotation_3d::StructureConstraint;
use crate::auxiliary::molecule::Molecule;
use crate::basis::ao::BasisAngularOrder;
use crate::bindings::python::integrals::PyStructureConstraint;
use crate::bindings::python::representation_analysis::slater_determinant::{
    PySlaterDeterminantComplex, PySlaterDeterminantReal,
};
use crate::io::format::qsym2_output;
use crate::io::{QSym2FileType, read_qsym2_binary, write_qsym2_binary};
use crate::target::determinant::SlaterDeterminant;
use crate::target::noci::basis::EagerBasis;
use crate::target::noci::multideterminant::MultiDeterminant;
use crate::target::noci::multideterminants::MultiDeterminants;

type C128 = Complex<f64>;

// ==================
// Struct definitions
// ==================

// -----------------
// Multi-determinant
// -----------------

// ~~~~
// Real
// ~~~~
//
/// Python-exposed structure to marshall real multi-determinant information between Rust and Python.
#[pyclass]
#[derive(Clone, Serialize, Deserialize)]
pub struct PyMultiDeterminantsReal {
    /// The basis of Slater determinants in which the multi-determinantal states are expressed.
    #[pyo3(get)]
    basis: Vec<PySlaterDeterminantReal>,

    /// The coefficients for the multi-determinantal states in the specified basis. Each column of
    /// the coefficient matrix contains the coefficients for one state.
    coefficients: Array2<f64>,

    /// The energies of the multi-determinantal states.
    energies: Array1<f64>,

    /// The density matrices for the multi-determinantal states in the specified basis.
    density_matrices: Option<Vec<Array2<f64>>>,

    /// The threshold for comparisons.
    #[pyo3(get)]
    threshold: f64,
}

#[pymethods]
impl PyMultiDeterminantsReal {
    /// Constructs a set of real Python-exposed multi-determinants.
    ///
    /// # Arguments
    ///
    /// * `basis` - The basis of Slater determinants in which the multi-determinantal states are
    /// expressed.
    /// * `coefficients` - The coefficients for the multi-determinantal states in the specified basis.
    /// Each column of the coefficient matrix contains the coefficients for one state.
    /// * `energies` - The energies of the multi-determinantal states.
    /// * `density_matrices` - The optional density matrices of the multi-determinantal states.
    /// * `threshold` - The threshold for comparisons.
    #[new]
    #[pyo3(signature = (basis, coefficients, energies, density_matrices, threshold))]
    pub fn new(
        basis: Vec<PySlaterDeterminantReal>,
        coefficients: Bound<'_, PyArray2<f64>>,
        energies: Bound<'_, PyArray1<f64>>,
        density_matrices: Option<Vec<Bound<'_, PyArray2<f64>>>>,
        threshold: f64,
    ) -> Self {
        let coefficients = coefficients.to_owned_array();
        let energies = energies.to_owned_array();
        let density_matrices = density_matrices.map(|denmats| {
            denmats
                .into_iter()
                .map(|denmat| denmat.to_owned_array())
                .collect_vec()
        });
        if let Some(ref denmats) = density_matrices {
            if denmats.len() != coefficients.ncols()
                || denmats.len() != energies.len()
                || coefficients.ncols() != energies.len()
            {
                panic!(
                    "Inconsistent numbers of multi-determinantal states in `coefficients`, `energies`, and `density_matrices`."
                )
            }
        };
        Self {
            basis,
            coefficients,
            energies,
            density_matrices,
            threshold,
        }
    }

    #[getter]
    pub fn coefficients<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(self.coefficients.to_pyarray(py))
    }

    #[getter]
    pub fn energies<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(self.energies.to_pyarray(py))
    }

    #[getter]
    pub fn density_matrices<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Option<Vec<Bound<'py, PyArray2<f64>>>>> {
        Ok(self.density_matrices.as_ref().map(|denmats| {
            denmats
                .iter()
                .map(|denmat| denmat.to_pyarray(py))
                .collect_vec()
        }))
    }

    /// Boolean indicating whether inner products involving these multi-determinantal states are
    /// complex-symmetric.
    pub fn complex_symmetric<'py>(&self, _py: Python<'py>) -> PyResult<bool> {
        let complex_symmetric_set = self
            .basis
            .iter()
            .map(|pydet| pydet.complex_symmetric)
            .collect::<HashSet<_>>();
        if complex_symmetric_set.len() != 1 {
            Err(PyRuntimeError::new_err(
                "Inconsistent complex-symmetric flags across basis functions.",
            ))
        } else {
            complex_symmetric_set.into_iter().next().ok_or_else(|| {
                PyRuntimeError::new_err("Unable to extract the complex-symmetric flag.")
            })
        }
    }

    /// Returns the coefficients for a particular state.
    pub fn state_coefficients<'py>(
        &self,
        py: Python<'py>,
        state_index: usize,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(self.coefficients.column(state_index).to_pyarray(py))
    }

    /// Returns the energy for a particular state.
    pub fn state_energy<'py>(&self, _py: Python<'py>, state_index: usize) -> PyResult<f64> {
        Ok(self.energies[state_index])
    }

    /// Returns the density matrix for a particular state.
    pub fn state_density_matrix<'py>(
        &self,
        py: Python<'py>,
        state_index: usize,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.density_matrices
            .as_ref()
            .ok_or_else(|| {
                PyRuntimeError::new_err(
                    "No multi-determinantal density matrices found.".to_string(),
                )
            })
            .map(|denmats| denmats[state_index].to_pyarray(py))
    }

    /// Saves the real Python-exposed multi-determinants as a binary file with `.qsym2.pymdet`
    /// extension.
    ///
    /// # Arguments
    ///
    /// * `name` - The name of the binary file to be saved without the `.qsym2.pymdet` extension.
    ///
    /// # Returns
    ///
    /// A return code indicating if the serialisation process has been successful.
    pub fn to_qsym2_binary<'py>(&self, _py: Python<'py>, name: PathBuf) -> PyResult<usize> {
        let mut path = name.to_path_buf();
        path.set_extension(QSym2FileType::Pymdet.ext());
        qsym2_output!(
            "Real Python-exposed multi-determinants saved as {}.",
            path.display().to_string()
        );
        write_qsym2_binary(name, QSym2FileType::Pymdet, self)
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))
    }

    /// Reads the real Python-exposed multi-determinants from a binary file with `.qsym2.pymdet`
    /// extension.
    ///
    /// # Arguments
    ///
    /// * `name` - The name of the binary file to be read from without the `.qsym2.pymdet` extension.
    ///
    /// # Returns
    ///
    /// A real Python-exposed multi-determinants structure.
    #[staticmethod]
    pub fn from_qsym2_binary(name: PathBuf) -> PyResult<Self> {
        let mut path = name.to_path_buf();
        path.set_extension(QSym2FileType::Pymdet.ext());
        qsym2_output!(
            "Real Python-exposed multi-determinants read in from {}.",
            path.display().to_string()
        );
        read_qsym2_binary(name, QSym2FileType::Pymdet)
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))
    }
}

impl PyMultiDeterminantsReal {
    /// Extracts the information in the [`PyMultiDeterminantsReal`] structure into a vector of
    /// `QSym2`'s native [`MultiDeterminant`] structures.
    ///
    /// # Arguments
    ///
    /// * `baos` - The [`BasisAngularOrder`]s for the basis set in which the Slater determinant is
    /// given, one for each explicit component per coefficient matrix.
    /// * `mol` - The molecule with which the Slater determinant is associated.
    ///
    /// # Returns
    ///
    /// A vector of [`MultiDeterminant`] structures, one for each multi-determinantal state
    /// contained in the Python version.
    ///
    /// # Errors
    ///
    /// Errors if the [`MultiDeterminant`] structures fail to build.
    pub fn to_qsym2_individuals<'b, 'a: 'b, SC>(
        &'b self,
        baos: &[&'a BasisAngularOrder],
        mol: &'a Molecule,
    ) -> Result<
        Vec<MultiDeterminant<'b, f64, EagerBasis<SlaterDeterminant<'b, f64, SC>>, SC>>,
        anyhow::Error,
    >
    where
        SC: StructureConstraint
            + Eq
            + Hash
            + Clone
            + fmt::Display
            + TryFrom<PyStructureConstraint, Error = anyhow::Error>,
    {
        let eager_basis = EagerBasis::builder()
            .elements(
                self.basis
                    .iter()
                    .map(|pydet| pydet.to_qsym2(baos, mol))
                    .collect::<Result<Vec<_>, _>>()?,
            )
            .build()?;
        let multidets = self
            .energies
            .iter()
            .zip(self.coefficients.columns())
            .map(|(e, c)| {
                MultiDeterminant::builder()
                    .basis(eager_basis.clone())
                    .coefficients(c.to_owned())
                    .energy(Ok(*e))
                    .threshold(self.threshold)
                    .build()
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(|err| format_err!(err));
        multidets
    }

    /// Extracts the information in the [`PyMultiDeterminantsReal`] structure into a `QSym2`'s
    /// native [`MultiDeterminants`] structure.
    ///
    /// # Arguments
    ///
    /// * `baos` - The [`BasisAngularOrder`]s for the basis set in which the Slater determinant is
    /// given, one for each explicit component per coefficient matrix.
    /// * `mol` - The molecule with which the Slater determinant is associated.
    ///
    /// # Returns
    ///
    /// A [`MultiDeterminants`] structure.
    ///
    /// # Errors
    ///
    /// Errors if the [`MultiDeterminants`] structure fails to build.
    pub fn to_qsym2_collection<'b, 'a: 'b, SC>(
        &'b self,
        baos: &[&'a BasisAngularOrder],
        mol: &'a Molecule,
    ) -> Result<
        MultiDeterminants<'b, f64, EagerBasis<SlaterDeterminant<'b, f64, SC>>, SC>,
        anyhow::Error,
    >
    where
        SC: StructureConstraint
            + Eq
            + Hash
            + Clone
            + fmt::Display
            + TryFrom<PyStructureConstraint, Error = anyhow::Error>,
    {
        let eager_basis = EagerBasis::builder()
            .elements(
                self.basis
                    .iter()
                    .map(|pydet| pydet.to_qsym2(baos, mol))
                    .collect::<Result<Vec<_>, _>>()?,
            )
            .build()?;
        MultiDeterminants::builder()
            .basis(eager_basis)
            .coefficients(self.coefficients.clone())
            .energies(Ok(self.energies.clone()))
            .threshold(self.threshold)
            .build()
            .map_err(|err| format_err!(err))
    }
}

// ~~~~~~~
// Complex
// ~~~~~~~
//
/// Python-exposed structure to marshall complex multi-determinant information between Rust and
/// Python.
#[pyclass]
#[derive(Clone, Serialize, Deserialize)]
pub struct PyMultiDeterminantsComplex {
    /// The basis of Slater determinants in which the multi-determinantal states are expressed.
    #[pyo3(get)]
    basis: Vec<PySlaterDeterminantComplex>,

    /// The coefficients for the multi-determinantal states in the specified basis. Each column of
    /// the coefficient matrix contains the coefficients for one state.
    coefficients: Array2<C128>,

    /// The energies of the multi-determinantal states.
    energies: Array1<C128>,

    /// The density matrices for the multi-determinantal states in the specified basis.
    density_matrices: Option<Vec<Array2<C128>>>,

    /// The threshold for comparisons.
    #[pyo3(get)]
    threshold: f64,
}

#[pymethods]
impl PyMultiDeterminantsComplex {
    /// Constructs a set of complex Python-exposed multi-determinants.
    ///
    /// # Arguments
    ///
    /// * `basis` - The basis of Slater determinants in which the multi-determinantal states are
    /// expressed.
    /// * `coefficients` - The coefficients for the multi-determinantal states in the specified basis.
    /// Each column of the coefficient matrix contains the coefficients for one state.
    /// * `energies` - The energies of the multi-determinantal states.
    /// * `density_matrices` - The optional density matrices of the multi-determinantal states.
    /// * `threshold` - The threshold for comparisons.
    #[new]
    #[pyo3(signature = (basis, coefficients, energies, density_matrices, threshold))]
    pub fn new(
        basis: Vec<PySlaterDeterminantComplex>,
        coefficients: Bound<'_, PyArray2<C128>>,
        energies: Bound<'_, PyArray1<C128>>,
        density_matrices: Option<Vec<Bound<'_, PyArray2<C128>>>>,
        threshold: f64,
    ) -> Self {
        let coefficients = coefficients.to_owned_array();
        let energies = energies.to_owned_array();
        let density_matrices = density_matrices.map(|denmats| {
            denmats
                .into_iter()
                .map(|denmat| denmat.to_owned_array())
                .collect_vec()
        });
        if let Some(ref denmats) = density_matrices {
            if denmats.len() != coefficients.ncols()
                || denmats.len() != energies.len()
                || coefficients.ncols() != energies.len()
            {
                panic!(
                    "Inconsistent numbers of multi-determinantal states in `coefficients`, `energies`, and `density_matrices`."
                )
            }
        };
        Self {
            basis,
            coefficients,
            energies,
            density_matrices,
            threshold,
        }
    }

    #[getter]
    pub fn coefficients<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<C128>>> {
        Ok(self.coefficients.to_pyarray(py))
    }

    #[getter]
    pub fn energies<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<C128>>> {
        Ok(self.energies.to_pyarray(py))
    }

    #[getter]
    pub fn density_matrices<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Option<Vec<Bound<'py, PyArray2<C128>>>>> {
        Ok(self.density_matrices.as_ref().map(|denmats| {
            denmats
                .iter()
                .map(|denmat| denmat.to_pyarray(py))
                .collect_vec()
        }))
    }

    /// Boolean indicating whether inner products involving these multi-determinantal states are
    /// complex-symmetric.
    pub fn complex_symmetric<'py>(&self, _py: Python<'py>) -> PyResult<bool> {
        let complex_symmetric_set = self
            .basis
            .iter()
            .map(|pydet| pydet.complex_symmetric)
            .collect::<HashSet<_>>();
        if complex_symmetric_set.len() != 1 {
            Err(PyRuntimeError::new_err(
                "Inconsistent complex-symmetric flags across basis functions.",
            ))
        } else {
            complex_symmetric_set.into_iter().next().ok_or_else(|| {
                PyRuntimeError::new_err("Unable to extract the complex-symmetric flag.")
            })
        }
    }

    /// Returns the coefficients for a particular state.
    pub fn state_coefficients<'py>(
        &self,
        py: Python<'py>,
        state_index: usize,
    ) -> PyResult<Bound<'py, PyArray1<C128>>> {
        Ok(self.coefficients.column(state_index).to_pyarray(py))
    }

    /// Returns the energy for a particular state.
    pub fn state_energy<'py>(&self, _py: Python<'py>, state_index: usize) -> PyResult<C128> {
        Ok(self.energies[state_index])
    }

    /// Returns the density matrix for a particular state.
    pub fn state_density_matrix<'py>(
        &self,
        py: Python<'py>,
        state_index: usize,
    ) -> PyResult<Bound<'py, PyArray2<C128>>> {
        self.density_matrices
            .as_ref()
            .ok_or_else(|| {
                PyRuntimeError::new_err(
                    "No multi-determinantal density matrices found.".to_string(),
                )
            })
            .map(|denmats| denmats[state_index].to_pyarray(py))
    }

    /// Saves the complex Python-exposed multi-determinants as a binary file with `.qsym2.pymdet`
    /// extension.
    ///
    /// # Arguments
    ///
    /// * `name` - The name of the binary file to be saved without the `.qsym2.pymdet` extension.
    ///
    /// # Returns
    ///
    /// A return code indicating if the serialisation process has been successful.
    pub fn to_qsym2_binary<'py>(&self, _py: Python<'py>, name: PathBuf) -> PyResult<usize> {
        let mut path = name.to_path_buf();
        path.set_extension(QSym2FileType::Pymdet.ext());
        qsym2_output!(
            "Complex Python-exposed multi-determinants saved as {}.",
            path.display().to_string()
        );
        write_qsym2_binary(name, QSym2FileType::Pymdet, self)
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))
    }

    /// Reads the complex Python-exposed multi-determinants from a binary file with `.qsym2.pymdet`
    /// extension.
    ///
    /// # Arguments
    ///
    /// * `name` - The name of the binary file to be read from without the `.qsym2.pymdet` extension.
    ///
    /// # Returns
    ///
    /// A complex Python-exposed multi-determinants structure.
    #[staticmethod]
    pub fn from_qsym2_binary(name: PathBuf) -> PyResult<Self> {
        let mut path = name.to_path_buf();
        path.set_extension(QSym2FileType::Pymdet.ext());
        qsym2_output!(
            "Complex Python-exposed multi-determinants read in from {}.",
            path.display().to_string()
        );
        read_qsym2_binary(name, QSym2FileType::Pymdet)
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))
    }
}

impl PyMultiDeterminantsComplex {
    /// Extracts the information in the [`PyMultiDeterminantsComplex`] structure into `QSym2`'s native
    /// [`MultiDeterminant`] structure.
    ///
    /// # Arguments
    ///
    /// * `baos` - The [`BasisAngularOrder`]s for the basis set in which the Slater determinant is
    /// given, one for each explicit component per coefficient matrix.
    /// * `mol` - The molecule with which the Slater determinant is associated.
    ///
    /// # Returns
    ///
    /// The A vector of [`MultiDeterminant`] structures, one for each multi-determinantal state
    /// contained in the Python version.
    ///
    /// # Errors
    ///
    /// Errors if the [`MultiDeterminant`] structures fail to build.
    pub fn to_qsym2_individuals<'b, 'a: 'b, SC>(
        &'b self,
        baos: &[&'a BasisAngularOrder],
        mol: &'a Molecule,
    ) -> Result<
        Vec<MultiDeterminant<'b, C128, EagerBasis<SlaterDeterminant<'b, C128, SC>>, SC>>,
        anyhow::Error,
    >
    where
        SC: StructureConstraint
            + Eq
            + Hash
            + Clone
            + fmt::Display
            + TryFrom<PyStructureConstraint, Error = anyhow::Error>,
    {
        let eager_basis = EagerBasis::builder()
            .elements(
                self.basis
                    .iter()
                    .map(|pydet| pydet.to_qsym2(baos, mol))
                    .collect::<Result<Vec<_>, _>>()?,
            )
            .build()?;
        let multidets = self
            .energies
            .iter()
            .zip(self.coefficients.columns())
            .map(|(e, c)| {
                MultiDeterminant::builder()
                    .basis(eager_basis.clone())
                    .coefficients(c.to_owned())
                    .energy(Ok(*e))
                    .threshold(self.threshold)
                    .build()
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(|err| format_err!(err));
        multidets
    }

    /// Extracts the information in the [`PyMultiDeterminantsComplex`] structure into a `QSym2`'s
    /// native [`MultiDeterminants`] structure.
    ///
    /// # Arguments
    ///
    /// * `baos` - The [`BasisAngularOrder`]s for the basis set in which the Slater determinant is
    /// given, one for each explicit component per coefficient matrix.
    /// * `mol` - The molecule with which the Slater determinant is associated.
    ///
    /// # Returns
    ///
    /// A [`MultiDeterminants`] structure.
    ///
    /// # Errors
    ///
    /// Errors if the [`MultiDeterminants`] structure fails to build.
    pub fn to_qsym2_collection<'b, 'a: 'b, SC>(
        &'b self,
        baos: &[&'a BasisAngularOrder],
        mol: &'a Molecule,
    ) -> Result<
        MultiDeterminants<'b, C128, EagerBasis<SlaterDeterminant<'b, C128, SC>>, SC>,
        anyhow::Error,
    >
    where
        SC: StructureConstraint
            + Eq
            + Hash
            + Clone
            + fmt::Display
            + TryFrom<PyStructureConstraint, Error = anyhow::Error>,
    {
        let eager_basis = EagerBasis::builder()
            .elements(
                self.basis
                    .iter()
                    .map(|pydet| pydet.to_qsym2(baos, mol))
                    .collect::<Result<Vec<_>, _>>()?,
            )
            .build()?;
        MultiDeterminants::builder()
            .basis(eager_basis)
            .coefficients(self.coefficients.clone())
            .energies(Ok(self.energies.clone()))
            .threshold(self.threshold)
            .build()
            .map_err(|err| format_err!(err))
    }
}

// ================
// Enum definitions
// ================

/// Python-exposed enumerated type to handle the union type
/// `PyMultiDeterminantsReal | PyMultiDeterminantsComplex` in Python.
#[derive(FromPyObject)]
pub enum PyMultiDeterminants {
    /// Variant for real Python-exposed multi-determinants.
    Real(PyMultiDeterminantsReal),

    /// Variant for complex Python-exposed multi-determinants.
    Complex(PyMultiDeterminantsComplex),
}

// =====================
// Functions definitions
// =====================

mod multideterminant_eager_basis;
mod multideterminant_orbit_basis_external_solver;
mod multideterminant_orbit_basis_internal_solver;

pub use multideterminant_eager_basis::rep_analyse_multideterminants_eager_basis;
pub use multideterminant_orbit_basis_external_solver::rep_analyse_multideterminants_orbit_basis_external_solver;
pub use multideterminant_orbit_basis_internal_solver::rep_analyse_multideterminants_orbit_basis_internal_solver;
