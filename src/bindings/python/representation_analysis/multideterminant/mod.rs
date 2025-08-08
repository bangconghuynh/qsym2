//! Python bindings for QSym² symmetry analysis of multi-determinants.

use std::collections::HashSet;
use std::fmt;
use std::hash::Hash;

use anyhow::format_err;
use ndarray::{Array1, Array2};
use num_complex::Complex;
use numpy::{PyArray1, PyArray2, PyArrayMethods, ToPyArray};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use crate::angmom::spinor_rotation_3d::StructureConstraint;
use crate::auxiliary::molecule::Molecule;
use crate::basis::ao::BasisAngularOrder;
use crate::bindings::python::integrals::PyStructureConstraint;
use crate::bindings::python::representation_analysis::slater_determinant::{
    PySlaterDeterminantComplex, PySlaterDeterminantReal,
};
use crate::target::determinant::SlaterDeterminant;
use crate::target::noci::basis::EagerBasis;
use crate::target::noci::multideterminant::MultiDeterminant;

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
/// Python-exposed structure to marshall real multi-determinant information between Rust and
/// Python.
///
/// # Constructor arguments
///
/// * `basis` - The basis of Slater determinants in which the multi-determinantal states are
/// expressed. Python type: `list[PySlaterDeterminantReal]`.
/// * `coefficients` - The coefficients for the multi-determinantal states in the specified basis.
/// Each column of the coefficient matrix contains the coefficients for one state.
/// Python type: `numpy.2darray[float]`.
/// * `energies` - The energies of the multi-determinantal states. Python type: `numpy.1darray[float]`.
/// * `threshold` - The threshold for comparisons. Python type: `float`.
#[pyclass]
#[derive(Clone)]
pub struct PyMultiDeterminantsReal {
    /// The basis of Slater determinants in which the multi-determinantal states are expressed.
    ///
    /// Python type: `list[PySlaterDeterminantReal]`.
    #[pyo3(get)]
    basis: Vec<PySlaterDeterminantReal>,

    /// The coefficients for the multi-determinantal states in the specified basis. Each column of
    /// the coefficient matrix contains the coefficients for one state.
    ///
    /// Python type: `numpy.2darray[float]`.
    coefficients: Array2<f64>,

    /// The energies of the multi-determinantal states.
    ///
    /// Python type: `numpy.1darray[float]`.
    energies: Array1<f64>,

    /// The threshold for comparisons.
    ///
    /// Python type: `float`.
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
    /// expressed. Python type: `list[PySlaterDeterminantReal]`.
    /// * `coefficients` - The coefficients for the multi-determinantal states in the specified basis.
    /// Each column of the coefficient matrix contains the coefficients for one state.
    /// Python type: `numpy.2darray[float]`.
    /// * `energies` - The energies of the multi-determinantal states. Python type: `numpy.1darray[float]`.
    /// * `threshold` - The threshold for comparisons. Python type: `float`.
    #[new]
    #[pyo3(signature = (basis, coefficients, energies, threshold))]
    pub fn new(
        basis: Vec<PySlaterDeterminantReal>,
        coefficients: Bound<'_, PyArray2<f64>>,
        energies: Bound<'_, PyArray1<f64>>,
        threshold: f64,
    ) -> Self {
        let multidet = Self {
            basis,
            coefficients: coefficients.to_owned_array(),
            energies: energies.to_owned_array(),
            threshold,
        };
        multidet
    }

    #[getter]
    pub fn coefficients<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(self.coefficients.to_pyarray(py))
    }

    #[getter]
    pub fn energies<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(self.energies.to_pyarray(py))
    }

    pub fn complex_symmetric<'py>(&self, py: Python<'py>) -> PyResult<bool> {
        let complex_symmetric_set = self
            .basis
            .iter()
            .map(|pydet| pydet.complex_symmetric(py))
            .collect::<Result<HashSet<_>, _>>()?;
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

    pub fn state_coefficients<'py>(
        &self,
        py: Python<'py>,
        state_index: usize,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(self.coefficients.column(state_index).to_pyarray(py))
    }

    pub fn state_energy<'py>(&self, _py: Python<'py>, state_index: usize) -> PyResult<f64> {
        Ok(self.energies[state_index])
    }
}

impl PyMultiDeterminantsReal {
    /// Extracts the information in the [`PyMultiDeterminantsReal`] structure into a vector of
    /// `QSym2`'s native [`MultiDeterminant`] structures.
    ///
    /// # Arguments
    ///
    /// * `bao` - The [`BasisAngularOrder`] for the basis set in which the Slater determinant is
    /// given.
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
    pub fn to_qsym2<'b, 'a: 'b, SC>(
        &'b self,
        bao: &'a BasisAngularOrder,
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
                    .map(|pydet| pydet.to_qsym2(bao, mol))
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
}

// ~~~~~~~
// Complex
// ~~~~~~~
//
/// Python-exposed structure to marshall complex multi-determinant information between Rust and
/// Python.
///
/// # Constructor arguments
///
/// * `basis` - The basis of Slater determinants in which the multi-determinantal states are
/// expressed. Python type: `list[PySlaterDeterminantComplex]`.
/// * `coefficients` - The coefficients for the multi-determinantal states in the specified basis.
/// Each column of the coefficient matrix contains the coefficients for one state.
/// Python type: `numpy.2darray[complex]`.
/// * `energies` - The energies of the multi-determinantal states. Python type:
/// `numpy.1darray[complex]`.
/// * `threshold` - The threshold for comparisons. Python type: `float`.
#[pyclass]
#[derive(Clone)]
pub struct PyMultiDeterminantsComplex {
    /// The basis of Slater determinants in which the multi-determinantal states are expressed.
    ///
    /// Python type: `list[PySlaterDeterminantReal]`.
    #[pyo3(get)]
    basis: Vec<PySlaterDeterminantComplex>,

    /// The coefficients for the multi-determinantal states in the specified basis. Each column of
    /// the coefficient matrix contains the coefficients for one state.
    ///
    /// Python type: `numpy.2darray[complex]`.
    coefficients: Array2<C128>,

    /// The energies of the multi-determinantal states.
    ///
    /// Python type: `numpy.1darray[complex]`.
    energies: Array1<C128>,

    /// The threshold for comparisons.
    ///
    /// Python type: `float`.
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
    /// expressed. Python type: `list[PySlaterDeterminantComplex]`.
    /// * `coefficients` - The coefficients for the multi-determinantal states in the specified basis.
    /// Each column of the coefficient matrix contains the coefficients for one state.
    /// Python type: `numpy.2darray[complex]`.
    /// * `energies` - The energies of the multi-determinantal states. Python type: `numpy.1darray[complex]`.
    /// * `threshold` - The threshold for comparisons. Python type: `float`.
    #[new]
    #[pyo3(signature = (basis, coefficients, energies, threshold))]
    pub fn new(
        basis: Vec<PySlaterDeterminantComplex>,
        coefficients: Bound<'_, PyArray2<C128>>,
        energies: Bound<'_, PyArray1<C128>>,
        threshold: f64,
    ) -> Self {
        let multidet = Self {
            basis,
            coefficients: coefficients.to_owned_array(),
            energies: energies.to_owned_array(),
            threshold,
        };
        multidet
    }

    #[getter]
    pub fn coefficients<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<C128>>> {
        Ok(self.coefficients.to_pyarray(py))
    }

    #[getter]
    pub fn energies<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<C128>>> {
        Ok(self.energies.to_pyarray(py))
    }

    pub fn complex_symmetric<'py>(&self, py: Python<'py>) -> PyResult<bool> {
        let complex_symmetric_set = self
            .basis
            .iter()
            .map(|pydet| pydet.complex_symmetric(py))
            .collect::<Result<HashSet<_>, _>>()?;
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

    pub fn state_coefficients<'py>(
        &self,
        py: Python<'py>,
        state_index: usize,
    ) -> PyResult<Bound<'py, PyArray1<C128>>> {
        Ok(self.coefficients.column(state_index).to_pyarray(py))
    }

    pub fn state_energy<'py>(&self, _py: Python<'py>, state_index: usize) -> PyResult<C128> {
        Ok(self.energies[state_index])
    }
}

impl PyMultiDeterminantsComplex {
    /// Extracts the information in the [`PyMultiDeterminantsComplex`] structure into `QSym2`'s native
    /// [`MultiDeterminant`] structure.
    ///
    /// # Arguments
    ///
    /// * `bao` - The [`BasisAngularOrder`] for the basis set in which the Slater determinant is
    /// given.
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
    pub fn to_qsym2<'b, 'a: 'b, SC>(
        &'b self,
        bao: &'a BasisAngularOrder,
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
                    .map(|pydet| pydet.to_qsym2(bao, mol))
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
