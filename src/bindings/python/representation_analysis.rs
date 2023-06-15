use anyhow::{self, bail, ensure, format_err};
use log;
use ndarray::{Array1, Array2};
use num_complex::Complex;
use numpy::{PyArray1, PyArray2};
use pyo3::exceptions::{PyIOError, PyRuntimeError};
use pyo3::prelude::*;

use crate::angmom::spinor_rotation_3d::SpinConstraint;
use crate::angmom::ANGMOM_INDICES;
use crate::aux::ao_basis::{BasisAngularOrder, BasisAtom, BasisShell, CartOrder, ShellOrder};
use crate::aux::molecule::Molecule;
use crate::drivers::representation_analysis::slater_determinant::{
    SlaterDeterminantRepAnalysisDriver, SlaterDeterminantRepAnalysisParams,
};
use crate::drivers::representation_analysis::CharacterTableDisplay;
use crate::drivers::symmetry_group_detection::SymmetryGroupDetectionResult;
use crate::drivers::QSym2Driver;
use crate::io::{read_qsym2, QSym2FileType};
use crate::symmetry::symmetry_group::{
    MagneticRepresentedSymmetryGroup, UnitaryRepresentedSymmetryGroup,
};
use crate::symmetry::symmetry_transformation::SymmetryTransformationKind;
use crate::target::determinant::SlaterDeterminant;

type C128 = Complex<f64>;

#[derive(FromPyObject)]
pub enum PyShellOrder {
    PureOrder(bool),

    CartOrder(Option<Vec<(u32, u32, u32)>>),
}

/// A Python-exposed structure to marshal basis angular order information between Python and Rust.
#[pyclass]
pub struct PyBasisAngularOrder {
    /// A vector of basis atoms. Each item in the vector is a tuple consisting of an atomic symbol
    /// and a vector of basis shell quartets whose components give:
    /// - the angular momentum symbol for the shell,
    /// - `true` if the shell is Cartesian, `false` if the shell is pure,
    /// - (this will be ignored if the shell is pure) `None` if the Cartesian functions are in
    /// lexicographic order, `Some(vec![[lx, ly, lz], ...])` to specify a custom Cartesian order.
    /// - (this will be ignored if the shell is Cartesian) `Some(increasingm)` to indicate the
    /// order of pure functions in the shell,
    basis_atoms: Vec<(String, Vec<(String, bool, PyShellOrder)>)>,
}

#[pymethods]
impl PyBasisAngularOrder {
    #[new]
    fn new(basis_atoms: Vec<(String, Vec<(String, bool, PyShellOrder)>)>) -> Self {
        Self { basis_atoms }
    }
}

impl PyBasisAngularOrder {
    fn to_qsym2<'b, 'a: 'b>(
        &'b self,
        mol: &'a Molecule,
    ) -> Result<BasisAngularOrder, anyhow::Error> {
        ensure!(
            self.basis_atoms.len() == mol.atoms.len(),
            "The number of basis atoms does not match the number of ordinary atoms."
        );
        let basis_atoms = self
            .basis_atoms
            .iter()
            .zip(mol.atoms.iter())
            .flat_map(|((element, basis_shells), atom)| {
                ensure!(
                    *element == atom.atomic_symbol,
                    "Expected element `{element}`, but found atom `{}`.",
                    atom.atomic_symbol
                );
                let bss = basis_shells
                    .iter()
                    .flat_map(|(angmom, cart, shell_order)| {
                        let l = ANGMOM_INDICES.get(angmom).unwrap();
                        let shl_ord = if *cart {
                            let cart_order = match shell_order {
                                PyShellOrder::CartOrder(cart_tuples_opt) => {
                                    if let Some(cart_tuples) = cart_tuples_opt {
                                        CartOrder::new(cart_tuples)?
                                    } else {
                                        CartOrder::lex(*l)
                                    }
                                },
                                PyShellOrder::PureOrder(_) => {
                                    log::error!("Cartesian shell order expected, but specification for pure shell order found.");
                                    bail!("Cartesian shell order expected, but specification for pure shell order found.")
                                }
                            };
                            ShellOrder::Cart(cart_order)
                        } else {
                            match shell_order {
                                PyShellOrder::PureOrder(increasingm) => {
                                    ShellOrder::Pure(*increasingm)
                                },
                                PyShellOrder::CartOrder(_) => {
                                    log::error!("Pure shell order expected, but specification for Cartesian shell order found.");
                                    bail!("Pure shell order expected, but specification for Cartesian shell order found.")
                                }
                            }
                        };
                        Ok::<_, anyhow::Error>(BasisShell::new(*l, shl_ord))
                    })
                    .collect::<Vec<_>>();
                Ok(BasisAtom::new(atom, &bss))
            })
            .collect::<Vec<_>>();
        Ok(BasisAngularOrder::new(&basis_atoms))
    }
}

#[pyclass]
#[derive(Clone)]
pub enum PySpinConstraint {
    Restricted,
    Unrestricted,
    Generalised,
}

impl From<PySpinConstraint> for SpinConstraint {
    fn from(pysc: PySpinConstraint) -> Self {
        match pysc {
            PySpinConstraint::Restricted => SpinConstraint::Restricted(2),
            PySpinConstraint::Unrestricted => SpinConstraint::Unrestricted(2, false),
            PySpinConstraint::Generalised => SpinConstraint::Generalised(2, false),
        }
    }
}

#[pyclass]
#[derive(Clone)]
pub struct PySlaterDeterminantReal {
    spin_constraint: PySpinConstraint,

    complex_symmetric: bool,

    coefficients: Vec<Array2<f64>>,

    occupations: Vec<Array1<f64>>,

    threshold: f64,

    mo_energies: Option<Vec<Array1<f64>>>,

    energy: Option<f64>,
}

#[pymethods]
impl PySlaterDeterminantReal {
    #[new]
    fn new(
        spin_constraint: PySpinConstraint,
        complex_symmetric: bool,
        coefficients: Vec<&PyArray2<f64>>,
        occupations: Vec<&PyArray1<f64>>,
        threshold: f64,
        mo_energies: Option<Vec<&PyArray1<f64>>>,
        energy: Option<f64>,
    ) -> Self {
        let det = Self {
            spin_constraint,
            complex_symmetric,
            coefficients: coefficients
                .iter()
                .map(|pyarr| pyarr.to_owned_array())
                .collect::<Vec<_>>(),
            occupations: occupations
                .iter()
                .map(|pyarr| pyarr.to_owned_array())
                .collect::<Vec<_>>(),
            threshold,
            mo_energies: mo_energies.map(|energies| {
                energies
                    .iter()
                    .map(|pyarr| pyarr.to_owned_array())
                    .collect::<Vec<_>>()
            }),
            energy,
        };
        det
    }
}

impl PySlaterDeterminantReal {
    fn to_qsym2<'b, 'a: 'b>(
        &'b self,
        bao: &'a BasisAngularOrder,
        mol: &'a Molecule,
    ) -> Result<SlaterDeterminant<f64>, anyhow::Error> {
        let det = SlaterDeterminant::<f64>::builder()
            .spin_constraint(self.spin_constraint.clone().into())
            .bao(bao)
            .complex_symmetric(self.complex_symmetric)
            .mol(mol)
            .coefficients(&self.coefficients)
            .occupations(&self.occupations)
            .mo_energies(self.mo_energies.clone())
            .energy(
                self.energy
                    .ok_or_else(|| "No determinantal energy set.".to_string()),
            )
            .threshold(self.threshold)
            .build()
            .map_err(|err| format_err!(err));
        det
    }
}

#[pyclass]
#[derive(Clone)]
pub struct PySlaterDeterminantComplex {
    spin_constraint: PySpinConstraint,

    complex_symmetric: bool,

    coefficients: Vec<Array2<C128>>,

    occupations: Vec<Array1<f64>>,

    threshold: f64,

    mo_energies: Option<Vec<Array1<C128>>>,

    energy: Option<C128>,
}

#[pymethods]
impl PySlaterDeterminantComplex {
    #[new]
    fn new(
        spin_constraint: PySpinConstraint,
        complex_symmetric: bool,
        coefficients: Vec<&PyArray2<C128>>,
        occupations: Vec<&PyArray1<f64>>,
        threshold: f64,
        mo_energies: Option<Vec<&PyArray1<C128>>>,
        energy: Option<C128>,
    ) -> Self {
        let det = Self {
            spin_constraint,
            complex_symmetric,
            coefficients: coefficients
                .iter()
                .map(|pyarr| pyarr.to_owned_array())
                .collect::<Vec<_>>(),
            occupations: occupations
                .iter()
                .map(|pyarr| pyarr.to_owned_array())
                .collect::<Vec<_>>(),
            threshold,
            mo_energies: mo_energies.map(|energies| {
                energies
                    .iter()
                    .map(|pyarr| pyarr.to_owned_array())
                    .collect::<Vec<_>>()
            }),
            energy,
        };
        det
    }
}

impl PySlaterDeterminantComplex {
    fn to_qsym2<'b, 'a: 'b>(
        &'b self,
        bao: &'a BasisAngularOrder,
        mol: &'a Molecule,
    ) -> Result<SlaterDeterminant<C128>, anyhow::Error> {
        let det = SlaterDeterminant::<C128>::builder()
            .spin_constraint(self.spin_constraint.clone().into())
            .bao(bao)
            .complex_symmetric(self.complex_symmetric)
            .mol(mol)
            .coefficients(&self.coefficients)
            .occupations(&self.occupations)
            .mo_energies(self.mo_energies.clone())
            .energy(
                self.energy
                    .ok_or_else(|| "No determinantal energy set.".to_string()),
            )
            .threshold(self.threshold)
            .build()
            .map_err(|err| format_err!(err));
        det
    }
}

#[derive(FromPyObject)]
pub enum PySlaterDeterminant {
    Real(PySlaterDeterminantReal),
    Complex(PySlaterDeterminantComplex),
}

#[derive(FromPyObject)]
pub enum PySAO<'a> {
    Real(&'a PyArray2<f64>),
    Complex(&'a PyArray2<C128>),
}

/// A Python-exposed function to perform representation symmetry analysis for Slater determinants.
#[pyfunction]
#[pyo3(signature = (inp_sym, pydet, pybao, pysao_spatial, integrality_threshold, linear_independence_threshold, use_magnetic_group, use_double_group, use_corepresentation, symmetry_transformation_kind, analyse_mo_symmetries=true, write_overlap_eigenvalues=true, write_character_table=true))]
pub(super) fn rep_analyse_slater_determinant(
    inp_sym: String,
    pydet: PySlaterDeterminant,
    pybao: &PyBasisAngularOrder,
    pysao_spatial: PySAO,
    integrality_threshold: f64,
    linear_independence_threshold: f64,
    use_magnetic_group: bool,
    use_double_group: bool,
    use_corepresentation: bool,
    symmetry_transformation_kind: SymmetryTransformationKind,
    analyse_mo_symmetries: bool,
    write_overlap_eigenvalues: bool,
    write_character_table: bool,
) -> PyResult<()> {
    let pd_res: SymmetryGroupDetectionResult = read_qsym2(&inp_sym, QSym2FileType::Sym)
        .map_err(|err| PyIOError::new_err(err.to_string()))?;
    let mol = &pd_res.pre_symmetry.recentred_molecule;
    let bao = pybao
        .to_qsym2(mol)
        .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
    match pydet {
        PySlaterDeterminant::Real(pydet_r) => {
            let sao_spatial = match pysao_spatial {
                PySAO::Real(pysao_r) => Ok(pysao_r.to_owned_array()),
                PySAO::Complex(_) => Err(PyRuntimeError::new_err(
                    "Real spatial AO overlap matrix expected.",
                )),
            }?;
            let det_r = pydet_r
                .to_qsym2(&bao, mol)
                .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
            let sda_params = SlaterDeterminantRepAnalysisParams::<f64>::builder()
                .integrality_threshold(integrality_threshold)
                .linear_independence_threshold(linear_independence_threshold)
                .use_magnetic_group(use_magnetic_group)
                .use_double_group(use_double_group)
                .symmetry_transformation_kind(symmetry_transformation_kind)
                .analyse_mo_symmetries(analyse_mo_symmetries)
                .write_overlap_eigenvalues(write_overlap_eigenvalues)
                .write_character_table(if write_character_table {
                    Some(CharacterTableDisplay::Symbolic)
                } else {
                    None
                })
                .build()
                .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
            if use_magnetic_group && use_corepresentation {
                let mut sda_driver = SlaterDeterminantRepAnalysisDriver::<
                    MagneticRepresentedSymmetryGroup,
                    f64,
                >::builder()
                .parameters(&sda_params)
                .determinant(&det_r)
                .sao_spatial(&sao_spatial)
                .symmetry_group(&pd_res)
                .build()
                .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
                sda_driver
                    .run()
                    .map_err(|err| PyRuntimeError::new_err(err.to_string()))?
            } else {
                let mut sda_driver = SlaterDeterminantRepAnalysisDriver::<
                    UnitaryRepresentedSymmetryGroup,
                    f64,
                >::builder()
                .parameters(&sda_params)
                .determinant(&det_r)
                .sao_spatial(&sao_spatial)
                .symmetry_group(&pd_res)
                .build()
                .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
                sda_driver
                    .run()
                    .map_err(|err| PyRuntimeError::new_err(err.to_string()))?
            };
        }
        PySlaterDeterminant::Complex(pydet_c) => {
            let sao_spatial_c = match pysao_spatial {
                PySAO::Real(pysao_r) => pysao_r.to_owned_array().mapv(Complex::from),
                PySAO::Complex(pysao_c) => pysao_c.to_owned_array(),
            };
            let det_c = pydet_c
                .to_qsym2(&bao, mol)
                .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
            let sda_params = SlaterDeterminantRepAnalysisParams::<C128>::builder()
                .integrality_threshold(integrality_threshold)
                .linear_independence_threshold(linear_independence_threshold)
                .use_magnetic_group(use_magnetic_group)
                .use_double_group(use_double_group)
                .symmetry_transformation_kind(symmetry_transformation_kind)
                .analyse_mo_symmetries(analyse_mo_symmetries)
                .write_overlap_eigenvalues(write_overlap_eigenvalues)
                .write_character_table(if write_character_table {
                    Some(CharacterTableDisplay::Symbolic)
                } else {
                    None
                })
                .build()
                .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
            if use_magnetic_group && use_corepresentation {
                let mut sda_driver = SlaterDeterminantRepAnalysisDriver::<
                    MagneticRepresentedSymmetryGroup,
                    C128,
                >::builder()
                .parameters(&sda_params)
                .determinant(&det_c)
                .sao_spatial(&sao_spatial_c)
                .symmetry_group(&pd_res)
                .build()
                .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
                sda_driver
                    .run()
                    .map_err(|err| PyRuntimeError::new_err(err.to_string()))?
            } else {
                let mut sda_driver = SlaterDeterminantRepAnalysisDriver::<
                    UnitaryRepresentedSymmetryGroup,
                    C128,
                >::builder()
                .parameters(&sda_params)
                .determinant(&det_c)
                .sao_spatial(&sao_spatial_c)
                .symmetry_group(&pd_res)
                .build()
                .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
                sda_driver
                    .run()
                    .map_err(|err| PyRuntimeError::new_err(err.to_string()))?
            };
        }
    }
    Ok(())
}
