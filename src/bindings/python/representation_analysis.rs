use std::path::PathBuf;

use anyhow::{self, bail, ensure, format_err};
use log;
use ndarray::{Array1, Array2};
use num_complex::Complex;
use numpy::{PyArray1, PyArray2};
use pyo3::exceptions::{PyIOError, PyRuntimeError};
use pyo3::prelude::*;

use crate::angmom::spinor_rotation_3d::SpinConstraint;
use crate::angmom::ANGMOM_INDICES;
use crate::aux::ao_basis::{
    BasisAngularOrder, BasisAtom, BasisShell, CartOrder, PureOrder, ShellOrder,
};
use crate::aux::molecule::Molecule;
use crate::drivers::representation_analysis::angular_function::AngularFunctionRepAnalysisParams;
use crate::drivers::representation_analysis::slater_determinant::{
    SlaterDeterminantRepAnalysisDriver, SlaterDeterminantRepAnalysisParams,
};
use crate::drivers::representation_analysis::{CharacterTableDisplay, MagneticSymmetryKind};
use crate::drivers::symmetry_group_detection::SymmetryGroupDetectionResult;
use crate::drivers::QSym2Driver;
use crate::io::{read_qsym2_binary, QSym2FileType};
use crate::symmetry::symmetry_group::{
    MagneticRepresentedSymmetryGroup, UnitaryRepresentedSymmetryGroup,
};
use crate::symmetry::symmetry_transformation::SymmetryTransformationKind;
use crate::target::determinant::SlaterDeterminant;

type C128 = Complex<f64>;

/// A Python-exposed enumerated type to handle the `ShellOrder` union type `bool |
/// Optional[list[tuple[int, int, int]]]` in Python.
#[derive(FromPyObject)]
pub enum PyShellOrder {
    /// Variant for pure shell order. The associated boolean indicates if the functions are
    /// arranged in increasing-$`m`$ order.
    ///
    /// Python type: `bool`
    PureOrder(bool),

    /// Variant for Cartesian shell order. If the associated `Option` is `None`, the order will be
    /// taken to be lexicographic. Otherwise, the order will be as specified by the $`(x, y, z)`$
    /// exponent tuples.
    ///
    /// Python type: Optional[list[tuple[int, int, int]]]
    CartOrder(Option<Vec<(u32, u32, u32)>>),
}

/// A Python-exposed structure to marshal basis angular order information between Python and Rust.
///
/// # Constructor arguments
///
/// * `basis_atoms` - A vector of tuples, each of which provides information for one basis
/// atom in the form `(element, basis_shells)`. Here:
///   * `element` is a string giving the element symbol of the atom, and
///   * `basis_shells` is a vector of tuples, each of which provides information for one basis
///   shell on the atom in the form `(angmom, cart, order)`. Here:
///     * `angmom` is a symbol such as `"S"` or `"P"` for the angular momentum of the shell,
///     * `cart` is a boolean indicating if the functions in the shell are Cartesian (`true`)
///     or pure / solid harmonics (`false`), and
///     * `order` specifies how the functions in the shell are ordered:
///       * if `cart` is `true`, `order` can be `None` for lexicographic order, or a list of
///       tuples `(lx, ly, lz)` specifying a custom order for the Cartesian functions where
///       `lx`, `ly`, and `lz` are the $`x`$-, $`y`$-, and $`z`$-exponents;
///       * if `cart` is `false`, `order` can be `true` for increasing-$`m`$ or `false` for
///       decreasing-$`m`$ order.
///
///   Python type:
///   `list[tuple[str, list[tuple[str, bool, bool | Optional[list[tuple[int, int, int]]]]]]]`.
#[pyclass]
pub struct PyBasisAngularOrder {
    /// A vector of basis atoms. Each item in the vector is a tuple consisting of an atomic symbol
    /// and a vector of basis shell quartets whose components give:
    /// - the angular momentum symbol for the shell,
    /// - `true` if the shell is Cartesian, `false` if the shell is pure,
    /// - if the shell is Cartesian, then this has two possibilities:
    ///   - either `None` if the Cartesian functions are in lexicographic order,
    ///   - or `Some(vec![[lx, ly, lz], ...])` to specify a custom Cartesian order.
    /// - if the shell is pure, then this is a boolean `increasingm` to indicate if the pure
    /// functions in the shell are arranged in increasing-$`m`$ order.
    ///
    /// Python type: `list[tuple[str, list[tuple[str, bool, bool | Optional[list[tuple[int, int, int]]]]]]]`
    basis_atoms: Vec<(String, Vec<(String, bool, PyShellOrder)>)>,
}

#[pymethods]
impl PyBasisAngularOrder {
    /// Constructs a new `PyBasisAngularOrder` structure.
    ///
    /// # Arguments
    ///
    /// * `basis_atoms` - A vector of tuples, each of which provides information for one basis
    /// atom in the form `(element, basis_shells)`. Here:
    ///   * `element` is a string giving the element symbol of the atom, and
    ///   * `basis_shells` is a vector of tuples, each of which provides information for one basis
    ///   shell on the atom in the form `(angmom, cart, order)`. Here:
    ///     * `angmom` is a symbol such as `"S"` or `"P"` for the angular momentum of the shell,
    ///     * `cart` is a boolean indicating if the functions in the shell are Cartesian (`true`)
    ///     or pure / solid harmonics (`false`), and
    ///     * `order` specifies how the functions in the shell are ordered:
    ///       * if `cart` is `true`, `order` can be `None` for lexicographic order, or a list of
    ///       tuples `(lx, ly, lz)` specifying a custom order for the Cartesian functions where
    ///       `lx`, `ly`, and `lz` are the $`x`$-, $`y`$-, and $`z`$-exponents;
    ///       * if `cart` is `false`, `order` can be `true` for increasing-$`m`$ or `false` for
    ///       decreasing-$`m`$ order.
    ///
    ///   Python type:
    ///   `list[tuple[str, list[tuple[str, bool, bool | Optional[list[tuple[int, int, int]]]]]]]`.
    #[new]
    fn new(basis_atoms: Vec<(String, Vec<(String, bool, PyShellOrder)>)>) -> Self {
        Self { basis_atoms }
    }
}

impl PyBasisAngularOrder {
    /// Extracts the information in the [`PyBasisAngularOrder`] structure into `QSym2`'s native
    /// [`BasisAngularOrder`] structure.
    ///
    /// # Arguments
    ///
    /// * `mol` - The molecule with which the basis set information is associated.
    ///
    /// # Returns
    ///
    /// The [`BasisAngularOrder`] structure with the same information.
    ///
    /// # Errors
    ///
    /// Errors if the number of atoms or the atom elements in `mol` do not match the number of
    /// atoms and atom elements in `self`, or if incorrect shell order types are specified.
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
                                    if *increasingm {
                                        ShellOrder::Pure(PureOrder::increasingm(*l))
                                    } else {
                                        ShellOrder::Pure(PureOrder::decreasingm(*l))
                                    }
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

/// A Python-exposed enumerated type to marshall basis spin constraint information between Rust and
/// Python.
#[pyclass]
#[derive(Clone)]
pub enum PySpinConstraint {
    /// Variant for restricted spin constraint. Only two spin spaces are exposed.
    Restricted,

    /// Variant for unrestricted spin constraint. Only two spin spaces arranged in decreasing-$`m`$
    /// order (*i.e.* $`(\alpha, \beta)`$) are exposed.
    Unrestricted,

    /// Variant for generalised spin constraint. Only two spin spaces arranged in decreasing-$`m`$
    /// order (*i.e.* $`(\alpha, \beta)`$) are exposed.
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

/// A Python-exposed structure to marshall real Slater determinant information between Rust and
/// Python.
///
/// # Constructor arguments
///
/// * `spin_constraint` - The spin constraint applied to the coefficients of the determinant.
/// Python type: `PySpinConstraint`.
/// * `complex_symmetric` - A boolean indicating if inner products involving this determinant
/// are complex-symmetric. Python type: `bool`.
/// * `coefficients` - The real coefficients for the molecular orbitals of this determinant.
/// Python type: `list[numpy.2darray[float]]`.
/// * `occupations` - The occupation patterns for the molecular orbitals. Python type:
/// `list[numpy.1darray[float]]`.
/// * `threshold` - The threshold for comparisons. Python type: `float`.
/// * `mo_energies` - The optional real molecular orbital energies. Python type:
/// `Optional[list[numpy.1darray[float]]]`.
/// * `energy` - The optional real determinantal energy. Python type: `Optional[float]`.
#[pyclass]
#[derive(Clone)]
pub struct PySlaterDeterminantReal {
    /// The spin constraint applied to the coefficients of the determinant.
    ///
    /// Python type: `PySpinConstraint`
    spin_constraint: PySpinConstraint,

    /// A boolean indicating if inner products involving this determinant are complex-symmetric.
    ///
    /// Python type: `bool`
    complex_symmetric: bool,

    /// The real coefficien`bool`ts for the molecular orbitals of this determinant.
    ///
    /// Python type: `list[numpy.2darray[float]]`
    coefficients: Vec<Array2<f64>>,

    /// The occupation patterns for the molecular orbitals.
    ///
    /// Python type: `list[numpy.1darray[float]]`
    occupations: Vec<Array1<f64>>,

    /// The threshold for comparisons.
    ///
    /// Python type: `float`
    threshold: f64,

    /// The optional real molecular orbital energies.
    ///
    /// Python type: `Optional[list[numpy.1darray[float]]]`
    mo_energies: Option<Vec<Array1<f64>>>,

    /// The optional real determinantal energy.
    ///
    /// Python type: `Optional[float]`
    energy: Option<f64>,
}

#[pymethods]
impl PySlaterDeterminantReal {
    /// Constructs a real Python-exposed Slater determinant.
    ///
    /// # Arguments
    ///
    /// * `spin_constraint` - The spin constraint applied to the coefficients of the determinant.
    /// Python type: `PySpinConstraint`.
    /// * `complex_symmetric` - A boolean indicating if inner products involving this determinant
    /// are complex-symmetric. Python type: `bool`.
    /// * `coefficients` - The real coefficients for the molecular orbitals of this determinant.
    /// Python type: `list[numpy.2darray[float]]`.
    /// * `occupations` - The occupation patterns for the molecular orbitals. Python type:
    /// `list[numpy.1darray[float]]`.
    /// * `threshold` - The threshold for comparisons. Python type: `float`.
    /// * `mo_energies` - The optional real molecular orbital energies. Python type:
    /// `Optional[list[numpy.1darray[float]]]`.
    /// * `energy` - The optional real determinantal energy. Python type: `Optional[float]`.
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
    /// Extracts the information in the [`PySlaterDeterminantReal`] structure into `QSym2`'s native
    /// [`SlaterDeterminant`] structure.
    ///
    /// # Arguments
    ///
    /// * `bao` - The [`BasisAngularOrder`] for the basis set in which the Slater determinant is
    /// given.
    /// * `mol` - The molecule with which the Slater determinant is associated.
    ///
    /// # Returns
    ///
    /// The [`SlaterDeterminant`] structure with the same information.
    ///
    /// # Errors
    ///
    /// Errors if the [`SlaterDeterminant`] fails to build.
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

/// A Python-exposed structure to marshall complex Slater determinant information between Rust and
/// Python.
///
/// # Constructor arguments
///
/// * `spin_constraint` - The spin constraint applied to the coefficients of the determinant.
/// Python type: `PySpinConstraint`.
/// * `complex_symmetric` - A boolean indicating if inner products involving this determinant
/// are complex-symmetric. Python type: `bool`.
/// * `coefficients` - The complex coefficients for the molecular orbitals of this determinant.
/// Python type: `list[numpy.2darray[float]]`.
/// * `occupations` - The occupation patterns for the molecular orbitals. Python type:
/// `list[numpy.1darray[float]]`.
/// * `threshold` - The threshold for comparisons. Python type: `float`.
/// * `mo_energies` - The optional complex molecular orbital energies. Python type:
/// `Optional[list[numpy.1darray[complex]]]`.
/// * `energy` - The optional complex determinantal energy. Python type: `Optional[complex]`.
#[pyclass]
#[derive(Clone)]
pub struct PySlaterDeterminantComplex {
    /// The spin constraint applied to the coefficients of the determinant.
    ///
    /// Python type: `PySpinConstraint`
    spin_constraint: PySpinConstraint,

    /// A boolean indicating if inner products involving this determinant are complex-symmetric.
    ///
    /// Python type: `bool`
    complex_symmetric: bool,

    /// The complex coefficients for the molecular orbitals of this determinant.
    ///
    /// Python type: `list[numpy.2darray[complex]]`
    coefficients: Vec<Array2<C128>>,

    /// The occupation patterns for the molecular orbitals.
    ///
    /// Python type: `list[numpy.1darray[float]]`
    occupations: Vec<Array1<f64>>,

    /// The threshold for comparisons.
    ///
    /// Python type: `float`
    threshold: f64,

    /// The optional complex molecular orbital energies.
    ///
    /// Python type: `Optional[list[numpy.1darray[complex]]]`
    mo_energies: Option<Vec<Array1<C128>>>,

    /// The optional complex determinantal energy.
    ///
    /// Python type: `Optional[complex]`
    energy: Option<C128>,
}

#[pymethods]
impl PySlaterDeterminantComplex {
    /// Constructs a complex Python-exposed Slater determinant.
    ///
    /// # Arguments
    ///
    /// * `spin_constraint` - The spin constraint applied to the coefficients of the determinant.
    /// Python type: `PySpinConstraint`.
    /// * `complex_symmetric` - A boolean indicating if inner products involving this determinant
    /// are complex-symmetric. Python type: `bool`.
    /// * `coefficients` - The complex coefficients for the molecular orbitals of this determinant.
    /// Python type: `list[numpy.2darray[float]]`.
    /// * `occupations` - The occupation patterns for the molecular orbitals. Python type:
    /// `list[numpy.1darray[float]]`.
    /// * `threshold` - The threshold for comparisons. Python type: `float`.
    /// * `mo_energies` - The optional complex molecular orbital energies. Python type:
    /// `Optional[list[numpy.1darray[complex]]]`.
    /// * `energy` - The optional complex determinantal energy. Python type: `Optional[complex]`.
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
    /// Extracts the information in the [`PySlaterDeterminantComplex`] structure into `QSym2`'s native
    /// [`SlaterDeterminant`] structure.
    ///
    /// # Arguments
    ///
    /// * `bao` - The [`BasisAngularOrder`] for the basis set in which the Slater determinant is
    /// given.
    /// * `mol` - The molecule with which the Slater determinant is associated.
    ///
    /// # Returns
    ///
    /// The [`SlaterDeterminant`] structure with the same information.
    ///
    /// # Errors
    ///
    /// Errors if the [`SlaterDeterminant`] fails to build.
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

/// A Python-exposed enumerated type to handle the union type
/// `PySlaterDeterminantReal | PySlaterDeterminantComplex` in Python.
#[derive(FromPyObject)]
pub enum PySlaterDeterminant {
    /// Variant for real Python-exposed Slater determinant.
    Real(PySlaterDeterminantReal),

    /// Variant for complex Python-exposed Slater determinant.
    Complex(PySlaterDeterminantComplex),
}

/// A Python-exposed enumerated type to handle the union type of numpy float arrays and numpy
/// complex arrays in Python.
#[derive(FromPyObject)]
pub enum PySAO<'a> {
    Real(&'a PyArray2<f64>),
    Complex(&'a PyArray2<C128>),
}

/// A Python-exposed function to perform representation symmetry analysis for real and complex
/// Slater determinants and log the result via the `qsym2-output` logger at the `INFO` level.
///
/// If `symmetry_transformation_kind` includes spin transformation, the provided determinant will
/// be augmented to generalised spin constraint automatically.
///
/// # Arguments
///
/// * `inp_sym` - A path to the [`QSym2FileType::Sym`] file containing the symmetry-group detection
/// result for the system. This will be used to construct abstract groups and character tables for
/// representation analysis. Python type: `str`.
/// * `pydet` - A Python-exposed Slater determinant whose coefficients are of type `float64` or
/// `complex128`. Python type: `PySlaterDeterminantReal | PySlaterDeterminantComplex`
/// * `sao_spatial` - The atomic-orbital overlap matrix whose elements are of type `float64` or
/// `complex128`. Python type: `numpy.2darray[float] | numpy.2darray[complex]`.
/// * `integrality_threshold` - The threshold for verifying if subspace multiplicities are
/// integral. Python type: `float`.
/// * `linear_independence_threshold` - The threshold for determining the linear independence
/// subspace via the non-zero eigenvalues of the orbit overlap matrix. Python type: `float`.
/// * `use_magnetic_group` - A boolean indicating if any magnetic group present should be used for
/// representation analysis. Otherwise, the unitary group will be used. Python type: `bool`.
/// * `use_double_group` - A boolean indicating if the double group of the prevailing symmetry
/// group is to be used for representation analysis instead. Python type: `bool`.
/// * `use_corepresentation` - A boolean indicating if corepresentations of magnetic groups are to
/// be used for representation analysis instead of unitary representations. Python type: `bool`.
/// * `symmetry_transformation_kind` - An enumerated type indicating the type of symmetry
/// transformations to be performed on the origin determinant to generate the orbit. If this
/// contains spin transformation, the determinant will be augmented to generalised spin constraint
/// automatically. Python type: `SymmetryTransformationKind`.
/// * `analyse_mo_symmetries` - A boolean indicating if the symmetries of individual molecular
/// orbitals are to be analysed. Python type: `bool`.
/// * `write_overlap_eigenvalues` - A boolean indicating if the eigenvalues of the determinant
/// orbit overlap matrix are to be written to the output. Python type: `bool`.
/// * `write_character_table` - A boolean indicating if the character table of the prevailing
/// symmetry group is to be printed out. Python type: `bool`.
/// * `infinite_order_to_finite` - The finite order with which infinite-order generators are to be
/// interpreted to form a finite subgroup of the prevailing infinite group. This finite subgroup
/// will be used for the symmetrisation. Python type: `Optional[int]`.
/// * `angular_function_integrality_threshold` - The threshold for verifying if subspace
/// multiplicities are integral for the symmetry analysis of angular functions. Python type:
/// `float`.
/// * `angular_function_linear_independence_threshold` - The threshold for determining the linear
/// independence subspace via the non-zero eigenvalues of the orbit overlap matrix for the symmetry
/// analysis of angular functions. Python type: `float`.
/// * `angular_function_max_angular_momentum` - The maximum angular momentum order to be used in
/// angular function symmetry analysis. Python type: `int`.
#[pyfunction]
#[pyo3(signature = (inp_sym, pydet, pybao, sao_spatial, integrality_threshold, linear_independence_threshold, use_magnetic_group, use_double_group, symmetry_transformation_kind, analyse_mo_symmetries=true, write_overlap_eigenvalues=true, write_character_table=true, infinite_order_to_finite=None, angular_function_integrality_threshold=1e-7, angular_function_linear_independence_threshold=1e-7, angular_function_max_angular_momentum=2))]
pub fn rep_analyse_slater_determinant(
    inp_sym: PathBuf,
    pydet: PySlaterDeterminant,
    pybao: &PyBasisAngularOrder,
    sao_spatial: PySAO,
    integrality_threshold: f64,
    linear_independence_threshold: f64,
    use_magnetic_group: Option<MagneticSymmetryKind>,
    use_double_group: bool,
    symmetry_transformation_kind: SymmetryTransformationKind,
    analyse_mo_symmetries: bool,
    write_overlap_eigenvalues: bool,
    write_character_table: bool,
    infinite_order_to_finite: Option<u32>,
    angular_function_integrality_threshold: f64,
    angular_function_linear_independence_threshold: f64,
    angular_function_max_angular_momentum: u32,
) -> PyResult<()> {
    let pd_res: SymmetryGroupDetectionResult = read_qsym2_binary(inp_sym, QSym2FileType::Sym)
        .map_err(|err| PyIOError::new_err(err.to_string()))?;
    let mol = &pd_res.pre_symmetry.recentred_molecule;
    let bao = pybao
        .to_qsym2(mol)
        .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
    let augment_to_generalised = matches!(
        symmetry_transformation_kind,
        SymmetryTransformationKind::Spin | SymmetryTransformationKind::SpinSpatial
    );
    let afa_params = AngularFunctionRepAnalysisParams::builder()
        .integrality_threshold(angular_function_integrality_threshold)
        .linear_independence_threshold(angular_function_linear_independence_threshold)
        .max_angular_momentum(angular_function_max_angular_momentum)
        .build()
        .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
    match (&pydet, &sao_spatial) {
        (PySlaterDeterminant::Real(pydet_r), PySAO::Real(pysao_r)) => {
            let sao_spatial = pysao_r.to_owned_array();
            let det_r = if augment_to_generalised {
                pydet_r
                    .to_qsym2(&bao, mol)
                    .map_err(|err| PyRuntimeError::new_err(err.to_string()))?
                    .to_generalised()
            } else {
                pydet_r
                    .to_qsym2(&bao, mol)
                    .map_err(|err| PyRuntimeError::new_err(err.to_string()))?
            };
            let sda_params = SlaterDeterminantRepAnalysisParams::<f64>::builder()
                .integrality_threshold(integrality_threshold)
                .linear_independence_threshold(linear_independence_threshold)
                .use_magnetic_group(use_magnetic_group.clone())
                .use_double_group(use_double_group)
                .symmetry_transformation_kind(symmetry_transformation_kind)
                .analyse_mo_symmetries(analyse_mo_symmetries)
                .write_overlap_eigenvalues(write_overlap_eigenvalues)
                .write_character_table(if write_character_table {
                    Some(CharacterTableDisplay::Symbolic)
                } else {
                    None
                })
                .infinite_order_to_finite(infinite_order_to_finite)
                .build()
                .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
            match &use_magnetic_group {
                Some(MagneticSymmetryKind::Corepresentation) => {
                    let mut sda_driver = SlaterDeterminantRepAnalysisDriver::<
                        MagneticRepresentedSymmetryGroup,
                        f64,
                    >::builder()
                    .parameters(&sda_params)
                    .angular_function_parameters(&afa_params)
                    .determinant(&det_r)
                    .sao_spatial(&sao_spatial)
                    .symmetry_group(&pd_res)
                    .build()
                    .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
                    sda_driver
                        .run()
                        .map_err(|err| PyRuntimeError::new_err(err.to_string()))?
                }
                Some(MagneticSymmetryKind::Representation) | None => {
                    let mut sda_driver = SlaterDeterminantRepAnalysisDriver::<
                        UnitaryRepresentedSymmetryGroup,
                        f64,
                    >::builder()
                    .parameters(&sda_params)
                    .angular_function_parameters(&afa_params)
                    .determinant(&det_r)
                    .sao_spatial(&sao_spatial)
                    .symmetry_group(&pd_res)
                    .build()
                    .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
                    sda_driver
                        .run()
                        .map_err(|err| PyRuntimeError::new_err(err.to_string()))?
                }
            };
        }
        (PySlaterDeterminant::Real(pydet_r), PySAO::Complex(pysao_c)) => {
            let det_r = if augment_to_generalised {
                pydet_r
                    .to_qsym2(&bao, mol)
                    .map_err(|err| PyRuntimeError::new_err(err.to_string()))?
                    .to_generalised()
            } else {
                pydet_r
                    .to_qsym2(&bao, mol)
                    .map_err(|err| PyRuntimeError::new_err(err.to_string()))?
            };
            let det_c: SlaterDeterminant<C128> = det_r.into();
            let sao_spatial_c = pysao_c.to_owned_array();
            let sda_params = SlaterDeterminantRepAnalysisParams::<f64>::builder()
                .integrality_threshold(integrality_threshold)
                .linear_independence_threshold(linear_independence_threshold)
                .use_magnetic_group(use_magnetic_group.clone())
                .use_double_group(use_double_group)
                .symmetry_transformation_kind(symmetry_transformation_kind)
                .analyse_mo_symmetries(analyse_mo_symmetries)
                .write_overlap_eigenvalues(write_overlap_eigenvalues)
                .write_character_table(if write_character_table {
                    Some(CharacterTableDisplay::Symbolic)
                } else {
                    None
                })
                .infinite_order_to_finite(infinite_order_to_finite)
                .build()
                .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
            match &use_magnetic_group {
                Some(MagneticSymmetryKind::Corepresentation) => {
                    let mut sda_driver = SlaterDeterminantRepAnalysisDriver::<
                        MagneticRepresentedSymmetryGroup,
                        C128,
                    >::builder()
                    .parameters(&sda_params)
                    .angular_function_parameters(&afa_params)
                    .determinant(&det_c)
                    .sao_spatial(&sao_spatial_c)
                    .symmetry_group(&pd_res)
                    .build()
                    .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
                    sda_driver
                        .run()
                        .map_err(|err| PyRuntimeError::new_err(err.to_string()))?
                }
                Some(MagneticSymmetryKind::Representation) | None => {
                    let mut sda_driver = SlaterDeterminantRepAnalysisDriver::<
                        UnitaryRepresentedSymmetryGroup,
                        C128,
                    >::builder()
                    .parameters(&sda_params)
                    .angular_function_parameters(&afa_params)
                    .determinant(&det_c)
                    .sao_spatial(&sao_spatial_c)
                    .symmetry_group(&pd_res)
                    .build()
                    .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
                    sda_driver
                        .run()
                        .map_err(|err| PyRuntimeError::new_err(err.to_string()))?
                }
            };
        }
        (PySlaterDeterminant::Complex(pydet_c), _) => {
            let det_c = if augment_to_generalised {
                pydet_c
                    .to_qsym2(&bao, mol)
                    .map_err(|err| PyRuntimeError::new_err(err.to_string()))?
                    .to_generalised()
            } else {
                pydet_c
                    .to_qsym2(&bao, mol)
                    .map_err(|err| PyRuntimeError::new_err(err.to_string()))?
            };
            let sao_spatial_c = match sao_spatial {
                PySAO::Real(pysao_r) => pysao_r.to_owned_array().mapv(Complex::from),
                PySAO::Complex(pysao_c) => pysao_c.to_owned_array(),
            };
            let sda_params = SlaterDeterminantRepAnalysisParams::<f64>::builder()
                .integrality_threshold(integrality_threshold)
                .linear_independence_threshold(linear_independence_threshold)
                .use_magnetic_group(use_magnetic_group.clone())
                .use_double_group(use_double_group)
                .symmetry_transformation_kind(symmetry_transformation_kind)
                .analyse_mo_symmetries(analyse_mo_symmetries)
                .write_overlap_eigenvalues(write_overlap_eigenvalues)
                .write_character_table(if write_character_table {
                    Some(CharacterTableDisplay::Symbolic)
                } else {
                    None
                })
                .infinite_order_to_finite(infinite_order_to_finite)
                .build()
                .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
            match &use_magnetic_group {
                Some(MagneticSymmetryKind::Corepresentation) => {
                    let mut sda_driver = SlaterDeterminantRepAnalysisDriver::<
                        MagneticRepresentedSymmetryGroup,
                        C128,
                    >::builder()
                    .parameters(&sda_params)
                    .angular_function_parameters(&afa_params)
                    .determinant(&det_c)
                    .sao_spatial(&sao_spatial_c)
                    .symmetry_group(&pd_res)
                    .build()
                    .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
                    sda_driver
                        .run()
                        .map_err(|err| PyRuntimeError::new_err(err.to_string()))?
                }
                Some(MagneticSymmetryKind::Representation) | None => {
                    let mut sda_driver = SlaterDeterminantRepAnalysisDriver::<
                        UnitaryRepresentedSymmetryGroup,
                        C128,
                    >::builder()
                    .parameters(&sda_params)
                    .angular_function_parameters(&afa_params)
                    .determinant(&det_c)
                    .sao_spatial(&sao_spatial_c)
                    .symmetry_group(&pd_res)
                    .build()
                    .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
                    sda_driver
                        .run()
                        .map_err(|err| PyRuntimeError::new_err(err.to_string()))?
                }
            };
        }
    }
    Ok(())
}
