//! Python bindings for QSym² atomic-orbital integral evaluations.

use anyhow::{self, bail, ensure, format_err};
use lazy_static::lazy_static;
#[cfg(feature = "integrals")]
use nalgebra::{Point3, Vector3};
#[cfg(feature = "integrals")]
use num_complex::Complex;
#[cfg(feature = "integrals")]
use numpy::{IntoPyArray, PyArray2, PyArray4};
use periodic_table;
#[cfg(feature = "integrals")]
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyType;
#[cfg(feature = "qchem")]
use regex::Regex;
use serde::{Deserialize, Serialize};

use crate::angmom::spinor_rotation_3d::{SpinConstraint, SpinOrbitCoupled};
use crate::auxiliary::molecule::Molecule;
use crate::basis::ao::{
    BasisAngularOrder, BasisAtom, BasisShell, CartOrder, PureOrder, ShellOrder, SpinorOrder,
};
#[cfg(feature = "integrals")]
use crate::basis::ao_integrals::{BasisSet, BasisShellContraction, GaussianContraction};
#[cfg(feature = "integrals")]
use crate::integrals::shell_tuple::build_shell_tuple_collection;
#[cfg(feature = "qchem")]
use crate::io::format::{log_title, qsym2_output, QSym2Output};

#[cfg(feature = "qchem")]
lazy_static! {
    static ref SP_PATH_RE: Regex =
        Regex::new(r"(.*sp)\\energy_function$").expect("Regex pattern invalid.");
}

/// Python-exposed enumerated type to handle the union type `(bool, bool) | (list[int], bool)` in
/// Python for specifying pure-spherical-harmonic order or spinor order.
#[derive(Clone, FromPyObject)]
pub enum PyPureSpinorOrder {
    /// Variant for standard pure or spinor shell order. The first associated boolean indicates if
    /// the functions are arranged in increasing-$`m`$ order, and the second associated boolean
    /// indicates if the shell is even with respect to spatial inversion.
    Standard((bool, bool)),

    /// Variant for custom pure or spinor shell order. The associated vector contains a sequence of
    /// integers specifying the order of $`m`$ values for pure or $`2m`$ values for spinor in the
    /// shell, and the associated boolean indicates if the shell is even with respect to spatial
    /// inversion.
    Custom((Vec<i32>, bool)),
}

/// Python-exposed enumerated type to handle the `ShellOrder` union type `bool |
/// Optional[list[tuple[int, int, int]]]` in Python.
#[derive(Clone, FromPyObject)]
pub enum PyShellOrder {
    /// Variant for pure or spinor shell order. The associated value is either a boolean indicating
    /// if the functions are arranged in increasing-$`m`$ order, or a sequence of integers specifying
    /// a custom $`m`$-order for pure or $`2m`$-order for spinor.
    ///
    /// Python type: `bool | list[int]`.
    PureSpinorOrder(PyPureSpinorOrder),

    /// Variant for Cartesian shell order. If the associated `Option` is `None`, the order will be
    /// taken to be lexicographic. Otherwise, the order will be as specified by the $`(x, y, z)`$
    /// exponent tuples.
    ///
    /// Python type: Optional[list[tuple[int, int, int]]].
    CartOrder(Option<Vec<(u32, u32, u32)>>),
}

// /// Enumerated type indicating the type of magnetic symmetry to be used for representation
// /// analysis.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[pyclass(eq, eq_int)]
pub enum ShellType {
    /// Variant indicating that unitary representations should be used for magnetic symmetry
    /// analysis.
    Pure,

    /// Variant indicating that magnetic corepresentations should be used for magnetic symmetry
    /// analysis.
    Spinor,

    Cartesian,
}

/// Python-exposed structure to marshal basis angular order information between Python and Rust.
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
///       `lx`, `ly`, and `lz` are the $`x`$-, $`y`$-, and $`z`$-exponents, respectively;
///       * if `cart` is `false`, `order` can be `true` for increasing-$`m`$ order, `false` for
///       decreasing-$`m`$ order, or a list of $`m`$ values for custom order.
///
///   Python type:
///   `list[tuple[str, list[tuple[str, bool, Optional[list[tuple[int, int, int]]] | bool | list[int]]]]]`.
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
    /// functions in the shell are arranged in increasing-$`m`$ order, or a list of $`m`$ values
    /// specifying a custom $`m`$ order.
    ///
    /// Python type: `list[tuple[str, list[tuple[str, bool, Optional[list[tuple[int, int, int]]] | bool | list[int]]]]]`.
    basis_atoms: Vec<(String, Vec<(u32, ShellType, PyShellOrder)>)>,
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
    ///       `lx`, `ly`, and `lz` are the $`x`$-, $`y`$-, and $`z`$-exponents, respectively;
    ///       * if `cart` is `false`, `order` can be `true` for increasing-$`m`$ order, `false` for
    ///       decreasing-$`m`$ order, or a list of $`m`$ values for custom order.
    ///
    ///   Python type:
    ///   `list[tuple[str, list[tuple[str, bool, bool | Optional[list[tuple[int, int, int]]]]]]]`.
    #[new]
    fn new(basis_atoms: Vec<(String, Vec<(u32, ShellType, PyShellOrder)>)>) -> Self {
        Self { basis_atoms }
    }

    /// Extracts basis angular order information from a Q-Chem HDF5 archive file.
    ///
    /// # Arguments
    ///
    /// * `filename` - A path to a Q-Chem HDF5 archive file. Python type: `str`.
    ///
    /// # Returns
    ///
    /// A sequence of `PyBasisAngularOrder` objects, one for each Q-Chem calculation found in the
    /// HDF5 archive file. Python type: `list[PyBasisAngularOrder]`.
    ///
    /// A summary showing how the `PyBasisAngularOrder` objects map onto the Q-Chem calculations in
    /// the HDF5 archive file is also logged at the `INFO` level.
    #[cfg(feature = "qchem")]
    #[classmethod]
    fn from_qchem_archive(_cls: &Bound<'_, PyType>, filename: &str) -> PyResult<Vec<Self>> {
        use hdf5;
        use indexmap::IndexMap;
        use num::ToPrimitive;

        let f = hdf5::File::open(filename).map_err(|err| PyValueError::new_err(err.to_string()))?;
        let mut sp_paths = f
            .group(".counters")
            .map_err(|err| PyValueError::new_err(err.to_string()))?
            .member_names()
            .map_err(|err| PyValueError::new_err(err.to_string()))?
            .iter()
            .filter_map(|path| {
                if SP_PATH_RE.is_match(path) {
                    let path = path.replace("\\", "/");
                    Some(path.replace("/energy_function", ""))
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        sp_paths.sort_by(|path_a, path_b| numeric_sort::cmp(path_a, path_b));

        let elements = periodic_table::periodic_table();

        log_title(&format!(
            "Basis angular order extraction from Q-Chem HDF5 archive files",
        ));
        let pybaos = sp_paths
            .iter()
            .map(|sp_path| {
                let sp_group = f
                    .group(sp_path)
                    .map_err(|err| PyValueError::new_err(err.to_string()))?;
                let shell_types = sp_group
                    .dataset("aobasis/shell_types")
                    .map_err(|err| PyValueError::new_err(err.to_string()))?
                    .read_1d::<i32>()
                    .map_err(|err| PyValueError::new_err(err.to_string()))?;
                let shell_to_atom_map = sp_group
                    .dataset("aobasis/shell_to_atom_map")
                    .map_err(|err| PyValueError::new_err(err.to_string()))?
                    .read_1d::<usize>()
                    .map_err(|err| PyValueError::new_err(err.to_string()))?
                    .iter()
                    .zip(shell_types.iter())
                    .flat_map(|(&idx, shell_type)| {
                        if *shell_type == -1 {
                            vec![idx, idx]
                        } else {
                            vec![idx]
                        }
                    })
                    .collect::<Vec<_>>();
                let nuclei = sp_group
                    .dataset("structure/nuclei")
                    .map_err(|err| PyValueError::new_err(err.to_string()))?
                    .read_1d::<usize>()
                    .map_err(|err| PyValueError::new_err(err.to_string()))?;

                let mut basis_atoms_map: IndexMap<usize, Vec<(u32, ShellType, PyShellOrder)>> =
                    IndexMap::new();
                shell_types.iter().zip(shell_to_atom_map.iter()).for_each(
                    |(shell_type, atom_idx)| {
                        if *shell_type == 0 {
                            // S shell
                            basis_atoms_map.entry(*atom_idx).or_insert(vec![]).push((
                                0,
                                ShellType::Cartesian,
                                PyShellOrder::CartOrder(Some(CartOrder::qchem(0).cart_tuples)),
                            ));
                        } else if *shell_type == 1 {
                            // P shell
                            basis_atoms_map.entry(*atom_idx).or_insert(vec![]).push((
                                1,
                                ShellType::Cartesian,
                                PyShellOrder::CartOrder(Some(CartOrder::qchem(1).cart_tuples)),
                            ));
                        } else if *shell_type == -1 {
                            // SP shell
                            basis_atoms_map
                                .entry(*atom_idx)
                                .or_insert(vec![])
                                .extend_from_slice(&[
                                    (
                                        0,
                                        ShellType::Cartesian,
                                        PyShellOrder::CartOrder(Some(
                                            CartOrder::qchem(0).cart_tuples,
                                        )),
                                    ),
                                    (
                                        1,
                                        ShellType::Cartesian,
                                        PyShellOrder::CartOrder(Some(
                                            CartOrder::qchem(1).cart_tuples,
                                        )),
                                    ),
                                ]);
                        } else if *shell_type < 0 {
                            // Cartesian D shell or higher
                            let l = shell_type.unsigned_abs();
                            // let l_usize = l
                            //     .to_usize()
                            //     .unwrap_or_else(|| panic!("Unable to convert the angular momentum value `|{shell_type}|` to `usize`."));
                            basis_atoms_map.entry(*atom_idx).or_insert(vec![]).push((
                                l,
                                ShellType::Cartesian,
                                PyShellOrder::CartOrder(Some(CartOrder::qchem(l).cart_tuples)),
                            ));
                        } else {
                            // Pure D shell or higher
                            let l = shell_type.unsigned_abs();
                            // let l_usize = l
                            //     .to_usize()
                            //     .unwrap_or_else(|| panic!("Unable to convert the angular momentum value `|{shell_type}|` to `usize`."));
                            basis_atoms_map.entry(*atom_idx).or_insert(vec![]).push((
                                l,
                                ShellType::Pure,
                                PyShellOrder::PureSpinorOrder(PyPureSpinorOrder::Standard((
                                    true,
                                    l % 2 == 0,
                                ))),
                            ));
                        }
                    },
                );
                let pybao = basis_atoms_map
                    .into_iter()
                    .map(|(atom_idx, v)| {
                        let element = elements
                            .get(nuclei[atom_idx])
                            .map(|el| el.symbol.to_string())
                            .ok_or_else(|| {
                                PyValueError::new_err(format!(
                                    "Unable to identify an element for atom index `{atom_idx}`."
                                ))
                            })?;
                        Ok((element, v))
                    })
                    .collect::<Result<Vec<_>, _>>()
                    .map(|basis_atoms| Self::new(basis_atoms));
                pybao
            })
            .collect::<Result<Vec<_>, _>>();

        let idx_width = sp_paths.len().ilog10().to_usize().unwrap_or(4).max(4) + 1;
        let sp_path_width = sp_paths
            .iter()
            .map(|sp_path| sp_path.chars().count())
            .max()
            .unwrap_or(10)
            .max(10);
        let table_width = idx_width + sp_path_width + 4;
        qsym2_output!("");
        "Each single-point calculation has associated with it a `PyBasisAngularOrder` object.\n\
        The table below shows the `PyBasisAngularOrder` index in the generated list and the\n\
        corresponding single-point calculation."
            .log_output_display();
        qsym2_output!("{}", "┈".repeat(table_width));
        qsym2_output!(" {:<idx_width$}  {:<}", "Index", "Q-Chem job");
        qsym2_output!("{}", "┈".repeat(table_width));
        sp_paths.iter().enumerate().for_each(|(i, sp_path)| {
            qsym2_output!(" {:<idx_width$}  {:<}", i, sp_path);
        });
        qsym2_output!("{}", "┈".repeat(table_width));
        qsym2_output!("");

        pybaos
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
    pub fn to_qsym2<'b, 'a: 'b>(
        &'b self,
        mol: &'a Molecule,
    ) -> Result<BasisAngularOrder<'b>, anyhow::Error> {
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
                        create_basis_shell(*angmom, cart, shell_order)
                    })
                    .collect::<Vec<_>>();
                Ok(BasisAtom::new(atom, &bss))
            })
            .collect::<Vec<_>>();
        Ok(BasisAngularOrder::new(&basis_atoms))
    }
}

/// Python-exposed enumerated type to marshall basis spin constraint information between Rust and
/// Python.
#[pyclass(eq, eq_int)]
#[derive(Clone, PartialEq, Eq, Hash)]
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

impl TryFrom<SpinConstraint> for PySpinConstraint {
    type Error = anyhow::Error;

    fn try_from(sc: SpinConstraint) -> Result<Self, Self::Error> {
        match sc {
            SpinConstraint::Restricted(2) => Ok(PySpinConstraint::Restricted),
            SpinConstraint::Unrestricted(2, false) => Ok(PySpinConstraint::Unrestricted),
            SpinConstraint::Generalised(2, false) => Ok(PySpinConstraint::Generalised),
            _ => Err(format_err!(
                "`PySpinConstraint` can only support two spin spaces."
            )),
        }
    }
}

/// Python-exposed enumerated type to marshall basis spin--orbit-coupled layout in the coupled
/// treatment of spin and spatial degrees of freedom between Rust and Python.
#[pyclass(eq, eq_int)]
#[derive(Clone, PartialEq, Eq, Hash)]
pub enum PySpinOrbitCoupled {
    /// Variant for $`j`$-adapted basis functions. Only two relativistic components are exposed.
    JAdapted,
}

impl From<PySpinOrbitCoupled> for SpinOrbitCoupled {
    fn from(pysoc: PySpinOrbitCoupled) -> Self {
        match pysoc {
            PySpinOrbitCoupled::JAdapted => SpinOrbitCoupled::JAdapted(2),
        }
    }
}

impl TryFrom<SpinOrbitCoupled> for PySpinOrbitCoupled {
    type Error = anyhow::Error;

    fn try_from(soc: SpinOrbitCoupled) -> Result<Self, Self::Error> {
        match soc {
            SpinOrbitCoupled::JAdapted(2) => Ok(PySpinOrbitCoupled::JAdapted),
            _ => Err(format_err!(
                "`PySpinOrbitCoupled` can only support two relativistic components."
            )),
        }
    }
}

/// Python-exposed enumerated type to handle the union type `PySpinConstraint | PySpinOrbitCoupled`
/// in Python.
#[derive(FromPyObject, Clone, PartialEq, Eq, Hash)]
pub enum PyStructureConstraint {
    /// Variant for Python-exposed spin constraint layout.
    SpinConstraint(PySpinConstraint),

    /// Variant for Python-exposed spin--orbit-coupled layout.
    SpinOrbitCoupled(PySpinOrbitCoupled),
}

impl TryFrom<SpinConstraint> for PyStructureConstraint {
    type Error = anyhow::Error;

    fn try_from(sc: SpinConstraint) -> Result<Self, Self::Error> {
        match sc {
            SpinConstraint::Restricted(2) => Ok(PyStructureConstraint::SpinConstraint(
                PySpinConstraint::Restricted,
            )),
            SpinConstraint::Unrestricted(2, false) => Ok(PyStructureConstraint::SpinConstraint(
                PySpinConstraint::Unrestricted,
            )),
            SpinConstraint::Generalised(2, false) => Ok(PyStructureConstraint::SpinConstraint(
                PySpinConstraint::Generalised,
            )),
            _ => Err(format_err!(
                "`PySpinConstraint` can only support two spin spaces."
            )),
        }
    }
}

impl TryFrom<PyStructureConstraint> for SpinConstraint {
    type Error = anyhow::Error;

    fn try_from(py_sc: PyStructureConstraint) -> Result<Self, Self::Error> {
        match py_sc {
            PyStructureConstraint::SpinConstraint(py_sc) => Ok(py_sc.into()),
            PyStructureConstraint::SpinOrbitCoupled(_) => Err(format_err!(
                "`SpinConstraint` cannot be created from `PySpinOrbitCoupled`."
            )),
        }
    }
}

impl TryFrom<SpinOrbitCoupled> for PyStructureConstraint {
    type Error = anyhow::Error;

    fn try_from(soc: SpinOrbitCoupled) -> Result<Self, Self::Error> {
        match soc {
            SpinOrbitCoupled::JAdapted(2) => Ok(PyStructureConstraint::SpinOrbitCoupled(
                PySpinOrbitCoupled::JAdapted,
            )),
            _ => Err(format_err!(
                "`PySpinOrbitCoupled` can only support two relativistic components."
            )),
        }
    }
}

impl TryFrom<PyStructureConstraint> for SpinOrbitCoupled {
    type Error = anyhow::Error;

    fn try_from(py_sc: PyStructureConstraint) -> Result<Self, Self::Error> {
        match py_sc {
            PyStructureConstraint::SpinOrbitCoupled(py_soc) => Ok(py_soc.into()),
            PyStructureConstraint::SpinConstraint(_) => Err(format_err!(
                "`SpinOrbitCoupled` cannot be created from `PySpinConstraint`."
            )),
        }
    }
}

#[cfg(feature = "integrals")]
#[pyclass]
#[derive(Clone)]
/// Python-exposed structure to marshall basis shell contraction information between Rust and
/// Python.
///
/// # Constructor arguments
///
/// * `basis_shell` - A triplet of the form `(angmom, cart, order)` where:
///     * `angmom` is a symbol such as `"S"` or `"P"` for the angular momentum of the shell,
///     * `cart` is a boolean indicating if the functions in the shell are Cartesian (`true`)
///     or pure / solid harmonics (`false`), and
///     * `order` specifies how the functions in the shell are ordered:
///       * if `cart` is `true`, `order` can be `None` for lexicographic order, or a list of
///       tuples `(lx, ly, lz)` specifying a custom order for the Cartesian functions where
///       `lx`, `ly`, and `lz` are the $`x`$-, $`y`$-, and $`z`$-exponents;
///       * if `cart` is `false`, `order` can be `true` for increasing-$`m`$ order, `false` for
///       decreasing-$`m`$ order, or a list of $`m`$ values for custom order.
///
///     Python type: `tuple[str, bool, bool | Optional[list[tuple[int, int, int]]]]`.
/// * `primitives` - A list of tuples, each of which contains the exponent and the contraction
/// coefficient of a Gaussian primitive in this shell. Python type: `list[tuple[float, float]]`.
/// * `cart_origin` - A fixed-size list of length 3 containing the Cartesian coordinates of the
/// origin $`\mathbf{R}`$ of this shell in Bohr radii. Python type: `list[float]`.
/// * `k` - An optional fixed-size list of length 3 containing the Cartesian components of the
/// $`\mathbf{k}`$ vector of this shell that appears in the complex phase factor
/// $`\exp[i\mathbf{k} \cdot (\mathbf{r} - \mathbf{R})]`$. Python type: `Optional[list[float]]`.
pub struct PyBasisShellContraction {
    /// A triplet of the form `(angmom, cart, order)` where:
    ///     * `angmom` is a symbol such as `"S"` or `"P"` for the angular momentum of the shell,
    ///     * `cart` is a boolean indicating if the functions in the shell are Cartesian (`true`)
    ///     or pure / solid harmonics (`false`), and
    ///     * `order` specifies how the functions in the shell are ordered:
    ///       * if `cart` is `true`, `order` can be `None` for lexicographic order, or a list of
    ///       tuples `(lx, ly, lz)` specifying a custom order for the Cartesian functions where
    ///       `lx`, `ly`, and `lz` are the $`x`$-, $`y`$-, and $`z`$-exponents;
    ///       * if `cart` is `false`, `order` can be `true` for increasing-$`m`$ order, `false` for
    ///       decreasing-$`m`$ order, or a list of $`m`$ values for custom order.
    ///
    /// Python type: `tuple[str, bool, bool | Optional[list[tuple[int, int, int]]]]`.
    pub basis_shell: (u32, ShellType, PyShellOrder),

    /// A list of tuples, each of which contains the exponent and the contraction coefficient of a
    /// Gaussian primitive in this shell.
    ///
    /// Python type: `list[tuple[float, float]]`.
    pub primitives: Vec<(f64, f64)>,

    /// A fixed-size list of length 3 containing the Cartesian coordinates of the origin
    /// $`\mathbf{R}`$ of this shell in Bohr radii.
    ///
    /// Python type: `list[float]`.
    pub cart_origin: [f64; 3],

    /// An optional fixed-size list of length 3 containing the Cartesian components of the
    /// $`\mathbf{k}`$ vector of this shell that appears in the complex phase factor
    /// $`\exp[i\mathbf{k} \cdot (\mathbf{r} - \mathbf{R})]`$.
    ///
    /// Python type: `Optional[list[float]]`.
    pub k: Option<[f64; 3]>,
}

#[cfg(feature = "integrals")]
#[pymethods]
impl PyBasisShellContraction {
    /// Creates a new `PyBasisShellContraction` structure.
    ///
    /// # Arguments
    ///
    /// * `basis_shell` - A triplet of the form `(angmom, cart, order)` where:
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
    ///     Python type: `tuple[str, bool, bool | Optional[list[tuple[int, int, int]]]]`.
    /// * `primitives` - A list of tuples, each of which contains the exponent and the contraction
    /// coefficient of a Gaussian primitive in this shell. Python type: `list[tuple[float, float]]`.
    /// * `cart_origin` - A fixed-size list of length 3 containing the Cartesian coordinates of the
    /// origin of this shell. Python type: `list[float]`.
    /// * `k` - An optional fixed-size list of length 3 containing the Cartesian components of the
    /// $`\mathbf{k}`$ vector of this shell. Python type: `Optional[list[float]]`.
    #[new]
    #[pyo3(signature = (basis_shell, primitives, cart_origin, k=None))]
    pub fn new(
        basis_shell: (u32, ShellType, PyShellOrder),
        primitives: Vec<(f64, f64)>,
        cart_origin: [f64; 3],
        k: Option<[f64; 3]>,
    ) -> Self {
        Self {
            basis_shell,
            primitives,
            cart_origin,
            k,
        }
    }
}

#[cfg(feature = "integrals")]
impl TryFrom<PyBasisShellContraction> for BasisShellContraction<f64, f64> {
    type Error = anyhow::Error;

    fn try_from(pybsc: PyBasisShellContraction) -> Result<Self, Self::Error> {
        let (order, cart, shell_order) = pybsc.basis_shell;
        let basis_shell = create_basis_shell(order, &cart, &shell_order)?;
        let contraction = GaussianContraction::<f64, f64> {
            primitives: pybsc.primitives,
        };
        let cart_origin = Point3::from_slice(&pybsc.cart_origin);
        let k = pybsc.k.map(|k| Vector3::from_row_slice(&k));
        Ok(Self {
            basis_shell,
            contraction,
            cart_origin,
            k,
        })
    }
}

// ================
// Helper functions
// ================

/// Creates a [`BasisShell`] structure from the `(angmom, cart, shell_order)` triplet.
///
/// # Arguments
/// * `order` is an integer indicating the order of the shell,
/// * `cart` is a boolean indicating if the functions in the shell are Cartesian (`true`)
/// or pure / solid harmonics (`false`), and
/// * `shell_order` specifies how the functions in the shell are ordered:
///   * if `cart` is `true`, `order` can be `None` for lexicographic order, or a list of
///   tuples `(lx, ly, lz)` specifying a custom order for the Cartesian functions where
///   `lx`, `ly`, and `lz` are the $`x`$-, $`y`$-, and $`z`$-exponents;
///   * if `cart` is `false`, `order` can be `true` for increasing-$`m`$ or `false` for
///   decreasing-$`m`$ order.
///
/// # Returns
///
/// A [`BasisShell`] structure.
///
/// # Errors
///
/// Errors if `angmom` is not a valid angular momentum, or if there is a mismatch between `cart`
/// and `shell_order`.
fn create_basis_shell(
    order: u32,
    shell_type: &ShellType,
    shell_order: &PyShellOrder,
) -> Result<BasisShell, anyhow::Error> {
    let shl_ord = match shell_type {
        ShellType::Cartesian => {
            let cart_order = match shell_order {
                PyShellOrder::CartOrder(cart_tuples_opt) => {
                    if let Some(cart_tuples) = cart_tuples_opt {
                        CartOrder::new(cart_tuples)?
                    } else {
                        CartOrder::lex(order)
                    }
                }
                PyShellOrder::PureSpinorOrder(_) => {
                    log::error!(
                        "Cartesian shell order expected, but specification for pure/spinor shell order found."
                    );
                    bail!(
                        "Cartesian shell order expected, but specification for pure/spinor shell order found."
                    )
                }
            };
            ShellOrder::Cart(cart_order)
        }
        ShellType::Pure => match shell_order {
            PyShellOrder::PureSpinorOrder(pypureorder) => match pypureorder {
                PyPureSpinorOrder::Standard((increasingm, _even)) => {
                    if *increasingm {
                        ShellOrder::Pure(PureOrder::increasingm(order))
                    } else {
                        ShellOrder::Pure(PureOrder::decreasingm(order))
                    }
                }
                PyPureSpinorOrder::Custom((mls, _even)) => ShellOrder::Pure(PureOrder::new(mls)?),
            },
            PyShellOrder::CartOrder(_) => {
                log::error!(
                    "Pure shell order expected, but specification for Cartesian shell order found."
                );
                bail!(
                    "Pure shell order expected, but specification for Cartesian shell order found."
                )
            }
        },
        ShellType::Spinor => match shell_order {
            PyShellOrder::PureSpinorOrder(pyspinororder) => match pyspinororder {
                PyPureSpinorOrder::Standard((increasingm, even)) => {
                    if *increasingm {
                        ShellOrder::Spinor(SpinorOrder::increasingm(order, *even))
                    } else {
                        ShellOrder::Spinor(SpinorOrder::decreasingm(order, *even))
                    }
                }
                PyPureSpinorOrder::Custom((two_mjs, even)) => {
                    ShellOrder::Spinor(SpinorOrder::new(two_mjs, *even)?)
                }
            },
            PyShellOrder::CartOrder(_) => {
                log::error!(
                    "Spinor shell order expected, but specification for Cartesian shell order found."
                );
                bail!(
                    "Spinor shell order expected, but specification for Cartesian shell order found."
                )
            }
        },
    };
    Ok::<_, anyhow::Error>(BasisShell::new(order, shl_ord))
}

// =================
// Exposed functions
// =================

#[cfg(feature = "integrals")]
#[pyfunction]
/// Calculates the real-valued two-centre overlap matrix for a basis set.
///
/// # Arguments
///
/// * `basis_set` - A list of lists of [`PyBasisShellContraction`]. Each inner list contains shells
/// on one atom. Python type: `list[list[PyBasisShellContraction]]`.
///
/// # Returns
///
/// A two-dimensional array containing the real two-centre overlap values.
///
/// # Panics
///
/// Panics if any shell contains a finite $`\mathbf{k}`$ vector.
pub fn calc_overlap_2c_real<'py>(
    py: Python<'py>,
    basis_set: Vec<Vec<PyBasisShellContraction>>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let bscs = BasisSet::new(
        basis_set
            .into_iter()
            .map(|basis_atom| {
                basis_atom
                    .into_iter()
                    .map(|pybsc| BasisShellContraction::<f64, f64>::try_from(pybsc))
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(|err| PyValueError::new_err(err.to_string()))?,
    );
    let sao_2c = py.allow_threads(|| {
        let stc = build_shell_tuple_collection![
            <s1, s2>;
            false, false;
            &bscs, &bscs;
            f64
        ];
        stc.overlap([0, 0])
            .pop()
            .expect("Unable to retrieve the two-centre overlap matrix.")
    });
    let pysao_2c = sao_2c.into_pyarray(py);
    Ok(pysao_2c)
}

#[cfg(feature = "integrals")]
#[pyfunction]
/// Calculates the complex-valued two-centre overlap matrix for a basis set.
///
/// # Arguments
///
/// * `basis_set` - A list of lists of [`PyBasisShellContraction`]. Each inner list contains shells
/// on one atom. Python type: `list[list[PyBasisShellContraction]]`.
/// * `complex_symmetric` - A boolean indicating if the complex-symmetric overlap is to be
/// calculated.
///
/// # Returns
///
/// A two-dimensional array containing the complex two-centre overlap values.
pub fn calc_overlap_2c_complex<'py>(
    py: Python<'py>,
    basis_set: Vec<Vec<PyBasisShellContraction>>,
    complex_symmetric: bool,
) -> PyResult<Bound<'py, PyArray2<Complex<f64>>>> {
    let bscs = BasisSet::new(
        basis_set
            .into_iter()
            .map(|basis_atom| {
                basis_atom
                    .into_iter()
                    .map(|pybsc| BasisShellContraction::<f64, f64>::try_from(pybsc))
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(|err| PyValueError::new_err(err.to_string()))?,
    );
    let sao_2c = py.allow_threads(|| {
        let stc = build_shell_tuple_collection![
            <s1, s2>;
            !complex_symmetric, false;
            &bscs, &bscs;
            Complex<f64>
        ];
        stc.overlap([0, 0])
            .pop()
            .expect("Unable to retrieve the two-centre overlap matrix.")
    });
    let pysao_2c = sao_2c.into_pyarray(py);
    Ok(pysao_2c)
}

#[cfg(feature = "integrals")]
#[pyfunction]
/// Calculates the real-valued four-centre overlap tensor for a basis set.
///
/// # Arguments
///
/// * `basis_set` - A list of lists of [`PyBasisShellContraction`]. Each inner list contains shells
/// on one atom. Python type: `list[list[PyBasisShellContraction]]`.
///
/// # Returns
///
/// A four-dimensional array containing the real four-centre overlap values.
///
/// # Panics
///
/// Panics if any shell contains a finite $`\mathbf{k}`$ vector.
pub fn calc_overlap_4c_real<'py>(
    py: Python<'py>,
    basis_set: Vec<Vec<PyBasisShellContraction>>,
) -> PyResult<Bound<'py, PyArray4<f64>>> {
    let bscs = BasisSet::new(
        basis_set
            .into_iter()
            .map(|basis_atom| {
                basis_atom
                    .into_iter()
                    .map(|pybsc| BasisShellContraction::<f64, f64>::try_from(pybsc))
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(|err| PyValueError::new_err(err.to_string()))?,
    );
    let sao_4c = py.allow_threads(|| {
        let stc = build_shell_tuple_collection![
            <s1, s2, s3, s4>;
            false, false, false, false;
            &bscs, &bscs, &bscs, &bscs;
            f64
        ];
        stc.overlap([0, 0, 0, 0])
            .pop()
            .expect("Unable to retrieve the four-centre overlap tensor.")
    });
    let pysao_4c = sao_4c.into_pyarray(py);
    Ok(pysao_4c)
}

#[cfg(feature = "integrals")]
#[pyfunction]
/// Calculates the complex-valued four-centre overlap tensor for a basis set.
///
/// # Arguments
///
/// * `basis_set` - A list of lists of [`PyBasisShellContraction`]. Each inner list contains shells
/// on one atom. Python type: `list[list[PyBasisShellContraction]]`.
/// * `complex_symmetric` - A boolean indicating if the complex-symmetric overlap tensor is to be
/// calculated.
///
/// # Returns
///
/// A four-dimensional array containing the complex four-centre overlap values.
pub fn calc_overlap_4c_complex<'py>(
    py: Python<'py>,
    basis_set: Vec<Vec<PyBasisShellContraction>>,
    complex_symmetric: bool,
) -> PyResult<Bound<'py, PyArray4<Complex<f64>>>> {
    let bscs = BasisSet::new(
        basis_set
            .into_iter()
            .map(|basis_atom| {
                basis_atom
                    .into_iter()
                    .map(|pybsc| BasisShellContraction::<f64, f64>::try_from(pybsc))
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(|err| PyValueError::new_err(err.to_string()))?,
    );
    let sao_4c = py.allow_threads(|| {
        let stc = build_shell_tuple_collection![
            <s1, s2, s3, s4>;
            !complex_symmetric, !complex_symmetric, false, false;
            &bscs, &bscs, &bscs, &bscs;
            Complex<f64>
        ];
        stc.overlap([0, 0, 0, 0])
            .pop()
            .expect("Unable to retrieve the four-centre overlap tensor.")
    });
    let pysao_4c = sao_4c.into_pyarray(py);
    Ok(pysao_4c)
}
