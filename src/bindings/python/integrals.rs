use anyhow::{self, bail, ensure};
use nalgebra::{Vector3, Point3};
use num_complex::Complex;
use numpy::{IntoPyArray, PyArray4};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::angmom::spinor_rotation_3d::SpinConstraint;
use crate::angmom::ANGMOM_INDICES;
use crate::auxiliary::molecule::Molecule;
use crate::basis::ao::{
    BasisAngularOrder, BasisAtom, BasisShell, CartOrder, PureOrder, ShellOrder,
};
use crate::basis::ao_integrals::{BasisShellContraction, GaussianContraction, BasisSet};
#[cfg(feature = "integrals")]
use crate::integrals::shell_tuple::build_shell_tuple_collection;

/// A Python-exposed enumerated type to handle the `ShellOrder` union type `bool |
/// Optional[list[tuple[int, int, int]]]` in Python.
#[derive(Clone, FromPyObject)]
pub enum PyShellOrder {
    /// Variant for pure shell order. The associated boolean indicates if the functions are
    /// arranged in increasing-$`m`$ order.
    ///
    /// Python type: `bool`.
    PureOrder(bool),

    /// Variant for Cartesian shell order. If the associated `Option` is `None`, the order will be
    /// taken to be lexicographic. Otherwise, the order will be as specified by the $`(x, y, z)`$
    /// exponent tuples.
    ///
    /// Python type: Optional[list[tuple[int, int, int]]].
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
    /// Python type: `list[tuple[str, list[tuple[str, bool, bool | Optional[list[tuple[int, int, int]]]]]]]`.
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
    pub(crate) fn to_qsym2<'b, 'a: 'b>(
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
                        create_basis_shell(angmom, *cart, shell_order)
                    })
                    .collect::<Vec<_>>();
                Ok(BasisAtom::new(atom, &bss))
            })
            .collect::<Vec<_>>();
        Ok(BasisAngularOrder::new(&basis_atoms))
    }
}

fn create_basis_shell(
    angmom: &str,
    cart: bool,
    shell_order: &PyShellOrder,
) -> Result<BasisShell, anyhow::Error> {
    let l = ANGMOM_INDICES
        .get(angmom)
        .unwrap_or_else(|| panic!("`{angmom}` is not a valid angular momentum."));
    let shl_ord = if cart {
        let cart_order = match shell_order {
            PyShellOrder::CartOrder(cart_tuples_opt) => {
                if let Some(cart_tuples) = cart_tuples_opt {
                    CartOrder::new(cart_tuples)?
                } else {
                    CartOrder::lex(*l)
                }
            }
            PyShellOrder::PureOrder(_) => {
                log::error!(
                    "Cartesian shell order expected, but specification for pure shell order found."
                );
                bail!(
                    "Cartesian shell order expected, but specification for pure shell order found."
                )
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
            }
            PyShellOrder::CartOrder(_) => {
                log::error!(
                    "Pure shell order expected, but specification for Cartesian shell order found."
                );
                bail!(
                    "Pure shell order expected, but specification for Cartesian shell order found."
                )
            }
        }
    };
    Ok::<_, anyhow::Error>(BasisShell::new(*l, shl_ord))
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

#[cfg(feature = "integrals")]
#[pyclass]
#[derive(Clone)]
pub struct PyBasisShellContraction {
    pub basis_shell: (String, bool, PyShellOrder),

    pub primitives: Vec<(f64, f64)>,

    pub cart_origin: [f64; 3],

    pub k: Option<[f64; 3]>,
}

#[pymethods]
impl PyBasisShellContraction {
    /// Creates a new `PyMolecule` structure.
    ///
    /// # Arguments
    ///
    /// * `atoms` - The ordinary atoms in the molecule. Python type: `list[tuple[str, tuple[float,
    /// float, float]]]`.
    /// * `threshold` - Threshold for comparing molecules. Python type: `float`.
    /// * `magnetic_field` - An optional uniform external magnetic field. Python type:
    /// `Optional[tuple[float, float, float]]`.
    /// * `electric_field` - An optional uniform external electric field. Python type:
    /// `Optional[tuple[float, float, float]]`.
    #[new]
    pub fn new(
        basis_shell: (String, bool, PyShellOrder),
        primitives: Vec<(f64, f64)>,
        cart_origin: [f64; 3],
        k: Option<[f64; 3]>,
    ) -> Self {
        Self {
            basis_shell,
            primitives,
            cart_origin,
            k
        }
    }
}

#[cfg(feature = "integrals")]
impl TryFrom<PyBasisShellContraction> for BasisShellContraction<f64, f64> {
    type Error = anyhow::Error;

    fn try_from(pybsc: PyBasisShellContraction) -> Result<Self, Self::Error> {
        let (angmom, cart, shell_order) = pybsc.basis_shell;
        let basis_shell = create_basis_shell(&angmom, cart, &shell_order)?;
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

#[cfg(feature = "integrals")]
#[pyfunction]
pub fn calc_overlap_4c_real<'py>(
    py: Python<'py>,
    basis_set: Vec<Vec<PyBasisShellContraction>>,
) -> PyResult<&'py PyArray4<f64>> {
    let bscs = BasisSet::new(
        basis_set
            .into_iter()
            .map(|basis_atom| {
                basis_atom
                    .into_iter()
                    .map(|pybsc| {
                        BasisShellContraction::<f64, f64>::try_from(pybsc)
                    })
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(|err| PyValueError::new_err(err.to_string()))?
    );
    let stc = build_shell_tuple_collection![
        <s1, s2, s3, s4>;
        false, false, false, false;
        &bscs, &bscs, &bscs, &bscs;
        f64
    ];
    let sao_4c = stc
        .overlap([0, 0, 0, 0])
        .pop()
        .expect("Unable to retrieve the four-centre overlap tensor.");
    let pysao_4c = sao_4c.into_pyarray(py);
    Ok(pysao_4c)
}

#[cfg(feature = "integrals")]
#[pyfunction]
pub fn calc_overlap_4c_complex<'py>(
    py: Python<'py>,
    basis_set: Vec<Vec<PyBasisShellContraction>>,
) -> PyResult<&'py PyArray4<Complex<f64>>> {
    let bscs = BasisSet::new(
        basis_set
            .into_iter()
            .map(|basis_atom| {
                basis_atom
                    .into_iter()
                    .map(|pybsc| {
                        BasisShellContraction::<f64, f64>::try_from(pybsc)
                    })
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(|err| PyValueError::new_err(err.to_string()))?
    );
    let stc = build_shell_tuple_collection![
        <s1, s2, s3, s4>;
        true, true, false, false;
        &bscs, &bscs, &bscs, &bscs;
        Complex<f64>
    ];
    let sao_4c = stc
        .overlap([0, 0, 0, 0])
        .pop()
        .expect("Unable to retrieve the four-centre overlap tensor.");
    let pysao_4c = sao_4c.into_pyarray(py);
    Ok(pysao_4c)
}
