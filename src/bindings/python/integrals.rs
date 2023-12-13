//! Python bindings for QSym² atomic-orbital integral evaluations.

use anyhow::{self, bail, ensure};
#[cfg(feature = "integrals")]
use nalgebra::{Point3, Vector3};
#[cfg(feature = "integrals")]
use num_complex::Complex;
#[cfg(feature = "integrals")]
use numpy::{IntoPyArray, PyArray2, PyArray4};
#[cfg(feature = "integrals")]
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::angmom::spinor_rotation_3d::SpinConstraint;
use crate::angmom::ANGMOM_INDICES;
use crate::auxiliary::molecule::Molecule;
use crate::basis::ao::{
    BasisAngularOrder, BasisAtom, BasisShell, CartOrder, PureOrder, ShellOrder,
};
#[cfg(feature = "integrals")]
use crate::basis::ao_integrals::{BasisSet, BasisShellContraction, GaussianContraction};
#[cfg(feature = "integrals")]
use crate::integrals::shell_tuple::build_shell_tuple_collection;

/// Python-exposed enumerated type to handle the union type `bool | list[int]` in Python.
#[derive(Clone, FromPyObject)]
pub enum PyPureOrder {
    /// Variant for standard pure shell order. The associated boolean indicates if the functions
    /// are arranged in increasing-$`m`$ order.
    Standard(bool),

    /// Variant for custom pure shell order. The associated vector contains a sequence of integers
    /// specifying the order of $`m`$ values in the shell.
    Custom(Vec<i32>),
}

/// Python-exposed enumerated type to handle the `ShellOrder` union type `bool |
/// Optional[list[tuple[int, int, int]]]` in Python.
#[derive(Clone, FromPyObject)]
pub enum PyShellOrder {
    /// Variant for pure shell order. The associated value is either a boolean indicating if the
    /// functions are arranged in increasing-$`m`$ order, or a sequence of integers specifying a
    /// custom $`m`$-order.
    ///
    /// Python type: `bool | list[int]`.
    PureOrder(PyPureOrder),

    /// Variant for Cartesian shell order. If the associated `Option` is `None`, the order will be
    /// taken to be lexicographic. Otherwise, the order will be as specified by the $`(x, y, z)`$
    /// exponent tuples.
    ///
    /// Python type: Optional[list[tuple[int, int, int]]].
    CartOrder(Option<Vec<(u32, u32, u32)>>),
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
    ///       `lx`, `ly`, and `lz` are the $`x`$-, $`y`$-, and $`z`$-exponents, respectively;
    ///       * if `cart` is `false`, `order` can be `true` for increasing-$`m`$ order, `false` for
    ///       decreasing-$`m`$ order, or a list of $`m`$ values for custom order.
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

/// Python-exposed enumerated type to marshall basis spin constraint information between Rust and
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
    pub basis_shell: (String, bool, PyShellOrder),

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
            k,
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

// ================
// Helper functions
// ================

/// Creates a [`BasisShell`] structure from the `(angmom, cart, shell_order)` triplet.
///
/// # Arguments
/// * `angmom` is a symbol such as `"S"` or `"P"` for the angular momentum of the shell,
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
            PyShellOrder::PureOrder(pypureorder) => match pypureorder {
                PyPureOrder::Standard(increasingm) => {
                    if *increasingm {
                        ShellOrder::Pure(PureOrder::increasingm(*l))
                    } else {
                        ShellOrder::Pure(PureOrder::decreasingm(*l))
                    }
                }
                PyPureOrder::Custom(mls) => ShellOrder::Pure(PureOrder::new(mls)?),
            },
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
) -> PyResult<&'py PyArray2<f64>> {
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
///
/// # Returns
///
/// A two-dimensional array containing the complex two-centre overlap values.
pub fn calc_overlap_2c_complex<'py>(
    py: Python<'py>,
    basis_set: Vec<Vec<PyBasisShellContraction>>,
) -> PyResult<&'py PyArray2<Complex<f64>>> {
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
            true, false;
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
) -> PyResult<&'py PyArray4<f64>> {
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
///
/// # Returns
///
/// A four-dimensional array containing the complex four-centre overlap values.
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
                    .map(|pybsc| BasisShellContraction::<f64, f64>::try_from(pybsc))
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(|err| PyValueError::new_err(err.to_string()))?,
    );
    let sao_4c = py.allow_threads(|| {
        let stc = build_shell_tuple_collection![
            <s1, s2, s3, s4>;
            true, true, false, false;
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
