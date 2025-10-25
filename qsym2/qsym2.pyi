import numpy as np
import numpy.typing as npt

from enum import Enum
from typing import Callable

type Py1DArray_f64 = np.ndarray[tuple[int], np.dtype[np.float64]]
type Py1DArray_c128 = np.ndarray[tuple[int], np.dtype[np.complex128]]
type Py2DArray_f64 = np.ndarray[tuple[int, int], np.dtype[np.float64]]
type Py2DArray_c128 = np.ndarray[tuple[int, int], np.dtype[np.complex128]]
type Py4DArray_f64 = np.ndarray[tuple[int, int, int, int], np.dtype[np.float64]]
type Py4DArray_c128 = np.ndarray[tuple[int, int, int, int], np.dtype[np.complex128]]

# ------------------------------
# representation_analysis/mod.rs
# ------------------------------

class MagneticSymmetryAnalysisKind(Enum):
    r"""
    Enumerated type indicating the type of magnetic symmetry to be used for representation analysis.
    """

    Representation = 0
    r"""Variant indicating that unitary representations should be used for magnetic symmetry analysis."""

    Corepresentation = 1
    r"""Variant indicating that magnetic corepresentations should be used for magnetic symmetry analysis."""

# ------------------------------
# symmetry_transformation/mod.rs
# ------------------------------

class SymmetryTransformationKind(Enum):
    r"""Enumerated type for managing the kind of symmetry transformation on an object."""

    Spatial = 0
    r"""Spatial-only transformation."""

    SpatialWithSpinTimeReversal = 1
    r"""Spatial-only transformation but with spin-including time reversal."""

    Spin = 2
    r"""Spin-only transformation."""

    SpinSpatial = 3
    r"""Spin-spatial coupled transformation."""

# ---------------
# analysis/mod.rs
# ---------------

class EigenvalueComparisonMode(Enum):
    r"""Enumerated type specifying the comparison mode for filtering out orbit overlap eigenvalues."""

    Real = 0
    r"""Compares the eigenvalues using only their real parts."""

    Modulus = 1
    r"""Compares the eigenvalues using their moduli."""

# -------------------------------------------
# bindings/python/symmetry_group_detection.rs
# -------------------------------------------

class PyMolecule:
    r"""
    Python-exposed structure to marshall molecular structure information between Rust and Python.
    """

    def __init__(
        self,
        atoms: list[tuple[str, tuple[float, float, float]]],
        threshold: float,
        magnetic_field: tuple[float, float, float] | None,
        electric_field: tuple[float, float, float] | None,
    ) -> None:
        r"""
        Creates a new `PyMolecule` structure.

        :param atoms: The ordinary atoms in the molecule.
        :param threshold: Threshold for comparing molecules.
        :param magnetic_field: An optional uniform external magnetic field.
        :param electric_field: An optional uniform external electric field.
        """

    atoms: list[tuple[str, tuple[float, float, float]]]
    r"""The ordinary atoms in the molecule."""

    threshold: float
    r"""Threshold for comparing molecules."""

    magnetic_field: tuple[float, float, float] | None
    r"""The uniform external magnetic field, if any."""

    electric_field: tuple[float, float, float] | None
    r"""The uniform external electric field, if any."""

class PySymmetryElementKind(Enum):
    r"""
    Python-exposed enumerated type to marshall symmetry element kind information one-way from Rust to Python.

    The possible variants are:
        * `PySymmetryElementKind.Proper`: variant denoting proper symmetry elements,
        * `PySymmetryElementKind.ProperTR`: variant denoting time-reversed proper symmetry elements,
        * `PySymmetryElementKind.ImproperMirrorPlane`: variant denoting improper symmetry elements
        (mirror-plane convention),
        * `PySymmetryElementKind.ImproperMirrorPlaneTR`: variant denoting time-reversed improper
        symmetry elements (mirror-plane convention).
    """

    Proper = 0
    r"""Variant denoting proper symmetry elements."""

    ProperTR = 1
    r"""Variant denoting time-reversed proper symmetry elements."""

    ImproperMirrorPlane = 2
    r"""Variant denoting improper symmetry elements (mirror-plane convention)."""

    ImproperMirrorPlaneTR = 3
    r"""Variant denoting time-reversed improper symmetry elements (mirror-plane convention)."""

class PySymmetry:
    r"""
    Python-exposed structure to marshall symmetry information one-way from Rust to Python.
    """

    def is_infinite(self) -> bool:
        r"""
        Returns a boolean indicating if the group is infinite.
        """

    def get_elements_of_kind(
        self, kind: PySymmetryElementKind
    ) -> dict[int, list[Py1DArray_f64]]:
        r"""
        Returns symmetry elements of all *finite* orders of a given kind.

        :param kind: The symmetry element kind.

        :return: A dictionary where the keys are integers indicating the orders of the elements and the values
        are vectors of one-dimensional arrays, each of which gives the axis of a symmetry element.
        If the order value is `-1`, then the associated elements have infinite order.
        """

    def get_generators_of_kind(
        self, kind: PySymmetryElementKind
    ) -> dict[int, list[Py1DArray_f64]]:
        r"""
        Returns symmetry generators of *finite*  and *infinite* orders of a given kind.

        :param kind: The symmetry generator kind.

        :return: A dictionary where the keys are integers indicating the orders of the generators and the values
        are vectors of one-dimensional arrays, each of which gives the axis of a symmetry generator.
        If the order value is `-1`, then the associated generators have infinite order.
        """

    group_name: str
    r"""The name of the symmetry group."""

def detect_symmetry_group(
    inp_xyz: str | None,
    inp_mol: PyMolecule | None,
    out_sym: str | None,
    moi_thresholds: list[float],
    distance_thresholds: list[float],
    time_reversal: bool,
    write_symmetry_elements: bool = True,
    fictitious_magnetic_field: tuple[float, float, float] | None = None,
    fictitious_electric_field: tuple[float, float, float] | None = None,
) -> tuple[PySymmetry, PySymmetry | None]:
    r"""
    Python-exposed function to perform symmetry-group detection and log the result via the
    `qsym2-output` logger at the `INFO` level.

    :param inp_xyz: An optional string providing the path to an XYZ file containing the molecule to be analysed.
        Only one of `inp_xyz` or `inp_mol` can be specified.
    :param inp_mol: An optional `PyMolecule` structure containing the molecule to be analysed.
        Only one of `inp_xyz` or `inp_mol` can be specified.
    :param out_sym: An optional name for the [`QSym2FileType::Sym`] file to be saved that contains the serialised results of the symmetry-group detection.
    :param moi_thresholds: Thresholds for comparing moments of inertia.
    :param distance_thresholds: Thresholds for comparing distances.
    :param time_reversal: A boolean indicating whether elements involving time reversal should also be considered.
    :param write_symmetry_elements: A boolean indicating if detected symmetry elements should be printed in the output.
    :param fictitious_magnetic_field: An optional fictitious uniform external magnetic field.
    :param fictitious_electric_field: An optional fictitious uniform external electric field.

    :return: A tuple of a [`PySymmetry`] for the unitary group and another optional [`PySymmetry`] for the magnetic group if requested.

    :error: Returns an error if any intermediate step in the symmetry-group detection procedure fails.
    """

# ------------------------------------------
# bindings/python/molecule_symmetrisation.rs
# ------------------------------------------

def symmetrise_molecule(
    inp_xyz: str | None,
    inp_mol: PyMolecule | None,
    out_target_sym: str | None = None,
    loose_moi_threshold: float = 1e-2,
    loose_distance_threshold: float = 1e-2,
    target_moi_threshold: float = 1e-7,
    target_distance_threshold: float = 1e-7,
    use_magnetic_group: bool = False,
    reorientate_molecule: bool = True,
    max_iterations: int = 50,
    consistent_target_symmetry_iterations: int = 10,
    verbose: int = 0,
    infinite_order_to_finite: int | None = None,
) -> PyMolecule:
    r"""
    Python-exposed function to perform molecule symmetrisation by bootstrapping and log the result
    via the `qsym2-output` logger at the `INFO` level.

    :param inp_xyz: An optional string providing the path to an XYZ file containing the molecule to be symmetrised. Only one of `inp_xyz` or `inp_mol` can be specified.
    :param inp_mol: An optional `PyMolecule` structure containing the molecule to be symmetrised. Only one of `inp_xyz` or `inp_mol` can be specified.
    :param out_target_sym: An optional path for a [`QSym2FileType::Sym`] file to be saved that contains the symmetry-group detection results of the symmetrised molecule at the target thresholds.
    :param loose_moi_threshold: The loose MoI threshold.
    :param loose_distance_threshold: The loose distance threshold.
    :param target_moi_threshold: The target (tight) MoI threshold.
    :param target_distance_threshold: The target (tight) distance threshold.
    :param use_magnetic_group: A boolean indicating if the magnetic group (*i.e.* the group including time-reversed operations) is to be used for the symmetrisation.
    :param reorientate_molecule: A boolean indicating if the molecule is also reoriented to align its principal axes with the Cartesian axes.
    :param max_iterations: The maximum number of iterations for the symmetrisation process.
    :param consistent_target_symmetry_iterations: The number of consecutive iterations during which the symmetry group at the target level of threshold must be consistently found for convergence to be reached, if this group cannot become identical to the symmetry group at the loose level of threshold.
    :param verbose: The print-out level.
    :param infinite_order_to_finite: The finite order with which infinite-order generators are to be interpreted to form a finite subgroup of the prevailing infinite group. This finite subgroup will be used for the symmetrisation.

    :return: The symmetrised molecule.

    :error: Errors if any intermediate step in the symmetrisation procedure fails.
    """

# ----------------------------
# bindings/python/integrals.rs
# ----------------------------

type PyPureSpinorOrder = tuple[bool, bool] | tuple[list[int], bool]

type PyShellOrder = PyPureSpinorOrder | list[tuple[int, int, int]] | None

class ShellType(Enum):
    r"""
    Python-exposed enumerated type indicating the shell type.

    The possible variants are:
        * `ShellType.Pure`: variant for a pure shell,
        * `ShellType.Cartesian`: variant for a Cartesian shell,
        * `ShellType.SpinorFermion`: variant for a spinor shell corresponding to a
        fermion without any additional balance symmetries,
        * `ShellType.SpinorFermionKineticBalance`: variant for a spinor shell corresponding to a
        fermion with the kinetic balance symmetry due to
        $`\mathbf{\sigma} \dot \hat{\mathbf{p}}`$,
        * `ShellType.SpinorAntifermion`: variant for a spinor shell
        corresponding to an antifermion without any additional balance symmetries,
        * `ShellType.SpinorAntifermionKineticBalance`: variant for a spinor shell
        describing an antifermion with the kinetic balance symmetry due to
        $`\mathbf{\sigma} \dot \hat{\mathbf{p}}`$.
    """

    Pure = 0
    r"""Variant for a pure shell."""

    Cartesian = 1
    r"""Variant for a Cartesian shell."""

    SpinorFermion = 2
    r"""Variant for a spinor shell corresponding to a fermion without any additional balance symmetries."""

    SpinorFermionKineticBalance = 3
    r"""Variant for a spinor shell corresponding to a fermion with the kinetic balance symmetry due to $`\mathbf{\sigma} \dot \hat{\mathbf{p}}`$."""

    SpinorAntifermion = 4
    r"""Variant for a spinor shell corresponding to an antifermion without any additional balance symmetries."""

    SpinorAntifermionKineticBalance = 5
    r"""Variant for a spinor shell describing an antifermion with the kinetic balance symmetry due to $`\mathbf{\sigma} \dot \hat{\mathbf{p}}`$."""

class PyBasisAngularOrder:
    """
    Python-exposed structure to marshall basis angular order information between Python and Rust.
    """

    def __init__(
        self, basis_atoms: list[tuple[str, list[tuple[int, ShellType, PyShellOrder]]]]
    ) -> None:
        r"""
        Constructs a new `PyBasisAngularOrder` structure.

        :param basis_atoms: A vector of tuples, each of which provides information for one basis atom in the form `(element, basis_shells)`.
            Here:
            * `element` is a string giving the element symbol of the atom, and
            * `basis_shells` is a vector of tuples, each of which provides information for one basis
            shell on the atom in the form `(angmom, shelltype, order)`. Here:
                * `angmom` is an integer specifying the angular momentum of the shell, the meaning of
                which depends on the shell type,
                * `shelltype` is an enumerated type with possible variants `ShellType.Pure`,
                `ShellType.Cartesian`, `ShellType.Spinor`, and `ShellType.SpinorKineticBalance`
                indicating the type of the shell, and
                * `order` specifies how the functions in the shell are ordered:
                    * if `shelltype` is `ShellType.Cartesian`, `order` can be `None` for lexicographic order, or a
                    list of tuples `(lx, ly, lz)` specifying a custom order for the Cartesian functions
                    where `lx`, `ly`, and `lz` are the $`x`$-, $`y`$-, and $`z`$-exponents, respectively;
                    * if `shelltype` is `ShellType.Pure`, `ShellType.SpinorFermion`,
                    `SpinorFermionKineticBalance`, `ShellType.SpinorAntifermion`, or
                    `ShellType.SpinorAntifermionKineticBalance`, `order` is a tuple of two values: the
                    first can be `true` for increasing-$`m`$ order, `false` for decreasing-$`m`$ order, or
                    a list of $`m`$ values for custom order, and the second is a boolean indicating whether
                    the spatial parts of the functions in the shell are even with respect to spatial inversion.
        r"""

    @classmethod
    def from_qchem_archive(cls, filename: str) -> list[PyBasisAngularOrder]:
        r"""
        Extracts basis angular order information from a Q-Chem HDF5 archive file.

        :param filename: A path to a Q-Chem HDF5 archive file.

        :return: A sequence of `PyBasisAngularOrder` objects, one for each Q-Chem calculation found in the HDF5 archive file.

        A summary showing how the `PyBasisAngularOrder` objects map onto the Q-Chem calculations in
        the HDF5 archive file is also logged at the `INFO` level.
        """

class PySpinConstraint(Enum):
    r"""
    Python-exposed enumerated type to marshall basis spin constraint information between Rust and Python.

    The possible variants are:
        * `PySpinConstraint.Restricted`: variant for restricted spin constraint. Only two spin spaces are exposed,
        * `PySpinConstraint.Unrestricted`: variant for unrestricted spin constraint. Only two spin spaces arranged
        in decreasing-$`m`$ order (*i.e.* $`(\alpha, \beta)`$) are exposed,
        * `PySpinConstraint.Generalised`: variant for generalised spin constraint. Only two spin spaces arranged in
        decreasing-$`m`$ order (*i.e.* $`(\alpha, \beta)`$) are exposed.
    """

    Restricted = 0
    r"""Variant for restricted spin constraint. Only two spin spaces are exposed."""

    Unrestricted = 1
    r"""Variant for unrestricted spin constraint. Only two spin spaces arranged in decreasing-$`m`$ order (*i.e.* $`(\alpha, \beta)`$) are exposed."""

    Generalised = 2
    r"""Variant for generalised spin constraint. Only two spin spaces arranged in decreasing-$`m`$ order (*i.e.* $`(\alpha, \beta)`$) are exposed."""

class PySpinOrbitCoupled:
    r"""
    Python-exposed enumerated type to marshall basis spin--orbit-coupled layout in the coupled
    treatment of spin and spatial degrees of freedom between Rust and Python.

    The possible variants are:
        * `PySpinOrbitCoupled.JAdapted2C`: variant for two-component $`j`$-adapted basis functions,
        * `PySpinOrbitCoupled.JAdapted4C`: variant for four-component $`j`$-adapted basis functions.
    """

type PyStructureConstraint = PySpinConstraint | PySpinOrbitCoupled

class PyBasisShellContraction:
    r"""
    Python-exposed structure to marshall basis shell contraction information between Rust and
    Python.
    """

    def __init__(
        self,
        basis_shell: tuple[int, ShellType, PyShellOrder],
        primitives: list[tuple[float, float]],
        cart_origin: tuple[float, float, float],
        k: tuple[float, float, float] | None,
    ) -> None:
        r"""
        Constructs a new `PyBasisAngularOrder` structure.

        :param basis_shell: A tuple which provides information for one basis shell on the atom in the form `(angmom, shelltype, order)`.
            Here:
            * `angmom` is an integer specifying the angular momentum of the shell, the meaning of
            which depends on the shell type,
            * `shelltype` is an enumerated type with possible variants `ShellType.Pure`,
            `ShellType.Cartesian`, `ShellType.Spinor`, and `ShellType.SpinorKineticBalance`
            indicating the type of the shell, and
            * `order` specifies how the functions in the shell are ordered:
                * if `shelltype` is `ShellType.Cartesian`, `order` can be `None` for lexicographic order, or a
                list of tuples `(lx, ly, lz)` specifying a custom order for the Cartesian functions
                where `lx`, `ly`, and `lz` are the $`x`$-, $`y`$-, and $`z`$-exponents, respectively;
                * if `shelltype` is `ShellType.Pure`, `ShellType.SpinorFermion`,
                `SpinorFermionKineticBalance`, `ShellType.SpinorAntifermion`, or
                `ShellType.SpinorAntifermionKineticBalance`, `order` is a tuple of two values: the
                first can be `true` for increasing-$`m`$ order, `false` for decreasing-$`m`$ order, or
                a list of $`m`$ values for custom order, and the second is a boolean indicating whether
        :param primitives: A list of tuples, each of which contains the exponent and the contraction coefficient of a Gaussian primitive in this shell.
        :param cart_origin: A fixed-size list of length 3 containing the Cartesian coordinates of the origin of this shell.
        :param k: An optional fixed-size list of length 3 containing the Cartesian components of the $`\mathbf{k}`$ vector of this shell.
        """

def calc_overlap_2c_real(
    basis_set: list[list[PyBasisShellContraction]],
) -> Py2DArray_f64:
    r"""
    Calculates the real-valued two-centre overlap matrix for a basis set.

    :param basis_set: A list of lists of [`PyBasisShellContraction`]. Each inner list contains shells on one atom.

    :return: A two-dimensional array containing the real two-centre overlap values.

    :panic: Panics if any shell contains a finite $`\mathbf{k}`$ vector.
    """

def calc_overlap_2c_complex(
    basis_set: list[list[PyBasisShellContraction]], complex_symmetric: bool
) -> Py2DArray_c128:
    r"""
    Calculates the complex-valued two-centre overlap matrix for a basis set.

    :param basis_set: A list of lists of [`PyBasisShellContraction`]. Each inner list contains shells on one atom.
    :param complex_symmetric: A boolean indicating if the complex-symmetric overlap is to be calculated.

    :return: A two-dimensional array containing the complex two-centre overlap values.
    """

def calc_overlap_4c_real(
    basis_set: list[list[PyBasisShellContraction]],
) -> Py4DArray_f64:
    r"""
    Calculates the real-valued four-centre overlap matrix for a basis set.

    :param basis_set: A list of lists of [`PyBasisShellContraction`]. Each inner list contains shells on one atom.

    :return: A four-dimensional array containing the real two-centre overlap values.

    :panic: Panics if any shell contains a finite $`\mathbf{k}`$ vector.
    """

def calc_overlap_4c_complex(
    basis_set: list[list[PyBasisShellContraction]], complex_symmetric: bool
) -> Py4DArray_c128:
    r"""
    Calculates the complex-valued four-centre overlap matrix for a basis set.

    :param basis_set: A list of lists of [`PyBasisShellContraction`]. Each inner list contains shells on one atom.
    :param complex_symmetric: A boolean indicating if the complex-symmetric overlap is to be calculated.

    :return: A four-dimensional array containing the complex four-centre overlap values.
    """

# -------------------------------------------------------------
# bindings/python/representation_analysis/slater_determinant.rs
# -------------------------------------------------------------

class PySlaterDeterminantReal:
    r"""
    Python-exposed structure to marshall real Slater determinant information between Rust and Python.
    """

    def __init__(
        self,
        structure_constraint: PyStructureConstraint,
        complex_symmetric: bool,
        coefficients: list[Py2DArray_f64],
        occupations: list[Py1DArray_f64],
        threshold: float,
        mo_energies: list[Py1DArray_f64] | None = None,
        energy: float | None = None,
    ) -> None:
        r"""
        Constructs a real Python-exposed Slater determinant.

        :param structure_constraint: The structure constraint applied to the coefficients of the determinant.
        :param complex_symmetric: A boolean indicating if inner products involving this determinant are complex-symmetric.
        :param coefficients: The real coefficient matrices for the molecular orbitals of this determinant, one for each spin space.
        :param occupations: The occupation arrays for the molecular orbitals, each of which is a one-dimensional array for each spin space.
        :param threshold: The threshold for comparisons.
        :param mo_energies: The optional real molecular orbital energy arrays, each of which is a one-dimensional array for each spin space.
        :param energy: The optional real determinantal energy.
        """

    complex_symmetric: bool
    r"""Boolean indicating if inner products involving this determinant are complex-symmetric."""

    coefficients: list[Py2DArray_f64]
    r"""The real coefficient matrices for the molecular orbitals of this determinant, one for each spin space."""

    occupations: list[Py1DArray_f64]
    r"""The occupation arrays for the molecular orbitals, each of which is a one-dimensional array for each spin space."""

    threshold: float
    r"""The threshold for comparisons."""

    mo_energies: list[Py1DArray_f64] | None
    r"""The real molecular orbital energy arrays, if any, each of which is a one-dimensional array for each spin space."""

    energy: float | None
    r"""The real determinantal energy, if any."""

class PySlaterDeterminantComplex:
    r"""
    Python-exposed structure to marshall complex Slater determinant information between Rust and Python.
    """

    def __init__(
        self,
        structure_constraint: PyStructureConstraint,
        complex_symmetric: bool,
        coefficients: list[Py2DArray_c128],
        occupations: list[Py1DArray_f64],
        threshold: float,
        mo_energies: list[Py1DArray_c128] | None = None,
        energy: complex | None = None,
    ) -> None:
        r"""
        Constructs a real Python-exposed Slater determinant.

        :param structure_constraint: The structure constraint applied to the coefficients of the determinant.
        :param complex_symmetric: A boolean indicating if inner products involving this determinant are complex-symmetric.
        :param coefficients: The complex coefficient matrices for the molecular orbitals of this determinant, one for each spin space.
        :param occupations: The occupation arrays for the molecular orbitals, each of which is a one-dimensional array for each spin space.
        :param threshold: The threshold for comparisons.
        :param mo_energies: The optional complex molecular orbital energy arrays, each of which is a one-dimensional array for each spin space.
        :param energy: The optional complex determinantal energy.
        """

    complex_symmetric: bool
    r"""Boolean indicating if inner products involving this determinant are complex-symmetric."""

    coefficients: list[Py2DArray_c128]
    r"""The complex coefficient matrices for the molecular orbitals of this determinant, one for each spin space."""

    occupations: list[Py1DArray_f64]
    r"""The occupation arrays for the molecular orbitals, each of which is a one-dimensional array for each spin space."""

    threshold: float
    r"""The threshold for comparisons."""

    mo_energies: list[Py1DArray_c128] | None
    r"""The complex molecular orbital energy arrays, if any, each of which is a one-dimensional array for each spin space."""

    energy: float | None
    r"""The complex determinantal energy, if any."""

type PySlaterDeterminant = PySlaterDeterminantReal | PySlaterDeterminantComplex

class PySlaterDeterminantRepAnalysisResult:
    r"""
    Python-exposed structure storing the results of Slater determinant representation analysis.
    """

    group: str
    r"""The group used for the representation analysis."""

    determinant_symmetry: str | None
    r"""The deduced overall symmetry of the determinant."""

    mo_symmetries: list[list[str | None]] | None
    r"""The deduced symmetries of the molecular orbitals constituting the determinant, if required."""

    determinant_density_symmetries: list[tuple[str, str | None]] | None
    r"""
    The deduced symmetries of the various densities constructible from the determinant, if
    required. In each tuple, the first element gives a description of the density corresponding
    to the symmetry result.
    """

    mo_density_symmetries: list[list[str | None]] | None
    r"""
    The deduced symmetries of the total densities of the molecular orbitals constituting the determinant,
    if required.
    """

def rep_analyse_slater_determinant(
    inp_sym: str,
    pydet: PySlaterDeterminant,
    pybaos: list[PyBasisAngularOrder],
    integrality_threshold: float,
    linear_independence_threshold: float,
    use_magnetic_group: MagneticSymmetryAnalysisKind | None,
    use_double_group: bool,
    use_cayley_table: bool,
    symmetry_transformation_kind: SymmetryTransformationKind,
    eigenvalue_comparison_mode: EigenvalueComparisonMode,
    sao: Py2DArray_f64 | Py2DArray_c128,
    sao_h: Py2DArray_f64 | Py2DArray_c128 | None = None,
    sao_spatial_4c: Py4DArray_f64 | Py4DArray_c128 | None = None,
    sao_spatial_4c_h: Py4DArray_f64 | Py4DArray_c128 | None = None,
    analyse_mo_symmetries: bool = True,
    analyse_mo_symmetry_projections: bool = True,
    analyse_mo_mirror_parities: bool = False,
    analyse_density_symmetries: bool = False,
    write_overlap_eigenvalues: bool = True,
    write_character_table: bool = True,
    infinite_order_to_finite: int | None = None,
    angular_function_integrality_threshold: float = 1e-7,
    angular_function_linear_independence_threshold: float = 1e-7,
    angular_function_max_angular_momentum: int = 2,
) -> PySlaterDeterminantRepAnalysisResult:
    r"""
    Python-exposed function to perform representation symmetry analysis for real and complex
    Slater determinants and log the result via the `qsym2-output` logger at the `INFO` level.

    If `symmetry_transformation_kind` includes spin transformation, the provided determinant will
    be augmented to generalised spin constraint automatically.

    :param inp_sym: A path to the [`QSym2FileType::Sym`] file containing the symmetry-group detection result for the system.
        This will be used to construct abstract groups and character tables for representation analysis.
    :param pydet: A Python-exposed Slater determinant whose coefficients are of type `float64` or `complex128`.
    :param pybaos: Python-exposed structures containing basis angular order information, one for each explicit component per coefficient matrix.
    :param integrality_threshold: The threshold for verifying if subspace multiplicities are integral.
    :param linear_independence_threshold: The threshold for determining the linear independence subspace via the non-zero eigenvalues of the orbit overlap matrix.
    :param use_magnetic_group: An option indicating if the magnetic group is to be used for symmetry analysis, and if so, whether unitary representations or unitary-antiunitary corepresentations should be used.
    :param use_double_group: A boolean indicating if the double group of the prevailing symmetry group is to be used for representation analysis instead.
    :param use_cayley_table: A boolean indicating if the Cayley table for the group, if available, should be used to speed up the calculation of orbit overlap matrices.
    :param symmetry_transformation_kind: An enumerated type indicating the type of symmetry transformations to be performed on the origin determinant to generate the orbit.
        If this contains spin transformation, the determinant will be augmented to generalised spin constraint automatically.
    :param eigenvalue_comparison_mode: An enumerated type indicating the mode of comparison of orbit overlap eigenvalues with the specified `linear_independence_threshold`.
    :param sao: The atomic-orbital overlap matrix whose elements are of type `float64` or `complex128`.
    :param sao_h: The optional complex-symmetric atomic-orbital overlap matrix whose elements are of type `float64` or `complex128`.
        This is required if antiunitary symmetry operations are involved.
    :param sao_spatial_4c: The optional atomic-orbital four-centre overlap matrix whose elements are of type `float64` or `complex128`.
    :param sao_spatial_4c_h: The optional complex-symmetric atomic-orbital four-centre overlap matrix whose elements are of type `float64` or `complex128`.
        This is required if antiunitary symmetry operations are involved.
    :param analyse_mo_symmetries: A boolean indicating if the symmetries of individual molecular orbitals are to be analysed.
    :param analyse_mo_symmetry_projections: A boolean indicating if the symmetry projections of individual molecular orbitals are to be analysed.
    :param analyse_mo_mirror_parities: A boolean indicating if the mirror parities of individual molecular orbitals are to be printed.
    :param analyse_density_symmetries: A boolean indicating if the symmetries of densities are to be analysed.
    :param write_overlap_eigenvalues: A boolean indicating if the eigenvalues of the determinant orbit overlap matrix are to be written to the output.
    :param write_character_table: A boolean indicating if the character table of the prevailing symmetry group is to be printed out.
    :param infinite_order_to_finite: The finite order with which infinite-order generators are to be interpreted to form a finite subgroup of the prevailing infinite group.
        This finite subgroup will be used for symmetry analysis.
    :param angular_function_integrality_threshold: The threshold for verifying if subspace multiplicities are integral for the symmetry analysis of angular functions.
    :param angular_function_linear_independence_threshold: The threshold for determining the linear independence subspace via the non-zero eigenvalues of the orbit overlap matrix for the symmetry analysis of angular functions.
    :param angular_function_max_angular_momentum: The maximum angular momentum order to be used in angular function symmetry analysis.

    :return: A Python-exposed [`PySlaterDeterminantRepAnalysisResult`] structure containing the results of the representation analysis.
    """

# --------------------------------------------------
# bindings/python/representation_analysis/density.rs
# --------------------------------------------------

class PyDensityReal:
    r"""
    Python-exposed structure to marshall real electron density information between Rust and Python.
    """

    def __init__(
        self,
        complex_symmetric: bool,
        density_matrix: Py2DArray_f64,
        threshold: float,
    ) -> None:
        r"""
        Constructs a real Python-exposed electron density.

        :param complex_symmetric: A boolean indicating if inner products involving this density are complex-symmetric.
        :param density_matrix: The real density matrix describing this density.
        :param threshold: The threshold for comparisons.
        """

    complex_symmetric: bool
    r"""Boolean indicating if inner products involving this density are complex-symmetric."""

    density_matrix: Py2DArray_f64
    r"""The real density matrix describing this density."""

    threshold: float
    r"""The threshold for comparisons."""

class PyDensityComplex:
    r"""
    Python-exposed structure to marshall complex electron density information between Rust and Python.
    """

    def __init__(
        self,
        complex_symmetric: bool,
        density_matrix: Py2DArray_c128,
        threshold: float,
    ) -> None:
        r"""
        Constructs a complex Python-exposed electron density.

        :param complex_symmetric: A boolean indicating if inner products involving this density are complex-symmetric.
        :param density_matrix: The complex density matrix describing this density.
        :param threshold: The threshold for comparisons.
        """

    complex_symmetric: bool
    r"""Boolean indicating if inner products involving this density are complex-symmetric."""

    density_matrix: Py2DArray_c128
    r"""The complex density matrix describing this density."""

    threshold: float
    r"""The threshold for comparisons."""

type PyDensity = PyDensityReal | PyDensityComplex

def rep_analyse_densities(
    inp_sym: str,
    pydens: list[tuple[str, PyDensity]],
    pybao: PyBasisAngularOrder,
    integrality_threshold: float,
    linear_independence_threshold: float,
    use_magnetic_group: MagneticSymmetryAnalysisKind | None,
    use_double_group: bool,
    use_cayley_table: bool,
    symmetry_transformation_kind: SymmetryTransformationKind,
    eigenvalue_comparison_mode: EigenvalueComparisonMode,
    sao_spatial_4c: Py4DArray_f64 | Py4DArray_c128,
    sao_spatial_4c_h: Py4DArray_f64 | Py4DArray_c128 | None = None,
    write_character_table: bool = True,
    infinite_order_to_finite: int | None = None,
    angular_function_integrality_threshold: float = 1e-7,
    angular_function_linear_independence_threshold: float = 1e-7,
    angular_function_max_angular_momentum: int = 2,
) -> None:
    r"""
    Python-exposed function to perform representation symmetry analysis for real and complex
    electron densities and log the result via the `qsym2-output` logger at the `INFO` level.

    :param inp_sym: A path to the [`QSym2FileType::Sym`] file containing the symmetry-group detection result for the system.
        This will be used to construct abstract groups and character tables for representation analysis.
    :param pydens: A sequence of Python-exposed electron densities whose density matrices are of type `float64` or `complex128`.
        Each density is accompanied by a description string.
    :param pybao: Python-exposed structure containing basis angular order information for the density matrices.
    :param integrality_threshold: The threshold for verifying if subspace multiplicities are integral.
    :param linear_independence_threshold: The threshold for determining the linear independence subspace via the non-zero eigenvalues of the orbit overlap matrix.
    :param use_magnetic_group: An option indicating if the magnetic group is to be used for symmetry analysis, and if so, whether unitary representations or unitary-antiunitary corepresentations should be used.
    :param use_double_group: A boolean indicating if the double group of the prevailing symmetry group is to be used for representation analysis instead.
    :param use_cayley_table: A boolean indicating if the Cayley table for the group, if available, should be used to speed up the calculation of orbit overlap matrices.
    :param symmetry_transformation_kind: An enumerated type indicating the type of symmetry transformations to be performed on the origin electron density to generate the orbit.
    :param eigenvalue_comparison_mode: An enumerated type indicating the mode of comparison of orbit overlap eigenvalues with the specified `linear_independence_threshold`.
    :param sao_spatial_4c: The atomic-orbital four-centre overlap matrix whose elements are of type `float64` or `complex128`.
    :param sao_spatial_4c_h: The optional complex-symmetric atomic-orbital four-centre overlap matrix whose elements are of type `float64` or `complex128`.
        This is required if antiunitary symmetry operations are involved.
    :param write_character_table: A boolean indicating if the character table of the prevailing symmetry group is to be printed out.
    :param infinite_order_to_finite: The finite order with which infinite-order generators are to be interpreted to form a finite subgroup of the prevailing infinite group.
        This finite subgroup will be used for symmetry analysis.
    :param angular_function_integrality_threshold: The threshold for verifying if subspace multiplicities are integral for the symmetry analysis of angular functions.
    :param angular_function_linear_independence_threshold: The threshold for determining the linear independence subspace via the non-zero eigenvalues of the orbit overlap matrix for the symmetry analysis of angular functions.
    :param angular_function_max_angular_momentum: The maximum angular momentum order to be used in angular function symmetry analysis.
    """

# -----------------------------------------------------------------
# bindings/python/representation_analysis/vibrational_coordinate.rs
# -----------------------------------------------------------------

class PyVibrationalCoordinateCollectionReal:
    r"""
    Python-exposed structure to marshall real vibrational coordinate collections between Rust and Python.
    """

    def __init__(
        self,
        coefficients: Py2DArray_f64,
        frequencies: Py1DArray_f64,
        threshold: float,
    ) -> None:
        r"""
        Constructs a real vibrational coordinate collection.

        :param coefficients: The real coefficients for the vibrational coordinates of this collection.
        :param frequencies: The real vibrational frequencies.
        :param threshold: The threshold for comparisons.
        """

    coefficients: Py2DArray_f64
    r"""The real coefficients for the vibrational coordinates of this collection."""

    frequencies: Py1DArray_f64
    r"""The real vibrational frequencies."""

    threshold: float
    r"""The threshold for comparisons."""

class PyVibrationalCoordinateCollectionComplex:
    r"""
    Python-exposed structure to marshall complex vibrational coordinate collections between Rust and Python.
    """

    def __init__(
        self,
        coefficients: Py2DArray_c128,
        frequencies: Py1DArray_c128,
        threshold: float,
    ) -> None:
        r"""
        Constructs a complex vibrational coordinate collection.

        :param coefficients: The complex coefficients for the vibrational coordinates of this collection.
        :param frequencies: The complex vibrational frequencies.
        :param threshold: The threshold for comparisons.
        """

    coefficients: Py2DArray_c128
    r"""The complex coefficients for the vibrational coordinates of this collection."""

    frequencies: Py1DArray_c128
    r"""The complex vibrational frequencies."""

    threshold: float
    r"""The threshold for comparisons."""

type PyVibrationalCoordinateCollection = (
    PyVibrationalCoordinateCollectionReal | PyVibrationalCoordinateCollectionComplex
)

def rep_analyse_vibrational_coordinate_collection(
    inp_sym: str,
    pyvibs: PyVibrationalCoordinateCollection,
    integrality_threshold: float,
    linear_independence_threshold: float,
    use_magnetic_group: MagneticSymmetryAnalysisKind | None,
    use_double_group: bool,
    use_cayley_table: bool,
    symmetry_transformation_kind: SymmetryTransformationKind,
    eigenvalue_comparison_mode: EigenvalueComparisonMode,
    write_character_table: bool = True,
    infinite_order_to_finite: int | None = None,
    angular_function_integrality_threshold: float = 1e-7,
    angular_function_linear_independence_threshold: float = 1e-7,
    angular_function_max_angular_momentum: int = 2,
) -> None:
    r"""
    Python-exposed function to perform representation symmetry analysis for real and complex
    vibrational coordinate collections and log the result via the `qsym2-output` logger at the
    `INFO` level.

    :param inp_sym: A path to the [`QSym2FileType::Sym`] file containing the symmetry-group detection result for the system.
        This will be used to construct abstract groups and character tables for representation analysis.
    :param pyvibs: A Python-exposed vibrational coordinate collection whose coefficients are of type `float64` or `complex128`.
    :param integrality_threshold: The threshold for verifying if subspace multiplicities are integral.
    :param linear_independence_threshold: The threshold for determining the linear independence subspace via the non-zero eigenvalues of the orbit overlap matrix.
    :param use_magnetic_group: An option indicating if the magnetic group is to be used for symmetry analysis, and if so, whether unitary representations or unitary-antiunitary corepresentations should be used.
    :param use_double_group: A boolean indicating if the double group of the prevailing symmetry group is to be used for representation analysis instead.
    :param use_cayley_table: A boolean indicating if the Cayley table for the group, if available, should be used to speed up the calculation of orbit overlap matrices.
    :param symmetry_transformation_kind: An enumerated type indicating the type of symmetry transformations to be performed on the origin vibrational coordinate to generate the orbit.
    :param eigenvalue_comparison_mode: An enumerated type indicating the mode of comparison of orbit overlap eigenvalues with the specified `linear_independence_threshold`.
    :param write_character_table: A boolean indicating if the character table of the prevailing symmetry group is to be printed out.
    :param infinite_order_to_finite: The finite order with which infinite-order generators are to be interpreted to form a finite subgroup of the prevailing infinite group.
        This finite subgroup will be used for symmetry analysis.
    :param angular_function_integrality_threshold: The threshold for verifying if subspace multiplicities are integral for the symmetry analysis of angular functions.
    :param angular_function_linear_independence_threshold: The threshold for determining the linear independence subspace via the non-zero eigenvalues of the orbit overlap matrix for the symmetry analysis of angular functions.
    :param angular_function_max_angular_momentum: The maximum angular momentum order to be used in angular function symmetry analysis.
    """

# ---------------------------------------------------------------
# bindings/python/representation_analysis/multideterminant/mod.rs
# ---------------------------------------------------------------

class PyMultiDeterminantsReal:
    r"""
    Python-exposed structure to marshall real multi-determinant information between Rust and Python.
    """

    def __init__(
        self,
        basis: list[PySlaterDeterminantReal],
        coefficients: Py2DArray_f64,
        energies: Py1DArray_f64,
        density_matrices: list[Py2DArray_f64] | None,
        threshold: float,
    ) -> None:
        r"""
        Constructs a set of real Python-exposed multi-determinants.

        :param basis: The basis of Slater determinants in which the multi-determinantal states are expressed.
        :param coefficients: The coefficients for the multi-determinantal states in the specified basis.
            Each column of the coefficient matrix contains the coefficients for one state.
        :param energies: The energies of the multi-determinantal states.
        :param density_matrices: The optional density matrices of the multi-determinantal states.
        :param threshold: The threshold for comparisons.
        """

    basis: list[PySlaterDeterminantReal]
    r"""The basis of Slater determinants in which the multi-determinantal states are expressed."""

    coefficients: Py2DArray_f64
    r"""
    The coefficients for the multi-determinantal states in the specified basis.
    Each column of the coefficient matrix contains the coefficients for one state.
    """

    energies: Py1DArray_f64
    r"""The energies of the multi-determinantal states."""

    density_matrices: list[Py2DArray_f64] | None
    r"""The density matrices for the multi-determinantal states in the specified basis, if any."""

    threshold: float
    r"""The threshold for comparisons."""

    def complex_symmetric(self) -> bool:
        r"""Boolean indicating whether inner products involving these multi-determinantal states are complex-symmetric."""

    def state_coefficients(self, state_index: int) -> Py1DArray_f64:
        r"""Returns the coefficients for a particular state."""

    def state_energy(self, state_index: int) -> float:
        r"""Returns the energy for a particular state."""

    def state_density_matrix(self, state_index: int) -> Py2DArray_f64:
        r"""Returns the density matrix for a particular state."""

class PyMultiDeterminantsComplex:
    r"""
    Python-exposed structure to marshall complex multi-determinant information between Rust and Python.
    """

    def __init__(
        self,
        basis: list[PySlaterDeterminantComplex],
        coefficients: Py2DArray_c128,
        energies: Py1DArray_c128,
        density_matrices: list[Py2DArray_c128] | None,
        threshold: float,
    ) -> None:
        r"""
        Constructs a set of complex Python-exposed multi-determinants.

        :param basis: The basis of Slater determinants in which the multi-determinantal states are expressed.
        :param coefficients: The coefficients for the multi-determinantal states in the specified basis.
            Each column of the coefficient matrix contains the coefficients for one state.
        :param energies: The energies of the multi-determinantal states.
        :param density_matrices: The optional density matrices of the multi-determinantal states.
        :param threshold: The threshold for comparisons.
        """

    basis: list[PySlaterDeterminantComplex]
    r"""The basis of Slater determinants in which the multi-determinantal states are expressed."""

    coefficients: Py2DArray_c128
    r"""
    The coefficients for the multi-determinantal states in the specified basis.
    Each column of the coefficient matrix contains the coefficients for one state.
    """

    energies: Py1DArray_c128
    r"""The energies of the multi-determinantal states."""

    density_matrices: list[Py2DArray_c128] | None
    r"""The density matrices for the multi-determinantal states in the specified basis, if any."""

    threshold: float
    r"""The threshold for comparisons."""

    def complex_symmetric(self) -> bool:
        r"""Boolean indicating whether inner products involving these multi-determinantal states are complex-symmetric."""

    def state_coefficients(self, state_index: int) -> Py1DArray_c128:
        r"""Returns the coefficients for a particular state."""

    def state_energy(self, state_index: int) -> float:
        r"""Returns the energy for a particular state."""

    def state_density_matrix(self, state_index: int) -> Py2DArray_c128:
        r"""Returns the density matrix for a particular state."""

type PyMultiDeterminants = PyMultiDeterminantsReal | PyMultiDeterminantsComplex

# ----------------------------------------------------------------------------------------
# bindings/python/representation_analysis/multideterminant/multideterminant_eager_basis.rs
# ----------------------------------------------------------------------------------------

def rep_analyse_multideterminants_eager_basis(
    inp_sym: str,
    pydets: list[PySlaterDeterminant],
    coefficients: Py2DArray_f64 | Py2DArray_c128,
    energies: Py1DArray_f64 | Py1DArray_c128,
    pybaos: list[PyBasisAngularOrder],
    integrality_threshold: float,
    linear_independence_threshold: float,
    use_magnetic_group: MagneticSymmetryAnalysisKind | None,
    use_double_group: bool,
    use_cayley_table: bool,
    symmetry_transformation_kind: SymmetryTransformationKind,
    eigenvalue_comparison_mode: EigenvalueComparisonMode,
    sao: Py2DArray_f64 | Py2DArray_c128,
    sao_h: Py2DArray_f64 | Py2DArray_c128 | None = None,
    write_overlap_eigenvalues: bool = True,
    write_character_table: bool = True,
    infinite_order_to_finite: int | None = None,
    angular_function_integrality_threshold: float = 1e-7,
    angular_function_linear_independence_threshold: float = 1e-7,
    angular_function_max_angular_momentum: int = 2,
) -> None:
    r"""
    Python-exposed function to perform representation symmetry analysis for real and complex
    multi-determinantal wavefunctions constructed from an eager basis of Slater determinants and log
    the result via the `qsym2-output` logger at the `INFO` level.

    If `symmetry_transformation_kind` includes spin transformation, the provided
    multi-determinantal wavefunctions will be augmented to generalised spin constraint
    automatically.

    :param inp_sym: A path to the [`QSym2FileType::Sym`] file containing the symmetry-group detection result for the system.
        This will be used to construct abstract groups and character tables for representation analysis.
    :param pydets: A list of Python-exposed Slater determinants whose coefficients are of type `float64` or `complex128`.
        These determinants serve as basis states for non-orthogonal configuration interaction to yield multi-determinantal
        wavefunctions, the symmetry of which will be analysed by this function.
    :param coefficients: The coefficient matrix where each column gives the linear combination coefficients for one multi-determinantal wavefunction.
        The number of rows must match the number of determinants specified in `pydets`.
        The elements are of type `float64` or `complex128`.
    :param energies: The `float64` or `complex128` energies of the multi-determinantal wavefunctions.
        The number of terms must match the number of columns of `coefficients`.
    :param pybaos: Python-exposed structures containing basis angular order information, one for each explicit component per coefficient matrix.
    :param integrality_threshold: The threshold for verifying if subspace multiplicities are integral.
    :param linear_independence_threshold: The threshold for determining the linear independence subspace via the non-zero eigenvalues of the orbit overlap matrix.
    :param use_magnetic_group: An option indicating if the magnetic group is to be used for symmetry analysis, and if so, whether unitary representations or unitary-antiunitary corepresentations should be used.
    :param use_double_group: A boolean indicating if the double group of the prevailing symmetry group is to be used for representation analysis instead.
    :param use_cayley_table: A boolean indicating if the Cayley table for the group, if available, should be used to speed up the calculation of orbit overlap matrices.
    :param symmetry_transformation_kind: An enumerated type indicating the type of symmetry transformations to be performed on the origin determinant to generate the orbit.
        If this contains spin transformation, the determinant will be augmented to generalised spin constraint automatically.
    :param eigenvalue_comparison_mode: An enumerated type indicating the mode of comparison of orbit overlap eigenvalues with the specified `linear_independence_threshold`.
    :param sao: The atomic-orbital overlap matrix whose elements are of type `float64` or `complex128`.
    :param sao_h: The optional complex-symmetric atomic-orbital overlap matrix whose elements are of type `float64` or `complex128`.
        This is required if antiunitary symmetry operations are involved.
    :param write_overlap_eigenvalues: A boolean indicating if the eigenvalues of the determinant orbit overlap matrix are to be written to the output.
    :param write_character_table: A boolean indicating if the character table of the prevailing symmetry group is to be printed out.
    :param infinite_order_to_finite: The finite order with which infinite-order generators are to be interpreted to form a finite subgroup of the prevailing infinite group.
        This finite subgroup will be used for symmetry analysis.
    :param angular_function_integrality_threshold: The threshold for verifying if subspace multiplicities are integral for the symmetry analysis of angular functions.
    :param angular_function_linear_independence_threshold: The threshold for determining the linear independence subspace via the non-zero eigenvalues of the orbit overlap matrix for the symmetry analysis of angular functions.
    :param angular_function_max_angular_momentum: The maximum angular momentum order to be used in angular function symmetry analysis.
    """

# --------------------------------------------------------------------------------------------------------
# bindings/python/representation_analysis/multideterminant/multideterminant_orbit_basis_internal_solver.rs
# --------------------------------------------------------------------------------------------------------

def rep_analyse_multideterminants_orbit_basis_internal_solver(
    inp_sym: str,
    pyorigins: list[PySlaterDeterminant],
    pybaos: list[PyBasisAngularOrder],
    sao: Py2DArray_f64 | Py2DArray_c128,
    enuc: float | complex,
    onee: Py2DArray_f64 | Py2DArray_c128,
    twoe: Py4DArray_f64 | Py4DArray_c128 | None,
    py_get_jk: Callable[
        [Py2DArray_f64 | Py2DArray_c128],
        tuple[Py2DArray_f64, Py2DArray_f64] | tuple[Py2DArray_c128, Py2DArray_c128],
    ]
    | None,
    thresh_offdiag: float,
    thresh_zeroov: float,
    integrality_threshold: float,
    linear_independence_threshold: float,
    use_magnetic_group: MagneticSymmetryAnalysisKind | None,
    use_double_group: bool,
    use_cayley_table: bool,
    symmetry_transformation_kind: SymmetryTransformationKind,
    eigenvalue_comparison_mode: EigenvalueComparisonMode,
    sao_h: Py2DArray_f64 | Py2DArray_c128 | None = None,
    write_overlap_eigenvalues: bool = True,
    write_character_table: bool = True,
    infinite_order_to_finite: int | None = None,
    angular_function_integrality_threshold: float = 1e-7,
    angular_function_linear_independence_threshold: float = 1e-7,
    angular_function_max_angular_momentum: int = 2,
) -> None:
    r"""
    Python-exposed function to run non-orthogonal configuration interaction using the internal
    solver and then perform representation symmetry analysis on the resulting real and complex
    multi-determinantal wavefunctions constructed from group-generated orbits and log the result via
    the `qsym2-output` logger at the `INFO` level.
   
    If `symmetry_transformation_kind` includes spin transformation, the provided
    multi-determinantal wavefunctions with spin constraint structure will be augmented to
    the generalised spin constraint automatically.
   
    :param inp_sym: A path to the [`QSym2FileType::Sym`] file containing the symmetry-group detection result for the system.
        This will be used to construct abstract groups and character tables for representation analysis.
    :param pyorigins: A list of Python-exposed Slater determinants whose coefficients are of type `float64` or `complex128`.
        These determinants serve as origins for group-generated orbits which serve as basis states for non-orthogonal configuration
        interaction to yield multi-determinantal wavefunctions, the symmetry of which will be analysed by this function.
    :param pybaos: Python-exposed structures containing basis angular order information, one for each explicit component per coefficient matrix.
    :param sao: The atomic-orbital overlap matrix whose elements are of type `float64` or `complex128`.
    :param enuc: The nuclear repulsion energy.
    :param onee: The one-electron integral matrix whose elements are of type `float64` or `complex128`.
    :param twoe: The two-electron integral tensor whose elements are of type `float64` or `complex128`.
        Either this or `py_get_jk` must be specified, but not both.
    :param py_get_jk: A Python function callable on a density matrix $`\mathbf{D}`$ to calculate the corresponding $`\mathbf{J}`$ and $`\mathbf{K}`$ matrices by contracting with appropriate two-electron integrals calculated on-the-fly.
        Either this or `twoe` must be specified, but not both.
    :param thresh_offdiag: Threshold for identifying non-zero off-diagonal elements in Löwdin pairing.
    :param thresh_zeroov: Threshold for identifying non-zero overlaps in Löwdin pairing.
    :param integrality_threshold: The threshold for verifying if subspace multiplicities are integral.
    :param linear_independence_threshold: The threshold for determining the linear independence subspace via the non-zero eigenvalues of the orbit overlap matrix.
    :param use_magnetic_group: An option indicating if the magnetic group is to be used for symmetry analysis, and if so, whether unitary representations or unitary-antiunitary corepresentations should be used.
    :param use_double_group: A boolean indicating if the double group of the prevailing symmetry group is to be used for representation analysis instead.
    :param use_cayley_table: A boolean indicating if the Cayley table for the group, if available, should be used to speed up the calculation of orbit overlap matrices.
    :param symmetry_transformation_kind: An enumerated type indicating the type of symmetry transformations to be performed on the origin determinant to generate the orbit.
        If this contains spin transformation, the determinant will be augmented to generalised spin constraint automatically.
    :param eigenvalue_comparison_mode: An enumerated type indicating the mode of comparison of orbit overlap eigenvalues with the specified `linear_independence_threshold`.
    :param sao_h: The optional complex-symmetric atomic-orbital overlap matrix whose elements are of type `float64` or `complex128`.
        This is required if antiunitary symmetry operations are involved.
    :param write_overlap_eigenvalues: A boolean indicating if the eigenvalues of the determinant orbit overlap matrix are to be written to the output.
    :param write_character_table: A boolean indicating if the character table of the prevailing symmetry group is to be printed out.
    :param infinite_order_to_finite: The finite order with which infinite-order generators are to be interpreted to form a finite subgroup of the prevailing infinite group.
        This finite subgroup will be used for symmetry analysis.
    :param angular_function_integrality_threshold: The threshold for verifying if subspace multiplicities are integral for the symmetry analysis of angular functions.
    :param angular_function_linear_independence_threshold: The threshold for determining the linear independence subspace via the non-zero eigenvalues of the orbit overlap matrix for the symmetry analysis of angular functions.
    :param angular_function_max_angular_momentum: The maximum angular momentum order to be used in angular function symmetry analysis.
    """
