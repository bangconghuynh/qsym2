from typing import Optional
import numpy as np
import numpy.typing as npt

from enum import Enum

# ---------------------------
# symmetry_group_detection.rs
# ---------------------------

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
    r""" The ordinary atoms in the molecule. """

    threshold: float
    r""" Threshold for comparing molecules. """

    magnetic_field: tuple[float, float, float] | None
    r""" The uniform external magnetic field, if any. """

    electric_field: tuple[float, float, float] | None
    r""" The uniform external electric field, if any. """

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
    r""" Variant denoting proper symmetry elements. """

    ProperTR = 1
    r""" Variant denoting time-reversed proper symmetry elements. """

    ImproperMirrorPlane = 2
    r""" Variant denoting improper symmetry elements (mirror-plane convention). """

    ImproperMirrorPlaneTR = 3
    r""" Variant denoting time-reversed improper symmetry elements (mirror-plane convention). """

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
    ) -> dict[int, list[npt.NDArray[np.float64]]]:
        r"""
        Returns symmetry elements of all *finite* orders of a given kind.

        # Arguments

        * `kind` - The symmetry element kind.

        # Returns

        A dictionary where the keys are integers indicating the orders of the elements and the values
        are vectors of one-dimensional arrays, each of which gives the axis of a symmetry element.
        If the order value is `-1`, then the associated elements have infinite order.
        """

    def get_generators_of_kind(
        self, kind: PySymmetryElementKind
    ) -> dict[int, list[npt.NDArray[np.float64]]]:
        r"""
        Returns symmetry generators of *finite*  and *infinite* orders of a given kind.

        :param kind: The symmetry generator kind.

        :return: A dictionary where the keys are integers indicating the orders of the generators and the values
        are vectors of one-dimensional arrays, each of which gives the axis of a symmetry generator.
        If the order value is `-1`, then the associated generators have infinite order.
        """

    group_name: str
    r""" The name of the symmetry group. """

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

# --------------------------
# molecule_symmetrisation.rs
# --------------------------

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

# ------------
# integrals.rs
# ------------

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
    r""" Variant for a pure shell. """

    Cartesian = 1
    r""" Variant for a Cartesian shell. """

    SpinorFermion = 2
    r""" Variant for a spinor shell corresponding to a fermion without any additional balance symmetries. """

    SpinorFermionKineticBalance = 3
    r""" Variant for a spinor shell corresponding to a fermion with the kinetic balance symmetry due to $`\mathbf{\sigma} \dot \hat{\mathbf{p}}`$. """

    SpinorAntifermion = 4
    r""" Variant for a spinor shell corresponding to an antifermion without any additional balance symmetries. """

    SpinorAntifermionKineticBalance = 5
    r""" Variant for a spinor shell describing an antifermion with the kinetic balance symmetry due to $`\mathbf{\sigma} \dot \hat{\mathbf{p}}`$. """

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
    r""" Variant for restricted spin constraint. Only two spin spaces are exposed. """

    Unrestricted = 1
    r""" Variant for unrestricted spin constraint. Only two spin spaces arranged in decreasing-$`m`$ order (*i.e.* $`(\alpha, \beta)`$) are exposed. """

    Generalised = 2
    r""" Variant for generalised spin constraint. Only two spin spaces arranged in decreasing-$`m`$ order (*i.e.* $`(\alpha, \beta)`$) are exposed. """

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
) -> npt.NDArray[np.float64]:
    r"""
    Calculates the real-valued two-centre overlap matrix for a basis set.

    :param basis_set: A list of lists of [`PyBasisShellContraction`]. Each inner list contains shells on one atom.

    :return: A two-dimensional array containing the real two-centre overlap values.

    :panic: Panics if any shell contains a finite $`\mathbf{k}`$ vector.
    """

def calc_overlap_2c_complex(
    basis_set: list[list[PyBasisShellContraction]], complex_symmetric: bool
) -> npt.NDArray[np.complex128]:
    r"""
    Calculates the complex-valued two-centre overlap matrix for a basis set.

    :param basis_set: A list of lists of [`PyBasisShellContraction`]. Each inner list contains shells on one atom.
    :param complex_symmetric: A boolean indicating if the complex-symmetric overlap is to be calculated.

    :return: A two-dimensional array containing the complex two-centre overlap values.
    """

def calc_overlap_4c_real(
    basis_set: list[list[PyBasisShellContraction]],
) -> npt.NDArray[np.float64]:
    r"""
    Calculates the real-valued four-centre overlap matrix for a basis set.

    :param basis_set: A list of lists of [`PyBasisShellContraction`]. Each inner list contains shells on one atom.

    :return: A four-dimensional array containing the real two-centre overlap values.

    :panic: Panics if any shell contains a finite $`\mathbf{k}`$ vector.
    """

def calc_overlap_4c_complex(
    basis_set: list[list[PyBasisShellContraction]], complex_symmetric: bool
) -> npt.NDArray[np.complex128]:
    r"""
    Calculates the complex-valued four-centre overlap matrix for a basis set.

    :param basis_set: A list of lists of [`PyBasisShellContraction`]. Each inner list contains shells on one atom.
    :param complex_symmetric: A boolean indicating if the complex-symmetric overlap is to be calculated.

    :return: A four-dimensional array containing the complex four-centre overlap values.
    """

# ---------------------------------------------
# representation_analysis/slater_determinant.rs
# ---------------------------------------------

class PySlaterDeterminantReal:
    r"""
    Python-exposed structure to marshall real Slater determinant information between Rust and Python.
    """

    def __init__(
        self,
        structure_constraint: PyStructureConstraint,
        complex_symmetric: bool,
        coefficients: list[npt.NDArray[np.float64]],
        occupations: list[npt.NDArray[np.float64]],
        threshold: float,
        mo_energies: list[npt.NDArray[np.float64]] | None,
        energy: float | None,
    ) -> None:
        r""" """
