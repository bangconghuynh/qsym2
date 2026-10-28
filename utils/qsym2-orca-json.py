import json
import logging

import numpy as np

from qsym2 import (
    EigenvalueComparisonMode,
    PyBasisAngularOrder,
    PyMolecule,
    PySlaterDeterminantReal,
    PySpinConstraint,
    ShellType,
    SymmetryTransformationKind,
    detect_symmetry_group,
    qsym2_output_contributors,
    qsym2_output_heading,
    rep_analyse_slater_determinant,
)


class MyFilter:
    def __init__(self, level):
        self.__level = level

    def filter(self, logRecord):
        return logRecord.levelno == self.__level


# ---------
# ORCA data
# ---------

# Make sure that the following two commands have been run:
# ```
# orca_2json tag.gbw
# orca_2json tag -property
# ```
# with the following configuration file `tag.json.conf`:
# ```
# {
#     "Basisset": true,
#     "MOCoefficients": true,
#     "1elIntegrals": ["S"]
# }
# ```
# This will generate two json files: `tag.json` and `tag.property.json`.
tag = "h6"

# -------
# Logging
# -------
logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.handlers[0].setLevel(logging.WARNING)

output_file_handler = logging.FileHandler(f"{tag}.sym.out", mode="w")
output_file_handler.addFilter(MyFilter(logging.INFO))
logger.addHandler(output_file_handler)

qsym2_output_heading()
qsym2_output_contributors()

# ------------
# Data loading
# ------------
with open(f"{tag}.json", "r") as f:
    data = json.load(f)

with open(f"{tag}.property.json", "r") as f:
    properties = json.load(f)

# ------------------------
# Symmetry-group detection
# ------------------------
atoms = [
    (atom["ElementLabel"], tuple(atom["Coords"])) for atom in data["Molecule"]["Atoms"]
]

pymol = PyMolecule(
    atoms=atoms,
    threshold=1e-7,
    magnetic_field=None,
    electric_field=None,
)

unisym, magsym = detect_symmetry_group(
    inp_xyz=None,
    inp_mol=pymol,
    out_sym=tag,
    moi_thresholds=[1e-6],
    distance_thresholds=[1e-6],
    time_reversal=False,
    write_symmetry_elements=False,
    fictitious_magnetic_field=None,
    fictitious_electric_field=None,
)

# -----------------
# Basis information
# -----------------
# Orca only uses spherical functions, so there is no pure/Cartesian ambiguity
# here.
ANGMOM_LABELS = {"S": 0, "P": 1, "D": 2, "F": 3, "G": 4, "H": 5, "I": 6}

def orca_pure_order(l: int) -> list[int]:
    r"""Generates the Orca order of spherical functions in a given shell.

    The spherical functions in a given shell of angular momentum :math:`l`
    are ordered in Orca according to :math:`0, +1, -1, \ldots, +l, -l`.
    """
    ms = []
    for m in range(l + 1):
        if m == 0:
            ms.append(m)
        else:
            ms.append(m)
            ms.append(-m)
    return ms

bao = [
    (
        atom["ElementLabel"],
        [
            (
                ANGMOM_LABELS[bshl["Shell"].upper()],
                ShellType.Pure,
                (
                    orca_pure_order(ANGMOM_LABELS[bshl["Shell"].upper()]),
                    ANGMOM_LABELS[bshl["Shell"].upper()] % 2 == 0,
                ),
            )
            for bshl in atom["Basis"]
        ],
    )
    for atom in data["Molecule"]["Atoms"]
]

pybao = PyBasisAngularOrder(bao)
sao_spatial: np.ndarray[tuple[int, int], np.dtype[np.float64]] = np.array(
    data["Molecule"]["S-Matrix"]
)

# ---------------------
# SCF state information
# ---------------------
c_all = np.array(
    [mo["MOCoefficients"] for mo in data["Molecule"]["MolecularOrbitals"]["MOs"]]
).T
occ_all = np.array(
    [mo["Occupancy"] for mo in data["Molecule"]["MolecularOrbitals"]["MOs"]]
)
mo_energy_all = np.array(
    [mo["OrbitalEnergy"] for mo in data["Molecule"]["MolecularOrbitals"]["MOs"]]
)

# Find indices where the occupation number changes from 0 to non-zero
spin_boundaries = np.concatenate(
    ([0], np.where((occ_all[:-1] == 0) & (occ_all[1:] != 0))[0] + 1, [len(occ_all)])
)

cs = []
occs = []
mo_energies = []
for i, spin_start in enumerate(spin_boundaries[:-1]):
    spin_end = spin_boundaries[i + 1]
    cs.append(c_all[:, spin_start:spin_end])
    occs.append(occ_all[spin_start:spin_end])
    mo_energies.append(mo_energy_all[spin_start:spin_end])

if data["Molecule"]["HFTyp"] == "UHF":
    spincons = PySpinConstraint.Unrestricted
elif data["Molecule"]["HFTyp"] == "RHF":
    spincons = PySpinConstraint.Restricted
else:
    raise RuntimeError("Unknown spin constraint.")

last_geometry_scf_energy_index = next(
    (
        i
        for i, energy_data in enumerate(properties["Geometries"][-1]["Energy"])
        if energy_data["Method"] == "SCF"
    ),
    None,
)
last_geometry_scf_energy = properties["Geometries"][-1]["Energy"][
    last_geometry_scf_energy_index
]["totalEnergy"][0][0]

pydet = PySlaterDeterminantReal(
    structure_constraint=spincons,
    complex_symmetric=False,
    coefficients=cs,
    occupations=occs,
    threshold=1e-7,
    mo_energies=mo_energies,
    energy=last_geometry_scf_energy,
)

# ---------------------------
# SCF representation analysis
# ---------------------------
rep_analyse_slater_determinant(
    # Data
    inp_sym=tag,
    pydet=pydet,
    pybaos=[pybao],
    sao=sao_spatial,
    sao_h=None,
    sao_spatial_4c=None,
    sao_spatial_4c_h=None,
    # Thresholds
    linear_independence_threshold=1e-7,
    integrality_threshold=1e-7,
    orbit_overlap_symmetry_threshold=1e-7,
    eigenvalue_comparison_mode=EigenvalueComparisonMode.Modulus,
    # Analysis options
    use_magnetic_group=None,
    use_double_group=False,
    use_cayley_table=True,
    symmetry_transformation_kind=SymmetryTransformationKind.Spatial,
    infinite_order_to_finite=4,
    # Other options
    write_character_table=True,
    write_overlap_eigenvalues=True,
    analyse_mo_symmetries=True,
    analyse_mo_mirror_parities=False,
    analyse_density_symmetries=False,
)
