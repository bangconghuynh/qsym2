import logging
import re

import numpy as np
import pytest
from pyscf import gto, scf

from qsym2 import (
    EigenvalueComparisonMode,
    PyMolecule,
    PySlaterDeterminantReal,
    PySpinConstraint,
    SymmetryTransformationKind,
    detect_symmetry_group,
    project_slater_determinant,
    rep_analyse_slater_determinant,
)

from ..integral_evaluation.test_integral_evaluation import extract_basis_information

FLOAT_RE = r"([+-]?\d+(\.\d+)?(e[+-]?\d+)?)"


# pyo3-log caches logger levels on first use, so this must be set before any QSym² call.
logging.getLogger("qsym2-output").setLevel(logging.INFO)


def test_project_slater_determinant_h62p(caplog):
    mol = gto.M(
        atom=r"""
        H         0.00000        0.00000        0.90000
        H         0.00000        0.00000       -0.90000
        H         0.00000        0.90000        0.00000
        H         0.00000       -0.90000        0.00000
        H         0.90000        0.00000        0.00000
        H        -0.90000        0.00000        0.00000
        """,
        unit="Angstrom",
        basis="6-31G",
        charge=2,
        spin=0,
        cart=False,
        verbose=0,
    )

    pymol = PyMolecule(
        atoms=mol._atom,
        threshold=1e-7,
        magnetic_field=None,
        electric_field=None,
    )

    _unisym, _magsym = detect_symmetry_group(
        inp_xyz=None,
        inp_mol=pymol,
        out_sym="mol",
        moi_thresholds=[1e-6],
        distance_thresholds=[1e-6],
        time_reversal=False,
        write_symmetry_elements=False,
        fictitious_magnetic_field=None,
        fictitious_electric_field=None,
    )

    rhf = scf.RHF(mol)
    rhf.conv_tol = 1e-13
    rhf.conv_check = False
    rhf.run()
    stable_i = False
    while not stable_i:
        mo1, _, stable_i, _ = rhf.stability(return_status=True)
        dm1 = rhf.make_rdm1(mo1, rhf.mo_occ)
        rhf = rhf.run(dm1)

    assert rhf.mo_coeff is not None
    ca = rhf.mo_coeff

    assert rhf.mo_occ is not None
    occa = rhf.mo_occ

    assert rhf.mo_energy is not None
    ea: np.ndarray[tuple[int], np.dtype[np.float64]] = rhf.mo_energy

    pybao, _ = extract_basis_information(mol)

    sao: np.ndarray[tuple[int, int], np.dtype[np.float64]] = mol.intor("int1e_ovlp")

    pydet = PySlaterDeterminantReal(
        structure_constraint=PySpinConstraint.Restricted,
        complex_symmetric=False,
        coefficients=[ca],
        occupations=[occa],
        threshold=1e-7,
        mo_energies=[ea],
        energy=rhf.e_tot,
    )

    res = rep_analyse_slater_determinant(
        # Data
        inp_sym="mol",
        pydet=pydet,
        pybaos=[pybao],
        sao=sao,
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

    assert res.group == "Oh"
    assert res.determinant_symmetry == "|A|_(1g) ⊕ |T|_(2g)"

    # ----------
    # Projection
    # ----------
    caplog.clear()
    _irreps, _multidets = project_slater_determinant(
        # General
        inp_sym="mol",
        pydet=pydet,
        projection_targets=["||T|_(2g)|", "||A|_(1g)|"],
        pybaos=[pybao],
        use_magnetic_group=None,
        use_double_group=False,
        symmetry_transformation_kind=SymmetryTransformationKind.Spatial,
        write_character_table=True,
        infinite_order_to_finite=None,
        # Energy
        calculate_energies=True,
        calculate_density_matrices=False,
        sao=sao,
        sao_h=None,
        enuc=rhf.energy_nuc(),
        onee=rhf.get_hcore(),
        twoe=None,
        py_get_jk=lambda dm: rhf.get_jk(dm=dm, hermi=0),
        thresh_offdiag=1e-6,
        thresh_zeroov=1e-6,
    )
    output = "\n".join(
        record.getMessage()
        for record in caplog.records
        if record.name == "qsym2-output"
    )

    lines = output.splitlines()
    start = next(
        i for i, line in enumerate(lines) if "Orbit-based symmetry projection summary" in line
    )
    symmetries = []
    for line in lines[start + 1 :]:
        m = re.match(rf"^\s+(\|.*)\s+{FLOAT_RE}\s+{FLOAT_RE}", line)
        if m is not None:
            symmetries.append((float(m.group(5)), m.group(1).strip()))
        elif len(symmetries) > 0:
            break

    assert symmetries[0][1] == "|T|_(2g)"
    assert np.abs(symmetries[0][0] - (-1.75428981)) < 1e-7

    assert symmetries[1][1] == "|A|_(1g)"
    assert np.abs(symmetries[1][0] - (-1.61158103)) < 1e-7


@pytest.mark.xfail
def test_project_slater_determinant_h62p_err():
    mol = gto.M(
        atom=r"""
        H         0.00000        0.00000        0.90000
        H         0.00000        0.00000       -0.90000
        H         0.00000        0.90000        0.00000
        H         0.00000       -0.90000        0.00000
        H         0.90000        0.00000        0.00000
        H        -0.90000        0.00000        0.00000
        """,
        unit="Angstrom",
        basis="6-31G",
        charge=2,
        spin=0,
        cart=False,
        verbose=0,
    )

    pymol = PyMolecule(
        atoms=mol._atom,
        threshold=1e-7,
        magnetic_field=None,
        electric_field=None,
    )

    _unisym, _magsym = detect_symmetry_group(
        inp_xyz=None,
        inp_mol=pymol,
        out_sym="mol",
        moi_thresholds=[1e-6],
        distance_thresholds=[1e-6],
        time_reversal=False,
        write_symmetry_elements=False,
        fictitious_magnetic_field=None,
        fictitious_electric_field=None,
    )

    rhf = scf.RHF(mol)
    rhf.conv_tol = 1e-13
    rhf.conv_check = False
    rhf.run()

    assert rhf.mo_coeff is not None
    ca = rhf.mo_coeff

    assert rhf.mo_occ is not None
    occa = rhf.mo_occ

    assert rhf.mo_energy is not None
    ea: np.ndarray[tuple[int], np.dtype[np.float64]] = rhf.mo_energy

    pybao, _ = extract_basis_information(mol)

    sao: np.ndarray[tuple[int, int], np.dtype[np.float64]] = mol.intor("int1e_ovlp")

    pydet = PySlaterDeterminantReal(
        structure_constraint=PySpinConstraint.Restricted,
        complex_symmetric=False,
        coefficients=[ca],
        occupations=[occa],
        threshold=1e-7,
        mo_energies=[ea],
        energy=rhf.e_tot,
    )

    # ----------
    # Projection
    # ----------
    # No `||T|_(2)|` in Oh
    # The raised Exception is pyo3_runtime.PanicException, but this cannot be
    # passed into pytest.raises(), so we mark this test as `xfail` instead.
    _irreps, _multidets = project_slater_determinant(
        # General
        inp_sym="mol",
        pydet=pydet,
        projection_targets=["||T|_(2)|"],
        pybaos=[pybao],
        use_magnetic_group=None,
        use_double_group=False,
        symmetry_transformation_kind=SymmetryTransformationKind.Spatial,
        write_character_table=True,
        infinite_order_to_finite=None,
        # Energy
        calculate_energies=True,
        calculate_density_matrices=False,
        sao=sao,
        sao_h=None,
        enuc=rhf.energy_nuc(),
        onee=rhf.get_hcore(),
        twoe=None,
        py_get_jk=lambda dm: rhf.get_jk(dm=dm, hermi=0),
        thresh_offdiag=1e-6,
        thresh_zeroov=1e-6,
    )
