import itertools
import os
import re

import numpy as np
from pyscf import gto, scf

from qsym2 import (
    EigenvalueComparisonMode,
    PyBasisAngularOrder,
    PyMolecule,
    PySlaterDeterminantReal,
    PySpinConstraint,
    ShellType,
    SymmetryTransformationKind,
    detect_symmetry_group,
    rep_analyse_slater_determinant,
)


def test_rep_analyse_slater_determinant_h62p():
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
        mo1 = rhf.stability()[0]
        dm1 = rhf.make_rdm1(mo1, rhf.mo_occ)
        rhf = rhf.run(dm1)

    assert rhf.mo_coeff is not None
    ca = rhf.mo_coeff

    assert rhf.mo_occ is not None
    occa = rhf.mo_occ

    assert rhf.mo_energy is not None
    ea: np.ndarray[tuple[int], np.dtype[np.float64]] = rhf.mo_energy


    def bao_from_mol(mol: gto.Mole) -> PyBasisAngularOrder:
        labels = mol.ao_labels()

        ANGMOM = ["S", "P", "D", "F", "G", "H", "I"]
        parsed_labels = []
        for label in labels:
            [i, atom, bas] = label.split()
            shell_opt = re.search(r"(\d+[a-z]).*", bas)
            assert shell_opt is not None
            shell = shell_opt.group(1)
            parsed_labels.append((i, atom, shell))

        shells = [shell for shell, _ in itertools.groupby(parsed_labels)]
        pybao = []
        for atom, atom_shells in itertools.groupby(shells, lambda shl: (shl[0], shl[1])):
            atom_shell_tuples = []
            for atom_shell in atom_shells:
                shell_code = atom_shell[2]
                angmom_opt = re.search(r"\d+([a-z])", shell_code)
                assert angmom_opt is not None
                angmom = int(ANGMOM.index(angmom_opt.group(1).upper()))
                if angmom == 1:
                    atom_shell_tuples.append((angmom, ShellType.Pure, ([+1, -1, 0], False)))
                else:
                    atom_shell_tuples.append(
                        (angmom, ShellType.Pure, (True, angmom % 2 == 0))
                    )
            pybao.append((atom[1], atom_shell_tuples))

        return PyBasisAngularOrder(pybao)

    pybao = bao_from_mol(mol)

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

    os.remove("mol.qsym2.sym")
