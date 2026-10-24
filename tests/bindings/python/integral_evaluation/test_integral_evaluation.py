import itertools
import re

import numpy as np
from pyscf import gto

from qsym2 import (
    PyBasisAngularOrder,
    PyBasisShellContraction,
    ShellType,
    calc_overlap_2c_complex,
    calc_overlap_2c_real,
    calc_overlap_4c_complex,
    calc_overlap_4c_real,
)


def extract_basis_information(
    mol: gto.Mole,
) -> tuple[PyBasisAngularOrder, list[list[PyBasisShellContraction]]]:
    r"""Extracts AO basis information from PySCF."""
    ANGMOM = ["S", "P", "D", "F", "G", "H", "I"]
    parsed_labels = []
    for label in mol.ao_labels():
        [i, atom, bas] = label.split()
        shell_opt = re.search(r"(\d+[a-z]).*", bas)
        assert shell_opt is not None
        shell = shell_opt.group(1)
        parsed_labels.append((i, atom, shell))

    shells = [shell for shell, _ in itertools.groupby(parsed_labels)]
    pybao = []
    basis_set = []
    shell_index = 0
    for atom, atom_shells in itertools.groupby(shells, lambda shl: (shl[0], shl[1])):
        atom_shell_tuples = []
        atom_bscs = []
        atom_index = int(atom[0])
        for atom_shell in atom_shells:
            shell_code = atom_shell[2]
            angmom_opt = re.search(r"\d+([a-z])", shell_code)
            assert angmom_opt is not None
            angmom = int(ANGMOM.index(angmom_opt.group(1).upper()))
            if angmom == 1:
                basis_shell = (angmom, ShellType.Pure, ([+1, -1, 0], False))
            else:
                basis_shell = (angmom, ShellType.Pure, (True, angmom % 2 == 0))
            atom_shell_tuples.append(basis_shell)

            bsc = PyBasisShellContraction(
                basis_shell=basis_shell,
                primitives=list(
                    zip(mol.bas_exp(shell_index), mol.bas_ctr_coeff(shell_index)[:, 0])
                ),
                cart_origin=mol._atom[atom_index][1],
                k=None,
            )
            atom_bscs.append(bsc)

            shell_index += 1
        pybao.append((atom[1], atom_shell_tuples))
        basis_set.append(atom_bscs)

    return PyBasisAngularOrder(pybao), basis_set


def test_calc_overlap_2c():
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
        verbose=5,
    )
    sao: np.ndarray[tuple[int, int], np.dtype[np.float64]] = mol.intor("int1e_ovlp")
    _bao, basis_set = extract_basis_information(mol)

    sao_qsym2_real = calc_overlap_2c_real(basis_set)
    sao_qsym2_complex = calc_overlap_2c_complex(basis_set, complex_symmetric=False)

    assert np.allclose(sao, sao_qsym2_real, atol=1e-8, rtol=1e-8)
    assert np.allclose(sao, sao_qsym2_complex, atol=1e-8, rtol=1e-8)


def test_calc_overlap_4c():
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
        verbose=5,
    )
    _bao, basis_set = extract_basis_information(mol)

    sao_4c_qsym2_real = calc_overlap_4c_real(basis_set)
    sao_4c_qsym2_complex = calc_overlap_4c_complex(basis_set, complex_symmetric=False)

    assert np.allclose(sao_4c_qsym2_real, sao_4c_qsym2_complex, atol=1e-8, rtol=1e-8)
