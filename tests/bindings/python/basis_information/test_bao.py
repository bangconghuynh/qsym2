import itertools
import pathlib
import re

from pyscf import gto

from qsym2 import PyBasisAngularOrder, ShellType


def test_bao_from_pyscf_spatial():
    mol = gto.M(
        atom=r"""
            C   -0.0000000    0.0000000    0.0000000
            H    0.6405128   -0.6405128    0.6405128
            H    0.6405128    0.6405128   -0.6405128
            H   -0.6405128    0.6405128    0.6405128
            H   -0.6405128   -0.6405128   -0.6405128
        """,
        unit="Angstrom",
        basis="6-31G*",
        charge=1,
        spin=1,
        cart=False,
    )

    def bao_from_mol(mol: gto.Mole) -> PyBasisAngularOrder:
        ANGMOM = ["S", "P", "D", "F", "G", "H", "I"]
        labels = mol.ao_labels()
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

    _ = bao_from_mol(mol)

    assert True


def test_bao_from_pyscf_spinorbitcoupled():
    mol = gto.M(
        atom=r"""
            C   -0.0000000    0.0000000    0.0000000
            H    0.6405128   -0.6405128    0.6405128
            H    0.6405128    0.6405128   -0.6405128
            H   -0.6405128    0.6405128    0.6405128
            H   -0.6405128   -0.6405128   -0.6405128
        """,
        unit="Angstrom",
        basis="6-31G*",
        charge=1,
        spin=1,
        cart=False,
    )

    def bao_from_mol(
        mol: gto.Mole,
    ) -> tuple[PyBasisAngularOrder, PyBasisAngularOrder]:
        ANGMOM = ["S", "P", "D", "F", "G", "H", "I"]
        parsed_labels = []
        for label in mol.spinor_labels():
            [i, atom, bas] = label.split()
            shell_opt = re.search(r"(.+),", bas)
            assert shell_opt is not None
            shell = shell_opt.group(1)
            parsed_labels.append((i, atom, shell))

        shells = [shell for shell, _ in itertools.groupby(parsed_labels)]
        pybao = []
        pybao_sp = []
        exps = []
        ctr_coeffs = []
        for shell_index in range(mol.nbas):
            for coeff in mol.bas_ctr_coeff(shell_index).T:
                exps.append(mol.bas_exp(shell_index))
                ctr_coeffs.append(coeff)
        shell_index = 0
        for atom, atom_shells in itertools.groupby(shells, lambda shl: (shl[0], shl[1])):
            atom_shell_tuples = []
            atom_shell_tuples_sp = []
            for atom_shell in atom_shells:
                shell_code = atom_shell[2]
                angmom_opt = re.search(r"([a-z])(\d+)\/(\d+)", shell_code)
                assert angmom_opt is not None
                spatial_index = ANGMOM.index(angmom_opt.group(1).upper())
                angmom = angmom_opt.group(2)
                basis_shell = (
                    int(angmom),
                    ShellType.SpinorFermion,
                    (True, spatial_index % 2 == 0),
                )
                basis_shell_sp = (
                    int(angmom),
                    ShellType.SpinorAntifermionKineticBalance,
                    (True, spatial_index % 2 == 0),
                )
                atom_shell_tuples.append(basis_shell)
                atom_shell_tuples_sp.append(basis_shell_sp)

                shell_index += 1
            pybao.append((atom[1], atom_shell_tuples))
            pybao_sp.append((atom[1], atom_shell_tuples_sp))

        return PyBasisAngularOrder(pybao), PyBasisAngularOrder(pybao_sp)

    _, _ = bao_from_mol(mol)

    assert True


def test_bao_from_qchem_archive():
    ROOT = pathlib.Path(__file__).resolve().parents[3]
    qchem_archive = f"{ROOT}/qchem/benzene_631gstar.qarchive.h5"
    _ = PyBasisAngularOrder.from_qchem_archive(qchem_archive)

    assert True
