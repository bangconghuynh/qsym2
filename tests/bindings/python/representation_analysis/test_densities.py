import itertools
import os
import re

from pyscf import gto, scf

from qsym2 import (
    EigenvalueComparisonMode,
    PyBasisAngularOrder,
    PyBasisShellContraction,
    PyDensityReal,
    PyMolecule,
    ShellType,
    SymmetryTransformationKind,
    calc_overlap_4c_real,
    detect_symmetry_group,
    rep_analyse_densities,
)


def test_rep_analyse_densities_h62p():
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
    dm_rhf = rhf.make_rdm1()

    uhf = scf.UHF(mol)
    uhf.conv_tol = 1e-13
    uhf.conv_check = False
    uhf.run()
    dm_uhf = uhf.make_rdm1()

    pydens = [
        (
            "rhf_density", PyDensityReal(
                complex_symmetric=False,
                density_matrix=dm_rhf,
                threshold=1e-7,
            )
        ),
        (
            "uhf_density_total", PyDensityReal(
                complex_symmetric=False,
                density_matrix=dm_uhf[0] + dm_uhf[1],
                threshold=1e-7,
            )
        ),
        (
            "uhf_density_spin", PyDensityReal(
                complex_symmetric=False,
                density_matrix=dm_uhf[0] - dm_uhf[1],
                threshold=1e-7,
            )
        ),
    ]


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

    pybao, basis_set = extract_basis_information(mol)

    sao_spatial_4c = calc_overlap_4c_real(basis_set)

    rep_analyse_densities(
        # Data
        inp_sym="mol",
        pydens=pydens,
        pybao=pybao,
        sao_spatial_4c=sao_spatial_4c,
        sao_spatial_4c_h=None,
        # Thresholds
        linear_independence_threshold=1e-7,
        integrality_threshold=1e-7,
        orbit_overlap_symmetry_threshold=1e-7,
        eigenvalue_comparison_mode=EigenvalueComparisonMode.Real,
        # Analysis options
        use_magnetic_group=None,
        use_double_group=False,
        use_cayley_table=True,
        symmetry_transformation_kind=SymmetryTransformationKind.Spatial,
        infinite_order_to_finite=None,
        # Other options
        write_character_table=True,
    )

    os.remove("mol.qsym2.sym")

    assert True
