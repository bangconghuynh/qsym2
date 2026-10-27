import itertools
import logging
import os
import re

import numpy as np
from pyscf import fci, gto, scf

from qsym2 import (
    EigenvalueComparisonMode,
    PyBasisAngularOrder,
    PyMolecule,
    PySlaterDeterminantReal,
    PySpinConstraint,
    ShellType,
    SymmetryTransformationKind,
    detect_symmetry_group,
    rep_analyse_multideterminants_eager_basis,
    rep_analyse_multideterminants_fci_basis,
    rep_analyse_multideterminants_orbit_basis_internal_solver,
    rep_analyse_slater_determinant,
)

FLOAT_RE = r"([+-]?\d+(\.\d+)?(e[+-]?\d+)?)"


# pyo3-log caches logger levels on first use, so this must be set before any QSym² call.
logging.getLogger("qsym2-output").setLevel(logging.INFO)


def test_rep_analyse_multideterminants_orbit_basis_internal_solver_h62p(caplog):
    try:
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

        # ----
        # NOCI
        # ----
        caplog.clear()
        _ = rep_analyse_multideterminants_orbit_basis_internal_solver(
            inp_sym="mol",
            pyorigins=[pydet],
            pybaos=[pybao],
            sao=sao,
            enuc=rhf.energy_nuc(),
            onee=rhf.get_hcore(),
            twoe=None,
            py_get_jk=lambda dm: rhf.get_jk(dm=dm, hermi=0),
            thresh_offdiag=1e-6,
            thresh_zeroov=1e-6,
            calculate_density_matrices=True,
            integrality_threshold=1e-7,
            linear_independence_threshold=1e-7,
            orbit_overlap_symmetry_threshold=1e-7,
            use_magnetic_group=None,
            use_double_group=False,
            use_cayley_table=True,
            symmetry_transformation_kind=SymmetryTransformationKind.Spatial,
            eigenvalue_comparison_mode=EigenvalueComparisonMode.Modulus,
            sao_h=None,
            write_overlap_eigenvalues=True,
            write_character_table=True,
            infinite_order_to_finite=None,
            angular_function_integrality_threshold=1e-7,
            angular_function_linear_independence_threshold=1e-7,
            angular_function_max_angular_momentum=2,
        )
        output = "\n".join(
            record.getMessage()
            for record in caplog.records
            if record.name == "qsym2-output"
        )

        lines = output.splitlines()
        start = next(
            i for i, line in enumerate(lines) if "> Multi-determinantal results" in line
        )
        symmetries = []
        for line in lines[start + 1 :]:
            m = re.match(rf"^\s*\d+\s+{FLOAT_RE}\s+(\|.*)\s+{FLOAT_RE}\s+{FLOAT_RE}", line)
            if m is not None:
                symmetries.append((float(m.group(1)), m.group(4).strip()))
            elif len(symmetries) > 0:
                break

        assert symmetries[0][1] == "|T|_(2g)"
        assert np.abs(symmetries[0][0] - (-1.75428981)) < 1e-7

        assert symmetries[1][1] == "|T|_(2g)"
        assert np.abs(symmetries[1][0] - (-1.75428981)) < 1e-7

        assert symmetries[2][1] == "|T|_(2g)"
        assert np.abs(symmetries[2][0] - (-1.75428981)) < 1e-7

        assert symmetries[3][1] == "|A|_(1g)"
        assert np.abs(symmetries[3][0] - (-1.61158103)) < 1e-7

    finally:
        os.remove("mol.qsym2.sym")


def test_rep_analyse_multideterminants_fci_basis_h4(caplog):
    try:
        d = 1.000
        mol = gto.M(
            atom=fr"""
            H    {+d/np.sqrt(3):+.7f}   {+d/np.sqrt(3):+.7f}   {+d/np.sqrt(3):+.7f}
            H    {-d/np.sqrt(3):+.7f}   {-d/np.sqrt(3):+.7f}   {+d/np.sqrt(3):+.7f}
            H    {-d/np.sqrt(3):+.7f}   {+d/np.sqrt(3):+.7f}   {-d/np.sqrt(3):+.7f}
            H    {+d/np.sqrt(3):+.7f}   {-d/np.sqrt(3):+.7f}   {-d/np.sqrt(3):+.7f}
            """,
            unit="Angstrom",
            basis="STO-3G",
            charge=0,
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

        uhf = scf.UHF(mol)
        uhf.conv_tol = 1e-13
        uhf.conv_check = False
        uhf.run()

        assert uhf.mo_coeff is not None
        ca, cb = uhf.mo_coeff[0], uhf.mo_coeff[1]

        assert uhf.mo_occ is not None
        occa, occb = uhf.mo_occ[0], uhf.mo_occ[1]

        assert uhf.mo_energy is not None
        mo_ea, mo_eb = uhf.mo_energy[0], uhf.mo_energy[1]

        e_tot = uhf.e_tot

        norb_a = ca.shape[1]
        norb_b = cb.shape[1]
        nelec_a, nelec_b = mol.nelec
        strings_a = fci.cistring.make_strings(range(norb_a), nelec_a)
        strings_b = fci.cistring.make_strings(range(norb_b), nelec_b)
        occ_bits_a = np.asarray([[(s >> i) & 1 for i in range(norb_a)] for s in strings_a], dtype=np.float64)
        occ_bits_b = np.asarray([[(s >> i) & 1 for i in range(norb_b)] for s in strings_b], dtype=np.float64)
        occs = list(itertools.product(occ_bits_a, occ_bits_b))
        nocc_a = len(occ_bits_a)
        nocc_b = len(occ_bits_b)


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

        pydet_uhf = PySlaterDeterminantReal(
            structure_constraint=PySpinConstraint.Unrestricted,
            complex_symmetric=False,
            coefficients=[ca, cb],
            occupations=[occa, occb],
            threshold=1e-7,
            mo_energies=[mo_ea, mo_eb],
            energy=e_tot,
        )

        # ---
        # FCI
        # ---
        cisolver = fci.FCI(uhf)
        es_fci, fcivec_fci = cisolver.kernel(nroots=300)
        es_fci = np.asarray(es_fci).reshape(-1)
        fcivec_fci = np.asarray(fcivec_fci).reshape(-1, nocc_a * nocc_b).T

        caplog.clear()
        rep_analyse_multideterminants_fci_basis(
            # Data
            inp_sym="mol",
            pydet=pydet_uhf,
            coefficients=fcivec_fci,
            energies=es_fci,
            occupation_patterns=occs,
            pybaos=[pybao],
            sao=sao,
            sao_h=None,
            # Thresholds
            integrality_threshold=1e-7,
            linear_independence_threshold=1e-7,
            orbit_overlap_symmetry_threshold=1e-7,
            # Analysis options
            use_magnetic_group=None,
            use_double_group=False,
            use_cayley_table=True,
            symmetry_transformation_kind=SymmetryTransformationKind.Spatial,
            eigenvalue_comparison_mode=EigenvalueComparisonMode.Modulus,
            infinite_order_to_finite=None,
            # Other options
            write_overlap_eigenvalues=True,
            write_character_table=True,
        )
        output = "\n".join(
            record.getMessage()
            for record in caplog.records
            if record.name == "qsym2-output"
        )

        lines = output.splitlines()
        start = next(
            i for i, line in enumerate(lines) if "> Multi-determinantal results" in line
        )
        symmetries = []
        for line in lines[start + 1 :]:
            m = re.match(rf"^\s*\d+\s+{FLOAT_RE}\s+(\|.*)\s+{FLOAT_RE}\s+{FLOAT_RE}", line)
            if m is not None:
                symmetries.append((float(m.group(1)), m.group(4).strip()))
            elif len(symmetries) > 0:
                break

        assert len(symmetries) == 36

        assert symmetries[0][1] == "|E|"
        assert np.abs(symmetries[0][0] - (-1.86690337)) < 1e-7

        assert symmetries[1][1] == "|E|"
        assert np.abs(symmetries[1][0] - (-1.86690337)) < 1e-7

        assert symmetries[2][1] == "|T|_(1)"
        assert np.abs(symmetries[2][0] - (-1.84218096)) < 1e-7

        assert symmetries[3][1] == "|T|_(1)"
        assert np.abs(symmetries[3][0] - (-1.84218096)) < 1e-7

        assert symmetries[4][1] == "|T|_(1)"
        assert np.abs(symmetries[4][0] - (-1.84218096)) < 1e-7

    finally:
        os.remove("mol.qsym2.sym")


def test_rep_analyse_multideterminants_eager_basis_h4(caplog):
    try:
        d = 1.000
        mol = gto.M(
            atom=fr"""
            H    {+d/np.sqrt(3):+.7f}   {+d/np.sqrt(3):+.7f}   {+d/np.sqrt(3):+.7f}
            H    {-d/np.sqrt(3):+.7f}   {-d/np.sqrt(3):+.7f}   {+d/np.sqrt(3):+.7f}
            H    {-d/np.sqrt(3):+.7f}   {+d/np.sqrt(3):+.7f}   {-d/np.sqrt(3):+.7f}
            H    {+d/np.sqrt(3):+.7f}   {-d/np.sqrt(3):+.7f}   {-d/np.sqrt(3):+.7f}
            """,
            unit="Angstrom",
            basis="STO-3G",
            charge=0,
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

        uhf = scf.UHF(mol)
        uhf.conv_tol = 1e-13
        uhf.conv_check = False
        uhf.run()

        assert uhf.mo_coeff is not None
        ca, cb = uhf.mo_coeff[0], uhf.mo_coeff[1]

        assert uhf.mo_occ is not None
        occa, occb = uhf.mo_occ[0], uhf.mo_occ[1]

        assert uhf.mo_energy is not None
        mo_ea, mo_eb = uhf.mo_energy[0], uhf.mo_energy[1]

        e_tot = uhf.e_tot

        norb_a = ca.shape[1]
        norb_b = cb.shape[1]
        nelec_a, nelec_b = mol.nelec
        strings_a = fci.cistring.make_strings(range(norb_a), nelec_a)
        strings_b = fci.cistring.make_strings(range(norb_b), nelec_b)
        occ_bits_a = np.asarray([[(s >> i) & 1 for i in range(norb_a)] for s in strings_a], dtype=np.float64)
        occ_bits_b = np.asarray([[(s >> i) & 1 for i in range(norb_b)] for s in strings_b], dtype=np.float64)
        occs = list(itertools.product(occ_bits_a, occ_bits_b))
        nocc_a = len(occ_bits_a)
        nocc_b = len(occ_bits_b)


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

        pydets = [
            PySlaterDeterminantReal(
                structure_constraint=PySpinConstraint.Unrestricted,
                complex_symmetric=False,
                coefficients=[ca, cb],
                occupations=occ,
                threshold=1e-7,
                mo_energies=[mo_ea, mo_eb],
                energy=e_tot,
            )
            for occ in occs
        ]

        # ---
        # FCI
        # ---
        cisolver = fci.FCI(uhf)
        es_fci, fcivec_fci = cisolver.kernel(nroots=300)
        es_fci = np.asarray(es_fci).reshape(-1)
        fcivec_fci = np.asarray(fcivec_fci).reshape(-1, nocc_a * nocc_b).T

        caplog.clear()
        rep_analyse_multideterminants_eager_basis(
            # Data
            inp_sym="mol",
            pydets=pydets,
            coefficients=fcivec_fci,
            energies=es_fci,
            pybaos=[pybao],
            sao=sao,
            sao_h=None,
            # Thresholds
            integrality_threshold=1e-7,
            linear_independence_threshold=1e-7,
            orbit_overlap_symmetry_threshold=1e-7,
            # Analysis options
            use_magnetic_group=None,
            use_double_group=False,
            use_cayley_table=True,
            symmetry_transformation_kind=SymmetryTransformationKind.Spatial,
            eigenvalue_comparison_mode=EigenvalueComparisonMode.Modulus,
            infinite_order_to_finite=None,
            # Other options
            write_overlap_eigenvalues=True,
            write_character_table=True,
        )
        output = "\n".join(
            record.getMessage()
            for record in caplog.records
            if record.name == "qsym2-output"
        )

        lines = output.splitlines()
        start = next(
            i for i, line in enumerate(lines) if "> Multi-determinantal results" in line
        )
        symmetries = []
        for line in lines[start + 1 :]:
            m = re.match(rf"^\s*\d+\s+{FLOAT_RE}\s+(\|.*)\s+{FLOAT_RE}\s+{FLOAT_RE}", line)
            if m is not None:
                symmetries.append((float(m.group(1)), m.group(4).strip()))
            elif len(symmetries) > 0:
                break

        assert len(symmetries) == 36

        assert symmetries[0][1] == "|E|"
        assert np.abs(symmetries[0][0] - (-1.86690337)) < 1e-7

        assert symmetries[1][1] == "|E|"
        assert np.abs(symmetries[1][0] - (-1.86690337)) < 1e-7

        assert symmetries[2][1] == "|T|_(1)"
        assert np.abs(symmetries[2][0] - (-1.84218096)) < 1e-7

        assert symmetries[3][1] == "|T|_(1)"
        assert np.abs(symmetries[3][0] - (-1.84218096)) < 1e-7

        assert symmetries[4][1] == "|T|_(1)"
        assert np.abs(symmetries[4][0] - (-1.84218096)) < 1e-7

    finally:
        os.remove("mol.qsym2.sym")
