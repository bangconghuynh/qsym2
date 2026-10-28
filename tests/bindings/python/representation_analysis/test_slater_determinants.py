import json
import pathlib

import numpy as np
import pytest
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

from ..integral_evaluation.test_integral_evaluation import extract_basis_information


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


def test_rep_analyse_slater_determinant_h62p_err():
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

    sao = np.eye(25)

    pydet = PySlaterDeterminantReal(
        structure_constraint=PySpinConstraint.Restricted,
        complex_symmetric=False,
        coefficients=[ca],
        occupations=[occa],
        threshold=1e-7,
        mo_energies=[ea],
        energy=rhf.e_tot,
    )

    with pytest.raises(RuntimeError):
        # Mismatched SAO dimensions: should be 24-by-24, not 25-by-25
        _ = rep_analyse_slater_determinant(
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


def test_rep_analyse_slater_determinant_h6_orca_json():
    # ------------
    # Data loading
    # ------------
    ROOT = pathlib.Path(__file__).resolve().parents[4]
    with open(f"{ROOT}/tests/orca/h6/h6.json", "r") as f:
        data = json.load(f)
    
    with open(f"{ROOT}/tests/orca/h6/h6.property.json", "r") as f:
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
    
    unisym, _magsym = detect_symmetry_group(
        inp_xyz=None,
        inp_mol=pymol,
        out_sym="h6",
        moi_thresholds=[1e-6],
        distance_thresholds=[1e-6],
        time_reversal=False,
        write_symmetry_elements=False,
        fictitious_magnetic_field=None,
        fictitious_electric_field=None,
    )

    assert unisym.group_name == "Oh"
    
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
    res = rep_analyse_slater_determinant(
        # Data
        inp_sym="h6",
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

    assert res.group == "Oh"
    assert res.determinant_symmetry == "|A|_(1g)"
