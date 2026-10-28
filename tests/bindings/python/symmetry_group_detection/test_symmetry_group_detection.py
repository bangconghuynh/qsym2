import pathlib

import pytest
from pyscf import gto

from qsym2 import PyMolecule, PySymmetryElementKind, detect_symmetry_group


def test_symmetry_group_detection_mol():
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

    pymol = PyMolecule(
        atoms=mol._atom,
        threshold=1e-7,
        magnetic_field=None,
        electric_field=None,
    )

    unisym, magsym = detect_symmetry_group(
        inp_xyz=None,
        inp_mol=pymol,
        out_sym="mol",
        moi_thresholds=[1e-6],
        distance_thresholds=[1e-6],
        time_reversal=True,
        write_symmetry_elements=True,
        fictitious_magnetic_field=None,
        fictitious_electric_field=None,
    )

    assert unisym.group_name == "Td"
    assert not unisym.is_infinite()

    assert len(unisym.get_elements_of_kind(PySymmetryElementKind.Proper)[2]) == 3
    assert len(unisym.get_elements_of_kind(PySymmetryElementKind.Proper)[3]) == 4
    assert len(unisym.get_elements_of_kind(PySymmetryElementKind.ImproperMirrorPlane)[1]) == 6
    assert len(unisym.get_elements_of_kind(PySymmetryElementKind.ImproperMirrorPlane)[4]) == 3

    assert len(unisym.get_generators_of_kind(PySymmetryElementKind.Proper)[3]) == 4
    assert len(unisym.get_generators_of_kind(PySymmetryElementKind.ImproperMirrorPlane)[1]) == 1

    assert magsym is not None
    assert magsym.group_name == "Td + θ·Td"


def test_symmetry_group_detection_xyz():
    ROOT = pathlib.Path(__file__).resolve().parents[4]
    unisym, magsym = detect_symmetry_group(
        inp_xyz=f"{ROOT}/tests/xyz/c60.xyz",
        inp_mol=None,
        out_sym="mol",
        moi_thresholds=[1e-6],
        distance_thresholds=[1e-6],
        time_reversal=True,
        write_symmetry_elements=True,
        fictitious_magnetic_field=(0.0, 0.0, 1.0),
        fictitious_electric_field=None,
    )

    assert unisym.group_name == "S10"
    assert not unisym.is_infinite()

    assert len(unisym.get_elements_of_kind(PySymmetryElementKind.Proper)[5]) == 1
    assert len(unisym.get_elements_of_kind(PySymmetryElementKind.ImproperMirrorPlane)[2]) == 1
    assert len(unisym.get_elements_of_kind(PySymmetryElementKind.ImproperMirrorPlane)[10]) == 1

    assert magsym is not None
    assert magsym.group_name == "D5d"
    assert len(magsym.get_elements_of_kind(PySymmetryElementKind.Proper)[5]) == 1
    assert len(magsym.get_elements_of_kind(PySymmetryElementKind.ProperTR)[2]) == 5
    assert len(magsym.get_elements_of_kind(PySymmetryElementKind.ImproperMirrorPlaneTR)[1]) == 5
    assert len(magsym.get_elements_of_kind(PySymmetryElementKind.ImproperMirrorPlane)[2]) == 1
    assert len(magsym.get_elements_of_kind(PySymmetryElementKind.ImproperMirrorPlane)[10]) == 1


def test_symmetry_group_detection_err():
    with pytest.raises(RuntimeError):
        _, _ = detect_symmetry_group(
            inp_xyz=None,
            inp_mol=None,
            out_sym="mol",
            moi_thresholds=[1e-6],
            distance_thresholds=[1e-6],
            time_reversal=True,
            write_symmetry_elements=True,
            fictitious_magnetic_field=(0.0, 0.0, 1.0),
            fictitious_electric_field=None,
        )

    with pytest.raises(RuntimeError):
        ROOT = pathlib.Path(__file__).resolve().parents[4]
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

        pymol = PyMolecule(
            atoms=mol._atom,
            threshold=1e-7,
            magnetic_field=None,
            electric_field=None,
        )

        _, _ = detect_symmetry_group(
            inp_xyz=f"{ROOT}/tests/xyz/c60.xyz",
            inp_mol=pymol,
            out_sym="mol",
            moi_thresholds=[1e-6],
            distance_thresholds=[1e-6],
            time_reversal=True,
            write_symmetry_elements=True,
            fictitious_magnetic_field=(0.0, 0.0, 1.0),
            fictitious_electric_field=None,
        )
