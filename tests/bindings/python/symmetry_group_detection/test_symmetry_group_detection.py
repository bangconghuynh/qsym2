import os

from pyscf import gto

from qsym2 import PyMolecule, PySymmetryElementKind, detect_symmetry_group


def test_symmetry_group_detection():
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

    os.remove("mol.qsym2.sym")
