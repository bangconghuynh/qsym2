import os

from qsym2 import PyMolecule, detect_symmetry_group, symmetrise_molecule


def test_symmetrisation_vf6_by():
    pymol = PyMolecule(
        atoms=[
            ("V", (+0.0, +0.0, +0.0)),
            ("F", (+1.0, +0.0, +0.0)),
            ("F", (-1.0, +0.0, +0.0)),
            ("F", (+0.0, +1.0, +0.0)),
            ("F", (+0.0, -1.0, +0.0)),
            ("F", (+0.0, +0.0, +1.1)),
            ("F", (+0.0, +0.0, -1.0)),
        ],
        threshold=1e-7,
        magnetic_field=(0.0, 0.1, 0.0),
        electric_field=None,
    )
    unisym, magsym = detect_symmetry_group(
        inp_xyz=None,
        inp_mol=pymol,
        out_sym="mol",
        moi_thresholds=[1e-8],
        distance_thresholds=[1e-8],
        time_reversal=True,
        write_symmetry_elements=True,
        fictitious_magnetic_field=None,
        fictitious_electric_field=None,
    )
    assert unisym.group_name == "Cs"
    assert magsym is not None
    assert magsym.group_name == "C2v"

    symmol = symmetrise_molecule(
        inp_xyz=None,
        inp_mol=pymol,
        out_target_sym=None,
        loose_moi_threshold=1e-1,
        loose_distance_threshold=1.5e-1,
        target_moi_threshold=1e-8,
        target_distance_threshold=1e-8,
        use_magnetic_group=False,
        reorientate_molecule=True,
        max_iterations=50,
        consistent_target_symmetry_iterations=10,
        verbose=2,
    )
    unisym, magsym = detect_symmetry_group(
        inp_xyz=None,
        inp_mol=symmol,
        out_sym="mol",
        moi_thresholds=[1e-8],
        distance_thresholds=[1e-8],
        time_reversal=True,
        write_symmetry_elements=True,
        fictitious_magnetic_field=None,
        fictitious_electric_field=None,
    )
    assert unisym.group_name == "C4h"
    assert magsym is not None
    assert magsym.group_name == "D4h"

    os.remove("mol.qsym2.sym")
