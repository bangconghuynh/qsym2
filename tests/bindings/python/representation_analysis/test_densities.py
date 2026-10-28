import logging
import re

import numpy as np
from pyscf import gto, scf

from qsym2 import (
    EigenvalueComparisonMode,
    PyDensityComplex,
    PyDensityReal,
    PyMolecule,
    SymmetryTransformationKind,
    calc_overlap_4c_complex,
    calc_overlap_4c_real,
    detect_symmetry_group,
    rep_analyse_densities,
)

from ..integral_evaluation.test_integral_evaluation import extract_basis_information

FLOAT_RE = r"([+-]?\d+(\.\d+)?(e[+-]?\d+)?)"


# pyo3-log caches logger levels on first use, so this must be set before any QSym² call.
logging.getLogger("qsym2-output").setLevel(logging.INFO)


def test_rep_analyse_densities_h62p(caplog):
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
    rhf_stable_i = False
    while not rhf_stable_i:
        mo1, _, rhf_stable_i, _ = rhf.stability(return_status=True)
        dm1 = rhf.make_rdm1(mo1, rhf.mo_occ)
        rhf = rhf.run(dm1)
    dm_rhf = rhf.make_rdm1()

    uhf = scf.UHF(mol)
    uhf.conv_tol = 1e-13
    uhf.conv_check = False
    uhf.run()
    uhf_stable_i = False
    while not uhf_stable_i:
        mo1, _, uhf_stable_i, _ = uhf.stability(return_status=True)
        dm1 = uhf.make_rdm1(mo1, uhf.mo_occ)
        uhf = uhf.run(dm1)
    dm_uhf = uhf.make_rdm1()

    pydens_real = [
        (
            "rhf_density_real", PyDensityReal(
                complex_symmetric=False,
                density_matrix=dm_rhf,
                threshold=1e-7,
            )
        ),
        (
            "uhf_density_total_real", PyDensityReal(
                complex_symmetric=False,
                density_matrix=dm_uhf[0] + dm_uhf[1],
                threshold=1e-7,
            )
        ),
        (
            "uhf_density_spin_real", PyDensityReal(
                complex_symmetric=False,
                density_matrix=dm_uhf[0] - dm_uhf[1],
                threshold=1e-7,
            )
        ),
    ]
    pydens_complex = [
        (
            "rhf_density_complex", PyDensityComplex(
                complex_symmetric=False,
                density_matrix=dm_rhf.astype(np.complex128),
                threshold=1e-7,
            )
        ),
        (
            "uhf_density_total_complex", PyDensityComplex(
                complex_symmetric=False,
                density_matrix=(dm_uhf[0] + dm_uhf[1]).astype(np.complex128),
                threshold=1e-7,
            )
        ),
        (
            "uhf_density_spin_complex", PyDensityComplex(
                complex_symmetric=False,
                density_matrix=(dm_uhf[0] - dm_uhf[1]).astype(np.complex128),
                threshold=1e-7,
            )
        ),
    ]

    pybao, basis_set = extract_basis_information(mol)

    sao_spatial_4c_real = calc_overlap_4c_real(basis_set)
    sao_spatial_4c_complex = calc_overlap_4c_complex(basis_set, complex_symmetric=False)

    caplog.clear()
    rep_analyse_densities(
        # Data
        inp_sym="mol",
        pydens=pydens_real,
        pybao=pybao,
        sao_spatial_4c=sao_spatial_4c_real,
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
    output = "\n".join(
        record.getMessage()
        for record in caplog.records
        if record.name == "qsym2-output"
    )

    lines = output.splitlines()
    start = next(
        i for i, line in enumerate(lines) if "> Density results" in line
    )
    symmetries = {}
    for line in lines[start + 1 :]:
        m = re.match(rf"^\s*\d+\s+(\S+)\s+(\|.*)\s+{FLOAT_RE}\s+{FLOAT_RE}", line)
        if m is not None:
            symmetries[m.group(1)] = m.group(2).strip()
        elif len(symmetries) > 0:
            break

    assert symmetries["rhf_density_real"] == "|A|_(1g) ⊕ |T|_(2g)"
    assert symmetries["uhf_density_total_real"] == "|A|_(1g) ⊕ |E|_(g)"
    assert symmetries["uhf_density_spin_real"] == "|E|_(g)"

    caplog.clear()
    rep_analyse_densities(
        # Data
        inp_sym="mol",
        pydens=pydens_complex,
        pybao=pybao,
        sao_spatial_4c=sao_spatial_4c_complex,
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
    output = "\n".join(
        record.getMessage()
        for record in caplog.records
        if record.name == "qsym2-output"
    )

    lines = output.splitlines()
    start = next(
        i for i, line in enumerate(lines) if "> Density results" in line
    )
    symmetries = {}
    for line in lines[start + 1 :]:
        m = re.match(rf"^\s*\d+\s+(\S+)\s+(\|.*)\s+{FLOAT_RE}{FLOAT_RE}i\s+{FLOAT_RE}{FLOAT_RE}i", line)
        if m is not None:
            symmetries[m.group(1)] = m.group(2).strip()
        elif len(symmetries) > 0:
            break

    assert symmetries["rhf_density_complex"] == "|A|_(1g) ⊕ |T|_(2g)"
    assert symmetries["uhf_density_total_complex"] == "|A|_(1g) ⊕ |E|_(g)"
    assert symmetries["uhf_density_spin_complex"] == "|E|_(g)"
