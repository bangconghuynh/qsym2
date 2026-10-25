import logging
import os
import re

import numpy as np
from pyscf import gto, scf
from pyscf.hessian import thermo

from qsym2 import (
    EigenvalueComparisonMode,
    PyMolecule,
    PyVibrationalCoordinateCollectionComplex,
    PyVibrationalCoordinateCollectionReal,
    SymmetryTransformationKind,
    detect_symmetry_group,
    rep_analyse_vibrational_coordinate_collection,
)

# pyo3-log caches logger levels on first use, so this must be set before any QSym² call.
logging.getLogger("qsym2-output").setLevel(logging.INFO)


def _parse_mode_symmetries(output: str) -> list[str]:
    r"""Extracts the per-mode symmetry labels from the vibrational coordinate results table."""
    lines = output.splitlines()
    start = next(
        i for i, line in enumerate(lines) if "> Vibrational coordinate results" in line
    )
    symmetries = []
    for line in lines[start + 1 :]:
        m = re.match(r"^\s*\d+\s+\S+\s+(\|\S*)", line)
        if m is not None:
            symmetries.append(m.group(1))
        elif symmetries:
            break
    return symmetries


def test_rep_analyse_vibrational_coordinate_collection_nh3(caplog):
    # Exact C3v geometry so that symmetry detection is not limited by rounded coordinates.
    r_h, z_n, z_h = 0.9428, 0.1118, -0.2609
    atoms = [("N", (0.0, 0.0, z_n))] + [
        ("H", (r_h * np.cos(theta), r_h * np.sin(theta), z_h))
        for theta in np.pi / 2 + np.arange(3) * 2 * np.pi / 3
    ]
    mol = gto.M(
        atom=atoms,
        unit="Angstrom",
        basis="sto-3g",
        verbose=0,
    )

    # ------------------
    # Frequency analysis
    # ------------------
    rhf = scf.RHF(mol)
    rhf.conv_tol = 1e-12
    rhf.run()

    hess = rhf.Hessian().kernel()
    results = thermo.harmonic_analysis(mol, hess)

    # `norm_mode` has shape (n_vib, n_atoms, 3).
    norm_mode = results["norm_mode"]
    n_vib = norm_mode.shape[0]
    coefficients = np.ascontiguousarray(norm_mode.reshape(n_vib, -1).T)

    freqs = np.asarray(results["freq_wavenumber"])
    frequencies = np.where(np.abs(freqs.imag) > 1e-8, -np.abs(freqs.imag), freqs.real)

    # Ammonia has 3N - 6 = 6 vibrational modes: 2A1 + 2E.
    assert coefficients.shape == (12, 6)
    assert frequencies.shape == (6,)

    pyvibs_real = PyVibrationalCoordinateCollectionReal(
        coefficients=coefficients,
        frequencies=frequencies,
        threshold=1e-7,
    )
    pyvibs_complex = PyVibrationalCoordinateCollectionComplex(
        coefficients=coefficients.astype(np.complex128),
        frequencies=frequencies.astype(np.complex128),
        threshold=1e-7,
    )

    # ------------------------
    # Symmetry group detection
    # ------------------------
    pymol = PyMolecule(
        atoms=mol._atom,
        threshold=1e-7,
        magnetic_field=None,
        electric_field=None,
    )
    unisym, _magsym = detect_symmetry_group(
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
    assert unisym.group_name == "C3v"

    # -----------------------------
    # Vibrational symmetry analysis
    # -----------------------------

    try:
        for pyvibs in [pyvibs_real, pyvibs_complex]:
            caplog.clear()
            rep_analyse_vibrational_coordinate_collection(
                # Data
                inp_sym="mol",
                pyvibs=pyvibs,
                # Thresholds
                integrality_threshold=1e-6,
                linear_independence_threshold=1e-6,
                orbit_overlap_symmetry_threshold=1e-6,
                eigenvalue_comparison_mode=EigenvalueComparisonMode.Modulus,
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
            # Modes are sorted by frequency; each E pair is listed as two degenerate modes.
            assert sorted(_parse_mode_symmetries(output)) == sorted(
                ["|A|_(1)"] * 2 + ["|E|"] * 4
            )
    finally:
        os.remove("mol.qsym2.sym")
