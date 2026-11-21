#!/usr/bin/env python3

import argparse
import logging
import textwrap

import cclib
import mendeleev
import numpy as np

from qsym2 import (
    EigenvalueComparisonMode,
    PyBasisAngularOrder,
    PyMolecule,
    PySlaterDeterminantReal,
    PySpinConstraint,
    ShellType,
    SymmetryTransformationKind,
    detect_symmetry_group,
    qsym2_output_contributors,
    qsym2_output_heading,
    rep_analyse_slater_determinant,
)


class MyFilter:
    def __init__(self, level):
        self.__level = level

    def filter(self, logRecord):
        return logRecord.levelno == self.__level


class RawTextArgumentDefaultsHelpFormatter(
    argparse.RawTextHelpFormatter, argparse.ArgumentDefaultsHelpFormatter
):
    pass


ANGS = ["S", "P", "D", "F", "G", "H"]
EV_TO_HARTREE = 0.03674930495120813


def orca_rep_analyse_slater_determinant(
    filename: str,
    moi_thresholds: list[float] | None = None,
    distance_thresholds: list[float] | None = None,
    linear_independence_threshold: float = 1e-6,
    orbit_overlap_symmetry_threshold: float = 1e-6,
    integrality_threshold: float = 1e-7,
):
    r"""Performs representation analysis for a Slater determinant and its
    molecular orbitals from an Orca output file.

    The results of the analysis are output to a log file at the `INFO` level.

    Parameters
    ----------
    filename :
        Orca output filename to be parsed.
    moi_thresholds :
        List of moment-of-inertia thresholds for symmetry group detection.
    distance_thresholds :
        List of distance thresholds for symmetry group detection.
    linear_independence_threshold :
        Threshold for truncating near-zero orbit-overlap eigenvalues.
    integrality_threshold :
        Threshold for determining whether a subspace multiplicity is an integer.
    """
    data = cclib.io.ccread(filename)

    pymol = PyMolecule(
        atoms=[
            (mendeleev.element(int(atomic_number)).symbol, tuple(coords))
            for atomic_number, coords in zip(
                data.atomnos, data.converged_geometries[-1]
            )
        ],
        threshold=1e-7,
    )

    if moi_thresholds is None:
        moi_thresholds = [1e-2, 1e-4, 1e-6]
    if distance_thresholds is None:
        distance_thresholds = [1e-2, 1e-4, 1e-6]

    _unisym, _magsym = detect_symmetry_group(
        inp_xyz=None,
        inp_mol=pymol,
        out_sym="mol",
        moi_thresholds=moi_thresholds,
        distance_thresholds=distance_thresholds,
        time_reversal=False,
        write_symmetry_elements=True,
        fictitious_magnetic_field=None,
        fictitious_electric_field=None,
    )

    def pure_order(l_symbol: str) -> list[int]:
        r"""Generates the Orca order of spherical functions in a given shell.

        The spherical functions in a given shell of angular momentum :math:`l`
        are ordered in Orca according to :math:`0, +1, -1, \ldots, +l, -l`.
        """
        l = ANGS.index(l_symbol)
        ms = []
        for m in range(l + 1):
            if m == 0:
                ms.append(m)
            else:
                ms.append(m)
                ms.append(-m)
        return ms

    # Orca only uses spherical functions, so there is no pure/Cartesian
    # ambiguity here.
    bao = [
        (
            mendeleev.element(int(atomic_number)).symbol,
            [
                (
                    bshl[0],
                    ShellType.Pure,
                    (pure_order(bshl[0]), ANGS.index(bshl[0]) % 2 == 0),
                )
                for bshl in batm
            ],
        )
        for atomic_number, batm in zip(data.atomnos, data.gbasis)
    ]
    pybao = PyBasisAngularOrder(bao)

    assert data is not None
    sao_spatial = data.aooverlaps

    na = (data.nelectrons + data.mult - 1) // 2
    nb = (data.nelectrons - data.mult + 1) // 2
    cs = [c.T for c in data.mocoeffs]
    es = [e * EV_TO_HARTREE for e in data.moenergies]

    if len(cs) == 1:
        os = [np.array([float(i < na) for i in range(data.nmo)])]
    else:
        os = [
            np.array([float(i < na) for i in range(data.nmo)]),
            np.array([float(i < nb) for i in range(data.nmo)]),
        ]

    scf_energies = data.scfenergies[-1] * EV_TO_HARTREE

    if len(cs) == 1:
        spincons = PySpinConstraint.Restricted
    else:
        spincons = PySpinConstraint.Unrestricted

    pydet_r = PySlaterDeterminantReal(spincons, False, cs, os, 1e-13, es, scf_energies)

    rep_analyse_slater_determinant(
        inp_sym="mol",
        pydet=pydet_r,
        pybaos=[pybao],
        sao=sao_spatial,
        sao_h=None,
        integrality_threshold=integrality_threshold,
        linear_independence_threshold=linear_independence_threshold,
        orbit_overlap_symmetry_threshold=orbit_overlap_symmetry_threshold,
        eigenvalue_comparison_mode=EigenvalueComparisonMode.Real,
        use_magnetic_group=None,
        use_double_group=False,
        symmetry_transformation_kind=SymmetryTransformationKind.Spatial,
        analyse_mo_symmetries=True,
        analyse_mo_mirror_parities=True,
        analyse_density_symmetries=False,
        use_cayley_table=True,
        write_overlap_eigenvalues=True,
        write_character_table=True,
    )


def main():
    parser = argparse.ArgumentParser(
        description=textwrap.dedent("""
            A simple script to run QSym² on an Orca output file.

            The Orca output file needs to have been generated with
                %output
                    PrintLevel Huge
                end
            in order for this script to work. If the output file contains
            multiple SCF results for multiple geometries, only the last set
            of results are analysed by QSym².
            """),
        formatter_class=RawTextArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-f", "--file", help="Orca output file to be parsed", type=str, required=True
    )
    parser.add_argument(
        "-o",
        "--output",
        help="QSym² output file to be written",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-m",
        "--moi_thresholds",
        nargs="*",
        type=float,
        default=[1e-2, 1e-4, 1e-6],
        help="Thresholds for moment-of-inertia comparisons (multiple values can be specified separated by spaces)",
    )
    parser.add_argument(
        "-d",
        "--distance_thresholds",
        nargs="*",
        type=float,
        default=[1e-2, 1e-4, 1e-6],
        help="Thresholds for distance comparisons (multiple values can be specified separated by spaces)",
    )
    parser.add_argument(
        "-l",
        "--linear_independence_threshold",
        type=float,
        default=1e-6,
        help="Threshold for determining linearly independent symmetry-equivalent partners and projecting out the null space",
    )
    parser.add_argument(
        "-i",
        "--integrality_threshold",
        type=float,
        default=1e-7,
        help="Threshold for determining integrality of irreducible (co)representation multiplicities",
    )
    args = parser.parse_args()

    logging.basicConfig()
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    root.handlers[0].setLevel(logging.WARNING)
    handler = logging.FileHandler(f"{args.output}", mode="w", encoding="utf-8")
    handler.addFilter(MyFilter(logging.INFO))
    root.addHandler(handler)

    qsym2_output_heading()
    qsym2_output_contributors()
    orca_rep_analyse_slater_determinant(
        filename=args.file,
        moi_thresholds=args.moi_thresholds,
        distance_thresholds=args.distance_thresholds,
        linear_independence_threshold=args.linear_independence_threshold,
        integrality_threshold=args.integrality_threshold,
    )


if __name__ == "__main__":
    main()
