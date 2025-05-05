---
title: "Molecular orbital symmetry in adamantane (Python)"
description: An illustration of QSym²'s representation analysis functionalities via molecular orbitals in adamantane
---

# Molecular orbital symmetry in adamantane (Python)

This tutorial demonstrates how QSym² can be used to obtain symmetry analysis information for self-consistent-field (SCF) molecular orbitals and the Slater determinants constructed from them.
In particular, we show how the Python interface of QSym² can be used with [PySCF](https://pyscf.org/) as the computation backend to obtain molecular orbital symmetry information for neutral and cationic adamantane.

In summary, we need to perform a symmetry-group detection calculation on the adamantane structure to obtain the symmetry group $\mathcal{G}$ which is then used to carry out representation symmetry analysis for the molecular orbitals obtained from a PySCF calculation.

In what follows, we slowly build up the content of a Python script named `adamantane.py` that runs an unrestricted Hartree&ndash;Fock (UHF) calculation using PySCF and then analyses the resulting determinant using QSym².
To prepare for this, create an empty text file with the name `adamantane.py` in a directory of your choosing:

```bash
touch adamantane.py
```

In addition, ensure that PySCF and the Python binding for QSym² are installed (see [Getting started/Installation/#Python-library compilation](../getting-started/installation.md/#python-library-compilation) for instructions).

## Neutral adamantane

### Symmetry-group detection

1. Construct an adamantane molecule in PySCF format by adding the following to `adamantane.py`, making sure to choose either Cartesian ordering or spherical ordering for the atomic-orbital basis functions:

    === "Cartesian ordering"

        ```py title="adamantane.py"
        from pyscf import gto

        mol = gto.M(
            atom=r"""
                C   -0.0000000    1.7842615    0.0000000;
                C    0.8927704    0.8927704   -0.8927704;
                C    1.7842615    0.0000000   -0.0000000;
                C    0.8927704   -0.8927704    0.8927704;
                C    0.0000000   -0.0000000    1.7842615;
                C   -1.7842615   -0.0000000    0.0000000;
                C   -0.8927704    0.8927704    0.8927704;
                C   -0.0000000    0.0000000   -1.7842615;
                C   -0.8927704   -0.8927704   -0.8927704;
                C    0.0000000   -1.7842615   -0.0000000;
                H   -0.6349304    2.4406159   -0.6349304;
                H    0.6349304    2.4406159    0.6349304;
                H    1.5345817    1.5345817   -1.5345817;
                H    2.4406159   -0.6349304   -0.6349304;
                H    2.4406159    0.6349304    0.6349304;
                H    1.5345817   -1.5345817    1.5345817;
                H    0.6349304    0.6349304    2.4406159;
                H   -0.6349304   -0.6349304    2.4406159;
                H   -2.4406159   -0.6349304    0.6349304;
                H   -2.4406159    0.6349304   -0.6349304;
                H   -1.5345817    1.5345817    1.5345817;
                H   -0.6349304    0.6349304   -2.4406159;
                H    0.6349304   -0.6349304   -2.4406159;
                H   -1.5345817   -1.5345817   -1.5345817;
                H   -0.6349304   -2.4406159    0.6349304;
                H    0.6349304   -2.4406159   -0.6349304;
            """,
            unit="Angstrom",
            basis="6-31G*",
            charge=0,
            spin=0,
            cart=True, #(1)!
        )
        ```

        1. :material-information-variant-circle: This boolean specifies that the atomic-orbital basis shells are all Cartesian.
        Then, within each shell, the PySCF convention is that the functions are arranged in lexicographic order.

    === "Spherical ordering"

        ```py title="adamantane.py"
        from pyscf import gto

        mol = gto.M(
            atom=r"""
                C   -0.0000000    1.7842615    0.0000000;
                C    0.8927704    0.8927704   -0.8927704;
                C    1.7842615    0.0000000   -0.0000000;
                C    0.8927704   -0.8927704    0.8927704;
                C    0.0000000   -0.0000000    1.7842615;
                C   -1.7842615   -0.0000000    0.0000000;
                C   -0.8927704    0.8927704    0.8927704;
                C   -0.0000000    0.0000000   -1.7842615;
                C   -0.8927704   -0.8927704   -0.8927704;
                C    0.0000000   -1.7842615   -0.0000000;
                H   -0.6349304    2.4406159   -0.6349304;
                H    0.6349304    2.4406159    0.6349304;
                H    1.5345817    1.5345817   -1.5345817;
                H    2.4406159   -0.6349304   -0.6349304;
                H    2.4406159    0.6349304    0.6349304;
                H    1.5345817   -1.5345817    1.5345817;
                H    0.6349304    0.6349304    2.4406159;
                H   -0.6349304   -0.6349304    2.4406159;
                H   -2.4406159   -0.6349304    0.6349304;
                H   -2.4406159    0.6349304   -0.6349304;
                H   -1.5345817    1.5345817    1.5345817;
                H   -0.6349304    0.6349304   -2.4406159;
                H    0.6349304   -0.6349304   -2.4406159;
                H   -1.5345817   -1.5345817   -1.5345817;
                H   -0.6349304   -2.4406159    0.6349304;
                H    0.6349304   -2.4406159   -0.6349304;
            """,
            unit="Angstrom",
            basis="6-31G*",
            charge=0,
            spin=0,
            cart=False, #(1)!
        )
        ```

        1. :material-information-variant-circle: This boolean specifies that the atomic-orbital basis shells are all spherical.
        Then, within each shell, the PySCF convention is that the functions are arranged in increasing-$m_l$ order, except for $P$ shells where the functions are arranged according to $p_x, p_y, p_z$, or $m_l = +1, -1, 0$.

2. Convert the PySCF molecule into an equivalent QSym² molecule by adding the following to `adamantane.py`:

    
    ```py title="adamantane.py"
    from qsym2 import PyMolecule

    pymol = PyMolecule(
        atoms=mol._atom,
        threshold=1e-7,
    )
    ```

3. We are almost ready to run symmetry-group detection on the above adamantane molecule using QSym².
However, the results from QSym² would not be visible because we have not configured Python to store these results anywhere.
We thus need to configure Python to log the output from QSym² to a file called `adamantane_symmetry.out` by adding the following to `adamantane.py`:

    ```py title="adamantane.py"
    import logging
    from qsym2 import qsym2_output_heading, qsym2_output_contributors

    output_filename = "adamantane_symmetry.out"
    class QSym2Filter(object):
        def __init__(self, level):
            self.__level = level

        def filter(self, logRecord):
            return logRecord.levelno == self.__level

    logging.basicConfig()
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    root.handlers[0].setLevel(logging.WARNING)
    handler = logging.FileHandler(
        output_filename,
        mode="w",
        encoding="utf-8",
    )
    handler.addFilter(QSym2Filter(logging.INFO))
    root.addHandler(handler)

    qsym2_output_heading()
    qsym2_output_contributors()
    ```

4. Then, add the following to `adamantane.py` to set up the symmetry-group detection calculation:

    ```py title="adamantane.py"
    from qsym2 import detect_symmetry_group

    unisym, magsym = detect_symmetry_group( #(1)!
        inp_xyz=None,
        inp_mol=pymol,
        out_sym="mol", #(2)!
        moi_thresholds=[1e-2, 1e-4, 1e-6],
        distance_thresholds=[1e-2, 1e-4, 1e-6],
        time_reversal=False,
        fictitious_magnetic_field=None,
        fictitious_electric_field=None,
        write_symmetry_elements=True,
    )
    ```

    1. :material-information-variant-circle: Explanations of the parameters for symmetry-group detection can be found at [User guide/Symmetry-group detection](../user-guide/symmetry-group-detection.md).
    2. :material-information-variant-circle: This instructs QSym² to save the symmetry-group detection results in a binary file named `mol.qsym2.sym`.
    This file will subsequently be read in by the representation analysis routine in QSym².

5. Run the `adamantane.py` script:

    ```bash
    python3 adamantane.py
    ```

    and verify that everything works so far.
    In particular, check that the `adamantane_symmetry.out` file has been generated and contains results.
    Verify also that a binary file named `mol.qsym2.sym` has been generated in the same location.


### Representation analysis for molecular orbitals

1. Instruct PySCF to run a UHF calculation by adding the following to `adamantane.py`:

    ```py title="adamantane.py"
    from pyscf import scf

    uhf = scf.UHF(mol)
    uhf.kernel()
    ```

After the execution of the above commands, the `uhf` object should contain all information about the UHF Slater determinant to which PySCF has converged.
We must now extract the relevant data from `uhf` into a format that QSym² can understand.

2. Let us first start with the easy part.
Add the following to `adamantane.py` to extract the molecular orbital coefficients and their occupation numbers and energies:

    ```py title="adamantane.py"
    ca = uhf.mo_coeff[0]
    cb = uhf.mo_coeff[1]

    occa = uhf.mo_occ[0]
    occb = uhf.mo_occ[1]

    ea = uhf.mo_energy[0]
    eb = uhf.mo_energy[1]
    ```

3. Next, we need to extract the [atomic-orbital basis angular order information](../user-guide/representation-analysis/basics.md/#atomic-orbital-basis-angular-order) of the calculation.
The required basis angular order information can be extracted by adding the following to `adamantane.py`, making sure to choose the correct atomic-orbital basis function ordering:

    === "Cartesian ordering"

        ```py title="adamantane.py"
        import itertools
        import re
        from qsym2 import PyBasisAngularOrder, ShellType

        def bao_from_ao_labels(labels: list[str]) -> PyBasisAngularOrder:
            r"""Extracts AO basis angular order information from the AO labels
            generated by PySCF.
            """

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
            for atom, atom_shells in itertools.groupby(
                shells, lambda shl: (shl[0], shl[1])
            ):
                atom_shell_tuples = []
                for atom_shell in atom_shells:
                    shell_code = atom_shell[2]
                    angmom_opt = re.search(r"\d+([a-z])", shell_code)
                    assert angmom_opt is not None
                    angmom = int(ANGMOM.index(angmom_opt.group(1).upper()))
                    atom_shell_tuples.append((angmom, ShellType.Cartesian, None)) #(1)!
                pybao.append((atom[1], atom_shell_tuples))

            return PyBasisAngularOrder(pybao)

        pybao = bao_from_ao_labels(mol.ao_labels())
        ```

        1. :material-information-variant-circle: For Cartesian functions, PySCF uses lexicographic ordering (see [here](https://pyscf.org/user/gto.html#ordering-of-basis-functions)).

    === "Spherical ordering"

        ```py title="adamantane.py"
        import itertools
        import re
        from qsym2 import PyBasisAngularOrder

        def bao_from_ao_labels(labels: list[str]) -> PyBasisAngularOrder:
            r"""Extracts AO basis angular order information from the AO labels
            generated by PySCF.
            """

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
            for atom, atom_shells in itertools.groupby(
                shells, lambda shl: (shl[0], shl[1])
            ):
                atom_shell_tuples = []
                for atom_shell in atom_shells:
                    shell_code = atom_shell[2]
                    angmom_opt = re.search(r"\d+([a-z])", shell_code)
                    assert angmom_opt is not None
                    angmom = int(ANGMOM.index(angmom_opt.group(1).upper()))
                    if angmom == 1: #(1)!
                        atom_shell_tuples.append(
                            (angmom, ShellType.Pure, ([+1, -1, 0], False))
                        )
                    else:
                        atom_shell_tuples.append((angmom, ShellType.Pure, (True, angmom % 2 == 0)))
                pybao.append((atom[1], atom_shell_tuples))

            return PyBasisAngularOrder(pybao)

        pybao = bao_from_ao_labels(mol.ao_labels())
        ```

        1. :material-information-variant-circle: For spherical functions, PySCF uses increasing-$m_l$ ordering, except for $P$ shells where the order is $p_x, p_y, p_z$, or $m_l = +1, -1, 0$ (see [here](https://pyscf.org/user/gto.html#ordering-of-basis-functions)).

4. Extract the [two-centre atomic-orbital overlap matrix](../user-guide/representation-analysis/slater-determinants.md/#basis-overlap-matrix) by adding the following to `adamantane.py`:

    ```py title="adamantane.py"
    sao_spatial = mol.intor("int1e_ovlp")
    ```

5. Put everything together into an object that QSym² can understand by adding the following to `adamantane.py`:

    ```py title="adamantane.py"
    from qsym2 import PySpinConstraint, PySlaterDeterminantReal

    pydet = PySlaterDeterminantReal( #(1)!
        structure_constraint=PySpinConstraint.Unrestricted,
        complex_symmetric=False,
        coefficients=[ca, cb],
        occupations=[occa, occb],
        threshold=1e-7,
        mo_energies=[ea, eb],
        energy=uhf.e_tot,
    )
    ```

    1. :material-information-variant-circle: The [`PySlaterDeterminantReal`](https://qsym2.dev/api/qsym2/bindings/python/representation_analysis/slater_determinant/struct.PySlaterDeterminantReal.html) class is only applicable to real Slater determinants.
    If complex Slater determinants are present, use [`PySlaterDeterminantComplex`](https://qsym2.dev/api/qsym2/bindings/python/representation_analysis/slater_determinant/struct.PySlaterDeterminantComplex.html) instead.
    </br></br>See [User guide/Representation analysis/Slater determinants/#Parameters](../user-guide/representation-analysis/slater-determinants.md/#parameters) for detailed explanations of the parameters.

6. Finally, instruct QSym² to perform representation analysis for the Slater determinant above together with its constituting molecular orbitals by adding the following to `adamantane.py`:

    ```py title="adamantane.py"
    from qsym2 import (
        rep_analyse_slater_determinant,
        EigenvalueComparisonMode,
        SymmetryTransformationKind,
    )

    rep_analyse_slater_determinant( #(1)!
        # Data
        inp_sym="mol", #(2)!
        pydet=pydet,
        pybao=pybao,
        sao=sao_spatial,
        sao_h=None,
        sao_spatial_4c=None,
        sao_spatial_4c_h=None,
        # Thresholds
        linear_independence_threshold=1e-7,
        integrality_threshold=1e-7,
        eigenvalue_comparison_mode=EigenvalueComparisonMode.Modulus,
        # Analysis options
        use_magnetic_group=None,
        use_double_group=False,
        use_cayley_table=True,
        symmetry_transformation_kind=SymmetryTransformationKind.Spatial,
        infinite_order_to_finite=None,
        # Other options
        write_character_table=True,
        write_overlap_eigenvalues=True,
        analyse_mo_symmetries=True,
        analyse_mo_mirror_parities=False,
        analyse_density_symmetries=False,
    )
    ```

    1. :material-information-variant-circle: See [User guide/Representation analysis/Slater determinants/#Parameters](../user-guide/representation-analysis/slater-determinants.md/#parameters) for detailed explanations of the parameters.
    2. :material-information-variant-circle: This instructs QSym² to read in the symmetry-group detection results from the binary file named `mol.qsym2.sym` that has been generated earlier by the `detect_symmetry_group` function (see [#Symmetry-group detection](#symmetry-group-detection)).

7. The script is now complete.
The entire sequence of symmetry-group detection and Slater determinant representation analysis can now be executed by running

    ```bash
    python3 adamantane.py
    ```

### Understanding QSym² results

1. Under the `Symmetry-Group Detection` section, inspect the `Threshold-scanning symmetry-group detection` subsection and identify the following:

    - the various threshold combinations,
    - the unitary group obtained for each threshold combination, and
    - the highest unitary group $\mathcal{G}$ found and the associated thresholds.

    Verify that $\mathcal{G}$ is indeed $\mathcal{T}_d$.

2. Under the `Slater Determinant Symmetry Analysis` section, inspect the `Character table of irreducible representations` section and verify that the character table for $\mathcal{G} = \mathcal{T}_d$ has been generated correctly.

    Then, inspect the `Conjugacy class transversal` subsection and verify that the representative elements of the conjugacy classes are sensible.

3. Inspect next the `Space-fixed spatial angular function symmetries in Td` subsection and identify how the standard spherical harmonics and Cartesian functions transform under $\mathcal{T}_d$.

    Then, compare these results to what is normally tabulated with standard character tables.

4. Inspect next the `Basis angular order` subsection and verify that the orders of the functions in the atomic-orbital shells are consistent with those reported by `print(mol.ao_labels())`.
This ensures that the basis angular order information has been extracted correctly.

    For more information, see [User guide/Representation analysis/Basics/#Atomic-orbital basis angular order](../user-guide/representation-analysis/basics.md/#atomic-orbital-basis-angular-order).

5. Inspect next the `Determinant orbit overlap eigenvalues` subsection.
This prints out the full eigenspectrum of the orbit overlap matrix of the Slater determinant being analysed (see [Section 2.4 of the QSym² paper](../about/authorship.md#publications)), and the position of the [linear-independence threshold](../user-guide/representation-analysis/basics.md/#linear-independence-threshold) relative to these eigenvalues.
Check if the linear-independence threshold has indeed been chosen sensibly with respect to the obtained eigenspectrum: is the gap between the eigenvalues immediately above and below the threshold larger than four orders of magnitude?

6. Finally, inspect the `Orbit-based symmetry analysis results` subsection and do the following:

    - identify the overall symmetry of the Slater determinant and check that it is consistent with the dimensionality indicated by the eigenspectrum and the linear-independence threshold; and
    - identify the symmetries of several molecular orbitals of interest and check if their degeneracies are consistent with their energies.


## Cationic adamantane

1. Repeat the above calculation and analysis for cationic adamantane:

    - the charge value in the construction of the `mol` object should be set to `1` and the spin value also to `1`; and
    - the QSym² analysis calculation can be run in exactly the same way as before.

2. Inspect the QSym² output file and determine the following:

    - the overall symmetry of the Slater determinant, and
    - the symmetries of the molecular orbitals that could be correlated to those examined in step 6 of the neutral adamantane case.

    Are these results in agreement with what you might have expected?
