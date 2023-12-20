---
title: Molecular orbital symmetry in benzene
description: An illustration of QSym²'s representation analysis functionalities via molecular orbitals in benzene
---

# Molecular orbital symmetry in benzene

This tutorial demonstrates how QSym² can be used to obtain symmetry analysis information for self-consistent-field (SCF) molecular orbitals and the Slater determinants constructed from them.
In particular, we show how QSym² can be used with Q-Chem HDF5 archive files to obtain molecular orbital symmetry information for neutral and cationic benzene.

!!! info "Q-Chem versions"

    The minimum version of Q-Chem that is capable of producing HDF5 archive files that can be read in by QSym², as tested by the author, is 5.4.2.
    However, the molecular orbital energies are not exported correctly to HDF5 archive files in this version.
    A newer version, 6.1.0, seems to have fixed this.


## Neutral benzene

In summary, we need to run a Q-Chem calculation on neutral benzene and save the resulting molecular orbitals in a HDF5 archive file that will then be read in by QSym² to perform symmetry analysis.

### Q-Chem calculation

1. Prepare a Q-Chem input file as follows:

    <div class="annotate" markdown>
    ``` title="benzene.inp"
    $molecule (1)
        0 1
        C    0.0000000    1.2116067   -0.6995215
        C    0.0000000    1.2116067    0.6995215
        C    0.0000000   -0.0000000   -1.3990430
        C    0.0000000    0.0000000    1.3990430
        C    0.0000000   -1.2116067    0.6995215
        C    0.0000000   -1.2116067   -0.6995215
        H    0.0000000    2.1489398   -1.2406910
        H    0.0000000    2.1489398    1.2406910
        H    0.0000000   -0.0000000   -2.4813820
        H    0.0000000    0.0000000    2.4813820
        H    0.0000000   -2.1489398    1.2406910
        H    0.0000000   -2.1489398   -1.2406910
    $end
    $rem
        BASIS 6-31G*
        METHOD TPSS (2)
        UNRESTRICTED true (3)
        SCF_GUESS core
        SCF_ALGORITHM diis
        SCF_CONVERGENCE 13 (4)
        PRINT_ORBITALS true
        SYMMETRY off (5)
        SYM_IGNORE true
    $end
    ```
    </div>

    1. :material-information-variant-circle: The non-standard orientation of the benzene molecule (with the unique axis along the $x$-direction rather than the conventional $z$-direction) has been deliberately chosen to illustrate the fact that QSym² does not require molecules to be in any predefined standard orientations in order for the representation analysis to work.
    2. :material-information-variant-circle: The TPSS exchange and correlation functionals are only for illustration purposes and can be replaced with other functionals.
    3. :material-information-variant-circle: The unrestricted spin constraint can be replaced with the restricted one for this closed-shell system &mdash; QSym² is able to handle all three types of spin constraint (restricted, unrestricted, generalised; see [`SpinConstraint`](https://qsym2.dev/api/qsym2/angmom/spinor_rotation_3d/enum.SpinConstraint.html) and [`PySpinConstraint`](https://qsym2.dev/api/qsym2/bindings/python/integrals/enum.PySpinConstraint.html) API documentations).
    4. :material-information-variant-circle: The very tight SCF convergence threshold ensures that the molecular orbitals have high numerical fidelity so that any symmetry breaking detected by QSym² is guaranteed to arise from the intrinsic nature of the orbitals, rather than from numerical noises. For larger systems, though, it might not be possible to converge SCF calculations so tightly, in which case there are [thresholds](../user-guide/representation-analysis/basics.md/#thresholds) that can be adjusted in QSym² to allow numerical noises to be ignored.
    5. :material-information-variant-circle: All symmetry considerations in Q-Chem have been turned off to avoid unwanted symmetrisation and reorientation of the molecule.
    
    This input file instructs Q-Chem to run a Kohn&ndash;Sham density-functional theory (KS DFT) calculation using the TPSS exchange and correlation functionals on a neutral benzene molecule placed in the $yz$-plane.

2. Run the above Q-Chem calculation, saving the generated scratch directory in the current directory for later use with QSym²:

    ``` bash
    QCSCRATCH=$(pwd) qchem -save benzene.inp benzene.out benzene_scratch
    ```

    The above command instructs Q-Chem to keep all scratch files in a new directory called `benzene_scratch` located in the current directory.
    Inside `benzene_scratch`, there should be a file called `qarchive.h5` which contains the results of the calculation and can be read in by QSym².
    If this file does not exist, check that the Q-Chem version being used is not too old.

### QSym² symmetry analysis

#### Running QSym²

1. Prepare a QSym² input file as follows:

    ```yaml title="benzene_symmetry.yml"
    symmetry_group_detection: !Parameters #(1)!
      moi_thresholds:
      - 0.0001
      - 0.00001
      distance_thresholds:
      - 0.0001
      - 0.00001
      time_reversal: false
      fictitious_magnetic_fields: null
      fictitious_electric_fields: null
      field_origin_com: false
      write_symmetry_elements: false
      result_save_name: null
    analysis_targets:
    - !SlaterDeterminant #(2)!
      source: !QChemArchive
        path: benzene_scratch/qarchive.h5
      control:
        integrality_threshold: 1e-6
        linear_independence_threshold: 1e-6
        analyse_mo_symmetries: true
        analyse_mo_mirror_parities: false
        analyse_density_symmetries: false
        use_magnetic_group: null
        use_double_group: false
        symmetry_transformation_kind: Spatial
        write_character_table: Symbolic
        write_overlap_eigenvalues: true
        eigenvalue_comparison_mode: Modulus
        infinite_order_to_finite: null
    ```

    1. :material-information-variant-circle: Explanations of the parameters for symmetry-group detection can be found at [User guide/Symmetry-group detection](../user-guide/symmetry-group-detection.md).
    2. :material-information-variant-circle: Explanations of the parameters for Slater determinant representation analysis can be found at [User guide/Representation analysis/Slater determinants](../user-guide/representation-analysis/slater-determinants.md).

    This input file instructs QSym² to first detect the symmetry group of the benzene molecule at various moment-of-inertia thresholds and distance thresholds.
    The highest symmetry group at the tightest threshold combination is then used for the subsequent representation analysis of the molecular orbitals found in the Q-Chem HDF5 archive file located at `benzene_scratch/qarchive.h5`.

2. Run the above QSym² calculation:

    ```bash
    qsym2 run -c benzene_symmetry.yml -o benzene_symmetry.out
    ```

    The above command instructs QSym² to run a symmetry analysis calculation based on the configuration defined in `benzene_symmetry.yml`, and then write the results of the calculation to a new plain-text file called `benzene_symmetry.out`, which can be opened by any text editor.


#### Understanding QSym² results

1. Under the `Symmetry-Group Detection` section, inspect the `Threshold-scanning symmetry-group detection` subsection and identify the following:

    - the various threshold combinations,
    - the unitary group obtained for each threshold combination, and
    - the highest unitary group $mathcal{G}$ found and the associated thresholds.

    Verify that $\mathcal{G}$ is indeed $\mathcal{D}_{6h}$.

2. Under the `Slater Determinant Symmetry Analysis` section, inspect the `Character table of irreducible representations` section and verify that the character table for $\mathcal{G}$ has been generated correctly.

    Then, inspect the `Conjugacy class transversal` subsection and verify that the representative elements of the conjugacy classes are indeed in accordance with the non-standard orientation of the benzene molecule specified [earlier](#q-chem-calculation).

3. Inspect next the `Space-fixed spatial angular function symmetries in D6h` subsection and identify how the standard spherical harmonics and Cartesian functions transform under $\mathcal{D}_{6h}$.

    Then, compare these results to what is normally tabulated with standard character tables.
    In particular, note how the Cartesian axis that transforms as $A_{2u}$ is now the $x$-axis instead of the $z$-axis, and the degenerate pair that transform as $E_{1u}$ are now $(y, z)$ instead of $(x, y)$. This is the consequence of the non-standard orientation of the benzene molecule in which the principal axis is the $x$-axis instead of the $z$-axis as per the standard convention.

4. Inspect next the `Basis angular order` subsection and verify that the orders of the functions in the atomic-orbital shells are consistent with those reported in the Q-Chem output file `benzene.out`.
For more information, see [User guide/Representation analysis/Basics/#Atomic-orbital basis angular order](../user-guide/representation-analysis/basics.md/#atomic-orbital-basis-angular-order).

5. Inspect next the `Determinant orbit overlap eigenvalues` subsection.
This prints out the full eigenspectrum of the orbit overlap matrix of the Slater determinant being analysed (see [Section 2.4 of the QSym² paper](../about/authorship.md#publications)), and the position of the [linear-independence threshold](../user-guide/representation-analysis/basics.md/#linear-independence-threshold) relative to these eigenvalues.
Check if the linear-independence threshold has indeed been chosen sensibly with respect to the obtained eigenspectrum: is the gap between the eigenvalues immediately above and below the threshold larger than four orders of magnitude?

6. Finally, inspect the `Orbit-based symmetry analysis results` subsection and do the following:

    - identify the overall symmetry of the Slater determinant and check that it is consistent with the dimensionality indicated by the eigenspectrum and the linear-independence threshold; and
    - identify the symmetries of several molecular orbitals of interest and, if possible, visualise them using Q-Chem/IQMol and check that the symmetry assignments make sense.


## Cationic benzene

1. Repeat the above calculation and analysis for cationic benzene:

    - the charge value in the Q-Chem input should be set to `1` and the multiplicity value to `2`; and
    - the QSym² analysis calculation can be run in exactly the same way as before.

2. Inspect the QSym² output file and determine the following:

    - the overall symmetry of the Slater determinant, and
    - the symmetries of the molecular orbitals that could be correlated to those examined in step 6 of the neutral benzene case.

    Are these results in agreement with what you might have expected?
