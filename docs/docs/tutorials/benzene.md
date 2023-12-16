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
