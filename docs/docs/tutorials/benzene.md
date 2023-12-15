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

### Q-Chem calculation

1. Prepare a Q-Chem input file as follows:

    ``` title="benzene.inp"
    $molecule
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
        METHOD TPSS
        UNRESTRICTED true
        SCF_GUESS core
        SCF_ALGORITHM diis
        SCF_CONVERGENCE 13
        PRINT_ORBITALS true
        SYMMETRY off
        SYM_IGNORE true
    $end
    ```
    
    This input file instructs Q-Chem to run a Kohn&ndash;Sham density-functional theory (KS DFT) calculation using the TPSS exchange and correlation functionals on a neutral benzene molecule placed in the $yz$-plane.
    There are a couple of points to note:
    
    - The non-standard orientation of the benzene molecule has been deliberately chosen to illustrate the fact that QSym² does not require molecules to be in any predefined standard orientations in order for the representation analysis to work.
    - The TPSS exchange and correlation functionals are only for illustration purposes and can be replaced with other functionals.
    - The unrestricted spin constraint can be replaced with the restricted one for this closed-shell system &mdash; QSym² is able to handle all three types of spin constraint (restricted, unrestricted, generalised; see [`SpinConstraint`](https://qsym2.dev/api/qsym2/angmom/spinor_rotation_3d/enum.SpinConstraint.html) and [`PySpinConstraint`](https://qsym2.dev/api/qsym2/bindings/python/integrals/enum.PySpinConstraint.html) API documentations).
    - The very tight SCF convergence threshold ensures that the molecular orbitals have high numerical fidelity so that any symmetry breaking detected by QSym² is guaranteed to arise from the intrinsic nature of the orbitals, rather than from numerical noises. For larger systems, though, it might not be possible to converge SCF calculations so tightly, in which case there are [thresholds](../user-guide/representation-analysis/basics.md/#thresholds) that can be adjusted in QSym² to allow numerical noises to be ignored.
    - All symmetry considerations in Q-Chem have been turned off to avoid unwanted symmetrisation and reorientation of the molecule.

2. Run the above Q-Chem calculation, saving the generated scratch directory in the current directory for later use with QSym²:

    ``` bash
    QCSCRATCH=$(pwd) qchem -save benzene.inp benzene.out benzene_scratch
    ```

    The above command instructs Q-Chem to keep all scratch files in a new directory called `benzene_scratch` located in the current directory.
    Inside `benzene_scratch`, there should be a file called `qarchive.h5` which contains the results of the calculation and can be read in by QSym².
    If this file does not exist, check that the Q-Chem version being used is not too old.
