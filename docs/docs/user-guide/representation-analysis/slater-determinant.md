---
title: Slater determinant
description: Configurable parameters for Slater determinant representation analysis
---

# Slater determinant

Let $\Psi_{\mathrm{SD}}$ be an $N_{\mathrm{e}}$-electron Slater determinant constructed from $N_{\mathrm{e}}$ occupied spin-orbitals $\chi_i(\mathbfit{x})$ written in terms of the composite spin-spatial coordinates $\mathbfit{x}$:

$$
    \Psi_{\mathrm{SD}} =
        \sqrt{N_{\mathrm{e}}!} \hat{\mathscr{A}}
        \left[ \prod_{i=1}^{N_{\mathrm{e}}} \chi_i(\mathbfit{x}_i) \right],
$$

where $\hat{\mathscr{A}}$ is the antisymmetriser in the symmetric group $\operatorname{Sym}(N_{\mathrm{e}})$ acting on the electron labels.
The Slater determinant $\Psi_{\mathrm{SD}}$ exists in some subspace of the $N_{\mathrm{e}}$-electron Hilbert space $\mathcal{H}_{N_{\mathrm{e}}}$ whilst the spin-orbitals $\chi_i(\mathbfit{x})$ each belong to some subspace of the one-electron Hilbert space $\mathcal{H}_{1}$.
QSym² is able to provide symmetry assignments for both the Slater determinant $\Psi_{\mathrm{SD}}$ and its constituting spin-orbitals $\chi_i(\mathbfit{x})$.
The mathematical details of this can be found in [Section 2.4.2 of the QSym² paper](../../about/authorship.md#publications).


## Requirements

### Basis overlap matrix

As explained in [Basics/Requirements/#Basis overlap matrix](basics.md/#basis-overlap-matrix), QSym² requires the overlap matrices of the bases chosen for (some subspaces of) $\mathcal{H}_{N_{\mathrm{e}}}$ and $\mathcal{H}_{1}$ in order to perform representation analysis on $\Psi_{\mathrm{SD}}$ and $\chi_i(\mathbfit{x})$, respectively.
As it turns out, since $\Psi_{\mathrm{SD}}$ is constructed from $\chi_i(\mathbfit{x})$, QSym² only requires the overlap matrix for the basis functions on $\mathcal{H}_{1}$ with respect to which the spin-orbitals $\chi_i(\mathbfit{x})$ are defined.
These basis functions are typically Gaussian atomic orbitals, and most, if not all, quantum-chemistry packages compute their overlaps as part of their inner working.
It is therefore more convenient to retrieve the atomic-orbital overlap matrix $\mathbfit{S}_{\mathcal{H}_{1}}$ (also written $\mathbfit{S}_{\mathrm{AO}}$) from quantum-chemistry packages whenever possible.
The ways in which $\mathbfit{S}_{\mathrm{AO}}$ can be read in by QSym² will be described below.

### Atomic-orbital basis angular order

As bases for $\mathcal{H}_{1}$ almost invariably consist of Gaussian atomic orbitals, QSym² requires information about their angular momenta and ordering conventions as described in [Basics/Requirements/#Atomic-orbital basis angular order](basics.md/#atomic-orbital-basis-angular-order).
Whenever possible, QSym² will attempt to construct the basis angular order information from available data, but if this cannot be done, then the required information must be provided manually (see [Basics/Requirements/#Atomic-orbital basis angular order](basics.md/#atomic-orbital-basis-angular-order) for details).


## Parameters

=== "Command-line interface"
    === "Source: Q-Chem HDF5 archive"
        ```yaml
        analysis_targets:
          - !SlaterDeterminant #(1)!
            source: !QchemArchive
              path: path/to/qchem/qarchive.h5
            control:
              # Thresholds
              integrality_threshold: 1e-7
              linear_independence_threshold: 1e-7
              eigenvalue_comparison_mode: Modulus
              # Analysis options
              use_magnetic_group: null
              use_double_group: false
              symmetry_transformation_kind: Spatial
              infinite_order_to_finite: null
              # Other options
              write_character_table: Symbolic
              write_overlap_eigenvalues: true
              analyse_mo_symmetries: true
              analyse_mo_mirror_parities: false
              analyse_density_symmetries: false
        ```

        1. :fontawesome-solid-users: This is just an example analysis target. The choices for symmetry transformation kinds can be specified in any analysis target.
        2. :fontawesome-solid-users: The possible options are:
            - `Spatial`: spatial transformation only,
            - `Spin`: spin transformation only,
            - `SpinSpatial`: coupled spin and spatial transformations

=== "Python"
    ```python
    from qsym2 import (
        rep_analyse_slater_determinant,
        SymmetryTransformationKind, #(1)!
    )

    rep_analyse_slater_determinant( #(2)!
        ...,
        symmetry_transformation_kind=SymmetryTransformationKind.Spatial, #(3)!
    )
    ```

    1. :fontawesome-solid-laptop-code: This is a Python-exposed Rust enum, [`SymmetryTransformationKind`](https://qsym2.dev/api/qsym2/symmetry/symmetry_transformation/enum.SymmetryTransformationKind.html), for indicating the kind of symmetry transformation to be applied on the target.
    2. :fontawesome-solid-users: This is just an example analysis driver function in Python. The choices for symmetry transformation kinds can be specified in any analysis driver function.
    3. :fontawesome-solid-users: The possible options are:
        - `SymmetryTransformationKind.Spatial`: spatial transformation only,
        - `SymmetryTransformationKind.Spin`: spin transformation only,
        - `SymmetryTransformationKind.SpinSpatial`: coupled spin and spatial transformations
