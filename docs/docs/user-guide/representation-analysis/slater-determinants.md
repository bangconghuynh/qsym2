---
title: Slater determinants
description: Configurable parameters for Slater determinant representation analysis
---

# Slater determinants

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
It is therefore more convenient to retrieve the atomic-orbital overlap matrix $\mathbfit{S}_{\mathcal{H}_{1}}$ (also written $\mathbfit{S}_{\mathrm{AO}}$) (and its complex-symmetric version, $\bar{\mathbfit{S}}_{\mathrm{AO}}$, whenever necessary), from quantum-chemistry packages whenever possible.
The ways in which $\mathbfit{S}_{\mathrm{AO}}$ can be read in by QSym² will be described below.

!!! warning "$\mathbfit{S}_{\mathrm{AO}}$ from Q-Chem HDF5 archives"

    Note that Q-Chem HDF5 archive files do store $\mathbfit{S}_{\mathrm{AO}}$, but the ordering of the atomic-orbital basis functions used to define this matrix (lexicographic order for Cartesian functions) is inconsistent with that used to define the molecular orbital coefficients (Q-Chem order for Cartesian functions).
    Hence, QSym² recomputes this matrix from the basis set information also stored in HDF5 archive files to ensure that the basis function ordering is consistent.

### Atomic-orbital basis angular order

As bases for $\mathcal{H}_{1}$ almost invariably consist of Gaussian atomic orbitals, QSym² requires information about their angular momenta and ordering conventions as described in [Basics/Requirements/#Atomic-orbital basis angular order](basics.md/#atomic-orbital-basis-angular-order).
Whenever possible, QSym² will attempt to construct the basis angular order information from available data, but if this cannot be done, then the required information must be provided manually (see [Basics/Requirements/#Atomic-orbital basis angular order](basics.md/#atomic-orbital-basis-angular-order) for details).


## Parameters

!!! info "Feature requirements"

    - Reading in Q-Chem HDF5 archive files requires the [`qchem` feature](../../getting-started/prerequisites.md/#rust-features).
    - Using the Python API requires the [`python` feature](../../getting-started/prerequisites.md/#rust-features).
    - Performing representation analysis for Slater determinants, molecular orbitals, and electron densities that are retrieved from Q-Chem HDF5 archive files requires the [`integrals` feature](../../getting-started/prerequisites.md/#rust-features) to recompute $\mathbfit{S}_{\mathrm{AO}}$ (see [#Basis overlap matrix](#basis-overlap-matrix)).


At the moment, QSym² offers three main ways to perform symmetry analysis for Slater determinants. They are:

- via the command-line interface reading in data from a Q-Chem HDF5 archive file,
- via the command-line interface reading in data from binary files,
- via the Python library API reading in data from Python data structures.

More methods might become possible in the future. The parameter specifications for the three existing methods are shown below.

=== "Command-line interface"
    === "Source: Q-Chem HDF5 archive"
        ```yaml
        analysis_targets:
          - !SlaterDeterminant #(1)!
            source: !QChemArchive #(2)!
              path: path/to/qchem/qarchive.h5 #(3)!
            control: #(4)!
              # Thresholds
              linear_independence_threshold: 1e-7 #(5)!
              integrality_threshold: 1e-7 #(6)!
              eigenvalue_comparison_mode: Modulus #(7)!
              # Analysis options
              use_magnetic_group: null #(8)!
              use_double_group: false #(9)!
              use_cayley_table: true #(10)!
              symmetry_transformation_kind: Spatial #(11)!
              infinite_order_to_finite: null #(12)!
              # Other options
              write_character_table: Symbolic #(13)!
              write_overlap_eigenvalues: true #(14)!
              analyse_mo_symmetries: true #(15)!
              analyse_mo_mirror_parities: false #(16)!
              analyse_density_symmetries: false #(17)!
        ```

        1. :fontawesome-solid-users: This specifies a Slater determinant analysis target.
        2. :fontawesome-solid-users: This specifies that the Slater determinant(s) to be analysed shall be retrieved from a Q-Chem HDF5 archive file.
        This file can be generated by running Q-Chem 5.4 or later with the `-save` option and located in the job scratch directory.
        This file may contain multiple Slater determinants such as those arising from multiple iterations of a geometry optimisation job, or from multiple jobs in a single Q-Chem run.
        3. :fontawesome-solid-users: This specifies the path to the Q-Chem HDF5 archive file to be analysed.
        4. :fontawesome-solid-users: This YAML dictionary contains all control parameters for the symmetry analysis of Slater determinants.
        5. :fontawesome-solid-users: This specifies a floating-point value for the linear independence threshold $\lambda^{\mathrm{thresh}}_{\mathbfit{S}}$.
        For more information, see [Basics/#Thresholds](basics.md/#thresholds).
        </br></br>:material-cog-sync-outline: Default: `1e-7`.
        6. :fontawesome-solid-users: This specifies a floating-point value for the integrality threshold $\lambda^{\mathrm{thresh}}_{\mathrm{int}}$.
        For more information, see [Basics/#Thresholds](basics.md/#thresholds).
        </br></br>:material-cog-sync-outline: Default: `1e-7`.
        7. :fontawesome-solid-users: This specifies the threshold comparison mode for the eigenvalues of the orbit overlap matrix $\mathbfit{S}$. The possible options are:
            - `Real`: this specifies the *real* comparison mode where the real parts of the eigenvalues are compared against the threshold,
            - `Modulus`: this specifies the *modulus* comparison mode where the absolute values of the eigenvalues are compared against the threshold.
        </li>For more information, see [Basics/#Thresholds](basics.md/#thresholds).
        </br></br>:material-cog-sync-outline: Default: `Modulus`.
        8. :fontawesome-solid-users: This specifies whether magnetic groups, if present, shall be used for symmetry analysis. The possible options are:
            - `null`: this specifies choice 1 of [Basics/Analysis options/#Magnetic groups](basics.md/#magnetic-groups) &mdash; use the irreducible representations of the unitary group $\mathcal{G}$,
            - `Representation`: this specifies choice 2 of [Basics/Analysis options/#Magnetic groups](basics.md/#magnetic-groups) &mdash; use the irreducible representations of the magnetic group $\mathcal{M}$, if $\mathcal{M}$ is available,
            - `Corepresentation`: this specifies choice 3 of [Basics/Analysis options/#Magnetic groups](basics.md/#magnetic-groups) &mdash; use the irreducible corepresentations of the magnetic group $\mathcal{M}$, if $\mathcal{M}$ is available.
        </li>For more information, see [Basics/Analysis options/#Magnetic groups](basics.md/#magnetic-groups).
        </br></br>:material-cog-sync-outline: Default: `null`.
        9. :fontawesome-solid-users: This is a boolean specifying if double groups shall be used for symmetry analysis. The possible options are:
            - `false`: use only conventional irreducible representations or corepresentations of $\mathcal{G}$,
            - `true`: use projective irreducible representations or corepresentations of $\mathcal{G}$ obtainable via its double cover $\mathcal{G}^*$.
        </li>For more information, see [Basics/Analysis options/#Double groups](basics.md/#double-groups).
        </br></br>:material-cog-sync-outline: Default: `false`.
        10. :fontawesome-solid-users: This is a boolean specifying if the Cayley table for the group, if available, should be used to speed up the computation of orbit overlap matrices.
        </br></br>:material-cog-sync-outline: Default: `true`.
        11. :fontawesome-solid-users: This specifies the kind of symmetry transformations to be applied to generate the orbit for symmetry analysis.
        The possible options are:
            - `Spatial`: spatial transformation only,
            - `SpatialWithSpinTimeReversal`: spatial transformation with spin-including time reversal,
            - `Spin`: spin transformation only,
            - `SpinSpatial`: coupled spin and spatial transformations.
        </li>For more information, see [Basics/Analysis options/#Transformation kinds](basics.md/#transformation-kinds).
        </br></br>:material-cog-sync-outline: Default: `Spatial`.
        12. :fontawesome-solid-users: This specifies the finite order $n$ to which all infinite-order symmetry elements, if any, are restricted. The possible options are:
            - `null`: do not restrict infinite-order symmetry elements to finite order,
            - a positive integer value: restrict all infinite-order symmetry elements to this finite order (this will be ignored if the system has no infinite-order symmetry elements).
        </li>For more information, see [Basics/Analysis options/#Infinite-order symmetry elements](basics.md/#infinite-order-symmetry-elements).
        </br></br>:material-cog-sync-outline: Default: `null`.
        13. :fontawesome-solid-users: This indicates if the character table of the prevailing symmetry group is to be printed in the output.
        The possible options are:
            - `null`: do not print character tables,
            - `Symbolic`: print character tables symbolically,
            - `Numerical`: print character tables numerically.
        </li></br>:material-cog-sync-outline: Default: `Symbolic`.
        14. :fontawesome-solid-users: This boolean indicates if the eigenspectrum of the overlap matrix for the Slater determinant orbit should be printed out.
        </br></br>:material-cog-sync-outline: Default: `true`.
        15. :fontawesome-solid-users: This boolean indicates if the constituting molecular orbitals (MOs) are also symmetry-analysed.
        </br></br>:material-cog-sync-outline: Default: `true`.
        16. :fontawesome-solid-users: This boolean indicates if MO mirror parities (*i.e.* parities w.r.t. any mirror planes present in the system) are to be analysed alongside MO symmetries.
        </br></br>:material-cog-sync-outline: Default: `false`.
        17. :fontawesome-solid-users: This boolean indicates if density symmetries are to be analysed alongside wavefunction symmetries. If `analyse_mo_symmetries` is set to `true`, then MO density symmetries are also analysed.
        </br></br>:material-cog-sync-outline: Default: `false`.

    === "Source: binary files"
        ```yaml
        analysis_targets:
          - !SlaterDeterminant #(1)!
            source: !Binaries #(2)!
              xyz: path/to/xyz #(3)!
              sao: path/to/2c/sao #(4)!
              sao_4c: null #(5)!
              coefficients: #(6)!
              - path/to/ca
              - path/to/cb
              occupations: #(7)!
              - path/to/occa
              - path/to/occb
              spin_constraint: !Unrestricted #(8)!
              - 2
              - false
              matrix_order: RowMajor #(9)!
              byte_order: LittleEndian #(10)!
              bao: #(11)!
                ...: ...
            control: #(12)!
              ...: ...
        ```

        1. :fontawesome-solid-users: This specifies a Slater determinant analysis target.
        2. :fontawesome-solid-users: This specifies that the Slater determinant to be analysed and related information shall be retrieved from respective binary files.
        These files can be generated from various sources, but must be modified to conform to a suitable format for reading in by QSym².
        </br></br>:fontawesome-solid-laptop-code: Under the hood, the following parameters are handled by the Rust struct [`BinariesSlaterDeterminantSource`](https://qsym2.dev/api/qsym2/interfaces/binaries/struct.BinariesSlaterDeterminantSource.html).
        3. :fontawesome-solid-users: This specifies the path to an XYZ file containing the geometry of the molecular system.
        4. :fontawesome-solid-users: This specifies the path to a binary file containing the real-valued two-centre atomic-orbital *spatial* overlap matrix.
        5. :fontawesome-solid-users: This specifies an optional path to a binary file containing the real-valued four-centre atomic-orbital *spatial* overlap matrix. This is only required for density symmetry analysis.
        6. :fontawesome-solid-users: This list specifies the paths to binary files containing molecular-orbital coefficient matrices. Each item in the list specifies the coefficient matrix for one spin space. The number of spin spaces must match that specified in the `spin_constraint` key.
        7. :fontawesome-solid-users: This list specifies the paths to binary files containing occupation numbers. Each item in the list specifies the occupation numbers of the molecular orbitals in one spin space. The number of spin spaces must match that specified in the `spin_constraint` key.
        8. :fontawesome-solid-laptop-code: Under the hood, this is handled by the Rust enum [`SpinConstraint`](https://qsym2.dev/api/qsym2/angmom/spinor_rotation_3d/enum.SpinConstraint.html).
        </br></br> Note that this does not yet support more general structure constraints since only real-valued Slater determinants can be specified via binary files at the moment.
        </br></br>:fontawesome-solid-users: This specifies the spin constraint applicable to the Slater determinant being analysed. The list of items that follows contains the associated values and is different for each spin constraint. The possible options are:
            - `!Restricted`: this specifies the *restricted* spin constraint where spatial molecular orbitals are identical across all spin spaces. This variant takes only one associated value:
                - `nspins`: this unsigned integer specifies the total number of spin spaces
            - `!Unrestricted`: this specifies the *unrestricted* spin constraint where spatial molecular orbitals can be different across different spin spaces. This variant takes two associated values:
                - `nspins`: this unsigned integer specifies the total number of spin spaces
                - `increasingm`: this boolean indicates whether the spin spaces are arranged in increasing-$m_s$ order (note: the typical $(\alpha, \beta)$ spin spaces are arranged in decreasing-$m_s$ order, and so this boolean should be set to `false`)
            - `!Generalised`: this specifies the *generalised* spin constraint where each spin-orbital is now expressed in a spin-spatial direct product basis. This variant takes two associated values:
                - `nspins`: this unsigned integer specifies the total number of spin spaces
                - `increasingm`: this boolean indicates whether the spin spaces are arranged in increasing-$m_s$ order (note: the typical $(\alpha, \beta)$ spin spaces are arranged in decreasing-$m_s$ order, and so this boolean should be set to `false`)
        9. :fontawesome-solid-users: This specifies the order in which matrix elements are packed in the supplied binary files. The possible options are:
            - `RowMajor`: C-like order where the first index is the slowest and the last index is the fastest,
            - `ColMajor`: Fortran-like order where the first index is the fastest and the last index is the slowest.
        </li></br>:material-cog-sync-outline: Default: `RowMajor`.
        10. :fontawesome-solid-users: This specifies the endianness of the byte values in the supplied binary files. The possible options are:
            - `LittleEndian`: the least-significant byte is stored at the smallest memory address,
            - `BigEndian`: the least-significant byte is stored at the largest memory address.
        </li> Most systems are little-endian, but this should be verified to ensure that the values in the binary files are read in correctly.
        </br></br>:material-cog-sync-outline: Default: `LittleEndian`.
        11. :fontawesome-solid-users: This YAML dictionary specifies the basis angular order information for the underlying calculation. For more information, see [Basics/Requirements/#Atomic-orbital basis angular order](basics.md/#atomic-orbital-basis-angular-order).
        12. :fontawesome-solid-users: This YAML dictionary contains all control parameters for the symmetry analysis of Slater determinants and is identical to that specified for Slater determinant Q-Chem HDF5 archive source.


=== "Python"
    ```python
    from qsym2 import (
        rep_analyse_slater_determinant,
        EigenvalueComparisonMode, #(1)!
        MagneticSymmetryAnalysisKind, #(2)!
        SymmetryTransformationKind, #(3)!
        PySpinConstraint, #(4)!
        PySpinOrbitCoupled, #(5)!
        PySlaterDeterminantReal,
        PySlaterDeterminantComplex,
    )

    ca = np.array([ #(6)!
        [+1.000, +0.000],
        [+0.000, +0.707],
        [+0.000, +0.707],
        ...
    ])
    cb = np.array([
        [+0.000, +0.707],
        [+1.000, +0.000],
        [+0.000, -0.707],
        ...
    ])
    
    occa = np.array([1.0, 1.0]) #(7)!
    occb = np.array([1.0, 0.0])

    ea = np.array([-0.51, -0.38]) #(8)!
    eb = np.array([-0.50, +0.02])

    pydet = PySlaterDeterminantReal( #(9)!
        structure_constraint=PySpinConstraint.Unrestricted, #(10)!
        complex_symmetric=False, #(11)!
        coefficients=[ca, cb], #(12)!
        occupations=[occa, occb], #(13)!
        threshold=1e-7, #(14)!
        mo_energies=[ea, eb], #(15)!
        energy=-1.30, #(16)!
    )

    sda_res = rep_analyse_slater_determinant( #(17)!
        # Data
        inp_sym="mol", #(18)!
        pydet=pydet, #(19)!
        pybao=pybao, #(20)!
        sao=sao_spatial, #(21)!
        sao_h=None, #(22)!
        sao_spatial_4c=None, #(23)!
        sao_spatial_4c_h=None, #(24)!
        # Thresholds
        linear_independence_threshold=1e-7, #(25)!
        integrality_threshold=1e-7, #(26)!
        eigenvalue_comparison_mode=EigenvalueComparisonMode.Modulus, #(27)!
        # Analysis options
        use_magnetic_group=None, #(28)!
        use_double_group=False, #(29)!
        use_cayley_table=True, #(30)!
        symmetry_transformation_kind=SymmetryTransformationKind.Spatial, #(31)!
        infinite_order_to_finite=None, #(32)!
        # Other options
        write_character_table=True, #(33)!
        write_overlap_eigenvalues=True, #(34)!
        analyse_mo_symmetries=True, #(35)!
        analyse_mo_mirror_parities=False, #(36)!
        analyse_density_symmetries=False, #(37)!
    ) #(38)!
    ```

    1. :fontawesome-solid-laptop-code: This is a Python-exposed Rust enum, [`EigenvalueComparisonMode`](https://qsym2.dev/api/qsym2/analysis/enum.EigenvalueComparisonMode.html), for indicating the mode of eigenvalue comparison. See [Basics/Thresholds/Linear independence threshold/#Comparison mode](basics.md/#comparison-mode) for further information.
    2. :fontawesome-solid-laptop-code: This is a Python-exposed Rust enum, [`MagneticSymmetryAnalysisKind`](https://qsym2.dev/api/qsym2/drivers/representation_analysis/enum.MagneticSymmetryAnalysisKind.html), for indicating the type of magnetic symmetry to be used for symmetry analysis. See [Basics/Analysis options/#Magnetic groups](basics.md/#magnetic-groups) for further information.
    3. :fontawesome-solid-laptop-code: This is a Python-exposed Rust enum, [`SymmetryTransformationKind`](https://qsym2.dev/api/qsym2/symmetry/symmetry_transformation/enum.SymmetryTransformationKind.html), for indicating the kind of symmetry transformation to be applied on the target. See [Basics/Analysis options/#Transformation kinds](basics.md/#transformation-kinds) for further information.
    4. :fontawesome-solid-laptop-code: This is a Python-exposed Rust enum, [`PySpinConstraint`](https://qsym2.dev/api/qsym2/bindings/python/integrals/enum.PySpinConstraint.html), for indicating the spin constraint applicable to the Slater determinant. In the Python API, only two spin spaces arranged in decreasing-$m_s$ order are permitted because Python enums do not support associated values.
    5. :fontawesome-solid-laptop-code: This is a Python-exposed Rust enum, [`PySpinOrbitCoupled`](https://qsym2.dev/api/qsym2/bindings/python/integrals/enum.PyOrbitCoupled.html), for indicating the spin--orbit-coupled structure applicable to the Slater determinant. In the Python API, only two-component j-adapted basis structures are permitted.
    6. :fontawesome-solid-users: This specifies a coefficient matrix for one spin space, which is a $N_{\mathrm{bas}} \times N_{\mathrm{MO}}$ `numpy` array. The number of basis functions, $N_{\mathrm{bas}}$, depends on the underlying spin constraint: for *generalised* spin constraint, this is twice the number of spatial basis functions, whereas for *restricted* and *unrestricted* spin constraints, this is the same as the number of spatial basis functions. Each column in the array specifies a molecular orbital which can be occupied or virtual as specified by the occupation numbers.
    7. :fontawesome-solid-users: This specifies an occupation number vector for one spin space, which is a one-dimensional `numpy` array of size $N_{\mathrm{MO}}$. Each value in this array gives the occupation number for the corresponding molecular orbital. Fractional values are allowed, but only when occupation numbers are either $0$ or $1$ can the Slater determinant symmetry be well-defined (otherwise the collection of fractionally occupied molecular orbitals does not actually form a single-determinantal wavefunction).
    8. :fontawesome-solid-users: This specifies an optional orbital energy vector for one spin space, which is a one-dimensional `numpy` array of size $N_{\mathrm{MO}}$. Each value in this array gives the orbital energy for the corresponding molecular orbital.
    9. :fontawesome-solid-users: [`PySlaterDeterminantReal`](https://qsym2.dev/api/qsym2/bindings/python/representation_analysis/slater_determinant/struct.PySlaterDeterminantReal.html) constructs a *real-valued* Slater determinant object. If a *complex-valued* Slater determinant is required instead, use [`PySlaterDeterminantComplex`](https://qsym2.dev/api/qsym2/bindings/python/representation_analysis/slater_determinant/struct.PySlaterDeterminantComplex.html).
    10. :fontawesome-solid-users: This specifies the structure constraint applicable to the Slater determinant being specified. The possible options are:
        - `PySpinConstraint.Restricted`: this specifies the *restricted* spin constraint where spatial molecular orbitals are identical across both spin spaces,
        - `PySpinConstraint.Unrestricted`: this specifies the *unrestricted* spin constraint where spatial molecular orbitals can be different across the two spin spaces,
        - `PySpinConstraint.Generalised`: this specifies the *generalised* spin constraint where each spin-orbital is now expressed in a spin-spatial direct product basis. Only two spin spaces are exposed to Python.
        - `PySpinOrbitCoupled.JAdapted`: this specifies the spin--orbit-coupled structure with j-adapted basis functions. Only two relativistic components (large and small) are exposed to Python.
    11. :fontawesome-solid-users: This specifies whether the Slater determinant is to be symmetry-analysed using the bilinear inner product instead of the conventional sesquilinear inner product.
    12. :fontawesome-solid-users: This specifies the coefficient matrices constituting this Slater determinant. Each matrix in this list is for one set of explicitly specified components in the underlying structure constraint.
    13. :fontawesome-solid-users: This specifies the occupation numbers for the specified molecular orbitals. Each vector in this list corresponds to one specified coefficient matrix.
    14. :fontawesome-solid-users: This specifies a threshold for comparing Slater determinants. This is of no consequence for symmetry analysis.
    15. :fontawesome-solid-users: This is optional.
    16. :fontawesome-solid-users: This is optional.
    17. :fontawesome-solid-users: This is the Python driver function for representation analysis of Slater determinants.
    </br></br>:fontawesome-solid-laptop-code: This is a Python-exposed Rust function, [`rep_analyse_slater_determinant`](https://qsym2.dev/api/qsym2/bindings/python/representation_analysis/slater_determinant/fn.rep_analyse_slater_determinant.html). See the API documentation of this function for more details.
    18. :fontawesome-solid-users: This specifies the path to the `.qsym2.sym` file that contains the serialised results of the symmetry-group detection (see the documentation for the `out_sym` parameter of the Python [`detect_symmetry_group`](https://qsym2.dev/api/qsym2/bindings/python/symmetry_group_detection/fn.detect_symmetry_group.html) function in [Symmetry-group detection/#Parameters](../symmetry-group-detection.md/#parameters)). This file should have been generated by the [`detect_symmetry_group`](https://qsym2.dev/api/qsym2/bindings/python/symmetry_group_detection/fn.detect_symmetry_group.html) function on the underlying molecular system prior to representation analysis.
    </br></br>This name does not need to contain the `.qsym2.sym` extension.
    </br></br>The symmetry results in this file will be used to construct the symmetry group $\mathcal{G}$ to be used in the subsequent representation analysis.
    19. :fontawesome-solid-users: This specifies the Slater determinant to be symmetry-analysed.
    20. :fontawesome-solid-users: This specifies the basis angular order information for the underlying basis. See [Basics/Requirements/#Atomic-orbital basis angular order](basics.md/#atomic-orbital-basis-angular-order) for details of how to specify this.
    21. :fontawesome-solid-users: This specifies the two-centre atomic-orbital overlap matrix as a two-dimensional `numpy` array. The dimensions of this matrix must be $n_{\mathrm{comps}}N_{\mathrm{bas}} \times n_{\mathrm{comps}}N_{\mathrm{bas}}$, where $N_{\mathrm{bas}}$ is the number of basis functions specified in the basis angular order information, and $n_{\mathrm{comps}}$ is either $1$ or the total number of explicit components per coefficient matrix.
    22. :fontawesome-solid-users: This specifies the optional complex-symmetric two-centre atomic-orbital spatial matrix as a two-dimensional `numpy` array. The dimensions of this matrix must be $n_{\mathrm{comps}}N_{\mathrm{bas}} \times n_{\mathrm{comps}}N_{\mathrm{bas}}$, where $N_{\mathrm{bas}}$ is the number of basis functions specified in the basis angular order information, and $n_{\mathrm{comps}}$ is either $1$ or the total number of explicit components per coefficient matrix. This is only required if antiunitary operations are ppresent.
    </br></br>:material-cog-sync-outline: Default: `None`.
    23. :fontawesome-solid-users: This specifies the optional four-centre atomic-orbital spatial overlap tensor as a four-dimensional `numpy` array. This is only required for density symmetry analysis, and currently only works for `PySpinConstraint` but not `PySpinOrbitCoupled` structure constraint.
    </br></br>:material-cog-sync-outline: Default: `None`.
    24. :fontawesome-solid-users: This specifies the optional complex-symmetric four-centre atomic-orbital spatial overlap tensor as a four-dimensional `numpy` array. This is only required for density symmetry analysis in the presence of antiunitary operations and currently only works for `PySpinConstraint` but not `PySpinOrbitCoupled` structure constraint.
    </br></br>:material-cog-sync-outline: Default: `None`.
    25. :fontawesome-solid-users: This specifies a floating-point value for the linear independence threshold $\lambda^{\mathrm{thresh}}_{\mathbfit{S}}$.
    For more information, see [Basics/#Thresholds](basics.md/#thresholds).
    26. :fontawesome-solid-users: This specifies a floating-point value for the integrality threshold $\lambda^{\mathrm{thresh}}_{\mathrm{int}}$.
    For more information, see [Basics/#Thresholds](basics.md/#thresholds).
    27. :fontawesome-solid-users: This specifies the threshold comparison mode for the eigenvalues of the orbit overlap matrix $\mathbfit{S}$. The possible options are:
        - `EigenvalueComparisonMode.Real`: this specifies the *real* comparison mode where the real parts of the eigenvalues are compared against the threshold,
        - `EigenvalueComparisonMode.Modulus`: this specifies the *modulus* comparison mode where the absolute values of the eigenvalues are compared against the threshold.
    </li>For more information, see [Basics/#Thresholds](basics.md/#thresholds).
    28. :fontawesome-solid-users: This specifies whether magnetic groups, if present, shall be used for symmetry analysis. The possible options are:
        - `None`: this specifies choice 1 of [Basics/Analysis options/#Magnetic groups](basics.md/#magnetic-groups) &mdash; use the irreducible representations of the unitary group $\mathcal{G}$,
        - `MagneticSymmetryAnalysisKind.Representation`: this specifies choice 2 of [Basics/Analysis options/#Magnetic groups](basics.md/#magnetic-groups) &mdash; use the irreducible representations of the magnetic group $\mathcal{M}$, if $\mathcal{M}$ is available,
        - `MagneticSymmetryAnalysisKind.Corepresentation`: this specifies choice 3 of [Basics/Analysis options/#Magnetic groups](basics.md/#magnetic-groups) &mdash; use the irreducible corepresentations of the magnetic group $\mathcal{M}$, if $\mathcal{M}$ is available.
    </li>For more information, see [Basics/Analysis options/#Magnetic groups](basics.md/#magnetic-groups).
    29. :fontawesome-solid-users: This is a boolean specifying if double groups shall be used for symmetry analysis. The possible options are:
        - `False`: use only conventional irreducible representations or corepresentations of $\mathcal{G}$,
        - `True`: use projective irreducible representations or corepresentations of $\mathcal{G}$ obtainable via its double cover $\mathcal{G}^*$.
    </li>For more information, see [Basics/Analysis options/#Double groups](basics.md/#double-groups).
    30. :fontawesome-solid-users: This is a boolean specifying if the Cayley table for the group, if available, should be used to speed up the computation of orbit overlap matrices.
    </br></br>:material-cog-sync-outline: Default: `True`.
    31. :fontawesome-solid-users: This specifies the kind of symmetry transformations to be applied to generate the orbit for symmetry analysis.
    The possible options are:
        - `SymmetryTransformationKind.Spatial`: spatial transformation only,
        - `SymmetryTransformationKind.SpatialWithSpinTimeReversal`: spatial transformation with spin-including time reversal,
        - `SymmetryTransformationKind.Spin`: spin transformation only,
        - `SymmetryTransformationKind.SpinSpatial`: coupled spin and spatial transformations.
    </li>For more information, see [Basics/Analysis options/#Transformation kinds](basics.md/#transformation-kinds).
    32. :fontawesome-solid-users: This specifies the finite order $n$ to which all infinite-order symmetry elements, if any, are restricted. The possible options are:
        - `None`: do not restrict infinite-order symmetry elements to finite order,
        - a positive integer value: restrict all infinite-order symmetry elements to this finite order (this will be ignored if the system has no infinite-order symmetry elements).
    </li>For more information, see [Basics/Analysis options/#Infinite-order symmetry elements](basics.md/#infinite-order-symmetry-elements).
    </br></br>:material-cog-sync-outline: Default: `None`.
    33. :fontawesome-solid-users: This boolean indicates if the *symbolic* character table of the prevailing symmetry group is to be printed in the output.
    </br></br>:material-cog-sync-outline: Default: `True`.
    34. :fontawesome-solid-users: This boolean indicates if the eigenspectrum of the overlap matrix for the Slater determinant orbit should be printed out.
    </br></br>:material-cog-sync-outline: Default: `True`.
    35. :fontawesome-solid-users: This boolean indicates if the constituting molecular orbitals (MOs) are also symmetry-analysed.
    </br></br>:material-cog-sync-outline: Default: `True`.
    36. :fontawesome-solid-users: This boolean indicates if MO mirror parities (*i.e.* parities w.r.t. any mirror planes present in the system) are to be analysed alongside MO symmetries.
    </br></br>:material-cog-sync-outline: Default: `False`.
    37. :fontawesome-solid-users: This boolean indicates if density symmetries are to be analysed alongside wavefunction symmetries. If `analyse_mo_symmetries` is set to `True`, then MO density symmetries are also analysed.
    </br></br>:material-cog-sync-outline: Default: `False`.
    38. :fontawesome-solid-laptop-code: :fontawesome-solid-users: The [`rep_analyse_slater_determinant`](https://qsym2.dev/api/qsym2/bindings/python/representation_analysis/slater_determinant/fn.rep_analyse_slater_determinant.html) function returns a single [`PySlaterDeterminantRepAnalysisResult`](https://qsym2.dev/api/qsym2/bindings/python/representation_analysis/slater_determinant/struct.PySlaterDeterminantRepAnalysisResult.html) object containing the Python-exposed results of the representation analysis.
