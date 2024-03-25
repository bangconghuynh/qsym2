---
title: Electron densities
description: Configurable parameters for electron density representation analysis
---

# Electron densities

Let $\rho(\mathbfit{r})$ be the one-electron density defined based on an $N_{\mathrm{e}}$-electron wavefunction $\Psi(\mathbfit{x}_1, \ldots, \mathbfit{x}_{N_{\mathrm{e}}})$ as

$$
    \rho(\mathbfit{r}) =
        N_{\mathrm{e}} \int
            \Psi(\mathbfit{r}, s, \mathbfit{x}_2, \ldots, \mathbfit{x}_{N_{\mathrm{e}}})^*
            \Psi(\mathbfit{r}, s, \mathbfit{x}_2, \ldots, \mathbfit{x}_{N_{\mathrm{e}}})
            \ \mathrm{d}s \ \mathrm{d}\mathbfit{x}_2 \ldots \mathrm{d}\mathbfit{x}_{N_{\mathrm{e}}}.
$$

In an atomic-orbital basis $\{ \phi_{\gamma}(\mathbfit{r}), \phi_{\delta}(\mathbfit{r}), \ldots \}$, the density $\rho(\mathbfit{r})$ can be expanded as

$$
    \rho(\mathbfit{r}) = \sum_{\gamma \delta}
        \phi_{\gamma}(\mathbfit{r}) \phi_{\delta}(\mathbfit{r}) P_{\delta \gamma},
$$

where $P_{\delta \gamma}$ are elements of the corresponding density matrix $\mathbfit{P}$ in this basis.
QSym² is able to provide symmetry assignments for electron densities and derived quantities &mdash; the mathematical details of this can be found in [Section 2.4.2 of the QSym² paper](../../about/authorship.md#publications).


## Requirements

### Basis overlap matrix

As explained in [Section 2.4.2 of the QSym² paper](../../about/authorship.md#publications), the symmetry analysis of electron densities requires four-centre overlap integrals:

$$
    \braket{\phi_{\gamma} \phi_{\delta} | \phi_{\gamma'} \phi_{\delta'}} = \int
        \phi_{\gamma}^*(\mathbfit{r}) \phi_{\delta}^*(\mathbfit{r})
        \phi_{\gamma'}(\mathbfit{r}) \phi_{\delta'}(\mathbfit{r})
        \ \mathrm{d}\mathbfit{r}.
$$

Few programs are known to have these integrals computed as part of their routine calculations.
QSym² therefore has implementations to calculate these integrals (and their complex-symmetric versions), provided that the full basis set information is provided (see [Integral evaluation](../integral-evaluation.md)).

### Atomic-orbital basis angular order

As electron densities are expanded in atomic-orbital bases in QSym², information about their angular momenta and ordering conventions as described in [Basics/Requirements/#Atomic-orbital basis angular order](basics.md/#atomic-orbital-basis-angular-order) is required.
Whenever possible, QSym² will attempt to construct the basis angular order information from available data, but if this cannot be done, then the required information must be provided manually (see [Basics/Requirements/#Atomic-orbital basis angular order](basics.md/#atomic-orbital-basis-angular-order) for details).

## Parameters

!!! info "Feature requirements"

    - Using the Python API requires the [`python` feature](../../getting-started/prerequisites.md/#rust-features).
    - Performing representation analysis for electron densities requires the [`integrals` feature](../../getting-started/prerequisites.md/#rust-features).

QSym² is able to perform symmetry analysis for electron densities that can arise from various sources.
In particular, electron densities constructed in Hartree&ndash;Fock theory or Kohn&ndash;Sham density-functional theory can already be symmetry-analysed alongside Slater determinants and molecular orbitals (see [Slater determinants](slater-determinants.md)).
On the other hand, electron densities that can be obtained in other theories (*e.g.* coupled-cluster or orbital-free density-functional theory) can be symmetry-analysed in QSym² via the Python library API reading in data from Python data structures.
The way to do this is shown below.

=== "Python"
    ```python
    from qsym2 import (
        rep_analyse_densities,
        EigenvalueComparisonMode, #(1)!
        MagneticSymmetryAnalysisKind, #(2)!
        SymmetryTransformationKind, #(3)!
        PyDensityReal,
        PyDensityComplex,
        PyBasisShellContraction,
        calc_overlap_4c_real,
        calc_overlap_4c_complex,
    )

    dmao_a = np.array([ #(4)!
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.5, 0.0, 0.0],
        [0.0, 0.0, 0.5, 0.0],
        [0.0, 0.0, 0.5, 0.8],
    ])
    dmao_b = np.array([
        [0.7, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.7, 0.0],
        [0.0, 0.0, 0.5, 0.8],
    ])

    pydens = [ #(5)!
        (
            "alpha_density", PyDensityReal( #(6)!
                complex_symmetric=False, #(7)!
                density_matrix=dmao_a, #(8)!
                threshold=1e-7, #(9)!
            )
        ),
        (
            "beta_density", PyDensityReal(
                complex_symmetric=False,
                density_matrix=dmao_b,
                threshold=1e-7,
            )
        ),
        (
            "total_density", PyDensityReal(
                complex_symmetric=False,
                density_matrix=dmao_a+dmao_b,
                threshold=1e-7,
            )
        ),
        (
            "spin_density", PyDensityReal(
                complex_symmetric=False,
                density_matrix=dmao_a-dmao_b,
                threshold=1e-7,
            )
        ),
    ]

    basis_set = ... #(10)!
    sao_spatial_4c = calc_overlap_4c_real(basis_set)

    rep_analyse_densities( #(11)!
        # Data
        inp_sym="mol", #(12)!
        pydens=pydens, #(13)!
        pybao=pybao, #(14)!
        sao_spatial_4c=sao_spatial_4c, #(15)!
        sao_spatial_4c_h=None, #(16)!
        # Thresholds
        linear_independence_threshold=1e-7, #(17)!
        integrality_threshold=1e-7, #(18)!
        eigenvalue_comparison_mode=EigenvalueComparisonMode.Modulus, #(19)!
        # Analysis options
        use_magnetic_group=None, #(20)!
        use_double_group=False, #(21)!
        use_cayley_table=True, #(22)!
        symmetry_transformation_kind=SymmetryTransformationKind.Spatial, #(23)!
        infinite_order_to_finite=None, #(24)!
        # Other options
        write_character_table=True, #(25)!
    )
    ```

    1. :fontawesome-solid-laptop-code: This is a Python-exposed Rust enum, [`EigenvalueComparisonMode`](https://qsym2.dev/api/qsym2/analysis/enum.EigenvalueComparisonMode.html), for indicating the mode of eigenvalue comparison. See [Basics/Thresholds/Linear independence threshold/#Comparison mode](basics.md/#comparison-mode) for further information.
    2. :fontawesome-solid-laptop-code: This is a Python-exposed Rust enum, [`MagneticSymmetryAnalysisKind`](https://qsym2.dev/api/qsym2/drivers/representation_analysis/enum.MagneticSymmetryAnalysisKind.html), for indicating the type of magnetic symmetry to be used for symmetry analysis. See [Basics/Analysis options/#Magnetic groups](basics.md/#magnetic-groups) for further information.
    3. :fontawesome-solid-laptop-code: This is a Python-exposed Rust enum, [`SymmetryTransformationKind`](https://qsym2.dev/api/qsym2/symmetry/symmetry_transformation/enum.SymmetryTransformationKind.html), for indicating the kind of symmetry transformation to be applied on the target. See [Basics/Analysis options/#Transformation kinds](basics.md/#transformation-kinds) for further information.
    4. :fontawesome-solid-users: This specifies a density matrix in the atomic-orbital basis, $\mathbfit{P}$, for one electron density, which is a $N_{\mathrm{bas}} \times N_{\mathrm{bas}}$ `numpy` array. The number of basis functions, $N_{\mathrm{bas}}$, is always the number of *spatial* basis functions because electron densities are entirely spatial quantities.
    5. :fontawesome-solid-users: This specifies a list of densities to be symmetry-analysed. Each element in this list is a tuple containing a brief string description for the density and a specification for the density itself.
    6. :fontawesome-solid-laptop-code: The classes [`PyDensityReal`](https://qsym2.dev/api/qsym2/bindings/python/representation_analysis/density/struct.PyDensityReal.html) and [`PyDensityComplex`](https://qsym2.dev/api/qsym2/bindings/python/representation_analysis/density/struct.PyDensityComplex.html) are Python-exposed Rust structures to marshall real or complex electron density information between Rust and Python.
    7. :fontawesome-solid-users: This specifies whether the electron densities are to be considered with respect to an inner product space where the conventional sesquilinear inner product has been replaced by a bilinear form.
    8. :fontawesome-solid-users: This specifies the density matrix $\mathbfit{P}$ for this density.
    9. :fontawesome-solid-users: This specifies a threshold for comparing electron densities. This is of no consequence for symmetry analysis.
    10. :fontawesome-solid-users: This specifies the basis set for evaluating the required four-centre overlap integrals. The details of this specification can be found in [Integral evalulation](../integral-evaluation.md).
    11. :fontawesome-solid-users: This is the Python driver function for representation analysis of electron densities.
    </br></br>:fontawesome-solid-laptop-code: This is a Python-exposed Rust function, [`rep_analyse_densities`](https://qsym2.dev/api/qsym2/bindings/python/representation_analysis/density/fn.rep_analyse_densities.html). See the API documentation of this function for more details.
    12. :fontawesome-solid-users: This specifies the path to the `.qsym2.sym` file that contains the serialised results of the symmetry-group detection (see the documentation for the `out_sym` parameter of the Python [`detect_symmetry_group`](https://qsym2.dev/api/qsym2/bindings/python/symmetry_group_detection/fn.detect_symmetry_group.html) function in [Symmetry-group detection/#Parameters](../symmetry-group-detection.md/#parameters)). This file should have been generated by the [`detect_symmetry_group`](https://qsym2.dev/api/qsym2/bindings/python/symmetry_group_detection/fn.detect_symmetry_group.html) function on the underlying molecular system prior to representation analysis.
    </br></br>This name does not need to contain the `.qsym2.sym` extension. 
    </br></br>The symmetry results in this file will be used to construct the symmetry group $\mathcal{G}$ to be used in the subsequent representation analysis.
    13. :fontawesome-solid-users: This specifies the electron densities to be symmetry-analysed.
    14. :fontawesome-solid-users: This specifies the basis angular order information for the underlying basis. See [Basics/Requirements/#Atomic-orbital basis angular order](basics.md/#atomic-orbital-basis-angular-order) for details of how to specify this.
    15. :fontawesome-solid-users: This specifies the four-centre atomic-orbital spatial overlap tensor as a four-dimensional `numpy` array.
    16. :fontawesome-solid-users: This specifies the optional complex-symmetric four-centre atomic-orbital spatial overlap tensor as a four-dimensional `numpy` array. This is only required for density symmetry analysis in the presence of antiunitary operations.
    </br></br>:material-cog-sync-outline: Default: `None`.
    17. :fontawesome-solid-users: This specifies a floating-point value for the linear independence threshold $\lambda^{\mathrm{thresh}}_{\mathbfit{S}}$.
    For more information, see [Basics/#Thresholds](basics.md/#thresholds).
    18. :fontawesome-solid-users: This specifies a floating-point value for the integrality threshold $\lambda^{\mathrm{thresh}}_{\mathrm{int}}$.
    For more information, see [Basics/#Thresholds](basics.md/#thresholds).
    19. :fontawesome-solid-users: This specifies the threshold comparison mode for the eigenvalues of the orbit overlap matrix $\mathbfit{S}$. The possible options are:
        - `EigenvalueComparisonMode.Real`: this specifies the *real* comparison mode where the real parts of the eigenvalues are compared against the threshold,
        - `EigenvalueComparisonMode.Modulus`: this specifies the *modulus* comparison mode where the absolute values of the eigenvalues are compared against the threshold.
    </li>For more information, see [Basics/#Thresholds](basics.md/#thresholds).
    20. :fontawesome-solid-users: This specifies whether magnetic groups, if present, shall be used for symmetry analysis. The possible options are:
        - `None`: this specifies choice 1 of [Basics/Analysis options/#Magnetic groups](basics.md/#magnetic-groups) &mdash; use the irreducible representations of the unitary group $\mathcal{G}$,
        - `MagneticSymmetryAnalysisKind.Representation`: this specifies choice 2 of [Basics/Analysis options/#Magnetic groups](basics.md/#magnetic-groups) &mdash; use the irreducible representations of the magnetic group $\mathcal{M}$, if $\mathcal{M}$ is available,
        - `MagneticSymmetryAnalysisKind.Corepresentation`: this specifies choice 3 of [Basics/Analysis options/#Magnetic groups](basics.md/#magnetic-groups) &mdash; use the irreducible corepresentations of the magnetic group $\mathcal{M}$, if $\mathcal{M}$ is available.
    </li>For more information, see [Basics/Analysis options/#Magnetic groups](basics.md/#magnetic-groups).
    21. :fontawesome-solid-users: This is a boolean specifying if double groups shall be used for symmetry analysis. The possible options are:
        - `False`: use only conventional irreducible representations or corepresentations of $\mathcal{G}$,
        - `True`: use projective irreducible representations or corepresentations of $\mathcal{G}$ obtainable via its double cover $\mathcal{G}^*$.
    </li>For more information, see [Basics/Analysis options/#Double groups](basics.md/#double-groups).
    22. :fontawesome-solid-users: This is a boolean specifying if the Cayley table for the group, if available, should be used to speed up the computation of orbit overlap matrices.
    </br></br>:material-cog-sync-outline: Default: `True`.
    23. :fontawesome-solid-users: This specifies the kind of symmetry transformations to be applied to generate the orbit for symmetry analysis.
    The possible options are:
        - `SymmetryTransformationKind.Spatial`: spatial transformation only,
        - `SymmetryTransformationKind.SpatialWithSpinTimeReversal`: spatial transformation with spin-including time reversal,
        - `SymmetryTransformationKind.Spin`: spin transformation only,
        - `SymmetryTransformationKind.SpinSpatial`: coupled spin and spatial transformations.
    </li>For more information, see [Basics/Analysis options/#Transformation kinds](basics.md/#transformation-kinds).
    24. :fontawesome-solid-users: This specifies the finite order $n$ to which all infinite-order symmetry elements, if any, are restricted. The possible options are:
        - `None`: do not restrict infinite-order symmetry elements to finite order,
        - a positive integer value: restrict all infinite-order symmetry elements to this finite order (this will be ignored if the system has no infinite-order symmetry elements).
    </li>For more information, see [Basics/Analysis options/#Infinite-order symmetry elements](basics.md/#infinite-order-symmetry-elements).
    </br></br>:material-cog-sync-outline: Default: `None`.
    25. :fontawesome-solid-users: This boolean indicates if the *symbolic* character table of the prevailing symmetry group is to be printed in the output.
    </li></br>:material-cog-sync-outline: Default: `True`.
