---
title: Real-space functions
description: Configurable parameters for real-space function representation analysis
---

# Real-space functions

Let $f(\mathbfit{r}): \mathbb{R}^3 \to \mathbb{F}$ be a real-valued ($\mathbb{F} = \mathbb{R}$) or complex-valued ($\mathbb{F} = \mathbb{C}$) function on $\mathbb{R}^3$ that can be specified on a grid.
QSym² is able to provide symmetry assignments for such functions using the orbit-based analysis formulation detailed in [Section 2.4.2 of the QSym² paper](../../about/authorship.md#publications).


## Requirements

### Basis overlap matrix

The symmetry analysis of real-spacre functions requires the following inner product to be defined:

$$
    \braket{f_1 | f_2} = \int
        f_{1}^*(\mathbfit{r}) f_{2}(\mathbfit{r}) w(\mathbfit{r})
        \ \mathrm{d}\mathbfit{r},
$$

where $w(\mathbfit{r})$ is an appropriate weight function.
On a grid $G$, the above integral can be approximated as

$$
    \braket{f_1 | f_2} \approx \sum_{\mathbfit{r}_i \in G}
        f_{1}^*(\mathbfit{r}_i) f_{2}(\mathbfit{r}_i) w(\mathbfit{r}_i) \Delta\mathbfit{r}_i,
$$

where the finite elements $\Delta\mathbfit{r}_i$ can be absorbed into the weight function.


### Function specification

The real-space function $f(\mathbfit{r})$ needs to be specifiable as a closure that takes in three real-valued arguments for the three input Cartesian coordinates and returns a scalar.


## Parameters

!!! info "Feature requirements"

    - As the symmetry analysis of real-space functions is still under development, its usage requires the [`sandbox` feature](../../getting-started/prerequisites.md/#rust-features).
    - Using the Python API requires the [`python` feature](../../getting-started/prerequisites.md/#rust-features).

QSym² is able to perform symmetry analysis for real-space functions that can arise from various sources, as long as they can be specified programmatically as a closure that takes in three real-valued arguments and returns a scalar.
The way to do this via the Python API is shown below.

=== "Python"
    ```python
    from qsym2 import (
        sandbox, #(1)!
        EigenvalueComparisonMode, #(2)!
        MagneticSymmetryAnalysisKind, #(3)!
        SymmetryTransformationKind, #(4)!
    )

    grid = np.array( #(5)!
        [
            [x, y, z]
            for x in np.arange(-1, 1.01, 0.1)
            for y in np.arange(-1, 1.01, 0.1)
            for z in np.arange(-1, 1.01, 0.1)
        ]
    ).T

    weight = np.array( #(6)!
        [np.exp(-(np.linalg.norm(r) ** 2)) for r in grid.T],
        dtype=np.complex128
    )

    def real_function(x, y, z): #(7)!
        return x * y + (x**2 - y**2)

    sandbox.rep_analyse_real_space_function_real( #(8)!
        # Data
        inp_sym="mol", #(9)!
        function=real_function, #(10)!
        grid_points=grid,
        weight=weight,
        # Thresholds
        linear_independence_threshold=1e-6, #(11)!
        integrality_threshold=1e-6, #(12)!
        eigenvalue_comparison_mode=EigenvalueComparisonMode.Modulus, #(13)!
        # Analysis options
        use_magnetic_group=None, #(14)!
        use_double_group=False, #(15)!
        use_cayley_table=True, #(16)!
        symmetry_transformation_kind=SymmetryTransformationKind.Spatial, #(17)!
        infinite_order_to_finite=None, #(18)!
        # Other options
        write_character_table=True, #(19)!
        write_overlap_eigenvalues=True, #(20)!
    )

    def complex_function(x, y, z): #(21)!
        return z**2 + 3j * x * (z**2 - y**2)

    sandbox.rep_analyse_real_space_function_complex( #(22)!
        # Data
        inp_sym="mol",
        function=complex_function,
        grid_points=grid,
        weight=weight,
        # Thresholds
        linear_independence_threshold=1e-6,
        integrality_threshold=1e-6,
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
    )
    ```

    1. :fontawesome-solid-laptop-code: This is a submodule of QSym² containing developmental features. See [Getting started/Prerequisites/Rust features/Developmental](../../getting-started/prerequisites.md/#developmental) for more information.
    2. :fontawesome-solid-laptop-code: This is a Python-exposed Rust enum, [`EigenvalueComparisonMode`](https://qsym2.dev/api/qsym2/analysis/enum.EigenvalueComparisonMode.html), for indicating the mode of eigenvalue comparison. See [Basics/Thresholds/Linear independence threshold/#Comparison mode](basics.md/#comparison-mode) for further information.
    3. :fontawesome-solid-laptop-code: This is a Python-exposed Rust enum, [`MagneticSymmetryAnalysisKind`](https://qsym2.dev/api/qsym2/drivers/representation_analysis/enum.MagneticSymmetryAnalysisKind.html), for indicating the type of magnetic symmetry to be used for symmetry analysis. See [Basics/Analysis options/#Magnetic groups](basics.md/#magnetic-groups) for further information.
    4. :fontawesome-solid-laptop-code: This is a Python-exposed Rust enum, [`SymmetryTransformationKind`](https://qsym2.dev/api/qsym2/symmetry/symmetry_transformation/enum.SymmetryTransformationKind.html), for indicating the kind of symmetry transformation to be applied on the target. See [Basics/Analysis options/#Transformation kinds](basics.md/#transformation-kinds) for further information.
    5. :fontawesome-solid-users: This specifies a $3 \times N$ array containing the Cartesian coordinates of the grid points at which the real-space function will be evaluated in the symmetry analysis. Each column of the array contains the coordinates of one grid point $\mathbfit{r}_i$.
    6. :fontawesome-solid-users: This specifies an $N$-element array containing the weight values $w(\mathbfit{r}_i)$ associated with the specified grid points $\mathbfit{r}_i$.
    7. :fontawesome-solid-users: This is an example real-valued real-space function: $f(\mathbfit{r}) = xy + (x^2 - y^2)$.
    8. :fontawesome-solid-users: This is the Python driver function for representation analysis of real-valued real-space functions.
    </br></br>:fontawesome-solid-laptop-code: This is a Python-exposed Rust function, [`rep_analyse_real_space_function_real`](https://qsym2.dev/api/qsym2/sandbox/bindings/python/representation_analysis/real_space_function/fn.rep_analyse_real_space_function_real.html). See the API documentation of this function for more details.
    9. :fontawesome-solid-users: This specifies the path to the `.qsym2.sym` file that contains the serialised results of the symmetry-group detection (see the documentation for the `out_sym` parameter of the Python [`detect_symmetry_group`](https://qsym2.dev/api/qsym2/bindings/python/symmetry_group_detection/fn.detect_symmetry_group.html) function in [Symmetry-group detection/#Parameters](../symmetry-group-detection.md/#parameters)). This file should have been generated by the [`detect_symmetry_group`](https://qsym2.dev/api/qsym2/bindings/python/symmetry_group_detection/fn.detect_symmetry_group.html) function on the underlying molecular system prior to representation analysis.
    </br></br>This name does not need to contain the `.qsym2.sym` extension. 
    </br></br>The symmetry results in this file will be used to construct the symmetry group $\mathcal{G}$ to be used in the subsequent representation analysis.
    10. :fontawesome-solid-users: This specifies the Python function that defines the real-valued real-space function to be symmetry-analysed.
    11. :fontawesome-solid-users: This specifies a floating-point value for the linear independence threshold $\lambda^{\mathrm{thresh}}_{\mathbfit{S}}$.
    For more information, see [Basics/#Thresholds](basics.md/#thresholds).
    12. :fontawesome-solid-users: This specifies a floating-point value for the integrality threshold $\lambda^{\mathrm{thresh}}_{\mathrm{int}}$.
    For more information, see [Basics/#Thresholds](basics.md/#thresholds).
    13. :fontawesome-solid-users: This specifies the threshold comparison mode for the eigenvalues of the orbit overlap matrix $\mathbfit{S}$. The possible options are:
        - `EigenvalueComparisonMode.Real`: this specifies the *real* comparison mode where the real parts of the eigenvalues are compared against the threshold,
        - `EigenvalueComparisonMode.Modulus`: this specifies the *modulus* comparison mode where the absolute values of the eigenvalues are compared against the threshold.
    </li>For more information, see [Basics/#Thresholds](basics.md/#thresholds).
    14. :fontawesome-solid-users: This specifies whether magnetic groups, if present, shall be used for symmetry analysis. The possible options are:
        - `None`: this specifies choice 1 of [Basics/Analysis options/#Magnetic groups](basics.md/#magnetic-groups) &mdash; use the irreducible representations of the unitary group $\mathcal{G}$,
        - `MagneticSymmetryAnalysisKind.Representation`: this specifies choice 2 of [Basics/Analysis options/#Magnetic groups](basics.md/#magnetic-groups) &mdash; use the irreducible representations of the magnetic group $\mathcal{M}$, if $\mathcal{M}$ is available,
        - `MagneticSymmetryAnalysisKind.Corepresentation`: this specifies choice 3 of [Basics/Analysis options/#Magnetic groups](basics.md/#magnetic-groups) &mdash; use the irreducible corepresentations of the magnetic group $\mathcal{M}$, if $\mathcal{M}$ is available.
    </li>For more information, see [Basics/Analysis options/#Magnetic groups](basics.md/#magnetic-groups).
    15. :fontawesome-solid-users: This is a boolean specifying if double groups shall be used for symmetry analysis. The possible options are:
        - `False`: use only conventional irreducible representations or corepresentations of $\mathcal{G}$,
        - `True`: use projective irreducible representations or corepresentations of $\mathcal{G}$ obtainable via its double cover $\mathcal{G}^*$.
    </li>For more information, see [Basics/Analysis options/#Double groups](basics.md/#double-groups).
    16. :fontawesome-solid-users: This is a boolean specifying if the Cayley table for the group, if available, should be used to speed up the computation of orbit overlap matrices.
    </br></br>:material-cog-sync-outline: Default: `True`.
    17. :fontawesome-solid-users: This specifies the kind of symmetry transformations to be applied to generate the orbit for symmetry analysis.
    The possible options are:
        - `SymmetryTransformationKind.Spatial`: spatial transformation only,
        - `SymmetryTransformationKind.SpatialWithSpinTimeReversal`: spatial transformation with spin-including time reversal,
        - `SymmetryTransformationKind.Spin`: spin transformation only,
        - `SymmetryTransformationKind.SpinSpatial`: coupled spin and spatial transformations.
    </li>For more information, see [Basics/Analysis options/#Transformation kinds](basics.md/#transformation-kinds).
    18. :fontawesome-solid-users: This specifies the finite order $n$ to which all infinite-order symmetry elements, if any, are restricted. The possible options are:
        - `None`: do not restrict infinite-order symmetry elements to finite order,
        - a positive integer value: restrict all infinite-order symmetry elements to this finite order (this will be ignored if the system has no infinite-order symmetry elements).
    </li>For more information, see [Basics/Analysis options/#Infinite-order symmetry elements](basics.md/#infinite-order-symmetry-elements).
    </br></br>:material-cog-sync-outline: Default: `None`.
    19. :fontawesome-solid-users: This boolean indicates if the *symbolic* character table of the prevailing symmetry group is to be printed in the output.
    </br></br>:material-cog-sync-outline: Default: `True`.
    20. :fontawesome-solid-users: This boolean indicates if the eigenspectrum of the overlap matrix for the real-space function orbit should be printed out.
    </br></br>:material-cog-sync-outline: Default: `True`.
    21. :fontawesome-solid-users: This is an example complex-valued real-space function: $f(\mathbfit{r}) = z^2 + 3i \times x(z^2 - y^2)$.
    22. :fontawesome-solid-users: This is the Python driver function for representation analysis of complex-valued real-space functions.
    </br></br>:fontawesome-solid-laptop-code: This is a Python-exposed Rust function, [`rep_analyse_real_space_function_complex`](https://qsym2.dev/api/qsym2/sandbox/bindings/python/representation_analysis/real_space_function/fn.rep_analyse_real_space_function_complex.html). See the API documentation of this function for more details.
