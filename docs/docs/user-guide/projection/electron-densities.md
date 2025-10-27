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
The projected density is thus

$$
    \hat{\mathscr{P}^{(\Gamma)}} \rho(\mathbfit{r})
        = \sum_{\gamma \delta}
            \phi_{\gamma}(\mathbfit{r}) \phi_{\delta}(\mathbfit{r})
            \frac{d_{\Gamma}}{|\mathcal{G}|} \sum_{i = 1}^{|\mathcal{G}|}
                \chi^{(\Gamma)}(g_i)^* P_{\delta \gamma}(g_i)
        = \sum_{\gamma \delta}
            \phi_{\gamma}(\mathbfit{r}) \phi_{\delta}(\mathbfit{r})
            P^{(\Gamma)}_{\delta \gamma},
$$

where $P_{\delta \gamma}(g_i)$ is the density matrix of the density transformed by $\hat{g}_i$ and

$$
    P^{(\Gamma)}_{\delta \gamma} =
        \frac{d_{\Gamma}}{|\mathcal{G}|} \sum_{i = 1}^{|\mathcal{G}|}
            \chi^{(\Gamma)}(g_i)^* P_{\delta \gamma}(g_i)
$$

is the resulting density matrix of the projected density.
It is clear that projected densities $\hat{\mathscr{P}}^{(\Gamma)} \rho(\mathbfit{r})$ are themselves densities.


## Requirements

As electron densities are expanded in atomic-orbital bases in QSym², information about their angular momenta and ordering conventions as described in [Representation analysis/Basics/Requirements/#Atomic-orbital basis angular order](../representation-analysis/basics.md/#atomic-orbital-basis-angular-order) is required.

## Parameters

!!! info "Feature requirements"

    - Using the Python API requires the [`python` feature](../../getting-started/prerequisites.md/#rust-features).

QSym² is able to perform symmetry projection for electron densities that can arise from various sources via the Python library API reading in data from Python data structures.
The way to do this is shown below.

=== "Python"
    ```python
    from qsym2 import (
        project_densities,
        SymmetryTransformationKind, #(1)!
        PyDensityReal,
        PyDensityComplex,
        MagneticSymmetryAnalysisKind, #(2)!
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

    pydens = [ #(3)!
        (
            "alpha_density", PyDensityReal(
                complex_symmetric=False,
                density_matrix=dmao_a,
                threshold=1e-7,
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

    result = project_densities( #(4)!
        # Data
        inp_sym="mol", #(5)!
        pydens=pydens, #(6)!
        projection_targets=[0, "||A|_(2g)|"], #(7)!
        pybao=pybao, #(8)!
        # Projection options
        use_magnetic_group=None, #(9)!
        use_double_group=False, #(10)!
        symmetry_transformation_kind=SymmetryTransformationKind.Spatial, #(11)!
        infinite_order_to_finite=None, #(12)!
        # Other options
        write_character_table=True, #(13)!
    ) #(14)!
    ```

    1. :fontawesome-solid-laptop-code: This is a Python-exposed Rust enum, [`SymmetryTransformationKind`](https://qsym2.dev/api/qsym2/symmetry/symmetry_transformation/enum.SymmetryTransformationKind.html), for indicating the kind of symmetry transformation to be applied on the target. See [Representation analysis/Basics/Analysis options/#Transformation kinds](../representation-analysis/basics.md/#transformation-kinds) for further information.
    2. :fontawesome-solid-laptop-code: This is a Python-exposed Rust enum, [`MagneticSymmetryAnalysisKind`](https://qsym2.dev/api/qsym2/drivers/representation_analysis/enum.MagneticSymmetryAnalysisKind.html), for indicating the type of magnetic symmetry to be used for symmetry projection. Note that projections using corepresentations of magnetic-represented groups are not yet supported. See [representation analysis/Basics/Analysis options/#Magnetic groups](../representation-analysis/basics.md/#magnetic-groups) for further information.
    3. :fontawesome-solid-users: This specifies a density matrix in the atomic-orbital basis, $\mathbfit{P}$, for one electron density, which is a $N_{\mathrm{bas}} \times N_{\mathrm{bas}}$ `numpy` array. The number of basis functions, $N_{\mathrm{bas}}$, is always the number of *spatial* basis functions because electron densities are entirely spatial quantities.
    4. :fontawesome-solid-users: This is the Python driver function for symmetry projection of electron densities.
    </br></br>:fontawesome-solid-laptop-code: This is a Python-exposed Rust function, [`project_densities`](https://qsym2.dev/api/qsym2/bindings/python/projection/density/fn.project_densities.html). See the API documentation of this function for more details.
    5. :fontawesome-solid-users: This specifies the path to the `.qsym2.sym` file that contains the serialised results of the symmetry-group detection (see the documentation for the `out_sym` parameter of the Python [`detect_symmetry_group`](https://qsym2.dev/api/qsym2/bindings/python/symmetry_group_detection/fn.detect_symmetry_group.html) function in [Symmetry-group detection/#Parameters](../symmetry-group-detection.md/#parameters)). This file should have been generated by the [`detect_symmetry_group`](https://qsym2.dev/api/qsym2/bindings/python/symmetry_group_detection/fn.detect_symmetry_group.html) function on the underlying molecular system prior to representation analysis.
    </br></br>This name does not need to contain the `.qsym2.sym` extension.
    </br></br>The symmetry results in this file will be used to construct the symmetry group $\mathcal{G}$ to be used in the subsequent representation analysis.
    6. :fontawesome-solid-users: This specifies a list of densities to be symmetry-projected. Each element in this list is a tuple containing a brief string description for the density and a specification for the density itself.
    7. :fontawesome-solid-users: This specifies the irreducible representations of the group onto which the provided densities are projected. Each element in this list is either an integer indicating the index of an irreducible representation in the group, or a string of the format `|^(presup)_(presub)|main|^(postsup)_(postsub)|` (note that four `|` characters are required), such as `||A|_(2g)|` or `||E|^(')_(1)|`, specifying the label of the irreducible representation.
    8. :fontawesome-solid-users: This specifies the basis angular order information for the underlying basis. See [Representation analysis/Basics/Requirements/#Atomic-orbital basis angular order](../representation-analysis/basics.md/#atomic-orbital-basis-angular-order) for details of how to specify this.
    9. :fontawesome-solid-users: This specifies whether magnetic groups, if present, shall be used for symmetry analysis. The possible options are:
        - `None`: this specifies choice 1 of [Representation analysis/Basics/Analysis options/#Magnetic groups](../representation-analysis/basics.md/#magnetic-groups) &mdash; use the irreducible representations of the unitary group $\mathcal{G}$,
        - `MagneticSymmetryAnalysisKind.Representation`: this specifies choice 2 of [Representation analysis/Basics/Analysis options/#Magnetic groups](../representation-analysis/basics.md/#magnetic-groups) &mdash; use the irreducible representations of the magnetic group $\mathcal{M}$, if $\mathcal{M}$ is available,
        - `MagneticSymmetryAnalysisKind.Corepresentation`: this specifies choice 3 of [Representation analysis/Basics/Analysis options/#Magnetic groups](../representation-analysis/basics.md/#magnetic-groups) &mdash; use the irreducible corepresentations of the magnetic group $\mathcal{M}$, if $\mathcal{M}$ is available.
    </li>For more information, see [Representation analysis/Basics/Analysis options/#Magnetic groups](../representation-analysis/basics.md/#magnetic-groups).
    10. :fontawesome-solid-users: This is a boolean specifying if double groups shall be used for symmetry analysis. The possible options are:
        - `False`: use only conventional irreducible representations or corepresentations of $\mathcal{G}$,
        - `True`: use projective irreducible representations or corepresentations of $\mathcal{G}$ obtainable via its double cover $\mathcal{G}^*$.
    </li>For more information, see [Representation analysis/Basics/Analysis options/#Double groups](../representation-analysis/basics.md/#double-groups).
    11. :fontawesome-solid-users: This specifies the kind of symmetry transformations to be applied to generate the orbit for symmetry analysis.
    The possible options are:
        - `SymmetryTransformationKind.Spatial`: spatial transformation only,
        - `SymmetryTransformationKind.SpatialWithSpinTimeReversal`: spatial transformation with spin-including time reversal,
        - `SymmetryTransformationKind.Spin`: spin transformation only,
        - `SymmetryTransformationKind.SpinSpatial`: coupled spin and spatial transformations.
    </li>For more information, see [Representation analysis/Basics/Analysis options/#Transformation kinds](../representation-analysis/basics.md/#transformation-kinds).
    12. :fontawesome-solid-users: This specifies the finite order $n$ to which all infinite-order symmetry elements, if any, are restricted. The possible options are:
        - `None`: do not restrict infinite-order symmetry elements to finite order,
        - a positive integer value: restrict all infinite-order symmetry elements to this finite order (this will be ignored if the system has no infinite-order symmetry elements).
    </li>For more information, see [Representation analysis/Basics/Analysis options/#Infinite-order symmetry elements](../representation-analysis/basics.md/#infinite-order-symmetry-elements).
    </br></br>:material-cog-sync-outline: Default: `None`.
    13. :fontawesome-solid-users: This boolean indicates if the *symbolic* character table of the prevailing symmetry group is to be printed in the output.
    </br></br>:material-cog-sync-outline: Default: `True`.
    14. :fontawesome-solid-users: The returned result is a list of tuples, each of which contains a string describing the density being projected and a dictionary a dictionary in which the keys are the irreducible representation labels and the values are the corresponding projected density.
