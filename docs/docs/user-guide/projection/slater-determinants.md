---
title: Slater determinants
description: Configurable parameters for Slater determinant symmetry projection
---

# Slater determinants

Let $\Psi_{\mathrm{SD}}$ be an $N_{\mathrm{e}}$-electron Slater determinant constructed from $N_{\mathrm{e}}$ occupied spin-orbitals $\chi_i(\mathbfit{x})$ written in terms of the composite spin-spatial coordinates $\mathbfit{x}$:

$$
    \Psi_{\mathrm{SD}} =
        \sqrt{N_{\mathrm{e}}!} \hat{\mathscr{A}}
        \left[ \prod_{i=1}^{N_{\mathrm{e}}} \chi_i(\mathbfit{x}_i) \right],
$$

where $\hat{\mathscr{A}}$ is the antisymmetriser in the symmetric group $\operatorname{Sym}(N_{\mathrm{e}})$ acting on the electron labels.

The projected Slater determinant is

$$
    \hat{\mathscr{P}^{(\Gamma)}} \Psi_{\mathrm{SD}}
        = \frac{d_{\Gamma}}{|\mathcal{G}|} \sum_{i = 1}^{|\mathcal{G}|}
                \chi^{(\Gamma)}(g_i)^* (\hat{g}_i \Psi_{\mathrm{SD}}),
$$

which is clearly no longer a Slater determinant but a linear combination of symmetry-equivalent ones that are in general non-orthogonal.


## Requirements

### Atomic-orbital basis angular order

As the molecular orbitals $\chi_i(\mathbfit{x}_i)$ are expressed in terms of Gaussian atomic orbitals, QSym² requires information about their angular momenta and ordering conventions as described in [Representation analysis/Basics/Requirements/#Atomic-orbital basis angular order](../representation-analysis/basics.md/#atomic-orbital-basis-angular-order) in order to symmetry-transform $\Psi_{\mathrm{SD}}$.

### Basis overlap matrix

Since symmetry projection is a linear-space operation, inner products are not required to be defined on the space in which $\Psi_{\mathrm{SD}}$ lives.
However, if one seeks to normalise the projected multi-determinantal wavefunction $\hat{\mathscr{P}^{(\Gamma)}} \Psi_{\mathrm{SD}}$, then one needs to define an appropriate inner product, which thus requires the specification of a basis overlap matrix as explained in [Representation analysis/Basics/Requirements/#Basis overlap matrix](../representation-analysis/basics.md/#basis-overlap-matrix).
This is also needed if one seeks to construct the density matrices of the resulting multi-determinantal wavefunctions.

## Parameters

!!! info "Feature requirements"

    - Using the Python API requires the [`python` feature](../../getting-started/prerequisites.md/#rust-features).


At the moment, QSym² offers one way to perform symmetry projection for Slater determinants, which is:

- via the Python library API reading in data from Python data structures.

More methods might become possible in the future. The parameter specifications for the one existing method are shown below.

=== "Python"
    ```python
    from qsym2 import (
        project_slater_determinant,
        SymmetryTransformationKind, #(1)!
        MagneticSymmetryAnalysisKind, #(2)!
        PySpinConstraint, #(3)!
        PySpinOrbitCoupled, #(4)!
        PySlaterDeterminantReal,
        PySlaterDeterminantComplex,
    )

    ca = np.array([ #(5)!
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
    
    occa = np.array([1.0, 1.0]) #(6)!
    occb = np.array([1.0, 0.0])

    ea = np.array([-0.51, -0.38]) #(7)!
    eb = np.array([-0.50, +0.02])

    pydet = PySlaterDeterminantReal( #(8)!
        structure_constraint=PySpinConstraint.Unrestricted,
        complex_symmetric=False,
        coefficients=[ca, cb],
        occupations=[occa, occb],
        threshold=1e-7,
        mo_energies=[ea, eb],
        energy=-1.30,
    )

    irreps, multidets = project_slater_determinant( #(9)!
        # Data
        inp_sym="mol", #(10)!
        pydet=pydet, #(11)!
        projection_targets=[0, "||A|_(2g)|"], #(12)!
        pybaos=[pybao], #(13)!
        sao=sao_spatial, #(14)!
        sao_h=None, #(15)!
        # Thresholds
        density_matrix_calculation_thresholds=(1e-7, 1e-7), #(16)!
        # Projection options
        use_magnetic_group=None, #(17)!
        use_double_group=False, #(18)!
        symmetry_transformation_kind=SymmetryTransformationKind.Spatial, #(19)!
        infinite_order_to_finite=None, #(20)!
        # Other options
        write_character_table=True, #(21)!
    ) #(22)!
    ```

    1. :fontawesome-solid-laptop-code: This is a Python-exposed Rust enum, [`SymmetryTransformationKind`](https://qsym2.dev/api/qsym2/symmetry/symmetry_transformation/enum.SymmetryTransformationKind.html), for indicating the kind of symmetry transformation to be applied on the target. See [Representation analysis/Basics/Analysis options/#Transformation kinds](../representation-analysis/basics.md/#transformation-kinds) for further information.
    2. :fontawesome-solid-laptop-code: This is a Python-exposed Rust enum, [`MagneticSymmetryAnalysisKind`](https://qsym2.dev/api/qsym2/drivers/representation_analysis/enum.MagneticSymmetryAnalysisKind.html), for indicating the type of magnetic symmetry to be used for symmetry projection. See [Representation analysis/Basics/Analysis options/#Magnetic groups](../representation-analysis/basics.md/#magnetic-groups) for further information.
    3. :fontawesome-solid-laptop-code: This is a Python-exposed Rust enum, [`PySpinConstraint`](https://qsym2.dev/api/qsym2/bindings/python/integrals/enum.PySpinConstraint.html), for indicating the spin constraint applicable to the Slater determinant. In the Python API, only two spin spaces arranged in decreasing-$m_s$ order are permitted because Python enums do not support associated values.
    4. :fontawesome-solid-laptop-code: This is a Python-exposed Rust enum, [`PySpinOrbitCoupled`](https://qsym2.dev/api/qsym2/bindings/python/integrals/enum.PySpinOrbitCoupled.html), for indicating the spin--orbit-coupled structure applicable to the Slater determinant. In the Python API, only two-component j-adapted basis structures are permitted.
    5. :fontawesome-solid-users: This specifies a coefficient matrix for one spin space, which is a $N_{\mathrm{bas}} \times N_{\mathrm{MO}}$ `numpy` array. The number of basis functions, $N_{\mathrm{bas}}$, depends on the underlying spin constraint: for *generalised* spin constraint, this is twice the number of spatial basis functions, whereas for *restricted* and *unrestricted* spin constraints, this is the same as the number of spatial basis functions. Each column in the array specifies a molecular orbital which can be occupied or virtual as specified by the occupation numbers.
    6. :fontawesome-solid-users: This specifies an occupation number vector for one spin space, which is a one-dimensional `numpy` array of size $N_{\mathrm{MO}}$. Each value in this array gives the occupation number for the corresponding molecular orbital. Fractional values are allowed, but only when occupation numbers are either $0$ or $1$ can the Slater determinant symmetry be well-defined (otherwise the collection of fractionally occupied molecular orbitals does not actually form a single-determinantal wavefunction).
    7. :fontawesome-solid-users: This specifies an optional orbital energy vector for one spin space, which is a one-dimensional `numpy` array of size $N_{\mathrm{MO}}$. Each value in this array gives the orbital energy for the corresponding molecular orbital.
    8. :fontawesome-solid-users: [`PySlaterDeterminantReal`](https://qsym2.dev/api/qsym2/bindings/python/representation_analysis/slater_determinant/struct.PySlaterDeterminantReal.html) constructs a *real-valued* Slater determinant object. If a *complex-valued* Slater determinant is required instead, use [`PySlaterDeterminantComplex`](https://qsym2.dev/api/qsym2/bindings/python/representation_analysis/slater_determinant/struct.PySlaterDeterminantComplex.html). For more information, see [Representation analysis/Slater determinants/Parameters/Python](../representation-analysis/slater-determinants.md#__tabbed_1_2).
    9. :fontawesome-solid-users: This is the Python driver function for symmetry projection of Slater determinants.
    </br></br>:fontawesome-solid-laptop-code: This is a Python-exposed Rust function, [`project_slater_determinant`](https://qsym2.dev/api/qsym2/bindings/python/projection/slater_determinant/fn.project_slater_determinant.html). See the API documentation of this function for more details.
    10. :fontawesome-solid-users: This specifies the path to the `.qsym2.sym` file that contains the serialised results of the symmetry-group detection (see the documentation for the `out_sym` parameter of the Python [`detect_symmetry_group`](https://qsym2.dev/api/qsym2/bindings/python/symmetry_group_detection/fn.detect_symmetry_group.html) function in [Symmetry-group detection/#Parameters](../symmetry-group-detection.md/#parameters)). This file should have been generated by the [`detect_symmetry_group`](https://qsym2.dev/api/qsym2/bindings/python/symmetry_group_detection/fn.detect_symmetry_group.html) function on the underlying molecular system prior to symmetry projection.
    </br></br>This name does not need to contain the `.qsym2.sym` extension.
    </br></br>The symmetry results in this file will be used to construct the symmetry group $\mathcal{G}$ to be used in the subsequent symmetry projection.
    11. :fontawesome-solid-users: This specifies the Slater determinant to be symmetry-projected.
    12. :fontawesome-solid-users: This specifies the irreducible representations of the group onto which the provided densities are projected. Each element in this list is either an integer indicating the index of an irreducible representation in the group, or a string of the format `|^(presup)_(presub)|main|^(postsup)_(postsub)|` (note that four `|` characters are required), such as `||A|_(2g)|` or `||E|^(')_(1)|`, specifying the label of the irreducible representation.
    13. :fontawesome-solid-users: This specifies the basis angular order information for the underlying basis. Each item in the list is for one explicit component in the coefficient matrices. See [Representation analysis/Basics/Requirements/#Atomic-orbital basis angular order](../representation-analysis/basics.md/#atomic-orbital-basis-angular-order) for details of how to specify this.
    14. :fontawesome-solid-users: This specifies the optional two-centre atomic-orbital overlap matrix as a two-dimensional `numpy` array. The dimensions of this matrix must be $n_{\mathrm{comps}}N_{\mathrm{bas}} \times n_{\mathrm{comps}}N_{\mathrm{bas}}$, where $N_{\mathrm{bas}}$ is the number of basis functions specified in the basis angular order information, and $n_{\mathrm{comps}}$ is either $1$ or the total number of explicit components per coefficient matrix.
    </br></br>This is only required for norm and density matrix calculations.
    15. :fontawesome-solid-users: This specifies the optional complex-symmetric two-centre atomic-orbital spatial matrix as a two-dimensional `numpy` array. The dimensions of this matrix must be $n_{\mathrm{comps}}N_{\mathrm{bas}} \times n_{\mathrm{comps}}N_{\mathrm{bas}}$, where $N_{\mathrm{bas}}$ is the number of basis functions specified in the basis angular order information, and $n_{\mathrm{comps}}$ is either $1$ or the total number of explicit components per coefficient matrix. This is only required if antiunitary operations are ppresent.
    </br></br>:material-cog-sync-outline: Default: `None`.
    16. :fontawesome-solid-users: This specifies an optional pair of thresholds for Löwdin pairing, one for checking zero off-diagonal values and one for checking zero overlaps, when computing multi-determinantal density matrices.
    <br></br>If `None`, no density matrices will be computed.
    17. :fontawesome-solid-users: This specifies whether magnetic groups, if present, shall be used for symmetry projection. The possible options are:
        - `None`: this specifies choice 1 of [Representation analysis/Basics/Analysis options/#Magnetic groups](../representation-analysis/basics.md/#magnetic-groups) &mdash; use the irreducible representations of the unitary group $\mathcal{G}$,
        - `MagneticSymmetryAnalysisKind.Representation`: this specifies choice 2 of [Representation analysis/Basics/Analysis options/#Magnetic groups](../representation-analysis/basics.md/#magnetic-groups) &mdash; use the irreducible representations of the magnetic group $\mathcal{M}$, if $\mathcal{M}$ is available,
        - `MagneticSymmetryAnalysisKind.Corepresentation`: this specifies choice 3 of [Representation analysis/Basics/Analysis options/#Magnetic groups](../representation-analysis/basics.md/#magnetic-groups) &mdash; use the irreducible corepresentations of the magnetic group $\mathcal{M}$, if $\mathcal{M}$ is available. However, this is not yet supported for symmetry projection.
    </li>For more information, see [Representation analysis/Basics/Analysis options/#Magnetic groups](../representation-analysis/basics.md/#magnetic-groups).
    18. :fontawesome-solid-users: This is a boolean specifying if double groups shall be used for symmetry projection. The possible options are:
        - `False`: use only conventional irreducible representations or corepresentations of $\mathcal{G}$,
        - `True`: use projective irreducible representations or corepresentations of $\mathcal{G}$ obtainable via its double cover $\mathcal{G}^*$.
    </li>For more information, see [Representation analysis/Basics/Analysis options/#Double groups](../representation-analysis/basics.md/#double-groups).
    19. :fontawesome-solid-users: This specifies the kind of symmetry transformations to be applied to generate the orbit for symmetry projection.
    The possible options are:
        - `SymmetryTransformationKind.Spatial`: spatial transformation only,
        - `SymmetryTransformationKind.SpatialWithSpinTimeReversal`: spatial transformation with spin-including time reversal,
        - `SymmetryTransformationKind.Spin`: spin transformation only,
        - `SymmetryTransformationKind.SpinSpatial`: coupled spin and spatial transformations.
    </li>For more information, see [Representation analysis/Basics/Analysis options/#Transformation kinds](../representation-analysis/basics.md/#transformation-kinds).
    20. :fontawesome-solid-users: This specifies the finite order $n$ to which all infinite-order symmetry elements, if any, are restricted. The possible options are:
        - `None`: do not restrict infinite-order symmetry elements to finite order,
        - a positive integer value: restrict all infinite-order symmetry elements to this finite order (this will be ignored if the system has no infinite-order symmetry elements).
    </li>For more information, see [Representation analysis/Basics/Analysis options/#Infinite-order symmetry elements](../representation-analysis/basics.md/#infinite-order-symmetry-elements).
    </br></br>:material-cog-sync-outline: Default: `None`.
    21. :fontawesome-solid-users: This boolean indicates if the *symbolic* character table of the prevailing symmetry group is to be printed in the output.
    </br></br>:material-cog-sync-outline: Default: `True`.
    22. The [`project_slater_determinant`](https://qsym2.dev/api/qsym2/bindings/python/projection/slater_determinant/fn.project_slater_determinant.html) function returns a tuple: where the first item is a list of the labels of the subspaces used for projection, and the second item is either a [`PyMultiDeterminantsReal`](https://qsym2.dev/api/qsym2/bindings/python/representation_analysis/multideterminant/struct.PyMultiDeterminantsReal.html) object or a [`PyMultiDeterminantsComplex`](https://qsym2.dev/api/qsym2/bindings/python/representation_analysis/multideterminant/struct.PyMultiDeterminantsComplex.html) containing the Slater determinant basis and the linear combination coefficients as a two-dimensional array with each column corresponding to one projected state.
