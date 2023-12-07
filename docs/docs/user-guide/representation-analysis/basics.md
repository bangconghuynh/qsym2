---
title: Basics
description: Basic information for representation analysis
---

# Basics

!!! note

    Unless stated otherwise, $\mathcal{G}$ denotes a generic symmetry group.

Let $V$ be a linear space and $\mathbfit{w} \in V$ be an element whose symmetry with respect to a group $\mathcal{G}$ is to be determined by QSym².
This amounts to identifying and characterising the linear subspace $W \subseteq V$ spanned by the orbit

$$
    \mathcal{G} \cdot \mathbfit{w} = \{ \hat{g}_i \mathbfit{w} \ :\ g_i \in \mathcal{G} \}.
$$

The mathematical details of this method are described in [Section 2.4 of the QSym² paper](../../about/authorship.md#publications).
Here, it suffices to summarise the key ingredients and considerations that are required for this method to work.


## Requirements

### Basis overlap matrix

Let $\mathcal{B}_V$ be a basis for the linear space $V$, and let us also endow $V$ with an inner product $\braket{\cdot | \cdot}$.
For any chosen $\mathcal{B}_V = \{ \mathbfit{v}_{\mu} \ :\ 1 \le \mu \le \dim V \}$, QSym² requires the *overlap matrix* $\mathbfit{S}_V$ as part of the representation analysis procedure:

$$
    S_{V, \mu \nu} = \braket{\mathbfit{v}_{\mu} | \mathbfit{v}_{\nu}}.
$$

A knowledge of $\mathbfit{S}_V$ lets QSym² determine the overlap matrix $\mathbfit{S}$ between the elements of the orbit $\mathcal{G} \cdot \mathbfit{w}$:

$$
    S_{ij} = \braket{\hat{g}_i \mathbfit{w} | \hat{g}_j \mathbfit{w}},
$$

which is then used to determine the representation matrices of the elements of $\mathcal{G}$ on $W$, as described in [Section 2.4 of the QSym² paper](../../about/authorship.md#publications).

Depending on the nature of the basis functions in $\mathcal{B}_V$, the overlap matrix $\mathbfit{S}_V$ might have already been computed by other quantum-chemistry calculation programs, in which case it can simply be read in by QSym².
But if $\mathbfit{S}_V$ is not readily available, then information about $\mathcal{B}_V$ must be made available to QSym² so that QSym² can compute $\mathbfit{S}_V$ on-the-fly.
The exact requirements for the various types of representation analysis that QSym² supports will be explained in the relevant sections in this guide.

### Atomic-orbital basis angular order

If $\mathcal{B}_V$ is a basis consisting of atomic orbitals, then, to carry out the transformations $\hat{g}_i \mathbfit{w}$ that are required to construct the orbit $\mathcal{G} \cdot \mathbfit{w}$ for representation analysis, QSym² needs to know how the basis atomic orbitals are transformed under spatial rotations.
This is equivalent to knowing the following for any shell of atomic orbitals:

- the atom on which the shell is centred,
- the angular momentum degree of the shell,
- whether the angular parts of the atomic orbitals in the shell (which are real solid harmonic functions) are expressed in Cartesian coordinates or in spherical polar coordinates, and
- the ordering of the functions in the shell.

These pieces of information are collectively referred to in QSym² as the *basis angular order* information which must be specified for any representation analysis performed on quantities expressed in terms of atomic orbitals.
Whenever possible, QSym² will attempt to construct this from available data, but if this cannot be done, then there are several ways to specify this manually, as shown below.

=== "Command-line interface"
    ```yaml
    analysis_targets:
      - !SlaterDeterminant
        source: !Binaries #(1)!
          ...
          bao: #(2)!
          - atom: [0, "O"] #(3)!
            basis_shells: #(4)!
            - l: 0 #(5)!
              shell_order: !PureIncreasingm #(6)!
            - l: 1
              shell_order: !PureDecreasingm
            - l: 3
              shell_order: !PureCustom #(7)!
              - 0
              - 1
              - -1
              - 2
              - -2
              - 3
              - -3
          - atom: [1, "H"]
            basis_shells:
            - l: 1
              shell_order: !CartLexicographic
            - l: 3
              shell_order: !CartQChem
            - l: 2
              shell_order: !CartCustom #(8)!
              - [2, 0, 0]
              - [0, 2, 0]
              - [0, 0, 2]
              - [1, 1, 0]
              - [1, 0, 1]
              - [0, 1, 1]
          - atom: [2, "H"]
            basis_shells:
            - l: 1
              shell_order: !CartLexicographic
            - l: 3
              shell_order: !CartQChem
            - l: 2
              shell_order: !CartCustom
              - [2, 0, 0]
              - [0, 2, 0]
              - [0, 0, 2]
              - [1, 1, 0]
              - [1, 0, 1]
              - [0, 1, 1]
    ```

    1. :fontawesome-solid-users: This is an example data source ([Slater determinant specified via binary coefficient files](slater-determinant.md)) where a manual specification of basis angular order is required. If other data sources for other analysis targets also require a manual specification of basis angular order, the format will be the same.
    2. :fontawesome-solid-users: Each item in this list specifies the angular order information for all shells on one atom in the molecule.</br></br>:fontawesome-solid-laptop-code: Under the hood, this key wraps around the [`InputBasisAngularOrder`](https://qsym2.dev/api/qsym2/interfaces/input/ao_basis/struct.InputBasisAngularOrder.html) struct which consists of a vector of [`InputBasisAtom`](https://qsym2.dev/api/qsym2/interfaces/input/ao_basis/struct.InputBasisAtom.html) structs.
    3. :fontawesome-solid-users: This key, `atom`, specifies the index and name of an atom in the basis set.
    4. :fontawesome-solid-users: This key, `basis_shells`, gives the ordered shells associated with this atom. Each item in this list specifies the angular momentum information of one shell centred on the prevailing atom.</br></br>:fontawesome-solid-laptop-code: Under the hood, this key is a vector of [`InputBasisShell`](https://qsym2.dev/api/qsym2/interfaces/input/ao_basis/struct.InputBasisShell.html) structs.
    5. :fontawesome-solid-users: This key, `l`, specifies the angular momentum degree of this shell.
    6. :fontawesome-solid-users: This key, `shell_order`, specifies the type and ordering of the basis functions in this shell. The following variants are supported:
        - `!PureIncreasingm`: the basis functions are pure real solid harmonics, arranged in increasing-$m_l$ order,
        - `!PureDecreasingm`: the basis functions are pure real solid harmonics, arranged in decreasing-$m_l$ order,
        - `!PureCustom`: the basis functions are pure real solid harmonics, arranged in a custom order to be specified by the $m_l$ values,
        - `!CartLexicographic`: the basis functions are Cartesian real solid harmonics, arranged in lexicographic order,
        - `!CartQChem`: the basis functions are Cartesian real solid harmonics, arranged in Q-Chem order,
        - `!CartCustom`: the basis functions are pure real solid harmonics, arranged in a custom order to be specified by the ordered exponent tuples.
    7. :fontawesome-solid-users: The order of the elements in this list specifies the $m_l$ order of the functions in this shell. Invalid $m_l$ values for a specified $l$ value (*i.e.* $\lvert m_l \rvert > l$) will result in an error. Invalid number of elements (*i.e.* not $2l + 1$) will also result in an error.
    8. :fontawesome-solid-users: Each element in this list is a tuple `[n_x, n_y, n_z]` containing the exponents of one Cartesian component: $x^{n_x} y^{n_y} z^{n_z}$. The order of the elements in this list specifies the order of the Cartesian components in this shell. Invalid exponents (*i.e.* $n_x + n_y + n_z \ne l$) will result in an error. Invalid number of elements (*i.e.* not $(l + 1)(l + 2)/2$) will also result in an error.

=== "Python"
    ```python
    from itertools import product
    from qsym2 import PyBasisAngularOrder

    def get_qchem_cartesian_order(l_degree: int) -> list[tuple[int, int, int]]: #(1)!
        r"""Returns the `Q-Chem`-ordered list of angular Cartesian functions of degree
        `l_degree`.

        :param l_degree: Degree of Cartesian functions.

        :returns: List of tuples of exponents of the Cartesian functions in the shell.
        """
        return [
            (tup.count(0), tup.count(1), tup.count(2))
            for tup in product(range(3), repeat=l_degree)
            if tup == tuple(sorted(tup, reverse=True))
        ]

    pybao = PyBasisAngularOrder([ #(2)!
        (
            "O", #(3)!
            [ #(4)!
                ("S", False, True), #(5)!
                ("P", False, False), #(6)!
                ("F", False, [0, 1, -1, 2, -2, 3, -3]), #(7)!
            ]
        ),
        (
            "H",
            [
                ("P", True, None), #(8)!
                ("F", True, get_qchem_cartesian_order(3)), #(9)!
                ("D", True, [ #(10)!
                    (2, 0, 0),
                    (0, 2, 0),
                    (0, 0, 2)
                    (1, 1, 0),
                    (1, 0, 1),
                    (0, 1, 1),
                ]),
            ]
        ),
        (
            "H",
            [
                ("P", True, None),
                ("F", True, get_qchem_cartesian_order(3)),
                ("D", True, [
                    (2, 0, 0),
                    (0, 2, 0),
                    (0, 0, 2)
                    (1, 1, 0),
                    (1, 0, 1),
                    (0, 1, 1),
                ]),
            ]
        ),
    ])
    ```

    1. :fontawesome-solid-laptop-code: :fontawesome-solid-users: This function provides a convenient Pythonic way to generate the Q-Chem order of Cartesian functions in a shell.
    2. :fontawesome-solid-users: Each item in this list specifies the angular order information for all shells on one atom in the molecule.</br></br>:fontawesome-solid-laptop-code: The [`PyBasisAngularOrder`](https://qsym2.dev/api/qsym2/bindings/python/integrals/struct.PyBasisAngularOrder.html) class is a Python-exposed Rust structure for marshalling basis angular order information between Python and Rust. This is subsequently converted to the pure Rust structure [`BasisAngularOrder`](https://qsym2.dev/api/qsym2/basis/ao/struct.BasisAngularOrder.html). Under the hood, the initialiser of this class takes in a list of tuples, each of which provides information for one basis atom. The API documentation for [`PyBasisAngularOrder`](https://qsym2.dev/api/qsym2/bindings/python/integrals/struct.PyBasisAngularOrder.html) can be consulted for further information.
    3. :fontawesome-solid-users: The first element of this tuple specifies the name of an atom in the order that it appears in the basis set.
    4. :fontawesome-solid-users: The second element of this tuple gives the ordered shells associated with this atom. Each item in this list is a tuple specifying the angular momentum information of one shell centred on the prevailing atom and has the form `(angmom, cart, order)` where:
        - `angmom` is a symbol such as `"S"` or `"P"` for the angular momentum of the shell,
        - `cart` is a boolean indicating if the functions in the shell are Cartesian (`True`)
        or pure / solid harmonics (`False`), and
        - `order` specifies how the functions in the shell are ordered:
            - if `cart` is `True`, `order` can be `None` for lexicographic order, or a list of
            tuples `(n_x, n_y, n_z)` specifying a custom order for the Cartesian functions where
            `n_x`, `n_y`, and `n_z` are the $x$-, $y$-, and $z$-exponents for a Cartesian component $x^{n_x} y^{n_y} z^{n_z}$;
            - if `cart` is `False`, `order` can be `True` for increasing-$m_l$ order, `False` for
            decreasing-$m_l$ order, or a list of $m_l$ values for custom order.
    5. :fontawesome-solid-users: This example specifies a spherical $S$-shell in which functions are arranged in increasing-$m_l$ order.
    6. :fontawesome-solid-users: This example specifies a spherical $P$-shell in which functions are arranged in decreasing-$m_l$ order.
    7. :fontawesome-solid-users: This example specifies a spherical $F$-shell in which functions are arranged in a custom $m_l$ order: $0, +1, -1, +2, -2, +3, -3$.
    8. :fontawesome-solid-users: This example specifies a Cartesian $P$-shell in which functions are arranged in lexicographic order.
    9. :fontawesome-solid-users: This example specifies a Cartesian $F$-shell in which functions are arranged in Q-Chem order.
    10. :fontawesome-solid-users: This example specifies a Cartesian $D$-shell in which functions are arranged in a custom order: $x^2, y^2, z^2, xy, xz, yz$.


## Thresholds

### Linear independence threshold

As explained in [Section 3.2.1 of the QSym² paper](../../about/authorship.md#publications), for every quantity $\mathbfit{w}$ that is to be symmetry-analysed via the orbit $\mathcal{G} \cdot \mathbfit{w}$, the overlap matrix $\mathbfit{S}$ between the elements in $\mathcal{G} \cdot \mathbfit{w}$ needs to be computed, and a threshold $\lambda^{\mathrm{thresh}}_{\mathbfit{S}}$ needs to be chosen to determine which of the eigenvalues of $\mathbfit{S}$ are non-zero.
This is so that linearly dependent elements in $\mathcal{G} \cdot \mathbfit{w}$ can be projected out and the space $W$ can be correctly identified.
The choice of the *linear independence threshold* $\lambda^{\mathrm{thresh}}_{\mathbfit{S}}$ is therefore vital for a numerically stable and meaningful symmetry analysis from QSym².

How does one pick a sensible value for this threshold?
The answer to this question depends on the eigenspectrum of $\mathbfit{S}$, and unfortunately, there is no *a priori* way to determine the best possible value of $\lambda^{\mathrm{thresh}}_{\mathbfit{S}}$ without examining this eigenspectrum first.
Fortunately, in all cases, QSym² can help with this by printing out the eigenvalue of $\mathbfit{S}$ immediately above the chosen threshold, $\lambda^{>}_{\mathbfit{S}}$, and also the eigenvalue of $\mathbfit{S}$ immediately below the chosen threshold, $\lambda^{<}_{\mathbfit{S}}$.

The author recommends that a good threshold choice is one for which

$$
    \log_{10}\lambda^{>}_{\mathbfit{S}} - \log_{10}\lambda^{<}_{\mathbfit{S}} \ge 3,
$$

*i.e.* the threshold cuts through a gap of at least three orders of magnitude in the eigenspectrum of $\mathbfit{S}$.
If such a gap does not exist, then $\mathcal{G} \cdot \mathbfit{w}$ contains all linearly independent elements, in which case:

- if $\mathbfit{w}$ has been tightly converged and is of a high numerical quality (relative to $\lambda^{\mathrm{thresh}}_{\mathbfit{S}}$), then it can be confidently concluded that $\mathbfit{w}$ transforms as the regular representation of $\mathcal{G}$;
- however, if $\mathbfit{w}$ has been rather poorly converged and is of a low numerical quality (relative to $\lambda^{\mathrm{thresh}}_{\mathbfit{S}}$), then any observed symmetry breaking could be artificial, and it is therefore advisable that $\mathbfit{w}$ be recomputed to a better numerical quality to ascertain the nature of any symmetry breaking.

#### Comparison mode

QSym² offers two modes of comparing the eigenvalues of $\mathbfit{S}$ with the threshold $\lambda^{\mathrm{thresh}}_{\mathbfit{S}}$:

- *real* mode: the real parts of the eigenvalues are compared against the threshold,
- *modulus* mode: the absolute values of the eigenvalues are compared against the threshold.

If the overlap matrix $\mathbfit{S}$ has negative or complex eigenvalues, the two comparison modes may give rise to different $W$ spaces and hence different symmetry assignments.
Care must therefore be taken in choosing the appropriate comparison mode for the system being studied.

### Integrality threshold

The crux of the symmetry analysis of $\mathbfit{w}$ with respect to a group $\mathcal{G}$ is the decomposition of the space $W \subseteq V$ spanned by the orbit $\mathcal{G} \cdot \mathbfit{w}$ into known irreducible representation spaces of $\mathcal{G}$ on $V$:

$$
    W = \bigoplus_{i} \Gamma_i^{\otimes k_i},
$$

where $\Gamma_i$ is an irreducible representation space of $\mathcal{G}$, $k_i$ its multiplicity in the decomposition of $W$, and the direct sum runs over all irreducible representation spaces of $\mathcal{G}$.
The decomposition of $W$ is then equivalent to finding the values of $k_i$ which must all be non-negative integers.
Numerically, however, the $k_i$ are represented and determined in QSym² as floating point numbers whose integrality must be verified.
This thus requires another threshold, $\lambda^{\mathrm{thresh}}_{\mathrm{int}}$, to be chosen.

In most cases, if $\mathbfit{w}$ is of a decent numerical quality and if the linear independence threshold $\lambda^{\mathrm{thresh}}_{\mathbfit{S}}$ described above has been chosen appropriately such that the space $W$ is well-behaved, then the default value of $\lambda^{\mathrm{thresh}}_{\mathrm{int}} = 10^{-7}$ should be more than good enough.
However, if QSym² complains about significant non-integrality in the obtained values of $k_i$, then, this is most likely symptomatic of a poorly chosen linear independence threshold $\lambda^{\mathrm{thresh}}_{\mathbfit{S}}$ which leads to an ill-formed $W$ space that admits an ill-defined decomposition into irreducible representation spaces of $\mathcal{G}$.
In this situation, rather than trying to unreasonably relax the integrality threshold $\lambda^{\mathrm{thresh}}_{\mathrm{int}}$, it is recommended that either the linear independence threshold $\lambda^{\mathrm{thresh}}_{\mathbfit{S}}$ should be revised or the quality of $\mathbfit{w}$ should be improved.

### Threshold specification

The above thresholds can be specified as follows.

=== "Command-line interface"
    ```yaml
    analysis_targets:
      - !SlaterDeterminant #(1)!
        source: ...
        control:
          ...: ...
          linear_independence_threshold: 1e-7 #(2)!
          integrality_threshold: 1e-7 #(3)!
          eigenvalue_comparison_mode: Modulus #(4)!
    ```

    1. :fontawesome-solid-users: This is just an example analysis target. The specification of thresholds can be specified in any analysis target.
    2. :fontawesome-solid-users: This specifies a floating-point value for the linear independence threshold $\lambda^{\mathrm{thresh}}_{\mathbfit{S}}$.
    3. :fontawesome-solid-users: This specifies a floating-point value for the interality threshold $\lambda^{\mathrm{thresh}}_{\mathrm{int}}$.
    4. :fontawesome-solid-users: This specifies the threshold comparison mode for the eigenvalues of the orbit overlap matrix $\mathbfit{S}$. The possible options are:
        - `Real`: this specifies the *real* comparison mode where the real parts of the eigenvalues are compared against the threshold,
        - `Modulus`: this specifies the *modulus* comparison mode where the absolute values of the eigenvalues are compared against the threshold.

=== "Python"
    ```python
    from qsym2 import (
        rep_analyse_slater_determinant,
        EigenvalueComparisonMode, #(1)!
    )

    rep_analyse_slater_determinant( #(2)!
        ...,
        linear_independence_threshold=1e-7, #(3)!
        integrality_threshold=1e-7, #(4)!
        eigenvalue_comparison_mode=EigenvalueComparisonMode.Modulus, #(5)!
    )
    ```

    1. :fontawesome-solid-laptop-code: This is a Python-exposed Rust enum, [`EigenvalueComparisonMode`](https://qsym2.dev/api/qsym2/analysis/enum.EigenvalueComparisonMode.html), for indicating the mode of eigenvalue comparison.
    2. :fontawesome-solid-users: This is just an example analysis driver function in Python. The specification of thresholds can be specified in any analysis driver function.
    3. :fontawesome-solid-users: This specifies a floating-point value for the linear independence threshold $\lambda^{\mathrm{thresh}}_{\mathbfit{S}}$.
    4. :fontawesome-solid-users: This specifies a floating-point value for the interality threshold $\lambda^{\mathrm{thresh}}_{\mathrm{int}}$.
    5. :fontawesome-solid-users: This specifies the threshold comparison mode for the eigenvalues of the orbit overlap matrix $\mathbfit{S}$. The possible options are:
        - `EigenvalueComparisonMode.Real`: this specifies the *real* comparison mode where the real parts of the eigenvalues are compared against the threshold,
        - `EigenvalueComparisonMode.Modulus`: this specifies the *modulus* comparison mode where the absolute values of the eigenvalues are compared against the threshold.

## Analysis options

QSym² offers multiple options for symmetry analysis in terms of group and transformation types that are available for all analysis targets (whether they are all physically meaningful for a particular target is a separate consideration).

### Magnetic groups

!!! warning "Special group notations in this section"

    In this section **only**, $\mathcal{G}$ denotes a unitary symmetry group and $\mathcal{M}$ denotes a magnetic group that admits $\mathcal{G}$ as its unitary halving subgroup.

As explained in [Symmetry-group detection/#External fields](../symmetry-group-detection.md/#external-fields), when time reversal is included in symmetry-group detection, QSym² has access to both the unitary group $\mathcal{G}$ and the magnetic group $\mathcal{M} = \mathcal{G} + \hat{a}\mathcal{G}$ of the system, if the latter is indeed present.
There are then three choices for symmetry analysis:

1. use the irreducible representations of the unitary group $\mathcal{G}$
2. use the irreducible representations of the magnetic group $\mathcal{M}$
3. use the irreducible corepresentations of the magnetic group $\mathcal{M}$

Choice 1 needs no further explanation.
Choice 3 makes use of [Wigner's corepresentation theory](../../methodologies/magnetic-corepresentations.md) and essentially considers how the joint orbit $\mathcal{M} \cdot \mathbfit{w} = \mathcal{G} \cdot \mathbfit{w} + \hat{a}\mathcal{G} \cdot \mathbfit{w}$ transforms as the irreducible corepresentations of $\mathcal{M}$ induced by the irreducible representations of the unitary halving subgroup $\mathcal{G}$.
Choice 3 is technically the proper way to handle antiunitary symmetry, but the information it gives can be rather limited since this choice honours the fact that characters of antiunitary operations do not remain invariant under a change of basis, and thus avoids explicitly characterising the symmetry of $\symbfit{w}$ under the antiunitary operations in $\mathcal{M}$.

However, there are times when it is desirable to characterise the symmetry of $\symbfit{w}$ under the antiunitary operations in $\mathcal{M}$, despite the above caveat of non-invariant antiunitary characters.
Choice 2 thus offers the possibility of treating the antiunitary elements in $\mathcal{M}$ *as though they were unitary* so that conventional representation theory can be used.
This is equivalent to considering a *unitary* group $\mathcal{M}'$ isomorphic to $\mathcal{M}$ that also contains $\mathcal{G}$ as its halving subgroup and characterising the space $W$ spanned by the orbit $\mathcal{M} \cdot \mathbfit{w}$ using the irreducible representations of $\mathcal{M}'$.
How meaningful this is depends on the nature of $\mathbfit{w}$:

- if $\mathbfit{w}$ is real-valued or can be made real-valued, then the orbits $\mathcal{M} \cdot \mathbfit{w}$ and $\mathcal{M}' \cdot \mathbfit{w}$ are identical (*i.e.* the antiunitary elements in $\mathcal{M}$ act on $\mathbfit{w}$ linearly), and the irreducible representations of $\mathcal{M}'$ are perfectly suitable for the symmetry characterisation of $\mathbfit{w}$;
- but if $\mathbfit{w}$ is complex-valued and cannot be made real, then the orbits $\mathcal{M} \cdot \mathbfit{w}$ and $\mathcal{M}' \cdot \mathbfit{w}$ differ, and although $W$ might still be decomposable in terms of the irreducible representations of $\mathcal{M}'$, it is in general unclear how the decomposition should be interpreted.

The above choices can be specified as follows.

=== "Command-line interface"
    ```yaml
    analysis_targets:
      - !SlaterDeterminant #(1)!
        source: ...
        control:
          ...: ...
          use_magnetic_group: null #(2)!
    ```

    1. :fontawesome-solid-users: This is just an example analysis target. The choices for magnetic group analysis can be specified in any analysis target.
    2. :fontawesome-solid-users: The possible options are:
        - `null`: this specifies choice 1 &mdash; use the irreducible representations of the unitary group $\mathcal{G}$,
        - `Representation`: this specifies choice 2 &mdash; use the irreducible representations of the magnetic group $\mathcal{M}$, if $\mathcal{M}$ is available,
        - `Corepresentation`: this specifies choice 3 &mdash; use the irreducible corepresentations of the magnetic group $\mathcal{M}$, if $\mathcal{M}$ is available.

=== "Python"
    ```python
    from qsym2 import (
        rep_analyse_slater_determinant,
        MagneticSymmetryAnalysisKind, #(1)!
    )

    rep_analyse_slater_determinant( #(2)!
        ...,
        use_magnetic_group=None, #(3)!
    )
    ```

    1. :fontawesome-solid-laptop-code: This is a Python-exposed Rust enum, [`MagneticSymmetryAnalysisKind`](https://qsym2.dev/api/qsym2/drivers/representation_analysis/enum.MagneticSymmetryAnalysisKind.html), for indicating the type of magnetic symmetry to be used for symmetry analysis.
    2. :fontawesome-solid-users: This is just an example analysis driver function in Python. The choices for magnetic group analysis can be specified in any analysis driver function.
    3. :fontawesome-solid-users: The possible options are:
        - `None`: this specifies choice 1 &mdash; use the irreducible representations of the unitary group $\mathcal{G}$,
        - `MagneticSymmetryAnalysisKind.Representation`: this specifies choice 2 &mdash; use the irreducible representations of the magnetic group $\mathcal{M}$, if $\mathcal{M}$ is available,
        - `MagneticSymmetryAnalysisKind.Corepresentation`: this specifies choice 3 &mdash; use the irreducible corepresentations of the magnetic group $\mathcal{M}$, if $\mathcal{M}$ is available.

### Double groups

!!! note

    From this section onwards, $\mathcal{G}$ reverts back to denoting a generic symmetry group.

QSym² is able to perform symmetry analysis based on projective representations and corepresentations.
As explained in [Methodologies/Projective (co)representations](../../methodologies/projective-reps-coreps.md), the projective irreducible representations or corepresentations of a group $\mathcal{G}$ can be obtain as conventional irreducible representations or corepresentations of its double cover $\mathcal{G}^*$.
Whether projective representations or corepresentations are required can be specified as follows.

=== "Command-line interface"
    ```yaml
    analysis_targets:
      - !SlaterDeterminant #(1)!
        source: ...
        control:
          ...: ...
          use_double_group: false #(2)!
    ```

    1. :fontawesome-solid-users: This is just an example analysis target. The choices for projective (co)representation analysis can be specified in any analysis target.
    2. :fontawesome-solid-users: This is a boolean. The possible options are:
        - `false`: use only conventional irreducible representations or corepresentations of $\mathcal{G}$,
        - `true`: use projective irreducible representations or corepresentations of $\mathcal{G}$ obtainable via its double cover $\mathcal{G}^*$.

=== "Python"
    ```python
    from qsym2 import (
        rep_analyse_slater_determinant,
    )

    rep_analyse_slater_determinant( #(1)!
        ...,
        use_double_group=False, #(2)!
    )
    ```

    1. :fontawesome-solid-users: This is just an example analysis driver function in Python. The choices for projective (co)representation analysis can be specified in any analysis driver function.
    2. :fontawesome-solid-users: This is a boolean. The possible options are:
        - `False`: use only conventional irreducible representations or corepresentations of $\mathcal{G}$,
        - `True`: use projective irreducible representations or corepresentations of $\mathcal{G}$ obtainable via its double cover $\mathcal{G}^*$.

### Transformation kinds

In QSym², every symmetry analysis target must implement five `Transformable` traits that define their transformation behaviours.
These traits are described below.

1. [`SpatialUnitaryTransformable`](https://qsym2.dev/api/qsym2/symmetry/symmetry_transformation/trait.SpatialUnitaryTransformable.html)

    This trait determines how the target behaves under spatial unitary transformations.
    If a target $\mathbfit{w}$ has a dependence on a spatial configuration-space vector $\mathbfit{r} \in \mathbb{R}^3$ such that one can write $\mathbfit{w} \equiv \mathbfit{w}(\mathbfit{r})$, and if a unitary operation $\hat{g}$ maps $\mathbfit{r}$ to $\hat{g}\mathbfit{r}$ on $\mathbb{R}^3$, then this trait allows one to obtain $\hat{g}\mathbfit{w}(\mathbfit{r}) = \mathbfit{w}(\hat{g}^{-1} \mathbfit{r})$.

2. [`SpinUnitaryTransformable`](https://qsym2.dev/api/qsym2/symmetry/symmetry_transformation/trait.SpinUnitaryTransformable.html)

    This trait determines how the target behaves under spin unitary transformations.
    If a target $\mathbfit{w}$ has a dependence on a spin configuration-space coordinate $s \in S$ such that one can write $\mathbfit{w} \equiv \mathbfit{w}(s)$, and if a unitary operation $\hat{g}$ maps $s$ to $\hat{g}s$ on $S$, then this trait allows one to obtain $\hat{g}\mathbfit{w}(s) = \mathbfit{w}(\hat{g}^{-1} s)$.

    Note that the exact structure of $S$ is often left unspecified because it is common for only the spin-transformation behaviours of $\mathbfit{w}(s)$ to be specified, but not its explicit form in terms of the spin coordinate $s$.

3. [`ComplexConjugationTransformable`](https://qsym2.dev/api/qsym2/symmetry/symmetry_transformation/trait.ComplexConjugationTransformable.html)

    This trait determines how the target behaves under complex conjugation.
    If a target $\mathbfit{w}$ is a function of a particular configuration-space coordinate $\mathbfit{x}$ such that one can write $\mathbfit{w}: \mathbfit{x} \mapsto \mathbfit{w}(\mathbfit{x}) \in \mathbb{C}$, then this trait allows one to obtain the complex conjugate $\mathbfit{w}^*(\mathbfit{x})$.

4. [`TimeReversalTransformable`](https://qsym2.dev/api/qsym2/symmetry/symmetry_transformation/trait.TimeReversalTransformable.html)

    This trait determines how the target behaves under the antiunitary action of time reversal.
    If a target implements the [`SpinUnitaryTransformable`](https://qsym2.dev/api/qsym2/symmetry/symmetry_transformation/trait.SpinUnitaryTransformable.html) and [`ComplexConjugationTransformable`](https://qsym2.dev/api/qsym2/symmetry/symmetry_transformation/trait.ComplexConjugationTransformable.html) traits as well as the [`DefaultTimeReversalTransformable`](https://qsym2.dev/api/qsym2/symmetry/symmetry_transformation/trait.DefaultTimeReversalTransformable.html) marker trait, then it also receives a default blanket implementation of the [`TimeReversalTransformable`](https://qsym2.dev/api/qsym2/symmetry/symmetry_transformation/trait.TimeReversalTransformable.html) trait which is a spin rotation by $\pi$ about the space-fixed $y$-axis followed by a complex conjugation.

5. [`SymmetryTransformable`](https://qsym2.dev/api/qsym2/symmetry/symmetry_transformation/trait.SymmetryTransformable.html)

    This trait offers multiple ways in which a target can be acted on by a [`SymmetryOperation`](https://qsym2.dev/api/qsym2/symmetry/symmetry_element/symmetry_operation/struct.SymmetryOperation.html).
    This trait requires the [`SpatialUnitaryTransformable`](https://qsym2.dev/api/qsym2/symmetry/symmetry_transformation/trait.SpatialUnitaryTransformable.html) and [`TimeReversalTransformable`](https://qsym2.dev/api/qsym2/symmetry/symmetry_transformation/trait.TimeReversalTransformable.html) traits to have been implemented.
    The possible types of transformations defined by this trait are:

    - spatial only,
    - spin only,
    - both spin and spatial.

Given a group $\mathcal{G}$, how its [`SymmetryOperation`](https://qsym2.dev/api/qsym2/symmetry/symmetry_element/symmetry_operation/struct.SymmetryOperation.html) elements act on a target $\mathbfit{w}$ based on the [`SymmetryTransformable`](https://qsym2.dev/api/qsym2/symmetry/symmetry_transformation/trait.SymmetryTransformable.html) trait to generate the orbit $\mathcal{G} \cdot \mathbfit{w}$ for symmetry analysis can be specified as follows.

=== "Command-line interface"
    ```yaml
    analysis_targets:
      - !SlaterDeterminant #(1)!
        source: ...
        control:
          ...: ...
          symmetry_transformation_kind: Spatial #(2)!
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

### Infinite-order symmetry elements

If the group $\mathcal{G}$ contains operations generated by one or more infinite-order symmetry elements, then the orbit $\mathcal{G} \cdot \mathbfit{w}$ is of infinite cardinality, and the numerical methods in QSym² are unable to directly characterise the space $W$ spanned by this orbit.
However, QSym² provides an option to consider a suitable subgroup $\mathcal{G}_n$ of $\mathcal{G}$ in which each infinite-order generating symmetry element is restricted to having a finite order $n$.

If $\mathcal{G}$ is an infinite linear group (*e.g.* $\mathcal{D}_{\infty h}$ or $\mathcal{C}_{\infty v}$), then there is only one infinite-order generating symmetry element (the $C_{\infty}$ axis).
Any positive choice of the finite integer order $n$ is permitted and results in a finite axial group $\mathcal{G}_n$ (*e.g.* $\mathcal{D}_{nh}$ or $\mathcal{C}_{nv}$) in which the $C_{\infty}$ axis has been replaced by the $C_{n}$ axis.

However, if $\mathcal{G}$ is the full rotation group $\mathsf{SO}(3)$ or the full roto-inversion group $\mathsf{O}(3)$, then there are in principle infinitely many $C_{\infty}$ axes.
In such a case, QSym² nominally chooses three orthogonal axes that are aligned with the three space-fixed Cartesian axes, *i.e.* $C_{\infty}^x$, $C_{\infty}^y$, and $C_{\infty}^z$, as the infinite-order generating elements for $\mathcal{G}$.
Then, choosing a value for $n$ restricts the order of *all three* generating elements simultaneously, giving $C_{n}^x$, $C_{n}^y$, and $C_{n}^z$.
This means that only two values of $n$ are possible:

- $n = 2$: $\mathsf{SO}(3) > \mathcal{D}_2$ and $\mathsf{O}(3) > \mathcal{D}_{2h}$,
- $n = 4$: $\mathsf{SO}(3) > \mathcal{O}$ and $\mathsf{O}(3) > \mathcal{O}_{h}$.

This is because the orthogonality between the three axes precludes other values of $n$ from fulfilling group closure.

In any case, the finite group $\mathcal{G}_n$ is used for subsequent symmetry analysis via the finite orbit $\mathcal{G}_n \cdot \mathbfit{w}$.
With an appropriately chosen value for $n$, the irreducible (co)representations of the finite subgroup $\mathcal{G}_n$ can be used to infer the symmetry of $\mathbfit{w}$ with respect to the full infinite group $\mathcal{G}$ unequivocally.
Examples of how this is achieved for the infinite groups $\mathcal{C}_{\infty v}$ and $\mathcal{C}_{\infty}$ are detailed in [Section 3.3.1 of the QSym² paper](../../about/authorship.md#publications).

The value $n$ can be specified as follows.

=== "Command-line interface"
    ```yaml
    analysis_targets:
      - !SlaterDeterminant #(1)!
        source: ...
        control:
          ...: ...
          infinite_order_to_finite: 8 #(2)!
    ```

    1. :fontawesome-solid-users: This is just an example analysis target. The finite order $n$ can be specified in any analysis target.
    2. :fontawesome-solid-users: This specifies the finite order $n$. The possible options are:
        - `null`: do not restrict infinite-order symmetry elements to finite order,
        - a positive integer value: restrict all infinite-order symmetry elements to this finite order (this will be ignored if the group has no infinite-order symmetry elements).

=== "Python"
    ```python
    from qsym2 import (
        rep_analyse_slater_determinant,
    )

    rep_analyse_slater_determinant( #(1)!
        ...,
        infinite_order_to_finite=8, #(2)!
    )
    ```

    1. :fontawesome-solid-users: This is just an example analysis driver function in Python. The finite order $n$ can be specified in any analysis driver function.
    2. :fontawesome-solid-users: This specifies the finite order $n$. The possible options are:
        - `None`: do not restrict infinite-order symmetry elements to finite order,
        - a positive integer value: restrict all infinite-order symmetry elements to this finite order (this will be ignored if the group has no infinite-order symmetry elements).
