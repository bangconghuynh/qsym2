---
title: Basics
description: Basic information for representation analysis
---

# Basics

Let $V$ be a linear space and $\mathbfit{w} \in V$ be an element whose symmetry with respect to a group $\mathcal{G}$ is to be determined by QSym².
This amounts to identifying and characterising the linear subspace $W \subseteq V$ spanned by the orbit

$$
    \mathcal{G} \cdot \mathbfit{w} = \{ \hat{g}_i \mathbfit{w} \ :\ g_i \in \mathcal{G} \}.
$$

The mathematical details of this method are described in [Section 2.4 of the QSym² paper](../../about/authorship.md#publications).
Here, it suffices to summarise the key ingredients that are required for this method to work.


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

which is then used to determine the representation matrices of the elements of $\mathcal{G}$ on $W$, as described in the [QSym² paper](../../about/authorship.md#publications).

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

=== "Binary"
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


## Linear independence threshold

As explained in [Section 3.2.1 of the QSym² paper](../../about/authorship.md#publications), for every quantity $\mathbfit{w}$ that is to be symmetry-analysed via the orbit $\mathcal{G} \cdot \mathbfit{w}$, the overlap matrix $\mathbfit{S}$ between the elements in $\mathcal{G} \cdot \mathbfit{w}$ needs to be computed, and a threshold $\lambda^{\mathrm{thresh}}_{\mathbfit{S}}$ needs to be chosen to determine which of the eigenvalues of $\mathbfit{S}$ are non-zero.
This is so that linearly dependent elements in $\mathcal{G} \cdot \mathbfit{w}$ can be projected out and the space $W$ can be correctly identified.
The choice of the *linear independence threshold* $\lambda^{\mathrm{thresh}}_{\mathbfit{S}}$ is therefore vital for a numerically stable and meaningful symmetry analysis from QSym².

How does one pick a sensible value for this threshold?
The answer to this question depends on the eigenspectrum of $\mathbfit{S}$, and unfortunately, there is no *a priori* way to determine the best possible value of $\lambda^{\mathrm{thresh}}_{\mathbfit{S}}$ without examining this eigenspectrum first.
Fortunately, in all cases, QSym² can help with this by printing out the eigenvalue of $\mathbfit{S}$ immediately above the chosen threshold, $\lambda^{>}_{\mathbfit{S}}$, and also the eigenvalue of $\mathbfit{S}$ immediately below the chosen threshold, $\lambda^{<}_{\mathbfit{S}}$.

The author recommends that a good threshold choice is one for which $\log_{10}\lambda^{>}_{\mathbfit{S}} - \log_{10}\lambda^{<}_{\mathbfit{S}} \ge 3$, *i.e.* the threshold cuts through a gap of at least three orders of magnitude in the eigenspectrum of $\mathbfit{S}$.
If such a gap does not exist, then $\mathcal{G} \cdot \mathbfit{w}$ contains all linearly independent elements.
In this case:

- if $\mathbfit{w}$ has been tightly converged and is of a high numerical quality (relative to $\lambda^{\mathrm{thresh}}_{\mathbfit{S}}$), then it can be concluded that $\mathbfit{w}$ transforms as the regular representation of $\mathcal{G}$;
- however, if $\mathbfit{w}$ has been rather poorly converged and is of a low numerical quality (relative to $\lambda^{\mathrm{thresh}}_{\mathbfit{S}}$), then any observed symmetry breaking could be artificial.
