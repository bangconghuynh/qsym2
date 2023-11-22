---
title: Basic requirements
description: Basic requirements of representation analysis
---

# Basic requirements

Let $V$ be a linear space and $\mathbfit{w} \in V$ be an element whose symmetry with respect to a group $\mathcal{G}$ is to be determined by QSym².
This amounts to identifying and characterising the linear subspace $W \subseteq V$ spanned by the orbit

$$
    \mathcal{G} \cdot \mathbfit{w} = \{ \hat{g}_i \mathbfit{w} \ :\ g_i \in \mathcal{G} \}.
$$

The mathematical details of this method are described in the [QSym² paper](../../about/authorship.md#publications).
Here, it suffices to summarise the key ingredients that are required for this method to work.

## Basis overlap matrix

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

Depending on the nature of the basis functions in $\mathcal{B}_V$, the overlap matrix $\mathbfit{S}_V$ might have already been computed by other quantum-chemistry calculatgion programs, in which case it can simply be read in by QSym².
But if $\mathbfit{S}_V$ is not readily available, then information about $\mathcal{B}_V$ must be made available to QSym² so that QSym² can compute $\mathbfit{S}_V$.
The exact requirements for the various types of representation analysis that QSym² supports will be explained in the relevant sections in this guide.

## Atomic-orbital basis angular order

If $\mathcal{B}_V$ is a basis consisting of atomic orbitals, then, to carry out the transformations $\hat{g}_i \mathbfit{w}$ that are required to construct the orbit $\mathcal{G} \cdot \mathbfit{w}$ for representation analysis, QSym² needs to know how the basis atomic orbitals are transformed under spatial rotations.
This is equivalent to knowing the following for any shell of atomic orbitals:

- the atom on which the shell is centred,
- the angular momentum degree of the shell,
- whether the angular parts of the atomic orbitals in the shell (which are real solid harmonic functions) are expressed in Cartesian coordinates or in spherical polar coordinates, and
- the ordering of the functions in the shell.

These pieces of information are collectively referred to as the *basis angular order* information in QSym², which must be specified for any representation analysis performed on quantities expressed in terms of atomic orbitals.
The possible ways to specify this are shown below.

=== "Binary"
    === "Parameter specification"
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
                  - [1, 1, 0]
                  - [1, 0, 1]
                  - [0, 2, 0]
                  - [0, 1, 1]
                  - [0, 0, 2]
              - atom: [1, "H"]
                basis_shells:
                - l: 1
                  shell_order: !CartLexicographic
                - l: 3
                  shell_order: !CartQChem
                - l: 2
                  shell_order: !CartCustom
                  - [2, 0, 0]
                  - [1, 1, 0]
                  - [1, 0, 1]
                  - [0, 2, 0]
                  - [0, 1, 1]
                  - [0, 0, 2]
        ```

        1. :fontawesome-solid-users: This is an example data source (Slater determinant specified via binary coefficient files) where a manual specification of basis angular order is required. If other data sources for other analysis targets also require a manual specification of basis angular order, the format will be the same.
        2. :fontawesome-solid-users: Each item in this list specifies the angular order information for all shells on one atom in the molecule.</br></br>:fontawesome-solid-laptop-code: Under the hood, this key wraps around the [`InputBasisAngularOrder`](https://qsym2.dev/api/qsym2/interfaces/input/ao_basis/struct.InputBasisAngularOrder.html) struct which consists of a vector of [`InputBasisAtom`](https://qsym2.dev/api/qsym2/interfaces/input/ao_basis/struct.InputBasisAtom.html) structs.
        3. :fontawesome-solid-users: This key, `atom`, specifies the index and name of an atom in the basis set.
        4. :fontawesome-solid-users: This key, `basis_shells`, gives the ordered shells associated with this atom. Each item in this list specifies the angular momentum information of one shell centred on the prevailing atom.</br></br>:fontawesome-solid-laptop-code: Under the hood, this key is a vector of [`InputBasisShell`](https://qsym2.dev/api/qsym2/interfaces/input/ao_basis/struct.InputBasisShell.html) structs.
        5. :fontawesome-solid-users: This key, `l`, specifies the angular momentum degree of this shell.
        6. :fontawesome-solid-users: This key, `shell_order`, specifies the type and ordering of the basis functions in this shell. The following variants are supported:
            - `!PureIncreasingm`: the basis functions are pure real solid harmonics, arranged in increasing $m_l$ order,
            - `!PureDecreasingm`: the basis functions are pure real solid harmonics, arranged in decreasing $m_l$ order,
            - `!PureCustom`: the basis functions are pure real solid harmonics, arranged in a custom order to be specified by the $m_l$ values,
            - `!CartLexicographic`: the basis functions are Cartesian real solid harmonics, arranged in lexicographic order,
            - `!CartQChem`: the basis functions are Cartesian real solid harmonics, arranged in Q-Chem order,
            - `!CartCustom`: the basis functions are pure real solid harmonics, arranged in a custom order to be specified by the ordered exponent tuples.
        7. :fontawesome-solid-users: The order of the elements in this list specifies the $m_l$ order of the functions in this shell. Invalid $m_l$ values for a specified $l$ value (*i.e.* $\lvert m_l \rvert > l$) will result in an error. Invalid number of elements (*i.e.* not $2l + 1$) will also result in an error.
        8. :fontawesome-solid-users: Each element in this list is a tuple `[n_x, n_y, n_z]` containing the exponents of one Cartesian component: $x^{n_x} y^{n_y} z^{n_z}$. The order of the elements in this list specifies the order of the Cartesian components in this shell. Invalid exponents (*i.e.* $n_x + n_y + n_z \ne l$) will result in an error. Invalid number of elements (*i.e.* not $(l + 1)(l + 2)/2$) will also result in an error.
