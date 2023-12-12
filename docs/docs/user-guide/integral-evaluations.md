---
title: Integral evaluations
description: Configurable parameters for integral evaluations
---

# Integral evaluations

## Overview

QSym² implements a generic [recursive algorithm by Honda *et al.*](https://doi.org/10.1063/1.459751) that is capable of calculating any $n$-centre overlap integrals and their derivatives over both Gaussian and London atomic-orbital basis functions.
These allow QSym² to perform representation analysis on quantities that require non-standard overlap matrices such as electron densities (see [Section 2.4.2 of the QSym² paper](../about/authorship.md#publications)) or current densities (to be detailed in a future publication).

The generality of this implementation is handled by Rust's [declarative macros](https://veykril.github.io/tlborm/introduction.html) that allow recursive codes for different integral patterns (*e.g.* two-centre or three-centre) to be written generically at the [Abstract Syntax Tree (AST)](https://en.wikipedia.org/wiki/Abstract_syntax_tree) level.
The generic codes are then expanded by the compiler into valid specific codes *before the main compilation of the program*.
The expanded specific codes thus enjoy the same level of compiler optimisation as though they had been written manually.


## Basis set information

QSym² requires a complete specification of the basis set to evaluate integrals.
At the moment, this functionality is only exposed to Python, but other ways to evaluate integrals (*e.g.* via the command-line interface saving the integrals as binary files) might be added in the future, if there are definitive use cases.
The basis set specification in the Python API is shown below by way of an example for hydrogen cyanide in 3-21G.

=== "Python"
    ```py
    from qsym2 import PyBasisShellContraction

    # HCN, 3-21G
    basis_set = [ #(1)!
        # H
        [ #(2)!
            PyBasisShellContraction( #(3)!
                basis_shell=("S", True, None), #(4)!
                primitives=[ #(5)!
                    (0.5447178000e+01, 0.1562849787e+00),
                    (0.8245472400e+00, 0.9046908767e+00),
                ],
                cart_origin=[0.0, 0.0, 0.0], #(6)!
                k=None, #(7)!
            ),
            PyBasisShellContraction(
                basis_shell=("S", True, None),
                primitives=[
                    (0.1831915800e+00, 1.0000000000e+00),
                ],
                cart_origin=[0.0, 0.0, 0.0],
                k=None,
            ),
        ],
        # C
        [
            PyBasisShellContraction(
                basis_shell=("S", True, None),
                primitives=[
                    (0.1722560000e+03, 0.6176690738e-01),
                    (0.2591090000e+02, 0.3587940429e+00),
                    (0.5533350000e+01, 0.7007130837e+00),
                ],
                cart_origin=[0.0, 0.0, 1.0],
                k=None,
            ),
            PyBasisShellContraction(
                basis_shell=("S", True, None),
                primitives=[
                    (0.3664980000e+01, -0.3958951621e+00),
                    (0.7705450000e+00,  0.1215834356e+01),
                ],
                cart_origin=[0.0, 0.0, 1.0],
                k=None,
            ),
            PyBasisShellContraction(
                basis_shell=("P", True, None), #(8)!
                primitives=[
                    (0.3664980000e+01, 0.2364599466e+00),
                    (0.7705450000e+00, 0.8606188057e+00),
                ],
                cart_origin=[0.0, 0.0, 1.0],
                k=None,
            ),
            PyBasisShellContraction(
                basis_shell=("S", True, None),
                primitives=[
                    (0.1958570000e+00, 1.0000000000e+00),
                ],
                cart_origin=[0.0, 0.0, 1.0],
                k=None,
            ),
            PyBasisShellContraction(
                basis_shell=("P", True, [(0, 0, 1), (0, 1, 0), (1, 0, 0)]), #(9)!
                primitives=[
                    (0.1958570000e+00, 1.0000000000e+00),
                ],
                cart_origin=[0.0, 0.0, 1.0],
                k=None,
            ),
        ],
        # N
        [
            PyBasisShellContraction(
                basis_shell=("S", True, None),
                primitives=[
                    (0.2427660000e+03, 0.5986570051e-01),
                    (0.3648510000e+02, 0.3529550030e+00),
                    (0.7814490000e+01, 0.7065130060e+00),
                ],
                cart_origin=[0.0, 0.0, 1.5],
                k=None,
            ),
            PyBasisShellContraction(
                basis_shell=("S", True, None),
                primitives=[
                    (0.5425220000e+01, -0.4133000774e+00),
                    (0.1149150000e+01,  0.1224417267e+01),
                ],
                cart_origin=[0.0, 0.0, 1.5],
                k=None,
            ),
            PyBasisShellContraction(
                basis_shell=("P", False, True), #(10)!
                primitives=[
                    (0.5425220000e+01, 0.2379720162e+00),
                    (0.1149150000e+01, 0.8589530586e+00),
                ],
                cart_origin=[0.0, 0.0, 1.5],
                k=None,
            ),
            PyBasisShellContraction(
                basis_shell=("S", True, None),
                primitives=[
                    (0.2832050000e+00, 1.0000000000e+00),
                ],
                cart_origin=[0.0, 0.0, 1.5],
                k=None,
            ),
            PyBasisShellContraction(
                basis_shell=("P", False, [0, +1, -1]), #(11)!
                primitives=[
                    (0.2832050000e+00, 1.0000000000e+00),
                ],
                cart_origin=[0.0, 0.0, 1.5],
                k=None,
            ),
        ],
    ]
    ```

    1. :fontawesome-solid-users: The entire basis set is represented by a Python list.
    Each item in this list is a list containing all shells centred on a single atom of the molecule.
    2. :fontawesome-solid-users: In this example, this inner list contains all shells centred on the hydrogen atom in the molecule.
    3. :fontawesome-solid-users: Each item in this inner list specifies the contraction information for one shell.
    </br></br>:fontawesome-solid-laptop-code: This class, [`PyBasisShellContraction`](https://qsym2.dev/api/qsym2/bindings/python/integrals/struct.PyBasisShellContraction.html), is a Python-exposed Rust structure to marshall basis shell contraction information between Rust and Python. Refer to the API documentation for this class for more information.
    4. :fontawesome-solid-users: The `basis_shell` parameter is a tuple specifying the angular momentum information of this shell and has the form `(angmom, cart, order)` where:
        - `angmom` is a symbol such as `"S"` or `"P"` for the angular momentum of the shell,
        - `cart` is a boolean indicating if the functions in the shell are Cartesian (`True`)
        or pure / solid harmonics (`False`), and
        - `order` specifies how the functions in the shell are ordered:
            - if `cart` is `True`, `order` can be `None` for lexicographic order, or a list of
            tuples `(n_x, n_y, n_z)` specifying a custom order for the Cartesian functions where
            `n_x`, `n_y`, and `n_z` are the $x$-, $y$-, and $z$-exponents for a Cartesian component $x^{n_x} y^{n_y} z^{n_z}$;
            - if `cart` is `False`, `order` can be `True` for increasing-$m_l$ order, `False` for
            decreasing-$m_l$ order, or a list of $m_l$ values for custom order.
        </li>:fontawesome-solid-laptop-code: Under the hood, this is handled by [`PyShellOrder`](https://qsym2.dev/api/qsym2/bindings/python/integrals/enum.PyShellOrder.html) which is a Python-exposed enumerated type to manage shell order information.
    5. :fontawesome-solid-users: The `primitives` parameter is a list specifying the Gaussian primitives making up this shell.
    Each item in this list is a tuple `(exponent, coefficient)` specifying the exponent and coefficient of a Gaussian primitive in the contraction.
    6. :fontawesome-solid-users: This specifies the Cartesian origin $\mathbfit{R}$ of the shell *in Bohr radii*.
    7. :fontawesome-solid-users: This optional fixed-size list of length 3 specifies the Cartesian components of the $\mathbfit{k}$ vector of this shell that appears in the complex phase factor $\exp[i\mathbfit{k} \cdot (\mathbfit{r} - \mathbfit{R})]$.
    8. :fontawesome-solid-users: This example specifies a Cartesian $P$-shell in which functions are arranged in lexicographic order.
    9. :fontawesome-solid-users: This example specifies a Cartesian $P$-shell in which functions are arranged in a custom order: $z, y, x$.
    10. :fontawesome-solid-users: This example specifies a spherical $P$-shell in which functions are arranged in increasing-$m_l$ order.
    11. :fontawesome-solid-users: This example specifies a spherical $P$-shell in which functions are arranged in a custom $m_l$ order: $0, +1, -1$.

## Integral calculations

At the moment, only two- and four-centre overlap integral calculations have been exposed to the Python API.
Other integral patterns can be exposed in the future should the need arise.
Obtaining these integrals via the Python API is very straightforward.


=== "Python"
    ```py
    from qsym2 import (
        calc_overlap_2c_real, #(1)!
        calc_overlap_2c_complex,
        calc_overlap_4c_real,
        calc_overlap_4c_complex,
    )

    basis_set = ... #(2)!

    sao_2c_r = calc_overlap_2c_real(basis_set) #(3)!
    sao_2c_c = calc_overlap_2c_complex(basis_set) #(4)!

    sao_4c_r = calc_overlap_4c_real(basis_set) #(5)!
    sao_4c_c = calc_overlap_4c_complex(basis_set) #(6)!
    ```

    1. :fontawesome-solid-laptop-code: These are Python-exposed functions to evaluate overlap integrals. The API documentation for these functions can be found [here](https://qsym2.dev/api/qsym2/bindings/python/integrals/index.html).
    2. :fontawesome-solid-users: The basis set information must be constructed as described [above](#basis-set-information).
    3. :fontawesome-solid-users: This evaluates the real-valued two-centre overlap matrix for the specified basis set and returns the results as a two-dimensional array. This is only applicable if the basis set comprises only Gaussian atomic orbitals (*i.e.* no $\mathbfit{k}$ vectors anywhere) with real contraction coefficients.
    4. :fontawesome-solid-users: This evaluates the complex-valued two-centre overlap matrix for the specified basis set and returns the results as a two-dimensional array.
    5. :fontawesome-solid-users: This evaluates the real-valued four-centre overlap tensor for the specified basis set and returns the results as a four-dimensional array. This is only applicable if the basis set comprises only Gaussian atomic orbitals (*i.e.* no $\mathbfit{k}$ vectors anywhere) with real contraction coefficients.
    6. :fontawesome-solid-users: This evaluates the complex-valued four-centre overlap matrix for the specified basis set and returns the results as a four-dimensional array.
