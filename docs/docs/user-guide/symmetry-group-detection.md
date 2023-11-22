---
title: Symmetry-group detection
description: Configurable parameters for symmetry-group detection
---

# Symmetry-group detection

## Overview

### Thresholds

QSym² uses an enhanced version of the [Beruski&ndash;Vidal algorithm](http://doi.wiley.com/10.1002/jcc.23493) to locate symmetry elements in molecular systems, possibly in the presence of external fields.
In the working of the algorithm, several types of numerical comparisons need to be made:

- comparisons between principal moments of inertia to classify the rotational symmetry of the system,
- comparisons of atomic coordinates to determine if a point transformation leaves the system invariant, and
- comparisons of normalised vector components to determine if a symmetry element has already been found.

Since all of the quantities being compared are over the field of real numbers represented computationally as 64-bit floating-point numbers, thresholds are required to account for numerical inaccuracies or uncertainties so that these numerical comparisons can be made stable and meaningful.
QSym² thus defines **two** types of thresholds for these comparisons:

- one for the comparisons between moments of inertia, and
- one for the comparisons of atomic coordinates and normalised vector components.

QSym² also allows for multiple thresholds to be specified for each type, so that the symmetry group detection routine can be repeated at various threshold combinations, and the highest symmetry group at the tighest threshold combination will eventually be selected for further analysis.

### External fields

QSym² is capable of determining symmetry groups in the presence of external electric and/or magnetic fields using a method of fictitious atoms.
A detailed description of how this is achieved can be found in the [QSym² paper](../about/authorship.md#publications).
An added benefit of this method is that QSym² is able to handle the symmetry of non-uniform fields mappable to collections of magnetic and electric dipoles that can be modelled by fictitious atoms.
QSym² thus allows the specification of arbitrary lists of fictitious magnetic and electric atoms that can be added to the molecular system being analysed.

With the inclusion of magnetic fields and hence the prospect of antiunitary symmetry operations, QSym² allows users to specify whether they wish to consider time reversal in symmetry analysis.

- If time reversal is omitted, the symmetry group obtained is the unitary group $\mathcal{G}$ comprising only unitary point symmetry operations of the system.
- If time reversal is instead included, the symmetry group obtained is the magnetic group $\mathcal{M}$ that takes $\mathcal{G}$ as its unitary halving subgroup: $\mathcal{M} = \mathcal{G} + \hat{a}\mathcal{G}$ where $\hat{a}$ is an antiunitary symmetry operation of the system.
    - $\mathcal{M}$ is called a *magnetic grey group* if it contains the time reversal operation $\hat{\theta}$. In this case, it is conventional and convenient to choose $\hat{a} = \hat{\theta}$. 
    - $\mathcal{M}$ is called a *magnetic black-and-white group* if it does **not** contain the time reversal operation.

### Result serialisation

QSym² supports the serialisation of symmetry group detection results as binary files.
This enables the results of symmetry group detection from one QSym² calculation to be read in by another QSym² calculation for further analysis.


## Parameters

The input parameters and their descriptions for each mode of running symmetry-group detection in QSym² are given below.
When an input parameter has a default value, the default value will be specified.

=== "Binary"
    === "Parameter specification"
        ```yaml
        symmetry_group_detection: !Parameters #(1)!
          moi_thresholds: #(2)!
          - 1e-2
          - 1e-4
          - 1e-6
          distance_thresholds: #(3)!
          - 1e-2
          - 1e-4
          - 1e-6
          time_reversal: false #(4)!
          fictitious_magnetic_fields: #(5)!
          - [[0, 0, 0], [0, 0, 1]]
          fictitious_electric_fields: null #(6)!
          field_origin_com: true #(7)!
          write_symmetry_elements: true #(8)!
          result_save_name: null #(9)!

        analysis_targets:
        - !MolecularSymmetry #(10)!
          xyz: /path/to/xyz/file #(11)!
        ```

        1. :fontawesome-solid-laptop-code: Under the hood, the `!Parameters` syntax indicates that the variant `Parameters` of the [`SymmetryGroupDetectionInputKind`](https://qsym2.dev/api/qsym2/interfaces/input/enum.SymmetryGroupDetectionInputKind.html#variant.Parameters) enum is being specified, which wraps around the [`SymmetryGroupDetectionParams`](https://qsym2.dev/api/qsym2/drivers/symmetry_group_detection/struct.SymmetryGroupDetectionParams.html) struct.
        2. :fontawesome-solid-users: Each element in this list is a threshold for moment-of-inertia comparisons. All pairs of thresholds, one from `moi_thresholds` and one from `distance_thresholds`, will be considered.</br></br>:material-cog-sync-outline: Default: `[1e-4, 1e-5, 1e-6]`.
        3. :fontawesome-solid-users: Each element in this list is a threshold for distance and geometry comparisons. All pairs of thresholds, one from `moi_thresholds` and one from `distance_thresholds`, will be considered.</br></br>:material-cog-sync-outline: Default: `[1e-4, 1e-5, 1e-6]`.
        4. :fontawesome-solid-users: This boolean indicates if time reversal is to be taken into account.</br></br>:material-cog-sync-outline: Default: `false`.
        5. :fontawesome-solid-users: Each element in this optional list is a tuple consisting of an origin $\mathbf{O}$ and a vector $\mathbf{v}$, for which a `magnetic(+)` special atom will be added at $\mathbf{O} + \mathbf{v}$, and a `magnetic(-)` special atom will be added at $\mathbf{O} - \mathbf{v}$. The fields described by these fictitious special atoms are not present in the system but added here only for symmetry analysis.</br></br>:material-cog-sync-outline: Default: `null`.
        6. :fontawesome-solid-users: Each element in this optional list is a tuple consisting of an origin $\mathbf{O}$ and a vector $\mathbf{v}$, for which an `electric(+)` special atom will be added at $\mathbf{O} + \mathbf{v}$. The fields described by these fictitious special atoms are not present in the system but added here only for symmetry analysis.</br></br>:material-cog-sync-outline: Default: `null`.
        7. :fontawesome-solid-users: This boolean indicates if the origins specified in `fictitious_magnetic_fields` and `fictitious_electric_fields` are to be taken relative to the molecule's centre of mass rather than to the space-fixed origin.</br></br>:material-cog-sync-outline: Default: `false`.
        8. :fontawesome-solid-users: This boolean indicates if a summary of the located symmetry elements is to be written to the output file.</br></br>:material-cog-sync-outline: Default: `false`.
        9. :fontawesome-solid-users: This optional string specifies a name for saving the result as a binary file of extension `.qsym2.sym`. If none is given, the result will not be saved.</br></br>:material-cog-sync-outline: Default: `null`.
        10. :fontawesome-solid-laptop-code: Under the hood, the specification of the target for symmetry-group detection in a YAML configuration file is handled by the `MolecularSymmetry` variant of the [`AnalysisTarget`](https://qsym2.dev/api/qsym2/interfaces/input/analysis/enum.AnalysisTarget.html) enum.
        11. :fontawesome-solid-users: This specifies a path to an XYZ file containing the input molecular structure for symmetry-group detection.

    === "Read from file"
        ```yaml
        symmetry_group_detection: !FromFile #(1)!
          /path/to/symmetry/result/file #(2)!
        ```

        1. :fontawesome-solid-laptop-code: Under the hood, the `!FromFile` syntax indicates that the variant `FromFile` of the [`SymmetryGroupDetectionInputKind`](https://qsym2.dev/api/qsym2/interfaces/input/enum.SymmetryGroupDetectionInputKind.html#variant.Parameters) enum is being specified.
        2. :fontawesome-solid-users: This path points to a `.qsym2.sym` file generated by QSym² from an earlier symmetry-group detection calculation. However, this path should be specified without the `.qsym2.sym` extension.</br>For example, if QSym² has generated the file `~/calc/water/h2o.qsym2.sym`, then this path should be `~/calc/water/h2o`.

=== "Python"
    ```py
    from qsym2 import detect_symmetry_group, PyMolecule

    pymol = PyMolecule( #(1)!
        atoms=[ #(2)!
            ("O", [0.0000000, -0.0184041,  0.0000000]),
            ("H", [0.0000000,  0.5383520, -0.7830361]),
            ("H", [0.0000000,  0.5383520,  0.7830361]),
        ],
        threshold=1e-7, #(3)!
        magnetic_field=[0.0, 0.1, 0.0], #(4)!
        electric_field=None, #(5)!
    )

    unisym, magsym = detect_symmetry_group( #(6)!
        inp_xyz=None, #(7)!
        inp_mol=pymol, #(8)!
        out_sym="mol", #(9)!
        moi_thresholds=[1e-2, 1e-4, 1e-6], #(10)!
        distance_thresholds=[1e-2, 1e-4, 1e-6], #(11)!
        time_reversal=False, #(12)!
        fictitious_magnetic_field=[0, 0, 1], #(13)!
        fictitious_electric_field=None, #(14)!
        write_symmetry_elements=True, #(15)!
    ) #(16)!
    ```

    1. :fontawesome-solid-laptop-code: :fontawesome-solid-users: This specifies the molecular system for symmetry-group detection. The [`PyMolecule`](https://qsym2.dev/api/qsym2/bindings/python/symmetry_group_detection/struct.PyMolecule.html) class, constructible in Python, contains geometry information that can be interpreted by QSym² on the Rust side. The [API documentation](https://qsym2.dev/api/qsym2/bindings/python/symmetry_group_detection/struct.PyMolecule.html) for this class can be consulted for further information.
    2. :fontawesome-solid-users: The coordinates of the atoms specified in this list can be in any units. QSym² does not care what the actual units are &mdash; symmetry properties are invariant to any change in length scale. The only exception is when QSym² evaluates molecular integrals: here, atomic units will be assumed.
    3. :fontawesome-solid-users: This specifies the threshold for comparing molecules. Note that this threshold is not the same as those specified in `detect_symmetry_group` below.
    4. :fontawesome-solid-users: This list gives the components of an optional uniform external magnetic field that is present in the system.</br></br> :material-cog-sync-outline: Default: `None`.
    5. :fontawesome-solid-users: This list gives the components of an optional uniform external electric field that is present in the system.</br></br> :material-cog-sync-outline: Default: `None`.
    6. :fontawesome-solid-users: The [`detect_symmetry_group`](https://qsym2.dev/api/qsym2/bindings/python/symmetry_group_detection/fn.detect_symmetry_group.html) function performs symmetry-group detection and logs the result via the `qsym2-output` logger at the `INFO` level. The [API documentation](https://qsym2.dev/api/qsym2/bindings/python/symmetry_group_detection/fn.detect_symmetry_group.html) for this function can be consulted for further information.
    7. :fontawesome-solid-users: This specifies a path to an XYZ file containing the geometry of the molecule whose symmetry group is to be determined. One and only one of `inp_xyz` or `inp_mol` must be specified, and the other must be `None`.
    8. :fontawesome-solid-users: This specifies a [`PyMolecule`](https://qsym2.dev/api/qsym2/bindings/python/symmetry_group_detection/struct.PyMolecule.html) object containing the molecular system whose symmetry group is to be determined. One and only one of `inp_xyz` or `inp_mol` must be specified, and the other must be `None`.
    9. :fontawesome-solid-users: This specifies an optional name for the `.qsym2.sym` file to be saved that contains the serialised results of the symmetry-group detection. This name does not need to contain the `.qsym2.sym` extension.
    10. :fontawesome-solid-users: Each element in this list is a threshold for moment-of-inertia comparisons. All pairs of thresholds, one from `moi_thresholds` and one from `distance_thresholds`, will be considered.
    11. :fontawesome-solid-users: Each element in this list is a threshold for distance and geometry comparisons. All pairs of thresholds, one from `moi_thresholds` and one from `distance_thresholds`, will be considered.
    12. :fontawesome-solid-users: This boolean indicates if time reversal is to be taken into account.
    13. :fontawesome-solid-users: This list gives the components of an optional fictitious uniform external magnetic field. This field is not present in the system but is added here only for symmetry analysis.</br></br> :material-cog-sync-outline: Default: `None`.
    14. :fontawesome-solid-users: This list gives the components of an optional fictitious uniform external electric field. This field is not present in the system but is added here only for symmetry analysis.</br></br> :material-cog-sync-outline: Default: `None`.
    15. :fontawesome-solid-users: This boolean indicates if a summary of the located symmetry elements is to be written to the output file. </br></br> :material-cog-sync-outline: Default: `True`.
    16. :fontawesome-solid-laptop-code: :fontawesome-solid-users: The [`detect_symmetry_group`](https://qsym2.dev/api/qsym2/bindings/python/symmetry_group_detection/fn.detect_symmetry_group.html) function returns a tuple of two objects:
        - `unisym`: a [`PySymmetry`](https://qsym2.dev/api/qsym2/bindings/python/symmetry_group_detection/struct.PySymmetry.html) object containing the unitary symmetry elements, and
        - `magsym`: an optional [`PySymmetry`](https://qsym2.dev/api/qsym2/bindings/python/symmetry_group_detection/struct.PySymmetry.html) object containing the magnetic symmetry elements, if requested by the `time_reversal` parameter.
