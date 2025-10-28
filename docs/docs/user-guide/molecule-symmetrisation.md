---
title: Molecule symmetrisation
description: Configurable parameters for molecule symmetrisation
---

# Molecule symmetrisation

## Overview

### Symmetrisation by 'bootstrapping'

QSym² offers a 'bootstrapping' algorithm to symmetrise molecular systems.
This algorithm symmetrises a molecule iteratively by defining two threshold levels: a `loose` level and a `target` level.
In every iteration, the following steps are performed:

1. The molecule is symmetry-analysed at the `target` level; any symmetry elements found are stashed and the symmetry group name, if any, is registered.
2. The molecule is symmetry-analysed at the `loose` level; any symmetry elements found are added to the stash and the symmetry group name, if any, is registered.
3. The convergence criteria (see below) are checked.
    - If convergence has been reached, the symmetrisation procedure is terminated.
    - If convergence has not been reached, the following steps are carried out.
4. All symmetry elements found in the stash are used to generate all possible symmetry operations which are then used to symmetrise the molecule: each symmetry operation is applied on the original molecule to produce a symmetry-equivalent copy, then all symmetry-equivalent copies are averaged to give the symmetrised molecule.
5. Repeat steps 1 to 4 above until convergence is reached.

### Convergence criteria

There are two convergence criteria for the symmetrisation procedure:

- **either** when the loose-threshold symmetry agrees with the target-threshold symmetry,
- **or** when the target-threshold symmetry contains more elements than the loose-threshold symmetry and has been consistently identified for a pre-specified number of consecutive iterations.

At least one criterion must be satisfied in order for convergence to be reached.

### Thresholds

As described in [Symmetry-group detection/#Thresholds](symmetry-group-detection.md/#thresholds), for each threshold level, two values are required: one for the comparisons between moments of inertia, and one for the comparisons of atomic coordinates and normalised vector components.
The choice of these thresholds affects the way the molecular system is symmetrised: the `loose` thresholds more or less determine the symmetry to which the system is symmetrised, and the `target` thresholds determine how well, or how tightly, the system is symmetrised.

### External fields

If external fields are present, they are also symmetrised together with the molecule, as the fictitious special atoms representing these fields are included in the symmetrisation protocol described above.
However, the magnitudes of these fields are **not** preserved during the symmetrisation for two reasons:

- when fictitious special atoms are constructed, their positions do not quite reflect field magnitudes but are rather chosen to ensure numerical stability in symmetry-group detection, and
- during the symmetrisation, all interatomic quantities such as bond lengths and bond angles are bound to vary so that the system can attain higher symmetry.

If the original field magnitudes are desired, it is trivial to use the positions of the fictitious special atoms after symmetrisation to reconstruct the external fields with the appropriate magnitudes.


## Parameters

!!! info "Feature requirements"

    - Using the Python API requires the [`python` feature](../getting-started/prerequisites.md/#rust-features).

The input parameters and their descriptions for each mode of running molecule symmetrisation in QSym² are given below.
When an input parameter has a default value, the default value will be specified.

=== "Command-line interface"
    ```yaml
    symmetry_group_detection: !Parameters #(1)!
      ...

    analysis_targets:
    - !MolecularSymmetry #(2)!
      xyz: /path/to/xyz/file #(3)!
      symmetrisation: #(4)!
        reorientate_molecule: true #(5)!
        loose_moi_threshold: 0.01 #(6)!
        loose_distance_threshold: 0.01 #(7)!
        target_moi_threshold: 1e-7 #(8)!
        target_distance_threshold: 1e-7 #(9)!
        max_iterations: 50 #(10)!
        consistent_target_symmetry_iterations: 10 #(11)!
        infinite_order_to_finite: null #(12)!
        use_magnetic_group: false #(13)!
        verbose: 0 #(14)!
        symmetrised_result_xyz: null #(15)!
        symmetrised_result_save_name: null #(16)!
    ```

    1. :fontawesome-solid-users: This specifies the parameters for [symmetry-group detection](symmetry-group-detection.md). These are required and will be used to perform a symmetry-group detection calculation on the symmetrised system. See the documentation for [symmetry-group detection](symmetry-group-detection.md) for more details.
    2. :fontawesome-solid-laptop-code: Under the hood, the specification of molecule symmetrisation in a YAML configuration file is handled by the `MolecularSymmetry` variant of the [`AnalysisTarget`](https://qsym2.dev/api/qsym2/interfaces/input/analysis/enum.AnalysisTarget.html) enum. This is so that a symmetry-group detection calculation can be performed on the symmetrised system.
    3. :fontawesome-solid-users: This specifies a path to an XYZ file containing the input molecular structure for symmetrisation.
    4. :fontawesome-solid-users: This key is optional: if it is omitted, no symmetrisation will be performed on the input molecular structure and only symmetry-group detection will be run (see [symmetry-group detection](symmetry-group-detection.md) for an illustration of this).</br></br>:fontawesome-solid-laptop-code: Under the hood, this wraps around the [`MoleculeSymmetrisationBootstrapParams`](https://qsym2.dev/api/qsym2/drivers/molecule_symmetrisation_bootstrap/struct.MoleculeSymmetrisationBootstrapParams.html) struct.
    5. :fontawesome-solid-users: This boolean indicates if the molecule is reoriented to align its principal axes with the space-fixed Cartesian axes at every iteration in the symmetrisation.</br></br> :material-cog-sync-outline: Default: `true`.
    6. :fontawesome-solid-users: This float specifies the `loose` moment-of-inertia threshold for the symmetrisation. The symmetry elements found at this threshold level will be used to bootstrap the symmetry of the molecule.</br></br> :material-cog-sync-outline: Default: `1e-2`.
    7. :fontawesome-solid-users: This float specifies the `loose` distance threshold for the symmetrisation. The symmetry elements found at this threshold level will be used to bootstrap the symmetry of the molecule.</br></br> :material-cog-sync-outline: Default: `1e-2`.
    8. :fontawesome-solid-users: This float specifies the `target` moment-of-inertia threshold for the symmetrisation.</br></br> :material-cog-sync-outline: Default: `1e-7`.
    9. :fontawesome-solid-users: This float specifies the `target` distance threshold for the symmetrisation.</br></br> :material-cog-sync-outline: Default: `1e-7`.
    10. :fontawesome-solid-users: This integer specifies the maximum number of symmetrisation iterations.</br></br> :material-cog-sync-outline: Default: `50`.
    11. :fontawesome-solid-users: This integer specifies the number of consecutive iterations during which the symmetry group at the `target` level of threshold must be consistently found for convergence to be reached, *if this group cannot become identical to the symmetry group at the `loose` level of threshold*.</br></br> :material-cog-sync-outline: Default: `10`.
    12. :fontawesome-solid-users: This optional integer specifies the finite order to which any infinite-order symmetry element is reduced, so that a finite number of symmetry operations can be used for the symmetrisation.</br></br> :material-cog-sync-outline: Default: `null`.
    13. :fontawesome-solid-users: This boolean indicates if any available magnetic group should be used for symmetrisation instead of the unitary group, *i.e.* if time-reversed operations, if any, should also be considered.</br></br> :material-cog-sync-outline: Default: `false`.
    14. :fontawesome-solid-users: This integer sppecifies the output verbosity level.</br></br> :material-cog-sync-outline: Default: `0`.
    15. :fontawesome-solid-users: This specifies an optional name (without the `.xyz` extension) for writing the symmetrised molecule to an XYZ file. If `null`, no XYZ files will be written.</br></br> :material-cog-sync-outline: Default: `null`.
    16. :fontawesome-solid-users: This specifies an optional name (without the `.qsym2.sym` extension) for saving the symmetry-group detection **verification** result of the symmetrised system as a binary file of type [`QSym2FileType::Sym`]. If `null`, the result will not be saved. Note that this is different from the `result_save_name` key of the [`symmetry_group_detection`](symmetry-group-detection.md) section.</br></br> :material-cog-sync-outline: Default: `null`.

=== "Python"
    ```py
    from qsym2 import symmetrise_molecule, PyMolecule

    pymol = PyMolecule( #(1)!
        atoms=[ #(2)!
            ("V", [+0.0, +0.0, +0.0]),
            ("F", [+1.0, +0.0, +0.0]),
            ("F", [-1.0, +0.0, +0.0]),
            ("F", [+0.0, +1.0, +0.0]),
            ("F", [+0.0, -1.0, +0.0]),
            ("F", [+0.0, +0.0, +1.1]),
            ("F", [+0.0, +0.0, -1.0]),
        ],
        threshold=1e-7, #(3)!
        magnetic_field=[0.0, 0.1, 0.0], #(4)!
        electric_field=None, #(5)!
    )

    symmol = symmetrise_molecule( #(6)!
        inp_xyz=None, #(7)!
        inp_mol=pymol, #(8)!
        out_target_sym=None, #(9)!
        loose_moi_threshold=1e-1, #(10)!
        loose_distance_threshold=1.5e-1, #(11)!
        target_moi_threshold=1e-8, #(12)!
        target_distance_threshold=1e-8, #(13)!
        use_magnetic_group=False, #(14)!
        reorientate_molecule=True, #(15)!
        max_iterations=50, #(16)!
        consistent_target_symmetry_iterations=10, #(17)!
        verbose=2, #(18)!
    ) #(19)!
    ```

    1. :fontawesome-solid-laptop-code: :fontawesome-solid-users: This specifies the molecular system for symmetry-group detection. The [`PyMolecule`](../python/symmetry-group-detection.md/#qsym2.PyMolecule) class, constructible in Python, contains geometry information that can be interpreted by QSym² on the Rust side.
    <br></br> :fontawesome-solid-laptop-code: The [Rust API documentation](https://qsym2.dev/api/qsym2/bindings/python/symmetry_group_detection/struct.PyMolecule.html) and [Python API documentation](../python/symmetry-group-detection.md/#qsym2.PyMolecule) for this class can be consulted for further information.
    2. :fontawesome-solid-users: The coordinates of the atoms specified in this list can be in any units. QSym² does not care what the actual units are &mdash; symmetry properties are invariant to any change in length scale. The only exception is when QSym² evaluates molecular integrals: here, atomic units will be assumed.
    3. :fontawesome-solid-users: This specifies the threshold for comparing molecules. Note that this threshold is not the same as those specified in `symmetrise_molecule` below.
    4. :fontawesome-solid-users: This list gives the components of an optional uniform external magnetic field that is present in the system. When external fields are present, they are also included in the symmetrisation process.</br></br> :material-cog-sync-outline: Default: `None`.
    5. :fontawesome-solid-users: This list gives the components of an optional uniform external electric field that is present in the system. When external fields are present, they are also included in the symmetrisation process.</br></br> :material-cog-sync-outline: Default: `None`.
    6. :fontawesome-solid-users: The [`symmetrise_molecule`](../python/molecule-symmetrisation.md/#qsym2.symmetrise_molecule) function performs molecule symmetrisation by bootstrapping and logs the result via the `qsym2-output` logger at the `INFO` level.
    <br></br> :fontawesome-solid-laptop-code: The [Rust API documentation](https://qsym2.dev/api/qsym2/bindings/python/molecule_symmetrisation/fn.symmetrise_molecule.html) and [Python API documentation](../python/molecule-symmetrisation.md/#qsym2.symmetrise_molecule) for this function can be consulted for further information.
    7. :fontawesome-solid-users: This specifies a path to an XYZ file containing the geometry of the molecule for symmetrisation. One and only one of `inp_xyz` or `inp_mol` must be specified, and the other must be `None`.
    8. :fontawesome-solid-users: This specifies a [`PyMolecule`](../python/symmetry-group-detection.md/#qsym2.PyMolecule) object containing the molecular system for symmetrisation. One and only one of `inp_xyz` or `inp_mol` must be specified, and the other must be `None`.
    9. :fontawesome-solid-users: This specifies an optional name for the `.qsym2.sym` file to be saved that contains the serialised results of the verification symmetry-group detection of the symmetrised molecule at the target thresholds. This name does not need to contain the `.qsym2.sym` extension.</br></br> :material-cog-sync-outline: Default: `None`.
    10. :fontawesome-solid-users: This float specifies the `loose` moment-of-inertia threshold for the symmetrisation. The symmetry elements found at this threshold level will be used to bootstrap the symmetry of the molecule.</br></br> :material-cog-sync-outline: Default: `1e-2`.
    11. :fontawesome-solid-users: This float specifies the `loose` distance threshold for the symmetrisation. The symmetry elements found at this threshold level will be used to bootstrap the symmetry of the molecule.</br></br> :material-cog-sync-outline: Default: `1e-2`.
    12. :fontawesome-solid-users: This float specifies the `target` moment-of-inertia threshold for the symmetrisation.</br></br> :material-cog-sync-outline: Default: `1e-7`.
    13. :fontawesome-solid-users: This float specifies the `target` distance threshold for the symmetrisation.</br></br> :material-cog-sync-outline: Default: `1e-7`.
    14. :fontawesome-solid-users: This boolean indicates if any available magnetic group should be used for symmetrisation instead of the unitary group, *i.e.* if time-reversed operations, if any, should also be considered.</br></br> :material-cog-sync-outline: Default: `False`.
    15. :fontawesome-solid-users: This boolean indicates if the molecule is reoriented to align its principal axes with the space-fixed Cartesian axes at every iteration in the symmetrisation.</br></br> :material-cog-sync-outline: Default: `True`.
    16. :fontawesome-solid-users: This integer specifies the maximum number of symmetrisation iterations.</br></br> :material-cog-sync-outline: Default: `50`.
    17. :fontawesome-solid-users: This integer specifies the number of consecutive iterations during which the symmetry group at the `target` level of threshold must be consistently found for convergence to be reached, *if this group cannot become identical to the symmetry group at the `loose` level of threshold*.</br></br> :material-cog-sync-outline: Default: `10`.
    18. :fontawesome-solid-users: This integer sppecifies the output verbosity level.</br></br> :material-cog-sync-outline: Default: `0`.
    19. :fontawesome-solid-laptop-code: :fontawesome-solid-users: The [`symmetrise_molecule`](../python/molecule-symmetrisation.md/#qsym2.symmetrise_molecule) function returns a [`PyMolecule`](../python/symmetry-group-detection.md/#qsym2.PyMolecule) obect containing the symmetrised molecular system.
