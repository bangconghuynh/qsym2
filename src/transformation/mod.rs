use ndarray::{Array, Axis, RemoveAxis};

use crate::aux::ao_basis::BasisAngularOrder;
use crate::permutation::Permutation;

#[cfg(test)]
#[path = "transformation_tests.rs"]
mod transformation_tests;

fn permute_array_by_atoms<D>(
    arr: &Array<f64, D>,
    atom_perm: &Permutation<usize>,
    axes: &[Axis],
    bao: &BasisAngularOrder,
) -> Array<f64, D>
where
    D: RemoveAxis,
{
    assert_eq!(
        atom_perm.rank(),
        bao.n_atoms(),
        "The rank of permutation does not match the number of atoms in the basis."
    );
    let atom_boundary_indices = bao.atom_boundary_indices();
    let permuted_shell_indices: Vec<usize> = atom_perm
        .image()
        .iter()
        .flat_map(|&i| {
            let (shell_min, shell_max) = atom_boundary_indices[i];
            shell_min..shell_max
        })
        .collect();

    let mut r = arr.clone();
    for axis in axes {
        r = r.select(*axis, &permuted_shell_indices);
    }
    r
}
