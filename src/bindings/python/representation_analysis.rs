use anyhow::{self, ensure, format_err};
use pyo3::prelude::*;

use crate::angmom::ANGMOM_INDICES;
use crate::aux::ao_basis::{BasisAngularOrder, BasisAtom, BasisShell, CartOrder, ShellOrder};
use crate::aux::molecule::Molecule;

/// A Python-exposed structure to marshal basis angular order information between Python and Rust.
#[pyclass]
pub struct PyBasisAngularOrder {
    /// A vector of basis atoms. Each item in the vector is a tuple consisting of an atomic symbol
    /// and a vector of basis shell quartets whose components give:
    /// - the angular momentum symbol for the shell,
    /// - `true` if the shell is Cartesian, `false` if the shell is pure,
    /// - (this will be ignored if the shell is Cartesian) `Some(increasingm)` to indicate the
    /// order of pure functions in the shell,
    /// - (this will be ignored if the shell is pure) `None` if the Cartesian functions are in
    /// lexicographic order, `Some(vec![[lx, ly, lz], ...])` to specify a custom Cartesian order.
    basis_atoms: Vec<(
        String,
        Vec<(String, bool, Option<bool>, Option<Vec<(u32, u32, u32)>>)>,
    )>,
}

#[pymethods]
impl PyBasisAngularOrder {
    #[new]
    fn new(
        basis_atoms: Vec<(
            String,
            Vec<(String, bool, Option<bool>, Option<Vec<(u32, u32, u32)>>)>,
        )>,
    ) -> Self {
        Self { basis_atoms }
    }
}

impl PyBasisAngularOrder {
    fn to_bao<'b, 'a: 'b>(&'b self, mol: &'a Molecule) -> Result<BasisAngularOrder, anyhow::Error> {
        ensure!(
            self.basis_atoms.len() == mol.atoms.len(),
            "The number of basis atoms does not match the number of ordinary atoms."
        );
        let basis_atoms = self
            .basis_atoms
            .iter()
            .zip(mol.atoms.iter())
            .flat_map(|((element, basis_shells), atom)| {
                ensure!(
                    *element == atom.atomic_symbol,
                    "Expected element `{element}`, but found atom `{}`.",
                    atom.atomic_symbol
                );
                let bss = basis_shells
                    .iter()
                    .flat_map(|(angmom, cart, increasingm, cart_ord)| {
                        let l = ANGMOM_INDICES.get(angmom).unwrap();
                        let shl_ord = if *cart {
                            let cart_order = if let Some(cart_tuples) = cart_ord {
                                CartOrder::new(cart_tuples)?
                            } else {
                                CartOrder::lex(*l)
                            };
                            ShellOrder::Cart(cart_order)
                        } else {
                            ShellOrder::Pure(increasingm.ok_or(format_err!(
                                "Pure shell specified, but no pure order found."
                            ))?)
                        };
                        Ok::<_, anyhow::Error>(BasisShell::new(*l, shl_ord))
                    })
                    .collect::<Vec<_>>();
                Ok(BasisAtom::new(atom, &bss))
            })
            .collect::<Vec<_>>();
        Ok(BasisAngularOrder::new(&basis_atoms))
    }
}

// /// A Python-exposed function to perform molecule symmetrisation.
// #[pyfunction]
// #[pyo3(signature = (inp_loose_sym, out_tight_sym, use_magnetic_group, target_moi_threshold, target_distance_threshold, reorientate_molecule, max_iterations, verbose, infinite_order_to_finite=None))]
// pub(super) fn rep_analyse_slater_determinant(
