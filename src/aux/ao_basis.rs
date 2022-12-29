use std::cmp::Ordering;
use std::collections::HashSet;
use std::convert::TryInto;
use std::fmt;
use std::slice::Iter;

use counter::Counter;
use derive_builder::Builder;
use itertools::Itertools;

use crate::aux::atom::Atom;
use crate::aux::misc::ProductRepeat;

#[cfg(test)]
#[path = "ao_basis_tests.rs"]
mod ao_basis_tests;

/// A struct to contain information about the ordering of Cartesian Gaussians of a certain rank.
#[derive(Clone, Builder, PartialEq, Eq, Hash)]
pub struct CartOrder {
    /// A sequence of $`(l_x, l_y, l_z)`$ tuples giving the ordering of the Cartesian Gaussians.
    #[builder(setter(custom))]
    cart_tuples: Vec<(u32, u32, u32)>,

    /// The rank of the Cartesian Gaussians.
    lcart: u32,
}

impl CartOrderBuilder {
    fn cart_tuples(&mut self, cart_tuples: &[(u32, u32, u32)]) -> &mut Self {
        let lcart = self.lcart.unwrap();
        assert!(cart_tuples.iter().all(|(lx, ly, lz)| lx + ly + lz == lcart));
        assert_eq!(
            cart_tuples.len(),
            ((lcart + 1) * (lcart + 2)).div_euclid(2) as usize
        );
        self.cart_tuples = Some(cart_tuples.to_vec());
        self
    }
}

impl CartOrder {
    /// Returns a builder to construct a new `CartOrder` struct.
    ///
    /// # Returns
    ///
    /// A builder to construct a new `CartOrder` struct.
    fn builder() -> CartOrderBuilder {
        CartOrderBuilder::default()
    }

    /// Constructs a new `CartOrder` struct for a specified rank with lexicographic order.
    ///
    /// # Arguments
    ///
    /// * lcart - The required Cartesian Gaussian rank.
    ///
    /// # Returns
    ///
    /// A `CartOrder` struct for a specified rank with lexicographic order.
    pub fn lex(lcart: u32) -> Self {
        let mut cart_tuples =
            Vec::with_capacity(((lcart + 1) * (lcart + 2)).div_euclid(2) as usize);
        for lx in (0..=lcart).rev() {
            for ly in (0..=(lcart - lx)).rev() {
                cart_tuples.push((lx, ly, lcart - lx - ly));
            }
        }
        Self::builder()
            .lcart(lcart)
            .cart_tuples(&cart_tuples)
            .build()
            .unwrap()
    }

    /// Constructs a new `CartOrder` struct for a specified rank with Q-Chem order.
    ///
    /// # Arguments
    ///
    /// * lcart - The required Cartesian Gaussian rank.
    ///
    /// # Returns
    ///
    /// A `CartOrder` struct for a specified rank with Q-Chem order.
    pub fn qchem(lcart: u32) -> Self {
        let cart_tuples: Vec<(u32, u32, u32)> = if lcart > 0 {
            (0..3)
                .product_repeat(lcart as usize)
                .filter_map(|tup| {
                    let mut tup_sorted = tup.clone();
                    tup_sorted.sort();
                    tup_sorted.reverse();
                    if tup == tup_sorted {
                        let lcartqns = tup.iter().collect::<Counter<_>>();
                        Some((
                            <usize as TryInto<u32>>::try_into(*(lcartqns.get(&0).unwrap_or(&0)))
                                .unwrap(),
                            <usize as TryInto<u32>>::try_into(*(lcartqns.get(&1).unwrap_or(&0)))
                                .unwrap(),
                            <usize as TryInto<u32>>::try_into(*(lcartqns.get(&2).unwrap_or(&0)))
                                .unwrap(),
                        ))
                    } else {
                        None
                    }
                })
                .collect()
        } else {
            vec![(0, 0, 0)]
        };
        Self::builder()
            .lcart(lcart)
            .cart_tuples(&cart_tuples)
            .build()
            .unwrap()
    }

    /// Verifies if this `CartOrder` struct is valid.
    ///
    /// # Returns
    ///
    /// A boolean indicating if this `CartOrder` struct is valid.
    pub fn verify(&self) -> bool {
        let cart_tuples_set = self.cart_tuples.iter().collect::<HashSet<_>>();
        let lcart = self.lcart;
        cart_tuples_set.len() == ((lcart + 1) * (lcart + 2)).div_euclid(2) as usize
            && cart_tuples_set
                .iter()
                .all(|(lx, ly, lz)| lx + ly + lz == lcart)
    }

    pub fn iter(&self) -> Iter<(u32, u32, u32)> {
        self.cart_tuples.iter()
    }
}

impl fmt::Display for CartOrder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Cartesian rank: {}", self.lcart)?;
        writeln!(f, "Order:")?;
        for cart_tuple in self.iter() {
            writeln!(f, "  {}", cart_tuple_to_str(cart_tuple, true))?;
        }
        Ok(())
    }
}

impl fmt::Debug for CartOrder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Cartesian rank: {}", self.lcart)?;
        writeln!(f, "Order:")?;
        for cart_tuple in self.iter() {
            writeln!(f, "  {:?}", cart_tuple)?;
        }
        Ok(())
    }
}

/// Translates a Cartesian exponent tuple to a human-understandable string.
///
/// # Arguments
///
/// * cart_tuple - A tuple of $`(l_x, l_y, l_z)`$ specifying the exponents of the Cartesian
/// components of the Cartesian Gaussian.
/// * flat - A flag indicating if the string representation is flat (*e.g.* `xxyz`) or compact
/// (*e.g.* `x^2yz`).
///
/// Returns
///
/// The string representation of the Cartesian exponent tuple.
fn cart_tuple_to_str(cart_tuple: &(u32, u32, u32), flat: bool) -> String {
    if cart_tuple.0 + cart_tuple.1 + cart_tuple.2 == 0u32 {
        "1".to_string()
    } else {
        let cart_array = [cart_tuple.0, cart_tuple.1, cart_tuple.2];
        let carts = ["x", "y", "z"];
        Itertools::intersperse(
            cart_array.iter().enumerate().map(|(i, &l)| {
                if flat {
                    carts[i].repeat(l as usize)
                } else {
                    match l.cmp(&1) {
                        Ordering::Greater => format!("{}^{}", carts[i], l),
                        Ordering::Equal => carts[i].to_string(),
                        Ordering::Less => "".to_string(),
                    }
                }
            }),
            "".to_string(),
        )
        .collect::<String>()
    }
}

/// An enum to indicate the type of the angular functions in a shell and how they are ordered.
enum ShellOrder {
    /// This variant indicates that the angular functions are real solid harmonics. The associated
    /// value is a flag indicating if the functions are arranged in increasing $`m`$ order.
    Pure(bool),

    /// This variant indicates that the angular functions are Cartesian functions. The associated
    /// value is a [`CartOrder`] struct containing the order of these functions.
    Cart(CartOrder),
}

/// A struct representing a shell in an atomic-orbital basis set.
struct BasisShell {
    /// A non-negative integer indicating the rank of the shell.
    l: u32,

    /// An enum indicating the type of the angular functions in a shell and how they are ordered.
    shell_order: ShellOrder,
}

impl BasisShell {
    /// The number of basis functions in this shell.
    fn n_funcs(&self) -> usize {
        let lsize = self.l as usize;
        match self.shell_order {
            ShellOrder::Pure(_) => 2 * lsize + 1,
            ShellOrder::Cart(_) => ((lsize + 1) * (lsize + 2)).div_euclid(2),
        }
    }
}

/// A struct containing the ordered sequence of the shells for an atom.
struct BasisAtom {
    /// An atom in the basis set.
    atom: Atom,

    /// The ordered shells associated with this atom.
    basis_shells: Vec<BasisShell>,
}

impl BasisAtom {
    /// The number of basis functions localised on this atom.
    fn n_funcs(&self) -> usize {
        self.basis_shells
            .iter()
            .map(|basis_shell| basis_shell.n_funcs())
            .sum()
    }

    /// The ordered tuples of 0-based indices indicating the starting (inclusive) and ending
    /// (exclusive) positions of the shells on this atom.
    fn shell_boundary_indices(&self) -> Vec<(usize, usize)> {
        self.basis_shells
            .iter()
            .scan(0, |acc, basis_shell| {
                let start_index = *acc;
                *acc += basis_shell.n_funcs();
                Some((start_index, *acc))
            })
            .collect::<Vec<_>>()
    }
}

/// A struct containing the angular momentum information of an atomic-orbital basis set that is
/// required for symmetry transformation to be performed.
pub struct BasisAngularOrder {
    /// An ordered sequence of [`BasisAtom`] in the order the atoms are defined in the molecule.
    basis_atoms: Vec<BasisAtom>,
}

impl BasisAngularOrder {
    /// The number of basis functions in this basis set.
    fn n_funcs(&self) -> usize {
        self.basis_atoms
            .iter()
            .map(|basis_atom| basis_atom.n_funcs())
            .sum()
    }

    /// The ordered tuples of 0-based indices indicating the starting (inclusive) and ending
    /// (exclusive) shell positions of the atoms in this basis set.
    fn atom_boundary_indices(&self) -> Vec<(usize, usize)> {
        self.basis_atoms
            .iter()
            .scan(0, |acc, basis_atom| {
                let start_index = *acc;
                *acc += basis_atom.n_funcs();
                Some((start_index, *acc))
            })
            .collect::<Vec<_>>()
    }

    /// The ordered tuples of 0-based indices indicating the starting (inclusive) and ending
    /// (exclusive) positions of the shells in this basis set.
    fn shell_boundary_indices(&self) -> Vec<(usize, usize)> {
        let atom_boundary_indices = self.atom_boundary_indices();
        self.basis_atoms
            .iter()
            .zip(atom_boundary_indices)
            .flat_map(|(basis_atom, (atom_start, _))| {
                basis_atom
                    .shell_boundary_indices()
                    .iter()
                    .map(|(shell_start, shell_end)| {
                        (shell_start + atom_start, shell_end + atom_start)
                    })
            })
            .collect::<Vec<_>>()
    }
}
