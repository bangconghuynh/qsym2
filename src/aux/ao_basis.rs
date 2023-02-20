use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::convert::TryInto;
use std::fmt;
use std::slice::Iter;

use counter::Counter;
use derive_builder::Builder;
use itertools::Itertools;

use crate::aux::atom::Atom;
use crate::aux::misc::ProductRepeat;
use crate::permutation::{permute_inplace, PermutableCollection, Permutation};

#[cfg(test)]
#[path = "ao_basis_tests.rs"]
mod ao_basis_tests;

// ------
// Shells
// ------

/// A struct to contain information about the ordering of Cartesian Gaussians of a certain rank.
#[derive(Clone, Builder, PartialEq, Eq, Hash)]
pub struct CartOrder {
    /// A sequence of $`(l_x, l_y, l_z)`$ tuples giving the ordering of the Cartesian Gaussians.
    #[builder(setter(custom))]
    cart_tuples: Vec<(u32, u32, u32)>,

    /// The rank of the Cartesian Gaussians.
    pub lcart: u32,
}

impl CartOrderBuilder {
    fn cart_tuples(&mut self, cart_tuples: &[(u32, u32, u32)]) -> &mut Self {
        let lcart = self.lcart.expect("`lcart` has not been set.");
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
    #[must_use]
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
            .expect("Unable to construct a `CartOrder` structure with lexicographic order.")
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
    #[must_use]
    pub fn qchem(lcart: u32) -> Self {
        let cart_tuples: Vec<(u32, u32, u32)> = if lcart > 0 {
            (0..3)
                .product_repeat(lcart as usize)
                .filter_map(|tup| {
                    let mut tup_sorted = tup.clone();
                    tup_sorted.sort_unstable();
                    tup_sorted.reverse();
                    if tup == tup_sorted {
                        let lcartqns = tup.iter().collect::<Counter<_>>();
                        Some((
                            <usize as TryInto<u32>>::try_into(*(lcartqns.get(&0).unwrap_or(&0)))
                                .expect("Unable to convert Cartesian x-exponent to `u32`."),
                            <usize as TryInto<u32>>::try_into(*(lcartqns.get(&1).unwrap_or(&0)))
                                .expect("Unable to convert Cartesian y-exponent to `u32`."),
                            <usize as TryInto<u32>>::try_into(*(lcartqns.get(&2).unwrap_or(&0)))
                                .expect("Unable to convert Cartesian z-exponent to `u32`."),
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
            .expect("Unable to construct a `CartOrder` structure with Q-Chem order.")
    }

    /// Verifies if this `CartOrder` struct is valid.
    ///
    /// # Returns
    ///
    /// A boolean indicating if this `CartOrder` struct is valid.
    #[must_use]
    pub fn verify(&self) -> bool {
        let cart_tuples_set = self.cart_tuples.iter().collect::<HashSet<_>>();
        let lcart = self.lcart;
        cart_tuples_set.len() == self.ncomps()
            && cart_tuples_set
                .iter()
                .all(|(lx, ly, lz)| lx + ly + lz == lcart)
    }

    pub fn iter(&self) -> Iter<(u32, u32, u32)> {
        self.cart_tuples.iter()
    }

    pub fn ncomps(&self) -> usize {
        let lcart = usize::try_from(self.lcart).unwrap_or_else(|_| {
            panic!(
                "Unable to convert the Cartesian degree {} to `usize`.",
                self.lcart
            )
        });
        ((lcart + 1) * (lcart + 2)).div_euclid(2)
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
            writeln!(f, "  {cart_tuple:?}")?;
        }
        Ok(())
    }
}

impl PermutableCollection for CartOrder {
    type Rank = usize;

    fn get_perm_of(&self, other: &Self) -> Option<Permutation<Self::Rank>> {
        let o_cart_tuples: HashMap<&(u32, u32, u32), usize> = other
            .cart_tuples
            .iter()
            .enumerate()
            .map(|(i, o_cart_tuple)| (o_cart_tuple, i))
            .collect();
        let image_opt: Option<Vec<Self::Rank>> = self
            .cart_tuples
            .iter()
            .map(|s_cart_tuple| {
                o_cart_tuples
                    .get(s_cart_tuple)
                    .copied()
            })
            .collect();
        image_opt.map(|image| Permutation::from_image(image))
    }

    fn permute(&self, perm: &Permutation<Self::Rank>) -> Self {
        let mut p_cartorder = self.clone();
        p_cartorder.permute_mut(perm);
        p_cartorder
    }

    fn permute_mut(&mut self, perm: &Permutation<Self::Rank>) {
        permute_inplace(&mut self.cart_tuples, perm);
    }
}

/// Translates a Cartesian exponent tuple to a human-understandable string.
///
/// # Arguments
///
/// * `cart_tuple` - A tuple of $`(l_x, l_y, l_z)`$ specifying the exponents of the Cartesian
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
                        Ordering::Greater => format!("{}^{l}", carts[i]),
                        Ordering::Equal => carts[i].to_string(),
                        Ordering::Less => String::new(),
                    }
                }
            }),
            String::new(),
        )
        .collect::<String>()
    }
}

/// An enum to indicate the type of the angular functions in a shell and how they are ordered.
#[derive(Clone, PartialEq, Eq, Hash)]
pub enum ShellOrder {
    /// This variant indicates that the angular functions are real solid harmonics. The associated
    /// value is a flag indicating if the functions are arranged in increasing $`m`$ order.
    Pure(bool),

    /// This variant indicates that the angular functions are Cartesian functions. The associated
    /// value is a [`CartOrder`] struct containing the order of these functions.
    Cart(CartOrder),
}

/// A struct representing a shell in an atomic-orbital basis set.
#[derive(Clone, Builder, PartialEq, Eq, Hash)]
pub struct BasisShell {
    /// A non-negative integer indicating the rank of the shell.
    #[builder(setter(custom))]
    pub l: u32,

    /// An enum indicating the type of the angular functions in a shell and how they are ordered.
    #[builder(setter(custom))]
    pub shell_order: ShellOrder,
}

impl BasisShellBuilder {
    fn l(&mut self, l: u32) -> &mut Self {
        if let Some(ShellOrder::Cart(cart_order)) = self.shell_order.as_ref() {
            assert_eq!(cart_order.lcart, l);
        }
        self.l = Some(l);
        self
    }

    fn shell_order(&mut self, shl_ord: ShellOrder) -> &mut Self {
        if let (ShellOrder::Cart(cart_order), Some(l)) = (shl_ord.clone(), self.l) {
            assert_eq!(cart_order.lcart, l);
        };
        self.shell_order = Some(shl_ord);
        self
    }
}

impl BasisShell {
    /// Returns a builder to construct a new [`BasisShell`].
    ///
    /// # Returns
    ///
    /// A builder to construct a new [`BasisShell`].
    fn builder() -> BasisShellBuilder {
        BasisShellBuilder::default()
    }

    pub fn new(l: u32, shl_ord: ShellOrder) -> Self {
        BasisShell::builder()
            .l(l)
            .shell_order(shl_ord)
            .build()
            .expect("Unable to construct a `BasisShell`.")
    }

    /// The number of basis functions in this shell.
    fn n_funcs(&self) -> usize {
        let lsize = self.l as usize;
        match self.shell_order {
            ShellOrder::Pure(_) => 2 * lsize + 1,
            ShellOrder::Cart(_) => ((lsize + 1) * (lsize + 2)).div_euclid(2),
        }
    }
}

// -----
// Atoms
// -----

/// A struct containing the ordered sequence of the shells for an atom.
#[derive(Clone, Builder, PartialEq, Eq, Hash)]
pub struct BasisAtom<'a> {
    /// An atom in the basis set.
    atom: &'a Atom,

    /// The ordered shells associated with this atom.
    #[builder(setter(custom))]
    basis_shells: Vec<BasisShell>,
}

impl<'a> BasisAtomBuilder<'a> {
    fn basis_shells(&mut self, bss: &[BasisShell]) -> &mut Self {
        self.basis_shells = Some(bss.to_vec());
        self
    }
}

impl<'a> BasisAtom<'a> {
    /// Returns a builder to construct a new [`BasisAtom`].
    ///
    /// # Returns
    ///
    /// A builder to construct a new [`BasisAtom`].
    fn builder() -> BasisAtomBuilder<'a> {
        BasisAtomBuilder::default()
    }

    pub fn new(atm: &'a Atom, bss: &[BasisShell]) -> Self {
        BasisAtom::builder()
            .atom(atm)
            .basis_shells(bss)
            .build()
            .expect("Unable to construct a `BasisAtom`.")
    }

    /// The number of basis functions localised on this atom.
    fn n_funcs(&self) -> usize {
        self.basis_shells.iter().map(BasisShell::n_funcs).sum()
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

// -----
// Basis
// -----

/// A struct containing the angular momentum information of an atomic-orbital basis set that is
/// required for symmetry transformation to be performed.
#[derive(Clone, Builder, PartialEq, Eq, Hash)]
pub struct BasisAngularOrder<'a> {
    /// An ordered sequence of [`BasisAtom`] in the order the atoms are defined in the molecule.
    #[builder(setter(custom))]
    basis_atoms: Vec<BasisAtom<'a>>,
}

impl<'a> BasisAngularOrderBuilder<'a> {
    fn basis_atoms(&mut self, batms: &[BasisAtom<'a>]) -> &mut Self {
        self.basis_atoms = Some(batms.to_vec());
        self
    }
}

impl<'a> BasisAngularOrder<'a> {
    /// Returns a builder to construct a new [`BasisAngularOrder`].
    ///
    /// # Returns
    ///
    /// A builder to construct a new [`BasisAngularOrder`].
    #[must_use]
    pub fn builder() -> BasisAngularOrderBuilder<'a> {
        BasisAngularOrderBuilder::default()
    }

    pub fn new(batms: &[BasisAtom<'a>]) -> Self {
        BasisAngularOrder::builder()
            .basis_atoms(batms)
            .build()
            .expect("Unable to construct a `BasisAngularOrder`.")
    }

    pub fn n_atoms(&self) -> usize {
        self.basis_atoms.len()
    }

    /// The number of basis functions in this basis set.
    pub fn n_funcs(&self) -> usize {
        self.basis_atoms.iter().map(BasisAtom::n_funcs).sum()
    }

    /// The ordered tuples of 0-based indices indicating the starting (inclusive) and ending
    /// (exclusive) shell positions of the atoms in this basis set.
    pub fn atom_boundary_indices(&self) -> Vec<(usize, usize)> {
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
    pub fn shell_boundary_indices(&self) -> Vec<(usize, usize)> {
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
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>()
    }

    pub fn basis_shells(&self) -> impl Iterator<Item = &BasisShell> + '_ {
        self.basis_atoms
            .iter()
            .flat_map(|basis_atom| basis_atom.basis_shells.iter())
    }
}

impl<'a> PermutableCollection for BasisAngularOrder<'a> {
    type Rank = usize;

    /// Determines the permutation of `BasisAtom`s to map `self` to `other`.
    ///
    /// # Arguments
    ///
    /// * `other` - Another `BasisAngularOrder` to be compared with `self`.
    ///
    /// # Returns
    ///
    /// Returns a permutation that permutes the *ordinary* atoms of `self` to give `other`, or
    /// `None` if no such permutation exists.
    fn get_perm_of(&self, other: &Self) -> Option<Permutation<Self::Rank>> {
        let o_basis_atoms: HashMap<&BasisAtom, usize> = other
            .basis_atoms
            .iter()
            .enumerate()
            .map(|(i, basis_atom)| (basis_atom, i))
            .collect();
        let image_opt: Option<Vec<Self::Rank>> = self
            .basis_atoms
            .iter()
            .map(|s_basis_atom| o_basis_atoms.get(s_basis_atom).copied())
            .collect();
        image_opt.map(|image| Permutation::from_image(image))
    }

    /// Permutes the ordinary atoms in this molecule and places them in a new molecule to be
    /// returned.
    ///
    /// # Arguments
    ///
    /// * `perm` - A permutation for the atoms.
    ///
    /// # Returns
    ///
    /// A new molecule with the permuted ordinary atoms.
    ///
    /// # Panics
    ///
    /// Panics if the rank of `perm` does not match the number of atoms in this molecule.
    fn permute(&self, perm: &Permutation<Self::Rank>) -> Self {
        let mut p_bao = self.clone();
        p_bao.permute_mut(perm);
        p_bao
    }

    /// Permutes in-place the ordinary atoms in this molecule.
    ///
    /// The in-place rearrangement implementation is taken from
    /// [here](https://stackoverflow.com/a/69774341/5112668).
    ///
    /// # Arguments
    ///
    /// * `perm` - A permutation for the atoms.
    ///
    /// # Panics
    ///
    /// Panics if the rank of `perm` does not match the number of atoms in this molecule.
    fn permute_mut(&mut self, perm: &Permutation<Self::Rank>) {
        permute_inplace(&mut self.basis_atoms, perm);
    }
}
