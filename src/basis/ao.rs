//! Atomic-orbital basis functions.

use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::convert::TryInto;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::slice::Iter;

use anyhow::{self, ensure, format_err};
use counter::Counter;
use derive_builder::Builder;
use itertools::{Itertools, izip};
use serde::{Deserialize, Serialize};

use crate::angmom::ANGMOM_LABELS;
use crate::auxiliary::atom::Atom;
use crate::auxiliary::misc::ProductRepeat;
use crate::permutation::{PermutableCollection, Permutation, permute_inplace};

#[cfg(test)]
#[path = "ao_tests.rs"]
mod ao_tests;

// -------------------
// Shell order structs
// -------------------

// ~~~~~~~~~
// PureOrder
// ~~~~~~~~~

/// Structure to contain information about the ordering of pure Gaussians of a certain rank.
#[derive(Clone, Builder, PartialEq, Eq, Hash)]
pub struct PureOrder {
    /// A sequence of $`m_l`$ values giving the ordering of the pure Gaussians.
    #[builder(setter(custom))]
    mls: Vec<i32>,

    /// The rank of the pure Gaussians.
    pub lpure: u32,

    /// The spatial parity of the pure Gaussians: `true` if even under spatial inversion and
    /// `false` if odd. By default, this is the same as the parity of `Self::lpure`.
    #[builder(default = "self.default_even()?")]
    pub even: bool,
}

impl PureOrderBuilder {
    fn mls(&mut self, mls: &[i32]) -> &mut Self {
        let lpure = self.lpure.expect("`lpure` has not been set.");
        assert!(
            mls.iter()
                .map(|m| m.unsigned_abs())
                .max()
                .expect("The maximum |m| value could not be determined.")
                == lpure
        );
        assert_eq!(mls.len(), (2 * lpure + 1) as usize);
        self.mls = Some(mls.to_vec());
        self
    }

    fn default_even(&self) -> Result<bool, String> {
        let lpure = self
            .lpure
            .ok_or_else(|| "`lpure` has not been set.".to_string())?;
        Ok(lpure % 2 == 0)
    }
}

impl PureOrder {
    /// Returns a builder to construct a new [`PureOrder`] structure.
    fn builder() -> PureOrderBuilder {
        PureOrderBuilder::default()
    }

    /// Constructs a new [`PureOrder`] structure from its constituting $`m_l`$ values. The spatial
    /// inversion parity is determined by the parity of the deduced $`l`$ value.
    pub fn new(mls: &[i32]) -> Result<Self, anyhow::Error> {
        let lpure = mls
            .iter()
            .map(|m| m.unsigned_abs())
            .max()
            .expect("The maximum |m| value could not be determined.");
        let pure_order = PureOrder::builder()
            .lpure(lpure)
            .mls(mls)
            .build()
            .map_err(|err| format_err!(err))?;
        ensure!(pure_order.verify(), "Invalid `PureOrder`.");
        Ok(pure_order)
    }

    /// Constructs a new [`PureOrder`] structure for a specified rank with increasing-$`m`$ order.
    /// The spatial inversion parity is determined by the parity of the deduced $`l`$ value.
    ///
    /// # Arguments
    ///
    /// * `lpure` - The required pure Gaussian rank.
    ///
    /// # Returns
    ///
    /// A [`PureOrder`] struct for a specified rank with increasing-$`m`$ order.
    #[must_use]
    pub fn increasingm(lpure: u32) -> Self {
        let lpure_i32 = i32::try_from(lpure).expect("`lpure` cannot be converted to `i32`.");
        let mls = (-lpure_i32..=lpure_i32).collect_vec();
        Self::builder()
            .lpure(lpure)
            .mls(&mls)
            .build()
            .expect("Unable to construct a `PureOrder` structure with increasing-m order.")
    }

    /// Constructs a new [`PureOrder`] structure for a specified rank with decreasing-$`m`$ order.
    /// The spatial inversion parity is determined by the parity of the deduced $`l`$ value.
    ///
    /// # Arguments
    ///
    /// * `lpure` - The required pure Gaussian rank.
    ///
    /// # Returns
    ///
    /// A [`PureOrder`] struct for a specified rank with decreasing-$`m`$ order.
    #[must_use]
    pub fn decreasingm(lpure: u32) -> Self {
        let lpure_i32 = i32::try_from(lpure).expect("`lpure` cannot be converted to `i32`.");
        let mls = (-lpure_i32..=lpure_i32).rev().collect_vec();
        Self::builder()
            .lpure(lpure)
            .mls(&mls)
            .build()
            .expect("Unable to construct a `PureOrder` structure with decreasing-m order.")
    }

    /// Constructs a new [`PureOrder`] structure for a specified rank with Molden order. The spatial
    /// inversion parity is determined by the parity of the deduced $`l`$ value.
    ///
    /// # Arguments
    ///
    /// * `lpure` - The required pure Gaussian rank.
    ///
    /// # Returns
    ///
    /// A [`PureOrder`] struct for a specified rank with Molden order.
    #[must_use]
    pub fn molden(lpure: u32) -> Self {
        let lpure_i32 = i32::try_from(lpure).expect("`lpure` cannot be converted to `i32`.");
        let mls = (0..=lpure_i32)
            .flat_map(|absm| {
                if absm == 0 {
                    vec![0]
                } else {
                    vec![absm, -absm]
                }
            })
            .collect_vec();
        Self::builder()
            .lpure(lpure)
            .mls(&mls)
            .build()
            .expect("Unable to construct a `PureOrder` structure with Molden order.")
    }

    /// Verifies if this [`PureOrder`] struct is valid.
    ///
    /// # Returns
    ///
    /// A boolean indicating if this [`PureOrder`] struct is valid.
    #[must_use]
    pub fn verify(&self) -> bool {
        let mls_set = self.mls.iter().collect::<HashSet<_>>();
        let lpure = self.lpure;
        mls_set.len() == self.ncomps() && mls_set.iter().all(|m| m.unsigned_abs() <= lpure)
    }

    /// Iterates over the constituent $`m_l`$ values.
    pub fn iter(&'_ self) -> Iter<'_, i32> {
        self.mls.iter()
    }

    /// Returns the number of pure components in the shell.
    pub fn ncomps(&self) -> usize {
        let lpure = usize::try_from(self.lpure).unwrap_or_else(|_| {
            panic!(
                "Unable to convert the pure degree {} to `usize`.",
                self.lpure
            )
        });
        2 * lpure + 1
    }

    /// Returns the $`m`$ value with a specified index in this shell.
    pub fn get_m_with_index(&self, i: usize) -> Option<i32> {
        self.mls.get(i).cloned()
    }
}

impl fmt::Display for PureOrder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Pure rank: {}", self.lpure)?;
        writeln!(f, "Order:")?;
        for m in self.iter() {
            writeln!(f, "  {m}")?;
        }
        Ok(())
    }
}

impl fmt::Debug for PureOrder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Pure rank: {}", self.lpure)?;
        writeln!(f, "Order:")?;
        for m in self.iter() {
            writeln!(f, "  {m:?}")?;
        }
        Ok(())
    }
}

impl PermutableCollection for PureOrder {
    type Rank = usize;

    fn get_perm_of(&self, other: &Self) -> Option<Permutation<Self::Rank>> {
        let o_mls: HashMap<&i32, usize> = other
            .mls
            .iter()
            .enumerate()
            .map(|(i, o_m)| (o_m, i))
            .collect();
        let image_opt: Option<Vec<Self::Rank>> =
            self.mls.iter().map(|s_m| o_mls.get(s_m).copied()).collect();
        image_opt.and_then(|image| Permutation::from_image(image).ok())
    }

    fn permute(&self, perm: &Permutation<Self::Rank>) -> Result<Self, anyhow::Error> {
        let mut p_pureorder = self.clone();
        p_pureorder.permute_mut(perm)?;
        Ok(p_pureorder)
    }

    fn permute_mut(&mut self, perm: &Permutation<Self::Rank>) -> Result<(), anyhow::Error> {
        permute_inplace(&mut self.mls, perm);
        Ok(())
    }
}

// ~~~~~~~~~
// CartOrder
// ~~~~~~~~~

/// Structure to contain information about the ordering of Cartesian Gaussians of a certain rank.
#[derive(Clone, Builder, PartialEq, Eq, Hash)]
pub struct CartOrder {
    /// A sequence of $`(l_x, l_y, l_z)`$ tuples giving the ordering of the Cartesian Gaussians.
    #[builder(setter(custom))]
    pub cart_tuples: Vec<(u32, u32, u32)>,

    /// The rank of the Cartesian Gaussians.
    pub lcart: u32,
}

impl CartOrderBuilder {
    fn cart_tuples(&mut self, cart_tuples: &[(u32, u32, u32)]) -> &mut Self {
        let lcart = self.lcart.expect("`lcart` has not been set.");
        assert!(
            cart_tuples.iter().all(|(lx, ly, lz)| lx + ly + lz == lcart),
            "Inconsistent total Cartesian orders between components."
        );
        assert_eq!(
            cart_tuples.len(),
            ((lcart + 1) * (lcart + 2)).div_euclid(2) as usize,
            "Unexpected number of components for `lcart` = {}.",
            lcart
        );
        self.cart_tuples = Some(cart_tuples.to_vec());
        self
    }
}

impl CartOrder {
    /// Returns a builder to construct a new [`CartOrder`] structure.
    fn builder() -> CartOrderBuilder {
        CartOrderBuilder::default()
    }

    /// Constructs a new [`CartOrder`] structure from its constituting tuples, each of which contains
    /// the $`x`$, $`y`$, and $`z`$ exponents for one Cartesian term.
    ///
    /// # Errors
    ///
    /// Errors if the Cartesian tuples are invalid (*e.g.* missing components or containing
    /// inconsistent components).
    pub fn new(cart_tuples: &[(u32, u32, u32)]) -> Result<Self, anyhow::Error> {
        let first_tuple = cart_tuples
            .first()
            .ok_or(format_err!("No Cartesian tuples found."))?;
        let lcart = first_tuple.0 + first_tuple.1 + first_tuple.2;
        let cart_order = CartOrder::builder()
            .lcart(lcart)
            .cart_tuples(cart_tuples)
            .build()
            .map_err(|err| format_err!(err))?;
        ensure!(cart_order.verify(), "Invalid `CartOrder`.");
        Ok(cart_order)
    }

    /// Constructs a new [`CartOrder`] structure for a specified rank with lexicographic order.
    ///
    /// # Arguments
    ///
    /// * `lcart` - The required Cartesian Gaussian rank.
    ///
    /// # Returns
    ///
    /// A [`CartOrder`] struct for a specified rank with lexicographic order.
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

    /// Constructs a new [`CartOrder`] structure for a specified rank with Molden order.
    ///
    /// # Arguments
    ///
    /// * `lcart` - The required Cartesian Gaussian rank up to 4.
    ///
    /// # Returns
    ///
    /// A [`CartOrder`] struct for a specified rank with Q-Chem order.
    ///
    /// # Panics
    ///
    /// Panics if `lcart` is greater than 4.
    #[must_use]
    pub fn molden(lcart: u32) -> Self {
        assert!(lcart <= 4, "`lcart` > 4 is not specified by Molden.");
        let cart_tuples: Vec<(u32, u32, u32)> = match lcart {
            0 => vec![(0, 0, 0)],
            1 => vec![(1, 0, 0), (0, 1, 0), (0, 0, 1)],
            2 => vec![
                (2, 0, 0),
                (0, 2, 0),
                (0, 0, 2),
                (1, 1, 0),
                (1, 0, 1),
                (0, 1, 1),
            ],
            3 => vec![
                (3, 0, 0),
                (0, 3, 0),
                (0, 0, 3),
                (1, 2, 0),
                (2, 1, 0),
                (2, 0, 1),
                (1, 0, 2),
                (0, 1, 2),
                (0, 2, 1),
                (1, 1, 1),
            ],
            4 => vec![
                (4, 0, 0),
                (0, 4, 0),
                (0, 0, 4),
                (3, 1, 0),
                (3, 0, 1),
                (1, 3, 0),
                (0, 3, 1),
                (1, 0, 3),
                (0, 1, 3),
                (2, 2, 0),
                (2, 0, 2),
                (0, 2, 2),
                (2, 1, 1),
                (1, 2, 1),
                (1, 1, 2),
            ],
            _ => panic!("`lcart` > 4 is not specified by Molden."),
        };
        Self::builder()
            .lcart(lcart)
            .cart_tuples(&cart_tuples)
            .build()
            .expect("Unable to construct a `CartOrder` structure with Molden order.")
    }

    /// Constructs a new [`CartOrder`] structure for a specified rank with Q-Chem order.
    ///
    /// # Arguments
    ///
    /// * `lcart` - The required Cartesian Gaussian rank.
    ///
    /// # Returns
    ///
    /// A [`CartOrder`] struct for a specified rank with Q-Chem order.
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

    /// Verifies if this [`CartOrder`] struct is valid.
    ///
    /// # Returns
    ///
    /// A boolean indicating if this [`CartOrder`] struct is valid.
    #[must_use]
    pub fn verify(&self) -> bool {
        let cart_tuples_set = self.cart_tuples.iter().collect::<HashSet<_>>();
        let lcart = self.lcart;
        cart_tuples_set.len() == self.ncomps()
            && cart_tuples_set
                .iter()
                .all(|(lx, ly, lz)| lx + ly + lz == lcart)
    }

    /// Iterates over the constituent tuples.
    pub fn iter(&'_ self) -> Iter<'_, (u32, u32, u32)> {
        self.cart_tuples.iter()
    }

    /// Returns the number of Cartesian components in the shell.
    pub fn ncomps(&self) -> usize {
        let lcart = usize::try_from(self.lcart).unwrap_or_else(|_| {
            panic!(
                "Unable to convert the Cartesian degree {} to `usize`.",
                self.lcart
            )
        });
        ((lcart + 1) * (lcart + 2)).div_euclid(2)
    }

    /// Returns the Cartesian component with a specified index in this shell.
    pub fn get_cart_tuple_with_index(&self, i: usize) -> Option<(u32, u32, u32)> {
        self.cart_tuples.get(i).cloned()
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
            .map(|s_cart_tuple| o_cart_tuples.get(s_cart_tuple).copied())
            .collect();
        image_opt.and_then(|image| Permutation::from_image(image).ok())
    }

    fn permute(&self, perm: &Permutation<Self::Rank>) -> Result<Self, anyhow::Error> {
        let mut p_cartorder = self.clone();
        p_cartorder.permute_mut(perm)?;
        Ok(p_cartorder)
    }

    fn permute_mut(&mut self, perm: &Permutation<Self::Rank>) -> Result<(), anyhow::Error> {
        permute_inplace(&mut self.cart_tuples, perm);
        Ok(())
    }
}

/// Translates a Cartesian exponent tuple to a human-understandable string.
///
/// # Arguments
///
/// * `cart_tuple` - A tuple of $`(l_x, l_y, l_z)`$ specifying the exponents of the Cartesian
///   components of the Cartesian Gaussian.
/// * flat - A flag indicating if the string representation is flat (*e.g.* `xxyz`) or compact
///   (*e.g.* `x^2yz`).
///
/// Returns
///
/// The string representation of the Cartesian exponent tuple.
pub(crate) fn cart_tuple_to_str(cart_tuple: &(u32, u32, u32), flat: bool) -> String {
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

// ~~~~~~~~~~~
// SpinorOrder
// ~~~~~~~~~~~

// SpinorParticleType enum
// .......................

/// Enumerated type for spinor particle types.
#[derive(Clone, PartialEq, Eq, Hash, Serialize, Deserialize, Debug)]
pub enum SpinorParticleType {
    /// Variant for spinors describing fermionic particles. The optional associated value contains
    /// any additional balance symmetry to be applied to the spinors.
    Fermion(Option<SpinorBalanceSymmetry>),

    /// Variant for spinors describing antifermionic particles. The optional associated value
    /// contains any additional balance symmetry to be applied to the spinors.
    Antifermion(Option<SpinorBalanceSymmetry>),
}

impl fmt::Display for SpinorParticleType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SpinorParticleType::Fermion(None) => {
                write!(f, "fermion")
            }
            SpinorParticleType::Fermion(Some(sbs)) => {
                write!(f, "fermion ({sbs})")
            }
            SpinorParticleType::Antifermion(None) => {
                write!(f, "antifermion")
            }
            SpinorParticleType::Antifermion(Some(sbs)) => {
                write!(f, "antifermion ({sbs})")
            }
        }
    }
}

// SpinorBalanceSymmetry enum
// ..........................

/// Enumerated type for spinor balance symmetry types.
#[derive(Clone, PartialEq, Eq, Hash, Serialize, Deserialize, Debug)]
pub enum SpinorBalanceSymmetry {
    /// Variant for kinetic balance, σ·p.
    KineticBalance,
}

impl fmt::Display for SpinorBalanceSymmetry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SpinorBalanceSymmetry::KineticBalance => {
                write!(f, "σ·p")
            }
        }
    }
}

/// Structure to contain information about the ordering of spinors of a certain rank.
#[derive(Clone, Builder, PartialEq, Eq, Hash)]
pub struct SpinorOrder {
    /// The angular momentum (times two) of the spinor Gaussians. This must be an odd integer.
    #[builder(setter(custom))]
    pub two_j: u32,

    /// A sequence of $`2m_j`$ values giving the ordering of the spinor Gaussians.
    #[builder(setter(custom))]
    two_mjs: Vec<i32>,

    /// The spatial inversion parity of the spinor Gaussians: `true` if even under spatial inversion
    /// and `false` if odd. This is intrinsic to the spinor Gaussians and independent of the
    /// particle type and any associated balance symmetry.
    pub even: bool,

    /// The particle type described by this spinor shell.
    pub particle_type: SpinorParticleType,
}

impl SpinorOrderBuilder {
    fn two_j(&mut self, two_j: u32) -> &mut Self {
        if two_j.rem_euclid(2) != 1 {
            panic!("`two_j` must be odd.")
        }
        self.two_j = Some(two_j);
        self
    }

    fn two_mjs(&mut self, two_mjs: &[i32]) -> &mut Self {
        let two_j = self.two_j.expect("`two_j` has not been set.");
        let max_two_m = two_mjs
            .iter()
            .map(|two_m| two_m.unsigned_abs())
            .max()
            .expect("The maximum |2m| value could not be determined.");
        assert_eq!(
            max_two_m, two_j,
            "The maximum |2m| value does not equal 2j = {two_j}."
        );
        assert_eq!(
            two_mjs.len(),
            (two_j + 1) as usize,
            "The number of 2m values specified does not equal 2j + 1 = {}",
            two_j + 1
        );
        self.two_mjs = Some(two_mjs.to_vec());
        self
    }
}

impl SpinorOrder {
    /// Returns a builder to construct a new [`SpinorOrder`] structure.
    fn builder() -> SpinorOrderBuilder {
        SpinorOrderBuilder::default()
    }

    /// Constructs a new [`SpinorOrder`] structure from its constituting $`2m_j`$ values and a
    /// specified spatial parity.
    pub fn new(
        two_mjs: &[i32],
        even: bool,
        particle_type: SpinorParticleType,
    ) -> Result<Self, anyhow::Error> {
        let two_j = two_mjs
            .iter()
            .map(|two_m| two_m.unsigned_abs())
            .max()
            .ok_or_else(|| format_err!("The maximum |2m| value could not be determined."))?;
        let spinor_order = SpinorOrder::builder()
            .two_j(two_j)
            .two_mjs(two_mjs)
            .even(even)
            .particle_type(particle_type)
            .build()
            .map_err(|err| format_err!(err))?;
        ensure!(spinor_order.verify(), "Invalid `SpinorOrder`.");
        Ok(spinor_order)
    }

    /// Constructs a new [`SpinorOrder`] structure for a specified angular momentum with
    /// increasing-$`m`$ order.
    ///
    /// # Arguments
    ///
    /// * `two_j` - The required spinor angular momentum (times two).
    /// * `even` - Boolean indicating whether the large-component spinors are even with respect to
    ///   spatial inversion.
    /// * `particle_type` - The associated particle type.
    ///
    /// # Returns
    ///
    /// A [`SpinorOrder`] struct for a specified angular momentum with increasing-$`m`$ order.
    #[must_use]
    pub fn increasingm(two_j: u32, even: bool, particle_type: SpinorParticleType) -> Self {
        let two_j_i32 = i32::try_from(two_j).expect("`two_j` cannot be converted to `i32`.");
        let two_mjs = (-two_j_i32..=two_j_i32).step_by(2).collect_vec();
        Self::builder()
            .two_j(two_j)
            .two_mjs(&two_mjs)
            .even(even)
            .particle_type(particle_type)
            .build()
            .expect("Unable to construct a `SpinorOrder` structure with increasing-m order.")
    }

    /// Constructs a new [`SpinorOrder`] structure for a specified angular momentum with
    /// decreasing-$`m`$ order.
    ///
    /// # Arguments
    ///
    /// * `two_j` - The required spinor angular momentum (times two).
    /// * `even` - Boolean indicating whether the large-component spinors are even with respect to
    ///   spatial inversion.
    /// * `particle_type` - The associated particle type.
    ///
    /// # Returns
    ///
    /// A [`SpinorOrder`] struct for a specified angular momentum with decreasing-$`m`$ order.
    #[must_use]
    pub fn decreasingm(two_j: u32, even: bool, particle_type: SpinorParticleType) -> Self {
        let two_j_i32 = i32::try_from(two_j).expect("`two_j` cannot be converted to `i32`.");
        let two_mjs = (-two_j_i32..=two_j_i32).rev().step_by(2).collect_vec();
        Self::builder()
            .two_j(two_j)
            .two_mjs(&two_mjs)
            .even(even)
            .particle_type(particle_type)
            .build()
            .expect("Unable to construct a `SpinorOrder` structure with decreasing-m order.")
    }

    /// Constructs a new [`SpinorOrder`] structure for a specified angular momentum with Molden order.
    ///
    /// # Arguments
    ///
    /// * `two_j` - The required spinor angular momentum (times two).
    /// * `even` - Boolean indicating whether the large-component spinors are even with respect to
    ///   spatial inversion.
    /// * `particle_type` - The associated particle type.
    ///
    /// # Returns
    ///
    /// A [`SpinorOrder`] struct for a specified angular momentum with Molden order.
    #[must_use]
    pub fn molden(two_j: u32, even: bool, particle_type: SpinorParticleType) -> Self {
        let two_j_i32 = i32::try_from(two_j).expect("`two_j` cannot be converted to `i32`.");
        let two_mjs = (1..=two_j_i32)
            .step_by(2)
            .flat_map(|abs_twom| vec![abs_twom, -abs_twom])
            .collect_vec();
        Self::builder()
            .two_j(two_j)
            .two_mjs(&two_mjs)
            .even(even)
            .particle_type(particle_type)
            .build()
            .expect("Unable to construct a `SpinorOrder` structure with Molden order.")
    }

    /// Verifies if this [`SpinorOrder`] struct is valid.
    ///
    /// # Returns
    ///
    /// A boolean indicating if this [`SpinorOrder`] struct is valid.
    #[must_use]
    pub fn verify(&self) -> bool {
        let two_mjs_set = self.two_mjs.iter().collect::<HashSet<_>>();
        let two_j = self.two_j;
        two_mjs_set.len() == self.ncomps()
            && two_mjs_set
                .iter()
                .all(|two_m| two_m.unsigned_abs() <= two_j)
    }

    /// Iterates over the constituent $`2m_j`$ values.
    pub fn iter(&'_ self) -> Iter<'_, i32> {
        self.two_mjs.iter()
    }

    /// Returns the number of pure components in the shell.
    pub fn ncomps(&self) -> usize {
        let two_j = usize::try_from(self.two_j).unwrap_or_else(|_| {
            panic!(
                "Unable to convert the two-j value {} to `usize`.",
                self.two_j
            )
        });
        two_j + 1
    }

    /// Returns the $`2m`$ value with a specified index in this shell.
    pub fn get_two_m_with_index(&self, i: usize) -> Option<i32> {
        self.two_mjs.get(i).cloned()
    }

    /// Returns the angular momentum quantum number of the spatial part of the spinors in this
    /// shell.
    pub fn l(&self) -> u32 {
        if self.two_j.rem_euclid(4) == 1 {
            // even l is to the left of j, and odd l to the right
            if self.even {
                (self.two_j - 1).div_euclid(2)
            } else {
                (self.two_j + 1).div_euclid(2)
            }
        } else {
            assert_eq!(self.two_j.rem_euclid(4), 3);
            // even l is to the right of j, and odd l to the left
            if self.even {
                (self.two_j + 1).div_euclid(2)
            } else {
                (self.two_j - 1).div_euclid(2)
            }
        }
    }

    /// Returns the value of $`a = 2(j - l)`$.
    pub fn a(&self) -> i8 {
        if self.two_j.rem_euclid(4) == 1 {
            // even l is to the left of j, and odd l to the right
            if self.even { 1 } else { -1 }
        } else {
            assert_eq!(self.two_j.rem_euclid(4), 3);
            // even l is to the right of j, and odd l to the left
            if self.even { -1 } else { 1 }
        }
    }
}

impl fmt::Display for SpinorOrder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "Angular momentum: {}/2 ({}; l = {}), {}",
            self.two_j,
            if self.a() == 1 { "+" } else { "-" },
            self.l(),
            self.particle_type,
        )?;
        writeln!(f, "Order:")?;
        for two_m in self.iter() {
            writeln!(f, "  {two_m}/2")?;
        }
        Ok(())
    }
}

impl fmt::Debug for SpinorOrder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "Angular momentum: {}/2 ({}; l = {}), {}",
            self.two_j,
            if self.a() == 1 { "+" } else { "-" },
            self.l(),
            self.particle_type,
        )?;
        writeln!(f, "Order:")?;
        for two_m in self.iter() {
            writeln!(f, "  {two_m:?}/2")?;
        }
        Ok(())
    }
}

impl PermutableCollection for SpinorOrder {
    type Rank = usize;

    fn get_perm_of(&self, other: &Self) -> Option<Permutation<Self::Rank>> {
        let o_two_mjs: HashMap<&i32, usize> = other
            .two_mjs
            .iter()
            .enumerate()
            .map(|(i, o_two_m)| (o_two_m, i))
            .collect();
        let image_opt: Option<Vec<Self::Rank>> = self
            .two_mjs
            .iter()
            .map(|s_two_m| o_two_mjs.get(s_two_m).copied())
            .collect();
        image_opt.and_then(|image| Permutation::from_image(image).ok())
    }

    fn permute(&self, perm: &Permutation<Self::Rank>) -> Result<Self, anyhow::Error> {
        let mut p_pureorder = self.clone();
        p_pureorder.permute_mut(perm)?;
        Ok(p_pureorder)
    }

    fn permute_mut(&mut self, perm: &Permutation<Self::Rank>) -> Result<(), anyhow::Error> {
        permute_inplace(&mut self.two_mjs, perm);
        Ok(())
    }
}

// ----------
// ShellOrder
// ----------

/// Enumerated type to indicate the type of the angular functions in a shell and how they are
/// ordered.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum ShellOrder {
    /// This variant indicates that the angular functions are real solid harmonics. The associated
    /// value is a [`PureOrder`] struct containing the order of these functions.
    Pure(PureOrder),

    /// This variant indicates that the angular functions are Cartesian functions. The associated
    /// value is a [`CartOrder`] struct containing the order of these functions.
    Cart(CartOrder),

    /// This variant indicates that the angular functions are spinors. The associated
    /// value is a [`SpinorOrder`] struct containing the order of these functions.
    Spinor(SpinorOrder),
}

impl fmt::Display for ShellOrder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ShellOrder::Pure(pure_order) => write!(
                f,
                "Pure ({}) ({})",
                if pure_order.even { "g" } else { "u" },
                pure_order.iter().map(|m| m.to_string()).join(", ")
            ),
            ShellOrder::Cart(cart_order) => write!(
                f,
                "Cart ({}) ({})",
                if cart_order.lcart % 2 == 0 { "g" } else { "u" },
                cart_order
                    .iter()
                    .map(|cart_tuple| { cart_tuple_to_str(cart_tuple, true) })
                    .join(", ")
            ),
            ShellOrder::Spinor(spinor_order) => write!(
                f,
                "Spinor ({}; l = {}), {}, ({})",
                if spinor_order.a() == 1 { "+" } else { "-" },
                spinor_order.l(),
                spinor_order.particle_type,
                spinor_order
                    .iter()
                    .map(|two_m| format!("{two_m}/2"))
                    .join(", ")
            ),
        }
    }
}

// ----------
// BasisShell
// ----------

/// Structure representing a shell in an atomic-orbital basis set.
#[derive(Clone, Builder, PartialEq, Eq, Hash, Debug)]
pub struct BasisShell {
    /// A non-negative integer indicating the rank of the shell. If this shell is pure, the rank
    /// equals its angular momentum. If this shell is Cartesian, the rank is the sum of the
    /// exponents of the Cartesian coordinates. If this shell is a spinor, the rank is twice its
    /// angular momentum.
    #[builder(setter(custom))]
    pub l: u32,

    /// An enum indicating the type of the angular functions in a shell and how they are ordered.
    #[builder(setter(custom))]
    pub shell_order: ShellOrder,
}

impl BasisShellBuilder {
    fn l(&mut self, l: u32) -> &mut Self {
        match self.shell_order.as_ref() {
            Some(ShellOrder::Pure(pure_order)) => assert_eq!(pure_order.lpure, l),
            Some(ShellOrder::Cart(cart_order)) => assert_eq!(cart_order.lcart, l),
            Some(ShellOrder::Spinor(spinor_order)) => assert_eq!(spinor_order.two_j, l),
            None => {}
        }
        self.l = Some(l);
        self
    }

    fn shell_order(&mut self, shl_ord: ShellOrder) -> &mut Self {
        match (&shl_ord, self.l) {
            (ShellOrder::Pure(pure_order), Some(l)) => assert_eq!(pure_order.lpure, l),
            (ShellOrder::Cart(cart_order), Some(l)) => assert_eq!(cart_order.lcart, l),
            (ShellOrder::Spinor(spinor_order), Some(l)) => assert_eq!(spinor_order.two_j, l),
            _ => {}
        }
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

    /// Constructs a new [`BasisShell`].
    ///
    /// # Arguments
    ///
    /// * `l` - The rank of this shell, which is equal to $`l_{\mathrm{pure}}`$ for a pure shell,
    ///   $`l_{\mathrm{cart}}`$ for a Cartesian shell, or $`2j`$ for a spinor shell.
    /// * `shl_ord` - A [`ShellOrder`] structure specifying the type and ordering of the basis
    ///   functions in this shell.
    pub fn new(l: u32, shl_ord: ShellOrder) -> Self {
        match &shl_ord {
            ShellOrder::Pure(pureorder) => assert_eq!(pureorder.lpure, l),
            ShellOrder::Cart(cartorder) => assert_eq!(cartorder.lcart, l),
            ShellOrder::Spinor(spinororder) => assert_eq!(spinororder.two_j, l),
        }
        BasisShell::builder()
            .l(l)
            .shell_order(shl_ord)
            .build()
            .expect("Unable to construct a `BasisShell`.")
    }

    /// The number of basis functions in this shell.
    pub fn n_funcs(&self) -> usize {
        let lsize = self.l as usize;
        match self.shell_order {
            ShellOrder::Pure(_) => 2 * lsize + 1,
            ShellOrder::Cart(_) => ((lsize + 1) * (lsize + 2)).div_euclid(2),
            ShellOrder::Spinor(_) => lsize + 1,
        }
    }
}

// ---------
// BasisAtom
// ---------

/// Structure containing the ordered sequence of the shells for an atom.
#[derive(Clone, Builder, PartialEq, Eq, Hash, Debug)]
pub struct BasisAtom<'a> {
    /// An atom in the basis set.
    pub(crate) atom: &'a Atom,

    /// The ordered shells associated with this atom.
    #[builder(setter(custom))]
    pub(crate) basis_shells: Vec<BasisShell>,
}

impl<'a> BasisAtomBuilder<'a> {
    pub(crate) fn basis_shells(&mut self, bss: &[BasisShell]) -> &mut Self {
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
    pub(crate) fn builder() -> BasisAtomBuilder<'a> {
        BasisAtomBuilder::default()
    }

    /// Constructs a new [`BasisAtom`].
    ///
    /// # Arguments
    ///
    /// * `atm` - A reference to an atom.
    /// * `bss` - A sequence of [`BasisShell`]s containing the basis functions localised on this
    ///   atom.
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

// -----------------
// BasisAngularOrder
// -----------------

/// Structure containing the angular momentum information of an atomic-orbital basis set that is
/// required for symmetry transformation to be performed.
#[derive(Clone, Builder)]
pub struct BasisAngularOrder<'a> {
    /// An ordered sequence of [`BasisAtom`] in the order the atoms are defined in the molecule.
    #[builder(setter(custom))]
    pub(crate) basis_atoms: Vec<BasisAtom<'a>>,
}

impl<'a> BasisAngularOrderBuilder<'a> {
    pub(crate) fn basis_atoms(&mut self, batms: &[BasisAtom<'a>]) -> &mut Self {
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
    pub(crate) fn builder() -> BasisAngularOrderBuilder<'a> {
        BasisAngularOrderBuilder::default()
    }

    /// Constructs a new [`BasisAngularOrder`] structure from the constituting [`BasisAtom`]s
    /// without any additional balance symmetry auxiliary information.
    ///
    /// # Arguments
    ///
    /// * `batms` - The constituent [`BasisAtom`]s.
    pub fn new(batms: &[BasisAtom<'a>]) -> Self {
        BasisAngularOrder::builder()
            .basis_atoms(batms)
            .build()
            .expect("Unable to construct a `BasisAngularOrder`.")
    }

    /// The number of atoms in the basis.
    pub fn n_atoms(&self) -> usize {
        self.basis_atoms.len()
    }

    /// The number of basis functions in this basis.
    pub fn n_funcs(&self) -> usize {
        self.basis_atoms.iter().map(BasisAtom::n_funcs).sum()
    }

    /// The ordered tuples of 0-based shell indices indicating the starting (inclusive) and ending
    /// (exclusive) shell positions of the atoms in this basis.
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

    /// The ordered tuples of 0-based function indices indicating the starting (inclusive) and
    /// ending (exclusive) positions of the shells in this basis.
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

    /// An iterator over the constituent [`BasisShell`]s in this basis.
    pub fn basis_shells(&self) -> impl Iterator<Item = &BasisShell> + '_ {
        self.basis_atoms
            .iter()
            .flat_map(|basis_atom| basis_atom.basis_shells.iter())
    }

    /// Determines the permutation of the functions in this [`BasisAngularOrder`] to map `self` to
    /// `other`, given that the shells themselves remain unchanged while only the functions in each
    /// shell are permuted.
    ///
    /// For example, consider `self`:
    /// ```text
    /// S (1)
    /// P (x, y, z)
    /// D (xx, xy, xz, yy, yz, zz)
    /// ```
    ///
    /// and `other`:
    /// ```text
    /// S (1)
    /// P (y, z, x)
    /// D (xx, xy, yy, xz, yz, zz)
    /// ```
    ///
    /// the mapping permutation is given by `π(0, 3, 1, 2, 4, 5, 7, 6, 8, 9)`.
    ///
    /// # Arguments
    ///
    /// * `other` - Another [`BasisAngularOrder`] to be compared against.
    ///
    /// # Returns
    ///
    /// The mapping permutation, if any.
    pub(crate) fn get_perm_of_functions_fixed_shells(
        &self,
        other: &Self,
    ) -> Result<Permutation<usize>, anyhow::Error> {
        if self.n_funcs() == other.n_funcs() && self.n_atoms() == other.n_atoms() {
            let s_shell_boundaries = self.shell_boundary_indices();
            let o_shell_boundaries = other.shell_boundary_indices();
            if s_shell_boundaries.len() == o_shell_boundaries.len() {
                izip!(
                    self.basis_shells(),
                    other.basis_shells(),
                    s_shell_boundaries.iter(),
                    o_shell_boundaries.iter()
                )
                .map(|(s_bs, o_bs, (s_start, s_end), (o_start, o_end))| {
                    if (s_start, s_end) == (o_start, o_end) {
                        let s_shl_ord = &s_bs.shell_order;
                        let o_shl_ord = &o_bs.shell_order;
                        match (s_shl_ord, o_shl_ord) {
                            (ShellOrder::Pure(s_po), ShellOrder::Pure(o_po)) => Ok(
                                s_po.get_perm_of(o_po)
                                    .unwrap()
                                    .image()
                                    .iter()
                                    .map(|x| s_start + x)
                                    .collect_vec(),
                            ),
                            (ShellOrder::Cart(s_co), ShellOrder::Cart(o_co)) => Ok(
                                s_co.get_perm_of(o_co)
                                    .unwrap()
                                    .image()
                                    .iter()
                                    .map(|x| s_start + x)
                                    .collect_vec(),
                            ),
                            _ => Err(format_err!("At least one pair of corresponding shells have mismatched pure/cart.")),
                        }
                    } else {
                        Err(format_err!("At least one pair of corresponding shells have mismatched boundary indices."))
                    }
                })
                .collect::<Result<Vec<_>, _>>()
                .and_then(|image_by_shells| {
                    let flattened_image = image_by_shells.into_iter().flatten().collect_vec();
                    Permutation::from_image(flattened_image)
                })
            } else {
                Err(format_err!("Mismatched numbers of shells."))
            }
        } else {
            Err(format_err!("Mismatched numbers of basis functions."))
        }
    }
}

impl<'a> PermutableCollection for BasisAngularOrder<'a> {
    type Rank = usize;

    /// Determines the permutation of [`BasisAtom`]s to map `self` to `other`.
    ///
    /// # Arguments
    ///
    /// * `other` - Another [`BasisAngularOrder`] to be compared with `self`.
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
        image_opt.and_then(|image| Permutation::from_image(image).ok())
    }

    fn permute(&self, perm: &Permutation<Self::Rank>) -> Result<Self, anyhow::Error> {
        let mut p_bao = self.clone();
        p_bao.permute_mut(perm)?;
        Ok(p_bao)
    }

    fn permute_mut(&mut self, perm: &Permutation<Self::Rank>) -> Result<(), anyhow::Error> {
        permute_inplace(&mut self.basis_atoms, perm);
        Ok(())
    }
}

impl<'a> fmt::Display for BasisAngularOrder<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let order_length = self
            .basis_shells()
            .map(|v| v.shell_order.to_string().chars().count())
            .max()
            .unwrap_or(20);
        let atom_index_length = self.n_atoms().to_string().chars().count();
        writeln!(f, "{}", "┈".repeat(17 + atom_index_length + order_length))?;
        writeln!(f, " {:>atom_index_length$}  Atom  Shell  Order", "#")?;
        writeln!(f, "{}", "┈".repeat(17 + atom_index_length + order_length))?;
        for (atm_i, batm) in self.basis_atoms.iter().enumerate() {
            let atm = batm.atom;
            for (shl_i, bshl) in batm.basis_shells.iter().enumerate() {
                if shl_i == 0 {
                    writeln!(
                        f,
                        " {:>atom_index_length$}  {:<4}  {:<5}  {:<order_length$}",
                        atm_i,
                        atm.atomic_symbol,
                        match &bshl.shell_order {
                            ShellOrder::Pure(_) | ShellOrder::Cart(_) => ANGMOM_LABELS
                                .get(usize::try_from(bshl.l).unwrap_or_else(|err| panic!("{err}")))
                                .copied()
                                .unwrap_or(&bshl.l.to_string())
                                .to_string(),
                            ShellOrder::Spinor(spinororder) => format!("{}/2", spinororder.two_j),
                        },
                        bshl.shell_order
                    )?;
                } else {
                    writeln!(
                        f,
                        " {:>atom_index_length$}  {:<4}  {:<5}  {:<order_length$}",
                        "",
                        "",
                        match &bshl.shell_order {
                            ShellOrder::Pure(_) | ShellOrder::Cart(_) => ANGMOM_LABELS
                                .get(usize::try_from(bshl.l).unwrap_or_else(|err| panic!("{err}")))
                                .copied()
                                .unwrap_or(&bshl.l.to_string())
                                .to_string(),
                            ShellOrder::Spinor(spinororder) => format!("{}/2", spinororder.two_j),
                        },
                        bshl.shell_order
                    )?;
                }
            }
        }
        writeln!(f, "{}", "┈".repeat(17 + atom_index_length + order_length))?;
        Ok(())
    }
}

impl<'a> PartialEq for BasisAngularOrder<'a> {
    fn eq(&self, other: &Self) -> bool {
        self.basis_atoms == other.basis_atoms
    }
}

impl<'a> Eq for BasisAngularOrder<'a> {}

impl<'a> Hash for BasisAngularOrder<'a> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.basis_atoms.hash(state);
    }
}

impl<'a> fmt::Debug for BasisAngularOrder<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{:?}", self.basis_atoms)
    }
}
