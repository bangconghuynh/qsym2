use anyhow::{self, ensure, format_err};
use derive_builder::Builder;
use serde::{Deserialize, Serialize};

use crate::aux::ao_basis::*;
use crate::aux::molecule::Molecule;

// ---------------
// InputShellOrder
// ---------------

/// A serialisable/deserialisable enumerated type to indicate the type of the angular functions in
/// a shell and how they are ordered.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) enum InputShellOrder {
    /// This variant indicates that the angular functions are real solid harmonics. The associated
    /// value is a flag indicating if the functions are arranged in increasing $`m`$ order.
    Pure(bool),

    /// This variant indicates that the angular functions are Cartesian functions arranged in
    /// lexicographic order.
    CartLexicographic,

    /// This variant indicates that the angular functions are Cartesian functions arranged in
    /// Q-Chem order.
    CartQChem,

    /// This variant indicates that the angular functions are Cartesian functions arranged in
    /// a custom order specified by the ordered exponent tuples.
    CartCustom(Vec<(u32, u32, u32)>),
}

impl InputShellOrder {
    /// Converts the [`InputShellOrder`] to a corresponding [`ShellOrder`].
    fn to_shell_order(&self, l: u32) -> ShellOrder {
        match self {
            InputShellOrder::Pure(increasing_m) => ShellOrder::Pure(*increasing_m),
            InputShellOrder::CartLexicographic => ShellOrder::Cart(CartOrder::lex(l)),
            InputShellOrder::CartQChem => ShellOrder::Cart(CartOrder::qchem(l)),
            InputShellOrder::CartCustom(cart_tuples) => ShellOrder::Cart(
                CartOrder::new(cart_tuples).expect("Invalid Cartesian tuples provided."),
            ),
        }
    }
}

// ---------------
// InputBasisShell
// ---------------

/// A serialisable/deserialisable structure representing a shell in an atomic-orbital basis set.
#[derive(Clone, Debug, Builder, Serialize, Deserialize)]
pub(crate) struct InputBasisShell {
    /// A non-negative integer indicating the rank of the shell.
    l: u32,

    /// An enum indicating the type of the angular functions in a shell and how they are ordered.
    shell_order: InputShellOrder,
}

impl InputBasisShell {
    /// Returns a builder to construct [`InputBasisShell`].
    pub(crate) fn builder() -> InputBasisShellBuilder {
        InputBasisShellBuilder::default()
    }

    /// Returns the number of basis functions in this shell.
    fn n_funcs(&self) -> usize {
        let lsize = self.l as usize;
        match self.shell_order {
            InputShellOrder::Pure(_) => 2 * lsize + 1,
            InputShellOrder::CartQChem
            | InputShellOrder::CartLexicographic
            | InputShellOrder::CartCustom(_) => ((lsize + 1) * (lsize + 2)).div_euclid(2),
        }
    }

    /// Converts the [`InputBasisShell`] to a corresponding [`BasisShell`].
    fn to_basis_shell(&self) -> BasisShell {
        BasisShell::new(self.l, self.shell_order.to_shell_order(self.l))
    }
}

// --------------
// InputBasisAtom
// --------------

/// A serialisable/deserialisable structure containing the ordered sequence of the shells for an
/// atom. However, unlike [`BasisAtom`], this structure does not contain a reference to the atom it
/// is describing, but instead it only contains an index and an owned string giving the element
/// name of the atom. This is only for serialisation/deserialisation purposes.
#[derive(Clone, Debug, Builder, Serialize, Deserialize)]
pub(crate) struct InputBasisAtom {
    /// The index and name of an atom in the basis set.
    atom: (usize, String),

    /// The ordered shells associated with this atom.
    basis_shells: Vec<InputBasisShell>,
}

impl InputBasisAtom {
    /// Returns a builder to construct [`InputBasisAtom`].
    pub(crate) fn builder() -> InputBasisAtomBuilder {
        InputBasisAtomBuilder::default()
    }

    /// Returns the number of basis functions localised on this atom.
    fn n_funcs(&self) -> usize {
        self.basis_shells.iter().map(InputBasisShell::n_funcs).sum()
    }

    /// Converts to a [`BasisAtom`] structure given a molecule.
    ///
    /// # Arguments
    ///
    /// * `mol` - A molecule to which the atom in this [`InputBasisAtom`] belongs.
    ///
    /// # Returns
    ///
    /// The corresponding [`BasisAtom`] structure.
    ///
    /// # Errors
    ///
    /// Errors if the atom index and name in this [`InputBasisAtom`] do not match the
    /// corresponding atom in `mol`.
    pub(crate) fn to_basis_atom<'a>(
        &self,
        mol: &'a Molecule,
    ) -> Result<BasisAtom<'a>, anyhow::Error> {
        let (atm_i, atm_s) = &self.atom;
        let atom = &mol
            .atoms
            .get(*atm_i)
            .ok_or(format_err!("Atom index {atm_i} not found."))?;
        ensure!(
            atom.atomic_symbol == *atm_s,
            "Mismatched element names: {} != {atm_s}.",
            atom.atomic_symbol
        );
        let bss = self
            .basis_shells
            .iter()
            .map(|inp_bs| inp_bs.to_basis_shell())
            .collect::<Vec<_>>();
        BasisAtom::builder()
            .atom(atom)
            .basis_shells(&bss)
            .build()
            .map_err(|err| format_err!(err))
    }
}

// ----------------------
// InputBasisAngularOrder
// ----------------------

/// A serialisable/deserialisable structure containing the angular momentum information of an
/// atomic-orbital basis set that is required for symmetry transformation to be performed.However,
/// unlike [`BasisAngularOrder`], this structure does not contain references to the atoms it is
/// describing. This is only for serialisation/deserialisation purposes.
///
/// The associated anonymous field is an ordered sequence of [`InputBasisAtom`] in the order the
/// atoms are defined in the molecule.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct InputBasisAngularOrder(pub(crate) Vec<InputBasisAtom>);

impl InputBasisAngularOrder {
    /// Returns the number of basis functions in this basis set.
    pub(crate) fn n_funcs(&self) -> usize {
        self.0.iter().map(InputBasisAtom::n_funcs).sum()
    }

    /// Converts to a [`BasisAngularOrder`] structure given a molecule.
    ///
    /// # Arguments
    ///
    /// * `mol` - A molecule to which the atoms in this [`InputBasisAngularOrder`] belong.
    ///
    /// # Returns
    ///
    /// The corresponding [`BasisAngularOrder`] structure.
    ///
    /// # Errors
    ///
    /// Errors if the atom indices and names in this [`InputBasisAngularOrder`] do not match
    /// those in `mol`.
    pub(crate) fn to_basis_angular_order<'a>(
        &self,
        mol: &'a Molecule,
    ) -> Result<BasisAngularOrder<'a>, anyhow::Error> {
        ensure!(
            mol.atoms.len() == self.0.len(),
            "Mismatched numbers of atoms: {} != {}.",
            mol.atoms.len(),
            self.0.len()
        );
        let basis_atoms = self
            .0
            .iter()
            .map(|batm| batm.to_basis_atom(mol))
            .collect::<Result<Vec<BasisAtom<'a>>, _>>()?;
        BasisAngularOrder::builder()
            .basis_atoms(&basis_atoms)
            .build()
            .map_err(|err| format_err!(err))
    }
}
