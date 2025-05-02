//! Human-readable specification of atomic-orbital basis information in QSym² input configuration.

use anyhow::{self, ensure, format_err};
use derive_builder::Builder;
use serde::{Deserialize, Serialize};

use crate::auxiliary::molecule::Molecule;
use crate::basis::ao::*;

// ---------------
// InputShellOrder
// ---------------

/// Serialisable/deserialisable enumerated type to indicate the type of the angular functions in
/// a shell and how they are ordered.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum InputShellOrder {
    /// This variant indicates that the angular functions are real solid harmonics arranged in
    /// increasing $`m`$ order.
    PureIncreasingm,

    /// This variant indicates that the angular functions are real solid harmonics arranged in
    /// decreasing $`m`$ order.
    PureDecreasingm,

    /// This variant indicates that the angular functions are real solid harmonics arranged in
    /// a custom order specified by the $`m_l`$ values.
    PureCustom(Vec<i32>),

    /// This variant indicates that the angular functions are Cartesian functions arranged in
    /// lexicographic order.
    CartLexicographic,

    /// This variant indicates that the angular functions are Cartesian functions arranged in
    /// Q-Chem order.
    CartQChem,

    /// This variant indicates that the angular functions are Cartesian functions arranged in
    /// a custom order specified by the ordered exponent tuples.
    CartCustom(Vec<(u32, u32, u32)>),

    /// This variant indicates that the angular functions are spinors arranged in increasing $`m`$
    /// order. The associated boolean indicates whether the spinors are even with respect to
    /// spatial inversion.
    SpinorIncreasingm(bool),

    /// This variant indicates that the angular functions are spinors arranged in decreasing $`m`$
    /// order. The associated boolean indicates whether the spinors are even with respect to
    /// spatial inversion.
    SpinorDecreasingm(bool),

    /// This variant indicates that the angular functions are spinors arranged in a custom order
    /// specified by the $`m_l`$ values.
    SpinorCustom(bool, Vec<i32>),
}

impl InputShellOrder {
    /// Converts the [`InputShellOrder`] to a corresponding [`ShellOrder`].
    pub fn to_shell_order(&self, l: u32) -> ShellOrder {
        match self {
            InputShellOrder::PureIncreasingm => ShellOrder::Pure(PureOrder::increasingm(l)),
            InputShellOrder::PureDecreasingm => ShellOrder::Pure(PureOrder::decreasingm(l)),
            InputShellOrder::PureCustom(mls) => {
                ShellOrder::Pure(PureOrder::new(mls).expect("Invalid ml sequence specified."))
            }
            InputShellOrder::CartLexicographic => ShellOrder::Cart(CartOrder::lex(l)),
            InputShellOrder::CartQChem => ShellOrder::Cart(CartOrder::qchem(l)),
            InputShellOrder::CartCustom(cart_tuples) => ShellOrder::Cart(
                CartOrder::new(cart_tuples).expect("Invalid Cartesian tuples provided."),
            ),
            InputShellOrder::SpinorIncreasingm(even) => {
                ShellOrder::Spinor(SpinorOrder::increasingm(l, *even))
            }
            InputShellOrder::SpinorDecreasingm(even) => {
                ShellOrder::Spinor(SpinorOrder::decreasingm(l, *even))
            }
            InputShellOrder::SpinorCustom(even, mls) => ShellOrder::Spinor(
                SpinorOrder::new(mls, *even).expect("Invalid 2mj sequence specified."),
            ),
        }
    }
}

// ---------------
// InputBasisShell
// ---------------

/// Serialisable/deserialisable structure representing a shell in an atomic-orbital basis set.
#[derive(Clone, Debug, Builder, Serialize, Deserialize)]
pub struct InputBasisShell {
    /// A non-negative integer indicating the rank of the shell.
    pub l: u32,

    /// An enum indicating the type of the angular functions in a shell and how they are ordered.
    pub shell_order: InputShellOrder,
}

impl InputBasisShell {
    /// Returns a builder to construct [`InputBasisShell`].
    pub fn builder() -> InputBasisShellBuilder {
        InputBasisShellBuilder::default()
    }

    /// Returns the number of basis functions in this shell.
    pub fn n_funcs(&self) -> usize {
        let lsize = self.l as usize;
        match self.shell_order {
            InputShellOrder::PureIncreasingm
            | InputShellOrder::PureDecreasingm
            | InputShellOrder::PureCustom(_) => 2 * lsize + 1, // lsize = l
            InputShellOrder::CartQChem
            | InputShellOrder::CartLexicographic
            | InputShellOrder::CartCustom(_) => ((lsize + 1) * (lsize + 2)).div_euclid(2),
            InputShellOrder::SpinorIncreasingm(_)
            | InputShellOrder::SpinorDecreasingm(_)
            | InputShellOrder::SpinorCustom(_, _) => lsize + 1, // lsize = 2j
        }
    }

    /// Converts the [`InputBasisShell`] to a corresponding [`BasisShell`].
    pub fn to_basis_shell(&self) -> BasisShell {
        BasisShell::new(self.l, self.shell_order.to_shell_order(self.l))
    }
}

// --------------
// InputBasisAtom
// --------------

/// Serialisable/deserialisable structure containing the ordered sequence of the shells for an
/// atom. However, unlike [`BasisAtom`], this structure does not contain a reference to the atom it
/// is describing, but instead it only contains an index and an owned string giving the element
/// name of the atom. This is only for serialisation/deserialisation purposes.
#[derive(Clone, Debug, Builder, Serialize, Deserialize)]
pub struct InputBasisAtom {
    /// The index and name of an atom in the basis set.
    pub atom: (usize, String),

    /// The ordered shells associated with this atom.
    pub basis_shells: Vec<InputBasisShell>,
}

impl InputBasisAtom {
    /// Returns a builder to construct [`InputBasisAtom`].
    pub fn builder() -> InputBasisAtomBuilder {
        InputBasisAtomBuilder::default()
    }

    /// Returns the number of basis functions localised on this atom.
    pub fn n_funcs(&self) -> usize {
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
    pub fn to_basis_atom<'a>(&self, mol: &'a Molecule) -> Result<BasisAtom<'a>, anyhow::Error> {
        let (atm_i, atm_s) = &self.atom;
        let atom = &mol
            .atoms
            .get(*atm_i)
            .ok_or(format_err!("Atom index {atm_i} not found."))?;
        ensure!(
            atom.atomic_symbol == *atm_s,
            "Mismatched element names: {} (expected) != {atm_s} (specified).",
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

/// Serialisable/deserialisable structure containing the angular momentum information of an
/// atomic-orbital basis set that is required for symmetry transformation to be performed.However,
/// unlike [`BasisAngularOrder`], this structure does not contain references to the atoms it is
/// describing. This is only for serialisation/deserialisation purposes.
///
/// The associated anonymous field is an ordered sequence of [`InputBasisAtom`] in the order the
/// atoms are defined in the molecule.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InputBasisAngularOrder(pub Vec<InputBasisAtom>);

impl InputBasisAngularOrder {
    /// Returns the number of basis functions in this basis set.
    pub fn n_funcs(&self) -> usize {
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
    pub fn to_basis_angular_order<'a>(
        &self,
        mol: &'a Molecule,
    ) -> Result<BasisAngularOrder<'a>, anyhow::Error> {
        ensure!(
            mol.atoms.len() == self.0.len(),
            "Mismatched numbers of atoms: {} (expected) != {} (specified).",
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

impl Default for InputBasisAngularOrder {
    fn default() -> Self {
        Self(vec![
            InputBasisAtom::builder()
                .atom((0, "H".to_string()))
                .basis_shells(vec![
                    InputBasisShell::builder()
                        .l(0)
                        .shell_order(InputShellOrder::PureIncreasingm)
                        .build()
                        .expect("Unable to construct a default input basis shell."),
                    InputBasisShell::builder()
                        .l(1)
                        .shell_order(InputShellOrder::PureDecreasingm)
                        .build()
                        .expect("Unable to construct a default input basis shell."),
                    InputBasisShell::builder()
                        .l(2)
                        .shell_order(InputShellOrder::PureCustom(vec![0, 1, -1, 2, -2]))
                        .build()
                        .expect("Unable to construct a default input basis shell."),
                ])
                .build()
                .expect("Unable to construct a default input basis atom."),
            InputBasisAtom::builder()
                .atom((1, "O".to_string()))
                .basis_shells(vec![
                    InputBasisShell::builder()
                        .l(1)
                        .shell_order(InputShellOrder::CartCustom(vec![
                            (0, 1, 0),
                            (1, 0, 0),
                            (0, 0, 1),
                        ]))
                        .build()
                        .expect("Unable to construct a default input basis shell."),
                    InputBasisShell::builder()
                        .l(2)
                        .shell_order(InputShellOrder::CartQChem)
                        .build()
                        .expect("Unable to construct a default input basis shell."),
                    InputBasisShell::builder()
                        .l(3)
                        .shell_order(InputShellOrder::CartLexicographic)
                        .build()
                        .expect("Unable to construct a default input basis shell."),
                ])
                .build()
                .expect("Unable to construct a default input basis atom."),
        ])
    }
}
