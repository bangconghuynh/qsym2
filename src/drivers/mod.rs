//! Drivers to carry out QSym² functionalities.

use anyhow;

pub mod molecule_symmetrisation;
pub mod molecule_sprucing;
pub mod representation_analysis;
pub mod symmetry_group_detection;

// =================
// Trait definitions
// =================

/// Trait defining behaviours of `QSym2` drivers.
pub trait QSym2Driver {
    /// The type of the parameter structure controlling the driver.
    type Params;

    /// The type of the successful outcome when executing the driver.
    type Outcome;

    /// Executes the driver and stores the result internally.
    fn run(&mut self) -> Result<(), anyhow::Error>;

    /// Returns the result of the driver execution.
    fn result(&self) -> Result<&Self::Outcome, anyhow::Error>;
}
