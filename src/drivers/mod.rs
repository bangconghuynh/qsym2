use anyhow;

pub mod molecule_symmetrisation;
pub mod representation_analysis;
pub mod symmetry_group_detection;

// =================
// Trait definitions
// =================

/// A trait for `QSym2` drivers.
pub trait QSym2Driver {
    /// The type of the successful outcome when executing the driver.
    type Outcome;

    /// Executes the driver and stores the result internally.
    fn run(&mut self) -> Result<(), anyhow::Error>;

    /// Returns the result of the driver execution.
    fn result(&self) -> Result<&Self::Outcome, anyhow::Error>;
}
