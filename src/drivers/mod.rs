use anyhow;

pub mod representation_analysis;
pub mod symmetry_group_detection;
pub mod molecule_symmetrisation;

// =================
// Trait definitions
// =================

/// A trait for $`\mathsf{QSym}^2`$ drivers.
pub trait QSym2Driver {
    /// The type of the successful outcome when executing the driver.
    type Outcome;

    /// Executes the driver and stores the result internally.
    fn run(&mut self) -> Result<(), anyhow::Error>;

    /// Returns the result of the driver execution.
    fn result(&self) -> Result<&Self::Outcome, anyhow::Error>;
}
