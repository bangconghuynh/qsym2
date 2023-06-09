// use std::error::Error;
use std::fmt;

use anyhow;
use log;

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

/// A trait for logging $`\mathsf{QSym}^2`$ outputs nicely.
pub trait QSym2Output: fmt::Debug + fmt::Display {
    /// Logs display output nicely.
    fn log_output_display(&self) {
        let lines = self.to_string();
        lines.lines().for_each(|line| {
            log::info!(target: "output", "{line}");
        })
    }

    /// Logs debug output nicely.
    fn log_output_debug(&self) {
        let lines = format!("{self:?}");
        lines.lines().for_each(|line| {
            log::info!(target: "output", "{line}");
        })
    }
}

// Blanket implementation
impl<T> QSym2Output for T where T: fmt::Debug + fmt::Display {}
