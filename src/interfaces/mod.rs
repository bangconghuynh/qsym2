use anyhow;

pub mod binaries;
pub mod cli;
pub mod input;
#[cfg(feature = "qchem")]
pub mod qchem;

pub trait InputHandle {
    /// Handles the input section and runs appropriate calculations.
    fn handle(&self) -> Result<(), anyhow::Error>;
}
