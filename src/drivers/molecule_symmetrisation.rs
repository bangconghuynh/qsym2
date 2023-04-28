use derive_builder::Builder;

// ==================
// Struct definitions
// ==================

// ----------
// Parameters
// ----------

/// A structure containing control parameters for symmetry-group detection.
#[derive(Clone, Builder, Debug)]
pub struct MoleculeSymmetrisationParams {
    /// Boolean indicating if a summetry of the located symmetry elements is to be written to the
    /// output file.
    use_magnetic_symmetry: bool,

    /// The maximum number of symmetrisation iterations.
    #[builder(default = "5")]
    max_iterations: usize,
}

impl MoleculeSymmetrisationParams {
    /// Returns a builder to construct a [`MoleculeSymmetrisationParams`] structure.
    pub fn builder() -> MoleculeSymmetrisationParamsBuilder {
        MoleculeSymmetrisationParamsBuilder::default()
    }
}
