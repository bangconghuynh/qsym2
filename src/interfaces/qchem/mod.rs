use serde::{Serialize, Deserialize};

mod hdf5;

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Input target: Slater determinant; source: Q-Chem archive
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// A serialisable/deserialisable structure containing control parameters for acquiring Slater
/// determinant(s) from a Q-Chem archive file.
#[derive(Clone, Serialize, Deserialize)]
pub struct QChemArchiveSlaterDeterminantSource {
    /// The path to the Q-Chem HDF5 archive file (`qarchive.h5`).
    pub(super) path: String,
}

impl Default for QChemArchiveSlaterDeterminantSource {
    fn default() -> Self {
        QChemArchiveSlaterDeterminantSource {
            path: "path/to/qchem/qarchive.h5".to_string(),
        }
    }
}
