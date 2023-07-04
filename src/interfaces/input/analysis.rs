use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::drivers::representation_analysis::slater_determinant::SlaterDeterminantRepAnalysisParams;
use crate::interfaces::custom::CustomSlaterDeterminantSource;
#[cfg(feature = "qchem")]
use crate::interfaces::qchem::QChemArchiveSlaterDeterminantSource;

// =======================================
// Representation analysis target controls
// =======================================

// -------------------------------------
// Representation analysis target choice
// -------------------------------------

/// A serialisable/deserialisable enumerated type representing possibilities of representation
/// analysis targets.
#[derive(Clone, Serialize, Deserialize)]
pub enum AnalysisTarget {
    /// Variant representing the choice of only performing a symmetry-group detection for a
    /// molecule. The associated value gives a path to an XYZ file containing the molecular
    /// structure for symmetry-group detection.
    MolecularSymmetry { xyz: PathBuf },

    /// Variant representing the choice of Slater determinant as the target for representation
    /// analysis. The associated structure contains the control parameters for this.
    SlaterDeterminant(SlaterDeterminantControl),
}

impl Default for AnalysisTarget {
    fn default() -> Self {
        AnalysisTarget::SlaterDeterminant(SlaterDeterminantControl::default())
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~
// Target: Slater determinant
// ~~~~~~~~~~~~~~~~~~~~~~~~~~

/// A serialisable/deserialisable structure containing control parameters for Slater determinant
/// representation analysis.
#[derive(Clone, Serialize, Deserialize, Default)]
pub struct SlaterDeterminantControl {
    /// The source of Slater determinant(s).
    pub source: SlaterDeterminantSource,

    /// The parameters for representation analysis.
    pub control: SlaterDeterminantRepAnalysisParams<f64>,
}

/// A serialisable/deserialisable enumerated type representing possibilities of Slater determinant
/// sources.
#[derive(Clone, Serialize, Deserialize)]
pub enum SlaterDeterminantSource {
    /// Slater determinant from Q-Chem HDF5 archive. This is only available when the `qchem`
    /// feature is enabled.
    #[cfg(feature = "qchem")]
    QChemArchive(QChemArchiveSlaterDeterminantSource),

    /// Slater determinant from a custom specification.
    Custom(CustomSlaterDeterminantSource),
}

impl Default for SlaterDeterminantSource {
    fn default() -> Self {
        SlaterDeterminantSource::Custom(CustomSlaterDeterminantSource::default())
    }
}
