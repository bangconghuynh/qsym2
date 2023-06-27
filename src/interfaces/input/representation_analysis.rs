use serde::{Deserialize, Serialize};

use crate::drivers::representation_analysis::slater_determinant::SlaterDeterminantRepAnalysisParams;
use crate::interfaces::input::ao_basis::InputBasisAngularOrder;

// =======================================
// Representation analysis target controls
// =======================================

// -------------------------------------
// Representation analysis target choice
// -------------------------------------

/// A serialisable/deserialisable enumerated type representing possibilities of representation
/// analysis targets.
#[derive(Clone, Serialize, Deserialize)]
pub(super) enum RepAnalysisTarget {
    /// Variant representing the choice of Slater determinant as the target for representation
    /// analysis. The associated structure contains the control parameters for this.
    SlaterDeterminant(SlaterDeterminantControl),
}

impl Default for RepAnalysisTarget {
    fn default() -> Self {
        RepAnalysisTarget::SlaterDeterminant(SlaterDeterminantControl::default())
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~
// Target: Slater determinant
// ~~~~~~~~~~~~~~~~~~~~~~~~~~

/// A serialisable/deserialisable structure containing control parameters for Slater determinant
/// representation analysis.
#[derive(Clone, Serialize, Deserialize, Default)]
pub(super) struct SlaterDeterminantControl {
    /// The source of Slater determinant(s).
    pub(super) source: SlaterDeterminantSource,

    /// The parameters for representation analysis.
    pub(super) control: SlaterDeterminantRepAnalysisParams<f64>,
}

/// A serialisable/deserialisable enumerated type representing possibilities of Slater determinant
/// sources.
#[derive(Clone, Serialize, Deserialize)]
pub(super) enum SlaterDeterminantSource {
    /// Slater determinant from Q-Chem scratch directory.
    QChemScratch(QChemScratchSlaterDeterminantSource),
}

impl Default for SlaterDeterminantSource {
    fn default() -> Self {
        SlaterDeterminantSource::QChemScratch(QChemScratchSlaterDeterminantSource::default())
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Target: Slater determinant; source: Q-Chem scratch
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// A serialisable/deserialisable structure containing control parameters for acquiring Slater
/// determinant(s) from a Q-Chem scratch directory.
#[derive(Clone, Serialize, Deserialize)]
pub(super) struct QChemScratchSlaterDeterminantSource {
    /// The path to the Q-Chem scratch directory.
    pub(super) scratch_path: String,

    /// The path to the Q-Chem `.fchk` file.
    pub(super) fchk_path: String,

    /// An optional specification for the basis angular order information. If `None`, this will be
    /// deduced automatically from any basis set information found in the specified Q-Chem scratch
    /// directory.
    pub(super) bao: Option<InputBasisAngularOrder>,
}

impl Default for QChemScratchSlaterDeterminantSource {
    fn default() -> Self {
        QChemScratchSlaterDeterminantSource {
            scratch_path: "path/to/qchem/job/scratch/directory".to_string(),
            fchk_path: "path/to/qchem/job/fchk/file".to_string(),
            bao: None,
        }
    }
}
