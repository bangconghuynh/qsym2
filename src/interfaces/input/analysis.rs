//! Symmetry analysis target controls from QSym² input configuration.

use std::path::PathBuf;

use anyhow;
use serde::{Deserialize, Serialize};

use crate::drivers::molecule_symmetrisation_bootstrap::MoleculeSymmetrisationBootstrapParams;
use crate::drivers::representation_analysis::angular_function::AngularFunctionRepAnalysisParams;
use crate::drivers::representation_analysis::slater_determinant::SlaterDeterminantRepAnalysisParams;
#[cfg(feature = "qchem")]
use crate::drivers::representation_analysis::vibrational_coordinate::VibrationalCoordinateRepAnalysisParams;
use crate::interfaces::binaries::BinariesSlaterDeterminantSource;
use crate::interfaces::input::SymmetryGroupDetectionInputKind;
#[cfg(feature = "qchem")]
use crate::interfaces::qchem::{
    QChemArchiveSlaterDeterminantSource, QChemArchiveVibrationalCoordinateSource,
};

// =======================================
// Representation analysis target controls
// =======================================

// -------------------------------------
// Representation analysis target choice
// -------------------------------------

/// Serialisable/deserialisable enumerated type representing possibilities of representation
/// analysis targets.
#[derive(Clone, Serialize, Deserialize)]
pub enum AnalysisTarget {
    /// Variant representing the choice of only performing a symmetry-group detection for a
    /// molecule.
    MolecularSymmetry {
        /// Path to an XYZ file containing the molecular structure for symmetry-group detection.
        xyz: PathBuf,

        /// Optional parameters for performing symmetrisation on the structure.
        symmetrisation: Option<MoleculeSymmetrisationBootstrapParams>,
    },

    /// Variant representing the choice of Slater determinant as the target for representation
    /// analysis. The associated structure contains the control parameters for this.
    SlaterDeterminant(SlaterDeterminantControl),

    /// Variant representing the choice of vibrational coordinates as the target for representation
    /// analysis. The associated structure contains the control parameters for this.
    #[cfg(feature = "qchem")]
    VibrationalCoordinates(VibrationalCoordinateControl),
}

impl AnalysisTarget {
    /// Returns a vector containing of all possible analysis targets populated with their default
    /// settings.
    pub(crate) fn all_default() -> Vec<Self> {
        vec![
            AnalysisTarget::MolecularSymmetry {
                xyz: PathBuf::from("path/to/xyz"),
                symmetrisation: Some(MoleculeSymmetrisationBootstrapParams::default()),
            },
            AnalysisTarget::SlaterDeterminant(SlaterDeterminantControl::default()),
            #[cfg(feature = "qchem")]
            AnalysisTarget::VibrationalCoordinates(VibrationalCoordinateControl::default()),
        ]
    }
}

impl Default for AnalysisTarget {
    fn default() -> Self {
        AnalysisTarget::SlaterDeterminant(SlaterDeterminantControl::default())
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~
// Target: Slater determinant
// ~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Trait for the handling of vibrational coordinate input sources.
pub(crate) trait SlaterDeterminantSourceHandle {
    type Outcome;

    /// Handles the Slater determinant input source and runs relevant calculations.
    fn sd_source_handle(
        &self,
        pd_params_inp: &SymmetryGroupDetectionInputKind,
        afa_params: &AngularFunctionRepAnalysisParams,
        sda_params: &SlaterDeterminantRepAnalysisParams<f64>,
    ) -> Result<Self::Outcome, anyhow::Error>;
}

/// Serialisable/deserialisable structure containing control parameters for Slater determinant
/// representation analysis.
#[derive(Clone, Serialize, Deserialize, Default)]
pub struct SlaterDeterminantControl {
    /// The source of Slater determinant(s).
    pub source: SlaterDeterminantSource,

    /// The parameters for representation analysis.
    pub control: SlaterDeterminantRepAnalysisParams<f64>,
}

/// Serialisable/deserialisable enumerated type representing possibilities of Slater determinant
/// sources.
#[derive(Clone, Serialize, Deserialize)]
pub enum SlaterDeterminantSource {
    /// Slater determinant from Q-Chem HDF5 archive. This is only available when the `qchem`
    /// feature is enabled.
    #[cfg(feature = "qchem")]
    QChemArchive(QChemArchiveSlaterDeterminantSource),

    /// Slater determinant from a binaries specification.
    Binaries(BinariesSlaterDeterminantSource),
}

#[cfg(feature = "qchem")]
impl Default for SlaterDeterminantSource {
    fn default() -> Self {
        SlaterDeterminantSource::QChemArchive(QChemArchiveSlaterDeterminantSource::default())
    }
}

#[cfg(not(feature = "qchem"))]
impl Default for SlaterDeterminantSource {
    fn default() -> Self {
        SlaterDeterminantSource::Binaries(BinariesSlaterDeterminantSource::default())
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Target: Vibrational coordinates
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Trait for the handling of vibrational coordinate input sources.
#[cfg(feature = "qchem")]
pub(crate) trait VibrationalCoordinateSourceHandle {
    type Outcome;

    /// Handles the vibrational coordinate input source and runs relevant calculations.
    fn vc_source_handle(
        &self,
        pd_params_inp: &SymmetryGroupDetectionInputKind,
        afa_params: &AngularFunctionRepAnalysisParams,
        vca_params: &VibrationalCoordinateRepAnalysisParams<f64>,
    ) -> Result<Self::Outcome, anyhow::Error>;
}

/// Serialisable/deserialisable structure containing control parameters for vibrational
/// coordinate representation analysis.
#[cfg(feature = "qchem")]
#[derive(Clone, Serialize, Deserialize, Default)]
pub struct VibrationalCoordinateControl {
    /// The source of vibrational coordinate(s).
    pub source: VibrationalCoordinateSource,

    /// The parameters for representation analysis.
    pub control: VibrationalCoordinateRepAnalysisParams<f64>,
}

/// Serialisable/deserialisable enumerated type representing possibilities of vibrational
/// coordinates sources.
#[cfg(feature = "qchem")]
#[derive(Clone, Serialize, Deserialize)]
pub enum VibrationalCoordinateSource {
    /// Vibrational coordinates from Q-Chem HDF5 archive. This is only available when the `qchem`
    /// feature is enabled.
    QChemArchive(QChemArchiveVibrationalCoordinateSource),
}

#[cfg(feature = "qchem")]
impl Default for VibrationalCoordinateSource {
    fn default() -> Self {
        VibrationalCoordinateSource::QChemArchive(QChemArchiveVibrationalCoordinateSource::default())
    }
}
