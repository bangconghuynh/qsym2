use serde::{Deserialize, Serialize};

use crate::drivers::representation_analysis::slater_determinant::SlaterDeterminantRepAnalysisParams;
use crate::drivers::symmetry_group_detection::SymmetryGroupDetectionParams;
#[allow(unused_imports)]
use crate::io::QSym2FileType;

/// An enumerated type representing possible input kinds for symmetry-group detection from a YAML
/// input file.
#[derive(Clone, Serialize, Deserialize)]
enum SymmetryGroupDetectionInputKind {
    /// Variant indicating that the parameters for the symmetry-group detection driver will be
    /// specified.
    Parameters(SymmetryGroupDetectionParams),

    /// Variant indicating that the symmetry-group detection results will be read in from a `QSym2`
    /// [`QSym2FileType:Sym`] binary file. The associated string gives the name of the file without
    /// its `.qsym2.sym` extension.
    FromFile(String),
}

impl Default for SymmetryGroupDetectionInputKind {
    fn default() -> Self {
        SymmetryGroupDetectionInputKind::Parameters(SymmetryGroupDetectionParams::default())
    }
}

/// A structure containing `QSym2` input parameters which can be serialised into and deserialised
/// from a YAML input file.
#[derive(Clone, Serialize, Deserialize)]
struct Input {
    /// Specification for symmetry-group detection. If `None`, no symmetry-group detection will be
    /// performed. If not `None`, then this either specifies the parameters for symmetry-group
    /// detection, or the name of a [`QSym2FileType:Sym`] binary file containing the symmetry-group
    /// detection results (without the `.qsym2.sym` extension).
    ///
    /// If not specified, this will be taken to be `None`.
    #[serde(default)]
    symmetry_group_detection: Option<SymmetryGroupDetectionInputKind>,

    /// Specification for Slater determinant representation analysis. If `None`, no Slater
    /// determinant representation analysis will be performed. If not `None`, then this specifies
    /// the parameters for Slater determinant representation analysis.
    ///
    /// # Default
    ///
    /// If not specified, this will be taken to be `None`.
    #[serde(default)]
    det_representation_analysis: Option<SlaterDeterminantRepAnalysisParams<f64>>,
}

impl Default for Input {
    fn default() -> Self {
        Input {
            symmetry_group_detection: Some(SymmetryGroupDetectionInputKind::default()),
            det_representation_analysis: Some(SlaterDeterminantRepAnalysisParams::<f64>::default()),
        }
    }
}

#[cfg(test)]
#[path = "input_tests.rs"]
mod input_tests;
