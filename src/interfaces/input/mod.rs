use serde::{Deserialize, Serialize};

use crate::drivers::representation_analysis::slater_determinant::SlaterDeterminantRepAnalysisParams;
use crate::drivers::symmetry_group_detection::SymmetryGroupDetectionParams;

#[derive(Clone, Serialize, Deserialize, Default)]
struct Input {
    #[serde(default)]
    symmetry_group_detection: SymmetryGroupDetectionParams,

    #[serde(default)]
    representation_analysis: SlaterDeterminantRepAnalysisParams<f64>,
}

#[cfg(test)]
#[path = "input_tests.rs"]
mod input_tests;
