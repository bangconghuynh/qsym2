use std::path::PathBuf;

use anyhow::{self, bail};
use serde::{Deserialize, Serialize};

use crate::drivers::representation_analysis::angular_function::AngularFunctionRepAnalysisParams;
use crate::drivers::symmetry_group_detection::{
    SymmetryGroupDetectionDriver, SymmetryGroupDetectionParams,
};
use crate::drivers::QSym2Driver;
#[cfg(feature = "qchem")]
use crate::interfaces::qchem::hdf5::QChemH5Driver;
#[allow(unused_imports)]
use crate::io::QSym2FileType;

use analysis::{AnalysisTarget, SlaterDeterminantSource};

pub mod analysis;
pub mod ao_basis;

#[cfg(test)]
#[path = "input_tests.rs"]
mod input_tests;

// ===============
// Driver controls
// ===============

// -------------------------------
// SymmetryGroupDetectionInputKind
// -------------------------------

/// An enumerated type representing possible input kinds for symmetry-group detection from a YAML
/// input file.
#[derive(Clone, Serialize, Deserialize)]
pub enum SymmetryGroupDetectionInputKind {
    /// Variant indicating that the parameters for the symmetry-group detection driver will be
    /// specified.
    Parameters(SymmetryGroupDetectionParams),

    /// Variant indicating that the symmetry-group detection results will be read in from a `QSym2`
    /// [`QSym2FileType:Sym`] binary file. The associated string gives the name of the file without
    /// its `.qsym2.sym` extension.
    FromFile(PathBuf),
}

impl Default for SymmetryGroupDetectionInputKind {
    fn default() -> Self {
        SymmetryGroupDetectionInputKind::Parameters(SymmetryGroupDetectionParams::default())
    }
}

// ==========
// Main input
// ==========

/// A structure containing `QSym2` input parameters which can be serialised into and deserialised
/// from a YAML input file.
#[derive(Clone, Serialize, Deserialize)]
pub struct Input {
    /// Specification for symmetry-group detection. This either specifies the parameters for
    /// symmetry-group detection, or the name of a [`QSym2FileType:Sym`] binary file containing the
    /// symmetry-group detection results (without the `.qsym2.sym` extension).
    pub symmetry_group_detection: SymmetryGroupDetectionInputKind,

    /// Specification for analysis target.
    pub analysis_target: AnalysisTarget,
}

impl Input {
    pub fn handle(&self) -> Result<(), anyhow::Error> {
        let pd_params_inp = &self.symmetry_group_detection;
        let mut afa_params = AngularFunctionRepAnalysisParams::default();
        match &self.analysis_target {
            AnalysisTarget::SlaterDeterminant(sd_control) => {
                let sd_source = &sd_control.source;
                let sda_params = &sd_control.control;
                afa_params.linear_independence_threshold = sda_params.linear_independence_threshold;
                afa_params.integrality_threshold = sda_params.integrality_threshold;
                match sd_source {
                    #[cfg(feature = "qchem")]
                    SlaterDeterminantSource::QChemArchive(qchemarchive_sd_source) => {
                        let qchemarchive_path = &qchemarchive_sd_source.path;
                        let mut qchem_h5_driver = QChemH5Driver::<f64>::builder()
                            .filename(qchemarchive_path.into())
                            .symmetry_group_detection_input(&pd_params_inp)
                            .angular_function_analysis_parameters(&afa_params)
                            .slater_det_rep_analysis_parameters(&sda_params)
                            .build()?;
                        qchem_h5_driver.run()
                    }
                    SlaterDeterminantSource::Custom(_) => Ok(()),
                }
            }
            AnalysisTarget::MolecularSymmetry { xyz } => {
                let pd_params = match pd_params_inp {
                    SymmetryGroupDetectionInputKind::Parameters(pd_params) => pd_params,
                    SymmetryGroupDetectionInputKind::FromFile(_) => {
                        bail!(
                            "It is pointless to provide a pre-calculated symmetry-group \
                            detection result when only symmetry-group detection is required."
                        )
                    }
                };
                let mut pd_driver = SymmetryGroupDetectionDriver::builder()
                    .parameters(pd_params)
                    .xyz(Some(xyz.into()))
                    .build()?;
                pd_driver.run()
            }
        }
    }
}

impl Default for Input {
    fn default() -> Self {
        Input {
            symmetry_group_detection: SymmetryGroupDetectionInputKind::default(),
            analysis_target: AnalysisTarget::default(),
        }
    }
}
