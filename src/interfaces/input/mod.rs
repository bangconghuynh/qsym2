use std::path::PathBuf;

use anyhow::{self, bail, Context};
use log;
use serde::{Deserialize, Serialize};

use crate::drivers::molecule_symmetrisation::MoleculeSymmetrisationDriver;
use crate::drivers::representation_analysis::angular_function::AngularFunctionRepAnalysisParams;
use crate::drivers::symmetry_group_detection::{
    SymmetryGroupDetectionDriver, SymmetryGroupDetectionParams,
};
use crate::drivers::QSym2Driver;
use crate::interfaces::input::analysis::{
    AnalysisTarget, SlaterDeterminantSource, SlaterDeterminantSourceHandle,
    VibrationalCoordinateSource, VibrationalCoordinateSourceHandle,
};
use crate::interfaces::InputHandle;
#[allow(unused_imports)]
use crate::io::{read_qsym2_binary, QSym2FileType};

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
    /// [`QSym2FileType::Sym`] binary file. The associated string gives the name of the file without
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
    /// symmetry-group detection, or the name of a [`QSym2FileType::Sym`] binary file containing the
    /// symmetry-group detection results (without the `.qsym2.sym` extension).
    pub symmetry_group_detection: SymmetryGroupDetectionInputKind,

    /// Specification for analysis target.
    pub analysis_target: AnalysisTarget,
}

impl InputHandle for Input {
    /// Handles the main input structure.
    fn handle(&self) -> Result<(), anyhow::Error> {
        let pd_params_inp = &self.symmetry_group_detection;
        let mut afa_params = AngularFunctionRepAnalysisParams::default();
        match &self.analysis_target {
            AnalysisTarget::MolecularSymmetry {
                xyz,
                symmetrisation,
            } => {
                log::debug!("Analysis target: Molecular symmetry");
                let pd_params = match pd_params_inp {
                    SymmetryGroupDetectionInputKind::Parameters(pd_params) => pd_params,
                    SymmetryGroupDetectionInputKind::FromFile(_) => {
                        bail!(
                            "It is pointless to provide a pre-calculated symmetry-group \
                            detection result when only symmetry-group detection is required."
                        )
                    }
                };
                log::debug!(
                    "Molecular symmetry group will be identified based on specified parameters."
                );
                let mut pd_driver = SymmetryGroupDetectionDriver::builder()
                    .parameters(pd_params)
                    .xyz(Some(xyz.into()))
                    .build()
                    .with_context(|| "Unable to construct a symmetry-group detection driver while handling molecular symmetry analysis target")?;
                let pd_res = pd_driver
                    .run()
                    .with_context(|| "Unable to execute the symmetry-group detection driver successfully while handling molecular symmetry analysis target");
                if let Some(ms_params) = symmetrisation.as_ref() {
                    log::debug!("Performing molecule symmetrisation...");
                    let pd_res = pd_driver
                        .result()
                        .with_context(|| "Unable to extract the target symmetry-group detection result for molecule symmetrisation while handling molecular symmetry analysis target")?;
                    let mut ms_driver = MoleculeSymmetrisationDriver::builder()
                        .parameters(ms_params)
                        .target_symmetry_result(pd_res)
                        .build()
                        .with_context(|| "Unable to construct a molecule symmetrisation driver while handling molecular symmetry analysis target")?;
                    let ms_res = ms_driver
                        .run()
                        .with_context(|| "Unable to execute the molecule symmetrisation driver successfully while handling molecular symmetry analysis target");
                    log::debug!("Performing molecule symmetrisation... Done.");
                    ms_res
                } else {
                    pd_res
                }
            }
            AnalysisTarget::SlaterDeterminant(sd_control) => {
                log::debug!("Analysis target: Slater determinant");
                let sd_source = &sd_control.source;
                let sda_params = &sd_control.control;
                afa_params.linear_independence_threshold = sda_params.linear_independence_threshold;
                afa_params.integrality_threshold = sda_params.integrality_threshold;
                match sd_source {
                    #[cfg(feature = "qchem")]
                    SlaterDeterminantSource::QChemArchive(qchemarchive_sd_source) => {
                        log::debug!("Slater determinant source: Q-Chem archive");
                        qchemarchive_sd_source
                            .sd_source_handle(&pd_params_inp, &afa_params, &sda_params)
                            .map(|_| ())
                    }
                    SlaterDeterminantSource::Binaries(binaries_sd_source) => {
                        log::debug!("Slater determinant source: binary files");
                        binaries_sd_source
                            .sd_source_handle(&pd_params_inp, &afa_params, &sda_params)
                            .map(|_| ())
                    }
                }
            }
            AnalysisTarget::VibrationalCoordinates(vc_control) => {
                log::debug!("Analysis target: vibrational coordinates");
                let vc_source = &vc_control.source;
                let vca_params = &vc_control.control;
                afa_params.linear_independence_threshold = vca_params.linear_independence_threshold;
                afa_params.integrality_threshold = vca_params.integrality_threshold;
                match vc_source {
                    #[cfg(feature = "qchem")]
                    VibrationalCoordinateSource::QChemArchive(qchemarchive_vc_source) => {
                        log::debug!("Vibrational coordinate source: Q-Chem archive");
                        qchemarchive_vc_source
                            .vc_source_handle(&pd_params_inp, &afa_params, &vca_params)
                            .map(|_| ())
                    }
                }
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
