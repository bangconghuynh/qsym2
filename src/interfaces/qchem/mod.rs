//! QSym² interfaces with Q-Chem.

use std::path::PathBuf;

use anyhow::Context;
use serde::{Deserialize, Serialize};

use crate::drivers::representation_analysis::angular_function::AngularFunctionRepAnalysisParams;
use crate::drivers::representation_analysis::slater_determinant::SlaterDeterminantRepAnalysisParams;
use crate::drivers::QSym2Driver;
use crate::drivers::representation_analysis::vibrational_coordinate::VibrationalCoordinateRepAnalysisParams;
use crate::interfaces::input::analysis::{
    SlaterDeterminantSourceHandle, VibrationalCoordinateSourceHandle,
};
use crate::interfaces::input::SymmetryGroupDetectionInputKind;
use crate::interfaces::qchem::hdf5::slater_determinant::QChemSlaterDeterminantH5Driver;
use crate::interfaces::qchem::hdf5::vibrational_coordinate::QChemVibrationH5Driver;

pub(crate) mod hdf5;

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Input target: Slater determinant; source: Q-Chem archive
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// A serialisable/deserialisable structure containing control parameters for acquiring Slater
/// determinant(s) from a Q-Chem archive file.
#[derive(Clone, Serialize, Deserialize)]
pub struct QChemArchiveSlaterDeterminantSource {
    /// The path to the Q-Chem HDF5 archive file (`qarchive.h5`).
    pub path: PathBuf,
}

impl Default for QChemArchiveSlaterDeterminantSource {
    fn default() -> Self {
        QChemArchiveSlaterDeterminantSource {
            path: PathBuf::from("path/to/qchem/qarchive.h5"),
        }
    }
}

impl SlaterDeterminantSourceHandle for QChemArchiveSlaterDeterminantSource {
    type Outcome = Vec<(String, String)>;

    fn sd_source_handle(
        &self,
        pd_params_inp: &SymmetryGroupDetectionInputKind,
        afa_params: &AngularFunctionRepAnalysisParams,
        sda_params: &SlaterDeterminantRepAnalysisParams<f64>,
    ) -> Result<Self::Outcome, anyhow::Error> {
        let qchemarchive_path = &self.path;
        let mut qchem_h5_driver = QChemSlaterDeterminantH5Driver::builder()
            .filename(qchemarchive_path.into())
            .symmetry_group_detection_input(&pd_params_inp)
            .angular_function_analysis_parameters(&afa_params)
            .rep_analysis_parameters(&sda_params)
            .build()
            .with_context(|| "Unable to construct a Q-Chem HDF5 driver when handling Q-Chem archive Slater determinant source")?;
        qchem_h5_driver.run().with_context(|| "Unable to execute the Q-Chem HDF5 driver successfully when handling Q-Chem archive Slater determinant source")?;
        qchem_h5_driver.result().cloned()
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Input target: vibrational coordinates; source: Q-Chem archive
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// A serialisable/deserialisable structure containing control parameters for acquiring vibrational
/// coordinate(s) from a Q-Chem archive file.
#[derive(Clone, Serialize, Deserialize)]
pub struct QChemArchiveVibrationalCoordinateSource {
    /// The path to the Q-Chem HDF5 archive file (`qarchive.h5`).
    pub path: PathBuf,
}

impl Default for QChemArchiveVibrationalCoordinateSource {
    fn default() -> Self {
        QChemArchiveVibrationalCoordinateSource {
            path: PathBuf::from("path/to/qchem/qarchive.h5"),
        }
    }
}

impl VibrationalCoordinateSourceHandle for QChemArchiveVibrationalCoordinateSource {
    type Outcome = Vec<(String, String)>;

    fn vc_source_handle(
        &self,
        pd_params_inp: &SymmetryGroupDetectionInputKind,
        afa_params: &AngularFunctionRepAnalysisParams,
        vca_params: &VibrationalCoordinateRepAnalysisParams<f64>,
    ) -> Result<Self::Outcome, anyhow::Error> {
        let qchemarchive_path = &self.path;
        let mut qchem_h5_driver = QChemVibrationH5Driver::builder()
            .filename(qchemarchive_path.into())
            .symmetry_group_detection_input(&pd_params_inp)
            .angular_function_analysis_parameters(&afa_params)
            .rep_analysis_parameters(&vca_params)
            .build()
            .with_context(|| "Unable to construct a Q-Chem HDF5 driver when handling Q-Chem archive vibrational coordinate source")?;
        qchem_h5_driver.run().with_context(|| "Unable to execute the Q-Chem HDF5 driver successfully when handling Q-Chem archive vibrational coordinate source")?;
        qchem_h5_driver.result().cloned()
    }
}
