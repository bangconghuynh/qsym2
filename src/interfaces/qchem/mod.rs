use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::drivers::QSym2Driver;
use crate::drivers::representation_analysis::angular_function::AngularFunctionRepAnalysisParams;
use crate::drivers::representation_analysis::slater_determinant::SlaterDeterminantRepAnalysisParams;
use crate::interfaces::input::analysis::SlaterDeterminantSourceHandle;
use crate::interfaces::input::SymmetryGroupDetectionInputKind;
use crate::interfaces::qchem::hdf5::QChemH5Driver;

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
    fn sd_source_handle(
        &self,
        pd_params_inp: &SymmetryGroupDetectionInputKind,
        afa_params: &AngularFunctionRepAnalysisParams,
        sda_params: &SlaterDeterminantRepAnalysisParams<f64>,
    ) -> Result<(), anyhow::Error> {
        let qchemarchive_path = &self.path;
        let mut qchem_h5_driver = QChemH5Driver::<f64>::builder()
            .filename(qchemarchive_path.into())
            .symmetry_group_detection_input(&pd_params_inp)
            .angular_function_analysis_parameters(&afa_params)
            .slater_det_rep_analysis_parameters(&sda_params)
            .build()?;
        qchem_h5_driver.run()
    }
}
