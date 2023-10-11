// use env_logger;
use hdf5;

use super::QChemVibrationH5SinglePointDriver;

use crate::drivers::representation_analysis::angular_function::AngularFunctionRepAnalysisParams;
use crate::drivers::representation_analysis::vibrational_coordinate::VibrationalCoordinateRepAnalysisParams;
use crate::drivers::representation_analysis::MagneticSymmetryAnalysisKind;
use crate::drivers::symmetry_group_detection::SymmetryGroupDetectionParams;
use crate::drivers::QSym2Driver;
use crate::interfaces::input::SymmetryGroupDetectionInputKind;
use crate::symmetry::symmetry_group::{
    MagneticRepresentedSymmetryGroup, UnitaryRepresentedSymmetryGroup,
};

const ROOT: &str = env!("CARGO_MANIFEST_DIR");

#[test]
fn test_interfaces_qchem_hdf5_vc_benzene_unitary() {
    // log4rs::init_file("log4rs.yml", Default::default()).unwrap();
    let name = format!("{ROOT}/tests/qchem/benzene_freq.qarchive.h5");
    let f = hdf5::File::open(&name).unwrap();
    let sp = f.group("job/1/geom_opt/iter/5/sp").unwrap();
    let pd_params = SymmetryGroupDetectionParams::default();
    let pd_params_inp = SymmetryGroupDetectionInputKind::Parameters(pd_params);
    let afa_params = AngularFunctionRepAnalysisParams::default();
    let vca_params = VibrationalCoordinateRepAnalysisParams::<f64>::default();
    let mut qchem_sp =
        QChemVibrationH5SinglePointDriver::<UnitaryRepresentedSymmetryGroup, f64>::builder()
            .sp_group(&sp)
            .energy_function_index("3")
            .symmetry_group_detection_input(&pd_params_inp)
            .angular_function_analysis_parameters(&afa_params)
            .rep_analysis_parameters(&vca_params)
            .build()
            .unwrap();
    assert!(qchem_sp.run().is_ok());
    let res = qchem_sp.result().unwrap();
    assert_eq!(res.0.group_name.as_ref().unwrap(), "D6h");

    let vc_syms = res.1.as_ref().unwrap();
    assert_eq!(vc_syms[0], "|E|_(2u)");
    assert_eq!(vc_syms[1], "|E|_(2u)");
    assert_eq!(vc_syms[24], "|B|_(1u)");
    assert_eq!(vc_syms[25], "|E|_(2g)");
    assert_eq!(vc_syms[26], "|E|_(2g)");
    assert_eq!(vc_syms[27], "|E|_(1u)");
    assert_eq!(vc_syms[28], "|E|_(1u)");
    assert_eq!(vc_syms[29], "|A|_(1g)");
}

#[test]
fn test_interfaces_qchem_hdf5_vc_benzene_magnetic() {
    // log4rs::init_file("log4rs.yml", Default::default()).unwrap();
    let name = format!("{ROOT}/tests/qchem/benzene_freq.qarchive.h5");
    let f = hdf5::File::open(&name).unwrap();
    let sp = f.group("job/1/geom_opt/iter/5/sp").unwrap();
    let mut pd_params = SymmetryGroupDetectionParams::default();
    pd_params.time_reversal = true;
    let pd_params_inp = SymmetryGroupDetectionInputKind::Parameters(pd_params);
    let afa_params = AngularFunctionRepAnalysisParams::default();
    let mut vca_params = VibrationalCoordinateRepAnalysisParams::<f64>::default();
    vca_params.use_magnetic_group = Some(MagneticSymmetryAnalysisKind::Corepresentation);
    let mut qchem_sp =
        QChemVibrationH5SinglePointDriver::<MagneticRepresentedSymmetryGroup, f64>::builder()
            .sp_group(&sp)
            .energy_function_index("3")
            .symmetry_group_detection_input(&pd_params_inp)
            .angular_function_analysis_parameters(&afa_params)
            .rep_analysis_parameters(&vca_params)
            .build()
            .unwrap();
    assert!(qchem_sp.run().is_ok());
    let res = qchem_sp.result().unwrap();
    assert_eq!(res.0.group_name.as_ref().unwrap(), "D6h + θ·D6h");

    let vc_syms = res.1.as_ref().unwrap();
    assert_eq!(vc_syms[0], "D[|E|_(2u)]");
    assert_eq!(vc_syms[1], "D[|E|_(2u)]");
    assert_eq!(vc_syms[24], "D[|B|_(1u)]");
    assert_eq!(vc_syms[25], "D[|E|_(2g)]");
    assert_eq!(vc_syms[26], "D[|E|_(2g)]");
    assert_eq!(vc_syms[27], "D[|E|_(1u)]");
    assert_eq!(vc_syms[28], "D[|E|_(1u)]");
    assert_eq!(vc_syms[29], "D[|A|_(1g)]");
}
