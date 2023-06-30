use hdf5;
use log4rs;
use nalgebra::{Point3, Vector3};

use super::{QChemH5Driver, QChemH5SinglePointDriver};
use crate::drivers::representation_analysis::angular_function::AngularFunctionRepAnalysisParams;
use crate::drivers::representation_analysis::slater_determinant::SlaterDeterminantRepAnalysisParams;
use crate::drivers::symmetry_group_detection::SymmetryGroupDetectionParams;
use crate::drivers::QSym2Driver;
use crate::symmetry::symmetry_group::{
    MagneticRepresentedSymmetryGroup, SymmetryGroupProperties, UnitaryRepresentedSymmetryGroup,
};

const ROOT: &str = env!("CARGO_MANIFEST_DIR");

#[test]
fn test_interfaces_qchem_hdf5_sp_vf63m_ms1() {
    // log4rs::init_file("log4rs.yml", Default::default()).unwrap();
    let name = format!("{ROOT}/tests/qchem/vf63m.qarchive.h5");
    let f = hdf5::File::open(&name).unwrap();
    let sp = f.group("job/1/sp").unwrap();
    let pd_params = SymmetryGroupDetectionParams::default();
    let afa_params = AngularFunctionRepAnalysisParams::default();
    let sda_params = SlaterDeterminantRepAnalysisParams::<f64>::default();
    let mut qchem_sp = QChemH5SinglePointDriver::<UnitaryRepresentedSymmetryGroup, f64>::builder()
        .sp_group(&sp)
        .symmetry_group_detection_parameters(&pd_params)
        .angular_function_analysis_parameters(&afa_params)
        .slater_det_rep_analysis_parameters(&sda_params)
        .build()
        .unwrap();
    assert!(qchem_sp.run().is_ok());
    let res = qchem_sp.result().unwrap();
    assert_eq!(res.0.group_name.as_ref().unwrap(), "Oh");
    assert_eq!(res.1.to_string(), "|T|_(1g)");
}

#[test]
fn test_interfaces_qchem_hdf5_sp_o2_ms0() {
    log4rs::init_file("log4rs.yml", Default::default()).unwrap();
    let name = format!("{ROOT}/tests/qchem/o2_ms0.qarchive.h5");
    let f = hdf5::File::open(&name).unwrap();
    let sp = f.group("job/1/sp").unwrap();
    let mut pd_params = SymmetryGroupDetectionParams::default();
    pd_params.time_reversal = true;

    let afa_params = AngularFunctionRepAnalysisParams::default();

    let mut sda_params = SlaterDeterminantRepAnalysisParams::<f64>::default();
    sda_params.use_magnetic_group = true;
    sda_params.infinite_order_to_finite = Some(8);

    let mut qchem_sp = QChemH5SinglePointDriver::<UnitaryRepresentedSymmetryGroup, f64>::builder()
        .sp_group(&sp)
        .symmetry_group_detection_parameters(&pd_params)
        .angular_function_analysis_parameters(&afa_params)
        .slater_det_rep_analysis_parameters(&sda_params)
        .build()
        .unwrap();
    assert!(qchem_sp.run().is_ok());
    let res = qchem_sp.result().unwrap();
    assert_eq!(res.0.group_name.as_ref().unwrap(), "D∞h + θ·D∞h");
    assert_eq!(
        res.1.to_string(),
        "^(+)|A|_(1g) ⊕ ^(+)|B|_(1g) ⊕ ^(+)|E|_(2g)"
    );
}

#[test]
fn test_interfaces_qchem_hdf5_sp_o2_ms0_fictitious_magnetic_field() {
    log4rs::init_file("log4rs.yml", Default::default()).unwrap();
    let name = format!("{ROOT}/tests/qchem/o2_ms0.qarchive.h5");
    let f = hdf5::File::open(&name).unwrap();
    let sp = f.group("job/1/sp").unwrap();
    let mut pd_params = SymmetryGroupDetectionParams::default();
    pd_params.time_reversal = true;
    pd_params.fictitious_magnetic_fields =
        Some(vec![(Point3::origin(), Vector3::new(0.0, 1.0, 0.0))]);
    pd_params.field_origin_com = true;

    let afa_params = AngularFunctionRepAnalysisParams::default();

    let mut sda_params = SlaterDeterminantRepAnalysisParams::<f64>::default();
    sda_params.use_magnetic_group = true;

    let mut qchem_sp = QChemH5SinglePointDriver::<MagneticRepresentedSymmetryGroup, f64>::builder()
        .sp_group(&sp)
        .symmetry_group_detection_parameters(&pd_params)
        .angular_function_analysis_parameters(&afa_params)
        .slater_det_rep_analysis_parameters(&sda_params)
        .build()
        .unwrap();
    assert!(qchem_sp.run().is_ok());
    let res = qchem_sp.result().unwrap();
    assert_eq!(res.0.group_name.as_ref().unwrap(), "D2h");
    assert_eq!(res.1.to_string(), "D[|A|_(g)]");
}

#[test]
fn test_interfaces_qchem_hdf5_full() {
    log4rs::init_file("log4rs.yml", Default::default()).unwrap();
    let name = format!("{ROOT}/tests/qchem/geomopt.qarchive.h5");
    let f = hdf5::File::open(&name).unwrap();
    let pd_params = SymmetryGroupDetectionParams::default();
    let mut afa_params = AngularFunctionRepAnalysisParams::default();
    afa_params.linear_independence_threshold = 1e-3;
    afa_params.integrality_threshold = 1e-5;
    let sda_params = SlaterDeterminantRepAnalysisParams::<f64>::default();
    let mut qchem_h5_driver = QChemH5Driver::<f64>::builder()
        .f(&f)
        .symmetry_group_detection_parameters(&pd_params)
        .angular_function_analysis_parameters(&afa_params)
        .slater_det_rep_analysis_parameters(&sda_params)
        .build()
        .unwrap();
    assert!(qchem_h5_driver.run().is_ok());
    let res = qchem_h5_driver.result().unwrap();
    assert_eq!(
        *res,
        vec![
            ("Cs".to_string(), "|A|^(')".to_string()),
            ("Cs".to_string(), "|A|^(')".to_string()),
            ("Cs".to_string(), "|A|^(')".to_string()),
            ("Cs".to_string(), "|A|^(')".to_string()),
            ("Cs".to_string(), "|A|^(')".to_string()),
            ("C2v".to_string(), "|A|_(1)".to_string()),
            ("C2v".to_string(), "|A|_(1)".to_string()),
            ("C2v".to_string(), "|A|_(1)".to_string()),
        ]
    );
}
