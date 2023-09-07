use log4rs;
use ndarray::array;
use num_complex::Complex;
use serial_test::serial;

use crate::chartab::chartab_symbols::DecomposedSymbol;
use crate::drivers::representation_analysis::angular_function::AngularFunctionRepAnalysisParams;
use crate::drivers::representation_analysis::vibrational_coordinate::{
    VibrationalCoordinateRepAnalysisDriver, VibrationalCoordinateRepAnalysisParams
};
use crate::drivers::representation_analysis::{
    CharacterTableDisplay, MagneticSymmetryAnalysisKind,
};
use crate::drivers::symmetry_group_detection::{
    SymmetryGroupDetectionDriver, SymmetryGroupDetectionParams,
};
use crate::drivers::QSym2Driver;
use crate::symmetry::symmetry_group::{
    MagneticRepresentedSymmetryGroup, UnitaryRepresentedSymmetryGroup,
};
use crate::symmetry::symmetry_symbols::{MullikenIrcorepSymbol, MullikenIrrepSymbol};
use crate::symmetry::symmetry_transformation::SymmetryTransformationKind;
use crate::target::vibration::VibrationalCoordinateCollection;

type C128 = Complex<f64>;

const ROOT: &str = env!("CARGO_MANIFEST_DIR");

#[test]
fn test_drivers_vibrational_coordinate_analysis_xef4() {
    log4rs::init_file("log4rs.yml", Default::default()).unwrap();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/xef4.xyz");

    let afa_params = AngularFunctionRepAnalysisParams::default();

    let pd_params = SymmetryGroupDetectionParams::builder()
        .moi_thresholds(&[1e-6])
        .distance_thresholds(&[1e-6])
        .field_origin_com(true)
        .time_reversal(true)
        .write_symmetry_elements(true)
        .build()
        .unwrap();
    let mut pd_driver = SymmetryGroupDetectionDriver::builder()
        .parameters(&pd_params)
        .xyz(Some(path.into()))
        .build()
        .unwrap();
    assert!(pd_driver.run().is_ok());
    let pd_res = pd_driver.result().unwrap();
    let mol_xef4 = &pd_res.pre_symmetry.recentred_molecule;

    #[rustfmt::skip]
    let vibrational_coordinates = array![
        [ 0.000,  0.000, -0.103, -0.126],
        [ 0.000,  0.000,  0.126, -0.103],
        [ 0.277,  0.000,  0.000,  0.000],
        [ 0.000,  0.000, -0.136,  0.474],
        [ 0.000,  0.000,  0.036,  0.492],
        [-0.480,  0.500,  0.000,  0.000],
        [ 0.000,  0.000,  0.492, -0.036],
        [ 0.000,  0.000, -0.474, -0.136],
        [-0.480, -0.500,  0.000,  0.000],
        [ 0.000,  0.000, -0.136,  0.474],
        [ 0.000,  0.000,  0.036,  0.492],
        [-0.480,  0.500,  0.000,  0.000],
        [ 0.000,  0.000,  0.492, -0.036],
        [ 0.000,  0.000, -0.474, -0.136],
        [-0.480, -0.500,  0.000,  0.000],
    ];
    let frequencies = array![-2841.05, -2438.27, -2205.51, -2205.51];
    let vibs = VibrationalCoordinateCollection::<f64>::builder()
        .coefficients(vibrational_coordinates)
        .frequencies(frequencies)
        .mol(mol_xef4)
        .threshold(1e-14)
        .build()
        .unwrap();

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // u Oh (ordinary, unitary) - spin spatial
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    let vca_params = VibrationalCoordinateRepAnalysisParams::<f64>::builder()
        .integrality_threshold(1e-3)
        .linear_independence_threshold(1e-3)
        .use_magnetic_group(None)
        .use_double_group(false)
        .symmetry_transformation_kind(SymmetryTransformationKind::Spatial)
        .write_character_table(Some(CharacterTableDisplay::Symbolic))
        .build()
        .unwrap();

    let mut vca_driver =
        VibrationalCoordinateRepAnalysisDriver::<UnitaryRepresentedSymmetryGroup, f64>::builder()
            .parameters(&vca_params)
            .angular_function_parameters(&afa_params)
            .vibrational_coordinate_collection(&vibs)
            .symmetry_group(pd_res)
            .build()
            .unwrap();
    assert!(vca_driver.run().is_ok());
    // assert_eq!(
    //     *sda_driver
    //         .result()
    //         .unwrap()
    //         .determinant_symmetry
    //         .as_ref()
    //         .unwrap(),
    //     DecomposedSymbol::<MullikenIrrepSymbol>::new("||T|_(1g)| ⊕ ||T|_(2g)|").unwrap()
    // );
}
