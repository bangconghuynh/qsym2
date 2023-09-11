use log4rs;
use nalgebra::{Point3, Vector3};
use ndarray::array;
use num_complex::Complex;

use crate::chartab::chartab_symbols::DecomposedSymbol;
use crate::drivers::representation_analysis::angular_function::AngularFunctionRepAnalysisParams;
use crate::drivers::representation_analysis::vibrational_coordinate::{
    VibrationalCoordinateRepAnalysisDriver, VibrationalCoordinateRepAnalysisParams,
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

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // u Oh (ordinary, unitary) - spatial
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
    assert_eq!(
        *vca_driver
            .result()
            .unwrap()
            .vibrational_coordinate_symmetries()[0]
            .as_ref()
            .unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new("||A|_(2u)|").unwrap()
    );
    assert_eq!(
        *vca_driver
            .result()
            .unwrap()
            .vibrational_coordinate_symmetries()[1]
            .as_ref()
            .unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new("||B|_(1u)|").unwrap()
    );
    assert_eq!(
        *vca_driver
            .result()
            .unwrap()
            .vibrational_coordinate_symmetries()[2]
            .as_ref()
            .unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new("||E|_(u)|").unwrap()
    );
    assert_eq!(
        *vca_driver
            .result()
            .unwrap()
            .vibrational_coordinate_symmetries()[3]
            .as_ref()
            .unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new("||E|_(u)|").unwrap()
    );

    let vibs_c: VibrationalCoordinateCollection<C128> = vibs.into();
    let mut vca_driver_c =
        VibrationalCoordinateRepAnalysisDriver::<UnitaryRepresentedSymmetryGroup, C128>::builder()
            .parameters(&vca_params)
            .angular_function_parameters(&afa_params)
            .vibrational_coordinate_collection(&vibs_c)
            .symmetry_group(pd_res)
            .build()
            .unwrap();
    assert!(vca_driver_c.run().is_ok());
    assert_eq!(
        *vca_driver_c
            .result()
            .unwrap()
            .vibrational_coordinate_symmetries()[0]
            .as_ref()
            .unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new("||A|_(2u)|").unwrap()
    );
    assert_eq!(
        *vca_driver_c
            .result()
            .unwrap()
            .vibrational_coordinate_symmetries()[1]
            .as_ref()
            .unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new("||B|_(1u)|").unwrap()
    );
    assert_eq!(
        *vca_driver_c
            .result()
            .unwrap()
            .vibrational_coordinate_symmetries()[2]
            .as_ref()
            .unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new("||E|_(u)|").unwrap()
    );
    assert_eq!(
        *vca_driver_c
            .result()
            .unwrap()
            .vibrational_coordinate_symmetries()[3]
            .as_ref()
            .unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new("||E|_(u)|").unwrap()
    );
}

#[test]
fn test_drivers_vibrational_coordinate_analysis_xef4_magnetic_field() {
    log4rs::init_file("log4rs.yml", Default::default()).unwrap();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/xef4.xyz");

    let afa_params = AngularFunctionRepAnalysisParams::default();

    let pd_params = SymmetryGroupDetectionParams::builder()
        .moi_thresholds(&[1e-6])
        .distance_thresholds(&[1e-6])
        .field_origin_com(true)
        .fictitious_magnetic_fields(Some(vec![(Point3::origin(), 0.1 * Vector3::z())]))
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

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // u C4h (ordinary, unitary) - spatial
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
    assert_eq!(
        *vca_driver
            .result()
            .unwrap()
            .vibrational_coordinate_symmetries()[0]
            .as_ref()
            .unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new("||A|_(u)|").unwrap()
    );
    assert_eq!(
        *vca_driver
            .result()
            .unwrap()
            .vibrational_coordinate_symmetries()[1]
            .as_ref()
            .unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new("||B|_(u)|").unwrap()
    );
    assert_eq!(
        *vca_driver
            .result()
            .unwrap()
            .vibrational_coordinate_symmetries()[2]
            .as_ref()
            .unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new("|_(a)|Γ|_(u)| ⊕ |_(b)|Γ|_(u)|").unwrap()
    );
    assert_eq!(
        *vca_driver
            .result()
            .unwrap()
            .vibrational_coordinate_symmetries()[3]
            .as_ref()
            .unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new("|_(a)|Γ|_(u)| ⊕ |_(b)|Γ|_(u)|").unwrap()
    );

    let vibs_c: VibrationalCoordinateCollection<C128> = vibs.clone().into();
    let mut vca_driver_c =
        VibrationalCoordinateRepAnalysisDriver::<UnitaryRepresentedSymmetryGroup, C128>::builder()
            .parameters(&vca_params)
            .angular_function_parameters(&afa_params)
            .vibrational_coordinate_collection(&vibs_c)
            .symmetry_group(pd_res)
            .build()
            .unwrap();
    assert!(vca_driver_c.run().is_ok());
    assert_eq!(
        *vca_driver_c
            .result()
            .unwrap()
            .vibrational_coordinate_symmetries()[0]
            .as_ref()
            .unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new("||A|_(u)|").unwrap()
    );
    assert_eq!(
        *vca_driver_c
            .result()
            .unwrap()
            .vibrational_coordinate_symmetries()[1]
            .as_ref()
            .unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new("||B|_(u)|").unwrap()
    );
    assert_eq!(
        *vca_driver_c
            .result()
            .unwrap()
            .vibrational_coordinate_symmetries()[2]
            .as_ref()
            .unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new("|_(a)|Γ|_(u)| ⊕ |_(b)|Γ|_(u)|").unwrap()
    );
    assert_eq!(
        *vca_driver_c
            .result()
            .unwrap()
            .vibrational_coordinate_symmetries()[3]
            .as_ref()
            .unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new("|_(a)|Γ|_(u)| ⊕ |_(b)|Γ|_(u)|").unwrap()
    );

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // m D4h(C4h) (bw, magnetic) - spatial
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    let vca_params = VibrationalCoordinateRepAnalysisParams::<f64>::builder()
        .integrality_threshold(1e-3)
        .linear_independence_threshold(1e-3)
        .use_magnetic_group(Some(MagneticSymmetryAnalysisKind::Corepresentation))
        .use_double_group(false)
        .symmetry_transformation_kind(SymmetryTransformationKind::Spatial)
        .write_character_table(Some(CharacterTableDisplay::Symbolic))
        .build()
        .unwrap();
    let mut vca_driver =
        VibrationalCoordinateRepAnalysisDriver::<MagneticRepresentedSymmetryGroup, f64>::builder()
            .parameters(&vca_params)
            .angular_function_parameters(&afa_params)
            .vibrational_coordinate_collection(&vibs)
            .symmetry_group(pd_res)
            .build()
            .unwrap();
    assert!(vca_driver.run().is_ok());
    assert_eq!(
        *vca_driver
            .result()
            .unwrap()
            .vibrational_coordinate_symmetries()[0]
            .as_ref()
            .unwrap(),
        DecomposedSymbol::<MullikenIrcorepSymbol>::new("||A|_(u)|").unwrap()
    );
    assert_eq!(
        *vca_driver
            .result()
            .unwrap()
            .vibrational_coordinate_symmetries()[1]
            .as_ref()
            .unwrap(),
        DecomposedSymbol::<MullikenIrcorepSymbol>::new("||B|_(u)|").unwrap()
    );
    assert_eq!(
        *vca_driver
            .result()
            .unwrap()
            .vibrational_coordinate_symmetries()[2]
            .as_ref()
            .unwrap(),
        DecomposedSymbol::<MullikenIrcorepSymbol>::new("|_(a)|Γ|_(u)| ⊕ |_(b)|Γ|_(u)|").unwrap()
    );
    assert_eq!(
        *vca_driver
            .result()
            .unwrap()
            .vibrational_coordinate_symmetries()[3]
            .as_ref()
            .unwrap(),
        DecomposedSymbol::<MullikenIrcorepSymbol>::new("|_(a)|Γ|_(u)| ⊕ |_(b)|Γ|_(u)|").unwrap()
    );

    let mut vca_driver_c =
        VibrationalCoordinateRepAnalysisDriver::<MagneticRepresentedSymmetryGroup, C128>::builder()
            .parameters(&vca_params)
            .angular_function_parameters(&afa_params)
            .vibrational_coordinate_collection(&vibs_c)
            .symmetry_group(pd_res)
            .build()
            .unwrap();
    assert!(vca_driver_c.run().is_ok());
    assert_eq!(
        *vca_driver_c
            .result()
            .unwrap()
            .vibrational_coordinate_symmetries()[0]
            .as_ref()
            .unwrap(),
        DecomposedSymbol::<MullikenIrcorepSymbol>::new("||A|_(u)|").unwrap()
    );
    assert_eq!(
        *vca_driver_c
            .result()
            .unwrap()
            .vibrational_coordinate_symmetries()[1]
            .as_ref()
            .unwrap(),
        DecomposedSymbol::<MullikenIrcorepSymbol>::new("||B|_(u)|").unwrap()
    );
    assert_eq!(
        *vca_driver_c
            .result()
            .unwrap()
            .vibrational_coordinate_symmetries()[2]
            .as_ref()
            .unwrap(),
        DecomposedSymbol::<MullikenIrcorepSymbol>::new("|_(a)|Γ|_(u)| ⊕ |_(b)|Γ|_(u)|").unwrap()
    );
    assert_eq!(
        *vca_driver_c
            .result()
            .unwrap()
            .vibrational_coordinate_symmetries()[3]
            .as_ref()
            .unwrap(),
        DecomposedSymbol::<MullikenIrcorepSymbol>::new("|_(a)|Γ|_(u)| ⊕ |_(b)|Γ|_(u)|").unwrap()
    );
}
