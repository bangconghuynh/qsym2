// use env_logger;
use ndarray::array;

use crate::analysis::RepAnalysis;
use crate::aux::atom::{Atom, ElementMap};
use crate::aux::geometry::Transform;
use crate::aux::molecule::Molecule;
use crate::chartab::chartab_symbols::DecomposedSymbol;
use crate::group::UnitaryRepresentedGroup;
use crate::symmetry::symmetry_core::{PreSymmetry, Symmetry};
use crate::symmetry::symmetry_group::SymmetryGroupProperties;
use crate::symmetry::symmetry_symbols::MullikenIrrepSymbol;
use crate::symmetry::symmetry_transformation::SymmetryTransformationKind;
use crate::target::vibration::vibration_analysis::VibrationalCoordinateSymmetryOrbit;
use crate::target::vibration::VibrationalCoordinate;

const ROOT: &str = env!("CARGO_MANIFEST_DIR");

#[test]
fn test_vibration_orbit_rep_analysis_nh3() {
    // env_logger::init();
    let emap = ElementMap::new();
    let atm_n = Atom::from_xyz("N 0.0000000 0.0000000 -0.0980283", &emap, 1e-6).unwrap();
    let atm_h0 = Atom::from_xyz("H  0.4881263   -0.8454595    0.2287327", &emap, 1e-6).unwrap();
    let atm_h1 = Atom::from_xyz("H  0.4881263    0.8454595    0.2287327", &emap, 1e-6).unwrap();
    let atm_h2 = Atom::from_xyz("H -0.9762526    0.0000000    0.2287327", &emap, 1e-6).unwrap();

    let mol_nh3 = Molecule::from_atoms(
        &[
            atm_n.clone(),
            atm_h0.clone(),
            atm_h1.clone(),
            atm_h2.clone(),
        ],
        1e-6,
    )
    .recentre();

    let presym = PreSymmetry::builder()
        .moi_threshold(1e-6)
        .molecule(&mol_nh3, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    let group_u_c3v = UnitaryRepresentedGroup::from_molecular_symmetry(&sym, None);
    println!("{}", group_u_c3v.irrep_character_table.as_ref().unwrap());
    let group_u_c3v_double = group_u_c3v.to_double_group();

    // --------------------
    // Mode 1: 1149.58 cm-1
    // --------------------
    #[rustfmt::skip]
    let c1 = array![
        0.000,  0.000, -0.118,
       -0.088,  0.152,  0.546,
       -0.088, -0.152,  0.546,
        0.175, -0.000,  0.546
    ];
    let vib1 = VibrationalCoordinate::<f64>::builder()
        .coefficients(c1)
        .mol(&mol_nh3)
        .threshold(1e-3)
        .build()
        .unwrap();

    let mut orbit_u_nh3_spatial_vib1 = VibrationalCoordinateSymmetryOrbit::builder()
        .group(&group_u_c3v)
        .origin(&vib1)
        .symmetry_transformation_kind(SymmetryTransformationKind::Spatial)
        .build()
        .unwrap();
    orbit_u_nh3_spatial_vib1
        .calc_smat(None)
        .calc_xmat(false);
    assert_eq!(
        orbit_u_nh3_spatial_vib1.analyse_rep().unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new("||A|_(1)|").unwrap()
    );

    // --------------------
    // Mode 2: 1867.67 cm-1
    // --------------------
    #[rustfmt::skip]
    let c2 = array![
       -0.070,  0.000,  0.000,
        0.550,  0.390,  0.109,
        0.550, -0.390,  0.109,
       -0.125, -0.000, -0.217,
    ];
    let vib2 = VibrationalCoordinate::<f64>::builder()
        .coefficients(c2)
        .mol(&mol_nh3)
        .threshold(1e-4)
        .build()
        .unwrap();

    let mut orbit_u_nh3_spatial_vib2 = VibrationalCoordinateSymmetryOrbit::builder()
        .group(&group_u_c3v)
        .origin(&vib2)
        .symmetry_transformation_kind(SymmetryTransformationKind::Spatial)
        .build()
        .unwrap();
    orbit_u_nh3_spatial_vib2
        .calc_smat(None)
        .calc_xmat(false);
    assert_eq!(
        orbit_u_nh3_spatial_vib2.analyse_rep().unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new("||E||").unwrap()
    );

    // --------------------
    // Mode 3: 1867.67 cm-1
    // --------------------
    #[rustfmt::skip]
    let c3 = array![
        0.000, -0.070,  0.000,
        0.390,  0.100, -0.188,
       -0.390,  0.100,  0.188,
       -0.000,  0.775, -0.000,
    ];
    let vib3 = VibrationalCoordinate::<f64>::builder()
        .coefficients(c3)
        .mol(&mol_nh3)
        .threshold(1e-4)
        .build()
        .unwrap();

    let mut orbit_u_nh3_spatial_vib3 = VibrationalCoordinateSymmetryOrbit::builder()
        .group(&group_u_c3v)
        .origin(&vib3)
        .symmetry_transformation_kind(SymmetryTransformationKind::Spatial)
        .build()
        .unwrap();
    orbit_u_nh3_spatial_vib3
        .calc_smat(None)
        .calc_xmat(false);
    assert_eq!(
        orbit_u_nh3_spatial_vib3.analyse_rep().unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new("||E||").unwrap()
    );

    // --------------------
    // Mode 4: 3376.47 cm-1
    // --------------------
    #[rustfmt::skip]
    let c4 = array![
        0.000,  0.000,  0.032,
       -0.279,  0.483, -0.147,
       -0.279, -0.483, -0.147,
        0.558,  0.000, -0.147,
    ];
    let vib4 = VibrationalCoordinate::<f64>::builder()
        .coefficients(c4)
        .mol(&mol_nh3)
        .threshold(1e-4)
        .build()
        .unwrap();

    let mut orbit_u_nh3_spatial_vib4 = VibrationalCoordinateSymmetryOrbit::builder()
        .group(&group_u_c3v)
        .origin(&vib4)
        .symmetry_transformation_kind(SymmetryTransformationKind::Spatial)
        .build()
        .unwrap();
    orbit_u_nh3_spatial_vib4
        .calc_smat(None)
        .calc_xmat(false);
    assert_eq!(
        orbit_u_nh3_spatial_vib4.analyse_rep().unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new("||A|_(1)|").unwrap()
    );

    // --------------------
    // Mode 5: 3517.10 cm-1
    // --------------------
    #[rustfmt::skip]
    let c5 = array![
       -0.082,  0.000,  0.000,
        0.182, -0.341,  0.127,
        0.182,  0.341,  0.127,
        0.773,  0.000, -0.254,
    ];
    let vib5 = VibrationalCoordinate::<f64>::builder()
        .coefficients(c5)
        .mol(&mol_nh3)
        .threshold(1e-4)
        .build()
        .unwrap();

    let mut orbit_u_nh3_spatial_vib5 = VibrationalCoordinateSymmetryOrbit::builder()
        .group(&group_u_c3v)
        .origin(&vib5)
        .symmetry_transformation_kind(SymmetryTransformationKind::Spatial)
        .build()
        .unwrap();
    orbit_u_nh3_spatial_vib5
        .calc_smat(None)
        .calc_xmat(false);
    assert_eq!(
        orbit_u_nh3_spatial_vib5.analyse_rep().unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new("||E||").unwrap()
    );

    // --------------------
    // Mode 6: 3517.10 cm-1
    // --------------------
    #[rustfmt::skip]
    let c6 = array![
        0.000, -0.082,  0.000,
       -0.341,  0.576, -0.220,
        0.341,  0.576,  0.220,
        0.000, -0.015,  0.000,
    ];
    let vib6 = VibrationalCoordinate::<f64>::builder()
        .coefficients(c6)
        .mol(&mol_nh3)
        .threshold(1e-4)
        .build()
        .unwrap();

    let mut orbit_u_nh3_spatial_vib6 = VibrationalCoordinateSymmetryOrbit::builder()
        .group(&group_u_c3v)
        .origin(&vib6)
        .symmetry_transformation_kind(SymmetryTransformationKind::Spatial)
        .build()
        .unwrap();
    orbit_u_nh3_spatial_vib6
        .calc_smat(None)
        .calc_xmat(false);
    assert_eq!(
        orbit_u_nh3_spatial_vib6.analyse_rep().unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new("||E||").unwrap()
    );
}

#[test]
fn test_vibration_orbit_rep_analysis_ch4() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/ch4.xyz");
    let thresh = 1e-6;
    let mol_ch4 = Molecule::from_xyz(&path, thresh);

    let presym = PreSymmetry::builder()
        .moi_threshold(1e-6)
        .molecule(&mol_ch4, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    let group_u_td = UnitaryRepresentedGroup::from_molecular_symmetry(&sym, None);

    // ---------------
    // Mode 1: 1530.98
    // ---------------
    #[rustfmt::skip]
    let c1 = array![
       -0.062, -0.108,  0.009,
        0.401,  0.421,  0.061,
       -0.034,  0.191,  0.291,
        0.369,  0.453, -0.112,
       -0.002,  0.223, -0.342,
    ];
    let vib1 = VibrationalCoordinate::<f64>::builder()
        .coefficients(c1)
        .mol(&mol_ch4)
        .threshold(1e-3)
        .build()
        .unwrap();

    let mut orbit_u_ch4_spatial_vib1 = VibrationalCoordinateSymmetryOrbit::builder()
        .group(&group_u_td)
        .origin(&vib1)
        .symmetry_transformation_kind(SymmetryTransformationKind::Spatial)
        .build()
        .unwrap();
    orbit_u_ch4_spatial_vib1
        .calc_smat(None)
        .calc_xmat(false);
    assert_eq!(
        orbit_u_ch4_spatial_vib1.analyse_rep().unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new("||T|_(2)|").unwrap()
    );

    // ---------------
    // Mode 2: 1530.98
    // ---------------
    #[rustfmt::skip]
    let c2 = array![
       -0.104,  0.062,  0.030,
        0.251, -0.046, -0.399,
        0.368, -0.434, -0.011,
        0.138,  0.067,  0.219,
        0.481, -0.322, -0.169,
    ];
    let vib2 = VibrationalCoordinate::<f64>::builder()
        .coefficients(c2)
        .mol(&mol_ch4)
        .threshold(1e-3)
        .build()
        .unwrap();

    let mut orbit_u_ch4_spatial_vib2 = VibrationalCoordinateSymmetryOrbit::builder()
        .group(&group_u_td)
        .origin(&vib2)
        .symmetry_transformation_kind(SymmetryTransformationKind::Spatial)
        .build()
        .unwrap();
    orbit_u_ch4_spatial_vib2
        .calc_smat(None)
        .calc_xmat(false);
    assert_eq!(
        orbit_u_ch4_spatial_vib2.analyse_rep().unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new("||T|_(2)|").unwrap()
    );

    // ---------------
    // Mode 3: 1530.98
    // ---------------
    #[rustfmt::skip]
    let c3 = array![
       -0.031,  0.008, -0.121,
       -0.149,  0.259,  0.288,
        0.330,  0.145,  0.402,
        0.301, -0.191,  0.431,
       -0.120, -0.305,  0.317,
    ];
    let vib3 = VibrationalCoordinate::<f64>::builder()
        .coefficients(c3)
        .mol(&mol_ch4)
        .threshold(1e-3)
        .build()
        .unwrap();

    let mut orbit_u_ch4_spatial_vib3 = VibrationalCoordinateSymmetryOrbit::builder()
        .group(&group_u_td)
        .origin(&vib3)
        .symmetry_transformation_kind(SymmetryTransformationKind::Spatial)
        .build()
        .unwrap();
    orbit_u_ch4_spatial_vib3
        .calc_smat(None)
        .calc_xmat(false);
    assert_eq!(
        orbit_u_ch4_spatial_vib3.analyse_rep().unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new("||T|_(2)|").unwrap()
    );

    // ---------------
    // Mode 4: 1714.14
    // ---------------
    #[rustfmt::skip]
    let c4 = array![
       -0.000, -0.000,  0.000,
        0.362,  0.345, -0.017,
        0.362, -0.345,  0.017,
       -0.362, -0.345, -0.017,
       -0.362,  0.345,  0.017,
    ];
    let vib4 = VibrationalCoordinate::<f64>::builder()
        .coefficients(c4)
        .mol(&mol_ch4)
        .threshold(1e-3)
        .build()
        .unwrap();

    let mut orbit_u_ch4_spatial_vib4 = VibrationalCoordinateSymmetryOrbit::builder()
        .group(&group_u_td)
        .origin(&vib4)
        .symmetry_transformation_kind(SymmetryTransformationKind::Spatial)
        .build()
        .unwrap();
    orbit_u_ch4_spatial_vib4
        .calc_smat(None)
        .calc_xmat(false);
    assert_eq!(
        orbit_u_ch4_spatial_vib4.analyse_rep().unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new("||E||").unwrap()
    );

    // ---------------
    // Mode 5: 1714.14
    // ---------------
    #[rustfmt::skip]
    let c5 = array![
       -0.000,  0.000, -0.000,
       -0.189,  0.219,  0.408,
       -0.189, -0.219, -0.408,
        0.189, -0.219,  0.408,
        0.189,  0.219, -0.408,
    ];
    let vib5 = VibrationalCoordinate::<f64>::builder()
        .coefficients(c5)
        .mol(&mol_ch4)
        .threshold(1e-3)
        .build()
        .unwrap();

    let mut orbit_u_ch4_spatial_vib5 = VibrationalCoordinateSymmetryOrbit::builder()
        .group(&group_u_td)
        .origin(&vib5)
        .symmetry_transformation_kind(SymmetryTransformationKind::Spatial)
        .build()
        .unwrap();
    orbit_u_ch4_spatial_vib5
        .calc_smat(None)
        .calc_xmat(false);
    assert_eq!(
        orbit_u_ch4_spatial_vib5.analyse_rep().unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new("||E||").unwrap()
    );

    // ---------------
    // Mode 6: 2984.48
    // ---------------
    #[rustfmt::skip]
    let c6 = array![
        0.000, -0.000, -0.000,
       -0.289,  0.289, -0.289,
       -0.289, -0.289,  0.289,
        0.289, -0.289, -0.289,
        0.289,  0.289,  0.289,
    ];
    let vib6 = VibrationalCoordinate::<f64>::builder()
        .coefficients(c6)
        .mol(&mol_ch4)
        .threshold(1e-3)
        .build()
        .unwrap();

    let mut orbit_u_ch4_spatial_vib6 = VibrationalCoordinateSymmetryOrbit::builder()
        .group(&group_u_td)
        .origin(&vib6)
        .symmetry_transformation_kind(SymmetryTransformationKind::Spatial)
        .build()
        .unwrap();
    orbit_u_ch4_spatial_vib6
        .calc_smat(None)
        .calc_xmat(false);
    assert_eq!(
        orbit_u_ch4_spatial_vib6.analyse_rep().unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new("||A|_(1)|").unwrap()
    );

    // ---------------
    // Mode 7: 3064.80
    // ---------------
    #[rustfmt::skip]
    let c7 = array![
       -0.002, -0.074,  0.055,
       -0.405,  0.390, -0.394,
        0.416,  0.402, -0.405,
       -0.055,  0.040,  0.065,
        0.066,  0.051,  0.077,
    ];
    let vib7 = VibrationalCoordinate::<f64>::builder()
        .coefficients(c7)
        .mol(&mol_ch4)
        .threshold(1e-3)
        .build()
        .unwrap();

    let mut orbit_u_ch4_spatial_vib7 = VibrationalCoordinateSymmetryOrbit::builder()
        .group(&group_u_td)
        .origin(&vib7)
        .symmetry_transformation_kind(SymmetryTransformationKind::Spatial)
        .build()
        .unwrap();
    orbit_u_ch4_spatial_vib7
        .calc_smat(None)
        .calc_xmat(false);
    assert_eq!(
        orbit_u_ch4_spatial_vib7.analyse_rep().unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new("||T|_(2)|").unwrap()
    );

    // ---------------
    // Mode 8: 3064.80
    // ---------------
    #[rustfmt::skip]
    let c8 = array![
       -0.071, -0.034, -0.048,
        0.256, -0.277,  0.260,
        0.166,  0.173, -0.190,
       -0.051,  0.031,  0.028,
        0.473,  0.481,  0.478,
    ];
    let vib8 = VibrationalCoordinate::<f64>::builder()
        .coefficients(c8)
        .mol(&mol_ch4)
        .threshold(1e-3)
        .build()
        .unwrap();

    let mut orbit_u_ch4_spatial_vib8 = VibrationalCoordinateSymmetryOrbit::builder()
        .group(&group_u_td)
        .origin(&vib8)
        .symmetry_transformation_kind(SymmetryTransformationKind::Spatial)
        .build()
        .unwrap();
    orbit_u_ch4_spatial_vib8
        .calc_smat(None)
        .calc_xmat(false);
    assert_eq!(
        orbit_u_ch4_spatial_vib8.analyse_rep().unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new("||T|_(2)|").unwrap()
    );

    // ---------------
    // Mode 9: 3064.80
    // ---------------
    #[rustfmt::skip]
    let c9 = array![
        0.059, -0.043, -0.056,
       -0.135,  0.139, -0.158,
       -0.218, -0.238,  0.218,
       -0.492,  0.495,  0.493,
        0.139,  0.119,  0.116,
    ];
    let vib9 = VibrationalCoordinate::<f64>::builder()
        .coefficients(c9)
        .mol(&mol_ch4)
        .threshold(1e-3)
        .build()
        .unwrap();

    let mut orbit_u_ch4_spatial_vib9 = VibrationalCoordinateSymmetryOrbit::builder()
        .group(&group_u_td)
        .origin(&vib9)
        .symmetry_transformation_kind(SymmetryTransformationKind::Spatial)
        .build()
        .unwrap();
    orbit_u_ch4_spatial_vib9
        .calc_smat(None)
        .calc_xmat(false);
    assert_eq!(
        orbit_u_ch4_spatial_vib9.analyse_rep().unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new("||T|_(2)|").unwrap()
    );
}
