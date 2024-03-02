use nalgebra::Vector3;
use num_complex::Complex;

use crate::analysis::{EigenvalueComparisonMode, RepAnalysis};
use crate::auxiliary::molecule::Molecule;
use crate::chartab::chartab_symbols::DecomposedSymbol;
use crate::group::UnitaryRepresentedGroup;
use crate::symmetry::symmetry_core::{PreSymmetry, Symmetry};
use crate::symmetry::symmetry_group::SymmetryGroupProperties;
use crate::symmetry::symmetry_symbols::MullikenIrrepSymbol;
use crate::symmetry::symmetry_transformation::SymmetryTransformationKind;
use crate::target::tensor::axialvector::axialvector_analysis::AxialVector3SymmetryOrbit;
use crate::target::tensor::axialvector::{AxialVector3, TimeParity};

const ROOT: &str = env!("CARGO_MANIFEST_DIR");
type C128 = Complex<f64>;

#[test]
fn test_axialvector_orbit_rep_analysis_vf6_oh() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/vf6.xyz");
    let mol = Molecule::from_xyz(&path, 1e-6);
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false).unwrap();
    let group_u_oh = UnitaryRepresentedGroup::from_molecular_symmetry(&sym, None).unwrap();

    // ~~~~~~~~~~~~~~~~~~~~~~~~
    // u Oh (ordinary, unitary)
    // ~~~~~~~~~~~~~~~~~~~~~~~~

    // --
    // Rx
    // --
    let rx = AxialVector3::<f64>::builder()
        .components(Vector3::x())
        .time_parity(TimeParity::Even)
        .threshold(1e-14)
        .build()
        .unwrap();
    let mut orbit_u_oh_spatial_rx = AxialVector3SymmetryOrbit::builder()
        .group(&group_u_oh)
        .origin(&rx)
        .integrality_threshold(1e-7)
        .linear_independence_threshold(1e-4)
        .symmetry_transformation_kind(SymmetryTransformationKind::Spatial)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
        .build()
        .unwrap();
    let _ = orbit_u_oh_spatial_rx
        .calc_smat(None, None)
        .unwrap()
        .calc_xmat(false);
    assert_eq!(
        orbit_u_oh_spatial_rx.analyse_rep().unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new("||T|_(1g)|").unwrap()
    );

    // --
    // Ry
    // --
    let ry = AxialVector3::<f64>::builder()
        .components(Vector3::y())
        .time_parity(TimeParity::Even)
        .threshold(1e-14)
        .build()
        .unwrap();
    let mut orbit_u_oh_spatial_ry = AxialVector3SymmetryOrbit::builder()
        .group(&group_u_oh)
        .origin(&ry)
        .integrality_threshold(1e-7)
        .linear_independence_threshold(1e-4)
        .symmetry_transformation_kind(SymmetryTransformationKind::Spatial)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
        .build()
        .unwrap();
    let _ = orbit_u_oh_spatial_ry
        .calc_smat(None, None)
        .unwrap()
        .calc_xmat(false);
    assert_eq!(
        orbit_u_oh_spatial_ry.analyse_rep().unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new("||T|_(1g)|").unwrap()
    );

    // --
    // Rz
    // --
    let rz = AxialVector3::<f64>::builder()
        .components(Vector3::z())
        .time_parity(TimeParity::Even)
        .threshold(1e-14)
        .build()
        .unwrap();
    let mut orbit_u_oh_spatial_rz = AxialVector3SymmetryOrbit::builder()
        .group(&group_u_oh)
        .origin(&rz)
        .integrality_threshold(1e-7)
        .linear_independence_threshold(1e-4)
        .symmetry_transformation_kind(SymmetryTransformationKind::Spatial)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
        .build()
        .unwrap();
    let _ = orbit_u_oh_spatial_rz
        .calc_smat(None, None)
        .unwrap()
        .calc_xmat(false);
    assert_eq!(
        orbit_u_oh_spatial_rz.analyse_rep().unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new("||T|_(1g)|").unwrap()
    );
}

#[test]
fn test_axialvector_orbit_rep_analysis_benzene_d6h_x() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/benzene.xyz");
    let mol = Molecule::from_xyz(&path, 1e-6);
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false).unwrap();
    let group_u_d6h_x = UnitaryRepresentedGroup::from_molecular_symmetry(&sym, None).unwrap();

    // ~~~~~~~~~~~~~~~~~~~~~~~~
    // u Oh (ordinary, unitary)
    // ~~~~~~~~~~~~~~~~~~~~~~~~

    // ----------------
    // Rx (Rz in D6h_z)
    // ----------------
    let rx: AxialVector3<C128> = AxialVector3::<f64>::builder()
        .components(Vector3::x())
        .time_parity(TimeParity::Even)
        .threshold(1e-14)
        .build()
        .unwrap()
        .into();
    let mut orbit_u_d6h_x_spatial_rx = AxialVector3SymmetryOrbit::builder()
        .group(&group_u_d6h_x)
        .origin(&rx)
        .integrality_threshold(1e-7)
        .linear_independence_threshold(1e-4)
        .symmetry_transformation_kind(SymmetryTransformationKind::Spatial)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Real)
        .build()
        .unwrap();
    let _ = orbit_u_d6h_x_spatial_rx
        .calc_smat(None, None)
        .unwrap()
        .calc_xmat(false);
    assert_eq!(
        orbit_u_d6h_x_spatial_rx.analyse_rep().unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new("||A|_(2g)|").unwrap()
    );

    // ----------------
    // Ry (Ry in D6h_z)
    // ----------------
    let ry: AxialVector3<C128> = AxialVector3::<f64>::builder()
        .components(Vector3::y())
        .time_parity(TimeParity::Even)
        .threshold(1e-14)
        .build()
        .unwrap()
        .into();
    let mut orbit_u_d6h_x_spatial_ry = AxialVector3SymmetryOrbit::builder()
        .group(&group_u_d6h_x)
        .origin(&ry)
        .integrality_threshold(1e-7)
        .linear_independence_threshold(1e-4)
        .symmetry_transformation_kind(SymmetryTransformationKind::Spatial)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Real)
        .build()
        .unwrap();
    let _ = orbit_u_d6h_x_spatial_ry
        .calc_smat(None, None)
        .unwrap()
        .calc_xmat(false);
    assert_eq!(
        orbit_u_d6h_x_spatial_ry.analyse_rep().unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new("||E|_(1g)|").unwrap()
    );

    // ----------------
    // Rz (Rx in D6h_z)
    // ----------------
    let rz: AxialVector3<C128> = AxialVector3::<f64>::builder()
        .components(Vector3::z())
        .time_parity(TimeParity::Even)
        .threshold(1e-14)
        .build()
        .unwrap()
        .into();
    let mut orbit_u_d6h_x_spatial_rz = AxialVector3SymmetryOrbit::builder()
        .group(&group_u_d6h_x)
        .origin(&rz)
        .integrality_threshold(1e-7)
        .linear_independence_threshold(1e-4)
        .symmetry_transformation_kind(SymmetryTransformationKind::Spatial)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Real)
        .build()
        .unwrap();
    let _ = orbit_u_d6h_x_spatial_rz
        .calc_smat(None, None)
        .unwrap()
        .calc_xmat(false);
    assert_eq!(
        orbit_u_d6h_x_spatial_rz.analyse_rep().unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new("||E|_(1g)|").unwrap()
    );
}

#[test]
fn test_axialvector_orbit_rep_analysis_nh3_c3v() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/nh3.xyz");
    let mol = Molecule::from_xyz(&path, 1e-6);
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-6)
        .molecule(&mol)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false).unwrap();
    let group_u_c3v = UnitaryRepresentedGroup::from_molecular_symmetry(&sym, None).unwrap();

    // --
    // Rx
    // --
    let rx: AxialVector3<C128> = AxialVector3::<f64>::builder()
        .components(Vector3::x())
        .time_parity(TimeParity::Even)
        .threshold(1e-14)
        .build()
        .unwrap()
        .into();
    let mut orbit_u_c3v_spatial_rx = AxialVector3SymmetryOrbit::builder()
        .group(&group_u_c3v)
        .origin(&rx)
        .integrality_threshold(1e-7)
        .linear_independence_threshold(1e-4)
        .symmetry_transformation_kind(SymmetryTransformationKind::Spatial)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Real)
        .build()
        .unwrap();
    let _ = orbit_u_c3v_spatial_rx
        .calc_smat(None, None)
        .unwrap()
        .calc_xmat(false);
    assert_eq!(
        orbit_u_c3v_spatial_rx.analyse_rep().unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new("||E||").unwrap()
    );

    // --
    // Ry
    // --
    let ry: AxialVector3<C128> = AxialVector3::<f64>::builder()
        .components(Vector3::y())
        .time_parity(TimeParity::Even)
        .threshold(1e-14)
        .build()
        .unwrap()
        .into();
    let mut orbit_u_c3v_spatial_ry = AxialVector3SymmetryOrbit::builder()
        .group(&group_u_c3v)
        .origin(&ry)
        .integrality_threshold(1e-7)
        .linear_independence_threshold(1e-4)
        .symmetry_transformation_kind(SymmetryTransformationKind::Spatial)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Real)
        .build()
        .unwrap();
    let _ = orbit_u_c3v_spatial_ry
        .calc_smat(None, None)
        .unwrap()
        .calc_xmat(false);
    assert_eq!(
        orbit_u_c3v_spatial_ry.analyse_rep().unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new("||E||").unwrap()
    );

    // --
    // Rz
    // --
    let rz: AxialVector3<C128> = AxialVector3::<f64>::builder()
        .components(Vector3::z())
        .time_parity(TimeParity::Even)
        .threshold(1e-14)
        .build()
        .unwrap()
        .into();
    let mut orbit_u_c3v_spatial_rz = AxialVector3SymmetryOrbit::builder()
        .group(&group_u_c3v)
        .origin(&rz)
        .integrality_threshold(1e-7)
        .linear_independence_threshold(1e-4)
        .symmetry_transformation_kind(SymmetryTransformationKind::Spatial)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Real)
        .build()
        .unwrap();
    let _ = orbit_u_c3v_spatial_rz
        .calc_smat(None, None)
        .unwrap()
        .calc_xmat(false);
    assert_eq!(
        orbit_u_c3v_spatial_rz.analyse_rep().unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new("||A|_(2)|").unwrap()
    );
}

#[test]
fn test_axialvector_orbit_rep_analysis_water_c2v_y() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/water.xyz");
    let mol = Molecule::from_xyz(&path, 1e-6);
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false).unwrap();
    let group_u_c2v_y = UnitaryRepresentedGroup::from_molecular_symmetry(&sym, None).unwrap();

    // ----------------
    // Rx (Rx in C2v_z)
    // ----------------
    let rx: AxialVector3<C128> = AxialVector3::<f64>::builder()
        .components(Vector3::x())
        .time_parity(TimeParity::Even)
        .threshold(1e-14)
        .build()
        .unwrap()
        .into();
    let mut orbit_u_c2v_y_spatial_rx = AxialVector3SymmetryOrbit::builder()
        .group(&group_u_c2v_y)
        .origin(&rx)
        .integrality_threshold(1e-7)
        .linear_independence_threshold(1e-4)
        .symmetry_transformation_kind(SymmetryTransformationKind::Spatial)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Real)
        .build()
        .unwrap();
    let _ = orbit_u_c2v_y_spatial_rx
        .calc_smat(None, None)
        .unwrap()
        .calc_xmat(false);
    assert_eq!(
        orbit_u_c2v_y_spatial_rx.analyse_rep().unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new("||B|_(2)|").unwrap()
    );

    // ----------------
    // Ry (Rz in C2v_z)
    // ----------------
    let ry: AxialVector3<C128> = AxialVector3::<f64>::builder()
        .components(Vector3::y())
        .time_parity(TimeParity::Even)
        .threshold(1e-14)
        .build()
        .unwrap()
        .into();
    let mut orbit_u_c2v_y_spatial_ry = AxialVector3SymmetryOrbit::builder()
        .group(&group_u_c2v_y)
        .origin(&ry)
        .integrality_threshold(1e-7)
        .linear_independence_threshold(1e-4)
        .symmetry_transformation_kind(SymmetryTransformationKind::Spatial)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Real)
        .build()
        .unwrap();
    let _ = orbit_u_c2v_y_spatial_ry
        .calc_smat(None, None)
        .unwrap()
        .calc_xmat(false);
    assert_eq!(
        orbit_u_c2v_y_spatial_ry.analyse_rep().unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new("||A|_(2)|").unwrap()
    );

    // ----------------
    // Rz (Ry in C2v_z)
    // ----------------
    let rz: AxialVector3<C128> = AxialVector3::<f64>::builder()
        .components(Vector3::z())
        .time_parity(TimeParity::Even)
        .threshold(1e-14)
        .build()
        .unwrap()
        .into();
    let mut orbit_u_c2v_y_spatial_rz = AxialVector3SymmetryOrbit::builder()
        .group(&group_u_c2v_y)
        .origin(&rz)
        .integrality_threshold(1e-7)
        .linear_independence_threshold(1e-4)
        .symmetry_transformation_kind(SymmetryTransformationKind::Spatial)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Real)
        .build()
        .unwrap();
    let _ = orbit_u_c2v_y_spatial_rz
        .calc_smat(None, None)
        .unwrap()
        .calc_xmat(false);
    assert_eq!(
        orbit_u_c2v_y_spatial_rz.analyse_rep().unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new("||B|_(1)|").unwrap()
    );
}

#[test]
fn test_axialvector_orbit_rep_analysis_bf3_magnetic_field_c3h() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/bf3.xyz");
    let mut mol = Molecule::from_xyz(&path, 1e-7);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 0.1)));
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false).unwrap();
    let group_u_c2v_y = UnitaryRepresentedGroup::from_molecular_symmetry(&sym, None).unwrap();

    // --
    // Rx
    // --
    let rx: AxialVector3<C128> = AxialVector3::<f64>::builder()
        .components(Vector3::x())
        .time_parity(TimeParity::Even)
        .threshold(1e-14)
        .build()
        .unwrap()
        .into();
    let mut orbit_u_c2v_y_spatial_rx = AxialVector3SymmetryOrbit::builder()
        .group(&group_u_c2v_y)
        .origin(&rx)
        .integrality_threshold(1e-7)
        .linear_independence_threshold(1e-4)
        .symmetry_transformation_kind(SymmetryTransformationKind::Spatial)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Real)
        .build()
        .unwrap();
    let _ = orbit_u_c2v_y_spatial_rx
        .calc_smat(None, None)
        .unwrap()
        .calc_xmat(false);
    assert_eq!(
        orbit_u_c2v_y_spatial_rx.analyse_rep().unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new("|_(a)|Γ|^('')| ⊕ |_(b)|Γ|^('')|").unwrap()
    );

    // --
    // Ry
    // --
    let ry: AxialVector3<C128> = AxialVector3::<f64>::builder()
        .components(Vector3::y())
        .time_parity(TimeParity::Even)
        .threshold(1e-14)
        .build()
        .unwrap()
        .into();
    let mut orbit_u_c2v_y_spatial_ry = AxialVector3SymmetryOrbit::builder()
        .group(&group_u_c2v_y)
        .origin(&ry)
        .integrality_threshold(1e-7)
        .linear_independence_threshold(1e-4)
        .symmetry_transformation_kind(SymmetryTransformationKind::Spatial)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Real)
        .build()
        .unwrap();
    let _ = orbit_u_c2v_y_spatial_ry
        .calc_smat(None, None)
        .unwrap()
        .calc_xmat(false);
    assert_eq!(
        orbit_u_c2v_y_spatial_ry.analyse_rep().unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new("|_(a)|Γ|^('')| ⊕ |_(b)|Γ|^('')|").unwrap()
    );

    // --
    // Rz
    // --
    let rz: AxialVector3<C128> = AxialVector3::<f64>::builder()
        .components(Vector3::z())
        .time_parity(TimeParity::Even)
        .threshold(1e-14)
        .build()
        .unwrap()
        .into();
    let mut orbit_u_c2v_y_spatial_rz = AxialVector3SymmetryOrbit::builder()
        .group(&group_u_c2v_y)
        .origin(&rz)
        .integrality_threshold(1e-7)
        .linear_independence_threshold(1e-4)
        .symmetry_transformation_kind(SymmetryTransformationKind::Spatial)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Real)
        .build()
        .unwrap();
    let _ = orbit_u_c2v_y_spatial_rz
        .calc_smat(None, None)
        .unwrap()
        .calc_xmat(false);
    assert_eq!(
        orbit_u_c2v_y_spatial_rz.analyse_rep().unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new("||A|^(')|").unwrap()
    );
}
