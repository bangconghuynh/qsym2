use crate::auxiliary::molecule::Molecule;
use crate::rotsym::{self, RotationalSymmetry};
use nalgebra::Vector3;

const ROOT: &str = env!("CARGO_MANIFEST_DIR");

#[test]
fn test_rotsym_c60() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c60.xyz");
    let mol = Molecule::from_xyz(&path, 1e-6);
    let com = mol.calc_com();
    let inertia = mol.calc_inertia_tensor(&com);
    let rotsym_result = rotsym::calc_rotational_symmetry(&inertia, 1e-6);
    assert!(matches!(rotsym_result, RotationalSymmetry::Spherical));
}

#[test]
fn test_rotsym_c60_field() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c60.xyz");
    let mut mol = Molecule::from_xyz(&path, 1e-6);
    mol.set_magnetic_field(Some(Vector3::new(2.0, 0.0, 0.0)));
    let com = mol.calc_com();
    let inertia = mol.calc_inertia_tensor(&com);
    let rotsym_result = rotsym::calc_rotational_symmetry(&inertia, 1e-6);
    assert!(matches!(
        rotsym_result,
        RotationalSymmetry::ProlateNonLinear
    ));
}

#[test]
fn test_rotsym_th() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/th.xyz");
    let mol = Molecule::from_xyz(&path, 1e-14);
    let com = mol.calc_com();
    let inertia = mol.calc_inertia_tensor(&com);
    let rotsym_result = rotsym::calc_rotational_symmetry(&inertia, 1e-14);
    assert!(matches!(rotsym_result, RotationalSymmetry::Spherical));
}

#[test]
fn test_rotsym_th_field() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/th.xyz");
    let mut mol = Molecule::from_xyz(&path, 1e-7);
    mol.set_magnetic_field(Some(Vector3::new(1.0, 0.0, 0.0)));
    let com = mol.calc_com();
    let inertia = mol.calc_inertia_tensor(&com);
    let rotsym_result = rotsym::calc_rotational_symmetry(&inertia, 1e-7);
    assert!(matches!(rotsym_result, RotationalSymmetry::ProlateLinear));

    mol.set_magnetic_field(None);
    mol.set_electric_field(Some(Vector3::new(0.0, 1.0, 0.0)));
    let com = mol.calc_com();
    let inertia = mol.calc_inertia_tensor(&com);
    let rotsym_result = rotsym::calc_rotational_symmetry(&inertia, 1e-7);
    assert!(matches!(rotsym_result, RotationalSymmetry::ProlateLinear));

    mol.set_magnetic_field(Some(Vector3::new(1.0, 0.0, 0.0)));
    let com = mol.calc_com();
    let inertia = mol.calc_inertia_tensor(&com);
    let rotsym_result = rotsym::calc_rotational_symmetry(&inertia, 1e-7);
    assert!(matches!(
        rotsym_result,
        RotationalSymmetry::AsymmetricPlanar
    ));
}

#[test]
fn test_rotsym_h8() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/h8.xyz");
    let mol = Molecule::from_xyz(&path, 1e-14);
    let com = mol.calc_com();
    let inertia = mol.calc_inertia_tensor(&com);
    let rotsym_result = rotsym::calc_rotational_symmetry(&inertia, 1e-14);
    assert!(matches!(
        rotsym_result,
        RotationalSymmetry::ProlateNonLinear
    ));
}

#[test]
fn test_rotsym_h8_field() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/h8.xyz");
    let mut mol = Molecule::from_xyz(&path, 1e-14);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    let com = mol.calc_com();
    let inertia = mol.calc_inertia_tensor(&com);
    let rotsym_result = rotsym::calc_rotational_symmetry(&inertia, 1e-14);
    assert!(matches!(
        rotsym_result,
        RotationalSymmetry::ProlateNonLinear
    ));

    mol.set_magnetic_field(Some(Vector3::new(0.0, 1.0, 0.0)));
    let com = mol.calc_com();
    let inertia = mol.calc_inertia_tensor(&com);
    let rotsym_result = rotsym::calc_rotational_symmetry(&inertia, 1e-14);
    assert!(matches!(
        rotsym_result,
        RotationalSymmetry::AsymmetricNonPlanar
    ));
}

#[test]
fn test_rotsym_n3() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/n3.xyz");
    let mol = Molecule::from_xyz(&path, 1e-12);
    let com = mol.calc_com();
    let inertia = mol.calc_inertia_tensor(&com);
    let rotsym_result = rotsym::calc_rotational_symmetry(&inertia, 1e-12);
    assert!(matches!(rotsym_result, RotationalSymmetry::ProlateLinear));
}

#[test]
fn test_rotsym_n3_field() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/n3.xyz");
    let mut mol = Molecule::from_xyz(&path, 1e-12);
    mol.set_magnetic_field(Some(Vector3::new(1.0, 0.0, 0.0)));
    let com = mol.calc_com();
    let inertia = mol.calc_inertia_tensor(&com);
    let rotsym_result = rotsym::calc_rotational_symmetry(&inertia, 1e-12);
    assert!(matches!(
        rotsym_result,
        RotationalSymmetry::AsymmetricPlanar
    ));

    mol.set_electric_field(Some(Vector3::new(0.0, 1.0, 0.0)));
    let com = mol.calc_com();
    let inertia = mol.calc_inertia_tensor(&com);
    let rotsym_result = rotsym::calc_rotational_symmetry(&inertia, 1e-12);
    assert!(matches!(
        rotsym_result,
        RotationalSymmetry::AsymmetricNonPlanar
    ));
}

#[test]
fn test_rotsym_h3() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/h3.xyz");
    let mol = Molecule::from_xyz(&path, 1e-6);
    let com = mol.calc_com();
    let inertia = mol.calc_inertia_tensor(&com);
    let rotsym_result = rotsym::calc_rotational_symmetry(&inertia, 1e-6);
    assert!(matches!(rotsym_result, RotationalSymmetry::OblatePlanar));
}

#[test]
fn test_rotsym_c3h3() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c3h3.xyz");
    let mol = Molecule::from_xyz(&path, 1e-6);
    let com = mol.calc_com();
    let inertia = mol.calc_inertia_tensor(&com);
    let rotsym_result = rotsym::calc_rotational_symmetry(&inertia, 1e-6);
    assert!(matches!(
        rotsym_result,
        RotationalSymmetry::AsymmetricPlanar
    ));
}

#[test]
fn test_rotsym_c3h3_field() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c3h3.xyz");
    let mut mol = Molecule::from_xyz(&path, 1e-6);
    mol.set_magnetic_field(Some(Vector3::new(1.0, 0.0, 0.0)));
    let com = mol.calc_com();
    let inertia = mol.calc_inertia_tensor(&com);
    let rotsym_result = rotsym::calc_rotational_symmetry(&inertia, 1e-6);
    assert!(matches!(
        rotsym_result,
        RotationalSymmetry::AsymmetricPlanar
    ));

    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    let com = mol.calc_com();
    let inertia = mol.calc_inertia_tensor(&com);
    let rotsym_result = rotsym::calc_rotational_symmetry(&inertia, 1e-6);
    assert!(matches!(
        rotsym_result,
        RotationalSymmetry::AsymmetricNonPlanar
    ));
}
