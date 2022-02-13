use crate::aux::molecule::Molecule;
use crate::rotsym::{self, RotationalSymmetry};

const ROOT: &str = env!("CARGO_MANIFEST_DIR");

#[test]
fn test_rotsym_c60 () {
    let path: String = format!("{}{}", ROOT, "/src/rotsym/xyz/c60.xyz");
    let mol = Molecule::from_xyz(&path);
    let com = mol.calc_com(0);
    let inertia = mol.calc_moi(&com, 0);
    let rotsym_result = rotsym::calc_rotational_symmetry(&inertia, 1e-6, 0, false);
    assert!(matches!(rotsym_result, RotationalSymmetry::Spherical));
}

#[test]
fn test_rotsym_th () {
    let path: String = format!("{}{}", ROOT, "/src/rotsym/xyz/th.xyz");
    let mol = Molecule::from_xyz(&path);
    let com = mol.calc_com(0);
    let inertia = mol.calc_moi(&com, 0);
    let rotsym_result = rotsym::calc_rotational_symmetry(&inertia, 1e-14, 0, true);
    assert!(matches!(rotsym_result, RotationalSymmetry::Spherical));
}

#[test]
fn test_rotsym_h8 () {
    let path: String = format!("{}{}", ROOT, "/src/rotsym/xyz/h8.xyz");
    let mol = Molecule::from_xyz(&path);
    let com = mol.calc_com(0);
    let inertia = mol.calc_moi(&com, 0);
    let rotsym_result = rotsym::calc_rotational_symmetry(&inertia, 1e-14, 0, true);
    assert!(matches!(rotsym_result, RotationalSymmetry::ProlateNonLinear));
}

#[test]
fn test_rotsym_n3 () {
    let path: String = format!("{}{}", ROOT, "/src/rotsym/xyz/n3.xyz");
    let mol = Molecule::from_xyz(&path);
    let com = mol.calc_com(0);
    let inertia = mol.calc_moi(&com, 0);
    let rotsym_result = rotsym::calc_rotational_symmetry(&inertia, 1e-12, 0, true);
    assert!(matches!(rotsym_result, RotationalSymmetry::ProlateLinear));
}

#[test]
fn test_rotsym_h3 () {
    let path: String = format!("{}{}", ROOT, "/src/rotsym/xyz/h3.xyz");
    let mol = Molecule::from_xyz(&path);
    let com = mol.calc_com(0);
    let inertia = mol.calc_moi(&com, 0);
    let rotsym_result = rotsym::calc_rotational_symmetry(&inertia, 1e-6, 0, true);
    assert!(matches!(rotsym_result, RotationalSymmetry::OblatePlanar));
}

#[test]
fn test_rotsym_c3h3 () {
    let path: String = format!("{}{}", ROOT, "/src/rotsym/xyz/c3h3.xyz");
    let mol = Molecule::from_xyz(&path);
    let com = mol.calc_com(0);
    let inertia = mol.calc_moi(&com, 0);
    let rotsym_result = rotsym::calc_rotational_symmetry(&inertia, 1e-6, 0, true);
    assert!(matches!(rotsym_result, RotationalSymmetry::AsymmetricPlanar));
}
