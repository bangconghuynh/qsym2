use rustyinspect::aux::molecule::Molecule;
use rustyinspect::rotsym::{self, RotationalSymmetry};

#[test]
fn test_c60 () {
    let mol = Molecule::from_xyz("tests/c60.xyz");
    let com = mol.calc_com(0);
    let inertia = mol.calc_moi(&com, 0);
    let rotsym_result = rotsym::calc_rotational_symmetry(&inertia, 1e-6, 0, false);
    assert!(matches!(rotsym_result, RotationalSymmetry::Spherical));
}

#[test]
fn test_th () {
    let mol = Molecule::from_xyz("tests/th.xyz");
    let com = mol.calc_com(0);
    let inertia = mol.calc_moi(&com, 0);
    let rotsym_result = rotsym::calc_rotational_symmetry(&inertia, 1e-14, 0, true);
    assert!(matches!(rotsym_result, RotationalSymmetry::Spherical));
}

#[test]
fn test_h8 () {
    let mol = Molecule::from_xyz("tests/h8.xyz");
    let com = mol.calc_com(0);
    let inertia = mol.calc_moi(&com, 0);
    let rotsym_result = rotsym::calc_rotational_symmetry(&inertia, 1e-14, 0, true);
    assert!(matches!(rotsym_result, RotationalSymmetry::ProlateNonLinear));
}

#[test]
fn test_n3 () {
    let mol = Molecule::from_xyz("tests/n3.xyz");
    let com = mol.calc_com(0);
    let inertia = mol.calc_moi(&com, 0);
    let rotsym_result = rotsym::calc_rotational_symmetry(&inertia, 1e-12, 0, true);
    assert!(matches!(rotsym_result, RotationalSymmetry::ProlateLinear));
}

#[test]
fn test_h3 () {
    let mol = Molecule::from_xyz("tests/h3.xyz");
    let com = mol.calc_com(0);
    let inertia = mol.calc_moi(&com, 0);
    let rotsym_result = rotsym::calc_rotational_symmetry(&inertia, 1e-6, 0, true);
    assert!(matches!(rotsym_result, RotationalSymmetry::OblatePlanar));
}

#[test]
fn test_c3h3 () {
    let mol = Molecule::from_xyz("tests/c3h3.xyz");
    let com = mol.calc_com(0);
    let inertia = mol.calc_moi(&com, 0);
    let rotsym_result = rotsym::calc_rotational_symmetry(&inertia, 1e-6, 0, true);
    assert!(matches!(rotsym_result, RotationalSymmetry::AsymmetricPlanar));
}
