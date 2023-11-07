use nalgebra::Vector3;

use crate::auxiliary::molecule::Molecule;

const ROOT: &str = env!("CARGO_MANIFEST_DIR");

#[test]
fn test_sea_c60() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c60.xyz");
    let mol = Molecule::from_xyz(&path, 1e-4);
    assert_eq!(mol.calc_sea_groups().len(), 1);
}

#[test]
fn test_sea_th() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/th.xyz");
    let mol = Molecule::from_xyz(&path, 1e-7);
    assert_eq!(mol.calc_sea_groups().len(), 1);
}

#[test]
fn test_sea_h8() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/h8.xyz");
    let mol = Molecule::from_xyz(&path, 1e-7);
    assert_eq!(mol.calc_sea_groups().len(), 1);
}

#[test]
fn test_sea_n3() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/n3.xyz");
    let mol = Molecule::from_xyz(&path, 1e-7);
    assert_eq!(mol.calc_sea_groups().len(), 3);
}

#[test]
fn test_sea_h3() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/h3.xyz");
    let mol = Molecule::from_xyz(&path, 1e-7);
    assert_eq!(mol.calc_sea_groups().len(), 1);
}

#[test]
fn test_sea_c3h3() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c3h3.xyz");
    let mol = Molecule::from_xyz(&path, 1e-7);
    assert_eq!(mol.calc_sea_groups().len(), 6);
}

#[test]
fn test_sea_th_magnetic_field() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/th.xyz");
    let mut mol = Molecule::from_xyz(&path, 1e-7);
    mol.set_magnetic_field(Some(Vector3::z()));
    assert_eq!(mol.calc_sea_groups().len(), 2);
}

#[test]
fn test_sea_th_electric_field() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/th.xyz");
    let mut mol = Molecule::from_xyz(&path, 1e-7);
    mol.set_electric_field(Some(Vector3::z()));
    assert_eq!(mol.calc_sea_groups().len(), 2);
}

#[test]
fn test_sea_th_magnetic_electric_field() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/th.xyz");
    let mut mol = Molecule::from_xyz(&path, 1e-7);
    mol.set_magnetic_field(Some(Vector3::y()));
    mol.set_electric_field(Some(Vector3::z()));
    assert_eq!(mol.calc_sea_groups().len(), 3);
}

#[test]
fn test_sea_hf_magnetic_field() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/hf.xyz");
    let mut mol = Molecule::from_xyz(&path, 1e-7);
    mol.set_magnetic_field(Some(Vector3::y()));
    assert_eq!(mol.calc_sea_groups().len(), 3);
}
