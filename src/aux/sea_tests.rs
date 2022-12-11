use crate::aux::molecule::Molecule;

const ROOT: &str = env!("CARGO_MANIFEST_DIR");

#[test]
fn test_sea_c60 () {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c60.xyz");
    let mol = Molecule::from_xyz(&path, 1e-4);
    assert_eq!(mol.calc_sea_groups().len(), 1);
}

#[test]
fn test_sea_th () {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/th.xyz");
    let mol = Molecule::from_xyz(&path, 1e-7);
    assert_eq!(mol.calc_sea_groups().len(), 1);
}

#[test]
fn test_sea_h8 () {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/h8.xyz");
    let mol = Molecule::from_xyz(&path, 1e-7);
    assert_eq!(mol.calc_sea_groups().len(), 1);
}

#[test]
fn test_sea_n3 () {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/n3.xyz");
    let mol = Molecule::from_xyz(&path, 1e-7);
    assert_eq!(mol.calc_sea_groups().len(), 3);
}

#[test]
fn test_sea_h3 () {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/h3.xyz");
    let mol = Molecule::from_xyz(&path, 1e-7);
    assert_eq!(mol.calc_sea_groups().len(), 1);
}

#[test]
fn test_sea_c3h3 () {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c3h3.xyz");
    let mol = Molecule::from_xyz(&path, 1e-7);
    assert_eq!(mol.calc_sea_groups().len(), 6);
}
