use crate::aux::geometry::Transform;
use crate::aux::molecule::Molecule;
use crate::symmetry::symmetry_core::Symmetry;
use crate::symmetry::symmetry_element::ElementOrder;
use env_logger;

const ROOT: &str = env!("CARGO_MANIFEST_DIR");

#[test]
fn test_point_group_detection_atom() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/th.xyz");
    let mut mol = Molecule::from_xyz(&path, 1e-7);
    mol.recentre_mut();
    let mut sym = Symmetry::builder()
        .moi_threshold(1e-14)
        .molecule(&mol)
        .build()
        .unwrap();
    sym.analyse();
}

#[test]
fn test_point_group_detection_c60() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c60.xyz");
    let mut mol = Molecule::from_xyz(&path, 1e-6);
    mol.recentre_mut();
    let mut sym = Symmetry::builder()
        .moi_threshold(1e-6)
        .molecule(&mol)
        .build()
        .unwrap();
    sym.analyse();
    assert_eq!(sym.point_group, Some("Ih".to_owned()));
    assert_eq!(sym.proper_generators[&ElementOrder::Int(5)].len(), 6);
    assert_eq!(sym.improper_generators[&ElementOrder::Int(2)].len(), 1);
}

#[test]
fn test_point_group_detection_ch4() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/ch4.xyz");
    let mut mol = Molecule::from_xyz(&path, 1e-6);
    mol.recentre_mut();
    let mut sym = Symmetry::builder()
        .moi_threshold(1e-6)
        .molecule(&mol)
        .build()
        .unwrap();
    sym.analyse();
    assert_eq!(sym.point_group, Some("Td".to_owned()));
    assert_eq!(sym.proper_generators[&ElementOrder::Int(3)].len(), 4);
    assert_eq!(sym.improper_generators[&ElementOrder::Int(1)].len(), 1);
    assert_eq!(sym.get_sigma_generators("d").unwrap().len(), 1);
}

#[test]
fn test_point_group_detection_vh2o6() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/vh2o6.xyz");
    let mut mol = Molecule::from_xyz(&path, 1e-12);
    mol.recentre_mut();
    let mut sym = Symmetry::builder()
        .moi_threshold(1e-12)
        .molecule(&mol)
        .build()
        .unwrap();
    sym.analyse();
    assert_eq!(sym.point_group, Some("Th".to_owned()));
    assert_eq!(sym.proper_generators[&ElementOrder::Int(3)].len(), 4);
    assert_eq!(sym.improper_generators[&ElementOrder::Int(2)].len(), 1);
}

#[test]
fn test_point_group_detection_vf6() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/vf6.xyz");
    let mut mol = Molecule::from_xyz(&path, 1e-12);
    mol.recentre_mut();
    let mut sym = Symmetry::builder()
        .moi_threshold(1e-12)
        .molecule(&mol)
        .build()
        .unwrap();
    sym.analyse();
    assert_eq!(sym.point_group, Some("Oh".to_owned()));
    assert_eq!(sym.proper_generators[&ElementOrder::Int(3)].len(), 4);
    assert_eq!(sym.proper_generators[&ElementOrder::Int(4)].len(), 1);
    assert_eq!(sym.improper_generators[&ElementOrder::Int(2)].len(), 1);
}
