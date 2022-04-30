use crate::aux::geometry::Transform;
use crate::aux::molecule::Molecule;
use crate::symmetry::symmetry_core::Symmetry;
use crate::symmetry::symmetry_element::{ElementOrder, SymmetryElementKind};
use nalgebra::Vector3;

const ROOT: &str = env!("CARGO_MANIFEST_DIR");

#[test]
fn test_symmetry_check_proper_improper_n3() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/n3.xyz");
    let mol = Molecule::from_xyz(&path, 1e-7);
    let sym = Symmetry::builder()
        .moi_threshold(1e-14)
        .molecule(&mol)
        .build()
        .unwrap();
    assert!(sym.check_proper(&ElementOrder::Int(2), &Vector3::new(1.0, 1.0, 1.0)));
    assert!(sym.check_proper(&ElementOrder::Int(15), &Vector3::new(1.0, 1.0, 1.0)));
    assert!(!sym.check_proper(&ElementOrder::Int(2), &Vector3::new(1.0, -1.0, 0.0)));

    let sig = SymmetryElementKind::ImproperMirrorPlane;
    let inv = SymmetryElementKind::ImproperInversionCentre;
    assert!(sym.check_improper(&ElementOrder::Int(1), &Vector3::new(-1.0, 0.0, 1.0), &sig));
    assert!(!sym.check_improper(&ElementOrder::Int(1), &Vector3::new(-1.0, 0.0, 1.0), &inv));
    assert!(sym.check_improper(&ElementOrder::Int(2), &Vector3::new(0.0, 1.0, -1.0), &inv));
    assert!(!sym.check_improper(&ElementOrder::Int(2), &Vector3::new(0.0, 1.0, -1.0), &sig));
}

#[test]
fn test_symmetry_check_proper_improper_h8() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/h8.xyz");
    let mut mol = Molecule::from_xyz(&path, 1e-7);
    mol.recentre_mut();
    let sym = Symmetry::builder()
        .moi_threshold(1e-14)
        .molecule(&mol)
        .build()
        .unwrap();
    assert!(sym.check_proper(&ElementOrder::Int(4), &Vector3::new(0.0, 0.0, 1.0)));
    assert!(sym.check_proper(&ElementOrder::Int(2), &Vector3::new(1.0, 0.0, 0.0)));
    assert!(sym.check_proper(&ElementOrder::Int(2), &Vector3::new(1.0, 1.0, 0.0)));

    let sig = SymmetryElementKind::ImproperMirrorPlane;
    let inv = SymmetryElementKind::ImproperInversionCentre;
    assert!(sym.check_improper(&ElementOrder::Int(1), &Vector3::new(0.0, 0.0, 1.0), &inv));
    assert!(sym.check_improper(&ElementOrder::Int(1), &Vector3::new(0.0, 0.0, 1.0), &sig));
    assert!(sym.check_improper(&ElementOrder::Int(1), &Vector3::new(0.0, 1.0, 0.0), &sig));
    assert!(sym.check_improper(&ElementOrder::Int(1), &Vector3::new(1.0, 1.0, 0.0), &sig));
}

#[test]
fn test_search_c2_spherical_c60() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c60.xyz");
    let mut mol = Molecule::from_xyz(&path, 1e-6);
    mol.recentre_mut();
    let mut sym = Symmetry::builder()
        .moi_threshold(1e-6)
        .molecule(&mol)
        .build()
        .unwrap();
    sym.analyse();
}
