use crate::auxiliary::molecule::Molecule;
use crate::symmetry::symmetry_core::PreSymmetry;
use crate::symmetry::symmetry_element::{INV, SIG};
use crate::symmetry::symmetry_element_order::ElementOrder;
use nalgebra::Vector3;

const ROOT: &str = env!("CARGO_MANIFEST_DIR");

#[test]
fn test_symmetry_check_proper_improper_n3() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/n3.xyz");
    let mol = Molecule::from_xyz(&path, 1e-7);
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-14)
        .molecule(&mol)
        .build()
        .unwrap();
    assert!(presym
        .check_proper(&ElementOrder::Int(2), &Vector3::new(1.0, 1.0, 1.0), false)
        .is_some());
    assert!(presym
        .check_proper(&ElementOrder::Int(15), &Vector3::new(1.0, 1.0, 1.0), false)
        .is_some());
    assert!(presym
        .check_proper(&ElementOrder::Int(2), &Vector3::new(1.0, -1.0, 0.0), false)
        .is_none());

    assert!(presym
        .check_improper(
            &ElementOrder::Int(1),
            &Vector3::new(-1.0, 0.0, 1.0),
            &SIG,
            false
        )
        .is_some());
    assert!(presym
        .check_improper(
            &ElementOrder::Int(1),
            &Vector3::new(-1.0, 0.0, 1.0),
            &INV,
            false
        )
        .is_none());
    assert!(presym
        .check_improper(
            &ElementOrder::Int(2),
            &Vector3::new(0.0, 1.0, -1.0),
            &INV,
            false
        )
        .is_some());
    assert!(presym
        .check_improper(
            &ElementOrder::Int(2),
            &Vector3::new(0.0, 1.0, -1.0),
            &SIG,
            false
        )
        .is_none());
}

#[test]
fn test_symmetry_check_proper_improper_h8() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/h8.xyz");
    let mol = Molecule::from_xyz(&path, 1e-7);
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-14)
        .molecule(&mol)
        .build()
        .unwrap();
    assert!(presym
        .check_proper(&ElementOrder::Int(4), &Vector3::new(0.0, 0.0, 1.0), false)
        .is_some());
    assert!(presym
        .check_proper(&ElementOrder::Int(2), &Vector3::new(1.0, 0.0, 0.0), false)
        .is_some());
    assert!(presym
        .check_proper(&ElementOrder::Int(2), &Vector3::new(1.0, 1.0, 0.0), false)
        .is_some());

    assert!(presym
        .check_improper(
            &ElementOrder::Int(1),
            &Vector3::new(0.0, 0.0, 1.0),
            &INV,
            false
        )
        .is_some());
    assert!(presym
        .check_improper(
            &ElementOrder::Int(1),
            &Vector3::new(0.0, 0.0, 1.0),
            &SIG,
            false
        )
        .is_some());
    assert!(presym
        .check_improper(
            &ElementOrder::Int(1),
            &Vector3::new(0.0, 1.0, 0.0),
            &SIG,
            false
        )
        .is_some());
    assert!(presym
        .check_improper(
            &ElementOrder::Int(1),
            &Vector3::new(1.0, 1.0, 0.0),
            &SIG,
            false
        )
        .is_some());
}
