use std::collections::HashSet;
use nalgebra::Vector3;

use crate::symmetry::symmetry_element::{SymmetryElement, SymmetryElementKind};

#[test]
fn test_finite_symmetry_element_constructor() {
    // ========================
    // Proper symmetry elements
    // ========================
    let c1 = SymmetryElement::builder()
        .threshold(1e-14)
        .order(1.0)
        .axis(Vector3::new(0.0, 2.0, 0.0))
        .kind(SymmetryElementKind::Proper)
        .build()
        .unwrap();
    assert_eq!(format!("{}", &c1), "C1");

    let c2 = SymmetryElement::builder()
        .threshold(1e-14)
        .order(2.0)
        .axis(Vector3::new(1.0, 1.0, 0.0))
        .kind(SymmetryElementKind::Proper)
        .build()
        .unwrap();
    assert_eq!(format!("{}", &c2), "C2(+0.707, +0.707, +0.000)");

    let c35 = SymmetryElement::builder()
        .threshold(1e-14)
        .order(3.5)
        .axis(Vector3::new(1.0, 1.0, 1.0))
        .kind(SymmetryElementKind::Proper)
        .build()
        .unwrap();
    assert_eq!(format!("{}", &c35), "C3.500(+0.577, +0.577, +0.577)");

    let ci = SymmetryElement::builder()
        .threshold(1e-14)
        .order(-1.0)
        .axis(Vector3::new(1.0, 0.0, -1.0))
        .kind(SymmetryElementKind::Proper)
        .build()
        .unwrap();
    assert_eq!(format!("{}", &ci), "C∞(+0.707, +0.000, -0.707)");

    // ==========================
    // Improper symmetry elements
    // ==========================
    let s1 = SymmetryElement::builder()
        .threshold(1e-14)
        .order(1.0)
        .axis(Vector3::new(0.0, 2.0, 0.0))
        .kind(SymmetryElementKind::ImproperMirrorPlane)
        .build()
        .unwrap();
    assert_eq!(format!("{}", &s1), "σ(+0.000, +1.000, +0.000)");

    let sd2 = SymmetryElement::builder()
        .threshold(1e-14)
        .order(2.0)
        .axis(Vector3::new(-1.0, 1.0, 0.0))
        .kind(SymmetryElementKind::ImproperInversionCentre)
        .build()
        .unwrap();
    assert_eq!(format!("{}", &sd2), "σ(-0.707, +0.707, +0.000)");

    let s2 = SymmetryElement::builder()
        .threshold(1e-14)
        .order(2.0)
        .axis(Vector3::new(2.0, 2.0, 1.0))
        .kind(SymmetryElementKind::ImproperMirrorPlane)
        .build()
        .unwrap();
    assert_eq!(format!("{}", &s2), "i");

    let sd1 = SymmetryElement::builder()
        .threshold(1e-14)
        .order(1.0)
        .axis(Vector3::new(1.0, 1.0, 1.0))
        .kind(SymmetryElementKind::ImproperInversionCentre)
        .build()
        .unwrap();
    assert_eq!(format!("{}", &sd1), "i");

    let s3 = SymmetryElement::builder()
        .threshold(1e-14)
        .order(3.0)
        .axis(Vector3::new(2.0, 2.0, 1.0))
        .kind(SymmetryElementKind::ImproperMirrorPlane)
        .build()
        .unwrap();
    assert_eq!(format!("{}", &s3), "S3(+0.667, +0.667, +0.333)");

    let sd3 = SymmetryElement::builder()
        .threshold(1e-14)
        .order(3.0)
        .axis(Vector3::new(1.0, 1.0, 1.0))
        .kind(SymmetryElementKind::ImproperInversionCentre)
        .build()
        .unwrap();
    assert_eq!(format!("{}", &sd3), "Ṡ3(+0.577, +0.577, +0.577)");
}

#[test]
fn test_finite_symmetry_element_improper_conversion() {
    let s1 = SymmetryElement::builder()
        .threshold(1e-14)
        .order(1.0)
        .axis(Vector3::new(0.0, 2.0, 0.0))
        .kind(SymmetryElementKind::ImproperMirrorPlane)
        .build()
        .unwrap();
    let sd2 = s1.convert_to_improper_kind(&SymmetryElementKind::ImproperInversionCentre);
    assert_eq!(format!("{}", &sd2), "σ(+0.000, -1.000, +0.000)");

    let sd1 = SymmetryElement::builder()
        .threshold(1e-14)
        .order(1.0)
        .axis(Vector3::new(1.0, 1.0, 1.0))
        .kind(SymmetryElementKind::ImproperInversionCentre)
        .build()
        .unwrap();
    let s2 = sd1.convert_to_improper_kind(&SymmetryElementKind::ImproperMirrorPlane);
    assert_eq!(format!("{}", &s2), "i");

    let s3 = SymmetryElement::builder()
        .threshold(1e-14)
        .order(3.0)
        .axis(Vector3::new(2.0, 2.0, 1.0))
        .kind(SymmetryElementKind::ImproperMirrorPlane)
        .build()
        .unwrap();
    let sd6 = s3.convert_to_improper_kind(&SymmetryElementKind::ImproperInversionCentre);
    assert_eq!(format!("{}", &sd6), "Ṡ6(-0.667, -0.667, -0.333)");

    let s4 = SymmetryElement::builder()
        .threshold(1e-14)
        .order(4.0)
        .axis(Vector3::new(1.0, 1.0, 1.0))
        .kind(SymmetryElementKind::ImproperMirrorPlane)
        .build()
        .unwrap();
    let sd4 = s4.convert_to_improper_kind(&SymmetryElementKind::ImproperInversionCentre);
    assert_eq!(format!("{}", &sd4), "Ṡ4(-0.577, -0.577, -0.577)");
}

#[test]
fn test_finite_symmetry_element_comparison() {
    // ===========
    // Proper only
    // ===========
    let c1 = SymmetryElement::builder()
        .threshold(1e-14)
        .order(1.0)
        .axis(Vector3::new(0.0, 2.0, 0.0))
        .kind(SymmetryElementKind::Proper)
        .build()
        .unwrap();
    let c1p = SymmetryElement::builder()
        .threshold(1e-14)
        .order(1.0)
        .axis(Vector3::new(1.0, 2.0, 0.0))
        .kind(SymmetryElementKind::Proper)
        .build()
        .unwrap();
    assert_eq!(c1, c1p);

    let c2 = SymmetryElement::builder()
        .threshold(1e-14)
        .order(2.0)
        .axis(Vector3::new(1.0, 2.0, 0.0))
        .kind(SymmetryElementKind::Proper)
        .build()
        .unwrap();
    let c2p = SymmetryElement::builder()
        .threshold(1e-14)
        .order(2.0)
        .axis(-Vector3::new(1.0, 2.0, 0.0))
        .kind(SymmetryElementKind::Proper)
        .build()
        .unwrap();
    assert_eq!(c2, c2p);
    assert_ne!(c1, c2);

    let c3 = SymmetryElement::builder()
        .threshold(1e-14)
        .order(3.0)
        .axis(Vector3::new(1.0, 2.0, 0.0))
        .kind(SymmetryElementKind::Proper)
        .build()
        .unwrap();
    let c3p = SymmetryElement::builder()
        .threshold(1e-14)
        .order(3.0)
        .axis(-Vector3::new(1.0, 2.0, 0.0))
        .kind(SymmetryElementKind::Proper)
        .build()
        .unwrap();
    assert_eq!(c3, c3p);

    // =============
    // Improper only
    // =============
    let s1 = SymmetryElement::builder()
        .threshold(1e-14)
        .order(1.0)
        .axis(Vector3::new(0.0, 2.0, 0.0))
        .kind(SymmetryElementKind::ImproperMirrorPlane)
        .build()
        .unwrap();
    let sd2 = s1.convert_to_improper_kind(&SymmetryElementKind::ImproperInversionCentre);
    assert_eq!(s1, sd2);

    let sd1 = SymmetryElement::builder()
        .threshold(1e-14)
        .order(1.0)
        .axis(Vector3::new(1.0, 1.0, 1.0))
        .kind(SymmetryElementKind::ImproperInversionCentre)
        .build()
        .unwrap();
    let s2 = sd1.convert_to_improper_kind(&SymmetryElementKind::ImproperMirrorPlane);
    assert_eq!(sd1, s2);
    assert_ne!(sd1, sd2);
    assert_ne!(s1, s2);

    let s3 = SymmetryElement::builder()
        .threshold(1e-14)
        .order(3.0)
        .axis(Vector3::new(2.0, 2.0, 1.0))
        .kind(SymmetryElementKind::ImproperMirrorPlane)
        .build()
        .unwrap();
    let sd6 = s3.convert_to_improper_kind(&SymmetryElementKind::ImproperInversionCentre);
    assert_eq!(s3, sd6);

    let s3p = SymmetryElement::builder()
        .threshold(1e-14)
        .order(3.0)
        .axis(-Vector3::new(2.0, 2.0, 1.0))
        .kind(SymmetryElementKind::ImproperMirrorPlane)
        .build()
        .unwrap();
    assert_eq!(s3, s3p);
    assert_ne!(s2, s3);

    // ===================
    // Proper and improper
    // ===================
    assert_ne!(c2, s2);
    assert_ne!(c2, sd2);
}

#[test]
fn test_finite_symmetry_element_hashset() {
    let mut element_set = HashSet::new();
    let c1 = SymmetryElement::builder()
        .threshold(1e-14)
        .order(1.0)
        .axis(Vector3::new(0.0, 2.0, 0.0))
        .kind(SymmetryElementKind::Proper)
        .build()
        .unwrap();
    element_set.insert(c1);
    assert_eq!(element_set.len(), 1);

    let c1p = SymmetryElement::builder()
        .threshold(1e-14)
        .order(1.0)
        .axis(Vector3::new(1.0, 2.0, 0.0))
        .kind(SymmetryElementKind::Proper)
        .build()
        .unwrap();
    element_set.insert(c1p);
    assert_eq!(element_set.len(), 1);

    let c2 = SymmetryElement::builder()
        .threshold(1e-14)
        .order(2.0)
        .axis(Vector3::new(1.0, 2.0, 0.0))
        .kind(SymmetryElementKind::Proper)
        .build()
        .unwrap();
    element_set.insert(c2);
    assert_eq!(element_set.len(), 2);

    let c2p = SymmetryElement::builder()
        .threshold(1e-14)
        .order(2.0)
        .axis(-Vector3::new(1.0, 2.0, 0.0))
        .kind(SymmetryElementKind::Proper)
        .build()
        .unwrap();
    element_set.insert(c2p);
    assert_eq!(element_set.len(), 2);

    let c3 = SymmetryElement::builder()
        .threshold(1e-14)
        .order(3.0)
        .axis(Vector3::new(1.0, 2.0, 0.0))
        .kind(SymmetryElementKind::Proper)
        .build()
        .unwrap();
    element_set.insert(c3);
    assert_eq!(element_set.len(), 3);

    let c3p = SymmetryElement::builder()
        .threshold(1e-14)
        .order(3.0)
        .axis(-Vector3::new(1.0, 2.0, 0.0))
        .kind(SymmetryElementKind::Proper)
        .build()
        .unwrap();
    element_set.insert(c3p);
    assert_eq!(element_set.len(), 3);

    let s1 = SymmetryElement::builder()
        .threshold(1e-14)
        .order(1.0)
        .axis(Vector3::new(0.0, 2.0, 0.0))
        .kind(SymmetryElementKind::ImproperMirrorPlane)
        .build()
        .unwrap();
    let sd2 = s1.convert_to_improper_kind(&SymmetryElementKind::ImproperInversionCentre);
    element_set.insert(s1);
    assert_eq!(element_set.len(), 4);
    element_set.insert(sd2);
    assert_eq!(element_set.len(), 4);

    let sd1 = SymmetryElement::builder()
        .threshold(1e-14)
        .order(1.0)
        .axis(Vector3::new(1.0, 1.0, 1.0))
        .kind(SymmetryElementKind::ImproperInversionCentre)
        .build()
        .unwrap();
    let s2 = sd1.convert_to_improper_kind(&SymmetryElementKind::ImproperMirrorPlane);
    element_set.insert(sd1);
    assert_eq!(element_set.len(), 5);
    element_set.insert(s2);
    assert_eq!(element_set.len(), 5);

    let s3 = SymmetryElement::builder()
        .threshold(1e-14)
        .order(3.0)
        .axis(Vector3::new(2.0, 2.0, 1.0))
        .kind(SymmetryElementKind::ImproperMirrorPlane)
        .build()
        .unwrap();
    let sd6 = s3.convert_to_improper_kind(&SymmetryElementKind::ImproperInversionCentre);
    element_set.insert(s3);
    assert_eq!(element_set.len(), 6);
    element_set.insert(sd6);
    assert_eq!(element_set.len(), 6);

    let s3p = SymmetryElement::builder()
        .threshold(1e-14)
        .order(3.0)
        .axis(-Vector3::new(2.0, 2.0, 1.0))
        .kind(SymmetryElementKind::ImproperMirrorPlane)
        .build()
        .unwrap();
    element_set.insert(s3p);
    assert_eq!(element_set.len(), 6);
}
