use crate::aux::misc;
use std::collections::HashSet;
use nalgebra::Vector3;

use crate::symmetry::symmetry_element::{SymmetryElement, SymmetryElementKind, ElementOrder};


#[test]
fn test_element_order_equality() {
    let order_1a = ElementOrder::new(1.0, 1e-14);
    let order_1b = ElementOrder::Int(1);
    assert_eq!(order_1a, order_1b);
    assert_eq!(misc::calculate_hash(&order_1a), misc::calculate_hash(&order_1b));

    let order_35a = ElementOrder::new(3.5, 1e-14);
    let order_35b = ElementOrder::Float(3.5, 1e-7);
    assert_eq!(order_35a, order_35b);
    assert_eq!(misc::calculate_hash(&order_35a), misc::calculate_hash(&order_35b));
    assert_ne!(order_35a, order_1b);

    let order_ia = ElementOrder::new(f64::INFINITY, 1e-14);
    let order_ib = ElementOrder::Inf;
    assert_eq!(order_ia, order_ib);
    assert_eq!(misc::calculate_hash(&order_ia), misc::calculate_hash(&order_ib));
    assert_ne!(order_ia, order_1b);
    assert_ne!(order_ia, order_35b);
}


#[test]
fn test_element_order_comparison() {
    let order_1 = ElementOrder::Int(1);
    let order_2 = ElementOrder::new(2.0, 1e-14);
    let order_35 = ElementOrder::new(3.5, 1e-14);
    let order_i = ElementOrder::Inf;

    assert!(order_1 < order_2);
    assert!(order_35 > order_2);
    assert!(order_i > order_35);
    assert!(order_1 < order_i);
}


#[test]
fn test_element_order_hashability() {
    let order_1a = ElementOrder::new(1.0, 1e-14);
    let order_1b = ElementOrder::Int(1);
    let order_2a = ElementOrder::new(2.0, 1e-14);
    let order_2b = ElementOrder::Int(2);
    let order_35a = ElementOrder::new(3.5, 1e-14);
    let order_35b = ElementOrder::Float(3.5, 1e-7);
    let order_ia = ElementOrder::new(f64::INFINITY, 1e-14);
    let order_ib = ElementOrder::Inf;

    let mut orders: HashSet<ElementOrder> = HashSet::new();
    orders.insert(order_1a);
    assert_eq!(orders.len(), 1);
    orders.insert(order_1b);
    assert_eq!(orders.len(), 1);

    orders.insert(order_2a);
    assert_eq!(orders.len(), 2);
    orders.insert(order_2b);
    assert_eq!(orders.len(), 2);

    orders.insert(order_35a);
    assert_eq!(orders.len(), 3);
    orders.insert(order_35b);
    assert_eq!(orders.len(), 3);

    orders.insert(order_ia);
    assert_eq!(orders.len(), 4);
    orders.insert(order_ib);
    assert_eq!(orders.len(), 4);
}

#[test]
fn test_finite_symmetry_element_constructor() {
    // ========================
    // Proper symmetry elements
    // ========================
    let c1 = SymmetryElement::builder()
        .threshold(1e-14)
        .order(ElementOrder::Int(1))
        .axis(Vector3::new(0.0, 2.0, 0.0))
        .kind(SymmetryElementKind::Proper)
        .build()
        .unwrap();
    assert_eq!(format!("{}", &c1), "C1");

    let c2 = SymmetryElement::builder()
        .threshold(1e-14)
        .order(ElementOrder::Int(2))
        .axis(Vector3::new(1.0, 1.0, 0.0))
        .kind(SymmetryElementKind::Proper)
        .build()
        .unwrap();
    assert_eq!(format!("{}", &c2), "C2(+0.707, +0.707, +0.000)");

    let c35 = SymmetryElement::builder()
        .threshold(1e-14)
        .order(ElementOrder::new(3.5, 1e-14))
        .axis(Vector3::new(1.0, 1.0, 1.0))
        .kind(SymmetryElementKind::Proper)
        .build()
        .unwrap();
    assert_eq!(format!("{}", &c35), "C3.500(+0.577, +0.577, +0.577)");

    let ci = SymmetryElement::builder()
        .threshold(1e-14)
        .order(ElementOrder::Inf)
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
        .order(ElementOrder::Int(1))
        .axis(Vector3::new(0.0, 2.0, 0.0))
        .kind(SymmetryElementKind::ImproperMirrorPlane)
        .build()
        .unwrap();
    assert_eq!(format!("{}", &s1), "σ(+0.000, +1.000, +0.000)");

    let sd2 = SymmetryElement::builder()
        .threshold(1e-14)
        .order(ElementOrder::Int(2))
        .axis(Vector3::new(-1.0, 1.0, 0.0))
        .kind(SymmetryElementKind::ImproperInversionCentre)
        .build()
        .unwrap();
    assert_eq!(format!("{}", &sd2), "σ(-0.707, +0.707, +0.000)");

    let s2 = SymmetryElement::builder()
        .threshold(1e-14)
        .order(ElementOrder::Int(2))
        .axis(Vector3::new(2.0, 2.0, 1.0))
        .kind(SymmetryElementKind::ImproperMirrorPlane)
        .build()
        .unwrap();
    assert_eq!(format!("{}", &s2), "i");

    let sd1 = SymmetryElement::builder()
        .threshold(1e-14)
        .order(ElementOrder::Int(1))
        .axis(Vector3::new(1.0, 1.0, 1.0))
        .kind(SymmetryElementKind::ImproperInversionCentre)
        .build()
        .unwrap();
    assert_eq!(format!("{}", &sd1), "i");
    assert_eq!(s2, sd1);

    let s3 = SymmetryElement::builder()
        .threshold(1e-14)
        .order(ElementOrder::Int(3))
        .axis(Vector3::new(2.0, 2.0, 1.0))
        .kind(SymmetryElementKind::ImproperMirrorPlane)
        .build()
        .unwrap();
    assert_eq!(format!("{}", &s3), "S3(+0.667, +0.667, +0.333)");

    let sd3 = SymmetryElement::builder()
        .threshold(1e-14)
        .order(ElementOrder::Int(3))
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
        .order(ElementOrder::Int(1))
        .axis(Vector3::new(0.0, 2.0, 0.0))
        .kind(SymmetryElementKind::ImproperMirrorPlane)
        .build()
        .unwrap();
    let sd2 = s1.convert_to_improper_kind(&SymmetryElementKind::ImproperInversionCentre);
    assert_eq!(format!("{}", &sd2), "σ(+0.000, -1.000, +0.000)");

    let sd1 = SymmetryElement::builder()
        .threshold(1e-14)
        .order(ElementOrder::Int(1))
        .axis(Vector3::new(1.0, 1.0, 1.0))
        .kind(SymmetryElementKind::ImproperInversionCentre)
        .build()
        .unwrap();
    let s2 = sd1.convert_to_improper_kind(&SymmetryElementKind::ImproperMirrorPlane);
    assert_eq!(format!("{}", &s2), "i");

    let s3 = SymmetryElement::builder()
        .threshold(1e-14)
        .order(ElementOrder::Int(3))
        .axis(Vector3::new(2.0, 2.0, 1.0))
        .kind(SymmetryElementKind::ImproperMirrorPlane)
        .build()
        .unwrap();
    let sd6 = s3.convert_to_improper_kind(&SymmetryElementKind::ImproperInversionCentre);
    assert_eq!(format!("{}", &sd6), "Ṡ6(-0.667, -0.667, -0.333)");

    let s4 = SymmetryElement::builder()
        .threshold(1e-14)
        .order(ElementOrder::Int(4))
        .axis(Vector3::new(1.0, 1.0, 1.0))
        .kind(SymmetryElementKind::ImproperMirrorPlane)
        .build()
        .unwrap();
    let sd4 = s4.convert_to_improper_kind(&SymmetryElementKind::ImproperInversionCentre);
    assert_eq!(sd4.order, ElementOrder::Int(4));
    assert_eq!(format!("{}", &sd4), "Ṡ4(-0.577, -0.577, -0.577)");

    let sd5 = SymmetryElement::builder()
        .threshold(1e-14)
        .order(ElementOrder::Int(5))
        .axis(Vector3::new(1.0, 2.0, 1.0))
        .kind(SymmetryElementKind::ImproperInversionCentre)
        .build()
        .unwrap();
    let s10 = sd5.convert_to_improper_kind(&SymmetryElementKind::ImproperMirrorPlane);
    assert_eq!(s10.order, ElementOrder::Int(10));
    assert_eq!(format!("{}", &s10), "S10(-0.408, -0.816, -0.408)");

    let sd7 = SymmetryElement::builder()
        .threshold(1e-14)
        .order(ElementOrder::Int(7))
        .axis(Vector3::new(1.0, 1.0, 1.0))
        .kind(SymmetryElementKind::ImproperInversionCentre)
        .build()
        .unwrap();
    let s14 = sd7.convert_to_improper_kind(&SymmetryElementKind::ImproperMirrorPlane);
    assert_eq!(s14.order, ElementOrder::Int(14));
}

#[test]
fn test_finite_symmetry_element_comparison() {
    // ===========
    // Proper only
    // ===========
    let c1 = SymmetryElement::builder()
        .threshold(1e-14)
        .order(ElementOrder::Int(1))
        .axis(Vector3::new(0.0, 2.0, 0.0))
        .kind(SymmetryElementKind::Proper)
        .build()
        .unwrap();
    let c1p = SymmetryElement::builder()
        .threshold(1e-14)
        .order(ElementOrder::Int(1))
        .axis(Vector3::new(1.0, 2.0, 0.0))
        .kind(SymmetryElementKind::Proper)
        .build()
        .unwrap();
    assert_eq!(c1, c1p);

    let c2 = SymmetryElement::builder()
        .threshold(1e-14)
        .order(ElementOrder::Int(2))
        .axis(Vector3::new(1.0, 2.0, 0.0))
        .kind(SymmetryElementKind::Proper)
        .build()
        .unwrap();
    let c2p = SymmetryElement::builder()
        .threshold(1e-14)
        .order(ElementOrder::Int(2))
        .axis(-Vector3::new(1.0, 2.0, 0.0))
        .kind(SymmetryElementKind::Proper)
        .build()
        .unwrap();
    assert_eq!(c2, c2p);
    assert_ne!(c1, c2);

    let c3 = SymmetryElement::builder()
        .threshold(1e-14)
        .order(ElementOrder::Int(3))
        .axis(Vector3::new(1.0, 2.0, 0.0))
        .kind(SymmetryElementKind::Proper)
        .build()
        .unwrap();
    let c3p = SymmetryElement::builder()
        .threshold(1e-14)
        .order(ElementOrder::Int(3))
        .axis(-Vector3::new(1.0, 2.0, 0.0))
        .kind(SymmetryElementKind::Proper)
        .build()
        .unwrap();
    assert_eq!(c3, c3p);

    // =============
    // Improper only
    // =============
    let s1 = SymmetryElement::builder()
        .threshold(1e-3)
        .order(ElementOrder::Int(1))
        .axis(Vector3::new(0.0, 2.0, 0.0))
        .kind(SymmetryElementKind::ImproperMirrorPlane)
        .build()
        .unwrap();
    let sd2 = s1.convert_to_improper_kind(&SymmetryElementKind::ImproperInversionCentre);
    assert_eq!(s1, sd2);

    let sd1 = SymmetryElement::builder()
        .threshold(1e-14)
        .order(ElementOrder::Int(1))
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
        .order(ElementOrder::Int(3))
        .axis(Vector3::new(2.0, 2.0, 1.0))
        .kind(SymmetryElementKind::ImproperMirrorPlane)
        .build()
        .unwrap();
    let sd6 = s3.convert_to_improper_kind(&SymmetryElementKind::ImproperInversionCentre);
    assert_eq!(sd6.order, ElementOrder::Int(6));
    assert_eq!(s3, sd6);

    let s3p = SymmetryElement::builder()
        .threshold(1e-14)
        .order(ElementOrder::Int(3))
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
        .order(ElementOrder::Int(1))
        .axis(Vector3::new(0.0, 2.0, 0.0))
        .kind(SymmetryElementKind::Proper)
        .build()
        .unwrap();
    element_set.insert(c1);
    assert_eq!(element_set.len(), 1);

    let c1p = SymmetryElement::builder()
        .threshold(1e-14)
        .order(ElementOrder::Int(1))
        .axis(Vector3::new(1.0, 2.0, 0.0))
        .kind(SymmetryElementKind::Proper)
        .build()
        .unwrap();
    element_set.insert(c1p);
    assert_eq!(element_set.len(), 1);

    let c2 = SymmetryElement::builder()
        .threshold(1e-14)
        .order(ElementOrder::Int(2))
        .axis(Vector3::new(1.0, 2.0, 0.0))
        .kind(SymmetryElementKind::Proper)
        .build()
        .unwrap();
    element_set.insert(c2);
    assert_eq!(element_set.len(), 2);

    let c2p = SymmetryElement::builder()
        .threshold(1e-14)
        .order(ElementOrder::Int(2))
        .axis(-Vector3::new(1.0, 2.0, 0.0))
        .kind(SymmetryElementKind::Proper)
        .build()
        .unwrap();
    element_set.insert(c2p);
    assert_eq!(element_set.len(), 2);

    let c3 = SymmetryElement::builder()
        .threshold(1e-14)
        .order(ElementOrder::Int(3))
        .axis(Vector3::new(1.0, 2.0, 0.0))
        .kind(SymmetryElementKind::Proper)
        .build()
        .unwrap();
    element_set.insert(c3);
    assert_eq!(element_set.len(), 3);

    let c3p = SymmetryElement::builder()
        .threshold(1e-14)
        .order(ElementOrder::Int(3))
        .axis(-Vector3::new(1.0, 2.0, 0.0))
        .kind(SymmetryElementKind::Proper)
        .build()
        .unwrap();
    element_set.insert(c3p);
    assert_eq!(element_set.len(), 3);

    let s1 = SymmetryElement::builder()
        .threshold(1e-14)
        .order(ElementOrder::Int(1))
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
        .order(ElementOrder::Int(1))
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
        .order(ElementOrder::Int(3))
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
        .order(ElementOrder::Int(3))
        .axis(-Vector3::new(2.0, 2.0, 1.0))
        .kind(SymmetryElementKind::ImproperMirrorPlane)
        .build()
        .unwrap();
    element_set.insert(s3p);
    assert_eq!(element_set.len(), 6);
}
