use std::collections::HashSet;
use nalgebra::Vector3;

use crate::aux::misc;
use crate::symmetry::symmetry_element::{ElementOrder, SymmetryElement, SymmetryElementKind};

#[test]
fn test_element_order_equality() {
    let order_1a = ElementOrder::new(1.0, 1e-14);
    let order_1b = ElementOrder::Int(1);
    assert_eq!(order_1a, order_1b);
    assert_eq!(
        misc::calculate_hash(&order_1a),
        misc::calculate_hash(&order_1b)
    );

    let order_ia = ElementOrder::new(f64::INFINITY, 1e-14);
    let order_ib = ElementOrder::Inf;
    assert_eq!(order_ia, order_ib);
    assert_eq!(
        misc::calculate_hash(&order_ia),
        misc::calculate_hash(&order_ib)
    );
    assert_ne!(order_ia, order_1b);
}

#[test]
fn test_element_order_comparison() {
    let order_1 = ElementOrder::Int(1);
    let order_2 = ElementOrder::new(2.0, 1e-14);
    let order_3 = ElementOrder::new(3.0, 1e-14);
    let order_i = ElementOrder::Inf;
    let order_ib = ElementOrder::new(f64::INFINITY, 1e-14);

    assert!(order_1 < order_2);
    assert!(order_3 > order_2);
    assert!(order_i > order_3);
    assert!(order_1 < order_i);
    assert!(order_i == order_ib);
}

#[test]
fn test_element_order_hashability() {
    let order_1a = ElementOrder::new(1.0, 1e-14);
    let order_1b = ElementOrder::Int(1);
    let order_2a = ElementOrder::new(2.0, 1e-14);
    let order_2b = ElementOrder::Int(2);
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

    orders.insert(order_ia);
    assert_eq!(orders.len(), 3);
    orders.insert(order_ib);
    assert_eq!(orders.len(), 3);
}

#[test]
fn test_symmetry_element_constructor() {
    // ========================
    // Proper symmetry elements
    // ========================
    let c1 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(1))
        .proper_power(1)
        .axis(Vector3::new(0.0, 2.0, 0.0))
        .kind(SymmetryElementKind::Proper)
        .build()
        .unwrap();
    assert_eq!(format!("{}", &c1), "E");
    assert_eq!(format!("{:?}", &c1), "C1(+0.000, +1.000, +0.000)");

    let c2 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(2))
        .proper_power(1)
        .axis(Vector3::new(1.0, 1.0, 0.0))
        .kind(SymmetryElementKind::Proper)
        .build()
        .unwrap();
    assert_eq!(format!("{}", &c2), "C2(+0.707, +0.707, +0.000)");
    assert_eq!(format!("{:?}", &c2), "C2(+0.707, +0.707, +0.000)");

    let c2p2 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(2))
        .proper_power(2)
        .axis(Vector3::new(1.0, 1.0, 0.0))
        .kind(SymmetryElementKind::Proper)
        .build()
        .unwrap();
    assert_eq!(format!("{}", &c2p2), "E");
    assert_eq!(format!("{:?}", &c2p2), "C2^2(+0.707, +0.707, +0.000)");

    let c3 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::new(3.0, 1e-14))
        .proper_power(1)
        .axis(Vector3::new(1.0, 1.0, 1.0))
        .kind(SymmetryElementKind::Proper)
        .build()
        .unwrap();
    assert_eq!(format!("{}", &c3), "C3(+0.577, +0.577, +0.577)");
    assert_eq!(format!("{:?}", &c3), "C3(+0.577, +0.577, +0.577)");

    let c3p2 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::new(3.0, 1e-14))
        .proper_power(2)
        .axis(Vector3::new(1.0, 1.0, 1.0))
        .kind(SymmetryElementKind::Proper)
        .build()
        .unwrap();
    assert_eq!(format!("{}", &c3p2), "C3^2(+0.577, +0.577, +0.577)");
    assert_eq!(format!("{:?}", &c3p2), "C3^2(+0.577, +0.577, +0.577)");

    let c3p3 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::new(3.0, 1e-14))
        .proper_power(3)
        .axis(Vector3::new(1.0, 1.0, 1.0))
        .kind(SymmetryElementKind::Proper)
        .build()
        .unwrap();
    assert_eq!(format!("{}", &c3p3), "E");
    assert_eq!(format!("{:?}", &c3p3), "C3^3(+0.577, +0.577, +0.577)");

    let ci = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Inf)
        .axis(Vector3::new(1.0, 0.0, -1.0))
        .kind(SymmetryElementKind::Proper)
        .build()
        .unwrap();
    assert_eq!(format!("{}", &ci), "C∞(+0.707, +0.000, -0.707)");
    assert_eq!(format!("{:?}", &ci), "C∞(+0.707, +0.000, -0.707)");

    let ci2 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Inf)
        .axis(Vector3::new(1.0, 0.0, 1.0))
        .proper_angle(0.12)
        .kind(SymmetryElementKind::Proper)
        .build()
        .unwrap();
    assert_eq!(format!("{}", &ci2), "C∞(+0.120)(+0.707, +0.000, +0.707)");
    assert_eq!(format!("{:?}", &ci2), "C∞(+0.120)(+0.707, +0.000, +0.707)");

    let ci3 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Inf)
        .axis(Vector3::new(1.0, 0.0, 1.0))
        .proper_angle(3.160)
        .kind(SymmetryElementKind::Proper)
        .build()
        .unwrap();
    assert_eq!(format!("{}", &ci3), "C∞(-3.123)(+0.707, +0.000, +0.707)");
    assert_eq!(format!("{:?}", &ci3), "C∞(-3.123)(+0.707, +0.000, +0.707)");

    // ==========================
    // Improper symmetry elements
    // ==========================
    let s1 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(1))
        .proper_power(1)
        .axis(Vector3::new(0.0, 2.0, 0.0))
        .kind(SymmetryElementKind::ImproperMirrorPlane)
        .build()
        .unwrap();
    assert_eq!(format!("{}", &s1), "σ(+0.000, +1.000, +0.000)");
    assert_eq!(format!("{:?}", &s1), "S1(+0.000, +1.000, +0.000)");

    let sd2 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(2))
        .proper_power(1)
        .axis(Vector3::new(-1.0, 1.0, 0.0))
        .kind(SymmetryElementKind::ImproperInversionCentre)
        .build()
        .unwrap();
    assert_eq!(format!("{}", &sd2), "σ(-0.707, +0.707, +0.000)");
    assert_eq!(format!("{:?}", &sd2), "Ṡ2(-0.707, +0.707, +0.000)");

    let sd2p2 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(2))
        .proper_power(2)
        .axis(Vector3::new(-1.0, 1.0, 0.0))
        .kind(SymmetryElementKind::ImproperInversionCentre)
        .build()
        .unwrap();
    assert_eq!(format!("{}", &sd2p2), "i");
    assert_eq!(format!("{:?}", &sd2p2), "iC2^2(-0.707, +0.707, +0.000)");

    let s2 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(2))
        .proper_power(1)
        .axis(Vector3::new(2.0, 2.0, 1.0))
        .kind(SymmetryElementKind::ImproperMirrorPlane)
        .build()
        .unwrap();
    assert_eq!(format!("{}", &s2), "i");
    assert_eq!(format!("{:?}", &s2), "S2(+0.667, +0.667, +0.333)");

    let s2p2 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(2))
        .proper_power(2)
        .axis(Vector3::new(2.0, 2.0, 1.0))
        .kind(SymmetryElementKind::ImproperMirrorPlane)
        .build()
        .unwrap();
    assert_eq!(format!("{}", &s2p2), "σ(+0.667, +0.667, +0.333)");
    assert_eq!(format!("{:?}", &s2p2), "σC2^2(+0.667, +0.667, +0.333)");

    let sd1 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(1))
        .proper_power(1)
        .axis(Vector3::new(1.0, 1.0, 1.0))
        .kind(SymmetryElementKind::ImproperInversionCentre)
        .build()
        .unwrap();
    assert_eq!(format!("{}", &sd1), "i");
    assert_eq!(format!("{:?}", &sd1), "Ṡ1(+0.577, +0.577, +0.577)");
    assert_eq!(s2, sd1);

    let s3 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(3))
        .proper_power(1)
        .axis(Vector3::new(2.0, 2.0, 1.0))
        .kind(SymmetryElementKind::ImproperMirrorPlane)
        .build()
        .unwrap();
    assert_eq!(format!("{}", &s3), "S3(+0.667, +0.667, +0.333)");
    assert_eq!(format!("{:?}", &s3), "S3(+0.667, +0.667, +0.333)");

    let s3p2 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(3))
        .proper_power(2)
        .axis(Vector3::new(2.0, 2.0, 1.0))
        .kind(SymmetryElementKind::ImproperMirrorPlane)
        .build()
        .unwrap();
    assert_eq!(format!("{}", &s3p2), "σC3^2(+0.667, +0.667, +0.333)");
    assert_eq!(format!("{:?}", &s3p2), "σC3^2(+0.667, +0.667, +0.333)");

    let s3p3 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(3))
        .proper_power(3)
        .axis(Vector3::new(2.0, 2.0, 1.0))
        .kind(SymmetryElementKind::ImproperMirrorPlane)
        .build()
        .unwrap();
    assert_eq!(format!("{}", &s3p3), "σ(+0.667, +0.667, +0.333)");
    assert_eq!(format!("{:?}", &s3p3), "σC3^3(+0.667, +0.667, +0.333)");

    let sd3 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(3))
        .proper_power(1)
        .axis(Vector3::new(1.0, 1.0, 1.0))
        .kind(SymmetryElementKind::ImproperInversionCentre)
        .build()
        .unwrap();
    assert_eq!(format!("{}", &sd3), "Ṡ3(+0.577, +0.577, +0.577)");
    assert_eq!(format!("{:?}", &sd3), "Ṡ3(+0.577, +0.577, +0.577)");

    let sd3p2 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(3))
        .proper_power(2)
        .axis(Vector3::new(1.0, 1.0, 1.0))
        .kind(SymmetryElementKind::ImproperInversionCentre)
        .build()
        .unwrap();
    assert_eq!(format!("{}", &sd3p2), "iC3^2(+0.577, +0.577, +0.577)");
    assert_eq!(format!("{:?}", &sd3p2), "iC3^2(+0.577, +0.577, +0.577)");

    let sd3p3 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(3))
        .proper_power(3)
        .axis(Vector3::new(1.0, 1.0, 1.0))
        .kind(SymmetryElementKind::ImproperInversionCentre)
        .build()
        .unwrap();
    assert_eq!(format!("{}", &sd3p3), "i");
    assert_eq!(format!("{:?}", &sd3p3), "iC3^3(+0.577, +0.577, +0.577)");

    let si = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Inf)
        .proper_power(1)
        .axis(Vector3::new(1.0, 0.0, -1.0))
        .kind(SymmetryElementKind::ImproperMirrorPlane)
        .build()
        .unwrap();
    assert_eq!(format!("{}", &si), "S∞(+0.707, +0.000, -0.707)");
    assert_eq!(format!("{:?}", &si), "σC∞(+0.707, +0.000, -0.707)");

    let sib = si.convert_to_improper_kind(&SymmetryElementKind::ImproperInversionCentre, false);
    assert_eq!(format!("{}", &sib), "Ṡ∞(+0.707, +0.000, -0.707)");
    assert_eq!(format!("{:?}", &sib), "iC∞(+0.707, +0.000, -0.707)");

    let si2 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Inf)
        .axis(Vector3::new(1.0, 0.0, 1.0))
        .proper_angle(0.121)
        .kind(SymmetryElementKind::ImproperMirrorPlane)
        .build()
        .unwrap();
    assert_eq!(format!("{}", &si2), "S∞(+0.121)(+0.707, +0.000, +0.707)");
    assert_eq!(format!("{:?}", &si2), "σC∞(+0.121)(+0.707, +0.000, +0.707)");

    let si2b = si2.convert_to_improper_kind(&SymmetryElementKind::ImproperInversionCentre, false);
    assert_eq!(format!("{}", &si2b), "Ṡ∞(-3.021)(+0.707, +0.000, +0.707)");
    assert_eq!(
        format!("{:?}", &si2b),
        "iC∞(-3.021)(+0.707, +0.000, +0.707)"
    );
}

#[test]
fn test_symmetry_element_finite_improper_conversion() {
    let s1 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(1))
        .proper_power(1)
        .axis(Vector3::new(0.0, 2.0, 0.0))
        .kind(SymmetryElementKind::ImproperMirrorPlane)
        .build()
        .unwrap();
    let sd2 = s1.convert_to_improper_kind(&SymmetryElementKind::ImproperInversionCentre, false);
    assert_eq!(format!("{}", &sd2), "σ(+0.000, +1.000, +0.000)");

    let sd1 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(1))
        .proper_power(1)
        .axis(Vector3::new(1.0, 1.0, 1.0))
        .kind(SymmetryElementKind::ImproperInversionCentre)
        .build()
        .unwrap();
    let s2 = sd1.convert_to_improper_kind(&SymmetryElementKind::ImproperMirrorPlane, false);
    assert_eq!(format!("{}", &s2), "i");

    let s3 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(3))
        .proper_power(1)
        .axis(Vector3::new(2.0, 2.0, 1.0))
        .kind(SymmetryElementKind::ImproperMirrorPlane)
        .build()
        .unwrap();
    let sd6 = s3.convert_to_improper_kind(&SymmetryElementKind::ImproperInversionCentre, false);
    assert_eq!(format!("{}", &sd6), "Ṡ6(+0.667, +0.667, +0.333)");

    let s4 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(4))
        .proper_power(1)
        .axis(Vector3::new(1.0, 1.0, 1.0))
        .kind(SymmetryElementKind::ImproperMirrorPlane)
        .build()
        .unwrap();
    let sd4 = s4.convert_to_improper_kind(&SymmetryElementKind::ImproperInversionCentre, false);
    assert_eq!(sd4.proper_order, ElementOrder::Int(4));
    assert_eq!(format!("{}", &sd4), "Ṡ4(+0.577, +0.577, +0.577)");

    let sd5 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(5))
        .proper_power(1)
        .axis(Vector3::new(1.0, 2.0, 1.0))
        .kind(SymmetryElementKind::ImproperInversionCentre)
        .build()
        .unwrap();
    let s10 = sd5.convert_to_improper_kind(&SymmetryElementKind::ImproperMirrorPlane, false);
    assert_eq!(s10.proper_order, ElementOrder::Int(10));
    assert_eq!(format!("{}", &s10), "S10(+0.408, +0.816, +0.408)");

    let sd7 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(7))
        .proper_power(1)
        .axis(Vector3::new(1.0, 1.0, 1.0))
        .kind(SymmetryElementKind::ImproperInversionCentre)
        .build()
        .unwrap();
    let s14 = sd7.convert_to_improper_kind(&SymmetryElementKind::ImproperMirrorPlane, false);
    assert_eq!(s14.proper_order, ElementOrder::Int(14));
}

#[test]
fn test_symmetry_element_finite_comparison() {
    // ===========
    // Proper only
    // ===========
    let c1 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(1))
        .proper_power(1)
        .axis(Vector3::new(0.0, 2.0, 0.0))
        .kind(SymmetryElementKind::Proper)
        .build()
        .unwrap();
    let c1p = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(1))
        .proper_power(1)
        .axis(Vector3::new(1.0, 2.0, 0.0))
        .kind(SymmetryElementKind::Proper)
        .build()
        .unwrap();
    assert_eq!(c1, c1p);

    let c1p2 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(1))
        .proper_power(2)
        .axis(Vector3::new(4.0, 2.0, 0.0))
        .kind(SymmetryElementKind::Proper)
        .build()
        .unwrap();
    assert_eq!(c1, c1p2);

    let c2 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(2))
        .proper_power(1)
        .axis(Vector3::new(1.0, 2.0, 0.0))
        .kind(SymmetryElementKind::Proper)
        .build()
        .unwrap();
    let c2p = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(2))
        .proper_power(1)
        .axis(-Vector3::new(1.0, 2.0, 0.0))
        .kind(SymmetryElementKind::Proper)
        .build()
        .unwrap();
    assert_eq!(c2, c2p);
    assert_ne!(c1, c2);

    let c3 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(3))
        .proper_power(1)
        .axis(Vector3::new(1.0, 2.0, 0.0))
        .kind(SymmetryElementKind::Proper)
        .build()
        .unwrap();
    let c3p = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(3))
        .proper_power(1)
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
        .proper_order(ElementOrder::Int(1))
        .proper_power(1)
        .axis(Vector3::new(0.0, 2.0, 0.0))
        .kind(SymmetryElementKind::ImproperMirrorPlane)
        .build()
        .unwrap();
    let sd2 = s1.convert_to_improper_kind(&SymmetryElementKind::ImproperInversionCentre, false);
    assert_eq!(s1, sd2);

    let sd1 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(1))
        .proper_power(1)
        .axis(Vector3::new(1.0, 1.0, 1.0))
        .kind(SymmetryElementKind::ImproperInversionCentre)
        .build()
        .unwrap();
    let s2 = sd1.convert_to_improper_kind(&SymmetryElementKind::ImproperMirrorPlane, false);
    assert_eq!(sd1, s2);
    assert_ne!(sd1, sd2);
    assert_ne!(s1, s2);

    let s3 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(3))
        .proper_power(1)
        .axis(Vector3::new(2.0, 2.0, 1.0))
        .kind(SymmetryElementKind::ImproperMirrorPlane)
        .build()
        .unwrap();
    let sd6 = s3.convert_to_improper_kind(&SymmetryElementKind::ImproperInversionCentre, false);
    assert_eq!(sd6.proper_order, ElementOrder::Int(6));
    assert_eq!(s3, sd6);

    let s3p = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(3))
        .proper_power(1)
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
fn test_symmetry_element_finite_power_comparison() {
    // ===========
    // Proper only
    // ===========
    let c1 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(1))
        .proper_power(1)
        .axis(Vector3::new(0.0, 2.0, 0.0))
        .kind(SymmetryElementKind::Proper)
        .build()
        .unwrap();
    let c1p = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(1))
        .proper_power(2)
        .axis(Vector3::new(1.0, 2.0, 0.0))
        .kind(SymmetryElementKind::Proper)
        .build()
        .unwrap();
    let c1p2 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(1))
        .proper_power(0)
        .axis(-Vector3::new(1.0, 2.0, 0.0))
        .kind(SymmetryElementKind::Proper)
        .build()
        .unwrap();
    assert_eq!(c1, c1p);
    assert_eq!(c1, c1p2);
    assert_eq!(c1p, c1p2);

    let c2 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(2))
        .proper_power(1)
        .axis(Vector3::new(1.0, 2.0, 0.0))
        .kind(SymmetryElementKind::Proper)
        .build()
        .unwrap();
    assert!(!c2.is_identity());
    let c2p = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(2))
        .proper_power(2)
        .axis(Vector3::new(1.0, 2.0, 0.0))
        .kind(SymmetryElementKind::Proper)
        .build()
        .unwrap();
    assert!(c2p.is_identity());
    let c2p2 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(2))
        .proper_power(3)
        .axis(-Vector3::new(1.0, 2.0, 0.0))
        .kind(SymmetryElementKind::Proper)
        .build()
        .unwrap();
    assert!(!c2p2.is_identity());
    assert_eq!(c2, c2p2);
    assert_eq!(c1, c2p);
    assert_ne!(c1, c2p2);

    let c4 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(4))
        .proper_power(1)
        .axis(Vector3::new(1.0, 1.0, 1.0))
        .kind(SymmetryElementKind::Proper)
        .build()
        .unwrap();
    let c4p2 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(4))
        .proper_power(2)
        .axis(-Vector3::new(1.0, 2.0, 0.0))
        .kind(SymmetryElementKind::Proper)
        .build()
        .unwrap();
    let c4p3 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(4))
        .proper_power(3)
        .axis(-Vector3::new(1.0, 1.0, 1.0))
        .kind(SymmetryElementKind::Proper)
        .build()
        .unwrap();
    let c4p4 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(4))
        .proper_power(4)
        .axis(-Vector3::new(2.0, 1.0, 1.0))
        .kind(SymmetryElementKind::Proper)
        .build()
        .unwrap();
    let c4p5 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(4))
        .proper_power(5)
        .axis(-Vector3::new(1.0, 1.0, 1.0))
        .kind(SymmetryElementKind::Proper)
        .build()
        .unwrap();
    assert!(c4p4.is_identity());
    assert_eq!(c4p2, c2);
    assert_eq!(c4p4, c1);
    assert_eq!(c4p4, c1p2);
    assert_eq!(c4, c4p3);
    assert_eq!(c4, c4p5);

    let c5 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(5))
        .proper_power(1)
        .axis(Vector3::new(2.0, 2.0, 1.0))
        .kind(SymmetryElementKind::Proper)
        .build()
        .unwrap();
    let c5p9 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(5))
        .proper_power(9)
        .axis(-Vector3::new(2.0, 2.0, 1.0))
        .kind(SymmetryElementKind::Proper)
        .build()
        .unwrap();
    assert_eq!(c5, c5p9);

    // =============
    // Improper only
    // =============
    let s1 = SymmetryElement::builder()
        .threshold(1e-3)
        .proper_order(ElementOrder::Int(1))
        .proper_power(1)
        .axis(Vector3::new(0.0, 2.0, 0.0))
        .kind(SymmetryElementKind::ImproperMirrorPlane)
        .build()
        .unwrap();
    assert!(s1.is_mirror_plane());
    let s1p2 = SymmetryElement::builder()
        .threshold(1e-3)
        .proper_order(ElementOrder::Int(1))
        .proper_power(2)
        .axis(Vector3::new(0.0, 2.0, 0.0))
        .kind(SymmetryElementKind::ImproperMirrorPlane)
        .build()
        .unwrap();
    assert!(s1p2.is_mirror_plane());
    assert_eq!(s1, s1p2);
    let sd2 = s1.convert_to_improper_kind(&SymmetryElementKind::ImproperInversionCentre, false);
    let sd2p = s1p2.convert_to_improper_kind(&SymmetryElementKind::ImproperInversionCentre, false);
    assert_eq!(sd2, sd2p);

    let s2 = SymmetryElement::builder()
        .threshold(1e-3)
        .proper_order(ElementOrder::Int(2))
        .proper_power(1)
        .axis(Vector3::new(0.0, 2.0, 0.0))
        .kind(SymmetryElementKind::ImproperMirrorPlane)
        .build()
        .unwrap();
    assert!(s2.is_inversion_centre());
    let s2p2 = SymmetryElement::builder()
        .threshold(1e-3)
        .proper_order(ElementOrder::Int(2))
        .proper_power(2)
        .axis(Vector3::new(0.0, 2.0, 0.0))
        .kind(SymmetryElementKind::ImproperMirrorPlane)
        .build()
        .unwrap();
    assert!(s2p2.is_mirror_plane());
    let s2p3 = SymmetryElement::builder()
        .threshold(1e-3)
        .proper_order(ElementOrder::Int(2))
        .proper_power(3)
        .axis(Vector3::new(0.0, 2.0, 0.0))
        .kind(SymmetryElementKind::ImproperMirrorPlane)
        .build()
        .unwrap();
    assert!(s2p3.is_inversion_centre());
    assert_eq!(s1, s2p2);
    assert_eq!(s2, s2p3);

    let sd2 = SymmetryElement::builder()
        .threshold(1e-3)
        .proper_order(ElementOrder::Int(2))
        .proper_power(1)
        .axis(Vector3::new(0.0, 2.0, 0.0))
        .kind(SymmetryElementKind::ImproperInversionCentre)
        .build()
        .unwrap();
    assert!(sd2.is_mirror_plane());
    let sd2b = sd2.convert_to_improper_kind(&SymmetryElementKind::ImproperMirrorPlane, true);
    let sd2p2 = SymmetryElement::builder()
        .threshold(1e-3)
        .proper_order(ElementOrder::Int(2))
        .proper_power(2)
        .axis(Vector3::new(1.0, 2.0, 0.0))
        .kind(SymmetryElementKind::ImproperInversionCentre)
        .build()
        .unwrap();
    assert!(sd2p2.is_inversion_centre());
    let sd2p2b = sd2p2.convert_to_improper_kind(&SymmetryElementKind::ImproperMirrorPlane, true);
    assert_eq!(sd2, s1);
    assert_eq!(sd2p2, s2);
    assert_eq!(sd2b, sd2);
    assert_eq!(sd2p2b, sd2p2);

    let s3 = SymmetryElement::builder()
        .threshold(1e-3)
        .proper_order(ElementOrder::Int(3))
        .proper_power(1)
        .axis(Vector3::new(1.0, 2.0, 1.0))
        .kind(SymmetryElementKind::ImproperMirrorPlane)
        .build()
        .unwrap();
    let s3p2 = SymmetryElement::builder()
        .threshold(1e-3)
        .proper_order(ElementOrder::Int(3))
        .proper_power(2)
        .axis(Vector3::new(1.0, 2.0, 1.0))
        .kind(SymmetryElementKind::ImproperMirrorPlane)
        .build()
        .unwrap();
    let s3p4 = SymmetryElement::builder()
        .threshold(1e-3)
        .proper_order(ElementOrder::Int(3))
        .proper_power(4)
        .axis(Vector3::new(1.0, 2.0, 1.0))
        .kind(SymmetryElementKind::ImproperMirrorPlane)
        .build()
        .unwrap();
    let s3p3 = SymmetryElement::builder()
        .threshold(1e-3)
        .proper_order(ElementOrder::Int(3))
        .proper_power(3)
        .axis(Vector3::new(1.0, 2.0, 1.0))
        .kind(SymmetryElementKind::ImproperMirrorPlane)
        .build()
        .unwrap();
    assert!(s3p3.is_mirror_plane());
    assert_eq!(s3, s3p2);
    assert_eq!(s3, s3p4);

    let s6p2 = SymmetryElement::builder()
        .threshold(1e-3)
        .proper_order(ElementOrder::Int(6))
        .proper_power(2)
        .axis(Vector3::new(1.0, 2.0, 1.0))
        .kind(SymmetryElementKind::ImproperMirrorPlane)
        .build()
        .unwrap();
    let s6p3 = SymmetryElement::builder()
        .threshold(1e-3)
        .proper_order(ElementOrder::Int(6))
        .proper_power(3)
        .axis(Vector3::new(2.0, 2.0, 1.0))
        .kind(SymmetryElementKind::ImproperMirrorPlane)
        .build()
        .unwrap();
    let s6p4 = SymmetryElement::builder()
        .threshold(1e-3)
        .proper_order(ElementOrder::Int(6))
        .proper_power(4)
        .axis(-Vector3::new(1.0, 2.0, 1.0))
        .kind(SymmetryElementKind::ImproperMirrorPlane)
        .build()
        .unwrap();
    assert!(s6p3.is_inversion_centre());
    assert_eq!(s3, s6p2);
    assert_eq!(s3, s6p4);

    // ===================
    // Proper and improper
    // ===================
    assert_ne!(c2, s2);
}

#[test]
fn test_symmetry_element_finite_hashset() {
    let mut element_set = HashSet::new();
    let c1 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(1))
        .proper_power(1)
        .axis(Vector3::new(0.0, 2.0, 0.0))
        .kind(SymmetryElementKind::Proper)
        .build()
        .unwrap();
    element_set.insert(c1);
    assert_eq!(element_set.len(), 1);

    let c1p = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(1))
        .proper_power(1)
        .axis(Vector3::new(1.0, 2.0, 0.0))
        .kind(SymmetryElementKind::Proper)
        .build()
        .unwrap();
    element_set.insert(c1p);
    assert_eq!(element_set.len(), 1);

    let c2 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(2))
        .proper_power(1)
        .axis(Vector3::new(1.0, 2.0, 0.0))
        .kind(SymmetryElementKind::Proper)
        .build()
        .unwrap();
    element_set.insert(c2);
    assert_eq!(element_set.len(), 2);

    let c2p = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(2))
        .proper_power(1)
        .axis(-Vector3::new(1.0, 2.0, 0.0))
        .kind(SymmetryElementKind::Proper)
        .build()
        .unwrap();
    element_set.insert(c2p);
    assert_eq!(element_set.len(), 2);

    let c3 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(3))
        .proper_power(1)
        .axis(Vector3::new(1.0, 2.0, 0.0))
        .kind(SymmetryElementKind::Proper)
        .build()
        .unwrap();
    element_set.insert(c3);
    assert_eq!(element_set.len(), 3);

    let c3p = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(3))
        .proper_power(1)
        .axis(-Vector3::new(1.0, 2.0, 0.0))
        .kind(SymmetryElementKind::Proper)
        .build()
        .unwrap();
    element_set.insert(c3p);
    assert_eq!(element_set.len(), 3);

    let s1 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(1))
        .proper_power(1)
        .axis(Vector3::new(0.0, 2.0, 0.0))
        .kind(SymmetryElementKind::ImproperMirrorPlane)
        .build()
        .unwrap();
    let sd2 = s1.convert_to_improper_kind(&SymmetryElementKind::ImproperInversionCentre, false);
    element_set.insert(s1);
    assert_eq!(element_set.len(), 4);
    element_set.insert(sd2);
    assert_eq!(element_set.len(), 4);

    let sd1 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(1))
        .proper_power(1)
        .axis(Vector3::new(1.0, 1.0, 1.0))
        .kind(SymmetryElementKind::ImproperInversionCentre)
        .build()
        .unwrap();
    let s2 = sd1.convert_to_improper_kind(&SymmetryElementKind::ImproperMirrorPlane, false);
    element_set.insert(sd1);
    assert_eq!(element_set.len(), 5);
    element_set.insert(s2);
    assert_eq!(element_set.len(), 5);

    let s3 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(3))
        .proper_power(1)
        .axis(Vector3::new(2.0, 2.0, 1.0))
        .kind(SymmetryElementKind::ImproperMirrorPlane)
        .build()
        .unwrap();
    let sd6 = s3.convert_to_improper_kind(&SymmetryElementKind::ImproperInversionCentre, false);
    element_set.insert(s3);
    assert_eq!(element_set.len(), 6);
    element_set.insert(sd6);
    assert_eq!(element_set.len(), 6);

    let s3p = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(3))
        .proper_power(1)
        .axis(-Vector3::new(2.0, 2.0, 1.0))
        .kind(SymmetryElementKind::ImproperMirrorPlane)
        .build()
        .unwrap();
    element_set.insert(s3p);
    assert_eq!(element_set.len(), 6);
}

#[test]
fn test_symmetry_element_cartesian_axis_closeness() {
    let s3 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(3))
        .proper_power(1)
        .axis(Vector3::new(1.0, 1.0, 1.0))
        .kind(SymmetryElementKind::ImproperMirrorPlane)
        .build()
        .unwrap();
    let (s3_closeness, s3_closest_axis) = s3.closeness_to_cartesian_axes();
    approx::assert_relative_eq!(s3_closeness, 1.0 - 1.0/3f64.sqrt());
    assert_eq!(s3_closest_axis, 0);

    let s4 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(3))
        .proper_power(1)
        .axis(Vector3::new(2.0, 1.0, 1.0))
        .kind(SymmetryElementKind::ImproperMirrorPlane)
        .build()
        .unwrap();
    let (s4_closeness, s4_closest_axis) = s4.closeness_to_cartesian_axes();
    approx::assert_relative_eq!(s4_closeness, 1.0 - 2.0/6f64.sqrt());
    assert_eq!(s4_closest_axis, 2);

    let s6 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(3))
        .proper_power(1)
        .axis(Vector3::new(0.0, 1.0, 0.0))
        .kind(SymmetryElementKind::Proper)
        .build()
        .unwrap();
    let (s6_closeness, s6_closest_axis) = s6.closeness_to_cartesian_axes();
    approx::assert_relative_eq!(s6_closeness, 0.0);
    assert_eq!(s6_closest_axis, 1);
}

#[test]
fn test_infinite_symmetry_element_comparison() {
    // ========================
    // Proper symmetry elements
    // ========================
    let ci1 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Inf)
        .axis(Vector3::new(0.0, 2.0, 0.0))
        .kind(SymmetryElementKind::Proper)
        .build()
        .unwrap();

    let ci2 = SymmetryElement::builder()
        .threshold(1e-7)
        .proper_order(ElementOrder::Inf)
        .axis(Vector3::new(0.0, 2.0, 0.0))
        .kind(SymmetryElementKind::Proper)
        .build()
        .unwrap();
    assert_eq!(ci1, ci2);

    let ci3 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Inf)
        .proper_angle(std::f64::consts::FRAC_PI_3)
        .axis(Vector3::new(0.0, 2.0, 1.0))
        .kind(SymmetryElementKind::Proper)
        .build()
        .unwrap();

    let ci4 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Inf)
        .proper_angle(2.0 * std::f64::consts::PI - std::f64::consts::FRAC_PI_3)
        .axis(-Vector3::new(0.0, 2.0, 1.0))
        .kind(SymmetryElementKind::Proper)
        .build()
        .unwrap();
    let ci4b = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Inf)
        .proper_angle(2.0 * std::f64::consts::PI - std::f64::consts::FRAC_PI_3)
        .axis(Vector3::new(0.0, 2.0, 1.0))
        .kind(SymmetryElementKind::Proper)
        .build()
        .unwrap();
    assert_eq!(ci3, ci4);
    assert_eq!(ci3, ci4b);

    // ==========================
    // Improper symmetry elements
    // ==========================
    let si1 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Inf)
        .axis(Vector3::new(0.0, 2.0, 0.0))
        .kind(SymmetryElementKind::ImproperMirrorPlane)
        .build()
        .unwrap();

    let si2 = SymmetryElement::builder()
        .threshold(1e-7)
        .proper_order(ElementOrder::Inf)
        .axis(Vector3::new(0.0, 2.0, 0.0))
        .kind(SymmetryElementKind::ImproperMirrorPlane)
        .build()
        .unwrap();
    let si2c = si2.convert_to_improper_kind(&SymmetryElementKind::ImproperInversionCentre, true);
    let si2b = SymmetryElement::builder()
        .threshold(1e-7)
        .proper_order(ElementOrder::Inf)
        .axis(Vector3::new(0.0, 2.0, 0.0))
        .kind(SymmetryElementKind::ImproperInversionCentre)
        .build()
        .unwrap();
    assert_eq!(si1, si2);
    assert_eq!(si1, si2b); // No proper angle specified, both conventions are the same.
    assert_eq!(si1, si2c);

    let si3 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Inf)
        .proper_angle(std::f64::consts::FRAC_PI_4)
        .axis(Vector3::new(1.0, 2.0, 1.0))
        .kind(SymmetryElementKind::ImproperMirrorPlane)
        .build()
        .unwrap();
    let si3b = si3.convert_to_improper_kind(&SymmetryElementKind::ImproperInversionCentre, false);
    let si3c = si3.convert_to_improper_kind(&SymmetryElementKind::ImproperInversionCentre, true);
    assert_eq!(si3, si3b);
    assert_eq!(si3, si3c);

    let si4 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Inf)
        .proper_angle(2.0 * std::f64::consts::PI - std::f64::consts::FRAC_PI_4)
        .axis(Vector3::new(1.0, 2.0, 1.0))
        .kind(SymmetryElementKind::ImproperMirrorPlane)
        .build()
        .unwrap();
    assert_eq!(si3, si4);

    let si5 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Inf)
        .proper_angle(2.0 * std::f64::consts::PI - std::f64::consts::FRAC_PI_4)
        .axis(Vector3::new(1.0, 2.0, 1.0))
        .kind(SymmetryElementKind::ImproperInversionCentre)
        .build()
        .unwrap();
    let si5b = si5.convert_to_improper_kind(&SymmetryElementKind::ImproperMirrorPlane, true);
    assert_ne!(si3, si5);
    assert_eq!(si5, si5b);
}