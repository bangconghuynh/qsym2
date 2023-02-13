use nalgebra::Vector3;
use std::collections::HashSet;

use crate::aux::misc;
use crate::symmetry::symmetry_element::{
    ElementOrder, SymmetryElement, INV, ROT, SIG, TRINV, TRROT, TRSIG,
};

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
        .kind(ROT)
        .build()
        .unwrap();
    assert_eq!(format!("{}", &c1), "E");
    assert_eq!(format!("{:?}", &c1), "C1(+0.000, +1.000, +0.000)");

    let tc1 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(1))
        .proper_power(1)
        .axis(Vector3::new(0.0, 2.0, 0.0))
        .kind(TRROT)
        .build()
        .unwrap();
    assert_eq!(format!("{}", &tc1), "θ");
    assert_eq!(format!("{:?}", &tc1), "θ·C1(+0.000, +1.000, +0.000)");

    let c2 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(2))
        .proper_power(1)
        .axis(Vector3::new(1.0, 1.0, 0.0))
        .kind(ROT)
        .build()
        .unwrap();
    assert_eq!(format!("{}", &c2), "C2(+0.707, +0.707, +0.000)");
    assert_eq!(format!("{:?}", &c2), "C2(+0.707, +0.707, +0.000)");

    let tc2 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(2))
        .proper_power(1)
        .axis(Vector3::new(1.0, 1.0, 0.0))
        .kind(TRROT)
        .build()
        .unwrap();
    assert_eq!(format!("{}", &tc2), "θ·C2(+0.707, +0.707, +0.000)");
    assert_eq!(format!("{:?}", &tc2), "θ·C2(+0.707, +0.707, +0.000)");

    let c2p2 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(2))
        .proper_power(2)
        .axis(Vector3::new(1.0, 1.0, 0.0))
        .kind(ROT)
        .build()
        .unwrap();
    assert_eq!(format!("{}", &c2p2), "E");
    assert_eq!(format!("{:?}", &c2p2), "C2^2(+0.707, +0.707, +0.000)");

    let tc2p2 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(2))
        .proper_power(2)
        .axis(Vector3::new(1.0, 1.0, 0.0))
        .kind(TRROT)
        .build()
        .unwrap();
    assert_eq!(format!("{}", &tc2p2), "θ");
    assert_eq!(format!("{:?}", &tc2p2), "θ·C2^2(+0.707, +0.707, +0.000)");

    let c3 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::new(3.0, 1e-14))
        .proper_power(1)
        .axis(Vector3::new(1.0, 1.0, 1.0))
        .kind(ROT)
        .build()
        .unwrap();
    assert_eq!(format!("{}", &c3), "C3(+0.577, +0.577, +0.577)");
    assert_eq!(format!("{:?}", &c3), "C3(+0.577, +0.577, +0.577)");

    let tc3 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::new(3.0, 1e-14))
        .proper_power(1)
        .axis(Vector3::new(1.0, 1.0, 1.0))
        .kind(TRROT)
        .build()
        .unwrap();
    assert_eq!(format!("{}", &tc3), "θ·C3(+0.577, +0.577, +0.577)");
    assert_eq!(format!("{:?}", &tc3), "θ·C3(+0.577, +0.577, +0.577)");

    let c3p2 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::new(3.0, 1e-14))
        .proper_power(2)
        .axis(Vector3::new(1.0, 1.0, 1.0))
        .kind(ROT)
        .build()
        .unwrap();
    assert_eq!(format!("{}", &c3p2), "C3^2(+0.577, +0.577, +0.577)");
    assert_eq!(format!("{:?}", &c3p2), "C3^2(+0.577, +0.577, +0.577)");

    let tc3p2 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::new(3.0, 1e-14))
        .proper_power(2)
        .axis(Vector3::new(1.0, 1.0, 1.0))
        .kind(TRROT)
        .build()
        .unwrap();
    assert_eq!(format!("{}", &tc3p2), "θ·C3^2(+0.577, +0.577, +0.577)");
    assert_eq!(format!("{:?}", &tc3p2), "θ·C3^2(+0.577, +0.577, +0.577)");

    let c3p3 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::new(3.0, 1e-14))
        .proper_power(3)
        .axis(Vector3::new(1.0, 1.0, 1.0))
        .kind(ROT)
        .build()
        .unwrap();
    assert_eq!(format!("{}", &c3p3), "E");
    assert_eq!(format!("{:?}", &c3p3), "C3^3(+0.577, +0.577, +0.577)");

    let tc3p3 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::new(3.0, 1e-14))
        .proper_power(3)
        .axis(Vector3::new(1.0, 1.0, 1.0))
        .kind(TRROT)
        .build()
        .unwrap();
    assert_eq!(format!("{}", &tc3p3), "θ");
    assert_eq!(format!("{:?}", &tc3p3), "θ·C3^3(+0.577, +0.577, +0.577)");

    let ci = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Inf)
        .axis(Vector3::new(1.0, 0.0, -1.0))
        .kind(ROT)
        .build()
        .unwrap();
    assert_eq!(format!("{}", &ci), "C∞(+0.707, +0.000, -0.707)");
    assert_eq!(format!("{:?}", &ci), "C∞(+0.707, +0.000, -0.707)");

    let tci = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Inf)
        .axis(Vector3::new(1.0, 0.0, -1.0))
        .kind(TRROT)
        .build()
        .unwrap();
    assert_eq!(format!("{}", &tci), "θ·C∞(+0.707, +0.000, -0.707)");
    assert_eq!(format!("{:?}", &tci), "θ·C∞(+0.707, +0.000, -0.707)");

    let ci2 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Inf)
        .axis(Vector3::new(1.0, 0.0, 1.0))
        .proper_angle(0.12)
        .kind(ROT)
        .build()
        .unwrap();
    assert_eq!(format!("{}", &ci2), "C∞(+0.120)(+0.707, +0.000, +0.707)");
    assert_eq!(format!("{:?}", &ci2), "C∞(+0.120)(+0.707, +0.000, +0.707)");

    let tci2 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Inf)
        .axis(Vector3::new(1.0, 0.0, 1.0))
        .proper_angle(0.12)
        .kind(TRROT)
        .build()
        .unwrap();
    assert_eq!(format!("{}", &tci2), "θ·C∞(+0.120)(+0.707, +0.000, +0.707)");
    assert_eq!(
        format!("{:?}", &tci2),
        "θ·C∞(+0.120)(+0.707, +0.000, +0.707)"
    );

    let ci3 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Inf)
        .axis(Vector3::new(1.0, 0.0, 1.0))
        .proper_angle(3.160)
        .kind(ROT)
        .build()
        .unwrap();
    assert_eq!(format!("{}", &ci3), "C∞(-3.123)(+0.707, +0.000, +0.707)");
    assert_eq!(format!("{:?}", &ci3), "C∞(-3.123)(+0.707, +0.000, +0.707)");

    let tci3 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Inf)
        .axis(Vector3::new(1.0, 0.0, 1.0))
        .proper_angle(3.160)
        .kind(TRROT)
        .build()
        .unwrap();
    assert_eq!(format!("{}", &tci3), "θ·C∞(-3.123)(+0.707, +0.000, +0.707)");
    assert_eq!(
        format!("{:?}", &tci3),
        "θ·C∞(-3.123)(+0.707, +0.000, +0.707)"
    );

    // ==========================
    // Improper symmetry elements
    // ==========================
    let s1 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(1))
        .proper_power(1)
        .axis(Vector3::new(0.0, 2.0, 0.0))
        .kind(SIG)
        .build()
        .unwrap();
    assert_eq!(format!("{}", &s1), "σ(+0.000, +1.000, +0.000)");
    assert_eq!(format!("{:?}", &s1), "S1(+0.000, +1.000, +0.000)");

    let ts1 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(1))
        .proper_power(1)
        .axis(Vector3::new(0.0, 2.0, 0.0))
        .kind(TRSIG)
        .build()
        .unwrap();
    assert_eq!(format!("{}", &ts1), "θ·σ(+0.000, +1.000, +0.000)");
    assert_eq!(format!("{:?}", &ts1), "θ·S1(+0.000, +1.000, +0.000)");

    let sd2 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(2))
        .proper_power(1)
        .axis(Vector3::new(-1.0, 1.0, 0.0))
        .kind(INV)
        .build()
        .unwrap();
    assert_eq!(format!("{}", &sd2), "σ(-0.707, +0.707, +0.000)");
    assert_eq!(format!("{:?}", &sd2), "Ṡ2(-0.707, +0.707, +0.000)");

    let tsd2 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(2))
        .proper_power(1)
        .axis(Vector3::new(-1.0, 1.0, 0.0))
        .kind(TRINV)
        .build()
        .unwrap();
    assert_eq!(format!("{}", &tsd2), "θ·σ(-0.707, +0.707, +0.000)");
    assert_eq!(format!("{:?}", &tsd2), "θ·Ṡ2(-0.707, +0.707, +0.000)");

    let sd2p2 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(2))
        .proper_power(2)
        .axis(Vector3::new(-1.0, 1.0, 0.0))
        .kind(INV)
        .build()
        .unwrap();
    assert_eq!(format!("{}", &sd2p2), "i");
    assert_eq!(format!("{:?}", &sd2p2), "iC2^2(-0.707, +0.707, +0.000)");

    let tsd2p2 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(2))
        .proper_power(2)
        .axis(Vector3::new(-1.0, 1.0, 0.0))
        .kind(TRINV)
        .build()
        .unwrap();
    assert_eq!(format!("{}", &tsd2p2), "θ·i");
    assert_eq!(format!("{:?}", &tsd2p2), "θ·iC2^2(-0.707, +0.707, +0.000)");

    let s2 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(2))
        .proper_power(1)
        .axis(Vector3::new(2.0, 2.0, 1.0))
        .kind(SIG)
        .build()
        .unwrap();
    assert_eq!(format!("{}", &s2), "i");
    assert_eq!(format!("{:?}", &s2), "S2(+0.667, +0.667, +0.333)");

    let ts2 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(2))
        .proper_power(1)
        .axis(Vector3::new(2.0, 2.0, 1.0))
        .kind(TRSIG)
        .build()
        .unwrap();
    assert_eq!(format!("{}", &ts2), "θ·i");
    assert_eq!(format!("{:?}", &ts2), "θ·S2(+0.667, +0.667, +0.333)");

    let s2p2 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(2))
        .proper_power(2)
        .axis(Vector3::new(2.0, 2.0, 1.0))
        .kind(SIG)
        .build()
        .unwrap();
    assert_eq!(format!("{}", &s2p2), "σ(+0.667, +0.667, +0.333)");
    assert_eq!(format!("{:?}", &s2p2), "σC2^2(+0.667, +0.667, +0.333)");

    let ts2p2 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(2))
        .proper_power(2)
        .axis(Vector3::new(2.0, 2.0, 1.0))
        .kind(TRSIG)
        .build()
        .unwrap();
    assert_eq!(format!("{}", &ts2p2), "θ·σ(+0.667, +0.667, +0.333)");
    assert_eq!(format!("{:?}", &ts2p2), "θ·σC2^2(+0.667, +0.667, +0.333)");

    let sd1 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(1))
        .proper_power(1)
        .axis(Vector3::new(1.0, 1.0, 1.0))
        .kind(INV)
        .build()
        .unwrap();
    assert_eq!(format!("{}", &sd1), "i");
    assert_eq!(format!("{:?}", &sd1), "Ṡ1(+0.577, +0.577, +0.577)");
    assert_eq!(s2, sd1);

    let tsd1 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(1))
        .proper_power(1)
        .axis(Vector3::new(1.0, 1.0, 1.0))
        .kind(TRINV)
        .build()
        .unwrap();
    assert_eq!(format!("{}", &tsd1), "θ·i");
    assert_eq!(format!("{:?}", &tsd1), "θ·Ṡ1(+0.577, +0.577, +0.577)");
    assert_eq!(ts2, tsd1);

    let s3 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(3))
        .proper_power(1)
        .axis(Vector3::new(2.0, 2.0, 1.0))
        .kind(SIG)
        .build()
        .unwrap();
    assert_eq!(format!("{}", &s3), "S3(+0.667, +0.667, +0.333)");
    assert_eq!(format!("{:?}", &s3), "S3(+0.667, +0.667, +0.333)");

    let ts3 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(3))
        .proper_power(1)
        .axis(Vector3::new(2.0, 2.0, 1.0))
        .kind(TRSIG)
        .build()
        .unwrap();
    assert_eq!(format!("{}", &ts3), "θ·S3(+0.667, +0.667, +0.333)");
    assert_eq!(format!("{:?}", &ts3), "θ·S3(+0.667, +0.667, +0.333)");

    let s3p2 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(3))
        .proper_power(2)
        .axis(Vector3::new(2.0, 2.0, 1.0))
        .kind(SIG)
        .build()
        .unwrap();
    assert_eq!(format!("{}", &s3p2), "σC3^2(+0.667, +0.667, +0.333)");
    assert_eq!(format!("{:?}", &s3p2), "σC3^2(+0.667, +0.667, +0.333)");

    let ts3p2 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(3))
        .proper_power(2)
        .axis(Vector3::new(2.0, 2.0, 1.0))
        .kind(TRSIG)
        .build()
        .unwrap();
    assert_eq!(format!("{}", &ts3p2), "θ·σC3^2(+0.667, +0.667, +0.333)");
    assert_eq!(format!("{:?}", &ts3p2), "θ·σC3^2(+0.667, +0.667, +0.333)");

    let s3p3 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(3))
        .proper_power(3)
        .axis(Vector3::new(2.0, 2.0, 1.0))
        .kind(SIG)
        .build()
        .unwrap();
    assert_eq!(format!("{}", &s3p3), "σ(+0.667, +0.667, +0.333)");
    assert_eq!(format!("{:?}", &s3p3), "σC3^3(+0.667, +0.667, +0.333)");

    let ts3p3 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(3))
        .proper_power(3)
        .axis(Vector3::new(2.0, 2.0, 1.0))
        .kind(TRSIG)
        .build()
        .unwrap();
    assert_eq!(format!("{}", &ts3p3), "θ·σ(+0.667, +0.667, +0.333)");
    assert_eq!(format!("{:?}", &ts3p3), "θ·σC3^3(+0.667, +0.667, +0.333)");

    let sd3 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(3))
        .proper_power(1)
        .axis(Vector3::new(1.0, 1.0, 1.0))
        .kind(INV)
        .build()
        .unwrap();
    assert_eq!(format!("{}", &sd3), "Ṡ3(+0.577, +0.577, +0.577)");
    assert_eq!(format!("{:?}", &sd3), "Ṡ3(+0.577, +0.577, +0.577)");

    let tsd3 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(3))
        .proper_power(1)
        .axis(Vector3::new(1.0, 1.0, 1.0))
        .kind(TRINV)
        .build()
        .unwrap();
    assert_eq!(format!("{}", &tsd3), "θ·Ṡ3(+0.577, +0.577, +0.577)");
    assert_eq!(format!("{:?}", &tsd3), "θ·Ṡ3(+0.577, +0.577, +0.577)");

    let sd3p2 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(3))
        .proper_power(2)
        .axis(Vector3::new(1.0, 1.0, 1.0))
        .kind(INV)
        .build()
        .unwrap();
    assert_eq!(format!("{}", &sd3p2), "iC3^2(+0.577, +0.577, +0.577)");
    assert_eq!(format!("{:?}", &sd3p2), "iC3^2(+0.577, +0.577, +0.577)");

    let tsd3p2 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(3))
        .proper_power(2)
        .axis(Vector3::new(1.0, 1.0, 1.0))
        .kind(TRINV)
        .build()
        .unwrap();
    assert_eq!(format!("{}", &tsd3p2), "θ·iC3^2(+0.577, +0.577, +0.577)");
    assert_eq!(format!("{:?}", &tsd3p2), "θ·iC3^2(+0.577, +0.577, +0.577)");

    let sd3p3 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(3))
        .proper_power(3)
        .axis(Vector3::new(1.0, 1.0, 1.0))
        .kind(INV)
        .build()
        .unwrap();
    assert_eq!(format!("{}", &sd3p3), "i");
    assert_eq!(format!("{:?}", &sd3p3), "iC3^3(+0.577, +0.577, +0.577)");

    let tsd3p3 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(3))
        .proper_power(3)
        .axis(Vector3::new(1.0, 1.0, 1.0))
        .kind(TRINV)
        .build()
        .unwrap();
    assert_eq!(format!("{}", &tsd3p3), "θ·i");
    assert_eq!(format!("{:?}", &tsd3p3), "θ·iC3^3(+0.577, +0.577, +0.577)");

    let si = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Inf)
        .proper_power(1)
        .axis(Vector3::new(1.0, 0.0, -1.0))
        .kind(SIG)
        .build()
        .unwrap();
    assert_eq!(format!("{}", &si), "S∞(+0.707, +0.000, -0.707)");
    assert_eq!(format!("{:?}", &si), "σC∞(+0.707, +0.000, -0.707)");

    let tsi = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Inf)
        .proper_power(1)
        .axis(Vector3::new(1.0, 0.0, -1.0))
        .kind(TRSIG)
        .build()
        .unwrap();
    assert_eq!(format!("{}", &tsi), "θ·S∞(+0.707, +0.000, -0.707)");
    assert_eq!(format!("{:?}", &tsi), "θ·σC∞(+0.707, +0.000, -0.707)");

    let sib = si.convert_to_improper_kind(&INV, false);
    assert_eq!(format!("{}", &sib), "Ṡ∞(+0.707, +0.000, -0.707)");
    assert_eq!(format!("{:?}", &sib), "iC∞(+0.707, +0.000, -0.707)");

    let tsib = tsi.convert_to_improper_kind(&INV, false);
    assert_eq!(format!("{}", &tsib), "θ·Ṡ∞(+0.707, +0.000, -0.707)");
    assert_eq!(format!("{:?}", &tsib), "θ·iC∞(+0.707, +0.000, -0.707)");

    let tsic = tsi.convert_to_improper_kind(&TRINV, false);
    assert_eq!(format!("{}", &tsic), "θ·Ṡ∞(+0.707, +0.000, -0.707)");
    assert_eq!(format!("{:?}", &tsic), "θ·iC∞(+0.707, +0.000, -0.707)");

    let si2 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Inf)
        .axis(Vector3::new(1.0, 0.0, 1.0))
        .proper_angle(0.121)
        .kind(SIG)
        .build()
        .unwrap();
    assert_eq!(format!("{}", &si2), "S∞(+0.121)(+0.707, +0.000, +0.707)");
    assert_eq!(format!("{:?}", &si2), "σC∞(+0.121)(+0.707, +0.000, +0.707)");

    let tsi2 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Inf)
        .axis(Vector3::new(1.0, 0.0, 1.0))
        .proper_angle(0.121)
        .kind(TRSIG)
        .build()
        .unwrap();
    assert_eq!(format!("{}", &tsi2), "θ·S∞(+0.121)(+0.707, +0.000, +0.707)");
    assert_eq!(
        format!("{:?}", &tsi2),
        "θ·σC∞(+0.121)(+0.707, +0.000, +0.707)"
    );

    let si2b = si2.convert_to_improper_kind(&INV, false);
    assert_eq!(format!("{}", &si2b), "Ṡ∞(-3.021)(+0.707, +0.000, +0.707)");
    assert_eq!(
        format!("{:?}", &si2b),
        "iC∞(-3.021)(+0.707, +0.000, +0.707)"
    );

    let tsi2b = tsi2.convert_to_improper_kind(&INV, false);
    assert_eq!(
        format!("{}", &tsi2b),
        "θ·Ṡ∞(-3.021)(+0.707, +0.000, +0.707)"
    );
    assert_eq!(
        format!("{:?}", &tsi2b),
        "θ·iC∞(-3.021)(+0.707, +0.000, +0.707)"
    );

    let tsi2c = tsi2.convert_to_improper_kind(&TRINV, false);
    assert_eq!(
        format!("{}", &tsi2c),
        "θ·Ṡ∞(-3.021)(+0.707, +0.000, +0.707)"
    );
    assert_eq!(
        format!("{:?}", &tsi2c),
        "θ·iC∞(-3.021)(+0.707, +0.000, +0.707)"
    );
}

#[test]
fn test_symmetry_element_finite_improper_conversion() {
    let s1 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(1))
        .proper_power(1)
        .axis(Vector3::new(0.0, 2.0, 0.0))
        .kind(SIG)
        .build()
        .unwrap();
    let sd2 = s1.convert_to_improper_kind(&INV, false);
    assert_eq!(format!("{}", &sd2), "σ(+0.000, +1.000, +0.000)");

    let ts1 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(1))
        .proper_power(1)
        .axis(Vector3::new(0.0, 2.0, 0.0))
        .kind(TRSIG)
        .build()
        .unwrap();
    let tsd2 = ts1.convert_to_improper_kind(&INV, false);
    assert_eq!(format!("{}", &tsd2), "θ·σ(+0.000, +1.000, +0.000)");

    let sd1 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(1))
        .proper_power(1)
        .axis(Vector3::new(1.0, 1.0, 1.0))
        .kind(INV)
        .build()
        .unwrap();
    let s2 = sd1.convert_to_improper_kind(&SIG, false);
    assert_eq!(format!("{}", &s2), "i");

    let tsd1 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(1))
        .proper_power(1)
        .axis(Vector3::new(1.0, 1.0, 1.0))
        .kind(TRINV)
        .build()
        .unwrap();
    let ts2 = tsd1.convert_to_improper_kind(&SIG, false);
    assert_eq!(format!("{}", &ts2), "θ·i");

    let s3 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(3))
        .proper_power(1)
        .axis(Vector3::new(2.0, 2.0, 1.0))
        .kind(SIG)
        .build()
        .unwrap();
    let sd6 = s3.convert_to_improper_kind(&INV, false);
    assert_eq!(format!("{}", &sd6), "Ṡ6(+0.667, +0.667, +0.333)");

    let ts3 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(3))
        .proper_power(1)
        .axis(Vector3::new(2.0, 2.0, 1.0))
        .kind(TRSIG)
        .build()
        .unwrap();
    let tsd6 = ts3.convert_to_improper_kind(&TRINV, false);
    assert_eq!(format!("{}", &tsd6), "θ·Ṡ6(+0.667, +0.667, +0.333)");

    let s4 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(4))
        .proper_power(1)
        .axis(Vector3::new(1.0, 1.0, 1.0))
        .kind(SIG)
        .build()
        .unwrap();
    let sd4 = s4.convert_to_improper_kind(&INV, false);
    assert_eq!(sd4.proper_order, ElementOrder::Int(4));
    assert_eq!(format!("{}", &sd4), "Ṡ4(+0.577, +0.577, +0.577)");

    let ts4 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(4))
        .proper_power(1)
        .axis(Vector3::new(1.0, 1.0, 1.0))
        .kind(TRSIG)
        .build()
        .unwrap();
    let tsd4 = ts4.convert_to_improper_kind(&INV, false);
    assert_eq!(tsd4.proper_order, ElementOrder::Int(4));
    assert_eq!(format!("{}", &tsd4), "θ·Ṡ4(+0.577, +0.577, +0.577)");

    let sd5 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(5))
        .proper_power(1)
        .axis(Vector3::new(1.0, 2.0, 1.0))
        .kind(INV)
        .build()
        .unwrap();
    let s10 = sd5.convert_to_improper_kind(&SIG, false);
    assert_eq!(s10.proper_order, ElementOrder::Int(10));
    assert_eq!(format!("{}", &s10), "S10(+0.408, +0.816, +0.408)");

    let tsd5 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(5))
        .proper_power(1)
        .axis(Vector3::new(1.0, 2.0, 1.0))
        .kind(TRINV)
        .build()
        .unwrap();
    let ts10 = tsd5.convert_to_improper_kind(&SIG, false);
    assert_eq!(ts10.proper_order, ElementOrder::Int(10));
    assert_eq!(format!("{}", &ts10), "θ·S10(+0.408, +0.816, +0.408)");

    let sd7 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(7))
        .proper_power(1)
        .axis(Vector3::new(1.0, 1.0, 1.0))
        .kind(INV)
        .build()
        .unwrap();
    let s14 = sd7.convert_to_improper_kind(&SIG, false);
    assert_eq!(s14.proper_order, ElementOrder::Int(14));

    let tsd7 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(7))
        .proper_power(1)
        .axis(Vector3::new(1.0, 1.0, 1.0))
        .kind(TRINV)
        .build()
        .unwrap();
    let ts14 = tsd7.convert_to_improper_kind(&SIG, false);
    assert_eq!(ts14.proper_order, ElementOrder::Int(14));
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
        .kind(ROT)
        .build()
        .unwrap();
    let c1p = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(1))
        .proper_power(1)
        .axis(Vector3::new(1.0, 2.0, 0.0))
        .kind(ROT)
        .build()
        .unwrap();
    assert_eq!(c1, c1p);

    let tc1 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(1))
        .proper_power(1)
        .axis(Vector3::new(0.0, 2.0, 0.0))
        .kind(TRROT)
        .build()
        .unwrap();
    let tc1p = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(1))
        .proper_power(1)
        .axis(Vector3::new(1.0, 2.0, 0.0))
        .kind(TRROT)
        .build()
        .unwrap();
    assert_eq!(tc1, tc1p);

    let c1p2 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(1))
        .proper_power(2)
        .axis(Vector3::new(4.0, 2.0, 0.0))
        .kind(ROT)
        .build()
        .unwrap();
    assert_eq!(c1, c1p2);

    let tc1p2 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(1))
        .proper_power(2)
        .axis(Vector3::new(4.0, 2.0, 0.0))
        .kind(TRROT)
        .build()
        .unwrap();
    assert_eq!(tc1, tc1p2);

    let c2 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(2))
        .proper_power(1)
        .axis(Vector3::new(1.0, 2.0, 0.0))
        .kind(ROT)
        .build()
        .unwrap();
    let c2p = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(2))
        .proper_power(1)
        .axis(-Vector3::new(1.0, 2.0, 0.0))
        .kind(ROT)
        .build()
        .unwrap();
    assert_eq!(c2, c2p);
    assert_ne!(c1, c2);

    let tc2 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(2))
        .proper_power(1)
        .axis(Vector3::new(1.0, 2.0, 0.0))
        .kind(TRROT)
        .build()
        .unwrap();
    let tc2p = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(2))
        .proper_power(1)
        .axis(-Vector3::new(1.0, 2.0, 0.0))
        .kind(TRROT)
        .build()
        .unwrap();
    assert_eq!(tc2, tc2p);
    assert_ne!(tc1, tc2);

    let c3 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(3))
        .proper_power(1)
        .axis(Vector3::new(1.0, 2.0, 0.0))
        .kind(ROT)
        .build()
        .unwrap();
    let c3p = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(3))
        .proper_power(1)
        .axis(-Vector3::new(1.0, 2.0, 0.0))
        .kind(ROT)
        .build()
        .unwrap();
    assert_eq!(c3, c3p);

    let tc3 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(3))
        .proper_power(1)
        .axis(Vector3::new(1.0, 2.0, 0.0))
        .kind(TRROT)
        .build()
        .unwrap();
    let tc3p = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(3))
        .proper_power(1)
        .axis(-Vector3::new(1.0, 2.0, 0.0))
        .kind(TRROT)
        .build()
        .unwrap();
    assert_eq!(tc3, tc3p);

    // =============
    // Improper only
    // =============
    let s1 = SymmetryElement::builder()
        .threshold(1e-3)
        .proper_order(ElementOrder::Int(1))
        .proper_power(1)
        .axis(Vector3::new(0.0, 2.0, 0.0))
        .kind(SIG)
        .build()
        .unwrap();
    let sd2 = s1.convert_to_improper_kind(&INV, false);
    assert_eq!(s1, sd2);

    let ts1 = SymmetryElement::builder()
        .threshold(1e-3)
        .proper_order(ElementOrder::Int(1))
        .proper_power(1)
        .axis(Vector3::new(0.0, 2.0, 0.0))
        .kind(TRSIG)
        .build()
        .unwrap();
    let tsd2 = ts1.convert_to_improper_kind(&INV, false);
    assert_eq!(ts1, tsd2);

    let sd1 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(1))
        .proper_power(1)
        .axis(Vector3::new(1.0, 1.0, 1.0))
        .kind(INV)
        .build()
        .unwrap();
    let s2 = sd1.convert_to_improper_kind(&SIG, false);
    assert_eq!(sd1, s2);
    assert_ne!(sd1, sd2);
    assert_ne!(s1, s2);

    let tsd1 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(1))
        .proper_power(1)
        .axis(Vector3::new(1.0, 1.0, 1.0))
        .kind(TRINV)
        .build()
        .unwrap();
    let ts2 = tsd1.convert_to_improper_kind(&SIG, false);
    assert_eq!(tsd1, ts2);
    assert_ne!(tsd1, tsd2);
    assert_ne!(ts1, ts2);

    let s3 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(3))
        .proper_power(1)
        .axis(Vector3::new(2.0, 2.0, 1.0))
        .kind(SIG)
        .build()
        .unwrap();
    let sd6 = s3.convert_to_improper_kind(&INV, false);
    assert_eq!(sd6.proper_order, ElementOrder::Int(6));
    assert_eq!(s3, sd6);

    let ts3 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(3))
        .proper_power(1)
        .axis(Vector3::new(2.0, 2.0, 1.0))
        .kind(TRSIG)
        .build()
        .unwrap();
    let tsd6 = ts3.convert_to_improper_kind(&INV, false);
    assert_eq!(tsd6.proper_order, ElementOrder::Int(6));
    assert_eq!(ts3, tsd6);

    let s3p = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(3))
        .proper_power(1)
        .axis(-Vector3::new(2.0, 2.0, 1.0))
        .kind(SIG)
        .build()
        .unwrap();
    assert_eq!(s3, s3p);
    assert_ne!(s2, s3);

    let ts3p = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(3))
        .proper_power(1)
        .axis(-Vector3::new(2.0, 2.0, 1.0))
        .kind(TRSIG)
        .build()
        .unwrap();
    assert_eq!(ts3, ts3p);
    assert_ne!(ts2, ts3);

    // ===================
    // Proper and improper
    // ===================
    assert_ne!(c2, s2);
    assert_ne!(c2, sd2);
    assert_ne!(tc2, ts2);
    assert_ne!(tc2, tsd2);
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
        .kind(ROT)
        .build()
        .unwrap();
    let c1p = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(1))
        .proper_power(2)
        .axis(Vector3::new(1.0, 2.0, 0.0))
        .kind(ROT)
        .build()
        .unwrap();
    let c1p2 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(1))
        .proper_power(0)
        .axis(-Vector3::new(1.0, 2.0, 0.0))
        .kind(ROT)
        .build()
        .unwrap();
    assert_eq!(c1, c1p);
    assert_eq!(c1, c1p2);
    assert_eq!(c1p, c1p2);
    assert!(c1.is_identity(false));

    let tc1 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(1))
        .proper_power(1)
        .axis(Vector3::new(0.0, 2.0, 0.0))
        .kind(TRROT)
        .build()
        .unwrap();
    let tc1p = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(1))
        .proper_power(2)
        .axis(Vector3::new(1.0, 2.0, 0.0))
        .kind(TRROT)
        .build()
        .unwrap();
    let tc1p2 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(1))
        .proper_power(0)
        .axis(-Vector3::new(1.0, 2.0, 0.0))
        .kind(TRROT)
        .build()
        .unwrap();
    assert_eq!(tc1, tc1p);
    assert_eq!(tc1, tc1p2);
    assert_eq!(tc1p, tc1p2);
    assert!(tc1.is_identity(true));

    let c2 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(2))
        .proper_power(1)
        .axis(Vector3::new(1.0, 2.0, 0.0))
        .kind(ROT)
        .build()
        .unwrap();
    assert!(!c2.is_identity(false));
    let c2p = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(2))
        .proper_power(2)
        .axis(Vector3::new(1.0, 2.0, 0.0))
        .kind(ROT)
        .build()
        .unwrap();
    assert!(c2p.is_identity(false));
    let c2p2 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(2))
        .proper_power(3)
        .axis(-Vector3::new(1.0, 2.0, 0.0))
        .kind(ROT)
        .build()
        .unwrap();
    assert!(!c2p2.is_identity(false));
    assert_eq!(c2, c2p2);
    assert_eq!(c1, c2p);
    assert_ne!(c1, c2p2);

    let tc2 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(2))
        .proper_power(1)
        .axis(Vector3::new(1.0, 2.0, 0.0))
        .kind(TRROT)
        .build()
        .unwrap();
    assert!(!tc2.is_identity(true));
    let tc2p = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(2))
        .proper_power(2)
        .axis(Vector3::new(1.0, 2.0, 0.0))
        .kind(TRROT)
        .build()
        .unwrap();
    assert!(tc2p.is_identity(true));
    let tc2p2 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(2))
        .proper_power(3)
        .axis(-Vector3::new(1.0, 2.0, 0.0))
        .kind(TRROT)
        .build()
        .unwrap();
    assert!(!tc2p2.is_identity(true));
    assert_eq!(tc2, tc2p2);
    assert_eq!(tc1, tc2p);
    assert_ne!(tc1, tc2p2);

    let c4 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(4))
        .proper_power(1)
        .axis(Vector3::new(1.0, 1.0, 1.0))
        .kind(ROT)
        .build()
        .unwrap();
    let c4p2 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(4))
        .proper_power(2)
        .axis(-Vector3::new(1.0, 2.0, 0.0))
        .kind(ROT)
        .build()
        .unwrap();
    let c4p3 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(4))
        .proper_power(3)
        .axis(-Vector3::new(1.0, 1.0, 1.0))
        .kind(ROT)
        .build()
        .unwrap();
    let c4p4 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(4))
        .proper_power(4)
        .axis(-Vector3::new(2.0, 1.0, 1.0))
        .kind(ROT)
        .build()
        .unwrap();
    let c4p5 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(4))
        .proper_power(5)
        .axis(-Vector3::new(1.0, 1.0, 1.0))
        .kind(ROT)
        .build()
        .unwrap();
    assert!(c4p4.is_identity(false));
    assert_eq!(c4p2, c2);
    assert_eq!(c4p4, c1);
    assert_eq!(c4p4, c1p2);
    assert_eq!(c4, c4p3);
    assert_eq!(c4, c4p5);

    let tc4 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(4))
        .proper_power(1)
        .axis(Vector3::new(1.0, 1.0, 1.0))
        .kind(TRROT)
        .build()
        .unwrap();
    let tc4p2 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(4))
        .proper_power(2)
        .axis(-Vector3::new(1.0, 2.0, 0.0))
        .kind(TRROT)
        .build()
        .unwrap();
    let tc4p3 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(4))
        .proper_power(3)
        .axis(-Vector3::new(1.0, 1.0, 1.0))
        .kind(TRROT)
        .build()
        .unwrap();
    let tc4p4 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(4))
        .proper_power(4)
        .axis(-Vector3::new(2.0, 1.0, 1.0))
        .kind(TRROT)
        .build()
        .unwrap();
    let tc4p5 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(4))
        .proper_power(5)
        .axis(-Vector3::new(1.0, 1.0, 1.0))
        .kind(TRROT)
        .build()
        .unwrap();
    assert!(tc4p4.is_identity(true));
    assert_eq!(tc4p2, tc2);
    assert_eq!(tc4p4, tc1);
    assert_eq!(tc4p4, tc1p2);
    assert_eq!(tc4, tc4p3);
    assert_eq!(tc4, tc4p5);

    let c5 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(5))
        .proper_power(1)
        .axis(Vector3::new(2.0, 2.0, 1.0))
        .kind(ROT)
        .build()
        .unwrap();
    let c5p9 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(5))
        .proper_power(9)
        .axis(-Vector3::new(2.0, 2.0, 1.0))
        .kind(ROT)
        .build()
        .unwrap();
    assert_eq!(c5, c5p9);

    let tc5 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(5))
        .proper_power(1)
        .axis(Vector3::new(2.0, 2.0, 1.0))
        .kind(TRROT)
        .build()
        .unwrap();
    let tc5p9 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(5))
        .proper_power(9)
        .axis(-Vector3::new(2.0, 2.0, 1.0))
        .kind(TRROT)
        .build()
        .unwrap();
    assert_eq!(tc5, tc5p9);

    // =============
    // Improper only
    // =============
    let s1 = SymmetryElement::builder()
        .threshold(1e-3)
        .proper_order(ElementOrder::Int(1))
        .proper_power(1)
        .axis(Vector3::new(0.0, 2.0, 0.0))
        .kind(SIG)
        .build()
        .unwrap();
    assert!(s1.is_mirror_plane(false));
    let s1p2 = SymmetryElement::builder()
        .threshold(1e-3)
        .proper_order(ElementOrder::Int(1))
        .proper_power(2)
        .axis(Vector3::new(0.0, 2.0, 0.0))
        .kind(SIG)
        .build()
        .unwrap();
    assert!(s1p2.is_mirror_plane(false));
    assert_eq!(s1, s1p2);
    let sd2 = s1.convert_to_improper_kind(&INV, false);
    let sd2p = s1p2.convert_to_improper_kind(&INV, false);
    assert_eq!(sd2, sd2p);

    let ts1 = SymmetryElement::builder()
        .threshold(1e-3)
        .proper_order(ElementOrder::Int(1))
        .proper_power(1)
        .axis(Vector3::new(0.0, 2.0, 0.0))
        .kind(TRSIG)
        .build()
        .unwrap();
    assert!(ts1.is_mirror_plane(true));
    let ts1p2 = SymmetryElement::builder()
        .threshold(1e-3)
        .proper_order(ElementOrder::Int(1))
        .proper_power(2)
        .axis(Vector3::new(0.0, 2.0, 0.0))
        .kind(TRSIG)
        .build()
        .unwrap();
    assert!(ts1p2.is_mirror_plane(true));
    assert_eq!(ts1, ts1p2);
    let tsd2 = ts1.convert_to_improper_kind(&INV, false);
    let tsd2p = ts1p2.convert_to_improper_kind(&INV, false);
    assert_eq!(tsd2, tsd2p);

    let s2 = SymmetryElement::builder()
        .threshold(1e-3)
        .proper_order(ElementOrder::Int(2))
        .proper_power(1)
        .axis(Vector3::new(0.0, 2.0, 0.0))
        .kind(SIG)
        .build()
        .unwrap();
    assert!(s2.is_inversion_centre(false));
    let s2p2 = SymmetryElement::builder()
        .threshold(1e-3)
        .proper_order(ElementOrder::Int(2))
        .proper_power(2)
        .axis(Vector3::new(0.0, 2.0, 0.0))
        .kind(SIG)
        .build()
        .unwrap();
    assert!(s2p2.is_mirror_plane(false));
    let s2p3 = SymmetryElement::builder()
        .threshold(1e-3)
        .proper_order(ElementOrder::Int(2))
        .proper_power(3)
        .axis(Vector3::new(0.0, 2.0, 0.0))
        .kind(SIG)
        .build()
        .unwrap();
    assert!(s2p3.is_inversion_centre(false));
    assert_eq!(s1, s2p2);
    assert_eq!(s2, s2p3);

    let ts2 = SymmetryElement::builder()
        .threshold(1e-3)
        .proper_order(ElementOrder::Int(2))
        .proper_power(1)
        .axis(Vector3::new(0.0, 2.0, 0.0))
        .kind(TRSIG)
        .build()
        .unwrap();
    assert!(ts2.is_inversion_centre(true));
    let ts2p2 = SymmetryElement::builder()
        .threshold(1e-3)
        .proper_order(ElementOrder::Int(2))
        .proper_power(2)
        .axis(Vector3::new(0.0, 2.0, 0.0))
        .kind(TRSIG)
        .build()
        .unwrap();
    assert!(ts2p2.is_mirror_plane(true));
    let ts2p3 = SymmetryElement::builder()
        .threshold(1e-3)
        .proper_order(ElementOrder::Int(2))
        .proper_power(3)
        .axis(Vector3::new(0.0, 2.0, 0.0))
        .kind(TRSIG)
        .build()
        .unwrap();
    assert!(ts2p3.is_inversion_centre(true));
    assert_eq!(ts1, ts2p2);
    assert_eq!(ts2, ts2p3);

    let sd2 = SymmetryElement::builder()
        .threshold(1e-3)
        .proper_order(ElementOrder::Int(2))
        .proper_power(1)
        .axis(Vector3::new(0.0, 2.0, 0.0))
        .kind(INV)
        .build()
        .unwrap();
    assert!(sd2.is_mirror_plane(false));
    let sd2b = sd2.convert_to_improper_kind(&SIG, true);
    let sd2p2 = SymmetryElement::builder()
        .threshold(1e-3)
        .proper_order(ElementOrder::Int(2))
        .proper_power(2)
        .axis(Vector3::new(1.0, 2.0, 0.0))
        .kind(INV)
        .build()
        .unwrap();
    assert!(sd2p2.is_inversion_centre(false));
    let sd2p2b = sd2p2.convert_to_improper_kind(&SIG, true);
    assert_eq!(sd2, s1);
    assert_eq!(sd2p2, s2);
    assert_eq!(sd2b, sd2);
    assert_eq!(sd2p2b, sd2p2);

    let tsd2 = SymmetryElement::builder()
        .threshold(1e-3)
        .proper_order(ElementOrder::Int(2))
        .proper_power(1)
        .axis(Vector3::new(0.0, 2.0, 0.0))
        .kind(TRINV)
        .build()
        .unwrap();
    assert!(tsd2.is_mirror_plane(true));
    let tsd2b = tsd2.convert_to_improper_kind(&SIG, true);
    let tsd2p2 = SymmetryElement::builder()
        .threshold(1e-3)
        .proper_order(ElementOrder::Int(2))
        .proper_power(2)
        .axis(Vector3::new(1.0, 2.0, 0.0))
        .kind(TRINV)
        .build()
        .unwrap();
    assert!(tsd2p2.is_inversion_centre(true));
    let tsd2p2b = tsd2p2.convert_to_improper_kind(&SIG, true);
    assert_eq!(tsd2, ts1);
    assert_eq!(tsd2p2, ts2);
    assert_eq!(tsd2b, tsd2);
    assert_eq!(tsd2p2b, tsd2p2);

    let s3 = SymmetryElement::builder()
        .threshold(1e-3)
        .proper_order(ElementOrder::Int(3))
        .proper_power(1)
        .axis(Vector3::new(1.0, 2.0, 1.0))
        .kind(SIG)
        .build()
        .unwrap();
    let s3p2 = SymmetryElement::builder()
        .threshold(1e-3)
        .proper_order(ElementOrder::Int(3))
        .proper_power(2)
        .axis(Vector3::new(1.0, 2.0, 1.0))
        .kind(SIG)
        .build()
        .unwrap();
    let s3p4 = SymmetryElement::builder()
        .threshold(1e-3)
        .proper_order(ElementOrder::Int(3))
        .proper_power(4)
        .axis(Vector3::new(1.0, 2.0, 1.0))
        .kind(SIG)
        .build()
        .unwrap();
    let s3p3 = SymmetryElement::builder()
        .threshold(1e-3)
        .proper_order(ElementOrder::Int(3))
        .proper_power(3)
        .axis(Vector3::new(1.0, 2.0, 1.0))
        .kind(SIG)
        .build()
        .unwrap();
    assert!(s3p3.is_mirror_plane(false));
    assert_eq!(s3, s3p2);
    assert_eq!(s3, s3p4);

    let ts3 = SymmetryElement::builder()
        .threshold(1e-3)
        .proper_order(ElementOrder::Int(3))
        .proper_power(1)
        .axis(Vector3::new(1.0, 2.0, 1.0))
        .kind(TRSIG)
        .build()
        .unwrap();
    let ts3p2 = SymmetryElement::builder()
        .threshold(1e-3)
        .proper_order(ElementOrder::Int(3))
        .proper_power(2)
        .axis(Vector3::new(1.0, 2.0, 1.0))
        .kind(TRSIG)
        .build()
        .unwrap();
    let ts3p4 = SymmetryElement::builder()
        .threshold(1e-3)
        .proper_order(ElementOrder::Int(3))
        .proper_power(4)
        .axis(Vector3::new(1.0, 2.0, 1.0))
        .kind(TRSIG)
        .build()
        .unwrap();
    let ts3p3 = SymmetryElement::builder()
        .threshold(1e-3)
        .proper_order(ElementOrder::Int(3))
        .proper_power(3)
        .axis(Vector3::new(1.0, 2.0, 1.0))
        .kind(TRSIG)
        .build()
        .unwrap();
    assert!(ts3p3.is_mirror_plane(true));
    assert_eq!(ts3, ts3p2);
    assert_eq!(ts3, ts3p4);

    let s6p2 = SymmetryElement::builder()
        .threshold(1e-3)
        .proper_order(ElementOrder::Int(6))
        .proper_power(2)
        .axis(Vector3::new(1.0, 2.0, 1.0))
        .kind(SIG)
        .build()
        .unwrap();
    let s6p3 = SymmetryElement::builder()
        .threshold(1e-3)
        .proper_order(ElementOrder::Int(6))
        .proper_power(3)
        .axis(Vector3::new(2.0, 2.0, 1.0))
        .kind(SIG)
        .build()
        .unwrap();
    let s6p4 = SymmetryElement::builder()
        .threshold(1e-3)
        .proper_order(ElementOrder::Int(6))
        .proper_power(4)
        .axis(-Vector3::new(1.0, 2.0, 1.0))
        .kind(SIG)
        .build()
        .unwrap();
    assert!(s6p3.is_inversion_centre(false));
    assert_eq!(s3, s6p2);
    assert_eq!(s3, s6p4);

    let ts6p2 = SymmetryElement::builder()
        .threshold(1e-3)
        .proper_order(ElementOrder::Int(6))
        .proper_power(2)
        .axis(Vector3::new(1.0, 2.0, 1.0))
        .kind(TRSIG)
        .build()
        .unwrap();
    let ts6p3 = SymmetryElement::builder()
        .threshold(1e-3)
        .proper_order(ElementOrder::Int(6))
        .proper_power(3)
        .axis(Vector3::new(2.0, 2.0, 1.0))
        .kind(TRSIG)
        .build()
        .unwrap();
    let ts6p4 = SymmetryElement::builder()
        .threshold(1e-3)
        .proper_order(ElementOrder::Int(6))
        .proper_power(4)
        .axis(-Vector3::new(1.0, 2.0, 1.0))
        .kind(TRSIG)
        .build()
        .unwrap();
    assert!(ts6p3.is_inversion_centre(true));
    assert_eq!(ts3, ts6p2);
    assert_eq!(ts3, ts6p4);

    // ===================
    // Proper and improper
    // ===================
    assert_ne!(c2, s2);
    assert_ne!(tc2, ts2);
}

#[test]
fn test_symmetry_element_finite_hashset() {
    let mut element_set = HashSet::new();
    let c1 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(1))
        .proper_power(1)
        .axis(Vector3::new(0.0, 2.0, 0.0))
        .kind(ROT)
        .build()
        .unwrap();
    element_set.insert(c1);
    assert_eq!(element_set.len(), 1);

    let c1p = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(1))
        .proper_power(1)
        .axis(Vector3::new(1.0, 2.0, 0.0))
        .kind(ROT)
        .build()
        .unwrap();
    element_set.insert(c1p);
    assert_eq!(element_set.len(), 1);

    let c2 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(2))
        .proper_power(1)
        .axis(Vector3::new(1.0, 2.0, 0.0))
        .kind(ROT)
        .build()
        .unwrap();
    element_set.insert(c2);
    assert_eq!(element_set.len(), 2);

    let c2p = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(2))
        .proper_power(1)
        .axis(-Vector3::new(1.0, 2.0, 0.0))
        .kind(ROT)
        .build()
        .unwrap();
    element_set.insert(c2p);
    assert_eq!(element_set.len(), 2);

    let c3 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(3))
        .proper_power(1)
        .axis(Vector3::new(1.0, 2.0, 0.0))
        .kind(ROT)
        .build()
        .unwrap();
    element_set.insert(c3);
    assert_eq!(element_set.len(), 3);

    let c3p = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(3))
        .proper_power(1)
        .axis(-Vector3::new(1.0, 2.0, 0.0))
        .kind(ROT)
        .build()
        .unwrap();
    element_set.insert(c3p);
    assert_eq!(element_set.len(), 3);

    let s1 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(1))
        .proper_power(1)
        .axis(Vector3::new(0.0, 2.0, 0.0))
        .kind(SIG)
        .build()
        .unwrap();
    let sd2 = s1.convert_to_improper_kind(&INV, false);
    element_set.insert(s1);
    assert_eq!(element_set.len(), 4);
    element_set.insert(sd2);
    assert_eq!(element_set.len(), 4);

    let sd1 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(1))
        .proper_power(1)
        .axis(Vector3::new(1.0, 1.0, 1.0))
        .kind(INV)
        .build()
        .unwrap();
    let s2 = sd1.convert_to_improper_kind(&SIG, false);
    element_set.insert(sd1);
    assert_eq!(element_set.len(), 5);
    element_set.insert(s2);
    assert_eq!(element_set.len(), 5);

    let s3 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(3))
        .proper_power(1)
        .axis(Vector3::new(2.0, 2.0, 1.0))
        .kind(SIG)
        .build()
        .unwrap();
    let sd6 = s3.convert_to_improper_kind(&INV, false);
    element_set.insert(s3);
    assert_eq!(element_set.len(), 6);
    element_set.insert(sd6);
    assert_eq!(element_set.len(), 6);

    let s3p = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(3))
        .proper_power(1)
        .axis(-Vector3::new(2.0, 2.0, 1.0))
        .kind(SIG)
        .build()
        .unwrap();
    element_set.insert(s3p);
    assert_eq!(element_set.len(), 6);

    let tc1 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(1))
        .proper_power(1)
        .axis(Vector3::new(0.0, 2.0, 0.0))
        .kind(TRROT)
        .build()
        .unwrap();
    element_set.insert(tc1);
    assert_eq!(element_set.len(), 7);

    let tc1p = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(1))
        .proper_power(1)
        .axis(Vector3::new(1.0, 2.0, 0.0))
        .kind(TRROT)
        .build()
        .unwrap();
    element_set.insert(tc1p);
    assert_eq!(element_set.len(), 7);

    let tc2 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(2))
        .proper_power(1)
        .axis(Vector3::new(1.0, 2.0, 0.0))
        .kind(TRROT)
        .build()
        .unwrap();
    element_set.insert(tc2);
    assert_eq!(element_set.len(), 8);

    let tc2p = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(2))
        .proper_power(1)
        .axis(-Vector3::new(1.0, 2.0, 0.0))
        .kind(TRROT)
        .build()
        .unwrap();
    element_set.insert(tc2p);
    assert_eq!(element_set.len(), 8);

    let tc3 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(3))
        .proper_power(1)
        .axis(Vector3::new(1.0, 2.0, 0.0))
        .kind(TRROT)
        .build()
        .unwrap();
    element_set.insert(tc3);
    assert_eq!(element_set.len(), 9);

    let tc3p = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(3))
        .proper_power(1)
        .axis(-Vector3::new(1.0, 2.0, 0.0))
        .kind(TRROT)
        .build()
        .unwrap();
    element_set.insert(tc3p);
    assert_eq!(element_set.len(), 9);

    let ts1 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(1))
        .proper_power(1)
        .axis(Vector3::new(0.0, 2.0, 0.0))
        .kind(TRSIG)
        .build()
        .unwrap();
    let tsd2 = ts1.convert_to_improper_kind(&INV, false);
    element_set.insert(ts1);
    assert_eq!(element_set.len(), 10);
    element_set.insert(tsd2);
    assert_eq!(element_set.len(), 10);

    let tsd1 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(1))
        .proper_power(1)
        .axis(Vector3::new(1.0, 1.0, 1.0))
        .kind(TRINV)
        .build()
        .unwrap();
    let ts2 = tsd1.convert_to_improper_kind(&SIG, false);
    element_set.insert(tsd1);
    assert_eq!(element_set.len(), 11);
    element_set.insert(ts2);
    assert_eq!(element_set.len(), 11);

    let ts3 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(3))
        .proper_power(1)
        .axis(Vector3::new(2.0, 2.0, 1.0))
        .kind(TRSIG)
        .build()
        .unwrap();
    let tsd6 = ts3.convert_to_improper_kind(&INV, false);
    element_set.insert(ts3);
    assert_eq!(element_set.len(), 12);
    element_set.insert(tsd6);
    assert_eq!(element_set.len(), 12);

    let ts3p = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(3))
        .proper_power(1)
        .axis(-Vector3::new(2.0, 2.0, 1.0))
        .kind(TRSIG)
        .build()
        .unwrap();
    element_set.insert(ts3p);
    assert_eq!(element_set.len(), 12);
}

#[test]
fn test_symmetry_element_cartesian_axis_closeness() {
    let s3 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(3))
        .proper_power(1)
        .axis(Vector3::new(1.0, 1.0, 1.0))
        .kind(SIG)
        .build()
        .unwrap();
    let (s3_closeness, s3_closest_axis) = s3.closeness_to_cartesian_axes();
    approx::assert_relative_eq!(s3_closeness, 1.0 - 1.0 / 3f64.sqrt());
    assert_eq!(s3_closest_axis, 0);

    let s4 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(3))
        .proper_power(1)
        .axis(Vector3::new(2.0, 1.0, 1.0))
        .kind(SIG)
        .build()
        .unwrap();
    let (s4_closeness, s4_closest_axis) = s4.closeness_to_cartesian_axes();
    approx::assert_relative_eq!(s4_closeness, 1.0 - 2.0 / 6f64.sqrt());
    assert_eq!(s4_closest_axis, 2);

    let s6 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(3))
        .proper_power(1)
        .axis(Vector3::new(0.0, 1.0, 0.0))
        .kind(ROT)
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
        .kind(ROT)
        .build()
        .unwrap();

    let ci2 = SymmetryElement::builder()
        .threshold(1e-7)
        .proper_order(ElementOrder::Inf)
        .axis(Vector3::new(0.0, 2.0, 0.0))
        .kind(ROT)
        .build()
        .unwrap();
    assert_eq!(ci1, ci2);

    let ci3 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Inf)
        .proper_angle(std::f64::consts::FRAC_PI_3)
        .axis(Vector3::new(0.0, 2.0, 1.0))
        .kind(ROT)
        .build()
        .unwrap();

    let ci4 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Inf)
        .proper_angle(2.0 * std::f64::consts::PI - std::f64::consts::FRAC_PI_3)
        .axis(-Vector3::new(0.0, 2.0, 1.0))
        .kind(ROT)
        .build()
        .unwrap();
    let ci4b = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Inf)
        .proper_angle(2.0 * std::f64::consts::PI - std::f64::consts::FRAC_PI_3)
        .axis(Vector3::new(0.0, 2.0, 1.0))
        .kind(ROT)
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
        .kind(SIG)
        .build()
        .unwrap();

    let si2 = SymmetryElement::builder()
        .threshold(1e-7)
        .proper_order(ElementOrder::Inf)
        .axis(Vector3::new(0.0, 2.0, 0.0))
        .kind(SIG)
        .build()
        .unwrap();
    let si2c = si2.convert_to_improper_kind(&INV, true);
    let si2b = SymmetryElement::builder()
        .threshold(1e-7)
        .proper_order(ElementOrder::Inf)
        .axis(Vector3::new(0.0, 2.0, 0.0))
        .kind(INV)
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
        .kind(SIG)
        .build()
        .unwrap();
    let si3b = si3.convert_to_improper_kind(&INV, false);
    let si3c = si3.convert_to_improper_kind(&INV, true);
    assert_eq!(si3, si3b);
    assert_eq!(si3, si3c);

    let si4 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Inf)
        .proper_angle(2.0 * std::f64::consts::PI - std::f64::consts::FRAC_PI_4)
        .axis(Vector3::new(1.0, 2.0, 1.0))
        .kind(SIG)
        .build()
        .unwrap();
    assert_eq!(si3, si4);

    let si5 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Inf)
        .proper_angle(2.0 * std::f64::consts::PI - std::f64::consts::FRAC_PI_4)
        .axis(Vector3::new(1.0, 2.0, 1.0))
        .kind(INV)
        .build()
        .unwrap();
    let si5b = si5.convert_to_improper_kind(&SIG, true);
    assert_ne!(si3, si5);
    assert_eq!(si5, si5b);
}
