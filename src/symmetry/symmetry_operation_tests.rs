use nalgebra::Vector3;

use crate::symmetry::symmetry_element::{
    ElementOrder, SymmetryElement, SymmetryElementKind, SymmetryOperation,
};

#[test]
fn test_symmetry_operation_constructor() {
    // ==========================
    // Proper symmetry operations
    // ==========================
    let c1_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(1))
        .proper_power(1)
        .axis(Vector3::new(0.0, 2.0, 0.0))
        .kind(SymmetryElementKind::Proper)
        .build()
        .unwrap();

    let c1 = SymmetryOperation::builder()
        .generating_element(c1_element.clone())
        .power(1)
        .build()
        .unwrap();
    assert!(c1.is_identity());

    let c1b = SymmetryOperation::builder()
        .generating_element(c1_element)
        .power(-3)
        .build()
        .unwrap();
    assert!(c1b.is_identity());

    let c2_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(2))
        .proper_power(1)
        .axis(Vector3::new(1.0, 1.0, 0.0))
        .kind(SymmetryElementKind::Proper)
        .build()
        .unwrap();

    let c2 = SymmetryOperation::builder()
        .generating_element(c2_element.clone())
        .power(1)
        .build()
        .unwrap();
    assert!(c2.is_binary_rotation());

    let c2p2 = SymmetryOperation::builder()
        .generating_element(c2_element.clone())
        .power(2)
        .build()
        .unwrap();
    assert!(c2p2.is_identity());

    let c2p2_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(2))
        .proper_power(2)
        .axis(Vector3::new(1.0, 1.0, 0.0))
        .kind(SymmetryElementKind::Proper)
        .build()
        .unwrap();

    let c2p2b = SymmetryOperation::builder()
        .generating_element(c2p2_element)
        .power(1)
        .build()
        .unwrap();
    assert!(c2p2b.is_identity());

    let c3_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::new(3.0, 1e-14))
        .proper_power(1)
        .axis(Vector3::new(1.0, 1.0, 1.0))
        .kind(SymmetryElementKind::Proper)
        .build()
        .unwrap();

    let c3 = SymmetryOperation::builder()
        .generating_element(c3_element.clone())
        .power(1)
        .build()
        .unwrap();
    assert!(!c3.is_identity());

    let c3p3 = SymmetryOperation::builder()
        .generating_element(c3_element)
        .power(-3)
        .build()
        .unwrap();
    assert!(c3p3.is_identity());

    let c3p2_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::new(3.0, 1e-14))
        .proper_power(2)
        .axis(Vector3::new(1.0, 1.0, 1.0))
        .kind(SymmetryElementKind::Proper)
        .build()
        .unwrap();

    let c3p6 = SymmetryOperation::builder()
        .generating_element(c3p2_element)
        .power(3)
        .build()
        .unwrap();
    assert!(c3p6.is_identity());

    let c4_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::new(4.0, 1e-14))
        .proper_power(1)
        .axis(Vector3::new(1.0, 1.0, 1.0))
        .kind(SymmetryElementKind::Proper)
        .build()
        .unwrap();

    let c4p2 = SymmetryOperation::builder()
        .generating_element(c4_element.clone())
        .power(2)
        .build()
        .unwrap();
    assert!(c4p2.is_binary_rotation());

    let c4p4 = SymmetryOperation::builder()
        .generating_element(c4_element)
        .power(-4)
        .build()
        .unwrap();
    assert!(c4p4.is_identity());

    let ci_element = SymmetryElement::builder()
        .threshold(1e-7)
        .proper_order(ElementOrder::Inf)
        .axis(Vector3::new(1.0, 0.0, -1.0))
        .proper_angle(2.0 * std::f64::consts::FRAC_PI_6)
        .kind(SymmetryElementKind::Proper)
        .build()
        .unwrap();

    let cip3 = SymmetryOperation::builder()
        .generating_element(ci_element.clone())
        .power(-3)
        .build()
        .unwrap();
    assert!(cip3.is_binary_rotation());

    let cip6 = SymmetryOperation::builder()
        .generating_element(ci_element.clone())
        .power(-6)
        .build()
        .unwrap();
    assert!(cip6.is_identity());

    let cip0 = SymmetryOperation::builder()
        .generating_element(ci_element)
        .power(0)
        .build()
        .unwrap();
    assert!(cip0.is_identity());

    // ==========================
    // Improper symmetry elements
    // ==========================
    let s1_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(1))
        .proper_power(1)
        .axis(Vector3::new(0.0, 2.0, 0.0))
        .kind(SymmetryElementKind::ImproperMirrorPlane)
        .build()
        .unwrap();

    let s1 = SymmetryOperation::builder()
        .generating_element(s1_element.clone())
        .power(1)
        .build()
        .unwrap();
    assert!(s1.is_reflection());

    let s1p2 = SymmetryOperation::builder()
        .generating_element(s1_element)
        .power(-2)
        .build()
        .unwrap();
    assert!(s1p2.is_identity());

    let sd2_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(2))
        .proper_power(1)
        .axis(Vector3::new(-1.0, 1.0, 0.0))
        .kind(SymmetryElementKind::ImproperInversionCentre)
        .build()
        .unwrap();

    let sd2 = SymmetryOperation::builder()
        .generating_element(sd2_element.clone())
        .power(1)
        .build()
        .unwrap();
    assert!(sd2.is_reflection());

    let sd2p2 = SymmetryOperation::builder()
        .generating_element(sd2_element)
        .power(2)
        .build()
        .unwrap();
    assert!(sd2p2.is_identity());

    let sd2pp2_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(2))
        .proper_power(2)
        .axis(Vector3::new(-1.0, 1.0, 0.0))
        .kind(SymmetryElementKind::ImproperInversionCentre)
        .build()
        .unwrap();

    let sd2pp2 = SymmetryOperation::builder()
        .generating_element(sd2pp2_element.clone())
        .power(1)
        .build()
        .unwrap();
    assert!(sd2pp2.is_inversion());

    let sd2pp2p6 = SymmetryOperation::builder()
        .generating_element(sd2pp2_element)
        .power(6)
        .build()
        .unwrap();
    assert!(sd2pp2p6.is_identity());

    let s2_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(2))
        .proper_power(1)
        .axis(Vector3::new(2.0, 2.0, 1.0))
        .kind(SymmetryElementKind::ImproperMirrorPlane)
        .build()
        .unwrap();

    let s2 = SymmetryOperation::builder()
        .generating_element(s2_element.clone())
        .power(1)
        .build()
        .unwrap();
    assert!(s2.is_inversion());

    let s2p2 = SymmetryOperation::builder()
        .generating_element(s2_element)
        .power(2)
        .build()
        .unwrap();
    assert!(s2p2.is_identity());

    let s2pp2_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(2))
        .proper_power(2)
        .axis(Vector3::new(2.0, 2.0, 1.0))
        .kind(SymmetryElementKind::ImproperMirrorPlane)
        .build()
        .unwrap();

    let s2pp2 = SymmetryOperation::builder()
        .generating_element(s2pp2_element.clone())
        .power(1)
        .build()
        .unwrap();
    assert!(s2pp2.is_reflection());

    let s2pp2p4 = SymmetryOperation::builder()
        .generating_element(s2pp2_element)
        .power(4)
        .build()
        .unwrap();
    assert!(s2pp2p4.is_identity());

    let sd1_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(1))
        .proper_power(1)
        .axis(Vector3::new(1.0, 1.0, 1.0))
        .kind(SymmetryElementKind::ImproperInversionCentre)
        .build()
        .unwrap();

    let sd1 = SymmetryOperation::builder()
        .generating_element(sd1_element.clone())
        .power(1)
        .build()
        .unwrap();
    assert!(sd1.is_inversion());

    let sd1p2 = SymmetryOperation::builder()
        .generating_element(sd1_element)
        .power(-2)
        .build()
        .unwrap();
    assert!(sd1p2.is_identity());

    let s3_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(3))
        .proper_power(1)
        .axis(Vector3::new(2.0, 2.0, 1.0))
        .kind(SymmetryElementKind::ImproperMirrorPlane)
        .build()
        .unwrap();

    let s3p3 = SymmetryOperation::builder()
        .generating_element(s3_element.clone())
        .power(-3)
        .build()
        .unwrap();
    assert!(s3p3.is_reflection());

    let s3p6 = SymmetryOperation::builder()
        .generating_element(s3_element)
        .power(6)
        .build()
        .unwrap();
    assert!(s3p6.is_identity());

    let s3pp2_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(3))
        .proper_power(2)
        .axis(Vector3::new(2.0, 2.0, 1.0))
        .kind(SymmetryElementKind::ImproperMirrorPlane)
        .build()
        .unwrap();

    let s3pp2p3 = SymmetryOperation::builder()
        .generating_element(s3pp2_element.clone())
        .power(3)
        .build()
        .unwrap();
    assert!(s3pp2p3.is_reflection());

    let s3pp2p6 = SymmetryOperation::builder()
        .generating_element(s3pp2_element)
        .power(-6)
        .build()
        .unwrap();
    assert!(s3pp2p6.is_identity());

    let s3pp3_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(3))
        .proper_power(3)
        .axis(Vector3::new(2.0, 2.0, 1.0))
        .kind(SymmetryElementKind::ImproperMirrorPlane)
        .build()
        .unwrap();

    let s3pp3 = SymmetryOperation::builder()
        .generating_element(s3pp3_element.clone())
        .power(1)
        .build()
        .unwrap();
    assert!(s3pp3.is_reflection());

    let s3pp3p2 = SymmetryOperation::builder()
        .generating_element(s3pp3_element)
        .power(2)
        .build()
        .unwrap();
    assert!(s3pp3p2.is_identity());

    let sd3_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(3))
        .proper_power(1)
        .axis(Vector3::new(1.0, 1.0, 1.0))
        .kind(SymmetryElementKind::ImproperInversionCentre)
        .build()
        .unwrap();

    let sd3p3 = SymmetryOperation::builder()
        .generating_element(sd3_element.clone())
        .power(3)
        .build()
        .unwrap();
    assert!(sd3p3.is_inversion());

    let sd3p6 = SymmetryOperation::builder()
        .generating_element(sd3_element)
        .power(6)
        .build()
        .unwrap();
    assert!(sd3p6.is_identity());

    let si_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Inf)
        .axis(Vector3::new(1.0, 0.0, 1.0))
        .proper_angle(2.0 * std::f64::consts::FRAC_PI_4)
        .kind(SymmetryElementKind::ImproperMirrorPlane)
        .build()
        .unwrap();

    let sip2 = SymmetryOperation::builder()
        .generating_element(si_element.clone())
        .power(2)
        .build()
        .unwrap();
    assert!(sip2.is_binary_rotation());

    let sip4 = SymmetryOperation::builder()
        .generating_element(si_element.clone())
        .power(4)
        .build()
        .unwrap();
    assert!(sip4.is_identity());

    let sib_element =
        si_element.convert_to_improper_kind(&SymmetryElementKind::ImproperInversionCentre, false);
    let sibp2 = SymmetryOperation::builder()
        .generating_element(sib_element.clone())
        .power(2)
        .build()
        .unwrap();
    assert!(sibp2.is_binary_rotation());

    let sibp4 = SymmetryOperation::builder()
        .generating_element(sib_element)
        .power(4)
        .build()
        .unwrap();
    assert!(sibp4.is_identity());
}

// #[test]
// fn test_finite_symmetry_element_improper_conversion() {
//     let s1 = SymmetryElement::builder()
//         .threshold(1e-14)
//         .proper_order(ElementOrder::Int(1))
//         .proper_power(1)
//         .axis(Vector3::new(0.0, 2.0, 0.0))
//         .kind(SymmetryElementKind::ImproperMirrorPlane)
//         .build()
//         .unwrap();
//     let sd2 = s1.convert_to_improper_kind(&SymmetryElementKind::ImproperInversionCentre, false);
//     assert_eq!(format!("{}", &sd2), "σ(+0.000, +1.000, +0.000)");

//     let sd1 = SymmetryElement::builder()
//         .threshold(1e-14)
//         .proper_order(ElementOrder::Int(1))
//         .proper_power(1)
//         .axis(Vector3::new(1.0, 1.0, 1.0))
//         .kind(SymmetryElementKind::ImproperInversionCentre)
//         .build()
//         .unwrap();
//     let s2 = sd1.convert_to_improper_kind(&SymmetryElementKind::ImproperMirrorPlane, false);
//     assert_eq!(format!("{}", &s2), "i");

//     let s3 = SymmetryElement::builder()
//         .threshold(1e-14)
//         .proper_order(ElementOrder::Int(3))
//         .proper_power(1)
//         .axis(Vector3::new(2.0, 2.0, 1.0))
//         .kind(SymmetryElementKind::ImproperMirrorPlane)
//         .build()
//         .unwrap();
//     let sd6 = s3.convert_to_improper_kind(&SymmetryElementKind::ImproperInversionCentre, false);
//     assert_eq!(format!("{}", &sd6), "Ṡ6(+0.667, +0.667, +0.333)");

//     let s4 = SymmetryElement::builder()
//         .threshold(1e-14)
//         .proper_order(ElementOrder::Int(4))
//         .proper_power(1)
//         .axis(Vector3::new(1.0, 1.0, 1.0))
//         .kind(SymmetryElementKind::ImproperMirrorPlane)
//         .build()
//         .unwrap();
//     let sd4 = s4.convert_to_improper_kind(&SymmetryElementKind::ImproperInversionCentre, false);
//     assert_eq!(sd4.proper_order, ElementOrder::Int(4));
//     assert_eq!(format!("{}", &sd4), "Ṡ4(+0.577, +0.577, +0.577)");

//     let sd5 = SymmetryElement::builder()
//         .threshold(1e-14)
//         .proper_order(ElementOrder::Int(5))
//         .proper_power(1)
//         .axis(Vector3::new(1.0, 2.0, 1.0))
//         .kind(SymmetryElementKind::ImproperInversionCentre)
//         .build()
//         .unwrap();
//     let s10 = sd5.convert_to_improper_kind(&SymmetryElementKind::ImproperMirrorPlane, false);
//     assert_eq!(s10.proper_order, ElementOrder::Int(10));
//     assert_eq!(format!("{}", &s10), "S10(+0.408, +0.816, +0.408)");

//     let sd7 = SymmetryElement::builder()
//         .threshold(1e-14)
//         .proper_order(ElementOrder::Int(7))
//         .proper_power(1)
//         .axis(Vector3::new(1.0, 1.0, 1.0))
//         .kind(SymmetryElementKind::ImproperInversionCentre)
//         .build()
//         .unwrap();
//     let s14 = sd7.convert_to_improper_kind(&SymmetryElementKind::ImproperMirrorPlane, false);
//     assert_eq!(s14.proper_order, ElementOrder::Int(14));
// }

// #[test]
// fn test_finite_symmetry_element_comparison() {
//     // ===========
//     // Proper only
//     // ===========
//     let c1 = SymmetryElement::builder()
//         .threshold(1e-14)
//         .proper_order(ElementOrder::Int(1))
//         .proper_power(1)
//         .axis(Vector3::new(0.0, 2.0, 0.0))
//         .kind(SymmetryElementKind::Proper)
//         .build()
//         .unwrap();
//     let c1p = SymmetryElement::builder()
//         .threshold(1e-14)
//         .proper_order(ElementOrder::Int(1))
//         .proper_power(1)
//         .axis(Vector3::new(1.0, 2.0, 0.0))
//         .kind(SymmetryElementKind::Proper)
//         .build()
//         .unwrap();
//     assert_eq!(c1, c1p);

//     let c1p2 = SymmetryElement::builder()
//         .threshold(1e-14)
//         .proper_order(ElementOrder::Int(1))
//         .proper_power(2)
//         .axis(Vector3::new(4.0, 2.0, 0.0))
//         .kind(SymmetryElementKind::Proper)
//         .build()
//         .unwrap();
//     assert_eq!(c1, c1p2);

//     let c2 = SymmetryElement::builder()
//         .threshold(1e-14)
//         .proper_order(ElementOrder::Int(2))
//         .proper_power(1)
//         .axis(Vector3::new(1.0, 2.0, 0.0))
//         .kind(SymmetryElementKind::Proper)
//         .build()
//         .unwrap();
//     let c2p = SymmetryElement::builder()
//         .threshold(1e-14)
//         .proper_order(ElementOrder::Int(2))
//         .proper_power(1)
//         .axis(-Vector3::new(1.0, 2.0, 0.0))
//         .kind(SymmetryElementKind::Proper)
//         .build()
//         .unwrap();
//     assert_eq!(c2, c2p);
//     assert_ne!(c1, c2);

//     let c3 = SymmetryElement::builder()
//         .threshold(1e-14)
//         .proper_order(ElementOrder::Int(3))
//         .proper_power(1)
//         .axis(Vector3::new(1.0, 2.0, 0.0))
//         .kind(SymmetryElementKind::Proper)
//         .build()
//         .unwrap();
//     let c3p = SymmetryElement::builder()
//         .threshold(1e-14)
//         .proper_order(ElementOrder::Int(3))
//         .proper_power(1)
//         .axis(-Vector3::new(1.0, 2.0, 0.0))
//         .kind(SymmetryElementKind::Proper)
//         .build()
//         .unwrap();
//     assert_eq!(c3, c3p);

//     // =============
//     // Improper only
//     // =============
//     let s1 = SymmetryElement::builder()
//         .threshold(1e-3)
//         .proper_order(ElementOrder::Int(1))
//         .proper_power(1)
//         .axis(Vector3::new(0.0, 2.0, 0.0))
//         .kind(SymmetryElementKind::ImproperMirrorPlane)
//         .build()
//         .unwrap();
//     let sd2 = s1.convert_to_improper_kind(&SymmetryElementKind::ImproperInversionCentre, false);
//     assert_eq!(s1, sd2);

//     let sd1 = SymmetryElement::builder()
//         .threshold(1e-14)
//         .proper_order(ElementOrder::Int(1))
//         .proper_power(1)
//         .axis(Vector3::new(1.0, 1.0, 1.0))
//         .kind(SymmetryElementKind::ImproperInversionCentre)
//         .build()
//         .unwrap();
//     let s2 = sd1.convert_to_improper_kind(&SymmetryElementKind::ImproperMirrorPlane, false);
//     assert_eq!(sd1, s2);
//     assert_ne!(sd1, sd2);
//     assert_ne!(s1, s2);

//     let s3 = SymmetryElement::builder()
//         .threshold(1e-14)
//         .proper_order(ElementOrder::Int(3))
//         .proper_power(1)
//         .axis(Vector3::new(2.0, 2.0, 1.0))
//         .kind(SymmetryElementKind::ImproperMirrorPlane)
//         .build()
//         .unwrap();
//     let sd6 = s3.convert_to_improper_kind(&SymmetryElementKind::ImproperInversionCentre, false);
//     assert_eq!(sd6.proper_order, ElementOrder::Int(6));
//     assert_eq!(s3, sd6);

//     let s3p = SymmetryElement::builder()
//         .threshold(1e-14)
//         .proper_order(ElementOrder::Int(3))
//         .proper_power(1)
//         .axis(-Vector3::new(2.0, 2.0, 1.0))
//         .kind(SymmetryElementKind::ImproperMirrorPlane)
//         .build()
//         .unwrap();
//     assert_eq!(s3, s3p);
//     assert_ne!(s2, s3);

//     // ===================
//     // Proper and improper
//     // ===================
//     assert_ne!(c2, s2);
//     assert_ne!(c2, sd2);
// }

// #[test]
// fn test_finite_symmetry_element_power_comparison() {
//     // ===========
//     // Proper only
//     // ===========
//     let c1 = SymmetryElement::builder()
//         .threshold(1e-14)
//         .proper_order(ElementOrder::Int(1))
//         .proper_power(1)
//         .axis(Vector3::new(0.0, 2.0, 0.0))
//         .kind(SymmetryElementKind::Proper)
//         .build()
//         .unwrap();
//     let c1p = SymmetryElement::builder()
//         .threshold(1e-14)
//         .proper_order(ElementOrder::Int(1))
//         .proper_power(2)
//         .axis(Vector3::new(1.0, 2.0, 0.0))
//         .kind(SymmetryElementKind::Proper)
//         .build()
//         .unwrap();
//     let c1p2 = SymmetryElement::builder()
//         .threshold(1e-14)
//         .proper_order(ElementOrder::Int(1))
//         .proper_power(0)
//         .axis(-Vector3::new(1.0, 2.0, 0.0))
//         .kind(SymmetryElementKind::Proper)
//         .build()
//         .unwrap();
//     assert_eq!(c1, c1p);
//     assert_eq!(c1, c1p2);
//     assert_eq!(c1p, c1p2);

//     let c2 = SymmetryElement::builder()
//         .threshold(1e-14)
//         .proper_order(ElementOrder::Int(2))
//         .proper_power(1)
//         .axis(Vector3::new(1.0, 2.0, 0.0))
//         .kind(SymmetryElementKind::Proper)
//         .build()
//         .unwrap();
//     assert!(!c2.is_identity());
//     let c2p = SymmetryElement::builder()
//         .threshold(1e-14)
//         .proper_order(ElementOrder::Int(2))
//         .proper_power(2)
//         .axis(Vector3::new(1.0, 2.0, 0.0))
//         .kind(SymmetryElementKind::Proper)
//         .build()
//         .unwrap();
//     assert!(c2p.is_identity());
//     let c2p2 = SymmetryElement::builder()
//         .threshold(1e-14)
//         .proper_order(ElementOrder::Int(2))
//         .proper_power(3)
//         .axis(-Vector3::new(1.0, 2.0, 0.0))
//         .kind(SymmetryElementKind::Proper)
//         .build()
//         .unwrap();
//     assert!(!c2p2.is_identity());
//     assert_eq!(c2, c2p2);
//     assert_eq!(c1, c2p);
//     assert_ne!(c1, c2p2);

//     let c4 = SymmetryElement::builder()
//         .threshold(1e-14)
//         .proper_order(ElementOrder::Int(4))
//         .proper_power(1)
//         .axis(Vector3::new(1.0, 1.0, 1.0))
//         .kind(SymmetryElementKind::Proper)
//         .build()
//         .unwrap();
//     let c4p2 = SymmetryElement::builder()
//         .threshold(1e-14)
//         .proper_order(ElementOrder::Int(4))
//         .proper_power(2)
//         .axis(-Vector3::new(1.0, 2.0, 0.0))
//         .kind(SymmetryElementKind::Proper)
//         .build()
//         .unwrap();
//     let c4p3 = SymmetryElement::builder()
//         .threshold(1e-14)
//         .proper_order(ElementOrder::Int(4))
//         .proper_power(3)
//         .axis(-Vector3::new(1.0, 1.0, 1.0))
//         .kind(SymmetryElementKind::Proper)
//         .build()
//         .unwrap();
//     let c4p4 = SymmetryElement::builder()
//         .threshold(1e-14)
//         .proper_order(ElementOrder::Int(4))
//         .proper_power(4)
//         .axis(-Vector3::new(2.0, 1.0, 1.0))
//         .kind(SymmetryElementKind::Proper)
//         .build()
//         .unwrap();
//     let c4p5 = SymmetryElement::builder()
//         .threshold(1e-14)
//         .proper_order(ElementOrder::Int(4))
//         .proper_power(5)
//         .axis(-Vector3::new(1.0, 1.0, 1.0))
//         .kind(SymmetryElementKind::Proper)
//         .build()
//         .unwrap();
//     assert!(c4p4.is_identity());
//     assert_eq!(c4p2, c2);
//     assert_eq!(c4p4, c1);
//     assert_eq!(c4p4, c1p2);
//     assert_eq!(c4, c4p3);
//     assert_eq!(c4, c4p5);

//     let c5 = SymmetryElement::builder()
//         .threshold(1e-14)
//         .proper_order(ElementOrder::Int(5))
//         .proper_power(1)
//         .axis(Vector3::new(2.0, 2.0, 1.0))
//         .kind(SymmetryElementKind::Proper)
//         .build()
//         .unwrap();
//     let c5p9 = SymmetryElement::builder()
//         .threshold(1e-14)
//         .proper_order(ElementOrder::Int(5))
//         .proper_power(9)
//         .axis(-Vector3::new(2.0, 2.0, 1.0))
//         .kind(SymmetryElementKind::Proper)
//         .build()
//         .unwrap();
//     assert_eq!(c5, c5p9);

//     // =============
//     // Improper only
//     // =============
//     let s1 = SymmetryElement::builder()
//         .threshold(1e-3)
//         .proper_order(ElementOrder::Int(1))
//         .proper_power(1)
//         .axis(Vector3::new(0.0, 2.0, 0.0))
//         .kind(SymmetryElementKind::ImproperMirrorPlane)
//         .build()
//         .unwrap();
//     assert!(s1.is_mirror_plane());
//     let s1p2 = SymmetryElement::builder()
//         .threshold(1e-3)
//         .proper_order(ElementOrder::Int(1))
//         .proper_power(2)
//         .axis(Vector3::new(0.0, 2.0, 0.0))
//         .kind(SymmetryElementKind::ImproperMirrorPlane)
//         .build()
//         .unwrap();
//     assert!(s1p2.is_mirror_plane());
//     assert_eq!(s1, s1p2);
//     let sd2 = s1.convert_to_improper_kind(&SymmetryElementKind::ImproperInversionCentre, false);
//     let sd2p = s1p2.convert_to_improper_kind(&SymmetryElementKind::ImproperInversionCentre, false);
//     assert_eq!(sd2, sd2p);

//     let s2 = SymmetryElement::builder()
//         .threshold(1e-3)
//         .proper_order(ElementOrder::Int(2))
//         .proper_power(1)
//         .axis(Vector3::new(0.0, 2.0, 0.0))
//         .kind(SymmetryElementKind::ImproperMirrorPlane)
//         .build()
//         .unwrap();
//     assert!(s2.is_inversion_centre());
//     let s2p2 = SymmetryElement::builder()
//         .threshold(1e-3)
//         .proper_order(ElementOrder::Int(2))
//         .proper_power(2)
//         .axis(Vector3::new(0.0, 2.0, 0.0))
//         .kind(SymmetryElementKind::ImproperMirrorPlane)
//         .build()
//         .unwrap();
//     assert!(s2p2.is_mirror_plane());
//     let s2p3 = SymmetryElement::builder()
//         .threshold(1e-3)
//         .proper_order(ElementOrder::Int(2))
//         .proper_power(3)
//         .axis(Vector3::new(0.0, 2.0, 0.0))
//         .kind(SymmetryElementKind::ImproperMirrorPlane)
//         .build()
//         .unwrap();
//     assert!(s2p3.is_inversion_centre());
//     assert_eq!(s1, s2p2);
//     assert_eq!(s2, s2p3);

//     let sd2 = SymmetryElement::builder()
//         .threshold(1e-3)
//         .proper_order(ElementOrder::Int(2))
//         .proper_power(1)
//         .axis(Vector3::new(0.0, 2.0, 0.0))
//         .kind(SymmetryElementKind::ImproperInversionCentre)
//         .build()
//         .unwrap();
//     assert!(sd2.is_mirror_plane());
//     let sd2b = sd2.convert_to_improper_kind(&SymmetryElementKind::ImproperMirrorPlane, true);
//     let sd2p2 = SymmetryElement::builder()
//         .threshold(1e-3)
//         .proper_order(ElementOrder::Int(2))
//         .proper_power(2)
//         .axis(Vector3::new(1.0, 2.0, 0.0))
//         .kind(SymmetryElementKind::ImproperInversionCentre)
//         .build()
//         .unwrap();
//     assert!(sd2p2.is_inversion_centre());
//     let sd2p2b = sd2p2.convert_to_improper_kind(&SymmetryElementKind::ImproperMirrorPlane, true);
//     assert_eq!(sd2, s1);
//     assert_eq!(sd2p2, s2);
//     assert_eq!(sd2b, sd2);
//     assert_eq!(sd2p2b, sd2p2);

//     let s3 = SymmetryElement::builder()
//         .threshold(1e-3)
//         .proper_order(ElementOrder::Int(3))
//         .proper_power(1)
//         .axis(Vector3::new(1.0, 2.0, 1.0))
//         .kind(SymmetryElementKind::ImproperMirrorPlane)
//         .build()
//         .unwrap();
//     let s3p2 = SymmetryElement::builder()
//         .threshold(1e-3)
//         .proper_order(ElementOrder::Int(3))
//         .proper_power(2)
//         .axis(Vector3::new(1.0, 2.0, 1.0))
//         .kind(SymmetryElementKind::ImproperMirrorPlane)
//         .build()
//         .unwrap();
//     let s3p4 = SymmetryElement::builder()
//         .threshold(1e-3)
//         .proper_order(ElementOrder::Int(3))
//         .proper_power(4)
//         .axis(Vector3::new(1.0, 2.0, 1.0))
//         .kind(SymmetryElementKind::ImproperMirrorPlane)
//         .build()
//         .unwrap();
//     let s3p3 = SymmetryElement::builder()
//         .threshold(1e-3)
//         .proper_order(ElementOrder::Int(3))
//         .proper_power(3)
//         .axis(Vector3::new(1.0, 2.0, 1.0))
//         .kind(SymmetryElementKind::ImproperMirrorPlane)
//         .build()
//         .unwrap();
//     assert!(s3p3.is_mirror_plane());
//     assert_eq!(s3, s3p2);
//     assert_eq!(s3, s3p4);

//     let s6p2 = SymmetryElement::builder()
//         .threshold(1e-3)
//         .proper_order(ElementOrder::Int(6))
//         .proper_power(2)
//         .axis(Vector3::new(1.0, 2.0, 1.0))
//         .kind(SymmetryElementKind::ImproperMirrorPlane)
//         .build()
//         .unwrap();
//     let s6p3 = SymmetryElement::builder()
//         .threshold(1e-3)
//         .proper_order(ElementOrder::Int(6))
//         .proper_power(3)
//         .axis(Vector3::new(2.0, 2.0, 1.0))
//         .kind(SymmetryElementKind::ImproperMirrorPlane)
//         .build()
//         .unwrap();
//     let s6p4 = SymmetryElement::builder()
//         .threshold(1e-3)
//         .proper_order(ElementOrder::Int(6))
//         .proper_power(4)
//         .axis(-Vector3::new(1.0, 2.0, 1.0))
//         .kind(SymmetryElementKind::ImproperMirrorPlane)
//         .build()
//         .unwrap();
//     assert!(s6p3.is_inversion_centre());
//     assert_eq!(s3, s6p2);
//     assert_eq!(s3, s6p4);

//     // ===================
//     // Proper and improper
//     // ===================
//     assert_ne!(c2, s2);
// }

// #[test]
// fn test_finite_symmetry_element_hashset() {
//     let mut element_set = HashSet::new();
//     let c1 = SymmetryElement::builder()
//         .threshold(1e-14)
//         .proper_order(ElementOrder::Int(1))
//         .proper_power(1)
//         .axis(Vector3::new(0.0, 2.0, 0.0))
//         .kind(SymmetryElementKind::Proper)
//         .build()
//         .unwrap();
//     element_set.insert(c1);
//     assert_eq!(element_set.len(), 1);

//     let c1p = SymmetryElement::builder()
//         .threshold(1e-14)
//         .proper_order(ElementOrder::Int(1))
//         .proper_power(1)
//         .axis(Vector3::new(1.0, 2.0, 0.0))
//         .kind(SymmetryElementKind::Proper)
//         .build()
//         .unwrap();
//     element_set.insert(c1p);
//     assert_eq!(element_set.len(), 1);

//     let c2 = SymmetryElement::builder()
//         .threshold(1e-14)
//         .proper_order(ElementOrder::Int(2))
//         .proper_power(1)
//         .axis(Vector3::new(1.0, 2.0, 0.0))
//         .kind(SymmetryElementKind::Proper)
//         .build()
//         .unwrap();
//     element_set.insert(c2);
//     assert_eq!(element_set.len(), 2);

//     let c2p = SymmetryElement::builder()
//         .threshold(1e-14)
//         .proper_order(ElementOrder::Int(2))
//         .proper_power(1)
//         .axis(-Vector3::new(1.0, 2.0, 0.0))
//         .kind(SymmetryElementKind::Proper)
//         .build()
//         .unwrap();
//     element_set.insert(c2p);
//     assert_eq!(element_set.len(), 2);

//     let c3 = SymmetryElement::builder()
//         .threshold(1e-14)
//         .proper_order(ElementOrder::Int(3))
//         .proper_power(1)
//         .axis(Vector3::new(1.0, 2.0, 0.0))
//         .kind(SymmetryElementKind::Proper)
//         .build()
//         .unwrap();
//     element_set.insert(c3);
//     assert_eq!(element_set.len(), 3);

//     let c3p = SymmetryElement::builder()
//         .threshold(1e-14)
//         .proper_order(ElementOrder::Int(3))
//         .proper_power(1)
//         .axis(-Vector3::new(1.0, 2.0, 0.0))
//         .kind(SymmetryElementKind::Proper)
//         .build()
//         .unwrap();
//     element_set.insert(c3p);
//     assert_eq!(element_set.len(), 3);

//     let s1 = SymmetryElement::builder()
//         .threshold(1e-14)
//         .proper_order(ElementOrder::Int(1))
//         .proper_power(1)
//         .axis(Vector3::new(0.0, 2.0, 0.0))
//         .kind(SymmetryElementKind::ImproperMirrorPlane)
//         .build()
//         .unwrap();
//     let sd2 = s1.convert_to_improper_kind(&SymmetryElementKind::ImproperInversionCentre, false);
//     element_set.insert(s1);
//     assert_eq!(element_set.len(), 4);
//     element_set.insert(sd2);
//     assert_eq!(element_set.len(), 4);

//     let sd1 = SymmetryElement::builder()
//         .threshold(1e-14)
//         .proper_order(ElementOrder::Int(1))
//         .proper_power(1)
//         .axis(Vector3::new(1.0, 1.0, 1.0))
//         .kind(SymmetryElementKind::ImproperInversionCentre)
//         .build()
//         .unwrap();
//     let s2 = sd1.convert_to_improper_kind(&SymmetryElementKind::ImproperMirrorPlane, false);
//     element_set.insert(sd1);
//     assert_eq!(element_set.len(), 5);
//     element_set.insert(s2);
//     assert_eq!(element_set.len(), 5);

//     let s3 = SymmetryElement::builder()
//         .threshold(1e-14)
//         .proper_order(ElementOrder::Int(3))
//         .proper_power(1)
//         .axis(Vector3::new(2.0, 2.0, 1.0))
//         .kind(SymmetryElementKind::ImproperMirrorPlane)
//         .build()
//         .unwrap();
//     let sd6 = s3.convert_to_improper_kind(&SymmetryElementKind::ImproperInversionCentre, false);
//     element_set.insert(s3);
//     assert_eq!(element_set.len(), 6);
//     element_set.insert(sd6);
//     assert_eq!(element_set.len(), 6);

//     let s3p = SymmetryElement::builder()
//         .threshold(1e-14)
//         .proper_order(ElementOrder::Int(3))
//         .proper_power(1)
//         .axis(-Vector3::new(2.0, 2.0, 1.0))
//         .kind(SymmetryElementKind::ImproperMirrorPlane)
//         .build()
//         .unwrap();
//     element_set.insert(s3p);
//     assert_eq!(element_set.len(), 6);
// }

// #[test]
// fn test_infinite_symmetry_element_comparison() {
//     // ========================
//     // Proper symmetry elements
//     // ========================
//     let ci1 = SymmetryElement::builder()
//         .threshold(1e-14)
//         .proper_order(ElementOrder::Inf)
//         .axis(Vector3::new(0.0, 2.0, 0.0))
//         .kind(SymmetryElementKind::Proper)
//         .build()
//         .unwrap();

//     let ci2 = SymmetryElement::builder()
//         .threshold(1e-7)
//         .proper_order(ElementOrder::Inf)
//         .axis(Vector3::new(0.0, 2.0, 0.0))
//         .kind(SymmetryElementKind::Proper)
//         .build()
//         .unwrap();
//     assert_eq!(ci1, ci2);

//     let ci3 = SymmetryElement::builder()
//         .threshold(1e-14)
//         .proper_order(ElementOrder::Inf)
//         .proper_angle(std::f64::consts::FRAC_PI_3)
//         .axis(Vector3::new(0.0, 2.0, 1.0))
//         .kind(SymmetryElementKind::Proper)
//         .build()
//         .unwrap();

//     let ci4 = SymmetryElement::builder()
//         .threshold(1e-14)
//         .proper_order(ElementOrder::Inf)
//         .proper_angle(2.0 * std::f64::consts::PI - std::f64::consts::FRAC_PI_3)
//         .axis(-Vector3::new(0.0, 2.0, 1.0))
//         .kind(SymmetryElementKind::Proper)
//         .build()
//         .unwrap();
//     let ci4b = SymmetryElement::builder()
//         .threshold(1e-14)
//         .proper_order(ElementOrder::Inf)
//         .proper_angle(2.0 * std::f64::consts::PI - std::f64::consts::FRAC_PI_3)
//         .axis(Vector3::new(0.0, 2.0, 1.0))
//         .kind(SymmetryElementKind::Proper)
//         .build()
//         .unwrap();
//     assert_eq!(ci3, ci4);
//     assert_eq!(ci3, ci4b);

//     // ==========================
//     // Improper symmetry elements
//     // ==========================
//     let si1 = SymmetryElement::builder()
//         .threshold(1e-14)
//         .proper_order(ElementOrder::Inf)
//         .axis(Vector3::new(0.0, 2.0, 0.0))
//         .kind(SymmetryElementKind::ImproperMirrorPlane)
//         .build()
//         .unwrap();

//     let si2 = SymmetryElement::builder()
//         .threshold(1e-7)
//         .proper_order(ElementOrder::Inf)
//         .axis(Vector3::new(0.0, 2.0, 0.0))
//         .kind(SymmetryElementKind::ImproperMirrorPlane)
//         .build()
//         .unwrap();
//     let si2c = si2.convert_to_improper_kind(&SymmetryElementKind::ImproperInversionCentre, true);
//     let si2b = SymmetryElement::builder()
//         .threshold(1e-7)
//         .proper_order(ElementOrder::Inf)
//         .axis(Vector3::new(0.0, 2.0, 0.0))
//         .kind(SymmetryElementKind::ImproperInversionCentre)
//         .build()
//         .unwrap();
//     assert_eq!(si1, si2);
//     assert_eq!(si1, si2b); // No proper angle specified, both conventions are the same.
//     assert_eq!(si1, si2c);

//     let si3 = SymmetryElement::builder()
//         .threshold(1e-14)
//         .proper_order(ElementOrder::Inf)
//         .proper_angle(std::f64::consts::FRAC_PI_4)
//         .axis(Vector3::new(1.0, 2.0, 1.0))
//         .kind(SymmetryElementKind::ImproperMirrorPlane)
//         .build()
//         .unwrap();
//     let si3b = si3.convert_to_improper_kind(&SymmetryElementKind::ImproperInversionCentre, false);
//     let si3c = si3.convert_to_improper_kind(&SymmetryElementKind::ImproperInversionCentre, true);
//     assert_eq!(si3, si3b);
//     assert_eq!(si3, si3c);

//     let si4 = SymmetryElement::builder()
//         .threshold(1e-14)
//         .proper_order(ElementOrder::Inf)
//         .proper_angle(2.0 * std::f64::consts::PI - std::f64::consts::FRAC_PI_4)
//         .axis(Vector3::new(1.0, 2.0, 1.0))
//         .kind(SymmetryElementKind::ImproperMirrorPlane)
//         .build()
//         .unwrap();
//     assert_eq!(si3, si4);

//     let si5 = SymmetryElement::builder()
//         .threshold(1e-14)
//         .proper_order(ElementOrder::Inf)
//         .proper_angle(2.0 * std::f64::consts::PI - std::f64::consts::FRAC_PI_4)
//         .axis(Vector3::new(1.0, 2.0, 1.0))
//         .kind(SymmetryElementKind::ImproperInversionCentre)
//         .build()
//         .unwrap();
//     let si5b = si5.convert_to_improper_kind(&SymmetryElementKind::ImproperMirrorPlane, true);
//     assert_ne!(si3, si5);
//     assert_eq!(si5, si5b);
// }
