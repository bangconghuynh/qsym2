use nalgebra::{Point3, Vector3};

use crate::symmetry::symmetry_element::{
    ElementOrder, SymmetryElement, SymmetryElementKind, SymmetryOperation, INV, SIG,
};
use fraction;
type F = fraction::Fraction;

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
    approx::assert_relative_eq!(
        c1.total_proper_angle,
        0.0,
        max_relative = c1.generating_element.threshold,
        epsilon = c1.generating_element.threshold
    );

    let c1b = SymmetryOperation::builder()
        .generating_element(c1_element)
        .power(-3)
        .build()
        .unwrap();
    assert!(c1b.is_identity());
    approx::assert_relative_eq!(
        c1b.total_proper_angle,
        0.0,
        max_relative = c1b.generating_element.threshold,
        epsilon = c1b.generating_element.threshold
    );

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
    approx::assert_relative_eq!(c2.total_proper_angle, std::f64::consts::PI);

    let c2p2 = SymmetryOperation::builder()
        .generating_element(c2_element.clone())
        .power(2)
        .build()
        .unwrap();
    assert!(c2p2.is_identity());
    approx::assert_relative_eq!(
        c2p2.total_proper_angle,
        0.0,
        max_relative = c2p2.generating_element.threshold,
        epsilon = c2p2.generating_element.threshold
    );

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
    approx::assert_relative_eq!(
        c2p2b.total_proper_angle,
        0.0,
        max_relative = c2p2b.generating_element.threshold,
        epsilon = c2p2b.generating_element.threshold
    );

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
    approx::assert_relative_eq!(
        c3.total_proper_angle,
        2.0 * std::f64::consts::FRAC_PI_3,
        max_relative = c3.generating_element.threshold,
        epsilon = c3.generating_element.threshold
    );

    let c3p3 = SymmetryOperation::builder()
        .generating_element(c3_element)
        .power(-3)
        .build()
        .unwrap();
    assert!(c3p3.is_identity());
    approx::assert_relative_eq!(
        c3p3.total_proper_angle,
        0.0,
        max_relative = c3p3.generating_element.threshold,
        epsilon = c3p3.generating_element.threshold
    );

    let c3pp2_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::new(3.0, 1e-14))
        .proper_power(2)
        .axis(Vector3::new(1.0, 1.0, 1.0))
        .kind(SymmetryElementKind::Proper)
        .build()
        .unwrap();

    let c3pp2p3 = SymmetryOperation::builder()
        .generating_element(c3pp2_element)
        .power(3)
        .build()
        .unwrap();
    assert!(c3pp2p3.is_identity());
    approx::assert_relative_eq!(
        c3pp2p3.total_proper_angle,
        0.0,
        max_relative = c3pp2p3.generating_element.threshold,
        epsilon = c3pp2p3.generating_element.threshold
    );

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
    approx::assert_relative_eq!(
        c4p2.total_proper_angle,
        std::f64::consts::PI,
        max_relative = c4p2.generating_element.threshold,
        epsilon = c4p2.generating_element.threshold
    );

    let c4p4 = SymmetryOperation::builder()
        .generating_element(c4_element)
        .power(-4)
        .build()
        .unwrap();
    assert!(c4p4.is_identity());
    approx::assert_relative_eq!(
        c4p4.total_proper_angle,
        0.0,
        max_relative = c4p4.generating_element.threshold,
        epsilon = c4p4.generating_element.threshold
    );

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
    approx::assert_relative_eq!(
        cip3.total_proper_angle,
        std::f64::consts::PI,
        max_relative = cip3.generating_element.threshold,
        epsilon = cip3.generating_element.threshold
    );

    let cip6 = SymmetryOperation::builder()
        .generating_element(ci_element.clone())
        .power(-6)
        .build()
        .unwrap();
    assert!(cip6.is_identity());
    approx::assert_relative_eq!(
        cip6.total_proper_angle,
        0.0,
        max_relative = cip6.generating_element.threshold,
        epsilon = cip6.generating_element.threshold
    );

    let cip0 = SymmetryOperation::builder()
        .generating_element(ci_element)
        .power(0)
        .build()
        .unwrap();
    assert!(cip0.is_identity());
    approx::assert_relative_eq!(
        cip0.total_proper_angle,
        0.0,
        max_relative = cip0.generating_element.threshold,
        epsilon = cip0.generating_element.threshold
    );

    // ============================
    // Improper symmetry operations
    // ============================
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
    approx::assert_relative_eq!(
        s1.total_proper_angle,
        0.0,
        max_relative = s1.generating_element.threshold,
        epsilon = s1.generating_element.threshold
    );

    let s1p2 = SymmetryOperation::builder()
        .generating_element(s1_element)
        .power(-2)
        .build()
        .unwrap();
    assert!(s1p2.is_identity());
    approx::assert_relative_eq!(
        s1p2.total_proper_angle,
        0.0,
        max_relative = s1p2.generating_element.threshold,
        epsilon = s1p2.generating_element.threshold
    );

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
    approx::assert_relative_eq!(
        sd2.total_proper_angle,
        std::f64::consts::PI,
        max_relative = sd2.generating_element.threshold,
        epsilon = sd2.generating_element.threshold
    );

    let sd2p2 = SymmetryOperation::builder()
        .generating_element(sd2_element)
        .power(2)
        .build()
        .unwrap();
    assert!(sd2p2.is_identity());
    approx::assert_relative_eq!(
        sd2p2.total_proper_angle,
        0.0,
        max_relative = sd2p2.generating_element.threshold,
        epsilon = sd2p2.generating_element.threshold
    );

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
    approx::assert_relative_eq!(
        sd2pp2.total_proper_angle,
        0.0,
        max_relative = sd2pp2.generating_element.threshold,
        epsilon = sd2pp2.generating_element.threshold
    );

    let sd2pp2p6 = SymmetryOperation::builder()
        .generating_element(sd2pp2_element)
        .power(6)
        .build()
        .unwrap();
    assert!(sd2pp2p6.is_identity());
    approx::assert_relative_eq!(
        sd2pp2p6.total_proper_angle,
        0.0,
        max_relative = sd2pp2p6.generating_element.threshold,
        epsilon = sd2pp2p6.generating_element.threshold
    );

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
    approx::assert_relative_eq!(
        s2.total_proper_angle,
        std::f64::consts::PI,
        max_relative = s2.generating_element.threshold,
        epsilon = s2.generating_element.threshold
    );

    let s2p2 = SymmetryOperation::builder()
        .generating_element(s2_element)
        .power(2)
        .build()
        .unwrap();
    assert!(s2p2.is_identity());
    approx::assert_relative_eq!(
        s2p2.total_proper_angle,
        0.0,
        max_relative = s2p2.generating_element.threshold,
        epsilon = s2p2.generating_element.threshold
    );

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
    approx::assert_relative_eq!(
        s2pp2.total_proper_angle,
        0.0,
        max_relative = s2pp2.generating_element.threshold,
        epsilon = s2pp2.generating_element.threshold
    );

    let s2pp2p4 = SymmetryOperation::builder()
        .generating_element(s2pp2_element)
        .power(4)
        .build()
        .unwrap();
    assert!(s2pp2p4.is_identity());
    approx::assert_relative_eq!(
        s2pp2p4.total_proper_angle,
        0.0,
        max_relative = s2pp2p4.generating_element.threshold,
        epsilon = s2pp2p4.generating_element.threshold
    );

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
    approx::assert_relative_eq!(
        sd1.total_proper_angle,
        0.0,
        max_relative = sd1.generating_element.threshold,
        epsilon = sd1.generating_element.threshold
    );

    let sd1p2 = SymmetryOperation::builder()
        .generating_element(sd1_element)
        .power(-2)
        .build()
        .unwrap();
    assert!(sd1p2.is_identity());
    approx::assert_relative_eq!(
        sd1p2.total_proper_angle,
        0.0,
        max_relative = sd1p2.generating_element.threshold,
        epsilon = sd1p2.generating_element.threshold
    );

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
    approx::assert_relative_eq!(
        s3p3.total_proper_angle,
        0.0,
        max_relative = s3p3.generating_element.threshold,
        epsilon = s3p3.generating_element.threshold
    );

    let s3p6 = SymmetryOperation::builder()
        .generating_element(s3_element)
        .power(6)
        .build()
        .unwrap();
    assert!(s3p6.is_identity());
    approx::assert_relative_eq!(
        s3p6.total_proper_angle,
        0.0,
        max_relative = s3p6.generating_element.threshold,
        epsilon = s3p6.generating_element.threshold
    );

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
    approx::assert_relative_eq!(
        s3pp2p3.total_proper_angle,
        0.0,
        max_relative = s3pp2p3.generating_element.threshold,
        epsilon = s3pp2p3.generating_element.threshold
    );

    let s3pp2p6 = SymmetryOperation::builder()
        .generating_element(s3pp2_element)
        .power(-6)
        .build()
        .unwrap();
    assert!(s3pp2p6.is_identity());
    approx::assert_relative_eq!(
        s3pp2p6.total_proper_angle,
        0.0,
        max_relative = s3pp2p6.generating_element.threshold,
        epsilon = s3pp2p6.generating_element.threshold
    );

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
    approx::assert_relative_eq!(
        s3pp3.total_proper_angle,
        0.0,
        max_relative = s3pp3.generating_element.threshold,
        epsilon = s3pp3.generating_element.threshold
    );

    let s3pp3p2 = SymmetryOperation::builder()
        .generating_element(s3pp3_element)
        .power(2)
        .build()
        .unwrap();
    assert!(s3pp3p2.is_identity());
    approx::assert_relative_eq!(
        s3pp3p2.total_proper_angle,
        0.0,
        max_relative = s3pp3p2.generating_element.threshold,
        epsilon = s3pp3p2.generating_element.threshold
    );

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
    approx::assert_relative_eq!(
        sd3p3.total_proper_angle,
        0.0,
        max_relative = sd3p3.generating_element.threshold,
        epsilon = sd3p3.generating_element.threshold
    );

    let sd3p6 = SymmetryOperation::builder()
        .generating_element(sd3_element)
        .power(6)
        .build()
        .unwrap();
    assert!(sd3p6.is_identity());
    approx::assert_relative_eq!(
        sd3p6.total_proper_angle,
        0.0,
        max_relative = sd3p6.generating_element.threshold,
        epsilon = sd3p6.generating_element.threshold
    );

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
    approx::assert_relative_eq!(
        sip2.total_proper_angle,
        std::f64::consts::PI,
        max_relative = sip2.generating_element.threshold,
        epsilon = sip2.generating_element.threshold
    );

    let sip4 = SymmetryOperation::builder()
        .generating_element(si_element.clone())
        .power(4)
        .build()
        .unwrap();
    assert!(sip4.is_identity());
    approx::assert_relative_eq!(
        sip4.total_proper_angle,
        0.0,
        max_relative = sip4.generating_element.threshold,
        epsilon = sip4.generating_element.threshold
    );

    let sib_element =
        si_element.convert_to_improper_kind(&SymmetryElementKind::ImproperInversionCentre, false);

    let sibp2 = SymmetryOperation::builder()
        .generating_element(sib_element.clone())
        .power(2)
        .build()
        .unwrap();
    assert!(sibp2.is_binary_rotation());
    approx::assert_relative_eq!(
        sibp2.total_proper_angle,
        std::f64::consts::PI,
        max_relative = sibp2.generating_element.threshold,
        epsilon = sibp2.generating_element.threshold
    );

    let sibp4 = SymmetryOperation::builder()
        .generating_element(sib_element)
        .power(4)
        .build()
        .unwrap();
    assert!(sibp4.is_identity());
    approx::assert_relative_eq!(
        sibp4.total_proper_angle,
        0.0,
        max_relative = sibp4.generating_element.threshold,
        epsilon = sibp4.generating_element.threshold
    );
}

#[test]
fn test_symmetry_operation_total_proper_fraction() {
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
    assert_eq!(c1.total_proper_fraction, Some(F::from(1u64)));

    let c1b = SymmetryOperation::builder()
        .generating_element(c1_element)
        .power(-3)
        .build()
        .unwrap();
    assert_eq!(c1b.total_proper_fraction, Some(F::from(1u64)));

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
    assert_eq!(c2.total_proper_fraction, Some(F::new(1u64, 2u64)));

    let c2p2 = SymmetryOperation::builder()
        .generating_element(c2_element.clone())
        .power(2)
        .build()
        .unwrap();
    assert_eq!(c2p2.total_proper_fraction, Some(F::from(1u64)));

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
    assert_eq!(c2p2b.total_proper_fraction, Some(F::from(1u64)));

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
    assert_eq!(c3.total_proper_fraction, Some(F::new(1u64, 3u64)));

    let c3p2 = SymmetryOperation::builder()
        .generating_element(c3_element.clone())
        .power(2)
        .build()
        .unwrap();
    assert_eq!(c3p2.total_proper_fraction, Some(F::new(2u64, 3u64)));

    let c3pm2 = SymmetryOperation::builder()
        .generating_element(c3_element.clone())
        .power(-2)
        .build()
        .unwrap();
    assert_eq!(c3pm2.total_proper_fraction, Some(F::new(1u64, 3u64)));

    let c3p3 = SymmetryOperation::builder()
        .generating_element(c3_element.clone())
        .power(3)
        .build()
        .unwrap();
    assert_eq!(c3p3.total_proper_fraction, Some(F::from(1u64)));

    let c3p4 = SymmetryOperation::builder()
        .generating_element(c3_element.clone())
        .power(4)
        .build()
        .unwrap();
    assert_eq!(c3p4.total_proper_fraction, Some(F::new(1u64, 3u64)));

    let c3pm4 = SymmetryOperation::builder()
        .generating_element(c3_element)
        .power(-4)
        .build()
        .unwrap();
    assert_eq!(c3pm4.total_proper_fraction, Some(F::new(2u64, 3u64)));

    let c3pp2_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::new(3.0, 1e-14))
        .proper_power(2)
        .axis(Vector3::new(1.0, 1.0, 1.0))
        .kind(SymmetryElementKind::Proper)
        .build()
        .unwrap();

    let c3pm6 = SymmetryOperation::builder()
        .generating_element(c3pp2_element)
        .power(-3)
        .build()
        .unwrap();
    assert_eq!(c3pm6.total_proper_fraction, Some(F::from(1u64)));

    let c4_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::new(4.0, 1e-14))
        .proper_power(1)
        .axis(Vector3::new(1.0, 1.0, 1.0))
        .kind(SymmetryElementKind::Proper)
        .build()
        .unwrap();

    let c4 = SymmetryOperation::builder()
        .generating_element(c4_element.clone())
        .power(1)
        .build()
        .unwrap();
    assert_eq!(c4.total_proper_fraction, Some(F::new(1u64, 4u64)));

    let c4p2 = SymmetryOperation::builder()
        .generating_element(c4_element.clone())
        .power(2)
        .build()
        .unwrap();
    assert_eq!(c4p2.total_proper_fraction, Some(F::new(1u64, 2u64)));

    let c4pm2 = SymmetryOperation::builder()
        .generating_element(c4_element.clone())
        .power(-2)
        .build()
        .unwrap();
    assert_eq!(c4pm2.total_proper_fraction, Some(F::new(1u64, 2u64)));

    let c4pm3 = SymmetryOperation::builder()
        .generating_element(c4_element.clone())
        .power(-3)
        .build()
        .unwrap();
    assert_eq!(c4pm3.total_proper_fraction, Some(F::new(1u64, 4u64)));

    let c4p4 = SymmetryOperation::builder()
        .generating_element(c4_element.clone())
        .power(4)
        .build()
        .unwrap();
    assert_eq!(c4p4.total_proper_fraction, Some(F::from(1u64)));

    let c7_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(7))
        .proper_power(1)
        .axis(Vector3::new(1.0, 1.0, 2.0))
        .kind(SymmetryElementKind::Proper)
        .build()
        .unwrap();

    let c7 = SymmetryOperation::builder()
        .generating_element(c7_element.clone())
        .power(1)
        .build()
        .unwrap();
    assert_eq!(c7.total_proper_fraction, Some(F::new(1u64, 7u64)));

    let c7p2 = SymmetryOperation::builder()
        .generating_element(c7_element.clone())
        .power(2)
        .build()
        .unwrap();
    assert_eq!(c7p2.total_proper_fraction, Some(F::new(2u64, 7u64)));

    let c7pm2 = SymmetryOperation::builder()
        .generating_element(c7_element.clone())
        .power(-2)
        .build()
        .unwrap();
    assert_eq!(c7pm2.total_proper_fraction, Some(F::new(5u64, 7u64)));

    let c7pm3 = SymmetryOperation::builder()
        .generating_element(c7_element.clone())
        .power(-3)
        .build()
        .unwrap();
    assert_eq!(c7pm3.total_proper_fraction, Some(F::new(4u64, 7u64)));

    let c7p4 = SymmetryOperation::builder()
        .generating_element(c7_element.clone())
        .power(4)
        .build()
        .unwrap();
    assert_eq!(c7p4.total_proper_fraction, Some(F::new(4u64, 7u64)));

    // ============================
    // Improper symmetry operations
    // ============================
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
    assert_eq!(s1.total_proper_fraction, Some(F::from(1u64)));

    let s1pm2 = SymmetryOperation::builder()
        .generating_element(s1_element)
        .power(-2)
        .build()
        .unwrap();
    assert_eq!(s1pm2.total_proper_fraction, Some(F::from(1u64)));

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
    assert_eq!(sd2.total_proper_fraction, Some(F::new(1u64, 2u64)));

    let sd2p2 = SymmetryOperation::builder()
        .generating_element(sd2_element)
        .power(2)
        .build()
        .unwrap();
    assert_eq!(sd2p2.total_proper_fraction, Some(F::from(1u64)));

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
    assert_eq!(sd2pp2.total_proper_fraction, Some(F::from(1u64)));

    let sd2pp2p6 = SymmetryOperation::builder()
        .generating_element(sd2pp2_element)
        .power(6)
        .build()
        .unwrap();
    assert_eq!(sd2pp2p6.total_proper_fraction, Some(F::from(1u64)));

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
    assert_eq!(s2.total_proper_fraction, Some(F::new(1u64, 2u64)));

    let s2p2 = SymmetryOperation::builder()
        .generating_element(s2_element)
        .power(2)
        .build()
        .unwrap();
    assert_eq!(s2p2.total_proper_fraction, Some(F::from(1u64)));

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
    assert_eq!(s2pp2.total_proper_fraction, Some(F::from(1u64)));

    let s2pp2p4 = SymmetryOperation::builder()
        .generating_element(s2pp2_element)
        .power(4)
        .build()
        .unwrap();
    assert_eq!(s2pp2p4.total_proper_fraction, Some(F::from(1u64)));

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
    assert_eq!(sd1.total_proper_fraction, Some(F::from(1u64)));

    let sd1pm2 = SymmetryOperation::builder()
        .generating_element(sd1_element)
        .power(-2)
        .build()
        .unwrap();
    assert_eq!(sd1pm2.total_proper_fraction, Some(F::from(1u64)));

    let s3_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(3))
        .proper_power(1)
        .axis(Vector3::new(2.0, 2.0, 1.0))
        .kind(SymmetryElementKind::ImproperMirrorPlane)
        .build()
        .unwrap();

    let s3 = SymmetryOperation::builder()
        .generating_element(s3_element.clone())
        .power(1)
        .build()
        .unwrap();
    assert_eq!(s3.total_proper_fraction, Some(F::new(1u64, 3u64)));

    let s3p3 = SymmetryOperation::builder()
        .generating_element(s3_element.clone())
        .power(-3)
        .build()
        .unwrap();
    assert_eq!(s3p3.total_proper_fraction, Some(F::from(1u64)));

    let s3p5 = SymmetryOperation::builder()
        .generating_element(s3_element.clone())
        .power(5)
        .build()
        .unwrap();
    assert!(!s3p5.is_proper());
    assert_eq!(s3p5.total_proper_fraction, Some(F::new(2u64, 3u64)));

    let s3pm5 = SymmetryOperation::builder()
        .generating_element(s3_element.clone())
        .power(-5)
        .build()
        .unwrap();
    assert!(!s3pm5.is_proper());
    assert_eq!(s3pm5.total_proper_fraction, Some(F::new(1u64, 3u64)));

    let s3p6 = SymmetryOperation::builder()
        .generating_element(s3_element)
        .power(6)
        .build()
        .unwrap();
    assert_eq!(s3p6.total_proper_fraction, Some(F::from(1u64)));

    let s3pp2_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(3))
        .proper_power(2)
        .axis(Vector3::new(2.0, 2.0, 1.0))
        .kind(SymmetryElementKind::ImproperMirrorPlane)
        .build()
        .unwrap();

    let s3pp2 = SymmetryOperation::builder()
        .generating_element(s3pp2_element.clone())
        .power(1)
        .build()
        .unwrap();
    assert_eq!(s3pp2.total_proper_fraction, Some(F::new(2u64, 3u64)));

    let s3pp2p3 = SymmetryOperation::builder()
        .generating_element(s3pp2_element.clone())
        .power(3)
        .build()
        .unwrap();
    assert_eq!(s3pp2p3.total_proper_fraction, Some(F::from(1u64)));

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
    assert_eq!(s3pp3.total_proper_fraction, Some(F::from(1u64)));

    let s3pp3p2 = SymmetryOperation::builder()
        .generating_element(s3pp3_element)
        .power(2)
        .build()
        .unwrap();
    assert_eq!(s3pp3p2.total_proper_fraction, Some(F::from(1u64)));

    let sd3_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(3))
        .proper_power(1)
        .axis(Vector3::new(1.0, 1.0, 1.0))
        .kind(SymmetryElementKind::ImproperInversionCentre)
        .build()
        .unwrap();

    let sd3 = SymmetryOperation::builder()
        .generating_element(sd3_element.clone())
        .power(1)
        .build()
        .unwrap();
    assert_eq!(sd3.total_proper_fraction, Some(F::new(1u64, 3u64)));

    let sd3p3 = SymmetryOperation::builder()
        .generating_element(sd3_element.clone())
        .power(3)
        .build()
        .unwrap();
    assert_eq!(sd3p3.total_proper_fraction, Some(F::from(1u64)));

    let sd3p6 = SymmetryOperation::builder()
        .generating_element(sd3_element)
        .power(6)
        .build()
        .unwrap();
    assert_eq!(sd3p6.total_proper_fraction, Some(F::from(1u64)));

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
    assert_eq!(sip2.total_proper_fraction, None);
}

#[test]
fn test_symmetry_operation_finite_improper_conversion() {
    // ============================
    // Improper symmetry operations
    // ============================
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
    approx::assert_relative_eq!(
        s1.total_proper_angle,
        0.0,
        max_relative = s1.generating_element.threshold,
        epsilon = s1.generating_element.threshold
    );
    assert_eq!(s1.total_proper_fraction, Some(F::from(1u64)));

    let s1c = s1.convert_to_improper_kind(&INV);
    approx::assert_relative_eq!(
        s1c.total_proper_angle,
        std::f64::consts::PI,
        max_relative = s1c.generating_element.threshold,
        epsilon = s1c.generating_element.threshold
    );
    assert_eq!(s1c.total_proper_fraction, Some(F::new(1u64, 2u64)));

    let s1pm2 = SymmetryOperation::builder()
        .generating_element(s1_element)
        .power(-2)
        .build()
        .unwrap();
    approx::assert_relative_eq!(
        s1pm2.total_proper_angle,
        0.0,
        max_relative = s1pm2.generating_element.threshold,
        epsilon = s1pm2.generating_element.threshold
    );
    assert_eq!(s1pm2.total_proper_fraction, Some(F::from(1u64)));

    let s1pm2c = s1pm2.convert_to_improper_kind(&SIG);
    approx::assert_relative_eq!(
        s1pm2c.total_proper_angle,
        0.0,
        max_relative = s1pm2c.generating_element.threshold,
        epsilon = s1pm2c.generating_element.threshold
    );
    assert_eq!(s1pm2c.total_proper_fraction, Some(F::from(1u64)));

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
    approx::assert_relative_eq!(
        sd2.total_proper_angle,
        std::f64::consts::PI,
        max_relative = sd2.generating_element.threshold,
        epsilon = sd2.generating_element.threshold
    );
    assert_eq!(sd2.total_proper_fraction, Some(F::new(1u64, 2u64)));

    let sd2c = sd2.convert_to_improper_kind(&SIG);
    approx::assert_relative_eq!(
        sd2c.total_proper_angle,
        0.0,
        max_relative = sd2c.generating_element.threshold,
        epsilon = sd2c.generating_element.threshold
    );
    assert_eq!(sd2c.total_proper_fraction, Some(F::from(1u64)));

    let sd2p2 = SymmetryOperation::builder()
        .generating_element(sd2_element)
        .power(2)
        .build()
        .unwrap();
    approx::assert_relative_eq!(
        sd2p2.total_proper_angle,
        0.0,
        max_relative = sd2p2.generating_element.threshold,
        epsilon = sd2p2.generating_element.threshold
    );
    assert_eq!(sd2p2.total_proper_fraction, Some(F::from(1u64)));

    let sd2p2c = sd2p2.convert_to_improper_kind(&SIG);
    approx::assert_relative_eq!(
        sd2p2c.total_proper_angle,
        0.0,
        max_relative = sd2p2c.generating_element.threshold,
        epsilon = sd2p2c.generating_element.threshold
    );
    assert_eq!(sd2p2c.total_proper_fraction, Some(F::from(1u64)));

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
    approx::assert_relative_eq!(
        sd2pp2.total_proper_angle,
        0.0,
        max_relative = sd2pp2.generating_element.threshold,
        epsilon = sd2pp2.generating_element.threshold
    );
    assert_eq!(sd2pp2.total_proper_fraction, Some(F::from(1u64)));

    let sd2pp2c = sd2pp2.convert_to_improper_kind(&SIG);
    approx::assert_relative_eq!(
        sd2pp2c.total_proper_angle,
        std::f64::consts::PI,
        max_relative = sd2pp2c.generating_element.threshold,
        epsilon = sd2pp2c.generating_element.threshold
    );
    assert_eq!(sd2pp2c.total_proper_fraction, Some(F::new(1u64, 2u64)));

    let sd2pp2p6 = SymmetryOperation::builder()
        .generating_element(sd2pp2_element)
        .power(6)
        .build()
        .unwrap();
    approx::assert_relative_eq!(
        sd2pp2p6.total_proper_angle,
        0.0,
        max_relative = sd2pp2p6.generating_element.threshold,
        epsilon = sd2pp2p6.generating_element.threshold
    );
    assert_eq!(sd2pp2p6.total_proper_fraction, Some(F::from(1u64)));

    let sd2pp2p6c = sd2pp2p6.convert_to_improper_kind(&SIG);
    approx::assert_relative_eq!(
        sd2pp2p6c.total_proper_angle,
        0.0,
        max_relative = sd2pp2p6c.generating_element.threshold,
        epsilon = sd2pp2p6c.generating_element.threshold
    );
    assert_eq!(sd2pp2p6c.total_proper_fraction, Some(F::from(1u64)));

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
    approx::assert_relative_eq!(
        s2.total_proper_angle,
        std::f64::consts::PI,
        max_relative = s2.generating_element.threshold,
        epsilon = s2.generating_element.threshold
    );
    assert_eq!(s2.total_proper_fraction, Some(F::new(1u64, 2u64)));

    let s2c = s2.convert_to_improper_kind(&INV);
    approx::assert_relative_eq!(
        s2c.total_proper_angle,
        0.0,
        max_relative = s2c.generating_element.threshold,
        epsilon = s2c.generating_element.threshold
    );
    assert_eq!(s2c.total_proper_fraction, Some(F::from(1u64)));

    let s2p2 = SymmetryOperation::builder()
        .generating_element(s2_element)
        .power(2)
        .build()
        .unwrap();
    approx::assert_relative_eq!(
        s2p2.total_proper_angle,
        0.0,
        max_relative = s2p2.generating_element.threshold,
        epsilon = s2p2.generating_element.threshold
    );
    assert_eq!(s2p2.total_proper_fraction, Some(F::from(1u64)));

    let s2p2c = s2p2.convert_to_improper_kind(&INV);
    approx::assert_relative_eq!(
        s2p2c.total_proper_angle,
        0.0,
        max_relative = s2p2c.generating_element.threshold,
        epsilon = s2p2c.generating_element.threshold
    );
    assert_eq!(s2p2c.total_proper_fraction, Some(F::from(1u64)));

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
    approx::assert_relative_eq!(
        sd1.total_proper_angle,
        0.0,
        max_relative = sd1.generating_element.threshold,
        epsilon = sd1.generating_element.threshold
    );
    assert_eq!(sd1.total_proper_fraction, Some(F::from(1u64)));

    let sd1c = sd1.convert_to_improper_kind(&SIG);
    approx::assert_relative_eq!(
        sd1c.total_proper_angle,
        std::f64::consts::PI,
        max_relative = sd1c.generating_element.threshold,
        epsilon = sd1c.generating_element.threshold
    );
    assert_eq!(sd1c.total_proper_fraction, Some(F::new(1u64, 2u64)));

    let sd1p2 = SymmetryOperation::builder()
        .generating_element(sd1_element)
        .power(-2)
        .build()
        .unwrap();
    approx::assert_relative_eq!(
        sd1p2.total_proper_angle,
        0.0,
        max_relative = sd1p2.generating_element.threshold,
        epsilon = sd1p2.generating_element.threshold
    );
    assert_eq!(sd1p2.total_proper_fraction, Some(F::from(1u64)));

    let sd1p2c = sd1p2.convert_to_improper_kind(&SIG);
    approx::assert_relative_eq!(
        sd1p2c.total_proper_angle,
        0.0,
        max_relative = sd1p2c.generating_element.threshold,
        epsilon = sd1p2.generating_element.threshold
    );
    assert_eq!(sd1p2c.total_proper_fraction, Some(F::from(1u64)));

    let s3_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(3))
        .proper_power(1)
        .axis(Vector3::new(2.0, 2.0, 1.0))
        .kind(SymmetryElementKind::ImproperMirrorPlane)
        .build()
        .unwrap();

    let s3 = SymmetryOperation::builder()
        .generating_element(s3_element.clone())
        .power(1)
        .build()
        .unwrap();
    approx::assert_relative_eq!(
        s3.total_proper_angle,
        2.0 * std::f64::consts::FRAC_PI_3,
        max_relative = s3.generating_element.threshold,
        epsilon = s3.generating_element.threshold
    );
    assert_eq!(s3.total_proper_fraction, Some(F::new(1u64, 3u64)));

    let s3c = s3.convert_to_improper_kind(&INV);
    approx::assert_relative_eq!(
        s3c.total_proper_angle,
        -std::f64::consts::FRAC_PI_3,
        max_relative = s3c.generating_element.threshold,
        epsilon = s3c.generating_element.threshold
    );
    assert_eq!(s3c.total_proper_fraction, Some(F::new(5u64, 6u64)));

    let s3p2 = SymmetryOperation::builder()
        .generating_element(s3_element.clone())
        .power(2)
        .build()
        .unwrap();
    assert!(s3p2.is_proper());
    approx::assert_relative_eq!(
        s3p2.total_proper_angle,
        -2.0 * std::f64::consts::FRAC_PI_3,
        max_relative = s3p2.generating_element.threshold,
        epsilon = s3p2.generating_element.threshold
    );
    assert_eq!(s3p2.total_proper_fraction, Some(F::new(2u64, 3u64)));

    let s3p2c = s3p2.convert_to_improper_kind(&INV);
    assert!(s3p2c.is_proper());
    approx::assert_relative_eq!(
        s3p2c.total_proper_angle,
        -2.0 * std::f64::consts::FRAC_PI_3,
        max_relative = s3p2c.generating_element.threshold,
        epsilon = s3p2c.generating_element.threshold
    );
    assert_eq!(s3p2c.total_proper_fraction, Some(F::new(2u64, 3u64)));

    let s3p3 = SymmetryOperation::builder()
        .generating_element(s3_element.clone())
        .power(3)
        .build()
        .unwrap();
    assert!(!s3p3.is_proper());
    approx::assert_relative_eq!(
        s3p3.total_proper_angle,
        0.0,
        max_relative = s3p3.generating_element.threshold,
        epsilon = s3p3.generating_element.threshold
    );
    assert_eq!(s3p3.total_proper_fraction, Some(F::from(1u64)));

    let s3p3c = s3p3.convert_to_improper_kind(&INV);
    assert!(!s3p3c.is_proper());
    approx::assert_relative_eq!(
        s3p3c.total_proper_angle,
        std::f64::consts::PI,
        max_relative = s3p2c.generating_element.threshold,
        epsilon = s3p2c.generating_element.threshold
    );
    assert_eq!(s3p3c.total_proper_fraction, Some(F::new(1u64, 2u64)));

    let s3p6 = SymmetryOperation::builder()
        .generating_element(s3_element)
        .power(6)
        .build()
        .unwrap();
    approx::assert_relative_eq!(
        s3p6.total_proper_angle,
        0.0,
        max_relative = s3p6.generating_element.threshold,
        epsilon = s3p6.generating_element.threshold
    );
    assert_eq!(s3p6.total_proper_fraction, Some(F::from(1u64)));

    let s3p6c = s3p6.convert_to_improper_kind(&INV);
    approx::assert_relative_eq!(
        s3p6c.total_proper_angle,
        0.0,
        max_relative = s3p6c.generating_element.threshold,
        epsilon = s3p6c.generating_element.threshold
    );
    assert_eq!(s3p6c.total_proper_fraction, Some(F::from(1u64)));

    let s3pp2_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(3))
        .proper_power(2)
        .axis(Vector3::new(2.0, 2.0, 1.0))
        .kind(SymmetryElementKind::ImproperMirrorPlane)
        .build()
        .unwrap();

    let s3pp2 = SymmetryOperation::builder()
        .generating_element(s3pp2_element.clone())
        .power(1)
        .build()
        .unwrap();
    approx::assert_relative_eq!(
        s3pp2.total_proper_angle,
        -2.0 * std::f64::consts::FRAC_PI_3,
        max_relative = s3pp2.generating_element.threshold,
        epsilon = s3pp2.generating_element.threshold
    );
    assert_eq!(s3pp2.total_proper_fraction, Some(F::new(2u64, 3u64)));

    let s3pp2c = s3pp2.convert_to_improper_kind(&INV);
    approx::assert_relative_eq!(
        s3pp2c.total_proper_angle,
        std::f64::consts::FRAC_PI_3,
        max_relative = s3pp2c.generating_element.threshold,
        epsilon = s3pp2c.generating_element.threshold
    );
    assert_eq!(s3pp2c.total_proper_fraction, Some(F::new(1u64, 6u64)));

    let s3pp2p3 = SymmetryOperation::builder()
        .generating_element(s3pp2_element.clone())
        .power(3)
        .build()
        .unwrap();
    approx::assert_relative_eq!(
        s3pp2p3.total_proper_angle,
        0.0,
        max_relative = s3pp2p3.generating_element.threshold,
        epsilon = s3pp2p3.generating_element.threshold
    );
    assert_eq!(s3pp2p3.total_proper_fraction, Some(F::from(1u64)));

    let s3pp2p3c = s3pp2p3.convert_to_improper_kind(&INV);
    approx::assert_relative_eq!(
        s3pp2p3c.total_proper_angle,
        std::f64::consts::PI,
        max_relative = s3pp2p3c.generating_element.threshold,
        epsilon = s3pp2p3c.generating_element.threshold
    );
    assert_eq!(s3pp2p3c.total_proper_fraction, Some(F::new(1u64, 2u64)));

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
    approx::assert_relative_eq!(
        sd3p3.total_proper_angle,
        0.0,
        max_relative = sd3p3.generating_element.threshold,
        epsilon = sd3p3.generating_element.threshold
    );
    assert_eq!(sd3p3.total_proper_fraction, Some(F::from(1u64)));

    let sd3p3c = sd3p3.convert_to_improper_kind(&SIG);
    approx::assert_relative_eq!(
        sd3p3c.total_proper_angle,
        std::f64::consts::PI,
        max_relative = sd3p3c.generating_element.threshold,
        epsilon = sd3p3c.generating_element.threshold
    );
    assert_eq!(sd3p3c.total_proper_fraction, Some(F::new(1u64, 2u64)));

    let s7_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(7))
        .proper_power(1)
        .axis(Vector3::new(2.0, 2.5, 1.0))
        .kind(SymmetryElementKind::ImproperMirrorPlane)
        .build()
        .unwrap();

    let s7 = SymmetryOperation::builder()
        .generating_element(s7_element.clone())
        .power(1)
        .build()
        .unwrap();
    approx::assert_relative_eq!(
        s7.total_proper_angle,
        2.0 / 7.0 * std::f64::consts::PI,
        max_relative = s7.generating_element.threshold,
        epsilon = s7.generating_element.threshold
    );
    assert_eq!(s7.total_proper_fraction, Some(F::new(1u64, 7u64)));

    let s7c = s7.convert_to_improper_kind(&INV);
    approx::assert_relative_eq!(
        s7c.total_proper_angle,
        -5.0 / 7.0 * std::f64::consts::PI,
        max_relative = s7c.generating_element.threshold,
        epsilon = s7c.generating_element.threshold
    );
    assert_eq!(s7c.total_proper_fraction, Some(F::new(9u64, 14u64)));

    let s7p2 = SymmetryOperation::builder()
        .generating_element(s7_element.clone())
        .power(2)
        .build()
        .unwrap();
    approx::assert_relative_eq!(
        s7p2.total_proper_angle,
        4.0 / 7.0 * std::f64::consts::PI,
        max_relative = s7p2.generating_element.threshold,
        epsilon = s7p2.generating_element.threshold
    );
    assert_eq!(s7p2.total_proper_fraction, Some(F::new(2u64, 7u64)));

    let s7p2c = s7p2.convert_to_improper_kind(&INV);
    approx::assert_relative_eq!(
        s7p2c.total_proper_angle,
        4.0 / 7.0 * std::f64::consts::PI,
        max_relative = s7p2c.generating_element.threshold,
        epsilon = s7p2c.generating_element.threshold
    );
    assert_eq!(s7p2c.total_proper_fraction, Some(F::new(2u64, 7u64)));

    let s7p5 = SymmetryOperation::builder()
        .generating_element(s7_element.clone())
        .power(5)
        .build()
        .unwrap();
    approx::assert_relative_eq!(
        s7p5.total_proper_angle,
        -4.0 / 7.0 * std::f64::consts::PI,
        max_relative = s7p5.generating_element.threshold,
        epsilon = s7p5.generating_element.threshold
    );
    assert_eq!(s7p5.total_proper_fraction, Some(F::new(5u64, 7u64)));

    let s7p5c = s7p5.convert_to_improper_kind(&INV);
    approx::assert_relative_eq!(
        s7p5c.total_proper_angle,
        3.0 / 7.0 * std::f64::consts::PI,
        max_relative = s7p5c.generating_element.threshold,
        epsilon = s7p5c.generating_element.threshold
    );
    assert_eq!(s7p5c.total_proper_fraction, Some(F::new(3u64, 14u64)));

    let s7pp2_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(7))
        .proper_power(2)
        .axis(Vector3::new(2.0, 2.5, 1.0))
        .kind(SymmetryElementKind::ImproperMirrorPlane)
        .build()
        .unwrap();

    let s7pp2 = SymmetryOperation::builder()
        .generating_element(s7pp2_element.clone())
        .power(1)
        .build()
        .unwrap();
    approx::assert_relative_eq!(
        s7pp2.total_proper_angle,
        4.0 / 7.0 * std::f64::consts::PI,
        max_relative = s7pp2.generating_element.threshold,
        epsilon = s7pp2.generating_element.threshold
    );
    assert_eq!(s7pp2.total_proper_fraction, Some(F::new(2u64, 7u64)));

    let s7pp2c = s7pp2.convert_to_improper_kind(&INV);
    approx::assert_relative_eq!(
        s7pp2c.total_proper_angle,
        -3.0 / 7.0 * std::f64::consts::PI,
        max_relative = s7pp2c.generating_element.threshold,
        epsilon = s7pp2c.generating_element.threshold
    );
    assert_eq!(s7pp2c.total_proper_fraction, Some(F::new(11u64, 14u64)));

    let si_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Inf)
        .axis(Vector3::new(1.0, 0.0, 1.0))
        .proper_angle(2.0 * std::f64::consts::FRAC_PI_4)
        .kind(SymmetryElementKind::ImproperMirrorPlane)
        .build()
        .unwrap();

    let si = SymmetryOperation::builder()
        .generating_element(si_element.clone())
        .power(1)
        .build()
        .unwrap();
    approx::assert_relative_eq!(
        si.total_proper_angle,
        std::f64::consts::FRAC_PI_2,
        max_relative = si.generating_element.threshold,
        epsilon = si.generating_element.threshold
    );
    assert_eq!(si.total_proper_fraction, None);

    let sic = si.convert_to_improper_kind(&INV);
    approx::assert_relative_eq!(
        sic.total_proper_angle,
        -std::f64::consts::FRAC_PI_2,
        max_relative = sic.generating_element.threshold,
        epsilon = sic.generating_element.threshold
    );
    assert_eq!(sic.total_proper_fraction, None);

    let sip2 = SymmetryOperation::builder()
        .generating_element(si_element.clone())
        .power(2)
        .build()
        .unwrap();
    approx::assert_relative_eq!(
        sip2.total_proper_angle,
        std::f64::consts::PI,
        max_relative = sip2.generating_element.threshold,
        epsilon = sip2.generating_element.threshold
    );
    assert_eq!(sip2.total_proper_fraction, None);

    let sip2c = sip2.convert_to_improper_kind(&INV);
    approx::assert_relative_eq!(
        sip2c.total_proper_angle,
        std::f64::consts::PI,
        max_relative = sip2c.generating_element.threshold,
        epsilon = sip2c.generating_element.threshold
    );
    assert_eq!(sip2c.total_proper_fraction, None);

    let sip4 = SymmetryOperation::builder()
        .generating_element(si_element.clone())
        .power(4)
        .build()
        .unwrap();
    approx::assert_relative_eq!(
        sip4.total_proper_angle,
        0.0,
        max_relative = sip4.generating_element.threshold,
        epsilon = sip4.generating_element.threshold
    );
    assert_eq!(sip4.total_proper_fraction, None);

    let sip4c = sip4.convert_to_improper_kind(&INV);
    approx::assert_relative_eq!(
        sip4c.total_proper_angle,
        0.0,
        max_relative = sip4c.generating_element.threshold,
        epsilon = sip4c.generating_element.threshold
    );
    assert_eq!(sip4c.total_proper_fraction, None);
}

#[test]
fn test_symmetry_operation_poles() {
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
    approx::assert_relative_eq!(c1.calc_pole(), Point3::origin());

    let c1b = SymmetryOperation::builder()
        .generating_element(c1_element)
        .power(-3)
        .build()
        .unwrap();
    approx::assert_relative_eq!(c1b.calc_pole(), Point3::origin());

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
    approx::assert_relative_eq!(c2.calc_pole(), Point3::new(1.0, 1.0, 0.0) / 2.0f64.sqrt());

    let c2pm1 = SymmetryOperation::builder()
        .generating_element(c2_element.clone())
        .power(-1)
        .build()
        .unwrap();
    approx::assert_relative_eq!(
        c2pm1.calc_pole(),
        Point3::new(1.0, 1.0, 0.0) / 2.0f64.sqrt()
    );

    let c2p2 = SymmetryOperation::builder()
        .generating_element(c2_element.clone())
        .power(2)
        .build()
        .unwrap();
    approx::assert_relative_eq!(c2p2.calc_pole(), Point3::origin());

    let c2b_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(2))
        .proper_power(1)
        .axis(Vector3::new(-1.0, 1.0, 0.0))
        .kind(SymmetryElementKind::Proper)
        .build()
        .unwrap();

    let c2b = SymmetryOperation::builder()
        .generating_element(c2b_element.clone())
        .power(1)
        .build()
        .unwrap();
    approx::assert_relative_eq!(c2b.calc_pole(), Point3::new(1.0, -1.0, 0.0) / 2.0f64.sqrt());

    let c2bpm1 = SymmetryOperation::builder()
        .generating_element(c2b_element.clone())
        .power(-1)
        .build()
        .unwrap();
    approx::assert_relative_eq!(
        c2bpm1.calc_pole(),
        Point3::new(1.0, -1.0, 0.0) / 2.0f64.sqrt()
    );

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
    approx::assert_relative_eq!(c3.calc_pole(), Point3::new(1.0, 1.0, 1.0) / 3.0f64.sqrt());

    let c3p2 = SymmetryOperation::builder()
        .generating_element(c3_element.clone())
        .power(2)
        .build()
        .unwrap();
    approx::assert_relative_eq!(
        c3p2.calc_pole(),
        -Point3::new(1.0, 1.0, 1.0) / 3.0f64.sqrt()
    );

    let c3pm1 = SymmetryOperation::builder()
        .generating_element(c3_element.clone())
        .power(-1)
        .build()
        .unwrap();
    approx::assert_relative_eq!(
        c3pm1.calc_pole(),
        -Point3::new(1.0, 1.0, 1.0) / 3.0f64.sqrt()
    );

    let c3pm2 = SymmetryOperation::builder()
        .generating_element(c3_element.clone())
        .power(-2)
        .build()
        .unwrap();
    approx::assert_relative_eq!(
        c3pm2.calc_pole(),
        Point3::new(1.0, 1.0, 1.0) / 3.0f64.sqrt()
    );

    let c3p3 = SymmetryOperation::builder()
        .generating_element(c3_element.clone())
        .power(3)
        .build()
        .unwrap();
    approx::assert_relative_eq!(c3p3.calc_pole(), Point3::origin());

    let c3p4 = SymmetryOperation::builder()
        .generating_element(c3_element.clone())
        .power(4)
        .build()
        .unwrap();
    approx::assert_relative_eq!(c3p4.calc_pole(), Point3::new(1.0, 1.0, 1.0) / 3.0f64.sqrt());

    let c3pm4 = SymmetryOperation::builder()
        .generating_element(c3_element)
        .power(-4)
        .build()
        .unwrap();
    approx::assert_relative_eq!(
        c3pm4.calc_pole(),
        -Point3::new(1.0, 1.0, 1.0) / 3.0f64.sqrt()
    );

    let c3pp2_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::new(3.0, 1e-14))
        .proper_power(2)
        .axis(Vector3::new(1.0, 1.0, 1.0))
        .kind(SymmetryElementKind::Proper)
        .build()
        .unwrap();

    let c3pp2 = SymmetryOperation::builder()
        .generating_element(c3pp2_element.clone())
        .power(1)
        .build()
        .unwrap();
    approx::assert_relative_eq!(
        c3pp2.calc_pole(),
        -Point3::new(1.0, 1.0, 1.0) / 3.0f64.sqrt()
    );

    let c3pp2p2 = SymmetryOperation::builder()
        .generating_element(c3pp2_element)
        .power(2)
        .build()
        .unwrap();
    approx::assert_relative_eq!(
        c3pp2p2.calc_pole(),
        Point3::new(1.0, 1.0, 1.0) / 3.0f64.sqrt()
    );

    let c4_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::new(4.0, 1e-14))
        .proper_power(1)
        .axis(Vector3::new(1.0, 1.0, -1.0))
        .kind(SymmetryElementKind::Proper)
        .build()
        .unwrap();

    let c4 = SymmetryOperation::builder()
        .generating_element(c4_element.clone())
        .power(1)
        .build()
        .unwrap();
    approx::assert_relative_eq!(c4.calc_pole(), Point3::new(1.0, 1.0, -1.0) / 3.0f64.sqrt());

    let c4p2 = SymmetryOperation::builder()
        .generating_element(c4_element.clone())
        .power(2)
        .build()
        .unwrap();
    approx::assert_relative_eq!(
        c4p2.calc_pole(),
        -Point3::new(1.0, 1.0, -1.0) / 3.0f64.sqrt()
    );

    let c4pm2 = SymmetryOperation::builder()
        .generating_element(c4_element.clone())
        .power(-2)
        .build()
        .unwrap();
    approx::assert_relative_eq!(
        c4pm2.calc_pole(),
        -Point3::new(1.0, 1.0, -1.0) / 3.0f64.sqrt()
    );

    let c4p3 = SymmetryOperation::builder()
        .generating_element(c4_element.clone())
        .power(3)
        .build()
        .unwrap();
    approx::assert_relative_eq!(
        c4p3.calc_pole(),
        -Point3::new(1.0, 1.0, -1.0) / 3.0f64.sqrt()
    );

    let c4p4 = SymmetryOperation::builder()
        .generating_element(c4_element.clone())
        .power(4)
        .build()
        .unwrap();
    approx::assert_relative_eq!(c4p4.calc_pole(), -Point3::origin());

    let c7_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(7))
        .proper_power(1)
        .axis(Vector3::new(1.0, 1.0, -2.0))
        .kind(SymmetryElementKind::Proper)
        .build()
        .unwrap();

    let c7 = SymmetryOperation::builder()
        .generating_element(c7_element.clone())
        .power(1)
        .build()
        .unwrap();
    approx::assert_relative_eq!(c7.calc_pole(), Point3::new(1.0, 1.0, -2.0) / 6.0f64.sqrt());

    let c7p2 = SymmetryOperation::builder()
        .generating_element(c7_element.clone())
        .power(2)
        .build()
        .unwrap();
    approx::assert_relative_eq!(
        c7p2.calc_pole(),
        Point3::new(1.0, 1.0, -2.0) / 6.0f64.sqrt()
    );

    let c7p3 = SymmetryOperation::builder()
        .generating_element(c7_element.clone())
        .power(3)
        .build()
        .unwrap();
    approx::assert_relative_eq!(
        c7p3.calc_pole(),
        Point3::new(1.0, 1.0, -2.0) / 6.0f64.sqrt()
    );

    let c7p4 = SymmetryOperation::builder()
        .generating_element(c7_element.clone())
        .power(4)
        .build()
        .unwrap();
    approx::assert_relative_eq!(
        c7p4.calc_pole(),
        -Point3::new(1.0, 1.0, -2.0) / 6.0f64.sqrt()
    );

    let c7p7 = SymmetryOperation::builder()
        .generating_element(c7_element)
        .power(7)
        .build()
        .unwrap();
    approx::assert_relative_eq!(c7p7.calc_pole(), -Point3::origin());

    let ci_element = SymmetryElement::builder()
        .threshold(1e-7)
        .proper_order(ElementOrder::Inf)
        .axis(Vector3::new(1.0, 0.0, -1.0))
        .proper_angle(2.0 * std::f64::consts::FRAC_PI_6)
        .kind(SymmetryElementKind::Proper)
        .build()
        .unwrap();

    let ci = SymmetryOperation::builder()
        .generating_element(ci_element.clone())
        .power(1)
        .build()
        .unwrap();
    approx::assert_relative_eq!(ci.calc_pole(), Point3::new(1.0, 0.0, -1.0) / 2.0f64.sqrt());

    let cip2 = SymmetryOperation::builder()
        .generating_element(ci_element.clone())
        .power(2)
        .build()
        .unwrap();
    approx::assert_relative_eq!(
        cip2.calc_pole(),
        Point3::new(1.0, 0.0, -1.0) / 2.0f64.sqrt()
    );

    let cip3 = SymmetryOperation::builder()
        .generating_element(ci_element.clone())
        .power(3)
        .build()
        .unwrap();
    approx::assert_relative_eq!(
        cip3.calc_pole(),
        -Point3::new(1.0, 0.0, -1.0) / 2.0f64.sqrt()
    );

    let cip4 = SymmetryOperation::builder()
        .generating_element(ci_element.clone())
        .power(4)
        .build()
        .unwrap();
    approx::assert_relative_eq!(
        cip4.calc_pole(),
        -Point3::new(1.0, 0.0, -1.0) / 2.0f64.sqrt()
    );

    let cip5 = SymmetryOperation::builder()
        .generating_element(ci_element.clone())
        .power(5)
        .build()
        .unwrap();
    approx::assert_relative_eq!(
        cip5.calc_pole(),
        -Point3::new(1.0, 0.0, -1.0) / 2.0f64.sqrt()
    );

    let cip6 = SymmetryOperation::builder()
        .generating_element(ci_element.clone())
        .power(6)
        .build()
        .unwrap();
    approx::assert_relative_eq!(cip6.calc_pole(), Point3::origin());

    // ============================
    // Improper symmetry operations
    // ============================
    let s1_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(1))
        .proper_power(1)
        .axis(Vector3::new(0.0, -2.0, 0.0))
        .kind(SymmetryElementKind::ImproperMirrorPlane)
        .build()
        .unwrap();

    let s1 = SymmetryOperation::builder()
        .generating_element(s1_element.clone())
        .power(1)
        .build()
        .unwrap();
    approx::assert_relative_eq!(s1.calc_pole(), Point3::new(0.0, 1.0, 0.0));

    let s1c = s1.convert_to_improper_kind(&INV);
    approx::assert_relative_eq!(s1.calc_pole(), s1c.calc_pole());

    let s1pm2 = SymmetryOperation::builder()
        .generating_element(s1_element)
        .power(-2)
        .build()
        .unwrap();
    approx::assert_relative_eq!(s1pm2.calc_pole(), Point3::origin());

    let s1pm2c = s1pm2.convert_to_improper_kind(&INV);
    approx::assert_relative_eq!(s1pm2.calc_pole(), s1pm2c.calc_pole());

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
    approx::assert_relative_eq!(sd2.calc_pole(), Point3::new(1.0, -1.0, 0.0) / 2.0f64.sqrt());

    let sd2c = sd2.convert_to_improper_kind(&SIG);
    approx::assert_relative_eq!(sd2.calc_pole(), sd2c.calc_pole());

    let sd2p2 = SymmetryOperation::builder()
        .generating_element(sd2_element)
        .power(2)
        .build()
        .unwrap();
    approx::assert_relative_eq!(sd2p2.calc_pole(), Point3::origin());

    let sd2p2c = sd2p2.convert_to_improper_kind(&SIG);
    approx::assert_relative_eq!(sd2p2.calc_pole(), sd2p2c.calc_pole());

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
    approx::assert_relative_eq!(sd2pp2.calc_pole(), Point3::origin());

    let sd2pp2c = sd2pp2.convert_to_improper_kind(&SIG);
    approx::assert_relative_eq!(sd2pp2.calc_pole(), sd2pp2c.calc_pole());

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
    approx::assert_relative_eq!(s2.calc_pole(), Point3::origin());

    let s2c = s2.convert_to_improper_kind(&INV);
    approx::assert_relative_eq!(s2.calc_pole(), s2c.calc_pole());

    let s2p2 = SymmetryOperation::builder()
        .generating_element(s2_element)
        .power(2)
        .build()
        .unwrap();
    approx::assert_relative_eq!(s2p2.calc_pole(), Point3::origin());

    let s2p2c = s2p2.convert_to_improper_kind(&INV);
    approx::assert_relative_eq!(s2p2.calc_pole(), s2p2c.calc_pole());

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
    approx::assert_relative_eq!(s2pp2.calc_pole(), Point3::new(2.0, 2.0, 1.0) / 3.0);

    let s2pp2c = s2pp2.convert_to_improper_kind(&INV);
    approx::assert_relative_eq!(s2pp2.calc_pole(), s2pp2c.calc_pole());

    let s2pp2p4 = SymmetryOperation::builder()
        .generating_element(s2pp2_element)
        .power(4)
        .build()
        .unwrap();
    approx::assert_relative_eq!(s2pp2p4.calc_pole(), Point3::origin());

    let s2pp2p4c = s2pp2p4.convert_to_improper_kind(&SIG);
    approx::assert_relative_eq!(s2pp2p4.calc_pole(), s2pp2p4c.calc_pole());

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
    approx::assert_relative_eq!(sd1.calc_pole(), Point3::origin());

    let sd1c = sd1.convert_to_improper_kind(&SIG);
    approx::assert_relative_eq!(sd1.calc_pole(), sd1c.calc_pole());

    let sd1pm2 = SymmetryOperation::builder()
        .generating_element(sd1_element)
        .power(-2)
        .build()
        .unwrap();
    approx::assert_relative_eq!(sd1pm2.calc_pole(), Point3::origin());

    let sd1pm2c = sd1pm2.convert_to_improper_kind(&SIG);
    approx::assert_relative_eq!(sd1pm2.calc_pole(), sd1pm2c.calc_pole());

    let s3_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(3))
        .proper_power(1)
        .axis(Vector3::new(2.0, 2.0, 1.0))
        .kind(SymmetryElementKind::ImproperMirrorPlane)
        .build()
        .unwrap();

    let s3 = SymmetryOperation::builder()
        .generating_element(s3_element.clone())
        .power(1)
        .build()
        .unwrap();
    approx::assert_relative_eq!(s3.calc_pole(), -Point3::new(2.0, 2.0, 1.0) / 3.0);

    let s3c = s3.convert_to_improper_kind(&INV);
    approx::assert_relative_eq!(s3.calc_pole(), s3c.calc_pole());

    let s3pm1 = SymmetryOperation::builder()
        .generating_element(s3_element.clone())
        .power(-1)
        .build()
        .unwrap();
    approx::assert_relative_eq!(s3pm1.calc_pole(), Point3::new(2.0, 2.0, 1.0) / 3.0);

    let s3pm1c = s3pm1.convert_to_improper_kind(&INV);
    approx::assert_relative_eq!(s3pm1.calc_pole(), s3pm1c.calc_pole());

    let s3p2 = SymmetryOperation::builder()
        .generating_element(s3_element.clone())
        .power(2)
        .build()
        .unwrap();
    approx::assert_relative_eq!(s3p2.calc_pole(), -Point3::new(2.0, 2.0, 1.0) / 3.0);

    let s3p2c = s3p2.convert_to_improper_kind(&INV);
    approx::assert_relative_eq!(s3p2.calc_pole(), s3p2c.calc_pole());

    let s3pm2 = SymmetryOperation::builder()
        .generating_element(s3_element.clone())
        .power(-2)
        .build()
        .unwrap();
    approx::assert_relative_eq!(s3pm2.calc_pole(), Point3::new(2.0, 2.0, 1.0) / 3.0);

    let s3pm2c = s3pm2.convert_to_improper_kind(&INV);
    approx::assert_relative_eq!(s3pm2.calc_pole(), s3pm2c.calc_pole());

    let s3p3 = SymmetryOperation::builder()
        .generating_element(s3_element.clone())
        .power(3)
        .build()
        .unwrap();
    approx::assert_relative_eq!(s3p3.calc_pole(), Point3::new(2.0, 2.0, 1.0) / 3.0);

    let s3p3c = s3p3.convert_to_improper_kind(&INV);
    approx::assert_relative_eq!(s3p3.calc_pole(), s3p3c.calc_pole());

    let s3pm3 = SymmetryOperation::builder()
        .generating_element(s3_element.clone())
        .power(-3)
        .build()
        .unwrap();
    approx::assert_relative_eq!(s3pm3.calc_pole(), Point3::new(2.0, 2.0, 1.0) / 3.0);

    let s3pm3c = s3pm3.convert_to_improper_kind(&INV);
    approx::assert_relative_eq!(s3pm3.calc_pole(), s3pm3c.calc_pole());

    let s3p5 = SymmetryOperation::builder()
        .generating_element(s3_element.clone())
        .power(5)
        .build()
        .unwrap();
    approx::assert_relative_eq!(s3p5.calc_pole(), Point3::new(2.0, 2.0, 1.0) / 3.0);

    let s3p5c = s3p5.convert_to_improper_kind(&INV);
    approx::assert_relative_eq!(s3p5.calc_pole(), s3p5c.calc_pole());

    let s3pm5 = SymmetryOperation::builder()
        .generating_element(s3_element.clone())
        .power(-5)
        .build()
        .unwrap();
    approx::assert_relative_eq!(s3pm5.calc_pole(), -Point3::new(2.0, 2.0, 1.0) / 3.0);

    let s3pm5c = s3pm5.convert_to_improper_kind(&INV);
    approx::assert_relative_eq!(s3pm5.calc_pole(), s3pm5c.calc_pole());

    let s3p6 = SymmetryOperation::builder()
        .generating_element(s3_element.clone())
        .power(6)
        .build()
        .unwrap();
    approx::assert_relative_eq!(s3p6.calc_pole(), Point3::origin());

    let s3pm6 = SymmetryOperation::builder()
        .generating_element(s3_element)
        .power(-6)
        .build()
        .unwrap();
    approx::assert_relative_eq!(s3pm6.calc_pole(), Point3::origin());

    let s3pp2_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(3))
        .proper_power(2)
        .axis(Vector3::new(2.0, 2.0, 1.0))
        .kind(SymmetryElementKind::ImproperMirrorPlane)
        .build()
        .unwrap();

    let s3pp2 = SymmetryOperation::builder()
        .generating_element(s3pp2_element.clone())
        .power(1)
        .build()
        .unwrap();
    approx::assert_relative_eq!(s3pp2.calc_pole(), -s3.calc_pole());

    let s3pp2p2 = SymmetryOperation::builder()
        .generating_element(s3pp2_element.clone())
        .power(2)
        .build()
        .unwrap();
    approx::assert_relative_eq!(s3pp2p2.calc_pole(), -s3p2.calc_pole());

    let s3pp2p3 = SymmetryOperation::builder()
        .generating_element(s3pp2_element.clone())
        .power(3)
        .build()
        .unwrap();
    approx::assert_relative_eq!(s3pp2p3.calc_pole(), s3p3.calc_pole());

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
    approx::assert_relative_eq!(s3pp3.calc_pole(), s3p3.calc_pole());

    let s3pp3p2 = SymmetryOperation::builder()
        .generating_element(s3pp3_element)
        .power(2)
        .build()
        .unwrap();
    approx::assert_relative_eq!(s3pp3p2.calc_pole(), Point3::origin());

    let sd3_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(3))
        .proper_power(1)
        .axis(Vector3::new(1.0, 1.0, 1.0))
        .kind(SymmetryElementKind::ImproperInversionCentre)
        .build()
        .unwrap();

    let sd3 = SymmetryOperation::builder()
        .generating_element(sd3_element.clone())
        .power(1)
        .build()
        .unwrap();
    approx::assert_relative_eq!(sd3.calc_pole(), Point3::new(1.0, 1.0, 1.0) / 3.0f64.sqrt());

    let sd3p2 = SymmetryOperation::builder()
        .generating_element(sd3_element.clone())
        .power(2)
        .build()
        .unwrap();
    approx::assert_relative_eq!(
        sd3p2.calc_pole(),
        -Point3::new(1.0, 1.0, 1.0) / 3.0f64.sqrt()
    );

    let sd3p3 = SymmetryOperation::builder()
        .generating_element(sd3_element)
        .power(3)
        .build()
        .unwrap();
    approx::assert_relative_eq!(sd3p3.calc_pole(), Point3::origin());

    let s7_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(7))
        .proper_power(1)
        .axis(Vector3::new(2.0, 2.0, -1.0))
        .kind(SymmetryElementKind::ImproperMirrorPlane)
        .build()
        .unwrap();

    let s7 = SymmetryOperation::builder()
        .generating_element(s7_element.clone())
        .power(1)
        .build()
        .unwrap();
    approx::assert_relative_eq!(s7.calc_pole(), -Point3::new(2.0, 2.0, -1.0) / 3.0);

    let s7c = s7.convert_to_improper_kind(&INV);
    approx::assert_relative_eq!(s7.calc_pole(), s7c.calc_pole());

    let s7pm1 = SymmetryOperation::builder()
        .generating_element(s7_element.clone())
        .power(-1)
        .build()
        .unwrap();
    approx::assert_relative_eq!(s7pm1.calc_pole(), Point3::new(2.0, 2.0, -1.0) / 3.0);

    let s7pm1c = s7pm1.convert_to_improper_kind(&INV);
    approx::assert_relative_eq!(s7pm1.calc_pole(), s7pm1c.calc_pole());

    let s7p4 = SymmetryOperation::builder()
        .generating_element(s7_element.clone())
        .power(4)
        .build()
        .unwrap();
    approx::assert_relative_eq!(s7p4.calc_pole(), -Point3::new(2.0, 2.0, -1.0) / 3.0);

    let s7p4c = s7p4.convert_to_improper_kind(&INV);
    approx::assert_relative_eq!(s7p4.calc_pole(), s7p4c.calc_pole());

    let s7pm4 = SymmetryOperation::builder()
        .generating_element(s7_element.clone())
        .power(-4)
        .build()
        .unwrap();
    approx::assert_relative_eq!(s7pm4.calc_pole(), Point3::new(2.0, 2.0, -1.0) / 3.0);

    let s7pm4c = s7pm4.convert_to_improper_kind(&INV);
    approx::assert_relative_eq!(s7pm4.calc_pole(), s7pm4c.calc_pole());

    let s7p5 = SymmetryOperation::builder()
        .generating_element(s7_element.clone())
        .power(5)
        .build()
        .unwrap();
    approx::assert_relative_eq!(s7p5.calc_pole(), Point3::new(2.0, 2.0, -1.0) / 3.0);

    let s7p5c = s7p5.convert_to_improper_kind(&INV);
    approx::assert_relative_eq!(s7p5.calc_pole(), s7p5c.calc_pole());

    let s7pm5 = SymmetryOperation::builder()
        .generating_element(s7_element)
        .power(-5)
        .build()
        .unwrap();
    approx::assert_relative_eq!(s7pm5.calc_pole(), -Point3::new(2.0, 2.0, -1.0) / 3.0);

    let s7pm5c = s7pm5.convert_to_improper_kind(&INV);
    approx::assert_relative_eq!(s7pm5.calc_pole(), s7pm5c.calc_pole());

    let si_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Inf)
        .axis(Vector3::new(1.0, 0.0, 1.0))
        .proper_angle(2.0 * std::f64::consts::PI / 5.0)
        .kind(SymmetryElementKind::ImproperMirrorPlane)
        .build()
        .unwrap();

    let si = SymmetryOperation::builder()
        .generating_element(si_element.clone())
        .power(1)
        .build()
        .unwrap();
    approx::assert_relative_eq!(si.calc_pole(), -Point3::new(1.0, 0.0, 1.0) / 2.0f64.sqrt());

    let sip2 = SymmetryOperation::builder()
        .generating_element(si_element.clone())
        .power(2)
        .build()
        .unwrap();
    approx::assert_relative_eq!(sip2.calc_pole(), Point3::new(1.0, 0.0, 1.0) / 2.0f64.sqrt());

    let sdi_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Inf)
        .axis(Vector3::new(1.0, 0.0, 1.0))
        .proper_angle(2.0 * std::f64::consts::PI / 5.0)
        .kind(SymmetryElementKind::ImproperInversionCentre)
        .build()
        .unwrap();

    let sdi = SymmetryOperation::builder()
        .generating_element(sdi_element.clone())
        .power(1)
        .build()
        .unwrap();
    approx::assert_relative_eq!(sdi.calc_pole(), Point3::new(1.0, 0.0, 1.0) / 2.0f64.sqrt());

    let sdip2 = SymmetryOperation::builder()
        .generating_element(sdi_element.clone())
        .power(2)
        .build()
        .unwrap();
    approx::assert_relative_eq!(
        sdip2.calc_pole(),
        Point3::new(1.0, 0.0, 1.0) / 2.0f64.sqrt()
    );
}

#[test]
fn test_symmetry_operation_comparisons() {
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

    let c1b = SymmetryOperation::builder()
        .generating_element(c1_element)
        .power(-3)
        .build()
        .unwrap();
    assert_eq!(c1, c1b);

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

    let c2pm1 = SymmetryOperation::builder()
        .generating_element(c2_element.clone())
        .power(-1)
        .build()
        .unwrap();
    assert_eq!(c2, c2pm1);

    let c2p2 = SymmetryOperation::builder()
        .generating_element(c2_element.clone())
        .power(2)
        .build()
        .unwrap();
    assert_eq!(c2p2, c1);

    let c2b_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(2))
        .proper_power(1)
        .axis(Vector3::new(-1.0, -1.0, 0.0))
        .kind(SymmetryElementKind::Proper)
        .build()
        .unwrap();

    let c2b = SymmetryOperation::builder()
        .generating_element(c2b_element)
        .power(1)
        .build()
        .unwrap();
    assert_eq!(c2, c2b);

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

    let c3p2 = SymmetryOperation::builder()
        .generating_element(c3_element.clone())
        .power(2)
        .build()
        .unwrap();

    let c3p3 = SymmetryOperation::builder()
        .generating_element(c3_element.clone())
        .power(3)
        .build()
        .unwrap();

    let c3pm1 = SymmetryOperation::builder()
        .generating_element(c3_element.clone())
        .power(-1)
        .build()
        .unwrap();

    let c3pm2 = SymmetryOperation::builder()
        .generating_element(c3_element.clone())
        .power(-2)
        .build()
        .unwrap();

    let c3pm3 = SymmetryOperation::builder()
        .generating_element(c3_element.clone())
        .power(-3)
        .build()
        .unwrap();

    assert_eq!(c3, c3pm2);
    assert_eq!(c3p2, c3pm1);
    assert_eq!(c3p3, c1);
    assert_eq!(c3pm3, c1);
    assert_ne!(c3, c3pm1);
    assert_ne!(c3, c3p2);

    let c3p4 = SymmetryOperation::builder()
        .generating_element(c3_element.clone())
        .power(4)
        .build()
        .unwrap();
    assert_eq!(c3p4, c3);

    let c3pp2_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::new(3.0, 1e-14))
        .proper_power(2)
        .axis(Vector3::new(1.0, 1.0, 1.0))
        .kind(SymmetryElementKind::Proper)
        .build()
        .unwrap();

    let c3pp2 = SymmetryOperation::builder()
        .generating_element(c3pp2_element.clone())
        .power(1)
        .build()
        .unwrap();
    assert_eq!(c3p2, c3pp2);

    let c3pp2p2 = SymmetryOperation::builder()
        .generating_element(c3pp2_element.clone())
        .power(2)
        .build()
        .unwrap();
    assert_eq!(c3, c3pp2p2);

    let c3pp2p3 = SymmetryOperation::builder()
        .generating_element(c3pp2_element.clone())
        .power(3)
        .build()
        .unwrap();
    assert_eq!(c1, c3pp2p3);

    let c4_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::new(4.0, 1e-14))
        .proper_power(1)
        .axis(Vector3::new(1.0, 1.0, -1.0))
        .kind(SymmetryElementKind::Proper)
        .build()
        .unwrap();

    let c4 = SymmetryOperation::builder()
        .generating_element(c4_element.clone())
        .power(1)
        .build()
        .unwrap();

    let c4pm1 = SymmetryOperation::builder()
        .generating_element(c4_element.clone())
        .power(-1)
        .build()
        .unwrap();

    let c4p2 = SymmetryOperation::builder()
        .generating_element(c4_element.clone())
        .power(2)
        .build()
        .unwrap();

    let c4pm2 = SymmetryOperation::builder()
        .generating_element(c4_element.clone())
        .power(-2)
        .build()
        .unwrap();

    let c4p3 = SymmetryOperation::builder()
        .generating_element(c4_element.clone())
        .power(3)
        .build()
        .unwrap();

    let c4pm3 = SymmetryOperation::builder()
        .generating_element(c4_element.clone())
        .power(-3)
        .build()
        .unwrap();

    let c4p4 = SymmetryOperation::builder()
        .generating_element(c4_element.clone())
        .power(4)
        .build()
        .unwrap();

    assert_eq!(c4p2, c4pm2);
    assert_eq!(c4p4, c1);
    assert_eq!(c4, c4pm3);
    assert_eq!(c4pm1, c4p3);

    let c4b_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::new(4.0, 1e-14))
        .proper_power(1)
        .axis(-Vector3::new(1.0, 1.0, -1.0))
        .kind(SymmetryElementKind::Proper)
        .build()
        .unwrap();

    let c4b = SymmetryOperation::builder()
        .generating_element(c4b_element.clone())
        .power(1)
        .build()
        .unwrap();

    let c4bpm1 = SymmetryOperation::builder()
        .generating_element(c4b_element.clone())
        .power(-1)
        .build()
        .unwrap();

    assert_eq!(c4b, c4pm1);
    assert_eq!(c4bpm1, c4);

    let c6_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(6))
        .proper_power(1)
        .axis(Vector3::new(1.0, 1.0, 1.0))
        .kind(SymmetryElementKind::Proper)
        .build()
        .unwrap();

    let c6 = SymmetryOperation::builder()
        .generating_element(c6_element.clone())
        .power(1)
        .build()
        .unwrap();

    let c6p2 = SymmetryOperation::builder()
        .generating_element(c6_element.clone())
        .power(2)
        .build()
        .unwrap();

    let c6pm11 = SymmetryOperation::builder()
        .generating_element(c6_element.clone())
        .power(-11)
        .build()
        .unwrap();

    let c6p6 = SymmetryOperation::builder()
        .generating_element(c6_element)
        .power(6)
        .build()
        .unwrap();

    assert_eq!(c6p2, c3);
    assert_eq!(c6p6, c1);
    assert_eq!(c6, c6pm11);

    let ci_element = SymmetryElement::builder()
        .threshold(1e-7)
        .proper_order(ElementOrder::Inf)
        .axis(-Vector3::new(1.0, 1.0, 1.0))
        .proper_angle(2.0 * std::f64::consts::FRAC_PI_6)
        .kind(SymmetryElementKind::Proper)
        .build()
        .unwrap();

    let ci = SymmetryOperation::builder()
        .generating_element(ci_element.clone())
        .power(1)
        .build()
        .unwrap();

    let cip2 = SymmetryOperation::builder()
        .generating_element(ci_element.clone())
        .power(2)
        .build()
        .unwrap();

    let cip3 = SymmetryOperation::builder()
        .generating_element(ci_element.clone())
        .power(3)
        .build()
        .unwrap();

    let cip4 = SymmetryOperation::builder()
        .generating_element(ci_element.clone())
        .power(4)
        .build()
        .unwrap();

    let cip5 = SymmetryOperation::builder()
        .generating_element(ci_element.clone())
        .power(5)
        .build()
        .unwrap();

    let cipm1 = SymmetryOperation::builder()
        .generating_element(ci_element.clone())
        .power(-1)
        .build()
        .unwrap();

    // They are not equal because they cannot be made to have the same hash.
    assert_ne!(cipm1, c6);
    assert_eq!(cipm1, cip5);

    let cipm2 = SymmetryOperation::builder()
        .generating_element(ci_element.clone())
        .power(-2)
        .build()
        .unwrap();
    assert_eq!(cipm2, cip4);

    let cipm3 = SymmetryOperation::builder()
        .generating_element(ci_element.clone())
        .power(-3)
        .build()
        .unwrap();
    assert_eq!(cipm3, cip3);

    let cipm4 = SymmetryOperation::builder()
        .generating_element(ci_element.clone())
        .power(-4)
        .build()
        .unwrap();
    assert_eq!(cipm4, cip2);

    let cipm5 = SymmetryOperation::builder()
        .generating_element(ci_element)
        .power(-5)
        .build()
        .unwrap();
    assert_eq!(cipm5, ci);

    // ============================
    // Improper symmetry operations
    // ============================
    let s1_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(1))
        .proper_power(1)
        .axis(Vector3::new(0.0, -2.0, 0.0))
        .kind(SymmetryElementKind::ImproperMirrorPlane)
        .build()
        .unwrap();

    let s1 = SymmetryOperation::builder()
        .generating_element(s1_element.clone())
        .power(1)
        .build()
        .unwrap();

    let s1c = s1.convert_to_improper_kind(&INV);
    assert_eq!(s1, s1c);

    let s1p2 = SymmetryOperation::builder()
        .generating_element(s1_element.clone())
        .power(2)
        .build()
        .unwrap();
    assert_eq!(s1p2, c1);

    let s1pm2 = SymmetryOperation::builder()
        .generating_element(s1_element.clone())
        .power(-2)
        .build()
        .unwrap();
    assert_eq!(s1pm2, c1);

    let s1pm2c = s1pm2.convert_to_improper_kind(&INV);
    assert_eq!(s1pm2c, c1);

    let s1p3 = SymmetryOperation::builder()
        .generating_element(s1_element)
        .power(3)
        .build()
        .unwrap();
    assert_eq!(s1p3, s1);

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

    let sd2c = sd2.convert_to_improper_kind(&SIG);
    assert_eq!(sd2, sd2c);

    let sd2p2 = SymmetryOperation::builder()
        .generating_element(sd2_element)
        .power(2)
        .build()
        .unwrap();
    assert_eq!(sd2p2, c1);

    let sd2p2c = sd2p2.convert_to_improper_kind(&SIG);
    assert_eq!(sd2p2c, c1);

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

    let sd2pp2c = sd2pp2.convert_to_improper_kind(&SIG);
    assert_eq!(sd2pp2, sd2pp2c);

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

    let s2c = s2.convert_to_improper_kind(&INV);
    assert_eq!(s2c, s2);

    let s2p2 = SymmetryOperation::builder()
        .generating_element(s2_element)
        .power(2)
        .build()
        .unwrap();
    assert_eq!(s2p2, c1);

    let s2p2c = s2p2.convert_to_improper_kind(&INV);
    assert_eq!(s2p2c, s2p2);

    let s2pp2_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(2))
        .proper_power(2)
        .axis(Vector3::new(0.0, 1.0, 0.0))
        .kind(SymmetryElementKind::ImproperMirrorPlane)
        .build()
        .unwrap();

    let s2pp2 = SymmetryOperation::builder()
        .generating_element(s2pp2_element.clone())
        .power(1)
        .build()
        .unwrap();
    assert_eq!(s2pp2, s1);

    let s2pp2c = s2pp2.convert_to_improper_kind(&INV);
    assert_eq!(s2pp2c, s2pp2);

    let s2pp2p4 = SymmetryOperation::builder()
        .generating_element(s2pp2_element)
        .power(4)
        .build()
        .unwrap();
    assert_eq!(s2pp2p4, c1);

    let s2pp2p4c = s2pp2p4.convert_to_improper_kind(&SIG);
    assert_eq!(s2pp2p4c, s2pp2p4);

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
    assert_eq!(sd1, sd2pp2);

    let sd1c = sd1.convert_to_improper_kind(&SIG);
    assert_eq!(sd1c, sd1);

    let sd1pm2 = SymmetryOperation::builder()
        .generating_element(sd1_element)
        .power(-2)
        .build()
        .unwrap();
    assert_eq!(sd1pm2, c1);

    let sd1pm2c = sd1pm2.convert_to_improper_kind(&SIG);
    assert_eq!(sd1pm2c, sd1pm2);

    let s3_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(3))
        .proper_power(1)
        .axis(Vector3::new(1.0, 1.0, 1.0))
        .kind(SymmetryElementKind::ImproperMirrorPlane)
        .build()
        .unwrap();

    let s3 = SymmetryOperation::builder()
        .generating_element(s3_element.clone())
        .power(1)
        .build()
        .unwrap();

    let s3c = s3.convert_to_improper_kind(&INV);
    assert_eq!(s3, s3c);

    let s3p2 = SymmetryOperation::builder()
        .generating_element(s3_element.clone())
        .power(2)
        .build()
        .unwrap();
    assert_eq!(s3p2, c3p2);

    let s3p2c = s3p2.convert_to_improper_kind(&INV);
    assert_eq!(s3p2, s3p2c);

    let s3pm1 = SymmetryOperation::builder()
        .generating_element(s3_element.clone())
        .power(-1)
        .build()
        .unwrap();

    let s3pm1c = s3pm1.convert_to_improper_kind(&INV);
    assert_eq!(s3pm1c, s3pm1);

    let s3p5 = SymmetryOperation::builder()
        .generating_element(s3_element.clone())
        .power(5)
        .build()
        .unwrap();
    assert_eq!(s3p5, s3pm1);

    let s3pm2 = SymmetryOperation::builder()
        .generating_element(s3_element.clone())
        .power(-2)
        .build()
        .unwrap();
    assert_eq!(s3pm2, c3pm2);

    let s3pm2c = s3pm2.convert_to_improper_kind(&INV);
    assert_eq!(s3pm2c, s3pm2);

    let s3p3 = SymmetryOperation::builder()
        .generating_element(s3_element.clone())
        .power(3)
        .build()
        .unwrap();

    let s3p3c = s3p3.convert_to_improper_kind(&INV);
    assert_eq!(s3p3, s3p3c);

    let s3pm3 = SymmetryOperation::builder()
        .generating_element(s3_element.clone())
        .power(-3)
        .build()
        .unwrap();
    assert_eq!(s3p3, s3pm3);

    let s3pm3c = s3pm3.convert_to_improper_kind(&INV);
    assert_eq!(s3pm3, s3pm3c);

    let s3pm5 = SymmetryOperation::builder()
        .generating_element(s3_element.clone())
        .power(-5)
        .build()
        .unwrap();
    assert_eq!(s3pm5, s3);

    let s3p6 = SymmetryOperation::builder()
        .generating_element(s3_element.clone())
        .power(6)
        .build()
        .unwrap();
    assert_eq!(s3p6, c3p3);

    let s3pp2_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(3))
        .proper_power(2)
        .axis(-Vector3::new(1.0, 1.0, 1.0))
        .kind(SymmetryElementKind::ImproperMirrorPlane)
        .build()
        .unwrap();

    let s3pp2 = SymmetryOperation::builder()
        .generating_element(s3pp2_element.clone())
        .power(1)
        .build()
        .unwrap();

    let s3pp2c = s3pp2.convert_to_improper_kind(&INV);
    assert_eq!(s3pp2, s3pp2c);

    let s3pp2p2 = SymmetryOperation::builder()
        .generating_element(s3pp2_element.clone())
        .power(2)
        .build()
        .unwrap();
    assert_eq!(s3pp2p2, c3pm1);

    let s3pp3_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(3))
        .proper_power(3)
        .axis(Vector3::new(1.0, 1.0, 1.0))
        .kind(SymmetryElementKind::ImproperMirrorPlane)
        .build()
        .unwrap();

    let s3pp3 = SymmetryOperation::builder()
        .generating_element(s3pp3_element.clone())
        .power(1)
        .build()
        .unwrap();
    assert_eq!(s3pp3, s3p3);

    let s3pp3p2 = SymmetryOperation::builder()
        .generating_element(s3pp3_element)
        .power(2)
        .build()
        .unwrap();
    assert_eq!(s3pp3p2, s3p6);

    let sd3_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(3))
        .proper_power(1)
        .axis(Vector3::new(1.0, 1.0, 1.0))
        .kind(SymmetryElementKind::ImproperInversionCentre)
        .build()
        .unwrap();

    let sd3 = SymmetryOperation::builder()
        .generating_element(sd3_element.clone())
        .power(1)
        .build()
        .unwrap();

    let s6pp5_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(6))
        .proper_power(5)
        .axis(Vector3::new(1.0, 1.0, 1.0))
        .kind(SymmetryElementKind::ImproperMirrorPlane)
        .build()
        .unwrap();

    let s6pp5 = SymmetryOperation::builder()
        .generating_element(s6pp5_element.clone())
        .power(1)
        .build()
        .unwrap();
    assert_eq!(sd3, s6pp5);

    let sd3p2 = SymmetryOperation::builder()
        .generating_element(sd3_element.clone())
        .power(2)
        .build()
        .unwrap();
    assert_eq!(sd3p2, c3p2);

    let sd3p3 = SymmetryOperation::builder()
        .generating_element(sd3_element)
        .power(3)
        .build()
        .unwrap();
    assert_eq!(sd3p3, s2);

    let s7_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(7))
        .proper_power(1)
        .axis(Vector3::new(2.0, 2.0, -1.0))
        .kind(SymmetryElementKind::ImproperMirrorPlane)
        .build()
        .unwrap();

    let s7 = SymmetryOperation::builder()
        .generating_element(s7_element.clone())
        .power(1)
        .build()
        .unwrap();

    let s7c = s7.convert_to_improper_kind(&INV);
    assert_eq!(s7, s7c);

    let s7pm1 = SymmetryOperation::builder()
        .generating_element(s7_element.clone())
        .power(-1)
        .build()
        .unwrap();

    let s7pm1c = s7pm1.convert_to_improper_kind(&INV);
    assert_eq!(s7pm1, s7pm1c);

    let s7pm3 = SymmetryOperation::builder()
        .generating_element(s7_element.clone())
        .power(-3)
        .build()
        .unwrap();

    let s7pm3c = s7pm3.convert_to_improper_kind(&INV);
    assert_eq!(s7pm3, s7pm3c);

    let s7p4 = SymmetryOperation::builder()
        .generating_element(s7_element.clone())
        .power(4)
        .build()
        .unwrap();

    let s7p4c = s7p4.convert_to_improper_kind(&INV);
    assert_eq!(s7p4, s7p4c);

    let s7pm4 = SymmetryOperation::builder()
        .generating_element(s7_element.clone())
        .power(-4)
        .build()
        .unwrap();

    let s7pm4c = s7pm4.convert_to_improper_kind(&INV);
    assert_eq!(s7pm4, s7pm4c);

    let s7p5 = SymmetryOperation::builder()
        .generating_element(s7_element.clone())
        .power(5)
        .build()
        .unwrap();

    let s7p5c = s7p5.convert_to_improper_kind(&INV);
    assert_eq!(s7p5, s7p5c);

    let s7pm5 = SymmetryOperation::builder()
        .generating_element(s7_element.clone())
        .power(-5)
        .build()
        .unwrap();

    let s7pm5c = s7pm5.convert_to_improper_kind(&INV);
    assert_eq!(s7pm5, s7pm5c);

    let s7p9 = SymmetryOperation::builder()
        .generating_element(s7_element.clone())
        .power(9)
        .build()
        .unwrap();
    assert_eq!(s7p9, s7pm5);

    let s7p10 = SymmetryOperation::builder()
        .generating_element(s7_element.clone())
        .power(10)
        .build()
        .unwrap();
    assert_eq!(s7p10, s7pm4);

    let s7p11 = SymmetryOperation::builder()
        .generating_element(s7_element.clone())
        .power(11)
        .build()
        .unwrap();
    assert_eq!(s7p11, s7pm3);


    let si_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Inf)
        .axis(Vector3::new(1.0, 0.0, 1.0))
        .proper_angle(2.0 * std::f64::consts::PI / 5.0)
        .kind(SymmetryElementKind::ImproperMirrorPlane)
        .build()
        .unwrap();

    let si = SymmetryOperation::builder()
        .generating_element(si_element.clone())
        .power(1)
        .build()
        .unwrap();

    let sic = si.convert_to_improper_kind(&INV);
    assert_eq!(si, sic);

    let sipm9 = SymmetryOperation::builder()
        .generating_element(si_element.clone())
        .power(-9)
        .build()
        .unwrap();

    assert_eq!(sipm9, si);

    let sip2 = SymmetryOperation::builder()
        .generating_element(si_element.clone())
        .power(2)
        .build()
        .unwrap();

    let sip2c = sip2.convert_to_improper_kind(&INV);
    assert_eq!(sip2, sip2c);

    let sipm8 = SymmetryOperation::builder()
        .generating_element(si_element.clone())
        .power(-8)
        .build()
        .unwrap();

    assert_eq!(sipm8, sip2);
}