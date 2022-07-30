use nalgebra::Vector3;

use crate::symmetry::symmetry_element::{
    ElementOrder, SymmetryElement, SymmetryElementKind, SymmetryOperation, SIG, INV
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
    approx::assert_relative_eq!(
        c3p6.total_proper_angle,
        0.0,
        max_relative = c3p6.generating_element.threshold,
        epsilon = c3p6.generating_element.threshold
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

    let c3p2_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::new(3.0, 1e-14))
        .proper_power(2)
        .axis(Vector3::new(1.0, 1.0, 1.0))
        .kind(SymmetryElementKind::Proper)
        .build()
        .unwrap();

    let c3pm6 = SymmetryOperation::builder()
        .generating_element(c3p2_element)
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

    let s1c = s1.convert_to_improper_kind(&INV);
    approx::assert_relative_eq!(
        s1c.total_proper_angle,
        std::f64::consts::PI,
        max_relative = s1c.generating_element.threshold,
        epsilon = s1c.generating_element.threshold
    );

    let s1p2 = SymmetryOperation::builder()
        .generating_element(s1_element)
        .power(-2)
        .build()
        .unwrap();
    approx::assert_relative_eq!(
        s1p2.total_proper_angle,
        0.0,
        max_relative = s1p2.generating_element.threshold,
        epsilon = s1p2.generating_element.threshold
    );

    let s1p2c = s1p2.convert_to_improper_kind(&SIG);
    approx::assert_relative_eq!(
        s1p2c.total_proper_angle,
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
    approx::assert_relative_eq!(
        sd2.total_proper_angle,
        std::f64::consts::PI,
        max_relative = sd2.generating_element.threshold,
        epsilon = sd2.generating_element.threshold
    );

    let sd2c = sd2.convert_to_improper_kind(&SIG);
    approx::assert_relative_eq!(
        sd2c.total_proper_angle,
        0.0,
        max_relative = sd2c.generating_element.threshold,
        epsilon = sd2c.generating_element.threshold
    );

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

    let sd2p2c = sd2p2.convert_to_improper_kind(&SIG);
    approx::assert_relative_eq!(
        sd2p2c.total_proper_angle,
        0.0,
        max_relative = sd2p2c.generating_element.threshold,
        epsilon = sd2p2c.generating_element.threshold
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
    approx::assert_relative_eq!(
        sd2pp2.total_proper_angle,
        0.0,
        max_relative = sd2pp2.generating_element.threshold,
        epsilon = sd2pp2.generating_element.threshold
    );

    let sd2pp2c = sd2pp2.convert_to_improper_kind(&SIG);
    approx::assert_relative_eq!(
        sd2pp2c.total_proper_angle,
        std::f64::consts::PI,
        max_relative = sd2pp2c.generating_element.threshold,
        epsilon = sd2pp2c.generating_element.threshold
    );

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

    let sd2pp2p6c = sd2pp2p6.convert_to_improper_kind(&SIG);
    approx::assert_relative_eq!(
        sd2pp2p6c.total_proper_angle,
        0.0,
        max_relative = sd2pp2p6c.generating_element.threshold,
        epsilon = sd2pp2p6c.generating_element.threshold
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
    approx::assert_relative_eq!(
        s2.total_proper_angle,
        std::f64::consts::PI,
        max_relative = s2.generating_element.threshold,
        epsilon = s2.generating_element.threshold
    );

    let s2c = s2.convert_to_improper_kind(&INV);
    approx::assert_relative_eq!(
        s2c.total_proper_angle,
        0.0,
        max_relative = s2c.generating_element.threshold,
        epsilon = s2c.generating_element.threshold
    );

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

    let s2p2c = s2p2.convert_to_improper_kind(&INV);
    approx::assert_relative_eq!(
        s2p2c.total_proper_angle,
        0.0,
        max_relative = s2p2c.generating_element.threshold,
        epsilon = s2p2c.generating_element.threshold
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
    approx::assert_relative_eq!(
        sd1.total_proper_angle,
        0.0,
        max_relative = sd1.generating_element.threshold,
        epsilon = sd1.generating_element.threshold
    );

    let sd1c = sd1.convert_to_improper_kind(&SIG);
    approx::assert_relative_eq!(
        sd1c.total_proper_angle,
        std::f64::consts::PI,
        max_relative = sd1c.generating_element.threshold,
        epsilon = sd1c.generating_element.threshold
    );

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

    let sd1p2c = sd1p2.convert_to_improper_kind(&SIG);
    approx::assert_relative_eq!(
        sd1p2c.total_proper_angle,
        0.0,
        max_relative = sd1p2c.generating_element.threshold,
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

    let s3c = s3.convert_to_improper_kind(&INV);
    approx::assert_relative_eq!(
        s3c.total_proper_angle,
        -std::f64::consts::FRAC_PI_3,
        max_relative = s3c.generating_element.threshold,
        epsilon = s3c.generating_element.threshold
    );

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

    let s3p2c = s3p2.convert_to_improper_kind(&INV);
    assert!(s3p2c.is_proper());
    approx::assert_relative_eq!(
        s3p2c.total_proper_angle,
        -2.0 * std::f64::consts::FRAC_PI_3,
        max_relative = s3p2c.generating_element.threshold,
        epsilon = s3p2c.generating_element.threshold
    );

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

    let s3p2c = s3p2.convert_to_improper_kind(&INV);
    assert!(s3p2c.is_proper());
    approx::assert_relative_eq!(
        s3p2c.total_proper_angle,
        -2.0 * std::f64::consts::FRAC_PI_3,
        max_relative = s3p2c.generating_element.threshold,
        epsilon = s3p2c.generating_element.threshold
    );

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

    let s3p3c = s3p3.convert_to_improper_kind(&INV);
    assert!(!s3p3c.is_proper());
    approx::assert_relative_eq!(
        s3p3c.total_proper_angle,
        std::f64::consts::PI,
        max_relative = s3p2c.generating_element.threshold,
        epsilon = s3p2c.generating_element.threshold
    );

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

    let s3p6c = s3p6.convert_to_improper_kind(&INV);
    approx::assert_relative_eq!(
        s3p6c.total_proper_angle,
        0.0,
        max_relative = s3p6c.generating_element.threshold,
        epsilon = s3p6c.generating_element.threshold
    );

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
        -2.0*std::f64::consts::FRAC_PI_3,
        max_relative = s3pp2.generating_element.threshold,
        epsilon = s3pp2.generating_element.threshold
    );

    let s3pp2c = s3pp2.convert_to_improper_kind(&INV);
    approx::assert_relative_eq!(
        s3pp2c.total_proper_angle,
        std::f64::consts::FRAC_PI_3,
        max_relative = s3pp2c.generating_element.threshold,
        epsilon = s3pp2c.generating_element.threshold
    );

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

    let s3pp2p3c = s3pp2p3.convert_to_improper_kind(&INV);
    approx::assert_relative_eq!(
        s3pp2p3c.total_proper_angle,
        std::f64::consts::PI,
        max_relative = s3pp2p3c.generating_element.threshold,
        epsilon = s3pp2p3c.generating_element.threshold
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
    approx::assert_relative_eq!(
        sd3p3.total_proper_angle,
        0.0,
        max_relative = sd3p3.generating_element.threshold,
        epsilon = sd3p3.generating_element.threshold
    );

    let sd3p3c = sd3p3.convert_to_improper_kind(&SIG);
    approx::assert_relative_eq!(
        sd3p3c.total_proper_angle,
        std::f64::consts::PI,
        max_relative = sd3p3c.generating_element.threshold,
        epsilon = sd3p3c.generating_element.threshold
    );

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

    let sic = si.convert_to_improper_kind(&INV);
    approx::assert_relative_eq!(
        sic.total_proper_angle,
        -std::f64::consts::FRAC_PI_2,
        max_relative = sic.generating_element.threshold,
        epsilon = sic.generating_element.threshold
    );

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

    let sip2c = sip2.convert_to_improper_kind(&INV);
    approx::assert_relative_eq!(
        sip2c.total_proper_angle,
        std::f64::consts::PI,
        max_relative = sip2c.generating_element.threshold,
        epsilon = sip2c.generating_element.threshold
    );

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

    let sip4c = sip4.convert_to_improper_kind(&INV);
    approx::assert_relative_eq!(
        sip4c.total_proper_angle,
        0.0,
        max_relative = sip4c.generating_element.threshold,
        epsilon = sip4c.generating_element.threshold
    );

    // let sib_element =
    //     si_element.convert_to_improper_kind(&SymmetryElementKind::ImproperInversionCentre, false);

    // let sibp2 = SymmetryOperation::builder()
    //     .generating_element(sib_element.clone())
    //     .power(2)
    //     .build()
    //     .unwrap();
    // assert!(sibp2.is_binary_rotation());
    // approx::assert_relative_eq!(
    //     sibp2.total_proper_angle,
    //     std::f64::consts::PI,
    //     max_relative = sibp2.generating_element.threshold,
    //     epsilon = sibp2.generating_element.threshold
    // );

    // let sibp4 = SymmetryOperation::builder()
    //     .generating_element(sib_element)
    //     .power(4)
    //     .build()
    //     .unwrap();
    // assert!(sibp4.is_identity());
    // approx::assert_relative_eq!(
    //     sibp4.total_proper_angle,
    //     0.0,
    //     max_relative = sibp4.generating_element.threshold,
    //     epsilon = sibp4.generating_element.threshold
    // );

}

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
