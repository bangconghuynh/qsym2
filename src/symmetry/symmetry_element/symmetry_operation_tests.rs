use nalgebra::{Point3, Vector3};
use num_traits::{Inv, Pow, Zero};
use std::collections::HashSet;

use crate::aux::geometry;
use crate::symmetry::symmetry_element::symmetry_operation::{
    FiniteOrder, SpecialSymmetryTransformation, SymmetryOperation,
};
use crate::symmetry::symmetry_element::{
    ElementOrder, RotationGroup, SymmetryElement, F, INV, ROT, SIG, SO3, SU2_0, SU2_1, TRINV,
    TRROT, TRSIG,
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
        .raw_axis(Vector3::new(0.0, 2.0, 0.0))
        .kind(ROT)
        .rotation_group(RotationGroup::SO3)
        .build()
        .unwrap();

    let c1 = SymmetryOperation::builder()
        .generating_element(c1_element.clone())
        .power(1)
        .build()
        .unwrap();
    assert!(c1.is_identity());
    assert_eq!(c1.order(), 1);
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
    assert_eq!(c1b.order(), 1);
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
        .raw_axis(Vector3::new(1.0, 1.0, 0.0))
        .kind(ROT)
        .rotation_group(RotationGroup::SO3)
        .build()
        .unwrap();

    let c2 = SymmetryOperation::builder()
        .generating_element(c2_element.clone())
        .power(1)
        .build()
        .unwrap();
    assert!(c2.is_spatial_binary_rotation());
    assert_eq!(c2.order(), 2);
    approx::assert_relative_eq!(c2.total_proper_angle, std::f64::consts::PI);

    let c2p2 = SymmetryOperation::builder()
        .generating_element(c2_element)
        .power(2)
        .build()
        .unwrap();
    assert!(c2p2.is_identity());
    assert_eq!(c2p2.order(), 1);
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
        .raw_axis(Vector3::new(1.0, 1.0, 0.0))
        .kind(ROT)
        .rotation_group(RotationGroup::SO3)
        .build()
        .unwrap();

    let c2p2b = SymmetryOperation::builder()
        .generating_element(c2p2_element)
        .power(1)
        .build()
        .unwrap();
    assert!(c2p2b.is_identity());
    assert_eq!(c2p2b.order(), 1);
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
        .raw_axis(Vector3::new(1.0, 1.0, 1.0))
        .kind(ROT)
        .rotation_group(RotationGroup::SO3)
        .build()
        .unwrap();

    let c3 = SymmetryOperation::builder()
        .generating_element(c3_element.clone())
        .power(1)
        .build()
        .unwrap();
    assert!(!c3.is_identity());
    assert_eq!(c3.order(), 3);
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
    assert_eq!(c3p3.order(), 1);
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
        .raw_axis(Vector3::new(1.0, 1.0, 1.0))
        .kind(ROT)
        .rotation_group(RotationGroup::SO3)
        .build()
        .unwrap();

    let c3pp2p3 = SymmetryOperation::builder()
        .generating_element(c3pp2_element)
        .power(3)
        .build()
        .unwrap();
    assert!(c3pp2p3.is_identity());
    assert_eq!(c3pp2p3.order(), 1);
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
        .raw_axis(Vector3::new(1.0, 1.0, 1.0))
        .kind(ROT)
        .rotation_group(RotationGroup::SO3)
        .build()
        .unwrap();

    let c4p2 = SymmetryOperation::builder()
        .generating_element(c4_element.clone())
        .power(2)
        .build()
        .unwrap();
    assert!(c4p2.is_spatial_binary_rotation());
    assert_eq!(c4p2.order(), 2);
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
    assert_eq!(c4p4.order(), 1);
    approx::assert_relative_eq!(
        c4p4.total_proper_angle,
        0.0,
        max_relative = c4p4.generating_element.threshold,
        epsilon = c4p4.generating_element.threshold
    );

    let ci_element = SymmetryElement::builder()
        .threshold(1e-7)
        .proper_order(ElementOrder::Inf)
        .raw_axis(Vector3::new(1.0, 0.0, -1.0))
        .proper_angle(2.0 * std::f64::consts::FRAC_PI_6)
        .kind(ROT)
        .rotation_group(RotationGroup::SO3)
        .build()
        .unwrap();

    let cip3 = SymmetryOperation::builder()
        .generating_element(ci_element.clone())
        .power(-3)
        .build()
        .unwrap();
    assert!(cip3.is_spatial_binary_rotation());
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
        .raw_axis(Vector3::new(0.0, 2.0, 0.0))
        .kind(SIG)
        .rotation_group(RotationGroup::SO3)
        .build()
        .unwrap();

    let s1 = SymmetryOperation::builder()
        .generating_element(s1_element.clone())
        .power(1)
        .build()
        .unwrap();
    assert!(s1.is_reflection());
    assert_eq!(s1.order(), 2);
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
    assert_eq!(s1p2.order(), 1);
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
        .raw_axis(Vector3::new(-1.0, 1.0, 0.0))
        .kind(INV)
        .rotation_group(RotationGroup::SO3)
        .build()
        .unwrap();

    let sd2 = SymmetryOperation::builder()
        .generating_element(sd2_element.clone())
        .power(1)
        .build()
        .unwrap();
    assert!(sd2.is_reflection());
    assert_eq!(sd2.order(), 2);
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
    assert_eq!(sd2p2.order(), 1);
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
        .raw_axis(Vector3::new(-1.0, 1.0, 0.0))
        .kind(INV)
        .rotation_group(RotationGroup::SO3)
        .build()
        .unwrap();

    let sd2pp2 = SymmetryOperation::builder()
        .generating_element(sd2pp2_element.clone())
        .power(1)
        .build()
        .unwrap();
    assert!(sd2pp2.is_inversion());
    assert_eq!(sd2pp2.order(), 2);
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
    assert_eq!(sd2pp2p6.order(), 1);
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
        .raw_axis(Vector3::new(2.0, 2.0, 1.0))
        .kind(SIG)
        .rotation_group(RotationGroup::SO3)
        .build()
        .unwrap();

    let s2 = SymmetryOperation::builder()
        .generating_element(s2_element.clone())
        .power(1)
        .build()
        .unwrap();
    assert!(s2.is_inversion());
    assert_eq!(s2.order(), 2);
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
    assert_eq!(s2p2.order(), 1);
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
        .raw_axis(Vector3::new(2.0, 2.0, 1.0))
        .kind(SIG)
        .rotation_group(RotationGroup::SO3)
        .build()
        .unwrap();

    let s2pp2 = SymmetryOperation::builder()
        .generating_element(s2pp2_element.clone())
        .power(1)
        .build()
        .unwrap();
    assert!(s2pp2.is_reflection());
    assert_eq!(s2pp2.order(), 2);
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
    assert_eq!(s2pp2p4.order(), 1);
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
        .raw_axis(Vector3::new(1.0, 1.0, 1.0))
        .kind(INV)
        .rotation_group(RotationGroup::SO3)
        .build()
        .unwrap();

    let sd1 = SymmetryOperation::builder()
        .generating_element(sd1_element.clone())
        .power(1)
        .build()
        .unwrap();
    assert!(sd1.is_inversion());
    assert_eq!(sd1.order(), 2);
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
    assert_eq!(sd1p2.order(), 1);
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
        .raw_axis(Vector3::new(2.0, 2.0, 1.0))
        .kind(SIG)
        .rotation_group(RotationGroup::SO3)
        .build()
        .unwrap();

    let s3p3 = SymmetryOperation::builder()
        .generating_element(s3_element.clone())
        .power(-3)
        .build()
        .unwrap();
    assert!(s3p3.is_reflection());
    assert_eq!(s3p3.order(), 2);
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
    assert_eq!(s3p6.order(), 1);
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
        .raw_axis(Vector3::new(2.0, 2.0, 1.0))
        .kind(SIG)
        .rotation_group(RotationGroup::SO3)
        .build()
        .unwrap();

    let s3pp2p3 = SymmetryOperation::builder()
        .generating_element(s3pp2_element.clone())
        .power(3)
        .build()
        .unwrap();
    assert!(s3pp2p3.is_reflection());
    assert_eq!(s3pp2p3.order(), 2);
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
    assert_eq!(s3pp2p6.order(), 1);
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
        .raw_axis(Vector3::new(2.0, 2.0, 1.0))
        .kind(SIG)
        .rotation_group(RotationGroup::SO3)
        .build()
        .unwrap();

    let s3pp3 = SymmetryOperation::builder()
        .generating_element(s3pp3_element.clone())
        .power(1)
        .build()
        .unwrap();
    assert!(s3pp3.is_reflection());
    assert_eq!(s3pp3.order(), 2);
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
    assert_eq!(s3pp3p2.order(), 1);
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
        .raw_axis(Vector3::new(1.0, 1.0, 1.0))
        .kind(INV)
        .rotation_group(RotationGroup::SO3)
        .build()
        .unwrap();

    let sd3p3 = SymmetryOperation::builder()
        .generating_element(sd3_element.clone())
        .power(3)
        .build()
        .unwrap();
    assert!(sd3p3.is_inversion());
    assert_eq!(sd3p3.order(), 2);
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
    assert_eq!(sd3p6.order(), 1);
    approx::assert_relative_eq!(
        sd3p6.total_proper_angle,
        0.0,
        max_relative = sd3p6.generating_element.threshold,
        epsilon = sd3p6.generating_element.threshold
    );

    let si_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Inf)
        .raw_axis(Vector3::new(1.0, 0.0, 1.0))
        .proper_angle(2.0 * std::f64::consts::FRAC_PI_4)
        .kind(SIG)
        .rotation_group(RotationGroup::SO3)
        .build()
        .unwrap();

    let sip2 = SymmetryOperation::builder()
        .generating_element(si_element.clone())
        .power(2)
        .build()
        .unwrap();
    assert!(sip2.is_spatial_binary_rotation());
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

    let sib_element = si_element.convert_to_improper_kind(&INV, false);

    let sibp2 = SymmetryOperation::builder()
        .generating_element(sib_element.clone())
        .power(2)
        .build()
        .unwrap();
    assert!(sibp2.is_spatial_binary_rotation());
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
        .raw_axis(Vector3::new(0.0, 2.0, 0.0))
        .kind(ROT)
        .rotation_group(RotationGroup::SO3)
        .build()
        .unwrap();

    let c1 = SymmetryOperation::builder()
        .generating_element(c1_element.clone())
        .power(1)
        .build()
        .unwrap();
    assert_eq!(c1.order(), 1);
    assert_eq!(c1.total_proper_fraction, Some(F::zero()));

    let c1b = SymmetryOperation::builder()
        .generating_element(c1_element)
        .power(-3)
        .build()
        .unwrap();
    assert_eq!(c1b.order(), 1);
    assert_eq!(c1b.total_proper_fraction, Some(F::zero()));

    let c2_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(2))
        .proper_power(1)
        .raw_axis(Vector3::new(1.0, 1.0, 0.0))
        .kind(ROT)
        .rotation_group(RotationGroup::SO3)
        .build()
        .unwrap();

    let c2 = SymmetryOperation::builder()
        .generating_element(c2_element.clone())
        .power(1)
        .build()
        .unwrap();
    assert_eq!(c2.order(), 2);
    assert_eq!(c2.total_proper_fraction, Some(F::new(1u32, 2u32)));

    let c2p2 = SymmetryOperation::builder()
        .generating_element(c2_element)
        .power(2)
        .build()
        .unwrap();
    assert_eq!(c2p2.order(), 1);
    assert_eq!(c2p2.total_proper_fraction, Some(F::zero()));

    let c2p2_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(2))
        .proper_power(2)
        .raw_axis(Vector3::new(1.0, 1.0, 0.0))
        .kind(ROT)
        .rotation_group(RotationGroup::SO3)
        .build()
        .unwrap();

    let c2p2b = SymmetryOperation::builder()
        .generating_element(c2p2_element)
        .power(1)
        .build()
        .unwrap();
    assert_eq!(c2p2b.order(), 1);
    assert_eq!(c2p2b.total_proper_fraction, Some(F::zero()));

    let c3_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::new(3.0, 1e-14))
        .proper_power(1)
        .raw_axis(Vector3::new(1.0, 1.0, 1.0))
        .kind(ROT)
        .rotation_group(RotationGroup::SO3)
        .build()
        .unwrap();

    let c3 = SymmetryOperation::builder()
        .generating_element(c3_element.clone())
        .power(1)
        .build()
        .unwrap();
    assert_eq!(c3.order(), 3);
    assert_eq!(c3.total_proper_fraction, Some(F::new(1u32, 3u32)));

    let c3p2 = SymmetryOperation::builder()
        .generating_element(c3_element.clone())
        .power(2)
        .build()
        .unwrap();
    assert_eq!(c3p2.order(), 3);
    assert_eq!(c3p2.total_proper_fraction, Some(F::new_neg(1u32, 3u32)));

    let c3pm2 = SymmetryOperation::builder()
        .generating_element(c3_element.clone())
        .power(-2)
        .build()
        .unwrap();
    assert_eq!(c3pm2.order(), 3);
    assert_eq!(c3pm2.total_proper_fraction, Some(F::new(1u32, 3u32)));

    let c3p3 = SymmetryOperation::builder()
        .generating_element(c3_element.clone())
        .power(3)
        .build()
        .unwrap();
    assert_eq!(c3p3.order(), 1);
    assert_eq!(c3p3.total_proper_fraction, Some(F::zero()));

    let c3p4 = SymmetryOperation::builder()
        .generating_element(c3_element.clone())
        .power(4)
        .build()
        .unwrap();
    assert_eq!(c3p4.order(), 3);
    assert_eq!(c3p4.total_proper_fraction, Some(F::new(1u32, 3u32)));

    let c3pm4 = SymmetryOperation::builder()
        .generating_element(c3_element)
        .power(-4)
        .build()
        .unwrap();
    assert_eq!(c3pm4.order(), 3);
    assert_eq!(c3pm4.total_proper_fraction, Some(F::new_neg(1u32, 3u32)));

    let c3pp2_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::new(3.0, 1e-14))
        .proper_power(2)
        .raw_axis(Vector3::new(1.0, 1.0, 1.0))
        .kind(ROT)
        .rotation_group(RotationGroup::SO3)
        .build()
        .unwrap();

    let c3pm6 = SymmetryOperation::builder()
        .generating_element(c3pp2_element)
        .power(-3)
        .build()
        .unwrap();
    assert_eq!(c3pm6.order(), 1);
    assert_eq!(c3pm6.total_proper_fraction, Some(F::zero()));

    let c4_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::new(4.0, 1e-14))
        .proper_power(1)
        .raw_axis(Vector3::new(1.0, 1.0, 1.0))
        .kind(ROT)
        .rotation_group(RotationGroup::SO3)
        .build()
        .unwrap();

    let c4 = SymmetryOperation::builder()
        .generating_element(c4_element.clone())
        .power(1)
        .build()
        .unwrap();
    assert_eq!(c4.order(), 4);
    assert_eq!(c4.total_proper_fraction, Some(F::new(1u32, 4u32)));

    let c4p2 = SymmetryOperation::builder()
        .generating_element(c4_element.clone())
        .power(2)
        .build()
        .unwrap();
    assert_eq!(c4p2.order(), 2);
    assert_eq!(c4p2.total_proper_fraction, Some(F::new(1u32, 2u32)));

    let c4pm2 = SymmetryOperation::builder()
        .generating_element(c4_element.clone())
        .power(-2)
        .build()
        .unwrap();
    assert_eq!(c4pm2.order(), 2);
    assert_eq!(c4pm2.total_proper_fraction, Some(F::new(1u32, 2u32)));

    let c4pm3 = SymmetryOperation::builder()
        .generating_element(c4_element.clone())
        .power(-3)
        .build()
        .unwrap();
    assert_eq!(c4pm3.order(), 4);
    assert_eq!(c4pm3.total_proper_fraction, Some(F::new(1u32, 4u32)));

    let c4p4 = SymmetryOperation::builder()
        .generating_element(c4_element)
        .power(4)
        .build()
        .unwrap();
    assert_eq!(c4p4.order(), 1);
    assert_eq!(c4p4.total_proper_fraction, Some(F::zero()));

    let c7_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(7))
        .proper_power(1)
        .raw_axis(Vector3::new(1.0, 1.0, 2.0))
        .kind(ROT)
        .rotation_group(RotationGroup::SO3)
        .build()
        .unwrap();

    let c7 = SymmetryOperation::builder()
        .generating_element(c7_element.clone())
        .power(1)
        .build()
        .unwrap();
    assert_eq!(c7.order(), 7);
    assert_eq!(c7.total_proper_fraction, Some(F::new(1u32, 7u32)));

    let c7p2 = SymmetryOperation::builder()
        .generating_element(c7_element.clone())
        .power(2)
        .build()
        .unwrap();
    assert_eq!(c7p2.order(), 7);
    assert_eq!(c7p2.total_proper_fraction, Some(F::new(2u32, 7u32)));

    let c7pm2 = SymmetryOperation::builder()
        .generating_element(c7_element.clone())
        .power(-2)
        .build()
        .unwrap();
    assert_eq!(c7pm2.order(), 7);
    assert_eq!(c7pm2.total_proper_fraction, Some(F::new_neg(2u32, 7u32)));

    let c7pm3 = SymmetryOperation::builder()
        .generating_element(c7_element.clone())
        .power(-3)
        .build()
        .unwrap();
    assert_eq!(c7pm3.order(), 7);
    assert_eq!(c7pm3.total_proper_fraction, Some(F::new_neg(3u32, 7u32)));

    let c7p4 = SymmetryOperation::builder()
        .generating_element(c7_element)
        .power(4)
        .build()
        .unwrap();
    assert_eq!(c7p4.order(), 7);
    assert_eq!(c7p4.total_proper_fraction, Some(F::new_neg(3u32, 7u32)));

    // ============================
    // Improper symmetry operations
    // ============================
    let s1_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(1))
        .proper_power(1)
        .raw_axis(Vector3::new(0.0, 2.0, 0.0))
        .kind(SIG)
        .rotation_group(RotationGroup::SO3)
        .build()
        .unwrap();

    let s1 = SymmetryOperation::builder()
        .generating_element(s1_element.clone())
        .power(1)
        .build()
        .unwrap();
    assert_eq!(s1.order(), 2);
    assert_eq!(s1.total_proper_fraction, Some(F::zero()));

    let s1pm2 = SymmetryOperation::builder()
        .generating_element(s1_element)
        .power(-2)
        .build()
        .unwrap();
    assert_eq!(s1pm2.order(), 1);
    assert_eq!(s1pm2.total_proper_fraction, Some(F::zero()));

    let sd2_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(2))
        .proper_power(1)
        .raw_axis(Vector3::new(-1.0, 1.0, 0.0))
        .kind(INV)
        .rotation_group(RotationGroup::SO3)
        .build()
        .unwrap();

    let sd2 = SymmetryOperation::builder()
        .generating_element(sd2_element.clone())
        .power(1)
        .build()
        .unwrap();
    assert_eq!(sd2.order(), 2);
    assert_eq!(sd2.total_proper_fraction, Some(F::new(1u32, 2u32)));

    let sd2p2 = SymmetryOperation::builder()
        .generating_element(sd2_element)
        .power(2)
        .build()
        .unwrap();
    assert_eq!(sd2p2.order(), 1);
    assert_eq!(sd2p2.total_proper_fraction, Some(F::zero()));

    let sd2pp2_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(2))
        .proper_power(2)
        .raw_axis(Vector3::new(-1.0, 1.0, 0.0))
        .kind(INV)
        .rotation_group(RotationGroup::SO3)
        .build()
        .unwrap();

    let sd2pp2 = SymmetryOperation::builder()
        .generating_element(sd2pp2_element.clone())
        .power(1)
        .build()
        .unwrap();
    assert_eq!(sd2pp2.order(), 2);
    assert_eq!(sd2pp2.total_proper_fraction, Some(F::zero()));

    let sd2pp2p6 = SymmetryOperation::builder()
        .generating_element(sd2pp2_element)
        .power(6)
        .build()
        .unwrap();
    assert_eq!(sd2pp2p6.order(), 1);
    assert_eq!(sd2pp2p6.total_proper_fraction, Some(F::zero()));

    let s2_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(2))
        .proper_power(1)
        .raw_axis(Vector3::new(2.0, 2.0, 1.0))
        .kind(SIG)
        .rotation_group(RotationGroup::SO3)
        .build()
        .unwrap();

    let s2 = SymmetryOperation::builder()
        .generating_element(s2_element.clone())
        .power(1)
        .build()
        .unwrap();
    assert_eq!(s2.order(), 2);
    assert_eq!(s2.total_proper_fraction, Some(F::new(1u32, 2u32)));

    let s2p2 = SymmetryOperation::builder()
        .generating_element(s2_element)
        .power(2)
        .build()
        .unwrap();
    assert_eq!(s2p2.order(), 1);
    assert_eq!(s2p2.total_proper_fraction, Some(F::zero()));

    let s2pp2_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(2))
        .proper_power(2)
        .raw_axis(Vector3::new(2.0, 2.0, 1.0))
        .kind(SIG)
        .rotation_group(RotationGroup::SO3)
        .build()
        .unwrap();

    let s2pp2 = SymmetryOperation::builder()
        .generating_element(s2pp2_element.clone())
        .power(1)
        .build()
        .unwrap();
    assert_eq!(s2pp2.order(), 2);
    assert_eq!(s2pp2.total_proper_fraction, Some(F::zero()));

    let s2pp2p4 = SymmetryOperation::builder()
        .generating_element(s2pp2_element)
        .power(4)
        .build()
        .unwrap();
    assert_eq!(s2pp2p4.order(), 1);
    assert_eq!(s2pp2p4.total_proper_fraction, Some(F::zero()));

    let sd1_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(1))
        .proper_power(1)
        .raw_axis(Vector3::new(1.0, 1.0, 1.0))
        .kind(INV)
        .rotation_group(RotationGroup::SO3)
        .build()
        .unwrap();

    let sd1 = SymmetryOperation::builder()
        .generating_element(sd1_element.clone())
        .power(1)
        .build()
        .unwrap();
    assert_eq!(sd1.order(), 2);
    assert_eq!(sd1.total_proper_fraction, Some(F::zero()));

    let sd1pm2 = SymmetryOperation::builder()
        .generating_element(sd1_element)
        .power(-2)
        .build()
        .unwrap();
    assert_eq!(sd1pm2.order(), 1);
    assert_eq!(sd1pm2.total_proper_fraction, Some(F::zero()));

    let s3_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(3))
        .proper_power(1)
        .raw_axis(Vector3::new(2.0, 2.0, 1.0))
        .kind(SIG)
        .rotation_group(RotationGroup::SO3)
        .build()
        .unwrap();

    let s3 = SymmetryOperation::builder()
        .generating_element(s3_element.clone())
        .power(1)
        .build()
        .unwrap();
    assert_eq!(s3.order(), 6);
    assert_eq!(s3.total_proper_fraction, Some(F::new(1u32, 3u32)));

    let s3p3 = SymmetryOperation::builder()
        .generating_element(s3_element.clone())
        .power(-3)
        .build()
        .unwrap();
    assert_eq!(s3p3.order(), 2);
    assert_eq!(s3p3.total_proper_fraction, Some(F::zero()));

    let s3p5 = SymmetryOperation::builder()
        .generating_element(s3_element.clone())
        .power(5)
        .build()
        .unwrap();
    assert!(!s3p5.is_proper());
    assert_eq!(s3p5.order(), 6);
    assert_eq!(s3p5.total_proper_fraction, Some(F::new_neg(1u32, 3u32)));

    let s3pm5 = SymmetryOperation::builder()
        .generating_element(s3_element.clone())
        .power(-5)
        .build()
        .unwrap();
    assert!(!s3pm5.is_proper());
    assert_eq!(s3pm5.order(), 6);
    assert_eq!(s3pm5.total_proper_fraction, Some(F::new(1u32, 3u32)));

    let s3p6 = SymmetryOperation::builder()
        .generating_element(s3_element)
        .power(6)
        .build()
        .unwrap();
    assert_eq!(s3p6.order(), 1);
    assert_eq!(s3p6.total_proper_fraction, Some(F::zero()));

    let s3pp2_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(3))
        .proper_power(2)
        .raw_axis(Vector3::new(2.0, 2.0, 1.0))
        .kind(SIG)
        .rotation_group(RotationGroup::SO3)
        .build()
        .unwrap();

    let s3pp2 = SymmetryOperation::builder()
        .generating_element(s3pp2_element.clone())
        .power(1)
        .build()
        .unwrap();
    assert_eq!(s3pp2.order(), 6);
    assert_eq!(s3pp2.total_proper_fraction, Some(F::new_neg(1u32, 3u32)));

    let s3pp2p3 = SymmetryOperation::builder()
        .generating_element(s3pp2_element)
        .power(3)
        .build()
        .unwrap();
    assert_eq!(s3pp2p3.order(), 2);
    assert_eq!(s3pp2p3.total_proper_fraction, Some(F::zero()));

    let s3pp3_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(3))
        .proper_power(3)
        .raw_axis(Vector3::new(2.0, 2.0, 1.0))
        .kind(SIG)
        .rotation_group(RotationGroup::SO3)
        .build()
        .unwrap();

    let s3pp3 = SymmetryOperation::builder()
        .generating_element(s3pp3_element.clone())
        .power(1)
        .build()
        .unwrap();
    assert_eq!(s3pp3.order(), 2);
    assert_eq!(s3pp3.total_proper_fraction, Some(F::zero()));

    let s3pp3p2 = SymmetryOperation::builder()
        .generating_element(s3pp3_element)
        .power(2)
        .build()
        .unwrap();
    assert_eq!(s3pp3p2.order(), 1);
    assert_eq!(s3pp3p2.total_proper_fraction, Some(F::zero()));

    let sd3_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(3))
        .proper_power(1)
        .raw_axis(Vector3::new(1.0, 1.0, 1.0))
        .kind(INV)
        .rotation_group(RotationGroup::SO3)
        .build()
        .unwrap();

    let sd3 = SymmetryOperation::builder()
        .generating_element(sd3_element.clone())
        .power(1)
        .build()
        .unwrap();
    assert_eq!(sd3.order(), 6);
    assert_eq!(sd3.total_proper_fraction, Some(F::new(1u32, 3u32)));

    let sd3p3 = SymmetryOperation::builder()
        .generating_element(sd3_element.clone())
        .power(3)
        .build()
        .unwrap();
    assert_eq!(sd3p3.order(), 2);
    assert_eq!(sd3p3.total_proper_fraction, Some(F::zero()));

    let sd3p6 = SymmetryOperation::builder()
        .generating_element(sd3_element)
        .power(6)
        .build()
        .unwrap();
    assert_eq!(sd3p6.order(), 1);
    assert_eq!(sd3p6.total_proper_fraction, Some(F::zero()));

    let si_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Inf)
        .raw_axis(Vector3::new(1.0, 0.0, 1.0))
        .proper_angle(2.0 * std::f64::consts::FRAC_PI_4)
        .kind(SIG)
        .rotation_group(RotationGroup::SO3)
        .build()
        .unwrap();

    let sip2 = SymmetryOperation::builder()
        .generating_element(si_element)
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
        .raw_axis(Vector3::new(0.0, 2.0, 0.0))
        .kind(SIG)
        .rotation_group(RotationGroup::SO3)
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
    assert_eq!(s1.total_proper_fraction, Some(F::zero()));

    let s1c = s1.convert_to_improper_kind(&INV);
    approx::assert_relative_eq!(
        s1c.total_proper_angle,
        std::f64::consts::PI,
        max_relative = s1c.generating_element.threshold,
        epsilon = s1c.generating_element.threshold
    );
    assert_eq!(s1c.total_proper_fraction, Some(F::new(1u32, 2u32)));

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
    assert_eq!(s1pm2.total_proper_fraction, Some(F::zero()));

    let s1pm2c = s1pm2.convert_to_improper_kind(&SIG);
    approx::assert_relative_eq!(
        s1pm2c.total_proper_angle,
        0.0,
        max_relative = s1pm2c.generating_element.threshold,
        epsilon = s1pm2c.generating_element.threshold
    );
    assert_eq!(s1pm2c.total_proper_fraction, Some(F::zero()));

    let sd2_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(2))
        .proper_power(1)
        .raw_axis(Vector3::new(-1.0, 1.0, 0.0))
        .kind(INV)
        .rotation_group(RotationGroup::SO3)
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
    assert_eq!(sd2.total_proper_fraction, Some(F::new(1u32, 2u32)));

    let sd2c = sd2.convert_to_improper_kind(&SIG);
    approx::assert_relative_eq!(
        sd2c.total_proper_angle,
        0.0,
        max_relative = sd2c.generating_element.threshold,
        epsilon = sd2c.generating_element.threshold
    );
    assert_eq!(sd2c.total_proper_fraction, Some(F::zero()));

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
    assert_eq!(sd2p2.total_proper_fraction, Some(F::zero()));

    let sd2p2c = sd2p2.convert_to_improper_kind(&SIG);
    approx::assert_relative_eq!(
        sd2p2c.total_proper_angle,
        0.0,
        max_relative = sd2p2c.generating_element.threshold,
        epsilon = sd2p2c.generating_element.threshold
    );
    assert_eq!(sd2p2c.total_proper_fraction, Some(F::zero()));

    let sd2pp2_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(2))
        .proper_power(2)
        .raw_axis(Vector3::new(-1.0, 1.0, 0.0))
        .kind(INV)
        .rotation_group(RotationGroup::SO3)
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
    assert_eq!(sd2pp2.total_proper_fraction, Some(F::zero()));

    let sd2pp2c = sd2pp2.convert_to_improper_kind(&SIG);
    approx::assert_relative_eq!(
        sd2pp2c.total_proper_angle,
        std::f64::consts::PI,
        max_relative = sd2pp2c.generating_element.threshold,
        epsilon = sd2pp2c.generating_element.threshold
    );
    assert_eq!(sd2pp2c.total_proper_fraction, Some(F::new(1u32, 2u32)));

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
    assert_eq!(sd2pp2p6.total_proper_fraction, Some(F::zero()));

    let sd2pp2p6c = sd2pp2p6.convert_to_improper_kind(&SIG);
    approx::assert_relative_eq!(
        sd2pp2p6c.total_proper_angle,
        0.0,
        max_relative = sd2pp2p6c.generating_element.threshold,
        epsilon = sd2pp2p6c.generating_element.threshold
    );
    assert_eq!(sd2pp2p6c.total_proper_fraction, Some(F::zero()));

    let s2_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(2))
        .proper_power(1)
        .raw_axis(Vector3::new(2.0, 2.0, 1.0))
        .kind(SIG)
        .rotation_group(RotationGroup::SO3)
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
    assert_eq!(s2.total_proper_fraction, Some(F::new(1u32, 2u32)));

    let s2c = s2.convert_to_improper_kind(&INV);
    approx::assert_relative_eq!(
        s2c.total_proper_angle,
        0.0,
        max_relative = s2c.generating_element.threshold,
        epsilon = s2c.generating_element.threshold
    );
    assert_eq!(s2c.total_proper_fraction, Some(F::zero()));

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
    assert_eq!(s2p2.total_proper_fraction, Some(F::zero()));

    let s2p2c = s2p2.convert_to_improper_kind(&INV);
    approx::assert_relative_eq!(
        s2p2c.total_proper_angle,
        0.0,
        max_relative = s2p2c.generating_element.threshold,
        epsilon = s2p2c.generating_element.threshold
    );
    assert_eq!(s2p2c.total_proper_fraction, Some(F::zero()));

    let sd1_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(1))
        .proper_power(1)
        .raw_axis(Vector3::new(1.0, 1.0, 1.0))
        .kind(INV)
        .rotation_group(RotationGroup::SO3)
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
    assert_eq!(sd1.total_proper_fraction, Some(F::zero()));

    let sd1c = sd1.convert_to_improper_kind(&SIG);
    approx::assert_relative_eq!(
        sd1c.total_proper_angle,
        std::f64::consts::PI,
        max_relative = sd1c.generating_element.threshold,
        epsilon = sd1c.generating_element.threshold
    );
    assert_eq!(sd1c.total_proper_fraction, Some(F::new(1u32, 2u32)));

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
    assert_eq!(sd1p2.total_proper_fraction, Some(F::zero()));

    let sd1p2c = sd1p2.convert_to_improper_kind(&SIG);
    approx::assert_relative_eq!(
        sd1p2c.total_proper_angle,
        0.0,
        max_relative = sd1p2c.generating_element.threshold,
        epsilon = sd1p2.generating_element.threshold
    );
    assert_eq!(sd1p2c.total_proper_fraction, Some(F::zero()));

    let s3_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(3))
        .proper_power(1)
        .raw_axis(Vector3::new(2.0, 2.0, 1.0))
        .kind(SIG)
        .rotation_group(RotationGroup::SO3)
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
    assert_eq!(s3.total_proper_fraction, Some(F::new(1u32, 3u32)));

    let s3c = s3.convert_to_improper_kind(&INV);
    approx::assert_relative_eq!(
        s3c.total_proper_angle,
        -std::f64::consts::FRAC_PI_3,
        max_relative = s3c.generating_element.threshold,
        epsilon = s3c.generating_element.threshold
    );
    assert_eq!(s3c.total_proper_fraction, Some(F::new_neg(1u32, 6u32)));

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
    assert_eq!(s3p2.total_proper_fraction, Some(F::new_neg(1u32, 3u32)));

    let s3p2c = s3p2.convert_to_improper_kind(&INV);
    assert!(s3p2c.is_proper());
    approx::assert_relative_eq!(
        s3p2c.total_proper_angle,
        -2.0 * std::f64::consts::FRAC_PI_3,
        max_relative = s3p2c.generating_element.threshold,
        epsilon = s3p2c.generating_element.threshold
    );
    assert_eq!(s3p2c.total_proper_fraction, Some(F::new_neg(1u32, 3u32)));

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
    assert_eq!(s3p3.total_proper_fraction, Some(F::zero()));

    let s3p3c = s3p3.convert_to_improper_kind(&INV);
    assert!(!s3p3c.is_proper());
    approx::assert_relative_eq!(
        s3p3c.total_proper_angle,
        std::f64::consts::PI,
        max_relative = s3p2c.generating_element.threshold,
        epsilon = s3p2c.generating_element.threshold
    );
    assert_eq!(s3p3c.total_proper_fraction, Some(F::new(1u32, 2u32)));

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
    assert_eq!(s3p6.total_proper_fraction, Some(F::zero()));

    let s3p6c = s3p6.convert_to_improper_kind(&INV);
    approx::assert_relative_eq!(
        s3p6c.total_proper_angle,
        0.0,
        max_relative = s3p6c.generating_element.threshold,
        epsilon = s3p6c.generating_element.threshold
    );
    assert_eq!(s3p6c.total_proper_fraction, Some(F::zero()));

    let s3pp2_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(3))
        .proper_power(2)
        .raw_axis(Vector3::new(2.0, 2.0, 1.0))
        .kind(SIG)
        .rotation_group(RotationGroup::SO3)
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
    assert_eq!(s3pp2.total_proper_fraction, Some(F::new_neg(1u32, 3u32)));

    let s3pp2c = s3pp2.convert_to_improper_kind(&INV);
    approx::assert_relative_eq!(
        s3pp2c.total_proper_angle,
        std::f64::consts::FRAC_PI_3,
        max_relative = s3pp2c.generating_element.threshold,
        epsilon = s3pp2c.generating_element.threshold
    );
    assert_eq!(s3pp2c.total_proper_fraction, Some(F::new(1u32, 6u32)));

    let s3pp2p3 = SymmetryOperation::builder()
        .generating_element(s3pp2_element)
        .power(3)
        .build()
        .unwrap();
    approx::assert_relative_eq!(
        s3pp2p3.total_proper_angle,
        0.0,
        max_relative = s3pp2p3.generating_element.threshold,
        epsilon = s3pp2p3.generating_element.threshold
    );
    assert_eq!(s3pp2p3.total_proper_fraction, Some(F::zero()));

    let s3pp2p3c = s3pp2p3.convert_to_improper_kind(&INV);
    approx::assert_relative_eq!(
        s3pp2p3c.total_proper_angle,
        std::f64::consts::PI,
        max_relative = s3pp2p3c.generating_element.threshold,
        epsilon = s3pp2p3c.generating_element.threshold
    );
    assert_eq!(s3pp2p3c.total_proper_fraction, Some(F::new(1u32, 2u32)));

    let sd3_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(3))
        .proper_power(1)
        .raw_axis(Vector3::new(1.0, 1.0, 1.0))
        .kind(INV)
        .rotation_group(RotationGroup::SO3)
        .build()
        .unwrap();

    let sd3p3 = SymmetryOperation::builder()
        .generating_element(sd3_element)
        .power(3)
        .build()
        .unwrap();
    approx::assert_relative_eq!(
        sd3p3.total_proper_angle,
        0.0,
        max_relative = sd3p3.generating_element.threshold,
        epsilon = sd3p3.generating_element.threshold
    );
    assert_eq!(sd3p3.total_proper_fraction, Some(F::zero()));

    let sd3p3c = sd3p3.convert_to_improper_kind(&SIG);
    approx::assert_relative_eq!(
        sd3p3c.total_proper_angle,
        std::f64::consts::PI,
        max_relative = sd3p3c.generating_element.threshold,
        epsilon = sd3p3c.generating_element.threshold
    );
    assert_eq!(sd3p3c.total_proper_fraction, Some(F::new(1u32, 2u32)));

    let s7_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(7))
        .proper_power(1)
        .raw_axis(Vector3::new(2.0, 2.5, 1.0))
        .kind(SIG)
        .rotation_group(RotationGroup::SO3)
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
    assert_eq!(s7.total_proper_fraction, Some(F::new(1u32, 7u32)));

    let s7c = s7.convert_to_improper_kind(&INV);
    approx::assert_relative_eq!(
        s7c.total_proper_angle,
        -5.0 / 7.0 * std::f64::consts::PI,
        max_relative = s7c.generating_element.threshold,
        epsilon = s7c.generating_element.threshold
    );
    assert_eq!(s7c.total_proper_fraction, Some(F::new_neg(5u32, 14u32)));

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
    assert_eq!(s7p2.total_proper_fraction, Some(F::new(2u32, 7u32)));

    let s7p2c = s7p2.convert_to_improper_kind(&INV);
    approx::assert_relative_eq!(
        s7p2c.total_proper_angle,
        4.0 / 7.0 * std::f64::consts::PI,
        max_relative = s7p2c.generating_element.threshold,
        epsilon = s7p2c.generating_element.threshold
    );
    assert_eq!(s7p2c.total_proper_fraction, Some(F::new(2u32, 7u32)));

    let s7p5 = SymmetryOperation::builder()
        .generating_element(s7_element)
        .power(5)
        .build()
        .unwrap();
    approx::assert_relative_eq!(
        s7p5.total_proper_angle,
        -4.0 / 7.0 * std::f64::consts::PI,
        max_relative = s7p5.generating_element.threshold,
        epsilon = s7p5.generating_element.threshold
    );
    assert_eq!(s7p5.total_proper_fraction, Some(F::new_neg(2u32, 7u32)));

    let s7p5c = s7p5.convert_to_improper_kind(&INV);
    approx::assert_relative_eq!(
        s7p5c.total_proper_angle,
        3.0 / 7.0 * std::f64::consts::PI,
        max_relative = s7p5c.generating_element.threshold,
        epsilon = s7p5c.generating_element.threshold
    );
    assert_eq!(s7p5c.total_proper_fraction, Some(F::new(3u32, 14u32)));

    let s7pp2_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(7))
        .proper_power(2)
        .raw_axis(Vector3::new(2.0, 2.5, 1.0))
        .kind(SIG)
        .rotation_group(RotationGroup::SO3)
        .build()
        .unwrap();

    let s7pp2 = SymmetryOperation::builder()
        .generating_element(s7pp2_element)
        .power(1)
        .build()
        .unwrap();
    approx::assert_relative_eq!(
        s7pp2.total_proper_angle,
        4.0 / 7.0 * std::f64::consts::PI,
        max_relative = s7pp2.generating_element.threshold,
        epsilon = s7pp2.generating_element.threshold
    );
    assert_eq!(s7pp2.total_proper_fraction, Some(F::new(2u32, 7u32)));

    let s7pp2c = s7pp2.convert_to_improper_kind(&INV);
    approx::assert_relative_eq!(
        s7pp2c.total_proper_angle,
        -3.0 / 7.0 * std::f64::consts::PI,
        max_relative = s7pp2c.generating_element.threshold,
        epsilon = s7pp2c.generating_element.threshold
    );
    assert_eq!(s7pp2c.total_proper_fraction, Some(F::new_neg(3u32, 14u32)));

    let si_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Inf)
        .raw_axis(Vector3::new(1.0, 0.0, 1.0))
        .proper_angle(2.0 * std::f64::consts::FRAC_PI_4)
        .kind(SIG)
        .rotation_group(RotationGroup::SO3)
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
        .generating_element(si_element)
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
        .raw_axis(Vector3::new(0.0, 2.0, 0.0))
        .kind(ROT)
        .rotation_group(RotationGroup::SO3)
        .build()
        .unwrap();

    let c1 = SymmetryOperation::builder()
        .generating_element(c1_element.clone())
        .power(1)
        .build()
        .unwrap();
    approx::assert_relative_eq!(c1.calc_pole(), Point3::from(Vector3::z()));

    let c1b = SymmetryOperation::builder()
        .generating_element(c1_element)
        .power(-3)
        .build()
        .unwrap();
    approx::assert_relative_eq!(c1b.calc_pole(), Point3::from(Vector3::z()));

    let c2_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(2))
        .proper_power(1)
        .raw_axis(Vector3::new(1.0, 1.0, 0.0))
        .kind(ROT)
        .rotation_group(RotationGroup::SO3)
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
        .generating_element(c2_element)
        .power(2)
        .build()
        .unwrap();
    approx::assert_relative_eq!(c2p2.calc_pole(), Point3::from(Vector3::z()));

    let c2b_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(2))
        .proper_power(1)
        .raw_axis(Vector3::new(-1.0, 1.0, 0.0))
        .kind(ROT)
        .rotation_group(RotationGroup::SO3)
        .build()
        .unwrap();

    let c2b = SymmetryOperation::builder()
        .generating_element(c2b_element.clone())
        .power(1)
        .build()
        .unwrap();
    approx::assert_relative_eq!(c2b.calc_pole(), Point3::new(1.0, -1.0, 0.0) / 2.0f64.sqrt());

    let c2bpm1 = SymmetryOperation::builder()
        .generating_element(c2b_element)
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
        .raw_axis(Vector3::new(1.0, 1.0, 1.0))
        .kind(ROT)
        .rotation_group(RotationGroup::SO3)
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
    approx::assert_relative_eq!(c3p3.calc_pole(), Point3::from(Vector3::z()));

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
        .raw_axis(Vector3::new(1.0, 1.0, 1.0))
        .kind(ROT)
        .rotation_group(RotationGroup::SO3)
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
        .raw_axis(Vector3::new(1.0, 1.0, -1.0))
        .kind(ROT)
        .rotation_group(RotationGroup::SO3)
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
        .generating_element(c4_element)
        .power(4)
        .build()
        .unwrap();
    approx::assert_relative_eq!(c4p4.calc_pole(), Point3::from(Vector3::z()));

    let c7_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(7))
        .proper_power(1)
        .raw_axis(Vector3::new(1.0, 1.0, -2.0))
        .kind(ROT)
        .rotation_group(RotationGroup::SO3)
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
    approx::assert_relative_eq!(c7p7.calc_pole(), Point3::from(Vector3::z()));

    let ci_element = SymmetryElement::builder()
        .threshold(1e-7)
        .proper_order(ElementOrder::Inf)
        .raw_axis(Vector3::new(1.0, 0.0, -1.0))
        .proper_angle(2.0 * std::f64::consts::FRAC_PI_6)
        .kind(ROT)
        .rotation_group(RotationGroup::SO3)
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
        .generating_element(ci_element)
        .power(6)
        .build()
        .unwrap();
    approx::assert_relative_eq!(cip6.calc_pole(), Point3::from(Vector3::z()));

    // ============================
    // Improper symmetry operations
    // ============================
    let s1_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(1))
        .proper_power(1)
        .raw_axis(Vector3::new(0.0, -2.0, 0.0))
        .kind(SIG)
        .rotation_group(RotationGroup::SO3)
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
    approx::assert_relative_eq!(s1pm2.calc_pole(), Point3::from(Vector3::z()));

    let s1pm2c = s1pm2.convert_to_improper_kind(&INV);
    approx::assert_relative_eq!(s1pm2.calc_pole(), s1pm2c.calc_pole());

    let sd2_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(2))
        .proper_power(1)
        .raw_axis(Vector3::new(-1.0, 1.0, 0.0))
        .kind(INV)
        .rotation_group(RotationGroup::SO3)
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
    approx::assert_relative_eq!(sd2p2.calc_pole(), Point3::from(Vector3::z()));

    let sd2p2c = sd2p2.convert_to_improper_kind(&SIG);
    approx::assert_relative_eq!(sd2p2.calc_pole(), sd2p2c.calc_pole());

    let sd2pp2_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(2))
        .proper_power(2)
        .raw_axis(Vector3::new(-1.0, 1.0, 0.0))
        .kind(INV)
        .rotation_group(RotationGroup::SO3)
        .build()
        .unwrap();

    let sd2pp2 = SymmetryOperation::builder()
        .generating_element(sd2pp2_element)
        .power(1)
        .build()
        .unwrap();
    approx::assert_relative_eq!(sd2pp2.calc_pole(), Point3::from(Vector3::z()));

    let sd2pp2c = sd2pp2.convert_to_improper_kind(&SIG);
    approx::assert_relative_eq!(sd2pp2.calc_pole(), sd2pp2c.calc_pole());

    let s2_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(2))
        .proper_power(1)
        .raw_axis(Vector3::new(2.0, 2.0, 1.0))
        .kind(SIG)
        .rotation_group(RotationGroup::SO3)
        .build()
        .unwrap();

    let s2 = SymmetryOperation::builder()
        .generating_element(s2_element.clone())
        .power(1)
        .build()
        .unwrap();
    approx::assert_relative_eq!(s2.calc_pole(), Point3::from(Vector3::z()));

    let s2c = s2.convert_to_improper_kind(&INV);
    approx::assert_relative_eq!(s2.calc_pole(), s2c.calc_pole());

    let s2p2 = SymmetryOperation::builder()
        .generating_element(s2_element)
        .power(2)
        .build()
        .unwrap();
    approx::assert_relative_eq!(s2p2.calc_pole(), Point3::from(Vector3::z()));

    let s2p2c = s2p2.convert_to_improper_kind(&INV);
    approx::assert_relative_eq!(s2p2.calc_pole(), s2p2c.calc_pole());

    let s2pp2_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(2))
        .proper_power(2)
        .raw_axis(Vector3::new(2.0, 2.0, 1.0))
        .kind(SIG)
        .rotation_group(RotationGroup::SO3)
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
    approx::assert_relative_eq!(s2pp2p4.calc_pole(), Point3::from(Vector3::z()));

    let s2pp2p4c = s2pp2p4.convert_to_improper_kind(&SIG);
    approx::assert_relative_eq!(s2pp2p4.calc_pole(), s2pp2p4c.calc_pole());

    let sd1_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(1))
        .proper_power(1)
        .raw_axis(Vector3::new(1.0, 1.0, 1.0))
        .kind(INV)
        .rotation_group(RotationGroup::SO3)
        .build()
        .unwrap();

    let sd1 = SymmetryOperation::builder()
        .generating_element(sd1_element.clone())
        .power(1)
        .build()
        .unwrap();
    approx::assert_relative_eq!(sd1.calc_pole(), Point3::from(Vector3::z()));

    let sd1c = sd1.convert_to_improper_kind(&SIG);
    approx::assert_relative_eq!(sd1.calc_pole(), sd1c.calc_pole());

    let sd1pm2 = SymmetryOperation::builder()
        .generating_element(sd1_element)
        .power(-2)
        .build()
        .unwrap();
    approx::assert_relative_eq!(sd1pm2.calc_pole(), Point3::from(Vector3::z()));

    let sd1pm2c = sd1pm2.convert_to_improper_kind(&SIG);
    approx::assert_relative_eq!(sd1pm2.calc_pole(), sd1pm2c.calc_pole());

    let s3_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(3))
        .proper_power(1)
        .raw_axis(Vector3::new(2.0, 2.0, 1.0))
        .kind(SIG)
        .rotation_group(RotationGroup::SO3)
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
    approx::assert_relative_eq!(s3p6.calc_pole(), Point3::from(Vector3::z()));

    let s3pm6 = SymmetryOperation::builder()
        .generating_element(s3_element)
        .power(-6)
        .build()
        .unwrap();
    approx::assert_relative_eq!(s3pm6.calc_pole(), Point3::from(Vector3::z()));

    let s3pp2_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(3))
        .proper_power(2)
        .raw_axis(Vector3::new(2.0, 2.0, 1.0))
        .kind(SIG)
        .rotation_group(RotationGroup::SO3)
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
        .generating_element(s3pp2_element)
        .power(3)
        .build()
        .unwrap();
    approx::assert_relative_eq!(s3pp2p3.calc_pole(), s3p3.calc_pole());

    let s3pp3_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(3))
        .proper_power(3)
        .raw_axis(Vector3::new(2.0, 2.0, 1.0))
        .kind(SIG)
        .rotation_group(RotationGroup::SO3)
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
    approx::assert_relative_eq!(s3pp3p2.calc_pole(), Point3::from(Vector3::z()));

    let sd3_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(3))
        .proper_power(1)
        .raw_axis(Vector3::new(1.0, 1.0, 1.0))
        .kind(INV)
        .rotation_group(RotationGroup::SO3)
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
    approx::assert_relative_eq!(sd3p3.calc_pole(), Point3::from(Vector3::z()));

    let s7_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(7))
        .proper_power(1)
        .raw_axis(Vector3::new(2.0, 2.0, -1.0))
        .kind(SIG)
        .rotation_group(RotationGroup::SO3)
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
        .raw_axis(Vector3::new(1.0, 0.0, 1.0))
        .proper_angle(2.0 * std::f64::consts::PI / 5.0)
        .kind(SIG)
        .rotation_group(RotationGroup::SO3)
        .build()
        .unwrap();

    let si = SymmetryOperation::builder()
        .generating_element(si_element.clone())
        .power(1)
        .build()
        .unwrap();
    approx::assert_relative_eq!(si.calc_pole(), -Point3::new(1.0, 0.0, 1.0) / 2.0f64.sqrt());

    let sip2 = SymmetryOperation::builder()
        .generating_element(si_element)
        .power(2)
        .build()
        .unwrap();
    approx::assert_relative_eq!(sip2.calc_pole(), Point3::new(1.0, 0.0, 1.0) / 2.0f64.sqrt());

    let sdi_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Inf)
        .raw_axis(Vector3::new(1.0, 0.0, 1.0))
        .proper_angle(2.0 * std::f64::consts::PI / 5.0)
        .kind(INV)
        .rotation_group(RotationGroup::SO3)
        .build()
        .unwrap();

    let sdi = SymmetryOperation::builder()
        .generating_element(sdi_element.clone())
        .power(1)
        .build()
        .unwrap();
    approx::assert_relative_eq!(sdi.calc_pole(), Point3::new(1.0, 0.0, 1.0) / 2.0f64.sqrt());

    let sdip2 = SymmetryOperation::builder()
        .generating_element(sdi_element)
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
        .raw_axis(Vector3::new(0.0, 2.0, 0.0))
        .kind(ROT)
        .rotation_group(RotationGroup::SO3)
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
        .raw_axis(Vector3::new(1.0, 1.0, 0.0))
        .kind(ROT)
        .rotation_group(RotationGroup::SO3)
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
        .generating_element(c2_element)
        .power(2)
        .build()
        .unwrap();
    assert_eq!(c2p2, c1);

    let c2b_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(2))
        .proper_power(1)
        .raw_axis(Vector3::new(-1.0, -1.0, 0.0))
        .kind(ROT)
        .rotation_group(RotationGroup::SO3)
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
        .raw_axis(Vector3::new(1.0, 1.0, 1.0))
        .kind(ROT)
        .rotation_group(RotationGroup::SO3)
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
        .generating_element(c3_element)
        .power(4)
        .build()
        .unwrap();
    assert_eq!(c3p4, c3);

    let c3pp2_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::new(3.0, 1e-14))
        .proper_power(2)
        .raw_axis(Vector3::new(1.0, 1.0, 1.0))
        .kind(ROT)
        .rotation_group(RotationGroup::SO3)
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
        .generating_element(c3pp2_element)
        .power(3)
        .build()
        .unwrap();
    assert_eq!(c1, c3pp2p3);

    let c4_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::new(4.0, 1e-14))
        .proper_power(1)
        .raw_axis(Vector3::new(1.0, 1.0, -1.0))
        .kind(ROT)
        .rotation_group(RotationGroup::SO3)
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
        .generating_element(c4_element)
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
        .raw_axis(-Vector3::new(1.0, 1.0, -1.0))
        .kind(ROT)
        .rotation_group(RotationGroup::SO3)
        .build()
        .unwrap();

    let c4b = SymmetryOperation::builder()
        .generating_element(c4b_element.clone())
        .power(1)
        .build()
        .unwrap();

    let c4bpm1 = SymmetryOperation::builder()
        .generating_element(c4b_element)
        .power(-1)
        .build()
        .unwrap();

    assert_eq!(c4b, c4pm1);
    assert_eq!(c4bpm1, c4);

    let c6_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(6))
        .proper_power(1)
        .raw_axis(Vector3::new(1.0, 1.0, 1.0))
        .kind(ROT)
        .rotation_group(RotationGroup::SO3)
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
        .raw_axis(-Vector3::new(1.0, 1.0, 1.0))
        .proper_angle(2.0 * std::f64::consts::FRAC_PI_6)
        .kind(ROT)
        .rotation_group(RotationGroup::SO3)
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
        .raw_axis(Vector3::new(0.0, -2.0, 0.0))
        .kind(SIG)
        .rotation_group(RotationGroup::SO3)
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
        .raw_axis(Vector3::new(-1.0, 1.0, 0.0))
        .kind(INV)
        .rotation_group(RotationGroup::SO3)
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
        .raw_axis(Vector3::new(-1.0, 1.0, 0.0))
        .kind(INV)
        .rotation_group(RotationGroup::SO3)
        .build()
        .unwrap();

    let sd2pp2 = SymmetryOperation::builder()
        .generating_element(sd2pp2_element)
        .power(1)
        .build()
        .unwrap();

    let sd2pp2c = sd2pp2.convert_to_improper_kind(&SIG);
    assert_eq!(sd2pp2, sd2pp2c);

    let s2_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(2))
        .proper_power(1)
        .raw_axis(Vector3::new(2.0, 2.0, 1.0))
        .kind(SIG)
        .rotation_group(RotationGroup::SO3)
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
        .raw_axis(Vector3::new(0.0, 1.0, 0.0))
        .kind(SIG)
        .rotation_group(RotationGroup::SO3)
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
        .raw_axis(Vector3::new(1.0, 1.0, 1.0))
        .kind(INV)
        .rotation_group(RotationGroup::SO3)
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
        .raw_axis(Vector3::new(1.0, 1.0, 1.0))
        .kind(SIG)
        .rotation_group(RotationGroup::SO3)
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
        .generating_element(s3_element)
        .power(6)
        .build()
        .unwrap();
    assert_eq!(s3p6, c3p3);

    let s3pp2_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(3))
        .proper_power(2)
        .raw_axis(-Vector3::new(1.0, 1.0, 1.0))
        .kind(SIG)
        .rotation_group(RotationGroup::SO3)
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
        .generating_element(s3pp2_element)
        .power(2)
        .build()
        .unwrap();
    assert_eq!(s3pp2p2, c3pm1);

    let s3pp3_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(3))
        .proper_power(3)
        .raw_axis(Vector3::new(1.0, 1.0, 1.0))
        .kind(SIG)
        .rotation_group(RotationGroup::SO3)
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
        .raw_axis(Vector3::new(1.0, 1.0, 1.0))
        .kind(INV)
        .rotation_group(RotationGroup::SO3)
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
        .raw_axis(Vector3::new(1.0, 1.0, 1.0))
        .kind(SIG)
        .rotation_group(RotationGroup::SO3)
        .build()
        .unwrap();

    let s6pp5 = SymmetryOperation::builder()
        .generating_element(s6pp5_element)
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
        .raw_axis(Vector3::new(2.0, 2.0, -1.0))
        .kind(SIG)
        .rotation_group(RotationGroup::SO3)
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
        .generating_element(s7_element)
        .power(11)
        .build()
        .unwrap();
    assert_eq!(s7p11, s7pm3);

    let si_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Inf)
        .raw_axis(Vector3::new(1.0, 0.0, 1.0))
        .proper_angle(2.0 * std::f64::consts::PI / 5.0)
        .kind(SIG)
        .rotation_group(RotationGroup::SO3)
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
        .generating_element(si_element)
        .power(-8)
        .build()
        .unwrap();

    assert_eq!(sipm8, sip2);
}

#[test]
fn test_symmetry_operation_to_quaternion() {
    // ==========================
    // Proper symmetry operations
    // ==========================
    let c3_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::new(3.0, 1e-14))
        .proper_power(1)
        .raw_axis(Vector3::new(2.0, -1.0, 1.0))
        .kind(ROT)
        .rotation_group(RotationGroup::SO3)
        .build()
        .unwrap();

    let c3 = SymmetryOperation::builder()
        .generating_element(c3_element.clone())
        .power(1)
        .build()
        .unwrap();

    let (c3_sca, c3_vec) = c3.calc_quaternion();
    approx::assert_relative_eq!(
        c3_sca,
        (0.5 * (2.0 * std::f64::consts::PI / 3.0)).cos(),
        epsilon = c3_element.threshold,
        max_relative = c3_element.threshold
    );
    approx::assert_relative_eq!(
        c3_vec,
        (0.5 * (2.0 * std::f64::consts::PI / 3.0)).sin() * Vector3::new(2.0, -1.0, 1.0)
            / (6.0f64.sqrt()),
        epsilon = c3_element.threshold,
        max_relative = c3_element.threshold
    );

    let c3p2 = SymmetryOperation::builder()
        .generating_element(c3_element.clone())
        .power(2)
        .build()
        .unwrap();

    let (c3p2_sca, c3p2_vec) = c3p2.calc_quaternion();
    approx::assert_relative_eq!(
        c3p2_sca,
        (0.5 * (2.0 * std::f64::consts::PI / 3.0)).cos(),
        epsilon = c3_element.threshold,
        max_relative = c3_element.threshold
    );
    approx::assert_relative_eq!(
        c3p2_vec,
        -(0.5 * (2.0 * std::f64::consts::PI / 3.0)).sin() * Vector3::new(2.0, -1.0, 1.0)
            / (6.0f64.sqrt()),
        epsilon = c3_element.threshold,
        max_relative = c3_element.threshold
    );

    let c3p3 = SymmetryOperation::builder()
        .generating_element(c3_element.clone())
        .power(3)
        .build()
        .unwrap();

    let (c3p3_sca, c3p3_vec) = c3p3.calc_quaternion();
    approx::assert_relative_eq!(
        c3p3_sca,
        (0.0f64).cos(),
        epsilon = c3_element.threshold,
        max_relative = c3_element.threshold
    );
    approx::assert_relative_eq!(
        c3p3_vec,
        Vector3::zeros(),
        epsilon = c3_element.threshold,
        max_relative = c3_element.threshold
    );

    // ============================
    // Improper symmetry operations
    // ============================
    let s3_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::new(3.0, 1e-14))
        .proper_power(1)
        .raw_axis(Vector3::new(2.0, -1.0, 1.0))
        .kind(SIG)
        .rotation_group(RotationGroup::SO3)
        .build()
        .unwrap();

    let s3 = SymmetryOperation::builder()
        .generating_element(s3_element.clone())
        .power(1)
        .build()
        .unwrap();

    let (s3_sca, s3_vec) = s3.calc_quaternion();
    approx::assert_relative_eq!(
        s3_sca,
        (0.5 * (std::f64::consts::PI / 3.0)).cos(),
        epsilon = s3_element.threshold,
        max_relative = s3_element.threshold
    );
    approx::assert_relative_eq!(
        s3_vec,
        -(0.5 * (std::f64::consts::PI / 3.0)).sin() * Vector3::new(2.0, -1.0, 1.0)
            / (6.0f64.sqrt()),
        epsilon = s3_element.threshold,
        max_relative = s3_element.threshold
    );

    let s3p2 = SymmetryOperation::builder()
        .generating_element(s3_element.clone())
        .power(2)
        .build()
        .unwrap();

    let (s3p2_sca, s3p2_vec) = s3p2.calc_quaternion();
    approx::assert_relative_eq!(
        s3p2_sca,
        (0.5 * (2.0 * std::f64::consts::PI / 3.0)).cos(),
        epsilon = s3_element.threshold,
        max_relative = s3_element.threshold
    );
    approx::assert_relative_eq!(
        s3p2_vec,
        -(0.5 * (2.0 * std::f64::consts::PI / 3.0)).sin() * Vector3::new(2.0, -1.0, 1.0)
            / (6.0f64.sqrt()),
        epsilon = s3_element.threshold,
        max_relative = s3_element.threshold
    );

    let s3p3 = SymmetryOperation::builder()
        .generating_element(s3_element.clone())
        .power(3)
        .build()
        .unwrap();

    let (s3p3_sca, s3p3_vec) = s3p3.calc_quaternion();
    approx::assert_relative_eq!(
        s3p3_sca,
        std::f64::consts::FRAC_PI_2.cos(),
        epsilon = s3_element.threshold,
        max_relative = s3_element.threshold
    );
    approx::assert_relative_eq!(
        s3p3_vec,
        std::f64::consts::FRAC_PI_2.sin() * Vector3::new(2.0, -1.0, 1.0) / (6.0f64.sqrt()),
        epsilon = s3_element.threshold,
        max_relative = s3_element.threshold
    );

    let sd3_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::new(3.0, 1e-14))
        .proper_power(1)
        .raw_axis(Vector3::new(2.0, -1.0, 1.0))
        .kind(INV)
        .rotation_group(RotationGroup::SO3)
        .build()
        .unwrap();

    let sd3 = SymmetryOperation::builder()
        .generating_element(sd3_element.clone())
        .power(1)
        .build()
        .unwrap();

    let (sd3_sca, sd3_vec) = sd3.calc_quaternion();
    approx::assert_relative_eq!(
        sd3_sca,
        (0.5 * (2.0 * std::f64::consts::PI / 3.0)).cos(),
        epsilon = sd3_element.threshold,
        max_relative = sd3_element.threshold
    );
    approx::assert_relative_eq!(
        sd3_vec,
        (0.5 * (2.0 * std::f64::consts::PI / 3.0)).sin() * Vector3::new(2.0, -1.0, 1.0)
            / (6.0f64.sqrt()),
        epsilon = sd3_element.threshold,
        max_relative = sd3_element.threshold
    );

    let sd3p2 = SymmetryOperation::builder()
        .generating_element(sd3_element.clone())
        .power(2)
        .build()
        .unwrap();

    let (sd3p2_sca, sd3p2_vec) = sd3p2.calc_quaternion();
    approx::assert_relative_eq!(
        sd3p2_sca,
        (0.5 * (2.0 * std::f64::consts::PI / 3.0)).cos(),
        epsilon = sd3_element.threshold,
        max_relative = sd3_element.threshold
    );
    approx::assert_relative_eq!(
        sd3p2_vec,
        -(0.5 * (2.0 * std::f64::consts::PI / 3.0)).sin() * Vector3::new(2.0, -1.0, 1.0)
            / (6.0f64.sqrt()),
        epsilon = sd3_element.threshold,
        max_relative = sd3_element.threshold
    );

    let sd3p3 = SymmetryOperation::builder()
        .generating_element(sd3_element.clone())
        .power(3)
        .build()
        .unwrap();

    let (sd3p3_sca, sd3p3_vec) = sd3p3.calc_quaternion();
    approx::assert_relative_eq!(
        sd3p3_sca,
        (0.0f64).cos(),
        epsilon = sd3_element.threshold,
        max_relative = sd3_element.threshold
    );
    approx::assert_relative_eq!(
        sd3p3_vec,
        Vector3::zeros(),
        epsilon = sd3_element.threshold,
        max_relative = sd3_element.threshold
    );
}

#[test]
fn test_symmetry_operation_from_quaternion() {
    // ==========================
    // Proper symmetry operations
    // ==========================
    let c4_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(4))
        .proper_power(1)
        .raw_axis(Vector3::new(2.0, -1.0, 1.0))
        .kind(ROT)
        .rotation_group(RotationGroup::SO3)
        .build()
        .unwrap();

    let c4p2 = SymmetryOperation::builder()
        .generating_element(c4_element.clone())
        .power(1)
        .build()
        .unwrap();

    let c4p2q = c4p2.calc_quaternion();
    let c4p2r = SymmetryOperation::from_quaternion(
        c4p2q,
        c4p2.is_proper(),
        c4_element.threshold,
        10,
        false,
        false,
        None,
    );
    assert_eq!(c4p2, c4p2r);

    let c7_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(7))
        .proper_power(1)
        .raw_axis(Vector3::new(2.0, -2.0, 1.0))
        .kind(ROT)
        .rotation_group(RotationGroup::SO3)
        .build()
        .unwrap();

    let c7 = SymmetryOperation::builder()
        .generating_element(c7_element.clone())
        .power(1)
        .build()
        .unwrap();

    let c7q = c7.calc_quaternion();
    let c7r = SymmetryOperation::from_quaternion(
        c7q,
        c7.is_proper(),
        c7_element.threshold,
        10,
        false,
        false,
        None,
    );
    assert_eq!(c7, c7r);

    let c7p2 = SymmetryOperation::builder()
        .generating_element(c7_element.clone())
        .power(2)
        .build()
        .unwrap();

    let c7p2q = c7p2.calc_quaternion();
    let c7p2r = SymmetryOperation::from_quaternion(
        c7p2q,
        c7p2.is_proper(),
        c7_element.threshold,
        10,
        false,
        false,
        None,
    );
    assert_eq!(c7p2, c7p2r);

    let c7p3 = SymmetryOperation::builder()
        .generating_element(c7_element.clone())
        .power(3)
        .build()
        .unwrap();

    let c7p3q = c7p3.calc_quaternion();
    let c7p3r = SymmetryOperation::from_quaternion(
        c7p3q,
        c7p3.is_proper(),
        c7_element.threshold,
        10,
        false,
        false,
        None,
    );
    assert_eq!(c7p3, c7p3r);

    let c7p7 = SymmetryOperation::builder()
        .generating_element(c7_element.clone())
        .power(7)
        .build()
        .unwrap();

    let c7p7q = c7p7.calc_quaternion();
    let c7p7r = SymmetryOperation::from_quaternion(
        c7p7q,
        c7p7.is_proper(),
        c7_element.threshold,
        10,
        false,
        false,
        None,
    );
    assert_eq!(c7p7, c7p7r);

    // ============================
    // Improper symmetry operations
    // ============================
    let s1_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(1))
        .proper_power(1)
        .raw_axis(Vector3::new(2.0, -1.0, 2.0))
        .kind(SIG)
        .rotation_group(RotationGroup::SO3)
        .build()
        .unwrap();

    let s1 = SymmetryOperation::builder()
        .generating_element(s1_element.clone())
        .power(1)
        .build()
        .unwrap();

    let s1q = s1.calc_quaternion();
    let s1r = SymmetryOperation::from_quaternion(
        s1q,
        s1.is_proper(),
        s1_element.threshold,
        10,
        false,
        false,
        None,
    );
    assert_eq!(s1, s1r);

    let s1pm4 = SymmetryOperation::builder()
        .generating_element(s1_element.clone())
        .power(-4)
        .build()
        .unwrap();

    let s1pm4q = s1pm4.calc_quaternion();
    let s1pm4r = SymmetryOperation::from_quaternion(
        s1pm4q,
        s1pm4.is_proper(),
        s1_element.threshold,
        10,
        false,
        false,
        None,
    );
    assert_eq!(s1pm4, s1pm4r);

    let s2_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(2))
        .proper_power(1)
        .raw_axis(Vector3::new(2.0, -1.0, 2.0))
        .kind(SIG)
        .rotation_group(RotationGroup::SO3)
        .build()
        .unwrap();

    let s2 = SymmetryOperation::builder()
        .generating_element(s2_element.clone())
        .power(1)
        .build()
        .unwrap();

    let s2q = s2.calc_quaternion();
    let s2r = SymmetryOperation::from_quaternion(
        s2q,
        s2.is_proper(),
        s2_element.threshold,
        10,
        false,
        false,
        None,
    );
    assert_eq!(s2, s2r);

    let s2p2 = SymmetryOperation::builder()
        .generating_element(s2_element.clone())
        .power(2)
        .build()
        .unwrap();

    let s2p2q = s2p2.calc_quaternion();
    let s2p2r = SymmetryOperation::from_quaternion(
        s2p2q,
        s2p2.is_proper(),
        s2_element.threshold,
        10,
        false,
        false,
        None,
    );
    assert_eq!(s2p2, s2p2r);

    let s3_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::new(3.0, 1e-14))
        .proper_power(1)
        .raw_axis(Vector3::new(2.0, -1.0, 1.0))
        .kind(SIG)
        .rotation_group(RotationGroup::SO3)
        .build()
        .unwrap();

    let s3 = SymmetryOperation::builder()
        .generating_element(s3_element.clone())
        .power(1)
        .build()
        .unwrap();

    let s3q = s3.calc_quaternion();
    let s3r = SymmetryOperation::from_quaternion(
        s3q,
        s3.is_proper(),
        s3_element.threshold,
        10,
        false,
        false,
        None,
    );
    assert_eq!(s3, s3r);

    let s3p2 = SymmetryOperation::builder()
        .generating_element(s3_element.clone())
        .power(2)
        .build()
        .unwrap();

    let s3p2q = s3p2.calc_quaternion();
    let s3p2r = SymmetryOperation::from_quaternion(
        s3p2q,
        s3p2.is_proper(),
        s3_element.threshold,
        10,
        false,
        false,
        None,
    );
    assert_eq!(s3p2, s3p2r);

    let s3pm1 = SymmetryOperation::builder()
        .generating_element(s3_element.clone())
        .power(-1)
        .build()
        .unwrap();

    let s3pm1q = s3pm1.calc_quaternion();
    let s3pm1r = SymmetryOperation::from_quaternion(
        s3pm1q,
        s3pm1.is_proper(),
        s3_element.threshold,
        10,
        false,
        false,
        None,
    );
    assert_eq!(s3pm1, s3pm1r);

    let s3p3 = SymmetryOperation::builder()
        .generating_element(s3_element.clone())
        .power(3)
        .build()
        .unwrap();

    let s3p3q = s3p3.calc_quaternion();
    let s3p3r = SymmetryOperation::from_quaternion(
        s3p3q,
        s3p3.is_proper(),
        s3_element.threshold,
        10,
        false,
        false,
        None,
    );
    assert_eq!(s3p3, s3p3r);

    let s17pp3_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(17))
        .proper_power(3)
        .raw_axis(Vector3::new(5.0, -1.0, 2.0))
        .kind(SIG)
        .rotation_group(RotationGroup::SO3)
        .build()
        .unwrap();

    let s17pp3 = SymmetryOperation::builder()
        .generating_element(s17pp3_element.clone())
        .power(1)
        .build()
        .unwrap();

    let s17pp3q = s17pp3.calc_quaternion();
    let s17pp3r = SymmetryOperation::from_quaternion(
        s17pp3q,
        s17pp3.is_proper(),
        s17pp3_element.threshold,
        17,
        false,
        false,
        None,
    );
    assert_eq!(s17pp3, s17pp3r);

    let sd11_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(11))
        .proper_power(1)
        .raw_axis(Vector3::new(2.0, -1.0, 1.0))
        .kind(INV)
        .rotation_group(RotationGroup::SO3)
        .build()
        .unwrap();

    let sd11 = SymmetryOperation::builder()
        .generating_element(sd11_element.clone())
        .power(1)
        .build()
        .unwrap();

    let sd11q = sd11.calc_quaternion();
    let sd11r = SymmetryOperation::from_quaternion(
        sd11q,
        sd11.is_proper(),
        sd11_element.threshold,
        11,
        false,
        false,
        None,
    );
    assert_eq!(sd11, sd11r);

    let sd11p6 = SymmetryOperation::builder()
        .generating_element(sd11_element.clone())
        .power(6)
        .build()
        .unwrap();

    let sd11p6q = sd11p6.calc_quaternion();
    let sd11p6r = SymmetryOperation::from_quaternion(
        sd11p6q,
        sd11p6.is_proper(),
        sd11_element.threshold,
        11,
        false,
        false,
        None,
    );
    assert_eq!(sd11p6, sd11p6r);
}

#[test]
fn test_symmetry_operation_su2_to_quaternion() {
    // ==========================
    // Proper symmetry operations
    // ==========================
    let c2_nsr_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::new(2.0, 1e-14))
        .proper_power(1)
        .raw_axis(Vector3::new(2.0, -1.0, -1.0))
        .kind(ROT)
        .rotation_group(RotationGroup::SU2(true))
        .build()
        .unwrap();

    let c2_nsr_p1 = SymmetryOperation::builder()
        .generating_element(c2_nsr_element.clone())
        .power(1)
        .build()
        .unwrap();
    assert!(c2_nsr_p1.is_spatial_binary_rotation());
    assert!(c2_nsr_p1.is_su2_class_1());

    let (c2_nsr_p1_sca, c2_nsr_p1_vec) = c2_nsr_p1.calc_quaternion();
    approx::assert_relative_eq!(
        c2_nsr_p1_sca,
        0.0,
        epsilon = c2_nsr_element.threshold,
        max_relative = c2_nsr_element.threshold
    );
    approx::assert_relative_eq!(
        c2_nsr_p1_vec,
        Vector3::new(2.0, -1.0, -1.0).normalize(),
        epsilon = c2_nsr_element.threshold,
        max_relative = c2_nsr_element.threshold
    );

    let c2_nsr_p2 = SymmetryOperation::builder()
        .generating_element(c2_nsr_element.clone())
        .power(2)
        .build()
        .unwrap();
    assert!(c2_nsr_p2.is_spatial_identity());
    assert!(c2_nsr_p2.is_su2_class_1());

    let (c2_nsr_p2_sca, c2_nsr_p2_vec) = c2_nsr_p2.calc_quaternion();
    approx::assert_relative_eq!(
        c2_nsr_p2_sca,
        -1.0,
        epsilon = c2_nsr_element.threshold,
        max_relative = c2_nsr_element.threshold
    );
    approx::assert_relative_eq!(
        c2_nsr_p2_vec,
        Vector3::zeros(),
        epsilon = c2_nsr_element.threshold,
        max_relative = c2_nsr_element.threshold
    );

    let c2_nsr_p3 = SymmetryOperation::builder()
        .generating_element(c2_nsr_element.clone())
        .power(3)
        .build()
        .unwrap();
    assert!(c2_nsr_p3.is_spatial_binary_rotation());
    assert!(!c2_nsr_p3.is_su2_class_1());

    let (c2_nsr_p3_sca, c2_nsr_p3_vec) = c2_nsr_p3.calc_quaternion();
    approx::assert_relative_eq!(
        c2_nsr_p3_sca,
        -c2_nsr_p1_sca,
        epsilon = c2_nsr_element.threshold,
        max_relative = c2_nsr_element.threshold
    );
    approx::assert_relative_eq!(
        c2_nsr_p3_vec,
        -c2_nsr_p1_vec,
        epsilon = c2_nsr_element.threshold,
        max_relative = c2_nsr_element.threshold
    );

    let c2_nsr_p4 = SymmetryOperation::builder()
        .generating_element(c2_nsr_element.clone())
        .power(4)
        .build()
        .unwrap();
    assert!(c2_nsr_p4.is_identity());

    let (c2_nsr_p4_sca, c2_nsr_p4_vec) = c2_nsr_p4.calc_quaternion();
    approx::assert_relative_eq!(
        c2_nsr_p4_sca,
        1.0,
        epsilon = c2_nsr_element.threshold,
        max_relative = c2_nsr_element.threshold
    );
    approx::assert_relative_eq!(
        c2_nsr_p4_vec,
        Vector3::zeros(),
        epsilon = c2_nsr_element.threshold,
        max_relative = c2_nsr_element.threshold
    );

    let c3_nsr_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::new(3.0, 1e-14))
        .proper_power(1)
        .raw_axis(Vector3::new(2.0, -1.0, 1.0))
        .kind(ROT)
        .rotation_group(RotationGroup::SU2(true))
        .build()
        .unwrap();

    let c3_nsr_p1 = SymmetryOperation::builder()
        .generating_element(c3_nsr_element.clone())
        .power(1)
        .build()
        .unwrap();

    let (c3_nsr_p1_sca, c3_nsr_p1_vec) = c3_nsr_p1.calc_quaternion();
    approx::assert_relative_eq!(
        c3_nsr_p1_sca,
        (0.5 * (2.0 * std::f64::consts::PI / 3.0)).cos(),
        epsilon = c3_nsr_element.threshold,
        max_relative = c3_nsr_element.threshold
    );
    approx::assert_relative_eq!(
        c3_nsr_p1_vec,
        (0.5 * (2.0 * std::f64::consts::PI / 3.0)).sin() * Vector3::new(2.0, -1.0, 1.0)
            / (6.0f64.sqrt()),
        epsilon = c3_nsr_element.threshold,
        max_relative = c3_nsr_element.threshold
    );

    let c3_nsr_p2 = SymmetryOperation::builder()
        .generating_element(c3_nsr_element.clone())
        .power(2)
        .build()
        .unwrap();

    let (c3_nsr_p2_sca, c3_nsr_p2_vec) = c3_nsr_p2.calc_quaternion();
    approx::assert_relative_eq!(
        c3_nsr_p2_sca,
        -(0.5 * (2.0 * std::f64::consts::PI / 3.0)).cos(),
        epsilon = c3_nsr_element.threshold,
        max_relative = c3_nsr_element.threshold
    );
    approx::assert_relative_eq!(
        c3_nsr_p2_vec,
        (0.5 * (2.0 * std::f64::consts::PI / 3.0)).sin() * Vector3::new(2.0, -1.0, 1.0)
            / (6.0f64.sqrt()),
        epsilon = c3_nsr_element.threshold,
        max_relative = c3_nsr_element.threshold
    );

    let c3_nsr_p3 = SymmetryOperation::builder()
        .generating_element(c3_nsr_element.clone())
        .power(3)
        .build()
        .unwrap();
    assert!(c3_nsr_p3.is_spatial_identity());

    let (c3_nsr_p3_sca, c3_nsr_p3_vec) = c3_nsr_p3.calc_quaternion();
    approx::assert_relative_eq!(
        c3_nsr_p3_sca,
        -1.0,
        epsilon = c3_nsr_element.threshold,
        max_relative = c3_nsr_element.threshold
    );
    approx::assert_relative_eq!(
        c3_nsr_p3_vec,
        Vector3::zeros(),
        epsilon = c3_nsr_element.threshold,
        max_relative = c3_nsr_element.threshold
    );

    let c3_nsr_p4 = SymmetryOperation::builder()
        .generating_element(c3_nsr_element.clone())
        .power(4)
        .build()
        .unwrap();

    let (c3_nsr_p4_sca, c3_nsr_p4_vec) = c3_nsr_p4.calc_quaternion();
    approx::assert_relative_eq!(
        c3_nsr_p4_sca,
        -c3_nsr_p1_sca,
        epsilon = c3_nsr_element.threshold,
        max_relative = c3_nsr_element.threshold
    );
    approx::assert_relative_eq!(
        c3_nsr_p4_vec,
        -c3_nsr_p1_vec,
        epsilon = c3_nsr_element.threshold,
        max_relative = c3_nsr_element.threshold
    );

    let c3_nsr_p5 = SymmetryOperation::builder()
        .generating_element(c3_nsr_element.clone())
        .power(5)
        .build()
        .unwrap();

    let (c3_nsr_p5_sca, c3_nsr_p5_vec) = c3_nsr_p5.calc_quaternion();
    approx::assert_relative_eq!(
        c3_nsr_p5_sca,
        -c3_nsr_p2_sca,
        epsilon = c3_nsr_element.threshold,
        max_relative = c3_nsr_element.threshold
    );
    approx::assert_relative_eq!(
        c3_nsr_p5_vec,
        -c3_nsr_p2_vec,
        epsilon = c3_nsr_element.threshold,
        max_relative = c3_nsr_element.threshold
    );

    let c3_nsr_p6 = SymmetryOperation::builder()
        .generating_element(c3_nsr_element.clone())
        .power(6)
        .build()
        .unwrap();
    assert!(c3_nsr_p6.is_identity());

    let (c3_nsr_p6_sca, c3_nsr_p6_vec) = c3_nsr_p6.calc_quaternion();
    approx::assert_relative_eq!(
        c3_nsr_p6_sca,
        1.0,
        epsilon = c3_nsr_element.threshold,
        max_relative = c3_nsr_element.threshold
    );
    approx::assert_relative_eq!(
        c3_nsr_p6_vec,
        Vector3::zeros(),
        epsilon = c3_nsr_element.threshold,
        max_relative = c3_nsr_element.threshold
    );

    // ============================
    // Improper symmetry operations
    // ============================
    let s1_nsr_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::new(1.0, 1e-14))
        .proper_power(1)
        .raw_axis(Vector3::new(2.0, -1.0, 1.0))
        .kind(SIG)
        .rotation_group(RotationGroup::SU2(true))
        .build()
        .unwrap();

    let s1_nsr_p1 = SymmetryOperation::builder()
        .generating_element(s1_nsr_element.clone())
        .power(1)
        .build()
        .unwrap();
    assert!(!s1_nsr_p1.is_su2_class_1());
    assert!(s1_nsr_p1.is_spatial_reflection());

    let (s1_nsr_p1_sca, s1_nsr_p1_vec) = s1_nsr_p1.calc_quaternion();
    approx::assert_relative_eq!(
        s1_nsr_p1_sca,
        0.0,
        epsilon = s1_nsr_element.threshold,
        max_relative = s1_nsr_element.threshold
    );
    approx::assert_relative_eq!(
        s1_nsr_p1_vec,
        Vector3::new(2.0, -1.0, 1.0) / 6.0f64.sqrt(),
        epsilon = s1_nsr_element.threshold,
        max_relative = s1_nsr_element.threshold
    );

    let s1_nsr_p2 = SymmetryOperation::builder()
        .generating_element(s1_nsr_element.clone())
        .power(2)
        .build()
        .unwrap();
    assert!(s1_nsr_p2.is_su2_class_1());
    assert!(s1_nsr_p2.is_spatial_identity());

    let (s1_nsr_p2_sca, s1_nsr_p2_vec) = s1_nsr_p2.calc_quaternion();
    approx::assert_relative_eq!(
        s1_nsr_p2_sca,
        -1.0,
        epsilon = s1_nsr_element.threshold,
        max_relative = s1_nsr_element.threshold
    );
    approx::assert_relative_eq!(
        s1_nsr_p2_vec,
        Vector3::zeros(),
        epsilon = s1_nsr_element.threshold,
        max_relative = s1_nsr_element.threshold
    );

    let s1_nsr_p3 = SymmetryOperation::builder()
        .generating_element(s1_nsr_element.clone())
        .power(3)
        .build()
        .unwrap();
    assert!(s1_nsr_p3.is_su2_class_1());
    assert!(s1_nsr_p3.is_spatial_reflection());

    let (s1_nsr_p3_sca, s1_nsr_p3_vec) = s1_nsr_p3.calc_quaternion();
    approx::assert_relative_eq!(
        s1_nsr_p3_sca,
        -s1_nsr_p1_sca,
        epsilon = s1_nsr_element.threshold,
        max_relative = s1_nsr_element.threshold
    );
    approx::assert_relative_eq!(
        s1_nsr_p3_vec,
        -s1_nsr_p1_vec,
        epsilon = s1_nsr_element.threshold,
        max_relative = s1_nsr_element.threshold
    );

    let s1_nsr_p4 = SymmetryOperation::builder()
        .generating_element(s1_nsr_element.clone())
        .power(4)
        .build()
        .unwrap();
    assert!(!s1_nsr_p4.is_su2_class_1());
    assert!(s1_nsr_p4.is_identity());

    let (s1_nsr_p4_sca, s1_nsr_p4_vec) = s1_nsr_p4.calc_quaternion();
    approx::assert_relative_eq!(
        s1_nsr_p4_sca,
        1.0,
        epsilon = s1_nsr_element.threshold,
        max_relative = s1_nsr_element.threshold
    );
    approx::assert_relative_eq!(
        s1_nsr_p4_vec,
        Vector3::zeros(),
        epsilon = s1_nsr_element.threshold,
        max_relative = s1_nsr_element.threshold
    );

    let s2_nsr_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::new(2.0, 1e-14))
        .proper_power(1)
        .raw_axis(Vector3::new(2.0, -1.0, 1.0))
        .kind(SIG)
        .rotation_group(RotationGroup::SU2(true))
        .build()
        .unwrap();

    let s2_nsr_p1 = SymmetryOperation::builder()
        .generating_element(s2_nsr_element.clone())
        .power(1)
        .build()
        .unwrap();
    assert!(!s2_nsr_p1.is_su2_class_1());
    assert!(s2_nsr_p1.is_spatial_inversion());
    assert!(!s2_nsr_p1.is_identity());

    let (s2_nsr_p1_sca, s2_nsr_p1_vec) = s2_nsr_p1.calc_quaternion();
    approx::assert_relative_eq!(
        s2_nsr_p1_sca,
        1.0,
        epsilon = s2_nsr_element.threshold,
        max_relative = s2_nsr_element.threshold
    );
    approx::assert_relative_eq!(
        s2_nsr_p1_vec,
        Vector3::zeros(),
        epsilon = s2_nsr_element.threshold,
        max_relative = s2_nsr_element.threshold
    );

    let s2_nsr_p2 = SymmetryOperation::builder()
        .generating_element(s2_nsr_element.clone())
        .power(2)
        .build()
        .unwrap();
    assert!(!s2_nsr_p2.is_su2_class_1());
    assert!(s2_nsr_p2.is_identity());

    let (s2_nsr_p2_sca, s2_nsr_p2_vec) = s2_nsr_p2.calc_quaternion();
    approx::assert_relative_eq!(
        s2_nsr_p2_sca,
        1.0,
        epsilon = s2_nsr_element.threshold,
        max_relative = s2_nsr_element.threshold
    );
    approx::assert_relative_eq!(
        s2_nsr_p2_vec,
        Vector3::zeros(),
        epsilon = s2_nsr_element.threshold,
        max_relative = s2_nsr_element.threshold
    );

    // ---------------
    // S3(n) = iC6(-n)
    // ---------------
    let s3_nsr_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::new(3.0, 1e-14))
        .proper_power(1)
        .raw_axis(Vector3::new(2.0, -1.0, 1.0))
        .kind(SIG)
        .rotation_group(RotationGroup::SU2(true))
        .build()
        .unwrap();

    let s3_nsr_p1 = SymmetryOperation::builder()
        .generating_element(s3_nsr_element.clone())
        .power(1)
        .build()
        .unwrap();
    assert!(!s3_nsr_p1.is_su2_class_1());

    let (s3_nsr_p1_sca, s3_nsr_p1_vec) = s3_nsr_p1.calc_quaternion();
    approx::assert_relative_eq!(
        s3_nsr_p1_sca,
        (0.5 * (std::f64::consts::PI / 3.0)).cos(),
        epsilon = s3_nsr_element.threshold,
        max_relative = s3_nsr_element.threshold
    );
    approx::assert_relative_eq!(
        s3_nsr_p1_vec,
        -(0.5 * (std::f64::consts::PI / 3.0)).sin() * Vector3::new(2.0, -1.0, 1.0)
            / (6.0f64.sqrt()),
        epsilon = s3_nsr_element.threshold,
        max_relative = s3_nsr_element.threshold
    );

    let s3_nsr_p2 = SymmetryOperation::builder()
        .generating_element(s3_nsr_element.clone())
        .power(2)
        .build()
        .unwrap();
    assert!(!s3_nsr_p2.is_su2_class_1());
    assert!(s3_nsr_p2.is_proper());

    let (s3_nsr_p2_sca, s3_nsr_p2_vec) = s3_nsr_p2.calc_quaternion();
    approx::assert_relative_eq!(
        s3_nsr_p2_sca,
        (0.5 * (2.0 * std::f64::consts::PI / 3.0)).cos(),
        epsilon = s3_nsr_element.threshold,
        max_relative = s3_nsr_element.threshold
    );
    approx::assert_relative_eq!(
        s3_nsr_p2_vec,
        -(0.5 * (2.0 * std::f64::consts::PI / 3.0)).sin() * Vector3::new(2.0, -1.0, 1.0)
            / (6.0f64.sqrt()),
        epsilon = s3_nsr_element.threshold,
        max_relative = s3_nsr_element.threshold
    );

    let s3_nsr_p3 = SymmetryOperation::builder()
        .generating_element(s3_nsr_element.clone())
        .power(3)
        .build()
        .unwrap();
    assert!(s3_nsr_p3.is_su2_class_1());
    assert!(!s3_nsr_p3.is_proper());
    assert!(s3_nsr_p3.is_spatial_reflection());

    let (s3_nsr_p3_sca, s3_nsr_p3_vec) = s3_nsr_p3.calc_quaternion();
    approx::assert_relative_eq!(
        s3_nsr_p3_sca,
        0.0,
        epsilon = s3_nsr_element.threshold,
        max_relative = s3_nsr_element.threshold
    );
    approx::assert_relative_eq!(
        s3_nsr_p3_vec,
        -Vector3::new(2.0, -1.0, 1.0) / (6.0f64.sqrt()),
        epsilon = s3_nsr_element.threshold,
        max_relative = s3_nsr_element.threshold
    );

    let s3_nsr_p4 = SymmetryOperation::builder()
        .generating_element(s3_nsr_element.clone())
        .power(4)
        .build()
        .unwrap();
    assert!(s3_nsr_p4.is_su2_class_1());
    assert!(s3_nsr_p4.is_proper());

    let (s3_nsr_p4_sca, s3_nsr_p4_vec) = s3_nsr_p4.calc_quaternion();
    approx::assert_relative_eq!(
        s3_nsr_p4_sca,
        -(0.5 * (2.0 * std::f64::consts::PI / 3.0)).cos(),
        epsilon = s3_nsr_element.threshold,
        max_relative = s3_nsr_element.threshold
    );
    approx::assert_relative_eq!(
        s3_nsr_p4_vec,
        -(0.5 * (2.0 * std::f64::consts::PI / 3.0)).sin() * Vector3::new(2.0, -1.0, 1.0)
            / (6.0f64.sqrt()),
        epsilon = s3_nsr_element.threshold,
        max_relative = s3_nsr_element.threshold
    );

    let s3_nsr_p5 = SymmetryOperation::builder()
        .generating_element(s3_nsr_element.clone())
        .power(5)
        .build()
        .unwrap();
    assert!(s3_nsr_p5.is_su2_class_1());
    assert!(!s3_nsr_p5.is_proper());

    let (s3_nsr_p5_sca, s3_nsr_p5_vec) = s3_nsr_p5.calc_quaternion();
    approx::assert_relative_eq!(
        s3_nsr_p5_sca,
        -(0.5 * (std::f64::consts::PI / 3.0)).cos(),
        epsilon = s3_nsr_element.threshold,
        max_relative = s3_nsr_element.threshold
    );
    approx::assert_relative_eq!(
        s3_nsr_p5_vec,
        -(0.5 * (std::f64::consts::PI / 3.0)).sin() * Vector3::new(2.0, -1.0, 1.0)
            / (6.0f64.sqrt()),
        epsilon = s3_nsr_element.threshold,
        max_relative = s3_nsr_element.threshold
    );

    let s3_nsr_p6 = SymmetryOperation::builder()
        .generating_element(s3_nsr_element.clone())
        .power(6)
        .build()
        .unwrap();
    assert!(s3_nsr_p6.is_su2_class_1());
    assert!(s3_nsr_p6.is_proper());
    assert!(s3_nsr_p6.is_spatial_identity());

    let (s3_nsr_p6_sca, s3_nsr_p6_vec) = s3_nsr_p6.calc_quaternion();
    approx::assert_relative_eq!(
        s3_nsr_p6_sca,
        -1.0,
        epsilon = s3_nsr_element.threshold,
        max_relative = s3_nsr_element.threshold
    );
    approx::assert_relative_eq!(
        s3_nsr_p6_vec,
        Vector3::zeros(),
        epsilon = s3_nsr_element.threshold,
        max_relative = s3_nsr_element.threshold
    );

    let s3_nsr_p7 = SymmetryOperation::builder()
        .generating_element(s3_nsr_element.clone())
        .power(7)
        .build()
        .unwrap();
    assert!(s3_nsr_p7.is_su2_class_1());
    assert!(!s3_nsr_p7.is_proper());

    let (s3_nsr_p7_sca, s3_nsr_p7_vec) = s3_nsr_p7.calc_quaternion();
    approx::assert_relative_eq!(
        s3_nsr_p7_sca,
        -s3_nsr_p1_sca,
        epsilon = s3_nsr_element.threshold,
        max_relative = s3_nsr_element.threshold
    );
    approx::assert_relative_eq!(
        s3_nsr_p7_vec,
        -s3_nsr_p1_vec,
        epsilon = s3_nsr_element.threshold,
        max_relative = s3_nsr_element.threshold
    );

    let s3_nsr_p8 = SymmetryOperation::builder()
        .generating_element(s3_nsr_element.clone())
        .power(8)
        .build()
        .unwrap();
    assert!(s3_nsr_p8.is_su2_class_1());
    assert!(s3_nsr_p8.is_proper());

    let (s3_nsr_p8_sca, s3_nsr_p8_vec) = s3_nsr_p8.calc_quaternion();
    approx::assert_relative_eq!(
        s3_nsr_p8_sca,
        -s3_nsr_p2_sca,
        epsilon = s3_nsr_element.threshold,
        max_relative = s3_nsr_element.threshold
    );
    approx::assert_relative_eq!(
        s3_nsr_p8_vec,
        -s3_nsr_p2_vec,
        epsilon = s3_nsr_element.threshold,
        max_relative = s3_nsr_element.threshold
    );

    let s3_nsr_p9 = SymmetryOperation::builder()
        .generating_element(s3_nsr_element.clone())
        .power(9)
        .build()
        .unwrap();
    assert!(!s3_nsr_p9.is_su2_class_1());
    assert!(!s3_nsr_p9.is_proper());

    let (s3_nsr_p9_sca, s3_nsr_p9_vec) = s3_nsr_p9.calc_quaternion();
    approx::assert_relative_eq!(
        s3_nsr_p9_sca,
        -s3_nsr_p3_sca,
        epsilon = s3_nsr_element.threshold,
        max_relative = s3_nsr_element.threshold
    );
    approx::assert_relative_eq!(
        s3_nsr_p9_vec,
        -s3_nsr_p3_vec,
        epsilon = s3_nsr_element.threshold,
        max_relative = s3_nsr_element.threshold
    );

    let s3_nsr_p10 = SymmetryOperation::builder()
        .generating_element(s3_nsr_element.clone())
        .power(10)
        .build()
        .unwrap();
    assert!(!s3_nsr_p10.is_su2_class_1());
    assert!(s3_nsr_p10.is_proper());

    let (s3_nsr_p10_sca, s3_nsr_p10_vec) = s3_nsr_p10.calc_quaternion();
    approx::assert_relative_eq!(
        s3_nsr_p10_sca,
        -s3_nsr_p4_sca,
        epsilon = s3_nsr_element.threshold,
        max_relative = s3_nsr_element.threshold
    );
    approx::assert_relative_eq!(
        s3_nsr_p10_vec,
        -s3_nsr_p4_vec,
        epsilon = s3_nsr_element.threshold,
        max_relative = s3_nsr_element.threshold
    );

    let s3_nsr_p11 = SymmetryOperation::builder()
        .generating_element(s3_nsr_element.clone())
        .power(11)
        .build()
        .unwrap();
    assert!(!s3_nsr_p11.is_su2_class_1());
    assert!(!s3_nsr_p11.is_proper());

    let (s3_nsr_p11_sca, s3_nsr_p11_vec) = s3_nsr_p11.calc_quaternion();
    approx::assert_relative_eq!(
        s3_nsr_p11_sca,
        -s3_nsr_p5_sca,
        epsilon = s3_nsr_element.threshold,
        max_relative = s3_nsr_element.threshold
    );
    approx::assert_relative_eq!(
        s3_nsr_p11_vec,
        -s3_nsr_p5_vec,
        epsilon = s3_nsr_element.threshold,
        max_relative = s3_nsr_element.threshold
    );

    let s3_nsr_p12 = SymmetryOperation::builder()
        .generating_element(s3_nsr_element.clone())
        .power(12)
        .build()
        .unwrap();
    assert!(!s3_nsr_p12.is_su2_class_1());
    assert!(s3_nsr_p12.is_proper());
    assert!(s3_nsr_p12.is_identity());

    let (s3_nsr_p12_sca, s3_nsr_p12_vec) = s3_nsr_p12.calc_quaternion();
    approx::assert_relative_eq!(
        s3_nsr_p12_sca,
        1.0,
        epsilon = s3_nsr_element.threshold,
        max_relative = s3_nsr_element.threshold
    );
    approx::assert_relative_eq!(
        s3_nsr_p12_vec,
        Vector3::zeros(),
        epsilon = s3_nsr_element.threshold,
        max_relative = s3_nsr_element.threshold
    );

    // --------------
    // Ṡ3(n) = iC3(n)
    // --------------
    let sd3_nsr_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::new(3.0, 1e-14))
        .proper_power(1)
        .raw_axis(Vector3::new(2.0, -1.0, 1.0))
        .kind(INV)
        .rotation_group(RotationGroup::SU2(true))
        .build()
        .unwrap();

    let sd3_nsr_p1 = SymmetryOperation::builder()
        .generating_element(sd3_nsr_element.clone())
        .power(1)
        .build()
        .unwrap();
    assert!(!sd3_nsr_p1.is_su2_class_1());
    assert!(!sd3_nsr_p1.is_proper());

    let (sd3_nsr_p1_sca, sd3_nsr_p1_vec) = sd3_nsr_p1.calc_quaternion();
    approx::assert_relative_eq!(
        sd3_nsr_p1_sca,
        (0.5 * (2.0 * std::f64::consts::PI / 3.0)).cos(),
        epsilon = sd3_nsr_element.threshold,
        max_relative = sd3_nsr_element.threshold
    );
    approx::assert_relative_eq!(
        sd3_nsr_p1_vec,
        (0.5 * (2.0 * std::f64::consts::PI / 3.0)).sin() * Vector3::new(2.0, -1.0, 1.0)
            / (6.0f64.sqrt()),
        epsilon = sd3_nsr_element.threshold,
        max_relative = sd3_nsr_element.threshold
    );

    let sd3_nsr_p2 = SymmetryOperation::builder()
        .generating_element(sd3_nsr_element.clone())
        .power(2)
        .build()
        .unwrap();
    assert!(sd3_nsr_p2.is_su2_class_1());
    assert!(sd3_nsr_p2.is_proper());

    let (sd3_nsr_p2_sca, sd3_nsr_p2_vec) = sd3_nsr_p2.calc_quaternion();
    approx::assert_relative_eq!(
        sd3_nsr_p2_sca,
        -(0.5 * (2.0 * std::f64::consts::PI / 3.0)).cos(),
        epsilon = sd3_nsr_element.threshold,
        max_relative = sd3_nsr_element.threshold
    );
    approx::assert_relative_eq!(
        sd3_nsr_p2_vec,
        (0.5 * (2.0 * std::f64::consts::PI / 3.0)).sin() * Vector3::new(2.0, -1.0, 1.0)
            / (6.0f64.sqrt()),
        epsilon = sd3_nsr_element.threshold,
        max_relative = sd3_nsr_element.threshold
    );

    let sd3_nsr_p3 = SymmetryOperation::builder()
        .generating_element(sd3_nsr_element.clone())
        .power(3)
        .build()
        .unwrap();
    assert!(sd3_nsr_p3.is_su2_class_1());
    assert!(!sd3_nsr_p3.is_proper());
    assert!(sd3_nsr_p3.is_spatial_inversion());

    let (sd3_nsr_p3_sca, sd3_nsr_p3_vec) = sd3_nsr_p3.calc_quaternion();
    approx::assert_relative_eq!(
        sd3_nsr_p3_sca,
        -1.0,
        epsilon = sd3_nsr_element.threshold,
        max_relative = sd3_nsr_element.threshold
    );
    approx::assert_relative_eq!(
        sd3_nsr_p3_vec,
        Vector3::zeros(),
        epsilon = sd3_nsr_element.threshold,
        max_relative = sd3_nsr_element.threshold
    );

    let sd3_nsr_p4 = SymmetryOperation::builder()
        .generating_element(sd3_nsr_element.clone())
        .power(4)
        .build()
        .unwrap();
    assert!(sd3_nsr_p4.is_su2_class_1());
    assert!(sd3_nsr_p4.is_proper());

    let (sd3_nsr_p4_sca, sd3_nsr_p4_vec) = sd3_nsr_p4.calc_quaternion();
    approx::assert_relative_eq!(
        sd3_nsr_p4_sca,
        -sd3_nsr_p1_sca,
        epsilon = sd3_nsr_element.threshold,
        max_relative = sd3_nsr_element.threshold
    );
    approx::assert_relative_eq!(
        sd3_nsr_p4_vec,
        -sd3_nsr_p1_vec,
        epsilon = sd3_nsr_element.threshold,
        max_relative = sd3_nsr_element.threshold
    );

    let sd3_nsr_p5 = SymmetryOperation::builder()
        .generating_element(sd3_nsr_element.clone())
        .power(5)
        .build()
        .unwrap();
    assert!(!sd3_nsr_p5.is_su2_class_1());
    assert!(!sd3_nsr_p5.is_proper());

    let (sd3_nsr_p5_sca, sd3_nsr_p5_vec) = sd3_nsr_p5.calc_quaternion();
    approx::assert_relative_eq!(
        sd3_nsr_p5_sca,
        -sd3_nsr_p2_sca,
        epsilon = sd3_nsr_element.threshold,
        max_relative = sd3_nsr_element.threshold
    );
    approx::assert_relative_eq!(
        sd3_nsr_p5_vec,
        -sd3_nsr_p2_vec,
        epsilon = sd3_nsr_element.threshold,
        max_relative = sd3_nsr_element.threshold
    );

    let sd3_nsr_p6 = SymmetryOperation::builder()
        .generating_element(sd3_nsr_element.clone())
        .power(6)
        .build()
        .unwrap();
    assert!(!sd3_nsr_p6.is_su2_class_1());
    assert!(sd3_nsr_p6.is_identity());

    let (sd3_nsr_p6_sca, sd3_nsr_p6_vec) = sd3_nsr_p6.calc_quaternion();
    approx::assert_relative_eq!(
        sd3_nsr_p6_sca,
        1.0,
        epsilon = sd3_nsr_element.threshold,
        max_relative = sd3_nsr_element.threshold
    );
    approx::assert_relative_eq!(
        sd3_nsr_p6_vec,
        Vector3::zeros(),
        epsilon = sd3_nsr_element.threshold,
        max_relative = sd3_nsr_element.threshold
    );
}

#[test]
fn test_symmetry_operation_su2_from_quaternion() {
    // ==========================
    // Proper symmetry operations
    // ==========================
    let c4_nsr_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(4))
        .proper_power(1)
        .raw_axis(Vector3::new(2.0, -1.0, 1.0))
        .kind(ROT)
        .rotation_group(RotationGroup::SU2(true))
        .build()
        .unwrap();

    let c4_nsr_p2 = SymmetryOperation::builder()
        .generating_element(c4_nsr_element.clone())
        .power(2)
        .build()
        .unwrap();

    let c4_nsr_p2_q = c4_nsr_p2.calc_quaternion();
    let c4_nsr_p2_r = SymmetryOperation::from_quaternion(
        c4_nsr_p2_q,
        c4_nsr_p2.is_proper(),
        c4_nsr_element.threshold,
        10,
        false,
        true,
        None,
    );
    assert_eq!(c4_nsr_p2, c4_nsr_p2_r);
    assert!(!c4_nsr_p2.is_su2_class_1());
    assert!(!c4_nsr_p2_r.is_su2_class_1());

    let c4_nsr_pm2 = SymmetryOperation::builder()
        .generating_element(c4_nsr_element.clone())
        .power(-2)
        .build()
        .unwrap();

    let c4_nsr_pm2_q = c4_nsr_pm2.calc_quaternion();
    let c4_nsr_pm2_r = SymmetryOperation::from_quaternion(
        c4_nsr_pm2_q,
        c4_nsr_pm2.is_proper(),
        c4_nsr_element.threshold,
        10,
        false,
        true,
        None,
    );
    assert_eq!(c4_nsr_pm2, c4_nsr_pm2_r);
    assert!(c4_nsr_pm2.is_su2_class_1());
    assert!(c4_nsr_pm2_r.is_su2_class_1());

    let c4_nsr_p4 = SymmetryOperation::builder()
        .generating_element(c4_nsr_element.clone())
        .power(4)
        .build()
        .unwrap();

    let c4_nsr_p4_q = c4_nsr_p4.calc_quaternion();
    let c4_nsr_p4_r = SymmetryOperation::from_quaternion(
        c4_nsr_p4_q,
        c4_nsr_p4.is_proper(),
        c4_nsr_element.threshold,
        10,
        false,
        true,
        None,
    );
    assert_eq!(c4_nsr_p4, c4_nsr_p4_r);
    assert!(c4_nsr_p4.is_su2_class_1());
    assert!(c4_nsr_p4_r.is_su2_class_1());

    let c4_nsr_p6 = SymmetryOperation::builder()
        .generating_element(c4_nsr_element.clone())
        .power(6)
        .build()
        .unwrap();

    let c4_nsr_p6_q = c4_nsr_p6.calc_quaternion();
    let c4_nsr_p6_r = SymmetryOperation::from_quaternion(
        c4_nsr_p6_q,
        c4_nsr_p6.is_proper(),
        c4_nsr_element.threshold,
        10,
        false,
        true,
        None,
    );
    assert_eq!(c4_nsr_p6, c4_nsr_p6_r);
    assert!(c4_nsr_p6_r.is_su2_class_1());

    let c7_isr_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(7))
        .proper_power(1)
        .raw_axis(Vector3::new(2.0, -2.0, 1.0))
        .kind(ROT)
        .rotation_group(RotationGroup::SU2(false))
        .build()
        .unwrap();

    let c7_isr_p1 = SymmetryOperation::builder()
        .generating_element(c7_isr_element.clone())
        .power(1)
        .build()
        .unwrap();

    let c7_isr_p1_q = c7_isr_p1.calc_quaternion();
    let c7_isr_p1_r = SymmetryOperation::from_quaternion(
        c7_isr_p1_q,
        c7_isr_p1.is_proper(),
        c7_isr_element.threshold,
        10,
        false,
        true,
        None,
    );
    assert_eq!(c7_isr_p1, c7_isr_p1_r);
    assert!(c7_isr_p1_r.is_su2_class_1());

    let c7_isr_p2 = SymmetryOperation::builder()
        .generating_element(c7_isr_element.clone())
        .power(2)
        .build()
        .unwrap();

    let c7_isr_p2_q = c7_isr_p2.calc_quaternion();
    let c7_isr_p2_r = SymmetryOperation::from_quaternion(
        c7_isr_p2_q,
        c7_isr_p2.is_proper(),
        c7_isr_element.threshold,
        10,
        false,
        true,
        None,
    );
    assert_eq!(c7_isr_p2, c7_isr_p2_r);
    assert!(!c7_isr_p2_r.is_su2_class_1());

    let c7_isr_p3 = SymmetryOperation::builder()
        .generating_element(c7_isr_element.clone())
        .power(3)
        .build()
        .unwrap();

    let c7_isr_p3_q = c7_isr_p3.calc_quaternion();
    let c7_isr_p3_r = SymmetryOperation::from_quaternion(
        c7_isr_p3_q,
        c7_isr_p3.is_proper(),
        c7_isr_element.threshold,
        10,
        false,
        true,
        None,
    );
    assert_eq!(c7_isr_p3, c7_isr_p3_r);
    assert!(c7_isr_p3_r.is_su2_class_1());

    let c7_isr_p7 = SymmetryOperation::builder()
        .generating_element(c7_isr_element.clone())
        .power(7)
        .build()
        .unwrap();

    let c7_isr_p7_q = c7_isr_p7.calc_quaternion();
    let c7_isr_p7_r = SymmetryOperation::from_quaternion(
        c7_isr_p7_q,
        c7_isr_p7.is_proper(),
        c7_isr_element.threshold,
        10,
        false,
        true,
        None,
    );
    assert_eq!(c7_isr_p7, c7_isr_p7_r);

    // ============================
    // Improper symmetry operations
    // ============================
    let s1_nsr_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(1))
        .proper_power(1)
        .raw_axis(Vector3::new(2.0, -1.0, 2.0))
        .kind(SIG)
        .rotation_group(RotationGroup::SU2(true))
        .build()
        .unwrap();

    let s1_nsr_p1 = SymmetryOperation::builder()
        .generating_element(s1_nsr_element.clone())
        .power(1)
        .build()
        .unwrap();

    let s1_nsr_p1_q = s1_nsr_p1.calc_quaternion();
    let s1_nsr_p1_r = SymmetryOperation::from_quaternion(
        s1_nsr_p1_q,
        s1_nsr_p1.is_proper(),
        s1_nsr_element.threshold,
        10,
        false,
        true,
        None,
    );
    assert_eq!(s1_nsr_p1, s1_nsr_p1_r);

    let s1_nsr_pm4 = SymmetryOperation::builder()
        .generating_element(s1_nsr_element.clone())
        .power(-4)
        .build()
        .unwrap();

    let s1_nsr_pm4_q = s1_nsr_pm4.calc_quaternion();
    let s1_nsr_pm4_r = SymmetryOperation::from_quaternion(
        s1_nsr_pm4_q,
        s1_nsr_pm4.is_proper(),
        s1_nsr_element.threshold,
        10,
        false,
        true,
        None,
    );
    assert_eq!(s1_nsr_pm4, s1_nsr_pm4_r);

    let s2_isr_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(2))
        .proper_power(1)
        .raw_axis(Vector3::new(2.0, -1.0, 2.0))
        .kind(SIG)
        .rotation_group(RotationGroup::SU2(false))
        .build()
        .unwrap();

    let s2_isr_p1 = SymmetryOperation::builder()
        .generating_element(s2_isr_element.clone())
        .power(1)
        .build()
        .unwrap();

    let s2_isr_p1_q = s2_isr_p1.calc_quaternion();
    let s2_isr_p1_r = SymmetryOperation::from_quaternion(
        s2_isr_p1_q,
        s2_isr_p1.is_proper(),
        s2_isr_element.threshold,
        10,
        false,
        true,
        None,
    );
    assert_eq!(s2_isr_p1, s2_isr_p1_r);

    let s2_isr_p2 = SymmetryOperation::builder()
        .generating_element(s2_isr_element.clone())
        .power(2)
        .build()
        .unwrap();

    let s2_isr_p2_q = s2_isr_p2.calc_quaternion();
    let s2_isr_p2_r = SymmetryOperation::from_quaternion(
        s2_isr_p2_q,
        s2_isr_p2.is_proper(),
        s2_isr_element.threshold,
        10,
        false,
        true,
        None,
    );
    assert_eq!(s2_isr_p2, s2_isr_p2_r);
    assert!(!s2_isr_p2_r.is_su2_class_1());

    let s3_isr_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::new(3.0, 1e-14))
        .proper_power(1)
        .raw_axis(Vector3::new(2.0, -1.0, 1.0))
        .kind(SIG)
        .rotation_group(RotationGroup::SU2(false))
        .build()
        .unwrap();

    let s3_isr_p1 = SymmetryOperation::builder()
        .generating_element(s3_isr_element.clone())
        .power(1)
        .build()
        .unwrap();

    let s3_isr_p1_q = s3_isr_p1.calc_quaternion();
    let s3_isr_p1_r = SymmetryOperation::from_quaternion(
        s3_isr_p1_q,
        s3_isr_p1.is_proper(),
        s3_isr_element.threshold,
        10,
        false,
        true,
        None,
    );
    assert_eq!(s3_isr_p1, s3_isr_p1_r);

    let s3_isr_p2 = SymmetryOperation::builder()
        .generating_element(s3_isr_element.clone())
        .power(2)
        .build()
        .unwrap();

    let s3_isr_p2_q = s3_isr_p2.calc_quaternion();
    let s3_isr_p2_r = SymmetryOperation::from_quaternion(
        s3_isr_p2_q,
        s3_isr_p2.is_proper(),
        s3_isr_element.threshold,
        10,
        false,
        true,
        None,
    );
    assert_eq!(s3_isr_p2, s3_isr_p2_r);

    let s3_isr_pm1 = SymmetryOperation::builder()
        .generating_element(s3_isr_element.clone())
        .power(-1)
        .build()
        .unwrap();

    let s3_isr_pm1_q = s3_isr_pm1.calc_quaternion();
    let s3_isr_pm1_r = SymmetryOperation::from_quaternion(
        s3_isr_pm1_q,
        s3_isr_pm1.is_proper(),
        s3_isr_element.threshold,
        10,
        false,
        true,
        None,
    );
    assert_eq!(s3_isr_pm1, s3_isr_pm1_r);

    let s3_isr_p3 = SymmetryOperation::builder()
        .generating_element(s3_isr_element.clone())
        .power(3)
        .build()
        .unwrap();

    let s3_isr_p3_q = s3_isr_p3.calc_quaternion();
    let s3_isr_p3_r = SymmetryOperation::from_quaternion(
        s3_isr_p3_q,
        s3_isr_p3.is_proper(),
        s3_isr_element.threshold,
        10,
        false,
        true,
        None,
    );
    assert_eq!(s3_isr_p3, s3_isr_p3_r);

    let s17_nsr_pp3_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(17))
        .proper_power(3)
        .raw_axis(Vector3::new(5.0, -1.0, 2.0))
        .kind(SIG)
        .rotation_group(RotationGroup::SU2(true))
        .build()
        .unwrap();

    let s17_nsr_pp3 = SymmetryOperation::builder()
        .generating_element(s17_nsr_pp3_element.clone())
        .power(1)
        .build()
        .unwrap();

    let s17_nsr_pp3_q = s17_nsr_pp3.calc_quaternion();
    let s17_nsr_pp3_r = SymmetryOperation::from_quaternion(
        s17_nsr_pp3_q,
        s17_nsr_pp3.is_proper(),
        s17_nsr_pp3_element.threshold,
        17,
        false,
        true,
        None,
    );
    assert_eq!(s17_nsr_pp3, s17_nsr_pp3_r);

    let sd11_isr_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(11))
        .proper_power(1)
        .raw_axis(Vector3::new(2.0, -1.0, 1.0))
        .kind(INV)
        .rotation_group(RotationGroup::SU2(false))
        .build()
        .unwrap();

    let sd11_isr_p1 = SymmetryOperation::builder()
        .generating_element(sd11_isr_element.clone())
        .power(1)
        .build()
        .unwrap();

    let sd11_isr_p1q = sd11_isr_p1.calc_quaternion();
    let sd11_isr_p1r = SymmetryOperation::from_quaternion(
        sd11_isr_p1q,
        sd11_isr_p1.is_proper(),
        sd11_isr_element.threshold,
        11,
        false,
        true,
        None,
    );
    assert_eq!(sd11_isr_p1, sd11_isr_p1r);

    let sd11_isr_p6 = SymmetryOperation::builder()
        .generating_element(sd11_isr_element.clone())
        .power(6)
        .build()
        .unwrap();

    let sd11_isr_p6_q = sd11_isr_p6.calc_quaternion();
    let sd11_isr_p6_r = SymmetryOperation::from_quaternion(
        sd11_isr_p6_q,
        sd11_isr_p6.is_proper(),
        sd11_isr_element.threshold,
        11,
        false,
        true,
        None,
    );
    assert_eq!(sd11_isr_p6, sd11_isr_p6_r);
}

#[test]
fn test_symmetry_operation_coaxial_composition() {
    // ==========================
    // Proper symmetry operations
    // ==========================
    let c5_element = SymmetryElement::builder()
        .threshold(1e-12)
        .proper_order(ElementOrder::Int(5))
        .proper_power(1)
        .raw_axis(Vector3::new(2.0, -1.0, 1.0))
        .kind(ROT)
        .rotation_group(RotationGroup::SO3)
        .build()
        .unwrap();

    let c5 = SymmetryOperation::builder()
        .generating_element(c5_element.clone())
        .power(1)
        .build()
        .unwrap();

    let c5pm1 = SymmetryOperation::builder()
        .generating_element(c5_element.clone())
        .power(-1)
        .build()
        .unwrap();
    assert!((&c5 * &c5pm1).is_identity());

    let c5p2 = SymmetryOperation::builder()
        .generating_element(c5_element.clone())
        .power(2)
        .build()
        .unwrap();
    assert_eq!(&c5 * &c5, c5p2);
    assert_eq!(&c5p2 * &c5pm1, c5);
    assert_eq!(&c5pm1 * &c5p2, c5);

    let c5p3 = SymmetryOperation::builder()
        .generating_element(c5_element)
        .power(3)
        .build()
        .unwrap();
    assert_eq!(&c5pm1 * &c5pm1, c5p3);
    assert!((&c5p3 * &c5p2).is_identity());
    assert_eq!(&c5p3 * &c5p3, c5);

    let c7_element = SymmetryElement::builder()
        .threshold(1e-12)
        .proper_order(ElementOrder::Int(7))
        .proper_power(1)
        .raw_axis(Vector3::new(2.0, -1.0, 1.0))
        .kind(ROT)
        .rotation_group(RotationGroup::SO3)
        .build()
        .unwrap();

    let c7 = SymmetryOperation::builder()
        .generating_element(c7_element.clone())
        .power(1)
        .build()
        .unwrap();

    let c35_element = SymmetryElement::builder()
        .threshold(1e-12)
        .proper_order(ElementOrder::Int(35))
        .proper_power(1)
        .raw_axis(Vector3::new(2.0, -1.0, 1.0))
        .kind(ROT)
        .rotation_group(RotationGroup::SO3)
        .build()
        .unwrap();

    let c35p12 = SymmetryOperation::builder()
        .generating_element(c35_element.clone())
        .power(12)
        .build()
        .unwrap();
    assert_eq!(&c5 * &c7, c35p12);

    let c7pm1 = SymmetryOperation::builder()
        .generating_element(c7_element)
        .power(-1)
        .build()
        .unwrap();

    let c35p2 = SymmetryOperation::builder()
        .generating_element(c35_element)
        .power(2)
        .build()
        .unwrap();
    assert_eq!(&c5 * &c7pm1, c35p2);

    let c10_element = SymmetryElement::builder()
        .threshold(1e-12)
        .proper_order(ElementOrder::Int(10))
        .proper_power(1)
        .raw_axis(Vector3::new(2.0, -1.0, 1.0))
        .kind(ROT)
        .rotation_group(RotationGroup::SO3)
        .build()
        .unwrap();

    let c10 = SymmetryOperation::builder()
        .generating_element(c10_element.clone())
        .power(1)
        .build()
        .unwrap();

    let c10p3 = SymmetryOperation::builder()
        .generating_element(c10_element)
        .power(3)
        .build()
        .unwrap();
    assert_eq!(&c5 * &c10, c10p3);

    // ============================
    // Improper symmetry operations
    // ============================
    let s5_element = SymmetryElement::builder()
        .threshold(1e-12)
        .proper_order(ElementOrder::Int(5))
        .proper_power(1)
        .raw_axis(Vector3::new(2.0, -1.0, 1.0))
        .kind(SIG)
        .rotation_group(RotationGroup::SO3)
        .build()
        .unwrap();

    let s5 = SymmetryOperation::builder()
        .generating_element(s5_element.clone())
        .power(1)
        .build()
        .unwrap();
    assert_eq!(&s5 * &s5, c5p2);

    let s5pp2_element = SymmetryElement::builder()
        .threshold(1e-12)
        .proper_order(ElementOrder::Int(5))
        .proper_power(2)
        .raw_axis(Vector3::new(2.0, -1.0, 1.0))
        .kind(SIG)
        .rotation_group(RotationGroup::SO3)
        .build()
        .unwrap();

    let s5pp2 = SymmetryOperation::builder()
        .generating_element(s5pp2_element)
        .power(1)
        .build()
        .unwrap();
    assert_eq!(&s5 * &c5, s5pp2);
    assert_eq!(&s5pp2 * &s5pp2, c5pm1);

    let s5p3 = SymmetryOperation::builder()
        .generating_element(s5_element.clone())
        .power(3)
        .build()
        .unwrap();
    assert_eq!(&s5 * &s5, c5p2);
    assert_eq!(&s5pp2 * &c5, s5p3);

    let s5p5 = SymmetryOperation::builder()
        .generating_element(s5_element)
        .power(5)
        .build()
        .unwrap();
    assert_eq!(&(&s5pp2 * &s5pp2) * &s5, s5p5);

    let s8_element = SymmetryElement::builder()
        .threshold(1e-12)
        .proper_order(ElementOrder::Int(8))
        .proper_power(1)
        .raw_axis(Vector3::new(2.0, -1.0, 1.0))
        .kind(SIG)
        .rotation_group(RotationGroup::SO3)
        .build()
        .unwrap();

    let s8 = SymmetryOperation::builder()
        .generating_element(s8_element)
        .power(1)
        .build()
        .unwrap();

    let c40_element = SymmetryElement::builder()
        .threshold(1e-12)
        .proper_order(ElementOrder::Int(40))
        .proper_power(1)
        .raw_axis(Vector3::new(2.0, -1.0, 1.0))
        .kind(ROT)
        .rotation_group(RotationGroup::SO3)
        .build()
        .unwrap();

    let c40p13 = SymmetryOperation::builder()
        .generating_element(c40_element)
        .power(13)
        .build()
        .unwrap();
    assert_eq!(&s5 * &s8, c40p13);

    let s40_element = SymmetryElement::builder()
        .threshold(1e-12)
        .proper_order(ElementOrder::Int(40))
        .proper_power(1)
        .raw_axis(Vector3::new(2.0, -1.0, 1.0))
        .kind(SIG)
        .rotation_group(RotationGroup::SO3)
        .build()
        .unwrap();

    let s40p21 = SymmetryOperation::builder()
        .generating_element(s40_element)
        .power(21)
        .build()
        .unwrap();
    assert_eq!(&c40p13 * &s5, s40p21);

    let s1_element = SymmetryElement::builder()
        .threshold(1e-12)
        .proper_order(ElementOrder::Int(1))
        .proper_power(1)
        .raw_axis(Vector3::new(2.0, -1.0, 1.0))
        .kind(SIG)
        .rotation_group(RotationGroup::SO3)
        .build()
        .unwrap();

    let s1 = SymmetryOperation::builder()
        .generating_element(s1_element)
        .power(1)
        .build()
        .unwrap();
    assert_eq!(&s1 * &c5, s5);

    let s2_element = SymmetryElement::builder()
        .threshold(1e-12)
        .proper_order(ElementOrder::Int(2))
        .proper_power(1)
        .raw_axis(Vector3::new(2.0, -1.0, 1.0))
        .kind(SIG)
        .rotation_group(RotationGroup::SO3)
        .build()
        .unwrap();

    let s2 = SymmetryOperation::builder()
        .generating_element(s2_element)
        .power(1)
        .build()
        .unwrap();

    let sd5_element = SymmetryElement::builder()
        .threshold(1e-12)
        .proper_order(ElementOrder::Int(5))
        .proper_power(1)
        .raw_axis(Vector3::new(2.0, -1.0, 1.0))
        .kind(INV)
        .rotation_group(RotationGroup::SO3)
        .build()
        .unwrap();

    let sd5 = SymmetryOperation::builder()
        .generating_element(sd5_element)
        .power(1)
        .build()
        .unwrap();
    assert_eq!(&s2 * &c5, sd5);
}

#[test]
fn test_symmetry_operation_noncoaxial_composition() {
    // ==========================
    // Proper symmetry operations
    // ==========================
    let c2x_element = SymmetryElement::builder()
        .threshold(1e-12)
        .proper_order(ElementOrder::Int(2))
        .proper_power(1)
        .raw_axis(Vector3::new(1.0, 0.0, 0.0))
        .kind(ROT)
        .rotation_group(RotationGroup::SO3)
        .build()
        .unwrap();

    let c2x = SymmetryOperation::builder()
        .generating_element(c2x_element)
        .power(1)
        .build()
        .unwrap();

    let c2y_element = SymmetryElement::builder()
        .threshold(1e-12)
        .proper_order(ElementOrder::Int(2))
        .proper_power(1)
        .raw_axis(Vector3::new(0.0, 1.0, 0.0))
        .kind(ROT)
        .rotation_group(RotationGroup::SO3)
        .build()
        .unwrap();

    let c2y = SymmetryOperation::builder()
        .generating_element(c2y_element)
        .power(1)
        .build()
        .unwrap();

    let c2z_element = SymmetryElement::builder()
        .threshold(1e-12)
        .proper_order(ElementOrder::Int(2))
        .proper_power(1)
        .raw_axis(Vector3::new(0.0, 0.0, 1.0))
        .kind(ROT)
        .rotation_group(RotationGroup::SO3)
        .build()
        .unwrap();

    let c2z = SymmetryOperation::builder()
        .generating_element(c2z_element)
        .power(1)
        .build()
        .unwrap();
    assert_eq!(&c2x * &c2y, c2z);
    assert_eq!(&c2x * &c2z, c2y);
    assert_eq!(&c2y * &c2z, c2x);

    let c3xyz_element = SymmetryElement::builder()
        .threshold(1e-12)
        .proper_order(ElementOrder::Int(3))
        .proper_power(1)
        .raw_axis(Vector3::new(1.0, 1.0, 1.0))
        .kind(ROT)
        .rotation_group(RotationGroup::SO3)
        .build()
        .unwrap();

    let c3xyz = SymmetryOperation::builder()
        .generating_element(c3xyz_element)
        .power(1)
        .build()
        .unwrap();

    let c2xz_element = SymmetryElement::builder()
        .threshold(1e-12)
        .proper_order(ElementOrder::Int(2))
        .proper_power(1)
        .raw_axis(Vector3::new(1.0, 0.0, 1.0))
        .kind(ROT)
        .rotation_group(RotationGroup::SO3)
        .build()
        .unwrap();

    let c2xz = SymmetryOperation::builder()
        .generating_element(c2xz_element)
        .power(1)
        .build()
        .unwrap();

    let c4z_element = SymmetryElement::builder()
        .threshold(1e-12)
        .proper_order(ElementOrder::Int(4))
        .proper_power(1)
        .raw_axis(Vector3::new(0.0, 0.0, 1.0))
        .kind(ROT)
        .rotation_group(RotationGroup::SO3)
        .build()
        .unwrap();

    let c4zpm1 = SymmetryOperation::builder()
        .generating_element(c4z_element)
        .power(-1)
        .build()
        .unwrap();
    assert_eq!(&c2xz * &c3xyz, c4zpm1);

    // ============================
    // Improper symmetry operations
    // ============================
    let s1x_element = SymmetryElement::builder()
        .threshold(1e-12)
        .proper_order(ElementOrder::Int(1))
        .proper_power(1)
        .raw_axis(Vector3::new(1.0, 0.0, 0.0))
        .kind(SIG)
        .rotation_group(RotationGroup::SO3)
        .build()
        .unwrap();

    let s1x = SymmetryOperation::builder()
        .generating_element(s1x_element)
        .power(1)
        .build()
        .unwrap();

    let s1y_element = SymmetryElement::builder()
        .threshold(1e-12)
        .proper_order(ElementOrder::Int(1))
        .proper_power(1)
        .raw_axis(Vector3::new(0.0, 1.0, 0.0))
        .kind(SIG)
        .rotation_group(RotationGroup::SO3)
        .build()
        .unwrap();

    let s1y = SymmetryOperation::builder()
        .generating_element(s1y_element)
        .power(1)
        .build()
        .unwrap();

    let s1z_element = SymmetryElement::builder()
        .threshold(1e-12)
        .proper_order(ElementOrder::Int(1))
        .proper_power(1)
        .raw_axis(Vector3::new(0.0, 0.0, 1.0))
        .kind(SIG)
        .rotation_group(RotationGroup::SO3)
        .build()
        .unwrap();

    let s1z = SymmetryOperation::builder()
        .generating_element(s1z_element)
        .power(1)
        .build()
        .unwrap();

    assert_eq!(&s1x * &s1y, c2z);
    assert_eq!(&s1x * &s1z, c2y);
    assert_eq!(&s1y * &s1z, c2x);

    let s1a_element = SymmetryElement::builder()
        .threshold(1e-10)
        .proper_order(ElementOrder::Int(1))
        .proper_power(1)
        .raw_axis(Vector3::new(1.0, 0.0, 0.0))
        .kind(SIG)
        .rotation_group(RotationGroup::SO3)
        .build()
        .unwrap();

    let s1a = SymmetryOperation::builder()
        .generating_element(s1a_element.clone())
        .power(1)
        .build()
        .unwrap();

    let s1b_element = SymmetryElement::builder()
        .threshold(1e-10)
        .proper_order(ElementOrder::Int(1))
        .proper_power(1)
        .raw_axis(Vector3::new(
            (2.0 * std::f64::consts::PI * 3.0 / 17.0).cos(),
            (2.0 * std::f64::consts::PI * 3.0 / 17.0).sin(),
            0.0,
        ))
        .kind(SIG)
        .rotation_group(RotationGroup::SO3)
        .build()
        .unwrap();

    let s1b = SymmetryOperation::builder()
        .generating_element(s1b_element.clone())
        .power(1)
        .build()
        .unwrap();

    let kaleidoscope_angle = s1a_element.raw_axis.dot(&s1b_element.raw_axis).acos();
    let rotation_axis = s1a_element.raw_axis.cross(&s1b_element.raw_axis);
    let fract =
        geometry::get_proper_fraction(2.0 * kaleidoscope_angle, 1e-10, 20).unwrap_or_else(|| {
            panic!(
                "Unable to obtain a proper fraction for angle {}.",
                2.0 * kaleidoscope_angle
            )
        });
    let c_element = SymmetryElement::builder()
        .threshold(1e-12)
        .proper_order(ElementOrder::Int(*fract.denom().unwrap()))
        .proper_power((*fract.numer().unwrap()).try_into().unwrap())
        .raw_axis(rotation_axis)
        .kind(ROT)
        .rotation_group(RotationGroup::SO3)
        .build()
        .unwrap();

    let c = SymmetryOperation::builder()
        .generating_element(c_element)
        .power(1)
        .build()
        .unwrap();
    assert_eq!(&s1b * &s1a, c);

    let s1c_element = SymmetryElement::builder()
        .threshold(1e-10)
        .proper_order(ElementOrder::Int(1))
        .proper_power(1)
        .raw_axis(Vector3::new(
            (std::f64::consts::PI * 3.0).sin(),
            -(std::f64::consts::PI * 3.0).cos(),
            0.0,
        ))
        .kind(SIG)
        .rotation_group(RotationGroup::SO3)
        .build()
        .unwrap();

    let s1c = SymmetryOperation::builder()
        .generating_element(s1c_element)
        .power(1)
        .build()
        .unwrap();

    let c1_element = SymmetryElement::builder()
        .threshold(1e-10)
        .proper_order(ElementOrder::Int(1))
        .proper_power(1)
        .raw_axis(Vector3::new(0.0, 0.0, 1.0))
        .kind(ROT)
        .rotation_group(RotationGroup::SO3)
        .build()
        .unwrap();

    let c1 = SymmetryOperation::builder()
        .generating_element(c1_element)
        .power(1)
        .build()
        .unwrap();

    let s1cc1 = &s1c * &c1;

    let s2d_element = SymmetryElement::builder()
        .threshold(1e-10)
        .proper_order(ElementOrder::Int(2))
        .proper_power(1)
        .raw_axis(Vector3::new(
            (std::f64::consts::PI * 3.0).sin(),
            -(std::f64::consts::PI * 3.0).cos(),
            0.0,
        ))
        .kind(INV)
        .rotation_group(RotationGroup::SO3)
        .build()
        .unwrap();

    let s2d = SymmetryOperation::builder()
        .generating_element(s2d_element)
        .power(1)
        .build()
        .unwrap();
    assert_eq!(s1cc1, s2d);
}

#[test]
fn test_symmetry_operation_su2_coaxial_composition() {
    // ==========================
    // Proper symmetry operations
    // ==========================
    let c2_nsr_element = SymmetryElement::builder()
        .threshold(1e-12)
        .proper_order(ElementOrder::Int(2))
        .proper_power(1)
        .raw_axis(Vector3::x())
        .kind(ROT)
        .rotation_group(RotationGroup::SU2(true))
        .build()
        .unwrap();

    let c2_nsr_p1 = SymmetryOperation::builder()
        .generating_element(c2_nsr_element.clone())
        .power(1)
        .build()
        .unwrap();

    let c2_nsr_p2 = SymmetryOperation::builder()
        .generating_element(c2_nsr_element.clone())
        .power(2)
        .build()
        .unwrap();

    assert!((&c2_nsr_p1 * &c2_nsr_p1).is_su2_class_1());
    assert_eq!(&c2_nsr_p1 * &c2_nsr_p1, c2_nsr_p2);

    assert!(!(&c2_nsr_p2 * &c2_nsr_p2).is_su2_class_1());
    assert!((&c2_nsr_p2 * &c2_nsr_p2).is_identity());

    let c2_isr_element = SymmetryElement::builder()
        .threshold(1e-12)
        .proper_order(ElementOrder::Int(2))
        .proper_power(1)
        .raw_axis(Vector3::x())
        .kind(ROT)
        .rotation_group(RotationGroup::SU2(false))
        .build()
        .unwrap();

    let c2_isr_p1 = SymmetryOperation::builder()
        .generating_element(c2_isr_element)
        .power(1)
        .build()
        .unwrap();
    assert!(!(&c2_nsr_p1 * &c2_isr_p1).is_su2_class_1());
    assert!((&c2_nsr_p1 * &c2_isr_p1).is_identity());

    let c3_nsr_element = SymmetryElement::builder()
        .threshold(1e-12)
        .proper_order(ElementOrder::Int(3))
        .proper_power(1)
        .raw_axis(Vector3::z())
        .kind(ROT)
        .rotation_group(RotationGroup::SU2(true))
        .build()
        .unwrap();

    let c3_nsr_p1 = SymmetryOperation::builder()
        .generating_element(c3_nsr_element.clone())
        .power(1)
        .build()
        .unwrap();

    let c3_nsr_p2 = SymmetryOperation::builder()
        .generating_element(c3_nsr_element.clone())
        .power(2)
        .build()
        .unwrap();

    let c3_nsr_p4 = SymmetryOperation::builder()
        .generating_element(c3_nsr_element)
        .power(4)
        .build()
        .unwrap();

    assert!((&c3_nsr_p1 * &c3_nsr_p2).is_su2_class_1());
    assert!((&c3_nsr_p1 * &c3_nsr_p2).is_spatial_identity());

    assert!((&c3_nsr_p2 * &c3_nsr_p2).is_su2_class_1());
    assert_eq!(&c3_nsr_p2 * &c3_nsr_p2, c3_nsr_p4);

    assert!(!(&c3_nsr_p4 * &c3_nsr_p1).is_su2_class_1());
    assert!((&c3_nsr_p4 * &c3_nsr_p2).is_identity());

    assert!((&c3_nsr_p4 * &c3_nsr_p2).is_identity());

    let c4_nsr_element = SymmetryElement::builder()
        .threshold(1e-12)
        .proper_order(ElementOrder::Int(4))
        .proper_power(1)
        .raw_axis(Vector3::x())
        .kind(ROT)
        .rotation_group(RotationGroup::SU2(true))
        .build()
        .unwrap();

    let c4_nsr_p1 = SymmetryOperation::builder()
        .generating_element(c4_nsr_element.clone())
        .power(1)
        .build()
        .unwrap();
    let c4_nsr_pm1 = SymmetryOperation::builder()
        .generating_element(c4_nsr_element)
        .power(-1)
        .build()
        .unwrap();
    let c2_nsr_p3 = SymmetryOperation::builder()
        .generating_element(c2_nsr_element)
        .power(3)
        .build()
        .unwrap();
    let c4_nsr_p2 = c4_nsr_p1.pow(2);
    let c4_nsr_pm2 = c4_nsr_pm1.pow(2);
    assert_eq!(c4_nsr_p2, c2_nsr_p1);
    assert_eq!(c4_nsr_pm2, c2_nsr_p3);

    let c4b_nsr_element = SymmetryElement::builder()
        .threshold(1e-12)
        .proper_order(ElementOrder::Int(4))
        .proper_power(1)
        .raw_axis(-Vector3::x())
        .kind(ROT)
        .rotation_group(RotationGroup::SU2(true))
        .build()
        .unwrap();

    let c4b_nsr_p1 = SymmetryOperation::builder()
        .generating_element(c4b_nsr_element)
        .power(1)
        .build()
        .unwrap();
    let c4b_nsr_p2 = c4b_nsr_p1.pow(2);
    assert!(c4b_nsr_p2.is_su2_class_1());
    assert_eq!(c4b_nsr_p2, c2_nsr_p3);

    let c5_nsr_element = SymmetryElement::builder()
        .threshold(1e-12)
        .proper_order(ElementOrder::Int(5))
        .proper_power(1)
        .raw_axis(Vector3::new(2.0, -1.0, 1.0))
        .kind(ROT)
        .rotation_group(RotationGroup::SU2(true))
        .build()
        .unwrap();

    let c5_nsr_p1 = SymmetryOperation::builder()
        .generating_element(c5_nsr_element.clone())
        .power(1)
        .build()
        .unwrap();
    let c5_nsr_pm1 = SymmetryOperation::builder()
        .generating_element(c5_nsr_element.clone())
        .power(-1)
        .build()
        .unwrap();
    let c5_nsr_p9 = SymmetryOperation::builder()
        .generating_element(c5_nsr_element.clone())
        .power(9)
        .build()
        .unwrap();
    assert_eq!(c5_nsr_pm1, c5_nsr_p9);
    assert!((&c5_nsr_p1 * &c5_nsr_pm1).is_identity());
    assert!((&c5_nsr_p1 * &c5_nsr_p9).is_identity());

    let c5_nsr_p2 = SymmetryOperation::builder()
        .generating_element(c5_nsr_element.clone())
        .power(2)
        .build()
        .unwrap();
    let c5_nsr_p3 = SymmetryOperation::builder()
        .generating_element(c5_nsr_element.clone())
        .power(3)
        .build()
        .unwrap();
    assert!((&c5_nsr_p2 * &c5_nsr_p3).is_su2_class_1());
    assert!((&c5_nsr_p2 * &c5_nsr_p3).is_spatial_identity());

    let c5_isr_element = SymmetryElement::builder()
        .threshold(1e-12)
        .proper_order(ElementOrder::Int(5))
        .proper_power(1)
        .raw_axis(Vector3::new(2.0, -1.0, 1.0))
        .kind(ROT)
        .rotation_group(RotationGroup::SU2(false))
        .build()
        .unwrap();

    let c5_isr_p1 = SymmetryOperation::builder()
        .generating_element(c5_isr_element)
        .power(1)
        .build()
        .unwrap();
    let c5_nsr_p4 = SymmetryOperation::builder()
        .generating_element(c5_nsr_element.clone())
        .power(4)
        .build()
        .unwrap();

    assert!((&c5_nsr_p2 * &c5_nsr_p4).is_su2_class_1());
    assert_eq!(&c5_nsr_p2 * &c5_nsr_p4, c5_isr_p1);

    let c7_isr_element = SymmetryElement::builder()
        .threshold(1e-12)
        .proper_order(ElementOrder::Int(7))
        .proper_power(1)
        .raw_axis(Vector3::new(2.0, -1.0, 1.0))
        .kind(ROT)
        .rotation_group(RotationGroup::SU2(false))
        .build()
        .unwrap();

    let c7_isr_p1 = SymmetryOperation::builder()
        .generating_element(c7_isr_element.clone())
        .power(1)
        .build()
        .unwrap();
    let c35_pp12_isr_element = SymmetryElement::builder()
        .threshold(1e-12)
        .proper_order(ElementOrder::Int(35))
        .proper_power(12)
        .raw_axis(Vector3::new(2.0, -1.0, 1.0))
        .kind(ROT)
        .rotation_group(RotationGroup::SU2(false))
        .build()
        .unwrap();
    let c35_pp12_isr_p1 = SymmetryOperation::builder()
        .generating_element(c35_pp12_isr_element)
        .power(1)
        .build()
        .unwrap();
    assert_eq!(&c5_nsr_p1 * &c7_isr_p1, c35_pp12_isr_p1);

    let c7_isr_pm1 = SymmetryOperation::builder()
        .generating_element(c7_isr_element)
        .power(-1)
        .build()
        .unwrap();
    let c35_pp2_isr_element = SymmetryElement::builder()
        .threshold(1e-12)
        .proper_order(ElementOrder::Int(35))
        .proper_power(2)
        .raw_axis(Vector3::new(2.0, -1.0, 1.0))
        .kind(ROT)
        .rotation_group(RotationGroup::SU2(false))
        .build()
        .unwrap();
    let c35_pp2_isr_p1 = SymmetryOperation::builder()
        .generating_element(c35_pp2_isr_element)
        .power(1)
        .build()
        .unwrap();
    assert_eq!(&c5_nsr_p1 * &c7_isr_pm1, c35_pp2_isr_p1);

    let c10_nsr_element = SymmetryElement::builder()
        .threshold(1e-12)
        .proper_order(ElementOrder::Int(10))
        .proper_power(1)
        .raw_axis(Vector3::new(2.0, -1.0, 1.0))
        .kind(ROT)
        .rotation_group(RotationGroup::SU2(true))
        .build()
        .unwrap();

    let c10_nsr_p4 = SymmetryOperation::builder()
        .generating_element(c10_nsr_element.clone())
        .power(4)
        .build()
        .unwrap();

    let c10_nsr_p6 = SymmetryOperation::builder()
        .generating_element(c10_nsr_element)
        .power(6)
        .build()
        .unwrap();
    assert_eq!(&c5_nsr_p1 * &c10_nsr_p4, c10_nsr_p6);

    // ============================
    // Improper symmetry operations
    // ============================
    let s5_nsr_element = SymmetryElement::builder()
        .threshold(1e-12)
        .proper_order(ElementOrder::Int(5))
        .proper_power(1)
        .raw_axis(Vector3::new(2.0, -1.0, 1.0))
        .kind(SIG)
        .rotation_group(RotationGroup::SU2(true))
        .build()
        .unwrap();

    let s5_nsr_p1 = SymmetryOperation::builder()
        .generating_element(s5_nsr_element)
        .power(1)
        .build()
        .unwrap();
    let c5_nsr_p7 = SymmetryOperation::builder()
        .generating_element(c5_nsr_element)
        .power(7)
        .build()
        .unwrap();
    assert!(!s5_nsr_p1.is_su2_class_1());
    assert!((&s5_nsr_p1 * &s5_nsr_p1).is_su2_class_1());
    assert_eq!(&s5_nsr_p1 * &s5_nsr_p1, c5_nsr_p7);

    let s5_pp2_nsr_element = SymmetryElement::builder()
        .threshold(1e-12)
        .proper_order(ElementOrder::Int(5))
        .proper_power(2)
        .raw_axis(Vector3::new(2.0, -1.0, 1.0))
        .kind(SIG)
        .rotation_group(RotationGroup::SU2(true))
        .build()
        .unwrap();

    let s5_pp2_nsr_p1 = SymmetryOperation::builder()
        .generating_element(s5_pp2_nsr_element)
        .power(1)
        .build()
        .unwrap();
    assert_eq!(&s5_nsr_p1 * &c5_nsr_p1, s5_pp2_nsr_p1);

    // =======================================
    // Proper and improper symmetry operations
    // =======================================
    let c2z_nsr_element = SymmetryElement::builder()
        .threshold(1e-12)
        .proper_order(ElementOrder::Int(2))
        .proper_power(1)
        .raw_axis(Vector3::z())
        .kind(ROT)
        .rotation_group(RotationGroup::SU2(true))
        .build()
        .unwrap();
    let c2z_nsr_p1 = SymmetryOperation::builder()
        .generating_element(c2z_nsr_element)
        .power(1)
        .build()
        .unwrap();

    let s4z_nsr_element = SymmetryElement::builder()
        .threshold(1e-12)
        .proper_order(ElementOrder::Int(4))
        .proper_power(1)
        .raw_axis(Vector3::z())
        .kind(SIG)
        .rotation_group(RotationGroup::SU2(true))
        .build()
        .unwrap();
    let s4z_nsr_p1 = SymmetryOperation::builder()
        .generating_element(s4z_nsr_element.clone())
        .power(1)
        .build()
        .unwrap();
    let s4z_nsr_pm1 = SymmetryOperation::builder()
        .generating_element(s4z_nsr_element)
        .power(-1)
        .build()
        .unwrap();
    assert_eq!(&c2z_nsr_p1 * &s4z_nsr_p1, s4z_nsr_pm1);
}

#[test]
fn test_symmetry_operation_su2_noncoaxial_composition() {
    // ----------------------------------------------------------------------------------------
    // D2*
    //
    // Reference: Table 8-6.2, Altmann, S. L. Rotations, Quaternions, and Double Groups. (Dover
    // Publications, Inc., 2005).
    // ----------------------------------------------------------------------------------------
    let c2x_nsr_element = SymmetryElement::builder()
        .threshold(1e-12)
        .proper_order(ElementOrder::Int(2))
        .proper_power(1)
        .raw_axis(Vector3::x())
        .kind(ROT)
        .rotation_group(RotationGroup::SU2(true))
        .build()
        .unwrap();
    let e_nsr = SymmetryOperation::builder()
        .generating_element(c2x_nsr_element.clone())
        .power(0)
        .build()
        .unwrap();
    let e_isr = SymmetryOperation::builder()
        .generating_element(c2x_nsr_element.clone())
        .power(2)
        .build()
        .unwrap();
    let c2x_nsr = SymmetryOperation::builder()
        .generating_element(c2x_nsr_element.clone())
        .power(1)
        .build()
        .unwrap();
    let c2x_isr = SymmetryOperation::builder()
        .generating_element(c2x_nsr_element)
        .power(3)
        .build()
        .unwrap();

    let c2y_nsr_element = SymmetryElement::builder()
        .threshold(1e-12)
        .proper_order(ElementOrder::Int(2))
        .proper_power(1)
        .raw_axis(Vector3::y())
        .kind(ROT)
        .rotation_group(RotationGroup::SU2(true))
        .build()
        .unwrap();
    let c2y_nsr = SymmetryOperation::builder()
        .generating_element(c2y_nsr_element.clone())
        .power(1)
        .build()
        .unwrap();
    let c2y_isr = SymmetryOperation::builder()
        .generating_element(c2y_nsr_element)
        .power(3)
        .build()
        .unwrap();

    let c2z_nsr_element = SymmetryElement::builder()
        .threshold(1e-12)
        .proper_order(ElementOrder::Int(2))
        .proper_power(1)
        .raw_axis(Vector3::z())
        .kind(ROT)
        .rotation_group(RotationGroup::SU2(true))
        .build()
        .unwrap();
    let c2z_nsr = SymmetryOperation::builder()
        .generating_element(c2z_nsr_element.clone())
        .power(1)
        .build()
        .unwrap();
    let c2z_isr = SymmetryOperation::builder()
        .generating_element(c2z_nsr_element)
        .power(3)
        .build()
        .unwrap();

    // e_nsr
    assert_eq!(&e_nsr * &e_nsr, e_nsr);
    assert_eq!(&e_nsr * &c2x_nsr, c2x_nsr);
    assert_eq!(&e_nsr * &c2y_nsr, c2y_nsr);
    assert_eq!(&e_nsr * &c2z_nsr, c2z_nsr);
    assert_eq!(&e_nsr * &e_isr, e_isr);
    assert_eq!(&e_nsr * &c2x_isr, c2x_isr);
    assert_eq!(&e_nsr * &c2y_isr, c2y_isr);
    assert_eq!(&e_nsr * &c2z_isr, c2z_isr);

    // c2x_nsr
    assert_eq!(&c2x_nsr * &e_nsr, c2x_nsr);
    assert_eq!(&c2x_nsr * &c2x_nsr, e_isr);
    assert_eq!(&c2x_nsr * &c2y_nsr, c2z_nsr);
    assert_eq!(&c2x_nsr * &c2z_nsr, c2y_isr);
    assert_eq!(&c2x_nsr * &e_isr, c2x_isr);
    assert_eq!(&c2x_nsr * &c2x_isr, e_nsr);
    assert_eq!(&c2x_nsr * &c2y_isr, c2z_isr);
    assert_eq!(&c2x_nsr * &c2z_isr, c2y_nsr);

    // c2y_nsr
    assert_eq!(&c2y_nsr * &e_nsr, c2y_nsr);
    assert_eq!(&c2y_nsr * &c2x_nsr, c2z_isr);
    assert_eq!(&c2y_nsr * &c2y_nsr, e_isr);
    assert_eq!(&c2y_nsr * &c2z_nsr, c2x_nsr);
    assert_eq!(&c2y_nsr * &e_isr, c2y_isr);
    assert_eq!(&c2y_nsr * &c2x_isr, c2z_nsr);
    assert_eq!(&c2y_nsr * &c2y_isr, e_nsr);
    assert_eq!(&c2y_nsr * &c2z_isr, c2x_isr);

    // c2z_nsr
    assert_eq!(&c2z_nsr * &e_nsr, c2z_nsr);
    assert_eq!(&c2z_nsr * &c2x_nsr, c2y_nsr);
    assert_eq!(&c2z_nsr * &c2y_nsr, c2x_isr);
    assert_eq!(&c2z_nsr * &c2z_nsr, e_isr);
    assert_eq!(&c2z_nsr * &e_isr, c2z_isr);
    assert_eq!(&c2z_nsr * &c2x_isr, c2y_isr);
    assert_eq!(&c2z_nsr * &c2y_isr, c2x_nsr);
    assert_eq!(&c2z_nsr * &c2z_isr, e_nsr);

    // e_isr
    assert_eq!(&e_isr * &e_nsr, e_isr);
    assert_eq!(&e_isr * &c2x_nsr, c2x_isr);
    assert_eq!(&e_isr * &c2y_nsr, c2y_isr);
    assert_eq!(&e_isr * &c2z_nsr, c2z_isr);
    assert_eq!(&e_isr * &e_isr, e_nsr);
    assert_eq!(&e_isr * &c2x_isr, c2x_nsr);
    assert_eq!(&e_isr * &c2y_isr, c2y_nsr);
    assert_eq!(&e_isr * &c2z_isr, c2z_nsr);

    // c2x_isr
    assert_eq!(&c2x_isr * &e_nsr, c2x_isr);
    assert_eq!(&c2x_isr * &c2x_nsr, e_nsr);
    assert_eq!(&c2x_isr * &c2y_nsr, c2z_isr);
    assert_eq!(&c2x_isr * &c2z_nsr, c2y_nsr);
    assert_eq!(&c2x_isr * &e_isr, c2x_nsr);
    assert_eq!(&c2x_isr * &c2x_isr, e_isr);
    assert_eq!(&c2x_isr * &c2y_isr, c2z_nsr);
    assert_eq!(&c2x_isr * &c2z_isr, c2y_isr);

    // c2y_isr
    assert_eq!(&c2y_isr * &e_nsr, c2y_isr);
    assert_eq!(&c2y_isr * &c2x_nsr, c2z_nsr);
    assert_eq!(&c2y_isr * &c2y_nsr, e_nsr);
    assert_eq!(&c2y_isr * &c2z_nsr, c2x_isr);
    assert_eq!(&c2y_isr * &e_isr, c2y_nsr);
    assert_eq!(&c2y_isr * &c2x_isr, c2z_isr);
    assert_eq!(&c2y_isr * &c2y_isr, e_isr);
    assert_eq!(&c2y_isr * &c2z_isr, c2x_nsr);

    // c2z_isr
    assert_eq!(&c2z_isr * &e_nsr, c2z_isr);
    assert_eq!(&c2z_isr * &c2x_nsr, c2y_isr);
    assert_eq!(&c2z_isr * &c2y_nsr, c2x_nsr);
    assert_eq!(&c2z_isr * &c2z_nsr, e_nsr);
    assert_eq!(&c2z_isr * &e_isr, c2z_nsr);
    assert_eq!(&c2z_isr * &c2x_isr, c2y_nsr);
    assert_eq!(&c2z_isr * &c2y_isr, c2x_isr);
    assert_eq!(&c2z_isr * &c2z_isr, e_isr);

    // --------------------------------------------------------------------------------------------
    // D3*
    //
    // We stick to the standard definition of the positive hemisphere here, instead of the modified
    // one used for Table 15-5.2 in Altmann, S. L. Rotations, Quaternions, and Double Groups.
    // (Dover Publications, Inc., 2005). The multiplications that we obtain will therefore be
    // different from the reference results in the homotopy classes of some of the the resultant C2
    // operations.
    //
    // The following tests are for development monitoring purposes. Should we manage to implement a
    // way to determine positive hemispheres consistently in the future, these tests can be used to
    // verify how the implementation changes the multiplication structure.
    // --------------------------------------------------------------------------------------------
    let c3_nsr_element = SymmetryElement::builder()
        .threshold(1e-12)
        .proper_order(ElementOrder::Int(3))
        .proper_power(1)
        .raw_axis(Vector3::z())
        .kind(ROT)
        .rotation_group(RotationGroup::SU2(true))
        .build()
        .unwrap();
    let c3p1_nsr = SymmetryOperation::builder()
        .generating_element(c3_nsr_element.clone())
        .power(1)
        .build()
        .unwrap();
    let c3pm1_nsr = SymmetryOperation::builder()
        .generating_element(c3_nsr_element.clone())
        .power(-1)
        .build()
        .unwrap();
    let c3p1_isr = SymmetryOperation::builder()
        .generating_element(c3_nsr_element.clone())
        .power(4)
        .build()
        .unwrap();
    let c3pm1_isr = SymmetryOperation::builder()
        .generating_element(c3_nsr_element)
        .power(-4)
        .build()
        .unwrap();

    let c21_nsr_element = SymmetryElement::builder()
        .threshold(1e-12)
        .proper_order(ElementOrder::Int(2))
        .proper_power(1)
        .raw_axis(Vector3::x())
        .kind(ROT)
        .rotation_group(RotationGroup::SU2(true))
        .build()
        .unwrap();
    let c21_nsr = SymmetryOperation::builder()
        .generating_element(c21_nsr_element.clone())
        .power(1)
        .build()
        .unwrap();
    let c21_isr = SymmetryOperation::builder()
        .generating_element(c21_nsr_element)
        .power(3)
        .build()
        .unwrap();

    let c22_nsr_element = SymmetryElement::builder()
        .threshold(1e-12)
        .proper_order(ElementOrder::Int(2))
        .proper_power(1)
        .raw_axis(Vector3::new(0.5, -(3.0f64.sqrt()) / 2.0, 0.0))
        .kind(ROT)
        .rotation_group(RotationGroup::SU2(true))
        .build()
        .unwrap();
    let c22_nsr = SymmetryOperation::builder()
        .generating_element(c22_nsr_element.clone())
        .power(1)
        .build()
        .unwrap();
    let c22_isr = SymmetryOperation::builder()
        .generating_element(c22_nsr_element)
        .power(3)
        .build()
        .unwrap();

    let c23_nsr_element = SymmetryElement::builder()
        .threshold(1e-12)
        .proper_order(ElementOrder::Int(2))
        .proper_power(1)
        .raw_axis(Vector3::new(0.5, 3.0f64.sqrt() / 2.0, 0.0))
        .kind(ROT)
        .rotation_group(RotationGroup::SU2(true))
        .build()
        .unwrap();
    let c23_nsr = SymmetryOperation::builder()
        .generating_element(c23_nsr_element.clone())
        .power(1)
        .build()
        .unwrap();
    let c23_isr = SymmetryOperation::builder()
        .generating_element(c23_nsr_element)
        .power(3)
        .build()
        .unwrap();

    // e_nsr
    assert_eq!(&e_nsr * &e_nsr, e_nsr);
    assert_eq!(&e_nsr * &c3p1_nsr, c3p1_nsr);
    assert_eq!(&e_nsr * &c3pm1_nsr, c3pm1_nsr);
    assert_eq!(&e_nsr * &c21_nsr, c21_nsr);
    assert_eq!(&e_nsr * &c22_nsr, c22_nsr);
    assert_eq!(&e_nsr * &c23_nsr, c23_nsr);
    assert_eq!(&e_nsr * &e_isr, e_isr);
    assert_eq!(&e_nsr * &c3p1_isr, c3p1_isr);
    assert_eq!(&e_nsr * &c3pm1_isr, c3pm1_isr);
    assert_eq!(&e_nsr * &c21_isr, c21_isr);
    assert_eq!(&e_nsr * &c22_isr, c22_isr);
    assert_eq!(&e_nsr * &c23_isr, c23_isr);

    // c3p1_nsr
    assert_eq!(&c3p1_nsr * &e_nsr, c3p1_nsr);
    assert_eq!(&c3p1_nsr * &c3p1_nsr, c3pm1_isr);
    assert_eq!(&c3p1_nsr * &c3pm1_nsr, e_nsr);
    assert_eq!(&c3p1_nsr * &c21_nsr, c23_nsr);
    assert_eq!(&c3p1_nsr * &c22_nsr, c21_nsr);
    assert_eq!(&c3p1_nsr * &c23_nsr, c22_isr);
    assert_eq!(&c3p1_nsr * &e_isr, c3p1_isr);
    assert_eq!(&c3p1_nsr * &c3p1_isr, c3pm1_nsr);
    assert_eq!(&c3p1_nsr * &c3pm1_isr, e_isr);
    assert_eq!(&c3p1_nsr * &c21_isr, c23_isr);
    assert_eq!(&c3p1_nsr * &c22_isr, c21_isr);
    assert_eq!(&c3p1_nsr * &c23_isr, c22_nsr);

    // c3pm1_nsr
    assert_eq!(&c3pm1_nsr * &e_nsr, c3pm1_nsr);
    assert_eq!(&c3pm1_nsr * &c3p1_nsr, e_nsr);
    assert_eq!(&c3pm1_nsr * &c3pm1_nsr, c3p1_isr);
    assert_eq!(&c3pm1_nsr * &c21_nsr, c22_nsr);
    assert_eq!(&c3pm1_nsr * &c22_nsr, c23_isr);
    assert_eq!(&c3pm1_nsr * &c23_nsr, c21_nsr);
    assert_eq!(&c3pm1_nsr * &e_isr, c3pm1_isr);
    assert_eq!(&c3pm1_nsr * &c3p1_isr, e_isr);
    assert_eq!(&c3pm1_nsr * &c3pm1_isr, c3p1_nsr);
    assert_eq!(&c3pm1_nsr * &c21_isr, c22_isr);
    assert_eq!(&c3pm1_nsr * &c22_isr, c23_nsr);
    assert_eq!(&c3pm1_nsr * &c23_isr, c21_isr);

    // c21_nsr
    assert_eq!(&c21_nsr * &e_nsr, c21_nsr);
    assert_eq!(&c21_nsr * &c3p1_nsr, c22_nsr);
    assert_eq!(&c21_nsr * &c3pm1_nsr, c23_nsr);
    assert_eq!(&c21_nsr * &c21_nsr, e_isr);
    assert_eq!(&c21_nsr * &c22_nsr, c3p1_isr);
    assert_eq!(&c21_nsr * &c23_nsr, c3pm1_isr);
    assert_eq!(&c21_nsr * &e_isr, c21_isr);
    assert_eq!(&c21_nsr * &c3p1_isr, c22_isr);
    assert_eq!(&c21_nsr * &c3pm1_isr, c23_isr);
    assert_eq!(&c21_nsr * &c21_isr, e_nsr);
    assert_eq!(&c21_nsr * &c22_isr, c3p1_nsr);
    assert_eq!(&c21_nsr * &c23_isr, c3pm1_nsr);

    // c22_nsr
    assert_eq!(&c22_nsr * &e_nsr, c22_nsr);
    assert_eq!(&c22_nsr * &c3p1_nsr, c23_isr);
    assert_eq!(&c22_nsr * &c3pm1_nsr, c21_nsr);
    assert_eq!(&c22_nsr * &c21_nsr, c3pm1_isr);
    assert_eq!(&c22_nsr * &c22_nsr, e_isr);
    assert_eq!(&c22_nsr * &c23_nsr, c3p1_nsr);
    assert_eq!(&c22_nsr * &e_isr, c22_isr);
    assert_eq!(&c22_nsr * &c3p1_isr, c23_nsr);
    assert_eq!(&c22_nsr * &c3pm1_isr, c21_isr);
    assert_eq!(&c22_nsr * &c21_isr, c3pm1_nsr);
    assert_eq!(&c22_nsr * &c22_isr, e_nsr);
    assert_eq!(&c22_nsr * &c23_isr, c3p1_isr);

    // c23_nsr
    assert_eq!(&c23_nsr * &e_nsr, c23_nsr);
    assert_eq!(&c23_nsr * &c3p1_nsr, c21_nsr);
    assert_eq!(&c23_nsr * &c3pm1_nsr, c22_isr);
    assert_eq!(&c23_nsr * &c21_nsr, c3p1_isr);
    assert_eq!(&c23_nsr * &c22_nsr, c3pm1_nsr);
    assert_eq!(&c23_nsr * &c23_nsr, e_isr);
    assert_eq!(&c23_nsr * &e_isr, c23_isr);
    assert_eq!(&c23_nsr * &c3p1_isr, c21_isr);
    assert_eq!(&c23_nsr * &c3pm1_isr, c22_nsr);
    assert_eq!(&c23_nsr * &c21_isr, c3p1_nsr);
    assert_eq!(&c23_nsr * &c22_isr, c3pm1_isr);
    assert_eq!(&c23_nsr * &c23_isr, e_nsr);
}

#[test]
fn test_symmetry_operation_time_reversal() {
    let tc2x_element = SymmetryElement::builder()
        .threshold(1e-12)
        .proper_order(ElementOrder::Int(2))
        .proper_power(1)
        .raw_axis(Vector3::new(1.0, 0.0, 0.0))
        .kind(TRROT)
        .rotation_group(RotationGroup::SO3)
        .build()
        .unwrap();

    let tc2x = SymmetryOperation::builder()
        .generating_element(tc2x_element)
        .power(1)
        .build()
        .unwrap();
    assert_eq!(tc2x.order(), 2);
    assert!(tc2x.is_antiunitary());
    assert!((&tc2x * &tc2x).is_identity());

    let c2x_element = SymmetryElement::builder()
        .threshold(1e-12)
        .proper_order(ElementOrder::Int(2))
        .proper_power(1)
        .raw_axis(Vector3::new(1.0, 0.0, 0.0))
        .kind(ROT)
        .rotation_group(RotationGroup::SO3)
        .build()
        .unwrap();
    let c2x = SymmetryOperation::builder()
        .generating_element(c2x_element)
        .power(1)
        .build()
        .unwrap();
    assert!(!c2x.is_antiunitary());

    let t = &tc2x * &c2x;
    assert_eq!(t.order(), 2);
    assert!(t.is_antiunitary());
    assert!(!t.is_identity());
    assert!(t.is_time_reversal());
    assert!((&t * &t).is_identity());

    let tc2y_element = SymmetryElement::builder()
        .threshold(1e-12)
        .proper_order(ElementOrder::Int(2))
        .proper_power(1)
        .raw_axis(Vector3::new(0.0, 1.0, 0.0))
        .kind(TRROT)
        .rotation_group(RotationGroup::SO3)
        .build()
        .unwrap();

    let tc2y = SymmetryOperation::builder()
        .generating_element(tc2y_element)
        .power(1)
        .build()
        .unwrap();

    let c2y_element = SymmetryElement::builder()
        .threshold(1e-12)
        .proper_order(ElementOrder::Int(2))
        .proper_power(1)
        .raw_axis(Vector3::new(0.0, 1.0, 0.0))
        .kind(ROT)
        .rotation_group(RotationGroup::SO3)
        .build()
        .unwrap();
    let c2y = SymmetryOperation::builder()
        .generating_element(c2y_element)
        .power(1)
        .build()
        .unwrap();

    assert_eq!(tc2y.order(), 2);
    assert_eq!(c2y.order(), 2);
    assert_eq!(&t * &c2y, tc2y);
    assert_eq!(&c2y * &t, tc2y);
    assert_eq!(&t * &tc2y, c2y);
    assert_eq!(&tc2y * &t, c2y);
    assert!(!(&tc2y * &tc2x).is_antiunitary());
    assert!((&c2y * &tc2x).is_antiunitary());
    assert!(!(&c2y * &tc2x).is_time_reversal());

    let sd1_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(1))
        .proper_power(1)
        .raw_axis(Vector3::new(1.0, 0.0, 0.0))
        .kind(INV)
        .rotation_group(RotationGroup::SO3)
        .build()
        .unwrap();

    let sd1 = SymmetryOperation::builder()
        .generating_element(sd1_element)
        .power(1)
        .build()
        .unwrap();
    assert_eq!(sd1.order(), 2);
    assert!(!sd1.is_antiunitary());

    let tsd1_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(1))
        .proper_power(1)
        .raw_axis(Vector3::new(1.0, 0.0, 0.0))
        .kind(TRINV)
        .rotation_group(RotationGroup::SO3)
        .build()
        .unwrap();

    let tsd1 = SymmetryOperation::builder()
        .generating_element(tsd1_element)
        .power(1)
        .build()
        .unwrap();
    assert_eq!(tsd1.order(), 2);
    assert!(tsd1.is_antiunitary());
    assert!(!tsd1.is_inversion());
    assert!(!tsd1.is_identity());
    assert!((&tsd1 * &sd1).is_time_reversal());
    assert!((&tsd1 * &t).is_inversion());
}

#[test]
fn test_symmetry_operation_exponentiation() {
    // ==========================
    // Proper symmetry operations
    // ==========================
    let c5_element = SymmetryElement::builder()
        .threshold(1e-12)
        .proper_order(ElementOrder::Int(5))
        .proper_power(1)
        .raw_axis(Vector3::new(2.0, -1.0, 1.0))
        .kind(ROT)
        .rotation_group(RotationGroup::SO3)
        .build()
        .unwrap();

    let c5 = SymmetryOperation::builder()
        .generating_element(c5_element.clone())
        .power(1)
        .build()
        .unwrap();

    let c5pm1 = SymmetryOperation::builder()
        .generating_element(c5_element.clone())
        .power(-1)
        .build()
        .unwrap();
    assert_eq!(c5pm1.order(), 5);
    assert_eq!((&c5).pow(-1), c5pm1);

    let c5p2 = SymmetryOperation::builder()
        .generating_element(c5_element)
        .power(2)
        .build()
        .unwrap();
    assert_eq!(c5p2.order(), 5);
    assert_eq!(c5pm1.pow(-2), c5p2);
    assert_eq!((&c5).pow(3), (&c5).pow(-2));
    assert_ne!((&c5).pow(3), (&c5).pow(-3));

    let c6_element = SymmetryElement::builder()
        .threshold(1e-12)
        .proper_order(ElementOrder::Int(6))
        .proper_power(1)
        .raw_axis(Vector3::new(1.0, 1.0, 1.0))
        .kind(ROT)
        .rotation_group(RotationGroup::SO3)
        .build()
        .unwrap();

    let c6 = SymmetryOperation::builder()
        .generating_element(c6_element)
        .power(1)
        .build()
        .unwrap();
    assert_eq!(c6.order(), 6);

    let c3_element = SymmetryElement::builder()
        .threshold(1e-12)
        .proper_order(ElementOrder::Int(3))
        .proper_power(1)
        .raw_axis(Vector3::new(1.0, 1.0, 1.0))
        .kind(ROT)
        .rotation_group(RotationGroup::SO3)
        .build()
        .unwrap();

    let c3 = SymmetryOperation::builder()
        .generating_element(c3_element)
        .power(1)
        .build()
        .unwrap();
    assert_eq!(c3.order(), 3);
    assert_eq!(c6.pow(2), c3);

    let c2_element = SymmetryElement::builder()
        .threshold(1e-12)
        .proper_order(ElementOrder::Int(2))
        .proper_power(1)
        .raw_axis(Vector3::new(1.0, 1.0, 0.0))
        .kind(ROT)
        .rotation_group(RotationGroup::SO3)
        .build()
        .unwrap();

    let c2 = SymmetryOperation::builder()
        .generating_element(c2_element.clone())
        .power(1)
        .build()
        .unwrap();

    let c2pm1 = SymmetryOperation::builder()
        .generating_element(c2_element)
        .power(-1)
        .build()
        .unwrap();

    assert_eq!(
        (&(&c2 * &c3) * &c2pm1).pow(2),
        &(&c2 * &c3.pow(2)) * &c2.pow(-1)
    );

    // ============================
    // Improper symmetry operations
    // ============================
    let s7_element = SymmetryElement::builder()
        .threshold(1e-12)
        .proper_order(ElementOrder::Int(7))
        .proper_power(1)
        .raw_axis(Vector3::new(2.0, -2.0, 1.0))
        .kind(SIG)
        .rotation_group(RotationGroup::SO3)
        .build()
        .unwrap();

    let s7 = SymmetryOperation::builder()
        .generating_element(s7_element.clone())
        .power(1)
        .build()
        .unwrap();
    assert_eq!(s7.order(), 14);
    assert!((&s7).pow(2).is_proper());
    assert!((&s7).pow(7).is_reflection());
    assert!((&s7).pow(14).is_identity());

    let s7p2 = SymmetryOperation::builder()
        .generating_element(s7_element)
        .power(2)
        .build()
        .unwrap();
    assert_eq!(s7p2.order(), 7);
    assert_eq!((&s7).pow(2), s7p2);

    // ===============================
    // Antiunitary symmetry operations
    // ===============================
    let ts5_element = SymmetryElement::builder()
        .threshold(1e-12)
        .proper_order(ElementOrder::Int(5))
        .proper_power(1)
        .raw_axis(Vector3::new(2.0, -2.0, 2.0))
        .kind(TRSIG)
        .rotation_group(RotationGroup::SO3)
        .build()
        .unwrap();

    let ts5 = SymmetryOperation::builder()
        .generating_element(ts5_element)
        .power(1)
        .build()
        .unwrap();
    assert_eq!(ts5.order(), 10);
    assert!((&ts5).pow(1).is_antiunitary());
    assert!(!(&ts5).pow(2).is_antiunitary());
    assert!(!(&ts5).pow(5).is_time_reversal());

    let tc5_element = SymmetryElement::builder()
        .threshold(1e-12)
        .proper_order(ElementOrder::Int(5))
        .proper_power(1)
        .raw_axis(Vector3::new(2.0, -2.0, 2.0))
        .kind(TRROT)
        .rotation_group(RotationGroup::SO3)
        .build()
        .unwrap();

    let tc5 = SymmetryOperation::builder()
        .generating_element(tc5_element)
        .power(1)
        .build()
        .unwrap();
    assert_eq!(tc5.order(), 10);
    assert!((&tc5).pow(1).is_antiunitary());
    assert!((&tc5).pow(5).is_time_reversal());
}

#[test]
fn test_symmetry_operation_invertibility() {
    let c5_element = SymmetryElement::builder()
        .threshold(1e-12)
        .proper_order(ElementOrder::Int(5))
        .proper_power(1)
        .raw_axis(Vector3::new(2.0, -1.0, 1.0))
        .kind(ROT)
        .rotation_group(RotationGroup::SO3)
        .build()
        .unwrap();

    let c5 = SymmetryOperation::builder()
        .generating_element(c5_element.clone())
        .power(1)
        .build()
        .unwrap();

    let c5pm1 = SymmetryOperation::builder()
        .generating_element(c5_element)
        .power(-1)
        .build()
        .unwrap();
    assert_eq!((&c5).inv(), c5pm1);
    assert!((&c5 * &(&c5).inv()).is_identity());

    let tc5_element = SymmetryElement::builder()
        .threshold(1e-12)
        .proper_order(ElementOrder::Int(5))
        .proper_power(1)
        .raw_axis(Vector3::new(2.0, -2.0, 2.0))
        .kind(TRROT)
        .rotation_group(RotationGroup::SO3)
        .build()
        .unwrap();

    let tc5 = SymmetryOperation::builder()
        .generating_element(tc5_element)
        .power(1)
        .build()
        .unwrap();
    assert_eq!((&c5).inv(), (&c5).pow(-1));
    assert!((&tc5 * &(&tc5).inv()).is_identity());
}

#[test]
fn test_symmetry_operation_hashability() {
    let mut symops: HashSet<SymmetryOperation> = HashSet::new();

    let c8_element = SymmetryElement::builder()
        .threshold(1e-12)
        .proper_order(ElementOrder::Int(8))
        .proper_power(1)
        .raw_axis(Vector3::new(1.0, -1.0, 1.0))
        .kind(ROT)
        .rotation_group(RotationGroup::SO3)
        .build()
        .unwrap();

    let c8 = SymmetryOperation::builder()
        .generating_element(c8_element)
        .power(1)
        .build()
        .unwrap();
    symops.insert((&c8).pow(1));
    symops.insert((&c8).pow(2));
    assert_eq!(symops.len(), 2);

    let c4_element = SymmetryElement::builder()
        .threshold(1e-12)
        .proper_order(ElementOrder::Int(4))
        .proper_power(1)
        .raw_axis(Vector3::new(-1.0, 1.0, -1.0))
        .kind(ROT)
        .rotation_group(RotationGroup::SO3)
        .build()
        .unwrap();

    let c4 = SymmetryOperation::builder()
        .generating_element(c4_element)
        .power(1)
        .build()
        .unwrap();
    assert!(symops.contains(&(&c4).pow(-1)));

    symops.insert((&c8).pow(-2));
    assert!(symops.contains(&c4));

    let s8_element = SymmetryElement::builder()
        .threshold(1e-12)
        .proper_order(ElementOrder::Int(8))
        .proper_power(1)
        .raw_axis(Vector3::new(1.0, -1.0, 1.0))
        .kind(SIG)
        .rotation_group(RotationGroup::SO3)
        .build()
        .unwrap();

    let s8 = SymmetryOperation::builder()
        .generating_element(s8_element)
        .power(1)
        .build()
        .unwrap();
    assert!(!symops.contains(&s8));
    assert!(symops.contains(&s8.pow(2)));

    let tc12_element = SymmetryElement::builder()
        .threshold(1e-12)
        .proper_order(ElementOrder::Int(12))
        .proper_power(1)
        .raw_axis(Vector3::new(1.0, -1.0, 1.0))
        .kind(TRROT)
        .rotation_group(RotationGroup::SO3)
        .build()
        .unwrap();

    let tc12 = SymmetryOperation::builder()
        .generating_element(tc12_element)
        .power(1)
        .build()
        .unwrap();
    assert!(!symops.contains(&tc12));
    assert!(!symops.contains(&tc12.pow(3)));
}

#[test]
fn test_symmetry_operation_su2_comparison() {
    let c5_nsr_element = SymmetryElement::builder()
        .threshold(1e-12)
        .proper_order(ElementOrder::Int(5))
        .proper_power(1)
        .raw_axis(Vector3::new(2.0, -1.0, 1.0))
        .kind(ROT)
        .rotation_group(RotationGroup::SU2(true))
        .build()
        .unwrap();

    let c5_isr_element = SymmetryElement::builder()
        .threshold(1e-12)
        .proper_order(ElementOrder::Int(5))
        .proper_power(1)
        .raw_axis(Vector3::new(2.0, -1.0, 1.0))
        .kind(ROT)
        .rotation_group(RotationGroup::SU2(false))
        .build()
        .unwrap();

    let c5_isr_p1 = SymmetryOperation::builder()
        .generating_element(c5_isr_element.clone())
        .power(1)
        .build()
        .unwrap();
    let c5_nsr_p6 = SymmetryOperation::builder()
        .generating_element(c5_nsr_element.clone())
        .power(6)
        .build()
        .unwrap();
    assert_eq!(c5_isr_p1, c5_nsr_p6);
    assert!(c5_isr_p1.is_su2_class_1());
    assert!(c5_nsr_p6.is_su2_class_1());

    let c5_isr_p2 = SymmetryOperation::builder()
        .generating_element(c5_isr_element.clone())
        .power(2)
        .build()
        .unwrap();
    let c5_nsr_p2 = SymmetryOperation::builder()
        .generating_element(c5_nsr_element.clone())
        .power(2)
        .build()
        .unwrap();
    assert_eq!(c5_isr_p2, c5_nsr_p2);
    assert!(!c5_isr_p2.is_su2_class_1());
    assert!(!c5_nsr_p2.is_su2_class_1());

    let c5_isr_p3 = SymmetryOperation::builder()
        .generating_element(c5_isr_element.clone())
        .power(3)
        .build()
        .unwrap();
    let c5_nsr_p8 = SymmetryOperation::builder()
        .generating_element(c5_nsr_element.clone())
        .power(8)
        .build()
        .unwrap();
    assert_eq!(c5_isr_p3, c5_nsr_p8);
    assert!(!c5_isr_p3.is_su2_class_1());
    assert!(!c5_nsr_p8.is_su2_class_1());

    let c5_isr_p4 = SymmetryOperation::builder()
        .generating_element(c5_isr_element.clone())
        .power(4)
        .build()
        .unwrap();
    let c5_nsr_p4 = SymmetryOperation::builder()
        .generating_element(c5_nsr_element.clone())
        .power(4)
        .build()
        .unwrap();
    assert_eq!(c5_isr_p4, c5_nsr_p4);
    assert!(c5_isr_p4.is_su2_class_1());
    assert!(c5_nsr_p4.is_su2_class_1());

    let c5_isr_p5 = SymmetryOperation::builder()
        .generating_element(c5_isr_element.clone())
        .power(5)
        .build()
        .unwrap();
    let c5_nsr_p10 = SymmetryOperation::builder()
        .generating_element(c5_nsr_element.clone())
        .power(10)
        .build()
        .unwrap();
    assert_eq!(c5_isr_p5, c5_nsr_p10);
    assert!(c5_isr_p5.is_identity());
    assert!(c5_nsr_p10.is_identity());

    let c5_isr_p6 = SymmetryOperation::builder()
        .generating_element(c5_isr_element.clone())
        .power(6)
        .build()
        .unwrap();
    let c5_nsr_p6 = SymmetryOperation::builder()
        .generating_element(c5_nsr_element.clone())
        .power(6)
        .build()
        .unwrap();
    assert_eq!(c5_isr_p6, c5_nsr_p6);
    assert!(c5_isr_p6.is_su2_class_1());
    assert!(c5_nsr_p6.is_su2_class_1());

    let c5_isr_p7 = SymmetryOperation::builder()
        .generating_element(c5_isr_element.clone())
        .power(7)
        .build()
        .unwrap();
    let c5_nsr_p2 = SymmetryOperation::builder()
        .generating_element(c5_nsr_element.clone())
        .power(2)
        .build()
        .unwrap();
    assert_eq!(c5_isr_p7, c5_nsr_p2);
    assert!(!c5_isr_p7.is_su2_class_1());
    assert!(!c5_nsr_p2.is_su2_class_1());

    let c5_isr_p8 = SymmetryOperation::builder()
        .generating_element(c5_isr_element.clone())
        .power(8)
        .build()
        .unwrap();
    let c5_nsr_p8 = SymmetryOperation::builder()
        .generating_element(c5_nsr_element.clone())
        .power(8)
        .build()
        .unwrap();
    assert_eq!(c5_isr_p8, c5_nsr_p8);
    assert!(!c5_isr_p8.is_su2_class_1());
    assert!(!c5_nsr_p8.is_su2_class_1());

    let c5_isr_p9 = SymmetryOperation::builder()
        .generating_element(c5_isr_element.clone())
        .power(9)
        .build()
        .unwrap();
    let c5_nsr_p4 = SymmetryOperation::builder()
        .generating_element(c5_nsr_element.clone())
        .power(4)
        .build()
        .unwrap();
    assert_eq!(c5_isr_p9, c5_nsr_p4);
    assert!(c5_isr_p9.is_su2_class_1());
    assert!(c5_nsr_p4.is_su2_class_1());

    let c5_isr_p10 = SymmetryOperation::builder()
        .generating_element(c5_isr_element)
        .power(10)
        .build()
        .unwrap();
    let c5_nsr_p0 = SymmetryOperation::builder()
        .generating_element(c5_nsr_element.clone())
        .power(0)
        .build()
        .unwrap();
    assert_eq!(c5_isr_p10, c5_nsr_p0);
    assert!(c5_isr_p10.is_identity());
    assert!(c5_nsr_p0.is_identity());

    let c5_nsr_pm1 = SymmetryOperation::builder()
        .generating_element(c5_nsr_element.clone())
        .power(-1)
        .build()
        .unwrap();
    let c5_nsr_p9 = SymmetryOperation::builder()
        .generating_element(c5_nsr_element.clone())
        .power(9)
        .build()
        .unwrap();
    assert_eq!(c5_nsr_pm1, c5_nsr_p9);
    assert!(!c5_nsr_pm1.is_su2_class_1());

    let c5_nsr_pm3 = SymmetryOperation::builder()
        .generating_element(c5_nsr_element.clone())
        .power(-3)
        .build()
        .unwrap();
    let c5_nsr_p7 = SymmetryOperation::builder()
        .generating_element(c5_nsr_element.clone())
        .power(7)
        .build()
        .unwrap();
    assert_eq!(c5_nsr_pm3, c5_nsr_p7);
    assert!(c5_nsr_pm3.is_su2_class_1());
    assert!(c5_nsr_p7.is_su2_class_1());

    let c5_nsr_pm5 = SymmetryOperation::builder()
        .generating_element(c5_nsr_element.clone())
        .power(-5)
        .build()
        .unwrap();
    let c5_nsr_p5 = SymmetryOperation::builder()
        .generating_element(c5_nsr_element.clone())
        .power(5)
        .build()
        .unwrap();
    assert_eq!(c5_nsr_pm5, c5_nsr_p5);
    assert!(c5_nsr_pm5.is_spatial_identity());
    assert!(c5_nsr_pm5.is_su2_class_1());
    assert!(c5_nsr_p5.is_su2_class_1());

    let c5_pp3_nsr_element = SymmetryElement::builder()
        .threshold(1e-12)
        .proper_order(ElementOrder::Int(5))
        .proper_power(3)
        .raw_axis(Vector3::new(2.0, -1.0, 1.0))
        .kind(ROT)
        .rotation_group(RotationGroup::SO3)
        .build()
        .unwrap()
        .to_su2(true)
        .unwrap();
    let c5_pp3_nsr_p2 = SymmetryOperation::builder()
        .generating_element(c5_pp3_nsr_element.clone())
        .power(2)
        .build()
        .unwrap();
    let c5_pp3_nsr_p7 = SymmetryOperation::builder()
        .generating_element(c5_pp3_nsr_element.clone())
        .power(7)
        .build()
        .unwrap();
    let c5_pp3_nsr_pm8 = SymmetryOperation::builder()
        .generating_element(c5_pp3_nsr_element)
        .power(-8)
        .build()
        .unwrap();
    assert_eq!(c5_pp3_nsr_p2, c5_pp3_nsr_p7);
    assert_eq!(c5_pp3_nsr_p2, c5_pp3_nsr_pm8);
    assert!(c5_pp3_nsr_p2.is_su2_class_1());
    assert!(c5_pp3_nsr_p7.is_su2_class_1());
    assert!(c5_pp3_nsr_pm8.is_su2_class_1());

    let c7_element = SymmetryElement::builder()
        .threshold(1e-12)
        .proper_order(ElementOrder::Int(7))
        .proper_power(1)
        .raw_axis(Vector3::new(2.0, -1.0, 1.0))
        .kind(ROT)
        .rotation_group(RotationGroup::SO3)
        .build()
        .unwrap();
    let c7_nsr_element = c7_element.to_su2(true).unwrap();

    let c5_nsr_p5 = SymmetryOperation::builder()
        .generating_element(c5_nsr_element.clone())
        .power(5)
        .build()
        .unwrap();
    let c7_nsr_p7 = SymmetryOperation::builder()
        .generating_element(c7_nsr_element.clone())
        .power(7)
        .build()
        .unwrap();
    assert_eq!(c5_nsr_p5, c7_nsr_p7);
    assert!(c5_nsr_p5.is_spatial_identity() && c5_nsr_p5.is_su2_class_1());

    let c5_nsr_pm5 = SymmetryOperation::builder()
        .generating_element(c5_nsr_element.clone())
        .power(-5)
        .build()
        .unwrap();
    let c7_nsr_pm7 = SymmetryOperation::builder()
        .generating_element(c7_nsr_element.clone())
        .power(-7)
        .build()
        .unwrap();
    assert_eq!(c5_nsr_pm5, c7_nsr_pm7);
    assert!(c5_nsr_pm5.is_spatial_identity() && c5_nsr_pm5.is_su2_class_1());

    let c5_nsr_p10 = SymmetryOperation::builder()
        .generating_element(c5_nsr_element)
        .power(10)
        .build()
        .unwrap();
    let c7_nsr_p14 = SymmetryOperation::builder()
        .generating_element(c7_nsr_element)
        .power(14)
        .build()
        .unwrap();
    assert_eq!(c5_nsr_p10, c7_nsr_p14);
    assert!(c5_nsr_p10.is_identity() && !c5_nsr_p10.is_su2_class_1());
}

#[test]
fn test_symmetry_operation_abbreviated_symbols() {
    // ==========================
    // Proper symmetry operations
    // ==========================
    let c1_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(1))
        .proper_power(1)
        .raw_axis(Vector3::new(0.0, 2.0, 0.0))
        .kind(ROT)
        .rotation_group(RotationGroup::SO3)
        .build()
        .unwrap();

    let c1p1 = SymmetryOperation::builder()
        .generating_element(c1_element.clone())
        .power(1)
        .build()
        .unwrap();
    assert_eq!(&c1p1.get_abbreviated_symbol(), "E");

    let c1pm1 = SymmetryOperation::builder()
        .generating_element(c1_element.clone())
        .power(-1)
        .build()
        .unwrap();
    assert_eq!(&c1pm1.get_abbreviated_symbol(), "E");

    let c1p2 = SymmetryOperation::builder()
        .generating_element(c1_element)
        .power(2)
        .build()
        .unwrap();
    assert_eq!(&c1p2.get_abbreviated_symbol(), "E");

    let c2_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(2))
        .proper_power(1)
        .raw_axis(Vector3::new(1.0, 1.0, 0.0))
        .kind(ROT)
        .rotation_group(RotationGroup::SO3)
        .build()
        .unwrap();

    let c2p1 = SymmetryOperation::builder()
        .generating_element(c2_element.clone())
        .power(1)
        .build()
        .unwrap();
    assert_eq!(&c2p1.get_abbreviated_symbol(), "C2");

    let c2pm1 = SymmetryOperation::builder()
        .generating_element(c2_element.clone())
        .power(-1)
        .build()
        .unwrap();
    assert_eq!(&c2pm1.get_abbreviated_symbol(), "C2");

    let c2p2 = SymmetryOperation::builder()
        .generating_element(c2_element.clone())
        .power(2)
        .build()
        .unwrap();
    assert_eq!(&c2p2.get_abbreviated_symbol(), "E");

    let c2pm2 = SymmetryOperation::builder()
        .generating_element(c2_element)
        .power(-2)
        .build()
        .unwrap();
    assert_eq!(&c2pm2.get_abbreviated_symbol(), "E");

    let c2b_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(2))
        .proper_power(1)
        .raw_axis(Vector3::new(1.0, 1.0, -1.0))
        .kind(ROT)
        .rotation_group(RotationGroup::SO3)
        .build()
        .unwrap();

    let c2bp1 = SymmetryOperation::builder()
        .generating_element(c2b_element.clone())
        .power(1)
        .build()
        .unwrap();
    assert_eq!(&c2bp1.get_abbreviated_symbol(), "C2");

    let c2bpm1 = SymmetryOperation::builder()
        .generating_element(c2b_element)
        .power(-1)
        .build()
        .unwrap();
    assert_eq!(&c2bpm1.get_abbreviated_symbol(), "C2");

    let c3_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(3))
        .proper_power(1)
        .raw_axis(Vector3::new(0.0, 1.0, 0.0))
        .kind(ROT)
        .rotation_group(RotationGroup::SO3)
        .build()
        .unwrap();

    let c3p1 = SymmetryOperation::builder()
        .generating_element(c3_element.clone())
        .power(1)
        .build()
        .unwrap();
    assert_eq!(&c3p1.get_abbreviated_symbol(), "C3");

    let c3pm1 = SymmetryOperation::builder()
        .generating_element(c3_element.clone())
        .power(-1)
        .build()
        .unwrap();
    assert_eq!(&c3pm1.get_abbreviated_symbol(), "C3^(-1)");

    let c3p2 = SymmetryOperation::builder()
        .generating_element(c3_element.clone())
        .power(2)
        .build()
        .unwrap();
    assert_eq!(&c3p2.get_abbreviated_symbol(), "C3^(-1)");

    let c3pm2 = SymmetryOperation::builder()
        .generating_element(c3_element)
        .power(-2)
        .build()
        .unwrap();
    assert_eq!(&c3pm2.get_abbreviated_symbol(), "C3");

    let c3b_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(3))
        .proper_power(1)
        .raw_axis(Vector3::new(1.0, 0.0, -1.0))
        .kind(ROT)
        .rotation_group(RotationGroup::SO3)
        .build()
        .unwrap();

    let c3bp1 = SymmetryOperation::builder()
        .generating_element(c3b_element.clone())
        .power(1)
        .build()
        .unwrap();
    assert_eq!(&c3bp1.get_abbreviated_symbol(), "C3^(-1)");

    let c3bpm1 = SymmetryOperation::builder()
        .generating_element(c3b_element.clone())
        .power(-1)
        .build()
        .unwrap();
    assert_eq!(&c3bpm1.get_abbreviated_symbol(), "C3");

    let c3bp2 = SymmetryOperation::builder()
        .generating_element(c3b_element.clone())
        .power(2)
        .build()
        .unwrap();
    assert_eq!(&c3bp2.get_abbreviated_symbol(), "C3");

    let c3bpm2 = SymmetryOperation::builder()
        .generating_element(c3b_element)
        .power(-2)
        .build()
        .unwrap();
    assert_eq!(&c3bpm2.get_abbreviated_symbol(), "C3^(-1)");

    let c4_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(4))
        .proper_power(1)
        .raw_axis(Vector3::new(1.0, 0.0, 0.0))
        .kind(ROT)
        .rotation_group(RotationGroup::SO3)
        .build()
        .unwrap();

    let c4p1 = SymmetryOperation::builder()
        .generating_element(c4_element.clone())
        .power(1)
        .build()
        .unwrap();
    assert_eq!(&c4p1.get_abbreviated_symbol(), "C4");

    let c4pm1 = SymmetryOperation::builder()
        .generating_element(c4_element.clone())
        .power(-1)
        .build()
        .unwrap();
    assert_eq!(&c4pm1.get_abbreviated_symbol(), "C4^(-1)");

    let c4p2 = SymmetryOperation::builder()
        .generating_element(c4_element.clone())
        .power(2)
        .build()
        .unwrap();
    assert_eq!(&c4p2.get_abbreviated_symbol(), "C2");

    let c4pm2 = SymmetryOperation::builder()
        .generating_element(c4_element.clone())
        .power(-2)
        .build()
        .unwrap();
    assert_eq!(&c4pm2.get_abbreviated_symbol(), "C2");

    let c4p3 = SymmetryOperation::builder()
        .generating_element(c4_element.clone())
        .power(3)
        .build()
        .unwrap();
    assert_eq!(&c4p3.get_abbreviated_symbol(), "C4^(-1)");

    let c4pm3 = SymmetryOperation::builder()
        .generating_element(c4_element)
        .power(-3)
        .build()
        .unwrap();
    assert_eq!(&c4pm3.get_abbreviated_symbol(), "C4");

    // ============================
    // Improper symmetry operations
    // ============================
    let s1_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(1))
        .proper_power(1)
        .raw_axis(Vector3::new(0.0, 2.0, 0.0))
        .kind(SIG)
        .rotation_group(RotationGroup::SO3)
        .build()
        .unwrap();

    let s1p1 = SymmetryOperation::builder()
        .generating_element(s1_element.clone())
        .power(1)
        .build()
        .unwrap();
    assert_eq!(&s1p1.get_abbreviated_symbol(), "σ");

    let s1pm1 = SymmetryOperation::builder()
        .generating_element(s1_element.clone())
        .power(-1)
        .build()
        .unwrap();
    assert_eq!(&s1pm1.get_abbreviated_symbol(), "σ");

    let s1p2 = SymmetryOperation::builder()
        .generating_element(s1_element.clone())
        .power(2)
        .build()
        .unwrap();
    assert_eq!(&s1p2.get_abbreviated_symbol(), "E");

    let s1pm2 = SymmetryOperation::builder()
        .generating_element(s1_element)
        .power(-2)
        .build()
        .unwrap();
    assert_eq!(&s1pm2.get_abbreviated_symbol(), "E");

    let sd1_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(1))
        .proper_power(1)
        .raw_axis(Vector3::new(0.0, 2.0, 0.0))
        .kind(INV)
        .rotation_group(RotationGroup::SO3)
        .build()
        .unwrap();

    let sd1p1 = SymmetryOperation::builder()
        .generating_element(sd1_element.clone())
        .power(1)
        .build()
        .unwrap();
    assert_eq!(&sd1p1.get_abbreviated_symbol(), "i");

    let sd1pm1 = SymmetryOperation::builder()
        .generating_element(sd1_element.clone())
        .power(-1)
        .build()
        .unwrap();
    assert_eq!(&sd1pm1.get_abbreviated_symbol(), "i");

    let sd1p2 = SymmetryOperation::builder()
        .generating_element(sd1_element.clone())
        .power(2)
        .build()
        .unwrap();
    assert_eq!(&sd1p2.get_abbreviated_symbol(), "E");

    let sd1pm2 = SymmetryOperation::builder()
        .generating_element(sd1_element)
        .power(-2)
        .build()
        .unwrap();
    assert_eq!(&sd1pm2.get_abbreviated_symbol(), "E");

    let s2_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(2))
        .proper_power(1)
        .raw_axis(Vector3::new(0.0, 1.0, -1.0))
        .kind(SIG)
        .rotation_group(RotationGroup::SO3)
        .build()
        .unwrap();

    let s2p1 = SymmetryOperation::builder()
        .generating_element(s2_element.clone())
        .power(1)
        .build()
        .unwrap();
    assert_eq!(&s2p1.get_abbreviated_symbol(), "i");

    let s2pm1 = SymmetryOperation::builder()
        .generating_element(s2_element.clone())
        .power(-1)
        .build()
        .unwrap();
    assert_eq!(&s2pm1.get_abbreviated_symbol(), "i");

    let s2p2 = SymmetryOperation::builder()
        .generating_element(s2_element.clone())
        .power(2)
        .build()
        .unwrap();
    assert_eq!(&s2p2.get_abbreviated_symbol(), "E");

    let s2pm2 = SymmetryOperation::builder()
        .generating_element(s2_element)
        .power(-2)
        .build()
        .unwrap();
    assert_eq!(&s2pm2.get_abbreviated_symbol(), "E");

    let sd2_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(2))
        .proper_power(1)
        .raw_axis(Vector3::new(0.0, 1.0, -1.0))
        .kind(INV)
        .rotation_group(RotationGroup::SO3)
        .build()
        .unwrap();

    let sd2p1 = SymmetryOperation::builder()
        .generating_element(sd2_element.clone())
        .power(1)
        .build()
        .unwrap();
    assert_eq!(&sd2p1.get_abbreviated_symbol(), "σ");

    let sd2pm1 = SymmetryOperation::builder()
        .generating_element(sd2_element.clone())
        .power(-1)
        .build()
        .unwrap();
    assert_eq!(&sd2pm1.get_abbreviated_symbol(), "σ");

    let sd2p2 = SymmetryOperation::builder()
        .generating_element(sd2_element.clone())
        .power(2)
        .build()
        .unwrap();
    assert_eq!(&sd2p2.get_abbreviated_symbol(), "E");

    let sd2pm2 = SymmetryOperation::builder()
        .generating_element(sd2_element)
        .power(-2)
        .build()
        .unwrap();
    assert_eq!(&sd2pm2.get_abbreviated_symbol(), "E");

    let s3_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(3))
        .proper_power(1)
        .raw_axis(Vector3::new(0.0, 0.0, 1.0))
        .kind(SIG)
        .rotation_group(RotationGroup::SO3)
        .build()
        .unwrap();

    let s3p1 = SymmetryOperation::builder()
        .generating_element(s3_element.clone())
        .power(1)
        .build()
        .unwrap();
    assert_eq!(&s3p1.get_abbreviated_symbol(), "S3");

    let s3pm1 = SymmetryOperation::builder()
        .generating_element(s3_element.clone())
        .power(-1)
        .build()
        .unwrap();
    assert_eq!(&s3pm1.get_abbreviated_symbol(), "S3^(-1)");

    let s3p2 = SymmetryOperation::builder()
        .generating_element(s3_element.clone())
        .power(2)
        .build()
        .unwrap();
    assert_eq!(&s3p2.get_abbreviated_symbol(), "C3^(-1)");

    let s3pm2 = SymmetryOperation::builder()
        .generating_element(s3_element.clone())
        .power(-2)
        .build()
        .unwrap();
    assert_eq!(&s3pm2.get_abbreviated_symbol(), "C3");

    let s3p3 = SymmetryOperation::builder()
        .generating_element(s3_element.clone())
        .power(3)
        .build()
        .unwrap();
    assert_eq!(&s3p3.get_abbreviated_symbol(), "σ");

    let s3pm3 = SymmetryOperation::builder()
        .generating_element(s3_element.clone())
        .power(-3)
        .build()
        .unwrap();
    assert_eq!(&s3pm3.get_abbreviated_symbol(), "σ");

    let s3p4 = SymmetryOperation::builder()
        .generating_element(s3_element.clone())
        .power(4)
        .build()
        .unwrap();
    assert_eq!(&s3p4.get_abbreviated_symbol(), "C3");

    let s3pm4 = SymmetryOperation::builder()
        .generating_element(s3_element.clone())
        .power(-4)
        .build()
        .unwrap();
    assert_eq!(&s3pm4.get_abbreviated_symbol(), "C3^(-1)");

    let s3p5 = SymmetryOperation::builder()
        .generating_element(s3_element.clone())
        .power(5)
        .build()
        .unwrap();
    assert_eq!(&s3p5.get_abbreviated_symbol(), "S3^(-1)");

    let s3pm5 = SymmetryOperation::builder()
        .generating_element(s3_element)
        .power(-5)
        .build()
        .unwrap();
    assert_eq!(&s3pm5.get_abbreviated_symbol(), "S3");

    let sd3_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(3))
        .proper_power(1)
        .raw_axis(Vector3::new(0.0, 0.0, 1.0))
        .kind(INV)
        .rotation_group(RotationGroup::SO3)
        .build()
        .unwrap();

    let sd3p1 = SymmetryOperation::builder()
        .generating_element(sd3_element.clone())
        .power(1)
        .build()
        .unwrap();
    assert_eq!(&sd3p1.get_abbreviated_symbol(), "Ṡ3");

    let sd3pm1 = SymmetryOperation::builder()
        .generating_element(sd3_element.clone())
        .power(-1)
        .build()
        .unwrap();
    assert_eq!(&sd3pm1.get_abbreviated_symbol(), "Ṡ3^(-1)");

    let sd3p2 = SymmetryOperation::builder()
        .generating_element(sd3_element.clone())
        .power(2)
        .build()
        .unwrap();
    assert_eq!(&sd3p2.get_abbreviated_symbol(), "C3^(-1)");

    let sd3pm2 = SymmetryOperation::builder()
        .generating_element(sd3_element.clone())
        .power(-2)
        .build()
        .unwrap();
    assert_eq!(&sd3pm2.get_abbreviated_symbol(), "C3");

    let sd3p3 = SymmetryOperation::builder()
        .generating_element(sd3_element.clone())
        .power(3)
        .build()
        .unwrap();
    assert_eq!(&sd3p3.get_abbreviated_symbol(), "i");

    let sd3pm3 = SymmetryOperation::builder()
        .generating_element(sd3_element.clone())
        .power(-3)
        .build()
        .unwrap();
    assert_eq!(&sd3pm3.get_abbreviated_symbol(), "i");

    let sd3p4 = SymmetryOperation::builder()
        .generating_element(sd3_element.clone())
        .power(4)
        .build()
        .unwrap();
    assert_eq!(&sd3p4.get_abbreviated_symbol(), "C3");

    let sd3pm4 = SymmetryOperation::builder()
        .generating_element(sd3_element.clone())
        .power(-4)
        .build()
        .unwrap();
    assert_eq!(&sd3pm4.get_abbreviated_symbol(), "C3^(-1)");

    let sd3p5 = SymmetryOperation::builder()
        .generating_element(sd3_element.clone())
        .power(5)
        .build()
        .unwrap();
    assert_eq!(&sd3p5.get_abbreviated_symbol(), "Ṡ3^(-1)");

    let sd3pm5 = SymmetryOperation::builder()
        .generating_element(sd3_element.clone())
        .power(-5)
        .build()
        .unwrap();
    assert_eq!(&sd3pm5.get_abbreviated_symbol(), "Ṡ3");

    let sd3p6 = SymmetryOperation::builder()
        .generating_element(sd3_element.clone())
        .power(6)
        .build()
        .unwrap();
    assert_eq!(&sd3p6.get_abbreviated_symbol(), "E");

    let sd3pm6 = SymmetryOperation::builder()
        .generating_element(sd3_element)
        .power(-6)
        .build()
        .unwrap();
    assert_eq!(&sd3pm6.get_abbreviated_symbol(), "E");

    let s4_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(4))
        .proper_power(1)
        .raw_axis(Vector3::new(0.0, 0.0, 1.0))
        .kind(SIG)
        .rotation_group(RotationGroup::SO3)
        .build()
        .unwrap();

    let s4p1 = SymmetryOperation::builder()
        .generating_element(s4_element.clone())
        .power(1)
        .build()
        .unwrap();
    assert_eq!(&s4p1.get_abbreviated_symbol(), "S4");

    let s4pm1 = SymmetryOperation::builder()
        .generating_element(s4_element.clone())
        .power(-1)
        .build()
        .unwrap();
    assert_eq!(&s4pm1.get_abbreviated_symbol(), "S4^(-1)");

    let s4p2 = SymmetryOperation::builder()
        .generating_element(s4_element.clone())
        .power(2)
        .build()
        .unwrap();
    assert_eq!(&s4p2.get_abbreviated_symbol(), "C2");

    let s4pm2 = SymmetryOperation::builder()
        .generating_element(s4_element.clone())
        .power(-2)
        .build()
        .unwrap();
    assert_eq!(&s4pm2.get_abbreviated_symbol(), "C2");

    let s4p3 = SymmetryOperation::builder()
        .generating_element(s4_element.clone())
        .power(3)
        .build()
        .unwrap();
    assert_eq!(&s4p3.get_abbreviated_symbol(), "S4^(-1)");

    let s4pm3 = SymmetryOperation::builder()
        .generating_element(s4_element.clone())
        .power(-3)
        .build()
        .unwrap();
    assert_eq!(&s4pm3.get_abbreviated_symbol(), "S4");

    let s4p4 = SymmetryOperation::builder()
        .generating_element(s4_element.clone())
        .power(4)
        .build()
        .unwrap();
    assert_eq!(&s4p4.get_abbreviated_symbol(), "E");

    let s4pm4 = SymmetryOperation::builder()
        .generating_element(s4_element)
        .power(-4)
        .build()
        .unwrap();
    assert_eq!(&s4pm4.get_abbreviated_symbol(), "E");

    let sd4_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(4))
        .proper_power(1)
        .raw_axis(Vector3::new(0.0, 0.0, 1.0))
        .kind(INV)
        .rotation_group(RotationGroup::SO3)
        .build()
        .unwrap();

    let sd4p1 = SymmetryOperation::builder()
        .generating_element(sd4_element.clone())
        .power(1)
        .build()
        .unwrap();
    assert_eq!(&sd4p1.get_abbreviated_symbol(), "Ṡ4");

    let sd4pm1 = SymmetryOperation::builder()
        .generating_element(sd4_element.clone())
        .power(-1)
        .build()
        .unwrap();
    assert_eq!(&sd4pm1.get_abbreviated_symbol(), "Ṡ4^(-1)");

    let sd4p2 = SymmetryOperation::builder()
        .generating_element(sd4_element.clone())
        .power(2)
        .build()
        .unwrap();
    assert_eq!(&sd4p2.get_abbreviated_symbol(), "C2");

    let sd4pm2 = SymmetryOperation::builder()
        .generating_element(sd4_element.clone())
        .power(-2)
        .build()
        .unwrap();
    assert_eq!(&sd4pm2.get_abbreviated_symbol(), "C2");

    let sd4p3 = SymmetryOperation::builder()
        .generating_element(sd4_element.clone())
        .power(3)
        .build()
        .unwrap();
    assert_eq!(&sd4p3.get_abbreviated_symbol(), "Ṡ4^(-1)");

    let sd4pm3 = SymmetryOperation::builder()
        .generating_element(sd4_element.clone())
        .power(-3)
        .build()
        .unwrap();
    assert_eq!(&sd4pm3.get_abbreviated_symbol(), "Ṡ4");

    let sd4p4 = SymmetryOperation::builder()
        .generating_element(sd4_element.clone())
        .power(4)
        .build()
        .unwrap();
    assert_eq!(&sd4p4.get_abbreviated_symbol(), "E");

    let sd4pm4 = SymmetryOperation::builder()
        .generating_element(sd4_element)
        .power(-4)
        .build()
        .unwrap();
    assert_eq!(&sd4pm4.get_abbreviated_symbol(), "E");

    let s7_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(7))
        .proper_power(1)
        .raw_axis(Vector3::new(0.0, 0.0, 1.0))
        .kind(SIG)
        .rotation_group(RotationGroup::SO3)
        .build()
        .unwrap();

    let s7p1 = SymmetryOperation::builder()
        .generating_element(s7_element.clone())
        .power(1)
        .build()
        .unwrap();
    assert_eq!(&s7p1.get_abbreviated_symbol(), "S7");

    let s7pm1 = SymmetryOperation::builder()
        .generating_element(s7_element.clone())
        .power(-1)
        .build()
        .unwrap();
    assert_eq!(&s7pm1.get_abbreviated_symbol(), "S7^(-1)");

    let s7p2 = SymmetryOperation::builder()
        .generating_element(s7_element.clone())
        .power(2)
        .build()
        .unwrap();
    assert_eq!(&s7p2.get_abbreviated_symbol(), "C7^2");

    let s7pm2 = SymmetryOperation::builder()
        .generating_element(s7_element.clone())
        .power(-2)
        .build()
        .unwrap();
    assert_eq!(&s7pm2.get_abbreviated_symbol(), "C7^(-2)");

    let s7p3 = SymmetryOperation::builder()
        .generating_element(s7_element.clone())
        .power(3)
        .build()
        .unwrap();
    assert_eq!(&s7p3.get_abbreviated_symbol(), "σC7^3");

    let s7pm3 = SymmetryOperation::builder()
        .generating_element(s7_element.clone())
        .power(-3)
        .build()
        .unwrap();
    assert_eq!(&s7pm3.get_abbreviated_symbol(), "σC7^(-3)");

    let s7p4 = SymmetryOperation::builder()
        .generating_element(s7_element.clone())
        .power(4)
        .build()
        .unwrap();
    assert_eq!(&s7p4.get_abbreviated_symbol(), "C7^(-3)");

    let s7pm4 = SymmetryOperation::builder()
        .generating_element(s7_element.clone())
        .power(-4)
        .build()
        .unwrap();
    assert_eq!(&s7pm4.get_abbreviated_symbol(), "C7^3");

    let s7p5 = SymmetryOperation::builder()
        .generating_element(s7_element.clone())
        .power(5)
        .build()
        .unwrap();
    assert_eq!(&s7p5.get_abbreviated_symbol(), "σC7^(-2)");

    let s7pm5 = SymmetryOperation::builder()
        .generating_element(s7_element.clone())
        .power(-5)
        .build()
        .unwrap();
    assert_eq!(&s7pm5.get_abbreviated_symbol(), "σC7^2");

    let s7p6 = SymmetryOperation::builder()
        .generating_element(s7_element.clone())
        .power(6)
        .build()
        .unwrap();
    assert_eq!(&s7p6.get_abbreviated_symbol(), "C7^(-1)");

    let s7pm6 = SymmetryOperation::builder()
        .generating_element(s7_element.clone())
        .power(-6)
        .build()
        .unwrap();
    assert_eq!(&s7pm6.get_abbreviated_symbol(), "C7");

    let s7p7 = SymmetryOperation::builder()
        .generating_element(s7_element.clone())
        .power(7)
        .build()
        .unwrap();
    assert_eq!(&s7p7.get_abbreviated_symbol(), "σ");

    let s7pm7 = SymmetryOperation::builder()
        .generating_element(s7_element)
        .power(-7)
        .build()
        .unwrap();
    assert_eq!(&s7pm7.get_abbreviated_symbol(), "σ");
}

#[test]
fn test_symmetry_operation_su2_abbreviated_symbols() {
    // ==========================
    // Proper symmetry operations
    // ==========================
    let c1_nsr_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(1))
        .proper_power(1)
        .raw_axis(Vector3::new(0.0, 2.0, 0.0))
        .kind(ROT)
        .rotation_group(RotationGroup::SU2(true))
        .build()
        .unwrap();

    let c1_nsr_p1 = SymmetryOperation::builder()
        .generating_element(c1_nsr_element.clone())
        .power(1)
        .build()
        .unwrap();
    assert_eq!(&c1_nsr_p1.get_abbreviated_symbol(), "E(Σ)");

    let c1_nsr_p2 = SymmetryOperation::builder()
        .generating_element(c1_nsr_element)
        .power(2)
        .build()
        .unwrap();
    assert_eq!(&c1_nsr_p2.get_abbreviated_symbol(), "E(Σ)");

    let c1_isr_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(1))
        .proper_power(1)
        .raw_axis(Vector3::new(0.0, 2.0, 0.0))
        .kind(ROT)
        .rotation_group(RotationGroup::SU2(false))
        .build()
        .unwrap();

    let c1_isr_p1 = SymmetryOperation::builder()
        .generating_element(c1_isr_element.clone())
        .power(1)
        .build()
        .unwrap();
    assert_eq!(&c1_isr_p1.get_abbreviated_symbol(), "E(QΣ)");

    let c1_isr_p2 = SymmetryOperation::builder()
        .generating_element(c1_isr_element)
        .power(2)
        .build()
        .unwrap();
    assert_eq!(&c1_isr_p2.get_abbreviated_symbol(), "E(Σ)");

    let c2_nsr_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(2))
        .proper_power(1)
        .raw_axis(Vector3::new(0.0, 2.0, 0.0))
        .kind(ROT)
        .rotation_group(RotationGroup::SU2(true))
        .build()
        .unwrap();

    let c2_nsr_p1 = SymmetryOperation::builder()
        .generating_element(c2_nsr_element.clone())
        .power(1)
        .build()
        .unwrap();
    assert_eq!(&c2_nsr_p1.get_abbreviated_symbol(), "C2(Σ)");

    let c2_nsr_p2 = SymmetryOperation::builder()
        .generating_element(c2_nsr_element.clone())
        .power(2)
        .build()
        .unwrap();
    assert_eq!(&c2_nsr_p2.get_abbreviated_symbol(), "E(QΣ)");

    let c2_nsr_p3 = SymmetryOperation::builder()
        .generating_element(c2_nsr_element.clone())
        .power(3)
        .build()
        .unwrap();
    assert_eq!(&c2_nsr_p3.get_abbreviated_symbol(), "C2(QΣ)");

    let c2_nsr_p4 = SymmetryOperation::builder()
        .generating_element(c2_nsr_element)
        .power(4)
        .build()
        .unwrap();
    assert_eq!(&c2_nsr_p4.get_abbreviated_symbol(), "E(Σ)");

    let c2_isr_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(2))
        .proper_power(1)
        .raw_axis(Vector3::new(0.0, -2.0, 0.0))
        .kind(ROT)
        .rotation_group(RotationGroup::SU2(false))
        .build()
        .unwrap();

    let c2_isr_p1 = SymmetryOperation::builder()
        .generating_element(c2_isr_element.clone())
        .power(1)
        .build()
        .unwrap();
    assert_eq!(&c2_isr_p1.get_abbreviated_symbol(), "C2(Σ)");

    let c2_isr_p2 = SymmetryOperation::builder()
        .generating_element(c2_isr_element.clone())
        .power(2)
        .build()
        .unwrap();
    assert_eq!(&c2_isr_p2.get_abbreviated_symbol(), "E(QΣ)");

    let c2_isr_p3 = SymmetryOperation::builder()
        .generating_element(c2_isr_element.clone())
        .power(3)
        .build()
        .unwrap();
    assert_eq!(&c2_isr_p3.get_abbreviated_symbol(), "C2(QΣ)");

    let c2_isr_p4 = SymmetryOperation::builder()
        .generating_element(c2_isr_element)
        .power(4)
        .build()
        .unwrap();
    assert_eq!(&c2_isr_p4.get_abbreviated_symbol(), "E(Σ)");

    let c3_nsr_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(3))
        .proper_power(1)
        .raw_axis(Vector3::new(0.0, 1.0, -1.0))
        .kind(ROT)
        .rotation_group(RotationGroup::SU2(true))
        .build()
        .unwrap();

    let c3_nsr_p1 = SymmetryOperation::builder()
        .generating_element(c3_nsr_element.clone())
        .power(1)
        .build()
        .unwrap();
    assert_eq!(&c3_nsr_p1.get_abbreviated_symbol(), "C3^(-1)(Σ)");

    let c3_nsr_p2 = SymmetryOperation::builder()
        .generating_element(c3_nsr_element.clone())
        .power(2)
        .build()
        .unwrap();
    assert_eq!(&c3_nsr_p2.get_abbreviated_symbol(), "C3(QΣ)");

    let c3_nsr_p3 = SymmetryOperation::builder()
        .generating_element(c3_nsr_element.clone())
        .power(3)
        .build()
        .unwrap();
    assert_eq!(&c3_nsr_p3.get_abbreviated_symbol(), "E(QΣ)");

    let c3_nsr_p4 = SymmetryOperation::builder()
        .generating_element(c3_nsr_element.clone())
        .power(4)
        .build()
        .unwrap();
    assert_eq!(&c3_nsr_p4.get_abbreviated_symbol(), "C3^(-1)(QΣ)");

    let c3_nsr_p5 = SymmetryOperation::builder()
        .generating_element(c3_nsr_element.clone())
        .power(5)
        .build()
        .unwrap();
    assert_eq!(&c3_nsr_p5.get_abbreviated_symbol(), "C3(Σ)");

    let c3_nsr_p6 = SymmetryOperation::builder()
        .generating_element(c3_nsr_element)
        .power(6)
        .build()
        .unwrap();
    assert_eq!(&c3_nsr_p6.get_abbreviated_symbol(), "E(Σ)");

    let c7_nsr_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(7))
        .proper_power(1)
        .raw_axis(Vector3::new(0.0, 1.0, 1.0))
        .kind(ROT)
        .rotation_group(RotationGroup::SU2(true))
        .build()
        .unwrap();

    let c7_nsr_p1 = SymmetryOperation::builder()
        .generating_element(c7_nsr_element.clone())
        .power(1)
        .build()
        .unwrap();
    assert_eq!(&c7_nsr_p1.get_abbreviated_symbol(), "C7(Σ)");

    let c7_nsr_p2 = SymmetryOperation::builder()
        .generating_element(c7_nsr_element.clone())
        .power(2)
        .build()
        .unwrap();
    assert_eq!(&c7_nsr_p2.get_abbreviated_symbol(), "C7^2(Σ)");

    let c7_nsr_p3 = SymmetryOperation::builder()
        .generating_element(c7_nsr_element.clone())
        .power(3)
        .build()
        .unwrap();
    assert_eq!(&c7_nsr_p3.get_abbreviated_symbol(), "C7^3(Σ)");

    let c7_nsr_p4 = SymmetryOperation::builder()
        .generating_element(c7_nsr_element.clone())
        .power(4)
        .build()
        .unwrap();
    assert_eq!(&c7_nsr_p4.get_abbreviated_symbol(), "C7^(-3)(QΣ)");

    let c7_nsr_p5 = SymmetryOperation::builder()
        .generating_element(c7_nsr_element.clone())
        .power(5)
        .build()
        .unwrap();
    assert_eq!(&c7_nsr_p5.get_abbreviated_symbol(), "C7^(-2)(QΣ)");

    let c7_nsr_p6 = SymmetryOperation::builder()
        .generating_element(c7_nsr_element.clone())
        .power(6)
        .build()
        .unwrap();
    assert_eq!(&c7_nsr_p6.get_abbreviated_symbol(), "C7^(-1)(QΣ)");

    let c7_nsr_p7 = SymmetryOperation::builder()
        .generating_element(c7_nsr_element.clone())
        .power(7)
        .build()
        .unwrap();
    assert_eq!(&c7_nsr_p7.get_abbreviated_symbol(), "E(QΣ)");

    let c7_nsr_p8 = SymmetryOperation::builder()
        .generating_element(c7_nsr_element.clone())
        .power(8)
        .build()
        .unwrap();
    assert_eq!(&c7_nsr_p8.get_abbreviated_symbol(), "C7(QΣ)");

    let c7_nsr_p9 = SymmetryOperation::builder()
        .generating_element(c7_nsr_element.clone())
        .power(9)
        .build()
        .unwrap();
    assert_eq!(&c7_nsr_p9.get_abbreviated_symbol(), "C7^2(QΣ)");

    let c7_nsr_p10 = SymmetryOperation::builder()
        .generating_element(c7_nsr_element.clone())
        .power(10)
        .build()
        .unwrap();
    assert_eq!(&c7_nsr_p10.get_abbreviated_symbol(), "C7^3(QΣ)");

    let c7_nsr_p11 = SymmetryOperation::builder()
        .generating_element(c7_nsr_element.clone())
        .power(11)
        .build()
        .unwrap();
    assert_eq!(&c7_nsr_p11.get_abbreviated_symbol(), "C7^(-3)(Σ)");

    let c7_nsr_p12 = SymmetryOperation::builder()
        .generating_element(c7_nsr_element.clone())
        .power(12)
        .build()
        .unwrap();
    assert_eq!(&c7_nsr_p12.get_abbreviated_symbol(), "C7^(-2)(Σ)");

    let c7_nsr_p13 = SymmetryOperation::builder()
        .generating_element(c7_nsr_element.clone())
        .power(13)
        .build()
        .unwrap();
    assert_eq!(&c7_nsr_p13.get_abbreviated_symbol(), "C7^(-1)(Σ)");

    let c7_nsr_p14 = SymmetryOperation::builder()
        .generating_element(c7_nsr_element)
        .power(14)
        .build()
        .unwrap();
    assert_eq!(&c7_nsr_p14.get_abbreviated_symbol(), "E(Σ)");

    // ============================
    // Improper symmetry operations
    // ============================
    let s1_nsr_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(1))
        .proper_power(1)
        .raw_axis(Vector3::new(0.0, 1.0, 1.0))
        .kind(SIG)
        .rotation_group(RotationGroup::SU2(true))
        .build()
        .unwrap();

    let s1_nsr_p1 = SymmetryOperation::builder()
        .generating_element(s1_nsr_element.clone())
        .power(1)
        .build()
        .unwrap();
    assert_eq!(&s1_nsr_p1.get_abbreviated_symbol(), "σ(Σ)");

    let s1_nsr_pm1 = SymmetryOperation::builder()
        .generating_element(s1_nsr_element.clone())
        .power(-1)
        .build()
        .unwrap();
    assert_eq!(&s1_nsr_pm1.get_abbreviated_symbol(), "σ(QΣ)");

    let s1_nsr_p2 = SymmetryOperation::builder()
        .generating_element(s1_nsr_element.clone())
        .power(2)
        .build()
        .unwrap();
    assert_eq!(&s1_nsr_p2.get_abbreviated_symbol(), "E(QΣ)");

    let s1_nsr_pm2 = SymmetryOperation::builder()
        .generating_element(s1_nsr_element.clone())
        .power(-2)
        .build()
        .unwrap();
    assert_eq!(&s1_nsr_pm2.get_abbreviated_symbol(), "E(QΣ)");

    let s1_nsr_p3 = SymmetryOperation::builder()
        .generating_element(s1_nsr_element.clone())
        .power(3)
        .build()
        .unwrap();
    assert_eq!(&s1_nsr_p3.get_abbreviated_symbol(), "σ(QΣ)");

    let s1_nsr_pm3 = SymmetryOperation::builder()
        .generating_element(s1_nsr_element.clone())
        .power(-3)
        .build()
        .unwrap();
    assert_eq!(&s1_nsr_pm3.get_abbreviated_symbol(), "σ(Σ)");

    let s1_nsr_p4 = SymmetryOperation::builder()
        .generating_element(s1_nsr_element.clone())
        .power(4)
        .build()
        .unwrap();
    assert_eq!(&s1_nsr_p4.get_abbreviated_symbol(), "E(Σ)");

    let s1_nsr_pm4 = SymmetryOperation::builder()
        .generating_element(s1_nsr_element)
        .power(-4)
        .build()
        .unwrap();
    assert_eq!(&s1_nsr_pm4.get_abbreviated_symbol(), "E(Σ)");

    let sd1_nsr_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(1))
        .proper_power(1)
        .raw_axis(Vector3::new(0.0, 1.0, 1.0))
        .kind(INV)
        .rotation_group(RotationGroup::SU2(true))
        .build()
        .unwrap();

    let sd1_nsr_p1 = SymmetryOperation::builder()
        .generating_element(sd1_nsr_element.clone())
        .power(1)
        .build()
        .unwrap();
    assert_eq!(&sd1_nsr_p1.get_abbreviated_symbol(), "i(Σ)");

    let sd1_nsr_pm1 = SymmetryOperation::builder()
        .generating_element(sd1_nsr_element.clone())
        .power(-1)
        .build()
        .unwrap();
    assert_eq!(&sd1_nsr_pm1.get_abbreviated_symbol(), "i(Σ)");

    let sd1_nsr_p2 = SymmetryOperation::builder()
        .generating_element(sd1_nsr_element.clone())
        .power(2)
        .build()
        .unwrap();
    assert_eq!(&sd1_nsr_p2.get_abbreviated_symbol(), "E(Σ)");

    let sd1_nsr_pm2 = SymmetryOperation::builder()
        .generating_element(sd1_nsr_element)
        .power(-2)
        .build()
        .unwrap();
    assert_eq!(&sd1_nsr_pm2.get_abbreviated_symbol(), "E(Σ)");

    let s2_nsr_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(2))
        .proper_power(1)
        .raw_axis(Vector3::new(0.0, 1.0, 1.0))
        .kind(SIG)
        .rotation_group(RotationGroup::SU2(true))
        .build()
        .unwrap();

    let s2_nsr_p1 = SymmetryOperation::builder()
        .generating_element(s2_nsr_element.clone())
        .power(1)
        .build()
        .unwrap();
    assert_eq!(&s2_nsr_p1.get_abbreviated_symbol(), "i(Σ)");

    let s2_nsr_pm1 = SymmetryOperation::builder()
        .generating_element(s2_nsr_element.clone())
        .power(-1)
        .build()
        .unwrap();
    assert_eq!(&s2_nsr_pm1.get_abbreviated_symbol(), "i(Σ)");

    let s2_nsr_p2 = SymmetryOperation::builder()
        .generating_element(s2_nsr_element.clone())
        .power(2)
        .build()
        .unwrap();
    assert_eq!(&s2_nsr_p2.get_abbreviated_symbol(), "E(Σ)");

    let s2_nsr_pm2 = SymmetryOperation::builder()
        .generating_element(s2_nsr_element)
        .power(-2)
        .build()
        .unwrap();
    assert_eq!(&s2_nsr_pm2.get_abbreviated_symbol(), "E(Σ)");

    let sd2_nsr_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(2))
        .proper_power(1)
        .raw_axis(Vector3::new(0.0, 1.0, 1.0))
        .kind(INV)
        .rotation_group(RotationGroup::SU2(true))
        .build()
        .unwrap();

    let sd2_nsr_p1 = SymmetryOperation::builder()
        .generating_element(sd2_nsr_element.clone())
        .power(1)
        .build()
        .unwrap();
    assert_eq!(&sd2_nsr_p1.get_abbreviated_symbol(), "σ(Σ)");

    let sd2_nsr_pm1 = SymmetryOperation::builder()
        .generating_element(sd2_nsr_element.clone())
        .power(-1)
        .build()
        .unwrap();
    assert_eq!(&sd2_nsr_pm1.get_abbreviated_symbol(), "σ(QΣ)");

    let sd2_nsr_p2 = SymmetryOperation::builder()
        .generating_element(sd2_nsr_element.clone())
        .power(2)
        .build()
        .unwrap();
    assert_eq!(&sd2_nsr_p2.get_abbreviated_symbol(), "E(QΣ)");

    let sd2_nsr_pm2 = SymmetryOperation::builder()
        .generating_element(sd2_nsr_element.clone())
        .power(-2)
        .build()
        .unwrap();
    assert_eq!(&sd2_nsr_pm2.get_abbreviated_symbol(), "E(QΣ)");

    let sd2_nsr_p3 = SymmetryOperation::builder()
        .generating_element(sd2_nsr_element.clone())
        .power(3)
        .build()
        .unwrap();
    assert_eq!(&sd2_nsr_p3.get_abbreviated_symbol(), "σ(QΣ)");

    let sd2_nsr_pm3 = SymmetryOperation::builder()
        .generating_element(sd2_nsr_element.clone())
        .power(-3)
        .build()
        .unwrap();
    assert_eq!(&sd2_nsr_pm3.get_abbreviated_symbol(), "σ(Σ)");

    let sd2_nsr_p4 = SymmetryOperation::builder()
        .generating_element(sd2_nsr_element.clone())
        .power(4)
        .build()
        .unwrap();
    assert_eq!(&sd2_nsr_p4.get_abbreviated_symbol(), "E(Σ)");

    let sd2_nsr_pm4 = SymmetryOperation::builder()
        .generating_element(sd2_nsr_element)
        .power(-4)
        .build()
        .unwrap();
    assert_eq!(&sd2_nsr_pm4.get_abbreviated_symbol(), "E(Σ)");

    let s3_nsr_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(3))
        .proper_power(1)
        .raw_axis(Vector3::new(0.0, 1.0, 1.0))
        .kind(SIG)
        .rotation_group(RotationGroup::SU2(true))
        .build()
        .unwrap();

    let s3_nsr_p1 = SymmetryOperation::builder()
        .generating_element(s3_nsr_element.clone())
        .power(1)
        .build()
        .unwrap();
    assert_eq!(&s3_nsr_p1.get_abbreviated_symbol(), "S3(Σ)");

    let s3_nsr_p2 = SymmetryOperation::builder()
        .generating_element(s3_nsr_element.clone())
        .power(2)
        .build()
        .unwrap();
    assert_eq!(&s3_nsr_p2.get_abbreviated_symbol(), "C3^(-1)(Σ)");

    let s3_nsr_p3 = SymmetryOperation::builder()
        .generating_element(s3_nsr_element.clone())
        .power(3)
        .build()
        .unwrap();
    assert_eq!(&s3_nsr_p3.get_abbreviated_symbol(), "σ(QΣ)");

    let s3_nsr_p4 = SymmetryOperation::builder()
        .generating_element(s3_nsr_element.clone())
        .power(4)
        .build()
        .unwrap();
    assert_eq!(&s3_nsr_p4.get_abbreviated_symbol(), "C3(QΣ)");

    let s3_nsr_p5 = SymmetryOperation::builder()
        .generating_element(s3_nsr_element.clone())
        .power(5)
        .build()
        .unwrap();
    assert_eq!(&s3_nsr_p5.get_abbreviated_symbol(), "S3^(-1)(QΣ)");

    let s3_nsr_p6 = SymmetryOperation::builder()
        .generating_element(s3_nsr_element.clone())
        .power(6)
        .build()
        .unwrap();
    assert_eq!(&s3_nsr_p6.get_abbreviated_symbol(), "E(QΣ)");

    let s3_nsr_p7 = SymmetryOperation::builder()
        .generating_element(s3_nsr_element.clone())
        .power(7)
        .build()
        .unwrap();
    assert_eq!(&s3_nsr_p7.get_abbreviated_symbol(), "S3(QΣ)");

    let s3_nsr_p8 = SymmetryOperation::builder()
        .generating_element(s3_nsr_element.clone())
        .power(8)
        .build()
        .unwrap();
    assert_eq!(&s3_nsr_p8.get_abbreviated_symbol(), "C3^(-1)(QΣ)");

    let s3_nsr_p9 = SymmetryOperation::builder()
        .generating_element(s3_nsr_element.clone())
        .power(9)
        .build()
        .unwrap();
    assert_eq!(&s3_nsr_p9.get_abbreviated_symbol(), "σ(Σ)");

    let s3_nsr_p10 = SymmetryOperation::builder()
        .generating_element(s3_nsr_element.clone())
        .power(10)
        .build()
        .unwrap();
    assert_eq!(&s3_nsr_p10.get_abbreviated_symbol(), "C3(Σ)");

    let s3_nsr_p11 = SymmetryOperation::builder()
        .generating_element(s3_nsr_element.clone())
        .power(11)
        .build()
        .unwrap();
    assert_eq!(&s3_nsr_p11.get_abbreviated_symbol(), "S3^(-1)(Σ)");

    let s3_nsr_p12 = SymmetryOperation::builder()
        .generating_element(s3_nsr_element)
        .power(12)
        .build()
        .unwrap();
    assert_eq!(&s3_nsr_p12.get_abbreviated_symbol(), "E(Σ)");

    let s4_nsr_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(4))
        .proper_power(1)
        .raw_axis(Vector3::new(0.0, 1.0, 1.0))
        .kind(SIG)
        .rotation_group(RotationGroup::SU2(true))
        .build()
        .unwrap();

    let s4_nsr_p1 = SymmetryOperation::builder()
        .generating_element(s4_nsr_element.clone())
        .power(1)
        .build()
        .unwrap();
    assert_eq!(&s4_nsr_p1.get_abbreviated_symbol(), "S4(Σ)");

    let s4_nsr_p2 = SymmetryOperation::builder()
        .generating_element(s4_nsr_element.clone())
        .power(2)
        .build()
        .unwrap();
    assert_eq!(&s4_nsr_p2.get_abbreviated_symbol(), "C2(QΣ)");

    let s4_nsr_p3 = SymmetryOperation::builder()
        .generating_element(s4_nsr_element.clone())
        .power(3)
        .build()
        .unwrap();
    assert_eq!(&s4_nsr_p3.get_abbreviated_symbol(), "S4^(-1)(QΣ)");

    let s4_nsr_p4 = SymmetryOperation::builder()
        .generating_element(s4_nsr_element.clone())
        .power(4)
        .build()
        .unwrap();
    assert_eq!(&s4_nsr_p4.get_abbreviated_symbol(), "E(QΣ)");

    let s4_nsr_p5 = SymmetryOperation::builder()
        .generating_element(s4_nsr_element.clone())
        .power(5)
        .build()
        .unwrap();
    assert_eq!(&s4_nsr_p5.get_abbreviated_symbol(), "S4(QΣ)");

    let s4_nsr_p6 = SymmetryOperation::builder()
        .generating_element(s4_nsr_element.clone())
        .power(6)
        .build()
        .unwrap();
    assert_eq!(&s4_nsr_p6.get_abbreviated_symbol(), "C2(Σ)");

    let s4_nsr_p7 = SymmetryOperation::builder()
        .generating_element(s4_nsr_element.clone())
        .power(7)
        .build()
        .unwrap();
    assert_eq!(&s4_nsr_p7.get_abbreviated_symbol(), "S4^(-1)(Σ)");

    let s4_nsr_p8 = SymmetryOperation::builder()
        .generating_element(s4_nsr_element)
        .power(8)
        .build()
        .unwrap();
    assert_eq!(&s4_nsr_p8.get_abbreviated_symbol(), "E(Σ)");

    let s5_nsr_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(5))
        .proper_power(1)
        .raw_axis(Vector3::new(1.0, 0.0, 0.0))
        .kind(SIG)
        .rotation_group(RotationGroup::SU2(true))
        .build()
        .unwrap();

    let s5_nsr_p1 = SymmetryOperation::builder()
        .generating_element(s5_nsr_element.clone())
        .power(1)
        .build()
        .unwrap();
    assert_eq!(&s5_nsr_p1.get_abbreviated_symbol(), "S5(Σ)");

    let s5_nsr_p2 = SymmetryOperation::builder()
        .generating_element(s5_nsr_element.clone())
        .power(2)
        .build()
        .unwrap();
    assert_eq!(&s5_nsr_p2.get_abbreviated_symbol(), "C5^2(QΣ)");

    let s5_nsr_p3 = SymmetryOperation::builder()
        .generating_element(s5_nsr_element.clone())
        .power(3)
        .build()
        .unwrap();
    assert_eq!(&s5_nsr_p3.get_abbreviated_symbol(), "σC5^(-2)(QΣ)");

    let s5_nsr_p4 = SymmetryOperation::builder()
        .generating_element(s5_nsr_element.clone())
        .power(4)
        .build()
        .unwrap();
    assert_eq!(&s5_nsr_p4.get_abbreviated_symbol(), "C5^(-1)(QΣ)");

    let s5_nsr_p5 = SymmetryOperation::builder()
        .generating_element(s5_nsr_element.clone())
        .power(5)
        .build()
        .unwrap();
    assert_eq!(&s5_nsr_p5.get_abbreviated_symbol(), "σ(Σ)");

    let s5_nsr_p6 = SymmetryOperation::builder()
        .generating_element(s5_nsr_element.clone())
        .power(6)
        .build()
        .unwrap();
    assert_eq!(&s5_nsr_p6.get_abbreviated_symbol(), "C5(Σ)");

    let s5_nsr_p7 = SymmetryOperation::builder()
        .generating_element(s5_nsr_element.clone())
        .power(7)
        .build()
        .unwrap();
    assert_eq!(&s5_nsr_p7.get_abbreviated_symbol(), "σC5^2(Σ)");

    let s5_nsr_p8 = SymmetryOperation::builder()
        .generating_element(s5_nsr_element.clone())
        .power(8)
        .build()
        .unwrap();
    assert_eq!(&s5_nsr_p8.get_abbreviated_symbol(), "C5^(-2)(Σ)");

    let s5_nsr_p9 = SymmetryOperation::builder()
        .generating_element(s5_nsr_element.clone())
        .power(9)
        .build()
        .unwrap();
    assert_eq!(&s5_nsr_p9.get_abbreviated_symbol(), "S5^(-1)(QΣ)");

    let s5_nsr_p10 = SymmetryOperation::builder()
        .generating_element(s5_nsr_element)
        .power(10)
        .build()
        .unwrap();
    assert_eq!(&s5_nsr_p10.get_abbreviated_symbol(), "E(QΣ)");

    assert_eq!(s5_nsr_p1.order(), 20);
}

#[test]
fn test_symmetry_operation_to_symmetry_element() {
    // ================================
    // Proper symmetry operations (SO3)
    // ================================
    let c3_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(3))
        .proper_power(1)
        .raw_axis(Vector3::new(0.0, 2.0, 0.0))
        .kind(ROT)
        .rotation_group(SO3)
        .build()
        .unwrap();

    let c3p2 = SymmetryOperation::builder()
        .generating_element(c3_element)
        .power(2)
        .build()
        .unwrap();
    assert_eq!(
        c3p2.to_symmetry_element().to_string(),
        "C3(+0.000, -1.000, +0.000)"
    );

    let tc3_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(3))
        .proper_power(1)
        .raw_axis(Vector3::new(0.0, 2.0, 0.0))
        .kind(TRROT)
        .rotation_group(SO3)
        .build()
        .unwrap();

    let tc3p2 = SymmetryOperation::builder()
        .generating_element(tc3_element)
        .power(2)
        .build()
        .unwrap();
    assert_eq!(
        tc3p2.to_symmetry_element().to_string(),
        "C3(+0.000, -1.000, +0.000)"
    );

    let c4_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(4))
        .proper_power(1)
        .raw_axis(Vector3::new(0.0, 2.0, -1.0))
        .kind(ROT)
        .rotation_group(SO3)
        .build()
        .unwrap();

    let c4p2 = SymmetryOperation::builder()
        .generating_element(c4_element)
        .power(2)
        .build()
        .unwrap();
    assert_eq!(c4p2.to_string(), "[C4(+0.000, +0.894, -0.447)]^2");
    assert_eq!(
        c4p2.to_symmetry_element().to_string(),
        "C2(+0.000, -0.894, +0.447)"
    );

    let tc4_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(4))
        .proper_power(1)
        .raw_axis(Vector3::new(0.0, 2.0, -1.0))
        .kind(TRROT)
        .rotation_group(SO3)
        .build()
        .unwrap();

    let tc4p3 = SymmetryOperation::builder()
        .generating_element(tc4_element)
        .power(3)
        .build()
        .unwrap();
    assert_eq!(tc4p3.to_string(), "[θ·C4(+0.000, +0.894, -0.447)]^3");
    assert_eq!(
        tc4p3.to_symmetry_element().to_string(),
        "θ·C4(+0.000, -0.894, +0.447)"
    );

    let c7_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(7))
        .proper_power(1)
        .raw_axis(Vector3::new(0.0, 2.0, -1.0))
        .kind(ROT)
        .rotation_group(SO3)
        .build()
        .unwrap();

    let c7p2 = SymmetryOperation::builder()
        .generating_element(c7_element.clone())
        .power(2)
        .build()
        .unwrap();
    assert_eq!(c7p2.to_string(), "[C7(+0.000, +0.894, -0.447)]^2");
    assert_eq!(
        c7p2.to_symmetry_element().to_string(),
        "C7^2(+0.000, +0.894, -0.447)"
    );

    let c7pm2 = SymmetryOperation::builder()
        .generating_element(c7_element.clone())
        .power(-2)
        .build()
        .unwrap();
    assert_eq!(c7pm2.to_string(), "[C7(+0.000, +0.894, -0.447)]^(-2)");
    assert_eq!(
        c7pm2.to_symmetry_element().to_string(),
        "C7^2(+0.000, -0.894, +0.447)"
    );

    let c7p4 = SymmetryOperation::builder()
        .generating_element(c7_element.clone())
        .power(4)
        .build()
        .unwrap();
    assert_eq!(c7p4.to_string(), "[C7(+0.000, +0.894, -0.447)]^4");
    assert_eq!(
        c7p4.to_symmetry_element().to_string(),
        "C7^3(+0.000, -0.894, +0.447)"
    );

    let c7p7 = SymmetryOperation::builder()
        .generating_element(c7_element)
        .power(7)
        .build()
        .unwrap();
    assert_eq!(c7p7.to_string(), "[C7(+0.000, +0.894, -0.447)]^7");
    assert_eq!(c7p7.to_symmetry_element().to_string(), "E");

    let tc7_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(7))
        .proper_power(1)
        .raw_axis(Vector3::new(0.0, 2.0, -1.0))
        .kind(TRROT)
        .rotation_group(SO3)
        .build()
        .unwrap();

    let tc7p2 = SymmetryOperation::builder()
        .generating_element(tc7_element.clone())
        .power(2)
        .build()
        .unwrap();
    assert_eq!(tc7p2.to_string(), "[θ·C7(+0.000, +0.894, -0.447)]^2");
    assert_eq!(
        tc7p2.to_symmetry_element().to_string(),
        "C7^2(+0.000, +0.894, -0.447)"
    );

    let tc7p5 = SymmetryOperation::builder()
        .generating_element(tc7_element)
        .power(5)
        .build()
        .unwrap();
    assert_eq!(tc7p5.to_string(), "[θ·C7(+0.000, +0.894, -0.447)]^5");
    assert_eq!(
        tc7p5.to_symmetry_element().to_string(),
        "θ·C7^2(+0.000, -0.894, +0.447)"
    );

    // ==================================
    // Improper symmetry operations (SO3)
    // ==================================
    let s3_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(3))
        .proper_power(1)
        .raw_axis(Vector3::new(0.0, 2.0, -1.0))
        .kind(SIG)
        .rotation_group(SO3)
        .build()
        .unwrap();

    let s3p2 = SymmetryOperation::builder()
        .generating_element(s3_element.clone())
        .power(2)
        .build()
        .unwrap();
    assert_eq!(s3p2.to_string(), "[S3(+0.000, +0.894, -0.447)]^2");
    assert_eq!(
        s3p2.to_symmetry_element().to_string(),
        "C3(+0.000, -0.894, +0.447)"
    );

    let s3p3 = SymmetryOperation::builder()
        .generating_element(s3_element.clone())
        .power(3)
        .build()
        .unwrap();
    assert_eq!(s3p3.to_string(), "[S3(+0.000, +0.894, -0.447)]^3");
    assert_eq!(
        s3p3.to_symmetry_element().to_string(),
        "σ(+0.000, -0.894, +0.447)"
    );

    let s3p4 = SymmetryOperation::builder()
        .generating_element(s3_element.clone())
        .power(4)
        .build()
        .unwrap();
    assert_eq!(s3p4.to_string(), "[S3(+0.000, +0.894, -0.447)]^4");
    assert_eq!(
        s3p4.to_symmetry_element().to_string(),
        "C3(+0.000, +0.894, -0.447)"
    );

    let s3p5 = SymmetryOperation::builder()
        .generating_element(s3_element.clone())
        .power(5)
        .build()
        .unwrap();
    assert_eq!(s3p5.to_string(), "[S3(+0.000, +0.894, -0.447)]^5");
    assert_eq!(
        s3p5.to_symmetry_element().to_string(),
        "S3(+0.000, -0.894, +0.447)"
    );

    let s3p6 = SymmetryOperation::builder()
        .generating_element(s3_element)
        .power(6)
        .build()
        .unwrap();
    assert_eq!(s3p6.to_string(), "[S3(+0.000, +0.894, -0.447)]^6");
    assert_eq!(s3p6.to_symmetry_element().to_string(), "E");

    let ts3_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(3))
        .proper_power(1)
        .raw_axis(Vector3::new(0.0, 2.0, -1.0))
        .kind(TRSIG)
        .rotation_group(SO3)
        .build()
        .unwrap();

    let ts3p2 = SymmetryOperation::builder()
        .generating_element(ts3_element.clone())
        .power(2)
        .build()
        .unwrap();
    assert_eq!(ts3p2.to_string(), "[θ·S3(+0.000, +0.894, -0.447)]^2");
    assert_eq!(
        ts3p2.to_symmetry_element().to_string(),
        "C3(+0.000, -0.894, +0.447)"
    );

    let ts3p3 = SymmetryOperation::builder()
        .generating_element(ts3_element)
        .power(3)
        .build()
        .unwrap();
    assert_eq!(ts3p3.to_string(), "[θ·S3(+0.000, +0.894, -0.447)]^3");
    assert_eq!(
        ts3p3.to_symmetry_element().to_string(),
        "θ·σ(+0.000, -0.894, +0.447)"
    );

    let sd5_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(5))
        .proper_power(1)
        .raw_axis(Vector3::new(0.0, 2.0, -1.0))
        .kind(INV)
        .rotation_group(SO3)
        .build()
        .unwrap();

    let sd5p2 = SymmetryOperation::builder()
        .generating_element(sd5_element.clone())
        .power(2)
        .build()
        .unwrap();
    assert_eq!(sd5p2.to_string(), "[Ṡ5(+0.000, +0.894, -0.447)]^2");
    assert_eq!(
        sd5p2.to_symmetry_element().to_string(),
        "C5^2(+0.000, +0.894, -0.447)"
    );

    let sd5p3 = SymmetryOperation::builder()
        .generating_element(sd5_element.clone())
        .power(3)
        .build()
        .unwrap();
    assert_eq!(sd5p3.to_string(), "[Ṡ5(+0.000, +0.894, -0.447)]^3");
    assert_eq!(
        sd5p3.to_symmetry_element().to_string(),
        "iC5^2(+0.000, -0.894, +0.447)"
    );

    let sd5p4 = SymmetryOperation::builder()
        .generating_element(sd5_element.clone())
        .power(4)
        .build()
        .unwrap();
    assert_eq!(sd5p4.to_string(), "[Ṡ5(+0.000, +0.894, -0.447)]^4");
    assert_eq!(
        sd5p4.to_symmetry_element().to_string(),
        "C5(+0.000, -0.894, +0.447)"
    );

    let sd5p5 = SymmetryOperation::builder()
        .generating_element(sd5_element.clone())
        .power(5)
        .build()
        .unwrap();
    assert_eq!(sd5p5.to_string(), "[Ṡ5(+0.000, +0.894, -0.447)]^5");
    assert_eq!(sd5p5.to_symmetry_element().to_string(), "i");

    let sd5p10 = SymmetryOperation::builder()
        .generating_element(sd5_element)
        .power(10)
        .build()
        .unwrap();
    assert_eq!(sd5p10.to_string(), "[Ṡ5(+0.000, +0.894, -0.447)]^10");
    assert_eq!(sd5p10.to_symmetry_element().to_string(), "E");

    let tsd5_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(5))
        .proper_power(1)
        .raw_axis(Vector3::new(0.0, 2.0, -1.0))
        .kind(TRINV)
        .rotation_group(SO3)
        .build()
        .unwrap();

    let tsd5p2 = SymmetryOperation::builder()
        .generating_element(tsd5_element.clone())
        .power(2)
        .build()
        .unwrap();
    assert_eq!(tsd5p2.to_string(), "[θ·Ṡ5(+0.000, +0.894, -0.447)]^2");
    assert_eq!(
        tsd5p2.to_symmetry_element().to_string(),
        "C5^2(+0.000, +0.894, -0.447)"
    );

    let tsd5p3 = SymmetryOperation::builder()
        .generating_element(tsd5_element)
        .power(3)
        .build()
        .unwrap();
    assert_eq!(tsd5p3.to_string(), "[θ·Ṡ5(+0.000, +0.894, -0.447)]^3");
    assert_eq!(
        tsd5p3.to_symmetry_element().to_string(),
        "θ·iC5^2(+0.000, -0.894, +0.447)"
    );

    // ================================
    // Proper symmetry operations (SU2)
    // ================================
    let c3_element_su2 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(3))
        .proper_power(1)
        .raw_axis(Vector3::new(0.0, 2.0, 0.0))
        .kind(ROT)
        .rotation_group(SU2_0)
        .build()
        .unwrap();

    let c3p2_su2 = SymmetryOperation::builder()
        .generating_element(c3_element_su2.clone())
        .power(2)
        .build()
        .unwrap();
    assert_eq!(
        c3p2_su2.to_symmetry_element().to_string(),
        "C3(QΣ)(+0.000, -1.000, +0.000)"
    );

    let c3p3_su2 = SymmetryOperation::builder()
        .generating_element(c3_element_su2.clone())
        .power(3)
        .build()
        .unwrap();
    assert_eq!(c3p3_su2.to_symmetry_element().to_string(), "E(QΣ)");

    let c3p4_su2 = SymmetryOperation::builder()
        .generating_element(c3_element_su2.clone())
        .power(4)
        .build()
        .unwrap();
    assert_eq!(
        c3p4_su2.to_symmetry_element().to_string(),
        "C3(QΣ)(+0.000, +1.000, +0.000)"
    );

    let c3p5_su2 = SymmetryOperation::builder()
        .generating_element(c3_element_su2.clone())
        .power(5)
        .build()
        .unwrap();
    assert_eq!(
        c3p5_su2.to_symmetry_element().to_string(),
        "C3(Σ)(+0.000, -1.000, +0.000)"
    );

    let c3p6_su2 = SymmetryOperation::builder()
        .generating_element(c3_element_su2)
        .power(6)
        .build()
        .unwrap();
    assert_eq!(c3p6_su2.to_symmetry_element().to_string(), "E(Σ)");

    let c4_element_su2 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(4))
        .proper_power(1)
        .raw_axis(Vector3::new(0.0, -1.0, 0.0))
        .kind(ROT)
        .rotation_group(SU2_0)
        .build()
        .unwrap();

    let c4p2_su2 = SymmetryOperation::builder()
        .generating_element(c4_element_su2.clone())
        .power(2)
        .build()
        .unwrap();
    assert_eq!(
        c4p2_su2.to_symmetry_element().to_string(),
        "C2(QΣ)(+0.000, +1.000, +0.000)"
    );

    let c4pm2_su2 = SymmetryOperation::builder()
        .generating_element(c4_element_su2)
        .power(-2)
        .build()
        .unwrap();
    assert_eq!(
        c4pm2_su2.to_symmetry_element().to_string(),
        "C2(Σ)(+0.000, +1.000, +0.000)"
    );

    let tc1_element_su2 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(1))
        .proper_power(1)
        .raw_axis(Vector3::new(0.0, 2.0, 1.0))
        .kind(TRROT)
        .rotation_group(SU2_0)
        .build()
        .unwrap();

    let tc1_su2 = SymmetryOperation::builder()
        .generating_element(tc1_element_su2.clone())
        .power(1)
        .build()
        .unwrap();
    assert_eq!(tc1_su2.to_symmetry_element().to_string(), "θ(Σ)");

    let tc1p2_su2 = SymmetryOperation::builder()
        .generating_element(tc1_element_su2.clone())
        .power(2)
        .build()
        .unwrap();
    assert_eq!(tc1p2_su2.to_symmetry_element().to_string(), "E(QΣ)");

    let tc1p3_su2 = SymmetryOperation::builder()
        .generating_element(tc1_element_su2.clone())
        .power(3)
        .build()
        .unwrap();
    assert_eq!(tc1p3_su2.to_symmetry_element().to_string(), "θ(QΣ)");

    let tc1p4_su2 = SymmetryOperation::builder()
        .generating_element(tc1_element_su2)
        .power(4)
        .build()
        .unwrap();
    assert_eq!(tc1p4_su2.to_symmetry_element().to_string(), "E(Σ)");

    let tc3_element_su2 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(3))
        .proper_power(1)
        .raw_axis(Vector3::new(1.0, 1.0, 1.0))
        .kind(TRROT)
        .rotation_group(SU2_0)
        .build()
        .unwrap();

    let tc3_su2 = SymmetryOperation::builder()
        .generating_element(tc3_element_su2.clone())
        .power(1)
        .build()
        .unwrap();
    assert_eq!(
        tc3_su2.to_symmetry_element().to_string(),
        "θ·C3(Σ)(+0.577, +0.577, +0.577)"
    );

    let tc3p2_su2 = SymmetryOperation::builder()
        .generating_element(tc3_element_su2)
        .power(2)
        .build()
        .unwrap();
    assert_eq!(
        tc3p2_su2.to_symmetry_element().to_string(),
        "C3(Σ)(-0.577, -0.577, -0.577)"
    );

    let tc4y_element_su2 = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(4))
        .proper_power(1)
        .raw_axis(Vector3::new(0.0, 1.0, 0.0))
        .kind(TRROT)
        .rotation_group(SU2_0)
        .build()
        .unwrap();

    let tc4y_su2 = SymmetryOperation::builder()
        .generating_element(tc4y_element_su2.clone())
        .power(1)
        .build()
        .unwrap();
    assert_eq!(
        tc4y_su2.to_symmetry_element().to_string(),
        "θ·C4(Σ)(+0.000, +1.000, +0.000)"
    );

    let tc4yp2_su2 = SymmetryOperation::builder()
        .generating_element(tc4y_element_su2.clone())
        .power(2)
        .build()
        .unwrap();
    assert_eq!(
        tc4yp2_su2.to_symmetry_element().to_string(),
        "C2(QΣ)(+0.000, +1.000, +0.000)"
    );

    let tc4yp3_su2 = SymmetryOperation::builder()
        .generating_element(tc4y_element_su2.clone())
        .power(3)
        .build()
        .unwrap();
    assert_eq!(
        tc4yp3_su2.to_symmetry_element().to_string(),
        "θ·C4(Σ)(+0.000, -1.000, +0.000)"
    );

    let tc4yp4_su2 = SymmetryOperation::builder()
        .generating_element(tc4y_element_su2.clone())
        .power(4)
        .build()
        .unwrap();
    assert_eq!(tc4yp4_su2.to_symmetry_element().to_string(), "E(QΣ)");

    let tc4yp5_su2 = SymmetryOperation::builder()
        .generating_element(tc4y_element_su2.clone())
        .power(5)
        .build()
        .unwrap();
    assert_eq!(
        tc4yp5_su2.to_symmetry_element().to_string(),
        "θ·C4(QΣ)(+0.000, +1.000, +0.000)"
    );

    let tc4yp6_su2 = SymmetryOperation::builder()
        .generating_element(tc4y_element_su2.clone())
        .power(6)
        .build()
        .unwrap();
    assert_eq!(
        tc4yp6_su2.to_symmetry_element().to_string(),
        "C2(Σ)(+0.000, +1.000, +0.000)"
    );

    let tc4yp7_su2 = SymmetryOperation::builder()
        .generating_element(tc4y_element_su2.clone())
        .power(7)
        .build()
        .unwrap();
    assert_eq!(
        tc4yp7_su2.to_symmetry_element().to_string(),
        "θ·C4(QΣ)(+0.000, -1.000, +0.000)"
    );

    let tc4yp8_su2 = SymmetryOperation::builder()
        .generating_element(tc4y_element_su2)
        .power(8)
        .build()
        .unwrap();
    assert_eq!(tc4yp8_su2.to_symmetry_element().to_string(), "E(Σ)");

    // let kc2x_element_su2 = SymmetryElement::builder()
    //     .threshold(1e-14)
    //     .proper_order(ElementOrder::Int(2))
    //     .proper_power(1)
    //     .raw_axis(Vector3::x())
    //     .kind(KROT)
    //     .rotation_group(SU2_0)
    //     .build()
    //     .unwrap();

    // let kc2x_su2 = SymmetryOperation::builder()
    //     .generating_element(kc2x_element_su2.clone())
    //     .power(1)
    //     .build()
    //     .unwrap();
    // assert_eq!(
    //     kc2x_su2.to_symmetry_element().to_string(),
    //     "K·C2(Σ)(+1.000, +0.000, +0.000)"
    // );

    // let kc2xp2_su2 = SymmetryOperation::builder()
    //     .generating_element(kc2x_element_su2.clone())
    //     .power(2)
    //     .build()
    //     .unwrap();
    // assert_eq!(kc2xp2_su2.to_symmetry_element().to_string(), "E(QΣ)");

    // let kc2xp3_su2 = SymmetryOperation::builder()
    //     .generating_element(kc2x_element_su2.clone())
    //     .power(3)
    //     .build()
    //     .unwrap();
    // assert_eq!(
    //     kc2xp3_su2.to_symmetry_element().to_string(),
    //     "K·C2(QΣ)(+1.000, +0.000, +0.000)"
    // );

    // let kc2xp4_su2 = SymmetryOperation::builder()
    //     .generating_element(kc2x_element_su2.clone())
    //     .power(4)
    //     .build()
    //     .unwrap();
    // assert_eq!(kc2xp4_su2.to_symmetry_element().to_string(), "E(Σ)");

    // let kc2y_element_su2 = SymmetryElement::builder()
    //     .threshold(1e-14)
    //     .proper_order(ElementOrder::Int(2))
    //     .proper_power(1)
    //     .raw_axis(Vector3::y())
    //     .kind(KROT)
    //     .rotation_group(SU2_0)
    //     .build()
    //     .unwrap();

    // let kc2y_su2 = SymmetryOperation::builder()
    //     .generating_element(kc2y_element_su2.clone())
    //     .power(1)
    //     .build()
    //     .unwrap();
    // assert_eq!(kc2y_su2.to_symmetry_element().to_string(), "θ(Σ)");

    // let kc2yp2_su2 = SymmetryOperation::builder()
    //     .generating_element(kc2y_element_su2.clone())
    //     .power(2)
    //     .build()
    //     .unwrap();
    // assert_eq!(kc2yp2_su2.to_symmetry_element().to_string(), "E(QΣ)");

    // let kc2yp3_su2 = SymmetryOperation::builder()
    //     .generating_element(kc2y_element_su2.clone())
    //     .power(3)
    //     .build()
    //     .unwrap();
    // assert_eq!(kc2yp3_su2.to_symmetry_element().to_string(), "θ(QΣ)");

    // let kc2yp4_su2 = SymmetryOperation::builder()
    //     .generating_element(kc2y_element_su2.clone())
    //     .power(4)
    //     .build()
    //     .unwrap();
    // assert_eq!(kc2yp4_su2.to_symmetry_element().to_string(), "E(Σ)");
}

#[test]
fn test_symmetry_operation_composition_time_reversal() {
    // ---
    // SO3
    // ---
    let tc1z_element_so3 = SymmetryElement::builder()
        .threshold(1e-12)
        .proper_order(ElementOrder::Int(1))
        .proper_power(1)
        .raw_axis(Vector3::z())
        .kind(TRROT)
        .rotation_group(SO3)
        .build()
        .unwrap();

    let tc1z_so3 = SymmetryOperation::builder()
        .generating_element(tc1z_element_so3)
        .power(1)
        .build()
        .unwrap();

    let tc1z_tc1z_so3 = (&tc1z_so3) * (&tc1z_so3);
    assert!(tc1z_tc1z_so3.is_identity());

    let tc1z_p2_so3 = (&tc1z_so3).pow(2);
    assert!(tc1z_p2_so3.is_identity());

    let tc1z_tc1z_tc1z_so3 = (&tc1z_tc1z_so3) * (&tc1z_so3);
    assert!(tc1z_tc1z_tc1z_so3.is_time_reversal());

    let tc1z_p3_so3 = (&tc1z_so3).pow(3);
    assert!(tc1z_p3_so3.is_time_reversal());

    let tc1z_tc1z_tc1z_tc1z_so3 = (&tc1z_tc1z_so3) * (&tc1z_tc1z_so3);
    assert!(tc1z_tc1z_tc1z_tc1z_so3.is_identity());

    let tc1z_p4_so3 = (&tc1z_so3).pow(4);
    assert!(tc1z_p4_so3.is_identity());

    let tc2z_element_so3 = SymmetryElement::builder()
        .threshold(1e-12)
        .proper_order(ElementOrder::Int(2))
        .proper_power(1)
        .raw_axis(Vector3::z())
        .kind(TRROT)
        .rotation_group(SO3)
        .build()
        .unwrap();

    let tc2z_so3 = SymmetryOperation::builder()
        .generating_element(tc2z_element_so3)
        .power(1)
        .build()
        .unwrap();

    let tc2z_tc2z_so3 = (&tc2z_so3) * (&tc2z_so3);
    assert!(tc2z_tc2z_so3.is_identity());

    let tc2z_p2_so3 = (&tc2z_so3).pow(2);
    assert!(tc2z_p2_so3.is_identity());

    let tc2z_tc2z_tc2z_so3 = (&tc2z_tc2z_so3) * (&tc2z_so3);
    assert!(!tc2z_tc2z_tc2z_so3.is_time_reversal());

    let tc2z_p3_so3 = (&tc2z_so3).pow(3);
    assert!(!tc2z_p3_so3.is_time_reversal());

    let tc2z_tc2z_tc2z_tc2z_so3 = (&tc2z_tc2z_so3) * (&tc2z_tc2z_so3);
    assert!(tc2z_tc2z_tc2z_tc2z_so3.is_identity());

    let tc2z_p4_so3 = (&tc2z_so3).pow(4);
    assert!(tc2z_p4_so3.is_identity());

    let tc3_element_so3 = SymmetryElement::builder()
        .threshold(1e-12)
        .proper_order(ElementOrder::Int(3))
        .proper_power(1)
        .raw_axis(Vector3::new(1.0, 1.0, 1.0))
        .kind(TRROT)
        .rotation_group(SO3)
        .build()
        .unwrap();

    let tc3_so3 = SymmetryOperation::builder()
        .generating_element(tc3_element_so3)
        .power(1)
        .build()
        .unwrap();

    let tc3_tc3_so3 = (&tc3_so3) * (&tc3_so3);
    assert!(!tc3_tc3_so3.is_identity());

    let tc3_p2_so3 = (&tc3_so3).pow(2);
    assert!(!tc3_p2_so3.is_identity());

    let tc3_tc3_tc3_so3 = (&tc3_tc3_so3) * (&tc3_so3);
    assert!(tc3_tc3_tc3_so3.is_time_reversal());

    let tc3_p3_so3 = (&tc3_so3).pow(3);
    assert!(tc3_p3_so3.is_time_reversal());

    // ---
    // SU2
    // ---
    let tc1z_element = SymmetryElement::builder()
        .threshold(1e-12)
        .proper_order(ElementOrder::Int(1))
        .proper_power(1)
        .raw_axis(Vector3::z())
        .kind(TRROT)
        .rotation_group(SU2_0)
        .build()
        .unwrap();

    let tc1z = SymmetryOperation::builder()
        .generating_element(tc1z_element)
        .power(1)
        .build()
        .unwrap();
    assert!(!tc1z.is_su2_class_1());
    assert!(tc1z.is_time_reversal());
    assert_eq!(tc1z.to_string(), "θ(Σ)");
    assert_eq!(tc1z.get_abbreviated_symbol(), "θ(Σ)");

    let tc1z_p2 = (&tc1z).pow(2);
    assert!(tc1z_p2.is_su2_class_1());
    assert!(!tc1z_p2.is_identity());
    assert_eq!(tc1z_p2.to_string(), "[θ(Σ)]^2");
    assert_eq!(tc1z_p2.get_abbreviated_symbol(), "E(QΣ)");

    let tc1z_tc1z = (&tc1z) * (&tc1z);
    assert!(tc1z_tc1z.is_su2_class_1());
    assert!(!tc1z_tc1z.is_identity());
    assert_eq!(tc1z_tc1z.to_string(), "E(QΣ)");
    assert_eq!(tc1z_tc1z.get_abbreviated_symbol(), "E(QΣ)");

    let tc1z_p3 = (&tc1z).pow(3);
    assert!(tc1z_p3.is_su2_class_1());
    assert!(!tc1z_p3.is_time_reversal());
    assert_eq!(tc1z_p3.to_string(), "[θ(Σ)]^3");
    assert_eq!(tc1z_p3.get_abbreviated_symbol(), "θ(QΣ)");

    let tc1z_tc1z_tc1z = (&tc1z_p2) * &tc1z;
    assert!(tc1z_tc1z_tc1z.is_su2_class_1());
    assert!(!tc1z_tc1z_tc1z.is_time_reversal());
    assert_eq!(tc1z_tc1z_tc1z.to_string(), "θ(QΣ)");
    assert_eq!(tc1z_tc1z_tc1z.get_abbreviated_symbol(), "θ(QΣ)");

    let tc1z_p4 = (&tc1z).pow(4);
    assert!(!tc1z_p4.is_su2_class_1());
    assert!(tc1z_p4.is_identity());
    assert_eq!(tc1z_p4.to_string(), "[θ(Σ)]^4");
    assert_eq!(tc1z_p4.get_abbreviated_symbol(), "E(Σ)");

    let tc1z_tc1z_tc1z_tc1z = (&tc1z_p2) * (&tc1z_p2);
    assert!(!tc1z_tc1z_tc1z_tc1z.is_su2_class_1());
    assert!(tc1z_tc1z_tc1z_tc1z.is_identity());
    assert_eq!(tc1z_tc1z_tc1z_tc1z.to_string(), "E(Σ)");
    assert_eq!(tc1z_tc1z_tc1z_tc1z.get_abbreviated_symbol(), "E(Σ)");

    let tc2z_element = SymmetryElement::builder()
        .threshold(1e-12)
        .proper_order(ElementOrder::Int(2))
        .proper_power(1)
        .raw_axis(Vector3::z())
        .kind(TRROT)
        .rotation_group(SU2_0)
        .build()
        .unwrap();

    let tc2z = SymmetryOperation::builder()
        .generating_element(tc2z_element)
        .power(1)
        .build()
        .unwrap();
    assert!(!tc2z.is_time_reversal());
    assert_eq!(tc2z.to_string(), "θ·C2(Σ)(+0.000, +0.000, +1.000)");
    assert_eq!(tc2z.get_abbreviated_symbol(), "θ·C2(Σ)");

    let tc2z_p2 = (&tc2z).pow(2);
    assert!(!tc2z_p2.is_su2_class_1());
    assert!(tc2z_p2.is_identity());
    assert_eq!(tc2z_p2.to_string(), "[θ·C2(Σ)(+0.000, +0.000, +1.000)]^2");
    assert_eq!(tc2z_p2.get_abbreviated_symbol(), "E(Σ)");

    let tc2z_tc2z = (&tc2z) * (&tc2z);
    assert!(!tc2z_tc2z.is_su2_class_1());
    assert!(tc2z_tc2z.is_identity());
    assert_eq!(tc2z_tc2z.to_string(), "E(Σ)");
    assert_eq!(tc2z_tc2z.get_abbreviated_symbol(), "E(Σ)");

    let tc2z_p3 = (&tc2z).pow(3);
    assert!(!tc2z_p3.is_su2_class_1());
    assert!(!tc2z_p3.is_time_reversal());
    assert_eq!(tc2z_p3.to_string(), "[θ·C2(Σ)(+0.000, +0.000, +1.000)]^3");
    assert_eq!(tc2z_p3.get_abbreviated_symbol(), "θ·C2(Σ)");

    let tc2z_tc2z_tc2z = (&tc2z_p2) * &tc2z;
    assert!(!tc2z_tc2z_tc2z.is_su2_class_1());
    assert!(!tc2z_tc2z_tc2z.is_time_reversal());
    assert_eq!(
        tc2z_tc2z_tc2z.to_string(),
        "θ·C2(Σ)(+0.000, +0.000, +1.000)"
    );
    assert_eq!(tc2z_tc2z_tc2z.get_abbreviated_symbol(), "θ·C2(Σ)");

    let tc2z_p4 = (&tc2z).pow(4);
    assert!(!tc2z_p4.is_su2_class_1());
    assert!(tc2z_p4.is_identity());
    assert_eq!(tc2z_p4.to_string(), "[θ·C2(Σ)(+0.000, +0.000, +1.000)]^4");
    assert_eq!(tc2z_p4.get_abbreviated_symbol(), "E(Σ)");

    let tc2z_tc2z_tc2z_tc2z = (&tc2z_p2) * (&tc2z_p2);
    assert!(!tc2z_tc2z_tc2z_tc2z.is_su2_class_1());
    assert!(tc2z_tc2z_tc2z_tc2z.is_identity());
    assert_eq!(tc2z_tc2z_tc2z_tc2z.to_string(), "E(Σ)");
    assert_eq!(tc2z_tc2z_tc2z_tc2z.get_abbreviated_symbol(), "E(Σ)");

    let tc3_element = SymmetryElement::builder()
        .threshold(1e-12)
        .proper_order(ElementOrder::Int(3))
        .proper_power(1)
        .raw_axis(Vector3::new(1.0, 1.0, 1.0))
        .kind(TRROT)
        .rotation_group(SU2_0)
        .build()
        .unwrap();

    let tc3 = SymmetryOperation::builder()
        .generating_element(tc3_element)
        .power(1)
        .build()
        .unwrap();
    assert!(!tc3.is_time_reversal());
    assert_eq!(tc3.to_string(), "θ·C3(Σ)(+0.577, +0.577, +0.577)");
    assert_eq!(tc3.get_abbreviated_symbol(), "θ·C3(Σ)");

    let tc3_p2 = (&tc3).pow(2);
    assert!(!tc3_p2.is_su2_class_1());
    assert_eq!(tc3_p2.to_string(), "[θ·C3(Σ)(+0.577, +0.577, +0.577)]^2");
    assert_eq!(tc3_p2.get_abbreviated_symbol(), "C3^(-1)(Σ)");

    let tc3_tc3 = (&tc3) * (&tc3);
    assert!(!tc3_tc3.is_su2_class_1());
    assert_eq!(tc3_tc3.to_string(), "C3(Σ)(-0.577, -0.577, -0.577)");
    assert_eq!(tc3_tc3.get_abbreviated_symbol(), "C3^(-1)(Σ)");

    let tc3_p3 = (&tc3).pow(3);
    assert!(!tc3_p3.is_su2_class_1());
    assert!(tc3_p3.is_time_reversal());
    assert_eq!(tc3_p3.to_string(), "[θ·C3(Σ)(+0.577, +0.577, +0.577)]^3");
    assert_eq!(tc3_p3.get_abbreviated_symbol(), "θ(Σ)");

    let tc3_tc3_tc3 = (&tc3_p2) * &tc3;
    assert!(!tc3_tc3_tc3.is_su2_class_1());
    assert!(tc3_tc3_tc3.is_time_reversal());
    assert_eq!(tc3_tc3_tc3.to_string(), "θ(Σ)");
    assert_eq!(tc3_tc3_tc3.get_abbreviated_symbol(), "θ(Σ)");

    let tc3_p4 = (&tc3).pow(4);
    assert!(tc3_p4.is_su2_class_1());
    assert_eq!(tc3_p4.to_string(), "[θ·C3(Σ)(+0.577, +0.577, +0.577)]^4");
    assert_eq!(tc3_p4.get_abbreviated_symbol(), "C3(QΣ)");

    let tc3_tc3_tc3_tc3 = (&tc3_p2) * (&tc3_p2);
    assert!(tc3_tc3_tc3_tc3.is_su2_class_1());
    assert_eq!(
        tc3_tc3_tc3_tc3.to_string(),
        "C3(QΣ)(+0.577, +0.577, +0.577)"
    );
    assert_eq!(tc3_tc3_tc3_tc3.get_abbreviated_symbol(), "C3(QΣ)");

    let c2z_element = SymmetryElement::builder()
        .threshold(1e-12)
        .proper_order(ElementOrder::Int(2))
        .proper_power(1)
        .raw_axis(Vector3::z())
        .kind(ROT)
        .rotation_group(SU2_0)
        .build()
        .unwrap();

    let c2z = SymmetryOperation::builder()
        .generating_element(c2z_element.clone())
        .power(1)
        .build()
        .unwrap();
    let c2zpm1 = SymmetryOperation::builder()
        .generating_element(c2z_element)
        .power(-1)
        .build()
        .unwrap();

    let tc2z_c2z = (&tc2z) * (&c2z);
    assert!(!tc2z_c2z.is_time_reversal());
    assert_eq!(tc2z_c2z.to_string(), "θ(QΣ)");
    assert_eq!(format!("{tc2z_c2z:?}"), "θ·C1(QΣ)(+0.000, +0.000, +1.000)");
    assert_eq!(tc2z_c2z.get_abbreviated_symbol(), "θ(QΣ)");

    let tc2z_c2zpm1 = (&tc2z) * (&c2zpm1);
    assert!(tc2z_c2zpm1.is_time_reversal());
    assert_eq!(tc2z_c2zpm1.to_string(), "θ(Σ)");
    assert_eq!(
        format!("{tc2z_c2zpm1:?}"),
        "θ·C1(Σ)(+0.000, +0.000, +1.000)"
    );
    assert_eq!(tc2z_c2zpm1.get_abbreviated_symbol(), "θ(Σ)");

    let tc3y_element = SymmetryElement::builder()
        .threshold(1e-12)
        .proper_order(ElementOrder::Int(3))
        .proper_power(1)
        .raw_axis(-Vector3::y())
        .kind(TRROT)
        .rotation_group(SU2_0)
        .build()
        .unwrap();

    let tc3y = SymmetryOperation::builder()
        .generating_element(tc3y_element)
        .power(1)
        .build()
        .unwrap();
    assert!(!tc3y.is_time_reversal());
    assert_eq!(tc3y.to_string(), "θ·C3(Σ)(+0.000, -1.000, +0.000)");
    assert_eq!(tc3y.get_abbreviated_symbol(), "θ·C3^(-1)(Σ)");

    let tc3y_p3 = tc3y.pow(3);
    assert!(tc3y_p3.is_time_reversal());
    assert_eq!(tc3y_p3.to_string(), "[θ·C3(Σ)(+0.000, -1.000, +0.000)]^3");
    assert_eq!(tc3y_p3.to_symmetry_element().to_string(), "θ(Σ)");
    assert_eq!(tc3y_p3.get_abbreviated_symbol(), "θ(Σ)");

    let c1_isr_element = SymmetryElement::builder()
        .threshold(1e-12)
        .proper_order(ElementOrder::Int(1))
        .proper_power(1)
        .raw_axis(Vector3::new(2.0, 1.0, 2.0))
        .kind(ROT)
        .rotation_group(SU2_1)
        .build()
        .unwrap();

    let c1_isr = SymmetryOperation::builder()
        .generating_element(c1_isr_element)
        .power(1)
        .build()
        .unwrap();

    let tc2z_c2z_c1_isr = (&tc2z_c2z) * (&c1_isr);
    assert!(tc2z_c2z_c1_isr.is_time_reversal());
    assert_eq!(tc2z_c2z_c1_isr.to_string(), "θ(Σ)");
    assert_eq!(tc2z_c2z_c1_isr.get_abbreviated_symbol(), "θ(Σ)");
}
