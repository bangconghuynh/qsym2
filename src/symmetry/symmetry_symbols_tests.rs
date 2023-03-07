use std::collections::HashSet;

use nalgebra::Vector3;

use crate::symmetry::symmetry_element::symmetry_operation::{
    SpecialSymmetryTransformation, SymmetryOperation,
};
use crate::symmetry::symmetry_element::{
    RotationGroup, SymmetryElement, SymmetryElementKind,
};
use crate::symmetry::symmetry_element_order::ElementOrder;
use crate::symmetry::symmetry_symbols::{
    MullikenIrcorepSymbol, MullikenIrrepSymbol, SymmetryClassSymbol,
};

#[test]
fn test_symmetry_symbols_mulliken() {
    let a = MullikenIrrepSymbol::new("A").unwrap();
    assert_eq!(a.to_string(), "|A|");

    let a2 = MullikenIrrepSymbol::new("||A||").unwrap();
    assert_eq!(a2.to_string(), "|A|");
    assert_eq!(a, a2);

    let b1dash = MullikenIrrepSymbol::new("||B|^(')_(1)|").unwrap();
    assert_eq!(b1dash.to_string(), "|B|^(')_(1)");
    assert_ne!(a, b1dash);

    let t1g = MullikenIrrepSymbol::new("||T|_(1g)|").unwrap();
    assert_eq!(t1g.to_string(), "|T|_(1g)");
}

#[test]
fn test_symmetry_symbols_class() {
    let c3_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::new(3.0, 1e-14))
        .proper_power(1)
        .raw_axis(Vector3::new(1.0, 1.0, 1.0))
        .kind(SymmetryElementKind::Proper(false))
        .rotationgroup(RotationGroup::SO3)
        .build()
        .unwrap();
    let c3 = SymmetryOperation::builder()
        .generating_element(c3_element.clone())
        .power(1)
        .build()
        .unwrap();

    let c3_cls = SymmetryClassSymbol::new("2||C3||", Some(c3)).unwrap();
    assert_eq!(c3_cls.to_string(), "2|C3|");
    assert!(c3_cls.is_proper());

    let i_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::new(2.0, 1e-14))
        .proper_power(1)
        .raw_axis(Vector3::new(1.0, 1.0, 1.0))
        .kind(SymmetryElementKind::ImproperMirrorPlane(false))
        .rotationgroup(RotationGroup::SO3)
        .build()
        .unwrap();
    let i = SymmetryOperation::builder()
        .generating_element(i_element.clone())
        .power(1)
        .build()
        .unwrap();

    let i_cls = SymmetryClassSymbol::new("1||i||", Some(i)).unwrap();
    assert_eq!(i_cls.to_string(), "|i|");
    assert!(!i_cls.is_proper());
    assert!(i_cls.is_inversion());
    assert!(!i_cls.is_time_reversal());

    let s_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::new(1.0, 1e-14))
        .proper_power(1)
        .raw_axis(Vector3::new(1.0, 1.0, 1.0))
        .kind(SymmetryElementKind::ImproperMirrorPlane(false))
        .rotationgroup(RotationGroup::SO3)
        .build()
        .unwrap();
    let s = SymmetryOperation::builder()
        .generating_element(s_element.clone())
        .power(1)
        .build()
        .unwrap();

    let s_cls = SymmetryClassSymbol::new("1||σ|_(h)|", Some(s)).unwrap();
    assert_eq!(s_cls.to_string(), "|σ|_(h)");
    assert!(!s_cls.is_proper());
    assert!(s_cls.is_reflection());
    assert!(!s_cls.is_time_reversal());
}

#[test]
fn test_symmetry_symbols_mulliken_ircorep_hashability() {
    let d1 = MullikenIrcorepSymbol::new("||E|_(g)| + ||T|_(2g)|").unwrap();
    let d2 = MullikenIrcorepSymbol::new("||T|_(2g)| + ||E|_(g)|").unwrap();
    let d3 = MullikenIrcorepSymbol::new("3||A|_(2g)| + 4||A|_(1g)|").unwrap();

    assert_eq!(format!("{d1}").as_str(), "D[|E|_(g) ⊕ |T|_(2g)]");
    assert_eq!(format!("{d2}").as_str(), "D[|E|_(g) ⊕ |T|_(2g)]");
    assert_eq!(format!("{d3}").as_str(), "D[4|A|_(1g) ⊕ 3|A|_(2g)]");

    assert_eq!(d1, d2);
    assert_ne!(d1, d3);

    let mut ds = HashSet::<MullikenIrcorepSymbol>::new();
    ds.insert(d1);
    assert_eq!(ds.len(), 1);
    ds.insert(d2);
    assert_eq!(ds.len(), 1);
    ds.insert(d3);
    assert_eq!(ds.len(), 2);

    let d4 = MullikenIrcorepSymbol::new("||T|_(2g)| + ||A|_(2g)| + ||A|_(1g)|").unwrap();
    let d5 = MullikenIrcorepSymbol::new("||A|_(1g)| + ||A|_(2g)| + ||T|_(2g)|").unwrap();
    assert_eq!(d4, d5);
    ds.insert(d4);
    assert_eq!(ds.len(), 3);
    ds.insert(d5);
    assert_eq!(ds.len(), 3);
}
