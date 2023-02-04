use nalgebra::Vector3;

use crate::symmetry::symmetry_element::symmetry_operation::{
    SpecialSymmetryTransformation, SymmetryOperation,
};
use crate::symmetry::symmetry_element::{SymmetryElement, SymmetryElementKind};
use crate::symmetry::symmetry_element_order::ElementOrder;
use crate::symmetry::symmetry_symbols::{ClassSymbol, MullikenIrrepSymbol};

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
        .axis(Vector3::new(1.0, 1.0, 1.0))
        .kind(SymmetryElementKind::Proper(false))
        .build()
        .unwrap();
    let c3 = SymmetryOperation::builder()
        .generating_element(c3_element.clone())
        .power(1)
        .build()
        .unwrap();

    let c3_cls = ClassSymbol::new("2||C3||", Some(c3)).unwrap();
    assert_eq!(c3_cls.to_string(), "2|C3|");
    assert!(c3_cls.is_proper());

    let i_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::new(2.0, 1e-14))
        .proper_power(1)
        .axis(Vector3::new(1.0, 1.0, 1.0))
        .kind(SymmetryElementKind::ImproperMirrorPlane(false))
        .build()
        .unwrap();
    let i = SymmetryOperation::builder()
        .generating_element(i_element.clone())
        .power(1)
        .build()
        .unwrap();

    let i_cls = ClassSymbol::new("1||i||", Some(i)).unwrap();
    assert_eq!(i_cls.to_string(), "|i|");
    assert!(!i_cls.is_proper());
    assert!(i_cls.is_inversion());
    assert!(!i_cls.is_time_reversal());

    let s_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::new(1.0, 1e-14))
        .proper_power(1)
        .axis(Vector3::new(1.0, 1.0, 1.0))
        .kind(SymmetryElementKind::ImproperMirrorPlane(false))
        .build()
        .unwrap();
    let s = SymmetryOperation::builder()
        .generating_element(s_element.clone())
        .power(1)
        .build()
        .unwrap();

    let s_cls = ClassSymbol::new("1||σ|_(h)|", Some(s)).unwrap();
    assert_eq!(s_cls.to_string(), "|σ|_(h)");
    assert!(!s_cls.is_proper());
    assert!(s_cls.is_reflection());
    assert!(!s_cls.is_time_reversal());
}
