use nalgebra::Vector3;

use crate::chartab::symmetry_symbols::{MullikenIrrepSymbol, ClassSymbol};
use crate::symmetry::symmetry_element_order::ElementOrder;
use crate::symmetry::symmetry_element::{
    SymmetryElement, SymmetryElementKind
};
use crate::symmetry::symmetry_element::symmetry_operation::{
    SpecialSymmetryTransformation, SymmetryOperation,
};


#[test]
fn test_symmetry_symbols_mulliken() {
    let a = MullikenIrrepSymbol::new("A").unwrap();
    assert_eq!(format!("{}", a), "|A|");

    let a2 = MullikenIrrepSymbol::new("||A||").unwrap();
    assert_eq!(format!("{}", a2), "|A|");
    assert_eq!(a, a2);

    let b1dash = MullikenIrrepSymbol::new("||B|^(')_(1)|").unwrap();
    assert_eq!(format!("{}", b1dash), "|B|^(')_(1)");
    assert_ne!(a, b1dash);

    let t1g = MullikenIrrepSymbol::new("||T|_(1g)|").unwrap();
    assert_eq!(format!("{}", t1g), "|T|_(1g)");
}

#[test]
fn test_symmetry_symbols_class() {
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

    let c3_cls = ClassSymbol::new("2||C3||", c3).unwrap();
    assert_eq!(format!("{}", c3_cls), "2|C3|");
    assert!(c3_cls.is_proper());

    let i_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::new(2.0, 1e-14))
        .proper_power(1)
        .axis(Vector3::new(1.0, 1.0, 1.0))
        .kind(SymmetryElementKind::ImproperMirrorPlane)
        .build()
        .unwrap();
    let i = SymmetryOperation::builder()
        .generating_element(i_element.clone())
        .power(1)
        .build()
        .unwrap();

    let i_cls = ClassSymbol::new("||i||", i).unwrap();
    assert_eq!(format!("{}", i_cls), "|i|");
    assert!(!i_cls.is_proper());
    assert!(i_cls.is_inversion());
    assert!(!i_cls.is_time_reversal());

    let s_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::new(1.0, 1e-14))
        .proper_power(1)
        .axis(Vector3::new(1.0, 1.0, 1.0))
        .kind(SymmetryElementKind::ImproperMirrorPlane)
        .build()
        .unwrap();
    let s = SymmetryOperation::builder()
        .generating_element(s_element.clone())
        .power(1)
        .build()
        .unwrap();

    let s_cls = ClassSymbol::new("||σ|_(h)|", s).unwrap();
    assert_eq!(format!("{}", s_cls), "|σ|_(h)");
    assert!(!s_cls.is_proper());
    assert!(s_cls.is_reflection());
    assert!(!s_cls.is_time_reversal());
}
