use nalgebra::Vector3;
use num_traits::Pow;

use crate::group::Group;
use crate::symmetry::symmetry_element::{SymmetryElement, SymmetryElementKind, SymmetryOperation};
use crate::symmetry::symmetry_element_order::ElementOrder;

#[test]
fn test_group_creation() {
    // =============
    // Cyclic groups
    // =============
    let c5_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(5))
        .proper_power(1)
        .axis(Vector3::new(1.0, 1.0, 2.0))
        .kind(SymmetryElementKind::Proper)
        .build()
        .unwrap();

    let c5 = SymmetryOperation::builder()
        .generating_element(c5_element.clone())
        .power(1)
        .build()
        .unwrap();

    let group_c5 = Group::<SymmetryOperation>::new("C5", (0..5).map(|k| c5.pow(k)).collect());
    let mut elements = group_c5.elements.keys();
    for i in 0..5 {
        let op = elements.next().unwrap();
        assert_eq!(*op, c5.pow(i));
    }
    let ctb_c5 = group_c5.cayley_table.unwrap();
    assert_eq!(ctb_c5, ctb_c5.t());

    let c29_element = SymmetryElement::builder()
        .threshold(1e-12)
        .proper_order(ElementOrder::Int(29))
        .proper_power(1)
        .axis(Vector3::new(1.0, 0.5, 2.0))
        .kind(SymmetryElementKind::Proper)
        .build()
        .unwrap();

    let c29 = SymmetryOperation::builder()
        .generating_element(c29_element.clone())
        .power(1)
        .build()
        .unwrap();

    let group_c29 = Group::<SymmetryOperation>::new("C29", (0..29).map(|k| c29.pow(k)).collect());
    let mut elements = group_c29.elements.keys();
    for i in 0..29 {
        let op = elements.next().unwrap();
        assert_eq!(*op, c29.pow(i));
    }
    let ctb_c29 = group_c29.cayley_table.unwrap();
    assert_eq!(ctb_c29, ctb_c29.t());
}
