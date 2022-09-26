use nalgebra::Vector3;
use num_traits::Pow;

use crate::group::Group;
use crate::symmetry::symmetry_element::{SymmetryElement, SymmetryElementKind, SymmetryOperation};
use crate::symmetry::symmetry_element_order::ElementOrder;

#[test]
fn test_group_creation() {
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

    let c5p0 = c5.pow(0);
    let c5p1 = c5.pow(1);
    let c5p2 = c5.pow(2);
    let c5p3 = c5.pow(3);
    let c5p4 = c5.pow(4);

    let group = Group::<SymmetryOperation>::new("C5", vec![c5p0, c5p1, c5p2, c5p3, c5p4]);
    let mut elements = group.elements.keys();
    for i in 0usize..5usize {
        let op = elements.next().unwrap();
        assert_eq!(*op, c5.pow(i as i32));
    }
    println!("{}", group.cayley_table.unwrap());
}
