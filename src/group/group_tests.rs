use std::panic;

use approx;
// use env_logger;
use itertools::Itertools;
use nalgebra::Vector3;
use num_traits::Pow;

use crate::aux::molecule::Molecule;
use crate::aux::template_molecules;
use crate::group::class::ClassProperties;
use crate::group::symmetry_group::SymmetryGroupProperties;
use crate::group::{
    Group, GroupProperties, GroupType, MagneticRepresentedGroup, UnitaryRepresentedGroup, BWGRP,
    GRGRP, ORGRP,
};
use crate::symmetry::symmetry_core::{PreSymmetry, Symmetry};
use crate::symmetry::symmetry_element::symmetry_operation::SpecialSymmetryTransformation;
use crate::symmetry::symmetry_element::{SymmetryElement, SymmetryOperation, ROT};
use crate::symmetry::symmetry_element_order::ElementOrder;

const ROOT: &str = env!("CARGO_MANIFEST_DIR");

#[test]
fn test_abstract_group_creation() {
    // =============
    // Cyclic groups
    // =============
    let c5_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::Int(5))
        .proper_power(1)
        .axis(Vector3::new(1.0, 1.0, 2.0))
        .kind(ROT)
        .build()
        .unwrap();

    let c5 = SymmetryOperation::builder()
        .generating_element(c5_element)
        .power(1)
        .build()
        .unwrap();

    let group_c5 = Group::<SymmetryOperation>::new("C5", (0..5).map(|k| (&c5).pow(k)).collect());
    let mut elements = group_c5.elements().keys();
    for i in 0..5 {
        let op = elements.next().unwrap();
        assert_eq!(*op, (&c5).pow(i));
    }
    assert!(group_c5.is_abelian());

    let c29_element = SymmetryElement::builder()
        .threshold(1e-12)
        .proper_order(ElementOrder::Int(29))
        .proper_power(1)
        .axis(Vector3::new(1.0, 0.5, 2.0))
        .kind(ROT)
        .build()
        .unwrap();

    let c29 = SymmetryOperation::builder()
        .generating_element(c29_element)
        .power(1)
        .build()
        .unwrap();

    let group_c29 =
        Group::<SymmetryOperation>::new("C29", (0..29).map(|k| (&c29).pow(k)).collect());
    let mut elements = group_c29.elements().keys();
    for i in 0..29 {
        let op = elements.next().unwrap();
        assert_eq!(*op, (&c29).pow(i));
    }
    assert!(group_c29.is_abelian());
}
