use std::panic;

use approx;
// use env_logger;
use itertools::Itertools;
use nalgebra::Vector3;
use num_traits::Pow;

use crate::auxiliary::molecule::Molecule;
use crate::auxiliary::template_molecules;
use crate::group::class::ClassProperties;
use crate::group::{
    EagerGroup, GroupProperties, GroupType, MagneticRepresentedGroup, UnitaryRepresentedGroup,
    BWGRP, BWGRP2, GRGRP, GRGRP2, ORGRP, ORGRP2,
};
use crate::permutation::IntoPermutation;
use crate::symmetry::symmetry_core::{PreSymmetry, Symmetry};
use crate::symmetry::symmetry_element::symmetry_operation::SpecialSymmetryTransformation;
use crate::symmetry::symmetry_element::{RotationGroup, SymmetryElement, SymmetryOperation, ROT};
use crate::symmetry::symmetry_element_order::ElementOrder;
use crate::symmetry::symmetry_group::SymmetryGroupProperties;

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
        .raw_axis(Vector3::new(1.0, 1.0, 2.0))
        .kind(ROT)
        .rotation_group(RotationGroup::SO3)
        .build()
        .unwrap();

    let c5 = SymmetryOperation::builder()
        .generating_element(c5_element)
        .power(1)
        .build()
        .unwrap();

    let group_c5 =
        EagerGroup::<SymmetryOperation>::new("C5", (0..5).map(|k| (&c5).pow(k)).collect()).unwrap();
    let mut elements = group_c5.elements().iter();
    for i in 0..5 {
        let op = elements.next().unwrap();
        assert_eq!(*op, (&c5).pow(i));
    }
    assert!(group_c5.is_abelian());

    let c29_element = SymmetryElement::builder()
        .threshold(1e-12)
        .proper_order(ElementOrder::Int(29))
        .proper_power(1)
        .raw_axis(Vector3::new(1.0, 0.5, 2.0))
        .kind(ROT)
        .rotation_group(RotationGroup::SO3)
        .build()
        .unwrap();

    let c29 = SymmetryOperation::builder()
        .generating_element(c29_element)
        .power(1)
        .build()
        .unwrap();

    let group_c29 =
        EagerGroup::<SymmetryOperation>::new("C29", (0..29).map(|k| (&c29).pow(k)).collect())
            .unwrap();
    let mut elements = group_c29.elements().iter();
    for i in 0..29 {
        let op = elements.next().unwrap();
        assert_eq!(*op, (&c29).pow(i));
    }
    assert!(group_c29.is_abelian());
}

#[test]
fn test_ur_group_from_molecular_symmetry() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/nh3.xyz");
    let mol = Molecule::from_xyz(&path, 1e-6);
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-6)
        .molecule(&mol)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false).unwrap();
    let group = UnitaryRepresentedGroup::from_molecular_symmetry(&sym, None).unwrap();
    assert_eq!(group.name(), "C3v".to_string());
    assert_eq!(group.order(), 6);
    assert_eq!(group.class_number(), 3);
}

#[test]
fn test_ur_group_element_to_conjugacy_class() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/cpnico.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    let presym = PreSymmetry::builder()
        .moi_threshold(thresh)
        .molecule(&mol)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false).unwrap();
    let group = UnitaryRepresentedGroup::from_molecular_symmetry(&sym, None).unwrap();
    assert_eq!(group.name(), "C5v".to_string());
    assert_eq!(group.order(), 10);
    assert_eq!(group.class_number(), 4);

    for element_i in 0..group.order() {
        let class_i = group.get_cc_of_element_index(element_i).unwrap();
        assert!(group.get_cc_index(class_i).unwrap().contains(&element_i));
    }
}

#[test]
fn test_ur_group_element_sort() {
    // H2O in yz-plane, with C2 axis along z - C2v
    let path: String = format!("{}{}", ROOT, "/tests/xyz/water_z.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    let presym = PreSymmetry::builder()
        .moi_threshold(thresh)
        .molecule(&mol)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false).unwrap();
    let group = UnitaryRepresentedGroup::from_molecular_symmetry(&sym, None).unwrap();
    approx::assert_relative_eq!(
        *group
            .elements()
            .get_index(2)
            .unwrap()
            .generating_element
            .raw_axis(),
        Vector3::new(0.0, 1.0, 0.0)
    );
    approx::assert_relative_eq!(
        *group
            .elements()
            .get_index(3)
            .unwrap()
            .generating_element
            .raw_axis(),
        Vector3::new(1.0, 0.0, 0.0)
    );

    // B2H6 - D2h
    let path: String = format!("{}{}", ROOT, "/tests/xyz/b2h6.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    let presym = PreSymmetry::builder()
        .moi_threshold(thresh)
        .molecule(&mol)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false).unwrap();
    let group = UnitaryRepresentedGroup::from_molecular_symmetry(&sym, None).unwrap();
    approx::assert_relative_eq!(
        *group
            .elements()
            .get_index(1)
            .unwrap()
            .generating_element
            .raw_axis(),
        Vector3::new(0.0, 0.0, 1.0)
    );
    approx::assert_relative_eq!(
        *group
            .elements()
            .get_index(2)
            .unwrap()
            .generating_element
            .raw_axis(),
        Vector3::new(0.0, 1.0, 0.0)
    );
    approx::assert_relative_eq!(
        *group
            .elements()
            .get_index(3)
            .unwrap()
            .generating_element
            .raw_axis(),
        Vector3::new(1.0, 0.0, 0.0)
    );
    assert!(group.elements().get_index(4).unwrap().is_inversion());
    approx::assert_relative_eq!(
        *group
            .elements()
            .get_index(5)
            .unwrap()
            .generating_element
            .raw_axis(),
        Vector3::new(0.0, 0.0, 1.0)
    );
    approx::assert_relative_eq!(
        *group
            .elements()
            .get_index(6)
            .unwrap()
            .generating_element
            .raw_axis(),
        Vector3::new(0.0, 1.0, 0.0)
    );
    approx::assert_relative_eq!(
        *group
            .elements()
            .get_index(7)
            .unwrap()
            .generating_element
            .raw_axis(),
        Vector3::new(1.0, 0.0, 0.0)
    );
}

// ============================================
// Abstract group from molecular symmetry tests
// ============================================

fn verify_abstract_group(
    group: &impl ClassProperties<GroupElement = SymmetryOperation>,
    name: &str,
    order: usize,
    class_number: usize,
    abelian: bool,
) {
    assert_eq!(group.name(), name);
    assert_eq!(group.order(), order);
    assert_eq!(group.class_number(), class_number);
    assert_eq!(group.is_abelian(), abelian);

    for element_i in 0..group.order() {
        let class_i = group.get_cc_of_element_index(element_i).unwrap();
        assert!(group.get_cc_index(class_i).unwrap().contains(&element_i));
    }

    // Test inverse conjugacy classes
    let ctb = group.cayley_table().expect("Cayley table not found.");
    for class_i in 0..group.class_number() {
        let inv_class_i = group
            .get_inverse_cc(class_i)
            .expect("Inverse conjugacy class not found.");
        assert!(
            group
                .get_cc_index(class_i)
                .unwrap()
                .iter()
                .cartesian_product(group.get_cc_index(inv_class_i).unwrap().iter())
                .filter(|&(&g, &inv_g)| { ctb[[g, inv_g]] == 0 })
                .collect::<Vec<_>>()
                .len()
                == group.get_cc_index(class_i).unwrap().len()
        )
    }
}

fn test_ur_ordinary_group(
    mol: &Molecule,
    thresh: f64,
    name: &str,
    order: usize,
    class_number: usize,
    abelian: bool,
) {
    let presym = PreSymmetry::builder()
        .moi_threshold(thresh)
        .molecule(mol)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false).unwrap();
    let group = UnitaryRepresentedGroup::from_molecular_symmetry(&sym, None).unwrap();
    assert_eq!(group.group_type(), ORGRP);
    verify_abstract_group(&group, name, order, class_number, abelian);

    // IntoPermutation
    group.elements().into_iter().for_each(|op| {
        assert!(op.act_permute(mol).is_some());
    });
}

fn test_ur_ordinary_double_group(
    mol: &Molecule,
    thresh: f64,
    name: &str,
    order: usize,
    class_number: usize,
    abelian: bool,
) {
    let presym = PreSymmetry::builder()
        .moi_threshold(thresh)
        .molecule(mol)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false).unwrap();
    let group = UnitaryRepresentedGroup::from_molecular_symmetry(&sym, None)
        .unwrap()
        .to_double_group()
        .unwrap();
    assert_eq!(group.group_type(), ORGRP2);
    verify_abstract_group(&group, name, order, class_number, abelian);
}

fn test_ur_magnetic_group(
    mol: &Molecule,
    thresh: f64,
    name: &str,
    order: usize,
    class_number: usize,
    abelian: bool,
    mag_group_type: GroupType,
) {
    let presym = PreSymmetry::builder()
        .moi_threshold(thresh)
        .molecule(mol)
        .build()
        .unwrap();
    let mut magsym = Symmetry::new();
    magsym.analyse(&presym, true).unwrap();
    let group = UnitaryRepresentedGroup::from_molecular_symmetry(&magsym, None).unwrap();
    assert_eq!(group.group_type(), mag_group_type);
    verify_abstract_group(&group, name, order, class_number, abelian);

    // IntoPermutation
    group.elements().into_iter().for_each(|op| {
        assert!(op.act_permute(mol).is_some());
    });
}

fn test_ur_magnetic_double_group(
    mol: &Molecule,
    thresh: f64,
    name: &str,
    order: usize,
    class_number: usize,
    abelian: bool,
    mag_group_type: GroupType,
) {
    let presym = PreSymmetry::builder()
        .moi_threshold(thresh)
        .molecule(mol)
        .build()
        .unwrap();
    let mut magsym = Symmetry::new();
    magsym.analyse(&presym, true).unwrap();
    let group = UnitaryRepresentedGroup::from_molecular_symmetry(&magsym, None)
        .unwrap()
        .to_double_group()
        .unwrap();
    assert_eq!(group.group_type(), mag_group_type);
    verify_abstract_group(&group, name, order, class_number, abelian);

    // IntoPermutation
    group.elements().into_iter().for_each(|op| {
        assert!(op.act_permute(mol).is_some());
    });
}

// fn test_mr_magnetic_group(
//     mol: &Molecule,
//     thresh: f64,
//     name: &str,
//     order: usize,
//     class_number: usize,
//     abelian: bool,
//     mag_group_type: GroupType,
// ) {
//     let presym = PreSymmetry::builder()
//         .moi_threshold(thresh)
//         .molecule(mol)
//         .build()
//         .unwrap();
//     let mut magsym = Symmetry::new();
//     magsym.analyse(&presym, true).unwrap();
//     let group = MagneticRepresentedGroup::from_molecular_symmetry(&magsym, None).unwrap();
//     assert_eq!(group.group_type(), mag_group_type);
//     verify_abstract_group(&group, name, order, class_number, abelian);
//
//     // IntoPermutation
//     group.elements().into_iter().for_each(|op| {
//         assert!(op.act_permute(mol).is_some());
//     });
// }

fn test_ur_ordinary_group_from_infinite(
    mol: &Molecule,
    finite_order: u32,
    thresh: f64,
    name: &str,
    _finite_name: &str,
    order: usize,
    class_number: usize,
    abelian: bool,
) {
    let presym = PreSymmetry::builder()
        .moi_threshold(thresh)
        .molecule(mol)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false).unwrap();
    let group = UnitaryRepresentedGroup::from_molecular_symmetry(&sym, Some(finite_order)).unwrap();
    verify_abstract_group(&group, name, order, class_number, abelian);
}

fn test_ur_magnetic_group_from_infinite(
    mol: &Molecule,
    finite_order: u32,
    thresh: f64,
    name: &str,
    _finite_name: &str,
    order: usize,
    class_number: usize,
    abelian: bool,
    mag_group_type: GroupType,
) {
    let presym = PreSymmetry::builder()
        .moi_threshold(thresh)
        .molecule(mol)
        .build()
        .unwrap();
    let mut magsym = Symmetry::new();
    magsym.analyse(&presym, true).unwrap();
    let group =
        UnitaryRepresentedGroup::from_molecular_symmetry(&magsym, Some(finite_order)).unwrap();
    assert_eq!(
        group
            .elements()
            .iter()
            .filter(|op| op.contains_time_reversal())
            .count(),
        group
            .elements()
            .iter()
            .filter(|op| !op.contains_time_reversal())
            .count(),
    );
    assert_eq!(group.group_type(), mag_group_type);
    verify_abstract_group(&group, name, order, class_number, abelian);
}

// fn test_mr_magnetic_group_from_infinite(
//     mol: &Molecule,
//     finite_order: u32,
//     thresh: f64,
//     name: &str,
//     _finite_name: &str,
//     order: usize,
//     class_number: usize,
//     abelian: bool,
//     mag_group_type: GroupType,
// ) {
//     let presym = PreSymmetry::builder()
//         .moi_threshold(thresh)
//         .molecule(mol)
//         .build()
//         .unwrap();
//     let mut magsym = Symmetry::new();
//     magsym.analyse(&presym, true).unwrap();
//     let group =
//         MagneticRepresentedGroup::from_molecular_symmetry(&magsym, Some(finite_order)).unwrap();
//     assert_eq!(
//         group
//             .elements()
//             .iter()
//             .filter(|op| op.contains_time_reversal())
//             .count(),
//         group
//             .elements()
//             .iter()
//             .filter(|op| !op.contains_time_reversal())
//             .count(),
//     );
//     assert_eq!(group.group_type(), mag_group_type);
//     verify_abstract_group(&group, name, order, class_number, abelian);
// }

fn test_ur_ordinary_group_class_order(mol: &Molecule, thresh: f64, class_order_str: &[&str]) {
    let presym = PreSymmetry::builder()
        .moi_threshold(thresh)
        .molecule(mol)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false).unwrap();
    let group = UnitaryRepresentedGroup::from_molecular_symmetry(&sym, None).unwrap();
    let classes = (0..group.class_number())
        .map(|i| {
            group
                .get_cc_symbol_of_index(i)
                .expect("Unable to retrieve all class symbols.")
                .to_string()
        })
        .collect_vec();
    assert_eq!(&classes, class_order_str);
}

fn test_ur_ordinary_double_group_class_order(
    mol: &Molecule,
    thresh: f64,
    class_order_str: &[&str],
) {
    let presym = PreSymmetry::builder()
        .moi_threshold(thresh)
        .molecule(mol)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false).unwrap();
    let group = UnitaryRepresentedGroup::from_molecular_symmetry(&sym, None)
        .unwrap()
        .to_double_group()
        .unwrap();
    let classes = (0..group.class_number())
        .map(|i| {
            group
                .get_cc_symbol_of_index(i)
                .expect("Unable to retrieve all class symbols.")
                .to_string()
        })
        .collect_vec();
    assert_eq!(&classes, class_order_str);
}

fn test_ur_magnetic_group_class_order(mol: &Molecule, thresh: f64, class_order_str: &[&str]) {
    let presym = PreSymmetry::builder()
        .moi_threshold(thresh)
        .molecule(mol)
        .build()
        .unwrap();
    let mut magsym = Symmetry::new();
    magsym.analyse(&presym, true).unwrap();
    let group = UnitaryRepresentedGroup::from_molecular_symmetry(&magsym, None).unwrap();
    let classes = (0..group.class_number())
        .map(|i| {
            group
                .get_cc_symbol_of_index(i)
                .expect("Unable to retrieve all class symbols.")
                .to_string()
        })
        .collect_vec();
    assert_eq!(&classes, class_order_str);
}

fn test_ur_magnetic_double_group_class_order(
    mol: &Molecule,
    thresh: f64,
    class_order_str: &[&str],
) {
    let presym = PreSymmetry::builder()
        .moi_threshold(thresh)
        .molecule(mol)
        .build()
        .unwrap();
    let mut magsym = Symmetry::new();
    magsym.analyse(&presym, true).unwrap();
    let group = UnitaryRepresentedGroup::from_molecular_symmetry(&magsym, None)
        .unwrap()
        .to_double_group()
        .unwrap();
    let classes = (0..group.class_number())
        .map(|i| {
            group
                .get_cc_symbol_of_index(i)
                .expect("Unable to retrieve all class symbols.")
                .to_string()
        })
        .collect_vec();
    assert_eq!(&classes, class_order_str);
}

// fn test_mr_magnetic_group_class_order(mol: &Molecule, thresh: f64, class_order_str: &[&str]) {
//     let presym = PreSymmetry::builder()
//         .moi_threshold(thresh)
//         .molecule(mol)
//         .build()
//         .unwrap();
//     let mut magsym = Symmetry::new();
//     magsym.analyse(&presym, true).unwrap();
//     let group = MagneticRepresentedGroup::from_molecular_symmetry(&magsym, None).unwrap();
//     let classes = (0..group.class_number())
//         .map(|i| {
//             group
//                 .get_cc_symbol_of_index(i)
//                 .expect("Unable to retrieve all class symbols.")
//                 .to_string()
//         })
//         .collect_vec();
//     assert_eq!(&classes, class_order_str);
// }

/********
Spherical
********/

#[test]
fn test_ur_group_spherical_atom_o3() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/th.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);

    test_ur_ordinary_group_from_infinite(&mol, 2, thresh, "O(3)", "D2h", 8, 8, true);

    test_ur_ordinary_group_from_infinite(&mol, 4, thresh, "O(3)", "Oh", 48, 10, false);

    let result = panic::catch_unwind(|| {
        test_ur_ordinary_group_from_infinite(&mol, 5, thresh, "?", "?", 48, 10, false);
    });
    assert!(result.is_err());

    let result = panic::catch_unwind(|| {
        test_ur_ordinary_group_from_infinite(&mol, 3, thresh, "?", "?", 48, 10, false);
    });
    assert!(result.is_err());
}

#[test]
fn test_ur_group_spherical_atom_grey_o3() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/th.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);

    test_ur_magnetic_group_from_infinite(
        &mol,
        2,
        thresh,
        "O(3) + θ·O(3)",
        "D2h + θ·D2h",
        16,
        16,
        true,
        GRGRP,
    );

    test_ur_magnetic_group_from_infinite(
        &mol,
        4,
        thresh,
        "O(3) + θ·O(3)",
        "Oh + θ·Oh",
        96,
        20,
        false,
        GRGRP,
    );

    let result = panic::catch_unwind(|| {
        test_ur_magnetic_group_from_infinite(&mol, 5, thresh, "?", "?", 48, 10, false, GRGRP);
    });
    assert!(result.is_err());

    let result = panic::catch_unwind(|| {
        test_ur_magnetic_group_from_infinite(&mol, 3, thresh, "?", "?", 48, 10, false, GRGRP);
    });
    assert!(result.is_err());
}

#[test]
fn test_ur_group_spherical_c60_ih() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c60.xyz");
    let thresh = 1e-5;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_group(&mol, thresh, "Ih", 120, 10, false);
}

#[test]
fn test_ur_group_spherical_c60_ih_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c60.xyz");
    let thresh = 1e-5;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_group_class_order(
        &mol,
        thresh,
        &[
            "|E|",
            "12|C5|",
            "12|C5^2|",
            "20|C3|",
            "15|C2|",
            "|i|",
            "12|S10|",
            "12|σC10^3|",
            "20|S6|",
            "15|σ|",
        ],
    );
}

#[test]
fn test_ur_group_spherical_c60_grey_ih() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c60.xyz");
    let thresh = 1e-5;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_magnetic_group(&mol, thresh, "Ih + θ·Ih", 240, 20, false, GRGRP);
}

#[test]
fn test_ur_group_spherical_c60_grey_ih_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c60.xyz");
    let thresh = 1e-5;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_magnetic_group_class_order(
        &mol,
        thresh,
        &[
            "|E|",
            "12|C5|",
            "12|C5^2|",
            "20|C3|",
            "15|C2|",
            "|i|",
            "12|S10|",
            "12|σC10^3|",
            "20|S6|",
            "15|σ|",
            "|θ|",
            "12|θ·C5|",
            "12|θ·C5^2|",
            "20|θ·C3|",
            "15|θ·C2|",
            "|θ·i|",
            "12|θ·S10|",
            "12|θ·σC10^3|",
            "20|θ·S6|",
            "15|θ·σ|",
        ],
    );
}

#[test]
fn test_ur_group_spherical_ch4_td() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/ch4.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_group(&mol, thresh, "Td", 24, 5, false);
}

#[test]
fn test_ur_group_spherical_ch4_td_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/ch4.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_group_class_order(&mol, thresh, &["|E|", "8|C3|", "3|C2|", "6|S4|", "6|σd|"]);
}

#[test]
fn test_ur_group_spherical_ch4_grey_td() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/ch4.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_magnetic_group(&mol, thresh, "Td + θ·Td", 48, 10, false, GRGRP);
}

#[test]
fn test_ur_group_spherical_ch4_grey_td_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/ch4.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_magnetic_group_class_order(
        &mol,
        thresh,
        &[
            "|E|",
            "8|C3|",
            "3|C2|",
            "6|S4|",
            "6|σd|",
            "|θ|",
            "8|θ·C3|",
            "3|θ·C2|",
            "6|θ·S4|",
            "6|θ·σd|",
        ],
    );
}

#[test]
fn test_ur_group_spherical_adamantane_td() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/adamantane.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_group(&mol, thresh, "Td", 24, 5, false);
}

#[test]
fn test_ur_group_spherical_adamantane_td_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/adamantane.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_magnetic_group_class_order(
        &mol,
        thresh,
        &[
            "|E|",
            "8|C3|",
            "3|C2|",
            "6|S4|",
            "6|σd|",
            "|θ|",
            "8|θ·C3|",
            "3|θ·C2|",
            "6|θ·S4|",
            "6|θ·σd|",
        ],
    );
}

#[test]
fn test_ur_group_spherical_adamantane_grey_td() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/adamantane.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_magnetic_group(&mol, thresh, "Td + θ·Td", 48, 10, false, GRGRP);
}

#[test]
fn test_ur_group_spherical_adamantane_grey_td_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/adamantane.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_magnetic_group_class_order(
        &mol,
        thresh,
        &[
            "|E|",
            "8|C3|",
            "3|C2|",
            "6|S4|",
            "6|σd|",
            "|θ|",
            "8|θ·C3|",
            "3|θ·C2|",
            "6|θ·S4|",
            "6|θ·σd|",
        ],
    );
}

#[test]
fn test_ur_group_spherical_c165_diamond_nanoparticle_td() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c165.xyz");
    let thresh = 1e-5;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_group(&mol, thresh, "Td", 24, 5, false);
}

#[test]
fn test_ur_group_spherical_c165_diamond_nanoparticle_td_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c165.xyz");
    let thresh = 1e-5;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_group_class_order(&mol, thresh, &["|E|", "8|C3|", "3|C2|", "6|S4|", "6|σd|"]);
}

#[test]
fn test_ur_group_spherical_c165_diamond_nanoparticle_grey_td() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c165.xyz");
    let thresh = 1e-5;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_magnetic_group(&mol, thresh, "Td + θ·Td", 48, 10, false, GRGRP);
}

#[test]
fn test_ur_group_spherical_c165_diamond_nanoparticle_grey_td_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c165.xyz");
    let thresh = 1e-5;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_magnetic_group_class_order(
        &mol,
        thresh,
        &[
            "|E|",
            "8|C3|",
            "3|C2|",
            "6|S4|",
            "6|σd|",
            "|θ|",
            "8|θ·C3|",
            "3|θ·C2|",
            "6|θ·S4|",
            "6|θ·σd|",
        ],
    );
}

#[test]
fn test_ur_group_spherical_vh2o6_th() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/vh2o6.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_group(&mol, thresh, "Th", 24, 8, false);
}

#[test]
fn test_ur_group_spherical_vh2o6_th_class_order() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/vh2o6.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_group_class_order(
        &mol,
        thresh,
        &[
            "|E|",
            "4|C3|",
            "4|C3|^(')",
            "3|C2|",
            "|i|",
            "4|S6|",
            "4|S6|^(')",
            "3|σh|",
        ],
    );
}

#[test]
fn test_ur_group_spherical_vh2o6_grey_th() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/vh2o6.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_magnetic_group(&mol, thresh, "Th + θ·Th", 48, 16, false, GRGRP);
}

#[test]
fn test_ur_group_spherical_vh2o6_grey_th_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/vh2o6.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_magnetic_group_class_order(
        &mol,
        thresh,
        &[
            "|E|",
            "4|C3|",
            "4|C3|^(')",
            "3|C2|",
            "|i|",
            "4|S6|",
            "4|S6|^(')",
            "3|σh|",
            "|θ|",
            "4|θ·C3|",
            "4|θ·C3|^(')",
            "3|θ·C2|",
            "|θ·i|",
            "4|θ·S6|",
            "4|θ·S6|^(')",
            "3|θ·σh|",
        ],
    );
}

#[test]
fn test_ur_group_spherical_vf6_oh() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/vf6.xyz");
    let thresh = 1e-12;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_group(&mol, thresh, "Oh", 48, 10, false);
}

#[test]
fn test_ur_group_spherical_vf6_oh_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/vf6.xyz");
    let thresh = 1e-12;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_group_class_order(
        &mol,
        thresh,
        &[
            "|E|",
            "6|C4|",
            "8|C3|",
            "3|C2|",
            "6|C2|^(')",
            "|i|",
            "8|S6|",
            "6|S4|",
            "3|σh|",
            "6|σd|",
        ],
    );
}

#[test]
fn test_ur_group_spherical_vf6_grey_oh() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/vf6.xyz");
    let thresh = 1e-12;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_magnetic_group(&mol, thresh, "Oh + θ·Oh", 96, 20, false, GRGRP);
}

#[test]
fn test_ur_group_spherical_vf6_grey_oh_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/vf6.xyz");
    let thresh = 1e-12;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_magnetic_group_class_order(
        &mol,
        thresh,
        &[
            "|E|",
            "6|C4|",
            "8|C3|",
            "3|C2|",
            "6|C2|^(')",
            "|i|",
            "8|S6|",
            "6|S4|",
            "3|σh|",
            "6|σd|",
            "|θ|",
            "6|θ·C4|",
            "8|θ·C3|",
            "3|θ·C2|",
            "6|θ·C2|^(')",
            "|θ·i|",
            "8|θ·S6|",
            "6|θ·S4|",
            "3|θ·σh|",
            "6|θ·σd|",
        ],
    );
}

/*********
Spherical*
*********/
#[test]
fn test_ur_group_spherical_c60_ih_double() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c60.xyz");
    let thresh = 1e-5;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_double_group(&mol, thresh, "Ih*", 240, 18, false);
}

#[test]
fn test_ur_group_spherical_c60_ih_double_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c60.xyz");
    let thresh = 1e-5;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_double_group_class_order(
        &mol,
        thresh,
        &[
            "|E(Σ)|",
            "|E(QΣ)|",
            "12|C5(Σ)|",
            "12|C5(QΣ)|",
            "12|C5^2(Σ)|",
            "12|C5^2(QΣ)|",
            "20|C3(Σ)|",
            "20|C3(QΣ)|",
            "15|C2(Σ), C2(QΣ)|",
            "|i(Σ)|",
            "|i(QΣ)|",
            "12|S10(Σ)|",
            "12|S10(QΣ)|",
            "12|σC10^3(Σ)|",
            "12|σC10^3(QΣ)|",
            "20|S6(Σ)|",
            "20|S6(QΣ)|",
            "15|σ(Σ), σ(QΣ)|",
        ],
    );
}

#[test]
fn test_ur_group_spherical_c60_grey_ih_double() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c60.xyz");
    let thresh = 1e-5;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_magnetic_double_group(&mol, thresh, "(Ih + θ·Ih)*", 480, 36, false, GRGRP2);
}

#[test]
fn test_ur_group_spherical_c60_grey_ih_double_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c60.xyz");
    let thresh = 1e-5;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_magnetic_double_group_class_order(
        &mol,
        thresh,
        &[
            "|E(Σ)|",
            "|E(QΣ)|",
            "12|C5(Σ)|",
            "12|C5(QΣ)|",
            "12|C5^2(Σ)|",
            "12|C5^2(QΣ)|",
            "20|C3(Σ)|",
            "20|C3(QΣ)|",
            "15|C2(Σ), C2(QΣ)|",
            "|i(Σ)|",
            "|i(QΣ)|",
            "12|S10(Σ)|",
            "12|S10(QΣ)|",
            "12|σC10^3(Σ)|",
            "12|σC10^3(QΣ)|",
            "20|S6(Σ)|",
            "20|S6(QΣ)|",
            "15|σ(Σ), σ(QΣ)|",
            "|θ(Σ)|",
            "|θ(QΣ)|",
            "12|θ·C5(Σ)|",
            "12|θ·C5(QΣ)|",
            "12|θ·C5^2(Σ)|",
            "12|θ·C5^2(QΣ)|",
            "20|θ·C3(Σ)|",
            "20|θ·C3(QΣ)|",
            "15|θ·C2(Σ), θ·C2(QΣ)|",
            "|θ·i(Σ)|",
            "|θ·i(QΣ)|",
            "12|θ·S10(Σ)|",
            "12|θ·S10(QΣ)|",
            "12|θ·σC10^3(Σ)|",
            "12|θ·σC10^3(QΣ)|",
            "20|θ·S6(Σ)|",
            "20|θ·S6(QΣ)|",
            "15|θ·σ(Σ), θ·σ(QΣ)|",
        ],
    );
}

#[test]
fn test_ur_group_spherical_ch4_td_double() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/ch4.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_double_group(&mol, thresh, "Td*", 48, 8, false);
}

#[test]
fn test_ur_group_spherical_ch4_td_double_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/ch4.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_double_group_class_order(
        &mol,
        thresh,
        &[
            "|E(Σ)|",
            "|E(QΣ)|",
            "8|C3(Σ)|",
            "8|C3(QΣ)|",
            "3|C2(Σ), C2(QΣ)|",
            "6|S4(Σ)|",
            "6|S4(QΣ)|",
            "6|σd(Σ), σd(QΣ)|",
        ],
    );
}

#[test]
fn test_ur_group_spherical_ch4_grey_td_double() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/ch4.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_magnetic_double_group(&mol, thresh, "(Td + θ·Td)*", 96, 16, false, GRGRP2);
}

#[test]
fn test_ur_group_spherical_ch4_grey_td_double_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/ch4.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_magnetic_double_group_class_order(
        &mol,
        thresh,
        &[
            "|E(Σ)|",
            "|E(QΣ)|",
            "8|C3(Σ)|",
            "8|C3(QΣ)|",
            "3|C2(Σ), C2(QΣ)|",
            "6|S4(Σ)|",
            "6|S4(QΣ)|",
            "6|σd(Σ), σd(QΣ)|",
            "|θ(Σ)|",
            "|θ(QΣ)|",
            "8|θ·C3(Σ)|",
            "8|θ·C3(QΣ)|",
            "3|θ·C2(Σ), θ·C2(QΣ)|",
            "6|θ·S4(Σ)|",
            "6|θ·S4(QΣ)|",
            "6|θ·σd(Σ), θ·σd(QΣ)|",
        ],
    );
}

#[test]
fn test_ur_group_spherical_vh2o6_th_double() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/vh2o6.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_double_group(&mol, thresh, "Th*", 48, 14, false);
}

#[test]
fn test_ur_group_spherical_vh2o6_th_double_class_order() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/vh2o6.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_double_group_class_order(
        &mol,
        thresh,
        &[
            "|E(Σ)|",
            "|E(QΣ)|",
            "4|C3(Σ)|",
            "4|C3(Σ)|^(')",
            "4|C3(QΣ)|",
            "4|C3(QΣ)|^(')",
            "3|C2(Σ), C2(QΣ)|",
            "|i(Σ)|",
            "|i(QΣ)|",
            "4|S6(Σ)|",
            "4|S6(Σ)|^(')",
            "4|S6(QΣ)|",
            "4|S6(QΣ)|^(')",
            "3|σh(Σ), σh(QΣ)|",
        ],
    );
}

#[test]
fn test_ur_group_spherical_vh2o6_grey_th_double() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/vh2o6.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_magnetic_double_group(&mol, thresh, "(Th + θ·Th)*", 96, 28, false, GRGRP2);
}

#[test]
fn test_ur_group_spherical_vh2o6_grey_th_double_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/vh2o6.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_magnetic_double_group_class_order(
        &mol,
        thresh,
        &[
            "|E(Σ)|",
            "|E(QΣ)|",
            "4|C3(Σ)|",
            "4|C3(Σ)|^(')",
            "4|C3(QΣ)|",
            "4|C3(QΣ)|^(')",
            "3|C2(Σ), C2(QΣ)|",
            "|i(Σ)|",
            "|i(QΣ)|",
            "4|S6(Σ)|",
            "4|S6(Σ)|^(')",
            "4|S6(QΣ)|",
            "4|S6(QΣ)|^(')",
            "3|σh(Σ), σh(QΣ)|",
            "|θ(Σ)|",
            "|θ(QΣ)|",
            "4|θ·C3(Σ)|",
            "4|θ·C3(Σ)|^(')",
            "4|θ·C3(QΣ)|",
            "4|θ·C3(QΣ)|^(')",
            "3|θ·C2(Σ), θ·C2(QΣ)|",
            "|θ·i(Σ)|",
            "|θ·i(QΣ)|",
            "4|θ·S6(Σ)|",
            "4|θ·S6(Σ)|^(')",
            "4|θ·S6(QΣ)|",
            "4|θ·S6(QΣ)|^(')",
            "3|θ·σh(Σ), θ·σh(QΣ)|",
        ],
    );
}

#[test]
fn test_ur_group_spherical_vf6_oh_double() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/vf6.xyz");
    let thresh = 1e-12;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_double_group(&mol, thresh, "Oh*", 96, 16, false);
}

#[test]
fn test_ur_group_spherical_vf6_oh_double_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/vf6.xyz");
    let thresh = 1e-12;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_double_group_class_order(
        &mol,
        thresh,
        &[
            "|E(Σ)|",
            "|E(QΣ)|",
            "6|C4(Σ)|",
            "6|C4(QΣ)|",
            "8|C3(Σ)|",
            "8|C3(QΣ)|",
            "3|C2(Σ), C2(QΣ)|",
            "6|C2(Σ), C2(QΣ)|^(')",
            "|i(Σ)|",
            "|i(QΣ)|",
            "8|S6(Σ)|",
            "8|S6(QΣ)|",
            "6|S4(Σ)|",
            "6|S4(QΣ)|",
            "3|σh(Σ), σh(QΣ)|",
            "6|σd(Σ), σd(QΣ)|",
        ],
    );
}

#[test]
fn test_ur_group_spherical_vf6_grey_oh_double() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/vf6.xyz");
    let thresh = 1e-12;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_magnetic_double_group(&mol, thresh, "(Oh + θ·Oh)*", 192, 32, false, GRGRP2);
}

#[test]
fn test_ur_group_spherical_vf6_grey_oh_double_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/vf6.xyz");
    let thresh = 1e-12;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_magnetic_double_group_class_order(
        &mol,
        thresh,
        &[
            "|E(Σ)|",
            "|E(QΣ)|",
            "6|C4(Σ)|",
            "6|C4(QΣ)|",
            "8|C3(Σ)|",
            "8|C3(QΣ)|",
            "3|C2(Σ), C2(QΣ)|",
            "6|C2(Σ), C2(QΣ)|^(')",
            "|i(Σ)|",
            "|i(QΣ)|",
            "8|S6(Σ)|",
            "8|S6(QΣ)|",
            "6|S4(Σ)|",
            "6|S4(QΣ)|",
            "3|σh(Σ), σh(QΣ)|",
            "6|σd(Σ), σd(QΣ)|",
            "|θ(Σ)|",
            "|θ(QΣ)|",
            "6|θ·C4(Σ)|",
            "6|θ·C4(QΣ)|",
            "8|θ·C3(Σ)|",
            "8|θ·C3(QΣ)|",
            "3|θ·C2(Σ), θ·C2(QΣ)|",
            "6|θ·C2(Σ), θ·C2(QΣ)|^(')",
            "|θ·i(Σ)|",
            "|θ·i(QΣ)|",
            "8|θ·S6(Σ)|",
            "8|θ·S6(QΣ)|",
            "6|θ·S4(Σ)|",
            "6|θ·S4(QΣ)|",
            "3|θ·σh(Σ), θ·σh(QΣ)|",
            "6|θ·σd(Σ), θ·σd(QΣ)|",
        ],
    );
}

/*****
Linear
*****/

#[test]
fn test_ur_group_linear_atom_magnetic_field_cinfh_small() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/th.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(1.0, 2.0, -1.0)));
    for n in 2usize..=12usize {
        if n % 2 == 0 {
            test_ur_ordinary_group_from_infinite(
                &mol,
                n as u32,
                thresh,
                "C∞h",
                format!("C{n}h").as_str(),
                2 * n,
                2 * n,
                true,
            );
        } else {
            test_ur_ordinary_group_from_infinite(
                &mol,
                n as u32,
                thresh,
                "C∞h",
                format!("C{}h", 2 * n).as_str(),
                4 * n,
                4 * n,
                true,
            );
        }
    }
}

#[test]
#[ignore]
fn test_ur_group_linear_atom_magnetic_field_cinfh_large() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/th.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(1.0, 2.0, -1.0)));
    for n in 13usize..=20usize {
        if n % 2 == 0 {
            test_ur_ordinary_group_from_infinite(
                &mol,
                n as u32,
                thresh,
                "C∞h",
                format!("C{n}h").as_str(),
                2 * n,
                2 * n,
                true,
            );
        } else {
            test_ur_ordinary_group_from_infinite(
                &mol,
                n as u32,
                thresh,
                "C∞h",
                format!("C{}h", 2 * n).as_str(),
                4 * n,
                4 * n,
                true,
            );
        }
    }
}

#[test]
fn test_ur_group_linear_atom_magnetic_field_bw_dinfh_cinfh_small() {
    /* The expected number of classes is deduced from the irrep structures of
     * the Dnh groups.
     * When n is even, the irreps are A1(g/u), A2(g/u), B1(g/u), B2(g/u), Ek(g/u)
     * where k = 1, ..., n/2 - 1.
     * When n is odd, the irreps are A1('/''), A2('/''), Ek('/'')
     * where k = 1, ..., n//2.
     */
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/th.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(1.0, 2.0, -1.0)));
    for n in 2usize..=12usize {
        if n % 2 == 0 {
            test_ur_magnetic_group_from_infinite(
                &mol,
                n as u32,
                thresh,
                "D∞h",
                format!("D{n}h").as_str(),
                4 * n,
                2 * (n / 2 - 1 + 4),
                n == 2,
                BWGRP,
            );
        } else {
            test_ur_magnetic_group_from_infinite(
                &mol,
                n as u32,
                thresh,
                "D∞h",
                format!("D{}h", 2 * n).as_str(),
                8 * n,
                2 * (n - 1 + 4),
                false,
                BWGRP,
            );
        }
    }
}

#[test]
#[ignore]
fn test_ur_group_linear_atom_magnetic_field_bw_dinfh_cinfh_large() {
    /* The expected number of classes is deduced from the irrep structures of
     * the Dnh groups.
     * When n is even, the irreps are A1(g/u), A2(g/u), B1(g/u), B2(g/u), Ek(g/u)
     * where k = 1, ..., n/2 - 1.
     * When n is odd, the irreps are A1('/''), A2('/''), Ek('/'')
     * where k = 1, ..., n//2.
     */
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/th.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(1.0, 2.0, -1.0)));
    for n in 12usize..=20usize {
        if n % 2 == 0 {
            test_ur_magnetic_group_from_infinite(
                &mol,
                n as u32,
                thresh,
                "D∞h",
                format!("D{n}h").as_str(),
                4 * n,
                2 * (n / 2 - 1 + 4),
                n == 2,
                BWGRP,
            );
        } else {
            test_ur_magnetic_group_from_infinite(
                &mol,
                n as u32,
                thresh,
                "D∞h",
                format!("D{}h", 2 * n).as_str(),
                8 * n,
                2 * (n - 1 + 4),
                false,
                BWGRP,
            );
        }
    }
}

#[test]
fn test_ur_group_linear_atom_electric_field_cinfv_small() {
    /* The expected number of classes is deduced from the irrep structures of
     * the Cnv groups.
     * When n is even, the irreps are A1, A2, B1, B2, Ek where k = 1, ..., n/2 - 1.
     * When n is odd, the irreps are A1, A2, Ek where k = 1, ..., n//2.
     */
    let path: String = format!("{}{}", ROOT, "/tests/xyz/th.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(Vector3::new(-1.0, 3.0, -2.0)));
    for n in 3usize..=12usize {
        test_ur_ordinary_group_from_infinite(
            &mol,
            n as u32,
            thresh,
            "C∞v",
            format!("C{n}v").as_str(),
            2 * n,
            ({
                if n % 2 == 0 {
                    n / 2 - 1
                } else {
                    n / 2
                }
            } + {
                if n % 2 == 0 {
                    4
                } else {
                    2
                }
            }),
            false,
        );
    }
}

#[test]
#[ignore]
fn test_ur_group_linear_atom_electric_field_cinfv_large() {
    /* The expected number of classes is deduced from the irrep structures of
     * the Cnv groups.
     * When n is even, the irreps are A1, A2, B1, B2, Ek where k = 1, ..., n/2 - 1.
     * When n is odd, the irreps are A1, A2, Ek where k = 1, ..., n//2.
     */
    let path: String = format!("{}{}", ROOT, "/tests/xyz/th.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(Vector3::new(-1.0, 3.0, -2.0)));
    for n in 13usize..=20usize {
        test_ur_ordinary_group_from_infinite(
            &mol,
            n as u32,
            thresh,
            "C∞v",
            format!("C{n}v").as_str(),
            2 * n,
            ({
                if n % 2 == 0 {
                    n / 2 - 1
                } else {
                    n / 2
                }
            } + {
                if n % 2 == 0 {
                    4
                } else {
                    2
                }
            }),
            false,
        );
    }
}

#[test]
fn test_ur_group_linear_c2h2_dinfh_small() {
    /* The expected number of classes is deduced from the irrep structures of
     * the Dnh groups.
     * When n is even, the irreps are A1(g/u), A2(g/u), B1(g/u), B2(g/u), Ek(g/u)
     * where k = 1, ..., n/2 - 1.
     * When n is odd, the irreps are A1('/''), A2('/''), Ek('/'')
     * where k = 1, ..., n//2.
     */
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c2h2.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    for n in 3usize..=12usize {
        if n % 2 == 0 {
            test_ur_ordinary_group_from_infinite(
                &mol,
                n as u32,
                thresh,
                "D∞h",
                format!("D{n}h").as_str(),
                4 * n,
                2 * (n / 2 - 1 + 4),
                false,
            );
        } else {
            test_ur_ordinary_group_from_infinite(
                &mol,
                n as u32,
                thresh,
                "D∞h",
                format!("D{}h", 2 * n).as_str(),
                8 * n,
                2 * (n - 1 + 4),
                false,
            );
        }
    }
}

#[test]
#[ignore]
fn test_ur_group_linear_c2h2_dinfh_large() {
    /* The expected number of classes is deduced from the irrep structures of
     * the Dnh groups.
     * When n is even, the irreps are A1(g/u), A2(g/u), B1(g/u), B2(g/u), Ek(g/u)
     * where k = 1, ..., n/2 - 1.
     * When n is odd, the irreps are A1('/''), A2('/''), Ek('/'')
     * where k = 1, ..., n//2.
     */
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c2h2.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    for n in 13usize..=20usize {
        if n % 2 == 0 {
            test_ur_ordinary_group_from_infinite(
                &mol,
                n as u32,
                thresh,
                "D∞h",
                format!("D{n}h").as_str(),
                4 * n,
                2 * (n / 2 - 1 + 4),
                false,
            );
        } else {
            test_ur_ordinary_group_from_infinite(
                &mol,
                n as u32,
                thresh,
                "D∞h",
                format!("D{}h", 2 * n).as_str(),
                8 * n,
                2 * (n - 1 + 4),
                false,
            );
        }
    }
}

#[test]
fn test_ur_group_linear_c2h2_grey_dinfh_small() {
    /* The expected number of classes is deduced from the irrep structures of
     * the Dnh groups.
     * When n is even, the irreps are A1(g/u), A2(g/u), B1(g/u), B2(g/u), Ek(g/u)
     * where k = 1, ..., n/2 - 1.
     * When n is odd, the irreps are A1('/''), A2('/''), Ek('/'')
     * where k = 1, ..., n//2.
     */
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c2h2.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    for n in 3usize..=12usize {
        if n % 2 == 0 {
            test_ur_magnetic_group_from_infinite(
                &mol,
                n as u32,
                thresh,
                "D∞h + θ·D∞h",
                format!("D{n}h + θ·D{n}h").as_str(),
                8 * n,
                4 * (n / 2 - 1 + 4),
                false,
                GRGRP,
            );
        } else {
            test_ur_magnetic_group_from_infinite(
                &mol,
                n as u32,
                thresh,
                "D∞h + θ·D∞h",
                format!("D{}h + θ·D{}h", 2 * n, 2 * n).as_str(),
                16 * n,
                4 * (n - 1 + 4),
                false,
                GRGRP,
            );
        }
    }
}

#[test]
#[ignore]
fn test_ur_group_linear_c2h2_grey_dinfh_large() {
    /* The expected number of classes is deduced from the irrep structures of
     * the Dnh groups.
     * When n is even, the irreps are A1(g/u), A2(g/u), B1(g/u), B2(g/u), Ek(g/u)
     * where k = 1, ..., n/2 - 1.
     * When n is odd, the irreps are A1('/''), A2('/''), Ek('/'')
     * where k = 1, ..., n//2.
     */
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c2h2.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    for n in 13usize..=20usize {
        if n % 2 == 0 {
            test_ur_magnetic_group_from_infinite(
                &mol,
                n as u32,
                thresh,
                "D∞h + θ·D∞h",
                format!("D{n}h + θ·D{n}h").as_str(),
                8 * n,
                4 * (n / 2 - 1 + 4),
                false,
                GRGRP,
            );
        } else {
            test_ur_magnetic_group_from_infinite(
                &mol,
                n as u32,
                thresh,
                "D∞h + θ·D∞h",
                format!("D{}h + θ·D{}h", 2 * n, 2 * n).as_str(),
                16 * n,
                4 * (n - 1 + 4),
                false,
                GRGRP,
            );
        }
    }
}

#[test]
fn test_ur_group_linear_c2h2_magnetic_field_cinfh_small() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c2h2.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);
    // Parallel field
    mol.set_magnetic_field(Some(Vector3::new(1.0, 1.0, 1.0)));
    for n in 2usize..=12usize {
        if n % 2 == 0 {
            test_ur_ordinary_group_from_infinite(
                &mol,
                n as u32,
                thresh,
                "C∞h",
                format!("C{n}h").as_str(),
                2 * n,
                2 * n,
                true,
            );
        } else {
            test_ur_ordinary_group_from_infinite(
                &mol,
                n as u32,
                thresh,
                "C∞h",
                format!("C{}h", 2 * n).as_str(),
                4 * n,
                4 * n,
                true,
            );
        }
    }
}

#[test]
#[ignore]
fn test_ur_group_linear_c2h2_magnetic_field_cinfh_large() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c2h2.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);
    // Parallel field
    mol.set_magnetic_field(Some(Vector3::new(1.0, 1.0, 1.0)));
    for n in 13usize..=20usize {
        if n % 2 == 0 {
            test_ur_ordinary_group_from_infinite(
                &mol,
                n as u32,
                thresh,
                "C∞h",
                format!("C{n}h").as_str(),
                2 * n,
                2 * n,
                true,
            );
        } else {
            test_ur_ordinary_group_from_infinite(
                &mol,
                n as u32,
                thresh,
                "C∞h",
                format!("C{}h", 2 * n).as_str(),
                4 * n,
                4 * n,
                true,
            );
        }
    }
}

#[test]
fn test_ur_group_linear_c2h2_magnetic_field_bw_dinfh_cinfh_small() {
    /* The expected number of classes is deduced from the irrep structures of
     * the Dnh groups.
     * When n is even, the irreps are A1(g/u), A2(g/u), B1(g/u), B2(g/u), Ek(g/u)
     * where k = 1, ..., n/2 - 1.
     * When n is odd, the irreps are A1('/''), A2('/''), Ek('/'')
     * where k = 1, ..., n//2.
     */
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c2h2.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(1.0, 1.0, 1.0)));
    for n in 2usize..=12usize {
        if n % 2 == 0 {
            test_ur_magnetic_group_from_infinite(
                &mol,
                n as u32,
                thresh,
                "D∞h",
                format!("D{n}h").as_str(),
                4 * n,
                2 * (n / 2 - 1 + 4),
                n == 2,
                BWGRP,
            );
        } else {
            test_ur_magnetic_group_from_infinite(
                &mol,
                n as u32,
                thresh,
                "D∞h",
                format!("D{}h", 2 * n).as_str(),
                8 * n,
                2 * (n - 1 + 4),
                false,
                BWGRP,
            );
        }
    }
}

#[test]
#[ignore]
fn test_ur_group_linear_c2h2_magnetic_field_bw_dinfh_cinfh_large() {
    /* The expected number of classes is deduced from the irrep structures of
     * the Dnh groups.
     * When n is even, the irreps are A1(g/u), A2(g/u), B1(g/u), B2(g/u), Ek(g/u)
     * where k = 1, ..., n/2 - 1.
     * When n is odd, the irreps are A1('/''), A2('/''), Ek('/'')
     * where k = 1, ..., n//2.
     */
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c2h2.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(1.0, 1.0, 1.0)));
    for n in 13usize..=20usize {
        if n % 2 == 0 {
            test_ur_magnetic_group_from_infinite(
                &mol,
                n as u32,
                thresh,
                "D∞h",
                format!("D{n}h").as_str(),
                4 * n,
                2 * (n / 2 - 1 + 4),
                n == 2,
                BWGRP,
            );
        } else {
            test_ur_magnetic_group_from_infinite(
                &mol,
                n as u32,
                thresh,
                "D∞h",
                format!("D{}h", 2 * n).as_str(),
                8 * n,
                2 * (n - 1 + 4),
                false,
                BWGRP,
            );
        }
    }
}

#[test]
fn test_ur_group_linear_c2h2_electric_field_cinfv_small() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c2h2.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);

    // Parallel field
    mol.set_electric_field(Some(Vector3::new(1.0, 1.0, 1.0)));
    for n in 3usize..=12usize {
        test_ur_ordinary_group_from_infinite(
            &mol,
            n as u32,
            thresh,
            "C∞v",
            format!("C{n}v").as_str(),
            2 * n,
            ({
                if n % 2 == 0 {
                    n / 2 - 1
                } else {
                    n / 2
                }
            } + {
                if n % 2 == 0 {
                    4
                } else {
                    2
                }
            }),
            false,
        );
    }
}

#[test]
#[ignore]
fn test_ur_group_linear_c2h2_electric_field_cinfv_large() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c2h2.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);

    // Parallel field
    mol.set_electric_field(Some(Vector3::new(1.0, 1.0, 1.0)));
    for n in 13usize..=20usize {
        test_ur_ordinary_group_from_infinite(
            &mol,
            n as u32,
            thresh,
            "C∞v",
            format!("C{n}v").as_str(),
            2 * n,
            ({
                if n % 2 == 0 {
                    n / 2 - 1
                } else {
                    n / 2
                }
            } + {
                if n % 2 == 0 {
                    4
                } else {
                    2
                }
            }),
            false,
        );
    }
}

#[test]
fn test_ur_group_linear_c2h2_electric_field_grey_cinfv_small() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c2h2.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);

    // Parallel field
    mol.set_electric_field(Some(Vector3::new(1.0, 1.0, 1.0)));
    for n in 3usize..=12usize {
        test_ur_magnetic_group_from_infinite(
            &mol,
            n as u32,
            thresh,
            "C∞v + θ·C∞v",
            format!("C{n}v + θ·C{n}v").as_str(),
            4 * n,
            2 * ({
                if n % 2 == 0 {
                    n / 2 - 1
                } else {
                    n / 2
                }
            } + {
                if n % 2 == 0 {
                    4
                } else {
                    2
                }
            }),
            false,
            GRGRP,
        );
    }
}

#[test]
#[ignore]
fn test_ur_group_linear_c2h2_electric_field_grey_cinfv_large() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c2h2.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);

    // Parallel field
    mol.set_electric_field(Some(Vector3::new(1.0, 1.0, 1.0)));
    for n in 13usize..=20usize {
        test_ur_magnetic_group_from_infinite(
            &mol,
            n as u32,
            thresh,
            "C∞v + θ·C∞v",
            format!("C{n}v + θ·C{n}v").as_str(),
            4 * n,
            2 * ({
                if n % 2 == 0 {
                    n / 2 - 1
                } else {
                    n / 2
                }
            } + {
                if n % 2 == 0 {
                    4
                } else {
                    2
                }
            }),
            false,
            GRGRP,
        );
    }
}

#[test]
fn test_ur_group_linear_n3_cinfv_small() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/n3.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    for n in 3usize..=12usize {
        test_ur_ordinary_group_from_infinite(
            &mol,
            n as u32,
            thresh,
            "C∞v",
            format!("C{n}v").as_str(),
            2 * n,
            ({
                if n % 2 == 0 {
                    n / 2 - 1
                } else {
                    n / 2
                }
            } + {
                if n % 2 == 0 {
                    4
                } else {
                    2
                }
            }),
            false,
        );
    }
}

#[test]
#[ignore]
fn test_ur_group_linear_n3_cinfv_large() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/n3.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    for n in 13usize..=20usize {
        test_ur_ordinary_group_from_infinite(
            &mol,
            n as u32,
            thresh,
            "C∞v",
            format!("C{n}v").as_str(),
            2 * n,
            ({
                if n % 2 == 0 {
                    n / 2 - 1
                } else {
                    n / 2
                }
            } + {
                if n % 2 == 0 {
                    4
                } else {
                    2
                }
            }),
            false,
        );
    }
}

#[test]
fn test_ur_group_linear_n3_grey_cinfv_small() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/n3.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    for n in 3usize..=12usize {
        test_ur_magnetic_group_from_infinite(
            &mol,
            n as u32,
            thresh,
            "C∞v + θ·C∞v",
            format!("C{n}v + θ·C{n}v").as_str(),
            4 * n,
            2 * ({
                if n % 2 == 0 {
                    n / 2 - 1
                } else {
                    n / 2
                }
            } + {
                if n % 2 == 0 {
                    4
                } else {
                    2
                }
            }),
            false,
            GRGRP,
        );
    }
}

#[test]
#[ignore]
fn test_ur_group_linear_n3_grey_cinfv_large() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/n3.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    for n in 13usize..=20usize {
        test_ur_magnetic_group_from_infinite(
            &mol,
            n as u32,
            thresh,
            "C∞v + θ·C∞v",
            format!("C{n}v + θ·C{n}v").as_str(),
            4 * n,
            2 * ({
                if n % 2 == 0 {
                    n / 2 - 1
                } else {
                    n / 2
                }
            } + {
                if n % 2 == 0 {
                    4
                } else {
                    2
                }
            }),
            false,
            GRGRP,
        );
    }
}

#[test]
fn test_ur_group_linear_n3_magnetic_field_cinf_small() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/n3.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);

    // Parallel field
    mol.set_magnetic_field(Some(Vector3::new(1.0, 1.0, 1.0)));
    for n in 2usize..=12usize {
        test_ur_ordinary_group_from_infinite(
            &mol,
            n as u32,
            thresh,
            "C∞",
            format!("C{n}").as_str(),
            n,
            n,
            true,
        );
    }
}

#[test]
#[ignore]
fn test_ur_group_linear_n3_magnetic_field_cinf_large() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/n3.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);

    // Parallel field
    mol.set_magnetic_field(Some(Vector3::new(1.0, 1.0, 1.0)));
    for n in 13usize..=20usize {
        test_ur_ordinary_group_from_infinite(
            &mol,
            n as u32,
            thresh,
            "C∞",
            format!("C{n}").as_str(),
            n,
            n,
            true,
        );
    }
}

#[test]
fn test_ur_group_linear_n3_magnetic_field_bw_cinfv_cinf_small() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/n3.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);

    // Parallel field
    mol.set_magnetic_field(Some(Vector3::new(1.0, 1.0, 1.0)));
    for n in 2usize..=12usize {
        test_ur_magnetic_group_from_infinite(
            &mol,
            n as u32,
            thresh,
            "C∞v",
            format!("C{n}v").as_str(),
            2 * n,
            ({
                if n % 2 == 0 {
                    n / 2 - 1
                } else {
                    n / 2
                }
            } + {
                if n % 2 == 0 {
                    4
                } else {
                    2
                }
            }),
            n == 2,
            BWGRP,
        );
    }
}

#[test]
#[ignore]
fn test_ur_group_linear_n3_magnetic_field_bw_cinfv_cinf_large() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/n3.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);

    // Parallel field
    mol.set_magnetic_field(Some(Vector3::new(1.0, 1.0, 1.0)));
    for n in 13usize..=20usize {
        test_ur_magnetic_group_from_infinite(
            &mol,
            n as u32,
            thresh,
            "C∞v",
            format!("C{n}v").as_str(),
            2 * n,
            ({
                if n % 2 == 0 {
                    n / 2 - 1
                } else {
                    n / 2
                }
            } + {
                if n % 2 == 0 {
                    4
                } else {
                    2
                }
            }),
            n == 2,
            BWGRP,
        );
    }
}

#[test]
fn test_ur_group_linear_n3_electric_field_cinfv_small() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/n3.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);

    // Parallel field
    mol.set_electric_field(Some(Vector3::new(1.0, 1.0, 1.0)));
    for n in 3usize..=12usize {
        test_ur_ordinary_group_from_infinite(
            &mol,
            n as u32,
            thresh,
            "C∞v",
            format!("C{n}v").as_str(),
            2 * n,
            ({
                if n % 2 == 0 {
                    n / 2 - 1
                } else {
                    n / 2
                }
            } + {
                if n % 2 == 0 {
                    4
                } else {
                    2
                }
            }),
            false,
        );
    }
}

#[test]
#[ignore]
fn test_ur_group_linear_n3_electric_field_cinfv_large() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/n3.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);

    // Parallel field
    mol.set_electric_field(Some(Vector3::new(1.0, 1.0, 1.0)));
    for n in 13usize..=20usize {
        test_ur_ordinary_group_from_infinite(
            &mol,
            n as u32,
            thresh,
            "C∞v",
            format!("C{n}v").as_str(),
            2 * n,
            ({
                if n % 2 == 0 {
                    n / 2 - 1
                } else {
                    n / 2
                }
            } + {
                if n % 2 == 0 {
                    4
                } else {
                    2
                }
            }),
            false,
        );
    }
}

#[test]
fn test_ur_group_linear_n3_electric_field_grey_cinfv_small() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/n3.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);

    // Parallel field
    mol.set_electric_field(Some(Vector3::new(1.0, 1.0, 1.0)));
    for n in 3usize..=12usize {
        test_ur_magnetic_group_from_infinite(
            &mol,
            n as u32,
            thresh,
            "C∞v + θ·C∞v",
            format!("C{n}v + θ·C{n}v").as_str(),
            4 * n,
            2 * ({
                if n % 2 == 0 {
                    n / 2 - 1
                } else {
                    n / 2
                }
            } + {
                if n % 2 == 0 {
                    4
                } else {
                    2
                }
            }),
            false,
            GRGRP,
        );
    }
}

#[test]
#[ignore]
fn test_ur_group_linear_n3_electric_field_grey_cinfv_large() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/n3.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);

    // Parallel field
    mol.set_electric_field(Some(Vector3::new(1.0, 1.0, 1.0)));
    for n in 13usize..=20usize {
        test_ur_magnetic_group_from_infinite(
            &mol,
            n as u32,
            thresh,
            "C∞v + θ·C∞v",
            format!("C{n}v + θ·C{n}v").as_str(),
            4 * n,
            2 * ({
                if n % 2 == 0 {
                    n / 2 - 1
                } else {
                    n / 2
                }
            } + {
                if n % 2 == 0 {
                    4
                } else {
                    2
                }
            }),
            false,
            GRGRP,
        );
    }
}

/********
Symmetric
********/

/*
Cn
*/

#[test]
fn test_ur_group_symmetric_ch4_magnetic_field_c3() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/ch4.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(1.0, 1.0, 1.0)));
    test_ur_ordinary_group(&mol, thresh, "C3", 3, 3, true);
}

#[test]
fn test_ur_group_symmetric_ch4_magnetic_field_c3_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/ch4.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(1.0, 1.0, 1.0)));
    test_ur_ordinary_group_class_order(&mol, thresh, &["|E|", "|C3|", "|C3^(-1)|"]);
}

#[test]
fn test_ur_group_symmetric_ch4_magnetic_field_bw_c3v_c3() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/ch4.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(1.0, 1.0, 1.0)));
    test_ur_magnetic_group(&mol, thresh, "C3v", 6, 3, false, BWGRP);
}

#[test]
fn test_ur_group_symmetric_ch4_magnetic_field_bw_c3v_c3_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/ch4.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(1.0, 1.0, 1.0)));
    test_ur_magnetic_group_class_order(&mol, thresh, &["|E|", "2|C3|", "3|θ·σv|"]);
}

#[test]
fn test_ur_group_symmetric_adamantane_magnetic_field_c3() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/adamantane.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.1, 0.1, 0.1)));
    test_ur_ordinary_group(&mol, thresh, "C3", 3, 3, true);
}

#[test]
fn test_ur_group_symmetric_adamantane_magnetic_field_c3_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/adamantane.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.1, 0.1, 0.1)));
    test_ur_ordinary_group_class_order(&mol, thresh, &["|E|", "|C3|", "|C3^(-1)|"]);
}

#[test]
fn test_ur_group_symmetric_adamantane_magnetic_field_bw_c3v_c3() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/adamantane.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.1, 0.1, 0.1)));
    test_ur_magnetic_group(&mol, thresh, "C3v", 6, 3, false, BWGRP);
}

#[test]
fn test_ur_group_symmetric_adamantane_magnetic_field_bw_c3v_c3_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/adamantane.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.1, 0.1, 0.1)));
    test_ur_magnetic_group_class_order(&mol, thresh, &["|E|", "2|C3|", "3|θ·σv|"]);
}

#[test]
fn test_ur_group_symmetric_vh2o6_electric_field_c3() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/vh2o6.xyz");
    let thresh = 1e-12;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(Vector3::new(-0.2, -0.2, -0.2)));
    test_ur_ordinary_group(&mol, thresh, "C3", 3, 3, true);
}

#[test]
fn test_ur_group_symmetric_vh2o6_electric_field_c3_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/vh2o6.xyz");
    let thresh = 1e-12;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(Vector3::new(-0.2, -0.2, -0.2)));
    test_ur_ordinary_group_class_order(&mol, thresh, &["|E|", "|C3|", "|C3^(-1)|"]);
}

#[test]
fn test_ur_group_symmetric_vh2o6_electric_field_grey_c3() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/vh2o6.xyz");
    let thresh = 1e-12;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(Vector3::new(-0.2, -0.2, -0.2)));
    test_ur_magnetic_group(&mol, thresh, "C3 + θ·C3", 6, 6, true, GRGRP);
}

#[test]
fn test_ur_group_symmetric_vh2o6_electric_field_grey_c3_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/vh2o6.xyz");
    let thresh = 1e-12;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(Vector3::new(-0.2, -0.2, -0.2)));
    test_ur_magnetic_group_class_order(
        &mol,
        thresh,
        &["|E|", "|C3|", "|C3^(-1)|", "|θ|", "|θ·C3|", "|θ·C3^(-1)|"],
    );
}

#[test]
fn test_ur_group_symmetric_65coronane_electric_field_c3() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/coronane65.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(Vector3::new(0.0, 0.0, -1.0)));
    test_ur_ordinary_group(&mol, thresh, "C3", 3, 3, true);
}

#[test]
fn test_ur_group_symmetric_65coronane_electric_field_c3_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/coronane65.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(Vector3::new(0.0, 0.0, -1.0)));
    test_ur_ordinary_group_class_order(&mol, thresh, &["|E|", "|C3|", "|C3^(-1)|"]);
}

#[test]
fn test_ur_group_symmetric_65coronane_electric_field_grey_c3() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/coronane65.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(Vector3::new(0.0, 0.0, -1.0)));
    test_ur_magnetic_group(&mol, thresh, "C3 + θ·C3", 6, 6, true, GRGRP);
}

#[test]
fn test_ur_group_symmetric_65coronane_electric_field_grey_c3_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/coronane65.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(Vector3::new(0.0, 0.0, -1.0)));
    test_ur_magnetic_group_class_order(
        &mol,
        thresh,
        &["|E|", "|C3|", "|C3^(-1)|", "|θ|", "|θ·C3|", "|θ·C3^(-1)|"],
    );
}

#[test]
fn test_ur_group_symmetric_h8_twisted_magnetic_field_c4() {
    // env_logger::init();
    let thresh = 1e-7;
    let mut mol = template_molecules::gen_twisted_h8(0.1);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 0.1)));
    test_ur_ordinary_group(&mol, thresh, "C4", 4, 4, true);
}

#[test]
fn test_ur_group_symmetric_h8_twisted_magnetic_field_c4_class_order() {
    let thresh = 1e-7;
    let mut mol = template_molecules::gen_twisted_h8(0.1);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 0.1)));
    test_ur_ordinary_group_class_order(&mol, thresh, &["|E|", "|C4|", "|C4^(-1)|", "|C2|"]);
}

#[test]
fn test_ur_group_symmetric_h8_twisted_magnetic_field_bw_d4_c4() {
    // env_logger::init();
    let thresh = 1e-7;
    let mut mol = template_molecules::gen_twisted_h8(0.1);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 0.1)));
    test_ur_magnetic_group(&mol, thresh, "D4", 8, 5, false, BWGRP);
}

#[test]
fn test_ur_group_symmetric_h8_twisted_magnetic_field_bw_d4_c4_class_order() {
    let thresh = 1e-7;
    let mut mol = template_molecules::gen_twisted_h8(0.1);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 0.1)));
    test_ur_magnetic_group_class_order(
        &mol,
        thresh,
        &["|E|", "2|C4|", "|C2|", "2|θ·C2|", "2|θ·C2|^(')"],
    );
}

#[test]
fn test_ur_group_symmetric_h8_twisted_electric_field_c4() {
    let thresh = 1e-7;
    let mut mol = template_molecules::gen_twisted_h8(0.1);
    mol.set_electric_field(Some(Vector3::new(0.0, 0.0, -0.1)));
    test_ur_ordinary_group(&mol, thresh, "C4", 4, 4, true);
}

#[test]
fn test_ur_group_symmetric_h8_twisted_electric_field_c4_class_order() {
    let thresh = 1e-7;
    let mut mol = template_molecules::gen_twisted_h8(0.1);
    mol.set_electric_field(Some(Vector3::new(0.0, 0.0, -0.1)));
    test_ur_ordinary_group_class_order(&mol, thresh, &["|E|", "|C4|", "|C4^(-1)|", "|C2|"]);
}

#[test]
fn test_ur_group_symmetric_h8_twisted_electric_field_grey_c4() {
    let thresh = 1e-7;
    let mut mol = template_molecules::gen_twisted_h8(0.1);
    mol.set_electric_field(Some(Vector3::new(0.0, 0.0, -0.1)));
    test_ur_magnetic_group(&mol, thresh, "C4 + θ·C4", 8, 8, true, GRGRP);
}

#[test]
fn test_ur_group_symmetric_h8_twisted_electric_field_grey_c4_class_order() {
    let thresh = 1e-7;
    let mut mol = template_molecules::gen_twisted_h8(0.1);
    mol.set_electric_field(Some(Vector3::new(0.0, 0.0, -0.1)));
    test_ur_magnetic_group_class_order(
        &mol,
        thresh,
        &[
            "|E|",
            "|C4|",
            "|C4^(-1)|",
            "|C2|",
            "|θ|",
            "|θ·C4|",
            "|θ·C4^(-1)|",
            "|θ·C2|",
        ],
    );
}

#[test]
fn test_ur_group_symmetric_cpnico_magnetic_field_c5() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/cpnico.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, -0.2)));
    test_ur_ordinary_group(&mol, thresh, "C5", 5, 5, true);
}

#[test]
fn test_ur_group_symmetric_cpnico_magnetic_field_c5_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/cpnico.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, -0.2)));
    test_ur_ordinary_group_class_order(
        &mol,
        thresh,
        &["|E|", "|C5|", "|C5^2|", "|C5^(-2)|", "|C5^(-1)|"],
    );
}

#[test]
fn test_ur_group_symmetric_cpnico_magnetic_field_bw_c5v_c5() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/cpnico.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, -0.2)));
    test_ur_magnetic_group(&mol, thresh, "C5v", 10, 4, false, BWGRP);
}

#[test]
fn test_ur_group_symmetric_cpnico_magnetic_field_bw_c5v_c5_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/cpnico.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, -0.2)));
    test_ur_magnetic_group_class_order(&mol, thresh, &["|E|", "2|C5|", "2|C5^2|", "5|θ·σv|"]);
}

#[test]
fn test_ur_group_symmetric_b7_magnetic_field_c6() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/b7.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 0.1)));
    test_ur_ordinary_group(&mol, thresh, "C6", 6, 6, true);
}

#[test]
fn test_ur_group_symmetric_b7_magnetic_field_c6_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/b7.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 0.1)));
    test_ur_ordinary_group_class_order(
        &mol,
        thresh,
        &["|E|", "|C6|", "|C6^(-1)|", "|C3|", "|C3^(-1)|", "|C2|"],
    );
}

#[test]
fn test_ur_group_symmetric_b7_magnetic_field_bw_c6v_c6() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/b7.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 0.1)));
    test_ur_magnetic_group(&mol, thresh, "C6v", 12, 6, false, BWGRP);
}

#[test]
fn test_ur_group_symmetric_b7_magnetic_field_bw_c6v_c6_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/b7.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 0.1)));
    test_ur_magnetic_group_class_order(
        &mol,
        thresh,
        &["|E|", "2|C6|", "2|C3|", "|C2|", "3|θ·σv|", "3|θ·σv|^(')"],
    );
}

#[test]
#[ignore]
fn test_ur_group_symmetric_arbitrary_half_sandwich_magnetic_field_cn() {
    // env_logger::init();
    let thresh = 1e-7;
    for n in 3..=32 {
        let mut mol = template_molecules::gen_arbitrary_half_sandwich(n);
        mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 0.1)));
        test_ur_ordinary_group(
            &mol,
            thresh,
            format!("C{n}").as_str(),
            n as usize,
            n as usize,
            true,
        );
    }
}

#[test]
#[ignore]
fn test_ur_group_symmetric_arbitrary_half_sandwich_magnetic_field_bw_cnv_cn() {
    let thresh = 1e-7;
    for n in 3..=32 {
        let mut mol = template_molecules::gen_arbitrary_half_sandwich(n);
        mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 0.1)));
        test_ur_magnetic_group(
            &mol,
            thresh,
            format!("C{n}v").as_str(),
            2 * n as usize,
            ({
                if n % 2 == 0 {
                    n / 2 - 1
                } else {
                    n / 2
                }
            } + {
                if n % 2 == 0 {
                    4
                } else {
                    2
                }
            }) as usize,
            false,
            BWGRP,
        );
    }
}

/***
Cn*
***/

#[test]
fn test_ur_group_symmetric_ch4_magnetic_field_c3_double() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/ch4.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(1.0, 1.0, 1.0)));
    test_ur_ordinary_double_group(&mol, thresh, "C3*", 6, 6, true);
}

#[test]
fn test_ur_group_symmetric_ch4_magnetic_field_c3_double_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/ch4.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(1.0, 1.0, 1.0)));
    test_ur_ordinary_double_group_class_order(
        &mol,
        thresh,
        &[
            "|E(Σ)|",
            "|E(QΣ)|",
            "|C3(Σ)|",
            "|C3(QΣ)|",
            "|C3^(-1)(Σ)|",
            "|C3^(-1)(QΣ)|",
        ],
    );
}

#[test]
fn test_ur_group_symmetric_adamantane_magnetic_field_bw_c3v_c3_double() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/adamantane.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.1, 0.1, 0.1)));
    test_ur_magnetic_double_group(&mol, thresh, "C3v*", 12, 6, false, BWGRP2);
}

#[test]
fn test_ur_group_symmetric_adamantane_magnetic_field_bw_c3v_c3_double_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/adamantane.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.1, 0.1, 0.1)));
    test_ur_magnetic_double_group_class_order(
        &mol,
        thresh,
        &[
            "|E(Σ)|",
            "|E(QΣ)|",
            "2|C3(Σ)|",
            "2|C3(QΣ)|",
            "3|θ·σv(Σ)|",
            "3|θ·σv(QΣ)|",
        ],
    );
}

#[test]
fn test_ur_group_symmetric_h8_twisted_magnetic_field_c4_double() {
    let thresh = 1e-7;
    let mut mol = template_molecules::gen_twisted_h8(0.1);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 0.1)));
    test_ur_ordinary_double_group(&mol, thresh, "C4*", 8, 8, true);
}

#[test]
fn test_ur_group_symmetric_h8_twisted_magnetic_field_c4_double_class_order() {
    let thresh = 1e-7;
    let mut mol = template_molecules::gen_twisted_h8(0.1);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 0.1)));
    test_ur_ordinary_double_group_class_order(
        &mol,
        thresh,
        &[
            "|E(Σ)|",
            "|E(QΣ)|",
            "|C4(Σ)|",
            "|C4(QΣ)|",
            "|C4^(-1)(Σ)|",
            "|C4^(-1)(QΣ)|",
            "|C2(Σ)|",
            "|C2(QΣ)|",
        ],
    );
}

#[test]
fn test_ur_group_symmetric_h8_twisted_magnetic_field_bw_d4_c4_double() {
    // env_logger::init();
    let thresh = 1e-7;
    let mut mol = template_molecules::gen_twisted_h8(0.1);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 0.1)));
    test_ur_magnetic_double_group(&mol, thresh, "D4*", 16, 7, false, BWGRP2);
}

#[test]
fn test_ur_group_symmetric_h8_twisted_magnetic_field_bw_d4_c4_double_class_order() {
    let thresh = 1e-7;
    let mut mol = template_molecules::gen_twisted_h8(0.1);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 0.1)));
    test_ur_magnetic_double_group_class_order(
        &mol,
        thresh,
        &[
            "|E(Σ)|",
            "|E(QΣ)|",
            "2|C4(Σ)|",
            "2|C4(QΣ)|",
            "|C2(Σ), C2(QΣ)|",
            "2|θ·C2(Σ), θ·C2(QΣ)|",
            "2|θ·C2(Σ), θ·C2(QΣ)|^(')",
        ],
    );
}

#[test]
fn test_ur_group_symmetric_cpnico_magnetic_field_c5_double() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/cpnico.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, -0.2)));
    test_ur_ordinary_double_group(&mol, thresh, "C5*", 10, 10, true);
}

#[test]
fn test_ur_group_symmetric_cpnico_magnetic_field_c5_double_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/cpnico.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, -0.2)));
    test_ur_ordinary_double_group_class_order(
        &mol,
        thresh,
        &[
            "|E(Σ)|",
            "|E(QΣ)|",
            "|C5(Σ)|",
            "|C5(QΣ)|",
            "|C5^2(Σ)|",
            "|C5^2(QΣ)|",
            "|C5^(-2)(Σ)|",
            "|C5^(-2)(QΣ)|",
            "|C5^(-1)(Σ)|",
            "|C5^(-1)(QΣ)|",
        ],
    );
}

#[test]
fn test_ur_group_symmetric_cpnico_magnetic_field_bw_c5v_c5_double() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/cpnico.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, -0.2)));
    test_ur_magnetic_double_group(&mol, thresh, "C5v*", 20, 8, false, BWGRP2);
}

#[test]
fn test_ur_group_symmetric_cpnico_magnetic_field_bw_c5v_c5_double_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/cpnico.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, -0.2)));
    test_ur_magnetic_double_group_class_order(
        &mol,
        thresh,
        &[
            "|E(Σ)|",
            "|E(QΣ)|",
            "2|C5(Σ)|",
            "2|C5(QΣ)|",
            "2|C5^2(Σ)|",
            "2|C5^2(QΣ)|",
            "5|θ·σv(Σ)|",
            "5|θ·σv(QΣ)|",
        ],
    );
}

#[test]
fn test_ur_group_symmetric_b7_magnetic_field_c6_double() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/b7.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 0.1)));
    test_ur_ordinary_double_group(&mol, thresh, "C6*", 12, 12, true);
}

#[test]
fn test_ur_group_symmetric_b7_magnetic_field_c6_double_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/b7.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 0.1)));
    test_ur_ordinary_double_group_class_order(
        &mol,
        thresh,
        &[
            "|E(Σ)|",
            "|E(QΣ)|",
            "|C6(Σ)|",
            "|C6(QΣ)|",
            "|C6^(-1)(Σ)|",
            "|C6^(-1)(QΣ)|",
            "|C3(Σ)|",
            "|C3(QΣ)|",
            "|C3^(-1)(Σ)|",
            "|C3^(-1)(QΣ)|",
            "|C2(Σ)|",
            "|C2(QΣ)|",
        ],
    );
}

#[test]
fn test_ur_group_symmetric_b7_magnetic_field_bw_c6v_c6_double() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/b7.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 0.1)));
    test_ur_magnetic_double_group(&mol, thresh, "C6v*", 24, 9, false, BWGRP2);
}

#[test]
fn test_ur_group_symmetric_b7_magnetic_field_bw_c6v_c6_double_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/b7.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 0.1)));
    test_ur_magnetic_double_group_class_order(
        &mol,
        thresh,
        &[
            "|E(Σ)|",
            "|E(QΣ)|",
            "2|C6(Σ)|",
            "2|C6(QΣ)|",
            "2|C3(Σ)|",
            "2|C3(QΣ)|",
            "|C2(Σ), C2(QΣ)|",
            "3|θ·σv(Σ), θ·σv(QΣ)|",
            "3|θ·σv(Σ), θ·σv(QΣ)|^(')",
        ],
    );
}

/*
Cnv
*/

#[test]
fn test_ur_group_symmetric_nh3_c3v() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/nh3.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_group(&mol, thresh, "C3v", 6, 3, false);
}

#[test]
fn test_ur_group_symmetric_nh3_c3v_class_order() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/nh3.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_group_class_order(&mol, thresh, &["|E|", "2|C3|", "3|σv|"]);
}

#[test]
fn test_ur_group_symmetric_nh3_grey_c3v() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/nh3.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_magnetic_group(&mol, thresh, "C3v + θ·C3v", 12, 6, false, GRGRP);
}

#[test]
fn test_ur_group_symmetric_nh3_grey_c3v_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/nh3.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_magnetic_group_class_order(
        &mol,
        thresh,
        &["|E|", "2|C3|", "3|σv|", "|θ|", "2|θ·C3|", "3|θ·σv|"],
    );
}

#[test]
fn test_ur_group_symmetric_bf3_electric_field_c3v() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/bf3.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    test_ur_ordinary_group(&mol, thresh, "C3v", 6, 3, false);
}

#[test]
fn test_ur_group_symmetric_bf3_electric_field_c3v_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/bf3.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    test_ur_ordinary_group_class_order(&mol, thresh, &["|E|", "2|C3|", "3|σv|"]);
}

#[test]
fn test_ur_group_symmetric_bf3_electric_field_grey_c3v() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/bf3.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    test_ur_magnetic_group(&mol, thresh, "C3v + θ·C3v", 12, 6, false, GRGRP);
}

#[test]
fn test_ur_group_symmetric_bf3_electric_field_grey_c3v_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/bf3.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    test_ur_magnetic_group_class_order(
        &mol,
        thresh,
        &["|E|", "2|C3|", "3|σv|", "|θ|", "2|θ·C3|", "3|θ·σv|"],
    );
}

#[test]
fn test_ur_group_symmetric_adamantane_electric_field_c3v() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/adamantane.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(Vector3::new(0.1, 0.1, 0.1)));
    test_ur_ordinary_group(&mol, thresh, "C3v", 6, 3, false);
}

#[test]
fn test_ur_group_symmetric_adamantane_electric_field_c3v_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/adamantane.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(Vector3::new(0.1, 0.1, 0.1)));
    test_ur_ordinary_group_class_order(&mol, thresh, &["|E|", "2|C3|", "3|σv|"]);
}

#[test]
fn test_ur_group_symmetric_adamantane_electric_field_grey_c3v() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/adamantane.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(Vector3::new(0.1, 0.1, 0.1)));
    test_ur_magnetic_group(&mol, thresh, "C3v + θ·C3v", 12, 6, false, GRGRP);
}

#[test]
fn test_ur_group_symmetric_adamantane_electric_field_grey_c3v_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/adamantane.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(Vector3::new(0.1, 0.1, 0.1)));
    test_ur_magnetic_group_class_order(
        &mol,
        thresh,
        &["|E|", "2|C3|", "3|σv|", "|θ|", "2|θ·C3|", "3|θ·σv|"],
    );
}

#[test]
fn test_ur_group_symmetric_ch4_electric_field_c3v() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/ch4.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(Vector3::new(1.0, 1.0, 1.0)));
    test_ur_ordinary_group(&mol, thresh, "C3v", 6, 3, false);
}

#[test]
fn test_ur_group_symmetric_ch4_electric_field_c3v_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/ch4.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(Vector3::new(1.0, 1.0, 1.0)));
    test_ur_ordinary_group_class_order(&mol, thresh, &["|E|", "2|C3|", "3|σv|"]);
}

#[test]
fn test_ur_group_symmetric_ch4_electric_field_grey_c3v() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/ch4.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(Vector3::new(1.0, 1.0, 1.0)));
    test_ur_magnetic_group(&mol, thresh, "C3v + θ·C3v", 12, 6, false, GRGRP);
}

#[test]
fn test_ur_group_symmetric_ch4_electric_field_grey_c3v_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/ch4.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(Vector3::new(1.0, 1.0, 1.0)));
    test_ur_magnetic_group_class_order(
        &mol,
        thresh,
        &["|E|", "2|C3|", "3|σv|", "|θ|", "2|θ·C3|", "3|θ·σv|"],
    );
}

#[test]
fn test_ur_group_symmetric_vf6_electric_field_c3v() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/vf6.xyz");
    let thresh = 1e-12;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(0.2 * Vector3::new(1.0, 1.0, 1.0)));
    test_ur_ordinary_group(&mol, thresh, "C3v", 6, 3, false);
}

#[test]
fn test_ur_group_symmetric_vf6_electric_field_c3v_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/vf6.xyz");
    let thresh = 1e-12;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(0.2 * Vector3::new(1.0, 1.0, 1.0)));
    test_ur_ordinary_group_class_order(&mol, thresh, &["|E|", "2|C3|", "3|σv|"]);
}

#[test]
fn test_ur_group_symmetric_vf6_electric_field_grey_c3v() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/vf6.xyz");
    let thresh = 1e-12;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(0.2 * Vector3::new(1.0, 1.0, 1.0)));
    test_ur_magnetic_group(&mol, thresh, "C3v + θ·C3v", 12, 6, false, GRGRP);
}

#[test]
fn test_ur_group_symmetric_vf6_electric_field_grey_c3v_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/vf6.xyz");
    let thresh = 1e-12;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(0.2 * Vector3::new(1.0, 1.0, 1.0)));
    test_ur_magnetic_group_class_order(
        &mol,
        thresh,
        &["|E|", "2|C3|", "3|σv|", "|θ|", "2|θ·C3|", "3|θ·σv|"],
    );
}

#[test]
fn test_ur_group_symmetric_sf5cl_c4v() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/sf5cl.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_group(&mol, thresh, "C4v", 8, 5, false);
}

#[test]
fn test_ur_group_symmetric_sf5cl_c4v_class_order() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/sf5cl.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_group_class_order(
        &mol,
        thresh,
        &["|E|", "2|C4|", "|C2|", "2|σv|", "2|σv|^(')"],
    );
}

#[test]
fn test_ur_group_symmetric_sf5cl_grey_c4v() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/sf5cl.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_magnetic_group(&mol, thresh, "C4v + θ·C4v", 16, 10, false, GRGRP);
}

#[test]
fn test_ur_group_symmetric_sf5cl_grey_c4v_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/sf5cl.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_magnetic_group_class_order(
        &mol,
        thresh,
        &[
            "|E|",
            "2|C4|",
            "|C2|",
            "2|σv|",
            "2|σv|^(')",
            "|θ|",
            "2|θ·C4|",
            "|θ·C2|",
            "2|θ·σv|",
            "2|θ·σv|^(')",
        ],
    );
}

#[test]
fn test_ur_group_symmetric_h8_electric_field_c4v() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/h8.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(0.2 * Vector3::new(0.0, 0.0, 1.0)));
    test_ur_ordinary_group(&mol, thresh, "C4v", 8, 5, false);
}

#[test]
fn test_ur_group_symmetric_h8_electric_field_c4v_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/h8.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(0.2 * Vector3::new(0.0, 0.0, 1.0)));
    test_ur_ordinary_group_class_order(
        &mol,
        thresh,
        &["|E|", "2|C4|", "|C2|", "2|σv|", "2|σv|^(')"],
    );
}

#[test]
fn test_ur_group_symmetric_h8_electric_field_grey_c4v() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/h8.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(0.2 * Vector3::new(0.0, 0.0, 1.0)));
    test_ur_magnetic_group(&mol, thresh, "C4v + θ·C4v", 16, 10, false, GRGRP);
}

#[test]
fn test_ur_group_symmetric_h8_electric_field_grey_c4v_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/h8.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(0.2 * Vector3::new(0.0, 0.0, 1.0)));
    test_ur_magnetic_group_class_order(
        &mol,
        thresh,
        &[
            "|E|",
            "2|C4|",
            "|C2|",
            "2|σv|",
            "2|σv|^(')",
            "|θ|",
            "2|θ·C4|",
            "|θ·C2|",
            "2|θ·σv|",
            "2|θ·σv|^(')",
        ],
    );
}

#[test]
fn test_ur_group_symmetric_vf6_electric_field_c4v() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/vf6.xyz");
    let thresh = 1e-12;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(0.2 * Vector3::new(1.0, 0.0, 0.0)));
    test_ur_ordinary_group(&mol, thresh, "C4v", 8, 5, false);
}

#[test]
fn test_ur_group_symmetric_vf6_electric_field_c4v_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/vf6.xyz");
    let thresh = 1e-12;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(0.2 * Vector3::new(1.0, 0.0, 0.0)));
    test_ur_ordinary_group_class_order(
        &mol,
        thresh,
        &["|E|", "2|C4|", "|C2|", "2|σv|", "2|σv|^(')"],
    );
}

#[test]
fn test_ur_group_symmetric_vf6_electric_field_grey_c4v() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/vf6.xyz");
    let thresh = 1e-12;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(0.2 * Vector3::new(1.0, 0.0, 0.0)));
    test_ur_magnetic_group(&mol, thresh, "C4v + θ·C4v", 16, 10, false, GRGRP);
}

#[test]
fn test_ur_group_symmetric_vf6_electric_field_grey_c4v_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/vf6.xyz");
    let thresh = 1e-12;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(0.2 * Vector3::new(1.0, 0.0, 0.0)));
    test_ur_magnetic_group_class_order(
        &mol,
        thresh,
        &[
            "|E|",
            "2|C4|",
            "|C2|",
            "2|σv|",
            "2|σv|^(')",
            "|θ|",
            "2|θ·C4|",
            "|θ·C2|",
            "2|θ·σv|",
            "2|θ·σv|^(')",
        ],
    );
}

#[test]
fn test_ur_group_symmetric_antiprism_pb10_electric_field_c4v() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/pb10.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    test_ur_ordinary_group(&mol, thresh, "C4v", 8, 5, false);
}

#[test]
fn test_ur_group_symmetric_antiprism_pb10_electric_field_c4v_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/pb10.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    test_ur_ordinary_group_class_order(
        &mol,
        thresh,
        &["|E|", "2|C4|", "|C2|", "2|σv|", "2|σv|^(')"],
    );
}

#[test]
fn test_ur_group_symmetric_antiprism_pb10_electric_field_grey_c4v() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/pb10.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    test_ur_magnetic_group(&mol, thresh, "C4v + θ·C4v", 16, 10, false, GRGRP);
}

#[test]
fn test_ur_group_symmetric_antiprism_pb10_electric_field_grey_c4v_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/pb10.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    test_ur_magnetic_group_class_order(
        &mol,
        thresh,
        &[
            "|E|",
            "2|C4|",
            "|C2|",
            "2|σv|",
            "2|σv|^(')",
            "|θ|",
            "2|θ·C4|",
            "|θ·C2|",
            "2|θ·σv|",
            "2|θ·σv|^(')",
        ],
    );
}

#[test]
fn test_ur_group_symmetric_cpnico_c5v() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/cpnico.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_group(&mol, thresh, "C5v", 10, 4, false);
}

#[test]
fn test_ur_group_symmetric_cpnico_c5v_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/cpnico.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_group_class_order(&mol, thresh, &["|E|", "2|C5|", "2|C5^2|", "5|σv|"]);
}

#[test]
fn test_ur_group_symmetric_cpnico_grey_c5v() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/cpnico.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_magnetic_group(&mol, thresh, "C5v + θ·C5v", 20, 8, false, GRGRP);
}

#[test]
fn test_ur_group_symmetric_cpnico_grey_c5v_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/cpnico.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_magnetic_group_class_order(
        &mol,
        thresh,
        &[
            "|E|",
            "2|C5|",
            "2|C5^2|",
            "5|σv|",
            "|θ|",
            "2|θ·C5|",
            "2|θ·C5^2|",
            "5|θ·σv|",
        ],
    );
}

#[test]
fn test_ur_group_symmetric_staggered_ferrocene_electric_field_c5v() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/staggered_ferrocene.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    test_ur_ordinary_group(&mol, thresh, "C5v", 10, 4, false);
}

#[test]
fn test_ur_group_symmetric_staggered_ferrocene_electric_field_c5v_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/staggered_ferrocene.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    test_ur_ordinary_group_class_order(&mol, thresh, &["|E|", "2|C5|", "2|C5^2|", "5|σv|"]);
}

#[test]
fn test_ur_group_symmetric_staggered_ferrocene_electric_field_grey_c5v() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/staggered_ferrocene.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    test_ur_magnetic_group(&mol, thresh, "C5v + θ·C5v", 20, 8, false, GRGRP);
}

#[test]
fn test_ur_group_symmetric_staggered_ferrocene_electric_field_grey_c5v_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/staggered_ferrocene.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    test_ur_magnetic_group_class_order(
        &mol,
        thresh,
        &[
            "|E|",
            "2|C5|",
            "2|C5^2|",
            "5|σv|",
            "|θ|",
            "2|θ·C5|",
            "2|θ·C5^2|",
            "5|θ·σv|",
        ],
    );
}

#[test]
fn test_ur_group_symmetric_c60_electric_field_c5v() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c60.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    test_ur_ordinary_group(&mol, thresh, "C5v", 10, 4, false);
}

#[test]
fn test_ur_group_symmetric_c60_electric_field_c5v_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c60.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    test_ur_ordinary_group_class_order(&mol, thresh, &["|E|", "2|C5|", "2|C5^2|", "5|σv|"]);
}

#[test]
fn test_ur_group_symmetric_c60_electric_field_grey_c5v() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c60.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    test_ur_magnetic_group(&mol, thresh, "C5v + θ·C5v", 20, 8, false, GRGRP);
}

#[test]
fn test_ur_group_symmetric_c60_electric_field_grey_c5v_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c60.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    test_ur_magnetic_group_class_order(
        &mol,
        thresh,
        &[
            "|E|",
            "2|C5|",
            "2|C5^2|",
            "5|σv|",
            "|θ|",
            "2|θ·C5|",
            "2|θ·C5^2|",
            "5|θ·σv|",
        ],
    );
}

#[test]
fn test_ur_group_symmetric_b7_c6v() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/b7.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_group(&mol, thresh, "C6v", 12, 6, false);
}

#[test]
fn test_ur_group_symmetric_b7_c6v_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/b7.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_group_class_order(
        &mol,
        thresh,
        &["|E|", "2|C6|", "2|C3|", "|C2|", "3|σv|", "3|σv|^(')"],
    );
}

#[test]
fn test_ur_group_symmetric_b7_grey_c6v() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/b7.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_magnetic_group(&mol, thresh, "C6v + θ·C6v", 24, 12, false, GRGRP);
}

#[test]
fn test_ur_group_symmetric_b7_grey_c6v_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/b7.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_magnetic_group_class_order(
        &mol,
        thresh,
        &[
            "|E|",
            "2|C6|",
            "2|C3|",
            "|C2|",
            "3|σv|",
            "3|σv|^(')",
            "|θ|",
            "2|θ·C6|",
            "2|θ·C3|",
            "|θ·C2|",
            "3|θ·σv|",
            "3|θ·σv|^(')",
        ],
    );
}

#[test]
fn test_ur_group_symmetric_au26_electric_field_c6v() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/au26.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    test_ur_ordinary_group(&mol, thresh, "C6v", 12, 6, false);
}

#[test]
fn test_ur_group_symmetric_au26_electric_field_c6v_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/au26.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    test_ur_ordinary_group_class_order(
        &mol,
        thresh,
        &["|E|", "2|C6|", "2|C3|", "|C2|", "3|σv|", "3|σv|^(')"],
    );
}

#[test]
fn test_ur_group_symmetric_au26_electric_field_grey_c6v() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/au26.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    test_ur_magnetic_group(&mol, thresh, "C6v + θ·C6v", 24, 12, false, GRGRP);
}

#[test]
fn test_ur_group_symmetric_au26_electric_field_grey_c6v_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/au26.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    test_ur_magnetic_group_class_order(
        &mol,
        thresh,
        &[
            "|E|",
            "2|C6|",
            "2|C3|",
            "|C2|",
            "3|σv|",
            "3|σv|^(')",
            "|θ|",
            "2|θ·C6|",
            "2|θ·C3|",
            "|θ·C2|",
            "3|θ·σv|",
            "3|θ·σv|^(')",
        ],
    );
}

#[test]
fn test_ur_group_symmetric_benzene_electric_field_c6v() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/benzene.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(Vector3::new(1.0, 0.0, 0.0)));
    test_ur_ordinary_group(&mol, thresh, "C6v", 12, 6, false);
}

#[test]
fn test_ur_group_symmetric_benzene_electric_field_c6v_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/benzene.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(Vector3::new(1.0, 0.0, 0.0)));
    test_ur_ordinary_group_class_order(
        &mol,
        thresh,
        &["|E|", "2|C6|", "2|C3|", "|C2|", "3|σv|", "3|σv|^(')"],
    );
}

#[test]
fn test_ur_group_symmetric_benzene_electric_field_grey_c6v() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/benzene.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(Vector3::new(1.0, 0.0, 0.0)));
    test_ur_magnetic_group(&mol, thresh, "C6v + θ·C6v", 24, 12, false, GRGRP);
}

#[test]
fn test_ur_group_symmetric_benzene_electric_field_grey_c6v_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/benzene.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(Vector3::new(1.0, 0.0, 0.0)));
    test_ur_magnetic_group_class_order(
        &mol,
        thresh,
        &[
            "|E|",
            "2|C6|",
            "2|C3|",
            "|C2|",
            "3|σv|",
            "3|σv|^(')",
            "|θ|",
            "2|θ·C6|",
            "2|θ·C3|",
            "|θ·C2|",
            "3|θ·σv|",
            "3|θ·σv|^(')",
        ],
    );
}

#[test]
#[ignore]
fn test_ur_group_symmetric_arbitrary_half_sandwich_cnv() {
    /* The expected number of classes is deduced from the irrep structures of
     * the Cnv groups.
     * When n is even, the irreps are A1, A2, B1, B2, Ek where k = 1, ..., n/2 - 1.
     * When n is odd, the irreps are A1, A2, Ek where k = 1, ..., n//2.
     */
    for n in 3..=32 {
        let mol = template_molecules::gen_arbitrary_half_sandwich(n);
        let thresh = 1e-7;
        test_ur_ordinary_group(
            &mol,
            thresh,
            format!("C{n}v").as_str(),
            2 * n as usize,
            ({
                if n % 2 == 0 {
                    n / 2 - 1
                } else {
                    n / 2
                }
            } + {
                if n % 2 == 0 {
                    4
                } else {
                    2
                }
            }) as usize,
            false,
        );
    }
}

#[test]
#[ignore]
fn test_ur_group_symmetric_arbitrary_half_sandwich_grey_cnv() {
    /* The expected number of classes is deduced from the irrep structures of
     * the Cnv groups.
     * When n is even, the irreps are A1, A2, B1, B2, Ek where k = 1, ..., n/2 - 1.
     * When n is odd, the irreps are A1, A2, Ek where k = 1, ..., n//2.
     */
    for n in 3..=32 {
        let mol = template_molecules::gen_arbitrary_half_sandwich(n);
        let thresh = 1e-7;
        test_ur_magnetic_group(
            &mol,
            thresh,
            format!("C{n}v + θ·C{n}v").as_str(),
            4 * n as usize,
            2 * ({
                if n % 2 == 0 {
                    n / 2 - 1
                } else {
                    n / 2
                }
            } + {
                if n % 2 == 0 {
                    4
                } else {
                    2
                }
            }) as usize,
            false,
            GRGRP,
        );
    }
}

#[test]
#[ignore]
fn test_ur_group_symmetric_arbitrary_staggered_sandwich_electric_field_cnv() {
    /* The expected number of classes is deduced from the irrep structures of
     * the Cnv groups.
     * When n is even, the irreps are A1, A2, B1, B2, Ek where k = 1, ..., n/2 - 1.
     * When n is odd, the irreps are A1, A2, Ek where k = 1, ..., n//2.
     */
    for n in 3..=20 {
        let mut mol = template_molecules::gen_arbitrary_twisted_sandwich(n, 0.5);
        mol.set_electric_field(Some(Vector3::new(0.0, 0.0, 1.0)));
        let thresh = 1e-7;
        test_ur_ordinary_group(
            &mol,
            thresh,
            format!("C{n}v").as_str(),
            2 * n as usize,
            ({
                if n % 2 == 0 {
                    n / 2 - 1
                } else {
                    n / 2
                }
            } + {
                if n % 2 == 0 {
                    4
                } else {
                    2
                }
            }) as usize,
            false,
        );
    }
}

#[test]
#[ignore]
fn test_ur_group_symmetric_arbitrary_staggered_sandwich_electric_field_grey_cnv() {
    /* The expected number of classes is deduced from the irrep structures of
     * the Cnv groups.
     * When n is even, the irreps are A1, A2, B1, B2, Ek where k = 1, ..., n/2 - 1.
     * When n is odd, the irreps are A1, A2, Ek where k = 1, ..., n//2.
     */
    for n in 3..=20 {
        let mut mol = template_molecules::gen_arbitrary_twisted_sandwich(n, 0.5);
        mol.set_electric_field(Some(Vector3::new(0.0, 0.0, 1.0)));
        let thresh = 1e-7;
        test_ur_magnetic_group(
            &mol,
            thresh,
            format!("C{n}v + θ·C{n}v").as_str(),
            4 * n as usize,
            2 * ({
                if n % 2 == 0 {
                    n / 2 - 1
                } else {
                    n / 2
                }
            } + {
                if n % 2 == 0 {
                    4
                } else {
                    2
                }
            }) as usize,
            false,
            GRGRP,
        );
    }
}

/***
Cnv*
***/

#[test]
fn test_ur_group_symmetric_nh3_c3v_double() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/nh3.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_double_group(&mol, thresh, "C3v*", 12, 6, false);
}

#[test]
fn test_ur_group_symmetric_nh3_c3v_double_class_order() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/nh3.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_double_group_class_order(
        &mol,
        thresh,
        &[
            "|E(Σ)|",
            "|E(QΣ)|",
            "2|C3(Σ)|",
            "2|C3(QΣ)|",
            "3|σv(Σ)|",
            "3|σv(QΣ)|",
        ],
    );
}

#[test]
fn test_ur_group_symmetric_vf6_electric_field_grey_c3v_double() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/vf6.xyz");
    let thresh = 1e-12;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(0.2 * Vector3::new(1.0, 1.0, 1.0)));
    test_ur_magnetic_double_group(&mol, thresh, "(C3v + θ·C3v)*", 24, 12, false, GRGRP2);
}

#[test]
fn test_ur_group_symmetric_vf6_electric_field_grey_c3v_double_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/vf6.xyz");
    let thresh = 1e-12;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(0.2 * Vector3::new(1.0, 1.0, 1.0)));
    test_ur_magnetic_double_group_class_order(
        &mol,
        thresh,
        &[
            "|E(Σ)|",
            "|E(QΣ)|",
            "2|C3(Σ)|",
            "2|C3(QΣ)|",
            "3|σv(Σ)|",
            "3|σv(QΣ)|",
            "|θ(Σ)|",
            "|θ(QΣ)|",
            "2|θ·C3(Σ)|",
            "2|θ·C3(QΣ)|",
            "3|θ·σv(Σ)|",
            "3|θ·σv(QΣ)|",
        ],
    );
}

#[test]
fn test_ur_group_symmetric_antiprism_pb10_electric_field_c4v_double() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/pb10.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    test_ur_ordinary_double_group(&mol, thresh, "C4v*", 16, 7, false);
}

#[test]
fn test_ur_group_symmetric_antiprism_pb10_electric_field_c4v_double_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/pb10.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    test_ur_ordinary_double_group_class_order(
        &mol,
        thresh,
        &[
            "|E(Σ)|",
            "|E(QΣ)|",
            "2|C4(Σ)|",
            "2|C4(QΣ)|",
            "|C2(Σ), C2(QΣ)|",
            "2|σv(Σ), σv(QΣ)|",
            "2|σv(Σ), σv(QΣ)|^(')",
        ],
    );
}

#[test]
fn test_ur_group_symmetric_antiprism_pb10_electric_field_grey_c4v_double() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/pb10.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    test_ur_magnetic_double_group(&mol, thresh, "(C4v + θ·C4v)*", 32, 14, false, GRGRP2);
}

#[test]
fn test_ur_group_symmetric_antiprism_pb10_electric_field_grey_c4v_double_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/pb10.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    test_ur_magnetic_double_group_class_order(
        &mol,
        thresh,
        &[
            "|E(Σ)|",
            "|E(QΣ)|",
            "2|C4(Σ)|",
            "2|C4(QΣ)|",
            "|C2(Σ), C2(QΣ)|",
            "2|σv(Σ), σv(QΣ)|",
            "2|σv(Σ), σv(QΣ)|^(')",
            "|θ(Σ)|",
            "|θ(QΣ)|",
            "2|θ·C4(Σ)|",
            "2|θ·C4(QΣ)|",
            "|θ·C2(Σ), θ·C2(QΣ)|",
            "2|θ·σv(Σ), θ·σv(QΣ)|",
            "2|θ·σv(Σ), θ·σv(QΣ)|^(')",
        ],
    );
}

#[test]
fn test_ur_group_symmetric_c60_electric_field_c5v_double() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c60.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    test_ur_ordinary_double_group(&mol, thresh, "C5v*", 20, 8, false);
}

#[test]
fn test_ur_group_symmetric_c60_electric_field_c5v_double_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c60.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    test_ur_ordinary_double_group_class_order(
        &mol,
        thresh,
        &[
            "|E(Σ)|",
            "|E(QΣ)|",
            "2|C5(Σ)|",
            "2|C5(QΣ)|",
            "2|C5^2(Σ)|",
            "2|C5^2(QΣ)|",
            "5|σv(Σ)|",
            "5|σv(QΣ)|",
        ],
    );
}

#[test]
fn test_ur_group_symmetric_c60_electric_field_grey_c5v_double() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c60.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    test_ur_magnetic_double_group(&mol, thresh, "(C5v + θ·C5v)*", 40, 16, false, GRGRP2);
}

#[test]
fn test_ur_group_symmetric_c60_electric_field_grey_c5v_double_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c60.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    test_ur_magnetic_double_group_class_order(
        &mol,
        thresh,
        &[
            "|E(Σ)|",
            "|E(QΣ)|",
            "2|C5(Σ)|",
            "2|C5(QΣ)|",
            "2|C5^2(Σ)|",
            "2|C5^2(QΣ)|",
            "5|σv(Σ)|",
            "5|σv(QΣ)|",
            "|θ(Σ)|",
            "|θ(QΣ)|",
            "2|θ·C5(Σ)|",
            "2|θ·C5(QΣ)|",
            "2|θ·C5^2(Σ)|",
            "2|θ·C5^2(QΣ)|",
            "5|θ·σv(Σ)|",
            "5|θ·σv(QΣ)|",
        ],
    );
}

#[test]
fn test_ur_group_symmetric_benzene_electric_field_c6v_double() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/benzene.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(Vector3::new(1.0, 0.0, 0.0)));
    test_ur_ordinary_double_group(&mol, thresh, "C6v*", 24, 9, false);
}

#[test]
fn test_ur_group_symmetric_benzene_electric_field_c6v_double_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/benzene.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(Vector3::new(1.0, 0.0, 0.0)));
    test_ur_ordinary_double_group_class_order(
        &mol,
        thresh,
        &[
            "|E(Σ)|",
            "|E(QΣ)|",
            "2|C6(Σ)|",
            "2|C6(QΣ)|",
            "2|C3(Σ)|",
            "2|C3(QΣ)|",
            "|C2(Σ), C2(QΣ)|",
            "3|σv(Σ), σv(QΣ)|",
            "3|σv(Σ), σv(QΣ)|^(')",
        ],
    );
}

#[test]
fn test_ur_group_symmetric_benzene_electric_field_grey_c6v_double() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/benzene.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(Vector3::new(1.0, 0.0, 0.0)));
    test_ur_magnetic_double_group(&mol, thresh, "(C6v + θ·C6v)*", 48, 18, false, GRGRP2);
}

#[test]
fn test_ur_group_symmetric_benzene_electric_field_grey_c6v_double_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/benzene.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(Vector3::new(1.0, 0.0, 0.0)));
    test_ur_magnetic_double_group_class_order(
        &mol,
        thresh,
        &[
            "|E(Σ)|",
            "|E(QΣ)|",
            "2|C6(Σ)|",
            "2|C6(QΣ)|",
            "2|C3(Σ)|",
            "2|C3(QΣ)|",
            "|C2(Σ), C2(QΣ)|",
            "3|σv(Σ), σv(QΣ)|",
            "3|σv(Σ), σv(QΣ)|^(')",
            "|θ(Σ)|",
            "|θ(QΣ)|",
            "2|θ·C6(Σ)|",
            "2|θ·C6(QΣ)|",
            "2|θ·C3(Σ)|",
            "2|θ·C3(QΣ)|",
            "|θ·C2(Σ), θ·C2(QΣ)|",
            "3|θ·σv(Σ), θ·σv(QΣ)|",
            "3|θ·σv(Σ), θ·σv(QΣ)|^(')",
        ],
    );
}

/*
Cnh
*/

#[test]
fn test_ur_group_symmetric_bf3_magnetic_field_c3h() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/bf3.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    test_ur_ordinary_group(&mol, thresh, "C3h", 6, 6, true);
}

#[test]
fn test_ur_group_symmetric_bf3_magnetic_field_c3h_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/bf3.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    test_ur_ordinary_group_class_order(
        &mol,
        thresh,
        &["|E|", "|C3|", "|C3^(-1)|", "|S3|", "|S3^(-1)|", "|σh|"],
    );
}

#[test]
fn test_ur_group_symmetric_bf3_magnetic_field_bw_d3h_c3h() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/bf3.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    test_ur_magnetic_group(&mol, thresh, "D3h", 12, 6, false, BWGRP);
}

#[test]
fn test_ur_group_symmetric_bf3_magnetic_field_bw_d3h_c3h_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/bf3.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    test_ur_magnetic_group_class_order(
        &mol,
        thresh,
        &["|E|", "2|C3|", "2|S3|", "|σh|", "3|θ·C2|", "3|θ·σv|"],
    );
}

#[test]
fn test_ur_group_symmetric_xef4_magnetic_field_c4h() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/xef4.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, -1.0)));
    test_ur_ordinary_group(&mol, thresh, "C4h", 8, 8, true);
}

#[test]
fn test_ur_group_symmetric_xef4_magnetic_field_c4h_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/xef4.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, -1.0)));
    test_ur_ordinary_group_class_order(
        &mol,
        thresh,
        &[
            "|E|",
            "|C4|",
            "|C4^(-1)|",
            "|C2|",
            "|i|",
            "|S4|",
            "|S4^(-1)|",
            "|σh|",
        ],
    );
}

#[test]
fn test_ur_group_symmetric_xef4_magnetic_field_bw_d4h_c4h() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/xef4.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, -1.0)));
    test_ur_magnetic_group(&mol, thresh, "D4h", 16, 10, false, BWGRP);
}

#[test]
fn test_ur_group_symmetric_xef4_magnetic_field_bw_d4h_c4h_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/xef4.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, -1.0)));
    test_ur_magnetic_group_class_order(
        &mol,
        thresh,
        &[
            "|E|",
            "2|C4|",
            "|C2|",
            "|i|",
            "2|S4|",
            "|σh|",
            "2|θ·C2|",
            "2|θ·C2|^(')",
            "2|θ·σv|",
            "2|θ·σv|^(')",
        ],
    );
}

#[test]
fn test_ur_group_symmetric_vf6_magnetic_field_c4h() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/vf6.xyz");
    let thresh = 1e-12;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    test_ur_ordinary_group(&mol, thresh, "C4h", 8, 8, true);
}

#[test]
fn test_ur_group_symmetric_vf6_magnetic_field_c4h_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/vf6.xyz");
    let thresh = 1e-12;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    test_ur_ordinary_group_class_order(
        &mol,
        thresh,
        &[
            "|E|",
            "|C4|",
            "|C4^(-1)|",
            "|C2|",
            "|i|",
            "|S4|",
            "|S4^(-1)|",
            "|σh|",
        ],
    );
}

#[test]
fn test_ur_group_symmetric_vf6_magnetic_field_bw_d4h_c4h() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/vf6.xyz");
    let thresh = 1e-12;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    test_ur_magnetic_group(&mol, thresh, "D4h", 16, 10, false, BWGRP);
}

#[test]
fn test_ur_group_symmetric_vf6_magnetic_field_bw_d4h_c4h_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/vf6.xyz");
    let thresh = 1e-12;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    test_ur_magnetic_group_class_order(
        &mol,
        thresh,
        &[
            "|E|",
            "2|C4|",
            "|C2|",
            "|i|",
            "2|S4|",
            "|σh|",
            "2|θ·C2|",
            "2|θ·C2|^(')",
            "2|θ·σv|",
            "2|θ·σv|^(')",
        ],
    );
}

#[test]
fn test_ur_group_symmetric_h8_magnetic_field_c4h() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/h8.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, 1e-7);
    mol.set_magnetic_field(Some(0.2 * Vector3::new(0.0, 0.0, 1.0)));
    test_ur_ordinary_group(&mol, thresh, "C4h", 8, 8, true);
}

#[test]
fn test_ur_group_symmetric_h8_magnetic_field_c4h_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/h8.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, 1e-7);
    mol.set_magnetic_field(Some(0.2 * Vector3::new(0.0, 0.0, 1.0)));
    test_ur_ordinary_group_class_order(
        &mol,
        thresh,
        &[
            "|E|",
            "|C4|",
            "|C4^(-1)|",
            "|C2|",
            "|i|",
            "|S4|",
            "|S4^(-1)|",
            "|σh|",
        ],
    );
}

#[test]
fn test_ur_group_symmetric_h8_magnetic_field_bw_d4h_c4h() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/h8.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, 1e-7);
    mol.set_magnetic_field(Some(0.2 * Vector3::new(0.0, 0.0, 1.0)));
    test_ur_magnetic_group(&mol, thresh, "D4h", 16, 10, false, BWGRP);
}

#[test]
fn test_ur_group_symmetric_h8_magnetic_field_bw_d4h_c4h_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/h8.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, 1e-7);
    mol.set_magnetic_field(Some(0.2 * Vector3::new(0.0, 0.0, 1.0)));
    test_ur_magnetic_group_class_order(
        &mol,
        thresh,
        &[
            "|E|",
            "2|C4|",
            "|C2|",
            "|i|",
            "2|S4|",
            "|σh|",
            "2|θ·C2|",
            "2|θ·C2|^(')",
            "2|θ·σv|",
            "2|θ·σv|^(')",
        ],
    );
}

#[test]
fn test_ur_group_symmetric_eclipsed_ferrocene_magnetic_field_c5h() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/eclipsed_ferrocene.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, -1.0)));
    test_ur_ordinary_group(&mol, thresh, "C5h", 10, 10, true);
}

#[test]
fn test_ur_group_symmetric_eclipsed_ferrocene_magnetic_field_c5h_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/eclipsed_ferrocene.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, -1.0)));
    test_ur_ordinary_group_class_order(
        &mol,
        thresh,
        &[
            "|E|",
            "|C5|",
            "|C5^2|",
            "|C5^(-2)|",
            "|C5^(-1)|",
            "|S5|",
            "|σC5^2|",
            "|σC5^(-2)|",
            "|S5^(-1)|",
            "|σh|",
        ],
    );
}

#[test]
fn test_ur_group_symmetric_eclipsed_ferrocene_magnetic_field_bw_d5h_c5h() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/eclipsed_ferrocene.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, -1.0)));
    test_ur_magnetic_group(&mol, thresh, "D5h", 20, 8, false, BWGRP);
}

#[test]
fn test_ur_group_symmetric_eclipsed_ferrocene_magnetic_field_bw_d5h_c5h_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/eclipsed_ferrocene.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, -1.0)));
    test_ur_magnetic_group_class_order(
        &mol,
        thresh,
        &[
            "|E|",
            "2|C5|",
            "2|C5^2|",
            "2|S5|",
            "2|σC5^2|",
            "|σh|",
            "5|θ·C2|",
            "5|θ·σv|",
        ],
    );
}

#[test]
fn test_ur_group_symmetric_benzene_magnetic_field_c6h() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/benzene.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(1.0, 0.0, 0.0)));
    test_ur_ordinary_group(&mol, thresh, "C6h", 12, 12, true);
}

#[test]
fn test_ur_group_symmetric_benzene_magnetic_field_c6h_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/benzene.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(1.0, 0.0, 0.0)));
    test_ur_ordinary_group_class_order(
        &mol,
        thresh,
        &[
            "|E|",
            "|C6|",
            "|C6^(-1)|",
            "|C3|",
            "|C3^(-1)|",
            "|C2|",
            "|i|",
            "|S6|",
            "|S6^(-1)|",
            "|S3|",
            "|S3^(-1)|",
            "|σh|",
        ],
    );
}

#[test]
fn test_ur_group_symmetric_benzene_magnetic_field_bw_d6h_c6h() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/benzene.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(1.0, 0.0, 0.0)));
    test_ur_magnetic_group(&mol, thresh, "D6h", 24, 12, false, BWGRP);
}

#[test]
fn test_ur_group_symmetric_benzene_magnetic_field_bw_d6h_c6h_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/benzene.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(1.0, 0.0, 0.0)));
    test_ur_magnetic_group_class_order(
        &mol,
        thresh,
        &[
            "|E|",
            "2|C6|",
            "2|C3|",
            "|C2|",
            "|i|",
            "2|S6|",
            "2|S3|",
            "|σh|",
            "3|θ·C2|",
            "3|θ·C2|^(')",
            "3|θ·σv|",
            "3|θ·σv|^(')",
        ],
    );
}

#[test]
#[ignore]
fn test_ur_group_symmetric_arbitrary_eclipsed_sandwich_magnetic_field_cnh() {
    // env_logger::init();
    for n in 3..=20 {
        let mut mol = template_molecules::gen_arbitrary_eclipsed_sandwich(n);
        let thresh = 1e-7;
        mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 0.1)));
        test_ur_ordinary_group(
            &mol,
            thresh,
            format!("C{n}h").as_str(),
            2 * n as usize,
            2 * n as usize,
            true,
        );
    }
}

#[test]
#[ignore]
fn test_ur_group_symmetric_arbitrary_eclipsed_sandwich_magnetic_field_bw_dnh_cnh() {
    // env_logger::init();
    for n in 3..=20 {
        let mut mol = template_molecules::gen_arbitrary_eclipsed_sandwich(n);
        let thresh = 1e-7;
        mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 0.1)));
        test_ur_magnetic_group(
            &mol,
            thresh,
            format!("D{n}h").as_str(),
            4 * n as usize,
            2 * ({
                if n % 2 == 0 {
                    n / 2 - 1
                } else {
                    n / 2
                }
            } + {
                if n % 2 == 0 {
                    4
                } else {
                    2
                }
            }) as usize,
            false,
            BWGRP,
        );
    }
}

/***
Cnh*
***/

#[test]
fn test_ur_group_symmetric_bf3_magnetic_field_c3h_double() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/bf3.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    test_ur_ordinary_double_group(&mol, thresh, "C3h*", 12, 12, true);
}

#[test]
fn test_ur_group_symmetric_bf3_magnetic_field_c3h_double_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/bf3.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    test_ur_ordinary_double_group_class_order(
        &mol,
        thresh,
        &[
            "|E(Σ)|",
            "|E(QΣ)|",
            "|C3(Σ)|",
            "|C3(QΣ)|",
            "|C3^(-1)(Σ)|",
            "|C3^(-1)(QΣ)|",
            "|S3(Σ)|",
            "|S3(QΣ)|",
            "|S3^(-1)(Σ)|",
            "|S3^(-1)(QΣ)|",
            "|σh(Σ)|",
            "|σh(QΣ)|",
        ],
    );
}

#[test]
fn test_ur_group_symmetric_bf3_magnetic_field_bw_d3h_c3h_double() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/bf3.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    test_ur_magnetic_double_group(&mol, thresh, "D3h*", 24, 9, false, BWGRP2);
}

#[test]
fn test_ur_group_symmetric_bf3_magnetic_field_bw_d3h_c3h_double_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/bf3.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    test_ur_magnetic_double_group_class_order(
        &mol,
        thresh,
        &[
            "|E(Σ)|",
            "|E(QΣ)|",
            "2|C3(Σ)|",
            "2|C3(QΣ)|",
            "2|S3(Σ)|",
            "2|S3(QΣ)|",
            "|σh(Σ), σh(QΣ)|",
            "3|θ·C2(Σ), θ·C2(QΣ)|",
            "3|θ·σv(Σ), θ·σv(QΣ)|",
        ],
    );
}

#[test]
fn test_ur_group_symmetric_xef4_magnetic_field_c4h_double() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/xef4.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, -1.0)));
    test_ur_ordinary_double_group(&mol, thresh, "C4h*", 16, 16, true);
}

#[test]
fn test_ur_group_symmetric_xef4_magnetic_field_c4h_double_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/xef4.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, -1.0)));
    test_ur_ordinary_double_group_class_order(
        &mol,
        thresh,
        &[
            "|E(Σ)|",
            "|E(QΣ)|",
            "|C4(Σ)|",
            "|C4(QΣ)|",
            "|C4^(-1)(Σ)|",
            "|C4^(-1)(QΣ)|",
            "|C2(Σ)|",
            "|C2(QΣ)|",
            "|i(Σ)|",
            "|i(QΣ)|",
            "|S4(Σ)|",
            "|S4(QΣ)|",
            "|S4^(-1)(Σ)|",
            "|S4^(-1)(QΣ)|",
            "|σh(Σ)|",
            "|σh(QΣ)|",
        ],
    );
}

#[test]
fn test_ur_group_symmetric_h8_magnetic_field_bw_d4h_c4h_double() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/h8.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, 1e-7);
    mol.set_magnetic_field(Some(0.2 * Vector3::new(0.0, 0.0, 1.0)));
    test_ur_magnetic_double_group(&mol, thresh, "D4h*", 32, 14, false, BWGRP2);
}

#[test]
fn test_ur_group_symmetric_h8_magnetic_field_bw_d4h_c4h_double_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/h8.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, 1e-7);
    mol.set_magnetic_field(Some(0.2 * Vector3::new(0.0, 0.0, 1.0)));
    test_ur_magnetic_double_group_class_order(
        &mol,
        thresh,
        &[
            "|E(Σ)|",
            "|E(QΣ)|",
            "2|C4(Σ)|",
            "2|C4(QΣ)|",
            "|C2(Σ), C2(QΣ)|",
            "|i(Σ)|",
            "|i(QΣ)|",
            "2|S4(Σ)|",
            "2|S4(QΣ)|",
            "|σh(Σ), σh(QΣ)|",
            "2|θ·C2(Σ), θ·C2(QΣ)|",
            "2|θ·C2(Σ), θ·C2(QΣ)|^(')",
            "2|θ·σv(Σ), θ·σv(QΣ)|",
            "2|θ·σv(Σ), θ·σv(QΣ)|^(')",
        ],
    );
}

#[test]
fn test_ur_group_symmetric_eclipsed_ferrocene_magnetic_field_c5h_double() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/eclipsed_ferrocene.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, -1.0)));
    test_ur_ordinary_double_group(&mol, thresh, "C5h*", 20, 20, true);
}

#[test]
fn test_ur_group_symmetric_eclipsed_ferrocene_magnetic_field_c5h_double_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/eclipsed_ferrocene.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, -1.0)));
    test_ur_ordinary_double_group_class_order(
        &mol,
        thresh,
        &[
            "|E(Σ)|",
            "|E(QΣ)|",
            "|C5(Σ)|",
            "|C5(QΣ)|",
            "|C5^2(Σ)|",
            "|C5^2(QΣ)|",
            "|C5^(-2)(Σ)|",
            "|C5^(-2)(QΣ)|",
            "|C5^(-1)(Σ)|",
            "|C5^(-1)(QΣ)|",
            "|S5(Σ)|",
            "|S5(QΣ)|",
            "|σC5^2(Σ)|",
            "|σC5^2(QΣ)|",
            "|σC5^(-2)(Σ)|",
            "|σC5^(-2)(QΣ)|",
            "|S5^(-1)(Σ)|",
            "|S5^(-1)(QΣ)|",
            "|σh(Σ)|",
            "|σh(QΣ)|",
        ],
    );
}

#[test]
fn test_ur_group_symmetric_eclipsed_ferrocene_magnetic_field_bw_d5h_c5h_double() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/eclipsed_ferrocene.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, -1.0)));
    test_ur_magnetic_double_group(&mol, thresh, "D5h*", 40, 13, false, BWGRP2);
}

#[test]
fn test_ur_group_symmetric_eclipsed_ferrocene_magnetic_field_bw_d5h_c5h_double_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/eclipsed_ferrocene.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, -1.0)));
    test_ur_magnetic_double_group_class_order(
        &mol,
        thresh,
        &[
            "|E(Σ)|",
            "|E(QΣ)|",
            "2|C5(Σ)|",
            "2|C5(QΣ)|",
            "2|C5^2(Σ)|",
            "2|C5^2(QΣ)|",
            "2|S5(Σ)|",
            "2|S5(QΣ)|",
            "2|σC5^2(Σ)|",
            "2|σC5^2(QΣ)|",
            "|σh(Σ), σh(QΣ)|",
            "5|θ·C2(Σ), θ·C2(QΣ)|",
            "5|θ·σv(Σ), θ·σv(QΣ)|",
        ],
    );
}

#[test]
fn test_ur_group_symmetric_benzene_magnetic_field_c6h_double() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/benzene.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(1.0, 0.0, 0.0)));
    test_ur_ordinary_double_group(&mol, thresh, "C6h*", 24, 24, true);
}

#[test]
fn test_ur_group_symmetric_benzene_magnetic_field_c6h_double_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/benzene.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(1.0, 0.0, 0.0)));
    test_ur_ordinary_double_group_class_order(
        &mol,
        thresh,
        &[
            "|E(Σ)|",
            "|E(QΣ)|",
            "|C6(Σ)|",
            "|C6(QΣ)|",
            "|C6^(-1)(Σ)|",
            "|C6^(-1)(QΣ)|",
            "|C3(Σ)|",
            "|C3(QΣ)|",
            "|C3^(-1)(Σ)|",
            "|C3^(-1)(QΣ)|",
            "|C2(Σ)|",
            "|C2(QΣ)|",
            "|i(Σ)|",
            "|i(QΣ)|",
            "|S6(Σ)|",
            "|S6(QΣ)|",
            "|S6^(-1)(Σ)|",
            "|S6^(-1)(QΣ)|",
            "|S3(Σ)|",
            "|S3(QΣ)|",
            "|S3^(-1)(Σ)|",
            "|S3^(-1)(QΣ)|",
            "|σh(Σ)|",
            "|σh(QΣ)|",
        ],
    );
}

#[test]
fn test_ur_group_symmetric_benzene_magnetic_field_bw_d6h_c6h_double() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/benzene.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(1.0, 0.0, 0.0)));
    test_ur_magnetic_double_group(&mol, thresh, "D6h*", 48, 18, false, BWGRP2);
}

#[test]
fn test_ur_group_symmetric_benzene_magnetic_field_bw_d6h_c6h_double_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/benzene.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(1.0, 0.0, 0.0)));
    test_ur_magnetic_double_group_class_order(
        &mol,
        thresh,
        &[
            "|E(Σ)|",
            "|E(QΣ)|",
            "2|C6(Σ)|",
            "2|C6(QΣ)|",
            "2|C3(Σ)|",
            "2|C3(QΣ)|",
            "|C2(Σ), C2(QΣ)|",
            "|i(Σ)|",
            "|i(QΣ)|",
            "2|S6(Σ)|",
            "2|S6(QΣ)|",
            "2|S3(Σ)|",
            "2|S3(QΣ)|",
            "|σh(Σ), σh(QΣ)|",
            "3|θ·C2(Σ), θ·C2(QΣ)|",
            "3|θ·C2(Σ), θ·C2(QΣ)|^(')",
            "3|θ·σv(Σ), θ·σv(QΣ)|",
            "3|θ·σv(Σ), θ·σv(QΣ)|^(')",
        ],
    );
}

/*
Dn
*/

#[test]
fn test_ur_group_symmetric_triphenyl_radical_d3() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/triphenylradical.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_group(&mol, thresh, "D3", 6, 3, false);
}

#[test]
fn test_ur_group_symmetric_triphenyl_radical_d3_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/triphenylradical.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_group_class_order(&mol, thresh, &["|E|", "2|C3|", "3|C2|"]);
}

#[test]
fn test_ur_group_symmetric_triphenyl_radical_grey_d3() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/triphenylradical.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_magnetic_group(&mol, thresh, "D3 + θ·D3", 12, 6, false, GRGRP);
}

#[test]
fn test_ur_group_symmetric_triphenyl_radical_grey_d3_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/triphenylradical.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_magnetic_group_class_order(
        &mol,
        thresh,
        &["|E|", "2|C3|", "3|C2|", "|θ|", "2|θ·C3|", "3|θ·C2|"],
    );
}

#[test]
fn test_ur_group_symmetric_h8_twisted_d4() {
    let thresh = 1e-7;
    let mol = template_molecules::gen_twisted_h8(0.1);
    test_ur_ordinary_group(&mol, thresh, "D4", 8, 5, false);
}

#[test]
fn test_ur_group_symmetric_h8_twisted_d4_class_order() {
    let thresh = 1e-7;
    let mol = template_molecules::gen_twisted_h8(0.1);
    test_ur_ordinary_group_class_order(
        &mol,
        thresh,
        &["|E|", "2|C4|", "|C2|", "2|C2|^(')", "2|C2|^('')"],
    );
}

#[test]
fn test_ur_group_symmetric_h8_twisted_grey_d4() {
    let thresh = 1e-7;
    let mol = template_molecules::gen_twisted_h8(0.1);
    test_ur_magnetic_group(&mol, thresh, "D4 + θ·D4", 16, 10, false, GRGRP);
}

#[test]
fn test_ur_group_symmetric_h8_twisted_grey_d4_class_order() {
    let thresh = 1e-7;
    let mol = template_molecules::gen_twisted_h8(0.1);
    test_ur_magnetic_group_class_order(
        &mol,
        thresh,
        &[
            "|E|",
            "2|C4|",
            "|C2|",
            "2|C2|^(')",
            "2|C2|^('')",
            "|θ|",
            "2|θ·C4|",
            "|θ·C2|",
            "2|θ·C2|^(')",
            "2|θ·C2|^('')",
        ],
    );
}

#[test]
fn test_ur_group_symmetric_c5ph5_d5() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c5ph5.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_group(&mol, thresh, "D5", 10, 4, false);
}

#[test]
fn test_ur_group_symmetric_c5ph5_d5_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c5ph5.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_group_class_order(&mol, thresh, &["|E|", "2|C5|", "2|C5^2|", "5|C2|"]);
}

#[test]
fn test_ur_group_symmetric_c5ph5_grey_d5() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c5ph5.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_magnetic_group(&mol, thresh, "D5 + θ·D5", 20, 8, false, GRGRP);
}

#[test]
fn test_ur_group_symmetric_c5ph5_grey_d5_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c5ph5.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_magnetic_group_class_order(
        &mol,
        thresh,
        &[
            "|E|",
            "2|C5|",
            "2|C5^2|",
            "5|C2|",
            "|θ|",
            "2|θ·C5|",
            "2|θ·C5^2|",
            "5|θ·C2|",
        ],
    );
}

#[test]
fn test_ur_group_symmetric_c6ph6_d6() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c6ph6.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_group(&mol, thresh, "D6", 12, 6, false);
}

#[test]
fn test_ur_group_symmetric_c6ph6_d6_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c6ph6.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_group_class_order(
        &mol,
        thresh,
        &["|E|", "2|C6|", "2|C3|", "|C2|", "3|C2|^(')", "3|C2|^('')"],
    );
}

#[test]
fn test_ur_group_symmetric_c6ph6_grey_d6() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c6ph6.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_magnetic_group(&mol, thresh, "D6 + θ·D6", 24, 12, false, GRGRP);
}

#[test]
fn test_ur_group_symmetric_c6ph6_grey_d6_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c6ph6.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_magnetic_group_class_order(
        &mol,
        thresh,
        &[
            "|E|",
            "2|C6|",
            "2|C3|",
            "|C2|",
            "3|C2|^(')",
            "3|C2|^('')",
            "|θ|",
            "2|θ·C6|",
            "2|θ·C3|",
            "|θ·C2|",
            "3|θ·C2|^(')",
            "3|θ·C2|^('')",
        ],
    );
}

#[test]
#[ignore]
fn test_ur_group_symmetric_arbitrary_twisted_sandwich_dn() {
    /* The expected number of classes is deduced from the irrep structures of
     * the Dn groups.
     * When n is even, the irreps are A1, A2, B1, B2, Ek where k = 1, ..., n/2 - 1.
     * When n is odd, the irreps are A1, A2, Ek where k = 1, ..., n//2.
     */
    let thresh = 1e-7;
    for n in 3..=20 {
        let mol = template_molecules::gen_arbitrary_twisted_sandwich(n, 0.1);
        test_ur_ordinary_group(
            &mol,
            thresh,
            format!("D{n}").as_str(),
            2 * n as usize,
            ({
                if n % 2 == 0 {
                    n / 2 - 1
                } else {
                    n / 2
                }
            } + {
                if n % 2 == 0 {
                    4
                } else {
                    2
                }
            }) as usize,
            false,
        );
    }
}

#[test]
#[ignore]
fn test_ur_group_symmetric_arbitrary_twisted_sandwich_grey_dn() {
    /* The expected number of classes is deduced from the irrep structures of
     * the Dn groups.
     * When n is even, the irreps are A1, A2, B1, B2, Ek where k = 1, ..., n/2 - 1.
     * When n is odd, the irreps are A1, A2, Ek where k = 1, ..., n//2.
     */
    let thresh = 1e-7;
    for n in 3..=20 {
        let mol = template_molecules::gen_arbitrary_twisted_sandwich(n, 0.1);
        test_ur_magnetic_group(
            &mol,
            thresh,
            format!("D{n} + θ·D{n}").as_str(),
            4 * n as usize,
            2 * ({
                if n % 2 == 0 {
                    n / 2 - 1
                } else {
                    n / 2
                }
            } + {
                if n % 2 == 0 {
                    4
                } else {
                    2
                }
            }) as usize,
            false,
            GRGRP,
        );
    }
}

/***
Dn*
***/
#[test]
fn test_ur_group_symmetric_triphenyl_radical_d3_double() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/triphenylradical.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_double_group(&mol, thresh, "D3*", 12, 6, false);
}

#[test]
fn test_ur_group_symmetric_triphenyl_radical_d3_double_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/triphenylradical.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_double_group_class_order(
        &mol,
        thresh,
        &[
            "|E(Σ)|",
            "|E(QΣ)|",
            "2|C3(Σ)|",
            "2|C3(QΣ)|",
            "3|C2(Σ)|",
            "3|C2(QΣ)|",
        ],
    );
}

#[test]
fn test_ur_group_symmetric_triphenyl_radical_grey_d3_double() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/triphenylradical.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_magnetic_double_group(&mol, thresh, "(D3 + θ·D3)*", 24, 12, false, GRGRP2);
}

#[test]
fn test_ur_group_symmetric_triphenyl_radical_grey_d3_double_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/triphenylradical.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_magnetic_double_group_class_order(
        &mol,
        thresh,
        &[
            "|E(Σ)|",
            "|E(QΣ)|",
            "2|C3(Σ)|",
            "2|C3(QΣ)|",
            "3|C2(Σ)|",
            "3|C2(QΣ)|",
            "|θ(Σ)|",
            "|θ(QΣ)|",
            "2|θ·C3(Σ)|",
            "2|θ·C3(QΣ)|",
            "3|θ·C2(Σ)|",
            "3|θ·C2(QΣ)|",
        ],
    );
}

#[test]
fn test_ur_group_symmetric_h8_twisted_d4_double() {
    let thresh = 1e-7;
    let mol = template_molecules::gen_twisted_h8(0.1);
    test_ur_ordinary_double_group(&mol, thresh, "D4*", 16, 7, false);
}

#[test]
fn test_ur_group_symmetric_h8_twisted_d4_double_class_order() {
    let thresh = 1e-7;
    let mol = template_molecules::gen_twisted_h8(0.1);
    test_ur_ordinary_double_group_class_order(
        &mol,
        thresh,
        &[
            "|E(Σ)|",
            "|E(QΣ)|",
            "2|C4(Σ)|",
            "2|C4(QΣ)|",
            "|C2(Σ), C2(QΣ)|",
            "2|C2(Σ), C2(QΣ)|^(')",
            "2|C2(Σ), C2(QΣ)|^('')",
        ],
    );
}

#[test]
fn test_ur_group_symmetric_h8_twisted_grey_d4_double() {
    let thresh = 1e-7;
    let mol = template_molecules::gen_twisted_h8(0.1);
    test_ur_magnetic_double_group(&mol, thresh, "(D4 + θ·D4)*", 32, 14, false, GRGRP2);
}

#[test]
fn test_ur_group_symmetric_h8_twisted_grey_d4_double_class_order() {
    let thresh = 1e-7;
    let mol = template_molecules::gen_twisted_h8(0.1);
    test_ur_magnetic_double_group_class_order(
        &mol,
        thresh,
        &[
            "|E(Σ)|",
            "|E(QΣ)|",
            "2|C4(Σ)|",
            "2|C4(QΣ)|",
            "|C2(Σ), C2(QΣ)|",
            "2|C2(Σ), C2(QΣ)|^(')",
            "2|C2(Σ), C2(QΣ)|^('')",
            "|θ(Σ)|",
            "|θ(QΣ)|",
            "2|θ·C4(Σ)|",
            "2|θ·C4(QΣ)|",
            "|θ·C2(Σ), θ·C2(QΣ)|",
            "2|θ·C2(Σ), θ·C2(QΣ)|^(')",
            "2|θ·C2(Σ), θ·C2(QΣ)|^('')",
        ],
    );
}

#[test]
fn test_ur_group_symmetric_c5ph5_d5_double() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c5ph5.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_double_group(&mol, thresh, "D5*", 20, 8, false);
}

#[test]
fn test_ur_group_symmetric_c5ph5_d5_double_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c5ph5.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_double_group_class_order(
        &mol,
        thresh,
        &[
            "|E(Σ)|",
            "|E(QΣ)|",
            "2|C5(Σ)|",
            "2|C5(QΣ)|",
            "2|C5^2(Σ)|",
            "2|C5^2(QΣ)|",
            "5|C2(Σ)|",
            "5|C2(QΣ)|",
        ],
    );
}

#[test]
fn test_ur_group_symmetric_c5ph5_grey_d5_double() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c5ph5.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_magnetic_double_group(&mol, thresh, "(D5 + θ·D5)*", 40, 16, false, GRGRP2);
}

#[test]
fn test_ur_group_symmetric_c5ph5_grey_d5_double_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c5ph5.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_magnetic_double_group_class_order(
        &mol,
        thresh,
        &[
            "|E(Σ)|",
            "|E(QΣ)|",
            "2|C5(Σ)|",
            "2|C5(QΣ)|",
            "2|C5^2(Σ)|",
            "2|C5^2(QΣ)|",
            "5|C2(Σ)|",
            "5|C2(QΣ)|",
            "|θ(Σ)|",
            "|θ(QΣ)|",
            "2|θ·C5(Σ)|",
            "2|θ·C5(QΣ)|",
            "2|θ·C5^2(Σ)|",
            "2|θ·C5^2(QΣ)|",
            "5|θ·C2(Σ)|",
            "5|θ·C2(QΣ)|",
        ],
    );
}

#[test]
fn test_ur_group_symmetric_c6ph6_d6_double() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c6ph6.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_double_group(&mol, thresh, "D6*", 24, 9, false);
}

#[test]
fn test_ur_group_symmetric_c6ph6_d6_double_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c6ph6.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_double_group_class_order(
        &mol,
        thresh,
        &[
            "|E(Σ)|",
            "|E(QΣ)|",
            "2|C6(Σ)|",
            "2|C6(QΣ)|",
            "2|C3(Σ)|",
            "2|C3(QΣ)|",
            "|C2(Σ), C2(QΣ)|",
            "3|C2(Σ), C2(QΣ)|^(')",
            "3|C2(Σ), C2(QΣ)|^('')",
        ],
    );
}

#[test]
fn test_ur_group_symmetric_c6ph6_grey_d6_double() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c6ph6.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_magnetic_double_group(&mol, thresh, "(D6 + θ·D6)*", 48, 18, false, GRGRP2);
}

#[test]
fn test_ur_group_symmetric_c6ph6_grey_d6_double_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c6ph6.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_magnetic_double_group_class_order(
        &mol,
        thresh,
        &[
            "|E(Σ)|",
            "|E(QΣ)|",
            "2|C6(Σ)|",
            "2|C6(QΣ)|",
            "2|C3(Σ)|",
            "2|C3(QΣ)|",
            "|C2(Σ), C2(QΣ)|",
            "3|C2(Σ), C2(QΣ)|^(')",
            "3|C2(Σ), C2(QΣ)|^('')",
            "|θ(Σ)|",
            "|θ(QΣ)|",
            "2|θ·C6(Σ)|",
            "2|θ·C6(QΣ)|",
            "2|θ·C3(Σ)|",
            "2|θ·C3(QΣ)|",
            "|θ·C2(Σ), θ·C2(QΣ)|",
            "3|θ·C2(Σ), θ·C2(QΣ)|^(')",
            "3|θ·C2(Σ), θ·C2(QΣ)|^('')",
        ],
    );
}

/*
Dnh
*/

#[test]
fn test_ur_group_symmetric_bf3_d3h() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/bf3.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_group(&mol, thresh, "D3h", 12, 6, false);
}

#[test]
fn test_ur_group_symmetric_bf3_d3h_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/bf3.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_group_class_order(
        &mol,
        thresh,
        &["|E|", "2|C3|", "3|C2|", "2|S3|", "|σh|", "3|σv|"],
    );
}

#[test]
fn test_ur_group_symmetric_bf3_grey_d3h() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/bf3.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_magnetic_group(&mol, thresh, "D3h + θ·D3h", 24, 12, false, GRGRP);
}

#[test]
fn test_ur_group_symmetric_bf3_grey_d3h_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/bf3.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_magnetic_group_class_order(
        &mol,
        thresh,
        &[
            "|E|",
            "2|C3|",
            "3|C2|",
            "2|S3|",
            "|σh|",
            "3|σv|",
            "|θ|",
            "2|θ·C3|",
            "3|θ·C2|",
            "2|θ·S3|",
            "|θ·σh|",
            "3|θ·σv|",
        ],
    );
}

#[test]
fn test_ur_group_symmetric_xef4_d4h() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/xef4.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_group(&mol, thresh, "D4h", 16, 10, false);
}

#[test]
fn test_ur_group_symmetric_xef4_d4h_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/xef4.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_group_class_order(
        &mol,
        thresh,
        &[
            "|E|",
            "2|C4|",
            "|C2|",
            "2|C2|^(')",
            "2|C2|^('')",
            "|i|",
            "2|S4|",
            "|σh|",
            "2|σv|",
            "2|σv|^(')",
        ],
    );
}

#[test]
fn test_ur_group_symmetric_xef4_grey_d4h() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/xef4.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_magnetic_group(&mol, thresh, "D4h + θ·D4h", 32, 20, false, GRGRP);
}

#[test]
fn test_ur_group_symmetric_xef4_grey_d4h_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/xef4.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_magnetic_group_class_order(
        &mol,
        thresh,
        &[
            "|E|",
            "2|C4|",
            "|C2|",
            "2|C2|^(')",
            "2|C2|^('')",
            "|i|",
            "2|S4|",
            "|σh|",
            "2|σv|",
            "2|σv|^(')",
            "|θ|",
            "2|θ·C4|",
            "|θ·C2|",
            "2|θ·C2|^(')",
            "2|θ·C2|^('')",
            "|θ·i|",
            "2|θ·S4|",
            "|θ·σh|",
            "2|θ·σv|",
            "2|θ·σv|^(')",
        ],
    );
}

#[test]
fn test_ur_group_symmetric_h8_d4h() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/h8.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_group(&mol, thresh, "D4h", 16, 10, false);
}

#[test]
fn test_ur_group_symmetric_h8_d4h_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/h8.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_group_class_order(
        &mol,
        thresh,
        &[
            "|E|",
            "2|C4|",
            "|C2|",
            "2|C2|^(')",
            "2|C2|^('')",
            "|i|",
            "2|S4|",
            "|σh|",
            "2|σv|",
            "2|σv|^(')",
        ],
    );
}

#[test]
fn test_ur_group_symmetric_h8_grey_d4h() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/h8.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_magnetic_group(&mol, thresh, "D4h + θ·D4h", 32, 20, false, GRGRP);
}

#[test]
fn test_ur_group_symmetric_h8_grey_d4h_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/h8.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_magnetic_group_class_order(
        &mol,
        thresh,
        &[
            "|E|",
            "2|C4|",
            "|C2|",
            "2|C2|^(')",
            "2|C2|^('')",
            "|i|",
            "2|S4|",
            "|σh|",
            "2|σv|",
            "2|σv|^(')",
            "|θ|",
            "2|θ·C4|",
            "|θ·C2|",
            "2|θ·C2|^(')",
            "2|θ·C2|^('')",
            "|θ·i|",
            "2|θ·S4|",
            "|θ·σh|",
            "2|θ·σv|",
            "2|θ·σv|^(')",
        ],
    );
}

#[test]
fn test_ur_group_symmetric_eclipsed_ferrocene_d5h() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/eclipsed_ferrocene.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_group(&mol, thresh, "D5h", 20, 8, false);
}

#[test]
fn test_ur_group_symmetric_eclipsed_ferrocene_d5h_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/eclipsed_ferrocene.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_group_class_order(
        &mol,
        thresh,
        &[
            "|E|",
            "2|C5|",
            "2|C5^2|",
            "5|C2|",
            "2|S5|",
            "2|σC5^2|",
            "|σh|",
            "5|σv|",
        ],
    );
}

#[test]
fn test_ur_group_symmetric_eclipsed_ferrocene_grey_d5h() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/eclipsed_ferrocene.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_magnetic_group(&mol, thresh, "D5h + θ·D5h", 40, 16, false, GRGRP);
}

#[test]
fn test_ur_group_symmetric_eclipsed_ferrocene_grey_d5h_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/eclipsed_ferrocene.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_magnetic_group_class_order(
        &mol,
        thresh,
        &[
            "|E|",
            "2|C5|",
            "2|C5^2|",
            "5|C2|",
            "2|S5|",
            "2|σC5^2|",
            "|σh|",
            "5|σv|",
            "|θ|",
            "2|θ·C5|",
            "2|θ·C5^2|",
            "5|θ·C2|",
            "2|θ·S5|",
            "2|θ·σC5^2|",
            "|θ·σh|",
            "5|θ·σv|",
        ],
    );
}

#[test]
fn test_ur_group_symmetric_benzene_d6h() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/benzene.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_group(&mol, thresh, "D6h", 24, 12, false);
}

#[test]
fn test_ur_group_symmetric_benzene_d6h_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/benzene.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    // The benzene molecule is in the yz-plane. Ordering of the symmetry elements based on their
    // closeness to principal axes means that the class ordering will appear different from that
    // found in standard character tables.
    test_ur_ordinary_group_class_order(
        &mol,
        thresh,
        &[
            "|E|",
            "2|C6|",
            "2|C3|",
            "3|C2|",
            "3|C2|^(')",
            "|C2|^('')",
            "|i|",
            "2|S6|",
            "2|S3|",
            "3|σv|",
            "3|σv|^(')",
            "|σh|",
        ],
    );
}

#[test]
fn test_ur_group_symmetric_benzene_grey_d6h() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/benzene.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_magnetic_group(&mol, thresh, "D6h + θ·D6h", 48, 24, false, GRGRP);
}

#[test]
fn test_ur_group_symmetric_benzene_grey_d6h_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/benzene.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    // The benzene molecule is in the yz-plane. Ordering of the symmetry elements based on their
    // closeness to principal axes means that the class ordering will appear different from that
    // found in standard character tables.
    test_ur_magnetic_group_class_order(
        &mol,
        thresh,
        &[
            "|E|",
            "2|C6|",
            "2|C3|",
            "3|C2|",
            "3|C2|^(')",
            "|C2|^('')",
            "|i|",
            "2|S6|",
            "2|S3|",
            "3|σv|",
            "3|σv|^(')",
            "|σh|",
            "|θ|",
            "2|θ·C6|",
            "2|θ·C3|",
            "3|θ·C2|",
            "3|θ·C2|^(')",
            "|θ·C2|^('')",
            "|θ·i|",
            "2|θ·S6|",
            "2|θ·S3|",
            "3|θ·σv|",
            "3|θ·σv|^(')",
            "|θ·σh|",
        ],
    );
}

// #[test]
// fn test_ur_group_symmetric_h100_d100h() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/h100.xyz");
//     let thresh = 1e-6;
//     let mol = Molecule::from_xyz(&path, thresh);
//     test_ur_ordinary_group(&mol, thresh, "D100h", 400, 106, false);
// }

#[test]
#[ignore]
fn test_ur_group_symmetric_arbitrary_eclipsed_sandwich_dnh() {
    /* The expected number of classes is deduced from the irrep structures of
     * the Dnh groups.
     * When n is even, the irreps are A1(g/u), A2(g/u), B1(g/u), B2(g/u), Ek(g/u)
     * where k = 1, ..., n/2 - 1.
     * When n is odd, the irreps are A1('/''), A2('/''), Ek('/'')
     * where k = 1, ..., n//2.
     */
    let thresh = 1e-7;
    for n in 3..=20 {
        let mol = template_molecules::gen_arbitrary_eclipsed_sandwich(n);
        test_ur_ordinary_group(
            &mol,
            thresh,
            format!("D{n}h").as_str(),
            4 * n as usize,
            2 * ({
                if n % 2 == 0 {
                    n / 2 - 1
                } else {
                    n / 2
                }
            } + {
                if n % 2 == 0 {
                    4
                } else {
                    2
                }
            }) as usize,
            false,
        );
    }
}

#[test]
#[ignore]
fn test_ur_group_symmetric_arbitrary_eclipsed_sandwich_grey_dnh() {
    /* The expected number of classes is deduced from the irrep structures of
     * the Dnh groups.
     * When n is even, the irreps are A1(g/u), A2(g/u), B1(g/u), B2(g/u), Ek(g/u)
     * where k = 1, ..., n/2 - 1.
     * When n is odd, the irreps are A1('/''), A2('/''), Ek('/'')
     * where k = 1, ..., n//2.
     */
    let thresh = 1e-7;
    for n in 3..=20 {
        let mol = template_molecules::gen_arbitrary_eclipsed_sandwich(n);
        test_ur_magnetic_group(
            &mol,
            thresh,
            format!("D{n}h + θ·D{n}h").as_str(),
            8 * n as usize,
            4 * ({
                if n % 2 == 0 {
                    n / 2 - 1
                } else {
                    n / 2
                }
            } + {
                if n % 2 == 0 {
                    4
                } else {
                    2
                }
            }) as usize,
            false,
            GRGRP,
        );
    }
}

/*
Dnh*
*/

#[test]
fn test_ur_group_symmetric_bf3_d3h_double() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/bf3.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_double_group(&mol, thresh, "D3h*", 24, 9, false);
}

#[test]
fn test_ur_group_symmetric_bf3_d3h_double_class_order() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/bf3.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_double_group_class_order(
        &mol,
        thresh,
        &[
            "|E(Σ)|",
            "|E(QΣ)|",
            "2|C3(Σ)|",
            "2|C3(QΣ)|",
            "3|C2(Σ), C2(QΣ)|",
            "2|S3(Σ)|",
            "2|S3(QΣ)|",
            "|σh(Σ), σh(QΣ)|",
            "3|σv(Σ), σv(QΣ)|",
        ],
    );
}

#[test]
fn test_ur_group_symmetric_bf3_grey_d3h_double() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/bf3.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_magnetic_double_group(&mol, thresh, "(D3h + θ·D3h)*", 48, 18, false, GRGRP2);
}

#[test]
fn test_ur_group_symmetric_bf3_grey_d3h_double_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/bf3.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_magnetic_double_group_class_order(
        &mol,
        thresh,
        &[
            "|E(Σ)|",
            "|E(QΣ)|",
            "2|C3(Σ)|",
            "2|C3(QΣ)|",
            "3|C2(Σ), C2(QΣ)|",
            "2|S3(Σ)|",
            "2|S3(QΣ)|",
            "|σh(Σ), σh(QΣ)|",
            "3|σv(Σ), σv(QΣ)|",
            "|θ(Σ)|",
            "|θ(QΣ)|",
            "2|θ·C3(Σ)|",
            "2|θ·C3(QΣ)|",
            "3|θ·C2(Σ), θ·C2(QΣ)|",
            "2|θ·S3(Σ)|",
            "2|θ·S3(QΣ)|",
            "|θ·σh(Σ), θ·σh(QΣ)|",
            "3|θ·σv(Σ), θ·σv(QΣ)|",
        ],
    );
}

#[test]
fn test_ur_group_symmetric_h8_d4h_double() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/h8.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_double_group(&mol, thresh, "D4h*", 32, 14, false);
}

#[test]
fn test_ur_group_symmetric_h8_d4h_double_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/h8.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_double_group_class_order(
        &mol,
        thresh,
        &[
            "|E(Σ)|",
            "|E(QΣ)|",
            "2|C4(Σ)|",
            "2|C4(QΣ)|",
            "|C2(Σ), C2(QΣ)|",
            "2|C2(Σ), C2(QΣ)|^(')",
            "2|C2(Σ), C2(QΣ)|^('')",
            "|i(Σ)|",
            "|i(QΣ)|",
            "2|S4(Σ)|",
            "2|S4(QΣ)|",
            "|σh(Σ), σh(QΣ)|",
            "2|σv(Σ), σv(QΣ)|",
            "2|σv(Σ), σv(QΣ)|^(')",
        ],
    );
}

#[test]
fn test_ur_group_symmetric_h8_grey_d4h_double() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/h8.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_magnetic_double_group(&mol, thresh, "(D4h + θ·D4h)*", 64, 28, false, GRGRP2);
}

#[test]
fn test_ur_group_symmetric_h8_grey_d4h_double_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/h8.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_magnetic_double_group_class_order(
        &mol,
        thresh,
        &[
            "|E(Σ)|",
            "|E(QΣ)|",
            "2|C4(Σ)|",
            "2|C4(QΣ)|",
            "|C2(Σ), C2(QΣ)|",
            "2|C2(Σ), C2(QΣ)|^(')",
            "2|C2(Σ), C2(QΣ)|^('')",
            "|i(Σ)|",
            "|i(QΣ)|",
            "2|S4(Σ)|",
            "2|S4(QΣ)|",
            "|σh(Σ), σh(QΣ)|",
            "2|σv(Σ), σv(QΣ)|",
            "2|σv(Σ), σv(QΣ)|^(')",
            "|θ(Σ)|",
            "|θ(QΣ)|",
            "2|θ·C4(Σ)|",
            "2|θ·C4(QΣ)|",
            "|θ·C2(Σ), θ·C2(QΣ)|",
            "2|θ·C2(Σ), θ·C2(QΣ)|^(')",
            "2|θ·C2(Σ), θ·C2(QΣ)|^('')",
            "|θ·i(Σ)|",
            "|θ·i(QΣ)|",
            "2|θ·S4(Σ)|",
            "2|θ·S4(QΣ)|",
            "|θ·σh(Σ), θ·σh(QΣ)|",
            "2|θ·σv(Σ), θ·σv(QΣ)|",
            "2|θ·σv(Σ), θ·σv(QΣ)|^(')",
        ],
    );
}

#[test]
fn test_ur_group_symmetric_eclipsed_ferrocene_d5h_double() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/eclipsed_ferrocene.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_double_group(&mol, thresh, "D5h*", 40, 13, false);
}

#[test]
fn test_ur_group_symmetric_eclipsed_ferrocene_d5h_double_class_order() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/eclipsed_ferrocene.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_double_group_class_order(
        &mol,
        thresh,
        &[
            "|E(Σ)|",
            "|E(QΣ)|",
            "2|C5(Σ)|",
            "2|C5(QΣ)|",
            "2|C5^2(Σ)|",
            "2|C5^2(QΣ)|",
            "5|C2(Σ), C2(QΣ)|",
            "2|S5(Σ)|",
            "2|S5(QΣ)|",
            "2|σC5^2(Σ)|",
            "2|σC5^2(QΣ)|",
            "|σh(Σ), σh(QΣ)|",
            "5|σv(Σ), σv(QΣ)|",
        ],
    );
}

#[test]
fn test_ur_group_symmetric_eclipsed_ferrocene_grey_d5h_double() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/eclipsed_ferrocene.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_magnetic_double_group(&mol, thresh, "(D5h + θ·D5h)*", 80, 26, false, GRGRP2);
}

#[test]
fn test_ur_group_symmetric_eclipsed_ferrocene_grey_d5h_double_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/eclipsed_ferrocene.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_magnetic_double_group_class_order(
        &mol,
        thresh,
        &[
            "|E(Σ)|",
            "|E(QΣ)|",
            "2|C5(Σ)|",
            "2|C5(QΣ)|",
            "2|C5^2(Σ)|",
            "2|C5^2(QΣ)|",
            "5|C2(Σ), C2(QΣ)|",
            "2|S5(Σ)|",
            "2|S5(QΣ)|",
            "2|σC5^2(Σ)|",
            "2|σC5^2(QΣ)|",
            "|σh(Σ), σh(QΣ)|",
            "5|σv(Σ), σv(QΣ)|",
            "|θ(Σ)|",
            "|θ(QΣ)|",
            "2|θ·C5(Σ)|",
            "2|θ·C5(QΣ)|",
            "2|θ·C5^2(Σ)|",
            "2|θ·C5^2(QΣ)|",
            "5|θ·C2(Σ), θ·C2(QΣ)|",
            "2|θ·S5(Σ)|",
            "2|θ·S5(QΣ)|",
            "2|θ·σC5^2(Σ)|",
            "2|θ·σC5^2(QΣ)|",
            "|θ·σh(Σ), θ·σh(QΣ)|",
            "5|θ·σv(Σ), θ·σv(QΣ)|",
        ],
    );
}

#[test]
fn test_ur_group_symmetric_benzene_d6h_double() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/benzene.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_double_group(&mol, thresh, "D6h*", 48, 18, false);
}

#[test]
fn test_ur_group_symmetric_benzene_d6h_double_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/benzene.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    // The benzene molecule is in the yz-plane. Ordering of the symmetry elements based on their
    // closeness to principal axes means that the class ordering will appear different from that
    // found in standard character tables.
    test_ur_ordinary_double_group_class_order(
        &mol,
        thresh,
        &[
            "|E(Σ)|",
            "|E(QΣ)|",
            "2|C6(Σ)|",
            "2|C6(QΣ)|",
            "2|C3(Σ)|",
            "2|C3(QΣ)|",
            "3|C2(Σ), C2(QΣ)|",
            "3|C2(Σ), C2(QΣ)|^(')",
            "|C2(Σ), C2(QΣ)|^('')",
            "|i(Σ)|",
            "|i(QΣ)|",
            "2|S6(Σ)|",
            "2|S6(QΣ)|",
            "2|S3(Σ)|",
            "2|S3(QΣ)|",
            "3|σv(Σ), σv(QΣ)|",
            "3|σv(Σ), σv(QΣ)|^(')",
            "|σh(Σ), σh(QΣ)|",
        ],
    );
}

#[test]
fn test_ur_group_symmetric_benzene_grey_d6h_double() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/benzene.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_magnetic_double_group(&mol, thresh, "(D6h + θ·D6h)*", 96, 36, false, GRGRP2);
}

#[test]
fn test_ur_group_symmetric_benzene_grey_d6h_double_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/benzene.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    // The benzene molecule is in the yz-plane. Ordering of the symmetry elements based on their
    // closeness to principal axes means that the class ordering will appear different from that
    // found in standard character tables.
    test_ur_magnetic_double_group_class_order(
        &mol,
        thresh,
        &[
            "|E(Σ)|",
            "|E(QΣ)|",
            "2|C6(Σ)|",
            "2|C6(QΣ)|",
            "2|C3(Σ)|",
            "2|C3(QΣ)|",
            "3|C2(Σ), C2(QΣ)|",
            "3|C2(Σ), C2(QΣ)|^(')",
            "|C2(Σ), C2(QΣ)|^('')",
            "|i(Σ)|",
            "|i(QΣ)|",
            "2|S6(Σ)|",
            "2|S6(QΣ)|",
            "2|S3(Σ)|",
            "2|S3(QΣ)|",
            "3|σv(Σ), σv(QΣ)|",
            "3|σv(Σ), σv(QΣ)|^(')",
            "|σh(Σ), σh(QΣ)|",
            "|θ(Σ)|",
            "|θ(QΣ)|",
            "2|θ·C6(Σ)|",
            "2|θ·C6(QΣ)|",
            "2|θ·C3(Σ)|",
            "2|θ·C3(QΣ)|",
            "3|θ·C2(Σ), θ·C2(QΣ)|",
            "3|θ·C2(Σ), θ·C2(QΣ)|^(')",
            "|θ·C2(Σ), θ·C2(QΣ)|^('')",
            "|θ·i(Σ)|",
            "|θ·i(QΣ)|",
            "2|θ·S6(Σ)|",
            "2|θ·S6(QΣ)|",
            "2|θ·S3(Σ)|",
            "2|θ·S3(QΣ)|",
            "3|θ·σv(Σ), θ·σv(QΣ)|",
            "3|θ·σv(Σ), θ·σv(QΣ)|^(')",
            "|θ·σh(Σ), θ·σh(QΣ)|",
        ],
    );
}

/*
Dnd
*/

#[test]
fn test_ur_group_symmetric_b2cl4_d2d() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/b2cl4.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_group(&mol, thresh, "D2d", 8, 5, false);
}

#[test]
fn test_ur_group_symmetric_b2cl4_d2d_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/b2cl4.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_group_class_order(
        &mol,
        thresh,
        &["|E|", "|C2|", "2|C2|^(')", "2|S4|", "2|σd|"],
    );
}

#[test]
fn test_ur_group_symmetric_b2cl4_grey_d2d() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/b2cl4.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_magnetic_group(&mol, thresh, "D2d + θ·D2d", 16, 10, false, GRGRP);
}

#[test]
fn test_ur_group_symmetric_b2cl4_grey_d2d_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/b2cl4.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_magnetic_group_class_order(
        &mol,
        thresh,
        &[
            "|E|",
            "|C2|",
            "2|C2|^(')",
            "2|S4|",
            "2|σd|",
            "|θ|",
            "|θ·C2|",
            "2|θ·C2|^(')",
            "2|θ·S4|",
            "2|θ·σd|",
        ],
    );
}

#[test]
fn test_ur_group_symmetric_s4n4_d2d() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/s4n4.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_group(&mol, thresh, "D2d", 8, 5, false);
}

#[test]
fn test_ur_group_symmetric_s4n4_d2d_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/s4n4.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_group_class_order(
        &mol,
        thresh,
        &["|E|", "|C2|", "2|C2|^(')", "2|S4|", "2|σd|"],
    );
}

#[test]
fn test_ur_group_symmetric_s4n4_grey_d2d() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/s4n4.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_magnetic_group(&mol, thresh, "D2d + θ·D2d", 16, 10, false, GRGRP);
}

#[test]
fn test_ur_group_symmetric_s4n4_grey_d2d_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/s4n4.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_magnetic_group_class_order(
        &mol,
        thresh,
        &[
            "|E|",
            "|C2|",
            "2|C2|^(')",
            "2|S4|",
            "2|σd|",
            "|θ|",
            "|θ·C2|",
            "2|θ·C2|^(')",
            "2|θ·S4|",
            "2|θ·σd|",
        ],
    );
}

#[test]
fn test_ur_group_symmetric_pbet4_d2d() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/pbet4.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_group(&mol, thresh, "D2d", 8, 5, false);
}

#[test]
fn test_ur_group_symmetric_pbet4_d2d_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/pbet4.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_group_class_order(
        &mol,
        thresh,
        &["|E|", "|C2|", "2|C2|^(')", "2|S4|", "2|σd|"],
    );
}

#[test]
fn test_ur_group_symmetric_pbet4_grey_d2d() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/pbet4.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_magnetic_group(&mol, thresh, "D2d + θ·D2d", 16, 10, false, GRGRP);
}

#[test]
fn test_ur_group_symmetric_pbet4_grey_d2d_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/pbet4.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_magnetic_group_class_order(
        &mol,
        thresh,
        &[
            "|E|",
            "|C2|",
            "2|C2|^(')",
            "2|S4|",
            "2|σd|",
            "|θ|",
            "|θ·C2|",
            "2|θ·C2|^(')",
            "2|θ·S4|",
            "2|θ·σd|",
        ],
    );
}

#[test]
fn test_ur_group_symmetric_allene_d2d() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/allene.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_group(&mol, thresh, "D2d", 8, 5, false);
}

#[test]
fn test_ur_group_symmetric_allene_d2d_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/allene.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_group_class_order(
        &mol,
        thresh,
        &["|E|", "|C2|", "2|C2|^(')", "2|S4|", "2|σd|"],
    );
}

#[test]
fn test_ur_group_symmetric_allene_grey_d2d() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/allene.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_magnetic_group(&mol, thresh, "D2d + θ·D2d", 16, 10, false, GRGRP);
}

#[test]
fn test_ur_group_symmetric_allene_grey_d2d_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/allene.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_magnetic_group_class_order(
        &mol,
        thresh,
        &[
            "|E|",
            "|C2|",
            "2|C2|^(')",
            "2|S4|",
            "2|σd|",
            "|θ|",
            "|θ·C2|",
            "2|θ·C2|^(')",
            "2|θ·S4|",
            "2|θ·σd|",
        ],
    );
}

#[test]
fn test_ur_group_symmetric_staggered_c2h6_d3d() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c2h6.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_group(&mol, thresh, "D3d", 12, 6, false);
}

#[test]
fn test_ur_group_symmetric_staggered_c2h6_d3d_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c2h6.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_group_class_order(
        &mol,
        thresh,
        &["|E|", "2|C3|", "3|C2|", "|i|", "2|S6|", "3|σd|"],
    );
}

#[test]
fn test_ur_group_symmetric_staggered_c2h6_grey_d3d() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c2h6.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_magnetic_group(&mol, thresh, "D3d + θ·D3d", 24, 12, false, GRGRP);
}

#[test]
fn test_ur_group_symmetric_staggered_c2h6_grey_d3d_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c2h6.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_magnetic_group_class_order(
        &mol,
        thresh,
        &[
            "|E|",
            "2|C3|",
            "3|C2|",
            "|i|",
            "2|S6|",
            "3|σd|",
            "|θ|",
            "2|θ·C3|",
            "3|θ·C2|",
            "|θ·i|",
            "2|θ·S6|",
            "3|θ·σd|",
        ],
    );
}

#[test]
fn test_ur_group_symmetric_cyclohexane_chair_d3d() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/cyclohexane_chair.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_group(&mol, thresh, "D3d", 12, 6, false);
}

#[test]
fn test_ur_group_symmetric_cyclohexane_chair_d3d_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/cyclohexane_chair.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_group_class_order(
        &mol,
        thresh,
        &["|E|", "2|C3|", "3|C2|", "|i|", "2|S6|", "3|σd|"],
    );
}

#[test]
fn test_ur_group_symmetric_cyclohexane_chair_grey_d3d() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/cyclohexane_chair.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_magnetic_group(&mol, thresh, "D3d + θ·D3d", 24, 12, false, GRGRP);
}

#[test]
fn test_ur_group_symmetric_cyclohexane_chair_grey_d3d_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/cyclohexane_chair.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_magnetic_group_class_order(
        &mol,
        thresh,
        &[
            "|E|",
            "2|C3|",
            "3|C2|",
            "|i|",
            "2|S6|",
            "3|σd|",
            "|θ|",
            "2|θ·C3|",
            "3|θ·C2|",
            "|θ·i|",
            "2|θ·S6|",
            "3|θ·σd|",
        ],
    );
}

#[test]
fn test_ur_group_symmetric_s8_d4d() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/s8.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_group(&mol, thresh, "D4d", 16, 7, false);
}

#[test]
fn test_ur_group_symmetric_s8_d4d_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/s8.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_group_class_order(
        &mol,
        thresh,
        &[
            "|E|",
            "2|C4|",
            "|C2|",
            "4|C2|^(')",
            "2|S8|",
            "2|σC8^3|",
            "4|σd|",
        ],
    );
}

#[test]
fn test_ur_group_symmetric_s8_grey_d4d() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/s8.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_magnetic_group(&mol, thresh, "D4d + θ·D4d", 32, 14, false, GRGRP);
}

#[test]
fn test_ur_group_symmetric_s8_grey_d4d_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/s8.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_magnetic_group_class_order(
        &mol,
        thresh,
        &[
            "|E|",
            "2|C4|",
            "|C2|",
            "4|C2|^(')",
            "2|S8|",
            "2|σC8^3|",
            "4|σd|",
            "|θ|",
            "2|θ·C4|",
            "|θ·C2|",
            "4|θ·C2|^(')",
            "2|θ·S8|",
            "2|θ·σC8^3|",
            "4|θ·σd|",
        ],
    );
}

#[test]
fn test_ur_group_symmetric_antiprism_h8_d4d() {
    let mol = template_molecules::gen_twisted_h8(std::f64::consts::FRAC_PI_4);
    let thresh = 1e-7;
    test_ur_ordinary_group(&mol, thresh, "D4d", 16, 7, false);
}

#[test]
fn test_ur_group_symmetric_antiprism_h8_d4d_class_order() {
    let mol = template_molecules::gen_twisted_h8(std::f64::consts::FRAC_PI_4);
    let thresh = 1e-7;
    test_ur_ordinary_group_class_order(
        &mol,
        thresh,
        &[
            "|E|",
            "2|C4|",
            "|C2|",
            "4|C2|^(')",
            "2|S8|",
            "2|σC8^3|",
            "4|σd|",
        ],
    );
}

#[test]
fn test_ur_group_symmetric_antiprism_h8_grey_d4d() {
    let mol = template_molecules::gen_twisted_h8(std::f64::consts::FRAC_PI_4);
    let thresh = 1e-7;
    test_ur_magnetic_group(&mol, thresh, "D4d + θ·D4d", 32, 14, false, GRGRP);
}

#[test]
fn test_ur_group_symmetric_antiprism_h8_grey_d4d_class_order() {
    let mol = template_molecules::gen_twisted_h8(std::f64::consts::FRAC_PI_4);
    let thresh = 1e-7;
    test_ur_magnetic_group_class_order(
        &mol,
        thresh,
        &[
            "|E|",
            "2|C4|",
            "|C2|",
            "4|C2|^(')",
            "2|S8|",
            "2|σC8^3|",
            "4|σd|",
            "|θ|",
            "2|θ·C4|",
            "|θ·C2|",
            "4|θ·C2|^(')",
            "2|θ·S8|",
            "2|θ·σC8^3|",
            "4|θ·σd|",
        ],
    );
}

#[test]
fn test_ur_group_symmetric_antiprism_pb10_d4d() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/pb10.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_group(&mol, thresh, "D4d", 16, 7, false);
}

#[test]
fn test_ur_group_symmetric_antiprism_pb10_d4d_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/pb10.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_group_class_order(
        &mol,
        thresh,
        &[
            "|E|",
            "2|C4|",
            "|C2|",
            "4|C2|^(')",
            "2|S8|",
            "2|σC8^3|",
            "4|σd|",
        ],
    );
}

#[test]
fn test_ur_group_symmetric_antiprism_pb10_grey_d4d() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/pb10.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_magnetic_group(&mol, thresh, "D4d + θ·D4d", 32, 14, false, GRGRP);
}

#[test]
fn test_ur_group_symmetric_antiprism_pb10_grey_d4d_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/pb10.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_magnetic_group_class_order(
        &mol,
        thresh,
        &[
            "|E|",
            "2|C4|",
            "|C2|",
            "4|C2|^(')",
            "2|S8|",
            "2|σC8^3|",
            "4|σd|",
            "|θ|",
            "2|θ·C4|",
            "|θ·C2|",
            "4|θ·C2|^(')",
            "2|θ·S8|",
            "2|θ·σC8^3|",
            "4|θ·σd|",
        ],
    );
}

#[test]
fn test_ur_group_symmetric_staggered_ferrocene_d5d() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/staggered_ferrocene.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_group(&mol, thresh, "D5d", 20, 8, false);
}

#[test]
fn test_ur_group_symmetric_staggered_ferrocene_d5d_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/staggered_ferrocene.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_group_class_order(
        &mol,
        thresh,
        &[
            "|E|",
            "2|C5|",
            "2|C5^2|",
            "5|C2|",
            "|i|",
            "2|S10|",
            "2|σC10^3|",
            "5|σd|",
        ],
    );
}

#[test]
fn test_ur_group_symmetric_staggered_ferrocene_grey_d5d() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/staggered_ferrocene.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_magnetic_group(&mol, thresh, "D5d + θ·D5d", 40, 16, false, GRGRP);
}

#[test]
fn test_ur_group_symmetric_staggered_ferrocene_grey_d5d_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/staggered_ferrocene.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_magnetic_group_class_order(
        &mol,
        thresh,
        &[
            "|E|",
            "2|C5|",
            "2|C5^2|",
            "5|C2|",
            "|i|",
            "2|S10|",
            "2|σC10^3|",
            "5|σd|",
            "|θ|",
            "2|θ·C5|",
            "2|θ·C5^2|",
            "5|θ·C2|",
            "|θ·i|",
            "2|θ·S10|",
            "2|θ·σC10^3|",
            "5|θ·σd|",
        ],
    );
}

#[test]
fn test_ur_group_symmetric_au26_d6d() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/au26.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_group(&mol, thresh, "D6d", 24, 9, false);
}

#[test]
fn test_ur_group_symmetric_au26_d6d_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/au26.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_group_class_order(
        &mol,
        thresh,
        &[
            "|E|",
            "2|C6|",
            "2|C3|",
            "|C2|",
            "6|C2|^(')",
            "2|S12|",
            "2|σC12^5|",
            "2|S4|",
            "6|σd|",
        ],
    );
}

#[test]
fn test_ur_group_symmetric_au26_grey_d6d() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/au26.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_magnetic_group(&mol, thresh, "D6d + θ·D6d", 48, 18, false, GRGRP);
}

#[test]
fn test_ur_group_symmetric_au26_grey_d6d_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/au26.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_magnetic_group_class_order(
        &mol,
        thresh,
        &[
            "|E|",
            "2|C6|",
            "2|C3|",
            "|C2|",
            "6|C2|^(')",
            "2|S12|",
            "2|σC12^5|",
            "2|S4|",
            "6|σd|",
            "|θ|",
            "2|θ·C6|",
            "2|θ·C3|",
            "|θ·C2|",
            "6|θ·C2|^(')",
            "2|θ·S12|",
            "2|θ·σC12^5|",
            "2|θ·S4|",
            "6|θ·σd|",
        ],
    );
}

#[test]
#[ignore]
fn test_ur_group_symmetric_arbitrary_staggered_sandwich_dnd() {
    let thresh = 1e-7;
    for n in 3..=20 {
        let mol = template_molecules::gen_arbitrary_twisted_sandwich(n, 0.5);
        test_ur_ordinary_group(
            &mol,
            thresh,
            format!("D{n}d").as_str(),
            4 * n as usize,
            (4 + n - 1) as usize,
            false,
        );
    }
}

#[test]
#[ignore]
fn test_ur_group_symmetric_arbitrary_staggered_sandwich_grey_dnd() {
    let thresh = 1e-7;
    for n in 3..=20 {
        let mol = template_molecules::gen_arbitrary_twisted_sandwich(n, 0.5);
        test_ur_magnetic_group(
            &mol,
            thresh,
            format!("D{n}d + θ·D{n}d").as_str(),
            8 * n as usize,
            2 * (4 + n - 1) as usize,
            false,
            GRGRP,
        );
    }
}

/***
Dnd*
***/

#[test]
fn test_ur_group_symmetric_b2cl4_d2d_double() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/b2cl4.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_double_group(&mol, thresh, "D2d*", 16, 7, false);
}

#[test]
fn test_ur_group_symmetric_b2cl4_d2d_double_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/b2cl4.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_double_group_class_order(
        &mol,
        thresh,
        &[
            "|E(Σ)|",
            "|E(QΣ)|",
            "|C2(Σ), C2(QΣ)|",
            "2|C2(Σ), C2(QΣ)|^(')",
            "2|S4(Σ)|",
            "2|S4(QΣ)|",
            "2|σd(Σ), σd(QΣ)|",
        ],
    );
}

#[test]
fn test_ur_group_symmetric_allene_grey_d2d_double() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/allene.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_magnetic_double_group(&mol, thresh, "(D2d + θ·D2d)*", 32, 14, false, GRGRP2);
}

#[test]
fn test_ur_group_symmetric_allene_grey_d2d_double_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/allene.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_magnetic_double_group_class_order(
        &mol,
        thresh,
        &[
            "|E(Σ)|",
            "|E(QΣ)|",
            "|C2(Σ), C2(QΣ)|",
            "2|C2(Σ), C2(QΣ)|^(')",
            "2|S4(Σ)|",
            "2|S4(QΣ)|",
            "2|σd(Σ), σd(QΣ)|",
            "|θ(Σ)|",
            "|θ(QΣ)|",
            "|θ·C2(Σ), θ·C2(QΣ)|",
            "2|θ·C2(Σ), θ·C2(QΣ)|^(')",
            "2|θ·S4(Σ)|",
            "2|θ·S4(QΣ)|",
            "2|θ·σd(Σ), θ·σd(QΣ)|",
        ],
    );
}

#[test]
fn test_ur_group_symmetric_staggered_c2h6_d3d_double() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c2h6.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_double_group(&mol, thresh, "D3d*", 24, 12, false);
}

#[test]
fn test_ur_group_symmetric_staggered_c2h6_d3d_double_class_order() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c2h6.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_double_group_class_order(
        &mol,
        thresh,
        &[
            "|E(Σ)|",
            "|E(QΣ)|",
            "2|C3(Σ)|",
            "2|C3(QΣ)|",
            "3|C2(Σ)|",
            "3|C2(QΣ)|",
            "|i(Σ)|",
            "|i(QΣ)|",
            "2|S6(Σ)|",
            "2|S6(QΣ)|",
            "3|σd(Σ)|",
            "3|σd(QΣ)|",
        ],
    );
}

#[test]
fn test_ur_group_symmetric_cyclohexane_chair_grey_d3d_double() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/cyclohexane_chair.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_magnetic_double_group(&mol, thresh, "(D3d + θ·D3d)*", 48, 24, false, GRGRP2);
}

#[test]
fn test_ur_group_symmetric_cyclohexane_chair_grey_d3d_double_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/cyclohexane_chair.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_magnetic_double_group_class_order(
        &mol,
        thresh,
        &[
            "|E(Σ)|",
            "|E(QΣ)|",
            "2|C3(Σ)|",
            "2|C3(QΣ)|",
            "3|C2(Σ)|",
            "3|C2(QΣ)|",
            "|i(Σ)|",
            "|i(QΣ)|",
            "2|S6(Σ)|",
            "2|S6(QΣ)|",
            "3|σd(Σ)|",
            "3|σd(QΣ)|",
            "|θ(Σ)|",
            "|θ(QΣ)|",
            "2|θ·C3(Σ)|",
            "2|θ·C3(QΣ)|",
            "3|θ·C2(Σ)|",
            "3|θ·C2(QΣ)|",
            "|θ·i(Σ)|",
            "|θ·i(QΣ)|",
            "2|θ·S6(Σ)|",
            "2|θ·S6(QΣ)|",
            "3|θ·σd(Σ)|",
            "3|θ·σd(QΣ)|",
        ],
    );
}

#[test]
fn test_ur_group_symmetric_antiprism_pb10_d4d_double() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/pb10.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_double_group(&mol, thresh, "D4d*", 32, 11, false);
}

#[test]
fn test_ur_group_symmetric_antiprism_pb10_d4d_double_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/pb10.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_double_group_class_order(
        &mol,
        thresh,
        &[
            "|E(Σ)|",
            "|E(QΣ)|",
            "2|C4(Σ)|",
            "2|C4(QΣ)|",
            "|C2(Σ), C2(QΣ)|",
            "4|C2(Σ), C2(QΣ)|^(')",
            "2|S8(Σ)|",
            "2|S8(QΣ)|",
            "2|σC8^3(Σ)|",
            "2|σC8^3(QΣ)|",
            "4|σd(Σ), σd(QΣ)|",
        ],
    );
}

#[test]
fn test_ur_group_symmetric_antiprism_pb10_grey_d4d_double() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/pb10.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_magnetic_double_group(&mol, thresh, "(D4d + θ·D4d)*", 64, 22, false, GRGRP2);
}

#[test]
fn test_ur_group_symmetric_antiprism_pb10_grey_d4d_double_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/pb10.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_magnetic_double_group_class_order(
        &mol,
        thresh,
        &[
            "|E(Σ)|",
            "|E(QΣ)|",
            "2|C4(Σ)|",
            "2|C4(QΣ)|",
            "|C2(Σ), C2(QΣ)|",
            "4|C2(Σ), C2(QΣ)|^(')",
            "2|S8(Σ)|",
            "2|S8(QΣ)|",
            "2|σC8^3(Σ)|",
            "2|σC8^3(QΣ)|",
            "4|σd(Σ), σd(QΣ)|",
            "|θ(Σ)|",
            "|θ(QΣ)|",
            "2|θ·C4(Σ)|",
            "2|θ·C4(QΣ)|",
            "|θ·C2(Σ), θ·C2(QΣ)|",
            "4|θ·C2(Σ), θ·C2(QΣ)|^(')",
            "2|θ·S8(Σ)|",
            "2|θ·S8(QΣ)|",
            "2|θ·σC8^3(Σ)|",
            "2|θ·σC8^3(QΣ)|",
            "4|θ·σd(Σ), θ·σd(QΣ)|",
        ],
    );
}

#[test]
fn test_ur_group_symmetric_staggered_ferrocene_d5d_double() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/staggered_ferrocene.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_double_group(&mol, thresh, "D5d*", 40, 16, false);
}

#[test]
fn test_ur_group_symmetric_staggered_ferrocene_d5d_double_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/staggered_ferrocene.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_double_group_class_order(
        &mol,
        thresh,
        &[
            "|E(Σ)|",
            "|E(QΣ)|",
            "2|C5(Σ)|",
            "2|C5(QΣ)|",
            "2|C5^2(Σ)|",
            "2|C5^2(QΣ)|",
            "5|C2(Σ)|",
            "5|C2(QΣ)|",
            "|i(Σ)|",
            "|i(QΣ)|",
            "2|S10(Σ)|",
            "2|S10(QΣ)|",
            "2|σC10^3(Σ)|",
            "2|σC10^3(QΣ)|",
            "5|σd(Σ)|",
            "5|σd(QΣ)|",
        ],
    );
}

#[test]
fn test_ur_group_symmetric_staggered_ferrocene_grey_d5d_double() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/staggered_ferrocene.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_magnetic_double_group(&mol, thresh, "(D5d + θ·D5d)*", 80, 32, false, GRGRP2);
}

#[test]
fn test_ur_group_symmetric_staggered_ferrocene_grey_d5d_double_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/staggered_ferrocene.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_magnetic_double_group_class_order(
        &mol,
        thresh,
        &[
            "|E(Σ)|",
            "|E(QΣ)|",
            "2|C5(Σ)|",
            "2|C5(QΣ)|",
            "2|C5^2(Σ)|",
            "2|C5^2(QΣ)|",
            "5|C2(Σ)|",
            "5|C2(QΣ)|",
            "|i(Σ)|",
            "|i(QΣ)|",
            "2|S10(Σ)|",
            "2|S10(QΣ)|",
            "2|σC10^3(Σ)|",
            "2|σC10^3(QΣ)|",
            "5|σd(Σ)|",
            "5|σd(QΣ)|",
            "|θ(Σ)|",
            "|θ(QΣ)|",
            "2|θ·C5(Σ)|",
            "2|θ·C5(QΣ)|",
            "2|θ·C5^2(Σ)|",
            "2|θ·C5^2(QΣ)|",
            "5|θ·C2(Σ)|",
            "5|θ·C2(QΣ)|",
            "|θ·i(Σ)|",
            "|θ·i(QΣ)|",
            "2|θ·S10(Σ)|",
            "2|θ·S10(QΣ)|",
            "2|θ·σC10^3(Σ)|",
            "2|θ·σC10^3(QΣ)|",
            "5|θ·σd(Σ)|",
            "5|θ·σd(QΣ)|",
        ],
    );
}

#[test]
fn test_ur_group_symmetric_au26_d6d_double() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/au26.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_double_group(&mol, thresh, "D6d*", 48, 15, false);
}

#[test]
fn test_ur_group_symmetric_au26_d6d_double_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/au26.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_double_group_class_order(
        &mol,
        thresh,
        &[
            "|E(Σ)|",
            "|E(QΣ)|",
            "2|C6(Σ)|",
            "2|C6(QΣ)|",
            "2|C3(Σ)|",
            "2|C3(QΣ)|",
            "|C2(Σ), C2(QΣ)|",
            "6|C2(Σ), C2(QΣ)|^(')",
            "2|S12(Σ)|",
            "2|S12(QΣ)|",
            "2|σC12^5(Σ)|",
            "2|σC12^5(QΣ)|",
            "2|S4(Σ)|",
            "2|S4(QΣ)|",
            "6|σd(Σ), σd(QΣ)|",
        ],
    );
}

#[test]
fn test_ur_group_symmetric_au26_grey_d6d_double() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/au26.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_magnetic_double_group(&mol, thresh, "(D6d + θ·D6d)*", 96, 30, false, GRGRP2);
}

#[test]
fn test_ur_group_symmetric_au26_grey_d6d_double_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/au26.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_magnetic_double_group_class_order(
        &mol,
        thresh,
        &[
            "|E(Σ)|",
            "|E(QΣ)|",
            "2|C6(Σ)|",
            "2|C6(QΣ)|",
            "2|C3(Σ)|",
            "2|C3(QΣ)|",
            "|C2(Σ), C2(QΣ)|",
            "6|C2(Σ), C2(QΣ)|^(')",
            "2|S12(Σ)|",
            "2|S12(QΣ)|",
            "2|σC12^5(Σ)|",
            "2|σC12^5(QΣ)|",
            "2|S4(Σ)|",
            "2|S4(QΣ)|",
            "6|σd(Σ), σd(QΣ)|",
            "|θ(Σ)|",
            "|θ(QΣ)|",
            "2|θ·C6(Σ)|",
            "2|θ·C6(QΣ)|",
            "2|θ·C3(Σ)|",
            "2|θ·C3(QΣ)|",
            "|θ·C2(Σ), θ·C2(QΣ)|",
            "6|θ·C2(Σ), θ·C2(QΣ)|^(')",
            "2|θ·S12(Σ)|",
            "2|θ·S12(QΣ)|",
            "2|θ·σC12^5(Σ)|",
            "2|θ·σC12^5(QΣ)|",
            "2|θ·S4(Σ)|",
            "2|θ·S4(QΣ)|",
            "6|θ·σd(Σ), θ·σd(QΣ)|",
        ],
    );
}

/*
S2n
*/

#[test]
fn test_ur_group_symmetric_b2cl4_magnetic_field_s4() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/b2cl4.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    test_ur_ordinary_group(&mol, thresh, "S4", 4, 4, true);
}

#[test]
fn test_ur_group_symmetric_b2cl4_magnetic_field_s4_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/b2cl4.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    test_ur_ordinary_group_class_order(&mol, thresh, &["|E|", "|C2|", "|S4|", "|S4^(-1)|"]);
}

#[test]
fn test_ur_group_symmetric_b2cl4_magnetic_field_bw_d2d_s4() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/b2cl4.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    test_ur_magnetic_group(&mol, thresh, "D2d", 8, 5, false, BWGRP);
}

#[test]
fn test_ur_group_symmetric_b2cl4_magnetic_field_bw_d2d_s4_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/b2cl4.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    test_ur_magnetic_group_class_order(
        &mol,
        thresh,
        &["|E|", "|C2|", "2|S4|", "2|θ·C2|", "2|θ·σd|"],
    );
}

#[test]
fn test_ur_group_symmetric_adamantane_magnetic_field_s4() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/adamantane.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(-0.1, 0.0, 0.0)));
    test_ur_ordinary_group(&mol, thresh, "S4", 4, 4, true);
}

#[test]
fn test_ur_group_symmetric_adamantane_magnetic_field_s4_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/adamantane.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(-0.1, 0.0, 0.0)));
    test_ur_ordinary_group_class_order(&mol, thresh, &["|E|", "|C2|", "|S4|", "|S4^(-1)|"]);
}

#[test]
fn test_ur_group_symmetric_adamantane_magnetic_field_bw_d2d_s4() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/adamantane.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(-0.1, 0.0, 0.0)));
    test_ur_magnetic_group(&mol, thresh, "D2d", 8, 5, false, BWGRP);
}

#[test]
fn test_ur_group_symmetric_adamantane_magnetic_field_bw_d2d_s4_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/adamantane.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(-0.1, 0.0, 0.0)));
    test_ur_magnetic_group_class_order(
        &mol,
        thresh,
        &["|E|", "|C2|", "2|S4|", "2|θ·C2|", "2|θ·σd|"],
    );
}

#[test]
fn test_ur_group_symmetric_ch4_magnetic_field_s4() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/ch4.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    test_ur_ordinary_group(&mol, thresh, "S4", 4, 4, true);
}

#[test]
fn test_ur_group_symmetric_ch4_magnetic_field_s4_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/ch4.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    test_ur_ordinary_group_class_order(&mol, thresh, &["|E|", "|C2|", "|S4|", "|S4^(-1)|"]);
}

#[test]
fn test_ur_group_symmetric_ch4_magnetic_field_bw_d2d_s4() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/ch4.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    test_ur_magnetic_group(&mol, thresh, "D2d", 8, 5, false, BWGRP);
}

#[test]
fn test_ur_group_symmetric_ch4_magnetic_field_bw_d2d_s4_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/ch4.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    test_ur_magnetic_group_class_order(
        &mol,
        thresh,
        &["|E|", "|C2|", "2|S4|", "2|θ·C2|", "2|θ·σd|"],
    );
}

#[test]
fn test_ur_group_symmetric_65coronane_s6() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/coronane65.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_group(&mol, thresh, "S6", 6, 6, true);
}

#[test]
fn test_ur_group_symmetric_65coronane_s6_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/coronane65.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_group_class_order(
        &mol,
        thresh,
        &["|E|", "|C3|", "|C3^(-1)|", "|i|", "|S6|", "|S6^(-1)|"],
    );
}

#[test]
fn test_ur_group_symmetric_65coronane_grey_s6() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/coronane65.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_magnetic_group(&mol, thresh, "S6 + θ·S6", 12, 12, true, GRGRP);
}

#[test]
fn test_ur_group_symmetric_65coronane_grey_s6_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/coronane65.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_magnetic_group_class_order(
        &mol,
        thresh,
        &[
            "|E|",
            "|C3|",
            "|C3^(-1)|",
            "|i|",
            "|S6|",
            "|S6^(-1)|",
            "|θ|",
            "|θ·C3|",
            "|θ·C3^(-1)|",
            "|θ·i|",
            "|θ·S6|",
            "|θ·S6^(-1)|",
        ],
    );
}

#[test]
fn test_ur_group_symmetric_65coronane_magnetic_field_s6() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/coronane65.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, -1.0)));
    test_ur_ordinary_group(&mol, thresh, "S6", 6, 6, true);
}

#[test]
fn test_ur_group_symmetric_65coronane_magnetic_field_s6_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/coronane65.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, -1.0)));
    test_ur_ordinary_group_class_order(
        &mol,
        thresh,
        &["|E|", "|C3|", "|C3^(-1)|", "|i|", "|S6|", "|S6^(-1)|"],
    );
}

#[test]
fn test_ur_group_symmetric_staggered_c2h6_magnetic_field_s6() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c2h6.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    test_ur_ordinary_group(&mol, thresh, "S6", 6, 6, true);
}

#[test]
fn test_ur_group_symmetric_staggered_c2h6_magnetic_field_s6_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c2h6.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    test_ur_ordinary_group_class_order(
        &mol,
        thresh,
        &["|E|", "|C3|", "|C3^(-1)|", "|i|", "|S6|", "|S6^(-1)|"],
    );
}

#[test]
fn test_ur_group_symmetric_staggered_c2h6_magnetic_field_bw_d3d_s6() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c2h6.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    test_ur_magnetic_group(&mol, thresh, "D3d", 12, 6, false, BWGRP);
}

#[test]
fn test_ur_group_symmetric_staggered_c2h6_magnetic_field_bw_d3d_s6_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c2h6.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    test_ur_magnetic_group_class_order(
        &mol,
        thresh,
        &["|E|", "2|C3|", "|i|", "2|S6|", "3|θ·C2|", "3|θ·σd|"],
    );
}

#[test]
fn test_ur_group_symmetric_c60_magnetic_field_s6() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c60.xyz");
    let thresh = 1e-5;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(
        -0.5773503107731,
        -0.1875926572335,
        0.7946543988441,
    )));
    test_ur_ordinary_group(&mol, thresh, "S6", 6, 6, true);
}

#[test]
fn test_ur_group_symmetric_c60_magnetic_field_s6_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c60.xyz");
    let thresh = 1e-5;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(
        -0.5773503107731,
        -0.1875926572335,
        0.7946543988441,
    )));
    test_ur_ordinary_group_class_order(
        &mol,
        thresh,
        &["|E|", "|C3|", "|C3^(-1)|", "|i|", "|S6|", "|S6^(-1)|"],
    );
}

#[test]
fn test_ur_group_symmetric_c60_magnetic_field_bw_d3d_s6() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c60.xyz");
    let thresh = 1e-5;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(
        -0.5773503107731,
        -0.1875926572335,
        0.7946543988441,
    )));
    test_ur_magnetic_group(&mol, thresh, "D3d", 12, 6, false, BWGRP);
}

#[test]
fn test_ur_group_symmetric_c60_magnetic_field_bw_d3d_s6_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c60.xyz");
    let thresh = 1e-5;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(
        -0.5773503107731,
        -0.1875926572335,
        0.7946543988441,
    )));
    test_ur_magnetic_group_class_order(
        &mol,
        thresh,
        &["|E|", "2|C3|", "|i|", "2|S6|", "3|θ·C2|", "3|θ·σd|"],
    );
}

#[test]
fn test_ur_group_symmetric_vh2o6_magnetic_field_s6() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/vh2o6.xyz");
    let thresh = 1e-12;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(-0.2, 0.2, 0.2)));
    test_ur_ordinary_group(&mol, thresh, "S6", 6, 6, true);
}

#[test]
fn test_ur_group_symmetric_vh2o6_magnetic_field_s6_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/vh2o6.xyz");
    let thresh = 1e-12;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(-0.2, 0.2, 0.2)));
    test_ur_ordinary_group_class_order(
        &mol,
        thresh,
        &["|E|", "|C3|", "|C3^(-1)|", "|i|", "|S6|", "|S6^(-1)|"],
    );
}

#[test]
fn test_ur_group_symmetric_vf6_magnetic_field_s6() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/vf6.xyz");
    let thresh = 1e-12;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(1.0, 1.0, 1.0)));
    test_ur_ordinary_group(&mol, thresh, "S6", 6, 6, true);
}

#[test]
fn test_ur_group_symmetric_vf6_magnetic_field_s6_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/vf6.xyz");
    let thresh = 1e-12;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(1.0, 1.0, 1.0)));
    test_ur_ordinary_group_class_order(
        &mol,
        thresh,
        &["|E|", "|C3|", "|C3^(-1)|", "|i|", "|S6|", "|S6^(-1)|"],
    );
}

#[test]
fn test_ur_group_symmetric_vf6_magnetic_field_bw_d3d_s6() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/vf6.xyz");
    let thresh = 1e-12;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(1.0, 1.0, 1.0)));
    test_ur_magnetic_group(&mol, thresh, "D3d", 12, 6, false, BWGRP);
}

#[test]
fn test_ur_group_symmetric_vf6_magnetic_field_bw_d3d_s6_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/vf6.xyz");
    let thresh = 1e-12;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(1.0, 1.0, 1.0)));
    test_ur_magnetic_group_class_order(
        &mol,
        thresh,
        &["|E|", "2|C3|", "|i|", "2|S6|", "3|θ·C2|", "3|θ·σd|"],
    );
}

#[test]
fn test_ur_group_symmetric_s8_magnetic_field_s8() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/s8.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    test_ur_ordinary_group(&mol, thresh, "S8", 8, 8, true);
}

#[test]
fn test_ur_group_symmetric_s8_magnetic_field_s8_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/s8.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    test_ur_ordinary_group_class_order(
        &mol,
        thresh,
        &[
            "|E|",
            "|C4|",
            "|C4^(-1)|",
            "|C2|",
            "|S8|",
            "|σC8^3|",
            "|σC8^(-3)|",
            "|S8^(-1)|",
        ],
    );
}

#[test]
fn test_ur_group_symmetric_s8_magnetic_field_bw_d4d_s8() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/s8.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    test_ur_magnetic_group(&mol, thresh, "D4d", 16, 7, false, BWGRP);
}

#[test]
fn test_ur_group_symmetric_s8_magnetic_field_bw_d4d_s8_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/s8.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    test_ur_magnetic_group_class_order(
        &mol,
        thresh,
        &[
            "|E|",
            "2|C4|",
            "|C2|",
            "2|S8|",
            "2|σC8^3|",
            "4|θ·C2|",
            "4|θ·σd|",
        ],
    );
}

#[test]
fn test_ur_group_symmetric_antiprism_pb10_magnetic_field_s8() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/pb10.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    test_ur_ordinary_group(&mol, thresh, "S8", 8, 8, true);
}

#[test]
fn test_ur_group_symmetric_antiprism_pb10_magnetic_field_s8_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/pb10.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    test_ur_ordinary_group_class_order(
        &mol,
        thresh,
        &[
            "|E|",
            "|C4|",
            "|C4^(-1)|",
            "|C2|",
            "|S8|",
            "|σC8^3|",
            "|σC8^(-3)|",
            "|S8^(-1)|",
        ],
    );
}

#[test]
fn test_ur_group_symmetric_antiprism_pb10_magnetic_field_bw_d4d_s8() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/pb10.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    test_ur_magnetic_group(&mol, thresh, "D4d", 16, 7, false, BWGRP);
}

#[test]
fn test_ur_group_symmetric_antiprism_pb10_magnetic_field_bw_d4d_s8_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/pb10.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    test_ur_magnetic_group_class_order(
        &mol,
        thresh,
        &[
            "|E|",
            "2|C4|",
            "|C2|",
            "2|S8|",
            "2|σC8^3|",
            "4|θ·C2|",
            "4|θ·σd|",
        ],
    );
}

#[test]
fn test_ur_group_symmetric_staggered_ferrocene_magnetic_field_s10() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/staggered_ferrocene.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    test_ur_ordinary_group(&mol, thresh, "S10", 10, 10, true);
}

#[test]
fn test_ur_group_symmetric_staggered_ferrocene_magnetic_field_s10_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/staggered_ferrocene.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    test_ur_ordinary_group_class_order(
        &mol,
        thresh,
        &[
            "|E|",
            "|C5|",
            "|C5^2|",
            "|C5^(-2)|",
            "|C5^(-1)|",
            "|i|",
            "|S10|",
            "|σC10^3|",
            "|σC10^(-3)|",
            "|S10^(-1)|",
        ],
    );
}

#[test]
fn test_ur_group_symmetric_staggered_ferrocene_magnetic_field_bw_d5d_s10() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/staggered_ferrocene.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    test_ur_magnetic_group(&mol, thresh, "D5d", 20, 8, false, BWGRP);
}

#[test]
fn test_ur_group_symmetric_staggered_ferrocene_magnetic_field_bw_d5d_s10_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/staggered_ferrocene.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    test_ur_magnetic_group_class_order(
        &mol,
        thresh,
        &[
            "|E|",
            "2|C5|",
            "2|C5^2|",
            "|i|",
            "2|S10|",
            "2|σC10^3|",
            "5|θ·C2|",
            "5|θ·σd|",
        ],
    );
}

#[test]
fn test_ur_group_symmetric_c60_magnetic_field_s10() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c60.xyz");
    let thresh = 1e-5;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    test_ur_ordinary_group(&mol, thresh, "S10", 10, 10, true);
}

#[test]
fn test_ur_group_symmetric_c60_magnetic_field_s10_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c60.xyz");
    let thresh = 1e-5;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    test_ur_ordinary_group_class_order(
        &mol,
        thresh,
        &[
            "|E|",
            "|C5|",
            "|C5^2|",
            "|C5^(-2)|",
            "|C5^(-1)|",
            "|i|",
            "|S10|",
            "|σC10^3|",
            "|σC10^(-3)|",
            "|S10^(-1)|",
        ],
    );
}

#[test]
fn test_ur_group_symmetric_au26_magnetic_field_s12() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/au26.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    test_ur_ordinary_group(&mol, thresh, "S12", 12, 12, true);
}

#[test]
fn test_ur_group_symmetric_au26_magnetic_field_s12_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/au26.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    test_ur_ordinary_group_class_order(
        &mol,
        thresh,
        &[
            "|E|",
            "|C6|",
            "|C6^(-1)|",
            "|C3|",
            "|C3^(-1)|",
            "|C2|",
            "|S12|",
            "|σC12^5|",
            "|σC12^(-5)|",
            "|S12^(-1)|",
            "|S4|",
            "|S4^(-1)|",
        ],
    );
}

#[test]
fn test_ur_group_symmetric_au26_magnetic_field_bw_d6d_s12() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/au26.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    test_ur_magnetic_group(&mol, thresh, "D6d", 24, 9, false, BWGRP);
}

#[test]
fn test_ur_group_symmetric_au26_magnetic_field_bw_d6d_s12_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/au26.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    test_ur_magnetic_group_class_order(
        &mol,
        thresh,
        &[
            "|E|",
            "2|C6|",
            "2|C3|",
            "|C2|",
            "2|S12|",
            "2|σC12^5|",
            "2|S4|",
            "6|θ·C2|",
            "6|θ·σd|",
        ],
    );
}

#[test]
#[ignore]
fn test_ur_group_symmetric_arbitrary_staggered_sandwich_magnetic_field_s2n() {
    let thresh = 1e-7;
    for n in 3..=20 {
        let mut mol = template_molecules::gen_arbitrary_twisted_sandwich(n, 0.5);
        mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
        test_ur_ordinary_group(
            &mol,
            thresh,
            format!("S{}", 2 * n).as_str(),
            2 * n as usize,
            2 * n as usize,
            true,
        );
    }
}

#[test]
#[ignore]
fn test_ur_group_symmetric_arbitrary_staggered_sandwich_magnetic_field_bw_dnd_s2n() {
    let thresh = 1e-7;
    for n in 3..=20 {
        let mut mol = template_molecules::gen_arbitrary_twisted_sandwich(n, 0.5);
        mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
        test_ur_magnetic_group(
            &mol,
            thresh,
            format!("D{n}d").as_str(),
            4 * n as usize,
            (4 + n - 1) as usize,
            false,
            BWGRP,
        );
    }
}

/***
S2n*
***/
#[test]
fn test_ur_group_symmetric_b2cl4_magnetic_field_s4_double() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/b2cl4.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    test_ur_ordinary_double_group(&mol, thresh, "S4*", 8, 8, true);
}

#[test]
fn test_ur_group_symmetric_b2cl4_magnetic_field_s4_double_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/b2cl4.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    test_ur_ordinary_double_group_class_order(
        &mol,
        thresh,
        &[
            "|E(Σ)|",
            "|E(QΣ)|",
            "|C2(Σ)|",
            "|C2(QΣ)|",
            "|S4(Σ)|",
            "|S4(QΣ)|",
            "|S4^(-1)(Σ)|",
            "|S4^(-1)(QΣ)|",
        ],
    );
}

#[test]
fn test_ur_group_symmetric_b2cl4_magnetic_field_bw_d2d_s4_double() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/b2cl4.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    test_ur_magnetic_double_group(&mol, thresh, "D2d*", 16, 7, false, BWGRP2);
}

#[test]
fn test_ur_group_symmetric_b2cl4_magnetic_field_bw_d2d_s4_double_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/b2cl4.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    test_ur_magnetic_double_group_class_order(
        &mol,
        thresh,
        &[
            "|E(Σ)|",
            "|E(QΣ)|",
            "|C2(Σ), C2(QΣ)|",
            "2|S4(Σ)|",
            "2|S4(QΣ)|",
            "2|θ·C2(Σ), θ·C2(QΣ)|",
            "2|θ·σd(Σ), θ·σd(QΣ)|",
        ],
    );
}

#[test]
fn test_ur_group_symmetric_vf6_magnetic_field_s6_double() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/vf6.xyz");
    let thresh = 1e-12;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(1.0, 1.0, 1.0)));
    test_ur_ordinary_double_group(&mol, thresh, "S6*", 12, 12, true);
}

#[test]
fn test_ur_group_symmetric_vf6_magnetic_field_s6_double_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/vf6.xyz");
    let thresh = 1e-12;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(1.0, 1.0, 1.0)));
    test_ur_ordinary_double_group_class_order(
        &mol,
        thresh,
        &[
            "|E(Σ)|",
            "|E(QΣ)|",
            "|C3(Σ)|",
            "|C3(QΣ)|",
            "|C3^(-1)(Σ)|",
            "|C3^(-1)(QΣ)|",
            "|i(Σ)|",
            "|i(QΣ)|",
            "|S6(Σ)|",
            "|S6(QΣ)|",
            "|S6^(-1)(Σ)|",
            "|S6^(-1)(QΣ)|",
        ],
    );
}

#[test]
fn test_ur_group_symmetric_vf6_magnetic_field_bw_d3d_s6_double() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/vf6.xyz");
    let thresh = 1e-12;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(1.0, 1.0, 1.0)));
    test_ur_magnetic_double_group(&mol, thresh, "D3d*", 24, 12, false, BWGRP2);
}

#[test]
fn test_ur_group_symmetric_vf6_magnetic_field_bw_d3d_s6_double_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/vf6.xyz");
    let thresh = 1e-12;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(1.0, 1.0, 1.0)));
    test_ur_magnetic_double_group_class_order(
        &mol,
        thresh,
        &[
            "|E(Σ)|",
            "|E(QΣ)|",
            "2|C3(Σ)|",
            "2|C3(QΣ)|",
            "|i(Σ)|",
            "|i(QΣ)|",
            "2|S6(Σ)|",
            "2|S6(QΣ)|",
            "3|θ·C2(Σ)|",
            "3|θ·C2(QΣ)|",
            "3|θ·σd(Σ)|",
            "3|θ·σd(QΣ)|",
        ],
    );
}

#[test]
fn test_ur_group_symmetric_antiprism_pb10_magnetic_field_s8_double() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/pb10.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    test_ur_ordinary_double_group(&mol, thresh, "S8*", 16, 16, true);
}

#[test]
fn test_ur_group_symmetric_antiprism_pb10_magnetic_field_s8_double_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/pb10.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    test_ur_ordinary_double_group_class_order(
        &mol,
        thresh,
        &[
            "|E(Σ)|",
            "|E(QΣ)|",
            "|C4(Σ)|",
            "|C4(QΣ)|",
            "|C4^(-1)(Σ)|",
            "|C4^(-1)(QΣ)|",
            "|C2(Σ)|",
            "|C2(QΣ)|",
            "|S8(Σ)|",
            "|S8(QΣ)|",
            "|σC8^3(Σ)|",
            "|σC8^3(QΣ)|",
            "|σC8^(-3)(Σ)|",
            "|σC8^(-3)(QΣ)|",
            "|S8^(-1)(Σ)|",
            "|S8^(-1)(QΣ)|",
        ],
    );
}

#[test]
fn test_ur_group_symmetric_antiprism_pb10_magnetic_field_bw_d4d_s8_double() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/pb10.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    test_ur_magnetic_double_group(&mol, thresh, "D4d*", 32, 11, false, BWGRP2);
}

#[test]
fn test_ur_group_symmetric_antiprism_pb10_magnetic_field_bw_d4d_s8_double_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/pb10.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    test_ur_magnetic_double_group_class_order(
        &mol,
        thresh,
        &[
            "|E(Σ)|",
            "|E(QΣ)|",
            "2|C4(Σ)|",
            "2|C4(QΣ)|",
            "|C2(Σ), C2(QΣ)|",
            "2|S8(Σ)|",
            "2|S8(QΣ)|",
            "2|σC8^3(Σ)|",
            "2|σC8^3(QΣ)|",
            "4|θ·C2(Σ), θ·C2(QΣ)|",
            "4|θ·σd(Σ), θ·σd(QΣ)|",
        ],
    );
}

#[test]
fn test_ur_group_symmetric_c60_magnetic_field_s10_double() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c60.xyz");
    let thresh = 1e-5;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    test_ur_ordinary_double_group(&mol, thresh, "S10*", 20, 20, true);
}

#[test]
fn test_ur_group_symmetric_c60_magnetic_field_s10_double_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c60.xyz");
    let thresh = 1e-5;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    test_ur_ordinary_double_group_class_order(
        &mol,
        thresh,
        &[
            "|E(Σ)|",
            "|E(QΣ)|",
            "|C5(Σ)|",
            "|C5(QΣ)|",
            "|C5^2(Σ)|",
            "|C5^2(QΣ)|",
            "|C5^(-2)(Σ)|",
            "|C5^(-2)(QΣ)|",
            "|C5^(-1)(Σ)|",
            "|C5^(-1)(QΣ)|",
            "|i(Σ)|",
            "|i(QΣ)|",
            "|S10(Σ)|",
            "|S10(QΣ)|",
            "|σC10^3(Σ)|",
            "|σC10^3(QΣ)|",
            "|σC10^(-3)(Σ)|",
            "|σC10^(-3)(QΣ)|",
            "|S10^(-1)(Σ)|",
            "|S10^(-1)(QΣ)|",
        ],
    );
}

#[test]
fn test_ur_group_symmetric_staggered_ferrocene_magnetic_field_bw_d5d_s10_double() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/staggered_ferrocene.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    test_ur_magnetic_double_group(&mol, thresh, "D5d*", 40, 16, false, BWGRP2);
}

#[test]
fn test_ur_group_symmetric_staggered_ferrocene_magnetic_field_bw_d5d_s10_double_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/staggered_ferrocene.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    test_ur_magnetic_double_group_class_order(
        &mol,
        thresh,
        &[
            "|E(Σ)|",
            "|E(QΣ)|",
            "2|C5(Σ)|",
            "2|C5(QΣ)|",
            "2|C5^2(Σ)|",
            "2|C5^2(QΣ)|",
            "|i(Σ)|",
            "|i(QΣ)|",
            "2|S10(Σ)|",
            "2|S10(QΣ)|",
            "2|σC10^3(Σ)|",
            "2|σC10^3(QΣ)|",
            "5|θ·C2(Σ)|",
            "5|θ·C2(QΣ)|",
            "5|θ·σd(Σ)|",
            "5|θ·σd(QΣ)|",
        ],
    );
}

#[test]
fn test_ur_group_symmetric_au26_magnetic_field_s12_double() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/au26.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    test_ur_ordinary_double_group(&mol, thresh, "S12*", 24, 24, true);
}

#[test]
fn test_ur_group_symmetric_au26_magnetic_field_s12_double_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/au26.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    test_ur_ordinary_double_group_class_order(
        &mol,
        thresh,
        &[
            "|E(Σ)|",
            "|E(QΣ)|",
            "|C6(Σ)|",
            "|C6(QΣ)|",
            "|C6^(-1)(Σ)|",
            "|C6^(-1)(QΣ)|",
            "|C3(Σ)|",
            "|C3(QΣ)|",
            "|C3^(-1)(Σ)|",
            "|C3^(-1)(QΣ)|",
            "|C2(Σ)|",
            "|C2(QΣ)|",
            "|S12(Σ)|",
            "|S12(QΣ)|",
            "|σC12^5(Σ)|",
            "|σC12^5(QΣ)|",
            "|σC12^(-5)(Σ)|",
            "|σC12^(-5)(QΣ)|",
            "|S12^(-1)(Σ)|",
            "|S12^(-1)(QΣ)|",
            "|S4(Σ)|",
            "|S4(QΣ)|",
            "|S4^(-1)(Σ)|",
            "|S4^(-1)(QΣ)|",
        ],
    );
}

#[test]
fn test_ur_group_symmetric_au26_magnetic_field_bw_d6d_s12_double() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/au26.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    test_ur_magnetic_double_group(&mol, thresh, "D6d*", 48, 15, false, BWGRP2);
}

#[test]
fn test_ur_group_symmetric_au26_magnetic_field_bw_d6d_s12_double_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/au26.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    test_ur_magnetic_double_group_class_order(
        &mol,
        thresh,
        &[
            "|E(Σ)|",
            "|E(QΣ)|",
            "2|C6(Σ)|",
            "2|C6(QΣ)|",
            "2|C3(Σ)|",
            "2|C3(QΣ)|",
            "|C2(Σ), C2(QΣ)|",
            "2|S12(Σ)|",
            "2|S12(QΣ)|",
            "2|σC12^5(Σ)|",
            "2|σC12^5(QΣ)|",
            "2|S4(Σ)|",
            "2|S4(QΣ)|",
            "6|θ·C2(Σ), θ·C2(QΣ)|",
            "6|θ·σd(Σ), θ·σd(QΣ)|",
        ],
    );
}

/*********
Asymmetric
*********/

/*
C2
*/

#[test]
fn test_ur_group_asymmetric_spiroketal_c2() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/spiroketal.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_group(&mol, thresh, "C2", 2, 2, true);
}

#[test]
fn test_ur_group_asymmetric_spiroketal_c2_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/spiroketal.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_group_class_order(&mol, thresh, &["|E|", "|C2|"]);
}

#[test]
fn test_ur_group_asymmetric_spiroketal_grey_c2() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/spiroketal.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_magnetic_group(&mol, thresh, "C2 + θ·C2", 4, 4, true, GRGRP);
}

#[test]
fn test_ur_group_asymmetric_spiroketal_grey_c2_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/spiroketal.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_magnetic_group_class_order(&mol, thresh, &["|E|", "|C2|", "|θ|", "|θ·C2|"]);
}

#[test]
fn test_ur_group_asymmetric_cyclohexene_c2() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/cyclohexene.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_group(&mol, thresh, "C2", 2, 2, true);
}

#[test]
fn test_ur_group_asymmetric_thf_c2() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/thf.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_group(&mol, thresh, "C2", 2, 2, true);
}

#[test]
fn test_ur_group_asymmetric_tartaricacid_c2() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/tartaricacid.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_group(&mol, thresh, "C2", 2, 2, true);
}

#[test]
fn test_ur_group_asymmetric_f2allene_c2() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/f2allene.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_group(&mol, thresh, "C2", 2, 2, true);
}

#[test]
fn test_ur_group_asymmetric_water_magnetic_field_c2() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/water.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 1.0, 0.0)));
    test_ur_ordinary_group(&mol, thresh, "C2", 2, 2, true);
}

#[test]
fn test_ur_group_asymmetric_water_magnetic_field_bw_c2v_c2() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/water.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 1.0, 0.0)));
    test_ur_magnetic_group(&mol, thresh, "C2v", 4, 4, true, BWGRP);
}

#[test]
fn test_ur_group_asymmetric_water_magnetic_field_bw_c2v_c2_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/water.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 1.0, 0.0)));
    test_ur_magnetic_group_class_order(&mol, thresh, &["|E|", "|C2|", "|θ·σv|", "|θ·σv|^(')"]);
}

#[test]
fn test_ur_group_asymmetric_pyridine_magnetic_field_c2() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/pyridine.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.2, 0.0)));
    test_ur_ordinary_group(&mol, thresh, "C2", 2, 2, true);
}

#[test]
fn test_ur_group_asymmetric_pyridine_magnetic_field_bw_c2v_c2() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/pyridine.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.2, 0.0)));
    test_ur_magnetic_group(&mol, thresh, "C2v", 4, 4, true, BWGRP);
}

#[test]
fn test_ur_group_asymmetric_pyridine_magnetic_field_bw_c2v_c2_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/pyridine.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.2, 0.0)));
    test_ur_magnetic_group_class_order(&mol, thresh, &["|E|", "|C2|", "|θ·σv|", "|θ·σv|^(')"]);
}

#[test]
fn test_ur_group_asymmetric_cyclobutene_magnetic_field_c2() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/cyclobutene.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.2, 0.0, 0.0)));
    test_ur_ordinary_group(&mol, thresh, "C2", 2, 2, true);
}

#[test]
fn test_ur_group_asymmetric_cyclobutene_magnetic_field_bw_c2v_c2() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/cyclobutene.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.2, 0.0, 0.0)));
    test_ur_magnetic_group(&mol, thresh, "C2v", 4, 4, true, BWGRP);
}

#[test]
fn test_ur_group_asymmetric_cyclobutene_magnetic_field_bw_c2v_c2_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/cyclobutene.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.2, 0.0, 0.0)));
    test_ur_magnetic_group_class_order(&mol, thresh, &["|E|", "|C2|", "|θ·σv|", "|θ·σv|^(')"]);
}

#[test]
fn test_ur_group_asymmetric_azulene_magnetic_field_c2() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/azulene.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 0.2)));
    test_ur_ordinary_group(&mol, thresh, "C2", 2, 2, true);
}

#[test]
fn test_ur_group_asymmetric_azulene_magnetic_field_bw_c2v_c2() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/azulene.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 0.2)));
    test_ur_magnetic_group(&mol, thresh, "C2v", 4, 4, true, BWGRP);
}

#[test]
fn test_ur_group_asymmetric_azulene_magnetic_field_bw_c2v_c2_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/azulene.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 0.2)));
    test_ur_magnetic_group_class_order(&mol, thresh, &["|E|", "|C2|", "|θ·σv|", "|θ·σv|^(')"]);
}

#[test]
fn test_ur_group_asymmetric_cis_cocl2h4o2_magnetic_field_c2() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/cis-cocl2h4o2.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.2, 0.0)));
    test_ur_ordinary_group(&mol, thresh, "C2", 2, 2, true);
}

#[test]
fn test_ur_group_asymmetric_cis_cocl2h4o2_magnetic_field_bw_c2v_c2() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/cis-cocl2h4o2.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.2, 0.0)));
    test_ur_magnetic_group(&mol, thresh, "C2v", 4, 4, true, BWGRP);
}

#[test]
fn test_ur_group_asymmetric_cis_cocl2h4o2_magnetic_field_bw_c2v_c2_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/cis-cocl2h4o2.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.2, 0.0)));
    test_ur_magnetic_group_class_order(&mol, thresh, &["|E|", "|C2|", "|θ·σv|", "|θ·σv|^(')"]);
}

#[test]
fn test_ur_group_asymmetric_cuneane_magnetic_field_c2() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/cuneane.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 0.2)));
    test_ur_ordinary_group(&mol, thresh, "C2", 2, 2, true);
}

#[test]
fn test_ur_group_asymmetric_cuneane_magnetic_field_bw_c2v_c2() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/cuneane.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 0.2)));
    test_ur_magnetic_group(&mol, thresh, "C2v", 4, 4, true, BWGRP);
}

#[test]
fn test_ur_group_asymmetric_cuneane_magnetic_field_bw_c2v_c2_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/cuneane.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 0.2)));
    test_ur_magnetic_group_class_order(&mol, thresh, &["|E|", "|C2|", "|θ·σv|", "|θ·σv|^(')"]);
}

/***
C2*
***/

#[test]
fn test_ur_group_asymmetric_spiroketal_c2_double() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/spiroketal.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_double_group(&mol, thresh, "C2*", 4, 4, true);
}

#[test]
fn test_ur_group_asymmetric_spiroketal_c2_double_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/spiroketal.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_double_group_class_order(
        &mol,
        thresh,
        &["|E(Σ)|", "|E(QΣ)|", "|C2(Σ)|", "|C2(QΣ)|"],
    );
}

#[test]
fn test_ur_group_asymmetric_spiroketal_grey_c2_double() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/spiroketal.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_magnetic_double_group(&mol, thresh, "(C2 + θ·C2)*", 8, 8, true, GRGRP2);
}

#[test]
fn test_ur_group_asymmetric_spiroketal_grey_c2_double_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/spiroketal.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_magnetic_double_group_class_order(
        &mol,
        thresh,
        &[
            "|E(Σ)|",
            "|E(QΣ)|",
            "|C2(Σ)|",
            "|C2(QΣ)|",
            "|θ(Σ)|",
            "|θ(QΣ)|",
            "|θ·C2(Σ)|",
            "|θ·C2(QΣ)|",
        ],
    );
}

/***
C2v
***/

#[test]
fn test_ur_group_asymmetric_water_c2v() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/water.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_group(&mol, thresh, "C2v", 4, 4, true);
}

#[test]
fn test_ur_group_asymmetric_water_c2v_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/water.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_group_class_order(&mol, thresh, &["|E|", "|C2|", "|σv|", "|σv|^(')"]);
}

#[test]
fn test_ur_group_asymmetric_water_grey_c2v() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/water.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_magnetic_group(&mol, thresh, "C2v + θ·C2v", 8, 8, true, GRGRP);
}

#[test]
fn test_ur_group_asymmetric_water_grey_c2v_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/water.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_magnetic_group_class_order(
        &mol,
        thresh,
        &[
            "|E|",
            "|C2|",
            "|σv|",
            "|σv|^(')",
            "|θ|",
            "|θ·C2|",
            "|θ·σv|",
            "|θ·σv|^(')",
        ],
    );
}

#[test]
fn test_ur_group_asymmetric_pyridine_c2v() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/pyridine.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_group(&mol, thresh, "C2v", 4, 4, true);
}

#[test]
fn test_ur_group_asymmetric_pyridine_c2v_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/pyridine.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_group_class_order(&mol, thresh, &["|E|", "|C2|", "|σv|", "|σv|^(')"]);
}

#[test]
fn test_ur_group_asymmetric_pyridine_grey_c2v() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/pyridine.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_magnetic_group(&mol, thresh, "C2v + θ·C2v", 8, 8, true, GRGRP);
}

#[test]
fn test_ur_group_asymmetric_pyridine_grey_c2v_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/pyridine.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_magnetic_group_class_order(
        &mol,
        thresh,
        &[
            "|E|",
            "|C2|",
            "|σv|",
            "|σv|^(')",
            "|θ|",
            "|θ·C2|",
            "|θ·σv|",
            "|θ·σv|^(')",
        ],
    );
}

#[test]
fn test_ur_group_asymmetric_cyclobutene_c2v() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/cyclobutene.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_group(&mol, thresh, "C2v", 4, 4, true);
}

#[test]
fn test_ur_group_asymmetric_azulene_c2v() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/azulene.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_group(&mol, thresh, "C2v", 4, 4, true);
}

#[test]
fn test_ur_group_asymmetric_cuneane_c2v() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/cuneane.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_group(&mol, thresh, "C2v", 4, 4, true);
}

#[test]
fn test_ur_group_asymmetric_bf3_electric_field_c2v() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/bf3.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(Vector3::new(0.2, 0.0, 0.0)));
    test_ur_ordinary_group(&mol, thresh, "C2v", 4, 4, true);
}

#[test]
fn test_ur_group_asymmetric_bf3_electric_field_grey_c2v() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/bf3.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(Vector3::new(0.2, 0.0, 0.0)));
    test_ur_magnetic_group(&mol, thresh, "C2v + θ·C2v", 8, 8, true, GRGRP);
}

/***
C2v*
***/

#[test]
fn test_ur_group_asymmetric_water_c2v_double() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/water.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_double_group(&mol, thresh, "C2v*", 8, 5, false);
}

#[test]
fn test_ur_group_asymmetric_water_c2v_double_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/water.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_double_group_class_order(
        &mol,
        thresh,
        &[
            "|E(Σ)|",
            "|E(QΣ)|",
            "|C2(Σ), C2(QΣ)|",
            "|σv(Σ), σv(QΣ)|",
            "|σv(Σ), σv(QΣ)|^(')",
        ],
    );
}

#[test]
fn test_ur_group_asymmetric_pyridine_grey_c2v_double() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/pyridine.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_magnetic_double_group(&mol, thresh, "(C2v + θ·C2v)*", 16, 10, false, GRGRP2);
}

#[test]
fn test_ur_group_asymmetric_pyridine_grey_c2v_double_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/pyridine.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_magnetic_double_group_class_order(
        &mol,
        thresh,
        &[
            "|E(Σ)|",
            "|E(QΣ)|",
            "|C2(Σ), C2(QΣ)|",
            "|σv(Σ), σv(QΣ)|",
            "|σv(Σ), σv(QΣ)|^(')",
            "|θ(Σ)|",
            "|θ(QΣ)|",
            "|θ·C2(Σ), θ·C2(QΣ)|",
            "|θ·σv(Σ), θ·σv(QΣ)|",
            "|θ·σv(Σ), θ·σv(QΣ)|^(')",
        ],
    );
}

/***
C2h
***/

#[test]
fn test_ur_group_asymmetric_h2o2_c2h() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/h2o2.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_group(&mol, thresh, "C2h", 4, 4, true);
}

#[test]
fn test_ur_group_asymmetric_h2o2_c2h_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/h2o2.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_group_class_order(&mol, thresh, &["|E|", "|C2|", "|i|", "|σh|"]);
}

#[test]
fn test_ur_group_asymmetric_h2o2_grey_c2h() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/h2o2.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_magnetic_group(&mol, thresh, "C2h + θ·C2h", 8, 8, true, GRGRP);
}

#[test]
fn test_ur_group_asymmetric_h2o2_grey_c2h_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/h2o2.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_magnetic_group_class_order(
        &mol,
        thresh,
        &[
            "|E|",
            "|C2|",
            "|i|",
            "|σh|",
            "|θ|",
            "|θ·C2|",
            "|θ·i|",
            "|θ·σh|",
        ],
    );
}

#[test]
fn test_ur_group_asymmetric_zethrene_c2h() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/zethrene.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_group(&mol, thresh, "C2h", 4, 4, true);
}

#[test]
fn test_ur_group_asymmetric_zethrene_c2h_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/zethrene.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_group_class_order(&mol, thresh, &["|E|", "|C2|", "|i|", "|σh|"]);
}

#[test]
fn test_ur_group_asymmetric_zethrene_grey_c2h() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/zethrene.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_magnetic_group(&mol, thresh, "C2h + θ·C2h", 8, 8, true, GRGRP);
}

#[test]
fn test_ur_group_asymmetric_zethrene_grey_c2h_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/zethrene.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_magnetic_group_class_order(
        &mol,
        thresh,
        &[
            "|E|",
            "|C2|",
            "|i|",
            "|σh|",
            "|θ|",
            "|θ·C2|",
            "|θ·i|",
            "|θ·σh|",
        ],
    );
}

#[test]
fn test_ur_group_asymmetric_distorted_vf6_magnetic_field_c2h() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/vf6_d2h.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(1.0, 0.0, 0.0)));
    test_ur_ordinary_group(&mol, thresh, "C2h", 4, 4, true);
}

#[test]
fn test_ur_group_asymmetric_distorted_vf6_magnetic_field_bw_d2h_c2h() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/vf6_d2h.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(1.0, 0.0, 0.0)));
    test_ur_magnetic_group(&mol, thresh, "D2h", 8, 8, true, BWGRP);
}

#[test]
fn test_ur_group_asymmetric_distorted_vf6_magnetic_field_bw_d2h_c2h_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/vf6_d2h.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(1.0, 0.0, 0.0)));
    test_ur_magnetic_group_class_order(
        &mol,
        thresh,
        &[
            "|E|",
            "|C2|",
            "|i|",
            "|σ|",
            "|θ·C2|",
            "|θ·C2|^(')",
            "|θ·σ|",
            "|θ·σ|^(')",
        ],
    );
}

#[test]
fn test_ur_group_asymmetric_b2h6_magnetic_field_c2h() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/b2h6.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(1.0, 0.0, 0.0)));
    test_ur_ordinary_group(&mol, thresh, "C2h", 4, 4, true);
}

#[test]
fn test_ur_group_asymmetric_b2h6_magnetic_field_bw_d2h_c2h() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/b2h6.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(1.0, 0.0, 0.0)));
    test_ur_magnetic_group(&mol, thresh, "D2h", 8, 8, true, BWGRP);
}

#[test]
fn test_ur_group_asymmetric_b2h6_magnetic_field_bw_d2h_c2h_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/b2h6.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(1.0, 0.0, 0.0)));
    test_ur_magnetic_group_class_order(
        &mol,
        thresh,
        &[
            "|E|",
            "|C2|",
            "|i|",
            "|σ|",
            "|θ·C2|",
            "|θ·C2|^(')",
            "|θ·σ|",
            "|θ·σ|^(')",
        ],
    );
}

#[test]
fn test_ur_group_asymmetric_naphthalene_magnetic_field_c2h() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/naphthalene.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 1.0, 0.0)));
    test_ur_ordinary_group(&mol, thresh, "C2h", 4, 4, true);
}

#[test]
fn test_ur_group_asymmetric_pyrene_magnetic_field_c2h() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/pyrene.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    test_ur_ordinary_group(&mol, thresh, "C2h", 4, 4, true);
}

#[test]
fn test_ur_group_asymmetric_pyrene_magnetic_field_bw_d2h_c2h() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/pyrene.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    test_ur_magnetic_group(&mol, thresh, "D2h", 8, 8, true, BWGRP);
}

#[test]
fn test_ur_group_asymmetric_pyrene_magnetic_field_bw_d2h_c2h_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/pyrene.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    test_ur_magnetic_group_class_order(
        &mol,
        thresh,
        &[
            "|E|",
            "|C2|",
            "|i|",
            "|σ|",
            "|θ·C2|",
            "|θ·C2|^(')",
            "|θ·σ|",
            "|θ·σ|^(')",
        ],
    );
}

#[test]
fn test_ur_group_asymmetric_c6o6_magnetic_field_c2h() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c6o6.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    test_ur_ordinary_group(&mol, thresh, "C2h", 4, 4, true);
}

#[test]
fn test_ur_group_asymmetric_c6o6_magnetic_field_bw_d2h_c2h() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c6o6.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    test_ur_magnetic_group(&mol, thresh, "D2h", 8, 8, true, BWGRP);
}

#[test]
fn test_ur_group_asymmetric_c6o6_magnetic_field_bw_d2h_c2h_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c6o6.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    test_ur_magnetic_group_class_order(
        &mol,
        thresh,
        &[
            "|E|",
            "|C2|",
            "|i|",
            "|σ|",
            "|θ·C2|",
            "|θ·C2|^(')",
            "|θ·σ|",
            "|θ·σ|^(')",
        ],
    );
}

/***
C2h*
***/

#[test]
fn test_ur_group_asymmetric_h2o2_c2h_double() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/h2o2.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_double_group(&mol, thresh, "C2h*", 8, 8, true);
}

#[test]
fn test_ur_group_asymmetric_h2o2_c2h_double_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/h2o2.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_double_group_class_order(
        &mol,
        thresh,
        &[
            "|E(Σ)|",
            "|E(QΣ)|",
            "|C2(Σ)|",
            "|C2(QΣ)|",
            "|i(Σ)|",
            "|i(QΣ)|",
            "|σh(Σ)|",
            "|σh(QΣ)|",
        ],
    );
}

#[test]
fn test_ur_group_asymmetric_h2o2_grey_c2h_double() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/h2o2.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_magnetic_double_group(&mol, thresh, "(C2h + θ·C2h)*", 16, 16, true, GRGRP2);
}

#[test]
fn test_ur_group_asymmetric_h2o2_grey_c2h_double_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/h2o2.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_magnetic_double_group_class_order(
        &mol,
        thresh,
        &[
            "|E(Σ)|",
            "|E(QΣ)|",
            "|C2(Σ)|",
            "|C2(QΣ)|",
            "|i(Σ)|",
            "|i(QΣ)|",
            "|σh(Σ)|",
            "|σh(QΣ)|",
            "|θ(Σ)|",
            "|θ(QΣ)|",
            "|θ·C2(Σ)|",
            "|θ·C2(QΣ)|",
            "|θ·i(Σ)|",
            "|θ·i(QΣ)|",
            "|θ·σh(Σ)|",
            "|θ·σh(QΣ)|",
        ],
    );
}

#[test]
fn test_ur_group_asymmetric_distorted_vf6_magnetic_field_bw_d2h_c2h_double() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/vf6_d2h.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(1.0, 0.0, 0.0)));
    test_ur_magnetic_double_group(&mol, thresh, "D2h*", 16, 10, false, BWGRP2);
}

#[test]
fn test_ur_group_asymmetric_distorted_vf6_magnetic_field_bw_d2h_c2h_double_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/vf6_d2h.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(1.0, 0.0, 0.0)));
    test_ur_magnetic_double_group_class_order(
        &mol,
        thresh,
        &[
            "|E(Σ)|",
            "|E(QΣ)|",
            "|C2(Σ), C2(QΣ)|",
            "|i(Σ)|",
            "|i(QΣ)|",
            "|σ(Σ), σ(QΣ)|",
            "|θ·C2(Σ), θ·C2(QΣ)|",
            "|θ·C2(Σ), θ·C2(QΣ)|^(')",
            "|θ·σ(Σ), θ·σ(QΣ)|",
            "|θ·σ(Σ), θ·σ(QΣ)|^(')",
        ],
    );
}

/*
Cs
*/

#[test]
fn test_ur_group_asymmetric_propene_cs() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/propene.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_group(&mol, thresh, "Cs", 2, 2, true);
}

#[test]
fn test_ur_group_asymmetric_propene_cs_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/propene.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_group_class_order(&mol, thresh, &["|E|", "|σh|"]);
}

#[test]
fn test_ur_group_asymmetric_propene_grey_cs() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/propene.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_magnetic_group(&mol, thresh, "Cs + θ·Cs", 4, 4, true, GRGRP);
}

#[test]
fn test_ur_group_asymmetric_propene_grey_cs_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/propene.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_magnetic_group_class_order(&mol, thresh, &["|E|", "|σh|", "|θ|", "|θ·σh|"]);
}

#[test]
fn test_ur_group_asymmetric_socl2_cs() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/socl2.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_group(&mol, thresh, "Cs", 2, 2, true);
}

#[test]
fn test_ur_group_asymmetric_socl2_cs_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/socl2.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_group_class_order(&mol, thresh, &["|E|", "|σh|"]);
}

#[test]
fn test_ur_group_asymmetric_socl2_grey_cs() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/socl2.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_magnetic_group(&mol, thresh, "Cs + θ·Cs", 4, 4, true, GRGRP);
}

#[test]
fn test_ur_group_asymmetric_socl2_grey_cs_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/socl2.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_magnetic_group_class_order(&mol, thresh, &["|E|", "|σh|", "|θ|", "|θ·σh|"]);
}

#[test]
fn test_ur_group_asymmetric_hocl_cs() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/hocl.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_group(&mol, thresh, "Cs", 2, 2, true);
}

#[test]
fn test_ur_group_asymmetric_hocn_cs() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/hocn.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_group(&mol, thresh, "Cs", 2, 2, true);
}

#[test]
fn test_ur_group_asymmetric_nh2f_cs() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/nh2f.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_group(&mol, thresh, "Cs", 2, 2, true);
}

#[test]
fn test_ur_group_asymmetric_phenol_cs() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/phenol.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_group(&mol, thresh, "Cs", 2, 2, true);
}

#[test]
fn test_ur_group_asymmetric_f_pyrrole_cs() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/f-pyrrole.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_group(&mol, thresh, "Cs", 2, 2, true);
}

#[test]
fn test_ur_group_asymmetric_n2o_cs() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/n2o.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_group(&mol, thresh, "Cs", 2, 2, true);
}

#[test]
fn test_ur_group_asymmetric_fclbenzene_cs() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/fclbenzene.xyz");
    let thresh = 1e-5;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_group(&mol, thresh, "Cs", 2, 2, true);
}

#[test]
fn test_ur_group_asymmetric_water_magnetic_field_cs() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/water.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(1.0, 0.0, 0.0)));
    test_ur_ordinary_group(&mol, thresh, "Cs", 2, 2, true);
}

#[test]
fn test_ur_group_asymmetric_water_magnetic_field_cs_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/water.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(1.0, 0.0, 0.0)));
    test_ur_ordinary_group_class_order(&mol, thresh, &["|E|", "|σh|"]);
}

#[test]
fn test_ur_group_asymmetric_water_magnetic_field_bw_c2v_cs() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/water.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(1.0, 0.0, 0.0)));
    test_ur_magnetic_group(&mol, thresh, "C2v", 4, 4, true, BWGRP);
}

#[test]
fn test_ur_group_asymmetric_water_magnetic_field_bw_c2v_cs_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/water.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(1.0, 0.0, 0.0)));
    test_ur_magnetic_group_class_order(&mol, thresh, &["|E|", "|σh|", "|θ·C2|", "|θ·σv|"]);
}

#[test]
fn test_ur_group_asymmetric_pyridine_magnetic_field_cs() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/pyridine.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 0.2)));
    test_ur_ordinary_group(&mol, thresh, "Cs", 2, 2, true);
}

#[test]
fn test_ur_group_asymmetric_pyridine_magnetic_field_bw_c2v_cs() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/pyridine.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 0.2)));
    test_ur_magnetic_group(&mol, thresh, "C2v", 4, 4, true, BWGRP);
}

#[test]
fn test_ur_group_asymmetric_pyridine_magnetic_field_bw_c2v_cs_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/pyridine.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 0.2)));
    test_ur_magnetic_group_class_order(&mol, thresh, &["|E|", "|σh|", "|θ·C2|", "|θ·σv|"]);
}

#[test]
fn test_ur_group_asymmetric_cyclobutene_magnetic_field_cs() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/cyclobutene.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.1, 0.0)));
    test_ur_ordinary_group(&mol, thresh, "Cs", 2, 2, true);
}

#[test]
fn test_ur_group_asymmetric_cyclobutene_magnetic_field_bw_c2v_cs() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/cyclobutene.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.1, 0.0)));
    test_ur_magnetic_group(&mol, thresh, "C2v", 4, 4, true, BWGRP);
}

#[test]
fn test_ur_group_asymmetric_cyclobutene_magnetic_field_bw_c2v_cs_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/cyclobutene.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.1, 0.0)));
    test_ur_magnetic_group_class_order(&mol, thresh, &["|E|", "|σh|", "|θ·C2|", "|θ·σv|"]);
}

#[test]
fn test_ur_group_asymmetric_azulene_magnetic_field_cs() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/azulene.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.1, 0.0)));
    test_ur_ordinary_group(&mol, thresh, "Cs", 2, 2, true);
}

#[test]
fn test_ur_group_asymmetric_azulene_magnetic_field_bw_c2v_cs() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/azulene.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.1, 0.0)));
    test_ur_magnetic_group(&mol, thresh, "C2v", 4, 4, true, BWGRP);
}

#[test]
fn test_ur_group_asymmetric_azulene_magnetic_field_bw_c2v_cs_class_order() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/azulene.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.1, 0.0)));
    test_ur_magnetic_group_class_order(&mol, thresh, &["|E|", "|σh|", "|θ·C2|", "|θ·σv|"]);
}

#[test]
fn test_ur_group_asymmetric_hf_magnetic_field_bw_c2v_cs() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/hf.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.1, 0.0, 0.0)));
    test_ur_magnetic_group(&mol, thresh, "C2v", 4, 4, true, BWGRP);
}

#[test]
fn test_ur_group_asymmetric_cis_cocl2h4o2_magnetic_field_cs() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/cis-cocl2h4o2.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 0.2)));
    test_ur_ordinary_group(&mol, thresh, "Cs", 2, 2, true);
}

#[test]
fn test_ur_group_asymmetric_cuneane_magnetic_field_cs() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/cuneane.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.5, 0.0)));
    test_ur_ordinary_group(&mol, thresh, "Cs", 2, 2, true);
}

#[test]
fn test_ur_group_asymmetric_cuneane_magnetic_field_bw_c2v_cs() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/cuneane.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.5, 0.0)));
    test_ur_magnetic_group(&mol, thresh, "C2v", 4, 4, true, BWGRP);
}

#[test]
fn test_ur_group_asymmetric_cuneane_magnetic_field_bw_c2v_cs_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/cuneane.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.5, 0.0)));
    test_ur_magnetic_group_class_order(&mol, thresh, &["|E|", "|σh|", "|θ·C2|", "|θ·σv|"]);
}

#[test]
fn test_ur_group_asymmetric_water_electric_field_cs() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/water.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(Vector3::new(1.0, 0.0, 0.0)));
    test_ur_ordinary_group(&mol, thresh, "Cs", 2, 2, true);
}

#[test]
fn test_ur_group_asymmetric_pyridine_electric_field_cs() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/pyridine.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(Vector3::new(0.0, 0.0, 0.2)));
    test_ur_ordinary_group(&mol, thresh, "Cs", 2, 2, true);
}

#[test]
fn test_ur_group_asymmetric_cyclobutene_electric_field_cs() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/cyclobutene.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(Vector3::new(0.0, 0.1, 0.0)));
    test_ur_ordinary_group(&mol, thresh, "Cs", 2, 2, true);
}

#[test]
fn test_ur_group_asymmetric_azulene_electric_field_cs() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/azulene.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(Vector3::new(0.0, 0.1, 0.0)));
    test_ur_ordinary_group(&mol, thresh, "Cs", 2, 2, true);
}

#[test]
fn test_ur_group_asymmetric_cis_cocl2h4o2_electric_field_cs() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/cis-cocl2h4o2.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(Vector3::new(0.0, 0.0, 0.2)));
    test_ur_ordinary_group(&mol, thresh, "Cs", 2, 2, true);
}

#[test]
fn test_ur_group_asymmetric_cuneane_electric_field_cs() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/cuneane.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(Vector3::new(0.0, 0.5, 0.0)));
    test_ur_ordinary_group(&mol, thresh, "Cs", 2, 2, true);
}

#[test]
fn test_ur_group_asymmetric_bf3_magnetic_field_cs() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/bf3.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.5, 0.0)));
    test_ur_ordinary_group(&mol, thresh, "Cs", 2, 2, true);
}

/// This is a special case: Cs point group in a symmetric top.
#[test]
fn test_ur_group_symmetric_ch4_magnetic_field_cs() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/ch4.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.1, 0.1, 0.0)));
    test_ur_ordinary_group(&mol, thresh, "Cs", 2, 2, true);
}

#[test]
fn test_ur_group_symmetric_ch4_magnetic_field_cs_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/ch4.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.1, 0.1, 0.0)));
    test_ur_ordinary_group_class_order(&mol, thresh, &["|E|", "|σh|"]);
}

#[test]
fn test_ur_group_symmetric_ch4_magnetic_field_bw_c2v_cs() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/ch4.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.1, 0.1, 0.0)));
    test_ur_magnetic_group(&mol, thresh, "C2v", 4, 4, true, BWGRP);
}

#[test]
fn test_ur_group_symmetric_ch4_magnetic_field_bw_c2v_cs_class_order() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/ch4.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.1, 0.1, 0.0)));
    test_ur_magnetic_group_class_order(&mol, thresh, &["|E|", "|σh|", "|θ·C2|", "|θ·σv|"]);
}

/// This is another special case: Cs point group in a symmetric top.
#[test]
fn test_ur_group_symmetric_ch4_electric_field_cs() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/ch4.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(Vector3::new(0.1, 0.1, 0.0)));
    test_ur_ordinary_group(&mol, thresh, "Cs", 2, 2, true);
}

#[test]
fn test_ur_group_asymmetric_atom_magnetic_electric_field_cs() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/th.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 0.2)));
    mol.set_electric_field(Some(Vector3::new(0.0, 0.1, 0.0)));
    test_ur_ordinary_group(&mol, thresh, "Cs", 2, 2, true);
}

/***
Cs*
***/

#[test]
fn test_ur_group_asymmetric_propene_cs_double() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/propene.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_double_group(&mol, thresh, "Cs*", 4, 4, true);
}

#[test]
fn test_ur_group_asymmetric_propene_cs_double_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/propene.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_double_group_class_order(
        &mol,
        thresh,
        &["|E(Σ)|", "|E(QΣ)|", "|σh(Σ)|", "|σh(QΣ)|"],
    );
}

#[test]
fn test_ur_group_asymmetric_propene_grey_cs_double() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/propene.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_magnetic_double_group(&mol, thresh, "(Cs + θ·Cs)*", 8, 8, true, GRGRP2);
}

#[test]
fn test_ur_group_asymmetric_propene_grey_cs_double_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/propene.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_magnetic_double_group_class_order(
        &mol,
        thresh,
        &[
            "|E(Σ)|",
            "|E(QΣ)|",
            "|σh(Σ)|",
            "|σh(QΣ)|",
            "|θ(Σ)|",
            "|θ(QΣ)|",
            "|θ·σh(Σ)|",
            "|θ·σh(QΣ)|",
        ],
    );
}

/*
D2
*/

#[test]
fn test_ur_group_asymmetric_i4_biphenyl_d2() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/i4-biphenyl.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_group(&mol, thresh, "D2", 4, 4, true);
}

#[test]
fn test_ur_group_asymmetric_i4_biphenyl_d2_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/i4-biphenyl.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_group_class_order(&mol, thresh, &["|E|", "|C2|", "|C2|^(')", "|C2|^('')"]);
}

#[test]
fn test_ur_group_asymmetric_i4_biphenyl_grey_d2() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/i4-biphenyl.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_magnetic_group(&mol, thresh, "D2 + θ·D2", 8, 8, true, GRGRP);
}

#[test]
fn test_ur_group_asymmetric_i4_biphenyl_grey_d2_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/i4-biphenyl.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_magnetic_group_class_order(
        &mol,
        thresh,
        &[
            "|E|",
            "|C2|",
            "|C2|^(')",
            "|C2|^('')",
            "|θ|",
            "|θ·C2|",
            "|θ·C2|^(')",
            "|θ·C2|^('')",
        ],
    );
}

#[test]
fn test_ur_group_asymmetric_twistane_d2() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/twistane.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_group(&mol, thresh, "D2", 4, 4, true);
}

#[test]
fn test_ur_group_asymmetric_22_paracyclophane_d2() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/paracyclophane22.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_group(&mol, thresh, "D2", 4, 4, true);
}

/*
D2*
*/

#[test]
fn test_ur_group_asymmetric_i4_biphenyl_d2_double() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/i4-biphenyl.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_double_group(&mol, thresh, "D2*", 8, 5, false);
}

#[test]
fn test_ur_group_asymmetric_i4_biphenyl_d2_double_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/i4-biphenyl.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_double_group_class_order(
        &mol,
        thresh,
        &[
            "|E(Σ)|",
            "|E(QΣ)|",
            "|C2(Σ), C2(QΣ)|",
            "|C2(Σ), C2(QΣ)|^(')",
            "|C2(Σ), C2(QΣ)|^('')",
        ],
    );
}

#[test]
fn test_ur_group_asymmetric_i4_biphenyl_grey_d2_double() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/i4-biphenyl.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_magnetic_double_group(&mol, thresh, "(D2 + θ·D2)*", 16, 10, false, GRGRP2);
}

#[test]
fn test_ur_group_asymmetric_i4_biphenyl_grey_d2_double_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/i4-biphenyl.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_magnetic_double_group_class_order(
        &mol,
        thresh,
        &[
            "|E(Σ)|",
            "|E(QΣ)|",
            "|C2(Σ), C2(QΣ)|",
            "|C2(Σ), C2(QΣ)|^(')",
            "|C2(Σ), C2(QΣ)|^('')",
            "|θ(Σ)|",
            "|θ(QΣ)|",
            "|θ·C2(Σ), θ·C2(QΣ)|",
            "|θ·C2(Σ), θ·C2(QΣ)|^(')",
            "|θ·C2(Σ), θ·C2(QΣ)|^('')",
        ],
    );
}

/***
D2h
***/

#[test]
fn test_ur_group_asymmetric_b2h6_d2h() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/b2h6.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_group(&mol, thresh, "D2h", 8, 8, true);
}

#[test]
fn test_ur_group_asymmetric_b2h6_d2h_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/b2h6.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_group_class_order(
        &mol,
        thresh,
        &[
            "|E|",
            "|C2|",
            "|C2|^(')",
            "|C2|^('')",
            "|i|",
            "|σ|",
            "|σ|^(')",
            "|σ|^('')",
        ],
    );
}

#[test]
fn test_ur_group_asymmetric_b2h6_grey_d2h() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/b2h6.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_magnetic_group(&mol, thresh, "D2h + θ·D2h", 16, 16, true, GRGRP);
}

#[test]
fn test_ur_group_asymmetric_b2h6_grey_d2h_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/b2h6.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_magnetic_group_class_order(
        &mol,
        thresh,
        &[
            "|E|",
            "|C2|",
            "|C2|^(')",
            "|C2|^('')",
            "|i|",
            "|σ|",
            "|σ|^(')",
            "|σ|^('')",
            "|θ|",
            "|θ·C2|",
            "|θ·C2|^(')",
            "|θ·C2|^('')",
            "|θ·i|",
            "|θ·σ|",
            "|θ·σ|^(')",
            "|θ·σ|^('')",
        ],
    );
}

#[test]
fn test_ur_group_asymmetric_naphthalene_d2h() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/naphthalene.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_group(&mol, thresh, "D2h", 8, 8, true);
}

#[test]
fn test_ur_group_asymmetric_naphthalene_d2h_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/naphthalene.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_group_class_order(
        &mol,
        thresh,
        &[
            "|E|",
            "|C2|",
            "|C2|^(')",
            "|C2|^('')",
            "|i|",
            "|σ|",
            "|σ|^(')",
            "|σ|^('')",
        ],
    );
}

#[test]
fn test_ur_group_asymmetric_naphthalene_grey_d2h() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/naphthalene.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_magnetic_group(&mol, thresh, "D2h + θ·D2h", 16, 16, true, GRGRP);
}

#[test]
fn test_ur_group_asymmetric_naphthalene_grey_d2h_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/naphthalene.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_magnetic_group_class_order(
        &mol,
        thresh,
        &[
            "|E|",
            "|C2|",
            "|C2|^(')",
            "|C2|^('')",
            "|i|",
            "|σ|",
            "|σ|^(')",
            "|σ|^('')",
            "|θ|",
            "|θ·C2|",
            "|θ·C2|^(')",
            "|θ·C2|^('')",
            "|θ·i|",
            "|θ·σ|",
            "|θ·σ|^(')",
            "|θ·σ|^('')",
        ],
    );
}

#[test]
fn test_ur_group_asymmetric_pyrene_d2h() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/pyrene.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_group(&mol, thresh, "D2h", 8, 8, true);
}

#[test]
fn test_ur_group_asymmetric_pyrene_d2h_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/pyrene.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_group_class_order(
        &mol,
        thresh,
        &[
            "|E|",
            "|C2|",
            "|C2|^(')",
            "|C2|^('')",
            "|i|",
            "|σ|",
            "|σ|^(')",
            "|σ|^('')",
        ],
    );
}

#[test]
fn test_ur_group_asymmetric_pyrene_grey_d2h() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/pyrene.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_magnetic_group(&mol, thresh, "D2h + θ·D2h", 16, 16, true, GRGRP);
}

#[test]
fn test_ur_group_asymmetric_pyrene_grey_d2h_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/pyrene.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_magnetic_group_class_order(
        &mol,
        thresh,
        &[
            "|E|",
            "|C2|",
            "|C2|^(')",
            "|C2|^('')",
            "|i|",
            "|σ|",
            "|σ|^(')",
            "|σ|^('')",
            "|θ|",
            "|θ·C2|",
            "|θ·C2|^(')",
            "|θ·C2|^('')",
            "|θ·i|",
            "|θ·σ|",
            "|θ·σ|^(')",
            "|θ·σ|^('')",
        ],
    );
}

#[test]
fn test_ur_group_asymmetric_c6o6_d2h() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c6o6.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_group(&mol, thresh, "D2h", 8, 8, true);
}

#[test]
fn test_ur_group_asymmetric_distorted_vf6_d2h() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/vf6_d2h.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_group(&mol, thresh, "D2h", 8, 8, true);
}

/***
D2h*
***/

#[test]
fn test_ur_group_asymmetric_b2h6_d2h_double() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/b2h6.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_double_group(&mol, thresh, "D2h*", 16, 10, false);
}

#[test]
fn test_ur_group_asymmetric_b2h6_d2h_double_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/b2h6.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_double_group_class_order(
        &mol,
        thresh,
        &[
            "|E(Σ)|",
            "|E(QΣ)|",
            "|C2(Σ), C2(QΣ)|",
            "|C2(Σ), C2(QΣ)|^(')",
            "|C2(Σ), C2(QΣ)|^('')",
            "|i(Σ)|",
            "|i(QΣ)|",
            "|σ(Σ), σ(QΣ)|",
            "|σ(Σ), σ(QΣ)|^(')",
            "|σ(Σ), σ(QΣ)|^('')",
        ],
    );
}

#[test]
fn test_ur_group_asymmetric_pyrene_grey_d2h_double() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/pyrene.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_magnetic_double_group(&mol, thresh, "(D2h + θ·D2h)*", 32, 20, false, GRGRP2);
}

#[test]
fn test_ur_group_asymmetric_pyrene_grey_d2h_double_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/pyrene.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_magnetic_double_group_class_order(
        &mol,
        thresh,
        &[
            "|E(Σ)|",
            "|E(QΣ)|",
            "|C2(Σ), C2(QΣ)|",
            "|C2(Σ), C2(QΣ)|^(')",
            "|C2(Σ), C2(QΣ)|^('')",
            "|i(Σ)|",
            "|i(QΣ)|",
            "|σ(Σ), σ(QΣ)|",
            "|σ(Σ), σ(QΣ)|^(')",
            "|σ(Σ), σ(QΣ)|^('')",
            "|θ(Σ)|",
            "|θ(QΣ)|",
            "|θ·C2(Σ), θ·C2(QΣ)|",
            "|θ·C2(Σ), θ·C2(QΣ)|^(')",
            "|θ·C2(Σ), θ·C2(QΣ)|^('')",
            "|θ·i(Σ)|",
            "|θ·i(QΣ)|",
            "|θ·σ(Σ), θ·σ(QΣ)|",
            "|θ·σ(Σ), θ·σ(QΣ)|^(')",
            "|θ·σ(Σ), θ·σ(QΣ)|^('')",
        ],
    );
}

/***
Ci
***/

#[test]
fn test_ur_group_asymmetric_meso_tartaricacid_ci() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/meso-tartaricacid.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_group(&mol, thresh, "Ci", 2, 2, true);
}

#[test]
fn test_ur_group_asymmetric_meso_tartaricacid_ci_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/meso-tartaricacid.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_group_class_order(&mol, thresh, &["|E|", "|i|"]);
}

#[test]
fn test_ur_group_asymmetric_meso_tartaricacid_grey_ci() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/meso-tartaricacid.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_magnetic_group(&mol, thresh, "Ci + θ·Ci", 4, 4, true, GRGRP);
}

#[test]
fn test_ur_group_asymmetric_meso_tartaricacid_grey_ci_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/meso-tartaricacid.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_magnetic_group_class_order(&mol, thresh, &["|E|", "|i|", "|θ|", "|θ·i|"]);
}

#[test]
fn test_ur_group_asymmetric_dibromodimethylcyclohexane_ci() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/dibromodimethylcyclohexane.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_group(&mol, thresh, "Ci", 2, 2, true);
}

#[test]
fn test_ur_group_asymmetric_h2o2_magnetic_field_ci() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/h2o2.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(1.0, 2.0, -1.0)));
    test_ur_ordinary_group(&mol, thresh, "Ci", 2, 2, true);
}

#[test]
fn test_ur_group_asymmetric_h2o2_magnetic_field_ci_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/h2o2.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(1.0, 2.0, -1.0)));
    test_ur_ordinary_group_class_order(&mol, thresh, &["|E|", "|i|"]);
}

#[test]
fn test_ur_group_asymmetric_h2o2_magnetic_field_bw_c2h_ci() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/h2o2_yz.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 2.0, -1.0)));
    test_ur_magnetic_group(&mol, thresh, "C2h", 4, 4, true, BWGRP);
}

#[test]
fn test_ur_group_asymmetric_h2o2_magnetic_field_bw_c2h_ci_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/h2o2_yz.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 2.0, -1.0)));
    test_ur_magnetic_group_class_order(&mol, thresh, &["|E|", "|i|", "|θ·C2|", "|θ·σh|"]);
}

#[test]
fn test_ur_group_symmetric_xef4_magnetic_field_ci() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/xef4.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(1.0, 2.0, -2.0)));
    test_ur_ordinary_group(&mol, thresh, "Ci", 2, 2, true);
}

#[test]
fn test_ur_group_asymmetric_c2h2_magnetic_field_ci() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c2h2.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(1.0, -1.0, 1.0)));
    test_ur_ordinary_group(&mol, thresh, "Ci", 2, 2, true);
}

/// This is a special case: Ci from S2 via symmetric top.
#[test]
fn test_ur_group_symmetric_vf6_magnetic_field_ci() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/vf6.xyz");
    let thresh = 1e-12;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(1.0, -2.0, 3.0)));
    test_ur_ordinary_group(&mol, thresh, "Ci", 2, 2, true);
}

/// This is a special case: Ci from S2 via symmetric top.
#[test]
fn test_ur_group_symmetric_c60_magnetic_field_ci() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c60.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(1.0, -2.0, 3.0)));
    test_ur_ordinary_group(&mol, thresh, "Ci", 2, 2, true);
}

/***
Ci*
***/

#[test]
fn test_ur_group_asymmetric_meso_tartaricacid_ci_double() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/meso-tartaricacid.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_double_group(&mol, thresh, "Ci*", 4, 4, true);
}

#[test]
fn test_ur_group_asymmetric_meso_tartaricacid_ci_double_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/meso-tartaricacid.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_double_group_class_order(
        &mol,
        thresh,
        &["|E(Σ)|", "|E(QΣ)|", "|i(Σ)|", "|i(QΣ)|"],
    );
}

#[test]
fn test_ur_group_asymmetric_meso_tartaricacid_grey_ci_double() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/meso-tartaricacid.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_magnetic_double_group(&mol, thresh, "(Ci + θ·Ci)*", 8, 8, true, GRGRP2);
}

#[test]
fn test_ur_group_asymmetric_meso_tartaricacid_grey_ci_double_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/meso-tartaricacid.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_magnetic_double_group_class_order(
        &mol,
        thresh,
        &[
            "|E(Σ)|",
            "|E(QΣ)|",
            "|i(Σ)|",
            "|i(QΣ)|",
            "|θ(Σ)|",
            "|θ(QΣ)|",
            "|θ·i(Σ)|",
            "|θ·i(QΣ)|",
        ],
    );
}

/***
C1
***/

#[test]
fn test_ur_group_asymmetric_butan1ol_c1() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/butan-1-ol.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_group(&mol, thresh, "C1", 1, 1, true);
}

#[test]
fn test_ur_group_asymmetric_butan1ol_c1_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/butan-1-ol.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_group_class_order(&mol, thresh, &["|E|"]);
}

#[test]
fn test_ur_group_asymmetric_butan1ol_grey_c1() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/butan-1-ol.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_magnetic_group(&mol, thresh, "C1 + θ·C1", 2, 2, true, GRGRP);
}

#[test]
fn test_ur_group_asymmetric_butan1ol_grey_c1_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/butan-1-ol.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_magnetic_group_class_order(&mol, thresh, &["|E|", "|θ|"]);
}

#[test]
fn test_ur_group_asymmetric_subst_5m_ring_c1() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/subst-5m-ring.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_group(&mol, thresh, "C1", 1, 1, true);
}

#[test]
fn test_ur_group_asymmetric_bf3_magnetic_field_c1() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/bf3.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(1.0, -1.0, 1.0)));
    test_ur_ordinary_group(&mol, thresh, "C1", 1, 1, true);
}

/// This is a special case: C1 via symmetric top.
#[test]
fn test_ur_group_symmetric_ch4_magnetic_field_c1() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/ch4.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(1.0, -3.0, 2.0)));
    test_ur_ordinary_group(&mol, thresh, "C1", 1, 1, true);
}

/// This is a special case: C1 via symmetric top.
#[test]
fn test_ur_group_symmetric_vf6_electric_field_c1() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/vf6.xyz");
    let thresh = 1e-12;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(Vector3::new(1.0, -2.0, 3.0)));
    test_ur_ordinary_group(&mol, thresh, "C1", 1, 1, true);
}

/// This is a special case: C1 via symmetric top.
#[test]
fn test_ur_group_symmetric_c60_electric_field_c1() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c60.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(Vector3::new(1.0, -2.0, 3.0)));
    test_ur_ordinary_group(&mol, thresh, "C1", 1, 1, true);
}

/***
C1*
***/

#[test]
fn test_ur_group_asymmetric_butan1ol_c1_double() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/butan-1-ol.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_double_group(&mol, thresh, "C1*", 2, 2, true);
}

#[test]
fn test_ur_group_asymmetric_butan1ol_c1_double_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/butan-1-ol.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_ordinary_double_group_class_order(&mol, thresh, &["|E(Σ)|", "|E(QΣ)|"]);
}

#[test]
fn test_ur_group_asymmetric_butan1ol_grey_c1_double() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/butan-1-ol.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_magnetic_double_group(&mol, thresh, "(C1 + θ·C1)*", 4, 4, true, GRGRP2);
}

#[test]
fn test_ur_group_asymmetric_butan1ol_grey_c1_double_class_order() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/butan-1-ol.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_ur_magnetic_double_group_class_order(
        &mol,
        thresh,
        &["|E(Σ)|", "|E(QΣ)|", "|θ(Σ)|", "|θ(QΣ)|"],
    );
}
