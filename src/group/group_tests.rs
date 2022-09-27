use std::collections::HashMap;

use nalgebra::Vector3;
use num_traits::Pow;

use crate::aux::molecule::Molecule;
use crate::aux::template_molecules;
use crate::group::{group_from_molecular_symmetry, Group};
use crate::symmetry::symmetry_core::{PreSymmetry, Symmetry};
use crate::symmetry::symmetry_element::{SymmetryElement, SymmetryElementKind, SymmetryOperation};
use crate::symmetry::symmetry_element_order::ElementOrder;

const ROOT: &str = env!("CARGO_MANIFEST_DIR");

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
    assert!(group_c5.is_abelian());

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
    assert!(group_c29.is_abelian());
}

#[test]
fn test_group_from_molecular_symmetry() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/nh3.xyz");
    let mol = Molecule::from_xyz(&path, 1e-6);
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-6)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::builder().build().unwrap();
    sym.analyse(&presym);
    let group = group_from_molecular_symmetry(sym, None, None);
    assert_eq!(group.name, "C3v".to_string());
    assert_eq!(group.order, 6);
    assert_eq!(group.class_number, Some(3));
}

fn test_abstract_group(
    mol: Molecule,
    thresh: f64,
    name: &str,
    order: usize,
    class_number: usize,
    abelian: bool,
) {
    let presym = PreSymmetry::builder()
        .moi_threshold(thresh)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::builder().build().unwrap();
    sym.analyse(&presym);
    let group = group_from_molecular_symmetry(sym, None, None);
    assert_eq!(group.name, name.to_string());
    assert_eq!(group.order, order);
    assert_eq!(group.class_number, Some(class_number));
    assert_eq!(group.is_abelian(), abelian);
}

/********
Spherical
********/

// #[test]
// fn test_abstract_group_spherical_atom_o3() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/th.xyz");
//     let mol = Molecule::from_xyz(&path, 1e-6);
//     let presym = PreSymmetry::builder()
//         .moi_threshold(1e-14)
//         .molecule(&mol, true)
//         .build()
//         .unwrap();
//     let mut sym = Symmetry::builder().build().unwrap();
//     sym.analyse(&presym);
//     assert_eq!(sym.point_group, Some("O(3)".to_owned()));
//     assert_eq!(sym.proper_generators[&ElementOrder::Inf].len(), 3);
// }

#[test]
fn test_abstract_group_spherical_c60_ih() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c60.xyz");
    let thresh = 1e-5;
    let mol = Molecule::from_xyz(&path, thresh);
    test_abstract_group(mol, thresh, "Ih", 120, 10, false);
}

#[test]
fn test_abstract_group_spherical_ch4_td() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/ch4.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    test_abstract_group(mol, thresh, "Td", 24, 5, false);
}

#[test]
fn test_abstract_group_spherical_adamantane_td() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/adamantane.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    test_abstract_group(mol, thresh, "Td", 24, 5, false);
}

#[test]
fn test_abstract_group_spherical_c165_diamond_nanoparticle_td() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c165.xyz");
    let thresh = 1e-5;
    let mol = Molecule::from_xyz(&path, thresh);
    test_abstract_group(mol, thresh, "Td", 24, 5, false);
}

#[test]
fn test_abstract_group_spherical_vh2o6_th() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/vh2o6.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    test_abstract_group(mol, thresh, "Th", 24, 8, false);
}

#[test]
fn test_abstract_group_spherical_vf6_oh() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/vf6.xyz");
    let thresh = 1e-12;
    let mol = Molecule::from_xyz(&path, thresh);
    test_abstract_group(mol, thresh, "Oh", 48, 10, false);
}

/********
Symmetric
********/

/*
Cn
*/

#[test]
fn test_abstract_group_symmetric_ch4_magnetic_field_c3() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/ch4.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(1.0, 1.0, 1.0)));
    test_abstract_group(mol, thresh, "C3", 3, 3, true);
}

#[test]
fn test_abstract_group_symmetric_adamantane_magnetic_field_c3() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/adamantane.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.1, 0.1, 0.1)));
    test_abstract_group(mol, thresh, "C3", 3, 3, true);
}

#[test]
fn test_abstract_group_symmetric_vh2o6_electric_field_c3() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/vh2o6.xyz");
    let thresh = 1e-12;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(Vector3::new(-0.2, -0.2, -0.2)));
    test_abstract_group(mol, thresh, "C3", 3, 3, true);
}

#[test]
fn test_abstract_group_symmetric_65coronane_electric_field_c3() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/coronane65.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(Vector3::new(0.0, 0.0, -1.0)));
    test_abstract_group(mol, thresh, "C3", 3, 3, true);
}

#[test]
fn test_abstract_group_symmetric_h8_twisted_magnetic_field_c4() {
    // env_logger::init();
    let thresh = 1e-7;
    let mut mol = template_molecules::gen_twisted_h8(0.1);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 0.1)));
    test_abstract_group(mol, thresh, "C4", 4, 4, true);
}

#[test]
fn test_abstract_group_symmetric_h8_twisted_electric_field_c4() {
    let thresh = 1e-7;
    let mut mol = template_molecules::gen_twisted_h8(0.1);
    mol.set_electric_field(Some(Vector3::new(0.0, 0.0, -0.1)));
    test_abstract_group(mol, thresh, "C4", 4, 4, true);
}

#[test]
fn test_abstract_group_symmetric_cpnico_magnetic_field_c5() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/cpnico.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, -0.2)));
    test_abstract_group(mol, thresh, "C5", 5, 5, true);
}

#[test]
fn test_abstract_group_symmetric_b7_magnetic_field_c6() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/b7.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 0.1)));
    test_abstract_group(mol, thresh, "C6", 6, 6, true);
}

#[test]
fn test_abstract_group_symmetric_arbitrary_half_sandwich_magnetic_field_cn() {
    let thresh = 1e-7;
    for n in 3..=32 {
        let mut mol = template_molecules::gen_arbitrary_half_sandwich(n);
        mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 0.1)));
        test_abstract_group(
            mol,
            thresh,
            format!("C{}", n).as_str(),
            n as usize,
            n as usize,
            true,
        );
    }
}

/*
Cnv
*/

#[test]
fn test_abstract_group_symmetric_nh3_c3v() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/nh3.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    test_abstract_group(mol, thresh, "C3v", 6, 3, false);
}

#[test]
fn test_abstract_group_symmetric_bf3_electric_field_c3v() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/bf3.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    test_abstract_group(mol, thresh, "C3v", 6, 3, false);
}

#[test]
fn test_abstract_group_symmetric_adamantane_electric_field_c3v() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/adamantane.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(Vector3::new(0.1, 0.1, 0.1)));
    test_abstract_group(mol, thresh, "C3v", 6, 3, false);
}

#[test]
fn test_abstract_group_symmetric_ch4_electric_field_c3v() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/ch4.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(Vector3::new(1.0, 1.0, 1.0)));
    test_abstract_group(mol, thresh, "C3v", 6, 3, false);
}

#[test]
fn test_abstract_group_symmetric_vf6_electric_field_c3v() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/vf6.xyz");
    let thresh = 1e-12;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(0.2 * Vector3::new(1.0, 1.0, 1.0)));
    test_abstract_group(mol, thresh, "C3v", 6, 3, false);
}

#[test]
fn test_abstract_group_symmetric_sf5cl_c4v() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/sf5cl.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_abstract_group(mol, thresh, "C4v", 8, 5, false);
}

#[test]
fn test_abstract_group_symmetric_h8_electric_field_c4v() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/h8.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(0.2 * Vector3::new(0.0, 0.0, 1.0)));
    test_abstract_group(mol, thresh, "C4v", 8, 5, false);
}

#[test]
fn test_abstract_group_symmetric_vf6_electric_field_c4v() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/vf6.xyz");
    let thresh = 1e-12;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(0.2 * Vector3::new(1.0, 0.0, 0.0)));
    test_abstract_group(mol, thresh, "C4v", 8, 5, false);
}

#[test]
fn test_abstract_group_symmetric_antiprism_pb10_electric_field_c4v() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/pb10.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    test_abstract_group(mol, thresh, "C4v", 8, 5, false);
}

#[test]
fn test_abstract_group_symmetric_cpnico_c5v() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/cpnico.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    test_abstract_group(mol, thresh, "C5v", 10, 4, false);
}

#[test]
fn test_abstract_group_symmetric_staggered_ferrocene_electric_field_c5v() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/staggered_ferrocene.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    test_abstract_group(mol, thresh, "C5v", 10, 4, false);
}

#[test]
fn test_abstract_group_symmetric_c60_electric_field_c5v() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c60.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    test_abstract_group(mol, thresh, "C5v", 10, 4, false);
}

#[test]
fn test_abstract_group_symmetric_b7_c6v() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/b7.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    test_abstract_group(mol, thresh, "C6v", 12, 6, false);
}

#[test]
fn test_abstract_group_symmetric_au26_electric_field_c6v() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/au26.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    test_abstract_group(mol, thresh, "C6v", 12, 6, false);
}

#[test]
fn test_abstract_group_symmetric_benzene_electric_field_c6v() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/benzene.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(Vector3::new(1.0, 0.0, 0.0)));
    test_abstract_group(mol, thresh, "C6v", 12, 6, false);
}

#[test]
fn test_abstract_group_symmetric_arbitrary_half_sandwich_cnv() {
    /* The expected number of classes is deduced from the irrep structures of
     * the Cnv groups.
     * When n is even, the irreps are A1, A2, B1, B2, Ek where k = 1, ..., n/2 - 1.
     * When n is odd, the irreps are A1, A2, Ek where k = 1, ..., n//2.
     */
    for n in 3..=32 {
        let mol = template_molecules::gen_arbitrary_half_sandwich(n);
        let thresh = 1e-7;
        test_abstract_group(
            mol,
            thresh,
            format!("C{}v", n).as_str(),
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
fn test_abstract_group_symmetric_arbitrary_staggered_sandwich_electric_field_cnv() {
    /* The expected number of classes is deduced from the irrep structures of
     * the Cnv groups.
     * When n is even, the irreps are A1, A2, B1, B2, Ek where k = 1, ..., n/2 - 1.
     * When n is odd, the irreps are A1, A2, Ek where k = 1, ..., n//2.
     */
    for n in 3..=20 {
        let mut mol = template_molecules::gen_arbitrary_twisted_sandwich(n, 0.5);
        mol.set_electric_field(Some(Vector3::new(0.0, 0.0, 1.0)));
        let thresh = 1e-7;
        test_abstract_group(
            mol,
            thresh,
            format!("C{}v", n).as_str(),
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

/*
Cnh
*/

#[test]
fn test_abstract_group_symmetric_bf3_magnetic_field_c3h() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/bf3.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    test_abstract_group(mol, thresh, "C3h", 6, 6, true);
}

#[test]
fn test_abstract_group_symmetric_xef4_magnetic_field_c4h() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/xef4.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, -1.0)));
    test_abstract_group(mol, thresh, "C4h", 8, 8, true);
}

#[test]
fn test_abstract_group_symmetric_vf6_magnetic_field_c4h() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/vf6.xyz");
    let thresh = 1e-12;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    test_abstract_group(mol, thresh, "C4h", 8, 8, true);
}

#[test]
fn test_abstract_group_symmetric_h8_magnetic_field_c4h() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/h8.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, 1e-7);
    mol.set_magnetic_field(Some(0.2 * Vector3::new(0.0, 0.0, 1.0)));
    test_abstract_group(mol, thresh, "C4h", 8, 8, true);
}

#[test]
fn test_abstract_group_symmetric_eclipsed_ferrocene_magnetic_field_c5h() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/eclipsed_ferrocene.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, -1.0)));
    test_abstract_group(mol, thresh, "C5h", 10, 10, true);
}

#[test]
fn test_abstract_group_symmetric_benzene_magnetic_field_c6h() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/benzene.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(1.0, 0.0, 0.0)));
    test_abstract_group(mol, thresh, "C6h", 12, 12, true);
}

#[test]
fn test_abstract_group_symmetric_arbitrary_eclipsed_sandwich_magnetic_field_cnh() {
    // env_logger::init();
    for n in 3..=20 {
        let mut mol = template_molecules::gen_arbitrary_eclipsed_sandwich(n);
        let thresh = 1e-7;
        mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 0.1)));
        test_abstract_group(
            mol,
            thresh,
            format!("C{}h", n).as_str(),
            2 * n as usize,
            2 * n as usize,
            true,
        );
    }
}
