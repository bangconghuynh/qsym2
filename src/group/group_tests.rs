use env_logger;
use itertools::Itertools;
use nalgebra::Vector3;
use num_traits::Pow;
use std::panic;

use crate::aux::molecule::Molecule;
use crate::aux::template_molecules;
use crate::group::{group_from_molecular_symmetry, Group};
use crate::symmetry::symmetry_core::{PreSymmetry, Symmetry};
use crate::symmetry::symmetry_element::{SymmetryElement, SymmetryElementKind, SymmetryOperation};
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
fn test_abstract_group_from_molecular_symmetry() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/nh3.xyz");
    let mol = Molecule::from_xyz(&path, 1e-6);
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-6)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::builder().build().unwrap();
    sym.analyse(&presym);
    let group = group_from_molecular_symmetry(sym, None);
    assert_eq!(group.name, "C3v".to_string());
    assert_eq!(group.order, 6);
    assert_eq!(group.class_number, Some(3));
}

#[test]
fn test_abstract_group_element_to_conjugacy_class() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/cpnico.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    let presym = PreSymmetry::builder()
        .moi_threshold(thresh)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::builder().build().unwrap();
    sym.analyse(&presym);
    let group = group_from_molecular_symmetry(sym, None);
    assert_eq!(group.name, "C5v".to_string());
    assert_eq!(group.order, 10);
    assert_eq!(group.class_number, Some(4));

    let conjugacy_classes = group.conjugacy_classes.unwrap();
    for (element_i, class_i) in group
        .element_to_conjugacy_classes
        .unwrap()
        .iter()
        .enumerate()
    {
        assert!(conjugacy_classes[*class_i].contains(&group.elements[element_i]));
    }
}

fn test_abstract_group(
    mol: &Molecule,
    thresh: f64,
    name: &str,
    order: usize,
    class_number: usize,
    abelian: bool,
) {
    let presym = PreSymmetry::builder()
        .moi_threshold(thresh)
        .molecule(mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::builder().build().unwrap();
    sym.analyse(&presym);
    let group = group_from_molecular_symmetry(sym, None);
    assert_eq!(group.name, name.to_string());
    assert_eq!(group.order, order);
    assert_eq!(group.class_number, Some(class_number));
    assert_eq!(group.is_abelian(), abelian);

    // Test element to conjugacy class
    let conjugacy_classes = group.conjugacy_classes.unwrap();
    for (element_i, class_i) in group
        .element_to_conjugacy_classes
        .unwrap()
        .iter()
        .enumerate()
    {
        assert!(conjugacy_classes[*class_i].contains(&group.elements[element_i]));
    }

    // Test inverse conjugacy classes
    let ctb = group.cayley_table.as_ref().unwrap();
    for (class_i, inv_class_i) in group
        .inverse_conjugacy_classes
        .as_ref()
        .unwrap()
        .iter()
        .enumerate()
    {
        assert!(conjugacy_classes[class_i]
            .iter()
            .cartesian_product(conjugacy_classes[*inv_class_i].iter())
            .filter(|(&g, &inv_g)| { ctb[[g, inv_g]] == 0 })
            .collect::<Vec<_>>()
            .len() == conjugacy_classes[class_i].len()
        );
    }
}

fn test_abstract_group_from_infinite_group(
    mol: &Molecule,
    finite_order: u32,
    thresh: f64,
    name: &str,
    finite_name: &str,
    order: usize,
    class_number: usize,
    abelian: bool,
) {
    let presym = PreSymmetry::builder()
        .moi_threshold(thresh)
        .molecule(mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::builder().build().unwrap();
    sym.analyse(&presym);
    let group = group_from_molecular_symmetry(sym, Some(finite_order));
    assert_eq!(group.name, name.to_string());
    assert_eq!(group.finite_subgroup_name, Some(finite_name.to_string()));
    assert_eq!(group.order, order);
    assert_eq!(group.class_number, Some(class_number));
    assert_eq!(group.is_abelian(), abelian);

    // Test element to conjugacy class
    let conjugacy_classes = group.conjugacy_classes.unwrap();
    for (element_i, class_i) in group
        .element_to_conjugacy_classes
        .unwrap()
        .iter()
        .enumerate()
    {
        assert!(conjugacy_classes[*class_i].contains(&group.elements[element_i]));
    }

    // Test inverse conjugacy classes
    let ctb = group.cayley_table.as_ref().unwrap();
    for (class_i, inv_class_i) in group
        .inverse_conjugacy_classes
        .as_ref()
        .unwrap()
        .iter()
        .enumerate()
    {
        assert!(conjugacy_classes[class_i]
            .iter()
            .cartesian_product(conjugacy_classes[*inv_class_i].iter())
            .filter(|(&g, &inv_g)| { ctb[[g, inv_g]] == 0 })
            .collect::<Vec<_>>()
            .len() == conjugacy_classes[class_i].len()
        );
    }
}

/********
Spherical
********/

#[test]
fn test_abstract_group_spherical_atom_o3() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/th.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);

    test_abstract_group_from_infinite_group(&mol, 2, thresh, "O(3)", "D2h", 8, 8, true);

    test_abstract_group_from_infinite_group(&mol, 4, thresh, "O(3)", "Oh", 48, 10, false);

    let result = panic::catch_unwind(|| {
        test_abstract_group_from_infinite_group(&mol, 5, thresh, "?", "?", 48, 10, false);
    });
    assert!(result.is_err());

    let result = panic::catch_unwind(|| {
        test_abstract_group_from_infinite_group(&mol, 3, thresh, "?", "?", 48, 10, false);
    });
    assert!(result.is_err());
}

#[test]
fn test_abstract_group_spherical_c60_ih() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c60.xyz");
    let thresh = 1e-5;
    let mol = Molecule::from_xyz(&path, thresh);
    test_abstract_group(&mol, thresh, "Ih", 120, 10, false);
}

#[test]
fn test_abstract_group_spherical_ch4_td() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/ch4.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    test_abstract_group(&mol, thresh, "Td", 24, 5, false);
}

#[test]
fn test_abstract_group_spherical_adamantane_td() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/adamantane.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    test_abstract_group(&mol, thresh, "Td", 24, 5, false);
}

#[test]
fn test_abstract_group_spherical_c165_diamond_nanoparticle_td() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c165.xyz");
    let thresh = 1e-5;
    let mol = Molecule::from_xyz(&path, thresh);
    test_abstract_group(&mol, thresh, "Td", 24, 5, false);
}

#[test]
fn test_abstract_group_spherical_vh2o6_th() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/vh2o6.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    test_abstract_group(&mol, thresh, "Th", 24, 8, false);
}

#[test]
fn test_abstract_group_spherical_vf6_oh() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/vf6.xyz");
    let thresh = 1e-12;
    let mol = Molecule::from_xyz(&path, thresh);
    test_abstract_group(&mol, thresh, "Oh", 48, 10, false);
}

/*****
Linear
*****/

#[test]
fn test_abstract_group_linear_atom_magnetic_field_cinfh() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/th.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(1.0, 2.0, -1.0)));
    for n in 2usize..=20usize {
        if n % 2 == 0 {
            test_abstract_group_from_infinite_group(
                &mol,
                n as u32,
                thresh,
                "C∞h",
                format!("C{}h", n).as_str(),
                2 * n,
                2 * n,
                true,
            );
        } else {
            test_abstract_group_from_infinite_group(
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
fn test_abstract_group_linear_atom_electric_field_cinfv() {
    /* The expected number of classes is deduced from the irrep structures of
     * the Cnv groups.
     * When n is even, the irreps are A1, A2, B1, B2, Ek where k = 1, ..., n/2 - 1.
     * When n is odd, the irreps are A1, A2, Ek where k = 1, ..., n//2.
     */
    let path: String = format!("{}{}", ROOT, "/tests/xyz/th.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(Vector3::new(-1.0, 3.0, -2.0)));
    for n in 3usize..=20usize {
        test_abstract_group_from_infinite_group(
            &mol,
            n as u32,
            thresh,
            "C∞v",
            format!("C{}v", n).as_str(),
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
            }) as usize,
            false,
        );
    }
}

#[test]
fn test_abstract_group_linear_c2h2_dinfh() {
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
    for n in 3usize..=20usize {
        if n % 2 == 0 {
            test_abstract_group_from_infinite_group(
                &mol,
                n as u32,
                thresh,
                "D∞h",
                format!("D{}h", n).as_str(),
                4 * n,
                2 * (n / 2 - 1 + 4) as usize,
                false,
            );
        } else {
            test_abstract_group_from_infinite_group(
                &mol,
                n as u32,
                thresh,
                "D∞h",
                format!("D{}h", 2 * n).as_str(),
                8 * n,
                2 * (n - 1 + 4) as usize,
                false,
            );
        }
    }
}

#[test]
fn test_abstract_group_linear_c2h2_magnetic_field_cinfh() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c2h2.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);
    // Parallel field
    mol.set_magnetic_field(Some(Vector3::new(1.0, 1.0, 1.0)));
    for n in 2usize..=20usize {
        if n % 2 == 0 {
            test_abstract_group_from_infinite_group(
                &mol,
                n as u32,
                thresh,
                "C∞h",
                format!("C{}h", n).as_str(),
                2 * n,
                2 * n as usize,
                true,
            );
        } else {
            test_abstract_group_from_infinite_group(
                &mol,
                n as u32,
                thresh,
                "C∞h",
                format!("C{}h", 2 * n).as_str(),
                4 * n,
                4 * n as usize,
                true,
            );
        }
    }
}

#[test]
fn test_abstract_group_linear_c2h2_electric_field_cinfv() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c2h2.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);

    // Parallel field
    mol.set_electric_field(Some(Vector3::new(1.0, 1.0, 1.0)));
    for n in 3usize..=20usize {
        test_abstract_group_from_infinite_group(
            &mol,
            n as u32,
            thresh,
            "C∞v",
            format!("C{}v", n).as_str(),
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
            }) as usize,
            false,
        );
    }
}

#[test]
fn test_abstract_group_linear_n3_cinfv() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/n3.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    for n in 3usize..=20usize {
        test_abstract_group_from_infinite_group(
            &mol,
            n as u32,
            thresh,
            "C∞v",
            format!("C{}v", n).as_str(),
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
            }) as usize,
            false,
        );
    }
}

#[test]
fn test_abstract_group_linear_n3_magnetic_field_cinf() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/n3.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);

    // Parallel field
    mol.set_magnetic_field(Some(Vector3::new(1.0, 1.0, 1.0)));
    for n in 2usize..=20usize {
        test_abstract_group_from_infinite_group(
            &mol,
            n as u32,
            thresh,
            "C∞",
            format!("C{}", n).as_str(),
            n,
            n,
            true,
        );
    }
}

#[test]
fn test_abstract_group_linear_n3_electric_field_cinfv() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/n3.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);

    // Parallel field
    mol.set_electric_field(Some(Vector3::new(1.0, 1.0, 1.0)));
    for n in 3usize..=20usize {
        test_abstract_group_from_infinite_group(
            &mol,
            n as u32,
            thresh,
            "C∞v",
            format!("C{}v", n).as_str(),
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
            }) as usize,
            false,
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
fn test_abstract_group_symmetric_ch4_magnetic_field_c3() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/ch4.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(1.0, 1.0, 1.0)));
    test_abstract_group(&mol, thresh, "C3", 3, 3, true);
}

#[test]
fn test_abstract_group_symmetric_adamantane_magnetic_field_c3() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/adamantane.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.1, 0.1, 0.1)));
    test_abstract_group(&mol, thresh, "C3", 3, 3, true);
}

#[test]
fn test_abstract_group_symmetric_vh2o6_electric_field_c3() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/vh2o6.xyz");
    let thresh = 1e-12;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(Vector3::new(-0.2, -0.2, -0.2)));
    test_abstract_group(&mol, thresh, "C3", 3, 3, true);
}

#[test]
fn test_abstract_group_symmetric_65coronane_electric_field_c3() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/coronane65.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(Vector3::new(0.0, 0.0, -1.0)));
    test_abstract_group(&mol, thresh, "C3", 3, 3, true);
}

#[test]
fn test_abstract_group_symmetric_h8_twisted_magnetic_field_c4() {
    // env_logger::init();
    let thresh = 1e-7;
    let mut mol = template_molecules::gen_twisted_h8(0.1);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 0.1)));
    test_abstract_group(&mol, thresh, "C4", 4, 4, true);
}

#[test]
fn test_abstract_group_symmetric_h8_twisted_electric_field_c4() {
    let thresh = 1e-7;
    let mut mol = template_molecules::gen_twisted_h8(0.1);
    mol.set_electric_field(Some(Vector3::new(0.0, 0.0, -0.1)));
    test_abstract_group(&mol, thresh, "C4", 4, 4, true);
}

#[test]
fn test_abstract_group_symmetric_cpnico_magnetic_field_c5() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/cpnico.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, -0.2)));
    test_abstract_group(&mol, thresh, "C5", 5, 5, true);
}

#[test]
fn test_abstract_group_symmetric_b7_magnetic_field_c6() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/b7.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 0.1)));
    test_abstract_group(&mol, thresh, "C6", 6, 6, true);
}

#[test]
fn test_abstract_group_symmetric_arbitrary_half_sandwich_magnetic_field_cn() {
    let thresh = 1e-7;
    for n in 3..=32 {
        let mut mol = template_molecules::gen_arbitrary_half_sandwich(n);
        mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 0.1)));
        test_abstract_group(
            &mol,
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
    test_abstract_group(&mol, thresh, "C3v", 6, 3, false);
}

#[test]
fn test_abstract_group_symmetric_bf3_electric_field_c3v() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/bf3.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    test_abstract_group(&mol, thresh, "C3v", 6, 3, false);
}

#[test]
fn test_abstract_group_symmetric_adamantane_electric_field_c3v() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/adamantane.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(Vector3::new(0.1, 0.1, 0.1)));
    test_abstract_group(&mol, thresh, "C3v", 6, 3, false);
}

#[test]
fn test_abstract_group_symmetric_ch4_electric_field_c3v() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/ch4.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(Vector3::new(1.0, 1.0, 1.0)));
    test_abstract_group(&mol, thresh, "C3v", 6, 3, false);
}

#[test]
fn test_abstract_group_symmetric_vf6_electric_field_c3v() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/vf6.xyz");
    let thresh = 1e-12;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(0.2 * Vector3::new(1.0, 1.0, 1.0)));
    test_abstract_group(&mol, thresh, "C3v", 6, 3, false);
}

#[test]
fn test_abstract_group_symmetric_sf5cl_c4v() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/sf5cl.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_abstract_group(&mol, thresh, "C4v", 8, 5, false);
}

#[test]
fn test_abstract_group_symmetric_h8_electric_field_c4v() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/h8.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(0.2 * Vector3::new(0.0, 0.0, 1.0)));
    test_abstract_group(&mol, thresh, "C4v", 8, 5, false);
}

#[test]
fn test_abstract_group_symmetric_vf6_electric_field_c4v() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/vf6.xyz");
    let thresh = 1e-12;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(0.2 * Vector3::new(1.0, 0.0, 0.0)));
    test_abstract_group(&mol, thresh, "C4v", 8, 5, false);
}

#[test]
fn test_abstract_group_symmetric_antiprism_pb10_electric_field_c4v() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/pb10.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    test_abstract_group(&mol, thresh, "C4v", 8, 5, false);
}

#[test]
fn test_abstract_group_symmetric_cpnico_c5v() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/cpnico.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    test_abstract_group(&mol, thresh, "C5v", 10, 4, false);
}

#[test]
fn test_abstract_group_symmetric_staggered_ferrocene_electric_field_c5v() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/staggered_ferrocene.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    test_abstract_group(&mol, thresh, "C5v", 10, 4, false);
}

#[test]
fn test_abstract_group_symmetric_c60_electric_field_c5v() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c60.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    test_abstract_group(&mol, thresh, "C5v", 10, 4, false);
}

#[test]
fn test_abstract_group_symmetric_b7_c6v() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/b7.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    test_abstract_group(&mol, thresh, "C6v", 12, 6, false);
}

#[test]
fn test_abstract_group_symmetric_au26_electric_field_c6v() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/au26.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    test_abstract_group(&mol, thresh, "C6v", 12, 6, false);
}

#[test]
fn test_abstract_group_symmetric_benzene_electric_field_c6v() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/benzene.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(Vector3::new(1.0, 0.0, 0.0)));
    test_abstract_group(&mol, thresh, "C6v", 12, 6, false);
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
            &mol,
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
            &mol,
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
    test_abstract_group(&mol, thresh, "C3h", 6, 6, true);
}

#[test]
fn test_abstract_group_symmetric_xef4_magnetic_field_c4h() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/xef4.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, -1.0)));
    test_abstract_group(&mol, thresh, "C4h", 8, 8, true);
}

#[test]
fn test_abstract_group_symmetric_vf6_magnetic_field_c4h() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/vf6.xyz");
    let thresh = 1e-12;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    test_abstract_group(&mol, thresh, "C4h", 8, 8, true);
}

#[test]
fn test_abstract_group_symmetric_h8_magnetic_field_c4h() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/h8.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, 1e-7);
    mol.set_magnetic_field(Some(0.2 * Vector3::new(0.0, 0.0, 1.0)));
    test_abstract_group(&mol, thresh, "C4h", 8, 8, true);
}

#[test]
fn test_abstract_group_symmetric_eclipsed_ferrocene_magnetic_field_c5h() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/eclipsed_ferrocene.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, -1.0)));
    test_abstract_group(&mol, thresh, "C5h", 10, 10, true);
}

#[test]
fn test_abstract_group_symmetric_benzene_magnetic_field_c6h() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/benzene.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(1.0, 0.0, 0.0)));
    test_abstract_group(&mol, thresh, "C6h", 12, 12, true);
}

#[test]
fn test_abstract_group_symmetric_arbitrary_eclipsed_sandwich_magnetic_field_cnh() {
    // env_logger::init();
    for n in 3..=20 {
        let mut mol = template_molecules::gen_arbitrary_eclipsed_sandwich(n);
        let thresh = 1e-7;
        mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 0.1)));
        test_abstract_group(
            &mol,
            thresh,
            format!("C{}h", n).as_str(),
            2 * n as usize,
            2 * n as usize,
            true,
        );
    }
}

/*
Dn
*/

#[test]
fn test_abstract_group_symmetric_triphenyl_radical_d3() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/triphenylradical.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_abstract_group(&mol, thresh, "D3", 6, 3, false);
}

#[test]
fn test_abstract_group_symmetric_h8_twisted_d4() {
    let thresh = 1e-7;
    let mol = template_molecules::gen_twisted_h8(0.1);
    test_abstract_group(&mol, thresh, "D4", 8, 5, false);
}

#[test]
fn test_abstract_group_symmetric_c5ph5_d5() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c5ph5.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_abstract_group(&mol, thresh, "D5", 10, 4, false);
}

#[test]
fn test_abstract_group_symmetric_c6ph6_d6() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c6ph6.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    test_abstract_group(&mol, thresh, "D6", 12, 6, false);
}

#[test]
fn test_abstract_group_symmetric_arbitrary_twisted_sandwich_dn() {
    /* The expected number of classes is deduced from the irrep structures of
     * the Dn groups.
     * When n is even, the irreps are A1, A2, B1, B2, Ek where k = 1, ..., n/2 - 1.
     * When n is odd, the irreps are A1, A2, Ek where k = 1, ..., n//2.
     */
    let thresh = 1e-7;
    for n in 3..=20 {
        let mol = template_molecules::gen_arbitrary_twisted_sandwich(n, 0.1);
        test_abstract_group(
            &mol,
            thresh,
            format!("D{}", n).as_str(),
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
Dnh
*/

#[test]
fn test_abstract_group_symmetric_bf3_d3h() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/bf3.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_abstract_group(&mol, thresh, "D3h", 12, 6, false);
}

#[test]
fn test_abstract_group_symmetric_xef4_d4h() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/xef4.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_abstract_group(&mol, thresh, "D4h", 16, 10, false);
}

#[test]
fn test_abstract_group_symmetric_h8_d4h() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/h8.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_abstract_group(&mol, thresh, "D4h", 16, 10, false);
}

#[test]
fn test_abstract_group_symmetric_eclipsed_ferrocene_d5h() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/eclipsed_ferrocene.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    test_abstract_group(&mol, thresh, "D5h", 20, 8, false);
}

#[test]
fn test_abstract_group_symmetric_benzene_d6h() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/benzene.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_abstract_group(&mol, thresh, "D6h", 24, 12, false);
}

#[test]
fn test_abstract_group_symmetric_arbitrary_eclipsed_sandwich_dnh() {
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
        test_abstract_group(
            &mol,
            thresh,
            format!("D{}h", n).as_str(),
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

/*
Dnd
*/

#[test]
fn test_abstract_group_symmetric_b2cl4_d2d() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/b2cl4.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_abstract_group(&mol, thresh, "D2d", 8, 5, false);
}

#[test]
fn test_abstract_group_symmetric_s4n4_d2d() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/s4n4.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_abstract_group(&mol, thresh, "D2d", 8, 5, false);
}

#[test]
fn test_abstract_group_symmetric_pbet4_d2d() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/pbet4.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_abstract_group(&mol, thresh, "D2d", 8, 5, false);
}

#[test]
fn test_abstract_group_symmetric_allene_d2d() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/allene.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_abstract_group(&mol, thresh, "D2d", 8, 5, false);
}

#[test]
fn test_abstract_group_symmetric_staggered_c2h6_d3d() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c2h6.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    test_abstract_group(&mol, thresh, "D3d", 12, 6, false);
}

#[test]
fn test_abstract_group_symmetric_cyclohexane_chair_d3d() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/cyclohexane_chair.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_abstract_group(&mol, thresh, "D3d", 12, 6, false);
}

#[test]
fn test_abstract_group_symmetric_s8_d4d() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/s8.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_abstract_group(&mol, thresh, "D4d", 16, 7, false);
}

#[test]
fn test_abstract_group_symmetric_antiprism_h8_d4d() {
    let mol = template_molecules::gen_twisted_h8(std::f64::consts::FRAC_PI_4);
    let thresh = 1e-7;
    test_abstract_group(&mol, thresh, "D4d", 16, 7, false);
}

#[test]
fn test_abstract_group_symmetric_antiprism_pb10_d4d() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/pb10.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_abstract_group(&mol, thresh, "D4d", 16, 7, false);
}

#[test]
fn test_abstract_group_symmetric_staggered_ferrocene_d5d() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/staggered_ferrocene.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    test_abstract_group(&mol, thresh, "D5d", 20, 8, false);
}

#[test]
fn test_abstract_group_symmetric_au26_d6d() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/au26.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    test_abstract_group(&mol, thresh, "D6d", 24, 9, false);
}

#[test]
fn test_abstract_group_symmetric_arbitrary_staggered_sandwich_dnd() {
    let thresh = 1e-7;
    for n in 3..=20 {
        let mol = template_molecules::gen_arbitrary_twisted_sandwich(n, 0.5);
        test_abstract_group(
            &mol,
            thresh,
            format!("D{}d", n).as_str(),
            4 * n as usize,
            (4 + n - 1) as usize,
            false,
        );
    }
}

/*
S2n
*/

#[test]
fn test_abstract_group_symmetric_b2cl4_magnetic_field_s4() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/b2cl4.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    test_abstract_group(&mol, thresh, "S4", 4, 4, true);
}

#[test]
fn test_abstract_group_symmetric_adamantane_magnetic_field_s4() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/adamantane.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(-0.1, 0.0, 0.0)));
    test_abstract_group(&mol, thresh, "S4", 4, 4, true);
}

#[test]
fn test_abstract_group_symmetric_ch4_magnetic_field_s4() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/ch4.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    test_abstract_group(&mol, thresh, "S4", 4, 4, true);
}

#[test]
fn test_abstract_group_symmetric_65coronane_s6() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/coronane65.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_abstract_group(&mol, thresh, "S6", 6, 6, true);
}

#[test]
fn test_abstract_group_symmetric_65coronane_magnetic_field_s6() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/coronane65.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, -1.0)));
    test_abstract_group(&mol, thresh, "S6", 6, 6, true);
}

#[test]
fn test_abstract_group_symmetric_staggered_c2h6_magnetic_field_s6() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c2h6.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    test_abstract_group(&mol, thresh, "S6", 6, 6, true);
}

#[test]
fn test_abstract_group_symmetric_c60_magnetic_field_s6() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c60.xyz");
    let thresh = 1e-5;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(
        -0.5773503107731,
        -0.1875926572335,
        0.7946543988441,
    )));
    test_abstract_group(&mol, thresh, "S6", 6, 6, true);
}

#[test]
fn test_abstract_group_symmetric_vh2o6_magnetic_field_s6() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/vh2o6.xyz");
    let thresh = 1e-12;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(-0.2, 0.2, 0.2)));
    test_abstract_group(&mol, thresh, "S6", 6, 6, true);
}

#[test]
fn test_abstract_group_symmetric_vf6_magnetic_field_s6() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/vf6.xyz");
    let thresh = 1e-12;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(1.0, 1.0, 1.0)));
    test_abstract_group(&mol, thresh, "S6", 6, 6, true);
}

#[test]
fn test_abstract_group_symmetric_s8_magnetic_field_s8() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/s8.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    test_abstract_group(&mol, thresh, "S8", 8, 8, true);
}

#[test]
fn test_abstract_group_symmetric_antiprism_pb10_magnetic_field_s8() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/pb10.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    test_abstract_group(&mol, thresh, "S8", 8, 8, true);
}

#[test]
fn test_abstract_group_symmetric_staggered_ferrocene_magnetic_field_s10() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/staggered_ferrocene.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    test_abstract_group(&mol, thresh, "S10", 10, 10, true);
}

#[test]
fn test_abstract_group_symmetric_c60_magnetic_field_s10() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c60.xyz");
    let thresh = 1e-5;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    test_abstract_group(&mol, thresh, "S10", 10, 10, true);
}

#[test]
fn test_abstract_group_symmetric_au26_magnetic_field_s12() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/au26.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    test_abstract_group(&mol, thresh, "S12", 12, 12, true);
}

#[test]
fn test_abstract_group_symmetric_arbitrary_staggered_sandwich_magnetic_field_s2n() {
    let thresh = 1e-7;
    for n in 3..=20 {
        let mut mol = template_molecules::gen_arbitrary_twisted_sandwich(n, 0.5);
        mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
        test_abstract_group(
            &mol,
            thresh,
            format!("S{}", 2 * n).as_str(),
            2 * n as usize,
            2 * n as usize,
            true,
        );
    }
}

/*********
Asymmetric
*********/

/*
C2
*/

#[test]
fn test_abstract_group_asymmetric_spiroketal_c2() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/spiroketal.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_abstract_group(&mol, thresh, "C2", 2, 2, true);
}

#[test]
fn test_abstract_group_asymmetric_cyclohexene_c2() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/cyclohexene.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_abstract_group(&mol, thresh, "C2", 2, 2, true);
}

#[test]
fn test_abstract_group_asymmetric_thf_c2() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/thf.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_abstract_group(&mol, thresh, "C2", 2, 2, true);
}

#[test]
fn test_abstract_group_asymmetric_tartaricacid_c2() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/tartaricacid.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_abstract_group(&mol, thresh, "C2", 2, 2, true);
}

#[test]
fn test_abstract_group_asymmetric_f2allene_c2() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/f2allene.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_abstract_group(&mol, thresh, "C2", 2, 2, true);
}

#[test]
fn test_abstract_group_asymmetric_water_magnetic_field_c2() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/water.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 1.0, 0.0)));
    test_abstract_group(&mol, thresh, "C2", 2, 2, true);
}

#[test]
fn test_abstract_group_asymmetric_pyridine_magnetic_field_c2() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/pyridine.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.2, 0.0)));
    test_abstract_group(&mol, thresh, "C2", 2, 2, true);
}

#[test]
fn test_abstract_group_asymmetric_cyclobutene_magnetic_field_c2() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/cyclobutene.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.2, 0.0, 0.0)));
    test_abstract_group(&mol, thresh, "C2", 2, 2, true);
}

#[test]
fn test_abstract_group_asymmetric_azulene_magnetic_field_c2() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/azulene.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 0.2)));
    test_abstract_group(&mol, thresh, "C2", 2, 2, true);
}

#[test]
fn test_abstract_group_asymmetric_cis_cocl2h4o2_magnetic_field_c2() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/cis-cocl2h4o2.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.2, 0.0)));
    test_abstract_group(&mol, thresh, "C2", 2, 2, true);
}

#[test]
fn test_abstract_group_asymmetric_cuneane_magnetic_field_c2() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/cuneane.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 0.2)));
    test_abstract_group(&mol, thresh, "C2", 2, 2, true);
}

/***
C2v
***/

#[test]
fn test_abstract_group_asymmetric_water_c2v() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/water.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_abstract_group(&mol, thresh, "C2v", 4, 4, true);
}

#[test]
fn test_abstract_group_asymmetric_pyridine_c2v() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/pyridine.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_abstract_group(&mol, thresh, "C2v", 4, 4, true);
}

#[test]
fn test_abstract_group_asymmetric_cyclobutene_c2v() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/cyclobutene.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_abstract_group(&mol, thresh, "C2v", 4, 4, true);
}

#[test]
fn test_abstract_group_asymmetric_azulene_c2v() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/azulene.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_abstract_group(&mol, thresh, "C2v", 4, 4, true);
}

#[test]
fn test_abstract_group_asymmetric_cuneane_c2v() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/cuneane.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_abstract_group(&mol, thresh, "C2v", 4, 4, true);
}

#[test]
fn test_abstract_group_asymmetric_bf3_electric_field_c2v() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/bf3.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(Vector3::new(0.2, 0.0, 0.0)));
    test_abstract_group(&mol, thresh, "C2v", 4, 4, true);
}

/***
C2h
***/

#[test]
fn test_abstract_group_asymmetric_h2o2_c2h() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/h2o2.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    test_abstract_group(&mol, thresh, "C2h", 4, 4, true);
}

#[test]
fn test_abstract_group_asymmetric_zethrene_c2h() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/zethrene.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    test_abstract_group(&mol, thresh, "C2h", 4, 4, true);
}

#[test]
fn test_abstract_group_asymmetric_distorted_vf6_magnetic_field_c2h() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/vf6_d2h.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(1.0, 0.0, 0.0)));
    test_abstract_group(&mol, thresh, "C2h", 4, 4, true);
}

#[test]
fn test_abstract_group_asymmetric_b2h6_magnetic_field_c2h() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/b2h6.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(1.0, 0.0, 0.0)));
    test_abstract_group(&mol, thresh, "C2h", 4, 4, true);
}

#[test]
fn test_abstract_group_asymmetric_naphthalene_magnetic_field_c2h() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/naphthalene.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 1.0, 0.0)));
    test_abstract_group(&mol, thresh, "C2h", 4, 4, true);
}

#[test]
fn test_abstract_group_asymmetric_pyrene_magnetic_field_c2h() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/pyrene.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    test_abstract_group(&mol, thresh, "C2h", 4, 4, true);
}

#[test]
fn test_abstract_group_asymmetric_c6o6_magnetic_field_c2h() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c6o6.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    test_abstract_group(&mol, thresh, "C2h", 4, 4, true);
}

/*
Cs
*/

#[test]
fn test_abstract_group_asymmetric_propene_cs() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/propene.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_abstract_group(&mol, thresh, "Cs", 2, 2, true);
}

#[test]
fn test_abstract_group_asymmetric_socl2_cs() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/socl2.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_abstract_group(&mol, thresh, "Cs", 2, 2, true);
}

#[test]
fn test_abstract_group_asymmetric_hocl_cs() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/hocl.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_abstract_group(&mol, thresh, "Cs", 2, 2, true);
}

#[test]
fn test_abstract_group_asymmetric_hocn_cs() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/hocn.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_abstract_group(&mol, thresh, "Cs", 2, 2, true);
}

#[test]
fn test_abstract_group_asymmetric_nh2f_cs() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/nh2f.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    test_abstract_group(&mol, thresh, "Cs", 2, 2, true);
}

#[test]
fn test_abstract_group_asymmetric_phenol_cs() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/phenol.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_abstract_group(&mol, thresh, "Cs", 2, 2, true);
}

#[test]
fn test_abstract_group_asymmetric_f_pyrrole_cs() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/f-pyrrole.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    test_abstract_group(&mol, thresh, "Cs", 2, 2, true);
}

#[test]
fn test_abstract_group_asymmetric_n2o_cs() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/n2o.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_abstract_group(&mol, thresh, "Cs", 2, 2, true);
}

#[test]
fn test_abstract_group_asymmetric_fclbenzene_cs() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/fclbenzene.xyz");
    let thresh = 1e-5;
    let mol = Molecule::from_xyz(&path, thresh);
    test_abstract_group(&mol, thresh, "Cs", 2, 2, true);
}

#[test]
fn test_abstract_group_asymmetric_water_magnetic_field_cs() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/water.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(1.0, 0.0, 0.0)));
    test_abstract_group(&mol, thresh, "Cs", 2, 2, true);
}

#[test]
fn test_abstract_group_asymmetric_pyridine_magnetic_field_cs() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/pyridine.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 0.2)));
    test_abstract_group(&mol, thresh, "Cs", 2, 2, true);
}

#[test]
fn test_abstract_group_asymmetric_cyclobutene_magnetic_field_cs() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/cyclobutene.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.1, 0.0)));
    test_abstract_group(&mol, thresh, "Cs", 2, 2, true);
}

#[test]
fn test_abstract_group_asymmetric_azulene_magnetic_field_cs() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/azulene.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.1, 0.0)));
    test_abstract_group(&mol, thresh, "Cs", 2, 2, true);
}

#[test]
fn test_abstract_group_asymmetric_cis_cocl2h4o2_magnetic_field_cs() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/cis-cocl2h4o2.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 0.2)));
    test_abstract_group(&mol, thresh, "Cs", 2, 2, true);
}

#[test]
fn test_abstract_group_asymmetric_cuneane_magnetic_field_cs() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/cuneane.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.5, 0.0)));
    test_abstract_group(&mol, thresh, "Cs", 2, 2, true);
}

#[test]
fn test_abstract_group_asymmetric_water_electric_field_cs() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/water.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(Vector3::new(1.0, 0.0, 0.0)));
    test_abstract_group(&mol, thresh, "Cs", 2, 2, true);
}

#[test]
fn test_abstract_group_asymmetric_pyridine_electric_field_cs() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/pyridine.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(Vector3::new(0.0, 0.0, 0.2)));
    test_abstract_group(&mol, thresh, "Cs", 2, 2, true);
}

#[test]
fn test_abstract_group_asymmetric_cyclobutene_electric_field_cs() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/cyclobutene.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(Vector3::new(0.0, 0.1, 0.0)));
    test_abstract_group(&mol, thresh, "Cs", 2, 2, true);
}

#[test]
fn test_abstract_group_asymmetric_azulene_electric_field_cs() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/azulene.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(Vector3::new(0.0, 0.1, 0.0)));
    test_abstract_group(&mol, thresh, "Cs", 2, 2, true);
}

#[test]
fn test_abstract_group_asymmetric_cis_cocl2h4o2_electric_field_cs() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/cis-cocl2h4o2.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(Vector3::new(0.0, 0.0, 0.2)));
    test_abstract_group(&mol, thresh, "Cs", 2, 2, true);
}

#[test]
fn test_abstract_group_asymmetric_cuneane_electric_field_cs() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/cuneane.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(Vector3::new(0.0, 0.5, 0.0)));
    test_abstract_group(&mol, thresh, "Cs", 2, 2, true);
}

#[test]
fn test_abstract_group_asymmetric_bf3_magnetic_field_cs() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/bf3.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.5, 0.0)));
    test_abstract_group(&mol, thresh, "Cs", 2, 2, true);
}

/// This is a special case: Cs point group in a symmetric top.
#[test]
fn test_abstract_group_symmetric_ch4_magnetic_field_cs() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/ch4.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.1, 0.1, 0.0)));
    test_abstract_group(&mol, thresh, "Cs", 2, 2, true);
}

/// This is another special case: Cs point group in a symmetric top.
#[test]
fn test_abstract_group_symmetric_ch4_electric_field_cs() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/ch4.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(Vector3::new(0.1, 0.1, 0.0)));
    test_abstract_group(&mol, thresh, "Cs", 2, 2, true);
}

#[test]
fn test_abstract_group_asymmetric_atom_magnetic_electric_field_cs() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/th.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 0.2)));
    mol.set_electric_field(Some(Vector3::new(0.0, 0.1, 0.0)));
    test_abstract_group(&mol, thresh, "Cs", 2, 2, true);
}

/*
D2
*/

#[test]
fn test_abstract_group_asymmetric_i4_biphenyl_d2() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/i4-biphenyl.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_abstract_group(&mol, thresh, "D2", 4, 4, true);
}

#[test]
fn test_abstract_group_asymmetric_twistane_d2() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/twistane.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_abstract_group(&mol, thresh, "D2", 4, 4, true);
}

#[test]
fn test_abstract_group_asymmetric_22_paracyclophane_d2() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/paracyclophane22.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_abstract_group(&mol, thresh, "D2", 4, 4, true);
}

/***
D2h
***/

#[test]
fn test_abstract_group_asymmetric_b2h6_d2h() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/b2h6.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_abstract_group(&mol, thresh, "D2h", 8, 8, true);
}

#[test]
fn test_abstract_group_asymmetric_naphthalene_d2h() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/naphthalene.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_abstract_group(&mol, thresh, "D2h", 8, 8, true);
}

#[test]
fn test_abstract_group_asymmetric_pyrene_d2h() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/pyrene.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_abstract_group(&mol, thresh, "D2h", 8, 8, true);
}

#[test]
fn test_abstract_group_asymmetric_c6o6_d2h() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c6o6.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_abstract_group(&mol, thresh, "D2h", 8, 8, true);
}

#[test]
fn test_abstract_group_asymmetric_distorted_vf6_d2h() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/vf6_d2h.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_abstract_group(&mol, thresh, "D2h", 8, 8, true);
}

/***
Ci
***/

#[test]
fn test_abstract_group_asymmetric_meso_tartaricacid_ci() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/meso-tartaricacid.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_abstract_group(&mol, thresh, "Ci", 2, 2, true);
}

#[test]
fn test_abstract_group_asymmetric_dibromodimethylcyclohexane_ci() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/dibromodimethylcyclohexane.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_abstract_group(&mol, thresh, "Ci", 2, 2, true);
}

#[test]
fn test_abstract_group_asymmetric_h2o2_magnetic_field_ci() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/h2o2.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(1.0, 2.0, -1.0)));
    test_abstract_group(&mol, thresh, "Ci", 2, 2, true);
}

#[test]
fn test_abstract_group_symmetric_xef4_magnetic_field_ci() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/xef4.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(1.0, 2.0, -2.0)));
    test_abstract_group(&mol, thresh, "Ci", 2, 2, true);
}

#[test]
fn test_abstract_group_asymmetric_c2h2_magnetic_field_ci() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c2h2.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(1.0, -1.0, 1.0)));
    test_abstract_group(&mol, thresh, "Ci", 2, 2, true);
}

/***
C1
***/

#[test]
fn test_abstract_group_asymmetric_butan1ol_c1() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/butan-1-ol.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_abstract_group(&mol, thresh, "C1", 1, 1, true);
}

#[test]
fn test_abstract_group_asymmetric_subst_5m_ring_c1() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/subst-5m-ring.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    test_abstract_group(&mol, thresh, "C1", 1, 1, true);
}

#[test]
fn test_abstract_group_asymmetric_bf3_magnetic_field_c1() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/bf3.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(1.0, -1.0, 1.0)));
    test_abstract_group(&mol, thresh, "C1", 1, 1, true);
}
