use crate::aux::molecule::Molecule;
use crate::aux::template_molecules;
use crate::symmetry::symmetry_core::{PreSymmetry, Symmetry};
use crate::symmetry::symmetry_element::{ROT, SIG, TRROT, TRSIG};
use crate::symmetry::symmetry_element_order::{ElementOrder, ORDER_1, ORDER_2};
use nalgebra::Vector3;

const ROOT: &str = env!("CARGO_MANIFEST_DIR");
use env_logger;

/********
Spherical
********/

#[test]
fn test_point_group_detection_spherical_atom_o3() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/th.xyz");
    let mol = Molecule::from_xyz(&path, 1e-6);
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-14)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("O(3)".to_owned()));
    assert_eq!(
        sym.get_generators(&ROT)
            .expect("No proper generators found.")[&ElementOrder::Inf]
            .len(),
        3
    );
}

#[test]
fn test_point_group_detection_spherical_c60_ih() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c60.xyz");
    let mol = Molecule::from_xyz(&path, 1e-6);
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-6)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("Ih".to_owned()));
    assert_eq!(
        sym.get_elements(&ROT).expect("No proper elements found.")[&ElementOrder::Int(5)].len(),
        6
    );
    assert_eq!(
        sym.get_elements(&ROT).expect("No proper elements found.")[&ElementOrder::Int(3)].len(),
        10
    );
    assert_eq!(
        sym.get_elements(&ROT).expect("No proper elements found.")[&ElementOrder::Int(2)].len(),
        15
    );
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(10)].len(),
        6
    );
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(6)].len(),
        10
    );
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(2)].len(),
        1
    );
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(1)].len(),
        15
    );
    assert_eq!(sym.get_sigma_elements("").unwrap().len(), 15);

    assert_eq!(
        sym.get_generators(&ROT)
            .expect("No proper generators found.")[&ElementOrder::Int(5)]
            .len(),
        6
    );
    assert_eq!(
        sym.get_generators(&SIG)
            .expect("No improper generators found.")[&ElementOrder::Int(2)]
            .len(),
        1
    );
}

#[test]
fn test_point_group_detection_spherical_ch4_td() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/ch4.xyz");
    let mol = Molecule::from_xyz(&path, 1e-6);
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-6)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("Td".to_owned()));
    assert_eq!(
        sym.get_elements(&ROT).expect("No proper elements found.")[&ElementOrder::Int(3)].len(),
        4
    );
    assert_eq!(
        sym.get_elements(&ROT).expect("No proper elements found.")[&ElementOrder::Int(2)].len(),
        3
    );
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(4)].len(),
        3
    );
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(1)].len(),
        6
    );
    assert_eq!(sym.get_sigma_elements("d").unwrap().len(), 6);

    assert_eq!(
        sym.get_generators(&ROT)
            .expect("No proper generators found.")[&ElementOrder::Int(3)]
            .len(),
        4
    );
    assert_eq!(
        sym.get_generators(&SIG)
            .expect("No improper generators found.")[&ElementOrder::Int(1)]
            .len(),
        1
    );
    assert_eq!(sym.get_sigma_generators("d").unwrap().len(), 1);
}

#[test]
fn test_point_group_detection_spherical_adamantane_td() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/adamantane.xyz");
    let mol = Molecule::from_xyz(&path, 1e-6);
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-6)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("Td".to_owned()));
    assert_eq!(
        sym.get_elements(&ROT).expect("No proper elements found.")[&ElementOrder::Int(3)].len(),
        4
    );
    assert_eq!(
        sym.get_elements(&ROT).expect("No proper elements found.")[&ElementOrder::Int(2)].len(),
        3
    );
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(4)].len(),
        3
    );
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(1)].len(),
        6
    );
    assert_eq!(sym.get_sigma_elements("d").unwrap().len(), 6);

    assert_eq!(
        sym.get_generators(&ROT)
            .expect("No proper generators found.")[&ElementOrder::Int(3)]
            .len(),
        4
    );
    assert_eq!(
        sym.get_generators(&SIG)
            .expect("No improper generators found.")[&ElementOrder::Int(1)]
            .len(),
        1
    );
    assert_eq!(sym.get_sigma_generators("d").unwrap().len(), 1);
}

#[test]
fn test_point_group_detection_spherical_c165_diamond_nanoparticle_td() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c165.xyz");
    let mol = Molecule::from_xyz(&path, 1e-6);
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-6)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("Td".to_owned()));
    assert_eq!(
        sym.get_elements(&ROT).expect("No proper elements found.")[&ElementOrder::Int(3)].len(),
        4
    );
    assert_eq!(
        sym.get_elements(&ROT).expect("No proper elements found.")[&ElementOrder::Int(2)].len(),
        3
    );
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(4)].len(),
        3
    );
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(1)].len(),
        6
    );
    assert_eq!(sym.get_sigma_elements("d").unwrap().len(), 6);

    assert_eq!(
        sym.get_generators(&ROT)
            .expect("No proper generators found.")[&ElementOrder::Int(3)]
            .len(),
        4
    );
    assert_eq!(
        sym.get_generators(&SIG)
            .expect("No improper generators found.")[&ElementOrder::Int(1)]
            .len(),
        1
    );
    assert_eq!(sym.get_sigma_generators("d").unwrap().len(), 1);
}

#[test]
fn test_point_group_detection_spherical_vh2o6_th() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/vh2o6.xyz");
    let mol = Molecule::from_xyz(&path, 1e-12);
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-12)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("Th".to_owned()));
    assert_eq!(
        sym.get_elements(&ROT).expect("No proper elements found.")[&ElementOrder::Int(3)].len(),
        4
    );
    assert_eq!(
        sym.get_elements(&ROT).expect("No proper elements found.")[&ElementOrder::Int(2)].len(),
        3
    );
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(6)].len(),
        4
    );
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(2)].len(),
        1
    );
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(1)].len(),
        3
    );
    assert_eq!(sym.get_sigma_elements("h").unwrap().len(), 3);

    assert_eq!(
        sym.get_generators(&ROT)
            .expect("No proper generators found.")[&ElementOrder::Int(3)]
            .len(),
        4
    );
    assert_eq!(
        sym.get_generators(&SIG)
            .expect("No improper generators found.")[&ElementOrder::Int(2)]
            .len(),
        1
    );
}

#[test]
fn test_point_group_detection_spherical_vf6_oh() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/vf6.xyz");
    let mol = Molecule::from_xyz(&path, 1e-12);
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-12)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("Oh".to_owned()));
    assert_eq!(
        sym.get_elements(&ROT).expect("No proper elements found.")[&ElementOrder::Int(4)].len(),
        3
    );
    assert_eq!(
        sym.get_elements(&ROT).expect("No proper elements found.")[&ElementOrder::Int(3)].len(),
        4
    );
    assert_eq!(
        sym.get_elements(&ROT).expect("No proper elements found.")[&ElementOrder::Int(2)].len(),
        9
    );
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(6)].len(),
        4
    );
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(4)].len(),
        3
    );
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(2)].len(),
        1
    );
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(1)].len(),
        9
    );
    assert_eq!(sym.get_sigma_elements("h").unwrap().len(), 3);
    assert_eq!(sym.get_sigma_elements("d").unwrap().len(), 6);

    assert_eq!(
        sym.get_generators(&ROT)
            .expect("No proper generators found.")[&ElementOrder::Int(3)]
            .len(),
        4
    );
    assert_eq!(
        sym.get_generators(&ROT)
            .expect("No proper generators found.")[&ElementOrder::Int(4)]
            .len(),
        1
    );
    assert_eq!(
        sym.get_generators(&SIG)
            .expect("No improper generators found.")[&ElementOrder::Int(2)]
            .len(),
        1
    );
}

/*****
Linear
*****/

#[test]
fn test_point_group_detection_linear_atom_magnetic_field_cinfh() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/th.xyz");
    let mut mol = Molecule::from_xyz(&path, 1e-7);
    mol.set_magnetic_field(Some(Vector3::new(1.0, 2.0, -1.0)));
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-14)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("C∞h".to_owned()));
    assert_eq!(
        sym.get_generators(&ROT)
            .expect("No proper generators found.")[&ElementOrder::Inf]
            .len(),
        1
    );
    assert_eq!(
        sym.get_generators(&SIG)
            .expect("No improper generators found.")[&ElementOrder::Int(1)]
            .len(),
        1
    );
    assert_eq!(sym.get_sigma_generators("h").unwrap().len(), 1);
}

#[test]
fn test_point_group_detection_linear_atom_magnetic_field_bw_dinfh() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/th.xyz");
    let mut mol = Molecule::from_xyz(&path, 1e-7);
    mol.set_magnetic_field(Some(Vector3::new(1.0, 2.0, -1.0)));
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-14)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut magsym = Symmetry::new();
    magsym.analyse(&presym, true);
    assert_eq!(magsym.point_group, Some("D∞h".to_owned()));
    assert_eq!(
        magsym
            .get_generators(&ROT)
            .expect("No proper generators found.")[&ElementOrder::Inf]
            .len(),
        1
    );
    assert!(magsym
        .get_generators(&ROT)
        .expect("No time-reversed proper generators found.")
        .get(&ORDER_2)
        .is_none());
    assert_eq!(
        magsym
            .get_generators(&TRROT)
            .expect("No time-reversed proper generators found.")[&ElementOrder::Int(2)]
            .len(),
        1
    );
    assert_eq!(
        magsym
            .get_generators(&SIG)
            .expect("No improper generators found.")[&ElementOrder::Int(1)]
            .len(),
        1
    );
    assert!(magsym.get_generators(&TRSIG).is_none());
    assert_eq!(magsym.get_sigma_generators("h").unwrap().len(), 1);
}

#[test]
fn test_point_group_detection_linear_atom_electric_field_cinfv() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/th.xyz");
    let mut mol = Molecule::from_xyz(&path, 1e-7);
    mol.set_electric_field(Some(Vector3::new(-1.0, 3.0, -2.0)));
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-14)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("C∞v".to_owned()));
    assert_eq!(
        sym.get_generators(&ROT)
            .expect("No proper generators found.")[&ElementOrder::Inf]
            .len(),
        1
    );
    assert_eq!(
        sym.get_generators(&SIG)
            .expect("No improper generators found.")[&ElementOrder::Int(1)]
            .len(),
        1
    );
    assert_eq!(sym.get_sigma_generators("v").unwrap().len(), 1);
}

#[test]
fn test_point_group_detection_linear_c2h2_dinfh() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c2h2.xyz");
    let mol = Molecule::from_xyz(&path, 1e-6);
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-6)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("D∞h".to_owned()));
    assert_eq!(
        sym.get_generators(&ROT)
            .expect("No proper generators found.")[&ElementOrder::Inf]
            .len(),
        1
    );
    assert_eq!(
        sym.get_generators(&ROT)
            .expect("No proper generators found.")[&ElementOrder::Int(2)]
            .len(),
        1
    );
    assert_eq!(
        sym.get_generators(&SIG)
            .expect("No improper generators found.")[&ElementOrder::Int(1)]
            .len(),
        1
    );
    assert_eq!(sym.get_sigma_generators("h").unwrap().len(), 1);
}

#[test]
fn test_point_group_detection_linear_c2h2_magnetic_field_cinfh() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c2h2.xyz");
    let mut mol = Molecule::from_xyz(&path, 1e-6);

    // Parallel field
    mol.set_magnetic_field(Some(Vector3::new(1.0, 1.0, 1.0)));
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-6)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("C∞h".to_owned()));
    assert_eq!(
        sym.get_generators(&ROT)
            .expect("No proper generators found.")[&ElementOrder::Inf]
            .len(),
        1
    );
    assert_eq!(
        sym.get_generators(&SIG)
            .expect("No improper generators found.")[&ElementOrder::Int(1)]
            .len(),
        1
    );
    assert_eq!(sym.get_sigma_generators("h").unwrap().len(), 1);
}

#[test]
fn test_point_group_detection_linear_c2h2_magnetic_field_bw_dinfh() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c2h2.xyz");
    let mut mol = Molecule::from_xyz(&path, 1e-6);

    // Parallel field
    mol.set_magnetic_field(Some(Vector3::new(1.0, 1.0, 1.0)));
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-6)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut magsym = Symmetry::new();
    magsym.analyse(&presym, true);
    assert_eq!(magsym.point_group, Some("D∞h".to_owned()));
    assert_eq!(
        magsym
            .get_generators(&ROT)
            .expect("No proper generators found.")[&ElementOrder::Inf]
            .len(),
        1
    );
    assert!(magsym
        .get_generators(&ROT)
        .expect("No time-reversed proper generators found.")
        .get(&ORDER_2)
        .is_none());
    assert_eq!(
        magsym
            .get_generators(&TRROT)
            .expect("No time-reversed proper generators found.")[&ElementOrder::Int(2)]
            .len(),
        1
    );
    assert_eq!(
        magsym
            .get_generators(&SIG)
            .expect("No improper generators found.")[&ElementOrder::Int(1)]
            .len(),
        1
    );
    assert!(magsym.get_generators(&TRSIG).is_none());
    assert_eq!(magsym.get_sigma_generators("h").unwrap().len(), 1);
}

#[test]
fn test_point_group_detection_linear_c2h2_electric_field_cinfv() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c2h2.xyz");
    let mut mol = Molecule::from_xyz(&path, 1e-6);

    // Parallel field
    mol.set_electric_field(Some(Vector3::new(1.0, 1.0, 1.0)));
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-6)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("C∞v".to_owned()));
    assert_eq!(
        sym.get_generators(&ROT)
            .expect("No proper generators found.")[&ElementOrder::Inf]
            .len(),
        1
    );
}

#[test]
fn test_point_group_detection_linear_n3_cinfv() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/n3.xyz");
    let mol = Molecule::from_xyz(&path, 1e-6);
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-6)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("C∞v".to_owned()));
    assert_eq!(
        sym.get_generators(&ROT)
            .expect("No proper generators found.")[&ElementOrder::Inf]
            .len(),
        1
    );
    assert_eq!(
        sym.get_generators(&SIG)
            .expect("No improper generators found.")[&ElementOrder::Int(1)]
            .len(),
        1
    );
    assert_eq!(sym.get_sigma_generators("v").unwrap().len(), 1);
}

#[test]
fn test_point_group_detection_linear_n3_magnetic_field_cinf() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/n3.xyz");
    let mut mol = Molecule::from_xyz(&path, 1e-6);

    // Parallel field
    mol.set_magnetic_field(Some(Vector3::new(1.0, 1.0, 1.0)));
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-6)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("C∞".to_owned()));
    assert_eq!(
        sym.get_generators(&ROT)
            .expect("No proper generators found.")[&ElementOrder::Inf]
            .len(),
        1
    );
}

#[test]
fn test_point_group_detection_linear_n3_magnetic_field_bw_cinfv() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/n3.xyz");
    let mut mol = Molecule::from_xyz(&path, 1e-6);

    // Parallel field
    mol.set_magnetic_field(Some(Vector3::new(1.0, 1.0, 1.0)));
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-6)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut magsym = Symmetry::new();
    magsym.analyse(&presym, true);
    assert_eq!(magsym.point_group, Some("C∞v".to_owned()));
    assert_eq!(
        magsym
            .get_generators(&ROT)
            .expect("No proper generators found.")[&ElementOrder::Inf]
            .len(),
        1
    );
    assert!(magsym.get_generators(&TRROT).is_none());
    assert!(magsym.get_generators(&SIG).is_none());
    assert_eq!(
        magsym
            .get_generators(&TRSIG)
            .expect("No time-reversed improper generators found.")[&ElementOrder::Int(1)]
            .len(),
        1
    );
    assert_eq!(magsym.get_sigma_generators("v").unwrap().len(), 1);
}

#[test]
fn test_point_group_detection_linear_n3_electric_field_cinfv() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/n3.xyz");
    let mut mol = Molecule::from_xyz(&path, 1e-6);

    // Parallel field
    mol.set_electric_field(Some(Vector3::new(1.0, 1.0, 1.0)));
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-6)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("C∞v".to_owned()));
    assert_eq!(
        sym.get_generators(&ROT)
            .expect("No proper generators found.")[&ElementOrder::Inf]
            .len(),
        1
    );
    assert_eq!(
        sym.get_generators(&SIG)
            .expect("No improper generators found.")[&ElementOrder::Int(1)]
            .len(),
        1
    );
    assert_eq!(sym.get_sigma_generators("v").unwrap().len(), 1);
}

/********
Symmetric
********/

/*
Cn
*/

#[test]
fn test_point_group_detection_symmetric_ch4_magnetic_field_c3() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/ch4.xyz");
    let mut mol = Molecule::from_xyz(&path, 1e-6);
    mol.set_magnetic_field(Some(Vector3::new(1.0, 1.0, 1.0)));
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-6)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("C3".to_owned()));
    assert_eq!(
        sym.get_elements(&ROT).expect("No proper elements found.")[&ElementOrder::Int(3)].len(),
        1
    );
    assert_eq!(
        sym.get_generators(&ROT)
            .expect("No proper generators found.")[&ElementOrder::Int(3)]
            .len(),
        1
    );
}

#[test]
fn test_point_group_detection_symmetric_ch4_magnetic_field_bw_c3v() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/ch4.xyz");
    let mut mol = Molecule::from_xyz(&path, 1e-6);
    mol.set_magnetic_field(Some(Vector3::new(1.0, 1.0, 1.0)));
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-6)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut magsym = Symmetry::new();
    magsym.analyse(&presym, true);
    assert_eq!(magsym.point_group, Some("C3v".to_owned()));
    assert_eq!(
        magsym
            .get_elements(&ROT)
            .expect("No proper elements found.")[&ElementOrder::Int(3)]
            .len(),
        1
    );
    assert!(magsym.get_elements(&TRROT).is_none());
    assert!(magsym.get_elements(&SIG).is_none());
    assert_eq!(
        magsym
            .get_elements(&TRSIG)
            .expect("No time-reversed improper elements found.")[&ElementOrder::Int(1)]
            .len(),
        3
    );
    assert_eq!(magsym.get_sigma_elements("v").unwrap().len(), 3);

    assert_eq!(
        magsym
            .get_generators(&ROT)
            .expect("No proper generators found.")[&ElementOrder::Int(3)]
            .len(),
        1
    );
    assert!(magsym.get_generators(&TRROT).is_none());
    assert!(magsym.get_generators(&SIG).is_none());
    assert_eq!(
        magsym
            .get_generators(&TRSIG)
            .expect("No improper generators found.")[&ElementOrder::Int(1)]
            .len(),
        1
    );
    assert_eq!(magsym.get_sigma_generators("v").unwrap().len(), 1);
}

#[test]
fn test_point_group_detection_symmetric_adamantane_magnetic_field_c3() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/adamantane.xyz");
    let mut mol = Molecule::from_xyz(&path, 1e-6);
    mol.set_magnetic_field(Some(Vector3::new(0.1, 0.1, 0.1)));
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-6)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("C3".to_owned()));
    assert_eq!(
        sym.get_elements(&ROT).expect("No proper elements found.")[&ElementOrder::Int(3)].len(),
        1
    );
    assert_eq!(
        sym.get_generators(&ROT)
            .expect("No proper generators found.")[&ElementOrder::Int(3)]
            .len(),
        1
    );
}

#[test]
fn test_point_group_detection_symmetric_adamantane_magnetic_field_bw_c3v() {
    env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/adamantane.xyz");
    let mut mol = Molecule::from_xyz(&path, 1e-6);
    mol.set_magnetic_field(Some(Vector3::new(0.1, 0.1, 0.1)));
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-6)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut magsym = Symmetry::new();
    magsym.analyse(&presym, true);
    assert_eq!(magsym.point_group, Some("C3v".to_owned()));
    assert_eq!(
        magsym
            .get_elements(&ROT)
            .expect("No proper elements found.")[&ElementOrder::Int(3)]
            .len(),
        1
    );
    assert!(magsym.get_elements(&TRROT).is_none());
    assert!(magsym.get_elements(&SIG).is_none());
    assert_eq!(
        magsym
            .get_elements(&TRSIG)
            .expect("No time-reversed improper elements found.")[&ElementOrder::Int(1)]
            .len(),
        3
    );
    assert_eq!(magsym.get_sigma_elements("v").unwrap().len(), 3);

    assert_eq!(
        magsym
            .get_generators(&ROT)
            .expect("No proper generators found.")[&ElementOrder::Int(3)]
            .len(),
        1
    );
    assert!(magsym.get_generators(&TRROT).is_none());
    assert!(magsym.get_generators(&SIG).is_none());
    assert_eq!(
        magsym
            .get_generators(&TRSIG)
            .expect("No improper generators found.")[&ElementOrder::Int(1)]
            .len(),
        1
    );
    assert_eq!(magsym.get_sigma_generators("v").unwrap().len(), 1);
}

#[test]
fn test_point_group_detection_symmetric_vh2o6_electric_field_c3() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/vh2o6.xyz");
    let mut mol = Molecule::from_xyz(&path, 1e-12);
    mol.set_electric_field(Some(Vector3::new(-0.2, -0.2, -0.2)));
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-12)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("C3".to_owned()));
    assert_eq!(
        sym.get_elements(&ROT).expect("No proper elements found.")[&ElementOrder::Int(3)].len(),
        1
    );
    assert_eq!(
        sym.get_generators(&ROT)
            .expect("No proper generators found.")[&ElementOrder::Int(3)]
            .len(),
        1
    );
}

#[test]
fn test_point_group_detection_symmetric_65coronane_electric_field_c3() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/coronane65.xyz");
    let mut mol = Molecule::from_xyz(&path, 1e-7);
    mol.set_electric_field(Some(Vector3::new(0.0, 0.0, -1.0)));
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("C3".to_owned()));
    assert_eq!(
        sym.get_elements(&ROT).expect("No proper elements found.")[&ElementOrder::Int(3)].len(),
        1
    );
    assert_eq!(
        sym.get_generators(&ROT)
            .expect("No proper generators found.")[&ElementOrder::Int(3)]
            .len(),
        1
    );
}

#[test]
fn test_point_group_detection_symmetric_h8_twisted_magnetic_field_c4() {
    // env_logger::init();
    let mut mol = template_molecules::gen_twisted_h8(0.1);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 0.1)));
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("C4".to_owned()));
    assert_eq!(
        sym.get_elements(&ROT).expect("No proper elements found.")[&ElementOrder::Int(4)].len(),
        1
    );
    assert_eq!(
        sym.get_generators(&ROT)
            .expect("No proper generators found.")[&ElementOrder::Int(4)]
            .len(),
        1
    );
}

#[test]
fn test_point_group_detection_symmetric_h8_twisted_electric_field_c4() {
    let mut mol = template_molecules::gen_twisted_h8(0.1);
    mol.set_electric_field(Some(Vector3::new(0.0, 0.0, -0.1)));
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("C4".to_owned()));
    assert_eq!(
        sym.get_elements(&ROT).expect("No proper elements found.")[&ElementOrder::Int(4)].len(),
        1
    );
    assert_eq!(
        sym.get_generators(&ROT)
            .expect("No proper generators found.")[&ElementOrder::Int(4)]
            .len(),
        1
    );
}

#[test]
fn test_point_group_detection_symmetric_cpnico_magnetic_field_c5() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/cpnico.xyz");
    let mut mol = Molecule::from_xyz(&path, 1e-7);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, -0.2)));
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("C5".to_owned()));
    assert_eq!(
        sym.get_elements(&ROT).expect("No proper elements found.")[&ElementOrder::Int(5)].len(),
        1
    );
    assert_eq!(
        sym.get_generators(&ROT)
            .expect("No proper generators found.")[&ElementOrder::Int(5)]
            .len(),
        1
    );
}

#[test]
fn test_point_group_detection_symmetric_b7_magnetic_field_c6() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/b7.xyz");
    let mut mol = Molecule::from_xyz(&path, 1e-6);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 0.1)));
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-6)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("C6".to_owned()));
    assert_eq!(
        sym.get_elements(&ROT).expect("No proper elements found.")[&ElementOrder::Int(6)].len(),
        1
    );
    assert_eq!(
        sym.get_generators(&ROT)
            .expect("No proper generators found.")[&ElementOrder::Int(6)]
            .len(),
        1
    );
}

#[test]
fn test_point_group_detection_symmetric_arbitrary_half_sandwich_magnetic_field_cn() {
    for n in 3..=32 {
        let mut mol = template_molecules::gen_arbitrary_half_sandwich(n);
        mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 0.1)));
        let presym = PreSymmetry::builder()
            .moi_threshold(1e-7)
            .molecule(&mol, true)
            .build()
            .unwrap();
        let mut sym = Symmetry::new();
        sym.analyse(&presym, false);
        assert_eq!(sym.point_group, Some(format!("C{n}")));
        assert_eq!(
            sym.get_elements(&ROT).expect("No proper elements found.")[&ElementOrder::Int(n)].len(),
            1
        );
        assert_eq!(
            sym.get_generators(&ROT)
                .expect("No proper generators found.")[&ElementOrder::Int(n)]
                .len(),
            1
        );
    }
}

/*
Cnv
*/

#[test]
fn test_point_group_detection_symmetric_nh3_c3v() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/nh3.xyz");
    let mol = Molecule::from_xyz(&path, 1e-6);
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-6)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("C3v".to_owned()));
    assert_eq!(
        sym.get_elements(&ROT).expect("No proper elements found.")[&ElementOrder::Int(3)].len(),
        1
    );
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(1)].len(),
        3
    );
    assert_eq!(sym.get_sigma_elements("v").unwrap().len(), 3);

    assert_eq!(
        sym.get_generators(&ROT)
            .expect("No proper generators found.")[&ElementOrder::Int(3)]
            .len(),
        1
    );
    assert_eq!(
        sym.get_generators(&SIG)
            .expect("No improper generators found.")[&ElementOrder::Int(1)]
            .len(),
        1
    );
    assert_eq!(sym.get_sigma_generators("v").unwrap().len(), 1);
}

#[test]
fn test_point_group_detection_symmetric_bf3_electric_field_c3v() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/bf3.xyz");
    let mut mol = Molecule::from_xyz(&path, 1e-7);
    mol.set_electric_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("C3v".to_owned()));
    assert_eq!(
        sym.get_elements(&ROT).expect("No proper elements found.")[&ElementOrder::Int(3)].len(),
        1
    );
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(1)].len(),
        3
    );
    assert_eq!(sym.get_sigma_elements("v").unwrap().len(), 3);

    assert_eq!(
        sym.get_generators(&ROT)
            .expect("No proper generators found.")[&ElementOrder::Int(3)]
            .len(),
        1
    );
    assert_eq!(
        sym.get_generators(&SIG)
            .expect("No improper generators found.")[&ElementOrder::Int(1)]
            .len(),
        1
    );
    assert_eq!(sym.get_sigma_generators("v").unwrap().len(), 1);
}

#[test]
fn test_point_group_detection_symmetric_adamantane_electric_field_c3v() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/adamantane.xyz");
    let mut mol = Molecule::from_xyz(&path, 1e-6);
    mol.set_electric_field(Some(Vector3::new(0.1, 0.1, 0.1)));
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-6)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("C3v".to_owned()));
    assert_eq!(
        sym.get_elements(&ROT).expect("No proper elements found.")[&ElementOrder::Int(3)].len(),
        1
    );
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(1)].len(),
        3
    );
    assert_eq!(sym.get_sigma_elements("v").unwrap().len(), 3);

    assert_eq!(
        sym.get_generators(&ROT)
            .expect("No proper generators found.")[&ElementOrder::Int(3)]
            .len(),
        1
    );
    assert_eq!(
        sym.get_generators(&SIG)
            .expect("No improper generators found.")[&ElementOrder::Int(1)]
            .len(),
        1
    );
    assert_eq!(sym.get_sigma_generators("v").unwrap().len(), 1);
}

#[test]
fn test_point_group_detection_symmetric_ch4_electric_field_c3v() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/ch4.xyz");
    let mut mol = Molecule::from_xyz(&path, 1e-6);
    mol.set_electric_field(Some(Vector3::new(1.0, 1.0, 1.0)));
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-6)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("C3v".to_owned()));
    assert_eq!(
        sym.get_elements(&ROT).expect("No proper elements found.")[&ElementOrder::Int(3)].len(),
        1
    );
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(1)].len(),
        3
    );
    assert_eq!(sym.get_sigma_elements("v").unwrap().len(), 3);

    assert_eq!(
        sym.get_generators(&ROT)
            .expect("No proper generators found.")[&ElementOrder::Int(3)]
            .len(),
        1
    );
    assert_eq!(
        sym.get_generators(&SIG)
            .expect("No improper generators found.")[&ElementOrder::Int(1)]
            .len(),
        1
    );
    assert_eq!(sym.get_sigma_generators("v").unwrap().len(), 1);
}

#[test]
fn test_point_group_detection_symmetric_vf6_electric_field_c3v() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/vf6.xyz");
    let mut mol = Molecule::from_xyz(&path, 1e-12);
    mol.set_electric_field(Some(0.2 * Vector3::new(1.0, 1.0, 1.0)));
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-12)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("C3v".to_owned()));
    assert_eq!(
        sym.get_elements(&ROT).expect("No proper elements found.")[&ElementOrder::Int(3)].len(),
        1
    );
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(1)].len(),
        3
    );
    assert_eq!(sym.get_sigma_elements("v").unwrap().len(), 3);

    assert_eq!(
        sym.get_generators(&ROT)
            .expect("No proper generators found.")[&ElementOrder::Int(3)]
            .len(),
        1
    );
    assert_eq!(
        sym.get_generators(&SIG)
            .expect("No improper generators found.")[&ElementOrder::Int(1)]
            .len(),
        1
    );
    assert_eq!(sym.get_sigma_generators("v").unwrap().len(), 1);
}

#[test]
fn test_point_group_detection_symmetric_sf5cl_c4v() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/sf5cl.xyz");
    let mol = Molecule::from_xyz(&path, 1e-7);
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("C4v".to_owned()));
    assert_eq!(
        sym.get_elements(&ROT).expect("No proper elements found.")[&ElementOrder::Int(4)].len(),
        1
    );
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(1)].len(),
        4
    );
    assert_eq!(sym.get_sigma_elements("v").unwrap().len(), 4);

    assert_eq!(
        sym.get_generators(&ROT)
            .expect("No proper generators found.")[&ElementOrder::Int(4)]
            .len(),
        1
    );
    assert_eq!(
        sym.get_generators(&SIG)
            .expect("No improper generators found.")[&ElementOrder::Int(1)]
            .len(),
        1
    );
    assert_eq!(sym.get_sigma_generators("v").unwrap().len(), 1);
}

#[test]
fn test_point_group_detection_symmetric_h8_electric_field_c4v() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/h8.xyz");
    let mut mol = Molecule::from_xyz(&path, 1e-7);
    mol.set_electric_field(Some(0.2 * Vector3::new(0.0, 0.0, 1.0)));
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("C4v".to_owned()));
    assert_eq!(
        sym.get_elements(&ROT).expect("No proper elements found.")[&ElementOrder::Int(4)].len(),
        1
    );
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(1)].len(),
        4
    );
    assert_eq!(sym.get_sigma_elements("v").unwrap().len(), 4);

    assert_eq!(
        sym.get_generators(&ROT)
            .expect("No proper generators found.")[&ElementOrder::Int(4)]
            .len(),
        1
    );
    assert_eq!(
        sym.get_generators(&SIG)
            .expect("No improper generators found.")[&ElementOrder::Int(1)]
            .len(),
        1
    );
    assert_eq!(sym.get_sigma_generators("v").unwrap().len(), 1);
}

#[test]
fn test_point_group_detection_symmetric_vf6_electric_field_c4v() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/vf6.xyz");
    let mut mol = Molecule::from_xyz(&path, 1e-12);
    mol.set_electric_field(Some(0.2 * Vector3::new(1.0, 0.0, 0.0)));
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-12)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("C4v".to_owned()));
    assert_eq!(
        sym.get_elements(&ROT).expect("No proper elements found.")[&ElementOrder::Int(4)].len(),
        1
    );
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(1)].len(),
        4
    );
    assert_eq!(sym.get_sigma_elements("v").unwrap().len(), 4);

    assert_eq!(
        sym.get_generators(&ROT)
            .expect("No proper generators found.")[&ElementOrder::Int(4)]
            .len(),
        1
    );
    assert_eq!(
        sym.get_generators(&SIG)
            .expect("No improper generators found.")[&ElementOrder::Int(1)]
            .len(),
        1
    );
    assert_eq!(sym.get_sigma_generators("v").unwrap().len(), 1);
}

#[test]
fn test_point_group_detection_symmetric_antiprism_pb10_electric_field_c4v() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/pb10.xyz");
    let mut mol = Molecule::from_xyz(&path, 1e-7);
    mol.set_electric_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("C4v".to_owned()));
    assert_eq!(
        sym.get_elements(&ROT).expect("No proper elements found.")[&ElementOrder::Int(4)].len(),
        1
    );
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(1)].len(),
        4
    );
    assert_eq!(sym.get_sigma_elements("v").unwrap().len(), 4);

    assert_eq!(
        sym.get_generators(&ROT)
            .expect("No proper generators found.")[&ElementOrder::Int(4)]
            .len(),
        1
    );
    assert_eq!(
        sym.get_generators(&SIG)
            .expect("No improper generators found.")[&ElementOrder::Int(1)]
            .len(),
        1
    );
    assert_eq!(sym.get_sigma_generators("v").unwrap().len(), 1);
}

#[test]
fn test_point_group_detection_symmetric_cpnico_c5v() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/cpnico.xyz");
    let mol = Molecule::from_xyz(&path, 1e-7);
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("C5v".to_owned()));
    assert_eq!(
        sym.get_elements(&ROT).expect("No proper elements found.")[&ElementOrder::Int(5)].len(),
        1
    );
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(1)].len(),
        5
    );
    assert_eq!(sym.get_sigma_elements("v").unwrap().len(), 5);

    assert_eq!(
        sym.get_generators(&ROT)
            .expect("No proper generators found.")[&ElementOrder::Int(5)]
            .len(),
        1
    );
    assert_eq!(
        sym.get_generators(&SIG)
            .expect("No improper generators found.")[&ElementOrder::Int(1)]
            .len(),
        1
    );
    assert_eq!(sym.get_sigma_generators("v").unwrap().len(), 1);
}

#[test]
fn test_point_group_detection_symmetric_staggered_ferrocene_electric_field_c5v() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/staggered_ferrocene.xyz");
    let mut mol = Molecule::from_xyz(&path, 1e-6);
    mol.set_electric_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("C5v".to_owned()));
    assert_eq!(
        sym.get_elements(&ROT).expect("No proper elements found.")[&ElementOrder::Int(5)].len(),
        1
    );
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(1)].len(),
        5
    );
    assert_eq!(sym.get_sigma_elements("v").unwrap().len(), 5);

    assert_eq!(
        sym.get_generators(&ROT)
            .expect("No proper generators found.")[&ElementOrder::Int(5)]
            .len(),
        1
    );
    assert_eq!(
        sym.get_generators(&SIG)
            .expect("No improper generators found.")[&ElementOrder::Int(1)]
            .len(),
        1
    );
    assert_eq!(sym.get_sigma_generators("v").unwrap().len(), 1);
}

#[test]
fn test_point_group_detection_symmetric_c60_electric_field_c5v() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c60.xyz");
    let mut mol = Molecule::from_xyz(&path, 1e-6);
    mol.set_electric_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-6)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("C5v".to_owned()));
    assert_eq!(
        sym.get_elements(&ROT).expect("No proper elements found.")[&ElementOrder::Int(5)].len(),
        1
    );
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(1)].len(),
        5
    );
    assert_eq!(sym.get_sigma_elements("v").unwrap().len(), 5);

    assert_eq!(
        sym.get_generators(&ROT)
            .expect("No proper generators found.")[&ElementOrder::Int(5)]
            .len(),
        1
    );
    assert_eq!(
        sym.get_generators(&SIG)
            .expect("No improper generators found.")[&ElementOrder::Int(1)]
            .len(),
        1
    );
    assert_eq!(sym.get_sigma_generators("v").unwrap().len(), 1);
}

#[test]
fn test_point_group_detection_symmetric_b7_c6v() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/b7.xyz");
    let mol = Molecule::from_xyz(&path, 1e-6);
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-6)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("C6v".to_owned()));
    assert_eq!(
        sym.get_elements(&ROT).expect("No proper elements found.")[&ElementOrder::Int(6)].len(),
        1
    );
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(1)].len(),
        6
    );
    assert_eq!(sym.get_sigma_elements("v").unwrap().len(), 6);

    assert_eq!(
        sym.get_generators(&ROT)
            .expect("No proper generators found.")[&ElementOrder::Int(6)]
            .len(),
        1
    );
    assert_eq!(
        sym.get_generators(&SIG)
            .expect("No improper generators found.")[&ElementOrder::Int(1)]
            .len(),
        1
    );
    assert_eq!(sym.get_sigma_generators("v").unwrap().len(), 1);
}

#[test]
fn test_point_group_detection_symmetric_au26_electric_field_c6v() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/au26.xyz");
    let mut mol = Molecule::from_xyz(&path, 1e-6);
    mol.set_electric_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("C6v".to_owned()));
    assert_eq!(
        sym.get_elements(&ROT).expect("No proper elements found.")[&ElementOrder::Int(6)].len(),
        1
    );
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(1)].len(),
        6
    );
    assert_eq!(sym.get_sigma_elements("v").unwrap().len(), 6);

    assert_eq!(
        sym.get_generators(&ROT)
            .expect("No proper generators found.")[&ElementOrder::Int(6)]
            .len(),
        1
    );
    assert_eq!(
        sym.get_generators(&SIG)
            .expect("No improper generators found.")[&ElementOrder::Int(1)]
            .len(),
        1
    );
    assert_eq!(sym.get_sigma_generators("v").unwrap().len(), 1);
}

#[test]
fn test_point_group_detection_symmetric_benzene_electric_field_c6v() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/benzene.xyz");
    let mut mol = Molecule::from_xyz(&path, 1e-7);
    mol.set_electric_field(Some(Vector3::new(1.0, 0.0, 0.0)));
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("C6v".to_owned()));
    assert_eq!(
        sym.get_elements(&ROT).expect("No proper elements found.")[&ElementOrder::Int(6)].len(),
        1
    );
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(1)].len(),
        6
    );
    assert_eq!(sym.get_sigma_elements("v").unwrap().len(), 6);

    assert_eq!(
        sym.get_generators(&ROT)
            .expect("No proper generators found.")[&ElementOrder::Int(6)]
            .len(),
        1
    );
    assert_eq!(
        sym.get_generators(&SIG)
            .expect("No improper generators found.")[&ElementOrder::Int(1)]
            .len(),
        1
    );
    assert_eq!(sym.get_sigma_generators("v").unwrap().len(), 1);
}

#[test]
fn test_point_group_detection_symmetric_arbitrary_half_sandwich_cnv() {
    for n in 3..=32 {
        let mol = template_molecules::gen_arbitrary_half_sandwich(n);
        let presym = PreSymmetry::builder()
            .moi_threshold(1e-7)
            .molecule(&mol, true)
            .build()
            .unwrap();
        let mut sym = Symmetry::new();
        sym.analyse(&presym, false);
        assert_eq!(sym.point_group, Some(format!("C{n}v")));
        assert_eq!(
            sym.get_elements(&ROT).expect("No proper elements found.")[&ElementOrder::Int(n)].len(),
            1
        );
        assert_eq!(
            sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(1)]
                .len() as u32,
            n
        );
        assert_eq!(sym.get_sigma_elements("v").unwrap().len() as u32, n);

        assert_eq!(
            sym.get_generators(&ROT)
                .expect("No proper generators found.")[&ElementOrder::Int(n)]
                .len(),
            1
        );
        assert_eq!(
            sym.get_generators(&SIG)
                .expect("No improper generators found.")[&ElementOrder::Int(1)]
                .len(),
            1
        );
        assert_eq!(sym.get_sigma_generators("v").unwrap().len(), 1);
    }
}

#[test]
fn test_point_group_detection_symmetric_arbitrary_staggered_sandwich_electric_field_cnv() {
    // env_logger::init();
    for n in 3..=20 {
        let mut mol = template_molecules::gen_arbitrary_twisted_sandwich(n, 0.5);
        mol.set_electric_field(Some(Vector3::new(0.0, 0.0, 1.0)));
        let presym = PreSymmetry::builder()
            .moi_threshold(1e-7)
            .molecule(&mol, true)
            .build()
            .unwrap();
        let mut sym = Symmetry::new();
        sym.analyse(&presym, false);
        assert_eq!(sym.point_group, Some(format!("C{n}v")));
        assert_eq!(
            sym.get_elements(&ROT).expect("No proper elements found.")[&ElementOrder::Int(n)].len(),
            1
        );
        assert_eq!(
            sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(1)]
                .len() as u32,
            n
        );
        assert_eq!(sym.get_sigma_elements("v").unwrap().len() as u32, n);

        assert_eq!(
            sym.get_generators(&ROT)
                .expect("No proper generators found.")[&ElementOrder::Int(n)]
                .len(),
            1
        );
        assert_eq!(
            sym.get_generators(&SIG)
                .expect("No improper generators found.")[&ElementOrder::Int(1)]
                .len(),
            1
        );
        assert_eq!(sym.get_sigma_generators("v").unwrap().len(), 1);
    }
}

/*
Cnh
*/

#[test]
fn test_point_group_detection_symmetric_bf3_magnetic_field_c3h() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/bf3.xyz");
    let mut mol = Molecule::from_xyz(&path, 1e-7);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("C3h".to_owned()));
    assert_eq!(
        sym.get_elements(&ROT).expect("No proper elements found.")[&ElementOrder::Int(3)].len(),
        1
    );
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(1)].len(),
        1
    );
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(3)].len(),
        1
    );
    assert_eq!(sym.get_sigma_elements("h").unwrap().len(), 1);

    assert_eq!(
        sym.get_generators(&ROT)
            .expect("No proper generators found.")[&ElementOrder::Int(3)]
            .len(),
        1
    );
    assert_eq!(
        sym.get_generators(&SIG)
            .expect("No improper generators found.")[&ElementOrder::Int(1)]
            .len(),
        1
    );
    assert_eq!(sym.get_sigma_generators("h").unwrap().len(), 1);
}

#[test]
fn test_point_group_detection_symmetric_xef4_magnetic_field_c4h() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/xef4.xyz");
    let mut mol = Molecule::from_xyz(&path, 1e-7);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, -1.0)));
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("C4h".to_owned()));
    assert_eq!(
        sym.get_elements(&ROT).expect("No proper elements found.")[&ElementOrder::Int(4)].len(),
        1
    );
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(1)].len(),
        1
    );
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(2)].len(),
        1
    );
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(4)].len(),
        1
    );
    assert_eq!(sym.get_sigma_elements("h").unwrap().len(), 1);

    assert_eq!(
        sym.get_generators(&ROT)
            .expect("No proper generators found.")[&ElementOrder::Int(4)]
            .len(),
        1
    );
    assert_eq!(
        sym.get_generators(&SIG)
            .expect("No improper generators found.")[&ElementOrder::Int(1)]
            .len(),
        1
    );
    assert_eq!(sym.get_sigma_generators("h").unwrap().len(), 1);
}

#[test]
fn test_point_group_detection_symmetric_vf6_magnetic_field_c4h() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/vf6.xyz");
    let mut mol = Molecule::from_xyz(&path, 1e-12);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-12)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("C4h".to_owned()));
    assert_eq!(
        sym.get_elements(&ROT).expect("No proper elements found.")[&ElementOrder::Int(4)].len(),
        1
    );
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(1)].len(),
        1
    );
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(2)].len(),
        1
    );
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(4)].len(),
        1
    );
    assert_eq!(sym.get_sigma_elements("h").unwrap().len(), 1);

    assert_eq!(
        sym.get_generators(&ROT)
            .expect("No proper generators found.")[&ElementOrder::Int(4)]
            .len(),
        1
    );
    assert_eq!(
        sym.get_generators(&SIG)
            .expect("No improper generators found.")[&ElementOrder::Int(1)]
            .len(),
        1
    );
    assert_eq!(sym.get_sigma_generators("h").unwrap().len(), 1);
}

#[test]
fn test_point_group_detection_symmetric_h8_magnetic_field_c4h() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/h8.xyz");
    let mut mol = Molecule::from_xyz(&path, 1e-7);
    mol.set_magnetic_field(Some(0.2 * Vector3::new(0.0, 0.0, 1.0)));
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("C4h".to_owned()));
    assert_eq!(
        sym.get_elements(&ROT).expect("No proper elements found.")[&ElementOrder::Int(4)].len(),
        1
    );
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(1)].len(),
        1
    );
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(2)].len(),
        1
    );
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(4)].len(),
        1
    );
    assert_eq!(sym.get_sigma_elements("h").unwrap().len(), 1);

    assert_eq!(
        sym.get_generators(&ROT)
            .expect("No proper generators found.")[&ElementOrder::Int(4)]
            .len(),
        1
    );
    assert_eq!(
        sym.get_generators(&SIG)
            .expect("No improper generators found.")[&ElementOrder::Int(1)]
            .len(),
        1
    );
    assert_eq!(sym.get_sigma_generators("h").unwrap().len(), 1);
}

#[test]
fn test_point_group_detection_symmetric_eclipsed_ferrocene_magnetic_field_c5h() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/eclipsed_ferrocene.xyz");
    let mut mol = Molecule::from_xyz(&path, 1e-7);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, -1.0)));
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("C5h".to_owned()));
    assert_eq!(
        sym.get_elements(&ROT).expect("No proper elements found.")[&ElementOrder::Int(5)].len(),
        1
    );
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(1)].len(),
        1
    );
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(5)].len(),
        1
    );
    assert_eq!(sym.get_sigma_elements("h").unwrap().len(), 1);

    assert_eq!(
        sym.get_generators(&ROT)
            .expect("No proper generators found.")[&ElementOrder::Int(5)]
            .len(),
        1
    );
    assert_eq!(
        sym.get_generators(&SIG)
            .expect("No improper generators found.")[&ElementOrder::Int(1)]
            .len(),
        1
    );
    assert_eq!(sym.get_sigma_generators("h").unwrap().len(), 1);
}

#[test]
fn test_point_group_detection_symmetric_benzene_magnetic_field_c6h() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/benzene.xyz");
    let mut mol = Molecule::from_xyz(&path, 1e-7);
    mol.set_magnetic_field(Some(Vector3::new(1.0, 0.0, 0.0)));
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("C6h".to_owned()));
    assert_eq!(
        sym.get_elements(&ROT).expect("No proper elements found.")[&ElementOrder::Int(6)].len(),
        1
    );
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(1)].len(),
        1
    );
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(2)].len(),
        1
    );
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(6)].len(),
        1
    );
    assert_eq!(sym.get_sigma_elements("h").unwrap().len(), 1);

    assert_eq!(
        sym.get_generators(&ROT)
            .expect("No proper generators found.")[&ElementOrder::Int(6)]
            .len(),
        1
    );
    assert_eq!(
        sym.get_generators(&SIG)
            .expect("No improper generators found.")[&ElementOrder::Int(1)]
            .len(),
        1
    );
    assert_eq!(sym.get_sigma_generators("h").unwrap().len(), 1);
}

#[test]
fn test_point_group_detection_symmetric_arbitrary_eclipsed_sandwich_magnetic_field_cnh() {
    // env_logger::init();
    for n in 3..=20 {
        let mut mol = template_molecules::gen_arbitrary_eclipsed_sandwich(n);
        mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 0.1)));
        let presym = PreSymmetry::builder()
            .moi_threshold(1e-7)
            .molecule(&mol, true)
            .build()
            .unwrap();
        let mut sym = Symmetry::new();
        sym.analyse(&presym, false);
        assert_eq!(sym.point_group, Some(format!("C{n}h")));
        assert_eq!(
            sym.get_elements(&ROT).expect("No proper elements found.")[&ElementOrder::Int(n)].len(),
            1
        );
        assert_eq!(
            sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(1)]
                .len(),
            1
        );
        assert_eq!(
            sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(n)]
                .len(),
            1
        );
        assert_eq!(sym.get_sigma_elements("h").unwrap().len(), 1);
        if n % 2 == 0 {
            assert_eq!(
                sym.get_elements(&ROT).expect("No proper elements found.")[&ElementOrder::Int(2)]
                    .len(),
                1
            );
            assert_eq!(
                sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(2)]
                    .len(),
                1
            );
        };

        assert_eq!(
            sym.get_generators(&ROT)
                .expect("No proper generators found.")[&ElementOrder::Int(n)]
                .len(),
            1
        );
        assert_eq!(
            sym.get_generators(&SIG)
                .expect("No improper generators found.")[&ElementOrder::Int(1)]
                .len(),
            1
        );
        assert_eq!(sym.get_sigma_generators("h").unwrap().len(), 1);
    }
}

/*
Dn
*/

#[test]
fn test_point_group_detection_symmetric_triphenyl_radical_d3() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/triphenylradical.xyz");
    let mol = Molecule::from_xyz(&path, 1e-7);
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("D3".to_owned()));
    assert_eq!(
        sym.get_elements(&ROT).expect("No proper elements found.")[&ElementOrder::Int(3)].len(),
        1
    );
    assert_eq!(
        sym.get_elements(&ROT).expect("No proper elements found.")[&ElementOrder::Int(2)].len(),
        3
    );

    assert_eq!(
        sym.get_generators(&ROT)
            .expect("No proper generators found.")[&ElementOrder::Int(3)]
            .len(),
        1
    );
    assert_eq!(
        sym.get_generators(&ROT)
            .expect("No proper generators found.")[&ElementOrder::Int(2)]
            .len(),
        1
    );
}

#[test]
fn test_point_group_detection_symmetric_h8_twisted_d4() {
    let mol = template_molecules::gen_twisted_h8(0.1);
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("D4".to_owned()));
    assert_eq!(
        sym.get_elements(&ROT).expect("No proper elements found.")[&ElementOrder::Int(4)].len(),
        1
    );
    assert_eq!(
        sym.get_generators(&ROT)
            .expect("No proper generators found.")[&ElementOrder::Int(4)]
            .len(),
        1
    );
    assert_eq!(
        sym.get_elements(&ROT).expect("No proper elements found.")[&ElementOrder::Int(2)].len(),
        5
    );
    assert_eq!(
        sym.get_generators(&ROT)
            .expect("No proper generators found.")[&ElementOrder::Int(2)]
            .len(),
        1
    );
}

#[test]
fn test_point_group_detection_symmetric_c5ph5_d5() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c5ph5.xyz");
    let mol = Molecule::from_xyz(&path, 1e-7);
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("D5".to_owned()));
    assert_eq!(
        sym.get_elements(&ROT).expect("No proper elements found.")[&ElementOrder::Int(5)].len(),
        1
    );
    assert_eq!(
        sym.get_elements(&ROT).expect("No proper elements found.")[&ElementOrder::Int(2)].len(),
        5
    );

    assert_eq!(
        sym.get_generators(&ROT)
            .expect("No proper generators found.")[&ElementOrder::Int(5)]
            .len(),
        1
    );
    assert_eq!(
        sym.get_generators(&ROT)
            .expect("No proper generators found.")[&ElementOrder::Int(2)]
            .len(),
        1
    );
}

#[test]
fn test_point_group_detection_symmetric_c6ph6_d6() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c6ph6.xyz");
    let mol = Molecule::from_xyz(&path, 1e-6);
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("D6".to_owned()));
    assert_eq!(
        sym.get_elements(&ROT).expect("No proper elements found.")[&ElementOrder::Int(6)].len(),
        1
    );
    assert_eq!(
        sym.get_elements(&ROT).expect("No proper elements found.")[&ElementOrder::Int(2)].len(),
        7
    );

    assert_eq!(
        sym.get_generators(&ROT)
            .expect("No proper generators found.")[&ElementOrder::Int(6)]
            .len(),
        1
    );
    assert_eq!(
        sym.get_generators(&ROT)
            .expect("No proper generators found.")[&ElementOrder::Int(2)]
            .len(),
        1
    );
}

#[test]
fn test_point_group_detection_symmetric_arbitrary_twisted_sandwich_dn() {
    // env_logger::init();
    for n in 3..=20 {
        let mol = template_molecules::gen_arbitrary_twisted_sandwich(n, 0.1);
        let presym = PreSymmetry::builder()
            .moi_threshold(1e-7)
            .molecule(&mol, true)
            .build()
            .unwrap();
        let mut sym = Symmetry::new();
        sym.analyse(&presym, false);
        assert_eq!(sym.point_group, Some(format!("D{n}")));
        assert_eq!(
            sym.get_elements(&ROT).expect("No proper elements found.")[&ElementOrder::Int(n)].len(),
            1
        );
        if n % 2 == 0 {
            assert_eq!(
                sym.get_elements(&ROT).expect("No proper elements found.")[&ElementOrder::Int(2)]
                    .len() as u32,
                n + 1
            );
        } else {
            assert_eq!(
                sym.get_elements(&ROT).expect("No proper elements found.")[&ElementOrder::Int(2)]
                    .len() as u32,
                n
            );
        };

        assert_eq!(
            sym.get_generators(&ROT)
                .expect("No proper generators found.")[&ElementOrder::Int(n)]
                .len(),
            1
        );
        assert_eq!(
            sym.get_generators(&ROT)
                .expect("No proper generators found.")[&ElementOrder::Int(2)]
                .len(),
            1
        );
    }
}

/*
Dnh
*/

#[test]
fn test_point_group_detection_symmetric_bf3_d3h() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/bf3.xyz");
    let mol = Molecule::from_xyz(&path, 1e-7);
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("D3h".to_owned()));
    assert_eq!(
        sym.get_elements(&ROT).expect("No proper elements found.")[&ElementOrder::Int(3)].len(),
        1
    );
    assert_eq!(
        sym.get_elements(&ROT).expect("No proper elements found.")[&ElementOrder::Int(2)].len(),
        3
    );
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(1)].len(),
        4
    );
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(3)].len(),
        1
    );
    assert_eq!(sym.get_sigma_elements("h").unwrap().len(), 1);
    assert_eq!(sym.get_sigma_elements("v").unwrap().len(), 3);

    assert_eq!(
        sym.get_generators(&ROT)
            .expect("No proper generators found.")[&ElementOrder::Int(3)]
            .len(),
        1
    );
    assert_eq!(
        sym.get_generators(&ROT)
            .expect("No proper generators found.")[&ElementOrder::Int(2)]
            .len(),
        1
    );
    assert_eq!(
        sym.get_generators(&SIG)
            .expect("No improper generators found.")[&ElementOrder::Int(1)]
            .len(),
        1
    );
    assert_eq!(sym.get_sigma_generators("h").unwrap().len(), 1);
}

#[test]
fn test_point_group_detection_symmetric_xef4_d4h() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/xef4.xyz");
    let mol = Molecule::from_xyz(&path, 1e-7);
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("D4h".to_owned()));
    assert_eq!(
        sym.get_elements(&ROT).expect("No proper elements found.")[&ElementOrder::Int(4)].len(),
        1
    );
    assert_eq!(
        sym.get_elements(&ROT).expect("No proper elements found.")[&ElementOrder::Int(2)].len(),
        5
    );
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(1)].len(),
        5
    );
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(2)].len(),
        1
    );
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(4)].len(),
        1
    );
    assert_eq!(sym.get_sigma_elements("h").unwrap().len(), 1);
    assert_eq!(sym.get_sigma_elements("v").unwrap().len(), 4);

    assert_eq!(
        sym.get_generators(&ROT)
            .expect("No proper generators found.")[&ElementOrder::Int(4)]
            .len(),
        1
    );
    assert_eq!(
        sym.get_generators(&ROT)
            .expect("No proper generators found.")[&ElementOrder::Int(2)]
            .len(),
        1
    );
    assert_eq!(
        sym.get_generators(&SIG)
            .expect("No improper generators found.")[&ElementOrder::Int(1)]
            .len(),
        1
    );
    assert_eq!(sym.get_sigma_generators("h").unwrap().len(), 1);
}

#[test]
fn test_point_group_detection_symmetric_h8_d4h() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/h8.xyz");
    let mol = Molecule::from_xyz(&path, 1e-7);
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("D4h".to_owned()));
    assert_eq!(
        sym.get_elements(&ROT).expect("No proper elements found.")[&ElementOrder::Int(4)].len(),
        1
    );
    assert_eq!(
        sym.get_elements(&ROT).expect("No proper elements found.")[&ElementOrder::Int(2)].len(),
        5
    );
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(1)].len(),
        5
    );
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(2)].len(),
        1
    );
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(4)].len(),
        1
    );
    assert_eq!(sym.get_sigma_elements("h").unwrap().len(), 1);
    assert_eq!(sym.get_sigma_elements("v").unwrap().len(), 4);

    assert_eq!(
        sym.get_generators(&ROT)
            .expect("No proper generators found.")[&ElementOrder::Int(4)]
            .len(),
        1
    );
    assert_eq!(
        sym.get_generators(&ROT)
            .expect("No proper generators found.")[&ElementOrder::Int(2)]
            .len(),
        1
    );
    assert_eq!(
        sym.get_generators(&SIG)
            .expect("No improper generators found.")[&ElementOrder::Int(1)]
            .len(),
        1
    );
    assert_eq!(sym.get_sigma_generators("h").unwrap().len(), 1);
}

#[test]
fn test_point_group_detection_symmetric_eclipsed_ferrocene_d5h() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/eclipsed_ferrocene.xyz");
    let mol = Molecule::from_xyz(&path, 1e-7);
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("D5h".to_owned()));
    assert_eq!(
        sym.get_elements(&ROT).expect("No proper elements found.")[&ElementOrder::Int(5)].len(),
        1
    );
    assert_eq!(
        sym.get_elements(&ROT).expect("No proper elements found.")[&ElementOrder::Int(2)].len(),
        5
    );
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(1)].len(),
        6
    );
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(5)].len(),
        1
    );
    assert_eq!(sym.get_sigma_elements("h").unwrap().len(), 1);
    assert_eq!(sym.get_sigma_elements("v").unwrap().len(), 5);

    assert_eq!(
        sym.get_generators(&ROT)
            .expect("No proper generators found.")[&ElementOrder::Int(5)]
            .len(),
        1
    );
    assert_eq!(
        sym.get_generators(&ROT)
            .expect("No proper generators found.")[&ElementOrder::Int(2)]
            .len(),
        1
    );
    assert_eq!(
        sym.get_generators(&SIG)
            .expect("No improper generators found.")[&ElementOrder::Int(1)]
            .len(),
        1
    );
    assert_eq!(sym.get_sigma_generators("h").unwrap().len(), 1);
}

#[test]
fn test_point_group_detection_symmetric_benzene_d6h() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/benzene.xyz");
    let mol = Molecule::from_xyz(&path, 1e-7);
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("D6h".to_owned()));
    assert_eq!(
        sym.get_elements(&ROT).expect("No proper elements found.")[&ElementOrder::Int(6)].len(),
        1
    );
    assert_eq!(
        sym.get_elements(&ROT).expect("No proper elements found.")[&ElementOrder::Int(2)].len(),
        7
    );
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(1)].len(),
        7
    );
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(2)].len(),
        1
    );
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(6)].len(),
        1
    );
    assert_eq!(sym.get_sigma_elements("h").unwrap().len(), 1);
    assert_eq!(sym.get_sigma_elements("v").unwrap().len(), 6);

    assert_eq!(
        sym.get_generators(&ROT)
            .expect("No proper generators found.")[&ElementOrder::Int(6)]
            .len(),
        1
    );
    assert_eq!(
        sym.get_generators(&ROT)
            .expect("No proper generators found.")[&ElementOrder::Int(2)]
            .len(),
        1
    );
    assert_eq!(
        sym.get_generators(&SIG)
            .expect("No improper generators found.")[&ElementOrder::Int(1)]
            .len(),
        1
    );
    assert_eq!(sym.get_sigma_generators("h").unwrap().len(), 1);
}

#[test]
fn test_point_group_detection_symmetric_h100_d100h() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/h100.xyz");
    let mol = Molecule::from_xyz(&path, 1e-6);
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("D100h".to_owned()));
    assert_eq!(
        sym.get_elements(&ROT).expect("No proper elements found.")[&ElementOrder::Int(100)].len(),
        1
    );
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(2)].len(),
        1
    );
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(100)].len(),
        1
    );
    assert_eq!(sym.get_sigma_elements("h").unwrap().len(), 1);
    assert_eq!(sym.get_sigma_elements("v").unwrap().len(), 100);

    assert_eq!(
        sym.get_generators(&ROT)
            .expect("No proper generators found.")[&ElementOrder::Int(100)]
            .len(),
        1
    );
    assert_eq!(
        sym.get_generators(&ROT)
            .expect("No proper generators found.")[&ElementOrder::Int(2)]
            .len(),
        1
    );
    assert_eq!(
        sym.get_generators(&SIG)
            .expect("No improper generators found.")[&ElementOrder::Int(1)]
            .len(),
        1
    );
    assert_eq!(sym.get_sigma_generators("h").unwrap().len(), 1);
}

#[test]
fn test_point_group_detection_symmetric_arbitrary_eclipsed_sandwich_dnh() {
    // env_logger::init();
    for n in 3..=20 {
        let mol = template_molecules::gen_arbitrary_eclipsed_sandwich(n);
        let presym = PreSymmetry::builder()
            .moi_threshold(1e-7)
            .molecule(&mol, true)
            .build()
            .unwrap();
        let mut sym = Symmetry::new();
        sym.analyse(&presym, false);
        assert_eq!(sym.point_group, Some(format!("D{n}h")));
        assert_eq!(
            sym.get_elements(&ROT).expect("No proper elements found.")[&ElementOrder::Int(n)].len(),
            1
        );
        assert_eq!(
            sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(1)]
                .len() as u32,
            n + 1
        );
        assert_eq!(
            sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(n)]
                .len(),
            1
        );
        assert_eq!(sym.get_sigma_elements("h").unwrap().len() as u32, 1);
        assert_eq!(sym.get_sigma_elements("v").unwrap().len() as u32, n);
        if n % 2 == 0 {
            assert_eq!(
                sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(2)]
                    .len(),
                1
            );
            assert_eq!(
                sym.get_elements(&ROT).expect("No proper elements found.")[&ElementOrder::Int(2)]
                    .len() as u32,
                n + 1
            );
        } else {
        }

        assert_eq!(
            sym.get_generators(&ROT)
                .expect("No proper generators found.")[&ElementOrder::Int(n)]
                .len(),
            1
        );
        assert_eq!(
            sym.get_generators(&ROT)
                .expect("No proper generators found.")[&ElementOrder::Int(2)]
                .len(),
            1
        );
        assert_eq!(
            sym.get_generators(&SIG)
                .expect("No improper generators found.")[&ElementOrder::Int(1)]
                .len(),
            1
        );
        assert_eq!(sym.get_sigma_generators("h").unwrap().len(), 1);
    }
}

/*
Dnd
*/

#[test]
fn test_point_group_detection_symmetric_b2cl4_d2d() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/b2cl4.xyz");
    let mol = Molecule::from_xyz(&path, 1e-7);
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("D2d".to_owned()));
    assert_eq!(
        sym.get_elements(&ROT).expect("No proper elements found.")[&ElementOrder::Int(2)].len(),
        3
    );
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(1)].len(),
        2
    );
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(4)].len(),
        1
    );
    assert_eq!(sym.get_sigma_elements("d").unwrap().len(), 2);

    assert_eq!(
        sym.get_generators(&ROT)
            .expect("No proper generators found.")[&ElementOrder::Int(2)]
            .len(),
        2
    );
    assert_eq!(
        sym.get_generators(&SIG)
            .expect("No improper generators found.")[&ElementOrder::Int(1)]
            .len(),
        1
    );
    assert_eq!(sym.get_sigma_generators("d").unwrap().len(), 1);
}

#[test]
fn test_point_group_detection_symmetric_s4n4_d2d() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/s4n4.xyz");
    let mol = Molecule::from_xyz(&path, 1e-7);
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("D2d".to_owned()));
    assert_eq!(
        sym.get_elements(&ROT).expect("No proper elements found.")[&ElementOrder::Int(2)].len(),
        3
    );
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(1)].len(),
        2
    );
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(4)].len(),
        1
    );
    assert_eq!(sym.get_sigma_elements("d").unwrap().len(), 2);

    assert_eq!(
        sym.get_generators(&ROT)
            .expect("No proper generators found.")[&ElementOrder::Int(2)]
            .len(),
        2
    );
    assert_eq!(
        sym.get_generators(&SIG)
            .expect("No improper generators found.")[&ElementOrder::Int(1)]
            .len(),
        1
    );
    assert_eq!(sym.get_sigma_generators("d").unwrap().len(), 1);
}

#[test]
fn test_point_group_detection_symmetric_pbet4_d2d() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/pbet4.xyz");
    let mol = Molecule::from_xyz(&path, 1e-7);
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("D2d".to_owned()));
    assert_eq!(
        sym.get_elements(&ROT).expect("No proper elements found.")[&ElementOrder::Int(2)].len(),
        3
    );
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(1)].len(),
        2
    );
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(4)].len(),
        1
    );
    assert_eq!(sym.get_sigma_elements("d").unwrap().len(), 2);

    assert_eq!(
        sym.get_generators(&ROT)
            .expect("No proper generators found.")[&ElementOrder::Int(2)]
            .len(),
        2
    );
    assert_eq!(
        sym.get_generators(&SIG)
            .expect("No improper generators found.")[&ElementOrder::Int(1)]
            .len(),
        1
    );
    assert_eq!(sym.get_sigma_generators("d").unwrap().len(), 1);
}

#[test]
fn test_point_group_detection_symmetric_allene_d2d() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/allene.xyz");
    let mol = Molecule::from_xyz(&path, 1e-7);
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("D2d".to_owned()));
    assert_eq!(
        sym.get_elements(&ROT).expect("No proper elements found.")[&ElementOrder::Int(2)].len(),
        3
    );
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(1)].len(),
        2
    );
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(4)].len(),
        1
    );
    assert_eq!(sym.get_sigma_elements("d").unwrap().len(), 2);

    assert_eq!(
        sym.get_generators(&ROT)
            .expect("No proper generators found.")[&ElementOrder::Int(2)]
            .len(),
        2
    );
    assert_eq!(
        sym.get_generators(&SIG)
            .expect("No improper generators found.")[&ElementOrder::Int(1)]
            .len(),
        1
    );
    assert_eq!(sym.get_sigma_generators("d").unwrap().len(), 1);
}

#[test]
fn test_point_group_detection_symmetric_staggered_c2h6_d3d() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c2h6.xyz");
    let mol = Molecule::from_xyz(&path, 1e-7);
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("D3d".to_owned()));
    assert_eq!(
        sym.get_elements(&ROT).expect("No proper elements found.")[&ElementOrder::Int(3)].len(),
        1
    );
    assert_eq!(
        sym.get_elements(&ROT).expect("No proper elements found.")[&ElementOrder::Int(2)].len(),
        3
    );
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(1)].len(),
        3
    );
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(2)].len(),
        1
    );
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(6)].len(),
        1
    );
    assert_eq!(sym.get_sigma_elements("d").unwrap().len(), 3);

    assert_eq!(
        sym.get_generators(&ROT)
            .expect("No proper generators found.")[&ElementOrder::Int(3)]
            .len(),
        1
    );
    assert_eq!(
        sym.get_generators(&ROT)
            .expect("No proper generators found.")[&ElementOrder::Int(2)]
            .len(),
        1
    );
    assert_eq!(
        sym.get_generators(&SIG)
            .expect("No improper generators found.")[&ElementOrder::Int(1)]
            .len(),
        1
    );
    assert_eq!(sym.get_sigma_generators("d").unwrap().len(), 1);
}

#[test]
fn test_point_group_detection_symmetric_cyclohexane_chair_d3d() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/cyclohexane_chair.xyz");
    let mol = Molecule::from_xyz(&path, 1e-7);
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("D3d".to_owned()));
    assert_eq!(
        sym.get_elements(&ROT).expect("No proper elements found.")[&ElementOrder::Int(3)].len(),
        1
    );
    assert_eq!(
        sym.get_elements(&ROT).expect("No proper elements found.")[&ElementOrder::Int(2)].len(),
        3
    );
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(1)].len(),
        3
    );
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(2)].len(),
        1
    );
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(6)].len(),
        1
    );
    assert_eq!(sym.get_sigma_elements("d").unwrap().len(), 3);

    assert_eq!(
        sym.get_generators(&ROT)
            .expect("No proper generators found.")[&ElementOrder::Int(3)]
            .len(),
        1
    );
    assert_eq!(
        sym.get_generators(&ROT)
            .expect("No proper generators found.")[&ElementOrder::Int(2)]
            .len(),
        1
    );
    assert_eq!(
        sym.get_generators(&SIG)
            .expect("No improper generators found.")[&ElementOrder::Int(1)]
            .len(),
        1
    );
    assert_eq!(sym.get_sigma_generators("d").unwrap().len(), 1);
}

#[test]
fn test_point_group_detection_symmetric_s8_d4d() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/s8.xyz");
    let mol = Molecule::from_xyz(&path, 1e-7);
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("D4d".to_owned()));
    assert_eq!(
        sym.get_elements(&ROT).expect("No proper elements found.")[&ElementOrder::Int(4)].len(),
        1
    );
    assert_eq!(
        sym.get_elements(&ROT).expect("No proper elements found.")[&ElementOrder::Int(2)].len(),
        5
    );
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(1)].len(),
        4
    );
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(8)].len(),
        1
    );
    assert_eq!(sym.get_sigma_elements("d").unwrap().len(), 4);

    assert_eq!(
        sym.get_generators(&ROT)
            .expect("No proper generators found.")[&ElementOrder::Int(4)]
            .len(),
        1
    );
    assert_eq!(
        sym.get_generators(&ROT)
            .expect("No proper generators found.")[&ElementOrder::Int(2)]
            .len(),
        1
    );
    assert_eq!(
        sym.get_generators(&SIG)
            .expect("No improper generators found.")[&ElementOrder::Int(1)]
            .len(),
        1
    );
    assert_eq!(sym.get_sigma_generators("d").unwrap().len(), 1);
}

#[test]
fn test_point_group_detection_symmetric_antiprism_h8_d4d() {
    let mol = template_molecules::gen_twisted_h8(std::f64::consts::FRAC_PI_4);
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("D4d".to_owned()));
    assert_eq!(
        sym.get_elements(&ROT).expect("No proper elements found.")[&ElementOrder::Int(4)].len(),
        1
    );
    assert_eq!(
        sym.get_elements(&ROT).expect("No proper elements found.")[&ElementOrder::Int(2)].len(),
        5
    );
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(1)].len(),
        4
    );
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(8)].len(),
        1
    );
    assert_eq!(sym.get_sigma_elements("d").unwrap().len(), 4);

    assert_eq!(
        sym.get_generators(&ROT)
            .expect("No proper generators found.")[&ElementOrder::Int(4)]
            .len(),
        1
    );
    assert_eq!(
        sym.get_generators(&ROT)
            .expect("No proper generators found.")[&ElementOrder::Int(2)]
            .len(),
        1
    );
    assert_eq!(
        sym.get_generators(&SIG)
            .expect("No improper generators found.")[&ElementOrder::Int(1)]
            .len(),
        1
    );
    assert_eq!(sym.get_sigma_generators("d").unwrap().len(), 1);
}

#[test]
fn test_point_group_detection_symmetric_antiprism_pb10_d4d() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/pb10.xyz");
    let mol = Molecule::from_xyz(&path, 1e-7);
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("D4d".to_owned()));
    assert_eq!(
        sym.get_elements(&ROT).expect("No proper elements found.")[&ElementOrder::Int(4)].len(),
        1
    );
    assert_eq!(
        sym.get_elements(&ROT).expect("No proper elements found.")[&ElementOrder::Int(2)].len(),
        5
    );
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(1)].len(),
        4
    );
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(8)].len(),
        1
    );
    assert_eq!(sym.get_sigma_elements("d").unwrap().len(), 4);

    assert_eq!(
        sym.get_generators(&ROT)
            .expect("No proper generators found.")[&ElementOrder::Int(4)]
            .len(),
        1
    );
    assert_eq!(
        sym.get_generators(&ROT)
            .expect("No proper generators found.")[&ElementOrder::Int(2)]
            .len(),
        1
    );
    assert_eq!(
        sym.get_generators(&SIG)
            .expect("No improper generators found.")[&ElementOrder::Int(1)]
            .len(),
        1
    );
    assert_eq!(sym.get_sigma_generators("d").unwrap().len(), 1);
}

#[test]
fn test_point_group_detection_symmetric_staggered_ferrocene_d5d() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/staggered_ferrocene.xyz");
    let mol = Molecule::from_xyz(&path, 1e-6);
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("D5d".to_owned()));
    assert_eq!(
        sym.get_elements(&ROT).expect("No proper elements found.")[&ElementOrder::Int(5)].len(),
        1
    );
    assert_eq!(
        sym.get_elements(&ROT).expect("No proper elements found.")[&ElementOrder::Int(2)].len(),
        5
    );
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(1)].len(),
        5
    );
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(2)].len(),
        1
    );
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(10)].len(),
        1
    );
    assert_eq!(sym.get_sigma_elements("d").unwrap().len(), 5);

    assert_eq!(
        sym.get_generators(&ROT)
            .expect("No proper generators found.")[&ElementOrder::Int(5)]
            .len(),
        1
    );
    assert_eq!(
        sym.get_generators(&ROT)
            .expect("No proper generators found.")[&ElementOrder::Int(2)]
            .len(),
        1
    );
    assert_eq!(
        sym.get_generators(&SIG)
            .expect("No improper generators found.")[&ElementOrder::Int(1)]
            .len(),
        1
    );
    assert_eq!(sym.get_sigma_generators("d").unwrap().len(), 1);
}

#[test]
fn test_point_group_detection_symmetric_au26_d6d() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/au26.xyz");
    let mol = Molecule::from_xyz(&path, 1e-6);
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("D6d".to_owned()));
    assert_eq!(
        sym.get_elements(&ROT).expect("No proper elements found.")[&ElementOrder::Int(6)].len(),
        1
    );
    assert_eq!(
        sym.get_elements(&ROT).expect("No proper elements found.")[&ElementOrder::Int(2)].len(),
        7
    );
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(1)].len(),
        6
    );
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(12)].len(),
        1
    );
    assert_eq!(sym.get_sigma_elements("d").unwrap().len(), 6);

    assert_eq!(
        sym.get_generators(&ROT)
            .expect("No proper generators found.")[&ElementOrder::Int(6)]
            .len(),
        1
    );
    assert_eq!(
        sym.get_generators(&ROT)
            .expect("No proper generators found.")[&ElementOrder::Int(2)]
            .len(),
        1
    );
    assert_eq!(
        sym.get_generators(&SIG)
            .expect("No improper generators found.")[&ElementOrder::Int(1)]
            .len(),
        1
    );
    assert_eq!(sym.get_sigma_generators("d").unwrap().len(), 1);
}

#[test]
fn test_point_group_detection_symmetric_arbitrary_staggered_sandwich_dnd() {
    // env_logger::init();
    for n in 3..=20 {
        let mol = template_molecules::gen_arbitrary_twisted_sandwich(n, 0.5);
        let presym = PreSymmetry::builder()
            .moi_threshold(1e-7)
            .molecule(&mol, true)
            .build()
            .unwrap();
        let mut sym = Symmetry::new();
        sym.analyse(&presym, false);
        assert_eq!(sym.point_group, Some(format!("D{n}d")));
        assert_eq!(
            sym.get_elements(&ROT).expect("No proper elements found.")[&ElementOrder::Int(n)].len(),
            1
        );
        assert_eq!(
            sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(1)]
                .len() as u32,
            n
        );
        assert_eq!(
            sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(2 * n)]
                .len(),
            1
        );
        assert_eq!(sym.get_sigma_elements("d").unwrap().len() as u32, n);
        if n % 2 == 0 {
            assert_eq!(
                sym.get_elements(&ROT).expect("No proper elements found.")[&ElementOrder::Int(2)]
                    .len() as u32,
                n + 1
            );
        } else {
            assert_eq!(
                sym.get_elements(&ROT).expect("No proper elements found.")[&ElementOrder::Int(2)]
                    .len() as u32,
                n
            );
            assert_eq!(
                sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(2)]
                    .len(),
                1
            );
        };

        assert_eq!(
            sym.get_generators(&ROT)
                .expect("No proper generators found.")[&ElementOrder::Int(n)]
                .len(),
            1
        );
        assert_eq!(
            sym.get_generators(&ROT)
                .expect("No proper generators found.")[&ElementOrder::Int(2)]
                .len(),
            1
        );
        assert_eq!(
            sym.get_generators(&SIG)
                .expect("No improper generators found.")[&ElementOrder::Int(1)]
                .len(),
            1
        );
        assert_eq!(sym.get_sigma_generators("d").unwrap().len(), 1);
    }
}

/*
S2n
*/

#[test]
fn test_point_group_detection_symmetric_b2cl4_magnetic_field_s4() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/b2cl4.xyz");
    let mut mol = Molecule::from_xyz(&path, 1e-7);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("S4".to_owned()));
    assert_eq!(
        sym.get_elements(&ROT).expect("No proper elements found.")[&ElementOrder::Int(2)].len(),
        1
    );
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(4)].len(),
        1
    );

    assert_eq!(
        sym.get_generators(&SIG)
            .expect("No improper generators found.")[&ElementOrder::Int(4)]
            .len(),
        1
    );
}

#[test]
fn test_point_group_detection_symmetric_adamantane_magnetic_field_s4() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/adamantane.xyz");
    let mut mol = Molecule::from_xyz(&path, 1e-6);
    mol.set_magnetic_field(Some(Vector3::new(-0.1, 0.0, 0.0)));
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-6)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("S4".to_owned()));
    assert_eq!(
        sym.get_elements(&ROT).expect("No proper elements found.")[&ElementOrder::Int(2)].len(),
        1
    );
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(4)].len(),
        1
    );

    assert_eq!(
        sym.get_generators(&SIG)
            .expect("No improper generators found.")[&ElementOrder::Int(4)]
            .len(),
        1
    );
}

#[test]
fn test_point_group_detection_symmetric_ch4_magnetic_field_s4() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/ch4.xyz");
    let mut mol = Molecule::from_xyz(&path, 1e-6);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-6)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("S4".to_owned()));
    assert_eq!(
        sym.get_elements(&ROT).expect("No proper elements found.")[&ElementOrder::Int(2)].len(),
        1
    );
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(4)].len(),
        1
    );

    assert_eq!(
        sym.get_generators(&SIG)
            .expect("No improper generators found.")[&ElementOrder::Int(4)]
            .len(),
        1
    );
}

#[test]
fn test_point_group_detection_symmetric_65coronane_s6() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/coronane65.xyz");
    let mol = Molecule::from_xyz(&path, 1e-7);
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("S6".to_owned()));
    assert_eq!(
        sym.get_elements(&ROT).expect("No proper elements found.")[&ElementOrder::Int(3)].len(),
        1
    );
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(6)].len(),
        1
    );
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(2)].len(),
        1
    );

    assert_eq!(
        sym.get_generators(&SIG)
            .expect("No improper generators found.")[&ElementOrder::Int(6)]
            .len(),
        1
    );
}

#[test]
fn test_point_group_detection_symmetric_65coronane_magnetic_field_s6() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/coronane65.xyz");
    let mut mol = Molecule::from_xyz(&path, 1e-7);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, -1.0)));
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("S6".to_owned()));
    assert_eq!(
        sym.get_elements(&ROT).expect("No proper elements found.")[&ElementOrder::Int(3)].len(),
        1
    );
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(6)].len(),
        1
    );
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(2)].len(),
        1
    );

    assert_eq!(
        sym.get_generators(&SIG)
            .expect("No improper generators found.")[&ElementOrder::Int(6)]
            .len(),
        1
    );
}

#[test]
fn test_point_group_detection_symmetric_staggered_c2h6_magnetic_field_s6() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c2h6.xyz");
    let mut mol = Molecule::from_xyz(&path, 1e-7);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("S6".to_owned()));
    assert_eq!(
        sym.get_elements(&ROT).expect("No proper elements found.")[&ElementOrder::Int(3)].len(),
        1
    );
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(6)].len(),
        1
    );
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(2)].len(),
        1
    );

    assert_eq!(
        sym.get_generators(&SIG)
            .expect("No improper generators found.")[&ElementOrder::Int(6)]
            .len(),
        1
    );
}

#[test]
fn test_point_group_detection_symmetric_c60_magnetic_field_s6() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c60.xyz");
    let mut mol = Molecule::from_xyz(&path, 1e-5);
    mol.set_magnetic_field(Some(Vector3::new(
        -0.5773503107731,
        -0.1875926572335,
        0.7946543988441,
    )));
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-6)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("S6".to_owned()));
    assert_eq!(
        sym.get_elements(&ROT).expect("No proper elements found.")[&ElementOrder::Int(3)].len(),
        1
    );
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(6)].len(),
        1
    );
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(2)].len(),
        1
    );

    assert_eq!(
        sym.get_generators(&SIG)
            .expect("No improper generators found.")[&ElementOrder::Int(6)]
            .len(),
        1
    );
}

#[test]
fn test_point_group_detection_symmetric_vh2o6_magnetic_field_s6() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/vh2o6.xyz");
    let mut mol = Molecule::from_xyz(&path, 1e-12);
    mol.set_magnetic_field(Some(Vector3::new(-0.2, 0.2, 0.2)));
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-12)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("S6".to_owned()));
    assert_eq!(
        sym.get_elements(&ROT).expect("No proper elements found.")[&ElementOrder::Int(3)].len(),
        1
    );
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(6)].len(),
        1
    );
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(2)].len(),
        1
    );

    assert_eq!(
        sym.get_generators(&SIG)
            .expect("No improper generators found.")[&ElementOrder::Int(6)]
            .len(),
        1
    );
}

#[test]
fn test_point_group_detection_symmetric_vf6_magnetic_field_s6() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/vf6.xyz");
    let mut mol = Molecule::from_xyz(&path, 1e-12);
    mol.set_magnetic_field(Some(Vector3::new(1.0, 1.0, 1.0)));
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-12)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("S6".to_owned()));
    assert_eq!(
        sym.get_elements(&ROT).expect("No proper elements found.")[&ElementOrder::Int(3)].len(),
        1
    );
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(6)].len(),
        1
    );
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(2)].len(),
        1
    );

    assert_eq!(
        sym.get_generators(&SIG)
            .expect("No improper generators found.")[&ElementOrder::Int(6)]
            .len(),
        1
    );
}

#[test]
fn test_point_group_detection_symmetric_s8_magnetic_field_s8() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/s8.xyz");
    let mut mol = Molecule::from_xyz(&path, 1e-7);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("S8".to_owned()));
    assert_eq!(
        sym.get_elements(&ROT).expect("No proper elements found.")[&ElementOrder::Int(4)].len(),
        1
    );
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(8)].len(),
        1
    );

    assert_eq!(
        sym.get_generators(&SIG)
            .expect("No improper generators found.")[&ElementOrder::Int(8)]
            .len(),
        1
    );
}

#[test]
fn test_point_group_detection_symmetric_antiprism_pb10_magnetic_field_s8() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/pb10.xyz");
    let mut mol = Molecule::from_xyz(&path, 1e-7);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("S8".to_owned()));
    assert_eq!(
        sym.get_elements(&ROT).expect("No proper elements found.")[&ElementOrder::Int(4)].len(),
        1
    );
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(8)].len(),
        1
    );

    assert_eq!(
        sym.get_generators(&SIG)
            .expect("No improper generators found.")[&ElementOrder::Int(8)]
            .len(),
        1
    );
}

#[test]
fn test_point_group_detection_symmetric_staggered_ferrocene_magnetic_field_s10() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/staggered_ferrocene.xyz");
    let mut mol = Molecule::from_xyz(&path, 1e-6);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("S10".to_owned()));
    assert_eq!(
        sym.get_elements(&ROT).expect("No proper elements found.")[&ElementOrder::Int(5)].len(),
        1
    );
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(10)].len(),
        1
    );
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(2)].len(),
        1
    );

    assert_eq!(
        sym.get_generators(&SIG)
            .expect("No improper generators found.")[&ElementOrder::Int(10)]
            .len(),
        1
    );
}

#[test]
fn test_point_group_detection_symmetric_c60_magnetic_field_s10() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c60.xyz");
    let mut mol = Molecule::from_xyz(&path, 1e-5);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-6)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("S10".to_owned()));
    assert_eq!(
        sym.get_elements(&ROT).expect("No proper elements found.")[&ElementOrder::Int(5)].len(),
        1
    );
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(10)].len(),
        1
    );
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(2)].len(),
        1
    );

    assert_eq!(
        sym.get_generators(&SIG)
            .expect("No improper generators found.")[&ElementOrder::Int(10)]
            .len(),
        1
    );
}

#[test]
fn test_point_group_detection_symmetric_au26_magnetic_field_s12() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/au26.xyz");
    let mut mol = Molecule::from_xyz(&path, 1e-6);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("S12".to_owned()));
    assert_eq!(
        sym.get_elements(&ROT).expect("No proper elements found.")[&ElementOrder::Int(6)].len(),
        1
    );
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(12)].len(),
        1
    );

    assert_eq!(
        sym.get_generators(&SIG)
            .expect("No improper generators found.")[&ElementOrder::Int(12)]
            .len(),
        1
    );
}

#[test]
fn test_point_group_detection_symmetric_arbitrary_staggered_sandwich_magnetic_field_s2n() {
    // env_logger::init();
    for n in 3..=20 {
        let mut mol = template_molecules::gen_arbitrary_twisted_sandwich(n, 0.5);
        mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
        let presym = PreSymmetry::builder()
            .moi_threshold(1e-7)
            .molecule(&mol, true)
            .build()
            .unwrap();
        let mut sym = Symmetry::new();
        sym.analyse(&presym, false);
        assert_eq!(sym.point_group, Some(format!("S{}", 2 * n)));
        assert_eq!(
            sym.get_elements(&ROT).expect("No proper elements found.")[&ElementOrder::Int(n)].len(),
            1
        );
        assert_eq!(
            sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(2 * n)]
                .len(),
            1
        );
        if n % 2 == 1 {
            assert_eq!(
                sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(2)]
                    .len(),
                1
            );
        };

        assert_eq!(
            sym.get_generators(&SIG)
                .expect("No improper generators found.")[&ElementOrder::Int(2 * n)]
                .len(),
            1
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
fn test_point_group_detection_asymmetric_spiroketal_c2() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/spiroketal.xyz");
    let mol = Molecule::from_xyz(&path, 1e-7);
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("C2".to_owned()));
    assert_eq!(
        sym.get_elements(&ROT).expect("No proper elements found.")[&ElementOrder::Int(2)].len(),
        1
    );
    assert_eq!(
        sym.get_generators(&ROT)
            .expect("No proper generators found.")[&ElementOrder::Int(2)]
            .len(),
        1
    );
}

#[test]
fn test_point_group_detection_asymmetric_cyclohexene_c2() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/cyclohexene.xyz");
    let mol = Molecule::from_xyz(&path, 1e-7);
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("C2".to_owned()));
    assert_eq!(
        sym.get_elements(&ROT).expect("No proper elements found.")[&ElementOrder::Int(2)].len(),
        1
    );
    assert_eq!(
        sym.get_generators(&ROT)
            .expect("No proper generators found.")[&ElementOrder::Int(2)]
            .len(),
        1
    );
}

#[test]
fn test_point_group_detection_asymmetric_thf_c2() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/thf.xyz");
    let mol = Molecule::from_xyz(&path, 1e-7);
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("C2".to_owned()));
    assert_eq!(
        sym.get_elements(&ROT).expect("No proper elements found.")[&ElementOrder::Int(2)].len(),
        1
    );
    assert_eq!(
        sym.get_generators(&ROT)
            .expect("No proper generators found.")[&ElementOrder::Int(2)]
            .len(),
        1
    );
}

#[test]
fn test_point_group_detection_asymmetric_tartaricacid_c2() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/tartaricacid.xyz");
    let mol = Molecule::from_xyz(&path, 1e-7);
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("C2".to_owned()));
    assert_eq!(
        sym.get_elements(&ROT).expect("No proper elements found.")[&ElementOrder::Int(2)].len(),
        1
    );
    assert_eq!(
        sym.get_generators(&ROT)
            .expect("No proper generators found.")[&ElementOrder::Int(2)]
            .len(),
        1
    );
}

#[test]
fn test_point_group_detection_asymmetric_f2allene_c2() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/f2allene.xyz");
    let mol = Molecule::from_xyz(&path, 1e-7);
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("C2".to_owned()));
    assert_eq!(
        sym.get_elements(&ROT).expect("No proper elements found.")[&ElementOrder::Int(2)].len(),
        1
    );
    assert_eq!(
        sym.get_generators(&ROT)
            .expect("No proper generators found.")[&ElementOrder::Int(2)]
            .len(),
        1
    );
}

#[test]
fn test_point_group_detection_asymmetric_water_magnetic_field_c2() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/water.xyz");
    let mut mol = Molecule::from_xyz(&path, 1e-7);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 1.0, 0.0)));
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("C2".to_owned()));
    assert_eq!(
        sym.get_elements(&ROT).expect("No proper elements found.")[&ElementOrder::Int(2)].len(),
        1
    );
    assert_eq!(
        sym.get_generators(&ROT)
            .expect("No proper generators found.")[&ElementOrder::Int(2)]
            .len(),
        1
    );
}

#[test]
fn test_point_group_detection_asymmetric_pyridine_magnetic_field_c2() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/pyridine.xyz");
    let mut mol = Molecule::from_xyz(&path, 1e-7);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.2, 0.0)));
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("C2".to_owned()));
    assert_eq!(
        sym.get_elements(&ROT).expect("No proper elements found.")[&ElementOrder::Int(2)].len(),
        1
    );
    assert_eq!(
        sym.get_generators(&ROT)
            .expect("No proper generators found.")[&ElementOrder::Int(2)]
            .len(),
        1
    );
}

#[test]
fn test_point_group_detection_asymmetric_cyclobutene_magnetic_field_c2() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/cyclobutene.xyz");
    let mut mol = Molecule::from_xyz(&path, 1e-7);
    mol.set_magnetic_field(Some(Vector3::new(0.2, 0.0, 0.0)));
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("C2".to_owned()));
    assert_eq!(
        sym.get_elements(&ROT).expect("No proper elements found.")[&ElementOrder::Int(2)].len(),
        1
    );
    assert_eq!(
        sym.get_generators(&ROT)
            .expect("No proper generators found.")[&ElementOrder::Int(2)]
            .len(),
        1
    );
}

#[test]
fn test_point_group_detection_asymmetric_azulene_magnetic_field_c2() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/azulene.xyz");
    let mut mol = Molecule::from_xyz(&path, 1e-7);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 0.2)));
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("C2".to_owned()));
    assert_eq!(
        sym.get_elements(&ROT).expect("No proper elements found.")[&ElementOrder::Int(2)].len(),
        1
    );
    assert_eq!(
        sym.get_generators(&ROT)
            .expect("No proper generators found.")[&ElementOrder::Int(2)]
            .len(),
        1
    );
}

#[test]
fn test_point_group_detection_asymmetric_cis_cocl2h4o2_magnetic_field_c2() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/cis-cocl2h4o2.xyz");
    let mut mol = Molecule::from_xyz(&path, 1e-7);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.2, 0.0)));
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("C2".to_owned()));
    assert_eq!(
        sym.get_elements(&ROT).expect("No proper elements found.")[&ElementOrder::Int(2)].len(),
        1
    );
    assert_eq!(
        sym.get_generators(&ROT)
            .expect("No proper generators found.")[&ElementOrder::Int(2)]
            .len(),
        1
    );
}

#[test]
fn test_point_group_detection_asymmetric_cuneane_magnetic_field_c2() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/cuneane.xyz");
    let mut mol = Molecule::from_xyz(&path, 1e-7);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 0.2)));
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("C2".to_owned()));
    assert_eq!(
        sym.get_elements(&ROT).expect("No proper elements found.")[&ElementOrder::Int(2)].len(),
        1
    );
    assert_eq!(
        sym.get_generators(&ROT)
            .expect("No proper generators found.")[&ElementOrder::Int(2)]
            .len(),
        1
    );
}

/***
C2v
***/

#[test]
fn test_point_group_detection_asymmetric_water_c2v() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/water.xyz");
    let mol = Molecule::from_xyz(&path, 1e-7);
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("C2v".to_owned()));
    assert_eq!(
        sym.get_elements(&ROT).expect("No proper elements found.")[&ElementOrder::Int(2)].len(),
        1
    );
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(1)].len(),
        2
    );
    assert_eq!(sym.get_sigma_elements("v").unwrap().len(), 2);

    assert_eq!(
        sym.get_generators(&ROT)
            .expect("No proper generators found.")[&ElementOrder::Int(2)]
            .len(),
        1
    );
    assert_eq!(
        sym.get_generators(&SIG)
            .expect("No improper generators found.")[&ElementOrder::Int(1)]
            .len(),
        1
    );
    assert_eq!(sym.get_sigma_generators("v").unwrap().len(), 1);
}

#[test]
fn test_point_group_detection_asymmetric_pyridine_c2v() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/pyridine.xyz");
    let mol = Molecule::from_xyz(&path, 1e-7);
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("C2v".to_owned()));
    assert_eq!(
        sym.get_elements(&ROT).expect("No proper elements found.")[&ElementOrder::Int(2)].len(),
        1
    );
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(1)].len(),
        2
    );
    assert_eq!(sym.get_sigma_elements("v").unwrap().len(), 2);

    assert_eq!(
        sym.get_generators(&ROT)
            .expect("No proper generators found.")[&ElementOrder::Int(2)]
            .len(),
        1
    );
    assert_eq!(
        sym.get_generators(&SIG)
            .expect("No improper generators found.")[&ElementOrder::Int(1)]
            .len(),
        1
    );
    assert_eq!(sym.get_sigma_generators("v").unwrap().len(), 1);
}

#[test]
fn test_point_group_detection_asymmetric_cyclobutene_c2v() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/cyclobutene.xyz");
    let mol = Molecule::from_xyz(&path, 1e-7);
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("C2v".to_owned()));
    assert_eq!(
        sym.get_elements(&ROT).expect("No proper elements found.")[&ElementOrder::Int(2)].len(),
        1
    );
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(1)].len(),
        2
    );
    assert_eq!(sym.get_sigma_elements("v").unwrap().len(), 2);

    assert_eq!(
        sym.get_generators(&ROT)
            .expect("No proper generators found.")[&ElementOrder::Int(2)]
            .len(),
        1
    );
    assert_eq!(
        sym.get_generators(&SIG)
            .expect("No improper generators found.")[&ElementOrder::Int(1)]
            .len(),
        1
    );
    assert_eq!(sym.get_sigma_generators("v").unwrap().len(), 1);
}

#[test]
fn test_point_group_detection_asymmetric_azulene_c2v() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/azulene.xyz");
    let mol = Molecule::from_xyz(&path, 1e-7);
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("C2v".to_owned()));
    assert_eq!(
        sym.get_elements(&ROT).expect("No proper elements found.")[&ElementOrder::Int(2)].len(),
        1
    );
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(1)].len(),
        2
    );
    assert_eq!(sym.get_sigma_elements("v").unwrap().len(), 2);

    assert_eq!(
        sym.get_generators(&ROT)
            .expect("No proper generators found.")[&ElementOrder::Int(2)]
            .len(),
        1
    );
    assert_eq!(
        sym.get_generators(&SIG)
            .expect("No improper generators found.")[&ElementOrder::Int(1)]
            .len(),
        1
    );
    assert_eq!(sym.get_sigma_generators("v").unwrap().len(), 1);
}

#[test]
fn test_point_group_detection_asymmetric_cuneane_c2v() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/cuneane.xyz");
    let mol = Molecule::from_xyz(&path, 1e-7);
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("C2v".to_owned()));
    assert_eq!(
        sym.get_elements(&ROT).expect("No proper elements found.")[&ElementOrder::Int(2)].len(),
        1
    );
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(1)].len(),
        2
    );
    assert_eq!(sym.get_sigma_elements("v").unwrap().len(), 2);

    assert_eq!(
        sym.get_generators(&ROT)
            .expect("No proper generators found.")[&ElementOrder::Int(2)]
            .len(),
        1
    );
    assert_eq!(
        sym.get_generators(&SIG)
            .expect("No improper generators found.")[&ElementOrder::Int(1)]
            .len(),
        1
    );
    assert_eq!(sym.get_sigma_generators("v").unwrap().len(), 1);
}

#[test]
fn test_point_group_detection_asymmetric_bf3_electric_field_c2v() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/bf3.xyz");
    let mut mol = Molecule::from_xyz(&path, 1e-7);
    mol.set_electric_field(Some(Vector3::new(0.2, 0.0, 0.0)));
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("C2v".to_owned()));
    assert_eq!(
        sym.get_elements(&ROT).expect("No proper elements found.")[&ElementOrder::Int(2)].len(),
        1
    );
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(1)].len(),
        2
    );
    assert_eq!(sym.get_sigma_elements("v").unwrap().len(), 2);

    assert_eq!(
        sym.get_generators(&ROT)
            .expect("No proper generators found.")[&ElementOrder::Int(2)]
            .len(),
        1
    );
    assert_eq!(
        sym.get_generators(&SIG)
            .expect("No improper generators found.")[&ElementOrder::Int(1)]
            .len(),
        1
    );
    assert_eq!(sym.get_sigma_generators("v").unwrap().len(), 1);
}

/***
C2h
***/

#[test]
fn test_point_group_detection_asymmetric_h2o2_c2h() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/h2o2.xyz");
    let mol = Molecule::from_xyz(&path, 1e-6);
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("C2h".to_owned()));
    assert_eq!(
        sym.get_elements(&ROT).expect("No proper elements found.")[&ElementOrder::Int(2)].len(),
        1
    );
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(1)].len(),
        1
    );
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(2)].len(),
        1
    );
    assert_eq!(sym.get_sigma_elements("h").unwrap().len(), 1);

    assert_eq!(
        sym.get_generators(&ROT)
            .expect("No proper generators found.")[&ElementOrder::Int(2)]
            .len(),
        1
    );
    assert_eq!(
        sym.get_generators(&SIG)
            .expect("No improper generators found.")[&ElementOrder::Int(1)]
            .len(),
        1
    );
    assert_eq!(sym.get_sigma_generators("h").unwrap().len(), 1);
}

#[test]
fn test_point_group_detection_asymmetric_zethrene_c2h() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/zethrene.xyz");
    let mol = Molecule::from_xyz(&path, 1e-6);
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("C2h".to_owned()));
    assert_eq!(
        sym.get_elements(&ROT).expect("No proper elements found.")[&ElementOrder::Int(2)].len(),
        1
    );
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(1)].len(),
        1
    );
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(2)].len(),
        1
    );
    assert_eq!(sym.get_sigma_elements("h").unwrap().len(), 1);

    assert_eq!(
        sym.get_generators(&ROT)
            .expect("No proper generators found.")[&ElementOrder::Int(2)]
            .len(),
        1
    );
    assert_eq!(
        sym.get_generators(&SIG)
            .expect("No improper generators found.")[&ElementOrder::Int(1)]
            .len(),
        1
    );
    assert_eq!(sym.get_sigma_generators("h").unwrap().len(), 1);
}

#[test]
fn test_point_group_detection_asymmetric_distorted_vf6_magnetic_field_c2h() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/vf6_d2h.xyz");
    let mut mol = Molecule::from_xyz(&path, 1e-7);
    mol.set_magnetic_field(Some(Vector3::new(1.0, 0.0, 0.0)));
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("C2h".to_owned()));
    assert_eq!(
        sym.get_elements(&ROT).expect("No proper elements found.")[&ElementOrder::Int(2)].len(),
        1
    );
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(1)].len(),
        1
    );
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(2)].len(),
        1
    );
    assert_eq!(sym.get_sigma_elements("h").unwrap().len(), 1);

    assert_eq!(
        sym.get_generators(&ROT)
            .expect("No proper generators found.")[&ElementOrder::Int(2)]
            .len(),
        1
    );
    assert_eq!(
        sym.get_generators(&SIG)
            .expect("No improper generators found.")[&ElementOrder::Int(1)]
            .len(),
        1
    );
    assert_eq!(sym.get_sigma_generators("h").unwrap().len(), 1);
}

#[test]
fn test_point_group_detection_asymmetric_b2h6_magnetic_field_c2h() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/b2h6.xyz");
    let mut mol = Molecule::from_xyz(&path, 1e-7);
    mol.set_magnetic_field(Some(Vector3::new(1.0, 0.0, 0.0)));
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("C2h".to_owned()));
    assert_eq!(
        sym.get_elements(&ROT).expect("No proper elements found.")[&ElementOrder::Int(2)].len(),
        1
    );
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(1)].len(),
        1
    );
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(2)].len(),
        1
    );
    assert_eq!(sym.get_sigma_elements("h").unwrap().len(), 1);

    assert_eq!(
        sym.get_generators(&ROT)
            .expect("No proper generators found.")[&ElementOrder::Int(2)]
            .len(),
        1
    );
    assert_eq!(
        sym.get_generators(&SIG)
            .expect("No improper generators found.")[&ElementOrder::Int(1)]
            .len(),
        1
    );
    assert_eq!(sym.get_sigma_generators("h").unwrap().len(), 1);
}

#[test]
fn test_point_group_detection_asymmetric_naphthalene_magnetic_field_c2h() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/naphthalene.xyz");
    let mut mol = Molecule::from_xyz(&path, 1e-7);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 1.0, 0.0)));
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("C2h".to_owned()));
    assert_eq!(
        sym.get_elements(&ROT).expect("No proper elements found.")[&ElementOrder::Int(2)].len(),
        1
    );
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(1)].len(),
        1
    );
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(2)].len(),
        1
    );
    assert_eq!(sym.get_sigma_elements("h").unwrap().len(), 1);

    assert_eq!(
        sym.get_generators(&ROT)
            .expect("No proper generators found.")[&ElementOrder::Int(2)]
            .len(),
        1
    );
    assert_eq!(
        sym.get_generators(&SIG)
            .expect("No improper generators found.")[&ElementOrder::Int(1)]
            .len(),
        1
    );
    assert_eq!(sym.get_sigma_generators("h").unwrap().len(), 1);
}

#[test]
fn test_point_group_detection_asymmetric_pyrene_magnetic_field_c2h() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/pyrene.xyz");
    let mut mol = Molecule::from_xyz(&path, 1e-7);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("C2h".to_owned()));
    assert_eq!(
        sym.get_elements(&ROT).expect("No proper elements found.")[&ElementOrder::Int(2)].len(),
        1
    );
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(1)].len(),
        1
    );
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(2)].len(),
        1
    );
    assert_eq!(sym.get_sigma_elements("h").unwrap().len(), 1);

    assert_eq!(
        sym.get_generators(&ROT)
            .expect("No proper generators found.")[&ElementOrder::Int(2)]
            .len(),
        1
    );
    assert_eq!(
        sym.get_generators(&SIG)
            .expect("No improper generators found.")[&ElementOrder::Int(1)]
            .len(),
        1
    );
    assert_eq!(sym.get_sigma_generators("h").unwrap().len(), 1);
}

#[test]
fn test_point_group_detection_asymmetric_c6o6_magnetic_field_c2h() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c6o6.xyz");
    let mut mol = Molecule::from_xyz(&path, 1e-7);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("C2h".to_owned()));
    assert_eq!(
        sym.get_elements(&ROT).expect("No proper elements found.")[&ElementOrder::Int(2)].len(),
        1
    );
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(1)].len(),
        1
    );
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(2)].len(),
        1
    );
    assert_eq!(sym.get_sigma_elements("h").unwrap().len(), 1);

    assert_eq!(
        sym.get_generators(&ROT)
            .expect("No proper generators found.")[&ElementOrder::Int(2)]
            .len(),
        1
    );
    assert_eq!(
        sym.get_generators(&SIG)
            .expect("No improper generators found.")[&ElementOrder::Int(1)]
            .len(),
        1
    );
    assert_eq!(sym.get_sigma_generators("h").unwrap().len(), 1);
}

/*
Cs
*/

#[test]
fn test_point_group_detection_asymmetric_propene_cs() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/propene.xyz");
    let mol = Molecule::from_xyz(&path, 1e-7);
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("Cs".to_owned()));
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(1)].len(),
        1
    );
    assert_eq!(sym.get_sigma_elements("h").unwrap().len(), 1);

    assert_eq!(
        sym.get_generators(&SIG)
            .expect("No improper generators found.")[&ElementOrder::Int(1)]
            .len(),
        1
    );
    assert_eq!(sym.get_sigma_generators("h").unwrap().len(), 1);
}

#[test]
fn test_point_group_detection_asymmetric_socl2_cs() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/socl2.xyz");
    let mol = Molecule::from_xyz(&path, 1e-7);
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("Cs".to_owned()));
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(1)].len(),
        1
    );
    assert_eq!(sym.get_sigma_elements("h").unwrap().len(), 1);

    assert_eq!(
        sym.get_generators(&SIG)
            .expect("No improper generators found.")[&ElementOrder::Int(1)]
            .len(),
        1
    );
    assert_eq!(sym.get_sigma_generators("h").unwrap().len(), 1);
}

#[test]
fn test_point_group_detection_asymmetric_hocl_cs() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/hocl.xyz");
    let mol = Molecule::from_xyz(&path, 1e-7);
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("Cs".to_owned()));
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(1)].len(),
        1
    );
    assert_eq!(sym.get_sigma_elements("h").unwrap().len(), 1);

    assert_eq!(
        sym.get_generators(&SIG)
            .expect("No improper generators found.")[&ElementOrder::Int(1)]
            .len(),
        1
    );
    assert_eq!(sym.get_sigma_generators("h").unwrap().len(), 1);
}

#[test]
fn test_point_group_detection_asymmetric_hocn_cs() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/hocn.xyz");
    let mol = Molecule::from_xyz(&path, 1e-7);
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("Cs".to_owned()));
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(1)].len(),
        1
    );
    assert_eq!(sym.get_sigma_elements("h").unwrap().len(), 1);

    assert_eq!(
        sym.get_generators(&SIG)
            .expect("No improper generators found.")[&ElementOrder::Int(1)]
            .len(),
        1
    );
    assert_eq!(sym.get_sigma_generators("h").unwrap().len(), 1);
}

#[test]
fn test_point_group_detection_asymmetric_nh2f_cs() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/nh2f.xyz");
    let mol = Molecule::from_xyz(&path, 1e-6);
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("Cs".to_owned()));
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(1)].len(),
        1
    );
    assert_eq!(sym.get_sigma_elements("h").unwrap().len(), 1);

    assert_eq!(
        sym.get_generators(&SIG)
            .expect("No improper generators found.")[&ElementOrder::Int(1)]
            .len(),
        1
    );
    assert_eq!(sym.get_sigma_generators("h").unwrap().len(), 1);
}

#[test]
fn test_point_group_detection_asymmetric_phenol_cs() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/phenol.xyz");
    let mol = Molecule::from_xyz(&path, 1e-7);
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("Cs".to_owned()));
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(1)].len(),
        1
    );
    assert_eq!(sym.get_sigma_elements("h").unwrap().len(), 1);

    assert_eq!(
        sym.get_generators(&SIG)
            .expect("No improper generators found.")[&ElementOrder::Int(1)]
            .len(),
        1
    );
    assert_eq!(sym.get_sigma_generators("h").unwrap().len(), 1);
}

#[test]
fn test_point_group_detection_asymmetric_f_pyrrole_cs() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/f-pyrrole.xyz");
    let mol = Molecule::from_xyz(&path, 1e-6);
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("Cs".to_owned()));
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(1)].len(),
        1
    );
    assert_eq!(sym.get_sigma_elements("h").unwrap().len(), 1);

    assert_eq!(
        sym.get_generators(&SIG)
            .expect("No improper generators found.")[&ElementOrder::Int(1)]
            .len(),
        1
    );
    assert_eq!(sym.get_sigma_generators("h").unwrap().len(), 1);
}

#[test]
fn test_point_group_detection_asymmetric_n2o_cs() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/n2o.xyz");
    let mol = Molecule::from_xyz(&path, 1e-7);
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("Cs".to_owned()));
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(1)].len(),
        1
    );
    assert_eq!(sym.get_sigma_elements("h").unwrap().len(), 1);

    assert_eq!(
        sym.get_generators(&SIG)
            .expect("No improper generators found.")[&ElementOrder::Int(1)]
            .len(),
        1
    );
    assert_eq!(sym.get_sigma_generators("h").unwrap().len(), 1);
}

#[test]
fn test_point_group_detection_asymmetric_fclbenzene_cs() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/fclbenzene.xyz");
    let mol = Molecule::from_xyz(&path, 1e-5);
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("Cs".to_owned()));
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(1)].len(),
        1
    );
    assert_eq!(sym.get_sigma_elements("h").unwrap().len(), 1);

    assert_eq!(
        sym.get_generators(&SIG)
            .expect("No improper generators found.")[&ElementOrder::Int(1)]
            .len(),
        1
    );
    assert_eq!(sym.get_sigma_generators("h").unwrap().len(), 1);
}

#[test]
fn test_point_group_detection_asymmetric_water_magnetic_field_cs() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/water.xyz");
    let mut mol = Molecule::from_xyz(&path, 1e-7);
    mol.set_magnetic_field(Some(Vector3::new(1.0, 0.0, 0.0)));
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("Cs".to_owned()));
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(1)].len(),
        1
    );
    assert_eq!(sym.get_sigma_elements("h").unwrap().len(), 1);

    assert_eq!(
        sym.get_generators(&SIG)
            .expect("No improper generators found.")[&ElementOrder::Int(1)]
            .len(),
        1
    );
    assert_eq!(sym.get_sigma_generators("h").unwrap().len(), 1);
}

#[test]
fn test_point_group_detection_asymmetric_pyridine_magnetic_field_cs() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/pyridine.xyz");
    let mut mol = Molecule::from_xyz(&path, 1e-7);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 0.2)));
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("Cs".to_owned()));
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(1)].len(),
        1
    );
    assert_eq!(sym.get_sigma_elements("h").unwrap().len(), 1);

    assert_eq!(
        sym.get_generators(&SIG)
            .expect("No improper generators found.")[&ElementOrder::Int(1)]
            .len(),
        1
    );
    assert_eq!(sym.get_sigma_generators("h").unwrap().len(), 1);
}

#[test]
fn test_point_group_detection_asymmetric_cyclobutene_magnetic_field_cs() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/cyclobutene.xyz");
    let mut mol = Molecule::from_xyz(&path, 1e-7);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.1, 0.0)));
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("Cs".to_owned()));
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(1)].len(),
        1
    );
    assert_eq!(sym.get_sigma_elements("h").unwrap().len(), 1);

    assert_eq!(
        sym.get_generators(&SIG)
            .expect("No improper generators found.")[&ElementOrder::Int(1)]
            .len(),
        1
    );
    assert_eq!(sym.get_sigma_generators("h").unwrap().len(), 1);
}

#[test]
fn test_point_group_detection_asymmetric_azulene_magnetic_field_cs() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/azulene.xyz");
    let mut mol = Molecule::from_xyz(&path, 1e-7);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.1, 0.0)));
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("Cs".to_owned()));
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(1)].len(),
        1
    );
    assert_eq!(sym.get_sigma_elements("h").unwrap().len(), 1);

    assert_eq!(
        sym.get_generators(&SIG)
            .expect("No improper generators found.")[&ElementOrder::Int(1)]
            .len(),
        1
    );
    assert_eq!(sym.get_sigma_generators("h").unwrap().len(), 1);
}

#[test]
fn test_point_group_detection_asymmetric_cis_cocl2h4o2_magnetic_field_cs() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/cis-cocl2h4o2.xyz");
    let mut mol = Molecule::from_xyz(&path, 1e-7);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 0.2)));
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("Cs".to_owned()));
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(1)].len(),
        1
    );
    assert_eq!(sym.get_sigma_elements("h").unwrap().len(), 1);

    assert_eq!(
        sym.get_generators(&SIG)
            .expect("No improper generators found.")[&ElementOrder::Int(1)]
            .len(),
        1
    );
    assert_eq!(sym.get_sigma_generators("h").unwrap().len(), 1);
}

#[test]
fn test_point_group_detection_asymmetric_cuneane_magnetic_field_cs() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/cuneane.xyz");
    let mut mol = Molecule::from_xyz(&path, 1e-7);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.5, 0.0)));
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("Cs".to_owned()));
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(1)].len(),
        1
    );
    assert_eq!(sym.get_sigma_elements("h").unwrap().len(), 1);

    assert_eq!(
        sym.get_generators(&SIG)
            .expect("No improper generators found.")[&ElementOrder::Int(1)]
            .len(),
        1
    );
    assert_eq!(sym.get_sigma_generators("h").unwrap().len(), 1);
}

#[test]
fn test_point_group_detection_asymmetric_water_electric_field_cs() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/water.xyz");
    let mut mol = Molecule::from_xyz(&path, 1e-7);
    mol.set_electric_field(Some(Vector3::new(1.0, 0.0, 0.0)));
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("Cs".to_owned()));
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(1)].len(),
        1
    );
    assert_eq!(sym.get_sigma_elements("h").unwrap().len(), 1);

    assert_eq!(
        sym.get_generators(&SIG)
            .expect("No improper generators found.")[&ElementOrder::Int(1)]
            .len(),
        1
    );
    assert_eq!(sym.get_sigma_generators("h").unwrap().len(), 1);
}

#[test]
fn test_point_group_detection_asymmetric_pyridine_electric_field_cs() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/pyridine.xyz");
    let mut mol = Molecule::from_xyz(&path, 1e-7);
    mol.set_electric_field(Some(Vector3::new(0.0, 0.0, 0.2)));
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("Cs".to_owned()));
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(1)].len(),
        1
    );
    assert_eq!(sym.get_sigma_elements("h").unwrap().len(), 1);

    assert_eq!(
        sym.get_generators(&SIG)
            .expect("No improper generators found.")[&ElementOrder::Int(1)]
            .len(),
        1
    );
    assert_eq!(sym.get_sigma_generators("h").unwrap().len(), 1);
}

#[test]
fn test_point_group_detection_asymmetric_cyclobutene_electric_field_cs() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/cyclobutene.xyz");
    let mut mol = Molecule::from_xyz(&path, 1e-7);
    mol.set_electric_field(Some(Vector3::new(0.0, 0.1, 0.0)));
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("Cs".to_owned()));
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(1)].len(),
        1
    );
    assert_eq!(sym.get_sigma_elements("h").unwrap().len(), 1);

    assert_eq!(
        sym.get_generators(&SIG)
            .expect("No improper generators found.")[&ElementOrder::Int(1)]
            .len(),
        1
    );
    assert_eq!(sym.get_sigma_generators("h").unwrap().len(), 1);
}

#[test]
fn test_point_group_detection_asymmetric_azulene_electric_field_cs() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/azulene.xyz");
    let mut mol = Molecule::from_xyz(&path, 1e-7);
    mol.set_electric_field(Some(Vector3::new(0.0, 0.1, 0.0)));
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("Cs".to_owned()));
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(1)].len(),
        1
    );
    assert_eq!(sym.get_sigma_elements("h").unwrap().len(), 1);

    assert_eq!(
        sym.get_generators(&SIG)
            .expect("No improper generators found.")[&ElementOrder::Int(1)]
            .len(),
        1
    );
    assert_eq!(sym.get_sigma_generators("h").unwrap().len(), 1);
}

#[test]
fn test_point_group_detection_asymmetric_cis_cocl2h4o2_electric_field_cs() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/cis-cocl2h4o2.xyz");
    let mut mol = Molecule::from_xyz(&path, 1e-7);
    mol.set_electric_field(Some(Vector3::new(0.0, 0.0, 0.2)));
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("Cs".to_owned()));
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(1)].len(),
        1
    );
    assert_eq!(sym.get_sigma_elements("h").unwrap().len(), 1);

    assert_eq!(
        sym.get_generators(&SIG)
            .expect("No improper generators found.")[&ElementOrder::Int(1)]
            .len(),
        1
    );
    assert_eq!(sym.get_sigma_generators("h").unwrap().len(), 1);
}

#[test]
fn test_point_group_detection_asymmetric_cuneane_electric_field_cs() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/cuneane.xyz");
    let mut mol = Molecule::from_xyz(&path, 1e-7);
    mol.set_electric_field(Some(Vector3::new(0.0, 0.5, 0.0)));
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("Cs".to_owned()));
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(1)].len(),
        1
    );
    assert_eq!(sym.get_sigma_elements("h").unwrap().len(), 1);

    assert_eq!(
        sym.get_generators(&SIG)
            .expect("No improper generators found.")[&ElementOrder::Int(1)]
            .len(),
        1
    );
    assert_eq!(sym.get_sigma_generators("h").unwrap().len(), 1);
}

#[test]
fn test_point_group_detection_asymmetric_bf3_magnetic_field_cs() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/bf3.xyz");
    let mut mol = Molecule::from_xyz(&path, 1e-7);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.5, 0.0)));
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("Cs".to_owned()));
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(1)].len(),
        1
    );
    assert_eq!(sym.get_sigma_elements("h").unwrap().len(), 1);

    assert_eq!(
        sym.get_generators(&SIG)
            .expect("No improper generators found.")[&ElementOrder::Int(1)]
            .len(),
        1
    );
    assert_eq!(sym.get_sigma_generators("h").unwrap().len(), 1);
}

/// This is a special case: Cs point group in a symmetric top.
#[test]
fn test_point_group_detection_symmetric_ch4_magnetic_field_cs() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/ch4.xyz");
    let mut mol = Molecule::from_xyz(&path, 1e-7);
    mol.set_magnetic_field(Some(Vector3::new(0.1, 0.1, 0.0)));
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("Cs".to_owned()));
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(1)].len(),
        1
    );
    assert_eq!(sym.get_sigma_elements("h").unwrap().len(), 1);

    assert_eq!(
        sym.get_generators(&SIG)
            .expect("No improper generators found.")[&ElementOrder::Int(1)]
            .len(),
        1
    );
    assert_eq!(sym.get_sigma_generators("h").unwrap().len(), 1);
}

/// This is another special case: Cs point group in a symmetric top.
#[test]
fn test_point_group_detection_symmetric_ch4_electric_field_cs() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/ch4.xyz");
    let mut mol = Molecule::from_xyz(&path, 1e-7);
    mol.set_electric_field(Some(Vector3::new(0.1, 0.1, 0.0)));
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("Cs".to_owned()));
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(1)].len(),
        1
    );
    assert_eq!(sym.get_sigma_elements("h").unwrap().len(), 1);

    assert_eq!(
        sym.get_generators(&SIG)
            .expect("No improper generators found.")[&ElementOrder::Int(1)]
            .len(),
        1
    );
    assert_eq!(sym.get_sigma_generators("h").unwrap().len(), 1);
}

#[test]
fn test_point_group_detection_asymmetric_atom_magnetic_electric_field_cs() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/th.xyz");
    let mut mol = Molecule::from_xyz(&path, 1e-7);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 0.2)));
    mol.set_electric_field(Some(Vector3::new(0.0, 0.1, 0.0)));
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("Cs".to_owned()));
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(1)].len(),
        1
    );
    assert_eq!(sym.get_sigma_elements("h").unwrap().len(), 1);

    assert_eq!(
        sym.get_generators(&SIG)
            .expect("No improper generators found.")[&ElementOrder::Int(1)]
            .len(),
        1
    );
    assert_eq!(sym.get_sigma_generators("h").unwrap().len(), 1);
}

/*
D2
*/

#[test]
fn test_point_group_detection_asymmetric_i4_biphenyl_d2() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/i4-biphenyl.xyz");
    let mol = Molecule::from_xyz(&path, 1e-7);
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("D2".to_owned()));
    assert_eq!(
        sym.get_elements(&ROT).expect("No proper elements found.")[&ElementOrder::Int(2)].len(),
        3
    );
    assert_eq!(
        sym.get_generators(&ROT)
            .expect("No proper generators found.")[&ElementOrder::Int(2)]
            .len(),
        2
    );
}

#[test]
fn test_point_group_detection_asymmetric_twistane_d2() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/twistane.xyz");
    let mol = Molecule::from_xyz(&path, 1e-7);
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("D2".to_owned()));
    assert_eq!(
        sym.get_elements(&ROT).expect("No proper elements found.")[&ElementOrder::Int(2)].len(),
        3
    );
    assert_eq!(
        sym.get_generators(&ROT)
            .expect("No proper generators found.")[&ElementOrder::Int(2)]
            .len(),
        2
    );
}

#[test]
fn test_point_group_detection_asymmetric_22_paracyclophane_d2() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/paracyclophane22.xyz");
    let mol = Molecule::from_xyz(&path, 1e-7);
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("D2".to_owned()));
    assert_eq!(
        sym.get_elements(&ROT).expect("No proper elements found.")[&ElementOrder::Int(2)].len(),
        3
    );
    assert_eq!(
        sym.get_generators(&ROT)
            .expect("No proper generators found.")[&ElementOrder::Int(2)]
            .len(),
        2
    );
}

/***
D2h
***/

#[test]
fn test_point_group_detection_asymmetric_b2h6_d2h() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/b2h6.xyz");
    let mol = Molecule::from_xyz(&path, 1e-7);
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("D2h".to_owned()));
    assert_eq!(
        sym.get_elements(&ROT).expect("No proper elements found.")[&ElementOrder::Int(2)].len(),
        3
    );
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(1)].len(),
        3
    );
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(2)].len(),
        1
    );
    assert_eq!(sym.get_sigma_elements("").unwrap().len(), 3);

    assert_eq!(
        sym.get_generators(&ROT)
            .expect("No proper generators found.")[&ElementOrder::Int(2)]
            .len(),
        2
    );
    assert_eq!(
        sym.get_generators(&SIG)
            .expect("No improper generators found.")[&ElementOrder::Int(1)]
            .len(),
        1
    );
    assert_eq!(sym.get_sigma_generators("").unwrap().len(), 1);
}

#[test]
fn test_point_group_detection_asymmetric_naphthalene_d2h() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/naphthalene.xyz");
    let mol = Molecule::from_xyz(&path, 1e-7);
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("D2h".to_owned()));
    assert_eq!(
        sym.get_elements(&ROT).expect("No proper elements found.")[&ElementOrder::Int(2)].len(),
        3
    );
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(1)].len(),
        3
    );
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(2)].len(),
        1
    );
    assert_eq!(sym.get_sigma_elements("").unwrap().len(), 3);

    assert_eq!(
        sym.get_generators(&ROT)
            .expect("No proper generators found.")[&ElementOrder::Int(2)]
            .len(),
        2
    );
    assert_eq!(
        sym.get_generators(&SIG)
            .expect("No improper generators found.")[&ElementOrder::Int(1)]
            .len(),
        1
    );
    assert_eq!(sym.get_sigma_generators("").unwrap().len(), 1);
}

#[test]
fn test_point_group_detection_asymmetric_pyrene_d2h() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/pyrene.xyz");
    let mol = Molecule::from_xyz(&path, 1e-7);
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("D2h".to_owned()));
    assert_eq!(
        sym.get_elements(&ROT).expect("No proper elements found.")[&ElementOrder::Int(2)].len(),
        3
    );
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(1)].len(),
        3
    );
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(2)].len(),
        1
    );
    assert_eq!(sym.get_sigma_elements("").unwrap().len(), 3);

    assert_eq!(
        sym.get_generators(&ROT)
            .expect("No proper generators found.")[&ElementOrder::Int(2)]
            .len(),
        2
    );
    assert_eq!(
        sym.get_generators(&SIG)
            .expect("No improper generators found.")[&ElementOrder::Int(1)]
            .len(),
        1
    );
    assert_eq!(sym.get_sigma_generators("").unwrap().len(), 1);
}

#[test]
fn test_point_group_detection_asymmetric_c6o6_d2h() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c6o6.xyz");
    let mol = Molecule::from_xyz(&path, 1e-7);
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("D2h".to_owned()));
    assert_eq!(
        sym.get_elements(&ROT).expect("No proper elements found.")[&ElementOrder::Int(2)].len(),
        3
    );
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(1)].len(),
        3
    );
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(2)].len(),
        1
    );
    assert_eq!(sym.get_sigma_elements("").unwrap().len(), 3);

    assert_eq!(
        sym.get_generators(&ROT)
            .expect("No proper generators found.")[&ElementOrder::Int(2)]
            .len(),
        2
    );
    assert_eq!(
        sym.get_generators(&SIG)
            .expect("No improper generators found.")[&ElementOrder::Int(1)]
            .len(),
        1
    );
    assert_eq!(sym.get_sigma_generators("").unwrap().len(), 1);
}

#[test]
fn test_point_group_detection_asymmetric_distorted_vf6_d2h() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/vf6_d2h.xyz");
    let mol = Molecule::from_xyz(&path, 1e-7);
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("D2h".to_owned()));
    assert_eq!(
        sym.get_elements(&ROT).expect("No proper elements found.")[&ElementOrder::Int(2)].len(),
        3
    );
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(1)].len(),
        3
    );
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(2)].len(),
        1
    );
    assert_eq!(sym.get_sigma_elements("").unwrap().len(), 3);

    assert_eq!(
        sym.get_generators(&ROT)
            .expect("No proper generators found.")[&ElementOrder::Int(2)]
            .len(),
        2
    );
    assert_eq!(
        sym.get_generators(&SIG)
            .expect("No improper generators found.")[&ElementOrder::Int(1)]
            .len(),
        1
    );
    assert_eq!(sym.get_sigma_generators("").unwrap().len(), 1);
}

/***
Ci
***/

#[test]
fn test_point_group_detection_asymmetric_meso_tartaricacid_ci() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/meso-tartaricacid.xyz");
    let mol = Molecule::from_xyz(&path, 1e-7);
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("Ci".to_owned()));
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(2)].len(),
        1
    );
    assert_eq!(
        sym.get_generators(&SIG)
            .expect("No improper generators found.")[&ElementOrder::Int(2)]
            .len(),
        1
    );
}

#[test]
fn test_point_group_detection_asymmetric_dibromodimethylcyclohexane_ci() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/dibromodimethylcyclohexane.xyz");
    let mol = Molecule::from_xyz(&path, 1e-7);
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("Ci".to_owned()));
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(2)].len(),
        1
    );
    assert_eq!(
        sym.get_generators(&SIG)
            .expect("No improper generators found.")[&ElementOrder::Int(2)]
            .len(),
        1
    );
}

#[test]
fn test_point_group_detection_asymmetric_h2o2_magnetic_field_ci() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/h2o2.xyz");
    let mut mol = Molecule::from_xyz(&path, 1e-6);
    mol.set_magnetic_field(Some(Vector3::new(1.0, 2.0, -1.0)));
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("Ci".to_owned()));
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(2)].len(),
        1
    );
    assert_eq!(
        sym.get_generators(&SIG)
            .expect("No improper generators found.")[&ElementOrder::Int(2)]
            .len(),
        1
    );
}

#[test]
fn test_point_group_detection_symmetric_xef4_magnetic_field_ci() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/xef4.xyz");
    let mut mol = Molecule::from_xyz(&path, 1e-7);
    mol.set_magnetic_field(Some(Vector3::new(1.0, 2.0, -2.0)));
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("Ci".to_owned()));
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(2)].len(),
        1
    );
    assert_eq!(
        sym.get_generators(&SIG)
            .expect("No improper generators found.")[&ElementOrder::Int(2)]
            .len(),
        1
    );
}

#[test]
fn test_point_group_detection_asymmetric_c2h2_magnetic_field_ci() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c2h2.xyz");
    let mut mol = Molecule::from_xyz(&path, 1e-6);
    mol.set_magnetic_field(Some(Vector3::new(1.0, -1.0, 1.0)));
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-6)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("Ci".to_owned()));
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(2)].len(),
        1
    );
    assert_eq!(
        sym.get_generators(&SIG)
            .expect("No improper generators found.")[&ElementOrder::Int(2)]
            .len(),
        1
    );
}

/// This is a special case: Ci from S2 via symmetric top.
#[test]
fn test_point_group_detection_symmetric_vf6_magnetic_field_ci() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/vf6.xyz");
    let mut mol = Molecule::from_xyz(&path, 1e-12);
    mol.set_magnetic_field(Some(Vector3::new(1.0, -2.0, 3.0)));
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-12)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("Ci".to_owned()));
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(2)].len(),
        1
    );
    assert_eq!(
        sym.get_generators(&SIG)
            .expect("No improper generators found.")[&ElementOrder::Int(2)]
            .len(),
        1
    );
}

/// This is a special case: Ci from S2 via symmetric top.
#[test]
fn test_point_group_detection_symmetric_c60_magnetic_field_ci() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c60.xyz");
    let mut mol = Molecule::from_xyz(&path, 1e-6);
    mol.set_magnetic_field(Some(Vector3::new(1.0, -2.0, 3.0)));
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-6)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("Ci".to_owned()));
    assert_eq!(
        sym.get_elements(&SIG).expect("No improper elements found.")[&ElementOrder::Int(2)].len(),
        1
    );
    assert_eq!(
        sym.get_generators(&SIG)
            .expect("No improper generators found.")[&ElementOrder::Int(2)]
            .len(),
        1
    );
}

/***
C1
***/

#[test]
fn test_point_group_detection_asymmetric_butan1ol_c1() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/butan-1-ol.xyz");
    let mol = Molecule::from_xyz(&path, 1e-7);
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("C1".to_owned()));
    assert_eq!(
        sym.get_elements(&ROT).expect("No proper elements found.")[&ElementOrder::Int(1)].len(),
        1
    );
    assert_eq!(
        sym.get_generators(&ROT)
            .expect("No proper generators found.")[&ElementOrder::Int(1)]
            .len(),
        1
    );
}

#[test]
fn test_point_group_detection_asymmetric_subst_5m_ring_c1() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/subst-5m-ring.xyz");
    let mol = Molecule::from_xyz(&path, 1e-7);
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("C1".to_owned()));
    assert_eq!(
        sym.get_elements(&ROT).expect("No proper elements found.")[&ElementOrder::Int(1)].len(),
        1
    );
    assert_eq!(
        sym.get_generators(&ROT)
            .expect("No proper generators found.")[&ElementOrder::Int(1)]
            .len(),
        1
    );
}

#[test]
fn test_point_group_detection_asymmetric_bf3_magnetic_field_c1() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/bf3.xyz");
    let mut mol = Molecule::from_xyz(&path, 1e-7);
    mol.set_magnetic_field(Some(Vector3::new(1.0, -1.0, 1.0)));
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("C1".to_owned()));
    assert_eq!(
        sym.get_elements(&ROT).expect("No proper elements found.")[&ElementOrder::Int(1)].len(),
        1
    );
    assert_eq!(
        sym.get_generators(&ROT)
            .expect("No proper generators found.")[&ElementOrder::Int(1)]
            .len(),
        1
    );
}

/// This is a special case: C1 via symmetric top.
#[test]
fn test_point_group_detection_symmetric_ch4_magnetic_field_c1() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/ch4.xyz");
    let mut mol = Molecule::from_xyz(&path, 1e-6);
    mol.set_magnetic_field(Some(Vector3::new(1.0, -3.0, 2.0)));
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-6)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("C1".to_owned()));
    assert_eq!(
        sym.get_elements(&ROT).expect("No proper elements found.")[&ElementOrder::Int(1)].len(),
        1
    );
    assert_eq!(
        sym.get_generators(&ROT)
            .expect("No proper generators found.")[&ElementOrder::Int(1)]
            .len(),
        1
    );
}

/// This is a special case: C1 via symmetric top.
#[test]
fn test_point_group_detection_symmetric_vf6_electric_field_c1() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/vf6.xyz");
    let mut mol = Molecule::from_xyz(&path, 1e-12);
    mol.set_electric_field(Some(Vector3::new(1.0, -2.0, 3.0)));
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-12)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("C1".to_owned()));
    assert_eq!(
        sym.get_elements(&ROT).expect("No proper elements found.")[&ElementOrder::Int(1)].len(),
        1
    );
    assert_eq!(
        sym.get_generators(&ROT)
            .expect("No proper generators found.")[&ElementOrder::Int(1)]
            .len(),
        1
    );
}

/// This is a special case: C1 via symmetric top.
#[test]
fn test_point_group_detection_symmetric_c60_electric_field_c1() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c60.xyz");
    let mut mol = Molecule::from_xyz(&path, 1e-6);
    mol.set_electric_field(Some(Vector3::new(1.0, -2.0, 3.0)));
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-6)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    assert_eq!(sym.point_group, Some("C1".to_owned()));
    assert_eq!(
        sym.get_elements(&ROT).expect("No proper elements found.")[&ElementOrder::Int(1)].len(),
        1
    );
    assert_eq!(
        sym.get_generators(&ROT)
            .expect("No proper generators found.")[&ElementOrder::Int(1)]
            .len(),
        1
    );
}
