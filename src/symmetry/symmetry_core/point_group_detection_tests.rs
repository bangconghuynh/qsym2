use crate::aux::molecule::Molecule;
use crate::aux::template_molecules;
use crate::symmetry::symmetry_core::{PreSymmetry, Symmetry};
use crate::symmetry::symmetry_element::ElementOrder;
use nalgebra::Vector3;

const ROOT: &str = env!("CARGO_MANIFEST_DIR");
use env_logger;

/********
Spherical
********/

#[test]
fn test_point_group_detection_atom_o3() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/th.xyz");
    let mol = Molecule::from_xyz(&path, 1e-6);
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-14)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::builder().build().unwrap();
    sym.analyse(&presym);
    assert_eq!(sym.point_group, Some("O(3)".to_owned()));
    assert_eq!(sym.proper_generators[&ElementOrder::Inf].len(), 3);
}

#[test]
fn test_point_group_detection_c60_ih() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c60.xyz");
    let mol = Molecule::from_xyz(&path, 1e-6);
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-6)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::builder().build().unwrap();
    sym.analyse(&presym);
    assert_eq!(sym.point_group, Some("Ih".to_owned()));
    assert_eq!(sym.proper_generators[&ElementOrder::Int(5)].len(), 6);
    assert_eq!(sym.improper_generators[&ElementOrder::Int(2)].len(), 1);
}

#[test]
fn test_point_group_detection_ch4_td() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/ch4.xyz");
    let mol = Molecule::from_xyz(&path, 1e-6);
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-6)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::builder().build().unwrap();
    sym.analyse(&presym);
    assert_eq!(sym.point_group, Some("Td".to_owned()));
    assert_eq!(sym.proper_generators[&ElementOrder::Int(3)].len(), 4);
    assert_eq!(sym.improper_generators[&ElementOrder::Int(1)].len(), 1);
    assert_eq!(sym.get_sigma_generators("d").unwrap().len(), 1);
}

#[test]
fn test_point_group_detection_adamantane_td() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/adamantane.xyz");
    let mol = Molecule::from_xyz(&path, 1e-6);
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-6)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::builder().build().unwrap();
    sym.analyse(&presym);
    assert_eq!(sym.point_group, Some("Td".to_owned()));
    assert_eq!(sym.proper_generators[&ElementOrder::Int(3)].len(), 4);
    assert_eq!(sym.improper_generators[&ElementOrder::Int(1)].len(), 1);
    assert_eq!(sym.get_sigma_generators("d").unwrap().len(), 1);
}

#[test]
fn test_point_group_detection_c165_diamond_nanoparticle_td() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c165.xyz");
    let mol = Molecule::from_xyz(&path, 1e-6);
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-6)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::builder().build().unwrap();
    sym.analyse(&presym);
    assert_eq!(sym.point_group, Some("Td".to_owned()));
    assert_eq!(sym.proper_generators[&ElementOrder::Int(3)].len(), 4);
    assert_eq!(sym.improper_generators[&ElementOrder::Int(1)].len(), 1);
    assert_eq!(sym.get_sigma_generators("d").unwrap().len(), 1);
}

#[test]
fn test_point_group_detection_vh2o6_th() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/vh2o6.xyz");
    let mol = Molecule::from_xyz(&path, 1e-12);
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-12)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::builder().build().unwrap();
    sym.analyse(&presym);
    assert_eq!(sym.point_group, Some("Th".to_owned()));
    assert_eq!(sym.proper_generators[&ElementOrder::Int(3)].len(), 4);
    assert_eq!(sym.improper_generators[&ElementOrder::Int(2)].len(), 1);
}

#[test]
fn test_point_group_detection_vf6_oh() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/vf6.xyz");
    let mol = Molecule::from_xyz(&path, 1e-12);
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-12)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::builder().build().unwrap();
    sym.analyse(&presym);
    assert_eq!(sym.point_group, Some("Oh".to_owned()));
    assert_eq!(sym.proper_generators[&ElementOrder::Int(3)].len(), 4);
    assert_eq!(sym.proper_generators[&ElementOrder::Int(4)].len(), 1);
    assert_eq!(sym.improper_generators[&ElementOrder::Int(2)].len(), 1);
}

/*****
Linear
*****/

#[test]
fn test_point_group_detection_atom_magnetic_field_cinfh() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/th.xyz");
    let mut mol = Molecule::from_xyz(&path, 1e-7);
    mol.set_magnetic_field(Some(Vector3::new(1.0, 2.0, -1.0)));
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-14)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::builder().build().unwrap();
    sym.analyse(&presym);
    assert_eq!(sym.point_group, Some("C∞h".to_owned()));
    assert_eq!(sym.proper_generators[&ElementOrder::Inf].len(), 1);
    assert_eq!(sym.improper_generators[&ElementOrder::Int(1)].len(), 1);
    assert_eq!(sym.get_sigma_generators("h").unwrap().len(), 1);
}

#[test]
fn test_point_group_detection_atom_electric_field_cinfv() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/th.xyz");
    let mut mol = Molecule::from_xyz(&path, 1e-7);
    mol.set_electric_field(Some(Vector3::new(-1.0, 3.0, -2.0)));
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-14)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::builder().build().unwrap();
    sym.analyse(&presym);
    assert_eq!(sym.point_group, Some("C∞v".to_owned()));
    assert_eq!(sym.proper_generators[&ElementOrder::Inf].len(), 1);
    assert_eq!(sym.improper_generators[&ElementOrder::Int(1)].len(), 1);
    assert_eq!(sym.get_sigma_generators("v").unwrap().len(), 1);
}

#[test]
fn test_point_group_detection_c2h2_dinfh() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c2h2.xyz");
    let mol = Molecule::from_xyz(&path, 1e-6);
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-6)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::builder().build().unwrap();
    sym.analyse(&presym);
    assert_eq!(sym.point_group, Some("D∞h".to_owned()));
    assert_eq!(sym.proper_generators[&ElementOrder::Inf].len(), 1);
    assert_eq!(sym.proper_generators[&ElementOrder::Int(2)].len(), 1);
    assert_eq!(sym.improper_generators[&ElementOrder::Int(1)].len(), 1);
    assert_eq!(sym.get_sigma_generators("h").unwrap().len(), 1);
}

#[test]
fn test_point_group_detection_c2h2_magnetic_field_dinfh() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c2h2.xyz");
    let mut mol = Molecule::from_xyz(&path, 1e-6);

    // Parallel field
    mol.set_magnetic_field(Some(Vector3::new(1.0, 1.0, 1.0)));
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-6)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::builder().build().unwrap();
    sym.analyse(&presym);
    assert_eq!(sym.point_group, Some("C∞h".to_owned()));
    assert_eq!(sym.proper_generators[&ElementOrder::Inf].len(), 1);
    assert_eq!(sym.improper_generators[&ElementOrder::Int(1)].len(), 1);
    assert_eq!(sym.get_sigma_generators("h").unwrap().len(), 1);

    // TODO: Finish this
    // Perpendicular field
    // mol.set_magnetic_field(Some(Vector3::new(1.0, -1.0, 0.0)));
    // let presym = PreSymmetry::builder()
    //     .moi_threshold(1e-6)
    //     .molecule(&mol, true)
    //     .build()
    //     .unwrap();
    // let mut sym = Symmetry::builder().build().unwrap();
    // sym.analyse(&presym);
    // assert_eq!(sym.point_group, Some("Cs".to_owned()));
    // assert_eq!(sym.proper_generators[&ElementOrder::Inf].len(), 1);
    // assert_eq!(sym.improper_generators[&ElementOrder::Int(1)].len(), 1);
    // assert_eq!(sym.get_sigma_generators("h").unwrap().len(), 1);
}

#[test]
fn test_point_group_detection_c2h2_electric_field_cinfv() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c2h2.xyz");
    let mut mol = Molecule::from_xyz(&path, 1e-6);

    // Parallel field
    mol.set_electric_field(Some(Vector3::new(1.0, 1.0, 1.0)));
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-6)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::builder().build().unwrap();
    sym.analyse(&presym);
    assert_eq!(sym.point_group, Some("C∞v".to_owned()));
    assert_eq!(sym.proper_generators[&ElementOrder::Inf].len(), 1);

    // TODO: Finish this
    // Perpendicular field
    // mol.set_magnetic_field(Some(Vector3::new(1.0, -1.0, 0.0)));
    // let presym = PreSymmetry::builder()
    //     .moi_threshold(1e-6)
    //     .molecule(&mol, true)
    //     .build()
    //     .unwrap();
    // let mut sym = Symmetry::builder().build().unwrap();
    // sym.analyse(&presym);
    // assert_eq!(sym.point_group, Some("Cs".to_owned()));
    // assert_eq!(sym.proper_generators[&ElementOrder::Inf].len(), 1);
    // assert_eq!(sym.improper_generators[&ElementOrder::Int(1)].len(), 1);
    // assert_eq!(sym.get_sigma_generators("h").unwrap().len(), 1);
}

#[test]
fn test_point_group_detection_n3_cinfv() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/n3.xyz");
    let mol = Molecule::from_xyz(&path, 1e-6);
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-6)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::builder().build().unwrap();
    sym.analyse(&presym);
    assert_eq!(sym.point_group, Some("C∞v".to_owned()));
    assert_eq!(sym.proper_generators[&ElementOrder::Inf].len(), 1);
    assert_eq!(sym.improper_generators[&ElementOrder::Int(1)].len(), 1);
    assert_eq!(sym.get_sigma_generators("v").unwrap().len(), 1);
}

#[test]
fn test_point_group_detection_n3_magnetic_field_cinf() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/n3.xyz");
    let mut mol = Molecule::from_xyz(&path, 1e-6);

    // Parallel field
    mol.set_magnetic_field(Some(Vector3::new(1.0, 1.0, 1.0)));
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-6)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::builder().build().unwrap();
    sym.analyse(&presym);
    assert_eq!(sym.point_group, Some("C∞".to_owned()));
    assert_eq!(sym.proper_generators[&ElementOrder::Inf].len(), 1);
}

#[test]
fn test_point_group_detection_n3_electric_field_cinfv() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/n3.xyz");
    let mut mol = Molecule::from_xyz(&path, 1e-6);

    // Parallel field
    mol.set_electric_field(Some(Vector3::new(1.0, 1.0, 1.0)));
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-6)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::builder().build().unwrap();
    sym.analyse(&presym);
    assert_eq!(sym.point_group, Some("C∞v".to_owned()));
    assert_eq!(sym.proper_generators[&ElementOrder::Inf].len(), 1);
    assert_eq!(sym.improper_generators[&ElementOrder::Int(1)].len(), 1);
    assert_eq!(sym.get_sigma_generators("v").unwrap().len(), 1);
}

/********
Symmetric
********/

/*
Cn
*/

#[test]
fn test_point_group_detection_ch4_magnetic_field_c3() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/ch4.xyz");
    let mut mol = Molecule::from_xyz(&path, 1e-6);
    mol.set_magnetic_field(Some(Vector3::new(1.0, 1.0, 1.0)));
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-6)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::builder().build().unwrap();
    sym.analyse(&presym);
    assert_eq!(sym.point_group, Some("C3".to_owned()));
    assert_eq!(sym.proper_elements[&ElementOrder::Int(3)].len(), 1);
    assert_eq!(sym.proper_generators[&ElementOrder::Int(3)].len(), 1);
}

#[test]
fn test_point_group_detection_adamantane_magnetic_field_c3() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/adamantane.xyz");
    let mut mol = Molecule::from_xyz(&path, 1e-6);
    mol.set_magnetic_field(Some(Vector3::new(0.1, 0.1, 0.1)));
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-6)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::builder().build().unwrap();
    sym.analyse(&presym);
    assert_eq!(sym.point_group, Some("C3".to_owned()));
    assert_eq!(sym.proper_elements[&ElementOrder::Int(3)].len(), 1);
    assert_eq!(sym.proper_generators[&ElementOrder::Int(3)].len(), 1);
}

#[test]
fn test_point_group_detection_vh2o6_electric_field_c3() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/vh2o6.xyz");
    let mut mol = Molecule::from_xyz(&path, 1e-12);
    mol.set_electric_field(Some(Vector3::new(-0.2, -0.2, -0.2)));
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-12)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::builder().build().unwrap();
    sym.analyse(&presym);
    assert_eq!(sym.point_group, Some("C3".to_owned()));
    assert_eq!(sym.proper_elements[&ElementOrder::Int(3)].len(), 1);
    assert_eq!(sym.proper_generators[&ElementOrder::Int(3)].len(), 1);
}

#[test]
fn test_point_group_detection_65coronane_electric_field_c3() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/coronane65.xyz");
    let mut mol = Molecule::from_xyz(&path, 1e-7);
    mol.set_electric_field(Some(Vector3::new(0.0, 0.0, -1.0)));
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::builder().build().unwrap();
    sym.analyse(&presym);
    assert_eq!(sym.point_group, Some("C3".to_owned()));
    assert_eq!(sym.proper_elements[&ElementOrder::Int(3)].len(), 1);
    assert_eq!(sym.proper_generators[&ElementOrder::Int(3)].len(), 1);
}

#[test]
fn test_point_group_detection_h8_twisted_magnetic_field_c4() {
    // env_logger::init();
    let mut mol = template_molecules::gen_twisted_h8(0.1);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 0.1)));
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::builder().build().unwrap();
    sym.analyse(&presym);
    assert_eq!(sym.point_group, Some("C4".to_owned()));
    assert_eq!(sym.proper_elements[&ElementOrder::Int(4)].len(), 1);
    assert_eq!(sym.proper_generators[&ElementOrder::Int(4)].len(), 1);
}

#[test]
fn test_point_group_detection_h8_twisted_electric_field_c4() {
    let mut mol = template_molecules::gen_twisted_h8(0.1);
    mol.set_electric_field(Some(Vector3::new(0.0, 0.0, -0.1)));
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::builder().build().unwrap();
    sym.analyse(&presym);
    assert_eq!(sym.point_group, Some("C4".to_owned()));
    assert_eq!(sym.proper_elements[&ElementOrder::Int(4)].len(), 1);
    assert_eq!(sym.proper_generators[&ElementOrder::Int(4)].len(), 1);
}

#[test]
fn test_point_group_detection_cpnico_magnetic_field_c5() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/cpnico.xyz");
    let mut mol = Molecule::from_xyz(&path, 1e-7);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, -0.2)));
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::builder().build().unwrap();
    sym.analyse(&presym);
    assert_eq!(sym.point_group, Some("C5".to_owned()));
    assert_eq!(sym.proper_elements[&ElementOrder::Int(5)].len(), 1);
    assert_eq!(sym.proper_generators[&ElementOrder::Int(5)].len(), 1);
}

#[test]
fn test_point_group_detection_b7_magnetic_field_c6() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/b7.xyz");
    let mut mol = Molecule::from_xyz(&path, 1e-6);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 0.1)));
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-6)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::builder().build().unwrap();
    sym.analyse(&presym);
    assert_eq!(sym.point_group, Some("C6".to_owned()));
    assert_eq!(sym.proper_elements[&ElementOrder::Int(6)].len(), 1);
    assert_eq!(sym.proper_generators[&ElementOrder::Int(6)].len(), 1);
}

#[test]
fn test_point_group_detection_arbitrary_half_sandwich_magnetic_field_cn() {
    for n in 3..=32 {
        let mut mol = template_molecules::gen_arbitrary_half_sandwich(n);
        mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 0.1)));
        let presym = PreSymmetry::builder()
            .moi_threshold(1e-7)
            .molecule(&mol, true)
            .build()
            .unwrap();
        let mut sym = Symmetry::builder().build().unwrap();
        sym.analyse(&presym);
        assert_eq!(sym.point_group, Some(format!("C{n}")));
        assert_eq!(sym.proper_elements[&ElementOrder::Int(n)].len(), 1);
        assert_eq!(sym.proper_generators[&ElementOrder::Int(n)].len(), 1);
    }
}


/*
Cnv
*/

#[test]
fn test_point_group_detection_nh3_c3v() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/nh3.xyz");
    let mol = Molecule::from_xyz(&path, 1e-6);
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-6)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::builder().build().unwrap();
    println!("{}", presym.rotational_symmetry);
    sym.analyse(&presym);
    assert_eq!(sym.point_group, Some("C3v".to_owned()));
    assert_eq!(sym.proper_elements[&ElementOrder::Int(3)].len(), 1);
    assert_eq!(sym.improper_elements[&ElementOrder::Int(1)].len(), 3);
    assert_eq!(sym.get_sigma_elements("v").unwrap().len(), 3);

    assert_eq!(sym.proper_generators[&ElementOrder::Int(3)].len(), 1);
    assert_eq!(sym.improper_generators[&ElementOrder::Int(1)].len(), 1);
    assert_eq!(sym.get_sigma_generators("v").unwrap().len(), 1);
}

#[test]
fn test_point_group_detection_bf3_electric_field_c3v() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/bf3.xyz");
    let mut mol = Molecule::from_xyz(&path, 1e-7);
    mol.set_electric_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::builder().build().unwrap();
    sym.analyse(&presym);
    assert_eq!(sym.point_group, Some("C3v".to_owned()));
    assert_eq!(sym.proper_elements[&ElementOrder::Int(3)].len(), 1);
    assert_eq!(sym.improper_elements[&ElementOrder::Int(1)].len(), 3);
    assert_eq!(sym.get_sigma_elements("v").unwrap().len(), 3);

    assert_eq!(sym.proper_generators[&ElementOrder::Int(3)].len(), 1);
    assert_eq!(sym.improper_generators[&ElementOrder::Int(1)].len(), 1);
    assert_eq!(sym.get_sigma_generators("v").unwrap().len(), 1);
}

#[test]
fn test_point_group_detection_adamantane_electric_field_c3v() {
    env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/adamantane.xyz");
    let mut mol = Molecule::from_xyz(&path, 1e-6);
    mol.set_electric_field(Some(Vector3::new(0.1, 0.1, 0.1)));
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-6)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::builder().build().unwrap();
    sym.analyse(&presym);
    assert_eq!(sym.point_group, Some("C3v".to_owned()));
    assert_eq!(sym.proper_elements[&ElementOrder::Int(3)].len(), 1);
    assert_eq!(sym.improper_elements[&ElementOrder::Int(1)].len(), 3);
    assert_eq!(sym.get_sigma_elements("v").unwrap().len(), 3);

    assert_eq!(sym.proper_generators[&ElementOrder::Int(3)].len(), 1);
    assert_eq!(sym.improper_generators[&ElementOrder::Int(1)].len(), 1);
    assert_eq!(sym.get_sigma_generators("v").unwrap().len(), 1);
}

#[test]
fn test_point_group_detection_sf5cl_c4v() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/sf5cl.xyz");
    let mol = Molecule::from_xyz(&path, 1e-7);
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::builder().build().unwrap();
    println!("{}", presym.rotational_symmetry);
    sym.analyse(&presym);
    assert_eq!(sym.point_group, Some("C4v".to_owned()));
    assert_eq!(sym.proper_elements[&ElementOrder::Int(4)].len(), 1);
    assert_eq!(sym.improper_elements[&ElementOrder::Int(1)].len(), 4);
    assert_eq!(sym.get_sigma_elements("v").unwrap().len(), 4);

    assert_eq!(sym.proper_generators[&ElementOrder::Int(4)].len(), 1);
    assert_eq!(sym.improper_generators[&ElementOrder::Int(1)].len(), 1);
    assert_eq!(sym.get_sigma_generators("v").unwrap().len(), 1);
}

#[test]
fn test_point_group_detection_cpnico_c5v() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/cpnico.xyz");
    let mol = Molecule::from_xyz(&path, 1e-7);
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::builder().build().unwrap();
    sym.analyse(&presym);
    assert_eq!(sym.point_group, Some("C5v".to_owned()));
    assert_eq!(sym.proper_elements[&ElementOrder::Int(5)].len(), 1);
    assert_eq!(sym.improper_elements[&ElementOrder::Int(1)].len(), 5);
    assert_eq!(sym.get_sigma_elements("v").unwrap().len(), 5);

    assert_eq!(sym.proper_generators[&ElementOrder::Int(5)].len(), 1);
    assert_eq!(sym.improper_generators[&ElementOrder::Int(1)].len(), 1);
    assert_eq!(sym.get_sigma_generators("v").unwrap().len(), 1);
}

#[test]
fn test_point_group_detection_b7_c6v() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/b7.xyz");
    let mol = Molecule::from_xyz(&path, 1e-6);
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-6)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::builder().build().unwrap();
    sym.analyse(&presym);
    assert_eq!(sym.point_group, Some("C6v".to_owned()));
    assert_eq!(sym.proper_elements[&ElementOrder::Int(6)].len(), 1);
    assert_eq!(sym.improper_elements[&ElementOrder::Int(1)].len(), 6);
    assert_eq!(sym.get_sigma_elements("v").unwrap().len(), 6);

    assert_eq!(sym.proper_generators[&ElementOrder::Int(6)].len(), 1);
    assert_eq!(sym.improper_generators[&ElementOrder::Int(1)].len(), 1);
    assert_eq!(sym.get_sigma_generators("v").unwrap().len(), 1);
}

#[test]
fn test_point_group_detection_arbitrary_half_sandwich_cnv() {
    for n in 3..=32 {
        let mol = template_molecules::gen_arbitrary_half_sandwich(n);
        let presym = PreSymmetry::builder()
            .moi_threshold(1e-7)
            .molecule(&mol, true)
            .build()
            .unwrap();
        let mut sym = Symmetry::builder().build().unwrap();
        sym.analyse(&presym);
        assert_eq!(sym.point_group, Some(format!("C{n}v")));
        assert_eq!(sym.proper_elements[&ElementOrder::Int(n)].len(), 1);
        assert_eq!(sym.improper_elements[&ElementOrder::Int(1)].len() as u32, n);
        assert_eq!(sym.get_sigma_elements("v").unwrap().len() as u32, n);

        assert_eq!(sym.proper_generators[&ElementOrder::Int(n)].len(), 1);
        assert_eq!(sym.improper_generators[&ElementOrder::Int(1)].len(), 1);
        assert_eq!(sym.get_sigma_generators("v").unwrap().len(), 1);
    }
}

/*
Cnh
*/

#[test]
fn test_point_group_detection_bf3_magnetic_field_c3h() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/bf3.xyz");
    let mut mol = Molecule::from_xyz(&path, 1e-7);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::builder().build().unwrap();
    sym.analyse(&presym);
    assert_eq!(sym.point_group, Some("C3h".to_owned()));
    assert_eq!(sym.proper_elements[&ElementOrder::Int(3)].len(), 1);
    assert_eq!(sym.improper_elements[&ElementOrder::Int(1)].len(), 1);
    assert_eq!(sym.improper_elements[&ElementOrder::Int(3)].len(), 1);
    assert_eq!(sym.get_sigma_elements("h").unwrap().len(), 1);

    assert_eq!(sym.proper_generators[&ElementOrder::Int(3)].len(), 1);
    assert_eq!(sym.improper_generators[&ElementOrder::Int(1)].len(), 1);
    assert_eq!(sym.get_sigma_generators("h").unwrap().len(), 1);
}

#[test]
fn test_point_group_detection_xef4_magnetic_field_c4h() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/xef4.xyz");
    let mut mol = Molecule::from_xyz(&path, 1e-7);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, -1.0)));
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::builder().build().unwrap();
    sym.analyse(&presym);
    assert_eq!(sym.point_group, Some("C4h".to_owned()));
    assert_eq!(sym.proper_elements[&ElementOrder::Int(4)].len(), 1);
    assert_eq!(sym.improper_elements[&ElementOrder::Int(1)].len(), 1);
    assert_eq!(sym.improper_elements[&ElementOrder::Int(2)].len(), 1);
    assert_eq!(sym.improper_elements[&ElementOrder::Int(4)].len(), 1);
    assert_eq!(sym.get_sigma_elements("h").unwrap().len(), 1);

    assert_eq!(sym.proper_generators[&ElementOrder::Int(4)].len(), 1);
    assert_eq!(sym.improper_generators[&ElementOrder::Int(1)].len(), 1);
    assert_eq!(sym.get_sigma_generators("h").unwrap().len(), 1);
}

#[test]
fn test_point_group_detection_eclipsed_ferrocene_magnetic_field_c5h() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/eclipsed_ferrocene.xyz");
    let mut mol = Molecule::from_xyz(&path, 1e-7);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, -1.0)));
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::builder().build().unwrap();
    sym.analyse(&presym);
    assert_eq!(sym.point_group, Some("C5h".to_owned()));
    assert_eq!(sym.proper_elements[&ElementOrder::Int(5)].len(), 1);
    assert_eq!(sym.improper_elements[&ElementOrder::Int(1)].len(), 1);
    assert_eq!(sym.improper_elements[&ElementOrder::Int(5)].len(), 1);
    assert_eq!(sym.get_sigma_elements("h").unwrap().len(), 1);

    assert_eq!(sym.proper_generators[&ElementOrder::Int(5)].len(), 1);
    assert_eq!(sym.improper_generators[&ElementOrder::Int(1)].len(), 1);
    assert_eq!(sym.get_sigma_generators("h").unwrap().len(), 1);
}

#[test]
fn test_point_group_detection_benzene_magnetic_field_c6h() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/benzene.xyz");
    let mut mol = Molecule::from_xyz(&path, 1e-7);
    mol.set_magnetic_field(Some(Vector3::new(1.0, 0.0, 0.0)));
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::builder().build().unwrap();
    sym.analyse(&presym);
    assert_eq!(sym.point_group, Some("C6h".to_owned()));
    assert_eq!(sym.proper_elements[&ElementOrder::Int(6)].len(), 1);
    assert_eq!(sym.improper_elements[&ElementOrder::Int(1)].len(), 1);
    assert_eq!(sym.improper_elements[&ElementOrder::Int(2)].len(), 1);
    assert_eq!(sym.improper_elements[&ElementOrder::Int(6)].len(), 1);
    assert_eq!(sym.get_sigma_elements("h").unwrap().len(), 1);

    assert_eq!(sym.proper_generators[&ElementOrder::Int(6)].len(), 1);
    assert_eq!(sym.improper_generators[&ElementOrder::Int(1)].len(), 1);
    assert_eq!(sym.get_sigma_generators("h").unwrap().len(), 1);
}

#[test]
fn test_point_group_detection_arbitrary_eclipsed_sandwich_magnetic_field_cnh() {
    // env_logger::init();
    for n in 3..=20 {
        let mut mol = template_molecules::gen_arbitrary_eclipsed_sandwich(n);
        mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 0.1)));
        let presym = PreSymmetry::builder()
            .moi_threshold(1e-7)
            .molecule(&mol, true)
            .build()
            .unwrap();
        let mut sym = Symmetry::builder().build().unwrap();
        sym.analyse(&presym);
        assert_eq!(sym.point_group, Some(format!("C{n}h")));
        assert_eq!(sym.proper_elements[&ElementOrder::Int(n)].len(), 1);
        assert_eq!(sym.improper_elements[&ElementOrder::Int(1)].len(), 1);
        assert_eq!(sym.improper_elements[&ElementOrder::Int(n)].len(), 1);
        assert_eq!(sym.get_sigma_elements("h").unwrap().len(), 1);
        if n % 2 == 0 {
            assert_eq!(sym.proper_elements[&ElementOrder::Int(2)].len(), 1);
            assert_eq!(sym.improper_elements[&ElementOrder::Int(2)].len(), 1);
        };

        assert_eq!(sym.proper_generators[&ElementOrder::Int(n)].len(), 1);
        assert_eq!(sym.improper_generators[&ElementOrder::Int(1)].len(), 1);
        assert_eq!(sym.get_sigma_generators("h").unwrap().len(), 1);
    }
}


/*
Dn
*/

#[test]
fn test_point_group_detection_triphenyl_radical_d3() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/triphenylradical.xyz");
    let mol = Molecule::from_xyz(&path, 1e-7);
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::builder().build().unwrap();
    sym.analyse(&presym);
    assert_eq!(sym.point_group, Some("D3".to_owned()));
    assert_eq!(sym.proper_elements[&ElementOrder::Int(3)].len(), 1);
    assert_eq!(sym.proper_elements[&ElementOrder::Int(2)].len(), 3);

    assert_eq!(sym.proper_generators[&ElementOrder::Int(3)].len(), 1);
    assert_eq!(sym.proper_generators[&ElementOrder::Int(2)].len(), 1);
}

#[test]
fn test_point_group_detection_h8_twisted_d4() {
    let mol = template_molecules::gen_twisted_h8(0.1);
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::builder().build().unwrap();
    sym.analyse(&presym);
    assert_eq!(sym.point_group, Some("D4".to_owned()));
    assert_eq!(sym.proper_elements[&ElementOrder::Int(4)].len(), 1);
    assert_eq!(sym.proper_generators[&ElementOrder::Int(4)].len(), 1);
    assert_eq!(sym.proper_elements[&ElementOrder::Int(2)].len(), 5);
    assert_eq!(sym.proper_generators[&ElementOrder::Int(2)].len(), 1);
}

#[test]
fn test_point_group_detection_c5ph5_d5() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c5ph5.xyz");
    let mol = Molecule::from_xyz(&path, 1e-7);
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::builder().build().unwrap();
    sym.analyse(&presym);
    assert_eq!(sym.point_group, Some("D5".to_owned()));
    assert_eq!(sym.proper_elements[&ElementOrder::Int(5)].len(), 1);
    assert_eq!(sym.proper_elements[&ElementOrder::Int(2)].len(), 5);

    assert_eq!(sym.proper_generators[&ElementOrder::Int(5)].len(), 1);
    assert_eq!(sym.proper_generators[&ElementOrder::Int(2)].len(), 1);
}

#[test]
fn test_point_group_detection_c6ph6_d6() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c6ph6.xyz");
    let mol = Molecule::from_xyz(&path, 1e-6);
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::builder().build().unwrap();
    sym.analyse(&presym);
    assert_eq!(sym.point_group, Some("D6".to_owned()));
    assert_eq!(sym.proper_elements[&ElementOrder::Int(6)].len(), 1);
    assert_eq!(sym.proper_elements[&ElementOrder::Int(2)].len(), 7);

    assert_eq!(sym.proper_generators[&ElementOrder::Int(6)].len(), 1);
    assert_eq!(sym.proper_generators[&ElementOrder::Int(2)].len(), 1);
}

#[test]
fn test_point_group_detection_arbitrary_twisted_sandwich_dn() {
    // env_logger::init();
    for n in 3..=20 {
        let mol = template_molecules::gen_arbitrary_twisted_sandwich(n, 0.1);
        let presym = PreSymmetry::builder()
            .moi_threshold(1e-7)
            .molecule(&mol, true)
            .build()
            .unwrap();
        let mut sym = Symmetry::builder().build().unwrap();
        sym.analyse(&presym);
        assert_eq!(sym.point_group, Some(format!("D{n}")));
        assert_eq!(sym.proper_elements[&ElementOrder::Int(n)].len(), 1);
        if n % 2 == 0 {
            assert_eq!(sym.proper_elements[&ElementOrder::Int(2)].len() as u32, n + 1);
        } else {
            assert_eq!(sym.proper_elements[&ElementOrder::Int(2)].len() as u32, n);
        };

        assert_eq!(sym.proper_generators[&ElementOrder::Int(n)].len(), 1);
        assert_eq!(sym.proper_generators[&ElementOrder::Int(2)].len(), 1);
    }
}


/*
Dnh
*/

#[test]
fn test_point_group_detection_bf3_d3h() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/bf3.xyz");
    let mol = Molecule::from_xyz(&path, 1e-7);
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::builder().build().unwrap();
    sym.analyse(&presym);
    assert_eq!(sym.point_group, Some("D3h".to_owned()));
    assert_eq!(sym.proper_elements[&ElementOrder::Int(3)].len(), 1);
    assert_eq!(sym.proper_elements[&ElementOrder::Int(2)].len(), 3);
    assert_eq!(sym.improper_elements[&ElementOrder::Int(1)].len(), 4);
    assert_eq!(sym.improper_elements[&ElementOrder::Int(3)].len(), 1);
    assert_eq!(sym.get_sigma_elements("h").unwrap().len(), 1);
    assert_eq!(sym.get_sigma_elements("v").unwrap().len(), 3);

    assert_eq!(sym.proper_generators[&ElementOrder::Int(3)].len(), 1);
    assert_eq!(sym.proper_generators[&ElementOrder::Int(2)].len(), 1);
    assert_eq!(sym.improper_generators[&ElementOrder::Int(1)].len(), 1);
    assert_eq!(sym.get_sigma_generators("h").unwrap().len(), 1);
}

#[test]
fn test_point_group_detection_xef4_d4h() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/xef4.xyz");
    let mol = Molecule::from_xyz(&path, 1e-7);
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::builder().build().unwrap();
    sym.analyse(&presym);
    assert_eq!(sym.point_group, Some("D4h".to_owned()));
    assert_eq!(sym.proper_elements[&ElementOrder::Int(4)].len(), 1);
    assert_eq!(sym.proper_elements[&ElementOrder::Int(2)].len(), 5);
    assert_eq!(sym.improper_elements[&ElementOrder::Int(1)].len(), 5);
    assert_eq!(sym.improper_elements[&ElementOrder::Int(2)].len(), 1);
    assert_eq!(sym.improper_elements[&ElementOrder::Int(4)].len(), 1);
    assert_eq!(sym.get_sigma_elements("h").unwrap().len(), 1);
    assert_eq!(sym.get_sigma_elements("v").unwrap().len(), 4);

    assert_eq!(sym.proper_generators[&ElementOrder::Int(4)].len(), 1);
    assert_eq!(sym.proper_generators[&ElementOrder::Int(2)].len(), 1);
    assert_eq!(sym.improper_generators[&ElementOrder::Int(1)].len(), 1);
    assert_eq!(sym.get_sigma_generators("h").unwrap().len(), 1);
}

#[test]
fn test_point_group_detection_eclipsed_ferrocene_d5h() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/eclipsed_ferrocene.xyz");
    let mol = Molecule::from_xyz(&path, 1e-7);
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::builder().build().unwrap();
    sym.analyse(&presym);
    assert_eq!(sym.point_group, Some("D5h".to_owned()));
    assert_eq!(sym.proper_elements[&ElementOrder::Int(5)].len(), 1);
    assert_eq!(sym.proper_elements[&ElementOrder::Int(2)].len(), 5);
    assert_eq!(sym.improper_elements[&ElementOrder::Int(1)].len(), 6);
    assert_eq!(sym.improper_elements[&ElementOrder::Int(5)].len(), 1);
    assert_eq!(sym.get_sigma_elements("h").unwrap().len(), 1);
    assert_eq!(sym.get_sigma_elements("v").unwrap().len(), 5);

    assert_eq!(sym.proper_generators[&ElementOrder::Int(5)].len(), 1);
    assert_eq!(sym.proper_generators[&ElementOrder::Int(2)].len(), 1);
    assert_eq!(sym.improper_generators[&ElementOrder::Int(1)].len(), 1);
    assert_eq!(sym.get_sigma_generators("h").unwrap().len(), 1);
}

#[test]
fn test_point_group_detection_benzene_d6h() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/benzene.xyz");
    let mol = Molecule::from_xyz(&path, 1e-7);
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::builder().build().unwrap();
    sym.analyse(&presym);
    assert_eq!(sym.point_group, Some("D6h".to_owned()));
    assert_eq!(sym.proper_elements[&ElementOrder::Int(6)].len(), 1);
    assert_eq!(sym.proper_elements[&ElementOrder::Int(2)].len(), 7);
    assert_eq!(sym.improper_elements[&ElementOrder::Int(1)].len(), 7);
    assert_eq!(sym.improper_elements[&ElementOrder::Int(2)].len(), 1);
    assert_eq!(sym.improper_elements[&ElementOrder::Int(6)].len(), 1);
    assert_eq!(sym.get_sigma_elements("h").unwrap().len(), 1);
    assert_eq!(sym.get_sigma_elements("v").unwrap().len(), 6);

    assert_eq!(sym.proper_generators[&ElementOrder::Int(6)].len(), 1);
    assert_eq!(sym.proper_generators[&ElementOrder::Int(2)].len(), 1);
    assert_eq!(sym.improper_generators[&ElementOrder::Int(1)].len(), 1);
    assert_eq!(sym.get_sigma_generators("h").unwrap().len(), 1);
}

#[test]
fn test_point_group_detection_arbitrary_eclipsed_sandwich_dnh() {
    // env_logger::init();
    for n in 3..=20 {
        let mol = template_molecules::gen_arbitrary_eclipsed_sandwich(n);
        let presym = PreSymmetry::builder()
            .moi_threshold(1e-7)
            .molecule(&mol, true)
            .build()
            .unwrap();
        let mut sym = Symmetry::builder().build().unwrap();
        sym.analyse(&presym);
        assert_eq!(sym.point_group, Some(format!("D{n}h")));
        assert_eq!(sym.proper_elements[&ElementOrder::Int(n)].len(), 1);
        assert_eq!(sym.improper_elements[&ElementOrder::Int(1)].len() as u32, n + 1);
        assert_eq!(sym.improper_elements[&ElementOrder::Int(n)].len(), 1);
        assert_eq!(sym.get_sigma_elements("h").unwrap().len() as u32, 1);
        assert_eq!(sym.get_sigma_elements("v").unwrap().len() as u32, n);
        if n % 2 == 0 {
            assert_eq!(sym.improper_elements[&ElementOrder::Int(2)].len(), 1);
            assert_eq!(sym.proper_elements[&ElementOrder::Int(2)].len() as u32, n + 1);
        } else {
        }

        assert_eq!(sym.proper_generators[&ElementOrder::Int(n)].len(), 1);
        assert_eq!(sym.proper_generators[&ElementOrder::Int(2)].len(), 1);
        assert_eq!(sym.improper_generators[&ElementOrder::Int(1)].len(), 1);
        assert_eq!(sym.get_sigma_generators("h").unwrap().len(), 1);
    }
}


/*
Dnd
*/

#[test]
fn test_point_group_detection_b2cl4_d2d() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/b2cl4.xyz");
    let mol = Molecule::from_xyz(&path, 1e-7);
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::builder().build().unwrap();
    sym.analyse(&presym);
    assert_eq!(sym.point_group, Some("D2d".to_owned()));
    assert_eq!(sym.proper_elements[&ElementOrder::Int(2)].len(), 3);
    assert_eq!(sym.improper_elements[&ElementOrder::Int(1)].len(), 2);
    assert_eq!(sym.improper_elements[&ElementOrder::Int(4)].len(), 1);
    assert_eq!(sym.get_sigma_elements("d").unwrap().len(), 2);

    assert_eq!(sym.proper_generators[&ElementOrder::Int(2)].len(), 2);
    assert_eq!(sym.improper_generators[&ElementOrder::Int(1)].len(), 1);
    assert_eq!(sym.get_sigma_generators("d").unwrap().len(), 1);
}

#[test]
fn test_point_group_detection_staggered_c2h6_d3d() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c2h6.xyz");
    let mol = Molecule::from_xyz(&path, 1e-7);
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::builder().build().unwrap();
    sym.analyse(&presym);
    assert_eq!(sym.point_group, Some("D3d".to_owned()));
    assert_eq!(sym.proper_elements[&ElementOrder::Int(3)].len(), 1);
    assert_eq!(sym.proper_elements[&ElementOrder::Int(2)].len(), 3);
    assert_eq!(sym.improper_elements[&ElementOrder::Int(1)].len(), 3);
    assert_eq!(sym.improper_elements[&ElementOrder::Int(2)].len(), 1);
    assert_eq!(sym.improper_elements[&ElementOrder::Int(6)].len(), 1);
    assert_eq!(sym.get_sigma_elements("d").unwrap().len(), 3);

    assert_eq!(sym.proper_generators[&ElementOrder::Int(3)].len(), 1);
    assert_eq!(sym.proper_generators[&ElementOrder::Int(2)].len(), 1);
    assert_eq!(sym.improper_generators[&ElementOrder::Int(1)].len(), 1);
    assert_eq!(sym.get_sigma_generators("d").unwrap().len(), 1);
}

#[test]
fn test_point_group_detection_s8_d4d() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/s8.xyz");
    let mol = Molecule::from_xyz(&path, 1e-7);
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::builder().build().unwrap();
    sym.analyse(&presym);
    assert_eq!(sym.point_group, Some("D4d".to_owned()));
    assert_eq!(sym.proper_elements[&ElementOrder::Int(4)].len(), 1);
    assert_eq!(sym.proper_elements[&ElementOrder::Int(2)].len(), 5);
    assert_eq!(sym.improper_elements[&ElementOrder::Int(1)].len(), 4);
    assert_eq!(sym.improper_elements[&ElementOrder::Int(8)].len(), 1);
    assert_eq!(sym.get_sigma_elements("d").unwrap().len(), 4);

    assert_eq!(sym.proper_generators[&ElementOrder::Int(4)].len(), 1);
    assert_eq!(sym.proper_generators[&ElementOrder::Int(2)].len(), 1);
    assert_eq!(sym.improper_generators[&ElementOrder::Int(1)].len(), 1);
    assert_eq!(sym.get_sigma_generators("d").unwrap().len(), 1);
}

#[test]
fn test_point_group_detection_staggered_ferrocene_d5d() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/staggered_ferrocene.xyz");
    let mol = Molecule::from_xyz(&path, 1e-6);
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::builder().build().unwrap();
    sym.analyse(&presym);
    assert_eq!(sym.point_group, Some("D5d".to_owned()));
    assert_eq!(sym.proper_elements[&ElementOrder::Int(5)].len(), 1);
    assert_eq!(sym.proper_elements[&ElementOrder::Int(2)].len(), 5);
    assert_eq!(sym.improper_elements[&ElementOrder::Int(1)].len(), 5);
    assert_eq!(sym.improper_elements[&ElementOrder::Int(2)].len(), 1);
    assert_eq!(sym.improper_elements[&ElementOrder::Int(10)].len(), 1);
    assert_eq!(sym.get_sigma_elements("d").unwrap().len(), 5);

    assert_eq!(sym.proper_generators[&ElementOrder::Int(5)].len(), 1);
    assert_eq!(sym.proper_generators[&ElementOrder::Int(2)].len(), 1);
    assert_eq!(sym.improper_generators[&ElementOrder::Int(1)].len(), 1);
    assert_eq!(sym.get_sigma_generators("d").unwrap().len(), 1);
}

#[test]
fn test_point_group_detection_au26_d6d() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/au26.xyz");
    let mol = Molecule::from_xyz(&path, 1e-6);
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::builder().build().unwrap();
    sym.analyse(&presym);
    assert_eq!(sym.point_group, Some("D6d".to_owned()));
    assert_eq!(sym.proper_elements[&ElementOrder::Int(6)].len(), 1);
    assert_eq!(sym.proper_elements[&ElementOrder::Int(2)].len(), 7);
    assert_eq!(sym.improper_elements[&ElementOrder::Int(1)].len(), 6);
    assert_eq!(sym.improper_elements[&ElementOrder::Int(12)].len(), 1);
    assert_eq!(sym.get_sigma_elements("d").unwrap().len(), 6);

    assert_eq!(sym.proper_generators[&ElementOrder::Int(6)].len(), 1);
    assert_eq!(sym.proper_generators[&ElementOrder::Int(2)].len(), 1);
    assert_eq!(sym.improper_generators[&ElementOrder::Int(1)].len(), 1);
    assert_eq!(sym.get_sigma_generators("d").unwrap().len(), 1);
}

#[test]
fn test_point_group_detection_arbitrary_staggered_sandwich_dnd() {
    // env_logger::init();
    for n in 3..=20 {
        let mol = template_molecules::gen_arbitrary_twisted_sandwich(n, 0.5);
        let presym = PreSymmetry::builder()
            .moi_threshold(1e-7)
            .molecule(&mol, true)
            .build()
            .unwrap();
        let mut sym = Symmetry::builder().build().unwrap();
        sym.analyse(&presym);
        assert_eq!(sym.point_group, Some(format!("D{n}d")));
        assert_eq!(sym.proper_elements[&ElementOrder::Int(n)].len(), 1);
        assert_eq!(sym.improper_elements[&ElementOrder::Int(1)].len() as u32, n);
        assert_eq!(sym.improper_elements[&ElementOrder::Int(2*n)].len(), 1);
        assert_eq!(sym.get_sigma_elements("d").unwrap().len() as u32, n);
        if n % 2 == 0 {
            assert_eq!(sym.proper_elements[&ElementOrder::Int(2)].len() as u32, n + 1);
        } else {
            assert_eq!(sym.proper_elements[&ElementOrder::Int(2)].len() as u32, n);
            assert_eq!(sym.improper_elements[&ElementOrder::Int(2)].len(), 1);
        };

        assert_eq!(sym.proper_generators[&ElementOrder::Int(n)].len(), 1);
        assert_eq!(sym.proper_generators[&ElementOrder::Int(2)].len(), 1);
        assert_eq!(sym.improper_generators[&ElementOrder::Int(1)].len(), 1);
        assert_eq!(sym.get_sigma_generators("d").unwrap().len(), 1);
    }
}


/*
S2n
*/

#[test]
fn test_point_group_detection_b2cl4_magnetic_field_s4() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/b2cl4.xyz");
    let mut mol = Molecule::from_xyz(&path, 1e-7);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::builder().build().unwrap();
    sym.analyse(&presym);
    assert_eq!(sym.point_group, Some("S4".to_owned()));
    assert_eq!(sym.proper_elements[&ElementOrder::Int(2)].len(), 1);
    assert_eq!(sym.improper_elements[&ElementOrder::Int(4)].len(), 1);

    assert_eq!(sym.improper_generators[&ElementOrder::Int(4)].len(), 1);
}

#[test]
fn test_point_group_detection_staggered_c2h6_magnetic_field_s6() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c2h6.xyz");
    let mut mol = Molecule::from_xyz(&path, 1e-7);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::builder().build().unwrap();
    sym.analyse(&presym);
    assert_eq!(sym.point_group, Some("S6".to_owned()));
    assert_eq!(sym.proper_elements[&ElementOrder::Int(3)].len(), 1);
    assert_eq!(sym.improper_elements[&ElementOrder::Int(6)].len(), 1);
    assert_eq!(sym.improper_elements[&ElementOrder::Int(2)].len(), 1);

    assert_eq!(sym.improper_generators[&ElementOrder::Int(6)].len(), 1);
}

#[test]
fn test_point_group_detection_s8_magnetic_field_s8() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/s8.xyz");
    let mut mol = Molecule::from_xyz(&path, 1e-7);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::builder().build().unwrap();
    sym.analyse(&presym);
    assert_eq!(sym.point_group, Some("S8".to_owned()));
    assert_eq!(sym.proper_elements[&ElementOrder::Int(4)].len(), 1);
    assert_eq!(sym.improper_elements[&ElementOrder::Int(8)].len(), 1);

    assert_eq!(sym.improper_generators[&ElementOrder::Int(8)].len(), 1);
}

#[test]
fn test_point_group_detection_staggered_ferrocene_magnetic_field_s10() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/staggered_ferrocene.xyz");
    let mut mol = Molecule::from_xyz(&path, 1e-6);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::builder().build().unwrap();
    sym.analyse(&presym);
    assert_eq!(sym.point_group, Some("S10".to_owned()));
    assert_eq!(sym.proper_elements[&ElementOrder::Int(5)].len(), 1);
    assert_eq!(sym.improper_elements[&ElementOrder::Int(10)].len(), 1);
    assert_eq!(sym.improper_elements[&ElementOrder::Int(2)].len(), 1);

    assert_eq!(sym.improper_generators[&ElementOrder::Int(10)].len(), 1);
}

#[test]
fn test_point_group_detection_au26_magnetic_field_s12() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/au26.xyz");
    let mut mol = Molecule::from_xyz(&path, 1e-6);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::builder().build().unwrap();
    sym.analyse(&presym);
    assert_eq!(sym.point_group, Some("S12".to_owned()));
    assert_eq!(sym.proper_elements[&ElementOrder::Int(6)].len(), 1);
    assert_eq!(sym.improper_elements[&ElementOrder::Int(12)].len(), 1);

    assert_eq!(sym.improper_generators[&ElementOrder::Int(12)].len(), 1);
}

#[test]
fn test_point_group_detection_arbitrary_staggered_sandwich_magnetic_field_s2n() {
    // env_logger::init();
    for n in 3..=20 {
        let mut mol = template_molecules::gen_arbitrary_twisted_sandwich(n, 0.5);
        mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
        let presym = PreSymmetry::builder()
            .moi_threshold(1e-7)
            .molecule(&mol, true)
            .build()
            .unwrap();
        let mut sym = Symmetry::builder().build().unwrap();
        sym.analyse(&presym);
        assert_eq!(sym.point_group, Some(format!("S{}", 2*n)));
        assert_eq!(sym.proper_elements[&ElementOrder::Int(n)].len(), 1);
        assert_eq!(sym.improper_elements[&ElementOrder::Int(2*n)].len(), 1);
        if n % 2 == 1 {
            assert_eq!(sym.improper_elements[&ElementOrder::Int(2)].len(), 1);
        };

        assert_eq!(sym.improper_generators[&ElementOrder::Int(2*n)].len(), 1);
    }
}
