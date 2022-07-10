use crate::aux::molecule::Molecule;
use crate::symmetry::symmetry_core::{Symmetry, PreSymmetry};
use crate::symmetry::symmetry_element::ElementOrder;
use nalgebra::Vector3;

const ROOT: &str = env!("CARGO_MANIFEST_DIR");
// use env_logger;

/*
Spherical
*/

#[test]
fn test_point_group_detection_atom() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/th.xyz");
    let mol = Molecule::from_xyz(&path, 1e-7);
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-14)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::builder()
        .build()
        .unwrap();
    sym.analyse(&presym);
    assert_eq!(sym.point_group, Some("O(3)".to_owned()));
    assert_eq!(sym.proper_generators[&ElementOrder::Inf].len(), 3);
}

#[test]
fn test_point_group_detection_c60() {
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
fn test_point_group_detection_ch4() {
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
fn test_point_group_detection_adamantane() {
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
fn test_point_group_detection_c165_diamond_nanoparticle() {
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
fn test_point_group_detection_vh2o6() {
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
fn test_point_group_detection_vf6() {
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


/*
Linear
*/

#[test]
fn test_point_group_detection_atom_magnetic_field() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/th.xyz");
    let mut mol = Molecule::from_xyz(&path, 1e-7);
    mol.set_magnetic_field(Some(Vector3::new(1.0, 2.0, -1.0)));
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-14)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::builder()
        .build()
        .unwrap();
    sym.analyse(&presym);
    assert_eq!(sym.point_group, Some("C∞h".to_owned()));
    assert_eq!(sym.proper_generators[&ElementOrder::Inf].len(), 1);
    assert_eq!(sym.improper_generators[&ElementOrder::Int(1)].len(), 1);
    assert_eq!(sym.get_sigma_generators("h").unwrap().len(), 1);
}

#[test]
fn test_point_group_detection_atom_electric_field() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/th.xyz");
    let mut mol = Molecule::from_xyz(&path, 1e-7);
    mol.set_electric_field(Some(Vector3::new(-1.0, 3.0, -2.0)));
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-14)
        .molecule(&mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::builder()
        .build()
        .unwrap();
    sym.analyse(&presym);
    assert_eq!(sym.point_group, Some("C∞v".to_owned()));
    assert_eq!(sym.proper_generators[&ElementOrder::Inf].len(), 1);
    assert_eq!(sym.improper_generators[&ElementOrder::Int(1)].len(), 1);
    assert_eq!(sym.get_sigma_generators("v").unwrap().len(), 1);
}

#[test]
fn test_point_group_detection_c2h2() {
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
fn test_point_group_detection_c2h2_magnetic_field() {
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
fn test_point_group_detection_c2h2_electric_field() {
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
fn test_point_group_detection_n3() {
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
fn test_point_group_detection_n3_magnetic_field() {
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
fn test_point_group_detection_n3_electric_field() {
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


/*
Symmetric
*/
