use crate::aux::atom::{Atom, ElementMap};
use crate::aux::geometry::{self, Transform};
use crate::symmetry::symmetry_element::SymmetryElementKind;
use crate::aux::molecule::Molecule;
use nalgebra::{Vector3, Reflection3};

const ROOT: &str = env!("CARGO_MANIFEST_DIR");

// #[test]
// fn test_transform_c60 () {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/c60.xyz");
//     let mol = Molecule::from_xyz(&path);
//     let sea_groups = mol.calc_sea_groups(1e-4, 0);
//     assert_eq!(sea_groups.len(), 1);
// }

#[test]
fn test_transform_th() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/th.xyz");
    let mut mol = Molecule::from_xyz(&path, 1e-7);
    mol.recentre_mut();
    let emap = ElementMap::new();
    let mol_ref = Molecule {
        atoms: vec![Atom::from_xyz("Th 0 0 0", &emap, 1e-7).unwrap()],
        magnetic_atoms: None,
        electric_atoms: None,
        threshold: 1e-7,
    };
    assert_eq!(mol, mol_ref);
}

#[test]
fn test_transform_h8() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/h8.xyz");
    let mut mol = Molecule::from_xyz(&path, 1e-7);
    mol.recentre_mut();
    let rotated_mol = mol.rotate(std::f64::consts::PI / 2.0, &Vector3::new(0.0, 0.0, 1.0));
    assert_eq!(mol, rotated_mol);

    let rotated_mol2 = mol.rotate(std::f64::consts::PI / 3.0, &Vector3::new(0.0, 0.0, 1.0));
    assert_ne!(mol, rotated_mol2);

    let rotated_mol3 = mol.rotate(std::f64::consts::PI, &Vector3::new(0.0, 1.0, 0.0));
    assert_eq!(mol, rotated_mol3);

    let rotated_mol4 = mol.rotate(std::f64::consts::PI / 2.0, &Vector3::new(0.0, 1.0, 0.0));
    assert_ne!(mol, rotated_mol4);
}

#[test]
fn test_transform_n3 () {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/n3.xyz");
    let mut mol = Molecule::from_xyz(&path, 1e-7);
    mol.recentre_mut();
    let rotated_mol = mol.rotate(0.1 * std::f64::consts::PI, &Vector3::new(1.0, 1.0, 1.0));
    assert_eq!(mol, rotated_mol);

    let rotated_mol2 = mol.rotate(std::f64::consts::PI, &Vector3::new(1.0, -1.0, 0.0));
    assert_ne!(mol, rotated_mol2);
}

#[test]
fn test_transform_c3h3 () {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c3h3.xyz");
    let mut mol = Molecule::from_xyz(&path, 1e-7);
    mol.recentre_mut();
    let rotated_mol = mol.rotate(2.0 * std::f64::consts::PI, &Vector3::new(0.0, 0.0, 1.0));
    assert_eq!(mol, rotated_mol);

    println!("{}", geometry::improper_rotation_matrix(0.0, &Vector3::new(0.0, 1.0, 0.0), 1, SymmetryElementKind::ImproperMirrorPlane));
}
