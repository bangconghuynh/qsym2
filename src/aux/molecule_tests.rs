use crate::aux::atom::{Atom, ElementMap};
use crate::aux::geometry::{Transform, IMINV, IMSIG};
use crate::aux::molecule::Molecule;
use std::collections::HashSet;
use nalgebra::Vector3;
use approx;

const ROOT: &str = env!("CARGO_MANIFEST_DIR");

#[test]
fn test_atom_comparisons() {
    let emap = ElementMap::new();
    let atom_0 = Atom::from_xyz("B 0.0 0.0 1.45823", &emap, 1e-4).unwrap();
    let atom_1 = Atom::from_xyz("B 0.0 0.0 1.45824", &emap, 1e-4).unwrap();
    let atom_2 = Atom::from_xyz("B 0.0 0.0 1.45825", &emap, 1e-4).unwrap();
    let atom_3 = Atom::from_xyz("B 0.0 0.0 1.45826", &emap, 1e-4).unwrap();
    let atom_4 = Atom::from_xyz("B 0.0 0.0 1.45827", &emap, 1e-4).unwrap();
    assert_eq!(atom_0, atom_1);
    assert_ne!(atom_0, atom_2);
    assert_ne!(atom_1, atom_2);
    assert_ne!(atom_1, atom_3);
    assert_eq!(atom_2, atom_3);
    assert_eq!(atom_2, atom_4);
    assert_eq!(atom_3, atom_4);
    let atom_set = HashSet::from([atom_0, atom_1, atom_2, atom_3, atom_4]);
    assert_eq!(atom_set.len(), 2);
}

#[test]
fn test_transform_c60() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c60.xyz");
    let mut mol = Molecule::from_xyz(&path, 1e-6);
    mol.recentre_mut();
    let rotated_mol = mol.rotate(2.0 * std::f64::consts::PI / 5.0, &Vector3::new(0.0, 0.0, 1.0));
    assert_eq!(mol, rotated_mol);
}

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
fn test_transform_n3() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/n3.xyz");
    let mut mol = Molecule::from_xyz(&path, 1e-7);
    mol.recentre_mut();
    let rotated_mol = mol.rotate(0.1 * std::f64::consts::PI, &Vector3::new(1.0, 1.0, 1.0));
    assert_eq!(mol, rotated_mol);

    let rotated_mol2 = mol.rotate(std::f64::consts::PI, &Vector3::new(1.0, -1.0, 0.0));
    assert_ne!(mol, rotated_mol2);

    let reflected_mol = mol.improper_rotate(
        0.0,
        &Vector3::new(1.0, -1.0, 0.0),
        &IMSIG,
    );
    assert_eq!(mol, reflected_mol);

    mol.set_magnetic_field(Some(Vector3::new(1.0, 1.0, 1.0)));
    let reflected_mol_mag = mol.improper_rotate(
        0.0,
        &Vector3::new(1.0, -1.0, 0.0),
        &IMSIG,
    );
    assert_ne!(mol, reflected_mol_mag);

    mol.set_magnetic_field(None);
    mol.set_electric_field(Some(Vector3::new(1.0, 1.0, 1.0)));
    let reflected_mol_ele = mol.improper_rotate(
        0.0,
        &Vector3::new(1.0, -1.0, 0.0),
        &IMSIG,
    );
    assert_eq!(mol, reflected_mol_ele);
}

#[test]
fn test_transform_c2h2() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c2h2.xyz");
    let mut mol = Molecule::from_xyz(&path, 1e-7);
    mol.recentre_mut();
    let rotated_mol = mol.rotate(std::f64::consts::PI, &Vector3::new(0.0, 1.0, -1.0));
    assert_eq!(mol, rotated_mol);

    let inverted_mol = mol.improper_rotate(
        0.0,
        &Vector3::new(1.0, 0.0, 0.0),
        &IMINV,
    );
    assert_eq!(mol, inverted_mol);

    mol.set_magnetic_field(Some(Vector3::new(1.0, 0.0, 0.0)));
    let inverted_mol_mag = mol.improper_rotate(
        0.0,
        &Vector3::new(1.0, 0.0, 0.0),
        &IMINV,
    );
    assert_eq!(mol, inverted_mol_mag);

    mol.set_magnetic_field(None);
    mol.set_electric_field(Some(Vector3::new(1.0, 0.0, 0.0)));
    let inverted_mol_ele = mol.improper_rotate(
        0.0,
        &Vector3::new(1.0, 0.0, 0.0),
        &IMINV,
    );
    assert_ne!(mol, inverted_mol_ele);
}

#[test]
fn test_transform_c3h3() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c3h3.xyz");
    let mut mol = Molecule::from_xyz(&path, 1e-7);
    mol.recentre_mut();
    let rotated_mol = mol.rotate(2.0 * std::f64::consts::PI, &Vector3::new(0.0, 0.0, 1.0));
    assert_eq!(mol, rotated_mol);
}

#[test]
fn test_calc_moi_h8() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/h8.xyz");
    let mol = Molecule::from_xyz(&path, 1e-7);
    let (mois, principal_axes) = mol.calc_moi();

    approx::assert_relative_eq!(
        mois[0], 4.031776,
        epsilon = 1e-6,
        max_relative = 1e-6
    );
    approx::assert_relative_eq!(
        mois[1], 10.07944,
        epsilon = 1e-6,
        max_relative = 1e-6
    );
    approx::assert_relative_eq!(
        mois[2], 10.07944,
        epsilon = 1e-6,
        max_relative = 1e-6
    );

    approx::assert_relative_eq!(
        principal_axes[0], Vector3::new(0.0, 0.0, 1.0),
        epsilon = 1e-6,
        max_relative = 1e-6
    );
}

#[test]
fn test_calc_moi_n3() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/n3.xyz");
    let mol = Molecule::from_xyz(&path, 1e-7);
    let (mois, principal_axes) = mol.calc_moi();

    approx::assert_relative_eq!(
        mois[0], 0.0,
        epsilon = 1e-13,
        max_relative = 1e-13
    );
    approx::assert_relative_eq!(
        mois[1], 196.09407999999996,
        epsilon = 1e-13,
        max_relative = 1e-13
    );
    approx::assert_relative_eq!(
        mois[2], 196.09407999999996,
        epsilon = 1e-13,
        max_relative = 1e-13
    );

    approx::assert_relative_eq!(
        principal_axes[0], Vector3::new(1.0, 1.0, 1.0) / 3.0_f64.sqrt(),
        epsilon = 1e-6,
        max_relative = 1e-6
    );
}
