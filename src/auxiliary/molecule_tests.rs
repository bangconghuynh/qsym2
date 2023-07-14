use std::collections::HashSet;

use approx;
use nalgebra::{Point3, Vector3};

use crate::auxiliary::atom::{Atom, AtomKind, ElementMap};
use crate::auxiliary::geometry::{Transform, IMINV, IMSIG};
use crate::auxiliary::molecule::Molecule;
use crate::permutation::{PermutableCollection, Permutation};

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
    let rotated_mol = mol.rotate(
        2.0 * std::f64::consts::PI / 5.0,
        &Vector3::new(0.0, 0.0, 1.0),
    );
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

    let reflected_mol = mol.improper_rotate(0.0, &Vector3::new(1.0, -1.0, 0.0), &IMSIG);
    assert_eq!(mol, reflected_mol);

    mol.set_magnetic_field(Some(Vector3::new(1.0, 1.0, 1.0)));
    let reflected_mol_mag = mol.improper_rotate(0.0, &Vector3::new(1.0, -1.0, 0.0), &IMSIG);
    assert_ne!(mol, reflected_mol_mag);

    mol.set_magnetic_field(None);
    mol.set_electric_field(Some(Vector3::new(1.0, 1.0, 1.0)));
    let reflected_mol_ele = mol.improper_rotate(0.0, &Vector3::new(1.0, -1.0, 0.0), &IMSIG);
    assert_eq!(mol, reflected_mol_ele);
}

#[test]
fn test_transform_c2h2() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c2h2.xyz");
    let mut mol = Molecule::from_xyz(&path, 1e-7);
    mol.recentre_mut();
    let rotated_mol = mol.rotate(std::f64::consts::PI, &Vector3::new(0.0, 1.0, -1.0));
    assert_eq!(mol, rotated_mol);

    let inverted_mol = mol.improper_rotate(0.0, &Vector3::new(1.0, 0.0, 0.0), &IMINV);
    assert_eq!(mol, inverted_mol);

    mol.set_magnetic_field(Some(Vector3::new(1.0, 0.0, 0.0)));
    let inverted_mol_mag = mol.improper_rotate(0.0, &Vector3::new(1.0, 0.0, 0.0), &IMINV);
    assert_eq!(mol, inverted_mol_mag);

    mol.set_magnetic_field(None);
    mol.set_electric_field(Some(Vector3::new(1.0, 0.0, 0.0)));
    let inverted_mol_ele = mol.improper_rotate(0.0, &Vector3::new(1.0, 0.0, 0.0), &IMINV);
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

    approx::assert_relative_eq!(mois[0], 4.031776, epsilon = 1e-6, max_relative = 1e-6);
    approx::assert_relative_eq!(mois[1], 10.07944, epsilon = 1e-6, max_relative = 1e-6);
    approx::assert_relative_eq!(mois[2], 10.07944, epsilon = 1e-6, max_relative = 1e-6);

    approx::assert_relative_eq!(
        principal_axes[0],
        Vector3::new(0.0, 0.0, 1.0),
        epsilon = 1e-6,
        max_relative = 1e-6
    );
}

#[test]
fn test_calc_moi_n3() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/n3.xyz");
    let mol = Molecule::from_xyz(&path, 1e-7);
    let (mois, principal_axes) = mol.calc_moi();

    approx::assert_relative_eq!(mois[0], 0.0, epsilon = 1e-13, max_relative = 1e-13);
    approx::assert_relative_eq!(
        mois[1],
        196.09407999999996,
        epsilon = 1e-13,
        max_relative = 1e-13
    );
    approx::assert_relative_eq!(
        mois[2],
        196.09407999999996,
        epsilon = 1e-13,
        max_relative = 1e-13
    );

    approx::assert_relative_eq!(
        principal_axes[0],
        Vector3::new(1.0, 1.0, 1.0) / 3.0_f64.sqrt(),
        epsilon = 1e-6,
        max_relative = 1e-6
    );
}

#[test]
fn test_reorientate_c2h2() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c2h2.xyz");
    let mut mol = Molecule::from_xyz(&path, 1e-7);
    mol.recentre_mut();
    mol.reorientate_mut(1e-7);
    let emap = ElementMap::new();
    let reoriented_mol = Molecule::from_atoms(
        &[
            Atom::new_ordinary(
                "C",
                Point3::new(0.0, 0.0, -(3.0f64.sqrt()) / 2.0),
                &emap,
                1e-7,
            ),
            Atom::new_ordinary("C", Point3::new(0.0, 0.0, 3.0f64.sqrt() / 2.0), &emap, 1e-7),
            Atom::new_ordinary(
                "H",
                Point3::new(0.0, 0.0, -(27.0f64.sqrt()) / 2.0),
                &emap,
                1e-7,
            ),
            Atom::new_ordinary(
                "H",
                Point3::new(0.0, 0.0, 27.0f64.sqrt() / 2.0),
                &emap,
                1e-7,
            ),
        ],
        1e-7,
    );
    assert_eq!(mol, reoriented_mol);
}

#[test]
fn test_reorientate_water() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/water.xyz");
    let mut mol = Molecule::from_xyz(&path, 1e-7);
    mol.recentre_mut();
    mol.reorientate_mut(1e-7);
    let emap = ElementMap::new();
    let reoriented_mol = Molecule::from_atoms(
        &[
            Atom::new_ordinary("O", Point3::new(0.0, 0.05741214, 0.0), &emap, 1e-7),
            Atom::new_ordinary("H", Point3::new(-0.79200060, -0.45566096, 0.0), &emap, 1e-7),
            Atom::new_ordinary("H", Point3::new(0.79200060, -0.45566096, 0.0), &emap, 1e-7),
        ],
        1e-7,
    );
    assert_eq!(mol, reoriented_mol);
}

#[test]
fn test_reorientate_benzene() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/benzene.xyz");
    let mut mol = Molecule::from_xyz(&path, 1e-7);
    mol.reorientate_mut(1e-7);
    let mut reoriented_mol = Molecule::from_xyz(&path, 1e-7);
    reoriented_mol.rotate_mut(
        -2.0 * std::f64::consts::PI / 3.0,
        &Vector3::new(1.0, 1.0, 1.0),
    );
    assert_eq!(mol, reoriented_mol);
}

#[test]
fn test_reorientate_vf6_field() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/vf6.xyz");
    let mut mol = Molecule::from_xyz(&path, 1e-7);
    mol.set_magnetic_field(Some(Vector3::x()));
    let mut reoriented_mol = mol.clone();
    mol.reorientate_mut(1e-7);
    reoriented_mol.rotate_mut(
        -2.0 * std::f64::consts::PI / 3.0,
        &Vector3::new(1.0, 1.0, 1.0),
    );
    assert_eq!(mol, reoriented_mol);
}

#[test]
fn test_molecule_get_perm_of() {
    let emap = ElementMap::new();
    let atom_0 = Atom::from_xyz("B 0.0 0.0 0.0", &emap, 1e-7).unwrap();
    let atom_0p = Atom::from_xyz("B 1.0 1.0 0.0", &emap, 1e-7).unwrap();
    let atom_1 = Atom::from_xyz("H 1.0 0.0 0.0", &emap, 1e-7).unwrap();
    let atom_2 = Atom::from_xyz("H 0.0 1.0 0.0", &emap, 1e-7).unwrap();
    let atom_3 = Atom::from_xyz("H -1.0 0.0 0.0", &emap, 1e-7).unwrap();
    let atom_4 = Atom::from_xyz("H 0.0 -1.0 0.0", &emap, 1e-7).unwrap();
    let mol1 = Molecule::from_atoms(
        &[
            atom_0.clone(),
            atom_1.clone(),
            atom_2.clone(),
            atom_3.clone(),
            atom_4.clone(),
        ],
        1e-7,
    );
    let mol2 = Molecule::from_atoms(
        &[
            atom_0.clone(),
            atom_2.clone(),
            atom_3.clone(),
            atom_4.clone(),
            atom_1.clone(),
        ],
        1e-7,
    );
    let mol3 = Molecule::from_atoms(
        &[
            atom_0,
            atom_1.clone(),
            atom_4.clone(),
            atom_3.clone(),
            atom_2.clone(),
        ],
        1e-7,
    );
    let mol4 = Molecule::from_atoms(&[atom_0p, atom_1, atom_4, atom_3, atom_2], 1e-7);

    assert_eq!(
        mol1.get_perm_of(&mol2),
        Some(Permutation::<usize>::from_image(vec![0, 4, 1, 2, 3]))
    );
    assert_eq!(
        mol1.get_perm_of(&mol3),
        Some(Permutation::<usize>::from_image(vec![0, 1, 4, 3, 2]))
    );
    assert_eq!(mol1.get_perm_of(&mol4), None);
    assert_eq!(
        mol2.get_perm_of(&mol1),
        Some(Permutation::<usize>::from_image(vec![0, 2, 3, 4, 1]))
    );
    assert_eq!(
        mol2.get_perm_of(&mol2),
        Some(Permutation::<usize>::from_image(vec![0, 1, 2, 3, 4]))
    );
    assert_eq!(
        mol2.get_perm_of(&mol3),
        Some(Permutation::<usize>::from_image(vec![0, 4, 3, 2, 1]))
    );
}

#[test]
fn test_molecule_permute() {
    let emap = ElementMap::new();
    let atom_0 = Atom::from_xyz("B 0.0 0.0 0.0", &emap, 1e-7).unwrap();
    let atom_1 = Atom::from_xyz("H 1.0 0.0 0.0", &emap, 1e-7).unwrap();
    let atom_2 = Atom::from_xyz("H 0.0 1.0 0.0", &emap, 1e-7).unwrap();
    let atom_3 = Atom::from_xyz("H -1.0 0.0 0.0", &emap, 1e-7).unwrap();
    let atom_4 = Atom::from_xyz("H 0.0 -1.0 0.0", &emap, 1e-7).unwrap();
    let mol1 = Molecule::from_atoms(
        &[
            atom_0.clone(),
            atom_1.clone(),
            atom_2.clone(),
            atom_3.clone(),
            atom_4.clone(),
        ],
        1e-7,
    );

    let perm = Permutation::<usize>::from_image(vec![0, 4, 3, 1, 2]);
    let mol2 = mol1.permute(&perm);
    assert_eq!(
        mol2.atoms,
        &[
            atom_0.clone(),
            atom_4.clone(),
            atom_3.clone(),
            atom_1.clone(),
            atom_2.clone(),
        ]
    );

    let perm2 = Permutation::<usize>::from_image(vec![1, 4, 3, 0, 2]);
    let mol3 = mol1.permute(&perm2);
    assert_eq!(
        mol3.atoms,
        &[
            atom_1.clone(),
            atom_4.clone(),
            atom_3.clone(),
            atom_0.clone(),
            atom_2.clone(),
        ]
    );
    assert_ne!(mol3.atoms, &[atom_0, atom_4, atom_3, atom_1, atom_2,]);
}

#[test]
fn test_molecule_permute_with_special_atoms() {
    let emap = ElementMap::new();
    let atom_0 = Atom::from_xyz("B 0.0 0.0 0.0", &emap, 1e-7).unwrap();
    let atom_1 = Atom::from_xyz("H 1.0 0.0 0.0", &emap, 1e-7).unwrap();
    let atom_2 = Atom::from_xyz("H 0.0 1.0 0.0", &emap, 1e-7).unwrap();
    let atom_3 = Atom::from_xyz("H -1.0 0.0 0.0", &emap, 1e-7).unwrap();
    let atom_4 = Atom::from_xyz("H 0.0 -1.0 0.0", &emap, 1e-7).unwrap();
    let atom_b1 =
        Atom::new_special(AtomKind::Magnetic(true), Point3::new(0.1, 0.1, 0.1), 1e-7).unwrap();
    let atom_b2 = Atom::new_special(
        AtomKind::Magnetic(false),
        Point3::new(-0.1, -0.1, -0.1),
        1e-7,
    )
    .unwrap();
    let atom_e1 =
        Atom::new_special(AtomKind::Electric(true), Point3::new(0.0, 0.0, 0.1), 1e-7).unwrap();
    let atom_e2 =
        Atom::new_special(AtomKind::Electric(false), Point3::new(0.0, 0.0, -0.1), 1e-7).unwrap();
    let mol1 = Molecule::from_atoms(
        &[
            atom_0.clone(),
            atom_1.clone(),
            atom_2.clone(),
            atom_3.clone(),
            atom_4.clone(),
            atom_b1.clone(),
            atom_b2.clone(),
            atom_e1.clone(),
        ],
        1e-7,
    );

    let perm = Permutation::<usize>::from_image(vec![0, 4, 3, 1, 2, 5, 6, 7]);
    let mol2 = mol1.permute(&perm);
    assert_eq!(
        mol2.atoms,
        &[
            atom_0.clone(),
            atom_4.clone(),
            atom_3.clone(),
            atom_1.clone(),
            atom_2.clone(),
        ]
    );
    assert_eq!(
        mol2.magnetic_atoms.unwrap(),
        &[atom_b1.clone(), atom_b2.clone(),]
    );
    assert_eq!(mol2.electric_atoms.unwrap(), &[atom_e1.clone(),]);

    let perm2 = Permutation::<usize>::from_image(vec![1, 4, 3, 0, 2, 6, 5, 7]);
    let mol3 = mol1.permute(&perm2);
    assert_eq!(
        mol3.atoms,
        &[
            atom_1.clone(),
            atom_4.clone(),
            atom_3.clone(),
            atom_0.clone(),
            atom_2.clone(),
        ]
    );
    assert_eq!(mol3.magnetic_atoms.unwrap(), &[atom_b2, atom_b1,]);
    assert_eq!(mol3.electric_atoms.unwrap(), &[atom_e1.clone(),]);

    let mol4 = Molecule::from_atoms(
        &[
            atom_0.clone(),
            atom_1.clone(),
            atom_2.clone(),
            atom_3.clone(),
            atom_4.clone(),
            atom_e1.clone(),
            atom_e2.clone(),
        ],
        1e-7,
    );

    let perm3 = Permutation::<usize>::from_image(vec![0, 1, 3, 4, 2, 6, 5]);
    let mol5 = mol4.permute(&perm3);
    assert_eq!(mol5.atoms, &[atom_0, atom_1, atom_3, atom_4, atom_2,]);
    assert_eq!(mol5.electric_atoms.unwrap(), &[atom_e2, atom_e1,]);
}
