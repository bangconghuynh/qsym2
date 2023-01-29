use nalgebra::Vector3;

use super::*;
use crate::aux::atom::{Atom, AtomKind};
use crate::aux::molecule::Molecule;
use crate::symmetry::symmetry_symbols::MullikenIrrepSymbol;

const ROOT: &str = env!("CARGO_MANIFEST_DIR");

/*
Dnh
*/
#[test]
fn test_character_table_construction_symmetric_bf3_rad_magnetic_field_bw_d3h_d3() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/bf3.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    let magnetic_atoms: Vec<Atom> = mol
        .atoms
        .iter()
        .filter_map(|atom| {
            if atom.atomic_symbol == "F" {
                Some([
                    Atom::new_special(
                        AtomKind::Magnetic(true),
                        1.1 * atom.coordinates,
                        thresh,
                    )
                    .expect("Unable to construct a special magnetic atom."),
                    Atom::new_special(
                        AtomKind::Magnetic(false),
                        0.9 * atom.coordinates,
                        thresh,
                    )
                    .expect("Unable to construct a special magnetic atom."),
                ])
            } else {
                None
            }
        })
        .flatten()
        .collect();
    mol.magnetic_atoms = Some(magnetic_atoms);
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|^(')_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||A|^(')_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||E|^(')|").unwrap(),
        MullikenIrrepSymbol::new("||A|^('')_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||A|^('')_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||E|^('')|").unwrap(),
    ];
    test_character_table_construction_magnetic(
        &mol,
        thresh,
        &expected_irreps,
        None,
    );
}

#[test]
fn test_character_table_construction_symmetric_bf3_alt_magnetic_field_bw_c2v_cs() {
    env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/bf3.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    let magnetic_atoms: Vec<Atom> = mol
        .atoms
        .iter()
        .enumerate()
        .filter_map(|(i, atom)| {
            if atom.atomic_symbol == "F" {
                Some([
                    Atom::new_special(
                        AtomKind::Magnetic(i % 2 == 0),
                        atom.coordinates + Vector3::new(0.0, 0.0, 1.0),
                        thresh,
                    )
                    .expect("Unable to construct a special magnetic atom."),
                    Atom::new_special(
                        AtomKind::Magnetic(i % 2 != 0),
                        atom.coordinates + Vector3::new(0.0, 0.0, -1.0),
                        thresh,
                    )
                    .expect("Unable to construct a special magnetic atom."),
                ])
            } else {
                None
            }
        })
        .flatten()
        .collect();
    mol.magnetic_atoms = Some(magnetic_atoms);
    verify_bw_c2v_cs(&mol, thresh);
}

#[test]
fn test_character_table_construction_symmetric_xef4_alt_magnetic_field_bw_d4h_d2h() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/xef4.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    let magnetic_atoms: Vec<Atom> = mol
        .atoms
        .iter()
        .enumerate()
        .filter_map(|(i, atom)| {
            if atom.atomic_symbol == "F" {
                Some([
                    Atom::new_special(
                        AtomKind::Magnetic(i % 2 == 0),
                        atom.coordinates + Vector3::new(0.0, 0.0, 1.0),
                        thresh,
                    )
                    .expect("Unable to construct a special magnetic atom."),
                    Atom::new_special(
                        AtomKind::Magnetic(i % 2 != 0),
                        atom.coordinates + Vector3::new(0.0, 0.0, -1.0),
                        thresh,
                    )
                    .expect("Unable to construct a special magnetic atom."),
                ])
            } else {
                None
            }
        })
        .flatten()
        .collect();
    mol.magnetic_atoms = Some(magnetic_atoms);

    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(1g)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(2g)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(1g)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(2g)|").unwrap(),
        MullikenIrrepSymbol::new("||E|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(1u)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(2u)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(1u)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(2u)|").unwrap(),
        MullikenIrrepSymbol::new("||E|_(u)|").unwrap(),
    ];
    test_character_table_construction_magnetic(
        &mol,
        thresh,
        &expected_irreps,
        None,
    );
}
