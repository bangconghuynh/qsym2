use std::collections::HashMap;

use approx;

use itertools::Itertools;
use nalgebra::Vector3;
use num::Complex;

use crate::aux::molecule::Molecule;
use crate::aux::template_molecules;
use crate::chartab::character::Character;
use crate::chartab::unityroot::UnityRoot;
use crate::chartab::CharacterTable;
use crate::group::group_from_molecular_symmetry;
use crate::symmetry::symmetry_core::{PreSymmetry, Symmetry};
use crate::symmetry::symmetry_element::SymmetryOperation;
use crate::symmetry::symmetry_symbols::{ClassSymbol, MathematicalSymbol, MullikenIrrepSymbol};

const ROOT: &str = env!("CARGO_MANIFEST_DIR");

// ================================================================
// Character table for abstract group from molecular symmetry tests
// ================================================================

fn test_character_table_validity(
    chartab: &CharacterTable<SymmetryOperation>,
    expected_irreps: &[MullikenIrrepSymbol],
    expected_chars_option: Option<
        HashMap<(&MullikenIrrepSymbol, &ClassSymbol<SymmetryOperation>), Character>,
    >,
) {
    // println!("{:?}", chartab);

    let order: usize = chartab
        .classes
        .keys()
        .map(|cc| cc.multiplicity().unwrap())
        .sum();

    // Sum of squared dimensions
    assert_eq!(
        order,
        chartab
            .irreps
            .keys()
            .map(|irrep| irrep.multiplicity().unwrap().pow(2))
            .sum()
    );

    // First orthogonality theorem (row-orthogonality)
    assert!(chartab
        .irreps
        .keys()
        .combinations_with_replacement(2)
        .all(|irreps_pair| {
            let irrep_i = irreps_pair[0];
            let irrep_j = irreps_pair[1];
            let i = *chartab.irreps.get(irrep_i).unwrap();
            let j = *chartab.irreps.get(irrep_j).unwrap();
            let inprod: Complex<f64> =
                chartab
                    .classes
                    .iter()
                    .fold(Complex::new(0.0f64, 0.0f64), |acc, (cc, &k)| {
                        acc + (cc.multiplicity().unwrap() as f64)
                            * chartab.characters[[i, k]].complex_value().conj()
                            * chartab.characters[[j, k]].complex_value()
                    })
                    / (order as f64);

            if i == j {
                approx::relative_eq!(inprod.re, 1.0, epsilon = 1e-14, max_relative = 1e-14)
                    && approx::relative_eq!(inprod.im, 0.0, epsilon = 1e-14, max_relative = 1e-14)
            } else {
                approx::relative_eq!(inprod.re, 0.0, epsilon = 1e-14, max_relative = 1e-14)
                    && approx::relative_eq!(inprod.im, 0.0, epsilon = 1e-14, max_relative = 1e-14)
            }
        }));

    // Second orthogonality theorem (column-orthogonality)
    assert!(chartab
        .classes
        .keys()
        .combinations_with_replacement(2)
        .all(|ccs_pair| {
            let cc_i = ccs_pair[0];
            let cc_j = ccs_pair[1];
            let i = *chartab.classes.get(cc_i).unwrap();
            let j = *chartab.classes.get(cc_j).unwrap();
            let inprod: Complex<f64> =
                chartab
                    .irreps
                    .iter()
                    .fold(Complex::new(0.0f64, 0.0f64), |acc, (_, &k)| {
                        acc + (cc_i.multiplicity().unwrap() as f64)
                            * chartab.characters[[k, i]].complex_value().conj()
                            * chartab.characters[[k, j]].complex_value()
                    })
                    / (order as f64);

            if i == j {
                approx::relative_eq!(inprod.re, 1.0, epsilon = 1e-14, max_relative = 1e-14)
                    && approx::relative_eq!(inprod.im, 0.0, epsilon = 1e-14, max_relative = 1e-14)
            } else {
                approx::relative_eq!(inprod.re, 0.0, epsilon = 1e-14, max_relative = 1e-14)
                    && approx::relative_eq!(inprod.im, 0.0, epsilon = 1e-14, max_relative = 1e-14)
            }
        }));

    // Expected irreps
    assert_eq!(
        chartab.irreps.keys().cloned().collect_vec(),
        expected_irreps
    );

    // Expected characters
    if let Some(expected_chars) = expected_chars_option {
        for ((irrep, cc), ref_char) in expected_chars.iter() {
            assert!(
                chartab.get_character(irrep, cc) == ref_char,
                "Character[({}, {})] = {} does not match {}.",
                irrep,
                cc,
                chartab.get_character(irrep, cc),
                ref_char
            );
        }
    }
}

fn test_character_table_construction(
    mol: &Molecule,
    thresh: f64,
    expected_irreps: &[MullikenIrrepSymbol],
    expected_chars_option: Option<
        HashMap<(&MullikenIrrepSymbol, &ClassSymbol<SymmetryOperation>), Character>,
    >,
) {
    let presym = PreSymmetry::builder()
        .moi_threshold(thresh)
        .molecule(mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    let group = group_from_molecular_symmetry(&sym, None);
    let chartab = group.character_table.as_ref().unwrap();
    test_character_table_validity(chartab, expected_irreps, expected_chars_option);
}

fn test_character_table_construction_magnetic(
    mol: &Molecule,
    thresh: f64,
    expected_irreps: &[MullikenIrrepSymbol],
    expected_chars_option: Option<
        HashMap<(&MullikenIrrepSymbol, &ClassSymbol<SymmetryOperation>), Character>,
    >,
) {
    let presym = PreSymmetry::builder()
        .moi_threshold(thresh)
        .molecule(mol, true)
        .build()
        .unwrap();
    let mut magsym = Symmetry::new();
    magsym.analyse(&presym, true);
    let group = group_from_molecular_symmetry(&magsym, None);
    let chartab = group.character_table.as_ref().unwrap();
    test_character_table_validity(chartab, expected_irreps, expected_chars_option);
}

fn test_character_table_construction_from_infinite_group(
    mol: &Molecule,
    finite_order: u32,
    thresh: f64,
    expected_irreps: &[MullikenIrrepSymbol],
    expected_chars_option: Option<
        HashMap<(&MullikenIrrepSymbol, &ClassSymbol<SymmetryOperation>), Character>,
    >,
) {
    let presym = PreSymmetry::builder()
        .moi_threshold(thresh)
        .molecule(mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    let group = group_from_molecular_symmetry(&sym, Some(finite_order));
    let chartab = group.character_table.as_ref().unwrap();
    test_character_table_validity(chartab, expected_irreps, expected_chars_option);
}

fn test_character_table_construction_from_infinite_magnetic_group(
    mol: &Molecule,
    finite_order: u32,
    thresh: f64,
    expected_irreps: &[MullikenIrrepSymbol],
    expected_chars_option: Option<
        HashMap<(&MullikenIrrepSymbol, &ClassSymbol<SymmetryOperation>), Character>,
    >,
) {
    let presym = PreSymmetry::builder()
        .moi_threshold(thresh)
        .molecule(mol, true)
        .build()
        .unwrap();
    let mut magsym = Symmetry::new();
    magsym.analyse(&presym, true);
    let group = group_from_molecular_symmetry(&magsym, Some(finite_order));
    let chartab = group.character_table.as_ref().unwrap();
    test_character_table_validity(chartab, expected_irreps, expected_chars_option);
}

/********
Spherical
********/
#[test]
fn test_character_table_construction_spherical_atom_o3() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/th.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);

    let d2h_expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(1g)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(2g)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(3g)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(u)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(1u)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(2u)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(3u)|").unwrap(),
    ];
    let c2 = ClassSymbol::<SymmetryOperation>::new("1||C2||", None).unwrap();
    let c2d = ClassSymbol::<SymmetryOperation>::new("1||C2|^(')|", None).unwrap();
    let d2h_expected_chars = HashMap::from([
        (
            (&d2h_expected_irreps[0], &c2),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&d2h_expected_irreps[1], &c2),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&d2h_expected_irreps[2], &c2),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&d2h_expected_irreps[3], &c2),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&d2h_expected_irreps[4], &c2),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&d2h_expected_irreps[5], &c2),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&d2h_expected_irreps[6], &c2),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&d2h_expected_irreps[7], &c2),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&d2h_expected_irreps[0], &c2d),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&d2h_expected_irreps[1], &c2d),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&d2h_expected_irreps[2], &c2d),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&d2h_expected_irreps[3], &c2d),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&d2h_expected_irreps[4], &c2d),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&d2h_expected_irreps[5], &c2d),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&d2h_expected_irreps[6], &c2d),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&d2h_expected_irreps[7], &c2d),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
    ]);
    test_character_table_construction_from_infinite_group(
        &mol,
        2,
        thresh,
        &d2h_expected_irreps,
        Some(d2h_expected_chars),
    );

    let oh_expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(1g)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(2g)|").unwrap(),
        MullikenIrrepSymbol::new("||E|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("||T|_(1g)|").unwrap(),
        MullikenIrrepSymbol::new("||T|_(2g)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(1u)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(2u)|").unwrap(),
        MullikenIrrepSymbol::new("||E|_(u)|").unwrap(),
        MullikenIrrepSymbol::new("||T|_(1u)|").unwrap(),
        MullikenIrrepSymbol::new("||T|_(2u)|").unwrap(),
    ];
    let c3 = ClassSymbol::<SymmetryOperation>::new("8||C3||", None).unwrap();
    let oh_expected_chars = HashMap::from([
        (
            (&oh_expected_irreps[0], &c3),
            Character::new(&[(UnityRoot::new(0, 3), 1)]),
        ),
        (
            (&oh_expected_irreps[1], &c3),
            Character::new(&[(UnityRoot::new(0, 3), 1)]),
        ),
        (
            (&oh_expected_irreps[2], &c3),
            Character::new(&[(UnityRoot::new(1, 3), 1), (UnityRoot::new(2, 3), 1)]),
        ),
        (
            (&oh_expected_irreps[3], &c3),
            Character::new(&[
                (UnityRoot::new(0, 3), 1),
                (UnityRoot::new(1, 3), 1),
                (UnityRoot::new(2, 3), 1),
            ]),
        ),
        (
            (&oh_expected_irreps[4], &c3),
            Character::new(&[
                (UnityRoot::new(0, 3), 1),
                (UnityRoot::new(1, 3), 1),
                (UnityRoot::new(2, 3), 1),
            ]),
        ),
        (
            (&oh_expected_irreps[5], &c3),
            Character::new(&[(UnityRoot::new(0, 3), 1)]),
        ),
        (
            (&oh_expected_irreps[6], &c3),
            Character::new(&[(UnityRoot::new(0, 3), 1)]),
        ),
        (
            (&oh_expected_irreps[7], &c3),
            Character::new(&[(UnityRoot::new(1, 3), 1), (UnityRoot::new(2, 3), 1)]),
        ),
        (
            (&oh_expected_irreps[8], &c3),
            Character::new(&[
                (UnityRoot::new(0, 3), 1),
                (UnityRoot::new(1, 3), 1),
                (UnityRoot::new(2, 3), 1),
            ]),
        ),
        (
            (&oh_expected_irreps[9], &c3),
            Character::new(&[
                (UnityRoot::new(0, 3), 1),
                (UnityRoot::new(1, 3), 1),
                (UnityRoot::new(2, 3), 1),
            ]),
        ),
    ]);
    test_character_table_construction_from_infinite_group(
        &mol,
        4,
        thresh,
        &oh_expected_irreps,
        Some(oh_expected_chars),
    );
}

#[test]
fn test_character_table_construction_spherical_atom_grey_o3() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/th.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);

    let grey_d2h_expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(1g)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(2g)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(3g)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(u)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(1u)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(2u)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(3u)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|A|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|B|_(1g)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|B|_(2g)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|B|_(3g)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|A|_(u)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|B|_(1u)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|B|_(2u)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|B|_(3u)|").unwrap(),
    ];
    let tc2 = ClassSymbol::<SymmetryOperation>::new("1||θ·C2||", None).unwrap();
    let tc2d = ClassSymbol::<SymmetryOperation>::new("1||θ·C2|^(')|", None).unwrap();
    let grey_d2h_expected_chars = HashMap::from([
        (
            (&grey_d2h_expected_irreps[0], &tc2),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&grey_d2h_expected_irreps[1], &tc2),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&grey_d2h_expected_irreps[2], &tc2),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&grey_d2h_expected_irreps[3], &tc2),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&grey_d2h_expected_irreps[4], &tc2),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&grey_d2h_expected_irreps[5], &tc2),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&grey_d2h_expected_irreps[6], &tc2),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&grey_d2h_expected_irreps[7], &tc2),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&grey_d2h_expected_irreps[8], &tc2),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&grey_d2h_expected_irreps[9], &tc2),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&grey_d2h_expected_irreps[10], &tc2),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&grey_d2h_expected_irreps[11], &tc2),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&grey_d2h_expected_irreps[12], &tc2),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&grey_d2h_expected_irreps[13], &tc2),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&grey_d2h_expected_irreps[14], &tc2),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&grey_d2h_expected_irreps[15], &tc2),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&grey_d2h_expected_irreps[0], &tc2d),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&grey_d2h_expected_irreps[1], &tc2d),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&grey_d2h_expected_irreps[2], &tc2d),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&grey_d2h_expected_irreps[3], &tc2d),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&grey_d2h_expected_irreps[4], &tc2d),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&grey_d2h_expected_irreps[5], &tc2d),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&grey_d2h_expected_irreps[6], &tc2d),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&grey_d2h_expected_irreps[7], &tc2d),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&grey_d2h_expected_irreps[8], &tc2d),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&grey_d2h_expected_irreps[9], &tc2d),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&grey_d2h_expected_irreps[10], &tc2d),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&grey_d2h_expected_irreps[11], &tc2d),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&grey_d2h_expected_irreps[12], &tc2d),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&grey_d2h_expected_irreps[13], &tc2d),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&grey_d2h_expected_irreps[14], &tc2d),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&grey_d2h_expected_irreps[15], &tc2d),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
    ]);
    test_character_table_construction_from_infinite_magnetic_group(
        &mol,
        2,
        thresh,
        &grey_d2h_expected_irreps,
        Some(grey_d2h_expected_chars),
    );

    let grey_oh_expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(1g)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(2g)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(1u)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(2u)|").unwrap(),
        MullikenIrrepSymbol::new("||E|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("||E|_(u)|").unwrap(),
        MullikenIrrepSymbol::new("||T|_(1g)|").unwrap(),
        MullikenIrrepSymbol::new("||T|_(2g)|").unwrap(),
        MullikenIrrepSymbol::new("||T|_(1u)|").unwrap(),
        MullikenIrrepSymbol::new("||T|_(2u)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|A|_(1g)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|A|_(2g)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|A|_(1u)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|A|_(2u)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|E|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|E|_(u)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|T|_(1g)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|T|_(2g)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|T|_(1u)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|T|_(2u)|").unwrap(),
    ];
    let tc3 = ClassSymbol::<SymmetryOperation>::new("8||θ·C3||", None).unwrap();
    let grey_oh_expected_chars = HashMap::from([
        (
            (&grey_oh_expected_irreps[0], &tc3),
            Character::new(&[(UnityRoot::new(0, 3), 1)]),
        ),
        (
            (&grey_oh_expected_irreps[1], &tc3),
            Character::new(&[(UnityRoot::new(0, 3), 1)]),
        ),
        (
            (&grey_oh_expected_irreps[4], &tc3),
            Character::new(&[(UnityRoot::new(1, 3), 1), (UnityRoot::new(2, 3), 1)]),
        ),
        (
            (&grey_oh_expected_irreps[6], &tc3),
            Character::new(&[
                (UnityRoot::new(0, 3), 1),
                (UnityRoot::new(1, 3), 1),
                (UnityRoot::new(2, 3), 1),
            ]),
        ),
        (
            (&grey_oh_expected_irreps[7], &tc3),
            Character::new(&[
                (UnityRoot::new(0, 3), 1),
                (UnityRoot::new(1, 3), 1),
                (UnityRoot::new(2, 3), 1),
            ]),
        ),
        (
            (&grey_oh_expected_irreps[2], &tc3),
            Character::new(&[(UnityRoot::new(0, 3), 1)]),
        ),
        (
            (&grey_oh_expected_irreps[3], &tc3),
            Character::new(&[(UnityRoot::new(0, 3), 1)]),
        ),
        (
            (&grey_oh_expected_irreps[5], &tc3),
            Character::new(&[(UnityRoot::new(1, 3), 1), (UnityRoot::new(2, 3), 1)]),
        ),
        (
            (&grey_oh_expected_irreps[8], &tc3),
            Character::new(&[
                (UnityRoot::new(0, 3), 1),
                (UnityRoot::new(1, 3), 1),
                (UnityRoot::new(2, 3), 1),
            ]),
        ),
        (
            (&grey_oh_expected_irreps[9], &tc3),
            Character::new(&[
                (UnityRoot::new(0, 3), 1),
                (UnityRoot::new(1, 3), 1),
                (UnityRoot::new(2, 3), 1),
            ]),
        ),
        (
            (&grey_oh_expected_irreps[10], &tc3),
            Character::new(&[(UnityRoot::new(1, 3), 1), (UnityRoot::new(2, 3), 1)]),
        ),
        (
            (&grey_oh_expected_irreps[11], &tc3),
            Character::new(&[(UnityRoot::new(1, 3), 1), (UnityRoot::new(2, 3), 1)]),
        ),
        (
            (&grey_oh_expected_irreps[14], &tc3),
            Character::new(&[(UnityRoot::new(0, 3), 1)]),
        ),
        (
            (&grey_oh_expected_irreps[16], &tc3),
            Character::new(&[
                (UnityRoot::new(0, 3), 1),
                (UnityRoot::new(1, 3), 1),
                (UnityRoot::new(2, 3), 1),
            ]),
        ),
        (
            (&grey_oh_expected_irreps[17], &tc3),
            Character::new(&[
                (UnityRoot::new(0, 3), 1),
                (UnityRoot::new(1, 3), 1),
                (UnityRoot::new(2, 3), 1),
            ]),
        ),
        (
            (&grey_oh_expected_irreps[12], &tc3),
            Character::new(&[(UnityRoot::new(1, 3), 1), (UnityRoot::new(2, 3), 1)]),
        ),
        (
            (&grey_oh_expected_irreps[13], &tc3),
            Character::new(&[(UnityRoot::new(1, 3), 1), (UnityRoot::new(2, 3), 1)]),
        ),
        (
            (&grey_oh_expected_irreps[15], &tc3),
            Character::new(&[(UnityRoot::new(0, 3), 1)]),
        ),
        (
            (&grey_oh_expected_irreps[18], &tc3),
            Character::new(&[
                (UnityRoot::new(0, 3), 1),
                (UnityRoot::new(1, 3), 1),
                (UnityRoot::new(2, 3), 1),
            ]),
        ),
        (
            (&grey_oh_expected_irreps[19], &tc3),
            Character::new(&[
                (UnityRoot::new(0, 3), 1),
                (UnityRoot::new(1, 3), 1),
                (UnityRoot::new(2, 3), 1),
            ]),
        ),
    ]);
    test_character_table_construction_from_infinite_magnetic_group(
        &mol,
        4,
        thresh,
        &grey_oh_expected_irreps,
        Some(grey_oh_expected_chars),
    );
}

#[test]
fn test_character_table_construction_spherical_c60_ih() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c60.xyz");
    let thresh = 1e-5;
    let mol = Molecule::from_xyz(&path, thresh);
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("||T|_(1g)|").unwrap(),
        MullikenIrrepSymbol::new("||T|_(2g)|").unwrap(),
        MullikenIrrepSymbol::new("||G|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("||H|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(u)|").unwrap(),
        MullikenIrrepSymbol::new("||T|_(1u)|").unwrap(),
        MullikenIrrepSymbol::new("||T|_(2u)|").unwrap(),
        MullikenIrrepSymbol::new("||G|_(u)|").unwrap(),
        MullikenIrrepSymbol::new("||H|_(u)|").unwrap(),
    ];
    let c5 = ClassSymbol::<SymmetryOperation>::new("12||C5||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &c5),
            Character::new(&[(UnityRoot::new(0, 10), 1)]),
        ),
        (
            (&expected_irreps[1], &c5),
            Character::new(&[
                (UnityRoot::new(0, 10), 1),
                (UnityRoot::new(2, 10), 1),
                (UnityRoot::new(8, 10), 1),
            ]),
        ),
        (
            (&expected_irreps[2], &c5),
            Character::new(&[
                (UnityRoot::new(0, 10), 1),
                (UnityRoot::new(4, 10), 1),
                (UnityRoot::new(6, 10), 1),
            ]),
        ),
        (
            (&expected_irreps[3], &c5),
            Character::new(&[(UnityRoot::new(5, 10), 1)]),
        ),
        (
            (&expected_irreps[4], &c5),
            Character::new(&[(UnityRoot::new(0, 10), 1), (UnityRoot::new(5, 10), 1)]),
        ),
        (
            (&expected_irreps[5], &c5),
            Character::new(&[(UnityRoot::new(0, 10), 1)]),
        ),
        (
            (&expected_irreps[6], &c5),
            Character::new(&[
                (UnityRoot::new(0, 10), 1),
                (UnityRoot::new(2, 10), 1),
                (UnityRoot::new(8, 10), 1),
            ]),
        ),
        (
            (&expected_irreps[7], &c5),
            Character::new(&[
                (UnityRoot::new(0, 10), 1),
                (UnityRoot::new(4, 10), 1),
                (UnityRoot::new(6, 10), 1),
            ]),
        ),
        (
            (&expected_irreps[8], &c5),
            Character::new(&[(UnityRoot::new(5, 10), 1)]),
        ),
        (
            (&expected_irreps[9], &c5),
            Character::new(&[(UnityRoot::new(0, 10), 1), (UnityRoot::new(5, 10), 1)]),
        ),
    ]);
    test_character_table_construction(&mol, thresh, &expected_irreps, Some(expected_chars));
}

#[test]
fn test_character_table_construction_spherical_c60_grey_ih() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c60.xyz");
    let thresh = 1e-5;
    let mol = Molecule::from_xyz(&path, thresh);
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(u)|").unwrap(),
        MullikenIrrepSymbol::new("||T|_(1g)|").unwrap(),
        MullikenIrrepSymbol::new("||T|_(2g)|").unwrap(),
        MullikenIrrepSymbol::new("||T|_(1u)|").unwrap(),
        MullikenIrrepSymbol::new("||T|_(2u)|").unwrap(),
        MullikenIrrepSymbol::new("||G|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("||G|_(u)|").unwrap(),
        MullikenIrrepSymbol::new("||H|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("||H|_(u)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|A|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|A|_(u)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|T|_(1g)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|T|_(2g)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|T|_(1u)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|T|_(2u)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|G|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|G|_(u)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|H|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|H|_(u)|").unwrap(),
    ];
    let tc5 = ClassSymbol::<SymmetryOperation>::new("12||θ·C5||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &tc5),
            Character::new(&[(UnityRoot::new(0, 10), 1)]),
        ),
        (
            (&expected_irreps[2], &tc5),
            Character::new(&[
                (UnityRoot::new(0, 10), 1),
                (UnityRoot::new(2, 10), 1),
                (UnityRoot::new(8, 10), 1),
            ]),
        ),
        (
            (&expected_irreps[3], &tc5),
            Character::new(&[
                (UnityRoot::new(0, 10), 1),
                (UnityRoot::new(4, 10), 1),
                (UnityRoot::new(6, 10), 1),
            ]),
        ),
        (
            (&expected_irreps[6], &tc5),
            Character::new(&[(UnityRoot::new(5, 10), 1)]),
        ),
        (
            (&expected_irreps[8], &tc5),
            Character::new(&[(UnityRoot::new(0, 10), 1), (UnityRoot::new(5, 10), 1)]),
        ),
        (
            (&expected_irreps[1], &tc5),
            Character::new(&[(UnityRoot::new(0, 10), 1)]),
        ),
        (
            (&expected_irreps[4], &tc5),
            Character::new(&[
                (UnityRoot::new(0, 10), 1),
                (UnityRoot::new(2, 10), 1),
                (UnityRoot::new(8, 10), 1),
            ]),
        ),
        (
            (&expected_irreps[5], &tc5),
            Character::new(&[
                (UnityRoot::new(0, 10), 1),
                (UnityRoot::new(4, 10), 1),
                (UnityRoot::new(6, 10), 1),
            ]),
        ),
        (
            (&expected_irreps[7], &tc5),
            Character::new(&[(UnityRoot::new(5, 10), 1)]),
        ),
        (
            (&expected_irreps[9], &tc5),
            Character::new(&[(UnityRoot::new(0, 10), 1), (UnityRoot::new(5, 10), 1)]),
        ),
        (
            (&expected_irreps[10], &tc5),
            Character::new(&[(UnityRoot::new(5, 10), 1)]),
        ),
        (
            (&expected_irreps[12], &tc5),
            Character::new(&[
                (UnityRoot::new(5, 10), 1),
                (UnityRoot::new(3, 10), 1),
                (UnityRoot::new(7, 10), 1),
            ]),
        ),
        (
            (&expected_irreps[13], &tc5),
            Character::new(&[
                (UnityRoot::new(5, 10), 1),
                (UnityRoot::new(1, 10), 1),
                (UnityRoot::new(9, 10), 1),
            ]),
        ),
        (
            (&expected_irreps[16], &tc5),
            Character::new(&[(UnityRoot::new(0, 10), 1)]),
        ),
        (
            (&expected_irreps[18], &tc5),
            Character::new(&[(UnityRoot::new(0, 10), 1), (UnityRoot::new(5, 10), 1)]),
        ),
        (
            (&expected_irreps[11], &tc5),
            Character::new(&[(UnityRoot::new(5, 10), 1)]),
        ),
        (
            (&expected_irreps[14], &tc5),
            Character::new(&[
                (UnityRoot::new(5, 10), 1),
                (UnityRoot::new(3, 10), 1),
                (UnityRoot::new(7, 10), 1),
            ]),
        ),
        (
            (&expected_irreps[15], &tc5),
            Character::new(&[
                (UnityRoot::new(5, 10), 1),
                (UnityRoot::new(9, 10), 1),
                (UnityRoot::new(1, 10), 1),
            ]),
        ),
        (
            (&expected_irreps[17], &tc5),
            Character::new(&[(UnityRoot::new(0, 10), 1)]),
        ),
        (
            (&expected_irreps[19], &tc5),
            Character::new(&[(UnityRoot::new(0, 10), 1), (UnityRoot::new(5, 10), 1)]),
        ),
    ]);
    test_character_table_construction_magnetic(
        &mol,
        thresh,
        &expected_irreps,
        Some(expected_chars),
    );
}

#[test]
fn test_character_table_construction_spherical_ch4_td() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/ch4.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||E||").unwrap(),
        MullikenIrrepSymbol::new("||T|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||T|_(2)|").unwrap(),
    ];
    let c3 = ClassSymbol::<SymmetryOperation>::new("8||C3||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &c3),
            Character::new(&[(UnityRoot::new(0, 3), 1)]),
        ),
        (
            (&expected_irreps[1], &c3),
            Character::new(&[(UnityRoot::new(0, 3), 1)]),
        ),
        (
            (&expected_irreps[2], &c3),
            Character::new(&[(UnityRoot::new(1, 3), 1), (UnityRoot::new(2, 3), 1)]),
        ),
        (
            (&expected_irreps[3], &c3),
            Character::new(&[
                (UnityRoot::new(0, 3), 1),
                (UnityRoot::new(1, 3), 1),
                (UnityRoot::new(2, 3), 1),
            ]),
        ),
        (
            (&expected_irreps[4], &c3),
            Character::new(&[
                (UnityRoot::new(0, 3), 1),
                (UnityRoot::new(1, 3), 1),
                (UnityRoot::new(2, 3), 1),
            ]),
        ),
    ]);
    test_character_table_construction(&mol, thresh, &expected_irreps, Some(expected_chars));
}

#[test]
fn test_character_table_construction_spherical_ch4_grey_td() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/ch4.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||E||").unwrap(),
        MullikenIrrepSymbol::new("||T|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||T|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|A|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|A|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|E||").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|T|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|T|_(2)|").unwrap(),
    ];
    let tc3 = ClassSymbol::<SymmetryOperation>::new("8||θ·C3||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &tc3),
            Character::new(&[(UnityRoot::new(0, 3), 1)]),
        ),
        (
            (&expected_irreps[1], &tc3),
            Character::new(&[(UnityRoot::new(0, 3), 1)]),
        ),
        (
            (&expected_irreps[2], &tc3),
            Character::new(&[(UnityRoot::new(1, 3), 1), (UnityRoot::new(2, 3), 1)]),
        ),
        (
            (&expected_irreps[3], &tc3),
            Character::new(&[
                (UnityRoot::new(0, 3), 1),
                (UnityRoot::new(1, 3), 1),
                (UnityRoot::new(2, 3), 1),
            ]),
        ),
        (
            (&expected_irreps[4], &tc3),
            Character::new(&[
                (UnityRoot::new(0, 3), 1),
                (UnityRoot::new(1, 3), 1),
                (UnityRoot::new(2, 3), 1),
            ]),
        ),
        (
            (&expected_irreps[5], &tc3),
            Character::new(&[(UnityRoot::new(1, 3), 1), (UnityRoot::new(2, 3), 1)]),
        ),
        (
            (&expected_irreps[6], &tc3),
            Character::new(&[(UnityRoot::new(1, 3), 1), (UnityRoot::new(2, 3), 1)]),
        ),
        (
            (&expected_irreps[7], &tc3),
            Character::new(&[(UnityRoot::new(0, 3), 1)]),
        ),
        (
            (&expected_irreps[8], &tc3),
            Character::new(&[
                (UnityRoot::new(0, 3), 1),
                (UnityRoot::new(1, 3), 1),
                (UnityRoot::new(2, 3), 1),
            ]),
        ),
        (
            (&expected_irreps[9], &tc3),
            Character::new(&[
                (UnityRoot::new(0, 3), 1),
                (UnityRoot::new(1, 3), 1),
                (UnityRoot::new(2, 3), 1),
            ]),
        ),
    ]);
    test_character_table_construction_magnetic(
        &mol,
        thresh,
        &expected_irreps,
        Some(expected_chars),
    );
}

#[test]
fn test_character_table_construction_spherical_adamantane_td() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/adamantane.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||E||").unwrap(),
        MullikenIrrepSymbol::new("||T|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||T|_(2)|").unwrap(),
    ];
    let c3 = ClassSymbol::<SymmetryOperation>::new("8||C3||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &c3),
            Character::new(&[(UnityRoot::new(0, 3), 1)]),
        ),
        (
            (&expected_irreps[1], &c3),
            Character::new(&[(UnityRoot::new(0, 3), 1)]),
        ),
        (
            (&expected_irreps[2], &c3),
            Character::new(&[(UnityRoot::new(1, 3), 1), (UnityRoot::new(2, 3), 1)]),
        ),
        (
            (&expected_irreps[3], &c3),
            Character::new(&[
                (UnityRoot::new(0, 3), 1),
                (UnityRoot::new(1, 3), 1),
                (UnityRoot::new(2, 3), 1),
            ]),
        ),
        (
            (&expected_irreps[4], &c3),
            Character::new(&[
                (UnityRoot::new(0, 3), 1),
                (UnityRoot::new(1, 3), 1),
                (UnityRoot::new(2, 3), 1),
            ]),
        ),
    ]);
    test_character_table_construction(&mol, thresh, &expected_irreps, Some(expected_chars));
}

#[test]
fn test_character_table_construction_spherical_adamantane_grey_td() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/adamantane.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||E||").unwrap(),
        MullikenIrrepSymbol::new("||T|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||T|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|A|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|A|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|E||").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|T|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|T|_(2)|").unwrap(),
    ];
    let tc3 = ClassSymbol::<SymmetryOperation>::new("8||θ·C3||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &tc3),
            Character::new(&[(UnityRoot::new(0, 3), 1)]),
        ),
        (
            (&expected_irreps[1], &tc3),
            Character::new(&[(UnityRoot::new(0, 3), 1)]),
        ),
        (
            (&expected_irreps[2], &tc3),
            Character::new(&[(UnityRoot::new(1, 3), 1), (UnityRoot::new(2, 3), 1)]),
        ),
        (
            (&expected_irreps[3], &tc3),
            Character::new(&[
                (UnityRoot::new(0, 3), 1),
                (UnityRoot::new(1, 3), 1),
                (UnityRoot::new(2, 3), 1),
            ]),
        ),
        (
            (&expected_irreps[4], &tc3),
            Character::new(&[
                (UnityRoot::new(0, 3), 1),
                (UnityRoot::new(1, 3), 1),
                (UnityRoot::new(2, 3), 1),
            ]),
        ),
        (
            (&expected_irreps[5], &tc3),
            Character::new(&[(UnityRoot::new(1, 3), 1), (UnityRoot::new(2, 3), 1)]),
        ),
        (
            (&expected_irreps[6], &tc3),
            Character::new(&[(UnityRoot::new(1, 3), 1), (UnityRoot::new(2, 3), 1)]),
        ),
        (
            (&expected_irreps[7], &tc3),
            Character::new(&[(UnityRoot::new(0, 3), 1)]),
        ),
        (
            (&expected_irreps[8], &tc3),
            Character::new(&[
                (UnityRoot::new(0, 3), 1),
                (UnityRoot::new(1, 3), 1),
                (UnityRoot::new(2, 3), 1),
            ]),
        ),
        (
            (&expected_irreps[9], &tc3),
            Character::new(&[
                (UnityRoot::new(0, 3), 1),
                (UnityRoot::new(1, 3), 1),
                (UnityRoot::new(2, 3), 1),
            ]),
        ),
    ]);
    test_character_table_construction_magnetic(
        &mol,
        thresh,
        &expected_irreps,
        Some(expected_chars),
    );
}

#[test]
fn test_character_table_construction_spherical_c165_diamond_nanoparticle_td() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c165.xyz");
    let thresh = 1e-5;
    let mol = Molecule::from_xyz(&path, thresh);
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||E||").unwrap(),
        MullikenIrrepSymbol::new("||T|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||T|_(2)|").unwrap(),
    ];
    let c3 = ClassSymbol::<SymmetryOperation>::new("8||C3||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &c3),
            Character::new(&[(UnityRoot::new(0, 3), 1)]),
        ),
        (
            (&expected_irreps[1], &c3),
            Character::new(&[(UnityRoot::new(0, 3), 1)]),
        ),
        (
            (&expected_irreps[2], &c3),
            Character::new(&[(UnityRoot::new(1, 3), 1), (UnityRoot::new(2, 3), 1)]),
        ),
        (
            (&expected_irreps[3], &c3),
            Character::new(&[
                (UnityRoot::new(0, 3), 1),
                (UnityRoot::new(1, 3), 1),
                (UnityRoot::new(2, 3), 1),
            ]),
        ),
        (
            (&expected_irreps[4], &c3),
            Character::new(&[
                (UnityRoot::new(0, 3), 1),
                (UnityRoot::new(1, 3), 1),
                (UnityRoot::new(2, 3), 1),
            ]),
        ),
    ]);
    test_character_table_construction(&mol, thresh, &expected_irreps, Some(expected_chars));
}

#[test]
fn test_character_table_construction_spherical_c165_diamond_nanoparticle_grey_td() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c165.xyz");
    let thresh = 1e-5;
    let mol = Molecule::from_xyz(&path, thresh);
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||E||").unwrap(),
        MullikenIrrepSymbol::new("||T|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||T|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|A|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|A|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|E||").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|T|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|T|_(2)|").unwrap(),
    ];
    let tc3 = ClassSymbol::<SymmetryOperation>::new("8||θ·C3||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &tc3),
            Character::new(&[(UnityRoot::new(0, 3), 1)]),
        ),
        (
            (&expected_irreps[1], &tc3),
            Character::new(&[(UnityRoot::new(0, 3), 1)]),
        ),
        (
            (&expected_irreps[2], &tc3),
            Character::new(&[(UnityRoot::new(1, 3), 1), (UnityRoot::new(2, 3), 1)]),
        ),
        (
            (&expected_irreps[3], &tc3),
            Character::new(&[
                (UnityRoot::new(0, 3), 1),
                (UnityRoot::new(1, 3), 1),
                (UnityRoot::new(2, 3), 1),
            ]),
        ),
        (
            (&expected_irreps[4], &tc3),
            Character::new(&[
                (UnityRoot::new(0, 3), 1),
                (UnityRoot::new(1, 3), 1),
                (UnityRoot::new(2, 3), 1),
            ]),
        ),
        (
            (&expected_irreps[5], &tc3),
            Character::new(&[(UnityRoot::new(1, 3), 1), (UnityRoot::new(2, 3), 1)]),
        ),
        (
            (&expected_irreps[6], &tc3),
            Character::new(&[(UnityRoot::new(1, 3), 1), (UnityRoot::new(2, 3), 1)]),
        ),
        (
            (&expected_irreps[7], &tc3),
            Character::new(&[(UnityRoot::new(0, 3), 1)]),
        ),
        (
            (&expected_irreps[8], &tc3),
            Character::new(&[
                (UnityRoot::new(0, 3), 1),
                (UnityRoot::new(1, 3), 1),
                (UnityRoot::new(2, 3), 1),
            ]),
        ),
        (
            (&expected_irreps[9], &tc3),
            Character::new(&[
                (UnityRoot::new(0, 3), 1),
                (UnityRoot::new(1, 3), 1),
                (UnityRoot::new(2, 3), 1),
            ]),
        ),
    ]);
    test_character_table_construction_magnetic(
        &mol,
        thresh,
        &expected_irreps,
        Some(expected_chars),
    );
}

#[test]
fn test_character_table_construction_spherical_vh2o6_th() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/vh2o6.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("||Γ|_(1g)|").unwrap(),
        MullikenIrrepSymbol::new("||Γ|_(2g)|").unwrap(),
        MullikenIrrepSymbol::new("||T|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(u)|").unwrap(),
        MullikenIrrepSymbol::new("||Γ|_(1u)|").unwrap(),
        MullikenIrrepSymbol::new("||Γ|_(2u)|").unwrap(),
        MullikenIrrepSymbol::new("||T|_(u)|").unwrap(),
    ];
    let c3 = ClassSymbol::<SymmetryOperation>::new("4||C3||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &c3),
            Character::new(&[(UnityRoot::new(0, 3), 1)]),
        ),
        (
            (&expected_irreps[1], &c3),
            Character::new(&[(UnityRoot::new(1, 3), 1)]),
        ),
        (
            (&expected_irreps[2], &c3),
            Character::new(&[(UnityRoot::new(2, 3), 1)]),
        ),
        (
            (&expected_irreps[3], &c3),
            Character::new(&[
                (UnityRoot::new(0, 3), 1),
                (UnityRoot::new(1, 3), 1),
                (UnityRoot::new(2, 3), 1),
            ]),
        ),
        (
            (&expected_irreps[4], &c3),
            Character::new(&[(UnityRoot::new(0, 3), 1)]),
        ),
        (
            (&expected_irreps[5], &c3),
            Character::new(&[(UnityRoot::new(1, 3), 1)]),
        ),
        (
            (&expected_irreps[6], &c3),
            Character::new(&[(UnityRoot::new(2, 3), 1)]),
        ),
        (
            (&expected_irreps[7], &c3),
            Character::new(&[
                (UnityRoot::new(0, 3), 1),
                (UnityRoot::new(1, 3), 1),
                (UnityRoot::new(2, 3), 1),
            ]),
        ),
    ]);
    test_character_table_construction(&mol, thresh, &expected_irreps, Some(expected_chars));
}

#[test]
fn test_character_table_construction_spherical_vh2o6_grey_th() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/vh2o6.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("||Γ|_(1g)|").unwrap(),
        MullikenIrrepSymbol::new("||Γ|_(2g)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(u)|").unwrap(),
        MullikenIrrepSymbol::new("||Γ|_(1u)|").unwrap(),
        MullikenIrrepSymbol::new("||Γ|_(2u)|").unwrap(),
        MullikenIrrepSymbol::new("||T|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("||T|_(u)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|A|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|Γ|_(1g)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|Γ|_(2g)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|A|_(u)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|Γ|_(1u)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|Γ|_(2u)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|T|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|T|_(u)|").unwrap(),
    ];
    let tc3 = ClassSymbol::<SymmetryOperation>::new("4||θ·C3||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &tc3),
            Character::new(&[(UnityRoot::new(0, 3), 1)]),
        ),
        (
            (&expected_irreps[1], &tc3),
            Character::new(&[(UnityRoot::new(1, 3), 1)]),
        ),
        (
            (&expected_irreps[2], &tc3),
            Character::new(&[(UnityRoot::new(2, 3), 1)]),
        ),
        (
            (&expected_irreps[6], &tc3),
            Character::new(&[
                (UnityRoot::new(0, 3), 1),
                (UnityRoot::new(1, 3), 1),
                (UnityRoot::new(2, 3), 1),
            ]),
        ),
        (
            (&expected_irreps[3], &tc3),
            Character::new(&[(UnityRoot::new(0, 3), 1)]),
        ),
        (
            (&expected_irreps[4], &tc3),
            Character::new(&[(UnityRoot::new(1, 3), 1)]),
        ),
        (
            (&expected_irreps[5], &tc3),
            Character::new(&[(UnityRoot::new(2, 3), 1)]),
        ),
        (
            (&expected_irreps[7], &tc3),
            Character::new(&[
                (UnityRoot::new(0, 3), 1),
                (UnityRoot::new(1, 3), 1),
                (UnityRoot::new(2, 3), 1),
            ]),
        ),
        (
            (&expected_irreps[8], &tc3),
            Character::new(&[(UnityRoot::new(3, 6), 1)]),
        ),
        (
            (&expected_irreps[9], &tc3),
            Character::new(&[(UnityRoot::new(5, 6), 1)]),
        ),
        (
            (&expected_irreps[10], &tc3),
            Character::new(&[(UnityRoot::new(1, 6), 1)]),
        ),
        (
            (&expected_irreps[12], &tc3),
            Character::new(&[
                (UnityRoot::new(0, 3), 1),
                (UnityRoot::new(1, 3), 1),
                (UnityRoot::new(2, 3), 1),
            ]),
        ),
        (
            (&expected_irreps[11], &tc3),
            Character::new(&[(UnityRoot::new(3, 6), 1)]),
        ),
        (
            (&expected_irreps[12], &tc3),
            Character::new(&[(UnityRoot::new(5, 6), 1)]),
        ),
        (
            (&expected_irreps[13], &tc3),
            Character::new(&[(UnityRoot::new(1, 6), 1)]),
        ),
        (
            (&expected_irreps[15], &tc3),
            Character::new(&[
                (UnityRoot::new(0, 3), 1),
                (UnityRoot::new(1, 3), 1),
                (UnityRoot::new(2, 3), 1),
            ]),
        ),
    ]);
    test_character_table_construction_magnetic(
        &mol,
        thresh,
        &expected_irreps,
        Some(expected_chars),
    );
}

#[test]
fn test_character_table_construction_spherical_vf6_oh() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/vf6.xyz");
    let thresh = 1e-12;
    let mol = Molecule::from_xyz(&path, thresh);
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(1g)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(2g)|").unwrap(),
        MullikenIrrepSymbol::new("||E|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("||T|_(1g)|").unwrap(),
        MullikenIrrepSymbol::new("||T|_(2g)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(1u)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(2u)|").unwrap(),
        MullikenIrrepSymbol::new("||E|_(u)|").unwrap(),
        MullikenIrrepSymbol::new("||T|_(1u)|").unwrap(),
        MullikenIrrepSymbol::new("||T|_(2u)|").unwrap(),
    ];
    let c3 = ClassSymbol::<SymmetryOperation>::new("8||C3||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &c3),
            Character::new(&[(UnityRoot::new(0, 3), 1)]),
        ),
        (
            (&expected_irreps[1], &c3),
            Character::new(&[(UnityRoot::new(0, 3), 1)]),
        ),
        (
            (&expected_irreps[2], &c3),
            Character::new(&[(UnityRoot::new(1, 3), 1), (UnityRoot::new(2, 3), 1)]),
        ),
        (
            (&expected_irreps[3], &c3),
            Character::new(&[
                (UnityRoot::new(0, 3), 1),
                (UnityRoot::new(1, 3), 1),
                (UnityRoot::new(2, 3), 1),
            ]),
        ),
        (
            (&expected_irreps[4], &c3),
            Character::new(&[
                (UnityRoot::new(0, 3), 1),
                (UnityRoot::new(1, 3), 1),
                (UnityRoot::new(2, 3), 1),
            ]),
        ),
        (
            (&expected_irreps[5], &c3),
            Character::new(&[(UnityRoot::new(0, 3), 1)]),
        ),
        (
            (&expected_irreps[6], &c3),
            Character::new(&[(UnityRoot::new(0, 3), 1)]),
        ),
        (
            (&expected_irreps[7], &c3),
            Character::new(&[(UnityRoot::new(1, 3), 1), (UnityRoot::new(2, 3), 1)]),
        ),
        (
            (&expected_irreps[8], &c3),
            Character::new(&[
                (UnityRoot::new(0, 3), 1),
                (UnityRoot::new(1, 3), 1),
                (UnityRoot::new(2, 3), 1),
            ]),
        ),
        (
            (&expected_irreps[9], &c3),
            Character::new(&[
                (UnityRoot::new(0, 3), 1),
                (UnityRoot::new(1, 3), 1),
                (UnityRoot::new(2, 3), 1),
            ]),
        ),
    ]);
    test_character_table_construction(&mol, thresh, &expected_irreps, Some(expected_chars));
}

#[test]
fn test_character_table_construction_spherical_vf6_grey_oh() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/vf6.xyz");
    let thresh = 1e-12;
    let mol = Molecule::from_xyz(&path, thresh);
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(1g)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(2g)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(1u)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(2u)|").unwrap(),
        MullikenIrrepSymbol::new("||E|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("||E|_(u)|").unwrap(),
        MullikenIrrepSymbol::new("||T|_(1g)|").unwrap(),
        MullikenIrrepSymbol::new("||T|_(2g)|").unwrap(),
        MullikenIrrepSymbol::new("||T|_(1u)|").unwrap(),
        MullikenIrrepSymbol::new("||T|_(2u)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|A|_(1g)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|A|_(2g)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|A|_(1u)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|A|_(2u)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|E|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|E|_(u)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|T|_(1g)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|T|_(2g)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|T|_(1u)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|T|_(2u)|").unwrap(),
    ];
    let tc3 = ClassSymbol::<SymmetryOperation>::new("8||θ·C3||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &tc3),
            Character::new(&[(UnityRoot::new(0, 3), 1)]),
        ),
        (
            (&expected_irreps[1], &tc3),
            Character::new(&[(UnityRoot::new(0, 3), 1)]),
        ),
        (
            (&expected_irreps[4], &tc3),
            Character::new(&[(UnityRoot::new(1, 3), 1), (UnityRoot::new(2, 3), 1)]),
        ),
        (
            (&expected_irreps[6], &tc3),
            Character::new(&[
                (UnityRoot::new(0, 3), 1),
                (UnityRoot::new(1, 3), 1),
                (UnityRoot::new(2, 3), 1),
            ]),
        ),
        (
            (&expected_irreps[7], &tc3),
            Character::new(&[
                (UnityRoot::new(0, 3), 1),
                (UnityRoot::new(1, 3), 1),
                (UnityRoot::new(2, 3), 1),
            ]),
        ),
        (
            (&expected_irreps[2], &tc3),
            Character::new(&[(UnityRoot::new(0, 3), 1)]),
        ),
        (
            (&expected_irreps[3], &tc3),
            Character::new(&[(UnityRoot::new(0, 3), 1)]),
        ),
        (
            (&expected_irreps[5], &tc3),
            Character::new(&[(UnityRoot::new(1, 3), 1), (UnityRoot::new(2, 3), 1)]),
        ),
        (
            (&expected_irreps[8], &tc3),
            Character::new(&[
                (UnityRoot::new(0, 3), 1),
                (UnityRoot::new(1, 3), 1),
                (UnityRoot::new(2, 3), 1),
            ]),
        ),
        (
            (&expected_irreps[9], &tc3),
            Character::new(&[
                (UnityRoot::new(0, 3), 1),
                (UnityRoot::new(1, 3), 1),
                (UnityRoot::new(2, 3), 1),
            ]),
        ),
        (
            (&expected_irreps[10], &tc3),
            Character::new(&[(UnityRoot::new(1, 3), 1), (UnityRoot::new(2, 3), 1)]),
        ),
        (
            (&expected_irreps[11], &tc3),
            Character::new(&[(UnityRoot::new(1, 3), 1), (UnityRoot::new(2, 3), 1)]),
        ),
        (
            (&expected_irreps[14], &tc3),
            Character::new(&[(UnityRoot::new(0, 3), 1)]),
        ),
        (
            (&expected_irreps[16], &tc3),
            Character::new(&[
                (UnityRoot::new(0, 3), 1),
                (UnityRoot::new(1, 3), 1),
                (UnityRoot::new(2, 3), 1),
            ]),
        ),
        (
            (&expected_irreps[17], &tc3),
            Character::new(&[
                (UnityRoot::new(0, 3), 1),
                (UnityRoot::new(1, 3), 1),
                (UnityRoot::new(2, 3), 1),
            ]),
        ),
        (
            (&expected_irreps[12], &tc3),
            Character::new(&[(UnityRoot::new(1, 3), 1), (UnityRoot::new(2, 3), 1)]),
        ),
        (
            (&expected_irreps[13], &tc3),
            Character::new(&[(UnityRoot::new(1, 3), 1), (UnityRoot::new(2, 3), 1)]),
        ),
        (
            (&expected_irreps[15], &tc3),
            Character::new(&[(UnityRoot::new(0, 3), 1)]),
        ),
        (
            (&expected_irreps[18], &tc3),
            Character::new(&[
                (UnityRoot::new(0, 3), 1),
                (UnityRoot::new(1, 3), 1),
                (UnityRoot::new(2, 3), 1),
            ]),
        ),
        (
            (&expected_irreps[19], &tc3),
            Character::new(&[
                (UnityRoot::new(0, 3), 1),
                (UnityRoot::new(1, 3), 1),
                (UnityRoot::new(2, 3), 1),
            ]),
        ),
    ]);
    test_character_table_construction_magnetic(
        &mol,
        thresh,
        &expected_irreps,
        Some(expected_chars),
    );
}

/*****
Linear
*****/
#[test]
fn test_character_table_construction_linear_atom_magnetic_field_cinfh() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/th.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(1.0, 2.0, -1.0)));
    for n in 2usize..=20usize {
        let m = if n % 2 == 0 {
            // Cnh for n even
            n
        } else {
            // C(2n)h for n odd
            2 * n
        };
        let mut expected_irreps = vec![MullikenIrrepSymbol::new("||A|_(g)|").unwrap()];
        expected_irreps.extend(
            (1..m.div_euclid(2))
                .map(|k| MullikenIrrepSymbol::new(&format!("||Γ|_({}g)|", k)).unwrap()),
        );
        expected_irreps.push(MullikenIrrepSymbol::new("||B|_(g)|").unwrap());
        expected_irreps.extend(
            (m.div_euclid(2)..(m - 1))
                .map(|k| MullikenIrrepSymbol::new(&format!("||Γ|_({}g)|", k)).unwrap()),
        );
        expected_irreps.push(MullikenIrrepSymbol::new("||A|_(u)|").unwrap());
        expected_irreps.extend(
            (1..m.div_euclid(2))
                .map(|k| MullikenIrrepSymbol::new(&format!("||Γ|_({}u)|", k)).unwrap()),
        );
        expected_irreps.push(MullikenIrrepSymbol::new("||B|_(u)|").unwrap());
        expected_irreps.extend(
            (m.div_euclid(2)..(m - 1))
                .map(|k| MullikenIrrepSymbol::new(&format!("||Γ|_({}u)|", k)).unwrap()),
        );
        test_character_table_construction_from_infinite_group(
            &mol,
            n as u32,
            thresh,
            &expected_irreps,
            None,
        );
    }
}

#[test]
fn test_character_table_construction_linear_atom_magnetic_field_bw_dinfh_cinfh() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/th.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(1.0, 2.0, -1.0)));
    for n in 2usize..=20usize {
        let m = if n % 2 == 0 {
            // Dnh for n even
            n
        } else {
            // D(2n)h for n odd
            2 * n
        };
        let mut expected_irreps = vec![
            MullikenIrrepSymbol::new("||A|_(1g)|").unwrap(),
            MullikenIrrepSymbol::new("||A|_(2g)|").unwrap(),
            MullikenIrrepSymbol::new("||B|_(1g)|").unwrap(),
            MullikenIrrepSymbol::new("||B|_(2g)|").unwrap(),
        ];
        if m.div_euclid(2) != 2 {
            expected_irreps.extend(
                (1..m.div_euclid(2))
                    .map(|k| MullikenIrrepSymbol::new(&format!("||E|_({}g)|", k)).unwrap()),
            );
        } else {
            expected_irreps.push(MullikenIrrepSymbol::new("||E|_(g)|").unwrap());
        }
        expected_irreps.extend(vec![
            MullikenIrrepSymbol::new("||A|_(1u)|").unwrap(),
            MullikenIrrepSymbol::new("||A|_(2u)|").unwrap(),
            MullikenIrrepSymbol::new("||B|_(1u)|").unwrap(),
            MullikenIrrepSymbol::new("||B|_(2u)|").unwrap(),
        ]);
        if m.div_euclid(2) != 2 {
            expected_irreps.extend(
                (1..m.div_euclid(2))
                    .map(|k| MullikenIrrepSymbol::new(&format!("||E|_({}u)|", k)).unwrap()),
            );
        } else {
            expected_irreps.push(MullikenIrrepSymbol::new("||E|_(u)|").unwrap());
        }
        test_character_table_construction_from_infinite_magnetic_group(
            &mol,
            n as u32,
            thresh,
            &expected_irreps,
            None,
        );
    }
}

#[test]
fn test_character_table_construction_linear_atom_electric_field_cinfv() {
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
        let expected_irreps = if n % 2 == 0 {
            let mut irreps = vec![
                MullikenIrrepSymbol::new("||A|_(1)|").unwrap(),
                MullikenIrrepSymbol::new("||A|_(2)|").unwrap(),
                MullikenIrrepSymbol::new("||B|_(1)|").unwrap(),
                MullikenIrrepSymbol::new("||B|_(2)|").unwrap(),
            ];
            if n > 4 {
                irreps.extend(
                    (1..n.div_euclid(2))
                        .map(|k| MullikenIrrepSymbol::new(&format!("||E|_({})|", k)).unwrap()),
                );
            } else {
                irreps.push(MullikenIrrepSymbol::new("||E||").unwrap());
            }
            irreps
        } else {
            let mut irreps = vec![
                MullikenIrrepSymbol::new("||A|_(1)|").unwrap(),
                MullikenIrrepSymbol::new("||A|_(2)|").unwrap(),
            ];
            if n > 3 {
                irreps.extend(
                    (1..=n.div_euclid(2))
                        .map(|k| MullikenIrrepSymbol::new(&format!("||E|_({})|", k)).unwrap()),
                );
            } else {
                irreps.push(MullikenIrrepSymbol::new("||E||").unwrap());
            };
            irreps
        };
        test_character_table_construction_from_infinite_group(
            &mol,
            n as u32,
            thresh,
            &expected_irreps,
            None,
        );
    }
}

// #[test]
// fn test_character_table_construction_linear_atom_electric_field_grey_cinfv() {
//     /* The expected number of classes is deduced from the irrep structures of
//      * the Cnv groups.
//      * When n is even, the irreps are A1, A2, B1, B2, Ek where k = 1, ..., n/2 - 1.
//      * When n is odd, the irreps are A1, A2, Ek where k = 1, ..., n//2.
//      */
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/th.xyz");
//     let thresh = 1e-7;
//     let mut mol = Molecule::from_xyz(&path, thresh);
//     mol.set_electric_field(Some(Vector3::new(-1.0, 3.0, -2.0)));
//     for n in 4usize..=20usize {
//         let expected_irreps = if n % 2 == 0 {
//             let mut irreps = vec![
//                 MullikenIrrepSymbol::new("||A|_(1)|").unwrap(),
//                 MullikenIrrepSymbol::new("||A|_(2)|").unwrap(),
//                 MullikenIrrepSymbol::new("||B|_(1)|").unwrap(),
//                 MullikenIrrepSymbol::new("||B|_(2)|").unwrap(),
//             ];
//             if n > 4 {
//                 irreps.extend(
//                     (1..n.div_euclid(2))
//                         .map(|k| MullikenIrrepSymbol::new(&format!("||E|_({})|", k)).unwrap()),
//                 );
//             } else {
//                 irreps.push(MullikenIrrepSymbol::new("||E||").unwrap());
//             }
//             let m_irreps = irreps.iter().map(|irrep| {
//                 MullikenIrrepSymbol::new(&format!(
//                         "|^(m)|{}|_({})|",
//                         irrep.main(),
//                         irrep.postsub()
//                 ))
//                     .unwrap()
//             }).collect_vec();
//             irreps.extend(m_irreps);
//             irreps
//         } else {
//             let mut irreps = vec![
//                 MullikenIrrepSymbol::new("||A|_(1)|").unwrap(),
//                 MullikenIrrepSymbol::new("||A|_(2)|").unwrap(),
//             ];
//             if n > 3 {
//                 irreps.extend(
//                     (1..=n.div_euclid(2))
//                         .map(|k| MullikenIrrepSymbol::new(&format!("||E|_({})|", k)).unwrap()),
//                 );
//             } else {
//                 irreps.push(MullikenIrrepSymbol::new("||E||").unwrap());
//             };
//             let m_irreps = irreps.iter().map(|irrep| {
//                 MullikenIrrepSymbol::new(&format!(
//                         "|^(m)|{}|_({})|",
//                         irrep.main(),
//                         irrep.postsub()
//                 ))
//                     .unwrap()
//             }).collect_vec();
//             irreps.extend(m_irreps);
//             irreps
//         };
//         for irrep in &expected_irreps {
//             println!("Ir: {}", irrep);
//         };
//         test_character_table_construction_from_infinite_magnetic_group(
//             &mol,
//             n as u32,
//             thresh,
//             &expected_irreps,
//             None,
//         );
//     }
// }

#[test]
fn test_character_table_construction_linear_c2h2_dinfh() {
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
        let m = if n % 2 == 0 {
            // Dnh for n even
            n
        } else {
            // D(2n)h for n odd
            2 * n
        };
        let mut expected_irreps = vec![];
        for parity in ["g", "u"] {
            let mut irreps = vec![
                MullikenIrrepSymbol::new(&format!("||A|_(1{parity})|")).unwrap(),
                MullikenIrrepSymbol::new(&format!("||A|_(2{parity})|")).unwrap(),
                MullikenIrrepSymbol::new(&format!("||B|_(1{parity})|")).unwrap(),
                MullikenIrrepSymbol::new(&format!("||B|_(2{parity})|")).unwrap(),
            ];
            if m > 4 {
                irreps.extend(
                    (1..m.div_euclid(2)).map(|k| {
                        MullikenIrrepSymbol::new(&format!("||E|_({k}{parity})|")).unwrap()
                    }),
                );
            } else {
                irreps.push(MullikenIrrepSymbol::new(&format!("||E|_({parity})|")).unwrap());
            }
            expected_irreps.extend(irreps)
        }
        test_character_table_construction_from_infinite_group(
            &mol,
            n as u32,
            thresh,
            &expected_irreps,
            None,
        );
    }
}

#[test]
fn test_character_table_construction_linear_c2h2_magnetic_field_cinfh() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c2h2.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);
    // Parallel field
    mol.set_magnetic_field(Some(Vector3::new(1.0, 1.0, 1.0)));
    for n in 2usize..=10usize {
        let m = if n % 2 == 0 {
            // Cnh for n even
            n
        } else {
            // C(2n)h for n odd
            2 * n
        };
        let mut expected_irreps = vec![MullikenIrrepSymbol::new("||A|_(g)|").unwrap()];
        expected_irreps.extend(
            (1..m.div_euclid(2))
                .map(|k| MullikenIrrepSymbol::new(&format!("||Γ|_({}g)|", k)).unwrap()),
        );
        expected_irreps.push(MullikenIrrepSymbol::new("||B|_(g)|").unwrap());
        expected_irreps.extend(
            (m.div_euclid(2)..(m - 1))
                .map(|k| MullikenIrrepSymbol::new(&format!("||Γ|_({}g)|", k)).unwrap()),
        );
        expected_irreps.push(MullikenIrrepSymbol::new("||A|_(u)|").unwrap());
        expected_irreps.extend(
            (1..m.div_euclid(2))
                .map(|k| MullikenIrrepSymbol::new(&format!("||Γ|_({}u)|", k)).unwrap()),
        );
        expected_irreps.push(MullikenIrrepSymbol::new("||B|_(u)|").unwrap());
        expected_irreps.extend(
            (m.div_euclid(2)..(m - 1))
                .map(|k| MullikenIrrepSymbol::new(&format!("||Γ|_({}u)|", k)).unwrap()),
        );
        test_character_table_construction_from_infinite_group(
            &mol,
            n as u32,
            thresh,
            &expected_irreps,
            None,
        );
    }
}

#[test]
fn test_character_table_construction_linear_c2h2_electric_field_cinfv() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c2h2.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);

    // Parallel field
    mol.set_electric_field(Some(Vector3::new(1.0, 1.0, 1.0)));
    for n in 3usize..=10usize {
        let expected_irreps = if n % 2 == 0 {
            let mut irreps = vec![
                MullikenIrrepSymbol::new("||A|_(1)|").unwrap(),
                MullikenIrrepSymbol::new("||A|_(2)|").unwrap(),
                MullikenIrrepSymbol::new("||B|_(1)|").unwrap(),
                MullikenIrrepSymbol::new("||B|_(2)|").unwrap(),
            ];
            if n > 4 {
                irreps.extend(
                    (1..n.div_euclid(2))
                        .map(|k| MullikenIrrepSymbol::new(&format!("||E|_({})|", k)).unwrap()),
                );
            } else {
                irreps.push(MullikenIrrepSymbol::new("||E||").unwrap());
            }
            irreps
        } else {
            let mut irreps = vec![
                MullikenIrrepSymbol::new("||A|_(1)|").unwrap(),
                MullikenIrrepSymbol::new("||A|_(2)|").unwrap(),
            ];
            if n > 3 {
                irreps.extend(
                    (1..=n.div_euclid(2))
                        .map(|k| MullikenIrrepSymbol::new(&format!("||E|_({})|", k)).unwrap()),
                );
            } else {
                irreps.push(MullikenIrrepSymbol::new("||E||").unwrap());
            };
            irreps
        };
        test_character_table_construction_from_infinite_group(
            &mol,
            n as u32,
            thresh,
            &expected_irreps,
            None,
        );
    }
}

#[test]
fn test_character_table_construction_linear_n3_cinfv() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/n3.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    for n in 3usize..=10usize {
        let expected_irreps = if n % 2 == 0 {
            let mut irreps = vec![
                MullikenIrrepSymbol::new("||A|_(1)|").unwrap(),
                MullikenIrrepSymbol::new("||A|_(2)|").unwrap(),
                MullikenIrrepSymbol::new("||B|_(1)|").unwrap(),
                MullikenIrrepSymbol::new("||B|_(2)|").unwrap(),
            ];
            if n > 4 {
                irreps.extend(
                    (1..n.div_euclid(2))
                        .map(|k| MullikenIrrepSymbol::new(&format!("||E|_({})|", k)).unwrap()),
                );
            } else {
                irreps.push(MullikenIrrepSymbol::new("||E||").unwrap());
            }
            irreps
        } else {
            let mut irreps = vec![
                MullikenIrrepSymbol::new("||A|_(1)|").unwrap(),
                MullikenIrrepSymbol::new("||A|_(2)|").unwrap(),
            ];
            if n > 3 {
                irreps.extend(
                    (1..=n.div_euclid(2))
                        .map(|k| MullikenIrrepSymbol::new(&format!("||E|_({})|", k)).unwrap()),
                );
            } else {
                irreps.push(MullikenIrrepSymbol::new("||E||").unwrap());
            };
            irreps
        };
        test_character_table_construction_from_infinite_group(
            &mol,
            n as u32,
            thresh,
            &expected_irreps,
            None,
        );
    }
}

#[test]
fn test_character_table_construction_linear_n3_magnetic_field_cinf() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/n3.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);

    // Parallel field
    mol.set_magnetic_field(Some(Vector3::new(1.0, 1.0, 1.0)));
    for n in 2usize..=10usize {
        let expected_irreps = if n % 2 == 0 {
            let mut irreps = vec![MullikenIrrepSymbol::new("||A||").unwrap()];
            irreps.extend(
                (1..(n / 2)).map(|i| MullikenIrrepSymbol::new(&format!("||Γ|_({})|", i)).unwrap()),
            );
            irreps.push(MullikenIrrepSymbol::new("||B||").unwrap());
            irreps.extend(
                ((n / 2)..(n - 1))
                    .map(|i| MullikenIrrepSymbol::new(&format!("||Γ|_({})|", i)).unwrap()),
            );
            irreps
        } else {
            let mut irreps = vec![MullikenIrrepSymbol::new("||A||").unwrap()];
            irreps.extend(
                (1..n).map(|i| MullikenIrrepSymbol::new(&format!("||Γ|_({})|", i)).unwrap()),
            );
            irreps
        };
        test_character_table_construction_from_infinite_group(
            &mol,
            n as u32,
            thresh,
            &expected_irreps,
            None,
        );
    }
}

#[test]
fn test_character_table_construction_linear_n3_electric_field_cinfv() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/n3.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);

    // Parallel field
    mol.set_electric_field(Some(Vector3::new(1.0, 1.0, 1.0)));
    for n in 3usize..=10usize {
        let expected_irreps = if n % 2 == 0 {
            let mut irreps = vec![
                MullikenIrrepSymbol::new("||A|_(1)|").unwrap(),
                MullikenIrrepSymbol::new("||A|_(2)|").unwrap(),
                MullikenIrrepSymbol::new("||B|_(1)|").unwrap(),
                MullikenIrrepSymbol::new("||B|_(2)|").unwrap(),
            ];
            if n > 4 {
                irreps.extend(
                    (1..n.div_euclid(2))
                        .map(|k| MullikenIrrepSymbol::new(&format!("||E|_({})|", k)).unwrap()),
                );
            } else {
                irreps.push(MullikenIrrepSymbol::new("||E||").unwrap());
            }
            irreps
        } else {
            let mut irreps = vec![
                MullikenIrrepSymbol::new("||A|_(1)|").unwrap(),
                MullikenIrrepSymbol::new("||A|_(2)|").unwrap(),
            ];
            if n > 3 {
                irreps.extend(
                    (1..=n.div_euclid(2))
                        .map(|k| MullikenIrrepSymbol::new(&format!("||E|_({})|", k)).unwrap()),
                );
            } else {
                irreps.push(MullikenIrrepSymbol::new("||E||").unwrap());
            };
            irreps
        };
        test_character_table_construction_from_infinite_group(
            &mol,
            n as u32,
            thresh,
            &expected_irreps,
            None,
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
fn test_character_table_construction_symmetric_ch4_magnetic_field_c3() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/ch4.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(1.0, 1.0, 1.0)));
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A||").unwrap(),
        MullikenIrrepSymbol::new("||Γ|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||Γ|_(2)|").unwrap(),
    ];
    let c3 = ClassSymbol::<SymmetryOperation>::new("1||C3||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &c3),
            Character::new(&[(UnityRoot::new(0, 3), 1)]),
        ),
        (
            (&expected_irreps[1], &c3),
            Character::new(&[(UnityRoot::new(1, 3), 1)]),
        ),
        (
            (&expected_irreps[2], &c3),
            Character::new(&[(UnityRoot::new(2, 3), 1)]),
        ),
    ]);
    test_character_table_construction(&mol, thresh, &expected_irreps, Some(expected_chars));
}

#[test]
fn test_character_table_construction_symmetric_adamantane_magnetic_field_c3() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/adamantane.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.1, 0.1, 0.1)));
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A||").unwrap(),
        MullikenIrrepSymbol::new("||Γ|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||Γ|_(2)|").unwrap(),
    ];
    let c3 = ClassSymbol::<SymmetryOperation>::new("1||C3||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &c3),
            Character::new(&[(UnityRoot::new(0, 3), 1)]),
        ),
        (
            (&expected_irreps[1], &c3),
            Character::new(&[(UnityRoot::new(1, 3), 1)]),
        ),
        (
            (&expected_irreps[2], &c3),
            Character::new(&[(UnityRoot::new(2, 3), 1)]),
        ),
    ]);
    test_character_table_construction(&mol, thresh, &expected_irreps, Some(expected_chars));
}

#[test]
fn test_character_table_construction_symmetric_vh2o6_electric_field_c3() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/vh2o6.xyz");
    let thresh = 1e-12;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(Vector3::new(-0.2, -0.2, -0.2)));
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A||").unwrap(),
        MullikenIrrepSymbol::new("||Γ|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||Γ|_(2)|").unwrap(),
    ];
    let c3 = ClassSymbol::<SymmetryOperation>::new("1||C3||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &c3),
            Character::new(&[(UnityRoot::new(0, 3), 1)]),
        ),
        (
            (&expected_irreps[1], &c3),
            Character::new(&[(UnityRoot::new(1, 3), 1)]),
        ),
        (
            (&expected_irreps[2], &c3),
            Character::new(&[(UnityRoot::new(2, 3), 1)]),
        ),
    ]);
    test_character_table_construction(&mol, thresh, &expected_irreps, Some(expected_chars));
}

#[test]
fn test_character_table_construction_symmetric_65coronane_electric_field_c3() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/coronane65.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(Vector3::new(0.0, 0.0, -1.0)));
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A||").unwrap(),
        MullikenIrrepSymbol::new("||Γ|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||Γ|_(2)|").unwrap(),
    ];
    let c3 = ClassSymbol::<SymmetryOperation>::new("1||C3||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &c3),
            Character::new(&[(UnityRoot::new(0, 3), 1)]),
        ),
        (
            (&expected_irreps[1], &c3),
            Character::new(&[(UnityRoot::new(1, 3), 1)]),
        ),
        (
            (&expected_irreps[2], &c3),
            Character::new(&[(UnityRoot::new(2, 3), 1)]),
        ),
    ]);
    test_character_table_construction(&mol, thresh, &expected_irreps, Some(expected_chars));
}

#[test]
fn test_character_table_construction_symmetric_h8_twisted_magnetic_field_c4() {
    // env_logger::init();
    let thresh = 1e-7;
    let mut mol = template_molecules::gen_twisted_h8(0.1);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 0.1)));
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A||").unwrap(),
        MullikenIrrepSymbol::new("||Γ|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||B||").unwrap(),
        MullikenIrrepSymbol::new("||Γ|_(2)|").unwrap(),
    ];
    let c4 = ClassSymbol::<SymmetryOperation>::new("1||C4||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &c4),
            Character::new(&[(UnityRoot::new(0, 4), 1)]),
        ),
        (
            (&expected_irreps[1], &c4),
            Character::new(&[(UnityRoot::new(1, 4), 1)]),
        ),
        (
            (&expected_irreps[2], &c4),
            Character::new(&[(UnityRoot::new(2, 4), 1)]),
        ),
        (
            (&expected_irreps[3], &c4),
            Character::new(&[(UnityRoot::new(3, 4), 1)]),
        ),
    ]);
    test_character_table_construction(&mol, thresh, &expected_irreps, Some(expected_chars));
}

#[test]
fn test_character_table_construction_symmetric_h8_twisted_electric_field_c4() {
    let thresh = 1e-7;
    let mut mol = template_molecules::gen_twisted_h8(0.1);
    mol.set_electric_field(Some(Vector3::new(0.0, 0.0, -0.1)));
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A||").unwrap(),
        MullikenIrrepSymbol::new("||Γ|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||B||").unwrap(),
        MullikenIrrepSymbol::new("||Γ|_(2)|").unwrap(),
    ];
    let c4 = ClassSymbol::<SymmetryOperation>::new("1||C4||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &c4),
            Character::new(&[(UnityRoot::new(0, 4), 1)]),
        ),
        (
            (&expected_irreps[1], &c4),
            Character::new(&[(UnityRoot::new(1, 4), 1)]),
        ),
        (
            (&expected_irreps[2], &c4),
            Character::new(&[(UnityRoot::new(2, 4), 1)]),
        ),
        (
            (&expected_irreps[3], &c4),
            Character::new(&[(UnityRoot::new(3, 4), 1)]),
        ),
    ]);
    test_character_table_construction(&mol, thresh, &expected_irreps, Some(expected_chars));
}

#[test]
fn test_character_table_construction_symmetric_cpnico_magnetic_field_c5() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/cpnico.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, -0.2)));
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A||").unwrap(),
        MullikenIrrepSymbol::new("||Γ|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||Γ|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||Γ|_(3)|").unwrap(),
        MullikenIrrepSymbol::new("||Γ|_(4)|").unwrap(),
    ];
    let c5 = ClassSymbol::<SymmetryOperation>::new("1||C5||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &c5),
            Character::new(&[(UnityRoot::new(0, 5), 1)]),
        ),
        (
            (&expected_irreps[1], &c5),
            Character::new(&[(UnityRoot::new(1, 5), 1)]),
        ),
        (
            (&expected_irreps[2], &c5),
            Character::new(&[(UnityRoot::new(2, 5), 1)]),
        ),
        (
            (&expected_irreps[3], &c5),
            Character::new(&[(UnityRoot::new(3, 5), 1)]),
        ),
        (
            (&expected_irreps[4], &c5),
            Character::new(&[(UnityRoot::new(4, 5), 1)]),
        ),
    ]);
    test_character_table_construction(&mol, thresh, &expected_irreps, Some(expected_chars));
}

#[test]
fn test_character_table_construction_symmetric_b7_magnetic_field_c6() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/b7.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 0.1)));
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A||").unwrap(),
        MullikenIrrepSymbol::new("||Γ|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||Γ|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||B||").unwrap(),
        MullikenIrrepSymbol::new("||Γ|_(3)|").unwrap(),
        MullikenIrrepSymbol::new("||Γ|_(4)|").unwrap(),
    ];
    let c6 = ClassSymbol::<SymmetryOperation>::new("1||C6||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &c6),
            Character::new(&[(UnityRoot::new(0, 6), 1)]),
        ),
        (
            (&expected_irreps[1], &c6),
            Character::new(&[(UnityRoot::new(1, 6), 1)]),
        ),
        (
            (&expected_irreps[2], &c6),
            Character::new(&[(UnityRoot::new(2, 6), 1)]),
        ),
        (
            (&expected_irreps[3], &c6),
            Character::new(&[(UnityRoot::new(3, 6), 1)]),
        ),
        (
            (&expected_irreps[4], &c6),
            Character::new(&[(UnityRoot::new(4, 6), 1)]),
        ),
        (
            (&expected_irreps[5], &c6),
            Character::new(&[(UnityRoot::new(5, 6), 1)]),
        ),
    ]);
    test_character_table_construction(&mol, thresh, &expected_irreps, Some(expected_chars));
}

#[test]
fn test_character_table_construction_symmetric_arbitrary_half_sandwich_magnetic_field_cn() {
    let thresh = 1e-7;
    for n in 3..=32 {
        let mut mol = template_molecules::gen_arbitrary_half_sandwich(n);
        mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 0.1)));
        let expected_irreps = if n % 2 == 0 {
            let mut irreps = vec![MullikenIrrepSymbol::new("||A||").unwrap()];
            irreps.extend(
                (1..(n / 2)).map(|i| MullikenIrrepSymbol::new(&format!("||Γ|_({})|", i)).unwrap()),
            );
            irreps.push(MullikenIrrepSymbol::new("||B||").unwrap());
            irreps.extend(
                ((n / 2)..(n - 1))
                    .map(|i| MullikenIrrepSymbol::new(&format!("||Γ|_({})|", i)).unwrap()),
            );
            irreps
        } else {
            let mut irreps = vec![MullikenIrrepSymbol::new("||A||").unwrap()];
            irreps.extend(
                (1..n).map(|i| MullikenIrrepSymbol::new(&format!("||Γ|_({})|", i)).unwrap()),
            );
            irreps
        };
        let cn = ClassSymbol::<SymmetryOperation>::new(&format!("1||C{}||", n), None).unwrap();
        let expected_chars: HashMap<_, _> = expected_irreps
            .iter()
            .enumerate()
            .map(|(i, irrep)| {
                (
                    (irrep, &cn),
                    Character::new(&[(UnityRoot::new(i as u32, n), 1)]),
                )
            })
            .collect();
        test_character_table_construction(&mol, thresh, &expected_irreps, Some(expected_chars));
    }
}

/*
Cnv
*/
#[test]
fn test_character_table_construction_symmetric_nh3_c3v() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/nh3.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||E||").unwrap(),
    ];
    let c3 = ClassSymbol::<SymmetryOperation>::new("2||C3||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &c3),
            Character::new(&[(UnityRoot::new(0, 3), 1)]),
        ),
        (
            (&expected_irreps[1], &c3),
            Character::new(&[(UnityRoot::new(0, 3), 1)]),
        ),
        (
            (&expected_irreps[2], &c3),
            Character::new(&[(UnityRoot::new(1, 3), 1), (UnityRoot::new(2, 3), 1)]),
        ),
    ]);
    test_character_table_construction(&mol, thresh, &expected_irreps, Some(expected_chars));
}

#[test]
fn test_character_table_construction_symmetric_bf3_electric_field_c3v() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/bf3.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||E||").unwrap(),
    ];
    let c3 = ClassSymbol::<SymmetryOperation>::new("2||C3||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &c3),
            Character::new(&[(UnityRoot::new(0, 3), 1)]),
        ),
        (
            (&expected_irreps[1], &c3),
            Character::new(&[(UnityRoot::new(0, 3), 1)]),
        ),
        (
            (&expected_irreps[2], &c3),
            Character::new(&[(UnityRoot::new(1, 3), 1), (UnityRoot::new(2, 3), 1)]),
        ),
    ]);
    test_character_table_construction(&mol, thresh, &expected_irreps, Some(expected_chars));
}

#[test]
fn test_character_table_construction_symmetric_adamantane_electric_field_c3v() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/adamantane.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(Vector3::new(0.1, 0.1, 0.1)));
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||E||").unwrap(),
    ];
    let c3 = ClassSymbol::<SymmetryOperation>::new("2||C3||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &c3),
            Character::new(&[(UnityRoot::new(0, 3), 1)]),
        ),
        (
            (&expected_irreps[1], &c3),
            Character::new(&[(UnityRoot::new(0, 3), 1)]),
        ),
        (
            (&expected_irreps[2], &c3),
            Character::new(&[(UnityRoot::new(1, 3), 1), (UnityRoot::new(2, 3), 1)]),
        ),
    ]);
    test_character_table_construction(&mol, thresh, &expected_irreps, Some(expected_chars));
}

#[test]
fn test_character_table_construction_symmetric_ch4_electric_field_c3v() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/ch4.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(Vector3::new(1.0, 1.0, 1.0)));
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||E||").unwrap(),
    ];
    let c3 = ClassSymbol::<SymmetryOperation>::new("2||C3||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &c3),
            Character::new(&[(UnityRoot::new(0, 3), 1)]),
        ),
        (
            (&expected_irreps[1], &c3),
            Character::new(&[(UnityRoot::new(0, 3), 1)]),
        ),
        (
            (&expected_irreps[2], &c3),
            Character::new(&[(UnityRoot::new(1, 3), 1), (UnityRoot::new(2, 3), 1)]),
        ),
    ]);
    test_character_table_construction(&mol, thresh, &expected_irreps, Some(expected_chars));
}

#[test]
fn test_character_table_construction_symmetric_vf6_electric_field_c3v() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/vf6.xyz");
    let thresh = 1e-12;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(0.2 * Vector3::new(1.0, 1.0, 1.0)));
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||E||").unwrap(),
    ];
    let c3 = ClassSymbol::<SymmetryOperation>::new("2||C3||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &c3),
            Character::new(&[(UnityRoot::new(0, 3), 1)]),
        ),
        (
            (&expected_irreps[1], &c3),
            Character::new(&[(UnityRoot::new(0, 3), 1)]),
        ),
        (
            (&expected_irreps[2], &c3),
            Character::new(&[(UnityRoot::new(1, 3), 1), (UnityRoot::new(2, 3), 1)]),
        ),
    ]);
    test_character_table_construction(&mol, thresh, &expected_irreps, Some(expected_chars));
}

#[test]
fn test_character_table_construction_symmetric_sf5cl_c4v() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/sf5cl.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||E||").unwrap(),
    ];
    let c4 = ClassSymbol::<SymmetryOperation>::new("2||C4||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &c4),
            Character::new(&[(UnityRoot::new(0, 4), 1)]),
        ),
        (
            (&expected_irreps[1], &c4),
            Character::new(&[(UnityRoot::new(0, 4), 1)]),
        ),
        (
            (&expected_irreps[2], &c4),
            Character::new(&[(UnityRoot::new(2, 4), 1)]),
        ),
        (
            (&expected_irreps[3], &c4),
            Character::new(&[(UnityRoot::new(2, 4), 1)]),
        ),
        (
            (&expected_irreps[4], &c4),
            Character::new(&[(UnityRoot::new(1, 4), 1), (UnityRoot::new(3, 4), 1)]),
        ),
    ]);
    test_character_table_construction(&mol, thresh, &expected_irreps, Some(expected_chars));
}

#[test]
fn test_character_table_construction_symmetric_h8_electric_field_c4v() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/h8.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(0.2 * Vector3::new(0.0, 0.0, 1.0)));
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||E||").unwrap(),
    ];
    let c4 = ClassSymbol::<SymmetryOperation>::new("2||C4||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &c4),
            Character::new(&[(UnityRoot::new(0, 4), 1)]),
        ),
        (
            (&expected_irreps[1], &c4),
            Character::new(&[(UnityRoot::new(0, 4), 1)]),
        ),
        (
            (&expected_irreps[2], &c4),
            Character::new(&[(UnityRoot::new(2, 4), 1)]),
        ),
        (
            (&expected_irreps[3], &c4),
            Character::new(&[(UnityRoot::new(2, 4), 1)]),
        ),
        (
            (&expected_irreps[4], &c4),
            Character::new(&[(UnityRoot::new(1, 4), 1), (UnityRoot::new(3, 4), 1)]),
        ),
    ]);
    test_character_table_construction(&mol, thresh, &expected_irreps, Some(expected_chars));
}

#[test]
fn test_character_table_construction_symmetric_vf6_electric_field_c4v() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/vf6.xyz");
    let thresh = 1e-12;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(0.2 * Vector3::new(1.0, 0.0, 0.0)));
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||E||").unwrap(),
    ];
    let c4 = ClassSymbol::<SymmetryOperation>::new("2||C4||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &c4),
            Character::new(&[(UnityRoot::new(0, 4), 1)]),
        ),
        (
            (&expected_irreps[1], &c4),
            Character::new(&[(UnityRoot::new(0, 4), 1)]),
        ),
        (
            (&expected_irreps[2], &c4),
            Character::new(&[(UnityRoot::new(2, 4), 1)]),
        ),
        (
            (&expected_irreps[3], &c4),
            Character::new(&[(UnityRoot::new(2, 4), 1)]),
        ),
        (
            (&expected_irreps[4], &c4),
            Character::new(&[(UnityRoot::new(1, 4), 1), (UnityRoot::new(3, 4), 1)]),
        ),
    ]);
    test_character_table_construction(&mol, thresh, &expected_irreps, Some(expected_chars));
}

#[test]
fn test_character_table_construction_symmetric_antiprism_pb10_electric_field_c4v() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/pb10.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||E||").unwrap(),
    ];
    let c4 = ClassSymbol::<SymmetryOperation>::new("2||C4||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &c4),
            Character::new(&[(UnityRoot::new(0, 4), 1)]),
        ),
        (
            (&expected_irreps[1], &c4),
            Character::new(&[(UnityRoot::new(0, 4), 1)]),
        ),
        (
            (&expected_irreps[2], &c4),
            Character::new(&[(UnityRoot::new(2, 4), 1)]),
        ),
        (
            (&expected_irreps[3], &c4),
            Character::new(&[(UnityRoot::new(2, 4), 1)]),
        ),
        (
            (&expected_irreps[4], &c4),
            Character::new(&[(UnityRoot::new(1, 4), 1), (UnityRoot::new(3, 4), 1)]),
        ),
    ]);
    test_character_table_construction(&mol, thresh, &expected_irreps, Some(expected_chars));
}

#[test]
fn test_character_table_construction_symmetric_cpnico_c5v() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/cpnico.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||E|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||E|_(2)|").unwrap(),
    ];
    let c5 = ClassSymbol::<SymmetryOperation>::new("2||C5||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &c5),
            Character::new(&[(UnityRoot::new(0, 5), 1)]),
        ),
        (
            (&expected_irreps[1], &c5),
            Character::new(&[(UnityRoot::new(0, 5), 1)]),
        ),
        (
            (&expected_irreps[2], &c5),
            Character::new(&[(UnityRoot::new(1, 5), 1), (UnityRoot::new(4, 5), 1)]),
        ),
        (
            (&expected_irreps[3], &c5),
            Character::new(&[(UnityRoot::new(2, 5), 1), (UnityRoot::new(3, 5), 1)]),
        ),
    ]);
    test_character_table_construction(&mol, thresh, &expected_irreps, Some(expected_chars));
}

#[test]
fn test_character_table_construction_symmetric_staggered_ferrocene_electric_field_c5v() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/staggered_ferrocene.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||E|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||E|_(2)|").unwrap(),
    ];
    let c5 = ClassSymbol::<SymmetryOperation>::new("2||C5||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &c5),
            Character::new(&[(UnityRoot::new(0, 5), 1)]),
        ),
        (
            (&expected_irreps[1], &c5),
            Character::new(&[(UnityRoot::new(0, 5), 1)]),
        ),
        (
            (&expected_irreps[2], &c5),
            Character::new(&[(UnityRoot::new(1, 5), 1), (UnityRoot::new(4, 5), 1)]),
        ),
        (
            (&expected_irreps[3], &c5),
            Character::new(&[(UnityRoot::new(2, 5), 1), (UnityRoot::new(3, 5), 1)]),
        ),
    ]);
    test_character_table_construction(&mol, thresh, &expected_irreps, Some(expected_chars));
}

#[test]
fn test_character_table_construction_symmetric_c60_electric_field_c5v() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c60.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||E|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||E|_(2)|").unwrap(),
    ];
    let c5 = ClassSymbol::<SymmetryOperation>::new("2||C5||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &c5),
            Character::new(&[(UnityRoot::new(0, 5), 1)]),
        ),
        (
            (&expected_irreps[1], &c5),
            Character::new(&[(UnityRoot::new(0, 5), 1)]),
        ),
        (
            (&expected_irreps[2], &c5),
            Character::new(&[(UnityRoot::new(1, 5), 1), (UnityRoot::new(4, 5), 1)]),
        ),
        (
            (&expected_irreps[3], &c5),
            Character::new(&[(UnityRoot::new(2, 5), 1), (UnityRoot::new(3, 5), 1)]),
        ),
    ]);
    test_character_table_construction(&mol, thresh, &expected_irreps, Some(expected_chars));
}

#[test]
fn test_character_table_construction_symmetric_b7_c6v() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/b7.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||E|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||E|_(2)|").unwrap(),
    ];
    let c6 = ClassSymbol::<SymmetryOperation>::new("2||C6||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &c6),
            Character::new(&[(UnityRoot::new(0, 6), 1)]),
        ),
        (
            (&expected_irreps[1], &c6),
            Character::new(&[(UnityRoot::new(0, 6), 1)]),
        ),
        (
            (&expected_irreps[2], &c6),
            Character::new(&[(UnityRoot::new(3, 6), 1)]),
        ),
        (
            (&expected_irreps[3], &c6),
            Character::new(&[(UnityRoot::new(3, 6), 1)]),
        ),
        (
            (&expected_irreps[4], &c6),
            Character::new(&[(UnityRoot::new(1, 6), 1), (UnityRoot::new(5, 6), 1)]),
        ),
        (
            (&expected_irreps[5], &c6),
            Character::new(&[(UnityRoot::new(2, 6), 1), (UnityRoot::new(4, 6), 1)]),
        ),
    ]);
    test_character_table_construction(&mol, thresh, &expected_irreps, Some(expected_chars));
}

#[test]
fn test_character_table_construction_symmetric_au26_electric_field_c6v() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/au26.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||E|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||E|_(2)|").unwrap(),
    ];
    let c6 = ClassSymbol::<SymmetryOperation>::new("2||C6||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &c6),
            Character::new(&[(UnityRoot::new(0, 6), 1)]),
        ),
        (
            (&expected_irreps[1], &c6),
            Character::new(&[(UnityRoot::new(0, 6), 1)]),
        ),
        (
            (&expected_irreps[2], &c6),
            Character::new(&[(UnityRoot::new(3, 6), 1)]),
        ),
        (
            (&expected_irreps[3], &c6),
            Character::new(&[(UnityRoot::new(3, 6), 1)]),
        ),
        (
            (&expected_irreps[4], &c6),
            Character::new(&[(UnityRoot::new(1, 6), 1), (UnityRoot::new(5, 6), 1)]),
        ),
        (
            (&expected_irreps[5], &c6),
            Character::new(&[(UnityRoot::new(2, 6), 1), (UnityRoot::new(4, 6), 1)]),
        ),
    ]);
    test_character_table_construction(&mol, thresh, &expected_irreps, Some(expected_chars));
}

#[test]
fn test_character_table_construction_symmetric_benzene_electric_field_c6v() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/benzene.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(Vector3::new(1.0, 0.0, 0.0)));
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||E|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||E|_(2)|").unwrap(),
    ];
    let c6 = ClassSymbol::<SymmetryOperation>::new("2||C6||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &c6),
            Character::new(&[(UnityRoot::new(0, 6), 1)]),
        ),
        (
            (&expected_irreps[1], &c6),
            Character::new(&[(UnityRoot::new(0, 6), 1)]),
        ),
        (
            (&expected_irreps[2], &c6),
            Character::new(&[(UnityRoot::new(3, 6), 1)]),
        ),
        (
            (&expected_irreps[3], &c6),
            Character::new(&[(UnityRoot::new(3, 6), 1)]),
        ),
        (
            (&expected_irreps[4], &c6),
            Character::new(&[(UnityRoot::new(1, 6), 1), (UnityRoot::new(5, 6), 1)]),
        ),
        (
            (&expected_irreps[5], &c6),
            Character::new(&[(UnityRoot::new(2, 6), 1), (UnityRoot::new(4, 6), 1)]),
        ),
    ]);
    test_character_table_construction(&mol, thresh, &expected_irreps, Some(expected_chars));
}

#[test]
fn test_character_table_construction_symmetric_arbitrary_half_sandwich_cnv() {
    /* The expected number of classes is deduced from the irrep structures of
     * the Cnv groups.
     * When n is even, the irreps are A1, A2, B1, B2, Ek where k = 1, ..., n/2 - 1.
     * When n is odd, the irreps are A1, A2, Ek where k = 1, ..., n//2.
     */
    for n in 3..=32 {
        let mol = template_molecules::gen_arbitrary_half_sandwich(n);
        let thresh = 1e-7;
        let expected_irreps = if n % 2 == 0 {
            let mut irreps = vec![
                MullikenIrrepSymbol::new("||A|_(1)|").unwrap(),
                MullikenIrrepSymbol::new("||A|_(2)|").unwrap(),
                MullikenIrrepSymbol::new("||B|_(1)|").unwrap(),
                MullikenIrrepSymbol::new("||B|_(2)|").unwrap(),
            ];
            if n > 4 {
                irreps.extend(
                    (1..n.div_euclid(2))
                        .map(|k| MullikenIrrepSymbol::new(&format!("||E|_({})|", k)).unwrap()),
                );
            } else {
                irreps.push(MullikenIrrepSymbol::new("||E||").unwrap());
            }
            irreps
        } else {
            let mut irreps = vec![
                MullikenIrrepSymbol::new("||A|_(1)|").unwrap(),
                MullikenIrrepSymbol::new("||A|_(2)|").unwrap(),
            ];
            if n > 3 {
                irreps.extend(
                    (1..=n.div_euclid(2))
                        .map(|k| MullikenIrrepSymbol::new(&format!("||E|_({})|", k)).unwrap()),
                );
            } else {
                irreps.push(MullikenIrrepSymbol::new("||E||").unwrap());
            };
            irreps
        };
        test_character_table_construction(&mol, thresh, &expected_irreps, None);
    }
}

/*
Cnh
*/
#[test]
fn test_character_table_construction_symmetric_bf3_magnetic_field_c3h() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/bf3.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|^(')|").unwrap(),
        MullikenIrrepSymbol::new("||Γ|^(')_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||Γ|^(')_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||A|^('')|").unwrap(),
        MullikenIrrepSymbol::new("||Γ|^('')_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||Γ|^('')_(2)|").unwrap(),
    ];
    let c3 = ClassSymbol::<SymmetryOperation>::new("1||C3||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &c3),
            Character::new(&[(UnityRoot::new(0, 3), 1)]),
        ),
        (
            (&expected_irreps[1], &c3),
            Character::new(&[(UnityRoot::new(1, 3), 1)]),
        ),
        (
            (&expected_irreps[2], &c3),
            Character::new(&[(UnityRoot::new(2, 3), 1)]),
        ),
        (
            (&expected_irreps[3], &c3),
            Character::new(&[(UnityRoot::new(0, 3), 1)]),
        ),
        (
            (&expected_irreps[4], &c3),
            Character::new(&[(UnityRoot::new(1, 3), 1)]),
        ),
        (
            (&expected_irreps[5], &c3),
            Character::new(&[(UnityRoot::new(2, 3), 1)]),
        ),
    ]);
    test_character_table_construction(&mol, thresh, &expected_irreps, Some(expected_chars));
}

#[test]
fn test_character_table_construction_symmetric_xef4_magnetic_field_c4h() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/xef4.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, -1.0)));
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("||Γ|_(1g)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("||Γ|_(2g)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(u)|").unwrap(),
        MullikenIrrepSymbol::new("||Γ|_(1u)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(u)|").unwrap(),
        MullikenIrrepSymbol::new("||Γ|_(2u)|").unwrap(),
    ];
    let c4 = ClassSymbol::<SymmetryOperation>::new("1||C4||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &c4),
            Character::new(&[(UnityRoot::new(0, 4), 1)]),
        ),
        (
            (&expected_irreps[1], &c4),
            Character::new(&[(UnityRoot::new(1, 4), 1)]),
        ),
        (
            (&expected_irreps[2], &c4),
            Character::new(&[(UnityRoot::new(2, 4), 1)]),
        ),
        (
            (&expected_irreps[3], &c4),
            Character::new(&[(UnityRoot::new(3, 4), 1)]),
        ),
        (
            (&expected_irreps[4], &c4),
            Character::new(&[(UnityRoot::new(0, 4), 1)]),
        ),
        (
            (&expected_irreps[5], &c4),
            Character::new(&[(UnityRoot::new(1, 4), 1)]),
        ),
        (
            (&expected_irreps[6], &c4),
            Character::new(&[(UnityRoot::new(2, 4), 1)]),
        ),
        (
            (&expected_irreps[7], &c4),
            Character::new(&[(UnityRoot::new(3, 4), 1)]),
        ),
    ]);
    test_character_table_construction(&mol, thresh, &expected_irreps, Some(expected_chars));
}

#[test]
fn test_character_table_construction_symmetric_vf6_magnetic_field_c4h() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/vf6.xyz");
    let thresh = 1e-12;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("||Γ|_(1g)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("||Γ|_(2g)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(u)|").unwrap(),
        MullikenIrrepSymbol::new("||Γ|_(1u)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(u)|").unwrap(),
        MullikenIrrepSymbol::new("||Γ|_(2u)|").unwrap(),
    ];
    let c4 = ClassSymbol::<SymmetryOperation>::new("1||C4||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &c4),
            Character::new(&[(UnityRoot::new(0, 4), 1)]),
        ),
        (
            (&expected_irreps[1], &c4),
            Character::new(&[(UnityRoot::new(1, 4), 1)]),
        ),
        (
            (&expected_irreps[2], &c4),
            Character::new(&[(UnityRoot::new(2, 4), 1)]),
        ),
        (
            (&expected_irreps[3], &c4),
            Character::new(&[(UnityRoot::new(3, 4), 1)]),
        ),
        (
            (&expected_irreps[4], &c4),
            Character::new(&[(UnityRoot::new(0, 4), 1)]),
        ),
        (
            (&expected_irreps[5], &c4),
            Character::new(&[(UnityRoot::new(1, 4), 1)]),
        ),
        (
            (&expected_irreps[6], &c4),
            Character::new(&[(UnityRoot::new(2, 4), 1)]),
        ),
        (
            (&expected_irreps[7], &c4),
            Character::new(&[(UnityRoot::new(3, 4), 1)]),
        ),
    ]);
    test_character_table_construction(&mol, thresh, &expected_irreps, Some(expected_chars));
}

#[test]
fn test_character_table_construction_symmetric_h8_magnetic_field_c4h() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/h8.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, 1e-7);
    mol.set_magnetic_field(Some(0.2 * Vector3::new(0.0, 0.0, 1.0)));
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("||Γ|_(1g)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("||Γ|_(2g)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(u)|").unwrap(),
        MullikenIrrepSymbol::new("||Γ|_(1u)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(u)|").unwrap(),
        MullikenIrrepSymbol::new("||Γ|_(2u)|").unwrap(),
    ];
    let c4 = ClassSymbol::<SymmetryOperation>::new("1||C4||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &c4),
            Character::new(&[(UnityRoot::new(0, 4), 1)]),
        ),
        (
            (&expected_irreps[1], &c4),
            Character::new(&[(UnityRoot::new(1, 4), 1)]),
        ),
        (
            (&expected_irreps[2], &c4),
            Character::new(&[(UnityRoot::new(2, 4), 1)]),
        ),
        (
            (&expected_irreps[3], &c4),
            Character::new(&[(UnityRoot::new(3, 4), 1)]),
        ),
        (
            (&expected_irreps[4], &c4),
            Character::new(&[(UnityRoot::new(0, 4), 1)]),
        ),
        (
            (&expected_irreps[5], &c4),
            Character::new(&[(UnityRoot::new(1, 4), 1)]),
        ),
        (
            (&expected_irreps[6], &c4),
            Character::new(&[(UnityRoot::new(2, 4), 1)]),
        ),
        (
            (&expected_irreps[7], &c4),
            Character::new(&[(UnityRoot::new(3, 4), 1)]),
        ),
    ]);
    test_character_table_construction(&mol, thresh, &expected_irreps, Some(expected_chars));
}

#[test]
fn test_character_table_construction_symmetric_eclipsed_ferrocene_magnetic_field_c5h() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/eclipsed_ferrocene.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, -1.0)));
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|^(')|").unwrap(),
        MullikenIrrepSymbol::new("||Γ|^(')_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||Γ|^(')_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||Γ|^(')_(3)|").unwrap(),
        MullikenIrrepSymbol::new("||Γ|^(')_(4)|").unwrap(),
        MullikenIrrepSymbol::new("||A|^('')|").unwrap(),
        MullikenIrrepSymbol::new("||Γ|^('')_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||Γ|^('')_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||Γ|^('')_(3)|").unwrap(),
        MullikenIrrepSymbol::new("||Γ|^('')_(4)|").unwrap(),
    ];
    let c5 = ClassSymbol::<SymmetryOperation>::new("1||C5||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &c5),
            Character::new(&[(UnityRoot::new(0, 5), 1)]),
        ),
        (
            (&expected_irreps[1], &c5),
            Character::new(&[(UnityRoot::new(1, 5), 1)]),
        ),
        (
            (&expected_irreps[2], &c5),
            Character::new(&[(UnityRoot::new(2, 5), 1)]),
        ),
        (
            (&expected_irreps[3], &c5),
            Character::new(&[(UnityRoot::new(3, 5), 1)]),
        ),
        (
            (&expected_irreps[4], &c5),
            Character::new(&[(UnityRoot::new(4, 5), 1)]),
        ),
        (
            (&expected_irreps[5], &c5),
            Character::new(&[(UnityRoot::new(0, 5), 1)]),
        ),
        (
            (&expected_irreps[6], &c5),
            Character::new(&[(UnityRoot::new(1, 5), 1)]),
        ),
        (
            (&expected_irreps[7], &c5),
            Character::new(&[(UnityRoot::new(2, 5), 1)]),
        ),
        (
            (&expected_irreps[8], &c5),
            Character::new(&[(UnityRoot::new(3, 5), 1)]),
        ),
        (
            (&expected_irreps[9], &c5),
            Character::new(&[(UnityRoot::new(4, 5), 1)]),
        ),
    ]);
    test_character_table_construction(&mol, thresh, &expected_irreps, Some(expected_chars));
}

#[test]
fn test_character_table_construction_symmetric_benzene_magnetic_field_c6h() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/benzene.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(1.0, 0.0, 0.0)));
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("||Γ|_(1g)|").unwrap(),
        MullikenIrrepSymbol::new("||Γ|_(2g)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("||Γ|_(3g)|").unwrap(),
        MullikenIrrepSymbol::new("||Γ|_(4g)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(u)|").unwrap(),
        MullikenIrrepSymbol::new("||Γ|_(1u)|").unwrap(),
        MullikenIrrepSymbol::new("||Γ|_(2u)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(u)|").unwrap(),
        MullikenIrrepSymbol::new("||Γ|_(3u)|").unwrap(),
        MullikenIrrepSymbol::new("||Γ|_(4u)|").unwrap(),
    ];
    let c6 = ClassSymbol::<SymmetryOperation>::new("1||C6||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &c6),
            Character::new(&[(UnityRoot::new(0, 6), 1)]),
        ),
        (
            (&expected_irreps[1], &c6),
            Character::new(&[(UnityRoot::new(1, 6), 1)]),
        ),
        (
            (&expected_irreps[2], &c6),
            Character::new(&[(UnityRoot::new(2, 6), 1)]),
        ),
        (
            (&expected_irreps[3], &c6),
            Character::new(&[(UnityRoot::new(3, 6), 1)]),
        ),
        (
            (&expected_irreps[4], &c6),
            Character::new(&[(UnityRoot::new(4, 6), 1)]),
        ),
        (
            (&expected_irreps[5], &c6),
            Character::new(&[(UnityRoot::new(5, 6), 1)]),
        ),
        (
            (&expected_irreps[6], &c6),
            Character::new(&[(UnityRoot::new(0, 6), 1)]),
        ),
        (
            (&expected_irreps[7], &c6),
            Character::new(&[(UnityRoot::new(1, 6), 1)]),
        ),
        (
            (&expected_irreps[8], &c6),
            Character::new(&[(UnityRoot::new(2, 6), 1)]),
        ),
        (
            (&expected_irreps[9], &c6),
            Character::new(&[(UnityRoot::new(3, 6), 1)]),
        ),
        (
            (&expected_irreps[10], &c6),
            Character::new(&[(UnityRoot::new(4, 6), 1)]),
        ),
        (
            (&expected_irreps[11], &c6),
            Character::new(&[(UnityRoot::new(5, 6), 1)]),
        ),
    ]);
    test_character_table_construction(&mol, thresh, &expected_irreps, Some(expected_chars));
}

#[test]
fn test_character_table_construction_symmetric_arbitrary_eclipsed_sandwich_magnetic_field_cnh() {
    // env_logger::init();
    for n in 3..=20 {
        let mut mol = template_molecules::gen_arbitrary_eclipsed_sandwich(n);
        let thresh = 1e-7;
        mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 0.1)));
        let expected_irreps = if n % 2 == 0 {
            let mut irreps = vec![MullikenIrrepSymbol::new("||A|_(g)|").unwrap()];
            irreps.extend(
                (1..n.div_euclid(2))
                    .map(|k| MullikenIrrepSymbol::new(&format!("||Γ|_({}g)|", k)).unwrap()),
            );
            irreps.push(MullikenIrrepSymbol::new("||B|_(g)|").unwrap());
            irreps.extend(
                (n.div_euclid(2)..(n - 1))
                    .map(|k| MullikenIrrepSymbol::new(&format!("||Γ|_({}g)|", k)).unwrap()),
            );
            irreps.push(MullikenIrrepSymbol::new("||A|_(u)|").unwrap());
            irreps.extend(
                (1..n.div_euclid(2))
                    .map(|k| MullikenIrrepSymbol::new(&format!("||Γ|_({}u)|", k)).unwrap()),
            );
            irreps.push(MullikenIrrepSymbol::new("||B|_(u)|").unwrap());
            irreps.extend(
                (n.div_euclid(2)..(n - 1))
                    .map(|k| MullikenIrrepSymbol::new(&format!("||Γ|_({}u)|", k)).unwrap()),
            );
            irreps
        } else {
            let mut irreps = vec![MullikenIrrepSymbol::new("||A|^(')|").unwrap()];
            irreps.extend(
                (1..n).map(|k| MullikenIrrepSymbol::new(&format!("||Γ|^(')_({})|", k)).unwrap()),
            );
            irreps.push(MullikenIrrepSymbol::new("||A|^('')|").unwrap());
            irreps.extend(
                (1..n).map(|k| MullikenIrrepSymbol::new(&format!("||Γ|^('')_({})|", k)).unwrap()),
            );
            irreps
        };
        test_character_table_construction(&mol, thresh, &expected_irreps, None);
    }
}

/*
Dn
*/
#[test]
fn test_character_table_construction_symmetric_triphenyl_radical_d3() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/triphenylradical.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||E||").unwrap(),
    ];
    let c3 = ClassSymbol::<SymmetryOperation>::new("2||C3||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &c3),
            Character::new(&[(UnityRoot::new(0, 3), 1)]),
        ),
        (
            (&expected_irreps[1], &c3),
            Character::new(&[(UnityRoot::new(0, 3), 1)]),
        ),
        (
            (&expected_irreps[2], &c3),
            Character::new(&[(UnityRoot::new(1, 3), 1), (UnityRoot::new(2, 3), 1)]),
        ),
    ]);
    test_character_table_construction(&mol, thresh, &expected_irreps, Some(expected_chars));
}

#[test]
fn test_character_table_construction_symmetric_h8_twisted_d4() {
    let thresh = 1e-7;
    let mol = template_molecules::gen_twisted_h8(0.1);
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||E||").unwrap(),
    ];
    let c4 = ClassSymbol::<SymmetryOperation>::new("2||C4||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &c4),
            Character::new(&[(UnityRoot::new(0, 4), 1)]),
        ),
        (
            (&expected_irreps[1], &c4),
            Character::new(&[(UnityRoot::new(0, 4), 1)]),
        ),
        (
            (&expected_irreps[2], &c4),
            Character::new(&[(UnityRoot::new(2, 4), 1)]),
        ),
        (
            (&expected_irreps[3], &c4),
            Character::new(&[(UnityRoot::new(2, 4), 1)]),
        ),
        (
            (&expected_irreps[4], &c4),
            Character::new(&[(UnityRoot::new(1, 4), 1), (UnityRoot::new(3, 4), 1)]),
        ),
    ]);
    test_character_table_construction(&mol, thresh, &expected_irreps, Some(expected_chars));
}

#[test]
fn test_character_table_construction_symmetric_c5ph5_d5() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c5ph5.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||E|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||E|_(2)|").unwrap(),
    ];
    let c5 = ClassSymbol::<SymmetryOperation>::new("2||C5||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &c5),
            Character::new(&[(UnityRoot::new(0, 5), 1)]),
        ),
        (
            (&expected_irreps[1], &c5),
            Character::new(&[(UnityRoot::new(0, 5), 1)]),
        ),
        (
            (&expected_irreps[2], &c5),
            Character::new(&[(UnityRoot::new(1, 5), 1), (UnityRoot::new(4, 5), 1)]),
        ),
        (
            (&expected_irreps[3], &c5),
            Character::new(&[(UnityRoot::new(2, 5), 1), (UnityRoot::new(3, 5), 1)]),
        ),
    ]);
    test_character_table_construction(&mol, thresh, &expected_irreps, Some(expected_chars));
}

#[test]
fn test_character_table_construction_symmetric_c6ph6_d6() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c6ph6.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||E|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||E|_(2)|").unwrap(),
    ];
    let c6 = ClassSymbol::<SymmetryOperation>::new("2||C6||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &c6),
            Character::new(&[(UnityRoot::new(0, 6), 1)]),
        ),
        (
            (&expected_irreps[1], &c6),
            Character::new(&[(UnityRoot::new(0, 6), 1)]),
        ),
        (
            (&expected_irreps[2], &c6),
            Character::new(&[(UnityRoot::new(3, 6), 1)]),
        ),
        (
            (&expected_irreps[3], &c6),
            Character::new(&[(UnityRoot::new(3, 6), 1)]),
        ),
        (
            (&expected_irreps[4], &c6),
            Character::new(&[(UnityRoot::new(1, 6), 1), (UnityRoot::new(5, 6), 1)]),
        ),
        (
            (&expected_irreps[5], &c6),
            Character::new(&[(UnityRoot::new(2, 6), 1), (UnityRoot::new(4, 6), 1)]),
        ),
    ]);
    test_character_table_construction(&mol, thresh, &expected_irreps, Some(expected_chars));
}

#[test]
fn test_character_table_construction_symmetric_arbitrary_twisted_sandwich_dn() {
    /* The expected number of classes is deduced from the irrep structures of
     * the Dn groups.
     * When n is even, the irreps are A1, A2, B1, B2, Ek where k = 1, ..., n/2 - 1.
     * When n is odd, the irreps are A1, A2, Ek where k = 1, ..., n//2.
     */
    let thresh = 1e-7;
    for n in 3..=20 {
        let mol = template_molecules::gen_arbitrary_twisted_sandwich(n, 0.1);
        let expected_irreps = if n % 2 == 0 {
            let mut irreps = vec![
                MullikenIrrepSymbol::new("||A|_(1)|").unwrap(),
                MullikenIrrepSymbol::new("||A|_(2)|").unwrap(),
                MullikenIrrepSymbol::new("||B|_(1)|").unwrap(),
                MullikenIrrepSymbol::new("||B|_(2)|").unwrap(),
            ];
            if n > 4 {
                irreps.extend(
                    (1..n.div_euclid(2))
                        .map(|k| MullikenIrrepSymbol::new(&format!("||E|_({})|", k)).unwrap()),
                );
            } else {
                irreps.push(MullikenIrrepSymbol::new("||E||").unwrap());
            }
            irreps
        } else {
            let mut irreps = vec![
                MullikenIrrepSymbol::new("||A|_(1)|").unwrap(),
                MullikenIrrepSymbol::new("||A|_(2)|").unwrap(),
            ];
            if n > 3 {
                irreps.extend(
                    (1..=n.div_euclid(2))
                        .map(|k| MullikenIrrepSymbol::new(&format!("||E|_({})|", k)).unwrap()),
                );
            } else {
                irreps.push(MullikenIrrepSymbol::new("||E||").unwrap());
            };
            irreps
        };
        test_character_table_construction(&mol, thresh, &expected_irreps, None);
    }
}

/*
Dnh
*/
#[test]
fn test_character_table_construction_symmetric_bf3_d3h() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/bf3.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|^(')_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||A|^(')_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||E|^(')|").unwrap(),
        MullikenIrrepSymbol::new("||A|^('')_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||A|^('')_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||E|^('')|").unwrap(),
    ];
    let c3 = ClassSymbol::<SymmetryOperation>::new("2||C3||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &c3),
            Character::new(&[(UnityRoot::new(0, 3), 1)]),
        ),
        (
            (&expected_irreps[1], &c3),
            Character::new(&[(UnityRoot::new(0, 3), 1)]),
        ),
        (
            (&expected_irreps[2], &c3),
            Character::new(&[(UnityRoot::new(1, 3), 1), (UnityRoot::new(2, 3), 1)]),
        ),
        (
            (&expected_irreps[3], &c3),
            Character::new(&[(UnityRoot::new(0, 3), 1)]),
        ),
        (
            (&expected_irreps[4], &c3),
            Character::new(&[(UnityRoot::new(0, 3), 1)]),
        ),
        (
            (&expected_irreps[5], &c3),
            Character::new(&[(UnityRoot::new(1, 3), 1), (UnityRoot::new(2, 3), 1)]),
        ),
    ]);
    test_character_table_construction(&mol, thresh, &expected_irreps, Some(expected_chars));
}

#[test]
fn test_character_table_construction_symmetric_xef4_d4h() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/xef4.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
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
    let c4 = ClassSymbol::<SymmetryOperation>::new("2||C4||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &c4),
            Character::new(&[(UnityRoot::new(0, 4), 1)]),
        ),
        (
            (&expected_irreps[1], &c4),
            Character::new(&[(UnityRoot::new(0, 4), 1)]),
        ),
        (
            (&expected_irreps[2], &c4),
            Character::new(&[(UnityRoot::new(2, 4), 1)]),
        ),
        (
            (&expected_irreps[3], &c4),
            Character::new(&[(UnityRoot::new(2, 4), 1)]),
        ),
        (
            (&expected_irreps[4], &c4),
            Character::new(&[(UnityRoot::new(1, 4), 1), (UnityRoot::new(3, 4), 1)]),
        ),
        (
            (&expected_irreps[5], &c4),
            Character::new(&[(UnityRoot::new(0, 4), 1)]),
        ),
        (
            (&expected_irreps[6], &c4),
            Character::new(&[(UnityRoot::new(0, 4), 1)]),
        ),
        (
            (&expected_irreps[7], &c4),
            Character::new(&[(UnityRoot::new(2, 4), 1)]),
        ),
        (
            (&expected_irreps[8], &c4),
            Character::new(&[(UnityRoot::new(2, 4), 1)]),
        ),
        (
            (&expected_irreps[9], &c4),
            Character::new(&[(UnityRoot::new(1, 4), 1), (UnityRoot::new(3, 4), 1)]),
        ),
    ]);
    test_character_table_construction(&mol, thresh, &expected_irreps, Some(expected_chars));
}

#[test]
fn test_character_table_construction_symmetric_h8_d4h() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/h8.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
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
    let c4 = ClassSymbol::<SymmetryOperation>::new("2||C4||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &c4),
            Character::new(&[(UnityRoot::new(0, 4), 1)]),
        ),
        (
            (&expected_irreps[1], &c4),
            Character::new(&[(UnityRoot::new(0, 4), 1)]),
        ),
        (
            (&expected_irreps[2], &c4),
            Character::new(&[(UnityRoot::new(2, 4), 1)]),
        ),
        (
            (&expected_irreps[3], &c4),
            Character::new(&[(UnityRoot::new(2, 4), 1)]),
        ),
        (
            (&expected_irreps[4], &c4),
            Character::new(&[(UnityRoot::new(1, 4), 1), (UnityRoot::new(3, 4), 1)]),
        ),
        (
            (&expected_irreps[5], &c4),
            Character::new(&[(UnityRoot::new(0, 4), 1)]),
        ),
        (
            (&expected_irreps[6], &c4),
            Character::new(&[(UnityRoot::new(0, 4), 1)]),
        ),
        (
            (&expected_irreps[7], &c4),
            Character::new(&[(UnityRoot::new(2, 4), 1)]),
        ),
        (
            (&expected_irreps[8], &c4),
            Character::new(&[(UnityRoot::new(2, 4), 1)]),
        ),
        (
            (&expected_irreps[9], &c4),
            Character::new(&[(UnityRoot::new(1, 4), 1), (UnityRoot::new(3, 4), 1)]),
        ),
    ]);
    test_character_table_construction(&mol, thresh, &expected_irreps, Some(expected_chars));
}

#[test]
fn test_character_table_construction_symmetric_eclipsed_ferrocene_d5h() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/eclipsed_ferrocene.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|^(')_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||A|^(')_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||E|^(')_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||E|^(')_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||A|^('')_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||A|^('')_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||E|^('')_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||E|^('')_(2)|").unwrap(),
    ];
    let c5 = ClassSymbol::<SymmetryOperation>::new("2||C5||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &c5),
            Character::new(&[(UnityRoot::new(0, 5), 1)]),
        ),
        (
            (&expected_irreps[1], &c5),
            Character::new(&[(UnityRoot::new(0, 5), 1)]),
        ),
        (
            (&expected_irreps[2], &c5),
            Character::new(&[(UnityRoot::new(1, 5), 1), (UnityRoot::new(4, 5), 1)]),
        ),
        (
            (&expected_irreps[3], &c5),
            Character::new(&[(UnityRoot::new(2, 5), 1), (UnityRoot::new(3, 5), 1)]),
        ),
        (
            (&expected_irreps[4], &c5),
            Character::new(&[(UnityRoot::new(0, 5), 1)]),
        ),
        (
            (&expected_irreps[5], &c5),
            Character::new(&[(UnityRoot::new(0, 5), 1)]),
        ),
        (
            (&expected_irreps[6], &c5),
            Character::new(&[(UnityRoot::new(1, 5), 1), (UnityRoot::new(4, 5), 1)]),
        ),
        (
            (&expected_irreps[7], &c5),
            Character::new(&[(UnityRoot::new(2, 5), 1), (UnityRoot::new(3, 5), 1)]),
        ),
    ]);
    test_character_table_construction(&mol, thresh, &expected_irreps, Some(expected_chars));
}

#[test]
fn test_character_table_construction_symmetric_benzene_d6h() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/benzene.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(1g)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(2g)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(1g)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(2g)|").unwrap(),
        MullikenIrrepSymbol::new("||E|_(1g)|").unwrap(),
        MullikenIrrepSymbol::new("||E|_(2g)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(1u)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(2u)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(1u)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(2u)|").unwrap(),
        MullikenIrrepSymbol::new("||E|_(1u)|").unwrap(),
        MullikenIrrepSymbol::new("||E|_(2u)|").unwrap(),
    ];
    let c6 = ClassSymbol::<SymmetryOperation>::new("2||C6||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &c6),
            Character::new(&[(UnityRoot::new(0, 6), 1)]),
        ),
        (
            (&expected_irreps[1], &c6),
            Character::new(&[(UnityRoot::new(0, 6), 1)]),
        ),
        (
            (&expected_irreps[2], &c6),
            Character::new(&[(UnityRoot::new(3, 6), 1)]),
        ),
        (
            (&expected_irreps[3], &c6),
            Character::new(&[(UnityRoot::new(3, 6), 1)]),
        ),
        (
            (&expected_irreps[4], &c6),
            Character::new(&[(UnityRoot::new(1, 6), 1), (UnityRoot::new(5, 6), 1)]),
        ),
        (
            (&expected_irreps[5], &c6),
            Character::new(&[(UnityRoot::new(2, 6), 1), (UnityRoot::new(4, 6), 1)]),
        ),
        (
            (&expected_irreps[6], &c6),
            Character::new(&[(UnityRoot::new(0, 6), 1)]),
        ),
        (
            (&expected_irreps[7], &c6),
            Character::new(&[(UnityRoot::new(0, 6), 1)]),
        ),
        (
            (&expected_irreps[8], &c6),
            Character::new(&[(UnityRoot::new(3, 6), 1)]),
        ),
        (
            (&expected_irreps[9], &c6),
            Character::new(&[(UnityRoot::new(3, 6), 1)]),
        ),
        (
            (&expected_irreps[10], &c6),
            Character::new(&[(UnityRoot::new(1, 6), 1), (UnityRoot::new(5, 6), 1)]),
        ),
        (
            (&expected_irreps[11], &c6),
            Character::new(&[(UnityRoot::new(2, 6), 1), (UnityRoot::new(4, 6), 1)]),
        ),
    ]);
    test_character_table_construction(&mol, thresh, &expected_irreps, Some(expected_chars));
}

#[test]
fn test_character_table_construction_symmetric_8_eclipsed_sandwich_d8h() {
    let thresh = 1e-7;
    let mol = template_molecules::gen_arbitrary_eclipsed_sandwich(8);
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(1g)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(2g)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(1g)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(2g)|").unwrap(),
        MullikenIrrepSymbol::new("||E|_(1g)|").unwrap(),
        MullikenIrrepSymbol::new("||E|_(2g)|").unwrap(),
        MullikenIrrepSymbol::new("||E|_(3g)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(1u)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(2u)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(1u)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(2u)|").unwrap(),
        MullikenIrrepSymbol::new("||E|_(1u)|").unwrap(),
        MullikenIrrepSymbol::new("||E|_(2u)|").unwrap(),
        MullikenIrrepSymbol::new("||E|_(3u)|").unwrap(),
    ];
    let c8 = ClassSymbol::<SymmetryOperation>::new("2||C8||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &c8),
            Character::new(&[(UnityRoot::new(0, 8), 1)]),
        ),
        (
            (&expected_irreps[1], &c8),
            Character::new(&[(UnityRoot::new(0, 8), 1)]),
        ),
        (
            (&expected_irreps[2], &c8),
            Character::new(&[(UnityRoot::new(4, 8), 1)]),
        ),
        (
            (&expected_irreps[3], &c8),
            Character::new(&[(UnityRoot::new(4, 8), 1)]),
        ),
        (
            (&expected_irreps[4], &c8),
            Character::new(&[(UnityRoot::new(1, 8), 1), (UnityRoot::new(7, 8), 1)]),
        ),
        (
            (&expected_irreps[5], &c8),
            Character::new(&[(UnityRoot::new(2, 8), 1), (UnityRoot::new(6, 8), 1)]),
        ),
        (
            (&expected_irreps[6], &c8),
            Character::new(&[(UnityRoot::new(3, 8), 1), (UnityRoot::new(5, 8), 1)]),
        ),
        (
            (&expected_irreps[7], &c8),
            Character::new(&[(UnityRoot::new(0, 8), 1)]),
        ),
        (
            (&expected_irreps[8], &c8),
            Character::new(&[(UnityRoot::new(0, 8), 1)]),
        ),
        (
            (&expected_irreps[9], &c8),
            Character::new(&[(UnityRoot::new(4, 8), 1)]),
        ),
        (
            (&expected_irreps[10], &c8),
            Character::new(&[(UnityRoot::new(4, 8), 1)]),
        ),
        (
            (&expected_irreps[11], &c8),
            Character::new(&[(UnityRoot::new(1, 8), 1), (UnityRoot::new(7, 8), 1)]),
        ),
        (
            (&expected_irreps[12], &c8),
            Character::new(&[(UnityRoot::new(2, 8), 1), (UnityRoot::new(6, 8), 1)]),
        ),
        (
            (&expected_irreps[13], &c8),
            Character::new(&[(UnityRoot::new(3, 8), 1), (UnityRoot::new(5, 8), 1)]),
        ),
    ]);
    test_character_table_construction(&mol, thresh, &expected_irreps, Some(expected_chars));
}

// #[test]
// fn test_character_table_construction_symmetric_h100_d100h() {
//     env_logger::init();
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/h100.xyz");
//     let thresh = 1e-6;
//     let mol = Molecule::from_xyz(&path, thresh);
//     test_character_table_construction(&mol, thresh);
// }

#[test]
fn test_character_table_construction_symmetric_arbitrary_eclipsed_sandwich_dnh() {
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
        let expected_irreps = if n % 2 == 0 {
            let mut irreps_gu = vec![];
            for parity in ["g", "u"] {
                let mut irreps = vec![
                    MullikenIrrepSymbol::new(&format!("||A|_(1{parity})|")).unwrap(),
                    MullikenIrrepSymbol::new(&format!("||A|_(2{parity})|")).unwrap(),
                    MullikenIrrepSymbol::new(&format!("||B|_(1{parity})|")).unwrap(),
                    MullikenIrrepSymbol::new(&format!("||B|_(2{parity})|")).unwrap(),
                ];
                if n > 4 {
                    irreps.extend((1..n.div_euclid(2)).map(|k| {
                        MullikenIrrepSymbol::new(&format!("||E|_({k}{parity})|")).unwrap()
                    }));
                } else {
                    irreps.push(MullikenIrrepSymbol::new(&format!("||E|_({parity})|")).unwrap());
                }
                irreps_gu.extend(irreps)
            }
            irreps_gu
        } else {
            let mut irreps_ddd = vec![];
            for parity in ["'", "''"] {
                let mut irreps = vec![
                    MullikenIrrepSymbol::new(&format!("||A|^({parity})_(1)|")).unwrap(),
                    MullikenIrrepSymbol::new(&format!("||A|^({parity})_(2)|")).unwrap(),
                ];
                if n > 3 {
                    irreps.extend((1..=n.div_euclid(2)).map(|k| {
                        MullikenIrrepSymbol::new(&format!("||E|^({parity})_({k})|")).unwrap()
                    }));
                } else {
                    irreps.push(MullikenIrrepSymbol::new(&format!("||E|^({parity})|")).unwrap());
                };
                irreps_ddd.extend(irreps)
            }
            irreps_ddd
        };
        test_character_table_construction(&mol, thresh, &expected_irreps, None);
    }
}

/*
Dnd
*/
#[test]
fn test_character_table_construction_symmetric_b2cl4_d2d() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/b2cl4.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||E||").unwrap(),
    ];
    let s4 = ClassSymbol::<SymmetryOperation>::new("2||S4||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &s4),
            Character::new(&[(UnityRoot::new(0, 4), 1)]),
        ),
        (
            (&expected_irreps[1], &s4),
            Character::new(&[(UnityRoot::new(0, 4), 1)]),
        ),
        (
            (&expected_irreps[2], &s4),
            Character::new(&[(UnityRoot::new(2, 4), 1)]),
        ),
        (
            (&expected_irreps[3], &s4),
            Character::new(&[(UnityRoot::new(2, 4), 1)]),
        ),
        (
            (&expected_irreps[4], &s4),
            Character::new(&[(UnityRoot::new(1, 4), 1), (UnityRoot::new(3, 4), 1)]),
        ),
    ]);
    test_character_table_construction(&mol, thresh, &expected_irreps, Some(expected_chars));
}

#[test]
fn test_character_table_construction_symmetric_s4n4_d2d() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/s4n4.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||E||").unwrap(),
    ];
    let s4 = ClassSymbol::<SymmetryOperation>::new("2||S4||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &s4),
            Character::new(&[(UnityRoot::new(0, 4), 1)]),
        ),
        (
            (&expected_irreps[1], &s4),
            Character::new(&[(UnityRoot::new(0, 4), 1)]),
        ),
        (
            (&expected_irreps[2], &s4),
            Character::new(&[(UnityRoot::new(2, 4), 1)]),
        ),
        (
            (&expected_irreps[3], &s4),
            Character::new(&[(UnityRoot::new(2, 4), 1)]),
        ),
        (
            (&expected_irreps[4], &s4),
            Character::new(&[(UnityRoot::new(1, 4), 1), (UnityRoot::new(3, 4), 1)]),
        ),
    ]);
    test_character_table_construction(&mol, thresh, &expected_irreps, Some(expected_chars));
}

#[test]
fn test_character_table_construction_symmetric_pbet4_d2d() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/pbet4.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||E||").unwrap(),
    ];
    let s4 = ClassSymbol::<SymmetryOperation>::new("2||S4||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &s4),
            Character::new(&[(UnityRoot::new(0, 4), 1)]),
        ),
        (
            (&expected_irreps[1], &s4),
            Character::new(&[(UnityRoot::new(0, 4), 1)]),
        ),
        (
            (&expected_irreps[2], &s4),
            Character::new(&[(UnityRoot::new(2, 4), 1)]),
        ),
        (
            (&expected_irreps[3], &s4),
            Character::new(&[(UnityRoot::new(2, 4), 1)]),
        ),
        (
            (&expected_irreps[4], &s4),
            Character::new(&[(UnityRoot::new(1, 4), 1), (UnityRoot::new(3, 4), 1)]),
        ),
    ]);
    test_character_table_construction(&mol, thresh, &expected_irreps, Some(expected_chars));
}

#[test]
fn test_character_table_construction_symmetric_allene_d2d() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/allene.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||E||").unwrap(),
    ];
    let s4 = ClassSymbol::<SymmetryOperation>::new("2||S4||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &s4),
            Character::new(&[(UnityRoot::new(0, 4), 1)]),
        ),
        (
            (&expected_irreps[1], &s4),
            Character::new(&[(UnityRoot::new(0, 4), 1)]),
        ),
        (
            (&expected_irreps[2], &s4),
            Character::new(&[(UnityRoot::new(2, 4), 1)]),
        ),
        (
            (&expected_irreps[3], &s4),
            Character::new(&[(UnityRoot::new(2, 4), 1)]),
        ),
        (
            (&expected_irreps[4], &s4),
            Character::new(&[(UnityRoot::new(1, 4), 1), (UnityRoot::new(3, 4), 1)]),
        ),
    ]);
    test_character_table_construction(&mol, thresh, &expected_irreps, Some(expected_chars));
}

#[test]
fn test_character_table_construction_symmetric_staggered_c2h6_d3d() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c2h6.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(1g)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(2g)|").unwrap(),
        MullikenIrrepSymbol::new("||E|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(1u)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(2u)|").unwrap(),
        MullikenIrrepSymbol::new("||E|_(u)|").unwrap(),
    ];
    let c3 = ClassSymbol::<SymmetryOperation>::new("2||C3||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &c3),
            Character::new(&[(UnityRoot::new(0, 3), 1)]),
        ),
        (
            (&expected_irreps[1], &c3),
            Character::new(&[(UnityRoot::new(0, 3), 1)]),
        ),
        (
            (&expected_irreps[2], &c3),
            Character::new(&[(UnityRoot::new(1, 3), 1), (UnityRoot::new(2, 3), 1)]),
        ),
        (
            (&expected_irreps[3], &c3),
            Character::new(&[(UnityRoot::new(0, 3), 1)]),
        ),
        (
            (&expected_irreps[4], &c3),
            Character::new(&[(UnityRoot::new(0, 3), 1)]),
        ),
        (
            (&expected_irreps[5], &c3),
            Character::new(&[(UnityRoot::new(1, 3), 1), (UnityRoot::new(2, 3), 1)]),
        ),
    ]);
    test_character_table_construction(&mol, thresh, &expected_irreps, Some(expected_chars));
}

#[test]
fn test_character_table_construction_symmetric_cyclohexane_chair_d3d() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/cyclohexane_chair.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(1g)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(2g)|").unwrap(),
        MullikenIrrepSymbol::new("||E|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(1u)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(2u)|").unwrap(),
        MullikenIrrepSymbol::new("||E|_(u)|").unwrap(),
    ];
    let c3 = ClassSymbol::<SymmetryOperation>::new("2||C3||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &c3),
            Character::new(&[(UnityRoot::new(0, 3), 1)]),
        ),
        (
            (&expected_irreps[1], &c3),
            Character::new(&[(UnityRoot::new(0, 3), 1)]),
        ),
        (
            (&expected_irreps[2], &c3),
            Character::new(&[(UnityRoot::new(1, 3), 1), (UnityRoot::new(2, 3), 1)]),
        ),
        (
            (&expected_irreps[3], &c3),
            Character::new(&[(UnityRoot::new(0, 3), 1)]),
        ),
        (
            (&expected_irreps[4], &c3),
            Character::new(&[(UnityRoot::new(0, 3), 1)]),
        ),
        (
            (&expected_irreps[5], &c3),
            Character::new(&[(UnityRoot::new(1, 3), 1), (UnityRoot::new(2, 3), 1)]),
        ),
    ]);
    test_character_table_construction(&mol, thresh, &expected_irreps, Some(expected_chars));
}

#[test]
fn test_character_table_construction_symmetric_s8_d4d() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/s8.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||E|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||E|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||E|_(3)|").unwrap(),
    ];
    let s8 = ClassSymbol::<SymmetryOperation>::new("2||S8||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &s8),
            Character::new(&[(UnityRoot::new(0, 8), 1)]),
        ),
        (
            (&expected_irreps[1], &s8),
            Character::new(&[(UnityRoot::new(0, 8), 1)]),
        ),
        (
            (&expected_irreps[2], &s8),
            Character::new(&[(UnityRoot::new(4, 8), 1)]),
        ),
        (
            (&expected_irreps[3], &s8),
            Character::new(&[(UnityRoot::new(4, 8), 1)]),
        ),
        (
            (&expected_irreps[4], &s8),
            Character::new(&[(UnityRoot::new(1, 8), 1), (UnityRoot::new(7, 8), 1)]),
        ),
        (
            (&expected_irreps[5], &s8),
            Character::new(&[(UnityRoot::new(2, 8), 1), (UnityRoot::new(6, 8), 1)]),
        ),
        (
            (&expected_irreps[6], &s8),
            Character::new(&[(UnityRoot::new(3, 8), 1), (UnityRoot::new(5, 8), 1)]),
        ),
    ]);
    test_character_table_construction(&mol, thresh, &expected_irreps, Some(expected_chars));
}

#[test]
fn test_character_table_construction_symmetric_antiprism_h8_d4d() {
    let mol = template_molecules::gen_twisted_h8(std::f64::consts::FRAC_PI_4);
    let thresh = 1e-7;
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||E|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||E|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||E|_(3)|").unwrap(),
    ];
    let s8 = ClassSymbol::<SymmetryOperation>::new("2||S8||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &s8),
            Character::new(&[(UnityRoot::new(0, 8), 1)]),
        ),
        (
            (&expected_irreps[1], &s8),
            Character::new(&[(UnityRoot::new(0, 8), 1)]),
        ),
        (
            (&expected_irreps[2], &s8),
            Character::new(&[(UnityRoot::new(4, 8), 1)]),
        ),
        (
            (&expected_irreps[3], &s8),
            Character::new(&[(UnityRoot::new(4, 8), 1)]),
        ),
        (
            (&expected_irreps[4], &s8),
            Character::new(&[(UnityRoot::new(1, 8), 1), (UnityRoot::new(7, 8), 1)]),
        ),
        (
            (&expected_irreps[5], &s8),
            Character::new(&[(UnityRoot::new(2, 8), 1), (UnityRoot::new(6, 8), 1)]),
        ),
        (
            (&expected_irreps[6], &s8),
            Character::new(&[(UnityRoot::new(3, 8), 1), (UnityRoot::new(5, 8), 1)]),
        ),
    ]);
    test_character_table_construction(&mol, thresh, &expected_irreps, Some(expected_chars));
}

#[test]
fn test_character_table_construction_symmetric_antiprism_pb10_d4d() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/pb10.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||E|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||E|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||E|_(3)|").unwrap(),
    ];
    let s8 = ClassSymbol::<SymmetryOperation>::new("2||S8||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &s8),
            Character::new(&[(UnityRoot::new(0, 8), 1)]),
        ),
        (
            (&expected_irreps[1], &s8),
            Character::new(&[(UnityRoot::new(0, 8), 1)]),
        ),
        (
            (&expected_irreps[2], &s8),
            Character::new(&[(UnityRoot::new(4, 8), 1)]),
        ),
        (
            (&expected_irreps[3], &s8),
            Character::new(&[(UnityRoot::new(4, 8), 1)]),
        ),
        (
            (&expected_irreps[4], &s8),
            Character::new(&[(UnityRoot::new(1, 8), 1), (UnityRoot::new(7, 8), 1)]),
        ),
        (
            (&expected_irreps[5], &s8),
            Character::new(&[(UnityRoot::new(2, 8), 1), (UnityRoot::new(6, 8), 1)]),
        ),
        (
            (&expected_irreps[6], &s8),
            Character::new(&[(UnityRoot::new(3, 8), 1), (UnityRoot::new(5, 8), 1)]),
        ),
    ]);
    test_character_table_construction(&mol, thresh, &expected_irreps, Some(expected_chars));
}

#[test]
fn test_character_table_construction_symmetric_staggered_ferrocene_d5d() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/staggered_ferrocene.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(1g)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(2g)|").unwrap(),
        MullikenIrrepSymbol::new("||E|_(1g)|").unwrap(),
        MullikenIrrepSymbol::new("||E|_(2g)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(1u)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(2u)|").unwrap(),
        MullikenIrrepSymbol::new("||E|_(1u)|").unwrap(),
        MullikenIrrepSymbol::new("||E|_(2u)|").unwrap(),
    ];
    let c5 = ClassSymbol::<SymmetryOperation>::new("2||C5||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &c5),
            Character::new(&[(UnityRoot::new(0, 5), 1)]),
        ),
        (
            (&expected_irreps[1], &c5),
            Character::new(&[(UnityRoot::new(0, 5), 1)]),
        ),
        (
            (&expected_irreps[2], &c5),
            Character::new(&[(UnityRoot::new(1, 5), 1), (UnityRoot::new(4, 5), 1)]),
        ),
        (
            (&expected_irreps[3], &c5),
            Character::new(&[(UnityRoot::new(2, 5), 1), (UnityRoot::new(3, 5), 1)]),
        ),
        (
            (&expected_irreps[4], &c5),
            Character::new(&[(UnityRoot::new(0, 5), 1)]),
        ),
        (
            (&expected_irreps[5], &c5),
            Character::new(&[(UnityRoot::new(0, 5), 1)]),
        ),
        (
            (&expected_irreps[6], &c5),
            Character::new(&[(UnityRoot::new(1, 5), 1), (UnityRoot::new(4, 5), 1)]),
        ),
        (
            (&expected_irreps[7], &c5),
            Character::new(&[(UnityRoot::new(2, 5), 1), (UnityRoot::new(3, 5), 1)]),
        ),
    ]);
    test_character_table_construction(&mol, thresh, &expected_irreps, Some(expected_chars));
}

#[test]
fn test_character_table_construction_symmetric_au26_d6d() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/au26.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||E|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||E|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||E|_(3)|").unwrap(),
        MullikenIrrepSymbol::new("||E|_(4)|").unwrap(),
        MullikenIrrepSymbol::new("||E|_(5)|").unwrap(),
    ];
    let s12 = ClassSymbol::<SymmetryOperation>::new("2||S12||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &s12),
            Character::new(&[(UnityRoot::new(0, 12), 1)]),
        ),
        (
            (&expected_irreps[1], &s12),
            Character::new(&[(UnityRoot::new(0, 12), 1)]),
        ),
        (
            (&expected_irreps[2], &s12),
            Character::new(&[(UnityRoot::new(6, 12), 1)]),
        ),
        (
            (&expected_irreps[3], &s12),
            Character::new(&[(UnityRoot::new(6, 12), 1)]),
        ),
        (
            (&expected_irreps[4], &s12),
            Character::new(&[(UnityRoot::new(1, 12), 1), (UnityRoot::new(11, 12), 1)]),
        ),
        (
            (&expected_irreps[5], &s12),
            Character::new(&[(UnityRoot::new(2, 12), 1), (UnityRoot::new(10, 12), 1)]),
        ),
        (
            (&expected_irreps[6], &s12),
            Character::new(&[(UnityRoot::new(3, 12), 1), (UnityRoot::new(9, 12), 1)]),
        ),
        (
            (&expected_irreps[7], &s12),
            Character::new(&[(UnityRoot::new(4, 12), 1), (UnityRoot::new(8, 12), 1)]),
        ),
        (
            (&expected_irreps[8], &s12),
            Character::new(&[(UnityRoot::new(5, 12), 1), (UnityRoot::new(7, 12), 1)]),
        ),
    ]);
    test_character_table_construction(&mol, thresh, &expected_irreps, Some(expected_chars));
}

#[test]
fn test_character_table_construction_symmetric_arbitrary_staggered_sandwich_dnd() {
    let thresh = 1e-7;
    for n in 3..=20 {
        let mol = template_molecules::gen_arbitrary_twisted_sandwich(n, 0.5);
        let expected_irreps = if n % 2 != 0 {
            // Odd n, g/u possible
            let mut irreps_gu = vec![];
            for parity in ["g", "u"] {
                let mut irreps = vec![
                    MullikenIrrepSymbol::new(&format!("||A|_(1{parity})|")).unwrap(),
                    MullikenIrrepSymbol::new(&format!("||A|_(2{parity})|")).unwrap(),
                ];
                if n > 4 {
                    irreps.extend((1..=n.div_euclid(2)).map(|k| {
                        MullikenIrrepSymbol::new(&format!("||E|_({k}{parity})|")).unwrap()
                    }));
                } else {
                    irreps.push(MullikenIrrepSymbol::new(&format!("||E|_({parity})|")).unwrap());
                }
                irreps_gu.extend(irreps)
            }
            irreps_gu
        } else {
            // Even n, no g/u
            let mut irreps = vec![
                MullikenIrrepSymbol::new("||A|_(1)|").unwrap(),
                MullikenIrrepSymbol::new("||A|_(2)|").unwrap(),
                MullikenIrrepSymbol::new("||B|_(1)|").unwrap(),
                MullikenIrrepSymbol::new("||B|_(2)|").unwrap(),
            ];
            if n > 2 {
                irreps.extend(
                    (1..n).map(|k| MullikenIrrepSymbol::new(&format!("||E|_({k})|")).unwrap()),
                );
            } else {
                irreps.push(MullikenIrrepSymbol::new("||E||").unwrap());
            };
            irreps
        };
        test_character_table_construction(&mol, thresh, &expected_irreps, None);
    }
}

/*
S2n
*/
#[test]
fn test_character_table_construction_symmetric_b2cl4_magnetic_field_s4() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/b2cl4.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A||").unwrap(),
        MullikenIrrepSymbol::new("||Γ|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||B||").unwrap(),
        MullikenIrrepSymbol::new("||Γ|_(2)|").unwrap(),
    ];
    let s4 = ClassSymbol::<SymmetryOperation>::new("1||S4||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &s4),
            Character::new(&[(UnityRoot::new(0, 4), 1)]),
        ),
        (
            (&expected_irreps[1], &s4),
            Character::new(&[(UnityRoot::new(1, 4), 1)]),
        ),
        (
            (&expected_irreps[2], &s4),
            Character::new(&[(UnityRoot::new(2, 4), 1)]),
        ),
        (
            (&expected_irreps[3], &s4),
            Character::new(&[(UnityRoot::new(3, 4), 1)]),
        ),
    ]);
    test_character_table_construction(&mol, thresh, &expected_irreps, Some(expected_chars));
}

#[test]
fn test_character_table_construction_symmetric_adamantane_magnetic_field_s4() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/adamantane.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(-0.1, 0.0, 0.0)));
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A||").unwrap(),
        MullikenIrrepSymbol::new("||Γ|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||B||").unwrap(),
        MullikenIrrepSymbol::new("||Γ|_(2)|").unwrap(),
    ];
    let s4 = ClassSymbol::<SymmetryOperation>::new("1||S4||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &s4),
            Character::new(&[(UnityRoot::new(0, 4), 1)]),
        ),
        (
            (&expected_irreps[1], &s4),
            Character::new(&[(UnityRoot::new(1, 4), 1)]),
        ),
        (
            (&expected_irreps[2], &s4),
            Character::new(&[(UnityRoot::new(2, 4), 1)]),
        ),
        (
            (&expected_irreps[3], &s4),
            Character::new(&[(UnityRoot::new(3, 4), 1)]),
        ),
    ]);
    test_character_table_construction(&mol, thresh, &expected_irreps, Some(expected_chars));
}

#[test]
fn test_character_table_construction_symmetric_ch4_magnetic_field_s4() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/ch4.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A||").unwrap(),
        MullikenIrrepSymbol::new("||Γ|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||B||").unwrap(),
        MullikenIrrepSymbol::new("||Γ|_(2)|").unwrap(),
    ];
    let s4 = ClassSymbol::<SymmetryOperation>::new("1||S4||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &s4),
            Character::new(&[(UnityRoot::new(0, 4), 1)]),
        ),
        (
            (&expected_irreps[1], &s4),
            Character::new(&[(UnityRoot::new(1, 4), 1)]),
        ),
        (
            (&expected_irreps[2], &s4),
            Character::new(&[(UnityRoot::new(2, 4), 1)]),
        ),
        (
            (&expected_irreps[3], &s4),
            Character::new(&[(UnityRoot::new(3, 4), 1)]),
        ),
    ]);
    test_character_table_construction(&mol, thresh, &expected_irreps, Some(expected_chars));
}

#[test]
fn test_character_table_construction_symmetric_65coronane_s6() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/coronane65.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("||Γ|_(1g)|").unwrap(),
        MullikenIrrepSymbol::new("||Γ|_(2g)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(u)|").unwrap(),
        MullikenIrrepSymbol::new("||Γ|_(1u)|").unwrap(),
        MullikenIrrepSymbol::new("||Γ|_(2u)|").unwrap(),
    ];
    let c3 = ClassSymbol::<SymmetryOperation>::new("1||C3||", None).unwrap();
    let s6 = ClassSymbol::<SymmetryOperation>::new("1||S6||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &c3),
            Character::new(&[(UnityRoot::new(0, 3), 1)]),
        ),
        (
            (&expected_irreps[1], &c3),
            Character::new(&[(UnityRoot::new(1, 3), 1)]),
        ),
        (
            (&expected_irreps[2], &c3),
            Character::new(&[(UnityRoot::new(2, 3), 1)]),
        ),
        (
            (&expected_irreps[3], &c3),
            Character::new(&[(UnityRoot::new(0, 3), 1)]),
        ),
        (
            (&expected_irreps[4], &c3),
            Character::new(&[(UnityRoot::new(1, 3), 1)]),
        ),
        (
            (&expected_irreps[5], &c3),
            Character::new(&[(UnityRoot::new(2, 3), 1)]),
        ),
        (
            (&expected_irreps[0], &s6),
            Character::new(&[(UnityRoot::new(0, 6), 1)]),
        ),
        (
            (&expected_irreps[1], &s6),
            Character::new(&[(UnityRoot::new(4, 6), 1)]),
        ),
        (
            (&expected_irreps[2], &s6),
            Character::new(&[(UnityRoot::new(2, 6), 1)]),
        ),
        (
            (&expected_irreps[3], &s6),
            Character::new(&[(UnityRoot::new(3, 6), 1)]),
        ),
        (
            (&expected_irreps[4], &s6),
            Character::new(&[(UnityRoot::new(1, 6), 1)]),
        ),
        (
            (&expected_irreps[5], &s6),
            Character::new(&[(UnityRoot::new(5, 6), 1)]),
        ),
    ]);
    test_character_table_construction(&mol, thresh, &expected_irreps, Some(expected_chars));
}

#[test]
fn test_character_table_construction_symmetric_65coronane_magnetic_field_s6() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/coronane65.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, -1.0)));
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("||Γ|_(1g)|").unwrap(),
        MullikenIrrepSymbol::new("||Γ|_(2g)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(u)|").unwrap(),
        MullikenIrrepSymbol::new("||Γ|_(1u)|").unwrap(),
        MullikenIrrepSymbol::new("||Γ|_(2u)|").unwrap(),
    ];
    let c3 = ClassSymbol::<SymmetryOperation>::new("1||C3||", None).unwrap();
    let s6 = ClassSymbol::<SymmetryOperation>::new("1||S6||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &c3),
            Character::new(&[(UnityRoot::new(0, 3), 1)]),
        ),
        (
            (&expected_irreps[1], &c3),
            Character::new(&[(UnityRoot::new(1, 3), 1)]),
        ),
        (
            (&expected_irreps[2], &c3),
            Character::new(&[(UnityRoot::new(2, 3), 1)]),
        ),
        (
            (&expected_irreps[3], &c3),
            Character::new(&[(UnityRoot::new(0, 3), 1)]),
        ),
        (
            (&expected_irreps[4], &c3),
            Character::new(&[(UnityRoot::new(1, 3), 1)]),
        ),
        (
            (&expected_irreps[5], &c3),
            Character::new(&[(UnityRoot::new(2, 3), 1)]),
        ),
        (
            (&expected_irreps[0], &s6),
            Character::new(&[(UnityRoot::new(0, 6), 1)]),
        ),
        (
            (&expected_irreps[1], &s6),
            Character::new(&[(UnityRoot::new(4, 6), 1)]),
        ),
        (
            (&expected_irreps[2], &s6),
            Character::new(&[(UnityRoot::new(2, 6), 1)]),
        ),
        (
            (&expected_irreps[3], &s6),
            Character::new(&[(UnityRoot::new(3, 6), 1)]),
        ),
        (
            (&expected_irreps[4], &s6),
            Character::new(&[(UnityRoot::new(1, 6), 1)]),
        ),
        (
            (&expected_irreps[5], &s6),
            Character::new(&[(UnityRoot::new(5, 6), 1)]),
        ),
    ]);
    test_character_table_construction(&mol, thresh, &expected_irreps, Some(expected_chars));
}

#[test]
fn test_character_table_construction_symmetric_staggered_c2h6_magnetic_field_s6() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c2h6.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("||Γ|_(1g)|").unwrap(),
        MullikenIrrepSymbol::new("||Γ|_(2g)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(u)|").unwrap(),
        MullikenIrrepSymbol::new("||Γ|_(1u)|").unwrap(),
        MullikenIrrepSymbol::new("||Γ|_(2u)|").unwrap(),
    ];
    let c3 = ClassSymbol::<SymmetryOperation>::new("1||C3||", None).unwrap();
    let s6 = ClassSymbol::<SymmetryOperation>::new("1||S6||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &c3),
            Character::new(&[(UnityRoot::new(0, 3), 1)]),
        ),
        (
            (&expected_irreps[1], &c3),
            Character::new(&[(UnityRoot::new(1, 3), 1)]),
        ),
        (
            (&expected_irreps[2], &c3),
            Character::new(&[(UnityRoot::new(2, 3), 1)]),
        ),
        (
            (&expected_irreps[3], &c3),
            Character::new(&[(UnityRoot::new(0, 3), 1)]),
        ),
        (
            (&expected_irreps[4], &c3),
            Character::new(&[(UnityRoot::new(1, 3), 1)]),
        ),
        (
            (&expected_irreps[5], &c3),
            Character::new(&[(UnityRoot::new(2, 3), 1)]),
        ),
        (
            (&expected_irreps[0], &s6),
            Character::new(&[(UnityRoot::new(0, 6), 1)]),
        ),
        (
            (&expected_irreps[1], &s6),
            Character::new(&[(UnityRoot::new(4, 6), 1)]),
        ),
        (
            (&expected_irreps[2], &s6),
            Character::new(&[(UnityRoot::new(2, 6), 1)]),
        ),
        (
            (&expected_irreps[3], &s6),
            Character::new(&[(UnityRoot::new(3, 6), 1)]),
        ),
        (
            (&expected_irreps[4], &s6),
            Character::new(&[(UnityRoot::new(1, 6), 1)]),
        ),
        (
            (&expected_irreps[5], &s6),
            Character::new(&[(UnityRoot::new(5, 6), 1)]),
        ),
    ]);
    test_character_table_construction(&mol, thresh, &expected_irreps, Some(expected_chars));
}

#[test]
fn test_character_table_construction_symmetric_c60_magnetic_field_s6() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c60.xyz");
    let thresh = 1e-5;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(
        -0.5773503107731,
        -0.1875926572335,
        0.7946543988441,
    )));
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("||Γ|_(1g)|").unwrap(),
        MullikenIrrepSymbol::new("||Γ|_(2g)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(u)|").unwrap(),
        MullikenIrrepSymbol::new("||Γ|_(1u)|").unwrap(),
        MullikenIrrepSymbol::new("||Γ|_(2u)|").unwrap(),
    ];
    let c3 = ClassSymbol::<SymmetryOperation>::new("1||C3||", None).unwrap();
    let s6 = ClassSymbol::<SymmetryOperation>::new("1||S6||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &c3),
            Character::new(&[(UnityRoot::new(0, 3), 1)]),
        ),
        (
            (&expected_irreps[1], &c3),
            Character::new(&[(UnityRoot::new(1, 3), 1)]),
        ),
        (
            (&expected_irreps[2], &c3),
            Character::new(&[(UnityRoot::new(2, 3), 1)]),
        ),
        (
            (&expected_irreps[3], &c3),
            Character::new(&[(UnityRoot::new(0, 3), 1)]),
        ),
        (
            (&expected_irreps[4], &c3),
            Character::new(&[(UnityRoot::new(1, 3), 1)]),
        ),
        (
            (&expected_irreps[5], &c3),
            Character::new(&[(UnityRoot::new(2, 3), 1)]),
        ),
        (
            (&expected_irreps[0], &s6),
            Character::new(&[(UnityRoot::new(0, 6), 1)]),
        ),
        (
            (&expected_irreps[1], &s6),
            Character::new(&[(UnityRoot::new(4, 6), 1)]),
        ),
        (
            (&expected_irreps[2], &s6),
            Character::new(&[(UnityRoot::new(2, 6), 1)]),
        ),
        (
            (&expected_irreps[3], &s6),
            Character::new(&[(UnityRoot::new(3, 6), 1)]),
        ),
        (
            (&expected_irreps[4], &s6),
            Character::new(&[(UnityRoot::new(1, 6), 1)]),
        ),
        (
            (&expected_irreps[5], &s6),
            Character::new(&[(UnityRoot::new(5, 6), 1)]),
        ),
    ]);
    test_character_table_construction(&mol, thresh, &expected_irreps, Some(expected_chars));
}

#[test]
fn test_character_table_construction_symmetric_vh2o6_magnetic_field_s6() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/vh2o6.xyz");
    let thresh = 1e-12;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(-0.2, 0.2, 0.2)));
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("||Γ|_(1g)|").unwrap(),
        MullikenIrrepSymbol::new("||Γ|_(2g)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(u)|").unwrap(),
        MullikenIrrepSymbol::new("||Γ|_(1u)|").unwrap(),
        MullikenIrrepSymbol::new("||Γ|_(2u)|").unwrap(),
    ];
    let c3 = ClassSymbol::<SymmetryOperation>::new("1||C3||", None).unwrap();
    let s6 = ClassSymbol::<SymmetryOperation>::new("1||S6||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &c3),
            Character::new(&[(UnityRoot::new(0, 3), 1)]),
        ),
        (
            (&expected_irreps[1], &c3),
            Character::new(&[(UnityRoot::new(1, 3), 1)]),
        ),
        (
            (&expected_irreps[2], &c3),
            Character::new(&[(UnityRoot::new(2, 3), 1)]),
        ),
        (
            (&expected_irreps[3], &c3),
            Character::new(&[(UnityRoot::new(0, 3), 1)]),
        ),
        (
            (&expected_irreps[4], &c3),
            Character::new(&[(UnityRoot::new(1, 3), 1)]),
        ),
        (
            (&expected_irreps[5], &c3),
            Character::new(&[(UnityRoot::new(2, 3), 1)]),
        ),
        (
            (&expected_irreps[0], &s6),
            Character::new(&[(UnityRoot::new(0, 6), 1)]),
        ),
        (
            (&expected_irreps[1], &s6),
            Character::new(&[(UnityRoot::new(4, 6), 1)]),
        ),
        (
            (&expected_irreps[2], &s6),
            Character::new(&[(UnityRoot::new(2, 6), 1)]),
        ),
        (
            (&expected_irreps[3], &s6),
            Character::new(&[(UnityRoot::new(3, 6), 1)]),
        ),
        (
            (&expected_irreps[4], &s6),
            Character::new(&[(UnityRoot::new(1, 6), 1)]),
        ),
        (
            (&expected_irreps[5], &s6),
            Character::new(&[(UnityRoot::new(5, 6), 1)]),
        ),
    ]);
    test_character_table_construction(&mol, thresh, &expected_irreps, Some(expected_chars));
}

#[test]
fn test_character_table_construction_symmetric_vf6_magnetic_field_s6() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/vf6.xyz");
    let thresh = 1e-12;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(1.0, 1.0, 1.0)));
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("||Γ|_(1g)|").unwrap(),
        MullikenIrrepSymbol::new("||Γ|_(2g)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(u)|").unwrap(),
        MullikenIrrepSymbol::new("||Γ|_(1u)|").unwrap(),
        MullikenIrrepSymbol::new("||Γ|_(2u)|").unwrap(),
    ];
    let c3 = ClassSymbol::<SymmetryOperation>::new("1||C3||", None).unwrap();
    let s6 = ClassSymbol::<SymmetryOperation>::new("1||S6||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &c3),
            Character::new(&[(UnityRoot::new(0, 3), 1)]),
        ),
        (
            (&expected_irreps[1], &c3),
            Character::new(&[(UnityRoot::new(1, 3), 1)]),
        ),
        (
            (&expected_irreps[2], &c3),
            Character::new(&[(UnityRoot::new(2, 3), 1)]),
        ),
        (
            (&expected_irreps[3], &c3),
            Character::new(&[(UnityRoot::new(0, 3), 1)]),
        ),
        (
            (&expected_irreps[4], &c3),
            Character::new(&[(UnityRoot::new(1, 3), 1)]),
        ),
        (
            (&expected_irreps[5], &c3),
            Character::new(&[(UnityRoot::new(2, 3), 1)]),
        ),
        (
            (&expected_irreps[0], &s6),
            Character::new(&[(UnityRoot::new(0, 6), 1)]),
        ),
        (
            (&expected_irreps[1], &s6),
            Character::new(&[(UnityRoot::new(4, 6), 1)]),
        ),
        (
            (&expected_irreps[2], &s6),
            Character::new(&[(UnityRoot::new(2, 6), 1)]),
        ),
        (
            (&expected_irreps[3], &s6),
            Character::new(&[(UnityRoot::new(3, 6), 1)]),
        ),
        (
            (&expected_irreps[4], &s6),
            Character::new(&[(UnityRoot::new(1, 6), 1)]),
        ),
        (
            (&expected_irreps[5], &s6),
            Character::new(&[(UnityRoot::new(5, 6), 1)]),
        ),
    ]);
    test_character_table_construction(&mol, thresh, &expected_irreps, Some(expected_chars));
}

#[test]
fn test_character_table_construction_symmetric_s8_magnetic_field_s8() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/s8.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A||").unwrap(),
        MullikenIrrepSymbol::new("||Γ|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||Γ|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||Γ|_(3)|").unwrap(),
        MullikenIrrepSymbol::new("||B||").unwrap(),
        MullikenIrrepSymbol::new("||Γ|_(4)|").unwrap(),
        MullikenIrrepSymbol::new("||Γ|_(5)|").unwrap(),
        MullikenIrrepSymbol::new("||Γ|_(6)|").unwrap(),
    ];
    let s8 = ClassSymbol::<SymmetryOperation>::new("1||S8||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &s8),
            Character::new(&[(UnityRoot::new(0, 8), 1)]),
        ),
        (
            (&expected_irreps[1], &s8),
            Character::new(&[(UnityRoot::new(1, 8), 1)]),
        ),
        (
            (&expected_irreps[2], &s8),
            Character::new(&[(UnityRoot::new(2, 8), 1)]),
        ),
        (
            (&expected_irreps[3], &s8),
            Character::new(&[(UnityRoot::new(3, 8), 1)]),
        ),
        (
            (&expected_irreps[4], &s8),
            Character::new(&[(UnityRoot::new(4, 8), 1)]),
        ),
        (
            (&expected_irreps[5], &s8),
            Character::new(&[(UnityRoot::new(5, 8), 1)]),
        ),
        (
            (&expected_irreps[6], &s8),
            Character::new(&[(UnityRoot::new(6, 8), 1)]),
        ),
        (
            (&expected_irreps[7], &s8),
            Character::new(&[(UnityRoot::new(7, 8), 1)]),
        ),
    ]);
    test_character_table_construction(&mol, thresh, &expected_irreps, Some(expected_chars));
}

#[test]
fn test_character_table_construction_symmetric_antiprism_pb10_magnetic_field_s8() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/pb10.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A||").unwrap(),
        MullikenIrrepSymbol::new("||Γ|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||Γ|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||Γ|_(3)|").unwrap(),
        MullikenIrrepSymbol::new("||B||").unwrap(),
        MullikenIrrepSymbol::new("||Γ|_(4)|").unwrap(),
        MullikenIrrepSymbol::new("||Γ|_(5)|").unwrap(),
        MullikenIrrepSymbol::new("||Γ|_(6)|").unwrap(),
    ];
    let s8 = ClassSymbol::<SymmetryOperation>::new("1||S8||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &s8),
            Character::new(&[(UnityRoot::new(0, 8), 1)]),
        ),
        (
            (&expected_irreps[1], &s8),
            Character::new(&[(UnityRoot::new(1, 8), 1)]),
        ),
        (
            (&expected_irreps[2], &s8),
            Character::new(&[(UnityRoot::new(2, 8), 1)]),
        ),
        (
            (&expected_irreps[3], &s8),
            Character::new(&[(UnityRoot::new(3, 8), 1)]),
        ),
        (
            (&expected_irreps[4], &s8),
            Character::new(&[(UnityRoot::new(4, 8), 1)]),
        ),
        (
            (&expected_irreps[5], &s8),
            Character::new(&[(UnityRoot::new(5, 8), 1)]),
        ),
        (
            (&expected_irreps[6], &s8),
            Character::new(&[(UnityRoot::new(6, 8), 1)]),
        ),
        (
            (&expected_irreps[7], &s8),
            Character::new(&[(UnityRoot::new(7, 8), 1)]),
        ),
    ]);
    test_character_table_construction(&mol, thresh, &expected_irreps, Some(expected_chars));
}

#[test]
fn test_character_table_construction_symmetric_staggered_ferrocene_magnetic_field_s10() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/staggered_ferrocene.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("||Γ|_(1g)|").unwrap(),
        MullikenIrrepSymbol::new("||Γ|_(2g)|").unwrap(),
        MullikenIrrepSymbol::new("||Γ|_(3g)|").unwrap(),
        MullikenIrrepSymbol::new("||Γ|_(4g)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(u)|").unwrap(),
        MullikenIrrepSymbol::new("||Γ|_(1u)|").unwrap(),
        MullikenIrrepSymbol::new("||Γ|_(2u)|").unwrap(),
        MullikenIrrepSymbol::new("||Γ|_(3u)|").unwrap(),
        MullikenIrrepSymbol::new("||Γ|_(4u)|").unwrap(),
    ];
    let c5 = ClassSymbol::<SymmetryOperation>::new("1||C5||", None).unwrap();
    let s10 = ClassSymbol::<SymmetryOperation>::new("1||S10||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &c5),
            Character::new(&[(UnityRoot::new(0, 5), 1)]),
        ),
        (
            (&expected_irreps[1], &c5),
            Character::new(&[(UnityRoot::new(1, 5), 1)]),
        ),
        (
            (&expected_irreps[2], &c5),
            Character::new(&[(UnityRoot::new(2, 5), 1)]),
        ),
        (
            (&expected_irreps[3], &c5),
            Character::new(&[(UnityRoot::new(3, 5), 1)]),
        ),
        (
            (&expected_irreps[4], &c5),
            Character::new(&[(UnityRoot::new(4, 5), 1)]),
        ),
        (
            (&expected_irreps[5], &c5),
            Character::new(&[(UnityRoot::new(0, 5), 1)]),
        ),
        (
            (&expected_irreps[6], &c5),
            Character::new(&[(UnityRoot::new(1, 5), 1)]),
        ),
        (
            (&expected_irreps[7], &c5),
            Character::new(&[(UnityRoot::new(2, 5), 1)]),
        ),
        (
            (&expected_irreps[8], &c5),
            Character::new(&[(UnityRoot::new(3, 5), 1)]),
        ),
        (
            (&expected_irreps[9], &c5),
            Character::new(&[(UnityRoot::new(4, 5), 1)]),
        ),
        (
            (&expected_irreps[0], &s10),
            Character::new(&[(UnityRoot::new(0, 10), 1)]),
        ),
        (
            (&expected_irreps[1], &s10),
            Character::new(&[(UnityRoot::new(6, 10), 1)]),
        ),
        (
            (&expected_irreps[2], &s10),
            Character::new(&[(UnityRoot::new(2, 10), 1)]),
        ),
        (
            (&expected_irreps[3], &s10),
            Character::new(&[(UnityRoot::new(8, 10), 1)]),
        ),
        (
            (&expected_irreps[4], &s10),
            Character::new(&[(UnityRoot::new(4, 10), 1)]),
        ),
        (
            (&expected_irreps[5], &s10),
            Character::new(&[(UnityRoot::new(5, 10), 1)]),
        ),
        (
            (&expected_irreps[6], &s10),
            Character::new(&[(UnityRoot::new(1, 10), 1)]),
        ),
        (
            (&expected_irreps[7], &s10),
            Character::new(&[(UnityRoot::new(7, 10), 1)]),
        ),
        (
            (&expected_irreps[8], &s10),
            Character::new(&[(UnityRoot::new(3, 10), 1)]),
        ),
        (
            (&expected_irreps[9], &s10),
            Character::new(&[(UnityRoot::new(9, 10), 1)]),
        ),
    ]);
    test_character_table_construction(&mol, thresh, &expected_irreps, Some(expected_chars));
}

#[test]
fn test_character_table_construction_symmetric_c60_magnetic_field_s10() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c60.xyz");
    let thresh = 1e-5;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("||Γ|_(1g)|").unwrap(),
        MullikenIrrepSymbol::new("||Γ|_(2g)|").unwrap(),
        MullikenIrrepSymbol::new("||Γ|_(3g)|").unwrap(),
        MullikenIrrepSymbol::new("||Γ|_(4g)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(u)|").unwrap(),
        MullikenIrrepSymbol::new("||Γ|_(1u)|").unwrap(),
        MullikenIrrepSymbol::new("||Γ|_(2u)|").unwrap(),
        MullikenIrrepSymbol::new("||Γ|_(3u)|").unwrap(),
        MullikenIrrepSymbol::new("||Γ|_(4u)|").unwrap(),
    ];
    let c5 = ClassSymbol::<SymmetryOperation>::new("1||C5||", None).unwrap();
    let s10 = ClassSymbol::<SymmetryOperation>::new("1||S10||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &c5),
            Character::new(&[(UnityRoot::new(0, 5), 1)]),
        ),
        (
            (&expected_irreps[1], &c5),
            Character::new(&[(UnityRoot::new(1, 5), 1)]),
        ),
        (
            (&expected_irreps[2], &c5),
            Character::new(&[(UnityRoot::new(2, 5), 1)]),
        ),
        (
            (&expected_irreps[3], &c5),
            Character::new(&[(UnityRoot::new(3, 5), 1)]),
        ),
        (
            (&expected_irreps[4], &c5),
            Character::new(&[(UnityRoot::new(4, 5), 1)]),
        ),
        (
            (&expected_irreps[5], &c5),
            Character::new(&[(UnityRoot::new(0, 5), 1)]),
        ),
        (
            (&expected_irreps[6], &c5),
            Character::new(&[(UnityRoot::new(1, 5), 1)]),
        ),
        (
            (&expected_irreps[7], &c5),
            Character::new(&[(UnityRoot::new(2, 5), 1)]),
        ),
        (
            (&expected_irreps[8], &c5),
            Character::new(&[(UnityRoot::new(3, 5), 1)]),
        ),
        (
            (&expected_irreps[9], &c5),
            Character::new(&[(UnityRoot::new(4, 5), 1)]),
        ),
        (
            (&expected_irreps[0], &s10),
            Character::new(&[(UnityRoot::new(0, 10), 1)]),
        ),
        (
            (&expected_irreps[1], &s10),
            Character::new(&[(UnityRoot::new(6, 10), 1)]),
        ),
        (
            (&expected_irreps[2], &s10),
            Character::new(&[(UnityRoot::new(2, 10), 1)]),
        ),
        (
            (&expected_irreps[3], &s10),
            Character::new(&[(UnityRoot::new(8, 10), 1)]),
        ),
        (
            (&expected_irreps[4], &s10),
            Character::new(&[(UnityRoot::new(4, 10), 1)]),
        ),
        (
            (&expected_irreps[5], &s10),
            Character::new(&[(UnityRoot::new(5, 10), 1)]),
        ),
        (
            (&expected_irreps[6], &s10),
            Character::new(&[(UnityRoot::new(1, 10), 1)]),
        ),
        (
            (&expected_irreps[7], &s10),
            Character::new(&[(UnityRoot::new(7, 10), 1)]),
        ),
        (
            (&expected_irreps[8], &s10),
            Character::new(&[(UnityRoot::new(3, 10), 1)]),
        ),
        (
            (&expected_irreps[9], &s10),
            Character::new(&[(UnityRoot::new(9, 10), 1)]),
        ),
    ]);
    test_character_table_construction(&mol, thresh, &expected_irreps, Some(expected_chars));
}

#[test]
fn test_character_table_construction_symmetric_au26_magnetic_field_s12() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/au26.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A||").unwrap(),
        MullikenIrrepSymbol::new("||Γ|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||Γ|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||Γ|_(3)|").unwrap(),
        MullikenIrrepSymbol::new("||Γ|_(4)|").unwrap(),
        MullikenIrrepSymbol::new("||Γ|_(5)|").unwrap(),
        MullikenIrrepSymbol::new("||B||").unwrap(),
        MullikenIrrepSymbol::new("||Γ|_(6)|").unwrap(),
        MullikenIrrepSymbol::new("||Γ|_(7)|").unwrap(),
        MullikenIrrepSymbol::new("||Γ|_(8)|").unwrap(),
        MullikenIrrepSymbol::new("||Γ|_(9)|").unwrap(),
        MullikenIrrepSymbol::new("||Γ|_(10)|").unwrap(),
    ];
    let s12 = ClassSymbol::<SymmetryOperation>::new("1||S12||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &s12),
            Character::new(&[(UnityRoot::new(0, 12), 1)]),
        ),
        (
            (&expected_irreps[1], &s12),
            Character::new(&[(UnityRoot::new(1, 12), 1)]),
        ),
        (
            (&expected_irreps[2], &s12),
            Character::new(&[(UnityRoot::new(2, 12), 1)]),
        ),
        (
            (&expected_irreps[3], &s12),
            Character::new(&[(UnityRoot::new(3, 12), 1)]),
        ),
        (
            (&expected_irreps[4], &s12),
            Character::new(&[(UnityRoot::new(4, 12), 1)]),
        ),
        (
            (&expected_irreps[5], &s12),
            Character::new(&[(UnityRoot::new(5, 12), 1)]),
        ),
        (
            (&expected_irreps[6], &s12),
            Character::new(&[(UnityRoot::new(6, 12), 1)]),
        ),
        (
            (&expected_irreps[7], &s12),
            Character::new(&[(UnityRoot::new(7, 12), 1)]),
        ),
        (
            (&expected_irreps[8], &s12),
            Character::new(&[(UnityRoot::new(8, 12), 1)]),
        ),
        (
            (&expected_irreps[9], &s12),
            Character::new(&[(UnityRoot::new(9, 12), 1)]),
        ),
        (
            (&expected_irreps[10], &s12),
            Character::new(&[(UnityRoot::new(10, 12), 1)]),
        ),
        (
            (&expected_irreps[11], &s12),
            Character::new(&[(UnityRoot::new(11, 12), 1)]),
        ),
    ]);
    test_character_table_construction(&mol, thresh, &expected_irreps, Some(expected_chars));
}

#[test]
fn test_character_table_construction_symmetric_arbitrary_staggered_sandwich_magnetic_field_s2n() {
    let thresh = 1e-7;
    for n in 3..=20 {
        let mut mol = template_molecules::gen_arbitrary_twisted_sandwich(n, 0.5);
        mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
        let expected_irreps = if n % 2 != 0 {
            // Odd n, g/u possible.
            let mut irreps_gu = vec![];
            for parity in ["g", "u"] {
                let mut irreps =
                    vec![MullikenIrrepSymbol::new(&format!("||A|_({parity})|")).unwrap()];
                irreps.extend(
                    (1..n).map(|k| {
                        MullikenIrrepSymbol::new(&format!("||Γ|_({k}{parity})|")).unwrap()
                    }),
                );
                irreps_gu.extend(irreps)
            }
            irreps_gu
        } else {
            // Even n, no g/u possible.
            let mut irreps = vec![MullikenIrrepSymbol::new("||A||").unwrap()];
            irreps
                .extend((1..n).map(|k| MullikenIrrepSymbol::new(&format!("||Γ|_({k})|")).unwrap()));
            irreps.push(MullikenIrrepSymbol::new("||B||").unwrap());
            irreps.extend(
                (n..(2 * n - 1))
                    .map(|k| MullikenIrrepSymbol::new(&format!("||Γ|_({k})|")).unwrap()),
            );
            irreps
        };
        test_character_table_construction(&mol, thresh, &expected_irreps, None);
    }
}

/*********
Asymmetric
*********/
/*
C2
*/
#[test]
fn test_character_table_construction_asymmetric_spiroketal_c2() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/spiroketal.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A||").unwrap(),
        MullikenIrrepSymbol::new("||B||").unwrap(),
    ];
    let c2 = ClassSymbol::<SymmetryOperation>::new("1||C2||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &c2),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[1], &c2),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
    ]);
    test_character_table_construction(&mol, thresh, &expected_irreps, Some(expected_chars));
}

#[test]
fn test_character_table_construction_asymmetric_cyclohexene_c2() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/cyclohexene.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A||").unwrap(),
        MullikenIrrepSymbol::new("||B||").unwrap(),
    ];
    let c2 = ClassSymbol::<SymmetryOperation>::new("1||C2||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &c2),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[1], &c2),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
    ]);
    test_character_table_construction(&mol, thresh, &expected_irreps, Some(expected_chars));
}

#[test]
fn test_character_table_construction_asymmetric_thf_c2() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/thf.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A||").unwrap(),
        MullikenIrrepSymbol::new("||B||").unwrap(),
    ];
    let c2 = ClassSymbol::<SymmetryOperation>::new("1||C2||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &c2),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[1], &c2),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
    ]);
    test_character_table_construction(&mol, thresh, &expected_irreps, Some(expected_chars));
}

#[test]
fn test_character_table_construction_asymmetric_tartaricacid_c2() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/tartaricacid.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A||").unwrap(),
        MullikenIrrepSymbol::new("||B||").unwrap(),
    ];
    let c2 = ClassSymbol::<SymmetryOperation>::new("1||C2||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &c2),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[1], &c2),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
    ]);
    test_character_table_construction(&mol, thresh, &expected_irreps, Some(expected_chars));
}

#[test]
fn test_character_table_construction_asymmetric_f2allene_c2() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/f2allene.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A||").unwrap(),
        MullikenIrrepSymbol::new("||B||").unwrap(),
    ];
    let c2 = ClassSymbol::<SymmetryOperation>::new("1||C2||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &c2),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[1], &c2),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
    ]);
    test_character_table_construction(&mol, thresh, &expected_irreps, Some(expected_chars));
}

#[test]
fn test_character_table_construction_asymmetric_water_magnetic_field_c2() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/water.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 1.0, 0.0)));
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A||").unwrap(),
        MullikenIrrepSymbol::new("||B||").unwrap(),
    ];
    let c2 = ClassSymbol::<SymmetryOperation>::new("1||C2||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &c2),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[1], &c2),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
    ]);
    test_character_table_construction(&mol, thresh, &expected_irreps, Some(expected_chars));
}

#[test]
fn test_character_table_construction_asymmetric_pyridine_magnetic_field_c2() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/pyridine.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.2, 0.0)));
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A||").unwrap(),
        MullikenIrrepSymbol::new("||B||").unwrap(),
    ];
    let c2 = ClassSymbol::<SymmetryOperation>::new("1||C2||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &c2),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[1], &c2),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
    ]);
    test_character_table_construction(&mol, thresh, &expected_irreps, Some(expected_chars));
}

#[test]
fn test_character_table_construction_asymmetric_cyclobutene_magnetic_field_c2() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/cyclobutene.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.2, 0.0, 0.0)));
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A||").unwrap(),
        MullikenIrrepSymbol::new("||B||").unwrap(),
    ];
    let c2 = ClassSymbol::<SymmetryOperation>::new("1||C2||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &c2),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[1], &c2),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
    ]);
    test_character_table_construction(&mol, thresh, &expected_irreps, Some(expected_chars));
}

#[test]
fn test_character_table_construction_asymmetric_azulene_magnetic_field_c2() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/azulene.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 0.2)));
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A||").unwrap(),
        MullikenIrrepSymbol::new("||B||").unwrap(),
    ];
    let c2 = ClassSymbol::<SymmetryOperation>::new("1||C2||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &c2),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[1], &c2),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
    ]);
    test_character_table_construction(&mol, thresh, &expected_irreps, Some(expected_chars));
}

#[test]
fn test_character_table_construction_asymmetric_cis_cocl2h4o2_magnetic_field_c2() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/cis-cocl2h4o2.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.2, 0.0)));
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A||").unwrap(),
        MullikenIrrepSymbol::new("||B||").unwrap(),
    ];
    let c2 = ClassSymbol::<SymmetryOperation>::new("1||C2||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &c2),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[1], &c2),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
    ]);
    test_character_table_construction(&mol, thresh, &expected_irreps, Some(expected_chars));
}

#[test]
fn test_character_table_construction_asymmetric_cuneane_magnetic_field_c2() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/cuneane.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 0.2)));
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A||").unwrap(),
        MullikenIrrepSymbol::new("||B||").unwrap(),
    ];
    let c2 = ClassSymbol::<SymmetryOperation>::new("1||C2||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &c2),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[1], &c2),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
    ]);
    test_character_table_construction(&mol, thresh, &expected_irreps, Some(expected_chars));
}

/***
C2v
***/
#[test]
fn test_character_table_construction_asymmetric_water_c2v() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/water.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(2)|").unwrap(),
    ];
    let c2 = ClassSymbol::<SymmetryOperation>::new("1||C2||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &c2),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[1], &c2),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[2], &c2),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[2], &c2),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
    ]);
    test_character_table_construction(&mol, thresh, &expected_irreps, Some(expected_chars));
}

#[test]
fn test_character_table_construction_asymmetric_pyridine_c2v() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/pyridine.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(2)|").unwrap(),
    ];
    let c2 = ClassSymbol::<SymmetryOperation>::new("1||C2||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &c2),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[1], &c2),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[2], &c2),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[2], &c2),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
    ]);
    test_character_table_construction(&mol, thresh, &expected_irreps, Some(expected_chars));
}

#[test]
fn test_character_table_construction_asymmetric_cyclobutene_c2v() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/cyclobutene.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(2)|").unwrap(),
    ];
    let c2 = ClassSymbol::<SymmetryOperation>::new("1||C2||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &c2),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[1], &c2),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[2], &c2),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[2], &c2),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
    ]);
    test_character_table_construction(&mol, thresh, &expected_irreps, Some(expected_chars));
}

#[test]
fn test_character_table_construction_asymmetric_azulene_c2v() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/azulene.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(2)|").unwrap(),
    ];
    let c2 = ClassSymbol::<SymmetryOperation>::new("1||C2||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &c2),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[1], &c2),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[2], &c2),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[2], &c2),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
    ]);
    test_character_table_construction(&mol, thresh, &expected_irreps, Some(expected_chars));
}

#[test]
fn test_character_table_construction_asymmetric_cuneane_c2v() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/cuneane.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(2)|").unwrap(),
    ];
    let c2 = ClassSymbol::<SymmetryOperation>::new("1||C2||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &c2),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[1], &c2),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[2], &c2),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[2], &c2),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
    ]);
    test_character_table_construction(&mol, thresh, &expected_irreps, Some(expected_chars));
}

#[test]
fn test_character_table_construction_asymmetric_bf3_electric_field_c2v() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/bf3.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(Vector3::new(0.2, 0.0, 0.0)));
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(2)|").unwrap(),
    ];
    let c2 = ClassSymbol::<SymmetryOperation>::new("1||C2||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &c2),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[1], &c2),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[2], &c2),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[2], &c2),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
    ]);
    test_character_table_construction(&mol, thresh, &expected_irreps, Some(expected_chars));
}

/***
C2h
***/
#[test]
fn test_character_table_construction_asymmetric_h2o2_c2h() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/h2o2.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(u)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(u)|").unwrap(),
    ];
    let c2 = ClassSymbol::<SymmetryOperation>::new("1||C2||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &c2),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[1], &c2),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[2], &c2),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[3], &c2),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
    ]);
    test_character_table_construction(&mol, thresh, &expected_irreps, Some(expected_chars));
}

#[test]
fn test_character_table_construction_asymmetric_zethrene_c2h() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/zethrene.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(u)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(u)|").unwrap(),
    ];
    let c2 = ClassSymbol::<SymmetryOperation>::new("1||C2||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &c2),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[1], &c2),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[2], &c2),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[3], &c2),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
    ]);
    test_character_table_construction(&mol, thresh, &expected_irreps, Some(expected_chars));
}

#[test]
fn test_character_table_construction_asymmetric_distorted_vf6_magnetic_field_c2h() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/vf6_d2h.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(1.0, 0.0, 0.0)));
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(u)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(u)|").unwrap(),
    ];
    let c2 = ClassSymbol::<SymmetryOperation>::new("1||C2||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &c2),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[1], &c2),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[2], &c2),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[3], &c2),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
    ]);
    test_character_table_construction(&mol, thresh, &expected_irreps, Some(expected_chars));
}

#[test]
fn test_character_table_construction_asymmetric_b2h6_magnetic_field_c2h() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/b2h6.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(1.0, 0.0, 0.0)));
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(u)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(u)|").unwrap(),
    ];
    let c2 = ClassSymbol::<SymmetryOperation>::new("1||C2||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &c2),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[1], &c2),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[2], &c2),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[3], &c2),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
    ]);
    test_character_table_construction(&mol, thresh, &expected_irreps, Some(expected_chars));
}

#[test]
fn test_character_table_construction_asymmetric_naphthalene_magnetic_field_c2h() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/naphthalene.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 1.0, 0.0)));
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(u)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(u)|").unwrap(),
    ];
    let c2 = ClassSymbol::<SymmetryOperation>::new("1||C2||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &c2),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[1], &c2),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[2], &c2),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[3], &c2),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
    ]);
    test_character_table_construction(&mol, thresh, &expected_irreps, Some(expected_chars));
}

#[test]
fn test_character_table_construction_asymmetric_pyrene_magnetic_field_c2h() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/pyrene.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(u)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(u)|").unwrap(),
    ];
    let c2 = ClassSymbol::<SymmetryOperation>::new("1||C2||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &c2),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[1], &c2),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[2], &c2),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[3], &c2),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
    ]);
    test_character_table_construction(&mol, thresh, &expected_irreps, Some(expected_chars));
}

#[test]
fn test_character_table_construction_asymmetric_c6o6_magnetic_field_c2h() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c6o6.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(u)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(u)|").unwrap(),
    ];
    let c2 = ClassSymbol::<SymmetryOperation>::new("1||C2||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &c2),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[1], &c2),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[2], &c2),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[3], &c2),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
    ]);
    test_character_table_construction(&mol, thresh, &expected_irreps, Some(expected_chars));
}

/*
Cs
*/
#[test]
fn test_character_table_construction_asymmetric_propene_cs() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/propene.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|^(')|").unwrap(),
        MullikenIrrepSymbol::new("||A|^('')|").unwrap(),
    ];
    let s = ClassSymbol::<SymmetryOperation>::new("1||σh||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &s),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[1], &s),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
    ]);
    test_character_table_construction(&mol, thresh, &expected_irreps, Some(expected_chars));
}

#[test]
fn test_character_table_construction_asymmetric_socl2_cs() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/socl2.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|^(')|").unwrap(),
        MullikenIrrepSymbol::new("||A|^('')|").unwrap(),
    ];
    let s = ClassSymbol::<SymmetryOperation>::new("1||σh||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &s),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[1], &s),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
    ]);
    test_character_table_construction(&mol, thresh, &expected_irreps, Some(expected_chars));
}

#[test]
fn test_character_table_construction_asymmetric_hocl_cs() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/hocl.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|^(')|").unwrap(),
        MullikenIrrepSymbol::new("||A|^('')|").unwrap(),
    ];
    let s = ClassSymbol::<SymmetryOperation>::new("1||σh||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &s),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[1], &s),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
    ]);
    test_character_table_construction(&mol, thresh, &expected_irreps, Some(expected_chars));
}

#[test]
fn test_character_table_construction_asymmetric_hocn_cs() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/hocn.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|^(')|").unwrap(),
        MullikenIrrepSymbol::new("||A|^('')|").unwrap(),
    ];
    let s = ClassSymbol::<SymmetryOperation>::new("1||σh||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &s),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[1], &s),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
    ]);
    test_character_table_construction(&mol, thresh, &expected_irreps, Some(expected_chars));
}

#[test]
fn test_character_table_construction_asymmetric_nh2f_cs() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/nh2f.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|^(')|").unwrap(),
        MullikenIrrepSymbol::new("||A|^('')|").unwrap(),
    ];
    let s = ClassSymbol::<SymmetryOperation>::new("1||σh||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &s),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[1], &s),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
    ]);
    test_character_table_construction(&mol, thresh, &expected_irreps, Some(expected_chars));
}

#[test]
fn test_character_table_construction_asymmetric_phenol_cs() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/phenol.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|^(')|").unwrap(),
        MullikenIrrepSymbol::new("||A|^('')|").unwrap(),
    ];
    let s = ClassSymbol::<SymmetryOperation>::new("1||σh||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &s),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[1], &s),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
    ]);
    test_character_table_construction(&mol, thresh, &expected_irreps, Some(expected_chars));
}

#[test]
fn test_character_table_construction_asymmetric_f_pyrrole_cs() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/f-pyrrole.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|^(')|").unwrap(),
        MullikenIrrepSymbol::new("||A|^('')|").unwrap(),
    ];
    let s = ClassSymbol::<SymmetryOperation>::new("1||σh||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &s),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[1], &s),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
    ]);
    test_character_table_construction(&mol, thresh, &expected_irreps, Some(expected_chars));
}

#[test]
fn test_character_table_construction_asymmetric_n2o_cs() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/n2o.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|^(')|").unwrap(),
        MullikenIrrepSymbol::new("||A|^('')|").unwrap(),
    ];
    let s = ClassSymbol::<SymmetryOperation>::new("1||σh||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &s),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[1], &s),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
    ]);
    test_character_table_construction(&mol, thresh, &expected_irreps, Some(expected_chars));
}

#[test]
fn test_character_table_construction_asymmetric_fclbenzene_cs() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/fclbenzene.xyz");
    let thresh = 1e-5;
    let mol = Molecule::from_xyz(&path, thresh);
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|^(')|").unwrap(),
        MullikenIrrepSymbol::new("||A|^('')|").unwrap(),
    ];
    let s = ClassSymbol::<SymmetryOperation>::new("1||σh||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &s),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[1], &s),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
    ]);
    test_character_table_construction(&mol, thresh, &expected_irreps, Some(expected_chars));
}

#[test]
fn test_character_table_construction_asymmetric_water_magnetic_field_cs() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/water.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(1.0, 0.0, 0.0)));
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|^(')|").unwrap(),
        MullikenIrrepSymbol::new("||A|^('')|").unwrap(),
    ];
    let s = ClassSymbol::<SymmetryOperation>::new("1||σh||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &s),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[1], &s),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
    ]);
    test_character_table_construction(&mol, thresh, &expected_irreps, Some(expected_chars));
}

#[test]
fn test_character_table_construction_asymmetric_pyridine_magnetic_field_cs() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/pyridine.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 0.2)));
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|^(')|").unwrap(),
        MullikenIrrepSymbol::new("||A|^('')|").unwrap(),
    ];
    let s = ClassSymbol::<SymmetryOperation>::new("1||σh||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &s),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[1], &s),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
    ]);
    test_character_table_construction(&mol, thresh, &expected_irreps, Some(expected_chars));
}

#[test]
fn test_character_table_construction_asymmetric_cyclobutene_magnetic_field_cs() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/cyclobutene.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.1, 0.0)));
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|^(')|").unwrap(),
        MullikenIrrepSymbol::new("||A|^('')|").unwrap(),
    ];
    let s = ClassSymbol::<SymmetryOperation>::new("1||σh||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &s),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[1], &s),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
    ]);
    test_character_table_construction(&mol, thresh, &expected_irreps, Some(expected_chars));
}

#[test]
fn test_character_table_construction_asymmetric_azulene_magnetic_field_cs() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/azulene.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.1, 0.0)));
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|^(')|").unwrap(),
        MullikenIrrepSymbol::new("||A|^('')|").unwrap(),
    ];
    let s = ClassSymbol::<SymmetryOperation>::new("1||σh||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &s),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[1], &s),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
    ]);
    test_character_table_construction(&mol, thresh, &expected_irreps, Some(expected_chars));
}

#[test]
fn test_character_table_construction_asymmetric_cis_cocl2h4o2_magnetic_field_cs() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/cis-cocl2h4o2.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 0.2)));
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|^(')|").unwrap(),
        MullikenIrrepSymbol::new("||A|^('')|").unwrap(),
    ];
    let s = ClassSymbol::<SymmetryOperation>::new("1||σh||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &s),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[1], &s),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
    ]);
    test_character_table_construction(&mol, thresh, &expected_irreps, Some(expected_chars));
}

#[test]
fn test_character_table_construction_asymmetric_cuneane_magnetic_field_cs() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/cuneane.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.5, 0.0)));
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|^(')|").unwrap(),
        MullikenIrrepSymbol::new("||A|^('')|").unwrap(),
    ];
    let s = ClassSymbol::<SymmetryOperation>::new("1||σh||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &s),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[1], &s),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
    ]);
    test_character_table_construction(&mol, thresh, &expected_irreps, Some(expected_chars));
}

#[test]
fn test_character_table_construction_asymmetric_water_electric_field_cs() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/water.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(Vector3::new(1.0, 0.0, 0.0)));
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|^(')|").unwrap(),
        MullikenIrrepSymbol::new("||A|^('')|").unwrap(),
    ];
    let s = ClassSymbol::<SymmetryOperation>::new("1||σh||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &s),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[1], &s),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
    ]);
    test_character_table_construction(&mol, thresh, &expected_irreps, Some(expected_chars));
}

#[test]
fn test_character_table_construction_asymmetric_pyridine_electric_field_cs() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/pyridine.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(Vector3::new(0.0, 0.0, 0.2)));
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|^(')|").unwrap(),
        MullikenIrrepSymbol::new("||A|^('')|").unwrap(),
    ];
    let s = ClassSymbol::<SymmetryOperation>::new("1||σh||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &s),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[1], &s),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
    ]);
    test_character_table_construction(&mol, thresh, &expected_irreps, Some(expected_chars));
}

#[test]
fn test_character_table_construction_asymmetric_cyclobutene_electric_field_cs() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/cyclobutene.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(Vector3::new(0.0, 0.1, 0.0)));
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|^(')|").unwrap(),
        MullikenIrrepSymbol::new("||A|^('')|").unwrap(),
    ];
    let s = ClassSymbol::<SymmetryOperation>::new("1||σh||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &s),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[1], &s),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
    ]);
    test_character_table_construction(&mol, thresh, &expected_irreps, Some(expected_chars));
}

#[test]
fn test_character_table_construction_asymmetric_azulene_electric_field_cs() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/azulene.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(Vector3::new(0.0, 0.1, 0.0)));
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|^(')|").unwrap(),
        MullikenIrrepSymbol::new("||A|^('')|").unwrap(),
    ];
    let s = ClassSymbol::<SymmetryOperation>::new("1||σh||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &s),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[1], &s),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
    ]);
    test_character_table_construction(&mol, thresh, &expected_irreps, Some(expected_chars));
}

#[test]
fn test_character_table_construction_asymmetric_cis_cocl2h4o2_electric_field_cs() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/cis-cocl2h4o2.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(Vector3::new(0.0, 0.0, 0.2)));
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|^(')|").unwrap(),
        MullikenIrrepSymbol::new("||A|^('')|").unwrap(),
    ];
    let s = ClassSymbol::<SymmetryOperation>::new("1||σh||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &s),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[1], &s),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
    ]);
    test_character_table_construction(&mol, thresh, &expected_irreps, Some(expected_chars));
}

#[test]
fn test_character_table_construction_asymmetric_cuneane_electric_field_cs() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/cuneane.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(Vector3::new(0.0, 0.5, 0.0)));
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|^(')|").unwrap(),
        MullikenIrrepSymbol::new("||A|^('')|").unwrap(),
    ];
    let s = ClassSymbol::<SymmetryOperation>::new("1||σh||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &s),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[1], &s),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
    ]);
    test_character_table_construction(&mol, thresh, &expected_irreps, Some(expected_chars));
}

#[test]
fn test_character_table_construction_asymmetric_bf3_magnetic_field_cs() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/bf3.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.5, 0.0)));
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|^(')|").unwrap(),
        MullikenIrrepSymbol::new("||A|^('')|").unwrap(),
    ];
    let s = ClassSymbol::<SymmetryOperation>::new("1||σh||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &s),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[1], &s),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
    ]);
    test_character_table_construction(&mol, thresh, &expected_irreps, Some(expected_chars));
}

/// This is a special case: Cs point group in a symmetric top.
#[test]
fn test_character_table_construction_symmetric_ch4_magnetic_field_cs() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/ch4.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.1, 0.1, 0.0)));
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|^(')|").unwrap(),
        MullikenIrrepSymbol::new("||A|^('')|").unwrap(),
    ];
    let s = ClassSymbol::<SymmetryOperation>::new("1||σh||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &s),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[1], &s),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
    ]);
    test_character_table_construction(&mol, thresh, &expected_irreps, Some(expected_chars));
}

/// This is another special case: Cs point group in a symmetric top.
#[test]
fn test_character_table_construction_symmetric_ch4_electric_field_cs() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/ch4.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(Vector3::new(0.1, 0.1, 0.0)));
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|^(')|").unwrap(),
        MullikenIrrepSymbol::new("||A|^('')|").unwrap(),
    ];
    let s = ClassSymbol::<SymmetryOperation>::new("1||σh||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &s),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[1], &s),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
    ]);
    test_character_table_construction(&mol, thresh, &expected_irreps, Some(expected_chars));
}

#[test]
fn test_character_table_construction_asymmetric_atom_magnetic_electric_field_cs() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/th.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 0.2)));
    mol.set_electric_field(Some(Vector3::new(0.0, 0.1, 0.0)));
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|^(')|").unwrap(),
        MullikenIrrepSymbol::new("||A|^('')|").unwrap(),
    ];
    let s = ClassSymbol::<SymmetryOperation>::new("1||σh||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &s),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[1], &s),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
    ]);
    test_character_table_construction(&mol, thresh, &expected_irreps, Some(expected_chars));
}

/*
D2
*/
#[test]
fn test_character_table_construction_asymmetric_i4_biphenyl_d2() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/i4-biphenyl.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A||").unwrap(),
        MullikenIrrepSymbol::new("||B|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(3)|").unwrap(),
    ];
    let c2 = ClassSymbol::<SymmetryOperation>::new("1||C2||", None).unwrap();
    let c2d = ClassSymbol::<SymmetryOperation>::new("1||C2|^(')|", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &c2),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[1], &c2),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[2], &c2),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[3], &c2),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[0], &c2d),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[1], &c2d),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[2], &c2d),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[3], &c2d),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
    ]);
    test_character_table_construction(&mol, thresh, &expected_irreps, Some(expected_chars));
}

#[test]
fn test_abstrainto_ct_group_asymmetric_twistane_d2() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/twistane.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A||").unwrap(),
        MullikenIrrepSymbol::new("||B|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(3)|").unwrap(),
    ];
    let c2 = ClassSymbol::<SymmetryOperation>::new("1||C2||", None).unwrap();
    let c2d = ClassSymbol::<SymmetryOperation>::new("1||C2|^(')|", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &c2),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[1], &c2),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[2], &c2),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[3], &c2),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[0], &c2d),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[1], &c2d),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[2], &c2d),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[3], &c2d),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
    ]);
    test_character_table_construction(&mol, thresh, &expected_irreps, Some(expected_chars));
}

#[test]
fn test_character_table_construction_asymmetric_22_paracyclophane_d2() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/paracyclophane22.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A||").unwrap(),
        MullikenIrrepSymbol::new("||B|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(3)|").unwrap(),
    ];
    let c2 = ClassSymbol::<SymmetryOperation>::new("1||C2||", None).unwrap();
    let c2d = ClassSymbol::<SymmetryOperation>::new("1||C2|^(')|", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &c2),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[1], &c2),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[2], &c2),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[3], &c2),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[0], &c2d),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[1], &c2d),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[2], &c2d),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[3], &c2d),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
    ]);
    test_character_table_construction(&mol, thresh, &expected_irreps, Some(expected_chars));
}

/***
D2h
***/
#[test]
fn test_character_table_construction_asymmetric_b2h6_d2h() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/b2h6.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(1g)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(2g)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(3g)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(u)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(1u)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(2u)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(3u)|").unwrap(),
    ];
    let c2 = ClassSymbol::<SymmetryOperation>::new("1||C2||", None).unwrap();
    let c2d = ClassSymbol::<SymmetryOperation>::new("1||C2|^(')|", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &c2),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[1], &c2),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[2], &c2),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[3], &c2),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[4], &c2),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[5], &c2),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[6], &c2),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[7], &c2),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[0], &c2d),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[1], &c2d),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[2], &c2d),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[3], &c2d),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[4], &c2d),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[5], &c2d),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[6], &c2d),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[7], &c2d),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
    ]);
    test_character_table_construction(&mol, thresh, &expected_irreps, Some(expected_chars));
}

#[test]
fn test_character_table_construction_asymmetric_naphthalene_d2h() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/naphthalene.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(1g)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(2g)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(3g)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(u)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(1u)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(2u)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(3u)|").unwrap(),
    ];
    let c2 = ClassSymbol::<SymmetryOperation>::new("1||C2||", None).unwrap();
    let c2d = ClassSymbol::<SymmetryOperation>::new("1||C2|^(')|", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &c2),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[1], &c2),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[2], &c2),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[3], &c2),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[4], &c2),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[5], &c2),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[6], &c2),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[7], &c2),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[0], &c2d),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[1], &c2d),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[2], &c2d),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[3], &c2d),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[4], &c2d),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[5], &c2d),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[6], &c2d),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[7], &c2d),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
    ]);
    test_character_table_construction(&mol, thresh, &expected_irreps, Some(expected_chars));
}

#[test]
fn test_character_table_construction_asymmetric_pyrene_d2h() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/pyrene.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(1g)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(2g)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(3g)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(u)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(1u)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(2u)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(3u)|").unwrap(),
    ];
    let c2 = ClassSymbol::<SymmetryOperation>::new("1||C2||", None).unwrap();
    let c2d = ClassSymbol::<SymmetryOperation>::new("1||C2|^(')|", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &c2),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[1], &c2),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[2], &c2),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[3], &c2),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[4], &c2),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[5], &c2),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[6], &c2),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[7], &c2),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[0], &c2d),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[1], &c2d),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[2], &c2d),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[3], &c2d),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[4], &c2d),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[5], &c2d),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[6], &c2d),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[7], &c2d),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
    ]);
    test_character_table_construction(&mol, thresh, &expected_irreps, Some(expected_chars));
}

#[test]
fn test_character_table_construction_asymmetric_c6o6_d2h() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c6o6.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(1g)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(2g)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(3g)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(u)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(1u)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(2u)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(3u)|").unwrap(),
    ];
    let c2 = ClassSymbol::<SymmetryOperation>::new("1||C2||", None).unwrap();
    let c2d = ClassSymbol::<SymmetryOperation>::new("1||C2|^(')|", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &c2),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[1], &c2),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[2], &c2),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[3], &c2),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[4], &c2),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[5], &c2),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[6], &c2),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[7], &c2),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[0], &c2d),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[1], &c2d),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[2], &c2d),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[3], &c2d),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[4], &c2d),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[5], &c2d),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[6], &c2d),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[7], &c2d),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
    ]);
    test_character_table_construction(&mol, thresh, &expected_irreps, Some(expected_chars));
}

#[test]
fn test_character_table_construction_asymmetric_distorted_vf6_d2h() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/vf6_d2h.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(1g)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(2g)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(3g)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(u)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(1u)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(2u)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(3u)|").unwrap(),
    ];
    let c2 = ClassSymbol::<SymmetryOperation>::new("1||C2||", None).unwrap();
    let c2d = ClassSymbol::<SymmetryOperation>::new("1||C2|^(')|", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &c2),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[1], &c2),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[2], &c2),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[3], &c2),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[4], &c2),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[5], &c2),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[6], &c2),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[7], &c2),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[0], &c2d),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[1], &c2d),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[2], &c2d),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[3], &c2d),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[4], &c2d),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[5], &c2d),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[6], &c2d),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[7], &c2d),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
    ]);
    test_character_table_construction(&mol, thresh, &expected_irreps, Some(expected_chars));
}

/***
Ci
***/
#[test]
fn test_character_table_construction_asymmetric_meso_tartaricacid_ci() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/meso-tartaricacid.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(u)|").unwrap(),
    ];
    let i = ClassSymbol::<SymmetryOperation>::new("1||i||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &i),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[1], &i),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
    ]);
    test_character_table_construction(&mol, thresh, &expected_irreps, Some(expected_chars));
}

#[test]
fn test_character_table_construction_asymmetric_dibromodimethylcyclohexane_ci() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/dibromodimethylcyclohexane.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(u)|").unwrap(),
    ];
    let i = ClassSymbol::<SymmetryOperation>::new("1||i||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &i),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[1], &i),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
    ]);
    test_character_table_construction(&mol, thresh, &expected_irreps, Some(expected_chars));
}

#[test]
fn test_character_table_construction_asymmetric_h2o2_magnetic_field_ci() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/h2o2.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(1.0, 2.0, -1.0)));
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(u)|").unwrap(),
    ];
    let i = ClassSymbol::<SymmetryOperation>::new("1||i||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &i),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[1], &i),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
    ]);
    test_character_table_construction(&mol, thresh, &expected_irreps, Some(expected_chars));
}

#[test]
fn test_character_table_construction_symmetric_xef4_magnetic_field_ci() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/xef4.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(1.0, 2.0, -2.0)));
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(u)|").unwrap(),
    ];
    let i = ClassSymbol::<SymmetryOperation>::new("1||i||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &i),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[1], &i),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
    ]);
    test_character_table_construction(&mol, thresh, &expected_irreps, Some(expected_chars));
}

#[test]
fn test_character_table_construction_asymmetric_c2h2_magnetic_field_ci() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c2h2.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(1.0, -1.0, 1.0)));
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(u)|").unwrap(),
    ];
    let i = ClassSymbol::<SymmetryOperation>::new("1||i||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &i),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[1], &i),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
    ]);
    test_character_table_construction(&mol, thresh, &expected_irreps, Some(expected_chars));
}

/***
C1
***/
#[test]
fn test_character_table_construction_asymmetric_butan1ol_c1() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/butan-1-ol.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    let expected_irreps = vec![MullikenIrrepSymbol::new("||A||").unwrap()];
    let e = ClassSymbol::<SymmetryOperation>::new("1||E||", None).unwrap();
    let expected_chars = HashMap::from([(
        (&expected_irreps[0], &e),
        Character::new(&[(UnityRoot::new(0, 1), 1)]),
    )]);
    test_character_table_construction(&mol, thresh, &expected_irreps, Some(expected_chars));
}

#[test]
fn test_character_table_construction_asymmetric_subst_5m_ring_c1() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/subst-5m-ring.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    let expected_irreps = vec![MullikenIrrepSymbol::new("||A||").unwrap()];
    let e = ClassSymbol::<SymmetryOperation>::new("1||E||", None).unwrap();
    let expected_chars = HashMap::from([(
        (&expected_irreps[0], &e),
        Character::new(&[(UnityRoot::new(0, 1), 1)]),
    )]);
    test_character_table_construction(&mol, thresh, &expected_irreps, Some(expected_chars));
}

#[test]
fn test_character_table_construction_asymmetric_bf3_magnetic_field_c1() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/bf3.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(1.0, -1.0, 1.0)));
    let expected_irreps = vec![MullikenIrrepSymbol::new("||A||").unwrap()];
    let e = ClassSymbol::<SymmetryOperation>::new("1||E||", None).unwrap();
    let expected_chars = HashMap::from([(
        (&expected_irreps[0], &e),
        Character::new(&[(UnityRoot::new(0, 1), 1)]),
    )]);
    test_character_table_construction(&mol, thresh, &expected_irreps, Some(expected_chars));
}
