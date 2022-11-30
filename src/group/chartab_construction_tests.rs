use std::collections::HashMap;

use approx;
use env_logger;
use itertools::Itertools;
use nalgebra::Vector3;
use num::Complex;

use crate::aux::molecule::Molecule;
use crate::aux::template_molecules;
use crate::chartab::character::Character;
use crate::chartab::unityroot::UnityRoot;
use crate::group::group_from_molecular_symmetry;
use crate::symmetry::symmetry_core::{PreSymmetry, Symmetry};
use crate::symmetry::symmetry_element::SymmetryOperation;
use crate::symmetry::symmetry_symbols::{ClassSymbol, MathematicalSymbol, MullikenIrrepSymbol};

const ROOT: &str = env!("CARGO_MANIFEST_DIR");

// ================================================================
// Character table for abstract group from molecular symmetry tests
// ================================================================

fn test_character_table_validity(
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
    let mut sym = Symmetry::builder().build().unwrap();
    sym.analyse(&presym);
    let group = group_from_molecular_symmetry(sym, None);
    let chartab = group.character_table.as_ref().unwrap();
    println!("{:?}", chartab);
    // println!("{}", chartab);

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
    assert_eq!(chartab.irreps.keys().cloned().collect_vec(), expected_irreps);

    // Expected characters
    if let Some(expected_chars) = expected_chars_option {
        for ((irrep, cc), ref_char) in expected_chars.iter() {
            println!("Comparing {:?} with {:?}...", chartab.get_character(irrep, cc), ref_char);
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

// fn test_character_table_construction_from_infinite_group(
//     mol: &Molecule,
//     finite_order: u32,
//     thresh: f64,
//     name: &str,
//     finite_name: &str,
//     order: usize,
//     class_number: usize,
//     abelian: bool,
// ) {
//     let presym = PreSymmetry::builder()
//         .moi_threshold(thresh)
//         .molecule(mol, true)
//         .build()
//         .unwrap();
//     let mut sym = Symmetry::builder().build().unwrap();
//     sym.analyse(&presym);
//     let group = group_from_molecular_symmetry(sym, Some(finite_order));
//     test_character_table_construction_validity(group, name, order, class_number, abelian);
// }

// /********
// Spherical
// ********/
// #[test]
// fn test_character_table_construction_spherical_atom_o3() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/th.xyz");
//     let thresh = 1e-7;
//     let mol = Molecule::from_xyz(&path, thresh);

//     test_character_table_construction_from_infinite_group(&mol, 2, thresh, "O(3)", "D2h", 8, 8, true);

//     test_character_table_construction_from_infinite_group(&mol, 4, thresh, "O(3)", "Oh", 48, 10, false);

//     let result = panic::catch_unwind(|| {
//         test_character_table_construction_from_infinite_group(&mol, 5, thresh, "?", "?", 48, 10, false);
//     });
//     assert!(result.is_err());

//     let result = panic::catch_unwind(|| {
//         test_character_table_construction_from_infinite_group(&mol, 3, thresh, "?", "?", 48, 10, false);
//     });
//     assert!(result.is_err());
// }

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
            Character::new(&[
                (UnityRoot::new(0, 10), 1),
            ]),
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
            Character::new(&[
                (UnityRoot::new(5, 10), 1),
            ]),
        ),
        (
            (&expected_irreps[4], &c5),
            Character::new(&[
                (UnityRoot::new(0, 10), 1),
                (UnityRoot::new(5, 10), 1),
            ]),
        ),
        (
            (&expected_irreps[5], &c5),
            Character::new(&[
                (UnityRoot::new(0, 10), 1),
            ]),
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
            Character::new(&[
                (UnityRoot::new(5, 10), 1),
            ]),
        ),
        (
            (&expected_irreps[9], &c5),
            Character::new(&[
                (UnityRoot::new(0, 10), 1),
                (UnityRoot::new(5, 10), 1),
            ]),
        ),
    ]);
    test_character_table_validity(&mol, thresh, &expected_irreps, Some(expected_chars));
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
            Character::new(&[
                (UnityRoot::new(0, 3), 1),
            ]),
        ),
        (
            (&expected_irreps[1], &c3),
            Character::new(&[
                (UnityRoot::new(0, 3), 1),
            ]),
        ),
        (
            (&expected_irreps[2], &c3),
            Character::new(&[
                (UnityRoot::new(1, 3), 1),
                (UnityRoot::new(2, 3), 1),
            ]),
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
    test_character_table_validity(&mol, thresh, &expected_irreps, Some(expected_chars));
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
            Character::new(&[
                (UnityRoot::new(0, 3), 1),
            ]),
        ),
        (
            (&expected_irreps[1], &c3),
            Character::new(&[
                (UnityRoot::new(0, 3), 1),
            ]),
        ),
        (
            (&expected_irreps[2], &c3),
            Character::new(&[
                (UnityRoot::new(1, 3), 1),
                (UnityRoot::new(2, 3), 1),
            ]),
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
    test_character_table_validity(&mol, thresh, &expected_irreps, Some(expected_chars));
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
            Character::new(&[
                (UnityRoot::new(0, 3), 1),
            ]),
        ),
        (
            (&expected_irreps[1], &c3),
            Character::new(&[
                (UnityRoot::new(0, 3), 1),
            ]),
        ),
        (
            (&expected_irreps[2], &c3),
            Character::new(&[
                (UnityRoot::new(1, 3), 1),
                (UnityRoot::new(2, 3), 1),
            ]),
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
    test_character_table_validity(&mol, thresh, &expected_irreps, Some(expected_chars));
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
            Character::new(&[
                (UnityRoot::new(0, 3), 1),
            ]),
        ),
        (
            (&expected_irreps[1], &c3),
            Character::new(&[
                (UnityRoot::new(1, 3), 1),
            ]),
        ),
        (
            (&expected_irreps[2], &c3),
            Character::new(&[
                (UnityRoot::new(2, 3), 1),
            ]),
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
            ]),
        ),
        (
            (&expected_irreps[5], &c3),
            Character::new(&[
                (UnityRoot::new(1, 3), 1),
            ]),
        ),
        (
            (&expected_irreps[6], &c3),
            Character::new(&[
                (UnityRoot::new(2, 3), 1),
            ]),
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
    test_character_table_validity(&mol, thresh, &expected_irreps, Some(expected_chars));
}

#[test]
fn test_character_table_construction_spherical_vf6_oh() {
    env_logger::init();
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
            Character::new(&[
                (UnityRoot::new(0, 3), 1),
            ]),
        ),
        (
            (&expected_irreps[1], &c3),
            Character::new(&[
                (UnityRoot::new(0, 3), 1),
            ]),
        ),
        (
            (&expected_irreps[2], &c3),
            Character::new(&[
                (UnityRoot::new(1, 3), 1),
                (UnityRoot::new(2, 3), 1),
            ]),
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
            Character::new(&[
                (UnityRoot::new(0, 3), 1),
            ]),
        ),
        (
            (&expected_irreps[6], &c3),
            Character::new(&[
                (UnityRoot::new(0, 3), 1),
            ]),
        ),
        (
            (&expected_irreps[7], &c3),
            Character::new(&[
                (UnityRoot::new(1, 3), 1),
                (UnityRoot::new(2, 3), 1),
            ]),
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
    test_character_table_validity(&mol, thresh, &expected_irreps, Some(expected_chars));
}

// /*****
// Linear
// *****/
// #[test]
// fn test_character_table_construction_linear_atom_magnetic_field_cinfh() {
//     // env_logger::init();
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/th.xyz");
//     let thresh = 1e-7;
//     let mut mol = Molecule::from_xyz(&path, thresh);
//     mol.set_magnetic_field(Some(Vector3::new(1.0, 2.0, -1.0)));
//     for n in 2usize..=20usize {
//         if n % 2 == 0 {
//             test_character_table_construction_from_infinite_group(
//                 &mol,
//                 n as u32,
//                 thresh,
//                 "C∞h",
//                 format!("C{}h", n).as_str(),
//                 2 * n,
//                 2 * n,
//                 true,
//             );
//         } else {
//             test_character_table_construction_from_infinite_group(
//                 &mol,
//                 n as u32,
//                 thresh,
//                 "C∞h",
//                 format!("C{}h", 2 * n).as_str(),
//                 4 * n,
//                 4 * n,
//                 true,
//             );
//         }
//     }
// }

// #[test]
// fn test_character_table_construction_linear_atom_electric_field_cinfv() {
//     /* The expected number of classes is deduced from the irrep structures of
//      * the Cnv groups.
//      * When n is even, the irreps are A1, A2, B1, B2, Ek where k = 1, ..., n/2 - 1.
//      * When n is odd, the irreps are A1, A2, Ek where k = 1, ..., n//2.
//      */
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/th.xyz");
//     let thresh = 1e-7;
//     let mut mol = Molecule::from_xyz(&path, thresh);
//     mol.set_electric_field(Some(Vector3::new(-1.0, 3.0, -2.0)));
//     for n in 3usize..=20usize {
//         test_character_table_construction_from_infinite_group(
//             &mol,
//             n as u32,
//             thresh,
//             "C∞v",
//             format!("C{}v", n).as_str(),
//             2 * n,
//             ({
//                 if n % 2 == 0 {
//                     n / 2 - 1
//                 } else {
//                     n / 2
//                 }
//             } + {
//                 if n % 2 == 0 {
//                     4
//                 } else {
//                     2
//                 }
//             }) as usize,
//             false,
//         );
//     }
// }

// #[test]
// fn test_character_table_construction_linear_c2h2_dinfh() {
//     /* The expected number of classes is deduced from the irrep structures of
//      * the Dnh groups.
//      * When n is even, the irreps are A1(g/u), A2(g/u), B1(g/u), B2(g/u), Ek(g/u)
//      * where k = 1, ..., n/2 - 1.
//      * When n is odd, the irreps are A1('/''), A2('/''), Ek('/'')
//      * where k = 1, ..., n//2.
//      */
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/c2h2.xyz");
//     let thresh = 1e-6;
//     let mol = Molecule::from_xyz(&path, thresh);
//     for n in 3usize..=20usize {
//         if n % 2 == 0 {
//             test_character_table_construction_from_infinite_group(
//                 &mol,
//                 n as u32,
//                 thresh,
//                 "D∞h",
//                 format!("D{}h", n).as_str(),
//                 4 * n,
//                 2 * (n / 2 - 1 + 4) as usize,
//                 false,
//             );
//         } else {
//             test_character_table_construction_from_infinite_group(
//                 &mol,
//                 n as u32,
//                 thresh,
//                 "D∞h",
//                 format!("D{}h", 2 * n).as_str(),
//                 8 * n,
//                 2 * (n - 1 + 4) as usize,
//                 false,
//             );
//         }
//     }
// }

// #[test]
// fn test_character_table_construction_linear_c2h2_magnetic_field_cinfh() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/c2h2.xyz");
//     let thresh = 1e-6;
//     let mut mol = Molecule::from_xyz(&path, thresh);
//     // Parallel field
//     mol.set_magnetic_field(Some(Vector3::new(1.0, 1.0, 1.0)));
//     for n in 2usize..=20usize {
//         if n % 2 == 0 {
//             test_character_table_construction_from_infinite_group(
//                 &mol,
//                 n as u32,
//                 thresh,
//                 "C∞h",
//                 format!("C{}h", n).as_str(),
//                 2 * n,
//                 2 * n as usize,
//                 true,
//             );
//         } else {
//             test_character_table_construction_from_infinite_group(
//                 &mol,
//                 n as u32,
//                 thresh,
//                 "C∞h",
//                 format!("C{}h", 2 * n).as_str(),
//                 4 * n,
//                 4 * n as usize,
//                 true,
//             );
//         }
//     }
// }

// #[test]
// fn test_character_table_construction_linear_c2h2_electric_field_cinfv() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/c2h2.xyz");
//     let thresh = 1e-6;
//     let mut mol = Molecule::from_xyz(&path, thresh);

//     // Parallel field
//     mol.set_electric_field(Some(Vector3::new(1.0, 1.0, 1.0)));
//     for n in 3usize..=20usize {
//         test_character_table_construction_from_infinite_group(
//             &mol,
//             n as u32,
//             thresh,
//             "C∞v",
//             format!("C{}v", n).as_str(),
//             2 * n,
//             ({
//                 if n % 2 == 0 {
//                     n / 2 - 1
//                 } else {
//                     n / 2
//                 }
//             } + {
//                 if n % 2 == 0 {
//                     4
//                 } else {
//                     2
//                 }
//             }) as usize,
//             false,
//         );
//     }
// }

// #[test]
// fn test_character_table_construction_linear_n3_cinfv() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/n3.xyz");
//     let thresh = 1e-6;
//     let mol = Molecule::from_xyz(&path, thresh);
//     for n in 3usize..=20usize {
//         test_character_table_construction_from_infinite_group(
//             &mol,
//             n as u32,
//             thresh,
//             "C∞v",
//             format!("C{}v", n).as_str(),
//             2 * n,
//             ({
//                 if n % 2 == 0 {
//                     n / 2 - 1
//                 } else {
//                     n / 2
//                 }
//             } + {
//                 if n % 2 == 0 {
//                     4
//                 } else {
//                     2
//                 }
//             }) as usize,
//             false,
//         );
//     }
// }

// #[test]
// fn test_character_table_construction_linear_n3_magnetic_field_cinf() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/n3.xyz");
//     let thresh = 1e-6;
//     let mut mol = Molecule::from_xyz(&path, thresh);

//     // Parallel field
//     mol.set_magnetic_field(Some(Vector3::new(1.0, 1.0, 1.0)));
//     for n in 2usize..=20usize {
//         test_character_table_construction_from_infinite_group(
//             &mol,
//             n as u32,
//             thresh,
//             "C∞",
//             format!("C{}", n).as_str(),
//             n,
//             n,
//             true,
//         );
//     }
// }

// #[test]
// fn test_character_table_construction_linear_n3_electric_field_cinfv() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/n3.xyz");
//     let thresh = 1e-6;
//     let mut mol = Molecule::from_xyz(&path, thresh);

//     // Parallel field
//     mol.set_electric_field(Some(Vector3::new(1.0, 1.0, 1.0)));
//     for n in 3usize..=20usize {
//         test_character_table_construction_from_infinite_group(
//             &mol,
//             n as u32,
//             thresh,
//             "C∞v",
//             format!("C{}v", n).as_str(),
//             2 * n,
//             ({
//                 if n % 2 == 0 {
//                     n / 2 - 1
//                 } else {
//                     n / 2
//                 }
//             } + {
//                 if n % 2 == 0 {
//                     4
//                 } else {
//                     2
//                 }
//             }) as usize,
//             false,
//         );
//     }
// }

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
            Character::new(&[
                (UnityRoot::new(0, 3), 1),
            ]),
        ),
        (
            (&expected_irreps[1], &c3),
            Character::new(&[
                (UnityRoot::new(1, 3), 1),
            ]),
        ),
        (
            (&expected_irreps[2], &c3),
            Character::new(&[
                (UnityRoot::new(2, 3), 1),
            ]),
        ),
    ]);
    test_character_table_validity(&mol, thresh, &expected_irreps, Some(expected_chars));
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
            Character::new(&[
                (UnityRoot::new(0, 3), 1),
            ]),
        ),
        (
            (&expected_irreps[1], &c3),
            Character::new(&[
                (UnityRoot::new(1, 3), 1),
            ]),
        ),
        (
            (&expected_irreps[2], &c3),
            Character::new(&[
                (UnityRoot::new(2, 3), 1),
            ]),
        ),
    ]);
    test_character_table_validity(&mol, thresh, &expected_irreps, Some(expected_chars));
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
            Character::new(&[
                (UnityRoot::new(0, 3), 1),
            ]),
        ),
        (
            (&expected_irreps[1], &c3),
            Character::new(&[
                (UnityRoot::new(1, 3), 1),
            ]),
        ),
        (
            (&expected_irreps[2], &c3),
            Character::new(&[
                (UnityRoot::new(2, 3), 1),
            ]),
        ),
    ]);
    test_character_table_validity(&mol, thresh, &expected_irreps, Some(expected_chars));
}

#[test]
fn test_character_table_construction_symmetric_65coronane_electric_field_c3() {
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
            Character::new(&[
                (UnityRoot::new(0, 3), 1),
            ]),
        ),
        (
            (&expected_irreps[1], &c3),
            Character::new(&[
                (UnityRoot::new(1, 3), 1),
            ]),
        ),
        (
            (&expected_irreps[2], &c3),
            Character::new(&[
                (UnityRoot::new(2, 3), 1),
            ]),
        ),
    ]);
    test_character_table_validity(&mol, thresh, &expected_irreps, Some(expected_chars));
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
            Character::new(&[
                (UnityRoot::new(0, 4), 1),
            ]),
        ),
        (
            (&expected_irreps[1], &c4),
            Character::new(&[
                (UnityRoot::new(1, 4), 1),
            ]),
        ),
        (
            (&expected_irreps[2], &c4),
            Character::new(&[
                (UnityRoot::new(2, 4), 1),
            ]),
        ),
        (
            (&expected_irreps[3], &c4),
            Character::new(&[
                (UnityRoot::new(3, 4), 1),
            ]),
        ),
    ]);
    test_character_table_validity(&mol, thresh, &expected_irreps, Some(expected_chars));
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
            Character::new(&[
                (UnityRoot::new(0, 4), 1),
            ]),
        ),
        (
            (&expected_irreps[1], &c4),
            Character::new(&[
                (UnityRoot::new(1, 4), 1),
            ]),
        ),
        (
            (&expected_irreps[2], &c4),
            Character::new(&[
                (UnityRoot::new(2, 4), 1),
            ]),
        ),
        (
            (&expected_irreps[3], &c4),
            Character::new(&[
                (UnityRoot::new(3, 4), 1),
            ]),
        ),
    ]);
    test_character_table_validity(&mol, thresh, &expected_irreps, Some(expected_chars));
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
            Character::new(&[
                (UnityRoot::new(0, 5), 1),
            ]),
        ),
        (
            (&expected_irreps[1], &c5),
            Character::new(&[
                (UnityRoot::new(1, 5), 1),
            ]),
        ),
        (
            (&expected_irreps[2], &c5),
            Character::new(&[
                (UnityRoot::new(2, 5), 1),
            ]),
        ),
        (
            (&expected_irreps[3], &c5),
            Character::new(&[
                (UnityRoot::new(3, 5), 1),
            ]),
        ),
        (
            (&expected_irreps[4], &c5),
            Character::new(&[
                (UnityRoot::new(4, 5), 1),
            ]),
        ),
    ]);
    test_character_table_validity(&mol, thresh, &expected_irreps, Some(expected_chars));
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
            Character::new(&[
                (UnityRoot::new(0, 6), 1),
            ]),
        ),
        (
            (&expected_irreps[1], &c6),
            Character::new(&[
                (UnityRoot::new(1, 6), 1),
            ]),
        ),
        (
            (&expected_irreps[2], &c6),
            Character::new(&[
                (UnityRoot::new(2, 6), 1),
            ]),
        ),
        (
            (&expected_irreps[3], &c6),
            Character::new(&[
                (UnityRoot::new(3, 6), 1),
            ]),
        ),
        (
            (&expected_irreps[4], &c6),
            Character::new(&[
                (UnityRoot::new(4, 6), 1),
            ]),
        ),
        (
            (&expected_irreps[5], &c6),
            Character::new(&[
                (UnityRoot::new(5, 6), 1),
            ]),
        ),
    ]);
    test_character_table_validity(&mol, thresh, &expected_irreps, Some(expected_chars));
}

#[test]
fn test_character_table_construction_symmetric_arbitrary_half_sandwich_magnetic_field_cn() {
    let thresh = 1e-7;
    for n in 3..=32 {
        let mut mol = template_molecules::gen_arbitrary_half_sandwich(n);
        mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 0.1)));
        let expected_irreps = if n % 2 == 0 {
            let mut irreps = vec![MullikenIrrepSymbol::new("||A||").unwrap()];
            irreps.extend((1..(n / 2)).map(|i| {
                MullikenIrrepSymbol::new(&format!("||Γ|_({})|", i)).unwrap()
            }));
            irreps.push(MullikenIrrepSymbol::new("||B||").unwrap());
            irreps.extend(((n/2)..(n-1)).map(|i| {
                MullikenIrrepSymbol::new(&format!("||Γ|_({})|", i)).unwrap()
            }));
            irreps
        } else {
            let mut irreps = vec![MullikenIrrepSymbol::new("||A||").unwrap()];
            irreps.extend((1..n).map(|i| {
                MullikenIrrepSymbol::new(&format!("||Γ|_({})|", i)).unwrap()
            }));
            irreps
        };
        let cn = ClassSymbol::<SymmetryOperation>::new(&format!("1||C{}||", n), None).unwrap();
        let expected_chars: HashMap<_, _> = expected_irreps.iter().enumerate().map(|(i, irrep)| {
            (
                (irrep, &cn),
                Character::new(&[
                    (UnityRoot::new(i as u64, n as u64), 1),
                ]),
            )
        }).collect();
        test_character_table_validity(&mol, thresh, &expected_irreps, Some(expected_chars));
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
            Character::new(&[
                (UnityRoot::new(0, 3), 1),
            ]),
        ),
        (
            (&expected_irreps[1], &c3),
            Character::new(&[
                (UnityRoot::new(0, 3), 1),
            ]),
        ),
        (
            (&expected_irreps[2], &c3),
            Character::new(&[
                (UnityRoot::new(1, 3), 1),
                (UnityRoot::new(2, 3), 1),
            ]),
        ),
    ]);
    test_character_table_validity(&mol, thresh, &expected_irreps, Some(expected_chars));
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
            Character::new(&[
                (UnityRoot::new(0, 3), 1),
            ]),
        ),
        (
            (&expected_irreps[1], &c3),
            Character::new(&[
                (UnityRoot::new(0, 3), 1),
            ]),
        ),
        (
            (&expected_irreps[2], &c3),
            Character::new(&[
                (UnityRoot::new(1, 3), 1),
                (UnityRoot::new(2, 3), 1),
            ]),
        ),
    ]);
    test_character_table_validity(&mol, thresh, &expected_irreps, Some(expected_chars));
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
            Character::new(&[
                (UnityRoot::new(0, 3), 1),
            ]),
        ),
        (
            (&expected_irreps[1], &c3),
            Character::new(&[
                (UnityRoot::new(0, 3), 1),
            ]),
        ),
        (
            (&expected_irreps[2], &c3),
            Character::new(&[
                (UnityRoot::new(1, 3), 1),
                (UnityRoot::new(2, 3), 1),
            ]),
        ),
    ]);
    test_character_table_validity(&mol, thresh, &expected_irreps, Some(expected_chars));
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
            Character::new(&[
                (UnityRoot::new(0, 3), 1),
            ]),
        ),
        (
            (&expected_irreps[1], &c3),
            Character::new(&[
                (UnityRoot::new(0, 3), 1),
            ]),
        ),
        (
            (&expected_irreps[2], &c3),
            Character::new(&[
                (UnityRoot::new(1, 3), 1),
                (UnityRoot::new(2, 3), 1),
            ]),
        ),
    ]);
    test_character_table_validity(&mol, thresh, &expected_irreps, Some(expected_chars));
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
            Character::new(&[
                (UnityRoot::new(0, 3), 1),
            ]),
        ),
        (
            (&expected_irreps[1], &c3),
            Character::new(&[
                (UnityRoot::new(0, 3), 1),
            ]),
        ),
        (
            (&expected_irreps[2], &c3),
            Character::new(&[
                (UnityRoot::new(1, 3), 1),
                (UnityRoot::new(2, 3), 1),
            ]),
        ),
    ]);
    test_character_table_validity(&mol, thresh, &expected_irreps, Some(expected_chars));
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
            Character::new(&[
                (UnityRoot::new(0, 4), 1),
            ]),
        ),
        (
            (&expected_irreps[1], &c4),
            Character::new(&[
                (UnityRoot::new(0, 4), 1),
            ]),
        ),
        (
            (&expected_irreps[2], &c4),
            Character::new(&[
                (UnityRoot::new(2, 4), 1),
            ]),
        ),
        (
            (&expected_irreps[3], &c4),
            Character::new(&[
                (UnityRoot::new(2, 4), 1),
            ]),
        ),
        (
            (&expected_irreps[4], &c4),
            Character::new(&[
                (UnityRoot::new(1, 4), 1),
                (UnityRoot::new(3, 4), 1),
            ]),
        ),
    ]);
    test_character_table_validity(&mol, thresh, &expected_irreps, Some(expected_chars));
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
            Character::new(&[
                (UnityRoot::new(0, 4), 1),
            ]),
        ),
        (
            (&expected_irreps[1], &c4),
            Character::new(&[
                (UnityRoot::new(0, 4), 1),
            ]),
        ),
        (
            (&expected_irreps[2], &c4),
            Character::new(&[
                (UnityRoot::new(2, 4), 1),
            ]),
        ),
        (
            (&expected_irreps[3], &c4),
            Character::new(&[
                (UnityRoot::new(2, 4), 1),
            ]),
        ),
        (
            (&expected_irreps[4], &c4),
            Character::new(&[
                (UnityRoot::new(1, 4), 1),
                (UnityRoot::new(3, 4), 1),
            ]),
        ),
    ]);
    test_character_table_validity(&mol, thresh, &expected_irreps, Some(expected_chars));
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
            Character::new(&[
                (UnityRoot::new(0, 4), 1),
            ]),
        ),
        (
            (&expected_irreps[1], &c4),
            Character::new(&[
                (UnityRoot::new(0, 4), 1),
            ]),
        ),
        (
            (&expected_irreps[2], &c4),
            Character::new(&[
                (UnityRoot::new(2, 4), 1),
            ]),
        ),
        (
            (&expected_irreps[3], &c4),
            Character::new(&[
                (UnityRoot::new(2, 4), 1),
            ]),
        ),
        (
            (&expected_irreps[4], &c4),
            Character::new(&[
                (UnityRoot::new(1, 4), 1),
                (UnityRoot::new(3, 4), 1),
            ]),
        ),
    ]);
    test_character_table_validity(&mol, thresh, &expected_irreps, Some(expected_chars));
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
            Character::new(&[
                (UnityRoot::new(0, 4), 1),
            ]),
        ),
        (
            (&expected_irreps[1], &c4),
            Character::new(&[
                (UnityRoot::new(0, 4), 1),
            ]),
        ),
        (
            (&expected_irreps[2], &c4),
            Character::new(&[
                (UnityRoot::new(2, 4), 1),
            ]),
        ),
        (
            (&expected_irreps[3], &c4),
            Character::new(&[
                (UnityRoot::new(2, 4), 1),
            ]),
        ),
        (
            (&expected_irreps[4], &c4),
            Character::new(&[
                (UnityRoot::new(1, 4), 1),
                (UnityRoot::new(3, 4), 1),
            ]),
        ),
    ]);
    test_character_table_validity(&mol, thresh, &expected_irreps, Some(expected_chars));
}

// #[test]
// fn test_character_table_construction_symmetric_cpnico_c5v() {
//     // env_logger::init();
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/cpnico.xyz");
//     let thresh = 1e-6;
//     let mol = Molecule::from_xyz(&path, thresh);
//     test_character_table_construction(&mol, thresh, "C5v", 10, 4, false);
// }

// #[test]
// fn test_character_table_construction_symmetric_cpnico_c5v_class_order() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/cpnico.xyz");
//     let thresh = 1e-6;
//     let mol = Molecule::from_xyz(&path, thresh);
//     test_character_table_construction_class_order(&mol, thresh, &["|E|", "2|C5|", "2|[C5]^2|", "5|σv|"]);
// }

// #[test]
// fn test_character_table_construction_symmetric_staggered_ferrocene_electric_field_c5v() {
//     // env_logger::init();
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/staggered_ferrocene.xyz");
//     let thresh = 1e-6;
//     let mut mol = Molecule::from_xyz(&path, thresh);
//     mol.set_electric_field(Some(Vector3::new(0.0, 0.0, 1.0)));
//     test_character_table_construction(&mol, thresh, "C5v", 10, 4, false);
// }

// #[test]
// fn test_character_table_construction_symmetric_staggered_ferrocene_electric_field_c5v_class_order() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/staggered_ferrocene.xyz");
//     let thresh = 1e-6;
//     let mut mol = Molecule::from_xyz(&path, thresh);
//     mol.set_electric_field(Some(Vector3::new(0.0, 0.0, 1.0)));
//     test_character_table_construction_class_order(&mol, thresh, &["|E|", "2|C5|", "2|[C5]^2|", "5|σv|"]);
// }

// #[test]
// fn test_character_table_construction_symmetric_c60_electric_field_c5v() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/c60.xyz");
//     let thresh = 1e-6;
//     let mut mol = Molecule::from_xyz(&path, thresh);
//     mol.set_electric_field(Some(Vector3::new(0.0, 0.0, 1.0)));
//     test_character_table_construction(&mol, thresh, "C5v", 10, 4, false);
// }

// #[test]
// fn test_character_table_construction_symmetric_c60_electric_field_c5v_class_order() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/c60.xyz");
//     let thresh = 1e-6;
//     let mut mol = Molecule::from_xyz(&path, thresh);
//     mol.set_electric_field(Some(Vector3::new(0.0, 0.0, 1.0)));
//     test_character_table_construction_class_order(&mol, thresh, &["|E|", "2|C5|", "2|[C5]^2|", "5|σv|"]);
// }

// #[test]
// fn test_character_table_construction_symmetric_b7_c6v() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/b7.xyz");
//     let thresh = 1e-6;
//     let mol = Molecule::from_xyz(&path, thresh);
//     test_character_table_construction(&mol, thresh, "C6v", 12, 6, false);
// }

// #[test]
// fn test_character_table_construction_symmetric_b7_c6v_class_order() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/b7.xyz");
//     let thresh = 1e-6;
//     let mol = Molecule::from_xyz(&path, thresh);
//     test_character_table_construction_class_order(
//         &mol,
//         thresh,
//         &["|E|", "2|C6|", "2|C3|", "|C2|", "3|σv|", "3|σv|^(')"],
//     );
// }

// #[test]
// fn test_character_table_construction_symmetric_au26_electric_field_c6v() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/au26.xyz");
//     let thresh = 1e-6;
//     let mut mol = Molecule::from_xyz(&path, thresh);
//     mol.set_electric_field(Some(Vector3::new(0.0, 0.0, 1.0)));
//     test_character_table_construction(&mol, thresh, "C6v", 12, 6, false);
// }

// #[test]
// fn test_character_table_construction_symmetric_au26_electric_field_c6v_class_order() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/au26.xyz");
//     let thresh = 1e-6;
//     let mut mol = Molecule::from_xyz(&path, thresh);
//     mol.set_electric_field(Some(Vector3::new(0.0, 0.0, 1.0)));
//     test_character_table_construction_class_order(
//         &mol,
//         thresh,
//         &["|E|", "2|C6|", "2|C3|", "|C2|", "3|σv|", "3|σv|^(')"],
//     );
// }

// #[test]
// fn test_character_table_construction_symmetric_benzene_electric_field_c6v() {
//     // env_logger::init();
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/benzene.xyz");
//     let thresh = 1e-7;
//     let mut mol = Molecule::from_xyz(&path, thresh);
//     mol.set_electric_field(Some(Vector3::new(1.0, 0.0, 0.0)));
//     test_character_table_construction(&mol, thresh, "C6v", 12, 6, false);
// }

// #[test]
// fn test_character_table_construction_symmetric_benzene_electric_field_c6v_class_order() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/benzene.xyz");
//     let thresh = 1e-7;
//     let mut mol = Molecule::from_xyz(&path, thresh);
//     mol.set_electric_field(Some(Vector3::new(1.0, 0.0, 0.0)));
//     test_character_table_construction_class_order(
//         &mol,
//         thresh,
//         &["|E|", "2|C6|", "2|C3|", "|C2|", "3|σv|", "3|σv|^(')"],
//     );
// }

// #[test]
// fn test_character_table_construction_symmetric_arbitrary_half_sandwich_cnv() {
//     /* The expected number of classes is deduced from the irrep structures of
//      * the Cnv groups.
//      * When n is even, the irreps are A1, A2, B1, B2, Ek where k = 1, ..., n/2 - 1.
//      * When n is odd, the irreps are A1, A2, Ek where k = 1, ..., n//2.
//      */
//     for n in 3..=32 {
//         let mol = template_molecules::gen_arbitrary_half_sandwich(n);
//         let thresh = 1e-7;
//         test_character_table_construction(
//             &mol,
//             thresh,
//             format!("C{}v", n).as_str(),
//             2 * n as usize,
//             ({
//                 if n % 2 == 0 {
//                     n / 2 - 1
//                 } else {
//                     n / 2
//                 }
//             } + {
//                 if n % 2 == 0 {
//                     4
//                 } else {
//                     2
//                 }
//             }) as usize,
//             false,
//         );
//     }
// }

// #[test]
// fn test_character_table_construction_symmetric_arbitrary_staggered_sandwich_electric_field_cnv() {
//     /* The expected number of classes is deduced from the irrep structures of
//      * the Cnv groups.
//      * When n is even, the irreps are A1, A2, B1, B2, Ek where k = 1, ..., n/2 - 1.
//      * When n is odd, the irreps are A1, A2, Ek where k = 1, ..., n//2.
//      */
//     for n in 3..=20 {
//         let mut mol = template_molecules::gen_arbitrary_twisted_sandwich(n, 0.5);
//         mol.set_electric_field(Some(Vector3::new(0.0, 0.0, 1.0)));
//         let thresh = 1e-7;
//         test_character_table_construction(
//             &mol,
//             thresh,
//             format!("C{}v", n).as_str(),
//             2 * n as usize,
//             ({
//                 if n % 2 == 0 {
//                     n / 2 - 1
//                 } else {
//                     n / 2
//                 }
//             } + {
//                 if n % 2 == 0 {
//                     4
//                 } else {
//                     2
//                 }
//             }) as usize,
//             false,
//         );
//     }
// }

// /*
// Cnh
// */
// #[test]
// fn test_character_table_construction_symmetric_bf3_magnetic_field_c3h() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/bf3.xyz");
//     let thresh = 1e-7;
//     let mut mol = Molecule::from_xyz(&path, thresh);
//     mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
//     test_character_table_construction(&mol, thresh, "C3h", 6, 6, true);
// }

// #[test]
// fn test_character_table_construction_symmetric_bf3_magnetic_field_c3h_class_order() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/bf3.xyz");
//     let thresh = 1e-7;
//     let mut mol = Molecule::from_xyz(&path, thresh);
//     mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
//     test_character_table_construction_class_order(
//         &mol,
//         thresh,
//         &["|E|", "|C3|", "|[C3]^2|", "|S3|", "|[S3]^5|", "|σh|"],
//     );
// }

// #[test]
// fn test_character_table_construction_symmetric_xef4_magnetic_field_c4h() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/xef4.xyz");
//     let thresh = 1e-7;
//     let mut mol = Molecule::from_xyz(&path, thresh);
//     mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, -1.0)));
//     test_character_table_construction(&mol, thresh, "C4h", 8, 8, true);
// }

// #[test]
// fn test_character_table_construction_symmetric_xef4_magnetic_field_c4h_class_order() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/xef4.xyz");
//     let thresh = 1e-7;
//     let mut mol = Molecule::from_xyz(&path, thresh);
//     mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, -1.0)));
//     test_character_table_construction_class_order(
//         &mol,
//         thresh,
//         &[
//             "|E|", "|C4|", "|[C4]^3|", "|C2|", "|i|", "|S4|", "|[S4]^3|", "|σh|",
//         ],
//     );
// }

// #[test]
// fn test_character_table_construction_symmetric_vf6_magnetic_field_c4h() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/vf6.xyz");
//     let thresh = 1e-12;
//     let mut mol = Molecule::from_xyz(&path, thresh);
//     mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
//     test_character_table_construction(&mol, thresh, "C4h", 8, 8, true);
// }

// #[test]
// fn test_character_table_construction_symmetric_vf6_magnetic_field_c4h_class_order() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/vf6.xyz");
//     let thresh = 1e-12;
//     let mut mol = Molecule::from_xyz(&path, thresh);
//     mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
//     test_character_table_construction_class_order(
//         &mol,
//         thresh,
//         &[
//             "|E|", "|C4|", "|[C4]^3|", "|C2|", "|i|", "|S4|", "|[S4]^3|", "|σh|",
//         ],
//     );
// }

// #[test]
// fn test_character_table_construction_symmetric_h8_magnetic_field_c4h() {
//     // env_logger::init();
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/h8.xyz");
//     let thresh = 1e-7;
//     let mut mol = Molecule::from_xyz(&path, 1e-7);
//     mol.set_magnetic_field(Some(0.2 * Vector3::new(0.0, 0.0, 1.0)));
//     test_character_table_construction(&mol, thresh, "C4h", 8, 8, true);
// }

// #[test]
// fn test_character_table_construction_symmetric_h8_magnetic_field_c4h_class_order() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/h8.xyz");
//     let thresh = 1e-7;
//     let mut mol = Molecule::from_xyz(&path, 1e-7);
//     mol.set_magnetic_field(Some(0.2 * Vector3::new(0.0, 0.0, 1.0)));
//     test_character_table_construction_class_order(
//         &mol,
//         thresh,
//         &[
//             "|E|", "|C4|", "|[C4]^3|", "|C2|", "|i|", "|S4|", "|[S4]^3|", "|σh|",
//         ],
//     );
// }

// #[test]
// fn test_character_table_construction_symmetric_eclipsed_ferrocene_magnetic_field_c5h() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/eclipsed_ferrocene.xyz");
//     let thresh = 1e-7;
//     let mut mol = Molecule::from_xyz(&path, thresh);
//     mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, -1.0)));
//     test_character_table_construction(&mol, thresh, "C5h", 10, 10, true);
// }

// #[test]
// fn test_character_table_construction_symmetric_eclipsed_ferrocene_magnetic_field_c5h_class_order() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/eclipsed_ferrocene.xyz");
//     let thresh = 1e-7;
//     let mut mol = Molecule::from_xyz(&path, thresh);
//     mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, -1.0)));
//     test_character_table_construction_class_order(
//         &mol,
//         thresh,
//         &[
//             "|E|", "|C5|", "|[C5]^2|", "|[C5]^3|", "|[C5]^4|", "|S5|", "|[S5]^3|", "|[S5]^7|",
//             "|[S5]^9|", "|σh|",
//         ],
//     );
// }

// #[test]
// fn test_character_table_construction_symmetric_benzene_magnetic_field_c6h() {
//     // env_logger::init();
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/benzene.xyz");
//     let thresh = 1e-7;
//     let mut mol = Molecule::from_xyz(&path, thresh);
//     mol.set_magnetic_field(Some(Vector3::new(1.0, 0.0, 0.0)));
//     test_character_table_construction(&mol, thresh, "C6h", 12, 12, true);
// }

// #[test]
// fn test_character_table_construction_symmetric_benzene_magnetic_field_c6h_class_order() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/benzene.xyz");
//     let thresh = 1e-7;
//     let mut mol = Molecule::from_xyz(&path, thresh);
//     mol.set_magnetic_field(Some(Vector3::new(1.0, 0.0, 0.0)));
//     test_character_table_construction_class_order(
//         &mol,
//         thresh,
//         &[
//             "|E|", "|C6|", "|[C6]^5|", "|C3|", "|[C3]^2|", "|C2|", "|i|", "|S6|", "|[S6]^5|",
//             "|S3|", "|[S3]^5|", "|σh|",
//         ],
//     );
// }

// #[test]
// fn test_character_table_construction_symmetric_arbitrary_eclipsed_sandwich_magnetic_field_cnh() {
//     // env_logger::init();
//     for n in 3..=20 {
//         let mut mol = template_molecules::gen_arbitrary_eclipsed_sandwich(n);
//         let thresh = 1e-7;
//         mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 0.1)));
//         test_character_table_construction(
//             &mol,
//             thresh,
//             format!("C{}h", n).as_str(),
//             2 * n as usize,
//             2 * n as usize,
//             true,
//         );
//     }
// }

// /*
// Dn
// */
// #[test]
// fn test_character_table_construction_symmetric_triphenyl_radical_d3() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/triphenylradical.xyz");
//     let thresh = 1e-7;
//     let mol = Molecule::from_xyz(&path, thresh);
//     test_character_table_construction(&mol, thresh, "D3", 6, 3, false);
// }

// #[test]
// fn test_character_table_construction_symmetric_triphenyl_radical_d3_class_order() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/triphenylradical.xyz");
//     let thresh = 1e-7;
//     let mol = Molecule::from_xyz(&path, thresh);
//     test_character_table_construction_class_order(&mol, thresh, &["|E|", "2|C3|", "3|C2|"]);
// }

// #[test]
// fn test_character_table_construction_symmetric_h8_twisted_d4() {
//     let thresh = 1e-7;
//     let mol = template_molecules::gen_twisted_h8(0.1);
//     test_character_table_construction(&mol, thresh, "D4", 8, 5, false);
// }

// #[test]
// fn test_character_table_construction_symmetric_h8_twisted_d4_class_order() {
//     let thresh = 1e-7;
//     let mol = template_molecules::gen_twisted_h8(0.1);
//     test_character_table_construction_class_order(
//         &mol,
//         thresh,
//         &["|E|", "2|C4|", "|C2|", "2|C2|^(')", "2|C2|^('')"],
//     );
// }

// #[test]
// fn test_character_table_construction_symmetric_c5ph5_d5() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/c5ph5.xyz");
//     let thresh = 1e-7;
//     let mol = Molecule::from_xyz(&path, thresh);
//     test_character_table_construction(&mol, thresh, "D5", 10, 4, false);
// }

// #[test]
// fn test_character_table_construction_symmetric_c5ph5_d5_class_order() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/c5ph5.xyz");
//     let thresh = 1e-7;
//     let mol = Molecule::from_xyz(&path, thresh);
//     test_character_table_construction_class_order(&mol, thresh, &["|E|", "2|C5|", "2|[C5]^2|", "5|C2|"]);
// }

// #[test]
// fn test_character_table_construction_symmetric_c6ph6_d6() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/c6ph6.xyz");
//     let thresh = 1e-6;
//     let mol = Molecule::from_xyz(&path, thresh);
//     test_character_table_construction(&mol, thresh, "D6", 12, 6, false);
// }

// #[test]
// fn test_character_table_construction_symmetric_c6ph6_d6_class_order() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/c6ph6.xyz");
//     let thresh = 1e-6;
//     let mol = Molecule::from_xyz(&path, thresh);
//     test_character_table_construction_class_order(
//         &mol,
//         thresh,
//         &["|E|", "2|C6|", "2|C3|", "|C2|", "3|C2|^(')", "3|C2|^('')"],
//     );
// }

// #[test]
// fn test_character_table_construction_symmetric_arbitrary_twisted_sandwich_dn() {
//     /* The expected number of classes is deduced from the irrep structures of
//      * the Dn groups.
//      * When n is even, the irreps are A1, A2, B1, B2, Ek where k = 1, ..., n/2 - 1.
//      * When n is odd, the irreps are A1, A2, Ek where k = 1, ..., n//2.
//      */
//     let thresh = 1e-7;
//     for n in 3..=20 {
//         let mol = template_molecules::gen_arbitrary_twisted_sandwich(n, 0.1);
//         test_character_table_construction(
//             &mol,
//             thresh,
//             format!("D{}", n).as_str(),
//             2 * n as usize,
//             ({
//                 if n % 2 == 0 {
//                     n / 2 - 1
//                 } else {
//                     n / 2
//                 }
//             } + {
//                 if n % 2 == 0 {
//                     4
//                 } else {
//                     2
//                 }
//             }) as usize,
//             false,
//         );
//     }
// }

// /*
// Dnh
// */
// #[test]
// fn test_character_table_construction_symmetric_bf3_d3h() {
//     env_logger::init();
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/bf3.xyz");
//     let thresh = 1e-7;
//     let mol = Molecule::from_xyz(&path, thresh);
//     test_character_table_validity(&mol, thresh);
// }

// #[test]
// fn test_character_table_construction_symmetric_bf3_d3h_class_order() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/bf3.xyz");
//     let thresh = 1e-7;
//     let mol = Molecule::from_xyz(&path, thresh);
//     test_character_table_construction_class_order(
//         &mol,
//         thresh,
//         &["|E|", "2|C3|", "3|C2|", "2|S3|", "|σh|", "3|σv|"],
//     );
// }

// #[test]
// fn test_character_table_construction_symmetric_xef4_d4h() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/xef4.xyz");
//     let thresh = 1e-7;
//     let mol = Molecule::from_xyz(&path, thresh);
//     test_character_table_validity(&mol, thresh);
// }

// #[test]
// fn test_character_table_construction_symmetric_xef4_d4h_class_order() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/xef4.xyz");
//     let thresh = 1e-7;
//     let mol = Molecule::from_xyz(&path, thresh);
//     test_character_table_construction_class_order(
//         &mol,
//         thresh,
//         &[
//             "|E|",
//             "2|C4|",
//             "|C2|",
//             "2|C2|^(')",
//             "2|C2|^('')",
//             "|i|",
//             "2|S4|",
//             "|σh|",
//             "2|σv|",
//             "2|σv|^(')",
//         ],
//     );
// }

// #[test]
// fn test_character_table_construction_symmetric_h8_d4h() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/h8.xyz");
//     let thresh = 1e-7;
//     let mol = Molecule::from_xyz(&path, thresh);
//     test_character_table_construction(&mol, thresh, "D4h", 16, 10, false);
// }

// #[test]
// fn test_character_table_construction_symmetric_h8_d4h_class_order() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/h8.xyz");
//     let thresh = 1e-7;
//     let mol = Molecule::from_xyz(&path, thresh);
//     test_character_table_construction_class_order(
//         &mol,
//         thresh,
//         &[
//             "|E|",
//             "2|C4|",
//             "|C2|",
//             "2|C2|^(')",
//             "2|C2|^('')",
//             "|i|",
//             "2|S4|",
//             "|σh|",
//             "2|σv|",
//             "2|σv|^(')",
//         ],
//     );
// }

// #[test]
// fn test_character_table_construction_symmetric_eclipsed_ferrocene_d5h() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/eclipsed_ferrocene.xyz");
//     let thresh = 1e-6;
//     let mol = Molecule::from_xyz(&path, thresh);
//     test_character_table_construction(&mol, thresh, "D5h", 20, 8, false);
// }

// #[test]
// fn test_character_table_construction_symmetric_eclipsed_ferrocene_d5h_class_order() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/eclipsed_ferrocene.xyz");
//     let thresh = 1e-6;
//     let mol = Molecule::from_xyz(&path, thresh);
//     test_character_table_construction_class_order(
//         &mol,
//         thresh,
//         &[
//             "|E|",
//             "2|C5|",
//             "2|[C5]^2|",
//             "5|C2|",
//             "2|S5|",
//             "2|[S5]^3|",
//             "|σh|",
//             "5|σv|",
//         ],
//     );
// }

// #[test]
// fn test_character_table_construction_symmetric_benzene_d6h() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/benzene.xyz");
//     let thresh = 1e-7;
//     let mol = Molecule::from_xyz(&path, thresh);
//     test_character_table_validity(&mol, thresh);
// }

// #[test]
// fn test_character_table_construction_symmetric_8_eclipsed_sandwich_d8h() {
//     let thresh = 1e-7;
//     let mol = template_molecules::gen_arbitrary_eclipsed_sandwich(8);
//     test_character_table_validity(&mol, thresh);
// }

// #[test]
// fn test_character_table_construction_symmetric_9_eclipsed_sandwich_d9h() {
//     env_logger::init();
//     let thresh = 1e-7;
//     let mol = template_molecules::gen_arbitrary_eclipsed_sandwich(9);
//     test_character_table_validity(&mol, thresh);
// }

// #[test]
// fn test_character_table_construction_symmetric_h100_d100h() {
//     env_logger::init();
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/h100.xyz");
//     let thresh = 1e-6;
//     let mol = Molecule::from_xyz(&path, thresh);
//     test_character_table_validity(&mol, thresh);
// }

// #[test]
// fn test_character_table_construction_symmetric_arbitrary_eclipsed_sandwich_dnh() {
//     /* The expected number of classes is deduced from the irrep structures of
//      * the Dnh groups.
//      * When n is even, the irreps are A1(g/u), A2(g/u), B1(g/u), B2(g/u), Ek(g/u)
//      * where k = 1, ..., n/2 - 1.
//      * When n is odd, the irreps are A1('/''), A2('/''), Ek('/'')
//      * where k = 1, ..., n//2.
//      */
//     let thresh = 1e-7;
//     for n in 3..=20 {
//         let mol = template_molecules::gen_arbitrary_eclipsed_sandwich(n);
//         test_character_table_construction(
//             &mol,
//             thresh,
//             format!("D{}h", n).as_str(),
//             4 * n as usize,
//             2 * ({
//                 if n % 2 == 0 {
//                     n / 2 - 1
//                 } else {
//                     n / 2
//                 }
//             } + {
//                 if n % 2 == 0 {
//                     4
//                 } else {
//                     2
//                 }
//             }) as usize,
//             false,
//         );
//     }
// }

// /*
// Dnd
// */
// #[test]
// fn test_character_table_construction_symmetric_b2cl4_d2d() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/b2cl4.xyz");
//     let thresh = 1e-7;
//     let mol = Molecule::from_xyz(&path, thresh);
//     test_character_table_construction(&mol, thresh, "D2d", 8, 5, false);
// }

// #[test]
// fn test_character_table_construction_symmetric_b2cl4_d2d_class_order() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/b2cl4.xyz");
//     let thresh = 1e-7;
//     let mol = Molecule::from_xyz(&path, thresh);
//     test_character_table_construction_class_order(
//         &mol,
//         thresh,
//         &["|E|", "|C2|", "2|C2|^(')", "2|S4|", "2|σd|"],
//     );
// }

// #[test]
// fn test_character_table_construction_symmetric_s4n4_d2d() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/s4n4.xyz");
//     let thresh = 1e-7;
//     let mol = Molecule::from_xyz(&path, thresh);
//     test_character_table_construction(&mol, thresh, "D2d", 8, 5, false);
// }

// #[test]
// fn test_character_table_construction_symmetric_s4n4_d2d_class_order() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/s4n4.xyz");
//     let thresh = 1e-7;
//     let mol = Molecule::from_xyz(&path, thresh);
//     test_character_table_construction_class_order(
//         &mol,
//         thresh,
//         &["|E|", "|C2|", "2|C2|^(')", "2|S4|", "2|σd|"],
//     );
// }

// #[test]
// fn test_character_table_construction_symmetric_pbet4_d2d() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/pbet4.xyz");
//     let thresh = 1e-7;
//     let mol = Molecule::from_xyz(&path, thresh);
//     test_character_table_construction(&mol, thresh, "D2d", 8, 5, false);
// }

// #[test]
// fn test_character_table_construction_symmetric_pbet4_d2d_class_order() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/pbet4.xyz");
//     let thresh = 1e-7;
//     let mol = Molecule::from_xyz(&path, thresh);
//     test_character_table_construction_class_order(
//         &mol,
//         thresh,
//         &["|E|", "|C2|", "2|C2|^(')", "2|S4|", "2|σd|"],
//     );
// }

// #[test]
// fn test_character_table_construction_symmetric_allene_d2d() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/allene.xyz");
//     let thresh = 1e-7;
//     let mol = Molecule::from_xyz(&path, thresh);
//     test_character_table_construction(&mol, thresh, "D2d", 8, 5, false);
// }

// #[test]
// fn test_character_table_construction_symmetric_allene_d2d_class_order() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/allene.xyz");
//     let thresh = 1e-7;
//     let mol = Molecule::from_xyz(&path, thresh);
//     test_character_table_construction_class_order(
//         &mol,
//         thresh,
//         &["|E|", "|C2|", "2|C2|^(')", "2|S4|", "2|σd|"],
//     );
// }

// #[test]
// fn test_character_table_construction_symmetric_staggered_c2h6_d3d() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/c2h6.xyz");
//     let thresh = 1e-6;
//     let mol = Molecule::from_xyz(&path, thresh);
//     test_character_table_construction(&mol, thresh, "D3d", 12, 6, false);
// }

// #[test]
// fn test_character_table_construction_symmetric_staggered_c2h6_d3d_class_order() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/c2h6.xyz");
//     let thresh = 1e-6;
//     let mol = Molecule::from_xyz(&path, thresh);
//     test_character_table_construction_class_order(
//         &mol,
//         thresh,
//         &["|E|", "2|C3|", "3|C2|", "|i|", "2|S6|", "3|σd|"],
//     );
// }

// #[test]
// fn test_character_table_construction_symmetric_cyclohexane_chair_d3d() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/cyclohexane_chair.xyz");
//     let thresh = 1e-7;
//     let mol = Molecule::from_xyz(&path, thresh);
//     test_character_table_construction(&mol, thresh, "D3d", 12, 6, false);
// }

// #[test]
// fn test_character_table_construction_symmetric_cyclohexane_chair_d3d_class_order() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/cyclohexane_chair.xyz");
//     let thresh = 1e-7;
//     let mol = Molecule::from_xyz(&path, thresh);
//     test_character_table_construction_class_order(
//         &mol,
//         thresh,
//         &["|E|", "2|C3|", "3|C2|", "|i|", "2|S6|", "3|σd|"],
//     );
// }

// #[test]
// fn test_character_table_construction_symmetric_s8_d4d() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/s8.xyz");
//     let thresh = 1e-7;
//     let mol = Molecule::from_xyz(&path, thresh);
//     test_character_table_construction(&mol, thresh, "D4d", 16, 7, false);
// }

// #[test]
// fn test_character_table_construction_symmetric_s8_d4d_class_order() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/s8.xyz");
//     let thresh = 1e-7;
//     let mol = Molecule::from_xyz(&path, thresh);
//     test_character_table_construction_class_order(
//         &mol,
//         thresh,
//         &[
//             "|E|",
//             "2|C4|",
//             "|C2|",
//             "4|C2|^(')",
//             "2|S8|",
//             "2|[S8]^3|",
//             "4|σd|",
//         ],
//     );
// }

// #[test]
// fn test_character_table_construction_symmetric_antiprism_h8_d4d() {
//     let mol = template_molecules::gen_twisted_h8(std::f64::consts::FRAC_PI_4);
//     let thresh = 1e-7;
//     test_character_table_construction(&mol, thresh, "D4d", 16, 7, false);
// }

// #[test]
// fn test_character_table_construction_symmetric_antiprism_h8_d4d_class_order() {
//     let mol = template_molecules::gen_twisted_h8(std::f64::consts::FRAC_PI_4);
//     let thresh = 1e-7;
//     test_character_table_construction_class_order(
//         &mol,
//         thresh,
//         &[
//             "|E|",
//             "2|C4|",
//             "|C2|",
//             "4|C2|^(')",
//             "2|S8|",
//             "2|[S8]^3|",
//             "4|σd|",
//         ],
//     );
// }

// #[test]
// fn test_character_table_construction_symmetric_antiprism_pb10_d4d() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/pb10.xyz");
//     let thresh = 1e-7;
//     let mol = Molecule::from_xyz(&path, thresh);
//     test_character_table_construction(&mol, thresh, "D4d", 16, 7, false);
// }

// #[test]
// fn test_character_table_construction_symmetric_antiprism_pb10_d4d_class_order() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/pb10.xyz");
//     let thresh = 1e-7;
//     let mol = Molecule::from_xyz(&path, thresh);
//     test_character_table_construction_class_order(
//         &mol,
//         thresh,
//         &[
//             "|E|",
//             "2|C4|",
//             "|C2|",
//             "4|C2|^(')",
//             "2|S8|",
//             "2|[S8]^3|",
//             "4|σd|",
//         ],
//     );
// }

// #[test]
// fn test_character_table_construction_symmetric_staggered_ferrocene_d5d() {
//     // env_logger::init();
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/staggered_ferrocene.xyz");
//     let thresh = 1e-6;
//     let mol = Molecule::from_xyz(&path, thresh);
//     test_character_table_construction(&mol, thresh, "D5d", 20, 8, false);
// }

// #[test]
// fn test_character_table_construction_symmetric_staggered_ferrocene_d5d_class_order() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/staggered_ferrocene.xyz");
//     let thresh = 1e-6;
//     let mol = Molecule::from_xyz(&path, thresh);
//     test_character_table_construction_class_order(
//         &mol,
//         thresh,
//         &[
//             "|E|",
//             "2|C5|",
//             "2|[C5]^2|",
//             "5|C2|",
//             "|i|",
//             "2|S10|",
//             "2|[S10]^3|",
//             "5|σd|",
//         ],
//     );
// }

// #[test]
// fn test_character_table_construction_symmetric_au26_d6d() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/au26.xyz");
//     let thresh = 1e-6;
//     let mol = Molecule::from_xyz(&path, thresh);
//     test_character_table_construction(&mol, thresh, "D6d", 24, 9, false);
// }

// #[test]
// fn test_character_table_construction_symmetric_au26_d6d_class_order() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/au26.xyz");
//     let thresh = 1e-6;
//     let mol = Molecule::from_xyz(&path, thresh);
//     test_character_table_construction_class_order(
//         &mol,
//         thresh,
//         &[
//             "|E|",
//             "2|C6|",
//             "2|C3|",
//             "|C2|",
//             "6|C2|^(')",
//             "2|S12|",
//             "2|[S12]^5|",
//             "2|S4|",
//             "6|σd|",
//         ],
//     );
// }

// #[test]
// fn test_character_table_construction_symmetric_arbitrary_staggered_sandwich_dnd() {
//     let thresh = 1e-7;
//     for n in 3..=20 {
//         let mol = template_molecules::gen_arbitrary_twisted_sandwich(n, 0.5);
//         test_character_table_construction(
//             &mol,
//             thresh,
//             format!("D{}d", n).as_str(),
//             4 * n as usize,
//             (4 + n - 1) as usize,
//             false,
//         );
//     }
// }

// /*
// S2n
// */
// #[test]
// fn test_character_table_construction_symmetric_b2cl4_magnetic_field_s4() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/b2cl4.xyz");
//     let thresh = 1e-7;
//     let mut mol = Molecule::from_xyz(&path, thresh);
//     mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
//     test_character_table_construction(&mol, thresh, "S4", 4, 4, true);
// }

// #[test]
// fn test_character_table_construction_symmetric_b2cl4_magnetic_field_s4_class_order() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/b2cl4.xyz");
//     let thresh = 1e-7;
//     let mut mol = Molecule::from_xyz(&path, thresh);
//     mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
//     test_character_table_construction_class_order(&mol, thresh, &["|E|", "|C2|", "|S4|", "|[S4]^3|"]);
// }

// #[test]
// fn test_character_table_construction_symmetric_adamantane_magnetic_field_s4() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/adamantane.xyz");
//     let thresh = 1e-7;
//     let mut mol = Molecule::from_xyz(&path, thresh);
//     mol.set_magnetic_field(Some(Vector3::new(-0.1, 0.0, 0.0)));
//     test_character_table_construction(&mol, thresh, "S4", 4, 4, true);
// }

// #[test]
// fn test_character_table_construction_symmetric_adamantane_magnetic_field_s4_class_order() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/adamantane.xyz");
//     let thresh = 1e-7;
//     let mut mol = Molecule::from_xyz(&path, thresh);
//     mol.set_magnetic_field(Some(Vector3::new(-0.1, 0.0, 0.0)));
//     test_character_table_construction_class_order(&mol, thresh, &["|E|", "|C2|", "|S4|", "|[S4]^3|"]);
// }

// #[test]
// fn test_character_table_construction_symmetric_ch4_magnetic_field_s4() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/ch4.xyz");
//     let thresh = 1e-6;
//     let mut mol = Molecule::from_xyz(&path, thresh);
//     mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
//     test_character_table_construction(&mol, thresh, "S4", 4, 4, true);
// }

// #[test]
// fn test_character_table_construction_symmetric_ch4_magnetic_field_s4_class_order() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/ch4.xyz");
//     let thresh = 1e-6;
//     let mut mol = Molecule::from_xyz(&path, thresh);
//     mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
//     test_character_table_construction_class_order(&mol, thresh, &["|E|", "|C2|", "|S4|", "|[S4]^3|"]);
// }

// #[test]
// fn test_character_table_construction_symmetric_65coronane_s6() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/coronane65.xyz");
//     let thresh = 1e-7;
//     let mol = Molecule::from_xyz(&path, thresh);
//     test_character_table_construction(&mol, thresh, "S6", 6, 6, true);
// }

// #[test]
// fn test_character_table_construction_symmetric_65coronane_s6_class_order() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/coronane65.xyz");
//     let thresh = 1e-7;
//     let mol = Molecule::from_xyz(&path, thresh);
//     test_character_table_construction_class_order(
//         &mol,
//         thresh,
//         &["|E|", "|C3|", "|[C3]^2|", "|i|", "|S6|", "|[S6]^5|"],
//     );
// }

// #[test]
// fn test_character_table_construction_symmetric_65coronane_magnetic_field_s6() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/coronane65.xyz");
//     let thresh = 1e-7;
//     let mut mol = Molecule::from_xyz(&path, thresh);
//     mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, -1.0)));
//     test_character_table_construction(&mol, thresh, "S6", 6, 6, true);
// }

// #[test]
// fn test_character_table_construction_symmetric_65coronane_magnetic_field_s6_class_order() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/coronane65.xyz");
//     let thresh = 1e-7;
//     let mut mol = Molecule::from_xyz(&path, thresh);
//     mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, -1.0)));
//     test_character_table_construction_class_order(
//         &mol,
//         thresh,
//         &["|E|", "|C3|", "|[C3]^2|", "|i|", "|S6|", "|[S6]^5|"],
//     );
// }

// #[test]
// fn test_character_table_construction_symmetric_staggered_c2h6_magnetic_field_s6() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/c2h6.xyz");
//     let thresh = 1e-7;
//     let mut mol = Molecule::from_xyz(&path, thresh);
//     mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
//     test_character_table_construction(&mol, thresh, "S6", 6, 6, true);
// }

// #[test]
// fn test_character_table_construction_symmetric_staggered_c2h6_magnetic_field_s6_class_order() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/c2h6.xyz");
//     let thresh = 1e-7;
//     let mut mol = Molecule::from_xyz(&path, thresh);
//     mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
//     test_character_table_construction_class_order(
//         &mol,
//         thresh,
//         &["|E|", "|C3|", "|[C3]^2|", "|i|", "|S6|", "|[S6]^5|"],
//     );
// }

// #[test]
// fn test_character_table_construction_symmetric_c60_magnetic_field_s6() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/c60.xyz");
//     let thresh = 1e-5;
//     let mut mol = Molecule::from_xyz(&path, thresh);
//     mol.set_magnetic_field(Some(Vector3::new(
//         -0.5773503107731,
//         -0.1875926572335,
//         0.7946543988441,
//     )));
//     test_character_table_construction(&mol, thresh, "S6", 6, 6, true);
// }

// #[test]
// fn test_character_table_construction_symmetric_c60_magnetic_field_s6_class_order() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/c60.xyz");
//     let thresh = 1e-5;
//     let mut mol = Molecule::from_xyz(&path, thresh);
//     mol.set_magnetic_field(Some(Vector3::new(
//         -0.5773503107731,
//         -0.1875926572335,
//         0.7946543988441,
//     )));
//     test_character_table_construction_class_order(
//         &mol,
//         thresh,
//         &["|E|", "|C3|", "|[C3]^2|", "|i|", "|S6|", "|[S6]^5|"],
//     );
// }

// #[test]
// fn test_character_table_construction_symmetric_vh2o6_magnetic_field_s6() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/vh2o6.xyz");
//     let thresh = 1e-12;
//     let mut mol = Molecule::from_xyz(&path, thresh);
//     mol.set_magnetic_field(Some(Vector3::new(-0.2, 0.2, 0.2)));
//     test_character_table_construction(&mol, thresh, "S6", 6, 6, true);
// }

// #[test]
// fn test_character_table_construction_symmetric_vh2o6_magnetic_field_s6_class_order() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/vh2o6.xyz");
//     let thresh = 1e-12;
//     let mut mol = Molecule::from_xyz(&path, thresh);
//     mol.set_magnetic_field(Some(Vector3::new(-0.2, 0.2, 0.2)));
//     test_character_table_construction_class_order(
//         &mol,
//         thresh,
//         &["|E|", "|C3|", "|[C3]^2|", "|i|", "|S6|", "|[S6]^5|"],
//     );
// }

// #[test]
// fn test_character_table_construction_symmetric_vf6_magnetic_field_s6() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/vf6.xyz");
//     let thresh = 1e-12;
//     let mut mol = Molecule::from_xyz(&path, thresh);
//     mol.set_magnetic_field(Some(Vector3::new(1.0, 1.0, 1.0)));
//     test_character_table_construction(&mol, thresh, "S6", 6, 6, true);
// }

// #[test]
// fn test_character_table_construction_symmetric_vf6_magnetic_field_s6_class_order() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/vf6.xyz");
//     let thresh = 1e-12;
//     let mut mol = Molecule::from_xyz(&path, thresh);
//     mol.set_magnetic_field(Some(Vector3::new(1.0, 1.0, 1.0)));
//     test_character_table_construction_class_order(
//         &mol,
//         thresh,
//         &["|E|", "|C3|", "|[C3]^2|", "|i|", "|S6|", "|[S6]^5|"],
//     );
// }

// #[test]
// fn test_character_table_construction_symmetric_s8_magnetic_field_s8() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/s8.xyz");
//     let thresh = 1e-7;
//     let mut mol = Molecule::from_xyz(&path, thresh);
//     mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
//     test_character_table_construction(&mol, thresh, "S8", 8, 8, true);
// }

// #[test]
// fn test_character_table_construction_symmetric_s8_magnetic_field_s8_class_order() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/s8.xyz");
//     let thresh = 1e-7;
//     let mut mol = Molecule::from_xyz(&path, thresh);
//     mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
//     test_character_table_construction_class_order(
//         &mol,
//         thresh,
//         &[
//             "|E|", "|C4|", "|[C4]^3|", "|C2|", "|S8|", "|[S8]^3|", "|[S8]^5|", "|[S8]^7|",
//         ],
//     );
// }

// #[test]
// fn test_character_table_construction_symmetric_antiprism_pb10_magnetic_field_s8() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/pb10.xyz");
//     let thresh = 1e-7;
//     let mut mol = Molecule::from_xyz(&path, thresh);
//     mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
//     test_character_table_construction(&mol, thresh, "S8", 8, 8, true);
// }

// #[test]
// fn test_character_table_construction_symmetric_antiprism_pb10_magnetic_field_s8_class_order() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/pb10.xyz");
//     let thresh = 1e-7;
//     let mut mol = Molecule::from_xyz(&path, thresh);
//     mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
//     test_character_table_construction_class_order(
//         &mol,
//         thresh,
//         &[
//             "|E|", "|C4|", "|[C4]^3|", "|C2|", "|S8|", "|[S8]^3|", "|[S8]^5|", "|[S8]^7|",
//         ],
//     );
// }

// #[test]
// fn test_character_table_construction_symmetric_staggered_ferrocene_magnetic_field_s10() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/staggered_ferrocene.xyz");
//     let thresh = 1e-6;
//     let mut mol = Molecule::from_xyz(&path, thresh);
//     mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
//     test_character_table_construction(&mol, thresh, "S10", 10, 10, true);
// }

// #[test]
// fn test_character_table_construction_symmetric_staggered_ferrocene_magnetic_field_s10_class_order() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/staggered_ferrocene.xyz");
//     let thresh = 1e-6;
//     let mut mol = Molecule::from_xyz(&path, thresh);
//     mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
//     test_character_table_construction_class_order(
//         &mol,
//         thresh,
//         &[
//             "|E|",
//             "|C5|",
//             "|[C5]^2|",
//             "|[C5]^3|",
//             "|[C5]^4|",
//             "|i|",
//             "|S10|",
//             "|[S10]^3|",
//             "|[S10]^7|",
//             "|[S10]^9|",
//         ],
//     );
// }

// #[test]
// fn test_character_table_construction_symmetric_c60_magnetic_field_s10() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/c60.xyz");
//     let thresh = 1e-5;
//     let mut mol = Molecule::from_xyz(&path, thresh);
//     mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
//     test_character_table_construction(&mol, thresh, "S10", 10, 10, true);
// }

// #[test]
// fn test_character_table_construction_symmetric_c60_magnetic_field_s10_class_order() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/c60.xyz");
//     let thresh = 1e-5;
//     let mut mol = Molecule::from_xyz(&path, thresh);
//     mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
//     test_character_table_construction_class_order(
//         &mol,
//         thresh,
//         &[
//             "|E|",
//             "|C5|",
//             "|[C5]^2|",
//             "|[C5]^3|",
//             "|[C5]^4|",
//             "|i|",
//             "|S10|",
//             "|[S10]^3|",
//             "|[S10]^7|",
//             "|[S10]^9|",
//         ],
//     );
// }

// #[test]
// fn test_character_table_construction_symmetric_au26_magnetic_field_s12() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/au26.xyz");
//     let thresh = 1e-6;
//     let mut mol = Molecule::from_xyz(&path, thresh);
//     mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
//     test_character_table_construction(&mol, thresh, "S12", 12, 12, true);
// }

// #[test]
// fn test_character_table_construction_symmetric_au26_magnetic_field_s12_class_order() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/au26.xyz");
//     let thresh = 1e-6;
//     let mut mol = Molecule::from_xyz(&path, thresh);
//     mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
//     test_character_table_construction_class_order(
//         &mol,
//         thresh,
//         &[
//             "|E|",
//             "|C6|",
//             "|[C6]^5|",
//             "|C3|",
//             "|[C3]^2|",
//             "|C2|",
//             "|S12|",
//             "|[S12]^5|",
//             "|[S12]^7|",
//             "|[S12]^11|",
//             "|[S12]^3|",
//             "|[S12]^9|",
//         ],
//     );
// }

// #[test]
// fn test_character_table_construction_symmetric_arbitrary_staggered_sandwich_magnetic_field_s2n() {
//     let thresh = 1e-7;
//     for n in 3..=20 {
//         let mut mol = template_molecules::gen_arbitrary_twisted_sandwich(n, 0.5);
//         mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
//         test_character_table_construction(
//             &mol,
//             thresh,
//             format!("S{}", 2 * n).as_str(),
//             2 * n as usize,
//             2 * n as usize,
//             true,
//         );
//     }
// }

// /*********
// Asymmetric
// *********/
// /*
// C2
// */
// #[test]
// fn test_character_table_construction_asymmetric_spiroketal_c2() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/spiroketal.xyz");
//     let thresh = 1e-7;
//     let mol = Molecule::from_xyz(&path, thresh);
//     test_character_table_construction(&mol, thresh, "C2", 2, 2, true);
// }

// #[test]
// fn test_character_table_construction_asymmetric_spiroketal_c2_class_order() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/spiroketal.xyz");
//     let thresh = 1e-7;
//     let mol = Molecule::from_xyz(&path, thresh);
//     test_character_table_construction_class_order(&mol, thresh, &["|E|", "|C2|"]);
// }

// #[test]
// fn test_character_table_construction_asymmetric_cyclohexene_c2() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/cyclohexene.xyz");
//     let thresh = 1e-7;
//     let mol = Molecule::from_xyz(&path, thresh);
//     test_character_table_construction(&mol, thresh, "C2", 2, 2, true);
// }

// #[test]
// fn test_character_table_construction_asymmetric_thf_c2() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/thf.xyz");
//     let thresh = 1e-7;
//     let mol = Molecule::from_xyz(&path, thresh);
//     test_character_table_construction(&mol, thresh, "C2", 2, 2, true);
// }

// #[test]
// fn test_character_table_construction_asymmetric_tartaricacid_c2() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/tartaricacid.xyz");
//     let thresh = 1e-7;
//     let mol = Molecule::from_xyz(&path, thresh);
//     test_character_table_construction(&mol, thresh, "C2", 2, 2, true);
// }

// #[test]
// fn test_character_table_construction_asymmetric_f2allene_c2() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/f2allene.xyz");
//     let thresh = 1e-7;
//     let mol = Molecule::from_xyz(&path, thresh);
//     test_character_table_construction(&mol, thresh, "C2", 2, 2, true);
// }

// #[test]
// fn test_character_table_construction_asymmetric_water_magnetic_field_c2() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/water.xyz");
//     let thresh = 1e-7;
//     let mut mol = Molecule::from_xyz(&path, thresh);
//     mol.set_magnetic_field(Some(Vector3::new(0.0, 1.0, 0.0)));
//     test_character_table_construction(&mol, thresh, "C2", 2, 2, true);
// }

// #[test]
// fn test_character_table_construction_asymmetric_pyridine_magnetic_field_c2() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/pyridine.xyz");
//     let thresh = 1e-7;
//     let mut mol = Molecule::from_xyz(&path, thresh);
//     mol.set_magnetic_field(Some(Vector3::new(0.0, 0.2, 0.0)));
//     test_character_table_construction(&mol, thresh, "C2", 2, 2, true);
// }

// #[test]
// fn test_character_table_construction_asymmetric_cyclobutene_magnetic_field_c2() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/cyclobutene.xyz");
//     let thresh = 1e-7;
//     let mut mol = Molecule::from_xyz(&path, thresh);
//     mol.set_magnetic_field(Some(Vector3::new(0.2, 0.0, 0.0)));
//     test_character_table_construction(&mol, thresh, "C2", 2, 2, true);
// }

// #[test]
// fn test_character_table_construction_asymmetric_azulene_magnetic_field_c2() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/azulene.xyz");
//     let thresh = 1e-7;
//     let mut mol = Molecule::from_xyz(&path, thresh);
//     mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 0.2)));
//     test_character_table_construction(&mol, thresh, "C2", 2, 2, true);
// }

// #[test]
// fn test_character_table_construction_asymmetric_cis_cocl2h4o2_magnetic_field_c2() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/cis-cocl2h4o2.xyz");
//     let thresh = 1e-7;
//     let mut mol = Molecule::from_xyz(&path, thresh);
//     mol.set_magnetic_field(Some(Vector3::new(0.0, 0.2, 0.0)));
//     test_character_table_construction(&mol, thresh, "C2", 2, 2, true);
// }

// #[test]
// fn test_character_table_construction_asymmetric_cuneane_magnetic_field_c2() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/cuneane.xyz");
//     let thresh = 1e-7;
//     let mut mol = Molecule::from_xyz(&path, thresh);
//     mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 0.2)));
//     test_character_table_construction(&mol, thresh, "C2", 2, 2, true);
// }

// /***
// C2v
// ***/
// #[test]
// fn test_character_table_construction_asymmetric_water_c2v() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/water.xyz");
//     let thresh = 1e-7;
//     let mol = Molecule::from_xyz(&path, thresh);
//     test_character_table_construction(&mol, thresh, "C2v", 4, 4, true);
// }

// #[test]
// fn test_character_table_construction_asymmetric_water_c2v_class_order() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/water.xyz");
//     let thresh = 1e-7;
//     let mol = Molecule::from_xyz(&path, thresh);
//     test_character_table_construction_class_order(&mol, thresh, &["|E|", "|C2|", "|σv|", "|σv|^(')"]);
// }

// #[test]
// fn test_character_table_construction_asymmetric_pyridine_c2v() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/pyridine.xyz");
//     let thresh = 1e-7;
//     let mol = Molecule::from_xyz(&path, thresh);
//     test_character_table_construction(&mol, thresh, "C2v", 4, 4, true);
// }

// #[test]
// fn test_character_table_construction_asymmetric_pyridine_c2v_class_order() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/pyridine.xyz");
//     let thresh = 1e-7;
//     let mol = Molecule::from_xyz(&path, thresh);
//     test_character_table_construction_class_order(&mol, thresh, &["|E|", "|C2|", "|σv|", "|σv|^(')"]);
// }

// #[test]
// fn test_character_table_construction_asymmetric_cyclobutene_c2v() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/cyclobutene.xyz");
//     let thresh = 1e-7;
//     let mol = Molecule::from_xyz(&path, thresh);
//     test_character_table_construction(&mol, thresh, "C2v", 4, 4, true);
// }

// #[test]
// fn test_character_table_construction_asymmetric_azulene_c2v() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/azulene.xyz");
//     let thresh = 1e-7;
//     let mol = Molecule::from_xyz(&path, thresh);
//     test_character_table_construction(&mol, thresh, "C2v", 4, 4, true);
// }

// #[test]
// fn test_character_table_construction_asymmetric_cuneane_c2v() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/cuneane.xyz");
//     let thresh = 1e-7;
//     let mol = Molecule::from_xyz(&path, thresh);
//     test_character_table_construction(&mol, thresh, "C2v", 4, 4, true);
// }

// #[test]
// fn test_character_table_construction_asymmetric_bf3_electric_field_c2v() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/bf3.xyz");
//     let thresh = 1e-7;
//     let mut mol = Molecule::from_xyz(&path, thresh);
//     mol.set_electric_field(Some(Vector3::new(0.2, 0.0, 0.0)));
//     test_character_table_construction(&mol, thresh, "C2v", 4, 4, true);
// }

// /***
// C2h
// ***/
// #[test]
// fn test_character_table_construction_asymmetric_h2o2_c2h() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/h2o2.xyz");
//     let thresh = 1e-6;
//     let mol = Molecule::from_xyz(&path, thresh);
//     test_character_table_construction(&mol, thresh, "C2h", 4, 4, true);
// }

// #[test]
// fn test_character_table_construction_asymmetric_h2o2_c2h_class_order() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/h2o2.xyz");
//     let thresh = 1e-6;
//     let mol = Molecule::from_xyz(&path, thresh);
//     test_character_table_construction_class_order(&mol, thresh, &["|E|", "|C2|", "|i|", "|σh|"]);
// }

// #[test]
// fn test_character_table_construction_asymmetric_zethrene_c2h() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/zethrene.xyz");
//     let thresh = 1e-6;
//     let mol = Molecule::from_xyz(&path, thresh);
//     test_character_table_construction(&mol, thresh, "C2h", 4, 4, true);
// }

// #[test]
// fn test_character_table_construction_asymmetric_zethrene_c2h_class_order() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/zethrene.xyz");
//     let thresh = 1e-6;
//     let mol = Molecule::from_xyz(&path, thresh);
//     test_character_table_construction_class_order(&mol, thresh, &["|E|", "|C2|", "|i|", "|σh|"]);
// }

// #[test]
// fn test_character_table_construction_asymmetric_distorted_vf6_magnetic_field_c2h() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/vf6_d2h.xyz");
//     let thresh = 1e-7;
//     let mut mol = Molecule::from_xyz(&path, thresh);
//     mol.set_magnetic_field(Some(Vector3::new(1.0, 0.0, 0.0)));
//     test_character_table_construction(&mol, thresh, "C2h", 4, 4, true);
// }

// #[test]
// fn test_character_table_construction_asymmetric_b2h6_magnetic_field_c2h() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/b2h6.xyz");
//     let thresh = 1e-7;
//     let mut mol = Molecule::from_xyz(&path, thresh);
//     mol.set_magnetic_field(Some(Vector3::new(1.0, 0.0, 0.0)));
//     test_character_table_construction(&mol, thresh, "C2h", 4, 4, true);
// }

// #[test]
// fn test_character_table_construction_asymmetric_naphthalene_magnetic_field_c2h() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/naphthalene.xyz");
//     let thresh = 1e-7;
//     let mut mol = Molecule::from_xyz(&path, thresh);
//     mol.set_magnetic_field(Some(Vector3::new(0.0, 1.0, 0.0)));
//     test_character_table_construction(&mol, thresh, "C2h", 4, 4, true);
// }

// #[test]
// fn test_character_table_construction_asymmetric_pyrene_magnetic_field_c2h() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/pyrene.xyz");
//     let thresh = 1e-7;
//     let mut mol = Molecule::from_xyz(&path, thresh);
//     mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
//     test_character_table_construction(&mol, thresh, "C2h", 4, 4, true);
// }

// #[test]
// fn test_character_table_construction_asymmetric_c6o6_magnetic_field_c2h() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/c6o6.xyz");
//     let thresh = 1e-7;
//     let mut mol = Molecule::from_xyz(&path, thresh);
//     mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
//     test_character_table_construction(&mol, thresh, "C2h", 4, 4, true);
// }

// /*
// Cs
// */
// #[test]
// fn test_character_table_construction_asymmetric_propene_cs() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/propene.xyz");
//     let thresh = 1e-7;
//     let mol = Molecule::from_xyz(&path, thresh);
//     test_character_table_construction(&mol, thresh, "Cs", 2, 2, true);
// }

// #[test]
// fn test_character_table_construction_asymmetric_propene_cs_class_order() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/propene.xyz");
//     let thresh = 1e-7;
//     let mol = Molecule::from_xyz(&path, thresh);
//     test_character_table_construction_class_order(&mol, thresh, &["|E|", "|σh|"]);
// }

// #[test]
// fn test_character_table_construction_asymmetric_socl2_cs() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/socl2.xyz");
//     let thresh = 1e-7;
//     let mol = Molecule::from_xyz(&path, thresh);
//     test_character_table_construction(&mol, thresh, "Cs", 2, 2, true);
// }

// #[test]
// fn test_character_table_construction_asymmetric_socl2_cs_class_order() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/socl2.xyz");
//     let thresh = 1e-7;
//     let mol = Molecule::from_xyz(&path, thresh);
//     test_character_table_construction_class_order(&mol, thresh, &["|E|", "|σh|"]);
// }

// #[test]
// fn test_character_table_construction_asymmetric_hocl_cs() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/hocl.xyz");
//     let thresh = 1e-7;
//     let mol = Molecule::from_xyz(&path, thresh);
//     test_character_table_construction(&mol, thresh, "Cs", 2, 2, true);
// }

// #[test]
// fn test_character_table_construction_asymmetric_hocn_cs() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/hocn.xyz");
//     let thresh = 1e-7;
//     let mol = Molecule::from_xyz(&path, thresh);
//     test_character_table_construction(&mol, thresh, "Cs", 2, 2, true);
// }

// #[test]
// fn test_character_table_construction_asymmetric_nh2f_cs() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/nh2f.xyz");
//     let thresh = 1e-6;
//     let mol = Molecule::from_xyz(&path, thresh);
//     test_character_table_construction(&mol, thresh, "Cs", 2, 2, true);
// }

// #[test]
// fn test_character_table_construction_asymmetric_phenol_cs() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/phenol.xyz");
//     let thresh = 1e-7;
//     let mol = Molecule::from_xyz(&path, thresh);
//     test_character_table_construction(&mol, thresh, "Cs", 2, 2, true);
// }

// #[test]
// fn test_character_table_construction_asymmetric_f_pyrrole_cs() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/f-pyrrole.xyz");
//     let thresh = 1e-6;
//     let mol = Molecule::from_xyz(&path, thresh);
//     test_character_table_construction(&mol, thresh, "Cs", 2, 2, true);
// }

// #[test]
// fn test_character_table_construction_asymmetric_n2o_cs() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/n2o.xyz");
//     let thresh = 1e-7;
//     let mol = Molecule::from_xyz(&path, thresh);
//     test_character_table_construction(&mol, thresh, "Cs", 2, 2, true);
// }

// #[test]
// fn test_character_table_construction_asymmetric_fclbenzene_cs() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/fclbenzene.xyz");
//     let thresh = 1e-5;
//     let mol = Molecule::from_xyz(&path, thresh);
//     test_character_table_construction(&mol, thresh, "Cs", 2, 2, true);
// }

// #[test]
// fn test_character_table_construction_asymmetric_water_magnetic_field_cs() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/water.xyz");
//     let thresh = 1e-7;
//     let mut mol = Molecule::from_xyz(&path, thresh);
//     mol.set_magnetic_field(Some(Vector3::new(1.0, 0.0, 0.0)));
//     test_character_table_construction(&mol, thresh, "Cs", 2, 2, true);
// }

// #[test]
// fn test_character_table_construction_asymmetric_water_magnetic_field_cs_class_order() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/water.xyz");
//     let thresh = 1e-7;
//     let mut mol = Molecule::from_xyz(&path, thresh);
//     mol.set_magnetic_field(Some(Vector3::new(1.0, 0.0, 0.0)));
//     test_character_table_construction_class_order(&mol, thresh, &["|E|", "|σh|"]);
// }

// #[test]
// fn test_character_table_construction_asymmetric_pyridine_magnetic_field_cs() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/pyridine.xyz");
//     let thresh = 1e-7;
//     let mut mol = Molecule::from_xyz(&path, thresh);
//     mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 0.2)));
//     test_character_table_construction(&mol, thresh, "Cs", 2, 2, true);
// }

// #[test]
// fn test_character_table_construction_asymmetric_cyclobutene_magnetic_field_cs() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/cyclobutene.xyz");
//     let thresh = 1e-7;
//     let mut mol = Molecule::from_xyz(&path, thresh);
//     mol.set_magnetic_field(Some(Vector3::new(0.0, 0.1, 0.0)));
//     test_character_table_construction(&mol, thresh, "Cs", 2, 2, true);
// }

// #[test]
// fn test_character_table_construction_asymmetric_azulene_magnetic_field_cs() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/azulene.xyz");
//     let thresh = 1e-7;
//     let mut mol = Molecule::from_xyz(&path, thresh);
//     mol.set_magnetic_field(Some(Vector3::new(0.0, 0.1, 0.0)));
//     test_character_table_construction(&mol, thresh, "Cs", 2, 2, true);
// }

// #[test]
// fn test_character_table_construction_asymmetric_cis_cocl2h4o2_magnetic_field_cs() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/cis-cocl2h4o2.xyz");
//     let thresh = 1e-7;
//     let mut mol = Molecule::from_xyz(&path, thresh);
//     mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 0.2)));
//     test_character_table_construction(&mol, thresh, "Cs", 2, 2, true);
// }

// #[test]
// fn test_character_table_construction_asymmetric_cuneane_magnetic_field_cs() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/cuneane.xyz");
//     let thresh = 1e-7;
//     let mut mol = Molecule::from_xyz(&path, thresh);
//     mol.set_magnetic_field(Some(Vector3::new(0.0, 0.5, 0.0)));
//     test_character_table_construction(&mol, thresh, "Cs", 2, 2, true);
// }

// #[test]
// fn test_character_table_construction_asymmetric_water_electric_field_cs() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/water.xyz");
//     let thresh = 1e-7;
//     let mut mol = Molecule::from_xyz(&path, thresh);
//     mol.set_electric_field(Some(Vector3::new(1.0, 0.0, 0.0)));
//     test_character_table_construction(&mol, thresh, "Cs", 2, 2, true);
// }

// #[test]
// fn test_character_table_construction_asymmetric_pyridine_electric_field_cs() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/pyridine.xyz");
//     let thresh = 1e-7;
//     let mut mol = Molecule::from_xyz(&path, thresh);
//     mol.set_electric_field(Some(Vector3::new(0.0, 0.0, 0.2)));
//     test_character_table_construction(&mol, thresh, "Cs", 2, 2, true);
// }

// #[test]
// fn test_character_table_construction_asymmetric_cyclobutene_electric_field_cs() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/cyclobutene.xyz");
//     let thresh = 1e-7;
//     let mut mol = Molecule::from_xyz(&path, thresh);
//     mol.set_electric_field(Some(Vector3::new(0.0, 0.1, 0.0)));
//     test_character_table_construction(&mol, thresh, "Cs", 2, 2, true);
// }

// #[test]
// fn test_character_table_construction_asymmetric_azulene_electric_field_cs() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/azulene.xyz");
//     let thresh = 1e-7;
//     let mut mol = Molecule::from_xyz(&path, thresh);
//     mol.set_electric_field(Some(Vector3::new(0.0, 0.1, 0.0)));
//     test_character_table_construction(&mol, thresh, "Cs", 2, 2, true);
// }

// #[test]
// fn test_character_table_construction_asymmetric_cis_cocl2h4o2_electric_field_cs() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/cis-cocl2h4o2.xyz");
//     let thresh = 1e-7;
//     let mut mol = Molecule::from_xyz(&path, thresh);
//     mol.set_electric_field(Some(Vector3::new(0.0, 0.0, 0.2)));
//     test_character_table_construction(&mol, thresh, "Cs", 2, 2, true);
// }

// #[test]
// fn test_character_table_construction_asymmetric_cuneane_electric_field_cs() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/cuneane.xyz");
//     let thresh = 1e-7;
//     let mut mol = Molecule::from_xyz(&path, thresh);
//     mol.set_electric_field(Some(Vector3::new(0.0, 0.5, 0.0)));
//     test_character_table_construction(&mol, thresh, "Cs", 2, 2, true);
// }

// #[test]
// fn test_character_table_construction_asymmetric_bf3_magnetic_field_cs() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/bf3.xyz");
//     let thresh = 1e-7;
//     let mut mol = Molecule::from_xyz(&path, thresh);
//     mol.set_magnetic_field(Some(Vector3::new(0.0, 0.5, 0.0)));
//     test_character_table_construction(&mol, thresh, "Cs", 2, 2, true);
// }

// /// This is a special case: Cs point group in a symmetric top.
// #[test]
// fn test_character_table_construction_symmetric_ch4_magnetic_field_cs() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/ch4.xyz");
//     let thresh = 1e-7;
//     let mut mol = Molecule::from_xyz(&path, thresh);
//     mol.set_magnetic_field(Some(Vector3::new(0.1, 0.1, 0.0)));
//     test_character_table_construction(&mol, thresh, "Cs", 2, 2, true);
// }

// #[test]
// fn test_character_table_construction_symmetric_ch4_magnetic_field_cs_class_order() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/ch4.xyz");
//     let thresh = 1e-7;
//     let mut mol = Molecule::from_xyz(&path, thresh);
//     mol.set_magnetic_field(Some(Vector3::new(0.1, 0.1, 0.0)));
//     test_character_table_construction_class_order(&mol, thresh, &["|E|", "|σh|"]);
// }

// /// This is another special case: Cs point group in a symmetric top.
// #[test]
// fn test_character_table_construction_symmetric_ch4_electric_field_cs() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/ch4.xyz");
//     let thresh = 1e-7;
//     let mut mol = Molecule::from_xyz(&path, thresh);
//     mol.set_electric_field(Some(Vector3::new(0.1, 0.1, 0.0)));
//     test_character_table_construction(&mol, thresh, "Cs", 2, 2, true);
// }

// #[test]
// fn test_character_table_construction_asymmetric_atom_magnetic_electric_field_cs() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/th.xyz");
//     let thresh = 1e-7;
//     let mut mol = Molecule::from_xyz(&path, thresh);
//     mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 0.2)));
//     mol.set_electric_field(Some(Vector3::new(0.0, 0.1, 0.0)));
//     test_character_table_construction(&mol, thresh, "Cs", 2, 2, true);
// }

// /*
// D2
// */
// #[test]
// fn test_character_table_construction_asymmetric_i4_biphenyl_d2() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/i4-biphenyl.xyz");
//     let thresh = 1e-7;
//     let mol = Molecule::from_xyz(&path, thresh);
//     test_character_table_construction(&mol, thresh, "D2", 4, 4, true);
// }

// #[test]
// fn test_character_table_construction_asymmetric_i4_biphenyl_d2_class_order() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/i4-biphenyl.xyz");
//     let thresh = 1e-7;
//     let mol = Molecule::from_xyz(&path, thresh);
//     test_character_table_construction_class_order(&mol, thresh, &["|E|", "|C2|", "|C2|^(')", "|C2|^('')"]);
// }

// #[test]
// fn test_abstrainto_ct_group_asymmetric_twistane_d2() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/twistane.xyz");
//     let thresh = 1e-7;
//     let mol = Molecule::from_xyz(&path, thresh);
//     test_character_table_construction(&mol, thresh, "D2", 4, 4, true);
// }

// #[test]
// fn test_character_table_construction_asymmetric_22_paracyclophane_d2() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/paracyclophane22.xyz");
//     let thresh = 1e-7;
//     let mol = Molecule::from_xyz(&path, thresh);
//     test_character_table_construction(&mol, thresh, "D2", 4, 4, true);
// }

// /***
// D2h
// ***/
// #[test]
// fn test_character_table_construction_asymmetric_b2h6_d2h() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/b2h6.xyz");
//     let thresh = 1e-7;
//     let mol = Molecule::from_xyz(&path, thresh);
//     test_character_table_construction(&mol, thresh, "D2h", 8, 8, true);
// }

// #[test]
// fn test_character_table_construction_asymmetric_b2h6_d2h_class_order() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/b2h6.xyz");
//     let thresh = 1e-7;
//     let mol = Molecule::from_xyz(&path, thresh);
//     test_character_table_construction_class_order(
//         &mol,
//         thresh,
//         &[
//             "|E|",
//             "|C2|",
//             "|C2|^(')",
//             "|C2|^('')",
//             "|i|",
//             "|σ|",
//             "|σ|^(')",
//             "|σ|^('')",
//         ],
//     );
// }

// #[test]
// fn test_character_table_construction_asymmetric_naphthalene_d2h() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/naphthalene.xyz");
//     let thresh = 1e-7;
//     let mol = Molecule::from_xyz(&path, thresh);
//     test_character_table_construction(&mol, thresh, "D2h", 8, 8, true);
// }

// #[test]
// fn test_character_table_construction_asymmetric_naphthalene_d2h_class_order() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/naphthalene.xyz");
//     let thresh = 1e-7;
//     let mol = Molecule::from_xyz(&path, thresh);
//     test_character_table_construction_class_order(
//         &mol,
//         thresh,
//         &[
//             "|E|",
//             "|C2|",
//             "|C2|^(')",
//             "|C2|^('')",
//             "|i|",
//             "|σ|",
//             "|σ|^(')",
//             "|σ|^('')",
//         ],
//     );
// }

// #[test]
// fn test_character_table_construction_asymmetric_pyrene_d2h() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/pyrene.xyz");
//     let thresh = 1e-7;
//     let mol = Molecule::from_xyz(&path, thresh);
//     test_character_table_construction(&mol, thresh, "D2h", 8, 8, true);
// }

// #[test]
// fn test_character_table_construction_asymmetric_pyrene_d2h_class_order() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/pyrene.xyz");
//     let thresh = 1e-7;
//     let mol = Molecule::from_xyz(&path, thresh);
//     test_character_table_construction_class_order(
//         &mol,
//         thresh,
//         &[
//             "|E|",
//             "|C2|",
//             "|C2|^(')",
//             "|C2|^('')",
//             "|i|",
//             "|σ|",
//             "|σ|^(')",
//             "|σ|^('')",
//         ],
//     );
// }

// #[test]
// fn test_character_table_construction_asymmetric_c6o6_d2h() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/c6o6.xyz");
//     let thresh = 1e-7;
//     let mol = Molecule::from_xyz(&path, thresh);
//     test_character_table_construction(&mol, thresh, "D2h", 8, 8, true);
// }

// #[test]
// fn test_character_table_construction_asymmetric_distorted_vf6_d2h() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/vf6_d2h.xyz");
//     let thresh = 1e-7;
//     let mol = Molecule::from_xyz(&path, thresh);
//     test_character_table_construction(&mol, thresh, "D2h", 8, 8, true);
// }

// /***
// Ci
// ***/
// #[test]
// fn test_character_table_construction_asymmetric_meso_tartaricacid_ci() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/meso-tartaricacid.xyz");
//     let thresh = 1e-7;
//     let mol = Molecule::from_xyz(&path, thresh);
//     test_character_table_construction(&mol, thresh, "Ci", 2, 2, true);
// }

// #[test]
// fn test_character_table_construction_asymmetric_meso_tartaricacid_ci_class_order() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/meso-tartaricacid.xyz");
//     let thresh = 1e-7;
//     let mol = Molecule::from_xyz(&path, thresh);
//     test_character_table_construction_class_order(&mol, thresh, &["|E|", "|i|"]);
// }

// #[test]
// fn test_character_table_construction_asymmetric_dibromodimethylcyclohexane_ci() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/dibromodimethylcyclohexane.xyz");
//     let thresh = 1e-7;
//     let mol = Molecule::from_xyz(&path, thresh);
//     test_character_table_construction(&mol, thresh, "Ci", 2, 2, true);
// }

// #[test]
// fn test_character_table_construction_asymmetric_h2o2_magnetic_field_ci() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/h2o2.xyz");
//     let thresh = 1e-6;
//     let mut mol = Molecule::from_xyz(&path, thresh);
//     mol.set_magnetic_field(Some(Vector3::new(1.0, 2.0, -1.0)));
//     test_character_table_construction(&mol, thresh, "Ci", 2, 2, true);
// }

// #[test]
// fn test_character_table_construction_asymmetric_h2o2_magnetic_field_ci_class_order() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/h2o2.xyz");
//     let thresh = 1e-6;
//     let mut mol = Molecule::from_xyz(&path, thresh);
//     mol.set_magnetic_field(Some(Vector3::new(1.0, 2.0, -1.0)));
//     test_character_table_construction_class_order(&mol, thresh, &["|E|", "|i|"]);
// }

// #[test]
// fn test_character_table_construction_symmetric_xef4_magnetic_field_ci() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/xef4.xyz");
//     let thresh = 1e-7;
//     let mut mol = Molecule::from_xyz(&path, thresh);
//     mol.set_magnetic_field(Some(Vector3::new(1.0, 2.0, -2.0)));
//     test_character_table_construction(&mol, thresh, "Ci", 2, 2, true);
// }

// #[test]
// fn test_character_table_construction_asymmetric_c2h2_magnetic_field_ci() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/c2h2.xyz");
//     let thresh = 1e-6;
//     let mut mol = Molecule::from_xyz(&path, thresh);
//     mol.set_magnetic_field(Some(Vector3::new(1.0, -1.0, 1.0)));
//     test_character_table_construction(&mol, thresh, "Ci", 2, 2, true);
// }

// /***
// C1
// ***/
// #[test]
// fn test_character_table_construction_asymmetric_butan1ol_c1() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/butan-1-ol.xyz");
//     let thresh = 1e-7;
//     let mol = Molecule::from_xyz(&path, thresh);
//     test_character_table_construction(&mol, thresh, "C1", 1, 1, true);
// }

// #[test]
// fn test_character_table_construction_asymmetric_butan1ol_c1_class_order() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/butan-1-ol.xyz");
//     let thresh = 1e-7;
//     let mol = Molecule::from_xyz(&path, thresh);
//     test_character_table_construction_class_order(&mol, thresh, &["|E|"]);
// }

// #[test]
// fn test_character_table_construction_asymmetric_subst_5m_ring_c1() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/subst-5m-ring.xyz");
//     let thresh = 1e-7;
//     let mol = Molecule::from_xyz(&path, thresh);
//     test_character_table_construction(&mol, thresh, "C1", 1, 1, true);
// }

// #[test]
// fn test_character_table_construction_asymmetric_bf3_magnetic_field_c1() {
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/bf3.xyz");
//     let thresh = 1e-7;
//     let mut mol = Molecule::from_xyz(&path, thresh);
//     mol.set_magnetic_field(Some(Vector3::new(1.0, -1.0, 1.0)));
//     test_character_table_construction(&mol, thresh, "C1", 1, 1, true);
// }
