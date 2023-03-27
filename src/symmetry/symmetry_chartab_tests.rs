use std::collections::HashMap;

use approx;
use itertools::Itertools;
use nalgebra::Vector3;
use num::Complex;
use num_traits::{ToPrimitive, Zero};

use crate::aux::molecule::Molecule;
use crate::aux::template_molecules;
use crate::chartab::character::Character;
use crate::chartab::chartab_group::CharacterProperties;
use crate::chartab::chartab_symbols::{CollectionSymbol, LinearSpaceSymbol, MathematicalSymbol};
use crate::chartab::unityroot::UnityRoot;
use crate::chartab::{CharacterTable, CorepCharacterTable, RepCharacterTable};
use crate::group::{GroupProperties, MagneticRepresentedGroup, UnitaryRepresentedGroup};
use crate::symmetry::symmetry_core::{PreSymmetry, Symmetry};
use crate::symmetry::symmetry_element::{SpecialSymmetryTransformation, SymmetryOperation};
use crate::symmetry::symmetry_group::SymmetryGroupProperties;
use crate::symmetry::symmetry_symbols::{
    MullikenIrcorepSymbol, MullikenIrrepSymbol, SymmetryClassSymbol,
};

const ROOT: &str = env!("CARGO_MANIFEST_DIR");

// ================================================================
// Character table for abstract group from molecular symmetry tests
// ================================================================

fn test_irrep_character_table_algebraic_validity(
    chartab: &RepCharacterTable<MullikenIrrepSymbol, SymmetryClassSymbol<SymmetryOperation>>,
) {
    let order: usize = chartab.classes.keys().map(|cc| cc.size()).sum();
    // Sum of squared dimensions
    assert_eq!(
        order,
        chartab
            .irreps
            .keys()
            .map(|irrep| irrep.dimensionality().pow(2))
            .sum()
    );

    // Square character table
    assert_eq!(chartab.array().nrows(), chartab.array().ncols());

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
                        acc + (cc.size() as f64)
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
                        acc + (cc_i.size() as f64)
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
}

fn test_irrep_character_table_validity(
    chartab: &RepCharacterTable<MullikenIrrepSymbol, SymmetryClassSymbol<SymmetryOperation>>,
    expected_irreps: &[MullikenIrrepSymbol],
    expected_chars_option: Option<
        HashMap<
            (
                &MullikenIrrepSymbol,
                &SymmetryClassSymbol<SymmetryOperation>,
            ),
            Character,
        >,
    >,
) {
    test_irrep_character_table_algebraic_validity(chartab);

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
                "Character[({}, {})] = {} (= {:?}) does not match {}.",
                irrep,
                cc,
                chartab.get_character(irrep, cc),
                chartab.get_character(irrep, cc),
                ref_char
            );
        }
    }
}

fn test_ircorep_character_table_algebraic_validity(
    chartab: &CorepCharacterTable<
        MullikenIrcorepSymbol,
        RepCharacterTable<MullikenIrrepSymbol, SymmetryClassSymbol<SymmetryOperation>>,
    >,
    group: &MagneticRepresentedGroup<
        SymmetryOperation,
        UnitaryRepresentedGroup<
            SymmetryOperation,
            MullikenIrrepSymbol,
            SymmetryClassSymbol<SymmetryOperation>,
        >,
        MullikenIrcorepSymbol,
    >,
) {
    // Theorem 7.5, Newmarch, J. D. Some character theory for groups of linear and antilinear
    // operators. J. Math. Phys. 24, 742–756 (1983).
    let mag_ctb = group
        .cayley_table()
        .expect("Cayley table not found for the magnetic group.");
    let zeta_2 = group
        .elements()
        .iter()
        .enumerate()
        .filter(|(op_idx, op)| op.is_antiunitary() && mag_ctb[(*op_idx, *op_idx)] == 0)
        .count();
    let uni_dim_sum = chartab
        .unitary_character_table
        .characters
        .column(0)
        .iter()
        .fold(Character::zero(), |acc, x| acc + x)
        .simplify();
    if approx::relative_eq!(
        zeta_2
            .to_f64()
            .unwrap_or_else(|| panic!("Unable to convert `{zeta_2}` to `f64`.")),
        uni_dim_sum.complex_value().re,
        max_relative = uni_dim_sum.threshold,
        epsilon = uni_dim_sum.threshold
    ) {
        assert!(chartab
            .intertwining_numbers
            .iter()
            .all(|(_, &intertwining_number)| intertwining_number == 1));
    }

    // Square character table
    assert_eq!(chartab.array().nrows(), chartab.array().ncols());

    // Sum of squared dimensions
    let unitary_order = chartab.get_order().div_euclid(2);
    assert_eq!(
        unitary_order,
        chartab
            .ircoreps
            .keys()
            .zip(chartab.intertwining_numbers.values())
            .map(|(ircorep, &intertwining_number)| ircorep
                .dimensionality()
                .pow(2)
                .div_euclid(intertwining_number.into()))
            .sum(),
    );
}

fn test_chartab_ordinary_group(
    mol: &Molecule,
    thresh: f64,
    expected_name: &str,
    expected_irreps: &[MullikenIrrepSymbol],
    expected_chars_option: Option<
        HashMap<
            (
                &MullikenIrrepSymbol,
                &SymmetryClassSymbol<SymmetryOperation>,
            ),
            Character,
        >,
    >,
) {
    let presym = PreSymmetry::builder()
        .moi_threshold(thresh)
        .molecule(mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    let group = UnitaryRepresentedGroup::from_molecular_symmetry(&sym, None);
    let chartab = group.character_table();
    println!("{chartab:?}");
    assert_eq!(chartab.name, expected_name);
    test_irrep_character_table_validity(chartab, expected_irreps, expected_chars_option);
}

fn test_chartab_ordinary_double_group(
    mol: &Molecule,
    thresh: f64,
    expected_name: &str,
    expected_irreps: &[MullikenIrrepSymbol],
    expected_chars_option: Option<
        HashMap<
            (
                &MullikenIrrepSymbol,
                &SymmetryClassSymbol<SymmetryOperation>,
            ),
            Character,
        >,
    >,
) {
    let presym = PreSymmetry::builder()
        .moi_threshold(thresh)
        .molecule(mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    let group = UnitaryRepresentedGroup::from_molecular_symmetry(&sym, None).to_double_group();
    let chartab = group.character_table();
    println!("{chartab:?}");
    assert_eq!(chartab.name, expected_name);
    test_irrep_character_table_validity(chartab, expected_irreps, expected_chars_option);
}

fn test_chartab_magnetic_group(
    mol: &Molecule,
    thresh: f64,
    expected_name: &str,
    expected_irreps: &[MullikenIrrepSymbol],
    expected_chars_option: Option<
        HashMap<
            (
                &MullikenIrrepSymbol,
                &SymmetryClassSymbol<SymmetryOperation>,
            ),
            Character,
        >,
    >,
) {
    let presym = PreSymmetry::builder()
        .moi_threshold(thresh)
        .molecule(mol, true)
        .build()
        .unwrap();
    let mut magsym = Symmetry::new();
    magsym.analyse(&presym, true);

    let unitary_group = UnitaryRepresentedGroup::from_molecular_symmetry(&magsym, None);
    let irrep_chartab = unitary_group.character_table();
    println!("{irrep_chartab:?}");
    assert_eq!(irrep_chartab.name, expected_name);
    test_irrep_character_table_validity(irrep_chartab, expected_irreps, expected_chars_option);

    let magnetic_group = MagneticRepresentedGroup::from_molecular_symmetry(&magsym, None);
    let ircorep_chartab = magnetic_group.character_table();
    println!("{ircorep_chartab:?}");
    assert_eq!(ircorep_chartab.name, expected_name);
    test_ircorep_character_table_algebraic_validity(ircorep_chartab, &magnetic_group);
}

fn test_chartab_magnetic_double_group(
    mol: &Molecule,
    thresh: f64,
    expected_name: &str,
    expected_irreps: &[MullikenIrrepSymbol],
    expected_chars_option: Option<
        HashMap<
            (
                &MullikenIrrepSymbol,
                &SymmetryClassSymbol<SymmetryOperation>,
            ),
            Character,
        >,
    >,
) {
    let presym = PreSymmetry::builder()
        .moi_threshold(thresh)
        .molecule(mol, true)
        .build()
        .unwrap();
    let mut magsym = Symmetry::new();
    magsym.analyse(&presym, true);

    let unitary_group =
        UnitaryRepresentedGroup::from_molecular_symmetry(&magsym, None).to_double_group();
    let irrep_chartab = unitary_group.character_table();
    println!("{irrep_chartab:?}");
    assert_eq!(irrep_chartab.name, expected_name);
    test_irrep_character_table_validity(irrep_chartab, expected_irreps, expected_chars_option);

    let magnetic_group =
        MagneticRepresentedGroup::from_molecular_symmetry(&magsym, None).to_double_group();
    let ircorep_chartab = magnetic_group.character_table();
    println!("{ircorep_chartab:?}");
    assert_eq!(ircorep_chartab.name, expected_name);
    test_ircorep_character_table_algebraic_validity(ircorep_chartab, &magnetic_group);
}

fn test_chartab_ordinary_group_from_infinite(
    mol: &Molecule,
    finite_order: u32,
    thresh: f64,
    expected_name: &str,
    expected_irreps: &[MullikenIrrepSymbol],
    expected_chars_option: Option<
        HashMap<
            (
                &MullikenIrrepSymbol,
                &SymmetryClassSymbol<SymmetryOperation>,
            ),
            Character,
        >,
    >,
) {
    let presym = PreSymmetry::builder()
        .moi_threshold(thresh)
        .molecule(mol, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    let group = UnitaryRepresentedGroup::from_molecular_symmetry(&sym, Some(finite_order));
    let chartab = group.irrep_character_table.as_ref().unwrap();
    println!("{chartab:?}");
    assert_eq!(chartab.name, expected_name);
    test_irrep_character_table_validity(chartab, expected_irreps, expected_chars_option);
}

fn test_chartab_magnetic_group_from_infinite(
    mol: &Molecule,
    finite_order: u32,
    thresh: f64,
    expected_name: &str,
    expected_irreps: &[MullikenIrrepSymbol],
    expected_chars_option: Option<
        HashMap<
            (
                &MullikenIrrepSymbol,
                &SymmetryClassSymbol<SymmetryOperation>,
            ),
            Character,
        >,
    >,
) {
    let presym = PreSymmetry::builder()
        .moi_threshold(thresh)
        .molecule(mol, true)
        .build()
        .unwrap();
    let mut magsym = Symmetry::new();
    magsym.analyse(&presym, true);

    let unitary_group =
        UnitaryRepresentedGroup::from_molecular_symmetry(&magsym, Some(finite_order));
    let irrep_chartab = unitary_group.character_table();
    assert_eq!(irrep_chartab.name, expected_name);
    test_irrep_character_table_validity(irrep_chartab, expected_irreps, expected_chars_option);

    let magnetic_group =
        MagneticRepresentedGroup::from_molecular_symmetry(&magsym, Some(finite_order));
    let ircorep_chartab = magnetic_group.character_table();
    // println!("{:?}", ircorep_chartab);
    assert_eq!(ircorep_chartab.name, expected_name);
    test_ircorep_character_table_algebraic_validity(ircorep_chartab, &magnetic_group);
}

/********
Spherical
********/
#[test]
fn test_chartab_spherical_atom_o3() {
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
    let c2 = SymmetryClassSymbol::<SymmetryOperation>::new("1||C2||", None).unwrap();
    let c2d = SymmetryClassSymbol::<SymmetryOperation>::new("1||C2|^(')|", None).unwrap();
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
    test_chartab_ordinary_group_from_infinite(
        &mol,
        2,
        thresh,
        "O(3) > D2h",
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
    let c3 = SymmetryClassSymbol::<SymmetryOperation>::new("8||C3||", None).unwrap();
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
    test_chartab_ordinary_group_from_infinite(
        &mol,
        4,
        thresh,
        "O(3) > Oh",
        &oh_expected_irreps,
        Some(oh_expected_chars),
    );
}

#[test]
fn test_chartab_spherical_atom_grey_o3() {
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
    let tc2 = SymmetryClassSymbol::<SymmetryOperation>::new("1||θ·C2||", None).unwrap();
    let tc2d = SymmetryClassSymbol::<SymmetryOperation>::new("1||θ·C2|^(')|", None).unwrap();
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
    test_chartab_magnetic_group_from_infinite(
        &mol,
        2,
        thresh,
        "O(3) + θ·O(3) > D2h + θ·D2h",
        &grey_d2h_expected_irreps,
        Some(grey_d2h_expected_chars),
    );

    let grey_oh_expected_irreps = vec![
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
        MullikenIrrepSymbol::new("|^(m)|A|_(1g)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|A|_(2g)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|E|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|T|_(1g)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|T|_(2g)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|A|_(1u)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|A|_(2u)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|E|_(u)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|T|_(1u)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|T|_(2u)|").unwrap(),
    ];
    let tc3 = SymmetryClassSymbol::<SymmetryOperation>::new("8||θ·C3||", None).unwrap();
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
            (&grey_oh_expected_irreps[2], &tc3),
            Character::new(&[(UnityRoot::new(1, 3), 1), (UnityRoot::new(2, 3), 1)]),
        ),
        (
            (&grey_oh_expected_irreps[3], &tc3),
            Character::new(&[
                (UnityRoot::new(0, 3), 1),
                (UnityRoot::new(1, 3), 1),
                (UnityRoot::new(2, 3), 1),
            ]),
        ),
        (
            (&grey_oh_expected_irreps[4], &tc3),
            Character::new(&[
                (UnityRoot::new(0, 3), 1),
                (UnityRoot::new(1, 3), 1),
                (UnityRoot::new(2, 3), 1),
            ]),
        ),
        (
            (&grey_oh_expected_irreps[5], &tc3),
            Character::new(&[(UnityRoot::new(0, 3), 1)]),
        ),
        (
            (&grey_oh_expected_irreps[6], &tc3),
            Character::new(&[(UnityRoot::new(0, 3), 1)]),
        ),
        (
            (&grey_oh_expected_irreps[7], &tc3),
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
            (&grey_oh_expected_irreps[12], &tc3),
            Character::new(&[(UnityRoot::new(0, 3), 1)]),
        ),
        (
            (&grey_oh_expected_irreps[13], &tc3),
            Character::new(&[
                (UnityRoot::new(0, 3), 1),
                (UnityRoot::new(1, 3), 1),
                (UnityRoot::new(2, 3), 1),
            ]),
        ),
        (
            (&grey_oh_expected_irreps[14], &tc3),
            Character::new(&[
                (UnityRoot::new(0, 3), 1),
                (UnityRoot::new(1, 3), 1),
                (UnityRoot::new(2, 3), 1),
            ]),
        ),
        (
            (&grey_oh_expected_irreps[15], &tc3),
            Character::new(&[(UnityRoot::new(1, 3), 1), (UnityRoot::new(2, 3), 1)]),
        ),
        (
            (&grey_oh_expected_irreps[16], &tc3),
            Character::new(&[(UnityRoot::new(1, 3), 1), (UnityRoot::new(2, 3), 1)]),
        ),
        (
            (&grey_oh_expected_irreps[17], &tc3),
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
    test_chartab_magnetic_group_from_infinite(
        &mol,
        4,
        thresh,
        "O(3) + θ·O(3) > Oh + θ·Oh",
        &grey_oh_expected_irreps,
        Some(grey_oh_expected_chars),
    );
}

#[test]
fn test_chartab_spherical_c60_ih() {
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
    let c5 = SymmetryClassSymbol::<SymmetryOperation>::new("12||C5||", None).unwrap();
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
    test_chartab_ordinary_group(&mol, thresh, "Ih", &expected_irreps, Some(expected_chars));
}

#[test]
fn test_chartab_spherical_c60_grey_ih() {
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
        MullikenIrrepSymbol::new("|^(m)|A|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|T|_(1g)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|T|_(2g)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|G|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|H|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|A|_(u)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|T|_(1u)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|T|_(2u)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|G|_(u)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|H|_(u)|").unwrap(),
    ];
    let tc5 = SymmetryClassSymbol::<SymmetryOperation>::new("12||θ·C5||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &tc5),
            Character::new(&[(UnityRoot::new(0, 10), 1)]),
        ),
        (
            (&expected_irreps[1], &tc5),
            Character::new(&[
                (UnityRoot::new(0, 10), 1),
                (UnityRoot::new(2, 10), 1),
                (UnityRoot::new(8, 10), 1),
            ]),
        ),
        (
            (&expected_irreps[2], &tc5),
            Character::new(&[
                (UnityRoot::new(0, 10), 1),
                (UnityRoot::new(4, 10), 1),
                (UnityRoot::new(6, 10), 1),
            ]),
        ),
        (
            (&expected_irreps[3], &tc5),
            Character::new(&[(UnityRoot::new(5, 10), 1)]),
        ),
        (
            (&expected_irreps[4], &tc5),
            Character::new(&[(UnityRoot::new(0, 10), 1), (UnityRoot::new(5, 10), 1)]),
        ),
        (
            (&expected_irreps[5], &tc5),
            Character::new(&[(UnityRoot::new(0, 10), 1)]),
        ),
        (
            (&expected_irreps[6], &tc5),
            Character::new(&[
                (UnityRoot::new(0, 10), 1),
                (UnityRoot::new(2, 10), 1),
                (UnityRoot::new(8, 10), 1),
            ]),
        ),
        (
            (&expected_irreps[7], &tc5),
            Character::new(&[
                (UnityRoot::new(0, 10), 1),
                (UnityRoot::new(4, 10), 1),
                (UnityRoot::new(6, 10), 1),
            ]),
        ),
        (
            (&expected_irreps[8], &tc5),
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
            (&expected_irreps[11], &tc5),
            Character::new(&[
                (UnityRoot::new(5, 10), 1),
                (UnityRoot::new(3, 10), 1),
                (UnityRoot::new(7, 10), 1),
            ]),
        ),
        (
            (&expected_irreps[12], &tc5),
            Character::new(&[
                (UnityRoot::new(5, 10), 1),
                (UnityRoot::new(1, 10), 1),
                (UnityRoot::new(9, 10), 1),
            ]),
        ),
        (
            (&expected_irreps[13], &tc5),
            Character::new(&[(UnityRoot::new(0, 10), 1)]),
        ),
        (
            (&expected_irreps[14], &tc5),
            Character::new(&[(UnityRoot::new(0, 10), 1), (UnityRoot::new(5, 10), 1)]),
        ),
        (
            (&expected_irreps[15], &tc5),
            Character::new(&[(UnityRoot::new(5, 10), 1)]),
        ),
        (
            (&expected_irreps[16], &tc5),
            Character::new(&[
                (UnityRoot::new(5, 10), 1),
                (UnityRoot::new(3, 10), 1),
                (UnityRoot::new(7, 10), 1),
            ]),
        ),
        (
            (&expected_irreps[17], &tc5),
            Character::new(&[
                (UnityRoot::new(5, 10), 1),
                (UnityRoot::new(9, 10), 1),
                (UnityRoot::new(1, 10), 1),
            ]),
        ),
        (
            (&expected_irreps[18], &tc5),
            Character::new(&[(UnityRoot::new(0, 10), 1)]),
        ),
        (
            (&expected_irreps[19], &tc5),
            Character::new(&[(UnityRoot::new(0, 10), 1), (UnityRoot::new(5, 10), 1)]),
        ),
    ]);
    test_chartab_magnetic_group(
        &mol,
        thresh,
        "Ih + θ·Ih",
        &expected_irreps,
        Some(expected_chars),
    );
}

#[test]
fn test_chartab_spherical_ch4_td() {
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
    let c3 = SymmetryClassSymbol::<SymmetryOperation>::new("8||C3||", None).unwrap();
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
    test_chartab_ordinary_group(&mol, thresh, "Td", &expected_irreps, Some(expected_chars));
}

#[test]
fn test_chartab_spherical_ch4_grey_td() {
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
    let tc3 = SymmetryClassSymbol::<SymmetryOperation>::new("8||θ·C3||", None).unwrap();
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
    test_chartab_magnetic_group(
        &mol,
        thresh,
        "Td + θ·Td",
        &expected_irreps,
        Some(expected_chars),
    );
}

#[test]
fn test_chartab_spherical_adamantane_td() {
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
    let c3 = SymmetryClassSymbol::<SymmetryOperation>::new("8||C3||", None).unwrap();
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
    test_chartab_ordinary_group(&mol, thresh, "Td", &expected_irreps, Some(expected_chars));
}

#[test]
fn test_chartab_spherical_adamantane_grey_td() {
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
    let tc3 = SymmetryClassSymbol::<SymmetryOperation>::new("8||θ·C3||", None).unwrap();
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
    test_chartab_magnetic_group(
        &mol,
        thresh,
        "Td + θ·Td",
        &expected_irreps,
        Some(expected_chars),
    );
}

#[test]
fn test_chartab_spherical_c165_diamond_nanoparticle_td() {
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
    let c3 = SymmetryClassSymbol::<SymmetryOperation>::new("8||C3||", None).unwrap();
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
    test_chartab_ordinary_group(&mol, thresh, "Td", &expected_irreps, Some(expected_chars));
}

#[test]
fn test_chartab_spherical_c165_diamond_nanoparticle_grey_td() {
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
    let tc3 = SymmetryClassSymbol::<SymmetryOperation>::new("8||θ·C3||", None).unwrap();
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
    test_chartab_magnetic_group(
        &mol,
        thresh,
        "Td + θ·Td",
        &expected_irreps,
        Some(expected_chars),
    );
}

#[test]
fn test_chartab_spherical_vh2o6_th() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/vh2o6.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|Γ|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|Γ|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("||T|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(u)|").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|Γ|_(u)|").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|Γ|_(u)|").unwrap(),
        MullikenIrrepSymbol::new("||T|_(u)|").unwrap(),
    ];
    let c3 = SymmetryClassSymbol::<SymmetryOperation>::new("4||C3||", None).unwrap();
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
    test_chartab_ordinary_group(&mol, thresh, "Th", &expected_irreps, Some(expected_chars));
}

#[test]
fn test_chartab_spherical_vh2o6_grey_th() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/vh2o6.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|Γ|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|Γ|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("||T|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(u)|").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|Γ|_(u)|").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|Γ|_(u)|").unwrap(),
        MullikenIrrepSymbol::new("||T|_(u)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|A|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)_(a)|Γ|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)_(b)|Γ|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|T|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|A|_(u)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)_(a)|Γ|_(u)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)_(b)|Γ|_(u)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|T|_(u)|").unwrap(),
    ];
    let tc3 = SymmetryClassSymbol::<SymmetryOperation>::new("4||θ·C3||", None).unwrap();
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
            (&expected_irreps[3], &tc3),
            Character::new(&[
                (UnityRoot::new(0, 3), 1),
                (UnityRoot::new(1, 3), 1),
                (UnityRoot::new(2, 3), 1),
            ]),
        ),
        (
            (&expected_irreps[4], &tc3),
            Character::new(&[(UnityRoot::new(0, 3), 1)]),
        ),
        (
            (&expected_irreps[5], &tc3),
            Character::new(&[(UnityRoot::new(1, 3), 1)]),
        ),
        (
            (&expected_irreps[6], &tc3),
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
            (&expected_irreps[11], &tc3),
            Character::new(&[
                (UnityRoot::new(0, 3), 1),
                (UnityRoot::new(1, 3), 1),
                (UnityRoot::new(2, 3), 1),
            ]),
        ),
        (
            (&expected_irreps[12], &tc3),
            Character::new(&[(UnityRoot::new(3, 6), 1)]),
        ),
        (
            (&expected_irreps[13], &tc3),
            Character::new(&[(UnityRoot::new(5, 6), 1)]),
        ),
        (
            (&expected_irreps[14], &tc3),
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
    test_chartab_magnetic_group(
        &mol,
        thresh,
        "Th + θ·Th",
        &expected_irreps,
        Some(expected_chars),
    );
}

#[test]
fn test_chartab_spherical_vf6_oh() {
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
    let c3 = SymmetryClassSymbol::<SymmetryOperation>::new("8||C3||", None).unwrap();
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
    test_chartab_ordinary_group(&mol, thresh, "Oh", &expected_irreps, Some(expected_chars));
}

#[test]
fn test_chartab_spherical_vf6_grey_oh() {
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
        MullikenIrrepSymbol::new("|^(m)|A|_(1g)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|A|_(2g)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|E|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|T|_(1g)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|T|_(2g)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|A|_(1u)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|A|_(2u)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|E|_(u)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|T|_(1u)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|T|_(2u)|").unwrap(),
    ];
    let tc3 = SymmetryClassSymbol::<SymmetryOperation>::new("8||θ·C3||", None).unwrap();
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
            Character::new(&[(UnityRoot::new(0, 3), 1)]),
        ),
        (
            (&expected_irreps[6], &tc3),
            Character::new(&[(UnityRoot::new(0, 3), 1)]),
        ),
        (
            (&expected_irreps[7], &tc3),
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
            (&expected_irreps[12], &tc3),
            Character::new(&[(UnityRoot::new(0, 3), 1)]),
        ),
        (
            (&expected_irreps[13], &tc3),
            Character::new(&[
                (UnityRoot::new(0, 3), 1),
                (UnityRoot::new(1, 3), 1),
                (UnityRoot::new(2, 3), 1),
            ]),
        ),
        (
            (&expected_irreps[14], &tc3),
            Character::new(&[
                (UnityRoot::new(0, 3), 1),
                (UnityRoot::new(1, 3), 1),
                (UnityRoot::new(2, 3), 1),
            ]),
        ),
        (
            (&expected_irreps[15], &tc3),
            Character::new(&[(UnityRoot::new(1, 3), 1), (UnityRoot::new(2, 3), 1)]),
        ),
        (
            (&expected_irreps[16], &tc3),
            Character::new(&[(UnityRoot::new(1, 3), 1), (UnityRoot::new(2, 3), 1)]),
        ),
        (
            (&expected_irreps[17], &tc3),
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
    test_chartab_magnetic_group(
        &mol,
        thresh,
        "Oh + θ·Oh",
        &expected_irreps,
        Some(expected_chars),
    );
}

/*****
Linear
*****/
#[test]
fn test_chartab_linear_atom_magnetic_field_cinfh() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/th.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(1.0, 2.0, -1.0)));
    verify_cinfh(&mol, thresh);
}

#[test]
fn test_chartab_linear_atom_magnetic_field_bw_dinfh_cinfh() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/th.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(1.0, 2.0, -1.0)));
    verify_bw_dinfh_cinfh(&mol, thresh);
}

#[test]
fn test_chartab_linear_atom_electric_field_cinfv() {
    /* The expected number of classes is deduced from the irrep structures of
     * the Cnv groups.
     * When n is even, the irreps are A1, A2, B1, B2, Ek where k = 1, ..., n/2 - 1.
     * When n is odd, the irreps are A1, A2, Ek where k = 1, ..., n//2.
     */
    let path: String = format!("{}{}", ROOT, "/tests/xyz/th.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(Vector3::new(-1.0, 3.0, -2.0)));
    verify_cinfv(&mol, thresh);
}

#[test]
fn test_chartab_linear_atom_electric_field_grey_cinfv() {
    // env_logger::init();
    /* The expected number of classes is deduced from the irrep structures of
     * the Cnv groups.
     * When n is even, the irreps are A1, A2, B1, B2, Ek where k = 1, ..., n/2 - 1.
     * When n is odd, the irreps are A1, A2, Ek where k = 1, ..., n//2.
     */
    let path: String = format!("{}{}", ROOT, "/tests/xyz/th.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(Vector3::new(-1.0, 3.0, -2.0)));
    verify_grey_cinfv(&mol, thresh);
}

#[test]
fn test_chartab_linear_c2h2_dinfh() {
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
    verify_dinfh(&mol, thresh);
}

#[test]
fn test_chartab_linear_c2h2_grey_dinfh() {
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
    verify_grey_dinfh(&mol, thresh);
}

#[test]
fn test_chartab_linear_c2h2_magnetic_field_cinfh() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c2h2.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);
    // Parallel field
    mol.set_magnetic_field(Some(Vector3::new(1.0, 1.0, 1.0)));
    verify_cinfh(&mol, thresh);
}

#[test]
fn test_chartab_linear_c2h2_magnetic_field_bw_dinfh_cinfh() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c2h2.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);
    // Parallel field
    mol.set_magnetic_field(Some(Vector3::new(1.0, 1.0, 1.0)));
    verify_bw_dinfh_cinfh(&mol, thresh);
}

#[test]
fn test_chartab_linear_c2h2_electric_field_cinfv() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c2h2.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);

    // Parallel field
    mol.set_electric_field(Some(Vector3::new(1.0, 1.0, 1.0)));
    verify_cinfv(&mol, thresh);
}

#[test]
fn test_chartab_linear_c2h2_electric_field_grey_cinfv() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c2h2.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);

    // Parallel field
    mol.set_electric_field(Some(Vector3::new(1.0, 1.0, 1.0)));
    verify_grey_cinfv(&mol, thresh);
}

#[test]
fn test_chartab_linear_n3_cinfv() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/n3.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    verify_cinfv(&mol, thresh);
}

#[test]
fn test_chartab_linear_n3_grey_cinfv() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/n3.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    verify_grey_cinfv(&mol, thresh);
}

#[test]
fn test_chartab_linear_n3_magnetic_field_cinf() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/n3.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);

    // Parallel field
    mol.set_magnetic_field(Some(Vector3::new(1.0, 1.0, 1.0)));
    verify_cinf(&mol, thresh);
}

#[test]
fn test_chartab_linear_n3_magnetic_field_bw_cinfv_cinf() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/n3.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);

    // Parallel field
    mol.set_magnetic_field(Some(Vector3::new(1.0, 1.0, 1.0)));
    verify_bw_cinfv_cinf(&mol, thresh);
}

#[test]
fn test_chartab_linear_n3_electric_field_cinfv() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/n3.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);

    // Parallel field
    mol.set_electric_field(Some(Vector3::new(1.0, 1.0, 1.0)));
    verify_cinfv(&mol, thresh);
}

#[test]
fn test_chartab_linear_n3_electric_field_grey_cinfv() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/n3.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);

    // Parallel field
    mol.set_electric_field(Some(Vector3::new(1.0, 1.0, 1.0)));
    verify_grey_cinfv(&mol, thresh);
}

/// Verifies the validity of the computed $`\mathcal{C}_{\infty v}`$ character table of irreps.
///
/// # Arguments
///
/// * `mol` - A reference to a [`Molecule`] structure.
/// * `thresh` - A threshold for symmetry detection.
///
/// # Panics
///
/// Panics when any expected condition is not fulfilled.
fn verify_cinfv(mol: &Molecule, thresh: f64) {
    for n in 3usize..=20usize {
        verify_cnv_from_cinfv(mol, thresh, n);
    }
}

/// Verifies the validity of the computed $`\mathcal{C}_{\infty v} + \theta\mathcal{C}_{\infty v}`$
/// character table of irreps.
///
/// # Arguments
///
/// * `mol` - A reference to a [`Molecule`] structure.
/// * `thresh` - A threshold for symmetry detection.
///
/// # Panics
///
/// Panics when any expected condition is not fulfilled.
fn verify_grey_cinfv(mol: &Molecule, thresh: f64) {
    for n in 2usize..=20usize {
        verify_grey_cnv_from_grey_cinfv(mol, thresh, n);
    }
}

/// Verifies the validity of the computed $`\mathcal{C}_{\infty v}(\mathcal{C}_{\infty})`$
/// character table of irreps.
///
/// # Arguments
///
/// * `mol` - A reference to a [`Molecule`] structure.
/// * `thresh` - A threshold for symmetry detection.
///
/// # Panics
///
/// Panics when any expected condition is not fulfilled.
fn verify_bw_cinfv_cinf(mol: &Molecule, thresh: f64) {
    for n in 3usize..=20usize {
        let expected_irreps = if n % 2 == 0 {
            let mut irreps = vec![
                MullikenIrrepSymbol::new("||A|_(1)|").unwrap(),
                MullikenIrrepSymbol::new("||A|_(2)|").unwrap(),
                MullikenIrrepSymbol::new("||B|_(1)|").unwrap(),
                MullikenIrrepSymbol::new("||B|_(2)|").unwrap(),
            ];
            if n == 4 {
                irreps.push(MullikenIrrepSymbol::new("||E||").unwrap());
            } else {
                irreps.extend(
                    (1..n.div_euclid(2))
                        .map(|k| MullikenIrrepSymbol::new(&format!("||E|_({})|", k)).unwrap()),
                );
            }
            irreps
        } else {
            let mut irreps = vec![
                MullikenIrrepSymbol::new("||A|_(1)|").unwrap(),
                MullikenIrrepSymbol::new("||A|_(2)|").unwrap(),
            ];
            if n == 3 {
                irreps.push(MullikenIrrepSymbol::new("||E||").unwrap());
            } else {
                irreps.extend(
                    (1..=n.div_euclid(2))
                        .map(|k| MullikenIrrepSymbol::new(&format!("||E|_({})|", k)).unwrap()),
                );
            };
            irreps
        };
        test_chartab_magnetic_group_from_infinite(
            &mol,
            n as u32,
            thresh,
            format!("C∞v > C{n}v").as_str(),
            &expected_irreps,
            None,
        );
    }
}

/// Verifies the validity of the computed $`\mathcal{C}_{\infty h}`$ character table of irreps.
///
/// # Arguments
///
/// * `mol` - A reference to a [`Molecule`] structure.
/// * `thresh` - A threshold for symmetry detection.
///
/// # Panics
///
/// Panics when any expected condition is not fulfilled.
fn verify_cinfh(mol: &Molecule, thresh: f64) {
    for n in 2usize..=10usize {
        let m = if n % 2 == 0 {
            // Cnh for n even
            n
        } else {
            // C(2n)h for n odd
            2 * n
        };
        let mut expected_irreps = vec![MullikenIrrepSymbol::new("||A|_(g)|").unwrap()];
        if m == 4 {
            expected_irreps.push(MullikenIrrepSymbol::new("|_(a)|Γ|_(g)|").unwrap());
        } else {
            expected_irreps.extend(
                (1..m.div_euclid(2))
                    .map(|k| MullikenIrrepSymbol::new(&format!("|_(a)|Γ|_({}g)|", k)).unwrap()),
            );
        }
        expected_irreps.push(MullikenIrrepSymbol::new("||B|_(g)|").unwrap());
        if m == 4 {
            expected_irreps.push(MullikenIrrepSymbol::new("|_(b)|Γ|_(g)|").unwrap());
        } else {
            expected_irreps.extend(
                (1..m.div_euclid(2))
                    .rev()
                    .map(|k| MullikenIrrepSymbol::new(&format!("|_(b)|Γ|_({}g)|", k)).unwrap()),
            );
        }
        expected_irreps.push(MullikenIrrepSymbol::new("||A|_(u)|").unwrap());
        if m == 4 {
            expected_irreps.push(MullikenIrrepSymbol::new("|_(a)|Γ|_(u)|").unwrap());
        } else {
            expected_irreps.extend(
                (1..m.div_euclid(2))
                    .map(|k| MullikenIrrepSymbol::new(&format!("|_(a)|Γ|_({}u)|", k)).unwrap()),
            );
        }
        expected_irreps.push(MullikenIrrepSymbol::new("||B|_(u)|").unwrap());
        if m == 4 {
            expected_irreps.push(MullikenIrrepSymbol::new("|_(b)|Γ|_(u)|").unwrap());
        } else {
            expected_irreps.extend(
                (1..m.div_euclid(2))
                    .rev()
                    .map(|k| MullikenIrrepSymbol::new(&format!("|_(b)|Γ|_({}u)|", k)).unwrap()),
            );
        }
        test_chartab_ordinary_group_from_infinite(
            &mol,
            n as u32,
            thresh,
            format!("C∞h > C{m}h").as_str(),
            &expected_irreps,
            None,
        );
    }
}

/// Verifies the validity of the computed $`\mathcal{D}_{\infty h}(\mathcal{C}_{\infty h})`$
/// character table of irreps.
///
/// # Arguments
///
/// * `mol` - A reference to a [`Molecule`] structure.
/// * `thresh` - A threshold for symmetry detection.
///
/// # Panics
///
/// Panics when any expected condition is not fulfilled.
fn verify_bw_dinfh_cinfh(mol: &Molecule, thresh: f64) {
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
        test_chartab_magnetic_group_from_infinite(
            &mol,
            n as u32,
            thresh,
            format!("D∞h > D{m}h").as_str(),
            &expected_irreps,
            None,
        );
    }
}

/// Verifies the validity of the computed $`\mathcal{D}_{\infty h}`$ character table of irreps.
///
/// # Arguments
///
/// * `mol` - A reference to a [`Molecule`] structure.
/// * `thresh` - A threshold for symmetry detection.
///
/// # Panics
///
/// Panics when any expected condition is not fulfilled.
fn verify_dinfh(mol: &Molecule, thresh: f64) {
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
            if m == 4 {
                irreps.push(MullikenIrrepSymbol::new(&format!("||E|_({parity})|")).unwrap());
            } else {
                irreps.extend(
                    (1..m.div_euclid(2)).map(|k| {
                        MullikenIrrepSymbol::new(&format!("||E|_({k}{parity})|")).unwrap()
                    }),
                );
            }
            expected_irreps.extend(irreps)
        }
        test_chartab_ordinary_group_from_infinite(
            &mol,
            n as u32,
            thresh,
            format!("D∞h > D{m}h").as_str(),
            &expected_irreps,
            None,
        );
    }
}

/// Verifies the validity of the computed $`\mathcal{D}_{\infty h} + \theta\mathcal{D}_{\infty h}`$
/// character table of irreps.
///
/// # Arguments
///
/// * `mol` - A reference to a [`Molecule`] structure.
/// * `thresh` - A threshold for symmetry detection.
///
/// # Panics
///
/// Panics when any expected condition is not fulfilled.
fn verify_grey_dinfh(mol: &Molecule, thresh: f64) {
    for n in 3usize..=20usize {
        let m = if n % 2 == 0 {
            // Dnh for n even
            n
        } else {
            // D(2n)h for n odd
            2 * n
        };
        let expected_irreps = {
            let mut irreps = ["g", "u"]
                .iter()
                .flat_map(|i_parity| {
                    let mut i_irreps = vec![
                        MullikenIrrepSymbol::new(&format!("||A|_(1{i_parity})|")).unwrap(),
                        MullikenIrrepSymbol::new(&format!("||A|_(2{i_parity})|")).unwrap(),
                        MullikenIrrepSymbol::new(&format!("||B|_(1{i_parity})|")).unwrap(),
                        MullikenIrrepSymbol::new(&format!("||B|_(2{i_parity})|")).unwrap(),
                    ];
                    if m == 4 {
                        i_irreps.push(
                            MullikenIrrepSymbol::new(&format!("||E|_({i_parity})|")).unwrap(),
                        );
                    } else {
                        i_irreps.extend((1..m.div_euclid(2)).map(|k| {
                            MullikenIrrepSymbol::new(&format!("||E|_({k}{i_parity})|")).unwrap()
                        }));
                    }
                    i_irreps
                })
                .collect_vec();
            let m_irreps = irreps
                .iter()
                .map(|irrep| {
                    MullikenIrrepSymbol::new(&format!(
                        "|^(m)|{}|^({})_({})|",
                        irrep.main(),
                        irrep.postsuper(),
                        irrep.postsub()
                    ))
                    .unwrap()
                })
                .collect_vec();
            irreps.extend(m_irreps);
            irreps
        };

        //         let mut expected_irreps = vec![
        //             MullikenIrrepSymbol::new(&format!("||A|_(1g)|")).unwrap(),
        //             MullikenIrrepSymbol::new(&format!("||A|_(2g)|")).unwrap(),
        //             MullikenIrrepSymbol::new(&format!("||B|_(1g)|")).unwrap(),
        //             MullikenIrrepSymbol::new(&format!("||B|_(2g)|")).unwrap(),
        //             MullikenIrrepSymbol::new(&format!("||A|_(1u)|")).unwrap(),
        //             MullikenIrrepSymbol::new(&format!("||A|_(2u)|")).unwrap(),
        //             MullikenIrrepSymbol::new(&format!("||B|_(1u)|")).unwrap(),
        //             MullikenIrrepSymbol::new(&format!("||B|_(2u)|")).unwrap(),
        //         ];
        //         if m == 4 {
        //             expected_irreps.push(MullikenIrrepSymbol::new(&format!("||E|_(g)|")).unwrap());
        //             expected_irreps.push(MullikenIrrepSymbol::new(&format!("||E|_(u)|")).unwrap());
        //         } else {
        //             expected_irreps.extend(
        //                 (1..m.div_euclid(2))
        //                     .map(|k| MullikenIrrepSymbol::new(&format!("||E|_({k}g)|")).unwrap()),
        //             );
        //             expected_irreps.extend(
        //                 (1..m.div_euclid(2))
        //                     .map(|k| MullikenIrrepSymbol::new(&format!("||E|_({k}u)|")).unwrap()),
        //             );
        //         }
        //         let m_irreps = expected_irreps
        //             .iter()
        //             .map(|irrep| {
        //                 MullikenIrrepSymbol::new(&format!("|^(m)|{}|_({})|", irrep.main(), irrep.postsub()))
        //                     .unwrap()
        //             })
        //             .collect_vec();
        //         expected_irreps.extend(m_irreps);
        test_chartab_magnetic_group_from_infinite(
            &mol,
            n as u32,
            thresh,
            format!("D∞h + θ·D∞h > D{m}h + θ·D{m}h").as_str(),
            &expected_irreps,
            None,
        );
    }
}

/// Verifies the validity of the computed $`\mathcal{C}_{\infty}`$ character table of irreps.
///
/// # Arguments
///
/// * `mol` - A reference to a [`Molecule`] structure.
/// * `thresh` - A threshold for symmetry detection.
///
/// # Panics
///
/// Panics when any expected condition is not fulfilled.
fn verify_cinf(mol: &Molecule, thresh: f64) {
    for n in 2usize..=10usize {
        let expected_irreps = if n % 2 == 0 {
            let mut irreps = vec![MullikenIrrepSymbol::new("||A||").unwrap()];
            if n == 4 {
                irreps.push(MullikenIrrepSymbol::new(&format!("|_(a)|Γ||")).unwrap());
            } else {
                irreps.extend(
                    (1..(n / 2))
                        .map(|i| MullikenIrrepSymbol::new(&format!("|_(a)|Γ|_({})|", i)).unwrap()),
                );
            }
            irreps.push(MullikenIrrepSymbol::new("||B||").unwrap());
            if n == 4 {
                irreps.push(MullikenIrrepSymbol::new(&format!("|_(b)|Γ||")).unwrap());
            } else {
                irreps.extend(
                    (1..(n / 2))
                        .rev()
                        .map(|i| MullikenIrrepSymbol::new(&format!("|_(b)|Γ|_({})|", i)).unwrap()),
                );
            }
            irreps
        } else {
            let mut irreps = vec![MullikenIrrepSymbol::new("||A||").unwrap()];
            if n == 3 {
                irreps.push(MullikenIrrepSymbol::new(&format!("|_(a)|Γ||")).unwrap());
                irreps.push(MullikenIrrepSymbol::new(&format!("|_(b)|Γ||")).unwrap());
            } else {
                irreps.extend(
                    (1..=n.div_euclid(2))
                        .map(|i| MullikenIrrepSymbol::new(&format!("|_(a)|Γ|_({})|", i)).unwrap()),
                );
                irreps.extend(
                    (1..=n.div_euclid(2))
                        .rev()
                        .map(|i| MullikenIrrepSymbol::new(&format!("|_(b)|Γ|_({})|", i)).unwrap()),
                );
            }
            irreps
        };
        test_chartab_ordinary_group_from_infinite(
            &mol,
            n as u32,
            thresh,
            format!("C∞ > C{n}").as_str(),
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
fn test_chartab_symmetric_ch4_magnetic_field_c3() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/ch4.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(1.0, 1.0, 1.0)));
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A||").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|Γ||").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|Γ||").unwrap(),
    ];
    let c3 = SymmetryClassSymbol::<SymmetryOperation>::new("1||C3||", None).unwrap();
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
    test_chartab_ordinary_group(&mol, thresh, "C3", &expected_irreps, Some(expected_chars));
}

#[test]
fn test_chartab_symmetric_ch4_magnetic_field_bw_c3v_c3() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/ch4.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(1.0, 1.0, 1.0)));
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||E||").unwrap(),
    ];
    let c3 = SymmetryClassSymbol::<SymmetryOperation>::new("2||C3||", None).unwrap();
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
    test_chartab_magnetic_group(&mol, thresh, "C3v", &expected_irreps, Some(expected_chars));
}

#[test]
fn test_chartab_symmetric_adamantane_magnetic_field_c3() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/adamantane.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.1, 0.1, 0.1)));
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A||").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|Γ||").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|Γ||").unwrap(),
    ];
    let c3 = SymmetryClassSymbol::<SymmetryOperation>::new("1||C3||", None).unwrap();
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
    test_chartab_ordinary_group(&mol, thresh, "C3", &expected_irreps, Some(expected_chars));
}

#[test]
fn test_chartab_symmetric_adamantane_magnetic_field_bw_c3v_c3() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/adamantane.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.1, 0.1, 0.1)));
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||E||").unwrap(),
    ];
    let c3 = SymmetryClassSymbol::<SymmetryOperation>::new("2||C3||", None).unwrap();
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
    test_chartab_magnetic_group(&mol, thresh, "C3v", &expected_irreps, Some(expected_chars));
}

#[test]
fn test_chartab_symmetric_vh2o6_electric_field_c3() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/vh2o6.xyz");
    let thresh = 1e-12;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(Vector3::new(-0.2, -0.2, -0.2)));
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A||").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|Γ||").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|Γ||").unwrap(),
    ];
    let c3 = SymmetryClassSymbol::<SymmetryOperation>::new("1||C3||", None).unwrap();
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
    test_chartab_ordinary_group(&mol, thresh, "C3", &expected_irreps, Some(expected_chars));
}

#[test]
fn test_chartab_symmetric_vh2o6_electric_field_grey_c3() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/vh2o6.xyz");
    let thresh = 1e-12;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(Vector3::new(-0.2, -0.2, -0.2)));
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A||").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|Γ||").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|Γ||").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|A||").unwrap(),
        MullikenIrrepSymbol::new("|^(m)_(a)|Γ||").unwrap(),
        MullikenIrrepSymbol::new("|^(m)_(b)|Γ||").unwrap(),
    ];
    let tc3 = SymmetryClassSymbol::<SymmetryOperation>::new("1||θ·C3||", None).unwrap();
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
            (&expected_irreps[3], &tc3),
            Character::new(&[(UnityRoot::new(3, 6), 1)]),
        ),
        (
            (&expected_irreps[4], &tc3),
            Character::new(&[(UnityRoot::new(5, 6), 1)]),
        ),
        (
            (&expected_irreps[5], &tc3),
            Character::new(&[(UnityRoot::new(1, 6), 1)]),
        ),
    ]);
    test_chartab_magnetic_group(
        &mol,
        thresh,
        "C3 + θ·C3",
        &expected_irreps,
        Some(expected_chars),
    );
}

#[test]
fn test_chartab_symmetric_65coronane_electric_field_c3() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/coronane65.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(Vector3::new(0.0, 0.0, -1.0)));
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A||").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|Γ||").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|Γ||").unwrap(),
    ];
    let c3 = SymmetryClassSymbol::<SymmetryOperation>::new("1||C3||", None).unwrap();
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
    test_chartab_ordinary_group(&mol, thresh, "C3", &expected_irreps, Some(expected_chars));
}

#[test]
fn test_chartab_symmetric_65coronane_electric_field_grey_c3() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/coronane65.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(Vector3::new(0.0, 0.0, -1.0)));
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A||").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|Γ||").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|Γ||").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|A||").unwrap(),
        MullikenIrrepSymbol::new("|^(m)_(a)|Γ||").unwrap(),
        MullikenIrrepSymbol::new("|^(m)_(b)|Γ||").unwrap(),
    ];
    let tc3 = SymmetryClassSymbol::<SymmetryOperation>::new("1||θ·C3||", None).unwrap();
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
            (&expected_irreps[3], &tc3),
            Character::new(&[(UnityRoot::new(3, 6), 1)]),
        ),
        (
            (&expected_irreps[4], &tc3),
            Character::new(&[(UnityRoot::new(5, 6), 1)]),
        ),
        (
            (&expected_irreps[5], &tc3),
            Character::new(&[(UnityRoot::new(1, 6), 1)]),
        ),
    ]);
    test_chartab_magnetic_group(
        &mol,
        thresh,
        "C3 + θ·C3",
        &expected_irreps,
        Some(expected_chars),
    );
}

#[test]
fn test_chartab_symmetric_h8_twisted_magnetic_field_c4() {
    // env_logger::init();
    let thresh = 1e-7;
    let mut mol = template_molecules::gen_twisted_h8(0.1);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 0.1)));
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A||").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|Γ||").unwrap(),
        MullikenIrrepSymbol::new("||B||").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|Γ||").unwrap(),
    ];
    let c4 = SymmetryClassSymbol::<SymmetryOperation>::new("1||C4||", None).unwrap();
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
    test_chartab_ordinary_group(&mol, thresh, "C4", &expected_irreps, Some(expected_chars));
}

#[test]
fn test_chartab_symmetric_h8_twisted_magnetic_field_bw_d4_c4() {
    // env_logger::init();
    let thresh = 1e-7;
    let mut mol = template_molecules::gen_twisted_h8(0.1);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 0.1)));
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||E||").unwrap(),
    ];
    let c4 = SymmetryClassSymbol::<SymmetryOperation>::new("2||C4||", None).unwrap();
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
    test_chartab_magnetic_group(&mol, thresh, "D4", &expected_irreps, Some(expected_chars));
}

#[test]
fn test_chartab_symmetric_h8_twisted_electric_field_c4() {
    let thresh = 1e-7;
    let mut mol = template_molecules::gen_twisted_h8(0.1);
    mol.set_electric_field(Some(Vector3::new(0.0, 0.0, -0.1)));
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A||").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|Γ||").unwrap(),
        MullikenIrrepSymbol::new("||B||").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|Γ||").unwrap(),
    ];
    let c4 = SymmetryClassSymbol::<SymmetryOperation>::new("1||C4||", None).unwrap();
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
    test_chartab_ordinary_group(&mol, thresh, "C4", &expected_irreps, Some(expected_chars));
}

#[test]
fn test_chartab_symmetric_h8_twisted_electric_field_grey_c4() {
    let thresh = 1e-7;
    let mut mol = template_molecules::gen_twisted_h8(0.1);
    mol.set_electric_field(Some(Vector3::new(0.0, 0.0, -0.1)));
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A||").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|Γ||").unwrap(),
        MullikenIrrepSymbol::new("||B||").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|Γ||").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|A||").unwrap(),
        MullikenIrrepSymbol::new("|^(m)_(a)|Γ||").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|B||").unwrap(),
        MullikenIrrepSymbol::new("|^(m)_(b)|Γ||").unwrap(),
    ];
    let tc4 = SymmetryClassSymbol::<SymmetryOperation>::new("1||θ·C4||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &tc4),
            Character::new(&[(UnityRoot::new(0, 4), 1)]),
        ),
        (
            (&expected_irreps[1], &tc4),
            Character::new(&[(UnityRoot::new(1, 4), 1)]),
        ),
        (
            (&expected_irreps[2], &tc4),
            Character::new(&[(UnityRoot::new(2, 4), 1)]),
        ),
        (
            (&expected_irreps[3], &tc4),
            Character::new(&[(UnityRoot::new(3, 4), 1)]),
        ),
        (
            (&expected_irreps[4], &tc4),
            Character::new(&[(UnityRoot::new(2, 4), 1)]),
        ),
        (
            (&expected_irreps[5], &tc4),
            Character::new(&[(UnityRoot::new(3, 4), 1)]),
        ),
        (
            (&expected_irreps[6], &tc4),
            Character::new(&[(UnityRoot::new(0, 4), 1)]),
        ),
        (
            (&expected_irreps[7], &tc4),
            Character::new(&[(UnityRoot::new(1, 4), 1)]),
        ),
    ]);
    test_chartab_magnetic_group(
        &mol,
        thresh,
        "C4 + θ·C4",
        &expected_irreps,
        Some(expected_chars),
    );
}

#[test]
fn test_chartab_symmetric_cpnico_magnetic_field_c5() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/cpnico.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, -0.2)));
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A||").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|Γ|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|Γ|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|Γ|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|Γ|_(1)|").unwrap(),
    ];
    let c5 = SymmetryClassSymbol::<SymmetryOperation>::new("1||C5||", None).unwrap();
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
    test_chartab_ordinary_group(&mol, thresh, "C5", &expected_irreps, Some(expected_chars));
}

#[test]
fn test_chartab_symmetric_cpnico_magnetic_field_bw_c5v_c5() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/cpnico.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, -0.2)));
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||E|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||E|_(2)|").unwrap(),
    ];
    let c5 = SymmetryClassSymbol::<SymmetryOperation>::new("2||C5||", None).unwrap();
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
    test_chartab_magnetic_group(&mol, thresh, "C5v", &expected_irreps, Some(expected_chars));
}

#[test]
fn test_chartab_symmetric_b7_magnetic_field_c6() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/b7.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 0.1)));
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A||").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|Γ|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|Γ|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||B||").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|Γ|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|Γ|_(1)|").unwrap(),
    ];
    let c6 = SymmetryClassSymbol::<SymmetryOperation>::new("1||C6||", None).unwrap();
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
    test_chartab_ordinary_group(&mol, thresh, "C6", &expected_irreps, Some(expected_chars));
}

#[test]
fn test_chartab_symmetric_b7_magnetic_field_bw_c6v_c6() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/b7.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 0.1)));
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||E|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||E|_(2)|").unwrap(),
    ];
    let c6 = SymmetryClassSymbol::<SymmetryOperation>::new("2||C6||", None).unwrap();
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
    test_chartab_magnetic_group(&mol, thresh, "C6v", &expected_irreps, Some(expected_chars));
}

#[test]
fn test_chartab_symmetric_arbitrary_half_sandwich_magnetic_field_cn() {
    let thresh = 1e-7;
    for n in 3..=32 {
        let mut mol = template_molecules::gen_arbitrary_half_sandwich(n);
        mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 0.1)));
        verify_cn(&mol, thresh, n)
    }
}

#[test]
fn test_chartab_symmetric_arbitrary_half_sandwich_magnetic_field_bw_cnv_cn() {
    let thresh = 1e-7;
    for n in 3..=32 {
        let mut mol = template_molecules::gen_arbitrary_half_sandwich(n);
        mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 0.1)));
        verify_bw_cnv_cn(&mol, thresh, n)
    }
}

/// Verifies the validity of the computed $`\mathcal{C}_{n}`$ character table of irreps.
///
/// # Arguments
///
/// * `mol` - A reference to a [`Molecule`] structure.
/// * `thresh` - A threshold for symmetry detection.
/// * `n` - The value of $`n`$.
///
/// # Panics
///
/// Panics when any expected condition is not fulfilled.
fn verify_cn(mol: &Molecule, thresh: f64, n: u32) {
    let expected_irreps = if n % 2 == 0 {
        let mut irreps = vec![MullikenIrrepSymbol::new("||A||").unwrap()];
        if n == 4 {
            irreps.push(MullikenIrrepSymbol::new(&format!("|_(a)|Γ||")).unwrap());
        } else {
            irreps.extend(
                (1..(n / 2))
                    .map(|i| MullikenIrrepSymbol::new(&format!("|_(a)|Γ|_({})|", i)).unwrap()),
            );
        }
        irreps.push(MullikenIrrepSymbol::new("||B||").unwrap());
        if n == 4 {
            irreps.push(MullikenIrrepSymbol::new(&format!("|_(b)|Γ||")).unwrap());
        } else {
            irreps.extend(
                (1..(n / 2))
                    .rev()
                    .map(|i| MullikenIrrepSymbol::new(&format!("|_(b)|Γ|_({})|", i)).unwrap()),
            );
        }
        irreps
    } else {
        let mut irreps = vec![MullikenIrrepSymbol::new("||A||").unwrap()];
        if n == 3 {
            irreps.push(MullikenIrrepSymbol::new(&format!("|_(a)|Γ||")).unwrap());
            irreps.push(MullikenIrrepSymbol::new(&format!("|_(b)|Γ||")).unwrap());
        } else {
            irreps.extend(
                (1..=n.div_euclid(2))
                    .map(|i| MullikenIrrepSymbol::new(&format!("|_(a)|Γ|_({})|", i)).unwrap()),
            );
            irreps.extend(
                (1..=n.div_euclid(2))
                    .rev()
                    .map(|i| MullikenIrrepSymbol::new(&format!("|_(b)|Γ|_({})|", i)).unwrap()),
            );
        }
        irreps
    };
    let cn = SymmetryClassSymbol::<SymmetryOperation>::new(&format!("1||C{}||", n), None).unwrap();
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
    test_chartab_ordinary_group(
        &mol,
        thresh,
        format!("C{n}").as_str(),
        &expected_irreps,
        Some(expected_chars),
    );
}

/// Verifies the validity of the computed $`\mathcal{C}_{nv}(\mathcal{C}_{n})`$ character table of
/// irreps.
///
/// # Arguments
///
/// * `mol` - A reference to a [`Molecule`] structure.
/// * `thresh` - A threshold for symmetry detection.
/// * `n` - The value of $`n`$.
///
/// # Panics
///
/// Panics when any expected condition is not fulfilled.
fn verify_bw_cnv_cn(mol: &Molecule, thresh: f64, n: u32) {
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
    test_chartab_magnetic_group(
        &mol,
        thresh,
        format!("C{n}v").as_str(),
        &expected_irreps,
        None,
    );
}

/*
Cnv
*/
#[test]
fn test_chartab_symmetric_nh3_c3v() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/nh3.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||E||").unwrap(),
    ];
    let c3 = SymmetryClassSymbol::<SymmetryOperation>::new("2||C3||", None).unwrap();
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
    test_chartab_ordinary_group(&mol, thresh, "C3v", &expected_irreps, Some(expected_chars));
}

#[test]
fn test_chartab_symmetric_nh3_grey_c3v() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/nh3.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||E||").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|A|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|A|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|E||").unwrap(),
    ];
    let tc3 = SymmetryClassSymbol::<SymmetryOperation>::new("2||θ·C3||", None).unwrap();
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
            Character::new(&[(UnityRoot::new(1, 3), 1), (UnityRoot::new(2, 3), 1)]),
        ),
        (
            (&expected_irreps[4], &tc3),
            Character::new(&[(UnityRoot::new(1, 3), 1), (UnityRoot::new(2, 3), 1)]),
        ),
        (
            (&expected_irreps[5], &tc3),
            Character::new(&[(UnityRoot::new(0, 3), 1)]),
        ),
    ]);
    test_chartab_magnetic_group(
        &mol,
        thresh,
        "C3v + θ·C3v",
        &expected_irreps,
        Some(expected_chars),
    );
}

#[test]
fn test_chartab_symmetric_bf3_electric_field_c3v() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/bf3.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||E||").unwrap(),
    ];
    let c3 = SymmetryClassSymbol::<SymmetryOperation>::new("2||C3||", None).unwrap();
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
    test_chartab_ordinary_group(&mol, thresh, "C3v", &expected_irreps, Some(expected_chars));
}

#[test]
fn test_chartab_symmetric_bf3_electric_field_grey_c3v() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/bf3.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||E||").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|A|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|A|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|E||").unwrap(),
    ];
    let tc3 = SymmetryClassSymbol::<SymmetryOperation>::new("2||θ·C3||", None).unwrap();
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
            Character::new(&[(UnityRoot::new(1, 3), 1), (UnityRoot::new(2, 3), 1)]),
        ),
        (
            (&expected_irreps[4], &tc3),
            Character::new(&[(UnityRoot::new(1, 3), 1), (UnityRoot::new(2, 3), 1)]),
        ),
        (
            (&expected_irreps[5], &tc3),
            Character::new(&[(UnityRoot::new(0, 3), 1)]),
        ),
    ]);
    test_chartab_magnetic_group(
        &mol,
        thresh,
        "C3v + θ·C3v",
        &expected_irreps,
        Some(expected_chars),
    );
}

#[test]
fn test_chartab_symmetric_adamantane_electric_field_c3v() {
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
    let c3 = SymmetryClassSymbol::<SymmetryOperation>::new("2||C3||", None).unwrap();
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
    test_chartab_ordinary_group(&mol, thresh, "C3v", &expected_irreps, Some(expected_chars));
}

#[test]
fn test_chartab_symmetric_adamantane_electric_field_grey_c3v() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/adamantane.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(Vector3::new(0.1, 0.1, 0.1)));
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||E||").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|A|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|A|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|E||").unwrap(),
    ];
    let tc3 = SymmetryClassSymbol::<SymmetryOperation>::new("2||θ·C3||", None).unwrap();
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
            Character::new(&[(UnityRoot::new(1, 3), 1), (UnityRoot::new(2, 3), 1)]),
        ),
        (
            (&expected_irreps[4], &tc3),
            Character::new(&[(UnityRoot::new(1, 3), 1), (UnityRoot::new(2, 3), 1)]),
        ),
        (
            (&expected_irreps[5], &tc3),
            Character::new(&[(UnityRoot::new(0, 3), 1)]),
        ),
    ]);
    test_chartab_magnetic_group(
        &mol,
        thresh,
        "C3v + θ·C3v",
        &expected_irreps,
        Some(expected_chars),
    );
}

#[test]
fn test_chartab_symmetric_ch4_electric_field_c3v() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/ch4.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(Vector3::new(1.0, 1.0, 1.0)));
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||E||").unwrap(),
    ];
    let c3 = SymmetryClassSymbol::<SymmetryOperation>::new("2||C3||", None).unwrap();
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
    test_chartab_ordinary_group(&mol, thresh, "C3v", &expected_irreps, Some(expected_chars));
}

#[test]
fn test_chartab_symmetric_ch4_electric_field_grey_c3v() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/ch4.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(Vector3::new(1.0, 1.0, 1.0)));
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||E||").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|A|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|A|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|E||").unwrap(),
    ];
    let tc3 = SymmetryClassSymbol::<SymmetryOperation>::new("2||θ·C3||", None).unwrap();
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
            Character::new(&[(UnityRoot::new(1, 3), 1), (UnityRoot::new(2, 3), 1)]),
        ),
        (
            (&expected_irreps[4], &tc3),
            Character::new(&[(UnityRoot::new(1, 3), 1), (UnityRoot::new(2, 3), 1)]),
        ),
        (
            (&expected_irreps[5], &tc3),
            Character::new(&[(UnityRoot::new(0, 3), 1)]),
        ),
    ]);
    test_chartab_magnetic_group(
        &mol,
        thresh,
        "C3v + θ·C3v",
        &expected_irreps,
        Some(expected_chars),
    );
}

#[test]
fn test_chartab_symmetric_vf6_electric_field_c3v() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/vf6.xyz");
    let thresh = 1e-12;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(0.2 * Vector3::new(1.0, 1.0, 1.0)));
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||E||").unwrap(),
    ];
    let c3 = SymmetryClassSymbol::<SymmetryOperation>::new("2||C3||", None).unwrap();
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
    test_chartab_ordinary_group(&mol, thresh, "C3v", &expected_irreps, Some(expected_chars));
}

#[test]
fn test_chartab_symmetric_vf6_electric_field_grey_c3v() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/vf6.xyz");
    let thresh = 1e-12;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(0.2 * Vector3::new(1.0, 1.0, 1.0)));
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||E||").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|A|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|A|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|E||").unwrap(),
    ];
    let tc3 = SymmetryClassSymbol::<SymmetryOperation>::new("2||θ·C3||", None).unwrap();
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
            Character::new(&[(UnityRoot::new(1, 3), 1), (UnityRoot::new(2, 3), 1)]),
        ),
        (
            (&expected_irreps[4], &tc3),
            Character::new(&[(UnityRoot::new(1, 3), 1), (UnityRoot::new(2, 3), 1)]),
        ),
        (
            (&expected_irreps[5], &tc3),
            Character::new(&[(UnityRoot::new(0, 3), 1)]),
        ),
    ]);
    test_chartab_magnetic_group(
        &mol,
        thresh,
        "C3v + θ·C3v",
        &expected_irreps,
        Some(expected_chars),
    );
}

#[test]
fn test_chartab_symmetric_sf5cl_c4v() {
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
    let c4 = SymmetryClassSymbol::<SymmetryOperation>::new("2||C4||", None).unwrap();
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
    test_chartab_ordinary_group(&mol, thresh, "C4v", &expected_irreps, Some(expected_chars));
}

#[test]
fn test_chartab_symmetric_sf5cl_grey_c4v() {
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
        MullikenIrrepSymbol::new("|^(m)|A|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|A|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|B|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|B|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|E||").unwrap(),
    ];
    let tc4 = SymmetryClassSymbol::<SymmetryOperation>::new("2||θ·C4||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &tc4),
            Character::new(&[(UnityRoot::new(0, 4), 1)]),
        ),
        (
            (&expected_irreps[1], &tc4),
            Character::new(&[(UnityRoot::new(0, 4), 1)]),
        ),
        (
            (&expected_irreps[2], &tc4),
            Character::new(&[(UnityRoot::new(2, 4), 1)]),
        ),
        (
            (&expected_irreps[3], &tc4),
            Character::new(&[(UnityRoot::new(2, 4), 1)]),
        ),
        (
            (&expected_irreps[4], &tc4),
            Character::new(&[(UnityRoot::new(1, 4), 1), (UnityRoot::new(3, 4), 1)]),
        ),
        (
            (&expected_irreps[5], &tc4),
            Character::new(&[(UnityRoot::new(2, 4), 1)]),
        ),
        (
            (&expected_irreps[6], &tc4),
            Character::new(&[(UnityRoot::new(2, 4), 1)]),
        ),
        (
            (&expected_irreps[7], &tc4),
            Character::new(&[(UnityRoot::new(0, 4), 1)]),
        ),
        (
            (&expected_irreps[8], &tc4),
            Character::new(&[(UnityRoot::new(0, 4), 1)]),
        ),
        (
            (&expected_irreps[9], &tc4),
            Character::new(&[(UnityRoot::new(1, 4), 1), (UnityRoot::new(3, 4), 1)]),
        ),
    ]);
    test_chartab_magnetic_group(
        &mol,
        thresh,
        "C4v + θ·C4v",
        &expected_irreps,
        Some(expected_chars),
    );
}

#[test]
fn test_chartab_symmetric_h8_electric_field_c4v() {
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
    let c4 = SymmetryClassSymbol::<SymmetryOperation>::new("2||C4||", None).unwrap();
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
    test_chartab_ordinary_group(&mol, thresh, "C4v", &expected_irreps, Some(expected_chars));
}

#[test]
fn test_chartab_symmetric_h8_electric_field_grey_c4v() {
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
        MullikenIrrepSymbol::new("|^(m)|A|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|A|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|B|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|B|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|E||").unwrap(),
    ];
    let tc4 = SymmetryClassSymbol::<SymmetryOperation>::new("2||θ·C4||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &tc4),
            Character::new(&[(UnityRoot::new(0, 4), 1)]),
        ),
        (
            (&expected_irreps[1], &tc4),
            Character::new(&[(UnityRoot::new(0, 4), 1)]),
        ),
        (
            (&expected_irreps[2], &tc4),
            Character::new(&[(UnityRoot::new(2, 4), 1)]),
        ),
        (
            (&expected_irreps[3], &tc4),
            Character::new(&[(UnityRoot::new(2, 4), 1)]),
        ),
        (
            (&expected_irreps[4], &tc4),
            Character::new(&[(UnityRoot::new(1, 4), 1), (UnityRoot::new(3, 4), 1)]),
        ),
        (
            (&expected_irreps[5], &tc4),
            Character::new(&[(UnityRoot::new(2, 4), 1)]),
        ),
        (
            (&expected_irreps[6], &tc4),
            Character::new(&[(UnityRoot::new(2, 4), 1)]),
        ),
        (
            (&expected_irreps[7], &tc4),
            Character::new(&[(UnityRoot::new(0, 4), 1)]),
        ),
        (
            (&expected_irreps[8], &tc4),
            Character::new(&[(UnityRoot::new(0, 4), 1)]),
        ),
        (
            (&expected_irreps[9], &tc4),
            Character::new(&[(UnityRoot::new(1, 4), 1), (UnityRoot::new(3, 4), 1)]),
        ),
    ]);
    test_chartab_magnetic_group(
        &mol,
        thresh,
        "C4v + θ·C4v",
        &expected_irreps,
        Some(expected_chars),
    );
}

#[test]
fn test_chartab_symmetric_vf6_electric_field_c4v() {
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
    let c4 = SymmetryClassSymbol::<SymmetryOperation>::new("2||C4||", None).unwrap();
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
    test_chartab_ordinary_group(&mol, thresh, "C4v", &expected_irreps, Some(expected_chars));
}

#[test]
fn test_chartab_symmetric_vf6_electric_field_grey_c4v() {
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
        MullikenIrrepSymbol::new("|^(m)|A|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|A|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|B|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|B|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|E||").unwrap(),
    ];
    let tc4 = SymmetryClassSymbol::<SymmetryOperation>::new("2||θ·C4||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &tc4),
            Character::new(&[(UnityRoot::new(0, 4), 1)]),
        ),
        (
            (&expected_irreps[1], &tc4),
            Character::new(&[(UnityRoot::new(0, 4), 1)]),
        ),
        (
            (&expected_irreps[2], &tc4),
            Character::new(&[(UnityRoot::new(2, 4), 1)]),
        ),
        (
            (&expected_irreps[3], &tc4),
            Character::new(&[(UnityRoot::new(2, 4), 1)]),
        ),
        (
            (&expected_irreps[4], &tc4),
            Character::new(&[(UnityRoot::new(1, 4), 1), (UnityRoot::new(3, 4), 1)]),
        ),
        (
            (&expected_irreps[5], &tc4),
            Character::new(&[(UnityRoot::new(2, 4), 1)]),
        ),
        (
            (&expected_irreps[6], &tc4),
            Character::new(&[(UnityRoot::new(2, 4), 1)]),
        ),
        (
            (&expected_irreps[7], &tc4),
            Character::new(&[(UnityRoot::new(0, 4), 1)]),
        ),
        (
            (&expected_irreps[8], &tc4),
            Character::new(&[(UnityRoot::new(0, 4), 1)]),
        ),
        (
            (&expected_irreps[9], &tc4),
            Character::new(&[(UnityRoot::new(1, 4), 1), (UnityRoot::new(3, 4), 1)]),
        ),
    ]);
    test_chartab_magnetic_group(
        &mol,
        thresh,
        "C4v + θ·C4v",
        &expected_irreps,
        Some(expected_chars),
    );
}

#[test]
fn test_chartab_symmetric_antiprism_pb10_electric_field_c4v() {
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
    let c4 = SymmetryClassSymbol::<SymmetryOperation>::new("2||C4||", None).unwrap();
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
    test_chartab_ordinary_group(&mol, thresh, "C4v", &expected_irreps, Some(expected_chars));
}

#[test]
fn test_chartab_symmetric_antiprism_pb10_electric_field_grey_c4v() {
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
        MullikenIrrepSymbol::new("|^(m)|A|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|A|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|B|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|B|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|E||").unwrap(),
    ];
    let tc4 = SymmetryClassSymbol::<SymmetryOperation>::new("2||θ·C4||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &tc4),
            Character::new(&[(UnityRoot::new(0, 4), 1)]),
        ),
        (
            (&expected_irreps[1], &tc4),
            Character::new(&[(UnityRoot::new(0, 4), 1)]),
        ),
        (
            (&expected_irreps[2], &tc4),
            Character::new(&[(UnityRoot::new(2, 4), 1)]),
        ),
        (
            (&expected_irreps[3], &tc4),
            Character::new(&[(UnityRoot::new(2, 4), 1)]),
        ),
        (
            (&expected_irreps[4], &tc4),
            Character::new(&[(UnityRoot::new(1, 4), 1), (UnityRoot::new(3, 4), 1)]),
        ),
        (
            (&expected_irreps[5], &tc4),
            Character::new(&[(UnityRoot::new(2, 4), 1)]),
        ),
        (
            (&expected_irreps[6], &tc4),
            Character::new(&[(UnityRoot::new(2, 4), 1)]),
        ),
        (
            (&expected_irreps[7], &tc4),
            Character::new(&[(UnityRoot::new(0, 4), 1)]),
        ),
        (
            (&expected_irreps[8], &tc4),
            Character::new(&[(UnityRoot::new(0, 4), 1)]),
        ),
        (
            (&expected_irreps[9], &tc4),
            Character::new(&[(UnityRoot::new(1, 4), 1), (UnityRoot::new(3, 4), 1)]),
        ),
    ]);
    test_chartab_magnetic_group(
        &mol,
        thresh,
        "C4v + θ·C4v",
        &expected_irreps,
        Some(expected_chars),
    );
}

#[test]
fn test_chartab_symmetric_cpnico_c5v() {
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
    let c5 = SymmetryClassSymbol::<SymmetryOperation>::new("2||C5||", None).unwrap();
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
    test_chartab_ordinary_group(&mol, thresh, "C5v", &expected_irreps, Some(expected_chars));
}

#[test]
fn test_chartab_symmetric_cpnico_grey_c5v() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/cpnico.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||E|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||E|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|A|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|A|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|E|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|E|_(2)|").unwrap(),
    ];
    let tc5 = SymmetryClassSymbol::<SymmetryOperation>::new("2||θ·C5||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &tc5),
            Character::new(&[(UnityRoot::new(0, 5), 1)]),
        ),
        (
            (&expected_irreps[1], &tc5),
            Character::new(&[(UnityRoot::new(0, 5), 1)]),
        ),
        (
            (&expected_irreps[2], &tc5),
            Character::new(&[(UnityRoot::new(1, 5), 1), (UnityRoot::new(4, 5), 1)]),
        ),
        (
            (&expected_irreps[3], &tc5),
            Character::new(&[(UnityRoot::new(2, 5), 1), (UnityRoot::new(3, 5), 1)]),
        ),
        (
            (&expected_irreps[4], &tc5),
            Character::new(&[(UnityRoot::new(5, 10), 1)]),
        ),
        (
            (&expected_irreps[5], &tc5),
            Character::new(&[(UnityRoot::new(5, 10), 1)]),
        ),
        (
            (&expected_irreps[6], &tc5),
            Character::new(&[(UnityRoot::new(3, 10), 1), (UnityRoot::new(7, 10), 1)]),
        ),
        (
            (&expected_irreps[7], &tc5),
            Character::new(&[(UnityRoot::new(1, 10), 1), (UnityRoot::new(9, 10), 1)]),
        ),
    ]);
    test_chartab_magnetic_group(
        &mol,
        thresh,
        "C5v + θ·C5v",
        &expected_irreps,
        Some(expected_chars),
    );
}

#[test]
fn test_chartab_symmetric_staggered_ferrocene_electric_field_c5v() {
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
    let c5 = SymmetryClassSymbol::<SymmetryOperation>::new("2||C5||", None).unwrap();
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
    test_chartab_ordinary_group(&mol, thresh, "C5v", &expected_irreps, Some(expected_chars));
}

#[test]
fn test_chartab_symmetric_staggered_ferrocene_electric_field_grey_c5v() {
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
        MullikenIrrepSymbol::new("|^(m)|A|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|A|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|E|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|E|_(2)|").unwrap(),
    ];
    let tc5 = SymmetryClassSymbol::<SymmetryOperation>::new("2||θ·C5||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &tc5),
            Character::new(&[(UnityRoot::new(0, 5), 1)]),
        ),
        (
            (&expected_irreps[1], &tc5),
            Character::new(&[(UnityRoot::new(0, 5), 1)]),
        ),
        (
            (&expected_irreps[2], &tc5),
            Character::new(&[(UnityRoot::new(1, 5), 1), (UnityRoot::new(4, 5), 1)]),
        ),
        (
            (&expected_irreps[3], &tc5),
            Character::new(&[(UnityRoot::new(2, 5), 1), (UnityRoot::new(3, 5), 1)]),
        ),
        (
            (&expected_irreps[4], &tc5),
            Character::new(&[(UnityRoot::new(5, 10), 1)]),
        ),
        (
            (&expected_irreps[5], &tc5),
            Character::new(&[(UnityRoot::new(5, 10), 1)]),
        ),
        (
            (&expected_irreps[6], &tc5),
            Character::new(&[(UnityRoot::new(3, 10), 1), (UnityRoot::new(7, 10), 1)]),
        ),
        (
            (&expected_irreps[7], &tc5),
            Character::new(&[(UnityRoot::new(1, 10), 1), (UnityRoot::new(9, 10), 1)]),
        ),
    ]);
    test_chartab_magnetic_group(
        &mol,
        thresh,
        "C5v + θ·C5v",
        &expected_irreps,
        Some(expected_chars),
    );
}

#[test]
fn test_chartab_symmetric_c60_electric_field_c5v() {
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
    let c5 = SymmetryClassSymbol::<SymmetryOperation>::new("2||C5||", None).unwrap();
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
    test_chartab_ordinary_group(&mol, thresh, "C5v", &expected_irreps, Some(expected_chars));
}

#[test]
fn test_chartab_symmetric_c60_electric_field_grey_c5v() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c60.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||E|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||E|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|A|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|A|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|E|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|E|_(2)|").unwrap(),
    ];
    let tc5 = SymmetryClassSymbol::<SymmetryOperation>::new("2||θ·C5||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &tc5),
            Character::new(&[(UnityRoot::new(0, 5), 1)]),
        ),
        (
            (&expected_irreps[1], &tc5),
            Character::new(&[(UnityRoot::new(0, 5), 1)]),
        ),
        (
            (&expected_irreps[2], &tc5),
            Character::new(&[(UnityRoot::new(1, 5), 1), (UnityRoot::new(4, 5), 1)]),
        ),
        (
            (&expected_irreps[3], &tc5),
            Character::new(&[(UnityRoot::new(2, 5), 1), (UnityRoot::new(3, 5), 1)]),
        ),
        (
            (&expected_irreps[4], &tc5),
            Character::new(&[(UnityRoot::new(5, 10), 1)]),
        ),
        (
            (&expected_irreps[5], &tc5),
            Character::new(&[(UnityRoot::new(5, 10), 1)]),
        ),
        (
            (&expected_irreps[6], &tc5),
            Character::new(&[(UnityRoot::new(3, 10), 1), (UnityRoot::new(7, 10), 1)]),
        ),
        (
            (&expected_irreps[7], &tc5),
            Character::new(&[(UnityRoot::new(1, 10), 1), (UnityRoot::new(9, 10), 1)]),
        ),
    ]);
    test_chartab_magnetic_group(
        &mol,
        thresh,
        "C5v + θ·C5v",
        &expected_irreps,
        Some(expected_chars),
    );
}

#[test]
fn test_chartab_symmetric_b7_c6v() {
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
    let c6 = SymmetryClassSymbol::<SymmetryOperation>::new("2||C6||", None).unwrap();
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
    test_chartab_ordinary_group(&mol, thresh, "C6v", &expected_irreps, Some(expected_chars));
}

#[test]
fn test_chartab_symmetric_b7_grey_c6v() {
    // env_logger::init();
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
        MullikenIrrepSymbol::new("|^(m)|A|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|A|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|B|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|B|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|E|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|E|_(2)|").unwrap(),
    ];
    let tc6 = SymmetryClassSymbol::<SymmetryOperation>::new("2||θ·C6||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &tc6),
            Character::new(&[(UnityRoot::new(0, 6), 1)]),
        ),
        (
            (&expected_irreps[1], &tc6),
            Character::new(&[(UnityRoot::new(0, 6), 1)]),
        ),
        (
            (&expected_irreps[2], &tc6),
            Character::new(&[(UnityRoot::new(3, 6), 1)]),
        ),
        (
            (&expected_irreps[3], &tc6),
            Character::new(&[(UnityRoot::new(3, 6), 1)]),
        ),
        (
            (&expected_irreps[4], &tc6),
            Character::new(&[(UnityRoot::new(1, 6), 1), (UnityRoot::new(5, 6), 1)]),
        ),
        (
            (&expected_irreps[5], &tc6),
            Character::new(&[(UnityRoot::new(2, 6), 1), (UnityRoot::new(4, 6), 1)]),
        ),
        (
            (&expected_irreps[6], &tc6),
            Character::new(&[(UnityRoot::new(3, 6), 1)]),
        ),
        (
            (&expected_irreps[7], &tc6),
            Character::new(&[(UnityRoot::new(3, 6), 1)]),
        ),
        (
            (&expected_irreps[8], &tc6),
            Character::new(&[(UnityRoot::new(0, 6), 1)]),
        ),
        (
            (&expected_irreps[9], &tc6),
            Character::new(&[(UnityRoot::new(0, 6), 1)]),
        ),
        (
            (&expected_irreps[10], &tc6),
            Character::new(&[(UnityRoot::new(2, 6), 1), (UnityRoot::new(4, 6), 1)]),
        ),
        (
            (&expected_irreps[11], &tc6),
            Character::new(&[(UnityRoot::new(1, 6), 1), (UnityRoot::new(5, 6), 1)]),
        ),
    ]);
    test_chartab_magnetic_group(
        &mol,
        thresh,
        "C6v + θ·C6v",
        &expected_irreps,
        Some(expected_chars),
    );
}

#[test]
fn test_chartab_symmetric_au26_electric_field_c6v() {
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
    let c6 = SymmetryClassSymbol::<SymmetryOperation>::new("2||C6||", None).unwrap();
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
    test_chartab_ordinary_group(&mol, thresh, "C6v", &expected_irreps, Some(expected_chars));
}

#[test]
fn test_chartab_symmetric_benzene_electric_field_c6v() {
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
    let c6 = SymmetryClassSymbol::<SymmetryOperation>::new("2||C6||", None).unwrap();
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
    test_chartab_ordinary_group(&mol, thresh, "C6v", &expected_irreps, Some(expected_chars));
}

#[test]
fn test_chartab_symmetric_au26_electric_field_grey_c6v() {
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
        MullikenIrrepSymbol::new("|^(m)|A|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|A|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|B|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|B|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|E|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|E|_(2)|").unwrap(),
    ];
    let tc6 = SymmetryClassSymbol::<SymmetryOperation>::new("2||θ·C6||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &tc6),
            Character::new(&[(UnityRoot::new(0, 6), 1)]),
        ),
        (
            (&expected_irreps[1], &tc6),
            Character::new(&[(UnityRoot::new(0, 6), 1)]),
        ),
        (
            (&expected_irreps[2], &tc6),
            Character::new(&[(UnityRoot::new(3, 6), 1)]),
        ),
        (
            (&expected_irreps[3], &tc6),
            Character::new(&[(UnityRoot::new(3, 6), 1)]),
        ),
        (
            (&expected_irreps[4], &tc6),
            Character::new(&[(UnityRoot::new(1, 6), 1), (UnityRoot::new(5, 6), 1)]),
        ),
        (
            (&expected_irreps[5], &tc6),
            Character::new(&[(UnityRoot::new(2, 6), 1), (UnityRoot::new(4, 6), 1)]),
        ),
        (
            (&expected_irreps[6], &tc6),
            Character::new(&[(UnityRoot::new(3, 6), 1)]),
        ),
        (
            (&expected_irreps[7], &tc6),
            Character::new(&[(UnityRoot::new(3, 6), 1)]),
        ),
        (
            (&expected_irreps[8], &tc6),
            Character::new(&[(UnityRoot::new(0, 6), 1)]),
        ),
        (
            (&expected_irreps[9], &tc6),
            Character::new(&[(UnityRoot::new(0, 6), 1)]),
        ),
        (
            (&expected_irreps[10], &tc6),
            Character::new(&[(UnityRoot::new(2, 6), 1), (UnityRoot::new(4, 6), 1)]),
        ),
        (
            (&expected_irreps[11], &tc6),
            Character::new(&[(UnityRoot::new(1, 6), 1), (UnityRoot::new(5, 6), 1)]),
        ),
    ]);
    test_chartab_magnetic_group(
        &mol,
        thresh,
        "C6v + θ·C6v",
        &expected_irreps,
        Some(expected_chars),
    );
}

#[test]
fn test_chartab_symmetric_arbitrary_half_sandwich_cnv() {
    /* The expected number of classes is deduced from the irrep structures of
     * the Cnv groups.
     * When n is even, the irreps are A1, A2, B1, B2, Ek where k = 1, ..., n/2 - 1.
     * When n is odd, the irreps are A1, A2, Ek where k = 1, ..., n//2.
     */
    for n in 3..=32 {
        let mol = template_molecules::gen_arbitrary_half_sandwich(n);
        let thresh = 1e-7;
        verify_cnv(&mol, thresh, n as usize);
    }
}

#[test]
fn test_chartab_symmetric_arbitrary_half_sandwich_grey_cnv() {
    /* The expected number of classes is deduced from the irrep structures of
     * the Cnv groups.
     * When n is even, the irreps are A1, A2, B1, B2, Ek where k = 1, ..., n/2 - 1.
     * When n is odd, the irreps are A1, A2, Ek where k = 1, ..., n//2.
     */
    for n in 3..=32 {
        let mol = template_molecules::gen_arbitrary_half_sandwich(n);
        let thresh = 1e-7;
        verify_grey_cnv(&mol, thresh, n as usize);
    }
}

/// Verifies the validity of the computed $`\mathcal{C}_{nv}`$ character table of irreps.
///
/// # Arguments
///
/// * `mol` - A reference to a [`Molecule`] structure.
/// * `thresh` - A threshold for symmetry detection.
/// * `n` - The value of $`n`$.
///
/// # Panics
///
/// Panics when any expected condition is not fulfilled.
fn verify_cnv(mol: &Molecule, thresh: f64, n: usize) {
    let expected_irreps = if n % 2 == 0 {
        let mut irreps = vec![
            MullikenIrrepSymbol::new("||A|_(1)|").unwrap(),
            MullikenIrrepSymbol::new("||A|_(2)|").unwrap(),
            MullikenIrrepSymbol::new("||B|_(1)|").unwrap(),
            MullikenIrrepSymbol::new("||B|_(2)|").unwrap(),
        ];
        if n == 4 {
            irreps.push(MullikenIrrepSymbol::new("||E||").unwrap());
        } else {
            irreps.extend(
                (1..n.div_euclid(2))
                    .map(|k| MullikenIrrepSymbol::new(&format!("||E|_({})|", k)).unwrap()),
            );
        }
        irreps
    } else {
        let mut irreps = vec![
            MullikenIrrepSymbol::new("||A|_(1)|").unwrap(),
            MullikenIrrepSymbol::new("||A|_(2)|").unwrap(),
        ];
        if n == 3 {
            irreps.push(MullikenIrrepSymbol::new("||E||").unwrap());
        } else {
            irreps.extend(
                (1..=n.div_euclid(2))
                    .map(|k| MullikenIrrepSymbol::new(&format!("||E|_({})|", k)).unwrap()),
            );
        };
        irreps
    };
    test_chartab_ordinary_group(
        &mol,
        thresh,
        format!("C{n}v").as_str(),
        &expected_irreps,
        None,
    );
}

/// Verifies the validity of the computed $`\mathcal{C}_{nv}`$ character table of irreps as a
/// subgroup of $`\mathcal{C}_{\infty v}`$.
///
/// # Arguments
///
/// * `mol` - A reference to a [`Molecule`] structure.
/// * `thresh` - A threshold for symmetry detection.
/// * `n` - The value of $`n`$.
///
/// # Panics
///
/// Panics when any expected condition is not fulfilled.
fn verify_cnv_from_cinfv(mol: &Molecule, thresh: f64, n: usize) {
    let expected_irreps = if n % 2 == 0 {
        let mut irreps = vec![
            MullikenIrrepSymbol::new("||A|_(1)|").unwrap(),
            MullikenIrrepSymbol::new("||A|_(2)|").unwrap(),
            MullikenIrrepSymbol::new("||B|_(1)|").unwrap(),
            MullikenIrrepSymbol::new("||B|_(2)|").unwrap(),
        ];
        if n == 4 {
            irreps.push(MullikenIrrepSymbol::new("||E||").unwrap());
        } else {
            irreps.extend(
                (1..n.div_euclid(2))
                    .map(|k| MullikenIrrepSymbol::new(&format!("||E|_({})|", k)).unwrap()),
            );
        }
        irreps
    } else {
        let mut irreps = vec![
            MullikenIrrepSymbol::new("||A|_(1)|").unwrap(),
            MullikenIrrepSymbol::new("||A|_(2)|").unwrap(),
        ];
        if n == 3 {
            irreps.push(MullikenIrrepSymbol::new("||E||").unwrap());
        } else {
            irreps.extend(
                (1..=n.div_euclid(2))
                    .map(|k| MullikenIrrepSymbol::new(&format!("||E|_({})|", k)).unwrap()),
            );
        };
        irreps
    };
    test_chartab_ordinary_group_from_infinite(
        &mol,
        n as u32,
        thresh,
        format!("C∞v > C{n}v").as_str(),
        &expected_irreps,
        None,
    );
}

/// Verifies the validity of the computed $`\mathcal{C}_{nv} + \theta\mathcal{C}_{nv}`$
/// character table of irreps.
///
/// # Arguments
///
/// * `mol` - A reference to a [`Molecule`] structure.
/// * `thresh` - A threshold for symmetry detection.
/// * `n` - The value of $`n`$.
///
/// # Panics
///
/// Panics when any expected condition is not fulfilled.
fn verify_grey_cnv(mol: &Molecule, thresh: f64, n: usize) {
    let expected_irreps = if n % 2 == 0 {
        let mut irreps = vec![
            MullikenIrrepSymbol::new("||A|_(1)|").unwrap(),
            MullikenIrrepSymbol::new("||A|_(2)|").unwrap(),
            MullikenIrrepSymbol::new("||B|_(1)|").unwrap(),
            MullikenIrrepSymbol::new("||B|_(2)|").unwrap(),
        ];
        if n == 4 {
            irreps.push(MullikenIrrepSymbol::new("||E||").unwrap());
        } else {
            irreps.extend(
                (1..n.div_euclid(2))
                    .map(|k| MullikenIrrepSymbol::new(&format!("||E|_({})|", k)).unwrap()),
            );
        }
        let m_irreps = irreps
            .iter()
            .map(|irrep| {
                MullikenIrrepSymbol::new(&format!("|^(m)|{}|_({})|", irrep.main(), irrep.postsub()))
                    .unwrap()
            })
            .collect_vec();
        irreps.extend(m_irreps);
        irreps
    } else {
        let mut irreps = vec![
            MullikenIrrepSymbol::new("||A|_(1)|").unwrap(),
            MullikenIrrepSymbol::new("||A|_(2)|").unwrap(),
        ];
        if n == 3 {
            irreps.push(MullikenIrrepSymbol::new("||E||").unwrap());
        } else {
            irreps.extend(
                (1..=n.div_euclid(2))
                    .map(|k| MullikenIrrepSymbol::new(&format!("||E|_({})|", k)).unwrap()),
            );
        };
        let m_irreps = irreps
            .iter()
            .map(|irrep| {
                MullikenIrrepSymbol::new(&format!("|^(m)|{}|_({})|", irrep.main(), irrep.postsub()))
                    .unwrap()
            })
            .collect_vec();
        irreps.extend(m_irreps);
        irreps
    };
    test_chartab_magnetic_group(
        mol,
        thresh,
        format!("C{n}v + θ·C{n}v").as_str(),
        &expected_irreps,
        None,
    );
}

/// Verifies the validity of the computed $`\mathcal{C}_{\infty v} + \theta\mathcal{C}_{\infty v}`$
/// character table of irreps.
///
/// # Arguments
///
/// * `mol` - A reference to a [`Molecule`] structure.
/// * `thresh` - A threshold for symmetry detection.
/// * `n` - The value of $`n`$.
///
/// # Panics
///
/// Panics when any expected condition is not fulfilled.
fn verify_grey_cnv_from_grey_cinfv(mol: &Molecule, thresh: f64, n: usize) {
    let expected_irreps = if n % 2 == 0 {
        let mut irreps = vec![
            MullikenIrrepSymbol::new("||A|_(1)|").unwrap(),
            MullikenIrrepSymbol::new("||A|_(2)|").unwrap(),
            MullikenIrrepSymbol::new("||B|_(1)|").unwrap(),
            MullikenIrrepSymbol::new("||B|_(2)|").unwrap(),
        ];
        if n == 4 {
            irreps.push(MullikenIrrepSymbol::new("||E||").unwrap());
        } else {
            irreps.extend(
                (1..n.div_euclid(2))
                    .map(|k| MullikenIrrepSymbol::new(&format!("||E|_({})|", k)).unwrap()),
            );
        }
        let m_irreps = irreps
            .iter()
            .map(|irrep| {
                MullikenIrrepSymbol::new(&format!("|^(m)|{}|_({})|", irrep.main(), irrep.postsub()))
                    .unwrap()
            })
            .collect_vec();
        irreps.extend(m_irreps);
        irreps
    } else {
        let mut irreps = vec![
            MullikenIrrepSymbol::new("||A|_(1)|").unwrap(),
            MullikenIrrepSymbol::new("||A|_(2)|").unwrap(),
        ];
        if n == 3 {
            irreps.push(MullikenIrrepSymbol::new("||E||").unwrap());
        } else {
            irreps.extend(
                (1..=n.div_euclid(2))
                    .map(|k| MullikenIrrepSymbol::new(&format!("||E|_({})|", k)).unwrap()),
            );
        };
        let m_irreps = irreps
            .iter()
            .map(|irrep| {
                MullikenIrrepSymbol::new(&format!("|^(m)|{}|_({})|", irrep.main(), irrep.postsub()))
                    .unwrap()
            })
            .collect_vec();
        irreps.extend(m_irreps);
        irreps
    };
    test_chartab_magnetic_group_from_infinite(
        mol,
        n as u32,
        thresh,
        format!("C∞v + θ·C∞v > C{n}v + θ·C{n}v").as_str(),
        &expected_irreps,
        None,
    );
}

/*
Cnh
*/
#[test]
fn test_chartab_symmetric_bf3_magnetic_field_c3h() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/bf3.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|^(')|").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|Γ|^(')|").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|Γ|^(')|").unwrap(),
        MullikenIrrepSymbol::new("||A|^('')|").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|Γ|^('')|").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|Γ|^('')|").unwrap(),
    ];
    let c3 = SymmetryClassSymbol::<SymmetryOperation>::new("1||C3||", None).unwrap();
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
    test_chartab_ordinary_group(&mol, thresh, "C3h", &expected_irreps, Some(expected_chars));
}

#[test]
fn test_chartab_symmetric_bf3_magnetic_field_bw_d3h_c3h() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/bf3.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|^(')_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||A|^(')_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||E|^(')|").unwrap(),
        MullikenIrrepSymbol::new("||A|^('')_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||A|^('')_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||E|^('')|").unwrap(),
    ];
    let c3 = SymmetryClassSymbol::<SymmetryOperation>::new("2||C3||", None).unwrap();
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
    test_chartab_magnetic_group(&mol, thresh, "D3h", &expected_irreps, Some(expected_chars));
}

#[test]
fn test_chartab_symmetric_xef4_magnetic_field_c4h() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/xef4.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, -1.0)));
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|Γ|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|Γ|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(u)|").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|Γ|_(u)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(u)|").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|Γ|_(u)|").unwrap(),
    ];
    let c4 = SymmetryClassSymbol::<SymmetryOperation>::new("1||C4||", None).unwrap();
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
    test_chartab_ordinary_group(&mol, thresh, "C4h", &expected_irreps, Some(expected_chars));
}

#[test]
fn test_chartab_symmetric_xef4_magnetic_field_bw_d4h_c4h() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/xef4.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, -1.0)));
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
    let c4 = SymmetryClassSymbol::<SymmetryOperation>::new("2||C4||", None).unwrap();
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
    test_chartab_magnetic_group(&mol, thresh, "D4h", &expected_irreps, Some(expected_chars));
}

#[test]
fn test_chartab_symmetric_vf6_magnetic_field_c4h() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/vf6.xyz");
    let thresh = 1e-12;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|Γ|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|Γ|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(u)|").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|Γ|_(u)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(u)|").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|Γ|_(u)|").unwrap(),
    ];
    let c4 = SymmetryClassSymbol::<SymmetryOperation>::new("1||C4||", None).unwrap();
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
    test_chartab_ordinary_group(&mol, thresh, "C4h", &expected_irreps, Some(expected_chars));
}

#[test]
fn test_chartab_symmetric_vf6_magnetic_field_bw_d4h_c4h() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/vf6.xyz");
    let thresh = 1e-12;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
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
    let c4 = SymmetryClassSymbol::<SymmetryOperation>::new("2||C4||", None).unwrap();
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
    test_chartab_magnetic_group(&mol, thresh, "D4h", &expected_irreps, Some(expected_chars));
}

#[test]
fn test_chartab_symmetric_h8_magnetic_field_c4h() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/h8.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, 1e-7);
    mol.set_magnetic_field(Some(0.2 * Vector3::new(0.0, 0.0, 1.0)));
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|Γ|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|Γ|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(u)|").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|Γ|_(u)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(u)|").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|Γ|_(u)|").unwrap(),
    ];
    let c4 = SymmetryClassSymbol::<SymmetryOperation>::new("1||C4||", None).unwrap();
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
    test_chartab_ordinary_group(&mol, thresh, "C4h", &expected_irreps, Some(expected_chars));
}

#[test]
fn test_chartab_symmetric_h8_magnetic_field_bw_d4h_c4h() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/h8.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, 1e-7);
    mol.set_magnetic_field(Some(0.2 * Vector3::new(0.0, 0.0, 1.0)));
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
    let c4 = SymmetryClassSymbol::<SymmetryOperation>::new("2||C4||", None).unwrap();
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
    test_chartab_magnetic_group(&mol, thresh, "D4h", &expected_irreps, Some(expected_chars));
}

#[test]
fn test_chartab_symmetric_eclipsed_ferrocene_magnetic_field_c5h() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/eclipsed_ferrocene.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, -1.0)));
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|^(')|").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|Γ|^(')_(1)|").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|Γ|^(')_(2)|").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|Γ|^(')_(2)|").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|Γ|^(')_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||A|^('')|").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|Γ|^('')_(1)|").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|Γ|^('')_(2)|").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|Γ|^('')_(2)|").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|Γ|^('')_(1)|").unwrap(),
    ];
    let c5 = SymmetryClassSymbol::<SymmetryOperation>::new("1||C5||", None).unwrap();
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
    test_chartab_ordinary_group(&mol, thresh, "C5h", &expected_irreps, Some(expected_chars));
}

#[test]
fn test_chartab_symmetric_eclipsed_ferrocene_magnetic_field_bw_d5h_c5h() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/eclipsed_ferrocene.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, -1.0)));
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
    let c5 = SymmetryClassSymbol::<SymmetryOperation>::new("2||C5||", None).unwrap();
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
    test_chartab_magnetic_group(&mol, thresh, "D5h", &expected_irreps, Some(expected_chars));
}

#[test]
fn test_chartab_symmetric_benzene_magnetic_field_c6h() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/benzene.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(1.0, 0.0, 0.0)));
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|Γ|_(1g)|").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|Γ|_(2g)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|Γ|_(2g)|").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|Γ|_(1g)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(u)|").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|Γ|_(1u)|").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|Γ|_(2u)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(u)|").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|Γ|_(2u)|").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|Γ|_(1u)|").unwrap(),
    ];
    let c6 = SymmetryClassSymbol::<SymmetryOperation>::new("1||C6||", None).unwrap();
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
    test_chartab_ordinary_group(&mol, thresh, "C6h", &expected_irreps, Some(expected_chars));
}

#[test]
fn test_chartab_symmetric_benzene_magnetic_field_bw_d6h_c6h() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/benzene.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(1.0, 0.0, 0.0)));
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
    let c6 = SymmetryClassSymbol::<SymmetryOperation>::new("2||C6||", None).unwrap();
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
    test_chartab_magnetic_group(&mol, thresh, "D6h", &expected_irreps, Some(expected_chars));
}

#[test]
fn test_chartab_symmetric_arbitrary_eclipsed_sandwich_magnetic_field_cnh() {
    // env_logger::init();
    for n in 3..=20 {
        let mut mol = template_molecules::gen_arbitrary_eclipsed_sandwich(n);
        let thresh = 1e-7;
        mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 0.1)));
        verify_cnh(&mol, thresh, n as usize);
    }
}

#[test]
fn test_chartab_symmetric_arbitrary_eclipsed_sandwich_magnetic_field_bw_dnh_cnh() {
    // env_logger::init();
    for n in 3..=20 {
        let mut mol = template_molecules::gen_arbitrary_eclipsed_sandwich(n);
        let thresh = 1e-7;
        mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 0.1)));
        verify_bw_dnh_cnh(&mol, thresh, n as usize);
    }
}

/// Verifies the validity of the computed $`\mathcal{C}_{nh}`$ character table of irreps.
///
/// # Arguments
///
/// * `mol` - A reference to a [`Molecule`] structure.
/// * `thresh` - A threshold for symmetry detection.
/// * `n` - The value of $`n`$.
///
/// # Panics
///
/// Panics when any expected condition is not fulfilled.
fn verify_cnh(mol: &Molecule, thresh: f64, n: usize) {
    let expected_irreps =
        if n % 2 == 0 {
            let mut irreps = vec![MullikenIrrepSymbol::new("||A|_(g)|").unwrap()];
            if n == 4 {
                irreps.push(MullikenIrrepSymbol::new("|_(a)|Γ|_(g)|").unwrap());
            } else {
                irreps
                    .extend((1..n.div_euclid(2)).map(|k| {
                        MullikenIrrepSymbol::new(&format!("|_(a)|Γ|_({}g)|", k)).unwrap()
                    }));
            }
            irreps.push(MullikenIrrepSymbol::new("||B|_(g)|").unwrap());
            if n == 4 {
                irreps.push(MullikenIrrepSymbol::new("|_(b)|Γ|_(g)|").unwrap());
            } else {
                irreps.extend(
                    (1..n.div_euclid(2))
                        .rev()
                        .map(|k| MullikenIrrepSymbol::new(&format!("|_(b)|Γ|_({}g)|", k)).unwrap()),
                );
            }
            irreps.push(MullikenIrrepSymbol::new("||A|_(u)|").unwrap());
            if n == 4 {
                irreps.push(MullikenIrrepSymbol::new("|_(a)|Γ|_(u)|").unwrap());
            } else {
                irreps
                    .extend((1..n.div_euclid(2)).map(|k| {
                        MullikenIrrepSymbol::new(&format!("|_(a)|Γ|_({}u)|", k)).unwrap()
                    }));
            }
            irreps.push(MullikenIrrepSymbol::new("||B|_(u)|").unwrap());
            if n == 4 {
                irreps.push(MullikenIrrepSymbol::new("|_(b)|Γ|_(u)|").unwrap());
            } else {
                irreps.extend(
                    (1..n.div_euclid(2))
                        .rev()
                        .map(|k| MullikenIrrepSymbol::new(&format!("|_(b)|Γ|_({}u)|", k)).unwrap()),
                );
            }
            irreps
        } else {
            let mut irreps = vec![MullikenIrrepSymbol::new("||A|^(')|").unwrap()];
            if n == 3 {
                irreps.push(MullikenIrrepSymbol::new("|_(a)|Γ|^(')|").unwrap());
                irreps.push(MullikenIrrepSymbol::new("|_(b)|Γ|^(')|").unwrap());
            } else {
                irreps.extend(
                    (1..=n.div_euclid(2)).map(|k| {
                        MullikenIrrepSymbol::new(&format!("|_(a)|Γ|^(')_({})|", k)).unwrap()
                    }),
                );
                irreps.extend(
                    (1..=n.div_euclid(2)).rev().map(|k| {
                        MullikenIrrepSymbol::new(&format!("|_(b)|Γ|^(')_({})|", k)).unwrap()
                    }),
                );
            }
            irreps.push(MullikenIrrepSymbol::new("||A|^('')|").unwrap());
            if n == 3 {
                irreps.push(MullikenIrrepSymbol::new("|_(a)|Γ|^('')|").unwrap());
                irreps.push(MullikenIrrepSymbol::new("|_(b)|Γ|^('')|").unwrap());
            } else {
                irreps.extend((1..=n.div_euclid(2)).map(|k| {
                    MullikenIrrepSymbol::new(&format!("|_(a)|Γ|^('')_({})|", k)).unwrap()
                }));
                irreps.extend((1..=n.div_euclid(2)).rev().map(|k| {
                    MullikenIrrepSymbol::new(&format!("|_(b)|Γ|^('')_({})|", k)).unwrap()
                }));
            }
            irreps
        };
    test_chartab_ordinary_group(
        &mol,
        thresh,
        format!("C{n}h").as_str(),
        &expected_irreps,
        None,
    );
}

/// Verifies the validity of the computed $`\mathcal{D}_{nh}(\mathcal{C}_{nh})`$ character table of
/// irreps.
///
/// # Arguments
///
/// * `mol` - A reference to a [`Molecule`] structure.
/// * `thresh` - A threshold for symmetry detection.
/// * `n` - The value of $`n`$.
///
/// # Panics
///
/// Panics when any expected condition is not fulfilled.
fn verify_bw_dnh_cnh(mol: &Molecule, thresh: f64, n: usize) {
    let expected_irreps =
        if n % 2 == 0 {
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
    test_chartab_magnetic_group(
        &mol,
        thresh,
        format!("D{n}h").as_str(),
        &expected_irreps,
        None,
    );
}

/*
Dn
*/
#[test]
fn test_chartab_symmetric_triphenyl_radical_d3() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/triphenylradical.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||E||").unwrap(),
    ];
    let c3 = SymmetryClassSymbol::<SymmetryOperation>::new("2||C3||", None).unwrap();
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
    test_chartab_ordinary_group(&mol, thresh, "D3", &expected_irreps, Some(expected_chars));
}

#[test]
fn test_chartab_symmetric_triphenyl_radical_grey_d3() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/triphenylradical.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||E||").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|A|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|A|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|E||").unwrap(),
    ];
    let tc3 = SymmetryClassSymbol::<SymmetryOperation>::new("2||θ·C3||", None).unwrap();
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
            Character::new(&[(UnityRoot::new(1, 3), 1), (UnityRoot::new(2, 3), 1)]),
        ),
        (
            (&expected_irreps[4], &tc3),
            Character::new(&[(UnityRoot::new(1, 3), 1), (UnityRoot::new(2, 3), 1)]),
        ),
        (
            (&expected_irreps[5], &tc3),
            Character::new(&[(UnityRoot::new(0, 3), 1)]),
        ),
    ]);
    test_chartab_magnetic_group(
        &mol,
        thresh,
        "D3 + θ·D3",
        &expected_irreps,
        Some(expected_chars),
    );
}

#[test]
fn test_chartab_symmetric_h8_twisted_d4() {
    let thresh = 1e-7;
    let mol = template_molecules::gen_twisted_h8(0.1);
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||E||").unwrap(),
    ];
    let c4 = SymmetryClassSymbol::<SymmetryOperation>::new("2||C4||", None).unwrap();
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
    test_chartab_ordinary_group(&mol, thresh, "D4", &expected_irreps, Some(expected_chars));
}

#[test]
fn test_chartab_symmetric_h8_twisted_grey_d4() {
    let thresh = 1e-7;
    let mol = template_molecules::gen_twisted_h8(0.1);
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||E||").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|A|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|A|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|B|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|B|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|E||").unwrap(),
    ];
    let tc4 = SymmetryClassSymbol::<SymmetryOperation>::new("2||θ·C4||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &tc4),
            Character::new(&[(UnityRoot::new(0, 4), 1)]),
        ),
        (
            (&expected_irreps[1], &tc4),
            Character::new(&[(UnityRoot::new(0, 4), 1)]),
        ),
        (
            (&expected_irreps[2], &tc4),
            Character::new(&[(UnityRoot::new(2, 4), 1)]),
        ),
        (
            (&expected_irreps[3], &tc4),
            Character::new(&[(UnityRoot::new(2, 4), 1)]),
        ),
        (
            (&expected_irreps[4], &tc4),
            Character::new(&[(UnityRoot::new(1, 4), 1), (UnityRoot::new(3, 4), 1)]),
        ),
        (
            (&expected_irreps[5], &tc4),
            Character::new(&[(UnityRoot::new(2, 4), 1)]),
        ),
        (
            (&expected_irreps[6], &tc4),
            Character::new(&[(UnityRoot::new(2, 4), 1)]),
        ),
        (
            (&expected_irreps[7], &tc4),
            Character::new(&[(UnityRoot::new(0, 4), 1)]),
        ),
        (
            (&expected_irreps[8], &tc4),
            Character::new(&[(UnityRoot::new(0, 4), 1)]),
        ),
        (
            (&expected_irreps[9], &tc4),
            Character::new(&[(UnityRoot::new(1, 4), 1), (UnityRoot::new(3, 4), 1)]),
        ),
    ]);
    test_chartab_magnetic_group(
        &mol,
        thresh,
        "D4 + θ·D4",
        &expected_irreps,
        Some(expected_chars),
    );
}

#[test]
fn test_chartab_symmetric_c5ph5_d5() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c5ph5.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||E|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||E|_(2)|").unwrap(),
    ];
    let c5 = SymmetryClassSymbol::<SymmetryOperation>::new("2||C5||", None).unwrap();
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
    test_chartab_ordinary_group(&mol, thresh, "D5", &expected_irreps, Some(expected_chars));
}

#[test]
fn test_chartab_symmetric_c5ph5_grey_d5() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c5ph5.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||E|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||E|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|A|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|A|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|E|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|E|_(2)|").unwrap(),
    ];
    let tc5 = SymmetryClassSymbol::<SymmetryOperation>::new("2||θ·C5||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &tc5),
            Character::new(&[(UnityRoot::new(0, 5), 1)]),
        ),
        (
            (&expected_irreps[1], &tc5),
            Character::new(&[(UnityRoot::new(0, 5), 1)]),
        ),
        (
            (&expected_irreps[2], &tc5),
            Character::new(&[(UnityRoot::new(1, 5), 1), (UnityRoot::new(4, 5), 1)]),
        ),
        (
            (&expected_irreps[3], &tc5),
            Character::new(&[(UnityRoot::new(2, 5), 1), (UnityRoot::new(3, 5), 1)]),
        ),
        (
            (&expected_irreps[4], &tc5),
            Character::new(&[(UnityRoot::new(5, 10), 1)]),
        ),
        (
            (&expected_irreps[5], &tc5),
            Character::new(&[(UnityRoot::new(5, 10), 1)]),
        ),
        (
            (&expected_irreps[6], &tc5),
            Character::new(&[(UnityRoot::new(3, 10), 1), (UnityRoot::new(7, 10), 1)]),
        ),
        (
            (&expected_irreps[7], &tc5),
            Character::new(&[(UnityRoot::new(1, 10), 1), (UnityRoot::new(9, 10), 1)]),
        ),
    ]);
    test_chartab_magnetic_group(
        &mol,
        thresh,
        "D5 + θ·D5",
        &expected_irreps,
        Some(expected_chars),
    );
}

#[test]
fn test_chartab_symmetric_c6ph6_d6() {
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
    let c6 = SymmetryClassSymbol::<SymmetryOperation>::new("2||C6||", None).unwrap();
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
    test_chartab_ordinary_group(&mol, thresh, "D6", &expected_irreps, Some(expected_chars));
}

#[test]
fn test_chartab_symmetric_c6ph6_grey_d6() {
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
        MullikenIrrepSymbol::new("|^(m)|A|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|A|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|B|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|B|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|E|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|E|_(2)|").unwrap(),
    ];
    let tc6 = SymmetryClassSymbol::<SymmetryOperation>::new("2||θ·C6||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &tc6),
            Character::new(&[(UnityRoot::new(0, 6), 1)]),
        ),
        (
            (&expected_irreps[1], &tc6),
            Character::new(&[(UnityRoot::new(0, 6), 1)]),
        ),
        (
            (&expected_irreps[2], &tc6),
            Character::new(&[(UnityRoot::new(3, 6), 1)]),
        ),
        (
            (&expected_irreps[3], &tc6),
            Character::new(&[(UnityRoot::new(3, 6), 1)]),
        ),
        (
            (&expected_irreps[4], &tc6),
            Character::new(&[(UnityRoot::new(1, 6), 1), (UnityRoot::new(5, 6), 1)]),
        ),
        (
            (&expected_irreps[5], &tc6),
            Character::new(&[(UnityRoot::new(2, 6), 1), (UnityRoot::new(4, 6), 1)]),
        ),
        (
            (&expected_irreps[6], &tc6),
            Character::new(&[(UnityRoot::new(3, 6), 1)]),
        ),
        (
            (&expected_irreps[7], &tc6),
            Character::new(&[(UnityRoot::new(3, 6), 1)]),
        ),
        (
            (&expected_irreps[8], &tc6),
            Character::new(&[(UnityRoot::new(0, 6), 1)]),
        ),
        (
            (&expected_irreps[9], &tc6),
            Character::new(&[(UnityRoot::new(0, 6), 1)]),
        ),
        (
            (&expected_irreps[10], &tc6),
            Character::new(&[(UnityRoot::new(2, 6), 1), (UnityRoot::new(4, 6), 1)]),
        ),
        (
            (&expected_irreps[11], &tc6),
            Character::new(&[(UnityRoot::new(1, 6), 1), (UnityRoot::new(5, 6), 1)]),
        ),
    ]);
    test_chartab_magnetic_group(
        &mol,
        thresh,
        "D6 + θ·D6",
        &expected_irreps,
        Some(expected_chars),
    );
}

#[test]
fn test_chartab_symmetric_arbitrary_twisted_sandwich_dn() {
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
            if n == 4 {
                irreps.push(MullikenIrrepSymbol::new("||E||").unwrap());
            } else {
                irreps.extend(
                    (1..n.div_euclid(2))
                        .map(|k| MullikenIrrepSymbol::new(&format!("||E|_({})|", k)).unwrap()),
                );
            }
            irreps
        } else {
            let mut irreps = vec![
                MullikenIrrepSymbol::new("||A|_(1)|").unwrap(),
                MullikenIrrepSymbol::new("||A|_(2)|").unwrap(),
            ];
            if n == 3 {
                irreps.push(MullikenIrrepSymbol::new("||E||").unwrap());
            } else {
                irreps.extend(
                    (1..=n.div_euclid(2))
                        .map(|k| MullikenIrrepSymbol::new(&format!("||E|_({})|", k)).unwrap()),
                );
            };
            irreps
        };
        test_chartab_ordinary_group(
            &mol,
            thresh,
            format!("D{n}").as_str(),
            &expected_irreps,
            None,
        );
    }
}

#[test]
fn test_chartab_symmetric_arbitrary_twisted_sandwich_grey_dn() {
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
            if n == 4 {
                irreps.push(MullikenIrrepSymbol::new("||E||").unwrap());
            } else {
                irreps.extend(
                    (1..n.div_euclid(2))
                        .map(|k| MullikenIrrepSymbol::new(&format!("||E|_({})|", k)).unwrap()),
                );
            }
            let m_irreps = irreps
                .iter()
                .map(|irrep| {
                    MullikenIrrepSymbol::new(&format!(
                        "|^(m)|{}|_({})|",
                        irrep.main(),
                        irrep.postsub()
                    ))
                    .unwrap()
                })
                .collect_vec();
            irreps.extend(m_irreps);
            irreps
        } else {
            let mut irreps = vec![
                MullikenIrrepSymbol::new("||A|_(1)|").unwrap(),
                MullikenIrrepSymbol::new("||A|_(2)|").unwrap(),
            ];
            if n == 3 {
                irreps.push(MullikenIrrepSymbol::new("||E||").unwrap());
            } else {
                irreps.extend(
                    (1..=n.div_euclid(2))
                        .map(|k| MullikenIrrepSymbol::new(&format!("||E|_({})|", k)).unwrap()),
                );
            };
            let m_irreps = irreps
                .iter()
                .map(|irrep| {
                    MullikenIrrepSymbol::new(&format!(
                        "|^(m)|{}|_({})|",
                        irrep.main(),
                        irrep.postsub()
                    ))
                    .unwrap()
                })
                .collect_vec();
            irreps.extend(m_irreps);
            irreps
        };
        test_chartab_magnetic_group(
            &mol,
            thresh,
            format!("D{n} + θ·D{n}").as_str(),
            &expected_irreps,
            None,
        );
    }
}

/*
Dnh
*/
#[test]
fn test_chartab_symmetric_bf3_d3h() {
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
    let c3 = SymmetryClassSymbol::<SymmetryOperation>::new("2||C3||", None).unwrap();
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
    test_chartab_ordinary_group(&mol, thresh, "D3h", &expected_irreps, Some(expected_chars));
}

#[test]
fn test_chartab_symmetric_bf3_grey_d3h() {
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
        MullikenIrrepSymbol::new("|^(m)|A|^(')_(1)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|A|^(')_(2)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|E|^(')|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|A|^('')_(1)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|A|^('')_(2)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|E|^('')|").unwrap(),
    ];
    let tc3 = SymmetryClassSymbol::<SymmetryOperation>::new("2||θ·C3||", None).unwrap();
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
            Character::new(&[(UnityRoot::new(0, 3), 1)]),
        ),
        (
            (&expected_irreps[4], &tc3),
            Character::new(&[(UnityRoot::new(0, 3), 1)]),
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
            Character::new(&[(UnityRoot::new(1, 3), 1), (UnityRoot::new(2, 3), 1)]),
        ),
        (
            (&expected_irreps[8], &tc3),
            Character::new(&[(UnityRoot::new(0, 3), 1)]),
        ),
        (
            (&expected_irreps[9], &tc3),
            Character::new(&[(UnityRoot::new(1, 3), 1), (UnityRoot::new(2, 3), 1)]),
        ),
        (
            (&expected_irreps[10], &tc3),
            Character::new(&[(UnityRoot::new(1, 3), 1), (UnityRoot::new(2, 3), 1)]),
        ),
        (
            (&expected_irreps[11], &tc3),
            Character::new(&[(UnityRoot::new(0, 3), 1)]),
        ),
    ]);
    test_chartab_magnetic_group(
        &mol,
        thresh,
        "D3h + θ·D3h",
        &expected_irreps,
        Some(expected_chars),
    );
}

#[test]
fn test_chartab_symmetric_xef4_d4h() {
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
    let c4 = SymmetryClassSymbol::<SymmetryOperation>::new("2||C4||", None).unwrap();
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
    test_chartab_ordinary_group(&mol, thresh, "D4h", &expected_irreps, Some(expected_chars));
}

#[test]
fn test_chartab_symmetric_xef4_grey_d4h() {
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
        MullikenIrrepSymbol::new("|^(m)|A|_(1g)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|A|_(2g)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|B|_(1g)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|B|_(2g)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|E|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|A|_(1u)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|A|_(2u)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|B|_(1u)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|B|_(2u)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|E|_(u)|").unwrap(),
    ];
    let tc4 = SymmetryClassSymbol::<SymmetryOperation>::new("2||θ·C4||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &tc4),
            Character::new(&[(UnityRoot::new(0, 4), 1)]),
        ),
        (
            (&expected_irreps[1], &tc4),
            Character::new(&[(UnityRoot::new(0, 4), 1)]),
        ),
        (
            (&expected_irreps[2], &tc4),
            Character::new(&[(UnityRoot::new(2, 4), 1)]),
        ),
        (
            (&expected_irreps[3], &tc4),
            Character::new(&[(UnityRoot::new(2, 4), 1)]),
        ),
        (
            (&expected_irreps[4], &tc4),
            Character::new(&[(UnityRoot::new(1, 4), 1), (UnityRoot::new(3, 4), 1)]),
        ),
        (
            (&expected_irreps[5], &tc4),
            Character::new(&[(UnityRoot::new(0, 4), 1)]),
        ),
        (
            (&expected_irreps[6], &tc4),
            Character::new(&[(UnityRoot::new(0, 4), 1)]),
        ),
        (
            (&expected_irreps[7], &tc4),
            Character::new(&[(UnityRoot::new(2, 4), 1)]),
        ),
        (
            (&expected_irreps[8], &tc4),
            Character::new(&[(UnityRoot::new(2, 4), 1)]),
        ),
        (
            (&expected_irreps[9], &tc4),
            Character::new(&[(UnityRoot::new(1, 4), 1), (UnityRoot::new(3, 4), 1)]),
        ),
        (
            (&expected_irreps[10], &tc4),
            Character::new(&[(UnityRoot::new(2, 4), 1)]),
        ),
        (
            (&expected_irreps[11], &tc4),
            Character::new(&[(UnityRoot::new(2, 4), 1)]),
        ),
        (
            (&expected_irreps[12], &tc4),
            Character::new(&[(UnityRoot::new(0, 4), 1)]),
        ),
        (
            (&expected_irreps[13], &tc4),
            Character::new(&[(UnityRoot::new(0, 4), 1)]),
        ),
        (
            (&expected_irreps[14], &tc4),
            Character::new(&[(UnityRoot::new(1, 4), 1), (UnityRoot::new(3, 4), 1)]),
        ),
        (
            (&expected_irreps[15], &tc4),
            Character::new(&[(UnityRoot::new(2, 4), 1)]),
        ),
        (
            (&expected_irreps[16], &tc4),
            Character::new(&[(UnityRoot::new(2, 4), 1)]),
        ),
        (
            (&expected_irreps[17], &tc4),
            Character::new(&[(UnityRoot::new(0, 4), 1)]),
        ),
        (
            (&expected_irreps[18], &tc4),
            Character::new(&[(UnityRoot::new(0, 4), 1)]),
        ),
        (
            (&expected_irreps[19], &tc4),
            Character::new(&[(UnityRoot::new(1, 4), 1), (UnityRoot::new(3, 4), 1)]),
        ),
    ]);
    test_chartab_magnetic_group(
        &mol,
        thresh,
        "D4h + θ·D4h",
        &expected_irreps,
        Some(expected_chars),
    );
}

#[test]
fn test_chartab_symmetric_h8_d4h() {
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
    let c4 = SymmetryClassSymbol::<SymmetryOperation>::new("2||C4||", None).unwrap();
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
    test_chartab_ordinary_group(&mol, thresh, "D4h", &expected_irreps, Some(expected_chars));
}

#[test]
fn test_chartab_symmetric_h8_grey_d4h() {
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
        MullikenIrrepSymbol::new("|^(m)|A|_(1g)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|A|_(2g)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|B|_(1g)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|B|_(2g)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|E|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|A|_(1u)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|A|_(2u)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|B|_(1u)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|B|_(2u)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|E|_(u)|").unwrap(),
    ];
    let tc4 = SymmetryClassSymbol::<SymmetryOperation>::new("2||θ·C4||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &tc4),
            Character::new(&[(UnityRoot::new(0, 4), 1)]),
        ),
        (
            (&expected_irreps[1], &tc4),
            Character::new(&[(UnityRoot::new(0, 4), 1)]),
        ),
        (
            (&expected_irreps[2], &tc4),
            Character::new(&[(UnityRoot::new(2, 4), 1)]),
        ),
        (
            (&expected_irreps[3], &tc4),
            Character::new(&[(UnityRoot::new(2, 4), 1)]),
        ),
        (
            (&expected_irreps[4], &tc4),
            Character::new(&[(UnityRoot::new(1, 4), 1), (UnityRoot::new(3, 4), 1)]),
        ),
        (
            (&expected_irreps[5], &tc4),
            Character::new(&[(UnityRoot::new(0, 4), 1)]),
        ),
        (
            (&expected_irreps[6], &tc4),
            Character::new(&[(UnityRoot::new(0, 4), 1)]),
        ),
        (
            (&expected_irreps[7], &tc4),
            Character::new(&[(UnityRoot::new(2, 4), 1)]),
        ),
        (
            (&expected_irreps[8], &tc4),
            Character::new(&[(UnityRoot::new(2, 4), 1)]),
        ),
        (
            (&expected_irreps[9], &tc4),
            Character::new(&[(UnityRoot::new(1, 4), 1), (UnityRoot::new(3, 4), 1)]),
        ),
        (
            (&expected_irreps[10], &tc4),
            Character::new(&[(UnityRoot::new(2, 4), 1)]),
        ),
        (
            (&expected_irreps[11], &tc4),
            Character::new(&[(UnityRoot::new(2, 4), 1)]),
        ),
        (
            (&expected_irreps[12], &tc4),
            Character::new(&[(UnityRoot::new(0, 4), 1)]),
        ),
        (
            (&expected_irreps[13], &tc4),
            Character::new(&[(UnityRoot::new(0, 4), 1)]),
        ),
        (
            (&expected_irreps[14], &tc4),
            Character::new(&[(UnityRoot::new(1, 4), 1), (UnityRoot::new(3, 4), 1)]),
        ),
        (
            (&expected_irreps[15], &tc4),
            Character::new(&[(UnityRoot::new(2, 4), 1)]),
        ),
        (
            (&expected_irreps[16], &tc4),
            Character::new(&[(UnityRoot::new(2, 4), 1)]),
        ),
        (
            (&expected_irreps[17], &tc4),
            Character::new(&[(UnityRoot::new(0, 4), 1)]),
        ),
        (
            (&expected_irreps[18], &tc4),
            Character::new(&[(UnityRoot::new(0, 4), 1)]),
        ),
        (
            (&expected_irreps[19], &tc4),
            Character::new(&[(UnityRoot::new(1, 4), 1), (UnityRoot::new(3, 4), 1)]),
        ),
    ]);
    test_chartab_magnetic_group(
        &mol,
        thresh,
        "D4h + θ·D4h",
        &expected_irreps,
        Some(expected_chars),
    );
}

#[test]
fn test_chartab_symmetric_eclipsed_ferrocene_d5h() {
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
    let c5 = SymmetryClassSymbol::<SymmetryOperation>::new("2||C5||", None).unwrap();
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
    test_chartab_ordinary_group(&mol, thresh, "D5h", &expected_irreps, Some(expected_chars));
}

#[test]
fn test_chartab_symmetric_eclipsed_ferrocene_grey_d5h() {
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
        MullikenIrrepSymbol::new("|^(m)|A|^(')_(1)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|A|^(')_(2)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|E|^(')_(1)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|E|^(')_(2)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|A|^('')_(1)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|A|^('')_(2)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|E|^('')_(1)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|E|^('')_(2)|").unwrap(),
    ];
    let tc5 = SymmetryClassSymbol::<SymmetryOperation>::new("2||θ·C5||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &tc5),
            Character::new(&[(UnityRoot::new(0, 5), 1)]),
        ),
        (
            (&expected_irreps[1], &tc5),
            Character::new(&[(UnityRoot::new(0, 5), 1)]),
        ),
        (
            (&expected_irreps[2], &tc5),
            Character::new(&[(UnityRoot::new(1, 5), 1), (UnityRoot::new(4, 5), 1)]),
        ),
        (
            (&expected_irreps[3], &tc5),
            Character::new(&[(UnityRoot::new(2, 5), 1), (UnityRoot::new(3, 5), 1)]),
        ),
        (
            (&expected_irreps[4], &tc5),
            Character::new(&[(UnityRoot::new(0, 5), 1)]),
        ),
        (
            (&expected_irreps[5], &tc5),
            Character::new(&[(UnityRoot::new(0, 5), 1)]),
        ),
        (
            (&expected_irreps[6], &tc5),
            Character::new(&[(UnityRoot::new(1, 5), 1), (UnityRoot::new(4, 5), 1)]),
        ),
        (
            (&expected_irreps[7], &tc5),
            Character::new(&[(UnityRoot::new(2, 5), 1), (UnityRoot::new(3, 5), 1)]),
        ),
        (
            (&expected_irreps[8], &tc5),
            Character::new(&[(UnityRoot::new(5, 10), 1)]),
        ),
        (
            (&expected_irreps[9], &tc5),
            Character::new(&[(UnityRoot::new(5, 10), 1)]),
        ),
        (
            (&expected_irreps[10], &tc5),
            Character::new(&[(UnityRoot::new(3, 10), 1), (UnityRoot::new(7, 10), 1)]),
        ),
        (
            (&expected_irreps[11], &tc5),
            Character::new(&[(UnityRoot::new(1, 10), 1), (UnityRoot::new(9, 10), 1)]),
        ),
        (
            (&expected_irreps[12], &tc5),
            Character::new(&[(UnityRoot::new(5, 10), 1)]),
        ),
        (
            (&expected_irreps[13], &tc5),
            Character::new(&[(UnityRoot::new(5, 10), 1)]),
        ),
        (
            (&expected_irreps[14], &tc5),
            Character::new(&[(UnityRoot::new(3, 10), 1), (UnityRoot::new(7, 10), 1)]),
        ),
        (
            (&expected_irreps[15], &tc5),
            Character::new(&[(UnityRoot::new(1, 10), 1), (UnityRoot::new(9, 10), 1)]),
        ),
    ]);
    test_chartab_magnetic_group(
        &mol,
        thresh,
        "D5h + θ·D5h",
        &expected_irreps,
        Some(expected_chars),
    );
}

#[test]
fn test_chartab_symmetric_benzene_d6h() {
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
    let c6 = SymmetryClassSymbol::<SymmetryOperation>::new("2||C6||", None).unwrap();
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
    test_chartab_ordinary_group(&mol, thresh, "D6h", &expected_irreps, Some(expected_chars));
}

#[test]
fn test_chartab_symmetric_benzene_grey_d6h() {
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
        MullikenIrrepSymbol::new("|^(m)|A|_(1g)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|A|_(2g)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|B|_(1g)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|B|_(2g)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|E|_(1g)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|E|_(2g)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|A|_(1u)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|A|_(2u)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|B|_(1u)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|B|_(2u)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|E|_(1u)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|E|_(2u)|").unwrap(),
    ];
    let tc6 = SymmetryClassSymbol::<SymmetryOperation>::new("2||θ·C6||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &tc6),
            Character::new(&[(UnityRoot::new(0, 6), 1)]),
        ),
        (
            (&expected_irreps[1], &tc6),
            Character::new(&[(UnityRoot::new(0, 6), 1)]),
        ),
        (
            (&expected_irreps[2], &tc6),
            Character::new(&[(UnityRoot::new(3, 6), 1)]),
        ),
        (
            (&expected_irreps[3], &tc6),
            Character::new(&[(UnityRoot::new(3, 6), 1)]),
        ),
        (
            (&expected_irreps[4], &tc6),
            Character::new(&[(UnityRoot::new(1, 6), 1), (UnityRoot::new(5, 6), 1)]),
        ),
        (
            (&expected_irreps[5], &tc6),
            Character::new(&[(UnityRoot::new(2, 6), 1), (UnityRoot::new(4, 6), 1)]),
        ),
        (
            (&expected_irreps[6], &tc6),
            Character::new(&[(UnityRoot::new(0, 6), 1)]),
        ),
        (
            (&expected_irreps[7], &tc6),
            Character::new(&[(UnityRoot::new(0, 6), 1)]),
        ),
        (
            (&expected_irreps[8], &tc6),
            Character::new(&[(UnityRoot::new(3, 6), 1)]),
        ),
        (
            (&expected_irreps[9], &tc6),
            Character::new(&[(UnityRoot::new(3, 6), 1)]),
        ),
        (
            (&expected_irreps[10], &tc6),
            Character::new(&[(UnityRoot::new(1, 6), 1), (UnityRoot::new(5, 6), 1)]),
        ),
        (
            (&expected_irreps[11], &tc6),
            Character::new(&[(UnityRoot::new(2, 6), 1), (UnityRoot::new(4, 6), 1)]),
        ),
        (
            (&expected_irreps[12], &tc6),
            Character::new(&[(UnityRoot::new(3, 6), 1)]),
        ),
        (
            (&expected_irreps[13], &tc6),
            Character::new(&[(UnityRoot::new(3, 6), 1)]),
        ),
        (
            (&expected_irreps[14], &tc6),
            Character::new(&[(UnityRoot::new(0, 6), 1)]),
        ),
        (
            (&expected_irreps[15], &tc6),
            Character::new(&[(UnityRoot::new(0, 6), 1)]),
        ),
        (
            (&expected_irreps[16], &tc6),
            Character::new(&[(UnityRoot::new(2, 6), 1), (UnityRoot::new(4, 6), 1)]),
        ),
        (
            (&expected_irreps[17], &tc6),
            Character::new(&[(UnityRoot::new(1, 6), 1), (UnityRoot::new(5, 6), 1)]),
        ),
        (
            (&expected_irreps[18], &tc6),
            Character::new(&[(UnityRoot::new(3, 6), 1)]),
        ),
        (
            (&expected_irreps[19], &tc6),
            Character::new(&[(UnityRoot::new(3, 6), 1)]),
        ),
        (
            (&expected_irreps[20], &tc6),
            Character::new(&[(UnityRoot::new(0, 6), 1)]),
        ),
        (
            (&expected_irreps[21], &tc6),
            Character::new(&[(UnityRoot::new(0, 6), 1)]),
        ),
        (
            (&expected_irreps[22], &tc6),
            Character::new(&[(UnityRoot::new(2, 6), 1), (UnityRoot::new(4, 6), 1)]),
        ),
        (
            (&expected_irreps[23], &tc6),
            Character::new(&[(UnityRoot::new(1, 6), 1), (UnityRoot::new(5, 6), 1)]),
        ),
    ]);
    test_chartab_magnetic_group(
        &mol,
        thresh,
        "D6h + θ·D6h",
        &expected_irreps,
        Some(expected_chars),
    );
}

#[test]
fn test_chartab_symmetric_8_eclipsed_sandwich_d8h() {
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
    let c8 = SymmetryClassSymbol::<SymmetryOperation>::new("2||C8||", None).unwrap();
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
    test_chartab_ordinary_group(&mol, thresh, "D8h", &expected_irreps, Some(expected_chars));
}

#[test]
fn test_chartab_symmetric_8_eclipsed_sandwich_grey_d8h() {
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
        MullikenIrrepSymbol::new("|^(m)|A|_(1g)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|A|_(2g)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|B|_(1g)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|B|_(2g)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|E|_(1g)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|E|_(2g)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|E|_(3g)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|A|_(1u)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|A|_(2u)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|B|_(1u)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|B|_(2u)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|E|_(1u)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|E|_(2u)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|E|_(3u)|").unwrap(),
    ];
    let tc8 = SymmetryClassSymbol::<SymmetryOperation>::new("2||θ·C8||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &tc8),
            Character::new(&[(UnityRoot::new(0, 8), 1)]),
        ),
        (
            (&expected_irreps[1], &tc8),
            Character::new(&[(UnityRoot::new(0, 8), 1)]),
        ),
        (
            (&expected_irreps[2], &tc8),
            Character::new(&[(UnityRoot::new(4, 8), 1)]),
        ),
        (
            (&expected_irreps[3], &tc8),
            Character::new(&[(UnityRoot::new(4, 8), 1)]),
        ),
        (
            (&expected_irreps[4], &tc8),
            Character::new(&[(UnityRoot::new(1, 8), 1), (UnityRoot::new(7, 8), 1)]),
        ),
        (
            (&expected_irreps[5], &tc8),
            Character::new(&[(UnityRoot::new(2, 8), 1), (UnityRoot::new(6, 8), 1)]),
        ),
        (
            (&expected_irreps[6], &tc8),
            Character::new(&[(UnityRoot::new(3, 8), 1), (UnityRoot::new(5, 8), 1)]),
        ),
        (
            (&expected_irreps[7], &tc8),
            Character::new(&[(UnityRoot::new(0, 8), 1)]),
        ),
        (
            (&expected_irreps[8], &tc8),
            Character::new(&[(UnityRoot::new(0, 8), 1)]),
        ),
        (
            (&expected_irreps[9], &tc8),
            Character::new(&[(UnityRoot::new(4, 8), 1)]),
        ),
        (
            (&expected_irreps[10], &tc8),
            Character::new(&[(UnityRoot::new(4, 8), 1)]),
        ),
        (
            (&expected_irreps[11], &tc8),
            Character::new(&[(UnityRoot::new(1, 8), 1), (UnityRoot::new(7, 8), 1)]),
        ),
        (
            (&expected_irreps[12], &tc8),
            Character::new(&[(UnityRoot::new(2, 8), 1), (UnityRoot::new(6, 8), 1)]),
        ),
        (
            (&expected_irreps[13], &tc8),
            Character::new(&[(UnityRoot::new(3, 8), 1), (UnityRoot::new(5, 8), 1)]),
        ),
        (
            (&expected_irreps[14], &tc8),
            Character::new(&[(UnityRoot::new(4, 8), 1)]),
        ),
        (
            (&expected_irreps[15], &tc8),
            Character::new(&[(UnityRoot::new(4, 8), 1)]),
        ),
        (
            (&expected_irreps[16], &tc8),
            Character::new(&[(UnityRoot::new(0, 8), 1)]),
        ),
        (
            (&expected_irreps[17], &tc8),
            Character::new(&[(UnityRoot::new(0, 8), 1)]),
        ),
        (
            (&expected_irreps[18], &tc8),
            Character::new(&[(UnityRoot::new(3, 8), 1), (UnityRoot::new(5, 8), 1)]),
        ),
        (
            (&expected_irreps[19], &tc8),
            Character::new(&[(UnityRoot::new(2, 8), 1), (UnityRoot::new(6, 8), 1)]),
        ),
        (
            (&expected_irreps[20], &tc8),
            Character::new(&[(UnityRoot::new(1, 8), 1), (UnityRoot::new(7, 8), 1)]),
        ),
        (
            (&expected_irreps[21], &tc8),
            Character::new(&[(UnityRoot::new(4, 8), 1)]),
        ),
        (
            (&expected_irreps[22], &tc8),
            Character::new(&[(UnityRoot::new(4, 8), 1)]),
        ),
        (
            (&expected_irreps[23], &tc8),
            Character::new(&[(UnityRoot::new(0, 8), 1)]),
        ),
        (
            (&expected_irreps[24], &tc8),
            Character::new(&[(UnityRoot::new(0, 8), 1)]),
        ),
        (
            (&expected_irreps[25], &tc8),
            Character::new(&[(UnityRoot::new(3, 8), 1), (UnityRoot::new(5, 8), 1)]),
        ),
        (
            (&expected_irreps[26], &tc8),
            Character::new(&[(UnityRoot::new(2, 8), 1), (UnityRoot::new(6, 8), 1)]),
        ),
        (
            (&expected_irreps[27], &tc8),
            Character::new(&[(UnityRoot::new(1, 8), 1), (UnityRoot::new(7, 8), 1)]),
        ),
    ]);
    test_chartab_magnetic_group(
        &mol,
        thresh,
        "D8h + θ·D8h",
        &expected_irreps,
        Some(expected_chars),
    );
}

// #[test]
// fn test_chartab_symmetric_h100_d100h() {
//     // env_logger::init();
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/h100.xyz");
//     let thresh = 1e-6;
//     let mol = Molecule::from_xyz(&path, thresh);
//     let presym = PreSymmetry::builder()
//         .moi_threshold(thresh)
//         .molecule(&mol, true)
//         .build()
//         .unwrap();
//     let mut sym = Symmetry::new();
//     sym.analyse(&presym, false);
//     let uni_group = UnitaryRepresentedGroup::from_molecular_symmetry(&sym, None);
//     let irrep_chartab = uni_group
//         .character_table();
//     println!("Irreps of unitary subgroup");
//     println!("{:?}", irrep_chartab);
// }

#[test]
fn test_chartab_symmetric_arbitrary_eclipsed_sandwich_dnh() {
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
        test_chartab_ordinary_group(
            &mol,
            thresh,
            format!("D{n}h").as_str(),
            &expected_irreps,
            None,
        );
    }
}

#[test]
fn test_chartab_symmetric_arbitrary_eclipsed_sandwich_grey_dnh() {
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
            let mut irreps = ["g", "u"]
                .iter()
                .flat_map(|i_parity| {
                    let mut i_irreps = vec![
                        MullikenIrrepSymbol::new(&format!("||A|_(1{i_parity})|")).unwrap(),
                        MullikenIrrepSymbol::new(&format!("||A|_(2{i_parity})|")).unwrap(),
                        MullikenIrrepSymbol::new(&format!("||B|_(1{i_parity})|")).unwrap(),
                        MullikenIrrepSymbol::new(&format!("||B|_(2{i_parity})|")).unwrap(),
                    ];
                    if n == 4 {
                        i_irreps.push(
                            MullikenIrrepSymbol::new(&format!("||E|_({i_parity})|")).unwrap(),
                        );
                    } else {
                        i_irreps.extend((1..n.div_euclid(2)).map(|k| {
                            MullikenIrrepSymbol::new(&format!("||E|_({k}{i_parity})|")).unwrap()
                        }));
                    }
                    i_irreps
                })
                .collect_vec();
            let m_irreps = irreps
                .iter()
                .map(|irrep| {
                    MullikenIrrepSymbol::new(&format!(
                        "|^(m)|{}|^({})_({})|",
                        irrep.main(),
                        irrep.postsuper(),
                        irrep.postsub()
                    ))
                    .unwrap()
                })
                .collect_vec();
            irreps.extend(m_irreps);
            irreps
        } else {
            let mut irreps = ["'", "''"]
                .iter()
                .flat_map(|s_parity| {
                    let mut s_irreps = vec![
                        MullikenIrrepSymbol::new(&format!("||A|^({s_parity})_(1)|")).unwrap(),
                        MullikenIrrepSymbol::new(&format!("||A|^({s_parity})_(2)|")).unwrap(),
                    ];
                    if n == 3 {
                        s_irreps.push(
                            MullikenIrrepSymbol::new(&format!("||E|^({s_parity})|")).unwrap(),
                        );
                    } else {
                        s_irreps.extend((1..=n.div_euclid(2)).map(|k| {
                            MullikenIrrepSymbol::new(&format!("||E|^({s_parity})_({k})|")).unwrap()
                        }));
                    }
                    s_irreps
                })
                .collect_vec();
            let m_irreps = irreps
                .iter()
                .map(|irrep| {
                    MullikenIrrepSymbol::new(&format!(
                        "|^(m)|{}|^({})_({})|",
                        irrep.main(),
                        irrep.postsuper(),
                        irrep.postsub()
                    ))
                    .unwrap()
                })
                .collect_vec();
            irreps.extend(m_irreps);
            irreps
        };
        test_chartab_magnetic_group(
            &mol,
            thresh,
            format!("D{n}h + θ·D{n}h").as_str(),
            &expected_irreps,
            None,
        );
    }
}

/*
Dnd
*/
#[test]
fn test_chartab_symmetric_b2cl4_d2d() {
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
    let s4 = SymmetryClassSymbol::<SymmetryOperation>::new("2||S4||", None).unwrap();
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
    test_chartab_ordinary_group(&mol, thresh, "D2d", &expected_irreps, Some(expected_chars));
}

#[test]
fn test_chartab_symmetric_b2cl4_grey_d2d() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/b2cl4.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||E||").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|A|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|A|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|B|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|B|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|E||").unwrap(),
    ];
    let ts4 = SymmetryClassSymbol::<SymmetryOperation>::new("2||θ·S4||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &ts4),
            Character::new(&[(UnityRoot::new(0, 4), 1)]),
        ),
        (
            (&expected_irreps[1], &ts4),
            Character::new(&[(UnityRoot::new(0, 4), 1)]),
        ),
        (
            (&expected_irreps[2], &ts4),
            Character::new(&[(UnityRoot::new(2, 4), 1)]),
        ),
        (
            (&expected_irreps[3], &ts4),
            Character::new(&[(UnityRoot::new(2, 4), 1)]),
        ),
        (
            (&expected_irreps[4], &ts4),
            Character::new(&[(UnityRoot::new(1, 4), 1), (UnityRoot::new(3, 4), 1)]),
        ),
        (
            (&expected_irreps[5], &ts4),
            Character::new(&[(UnityRoot::new(2, 4), 1)]),
        ),
        (
            (&expected_irreps[6], &ts4),
            Character::new(&[(UnityRoot::new(2, 4), 1)]),
        ),
        (
            (&expected_irreps[7], &ts4),
            Character::new(&[(UnityRoot::new(0, 4), 1)]),
        ),
        (
            (&expected_irreps[8], &ts4),
            Character::new(&[(UnityRoot::new(0, 4), 1)]),
        ),
        (
            (&expected_irreps[9], &ts4),
            Character::new(&[(UnityRoot::new(1, 4), 1), (UnityRoot::new(3, 4), 1)]),
        ),
    ]);
    test_chartab_magnetic_group(
        &mol,
        thresh,
        "D2d + θ·D2d",
        &expected_irreps,
        Some(expected_chars),
    );
}

#[test]
fn test_chartab_symmetric_s4n4_d2d() {
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
    let s4 = SymmetryClassSymbol::<SymmetryOperation>::new("2||S4||", None).unwrap();
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
    test_chartab_ordinary_group(&mol, thresh, "D2d", &expected_irreps, Some(expected_chars));
}

#[test]
fn test_chartab_symmetric_pbet4_d2d() {
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
    let s4 = SymmetryClassSymbol::<SymmetryOperation>::new("2||S4||", None).unwrap();
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
    test_chartab_ordinary_group(&mol, thresh, "D2d", &expected_irreps, Some(expected_chars));
}

#[test]
fn test_chartab_symmetric_s4n4_grey_d2d() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/s4n4.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||E||").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|A|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|A|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|B|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|B|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|E||").unwrap(),
    ];
    let ts4 = SymmetryClassSymbol::<SymmetryOperation>::new("2||θ·S4||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &ts4),
            Character::new(&[(UnityRoot::new(0, 4), 1)]),
        ),
        (
            (&expected_irreps[1], &ts4),
            Character::new(&[(UnityRoot::new(0, 4), 1)]),
        ),
        (
            (&expected_irreps[2], &ts4),
            Character::new(&[(UnityRoot::new(2, 4), 1)]),
        ),
        (
            (&expected_irreps[3], &ts4),
            Character::new(&[(UnityRoot::new(2, 4), 1)]),
        ),
        (
            (&expected_irreps[4], &ts4),
            Character::new(&[(UnityRoot::new(1, 4), 1), (UnityRoot::new(3, 4), 1)]),
        ),
        (
            (&expected_irreps[5], &ts4),
            Character::new(&[(UnityRoot::new(2, 4), 1)]),
        ),
        (
            (&expected_irreps[6], &ts4),
            Character::new(&[(UnityRoot::new(2, 4), 1)]),
        ),
        (
            (&expected_irreps[7], &ts4),
            Character::new(&[(UnityRoot::new(0, 4), 1)]),
        ),
        (
            (&expected_irreps[8], &ts4),
            Character::new(&[(UnityRoot::new(0, 4), 1)]),
        ),
        (
            (&expected_irreps[9], &ts4),
            Character::new(&[(UnityRoot::new(1, 4), 1), (UnityRoot::new(3, 4), 1)]),
        ),
    ]);
    test_chartab_magnetic_group(
        &mol,
        thresh,
        "D2d + θ·D2d",
        &expected_irreps,
        Some(expected_chars),
    );
}

#[test]
fn test_chartab_symmetric_allene_d2d() {
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
    let s4 = SymmetryClassSymbol::<SymmetryOperation>::new("2||S4||", None).unwrap();
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
    test_chartab_ordinary_group(&mol, thresh, "D2d", &expected_irreps, Some(expected_chars));
}

#[test]
fn test_chartab_symmetric_allene_grey_d2d() {
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
        MullikenIrrepSymbol::new("|^(m)|A|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|A|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|B|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|B|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|E||").unwrap(),
    ];
    let ts4 = SymmetryClassSymbol::<SymmetryOperation>::new("2||θ·S4||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &ts4),
            Character::new(&[(UnityRoot::new(0, 4), 1)]),
        ),
        (
            (&expected_irreps[1], &ts4),
            Character::new(&[(UnityRoot::new(0, 4), 1)]),
        ),
        (
            (&expected_irreps[2], &ts4),
            Character::new(&[(UnityRoot::new(2, 4), 1)]),
        ),
        (
            (&expected_irreps[3], &ts4),
            Character::new(&[(UnityRoot::new(2, 4), 1)]),
        ),
        (
            (&expected_irreps[4], &ts4),
            Character::new(&[(UnityRoot::new(1, 4), 1), (UnityRoot::new(3, 4), 1)]),
        ),
        (
            (&expected_irreps[5], &ts4),
            Character::new(&[(UnityRoot::new(2, 4), 1)]),
        ),
        (
            (&expected_irreps[6], &ts4),
            Character::new(&[(UnityRoot::new(2, 4), 1)]),
        ),
        (
            (&expected_irreps[7], &ts4),
            Character::new(&[(UnityRoot::new(0, 4), 1)]),
        ),
        (
            (&expected_irreps[8], &ts4),
            Character::new(&[(UnityRoot::new(0, 4), 1)]),
        ),
        (
            (&expected_irreps[9], &ts4),
            Character::new(&[(UnityRoot::new(1, 4), 1), (UnityRoot::new(3, 4), 1)]),
        ),
    ]);
    test_chartab_magnetic_group(
        &mol,
        thresh,
        "D2d + θ·D2d",
        &expected_irreps,
        Some(expected_chars),
    );
}

#[test]
fn test_chartab_symmetric_staggered_c2h6_d3d() {
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
    let c3 = SymmetryClassSymbol::<SymmetryOperation>::new("2||C3||", None).unwrap();
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
    test_chartab_ordinary_group(&mol, thresh, "D3d", &expected_irreps, Some(expected_chars));
}

#[test]
fn test_chartab_symmetric_cyclohexane_chair_d3d() {
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
    let c3 = SymmetryClassSymbol::<SymmetryOperation>::new("2||C3||", None).unwrap();
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
    test_chartab_ordinary_group(&mol, thresh, "D3d", &expected_irreps, Some(expected_chars));
}

#[test]
fn test_chartab_symmetric_cyclohexane_chair_grey_d3d() {
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
        MullikenIrrepSymbol::new("|^(m)|A|_(1g)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|A|_(2g)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|E|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|A|_(1u)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|A|_(2u)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|E|_(u)|").unwrap(),
    ];
    let tc3 = SymmetryClassSymbol::<SymmetryOperation>::new("2||θ·C3||", None).unwrap();
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
            Character::new(&[(UnityRoot::new(0, 3), 1)]),
        ),
        (
            (&expected_irreps[4], &tc3),
            Character::new(&[(UnityRoot::new(0, 3), 1)]),
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
            Character::new(&[(UnityRoot::new(1, 3), 1), (UnityRoot::new(2, 3), 1)]),
        ),
        (
            (&expected_irreps[8], &tc3),
            Character::new(&[(UnityRoot::new(0, 3), 1)]),
        ),
        (
            (&expected_irreps[9], &tc3),
            Character::new(&[(UnityRoot::new(1, 3), 1), (UnityRoot::new(2, 3), 1)]),
        ),
        (
            (&expected_irreps[10], &tc3),
            Character::new(&[(UnityRoot::new(1, 3), 1), (UnityRoot::new(2, 3), 1)]),
        ),
        (
            (&expected_irreps[11], &tc3),
            Character::new(&[(UnityRoot::new(0, 3), 1)]),
        ),
    ]);
    test_chartab_magnetic_group(
        &mol,
        thresh,
        "D3d + θ·D3d",
        &expected_irreps,
        Some(expected_chars),
    );
}

#[test]
fn test_chartab_symmetric_s8_d4d() {
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
    let s8 = SymmetryClassSymbol::<SymmetryOperation>::new("2||S8||", None).unwrap();
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
    test_chartab_ordinary_group(&mol, thresh, "D4d", &expected_irreps, Some(expected_chars));
}

#[test]
fn test_chartab_symmetric_s8_grey_d4d() {
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
        MullikenIrrepSymbol::new("|^(m)|A|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|A|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|B|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|B|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|E|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|E|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|E|_(3)|").unwrap(),
    ];
    let ts8 = SymmetryClassSymbol::<SymmetryOperation>::new("2||θ·S8||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &ts8),
            Character::new(&[(UnityRoot::new(0, 8), 1)]),
        ),
        (
            (&expected_irreps[1], &ts8),
            Character::new(&[(UnityRoot::new(0, 8), 1)]),
        ),
        (
            (&expected_irreps[2], &ts8),
            Character::new(&[(UnityRoot::new(4, 8), 1)]),
        ),
        (
            (&expected_irreps[3], &ts8),
            Character::new(&[(UnityRoot::new(4, 8), 1)]),
        ),
        (
            (&expected_irreps[4], &ts8),
            Character::new(&[(UnityRoot::new(1, 8), 1), (UnityRoot::new(7, 8), 1)]),
        ),
        (
            (&expected_irreps[5], &ts8),
            Character::new(&[(UnityRoot::new(2, 8), 1), (UnityRoot::new(6, 8), 1)]),
        ),
        (
            (&expected_irreps[6], &ts8),
            Character::new(&[(UnityRoot::new(3, 8), 1), (UnityRoot::new(5, 8), 1)]),
        ),
        (
            (&expected_irreps[7], &ts8),
            Character::new(&[(UnityRoot::new(4, 8), 1)]),
        ),
        (
            (&expected_irreps[8], &ts8),
            Character::new(&[(UnityRoot::new(4, 8), 1)]),
        ),
        (
            (&expected_irreps[9], &ts8),
            Character::new(&[(UnityRoot::new(0, 8), 1)]),
        ),
        (
            (&expected_irreps[10], &ts8),
            Character::new(&[(UnityRoot::new(0, 8), 1)]),
        ),
        (
            (&expected_irreps[11], &ts8),
            Character::new(&[(UnityRoot::new(3, 8), 1), (UnityRoot::new(5, 8), 1)]),
        ),
        (
            (&expected_irreps[12], &ts8),
            Character::new(&[(UnityRoot::new(2, 8), 1), (UnityRoot::new(6, 8), 1)]),
        ),
        (
            (&expected_irreps[13], &ts8),
            Character::new(&[(UnityRoot::new(1, 8), 1), (UnityRoot::new(7, 8), 1)]),
        ),
    ]);
    test_chartab_magnetic_group(
        &mol,
        thresh,
        "D4d + θ·D4d",
        &expected_irreps,
        Some(expected_chars),
    );
}

#[test]
fn test_chartab_symmetric_antiprism_h8_d4d() {
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
    let s8 = SymmetryClassSymbol::<SymmetryOperation>::new("2||S8||", None).unwrap();
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
    test_chartab_ordinary_group(&mol, thresh, "D4d", &expected_irreps, Some(expected_chars));
}

#[test]
fn test_chartab_symmetric_antiprism_h8_grey_d4d() {
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
        MullikenIrrepSymbol::new("|^(m)|A|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|A|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|B|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|B|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|E|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|E|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|E|_(3)|").unwrap(),
    ];
    let ts8 = SymmetryClassSymbol::<SymmetryOperation>::new("2||θ·S8||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &ts8),
            Character::new(&[(UnityRoot::new(0, 8), 1)]),
        ),
        (
            (&expected_irreps[1], &ts8),
            Character::new(&[(UnityRoot::new(0, 8), 1)]),
        ),
        (
            (&expected_irreps[2], &ts8),
            Character::new(&[(UnityRoot::new(4, 8), 1)]),
        ),
        (
            (&expected_irreps[3], &ts8),
            Character::new(&[(UnityRoot::new(4, 8), 1)]),
        ),
        (
            (&expected_irreps[4], &ts8),
            Character::new(&[(UnityRoot::new(1, 8), 1), (UnityRoot::new(7, 8), 1)]),
        ),
        (
            (&expected_irreps[5], &ts8),
            Character::new(&[(UnityRoot::new(2, 8), 1), (UnityRoot::new(6, 8), 1)]),
        ),
        (
            (&expected_irreps[6], &ts8),
            Character::new(&[(UnityRoot::new(3, 8), 1), (UnityRoot::new(5, 8), 1)]),
        ),
        (
            (&expected_irreps[7], &ts8),
            Character::new(&[(UnityRoot::new(4, 8), 1)]),
        ),
        (
            (&expected_irreps[8], &ts8),
            Character::new(&[(UnityRoot::new(4, 8), 1)]),
        ),
        (
            (&expected_irreps[9], &ts8),
            Character::new(&[(UnityRoot::new(0, 8), 1)]),
        ),
        (
            (&expected_irreps[10], &ts8),
            Character::new(&[(UnityRoot::new(0, 8), 1)]),
        ),
        (
            (&expected_irreps[11], &ts8),
            Character::new(&[(UnityRoot::new(3, 8), 1), (UnityRoot::new(5, 8), 1)]),
        ),
        (
            (&expected_irreps[12], &ts8),
            Character::new(&[(UnityRoot::new(2, 8), 1), (UnityRoot::new(6, 8), 1)]),
        ),
        (
            (&expected_irreps[13], &ts8),
            Character::new(&[(UnityRoot::new(1, 8), 1), (UnityRoot::new(7, 8), 1)]),
        ),
    ]);
    test_chartab_magnetic_group(
        &mol,
        thresh,
        "D4d + θ·D4d",
        &expected_irreps,
        Some(expected_chars),
    );
}

#[test]
fn test_chartab_symmetric_antiprism_pb10_d4d() {
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
    let s8 = SymmetryClassSymbol::<SymmetryOperation>::new("2||S8||", None).unwrap();
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
    test_chartab_ordinary_group(&mol, thresh, "D4d", &expected_irreps, Some(expected_chars));
}

#[test]
fn test_chartab_symmetric_antiprism_pb10_grey_d4d() {
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
        MullikenIrrepSymbol::new("|^(m)|A|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|A|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|B|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|B|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|E|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|E|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|E|_(3)|").unwrap(),
    ];
    let ts8 = SymmetryClassSymbol::<SymmetryOperation>::new("2||θ·S8||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &ts8),
            Character::new(&[(UnityRoot::new(0, 8), 1)]),
        ),
        (
            (&expected_irreps[1], &ts8),
            Character::new(&[(UnityRoot::new(0, 8), 1)]),
        ),
        (
            (&expected_irreps[2], &ts8),
            Character::new(&[(UnityRoot::new(4, 8), 1)]),
        ),
        (
            (&expected_irreps[3], &ts8),
            Character::new(&[(UnityRoot::new(4, 8), 1)]),
        ),
        (
            (&expected_irreps[4], &ts8),
            Character::new(&[(UnityRoot::new(1, 8), 1), (UnityRoot::new(7, 8), 1)]),
        ),
        (
            (&expected_irreps[5], &ts8),
            Character::new(&[(UnityRoot::new(2, 8), 1), (UnityRoot::new(6, 8), 1)]),
        ),
        (
            (&expected_irreps[6], &ts8),
            Character::new(&[(UnityRoot::new(3, 8), 1), (UnityRoot::new(5, 8), 1)]),
        ),
        (
            (&expected_irreps[7], &ts8),
            Character::new(&[(UnityRoot::new(4, 8), 1)]),
        ),
        (
            (&expected_irreps[8], &ts8),
            Character::new(&[(UnityRoot::new(4, 8), 1)]),
        ),
        (
            (&expected_irreps[9], &ts8),
            Character::new(&[(UnityRoot::new(0, 8), 1)]),
        ),
        (
            (&expected_irreps[10], &ts8),
            Character::new(&[(UnityRoot::new(0, 8), 1)]),
        ),
        (
            (&expected_irreps[11], &ts8),
            Character::new(&[(UnityRoot::new(3, 8), 1), (UnityRoot::new(5, 8), 1)]),
        ),
        (
            (&expected_irreps[12], &ts8),
            Character::new(&[(UnityRoot::new(2, 8), 1), (UnityRoot::new(6, 8), 1)]),
        ),
        (
            (&expected_irreps[13], &ts8),
            Character::new(&[(UnityRoot::new(1, 8), 1), (UnityRoot::new(7, 8), 1)]),
        ),
    ]);
    test_chartab_magnetic_group(
        &mol,
        thresh,
        "D4d + θ·D4d",
        &expected_irreps,
        Some(expected_chars),
    );
}

#[test]
fn test_chartab_symmetric_staggered_ferrocene_d5d() {
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
    let c5 = SymmetryClassSymbol::<SymmetryOperation>::new("2||C5||", None).unwrap();
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
    test_chartab_ordinary_group(&mol, thresh, "D5d", &expected_irreps, Some(expected_chars));
}

#[test]
fn test_chartab_symmetric_staggered_ferrocene_grey_d5d() {
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
        MullikenIrrepSymbol::new("|^(m)|A|_(1g)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|A|_(2g)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|E|_(1g)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|E|_(2g)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|A|_(1u)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|A|_(2u)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|E|_(1u)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|E|_(2u)|").unwrap(),
    ];
    let tc5 = SymmetryClassSymbol::<SymmetryOperation>::new("2||θ·C5||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &tc5),
            Character::new(&[(UnityRoot::new(0, 5), 1)]),
        ),
        (
            (&expected_irreps[1], &tc5),
            Character::new(&[(UnityRoot::new(0, 5), 1)]),
        ),
        (
            (&expected_irreps[2], &tc5),
            Character::new(&[(UnityRoot::new(1, 5), 1), (UnityRoot::new(4, 5), 1)]),
        ),
        (
            (&expected_irreps[3], &tc5),
            Character::new(&[(UnityRoot::new(2, 5), 1), (UnityRoot::new(3, 5), 1)]),
        ),
        (
            (&expected_irreps[4], &tc5),
            Character::new(&[(UnityRoot::new(0, 5), 1)]),
        ),
        (
            (&expected_irreps[5], &tc5),
            Character::new(&[(UnityRoot::new(0, 5), 1)]),
        ),
        (
            (&expected_irreps[6], &tc5),
            Character::new(&[(UnityRoot::new(1, 5), 1), (UnityRoot::new(4, 5), 1)]),
        ),
        (
            (&expected_irreps[7], &tc5),
            Character::new(&[(UnityRoot::new(2, 5), 1), (UnityRoot::new(3, 5), 1)]),
        ),
        (
            (&expected_irreps[8], &tc5),
            Character::new(&[(UnityRoot::new(5, 10), 1)]),
        ),
        (
            (&expected_irreps[9], &tc5),
            Character::new(&[(UnityRoot::new(5, 10), 1)]),
        ),
        (
            (&expected_irreps[10], &tc5),
            Character::new(&[(UnityRoot::new(3, 10), 1), (UnityRoot::new(7, 10), 1)]),
        ),
        (
            (&expected_irreps[11], &tc5),
            Character::new(&[(UnityRoot::new(1, 10), 1), (UnityRoot::new(9, 10), 1)]),
        ),
        (
            (&expected_irreps[12], &tc5),
            Character::new(&[(UnityRoot::new(5, 10), 1)]),
        ),
        (
            (&expected_irreps[13], &tc5),
            Character::new(&[(UnityRoot::new(5, 10), 1)]),
        ),
        (
            (&expected_irreps[14], &tc5),
            Character::new(&[(UnityRoot::new(3, 10), 1), (UnityRoot::new(7, 10), 1)]),
        ),
        (
            (&expected_irreps[15], &tc5),
            Character::new(&[(UnityRoot::new(1, 10), 1), (UnityRoot::new(9, 10), 1)]),
        ),
    ]);
    test_chartab_magnetic_group(
        &mol,
        thresh,
        "D5d + θ·D5d",
        &expected_irreps,
        Some(expected_chars),
    );
}

#[test]
fn test_chartab_symmetric_au26_d6d() {
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
    let s12 = SymmetryClassSymbol::<SymmetryOperation>::new("2||S12||", None).unwrap();
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
    test_chartab_ordinary_group(&mol, thresh, "D6d", &expected_irreps, Some(expected_chars));
}

#[test]
fn test_chartab_symmetric_au26_grey_d6d() {
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
        MullikenIrrepSymbol::new("|^(m)|A|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|A|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|B|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|B|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|E|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|E|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|E|_(3)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|E|_(4)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|E|_(5)|").unwrap(),
    ];
    let ts12 = SymmetryClassSymbol::<SymmetryOperation>::new("2||θ·S12||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &ts12),
            Character::new(&[(UnityRoot::new(0, 12), 1)]),
        ),
        (
            (&expected_irreps[1], &ts12),
            Character::new(&[(UnityRoot::new(0, 12), 1)]),
        ),
        (
            (&expected_irreps[2], &ts12),
            Character::new(&[(UnityRoot::new(6, 12), 1)]),
        ),
        (
            (&expected_irreps[3], &ts12),
            Character::new(&[(UnityRoot::new(6, 12), 1)]),
        ),
        (
            (&expected_irreps[4], &ts12),
            Character::new(&[(UnityRoot::new(1, 12), 1), (UnityRoot::new(11, 12), 1)]),
        ),
        (
            (&expected_irreps[5], &ts12),
            Character::new(&[(UnityRoot::new(2, 12), 1), (UnityRoot::new(10, 12), 1)]),
        ),
        (
            (&expected_irreps[6], &ts12),
            Character::new(&[(UnityRoot::new(3, 12), 1), (UnityRoot::new(9, 12), 1)]),
        ),
        (
            (&expected_irreps[7], &ts12),
            Character::new(&[(UnityRoot::new(4, 12), 1), (UnityRoot::new(8, 12), 1)]),
        ),
        (
            (&expected_irreps[8], &ts12),
            Character::new(&[(UnityRoot::new(5, 12), 1), (UnityRoot::new(7, 12), 1)]),
        ),
        (
            (&expected_irreps[9], &ts12),
            Character::new(&[(UnityRoot::new(6, 12), 1)]),
        ),
        (
            (&expected_irreps[10], &ts12),
            Character::new(&[(UnityRoot::new(6, 12), 1)]),
        ),
        (
            (&expected_irreps[11], &ts12),
            Character::new(&[(UnityRoot::new(0, 12), 1)]),
        ),
        (
            (&expected_irreps[12], &ts12),
            Character::new(&[(UnityRoot::new(0, 12), 1)]),
        ),
        (
            (&expected_irreps[13], &ts12),
            Character::new(&[(UnityRoot::new(5, 12), 1), (UnityRoot::new(7, 12), 1)]),
        ),
        (
            (&expected_irreps[14], &ts12),
            Character::new(&[(UnityRoot::new(4, 12), 1), (UnityRoot::new(8, 12), 1)]),
        ),
        (
            (&expected_irreps[15], &ts12),
            Character::new(&[(UnityRoot::new(3, 12), 1), (UnityRoot::new(9, 12), 1)]),
        ),
        (
            (&expected_irreps[16], &ts12),
            Character::new(&[(UnityRoot::new(2, 12), 1), (UnityRoot::new(10, 12), 1)]),
        ),
        (
            (&expected_irreps[17], &ts12),
            Character::new(&[(UnityRoot::new(1, 12), 1), (UnityRoot::new(11, 12), 1)]),
        ),
    ]);
    test_chartab_magnetic_group(
        &mol,
        thresh,
        "D6d + θ·D6d",
        &expected_irreps,
        Some(expected_chars),
    );
}

#[test]
fn test_chartab_symmetric_arbitrary_staggered_sandwich_dnd() {
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
                if n == 3 {
                    irreps.push(MullikenIrrepSymbol::new(&format!("||E|_({parity})|")).unwrap());
                } else {
                    irreps.extend((1..=n.div_euclid(2)).map(|k| {
                        MullikenIrrepSymbol::new(&format!("||E|_({k}{parity})|")).unwrap()
                    }));
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
            if n == 2 {
                irreps.push(MullikenIrrepSymbol::new("||E||").unwrap());
            } else {
                irreps.extend(
                    (1..n).map(|k| MullikenIrrepSymbol::new(&format!("||E|_({k})|")).unwrap()),
                );
            };
            irreps
        };
        test_chartab_ordinary_group(
            &mol,
            thresh,
            format!("D{n}d").as_str(),
            &expected_irreps,
            None,
        );
    }
}

#[test]
fn test_chartab_symmetric_arbitrary_staggered_sandwich_grey_dnd() {
    let thresh = 1e-7;
    for n in 3..=20 {
        let mol = template_molecules::gen_arbitrary_twisted_sandwich(n, 0.5);
        let expected_irreps = if n % 2 != 0 {
            // Odd n, g/u possible
            let mut irreps = ["g", "u"]
                .iter()
                .flat_map(|i_parity| {
                    let mut i_irreps = vec![
                        MullikenIrrepSymbol::new(&format!("||A|_(1{i_parity})|")).unwrap(),
                        MullikenIrrepSymbol::new(&format!("||A|_(2{i_parity})|")).unwrap(),
                    ];
                    if n == 3 {
                        i_irreps.push(
                            MullikenIrrepSymbol::new(&format!("||E|_({i_parity})|")).unwrap(),
                        );
                    } else {
                        i_irreps.extend((1..=n.div_euclid(2)).map(|k| {
                            MullikenIrrepSymbol::new(&format!("||E|_({k}{i_parity})|")).unwrap()
                        }));
                    }
                    i_irreps
                })
                .collect_vec();
            let m_irreps = irreps
                .iter()
                .map(|irrep| {
                    MullikenIrrepSymbol::new(&format!(
                        "|^(m)|{}|^({})_({})|",
                        irrep.main(),
                        irrep.postsuper(),
                        irrep.postsub()
                    ))
                    .unwrap()
                })
                .collect_vec();
            irreps.extend(m_irreps);
            irreps
        } else {
            // Even n, no g/u
            let mut irreps = vec![
                MullikenIrrepSymbol::new("||A|_(1)|").unwrap(),
                MullikenIrrepSymbol::new("||A|_(2)|").unwrap(),
                MullikenIrrepSymbol::new("||B|_(1)|").unwrap(),
                MullikenIrrepSymbol::new("||B|_(2)|").unwrap(),
            ];
            if n == 2 {
                irreps.push(MullikenIrrepSymbol::new("||E||").unwrap());
            } else {
                irreps.extend(
                    (1..n).map(|k| MullikenIrrepSymbol::new(&format!("||E|_({k})|")).unwrap()),
                );
            };
            let m_irreps = irreps
                .iter()
                .map(|irrep| {
                    MullikenIrrepSymbol::new(&format!(
                        "|^(m)|{}|^({})_({})|",
                        irrep.main(),
                        irrep.postsuper(),
                        irrep.postsub()
                    ))
                    .unwrap()
                })
                .collect_vec();
            irreps.extend(m_irreps);
            irreps
        };
        test_chartab_magnetic_group(
            &mol,
            thresh,
            format!("D{n}d + θ·D{n}d").as_str(),
            &expected_irreps,
            None,
        );
    }
}

/***
Dnd*
***/

#[test]
fn test_chartab_symmetric_b2cl4_d2d_double() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/b2cl4.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||E||").unwrap(),
        MullikenIrrepSymbol::new("||E~|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||E~|_(2)|").unwrap(),
    ];
    let s4 = SymmetryClassSymbol::<SymmetryOperation>::new("2||S4(Σ)||", None).unwrap();
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
        (
            (&expected_irreps[5], &s4),
            Character::new(&[(UnityRoot::new(1, 8), 1), (UnityRoot::new(7, 8), 1)]),
        ),
        (
            (&expected_irreps[6], &s4),
            Character::new(&[(UnityRoot::new(3, 8), 1), (UnityRoot::new(5, 8), 1)]),
        ),
    ]);
    test_chartab_ordinary_double_group(
        &mol,
        thresh,
        "D2d*",
        &expected_irreps,
        Some(expected_chars),
    );
}

#[test]
fn test_chartab_symmetric_b2cl4_grey_d2d_double() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/b2cl4.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||E||").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|A|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|A|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|B|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|B|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|E||").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|E~|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|E~|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|E~|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|E~|_(2)|").unwrap(),
    ];
    let ts4 = SymmetryClassSymbol::<SymmetryOperation>::new("2||θ·S4(Σ)||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &ts4),
            Character::new(&[(UnityRoot::new(0, 4), 1)]),
        ),
        (
            (&expected_irreps[1], &ts4),
            Character::new(&[(UnityRoot::new(0, 4), 1)]),
        ),
        (
            (&expected_irreps[2], &ts4),
            Character::new(&[(UnityRoot::new(2, 4), 1)]),
        ),
        (
            (&expected_irreps[3], &ts4),
            Character::new(&[(UnityRoot::new(2, 4), 1)]),
        ),
        (
            (&expected_irreps[4], &ts4),
            Character::new(&[(UnityRoot::new(1, 4), 1), (UnityRoot::new(3, 4), 1)]),
        ),
        (
            (&expected_irreps[5], &ts4),
            Character::new(&[(UnityRoot::new(2, 4), 1)]),
        ),
        (
            (&expected_irreps[6], &ts4),
            Character::new(&[(UnityRoot::new(2, 4), 1)]),
        ),
        (
            (&expected_irreps[7], &ts4),
            Character::new(&[(UnityRoot::new(0, 4), 1)]),
        ),
        (
            (&expected_irreps[8], &ts4),
            Character::new(&[(UnityRoot::new(0, 4), 1)]),
        ),
        (
            (&expected_irreps[9], &ts4),
            Character::new(&[(UnityRoot::new(1, 4), 1), (UnityRoot::new(3, 4), 1)]),
        ),
        (
            (&expected_irreps[10], &ts4),
            Character::new(&[(UnityRoot::new(1, 8), 1), (UnityRoot::new(3, 8), 1)]),
        ),
        (
            (&expected_irreps[11], &ts4),
            Character::new(&[(UnityRoot::new(5, 8), 1), (UnityRoot::new(7, 8), 1)]),
        ),
        (
            (&expected_irreps[12], &ts4),
            Character::new(&[(UnityRoot::new(5, 8), 1), (UnityRoot::new(7, 8), 1)]),
        ),
        (
            (&expected_irreps[13], &ts4),
            Character::new(&[(UnityRoot::new(1, 8), 1), (UnityRoot::new(3, 8), 1)]),
        ),
    ]);
    test_chartab_magnetic_double_group(
        &mol,
        thresh,
        "(D2d + θ·D2d)*",
        &expected_irreps,
        Some(expected_chars),
    );
}

#[test]
fn test_chartab_symmetric_cyclohexane_chair_d3d_double() {
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
        MullikenIrrepSymbol::new("|_(a)|Γ~|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|Γ~|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("||E~|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|Γ~|_(u)|").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|Γ~|_(u)|").unwrap(),
        MullikenIrrepSymbol::new("||E~|_(u)|").unwrap(),
    ];
    let c3 = SymmetryClassSymbol::<SymmetryOperation>::new("2||C3(Σ)||", None).unwrap();
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
        (
            (&expected_irreps[6], &c3),
            Character::new(&[(UnityRoot::new(1, 3), 1), (UnityRoot::new(2, 3), 1)]),
        ),
        (
            (&expected_irreps[7], &c3),
            Character::new(&[(UnityRoot::new(1, 3), 1), (UnityRoot::new(2, 3), 1)]),
        ),
        (
            (&expected_irreps[8], &c3),
            Character::new(&[(UnityRoot::new(0, 3), 1)]),
        ),
        (
            (&expected_irreps[9], &c3),
            Character::new(&[(UnityRoot::new(1, 3), 1), (UnityRoot::new(2, 3), 1)]),
        ),
        (
            (&expected_irreps[10], &c3),
            Character::new(&[(UnityRoot::new(1, 3), 1), (UnityRoot::new(2, 3), 1)]),
        ),
        (
            (&expected_irreps[11], &c3),
            Character::new(&[(UnityRoot::new(0, 3), 1)]),
        ),
    ]);
    test_chartab_ordinary_double_group(
        &mol,
        thresh,
        "D3d*",
        &expected_irreps,
        Some(expected_chars),
    );
}

#[test]
fn test_chartab_symmetric_cyclohexane_chair_grey_d3d_double() {
    // env_logger::init();
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
        MullikenIrrepSymbol::new("|^(m)|A|_(1g)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|A|_(2g)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|E|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|A|_(1u)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|A|_(2u)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|E|_(u)|").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|Γ~|_(1g)|").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|Γ~|_(2g)|").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|E~|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|Γ~|_(1u)|").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|Γ~|_(2u)|").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|E~|_(u)|").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|Γ~|_(2g)|").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|Γ~|_(1g)|").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|E~|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|Γ~|_(2u)|").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|Γ~|_(1u)|").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|E~|_(u)|").unwrap(),
    ];
    let tc3 = SymmetryClassSymbol::<SymmetryOperation>::new("2||θ·C3(Σ)||", None).unwrap();
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
            Character::new(&[(UnityRoot::new(0, 3), 1)]),
        ),
        (
            (&expected_irreps[4], &tc3),
            Character::new(&[(UnityRoot::new(0, 3), 1)]),
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
            Character::new(&[(UnityRoot::new(1, 3), 1), (UnityRoot::new(2, 3), 1)]),
        ),
        (
            (&expected_irreps[8], &tc3),
            Character::new(&[(UnityRoot::new(0, 3), 1)]),
        ),
        (
            (&expected_irreps[9], &tc3),
            Character::new(&[(UnityRoot::new(1, 3), 1), (UnityRoot::new(2, 3), 1)]),
        ),
        (
            (&expected_irreps[10], &tc3),
            Character::new(&[(UnityRoot::new(1, 3), 1), (UnityRoot::new(2, 3), 1)]),
        ),
        (
            (&expected_irreps[11], &tc3),
            Character::new(&[(UnityRoot::new(0, 3), 1)]),
        ),
        (
            (&expected_irreps[12], &tc3),
            Character::new(&[(UnityRoot::new(3, 4), 1)]),
        ),
        (
            (&expected_irreps[13], &tc3),
            Character::new(&[(UnityRoot::new(3, 4), 1)]),
        ),
        (
            (&expected_irreps[14], &tc3),
            Character::new(&[(UnityRoot::new(1, 4), 1)]),
        ),
        (
            (&expected_irreps[15], &tc3),
            Character::new(&[(UnityRoot::new(3, 4), 1)]),
        ),
        (
            (&expected_irreps[16], &tc3),
            Character::new(&[(UnityRoot::new(3, 4), 1)]),
        ),
        (
            (&expected_irreps[17], &tc3),
            Character::new(&[(UnityRoot::new(1, 4), 1)]),
        ),
        (
            (&expected_irreps[18], &tc3),
            Character::new(&[(UnityRoot::new(1, 4), 1)]),
        ),
        (
            (&expected_irreps[19], &tc3),
            Character::new(&[(UnityRoot::new(1, 4), 1)]),
        ),
        (
            (&expected_irreps[20], &tc3),
            Character::new(&[(UnityRoot::new(3, 4), 1)]),
        ),
        (
            (&expected_irreps[21], &tc3),
            Character::new(&[(UnityRoot::new(1, 4), 1)]),
        ),
        (
            (&expected_irreps[22], &tc3),
            Character::new(&[(UnityRoot::new(1, 4), 1)]),
        ),
        (
            (&expected_irreps[23], &tc3),
            Character::new(&[(UnityRoot::new(3, 4), 1)]),
        ),
    ]);
    test_chartab_magnetic_double_group(
        &mol,
        thresh,
        "(D3d + θ·D3d)*",
        &expected_irreps,
        Some(expected_chars),
    );
}

#[test]
fn test_chartab_symmetric_antiprism_pb10_d4d_double() {
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
        MullikenIrrepSymbol::new("||E~|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||E~|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||E~|_(3)|").unwrap(),
        MullikenIrrepSymbol::new("||E~|_(4)|").unwrap(),
    ];
    let s8 = SymmetryClassSymbol::<SymmetryOperation>::new("2||S8(Σ)||", None).unwrap();
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
        (
            (&expected_irreps[7], &s8),
            Character::new(&[(UnityRoot::new(1, 16), 1), (UnityRoot::new(15, 16), 1)]),
        ),
        (
            (&expected_irreps[8], &s8),
            Character::new(&[(UnityRoot::new(3, 16), 1), (UnityRoot::new(13, 16), 1)]),
        ),
        (
            (&expected_irreps[9], &s8),
            Character::new(&[(UnityRoot::new(5, 16), 1), (UnityRoot::new(11, 16), 1)]),
        ),
        (
            (&expected_irreps[10], &s8),
            Character::new(&[(UnityRoot::new(7, 16), 1), (UnityRoot::new(9, 16), 1)]),
        ),
    ]);
    test_chartab_ordinary_double_group(
        &mol,
        thresh,
        "D4d*",
        &expected_irreps,
        Some(expected_chars),
    );
}

#[test]
fn test_chartab_symmetric_antiprism_pb10_grey_d4d_double() {
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
        MullikenIrrepSymbol::new("|^(m)|A|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|A|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|B|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|B|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|E|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|E|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|E|_(3)|").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|E~|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|E~|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|E~|_(3)|").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|E~|_(4)|").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|E~|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|E~|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|E~|_(3)|").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|E~|_(4)|").unwrap(),
    ];
    let ts8 = SymmetryClassSymbol::<SymmetryOperation>::new("2||θ·S8(Σ)||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &ts8),
            Character::new(&[(UnityRoot::new(0, 8), 1)]),
        ),
        (
            (&expected_irreps[1], &ts8),
            Character::new(&[(UnityRoot::new(0, 8), 1)]),
        ),
        (
            (&expected_irreps[2], &ts8),
            Character::new(&[(UnityRoot::new(4, 8), 1)]),
        ),
        (
            (&expected_irreps[3], &ts8),
            Character::new(&[(UnityRoot::new(4, 8), 1)]),
        ),
        (
            (&expected_irreps[4], &ts8),
            Character::new(&[(UnityRoot::new(1, 8), 1), (UnityRoot::new(7, 8), 1)]),
        ),
        (
            (&expected_irreps[5], &ts8),
            Character::new(&[(UnityRoot::new(2, 8), 1), (UnityRoot::new(6, 8), 1)]),
        ),
        (
            (&expected_irreps[6], &ts8),
            Character::new(&[(UnityRoot::new(3, 8), 1), (UnityRoot::new(5, 8), 1)]),
        ),
        (
            (&expected_irreps[7], &ts8),
            Character::new(&[(UnityRoot::new(4, 8), 1)]),
        ),
        (
            (&expected_irreps[8], &ts8),
            Character::new(&[(UnityRoot::new(4, 8), 1)]),
        ),
        (
            (&expected_irreps[9], &ts8),
            Character::new(&[(UnityRoot::new(0, 8), 1)]),
        ),
        (
            (&expected_irreps[10], &ts8),
            Character::new(&[(UnityRoot::new(0, 8), 1)]),
        ),
        (
            (&expected_irreps[11], &ts8),
            Character::new(&[(UnityRoot::new(3, 8), 1), (UnityRoot::new(5, 8), 1)]),
        ),
        (
            (&expected_irreps[12], &ts8),
            Character::new(&[(UnityRoot::new(2, 8), 1), (UnityRoot::new(6, 8), 1)]),
        ),
        (
            (&expected_irreps[13], &ts8),
            Character::new(&[(UnityRoot::new(1, 8), 1), (UnityRoot::new(7, 8), 1)]),
        ),
        (
            (&expected_irreps[14], &ts8),
            Character::new(&[(UnityRoot::new(3, 16), 1), (UnityRoot::new(5, 16), 1)]),
        ),
        (
            (&expected_irreps[15], &ts8),
            Character::new(&[(UnityRoot::new(1, 16), 1), (UnityRoot::new(7, 16), 1)]),
        ),
        (
            (&expected_irreps[16], &ts8),
            Character::new(&[(UnityRoot::new(9, 16), 1), (UnityRoot::new(15, 16), 1)]),
        ),
        (
            (&expected_irreps[17], &ts8),
            Character::new(&[(UnityRoot::new(11, 16), 1), (UnityRoot::new(13, 16), 1)]),
        ),
        (
            (&expected_irreps[18], &ts8),
            Character::new(&[(UnityRoot::new(11, 16), 1), (UnityRoot::new(13, 16), 1)]),
        ),
        (
            (&expected_irreps[19], &ts8),
            Character::new(&[(UnityRoot::new(9, 16), 1), (UnityRoot::new(15, 16), 1)]),
        ),
        (
            (&expected_irreps[20], &ts8),
            Character::new(&[(UnityRoot::new(1, 16), 1), (UnityRoot::new(7, 16), 1)]),
        ),
        (
            (&expected_irreps[21], &ts8),
            Character::new(&[(UnityRoot::new(3, 16), 1), (UnityRoot::new(5, 16), 1)]),
        ),
    ]);
    test_chartab_magnetic_double_group(
        &mol,
        thresh,
        "(D4d + θ·D4d)*",
        &expected_irreps,
        Some(expected_chars),
    );
}

#[test]
fn test_chartab_symmetric_staggered_ferrocene_d5d_double() {
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
        MullikenIrrepSymbol::new("|_(a)|Γ~|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|Γ~|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("||E~|_(1g)|").unwrap(),
        MullikenIrrepSymbol::new("||E~|_(2g)|").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|Γ~|_(u)|").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|Γ~|_(u)|").unwrap(),
        MullikenIrrepSymbol::new("||E~|_(1u)|").unwrap(),
        MullikenIrrepSymbol::new("||E~|_(2u)|").unwrap(),
    ];
    let c5 = SymmetryClassSymbol::<SymmetryOperation>::new("2||C5(Σ)||", None).unwrap();
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
    test_chartab_ordinary_double_group(
        &mol,
        thresh,
        "D5d*",
        &expected_irreps,
        Some(expected_chars),
    );
}

#[test]
fn test_chartab_symmetric_staggered_ferrocene_grey_d5d_double() {
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
        MullikenIrrepSymbol::new("|^(m)|A|_(1g)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|A|_(2g)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|E|_(1g)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|E|_(2g)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|A|_(1u)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|A|_(2u)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|E|_(1u)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|E|_(2u)|").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|Γ~|_(1g)|").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|Γ~|_(2g)|").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|E~|_(1g)|").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|E~|_(2g)|").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|Γ~|_(1u)|").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|Γ~|_(2u)|").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|E~|_(1u)|").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|E~|_(2u)|").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|Γ~|_(2g)|").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|Γ~|_(1g)|").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|E~|_(1g)|").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|E~|_(2g)|").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|Γ~|_(2u)|").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|Γ~|_(1u)|").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|E~|_(1u)|").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|E~|_(2u)|").unwrap(),
    ];
    let tc5 = SymmetryClassSymbol::<SymmetryOperation>::new("2||θ·C5(Σ)||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &tc5),
            Character::new(&[(UnityRoot::new(0, 5), 1)]),
        ),
        (
            (&expected_irreps[1], &tc5),
            Character::new(&[(UnityRoot::new(0, 5), 1)]),
        ),
        (
            (&expected_irreps[2], &tc5),
            Character::new(&[(UnityRoot::new(1, 5), 1), (UnityRoot::new(4, 5), 1)]),
        ),
        (
            (&expected_irreps[3], &tc5),
            Character::new(&[(UnityRoot::new(2, 5), 1), (UnityRoot::new(3, 5), 1)]),
        ),
        (
            (&expected_irreps[4], &tc5),
            Character::new(&[(UnityRoot::new(0, 5), 1)]),
        ),
        (
            (&expected_irreps[5], &tc5),
            Character::new(&[(UnityRoot::new(0, 5), 1)]),
        ),
        (
            (&expected_irreps[6], &tc5),
            Character::new(&[(UnityRoot::new(1, 5), 1), (UnityRoot::new(4, 5), 1)]),
        ),
        (
            (&expected_irreps[7], &tc5),
            Character::new(&[(UnityRoot::new(2, 5), 1), (UnityRoot::new(3, 5), 1)]),
        ),
        (
            (&expected_irreps[8], &tc5),
            Character::new(&[(UnityRoot::new(5, 10), 1)]),
        ),
        (
            (&expected_irreps[9], &tc5),
            Character::new(&[(UnityRoot::new(5, 10), 1)]),
        ),
        (
            (&expected_irreps[10], &tc5),
            Character::new(&[(UnityRoot::new(3, 10), 1), (UnityRoot::new(7, 10), 1)]),
        ),
        (
            (&expected_irreps[11], &tc5),
            Character::new(&[(UnityRoot::new(1, 10), 1), (UnityRoot::new(9, 10), 1)]),
        ),
        (
            (&expected_irreps[12], &tc5),
            Character::new(&[(UnityRoot::new(5, 10), 1)]),
        ),
        (
            (&expected_irreps[13], &tc5),
            Character::new(&[(UnityRoot::new(5, 10), 1)]),
        ),
        (
            (&expected_irreps[14], &tc5),
            Character::new(&[(UnityRoot::new(3, 10), 1), (UnityRoot::new(7, 10), 1)]),
        ),
        (
            (&expected_irreps[15], &tc5),
            Character::new(&[(UnityRoot::new(1, 10), 1), (UnityRoot::new(9, 10), 1)]),
        ),
        (
            (&expected_irreps[16], &tc5),
            Character::new(&[(UnityRoot::new(15, 20), 1)]),
        ),
        (
            (&expected_irreps[17], &tc5),
            Character::new(&[(UnityRoot::new(15, 20), 1)]),
        ),
        (
            (&expected_irreps[18], &tc5),
            Character::new(&[(UnityRoot::new(3, 20), 1), (UnityRoot::new(7, 20), 1)]),
        ),
        (
            (&expected_irreps[19], &tc5),
            Character::new(&[(UnityRoot::new(11, 20), 1), (UnityRoot::new(19, 20), 1)]),
        ),
        (
            (&expected_irreps[20], &tc5),
            Character::new(&[(UnityRoot::new(15, 20), 1)]),
        ),
        (
            (&expected_irreps[21], &tc5),
            Character::new(&[(UnityRoot::new(15, 20), 1)]),
        ),
        (
            (&expected_irreps[22], &tc5),
            Character::new(&[(UnityRoot::new(3, 20), 1), (UnityRoot::new(7, 20), 1)]),
        ),
        (
            (&expected_irreps[23], &tc5),
            Character::new(&[(UnityRoot::new(11, 20), 1), (UnityRoot::new(19, 20), 1)]),
        ),
        (
            (&expected_irreps[24], &tc5),
            Character::new(&[(UnityRoot::new(5, 20), 1)]),
        ),
        (
            (&expected_irreps[25], &tc5),
            Character::new(&[(UnityRoot::new(5, 20), 1)]),
        ),
        (
            (&expected_irreps[26], &tc5),
            Character::new(&[(UnityRoot::new(13, 20), 1), (UnityRoot::new(17, 20), 1)]),
        ),
        (
            (&expected_irreps[27], &tc5),
            Character::new(&[(UnityRoot::new(1, 20), 1), (UnityRoot::new(9, 20), 1)]),
        ),
        (
            (&expected_irreps[28], &tc5),
            Character::new(&[(UnityRoot::new(5, 20), 1)]),
        ),
        (
            (&expected_irreps[29], &tc5),
            Character::new(&[(UnityRoot::new(5, 20), 1)]),
        ),
        (
            (&expected_irreps[30], &tc5),
            Character::new(&[(UnityRoot::new(13, 20), 1), (UnityRoot::new(17, 20), 1)]),
        ),
        (
            (&expected_irreps[31], &tc5),
            Character::new(&[(UnityRoot::new(1, 20), 1), (UnityRoot::new(9, 20), 1)]),
        ),
    ]);
    test_chartab_magnetic_double_group(
        &mol,
        thresh,
        "(D5d + θ·D5d)*",
        &expected_irreps,
        Some(expected_chars),
    );
}

#[test]
fn test_chartab_symmetric_au26_d6d_double() {
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
        MullikenIrrepSymbol::new("||E~|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||E~|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||E~|_(3)|").unwrap(),
        MullikenIrrepSymbol::new("||E~|_(4)|").unwrap(),
        MullikenIrrepSymbol::new("||E~|_(5)|").unwrap(),
        MullikenIrrepSymbol::new("||E~|_(6)|").unwrap(),
    ];
    let s12 = SymmetryClassSymbol::<SymmetryOperation>::new("2||S12(Σ)||", None).unwrap();
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
        (
            (&expected_irreps[9], &s12),
            Character::new(&[(UnityRoot::new(1, 24), 1), (UnityRoot::new(23, 24), 1)]),
        ),
        (
            (&expected_irreps[10], &s12),
            Character::new(&[(UnityRoot::new(3, 24), 1), (UnityRoot::new(21, 24), 1)]),
        ),
        (
            (&expected_irreps[11], &s12),
            Character::new(&[(UnityRoot::new(5, 24), 1), (UnityRoot::new(19, 24), 1)]),
        ),
        (
            (&expected_irreps[12], &s12),
            Character::new(&[(UnityRoot::new(7, 24), 1), (UnityRoot::new(17, 24), 1)]),
        ),
        (
            (&expected_irreps[13], &s12),
            Character::new(&[(UnityRoot::new(9, 24), 1), (UnityRoot::new(15, 24), 1)]),
        ),
        (
            (&expected_irreps[14], &s12),
            Character::new(&[(UnityRoot::new(11, 24), 1), (UnityRoot::new(13, 24), 1)]),
        ),
    ]);
    test_chartab_ordinary_double_group(
        &mol,
        thresh,
        "D6d*",
        &expected_irreps,
        Some(expected_chars),
    );
}

#[test]
fn test_chartab_symmetric_au26_grey_d6d_double() {
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
        MullikenIrrepSymbol::new("|^(m)|A|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|A|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|B|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|B|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|E|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|E|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|E|_(3)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|E|_(4)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|E|_(5)|").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|E~|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|E~|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|E~|_(3)|").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|E~|_(4)|").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|E~|_(5)|").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|E~|_(6)|").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|E~|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|E~|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|E~|_(3)|").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|E~|_(4)|").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|E~|_(5)|").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|E~|_(6)|").unwrap(),
    ];
    let ts12 = SymmetryClassSymbol::<SymmetryOperation>::new("2||θ·S12(Σ)||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &ts12),
            Character::new(&[(UnityRoot::new(0, 12), 1)]),
        ),
        (
            (&expected_irreps[1], &ts12),
            Character::new(&[(UnityRoot::new(0, 12), 1)]),
        ),
        (
            (&expected_irreps[2], &ts12),
            Character::new(&[(UnityRoot::new(6, 12), 1)]),
        ),
        (
            (&expected_irreps[3], &ts12),
            Character::new(&[(UnityRoot::new(6, 12), 1)]),
        ),
        (
            (&expected_irreps[4], &ts12),
            Character::new(&[(UnityRoot::new(1, 12), 1), (UnityRoot::new(11, 12), 1)]),
        ),
        (
            (&expected_irreps[5], &ts12),
            Character::new(&[(UnityRoot::new(2, 12), 1), (UnityRoot::new(10, 12), 1)]),
        ),
        (
            (&expected_irreps[6], &ts12),
            Character::new(&[(UnityRoot::new(3, 12), 1), (UnityRoot::new(9, 12), 1)]),
        ),
        (
            (&expected_irreps[7], &ts12),
            Character::new(&[(UnityRoot::new(4, 12), 1), (UnityRoot::new(8, 12), 1)]),
        ),
        (
            (&expected_irreps[8], &ts12),
            Character::new(&[(UnityRoot::new(5, 12), 1), (UnityRoot::new(7, 12), 1)]),
        ),
        (
            (&expected_irreps[9], &ts12),
            Character::new(&[(UnityRoot::new(6, 12), 1)]),
        ),
        (
            (&expected_irreps[10], &ts12),
            Character::new(&[(UnityRoot::new(6, 12), 1)]),
        ),
        (
            (&expected_irreps[11], &ts12),
            Character::new(&[(UnityRoot::new(0, 12), 1)]),
        ),
        (
            (&expected_irreps[12], &ts12),
            Character::new(&[(UnityRoot::new(0, 12), 1)]),
        ),
        (
            (&expected_irreps[13], &ts12),
            Character::new(&[(UnityRoot::new(5, 12), 1), (UnityRoot::new(7, 12), 1)]),
        ),
        (
            (&expected_irreps[14], &ts12),
            Character::new(&[(UnityRoot::new(4, 12), 1), (UnityRoot::new(8, 12), 1)]),
        ),
        (
            (&expected_irreps[15], &ts12),
            Character::new(&[(UnityRoot::new(3, 12), 1), (UnityRoot::new(9, 12), 1)]),
        ),
        (
            (&expected_irreps[16], &ts12),
            Character::new(&[(UnityRoot::new(2, 12), 1), (UnityRoot::new(10, 12), 1)]),
        ),
        (
            (&expected_irreps[17], &ts12),
            Character::new(&[(UnityRoot::new(1, 12), 1), (UnityRoot::new(11, 12), 1)]),
        ),
        (
            (&expected_irreps[18], &ts12),
            Character::new(&[(UnityRoot::new(5, 24), 1), (UnityRoot::new(7, 24), 1)]),
        ),
        (
            (&expected_irreps[19], &ts12),
            Character::new(&[(UnityRoot::new(3, 24), 1), (UnityRoot::new(9, 24), 1)]),
        ),
        (
            (&expected_irreps[20], &ts12),
            Character::new(&[(UnityRoot::new(1, 24), 1), (UnityRoot::new(11, 24), 1)]),
        ),
        (
            (&expected_irreps[21], &ts12),
            Character::new(&[(UnityRoot::new(13, 24), 1), (UnityRoot::new(23, 24), 1)]),
        ),
        (
            (&expected_irreps[22], &ts12),
            Character::new(&[(UnityRoot::new(15, 24), 1), (UnityRoot::new(21, 24), 1)]),
        ),
        (
            (&expected_irreps[23], &ts12),
            Character::new(&[(UnityRoot::new(17, 24), 1), (UnityRoot::new(19, 24), 1)]),
        ),
        (
            (&expected_irreps[24], &ts12),
            Character::new(&[(UnityRoot::new(17, 24), 1), (UnityRoot::new(19, 24), 1)]),
        ),
        (
            (&expected_irreps[25], &ts12),
            Character::new(&[(UnityRoot::new(15, 24), 1), (UnityRoot::new(21, 24), 1)]),
        ),
        (
            (&expected_irreps[26], &ts12),
            Character::new(&[(UnityRoot::new(13, 24), 1), (UnityRoot::new(23, 24), 1)]),
        ),
        (
            (&expected_irreps[27], &ts12),
            Character::new(&[(UnityRoot::new(1, 24), 1), (UnityRoot::new(11, 24), 1)]),
        ),
        (
            (&expected_irreps[28], &ts12),
            Character::new(&[(UnityRoot::new(3, 24), 1), (UnityRoot::new(9, 24), 1)]),
        ),
        (
            (&expected_irreps[29], &ts12),
            Character::new(&[(UnityRoot::new(5, 24), 1), (UnityRoot::new(7, 24), 1)]),
        ),
    ]);
    test_chartab_magnetic_double_group(
        &mol,
        thresh,
        "(D6d + θ·D6d)*",
        &expected_irreps,
        Some(expected_chars),
    );
}

/*
S2n
*/
#[test]
fn test_chartab_symmetric_b2cl4_magnetic_field_s4() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/b2cl4.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A||").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|Γ||").unwrap(),
        MullikenIrrepSymbol::new("||B||").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|Γ||").unwrap(),
    ];
    let s4 = SymmetryClassSymbol::<SymmetryOperation>::new("1||S4||", None).unwrap();
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
    test_chartab_ordinary_group(&mol, thresh, "S4", &expected_irreps, Some(expected_chars));
}

#[test]
fn test_chartab_symmetric_b2cl4_magnetic_field_bw_d2d_s4() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/b2cl4.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||E||").unwrap(),
    ];
    let s4 = SymmetryClassSymbol::<SymmetryOperation>::new("2||S4||", None).unwrap();
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
    test_chartab_magnetic_group(&mol, thresh, "D2d", &expected_irreps, Some(expected_chars));
}

#[test]
fn test_chartab_symmetric_adamantane_magnetic_field_s4() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/adamantane.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(-0.1, 0.0, 0.0)));
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A||").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|Γ||").unwrap(),
        MullikenIrrepSymbol::new("||B||").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|Γ||").unwrap(),
    ];
    let s4 = SymmetryClassSymbol::<SymmetryOperation>::new("1||S4||", None).unwrap();
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
    test_chartab_ordinary_group(&mol, thresh, "S4", &expected_irreps, Some(expected_chars));
}

#[test]
fn test_chartab_symmetric_adamantane_magnetic_field_bw_d2d_s4() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/adamantane.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(-0.1, 0.0, 0.0)));
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||E||").unwrap(),
    ];
    let s4 = SymmetryClassSymbol::<SymmetryOperation>::new("2||S4||", None).unwrap();
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
    test_chartab_magnetic_group(&mol, thresh, "D2d", &expected_irreps, Some(expected_chars));
}

#[test]
fn test_chartab_symmetric_ch4_magnetic_field_s4() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/ch4.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A||").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|Γ||").unwrap(),
        MullikenIrrepSymbol::new("||B||").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|Γ||").unwrap(),
    ];
    let s4 = SymmetryClassSymbol::<SymmetryOperation>::new("1||S4||", None).unwrap();
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
    test_chartab_ordinary_group(&mol, thresh, "S4", &expected_irreps, Some(expected_chars));
}

#[test]
fn test_chartab_symmetric_ch4_magnetic_field_bw_d2d_s4() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/ch4.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||E||").unwrap(),
    ];
    let s4 = SymmetryClassSymbol::<SymmetryOperation>::new("2||S4||", None).unwrap();
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
    test_chartab_magnetic_group(&mol, thresh, "D2d", &expected_irreps, Some(expected_chars));
}

#[test]
fn test_chartab_symmetric_65coronane_s6() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/coronane65.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|Γ|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|Γ|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(u)|").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|Γ|_(u)|").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|Γ|_(u)|").unwrap(),
    ];
    let c3 = SymmetryClassSymbol::<SymmetryOperation>::new("1||C3||", None).unwrap();
    let s6 = SymmetryClassSymbol::<SymmetryOperation>::new("1||S6||", None).unwrap();
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
    test_chartab_ordinary_group(&mol, thresh, "S6", &expected_irreps, Some(expected_chars));
}

#[test]
fn test_chartab_symmetric_65coronane_grey_s6() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/coronane65.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|Γ|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|Γ|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(u)|").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|Γ|_(u)|").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|Γ|_(u)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|A|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)_(a)|Γ|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)_(b)|Γ|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|A|_(u)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)_(a)|Γ|_(u)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)_(b)|Γ|_(u)|").unwrap(),
    ];
    let tc3 = SymmetryClassSymbol::<SymmetryOperation>::new("1||θ·C3||", None).unwrap();
    let ts6 = SymmetryClassSymbol::<SymmetryOperation>::new("1||θ·S6||", None).unwrap();
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
            (&expected_irreps[0], &ts6),
            Character::new(&[(UnityRoot::new(0, 6), 1)]),
        ),
        (
            (&expected_irreps[1], &ts6),
            Character::new(&[(UnityRoot::new(4, 6), 1)]),
        ),
        (
            (&expected_irreps[2], &ts6),
            Character::new(&[(UnityRoot::new(2, 6), 1)]),
        ),
        (
            (&expected_irreps[3], &ts6),
            Character::new(&[(UnityRoot::new(3, 6), 1)]),
        ),
        (
            (&expected_irreps[4], &ts6),
            Character::new(&[(UnityRoot::new(1, 6), 1)]),
        ),
        (
            (&expected_irreps[5], &ts6),
            Character::new(&[(UnityRoot::new(5, 6), 1)]),
        ),
        (
            (&expected_irreps[6], &tc3),
            Character::new(&[(UnityRoot::new(3, 6), 1)]),
        ),
        (
            (&expected_irreps[7], &tc3),
            Character::new(&[(UnityRoot::new(5, 6), 1)]),
        ),
        (
            (&expected_irreps[8], &tc3),
            Character::new(&[(UnityRoot::new(1, 6), 1)]),
        ),
        (
            (&expected_irreps[9], &tc3),
            Character::new(&[(UnityRoot::new(3, 6), 1)]),
        ),
        (
            (&expected_irreps[10], &tc3),
            Character::new(&[(UnityRoot::new(5, 6), 1)]),
        ),
        (
            (&expected_irreps[11], &tc3),
            Character::new(&[(UnityRoot::new(1, 6), 1)]),
        ),
        (
            (&expected_irreps[6], &ts6),
            Character::new(&[(UnityRoot::new(3, 6), 1)]),
        ),
        (
            (&expected_irreps[7], &ts6),
            Character::new(&[(UnityRoot::new(1, 6), 1)]),
        ),
        (
            (&expected_irreps[8], &ts6),
            Character::new(&[(UnityRoot::new(5, 6), 1)]),
        ),
        (
            (&expected_irreps[9], &ts6),
            Character::new(&[(UnityRoot::new(0, 6), 1)]),
        ),
        (
            (&expected_irreps[10], &ts6),
            Character::new(&[(UnityRoot::new(2, 3), 1)]),
        ),
        (
            (&expected_irreps[11], &ts6),
            Character::new(&[(UnityRoot::new(1, 3), 1)]),
        ),
    ]);
    test_chartab_magnetic_group(
        &mol,
        thresh,
        "S6 + θ·S6",
        &expected_irreps,
        Some(expected_chars),
    );
}

#[test]
fn test_chartab_symmetric_65coronane_magnetic_field_s6() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/coronane65.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, -1.0)));
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|Γ|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|Γ|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(u)|").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|Γ|_(u)|").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|Γ|_(u)|").unwrap(),
    ];
    let c3 = SymmetryClassSymbol::<SymmetryOperation>::new("1||C3||", None).unwrap();
    let s6 = SymmetryClassSymbol::<SymmetryOperation>::new("1||S6||", None).unwrap();
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
    test_chartab_ordinary_group(&mol, thresh, "S6", &expected_irreps, Some(expected_chars));
}

#[test]
fn test_chartab_symmetric_staggered_c2h6_magnetic_field_s6() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c2h6.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|Γ|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|Γ|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(u)|").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|Γ|_(u)|").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|Γ|_(u)|").unwrap(),
    ];
    let c3 = SymmetryClassSymbol::<SymmetryOperation>::new("1||C3||", None).unwrap();
    let s6 = SymmetryClassSymbol::<SymmetryOperation>::new("1||S6||", None).unwrap();
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
    test_chartab_ordinary_group(&mol, thresh, "S6", &expected_irreps, Some(expected_chars));
}

#[test]
fn test_chartab_symmetric_staggered_c2h6_magnetic_field_bw_d3d_s6() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c2h6.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(1g)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(2g)|").unwrap(),
        MullikenIrrepSymbol::new("||E|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(1u)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(2u)|").unwrap(),
        MullikenIrrepSymbol::new("||E|_(u)|").unwrap(),
    ];
    let c3 = SymmetryClassSymbol::<SymmetryOperation>::new("2||C3||", None).unwrap();
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
    test_chartab_magnetic_group(&mol, thresh, "D3d", &expected_irreps, Some(expected_chars));
}

#[test]
fn test_chartab_symmetric_c60_magnetic_field_s6() {
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
        MullikenIrrepSymbol::new("|_(a)|Γ|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|Γ|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(u)|").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|Γ|_(u)|").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|Γ|_(u)|").unwrap(),
    ];
    let c3 = SymmetryClassSymbol::<SymmetryOperation>::new("1||C3||", None).unwrap();
    let s6 = SymmetryClassSymbol::<SymmetryOperation>::new("1||S6||", None).unwrap();
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
    test_chartab_ordinary_group(&mol, thresh, "S6", &expected_irreps, Some(expected_chars));
}

#[test]
fn test_chartab_symmetric_c60_magnetic_field_bw_d3d_s6() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c60.xyz");
    let thresh = 1e-5;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(
        -0.5773503107731,
        -0.1875926572335,
        0.7946543988441,
    )));
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(1g)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(2g)|").unwrap(),
        MullikenIrrepSymbol::new("||E|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(1u)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(2u)|").unwrap(),
        MullikenIrrepSymbol::new("||E|_(u)|").unwrap(),
    ];
    let c3 = SymmetryClassSymbol::<SymmetryOperation>::new("2||C3||", None).unwrap();
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
    test_chartab_magnetic_group(&mol, thresh, "D3d", &expected_irreps, Some(expected_chars));
}

#[test]
fn test_chartab_symmetric_vh2o6_magnetic_field_s6() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/vh2o6.xyz");
    let thresh = 1e-12;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(-0.2, 0.2, 0.2)));
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|Γ|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|Γ|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(u)|").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|Γ|_(u)|").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|Γ|_(u)|").unwrap(),
    ];
    let c3 = SymmetryClassSymbol::<SymmetryOperation>::new("1||C3||", None).unwrap();
    let s6 = SymmetryClassSymbol::<SymmetryOperation>::new("1||S6||", None).unwrap();
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
    test_chartab_ordinary_group(&mol, thresh, "S6", &expected_irreps, Some(expected_chars));
}

#[test]
fn test_chartab_symmetric_vf6_magnetic_field_s6() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/vf6.xyz");
    let thresh = 1e-12;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(1.0, 1.0, 1.0)));
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|Γ|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|Γ|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(u)|").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|Γ|_(u)|").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|Γ|_(u)|").unwrap(),
    ];
    let c3 = SymmetryClassSymbol::<SymmetryOperation>::new("1||C3||", None).unwrap();
    let s6 = SymmetryClassSymbol::<SymmetryOperation>::new("1||S6||", None).unwrap();
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
    test_chartab_ordinary_group(&mol, thresh, "S6", &expected_irreps, Some(expected_chars));
}

#[test]
fn test_chartab_symmetric_vf6_magnetic_field_bw_d3d_s6() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/vf6.xyz");
    let thresh = 1e-12;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(1.0, 1.0, 1.0)));
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(1g)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(2g)|").unwrap(),
        MullikenIrrepSymbol::new("||E|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(1u)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(2u)|").unwrap(),
        MullikenIrrepSymbol::new("||E|_(u)|").unwrap(),
    ];
    let c3 = SymmetryClassSymbol::<SymmetryOperation>::new("2||C3||", None).unwrap();
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
    test_chartab_magnetic_group(&mol, thresh, "D3d", &expected_irreps, Some(expected_chars));
}

#[test]
fn test_chartab_symmetric_s8_magnetic_field_s8() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/s8.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A||").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|Γ|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|Γ|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|Γ|_(3)|").unwrap(),
        MullikenIrrepSymbol::new("||B||").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|Γ|_(3)|").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|Γ|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|Γ|_(1)|").unwrap(),
    ];
    let s8 = SymmetryClassSymbol::<SymmetryOperation>::new("1||S8||", None).unwrap();
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
    test_chartab_ordinary_group(&mol, thresh, "S8", &expected_irreps, Some(expected_chars));
}

#[test]
fn test_chartab_symmetric_s8_magnetic_field_bw_d4d_s8() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/s8.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||E|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||E|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||E|_(3)|").unwrap(),
    ];
    let s8 = SymmetryClassSymbol::<SymmetryOperation>::new("2||S8||", None).unwrap();
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
    test_chartab_magnetic_group(&mol, thresh, "D4d", &expected_irreps, Some(expected_chars));
}

#[test]
fn test_chartab_symmetric_antiprism_pb10_magnetic_field_s8() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/pb10.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A||").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|Γ|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|Γ|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|Γ|_(3)|").unwrap(),
        MullikenIrrepSymbol::new("||B||").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|Γ|_(3)|").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|Γ|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|Γ|_(1)|").unwrap(),
    ];
    let s8 = SymmetryClassSymbol::<SymmetryOperation>::new("1||S8||", None).unwrap();
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
    test_chartab_ordinary_group(&mol, thresh, "S8", &expected_irreps, Some(expected_chars));
}

#[test]
fn test_chartab_symmetric_antiprism_pb10_magnetic_field_bw_d4d_s8() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/pb10.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||E|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||E|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||E|_(3)|").unwrap(),
    ];
    let s8 = SymmetryClassSymbol::<SymmetryOperation>::new("2||S8||", None).unwrap();
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
    test_chartab_magnetic_group(&mol, thresh, "D4d", &expected_irreps, Some(expected_chars));
}

#[test]
fn test_chartab_symmetric_staggered_ferrocene_magnetic_field_s10() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/staggered_ferrocene.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|Γ|_(1g)|").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|Γ|_(2g)|").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|Γ|_(2g)|").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|Γ|_(1g)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(u)|").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|Γ|_(1u)|").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|Γ|_(2u)|").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|Γ|_(2u)|").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|Γ|_(1u)|").unwrap(),
    ];
    let c5 = SymmetryClassSymbol::<SymmetryOperation>::new("1||C5||", None).unwrap();
    let s10 = SymmetryClassSymbol::<SymmetryOperation>::new("1||S10||", None).unwrap();
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
    test_chartab_ordinary_group(&mol, thresh, "S10", &expected_irreps, Some(expected_chars));
}

#[test]
fn test_chartab_symmetric_staggered_ferrocene_magnetic_field_bw_d5d_s10() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/staggered_ferrocene.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
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
    let c5 = SymmetryClassSymbol::<SymmetryOperation>::new("2||C5||", None).unwrap();
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
    test_chartab_magnetic_group(&mol, thresh, "D5d", &expected_irreps, Some(expected_chars));
}

#[test]
fn test_chartab_symmetric_c60_magnetic_field_s10() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c60.xyz");
    let thresh = 1e-5;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|Γ|_(1g)|").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|Γ|_(2g)|").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|Γ|_(2g)|").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|Γ|_(1g)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(u)|").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|Γ|_(1u)|").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|Γ|_(2u)|").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|Γ|_(2u)|").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|Γ|_(1u)|").unwrap(),
    ];
    let c5 = SymmetryClassSymbol::<SymmetryOperation>::new("1||C5||", None).unwrap();
    let s10 = SymmetryClassSymbol::<SymmetryOperation>::new("1||S10||", None).unwrap();
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
    test_chartab_ordinary_group(&mol, thresh, "S10", &expected_irreps, Some(expected_chars));
}

#[test]
fn test_chartab_symmetric_c60_magnetic_field_bw_d5d_s10() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c60.xyz");
    let thresh = 1e-5;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
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
    let c5 = SymmetryClassSymbol::<SymmetryOperation>::new("2||C5||", None).unwrap();
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
    test_chartab_magnetic_group(&mol, thresh, "D5d", &expected_irreps, Some(expected_chars));
}

#[test]
fn test_chartab_symmetric_au26_magnetic_field_s12() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/au26.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A||").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|Γ|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|Γ|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|Γ|_(3)|").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|Γ|_(4)|").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|Γ|_(5)|").unwrap(),
        MullikenIrrepSymbol::new("||B||").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|Γ|_(5)|").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|Γ|_(4)|").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|Γ|_(3)|").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|Γ|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|Γ|_(1)|").unwrap(),
    ];
    let s12 = SymmetryClassSymbol::<SymmetryOperation>::new("1||S12||", None).unwrap();
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
    test_chartab_ordinary_group(&mol, thresh, "S12", &expected_irreps, Some(expected_chars));
}

#[test]
fn test_chartab_symmetric_au26_magnetic_field_bw_d6d_s12() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/au26.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
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
    let s12 = SymmetryClassSymbol::<SymmetryOperation>::new("2||S12||", None).unwrap();
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
    test_chartab_magnetic_group(&mol, thresh, "D6d", &expected_irreps, Some(expected_chars));
}

#[test]
fn test_chartab_symmetric_arbitrary_staggered_sandwich_magnetic_field_s2n() {
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
                if n == 3 {
                    irreps
                        .push(MullikenIrrepSymbol::new(&format!("|_(a)|Γ|_({parity})|")).unwrap());
                    irreps
                        .push(MullikenIrrepSymbol::new(&format!("|_(b)|Γ|_({parity})|")).unwrap());
                } else {
                    irreps.extend((1..=n.div_euclid(2)).map(|k| {
                        MullikenIrrepSymbol::new(&format!("|_(a)|Γ|_({k}{parity})|")).unwrap()
                    }));
                    irreps.extend((1..=n.div_euclid(2)).rev().map(|k| {
                        MullikenIrrepSymbol::new(&format!("|_(b)|Γ|_({k}{parity})|")).unwrap()
                    }));
                }
                irreps_gu.extend(irreps)
            }
            irreps_gu
        } else {
            // Even n, no g/u possible.
            let mut irreps = vec![MullikenIrrepSymbol::new("||A||").unwrap()];
            irreps.extend(
                (1..n).map(|k| MullikenIrrepSymbol::new(&format!("|_(a)|Γ|_({k})|")).unwrap()),
            );
            irreps.push(MullikenIrrepSymbol::new("||B||").unwrap());
            irreps.extend(
                (1..n)
                    .rev()
                    .map(|k| MullikenIrrepSymbol::new(&format!("|_(b)|Γ|_({k})|")).unwrap()),
            );
            irreps
        };
        test_chartab_ordinary_group(
            &mol,
            thresh,
            format!("S{}", 2 * n).as_str(),
            &expected_irreps,
            None,
        );
    }
}

#[test]
fn test_chartab_symmetric_arbitrary_staggered_sandwich_magnetic_field_bw_dnd_s2n() {
    // env_logger::init();
    let thresh = 1e-7;
    for n in 3..=20 {
        let mut mol = template_molecules::gen_arbitrary_twisted_sandwich(n, 0.5);
        mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
        let expected_irreps = if n % 2 != 0 {
            // Odd n, g/u possible
            let mut irreps_gu = vec![];
            for parity in ["g", "u"] {
                let mut irreps = vec![
                    MullikenIrrepSymbol::new(&format!("||A|_(1{parity})|")).unwrap(),
                    MullikenIrrepSymbol::new(&format!("||A|_(2{parity})|")).unwrap(),
                ];
                if n == 3 {
                    irreps.push(MullikenIrrepSymbol::new(&format!("||E|_({parity})|")).unwrap());
                } else {
                    irreps.extend((1..=n.div_euclid(2)).map(|k| {
                        MullikenIrrepSymbol::new(&format!("||E|_({k}{parity})|")).unwrap()
                    }));
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
            if n == 2 {
                irreps.push(MullikenIrrepSymbol::new("||E||").unwrap());
            } else {
                irreps.extend(
                    (1..n).map(|k| MullikenIrrepSymbol::new(&format!("||E|_({k})|")).unwrap()),
                );
            };
            irreps
        };
        test_chartab_magnetic_group(
            &mol,
            thresh,
            format!("D{n}d").as_str(),
            &expected_irreps,
            None,
        );
    }
}

/***
S2n*
***/
#[test]
fn test_chartab_symmetric_b2cl4_magnetic_field_s4_double() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/b2cl4.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A||").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|Γ||").unwrap(),
        MullikenIrrepSymbol::new("||B||").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|Γ||").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|Γ~|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|Γ~|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|Γ~|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|Γ~|_(1)|").unwrap(),
    ];
    let s4 = SymmetryClassSymbol::<SymmetryOperation>::new("1||S4(Σ)||", None).unwrap();
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
        (
            (&expected_irreps[4], &s4),
            Character::new(&[(UnityRoot::new(1, 8), 1)]),
        ),
        (
            (&expected_irreps[5], &s4),
            Character::new(&[(UnityRoot::new(3, 8), 1)]),
        ),
        (
            (&expected_irreps[6], &s4),
            Character::new(&[(UnityRoot::new(5, 8), 1)]),
        ),
        (
            (&expected_irreps[7], &s4),
            Character::new(&[(UnityRoot::new(7, 8), 1)]),
        ),
    ]);
    test_chartab_ordinary_double_group(&mol, thresh, "S4*", &expected_irreps, Some(expected_chars));
}

#[test]
fn test_chartab_symmetric_b2cl4_magnetic_field_bw_d2d_s4_double() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/b2cl4.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||E||").unwrap(),
        MullikenIrrepSymbol::new("||E~|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||E~|_(2)|").unwrap(),
    ];
    let s4 = SymmetryClassSymbol::<SymmetryOperation>::new("2||S4(Σ)||", None).unwrap();
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
        (
            (&expected_irreps[5], &s4),
            Character::new(&[(UnityRoot::new(1, 8), 1), (UnityRoot::new(7, 8), 1)]),
        ),
        (
            (&expected_irreps[6], &s4),
            Character::new(&[(UnityRoot::new(3, 8), 1), (UnityRoot::new(5, 8), 1)]),
        ),
    ]);
    test_chartab_magnetic_double_group(
        &mol,
        thresh,
        "D2d*",
        &expected_irreps,
        Some(expected_chars),
    );
}

#[test]
fn test_chartab_symmetric_vf6_magnetic_field_s6_double() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/vf6.xyz");
    let thresh = 1e-12;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(1.0, 1.0, 1.0)));
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|Γ|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|Γ|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(u)|").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|Γ|_(u)|").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|Γ|_(u)|").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|Γ~|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("||B~|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|Γ~|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|Γ~|_(u)|").unwrap(),
        MullikenIrrepSymbol::new("||B~|_(u)|").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|Γ~|_(u)|").unwrap(),
    ];
    let c3 = SymmetryClassSymbol::<SymmetryOperation>::new("1||C3(Σ)||", None).unwrap();
    let s6 = SymmetryClassSymbol::<SymmetryOperation>::new("1||S6(Σ)||", None).unwrap();
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
            (&expected_irreps[6], &c3),
            Character::new(&[(UnityRoot::new(1, 6), 1)]),
        ),
        (
            (&expected_irreps[7], &c3),
            Character::new(&[(UnityRoot::new(3, 6), 1)]),
        ),
        (
            (&expected_irreps[8], &c3),
            Character::new(&[(UnityRoot::new(5, 6), 1)]),
        ),
        (
            (&expected_irreps[9], &c3),
            Character::new(&[(UnityRoot::new(1, 6), 1)]),
        ),
        (
            (&expected_irreps[10], &c3),
            Character::new(&[(UnityRoot::new(3, 6), 1)]),
        ),
        (
            (&expected_irreps[11], &c3),
            Character::new(&[(UnityRoot::new(5, 6), 1)]),
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
        (
            (&expected_irreps[6], &s6),
            Character::new(&[(UnityRoot::new(5, 6), 1)]),
        ),
        (
            (&expected_irreps[7], &s6),
            Character::new(&[(UnityRoot::new(3, 6), 1)]),
        ),
        (
            (&expected_irreps[8], &s6),
            Character::new(&[(UnityRoot::new(1, 6), 1)]),
        ),
        (
            (&expected_irreps[9], &s6),
            Character::new(&[(UnityRoot::new(2, 6), 1)]),
        ),
        (
            (&expected_irreps[10], &s6),
            Character::new(&[(UnityRoot::new(0, 6), 1)]),
        ),
        (
            (&expected_irreps[11], &s6),
            Character::new(&[(UnityRoot::new(4, 6), 1)]),
        ),
    ]);
    test_chartab_ordinary_double_group(&mol, thresh, "S6*", &expected_irreps, Some(expected_chars));
}

#[test]
fn test_chartab_symmetric_vf6_magnetic_field_bw_d3d_s6_double() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/vf6.xyz");
    let thresh = 1e-12;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(1.0, 1.0, 1.0)));
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(1g)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(2g)|").unwrap(),
        MullikenIrrepSymbol::new("||E|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(1u)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(2u)|").unwrap(),
        MullikenIrrepSymbol::new("||E|_(u)|").unwrap(),
        MullikenIrrepSymbol::new("||B~|_(1g)|").unwrap(),
        MullikenIrrepSymbol::new("||B~|_(2g)|").unwrap(),
        MullikenIrrepSymbol::new("||E~|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("||B~|_(1u)|").unwrap(),
        MullikenIrrepSymbol::new("||B~|_(2u)|").unwrap(),
        MullikenIrrepSymbol::new("||E~|_(u)|").unwrap(),
    ];
    let c3 = SymmetryClassSymbol::<SymmetryOperation>::new("2||C3(Σ)||", None).unwrap();
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
        (
            (&expected_irreps[6], &c3),
            Character::new(&[(UnityRoot::new(1, 3), 1), (UnityRoot::new(2, 3), 1)]),
        ),
        (
            (&expected_irreps[7], &c3),
            Character::new(&[(UnityRoot::new(1, 3), 1), (UnityRoot::new(2, 3), 1)]),
        ),
        (
            (&expected_irreps[8], &c3),
            Character::new(&[(UnityRoot::new(0, 3), 1)]),
        ),
        (
            (&expected_irreps[9], &c3),
            Character::new(&[(UnityRoot::new(1, 3), 1), (UnityRoot::new(2, 3), 1)]),
        ),
        (
            (&expected_irreps[10], &c3),
            Character::new(&[(UnityRoot::new(1, 3), 1), (UnityRoot::new(2, 3), 1)]),
        ),
        (
            (&expected_irreps[11], &c3),
            Character::new(&[(UnityRoot::new(0, 3), 1)]),
        ),
    ]);
    test_chartab_magnetic_double_group(
        &mol,
        thresh,
        "D3d*",
        &expected_irreps,
        Some(expected_chars),
    );
}

#[test]
fn test_chartab_symmetric_antiprism_pb10_magnetic_field_s8_double() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/pb10.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A||").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|Γ|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|Γ|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|Γ|_(3)|").unwrap(),
        MullikenIrrepSymbol::new("||B||").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|Γ|_(3)|").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|Γ|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|Γ|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|Γ~|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|Γ~|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|Γ~|_(3)|").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|Γ~|_(4)|").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|Γ~|_(4)|").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|Γ~|_(3)|").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|Γ~|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|Γ~|_(1)|").unwrap(),
    ];
    let s8 = SymmetryClassSymbol::<SymmetryOperation>::new("1||S8(Σ)||", None).unwrap();
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
        (
            (&expected_irreps[8], &s8),
            Character::new(&[(UnityRoot::new(1, 16), 1)]),
        ),
        (
            (&expected_irreps[9], &s8),
            Character::new(&[(UnityRoot::new(3, 16), 1)]),
        ),
        (
            (&expected_irreps[10], &s8),
            Character::new(&[(UnityRoot::new(5, 16), 1)]),
        ),
        (
            (&expected_irreps[11], &s8),
            Character::new(&[(UnityRoot::new(7, 16), 1)]),
        ),
        (
            (&expected_irreps[12], &s8),
            Character::new(&[(UnityRoot::new(9, 16), 1)]),
        ),
        (
            (&expected_irreps[13], &s8),
            Character::new(&[(UnityRoot::new(11, 16), 1)]),
        ),
        (
            (&expected_irreps[14], &s8),
            Character::new(&[(UnityRoot::new(13, 16), 1)]),
        ),
        (
            (&expected_irreps[15], &s8),
            Character::new(&[(UnityRoot::new(15, 16), 1)]),
        ),
    ]);
    test_chartab_ordinary_double_group(&mol, thresh, "S8*", &expected_irreps, Some(expected_chars));
}

#[test]
fn test_chartab_symmetric_antiprism_pb10_magnetic_field_bw_d4d_s8_double() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/pb10.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||E|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||E|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||E|_(3)|").unwrap(),
        MullikenIrrepSymbol::new("||E~|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||E~|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||E~|_(3)|").unwrap(),
        MullikenIrrepSymbol::new("||E~|_(4)|").unwrap(),
    ];
    let s8 = SymmetryClassSymbol::<SymmetryOperation>::new("2||S8(Σ)||", None).unwrap();
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
        (
            (&expected_irreps[7], &s8),
            Character::new(&[(UnityRoot::new(1, 16), 1), (UnityRoot::new(15, 16), 1)]),
        ),
        (
            (&expected_irreps[8], &s8),
            Character::new(&[(UnityRoot::new(3, 16), 1), (UnityRoot::new(13, 16), 1)]),
        ),
        (
            (&expected_irreps[9], &s8),
            Character::new(&[(UnityRoot::new(5, 16), 1), (UnityRoot::new(11, 16), 1)]),
        ),
        (
            (&expected_irreps[10], &s8),
            Character::new(&[(UnityRoot::new(7, 16), 1), (UnityRoot::new(9, 16), 1)]),
        ),
    ]);
    test_chartab_magnetic_double_group(
        &mol,
        thresh,
        "D4d*",
        &expected_irreps,
        Some(expected_chars),
    );
}

#[test]
fn test_chartab_symmetric_staggered_ferrocene_magnetic_field_s10_double() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/staggered_ferrocene.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|Γ|_(1g)|").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|Γ|_(2g)|").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|Γ|_(2g)|").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|Γ|_(1g)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(u)|").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|Γ|_(1u)|").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|Γ|_(2u)|").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|Γ|_(2u)|").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|Γ|_(1u)|").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|Γ~|_(1g)|").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|Γ~|_(2g)|").unwrap(),
        MullikenIrrepSymbol::new("||B~|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|Γ~|_(2g)|").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|Γ~|_(1g)|").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|Γ~|_(1u)|").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|Γ~|_(2u)|").unwrap(),
        MullikenIrrepSymbol::new("||B~|_(u)|").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|Γ~|_(2u)|").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|Γ~|_(1u)|").unwrap(),
    ];
    let c5 = SymmetryClassSymbol::<SymmetryOperation>::new("1||C5(Σ)||", None).unwrap();
    let s10 = SymmetryClassSymbol::<SymmetryOperation>::new("1||S10(Σ)||", None).unwrap();
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
            (&expected_irreps[10], &c5),
            Character::new(&[(UnityRoot::new(1, 10), 1)]),
        ),
        (
            (&expected_irreps[11], &c5),
            Character::new(&[(UnityRoot::new(3, 10), 1)]),
        ),
        (
            (&expected_irreps[12], &c5),
            Character::new(&[(UnityRoot::new(5, 10), 1)]),
        ),
        (
            (&expected_irreps[13], &c5),
            Character::new(&[(UnityRoot::new(7, 10), 1)]),
        ),
        (
            (&expected_irreps[14], &c5),
            Character::new(&[(UnityRoot::new(9, 10), 1)]),
        ),
        (
            (&expected_irreps[15], &c5),
            Character::new(&[(UnityRoot::new(1, 10), 1)]),
        ),
        (
            (&expected_irreps[16], &c5),
            Character::new(&[(UnityRoot::new(3, 10), 1)]),
        ),
        (
            (&expected_irreps[17], &c5),
            Character::new(&[(UnityRoot::new(5, 10), 1)]),
        ),
        (
            (&expected_irreps[18], &c5),
            Character::new(&[(UnityRoot::new(7, 10), 1)]),
        ),
        (
            (&expected_irreps[19], &c5),
            Character::new(&[(UnityRoot::new(9, 10), 1)]),
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
        (
            (&expected_irreps[10], &s10),
            Character::new(&[(UnityRoot::new(4, 5), 1)]),
        ),
        (
            (&expected_irreps[11], &s10),
            Character::new(&[(UnityRoot::new(2, 5), 1)]),
        ),
        (
            (&expected_irreps[12], &s10),
            Character::new(&[(UnityRoot::new(0, 5), 1)]),
        ),
        (
            (&expected_irreps[13], &s10),
            Character::new(&[(UnityRoot::new(3, 5), 1)]),
        ),
        (
            (&expected_irreps[14], &s10),
            Character::new(&[(UnityRoot::new(1, 5), 1)]),
        ),
        (
            (&expected_irreps[15], &s10),
            Character::new(&[(UnityRoot::new(3, 10), 1)]),
        ),
        (
            (&expected_irreps[16], &s10),
            Character::new(&[(UnityRoot::new(9, 10), 1)]),
        ),
        (
            (&expected_irreps[17], &s10),
            Character::new(&[(UnityRoot::new(5, 10), 1)]),
        ),
        (
            (&expected_irreps[18], &s10),
            Character::new(&[(UnityRoot::new(1, 10), 1)]),
        ),
        (
            (&expected_irreps[19], &s10),
            Character::new(&[(UnityRoot::new(7, 10), 1)]),
        ),
    ]);
    test_chartab_ordinary_double_group(
        &mol,
        thresh,
        "S10*",
        &expected_irreps,
        Some(expected_chars),
    );
}

#[test]
fn test_chartab_symmetric_c60_magnetic_field_bw_d5d_s10_double() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c60.xyz");
    let thresh = 1e-5;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(1g)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(2g)|").unwrap(),
        MullikenIrrepSymbol::new("||E|_(1g)|").unwrap(),
        MullikenIrrepSymbol::new("||E|_(2g)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(1u)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(2u)|").unwrap(),
        MullikenIrrepSymbol::new("||E|_(1u)|").unwrap(),
        MullikenIrrepSymbol::new("||E|_(2u)|").unwrap(),
        MullikenIrrepSymbol::new("||B~|_(1g)|").unwrap(),
        MullikenIrrepSymbol::new("||B~|_(2g)|").unwrap(),
        MullikenIrrepSymbol::new("||E~|_(1g)|").unwrap(),
        MullikenIrrepSymbol::new("||E~|_(2g)|").unwrap(),
        MullikenIrrepSymbol::new("||B~|_(1u)|").unwrap(),
        MullikenIrrepSymbol::new("||B~|_(2u)|").unwrap(),
        MullikenIrrepSymbol::new("||E~|_(1u)|").unwrap(),
        MullikenIrrepSymbol::new("||E~|_(2u)|").unwrap(),
    ];
    let c5 = SymmetryClassSymbol::<SymmetryOperation>::new("2||C5(Σ)||", None).unwrap();
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
        (
            (&expected_irreps[8], &c5),
            Character::new(&[(UnityRoot::new(5, 10), 1)]),
        ),
        (
            (&expected_irreps[9], &c5),
            Character::new(&[(UnityRoot::new(5, 10), 1)]),
        ),
        (
            (&expected_irreps[10], &c5),
            Character::new(&[(UnityRoot::new(1, 10), 1), (UnityRoot::new(9, 10), 1)]),
        ),
        (
            (&expected_irreps[11], &c5),
            Character::new(&[(UnityRoot::new(3, 10), 1), (UnityRoot::new(7, 10), 1)]),
        ),
        (
            (&expected_irreps[12], &c5),
            Character::new(&[(UnityRoot::new(5, 10), 1)]),
        ),
        (
            (&expected_irreps[13], &c5),
            Character::new(&[(UnityRoot::new(5, 10), 1)]),
        ),
        (
            (&expected_irreps[14], &c5),
            Character::new(&[(UnityRoot::new(1, 10), 1), (UnityRoot::new(9, 10), 1)]),
        ),
        (
            (&expected_irreps[15], &c5),
            Character::new(&[(UnityRoot::new(3, 10), 1), (UnityRoot::new(7, 10), 1)]),
        ),
    ]);
    test_chartab_magnetic_double_group(
        &mol,
        thresh,
        "D5d*",
        &expected_irreps,
        Some(expected_chars),
    );
}

#[test]
fn test_chartab_symmetric_au26_magnetic_field_s12_double() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/au26.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A||").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|Γ|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|Γ|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|Γ|_(3)|").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|Γ|_(4)|").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|Γ|_(5)|").unwrap(),
        MullikenIrrepSymbol::new("||B||").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|Γ|_(5)|").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|Γ|_(4)|").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|Γ|_(3)|").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|Γ|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|Γ|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|Γ~|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|Γ~|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|Γ~|_(3)|").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|Γ~|_(4)|").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|Γ~|_(5)|").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|Γ~|_(6)|").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|Γ~|_(6)|").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|Γ~|_(5)|").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|Γ~|_(4)|").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|Γ~|_(3)|").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|Γ~|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|Γ~|_(1)|").unwrap(),
    ];
    let s12 = SymmetryClassSymbol::<SymmetryOperation>::new("1||S12(Σ)||", None).unwrap();
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
        (
            (&expected_irreps[12], &s12),
            Character::new(&[(UnityRoot::new(1, 24), 1)]),
        ),
        (
            (&expected_irreps[13], &s12),
            Character::new(&[(UnityRoot::new(3, 24), 1)]),
        ),
        (
            (&expected_irreps[14], &s12),
            Character::new(&[(UnityRoot::new(5, 24), 1)]),
        ),
        (
            (&expected_irreps[15], &s12),
            Character::new(&[(UnityRoot::new(7, 24), 1)]),
        ),
        (
            (&expected_irreps[16], &s12),
            Character::new(&[(UnityRoot::new(9, 24), 1)]),
        ),
        (
            (&expected_irreps[17], &s12),
            Character::new(&[(UnityRoot::new(11, 24), 1)]),
        ),
        (
            (&expected_irreps[18], &s12),
            Character::new(&[(UnityRoot::new(13, 24), 1)]),
        ),
        (
            (&expected_irreps[19], &s12),
            Character::new(&[(UnityRoot::new(15, 24), 1)]),
        ),
        (
            (&expected_irreps[20], &s12),
            Character::new(&[(UnityRoot::new(17, 24), 1)]),
        ),
        (
            (&expected_irreps[21], &s12),
            Character::new(&[(UnityRoot::new(19, 24), 1)]),
        ),
        (
            (&expected_irreps[22], &s12),
            Character::new(&[(UnityRoot::new(21, 24), 1)]),
        ),
        (
            (&expected_irreps[23], &s12),
            Character::new(&[(UnityRoot::new(23, 24), 1)]),
        ),
    ]);
    test_chartab_ordinary_double_group(
        &mol,
        thresh,
        "S12*",
        &expected_irreps,
        Some(expected_chars),
    );
}

#[test]
fn test_chartab_symmetric_au26_magnetic_field_bw_d6d_s12_double() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/au26.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
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
        MullikenIrrepSymbol::new("||E~|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||E~|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||E~|_(3)|").unwrap(),
        MullikenIrrepSymbol::new("||E~|_(4)|").unwrap(),
        MullikenIrrepSymbol::new("||E~|_(5)|").unwrap(),
        MullikenIrrepSymbol::new("||E~|_(6)|").unwrap(),
    ];
    let s12 = SymmetryClassSymbol::<SymmetryOperation>::new("2||S12(Σ)||", None).unwrap();
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
        (
            (&expected_irreps[9], &s12),
            Character::new(&[(UnityRoot::new(1, 24), 1), (UnityRoot::new(23, 24), 1)]),
        ),
        (
            (&expected_irreps[10], &s12),
            Character::new(&[(UnityRoot::new(3, 24), 1), (UnityRoot::new(21, 24), 1)]),
        ),
        (
            (&expected_irreps[11], &s12),
            Character::new(&[(UnityRoot::new(5, 24), 1), (UnityRoot::new(19, 24), 1)]),
        ),
        (
            (&expected_irreps[12], &s12),
            Character::new(&[(UnityRoot::new(7, 24), 1), (UnityRoot::new(17, 24), 1)]),
        ),
        (
            (&expected_irreps[13], &s12),
            Character::new(&[(UnityRoot::new(9, 24), 1), (UnityRoot::new(15, 24), 1)]),
        ),
        (
            (&expected_irreps[14], &s12),
            Character::new(&[(UnityRoot::new(11, 24), 1), (UnityRoot::new(13, 24), 1)]),
        ),
    ]);
    test_chartab_magnetic_double_group(
        &mol,
        thresh,
        "D6d*",
        &expected_irreps,
        Some(expected_chars),
    );
}

/*********
Asymmetric
*********/
/*
C2
*/
#[test]
fn test_chartab_asymmetric_spiroketal_c2() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/spiroketal.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    verify_c2(&mol, thresh);
}

#[test]
fn test_chartab_asymmetric_spiroketal_grey_c2() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/spiroketal.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    verify_grey_c2(&mol, thresh);
}

#[test]
fn test_chartab_asymmetric_cyclohexene_c2() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/cyclohexene.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    verify_c2(&mol, thresh);
}

#[test]
fn test_chartab_asymmetric_cyclohexene_grey_c2() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/cyclohexene.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    verify_grey_c2(&mol, thresh);
}

#[test]
fn test_chartab_asymmetric_thf_c2() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/thf.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    verify_c2(&mol, thresh);
}

#[test]
fn test_chartab_asymmetric_thf_grey_c2() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/thf.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    verify_grey_c2(&mol, thresh);
}

#[test]
fn test_chartab_asymmetric_tartaricacid_c2() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/tartaricacid.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    verify_c2(&mol, thresh);
}

#[test]
fn test_chartab_asymmetric_tartaricacid_grey_c2() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/tartaricacid.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    verify_grey_c2(&mol, thresh);
}

#[test]
fn test_chartab_asymmetric_f2allene_c2() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/f2allene.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    verify_c2(&mol, thresh);
}

#[test]
fn test_chartab_asymmetric_f2allene_grey_c2() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/f2allene.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    verify_grey_c2(&mol, thresh);
}

#[test]
fn test_chartab_asymmetric_water_magnetic_field_c2() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/water.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 1.0, 0.0)));
    verify_c2(&mol, thresh);
}

#[test]
fn test_chartab_asymmetric_water_magnetic_field_bw_c2v_c2() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/water.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 1.0, 0.0)));
    verify_bw_c2v_c2(&mol, thresh);
}

#[test]
fn test_chartab_asymmetric_pyridine_magnetic_field_c2() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/pyridine.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.2, 0.0)));
    verify_c2(&mol, thresh);
}

#[test]
fn test_chartab_asymmetric_pyridine_magnetic_field_bw_c2v_c2() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/pyridine.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.2, 0.0)));
    verify_bw_c2v_c2(&mol, thresh);
}

#[test]
fn test_chartab_asymmetric_cyclobutene_magnetic_field_c2() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/cyclobutene.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.2, 0.0, 0.0)));
    verify_c2(&mol, thresh);
}

#[test]
fn test_chartab_asymmetric_cyclobutene_magnetic_field_bw_c2v_c2() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/cyclobutene.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.2, 0.0, 0.0)));
    verify_bw_c2v_c2(&mol, thresh);
}

#[test]
fn test_chartab_asymmetric_azulene_magnetic_field_c2() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/azulene.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 0.2)));
    verify_c2(&mol, thresh);
}

#[test]
fn test_chartab_asymmetric_azulene_magnetic_field_bw_c2v_c2() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/azulene.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 0.2)));
    verify_bw_c2v_c2(&mol, thresh);
}

#[test]
fn test_chartab_asymmetric_cis_cocl2h4o2_magnetic_field_c2() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/cis-cocl2h4o2.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.2, 0.0)));
    verify_c2(&mol, thresh);
}

#[test]
fn test_chartab_asymmetric_cis_cocl2h4o2_magnetic_field_bw_c2v_c2() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/cis-cocl2h4o2.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.2, 0.0)));
    verify_bw_c2v_c2(&mol, thresh);
}

#[test]
fn test_chartab_asymmetric_cuneane_magnetic_field_c2() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/cuneane.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 0.2)));
    verify_c2(&mol, thresh);
}

#[test]
fn test_chartab_asymmetric_cuneane_magnetic_field_bw_c2v_c2() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/cuneane.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 0.2)));
    verify_bw_c2v_c2(&mol, thresh);
}

/// Verifies the validity of the computed $`\mathcal{C}_{2}`$ character table of irreps.
///
/// # Arguments
///
/// * `mol` - A reference to a [`Molecule`] structure.
/// * `thresh` - A threshold for symmetry detection.
///
/// # Panics
///
/// Panics when any expected condition is not fulfilled.
fn verify_c2(mol: &Molecule, thresh: f64) {
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A||").unwrap(),
        MullikenIrrepSymbol::new("||B||").unwrap(),
    ];
    let c2 = SymmetryClassSymbol::<SymmetryOperation>::new("1||C2||", None).unwrap();
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
    test_chartab_ordinary_group(&mol, thresh, "C2", &expected_irreps, Some(expected_chars));
}

/// Verifies the validity of the computed $`\mathcal{C}_{2} + \theta\mathcal{C}_{2}`$ character
/// table of irreps.
///
/// # Arguments
///
/// * `mol` - A reference to a [`Molecule`] structure.
/// * `thresh` - A threshold for symmetry detection.
///
/// # Panics
///
/// Panics when any expected condition is not fulfilled.
fn verify_grey_c2(mol: &Molecule, thresh: f64) {
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A||").unwrap(),
        MullikenIrrepSymbol::new("||B||").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|A||").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|B||").unwrap(),
    ];
    let tc2 = SymmetryClassSymbol::<SymmetryOperation>::new("1||θ·C2||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &tc2),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[1], &tc2),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[2], &tc2),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[3], &tc2),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
    ]);
    test_chartab_magnetic_group(
        &mol,
        thresh,
        "C2 + θ·C2",
        &expected_irreps,
        Some(expected_chars),
    );
}

/***
C2*
***/
#[test]
fn test_chartab_asymmetric_spiroketal_c2_double() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/spiroketal.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A||").unwrap(),
        MullikenIrrepSymbol::new("||B||").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|Γ~||").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|Γ~||").unwrap(),
    ];
    let c2 = SymmetryClassSymbol::<SymmetryOperation>::new("1||C2(Σ)||", None).unwrap();
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
            Character::new(&[(UnityRoot::new(1, 4), 1)]),
        ),
        (
            (&expected_irreps[3], &c2),
            Character::new(&[(UnityRoot::new(3, 4), 1)]),
        ),
    ]);
    test_chartab_ordinary_double_group(&mol, thresh, "C2*", &expected_irreps, Some(expected_chars));
}

#[test]
fn test_chartab_asymmetric_spiroketal_grey_c2_double() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/spiroketal.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A||").unwrap(),
        MullikenIrrepSymbol::new("||B||").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|A||").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|B||").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|Γ~|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|Γ~|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|Γ~|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|Γ~|_(1)|").unwrap(),
    ];
    let t = SymmetryClassSymbol::<SymmetryOperation>::new("1||θ(Σ)||", None).unwrap();
    let tc2 = SymmetryClassSymbol::<SymmetryOperation>::new("1||θ·C2(Σ)||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &t),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[1], &t),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[2], &t),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[3], &t),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[4], &t),
            Character::new(&[(UnityRoot::new(1, 4), 1)]),
        ),
        (
            (&expected_irreps[5], &t),
            Character::new(&[(UnityRoot::new(1, 4), 1)]),
        ),
        (
            (&expected_irreps[6], &t),
            Character::new(&[(UnityRoot::new(3, 4), 1)]),
        ),
        (
            (&expected_irreps[7], &t),
            Character::new(&[(UnityRoot::new(3, 4), 1)]),
        ),
        (
            (&expected_irreps[0], &tc2),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[1], &tc2),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[2], &tc2),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[3], &tc2),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[4], &tc2),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[5], &tc2),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[6], &tc2),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[7], &tc2),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
    ]);
    test_chartab_magnetic_double_group(
        &mol,
        thresh,
        "(C2 + θ·C2)*",
        &expected_irreps,
        Some(expected_chars),
    );
}

#[test]
fn test_chartab_asymmetric_cis_cocl2h4o2_magnetic_field_bw_c2v_c2_double() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/cis-cocl2h4o2.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.2, 0.0)));
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||E~||").unwrap(),
    ];
    let c2 = SymmetryClassSymbol::<SymmetryOperation>::new("1||C2(Σ), C2(QΣ)||", None).unwrap();
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
            Character::new(&[(UnityRoot::new(0, 2), 1), (UnityRoot::new(1, 2), 1)]),
        ),
    ]);
    test_chartab_magnetic_double_group(
        &mol,
        thresh,
        "C2v*",
        &expected_irreps,
        Some(expected_chars),
    );
}

/***
C2v
***/
#[test]
fn test_chartab_asymmetric_water_c2v() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/water.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    verify_c2v(&mol, thresh);
}

#[test]
fn test_chartab_asymmetric_water_grey_c2v() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/water.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    verify_grey_c2v(&mol, thresh);
}

#[test]
fn test_chartab_asymmetric_pyridine_c2v() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/pyridine.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    verify_c2v(&mol, thresh);
}

#[test]
fn test_chartab_asymmetric_pyridine_grey_c2v() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/pyridine.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    verify_grey_c2v(&mol, thresh);
}

#[test]
fn test_chartab_asymmetric_cyclobutene_c2v() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/cyclobutene.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    verify_c2v(&mol, thresh);
}

#[test]
fn test_chartab_asymmetric_cyclobutene_grey_c2v() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/cyclobutene.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    verify_grey_c2v(&mol, thresh);
}

#[test]
fn test_chartab_asymmetric_azulene_c2v() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/azulene.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    verify_c2v(&mol, thresh);
}

#[test]
fn test_chartab_asymmetric_azulene_grey_c2v() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/azulene.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    verify_grey_c2v(&mol, thresh);
}

#[test]
fn test_chartab_asymmetric_cuneane_c2v() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/cuneane.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    verify_c2v(&mol, thresh);
}

#[test]
fn test_chartab_asymmetric_cuneane_grey_c2v() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/cuneane.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    verify_grey_c2v(&mol, thresh);
}

#[test]
fn test_chartab_asymmetric_bf3_electric_field_c2v() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/bf3.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(Vector3::new(0.2, 0.0, 0.0)));
    verify_c2v(&mol, thresh);
}

#[test]
fn test_chartab_asymmetric_bf3_electric_field_grey_c2v() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/bf3.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(Vector3::new(0.2, 0.0, 0.0)));
    verify_grey_c2v(&mol, thresh);
}

/// Verifies the validity of the computed $`\mathcal{C}_{2v}`$ character table of irreps.
///
/// # Arguments
///
/// * `mol` - A reference to a [`Molecule`] structure.
/// * `thresh` - A threshold for symmetry detection.
/// * `magnetic` - A flag indicating if this is a magnetic group.
///
/// # Panics
///
/// Panics when any expected condition is not fulfilled.
fn verify_c2v(mol: &Molecule, thresh: f64) {
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(2)|").unwrap(),
    ];
    let c2 = SymmetryClassSymbol::<SymmetryOperation>::new("1||C2||", None).unwrap();
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
    ]);
    test_chartab_ordinary_group(&mol, thresh, "C2v", &expected_irreps, Some(expected_chars));
}

/// Verifies the validity of the computed $`\mathcal{C}_{2v}(\mathcal{C}_{2})`$ character table of irreps.
///
/// # Arguments
///
/// * `mol` - A reference to a [`Molecule`] structure.
/// * `thresh` - A threshold for symmetry detection.
/// * `magnetic` - A flag indicating if this is a magnetic group.
///
/// # Panics
///
/// Panics when any expected condition is not fulfilled.
fn verify_bw_c2v_c2(mol: &Molecule, thresh: f64) {
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(2)|").unwrap(),
    ];
    let c2 = SymmetryClassSymbol::<SymmetryOperation>::new("1||C2||", None).unwrap();
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
    ]);
    test_chartab_magnetic_group(&mol, thresh, "C2v", &expected_irreps, Some(expected_chars));
}

/// Verifies the validity of the computed $`\mathcal{C}_{2v}(\mathcal{C}_{s})`$ character table of irreps.
///
/// # Arguments
///
/// * `mol` - A reference to a [`Molecule`] structure.
/// * `thresh` - A threshold for symmetry detection.
/// * `magnetic` - A flag indicating if this is a magnetic group.
///
/// # Panics
///
/// Panics when any expected condition is not fulfilled.
fn verify_bw_c2v_cs(mol: &Molecule, thresh: f64) {
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|^(')_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||A|^(')_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||A|^('')_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||A|^('')_(2)|").unwrap(),
    ];
    let tc2 = SymmetryClassSymbol::<SymmetryOperation>::new("1||θ·C2||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &tc2),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[1], &tc2),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[2], &tc2),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[3], &tc2),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
    ]);
    test_chartab_magnetic_group(&mol, thresh, "C2v", &expected_irreps, Some(expected_chars));
}

/// Verifies the validity of the computed $`\mathcal{C}_{2v} + \theta\mathcal{C}_{2v}`$ character
/// table of irreps.
///
/// # Arguments
///
/// * `mol` - A reference to a [`Molecule`] structure.
/// * `thresh` - A threshold for symmetry detection.
///
/// # Panics
///
/// Panics when any expected condition is not fulfilled.
fn verify_grey_c2v(mol: &Molecule, thresh: f64) {
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|A|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|A|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|B|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|B|_(2)|").unwrap(),
    ];
    let tc2 = SymmetryClassSymbol::<SymmetryOperation>::new("1||θ·C2||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &tc2),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[1], &tc2),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[2], &tc2),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[3], &tc2),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[4], &tc2),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[5], &tc2),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[6], &tc2),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[7], &tc2),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
    ]);
    test_chartab_magnetic_group(
        &mol,
        thresh,
        "C2v + θ·C2v",
        &expected_irreps,
        Some(expected_chars),
    );
}

/***
C2v*
***/
#[test]
fn test_chartab_asymmetric_water_c2v_double() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/water.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||E~||").unwrap(),
    ];
    let c2 = SymmetryClassSymbol::<SymmetryOperation>::new("1||C2(Σ), C2(QΣ)||", None).unwrap();
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
            Character::new(&[(UnityRoot::new(0, 2), 1), (UnityRoot::new(1, 2), 1)]),
        ),
    ]);
    test_chartab_ordinary_double_group(
        &mol,
        thresh,
        "C2v*",
        &expected_irreps,
        Some(expected_chars),
    );
}

#[test]
fn test_chartab_asymmetric_bf3_electric_field_grey_c2v_double() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/bf3.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(Vector3::new(0.2, 0.0, 0.0)));
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|A|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|A|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|B|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|B|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|E~||").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|E~||").unwrap(),
    ];
    let t = SymmetryClassSymbol::<SymmetryOperation>::new("1||θ(Σ)||", None).unwrap();
    let tc2 =
        SymmetryClassSymbol::<SymmetryOperation>::new("1||θ·C2(Σ), θ·C2(QΣ)||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &t),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[1], &t),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[2], &t),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[3], &t),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[4], &t),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[5], &t),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[6], &t),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[7], &t),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[8], &t),
            Character::new(&[(UnityRoot::new(1, 4), 2)]),
        ),
        (
            (&expected_irreps[9], &t),
            Character::new(&[(UnityRoot::new(3, 4), 2)]),
        ),
        (
            (&expected_irreps[0], &tc2),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[1], &tc2),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[2], &tc2),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[3], &tc2),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[4], &tc2),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[5], &tc2),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[6], &tc2),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[7], &tc2),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[8], &tc2),
            Character::new(&[(UnityRoot::new(0, 2), 1), (UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[9], &tc2),
            Character::new(&[(UnityRoot::new(0, 2), 1), (UnityRoot::new(1, 2), 1)]),
        ),
    ]);
    test_chartab_magnetic_double_group(
        &mol,
        thresh,
        "(C2v + θ·C2v)*",
        &expected_irreps,
        Some(expected_chars),
    );
}

/***
C2h
***/
#[test]
fn test_chartab_asymmetric_h2o2_c2h() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/h2o2.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    verify_c2h(&mol, thresh);
}

#[test]
fn test_chartab_asymmetric_h2o2_grey_c2h() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/h2o2.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    verify_grey_c2h(&mol, thresh);
}

#[test]
fn test_chartab_asymmetric_zethrene_c2h() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/zethrene.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    verify_c2h(&mol, thresh);
}

#[test]
fn test_chartab_asymmetric_zethrene_grey_c2h() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/zethrene.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    verify_grey_c2h(&mol, thresh);
}

#[test]
fn test_chartab_asymmetric_distorted_vf6_magnetic_field_c2h() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/vf6_d2h.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(1.0, 0.0, 0.0)));
    verify_c2h(&mol, thresh);
}

#[test]
fn test_chartab_asymmetric_distorted_vf6_magnetic_field_bw_d2h_c2h() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/vf6_d2h.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(1.0, 0.0, 0.0)));
    verify_d2h(&mol, thresh, true);
}

#[test]
fn test_chartab_asymmetric_b2h6_magnetic_field_c2h() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/b2h6.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(1.0, 0.0, 0.0)));
    verify_c2h(&mol, thresh);
}

#[test]
fn test_chartab_asymmetric_b2h6_magnetic_field_bw_d2h_c2h() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/b2h6.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(1.0, 0.0, 0.0)));
    verify_d2h(&mol, thresh, true);
}

#[test]
fn test_chartab_asymmetric_naphthalene_magnetic_field_c2h() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/naphthalene.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 1.0, 0.0)));
    verify_c2h(&mol, thresh);
}

#[test]
fn test_chartab_asymmetric_naphthalene_magnetic_field_bw_d2h_c2h() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/naphthalene.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 1.0, 0.0)));
    verify_d2h(&mol, thresh, true);
}

#[test]
fn test_chartab_asymmetric_pyrene_magnetic_field_c2h() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/pyrene.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    verify_c2h(&mol, thresh);
}

#[test]
fn test_chartab_asymmetric_pyrene_magnetic_field_bw_d2h_c2h() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/pyrene.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    verify_d2h(&mol, thresh, true);
}

#[test]
fn test_chartab_asymmetric_c6o6_magnetic_field_c2h() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c6o6.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    verify_c2h(&mol, thresh);
}

#[test]
fn test_chartab_asymmetric_c6o6_magnetic_field_bw_d2h_c2h() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c6o6.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    verify_d2h(&mol, thresh, true);
}

/// Verifies the validity of the computed $`\mathcal{C}_{2h}`$ character table of irreps.
///
/// # Arguments
///
/// * `mol` - A reference to a [`Molecule`] structure.
/// * `thresh` - A threshold for symmetry detection.
///
/// # Panics
///
/// Panics when any expected condition is not fulfilled.
fn verify_c2h(mol: &Molecule, thresh: f64) {
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(u)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(u)|").unwrap(),
    ];
    let c2 = SymmetryClassSymbol::<SymmetryOperation>::new("1||C2||", None).unwrap();
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
    test_chartab_ordinary_group(&mol, thresh, "C2h", &expected_irreps, Some(expected_chars));
}

/// Verifies the validity of the computed $`\mathcal{C}_{2h} + \theta\mathcal{C}_{2h}`$ character
/// table of irreps.
///
/// # Arguments
///
/// * `mol` - A reference to a [`Molecule`] structure.
/// * `thresh` - A threshold for symmetry detection.
///
/// # Panics
///
/// Panics when any expected condition is not fulfilled.
fn verify_grey_c2h(mol: &Molecule, thresh: f64) {
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(u)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(u)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|A|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|B|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|A|_(u)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|B|_(u)|").unwrap(),
    ];
    let tc2 = SymmetryClassSymbol::<SymmetryOperation>::new("1||θ·C2||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &tc2),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[1], &tc2),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[2], &tc2),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[3], &tc2),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[4], &tc2),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[5], &tc2),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[6], &tc2),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[7], &tc2),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
    ]);
    test_chartab_magnetic_group(
        &mol,
        thresh,
        "C2h + θ·C2h",
        &expected_irreps,
        Some(expected_chars),
    );
}

/***
C2h*
***/
#[test]
fn test_chartab_asymmetric_h2o2_c2h_double() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/h2o2.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(u)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(u)|").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|Γ~|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|Γ~|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|Γ~|_(u)|").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|Γ~|_(u)|").unwrap(),
    ];
    let c2 = SymmetryClassSymbol::<SymmetryOperation>::new("1||C2(Σ)||", None).unwrap();
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
    test_chartab_ordinary_double_group(
        &mol,
        thresh,
        "C2h*",
        &expected_irreps,
        Some(expected_chars),
    );
}

#[test]
fn test_chartab_asymmetric_h2o2_grey_c2h_double() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/h2o2.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(u)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(u)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|A|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|B|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|A|_(u)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|B|_(u)|").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|Γ~|_(1g)|").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|Γ~|_(2g)|").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|Γ~|_(1u)|").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|Γ~|_(2u)|").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|Γ~|_(2g)|").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|Γ~|_(1g)|").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|Γ~|_(2u)|").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|Γ~|_(1u)|").unwrap(),
    ];
    let t = SymmetryClassSymbol::<SymmetryOperation>::new("1||θ(Σ)||", None).unwrap();
    let tc2 = SymmetryClassSymbol::<SymmetryOperation>::new("1||θ·C2(Σ)||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &t),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[1], &t),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[2], &t),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[3], &t),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[4], &t),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[5], &t),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[6], &t),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[7], &t),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[8], &t),
            Character::new(&[(UnityRoot::new(1, 4), 1)]),
        ),
        (
            (&expected_irreps[9], &t),
            Character::new(&[(UnityRoot::new(1, 4), 1)]),
        ),
        (
            (&expected_irreps[10], &t),
            Character::new(&[(UnityRoot::new(1, 4), 1)]),
        ),
        (
            (&expected_irreps[11], &t),
            Character::new(&[(UnityRoot::new(1, 4), 1)]),
        ),
        (
            (&expected_irreps[12], &t),
            Character::new(&[(UnityRoot::new(3, 4), 1)]),
        ),
        (
            (&expected_irreps[13], &t),
            Character::new(&[(UnityRoot::new(3, 4), 1)]),
        ),
        (
            (&expected_irreps[14], &t),
            Character::new(&[(UnityRoot::new(3, 4), 1)]),
        ),
        (
            (&expected_irreps[15], &t),
            Character::new(&[(UnityRoot::new(3, 4), 1)]),
        ),
        (
            (&expected_irreps[0], &tc2),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[1], &tc2),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[2], &tc2),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[3], &tc2),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[4], &tc2),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[5], &tc2),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[6], &tc2),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[7], &tc2),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[8], &tc2),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[9], &tc2),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[10], &tc2),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[11], &tc2),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[12], &tc2),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[13], &tc2),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[14], &tc2),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[15], &tc2),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
    ]);
    test_chartab_magnetic_double_group(
        &mol,
        thresh,
        "(C2h + θ·C2h)*",
        &expected_irreps,
        Some(expected_chars),
    );
}

#[test]
fn test_chartab_asymmetric_naphthalene_magnetic_field_bw_d2h_c2h_double() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/naphthalene.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 1.0, 0.0)));
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(1g)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(2g)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(1g)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(2g)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(1u)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(2u)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(1u)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(2u)|").unwrap(),
        MullikenIrrepSymbol::new("||E~|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("||E~|_(u)|").unwrap(),
    ];
    let c2 = SymmetryClassSymbol::<SymmetryOperation>::new("1||C2(Σ), C2(QΣ)||", None).unwrap();
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
            (&expected_irreps[8], &c2),
            Character::new(&[(UnityRoot::new(0, 2), 1), (UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[9], &c2),
            Character::new(&[(UnityRoot::new(0, 2), 1), (UnityRoot::new(1, 2), 1)]),
        ),
    ]);
    test_chartab_magnetic_double_group(
        &mol,
        thresh,
        "D2h*",
        &expected_irreps,
        Some(expected_chars),
    );
}

/*
Cs
*/
#[test]
fn test_chartab_asymmetric_propene_cs() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/propene.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    verify_cs(&mol, thresh);
}

#[test]
fn test_chartab_asymmetric_propene_grey_cs() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/propene.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    verify_grey_cs(&mol, thresh);
}

#[test]
fn test_chartab_asymmetric_socl2_cs() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/socl2.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    verify_cs(&mol, thresh);
}

#[test]
fn test_chartab_asymmetric_socl2_grey_cs() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/socl2.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    verify_grey_cs(&mol, thresh);
}

#[test]
fn test_chartab_asymmetric_hocl_cs() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/hocl.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    verify_cs(&mol, thresh);
}

#[test]
fn test_chartab_asymmetric_hocl_grey_cs() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/hocl.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    verify_grey_cs(&mol, thresh);
}

#[test]
fn test_chartab_asymmetric_hocn_cs() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/hocn.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    verify_cs(&mol, thresh);
}

#[test]
fn test_chartab_asymmetric_hocn_grey_cs() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/hocn.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    verify_grey_cs(&mol, thresh);
}

#[test]
fn test_chartab_asymmetric_nh2f_cs() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/nh2f.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    verify_cs(&mol, thresh);
}

#[test]
fn test_chartab_asymmetric_nh2f_grey_cs() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/nh2f.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    verify_grey_cs(&mol, thresh);
}

#[test]
fn test_chartab_asymmetric_phenol_cs() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/phenol.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    verify_cs(&mol, thresh);
}

#[test]
fn test_chartab_asymmetric_phenol_grey_cs() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/phenol.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    verify_grey_cs(&mol, thresh);
}

#[test]
fn test_chartab_asymmetric_f_pyrrole_cs() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/f-pyrrole.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    verify_cs(&mol, thresh);
}

#[test]
fn test_chartab_asymmetric_f_pyrrole_grey_cs() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/f-pyrrole.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    verify_grey_cs(&mol, thresh);
}

#[test]
fn test_chartab_asymmetric_n2o_cs() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/n2o.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    verify_cs(&mol, thresh);
}

#[test]
fn test_chartab_asymmetric_n2o_grey_cs() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/n2o.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    verify_grey_cs(&mol, thresh);
}

#[test]
fn test_chartab_asymmetric_fclbenzene_cs() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/fclbenzene.xyz");
    let thresh = 1e-5;
    let mol = Molecule::from_xyz(&path, thresh);
    verify_cs(&mol, thresh);
}

#[test]
fn test_chartab_asymmetric_fclbenzene_grey_cs() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/fclbenzene.xyz");
    let thresh = 1e-5;
    let mol = Molecule::from_xyz(&path, thresh);
    verify_grey_cs(&mol, thresh);
}

#[test]
fn test_chartab_asymmetric_water_magnetic_field_cs() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/water.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(1.0, 0.0, 0.0)));
    verify_cs(&mol, thresh);
}

#[test]
fn test_chartab_asymmetric_water_magnetic_field_bw_c2v_cs() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/water.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(1.0, 0.0, 0.0)));
    verify_bw_c2v_cs(&mol, thresh);
}

#[test]
fn test_chartab_asymmetric_pyridine_magnetic_field_cs() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/pyridine.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 0.2)));
    verify_cs(&mol, thresh);
}

#[test]
fn test_chartab_asymmetric_pyridine_magnetic_field_bw_c2v_cs() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/pyridine.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 0.2)));
    verify_bw_c2v_cs(&mol, thresh);
}

#[test]
fn test_chartab_asymmetric_cyclobutene_magnetic_field_cs() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/cyclobutene.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.1, 0.0)));
    verify_cs(&mol, thresh);
}

#[test]
fn test_chartab_asymmetric_cyclobutene_magnetic_field_bw_c2v_cs() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/cyclobutene.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.1, 0.0)));
    verify_bw_c2v_cs(&mol, thresh);
}

#[test]
fn test_chartab_asymmetric_azulene_magnetic_field_cs() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/azulene.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.1, 0.0)));
    verify_cs(&mol, thresh);
}

#[test]
fn test_chartab_asymmetric_azulene_magnetic_field_bw_c2v_cs() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/azulene.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.1, 0.0)));
    verify_bw_c2v_cs(&mol, thresh);
}

#[test]
fn test_chartab_asymmetric_cis_cocl2h4o2_magnetic_field_cs() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/cis-cocl2h4o2.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 0.2)));
    verify_cs(&mol, thresh);
}

#[test]
fn test_chartab_asymmetric_cis_cocl2h4o2_magnetic_field_bw_c2v_cs() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/cis-cocl2h4o2.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 0.2)));
    verify_bw_c2v_cs(&mol, thresh);
}

#[test]
fn test_chartab_asymmetric_cuneane_magnetic_field_cs() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/cuneane.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.5, 0.0)));
    verify_cs(&mol, thresh);
}

#[test]
fn test_chartab_asymmetric_cuneane_magnetic_field_bw_c2v_cs() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/cuneane.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.5, 0.0)));
    verify_bw_c2v_cs(&mol, thresh);
}

#[test]
fn test_chartab_asymmetric_water_electric_field_cs() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/water.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(Vector3::new(1.0, 0.0, 0.0)));
    verify_cs(&mol, thresh);
}

#[test]
fn test_chartab_asymmetric_water_electric_field_grey_cs() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/water.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(Vector3::new(1.0, 0.0, 0.0)));
    verify_grey_cs(&mol, thresh);
}

#[test]
fn test_chartab_asymmetric_pyridine_electric_field_cs() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/pyridine.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(Vector3::new(0.0, 0.0, 0.2)));
    verify_cs(&mol, thresh);
}

#[test]
fn test_chartab_asymmetric_pyridine_electric_field_grey_cs() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/pyridine.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(Vector3::new(0.0, 0.0, 0.2)));
    verify_grey_cs(&mol, thresh);
}

#[test]
fn test_chartab_asymmetric_cyclobutene_electric_field_cs() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/cyclobutene.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(Vector3::new(0.0, 0.1, 0.0)));
    verify_cs(&mol, thresh);
}

#[test]
fn test_chartab_asymmetric_cyclobutene_electric_field_grey_cs() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/cyclobutene.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(Vector3::new(0.0, 0.1, 0.0)));
    verify_grey_cs(&mol, thresh);
}

#[test]
fn test_chartab_asymmetric_azulene_electric_field_cs() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/azulene.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(Vector3::new(0.0, 0.1, 0.0)));
    verify_cs(&mol, thresh);
}

#[test]
fn test_chartab_asymmetric_azulene_electric_field_grey_cs() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/azulene.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(Vector3::new(0.0, 0.1, 0.0)));
    verify_grey_cs(&mol, thresh);
}

#[test]
fn test_chartab_asymmetric_cis_cocl2h4o2_electric_field_cs() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/cis-cocl2h4o2.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(Vector3::new(0.0, 0.0, 0.2)));
    verify_cs(&mol, thresh);
}

#[test]
fn test_chartab_asymmetric_cis_cocl2h4o2_electric_field_grey_cs() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/cis-cocl2h4o2.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(Vector3::new(0.0, 0.0, 0.2)));
    verify_grey_cs(&mol, thresh);
}

#[test]
fn test_chartab_asymmetric_cuneane_electric_field_cs() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/cuneane.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(Vector3::new(0.0, 0.5, 0.0)));
    verify_cs(&mol, thresh);
}

#[test]
fn test_chartab_asymmetric_cuneane_electric_field_grey_cs() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/cuneane.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(Vector3::new(0.0, 0.5, 0.0)));
    verify_grey_cs(&mol, thresh);
}

#[test]
fn test_chartab_asymmetric_bf3_magnetic_field_cs() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/bf3.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.5, 0.0)));
    verify_cs(&mol, thresh);
}

#[test]
fn test_chartab_asymmetric_bf3_magnetic_field_bw_c2v_cs() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/bf3.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.5, 0.0)));
    verify_bw_c2v_cs(&mol, thresh);
}

/// This is a special case: Cs point group in a symmetric top.
#[test]
fn test_chartab_symmetric_ch4_magnetic_field_cs() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/ch4.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.1, 0.1, 0.0)));
    verify_cs(&mol, thresh);
}

#[test]
fn test_chartab_symmetric_ch4_magnetic_field_bw_c2v_cs() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/ch4.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.1, 0.1, 0.0)));
    verify_bw_c2v_cs(&mol, thresh);
}

/// This is another special case: Cs point group in a symmetric top.
#[test]
fn test_chartab_symmetric_ch4_electric_field_cs() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/ch4.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(Vector3::new(0.1, 0.1, 0.0)));
    verify_cs(&mol, thresh);
}

/// This is another special case: Cs point group in a symmetric top.
#[test]
fn test_chartab_symmetric_ch4_electric_field_grey_cs() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/ch4.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_electric_field(Some(Vector3::new(0.1, 0.1, 0.0)));
    verify_grey_cs(&mol, thresh);
}

#[test]
fn test_chartab_asymmetric_atom_magnetic_electric_field_cs() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/th.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 0.2)));
    mol.set_electric_field(Some(Vector3::new(0.0, 0.1, 0.0)));
    verify_cs(&mol, thresh);
}

#[test]
fn test_chartab_asymmetric_atom_magnetic_electric_field_bw_c2v_cs() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/th.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 0.2)));
    mol.set_electric_field(Some(Vector3::new(0.0, 0.1, 0.0)));
    verify_bw_c2v_cs(&mol, thresh);
}

/// Verifies the validity of the computed $`\mathcal{C}_{s}`$ character table of irreps.
///
/// # Arguments
///
/// * `mol` - A reference to a [`Molecule`] structure.
/// * `thresh` - A threshold for symmetry detection.
///
/// # Panics
///
/// Panics when any expected condition is not fulfilled.
fn verify_cs(mol: &Molecule, thresh: f64) {
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|^(')|").unwrap(),
        MullikenIrrepSymbol::new("||A|^('')|").unwrap(),
    ];
    let s = SymmetryClassSymbol::<SymmetryOperation>::new("1||σh||", None).unwrap();
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
    test_chartab_ordinary_group(&mol, thresh, "Cs", &expected_irreps, Some(expected_chars));
}

/// Verifies the validity of the computed $`\mathcal{C}_{s} + \theta\mathcal{C}_{s}`$ character
/// table of irreps.
///
/// # Arguments
///
/// * `mol` - A reference to a [`Molecule`] structure.
/// * `thresh` - A threshold for symmetry detection.
///
/// # Panics
///
/// Panics when any expected condition is not fulfilled.
fn verify_grey_cs(mol: &Molecule, thresh: f64) {
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|^(')|").unwrap(),
        MullikenIrrepSymbol::new("||A|^('')|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|A|^(')|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|A|^('')|").unwrap(),
    ];
    let ts = SymmetryClassSymbol::<SymmetryOperation>::new("1||θ·σh||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &ts),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[1], &ts),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[2], &ts),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[3], &ts),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
    ]);
    test_chartab_magnetic_group(
        &mol,
        thresh,
        "Cs + θ·Cs",
        &expected_irreps,
        Some(expected_chars),
    );
}

/***
Cs*
***/
#[test]
fn test_chartab_asymmetric_propene_cs_double() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/propene.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|^(')|").unwrap(),
        MullikenIrrepSymbol::new("||A|^('')|").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|Γ~||").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|Γ~||").unwrap(),
    ];
    let s = SymmetryClassSymbol::<SymmetryOperation>::new("1||σh(Σ)||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &s),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[1], &s),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[2], &s),
            Character::new(&[(UnityRoot::new(1, 4), 1)]),
        ),
        (
            (&expected_irreps[3], &s),
            Character::new(&[(UnityRoot::new(3, 4), 1)]),
        ),
    ]);
    test_chartab_ordinary_double_group(&mol, thresh, "Cs*", &expected_irreps, Some(expected_chars));
}

#[test]
fn test_chartab_asymmetric_propene_grey_cs_double() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/propene.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|^(')|").unwrap(),
        MullikenIrrepSymbol::new("||A|^('')|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|A|^(')|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|A|^('')|").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|Γ~|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|Γ~|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|Γ~|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|Γ~|_(1)|").unwrap(),
    ];
    let t = SymmetryClassSymbol::<SymmetryOperation>::new("1||θ(Σ)||", None).unwrap();
    let ts = SymmetryClassSymbol::<SymmetryOperation>::new("1||θ·σh(Σ)||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &t),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[1], &t),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[2], &t),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[3], &t),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[4], &t),
            Character::new(&[(UnityRoot::new(1, 4), 1)]),
        ),
        (
            (&expected_irreps[5], &t),
            Character::new(&[(UnityRoot::new(1, 4), 1)]),
        ),
        (
            (&expected_irreps[6], &t),
            Character::new(&[(UnityRoot::new(3, 4), 1)]),
        ),
        (
            (&expected_irreps[7], &t),
            Character::new(&[(UnityRoot::new(3, 4), 1)]),
        ),
        (
            (&expected_irreps[0], &ts),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[1], &ts),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[2], &ts),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[3], &ts),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
    ]);
    test_chartab_magnetic_double_group(
        &mol,
        thresh,
        "(Cs + θ·Cs)*",
        &expected_irreps,
        Some(expected_chars),
    );
}

/*
D2
*/
#[test]
fn test_chartab_asymmetric_i4_biphenyl_d2() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/i4-biphenyl.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    verify_d2(&mol, thresh);
}

#[test]
fn test_chartab_asymmetric_i4_biphenyl_grey_d2() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/i4-biphenyl.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    verify_grey_d2(&mol, thresh);
}

#[test]
fn test_chartab_asymmetric_twistane_d2() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/twistane.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    verify_d2(&mol, thresh);
}

#[test]
fn test_chartab_asymmetric_twistane_grey_d2() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/twistane.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    verify_grey_d2(&mol, thresh);
}

#[test]
fn test_chartab_asymmetric_22_paracyclophane_d2() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/paracyclophane22.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    verify_d2(&mol, thresh);
}

#[test]
fn test_chartab_asymmetric_22_paracyclophane_grey_d2() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/paracyclophane22.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    verify_grey_d2(&mol, thresh);
}

/// Verifies the validity of the computed $`\mathcal{D}_{2}`$ character table of irreps.
///
/// # Arguments
///
/// * `mol` - A reference to a [`Molecule`] structure.
/// * `thresh` - A threshold for symmetry detection.
///
/// # Panics
///
/// Panics when any expected condition is not fulfilled.
fn verify_d2(mol: &Molecule, thresh: f64) {
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A||").unwrap(),
        MullikenIrrepSymbol::new("||B|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(3)|").unwrap(),
    ];
    let c2 = SymmetryClassSymbol::<SymmetryOperation>::new("1||C2||", None).unwrap();
    let c2d = SymmetryClassSymbol::<SymmetryOperation>::new("1||C2|^(')|", None).unwrap();
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
    test_chartab_ordinary_group(&mol, thresh, "D2", &expected_irreps, Some(expected_chars));
}

/// Verifies the validity of the computed $`\mathcal{D}_{2} + \theta\mathcal{D}_{2}`$ character
/// table of irreps.
///
/// # Arguments
///
/// * `mol` - A reference to a [`Molecule`] structure.
/// * `thresh` - A threshold for symmetry detection.
///
/// # Panics
///
/// Panics when any expected condition is not fulfilled.
fn verify_grey_d2(mol: &Molecule, thresh: f64) {
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A||").unwrap(),
        MullikenIrrepSymbol::new("||B|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(3)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|A||").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|B|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|B|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|B|_(3)|").unwrap(),
    ];
    let tc2 = SymmetryClassSymbol::<SymmetryOperation>::new("1||θ·C2||", None).unwrap();
    let tc2d = SymmetryClassSymbol::<SymmetryOperation>::new("1||θ·C2|^(')|", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &tc2),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[1], &tc2),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[2], &tc2),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[3], &tc2),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[0], &tc2d),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[1], &tc2d),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[2], &tc2d),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[3], &tc2d),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[4], &tc2),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[5], &tc2),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[6], &tc2),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[7], &tc2),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[4], &tc2d),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[5], &tc2d),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[6], &tc2d),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[7], &tc2d),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
    ]);
    test_chartab_magnetic_group(
        &mol,
        thresh,
        "D2 + θ·D2",
        &expected_irreps,
        Some(expected_chars),
    );
}

/***
D2*
***/
#[test]
fn test_chartab_asymmetric_i4_biphenyl_d2_double() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/i4-biphenyl.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A||").unwrap(),
        MullikenIrrepSymbol::new("||B|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(3)|").unwrap(),
        MullikenIrrepSymbol::new("||E~||").unwrap(),
    ];
    let c2 = SymmetryClassSymbol::<SymmetryOperation>::new("1||C2(Σ), C2(QΣ)||", None).unwrap();
    let c2d =
        SymmetryClassSymbol::<SymmetryOperation>::new("1||C2(Σ), C2(QΣ)|^(')|", None).unwrap();
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
            Character::new(&[(UnityRoot::new(0, 2), 1), (UnityRoot::new(1, 2), 1)]),
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
            Character::new(&[(UnityRoot::new(0, 2), 1), (UnityRoot::new(1, 2), 1)]),
        ),
    ]);
    test_chartab_ordinary_double_group(&mol, thresh, "D2*", &expected_irreps, Some(expected_chars));
}

#[test]
fn test_chartab_asymmetric_i4_biphenyl_grey_d2_double() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/i4-biphenyl.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A||").unwrap(),
        MullikenIrrepSymbol::new("||B|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("||B|_(3)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|A||").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|B|_(1)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|B|_(2)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|B|_(3)|").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|E~||").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|E~||").unwrap(),
    ];
    let t = SymmetryClassSymbol::<SymmetryOperation>::new("1||θ(Σ)||", None).unwrap();
    let tc2 =
        SymmetryClassSymbol::<SymmetryOperation>::new("1||θ·C2(Σ), θ·C2(QΣ)||", None).unwrap();
    let tc2d =
        SymmetryClassSymbol::<SymmetryOperation>::new("1||θ·C2(Σ), θ·C2(QΣ)|^(')|", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &t),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[1], &t),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[2], &t),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[3], &t),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[0], &tc2),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[1], &tc2),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[2], &tc2),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[3], &tc2),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[0], &tc2d),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[1], &tc2d),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[2], &tc2d),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[3], &tc2d),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[4], &t),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[5], &t),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[6], &t),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[7], &t),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[4], &tc2),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[5], &tc2),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[6], &tc2),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[7], &tc2),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[4], &tc2d),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[5], &tc2d),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[6], &tc2d),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[7], &tc2d),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[8], &t),
            Character::new(&[(UnityRoot::new(1, 4), 2)]),
        ),
        (
            (&expected_irreps[9], &t),
            Character::new(&[(UnityRoot::new(3, 4), 2)]),
        ),
        (
            (&expected_irreps[8], &tc2),
            Character::new(&[(UnityRoot::new(0, 2), 1), (UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[9], &tc2),
            Character::new(&[(UnityRoot::new(0, 2), 1), (UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[8], &tc2d),
            Character::new(&[(UnityRoot::new(0, 2), 1), (UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[9], &tc2d),
            Character::new(&[(UnityRoot::new(0, 2), 1), (UnityRoot::new(1, 2), 1)]),
        ),
    ]);
    test_chartab_magnetic_double_group(
        &mol,
        thresh,
        "(D2 + θ·D2)*",
        &expected_irreps,
        Some(expected_chars),
    );
}

/***
D2h
***/
#[test]
fn test_chartab_asymmetric_b2h6_d2h() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/b2h6.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    verify_d2h(&mol, thresh, false);
}

#[test]
fn test_chartab_asymmetric_b2h6_grey_d2h() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/b2h6.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    verify_grey_d2h(&mol, thresh);
}

#[test]
fn test_chartab_asymmetric_naphthalene_d2h() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/naphthalene.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    verify_d2h(&mol, thresh, false);
}

#[test]
fn test_chartab_asymmetric_naphthalene_grey_d2h() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/naphthalene.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    verify_grey_d2h(&mol, thresh);
}

#[test]
fn test_chartab_asymmetric_pyrene_d2h() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/pyrene.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    verify_d2h(&mol, thresh, false);
}

#[test]
fn test_chartab_asymmetric_pyrene_grey_d2h() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/pyrene.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    verify_grey_d2h(&mol, thresh);
}

#[test]
fn test_chartab_asymmetric_c6o6_d2h() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c6o6.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    verify_d2h(&mol, thresh, false);
}

#[test]
fn test_chartab_asymmetric_c6o6_grey_d2h() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c6o6.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    verify_grey_d2h(&mol, thresh);
}

#[test]
fn test_chartab_asymmetric_distorted_vf6_d2h() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/vf6_d2h.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    verify_d2h(&mol, thresh, false);
}

#[test]
fn test_chartab_asymmetric_distorted_vf6_grey_d2h() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/vf6_d2h.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    verify_grey_d2h(&mol, thresh);
}

/// Verifies the validity of the computed $`\mathcal{D}_{2h}`$ character table of irreps.
///
/// # Arguments
///
/// * `mol` - A reference to a [`Molecule`] structure.
/// * `thresh` - A threshold for symmetry detection.
/// * `magnetic` - A flag indicating if this is a magnetic group.
///
/// # Panics
///
/// Panics when any expected condition is not fulfilled.
fn verify_d2h(mol: &Molecule, thresh: f64, magnetic: bool) {
    if magnetic {
        let expected_irreps = vec![
            MullikenIrrepSymbol::new("||A|_(1g)|").unwrap(),
            MullikenIrrepSymbol::new("||A|_(2g)|").unwrap(),
            MullikenIrrepSymbol::new("||B|_(1g)|").unwrap(),
            MullikenIrrepSymbol::new("||B|_(2g)|").unwrap(),
            MullikenIrrepSymbol::new("||A|_(1u)|").unwrap(),
            MullikenIrrepSymbol::new("||A|_(2u)|").unwrap(),
            MullikenIrrepSymbol::new("||B|_(1u)|").unwrap(),
            MullikenIrrepSymbol::new("||B|_(2u)|").unwrap(),
        ];
        let c2 = SymmetryClassSymbol::<SymmetryOperation>::new("1||C2||", None).unwrap();
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
        ]);
        test_chartab_magnetic_group(&mol, thresh, "D2h", &expected_irreps, Some(expected_chars));
    } else {
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
        let c2 = SymmetryClassSymbol::<SymmetryOperation>::new("1||C2||", None).unwrap();
        let c2d = SymmetryClassSymbol::<SymmetryOperation>::new("1||C2|^(')|", None).unwrap();
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
        test_chartab_ordinary_group(&mol, thresh, "D2h", &expected_irreps, Some(expected_chars));
    }
}

/// Verifies the validity of the computed $`\mathcal{D}_{2h} + \theta\mathcal{D}_{2h}`$ character
/// table of irreps.
///
/// # Arguments
///
/// * `mol` - A reference to a [`Molecule`] structure.
/// * `thresh` - A threshold for symmetry detection.
///
/// # Panics
///
/// Panics when any expected condition is not fulfilled.
fn verify_grey_d2h(mol: &Molecule, thresh: f64) {
    let expected_irreps = vec![
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
    let tc2 = SymmetryClassSymbol::<SymmetryOperation>::new("1||θ·C2||", None).unwrap();
    let tc2d = SymmetryClassSymbol::<SymmetryOperation>::new("1||θ·C2|^(')|", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &tc2),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[1], &tc2),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[2], &tc2),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[3], &tc2),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[4], &tc2),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[5], &tc2),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[6], &tc2),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[7], &tc2),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[0], &tc2d),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[1], &tc2d),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[2], &tc2d),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[3], &tc2d),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[4], &tc2d),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[5], &tc2d),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[6], &tc2d),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[7], &tc2d),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[8], &tc2),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[9], &tc2),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[10], &tc2),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[11], &tc2),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[12], &tc2),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[13], &tc2),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[14], &tc2),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[15], &tc2),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[8], &tc2d),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[9], &tc2d),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[10], &tc2d),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[11], &tc2d),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[12], &tc2d),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[13], &tc2d),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[14], &tc2d),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[15], &tc2d),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
    ]);
    test_chartab_magnetic_group(
        &mol,
        thresh,
        "D2h + θ·D2h",
        &expected_irreps,
        Some(expected_chars),
    );
}

/***
D2h*
***/
#[test]
fn test_chartab_asymmetric_b2h6_d2h_double() {
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
        MullikenIrrepSymbol::new("||E~|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("||E~|_(u)|").unwrap(),
    ];
    let c2 = SymmetryClassSymbol::<SymmetryOperation>::new("1||C2(Σ), C2(QΣ)||", None).unwrap();
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
            (&expected_irreps[8], &c2),
            Character::new(&[(UnityRoot::new(0, 2), 1), (UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[9], &c2),
            Character::new(&[(UnityRoot::new(0, 2), 1), (UnityRoot::new(1, 2), 1)]),
        ),
    ]);
    test_chartab_ordinary_double_group(
        &mol,
        thresh,
        "D2h*",
        &expected_irreps,
        Some(expected_chars),
    );
}

#[test]
fn test_chartab_asymmetric_b2h6_grey_d2h_double() {
    // env_logger::init();
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
        MullikenIrrepSymbol::new("|^(m)|A|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|B|_(1g)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|B|_(2g)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|B|_(3g)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|A|_(u)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|B|_(1u)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|B|_(2u)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|B|_(3u)|").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|E~|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|E~|_(u)|").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|E~|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|E~|_(u)|").unwrap(),
    ];
    let t = SymmetryClassSymbol::<SymmetryOperation>::new("1||θ(Σ)||", None).unwrap();
    let tc2 =
        SymmetryClassSymbol::<SymmetryOperation>::new("1||θ·C2(Σ), θ·C2(QΣ)||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &t),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[1], &t),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[2], &t),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[3], &t),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[4], &t),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[5], &t),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[6], &t),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[7], &t),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[8], &t),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[9], &t),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[10], &t),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[11], &t),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[12], &t),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[13], &t),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[14], &t),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[15], &t),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[16], &t),
            Character::new(&[(UnityRoot::new(1, 4), 2)]),
        ),
        (
            (&expected_irreps[17], &t),
            Character::new(&[(UnityRoot::new(1, 4), 2)]),
        ),
        (
            (&expected_irreps[18], &t),
            Character::new(&[(UnityRoot::new(3, 4), 2)]),
        ),
        (
            (&expected_irreps[19], &t),
            Character::new(&[(UnityRoot::new(3, 4), 2)]),
        ),
        (
            (&expected_irreps[0], &tc2),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[1], &tc2),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[2], &tc2),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[3], &tc2),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[4], &tc2),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[5], &tc2),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[6], &tc2),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[7], &tc2),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[8], &tc2),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[9], &tc2),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[10], &tc2),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[11], &tc2),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[12], &tc2),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[13], &tc2),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[14], &tc2),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[15], &tc2),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[16], &tc2),
            Character::new(&[(UnityRoot::new(0, 2), 1), (UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[17], &tc2),
            Character::new(&[(UnityRoot::new(0, 2), 1), (UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[18], &tc2),
            Character::new(&[(UnityRoot::new(0, 2), 1), (UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[19], &tc2),
            Character::new(&[(UnityRoot::new(0, 2), 1), (UnityRoot::new(1, 2), 1)]),
        ),
    ]);
    test_chartab_magnetic_double_group(
        &mol,
        thresh,
        "(D2h + θ·D2h)*",
        &expected_irreps,
        Some(expected_chars),
    );
}

/***
Ci
***/
#[test]
fn test_chartab_asymmetric_meso_tartaricacid_ci() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/meso-tartaricacid.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    verify_ci(&mol, thresh);
}

#[test]
fn test_chartab_asymmetric_meso_tartaricacid_grey_ci() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/meso-tartaricacid.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    verify_grey_ci(&mol, thresh);
}

#[test]
fn test_chartab_asymmetric_dibromodimethylcyclohexane_ci() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/dibromodimethylcyclohexane.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    verify_ci(&mol, thresh);
}

#[test]
fn test_chartab_asymmetric_dibromodimethylcyclohexane_grey_ci() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/dibromodimethylcyclohexane.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    verify_grey_ci(&mol, thresh);
}

#[test]
fn test_chartab_asymmetric_h2o2_magnetic_field_ci() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/h2o2.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(1.0, 2.0, -1.0)));
    verify_ci(&mol, thresh);
}

#[test]
fn test_chartab_asymmetric_h2o2_magnetic_field_bw_c2h_ci() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/h2o2_yz.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 2.0, -1.0)));
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(1g)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(2g)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(1u)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(2u)|").unwrap(),
    ];
    let tc2 = SymmetryClassSymbol::<SymmetryOperation>::new("1||θ·C2||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &tc2),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[1], &tc2),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[2], &tc2),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[3], &tc2),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
    ]);
    test_chartab_magnetic_group(&mol, thresh, "C2h", &expected_irreps, Some(expected_chars));
}

#[test]
fn test_chartab_symmetric_xef4_magnetic_field_ci() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/xef4.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(1.0, 2.0, -2.0)));
    verify_ci(&mol, thresh);
}

#[test]
fn test_chartab_asymmetric_c2h2_magnetic_field_ci() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c2h2.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(1.0, -1.0, 1.0)));
    verify_ci(&mol, thresh);
}

/// Verifies the validity of the computed $`\mathcal{C}_{i}`$ character table of irreps.
///
/// # Arguments
///
/// * `mol` - A reference to a [`Molecule`] structure.
/// * `thresh` - A threshold for symmetry detection.
///
/// # Panics
///
/// Panics when any expected condition is not fulfilled.
fn verify_ci(mol: &Molecule, thresh: f64) {
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(u)|").unwrap(),
    ];
    let i = SymmetryClassSymbol::<SymmetryOperation>::new("1||i||", None).unwrap();
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
    test_chartab_ordinary_group(&mol, thresh, "Ci", &expected_irreps, Some(expected_chars));
}

/// Verifies the validity of the computed $`\mathcal{C}_{i} + \theta\mathcal{C}_{i}`$ character
/// table of irreps.
///
/// # Arguments
///
/// * `mol` - A reference to a [`Molecule`] structure.
/// * `thresh` - A threshold for symmetry detection.
///
/// # Panics
///
/// Panics when any expected condition is not fulfilled.
fn verify_grey_ci(mol: &Molecule, thresh: f64) {
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(u)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|A|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|A|_(u)|").unwrap(),
    ];
    let ti = SymmetryClassSymbol::<SymmetryOperation>::new("1||θ·i||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &ti),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[1], &ti),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[2], &ti),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[3], &ti),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
    ]);
    test_chartab_magnetic_group(
        &mol,
        thresh,
        "Ci + θ·Ci",
        &expected_irreps,
        Some(expected_chars),
    );
}

/***
Ci*
***/
#[test]
fn test_chartab_asymmetric_meso_tartaricacid_ci_double() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/meso-tartaricacid.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(u)|").unwrap(),
        MullikenIrrepSymbol::new("||A~|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("||A~|_(u)|").unwrap(),
    ];
    let i = SymmetryClassSymbol::<SymmetryOperation>::new("1||i(Σ)||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &i),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[1], &i),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[2], &i),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[3], &i),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
    ]);
    test_chartab_ordinary_double_group(&mol, thresh, "Ci*", &expected_irreps, Some(expected_chars));
}

#[test]
fn test_chartab_asymmetric_meso_tartaricacid_grey_ci_double() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/meso-tartaricacid.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(u)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|A|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|A|_(u)|").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|Γ~|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|Γ~|_(u)|").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|Γ~|_(g)|").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|Γ~|_(u)|").unwrap(),
    ];
    let ti = SymmetryClassSymbol::<SymmetryOperation>::new("1||θ·i(Σ)||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &ti),
            Character::new(&[(UnityRoot::new(0, 4), 1)]),
        ),
        (
            (&expected_irreps[1], &ti),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[2], &ti),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[3], &ti),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[4], &ti),
            Character::new(&[(UnityRoot::new(1, 4), 1)]),
        ),
        (
            (&expected_irreps[5], &ti),
            Character::new(&[(UnityRoot::new(3, 4), 1)]),
        ),
        (
            (&expected_irreps[6], &ti),
            Character::new(&[(UnityRoot::new(3, 4), 1)]),
        ),
        (
            (&expected_irreps[7], &ti),
            Character::new(&[(UnityRoot::new(1, 4), 1)]),
        ),
    ]);
    test_chartab_magnetic_double_group(
        &mol,
        thresh,
        "(Ci + θ·Ci)*",
        &expected_irreps,
        Some(expected_chars),
    );
}

#[test]
fn test_chartab_asymmetric_h2o2_magnetic_field_bw_c2h_ci_double() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/h2o2_yz.xyz");
    let thresh = 1e-6;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 2.0, -1.0)));
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A|_(1g)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(2g)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(1u)|").unwrap(),
        MullikenIrrepSymbol::new("||A|_(2u)|").unwrap(),
        MullikenIrrepSymbol::new("||A~|_(1g)|").unwrap(),
        MullikenIrrepSymbol::new("||A~|_(2g)|").unwrap(),
        MullikenIrrepSymbol::new("||A~|_(1u)|").unwrap(),
        MullikenIrrepSymbol::new("||A~|_(2u)|").unwrap(),
    ];
    let tc2 = SymmetryClassSymbol::<SymmetryOperation>::new("1||θ·C2(Σ)||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &tc2),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[1], &tc2),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[2], &tc2),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[3], &tc2),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[4], &tc2),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[5], &tc2),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[6], &tc2),
            Character::new(&[(UnityRoot::new(0, 2), 1)]),
        ),
        (
            (&expected_irreps[7], &tc2),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
    ]);
    test_chartab_magnetic_double_group(
        &mol,
        thresh,
        "C2h*",
        &expected_irreps,
        Some(expected_chars),
    );
}

/***
C1
***/
#[test]
fn test_chartab_asymmetric_butan1ol_c1() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/butan-1-ol.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    verify_c1(&mol, thresh);
}

#[test]
fn test_chartab_asymmetric_butan1ol_grey_c1() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/butan-1-ol.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    verify_grey_c1(&mol, thresh);
}

#[test]
fn test_chartab_asymmetric_subst_5m_ring_c1() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/subst-5m-ring.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    verify_c1(&mol, thresh);
}

#[test]
fn test_chartab_asymmetric_subst_5m_ring_grey_c1() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/subst-5m-ring.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    verify_grey_c1(&mol, thresh);
}

#[test]
fn test_chartab_asymmetric_bf3_magnetic_field_c1() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/bf3.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(1.0, -1.0, 1.0)));
    verify_c1(&mol, thresh);
}

/// Verifies the validity of the computed $`\mathcal{C}_{1}`$ character table of irreps.
///
/// # Arguments
///
/// * `mol` - A reference to a [`Molecule`] structure.
/// * `thresh` - A threshold for symmetry detection.
///
/// # Panics
///
/// Panics when any expected condition is not fulfilled.
fn verify_c1(mol: &Molecule, thresh: f64) {
    let expected_irreps = vec![MullikenIrrepSymbol::new("||A||").unwrap()];
    let e = SymmetryClassSymbol::<SymmetryOperation>::new("1||E||", None).unwrap();
    let expected_chars = HashMap::from([(
        (&expected_irreps[0], &e),
        Character::new(&[(UnityRoot::new(0, 1), 1)]),
    )]);
    test_chartab_ordinary_group(&mol, thresh, "C1", &expected_irreps, Some(expected_chars));
}

/// Verifies the validity of the computed $`\mathcal{C}_{1} + \theta\mathcal{C}_{1}`$ character
/// table of irreps.
///
/// # Arguments
///
/// * `mol` - A reference to a [`Molecule`] structure.
/// * `thresh` - A threshold for symmetry detection.
///
/// # Panics
///
/// Panics when any expected condition is not fulfilled.
fn verify_grey_c1(mol: &Molecule, thresh: f64) {
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A||").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|A||").unwrap(),
    ];
    let t = SymmetryClassSymbol::<SymmetryOperation>::new("1||θ||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &t),
            Character::new(&[(UnityRoot::new(0, 1), 1)]),
        ),
        (
            (&expected_irreps[1], &t),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
    ]);
    test_chartab_magnetic_group(
        &mol,
        thresh,
        "C1 + θ·C1",
        &expected_irreps,
        Some(expected_chars),
    );
}

/***
C1*
***/
#[test]
fn test_chartab_asymmetric_butan1ol_c1_double() {
    // env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/butan-1-ol.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A||").unwrap(),
        MullikenIrrepSymbol::new("||A~||").unwrap(),
    ];
    let e = SymmetryClassSymbol::<SymmetryOperation>::new("1||E(Σ)||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &e),
            Character::new(&[(UnityRoot::new(0, 1), 1)]),
        ),
        (
            (&expected_irreps[1], &e),
            Character::new(&[(UnityRoot::new(0, 1), 1)]),
        ),
    ]);
    test_chartab_ordinary_double_group(&mol, thresh, "C1*", &expected_irreps, Some(expected_chars));
}

#[test]
fn test_chartab_asymmetric_butan1ol_grey_c1_double() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/butan-1-ol.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    let expected_irreps = vec![
        MullikenIrrepSymbol::new("||A||").unwrap(),
        MullikenIrrepSymbol::new("|^(m)|A||").unwrap(),
        MullikenIrrepSymbol::new("|_(a)|Γ~||").unwrap(),
        MullikenIrrepSymbol::new("|_(b)|Γ~||").unwrap(),
    ];
    let t = SymmetryClassSymbol::<SymmetryOperation>::new("1||θ(Σ)||", None).unwrap();
    let expected_chars = HashMap::from([
        (
            (&expected_irreps[0], &t),
            Character::new(&[(UnityRoot::new(0, 1), 1)]),
        ),
        (
            (&expected_irreps[1], &t),
            Character::new(&[(UnityRoot::new(1, 2), 1)]),
        ),
        (
            (&expected_irreps[2], &t),
            Character::new(&[(UnityRoot::new(1, 4), 1)]),
        ),
        (
            (&expected_irreps[3], &t),
            Character::new(&[(UnityRoot::new(3, 4), 1)]),
        ),
    ]);
    test_chartab_magnetic_double_group(
        &mol,
        thresh,
        "(C1 + θ·C1)*",
        &expected_irreps,
        Some(expected_chars),
    );
}

#[cfg(test)]
#[path = "symmetry_chartab_nonuniform_b_tests.rs"]
mod symmetry_chartab_nonuniform_b_tests;
