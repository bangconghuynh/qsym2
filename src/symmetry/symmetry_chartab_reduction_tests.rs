use itertools::Itertools;
use nalgebra::{Rotation3, Vector3};
use num_traits::ToPrimitive;

use crate::aux::atom::{Atom, AtomKind};
use crate::aux::molecule::Molecule;
use crate::chartab::chartab_group::CharacterProperties;
use crate::chartab::chartab_symbols::{DecomposedSymbol, ReducibleLinearSpaceSymbol};
use crate::chartab::{CharacterTable, SubspaceDecomposable};
use crate::group::{MagneticRepresentedGroup, UnitaryRepresentedGroup};
use crate::symmetry::symmetry_core::{PreSymmetry, Symmetry};
use crate::symmetry::symmetry_group::SymmetryGroupProperties;
use crate::symmetry::symmetry_symbols::MullikenIrrepSymbol;

const ROOT: &str = env!("CARGO_MANIFEST_DIR");

#[test]
fn test_chartab_reduction_vf6_oh() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/vf6.xyz");
    let thresh = 1e-12;
    let mol = Molecule::from_xyz(&path, thresh);
    let presym = PreSymmetry::builder()
        .moi_threshold(thresh)
        .molecule(&mol)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false).unwrap();
    let group = UnitaryRepresentedGroup::from_molecular_symmetry(&sym, None);
    let chartab = group.character_table();

    // Single irreps
    chartab.get_all_rows().iter().for_each(|test_irrep| {
        let reduction = chartab
            .reduce_characters(
                &chartab
                    .get_all_cols()
                    .iter()
                    .map(|cc_symbol| {
                        (
                            cc_symbol,
                            chartab
                                .get_character(&test_irrep, cc_symbol)
                                .complex_value(),
                        )
                    })
                    .collect::<Vec<_>>(),
                thresh,
            )
            .unwrap();
        assert_eq!(reduction.subspaces().len(), 1);
        assert!(reduction.subspaces().contains(&(test_irrep, &1)));
    });

    // Direct products
    let a1g = MullikenIrrepSymbol::new("||A|_(1g)|").unwrap();
    let a2g = MullikenIrrepSymbol::new("||A|_(2g)|").unwrap();
    let eg = MullikenIrrepSymbol::new("||E|_(g)|").unwrap();
    let t1g = MullikenIrrepSymbol::new("||T|_(1g)|").unwrap();
    let t2g = MullikenIrrepSymbol::new("||T|_(2g)|").unwrap();
    let a1u = MullikenIrrepSymbol::new("||A|_(1u)|").unwrap();
    let a2u = MullikenIrrepSymbol::new("||A|_(2u)|").unwrap();
    let eu = MullikenIrrepSymbol::new("||E|_(u)|").unwrap();
    let t1u = MullikenIrrepSymbol::new("||T|_(1u)|").unwrap();
    let t2u = MullikenIrrepSymbol::new("||T|_(2u)|").unwrap();
    let expected_direct_products = vec![
        ((&a1g, &a1g), vec![(a1g.clone(), 1)]),
        ((&a1g, &a2g), vec![(a2g.clone(), 1)]),
        ((&a1g, &eg), vec![(eg.clone(), 1)]),
        ((&a1g, &t1g), vec![(t1g.clone(), 1)]),
        ((&a1g, &t2g), vec![(t2g.clone(), 1)]),
        ((&a2g, &a2g), vec![(a1g.clone(), 1)]),
        ((&a2g, &eg), vec![(eg.clone(), 1)]),
        ((&a2g, &t1g), vec![(t2g.clone(), 1)]),
        ((&a2g, &t2g), vec![(t1g.clone(), 1)]),
        (
            (&eg, &eg),
            vec![(a1g.clone(), 1), (a2g.clone(), 1), (eg.clone(), 1)],
        ),
        ((&eg, &t1g), vec![(t1g.clone(), 1), (t2g.clone(), 1)]),
        ((&eg, &t2g), vec![(t1g.clone(), 1), (t2g.clone(), 1)]),
        (
            (&t1g, &t1g),
            vec![
                (a1g.clone(), 1),
                (eg.clone(), 1),
                (t1g.clone(), 1),
                (t2g.clone(), 1),
            ],
        ),
        (
            (&t1g, &t2g),
            vec![
                (a2g.clone(), 1),
                (eg.clone(), 1),
                (t1g.clone(), 1),
                (t2g.clone(), 1),
            ],
        ),
        (
            (&t2g, &t2g),
            vec![
                (a1g.clone(), 1),
                (eg.clone(), 1),
                (t1g.clone(), 1),
                (t2g.clone(), 1),
            ],
        ),
        ((&a1g, &a1u), vec![(a1u.clone(), 1)]),
        ((&a1g, &a2u), vec![(a2u.clone(), 1)]),
        ((&a1g, &eu), vec![(eu.clone(), 1)]),
        ((&a1g, &t1u), vec![(t1u.clone(), 1)]),
        ((&a1g, &t2u), vec![(t2u.clone(), 1)]),
        ((&a2g, &a1u), vec![(a2u.clone(), 1)]),
        ((&a2g, &a2u), vec![(a1u.clone(), 1)]),
        ((&a2g, &eu), vec![(eu.clone(), 1)]),
        ((&a2g, &t1u), vec![(t2u.clone(), 1)]),
        ((&a2g, &t2u), vec![(t1u.clone(), 1)]),
        ((&eg, &a1u), vec![(eu.clone(), 1)]),
        ((&eg, &a2u), vec![(eu.clone(), 1)]),
        (
            (&eg, &eu),
            vec![(a1u.clone(), 1), (a2u.clone(), 1), (eu.clone(), 1)],
        ),
        ((&eg, &t1u), vec![(t1u.clone(), 1), (t2u.clone(), 1)]),
        ((&eg, &t2u), vec![(t1u.clone(), 1), (t2u.clone(), 1)]),
        ((&t1g, &a1u), vec![(t1u.clone(), 1)]),
        ((&t1g, &a2u), vec![(t2u.clone(), 1)]),
        ((&t1g, &eu), vec![(t1u.clone(), 1), (t2u.clone(), 1)]),
        (
            (&t1g, &t1u),
            vec![
                (a1u.clone(), 1),
                (eu.clone(), 1),
                (t1u.clone(), 1),
                (t2u.clone(), 1),
            ],
        ),
        (
            (&t1g, &t2u),
            vec![
                (a2u.clone(), 1),
                (eu.clone(), 1),
                (t1u.clone(), 1),
                (t2u.clone(), 1),
            ],
        ),
        ((&t2g, &a1u), vec![(t2u.clone(), 1)]),
        ((&t2g, &a2u), vec![(t1u.clone(), 1)]),
        ((&t2g, &eu), vec![(t1u.clone(), 1), (t2u.clone(), 1)]),
        (
            (&t2g, &t1u),
            vec![
                (a2u.clone(), 1),
                (eu.clone(), 1),
                (t1u.clone(), 1),
                (t2u.clone(), 1),
            ],
        ),
        (
            (&t2g, &t2u),
            vec![
                (a1u.clone(), 1),
                (eu.clone(), 1),
                (t1u.clone(), 1),
                (t2u.clone(), 1),
            ],
        ),
        ((&a1u, &a1u), vec![(a1g.clone(), 1)]),
        ((&a1u, &a2u), vec![(a2g.clone(), 1)]),
        ((&a1u, &eu), vec![(eg.clone(), 1)]),
        ((&a1u, &t1u), vec![(t1g.clone(), 1)]),
        ((&a1u, &t2u), vec![(t2g.clone(), 1)]),
        ((&a2u, &a2u), vec![(a1g.clone(), 1)]),
        ((&a2u, &eu), vec![(eg.clone(), 1)]),
        ((&a2u, &t1u), vec![(t2g.clone(), 1)]),
        ((&a2u, &t2u), vec![(t1g.clone(), 1)]),
        (
            (&eu, &eu),
            vec![(a1g.clone(), 1), (a2g.clone(), 1), (eg.clone(), 1)],
        ),
        ((&eu, &t1u), vec![(t1g.clone(), 1), (t2g.clone(), 1)]),
        ((&eu, &t2u), vec![(t1g.clone(), 1), (t2g.clone(), 1)]),
        (
            (&t1u, &t1u),
            vec![
                (a1g.clone(), 1),
                (eg.clone(), 1),
                (t1g.clone(), 1),
                (t2g.clone(), 1),
            ],
        ),
        (
            (&t1u, &t2u),
            vec![
                (a2g.clone(), 1),
                (eg.clone(), 1),
                (t1g.clone(), 1),
                (t2g.clone(), 1),
            ],
        ),
        (
            (&t2u, &t2u),
            vec![
                (a1g.clone(), 1),
                (eg.clone(), 1),
                (t1g.clone(), 1),
                (t2g.clone(), 1),
            ],
        ),
    ];

    expected_direct_products
        .iter()
        .for_each(|((irrep_0, irrep_1), res)| {
            let irrep_0_chars = chartab.get_row(irrep_0).map(|x| x.complex_value());
            let irrep_1_chars = chartab.get_row(irrep_1).map(|x| x.complex_value());
            let direct_product = irrep_0_chars * irrep_1_chars;

            let reduction = chartab
                .reduce_characters(
                    &chartab
                        .get_all_cols()
                        .iter()
                        .enumerate()
                        .map(|(i, cc_symbol)| (cc_symbol, direct_product[i]))
                        .collect::<Vec<_>>(),
                    thresh,
                )
                .unwrap();
            assert_eq!(
                reduction,
                DecomposedSymbol::<MullikenIrrepSymbol>::from_subspaces(res)
            );
        });
}

#[test]
fn test_chartab_reduction_vf6_oh_double() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/vf6.xyz");
    let thresh = 1e-12;
    let mol = Molecule::from_xyz(&path, thresh);
    let presym = PreSymmetry::builder()
        .moi_threshold(thresh)
        .molecule(&mol)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false).unwrap();
    let group = UnitaryRepresentedGroup::from_molecular_symmetry(&sym, None).to_double_group();
    let chartab = group.character_table();

    // Direct products
    let a1g = MullikenIrrepSymbol::new("||A|_(1g)|").unwrap();
    let a2g = MullikenIrrepSymbol::new("||A|_(2g)|").unwrap();
    let eg = MullikenIrrepSymbol::new("||E|_(g)|").unwrap();
    let t1g = MullikenIrrepSymbol::new("||T|_(1g)|").unwrap();
    let t2g = MullikenIrrepSymbol::new("||T|_(2g)|").unwrap();
    let a1u = MullikenIrrepSymbol::new("||A|_(1u)|").unwrap();
    let a2u = MullikenIrrepSymbol::new("||A|_(2u)|").unwrap();
    let eu = MullikenIrrepSymbol::new("||E|_(u)|").unwrap();
    let t1u = MullikenIrrepSymbol::new("||T|_(1u)|").unwrap();
    let t2u = MullikenIrrepSymbol::new("||T|_(2u)|").unwrap();
    let ep1g = MullikenIrrepSymbol::new("||E~|_(1g)|").unwrap();
    let ep2g = MullikenIrrepSymbol::new("||E~|_(2g)|").unwrap();
    let gpg = MullikenIrrepSymbol::new("||G~|_(g)|").unwrap();
    let ep1u = MullikenIrrepSymbol::new("||E~|_(1u)|").unwrap();
    let ep2u = MullikenIrrepSymbol::new("||E~|_(2u)|").unwrap();
    let gpu = MullikenIrrepSymbol::new("||G~|_(u)|").unwrap();
    let expected_direct_products = vec![
        ((&a1g, &ep1g), vec![(ep1g.clone(), 1)]),
        ((&a1g, &ep2g), vec![(ep2g.clone(), 1)]),
        ((&a1g, &gpg), vec![(gpg.clone(), 1)]),
        ((&a2g, &ep1g), vec![(ep2g.clone(), 1)]),
        ((&a2g, &ep2g), vec![(ep1g.clone(), 1)]),
        ((&a2g, &gpg), vec![(gpg.clone(), 1)]),
        ((&eg, &ep1g), vec![(gpg.clone(), 1)]),
        ((&eg, &ep2g), vec![(gpg.clone(), 1)]),
        (
            (&eg, &gpg),
            vec![(ep1g.clone(), 1), (ep2g.clone(), 1), (gpg.clone(), 1)],
        ),
        ((&t1g, &ep1g), vec![(ep1g.clone(), 1), (gpg.clone(), 1)]),
        ((&t1g, &ep2g), vec![(ep2g.clone(), 1), (gpg.clone(), 1)]),
        (
            (&t1g, &gpg),
            vec![(ep1g.clone(), 1), (ep2g.clone(), 1), (gpg.clone(), 2)],
        ),
        ((&t2g, &ep1g), vec![(ep2g.clone(), 1), (gpg.clone(), 1)]),
        ((&t2g, &ep2g), vec![(ep1g.clone(), 1), (gpg.clone(), 1)]),
        (
            (&t2g, &gpg),
            vec![(ep1g.clone(), 1), (ep2g.clone(), 1), (gpg.clone(), 2)],
        ),
        ((&a1u, &ep1g), vec![(ep1u.clone(), 1)]),
        ((&a1u, &ep2g), vec![(ep2u.clone(), 1)]),
        ((&a1u, &gpg), vec![(gpu.clone(), 1)]),
        ((&a2u, &ep1g), vec![(ep2u.clone(), 1)]),
        ((&a2u, &ep2g), vec![(ep1u.clone(), 1)]),
        ((&a2u, &gpg), vec![(gpu.clone(), 1)]),
        ((&eu, &ep1g), vec![(gpu.clone(), 1)]),
        ((&eu, &ep2g), vec![(gpu.clone(), 1)]),
        (
            (&eu, &gpg),
            vec![(ep1u.clone(), 1), (ep2u.clone(), 1), (gpu.clone(), 1)],
        ),
        ((&t1u, &ep1g), vec![(ep1u.clone(), 1), (gpu.clone(), 1)]),
        ((&t1u, &ep2g), vec![(ep2u.clone(), 1), (gpu.clone(), 1)]),
        (
            (&t1u, &gpg),
            vec![(ep1u.clone(), 1), (ep2u.clone(), 1), (gpu.clone(), 2)],
        ),
        ((&t2u, &ep1g), vec![(ep2u.clone(), 1), (gpu.clone(), 1)]),
        ((&t2u, &ep2g), vec![(ep1u.clone(), 1), (gpu.clone(), 1)]),
        (
            (&t2u, &gpg),
            vec![(ep1u.clone(), 1), (ep2u.clone(), 1), (gpu.clone(), 2)],
        ),
        ((&ep1g, &ep1g), vec![(a1g.clone(), 1), (t1g.clone(), 1)]),
        ((&ep1g, &ep2g), vec![(a2g.clone(), 1), (t2g.clone(), 1)]),
        (
            (&ep1g, &gpg),
            vec![(eg.clone(), 1), (t1g.clone(), 1), (t2g.clone(), 1)],
        ),
        ((&ep2g, &ep2g), vec![(a1g.clone(), 1), (t1g.clone(), 1)]),
        (
            (&ep2g, &gpg),
            vec![(eg.clone(), 1), (t1g.clone(), 1), (t2g.clone(), 1)],
        ),
        (
            (&gpg, &gpg),
            vec![
                (a1g.clone(), 1),
                (a2g.clone(), 1),
                (eg.clone(), 1),
                (t1g.clone(), 2),
                (t2g.clone(), 2),
            ],
        ),
    ];

    expected_direct_products
        .iter()
        .for_each(|((irrep_0, irrep_1), res)| {
            let irrep_0_chars = chartab.get_row(irrep_0).map(|x| x.complex_value());
            let irrep_1_chars = chartab.get_row(irrep_1).map(|x| x.complex_value());
            let direct_product = irrep_0_chars * irrep_1_chars;

            let reduction = chartab
                .reduce_characters(
                    &chartab
                        .get_all_cols()
                        .iter()
                        .enumerate()
                        .map(|(i, cc_symbol)| (cc_symbol, direct_product[i]))
                        .collect::<Vec<_>>(),
                    thresh,
                )
                .unwrap();
            assert_eq!(
                reduction,
                DecomposedSymbol::<MullikenIrrepSymbol>::from_subspaces(res),
            );
        });
}

#[test]
fn test_chartab_reduction_b2cl4_magnetic_field_s4_double() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/b2cl4.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    let presym = PreSymmetry::builder()
        .moi_threshold(thresh)
        .molecule(&mol)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false).unwrap();
    let group = UnitaryRepresentedGroup::from_molecular_symmetry(&sym, None).to_double_group();
    let chartab = group.character_table();

    // Direct products
    let a = MullikenIrrepSymbol::new("||A||").unwrap();
    let ag = MullikenIrrepSymbol::new("|_(a)|Γ||").unwrap();
    let b = MullikenIrrepSymbol::new("||B||").unwrap();
    let bg = MullikenIrrepSymbol::new("|_(b)|Γ||").unwrap();
    let agp1 = MullikenIrrepSymbol::new("|_(a)|Γ~|_(1)|").unwrap();
    let agp2 = MullikenIrrepSymbol::new("|_(a)|Γ~|_(2)|").unwrap();
    let bgp2 = MullikenIrrepSymbol::new("|_(b)|Γ~|_(2)|").unwrap();
    let bgp1 = MullikenIrrepSymbol::new("|_(b)|Γ~|_(1)|").unwrap();

    let expected_direct_products = vec![
        ((&a, &a), vec![(a.clone(), 1)]),
        ((&a, &b), vec![(b.clone(), 1)]),
        ((&a, &ag), vec![(ag.clone(), 1)]),
        ((&a, &bg), vec![(bg.clone(), 1)]),
        ((&a, &agp1), vec![(agp1.clone(), 1)]),
        ((&a, &bgp1), vec![(bgp1.clone(), 1)]),
        ((&a, &agp2), vec![(agp2.clone(), 1)]),
        ((&a, &bgp2), vec![(bgp2.clone(), 1)]),
        ((&b, &b), vec![(a.clone(), 1)]),
        ((&b, &ag), vec![(bg.clone(), 1)]),
        ((&b, &bg), vec![(ag.clone(), 1)]),
        ((&b, &agp1), vec![(bgp2.clone(), 1)]),
        ((&b, &bgp1), vec![(agp2.clone(), 1)]),
        ((&b, &agp2), vec![(bgp1.clone(), 1)]),
        ((&b, &bgp2), vec![(agp1.clone(), 1)]),
        ((&ag, &ag), vec![(b.clone(), 1)]),
        ((&ag, &bg), vec![(a.clone(), 1)]),
        ((&ag, &agp1), vec![(agp2.clone(), 1)]),
        ((&ag, &bgp1), vec![(agp1.clone(), 1)]),
        ((&ag, &agp2), vec![(bgp2.clone(), 1)]),
        ((&ag, &bgp2), vec![(bgp1.clone(), 1)]),
        ((&bg, &bg), vec![(b.clone(), 1)]),
        ((&bg, &agp1), vec![(bgp1.clone(), 1)]),
        ((&bg, &bgp1), vec![(bgp2.clone(), 1)]),
        ((&bg, &agp2), vec![(agp1.clone(), 1)]),
        ((&bg, &bgp2), vec![(agp2.clone(), 1)]),
        ((&agp1, &agp1), vec![(ag.clone(), 1)]),
        ((&agp1, &bgp1), vec![(a.clone(), 1)]),
        ((&agp1, &agp2), vec![(b.clone(), 1)]),
        ((&agp1, &bgp2), vec![(bg.clone(), 1)]),
        ((&bgp1, &bgp1), vec![(bg.clone(), 1)]),
        ((&bgp1, &agp2), vec![(ag.clone(), 1)]),
        ((&bgp1, &bgp2), vec![(b.clone(), 1)]),
        ((&agp2, &agp2), vec![(bg.clone(), 1)]),
        ((&agp2, &bgp2), vec![(a.clone(), 1)]),
        ((&bgp2, &bgp2), vec![(ag.clone(), 1)]),
    ];

    expected_direct_products
        .iter()
        .for_each(|((irrep_0, irrep_1), res)| {
            let irrep_0_chars = chartab.get_row(irrep_0).map(|x| x.complex_value());
            let irrep_1_chars = chartab.get_row(irrep_1).map(|x| x.complex_value());
            let direct_product = irrep_0_chars * irrep_1_chars;

            let reduction = chartab
                .reduce_characters(
                    &chartab
                        .get_all_cols()
                        .iter()
                        .enumerate()
                        .map(|(i, cc_symbol)| (cc_symbol, direct_product[i]))
                        .collect::<Vec<_>>(),
                    thresh,
                )
                .unwrap();
            assert_eq!(
                reduction,
                DecomposedSymbol::<MullikenIrrepSymbol>::from_subspaces(res),
            );
        });
}

#[test]
fn test_chartab_reduction_bf3_magnetic_field_bw_d3h_c3h() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/bf3.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    mol.set_magnetic_field(Some(Vector3::new(0.0, 0.0, 1.0)));
    let presym = PreSymmetry::builder()
        .moi_threshold(thresh)
        .molecule(&mol)
        .build()
        .unwrap();
    let mut magsym = Symmetry::new();
    magsym.analyse(&presym, true).unwrap();
    let magnetic_group = MagneticRepresentedGroup::from_molecular_symmetry(&magsym, None);
    let ircorep_chartab = magnetic_group.character_table();

    // Single ircoreps
    ircorep_chartab
        .get_all_rows()
        .iter()
        .for_each(|test_ircorep| {
            let reduction = ircorep_chartab
                .reduce_characters(
                    &ircorep_chartab
                        .get_all_cols()
                        .iter()
                        .map(|cc_symbol| {
                            (
                                cc_symbol,
                                ircorep_chartab
                                    .get_character(&test_ircorep, cc_symbol)
                                    .complex_value(),
                            )
                        })
                        .collect::<Vec<_>>(),
                    thresh,
                )
                .unwrap();
            assert_eq!(reduction.subspaces().len(), 1);
            assert!(reduction.subspaces().contains(&(test_ircorep, &1)));
        });

    // Two ircoreps
    ircorep_chartab
        .get_all_rows()
        .iter()
        .combinations_with_replacement(2)
        .for_each(|test_ircoreps| {
            let test_ircorep_i = test_ircoreps[0];
            let test_ircorep_j = test_ircoreps[1];
            let reduction = ircorep_chartab
                .reduce_characters(
                    &ircorep_chartab
                        .get_all_cols()
                        .iter()
                        .map(|cc_symbol| {
                            (
                                cc_symbol,
                                ircorep_chartab
                                    .get_character(&test_ircorep_i, cc_symbol)
                                    .complex_value()
                                    + ircorep_chartab
                                        .get_character(&test_ircorep_j, cc_symbol)
                                        .complex_value(),
                            )
                        })
                        .collect::<Vec<_>>(),
                    thresh,
                )
                .unwrap();
            if test_ircorep_i == test_ircorep_j {
                assert_eq!(reduction.subspaces().len(), 1);
                assert!(reduction.subspaces().contains(&(test_ircorep_i, &2)));
            } else {
                assert_eq!(reduction.subspaces().len(), 2);
                assert!(reduction.subspaces().contains(&(test_ircorep_i, &1)));
                assert!(reduction.subspaces().contains(&(test_ircorep_j, &1)));
            }
        });
}

#[test]
fn test_chartab_reduction_h8_alt_x_magnetic_field_bw_c4h_c2h() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/h8.xyz");
    let thresh = 1e-7;
    let mut mol = Molecule::from_xyz(&path, thresh);
    let magnetic_atoms: Vec<Atom> = mol
        .atoms
        .iter()
        .enumerate()
        .map(|(i, atom)| {
            let direction_id = i.div_euclid(2);
            [
                Atom::new_special(
                    AtomKind::Magnetic(i % 2 == 0),
                    atom.coordinates
                        + Rotation3::new(
                            Vector3::z()
                                * (std::f64::consts::FRAC_PI_2 * direction_id.to_f64().unwrap()),
                        ) * (0.1 * Vector3::x()),
                    thresh,
                )
                .expect("Unable to construct a special magnetic atom."),
                Atom::new_special(
                    AtomKind::Magnetic(i % 2 != 0),
                    atom.coordinates
                        - Rotation3::new(
                            Vector3::z()
                                * (std::f64::consts::FRAC_PI_2 * direction_id.to_f64().unwrap()),
                        ) * (0.1 * Vector3::x()),
                    thresh,
                )
                .expect("Unable to construct a special magnetic atom."),
            ]
        })
        .flatten()
        .collect();
    mol.magnetic_atoms = Some(magnetic_atoms);
    let presym = PreSymmetry::builder()
        .moi_threshold(thresh)
        .molecule(&mol)
        .build()
        .unwrap();
    let mut magsym = Symmetry::new();
    magsym.analyse(&presym, true).unwrap();
    let magnetic_group = MagneticRepresentedGroup::from_molecular_symmetry(&magsym, None);
    let ircorep_chartab = magnetic_group.character_table();

    // Single ircoreps
    ircorep_chartab
        .get_all_rows()
        .iter()
        .for_each(|test_ircorep| {
            let reduction = ircorep_chartab
                .reduce_characters(
                    &ircorep_chartab
                        .get_all_cols()
                        .iter()
                        .map(|cc_symbol| {
                            (
                                cc_symbol,
                                ircorep_chartab
                                    .get_character(&test_ircorep, cc_symbol)
                                    .complex_value(),
                            )
                        })
                        .collect::<Vec<_>>(),
                    thresh,
                )
                .unwrap();
            assert_eq!(reduction.subspaces().len(), 1);
            assert!(reduction.subspaces().contains(&(test_ircorep, &1)));
        });

    // Two ircoreps
    ircorep_chartab
        .get_all_rows()
        .iter()
        .combinations_with_replacement(2)
        .for_each(|test_ircoreps| {
            let test_ircorep_i = test_ircoreps[0];
            let test_ircorep_j = test_ircoreps[1];
            let reduction = ircorep_chartab
                .reduce_characters(
                    &ircorep_chartab
                        .get_all_cols()
                        .iter()
                        .map(|cc_symbol| {
                            (
                                cc_symbol,
                                ircorep_chartab
                                    .get_character(&test_ircorep_i, cc_symbol)
                                    .complex_value()
                                    + ircorep_chartab
                                        .get_character(&test_ircorep_j, cc_symbol)
                                        .complex_value(),
                            )
                        })
                        .collect::<Vec<_>>(),
                    thresh,
                )
                .unwrap();
            if test_ircorep_i == test_ircorep_j {
                assert_eq!(reduction.subspaces().len(), 1);
                assert!(reduction.subspaces().contains(&(test_ircorep_i, &2)));
            } else {
                assert_eq!(reduction.subspaces().len(), 2);
                assert!(reduction.subspaces().contains(&(test_ircorep_i, &1)));
                assert!(reduction.subspaces().contains(&(test_ircorep_j, &1)));
            }
        });
}
