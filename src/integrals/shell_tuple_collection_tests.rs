use env_logger;

use byteorder::LittleEndian;
use nalgebra::{Point3, Vector3};
use ndarray::{array, Array2, Array3};
use ndarray_linalg::assert_close_l2;
use num_complex::Complex;

use crate::auxiliary::molecule::Molecule;
use crate::basis::ao::*;
use crate::basis::ao_integrals::*;
use crate::io::numeric::NumericReader;

const ROOT: &str = env!("CARGO_MANIFEST_DIR");

type C128 = Complex<f64>;

#[test]
fn test_integrals_shell_tuple_collection() {
    let bs0 = BasisShell::new(1, ShellOrder::Cart(CartOrder::lex(1)));
    let gc0 = GaussianContraction::<f64, f64> {
        primitives: vec![(0.1, 0.1), (0.2, 0.2)],
    };
    let bsc0 = BasisShellContraction::<f64, f64> {
        basis_shell: bs0,
        contraction: gc0,
        cart_origin: Point3::new(1.0, 0.0, 0.0),
        k: None,
    };
    let bs1 = BasisShell::new(2, ShellOrder::Cart(CartOrder::lex(2)));
    let gc1 = GaussianContraction::<f64, f64> {
        primitives: vec![(0.3, 0.3), (0.4, 0.4), (0.5, 0.5)],
    };
    let bsc1 = BasisShellContraction::<f64, f64> {
        basis_shell: bs1,
        contraction: gc1,
        cart_origin: Point3::new(2.0, 1.0, 1.0),
        k: Some(Vector3::z()),
    };
    let bscs_1 = BasisSet::new(vec![vec![bsc0.clone()]]);
    let bscs_2 = BasisSet::new(vec![vec![bsc0], vec![bsc1]]);
    let stc = build_shell_tuple_collection![
        <s1, s2, s3, s4, s5>;
        true, true, false, true, false;
        &bscs_2, &bscs_1, &bscs_2, &bscs_2, &bscs_2;
        C128
    ];
    assert_eq!(stc.lmax(), 2);
    assert_eq!(stc.ccs, [true, true, false, true, false]);
    assert_eq!(stc.unique_shell_tuples_iter([1, 1, 0, 2, 0]).count(), 12);
    assert_eq!(
        stc.unique_shell_tuples_iter([1, 1, 0, 2, 0])
            .flat_map(|(_, _, equiv_perms)| equiv_perms)
            .count(),
        16
    );
    assert_eq!(stc.function_all_shell_shape, [9, 3, 9, 9, 9]);
}

#[test]
fn test_integrals_shell_tuple_collection_overlap_2c_h2() {
    // ~~~~~~~~~~~~~~~~~
    // H2, STO-3G
    // Reference: Q-Chem
    // ~~~~~~~~~~~~~~~~~
    let bs_cs = BasisShell::new(0, ShellOrder::Cart(CartOrder::lex(0)));
    let gc_h_sto3g_1s = GaussianContraction::<f64, f64> {
        primitives: vec![
            (3.4252509140, 0.1543289673),
            (0.6239137298, 0.5353281423),
            (0.1688554040, 0.4446345422),
        ],
    };

    let bsc0 = BasisShellContraction::<f64, f64> {
        basis_shell: bs_cs.clone(),
        contraction: gc_h_sto3g_1s.clone(),
        cart_origin: Point3::new(0.0, 0.0, 0.0),
        k: None,
    };
    let bsc1 = BasisShellContraction::<f64, f64> {
        basis_shell: bs_cs.clone(),
        contraction: gc_h_sto3g_1s,
        cart_origin: Point3::new(0.0, 0.0, 1.0),
        k: None,
    };

    let bscs = BasisSet::new(vec![vec![bsc0], vec![bsc1]]);
    let stc = build_shell_tuple_collection![
        <s1, s2>;
        true, false;
        &bscs, &bscs;
        f64
    ];
    let ovs = stc.overlap([0, 0]);

    #[rustfmt::skip]
    assert_close_l2!(
        &ovs[0],
        &array![
            [1.0000000, 0.7965883],
            [0.7965883, 1.0000000],
        ],
        1e-7
    );
}

#[test]
fn test_integrals_shell_tuple_collection_overlap_2c_bf3() {
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // BF3, cc-pVTZ (optimised contraction)
    // Reference: libint
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    let mol = Molecule::from_xyz(&format!("{ROOT}/tests/xyz/bf3.xyz"), 1e-7);

    let bscs = BasisSet::<f64, f64>::from_bse(&mol, "cc-pVTZ", true, true, 0, false, true).unwrap();
    let stc = build_shell_tuple_collection![
        <s1, s2>;
        true, false;
        &bscs, &bscs;
        f64
    ];
    let ovs = stc.overlap([0, 0]);

    let sao_v = NumericReader::<_, LittleEndian, f64>::from_file(format!(
        "{ROOT}/tests/binaries/integrals/bf3_sao/sao_libint"
    ))
    .unwrap()
    .collect::<Vec<_>>();
    let sao = Array2::from_shape_vec((140, 140), sao_v).unwrap();
    assert_close_l2!(&ovs[0], &sao, 1e-7);
}

#[test]
fn test_integrals_shell_tuple_collection_overlap_2c_benzene() {
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // Benzene, cc-pVQZ (optimised contraction)
    // Reference: libint
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    let mol = Molecule::from_xyz(&format!("{ROOT}/tests/xyz/benzene.xyz"), 1e-7);

    let bscs = BasisSet::<f64, f64>::from_bse(&mol, "cc-pVQZ", true, true, 0, false, true).unwrap();
    let stc = build_shell_tuple_collection![
        <s1, s2>;
        false, false;
        &bscs, &bscs;
        f64
    ];
    let ovs = stc.overlap([0, 0]);
    let sao_v = NumericReader::<_, LittleEndian, f64>::from_file(format!(
        "{ROOT}/tests/binaries/integrals/benzene_sao/sao_libint_opt"
    ))
    .unwrap()
    .collect::<Vec<_>>();
    let sao = Array2::from_shape_vec((630, 630), sao_v).unwrap();
    assert_close_l2!(&ovs[0], &sao, 3e-7);
}

#[test]
fn test_integrals_shell_tuple_collection_overlap_3c_h6() {
    // ~~~~~~~~~~~~~~~~
    // H6 (octahedral)
    // Reference: QUEST
    // ~~~~~~~~~~~~~~~~
    let mol = Molecule::from_xyz(&format!("{ROOT}/tests/xyz/h6_oct.xyz"), 1e-7);

    for cart in [true, false] {
        for (bas_name, bas_sym) in [("STO-3G", "sto3g"), ("6-31G*", "631gs")] {
            let mut bscs = BasisSet::<f64, f64>::from_bse(
                &mol, bas_name, cart,  // cart
                false, // optimised contraction
                0,     // version
                true,  // mol_bohr
                true,  // force renormalisation
            )
            .unwrap();

            // QUEST-specific: shells are grouped by angular momentum.
            bscs.sort_by_angular_momentum();

            // QUEST-specific: P functions are always in Cartesian order.
            if !cart {
                bscs.all_shells_mut().for_each(|bsc| {
                    if bsc.basis_shell.l == 1 {
                        bsc.basis_shell.shell_order = ShellOrder::Cart(CartOrder::lex(1))
                    }
                });
            }

            let stc = build_shell_tuple_collection![
                <s1, s2, s3>;
                false, false, false;
                &bscs, &bscs, &bscs;
                f64
            ];
            let ovs = stc.overlap([0, 0, 0]);
            let sao_v = NumericReader::<_, LittleEndian, f64>::from_file(format!(
                "{ROOT}/tests/binaries/integrals/h6_3c/{}_gao_H6_{bas_sym}_contracted",
                if cart { "cartesian" } else { "spherical" }
            ))
            .unwrap()
            .collect::<Vec<_>>();
            let sao_3c = Array3::from_shape_vec(ovs[0].raw_dim(), sao_v).unwrap();
            assert_close_l2!(&ovs[0], &sao_3c, 1e-7);
        }
    }
}

#[test]
fn test_integrals_shell_tuple_collection_overlap_3c_li() {
    // ~~~~~~~~~~~~~~~~
    // Li
    // Reference: QUEST
    // ~~~~~~~~~~~~~~~~
    let mol = Molecule::from_xyz(&format!("{ROOT}/tests/xyz/li.xyz"), 1e-7);

    for cart in [true, false] {
        for (bas_name, bas_sym) in [("STO-3G", "sto3g"), ("6-31G*", "631gs")] {
            let mut bscs = BasisSet::<f64, f64>::from_bse(
                &mol, bas_name, cart,  // cart
                false, // optimised contraction
                0,     // version
                true,  // mol_bohr
                true,  // force renormalisation
            )
            .unwrap();

            // QUEST-specific: shells are grouped by angular momentum.
            bscs.sort_by_angular_momentum();

            // QUEST-specific: P functions are always in Cartesian order.
            if !cart {
                bscs.all_shells_mut().for_each(|bsc| {
                    if bsc.basis_shell.l == 1 {
                        bsc.basis_shell.shell_order = ShellOrder::Cart(CartOrder::lex(1))
                    }
                });
            }

            let stc = build_shell_tuple_collection![
                <s1, s2, s3>;
                false, false, false;
                &bscs, &bscs, &bscs;
                f64
            ];
            let ovs = stc.overlap([0, 0, 0]);
            let sao_v = NumericReader::<_, LittleEndian, f64>::from_file(format!(
                "{ROOT}/tests/binaries/integrals/li_3c/{}_gao_Li_{bas_sym}_contracted",
                if cart { "cartesian" } else { "spherical" }
            ))
            .unwrap()
            .collect::<Vec<_>>();
            let sao_3c = Array3::from_shape_vec(ovs[0].raw_dim(), sao_v).unwrap();
            assert_close_l2!(&ovs[0], &sao_3c, 1e-7);
        }
    }
}

#[test]
fn test_integrals_shell_tuple_collection_overlap_3c_li2() {
    // ~~~~~~~~~~~~~~~~
    // Li2
    // Reference: QUEST
    // ~~~~~~~~~~~~~~~~
    let mol = Molecule::from_xyz(&format!("{ROOT}/tests/xyz/li2.xyz"), 1e-7);

    for cart in [true, false] {
        for (bas_name, bas_sym) in [("STO-3G", "sto3g"), ("6-31G*", "631gs")] {
            let mut bscs = BasisSet::<f64, f64>::from_bse(
                &mol, bas_name, cart,  // cart
                false, // optimised contraction
                0,     // version
                true,  // mol_bohr
                true,  // force renormalisation
            )
            .unwrap();

            // QUEST-specific: shells are grouped by angular momentum on each atom.
            bscs.sort_by_angular_momentum();

            // QUEST-specific: P functions are always in Cartesian order.
            if !cart {
                bscs.all_shells_mut().for_each(|bsc| {
                    if bsc.basis_shell.l == 1 {
                        bsc.basis_shell.shell_order = ShellOrder::Cart(CartOrder::lex(1))
                    }
                });
            }

            let stc = build_shell_tuple_collection![
                <s1, s2, s3>;
                false, false, false;
                &bscs, &bscs, &bscs;
                f64
            ];
            let ovs = stc.overlap([0, 0, 0]);
            let sao_v = NumericReader::<_, LittleEndian, f64>::from_file(format!(
                "{ROOT}/tests/binaries/integrals/li2_3c/{}_gao_Li2_{bas_sym}_contracted",
                if cart { "cartesian" } else { "spherical" }
            ))
            .unwrap()
            .collect::<Vec<_>>();
            let sao_3c = Array3::from_shape_vec(ovs[0].raw_dim(), sao_v).unwrap();
            assert_close_l2!(&ovs[0], &sao_3c, 1e-7);
        }
    }
}
