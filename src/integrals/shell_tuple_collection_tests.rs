use byteorder::LittleEndian;
use duplicate::duplicate_item;
use nalgebra::{Point3, Vector3};
use ndarray::{array, Array2, Array3};
use ndarray_linalg::assert::close_l2;
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
    assert_eq!(stc.angular_all_shell_shape, [9, 3, 9, 9, 9]);
}

#[test]
fn test_integrals_shell_tuple_collection_overlap_gao_2c_h2() {
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
    close_l2(
        &ovs[0],
        &array![
            [1.0000000, 0.7965883],
            [0.7965883, 1.0000000],
        ],
        1e-7
    );
}

#[test]
fn test_integrals_shell_tuple_collection_overlap_gao_2c_bf3() {
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
        "{ROOT}/tests/binaries/integrals/bf3_2c/sao_libint"
    ))
    .unwrap()
    .collect::<Vec<_>>();
    let sao = Array2::from_shape_vec((140, 140), sao_v).unwrap();
    close_l2(&ovs[0], &sao, 1e-7);
}

#[test]
fn test_integrals_shell_tuple_collection_overlap_gao_2c_benzene() {
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
        "{ROOT}/tests/binaries/integrals/benzene_2c/sao_libint_opt"
    ))
    .unwrap()
    .collect::<Vec<_>>();
    let sao = Array2::from_shape_vec((630, 630), sao_v).unwrap();
    close_l2(&ovs[0], &sao, 3e-7);
}

#[duplicate_item(
    [
        gao_3c_test_name [ test_integrals_shell_tuple_collection_overlap_gao_3c_h6 ]
        mol_name_1 [ "h6_oct" ]
        dir_name [ "h6_3c" ]
        mol_name_2 [ "H6" ]
    ]
    [
        gao_3c_test_name [ test_integrals_shell_tuple_collection_overlap_gao_3c_li ]
        mol_name_1 [ "li" ]
        dir_name [ "li_3c" ]
        mol_name_2 [ "Li" ]
    ]
    [
        gao_3c_test_name [ test_integrals_shell_tuple_collection_overlap_gao_3c_li2 ]
        mol_name_1 [ "li2" ]
        dir_name [ "li2_3c" ]
        mol_name_2 [ "Li2" ]
    ]
    [
        gao_3c_test_name [ test_integrals_shell_tuple_collection_overlap_gao_3c_bf3_offcentre ]
        mol_name_1 [ "bf3_offcentre" ]
        dir_name [ "bf3_offcentre_3c" ]
        mol_name_2 [ "BF3" ]
    ]
    [
        gao_3c_test_name [ test_integrals_shell_tuple_collection_overlap_gao_3c_cr ]
        mol_name_1 [ "cr" ]
        dir_name [ "cr_3c" ]
        mol_name_2 [ "Cr" ]
    ]
)]
#[test]
fn gao_3c_test_name() {
    // ~~~~~~~~~~~~~~~~
    // Reference: QUEST
    // ~~~~~~~~~~~~~~~~
    let mol_name = mol_name_1;
    let mol = Molecule::from_xyz(&format!("{ROOT}/tests/xyz/{mol_name}.xyz"), 1e-7);

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
            let dir = dir_name;
            let mol_name = mol_name_2;
            let sao_v = NumericReader::<_, LittleEndian, f64>::from_file(format!(
                "{ROOT}/tests/binaries/integrals/{dir}/{}_gao_{mol_name}_{bas_sym}_contracted",
                if cart { "cartesian" } else { "spherical" }
            ))
            .unwrap()
            .collect::<Vec<_>>();
            let sao_3c = Array3::from_shape_vec(ovs[0].raw_dim(), sao_v).unwrap();
            close_l2(&ovs[0], &sao_3c, 1e-7);
        }
    }
}

#[duplicate_item(
    [
        lao_2c_test_name [ test_integrals_shell_tuple_collection_overlap_lao_2c_h2 ]
        mol_name_1 [ "h2_x" ]
        dir_name [ "h2_2c" ]
        mol_name_2 [ "H2" ]
        bases [
            ("STO-3G", "sto3g"),
            ("6-31G*", "631gs"),
            ("cc-pVTZ", "ccpvtz"),
        ]
    ]
    [
        lao_2c_test_name [ test_integrals_shell_tuple_collection_overlap_lao_2c_h6 ]
        mol_name_1 [ "h6_oct" ]
        dir_name [ "h6_2c" ]
        mol_name_2 [ "H6" ]
        bases [
            ("STO-3G", "sto3g"),
            ("6-31G*", "631gs"),
            ("cc-pVTZ", "ccpvtz"),
        ]
    ]
    [
        lao_2c_test_name [ test_integrals_shell_tuple_collection_overlap_lao_2c_li ]
        mol_name_1 [ "li" ]
        dir_name [ "li_2c" ]
        mol_name_2 [ "Li" ]
        bases [
            ("STO-3G", "sto3g"),
            ("6-31G*", "631gs"),
            ("cc-pVTZ", "ccpvtz"),
        ]
    ]
    [
        lao_2c_test_name [ test_integrals_shell_tuple_collection_overlap_lao_2c_li2 ]
        mol_name_1 [ "li2" ]
        dir_name [ "li2_2c" ]
        mol_name_2 [ "Li2" ]
        bases [
            ("STO-3G", "sto3g"),
            ("6-31G*", "631gs"),
            ("cc-pVTZ", "ccpvtz"),
        ]
    ]
    [
        lao_2c_test_name [ test_integrals_shell_tuple_collection_overlap_lao_2c_li6 ]
        mol_name_1 [ "li6" ]
        dir_name [ "li6_2c" ]
        mol_name_2 [ "Li6" ]
        bases [
            ("STO-3G", "sto3g"),
            ("6-31G*", "631gs"),
            ("cc-pVTZ", "ccpvtz"),
        ]
    ]
    [
        lao_2c_test_name [ test_integrals_shell_tuple_collection_overlap_lao_2c_bf3 ]
        mol_name_1 [ "bf3_offcentre" ]
        dir_name [ "bf3_offcentre_2c" ]
        mol_name_2 [ "BF3" ]
        bases [
            ("STO-3G", "sto3g"),
            ("6-31G*", "631gs"),
            ("cc-pVTZ", "ccpvtz"),
        ]
    ]
    [
        lao_2c_test_name [ test_integrals_shell_tuple_collection_overlap_lao_2c_cr ]
        mol_name_1 [ "cr" ]
        dir_name [ "cr_2c" ]
        mol_name_2 [ "Cr" ]
        bases [
            ("STO-3G", "sto3g"),
            ("6-31G*", "631gs"),
            ("cc-pVTZ", "ccpvtz"),
            ("cc-pVTZ", "ccpvtz"), // QUEST swaps the two D shells. These have been manually
                                   // swapped back in the creation of QUEST reference data.
        ]
    ]
    [
        lao_2c_test_name [ test_integrals_shell_tuple_collection_overlap_lao_2c_cr2 ]
        mol_name_1 [ "cr2" ]
        dir_name [ "cr2_2c" ]
        mol_name_2 [ "Cr2" ]
        bases [
            ("STO-3G", "sto3g"),
            ("6-31G*", "631gs"),
            ("cc-pVTZ", "ccpvtz"), // QUEST swaps the two D shells. These have been manually
                                   // swapped back in the creation of QUEST reference data.
        ]
    ]
)]
#[test]
fn lao_2c_test_name() {
    // ~~~~~~~~~~~~~~~~
    // Reference: QUEST
    // ~~~~~~~~~~~~~~~~
    let mol_name = mol_name_1;
    let mol = Molecule::from_xyz(&format!("{ROOT}/tests/xyz/{mol_name}.xyz"), 1e-7);

    for cart in [true] {
        for (bas_name, bas_sym) in [bases] {
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

            bscs.apply_magnetic_field(&Vector3::new(0.0, 0.0, 0.1), &Point3::origin());

            let stc = build_shell_tuple_collection![
                <s1, s2>;
                true, false;
                &bscs, &bscs;
                C128
            ];
            let ovs = stc.overlap([0, 0]);
            let dir = dir_name;
            let mol_name = mol_name_2;
            let sao_v = NumericReader::<_, LittleEndian, C128>::from_file(format!(
                "{ROOT}/tests/binaries/integrals/{dir}/{}_lao_{mol_name}_{bas_sym}_contracted",
                if cart { "cartesian" } else { "spherical" }
            ))
            .unwrap()
            .collect::<Vec<_>>();
            let sao_2c = Array2::from_shape_vec(ovs[0].raw_dim(), sao_v).unwrap();
            close_l2(&ovs[0], &sao_2c, 1e-7);
        }
    }
}
