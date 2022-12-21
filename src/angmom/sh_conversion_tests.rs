use approx;
use itertools::Itertools;
use ndarray::{array, Array2};
use num::Complex;
use num_traits::{One, Zero};

use crate::angmom::sh_conversion::{
    complexc, complexcinv, norm_cart_gaussian, norm_sph_gaussian, sh_c2r_mat, sh_cart2cl_mat,
    sh_cart2r, sh_cart2rl_mat, sh_cl2cart_mat, sh_r2c_mat, sh_r2cart, sh_rl2cart_mat, CartOrder,
};

type C128 = Complex<f64>;

#[test]
fn test_sh_conversion_cartorder() {
    // =========
    // lcart = 0
    // =========
    let co_0_lex = CartOrder::lex(0);
    assert_eq!(co_0_lex.cart_tuples, vec![(0, 0, 0)]);

    let co_0_qchem = CartOrder::qchem(0);
    assert_eq!(co_0_qchem.cart_tuples, vec![(0, 0, 0)]);

    // =========
    // lcart = 1
    // =========
    let co_1_lex = CartOrder::lex(1);
    assert_eq!(co_1_lex.cart_tuples, vec![(1, 0, 0), (0, 1, 0), (0, 0, 1)]);

    let co_1_qchem = CartOrder::qchem(1);
    assert_eq!(
        co_1_qchem.cart_tuples,
        vec![(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    );

    // =========
    // lcart = 2
    // =========
    let co_2_lex = CartOrder::lex(2);
    assert_eq!(
        co_2_lex.cart_tuples,
        vec![
            (2, 0, 0),
            (1, 1, 0),
            (1, 0, 1),
            (0, 2, 0),
            (0, 1, 1),
            (0, 0, 2),
        ]
    );

    let co_2_qchem = CartOrder::qchem(2);
    assert_eq!(
        co_2_qchem.cart_tuples,
        vec![
            (2, 0, 0),
            (1, 1, 0),
            (0, 2, 0),
            (1, 0, 1),
            (0, 1, 1),
            (0, 0, 2),
        ]
    );

    // =========
    // lcart = 3
    // =========
    let co_3_lex = CartOrder::lex(3);
    assert_eq!(
        co_3_lex.cart_tuples,
        vec![
            (3, 0, 0),
            (2, 1, 0),
            (2, 0, 1),
            (1, 2, 0),
            (1, 1, 1),
            (1, 0, 2),
            (0, 3, 0),
            (0, 2, 1),
            (0, 1, 2),
            (0, 0, 3),
        ]
    );

    let co_3_qchem = CartOrder::qchem(3);
    assert_eq!(
        co_3_qchem.cart_tuples,
        vec![
            (3, 0, 0),
            (2, 1, 0),
            (1, 2, 0),
            (0, 3, 0),
            (2, 0, 1),
            (1, 1, 1),
            (0, 2, 1),
            (1, 0, 2),
            (0, 1, 2),
            (0, 0, 3),
        ]
    );

    // =========
    // lcart = 4
    // =========
    let co_4_lex = CartOrder::lex(4);
    assert_eq!(
        co_4_lex.cart_tuples,
        vec![
            (4, 0, 0),
            (3, 1, 0),
            (3, 0, 1),
            (2, 2, 0),
            (2, 1, 1),
            (2, 0, 2),
            (1, 3, 0),
            (1, 2, 1),
            (1, 1, 2),
            (1, 0, 3),
            (0, 4, 0),
            (0, 3, 1),
            (0, 2, 2),
            (0, 1, 3),
            (0, 0, 4),
        ]
    );

    let co_4_qchem = CartOrder::qchem(4);
    assert_eq!(
        co_4_qchem.cart_tuples,
        vec![
            (4, 0, 0),
            (3, 1, 0),
            (2, 2, 0),
            (1, 3, 0),
            (0, 4, 0),
            (3, 0, 1),
            (2, 1, 1),
            (1, 2, 1),
            (0, 3, 1),
            (2, 0, 2),
            (1, 1, 2),
            (0, 2, 2),
            (1, 0, 3),
            (0, 1, 3),
            (0, 0, 4),
        ]
    );
}

#[test]
fn test_sh_conversion_complexc() {
    // =====
    // l = 0
    // =====
    for lcartqns in (0..=20).combinations_with_replacement(3) {
        let lx = lcartqns[0];
        let ly = lcartqns[1];
        let lz = lcartqns[2];
        if lx == 0 && ly == 0 && lz == 0 {
            assert!((complexc((0, 0), (lx, ly, lz), true) - C128::one()).norm() < 1e-14);
        } else {
            let lcart = lx + ly + lz;
            if lcart.rem_euclid(2) != 0
                || (lx.rem_euclid(2) != 0 || ly.rem_euclid(2) != 0 || lz.rem_euclid(2) != 0)
            {
                assert!((complexc((0, 0), (lx, ly, lz), true) - C128::zero()).norm() < 1e-14);
            } else {
                assert!((complexc((0, 0), (lx, ly, lz), true) - C128::zero()).norm() >= 1e-14);
            }
        }
    }

    let ntilde_2 = norm_sph_gaussian(2, 1.0);
    let n_200 = norm_cart_gaussian((2, 0, 0), 1.0);
    let complexc_00_200_ref = ntilde_2 / (n_200 * (4.0 * std::f64::consts::PI).sqrt());
    assert!((complexc((0, 0), (2, 0, 0), true) - complexc_00_200_ref).norm() < 1e-14);
    assert!((complexc((0, 0), (0, 2, 0), true) - complexc_00_200_ref).norm() < 1e-14);
    assert!((complexc((0, 0), (0, 0, 2), true) - complexc_00_200_ref).norm() < 1e-14);

    // =====
    // l = 1
    // =====
    // Specific values
    assert!((complexc((1, 0), (0, 0, 1), true) - C128::one()).norm() < 1e-14);

    assert!(
        (complexc((1, 1), (1, 0, 0), true) - C128::new(-1.0 / 2.0f64.sqrt(), 0.0)).norm() < 1e-14
    );
    assert!(
        (complexc((1, 1), (0, 1, 0), true) - C128::new(0.0, -1.0 / 2.0f64.sqrt())).norm() < 1e-14
    );

    assert!(
        (complexc((1, -1), (1, 0, 0), true) - C128::new(1.0 / 2.0f64.sqrt(), 0.0)).norm() < 1e-14
    );
    assert!(
        (complexc((1, -1), (0, 1, 0), true) - C128::new(0.0, -1.0 / 2.0f64.sqrt())).norm() < 1e-14
    );

    // Generic values
    for lcartqns in (0..=20).combinations_with_replacement(3) {
        let lx = lcartqns[0];
        let ly = lcartqns[1];
        let lz = lcartqns[2];
        let lcart: u32 = lx + ly + lz;
        if lcart == 1 {
            continue;
        } else if lcart.rem_euclid(2) == 0 {
            for m in -1..=1 {
                assert!((complexc((1, m), (lx, ly, lz), true) - C128::zero()).norm() < 1e-14);
            }
        }
    }

    // =====
    // l = 2
    // =====
    // Specific values
    assert!((complexc((2, 0), (0, 0, 2), true) - C128::one()).norm() < 1e-14);
    assert!((complexc((2, 0), (2, 0, 0), true) - C128::new(-0.5, 0.0)).norm() < 1e-14);
    assert!((complexc((2, 0), (0, 2, 0), true) - C128::new(-0.5, 0.0)).norm() < 1e-14);

    assert!(
        (complexc((2, 1), (1, 0, 1), true) - C128::new(-1.0 / 2.0f64.sqrt(), 0.0)).norm() < 1e-14
    );
    assert!(
        (complexc((2, 1), (0, 1, 1), true) - C128::new(0.0, -1.0 / 2.0f64.sqrt())).norm() < 1e-14
    );

    assert!(
        (complexc((2, -1), (1, 0, 1), true) - C128::new(1.0 / 2.0f64.sqrt(), 0.0)).norm() < 1e-14
    );
    assert!(
        (complexc((2, -1), (0, 1, 1), true) - C128::new(0.0, -1.0 / 2.0f64.sqrt())).norm() < 1e-14
    );

    assert!(
        (complexc((2, 2), (2, 0, 0), true) - C128::new((3.0f64 / 8.0f64).sqrt(), 0.0)).norm()
            < 1e-14
    );
    assert!(
        (complexc((2, 2), (0, 2, 0), true) - C128::new(-(3.0f64 / 8.0f64).sqrt(), 0.0)).norm()
            < 1e-14
    );
    assert!(
        (complexc((2, 2), (1, 1, 0), true) - C128::new(0.0, 1.0 / 2.0f64.sqrt())).norm() < 1e-14
    );

    assert!((complexc((2, -2), (0, 0, 0), true) - C128::zero()).norm() < 1e-14);
    assert!(
        (complexc((2, -2), (2, 0, 0), true) - C128::new((3.0f64 / 8.0f64).sqrt(), 0.0)).norm()
            < 1e-14
    );
    assert!(
        (complexc((2, -2), (0, 2, 0), true) - C128::new(-(3.0f64 / 8.0f64).sqrt(), 0.0)).norm()
            < 1e-14
    );
    assert!(
        (complexc((2, -2), (1, 1, 0), true) - C128::new(0.0, -1.0 / 2.0f64.sqrt())).norm() < 1e-14
    );

    // Generic values
    for lcartqns in (0..=20).combinations_with_replacement(3) {
        let lx = lcartqns[0];
        let ly = lcartqns[1];
        let lz = lcartqns[2];
        let lcart: u32 = lx + ly + lz;
        if lcart == 1 {
            continue;
        } else if lcart.rem_euclid(2) != 0 {
            // println!("{}, {}, {} -- {}", 1, 0, 0, complexc((1, 1), (1, 0, 0), true));
            for m in -2..=2 {
                assert!((complexc((2, m), (lx, ly, lz), true) - C128::zero()).norm() < 1e-14);
            }
        }
    }

    // =====
    // l = 3
    // =====
    // Specific values
    assert!((complexc((3, 0), (0, 0, 3), true) - C128::one()).norm() < 1e-14);
    assert!(
        (complexc((3, 0), (2, 0, 1), true) - C128::new(-3.0 / (2.0 * 5.0f64.sqrt()), 0.0)).norm()
            < 1e-14
    );
    assert!(
        (complexc((3, 0), (0, 2, 1), true) - C128::new(-3.0 / (2.0 * 5.0f64.sqrt()), 0.0)).norm()
            < 1e-14
    );

    assert!(
        (complexc((3, 1), (1, 0, 2), true) - C128::new(-(3.0f64 / 5.0f64).sqrt(), 0.0)).norm()
            < 1e-14
    );
    assert!(
        (complexc((3, 1), (0, 1, 2), true) - C128::new(0.0, -(3.0f64 / 5.0f64).sqrt())).norm()
            < 1e-14
    );
    assert!(
        (complexc((3, 1), (3, 0, 0), true) - C128::new(3.0f64.sqrt() / 4.0, 0.0)).norm() < 1e-14
    );
    assert!(
        (complexc((3, 1), (0, 3, 0), true) - C128::new(0.0, 3.0f64.sqrt() / 4.0)).norm() < 1e-14
    );
    assert!(
        (complexc((3, 1), (1, 2, 0), true) - C128::new(3.0f64.sqrt() / (4.0 * 5.0f64.sqrt()), 0.0))
            .norm()
            < 1e-14
    );
    assert!(
        (complexc((3, 1), (2, 1, 0), true) - C128::new(0.0, 3.0f64.sqrt() / (4.0 * 5.0f64.sqrt())))
            .norm()
            < 1e-14
    );

    assert!(
        (complexc((3, -1), (1, 0, 2), true) - C128::new((3.0f64 / 5.0f64).sqrt(), 0.0)).norm()
            < 1e-14
    );
    assert!(
        (complexc((3, -1), (0, 1, 2), true) - C128::new(0.0, -(3.0f64 / 5.0f64).sqrt())).norm()
            < 1e-14
    );
    assert!(
        (complexc((3, -1), (3, 0, 0), true) - C128::new(-3.0f64.sqrt() / 4.0, 0.0)).norm() < 1e-14
    );
    assert!(
        (complexc((3, -1), (0, 3, 0), true) - C128::new(0.0, 3.0f64.sqrt() / 4.0)).norm() < 1e-14
    );
    assert!(
        (complexc((3, -1), (1, 2, 0), true)
            - C128::new(-3.0f64.sqrt() / (4.0 * 5.0f64.sqrt()), 0.0))
        .norm()
            < 1e-14
    );
    assert!(
        (complexc((3, -1), (2, 1, 0), true)
            - C128::new(0.0, 3.0f64.sqrt() / (4.0 * 5.0f64.sqrt())))
        .norm()
            < 1e-14
    );

    assert!(
        (complexc((3, 2), (2, 0, 1), true) - C128::new((3.0f64 / 8.0f64).sqrt(), 0.0)).norm()
            < 1e-14
    );
    assert!(
        (complexc((3, 2), (0, 2, 1), true) - C128::new(-(3.0f64 / 8.0f64).sqrt(), 0.0)).norm()
            < 1e-14
    );
    assert!(
        (complexc((3, 2), (1, 1, 1), true) - C128::new(0.0, 1.0 / (2.0f64).sqrt())).norm() < 1e-14
    );

    assert!(
        (complexc((3, -2), (2, 0, 1), true) - C128::new((3.0f64 / 8.0f64).sqrt(), 0.0)).norm()
            < 1e-14
    );
    assert!(
        (complexc((3, -2), (0, 2, 1), true) - C128::new(-(3.0f64 / 8.0f64).sqrt(), 0.0)).norm()
            < 1e-14
    );
    assert!(
        (complexc((3, -2), (1, 1, 1), true) - C128::new(0.0, -1.0 / (2.0f64).sqrt())).norm()
            < 1e-14
    );

    assert!(
        (complexc((3, 3), (3, 0, 0), true) - C128::new(-5.0f64.sqrt() / 4.0, 0.0)).norm() < 1e-14
    );
    assert!(
        (complexc((3, 3), (0, 3, 0), true) - C128::new(0.0, 5.0f64.sqrt() / 4.0)).norm() < 1e-14
    );
    assert!((complexc((3, 3), (1, 2, 0), true) - C128::new(3.0 / 4.0, 0.0)).norm() < 1e-14);
    assert!((complexc((3, 3), (2, 1, 0), true) - C128::new(0.0, -3.0 / 4.0)).norm() < 1e-14);

    assert!(
        (complexc((3, -3), (3, 0, 0), true) - C128::new(5.0f64.sqrt() / 4.0, 0.0)).norm() < 1e-14
    );
    assert!(
        (complexc((3, -3), (0, 3, 0), true) - C128::new(0.0, 5.0f64.sqrt() / 4.0)).norm() < 1e-14
    );
    assert!((complexc((3, -3), (1, 2, 0), true) - C128::new(-3.0 / 4.0, 0.0)).norm() < 1e-14);
    assert!((complexc((3, -3), (2, 1, 0), true) - C128::new(0.0, -3.0 / 4.0)).norm() < 1e-14);

    // Generic values
    for lcartqns in (0..=20).combinations_with_replacement(3) {
        let lx = lcartqns[0];
        let ly = lcartqns[1];
        let lz = lcartqns[2];
        let lcart: u32 = lx + ly + lz;
        if lcart == 3 {
            continue;
        } else if lcart.rem_euclid(2) == 0 || lcart < 3 {
            for m in -3..=3 {
                assert!((complexc((3, m), (lx, ly, lz), true) - C128::zero()).norm() < 1e-14);
            }
        }
    }
}

#[test]
fn test_sh_conversion_complexcinv() {
    // =========
    // lcart = 0
    // =========
    assert!((complexcinv((0, 0, 0), (0, 0), true) - C128::one()).norm() < 1e-14);

    // =========
    // lcart = 1
    // =========
    assert!((complexcinv((0, 0, 1), (1, 0), true) - C128::one()).norm() < 1e-14);
    assert!((complexcinv((1, 0, 0), (0, 0), true) - C128::zero()).norm() < 1e-14);
    assert!(
        (complexcinv((1, 0, 0), (1, 1), true) - C128::new(-1.0 / 2.0f64.sqrt(), 0.0)).norm()
            < 1e-14
    );
    assert!(
        (complexcinv((1, 0, 0), (1, -1), true) - C128::new(1.0 / 2.0f64.sqrt(), 0.0)).norm()
            < 1e-14
    );
    assert!(
        (complexcinv((0, 1, 0), (1, 1), true) - C128::new(0.0, 1.0 / 2.0f64.sqrt())).norm() < 1e-14
    );
    assert!(
        (complexcinv((0, 1, 0), (1, -1), true) - C128::new(0.0, 1.0 / 2.0f64.sqrt())).norm()
            < 1e-14
    );

    // =========
    // lcart = 2
    // =========
    assert!((complexcinv((1, 1, 0), (1, -1), true) - C128::zero()).norm() < 1e-14);

    let ntilde_2 = norm_sph_gaussian(2, 1.0);
    let n_200 = norm_cart_gaussian((2, 0, 0), 1.0);
    assert!((complexcinv((0, 0, 2), (2, 0), true) - C128::new(2.0 / 3.0, 0.0)).norm() < 1e-14);
    assert!(
        (complexcinv((0, 0, 2), (0, 0), true)
            - C128::new(
                1.0 / 3.0 * n_200 / ntilde_2 * (4.0 * std::f64::consts::PI).sqrt(),
                0.0
            ))
        .norm()
            < 1e-14
    );

    assert!((complexcinv((0, 0, 2), (2, 0), true) - C128::new(2.0 / 3.0, 0.0)).norm() < 1e-14);
    assert!(
        (complexcinv((0, 0, 2), (0, 0), true)
            - C128::new(
                1.0 / 3.0 * n_200 / ntilde_2 * (4.0 * std::f64::consts::PI).sqrt(),
                0.0
            ))
        .norm()
            < 1e-14
    );

    assert!((complexcinv((2, 0, 0), (2, 0), true) - C128::new(-1.0 / 3.0, 0.0)).norm() < 1e-14);
    assert!(
        (complexcinv((2, 0, 0), (0, 0), true)
            - C128::new(
                1.0 / 3.0 * n_200 / ntilde_2 * (4.0 * std::f64::consts::PI).sqrt(),
                0.0
            ))
        .norm()
            < 1e-14
    );
    assert!(
        (complexcinv((2, 0, 0), (2, 2), true) - C128::new(1.0 / 6.0f64.sqrt(), 0.0)).norm() < 1e-14
    );
    assert!(
        (complexcinv((2, 0, 0), (2, -2), true) - C128::new(1.0 / 6.0f64.sqrt(), 0.0)).norm()
            < 1e-14
    );

    assert!(
        (complexcinv((0, 2, 0), (0, 0), true)
            - C128::new(
                1.0 / 3.0 * n_200 / ntilde_2 * (4.0 * std::f64::consts::PI).sqrt(),
                0.0
            ))
        .norm()
            < 1e-14
    );
    assert!((complexcinv((0, 2, 0), (2, 0), true) - C128::new(-1.0 / 3.0, 0.0)).norm() < 1e-14);
    assert!(
        (complexcinv((0, 2, 0), (2, 2), true) - C128::new(-1.0 / 6.0f64.sqrt(), 0.0)).norm()
            < 1e-14
    );
    assert!(
        (complexcinv((0, 2, 0), (2, -2), true) - C128::new(-1.0 / 6.0f64.sqrt(), 0.0)).norm()
            < 1e-14
    );
}

#[test]
fn test_sh_conversion_c2r() {
    let sq2 = 2.0f64.sqrt();

    let c2r0 = sh_c2r_mat(0, true, true);
    assert_eq!(c2r0.shape(), &[1, 1]);
    assert_eq!(c2r0[(0, 0)], C128::from(1.0));

    let c2r1 = sh_c2r_mat(1, true, true);
    assert_eq!(c2r1.shape(), &[3, 3]);
    let c2r1_ref = array![
        [
            C128::new(0.0, -1.0 / sq2),
            C128::zero(),
            C128::new(0.0, -1.0 / sq2)
        ],
        [C128::zero(), C128::one(), C128::zero()],
        [
            C128::new(1.0 / sq2, 0.0),
            C128::zero(),
            C128::new(-1.0 / sq2, 0.0)
        ],
    ];
    assert_eq!(c2r1, c2r1_ref);

    let c2r1_decreasingm = sh_c2r_mat(1, true, false);
    assert_eq!(c2r1_decreasingm.shape(), &[3, 3]);
    let c2r1_decreasingm_ref = array![
        [
            C128::new(-1.0 / sq2, 0.0),
            C128::zero(),
            C128::new(1.0 / sq2, 0.0)
        ],
        [C128::zero(), C128::one(), C128::zero()],
        [
            C128::new(0.0, -1.0 / sq2),
            C128::zero(),
            C128::new(0.0, -1.0 / sq2)
        ],
    ];
    assert_eq!(c2r1_decreasingm, c2r1_decreasingm_ref);

    let c2r2 = sh_c2r_mat(2, true, true);
    assert_eq!(c2r2.shape(), &[5, 5]);
    let c2r2_ref = array![
        [
            C128::new(0.0, -1.0 / sq2),
            C128::zero(),
            C128::zero(),
            C128::zero(),
            C128::new(0.0, 1.0 / sq2)
        ],
        [
            C128::zero(),
            C128::new(0.0, -1.0 / sq2),
            C128::zero(),
            C128::new(0.0, -1.0 / sq2),
            C128::zero()
        ],
        [
            C128::zero(),
            C128::zero(),
            C128::one(),
            C128::zero(),
            C128::zero()
        ],
        [
            C128::zero(),
            C128::new(1.0 / sq2, 0.0),
            C128::zero(),
            C128::new(-1.0 / sq2, 0.0),
            C128::zero()
        ],
        [
            C128::new(1.0 / sq2, 0.0),
            C128::zero(),
            C128::zero(),
            C128::zero(),
            C128::new(1.0 / sq2, 0.0)
        ],
    ];
    assert_eq!(c2r2, c2r2_ref);
}

#[test]
fn test_sh_conversion_r2c() {
    let sq2 = 2.0f64.sqrt();

    let r2c0 = sh_r2c_mat(0, true, true);
    assert_eq!(r2c0.shape(), &[1, 1]);
    assert_eq!(r2c0[(0, 0)], C128::from(1.0));

    let r2c1 = sh_r2c_mat(1, true, true);
    assert_eq!(r2c1.shape(), &[3, 3]);
    let r2c1_ref = array![
        [
            C128::new(0.0, 1.0 / sq2),
            C128::zero(),
            C128::new(1.0 / sq2, 0.0)
        ],
        [C128::zero(), C128::one(), C128::zero()],
        [
            C128::new(0.0, 1.0 / sq2),
            C128::zero(),
            C128::new(-1.0 / sq2, 0.0)
        ],
    ];
    assert_eq!(r2c1, r2c1_ref);

    let r2c2 = sh_r2c_mat(2, true, true);
    assert_eq!(r2c2.shape(), &[5, 5]);
    let r2c2_ref = array![
        [
            C128::new(0.0, 1.0 / sq2),
            C128::zero(),
            C128::zero(),
            C128::zero(),
            C128::new(1.0 / sq2, 0.0)
        ],
        [
            C128::zero(),
            C128::new(0.0, 1.0 / sq2),
            C128::zero(),
            C128::new(1.0 / sq2, 0.0),
            C128::zero()
        ],
        [
            C128::zero(),
            C128::zero(),
            C128::one(),
            C128::zero(),
            C128::zero()
        ],
        [
            C128::zero(),
            C128::new(0.0, 1.0 / sq2),
            C128::zero(),
            C128::new(-1.0 / sq2, 0.0),
            C128::zero()
        ],
        [
            C128::new(0.0, -1.0 / sq2),
            C128::zero(),
            C128::zero(),
            C128::zero(),
            C128::new(1.0 / sq2, 0.0)
        ],
    ];
    assert_eq!(r2c2, r2c2_ref);
}

#[test]
fn test_sh_conversion_cl2cart() {
    let sq2 = 2.0f64.sqrt();

    let umat00 = sh_cl2cart_mat(0, 0, CartOrder::lex(0), true, true);
    assert_eq!(umat00.shape(), &[1, 1]);
    assert_eq!(umat00[(0, 0)], C128::from(1.0));

    let umat11 = sh_cl2cart_mat(1, 1, CartOrder::lex(1), true, true);
    assert_eq!(umat11.shape(), &[3, 3]);
    let umat11_ref = array![
        [
            C128::new(1.0 / sq2, 0.0),
            C128::zero(),
            C128::new(-1.0 / sq2, 0.0)
        ],
        [
            C128::new(0.0, -1.0 / sq2),
            C128::zero(),
            C128::new(0.0, -1.0 / sq2)
        ],
        [C128::zero(), C128::one(), C128::zero()],
    ];
    approx::assert_relative_eq!(
        (&umat11 - &umat11_ref).map(|x| x.norm_sqr()).sum().sqrt(),
        0.0,
        epsilon = 1e-14,
        max_relative = 1e-14
    );

    let vmat11 = sh_cart2cl_mat(1, 1, CartOrder::lex(1), true, true);
    approx::assert_relative_eq!(
        (vmat11.dot(&umat11) - Array2::<C128>::eye(3))
            .map(|x| x.norm_sqr())
            .sum()
            .sqrt(),
        0.0,
        epsilon = 1e-14,
        max_relative = 1e-14
    );

    let umat21 = sh_cl2cart_mat(2, 1, CartOrder::lex(2), true, true);
    approx::assert_relative_eq!(
        umat21.map(|x| x.norm_sqr()).sum().sqrt(),
        0.0,
        epsilon = 1e-14,
        max_relative = 1e-14
    );

    let umat22 = sh_cl2cart_mat(2, 2, CartOrder::lex(2), true, true);
    let vmat22 = sh_cart2cl_mat(2, 2, CartOrder::lex(2), true, true);
    approx::assert_relative_eq!(
        (vmat22.dot(&umat22) - Array2::<C128>::eye(5))
            .map(|x| x.norm_sqr())
            .sum()
            .sqrt(),
        0.0,
        epsilon = 1e-14,
        max_relative = 1e-14
    );

    let umat20 = sh_cl2cart_mat(2, 0, CartOrder::lex(2), true, true);
    let vmat02 = sh_cart2cl_mat(0, 2, CartOrder::lex(2), true, true);
    approx::assert_relative_eq!(
        (vmat02.dot(&umat20) - Array2::<C128>::eye(1))
            .map(|x| x.norm_sqr())
            .sum()
            .sqrt(),
        0.0,
        epsilon = 1e-14,
        max_relative = 1e-14
    );
}

#[test]
fn test_sh_conversion_cart2cl() {
    let sq2 = 2.0f64.sqrt();
    let sq6 = 6.0f64.sqrt();

    let vmat00 = sh_cart2cl_mat(0, 0, CartOrder::lex(0), true, true);
    assert_eq!(vmat00.shape(), &[1, 1]);
    assert_eq!(vmat00[(0, 0)], C128::from(1.0));

    let vmat11 = sh_cart2cl_mat(1, 1, CartOrder::lex(1), true, true);
    assert_eq!(vmat11.shape(), &[3, 3]);
    let vmat11_ref = array![
        [
            C128::new(1.0 / sq2, 0.0),
            C128::new(0.0, 1.0 / sq2),
            C128::zero(),
        ],
        [C128::zero(), C128::zero(), C128::one()],
        [
            C128::new(-1.0 / sq2, 0.0),
            C128::new(0.0, 1.0 / sq2),
            C128::zero(),
        ],
    ];
    approx::assert_relative_eq!(
        (&vmat11 - &vmat11_ref).map(|x| x.norm_sqr()).sum().sqrt(),
        0.0,
        epsilon = 1e-14,
        max_relative = 1e-14
    );

    let vmat01 = sh_cart2cl_mat(0, 1, CartOrder::lex(1), true, true);
    assert_eq!(vmat01.shape(), &[1, 3]);
    approx::assert_relative_eq!(
        vmat01.map(|x| x.norm_sqr()).sum().sqrt(),
        0.0,
        epsilon = 1e-14,
        max_relative = 1e-14
    );

    let vmat22 = sh_cart2cl_mat(2, 2, CartOrder::lex(2), true, true);
    assert_eq!(vmat22.shape(), &[5, 6]);
    let vmat22_ref = array![
        [
            C128::new(1.0 / sq6, 0.0),
            C128::new(0.0, 1.0 / sq2),
            C128::zero(),
            C128::new(-1.0 / sq6, 0.0),
            C128::zero(),
            C128::zero(),
        ],
        [
            C128::zero(),
            C128::zero(),
            C128::new(1.0 / sq2, 0.0),
            C128::zero(),
            C128::new(0.0, 1.0 / sq2),
            C128::zero(),
        ],
        [
            C128::new(-1.0 / 3.0, 0.0),
            C128::zero(),
            C128::zero(),
            C128::new(-1.0 / 3.0, 0.0),
            C128::zero(),
            C128::new(2.0 / 3.0, 0.0),
        ],
        [
            C128::zero(),
            C128::zero(),
            C128::new(-1.0 / sq2, 0.0),
            C128::zero(),
            C128::new(0.0, 1.0 / sq2),
            C128::zero(),
        ],
        [
            C128::new(1.0 / sq6, 0.0),
            C128::new(0.0, -1.0 / sq2),
            C128::zero(),
            C128::new(-1.0 / sq6, 0.0),
            C128::zero(),
            C128::zero(),
        ],
    ];
    approx::assert_relative_eq!(
        (&vmat22 - &vmat22_ref).map(|x| x.norm_sqr()).sum().sqrt(),
        0.0,
        epsilon = 1e-14,
        max_relative = 1e-14
    );

    let vmat02 = sh_cart2cl_mat(0, 2, CartOrder::lex(2), true, true);
    assert_eq!(vmat02.shape(), &[1, 6]);

    let ntilde = norm_sph_gaussian(2, 1.0);
    let n = norm_cart_gaussian((2, 0, 0), 1.0);
    let temp = C128::new(
        1.0 / 3.0 * n / ntilde * (4.0 * std::f64::consts::PI).sqrt(),
        0.0,
    );
    let vmat02_ref = array![[temp, C128::zero(), C128::zero(), temp, C128::zero(), temp,]];
    approx::assert_relative_eq!(
        (&vmat02 - &vmat02_ref).map(|x| x.norm_sqr()).sum().sqrt(),
        0.0,
        epsilon = 1e-14,
        max_relative = 1e-14
    );
}

#[test]
fn test_sh_conversion_rl2cart() {
    let wmat00 = sh_rl2cart_mat(0, 0, CartOrder::lex(0), true, true);
    assert_eq!(wmat00.shape(), &[1, 1]);
    assert_eq!(wmat00[(0, 0)], 1.0);

    let wmat11 = sh_rl2cart_mat(1, 1, CartOrder::lex(1), true, true);
    assert_eq!(wmat11.shape(), &[3, 3]);
    let wmat11_ref = array![[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],];
    approx::assert_relative_eq!(
        (&wmat11 - &wmat11_ref).map(|x| x * x).sum().sqrt(),
        0.0,
        epsilon = 1e-14,
        max_relative = 1e-14
    );

    let xmat11 = sh_cart2rl_mat(1, 1, CartOrder::lex(1), true, true);
    approx::assert_relative_eq!(
        (xmat11.dot(&wmat11) - Array2::<f64>::eye(3))
            .map(|x| x * x)
            .sum()
            .sqrt(),
        0.0,
        epsilon = 1e-14,
        max_relative = 1e-14
    );

    let wmat22 = sh_rl2cart_mat(2, 2, CartOrder::lex(2), true, true);
    assert_eq!(wmat22.shape(), &[6, 5]);
    let wmat22_ref = array![
        [0.0, 0.0, -0.5, 0.0, 3.0f64.sqrt() / 2.0],
        [1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, -0.5, 0.0, -3.0f64.sqrt() / 2.0],
        [0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0],
    ];
    approx::assert_relative_eq!(
        (&wmat22 - &wmat22_ref).map(|x| x * x).sum().sqrt(),
        0.0,
        epsilon = 1e-14,
        max_relative = 1e-14
    );

    let wmat20 = sh_rl2cart_mat(2, 0, CartOrder::lex(2), true, true);
    assert_eq!(wmat20.shape(), &[6, 1]);
    let wmat20_ref = array![
        [1.0 / 5.0f64.sqrt()],
        [0.0],
        [0.0],
        [1.0 / 5.0f64.sqrt()],
        [0.0],
        [1.0 / 5.0f64.sqrt()],
    ];
    approx::assert_relative_eq!(
        (&wmat20 - &wmat20_ref).map(|x| x * x).sum().sqrt(),
        0.0,
        epsilon = 1e-14,
        max_relative = 1e-14
    );
}

#[test]
fn test_sh_conversion_cart2rl() {
    let sq3 = 3.0f64.sqrt();
    let sq5 = 5.0f64.sqrt();

    let xmat00 = sh_cart2rl_mat(0, 0, CartOrder::lex(0), true, true);
    assert_eq!(xmat00.shape(), &[1, 1]);
    assert_eq!(xmat00[(0, 0)], 1.0);

    let xmat11 = sh_cart2rl_mat(1, 1, CartOrder::lex(1), true, true);
    assert_eq!(xmat11.shape(), &[3, 3]);
    let xmat11_ref = array![[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0],];
    approx::assert_relative_eq!(
        (&xmat11 - &xmat11_ref).map(|x| x * x).sum().sqrt(),
        0.0,
        epsilon = 1e-14,
        max_relative = 1e-14
    );

    let xmat22 = sh_cart2rl_mat(2, 2, CartOrder::lex(2), true, true);
    assert_eq!(xmat22.shape(), &[5, 6]);
    let xmat22_ref = array![
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [-1.0 / 3.0, 0.0, 0.0, -1.0 / 3.0, 0.0, 2.0 / 3.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        [1.0 / sq3, 0.0, 0.0, -1.0 / sq3, 0.0, 0.0],
    ];
    approx::assert_relative_eq!(
        (&xmat22 - &xmat22_ref).map(|x| x * x).sum().sqrt(),
        0.0,
        epsilon = 1e-14,
        max_relative = 1e-14
    );

    let xmat02 = sh_cart2rl_mat(0, 2, CartOrder::lex(2), true, true);
    assert_eq!(xmat02.shape(), &[1, 6]);
    let xmat02_ref = array![[sq5 / 3.0, 0.0, 0.0, sq5 / 3.0, 0.0, sq5 / 3.0],];
    approx::assert_relative_eq!(
        (&xmat02 - &xmat02_ref).map(|x| x * x).sum().sqrt(),
        0.0,
        epsilon = 1e-14,
        max_relative = 1e-14
    );
}

#[test]
fn test_sh_conversion_r2cart() {
    let wmats1l = sh_r2cart(1, CartOrder::lex(1), true, true);
    assert_eq!(wmats1l.len(), 1);
    for (i, wmat1l) in wmats1l.iter().enumerate() {
        let l = 1u32.checked_sub(2 * (i as u32)).unwrap();
        approx::assert_relative_eq!(
            (wmat1l - &sh_rl2cart_mat(1, l, CartOrder::lex(1), true, true))
                .map(|x| x * x)
                .sum()
                .sqrt(),
            0.0,
            epsilon = 1e-14,
            max_relative = 1e-14
        );
    }

    let wmats2l = sh_r2cart(2, CartOrder::lex(2), true, true);
    assert_eq!(wmats2l.len(), 2);
    for (i, wmat2l) in wmats2l.iter().enumerate() {
        let l = 2u32.checked_sub(2 * (i as u32)).unwrap();
        approx::assert_relative_eq!(
            (wmat2l - &sh_rl2cart_mat(2, l, CartOrder::lex(2), true, true))
                .map(|x| x * x)
                .sum()
                .sqrt(),
            0.0,
            epsilon = 1e-14,
            max_relative = 1e-14
        );
    }

    let wmats3l = sh_r2cart(3, CartOrder::lex(3), true, true);
    assert_eq!(wmats3l.len(), 2);
    for (i, wmat3l) in wmats3l.iter().enumerate() {
        let l = 3u32.checked_sub(2 * (i as u32)).unwrap();
        approx::assert_relative_eq!(
            (wmat3l - &sh_rl2cart_mat(3, l, CartOrder::lex(3), true, true))
                .map(|x| x * x)
                .sum()
                .sqrt(),
            0.0,
            epsilon = 1e-14,
            max_relative = 1e-14
        );
    }

    let wmats4l = sh_r2cart(4, CartOrder::lex(4), true, true);
    assert_eq!(wmats4l.len(), 3);
    for (i, wmat4l) in wmats4l.iter().enumerate() {
        let l = 4u32.checked_sub(2 * (i as u32)).unwrap();
        approx::assert_relative_eq!(
            (wmat4l - &sh_rl2cart_mat(4, l, CartOrder::lex(4), true, true))
                .map(|x| x * x)
                .sum()
                .sqrt(),
            0.0,
            epsilon = 1e-14,
            max_relative = 1e-14
        );
    }

    let wmats5l = sh_r2cart(5, CartOrder::lex(5), true, true);
    assert_eq!(wmats5l.len(), 3);
    for (i, wmat5l) in wmats5l.iter().enumerate() {
        let l = 5u32.checked_sub(2 * (i as u32)).unwrap();
        approx::assert_relative_eq!(
            (wmat5l - &sh_rl2cart_mat(5, l, CartOrder::lex(5), true, true))
                .map(|x| x * x)
                .sum()
                .sqrt(),
            0.0,
            epsilon = 1e-14,
            max_relative = 1e-14
        );
    }
}

#[test]
fn test_sh_conversion_cart2r() {
    let xmatsl1 = sh_cart2r(1, CartOrder::lex(1), true, true);
    assert_eq!(xmatsl1.len(), 1);
    for (i, xmatl1) in xmatsl1.iter().enumerate() {
        let l = 1u32.checked_sub(2 * (i as u32)).unwrap();
        approx::assert_relative_eq!(
            (xmatl1 - &sh_cart2rl_mat(l, 1, CartOrder::lex(1), true, true))
                .map(|x| x * x)
                .sum()
                .sqrt(),
            0.0,
            epsilon = 1e-14,
            max_relative = 1e-14
        );
    }

    let xmatsl2 = sh_cart2r(2, CartOrder::lex(2), true, true);
    assert_eq!(xmatsl2.len(), 2);
    for (i, xmatl2) in xmatsl2.iter().enumerate() {
        let l = 2u32.checked_sub(2 * (i as u32)).unwrap();
        approx::assert_relative_eq!(
            (xmatl2 - &sh_cart2rl_mat(l, 2, CartOrder::lex(2), true, true))
                .map(|x| x * x)
                .sum()
                .sqrt(),
            0.0,
            epsilon = 1e-14,
            max_relative = 1e-14
        );
    }

    let xmatsl3 = sh_cart2r(3, CartOrder::lex(3), true, true);
    assert_eq!(xmatsl3.len(), 2);
    for (i, xmatl3) in xmatsl3.iter().enumerate() {
        let l = 3u32.checked_sub(2 * (i as u32)).unwrap();
        approx::assert_relative_eq!(
            (xmatl3 - &sh_cart2rl_mat(l, 3, CartOrder::lex(3), true, true))
                .map(|x| x * x)
                .sum()
                .sqrt(),
            0.0,
            epsilon = 1e-14,
            max_relative = 1e-14
        );
    }

    let xmatsl4 = sh_cart2r(4, CartOrder::lex(4), true, true);
    assert_eq!(xmatsl4.len(), 3);
    for (i, xmatl4) in xmatsl4.iter().enumerate() {
        let l = 4u32.checked_sub(2 * (i as u32)).unwrap();
        approx::assert_relative_eq!(
            (xmatl4 - &sh_cart2rl_mat(l, 4, CartOrder::lex(4), true, true))
                .map(|x| x * x)
                .sum()
                .sqrt(),
            0.0,
            epsilon = 1e-14,
            max_relative = 1e-14
        );
    }

    let xmatsl5 = sh_cart2r(5, CartOrder::lex(5), true, true);
    assert_eq!(xmatsl5.len(), 3);
    for (i, xmatl5) in xmatsl5.iter().enumerate() {
        let l = 5u32.checked_sub(2 * (i as u32)).unwrap();
        approx::assert_relative_eq!(
            (xmatl5 - &sh_cart2rl_mat(l, 5, CartOrder::lex(5), true, true))
                .map(|x| x * x)
                .sum()
                .sqrt(),
            0.0,
            epsilon = 1e-14,
            max_relative = 1e-14
        );
    }
}
