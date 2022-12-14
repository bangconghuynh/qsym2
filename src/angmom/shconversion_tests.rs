use approx;
use itertools::Itertools;
use ndarray::array;
use num::Complex;
use num_traits::{One, Zero};

use crate::angmom::shconversion::{
    complexc, complexcinv, norm_cart_gaussian, norm_sph_gaussian, sh_c2r_mat, sh_r2c_mat
};

type C128 = Complex<f64>;

#[test]
fn test_shconversion_complexc() {
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
fn test_shconversion_complexcinv() {
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
fn test_shconversion_sh_c2r() {
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
        [C128::zero(), C128::zero(), C128::one(), C128::zero(), C128::zero()],
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
fn test_shconversion_sh_r2c() {
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
        [C128::zero(), C128::zero(), C128::one(), C128::zero(), C128::zero()],
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
