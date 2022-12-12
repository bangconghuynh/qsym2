use itertools::Itertools;
use num::Complex;
use num_traits::{One, Zero};

use crate::angmom::shconversion::{complexc, complexcinv, norm_cart_gaussian, norm_sph_gaussian};

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
            assert!((complexc((0, 0), (lx, ly, lz), true) - Complex::<f64>::one()).norm() < 1e-14);
        } else {
            let lcart = lx + ly + lz;
            if lcart.rem_euclid(2) != 0
                || (lx.rem_euclid(2) != 0 || ly.rem_euclid(2) != 0 || lz.rem_euclid(2) != 0)
            {
                assert!(
                    (complexc((0, 0), (lx, ly, lz), true) - Complex::<f64>::zero()).norm() < 1e-14
                );
            } else {
                assert!(
                    (complexc((0, 0), (lx, ly, lz), true) - Complex::<f64>::zero()).norm() >= 1e-14
                );
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
    assert!((complexc((1, 0), (0, 0, 1), true) - Complex::<f64>::one()).norm() < 1e-14);

    assert!(
        (complexc((1, 1), (1, 0, 0), true) - Complex::<f64>::new(-1.0 / 2.0f64.sqrt(), 0.0)).norm()
            < 1e-14
    );
    assert!(
        (complexc((1, 1), (0, 1, 0), true) - Complex::<f64>::new(0.0, -1.0 / 2.0f64.sqrt())).norm()
            < 1e-14
    );

    assert!(
        (complexc((1, -1), (1, 0, 0), true) - Complex::<f64>::new(1.0 / 2.0f64.sqrt(), 0.0)).norm()
            < 1e-14
    );
    assert!(
        (complexc((1, -1), (0, 1, 0), true) - Complex::<f64>::new(0.0, -1.0 / 2.0f64.sqrt()))
            .norm()
            < 1e-14
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
                assert!(
                    (complexc((1, m), (lx, ly, lz), true) - Complex::<f64>::zero()).norm() < 1e-14
                );
            }
        }
    }

    // =====
    // l = 2
    // =====
    // Specific values
    assert!((complexc((2, 0), (0, 0, 2), true) - Complex::<f64>::one()).norm() < 1e-14);
    assert!((complexc((2, 0), (2, 0, 0), true) - Complex::<f64>::new(-0.5, 0.0)).norm() < 1e-14);
    assert!((complexc((2, 0), (0, 2, 0), true) - Complex::<f64>::new(-0.5, 0.0)).norm() < 1e-14);

    assert!(
        (complexc((2, 1), (1, 0, 1), true) - Complex::<f64>::new(-1.0 / 2.0f64.sqrt(), 0.0)).norm()
            < 1e-14
    );
    assert!(
        (complexc((2, 1), (0, 1, 1), true) - Complex::<f64>::new(0.0, -1.0 / 2.0f64.sqrt())).norm()
            < 1e-14
    );

    assert!(
        (complexc((2, -1), (1, 0, 1), true) - Complex::<f64>::new(1.0 / 2.0f64.sqrt(), 0.0)).norm()
            < 1e-14
    );
    assert!(
        (complexc((2, -1), (0, 1, 1), true) - Complex::<f64>::new(0.0, -1.0 / 2.0f64.sqrt()))
            .norm()
            < 1e-14
    );

    assert!(
        (complexc((2, 2), (2, 0, 0), true) - Complex::<f64>::new((3.0f64 / 8.0f64).sqrt(), 0.0))
            .norm()
            < 1e-14
    );
    assert!(
        (complexc((2, 2), (0, 2, 0), true) - Complex::<f64>::new(-(3.0f64 / 8.0f64).sqrt(), 0.0))
            .norm()
            < 1e-14
    );
    assert!(
        (complexc((2, 2), (1, 1, 0), true) - Complex::<f64>::new(0.0, 1.0 / 2.0f64.sqrt())).norm()
            < 1e-14
    );

    assert!((complexc((2, -2), (0, 0, 0), true) - Complex::<f64>::zero()).norm() < 1e-14);
    assert!(
        (complexc((2, -2), (2, 0, 0), true) - Complex::<f64>::new((3.0f64 / 8.0f64).sqrt(), 0.0))
            .norm()
            < 1e-14
    );
    assert!(
        (complexc((2, -2), (0, 2, 0), true) - Complex::<f64>::new(-(3.0f64 / 8.0f64).sqrt(), 0.0))
            .norm()
            < 1e-14
    );
    assert!(
        (complexc((2, -2), (1, 1, 0), true) - Complex::<f64>::new(0.0, -1.0 / 2.0f64.sqrt()))
            .norm()
            < 1e-14
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
                assert!(
                    (complexc((2, m), (lx, ly, lz), true) - Complex::<f64>::zero()).norm() < 1e-14
                );
            }
        }
    }

    // =====
    // l = 3
    // =====
    // Specific values
    assert!((complexc((3, 0), (0, 0, 3), true) - Complex::<f64>::one()).norm() < 1e-14);
    assert!(
        (complexc((3, 0), (2, 0, 1), true)
            - Complex::<f64>::new(-3.0 / (2.0 * 5.0f64.sqrt()), 0.0))
        .norm()
            < 1e-14
    );
    assert!(
        (complexc((3, 0), (0, 2, 1), true)
            - Complex::<f64>::new(-3.0 / (2.0 * 5.0f64.sqrt()), 0.0))
        .norm()
            < 1e-14
    );

    assert!(
        (complexc((3, 1), (1, 0, 2), true) - Complex::<f64>::new(-(3.0f64 / 5.0f64).sqrt(), 0.0))
            .norm()
            < 1e-14
    );
    assert!(
        (complexc((3, 1), (0, 1, 2), true) - Complex::<f64>::new(0.0, -(3.0f64 / 5.0f64).sqrt()))
            .norm()
            < 1e-14
    );
    assert!(
        (complexc((3, 1), (3, 0, 0), true) - Complex::<f64>::new(3.0f64.sqrt() / 4.0, 0.0)).norm()
            < 1e-14
    );
    assert!(
        (complexc((3, 1), (0, 3, 0), true) - Complex::<f64>::new(0.0, 3.0f64.sqrt() / 4.0)).norm()
            < 1e-14
    );
    assert!(
        (complexc((3, 1), (1, 2, 0), true)
            - Complex::<f64>::new(3.0f64.sqrt() / (4.0 * 5.0f64.sqrt()), 0.0))
        .norm()
            < 1e-14
    );
    assert!(
        (complexc((3, 1), (2, 1, 0), true)
            - Complex::<f64>::new(0.0, 3.0f64.sqrt() / (4.0 * 5.0f64.sqrt())))
        .norm()
            < 1e-14
    );

    assert!(
        (complexc((3, -1), (1, 0, 2), true) - Complex::<f64>::new((3.0f64 / 5.0f64).sqrt(), 0.0))
            .norm()
            < 1e-14
    );
    assert!(
        (complexc((3, -1), (0, 1, 2), true) - Complex::<f64>::new(0.0, -(3.0f64 / 5.0f64).sqrt()))
            .norm()
            < 1e-14
    );
    assert!(
        (complexc((3, -1), (3, 0, 0), true) - Complex::<f64>::new(-3.0f64.sqrt() / 4.0, 0.0))
            .norm()
            < 1e-14
    );
    assert!(
        (complexc((3, -1), (0, 3, 0), true) - Complex::<f64>::new(0.0, 3.0f64.sqrt() / 4.0)).norm()
            < 1e-14
    );
    assert!(
        (complexc((3, -1), (1, 2, 0), true)
            - Complex::<f64>::new(-3.0f64.sqrt() / (4.0 * 5.0f64.sqrt()), 0.0))
        .norm()
            < 1e-14
    );
    assert!(
        (complexc((3, -1), (2, 1, 0), true)
            - Complex::<f64>::new(0.0, 3.0f64.sqrt() / (4.0 * 5.0f64.sqrt())))
        .norm()
            < 1e-14
    );

    assert!(
        (complexc((3, 2), (2, 0, 1), true) - Complex::<f64>::new((3.0f64 / 8.0f64).sqrt(), 0.0))
            .norm()
            < 1e-14
    );
    assert!(
        (complexc((3, 2), (0, 2, 1), true) - Complex::<f64>::new(-(3.0f64 / 8.0f64).sqrt(), 0.0))
            .norm()
            < 1e-14
    );
    assert!(
        (complexc((3, 2), (1, 1, 1), true) - Complex::<f64>::new(0.0, 1.0 / (2.0f64).sqrt()))
            .norm()
            < 1e-14
    );

    assert!(
        (complexc((3, -2), (2, 0, 1), true) - Complex::<f64>::new((3.0f64 / 8.0f64).sqrt(), 0.0))
            .norm()
            < 1e-14
    );
    assert!(
        (complexc((3, -2), (0, 2, 1), true) - Complex::<f64>::new(-(3.0f64 / 8.0f64).sqrt(), 0.0))
            .norm()
            < 1e-14
    );
    assert!(
        (complexc((3, -2), (1, 1, 1), true) - Complex::<f64>::new(0.0, -1.0 / (2.0f64).sqrt()))
            .norm()
            < 1e-14
    );

    assert!(
        (complexc((3, 3), (3, 0, 0), true) - Complex::<f64>::new(-5.0f64.sqrt() / 4.0, 0.0)).norm()
            < 1e-14
    );
    assert!(
        (complexc((3, 3), (0, 3, 0), true) - Complex::<f64>::new(0.0, 5.0f64.sqrt() / 4.0)).norm()
            < 1e-14
    );
    assert!(
        (complexc((3, 3), (1, 2, 0), true) - Complex::<f64>::new(3.0 / 4.0, 0.0)).norm() < 1e-14
    );
    assert!(
        (complexc((3, 3), (2, 1, 0), true) - Complex::<f64>::new(0.0, -3.0 / 4.0)).norm() < 1e-14
    );

    assert!(
        (complexc((3, -3), (3, 0, 0), true) - Complex::<f64>::new(5.0f64.sqrt() / 4.0, 0.0)).norm()
            < 1e-14
    );
    assert!(
        (complexc((3, -3), (0, 3, 0), true) - Complex::<f64>::new(0.0, 5.0f64.sqrt() / 4.0)).norm()
            < 1e-14
    );
    assert!(
        (complexc((3, -3), (1, 2, 0), true) - Complex::<f64>::new(-3.0 / 4.0, 0.0)).norm() < 1e-14
    );
    assert!(
        (complexc((3, -3), (2, 1, 0), true) - Complex::<f64>::new(0.0, -3.0 / 4.0)).norm() < 1e-14
    );

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
                assert!(
                    (complexc((3, m), (lx, ly, lz), true) - Complex::<f64>::zero()).norm() < 1e-14
                );
            }
        }
    }
}

#[test]
fn test_shconversion_complexcinv() {
    // =========
    // lcart = 0
    // =========
    assert!((complexcinv((0, 0, 0), (0, 0), true) - Complex::<f64>::one()).norm() < 1e-14);

    // =========
    // lcart = 1
    // =========
    assert!((complexcinv((0, 0, 1), (1, 0), true) - Complex::<f64>::one()).norm() < 1e-14);
    assert!((complexcinv((1, 0, 0), (0, 0), true) - Complex::<f64>::zero()).norm() < 1e-14);
    assert!(
        (complexcinv((1, 0, 0), (1, 1), true) - Complex::<f64>::new(-1.0 / 2.0f64.sqrt(), 0.0))
            .norm()
            < 1e-14
    );
    assert!(
        (complexcinv((1, 0, 0), (1, -1), true) - Complex::<f64>::new(1.0 / 2.0f64.sqrt(), 0.0))
            .norm()
            < 1e-14
    );
    assert!(
        (complexcinv((0, 1, 0), (1, 1), true) - Complex::<f64>::new(0.0, 1.0 / 2.0f64.sqrt()))
            .norm()
            < 1e-14
    );
    assert!(
        (complexcinv((0, 1, 0), (1, -1), true) - Complex::<f64>::new(0.0, 1.0 / 2.0f64.sqrt()))
            .norm()
            < 1e-14
    );

    // =========
    // lcart = 2
    // =========
    assert!((complexcinv((1, 1, 0), (1, -1), true) - Complex::<f64>::zero()).norm() < 1e-14);

    let ntilde_2 = norm_sph_gaussian(2, 1.0);
    let n_200 = norm_cart_gaussian((2, 0, 0), 1.0);
    assert!(
        (complexcinv((0, 0, 2), (2, 0), true) - Complex::<f64>::new(2.0 / 3.0, 0.0)).norm() < 1e-14
    );
    assert!(
        (complexcinv((0, 0, 2), (0, 0), true)
            - Complex::<f64>::new(
                1.0 / 3.0 * n_200 / ntilde_2 * (4.0 * std::f64::consts::PI).sqrt(),
                0.0
            ))
        .norm()
            < 1e-14
    );

    assert!(
        (complexcinv((0, 0, 2), (2, 0), true) - Complex::<f64>::new(2.0 / 3.0, 0.0)).norm() < 1e-14
    );
    assert!(
        (complexcinv((0, 0, 2), (0, 0), true)
            - Complex::<f64>::new(
                1.0 / 3.0 * n_200 / ntilde_2 * (4.0 * std::f64::consts::PI).sqrt(),
                0.0
            ))
        .norm()
            < 1e-14
    );

    assert!(
        (complexcinv((2, 0, 0), (2, 0), true) - Complex::<f64>::new(-1.0 / 3.0, 0.0)).norm()
            < 1e-14
    );
    assert!(
        (complexcinv((2, 0, 0), (0, 0), true)
            - Complex::<f64>::new(
                1.0 / 3.0 * n_200 / ntilde_2 * (4.0 * std::f64::consts::PI).sqrt(),
                0.0
            ))
        .norm()
            < 1e-14
    );
    assert!(
        (complexcinv((2, 0, 0), (2, 2), true) - Complex::<f64>::new(1.0 / 6.0f64.sqrt(), 0.0))
            .norm()
            < 1e-14
    );
    assert!(
        (complexcinv((2, 0, 0), (2, -2), true) - Complex::<f64>::new(1.0 / 6.0f64.sqrt(), 0.0))
            .norm()
            < 1e-14
    );

    assert!(
        (complexcinv((0, 2, 0), (0, 0), true)
            - Complex::<f64>::new(
                1.0 / 3.0 * n_200 / ntilde_2 * (4.0 * std::f64::consts::PI).sqrt(),
                0.0
            ))
        .norm()
            < 1e-14
    );
    assert!(
        (complexcinv((0, 2, 0), (2, 0), true) - Complex::<f64>::new(-1.0 / 3.0, 0.0)).norm()
            < 1e-14
    );
    assert!(
        (complexcinv((0, 2, 0), (2, 2), true) - Complex::<f64>::new(-1.0 / 6.0f64.sqrt(), 0.0))
            .norm()
            < 1e-14
    );
    assert!(
        (complexcinv((0, 2, 0), (2, -2), true) - Complex::<f64>::new(-1.0 / 6.0f64.sqrt(), 0.0))
            .norm()
            < 1e-14
    );
}
