use approx;
use nalgebra::{ComplexField, Vector3};
use ndarray::{Array2, array};
use num::{Complex, Zero};
use proptest::prelude::*;

use crate::angmom::sh_conversion::sh_r2c_mat;
use crate::angmom::sh_rotation_3d::{rlmat, rmat};
use crate::angmom::spinor_rotation_3d::{
    dmat_angleaxis, dmat_angleaxis_gen_double, dmat_angleaxis_gen_single, dmat_euler,
    dmat_euler_gen,
};
use crate::basis::ao::PureOrder;

type C128 = Complex<f64>;

#[test]
fn test_spinor_rotation_3d_dmat_euler() {
    let d1 = dmat_euler((0.0, 0.0, 2.0 * std::f64::consts::PI), false);
    let d1_ref = array![
        [C128::from(-1.0), C128::zero()],
        [C128::zero(), C128::from(-1.0)],
    ];
    approx::assert_relative_eq!(
        (d1 - d1_ref).map(|x| x.norm_sqr()).sum().sqrt(),
        0.0,
        epsilon = 1e-14,
        max_relative = 1e-14
    );

    let d2 = dmat_euler((0.0, 2.0 * std::f64::consts::PI, 0.0), false);
    let d2_ref = array![
        [C128::from(-1.0), C128::zero()],
        [C128::zero(), C128::from(-1.0)],
    ];
    approx::assert_relative_eq!(
        (d2 - d2_ref).map(|x| x.norm_sqr()).sum().sqrt(),
        0.0,
        epsilon = 1e-14,
        max_relative = 1e-14
    );

    let d3 = dmat_euler((2.0 * std::f64::consts::PI, 0.0, 0.0), false);
    let d3_ref = array![
        [C128::from(-1.0), C128::zero()],
        [C128::zero(), C128::from(-1.0)],
    ];
    approx::assert_relative_eq!(
        (d3 - d3_ref).map(|x| x.norm_sqr()).sum().sqrt(),
        0.0,
        epsilon = 1e-14,
        max_relative = 1e-14
    );
}

proptest! {
    #[test]
    fn test_spinor_rotation_3d_dmat_euler_arbitrary(
        angle in (-1000000.0..1000000.0f64)
    ) {
        let d1 = dmat_euler((angle, 0.0, -angle), false);
        let d1b = dmat_euler((0.0, 0.0, 0.0), false);
        approx::assert_relative_eq!(
            (d1 - d1b).map(|x| x.norm_sqr()).sum().sqrt(),
            0.0,
            epsilon = 1e-14 * angle.abs().max(1.0),
            max_relative = 1e-14 * angle.abs().max(1.0)
        );

        let d2 = dmat_euler((angle, 0.0, 0.0), false);
        let d2b = dmat_euler((0.0, 0.0, angle + 2.0 * std::f64::consts::PI), false);
        approx::assert_relative_eq!(
            (d2 + d2b).map(|x| x.norm_sqr()).sum().sqrt(),
            0.0,
            epsilon = 1e-14 * angle.abs().max(1.0),
            max_relative = 1e-14 * angle.abs().max(1.0)
        );
    }
}

proptest! {
    #[test]
    fn test_spinor_rotation_3d_dmat_angleaxis(
        nx in (-1000000.0..1000000.0f64),
        ny in (-1000000.0..1000000.0f64),
        nz in (-1000000.0..1000000.0f64),
        angle in (-1000000.0..1000000.0f64)
    ) {
        let mag = (nx.powi(2) + ny.powi(2) + nz.powi(2)).sqrt();
        prop_assume!(
            approx::relative_ne!(
                mag,
                0.0,
                epsilon = 1e-14,
                max_relative = 1e-14
            )
        );

        let d1 = dmat_angleaxis(
            2.0 * std::f64::consts::PI,
            Vector3::new(nx, ny, nz),
            false,
        );
        let d1_ref = array![
            [C128::from(-1.0), C128::zero()],
            [C128::zero(), C128::from(-1.0)],
        ];
        approx::assert_relative_eq!(
            (d1 - d1_ref).map(|x| x.norm_sqr()).sum().sqrt(),
            0.0,
            epsilon = 1e-14,
            max_relative = 1e-14
        );

        let d2 = dmat_angleaxis(
            angle,
            Vector3::new(0.0, 1.0, 0.0),
            false,
        );
        let d2b = dmat_euler((0.0, angle, 0.0), false);
        approx::assert_relative_eq!(
            (d2 - d2b).map(|x| x.norm_sqr()).sum().sqrt(),
            0.0,
            epsilon = 1e-14 * angle.abs().max(1.0),
            max_relative = 1e-14 * angle.abs().max(1.0)
        );

        let d3 = dmat_angleaxis(
            angle,
            Vector3::new(0.0, 0.0, 1.0),
            false,
        );
        let d3b = dmat_euler((0.0, 0.0, angle), false);
        let d3c = dmat_euler((angle, 0.0, 0.0), false);
        approx::assert_relative_eq!(
            (&d3 - &d3b).map(|x| x.norm_sqr()).sum().sqrt(),
            0.0,
            epsilon = 1e-14 * angle.abs().max(1.0),
            max_relative = 1e-14 * angle.abs().max(1.0)
        );
        approx::assert_relative_eq!(
            (&d3 - &d3c).map(|x| x.norm_sqr()).sum().sqrt(),
            0.0,
            epsilon = 1e-14 * angle.abs().max(1.0),
            max_relative = 1e-14 * angle.abs().max(1.0)
        );

        let d4 = dmat_angleaxis(
            angle,
            Vector3::new(nx, ny, nz),
            false,
        );
        let d4b = dmat_angleaxis(
            angle + 2.0 * std::f64::consts::PI,
            Vector3::new(nx, ny, nz),
            false,
        );
        let d4c = dmat_angleaxis(
            angle + 4.0 * std::f64::consts::PI,
            Vector3::new(nx, ny, nz),
            false,
        );
        approx::assert_relative_eq!(
            (&d4 + &d4b).map(|x| x.norm_sqr()).sum().sqrt(),
            0.0,
            epsilon = 1e-14 * angle.abs().max(1.0),
            max_relative = 1e-14 * angle.abs().max(1.0)
        );
        approx::assert_relative_eq!(
            (&d4 - &d4c).map(|x| x.norm_sqr()).sum().sqrt(),
            0.0,
            epsilon = 1e-14 * angle.abs().max(1.0),
            max_relative = 1e-14 * angle.abs().max(1.0)
        );
    }
}

proptest! {
    #[test]
    fn test_spinor_rotation_3d_dmat_euler_gen(
        alpha in (-1000000.0..1000000.0f64),
        beta in (-1000000.0..1000000.0f64),
        gamma in (-1000000.0..1000000.0f64),
    ) {
        let d1 = dmat_euler((alpha, beta, gamma), false);
        let d1gen = dmat_euler_gen(1, (alpha, beta, gamma), false);
        approx::assert_relative_eq!(
            (&d1 - &d1gen).map(|x| x.norm_sqr()).sum().sqrt(),
            0.0,
            epsilon = 1e-14,
            max_relative = 1e-14
        );
    }
}

proptest! {
    #[test]
    fn test_spinor_rotation_3d_dmat_angleaxis_gen_double(
        nx in (-1000000.0..1000000.0f64),
        ny in (-1000000.0..1000000.0f64),
        nz in (-1000000.0..1000000.0f64),
        angle in (-1000000.0..1000000.0f64)
    ) {
        let mag = (nx.powi(2) + ny.powi(2) + nz.powi(2)).sqrt();
        prop_assume!(
            approx::relative_ne!(
                mag,
                0.0,
                epsilon = 1e-14,
                max_relative = 1e-14
            )
        );

        // j = 0
        let d1a = dmat_angleaxis_gen_double(0, angle, Vector3::new(nx, ny, nz), false);
        let d1b = dmat_angleaxis_gen_double(
            0, angle + 2.0 * std::f64::consts::PI, Vector3::new(nx, ny, nz), false
        );
        approx::assert_relative_eq!(
            (&d1a - &d1b).map(|x| x.norm_sqr()).sum().sqrt(),
            0.0,
            epsilon = 1e-14 * angle.abs().max(1.0),
            max_relative = 1e-14 * angle.abs().max(1.0)
        );

        // j = 1/2
        let d2a = dmat_angleaxis_gen_double(1, angle, Vector3::new(nx, ny, nz), false);
        let d2b = dmat_angleaxis_gen_double(
            1, angle + 2.0 * std::f64::consts::PI, Vector3::new(nx, ny, nz), false
        );
        let d2c = dmat_angleaxis_gen_double(
            1, angle + 4.0 * std::f64::consts::PI, Vector3::new(nx, ny, nz), false
        );
        approx::assert_relative_eq!(
            (&d2a + &d2b).map(|x| x.norm_sqr()).sum().sqrt(),
            0.0,
            epsilon = 1e-14 * angle.abs().max(1.0),
            max_relative = 1e-14 * angle.abs().max(1.0)
        );
        approx::assert_relative_eq!(
            (&d2a - &d2c).map(|x| x.norm_sqr()).sum().sqrt(),
            0.0,
            epsilon = 1e-14 * angle.abs().max(1.0),
            max_relative = 1e-14 * angle.abs().max(1.0)
        );

        // j = 1
        let d3a = dmat_angleaxis_gen_double(2, angle, Vector3::new(nx, ny, nz), false);
        let d3b = dmat_angleaxis_gen_double(
            2, angle + 2.0 * std::f64::consts::PI, Vector3::new(nx, ny, nz), false
        );
        approx::assert_relative_eq!(
            (&d3a - &d3b).map(|x| x.norm_sqr()).sum().sqrt(),
            0.0,
            epsilon = 1e-14 * angle.abs().max(1.0),
            max_relative = 1e-14 * angle.abs().max(1.0)
        );
    }
}

proptest! {
    #[test]
    fn test_spinor_rotation_3d_dmat_angleaxis_gen_double_arbitrary(
        twoj in (0..47u32),
        nx in (-1000000.0..1000000.0f64),
        ny in (-1000000.0..1000000.0f64),
        nz in (-1000000.0..1000000.0f64),
        angle in (-10000.0..10000.0f64)
    ) {
        // proptest begins to fail for twoj from 47 and up.
        let mag = (nx.powi(2) + ny.powi(2) + nz.powi(2)).sqrt();
        prop_assume!(
            approx::relative_ne!(
                mag,
                0.0,
                epsilon = 1e-14,
                max_relative = 1e-14
            )
        );
        let da = dmat_angleaxis_gen_double(twoj, angle, Vector3::new(nx, ny, nz), false);
        let db = dmat_angleaxis_gen_double(
            twoj, angle + 2.0 * std::f64::consts::PI, Vector3::new(nx, ny, nz), false
        );
        if twoj % 2 == 0 {
            approx::assert_relative_eq!(
                (da - db).map(|x| x.norm_sqr()).sum().sqrt(),
                0.0,
                epsilon = 1e-11 * angle.abs().max(1.0) * twoj as f64,
                max_relative = 1e-11 * angle.abs().max(1.0) * twoj as f64
            );
        } else {
            approx::assert_relative_eq!(
                (da + db).map(|x| x.norm_sqr()).sum().sqrt(),
                0.0,
                epsilon = 1e-11 * angle.abs().max(1.0) * twoj as f64,
                max_relative = 1e-11 * angle.abs().max(1.0) * twoj as f64
            );
        }
    }
}

proptest! {
    #[test]
    fn test_spinor_rotation_3d_dmat_angleaxis_gen_single(
        nx in (-1000000.0..1000000.0f64),
        ny in (-1000000.0..1000000.0f64),
        nz in (-1000000.0..1000000.0f64),
        angle in (-1000000.0..1000000.0f64)
    ) {
        let mag = (nx.powi(2) + ny.powi(2) + nz.powi(2)).sqrt();
        prop_assume!(
            approx::relative_ne!(
                mag,
                0.0,
                epsilon = 1e-14,
                max_relative = 1e-14
            )
        );

        // j = 0
        let d1a = dmat_angleaxis_gen_single(0, angle, Vector3::new(nx, ny, nz), false);
        let d1b = dmat_angleaxis_gen_single(
            0, angle + 2.0 * std::f64::consts::PI, Vector3::new(nx, ny, nz), false
        );
        approx::assert_relative_eq!(
            (&d1a - &d1b).map(|x| x.norm_sqr()).sum().sqrt(),
            0.0,
            epsilon = 1e-14 * angle.abs().max(1.0),
            max_relative = 1e-14 * angle.abs().max(1.0)
        );

        // j = 1/2
        let d2a = dmat_angleaxis_gen_single(1, angle, Vector3::new(nx, ny, nz), false);
        let d2b = dmat_angleaxis_gen_single(
            1, angle + 2.0 * std::f64::consts::PI, Vector3::new(nx, ny, nz), false
        );
        let d2c = dmat_angleaxis_gen_single(
            1, angle + 4.0 * std::f64::consts::PI, Vector3::new(nx, ny, nz), false
        );
        approx::assert_relative_eq!(
            (&d2a - &d2b).map(|x| x.norm_sqr()).sum().sqrt(),
            0.0,
            epsilon = 1e-14 * angle.abs().max(1.0),
            max_relative = 1e-14 * angle.abs().max(1.0)
        );
        approx::assert_relative_eq!(
            (&d2a - &d2c).map(|x| x.norm_sqr()).sum().sqrt(),
            0.0,
            epsilon = 1e-14 * angle.abs().max(1.0),
            max_relative = 1e-14 * angle.abs().max(1.0)
        );

        // j = 1
        let d3a = dmat_angleaxis_gen_single(2, angle, Vector3::new(nx, ny, nz), false);
        let d3b = dmat_angleaxis_gen_single(
            2, angle + 2.0 * std::f64::consts::PI, Vector3::new(nx, ny, nz), false
        );
        approx::assert_relative_eq!(
            (&d3a - &d3b).map(|x| x.norm_sqr()).sum().sqrt(),
            0.0,
            epsilon = 1e-14 * angle.abs().max(1.0),
            max_relative = 1e-14 * angle.abs().max(1.0)
        );

        // j = 3/2
        let d4a = dmat_angleaxis_gen_single(3, angle, Vector3::new(nx, ny, nz), false);
        let d4b = dmat_angleaxis_gen_single(
            3, angle + 2.0 * std::f64::consts::PI, Vector3::new(nx, ny, nz), false
        );
        let d4c = dmat_angleaxis_gen_single(
            3, angle + 4.0 * std::f64::consts::PI, Vector3::new(nx, ny, nz), false
        );
        approx::assert_relative_eq!(
            (&d4a - &d4b).map(|x| x.norm_sqr()).sum().sqrt(),
            0.0,
            epsilon = 1e-14 * angle.abs().max(1.0),
            max_relative = 1e-14 * angle.abs().max(1.0)
        );
        approx::assert_relative_eq!(
            (&d4a - &d4c).map(|x| x.norm_sqr()).sum().sqrt(),
            0.0,
            epsilon = 1e-14 * angle.abs().max(1.0),
            max_relative = 1e-14 * angle.abs().max(1.0)
        );

    }
}

proptest! {
    #[test]
    fn test_spinor_rotation_3d_dmat_angleaxis_gen_single_arbitrary(
        twoj in (0..47u32),
        nx in (-1000000.0..1000000.0f64),
        ny in (-1000000.0..1000000.0f64),
        nz in (-1000000.0..1000000.0f64),
        angle in (-10000.0..10000.0f64)
    ) {
        // proptest begins to fail for twoj from 47 and up.
        let mag = (nx.powi(2) + ny.powi(2) + nz.powi(2)).sqrt();
        prop_assume!(
            approx::relative_ne!(
                mag,
                0.0,
                epsilon = 1e-14,
                max_relative = 1e-14
            )
        );
        let da = dmat_angleaxis_gen_single(twoj, angle, Vector3::new(nx, ny, nz), false);
        let db = dmat_angleaxis_gen_single(
            twoj, angle + 2.0 * std::f64::consts::PI, Vector3::new(nx, ny, nz), false
        );
        approx::assert_relative_eq!(
            (da - db).map(|x| x.norm_sqr()).sum().sqrt(),
            0.0,
            epsilon = 1e-11 * angle.abs().max(1.0) * twoj as f64,
            max_relative = 1e-11 * angle.abs().max(1.0) * twoj as f64
        );
    }
}

proptest! {
    #[test]
    fn test_spinor_rotation_3d_dmat_angleaxis_gen_single_compare_with_rlmat(
        nx in (-1000000.0..1000000.0f64),
        ny in (-1000000.0..1000000.0f64),
        nz in (-1000000.0..1000000.0f64),
        angle in (-10000.0..10000.0f64)
    ) {
        let mag = (nx.powi(2) + ny.powi(2) + nz.powi(2)).sqrt();
        prop_assume!(
            approx::relative_ne!(
                mag,
                0.0,
                epsilon = 1e-14,
                max_relative = 1e-14
            )
        );

        let axis = Vector3::new(nx, ny, nz).normalize();
        let r = rmat(angle, axis);

        let mut rls_r = vec![Array2::eye(1), r.clone()];

        for l in 2..=23 {
            let rl_r = rlmat(l, &r, &rls_r[l as usize -1]);
            let rl = rl_r.mapv(Complex::from);
            rls_r.push(rl_r);
            let dmatl_c = dmat_angleaxis_gen_single(2*l, angle, axis, true);
            let t = sh_r2c_mat(l, true, &PureOrder::increasingm(l));
            let dmatl_r = t.t().mapv(|v| v.conj()).dot(&dmatl_c).dot(&t);
            approx::assert_relative_eq!(
                (&rl - &dmatl_r).map(|x| x.abs().powi(2)).sum().sqrt(),
                0.0,
                epsilon = 1e-11 * angle.abs().max(1.0) * (2 * l) as f64,
                max_relative = 1e-11 * angle.abs().max(1.0) * (2 * l) as f64
            );
        }
    }
}

/// Returns the $`j = 1/2`$ matrix of the $`\mathsf{SU}(2)`$ element
/// $`U(\phi\hat{\mathbf{n}}) = \cos\frac{\phi}{2} - i\sin\frac{\phi}{2}\,\hat{\mathbf{n}}\cdot\boldsymbol{\sigma}`$
/// in increasing-$`m`$ order, without any normalisation of $`\phi`$.
fn su2_closed_form(angle: f64, axis: Vector3<f64>) -> Array2<C128> {
    let n = axis.normalize();
    let i = C128::i();
    let (s, c) = (0.5 * angle).sin_cos();
    array![
        [c + i * n.z * s, (n.y - i * n.x) * s],
        [-(n.y + i * n.x) * s, c - i * n.z * s],
    ]
}

/// Returns the Frobenius distance between two matrices.
fn frobenius_distance(a: &Array2<C128>, b: &Array2<C128>) -> f64 {
    (a - b).map(|x| x.norm_sqr()).sum().sqrt()
}

#[test]
fn test_spinor_rotation_3d_dmat_angleaxis_gen_single_c2_icosahedral_axis() {
    // Regression test: the spin-1/2 matrix of a binary rotation used to have a sign decided by
    // rounding, because the homotopy class was chosen from the sign of a scalar part that is
    // exactly zero for binary rotations. This is a two-fold rotation axis of the icosahedral group
    // in the Point Group Atlas (https://valbert4.github.io/point-group-atlas) frame, together with
    // two axes moved by 1e-15.
    let phi = (1.0 + 5f64.sqrt()) / 2.0;
    // C2 axis (element 47) of https://valbert4.github.io/point-group-atlas/rotation/I/elements/.
    let pole = Vector3::new(-0.5, 0.5 / phi, 0.5 * phi);
    for n in [pole, -pole] {
        for d in [
            Vector3::zeros(),
            Vector3::new(0.0, 1e-15, 0.0),
            Vector3::new(0.0, 0.0, 1e-15),
        ] {
            for angle in [
                std::f64::consts::PI,
                2.0 * std::f64::consts::PI / 5.0,
                4.0 * std::f64::consts::PI / 5.0,
                2.0 * std::f64::consts::PI / 3.0,
            ] {
                let d_half = dmat_angleaxis_gen_single(1, angle, n + d, true);
                let u = su2_closed_form(angle, n + d);
                approx::assert_relative_eq!(
                    frobenius_distance(&d_half, &u),
                    0.0,
                    epsilon = 1e-13,
                    max_relative = 1e-13
                );
            }
        }
    }
}

#[test]
fn test_spinor_rotation_3d_dmat_angleaxis_gen_single_c2_cartesian_and_diagonal_axes() {
    let s2 = std::f64::consts::FRAC_1_SQRT_2;
    let s3 = 1.0 / 3f64.sqrt();
    let axes = [
        Vector3::x(),
        Vector3::y(),
        Vector3::z(),
        -Vector3::x(),
        -Vector3::y(),
        -Vector3::z(),
        Vector3::new(s2, s2, 0.0),
        Vector3::new(s2, -s2, 0.0),
        Vector3::new(s2, 0.0, s2),
        Vector3::new(-s2, 0.0, s2),
        Vector3::new(0.0, s2, s2),
        Vector3::new(0.0, s2, -s2),
        Vector3::new(s3, s3, s3),
        Vector3::new(s3, -s3, s3),
        Vector3::new(-s3, -s3, s3),
        Vector3::new(s3, s3, -s3),
    ];
    for axis in axes {
        let u = su2_closed_form(std::f64::consts::PI, axis);
        let d_half = dmat_angleaxis_gen_single(1, std::f64::consts::PI, axis, true);
        approx::assert_relative_eq!(
            frobenius_distance(&d_half, &u),
            0.0,
            epsilon = 1e-13,
            max_relative = 1e-13
        );
        // A binary rotation by -π is the same rotation in homotopy class 0.
        let d_half_neg = dmat_angleaxis_gen_single(1, -std::f64::consts::PI, axis, true);
        approx::assert_relative_eq!(
            frobenius_distance(&d_half_neg, &u),
            0.0,
            epsilon = 1e-13,
            max_relative = 1e-13
        );
    }
}

proptest! {
    #[test]
    fn test_spinor_rotation_3d_dmat_angleaxis_gen_single_matches_su2_closed_form(
        nx in (-1.0..1.0f64),
        ny in (-1.0..1.0f64),
        nz in (-1.0..1.0f64),
        frac in (-0.5..0.5f64),
    ) {
        let axis = Vector3::new(nx, ny, nz);
        prop_assume!(axis.norm() > 1e-3);
        let angle = 2.0 * std::f64::consts::PI * frac;
        let d_half = dmat_angleaxis_gen_single(1, angle, axis, true);
        let u = su2_closed_form(angle, axis);
        approx::assert_relative_eq!(
            frobenius_distance(&d_half, &u),
            0.0,
            epsilon = 1e-12,
            max_relative = 1e-12
        );
    }
}

proptest! {
    #[test]
    fn test_spinor_rotation_3d_dmat_angleaxis_gen_single_c2_closed_form(
        nx in (-1.0..1.0f64),
        ny in (-1.0..1.0f64),
        nz in (-1.0..1.0f64),
    ) {
        let axis = Vector3::new(nx, ny, nz);
        prop_assume!(axis.norm() > 1e-3);
        let d_half = dmat_angleaxis_gen_single(1, std::f64::consts::PI, axis, true);
        let u = su2_closed_form(std::f64::consts::PI, axis);
        approx::assert_relative_eq!(
            frobenius_distance(&d_half, &u),
            0.0,
            epsilon = 1e-12,
            max_relative = 1e-12
        );
    }
}

proptest! {
    #[test]
    fn test_spinor_rotation_3d_dmat_angleaxis_gen_single_c2_homomorphism(
        nx in (-1.0..1.0f64),
        ny in (-1.0..1.0f64),
        nz in (-1.0..1.0f64),
        mx in (-1.0..1.0f64),
        my in (-1.0..1.0f64),
        mz in (-1.0..1.0f64),
    ) {
        // For perpendicular unit vectors n and m, U(πn) U(πm) = U(π n×m) exactly in SU(2), so the
        // same must hold for every D^(j).
        let n = Vector3::new(nx, ny, nz);
        prop_assume!(n.norm() > 1e-3);
        let n = n.normalize();
        let m = Vector3::new(mx, my, mz);
        let m = m - n * n.dot(&m);
        prop_assume!(m.norm() > 1e-3);
        let m = m.normalize();
        let nxm = n.cross(&m);
        for twoj in 1..=6u32 {
            let dn = dmat_angleaxis_gen_single(twoj, std::f64::consts::PI, n, true);
            let dm = dmat_angleaxis_gen_single(twoj, std::f64::consts::PI, m, true);
            let dnm = dmat_angleaxis_gen_single(twoj, std::f64::consts::PI, nxm, true);
            approx::assert_relative_eq!(
                frobenius_distance(&dn.dot(&dm), &dnm),
                0.0,
                epsilon = 1e-11,
                max_relative = 1e-11
            );
        }
    }
}
