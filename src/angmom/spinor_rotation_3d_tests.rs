use approx;
use nalgebra::Vector3;
use ndarray::array;
use num::{Complex, Zero};
use proptest::prelude::*;

use crate::angmom::spinor_rotation_3d::{
    dmat_angleaxis, dmat_angleaxis_gen, dmat_euler, dmat_euler_gen,
};

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
            epsilon = 1e-14,
            max_relative = 1e-14
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
            epsilon = 1e-14,
            max_relative = 1e-14
        );
        approx::assert_relative_eq!(
            (&d3 - &d3c).map(|x| x.norm_sqr()).sum().sqrt(),
            0.0,
            epsilon = 1e-14,
            max_relative = 1e-14
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
    fn test_spinor_rotation_3d_dmat_angleaxis_gen(
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
        let d1a = dmat_angleaxis_gen(0, angle, Vector3::new(nx, ny, nz), false);
        let d1b = dmat_angleaxis_gen(
            0, angle + 2.0 * std::f64::consts::PI, Vector3::new(nx, ny, nz), false
        );
        approx::assert_relative_eq!(
            (&d1a - &d1b).map(|x| x.norm_sqr()).sum().sqrt(),
            0.0,
            epsilon = 1e-14 * angle.abs().max(1.0),
            max_relative = 1e-14 * angle.abs().max(1.0)
        );

        // j = 1/2
        let d2a = dmat_angleaxis_gen(1, angle, Vector3::new(nx, ny, nz), false);
        let d2b = dmat_angleaxis_gen(
            1, angle + 2.0 * std::f64::consts::PI, Vector3::new(nx, ny, nz), false
        );
        let d2c = dmat_angleaxis_gen(
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
        let d3a = dmat_angleaxis_gen(2, angle, Vector3::new(nx, ny, nz), false);
        let d3b = dmat_angleaxis_gen(
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
    fn test_spinor_rotation_3d_dmat_angleaxis_gen_arbitrary(
        twoj in (0..47u32),
        nx in (-1000000.0..1000000.0f64),
        ny in (-1000000.0..1000000.0f64),
        nz in (-1000000.0..1000000.0f64),
        angle in (-1000000.0..1000000.0f64)
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
        let da = dmat_angleaxis_gen(twoj, angle, Vector3::new(nx, ny, nz), false);
        let db = dmat_angleaxis_gen(
            twoj, angle + 2.0 * std::f64::consts::PI, Vector3::new(nx, ny, nz), false
        );
        if twoj % 2 == 0 {
            approx::assert_relative_eq!(
                (da - db).map(|x| x.norm_sqr()).sum().sqrt(),
                0.0,
                epsilon = 1e-12 * angle.abs().max(1.0),
                max_relative = 1e-14 * angle.abs().max(1.0)
            );
        } else {
            approx::assert_relative_eq!(
                (da + db).map(|x| x.norm_sqr()).sum().sqrt(),
                0.0,
                epsilon = 1e-12 * angle.abs().max(1.0),
                max_relative = 1e-14 * angle.abs().max(1.0)
            );
        }
    }
}
