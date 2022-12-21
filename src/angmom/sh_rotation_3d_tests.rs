use approx;
use nalgebra::Vector3;
use ndarray::array;

use crate::angmom::sh_rotation_3d::{rlmat, rmat};

#[test]
fn test_sh_rotation_3d_rmat() {
    let sq3 = 3.0f64.sqrt();

    let rmat_pi_6_z = rmat(std::f64::consts::FRAC_PI_6, Vector3::new(0.0, 0.0, 1.0));
    let rmat_pi_6_z_ref = array![
        [sq3 / 2.0, 0.0, 0.5],
        [0.0, 1.0, 0.0],
        [-0.5, 0.0, sq3 / 2.0],
    ];
    approx::assert_relative_eq!(
        (rmat_pi_6_z - rmat_pi_6_z_ref).map(|x| x * x).sum().sqrt(),
        0.0,
        epsilon = 1e-14,
        max_relative = 1e-14
    );

    let rmat_pi_3_y = rmat(std::f64::consts::FRAC_PI_3, Vector3::new(0.0, 1.0, 0.0));
    let rmat_pi_3_y_ref = array![
        [1.0, 0.0, 0.0],
        [0.0, 0.5, -sq3 / 2.0],
        [0.0, sq3 / 2.0, 0.5],
    ];
    approx::assert_relative_eq!(
        (rmat_pi_3_y - rmat_pi_3_y_ref).map(|x| x * x).sum().sqrt(),
        0.0,
        epsilon = 1e-14,
        max_relative = 1e-14
    );

    let rmat_pi_2_x = rmat(std::f64::consts::FRAC_PI_2, Vector3::new(1.0, 0.0, 0.0));
    let rmat_pi_2_x_ref = array![[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0],];
    approx::assert_relative_eq!(
        (rmat_pi_2_x - rmat_pi_2_x_ref).map(|x| x * x).sum().sqrt(),
        0.0,
        epsilon = 1e-14,
        max_relative = 1e-14
    );
}

#[test]
fn test_sh_rotation_3d_rlmat() {
    let r = rmat(std::f64::consts::FRAC_PI_2, Vector3::new(0.0, 0.0, 1.0));
    let r2 = rlmat(2, &r, &r);
    let r2_ref = array![
        [-1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, -1.0]
    ];
    approx::assert_relative_eq!(
        (&r2 - &r2_ref).map(|x| x * x).sum().sqrt(),
        0.0,
        epsilon = 1e-14,
        max_relative = 1e-14
    );

    let r3 = rlmat(3, &r, &r2);
    let r3_ref = array![
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
        [0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ];
    approx::assert_relative_eq!(
        (&r3 - &r3_ref).map(|x| x * x).sum().sqrt(),
        0.0,
        epsilon = 1e-14,
        max_relative = 1e-14
    );

    let r = rmat(
        2.0 * std::f64::consts::FRAC_PI_3,
        Vector3::new(1.0, 1.0, 1.0),
    );
    let r2 = rlmat(2, &r, &r);
    let sq3 = 3.0f64.sqrt();
    let r2_ref = array![
        [0.0, 0.0, 0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, -0.5, 0.0, -sq3 / 2.0],
        [0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, sq3 / 2.0, 0.0, -0.5]
    ];
    approx::assert_relative_eq!(
        (&r2 - &r2_ref).map(|x| x * x).sum().sqrt(),
        0.0,
        epsilon = 1e-14,
        max_relative = 1e-14
    );
}
