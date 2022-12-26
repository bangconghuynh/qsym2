use approx;
use nalgebra::Vector3;
use ndarray::array;
use num::{Complex, Zero};

use crate::angmom::spinor_rotation_3d::{dmat_angleaxis, dmat_euler};

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

    let d4 = dmat_euler((0.304, 0.0, -0.304), false);
    let d4b = dmat_euler((0.0, 0.0, 0.0), false);
    approx::assert_relative_eq!(
        (d4 - d4b).map(|x| x.norm_sqr()).sum().sqrt(),
        0.0,
        epsilon = 1e-14,
        max_relative = 1e-14
    );

    let d5 = dmat_euler((0.128, 0.0, 0.0), false);
    let d5b = dmat_euler((0.0, 0.0, 0.128 + 2.0 * std::f64::consts::PI), false);
    approx::assert_relative_eq!(
        (d5 + d5b).map(|x| x.norm_sqr()).sum().sqrt(),
        0.0,
        epsilon = 1e-14,
        max_relative = 1e-14
    );
}

#[test]
fn test_spinor_rotation_3d_dmat_angleaxis() {
    let d1 = dmat_angleaxis(
        2.0 * std::f64::consts::PI,
        Vector3::new(0.3, 0.2, 0.8),
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
        2.0 * std::f64::consts::PI,
        Vector3::new(1.3, 0.4, 0.1),
        false,
    );
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

    let d3 = dmat_angleaxis(
        2.0 / 7.0 * std::f64::consts::PI,
        Vector3::new(0.0, 1.0, 0.0),
        false,
    );
    let d3b = dmat_euler((0.0, 2.0 / 7.0 * std::f64::consts::PI, 0.0), false);
    approx::assert_relative_eq!(
        (d3 - d3b).map(|x| x.norm_sqr()).sum().sqrt(),
        0.0,
        epsilon = 1e-14,
        max_relative = 1e-14
    );

    let d4 = dmat_angleaxis(
        2.0 / 5.0 * std::f64::consts::PI,
        Vector3::new(0.0, 0.0, 1.0),
        false,
    );
    let d4b = dmat_euler((0.0, 0.0, 2.0 / 5.0 * std::f64::consts::PI), false);
    let d4c = dmat_euler((2.0 / 5.0 * std::f64::consts::PI, 0.0, 0.0), false);
    approx::assert_relative_eq!(
        (&d4 - &d4b).map(|x| x.norm_sqr()).sum().sqrt(),
        0.0,
        epsilon = 1e-14,
        max_relative = 1e-14
    );
    approx::assert_relative_eq!(
        (&d4 - &d4c).map(|x| x.norm_sqr()).sum().sqrt(),
        0.0,
        epsilon = 1e-14,
        max_relative = 1e-14
    );
}
