use approx;
use ndarray::array;
use nalgebra::Vector3;

use crate::angmom::shrotation_3d::rmat;

#[test]
fn test_shrotation_3d_rmat() {
    let rmat_pi_6_z = rmat(std::f64::consts::FRAC_PI_6, Vector3::new(0.0, 0.0, 1.0));
    let sq3 = 3.0f64.sqrt();
    let rmat_pi_6_z_ref = array![
        [sq3 / 2.0, 0.0,       0.5],
        [      0.0, 1.0,       0.0],
        [     -0.5, 0.0, sq3 / 2.0],
    ];
    approx::assert_relative_eq!(
        (rmat_pi_6_z - rmat_pi_6_z_ref)
            .map(|x| x * x)
            .sum()
            .sqrt(),
        0.0,
        epsilon = 1e-14,
        max_relative = 1e-14
    );
}
