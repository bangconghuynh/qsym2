use nalgebra::Vector3;


/// Returns the rotation angle adjusted to be in the interval $(-\pi, +\pi]$.
///
/// # Arguments
///
/// * `rot_ang` - A rotation angle.
/// * `thresh` - A threshold for comparisons.
///
/// # Returns
///
/// The normalised rotation angle.
pub fn normalise_rotation_angle(rot_ang: f64, thresh: f64) -> f64 {
    let mut norm_rot_ang = rot_ang.clone();
    while norm_rot_ang > std::f64::consts::PI + thresh {
        norm_rot_ang -= 2.0 * std::f64::consts::PI;
    }
    while norm_rot_ang <= -std::f64::consts::PI + thresh {
        norm_rot_ang += 2.0 * std::f64::consts::PI;
    }
    norm_rot_ang
}


/// Returns the positive pole of a rotation axis.
///
/// The definition of positive poles can be found in S.L. Altmann, Rotations,
/// Quaternions, and Double Groups (Dover Publications, Inc., New York, 2005)
/// (Chapter 9).
///
/// # Arguments
///
/// * axis - An axis of rotation (proper or improper).
/// * thresh - Threshold for comparisons.
///
/// # Returns
///
/// The positive pole of `axis`.
pub fn get_positive_pole(axis: &Vector3<f64>, thresh: f64) -> Vector3<f64> {
    approx::assert_relative_eq!(axis.norm(), 1.0, epsilon = thresh, max_relative = thresh);
    let mut pole = axis.clone();
    if pole[2].abs() > thresh {
        pole *= pole[2].signum();
    } else if pole[0].abs() > thresh {
        pole *= pole[0].signum();
    } else {
        assert!(pole[1].abs() > thresh);
        pole *= pole[1].signum();
    }
    pole
}
