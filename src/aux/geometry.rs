use crate::symmetry::symmetry_element::SymmetryElementKind;
use nalgebra::{ClosedMul, Matrix3, Rotation3, Scalar, Transform3, UnitVector3, Vector3};

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

/// Computes the outer product between two three-dimensional vectors.
///
/// # Arguments
///
/// * vec1 - The first vector, $\boldsymbol{v}_1$.
/// * vec2 - The second vector, $\boldsymbol{v}_2$.
///
/// # Returns
///
/// The outer product $\boldsymbol{v}_1 \otimes \boldsymbol{v}_2$.
fn outer<T: Scalar + ClosedMul + Copy>(vec1: &Vector3<T>, vec2: &Vector3<T>) -> Matrix3<T> {
    let outer_product_iter: Vec<T> = vec2
        .iter()
        .map(|&item_x| vec1.iter().map(move |&item_y| item_x * item_y))
        .flatten()
        .collect();
    Matrix3::from_iterator(outer_product_iter)
}

/// Returns a $3 \times 3$ rotation matrix in $\mathbb{R}^3$ corresponding to a
/// rotation through `angle` about `axis` raised to the power `power`.
///
/// # Arguments
///
/// * angle - The angle of rotation.
/// * axis - The axis of rotation.
/// * power - The power of rotation.
///
/// # Returns
///
/// The rotation matrix.
pub fn proper_rotation_matrix(angle: f64, axis: &Vector3<f64>, power: i8) -> Matrix3<f64> {
    let normalised_axis = UnitVector3::new_normalize(*axis);
    Rotation3::from_axis_angle(&normalised_axis, (power as f64) * angle).into_inner()
}

/// Returns a $3 \times 3$ transformation matrix in $\mathbb{R}^3$ corresponding
/// to an improper rotation through `angle` about `axis` raised to the power
/// `power`.
///
/// # Arguments
///
/// * angle - The angle of rotation.
/// * axis - The axis of rotation.
/// * power - The power of transformation.
/// * kind - The convention in which the improper rotation is defined.
///
/// # Returns
///
/// The transformation matrix.
pub fn improper_rotation_matrix(
    angle: f64,
    axis: &Vector3<f64>,
    power: i8,
    kind: SymmetryElementKind,
) -> Matrix3<f64> {
    let rotmat = proper_rotation_matrix(angle, axis, power);
    let normalised_axis = UnitVector3::new_normalize(*axis);
    match kind {
        SymmetryElementKind::ImproperMirrorPlane => {
            let refmat = Matrix3::identity()
                - 2.0 * ((power % 2) as f64) * outer(&normalised_axis, &normalised_axis);
            refmat * rotmat
        }
        SymmetryElementKind::ImproperInversionCentre => {
            if power % 2 == 1 {
                -rotmat
            } else {
                rotmat
            }
        }
        _ => panic!("Only improper kinds are allowed."),
    }
}

pub trait Transform {
    /// Transforms in-place the coordinates about the origin by a given
    /// transformation.
    ///
    /// # Arguments
    ///
    /// * mat - A three-dimensional transformation matrix.
    fn transform_mut(self: &mut Self, mat: &Matrix3<f64>);

    /// Rotates in-place the coordinates through `angle` about `axis`.
    ///
    /// # Arguments
    ///
    /// * angle - The angle of rotation.
    /// * axis - The axis of rotation.
    fn rotate_mut(self: &mut Self, angle: f64, axis: &Vector3<f64>);

    /// Improper-rotates in-place the coordinates through `angle` about `axis`.
    ///
    /// # Arguments
    ///
    /// * angle - The angle of rotation.
    /// * axis - The axis of rotation.
    /// * kind - The convention in which the improper rotation is defined.
    fn improper_rotate_mut(
        self: &mut Self,
        angle: f64,
        axis: &Vector3<f64>,
        kind: SymmetryElementKind,
    );

    /// Translates in-place the coordinates by a specified translation vector in
    /// three dimensions.
    ///
    /// # Arguments
    ///
    /// * tvec - The translation vector.
    fn translate_mut(self: &mut Self, tvec: &Vector3<f64>);

    /// Recentres in-place to put the centre of mass at the origin.
    fn recentre_mut(self: &mut Self);

    /// Clones and transforms the coordinates about the origin by a given
    /// transformation.
    ///
    /// # Arguments
    ///
    /// * mat - A three-dimensional transformation matrix.
    ///
    /// # Returns
    ///
    /// A transformed copy.
    fn transform(self: &Self, mat: &Matrix3<f64>) -> Self;

    /// Clones and rotates the coordinates through `angle` about `axis`.
    ///
    /// # Arguments
    ///
    /// * angle - The angle of rotation.
    /// * axis - The axis of rotation.
    ///
    /// # Returns
    ///
    /// A rotated copy.
    fn rotate(self: &Self, angle: f64, axis: &Vector3<f64>) -> Self;

    /// Clones and improper-rotates the coordinates through `angle` about `axis`.
    ///
    /// # Arguments
    ///
    /// * angle - The angle of rotation.
    /// * axis - The axis of rotation.
    /// * kind - The convention in which the improper rotation is defined.
    ///
    /// # Returns
    ///
    /// An improper-rotated copy.
    fn improper_rotate(
        self: &Self,
        angle: f64,
        axis: &Vector3<f64>,
        kind: SymmetryElementKind,
    ) -> Self;

    /// Clones and translates in-place the coordinates by a specified
    /// translation in three dimensions.
    ///
    /// # Arguments
    ///
    /// * tvec - The translation vector.
    ///
    /// # Returns
    ///
    /// A translated copy.
    fn translate(self: &Self, tvec: &Vector3<f64>) -> Self;

    /// Clones and recentres to put the centre of mass at the origin.
    ///
    /// # Returns
    ///
    /// A recentred copy.
    fn recentre(self: &Self) -> Self;
}
