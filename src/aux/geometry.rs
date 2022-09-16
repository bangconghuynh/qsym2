use crate::aux::atom::Atom;
use crate::aux::misc::HashableFloat;
use crate::symmetry::symmetry_element::SymmetryElementKind;
use itertools::{self, Itertools};
use nalgebra::{ClosedMul, Matrix3, Point3, Rotation3, Scalar, UnitVector3, Vector3};
use std::collections::HashSet;
use fraction;
use approx;

type F32 = fraction::GenericFraction<u32>;

#[cfg(test)]
#[path = "geometry_tests.rs"]
mod geometry_tests;

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
    let mut pole = axis.normalize();
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

pub fn get_proper_fraction(angle: f64, thresh: f64, max_trial_power: u32) -> F32 {
    let normalised_angle = normalise_rotation_angle(angle, thresh);
    let positive_normalised_angle = if normalised_angle >= 0.0 {
        normalised_angle
    } else {
        2.0 * std::f64::consts::PI + normalised_angle
    };
    let rational_order = (2.0 * std::f64::consts::PI) / positive_normalised_angle;
    println!("Rat ord, max pow: {rational_order}, {max_trial_power}");
    let mut power: u32 = 1;
    while approx::relative_ne!(
        rational_order * (power as f64),
        (rational_order * (power as f64)).round(),
        max_relative = thresh,
        epsilon = thresh
    ) && power < max_trial_power {
        power += 1;
        println!("{}, {}", power, rational_order * (power as f64));
    }
    let order = if approx::relative_eq!(
        rational_order * (power as f64),
        (rational_order * (power as f64)).round(),
        max_relative = thresh,
        epsilon = thresh
    ) {
        Ok((rational_order * (power as f64)).round() as u32)
    } else {
        Err(format!("No proper fractions can be found."))
    };
    F32::new(power, order.unwrap())
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
    kind: &SymmetryElementKind,
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

/// Checks if a sequence of atoms are vertices of a regular polygon.
///
/// # Arguments
///
/// * atoms - A sequence of atoms to be tested.
///
/// # Returns
///
/// A flag indicating if the atoms form the vertices of a regular polygon.
pub fn check_regular_polygon(atoms: &[&Atom]) -> bool {
    assert!(
        atoms.len() >= 3,
        "Polygons can only be formed by three atoms or more."
    );

    let tot_m: f64 = atoms.iter().fold(0.0, |acc, atom| acc + atom.atomic_mass);
    let com: Point3<f64> = atoms.iter().fold(Point3::origin(), |acc, atom| {
        acc + (&atom.coordinates * atom.atomic_mass - Point3::origin())
    }) / tot_m;

    let radial_dists: HashSet<(u64, i16, i8)> = atoms
        .iter()
        .map(|atom| {
            (&atom.coordinates - &com)
                .norm()
                .round_factor(atom.threshold)
                .integer_decode()
        })
        .collect();

    // Check if all atoms are equidistant from the centre of mass
    if radial_dists.len() != 1 {
        false
    } else {
        let regular_angle = 2.0 * std::f64::consts::PI / (atoms.len() as f64);
        let thresh = atoms
            .iter()
            .fold(0.0_f64, |acc, atom| acc.max(atom.threshold));
        let mut rad_vectors: Vec<Vector3<f64>> =
            atoms.iter().map(|atom| atom.coordinates - &com).collect();
        let (vec_i, vec_j) = itertools::iproduct!(rad_vectors.iter(), rad_vectors.iter())
            .max_by(|&(v_i1, v_j1), &(v_i2, v_j2)| {
                v_i1.cross(v_j1)
                    .norm()
                    .partial_cmp(&v_i2.cross(v_j2).norm())
                    .unwrap()
            })
            .unwrap();
        let normal = UnitVector3::new_normalize(vec_i.cross(vec_j));
        if normal.norm() < thresh { return false }

        let vec0 = atoms[0].coordinates - &com;
        rad_vectors.sort_by(|a, b| {
            get_anticlockwise_angle(&vec0, a, &normal, thresh)
                .partial_cmp(&get_anticlockwise_angle(&vec0, b, &normal, thresh))
                .unwrap()
        });
        let vector_pairs: Vec<(&Vector3<f64>, &Vector3<f64>)> =
            rad_vectors.iter().circular_tuple_windows().collect();
        let mut angles: HashSet<(u64, i16, i8)> = vector_pairs
            .iter()
            .map(|(v1, v2)| {
                get_anticlockwise_angle(*v1, *v2, &normal, thresh)
                    .round_factor(thresh)
                    .integer_decode()
            })
            .collect();
        angles.insert(regular_angle.round_factor(thresh).integer_decode());

        angles.len() == 1
    }
}

/// Returns the anticlockwise angle $\phi$ from `vec1` to `vec2` when viewed down
/// the `normal` vector.
///
/// This is only well-defined in $\mathbb{R}^3$. The range of the anticlockwise
/// angle is $[0, 2\pi]$.
///
/// # Arguments
///
/// * vec1 - The first vector.
/// * vec2 - The second vector.
/// * normal - A normal unit vector defining the view.
///
/// # Returns
///
/// The anticlockwise angle $\phi$.
fn get_anticlockwise_angle(
    vec1: &Vector3<f64>,
    vec2: &Vector3<f64>,
    normal: &UnitVector3<f64>,
    thresh: f64,
) -> f64 {
    assert!(thresh >= std::f64::EPSILON);
    assert!(vec1.norm() >= thresh);
    assert!(vec2.norm() >= thresh);
    let dot = vec1.dot(&vec2);
    let det = normal.into_inner().dot(&vec1.cross(&vec2));
    let mut angle = det.atan2(dot);
    while angle < -thresh {
        angle += 2.0 * std::f64::consts::PI;
    }
    angle
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
        kind: &SymmetryElementKind,
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
        kind: &SymmetryElementKind,
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
