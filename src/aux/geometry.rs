use std::collections::HashSet;
use std::fmt;

use approx;
use derive_builder::Builder;
use fraction;
use itertools::{self, Itertools};
use nalgebra::{ClosedMul, Matrix3, Point3, Rotation3, Scalar, UnitVector3, Vector3};
use num_traits::{One, ToPrimitive};

use crate::aux::atom::Atom;
use crate::aux::misc::HashableFloat;

type F32 = fraction::GenericFraction<u32>;

#[cfg(test)]
#[path = "geometry_tests.rs"]
mod geometry_tests;

// ================
// Enum definitions
// ================

/// An enum to classify the type of improper rotation given an angle and axis.
pub enum ImproperRotationKind {
    /// The improper rotation is a rotation by the specified angle and axis followed by a
    /// reflection in a mirror plane perpendicular to the axis.
    MirrorPlane,

    /// The improper rotation is a rotation by the specified angle and axis followed by an
    /// inversion through the centre of inversion.
    InversionCentre,
}

pub const IMSIG: ImproperRotationKind = ImproperRotationKind::MirrorPlane;
pub const IMINV: ImproperRotationKind = ImproperRotationKind::InversionCentre;

// =================
// Utility functions
// =================

/// Returns the rotation angle adjusted to be in the interval $`(-\pi, +\pi]`$.
///
/// # Arguments
///
/// * `rot_ang` - A rotation angle.
/// * `thresh` - A threshold for comparisons.
///
/// # Returns
///
/// The normalised rotation angle.
#[must_use]
pub fn normalise_rotation_angle(rot_ang: f64, thresh: f64) -> (f64, u32) {
    let frac_1_2 = 1.0 / 2.0;
    let fraction = rot_ang / (2.0 * std::f64::consts::PI);
    if fraction > frac_1_2 + thresh {
        let integer_part = fraction.trunc().to_u32().unwrap_or_else(|| {
            panic!("Unable to convert the integer part of `{fraction}` to `u32`.")
        });
        let x = if fraction.fract() <= frac_1_2 + thresh {
            integer_part
        } else {
            integer_part + 1
        };
        (rot_ang - 2.0 * std::f64::consts::PI * f64::from(x), x)
    } else if fraction <= -frac_1_2 + thresh {
        let integer_part = (-fraction).trunc().to_u32().unwrap_or_else(|| {
            panic!("Unable to convert the integer part of `{fraction}` to `u32`.")
        });
        let x = if (-fraction).fract() < frac_1_2 - thresh {
            integer_part
        } else {
            integer_part + 1
        };
        (rot_ang + 2.0 * std::f64::consts::PI * f64::from(x), x)
    } else {
        (rot_ang, 0)
    }
}

pub fn normalise_rotation_fraction(fraction: F32) -> (F32, u32) {
    // Consider a fraction f.
    //
    // If f > 1/2, we seek a positive integer x such that
    //  -1/2 < f - x <= 1/2.
    // It turns out that x ∈ [f - 1/2, f + 1/2).
    //
    // If f <= -1/2, we seek a positive integer x such that
    //  -1/2 < f + x <= 1/2.
    // It turns out that x ∈ (-f - 1/2, -f + 1/2].
    //
    // If the proper rotation corresponding to f is reached from the identity
    // via a continuous path in the parametric ball, x gives the number of times
    // this path goes through a podal-antipodal jump, and thus whether x is even
    // corresponds to whether this homotopy path is of class 0.
    //
    // See S.L. Altmann, Rotations, Quaternions, and Double Groups (Dover
    // Publications, Inc., New York, 2005) for further information.
    let frac_1_2 = F32::new(1u32, 2u32);
    if fraction > frac_1_2 {
        let integer_part = fraction.trunc();
        let x = if fraction.fract() <= frac_1_2 {
            integer_part
        } else {
            integer_part + F32::one()
        };
        (
            fraction - x,
            x.to_u32()
                .expect("Unable to convert the 2π-turn number to `u32`."),
        )
    } else if fraction <= -frac_1_2 {
        let integer_part = (-fraction).trunc();
        let x = if (-fraction).fract() < frac_1_2 {
            integer_part
        } else {
            integer_part + F32::one()
        };
        (
            fraction + x,
            x.to_u32()
                .expect("Unable to convert the 2π-turn number to `u32`."),
        )
    } else {
        (fraction, 0)
    }
}

/// Returns the rotation angle adjusted to be in the interval $(-2\pi, +2\pi]$.
///
/// This is important to distinguish operations in double groups.
///
/// # Arguments
///
/// * `rot_ang` - A rotation angle.
/// * `thresh` - A threshold for comparisons.
///
/// # Returns
///
/// The normalised double-rotation angle.
#[must_use]
pub fn normalise_rotation_double_angle(rot_ang: f64, thresh: f64) -> f64 {
    let mut norm_rot_ang = rot_ang;
    while norm_rot_ang > 2.0 * std::f64::consts::PI + thresh {
        norm_rot_ang -= 4.0 * std::f64::consts::PI;
    }
    while norm_rot_ang <= -2.0 * std::f64::consts::PI + thresh {
        norm_rot_ang += 4.0 * std::f64::consts::PI;
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
///
/// # Panics
///
/// Panics if the resulting pole is a null vector.
#[must_use]
pub fn get_standard_positive_pole(axis: &Vector3<f64>, thresh: f64) -> Vector3<f64> {
    let poshem = PositiveHemisphere::new_standard_cartesian();
    poshem.get_positive_pole(axis, thresh)
    // let mut pole = axis.normalize();
    // if pole[2].abs() > thresh {
    //     pole *= pole[2].signum();
    // } else if pole[0].abs() > thresh {
    //     pole *= pole[0].signum();
    // } else {
    //     assert!(pole[1].abs() > thresh);
    //     pole *= pole[1].signum();
    // }
    // pole
}

/// Check if a rotation axis is in the positive hemisphere.
///
/// The definition of the positive hemisphere can be found in S.L. Altmann, Rotations,
/// Quaternions, and Double Groups (Dover Publications, Inc., New York, 2005)
/// (Chapter 9).
///
/// # Arguments
///
/// * axis - An axis of rotation.
/// * thresh - Threshold for comparisons.
///
/// # Returns
///
/// Returns `true` if `axis` is in the positive hemisphere.
///
/// # Panics
///
/// Panics if the axis is a null vector.
#[must_use]
pub fn check_standard_positive_pole(axis: &Vector3<f64>, thresh: f64) -> bool {
    let poshem = PositiveHemisphere::new_standard_cartesian();
    poshem.check_positive_pole(axis, thresh)
    // let normalised_axis = axis.normalize();
    // normalised_axis[2] > thresh
    //     || (approx::relative_eq!(
    //         normalised_axis[2],
    //         0.0,
    //         max_relative = thresh,
    //         epsilon = thresh
    //     ) && normalised_axis[0] > thresh)
    //     || (approx::relative_eq!(
    //         normalised_axis[2],
    //         0.0,
    //         max_relative = thresh,
    //         epsilon = thresh
    //     ) && approx::relative_eq!(
    //         normalised_axis[0],
    //         0.0,
    //         max_relative = thresh,
    //         epsilon = thresh
    //     ) && normalised_axis[1] > thresh)
}

/// Determines the reduced fraction $`k/n`$ where $`k`$ and $`n`$ are both integers representing a
/// proper rotation $`C_n^k`$ corresponding to a specified rotation angle.
///
/// # Arguments
///
/// * `angle` - An angle of rotation.
/// * `thresh` - A threshold for checking if a floating point number is integral.
/// * `max_trial_power` - Maximum power $`k`$ to try.
///
/// # Returns
///
/// An [`Option`] wrapping the required fraction.
///
/// # Panics
///
/// Panics if the deduced order $`n`$ is negative.
#[must_use]
pub fn get_proper_fraction(angle: f64, thresh: f64, max_trial_power: u32) -> Option<F32> {
    let (normalised_angle, _) = normalise_rotation_angle(angle, thresh);
    // let positive_normalised_angle = if normalised_angle >= 0.0 {
    //     normalised_angle
    // } else {
    //     2.0 * std::f64::consts::PI + normalised_angle
    // };
    let rational_order = (2.0 * std::f64::consts::PI) / normalised_angle.abs();
    let mut power: u32 = 1;
    while approx::relative_ne!(
        rational_order * (f64::from(power)),
        (rational_order * (f64::from(power))).round(),
        max_relative = thresh,
        epsilon = thresh
    ) && power < max_trial_power
    {
        power += 1;
    }
    if approx::relative_eq!(
        rational_order * (f64::from(power)),
        (rational_order * (f64::from(power))).round(),
        max_relative = thresh,
        epsilon = thresh
    ) {
        let orderf64 = (rational_order * (f64::from(power))).round();
        assert!(orderf64.is_sign_positive());
        assert!(orderf64 <= f64::from(u32::MAX));
        #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
        let order = orderf64 as u32;
        if normalised_angle > 0.0 {
            Some(F32::new(power, order))
        } else {
            Some(F32::new_neg(power, order))
        }
    } else {
        None
    }
    // let rational_order = (2.0 * std::f64::consts::PI) / positive_normalised_angle;
    // let mut power: u32 = 1;
    // while approx::relative_ne!(
    //     rational_order * (f64::from(power)),
    //     (rational_order * (f64::from(power))).round(),
    //     max_relative = thresh,
    //     epsilon = thresh
    // ) && power < max_trial_power
    // {
    //     power += 1;
    // }
    // if approx::relative_eq!(
    //     rational_order * (f64::from(power)),
    //     (rational_order * (f64::from(power))).round(),
    //     max_relative = thresh,
    //     epsilon = thresh
    // ) {
    //     let orderf64 = (rational_order * (f64::from(power))).round();
    //     assert!(orderf64.is_sign_positive());
    //     assert!(orderf64 <= f64::from(u32::MAX));
    //     #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
    //     let order = orderf64 as u32;
    //     Some(F32::new(power, order))
    // } else {
    //     None
    // }
}

/// Computes the outer product between two three-dimensional vectors.
///
/// # Arguments
///
/// * `vec1` - The first vector, $`\mathbf{v}_1`$.
/// * `vec2` - The second vector, $`\mathbf{v}_2`$.
///
/// # Returns
///
/// The outer product $`\mathbf{v}_1 \otimes \mathbf{v}_2`$.
fn outer<T: Scalar + ClosedMul + Copy>(vec1: &Vector3<T>, vec2: &Vector3<T>) -> Matrix3<T> {
    let outer_product_iter: Vec<T> = vec2
        .iter()
        .flat_map(|&item_x| vec1.iter().map(move |&item_y| item_x * item_y))
        .collect();
    Matrix3::from_iterator(outer_product_iter)
}

/// Returns a $3 \times 3$ rotation matrix in $\mathbb{R}^3$ corresponding to a
/// rotation through `angle` about `axis` raised to the power `power`.
///
/// # Arguments
///
/// * `angle` - The angle of rotation.
/// * `axis` - The axis of rotation.
/// * `power` - The power of rotation.
///
/// # Returns
///
/// The rotation matrix.
#[must_use]
pub fn proper_rotation_matrix(angle: f64, axis: &Vector3<f64>, power: i8) -> Matrix3<f64> {
    let normalised_axis = UnitVector3::new_normalize(*axis);
    Rotation3::from_axis_angle(&normalised_axis, (f64::from(power)) * angle).into_inner()
}

/// Returns a $3 \times 3$ transformation matrix in $\mathbb{R}^3$ corresponding
/// to an improper rotation through `angle` about `axis` raised to the power
/// `power`.
///
/// # Arguments
///
/// * `angle` - The angle of rotation.
/// * `axis` - The axis of rotation.
/// * `power` - The power of transformation.
/// * `kind` - The convention in which the improper rotation is defined.
///
/// # Returns
///
/// The transformation matrix.
///
/// # Panics
///
/// Panics if `kind` is not one of the improper kinds, or if `kind` contains time reversal.
#[must_use]
pub fn improper_rotation_matrix(
    angle: f64,
    axis: &Vector3<f64>,
    power: i8,
    kind: &ImproperRotationKind,
) -> Matrix3<f64> {
    let rotmat = proper_rotation_matrix(angle, axis, power);
    let normalised_axis = UnitVector3::new_normalize(*axis);
    match kind {
        ImproperRotationKind::MirrorPlane => {
            let refmat = Matrix3::identity()
                - 2.0 * (f64::from(power % 2)) * outer(&normalised_axis, &normalised_axis);
            refmat * rotmat
        }
        ImproperRotationKind::InversionCentre => {
            if power % 2 == 1 {
                -rotmat
            } else {
                rotmat
            }
        }
    }
}

/// Checks if a sequence of atoms are vertices of a regular polygon.
///
/// # Arguments
///
/// * `atoms` - A sequence of atoms to be tested.
///
/// # Returns
///
/// A flag indicating if the atoms form the vertices of a regular polygon.
///
/// # Panics
///
/// Panics if `atoms` contains fewer than three atoms.
#[must_use]
pub fn check_regular_polygon(atoms: &[&Atom]) -> bool {
    assert!(
        atoms.len() >= 3,
        "Polygons can only be formed by three atoms or more."
    );

    let tot_m: f64 = atoms.iter().fold(0.0, |acc, atom| acc + atom.atomic_mass);
    let com: Point3<f64> = atoms.iter().fold(Point3::origin(), |acc, atom| {
        acc + (atom.coordinates * atom.atomic_mass - Point3::origin())
    }) / tot_m;

    let radial_dists: HashSet<(u64, i16, i8)> = atoms
        .iter()
        .map(|atom| {
            (atom.coordinates - com)
                .norm()
                .round_factor(atom.threshold)
                .integer_decode()
        })
        .collect();

    // Check if all atoms are equidistant from the centre of mass
    if radial_dists.len() == 1 {
        let regular_angle = 2.0 * std::f64::consts::PI
            / atoms
                .len()
                .to_f64()
                .unwrap_or_else(|| panic!("Unable to convert `{}` to `f64`.", atoms.len()));
        let thresh = atoms
            .iter()
            .fold(0.0_f64, |acc, atom| acc.max(atom.threshold));
        let mut rad_vectors: Vec<Vector3<f64>> =
            atoms.iter().map(|atom| atom.coordinates - com).collect();
        let (vec_i, vec_j) = itertools::iproduct!(rad_vectors.iter(), rad_vectors.iter())
            .max_by(|&(v_i1, v_j1), &(v_i2, v_j2)| {
                v_i1.cross(v_j1)
                    .norm()
                    .partial_cmp(&v_i2.cross(v_j2).norm())
                    .expect("Unable to compare the cross products of two vector pairs.")
            })
            .expect("Unable to find the vector pair with the largest norm cross product.");
        let normal = UnitVector3::new_normalize(vec_i.cross(vec_j));
        if normal.norm() < thresh {
            return false;
        }

        let vec0 = atoms[0].coordinates - com;
        rad_vectors.sort_by(|a, b| {
            get_anticlockwise_angle(&vec0, a, &normal, thresh)
                .partial_cmp(&get_anticlockwise_angle(&vec0, b, &normal, thresh))
                .unwrap_or_else(|| {
                    panic!(
                        "Unable to compare anticlockwise angles of {a} and {b} relative to {vec0}."
                    )
                })
        });
        let vector_pairs: Vec<(&Vector3<f64>, &Vector3<f64>)> =
            rad_vectors.iter().circular_tuple_windows().collect();
        let mut angles: HashSet<(u64, i16, i8)> = vector_pairs
            .iter()
            .map(|(v1, v2)| {
                get_anticlockwise_angle(v1, v2, &normal, thresh)
                    .round_factor(thresh)
                    .integer_decode()
            })
            .collect();
        angles.insert(regular_angle.round_factor(thresh).integer_decode());

        angles.len() == 1
    } else {
        false
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
    let dot = vec1.dot(vec2);
    let det = normal.into_inner().dot(&vec1.cross(vec2));
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
    fn transform_mut(&mut self, mat: &Matrix3<f64>);

    /// Rotates in-place the coordinates through `angle` about `axis`.
    ///
    /// # Arguments
    ///
    /// * angle - The angle of rotation.
    /// * axis - The axis of rotation.
    fn rotate_mut(&mut self, angle: f64, axis: &Vector3<f64>);

    /// Improper-rotates in-place the coordinates through `angle` about `axis`.
    ///
    /// # Arguments
    ///
    /// * angle - The angle of rotation.
    /// * axis - The axis of rotation.
    /// * kind - The convention in which the improper rotation is defined.
    fn improper_rotate_mut(&mut self, angle: f64, axis: &Vector3<f64>, kind: &ImproperRotationKind);

    /// Translates in-place the coordinates by a specified translation vector in
    /// three dimensions.
    ///
    /// # Arguments
    ///
    /// * tvec - The translation vector.
    fn translate_mut(&mut self, tvec: &Vector3<f64>);

    /// Recentres in-place to put the centre of mass at the origin.
    fn recentre_mut(&mut self);

    /// Reverses time by reversing in-place the polarity of any magnetic special atoms.
    fn reverse_time_mut(&mut self);

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
    #[must_use]
    fn transform(&self, mat: &Matrix3<f64>) -> Self;

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
    #[must_use]
    fn rotate(&self, angle: f64, axis: &Vector3<f64>) -> Self;

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
    #[must_use]
    fn improper_rotate(&self, angle: f64, axis: &Vector3<f64>, kind: &ImproperRotationKind)
        -> Self;

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
    #[must_use]
    fn translate(&self, tvec: &Vector3<f64>) -> Self;

    /// Clones and recentres to put the centre of mass at the origin.
    ///
    /// # Returns
    ///
    /// A recentred copy.
    #[must_use]
    fn recentre(&self) -> Self;

    /// Clones the molecule and reverses time by reversing the polarity of any magnetic special
    /// atoms.
    ///
    /// # Returns
    ///
    /// A time-reversed copy.
    #[must_use]
    fn reverse_time(&self) -> Self;
}

// ===================
// Positive Hemisphere
// ===================

#[derive(Clone, Debug)]
enum ImproperOrdering {
    Greater,
    GreaterEqual,
    Less,
    LessEqual,
    Equal,
}

#[derive(Debug, Clone)]
enum CartesianCoordinate {
    X,
    Y,
    Z,
}

impl CartesianCoordinate {
    fn to_index(&self) -> usize {
        match self {
            CartesianCoordinate::X => 0,
            CartesianCoordinate::Y => 1,
            CartesianCoordinate::Z => 2,
        }
    }
}

impl fmt::Display for CartesianCoordinate {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            CartesianCoordinate::X => write!(f, "x"),
            CartesianCoordinate::Y => write!(f, "y"),
            CartesianCoordinate::Z => write!(f, "z"),
        }
    }
}

#[derive(Clone)]
pub struct CartesianConditions {
    conditions: Vec<Vec<(CartesianCoordinate, ImproperOrdering, f64)>>,
}

impl CartesianConditions {
    fn check(&self, vec: &Vector3<f64>, thresh: f64) -> bool {
        self.conditions.iter().any(|condition_set| {
            condition_set.iter().all(|(i, order, target)| match order {
                ImproperOrdering::Greater => vec[i.to_index()] > target + thresh,
                ImproperOrdering::GreaterEqual => vec[i.to_index()] > target - thresh,
                ImproperOrdering::Less => vec[i.to_index()] < target - thresh,
                ImproperOrdering::LessEqual => vec[i.to_index()] < target + thresh,
                ImproperOrdering::Equal => approx::relative_eq!(
                    vec[i.to_index()],
                    target,
                    max_relative = thresh,
                    epsilon = thresh
                ),
            })
        })
    }

    fn new_standard() -> Self {
        Self {
            conditions: vec![
                vec![(
                    CartesianCoordinate::Z,
                    ImproperOrdering::Greater,
                    0.0,
                )],
                vec![
                    (CartesianCoordinate::Z, ImproperOrdering::Equal, 0.0),
                    (CartesianCoordinate::X, ImproperOrdering::Greater, 0.0),
                ],
                vec![
                    (CartesianCoordinate::Z, ImproperOrdering::Equal, 0.0),
                    (CartesianCoordinate::X, ImproperOrdering::Equal, 0.0),
                    (CartesianCoordinate::Y, ImproperOrdering::Greater, 0.0),
                ],
            ],
        }
    }
}

impl Default for CartesianConditions {
    fn default() -> Self {
        Self::new_standard()
    }
}

#[derive(Clone)]
pub enum SphericalCoordinate {
    Theta,
    Phi,
}

impl fmt::Display for SphericalCoordinate {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            SphericalCoordinate::Theta => write!(f, "θ"),
            SphericalCoordinate::Phi => write!(f, "φ"),
        }
    }
}

#[derive(Clone, Builder)]
pub struct SphericalConditions {
    #[builder(setter(custom))]
    z_basis: Vector3<f64>,

    #[builder(setter(custom))]
    x_basis: Vector3<f64>,

    #[builder(setter(custom))]
    conditions: Vec<Vec<(SphericalCoordinate, ImproperOrdering, f64)>>,
}

impl SphericalConditionsBuilder {
    fn z_basis(&mut self, z_bas: Vector3<f64>) -> &mut Self {
        self.z_basis = Some(z_bas.normalize());
        self
    }

    fn x_basis(&mut self, x_bas: Vector3<f64>) -> &mut Self {
        self.x_basis = Some(x_bas.normalize());
        self
    }

    fn conditions(
        &mut self,
        conds: &[Vec<(SphericalCoordinate, ImproperOrdering, f64)>],
    ) -> &mut Self {
        self.conditions = Some(conds.to_vec());
        self
    }
}

impl SphericalConditions {
    fn builder() -> SphericalConditionsBuilder {
        SphericalConditionsBuilder::default()
    }

    fn get_component(&self, vec: &Vector3<f64>, coord: &SphericalCoordinate, thresh: f64) -> f64 {
        match coord {
            SphericalCoordinate::Theta => {
                let cos_theta = vec.dot(&self.z_basis) / (vec.norm() * self.z_basis.norm());
                if approx::relative_eq!(cos_theta, 1.0, epsilon = thresh, max_relative = thresh) {
                    1.0f64.acos()
                } else if approx::relative_eq!(
                    cos_theta,
                    -1.0,
                    epsilon = thresh,
                    max_relative = thresh
                ) {
                    (-1.0f64).acos()
                } else {
                    cos_theta.acos()
                }
            }
            SphericalCoordinate::Phi => {
                let y_vector = self.z_basis.cross(&self.x_basis).normalize();
                let xy_vec = vec - vec.dot(&self.z_basis) / self.z_basis.norm() * self.z_basis;
                let sgn_y = xy_vec.dot(&y_vector).signum();
                let cos_phi = xy_vec.dot(&self.x_basis) / (xy_vec.norm() * self.x_basis.norm());
                if approx::relative_eq!(cos_phi, 1.0, epsilon = thresh, max_relative = thresh) {
                    sgn_y * 1.0f64.acos()
                } else if approx::relative_eq!(
                    cos_phi,
                    -1.0,
                    epsilon = thresh,
                    max_relative = thresh
                ) {
                    sgn_y * (-1.0f64).acos()
                } else {
                    sgn_y * cos_phi.acos()
                }
            }
        }
    }

    fn check(&self, vec: &Vector3<f64>, thresh: f64) -> bool {
        self.conditions.iter().any(|condition_set| {
            condition_set.iter().all(|(i, order, target)| {
                let component = self.get_component(vec, i, thresh);
                match order {
                    ImproperOrdering::Greater => component > target + thresh,
                    ImproperOrdering::GreaterEqual => component > target - thresh,
                    ImproperOrdering::Less => component < target - thresh,
                    ImproperOrdering::LessEqual => component < target + thresh,
                    ImproperOrdering::Equal => approx::relative_eq!(
                        component,
                        target,
                        max_relative = thresh,
                        epsilon = thresh
                    ),
                }
            })
        })
    }

    fn new_standard() -> Self {
        let half_pi = 0.5 * std::f64::consts::PI;
        let conditions = vec![
            vec![
                (
                    SphericalCoordinate::Theta,
                    ImproperOrdering::GreaterEqual,
                    0.0,
                ),
                (SphericalCoordinate::Theta, ImproperOrdering::Less, half_pi),
            ],
            vec![
                (SphericalCoordinate::Theta, ImproperOrdering::Equal, half_pi),
                (
                    SphericalCoordinate::Phi,
                    ImproperOrdering::Greater,
                    -half_pi,
                ),
                (
                    SphericalCoordinate::Phi,
                    ImproperOrdering::LessEqual,
                    half_pi,
                ),
            ],
        ];
        Self::builder()
            .z_basis(Vector3::z())
            .x_basis(Vector3::x())
            .conditions(&conditions)
            .build()
            .expect("Unable to construct a set of spherical-coordinate conditions.")
    }

    fn new_disjoint_equatorial_arcs(
        z_basis: Vector3<f64>,
        x_basis: Vector3<f64>,
        n: usize,
    ) -> Self {
        assert!(n > 0);
        let n_f64 = n
            .to_f64()
            .expect("Unable to convert the number of arcs to `f64`.");
        let half_arc = std::f64::consts::PI / (2.0 * n_f64);
        let sep = 2.0 * std::f64::consts::PI / n_f64;
        let half_pi = 0.5 * std::f64::consts::PI;

        let mut conditions = vec![vec![
            (
                SphericalCoordinate::Theta,
                ImproperOrdering::GreaterEqual,
                0.0,
            ),
            (SphericalCoordinate::Theta, ImproperOrdering::Less, half_pi),
        ]];

        let phi_conditions = (0..n)
            .map(|i| {
                let (centre, _) = normalise_rotation_angle(i.to_f64().unwrap() * sep, f64::EPSILON);
                let min_exc = centre - half_arc;
                let max_inc = centre + half_arc;
                vec![
                    (SphericalCoordinate::Theta, ImproperOrdering::Equal, half_pi),
                    (SphericalCoordinate::Phi, ImproperOrdering::Greater, min_exc),
                    (
                        SphericalCoordinate::Phi,
                        ImproperOrdering::LessEqual,
                        max_inc,
                    ),
                ]
            })
            .collect_vec();

        conditions.extend(phi_conditions.into_iter());
        Self::builder()
            .z_basis(z_basis)
            .x_basis(x_basis)
            .conditions(&conditions)
            .build()
            .expect("Unable to construct a set of spherical-coordinate conditions.")
    }
}

#[derive(Clone)]
pub enum PositiveHemisphere {
    Cartesian(CartesianConditions),
    Spherical(SphericalConditions),
}

impl PositiveHemisphere {
    pub fn check_positive_pole(&self, axis: &Vector3<f64>, thresh: f64) -> bool {
        let normalised_axis = axis.normalize();
        match self {
            PositiveHemisphere::Cartesian(cart_conditions) => {
                cart_conditions.check(&normalised_axis, thresh)
            }
            PositiveHemisphere::Spherical(sph_conditions) => {
                sph_conditions.check(&normalised_axis, thresh)
            }
        }
    }

    pub fn get_positive_pole(&self, axis: &Vector3<f64>, thresh: f64) -> Vector3<f64> {
        let normalised_axis = axis.normalize();
        if self.check_positive_pole(&normalised_axis, thresh) {
            normalised_axis
        } else {
            -normalised_axis
        }
    }

    pub fn new_standard_cartesian() -> Self {
        Self::Cartesian(CartesianConditions::new_standard())
    }

    pub fn new_standard_spherical() -> Self {
        Self::Spherical(SphericalConditions::new_standard())
    }

    pub fn new_spherical_disjoint_equatorial_arcs(
        z_basis: Vector3<f64>,
        x_basis: Vector3<f64>,
        n: usize,
    ) -> Self {
        Self::Spherical(SphericalConditions::new_disjoint_equatorial_arcs(
            z_basis, x_basis, n,
        ))
    }
}
