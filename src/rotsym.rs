use nalgebra as na;
use std::fmt;


/// Calculates the absolute or relative difference between two `T`.
///
/// # Arguments
///
/// * `a` - A value of type `T`.
/// * `b` - Another value of type `T`.
/// * `abs_compare` - A flag indicating if absolute or relative difference is
///     to be calculated.
///
/// # Returns
///
/// The required difference.
fn diff<T: na::RealField + Copy>(a: T, b: T, abs_compare: bool) -> T {
    if abs_compare {
        (a - b).abs()
    } else {
        (a - b).abs() / (a + b).abs()
    }
}


/// An enum to classify the types of rotational symmetry of a molecular system
/// based on its principal moments of inertia.
pub enum RotationalSymmetry {
    /// All three principal moments of inertia are identical.
    Spherical,
    /// The unique principal moment of inertia is the largest, the other two
    /// are equal and sum to the unique one.
    OblatePlanar,
    /// The unique principal moment of inertia is the largest, the other two
    /// are equal but do not sum to the unique one.
    OblateNonPlanar,
    /// The unique principal moment of inertia is zero, the other two are equal.
    ProlateLinear,
    /// The unique principal moment of inertia is the smallest but non-zero,
    /// the other two are equal.
    ProlateNonLinear,
    /// The largest principal moment of inertia is the sum of the other two,
    /// but they are all distinct.
    AsymmetricPlanar,
    /// All principal moments of inertia are distinct and do not have any
    /// special relations between them.
    AsymmetricNonPlanar,
}

impl fmt::Display for RotationalSymmetry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RotationalSymmetry::Spherical => write!(f, "Spherical"),
            RotationalSymmetry::OblatePlanar => write!(f, "Oblate, planar"),
            RotationalSymmetry::OblateNonPlanar => write!(f, "Oblate, non-planar"),
            RotationalSymmetry::ProlateLinear => write!(f, "Prolate, linear"),
            RotationalSymmetry::ProlateNonLinear => write!(f, "Prolate, non-linear"),
            RotationalSymmetry::AsymmetricPlanar => write!(f, "Asymmetric, planar"),
            RotationalSymmetry::AsymmetricNonPlanar => write!(f, "Asymmetric, non-planar"),
        }
    }
}


/// Determines the rotational symmetry given an inertia tensor.
///
/// # Arguments
///
/// * `inertia_tensor` - An inertia tensor which is a $3 \times 3$ matrix.
/// * `thresh` - A threshold for comparing moments of inertia.
/// * `verbose` - The print level.
/// * `abs_compare` - A flag indicating if absolute or relative difference
///     should be used in moment of inertia comparisons.
///
/// # Returns
///
/// The rotational symmetry as one of the [`RotationalSymmetry`] variants.
pub fn calc_rotational_symmetry(
    inertia_tensor: &na::Matrix3<f64>,
    thresh: f64,
    verbose: u64,
    abs_compare: bool,
) -> RotationalSymmetry {
    let moi_mat = inertia_tensor.symmetric_eigenvalues();
    let mut moi: Vec<&f64> = moi_mat.iter().collect();
    moi.sort_by(|a, b| a.partial_cmp(b).unwrap());
    if verbose > 0 {
        println!(
            "Moments of inertia:\n {:.6}\n {:.6}\n {:.6}",
            moi[0], moi[1], moi[2]
        );
        if abs_compare {
            println!("Threshold for absolute MoI comparison: {:.3e}", thresh);
        } else {
            println!("Threshold for relative MoI comparison: {:.3e}", thresh);
        }
    }
    if diff(*moi[0], *moi[1], abs_compare) < thresh {
        if diff(*moi[1], *moi[2], abs_compare) < thresh {
            return RotationalSymmetry::Spherical;
        }
        if diff(*moi[2], *moi[0] + *moi[1], abs_compare) < thresh {
            return RotationalSymmetry::OblatePlanar;
        }
        return RotationalSymmetry::OblateNonPlanar;
    }
    if diff(*moi[1], *moi[2], abs_compare) < thresh {
        if moi[0].abs() < thresh {
            return RotationalSymmetry::ProlateLinear;
        }
        return RotationalSymmetry::ProlateNonLinear;
    }
    if diff(*moi[2], *moi[0] + *moi[1], abs_compare) < thresh {
        return RotationalSymmetry::AsymmetricPlanar;
    }
    RotationalSymmetry::AsymmetricNonPlanar
}
