//! Rotational symmetry based on moments of inertia.

use std::fmt;

use approx;
use log;
use nalgebra as na;
use serde::{Deserialize, Serialize};

#[cfg(test)]
#[path = "rotsym_tests.rs"]
mod rotsym_tests;

/// Enumerated type to classify the types of rotational symmetry of a molecular system based on its
/// principal moments of inertia.
#[derive(Clone, Debug, Serialize, Deserialize)]
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
/// * `inertia_tensor` - An inertia tensor which is a $`3 \times 3`$ matrix.
/// * `thresh` - A threshold for comparing moments of inertia.
///
/// # Returns
///
/// The rotational symmetry as one of the [`RotationalSymmetry`] variants.
///
/// # Panics
///
/// Panics when the moments of inertia cannot be compared.
#[must_use]
pub fn calc_rotational_symmetry(
    inertia_tensor: &na::Matrix3<f64>,
    thresh: f64,
) -> RotationalSymmetry {
    let moi_mat = inertia_tensor.symmetric_eigenvalues();
    let mut moi: Vec<&f64> = moi_mat.iter().collect();
    moi.sort_by(|a, b| {
        a.partial_cmp(b)
            .unwrap_or_else(|| panic!("`{a}` and `{b}` cannot be compared."))
    });
    log::debug!("Moments of inertia:");
    for component in &moi {
        log::debug!("  {component:+.14}");
    }
    if approx::relative_eq!(*moi[0], *moi[1], epsilon = thresh, max_relative = thresh) {
        if approx::relative_eq!(*moi[1], *moi[2], epsilon = thresh, max_relative = thresh) {
            return RotationalSymmetry::Spherical;
        }
        if approx::relative_eq!(
            *moi[2],
            *moi[0] + *moi[1],
            epsilon = thresh,
            max_relative = thresh
        ) {
            return RotationalSymmetry::OblatePlanar;
        }
        return RotationalSymmetry::OblateNonPlanar;
    }
    if approx::relative_eq!(*moi[1], *moi[2], epsilon = thresh, max_relative = thresh) {
        if moi[0].abs() < thresh {
            return RotationalSymmetry::ProlateLinear;
        }
        return RotationalSymmetry::ProlateNonLinear;
    }
    if approx::relative_eq!(
        *moi[2],
        *moi[0] + *moi[1],
        epsilon = thresh,
        max_relative = thresh
    ) {
        return RotationalSymmetry::AsymmetricPlanar;
    }
    RotationalSymmetry::AsymmetricNonPlanar
}
