//! Geometrical symmetry elements.

use std::convert::TryInto;
use std::fmt;
use std::hash::{Hash, Hasher};

use anyhow::{ensure, format_err};
use approx;
use derive_builder::Builder;
use fraction;
use log;
use nalgebra::Vector3;
use num::integer::gcd;
use num_traits::{ToPrimitive, Zero};
use serde::{Deserialize, Serialize};

use crate::auxiliary::geometry;
use crate::auxiliary::misc::{self, HashableFloat};
use crate::symmetry::symmetry_element_order::ElementOrder;

type F = fraction::GenericFraction<u32>;

pub mod symmetry_operation;
pub use symmetry_operation::*;

#[cfg(test)]
mod symmetry_element_tests;

// ====================================
// Enum definitions and implementations
// ====================================

/// Enumerated type to classify the type of the antiunitary term that contributes to a symmetry
/// element.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AntiunitaryKind {
    /// Variant for the antiunitary term being a complex-conjugation operation.
    ComplexConjugation,

    /// Variant for the antiunitary term being a time-reversal operation.
    TimeReversal,
}

/// Enumerated type to classify the types of symmetry element.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SymmetryElementKind {
    /// Proper symmetry element which consists of just a proper rotation axis.
    ///
    /// The associated option indicates the type of the associated antiunitary operation, if any.
    Proper(Option<AntiunitaryKind>),

    /// Improper symmetry element in the mirror-plane convention, which consists of a proper
    /// rotation axis and an orthogonal mirror plane.
    ///
    /// The associated option indicates the type of the associated antiunitary operation, if any.
    ImproperMirrorPlane(Option<AntiunitaryKind>),

    /// Improper symmetry element in the inversion-centre convention, which consists of a proper
    /// rotation axis and an inversion centre.
    ///
    /// The associated option indicates the type of the associated antiunitary operation, if any.
    ImproperInversionCentre(Option<AntiunitaryKind>),
}

impl SymmetryElementKind {
    /// Indicates if the antiunitary part of this element contains a time-reversal operation.
    #[must_use]
    pub fn contains_time_reversal(&self) -> bool {
        match self {
            Self::Proper(tr)
            | Self::ImproperMirrorPlane(tr)
            | Self::ImproperInversionCentre(tr) => *tr == Some(AntiunitaryKind::TimeReversal),
        }
    }

    /// Indicates if the antiunitary part of this element contains a pure complex conjugation.
    #[must_use]
    pub fn contains_complex_conjugation(&self) -> bool {
        match self {
            Self::Proper(tr)
            | Self::ImproperMirrorPlane(tr)
            | Self::ImproperInversionCentre(tr) => *tr == Some(AntiunitaryKind::ComplexConjugation),
        }
    }

    /// Indicates if an antiunitary operation is associated with this element.
    #[must_use]
    pub fn contains_antiunitary(&self) -> bool {
        match self {
            Self::Proper(au)
            | Self::ImproperMirrorPlane(au)
            | Self::ImproperInversionCentre(au) => au.is_some(),
        }
    }

    /// Converts the current kind to the desired time-reversal form.
    ///
    /// # Arguments
    ///
    /// * `tr` - A boolean indicating whether time reversal is included (`true`) or removed
    /// (`false`).
    ///
    /// # Returns
    ///
    /// A copy of the current kind with or without the time-reversal antiunitary operation.
    #[must_use]
    pub fn to_tr(&self, tr: bool) -> Self {
        if tr {
            self.to_antiunitary(Some(AntiunitaryKind::TimeReversal))
        } else {
            self.to_antiunitary(None)
        }
    }

    /// Converts the associated antiunitary operation to the desired kind.
    ///
    /// # Arguments
    ///
    /// * `au` - An option containing the desired antiunitary kind.
    ///
    /// # Returns
    ///
    /// A new symmetry element kind with the desired antiunitary kind.
    pub fn to_antiunitary(&self, au: Option<AntiunitaryKind>) -> Self {
        match self {
            Self::Proper(_) => Self::Proper(au),
            Self::ImproperMirrorPlane(_) => Self::ImproperMirrorPlane(au),
            Self::ImproperInversionCentre(_) => Self::ImproperInversionCentre(au),
        }
    }
}

/// Enumerated type to indicate the rotation group to which the unitary proper rotation part of a
/// symmetry element belongs.
#[derive(Clone, Hash, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub enum RotationGroup {
    /// Variant indicating that the proper part of the symmetry element generates rotations in
    /// $`\mathsf{SO}(3)`$.
    SO3,

    /// Variant indicating that the proper part of the symmetry element generates rotations in
    /// $`\mathsf{SU}(2)`$. The associated boolean indicates whether the proper part of the element
    /// itself is connected to the identity via a homotopy path of class 0 (`true`) or class 1
    /// (`false`).
    SU2(bool),
}

impl RotationGroup {
    /// Indicates if the rotation is in $`\mathsf{SU}(2)`$.
    pub fn is_su2(&self) -> bool {
        matches!(self, RotationGroup::SU2(_))
    }

    /// Indicates if the rotation is in $`\mathsf{SU}(2)`$ and connected to the
    /// identity via a homotopy path of class 1.
    pub fn is_su2_class_1(&self) -> bool {
        matches!(self, RotationGroup::SU2(false))
    }
}

// ======================================
// Struct definitions and implementations
// ======================================

pub struct SymmetryElementKindConversionError(String);

impl fmt::Debug for SymmetryElementKindConversionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SymmetryElementKindConversionError")
            .field("Message", &self.0)
            .finish()
    }
}

impl fmt::Display for SymmetryElementKindConversionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SymmetryElementKindConversionError with message: {}",
            &self.0
        )
    }
}

impl std::error::Error for SymmetryElementKindConversionError {}

impl TryInto<geometry::ImproperRotationKind> for SymmetryElementKind {
    type Error = SymmetryElementKindConversionError;

    fn try_into(self) -> Result<geometry::ImproperRotationKind, Self::Error> {
        match self {
            Self::Proper(_) => Err(SymmetryElementKindConversionError(
                "Unable to convert a proper element to an `ImproperRotationKind`.".to_string(),
            )),
            Self::ImproperMirrorPlane(_) => Ok(geometry::ImproperRotationKind::MirrorPlane),
            Self::ImproperInversionCentre(_) => Ok(geometry::ImproperRotationKind::InversionCentre),
        }
    }
}

impl fmt::Display for SymmetryElementKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Proper(au) => match au {
                None => write!(f, "Proper"),
                Some(AntiunitaryKind::TimeReversal) => write!(f, "Time-reversed proper"),
                Some(AntiunitaryKind::ComplexConjugation) => write!(f, "Complex-conjugated proper"),
            },
            Self::ImproperMirrorPlane(au) => match au {
                None => write!(f, "Improper (mirror-plane convention)"),
                Some(AntiunitaryKind::TimeReversal) => {
                    write!(f, "Time-reversed improper (mirror-plane convention)")
                }
                Some(AntiunitaryKind::ComplexConjugation) => {
                    write!(f, "Complex-conjugated improper (mirror-plane convention)")
                }
            },
            Self::ImproperInversionCentre(au) => match au {
                None => write!(f, "Improper (inversion-centre convention)"),
                Some(AntiunitaryKind::TimeReversal) => {
                    write!(f, "Time-reversed improper (inversion-centre convention)")
                }
                Some(AntiunitaryKind::ComplexConjugation) => {
                    write!(
                        f,
                        "Complex-conjugated improper (inversion-centre convention)"
                    )
                }
            },
        }
    }
}

/// Structure for storing and managing symmetry elements.
///
/// Each symmetry element is a geometrical object in $`\mathbb{R}^3`$ that encodes the following
/// pieces of information:
///
/// * the axis of rotation $`\hat{\mathbf{n}}`$,
/// * the angle of rotation $`\phi`$,
/// * the associated improper operation, if any, and
/// * the associated antiunitary operation, if any.
///
/// These pieces of information can be stored in the following representation of a symmetry element
/// $`\hat{g}`$:
///
/// ```math
/// \hat{g} = \hat{\alpha} \hat{\gamma} \hat{C}_n^k,
/// ```
///
/// where
/// * $`n \in \mathbb{N}_{+}`$, $`k \in \mathbb{Z}/n\mathbb{Z}`$ such that
/// $`\lfloor -n/2 \rfloor < k \le \lfloor n/2 \rfloor`$,
/// * $`\hat{\gamma}`$ is either the identity $`\hat{e}`$, the inversion operation $`\hat{i}`$, or
/// a reflection operation $`\hat{\sigma}`$ perpendicular to the axis of rotation,
/// * $`\hat{\alpha}`$ is either the identity $`\hat{e}`$, the complex conjugation $`\hat{K}`$, or
/// the time reversal $`\hat{\theta}`$.
///
/// We shall refer to $`\hat{C}_n^k`$ as the *unitary proper rotation part*, $`\hat{\gamma}`$ the
/// *improper rotation part*, and $`\hat{\alpha}`$ the *antiunitary part* of the symmetry element.
///
/// With this definition, the above pieces of information required to specify a geometrical symmetry
/// element are given as follows:
///
/// * the axis of rotation $`\hat{\mathbf{n}}`$ is given by the axis of $`\hat{C}_n^k`$,
/// * the angle of rotation $`\phi = 2\pi k/n \in (-\pi, \pi]`$,
/// * the improper contribution $`\hat{\gamma}`$,
/// * the antiunitary contribution $`\hat{\alpha}`$.
///
/// This definition also allows the unitary part of $`\hat{g}`$ (*i.e.* the composition
/// $`\hat{\gamma} \hat{C}_n^k`$) to be interpreted as an element of either $`\mathsf{O}(3)`$ or
/// $`\mathsf{SU}'(2)`$, which means that the unitary part of $`\hat{g}`$ is also a symmetry
/// operation in the corresponding group, and a rather special one that can be used to generate
/// other symmetry operations of the group. $`\hat{g}`$ thus serves as a bridge between molecular
/// symmetry and abstract group theory.
///
/// There is one small caveat: for infinite-order elements, $`n`$ and $`k`$ can no longer be used
/// to give the angle of rotation. There must thus be a mechanism to allow for infinite-order
/// elements to be interpreted as an arbitrary finite-order one. An explicit specification of the
/// angle of rotation $`\phi`$ seems to be the best way to do this. In other words, the angle of
/// rotation of each element is specified by either a tuple of integers $`(k, n)`$ or a
/// floating-point number $`\phi`$.
#[derive(Builder, Clone, Serialize, Deserialize)]
pub struct SymmetryElement {
    /// The rotational order $`n`$ of the unitary proper rotation part of the symmetry element. This
    /// can be finite or infinite, and will determine whether the proper power is `None` or
    /// contains an integer value.
    ///
    /// The unitary proper rotation does not include any additional unitary rotations introduced by
    /// the antiunitary part of the symmetry element (*e.g.* time reversal).
    #[builder(setter(name = "proper_order"))]
    raw_proper_order: ElementOrder,

    /// The power $`k \in \mathbb{Z}/n\mathbb{Z}`$ of the unitary proper rotation part of the
    /// symmetry element such that $`\lfloor -n/2 \rfloor < k <= \lfloor n/2 \rfloor`$. This is only
    /// defined if [`Self::raw_proper order`] is finite.
    ///
    /// The unitary proper rotation does not include any additional unitary rotations introduced by
    /// the antiunitary part of the symmetry element (*e.g.* time reversal).
    #[builder(setter(custom, name = "proper_power"), default = "None")]
    raw_proper_power: Option<i32>,

    /// The normalised axis of the unitary proper rotation part of the symmetry element whose
    /// direction is as specified at construction.
    #[builder(setter(custom))]
    raw_axis: Vector3<f64>,

    /// The spatial and antiunitary kind of the symmetry element.
    #[builder(default = "SymmetryElementKind::Proper(None)")]
    kind: SymmetryElementKind,

    /// The rotation group in which the unitary proper rotation part of the symmetry element shall
    /// be interpreted.
    rotation_group: RotationGroup,

    /// A boolean indicating whether the symmetry element is a generator of the group to which it
    /// belongs.
    #[builder(default = "false")]
    generator: bool,

    /// A threshold for approximate equality comparisons.
    #[builder(setter(custom))]
    threshold: f64,

    /// An additional superscript for distinguishing symmetry elements.
    #[builder(default = "String::new()")]
    pub(crate) additional_superscript: String,

    /// An additional subscript for distinguishing symmetry elements.
    #[builder(default = "String::new()")]
    pub(crate) additional_subscript: String,

    /// The fraction $`k/n \in (-1/2, 1/2]`$ of the unitary proper rotation, represented exactly
    /// for hashing and comparison purposes.
    ///
    /// The unitary proper rotation does not include any additional unitary rotations introduced by
    /// the antiunitary part of the symmetry element (*e.g.* time reversal).
    ///
    /// This is not defined for infinite-order elements and cannot be set arbitrarily.
    #[builder(setter(skip), default = "self.calc_proper_fraction()")]
    proper_fraction: Option<F>,

    /// The normalised proper angle $`\phi \in (-\pi, \pi]`$ corresponding to the unitary proper
    /// rotation.
    ///
    /// The unitary proper rotation does not include any additional unitary rotations introduced by
    /// the antiunitary part of the symmetry element (*e.g.* time reversal).
    ///
    /// This can be set arbitrarily only for infinite-order elements.
    #[builder(setter(custom), default = "self.calc_proper_angle()")]
    proper_angle: Option<f64>,
}

impl SymmetryElementBuilder {
    /// Sets the power of the unitary proper rotation part of the element.
    ///
    /// # Arguments
    ///
    /// * `prop_pow` - A proper power to be set. This will be folded into the interval
    /// $`(\lfloor -n/2 \rfloor, \lfloor n/2 \rfloor]`$.
    pub fn proper_power(&mut self, prop_pow: i32) -> &mut Self {
        let raw_proper_order = self
            .raw_proper_order
            .as_ref()
            .expect("Proper order has not been set.");
        self.raw_proper_power = match raw_proper_order {
            ElementOrder::Int(io) => {
                let io_i32 =
                    i32::try_from(*io).expect("Unable to convert the integer order to `i32`.");
                let residual = prop_pow.rem_euclid(io_i32);
                if residual > io_i32.div_euclid(2) {
                    Some(Some(residual - io_i32))
                } else {
                    Some(Some(residual))
                }
            }
            ElementOrder::Inf => None,
        };
        self
    }

    /// Sets the rotation angle of the unitary proper rotation part of the infinite-order element.
    ///
    /// # Arguments
    ///
    /// * `ang` - A proper rotation angle to be set. This will be folded into the interval
    /// $`(-\pi, \pi]`$.
    ///
    /// # Panics
    ///
    /// Panics when `self` is of finite order.
    pub fn proper_angle(&mut self, ang: f64) -> &mut Self {
        let proper_order = self
            .raw_proper_order
            .as_ref()
            .expect("Proper order has not been set.");
        self.proper_angle = match proper_order {
            ElementOrder::Int(_) => panic!(
                "Arbitrary proper rotation angles can only be set for infinite-order elements."
            ),
            ElementOrder::Inf => {
                let (normalised_rotation_angle, _) = geometry::normalise_rotation_angle(
                    ang,
                    self.threshold.expect("Threshold value has not been set."),
                );
                Some(Some(normalised_rotation_angle))
            }
        };
        self
    }

    /// Sets the raw axis of the element.
    ///
    /// # Arguments
    ///
    /// * `axs` - The raw axis which will be normalised.
    pub fn raw_axis(&mut self, axs: Vector3<f64>) -> &mut Self {
        let thresh = self.threshold.expect("Threshold value has not been set.");
        if approx::abs_diff_eq!(axs.norm(), 1.0, epsilon = thresh) {
            self.raw_axis = Some(axs);
        } else {
            log::warn!("Axis not normalised. Normalising...");
            self.raw_axis = Some(axs.normalize());
        }
        self
    }

    /// Sets the comparison threshold of the element.
    ///
    /// # Arguments
    ///
    /// * `thresh` - The comparison threshold..
    pub fn threshold(&mut self, thresh: f64) -> &mut Self {
        if thresh >= 0.0 {
            self.threshold = Some(thresh);
        } else {
            log::error!(
                "Threshold value `{}` is invalid. Threshold must be non-negative.",
                thresh
            );
            self.threshold = None;
        }
        self
    }

    /// Calculates the fraction $`k/n \in (-1/2, 1/2]`$ of the unitary proper rotation, represented
    /// exactly for hashing and comparison purposes.
    ///
    /// The unitary proper rotation does not include any additional unitary rotations introduced by
    /// the antiunitary part of the symmetry element (*e.g.* time reversal).
    ///
    /// This is not defined for infinite-order elements and cannot be set arbitrarily.
    fn calc_proper_fraction(&self) -> Option<F> {
        let raw_proper_order = self
            .raw_proper_order
            .as_ref()
            .expect("Proper order has not been set.");
        match raw_proper_order {
            ElementOrder::Int(io) => {
                // The generating element has a proper fraction, pp/n.
                //
                // If pp/n > 1/2, we seek a positive integer x such that
                //  -1/2 < pp/n - x <= 1/2.
                // It turns out that x ∈ [pp/n - 1/2, pp/n + 1/2).
                //
                // If pp/n <= -1/2, we seek a positive integer x such that
                //  -1/2 < pp/n + x <= 1/2.
                // It turns out that x ∈ (-pp/n - 1/2, -pp/n + 1/2].
                //
                // x is then used to bring pp/n back into the (-1/2, 1/2] interval.
                //
                // See S.L. Altmann, Rotations, Quaternions, and Double Groups (Dover
                // Publications, Inc., New York, 2005) for further information.
                let pp = self
                    .raw_proper_power
                    .expect("Proper power has not been set.")
                    .expect("No proper powers found.");
                let total_proper_fraction = if pp >= 0 {
                    F::new(pp.unsigned_abs(), *io)
                } else {
                    F::new_neg(pp.unsigned_abs(), *io)
                };
                Some(total_proper_fraction)
            }
            ElementOrder::Inf => None,
        }
    }

    /// Calculates the normalised proper angle $`\phi \in (-\pi, \pi]`$ corresponding to the unitary
    /// proper rotation.
    ///
    /// The unitary proper rotation does not include any additional unitary rotations introduced by
    /// the antiunitary part of the symmetry element (*e.g.* time reversal).
    ///
    /// This can be set arbitrarily only for infinite-order elements.
    fn calc_proper_angle(&self) -> Option<f64> {
        let raw_proper_order = self
            .raw_proper_order
            .as_ref()
            .expect("Proper order has not been set.");
        match raw_proper_order {
            ElementOrder::Int(io) => {
                let pp = self
                    .raw_proper_power
                    .expect("Proper power has not been set.")
                    .expect("No proper powers found.");
                let total_proper_fraction = if pp >= 0 {
                    F::new(pp.unsigned_abs(), *io)
                } else {
                    F::new_neg(pp.unsigned_abs(), *io)
                };
                Some(
                    total_proper_fraction
                        .to_f64()
                        .expect("Unable to convert the proper fraction to `f64`.")
                        * 2.0
                        * std::f64::consts::PI,
                )
            }
            ElementOrder::Inf => self.proper_angle.unwrap_or(None),
        }
    }
}

impl SymmetryElement {
    /// Returns a builder to construct a new symmetry element.
    ///
    /// # Returns
    ///
    /// A builder to construct a new symmetry element.
    #[must_use]
    pub fn builder() -> SymmetryElementBuilder {
        SymmetryElementBuilder::default()
    }

    /// Returns the raw order of the unitary proper rotation. This might not be equal to the value of
    /// $`n`$ if the fraction $`k/n`$ has been reduced.
    pub fn raw_proper_order(&self) -> &ElementOrder {
        &self.raw_proper_order
    }

    /// Returns the raw power of the unitary proper rotation. This might not be equal to the value of
    /// $`k`$ if the fraction $`k/n`$ has been reduced.
    pub fn raw_proper_power(&self) -> Option<&i32> {
        self.raw_proper_power.as_ref()
    }

    /// Returns the raw axis of the unitary proper rotation.
    pub fn raw_axis(&self) -> &Vector3<f64> {
        &self.raw_axis
    }

    /// Returns the axis of the unitary proper rotation in the standard positive hemisphere.
    pub fn standard_positive_axis(&self) -> Vector3<f64> {
        geometry::get_standard_positive_pole(&self.raw_axis, self.threshold)
    }

    /// Returns the axis of the unitary proper rotation multiplied by the sign of the rotation angle.
    /// If the proper rotation is a binary rotation, then the positive axis is always returned.
    pub fn signed_axis(&self) -> Vector3<f64> {
        let au = self.antiunitary_part();
        if self.is_o3_binary_rotation_axis(au) || self.is_o3_mirror_plane(au) {
            self.standard_positive_axis()
        } else {
            self.proper_fraction
                .map(|frac| {
                    frac.signum()
                        .to_f64()
                        .expect("Unable to obtain the sign of the proper fraction.")
                })
                .or_else(|| self.proper_angle.map(|proper_angle| proper_angle.signum())).map(|signum| signum * self.raw_axis)
                .unwrap_or_else(|| {
                    log::warn!("No rotation signs could be obtained. The positive axis will be used for the signed axis.");
                    self.standard_positive_axis()
                })
        }
    }

    /// Returns the pole of the unitary proper rotation part of the element while leaving any improper
    /// and antiunitary contributions intact.
    ///
    /// If the unitary proper rotation part if a binary rotation, the pole is always in the standard
    /// positive hemisphere.
    ///
    /// # Returns
    ///
    /// The position vector of the proper rotation pole.
    pub fn proper_rotation_pole(&self) -> Vector3<f64> {
        match *self.raw_proper_order() {
            ElementOrder::Int(_) => {
                let frac_1_2 = F::new(1u32, 2u32);
                let proper_fraction = self.proper_fraction.expect("No proper fractions found.");
                if proper_fraction == frac_1_2 {
                    // Binary proper rotations
                    geometry::get_standard_positive_pole(&self.raw_axis, self.threshold)
                } else if proper_fraction > F::zero() {
                    // Positive proper rotation angles
                    self.raw_axis
                } else if proper_fraction < F::zero() {
                    // Negative proper rotation angles
                    -self.raw_axis
                } else {
                    // Identity or inversion
                    assert!(proper_fraction.is_zero());
                    Vector3::zeros()
                }
            }
            ElementOrder::Inf => {
                if approx::abs_diff_eq!(
                    self.proper_angle.expect("No proper angles found."),
                    std::f64::consts::PI,
                    epsilon = self.threshold
                ) {
                    // Binary proper rotations
                    geometry::get_standard_positive_pole(&self.raw_axis, self.threshold)
                } else if approx::abs_diff_ne!(
                    self.proper_angle.expect("No proper angles found."),
                    0.0,
                    epsilon = self.threshold
                ) {
                    self.proper_angle.expect("No proper angles found.").signum() * self.raw_axis
                } else {
                    approx::assert_abs_diff_eq!(
                        self.proper_angle.expect("No proper angles found."),
                        0.0,
                        epsilon = self.threshold
                    );
                    Vector3::zeros()
                }
            }
        }
    }

    /// Returns the unitary proper fraction for this element, if any.
    ///
    /// The element lacks a unitary proper fraction if it is infinite-order.
    pub fn proper_fraction(&self) -> Option<&F> {
        self.proper_fraction.as_ref()
    }

    /// Returns the unitary proper angle for this element, if any.
    ///
    /// The element lacks a unitary proper angle if it is infinite-order and the rotation angle has
    /// not been set.
    pub fn proper_angle(&self) -> Option<f64> {
        self.proper_angle
    }

    /// Returns the spatial and antiunitary kind of this element.
    pub fn kind(&self) -> &SymmetryElementKind {
        &self.kind
    }

    /// Returns the rotation group and possibly the identity-connected homotopy class in which the
    /// proper rotation part of this element is to be interpreted.
    pub fn rotation_group(&self) -> &RotationGroup {
        &self.rotation_group
    }

    /// Returns a boolean indicating if the element is a generator of a group.
    pub fn is_generator(&self) -> bool {
        self.generator
    }

    /// Returns the threshold for approximate comparisons.
    pub fn threshold(&self) -> f64 {
        self.threshold
    }

    /// Checks if the symmetry element contains a time-reversal operator as the antiunitary part.
    ///
    /// # Returns
    ///
    /// A boolean indicating if the symmetry element contains a time-reversal operator as the
    /// antiunitary part.
    #[must_use]
    pub fn contains_time_reversal(&self) -> bool {
        self.kind.contains_time_reversal()
    }

    /// Indicates if an antiunitary operation is associated with this element.
    #[must_use]
    pub fn contains_antiunitary(&self) -> bool {
        self.kind.contains_antiunitary()
    }

    /// Returns the antiunitary part of the element, if any.
    ///
    /// # Returns
    ///
    /// Returns `None` if the symmetry element has no antiunitary parts, or `Some` wrapping around
    /// the antiunitary kind if the symmetry element contains an antiunitary part.
    pub fn antiunitary_part(&self) -> Option<AntiunitaryKind> {
        match self.kind {
            SymmetryElementKind::Proper(au)
            | SymmetryElementKind::ImproperMirrorPlane(au)
            | SymmetryElementKind::ImproperInversionCentre(au) => au,
        }
    }

    /// Checks if the unitary proper rotation part of the element is in $`\mathsf{SU}(2)`$.
    #[must_use]
    pub fn is_su2(&self) -> bool {
        self.rotation_group.is_su2()
    }

    /// Checks if the unitary proper rotation part of the element is in $`\mathsf{SU}(2)`$ and
    /// connected to the identity via a homotopy path of class 1.
    ///
    /// See S.L. Altmann, Rotations, Quaternions, and Double Groups (Dover Publications, Inc., New
    /// York, 2005) for further information.
    #[must_use]
    pub fn is_su2_class_1(&self) -> bool {
        self.rotation_group.is_su2_class_1()
    }

    /// Checks if the spatial part of the symmetry element is proper and has the specified
    /// antiunitary attribute.
    ///
    /// More specifically, this checks if the combination $`\hat{\gamma} \hat{C}_n^k`$ is proper,
    /// and if $`\hat{\alpha}`$ is as specified by `au`.
    ///
    /// # Arguments
    ///
    /// * `au` - An `Option` for the desired antiunitary kind.
    ///
    /// # Returns
    ///
    /// A boolean indicating if the symmetry element is proper and has the specified antiunitary
    /// attribute.
    #[must_use]
    pub fn is_o3_proper(&self, au: Option<AntiunitaryKind>) -> bool {
        self.kind == SymmetryElementKind::Proper(au)
    }

    /// Checks if the symmetry element is spatially an identity element and has the specified
    /// antiunitary attribute.
    ///
    /// More specifically, this checks if the combination $`\hat{\gamma} \hat{C}_n^k`$ is the
    /// identity, and if $`\hat{\alpha}`$ is as specified by `au`. Any rotation in $`\hat{\alpha}`$
    /// will not be absorbed into $`\hat{C}_n^k`$.
    ///
    /// # Arguments
    ///
    /// * `au` - An `Option` for the desired antiunitary kind.
    ///
    /// # Returns
    ///
    /// A boolean indicating if this symmetry element is spatially the identity and has the
    /// specified antiunitary attribute.
    #[must_use]
    pub fn is_o3_identity(&self, au: Option<AntiunitaryKind>) -> bool {
        self.kind == SymmetryElementKind::Proper(au)
            && self
                .proper_fraction
                .map(|frac| frac.is_zero())
                .or_else(|| {
                    self.proper_angle.map(|proper_angle| {
                        approx::abs_diff_eq!(proper_angle, 0.0, epsilon = self.threshold,)
                    })
                })
                .unwrap_or(false)
    }

    /// Checks if the symmetry element is spatially an inversion centre and has the specified
    /// antiunitary attribute.
    ///
    /// More specifically, this checks if the combination $`\hat{\gamma} \hat{C}_n^k`$ is the
    /// spatial inversion, and if $`\hat{\alpha}`$ is as specified by `au`. Any rotation in
    /// $`\hat{\alpha}`$ will not be absorbed into $`\hat{C}_n^k`$.
    ///
    /// # Arguments
    ///
    /// * `au` - An `Option` for the desired antiunitary kind.
    ///
    /// # Returns
    ///
    /// A boolean indicating if this symmetry element is spatially an inversion centre and has the
    /// specified antiunitary attribute.
    #[must_use]
    pub fn is_o3_inversion_centre(&self, au: Option<AntiunitaryKind>) -> bool {
        match self.kind {
            SymmetryElementKind::ImproperMirrorPlane(sig_au) => {
                sig_au == au
                    && self
                        .proper_fraction
                        .map(|frac| frac == F::new(1u32, 2u32))
                        .or_else(|| {
                            self.proper_angle.map(|proper_angle| {
                                approx::abs_diff_eq!(
                                    proper_angle,
                                    std::f64::consts::PI,
                                    epsilon = self.threshold,
                                )
                            })
                        })
                        .unwrap_or(false)
            }
            SymmetryElementKind::ImproperInversionCentre(inv_au) => {
                inv_au == au
                    && self
                        .proper_fraction
                        .map(|frac| frac.is_zero())
                        .or_else(|| {
                            self.proper_angle.map(|proper_angle| {
                                approx::abs_diff_eq!(proper_angle, 0.0, epsilon = self.threshold,)
                            })
                        })
                        .unwrap_or(false)
            }
            _ => false,
        }
    }

    /// Checks if the symmetry element is spatially a binary rotation axis and has the specified
    /// antiunitary attribute.
    ///
    /// More specifically, this checks if the combination $`\hat{\gamma} \hat{C}_n^k`$ is a binary
    /// rotation, and if $`\hat{\alpha}`$ is as specified by `au`. Any rotation in $`\hat{\alpha}`$
    /// will not be absorbed into $`\hat{C}_n^k`$.
    ///
    /// # Arguments
    ///
    /// * `au` - An `Option` for the desired antiunitary kind.
    ///
    /// # Returns
    ///
    /// A boolean indicating if this symmetry element is spatially a binary rotation axis and has
    /// the specified antiunitary attribute.
    #[must_use]
    pub fn is_o3_binary_rotation_axis(&self, au: Option<AntiunitaryKind>) -> bool {
        self.kind == SymmetryElementKind::Proper(au)
            && self
                .proper_fraction
                .map(|frac| frac == F::new(1u32, 2u32))
                .or_else(|| {
                    self.proper_angle.map(|proper_angle| {
                        approx::abs_diff_eq!(
                            proper_angle,
                            std::f64::consts::PI,
                            epsilon = self.threshold,
                        )
                    })
                })
                .unwrap_or(false)
    }

    /// Checks if the symmetry element is spatially a mirror plane and has the specified
    /// antiunitary attribute.
    ///
    /// More specifically, this checks if the combination $`\hat{\gamma} \hat{C}_n^k`$ is a
    /// reflection, and if $`\hat{\alpha}`$ is as specified by `au`. Any rotation in
    /// $`\hat{\alpha}`$ will not be absorbed into $`\hat{C}_n^k`$.
    ///
    /// # Arguments
    ///
    /// * `au` - An `Option` for the desired antiunitary kind.
    ///
    /// # Returns
    ///
    /// A boolean indicating if this symmetry element is spatially a mirror plane and has the
    /// specified antiunitary attribute.
    #[must_use]
    pub fn is_o3_mirror_plane(&self, au: Option<AntiunitaryKind>) -> bool {
        match self.kind {
            SymmetryElementKind::ImproperMirrorPlane(sig_au) => {
                sig_au == au
                    && self
                        .proper_fraction
                        .map(|frac| frac.is_zero())
                        .or_else(|| {
                            self.proper_angle.map(|proper_angle| {
                                approx::abs_diff_eq!(proper_angle, 0.0, epsilon = self.threshold,)
                            })
                        })
                        .unwrap_or(false)
            }
            SymmetryElementKind::ImproperInversionCentre(inv_au) => {
                inv_au == au
                    && self
                        .proper_fraction
                        .map(|frac| frac == F::new(1u32, 2u32))
                        .or_else(|| {
                            self.proper_angle.map(|proper_angle| {
                                approx::abs_diff_eq!(
                                    proper_angle,
                                    std::f64::consts::PI,
                                    epsilon = self.threshold,
                                )
                            })
                        })
                        .unwrap_or(false)
            }
            _ => false,
        }
    }

    /// Returns the full symbol for this symmetry element, which does not classify certain
    /// improper rotation axes into inversion centres or mirror planes, but which does simplify
    /// the power/order ratio, and which displays only the absolute value of the power since
    /// symmetry elements do not distinguish senses of rotations since rotations of oposite
    /// directions are inverses of each other, both of which must exist in the group.
    ///
    /// Some additional symbols that can be unconventional include:
    ///
    /// * `θ`: time reversal,
    /// * `K`: complex conjugation,
    /// * `(Σ)`: the spatial part is in homotopy class 0 of $`\mathsf{SU}'(2)`$,
    /// * `(QΣ)`: the spatial part is in homotopy class 1 of $`\mathsf{SU}'(2)`$.
    ///
    /// See [`RotationGroup`] for further information.
    ///
    /// # Returns
    ///
    /// The full symbol for this symmetry element.
    #[must_use]
    pub fn get_full_symbol(&self) -> String {
        let au_sym = if self.kind.contains_time_reversal() {
            "θ·"
        } else if self.kind.contains_complex_conjugation() {
            "K·"
        } else {
            ""
        };
        let main_symbol: String = match self.kind {
            SymmetryElementKind::Proper(_) => {
                format!("{au_sym}C")
            }
            SymmetryElementKind::ImproperMirrorPlane(_) => {
                if *self.raw_proper_order() != ElementOrder::Inf
                    && (*self
                        .proper_fraction
                        .expect("No proper fractions found for a finite-order element.")
                        .numer()
                        .expect("Unable to extract the numerator of the proper fraction.")
                        == 1
                        || self
                            .proper_fraction
                            .expect("No proper fractions found for a finite-order element.")
                            .is_zero())
                {
                    format!("{au_sym}S")
                } else {
                    format!("{au_sym}σC")
                }
            }
            SymmetryElementKind::ImproperInversionCentre(_) => {
                if *self.raw_proper_order() != ElementOrder::Inf
                    && (*self
                        .proper_fraction
                        .expect("No proper fractions found for a finite-order element.")
                        .numer()
                        .expect("Unable to extract the numerator of the proper fraction.")
                        == 1
                        || self
                            .proper_fraction
                            .expect("No proper fractions found for a finite-order element.")
                            .is_zero())
                {
                    format!("{au_sym}Ṡ")
                } else {
                    format!("{au_sym}iC")
                }
            }
        };

        let su2_sym = if self.is_su2_class_1() {
            "(QΣ)"
        } else if self.is_su2() {
            "(Σ)"
        } else {
            ""
        };

        if let Some(proper_fraction) = self.proper_fraction {
            if proper_fraction.is_zero() {
                format!("{main_symbol}1{su2_sym}")
            } else {
                let proper_order = proper_fraction
                    .denom()
                    .expect("Unable to extract the denominator of the proper fraction.")
                    .to_string();
                let proper_power = {
                    let pow = *proper_fraction
                        .numer()
                        .expect("Unable to extract the numerator of the proper fraction.");
                    if pow > 1 {
                        format!("^{pow}")
                    } else {
                        String::new()
                    }
                };
                format!("{main_symbol}{proper_order}{proper_power}{su2_sym}")
            }
        } else {
            assert_eq!(*self.raw_proper_order(), ElementOrder::Inf);
            let proper_angle = if let Some(proper_angle) = self.proper_angle {
                format!("({:+.3})", proper_angle.abs())
            } else {
                String::new()
            };
            format!(
                "{main_symbol}{}{proper_angle}{su2_sym}",
                self.raw_proper_order()
            )
        }
    }

    /// Returns the simplified symbol for this symmetry element, which classifies special symmetry
    /// elements (identity, inversion centre, time reversal, mirror planes), and which simplifies
    /// the power/order ratio and displays only the absolute value of the power since symmetry
    /// elements do not distinguish senses of rotations, as rotations of oposite directions are
    /// inverses of each other, both of which must exist in the group.
    ///
    /// # Returns
    ///
    /// The simplified symbol for this symmetry element.
    #[must_use]
    pub fn get_simplified_symbol(&self) -> String {
        let (main_symbol, needs_power) = match self.kind {
            SymmetryElementKind::Proper(au) => {
                if self.is_o3_identity(au) {
                    match au {
                        None => ("E".to_owned(), false),
                        Some(AntiunitaryKind::TimeReversal) => ("θ".to_owned(), false),
                        Some(AntiunitaryKind::ComplexConjugation) => ("K".to_owned(), false),
                    }
                } else {
                    let au_sym = match au {
                        None => "",
                        Some(AntiunitaryKind::TimeReversal) => "θ·",
                        Some(AntiunitaryKind::ComplexConjugation) => "K·",
                    };
                    (format!("{au_sym}C"), true)
                }
            }
            SymmetryElementKind::ImproperMirrorPlane(au) => {
                let au_sym = match au {
                    None => "",
                    Some(AntiunitaryKind::TimeReversal) => "θ·",
                    Some(AntiunitaryKind::ComplexConjugation) => "K·",
                };
                if self.is_o3_mirror_plane(au) {
                    (format!("{au_sym}σ"), false)
                } else if self.is_o3_inversion_centre(au) {
                    (format!("{au_sym}i"), false)
                } else if *self.raw_proper_order() == ElementOrder::Inf
                    || *self
                        .proper_fraction
                        .expect("No proper fractions found for a finite-order element.")
                        .numer()
                        .expect("Unable to extract the numerator of the proper fraction.")
                        == 1
                {
                    (format!("{au_sym}S"), false)
                } else {
                    (format!("{au_sym}σC"), true)
                }
            }
            SymmetryElementKind::ImproperInversionCentre(au) => {
                let au_sym = match au {
                    None => "",
                    Some(AntiunitaryKind::TimeReversal) => "θ·",
                    Some(AntiunitaryKind::ComplexConjugation) => "K·",
                };
                if self.is_o3_mirror_plane(au) {
                    (format!("{au_sym}σ"), false)
                } else if self.is_o3_inversion_centre(au) {
                    (format!("{au_sym}i"), false)
                } else if *self.raw_proper_order() == ElementOrder::Inf
                    || *self
                        .proper_fraction
                        .expect("No proper fractions found for a finite-order element.")
                        .numer()
                        .expect("Unable to extract the numerator of the proper fraction.")
                        == 1
                {
                    (format!("{au_sym}Ṡ"), false)
                } else {
                    (format!("{au_sym}iC"), true)
                }
            }
        };

        let su2_sym = if self.is_su2_class_1() {
            "(QΣ)"
        } else if self.is_su2() {
            "(Σ)"
        } else {
            ""
        };

        if let Some(proper_fraction) = self.proper_fraction {
            let au = self.antiunitary_part();
            let proper_order = if self.is_o3_identity(au)
                || self.is_o3_inversion_centre(au)
                || self.is_o3_mirror_plane(au)
            {
                String::new()
            } else {
                proper_fraction
                    .denom()
                    .expect("Unable to extract the denominator of the proper fraction.")
                    .to_string()
            };

            let proper_power = if needs_power {
                let pow = *proper_fraction
                    .numer()
                    .expect("Unable to extract the numerator of the proper fraction.");
                if pow > 1 {
                    format!("^{pow}")
                } else {
                    String::new()
                }
            } else {
                String::new()
            };
            format!(
                "{main_symbol}{}{proper_order}{proper_power}{}{su2_sym}",
                self.additional_superscript, self.additional_subscript
            )
        } else {
            assert_eq!(*self.raw_proper_order(), ElementOrder::Inf);
            let proper_angle = if let Some(proper_angle) = self.proper_angle {
                format!("({:+.3})", proper_angle.abs())
            } else {
                String::new()
            };
            format!(
                "{main_symbol}{}{}{proper_angle}{}{su2_sym}",
                self.additional_superscript,
                *self.raw_proper_order(),
                self.additional_subscript
            )
        }
    }

    /// Returns the simplified symbol for this symmetry element, which classifies special symmetry
    /// elements (identity, inversion centre, mirror planes), and which simplifies the power/order
    /// ratio and displays only the absolute value of the power since symmetry elements do not
    /// distinguish senses of rotations. Rotations of oposite directions are inverses of each
    /// other, both of which must exist in the group.
    ///
    /// # Returns
    ///
    /// The simplified symbol for this symmetry element.
    #[must_use]
    pub fn get_simplified_symbol_signed_power(&self) -> String {
        let (main_symbol, needs_power) = match self.kind {
            SymmetryElementKind::Proper(au) => {
                if self.is_o3_identity(au) {
                    match au {
                        None => ("E".to_owned(), false),
                        Some(AntiunitaryKind::TimeReversal) => ("θ".to_owned(), false),
                        Some(AntiunitaryKind::ComplexConjugation) => ("K".to_owned(), false),
                    }
                } else {
                    let au_sym = match au {
                        None => "",
                        Some(AntiunitaryKind::TimeReversal) => "θ·",
                        Some(AntiunitaryKind::ComplexConjugation) => "K·",
                    };
                    (format!("{au_sym}C"), true)
                }
            }
            SymmetryElementKind::ImproperMirrorPlane(au) => {
                let au_sym = match au {
                    None => "",
                    Some(AntiunitaryKind::TimeReversal) => "θ·",
                    Some(AntiunitaryKind::ComplexConjugation) => "K·",
                };
                if self.is_o3_mirror_plane(au) {
                    (format!("{au_sym}σ"), false)
                } else if self.is_o3_inversion_centre(au) {
                    (format!("{au_sym}i"), false)
                } else if *self.raw_proper_order() == ElementOrder::Inf
                    || *self
                        .proper_fraction
                        .expect("No proper fractions found for a finite-order element.")
                        .numer()
                        .expect("Unable to extract the numerator of the proper fraction.")
                        == 1
                {
                    (format!("{au_sym}S"), true)
                } else {
                    (format!("{au_sym}σC"), true)
                }
            }
            SymmetryElementKind::ImproperInversionCentre(au) => {
                let au_sym = match au {
                    None => "",
                    Some(AntiunitaryKind::TimeReversal) => "θ·",
                    Some(AntiunitaryKind::ComplexConjugation) => "K·",
                };
                if self.is_o3_mirror_plane(au) {
                    (format!("{au_sym}σ"), false)
                } else if self.is_o3_inversion_centre(au) {
                    (format!("{au_sym}i"), false)
                } else if *self.raw_proper_order() == ElementOrder::Inf
                    || *self
                        .proper_fraction
                        .expect("No proper fractions found for a finite-order element.")
                        .numer()
                        .expect("Unable to extract the numerator of the proper fraction.")
                        == 1
                {
                    (format!("{au_sym}Ṡ"), true)
                } else {
                    (format!("{au_sym}iC"), true)
                }
            }
        };

        let su2_sym = if self.is_su2_class_1() {
            "(QΣ)"
        } else if self.is_su2() {
            "(Σ)"
        } else {
            ""
        };

        if let Some(proper_fraction) = self.proper_fraction {
            let au = self.antiunitary_part();
            let proper_order = if self.is_o3_identity(au)
                || self.is_o3_inversion_centre(au)
                || self.is_o3_mirror_plane(au)
            {
                String::new()
            } else {
                proper_fraction
                    .denom()
                    .expect("Unable to extract the denominator of the proper fraction.")
                    .to_string()
            };

            let proper_power = if needs_power {
                let pow = *proper_fraction
                    .numer()
                    .expect("Unable to extract the numerator of the proper fraction.");
                if !geometry::check_standard_positive_pole(
                    &self.proper_rotation_pole(),
                    self.threshold,
                ) {
                    format!("^(-{pow})")
                } else if pow > 1 {
                    format!("^{pow}")
                } else {
                    String::new()
                }
            } else {
                String::new()
            };
            format!(
                "{main_symbol}{}{proper_order}{proper_power}{}{su2_sym}",
                self.additional_superscript, self.additional_subscript
            )
        } else {
            assert_eq!(*self.raw_proper_order(), ElementOrder::Inf);
            let proper_angle = if let Some(proper_angle) = self.proper_angle {
                format!("({:+.3})", proper_angle.abs())
            } else {
                String::new()
            };
            format!(
                "{main_symbol}{}{}{proper_angle}{}{su2_sym}",
                self.additional_superscript,
                *self.raw_proper_order(),
                self.additional_subscript
            )
        }
    }

    /// Returns a copy of the current improper symmetry element that has been converted to the
    /// required improper kind. For $`\mathsf{SU}'(2)`$ elements, the conversion will be carried
    /// out in the same homotopy class. The antiunitary part will be kept unchanged.
    ///
    /// To convert between the two improper kinds, we essentially seek integers
    /// $`n, n' \in \mathbb{N}_{+}`$ and $`k \in \mathbb{Z}/n\mathbb{Z}`$,
    /// $`k' \in \mathbb{Z}/n'\mathbb{Z}`$, such that
    ///
    /// ```math
    /// \sigma C_n^k = i C_{n'}^{k'},
    /// ```
    ///
    /// where the axes of all involved elements are parallel. By noting that
    /// $`\sigma = i C_2`$ and that $`k`$ and $`k'`$ must have opposite signs, we can easily show
    /// that, for $`k \ge 0, k' < 0`$,
    ///
    /// ```math
    /// \begin{aligned}
    ///     n' &= \frac{2n}{\operatorname{gcd}(2n, n - 2k)},\\
    ///     k' &= -\frac{n - 2k}{\operatorname{gcd}(2n, n - 2k)},
    /// \end{aligned}
    /// ```
    ///
    /// whereas for $`k < 0, k' \ge 0`$,
    ///
    /// ```math
    /// \begin{aligned}
    ///     n' &= \frac{2n}{\operatorname{gcd}(2n, n + 2k)},\\
    ///     k' &= \frac{n + 2k}{\operatorname{gcd}(2n, n + 2k)}.
    /// \end{aligned}
    /// ```
    ///
    /// The above relations are self-inverse. It can be further shown that
    /// $`\operatorname{gcd}(n', k') = 1`$. Hence, for symmetry *element* conversions, we can simply
    /// take $`k' = 1`$. This is because a symmetry element plays the role of a generator, and the
    /// coprimality of $`n'`$ and $`k'`$ means that $`i C_{n'}^{1}`$ is as valid a generator as
    /// $`i C_{n'}^{k'}`$.
    ///
    /// # Arguments
    ///
    /// * `improper_kind` - The improper kind to which `self` is to be converted. There is no need
    /// to make sure the time reversal specification in `improper_kind` matches that of `self` as
    /// the conversion will take care of this.
    /// * `preserves_power` - Boolean indicating if the proper rotation power $`k'`$ should be
    /// preserved or should be set to $`1`$.
    ///
    /// # Returns
    ///
    /// A copy of the current improper symmetry element that has been converted.
    ///
    /// # Panics
    ///
    /// Panics when `self` is not an improper element, or when `improper_kind` is not one of the
    /// improper variants.
    #[must_use]
    pub fn convert_to_improper_kind(
        &self,
        improper_kind: &SymmetryElementKind,
        preserves_power: bool,
    ) -> Self {
        let au = self.antiunitary_part();
        assert!(
            !self.is_o3_proper(au),
            "Only improper elements can be converted."
        );
        let improper_kind = improper_kind.to_antiunitary(au);
        assert!(
            !matches!(improper_kind, SymmetryElementKind::Proper(_)),
            "`improper_kind` must be one of the improper variants."
        );

        // self.kind and improper_kind must now have the same antiunitary part.

        if self.kind == improper_kind {
            return self.clone();
        }

        let (dest_order, dest_proper_power) = match *self.raw_proper_order() {
            ElementOrder::Int(_) => {
                let proper_fraction = self
                    .proper_fraction
                    .expect("No proper fractions found for an element with integer order.");
                let n = *proper_fraction.denom().unwrap();
                let k = if proper_fraction.is_sign_negative() {
                    -i32::try_from(*proper_fraction.numer().unwrap_or_else(|| {
                        panic!("Unable to retrieve the numerator of {proper_fraction:?}.")
                    }))
                    .expect("Unable to convert the numerator of the proper fraction to `i32`.")
                } else {
                    i32::try_from(*proper_fraction.numer().unwrap_or_else(|| {
                        panic!("Unable to retrieve the numerator of {proper_fraction:?}.")
                    }))
                    .expect("Unable to convert the numerator of the proper fraction to `i32`.")
                };

                if k >= 0 {
                    // k >= 0, k2 < 0
                    let n_m_2k = n
                        .checked_sub(2 * k.unsigned_abs())
                        .expect("The value of `n - 2k` is negative.");
                    let n2 = ElementOrder::Int(2 * n / (gcd(2 * n, n_m_2k)));
                    let k2: i32 = if preserves_power {
                        -i32::try_from(n_m_2k / gcd(2 * n, n_m_2k))
                            .expect("Unable to convert `k'` to `i32`.")
                    } else {
                        1
                    };
                    (n2, k2)
                } else {
                    // k < 0, k2 >= 0
                    let n_p_2k = n
                        .checked_sub(2 * k.unsigned_abs())
                        .expect("The value of `n + 2k` is negative.");
                    let n2 = ElementOrder::Int(2 * n / (gcd(2 * n, n_p_2k)));
                    let k2: i32 = if preserves_power {
                        i32::try_from(n_p_2k / (gcd(2 * n, n_p_2k)))
                            .expect("Unable to convert `k'` to `i32`.")
                    } else {
                        1
                    };
                    (n2, k2)
                }
            }
            ElementOrder::Inf => (ElementOrder::Inf, 1),
        };

        match dest_order {
            ElementOrder::Int(_) => Self::builder()
                .threshold(self.threshold)
                .proper_order(dest_order)
                .proper_power(dest_proper_power)
                .raw_axis(self.raw_axis)
                .kind(improper_kind)
                .rotation_group(self.rotation_group.clone())
                .generator(self.generator)
                .additional_superscript(self.additional_superscript.clone())
                .additional_subscript(self.additional_subscript.clone())
                .build()
                .expect("Unable to construct a symmetry element."),

            ElementOrder::Inf => {
                if let Some(ang) = self.proper_angle {
                    Self::builder()
                        .threshold(self.threshold)
                        .proper_order(dest_order)
                        .proper_power(dest_proper_power)
                        .proper_angle(-std::f64::consts::PI + ang)
                        .raw_axis(self.raw_axis)
                        .kind(improper_kind)
                        .rotation_group(self.rotation_group.clone())
                        .generator(self.generator)
                        .additional_superscript(self.additional_superscript.clone())
                        .additional_subscript(self.additional_subscript.clone())
                        .build()
                        .expect("Unable to construct a symmetry element.")
                } else {
                    Self::builder()
                        .threshold(self.threshold)
                        .proper_order(dest_order)
                        .proper_power(dest_proper_power)
                        .raw_axis(self.raw_axis)
                        .kind(improper_kind)
                        .rotation_group(self.rotation_group.clone())
                        .generator(self.generator)
                        .additional_superscript(self.additional_superscript.clone())
                        .additional_subscript(self.additional_subscript.clone())
                        .build()
                        .expect("Unable to construct a symmetry element.")
                }
            }
        }
    }

    /// Converts the symmetry element to one with the desired time-reversal property.
    ///
    /// # Arguments
    ///
    /// * `tr` - A boolean indicating if time reversal is to be included.
    ///
    /// # Returns
    ///
    /// A new symmetry element with or without time reversal as indicated by `tr`.
    pub fn to_tr(&self, tr: bool) -> Self {
        let mut c_self = self.clone();
        c_self.kind = c_self.kind.to_tr(tr);
        c_self
    }

    /// Returns a copy of the current antiunitary symmetry element that has been converted to the
    /// required antiunitary kind. The improper part will be kept unchanged.
    ///
    /// # Arguments
    ///
    /// * `au` - The antiunitary kind to which `self` is to be converted.
    ///
    /// # Returns
    ///
    /// A copy of the current antiunitary symmetry element that has been converted.
    ///
    /// # Errors
    ///
    /// Errors when `self` is not an antiunitary element.
    pub fn convert_to_antiunitary_kind(&self, au: &AntiunitaryKind) -> Result<Self, anyhow::Error> {
        if let Some(self_au) = self.antiunitary_part() {
            if self_au == *au {
                Ok(self.clone())
            } else {
                ensure!(self.is_su2(), "The unitary proper rotation part of this element is not in SU(2) and therefore not able to support any time-reversal-induced rotation.");
                let inv_self_au = SymmetryElementKind::ImproperInversionCentre(Some(self_au));

                // gamma is either e or i
                let alpha_gamma_c = if self.is_o3_proper(Some(self_au)) {
                    self.clone()
                } else {
                    self.convert_to_improper_kind(&inv_self_au, false)
                };

                // `c_op` is the unitary proper rotation part
                let mut c = alpha_gamma_c.clone();
                c.kind = SymmetryElementKind::Proper(None);
                let c_op = SymmetryOperation::builder()
                    .generating_element(c)
                    .power(1)
                    .build()?;

                // `induced_c_op` is the extra rotation required for the conversion
                let r_pi_y = SymmetryElement::builder()
                    .threshold(c_op.generating_element.threshold)
                    .proper_order(ElementOrder::Int(2))
                    .proper_power(1)
                    .raw_axis(Vector3::y())
                    .kind(ROT)
                    .rotation_group(RotationGroup::SU2(true))
                    .build()?;
                let induced_c_op = match self_au {
                    AntiunitaryKind::TimeReversal => SymmetryOperation::builder()
                        .generating_element(r_pi_y)
                        .power(1)
                        .build()?,
                    AntiunitaryKind::ComplexConjugation => SymmetryOperation::builder()
                        .generating_element(r_pi_y)
                        .power(-1)
                        .build()?,
                };

                // `c_prime_op` is the resulting rotation after the conversion
                let c_prime_op = induced_c_op * c_op;

                let converted_alpha_gamma_c_prime = match self.kind {
                    SymmetryElementKind::Proper(_) => {
                        let mut converted_alpha_e_c_prime = c_prime_op.to_symmetry_element();
                        converted_alpha_e_c_prime.kind =
                            SymmetryElementKind::Proper(Some(au.clone()));
                        converted_alpha_e_c_prime
                    }
                    SymmetryElementKind::ImproperInversionCentre(_) => {
                        let mut converted_alpha_i_c_prime = c_prime_op.to_symmetry_element();
                        converted_alpha_i_c_prime.kind =
                            SymmetryElementKind::ImproperInversionCentre(Some(au.clone()));
                        converted_alpha_i_c_prime
                    }
                    SymmetryElementKind::ImproperMirrorPlane(_) => {
                        let mut converted_alpha_i_c_prime = c_prime_op.to_symmetry_element();
                        converted_alpha_i_c_prime.kind =
                            SymmetryElementKind::ImproperInversionCentre(Some(au.clone()));
                        let converted_alpha_sigma_c_prime_prime = converted_alpha_i_c_prime
                            .convert_to_improper_kind(
                                &SymmetryElementKind::ImproperMirrorPlane(Some(au.clone())),
                                false,
                            );
                        converted_alpha_sigma_c_prime_prime
                    }
                };
                Ok(converted_alpha_gamma_c_prime)
            }
        } else {
            Err(format_err!(
                "This element does not contain an antiunitary part."
            ))
        }
    }

    fn standardise(&self) -> Self {
        let au = self.antiunitary_part();
        let improper_conv = !self.is_o3_proper(au);
        let antiunitary_conv = self.contains_antiunitary();
        let su2 = self.is_su2();
        match (improper_conv, antiunitary_conv, su2) {
            (false, false, _) => self.clone(),
            (true, false, _) => self
                .convert_to_improper_kind(&SymmetryElementKind::ImproperInversionCentre(au), false),
            (false, true, true) => self
                .convert_to_antiunitary_kind(&K)
                .expect("Unable to convert to the complex conjugation convention."),
            (false, true, false) => self.clone(),
            (true, true, true) => self
                .convert_to_improper_kind(
                    &SymmetryElementKind::ImproperInversionCentre(Some(K)),
                    true,
                )
                .convert_to_antiunitary_kind(&K)
                .expect("Unable to convert to the complex conjugation convention."),
            (true, true, false) => self.convert_to_improper_kind(
                &SymmetryElementKind::ImproperInversionCentre(Some(K)),
                false,
            ),
        }
    }

    /// Convert the proper rotation of the current element to one in $`\mathsf{SU}(2)`$.
    ///
    /// # Arguments
    ///
    /// * `normal` - A boolean indicating whether the resultant $`\mathsf{SU}(2)`$ proper rotation
    /// is of homotopy class 0 (`true`) or 1 (`false`) when connected to the identity.
    ///
    /// # Returns
    ///
    /// A symmetry element in $`\mathsf{SU}(2)`$, or `None` if the current symmetry element
    /// is already in $`\mathsf{SU}(2)`$.
    pub fn to_su2(&self, normal: bool) -> Option<Self> {
        if self.is_su2() {
            None
        } else {
            let mut element = self.clone();
            element.rotation_group = RotationGroup::SU2(normal);
            Some(element)
        }
    }

    /// The closeness of the symmetry element's axis to one of the three space-fixed Cartesian axes.
    ///
    /// # Returns
    ///
    /// A tuple of two values:
    /// - A value $`\gamma \in [0, 1-1/\sqrt{3}]`$ indicating how close the axis is to one of the
    /// three Cartesian axes. The closer $`\gamma`$ is to $`0`$, the closer the alignment.
    /// - An index for the closest axis: `0` for $`z`$, `1` for $`y`$, `2` for $`x`$.
    ///
    /// # Panics
    ///
    /// Panics when $`\gamma`$ is outside the required closed interval $`[0, 1-1/\sqrt{3}]`$ by
    /// more than the threshold value in `self`.
    #[must_use]
    pub fn closeness_to_cartesian_axes(&self) -> (f64, usize) {
        let pos_axis = self.standard_positive_axis();
        let rev_pos_axis = Vector3::new(pos_axis[2], pos_axis[1], pos_axis[0]);
        let (amax_arg, amax_val) = rev_pos_axis.abs().argmax();
        let axis_closeness = 1.0 - amax_val;
        let thresh = self.threshold;
        assert!(
            -thresh <= axis_closeness && axis_closeness <= (1.0 - 1.0 / 3.0f64.sqrt() + thresh)
        );

        // closest axis: 0 for z, 1 for y, 2 for x
        // This is so that z axes are preferred.
        let closest_axis = amax_arg;
        (axis_closeness, closest_axis)
    }
}

impl fmt::Debug for SymmetryElement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let signed_axis = self.signed_axis();
        let x = if signed_axis[0].abs() < 5e-4 {
            0.000
        } else {
            signed_axis[0] + 0.0
        };
        let y = if signed_axis[1].abs() < 5e-4 {
            0.000
        } else {
            signed_axis[1] + 0.0
        };
        let z = if signed_axis[2].abs() < 5e-4 {
            0.000
        } else {
            signed_axis[2] + 0.0
        };
        write!(f, "{}({x:+.3}, {y:+.3}, {z:+.3})", self.get_full_symbol(),)
    }
}

impl fmt::Display for SymmetryElement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let au = self.antiunitary_part();
        if self.is_o3_identity(au) || self.is_o3_inversion_centre(au) {
            write!(f, "{}", self.get_simplified_symbol())
        } else {
            let signed_axis = self.signed_axis();
            let x = if signed_axis[0].abs() < 5e-4 {
                0.000
            } else {
                signed_axis[0] + 0.0
            };
            let y = if signed_axis[1].abs() < 5e-4 {
                0.000
            } else {
                signed_axis[1] + 0.0
            };
            let z = if signed_axis[2].abs() < 5e-4 {
                0.000
            } else {
                signed_axis[2] + 0.0
            };
            write!(
                f,
                "{}({x:+.3}, {y:+.3}, {z:+.3})",
                self.get_simplified_symbol(),
            )
        }
    }
}

impl PartialEq for SymmetryElement {
    /// Two symmetry elements are equal if and only if the following conditions are all satisfied:
    ///
    /// * they are both in the same rotation group and belong to the same homotopy class;
    /// * they are both proper or improper;
    /// * they are both unitary or antiunitary;
    /// * their axes are either parallel or anti-parallel;
    /// * their proper rotation angles have equal absolute values.
    ///
    /// For improper elements, proper rotation angles are taken in the inversion centre convention.
    /// For antiunitary elements, proper rotation angles are taken in the complex conjugation
    /// convention.
    ///
    /// Thus, symmetry element equality is less strict than symmetry operation equality. This is so
    /// that parallel or anti-parallel symmetry elements with the same spatial and time-reversal
    /// parities and angle of rotation are deemed identical, thus facilitating symmetry detection
    /// where one does not yet care much about directions of rotations.
    #[allow(clippy::too_many_lines)]
    fn eq(&self, other: &Self) -> bool {
        if self.is_su2() != other.is_su2() {
            // Different rotation groups for the rotation parts. This will not change in any
            // convention.
            return false;
        }

        if self.contains_antiunitary() != other.contains_antiunitary() {
            // Different antiunitarities. This will not change in any convention.
            return false;
        }

        let s_au = self.antiunitary_part();
        let o_au = other.antiunitary_part();

        if self.is_o3_proper(s_au) != other.is_o3_proper(o_au) {
            // Different spatial parities. This will not change in any convention.
            return false;
        }

        if self.is_o3_identity(s_au) && other.is_o3_identity(o_au) {
            // Both are spatial identity. The equality depends on whether s_au == o_au or not.
            return s_au == o_au && misc::calculate_hash(self) == misc::calculate_hash(other);
        }

        if self.is_o3_inversion_centre(s_au) && other.is_o3_inversion_centre(o_au) {
            // Both are spatial inversion centre. The equality depends on whether s_au == o_au or not.
            return s_au == o_au && misc::calculate_hash(self) == misc::calculate_hash(other);
        }

        let thresh = (self.threshold * other.threshold).sqrt();

        let std_self = self.standardise();
        let std_other = other.standardise();

        let result = {
            // Parallel or anti-parallel axes.
            let similar_poles = approx::relative_eq!(
                geometry::get_standard_positive_pole(&std_self.raw_axis, thresh),
                geometry::get_standard_positive_pole(&std_other.raw_axis, thresh),
                epsilon = thresh,
                max_relative = thresh
            );

            // Same angle of rotation (irrespective of signs).
            let similar_angles = match (*std_self.raw_proper_order(), *std_other.raw_proper_order())
            {
                (ElementOrder::Inf, ElementOrder::Inf) => {
                    match (std_self.proper_angle, std_other.proper_angle) {
                        (Some(s_angle), Some(o_angle)) => {
                            approx::relative_eq!(
                                s_angle.abs(),
                                o_angle.abs(),
                                epsilon = thresh,
                                max_relative = thresh
                            )
                        }
                        (None, None) => similar_poles,
                        _ => false,
                    }
                }
                (ElementOrder::Int(_), ElementOrder::Int(_)) => {
                    let s_proper_fraction = std_self
                        .proper_fraction
                        .expect("Proper fraction for `std_self` not found.");
                    let o_proper_fraction = std_other
                        .proper_fraction
                        .expect("Proper fraction for `std_other` not found.");
                    s_proper_fraction.abs() == o_proper_fraction.abs()
                }
                _ => false,
            };

            similar_poles && similar_angles
        };

        // let result = if self.is_o3_proper(s_au) {
        //     // Proper.
        //
        //     // Parallel or anti-parallel axes.
        //     let similar_poles = approx::relative_eq!(
        //         geometry::get_standard_positive_pole(&self.raw_axis, thresh),
        //         geometry::get_standard_positive_pole(&other.raw_axis, thresh),
        //         epsilon = thresh,
        //         max_relative = thresh
        //     );
        //
        //     // Same angle of rotation (irrespective of signs).
        //     let similar_angles = match (*self.raw_proper_order(), *other.raw_proper_order()) {
        //         (ElementOrder::Inf, ElementOrder::Inf) => {
        //             match (self.proper_angle, other.proper_angle) {
        //                 (Some(s_angle), Some(o_angle)) => {
        //                     approx::relative_eq!(
        //                         s_angle.abs(),
        //                         o_angle.abs(),
        //                         epsilon = thresh,
        //                         max_relative = thresh
        //                     )
        //                 }
        //                 (None, None) => similar_poles,
        //                 _ => false,
        //             }
        //         }
        //         (ElementOrder::Int(_), ElementOrder::Int(_)) => {
        //             let c_proper_fraction = self
        //                 .proper_fraction
        //                 .expect("Proper fraction for `self` not found.");
        //             let o_proper_fraction = other
        //                 .proper_fraction
        //                 .expect("Proper fraction for `other` not found.");
        //             c_proper_fraction.abs() == o_proper_fraction.abs()
        //         }
        //         _ => false,
        //     };
        //
        //     similar_poles && similar_angles
        // } else {
        //     // Improper => convert to inversion-centre convention.
        //     let inv_au = SymmetryElementKind::ImproperInversionCentre(au);
        //     let c_self = self.convert_to_improper_kind(&inv_au, false);
        //     let c_other = other.convert_to_improper_kind(&inv_au, false);
        //
        //     // Parallel or anti-parallel axes.
        //     let similar_poles = approx::relative_eq!(
        //         geometry::get_standard_positive_pole(&c_self.raw_axis, thresh),
        //         geometry::get_standard_positive_pole(&c_other.raw_axis, thresh),
        //         epsilon = thresh,
        //         max_relative = thresh
        //     );
        //
        //     // Same angle of rotation (irrespective of signs).
        //     let similar_angles = match (*c_self.raw_proper_order(), *c_other.raw_proper_order()) {
        //         (ElementOrder::Inf, ElementOrder::Inf) => {
        //             match (c_self.proper_angle, c_other.proper_angle) {
        //                 (Some(s_angle), Some(o_angle)) => {
        //                     approx::relative_eq!(
        //                         s_angle.abs(),
        //                         o_angle.abs(),
        //                         epsilon = thresh,
        //                         max_relative = thresh
        //                     )
        //                 }
        //                 (None, None) => similar_poles,
        //                 _ => false,
        //             }
        //         }
        //         (ElementOrder::Int(_), ElementOrder::Int(_)) => {
        //             let c_proper_fraction = c_self
        //                 .proper_fraction
        //                 .expect("Proper fraction for `c_self` not found.");
        //             let o_proper_fraction = c_other
        //                 .proper_fraction
        //                 .expect("Proper fraction for `c_other` not found.");
        //             c_proper_fraction.abs() == o_proper_fraction.abs()
        //         }
        //         _ => false,
        //     };
        //
        //     similar_poles && similar_angles
        // };

        result && (misc::calculate_hash(self) == misc::calculate_hash(other))
    }
}

impl Eq for SymmetryElement {}

impl Hash for SymmetryElement {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let std_self = self.standardise();

        std_self.rotation_group.hash(state);

        let au = std_self.antiunitary_part();
        au.hash(state);

        std_self.is_o3_proper(au).hash(state);

        if std_self.is_o3_identity(au) || std_self.is_o3_inversion_centre(au) {
            true.hash(state);
        // } else if self.kind == SymmetryElementKind::ImproperMirrorPlane(au) {
        //     let c_self = self
        //         .convert_to_improper_kind(&SymmetryElementKind::ImproperInversionCentre(au), false);
        //     let pole = geometry::get_standard_positive_pole(&c_self.raw_axis, c_self.threshold);
        //     pole[0]
        //         .round_factor(self.threshold)
        //         .integer_decode()
        //         .hash(state);
        //     pole[1]
        //         .round_factor(self.threshold)
        //         .integer_decode()
        //         .hash(state);
        //     pole[2]
        //         .round_factor(self.threshold)
        //         .integer_decode()
        //         .hash(state);
        //     if let ElementOrder::Inf = *c_self.raw_proper_order() {
        //         if let Some(angle) = c_self.proper_angle {
        //             angle
        //                 .abs()
        //                 .round_factor(self.threshold)
        //                 .integer_decode()
        //                 .hash(state);
        //         } else {
        //             0.hash(state);
        //         }
        //     } else {
        //         c_self
        //             .proper_fraction
        //             .expect("No proper fractions for `c_self` found.")
        //             .abs()
        //             .hash(state);
        //     };
        } else {
            let pole = geometry::get_standard_positive_pole(&std_self.raw_axis, std_self.threshold);
            pole[0]
                .round_factor(std_self.threshold)
                .integer_decode()
                .hash(state);
            pole[1]
                .round_factor(std_self.threshold)
                .integer_decode()
                .hash(state);
            pole[2]
                .round_factor(std_self.threshold)
                .integer_decode()
                .hash(state);
            if let ElementOrder::Inf = *std_self.raw_proper_order() {
                if let Some(angle) = std_self.proper_angle {
                    angle
                        .abs()
                        .round_factor(std_self.threshold)
                        .integer_decode()
                        .hash(state);
                } else {
                    0.hash(state);
                }
            } else {
                std_self
                    .proper_fraction
                    .expect("No proper fractions for `self` found.")
                    .abs()
                    .hash(state);
            };
        };
    }
}

/// Time-reversal antiunitary kind.
pub const TR: AntiunitaryKind = AntiunitaryKind::TimeReversal;

/// Complex-conjugation antiunitary kind.
pub const K: AntiunitaryKind = AntiunitaryKind::ComplexConjugation;

/// Proper rotation symmetry element kind.
pub const ROT: SymmetryElementKind = SymmetryElementKind::Proper(None);

/// Improper symmetry element kind in the mirror-plane convention.
pub const SIG: SymmetryElementKind = SymmetryElementKind::ImproperMirrorPlane(None);

/// Improper symmetry element kind in the inversion-centre convention.
pub const INV: SymmetryElementKind = SymmetryElementKind::ImproperInversionCentre(None);

/// Time-reversed proper rotation symmetry element kind.
pub const TRROT: SymmetryElementKind = SymmetryElementKind::Proper(Some(TR));

/// Time-reversed improper symmetry element kind in the mirror-plane convention.
pub const TRSIG: SymmetryElementKind = SymmetryElementKind::ImproperMirrorPlane(Some(TR));

/// Time-reversed improper symmetry element kind in the inversion-centre convention.
pub const TRINV: SymmetryElementKind = SymmetryElementKind::ImproperInversionCentre(Some(TR));

/// Complex-conjugated proper rotation symmetry element kind.
pub const KROT: SymmetryElementKind = SymmetryElementKind::Proper(Some(K));

/// Complex-conjugated improper symmetry element kind in the mirror-plane convention.
pub const KSIG: SymmetryElementKind = SymmetryElementKind::ImproperMirrorPlane(Some(K));

/// Complex-conjugated improper symmetry element kind in the inversion-centre convention.
pub const KINV: SymmetryElementKind = SymmetryElementKind::ImproperInversionCentre(Some(K));

/// Rotation group $`\mathsf{SO}(3)`$.
pub const SO3: RotationGroup = RotationGroup::SO3;

/// Rotation group $`\mathsf{SU}(2)`$, homotopy path of class 0.
pub const SU2_0: RotationGroup = RotationGroup::SU2(true);

/// Rotation group $`\mathsf{SU}(2)`$, homotopy path of class 1.
pub const SU2_1: RotationGroup = RotationGroup::SU2(false);
