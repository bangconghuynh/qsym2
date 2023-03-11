use std::convert::TryInto;
use std::fmt;
use std::hash::{Hash, Hasher};

use approx;
use derive_builder::Builder;
use fraction;
use log;
use nalgebra::Vector3;
use num::integer::gcd;
use num_traits::{ToPrimitive, Zero};

use crate::aux::geometry;
use crate::aux::misc::{self, HashableFloat};
use crate::symmetry::symmetry_element_order::ElementOrder;

type F = fraction::GenericFraction<u32>;

pub mod symmetry_operation;
pub use symmetry_operation::*;

#[cfg(test)]
mod symmetry_element_tests;

// ====================================
// Enum definitions and implementations
// ====================================

/// An enum to classify the types of symmetry element.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum SymmetryElementKind {
    /// Proper symmetry element which consists of just a proper rotation axis.
    ///
    /// The associated boolean indicates whether there is a time-reversal operation associated with
    /// this element.
    Proper(bool),

    /// Improper symmetry element in the mirror-plane convention, which consists
    /// of a proper rotation axis and an orthogonal mirror plane.
    ///
    /// The associated boolean indicates whether there is a time-reversal operation associated with
    /// this element.
    ImproperMirrorPlane(bool),

    /// Improper symmetry element in the inversion-centre convention, which
    /// consists of a proper rotation axis and an inversion centre.
    ///
    /// The associated boolean indicates whether there is a time-reversal operation associated with
    /// this element.
    ImproperInversionCentre(bool),
}

impl SymmetryElementKind {
    /// Indicates if a time-reversal operation is associated with this element.
    #[must_use]
    pub fn contains_time_reversal(&self) -> bool {
        match self {
            Self::Proper(tr)
            | Self::ImproperMirrorPlane(tr)
            | Self::ImproperInversionCentre(tr) => *tr,
        }
    }

    /// Converts the current kind to the desired time-reversal form.
    ///
    /// # Arguments
    ///
    /// * `tr` - A flag indicating whether time reversal is included or not.
    ///
    /// # Returns
    ///
    /// A copy of the current kind with the desired time-reversal flag.
    #[must_use]
    pub fn to_tr(&self, tr: bool) -> Self {
        match self {
            Self::Proper(_) => Self::Proper(tr),
            Self::ImproperMirrorPlane(_) => Self::ImproperMirrorPlane(tr),
            Self::ImproperInversionCentre(_) => Self::ImproperInversionCentre(tr),
        }
    }
}

/// An enumerated type to signify whether a spatial symmetry operation has an associated spin
/// rotation.
#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub enum RotationGroup {
    /// Variant indicating that the proper part of the symmetry element generates rotations in
    /// $`mathsf{SO}(3)`$.
    SO3,

    /// Variant indicating that the proper part of the symmetry element generates rotations in
    /// $`\mathsf{SU}(2)`$. The associated boolean indicates whether the proper part of the element
    /// itself is connected to the identity via a homotopy path of class 0 (`true`) or class 1
    /// (`false`).
    SU2(bool),
}

impl RotationGroup {
    /// Indicates if the rotation is in $`\mathsf{SU}(2)`$.
    fn is_su2(&self) -> bool {
        matches!(self, RotationGroup::SU2(_))
    }

    /// Indicates if the rotation is in $`\mathsf{SU}(2)`$ and connected to the
    /// identity via a homotopy path of class 1.
    fn is_su2_class_1(&self) -> bool {
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
            Self::Proper(tr) => {
                if *tr {
                    write!(f, "Time-reversed proper")
                } else {
                    write!(f, "Proper")
                }
            }
            Self::ImproperMirrorPlane(tr) => {
                if *tr {
                    write!(f, "Time-reversed improper (mirror-plane convention)")
                } else {
                    write!(f, "Improper (mirror-plane convention)")
                }
            }
            Self::ImproperInversionCentre(tr) => {
                if *tr {
                    write!(f, "Time-reversed improper (inversion-centre convention)")
                } else {
                    write!(f, "Improper (inversion-centre convention)")
                }
            }
        }
    }
}

/// A struct for storing and managing symmetry elements.
///
/// Each symmetry element is a geometrical object in $`\mathbb{R}^3`$ that encodes
/// three pieces of information:
///
/// * the axis of rotation $`\hat{\mathbf{n}}`$,
/// * the angle of rotation $`\phi`$, and
/// * whether the element is proper or improper.
///
/// These three pieces of information can be stored in the following
/// representation of a symmetry element $`\hat{g}`$:
///
/// ```math
/// \hat{g} = \hat{\gamma} \hat{C}_n^k,
/// ```
///
/// where $`n \in \mathbb{N}_{+}`$, $`k \in \mathbb{Z}/n\mathbb{Z} = \{1, \ldots, n\}`$,
/// and $`\hat{\gamma}`$ is either the identity $`\hat{e}`$, the inversion operation
/// $`\hat{i}`$, or a reflection operation $`\hat{\sigma}`$. With this definition,
/// the three pieces of information required to specify a geometrical symmetry
/// element are given as follows:
///
/// * the axis of rotation $`\hat{\mathbf{n}}`$ is given by the axis of $`\hat{C}_n^k`$,
/// * the angle of rotation $`\phi = 2\pi k/n \in (0, \pi] \cup \lbrace2\pi\rbrace`$, and
/// * whether the element is proper or improper is given by $`\hat{\gamma}`$.
///
/// This definition, however, also allows $`\hat{g}`$ to be interpreted as
/// an element of $`O(3)`$, which means that $`\hat{g}`$ is also a symmetry
/// operation, and a rather special one that can be used to generate other
/// symmetry operations of the group. $`\hat{g}`$ thus serves as a bridge between
/// molecular symmetry and abstract group theory.
///
/// There is one small caveat: for infinite-order elements, $`n`$ and $`k`$ can
/// no longer be used to give the angle of rotation. There must thus be a
/// mechanism to allow for infinite-order elements to be interpreted as an
/// arbitrary finite-order one. An explicit specification of the angle of rotation
/// seems to be the best way to do this.
#[derive(Builder, Clone)]
pub struct SymmetryElement {
    /// The rotational order $`n`$ of the proper symmetry element.
    pub proper_order: ElementOrder,

    /// The power $`k \in \mathbb{Z}/n\mathbb{Z} = \{1, \ldots, n\}`$ of the proper
    /// symmetry element. This is only defined if [`Self::proper_order`] is finite.
    #[builder(setter(custom), default = "None")]
    pub proper_power: Option<i32>,

    /// The normalised axis of the symmetry element.
    #[builder(setter(custom))]
    pub raw_axis: Vector3<f64>,

    /// The spatial and time-reversal kind of the symmetry element.
    #[builder(default = "SymmetryElementKind::Proper(false)")]
    pub kind: SymmetryElementKind,

    /// The associated spin rotation of the symmetry element, if any.
    pub rotationgroup: RotationGroup,

    /// A flag indicating whether the symmetry element is a generator of the
    /// group to which it belongs.
    #[builder(default = "false")]
    pub generator: bool,

    /// A threshold for approximate equality comparisons.
    #[builder(setter(custom))]
    pub threshold: f64,

    /// An additional superscript for distinguishing the symmetry element.
    #[builder(default = "String::new()")]
    pub additional_superscript: String,

    /// An additional subscript for distinguishing the symmetry element.
    #[builder(default = "String::new()")]
    pub additional_subscript: String,

    /// The fraction $`k/n \in (0, 1]`$ of the proper rotation, represented exactly
    /// for hashing and comparison purposes.
    ///
    /// This is not defined for infinite-order elements.
    ///
    /// Note that the definitions of [`Self::proper_fraction`] and
    /// [`Self::proper_angle`] differ, so that [`Self::proper_fraction`] can facilitate
    /// positive-only comparisons, whereas [`Self::proper_angle`] gives the rotation
    /// angle in the conventional range that puts the identity rotation at the centre
    /// of the range.
    #[builder(setter(skip), default = "self.calc_proper_fraction()")]
    proper_fraction: Option<F>,

    /// The normalised proper angle corresponding to the proper rotation
    /// $`\hat{C}_n^k`$.
    ///
    /// Note that the definitions of [`Self::proper_fraction`] and
    /// [`Self::proper_angle`] differ, so that [`Self::proper_fraction`] can facilitate
    /// positive-only comparisons, whereas [`Self::proper_angle`] gives the rotation
    /// angle in the conventional range of $`(-\pi, +\pi]`$ that puts the identity rotation at
    /// the centre of the range.
    #[builder(setter(custom), default = "self.calc_proper_angle()")]
    proper_angle: Option<f64>,
}

impl SymmetryElementBuilder {
    pub fn proper_power(&mut self, prop_pow: i32) -> &mut Self {
        let proper_order = self
            .proper_order
            .as_ref()
            .expect("Proper order has not been set.");
        self.proper_power = match proper_order {
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

    /// # Panics
    ///
    /// Panics when `self` is of finite order.
    pub fn proper_angle(&mut self, ang: f64) -> &mut Self {
        let proper_order = self
            .proper_order
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
            },
        };
        self
    }

    fn calc_proper_fraction(&self) -> Option<F> {
        let proper_order = self
            .proper_order
            .as_ref()
            .expect("Proper order has not been set.");
        match proper_order {
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
                    .proper_power
                    .expect("Proper power has not been set.")
                    .expect("No proper powers found.");
                let total_proper_fraction = if pp >= 0 {
                    F::new(pp.unsigned_abs(), *io)
                } else {
                    F::new_neg(pp.unsigned_abs(), *io)
                };
                Some(total_proper_fraction)
                // let frac_1_2 = F::new(1u32, 2u32);
                // if total_proper_fraction > frac_1_2 {
                //     let integer_part = total_proper_fraction.trunc();
                //     let x = if total_proper_fraction.fract() <= frac_1_2 {
                //         integer_part
                //     } else {
                //         integer_part + F::one()
                //     };
                //     Some(total_proper_fraction - x)
                // } else if total_proper_fraction <= -frac_1_2 {
                //     let integer_part = (-total_proper_fraction).trunc();
                //     let x = if (-total_proper_fraction).fract() < frac_1_2 {
                //         integer_part
                //     } else {
                //         integer_part + F::one()
                //     };
                //     Some(total_proper_fraction + x)
                // } else {
                //     Some(total_proper_fraction)
                // }
            }
            ElementOrder::Inf => None,
        }
    }

    fn calc_proper_angle(&self) -> Option<f64> {
        let proper_order = self
            .proper_order
            .as_ref()
            .expect("Proper order has not been set.");
        match proper_order {
            ElementOrder::Int(io) => {
                let pp = self
                    .proper_power
                    .expect("Proper power has not been set.")
                    .expect("No proper powers found.");
                let total_proper_fraction = if pp >= 0 {
                    F::new(pp.unsigned_abs(), *io)
                } else {
                    F::new_neg(pp.unsigned_abs(), *io)
                };
                // let frac_1_2 = F::new(1u32, 2u32);
                // let proper_fraction = if total_proper_fraction > frac_1_2 {
                //     let integer_part = total_proper_fraction.trunc();
                //     let x = if total_proper_fraction.fract() <= frac_1_2 {
                //         integer_part
                //     } else {
                //         integer_part + F::one()
                //     };
                //     total_proper_fraction - x
                // } else if total_proper_fraction <= -frac_1_2 {
                //     let integer_part = (-total_proper_fraction).trunc();
                //     let x = if (-total_proper_fraction).fract() < frac_1_2 {
                //         integer_part
                //     } else {
                //         integer_part + F::one()
                //     };
                //     total_proper_fraction + x
                // } else {
                //     total_proper_fraction
                // };
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

    pub fn raw_axis(&mut self, axs: Vector3<f64>) -> &mut Self {
        let thresh = self.threshold.expect("Threshold value has not been set.");
        if approx::relative_eq!(axs.norm(), 1.0, epsilon = thresh, max_relative = thresh) {
            self.raw_axis = Some(axs);
        } else {
            log::warn!("Axis not normalised. Normalising...");
            self.raw_axis = Some(axs.normalize());
        }
        self
    }

    pub fn threshold(&mut self, thresh: f64) -> &mut Self {
        if thresh >= 0.0 {
            self.threshold = Some(thresh);
        } else {
            log::error!(
                "Threshold value {} is invalid. Threshold must be non-negative.",
                thresh
            );
            self.threshold = None;
        }
        self
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

    pub fn positive_axis(&self) -> Vector3<f64> {
        geometry::get_positive_pole(&self.raw_axis, self.threshold)
    }

    pub fn signed_axis(&self) -> Vector3<f64> {
        let tr = self.contains_time_reversal();
        if self.is_o3_binary_rotation_axis(tr) || self.is_o3_mirror_plane(tr) {
            self.positive_axis()
        } else {
            self.proper_fraction
                .map(|frac| {
                    frac.signum()
                        .to_f64()
                        .expect("Unable to obtain the sign of the proper fraction.")
                })
                .or_else(|| self.proper_angle.map(|proper_angle| proper_angle.signum()))
                .and_then(|signum| Some(signum * self.raw_axis))
                .unwrap_or_else(|| {
                    log::warn!("No rotation signs could be obtained. The positive axis will be used for the signed axis.");
                    self.positive_axis()
                })
        }
    }

    pub fn proper_fraction(&self) -> Option<&F> {
        self.proper_fraction.as_ref()
    }

    pub fn proper_angle(&self) -> Option<f64> {
        self.proper_angle
    }

    /// Checks if the symmetry element contains a time-reversal operator.
    ///
    /// # Returns
    ///
    /// A flag indicating if the symmetry element contains a time-reversal operator.
    #[must_use]
    pub fn contains_time_reversal(&self) -> bool {
        self.kind.contains_time_reversal()
    }

    /// Checks if the symmetry element contains an active spin rotation.
    #[must_use]
    fn is_su2(&self) -> bool {
        self.rotationgroup.is_su2()
    }

    /// Checks if the symmetry element contains an inverse spin rotation.
    #[must_use]
    fn is_su2_class_1(&self) -> bool {
        self.rotationgroup.is_su2_class_1()
    }

    /// Checks if the spatial part of the symmetry element is proper and has the specified
    /// time-reversal attribute.
    ///
    /// # Arguments
    ///
    /// * `tr` - A flag indicating if time reversal is to be considered.
    ///
    /// # Returns
    ///
    /// A flag indicating if the symmetry element is proper and has the specified time-reversal
    /// attribute.
    #[must_use]
    pub fn is_o3_proper(&self, tr: bool) -> bool {
        self.kind == SymmetryElementKind::Proper(tr)
    }

    /// Checks if the symmetry element is spatially an identity element and has the specified
    /// time-reversal attribute.
    ///
    /// # Arguments
    ///
    /// * `tr` - A flag indicating if time reversal is to be considered.
    ///
    /// # Returns
    ///
    /// A flag indicating if this symmetry element is spatially an identity element and has the
    /// specified time-reversal attribute.
    #[must_use]
    pub fn is_o3_identity(&self, tr: bool) -> bool {
        self.kind == SymmetryElementKind::Proper(tr)
            && self
                .proper_fraction
                .map(|frac| frac.is_zero())
                .or_else(|| {
                    self.proper_angle.map(|proper_angle| {
                        approx::relative_eq!(
                            proper_angle,
                            0.0,
                            epsilon = self.threshold,
                            max_relative = self.threshold
                        )
                    })
                })
                .unwrap_or(false)
    }

    /// Checks if the symmetry element is spatially an inversion centre and has the specified
    /// time-reversal attribute.
    ///
    /// # Arguments
    ///
    /// * `tr` - A flag indicating if time reversal is to be considered.
    ///
    /// # Returns
    ///
    /// A flag indicating if this symmetry element is an inversion centre and has the specified
    /// time-reversal attribute.
    #[must_use]
    pub fn is_o3_inversion_centre(&self, tr: bool) -> bool {
        match self.kind {
            SymmetryElementKind::ImproperMirrorPlane(sig_tr) => {
                sig_tr == tr
                    && self
                        .proper_fraction
                        .map(|frac| frac == F::new(1u32, 2u32))
                        .or_else(|| {
                            self.proper_angle.map(|proper_angle| {
                                approx::relative_eq!(
                                    proper_angle,
                                    std::f64::consts::PI,
                                    epsilon = self.threshold,
                                    max_relative = self.threshold
                                )
                            })
                        })
                        .unwrap_or(false)
            }
            SymmetryElementKind::ImproperInversionCentre(inv_tr) => {
                inv_tr == tr
                    && self
                        .proper_fraction
                        .map(|frac| frac.is_zero())
                        .or_else(|| {
                            self.proper_angle.map(|proper_angle| {
                                approx::relative_eq!(
                                    proper_angle,
                                    0.0,
                                    epsilon = self.threshold,
                                    max_relative = self.threshold
                                )
                            })
                        })
                        .unwrap_or(false)
            }
            _ => false,
        }
    }

    /// Checks if the symmetry element is spatially a binary rotation axis and has the specified
    /// time-reversal attribute.
    ///
    /// # Arguments
    ///
    /// * `tr` - A flag indicating if time reversal is to be considered.
    ///
    /// # Returns
    ///
    /// A flag indicating if this symmetry element is spatially a binary rotation axis and has the
    /// specified time-reversal attribute.
    #[must_use]
    pub fn is_o3_binary_rotation_axis(&self, tr: bool) -> bool {
        self.kind == SymmetryElementKind::Proper(tr)
            && self
                .proper_fraction
                .map(|frac| frac == F::new(1u32, 2u32))
                .or_else(|| {
                    self.proper_angle.map(|proper_angle| {
                        approx::relative_eq!(
                            proper_angle,
                            std::f64::consts::PI,
                            epsilon = self.threshold,
                            max_relative = self.threshold
                        )
                    })
                })
                .unwrap_or(false)
    }

    /// Checks if the symmetry element is spatially a mirror plane and has the specified
    /// time-reversal attribute.
    ///
    /// # Arguments
    ///
    /// * `tr` - A flag indicating if time reversal is to be considered.
    ///
    /// # Returns
    ///
    /// A flag indicating if this symmetry element is spatially a mirror plane and has the
    /// specified time-reversal attribute.
    #[must_use]
    pub fn is_o3_mirror_plane(&self, tr: bool) -> bool {
        match self.kind {
            SymmetryElementKind::ImproperMirrorPlane(sig_tr) => {
                sig_tr == tr
                    && self
                        .proper_fraction
                        .map(|frac| frac.is_zero())
                        .or_else(|| {
                            self.proper_angle.map(|proper_angle| {
                                approx::relative_eq!(
                                    proper_angle,
                                    0.0,
                                    epsilon = self.threshold,
                                    max_relative = self.threshold
                                )
                            })
                        })
                        .unwrap_or(false)
            }
            SymmetryElementKind::ImproperInversionCentre(inv_tr) => {
                inv_tr == tr
                    && self
                        .proper_fraction
                        .map(|frac| frac == F::new(1u32, 2u32))
                        .or_else(|| {
                            self.proper_angle.map(|proper_angle| {
                                approx::relative_eq!(
                                    proper_angle,
                                    std::f64::consts::PI,
                                    epsilon = self.threshold,
                                    max_relative = self.threshold
                                )
                            })
                        })
                        .unwrap_or(false)
            }
            _ => false,
        }
    }

    /// Returns the full symbol for this symmetry element, which does not
    /// classify certain improper rotation axes into inversion centres or mirror
    /// planes, but which does simplify the power/order ratio, and which displays only the absolute
    /// value of the power since symmetry elements do not distinguish senses of rotations.
    /// Rotations of oposite directions are inverses of each other, both of which must exist in the
    /// group.
    ///
    /// Some additional symbols that can be unconventional include:
    ///
    /// * `θ`: time reversal,
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
        let tr_sym = if self.kind.contains_time_reversal() {
            "θ·"
        } else {
            ""
        };
        let main_symbol: String = match self.kind {
            SymmetryElementKind::Proper(_) => {
                format!("{tr_sym}C")
            }
            SymmetryElementKind::ImproperMirrorPlane(_) => {
                if self.proper_order != ElementOrder::Inf
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
                    format!("{tr_sym}S")
                } else {
                    format!("{tr_sym}σC")
                }
            }
            SymmetryElementKind::ImproperInversionCentre(_) => {
                if self.proper_order != ElementOrder::Inf
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
                    format!("{tr_sym}Ṡ")
                } else {
                    format!("{tr_sym}iC")
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
            assert_eq!(self.proper_order, ElementOrder::Inf);
            let proper_angle = if let Some(proper_angle) = self.proper_angle {
                format!("({:+.3})", proper_angle.abs())
            } else {
                String::new()
            };
            format!("{main_symbol}{}{proper_angle}{su2_sym}", self.proper_order)
        }
    }

    /// Returns the detailed symbol for this symmetry element, which classifies
    /// special symmetry elements (identity, inversion centre, mirror planes), and which simplifies
    /// the power/order ratio and displays only the absolute value of the power since symmetry
    /// elements do not distinguish senses of rotations. Rotations of oposite directions are
    /// inverses of each other, both of which must exist in the group.
    ///
    /// # Returns
    ///
    /// The detailed symbol for this symmetry element.
    #[must_use]
    pub fn get_simplified_symbol(&self) -> String {
        let (main_symbol, needs_power) = match self.kind {
            SymmetryElementKind::Proper(tr) => {
                if self.is_o3_identity(tr) {
                    if tr {
                        ("θ".to_owned(), false)
                    } else {
                        ("E".to_owned(), false)
                    }
                } else {
                    (format!("{}C", if tr { "θ·" } else { "" }), true)
                }
            }
            SymmetryElementKind::ImproperMirrorPlane(tr) => {
                let tr_sym = if tr { "θ·" } else { "" };
                if self.is_o3_mirror_plane(tr) {
                    (format!("{tr_sym}σ"), false)
                } else if self.is_o3_inversion_centre(tr) {
                    (format!("{tr_sym}i"), false)
                } else if self.proper_order == ElementOrder::Inf
                    || *self
                        .proper_fraction
                        .expect("No proper fractions found for a finite-order element.")
                        .numer()
                        .expect("Unable to extract the numerator of the proper fraction.")
                        == 1
                {
                    (format!("{tr_sym}S"), false)
                } else {
                    (format!("{tr_sym}σC"), true)
                }
            }
            SymmetryElementKind::ImproperInversionCentre(tr) => {
                let tr_sym = if tr { "θ·" } else { "" };
                if self.is_o3_mirror_plane(tr) {
                    (format!("{tr_sym}σ"), false)
                } else if self.is_o3_inversion_centre(tr) {
                    (format!("{tr_sym}i"), false)
                } else if self.proper_order == ElementOrder::Inf
                    || *self
                        .proper_fraction
                        .expect("No proper fractions found for a finite-order element.")
                        .numer()
                        .expect("Unable to extract the numerator of the proper fraction.")
                        == 1
                {
                    (format!("{tr_sym}Ṡ"), false)
                } else {
                    (format!("{tr_sym}iC"), true)
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
            let tr = self.contains_time_reversal();
            let proper_order = if self.is_o3_identity(tr)
                || self.is_o3_inversion_centre(tr)
                || self.is_o3_mirror_plane(tr)
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
                "{main_symbol}{}{proper_order}{proper_power}{su2_sym}{}",
                self.additional_superscript, self.additional_subscript
            )
        } else {
            assert_eq!(self.proper_order, ElementOrder::Inf);
            let proper_angle = if let Some(proper_angle) = self.proper_angle {
                format!("({:+.3})", proper_angle.abs())
            } else {
                String::new()
            };
            format!(
                "{main_symbol}{}{}{proper_angle}{su2_sym}{}",
                self.additional_superscript, self.proper_order, self.additional_subscript
            )
        }
    }

    /// Returns a copy of the current improper symmetry element that has been
    /// converted to the required improper kind.
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
    /// $`\sigma = i C_2`$, we can easily show that
    ///
    /// ```math
    /// \begin{aligned}
    ///     n' &= \frac{2n}{\operatorname{gcd}(2n, n + 2k)},\\
    ///     k' &= \frac{n + 2k}{\operatorname{gcd}(2n, n + 2k)} \mod n'.
    /// \end{aligned}
    /// ```
    ///
    /// The above relations are self-inverse. It can be further shown that
    /// $`\operatorname{gcd}(n', k') = 1`$. Hence, for symmetry *element*
    /// conversions, we can simply take $`k' = 1`$. This is because a symmetry
    /// element plays the role of a generator, and the coprimality of $`n'`$ and
    /// $`k'`$ means that $`i C_{n'}^{1}`$ is as valid a generator as
    /// $`i C_{n'}^{k'}`$.
    ///
    /// # Arguments
    ///
    /// * `improper_kind` - The improper kind to which `self` is to be converted. There is no need
    /// to make sure the time reversal specification in `improper_kind` matches that of `self` as
    /// the conversion will take care of this.
    /// * `preserves_power` - Flag indicating if the proper rotation power $`k'`$
    /// should be preserved or should be set to $`1`$.
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
        let tr = self.contains_time_reversal();
        assert!(
            !self.is_o3_proper(tr),
            "Only improper elements can be converted."
        );
        let improper_kind = improper_kind.to_tr(tr);
        assert!(
            !matches!(improper_kind, SymmetryElementKind::Proper(_)),
            "`improper_kind` must be one of the improper variants."
        );

        if self.kind == improper_kind {
            return self.clone();
        }

        let (dest_order, dest_proper_power) = match self.proper_order {
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
                .rotationgroup(self.rotationgroup.clone())
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
                        .rotationgroup(self.rotationgroup.clone())
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
                        .rotationgroup(self.rotationgroup.clone())
                        .generator(self.generator)
                        .additional_superscript(self.additional_superscript.clone())
                        .additional_subscript(self.additional_subscript.clone())
                        .build()
                        .expect("Unable to construct a symmetry element.")
                }
            }
        }
    }

    /// Adds spin rotation to the current element if none is present.
    ///
    /// # Arguments
    ///
    /// * `normal` - A boolean indicating whether the added spin rotation is normal or inverse.
    ///
    /// # Returns
    ///
    /// A symmetry element with the added spin rotation if none is present, or `None` if the
    /// current symmetry element already has an associated spin rotation.
    pub fn to_su2(&self, normal: bool) -> Option<Self> {
        if self.is_su2() {
            None
        } else {
            let mut element = self.clone();
            element.rotationgroup = RotationGroup::SU2(normal);
            Some(element)
        }
    }

    /// The closeness of the symmetry element's axis to one of the
    /// three space-fixed Cartesian axes.
    ///
    /// # Returns
    ///
    /// A tuple of two values:
    /// - A value $`\gamma \in [0, 1-1/\sqrt{3}]`$ indicating how
    /// close the axis is to one of the three Cartesian axes. The closer
    /// $`\gamma`$ is to $`0`$, the closer the alignment.
    /// - An index for the closest axis: `0` for $`z`$, `1` for $`y`$, `2`
    /// for $`x`$.
    ///
    /// # Panics
    ///
    /// Panics when $`\gamma`$ is outside the required closed interval $`[0, 1-1/\sqrt{3}]`$ by
    /// more than the threshold value in `self`.
    #[must_use]
    pub fn closeness_to_cartesian_axes(&self) -> (f64, usize) {
        let pos_axis = self.positive_axis();
        let rev_pos_axis = Vector3::new(pos_axis[(2)], pos_axis[(1)], pos_axis[(0)]);
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
        write!(
            f,
            "{}({:+.3}, {:+.3}, {:+.3})",
            self.get_full_symbol(),
            signed_axis[0] + 0.0,
            signed_axis[1] + 0.0,
            signed_axis[2] + 0.0
        )
    }
}

impl fmt::Display for SymmetryElement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let tr = self.contains_time_reversal();
        if self.is_o3_identity(tr) || self.is_o3_inversion_centre(tr) {
            write!(f, "{}", self.get_simplified_symbol())
        } else {
            let signed_axis = self.signed_axis();
            write!(
                f,
                "{}({:+.3}, {:+.3}, {:+.3})",
                self.get_simplified_symbol(),
                signed_axis[0] + 0.0,
                signed_axis[1] + 0.0,
                signed_axis[2] + 0.0
            )
        }
    }
}

impl PartialEq for SymmetryElement {
    /// Two symmetry elements are equal if and only if the following conditions
    /// are all satisfied:
    ///
    /// * they are both in the same rotation group and belong to the same homotopy class;
    /// * they are both proper or improper;
    /// * they both have the same time reversal properties;
    /// * their axes are either parallel or anti-parallel;
    /// * their proper rotation angles have equal absolute values.
    ///
    /// For improper elements, proper rotation angles are taken in the inversion
    /// centre convention.
    ///
    /// Thus, symmetry element equality is less strict than symmetry operation equality. This is so
    /// that parallel or anti-parallel symmetry elements with the same spatial and time-reversal
    /// parities and angle of rotation are deemed identical, thus facilitating symmetry detection
    /// where one does not yet care much about directions of rotations.
    #[allow(clippy::too_many_lines)]
    fn eq(&self, other: &Self) -> bool {
        if self.rotationgroup != other.rotationgroup {
            // Different rotation groups or homotopy classes.
            return false;
        }

        if self.contains_time_reversal() != other.contains_time_reversal() {
            // Different time-reversal parities.
            return false;
        }

        let tr = self.contains_time_reversal();

        if self.is_o3_proper(tr) != other.is_o3_proper(tr) {
            // Different spatial parities.
            return false;
        }

        if self.is_o3_identity(tr) && other.is_o3_identity(tr) {
            // Both are spatial identity.
            assert_eq!(
                misc::calculate_hash(self),
                misc::calculate_hash(other),
                "{self} and {other} have unequal hashes."
            );
            return true;
        }

        if self.is_o3_inversion_centre(tr) && other.is_o3_inversion_centre(tr) {
            // Both are spatial inversion centre.
            assert_eq!(
                misc::calculate_hash(self),
                misc::calculate_hash(other),
                "{self} and {other} have unequal hashes."
            );
            return true;
        }

        let thresh = (self.threshold * other.threshold).sqrt();

        let result = if self.is_o3_proper(tr) {
            // Proper.

            // Parallel or anti-parallel axes.
            let similar_poles = approx::relative_eq!(
                geometry::get_positive_pole(&self.raw_axis, thresh),
                geometry::get_positive_pole(&other.raw_axis, thresh),
                epsilon = thresh,
                max_relative = thresh
            );

            // Same angle of rotation (irrespective of signs).
            let similar_angles = match (self.proper_order, other.proper_order) {
                (ElementOrder::Inf, ElementOrder::Inf) => {
                    match (self.proper_angle, other.proper_angle) {
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
                    let c_proper_fraction = self
                        .proper_fraction
                        .expect("Proper fraction for `self` not found.");
                    let o_proper_fraction = other
                        .proper_fraction
                        .expect("Proper fraction for `other` not found.");
                    c_proper_fraction.abs() == o_proper_fraction.abs()
                }
                _ => false,
            };

            similar_poles && similar_angles
        } else {
            // Improper => convert to inversion-centre convention.
            let inv_tr = SymmetryElementKind::ImproperInversionCentre(tr);
            let c_self = self.convert_to_improper_kind(&inv_tr, false);
            let c_other = other.convert_to_improper_kind(&inv_tr, false);

            // Parallel or anti-parallel axes.
            let similar_poles = approx::relative_eq!(
                geometry::get_positive_pole(&c_self.raw_axis, thresh),
                geometry::get_positive_pole(&c_other.raw_axis, thresh),
                epsilon = thresh,
                max_relative = thresh
            );

            // Same angle of rotation (irrespective of signs).
            let similar_angles = match (c_self.proper_order, c_other.proper_order) {
                (ElementOrder::Inf, ElementOrder::Inf) => {
                    match (c_self.proper_angle, c_other.proper_angle) {
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
                    let c_proper_fraction = c_self
                        .proper_fraction
                        .expect("Proper fraction for `c_self` not found.");
                    let o_proper_fraction = c_other
                        .proper_fraction
                        .expect("Proper fraction for `c_other` not found.");
                    c_proper_fraction.abs() == o_proper_fraction.abs()
                }
                _ => false,
            };

            similar_poles && similar_angles
        };

        if result {
            assert_eq!(
                misc::calculate_hash(self),
                misc::calculate_hash(other),
                "`{self}` and `{other}` have unequal hashes."
            );
        }
        result
    }
}

impl Eq for SymmetryElement {}

impl Hash for SymmetryElement {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.rotationgroup.hash(state);

        let tr = self.contains_time_reversal();
        tr.hash(state);

        self.is_o3_proper(tr).hash(state);

        if self.is_o3_identity(tr) || self.is_o3_inversion_centre(tr) {
            true.hash(state);
        } else if self.kind == SymmetryElementKind::ImproperMirrorPlane(tr) {
            let c_self = self
                .convert_to_improper_kind(&SymmetryElementKind::ImproperInversionCentre(tr), false);
            let pole = geometry::get_positive_pole(&c_self.raw_axis, c_self.threshold);
            pole[0]
                .round_factor(self.threshold)
                .integer_decode()
                .hash(state);
            pole[1]
                .round_factor(self.threshold)
                .integer_decode()
                .hash(state);
            pole[2]
                .round_factor(self.threshold)
                .integer_decode()
                .hash(state);
            if let ElementOrder::Inf = c_self.proper_order {
                if let Some(angle) = c_self.proper_angle {
                    angle
                        .abs()
                        .round_factor(self.threshold)
                        .integer_decode()
                        .hash(state);
                } else {
                    0.hash(state);
                }
            } else {
                c_self
                    .proper_fraction
                    .expect("No proper fractions for `c_self` found.")
                    .abs()
                    .hash(state);
            };
        } else {
            let pole = geometry::get_positive_pole(&self.raw_axis, self.threshold);
            pole[0]
                .round_factor(self.threshold)
                .integer_decode()
                .hash(state);
            pole[1]
                .round_factor(self.threshold)
                .integer_decode()
                .hash(state);
            pole[2]
                .round_factor(self.threshold)
                .integer_decode()
                .hash(state);
            if let ElementOrder::Inf = self.proper_order {
                if let Some(angle) = self.proper_angle {
                    angle
                        .abs()
                        .round_factor(self.threshold)
                        .integer_decode()
                        .hash(state);
                } else {
                    0.hash(state);
                }
            } else {
                self.proper_fraction
                    .expect("No proper fractions for `self` found.")
                    .abs()
                    .hash(state);
            };
        };
    }
}

pub const ROT: SymmetryElementKind = SymmetryElementKind::Proper(false);
pub const SIG: SymmetryElementKind = SymmetryElementKind::ImproperMirrorPlane(false);
pub const INV: SymmetryElementKind = SymmetryElementKind::ImproperInversionCentre(false);
pub const TRROT: SymmetryElementKind = SymmetryElementKind::Proper(true);
pub const TRSIG: SymmetryElementKind = SymmetryElementKind::ImproperMirrorPlane(true);
pub const TRINV: SymmetryElementKind = SymmetryElementKind::ImproperInversionCentre(true);
pub const SO3: RotationGroup = RotationGroup::SO3;
pub const SU2_0: RotationGroup = RotationGroup::SU2(true);
pub const SU2_1: RotationGroup = RotationGroup::SU2(false);
