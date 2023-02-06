use std::fmt;
use std::hash::{Hash, Hasher};
use std::ops::Mul;

use approx;
use derive_builder::Builder;
use fraction;
use nalgebra::{Point3, Vector3};
use num_traits::Pow;

use crate::aux::geometry;
use crate::aux::misc::{self, HashableFloat};
use crate::group::FiniteOrder;
use crate::symmetry::symmetry_element::{SymmetryElement, SymmetryElementKind, INV};
use crate::symmetry::symmetry_element_order::ElementOrder;

type F = fraction::GenericFraction<u32>;
type Quaternion = (f64, Vector3<f64>);

#[cfg(test)]
#[path = "symmetry_operation_tests.rs"]
mod symmetry_operation_tests;

/// A trait for special symmetry transformations.
pub trait SpecialSymmetryTransformation {
    /// Checks if the symmetry operation is proper or not.
    ///
    /// # Returns
    ///
    /// A flag indicating if the symmetry operation is proper.
    fn is_proper(&self) -> bool;

    /// Checks if the symmetry operation is antiunitary or not.
    ///
    /// # Returns
    ///
    /// A flag indicating if the symmetry oppperation is antiunitary.
    fn is_antiunitary(&self) -> bool;

    /// Checks if the symmetry operation is the identity.
    ///
    /// # Returns
    ///
    /// A flag indicating if this symmetry operation is the identity.
    fn is_identity(&self) -> bool;

    /// Checks if the symmetry operation is a pure time-reversal.
    ///
    /// # Returns
    ///
    /// A flag indicating if this symmetry operation is a pure time-reversal.
    fn is_time_reversal(&self) -> bool;

    /// Checks if the symmetry operation is an inversion.
    ///
    /// # Returns
    ///
    /// A flag indicating if this symmetry operation is an inversion.
    fn is_inversion(&self) -> bool;

    /// Checks if the symmetry operation is an inversion accompanied by a time reversal.
    ///
    /// # Returns
    ///
    /// A flag indicating if this symmetry operation is an inversion accompanied by a time reversal.
    fn is_tr_inversion(&self) -> bool;

    /// Checks if the symmetry operation is a binary rotation.
    ///
    /// # Returns
    ///
    /// A flag indicating if this symmetry operation is a binary rotation.
    fn is_binary_rotation(&self) -> bool;

    /// Checks if the symmetry operation is a binary rotation accompanied by a time reversal.
    ///
    /// # Returns
    ///
    /// A flag indicating if this symmetry operation is a binary rotation.
    fn is_tr_binary_rotation(&self) -> bool;

    /// Checks if the symmetry operation is a reflection.
    ///
    /// # Returns
    ///
    /// A flag indicating if this symmetry operation is a reflection.
    fn is_reflection(&self) -> bool;

    /// Checks if the symmetry operation is a reflection.
    ///
    /// # Returns
    ///
    /// A flag indicating if this symmetry operation is a reflection accompanied by a time reversal.
    fn is_tr_reflection(&self) -> bool;
}

/// A struct for managing symmetry operations generated from symmetry elements.
///
/// These symmetry operations are limited to members of $`O(3)`$ only.
///
/// A symmetry element serves as a generator for symmetry operations. Thus,
/// a symmetry element together with an integer indicating the number of times
/// the symmetry element is applied specifies a symmetry operation.
#[derive(Builder, Clone)]
pub struct SymmetryOperation {
    /// The generating symmetry element for this symmetry operation.
    pub generating_element: SymmetryElement,

    /// The integral power indicating the number of times
    /// [`Self::generating_element`] is applied to form the symmetry operation.
    pub power: i32,

    /// The total proper rotation angle associated witrh this operation (after
    /// taking into account the power of the operation).
    ///
    /// This is simply the proper rotation angle of [`Self::generating_element`]
    /// multiplied by [`Self::power`].
    ///
    /// This angle lies in the open interval $`(-\pi, \pi]`$. For improper
    /// operations, this angle depends on the convention used to describe the
    /// [`Self::generating_element`].
    ///
    /// Note that the definitions of [`Self::total_proper_fraction`] and
    /// [`Self::total_proper_angle`] differ, so that
    /// [`Self::total_proper_fraction`] can facilitate positive-only comparisons,
    /// whereas [`Self::total_proper_angle`] gives the rotation angle in the
    /// conventional range that puts the identity rotation at the centre
    /// of the range.
    #[builder(setter(skip), default = "self.calc_total_proper_angle()")]
    total_proper_angle: f64,

    /// The fraction $`pk/n \in (0, 1]`$ of the proper rotation, represented
    /// exactly for hashing and comparison purposes.
    ///
    /// This is not defined for operations with infinite-order generating
    /// elements.
    ///
    /// Note that the definitions of [`Self::total_proper_fraction`] and
    /// [`Self::total_proper_angle`] differ, so that
    /// [`Self::total_proper_fraction`] can facilitate positive-only comparisons,
    /// whereas [`Self::total_proper_angle`] gives the rotation angle in the
    /// conventional range that puts the identity rotation at the centre
    /// of the range.
    #[builder(setter(skip), default = "self.calc_total_proper_fraction()")]
    pub total_proper_fraction: Option<F>,
}

impl SymmetryOperationBuilder {
    fn calc_total_proper_angle(&self) -> f64 {
        geometry::normalise_rotation_angle(
            self.generating_element
                .as_ref()
                .expect("Generating element has not been set.")
                .proper_angle
                .expect("Proper angle has not been set.")
                * (f64::from(self.power.expect("Power has not been set."))),
            self.generating_element
                .as_ref()
                .expect("Generating element has not been set.")
                .threshold,
        )
    }

    fn calc_total_proper_fraction(&self) -> Option<F> {
        match self
            .generating_element
            .as_ref()
            .expect("Generating element has not been set.")
            .proper_fraction
        {
            Some(frac) => {
                let pow = self.power.expect("Power has not been set.");
                let unnormalised_frac = if pow >= 0 {
                    (frac * F::new(pow.unsigned_abs(), 1u32)).fract()
                } else {
                    F::from(1u32) - (frac * F::new(pow.unsigned_abs(), 1u32)).fract()
                };
                if unnormalised_frac == F::from(0u64) {
                    Some(F::from(1u64))
                } else {
                    Some(unnormalised_frac)
                }
            }
            None => None,
        }
    }
}

impl SymmetryOperation {
    /// Returns a builder to construct a new symmetry operation.
    ///
    /// # Returns
    ///
    /// A builder to construct a new symmetry operation.
    #[must_use]
    pub fn builder() -> SymmetryOperationBuilder {
        SymmetryOperationBuilder::default()
    }

    /// Constructs a finite-order-element-generated symmetry operation from a
    /// quaternion.
    ///
    /// The rotation angle encoded in the quaternion is taken to be non-negative
    /// and assigned as the proper rotation angle associated with the element
    /// generating the operation.
    ///
    /// If an improper operation is required, its generator will be constructed
    /// in the inversion-centre convention.
    ///
    /// # Arguments
    ///
    /// * `qtn` - A quaternion encoding the proper rotation associated with the
    /// generating element of the operation to be constructed.
    /// * `proper` - A flag indicating if the operation is proper or improper.
    /// * `thresh` - Threshold for comparisons.
    /// * `tr` - A flag indicating if the resulting symmetry operation should be accompanied by a
    /// time-reversal operator.
    ///
    /// # Returns
    ///
    /// The constructed symmetry operation.
    ///
    /// # Panics
    ///
    /// Panics when the scalar part of the provided quaternion lies outside $`[0, 1]`$ by more than
    /// the specified threshold `thresh`, or when the rotation angle associated with the quaternion
    /// cannot be gracefully converted into an integer tuple of order and power.
    #[must_use]
    pub fn from_quaternion(
        qtn: Quaternion,
        proper: bool,
        thresh: f64,
        max_trial_power: u32,
        tr: bool,
    ) -> Self {
        let (scalar_part, vector_part) = qtn;
        assert!(-thresh <= scalar_part && scalar_part <= 1.0 + thresh);
        let (axis, order, power) =
            if approx::relative_eq!(scalar_part, 1.0, epsilon = thresh, max_relative = thresh) {
                // Zero-degree rotation, i.e. identity or inversion
                (Vector3::new(0.0, 0.0, 1.0), 1u32, 1u32)
            } else {
                let positive_normalised_angle = 2.0 * scalar_part.acos(); // in [0, π]
                let axis = vector_part / (0.5 * positive_normalised_angle).sin();
                let proper_fraction = geometry::get_proper_fraction(
                positive_normalised_angle,
                thresh,
                max_trial_power,
            )
            .unwrap_or_else(|| {
                panic!("No proper fraction could be found for angle `{positive_normalised_angle}`.")
            });
                (
                    axis,
                    *proper_fraction.denom().unwrap_or_else(|| {
                        panic!("Unable to extract the denominator of `{proper_fraction}`.")
                    }),
                    *proper_fraction.numer().unwrap_or_else(|| {
                        panic!("Unable to extract the numerator of `{proper_fraction}`.")
                    }),
                )
            };

        let kind = if proper {
            SymmetryElementKind::Proper(tr)
        } else {
            SymmetryElementKind::ImproperInversionCentre(tr)
        };

        let element = SymmetryElement::builder()
            .threshold(thresh)
            .proper_order(ElementOrder::Int(order))
            .proper_power(power)
            .axis(axis)
            .kind(kind)
            .build()
            .expect("Unable to construct a symmetry element.");

        SymmetryOperation::builder()
            .generating_element(element)
            .power(1)
            .build()
            .expect("Unable to construct a symmetry operation.")
    }

    /// Finds the pole associated with this operation.
    ///
    /// This is the point on the unit sphere that is left invariant by the operation.
    ///
    /// For improper elements, the inversion-centre convention is used to define
    /// the pole. This allows a proper rotation and its improper partner to have the
    /// same pole, thus facilitating the consistent specification of poles for the
    /// identity / inversion and binary rotations / reflections.
    ///
    /// Note that binary rotations / reflections have unique poles on the positive
    /// hemisphere (*i.e.*, $`C_2(\hat{\mathbf{n}}) = C_2^{-1}(\hat{\mathbf{n}})`$
    /// and $`\sigma(\hat{\mathbf{n}}) = \sigma^{-1}(\hat{\mathbf{n}})`$).
    ///
    /// See S.L. Altmann, Rotations, Quaternions, and Double Groups (Dover
    /// Publications, Inc., New York, 2005) (Chapter 9) for further information.
    ///
    /// # Returns
    ///
    /// The pole associated with this operation.
    ///
    /// # Panics
    ///
    /// Panics when no total proper fractions could be found for this operation.
    #[must_use]
    pub fn calc_pole(&self) -> Point3<f64> {
        let op = if self.is_proper() {
            self.clone()
        } else {
            self.convert_to_improper_kind(&INV)
        };
        match op.generating_element.proper_order {
            ElementOrder::Int(_) => {
                let frac_1_2 = F::new(1u32, 2u32);
                if op
                    .total_proper_fraction
                    .expect("No total proper fractions found.")
                    == frac_1_2
                {
                    // Binary rotations or reflections
                    Point3::from(geometry::get_positive_pole(
                        &op.generating_element.axis,
                        op.generating_element.threshold,
                    ))
                } else if op
                    .total_proper_fraction
                    .expect("No total proper fractions found.")
                    < frac_1_2
                {
                    // Positive rotation angles
                    Point3::from(op.generating_element.axis)
                } else if op
                    .total_proper_fraction
                    .expect("No total proper fractions found.")
                    < F::from(1u64)
                {
                    // Negative rotation angles
                    Point3::from(-op.generating_element.axis)
                } else {
                    assert_eq!(
                        op.total_proper_fraction
                            .expect("No total proper fractions found."),
                        F::from(1u64)
                    );
                    Point3::origin()
                }
            }
            ElementOrder::Inf => {
                if approx::relative_eq!(
                    op.total_proper_angle,
                    std::f64::consts::PI,
                    max_relative = op.generating_element.threshold,
                    epsilon = op.generating_element.threshold
                ) {
                    // Binary rotations or reflections
                    Point3::from(geometry::get_positive_pole(
                        &op.generating_element.axis,
                        op.generating_element.threshold,
                    ))
                } else if approx::relative_ne!(
                    op.total_proper_angle,
                    0.0,
                    max_relative = op.generating_element.threshold,
                    epsilon = op.generating_element.threshold
                ) {
                    Point3::from(op.total_proper_angle.signum() * op.generating_element.axis)
                } else {
                    approx::assert_relative_eq!(
                        op.total_proper_angle,
                        0.0,
                        max_relative = op.generating_element.threshold,
                        epsilon = op.generating_element.threshold
                    );
                    Point3::origin()
                }
            }
        }
    }

    /// Finds the quaternion associated with this operation.
    ///
    /// The rotation angle encoded in the quaternion is taken to be non-negative
    /// and assigned as the proper rotation angle associated with the element
    /// generating the operation.
    ///
    /// If this is an operation generated from an improper element, the
    /// inversion-centre convention will be used.
    ///
    /// See S.L. Altmann, Rotations, Quaternions, and Double Groups (Dover
    /// Publications, Inc., New York, 2005) (Chapter 9) for further information.
    ///
    /// # Returns
    ///
    /// The quaternion associated with this operation.
    ///
    /// # Panics
    ///
    /// Panics if the calculated scalar part of the quaternion lies outside the closed interval
    /// $`[0, 1]`$ by more than the threshold value stored in the generating element in `self`.
    #[must_use]
    pub fn calc_quaternion(&self) -> Quaternion {
        let c_self = match self.generating_element.kind {
            SymmetryElementKind::Proper(_) | SymmetryElementKind::ImproperInversionCentre(_) => {
                self.clone()
            }
            SymmetryElementKind::ImproperMirrorPlane(_) => self.convert_to_improper_kind(&INV),
        };

        // We only need the absolute value of the angle. Its sign information is
        // encoded in the pole.
        let abs_angle = c_self.total_proper_angle.abs();
        let scalar_part = (0.5 * abs_angle).cos();
        let vector_part = (0.5 * abs_angle).sin() * c_self.calc_pole().coords;
        assert!(
            -self.generating_element.threshold <= scalar_part
                && scalar_part <= 1.0 + self.generating_element.threshold
        );
        (scalar_part, vector_part)
    }

    /// Returns a copy of the current symmetry operation with the generating element
    /// converted to the requested improper kind (power-preserving), provided that
    /// it is an improper element.
    ///
    /// # Arguments
    ///
    /// * `improper_kind` - The improper kind to which `self` is to be converted. There is no need to
    /// make sure the time reversal specification in `improper_kind` matches that of the generating
    /// element of `self` as the conversion will take care of this.
    ///
    /// # Panics
    ///
    /// Panics if the converted symmetry operation cannot be constructed.
    #[must_use]
    pub fn convert_to_improper_kind(&self, improper_kind: &SymmetryElementKind) -> Self {
        let c_element = self
            .generating_element
            .convert_to_improper_kind(improper_kind, true);
        Self::builder()
            .generating_element(c_element)
            .power(self.power)
            .build()
            .expect("Unable to construct a symmetry operation.")
    }

    /// Generates the abbreviated symbol for this symmetry operation, which classifies
    /// certain improper axes into inversion centres or mirror planes,
    #[must_use]
    pub fn get_abbreviated_symbol(&self) -> String {
        if self.power == 1 {
            self.generating_element.get_detailed_symbol()
        } else {
            format!(
                "[{}]^{}",
                self.generating_element.get_detailed_symbol(),
                self.power
            )
        }
    }
}

impl FiniteOrder for SymmetryOperation {
    type Int = u32;

    /// Calculates the order of this symmetry operation.
    fn order(&self) -> Self::Int {
        let denom = *self
            .total_proper_fraction
            .expect("No total proper fractions found.")
            .denom()
            .expect("Unable to extract the denominator.");
        if (self.is_proper() && !self.is_antiunitary()) || denom.rem_euclid(2) == 0 {
            denom
        } else {
            2 * denom
        }
    }
}

impl SpecialSymmetryTransformation for SymmetryOperation {
    /// Checks if the spatial part of the symmetry operation is proper or not.
    ///
    /// # Returns
    ///
    /// A flag indicating if the spatial part of the symmetry operation is proper.
    fn is_proper(&self) -> bool {
        self.generating_element.is_proper(true)
            || self.generating_element.is_proper(false)
            || (self.power % 2 == 0)
    }

    /// Checks if the symmetry operation is antiunitary or not.
    ///
    /// # Returns
    ///
    /// A flag indicating if the symmetry oppperation is antiunitary.
    fn is_antiunitary(&self) -> bool {
        self.generating_element.contains_time_reversal() && self.power % 2 == 1
    }

    /// Checks if the whole symmetry operation is the identity.
    ///
    /// # Returns
    ///
    /// A flag indicating if this symmetry operation is the identity.
    fn is_identity(&self) -> bool {
        self.is_proper()
            && !self.is_antiunitary()
            && match self.generating_element.proper_order {
                ElementOrder::Int(_) => self.total_proper_fraction == Some(F::from(1u64)),
                ElementOrder::Inf => {
                    approx::relative_eq!(
                        geometry::normalise_rotation_angle(
                            self.generating_element
                                .proper_angle
                                .expect("No proper angles found for the generating element.")
                                * (f64::from(self.power)),
                            self.generating_element.threshold
                        ) % (2.0 * std::f64::consts::PI),
                        0.0,
                        max_relative = self.generating_element.threshold,
                        epsilon = self.generating_element.threshold
                    )
                }
            }
    }

    /// Checks if the whole symmetry operation is an inversion.
    ///
    /// # Returns
    ///
    /// A flag indicating if this symmetry operation is an inversion.
    fn is_inversion(&self) -> bool {
        !self.is_proper()
            && !self.is_antiunitary()
            && match self.generating_element.kind {
                SymmetryElementKind::ImproperMirrorPlane(false) => {
                    if let ElementOrder::Int(_) = self.generating_element.proper_order {
                        self.total_proper_fraction == Some(F::new(1u32, 2u32))
                    } else {
                        approx::relative_eq!(
                            geometry::normalise_rotation_angle(
                                self.generating_element
                                    .proper_angle
                                    .expect("No proper angles found for the generating element.")
                                    * (f64::from(self.power)),
                                self.generating_element.threshold
                            ) % (2.0 * std::f64::consts::PI),
                            std::f64::consts::PI,
                            max_relative = self.generating_element.threshold,
                            epsilon = self.generating_element.threshold
                        )
                    }
                }
                SymmetryElementKind::ImproperInversionCentre(false) => {
                    if let ElementOrder::Int(_) = self.generating_element.proper_order {
                        self.total_proper_fraction == Some(F::from(1u64))
                    } else {
                        approx::relative_eq!(
                            geometry::normalise_rotation_angle(
                                self.generating_element
                                    .proper_angle
                                    .expect("No proper angles found for the generating element.")
                                    * (f64::from(self.power)),
                                self.generating_element.threshold
                            ) % (2.0 * std::f64::consts::PI),
                            0.0,
                            max_relative = self.generating_element.threshold,
                            epsilon = self.generating_element.threshold
                        )
                    }
                }
                _ => false,
            }
    }

    /// Checks if the whole symmetry operation is an inversion accompanied by a time reversal.
    ///
    /// # Returns
    ///
    /// A flag indicating if this symmetry operation is an inversion accompanied by a time reversal.
    fn is_tr_inversion(&self) -> bool {
        !self.is_proper()
            && self.is_antiunitary()
            && match self.generating_element.kind {
                SymmetryElementKind::ImproperMirrorPlane(true) => {
                    if let ElementOrder::Int(_) = self.generating_element.proper_order {
                        self.total_proper_fraction == Some(F::new(1u32, 2u32))
                    } else {
                        approx::relative_eq!(
                            geometry::normalise_rotation_angle(
                                self.generating_element
                                    .proper_angle
                                    .expect("No proper angles found for the generating element.")
                                    * (f64::from(self.power)),
                                self.generating_element.threshold
                            ) % (2.0 * std::f64::consts::PI),
                            std::f64::consts::PI,
                            max_relative = self.generating_element.threshold,
                            epsilon = self.generating_element.threshold
                        )
                    }
                }
                SymmetryElementKind::ImproperInversionCentre(true) => {
                    if let ElementOrder::Int(_) = self.generating_element.proper_order {
                        self.total_proper_fraction == Some(F::from(1u64))
                    } else {
                        approx::relative_eq!(
                            geometry::normalise_rotation_angle(
                                self.generating_element
                                    .proper_angle
                                    .expect("No proper angles found for the generating element.")
                                    * (f64::from(self.power)),
                                self.generating_element.threshold
                            ) % (2.0 * std::f64::consts::PI),
                            0.0,
                            max_relative = self.generating_element.threshold,
                            epsilon = self.generating_element.threshold
                        )
                    }
                }
                _ => false,
            }
    }

    /// Checks if the whole symmetry operation is a binary rotation.
    ///
    /// # Returns
    ///
    /// A flag indicating if this symmetry operation is a binary rotation.
    fn is_binary_rotation(&self) -> bool {
        self.is_proper()
            && !self.is_antiunitary()
            && match self.generating_element.proper_order {
                ElementOrder::Int(_) => self.total_proper_fraction == Some(F::new(1u32, 2u32)),
                ElementOrder::Inf => {
                    approx::relative_eq!(
                        geometry::normalise_rotation_angle(
                            self.generating_element
                                .proper_angle
                                .expect("No proper angles found for the generating element.")
                                * (f64::from(self.power)),
                            self.generating_element.threshold
                        ) % (2.0 * std::f64::consts::PI),
                        std::f64::consts::PI,
                        max_relative = self.generating_element.threshold,
                        epsilon = self.generating_element.threshold
                    )
                }
            }
    }

    /// Checks if the whole symmetry operation is a binary rotation accompanied by a time reversal.
    ///
    /// # Returns
    ///
    /// A flag indicating if this symmetry operation is a binary rotation.
    fn is_tr_binary_rotation(&self) -> bool {
        self.is_proper()
            && self.is_antiunitary()
            && match self.generating_element.proper_order {
                ElementOrder::Int(_) => self.total_proper_fraction == Some(F::new(1u32, 2u32)),
                ElementOrder::Inf => {
                    approx::relative_eq!(
                        geometry::normalise_rotation_angle(
                            self.generating_element
                                .proper_angle
                                .expect("No proper angles found for the generating element.")
                                * (f64::from(self.power)),
                            self.generating_element.threshold
                        ) % (2.0 * std::f64::consts::PI),
                        std::f64::consts::PI,
                        max_relative = self.generating_element.threshold,
                        epsilon = self.generating_element.threshold
                    )
                }
            }
    }

    /// Checks if the whole symmetry operation is a reflection.
    ///
    /// # Returns
    ///
    /// A flag indicating if this symmetry operation is a reflection.
    fn is_reflection(&self) -> bool {
        !self.is_proper()
            && !self.is_antiunitary()
            && match self.generating_element.kind {
                SymmetryElementKind::ImproperMirrorPlane(false) => {
                    if let ElementOrder::Int(_) = self.generating_element.proper_order {
                        self.total_proper_fraction == Some(F::from(1u64))
                    } else {
                        approx::relative_eq!(
                            geometry::normalise_rotation_angle(
                                self.generating_element
                                    .proper_angle
                                    .expect("No proper angles found for the generating element.")
                                    * (f64::from(self.power)),
                                self.generating_element.threshold
                            ) % (2.0 * std::f64::consts::PI),
                            0.0,
                            max_relative = self.generating_element.threshold,
                            epsilon = self.generating_element.threshold
                        )
                    }
                }
                SymmetryElementKind::ImproperInversionCentre(false) => {
                    if let ElementOrder::Int(_) = self.generating_element.proper_order {
                        self.total_proper_fraction == Some(F::new(1u32, 2u32))
                    } else {
                        approx::relative_eq!(
                            geometry::normalise_rotation_angle(
                                self.generating_element
                                    .proper_angle
                                    .expect("No proper angles found for the generating element.")
                                    * (f64::from(self.power)),
                                self.generating_element.threshold
                            ) % (2.0 * std::f64::consts::PI),
                            std::f64::consts::PI,
                            max_relative = self.generating_element.threshold,
                            epsilon = self.generating_element.threshold
                        )
                    }
                }
                _ => false,
            }
    }

    /// Checks if the whole symmetry operation is a pure time-reversal.
    ///
    /// # Returns
    ///
    /// A flag indicating if this symmetry operation is a pure time-reversal.
    fn is_time_reversal(&self) -> bool {
        self.is_proper()
            && self.is_antiunitary()
            && match self.generating_element.proper_order {
                ElementOrder::Int(_) => self.total_proper_fraction == Some(F::from(1u64)),
                ElementOrder::Inf => {
                    approx::relative_eq!(
                        geometry::normalise_rotation_angle(
                            self.generating_element
                                .proper_angle
                                .expect("No proper angles found for the generating element.")
                                * (f64::from(self.power)),
                            self.generating_element.threshold
                        ) % (2.0 * std::f64::consts::PI),
                        0.0,
                        max_relative = self.generating_element.threshold,
                        epsilon = self.generating_element.threshold
                    )
                }
            }
    }

    /// Checks if the whole symmetry operation is a reflection accompanied by a time reversal.
    ///
    /// # Returns
    ///
    /// A flag indicating if this symmetry operation is a reflection accompanied by a time reversal.
    fn is_tr_reflection(&self) -> bool {
        !self.is_proper()
            && self.is_antiunitary()
            && match self.generating_element.kind {
                SymmetryElementKind::ImproperMirrorPlane(true) => {
                    if let ElementOrder::Int(_) = self.generating_element.proper_order {
                        self.total_proper_fraction == Some(F::from(1u64))
                    } else {
                        approx::relative_eq!(
                            geometry::normalise_rotation_angle(
                                self.generating_element
                                    .proper_angle
                                    .expect("No proper angles found for the generating element.")
                                    * (f64::from(self.power)),
                                self.generating_element.threshold
                            ) % (2.0 * std::f64::consts::PI),
                            0.0,
                            max_relative = self.generating_element.threshold,
                            epsilon = self.generating_element.threshold
                        )
                    }
                }
                SymmetryElementKind::ImproperInversionCentre(true) => {
                    if let ElementOrder::Int(_) = self.generating_element.proper_order {
                        self.total_proper_fraction == Some(F::new(1u32, 2u32))
                    } else {
                        approx::relative_eq!(
                            geometry::normalise_rotation_angle(
                                self.generating_element
                                    .proper_angle
                                    .expect("No proper angles found for the generating element.")
                                    * (f64::from(self.power)),
                                self.generating_element.threshold
                            ) % (2.0 * std::f64::consts::PI),
                            std::f64::consts::PI,
                            max_relative = self.generating_element.threshold,
                            epsilon = self.generating_element.threshold
                        )
                    }
                }
                _ => false,
            }
    }
}

impl fmt::Debug for SymmetryOperation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.power == 1 {
            write!(f, "{:?}", self.generating_element)
        } else {
            write!(f, "[{:?}]^{}", self.generating_element, self.power)
        }
    }
}

impl fmt::Display for SymmetryOperation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.power == 1 {
            write!(f, "{}", self.generating_element)
        } else {
            write!(f, "[{}]^{}", self.generating_element, self.power)
        }
    }
}

impl PartialEq for SymmetryOperation {
    fn eq(&self, other: &Self) -> bool {
        if (self.generating_element.proper_order == ElementOrder::Inf)
            != (other.generating_element.proper_order == ElementOrder::Inf)
        {
            // We disable comparisons between operations with infinite-order and
            // finite-order generating elements, because they cannot be made to
            // have the same hashes without losing the fidelity of exact-fraction
            // representations for operations with finite-order generating elements.
            return false;
        }

        if self.is_antiunitary() != other.is_antiunitary() {
            return false;
        }

        if self.is_proper() != other.is_proper() {
            return false;
        }

        if self.is_time_reversal() && other.is_time_reversal() {
            assert_eq!(misc::calculate_hash(self), misc::calculate_hash(other));
            return true;
        }

        if self.is_identity() && other.is_identity() {
            assert_eq!(misc::calculate_hash(self), misc::calculate_hash(other));
            return true;
        }

        if self.is_inversion() && other.is_inversion() {
            assert_eq!(misc::calculate_hash(self), misc::calculate_hash(other));
            return true;
        }

        if self.is_tr_inversion() && other.is_tr_inversion() {
            assert_eq!(misc::calculate_hash(self), misc::calculate_hash(other));
            return true;
        }

        let thresh =
            (self.generating_element.threshold * other.generating_element.threshold).sqrt();

        let result = if (self.is_binary_rotation() && other.is_binary_rotation())
            || (self.is_reflection() && other.is_reflection())
            || (self.is_tr_binary_rotation() && other.is_tr_binary_rotation())
            || (self.is_tr_reflection() && other.is_tr_reflection())
        {
            approx::relative_eq!(
                self.calc_pole(),
                other.calc_pole(),
                epsilon = thresh,
                max_relative = thresh
            )
        } else {
            let c_self = if self.is_proper() {
                self.clone()
            } else {
                self.convert_to_improper_kind(&INV)
            };
            let c_other = if other.is_proper() {
                other.clone()
            } else {
                other.convert_to_improper_kind(&INV)
            };

            let angle_comparison = if let Some(s_frac) = c_self.total_proper_fraction {
                if let Some(o_frac) = c_other.total_proper_fraction {
                    let abs_s_frac = if s_frac < F::new(1u32, 2u32) {
                        s_frac
                    } else {
                        F::from(1u64) - s_frac
                    };
                    let abs_o_frac = if o_frac < F::new(1u32, 2u32) {
                        o_frac
                    } else {
                        F::from(1u64) - o_frac
                    };
                    abs_s_frac == abs_o_frac
                } else {
                    approx::relative_eq!(
                        c_self.total_proper_angle.abs(),
                        c_other.total_proper_angle.abs(),
                        epsilon = thresh,
                        max_relative = thresh
                    )
                }
            } else {
                approx::relative_eq!(
                    c_self.total_proper_angle.abs(),
                    c_other.total_proper_angle.abs(),
                    epsilon = thresh,
                    max_relative = thresh
                )
            };

            angle_comparison
                && approx::relative_eq!(
                    self.calc_pole(),
                    other.calc_pole(),
                    epsilon = thresh,
                    max_relative = thresh
                )
        };

        if result {
            assert_eq!(
                misc::calculate_hash(self),
                misc::calculate_hash(other),
                "{self} and {other} have unequal hashes.",
            );
        }
        result
    }
}

impl Eq for SymmetryOperation {}

impl Hash for SymmetryOperation {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let c_self = match self.generating_element.kind {
            SymmetryElementKind::Proper(_) | SymmetryElementKind::ImproperInversionCentre(_) => {
                self.clone()
            }
            SymmetryElementKind::ImproperMirrorPlane(_) => self.convert_to_improper_kind(&INV),
        };
        c_self.is_antiunitary().hash(state);
        c_self.is_proper().hash(state);
        if c_self.is_identity()
            || c_self.is_inversion()
            || c_self.is_time_reversal()
            || c_self.is_tr_inversion()
        {
            true.hash(state);
        } else {
            let pole = c_self.calc_pole();
            pole[0]
                .round_factor(c_self.generating_element.threshold)
                .integer_decode()
                .hash(state);
            pole[1]
                .round_factor(c_self.generating_element.threshold)
                .integer_decode()
                .hash(state);
            pole[2]
                .round_factor(c_self.generating_element.threshold)
                .integer_decode()
                .hash(state);

            if !c_self.is_binary_rotation()
                && !c_self.is_reflection()
                && !c_self.is_tr_binary_rotation()
                && !c_self.is_tr_reflection()
            {
                if let Some(frac) = c_self.total_proper_fraction {
                    // frac lies in (0, 1/2) ∪ (1/2, 1).
                    // 1/2 and 1 are excluded because this is not an identity,
                    // inversion, binary rotation, or reflection.
                    let abs_frac = if frac < F::new(1u32, 2u32) {
                        frac
                    } else {
                        F::from(1u32) - frac
                    };
                    abs_frac.hash(state);
                } else {
                    // self.total_proper_angle lies in (-π, 0) ∪ (0, π).
                    // 0 and π are excluded because this is not an identity,
                    // inversion, binary rotation, or reflection.
                    let abs_ang = c_self.total_proper_angle.abs();
                    abs_ang
                        .round_factor(c_self.generating_element.threshold)
                        .integer_decode()
                        .hash(state);
                };
            }
        };
    }
}

impl Mul<&'_ SymmetryOperation> for &SymmetryOperation {
    type Output = SymmetryOperation;

    fn mul(self, rhs: &SymmetryOperation) -> Self::Output {
        let (q1_s, q1_v) = self.calc_quaternion();
        let (q2_s, q2_v) = rhs.calc_quaternion();

        let q3_s = q1_s * q2_s - q1_v.dot(&q2_v);
        let q3_v = q1_s * q2_v + q2_s * q1_v + q1_v.cross(&q2_v);

        let q3 = if q3_s >= 0.0 {
            (q3_s, q3_v)
        } else {
            (-q3_s, -q3_v)
        };

        let proper = self.is_proper() == rhs.is_proper();
        let thresh = (self.generating_element.threshold * rhs.generating_element.threshold).sqrt();
        let max_trial_power = u32::MAX;
        SymmetryOperation::from_quaternion(
            q3,
            proper,
            thresh,
            max_trial_power,
            self.is_antiunitary() != rhs.is_antiunitary(),
        )
    }
}

impl Mul<&'_ SymmetryOperation> for SymmetryOperation {
    type Output = SymmetryOperation;

    fn mul(self, rhs: &SymmetryOperation) -> Self::Output {
        &self * rhs
    }
}

impl Mul<SymmetryOperation> for SymmetryOperation {
    type Output = SymmetryOperation;

    fn mul(self, rhs: SymmetryOperation) -> Self::Output {
        &self * &rhs
    }
}

impl Mul<SymmetryOperation> for &SymmetryOperation {
    type Output = SymmetryOperation;

    fn mul(self, rhs: SymmetryOperation) -> Self::Output {
        self * &rhs
    }
}

impl Pow<i32> for &SymmetryOperation {
    type Output = SymmetryOperation;

    fn pow(self, rhs: i32) -> Self::Output {
        SymmetryOperation::builder()
            .generating_element(self.generating_element.clone())
            .power(self.power * rhs)
            .build()
            .expect("Unable to construct a symmetry operation.")
    }
}

impl Pow<i32> for SymmetryOperation {
    type Output = SymmetryOperation;

    fn pow(self, rhs: i32) -> Self::Output {
        (&self).pow(rhs)
    }
}
