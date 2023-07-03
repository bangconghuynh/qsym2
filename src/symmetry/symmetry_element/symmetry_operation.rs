use std::fmt;
use std::hash::{Hash, Hasher};
use std::ops::Mul;

use approx;
use derive_builder::Builder;
use fraction;
use nalgebra::{Point3, Vector3};
use ndarray::{Array2, Axis, ShapeBuilder};
use num_traits::{Inv, Pow, Zero};
use ordered_float::OrderedFloat;
use serde::{Deserialize, Serialize};

use crate::aux::geometry::{
    self, improper_rotation_matrix, proper_rotation_matrix, PositiveHemisphere, Transform, IMINV,
};
use crate::aux::misc::{self, HashableFloat};
use crate::group::FiniteOrder;
use crate::permutation::{IntoPermutation, PermutableCollection, Permutation};
use crate::symmetry::symmetry_element::{
    AntiunitaryKind, SymmetryElement, SymmetryElementKind, INV, ROT, SIG, SO3, SU2_0, SU2_1, TRINV,
    TRROT, TRSIG,
};
use crate::symmetry::symmetry_element_order::ElementOrder;

type F = fraction::GenericFraction<u32>;
type Quaternion = (f64, Vector3<f64>);

#[cfg(test)]
#[path = "symmetry_operation_tests.rs"]
mod symmetry_operation_tests;

// =================
// Trait definitions
// =================

/// A trait for special symmetry transformations.
pub trait SpecialSymmetryTransformation {
    // =================
    // Group-theoretical
    // =================

    /// Checks if the proper rotation part of the symmetry operation is in $`\mathsf{SU}(2)`$.
    ///
    /// # Returns
    ///
    /// A boolean indicating if this symmetry operation contains an $`\mathsf{SU}(2)`$ proper
    /// rotation.
    fn is_su2(&self) -> bool;

    /// Checks if the proper rotation part of the symmetry operation is in $`\mathsf{SU}(2)`$ and
    /// connected to the identity via a homotopy path of class 1.
    ///
    /// # Returns
    ///
    /// A boolean indicating if this symmetry operation contains an $`\mathsf{SU}(2)`$ proper
    /// rotation connected to the identity via a homotopy path of class 1.
    fn is_su2_class_1(&self) -> bool;

    // ============
    // Spatial part
    // ============

    /// Checks if the spatial part of the symmetry operation is proper.
    ///
    /// # Returns
    ///
    /// A boolean indicating if the spatial part of the symmetry operation is proper.
    fn is_proper(&self) -> bool;

    /// Checks if the spatial part of the symmetry operation is the spatial identity.
    ///
    /// # Returns
    ///
    /// A boolean indicating if the spatial part of the symmetry operation is the spatial identity.
    fn is_spatial_identity(&self) -> bool;

    /// Checks if the spatial part of the symmetry operation is a spatial binary rotation.
    ///
    /// # Returns
    ///
    /// A boolean indicating if the spatial part of the symmetry operation is a spatial binary
    /// rotation.
    fn is_spatial_binary_rotation(&self) -> bool;

    /// Checks if the spatial part of the symmetry operation is the spatial inversion.
    ///
    /// # Returns
    ///
    /// A boolean indicating if the spatial part of the symmetry operation is the spatial inversion.
    fn is_spatial_inversion(&self) -> bool;

    /// Checks if the spatial part of the symmetry operation is a spatial reflection.
    ///
    /// # Returns
    ///
    /// A boolean indicating if the spatial part of the symmetry operation is a spatial reflection.
    fn is_spatial_reflection(&self) -> bool;

    // ==================
    // Time-reversal part
    // ==================

    /// Checks if the symmetry operation is antiunitary.
    ///
    /// # Returns
    ///
    /// A boolean indicating if the symmetry oppperation is antiunitary.
    fn is_antiunitary(&self) -> bool;

    // ==========================
    // Overall - provided methods
    // ==========================

    /// Checks if the symmetry operation is the identity in $`\mathsf{O}(3)`$, `E`, or
    /// in $`\mathsf{SU}(2)`$, `E(Σ)`.
    ///
    /// # Returns
    ///
    /// A boolean indicating if this symmetry operation is the identity.
    fn is_identity(&self) -> bool {
        self.is_spatial_identity() && !self.is_antiunitary() && !self.is_su2_class_1()
    }

    /// Checks if the symmetry operation is a pure time-reversal in $`\mathsf{O}(3)`$, `θ`, or
    /// in $`\mathsf{SU}(2)`$, `θ(Σ)`.
    ///
    /// # Returns
    ///
    /// A boolean indicating if this symmetry operation is a pure time-reversal.
    fn is_time_reversal(&self) -> bool {
        self.is_spatial_identity() && self.is_antiunitary() && !self.is_su2_class_1()
    }

    /// Checks if the symmetry operation is an inversion in $`\mathsf{O}(3)`$, `i`, but not in
    /// $`\mathsf{SU}(2)`$, `i(Σ)`.
    ///
    /// # Returns
    ///
    /// A boolean indicating if this symmetry operation is an inversion in $`\mathsf{O}(3)`$.
    fn is_inversion(&self) -> bool {
        self.is_spatial_inversion() && !self.is_antiunitary() && !self.is_su2()
    }

    /// Checks if the symmetry operation is a reflection in $`\mathsf{O}(3)`$, `σ`, but not in
    /// $`\mathsf{SU}(2)`$, `σ(Σ)`.
    ///
    /// # Returns
    ///
    /// A boolean indicating if this symmetry operation is a reflection in $`\mathsf{O}(3)`$.
    fn is_reflection(&self) -> bool {
        self.is_spatial_reflection() && !self.is_antiunitary() && !self.is_su2()
    }
}

// ======================================
// Struct definitions and implementations
// ======================================

/// A structure for managing symmetry operations generated from symmetry elements.
///
/// A symmetry element serves as a generator for symmetry operations. Thus, a symmetry element
/// together with a signed integer indicating the number of times the symmetry element is applied
/// (positively or negatively) specifies a symmetry operation.
#[derive(Builder, Clone, Serialize, Deserialize)]
pub struct SymmetryOperation {
    /// The generating symmetry element for this symmetry operation.
    pub generating_element: SymmetryElement,

    /// The integral power indicating the number of times
    /// [`Self::generating_element`] is applied to form the symmetry operation.
    pub power: i32,

    /// The total proper rotation angle associated with this operation (after taking into account
    /// the power of the operation).
    ///
    /// This is simply the proper rotation angle of [`Self::generating_element`] multiplied by
    /// [`Self::power`] and then folded onto the open interval $`(-\pi, \pi]`$.
    ///
    /// This angle lies in the open interval $`(-\pi, \pi]`$. For improper operations, this angle
    /// depends on the convention used to describe the [`Self::generating_element`].
    #[builder(setter(skip), default = "self.calc_total_proper_angle()")]
    total_proper_angle: f64,

    /// The fraction $`pk/n \in (-1/2, 1/2]`$ of the proper rotation, represented exactly for
    /// hashing and comparison purposes.
    ///
    /// This is not defined for operations with infinite-order generating elements.
    #[builder(setter(skip), default = "self.calc_total_proper_fraction()")]
    pub(crate) total_proper_fraction: Option<F>,

    /// The positive hemisphere used for distinguishing positive and negative rotation poles. If
    /// `None`, the standard positive hemisphere as defined in S.L. Altmann, Rotations,
    /// Quaternions, and Double Groups (Dover Publications, Inc., New York, 2005) is used.
    #[builder(default = "None")]
    pub positive_hemisphere: Option<PositiveHemisphere>,
}

impl SymmetryOperationBuilder {
    fn calc_total_proper_angle(&self) -> f64 {
        let (total_proper_angle, _) = geometry::normalise_rotation_angle(
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
        );
        total_proper_angle
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
                let (total_proper_fraction, _) =
                    geometry::normalise_rotation_fraction(frac * F::from(pow));
                Some(total_proper_fraction)
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
    pub(crate) fn builder() -> SymmetryOperationBuilder {
        SymmetryOperationBuilder::default()
    }

    /// Constructs a finite-order-element-generated symmetry operation from a quaternion.
    ///
    /// The rotation angle encoded in the quaternion is taken to be non-negative and assigned as
    /// the proper rotation angle associated with the element generating the operation.
    ///
    /// If an improper operation is required, its generator will be constructed in the
    /// inversion-centre convention.
    ///
    /// # Arguments
    ///
    /// * `qtn` - A quaternion encoding the proper rotation associated with the
    /// generating element of the operation to be constructed.
    /// * `proper` - A boolean indicating if the operation is proper or improper.
    /// * `thresh` - Threshold for comparisons.
    /// * `tr` - A boolean indicating if the resulting symmetry operation should be accompanied by
    /// a time-reversal operation.
    /// * `su2` - A boolean indicating if the resulting symmetry operation is to contain a proper
    /// rotation in $`\mathsf{SU}(2)`$. The homotopy class of the operation will be deduced from
    /// the specified quaternion.
    /// * `poshem` - An option containing any custom positive hemisphere used to distinguish
    /// positive and negative rotation poles.
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
        su2: bool,
        poshem: Option<PositiveHemisphere>,
    ) -> Self {
        let (scalar_part, vector_part) = qtn;
        let kind = if proper {
            if tr {
                TRROT
            } else {
                ROT
            }
        } else if tr {
            TRINV
        } else {
            INV
        };
        let element = if su2 {
            // SU(2)
            assert!(
                -1.0 - thresh <= scalar_part && scalar_part <= 1.0 + thresh,
                "The scalar part of the quaternion must be in the interval [-1, +1]."
            );
            let (axis, order, power, su2_grp) = if approx::relative_eq!(
                scalar_part,
                1.0,
                epsilon = thresh,
                max_relative = thresh
            ) {
                // Zero-degree rotation, i.e. identity or inversion, class 0
                (Vector3::z(), 1u32, 1u32, SU2_0)
            } else if approx::relative_eq!(
                scalar_part,
                -1.0,
                epsilon = thresh,
                max_relative = thresh
            ) {
                // 360-degree rotation, i.e. identity or inversion, class 1
                (Vector3::z(), 1u32, 1u32, SU2_1)
            } else if approx::relative_eq!(
                scalar_part,
                0.0,
                epsilon = thresh,
                max_relative = thresh
            ) {
                // 180-degree rotation, i.e. binary rotation or reflection. Whether the resultant
                // operation is in class 0 or class 1 depends on whether the vector part is in the
                // positive hemisphere or negative hemisphere.
                let positive_axis = poshem
                    .as_ref()
                    .cloned()
                    .unwrap_or_default()
                    .get_positive_pole(&vector_part, thresh);
                (
                    positive_axis,
                    2u32,
                    1u32,
                    if poshem
                        .as_ref()
                        .cloned()
                        .unwrap_or_default()
                        .check_positive_pole(&vector_part, thresh)
                    {
                        SU2_0
                    } else {
                        SU2_1
                    },
                )
            } else {
                // scalar_part != 0, 1, or -1
                let (standardised_scalar_part, standardised_vector_part, su2_grp) =
                    if scalar_part > 0.0 {
                        (scalar_part, vector_part, SU2_0)
                    } else {
                        (-scalar_part, -vector_part, SU2_1)
                    };
                let half_proper_angle = standardised_scalar_part.acos();
                let proper_angle = 2.0 * half_proper_angle;
                let axis = standardised_vector_part / half_proper_angle.sin();
                let proper_fraction =
                    geometry::get_proper_fraction(proper_angle, thresh, max_trial_power)
                        .unwrap_or_else(|| {
                            panic!("No proper fraction could be found for angle `{proper_angle}`.")
                        });
                (
                    axis,
                    *proper_fraction.denom().unwrap_or_else(|| {
                        panic!("Unable to extract the denominator of `{proper_fraction}`.")
                    }),
                    *proper_fraction.numer().unwrap_or_else(|| {
                        panic!("Unable to extract the numerator of `{proper_fraction}`.")
                    }),
                    su2_grp,
                )
            };
            SymmetryElement::builder()
                .threshold(thresh)
                .proper_order(ElementOrder::Int(order))
                .proper_power(
                    power
                        .try_into()
                        .expect("Unable to convert the proper power to `i32`."),
                )
                .raw_axis(axis)
                .kind(kind)
                .rotation_group(su2_grp)
                .build()
                .unwrap_or_else(|_|
                    panic!("Unable to construct a symmetry element of kind `{kind}` with the proper part in SU(2).")
                )
        } else {
            // SO(3)
            assert!(
                -thresh <= scalar_part && scalar_part <= 1.0 + thresh,
                "The scalar part of the quaternion must be in the interval [0, +1] when only SO(3) rotations are considered."
            );
            let (axis, order, power) = if approx::relative_eq!(
                scalar_part,
                1.0,
                epsilon = thresh,
                max_relative = thresh
            ) {
                // Zero-degree rotation, i.e. identity or inversion
                (Vector3::z(), 1u32, 1i32)
            } else {
                let half_proper_angle = scalar_part.acos(); // acos returns values in [0, π]
                let proper_angle = 2.0 * half_proper_angle;
                let axis = vector_part / half_proper_angle.sin();
                let proper_fraction =
                    geometry::get_proper_fraction(proper_angle, thresh, max_trial_power)
                        .unwrap_or_else(|| {
                            panic!("No proper fraction could be found for angle `{proper_angle}`.")
                        });
                let proper_power = if proper_fraction.is_sign_positive() {
                    i32::try_from(*proper_fraction.numer().unwrap_or_else(|| {
                        panic!("Unable to extract the numerator of `{proper_fraction}`.")
                    }))
                    .expect("Unable to convert the numerator of the proper fraction to `i32`.")
                } else {
                    -i32::try_from(*proper_fraction.numer().unwrap_or_else(|| {
                        panic!("Unable to extract the numerator of `{proper_fraction}`.")
                    }))
                    .expect("Unable to convert the numerator of the proper fraction to `i32`.")
                };
                (
                    axis,
                    *proper_fraction.denom().unwrap_or_else(|| {
                        panic!("Unable to extract the denominator of `{proper_fraction}`.")
                    }),
                    proper_power,
                )
            };

            SymmetryElement::builder()
                .threshold(thresh)
                .proper_order(ElementOrder::Int(order))
                .proper_power(power)
                .raw_axis(axis)
                .kind(kind)
                .rotation_group(SO3)
                .build()
                .expect(
                    "Unable to construct a symmetry element without an associated spin rotation.",
                )
        };

        SymmetryOperation::builder()
            .generating_element(element)
            .power(1)
            .positive_hemisphere(poshem)
            .build()
            .unwrap_or_else(|_|
                panic!(
                    "Unable to construct a symmetry operation of kind `{kind}` with {} rotation from the quaternion `{qtn:?}`.",
                    if su2 { "SU(2)" } else { "SO(3)" }
                )
            )
    }

    /// Finds the quaternion associated with this operation.
    ///
    /// The rotation angle encoded in the quaternion is taken to be non-negative and assigned as
    /// the proper rotation angle associated with the element generating the operation.
    ///
    /// If this is an operation generated from an improper element, the inversion-centre convention
    /// will be used to determine the angle of proper rotation.
    ///
    /// Both $`\mathsf{SO}(3)`$ and $`\mathsf{SU}(2)`$ proper rotations are supported. For
    /// $`\mathsf{SO}(3)`$ proper rotations, only quaternions in the standardised form are
    /// returned.
    ///
    /// See S.L. Altmann, Rotations, Quaternions, and Double Groups (Dover Publications, Inc., New
    /// York, 2005) (Chapter 9) for further information.
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
        let c_self = if self.is_proper() {
            self.clone()
        } else {
            // Time-reversal does not matter here.
            self.convert_to_improper_kind(&INV)
        };
        debug_assert_eq!(
            self.is_su2_class_1(),
            c_self.is_su2_class_1(),
            "`{self}` and `{c_self}` are in different homotopy classes."
        );

        // We only need the absolute value of the angle. Its sign information is
        // encoded in the pole. `abs_angle` thus lies in [0, π], and so
        //  cos(abs_angle/2) >= 0 and sin(abs_angle/2) >= 0.
        // The scalar part is guaranteed to be in [0, 1].
        // For binary rotations, the scalar part is zero, but the definition of pole ensures that
        // the vector part still lies in the positive hemisphere.
        let abs_angle = c_self.total_proper_angle.abs();
        let scalar_part = (0.5 * abs_angle).cos();
        let vector_part = (0.5 * abs_angle).sin() * c_self.calc_pole().coords;
        debug_assert!(
            -self.generating_element.threshold <= scalar_part
                && scalar_part <= 1.0 + self.generating_element.threshold
        );
        debug_assert!(if approx::relative_eq!(
            scalar_part,
            0.0,
            max_relative = c_self.generating_element.threshold,
            epsilon = c_self.generating_element.threshold
        ) {
            c_self
                .positive_hemisphere
                .as_ref()
                .cloned()
                .unwrap_or_default()
                .check_positive_pole(&vector_part, c_self.generating_element.threshold)
        } else {
            true
        },);

        if self.is_su2_class_1() {
            (-scalar_part, -vector_part)
        } else {
            (scalar_part, vector_part)
        }
    }

    /// Finds the pole associated with this operation with respect to the positive hemisphere
    /// defined in [`Self::positive_hemisphere`].
    ///
    /// This is the point on the unit sphere that is left invariant by the operation.
    ///
    /// For improper operations, the inversion-centre convention is used to define
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
            // Time-reversal does not matter here.
            self.convert_to_improper_kind(&INV)
        };
        match *op.generating_element.raw_proper_order() {
            ElementOrder::Int(_) => {
                let frac_1_2 = F::new(1u32, 2u32);
                let total_proper_fraction = op
                    .total_proper_fraction
                    .expect("No total proper fractions found.");
                if total_proper_fraction == frac_1_2 {
                    // Binary rotations or reflections
                    Point3::from(
                        self.positive_hemisphere
                            .as_ref()
                            .cloned()
                            .unwrap_or_default()
                            .get_positive_pole(
                                &op.generating_element.raw_axis,
                                op.generating_element.threshold,
                            ),
                    )
                } else if total_proper_fraction > F::zero() {
                    // Positive rotation angles
                    Point3::from(op.generating_element.raw_axis)
                } else if total_proper_fraction < F::zero() {
                    // Negative rotation angles
                    Point3::from(-op.generating_element.raw_axis)
                } else {
                    // Identity or inversion
                    assert!(total_proper_fraction.is_zero());
                    Point3::from(Vector3::z())
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
                    Point3::from(
                        self.positive_hemisphere
                            .as_ref()
                            .cloned()
                            .unwrap_or_default()
                            .get_positive_pole(
                                &op.generating_element.raw_axis,
                                op.generating_element.threshold,
                            ),
                    )
                } else if approx::relative_ne!(
                    op.total_proper_angle,
                    0.0,
                    max_relative = op.generating_element.threshold,
                    epsilon = op.generating_element.threshold
                ) {
                    Point3::from(op.total_proper_angle.signum() * op.generating_element.raw_axis)
                } else {
                    approx::assert_relative_eq!(
                        op.total_proper_angle,
                        0.0,
                        max_relative = op.generating_element.threshold,
                        epsilon = op.generating_element.threshold
                    );
                    Point3::from(Vector3::z())
                }
            }
        }
    }

    /// Finds the pole associated with the proper rotation of this operation.
    ///
    /// This is the point on the unit sphere that is left invariant by the proper rotation part of
    /// the operation.
    ///
    /// For improper operations, no conversions will be performed, unlike in [`Self::calc_pole`].
    ///
    /// Note that binary rotations have unique poles on the positive hemisphere (*i.e.*,
    /// $`C_2(\hat{\mathbf{n}}) = C_2^{-1}(\hat{\mathbf{n}})`$ and
    /// $`\sigma(\hat{\mathbf{n}}) = \sigma^{-1}(\hat{\mathbf{n}})`$).
    ///
    /// See S.L. Altmann, Rotations, Quaternions, and Double Groups (Dover
    /// Publications, Inc., New York, 2005) (Chapter 9) for further information.
    ///
    /// # Returns
    ///
    /// The pole associated with the proper rotation of this operation.
    ///
    /// # Panics
    ///
    /// Panics when no total proper fractions could be found for this operation.
    #[must_use]
    pub fn calc_proper_rotation_pole(&self) -> Point3<f64> {
        match *self.generating_element.raw_proper_order() {
            ElementOrder::Int(_) => {
                let frac_1_2 = F::new(1u32, 2u32);
                let total_proper_fraction = self
                    .total_proper_fraction
                    .expect("No total proper fractions found.");
                if total_proper_fraction == frac_1_2 {
                    // Binary rotations or reflections
                    Point3::from(
                        self.positive_hemisphere
                            .as_ref()
                            .cloned()
                            .unwrap_or_default()
                            .get_positive_pole(
                                &self.generating_element.raw_axis,
                                self.generating_element.threshold,
                            ),
                    )
                } else if total_proper_fraction > F::zero() {
                    // Positive rotation angles
                    Point3::from(self.generating_element.raw_axis)
                } else if total_proper_fraction < F::zero() {
                    // Negative rotation angles
                    Point3::from(-self.generating_element.raw_axis)
                } else {
                    // Identity or inversion
                    assert!(total_proper_fraction.is_zero());
                    Point3::from(Vector3::z())
                }
            }
            ElementOrder::Inf => {
                if approx::relative_eq!(
                    self.total_proper_angle,
                    std::f64::consts::PI,
                    max_relative = self.generating_element.threshold,
                    epsilon = self.generating_element.threshold
                ) {
                    // Binary rotations or reflections
                    Point3::from(
                        self.positive_hemisphere
                            .as_ref()
                            .cloned()
                            .unwrap_or_default()
                            .get_positive_pole(
                                &self.generating_element.raw_axis,
                                self.generating_element.threshold,
                            ),
                    )
                } else if approx::relative_ne!(
                    self.total_proper_angle,
                    0.0,
                    max_relative = self.generating_element.threshold,
                    epsilon = self.generating_element.threshold
                ) {
                    Point3::from(
                        self.total_proper_angle.signum() * self.generating_element.raw_axis,
                    )
                } else {
                    approx::assert_relative_eq!(
                        self.total_proper_angle,
                        0.0,
                        max_relative = self.generating_element.threshold,
                        epsilon = self.generating_element.threshold
                    );
                    Point3::from(Vector3::z())
                }
            }
        }
    }

    /// Finds the pole angle associated with this operation.
    ///
    /// This is the non-negative angle that, together with the pole, uniquely determines the proper
    /// part of this operation. This angle lies in the interval $`[0, \pi]`$.
    ///
    /// For improper operations, the inversion-centre convention is used to define the pole angle.
    /// This allows a proper rotation and its improper partner to have the same pole angle, thus
    /// facilitating the consistent specification of poles for the identity / inversion and binary
    /// rotations / reflections.
    ///
    /// # Returns
    ///
    /// The pole angle associated with this operation.
    ///
    /// # Panics
    ///
    /// Panics when no total proper fractions could be found for this operation.
    #[must_use]
    pub fn calc_pole_angle(&self) -> f64 {
        let c_self = if self.is_proper() {
            self.clone()
        } else {
            // Time-reversal does not matter here.
            self.convert_to_improper_kind(&INV)
        };

        c_self.total_proper_angle.abs()
    }

    /// Returns a copy of the current symmetry operation with the generating element
    /// converted to the requested improper kind (power-preserving), provided that
    /// it is an improper element.
    ///
    /// # Arguments
    ///
    /// * `improper_kind` - The improper kind to which `self` is to be converted. There is no need
    /// to make sure the time reversal specification in `improper_kind` matches that of the
    /// generating element of `self` as the conversion will take care of this.
    ///
    /// # Panics
    ///
    /// Panics if the converted symmetry operation cannot be constructed.
    #[must_use]
    pub fn convert_to_improper_kind(&self, improper_kind: &SymmetryElementKind) -> Self {
        let c_element = self
            .generating_element
            .convert_to_improper_kind(improper_kind, true);
        debug_assert_eq!(
            self.generating_element.is_su2_class_1(),
            c_element.is_su2_class_1()
        );
        Self::builder()
            .generating_element(c_element)
            .power(self.power)
            .positive_hemisphere(self.positive_hemisphere.clone())
            .build()
            .expect("Unable to construct a symmetry operation.")
    }

    /// Converts the current symmetry operation $`O`$ to an equivalent symmetry element $`E`$ such
    /// that $`O = E^1`$.
    ///
    /// The proper rotation axis of $`E`$ is the proper rotation pole (*not* the overall pole) of
    /// $`O`$, and the proper rotation angle of $`E`$ is the total proper rotation angle of $`O`$,
    /// either as an (order, power) integer tuple or an angle floating-point number.
    ///
    /// If $`O`$ is improper, then the improper generating element for $`E`$ is the same as that in
    /// the generating element of $`O`$.
    ///
    /// # Returns
    ///
    /// The equivalent symmetry element $`E`$.
    pub fn to_symmetry_element(&self) -> SymmetryElement {
        let kind = if self.is_proper() {
            let tr = self.is_antiunitary();
            if tr {
                TRROT
            } else {
                ROT
            }
        } else {
            self.generating_element.kind
        };
        let additional_superscript = if self.is_proper() {
            String::new()
        } else {
            self.generating_element.additional_superscript.clone()
        };
        let additional_subscript = if self.is_proper() {
            String::new()
        } else {
            self.generating_element.additional_subscript.clone()
        };
        let rotation_group = if self.is_su2_class_1() {
            SU2_1
        } else if self.is_su2() {
            SU2_0
        } else {
            SO3
        };
        let axis = if self.is_spatial_reflection() {
            self.positive_hemisphere
                .as_ref()
                .cloned()
                .unwrap_or_default()
                .get_positive_pole(
                    self.generating_element.raw_axis(),
                    self.generating_element.threshold,
                )
        } else {
            self.calc_proper_rotation_pole().coords
        };

        if let Some(total_proper_fraction) = self.total_proper_fraction {
            let proper_order = *total_proper_fraction
                .denom()
                .expect("Unable to extract the denominator of the total proper fraction.");
            let numer = *total_proper_fraction
                .numer()
                .expect("Unable to extract the numerator of the total proper fraction.");
            let proper_power =
                i32::try_from(numer).expect("Unable to convert the numerator to `i32`.");
            SymmetryElement::builder()
                .threshold(self.generating_element.threshold())
                .proper_order(ElementOrder::Int(proper_order))
                .proper_power(proper_power)
                .raw_axis(axis)
                .kind(kind)
                .rotation_group(rotation_group)
                .additional_superscript(additional_superscript)
                .additional_subscript(additional_subscript)
                .build()
                .unwrap()
        } else {
            let proper_angle = self.total_proper_angle;
            SymmetryElement::builder()
                .threshold(self.generating_element.threshold())
                .proper_order(ElementOrder::Inf)
                .proper_angle(proper_angle)
                .raw_axis(axis)
                .kind(kind)
                .rotation_group(rotation_group)
                .additional_superscript(additional_superscript)
                .additional_subscript(additional_subscript)
                .build()
                .unwrap()
        }
    }

    /// Generates the abbreviated symbol for this symmetry operation.
    #[must_use]
    pub fn get_abbreviated_symbol(&self) -> String {
        self.to_symmetry_element()
            .get_simplified_symbol_signed_power()
    }

    /// Returns the representation matrix for the spatial part of this symmetry operation.
    ///
    /// This representation matrix is in the basis of coordinate *functions* $`(y, z, x)`$.
    #[must_use]
    pub fn get_3d_spatial_matrix(&self) -> Array2<f64> {
        if self.is_proper() {
            if self.is_identity() || self.is_time_reversal() {
                Array2::<f64>::eye(3)
            } else {
                let angle = self.calc_pole_angle();
                let axis = self.calc_pole().coords;
                let mat = proper_rotation_matrix(angle, &axis, 1);

                // nalgebra matrix iter is column-major.
                Array2::<f64>::from_shape_vec(
                    (3, 3).f(),
                    mat.iter().copied().collect::<Vec<_>>(),
                )
                .unwrap_or_else(
                    |_| panic!(
                        "Unable to construct a three-dimensional rotation matrix for angle {angle} and axis {axis}."
                    )
                )
                .select(Axis(0), &[1, 2, 0])
                .select(Axis(1), &[1, 2, 0])
            }
        } else if self.is_spatial_inversion() {
            -Array2::<f64>::eye(3)
        } else {
            // Pole and pole angle are obtained in the inversion-centre convention.
            let angle = self.calc_pole_angle();
            let axis = self.calc_pole().coords;
            let mat = improper_rotation_matrix(angle, &axis, 1, &IMINV);

            // nalgebra matrix iter is column-major.
            Array2::<f64>::from_shape_vec(
                (3, 3).f(),
                mat.iter().copied().collect::<Vec<_>>(),
            )
            .unwrap_or_else(
                |_| panic!(
                    "Unable to construct a three-dimensional improper rotation matrix for angle {angle} and axis {axis}."
                )
            )
            .select(Axis(0), &[1, 2, 0])
            .select(Axis(1), &[1, 2, 0])
        }
    }

    /// Convert the proper rotation of the current operation to one in hopotopy class 0 of
    /// $`\mathsf{SU}(2)`$.
    ///
    /// # Returns
    ///
    /// A symmetry element in $`\mathsf{SU}(2)`$.
    pub fn to_su2_class_0(&self) -> Self {
        let q_identity = Self::from_quaternion(
            (-1.0, -Vector3::z()),
            true,
            self.generating_element.threshold(),
            1,
            false,
            true,
            None,
        );
        if self.is_su2() {
            if self.is_su2_class_1() {
                self * q_identity
            } else {
                self.clone()
            }
        } else {
            let mut op = self.clone();
            op.generating_element.rotation_group = SU2_0;
            if op.is_su2_class_1() {
                let mut q_op = op * q_identity;
                if !q_op.is_proper() {
                    q_op = q_op.convert_to_improper_kind(&SIG);
                }
                q_op
            } else {
                op
            }
        }
    }

    /// Sets the positive hemisphere governing this symmetry operation.
    ///
    /// # Arguments
    ///
    /// * `poshem` - An `Option` containing a custom positive hemisphere, if any.
    pub fn set_positive_hemisphere(&mut self, poshem: Option<&PositiveHemisphere>) {
        self.positive_hemisphere = poshem.cloned();
    }
}

// =====================
// Trait implementations
// =====================

impl FiniteOrder for SymmetryOperation {
    type Int = u32;

    /// Calculates the order of this symmetry operation.
    fn order(&self) -> Self::Int {
        let denom = *self
            .total_proper_fraction
            .expect("No total proper fractions found.")
            .denom()
            .expect("Unable to extract the denominator.");
        let spatial_order =
            if (self.is_proper() && !self.is_antiunitary()) || denom.rem_euclid(2) == 0 {
                denom
            } else {
                2 * denom
            };
        if self.is_su2() {
            2 * spatial_order
        } else {
            spatial_order
        }
    }
}

impl SpecialSymmetryTransformation for SymmetryOperation {
    // ============
    // Spatial part
    // ============

    /// Checks if the spatial part of the symmetry operation is proper.
    ///
    /// # Returns
    ///
    /// A boolean indicating if the spatial part of the symmetry operation is proper.
    fn is_proper(&self) -> bool {
        let au = self.generating_element.contains_antiunitary();
        self.generating_element.is_o3_proper(au) || self.power.rem_euclid(2) == 0
    }

    /// Checks if the spatial part of the symmetry operation is the spatial identity.
    ///
    /// # Returns
    ///
    /// A boolean indicating if the spatial part of the symmetry operation is the spatial identity.
    fn is_spatial_identity(&self) -> bool {
        self.is_proper()
            && match *self.generating_element.raw_proper_order() {
                ElementOrder::Int(_) => self
                    .total_proper_fraction
                    .expect("Total proper fraction not found for a finite-order operation.")
                    .is_zero(),
                ElementOrder::Inf => approx::relative_eq!(
                    self.total_proper_angle,
                    0.0,
                    max_relative = self.generating_element.threshold,
                    epsilon = self.generating_element.threshold
                ),
            }
    }

    /// Checks if the spatial part of the symmetry operation is a spatial binary rotation.
    ///
    /// # Returns
    ///
    /// A boolean indicating if the spatial part of the symmetry operation is a spatial binary
    /// rotation.
    fn is_spatial_binary_rotation(&self) -> bool {
        self.is_proper()
            && match *self.generating_element.raw_proper_order() {
                ElementOrder::Int(_) => {
                    self.total_proper_fraction
                        .expect("Total proper fraction not found for a finite-order operation.")
                        == F::new(1u32, 2u32)
                }
                ElementOrder::Inf => {
                    approx::relative_eq!(
                        self.total_proper_angle,
                        std::f64::consts::PI,
                        max_relative = self.generating_element.threshold,
                        epsilon = self.generating_element.threshold
                    )
                }
            }
    }

    /// Checks if the spatial part of the symmetry operation is the spatial inversion.
    ///
    /// # Returns
    ///
    /// A boolean indicating if the spatial part of the symmetry operation is the spatial inversion.
    fn is_spatial_inversion(&self) -> bool {
        !self.is_proper()
            && match self.generating_element.kind {
                SymmetryElementKind::ImproperMirrorPlane(_) => {
                    if let ElementOrder::Int(_) = *self.generating_element.raw_proper_order() {
                        self.total_proper_fraction
                            .expect("Total proper fraction not found for a finite-order operation.")
                            == F::new(1u32, 2u32)
                    } else {
                        approx::relative_eq!(
                            self.total_proper_angle,
                            std::f64::consts::PI,
                            max_relative = self.generating_element.threshold,
                            epsilon = self.generating_element.threshold
                        )
                    }
                }
                SymmetryElementKind::ImproperInversionCentre(_) => {
                    if let ElementOrder::Int(_) = *self.generating_element.raw_proper_order() {
                        self.total_proper_fraction
                            .expect("Total proper fraction not found for a finite-order operation.")
                            .is_zero()
                    } else {
                        approx::relative_eq!(
                            self.total_proper_angle,
                            0.0,
                            max_relative = self.generating_element.threshold,
                            epsilon = self.generating_element.threshold
                        )
                    }
                }
                _ => false,
            }
    }

    /// Checks if the spatial part of the symmetry operation is a spatial reflection.
    ///
    /// # Returns
    ///
    /// A boolean indicating if the spatial part of the symmetry operation is a spatial reflection.
    fn is_spatial_reflection(&self) -> bool {
        !self.is_proper()
            && match self.generating_element.kind {
                SymmetryElementKind::ImproperMirrorPlane(_) => {
                    if let ElementOrder::Int(_) = *self.generating_element.raw_proper_order() {
                        self.total_proper_fraction
                            .expect("Total proper fraction not found for a finite-order operation.")
                            .is_zero()
                    } else {
                        approx::relative_eq!(
                            self.total_proper_angle,
                            0.0,
                            max_relative = self.generating_element.threshold,
                            epsilon = self.generating_element.threshold
                        )
                    }
                }
                SymmetryElementKind::ImproperInversionCentre(_) => {
                    if let ElementOrder::Int(_) = self.generating_element.raw_proper_order() {
                        self.total_proper_fraction
                            .expect("Total proper fraction not found for a finite-order operation.")
                            == F::new(1u32, 2u32)
                    } else {
                        approx::relative_eq!(
                            self.total_proper_angle,
                            std::f64::consts::PI,
                            max_relative = self.generating_element.threshold,
                            epsilon = self.generating_element.threshold
                        )
                    }
                }
                _ => false,
            }
    }

    // ==================
    // Time-reversal part
    // ==================

    /// Checks if the symmetry operation is antiunitary or not.
    ///
    /// # Returns
    ///
    /// A boolean indicating if the symmetry oppperation is antiunitary.
    fn is_antiunitary(&self) -> bool {
        self.generating_element.contains_time_reversal() && self.power.rem_euclid(2) == 1
    }

    // ==================
    // Spin rotation part
    // ==================

    /// Checks if the proper rotation part of the symmetry operation is in $`\mathsf{SU}(2)`$.
    ///
    /// # Returns
    ///
    /// A boolean indicating if this symmetry operation contains an $`\mathsf{SU}(2)`$ proper
    /// rotation.
    fn is_su2(&self) -> bool {
        self.generating_element.rotation_group.is_su2()
    }

    /// Checks if the proper rotation part of the symmetry operation is in $`\mathsf{SU}(2)`$ and
    /// connected to the identity via a homotopy path of class 1.
    ///
    /// # Returns
    ///
    /// A boolean indicating if this symmetry operation contains an $`\mathsf{SU}(2)`$ proper
    /// rotation connected to the identity via a homotopy path of class 1.
    fn is_su2_class_1(&self) -> bool {
        if self.is_su2() {
            // The following is wrong, because `self.is_proper()` takes into account the power applied
            // to the spatial part, but not yet to the spin rotation part. Then, for example,
            // [QΣ·S3(+0.816, -0.408, +0.408)]^2 would become Σ'·[C3(+0.816, -0.408, +0.408)]^2 where
            // Σ' is the associated spin rotation of [C3(+0.816, -0.408, +0.408)]^2, which is not the
            // same as Σ^2.
            // let c_self = if self.is_proper() {
            //     self.clone()
            // } else {
            //     self.convert_to_improper_kind(&INV)
            // };
            //
            // The following is correct.
            let c_self = match self.generating_element.kind {
                SymmetryElementKind::Proper(_)
                | SymmetryElementKind::ImproperInversionCentre(_) => self.clone(),
                SymmetryElementKind::ImproperMirrorPlane(au) => {
                    self.convert_to_improper_kind(&SymmetryElementKind::ImproperInversionCentre(au))
                }
            };
            let generating_element_au = c_self.generating_element.contains_antiunitary();
            let spatial_proper_identity = c_self
                .generating_element
                .is_o3_identity(generating_element_au)
                || c_self
                    .generating_element
                    .is_o3_inversion_centre(generating_element_au);

            let inverse_from_time_reversal =
                if self.is_su2() && generating_element_au == Some(AntiunitaryKind::TimeReversal) {
                    self.power.rem_euclid(4) == 2 || self.power.rem_euclid(4) == 3
                } else {
                    false
                };

            let inverse_from_rotation_group = if spatial_proper_identity {
                // The proper part of the generating element is the identity. In this case, no
                // matter the value of proper power, the result is always the identity.
                false
            } else {
                let thresh = c_self.generating_element.threshold;
                let odd_jumps_from_angle = c_self
                    .generating_element
                    .proper_fraction
                    .map(|frac| {
                        let pow = c_self.power;
                        let total_unormalised_proper_fraction = frac * F::from(pow);
                        let (_, x) = geometry::normalise_rotation_fraction(
                            total_unormalised_proper_fraction,
                        );
                        x.rem_euclid(2) == 1
                    })
                    .unwrap_or_else(|| {
                        let total_unormalised_proper_angle = c_self
                            .generating_element
                            .proper_angle
                            .expect("Proper angle of generating element not found.")
                            * f64::from(c_self.power);
                        let (_, x) = geometry::normalise_rotation_angle(
                            total_unormalised_proper_angle,
                            thresh,
                        );
                        x.rem_euclid(2) == 1
                    });
                let single_jump_from_c2 = (c_self.is_spatial_binary_rotation()
                    || c_self.is_spatial_reflection())
                    && !self
                        .positive_hemisphere
                        .as_ref()
                        .cloned()
                        .unwrap_or_default()
                        .check_positive_pole(c_self.generating_element.raw_axis(), thresh);
                odd_jumps_from_angle != single_jump_from_c2
            };

            let intrinsic_inverse = c_self.generating_element.rotation_group().is_su2_class_1()
                && c_self.power.rem_euclid(2) == 1;

            let inverse_count = [
                inverse_from_time_reversal,
                inverse_from_rotation_group,
                intrinsic_inverse,
            ]
            .into_iter()
            .filter(|&inverse| inverse)
            .count();

            inverse_count.rem_euclid(2) == 1
        } else {
            false
        }
    }
}

impl fmt::Debug for SymmetryOperation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.power == 1 {
            write!(f, "{:?}", self.generating_element)
        } else if self.power >= 0 {
            write!(f, "[{:?}]^{}", self.generating_element, self.power)
        } else {
            write!(f, "[{:?}]^({})", self.generating_element, self.power)
        }
    }
}

impl fmt::Display for SymmetryOperation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.power == 1 {
            write!(f, "{}", self.generating_element)
        } else if self.power >= 0 {
            write!(f, "[{}]^{}", self.generating_element, self.power)
        } else {
            write!(f, "[{}]^({})", self.generating_element, self.power)
        }
    }
}

impl PartialEq for SymmetryOperation {
    fn eq(&self, other: &Self) -> bool {
        if (*self.generating_element.raw_proper_order() == ElementOrder::Inf)
            != (*other.generating_element.raw_proper_order() == ElementOrder::Inf)
        {
            // We disable comparisons between operations with infinite-order and
            // finite-order generating elements, because they cannot be made to
            // have the same hashes without losing the fidelity of exact-fraction
            // representations for operations with finite-order generating elements.
            return false;
        }

        // =================
        // Group-theoretical
        // =================

        if self.is_su2() != other.is_su2() {
            return false;
        }

        if self.is_su2_class_1() != other.is_su2_class_1() {
            return false;
        }

        // ==========================
        // Special general operations
        // ==========================

        if self.is_proper() != other.is_proper() {
            return false;
        }

        if self.is_antiunitary() != other.is_antiunitary() {
            return false;
        }

        // ===========================
        // Special specific operations
        // ===========================

        // At this stage, `self` and `other` must have the same spatial parity, unitarity, and
        // SO3/SU2 properties.
        if self.is_spatial_identity() && other.is_spatial_identity() {
            assert_eq!(misc::calculate_hash(self), misc::calculate_hash(other));
            return true;
        }

        // ======
        // Others
        // ======

        let thresh =
            (self.generating_element.threshold * other.generating_element.threshold).sqrt();

        let result = if (self.is_spatial_binary_rotation() && other.is_spatial_binary_rotation())
            || (self.is_spatial_reflection() && other.is_spatial_reflection())
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
                // Time-reversal does not matter here.
                self.convert_to_improper_kind(&INV)
            };
            let c_other = if other.is_proper() {
                other.clone()
            } else {
                // Time-reversal does not matter here.
                other.convert_to_improper_kind(&INV)
            };

            let angle_comparison = if let (Some(s_frac), Some(o_frac)) =
                (c_self.total_proper_fraction, c_other.total_proper_fraction)
            {
                s_frac.abs() == o_frac.abs()
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
                "`{self}` and `{other}` have unequal hashes.",
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
        // ==========================
        // Special general operations
        // ==========================
        c_self.is_proper().hash(state);
        c_self.is_antiunitary().hash(state);
        c_self.is_su2().hash(state);
        c_self.is_su2_class_1().hash(state);

        // ===========================
        // Special specific operations
        // ===========================
        if c_self.is_spatial_identity() {
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

            if !c_self.is_spatial_binary_rotation() && !c_self.is_spatial_reflection() {
                if let Some(frac) = c_self.total_proper_fraction {
                    // self.total_proper_fraction lies in (-1/2, 0) ∪ (0, 1/2).
                    // 0 and 1/2 are excluded because this is not an identity,
                    // inversion, binary rotation, or reflection.
                    frac.abs().hash(state);
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
        assert_eq!(
            self.is_su2(),
            rhs.is_su2(),
            "`self` and `rhs` must both have or not have associated spin rotations."
        );
        assert_eq!(self.positive_hemisphere, rhs.positive_hemisphere);
        let su2 = self.is_su2();
        let (q1_s, q1_v) = self.calc_quaternion();
        let (q2_s, q2_v) = rhs.calc_quaternion();

        let q3_s = q1_s * q2_s - q1_v.dot(&q2_v);
        let q3_v = q1_s * q2_v + q2_s * q1_v + q1_v.cross(&q2_v);

        // Is the resulting operation proper?
        let proper = self.is_proper() == rhs.is_proper();

        // Does the resulting operation contain a time reversal?
        let tr = self.is_antiunitary() != rhs.is_antiunitary();

        // Does the resulting operation pick up a quaternion sign change due to θ^2?
        let tr2 = self.is_antiunitary() && rhs.is_antiunitary();

        let thresh = (self.generating_element.threshold * rhs.generating_element.threshold).sqrt();
        let max_trial_power = u32::MAX;

        let q3 = if su2 {
            if tr2 {
                (-q3_s, -q3_v)
            } else {
                (q3_s, q3_v)
            }
        } else if q3_s >= 0.0 {
            (q3_s, q3_v)
        } else {
            (-q3_s, -q3_v)
        };

        SymmetryOperation::from_quaternion(
            q3,
            proper,
            thresh,
            max_trial_power,
            tr,
            su2,
            self.positive_hemisphere.clone(),
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
            .positive_hemisphere(self.positive_hemisphere.clone())
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

impl Inv for &SymmetryOperation {
    type Output = SymmetryOperation;

    fn inv(self) -> Self::Output {
        SymmetryOperation::builder()
            .generating_element(self.generating_element.clone())
            .power(-self.power)
            .positive_hemisphere(self.positive_hemisphere.clone())
            .build()
            .expect("Unable to construct an inverse symmetry operation.")
    }
}

impl Inv for SymmetryOperation {
    type Output = SymmetryOperation;

    fn inv(self) -> Self::Output {
        (&self).inv()
    }
}

impl<M> IntoPermutation<M> for SymmetryOperation
where
    M: Transform + PermutableCollection<Rank = usize>,
{
    fn act_permute(&self, rhs: &M) -> Option<Permutation<usize>> {
        let angle = self.calc_pole_angle();
        let axis = self.calc_pole().coords;
        let mut t_mol = if self.is_proper() {
            rhs.rotate(angle, &axis)
        } else {
            rhs.improper_rotate(angle, &axis, &IMINV)
        };
        if self.is_antiunitary() {
            t_mol.reverse_time_mut();
        }
        rhs.get_perm_of(&t_mol)
    }
}

// =================
// Utility functions
// =================
/// Sorts symmetry operations in-place based on:
///
/// * whether they are unitary or antiunitary
/// * whether they are proper or improper
/// * whether they are the identity or inversion
/// * whether they are a spatial binary rotation or spatial reflection
/// * their orders
/// * their powers
/// * their closeness to Cartesian axes
/// * the axes of closest inclination
/// * whether they are of homotopy class 1 in $`\mathsf{SU}'(2)`$.
///
/// # Arguments
///
/// * `operations` - A mutable reference to a vector of symmetry operations.
pub(crate) fn sort_operations(operations: &mut [SymmetryOperation]) {
    operations.sort_by_key(|op| {
        let (axis_closeness, closest_axis) = op.generating_element.closeness_to_cartesian_axes();
        let c_op = if op.is_proper()
            || op.generating_element.kind == SIG
            || op.generating_element.kind == TRSIG
        {
            op.clone()
        } else if op.is_antiunitary() {
            op.convert_to_improper_kind(&TRSIG)
        } else {
            op.convert_to_improper_kind(&SIG)
        };

        let total_proper_fraction = c_op
            .total_proper_fraction
            .expect("No total proper fractions found.");
        let denom = i64::try_from(
            *total_proper_fraction
                .denom()
                .expect("The denominator of the total proper fraction cannot be extracted."),
        )
        .unwrap_or_else(|_| {
            panic!("Unable to convert the denominator of `{total_proper_fraction:?}` to `i64`.")
        });
        let numer = i64::try_from(
            *total_proper_fraction
                .numer()
                .expect("The numerator of the total proper fraction cannot be extracted."),
        )
        .unwrap_or_else(|_| {
            panic!("Unable to convert the numerator of `{total_proper_fraction:?}` to `i64`.")
        });

        let negative_rotation = !c_op
            .positive_hemisphere
            .as_ref()
            .cloned()
            .unwrap_or_default()
            .check_positive_pole(
                &c_op.calc_proper_rotation_pole().coords,
                c_op.generating_element.threshold(),
            );

        (
            c_op.is_antiunitary(),
            !c_op.is_proper(),
            !(c_op.is_spatial_identity() || c_op.is_spatial_inversion()),
            c_op.is_spatial_binary_rotation() || c_op.is_spatial_reflection(),
            -denom,
            negative_rotation,
            if negative_rotation { -numer } else { numer },
            OrderedFloat(axis_closeness),
            closest_axis,
            c_op.is_su2_class_1(),
        )
    });
}
