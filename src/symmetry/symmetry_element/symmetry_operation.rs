use std::fmt;
use std::hash::{Hash, Hasher};
use std::ops::Mul;

use approx;
use derive_builder::Builder;
use fraction;
use nalgebra::{Point3, Vector3};
use ndarray::{Array2, Axis, ShapeBuilder};
use num::ToPrimitive;
use num_traits::{Inv, One, Pow, Zero};

use crate::aux::geometry::{
    self, improper_rotation_matrix, proper_rotation_matrix, Transform, IMINV,
};
use crate::aux::misc::{self, HashableFloat};
use crate::group::FiniteOrder;
use crate::permutation::{IntoPermutation, PermutableCollection, Permutation};
use crate::symmetry::symmetry_element::{RotationGroup, SymmetryElement, SymmetryElementKind, INV};
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

    /// Checks if the symmetry operation contains an active associated spin rotation (normal or
    /// inverse).
    ///
    /// # Returns
    ///
    /// A flag indicating if this symmetry operation contains an active associated spin rotation.
    fn is_su2(&self) -> bool;

    /// Checks if the symmetry operation contains an active and inverse associated spin rotation.
    ///
    /// # Returns
    ///
    /// A flag indicating if this symmetry operation contains an active and inverse associated
    /// spin rotation.
    fn is_su2_class_1(&self) -> bool;

    // ============
    // Spatial part
    // ============

    /// Checks if the symmetry operation is proper or not.
    ///
    /// # Returns
    ///
    /// A flag indicating if the symmetry operation is proper.
    fn is_proper(&self) -> bool;

    fn is_spatial_identity(&self) -> bool;

    fn is_spatial_binary_rotation(&self) -> bool;

    fn is_spatial_inversion(&self) -> bool;

    fn is_spatial_reflection(&self) -> bool;

    // ==================
    // Time-reversal part
    // ==================

    /// Checks if the symmetry operation is antiunitary or not.
    ///
    /// # Returns
    ///
    /// A flag indicating if the symmetry oppperation is antiunitary.
    fn is_antiunitary(&self) -> bool;

    // ==========================
    // Overall - provided methods
    // ==========================

    /// Checks if the symmetry operation is the identity in $`\mathsf{O}(3)`$ or $`\mathsf{SU}(2)`$.
    ///
    /// # Returns
    ///
    /// A flag indicating if this symmetry operation is the identity.
    fn is_identity(&self) -> bool {
        self.is_spatial_identity() && !self.is_antiunitary() && !self.is_su2_class_1()
    }

    /// Checks if the symmetry operation is a pure time-reversal.
    ///
    /// # Returns
    ///
    /// A flag indicating if this symmetry operation is a pure time-reversal.
    fn is_time_reversal(&self) -> bool {
        self.is_spatial_identity() && self.is_antiunitary() && !self.is_su2_class_1()
    }

    /// Checks if the symmetry operation is an inversion in $`\mathsf{O}(3)`$.
    ///
    /// # Returns
    ///
    /// A flag indicating if this symmetry operation is an inversion.
    fn is_inversion(&self) -> bool {
        self.is_spatial_inversion() && !self.is_antiunitary() && !self.is_su2()
    }

    // /// Checks if the symmetry operation is an inversion accompanied by a time reversal.
    // ///
    // /// # Returns
    // ///
    // /// A flag indicating if this symmetry operation is an inversion accompanied by a time reversal.
    // fn is_tr_inversion(&self) -> bool {
    //     self.is_spatial_inversion() && self.is_antiunitary() && !self.is_su2_class_1()
    // }

    // /// Checks if the symmetry operation is a binary rotation.
    // ///
    // /// # Returns
    // ///
    // /// A flag indicating if this symmetry operation is a binary rotation.
    // fn is_binary_rotation(&self) -> bool {
    //     self.is_spatial_binary_rotation() && !self.is_antiunitary() && !self.is_su2()
    // }

    // /// Checks if the symmetry operation is a binary rotation accompanied by a time reversal.
    // ///
    // /// # Returns
    // ///
    // /// A flag indicating if this symmetry operation is a binary rotation.
    // fn is_tr_binary_rotation(&self) -> bool {
    //     self.is_spatial_binary_rotation() && self.is_antiunitary() && !self.is_su2()
    // }

    /// Checks if the symmetry operation is a reflection.
    ///
    /// # Returns
    ///
    /// A flag indicating if this symmetry operation is a reflection.
    fn is_reflection(&self) -> bool {
        self.is_spatial_reflection() && !self.is_antiunitary() && !self.is_su2()
    }

    ///// Checks if the symmetry operation is a reflection.
    /////
    ///// # Returns
    /////
    ///// A flag indicating if this symmetry operation is a reflection accompanied by a time reversal.
    //fn is_tr_reflection(&self) -> bool {
    //    self.is_spatial_reflection() && self.is_antiunitary() && !self.is_su2()
    //}
}

// ======================================
// Struct definitions and implementations
// ======================================

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

    /// The total proper rotation angle associated with this operation (after
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
                let mut normalised_frac = frac * F::from(pow);
                let frac_1_2 = F::new(1u32, 2u32);
                while normalised_frac > frac_1_2 {
                    normalised_frac -= F::one();
                }
                while normalised_frac <= -frac_1_2 {
                    normalised_frac += F::one();
                }
                Some(normalised_frac)
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
    /// * `sr` - A flag indicating if the resulting symmetry operation should be accompanied by a
    /// spin rotation.
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
    ) -> Self {
        let (scalar_part, vector_part) = qtn;
        let kind = if proper {
            SymmetryElementKind::Proper(tr)
        } else {
            SymmetryElementKind::ImproperInversionCentre(tr)
        };
        let element = if su2 {
            log::debug!(
                "Constructing a symmetry element of kind `{kind}` with the proper part in SU(3)..."
            );
            assert!(
                -1.0 - thresh <= scalar_part && scalar_part <= 1.0 + thresh,
                "The scalar part of the quaternion must be in the interval [-1, +1]."
            );
            let (axis, order, power, normal) = if approx::relative_eq!(
                scalar_part,
                1.0,
                epsilon = thresh,
                max_relative = thresh
            ) {
                // Zero-degree rotation, i.e. identity or inversion, class 0
                (Vector3::new(0.0, 0.0, 1.0), 1u32, 1u32, true)
            } else if approx::relative_eq!(
                scalar_part,
                -1.0,
                epsilon = thresh,
                max_relative = thresh
            ) {
                // 360-degree rotation, i.e. identity or inversion, class 1
                (Vector3::new(0.0, 0.0, 1.0), 1u32, 1u32, false)
            } else if approx::relative_eq!(
                scalar_part,
                0.0,
                epsilon = thresh,
                max_relative = thresh
            ) {
                // 180-degree rotation, i.e. binary rotation or reflection. Whether the resultant
                // operation is in class 0 or class 1 depends on whether the vector part is in the
                // positive hemisphere or negative hemisphere.
                let positive_axis = geometry::get_positive_pole(&vector_part, thresh);
                (
                    positive_axis,
                    2u32,
                    1u32,
                    geometry::check_positive_pole(&vector_part, thresh),
                )
            } else {
                // scalar_part != 0, 1, or -1
                // scalar_part = cos(ϕ/2) = λ
                // If scalar_part > 0, ϕ/2 = argcos(|λ|)
                // If scalar_part < 0, ϕ/2 = argcos(|λ|) + π
                // Once ϕ has been found, the vector_part can be used to work out the axis.
                // let half_spatial_angle = scalar_part.abs().acos();
                // let spatial_positive_normalised_angle = 2.0 * half_spatial_angle;
                // let spatial_axis = if scalar_part > 0.0 {
                //     vector_part / half_spatial_angle.sin()
                // } else {
                //     vector_part / (half_spatial_angle + std::f64::consts::PI).sin()
                // };
                // let spatial_proper_fraction = geometry::get_proper_fraction(
                //     spatial_positive_normalised_angle,
                //     thresh,
                //     max_trial_power,
                // )
                // .unwrap_or_else(|| {
                //     panic!("No proper fraction could be found for angle `{spatial_positive_normalised_angle}`.")
                // });
                let (standardised_scalar_part, standardised_vector_part, normal) =
                    if scalar_part > 0.0 {
                        (scalar_part, vector_part, true)
                    } else {
                        (-scalar_part, -vector_part, false)
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
                    normal,
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
                .rotationgroup(RotationGroup::SU2(normal))
                .build()
                .unwrap_or_else(|_|
                    panic!("Unable to construct a symmetry element of kind `{kind}` with the proper part in SU(2).")
                )
        } else {
            log::debug!(
                "Constructing a symmetry element of kind `{kind}` with the proper part in SO(3)..."
            );
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
                (Vector3::new(0.0, 0.0, 1.0), 1u32, 1u32)
            } else {
                let half_proper_angle = scalar_part.acos(); // acos returns values in [0, π]
                let proper_angle = 2.0 * half_proper_angle;
                let axis = vector_part / half_proper_angle.sin();
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
                .rotationgroup(RotationGroup::SO3)
                .build()
                .expect(
                    "Unable to construct a symmetry element without an associated spin rotation.",
                )
        };

        SymmetryOperation::builder()
            .generating_element(element)
            .power(1)
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
        // let c_self = match self.generating_element.kind {
        //     SymmetryElementKind::Proper(_) | SymmetryElementKind::ImproperInversionCentre(_) => {
        //         self.clone()
        //     }
        //     SymmetryElementKind::ImproperMirrorPlane(_) => self.convert_to_improper_kind(&INV),
        // };
        let c_self = if self.is_proper() {
            self.clone()
        } else {
            // Time-reversal does not matter here.
            self.convert_to_improper_kind(&INV)
        };
        assert_eq!(self.is_su2_class_1(), c_self.is_su2_class_1());

        // We only need the absolute value of the angle. Its sign information is
        // encoded in the pole. `abs_angle` thus lies in [0, π], and so
        //  cos(abs_angle/2) >= 0 and sin(abs_angle/2) >= 0.
        // The scalar part is guaranteed to be in [0, 1].
        // For binary rotations, the scalar part is zero, but the definition of pole ensures that
        // the vector part still lies in the positive hemisphere.
        let abs_angle = c_self.total_proper_angle.abs();
        let scalar_part = (0.5 * abs_angle).cos();
        let vector_part = (0.5 * abs_angle).sin() * c_self.calc_pole().coords;
        assert!(
            -self.generating_element.threshold <= scalar_part
                && scalar_part <= 1.0 + self.generating_element.threshold
        );
        if approx::relative_eq!(
            scalar_part,
            0.0,
            max_relative = c_self.generating_element.threshold,
            epsilon = c_self.generating_element.threshold
        ) {
            assert!(geometry::check_positive_pole(
                &vector_part,
                c_self.generating_element.threshold
            ));
        }

        if self.is_su2_class_1() {
            // println!(
            //     "Calc Q for {self}: {abs_angle} {} => {}, {}",
            //     c_self.calc_pole().coords,
            //     -scalar_part,
            //     -vector_part
            // );
            (-scalar_part, -vector_part)
        } else {
            // println!(
            //     "Calc Q for {self}: {abs_angle} => {}, {}",
            //     scalar_part, vector_part
            // );
            (scalar_part, vector_part)
        }
    }

    /// Finds the pole associated with this operation.
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
        match op.generating_element.proper_order {
            ElementOrder::Int(_) => {
                let frac_1_2 = F::new(1u32, 2u32);
                let total_proper_fraction = op
                    .total_proper_fraction
                    .expect("No total proper fractions found.");
                if total_proper_fraction == frac_1_2 {
                    // Binary rotations or reflections
                    Point3::from(geometry::get_positive_pole(
                        &op.generating_element.raw_axis,
                        op.generating_element.threshold,
                    ))
                } else if total_proper_fraction > F::zero() {
                    // Positive rotation angles
                    Point3::from(op.generating_element.raw_axis)
                } else if total_proper_fraction < F::zero() {
                    // Negative rotation angles
                    Point3::from(-op.generating_element.raw_axis)
                } else {
                    // Identity or inversion
                    assert!(total_proper_fraction.is_zero());
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
                        &op.generating_element.raw_axis,
                        op.generating_element.threshold,
                    ))
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
                    Point3::origin()
                }
            }
        }
    }

    /// Finds the pole angle associated with this operation.
    ///
    /// This is the angle that, together with the pole, uniquely determines the proper part of this
    /// operation. This angle lies in the interval $`[0, \pi]`$.
    ///
    /// For improper operations, the inversion-centre convention is used to define
    /// the pole angle. This allows a proper rotation and its improper partner to have the
    /// same pole angle, thus facilitating the consistent specification of poles for the
    /// identity / inversion and binary rotations / reflections.
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
        let c_self = match self.generating_element.kind {
            SymmetryElementKind::Proper(_) | SymmetryElementKind::ImproperInversionCentre(_) => {
                self.clone()
            }
            SymmetryElementKind::ImproperMirrorPlane(_) => self.convert_to_improper_kind(&INV),
        };

        c_self.total_proper_angle.abs()
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
        assert_eq!(
            self.generating_element.is_su2_class_1(),
            c_element.is_su2_class_1()
        );
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
            self.generating_element.get_simplified_symbol()
        } else {
            format!(
                "[{}]^{}",
                self.generating_element.get_simplified_symbol(),
                self.power
            )
        }
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
        } else {
            if self.is_spatial_inversion() {
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
    }

    /// Finds the pole double-angle associated with this operation.
    ///
    /// This is the angle that, together with the pole, uniquely determines the proper part of this
    /// operation in double groups. This angle lies in the interval $`[0, 2\pi]`$.
    ///
    /// For improper operations, the inversion-centre convention is used to define
    /// the pole angle. This allows a proper rotation and its improper partner to have the
    /// same pole angle, thus facilitating the consistent specification of poles for the
    /// identity / inversion and binary rotations / reflections.
    ///
    /// # Returns
    ///
    /// The pole angle associated with this operation.
    ///
    /// # Panics
    ///
    /// Panics when no total proper fractions could be found for this operation.
    #[must_use]
    pub fn calc_pole_double_angle(&self) -> f64 {
        let c_self = match self.generating_element.kind {
            SymmetryElementKind::Proper(_) | SymmetryElementKind::ImproperInversionCentre(_) => {
                self.clone()
            }
            SymmetryElementKind::ImproperMirrorPlane(_) => self.convert_to_improper_kind(&INV),
        };

        geometry::normalise_rotation_double_angle(
            c_self
                .generating_element
                .proper_angle
                .expect("Proper angle has not been set.")
                * (f64::from(self.power)),
            c_self.generating_element.threshold,
        )
        .abs()
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

    /// Checks if the spatial part of the symmetry operation is proper or not.
    ///
    /// # Returns
    ///
    /// A flag indicating if the spatial part of the symmetry operation is proper.
    fn is_proper(&self) -> bool {
        self.generating_element.is_o3_proper(true)
            || self.generating_element.is_o3_proper(false)
            || (self.power.rem_euclid(2) == 0)
    }

    fn is_spatial_identity(&self) -> bool {
        self.is_proper()
            && match self.generating_element.proper_order {
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

    fn is_spatial_binary_rotation(&self) -> bool {
        self.is_proper()
            && match self.generating_element.proper_order {
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

    fn is_spatial_inversion(&self) -> bool {
        !self.is_proper()
            && match self.generating_element.kind {
                SymmetryElementKind::ImproperMirrorPlane(_) => {
                    if let ElementOrder::Int(_) = self.generating_element.proper_order {
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
                    if let ElementOrder::Int(_) = self.generating_element.proper_order {
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

    fn is_spatial_reflection(&self) -> bool {
        !self.is_proper()
            && match self.generating_element.kind {
                SymmetryElementKind::ImproperMirrorPlane(_) => {
                    if let ElementOrder::Int(_) = self.generating_element.proper_order {
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
                    if let ElementOrder::Int(_) = self.generating_element.proper_order {
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
    /// A flag indicating if the symmetry oppperation is antiunitary.
    fn is_antiunitary(&self) -> bool {
        self.generating_element.contains_time_reversal() && self.power.rem_euclid(2) == 1
    }

    // ==================
    // Spin rotation part
    // ==================

    fn is_su2(&self) -> bool {
        self.generating_element.rotationgroup.is_su2()
    }

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
            let c_self = match self.generating_element.kind {
                SymmetryElementKind::Proper(_) | SymmetryElementKind::ImproperInversionCentre(_) => {
                    self.clone()
                }
                SymmetryElementKind::ImproperMirrorPlane(tr) => {
                    self.convert_to_improper_kind(&SymmetryElementKind::ImproperInversionCentre(tr))
                }
            };
            let generating_element_tr = c_self.generating_element.contains_time_reversal();
            let spatial_proper_identity = c_self
                .generating_element
                .is_o3_identity(generating_element_tr)
                || c_self
                    .generating_element
                    .is_o3_inversion_centre(generating_element_tr);
            let inverse_from_rotationgroup = if spatial_proper_identity {
                // The proper part of the generating element is the identity. In this case, no
                // matter the value of proper power, the result is always the identity.
                false
            } else {
                c_self
                .generating_element
                .proper_fraction
                .map(|frac| {
                    // The generating element has a proper fraction, k/n, which becomes kp/n when
                    // raised to the proper power p.
                    //
                    // If kp/n > 1/2, we seek a positive integer x such that
                    //  -1/2 < kp/n - x <= 1/2.
                    // It turns out that x ∈ [kp/n - 1/2, kp/n + 1/2).
                    //
                    // If kp/n <= -1/2, we seek a positive integer x such that
                    //  -1/2 < kp/n + x <= 1/2.
                    // It turns out that x ∈ (-kp/n - 1/2, -kp/n + 1/2].
                    //
                    // If the proper rotation corresponding to kp/n is reached from the identity
                    // via a continuous path in the parametric ball, x gives the number of times
                    // this path goes through a podal-antipodal jump, and thus whether x is even
                    // corresponds to whether this homotopy path is of class 0.
                    //
                    // See S.L. Altmann, Rotations, Quaternions, and Double Groups (Dover
                    // Publications, Inc., New York, 2005) for further information.
                    let pow = c_self.power;
                    let total_proper_fraction = frac * F::from(pow);
                    let frac_1_2 = F::new(1u32, 2u32);
                    let x = if total_proper_fraction > frac_1_2 {
                        let integer_part = total_proper_fraction
                                .trunc()
                                .to_u32()
                                .unwrap_or_else(|| panic!("Unable to convert the integer part of `{total_proper_fraction}` to `u32`."));
                        if total_proper_fraction.fract() <= frac_1_2 {
                            integer_part
                        } else {
                            integer_part + 1
                        }
                    } else if total_proper_fraction <= -frac_1_2 {
                        let integer_part = (-total_proper_fraction)
                                .trunc()
                                .to_u32()
                                .unwrap_or_else(|| panic!("Unable to convert the integer part of `{total_proper_fraction}` to `u32`."));
                        if (-total_proper_fraction).fract() < frac_1_2 {
                            integer_part
                        } else {
                            integer_part + 1
                        }
                    } else {
                        0
                    };
                    x.rem_euclid(2) == 1
                })
                .unwrap_or_else(|| {
                    let total_proper_angle = c_self
                        .generating_element
                        .proper_angle
                        .expect("Proper angle of generating element not found.")
                        * f64::from(c_self.power);
                    let thresh = c_self.generating_element.threshold;
                    let total_proper_fraction = total_proper_angle / 2.0 * std::f64::consts::PI;
                    let frac_1_2 = 1.0 / 2.0;
                    let x = if total_proper_fraction > frac_1_2 + thresh {
                        let integer_part = total_proper_fraction
                                .trunc()
                                .to_u32()
                                .unwrap_or_else(|| panic!("Unable to convert the integer part of `{total_proper_fraction}` to `u32`."));
                        if total_proper_fraction.fract() <= frac_1_2 + thresh {
                            integer_part
                        } else {
                            integer_part + 1
                        }
                    } else if total_proper_fraction <= -frac_1_2 + thresh {
                        let integer_part = (-total_proper_fraction)
                                .trunc()
                                .to_u32()
                                .unwrap_or_else(|| panic!("Unable to convert the integer part of `{total_proper_fraction}` to `u32`."));
                        if (-total_proper_fraction).fract() < frac_1_2 - thresh {
                            integer_part
                        } else {
                            integer_part + 1
                        }
                    } else {
                        0
                    };
                    x.rem_euclid(2) == 1
                })
            };
            let intrinsic_inverse = c_self.generating_element.rotationgroup.is_su2_class_1()
                && c_self.power.rem_euclid(2) == 1;
            // println!("Inv from rot : {inverse_from_rotationgroup}");
            // println!("Intrinsic inv: {intrinsic_inverse}");
            inverse_from_rotationgroup != intrinsic_inverse
        } else {
            false
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
        let su2 = self.is_su2();
        let (q1_s, q1_v) = self.calc_quaternion();
        let (q2_s, q2_v) = rhs.calc_quaternion();

        let q3_s = q1_s * q2_s - q1_v.dot(&q2_v);
        let q3_v = q1_s * q2_v + q2_s * q1_v + q1_v.cross(&q2_v);

        let q3 = if su2 || q3_s >= 0.0 {
            (q3_s, q3_v)
        } else {
            (-q3_s, -q3_v)
        };

        // println!("Q1: {q1_s}, {q1_v:?}");
        // println!("Q2: {q2_s}, {q2_v:?}");
        // println!("Q3: {q3:?}");

        let proper = self.is_proper() == rhs.is_proper();
        let tr = self.is_antiunitary() != rhs.is_antiunitary();
        let thresh = (self.generating_element.threshold * rhs.generating_element.threshold).sqrt();
        let max_trial_power = u32::MAX;
        SymmetryOperation::from_quaternion(q3, proper, thresh, max_trial_power, tr, su2)
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

impl Inv for &SymmetryOperation {
    type Output = SymmetryOperation;

    fn inv(self) -> Self::Output {
        SymmetryOperation::builder()
            .generating_element(self.generating_element.clone())
            .power(-self.power)
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
