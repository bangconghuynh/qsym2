use crate::aux::geometry;
use crate::symmetry::symmetry_element::{SymmetryElement, SymmetryElementKind};
use crate::symmetry::symmetry_element_order::ElementOrder;
use approx;
use derive_builder::Builder;
use fraction;

type F = fraction::Fraction;

#[cfg(test)]
#[path = "symmetry_operation_tests.rs"]
mod symmetry_operation_tests;

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
    #[builder(setter(skip), default = "self.calc_total_proper_angle()")]
    total_proper_angle: f64,
}

impl SymmetryOperationBuilder {
    fn calc_total_proper_angle(&self) -> f64 {
        geometry::normalise_rotation_angle(
            self.generating_element
                .as_ref()
                .unwrap()
                .proper_angle
                .unwrap()
                * (self.power.unwrap() as f64),
            self.generating_element.as_ref().unwrap().threshold,
        )
    }
}

impl SymmetryOperation {
    /// Returns a builder to construct a new symmetry operation.
    ///
    /// # Returns
    ///
    /// A builder to construct a new symmetry operation.
    pub fn builder() -> SymmetryOperationBuilder {
        SymmetryOperationBuilder::default()
    }

    /// Checks if the symmetry operation is proper or not.
    ///
    /// # Returns
    ///
    /// A flag indicating if the symmetry operation is proper.
    pub fn is_proper(&self) -> bool {
        self.generating_element.is_proper() || (self.power % 2 == 0)
    }

    /// Checks if the symmetry operation is the identity.
    ///
    /// # Returns
    ///
    /// A flag indicating if this symmetry operation is the identity.
    pub fn is_identity(&self) -> bool {
        self.is_proper()
            && match self.generating_element.proper_order {
                ElementOrder::Int(io) => {
                    let pos_pow = (self.power % (io as i32)) as u64;
                    (self.generating_element.proper_fraction.unwrap() * F::from(pos_pow)).fract()
                        == F::from(0u64)
                }
                ElementOrder::Inf => {
                    approx::relative_eq!(
                        geometry::normalise_rotation_angle(
                            self.generating_element.proper_angle.unwrap() * (self.power as f64),
                            self.generating_element.threshold
                        ) % (2.0 * std::f64::consts::PI),
                        0.0,
                        max_relative = self.generating_element.threshold,
                        epsilon = self.generating_element.threshold
                    )
                }
            }
    }

    /// Checks if the symmetry operation is an inversion.
    ///
    /// # Returns
    ///
    /// A flag indicating if this symmetry operation is an inversion.
    pub fn is_inversion(&self) -> bool {
        !self.is_proper()
            && match self.generating_element.kind {
                SymmetryElementKind::ImproperMirrorPlane => {
                    if let ElementOrder::Int(io) = self.generating_element.proper_order {
                        let pos_pow = (self.power % (io as i32)) as u64;
                        (self.generating_element.proper_fraction.unwrap() * F::from(pos_pow))
                            .fract()
                            == F::new(1u64, 2u64)
                    } else {
                        approx::relative_eq!(
                            geometry::normalise_rotation_angle(
                                self.generating_element.proper_angle.unwrap() * (self.power as f64),
                                self.generating_element.threshold
                            ) % (2.0 * std::f64::consts::PI),
                            std::f64::consts::PI,
                            max_relative = self.generating_element.threshold,
                            epsilon = self.generating_element.threshold
                        )
                    }
                }
                SymmetryElementKind::ImproperInversionCentre => {
                    if let ElementOrder::Int(io) = self.generating_element.proper_order {
                        let pos_pow = (self.power % (io as i32)) as u64;
                        (self.generating_element.proper_fraction.unwrap() * F::from(pos_pow))
                            .fract()
                            == F::from(0u64)
                    } else {
                        approx::relative_eq!(
                            geometry::normalise_rotation_angle(
                                self.generating_element.proper_angle.unwrap() * (self.power as f64),
                                self.generating_element.threshold
                            ) % (2.0 * std::f64::consts::PI),
                            0.0,
                            max_relative = self.generating_element.threshold,
                            epsilon = self.generating_element.threshold
                        )
                    }
                }
                SymmetryElementKind::Proper => false,
            }
    }

    /// Checks if the symmetry operation is a binary rotation.
    ///
    /// # Returns
    ///
    /// A flag indicating if this symmetry operation is a binary rotation.
    pub fn is_binary_rotation(&self) -> bool {
        self.is_proper()
            && match self.generating_element.proper_order {
                ElementOrder::Int(io) => {
                    let pos_pow = (self.power % (io as i32)) as u64;
                    (self.generating_element.proper_fraction.unwrap() * F::from(pos_pow)).fract()
                        == F::new(1u64, 2u64)
                }
                ElementOrder::Inf => {
                    approx::relative_eq!(
                        geometry::normalise_rotation_angle(
                            self.generating_element.proper_angle.unwrap() * (self.power as f64),
                            self.generating_element.threshold
                        ) % (2.0 * std::f64::consts::PI),
                        std::f64::consts::PI,
                        max_relative = self.generating_element.threshold,
                        epsilon = self.generating_element.threshold
                    )
                }
            }
    }

    /// Checks if the symmetry operation is a reflection.
    ///
    /// # Returns
    ///
    /// A flag indicating if this symmetry element is a mirror plane.
    pub fn is_reflection(&self) -> bool {
        !self.is_proper()
            && match self.generating_element.kind {
                SymmetryElementKind::ImproperMirrorPlane => {
                    if let ElementOrder::Int(io) = self.generating_element.proper_order {
                        let pos_pow = (self.power % (io as i32)) as u64;
                        (self.generating_element.proper_fraction.unwrap() * F::from(pos_pow))
                            .fract()
                            == F::from(0u64)
                    } else {
                        approx::relative_eq!(
                            geometry::normalise_rotation_angle(
                                self.generating_element.proper_angle.unwrap() * (self.power as f64),
                                self.generating_element.threshold
                            ) % (2.0 * std::f64::consts::PI),
                            0.0,
                            max_relative = self.generating_element.threshold,
                            epsilon = self.generating_element.threshold
                        )
                    }
                }
                SymmetryElementKind::ImproperInversionCentre => {
                    if let ElementOrder::Int(io) = self.generating_element.proper_order {
                        let pos_pow = (self.power % (io as i32)) as u64;
                        (self.generating_element.proper_fraction.unwrap() * F::from(pos_pow))
                            .fract()
                            == F::new(1u64, 2u64)
                    } else {
                        approx::relative_eq!(
                            geometry::normalise_rotation_angle(
                                self.generating_element.proper_angle.unwrap() * (self.power as f64),
                                self.generating_element.threshold
                            ) % (2.0 * std::f64::consts::PI),
                            std::f64::consts::PI,
                            max_relative = self.generating_element.threshold,
                            epsilon = self.generating_element.threshold
                        )
                    }
                }
                SymmetryElementKind::Proper => false,
            }
    }
}

//    /// Returns the standard symbol for this symmetry element, which does not
//    /// classify certain improper rotation axes into inversion centres or mirror
//    /// planes.
//    ///
//    /// # Returns
//    ///
//    /// The standard symbol for this symmetry element.
//    pub fn get_standard_symbol(&self) -> String {
//        let main_symbol: String = match self.kind {
//            SymmetryElementKind::Proper => "C".to_owned(),
//            SymmetryElementKind::ImproperMirrorPlane => "S".to_owned(),
//            SymmetryElementKind::ImproperInversionCentre => "Ṡ".to_owned(),
//        };
//        format!("{}{}", main_symbol, self.order)
//    }

//    /// Returns the detailed symbol for this symmetry element, which classifies
//    /// certain improper rotation axes into inversion centres or mirror planes.
//    ///
//    /// # Returns
//    ///
//    /// The detailed symbol for this symmetry element.
//    pub fn get_detailed_symbol(&self) -> String {
//        let main_symbol: String = match self.kind {
//            SymmetryElementKind::Proper => "C".to_owned(),
//            SymmetryElementKind::ImproperMirrorPlane => {
//                if self.order == ElementOrder::Int(1) {
//                    "σ".to_owned()
//                } else if self.order == ElementOrder::Int(2) {
//                    "i".to_owned()
//                } else {
//                    "S".to_owned()
//                }
//            }
//            SymmetryElementKind::ImproperInversionCentre => {
//                if self.order == ElementOrder::Int(1) {
//                    "i".to_owned()
//                } else if self.order == ElementOrder::Int(2) {
//                    "σ".to_owned()
//                } else {
//                    "Ṡ".to_owned()
//                }
//            }
//        };

//        let order_string: String =
//            if self.is_proper() || (!self.is_inversion_centre() && !self.is_mirror_plane()) {
//                format!("{}", self.order)
//            } else {
//                "".to_owned()
//            };
//        main_symbol + &self.additional_superscript + &order_string + &self.additional_subscript
//    }

//    /// Returns a copy of the current improper symmetry element that has been
//    /// converted to the required improper kind.
//    ///
//    /// To convert between the two improper kinds, we essentially seek integers
//    /// $`n, n' \in \mathbb{N}_{+}`$ and $`k \in \mathbb{Z}/n\mathbb{Z}`$,
//    /// $`k' \in \mathbb{Z}/n'\mathbb{Z}`$, such that
//    ///
//    /// ```math
//    /// \sigma C_n^k = i C_{n'}^{k'},
//    /// ```
//    ///
//    /// where the axes of all involved elements are parallel. By noting that
//    /// $`\sigma = i C_2`$, we can easily show that
//    ///
//    /// ```math
//    /// \begin{aligned}
//    ///     n' &= \frac{2n}{\operatorname{gcd}(2n, n + 2k)},\\
//    ///     k' &= \frac{n + 2k}{\operatorname{gcd}(2n, n + 2k)} \mod n'.
//    /// \end{aligned}
//    /// ```
//    ///
//    /// The above relations are self-inversed. It can be further shown that
//    /// $`\operatorname{gcd}(n', k') = 1`$. Hence, for symmetry *element*
//    /// conversions, we can simply take $`k' = 1`$. This is because a symmetry
//    /// element plays the role of a generator, and the coprimality of $`n'`$ and
//    /// $`k'`$ means that $`i C_{n'}`$ is as valid a generator as
//    /// $`i C_{n'}^{k'}`$.
//    ///
//    /// # Arguments
//    ///
//    /// * improper_kind - Reference to the required improper kind.
//    ///
//    /// # Returns
//    ///
//    /// A copy of the current improper symmetry element that has been converted.
//    pub fn convert_to_improper_kind(&self, improper_kind: &SymmetryElementKind) -> Self {
//        assert!(
//            !self.is_proper(),
//            "Only improper elements can be converted."
//        );
//        assert_ne!(
//            *improper_kind,
//            SymmetryElementKind::Proper,
//            "`improper_kind` must be one of the improper variants."
//        );

//        if self.kind == *improper_kind {
//            return self.clone();
//        }

//        let dest_order = match self.order {
//            ElementOrder::Int(order_int) => ElementOrder::Int(
//                2 * order_int / (gcd(2 * order_int, order_int + 2)),
//            ),
//            ElementOrder::Inf => ElementOrder::Inf,
//            ElementOrder::Float(_, _) => {
//                panic!();
//            }
//        };
//        Self::builder()
//            .threshold(self.threshold)
//            .order(dest_order)
//            .axis(-self.axis)
//            .kind(improper_kind.clone())
//            .generator(self.generator)
//            .additional_superscript(self.additional_superscript.clone())
//            .additional_subscript(self.additional_subscript.clone())
//            .build()
//            .unwrap()
//    }
//}

//impl fmt::Display for SymmetryElement {
//    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
//        if self.is_identity() || self.is_inversion_centre() {
//            write!(f, "{}", self.get_detailed_symbol())
//        } else {
//            write!(
//                f,
//                "{}({:+.3}, {:+.3}, {:+.3})",
//                self.get_detailed_symbol(),
//                self.axis[0] + 0.0,
//                self.axis[1] + 0.0,
//                self.axis[2] + 0.0
//            )
//        }
//    }
//}

//impl fmt::Debug for SymmetryElement {
//    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
//        if self.is_identity() || self.is_inversion_centre() {
//            write!(f, "{}", self.get_detailed_symbol())
//        } else {
//            write!(
//                f,
//                "{}({:+.3}, {:+.3}, {:+.3})",
//                self.get_detailed_symbol(),
//                self.axis[0] + 0.0,
//                self.axis[1] + 0.0,
//                self.axis[2] + 0.0
//            )
//        }
//    }
//}

//impl PartialEq for SymmetryElement {
//    fn eq(&self, other: &Self) -> bool {
//        if self.is_proper() != other.is_proper() {
//            return false;
//        }

//        if self.is_identity() && other.is_identity() {
//            assert_eq!(misc::calculate_hash(self), misc::calculate_hash(other));
//            return true;
//        }

//        if self.is_inversion_centre() && other.is_inversion_centre() {
//            assert_eq!(misc::calculate_hash(self), misc::calculate_hash(other));
//            return true;
//        }

//        let thresh = (self.threshold * other.threshold).sqrt();

//        if self.kind != other.kind {
//            let converted_other = other.convert_to_improper_kind(&self.kind);
//            let result = (self.order == converted_other.order)
//                && (approx::relative_eq!(
//                    self.axis,
//                    converted_other.axis,
//                    epsilon = thresh,
//                    max_relative = thresh
//                ) || approx::relative_eq!(
//                    self.axis,
//                    -converted_other.axis,
//                    epsilon = thresh,
//                    max_relative = thresh
//                ));
//            if result {
//                assert_eq!(misc::calculate_hash(self), misc::calculate_hash(other));
//            }
//            return result;
//        }
//        let result = (self.order == other.order)
//            && (approx::relative_eq!(
//                self.axis,
//                other.axis,
//                epsilon = thresh,
//                max_relative = thresh
//            ) || approx::relative_eq!(
//                self.axis,
//                -other.axis,
//                epsilon = thresh,
//                max_relative = thresh
//            ));
//        if result {
//            assert_eq!(misc::calculate_hash(self), misc::calculate_hash(other));
//        }
//        result
//    }
//}

//impl Eq for SymmetryElement {}

//impl Hash for SymmetryElement {
//    fn hash<H: Hasher>(&self, state: &mut H) {
//        self.is_proper().hash(state);
//        if self.is_identity() || self.is_inversion_centre() {
//            true.hash(state);
//        } else {
//            match self.kind {
//                SymmetryElementKind::ImproperMirrorPlane => {
//                    let c_self = self
//                        .convert_to_improper_kind(&SymmetryElementKind::ImproperInversionCentre);
//                    c_self.order.hash(state);
//                    let pole = geometry::get_positive_pole(&c_self.axis, c_self.threshold);
//                    pole[0]
//                        .round_factor(self.threshold)
//                        .integer_decode()
//                        .hash(state);
//                    pole[1]
//                        .round_factor(self.threshold)
//                        .integer_decode()
//                        .hash(state);
//                    pole[2]
//                        .round_factor(self.threshold)
//                        .integer_decode()
//                        .hash(state);
//                }
//                _ => {
//                    self.order.hash(state);
//                    let pole = geometry::get_positive_pole(&self.axis, self.threshold);
//                    pole[0]
//                        .round_factor(self.threshold)
//                        .integer_decode()
//                        .hash(state);
//                    pole[1]
//                        .round_factor(self.threshold)
//                        .integer_decode()
//                        .hash(state);
//                    pole[2]
//                        .round_factor(self.threshold)
//                        .integer_decode()
//                        .hash(state);
//                }
//            };
//        }
//    }
//}

//pub const SIG: SymmetryElementKind = SymmetryElementKind::ImproperMirrorPlane;
//pub const INV: SymmetryElementKind = SymmetryElementKind::ImproperInversionCentre;
