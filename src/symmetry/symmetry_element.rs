use approx;
use num::integer::gcd;
use derive_builder::Builder;
use log;
use nalgebra::Vector3;
use std::cmp::Ordering;
use std::fmt;
use std::hash::{Hash, Hasher};

use crate::aux::geometry;
use crate::aux::misc::{self, HashableFloat};

#[cfg(test)]
#[path = "symmetry_element_tests.rs"]
mod symmetry_element_tests;

/// An enum to classify the types of symmetry element.
#[derive(Clone, Debug, PartialEq)]
pub enum SymmetryElementKind {
    /// Proper symmetry element which consists of just a proper rotation axis.
    Proper,

    /// Improper symmetry element in the mirror-plane convention, which consists
    /// of a proper rotation axis and an orthogonal mirror plane.
    ImproperMirrorPlane,

    /// Improper symmetry element in the inversion-centre convention, which
    /// consists of a proper rotation axis and an inversion centre.
    ImproperInversionCentre,
}

/// An enum to handle symmetry element orders which can be integers, floats, or infinity.
#[derive(Clone, Debug)]
pub enum ElementOrder {
    /// Positive integer order.
    Int(u32),

    /// Positive floating point order and a threshold for comparisons.
    Float(f64, f64),

    /// Infinite order.
    Inf,
}

impl ElementOrder {
    pub fn new(order: f64, thresh: f64) -> Self {
        assert!(
            order.is_sign_positive(),
            "Order value {} is invalid. Order values must be strictly positive.",
            order
        );
        if order.is_infinite() {
            return Self::Inf;
        }
        let rounded_order = order.round_factor(thresh);
        if approx::relative_eq!(
            rounded_order,
            rounded_order.round(),
            epsilon = thresh,
            max_relative = thresh
        ) {
            return Self::Int(rounded_order as u32);
        }
        Self::Float(rounded_order, thresh)
    }

    pub fn to_float(&self) -> f64 {
        match self {
            Self::Int(s_i) => *s_i as f64,
            Self::Float(s_f, _) => *s_f,
            Self::Inf => f64::INFINITY,
        }
    }
}

impl PartialEq for ElementOrder {
    fn eq(&self, other: &Self) -> bool {
        match &self {
            Self::Int(s_i) => match &other {
                Self::Int(o_i) => {
                    return s_i == o_i;
                }
                Self::Float(o_f, o_thresh) => {
                    return approx::relative_eq!(
                        *s_i as f64,
                        *o_f,
                        epsilon = *o_thresh,
                        max_relative = *o_thresh
                    );
                }
                Self::Inf => return false,
            },
            Self::Float(s_f, s_thresh) => match &other {
                Self::Int(o_i) => {
                    return approx::relative_eq!(
                        *s_f,
                        *o_i as f64,
                        epsilon = *s_thresh,
                        max_relative = *s_thresh
                    );
                }
                Self::Float(o_f, o_thresh) => {
                    return approx::relative_eq!(
                        *s_f,
                        *o_f,
                        epsilon = (*s_thresh * *o_thresh).sqrt(),
                        max_relative = (*s_thresh * *o_thresh).sqrt(),
                    );
                }
                Self::Inf => {
                    return s_f.is_infinite() && s_f.is_sign_positive();
                }
            },
            Self::Inf => match &other {
                Self::Int(_) => return false,
                Self::Float(o_f, _) => {
                    return o_f.is_infinite() && o_f.is_sign_positive();
                }
                Self::Inf => return true,
            },
        }
    }
}

impl Eq for ElementOrder {}

impl PartialOrd for ElementOrder {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.to_float().partial_cmp(&other.to_float())?)
    }
}

impl Ord for ElementOrder {
    fn cmp(&self, other: &Self) -> Ordering {
        self.to_float().total_cmp(&other.to_float())
    }
}

impl Hash for ElementOrder {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match &self {
            Self::Int(s_i) => {
                s_i.hash(state);
            }
            Self::Float(s_f, s_thresh) => {
                let factor = 1.0 / s_thresh;
                s_f.round_factor(factor).integer_decode().hash(state);
            }
            Self::Inf => {
                f64::INFINITY.integer_decode().hash(state);
            }
        }
    }
}

impl fmt::Display for ElementOrder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self {
            Self::Int(s_i) => write!(f, "{}", s_i),
            Self::Float(s_f, s_thresh) => {
                match approx::relative_eq!(
                    *s_f,
                    s_f.round(),
                    epsilon = *s_thresh,
                    max_relative = *s_thresh
                ) {
                    true => write!(f, "{:.0}", s_f),
                    false => write!(f, "{:.3}", s_f),
                }
            }
            Self::Inf => write!(f, "{}", "∞".to_owned()),
        }
    }
}

/// A struct for storing and managing symmetry elements.
#[derive(Builder, Clone)]
pub struct SymmetryElement {
    /// The rotational order of the symmetry element defined by $2\pi/\phi$
    /// where $\phi \in (0, \pi] \cup \lbrace2\pi\rbrace$ is the positive angle
    /// of the rotation about [`Self::axis`] associated with this element. This
    /// is **not** necessarily an integer, and can also take the special value
    /// of `-1.0` to indicate that this symmetry element is of infinite order.
    pub order: ElementOrder,

    /// The normalised axis of the symmetry element.
    #[builder(setter(custom))]
    pub axis: Vector3<f64>,

    /// The kind of the symmetry element.
    #[builder(default = "SymmetryElementKind::Proper")]
    kind: SymmetryElementKind,

    /// A flag indicating whether the symmetry element is a generator of the
    /// group to which it belongs.
    #[builder(default = "false")]
    generator: bool,

    /// A threshold for approximate equality comparisons.
    #[builder(setter(custom))]
    threshold: f64,

    /// An additional superscript for distinguishing the symmetry element.
    #[builder(default = "\"\".to_owned()")]
    pub additional_superscript: String,

    /// An additional subscript for distinguishing the symmetry element.
    #[builder(default = "\"\".to_owned()")]
    pub additional_subscript: String,
}

impl SymmetryElementBuilder {
    pub fn axis(&mut self, axs: Vector3<f64>) -> &mut Self {
        let thresh = self.threshold.unwrap();
        if approx::relative_eq!(axs.norm(), 1.0, epsilon = thresh, max_relative = thresh) {
            self.axis = Some(axs);
        } else {
            log::warn!("Axis not normalised. Normalising...");
            self.axis = Some(axs.normalize());
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
    pub fn builder() -> SymmetryElementBuilder {
        SymmetryElementBuilder::default()
    }

    /// Checks if the symmetry element is proper or not.
    ///
    /// # Returns
    ///
    /// A flag indicating if the symmetry element is proper.
    pub fn is_proper(&self) -> bool {
        self.kind == SymmetryElementKind::Proper
    }

    /// Checks if the symmetry element is an identity element.
    ///
    /// # Returns
    ///
    /// A flag indicating if this symmetry element is an identity element.
    pub fn is_identity(&self) -> bool {
        self.kind == SymmetryElementKind::Proper && self.order == ElementOrder::Int(1)
    }

    /// Checks if the symmetry element is an inversion centre.
    ///
    /// # Returns
    ///
    /// A flag indicating if this symmetry element is an inversion centre.
    pub fn is_inversion_centre(&self) -> bool {
        (matches!(self.kind, SymmetryElementKind::ImproperMirrorPlane)
            && self.order == ElementOrder::Int(2))
            || (self.kind == SymmetryElementKind::ImproperInversionCentre
                && self.order == ElementOrder::Int(1))
    }

    /// Checks if the symmetry element is a binary rotation axis.
    ///
    /// # Returns
    ///
    /// A flag indicating if this symmetry element is a binary rotation axis.
    pub fn is_binary_rotation_axis(&self) -> bool {
        self.kind == SymmetryElementKind::Proper && self.order == ElementOrder::Int(2)
    }

    /// Checks if the symmetry element is a mirror plane.
    ///
    /// # Returns
    ///
    /// A flag indicating if this symmetry element is a mirror plane.
    pub fn is_mirror_plane(&self) -> bool {
        (matches!(self.kind, SymmetryElementKind::ImproperMirrorPlane)
            && self.order == ElementOrder::Int(1))
            || (self.kind == SymmetryElementKind::ImproperInversionCentre
                && self.order == ElementOrder::Int(2))
    }

    /// Returns the standard symbol for this symmetry element, which does not
    /// classify certain improper rotation axes into inversion centres or mirror
    /// planes.
    ///
    /// # Returns
    ///
    /// The standard symbol for this symmetry element.
    pub fn get_standard_symbol(&self) -> String {
        let main_symbol: String = match self.kind {
            SymmetryElementKind::Proper => "C".to_owned(),
            SymmetryElementKind::ImproperMirrorPlane => "S".to_owned(),
            SymmetryElementKind::ImproperInversionCentre => "Ṡ".to_owned(),
        };
        format!("{}{}", main_symbol, self.order)
    }

    /// Returns the detailed symbol for this symmetry element, which classifies
    /// certain improper rotation axes into inversion centres or mirror planes.
    ///
    /// # Returns
    ///
    /// The detailed symbol for this symmetry element.
    pub fn get_detailed_symbol(&self) -> String {
        let main_symbol: String = match self.kind {
            SymmetryElementKind::Proper => "C".to_owned(),
            SymmetryElementKind::ImproperMirrorPlane => {
                if self.order == ElementOrder::Int(1) {
                    "σ".to_owned()
                } else if self.order == ElementOrder::Int(2) {
                    "i".to_owned()
                } else {
                    "S".to_owned()
                }
            }
            SymmetryElementKind::ImproperInversionCentre => {
                if self.order == ElementOrder::Int(1) {
                    "i".to_owned()
                } else if self.order == ElementOrder::Int(2) {
                    "σ".to_owned()
                } else {
                    "Ṡ".to_owned()
                }
            }
        };

        let order_string: String =
            if self.is_proper() || (!self.is_inversion_centre() && !self.is_mirror_plane()) {
                format!("{}", self.order)
            } else {
                "".to_owned()
            };
        main_symbol + &self.additional_superscript + &order_string + &self.additional_subscript
    }

    /// Returns a copy of the current improper symmetry element that has been
    /// converted to the required improper kind.
    ///
    /// # Arguments
    ///
    /// * improper_kind - Reference to the required improper kind.
    ///
    /// # Returns
    ///
    /// A copy of the current improper symmetry element that has been converted.
    pub fn convert_to_improper_kind(&self, improper_kind: &SymmetryElementKind) -> Self {
        assert!(
            !self.is_proper(),
            "Only improper elements can be converted."
        );
        assert_ne!(
            *improper_kind,
            SymmetryElementKind::Proper,
            "`improper_kind` must be one of the improper variants."
        );

        if self.kind == *improper_kind {
            return self.clone();
        }

        let dest_order = match self.order {
            ElementOrder::Int(order_int) => ElementOrder::Int(2 * order_int / (gcd(2 * order_int, order_int + 2))),
            ElementOrder::Inf => ElementOrder::Inf,
            ElementOrder::Float(_, _) => { panic!(); },
            // let self_basic_angle = geometry::normalise_rotation_angle(
            //     2.0 * std::f64::consts::PI / self.order.to_float(),
            //     self.threshold,
            // );
            // let dest_basic_angle = std::f64::consts::PI - self_basic_angle;
            // if dest_basic_angle.abs() > self.threshold {
            //     ElementOrder::new(
            //         2.0 * std::f64::consts::PI / dest_basic_angle,
            //         self.threshold,
            //     )
            // } else {
            //     ElementOrder::Int(1)
            // }
        };
        Self::builder()
            .threshold(self.threshold)
            .order(dest_order)
            .axis(-self.axis)
            .kind(improper_kind.clone())
            .generator(self.generator)
            .additional_superscript(self.additional_superscript.clone())
            .additional_subscript(self.additional_subscript.clone())
            .build()
            .unwrap()
    }
}

impl fmt::Display for SymmetryElement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_identity() || self.is_inversion_centre() {
            write!(f, "{}", self.get_detailed_symbol())
        } else {
            write!(
                f,
                "{}({:+.3}, {:+.3}, {:+.3})",
                self.get_detailed_symbol(),
                self.axis[0] + 0.0,
                self.axis[1] + 0.0,
                self.axis[2] + 0.0
            )
        }
    }
}

impl fmt::Debug for SymmetryElement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_identity() || self.is_inversion_centre() {
            write!(f, "{}", self.get_detailed_symbol())
        } else {
            write!(
                f,
                "{}({:+.3}, {:+.3}, {:+.3})",
                self.get_detailed_symbol(),
                self.axis[0] + 0.0,
                self.axis[1] + 0.0,
                self.axis[2] + 0.0
            )
        }
    }
}

impl PartialEq for SymmetryElement {
    fn eq(&self, other: &Self) -> bool {
        if self.is_proper() != other.is_proper() {
            return false;
        }

        if self.is_identity() && other.is_identity() {
            assert_eq!(misc::calculate_hash(self), misc::calculate_hash(other));
            return true;
        }

        if self.is_inversion_centre() && other.is_inversion_centre() {
            assert_eq!(misc::calculate_hash(self), misc::calculate_hash(other));
            return true;
        }

        let thresh = (self.threshold * other.threshold).sqrt();

        if self.kind != other.kind {
            let converted_other = other.convert_to_improper_kind(&self.kind);
            let result = (self.order == converted_other.order)
                && (approx::relative_eq!(
                    self.axis,
                    converted_other.axis,
                    epsilon = thresh,
                    max_relative = thresh
                ) || approx::relative_eq!(
                    self.axis,
                    -converted_other.axis,
                    epsilon = thresh,
                    max_relative = thresh
                ));
            if result {
                assert_eq!(misc::calculate_hash(self), misc::calculate_hash(other));
            }
            return result;
        }
        let result = (self.order == other.order)
            && (approx::relative_eq!(
                self.axis,
                other.axis,
                epsilon = thresh,
                max_relative = thresh
            ) || approx::relative_eq!(
                self.axis,
                -other.axis,
                epsilon = thresh,
                max_relative = thresh
            ));
        if result {
            assert_eq!(misc::calculate_hash(self), misc::calculate_hash(other));
        }
        result
    }
}

impl Eq for SymmetryElement {}

impl Hash for SymmetryElement {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.is_proper().hash(state);
        if self.is_identity() || self.is_inversion_centre() {
            true.hash(state);
        } else {
            match self.kind {
                SymmetryElementKind::ImproperMirrorPlane => {
                    let c_self = self
                        .convert_to_improper_kind(&SymmetryElementKind::ImproperInversionCentre);
                    c_self.order.hash(state);
                    let pole = geometry::get_positive_pole(&c_self.axis, c_self.threshold);
                    pole[0].round_factor(self.threshold).integer_decode().hash(state);
                    pole[1].round_factor(self.threshold).integer_decode().hash(state);
                    pole[2].round_factor(self.threshold).integer_decode().hash(state);
                }
                _ => {
                    self.order.hash(state);
                    let pole = geometry::get_positive_pole(&self.axis, self.threshold);
                    pole[0].round_factor(self.threshold).integer_decode().hash(state);
                    pole[1].round_factor(self.threshold).integer_decode().hash(state);
                    pole[2].round_factor(self.threshold).integer_decode().hash(state);
                }
            };
        }
    }
}

pub const ORDER_1: ElementOrder = ElementOrder::Int(1);
pub const ORDER_2: ElementOrder = ElementOrder::Int(2);
pub const ORDER_I: ElementOrder = ElementOrder::Inf;

pub const SIG: SymmetryElementKind = SymmetryElementKind::ImproperMirrorPlane;
pub const INV: SymmetryElementKind = SymmetryElementKind::ImproperInversionCentre;
