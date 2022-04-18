use nalgebra::Vector3;
use std::fmt;
use std::hash::{Hash, Hasher};
use approx;
use log;
use derive_builder::Builder;

use crate::aux::geometry;
use crate::aux::misc;
use crate::aux::misc::HashableFloat;

#[cfg(test)]
#[path = "symmetry_elements_tests.rs"]
mod symmetry_elements_tests;

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

/// A struct for storing and managing symmetry elements.
#[derive(Builder, Clone)]
pub struct SymmetryElement {
    /// The rotational order of the symmetry element defined by $2\pi/\phi$
    /// where $\phi \in (0, \pi] \cup \lbrace2\pi\rbrace$ is the positive angle
    /// of the rotation about [`Self::axis`] associated with this element. This
    /// is **not** necessarily an integer, and can also take the special value
    /// of `-1.0` to indicate that this symmetry element is of infinite order.
    #[builder(setter(custom))]
    order: f64,

    /// The normalised axis of the symmetry element.
    #[builder(setter(custom))]
    axis: Vector3<f64>,

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
    additional_superscript: String,

    /// An additional subscript for distinguishing the symmetry element.
    #[builder(default = "\"\".to_owned()")]
    additional_subscript: String,
}

impl SymmetryElementBuilder {
    pub fn order(&mut self, ord: f64) -> &mut Self {
        let thresh = self.threshold.unwrap();
        if ord > thresh || approx::relative_eq!(ord, -1.0, epsilon = thresh, max_relative = thresh)
        {
            self.order = Some(ord);
        } else {
            log::error!(
                "Order value {} is invalid. Order must be positive or -1.0.",
                ord
            );
            self.order = None;
        }
        self
    }

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
    fn is_proper(&self) -> bool {
        self.kind == SymmetryElementKind::Proper
    }

    /// Checks if the symmetry element is an identity element.
    ///
    /// # Returns
    ///
    /// A flag indicating if this symmetry element is an identity element.
    fn is_identity(&self) -> bool {
        self.kind == SymmetryElementKind::Proper
            && approx::relative_eq!(
                self.order,
                1.0,
                epsilon = self.threshold,
                max_relative = self.threshold
            )
    }

    /// Checks if the symmetry element is an inversion centre.
    ///
    /// # Returns
    ///
    /// A flag indicating if this symmetry element is an inversion centre.
    fn is_inversion_centre(&self) -> bool {
        (matches!(self.kind, SymmetryElementKind::ImproperMirrorPlane)
            && approx::relative_eq!(
                self.order,
                2.0,
                epsilon = self.threshold,
                max_relative = self.threshold
            ))
            || (self.kind == SymmetryElementKind::ImproperInversionCentre
                && approx::relative_eq!(
                    self.order,
                    1.0,
                    epsilon = self.threshold,
                    max_relative = self.threshold
                ))
    }

    /// Checks if the symmetry element is a binary rotation axis.
    ///
    /// # Returns
    ///
    /// A flag indicating if this symmetry element is a binary rotation axis.
    fn is_binary_rotation_axis(&self) -> bool {
        self.kind == SymmetryElementKind::Proper
            && approx::relative_eq!(
                self.order,
                2.0,
                epsilon = self.threshold,
                max_relative = self.threshold
            )
    }

    /// Checks if the symmetry element is a mirror plane.
    ///
    /// # Returns
    ///
    /// A flag indicating if this symmetry element is a mirror plane.
    fn is_mirror_plane(&self) -> bool {
        (matches!(self.kind, SymmetryElementKind::ImproperMirrorPlane)
            && approx::relative_eq!(
                self.order,
                1.0,
                epsilon = self.threshold,
                max_relative = self.threshold
            ))
            || (self.kind == SymmetryElementKind::ImproperInversionCentre
                && approx::relative_eq!(
                    self.order,
                    2.0,
                    epsilon = self.threshold,
                    max_relative = self.threshold
                ))
    }

    /// Returns the standard symbol for this symmetry element, which does not
    /// classify certain improper rotation axes into inversion centres or mirror
    /// planes.
    ///
    /// # Returns
    ///
    /// The standard symbol for this symmetry element.
    fn get_standard_symbol(&self) -> String {
        let main_symbol: String = match self.kind {
            SymmetryElementKind::Proper => "C".to_owned(),
            SymmetryElementKind::ImproperMirrorPlane => "S".to_owned(),
            SymmetryElementKind::ImproperInversionCentre => "Ṡ".to_owned(),
        };

        let order_string: String = match approx::relative_eq!(
            self.order,
            self.order.round(),
            epsilon = self.threshold,
            max_relative = self.threshold
        ) {
            true => match approx::relative_eq!(
                self.order,
                -1.0,
                epsilon = self.threshold,
                max_relative = self.threshold
            ) {
                true => "∞".to_owned(),
                false => format!("{:.0}", self.order),
            },
            false => format!("{:.3}", self.order),
        };
        main_symbol + &order_string
    }

    /// Returns the detailed symbol for this symmetry element, which classifies
    /// certain improper rotation axes into inversion centres or mirror planes.
    ///
    /// # Returns
    ///
    /// The detailed symbol for this symmetry element.
    fn get_detailed_symbol(&self) -> String {
        let main_symbol: String = match self.kind {
            SymmetryElementKind::Proper => "C".to_owned(),
            SymmetryElementKind::ImproperMirrorPlane => {
                if approx::relative_eq!(
                    self.order,
                    1.0,
                    epsilon = self.threshold,
                    max_relative = self.threshold
                ) {
                    "σ".to_owned()
                } else if approx::relative_eq!(
                    self.order,
                    2.0,
                    epsilon = self.threshold,
                    max_relative = self.threshold
                ) {
                    "i".to_owned()
                } else {
                    "S".to_owned()
                }
            }
            SymmetryElementKind::ImproperInversionCentre => {
                if approx::relative_eq!(
                    self.order,
                    1.0,
                    epsilon = self.threshold,
                    max_relative = self.threshold
                ) {
                    "i".to_owned()
                } else if approx::relative_eq!(
                    self.order,
                    2.0,
                    epsilon = self.threshold,
                    max_relative = self.threshold
                ) {
                    "σ".to_owned()
                } else {
                    "Ṡ".to_owned()
                }
            }
        };

        let order_string: String =
            if (self.is_proper() && self.order > 0.0) || (!self.is_proper() && self.order > 2.0) {
                match approx::relative_eq!(
                    self.order,
                    self.order.round(),
                    epsilon = self.threshold,
                    max_relative = self.threshold
                ) {
                    true => match approx::relative_eq!(
                        self.order,
                        -1.0,
                        epsilon = self.threshold,
                        max_relative = self.threshold
                    ) {
                        true => "∞".to_owned(),
                        false => format!("{:.0}", self.order),
                    },
                    false => format!("{:.3}", self.order),
                }
            } else if approx::relative_eq!(
                self.order,
                -1.0,
                epsilon = self.threshold,
                max_relative = self.threshold
            ) {
                "∞".to_owned()
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
    fn convert_to_improper_kind(&self, improper_kind: &SymmetryElementKind) -> Self {
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

        let mut dest_order = self.order.clone();
        if self.order > 0.0 {
            let self_basic_angle = geometry::normalise_rotation_angle(
                2.0 * std::f64::consts::PI / self.order,
                self.threshold,
            );
            let dest_basic_angle = std::f64::consts::PI - self_basic_angle;
            if dest_basic_angle.abs() > self.threshold {
                dest_order = 2.0 * std::f64::consts::PI / dest_basic_angle;
            } else {
                dest_order = 1.0;
            }
        }
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
        // if (self.is_binary_rotation_axis() && other.is_binary_rotation_axis())
        //     || (self.is_mirror_plane() && other.is_mirror_plane())
        // {
        //     let result = approx::relative_eq!(
        //         self.axis,
        //         other.axis,
        //         epsilon = thresh,
        //         max_relative = thresh
        //     ) || approx::relative_eq!(
        //         self.axis,
        //         -other.axis,
        //         epsilon = thresh,
        //         max_relative = thresh
        //     );
        //     if result {
        //         assert_eq!(misc::calculate_hash(self), misc::calculate_hash(other));
        //     }
        //     return result;
        // }

        if self.kind != other.kind {
            let converted_other = other.convert_to_improper_kind(&self.kind);
            let result = approx::relative_eq!(
                self.order,
                converted_other.order,
                epsilon = thresh,
                max_relative = thresh
            ) && (
                approx::relative_eq!(
                    self.axis,
                    converted_other.axis,
                    epsilon = thresh,
                    max_relative = thresh
                ) || approx::relative_eq!(
                    self.axis,
                    -converted_other.axis,
                    epsilon = thresh,
                    max_relative = thresh
                )
            );
            if result {
                assert_eq!(misc::calculate_hash(self), misc::calculate_hash(other));
            }
            return result;
        }
        let result = approx::relative_eq!(
            self.order,
            other.order,
            epsilon = thresh,
            max_relative = thresh
        ) && (
            approx::relative_eq!(
                self.axis,
                other.axis,
                epsilon = thresh,
                max_relative = thresh
            ) || approx::relative_eq!(
                self.axis,
                -other.axis,
                epsilon = thresh,
                max_relative = thresh
            )
        );
        if result {
            assert_eq!(misc::calculate_hash(self), misc::calculate_hash(other));
        }
        result
    }
}


impl Eq for SymmetryElement { }


impl Hash for SymmetryElement {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.is_proper().hash(state);
        if self.is_identity() || self.is_inversion_centre() {
            true.hash(state);
        } else {
            let factor = 1.0 / self.threshold;
            self.order.round_factor(factor).integer_decode().hash(state);
            let pole = geometry::get_positive_pole(&self.axis, self.threshold);
            pole[0].round_factor(factor).integer_decode().hash(state);
            pole[1].round_factor(factor).integer_decode().hash(state);
            pole[2].round_factor(factor).integer_decode().hash(state);
        }
    }
}
