use std::cmp;
use std::convert::TryInto;
use std::fmt;
use std::hash::{Hash, Hasher};

use approx;
use derive_builder::Builder;
use fraction;
use log;
use nalgebra::Vector3;
use num::integer::gcd;

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
pub enum AssociatedSpinRotation {
    /// Variant indicating that no associated spin rotation shall be taken into account.
    Ignored,

    /// Variant indicating that the associated spin rotation shall be taken into account, with the
    /// accompanying boolean indicating whether the spin rotation is **normal** (*i.e.* its rotation
    /// angle is the same as that of the proper rotation in the spatial symmetry operation), or
    /// **inverse** (*i.e. its rotation angle is $`\theta + 2\pi`$, where $`\theta`$ is the proper
    /// rotation angle renormalised to be in the $`[0, 2\pi)`$ range).
    Active(bool),
}

impl AssociatedSpinRotation {
    /// Indicates if the associated spin rotation is an active spin rotation.
    fn is_active_spin_rotation(&self) -> bool {
        match self {
            AssociatedSpinRotation::Active(_) => true,
            AssociatedSpinRotation::Ignored => false,
        }
    }

    /// Indicates if the associated spin rotation is an inverse spin rotation.
    fn is_inverse_spin_rotation(&self) -> bool {
        match self {
            AssociatedSpinRotation::Active(false) => true,
            AssociatedSpinRotation::Active(true) | AssociatedSpinRotation::Ignored => false,
        }
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
    pub proper_power: Option<u32>,

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

    /// The normalised axis of the symmetry element.
    #[builder(setter(custom))]
    pub axis: Vector3<f64>,

    /// The spatial and time-reversal kind of the symmetry element.
    #[builder(default = "SymmetryElementKind::Proper(false)")]
    pub kind: SymmetryElementKind,

    /// The associated spin rotation of the symmetry element, if any.
    pub spinrot: AssociatedSpinRotation,

    /// A flag indicating whether the symmetry element is a generator of the
    /// group to which it belongs.
    #[builder(default = "false")]
    generator: bool,

    /// A threshold for approximate equality comparisons.
    #[builder(setter(custom))]
    pub threshold: f64,

    /// An additional superscript for distinguishing the symmetry element.
    #[builder(default = "String::new()")]
    pub additional_superscript: String,

    /// An additional subscript for distinguishing the symmetry element.
    #[builder(default = "String::new()")]
    pub additional_subscript: String,
}

impl SymmetryElementBuilder {
    pub fn proper_power(&mut self, prop_pow: u32) -> &mut Self {
        let proper_order = self
            .proper_order
            .as_ref()
            .expect("Proper order has not been set.");
        self.proper_power = match proper_order {
            ElementOrder::Int(io) => {
                let residual = prop_pow % io;
                if residual == 0 {
                    Some(Some(*io))
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
            ElementOrder::Inf => Some(Some(geometry::normalise_rotation_angle(
                ang,
                self.threshold.expect("Threshold value has not been set."),
            ))),
        };
        self
    }

    fn calc_proper_fraction(&self) -> Option<F> {
        let proper_order = self
            .proper_order
            .as_ref()
            .expect("Proper order has not been set.");
        match proper_order {
            ElementOrder::Int(io) => Some(F::new(
                self.proper_power
                    .expect("Proper power has not been set.")
                    .expect("No proper powers found."),
                *io,
            )),
            ElementOrder::Inf => None,
        }
    }

    fn calc_proper_angle(&self) -> Option<f64> {
        let proper_order = self
            .proper_order
            .as_ref()
            .expect("Proper order has not been set.");
        match proper_order {
            ElementOrder::Int(io) => Some(geometry::normalise_rotation_angle(
                (f64::from(
                    self.proper_power
                        .expect("Proper power has not been set.")
                        .expect("No proper powers found."),
                ) / (f64::from(*io)))
                    * 2.0
                    * std::f64::consts::PI,
                self.threshold.expect("Threshold value has not been set."),
            )),
            ElementOrder::Inf => self.proper_angle.unwrap_or(None),
        }
    }

    pub fn axis(&mut self, axs: Vector3<f64>) -> &mut Self {
        let thresh = self.threshold.expect("Threshold value has not been set.");
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
    #[must_use]
    pub fn builder() -> SymmetryElementBuilder {
        SymmetryElementBuilder::default()
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
    fn contains_active_spin_rotation(&self) -> bool {
        self.spinrot.is_active_spin_rotation()
    }

    /// Checks if the symmetry element contains an inverse spin rotation.
    #[must_use]
    fn contains_inverse_spin_rotation(&self) -> bool {
        self.spinrot.is_inverse_spin_rotation()
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
    pub fn is_nonsr_proper(&self, tr: bool) -> bool {
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
    pub fn is_nonsr_identity(&self, tr: bool) -> bool {
        self.kind == SymmetryElementKind::Proper(tr)
            && self.proper_fraction == Some(F::from(1))
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
    pub fn is_nonsr_inversion_centre(&self, tr: bool) -> bool {
        (self.kind == SymmetryElementKind::ImproperMirrorPlane(tr)
         && self.proper_fraction == Some(F::new(1u32, 2u32)))
            || (self.kind == SymmetryElementKind::ImproperInversionCentre(tr)
                && self.proper_fraction == Some(F::from(1)))
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
    pub fn is_nonsr_binary_rotation_axis(&self, tr: bool) -> bool {
        self.kind == SymmetryElementKind::Proper(tr)
            && self.proper_fraction == Some(F::new(1u32, 2u32))
    }

    /// Checks if the symmetry element is spatially a mirror plane and has the specified time-reversal
    /// attribute.
    ///
    /// # Arguments
    ///
    /// * `tr` - A flag indicating if time reversal is to be considered.
    ///
    /// # Returns
    ///
    /// A flag indicating if this symmetry element is spatially a mirror plane and has the specified
    /// time-reversal attribute.
    #[must_use]
    pub fn is_nonsr_mirror_plane(&self, tr: bool) -> bool {
        (self.kind == SymmetryElementKind::ImproperMirrorPlane(tr)
         && self.proper_fraction == Some(F::from(1)))
            || (self.kind == SymmetryElementKind::ImproperInversionCentre(tr)
                && self.proper_fraction == Some(F::new(1u32, 2u32)))
    }

    /// Returns the standard symbol for this symmetry element, which does not
    /// classify certain improper rotation axes into inversion centres or mirror
    /// planes.
    ///
    /// Some additional symbols that can be unconventional include:
    ///
    /// * `θ`: time reversal,
    /// * `Σ`: the normal spin rotation associated with the proper rotation of this element,
    /// * `QΣ`: the inverse spin rotation associated with the proper rotation of this element.
    ///
    /// See [`AssociatedSpinRotation`] for further information.
    ///
    /// # Returns
    ///
    /// The standard symbol for this symmetry element.
    #[must_use]
    pub fn get_standard_symbol(&self) -> String {
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
                if self.proper_order != ElementOrder::Inf && self.proper_power == Some(1) {
                    format!("{tr_sym}S")
                } else {
                    format!("{tr_sym}σC")
                }
            }
            SymmetryElementKind::ImproperInversionCentre(_) => {
                if self.proper_order != ElementOrder::Inf && self.proper_power == Some(1) {
                    format!("{tr_sym}Ṡ")
                } else {
                    format!("{tr_sym}iC")
                }
            }
        };
        let proper_power = if let Some(pow) = self.proper_power {
            if pow > 1 {
                format!("^{pow}")
            } else {
                String::new()
            }
        } else {
            String::new()
        };
        let sr_sym = match self.spinrot {
            AssociatedSpinRotation::Ignored => "",
            AssociatedSpinRotation::Active(true) => "Σ·",
            AssociatedSpinRotation::Active(false) => "QΣ·",
        };
        format!("{sr_sym}{main_symbol}{}{proper_power}", self.proper_order)
    }

    /// Returns the detailed symbol for this symmetry element, which classifies
    /// special symmetry elements (identity, inversion centre, mirror planes).
    ///
    /// # Returns
    ///
    /// The detailed symbol for this symmetry element.
    #[must_use]
    pub fn get_detailed_symbol(&self) -> String {
        let (main_symbol, needs_power) = match self.kind {
            SymmetryElementKind::Proper(tr) => {
                if self.is_nonsr_identity(tr) {
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
                if self.is_nonsr_mirror_plane(tr) {
                    (format!("{tr_sym}σ"), false)
                } else if self.is_nonsr_inversion_centre(tr) {
                    (format!("{tr_sym}i"), false)
                } else if self.proper_order == ElementOrder::Inf || self.proper_power == Some(1) {
                    (format!("{tr_sym}S"), false)
                } else {
                    (format!("{tr_sym}σC"), true)
                }
            }
            SymmetryElementKind::ImproperInversionCentre(tr) => {
                let tr_sym = if tr { "θ·" } else { "" };
                if self.is_nonsr_mirror_plane(tr) {
                    (format!("{tr_sym}σ"), false)
                } else if self.is_nonsr_inversion_centre(tr) {
                    (format!("{tr_sym}i"), false)
                } else if self.proper_order == ElementOrder::Inf || self.proper_power == Some(1) {
                    (format!("{tr_sym}Ṡ"), false)
                } else {
                    (format!("{tr_sym}iC"), true)
                }
            }
        };

        let order_string: String = if self.is_nonsr_identity(false)
            || self.is_nonsr_inversion_centre(false)
            || self.is_nonsr_mirror_plane(false)
            || self.is_nonsr_identity(true)
            || self.is_nonsr_inversion_centre(true)
            || self.is_nonsr_mirror_plane(true)
        {
            String::new()
        } else {
            format!("{}", self.proper_order)
        };

        let proper_power = if needs_power {
            match self.proper_order {
                ElementOrder::Int(_) => {
                    if let Some(pow) = self.proper_power {
                        if pow > 1 {
                            format!("^{pow}")
                        } else {
                            String::new()
                        }
                    } else {
                        String::new()
                    }
                }
                ElementOrder::Inf => String::new(),
            }
        } else {
            String::new()
        };

        let sr_sym = match self.spinrot {
            AssociatedSpinRotation::Ignored => "",
            AssociatedSpinRotation::Active(true) => "Σ·",
            AssociatedSpinRotation::Active(false) => "QΣ·",
        };
        sr_sym.to_owned()
            + &main_symbol
            + &self.additional_superscript
            + &order_string
            + &proper_power
            + &self.additional_subscript
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
        assert!(
            !(self.is_nonsr_proper(false) || self.is_nonsr_proper(true)),
            "Only improper elements can be converted."
        );
        let improper_kind = improper_kind.to_tr(self.contains_time_reversal());
        assert!(
            !matches!(improper_kind, SymmetryElementKind::Proper(_)),
            "`improper_kind` must be one of the improper variants."
        );

        if self.kind == improper_kind {
            return self.clone();
        }

        let dest_order = match self.proper_order {
            ElementOrder::Int(order_int) => ElementOrder::Int(
                2 * order_int
                    / (gcd(
                        2 * order_int,
                        order_int + 2 * self.proper_power.expect("No proper powers found."),
                    )),
            ),
            ElementOrder::Inf => ElementOrder::Inf,
        };
        let dest_proper_power = if preserves_power {
            match self.proper_order {
                ElementOrder::Int(order_int) => {
                    let pow = self.proper_power.expect("No proper powers found.");
                    (order_int + 2 * pow) / (gcd(2 * order_int, order_int + 2 * pow))
                }
                ElementOrder::Inf => 1,
            }
        } else {
            1
        };
        match dest_order {
            ElementOrder::Int(_) => Self::builder()
                .threshold(self.threshold)
                .proper_order(dest_order)
                .proper_power(dest_proper_power)
                .axis(self.axis)
                .kind(improper_kind)
                .spinrot(self.spinrot.clone())
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
                        .proper_angle(std::f64::consts::PI + ang)
                        .axis(self.axis)
                        .kind(improper_kind)
                        .spinrot(self.spinrot.clone())
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
                        .axis(self.axis)
                        .kind(improper_kind)
                        .spinrot(self.spinrot.clone())
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
    pub fn add_spin_rotation(&self, normal: bool) -> Option<Self> {
        if self.contains_active_spin_rotation() {
            None
        } else {
            let mut element = self.clone();
            element.spinrot = AssociatedSpinRotation::Active(normal);
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
        let normalised_axis = self.axis.normalize();
        let rev_normalised_axis = Vector3::new(
            normalised_axis[(2)],
            normalised_axis[(1)],
            normalised_axis[(0)],
        );
        let (amax_arg, amax_val) = rev_normalised_axis.abs().argmax();
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

impl fmt::Display for SymmetryElement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let proper_angle = match self.proper_order {
            ElementOrder::Inf => {
                if let Some(ang) = self.proper_angle {
                    format!("({ang:+.3})")
                } else {
                    String::new()
                }
            }
            ElementOrder::Int(_) => String::new(),
        };
        if self.is_nonsr_identity(false)
            || self.is_nonsr_inversion_centre(false)
            || self.is_nonsr_identity(true)
            || self.is_nonsr_inversion_centre(true)
        {
            write!(f, "{}", self.get_detailed_symbol())
        } else {
            write!(
                f,
                "{}{}({:+.3}, {:+.3}, {:+.3})",
                self.get_detailed_symbol(),
                proper_angle,
                self.axis[0] + 0.0,
                self.axis[1] + 0.0,
                self.axis[2] + 0.0,
            )
        }
    }
}

impl fmt::Debug for SymmetryElement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let proper_angle = match self.proper_order {
            ElementOrder::Inf => {
                if let Some(ang) = self.proper_angle {
                    format!("({ang:+.3})")
                } else {
                    String::new()
                }
            }
            ElementOrder::Int(_) => String::new(),
        };
        write!(
            f,
            "{}{}({:+.3}, {:+.3}, {:+.3})",
            self.get_standard_symbol(),
            proper_angle,
            self.axis[0] + 0.0,
            self.axis[1] + 0.0,
            self.axis[2] + 0.0,
        )
    }
}

impl PartialEq for SymmetryElement {
    /// Two symmetry elements are equal if and only if the following conditions
    /// are all satisfied:
    ///
    /// * they are both proper or improper;
    /// * they both have the same time reversal properties;
    /// * their axes are either parallel or anti-parallel;
    /// * their proper rotation angles have equal absolute values.
    ///
    /// For improper elements, proper rotation angles are taken in the inversion
    /// centre convention.
    #[allow(clippy::too_many_lines)]
    fn eq(&self, other: &Self) -> bool {
        if self.spinrot != other.spinrot {
            return false;
        }

        if self.contains_time_reversal() != other.contains_time_reversal() {
            return false;
        }

        let tr = self.contains_time_reversal();

        if self.is_nonsr_proper(tr) != other.is_nonsr_proper(tr) {
            return false;
        }

        if self.is_nonsr_identity(tr) && other.is_nonsr_identity(tr) {
            assert_eq!(
                misc::calculate_hash(self),
                misc::calculate_hash(other),
                "{self} and {other} have unequal hashes."
            );
            return true;
        }

        if self.is_nonsr_inversion_centre(tr) && other.is_nonsr_inversion_centre(tr) {
            assert_eq!(
                misc::calculate_hash(self),
                misc::calculate_hash(other),
                "{self} and {other} have unequal hashes."
            );
            return true;
        }

        let thresh = (self.threshold * other.threshold).sqrt();

        if self.kind != other.kind {
            let converted_other = other.convert_to_improper_kind(&self.kind, false);
            let result = approx::relative_eq!(
                geometry::get_positive_pole(&self.axis, thresh),
                geometry::get_positive_pole(&converted_other.axis, thresh),
                epsilon = thresh,
                max_relative = thresh
            ) && if let ElementOrder::Inf = self.proper_order {
                if let ElementOrder::Inf = converted_other.proper_order {
                    if let Some(s_angle) = self.proper_angle {
                        if let Some(o_angle) = converted_other.proper_angle {
                            approx::relative_eq!(
                                s_angle,
                                o_angle,
                                epsilon = thresh,
                                max_relative = thresh
                            )
                        } else {
                            false
                        }
                    } else {
                        converted_other.proper_angle.is_none()
                    }
                } else {
                    false
                }
            } else if let ElementOrder::Inf = converted_other.proper_order {
                false
            } else {
                (self.proper_fraction == converted_other.proper_fraction)
                    || (self
                        .proper_fraction
                        .expect("No proper fractions found for `self`.")
                        + converted_other
                            .proper_fraction
                            .expect("No proper fractions found for `converted_other`.")
                        == F::from(1u32))
            };
            if result {
                assert_eq!(
                    misc::calculate_hash(self),
                    misc::calculate_hash(other),
                    "{self} and {other} have unequal hashes."
                );
            }
            return result;
        }

        let result = approx::relative_eq!(
            geometry::get_positive_pole(&self.axis, thresh),
            geometry::get_positive_pole(&other.axis, thresh),
            epsilon = thresh,
            max_relative = thresh
        ) && if let ElementOrder::Inf = self.proper_order {
            if let ElementOrder::Inf = other.proper_order {
                if let Some(s_angle) = self.proper_angle {
                    if let Some(o_angle) = other.proper_angle {
                        approx::relative_eq!(
                            s_angle.abs(),
                            o_angle.abs(),
                            epsilon = thresh,
                            max_relative = thresh
                        )
                    } else {
                        false
                    }
                } else {
                    other.proper_angle.is_none()
                }
            } else {
                false
            }
        } else if let ElementOrder::Inf = other.proper_order {
            false
        } else {
            (self.proper_fraction == other.proper_fraction)
                || (self
                    .proper_fraction
                    .expect("No proper fractions found for `self`.")
                    + other
                        .proper_fraction
                        .expect("No proper fractions found for `other`.")
                    == F::from(1u64))
        };
        if result {
            assert_eq!(
                misc::calculate_hash(self),
                misc::calculate_hash(other),
                "{self} and {other} have unequal hashes."
            );
        }
        result
    }
}

impl Eq for SymmetryElement {}

impl Hash for SymmetryElement {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.spinrot.hash(state);
        let tr = self.contains_time_reversal();
        tr.hash(state);
        self.is_nonsr_proper(tr).hash(state);
        if self.is_nonsr_identity(tr) || self.is_nonsr_inversion_centre(tr) {
            true.hash(state);
        } else if self.kind == SymmetryElementKind::ImproperMirrorPlane(tr) {
            let c_self = self
                .convert_to_improper_kind(&SymmetryElementKind::ImproperInversionCentre(tr), false);
            let pole = geometry::get_positive_pole(&c_self.axis, c_self.threshold);
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
                cmp::min_by(
                    c_self.proper_fraction.expect("No proper fractions found."),
                    F::from(1u64) - c_self.proper_fraction.expect("No proper fractions found."),
                    |a, b| {
                        a.partial_cmp(b)
                            .unwrap_or_else(|| panic!("{a} and {b} cannot be compared."))
                    },
                )
                .hash(state);
            };
        } else {
            let pole = geometry::get_positive_pole(&self.axis, self.threshold);
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
                cmp::min_by(
                    self.proper_fraction.expect("No proper fractions found."),
                    F::from(1u64) - self.proper_fraction.expect("No proper fractions found."),
                    |a, b| {
                        a.partial_cmp(b)
                            .unwrap_or_else(|| panic!("{a} and {b} cannot be compared."))
                    },
                )
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
