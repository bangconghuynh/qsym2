use std::cmp;
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

/// An enum to classify the types of symmetry element.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
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

impl fmt::Display for SymmetryElementKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Proper => write!(f, "Proper"),
            Self::ImproperMirrorPlane => write!(f, "Improper (mirror-plane convention)"),
            Self::ImproperInversionCentre => write!(f, "Improper (inversion-centre convention)"),
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
    /// angle in the conventional range that puts the identity rotation at the centre
    /// of the range.
    #[builder(setter(custom), default = "self.calc_proper_angle()")]
    proper_angle: Option<f64>,

    /// The normalised axis of the symmetry element.
    #[builder(setter(custom))]
    pub axis: Vector3<f64>,

    /// The kind of the symmetry element.
    #[builder(default = "SymmetryElementKind::Proper")]
    pub kind: SymmetryElementKind,

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
                (f64::from(self
                    .proper_power
                    .expect("Proper power has not been set.")
                    .expect("No proper powers found."))
                    / (f64::from(*io)))
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

    /// Checks if the symmetry element is proper or not.
    ///
    /// # Returns
    ///
    /// A flag indicating if the symmetry element is proper.
    #[must_use]
    pub fn is_proper(&self) -> bool {
        self.kind == SymmetryElementKind::Proper
    }

    /// Checks if the symmetry element is an identity element.
    ///
    /// # Returns
    ///
    /// A flag indicating if this symmetry element is an identity element.
    #[must_use]
    pub fn is_identity(&self) -> bool {
        self.kind == SymmetryElementKind::Proper && self.proper_fraction == Some(F::from(1))
    }

    /// Checks if the symmetry element is an inversion centre.
    ///
    /// # Returns
    ///
    /// A flag indicating if this symmetry element is an inversion centre.
    #[must_use]
    pub fn is_inversion_centre(&self) -> bool {
        (self.kind == SymmetryElementKind::ImproperMirrorPlane
            && self.proper_fraction == Some(F::new(1u32, 2u32)))
            || (self.kind == SymmetryElementKind::ImproperInversionCentre
                && self.proper_fraction == Some(F::from(1)))
    }

    /// Checks if the symmetry element is a binary rotation axis.
    ///
    /// # Returns
    ///
    /// A flag indicating if this symmetry element is a binary rotation axis.
    #[must_use]
    pub fn is_binary_rotation_axis(&self) -> bool {
        self.kind == SymmetryElementKind::Proper && self.proper_fraction == Some(F::new(1u32, 2u32))
    }

    /// Checks if the symmetry element is a mirror plane.
    ///
    /// # Returns
    ///
    /// A flag indicating if this symmetry element is a mirror plane.
    #[must_use]
    pub fn is_mirror_plane(&self) -> bool {
        (matches!(self.kind, SymmetryElementKind::ImproperMirrorPlane)
            && self.proper_fraction == Some(F::from(1)))
            || (self.kind == SymmetryElementKind::ImproperInversionCentre
                && self.proper_fraction == Some(F::new(1u32, 2u32)))
    }

    /// Returns the standard symbol for this symmetry element, which does not
    /// classify certain improper rotation axes into inversion centres or mirror
    /// planes.
    ///
    /// # Returns
    ///
    /// The standard symbol for this symmetry element.
    #[must_use]
    pub fn get_standard_symbol(&self) -> String {
        let main_symbol: String = match self.kind {
            SymmetryElementKind::Proper => "C".to_owned(),
            SymmetryElementKind::ImproperMirrorPlane => {
                if self.proper_order != ElementOrder::Inf && self.proper_power == Some(1) {
                    "S".to_owned()
                } else {
                    "σC".to_owned()
                }
            }
            SymmetryElementKind::ImproperInversionCentre => {
                if self.proper_order != ElementOrder::Inf && self.proper_power == Some(1) {
                    "Ṡ".to_owned()
                } else {
                    "iC".to_owned()
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
        format!("{main_symbol}{}{proper_power}", self.proper_order)
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
            SymmetryElementKind::Proper => {
                if self.is_identity() {
                    ("E".to_owned(), false)
                } else {
                    ("C".to_owned(), true)
                }
            }
            SymmetryElementKind::ImproperMirrorPlane => {
                if self.is_mirror_plane() {
                    ("σ".to_owned(), false)
                } else if self.is_inversion_centre() {
                    ("i".to_owned(), false)
                } else if self.proper_order == ElementOrder::Inf || self.proper_power == Some(1) {
                    ("S".to_owned(), false)
                } else {
                    ("σC".to_owned(), true)
                }
            }
            SymmetryElementKind::ImproperInversionCentre => {
                if self.is_mirror_plane() {
                    ("σ".to_owned(), false)
                } else if self.is_inversion_centre() {
                    ("i".to_owned(), false)
                } else if self.proper_order == ElementOrder::Inf || self.proper_power == Some(1) {
                    ("Ṡ".to_owned(), false)
                } else {
                    ("iC".to_owned(), true)
                }
            }
        };

        let order_string: String =
            if !self.is_identity() && !self.is_inversion_centre() && !self.is_mirror_plane() {
                format!("{}", self.proper_order)
            } else {
                String::new()
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

        main_symbol
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
    /// * `improper_kind` - Reference to the required improper kind.
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
                .kind(improper_kind.clone())
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
                        .kind(improper_kind.clone())
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
                        .kind(improper_kind.clone())
                        .generator(self.generator)
                        .additional_superscript(self.additional_superscript.clone())
                        .additional_subscript(self.additional_subscript.clone())
                        .build()
                        .expect("Unable to construct a symmetry element.")
                }
            }
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
        if self.is_identity() || self.is_inversion_centre() {
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
    /// * their axes are either parallel or anti-parallel;
    /// * their proper rotation angles have equal absolute values.
    ///
    /// For improper elements, proper rotation angles are taken in the inversion
    /// centre convention.
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
                assert_eq!(misc::calculate_hash(self), misc::calculate_hash(other));
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
        } else if self.kind == SymmetryElementKind::ImproperMirrorPlane {
            let c_self = self
                .convert_to_improper_kind(&SymmetryElementKind::ImproperInversionCentre, false);
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

pub const ROT: SymmetryElementKind = SymmetryElementKind::Proper;
pub const SIG: SymmetryElementKind = SymmetryElementKind::ImproperMirrorPlane;
pub const INV: SymmetryElementKind = SymmetryElementKind::ImproperInversionCentre;
