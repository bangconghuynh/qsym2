use std::collections::hash_map::Entry::Vacant;
use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::fmt;

use anyhow::{self, ensure, format_err};
use derive_builder::Builder;
use indexmap::IndexSet;
use itertools::Itertools;
use log;
use nalgebra::{Point3, Vector3};
use rayon::prelude::*;

use crate::aux::atom::Atom;
use crate::aux::geometry::{self, Transform};
use crate::aux::molecule::Molecule;
use crate::rotsym::{self, RotationalSymmetry};
use crate::symmetry::symmetry_element::symmetry_operation::{
    sort_operations, SpecialSymmetryTransformation, SymmetryOperation,
};
use crate::symmetry::symmetry_element::{
    AntiunitaryKind, SymmetryElement, SymmetryElementKind, ROT, SIG, SO3, TRROT, TRSIG,
};
use crate::symmetry::symmetry_element_order::{ElementOrder, ORDER_1, ORDER_2};
use crate::symmetry::symmetry_symbols::deduce_sigma_symbol;

#[cfg(test)]
mod symmetry_core_tests;

#[cfg(test)]
mod symmetry_group_detection_tests;

#[derive(Debug)]
pub struct PointGroupDetectionError(pub String);

impl fmt::Display for PointGroupDetectionError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Point-group detection error: {}.", self.0)
    }
}

impl Error for PointGroupDetectionError {}

/// A struct for storing and managing information required for symmetry analysis.
#[derive(Clone, Builder, Debug)]
pub struct PreSymmetry {
    /// The molecule to be symmetry-analysed. This molecule will have been
    /// translated to put its centre of mass at the origin.
    #[builder(setter(custom))]
    pub molecule: Molecule,

    /// The rotational symmetry of [`Self::molecule`] based on its moments of
    /// inertia.
    #[builder(setter(skip), default = "self.calc_rotational_symmetry()")]
    pub rotational_symmetry: RotationalSymmetry,

    /// The groups of symmetry-equivalent atoms in [`Self::molecule`].
    #[builder(setter(skip), default = "self.calc_sea_groups()")]
    pub sea_groups: Vec<Vec<Atom>>,

    /// Threshold for relative comparisons of moments of inertia.
    #[builder(setter(custom))]
    pub moi_threshold: f64,

    /// Threshold for relative distance comparisons.
    #[builder(setter(skip), default = "self.get_dist_threshold()")]
    pub dist_threshold: f64,
}

impl PreSymmetryBuilder {
    /// Initialises the molecule to be symmetry-analysed.
    ///
    /// # Arguments
    ///
    /// * molecule - The molecule to be symmetry-analysed.
    /// * recentre - A flag indicating if the molecule shall be recentred.
    ///
    /// # Returns
    /// A mutable borrow of `[Self]`.
    pub fn molecule(&mut self, molecule: &Molecule, recentre: bool) -> &mut Self {
        if recentre {
            // The Symmetry struct now owns a recentred copy of `molecule`.
            self.molecule = Some(molecule.recentre());
        } else {
            self.molecule = Some(molecule.clone());
        }
        self
    }

    /// Initialises the threshold for moment-of-inertia comparisons.
    ///
    /// # Arguments
    ///
    /// * thresh - The threshold for moment-of-inertia comparisons.
    ///
    /// # Returns
    /// A mutable borrow of `[Self]`.
    pub fn moi_threshold(&mut self, thresh: f64) -> &mut Self {
        if thresh >= f64::EPSILON {
            self.moi_threshold = Some(thresh);
        } else {
            log::error!(
                "Threshold value {} is invalid. Threshold must be at least the machine epsilon.",
                thresh
            );
            self.moi_threshold = None;
        }
        self
    }

    fn calc_rotational_symmetry(&self) -> RotationalSymmetry {
        let com = self
            .molecule
            .as_ref()
            .expect("A molecule has not been set.")
            .calc_com();
        let inertia = self
            .molecule
            .as_ref()
            .expect("A molecule has not been set.")
            .calc_inertia_tensor(&com);
        approx::assert_relative_eq!(
            com,
            Point3::origin(),
            epsilon = self
                .molecule
                .as_ref()
                .expect("A molecule has not been set.")
                .threshold,
            max_relative = self
                .molecule
                .as_ref()
                .expect("A molecule has not been set.")
                .threshold
        );
        rotsym::calc_rotational_symmetry(
            &inertia,
            self.moi_threshold.expect("MoI threshold has not been set."),
        )
    }

    fn calc_sea_groups(&self) -> Vec<Vec<Atom>> {
        self.molecule
            .as_ref()
            .expect("A molecule has not been set.")
            .calc_sea_groups()
    }

    fn get_dist_threshold(&self) -> f64 {
        self.molecule
            .as_ref()
            .expect("A molecule has not been set.")
            .threshold
    }
}

impl PreSymmetry {
    /// Returns a builder to construct a new pre-symmetry struct.
    ///
    /// # Returns
    ///
    /// A builder to construct a new pre-symmetry struct.
    #[must_use]
    pub fn builder() -> PreSymmetryBuilder {
        PreSymmetryBuilder::default()
    }

    /// Checks for the existence of the proper symmetry element $`C_n`$  or $`\theta C_n`$ along
    /// `axis` in `[Self::molecule]`.
    ///
    /// Non-time-reversed elements are always preferred.
    ///
    /// # Arguments
    ///
    /// * `order` - The geometrical order $`n`$ of the rotation axis. Only finite
    /// orders are supported.
    /// * `axis` - The rotation axis.
    /// * `tr` - A flag indicating if time reversal should also be considered in case the
    /// non-time-reversed symmetry element does not exist.
    ///
    /// # Returns
    ///
    /// An [`Option`] containing the proper kind if the $`C_n`$ or $`\theta C_n`$ element exists in
    /// `[Self::molecule]`. If not, [`None`] is returned.
    #[allow(clippy::trivially_copy_pass_by_ref)]
    fn check_proper(
        &self,
        order: &ElementOrder,
        axis: &Vector3<f64>,
        tr: bool,
    ) -> Option<SymmetryElementKind> {
        assert_ne!(
            *order,
            ElementOrder::Inf,
            "This method does not work for infinite-order elements."
        );
        let angle = 2.0 * std::f64::consts::PI / order.to_float();
        let rotated_mol = self.molecule.rotate(angle, axis);
        if rotated_mol == self.molecule {
            Some(ROT)
        } else if tr {
            let tr_rotated_mol = rotated_mol.reverse_time();
            if tr_rotated_mol == self.molecule {
                Some(TRROT)
            } else {
                None
            }
        } else {
            None
        }
    }

    /// Checks for the existence of the improper symmetry element $`S_n`$, $`\dot{S}_n`$,
    /// $`\theta S_n`$, or $`\theta \dot{S}_n`$ along `axis` in `[Self::molecule]`.
    ///
    /// Non-time-reversed elements are always preferred.
    ///
    /// # Arguments
    ///
    /// * `order` - The geometrical order $`n`$ of the improper rotation axis. Only
    /// finite orders are supported.
    /// * `axis` - The rotation axis.
    /// * `kind` - The convention in which the improper element is defined. The time reversal
    /// property of this does not matter.
    /// * `tr` - A flag indicating if time reversal should also be considered in case the
    /// non-time-reversed symmetry element does not exist.
    ///
    /// # Returns
    ///
    /// An [`Option`] containing the improper kind if the $`S_n`$, $`\theta S_n`$, $`\theta S_n`$,
    /// or $`\theta \dot{S}_n`$ element exists in `[Self::molecule]`. If not, [`None`] is returned.
    #[allow(clippy::trivially_copy_pass_by_ref)]
    fn check_improper(
        &self,
        order: &ElementOrder,
        axis: &Vector3<f64>,
        kind: &SymmetryElementKind,
        tr: bool,
    ) -> Option<SymmetryElementKind> {
        assert_ne!(
            *order,
            ElementOrder::Inf,
            "This method does not work for infinite-order elements."
        );
        let angle = 2.0 * std::f64::consts::PI / order.to_float();
        let transformed_mol = self.molecule.improper_rotate(
            angle,
            axis,
            &kind.to_tr(false).try_into().unwrap_or_else(|err| {
                log::error!("Error detected: {err}.");
                panic!("Error detected: {err}.")
            }),
        );
        if transformed_mol == self.molecule {
            Some(kind.to_tr(false))
        } else if tr {
            let tr_transformed_mol = transformed_mol.reverse_time();
            if tr_transformed_mol == self.molecule {
                Some(kind.to_tr(true))
            } else {
                None
            }
        } else {
            None
        }
    }
}

/// A struct for storing and managing symmetry analysis results.
#[derive(Builder, Clone, Debug)]
pub struct Symmetry {
    /// The determined point group in Schönflies notation.
    #[builder(setter(skip, strip_option), default = "None")]
    pub group_name: Option<String>,

    /// The symmetry elements found.
    ///
    /// Each entry in the hash map is for one kind of symmetry elements: the key gives the kind,
    /// and the value is a hash map where each key gives the order and the corresponding value
    /// gives the [`HashSet`] of the elements with that order.
    ///
    /// Note that for improper elements, the mirror-plane convention is preferred.
    #[builder(setter(skip), default = "HashMap::new()")]
    pub elements: HashMap<SymmetryElementKind, HashMap<ElementOrder, IndexSet<SymmetryElement>>>,

    /// The symmetry generators found.
    ///
    /// Each entry in the hash map is for one kind of symmetry generators: the key gives the kind,
    /// and the value is a hash map where each key gives the order and the corresponding value
    /// gives the [`HashSet`] of the generators with that order.
    ///
    /// Note that for improper generatrors, the mirror-plane convention is preferred.
    #[builder(setter(skip), default = "HashMap::new()")]
    pub generators: HashMap<SymmetryElementKind, HashMap<ElementOrder, IndexSet<SymmetryElement>>>,
}

impl Symmetry {
    /// Returns a builder to construct a new symmetry struct.
    ///
    /// # Returns
    ///
    /// A builder to construct a new symmetry struct.
    #[must_use]
    fn builder() -> SymmetryBuilder {
        SymmetryBuilder::default()
    }

    /// Construct a new and empty symmetry struct.
    #[must_use]
    pub fn new() -> Self {
        Symmetry::builder()
            .build()
            .expect("Unable to construct a `Symmetry` structure.")
    }

    /// Performs point-group detection analysis.
    ///
    /// # Arguments
    ///
    /// * `presym` - A pre-symmetry-analysis structure containing the molecule and its rotational
    /// symmetry required for point-group detection.
    /// * `tr` - A flag indicating if time reversal should also be considered. A time-reversed
    /// symmetry element will only be considered if its non-time-reversed version turns out to be
    /// not a symmetry element.
    pub fn analyse(
        &mut self,
        presym: &PreSymmetry,
        tr: bool,
    ) -> Result<&mut Self, anyhow::Error> {
        log::debug!("Rotational symmetry found: {}", presym.rotational_symmetry);

        if tr {
            log::debug!("Antiunitary symmetry generated by time reversal will be considered.");
        };

        // Add the identity, which must always exist.
        let c1 = SymmetryElement::builder()
            .threshold(presym.dist_threshold)
            .proper_order(ORDER_1)
            .proper_power(1)
            .raw_axis(Vector3::new(0.0, 0.0, 1.0))
            .kind(ROT)
            .rotation_group(SO3)
            .build()
            .expect("Unable to construct the identity element.");
        self.add_proper(ORDER_1, c1.raw_axis(), false, presym.dist_threshold, false);

        // Identify all symmetry elements and generators
        match &presym.rotational_symmetry {
            RotationalSymmetry::Spherical => self.analyse_spherical(presym, tr)?,
            RotationalSymmetry::ProlateLinear => self.analyse_linear(presym, tr)?,
            RotationalSymmetry::OblatePlanar
            | RotationalSymmetry::OblateNonPlanar
            | RotationalSymmetry::ProlateNonLinear => self.analyse_symmetric(presym, tr)?,
            RotationalSymmetry::AsymmetricPlanar | RotationalSymmetry::AsymmetricNonPlanar => {
                self.analyse_asymmetric(presym, tr)?
            }
        }

        if tr {
            if self.get_elements(&TRROT).is_none()
                && self.get_elements(&TRSIG).is_none()
                && self.get_generators(&TRROT).is_none()
                && self.get_generators(&TRSIG).is_none()
            {
                log::debug!("Antiunitary symmetry requested, but so far only non-time-reversed elements found.");
                // Time-reversal requested, but the above analysis gives only non-time-reversed
                // elements, which means the system must also contain time reversal as a symmetry
                // operation. This implies that the group is a grey group.
                if presym.molecule == presym.molecule.reverse_time() {
                    log::debug!("Time reversal is a symmetry element. This is a grey group.");
                    // Add time-reversed copies of proper elements
                    self.elements.insert(
                        TRROT,
                        self.get_elements(&ROT)
                            .expect("No proper elements found.")
                            .iter()
                            .map(|(order, proper_elements)| {
                                let tr_proper_elements = proper_elements
                                    .iter()
                                    .map(|proper_element| {
                                        proper_element.to_tr(true)
                                        // let mut tr_proper_element = proper_element.clone();
                                        // tr_proper_element.kind = proper_element.kind.to_tr(true);
                                        // tr_proper_element
                                    })
                                    .collect::<IndexSet<_>>();
                                (*order, tr_proper_elements)
                            })
                            .collect::<HashMap<_, _>>(),
                    );
                    log::debug!("Time-reversed copies of all proper elements added.");

                    // Add the time-reversal element as a generator
                    self.add_proper(ORDER_1, c1.raw_axis(), true, presym.dist_threshold, true);

                    // Add time-reversed copies of improper elements, if any
                    if self.get_elements(&SIG).is_some() {
                        self.elements.insert(
                            TRSIG,
                            self.get_elements(&SIG)
                                .expect("No improper elements found.")
                                .iter()
                                .map(|(order, improper_elements)| {
                                    let tr_improper_elements = improper_elements
                                        .iter()
                                        .map(|improper_element| {
                                            improper_element.to_tr(true)
                                            // let mut tr_improper_element = improper_element.clone();
                                            // tr_improper_element.kind =
                                            //     improper_element.kind.to_tr(true);
                                            // tr_improper_element
                                        })
                                        .collect::<IndexSet<_>>();
                                    (*order, tr_improper_elements)
                                })
                                .collect::<HashMap<_, _>>(),
                        );
                        log::debug!("Time-reversed copies of all improper elements added.");
                    }

                    // Rename the group to include the antiunitary coset generated by time reversal
                    let unitary_group = self
                        .group_name
                        .as_ref()
                        .expect("No point groups found.")
                        .clone();
                    self.set_group_name(format!("{unitary_group} + θ·{unitary_group}"));
                } else {
                    log::debug!(
                        "Time reversal is not a symmetry element. This is an ordinary group."
                    );
                }
            } else {
                log::debug!(
                    "Antiunitary symmetry requested and some non-time-reversed elements found."
                );
                log::debug!(
                    "Time reversal is not a symmetry element. This is a black-and-white group."
                );
            }
        }
        Ok(self)
    }

    /// Adds a proper symmetry element to this struct.
    ///
    /// # Arguments
    ///
    /// * order - The order of the proper symmetry element.
    /// * axis - The axis of rotation of the proper symmetry element.
    /// * generator - A flag indicating if this element should be added as a generator.
    ///
    /// # Returns
    ///
    /// `true` if the specified element is not present and has just been added,
    /// `false` otherwise.
    fn add_proper(
        &mut self,
        order: ElementOrder,
        axis: &Vector3<f64>,
        generator: bool,
        threshold: f64,
        tr: bool,
    ) -> bool {
        let positive_axis = geometry::get_standard_positive_pole(axis, threshold).normalize();
        let proper_kind = if tr { TRROT } else { ROT };
        let element = SymmetryElement::builder()
            .threshold(threshold)
            .proper_order(order)
            .proper_power(1)
            .raw_axis(positive_axis)
            .kind(proper_kind)
            .rotation_group(SO3)
            .generator(generator)
            .build()
            .expect("Unable to construct a proper element.");
        let simplified_symbol = element.get_simplified_symbol();
        let full_symbol = element.get_full_symbol();
        let result = if generator {
            if let Vacant(proper_generators) = self.generators.entry(proper_kind.clone()) {
                proper_generators.insert(HashMap::from([(order, IndexSet::from([element]))]));
                true
            } else {
                let proper_generators =
                    self.generators.get_mut(&proper_kind).unwrap_or_else(|| {
                        panic!(
                            "{} generators not found.",
                            if tr { "Time-reversed proper" } else { "Proper" }
                        )
                    });

                if let Vacant(proper_generators_order) = proper_generators.entry(order) {
                    proper_generators_order.insert(IndexSet::from([element]));
                    true
                } else {
                    proper_generators
                        .get_mut(&order)
                        .unwrap_or_else(|| {
                            panic!(
                                "Proper generators {}C{order} not found.",
                                if tr { "θ" } else { "" }
                            )
                        })
                        .insert(element)
                }
            }
        } else if let Vacant(proper_elements) = self.elements.entry(proper_kind.clone()) {
            proper_elements.insert(HashMap::from([(order, IndexSet::from([element]))]));
            true
        } else {
            let proper_elements = self.elements.get_mut(&proper_kind).unwrap_or_else(|| {
                panic!(
                    "{} elements not found.",
                    if tr { "Time-reversed proper" } else { "Proper" }
                )
            });

            if let Vacant(e) = proper_elements.entry(order) {
                e.insert(IndexSet::from([element]));
                true
            } else {
                proper_elements
                    .get_mut(&order)
                    .unwrap_or_else(|| {
                        panic!(
                            "Proper elements {}C{order} not found.",
                            if tr { "θ" } else { "" }
                        )
                    })
                    .insert(element)
            }
        };

        let dest_str = if generator {
            "generator".to_owned()
        } else {
            "element".to_owned()
        };
        if result {
            log::debug!(
                "Proper rotation {} ({}): {} axis along ({:+.3}, {:+.3}, {:+.3}) added.",
                dest_str,
                simplified_symbol,
                full_symbol,
                positive_axis[0] + 0.0,
                positive_axis[1] + 0.0,
                positive_axis[2] + 0.0,
            );
        }
        result
    }

    /// Adds an improper symmetry element to this struct.
    ///
    /// The improper symmetry element can be defined in whichever convention
    /// specified by `kind`, but it will always be added in the mirror-plane
    /// convention.
    ///
    /// # Arguments
    ///
    /// * order - The order of the improper symmetry element in the convention
    ///     specified by `kind`.
    /// * axis - The axis of the improper symmetry element.
    /// * generator - A flag indicating if this element should be added as a generator.
    /// * kind - The convention in which the improper symmetry element is defined.
    /// * sigma - An optional additional string indicating the type of mirror
    ///     plane in the case the improper element is a mirror plane.
    ///
    /// # Returns
    ///
    /// `true` if the specified element is not present and has just been added,
    /// `false` otherwise.
    #[allow(clippy::too_many_lines)]
    fn add_improper(
        &mut self,
        order: ElementOrder,
        axis: &Vector3<f64>,
        generator: bool,
        kind: SymmetryElementKind,
        sigma: Option<String>,
        threshold: f64,
        tr: bool,
    ) -> bool {
        let positive_axis = geometry::get_standard_positive_pole(axis, threshold).normalize();
        let mirror_kind = if tr { TRSIG } else { SIG };
        let element = if let Some(sigma_str) = sigma {
            assert!(sigma_str == "d" || sigma_str == "v" || sigma_str == "h");
            let mut sym_ele = SymmetryElement::builder()
                .threshold(threshold)
                .proper_order(order)
                .proper_power(1)
                .raw_axis(positive_axis)
                .kind(kind.to_tr(tr))
                .rotation_group(SO3)
                .generator(generator)
                .build()
                .expect("Unable to construct an improper symmetry element.")
                .convert_to_improper_kind(&mirror_kind, false);
            if *sym_ele.raw_proper_order() == ElementOrder::Int(1) {
                sym_ele.additional_subscript = sigma_str;
            }
            sym_ele
        } else {
            SymmetryElement::builder()
                .threshold(threshold)
                .proper_order(order)
                .proper_power(1)
                .raw_axis(positive_axis)
                .kind(kind.to_tr(tr))
                .rotation_group(SO3)
                .generator(generator)
                .build()
                .expect("Unable to construct an improper symmetry element.")
                .convert_to_improper_kind(&mirror_kind, false)
        };
        let order = *element.raw_proper_order();
        let simplified_symbol = element.get_simplified_symbol();
        let full_symbol = element.get_full_symbol();
        let au = if tr {
            Some(AntiunitaryKind::TimeReversal)
        } else {
            None
        };
        let is_o3_mirror_plane = element.is_o3_mirror_plane(au);
        let is_o3_inversion_centre = element.is_o3_inversion_centre(au);
        let improper_kind = if tr { TRSIG } else { SIG };
        let result = if generator {
            if let Vacant(improper_generators) = self.generators.entry(improper_kind.clone()) {
                improper_generators.insert(HashMap::from([(order, IndexSet::from([element]))]));
                true
            } else {
                let improper_generators =
                    self.generators.get_mut(&improper_kind).unwrap_or_else(|| {
                        panic!(
                            "{} generators not found.",
                            if tr {
                                "Time-reversed improper"
                            } else {
                                "Improper"
                            }
                        )
                    });

                if let Vacant(e) = improper_generators.entry(order) {
                    e.insert(IndexSet::from([element]));
                    true
                } else {
                    improper_generators
                        .get_mut(&order)
                        .unwrap_or_else(|| {
                            panic!(
                                "Improper generators {}S{order} not found.",
                                if tr { "θ" } else { "" }
                            )
                        })
                        .insert(element)
                }
            }
        } else if let Vacant(improper_elements) = self.elements.entry(improper_kind.clone()) {
            improper_elements.insert(HashMap::from([(order, IndexSet::from([element]))]));
            true
        } else {
            let improper_elements = self.elements.get_mut(&improper_kind).unwrap_or_else(|| {
                panic!(
                    "{} elements not found.",
                    if tr {
                        "Time-reversed improper"
                    } else {
                        "Improper"
                    }
                )
            });

            if let Vacant(e) = improper_elements.entry(order) {
                e.insert(IndexSet::from([element]));
                true
            } else {
                improper_elements
                    .get_mut(&order)
                    .unwrap_or_else(|| {
                        panic!(
                            "Improper elements {}S{order} not found.",
                            if tr { "θ" } else { "" }
                        )
                    })
                    .insert(element)
            }
        };
        let dest_str = if generator {
            "generator".to_owned()
        } else {
            "element".to_owned()
        };
        if result {
            if is_o3_mirror_plane {
                log::debug!(
                    "Mirror plane {} ({}): {} axis along ({:+.3}, {:+.3}, {:+.3}) added.",
                    dest_str,
                    simplified_symbol,
                    full_symbol,
                    positive_axis[0] + 0.0,
                    positive_axis[1] + 0.0,
                    positive_axis[2] + 0.0,
                );
            } else if is_o3_inversion_centre {
                log::debug!(
                    "Inversion centre {} ({}): {} axis along ({:+.3}, {:+.3}, {:+.3}) added.",
                    dest_str,
                    simplified_symbol,
                    full_symbol,
                    positive_axis[0] + 0.0,
                    positive_axis[1] + 0.0,
                    positive_axis[2] + 0.0,
                );
            } else {
                log::debug!(
                    "Improper rotation {} ({}): {} axis along ({:+.3}, {:+.3}, {:+.3}) added.",
                    dest_str,
                    simplified_symbol,
                    full_symbol,
                    positive_axis[0] + 0.0,
                    positive_axis[1] + 0.0,
                    positive_axis[2] + 0.0,
                );
            }
        }
        result
    }

    /// Sets the name of the symmetry group.
    ///
    /// # Arguments
    ///
    /// * `name` - The name to be given to the symmetry group.
    fn set_group_name(&mut self, name: String) {
        self.group_name = Some(name);
        log::debug!(
            "Symmetry group determined: {}",
            self.group_name.as_ref().expect("No symmetry groups found.")
        );
    }

    /// Obtains elements of a particular kind.
    ///
    /// # Arguments
    ///
    /// * `kind` - An element kind to be obtained.
    ///
    /// # Returns
    ///
    /// An optional shared reference to the hash map of the required element kind.
    #[must_use]
    pub fn get_elements(
        &self,
        kind: &SymmetryElementKind,
    ) -> Option<&HashMap<ElementOrder, IndexSet<SymmetryElement>>> {
        self.elements.get(kind)
    }

    /// Obtains elements of a particular kind (mutable).
    ///
    /// # Arguments
    ///
    /// * `kind` - An element kind to be obtained.
    ///
    /// # Returns
    ///
    /// An optional exclusive reference to the hash map of the required element kind.
    #[must_use]
    pub fn get_elements_mut(
        &mut self,
        kind: &SymmetryElementKind,
    ) -> Option<&mut HashMap<ElementOrder, IndexSet<SymmetryElement>>> {
        self.elements.get_mut(kind)
    }

    /// Obtains generators of a particular kind.
    ///
    /// # Arguments
    ///
    /// * `kind` - A generator kind to be obtained.
    ///
    /// # Returns
    ///
    /// An optional shared reference to the hash map of the required generator kind.
    #[must_use]
    pub fn get_generators(
        &self,
        kind: &SymmetryElementKind,
    ) -> Option<&HashMap<ElementOrder, IndexSet<SymmetryElement>>> {
        self.generators.get(kind)
    }

    /// Obtains generators of a particular kind (mutable).
    ///
    /// # Arguments
    ///
    /// * `kind` - A generator kind to be obtained.
    ///
    /// # Returns
    ///
    /// An optional exclusive reference to the hash map of the required generator kind.
    #[must_use]
    pub fn get_generators_mut(
        &mut self,
        kind: &SymmetryElementKind,
    ) -> Option<&mut HashMap<ElementOrder, IndexSet<SymmetryElement>>> {
        self.generators.get_mut(kind)
    }

    /// Obtains mirror-plane elements by their type (`"h"`, `"v"`, `"d"`, or `""`), including both
    /// time-reversed and non-time-reversed variants.
    ///
    /// # Returns
    ///
    /// An option containing the set of the required mirror-plane element type, if exists. If not,
    /// then `None` is returned.
    #[must_use]
    pub fn get_sigma_elements(&self, sigma: &str) -> Option<HashSet<&SymmetryElement>> {
        self.get_improper(&ORDER_1)
            .map(|sigma_elements| {
                sigma_elements
                    .iter()
                    .filter_map(|ele| {
                        if ele.additional_subscript == sigma {
                            Some(*ele)
                        } else {
                            None
                        }
                    })
                    .collect::<HashSet<_>>()
            })
            .filter(|sigma_elements| !sigma_elements.is_empty())
    }

    /// Obtains mirror-plane generators by their type (`"h"`, `"v"`, `"d"`, or `""`), including both
    /// time-reversed and non-time-reversed variants.
    ///
    /// # Returns
    ///
    /// A set of the required mirror-plane generator type, if exists.
    #[must_use]
    pub fn get_sigma_generators(&self, sigma: &str) -> Option<HashSet<&SymmetryElement>> {
        let mut sigma_generators: HashSet<&SymmetryElement> = HashSet::new();
        if let Some(improper_generators) = self.get_generators(&SIG) {
            if let Some(sigmas) = improper_generators.get(&ORDER_1) {
                sigma_generators.extend(
                    sigmas
                        .iter()
                        .filter(|ele| ele.additional_subscript == sigma),
                );
            }
        }
        if let Some(tr_improper_generators) = self.get_generators(&TRSIG) {
            if let Some(sigmas) = tr_improper_generators.get(&ORDER_1) {
                sigma_generators.extend(
                    sigmas
                        .iter()
                        .filter(|ele| ele.additional_subscript == sigma),
                );
            }
        }
        if sigma_generators.is_empty() {
            None
        } else {
            Some(sigma_generators)
        }
    }

    /// Obtains the highest proper rotation order.
    ///
    /// # Returns
    ///
    /// The highest proper rotation order.
    #[must_use]
    pub fn get_max_proper_order(&self) -> ElementOrder {
        *self
            .get_generators(&ROT)
            .unwrap_or(&HashMap::new())
            .keys()
            .chain(self.get_elements(&ROT).unwrap_or(&HashMap::new()).keys())
            .chain(
                self.get_generators(&TRROT)
                    .unwrap_or(&HashMap::new())
                    .keys(),
            )
            .chain(self.get_elements(&TRROT).unwrap_or(&HashMap::new()).keys())
            .max()
            .expect("No highest proper rotation order could be obtained.")
    }

    /// Obtains all proper elements of a certain order (both time-reversed and non-time-reversed).
    ///
    /// # Arguments
    ///
    /// * `order` - The required order of elements.
    ///
    /// # Returns
    ///
    /// An optional hash set of proper elements of the required order. If no such elements exist,
    /// `None` will be returned.
    #[must_use]
    pub fn get_proper(&self, order: &ElementOrder) -> Option<HashSet<&SymmetryElement>> {
        let opt_proper_elements = self
            .get_elements(&ROT)
            .map(|proper_elements| proper_elements.get(order))
            .unwrap_or_default();
        let opt_tr_proper_elements = self
            .get_elements(&TRROT)
            .map(|tr_proper_elements| tr_proper_elements.get(order))
            .unwrap_or_default();

        match (opt_proper_elements, opt_tr_proper_elements) {
            (None, None) => None,
            (Some(proper_elements), None) => Some(proper_elements.iter().collect::<HashSet<_>>()),
            (None, Some(tr_proper_elements)) => {
                Some(tr_proper_elements.iter().collect::<HashSet<_>>())
            }
            (Some(proper_elements), Some(tr_proper_elements)) => Some(
                proper_elements
                    .iter()
                    .chain(tr_proper_elements.iter())
                    .collect::<HashSet<_>>(),
            ),
        }
    }

    /// Obtains all improper elements of a certain order (both time-reversed and non-time-reversed).
    ///
    /// # Arguments
    ///
    /// * `order` - The required order of elements.
    ///
    /// # Returns
    ///
    /// An optional hash set of improper elements of the required order. If no such elements exist,
    /// `None` will be returned.
    #[must_use]
    pub fn get_improper(&self, order: &ElementOrder) -> Option<HashSet<&SymmetryElement>> {
        let opt_improper_elements = self
            .get_elements(&SIG)
            .map(|improper_elements| improper_elements.get(order))
            .unwrap_or_default();
        let opt_tr_improper_elements = self
            .get_elements(&TRSIG)
            .map(|tr_improper_elements| tr_improper_elements.get(order))
            .unwrap_or_default();

        match (opt_improper_elements, opt_tr_improper_elements) {
            (None, None) => None,
            (Some(improper_elements), None) => {
                Some(improper_elements.iter().collect::<HashSet<_>>())
            }
            (None, Some(tr_improper_elements)) => {
                Some(tr_improper_elements.iter().collect::<HashSet<_>>())
            }
            (Some(improper_elements), Some(tr_improper_elements)) => Some(
                improper_elements
                    .iter()
                    .chain(tr_improper_elements.iter())
                    .collect::<HashSet<_>>(),
            ),
        }
    }

    /// Obtains a proper principal element, *i.e.* a time-reversed or non-time-reversed proper
    /// element with the highest order.
    ///
    /// If there are several such elements, the element to be returned will be randomly chosen but
    /// with any non-time-reversed ones prioritised.
    ///
    /// # Returns
    ///
    /// A proper principal element.
    ///
    /// # Panics
    ///
    /// Panics if no proper elements or generators can be found.
    #[must_use]
    pub fn get_proper_principal_element(&self) -> &SymmetryElement {
        let max_ord = self.get_max_proper_order();
        let principal_elements = self.get_proper(&max_ord).unwrap_or_else(|| {
            let opt_proper_generators = self
                .get_generators(&ROT)
                .map(|proper_generators| proper_generators.get(&max_ord))
                .unwrap_or_default();
            let opt_tr_proper_generators = self
                .get_elements(&TRROT)
                .map(|tr_proper_generators| tr_proper_generators.get(&max_ord))
                .unwrap_or_default();

            match (opt_proper_generators, opt_tr_proper_generators) {
                (None, None) => panic!("No proper elements found."),
                (Some(proper_generators), None) => proper_generators.iter().collect::<HashSet<_>>(),
                (None, Some(tr_proper_generators)) => {
                    tr_proper_generators.iter().collect::<HashSet<_>>()
                }
                (Some(proper_generators), Some(tr_proper_generators)) => proper_generators
                    .iter()
                    .chain(tr_proper_generators.iter())
                    .collect::<HashSet<_>>(),
            }
        });
        principal_elements
            .iter()
            .find(|ele| !ele.contains_time_reversal())
            .unwrap_or_else(|| {
                principal_elements
                    .iter()
                    .next()
                    .expect("No proper principal elements found.")
            })
    }

    /// Determines if this group is an infinite group.
    ///
    /// # Returns
    ///
    /// A boolean indicating if this group is an infinite group.
    #[must_use]
    pub fn is_infinite(&self) -> bool {
        self.get_max_proper_order() == ElementOrder::Inf
            || *self
                .get_generators(&ROT)
                .unwrap_or(&HashMap::new())
                .keys()
                .chain(self.get_elements(&ROT).unwrap_or(&HashMap::new()).keys())
                .chain(
                    self.get_generators(&TRROT)
                        .unwrap_or(&HashMap::new())
                        .keys(),
                )
                .chain(self.get_elements(&TRROT).unwrap_or(&HashMap::new()).keys())
                .max()
                .unwrap_or(&ElementOrder::Int(0))
                == ElementOrder::Inf
    }

    /// Returns the total number of symmetry elements (*NOT* symmetry operations). In
    /// infinite-order groups, this is the sum of the number of discrete symmetry elements and the
    /// number of discrete symmetry generators.
    pub fn n_elements(&self) -> usize {
        let n_elements = self.elements
            .values()
            .flat_map(|kind_elements| kind_elements.values())
            .flatten()
            .count();
        if self.is_infinite() {
            n_elements + self.generators
            .values()
            .flat_map(|kind_elements| kind_elements.values())
            .flatten()
            .count()
        } else {
            n_elements
        }
    }

    /// Generates all possible symmetry operations from the available symmetry elements.
    ///
    /// # Arguments
    ///
    /// * `infinite_order_to_finite` - A finite order to interpret infinite-order generators of
    /// infinite groups.
    ///
    /// # Returns
    ///
    /// A vector of generated symmetry operations.
    ///
    /// # Panics
    ///
    /// Panics if the group is infinite but `infinite_order_to_finite` is `None`, or if the finite
    /// order specified in `infinite_order_to_finite` is incompatible with the infinite group.
    #[allow(clippy::too_many_lines)]
    pub fn generate_all_operations(
        &self,
        infinite_order_to_finite: Option<u32>,
    ) -> Vec<SymmetryOperation> {
        let handles_infinite_group = if self.is_infinite() {
            assert!(infinite_order_to_finite.is_some());
            infinite_order_to_finite
        } else {
            None
        };

        if let Some(finite_order) = handles_infinite_group {
            let group_name = self.group_name.as_ref().expect("Group name not found.");
            if group_name.contains("O(3)") {
                if !matches!(finite_order, 2 | 4) {
                    log::error!(
                        "Finite order of `{finite_order}` is not yet supported for `{group_name}`."
                    );
                }
                assert!(
                    matches!(finite_order, 2 | 4),
                    "Finite order of `{finite_order}` is not yet supported for `{group_name}`."
                );
            }
        }

        let id_element = self
            .get_elements(&ROT)
            .unwrap_or(&HashMap::new())
            .get(&ORDER_1)
            .expect("No identity elements found.")
            .iter()
            .next()
            .expect("No identity elements found.")
            .clone();

        let id_operation = SymmetryOperation::builder()
            .generating_element(id_element)
            .power(1)
            .build()
            .expect("Unable to construct an identity operation.");

        let empty_elements: HashMap<ElementOrder, IndexSet<SymmetryElement>> = HashMap::new();

        // Finite proper operations
        let mut proper_orders = self
            .get_elements(&ROT)
            .unwrap_or(&empty_elements)
            .keys()
            .collect::<Vec<_>>();
        proper_orders.sort_by(|a, b| {
            a.partial_cmp(b)
                .unwrap_or_else(|| panic!("`{a}` and `{b}` cannot be compared."))
        });
        let proper_operations =
            proper_orders
                .iter()
                .fold(vec![id_operation], |mut acc, proper_order| {
                    self.get_elements(&ROT)
                        .unwrap_or(&empty_elements)
                        .get(proper_order)
                        .unwrap_or_else(|| panic!("Proper elements C{proper_order} not found."))
                        .iter()
                        .for_each(|proper_element| {
                            if let ElementOrder::Int(io) = proper_order {
                                acc.extend((1..*io).map(|power| {
                                    SymmetryOperation::builder()
                                        .generating_element(proper_element.clone())
                                        .power(power.try_into().unwrap_or_else(|_| {
                                            panic!("Unable to convert `{power}` to `i32`.")
                                        }))
                                        .build()
                                        .expect("Unable to construct a symmetry operation.")
                                }));
                            }
                        });
                    acc
                });

        // Finite proper operations from generators
        let proper_operations_from_generators = if let Some(fin_ord) = handles_infinite_group {
            self.get_generators(&ROT)
                .unwrap_or(&empty_elements)
                .par_iter()
                .fold(std::vec::Vec::new, |mut acc, (order, proper_generators)| {
                    for proper_generator in proper_generators.iter() {
                        let finite_order = match order {
                            ElementOrder::Int(io) => *io,
                            ElementOrder::Inf => fin_ord,
                        };
                        let finite_proper_element = SymmetryElement::builder()
                            .threshold(proper_generator.threshold())
                            .proper_order(ElementOrder::Int(finite_order))
                            .proper_power(1)
                            .raw_axis(proper_generator.raw_axis().clone())
                            .kind(proper_generator.kind().clone())
                            .rotation_group(proper_generator.rotation_group().clone())
                            .additional_superscript(proper_generator.additional_superscript.clone())
                            .additional_subscript(proper_generator.additional_subscript.clone())
                            .build()
                            .expect("Unable to construct a symmetry element.");
                        acc.extend((1..finite_order).map(|power| {
                            SymmetryOperation::builder()
                                .generating_element(finite_proper_element.clone())
                                .power(power.try_into().unwrap_or_else(|_| {
                                    panic!("Unable to convert `{power}` to `i32`.")
                                }))
                                .build()
                                .expect("Unable to construct a symmetry operation.")
                        }));
                    }
                    acc
                })
                .reduce(std::vec::Vec::new, |mut acc, vec| {
                    acc.extend(vec);
                    acc
                })
        } else {
            vec![]
        };

        // Finite time-reversed proper operations
        let mut tr_proper_orders = self
            .get_elements(&TRROT)
            .unwrap_or(&empty_elements)
            .keys()
            .collect::<Vec<_>>();
        tr_proper_orders.sort_by(|a, b| {
            a.partial_cmp(b)
                .unwrap_or_else(|| panic!("`{a}` and `{b}` cannot be compared."))
        });
        let tr_proper_operations =
            tr_proper_orders
                .iter()
                .fold(vec![], |mut acc, tr_proper_order| {
                    self.get_elements(&TRROT)
                        .unwrap_or(&empty_elements)
                        .get(tr_proper_order)
                        .unwrap_or_else(|| {
                            panic!("Proper elements θ·C{tr_proper_order} not found.")
                        })
                        .iter()
                        .for_each(|tr_proper_element| {
                            if let ElementOrder::Int(io) = tr_proper_order {
                                acc.extend((1..(2 * *io)).step_by(2).map(|power| {
                                    SymmetryOperation::builder()
                                        .generating_element(tr_proper_element.clone())
                                        .power(power.try_into().unwrap_or_else(|_| {
                                            panic!("Unable to convert `{power}` to `i32`.")
                                        }))
                                        .build()
                                        .expect("Unable to construct a symmetry operation.")
                                }));
                            }
                        });
                    acc
                });

        // Finite time-reversed proper operations from generators
        let tr_proper_operations_from_generators = if let Some(fin_ord) = handles_infinite_group {
            self.get_generators(&TRROT)
                .unwrap_or(&empty_elements)
                .par_iter()
                .fold(
                    std::vec::Vec::new,
                    |mut acc, (order, tr_proper_generators)| {
                        for tr_proper_generator in tr_proper_generators.iter() {
                            let finite_order = match order {
                                ElementOrder::Int(io) => *io,
                                ElementOrder::Inf => fin_ord,
                            };
                            let finite_tr_proper_element = SymmetryElement::builder()
                                .threshold(tr_proper_generator.threshold())
                                .proper_order(ElementOrder::Int(finite_order))
                                .proper_power(1)
                                .raw_axis(tr_proper_generator.raw_axis().clone())
                                .kind(tr_proper_generator.kind().clone())
                                .rotation_group(tr_proper_generator.rotation_group().clone())
                                .additional_superscript(
                                    tr_proper_generator.additional_superscript.clone(),
                                )
                                .additional_subscript(
                                    tr_proper_generator.additional_subscript.clone(),
                                )
                                .build()
                                .expect("Unable to construct a symmetry element.");
                            acc.extend((1..finite_order).map(|power| {
                                SymmetryOperation::builder()
                                    .generating_element(finite_tr_proper_element.clone())
                                    .power(power.try_into().unwrap_or_else(|_| {
                                        panic!("Unable to convert `{power}` to `i32`.")
                                    }))
                                    .build()
                                    .expect("Unable to construct a symmetry operation.")
                            }));
                        }
                        acc
                    },
                )
                .reduce(std::vec::Vec::new, |mut acc, vec| {
                    acc.extend(vec);
                    acc
                })
        } else {
            vec![]
        };

        // Finite improper operations
        let mut improper_orders = self
            .get_elements(&SIG)
            .unwrap_or(&empty_elements)
            .keys()
            .collect::<Vec<_>>();
        improper_orders.sort_by(|a, b| {
            a.partial_cmp(b)
                .unwrap_or_else(|| panic!("`{a}` and `{b}` cannot be compared."))
        });
        let improper_operations = improper_orders
            .iter()
            .fold(vec![], |mut acc, improper_order| {
                self.get_elements(&SIG)
                    .unwrap_or(&empty_elements)
                    .get(improper_order)
                    .unwrap_or_else(|| panic!("Improper elements S{improper_order} not found."))
                    .iter()
                    .for_each(|improper_element| {
                        if let ElementOrder::Int(io) = improper_order {
                            acc.extend((1..(2 * *io)).step_by(2).map(|power| {
                                SymmetryOperation::builder()
                                    .generating_element(improper_element.clone())
                                    .power(power.try_into().unwrap_or_else(|_| {
                                        panic!("Unable to convert `{power}` to `i32`.")
                                    }))
                                    .build()
                                    .expect("Unable to construct a symmetry operation.")
                            }));
                        }
                    });
                acc
            });

        // Finite improper operations from generators
        let improper_operations_from_generators = if let Some(fin_ord) = handles_infinite_group {
            self.get_generators(&SIG)
                .unwrap_or(&empty_elements)
                .par_iter()
                .fold(
                    std::vec::Vec::new,
                    |mut acc, (order, improper_generators)| {
                        for improper_generator in improper_generators.iter() {
                            let finite_order = match order {
                                ElementOrder::Int(io) => *io,
                                ElementOrder::Inf => fin_ord,
                            };
                            let finite_improper_element = SymmetryElement::builder()
                                .threshold(improper_generator.threshold())
                                .proper_order(ElementOrder::Int(finite_order))
                                .proper_power(1)
                                .raw_axis(improper_generator.raw_axis().clone())
                                .kind(improper_generator.kind().clone())
                                .rotation_group(improper_generator.rotation_group().clone())
                                .additional_superscript(
                                    improper_generator.additional_superscript.clone(),
                                )
                                .additional_subscript(
                                    improper_generator.additional_subscript.clone(),
                                )
                                .build()
                                .expect("Unable to construct a symmetry element.");
                            acc.extend((1..(2 * finite_order)).step_by(2).map(|power| {
                                SymmetryOperation::builder()
                                    .generating_element(finite_improper_element.clone())
                                    .power(power.try_into().unwrap_or_else(|_| {
                                        panic!("Unable to convert `{power}` to `i32`.")
                                    }))
                                    .build()
                                    .expect("Unable to construct a symmetry operation.")
                            }));
                        }
                        acc
                    },
                )
                .reduce(std::vec::Vec::new, |mut acc, vec| {
                    acc.extend(vec);
                    acc
                })
        } else {
            vec![]
        };

        // Finite time-reversed improper operations
        let mut tr_improper_orders = self
            .get_elements(&TRSIG)
            .unwrap_or(&empty_elements)
            .keys()
            .collect::<Vec<_>>();
        tr_improper_orders.sort_by(|a, b| {
            a.partial_cmp(b)
                .unwrap_or_else(|| panic!("`{a}` and `{b}` cannot be compared."))
        });
        let tr_improper_operations =
            tr_improper_orders
                .iter()
                .fold(vec![], |mut acc, tr_improper_order| {
                    self.get_elements(&TRSIG)
                        .unwrap_or(&empty_elements)
                        .get(tr_improper_order)
                        .unwrap_or_else(|| {
                            panic!("Improper elements θ·S{tr_improper_order} not found.")
                        })
                        .iter()
                        .for_each(|tr_improper_element| {
                            if let ElementOrder::Int(io) = tr_improper_order {
                                acc.extend((1..(2 * *io)).step_by(2).map(|power| {
                                    SymmetryOperation::builder()
                                        .generating_element(tr_improper_element.clone())
                                        .power(power.try_into().unwrap_or_else(|_| {
                                            panic!("Unable to convert `{power}` to `i32`.")
                                        }))
                                        .build()
                                        .expect("Unable to construct a symmetry operation.")
                                }));
                            }
                        });
                    acc
                });

        // Finite time-reversed improper operations from generators
        let tr_improper_operations_from_generators = if let Some(fin_ord) = handles_infinite_group {
            self.get_generators(&TRSIG)
                .unwrap_or(&empty_elements)
                .par_iter()
                .fold(
                    std::vec::Vec::new,
                    |mut acc, (order, tr_improper_generators)| {
                        for tr_improper_generator in tr_improper_generators.iter() {
                            let finite_order = match order {
                                ElementOrder::Int(io) => *io,
                                ElementOrder::Inf => fin_ord,
                            };
                            let finite_tr_improper_element = SymmetryElement::builder()
                                .threshold(tr_improper_generator.threshold())
                                .proper_order(ElementOrder::Int(finite_order))
                                .proper_power(1)
                                .raw_axis(tr_improper_generator.raw_axis().clone())
                                .kind(tr_improper_generator.kind().clone())
                                .rotation_group(tr_improper_generator.rotation_group().clone())
                                .additional_superscript(
                                    tr_improper_generator.additional_superscript.clone(),
                                )
                                .additional_subscript(
                                    tr_improper_generator.additional_subscript.clone(),
                                )
                                .build()
                                .expect("Unable to construct a symmetry element.");
                            acc.extend((1..(2 * finite_order)).step_by(2).map(|power| {
                                SymmetryOperation::builder()
                                    .generating_element(finite_tr_improper_element.clone())
                                    .power(power.try_into().unwrap_or_else(|_| {
                                        panic!("Unable to convert `{power}` to `i32`.")
                                    }))
                                    .build()
                                    .expect("Unable to construct a symmetry operation.")
                            }));
                        }
                        acc
                    },
                )
                .reduce(std::vec::Vec::new, |mut acc, vec| {
                    acc.extend(vec);
                    acc
                })
        } else {
            vec![]
        };

        let operations: IndexSet<_> = if handles_infinite_group.is_none() {
            proper_operations
                .into_iter()
                .chain(proper_operations_from_generators)
                .chain(improper_operations)
                .chain(improper_operations_from_generators)
                .chain(tr_proper_operations)
                .chain(tr_proper_operations_from_generators)
                .chain(tr_improper_operations)
                .chain(tr_improper_operations_from_generators)
                .collect()
        } else {
            // Fulfil group closure
            log::debug!("Fulfilling closure for a finite subgroup of an infinite group...");
            let mut existing_operations: IndexSet<_> = proper_operations
                .into_iter()
                .chain(proper_operations_from_generators)
                .chain(improper_operations)
                .chain(improper_operations_from_generators)
                .chain(tr_proper_operations)
                .chain(tr_proper_operations_from_generators)
                .chain(tr_improper_operations)
                .chain(tr_improper_operations_from_generators)
                .collect();
            let mut extra_operations = HashSet::<SymmetryOperation>::new();
            let mut npasses = 0;
            let mut nstable = 0;

            let principal_element = self.get_proper_principal_element();
            while nstable < 2 || npasses == 0 {
                let n_extra_operations = extra_operations.len();
                existing_operations.extend(extra_operations);

                npasses += 1;
                log::debug!(
                    "Generating all group elements: {} pass{}, {} element{} (of which {} {} new)",
                    npasses,
                    {
                        if npasses > 1 {
                            "es"
                        } else {
                            ""
                        }
                    }
                    .to_string(),
                    existing_operations.len(),
                    {
                        if existing_operations.len() > 1 {
                            "s"
                        } else {
                            ""
                        }
                    }
                    .to_string(),
                    n_extra_operations,
                    {
                        if n_extra_operations == 1 {
                            "is"
                        } else {
                            "are"
                        }
                    }
                    .to_string(),
                );

                extra_operations = existing_operations
                    .iter()
                    .combinations_with_replacement(2)
                    .par_bridge()
                    .filter_map(|op_pairs| {
                        let op_i_ref = op_pairs[0];
                        let op_j_ref = op_pairs[1];
                        let op_k = op_i_ref * op_j_ref;
                        if existing_operations.contains(&op_k) {
                            None
                        } else if op_k.is_proper() {
                            Some(op_k)
                        } else if op_k.is_spatial_reflection()
                            && op_k.generating_element.additional_subscript.is_empty()
                        {
                            if let Some(sigma_symbol) = deduce_sigma_symbol(
                                &op_k.generating_element.raw_axis(),
                                principal_element,
                                op_k.generating_element.threshold(),
                                false,
                            ) {
                                let mut op_k_sym = op_k.convert_to_improper_kind(&SIG);
                                op_k_sym.generating_element.additional_subscript = sigma_symbol;
                                Some(op_k_sym)
                            } else {
                                Some(op_k.convert_to_improper_kind(&SIG))
                            }
                        } else {
                            Some(op_k.convert_to_improper_kind(&SIG))
                        }
                    })
                    .collect();
                if extra_operations.is_empty() {
                    nstable += 1;
                } else {
                    nstable = 0;
                }
            }

            assert_eq!(extra_operations.len(), 0);
            log::debug!(
                "Group closure reached with {} elements.",
                existing_operations.len()
            );
            existing_operations
        };

        let mut sorted_operations: Vec<SymmetryOperation> = operations.into_iter().collect();
        sort_operations(&mut sorted_operations);
        sorted_operations
    }
}

impl Default for Symmetry {
    fn default() -> Self {
        Self::new()
    }
}

/// Locates all proper rotation elements present in [`PreSymmetry::molecule`]
///
/// # Arguments
///
/// * `presym` - A pre-symmetry-analysis struct containing information about
/// the molecular system.
/// * `sym` - A symmetry struct to store the proper rotation elements found.
/// * `asymmetric` - If `true`, the search assumes that the group is one of the
/// Abelian point groups for which the highest possible rotation order is $`2`$
/// and there can be at most three $`C_2`$ axes.
/// * `tr` - A flag indicating if time reversal should also be considered.
#[allow(clippy::too_many_lines)]
fn _search_proper_rotations(
    presym: &PreSymmetry,
    sym: &mut Symmetry,
    asymmetric: bool,
    tr: bool,
) -> Result<(), anyhow::Error> {
    log::debug!("==============================");
    log::debug!("Proper rotation search begins.");
    log::debug!("==============================");
    let mut linear_sea_groups: Vec<&Vec<Atom>> = vec![];
    let mut count_c2: usize = 0;
    log::debug!("++++++++++++++++++++++++++");
    log::debug!("SEA group analysis begins.");
    log::debug!("++++++++++++++++++++++++++");
    for sea_group in &presym.sea_groups {
        if asymmetric && count_c2 == 3 {
            break;
        }
        let k_sea = sea_group.len();
        match k_sea {
            1 => {
                continue;
            }
            2 => {
                log::debug!("A linear SEA set detected: {:?}.", sea_group);
                linear_sea_groups.push(sea_group);
            }
            _ => {
                let sea_mol = Molecule::from_atoms(sea_group, presym.dist_threshold);
                let (sea_mois, sea_axes) = sea_mol.calc_moi();
                // Search for high-order rotation axes
                if approx::relative_eq!(
                    sea_mois[0] + sea_mois[1],
                    sea_mois[2],
                    epsilon = presym.moi_threshold,
                    max_relative = presym.moi_threshold,
                ) {
                    // Planar SEA
                    let k_fac_range: Vec<_> = if approx::relative_eq!(
                        sea_mois[0],
                        sea_mois[1],
                        epsilon = presym.moi_threshold,
                        max_relative = presym.moi_threshold,
                    ) {
                        // Regular k-sided polygon
                        log::debug!(
                            "A regular {}-sided polygon SEA set detected: {:?}.",
                            k_sea,
                            sea_group
                        );
                        let mut divisors = divisors::get_divisors(k_sea);
                        divisors.push(k_sea);
                        divisors
                    } else {
                        // Irregular k-sided polygon
                        log::debug!(
                            "An irregular {}-sided polygon SEA set detected: {:?}.",
                            k_sea,
                            sea_group
                        );
                        divisors::get_divisors(k_sea)
                    };
                    for k_fac in &k_fac_range {
                        if let Some(proper_kind) =
                            presym.check_proper(
                                &ElementOrder::Int((*k_fac).try_into().unwrap_or_else(|_| {
                                    panic!("Unable to convert {k_fac} to `u32`.")
                                })),
                                &sea_axes[2],
                                tr,
                            )
                        {
                            match *k_fac {
                                2 => {
                                    count_c2 += usize::from(sym.add_proper(
                                        ElementOrder::Int((*k_fac).try_into().unwrap_or_else(
                                            |_| panic!("Unable to convert {k_fac} to `u32`."),
                                        )),
                                        &sea_axes[2],
                                        false,
                                        presym.dist_threshold,
                                        proper_kind.contains_time_reversal(),
                                    ));
                                }
                                _ => {
                                    sym.add_proper(
                                        ElementOrder::Int((*k_fac).try_into().unwrap_or_else(
                                            |_| panic!("Unable to convert {k_fac} to `u32`."),
                                        )),
                                        &sea_axes[2],
                                        false,
                                        presym.dist_threshold,
                                        proper_kind.contains_time_reversal(),
                                    );
                                }
                            }
                        }
                    }
                } else {
                    // Polyhedral SEA
                    if approx::relative_eq!(
                        sea_mois[1],
                        sea_mois[2],
                        epsilon = presym.moi_threshold,
                        max_relative = presym.moi_threshold,
                    ) {
                        // The number of atoms in this SEA group must be even.
                        ensure!(
                            k_sea % 2 == 0,
                            "Unexpected odd number of atoms in this SEA group."
                        );
                        if approx::relative_eq!(
                            sea_mois[0],
                            sea_mois[1],
                            epsilon = presym.moi_threshold,
                            max_relative = presym.moi_threshold,
                        ) {
                            // Spherical top SEA
                            log::debug!("A spherical top SEA set detected: {:?}", sea_group);
                            let sea_presym = PreSymmetry::builder()
                                .moi_threshold(presym.moi_threshold)
                                .molecule(&sea_mol, true)
                                .build()
                                .expect("Unable to construct a `PreSymmetry` structure.");
                            let mut sea_sym = Symmetry::builder()
                                .build()
                                .expect("Unable to construct a default `Symmetry` structure.");
                            log::debug!("-----------------------------------------------");
                            log::debug!("Symmetry analysis for spherical top SEA begins.");
                            log::debug!("-----------------------------------------------");
                            sea_sym.analyse(&sea_presym, tr)?;
                            log::debug!("---------------------------------------------");
                            log::debug!("Symmetry analysis for spherical top SEA ends.");
                            log::debug!("---------------------------------------------");
                            for (order, proper_elements) in sea_sym
                                .get_elements(&ROT)
                                .unwrap_or(&HashMap::new())
                                .iter()
                                .chain(
                                    sea_sym
                                        .get_elements(&TRROT)
                                        .unwrap_or(&HashMap::new())
                                        .iter(),
                                )
                            {
                                for proper_element in proper_elements {
                                    if let Some(proper_kind) =
                                        presym.check_proper(order, &proper_element.raw_axis(), tr)
                                    {
                                        sym.add_proper(
                                            *order,
                                            proper_element.raw_axis(),
                                            false,
                                            presym.dist_threshold,
                                            proper_kind.contains_time_reversal(),
                                        );
                                    }
                                }
                            }
                        } else {
                            // Prolate symmetric top
                            log::debug!("A prolate symmetric top SEA set detected.");
                            for k_fac in divisors::get_divisors(k_sea / 2)
                                .iter()
                                .chain(vec![k_sea / 2].iter())
                            {
                                let k_fac_order =
                                    ElementOrder::Int((*k_fac).try_into().unwrap_or_else(|_| {
                                        panic!("Unable to convert {k_fac} to u32.")
                                    }));
                                if let Some(proper_kind) =
                                    presym.check_proper(&k_fac_order, &sea_axes[0], tr)
                                {
                                    if *k_fac == 2 {
                                        count_c2 += usize::from(sym.add_proper(
                                            k_fac_order,
                                            &sea_axes[0],
                                            false,
                                            presym.dist_threshold,
                                            proper_kind.contains_time_reversal(),
                                        ));
                                    } else {
                                        sym.add_proper(
                                            k_fac_order,
                                            &sea_axes[0],
                                            false,
                                            presym.dist_threshold,
                                            proper_kind.contains_time_reversal(),
                                        );
                                    }
                                }
                            }
                        }
                    } else if approx::relative_eq!(
                        sea_mois[0],
                        sea_mois[1],
                        epsilon = presym.moi_threshold,
                        max_relative = presym.moi_threshold,
                    ) {
                        // Oblate symmetry top
                        log::debug!("An oblate symmetric top SEA set detected.");
                        ensure!(
                            k_sea % 2 == 0,
                            "Unexpected odd number of atoms in this SEA group."
                        );
                        for k_fac in divisors::get_divisors(k_sea / 2)
                            .iter()
                            .chain(vec![k_sea / 2].iter())
                        {
                            let k_fac_order =
                                ElementOrder::Int((*k_fac).try_into().map_err(|_| {
                                    format_err!("Unable to convert `{k_fac}` to `u32`.")
                                })?);
                            if let Some(proper_kind) =
                                presym.check_proper(&k_fac_order, &sea_axes[2], tr)
                            {
                                if *k_fac == 2 {
                                    count_c2 += usize::from(sym.add_proper(
                                        k_fac_order,
                                        &sea_axes[2],
                                        false,
                                        presym.dist_threshold,
                                        proper_kind.contains_time_reversal(),
                                    ));
                                } else {
                                    sym.add_proper(
                                        k_fac_order,
                                        &sea_axes[2],
                                        false,
                                        presym.dist_threshold,
                                        proper_kind.contains_time_reversal(),
                                    );
                                }
                            }
                        }
                    } else {
                        // Asymmetric top
                        log::debug!("An asymmetric top SEA set detected.");
                        for sea_axis in &sea_axes {
                            if let Some(proper_kind) = presym.check_proper(&ORDER_2, sea_axis, tr) {
                                count_c2 += usize::from(sym.add_proper(
                                    ORDER_2,
                                    sea_axis,
                                    false,
                                    presym.dist_threshold,
                                    proper_kind.contains_time_reversal(),
                                ));
                            }
                        }
                    }
                }
            }
        } // end match k_sea

        // Search for any remaining C2 axes
        for atom2s in sea_group.iter().combinations(2) {
            if asymmetric && count_c2 == 3 {
                break;
            }
            let atom_i_pos = atom2s[0].coordinates;
            let atom_j_pos = atom2s[1].coordinates;

            // Case B: C2 might cross through any two atoms
            if let Some(proper_kind) = presym.check_proper(&ORDER_2, &atom_i_pos.coords, tr) {
                log::debug!("Case B: C2 crosses through any two atoms.");
                count_c2 += usize::from(sym.add_proper(
                    ORDER_2,
                    &atom_i_pos.coords,
                    false,
                    presym.dist_threshold,
                    proper_kind.contains_time_reversal(),
                ));
            }

            // Case A: C2 might cross through the midpoint of two atoms
            let midvec = 0.5 * (atom_i_pos.coords + atom_j_pos.coords);
            let c2_check = presym.check_proper(&ORDER_2, &midvec, tr);
            if midvec.norm() > presym.dist_threshold && c2_check.is_some() {
                log::debug!("Case A: C2 crosses through the midpoint of two atoms.");
                count_c2 += usize::from(
                    sym.add_proper(
                        ORDER_2,
                        &midvec,
                        false,
                        presym.dist_threshold,
                        c2_check
                            .expect("Expected C2 not found.")
                            .contains_time_reversal(),
                    ),
                );
            } else if let Some(electric_atoms) = &presym.molecule.electric_atoms {
                let com = presym.molecule.calc_com();
                let e_vector = electric_atoms[0].coordinates - com;
                if let Some(proper_kind) = presym.check_proper(&ORDER_2, &e_vector, tr) {
                    count_c2 += usize::from(sym.add_proper(
                        ORDER_2,
                        &e_vector,
                        false,
                        presym.dist_threshold,
                        proper_kind.contains_time_reversal(),
                    ));
                }
            }
        }
    } // end for sea_group in presym.sea_groups.iter()
    log::debug!("++++++++++++++++++++++++");
    log::debug!("SEA group analysis ends.");
    log::debug!("++++++++++++++++++++++++");

    // Search for any remaining C2 axes.
    if asymmetric && count_c2 == 3 {
    } else {
        // Case C: Molecules with two or more sets of non-parallel linear diatomic SEA groups
        if linear_sea_groups.len() >= 2 {
            log::debug!("Case C: Molecules with two or more sets of non-parallel linear diatomic SEA groups.");
            let normal_option = linear_sea_groups.iter().combinations(2).find_map(|pair| {
                let vec_0 = pair[0][1].coordinates - pair[0][0].coordinates;
                let vec_1 = pair[1][1].coordinates - pair[1][0].coordinates;
                let trial_normal = vec_0.cross(&vec_1);
                if trial_normal.norm() > presym.dist_threshold {
                    Some(trial_normal)
                } else {
                    None
                }
            });
            if let Some(normal) = normal_option {
                if let Some(proper_kind) = presym.check_proper(&ORDER_2, &normal, tr) {
                    sym.add_proper(
                        ORDER_2,
                        &normal,
                        false,
                        presym.dist_threshold,
                        proper_kind.contains_time_reversal(),
                    );
                }
            }
        }
    }
    log::debug!("============================");
    log::debug!("Proper rotation search ends.");
    log::debug!("============================");

    Ok(())
}

mod symmetry_core_asymmetric;
mod symmetry_core_linear;
mod symmetry_core_spherical;
mod symmetry_core_symmetric;
