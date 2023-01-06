use std::collections::hash_map::Entry::Vacant;
use std::collections::{HashMap, HashSet};

use derive_builder::Builder;
use itertools::Itertools;
use log;
use nalgebra::{Point3, Vector3};

use crate::aux::atom::Atom;
use crate::aux::geometry::{self, Transform};
use crate::aux::molecule::Molecule;
use crate::rotsym::{self, RotationalSymmetry};
use crate::symmetry::symmetry_element::{
    SymmetryElement, SymmetryElementKind, ROT, SIG, TRROT, TRSIG,
};
use crate::symmetry::symmetry_element_order::{ElementOrder, ORDER_1, ORDER_2};

#[cfg(test)]
mod symmetry_core_tests;

#[cfg(test)]
mod point_group_detection_tests;

/// A struct for storing and managing information required for symmetry analysis.
#[derive(Builder)]
pub struct PreSymmetry {
    /// The molecule to be symmetry-analysed. This molecule will have bee
    /// translated to put its centre of mass at the origin.
    #[builder(setter(custom))]
    molecule: Molecule,

    /// The rotational symmetry of [`Self::molecule`] based on its moments of
    /// inertia.
    #[builder(setter(skip), default = "self.calc_rotational_symmetry()")]
    rotational_symmetry: RotationalSymmetry,

    /// The groups of symmetry-equivalent atoms in [`Self::molecule`].
    #[builder(setter(skip), default = "self.calc_sea_groups()")]
    sea_groups: Vec<Vec<Atom>>,

    /// Threshold for relative comparisons of moments of inertia.
    #[builder(setter(custom))]
    moi_threshold: f64,

    /// Threshold for relative distance comparisons.
    #[builder(setter(skip), default = "self.get_dist_threshold()")]
    dist_threshold: f64,
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
            Some(SymmetryElementKind::Proper(false))
        } else if tr {
            let tr_rotated_mol = rotated_mol.reverse_time();
            if tr_rotated_mol == self.molecule {
                Some(SymmetryElementKind::Proper(true))
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
        let transformed_mol = self
            .molecule
            .improper_rotate(angle, axis, &kind.to_tr(false));
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
#[derive(Builder, Debug)]
pub struct Symmetry {
    /// The determined point group in Schönflies notation.
    #[builder(setter(skip, strip_option), default = "None")]
    pub point_group: Option<String>,

    /// The symmetry elements found.
    ///
    /// Each entry in the hash map is for one kind of symmetry elements: the key gives the kind,
    /// and the value is a hash map where each key gives the order and the corresponding value
    /// gives the [`HashSet`] of the elements with that order.
    ///
    /// Note that for improper elements, the mirror-plane convention is preferred.
    #[builder(setter(skip), default = "HashMap::new()")]
    pub elements: HashMap<SymmetryElementKind, HashMap<ElementOrder, HashSet<SymmetryElement>>>,

    /// The symmetry generators found.
    ///
    /// Each entry in the hash map is for one kind of symmetry generators: the key gives the kind,
    /// and the value is a hash map where each key gives the order and the corresponding value
    /// gives the [`HashSet`] of the generators with that order.
    ///
    /// Note that for improper generatrors, the mirror-plane convention is preferred.
    #[builder(setter(skip), default = "HashMap::new()")]
    pub generators: HashMap<SymmetryElementKind, HashMap<ElementOrder, HashSet<SymmetryElement>>>,
}

impl Symmetry {
    /// Returns a builder to construct a new symmetry struct.
    ///
    /// # Returns
    ///
    /// A builder to construct a new symmetry struct.
    #[must_use]
    pub fn builder() -> SymmetryBuilder {
        SymmetryBuilder::default()
    }

    pub fn new() -> Self {
        Symmetry::builder()
            .build()
            .expect("Unable to construct a `Symmetry` structure.")
    }

    /// Performs point-group detection analysis.
    ///
    /// # Arguments
    ///
    /// * presym - A pre-symmetry-analysis struct containing the molecule
    /// and its rotational symmetry required for point-group detection.
    pub fn analyse(&mut self, presym: &PreSymmetry, tr: bool) {
        log::debug!("Rotational symmetry found: {}", presym.rotational_symmetry);

        // Add the identity, which must always exist.
        let c1 = SymmetryElement::builder()
            .threshold(presym.dist_threshold)
            .proper_order(ORDER_1)
            .proper_power(1)
            .axis(Vector3::new(0.0, 0.0, 1.0))
            .kind(SymmetryElementKind::Proper(false))
            .build()
            .expect("Unable to construct the identity element.");
        if let Vacant(proper_elements) = self.elements.entry(ROT) {
            proper_elements.insert(HashMap::from([(ORDER_1, HashSet::from([c1]))]));
        } else {
            self.elements
                .get_mut(&ROT)
                .expect("Unable to add proper elements.")
                .insert(ORDER_1, HashSet::from([c1]));
        };

        match &presym.rotational_symmetry {
            RotationalSymmetry::Spherical => self.analyse_spherical(presym, tr),
            RotationalSymmetry::ProlateLinear => self.analyse_linear(presym, tr),
            RotationalSymmetry::OblatePlanar
            | RotationalSymmetry::OblateNonPlanar
            | RotationalSymmetry::ProlateNonLinear => self.analyse_symmetric(presym, tr),
            RotationalSymmetry::AsymmetricPlanar | RotationalSymmetry::AsymmetricNonPlanar => {
                self.analyse_asymmetric(presym, tr);
            }
        }
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
        axis: Vector3<f64>,
        generator: bool,
        threshold: f64,
        tr: bool,
    ) -> bool {
        let positive_axis = geometry::get_positive_pole(&axis, threshold).normalize();
        let element = SymmetryElement::builder()
            .threshold(threshold)
            .proper_order(order)
            .proper_power(1)
            .axis(positive_axis)
            .kind(SymmetryElementKind::Proper(tr))
            .generator(generator)
            .build()
            .expect("Unable to construct a proper element.");
        let detailed_symbol = element.get_detailed_symbol();
        let standard_symbol = element.get_standard_symbol();
        let proper_kind = if tr { TRROT } else { ROT };
        let result = if generator {
            if let Vacant(proper_generators) = self.generators.entry(proper_kind.clone()) {
                proper_generators.insert(HashMap::from([(order, HashSet::from([element]))]));
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
                    proper_generators_order.insert(HashSet::from([element]));
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
            proper_elements.insert(HashMap::from([(order, HashSet::from([element]))]));
            true
        } else {
            let proper_elements = self.elements.get_mut(&proper_kind).unwrap_or_else(|| {
                panic!(
                    "{} elements not found.",
                    if tr { "Time-reversed proper" } else { "Proper" }
                )
            });

            if let Vacant(e) = proper_elements.entry(order) {
                e.insert(HashSet::from([element]));
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
                detailed_symbol,
                standard_symbol,
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
        axis: Vector3<f64>,
        generator: bool,
        kind: SymmetryElementKind,
        sigma: Option<String>,
        threshold: f64,
        tr: bool,
    ) -> bool {
        let positive_axis = geometry::get_positive_pole(&axis, threshold).normalize();
        let element = if let Some(sigma_str) = sigma {
            assert!(sigma_str == "d" || sigma_str == "v" || sigma_str == "h");
            let mut sym_ele = SymmetryElement::builder()
                .threshold(threshold)
                .proper_order(order)
                .proper_power(1)
                .axis(positive_axis)
                .kind(kind.to_tr(tr))
                .generator(generator)
                .build()
                .expect("Unable to construct an improper symmetry element.")
                .convert_to_improper_kind(&SymmetryElementKind::ImproperMirrorPlane(tr), false);
            if sym_ele.proper_order == ElementOrder::Int(1) {
                sym_ele.additional_subscript = sigma_str;
            }
            sym_ele
        } else {
            SymmetryElement::builder()
                .threshold(threshold)
                .proper_order(order)
                .proper_power(1)
                .axis(positive_axis)
                .kind(kind.to_tr(tr))
                .generator(generator)
                .build()
                .expect("Unable to construct an improper symmetry element.")
                .convert_to_improper_kind(&SymmetryElementKind::ImproperMirrorPlane(tr), false)
        };
        let order = element.proper_order;
        let detailed_symbol = element.get_detailed_symbol();
        let standard_symbol = element.get_standard_symbol();
        let is_mirror_plane = element.is_mirror_plane(tr);
        let is_inversion_centre = element.is_inversion_centre(tr);
        let improper_kind = if tr { TRSIG } else { SIG };
        let result = if generator {
            if let Vacant(improper_generators) = self.generators.entry(improper_kind.clone()) {
                improper_generators.insert(HashMap::from([(order, HashSet::from([element]))]));
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
                    e.insert(HashSet::from([element]));
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
            improper_elements.insert(HashMap::from([(order, HashSet::from([element]))]));
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
                e.insert(HashSet::from([element]));
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
            if is_mirror_plane {
                log::debug!(
                    "Mirror plane {} ({}): {} axis along ({:+.3}, {:+.3}, {:+.3}) added.",
                    dest_str,
                    detailed_symbol,
                    standard_symbol,
                    positive_axis[0] + 0.0,
                    positive_axis[1] + 0.0,
                    positive_axis[2] + 0.0,
                );
            } else if is_inversion_centre {
                log::debug!(
                    "Inversion centre {} ({}): {} axis along ({:+.3}, {:+.3}, {:+.3}) added.",
                    dest_str,
                    detailed_symbol,
                    standard_symbol,
                    positive_axis[0] + 0.0,
                    positive_axis[1] + 0.0,
                    positive_axis[2] + 0.0,
                );
            } else {
                log::debug!(
                    "Improper rotation {} ({}): {} axis along ({:+.3}, {:+.3}, {:+.3}) added.",
                    dest_str,
                    detailed_symbol,
                    standard_symbol,
                    positive_axis[0] + 0.0,
                    positive_axis[1] + 0.0,
                    positive_axis[2] + 0.0,
                );
            }
        }
        result
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
    ) -> Option<&HashMap<ElementOrder, HashSet<SymmetryElement>>> {
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
    ) -> Option<&mut HashMap<ElementOrder, HashSet<SymmetryElement>>> {
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
    ) -> Option<&HashMap<ElementOrder, HashSet<SymmetryElement>>> {
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
    ) -> Option<&mut HashMap<ElementOrder, HashSet<SymmetryElement>>> {
        self.generators.get_mut(kind)
    }

    /// Obtains mirror-plane elements by their type (`"h"`, `"v"`, `"d"`, or `""`), including both
    /// time-reversed and non-time-reversed variants.
    ///
    /// # Returns
    ///
    /// A set of the required mirror-plane element type, if exists.
    #[must_use]
    pub fn get_sigma_elements(&self, sigma: &str) -> Option<HashSet<&SymmetryElement>> {
        self.get_improper(&ORDER_1).map(|sigma_elements| {
            sigma_elements
                .iter()
                .filter_map(|ele| {
                    if ele.additional_subscript == sigma {
                        Some(*ele)
                    } else {
                        None
                    }
                })
                .collect()
        })
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
    pub fn get_proper(&self, order: &ElementOrder) -> Option<HashSet<&SymmetryElement>> {
        let opt_proper_elements = self
            .get_elements(&ROT)
            .map(|proper_elements| proper_elements.get(&order))
            .unwrap_or_default();
        let opt_tr_proper_elements = self
            .get_elements(&TRROT)
            .map(|tr_proper_elements| tr_proper_elements.get(&order))
            .unwrap_or_default();

        match (opt_proper_elements, opt_tr_proper_elements) {
            (None, None) => None,
            (Some(proper_elements), None) => Some(HashSet::from_iter(proper_elements.iter())),
            (None, Some(tr_proper_elements)) => Some(HashSet::from_iter(tr_proper_elements.iter())),
            (Some(proper_elements), Some(tr_proper_elements)) => Some(HashSet::from_iter(
                proper_elements.iter().chain(tr_proper_elements.iter()),
            )),
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
    pub fn get_improper(&self, order: &ElementOrder) -> Option<HashSet<&SymmetryElement>> {
        let opt_improper_elements = self
            .get_elements(&SIG)
            .map(|improper_elements| improper_elements.get(&order))
            .unwrap_or_default();
        let opt_tr_improper_elements = self
            .get_elements(&TRSIG)
            .map(|tr_improper_elements| tr_improper_elements.get(&order))
            .unwrap_or_default();

        match (opt_improper_elements, opt_tr_improper_elements) {
            (None, None) => None,
            (Some(improper_elements), None) => Some(HashSet::from_iter(improper_elements.iter())),
            (None, Some(tr_improper_elements)) => {
                Some(HashSet::from_iter(tr_improper_elements.iter()))
            }
            (Some(improper_elements), Some(tr_improper_elements)) => Some(HashSet::from_iter(
                improper_elements.iter().chain(tr_improper_elements.iter()),
            )),
        }
    }

    /// Obtains a proper principal element, *i.e.* a time-reversed or non-time-reversed proper
    /// element with the highest order.
    ///
    /// If there are several such elements, the element to be returned will be randomly chosen.
    ///
    /// # Returns
    ///
    /// A proper principal element.
    pub fn get_proper_principal_element(&self) -> &SymmetryElement {
        let max_ord = self.get_max_proper_order();
        self.get_proper(&max_ord)
            .expect("No proper elements found.")
            .iter()
            .next()
            .expect("No proper principal elements found.")
    }

    /// Determines if this group is an infinite group.
    ///
    /// # Returns
    ///
    /// A flag indicating if this group is an infinite group.
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
fn _search_proper_rotations(presym: &PreSymmetry, sym: &mut Symmetry, asymmetric: bool, tr: bool) {
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
                                        sea_axes[2],
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
                                        sea_axes[2],
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
                        assert_eq!(k_sea % 2, 0);
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
                            sea_sym.analyse(&sea_presym, tr);
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
                                        presym.check_proper(order, &proper_element.axis, tr)
                                    {
                                        sym.add_proper(
                                            *order,
                                            proper_element.axis,
                                            false,
                                            presym.dist_threshold,
                                            proper_kind.contains_time_reversal(),
                                        );
                                    }
                                }
                            }

                            // BCH Jan 2023: The following shouldn't be here as this function is
                            // only to locate proper rotations. Including the following results in
                            // premature additions of mirror planes in
                            // adamantane_magnetic_field_bw_c3v case that cause the main symmetric
                            // algorithm to later fail to register these mirror planes.
                            // for (order, improper_elements) in sea_sym
                            //     .get_elements(&SIG)
                            //     .unwrap_or(&HashMap::new())
                            //     .iter()
                            //     .chain(
                            //         sea_sym
                            //             .get_elements(&TRSIG)
                            //             .unwrap_or(&HashMap::new())
                            //             .iter(),
                            //     )
                            // {
                            //     for improper_element in improper_elements {
                            //         if let Some(improper_kind) = presym.check_improper(
                            //             order,
                            //             &improper_element.axis,
                            //             &SIG,
                            //             tr,
                            //         ) {
                            //             log::debug!(
                            //                 "Check improper passed for {}.",
                            //                 improper_element
                            //             );
                            //             sym.add_improper(
                            //                 *order,
                            //                 improper_element.axis,
                            //                 false,
                            //                 SIG.clone(),
                            //                 None,
                            //                 presym.dist_threshold,
                            //                 improper_kind.contains_time_reversal(),
                            //             );
                            //         }
                            //     }
                            // }
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
                                            sea_axes[0],
                                            false,
                                            presym.dist_threshold,
                                            proper_kind.contains_time_reversal(),
                                        ));
                                    } else {
                                        sym.add_proper(
                                            k_fac_order,
                                            sea_axes[0],
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
                        assert_eq!(k_sea % 2, 0);
                        for k_fac in divisors::get_divisors(k_sea / 2)
                            .iter()
                            .chain(vec![k_sea / 2].iter())
                        {
                            let k_fac_order =
                                ElementOrder::Int((*k_fac).try_into().unwrap_or_else(|_| {
                                    panic!("Unable to convert {k_fac} to u32.")
                                }));
                            if let Some(proper_kind) =
                                presym.check_proper(&k_fac_order, &sea_axes[2], tr)
                            {
                                if *k_fac == 2 {
                                    count_c2 += usize::from(sym.add_proper(
                                        k_fac_order,
                                        sea_axes[2],
                                        false,
                                        presym.dist_threshold,
                                        proper_kind.contains_time_reversal(),
                                    ));
                                } else {
                                    sym.add_proper(
                                        k_fac_order,
                                        sea_axes[2],
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
                                    *sea_axis,
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
                count_c2 += usize::from(sym.add_proper(
                    ORDER_2,
                    atom_i_pos.coords,
                    false,
                    presym.dist_threshold,
                    proper_kind.contains_time_reversal(),
                ));
            }

            // Case A: C2 might cross through the midpoint of two atoms
            let midvec = 0.5 * (atom_i_pos.coords + atom_j_pos.coords);
            let c2_check = presym.check_proper(&ORDER_2, &midvec, tr);
            if midvec.norm() > presym.dist_threshold && c2_check.is_some() {
                count_c2 += usize::from(
                    sym.add_proper(
                        ORDER_2,
                        midvec,
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
                        e_vector,
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

    if asymmetric && count_c2 == 3 {
    } else {
        // Search for any remaining C2 axes.
        // Case C: Molecules with two or more sets of non-parallel linear diatomic SEA groups
        if linear_sea_groups.len() >= 2 {
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
                        normal,
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
}

mod symmetry_core_asymmetric;
mod symmetry_core_linear;
mod symmetry_core_spherical;
mod symmetry_core_symmetric;
