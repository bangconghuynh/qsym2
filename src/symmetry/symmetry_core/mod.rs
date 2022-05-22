use crate::aux::geometry::{self, Transform};
use crate::aux::molecule::Molecule;
use crate::rotsym::{self, RotationalSymmetry};
use crate::symmetry::symmetry_element::{ElementOrder, SymmetryElement, SymmetryElementKind};
use log;
use nalgebra::{Point3, Vector3};
use std::collections::{HashMap, HashSet};

use derive_builder::Builder;

#[cfg(test)]
#[path = "symmetry_core_tests.rs"]
mod symmetry_core_tests;

#[cfg(test)]
#[path = "point_group_detection_tests.rs"]
mod point_group_detection_tests;

/// A struct for storing and managing symmetry information.
#[derive(Builder, Debug)]
pub struct Symmetry {
    /// The molecule associated with this [`Symmetry`] struct.
    #[builder(setter(custom))]
    molecule: Molecule,

    /// The static electric field vector being applied to [`Self::molecule`].
    #[builder(setter(strip_option), default = "None")]
    electric_field: Option<Vector3<f64>>,

    /// The static magnetic field vector being applied to [`Self::molecule`].
    #[builder(setter(strip_option), default = "None")]
    magnetic_field: Option<Vector3<f64>>,

    /// The rotational symmetry of [`Self::molecule`] based on its moments of
    /// inertia.
    #[builder(setter(skip, strip_option), default = "None")]
    rotational_symmetry: Option<RotationalSymmetry>,

    /// The point group of [`Self::molecule`] in the presence of any
    /// [`Self::electric_field`] and [`Self::magnetic_field`] in Schönflies
    /// notation.
    #[builder(setter(skip, strip_option), default = "None")]
    point_group: Option<String>,

    /// The groups of symmetry-equivalent atoms in [`Self::molecule`] in the
    /// presence of any [`Self::electric_field`] and [`Self::magnetic_field`].
    #[builder(setter(skip, strip_option), default = "None")]
    sea_groups: Option<Vec<Vec<usize>>>,

    /// The proper generators possessed by [`Self::molecule`] in the presence of
    /// any [`Self::electric_field`] and [`Self::magnetic_field`].
    ///
    /// Each key gives the order and the matching value gives the [`HashSet`] of
    /// the corresponding proper generators.
    #[builder(setter(skip), default = "HashMap::new()")]
    proper_generators: HashMap<ElementOrder, HashSet<SymmetryElement>>,

    /// The improper generators possessed by [`Self::molecule`] in the presence
    /// of any [`Self::electric_field`] and [`Self::magnetic_field`]. These
    /// generators are always defined in the mirror-plane convention.
    ///
    /// Each key gives the order and the matching value gives the [`HashSet`] of
    /// the corresponding improper generators.
    #[builder(setter(skip), default = "HashMap::new()")]
    improper_generators: HashMap<ElementOrder, HashSet<SymmetryElement>>,

    /// The proper elements possessed by [`Self::molecule`] in the presence of
    /// any [`Self::electric_field`] and [`Self::magnetic_field`].
    ///
    /// Each key gives the order and the matching value gives the [`HashSet`] of
    /// the corresponding proper elements.
    #[builder(setter(skip), default = "Self::default_proper_elements()")]
    proper_elements: HashMap<ElementOrder, HashSet<SymmetryElement>>,

    /// The improper elements possessed by [`Self::molecule`] in the presence
    /// of any [`Self::electric_field`] and [`Self::magnetic_field`]. These
    /// elements are always defined in the mirror-plane convention.
    ///
    /// Each key gives the order and the matching value gives the [`HashSet`] of
    /// the corresponding improper elements.
    #[builder(setter(skip), default = "HashMap::new()")]
    improper_elements: HashMap<ElementOrder, HashSet<SymmetryElement>>,

    /// Threshold for relative comparisons of moments of inertia.
    #[builder(setter(custom))]
    moi_threshold: f64,
}

impl SymmetryBuilder {
    fn default_proper_elements() -> HashMap<ElementOrder, HashSet<SymmetryElement>> {
        let mut proper_elements = HashMap::new();
        let c1 = SymmetryElement::builder()
            .threshold(1e-14)
            .order(ElementOrder::Int(1))
            .axis(Vector3::new(0.0, 0.0, 1.0))
            .kind(SymmetryElementKind::Proper)
            .build()
            .unwrap();
        let mut identity_element_set = HashSet::new();
        identity_element_set.insert(c1);
        proper_elements.insert(ElementOrder::Int(1), identity_element_set);
        proper_elements
    }

    pub fn molecule(&mut self, molecule: &Molecule) -> &mut Self {
        // The Symmetry struct now owns a copy of `molecule`.
        self.molecule = Some(molecule.clone());
        self
    }

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
}

impl Symmetry {
    /// Returns a builder to construct a new symmetry struct.
    ///
    /// # Returns
    ///
    /// A builder to construct a new symmetry struct.
    pub fn builder() -> SymmetryBuilder {
        SymmetryBuilder::default()
    }

    /// Sets the electric field vector applied to [`Self::molecule`].
    ///
    /// # Arguments
    ///
    /// * e_vector - An option for a vector.
    pub fn set_electric_field(&mut self, e_vector: Option<Vector3<f64>>) {
        log::warn!("Electric field modified. Resetting symmetry...");
        self.reset_symmetry();
        match e_vector {
            Some(vec) => {
                if approx::relative_eq!(
                    vec.norm(),
                    0.0,
                    epsilon = self.molecule.threshold,
                    max_relative = self.molecule.threshold
                ) {
                    self.electric_field = e_vector;
                } else {
                    self.electric_field = None
                };
            }
            None => self.electric_field = None,
        }
        self.molecule.set_electric_field(self.electric_field);
    }

    /// Sets the magnetic field vector applied to [`Self::molecule`].
    ///
    /// # Arguments
    ///
    /// * b_vector - An option for a vector.
    pub fn set_magnetic_field(&mut self, b_vector: Option<Vector3<f64>>) {
        log::warn!("Magnetic field modified. Resetting symmetry...");
        self.reset_symmetry();
        match b_vector {
            Some(vec) => {
                if approx::relative_eq!(
                    vec.norm(),
                    0.0,
                    epsilon = self.molecule.threshold,
                    max_relative = self.molecule.threshold
                ) {
                    self.magnetic_field = b_vector;
                } else {
                    self.magnetic_field = None
                };
            }
            None => self.magnetic_field = None,
        }
        self.molecule.set_magnetic_field(self.magnetic_field);
    }

    /// Clears all symmetry-analysed fields.
    fn reset_symmetry(&mut self) {
        self.sea_groups = None;
        self.point_group = None;
        self.proper_generators = HashMap::new();
        self.improper_generators = HashMap::new();
        self.proper_elements = SymmetryBuilder::default_proper_elements();
        self.improper_elements = HashMap::new();
    }

    /// Performs point-group detection analysis.
    ///
    /// This sets the fields [`Self::rotational_symmetry`], [`Self::sea_groups`].
    pub fn analyse(&mut self) {
        let com = self.molecule.calc_com(0);
        let inertia = self.molecule.calc_inertia_tensor(&com, 0);
        approx::assert_relative_eq!(
            com,
            Point3::origin(),
            epsilon = self.molecule.threshold,
            max_relative = self.molecule.threshold
        );
        self.rotational_symmetry = Some(rotsym::calc_rotational_symmetry(
            &inertia,
            self.moi_threshold,
            0,
        ));
        let moi_mat = inertia.symmetric_eigenvalues();
        let mut moi: Vec<&f64> = moi_mat.iter().collect();
        moi.sort_by(|a, b| a.partial_cmp(b).unwrap());
        self.sea_groups = Some(self.molecule.calc_sea_groups(0));

        match &self.rotational_symmetry {
            Some(rotsym) => match rotsym {
                RotationalSymmetry::Spherical => self.analyse_spherical(),
                _ => {}
            },
            _ => {}
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
    fn add_proper(&mut self, order: ElementOrder, axis: Vector3<f64>, generator: bool) -> bool {
        let positive_axis = geometry::get_positive_pole(&axis, self.molecule.threshold).normalize();
        let element = SymmetryElement::builder()
            .threshold(self.molecule.threshold)
            .order(order.clone())
            .axis(positive_axis)
            .kind(SymmetryElementKind::Proper)
            .generator(generator)
            .build()
            .unwrap();
        let detailed_symbol = element.get_detailed_symbol();
        let standard_symbol = element.get_standard_symbol();
        let result = if generator {
            if self.proper_generators.contains_key(&order) {
                self.proper_generators
                    .get_mut(&order)
                    .unwrap()
                    .insert(element)
            } else {
                self.proper_generators
                    .insert(order, HashSet::from([element]));
                true
            }
        } else {
            if self.proper_elements.contains_key(&order) {
                self.proper_elements
                    .get_mut(&order)
                    .unwrap()
                    .insert(element)
            } else {
                self.proper_elements.insert(order, HashSet::from([element]));
                true
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
    fn add_improper(
        &mut self,
        order: ElementOrder,
        axis: Vector3<f64>,
        generator: bool,
        kind: SymmetryElementKind,
        sigma: Option<String>,
    ) -> bool {
        let positive_axis = geometry::get_positive_pole(&axis, self.molecule.threshold).normalize();
        let element = if let Some(sigma_str) = sigma {
            assert!(sigma_str == "d" || sigma_str == "v" || sigma_str == "h");
            SymmetryElement::builder()
                .threshold(self.molecule.threshold)
                .order(order.clone())
                .axis(positive_axis)
                .kind(kind)
                .generator(generator)
                .additional_subscript(sigma_str)
                .build()
                .unwrap()
                .convert_to_improper_kind(&SymmetryElementKind::ImproperMirrorPlane)
        } else {
            SymmetryElement::builder()
                .threshold(self.molecule.threshold)
                .order(order.clone())
                .axis(positive_axis)
                .kind(kind)
                .generator(generator)
                .build()
                .unwrap()
                .convert_to_improper_kind(&SymmetryElementKind::ImproperMirrorPlane)
        };
        let detailed_symbol = element.get_detailed_symbol();
        let standard_symbol = element.get_standard_symbol();
        let is_mirror_plane = element.is_mirror_plane();
        let is_inversion_centre = element.is_inversion_centre();
        let result = if generator {
            if self.improper_generators.contains_key(&order) {
                self.improper_generators
                    .get_mut(&order)
                    .unwrap()
                    .insert(element)
            } else {
                self.improper_generators
                    .insert(order, HashSet::from([element]));
                true
            }
        } else {
            if self.improper_elements.contains_key(&order) {
                self.improper_elements
                    .get_mut(&order)
                    .unwrap()
                    .insert(element)
            } else {
                self.improper_elements
                    .insert(order, HashSet::from([element]));
                true
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

    /// Checks for the existence of the proper symmetry element $C_n$ along
    /// `axis` in `[Self::molecule]`.
    ///
    /// # Arguments
    ///
    /// * order - The geometrical order $n$ of the rotation axis. Only finite
    /// orders are supported.
    /// * axis - The rotation axis.
    ///
    /// # Returns
    ///
    /// A flag indicating if the $C_n$ element exists in `[Self::molecule]`.
    fn check_proper(&self, order: &ElementOrder, axis: &Vector3<f64>) -> bool {
        assert_ne!(
            *order,
            ElementOrder::Inf,
            "This method does not work for infinite-order elements."
        );
        let angle = 2.0 * std::f64::consts::PI / order.to_float();
        let rotated_mol = self.molecule.rotate(angle, axis);
        rotated_mol == self.molecule
    }

    /// Checks for the existence of the improper symmetry element $S_n$ or
    /// $\dot{S}_n$ along `axis` in `[Self::molecule]`.
    ///
    /// # Arguments
    ///
    /// * order - The geometrical order $n$ of the improper rotation axis. Only
    /// finite orders are supported.
    /// * axis - The rotation axis.
    /// * kind - The convention in which the improper element is defined.
    ///
    /// # Returns
    ///
    /// A flag indicating if the improper element exists in `[Self::molecule]`.
    fn check_improper(
        &self,
        order: &ElementOrder,
        axis: &Vector3<f64>,
        kind: &SymmetryElementKind,
    ) -> bool {
        assert_ne!(
            *order,
            ElementOrder::Inf,
            "This method does not work for infinite-order elements."
        );
        let angle = 2.0 * std::f64::consts::PI / order.to_float();
        let transformed_mol = self.molecule.improper_rotate(angle, axis, kind);
        transformed_mol == self.molecule
    }

    pub fn get_sigma_elements(&self, sigma: &str) -> Option<HashSet<&SymmetryElement>> {
        let order_1 = &ElementOrder::Int(1);
        if self.improper_elements.contains_key(&order_1) {
            Some(
                self.improper_elements[&order_1]
                    .iter()
                    .filter(|ele| ele.additional_subscript == sigma)
                    .collect(),
            )
        } else {
            None
        }
    }

    pub fn get_sigma_generators(&self, sigma: &str) -> Option<HashSet<&SymmetryElement>> {
        let order_1 = &ElementOrder::Int(1);
        if self.improper_elements.contains_key(&order_1) {
            Some(
                self.improper_generators[&order_1]
                    .iter()
                    .filter(|ele| ele.additional_subscript == sigma)
                    .collect(),
            )
        } else {
            None
        }
    }
}

mod symmetry_core_spherical;
