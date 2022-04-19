use crate::aux::atom::Atom;
use crate::aux::molecule::Molecule;
use crate::rotsym::RotationalSymmetry;
use crate::symmetry::symmetry_element::{ElementOrder, SymmetryElement, SymmetryElementKind};
use log;
use nalgebra::Vector3;
use std::collections::{HashMap, HashSet};

use derive_builder::Builder;

#[cfg(test)]
#[path = "symmetry_core_tests.rs"]
mod symmetry_core_tests;

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
    sea_groups: Option<Vec<Vec<Atom>>>,

    /// The proper generators possessed by [`Self::molecule`] in the presence of
    /// any [`Self::electric_field`] and [`Self::magnetic_field`].
    ///
    /// Each key gives the order and the matching value gives the [`HashSet`] of
    /// the corresponding proper generators.
    #[builder(setter(skip), default = "HashMap::new()")]
    proper_generators: HashMap<ElementOrder, HashSet<SymmetryElement>>,

    /// The improper generators possessed by [`Self::molecule`] in the presence
    /// of any [`Self::electric_field`] and [`Self::magnetic_field`].
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
    /// of any [`Self::electric_field`] and [`Self::magnetic_field`].
    ///
    /// Each key gives the order and the matching value gives the [`HashSet`] of
    /// the corresponding improper elements.
    #[builder(setter(skip), default = "HashMap::new()")]
    improper_elements: HashMap<ElementOrder, HashSet<SymmetryElement>>,

    /// Threshold for relative comparisons.
    #[builder(setter(custom))]
    threshold: f64,

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
        self.molecule = Some(molecule.clone());
        self
    }

    pub fn threshold(&mut self, thresh: f64) -> &mut Self {
        if thresh >= f64::EPSILON {
            self.threshold = Some(thresh);
        } else {
            log::error!(
                "Threshold value {} is invalid. Threshold must be at least the machine epsilon.",
                thresh
            );
            self.threshold = None;
        }
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
                    epsilon = self.threshold,
                    max_relative = self.threshold
                ) {
                    self.electric_field = e_vector;
                } else {
                    self.electric_field = None
                };
            }
            None => self.electric_field = None,
        }
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
                    epsilon = self.threshold,
                    max_relative = self.threshold
                ) {
                    self.magnetic_field = b_vector;
                } else {
                    self.magnetic_field = None
                };
            }
            None => self.magnetic_field = None,
        }
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

    fn analyse(&mut self) {}
}
