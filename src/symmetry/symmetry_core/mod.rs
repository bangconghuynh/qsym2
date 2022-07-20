use crate::aux::atom::Atom;
use crate::aux::geometry::{self, Transform};
use crate::aux::molecule::Molecule;
use crate::rotsym::{self, RotationalSymmetry};
use crate::symmetry::symmetry_element::{SymmetryElement, SymmetryElementKind, SIG};
use crate::symmetry::symmetry_element_order::{ElementOrder, ORDER_2};
use log;
use nalgebra::{Point3, Vector3};
use itertools::Itertools;
use std::collections::{HashMap, HashSet};

use derive_builder::Builder;

#[cfg(test)]
#[path = "symmetry_core_tests.rs"]
mod symmetry_core_tests;

#[path = "point_group_detection_tests.rs"]
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
        let com = self.molecule.as_ref().unwrap().calc_com(0);
        let inertia = self.molecule.as_ref().unwrap().calc_inertia_tensor(&com, 0);
        approx::assert_relative_eq!(
            com,
            Point3::origin(),
            epsilon = self.molecule.as_ref().unwrap().threshold,
            max_relative = self.molecule.as_ref().unwrap().threshold
        );
        rotsym::calc_rotational_symmetry(&inertia, self.moi_threshold.unwrap(), 0)
    }

    fn calc_sea_groups(&self) -> Vec<Vec<Atom>> {
        self.molecule.as_ref().unwrap().calc_sea_groups(0)
    }

    fn get_dist_threshold(&self) -> f64 {
        self.molecule.as_ref().unwrap().threshold
    }
}

impl PreSymmetry {
    /// Returns a builder to construct a new pre-symmetry struct.
    ///
    /// # Returns
    ///
    /// A builder to construct a new pre-symmetry struct.
    pub fn builder() -> PreSymmetryBuilder {
        PreSymmetryBuilder::default()
    }

    /// Checks for the existence of the proper symmetry element $`C_n`$ along
    /// `axis` in `[Self::molecule]`.
    ///
    /// # Arguments
    ///
    /// * order - The geometrical order $`n`$ of the rotation axis. Only finite
    /// orders are supported.
    /// * axis - The rotation axis.
    ///
    /// # Returns
    ///
    /// A flag indicating if the $`C_n`$ element exists in `[Self::molecule]`.
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

    /// Checks for the existence of the improper symmetry element $`S_n`$ or
    /// $`\dot{S}_n`$ along `axis` in `[Self::molecule]`.
    ///
    /// # Arguments
    ///
    /// * order - The geometrical order $`n`$ of the improper rotation axis. Only
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
}

/// A struct for storing and managing symmetry analysis results.
#[derive(Builder, Debug)]
pub struct Symmetry {
    /// The determined point group in Schönflies notation.
    #[builder(setter(skip, strip_option), default = "None")]
    point_group: Option<String>,

    /// The proper generators found.
    ///
    /// Each key gives the order and the matching value gives the [`HashSet`] of
    /// the corresponding proper generators.
    #[builder(setter(skip), default = "HashMap::new()")]
    proper_generators: HashMap<ElementOrder, HashSet<SymmetryElement>>,

    /// The improper generators found. These generators are always defined in
    /// the mirror-plane convention.
    ///
    /// Each key gives the order and the matching value gives the [`HashSet`] of
    /// the corresponding improper generators.
    #[builder(setter(skip), default = "HashMap::new()")]
    improper_generators: HashMap<ElementOrder, HashSet<SymmetryElement>>,

    /// The proper elements found.
    ///
    /// Each key gives the order and the matching value gives the [`HashSet`] of
    /// the corresponding proper elements.
    #[builder(setter(skip), default = "Self::default_proper_elements()")]
    proper_elements: HashMap<ElementOrder, HashSet<SymmetryElement>>,

    /// The improper elements found. These elements are always defined in the
    /// mirror-plane convention.
    ///
    /// Each key gives the order and the matching value gives the [`HashSet`] of
    /// the corresponding improper elements.
    #[builder(setter(skip), default = "HashMap::new()")]
    improper_elements: HashMap<ElementOrder, HashSet<SymmetryElement>>,
}

impl SymmetryBuilder {
    fn default_proper_elements() -> HashMap<ElementOrder, HashSet<SymmetryElement>> {
        let mut proper_elements = HashMap::new();
        let c1 = SymmetryElement::builder()
            .threshold(1e-14)
            .order(ElementOrder::Int(1))
            .proper_power(1)
            .axis(Vector3::new(0.0, 0.0, 1.0))
            .kind(SymmetryElementKind::Proper)
            .build()
            .unwrap();
        let mut identity_element_set = HashSet::new();
        identity_element_set.insert(c1);
        proper_elements.insert(ElementOrder::Int(1), identity_element_set);
        proper_elements
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

    /// Performs point-group detection analysis.
    ///
    /// # Arguments
    ///
    /// * presym - A pre-symmetry-analysis struct containing the molecule
    /// and its rotational symmetry required for point-group detection.
    pub fn analyse(&mut self, presym: &PreSymmetry) {
        log::debug!("Rotational symmetry found: {}", presym.rotational_symmetry);
        match &presym.rotational_symmetry {
            RotationalSymmetry::Spherical => self.analyse_spherical(presym),
            RotationalSymmetry::ProlateLinear => self.analyse_linear(presym),
            RotationalSymmetry::OblatePlanar
            | RotationalSymmetry::OblateNonPlanar
            | RotationalSymmetry::ProlateNonLinear => self.analyse_symmetric(presym),
            RotationalSymmetry::AsymmetricPlanar
            | RotationalSymmetry::AsymmetricNonPlanar => self.analyse_asymmetric(presym),
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
    ) -> bool {
        let positive_axis = geometry::get_positive_pole(&axis, threshold).normalize();
        let element = SymmetryElement::builder()
            .threshold(threshold)
            .order(order.clone())
            .proper_power(1)
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
        threshold: f64,
    ) -> bool {
        let positive_axis = geometry::get_positive_pole(&axis, threshold).normalize();
        let element = if let Some(sigma_str) = sigma {
            assert!(sigma_str == "d" || sigma_str == "v" || sigma_str == "h");
            let mut sym_ele = SymmetryElement::builder()
                .threshold(threshold)
                .order(order.clone())
                .proper_power(1)
                .axis(positive_axis)
                .kind(kind)
                .generator(generator)
                .build()
                .unwrap()
                .convert_to_improper_kind(&SymmetryElementKind::ImproperMirrorPlane, false);
            if sym_ele.order == ElementOrder::Int(1) {
                sym_ele.additional_subscript = sigma_str;
            }
            sym_ele
        } else {
            SymmetryElement::builder()
                .threshold(threshold)
                .order(order.clone())
                .proper_power(1)
                .axis(positive_axis)
                .kind(kind)
                .generator(generator)
                .build()
                .unwrap()
                .convert_to_improper_kind(&SymmetryElementKind::ImproperMirrorPlane, false)
        };
        let sig_order = element.order.clone();
        let detailed_symbol = element.get_detailed_symbol();
        let standard_symbol = element.get_standard_symbol();
        let is_mirror_plane = element.is_mirror_plane();
        let is_inversion_centre = element.is_inversion_centre();
        let result = if generator {
            if self.improper_generators.contains_key(&sig_order) {
                self.improper_generators
                    .get_mut(&sig_order)
                    .unwrap()
                    .insert(element)
            } else {
                self.improper_generators
                    .insert(sig_order, HashSet::from([element]));
                true
            }
        } else {
            if self.improper_elements.contains_key(&sig_order) {
                self.improper_elements
                    .get_mut(&sig_order)
                    .unwrap()
                    .insert(element)
            } else {
                self.improper_elements
                    .insert(sig_order, HashSet::from([element]));
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

    /// Obtains mirror-plane elements by their type (`"h"`, `"v"`, `"d"`, or `""`).
    ///
    /// # Returns
    ///
    /// A set of the required mirror-plane element type, if exists.
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

    /// Obtains mirror-plane generators by their type (`"h"`, `"v"`, `"d"`, or `""`).
    ///
    /// # Returns
    ///
    /// A set of the required mirror-plane generator type, if exists.
    pub fn get_sigma_generators(&self, sigma: &str) -> Option<HashSet<&SymmetryElement>> {
        let order_1 = &ElementOrder::Int(1);
        if self.improper_generators.contains_key(&order_1) {
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

    /// Obtains the highest proper rotation order.
    ///
    /// # Returns
    ///
    /// The highest proper rotation order.
    pub fn get_max_proper_order(&self) -> ElementOrder {
        self.proper_generators
            .keys()
            .chain(self.proper_elements.keys())
            .max()
            .unwrap()
            .clone()
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
fn _search_proper_rotations(presym: &PreSymmetry, sym: &mut Symmetry, asymmetric: bool) {
    let mut linear_sea_groups: Vec<&Vec<Atom>> = vec![];
    let mut count_c2: usize = 0;
    for sea_group in presym.sea_groups.iter() {
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
                    for k_fac in k_fac_range.iter() {
                        if presym.check_proper(
                            &ElementOrder::Int((*k_fac).try_into().unwrap()),
                            &sea_axes[2],
                        ) {
                            match *k_fac {
                                2 => {
                                    count_c2 += sym.add_proper(
                                        ElementOrder::Int(*k_fac as u32),
                                        sea_axes[2].clone(),
                                        false,
                                        presym.dist_threshold,
                                    ) as usize;
                                }
                                _ => {
                                    sym.add_proper(
                                        ElementOrder::Int(*k_fac as u32),
                                        sea_axes[2].clone(),
                                        false,
                                        presym.dist_threshold,
                                    ) as usize;
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
                            log::debug!("A spherical top SEA set detected.");
                            let sea_presym = PreSymmetry::builder()
                                .moi_threshold(presym.moi_threshold)
                                .molecule(&sea_mol, true)
                                .build()
                                .unwrap();
                            let mut sea_sym = Symmetry::builder().build().unwrap();
                            log::debug!("Symmetry analysis for spherical top SEA begins.");
                            log::debug!("-----------------------------------------------");
                            sea_sym.analyse(&sea_presym);
                            log::debug!("Symmetry analysis for spherical top SEA ends.");
                            log::debug!("---------------------------------------------");
                            for (order, proper_elements) in sea_sym.proper_elements.iter() {
                                for proper_element in proper_elements {
                                    if presym.check_proper(&order, &proper_element.axis) {
                                        sym.add_proper(
                                            order.clone(),
                                            proper_element.axis,
                                            false,
                                            presym.dist_threshold,
                                        );
                                    }
                                }
                            }
                            for (order, improper_elements) in sea_sym.improper_elements.iter() {
                                for improper_element in improper_elements {
                                    if presym.check_improper(&order, &improper_element.axis, &SIG) {
                                        sym.add_improper(
                                            order.clone(),
                                            improper_element.axis,
                                            false,
                                            SIG.clone(),
                                            None,
                                            presym.dist_threshold,
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
                                let k_fac_order = ElementOrder::Int(*k_fac as u32);
                                if presym.check_proper(&k_fac_order, &sea_axes[0]) {
                                    if *k_fac == 2 {
                                        count_c2 += sym.add_proper(
                                            k_fac_order,
                                            sea_axes[0],
                                            false,
                                            presym.dist_threshold,
                                        )
                                            as usize;
                                    } else {
                                        sym.add_proper(
                                            k_fac_order,
                                            sea_axes[0],
                                            false,
                                            presym.dist_threshold,
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
                            let k_fac_order = ElementOrder::Int(*k_fac as u32);
                            if presym.check_proper(&k_fac_order, &sea_axes[2]) {
                                if *k_fac == 2 {
                                    count_c2 += sym.add_proper(
                                        k_fac_order,
                                        sea_axes[2],
                                        false,
                                        presym.dist_threshold,
                                    ) as usize;
                                } else {
                                    sym.add_proper(
                                        k_fac_order,
                                        sea_axes[2],
                                        false,
                                        presym.dist_threshold,
                                    );
                                }
                            }
                        }
                    } else {
                        // Asymmetric top
                        log::debug!("An asymmetric top SEA set detected.");
                        for sea_axis in sea_axes.iter() {
                            if presym.check_proper(&ORDER_2, sea_axis) {
                                count_c2 += sym.add_proper(
                                    ORDER_2.clone(),
                                    *sea_axis,
                                    false,
                                    presym.dist_threshold,
                                ) as usize;
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
            } else {
                let atom_i_pos = atom2s[0].coordinates;
                let atom_j_pos = atom2s[1].coordinates;

                // Case B: C2 might cross through any two atoms
                if presym.check_proper(&ORDER_2, &atom_i_pos.coords) {
                    count_c2 += sym.add_proper(
                        ORDER_2.clone(),
                        atom_i_pos.coords,
                        false,
                        presym.dist_threshold,
                    ) as usize;
                }

                // Case A: C2 might cross through the midpoint of two atoms
                let midvec = 0.5 * (&atom_i_pos.coords + &atom_j_pos.coords);
                if midvec.norm() > presym.dist_threshold && presym.check_proper(&ORDER_2, &midvec) {
                    count_c2 +=
                        sym.add_proper(ORDER_2.clone(), midvec, false, presym.dist_threshold)
                            as usize;
                } else if let Some(electric_atoms) = &presym.molecule.electric_atoms {
                    let e_vector = electric_atoms[0].coordinates - electric_atoms[1].coordinates;
                    if presym.check_proper(&ORDER_2, &e_vector) {
                        count_c2 +=
                            sym.add_proper(ORDER_2.clone(), e_vector, false, presym.dist_threshold)
                                as usize;
                    }
                }
            }
        }
    } // end for sea_group in presym.sea_groups.iter()

    if asymmetric && count_c2 == 3 {
        return;
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
                if presym.check_proper(&ORDER_2, &normal) {
                    sym.add_proper(ORDER_2.clone(), normal, false, presym.dist_threshold);
                }
            }
        }
    }
}

mod symmetry_core_linear;
mod symmetry_core_spherical;
mod symmetry_core_symmetric;
mod symmetry_core_asymmetric;
