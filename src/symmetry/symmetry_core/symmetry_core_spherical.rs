use super::Symmetry;
use crate::aux::geometry;
use crate::rotsym::RotationalSymmetry;
use crate::symmetry::symmetry_element::{ElementOrder, SymmetryElementKind};
use itertools::{self, Itertools};
use log;
use nalgebra::Vector3;
use std::collections::HashSet;

impl Symmetry {
    /// Locates and adds all possible and distinct $C_2$ axes present in the
    /// molecule in `sym`, provided that `sym` is a spherical top.
    fn search_c2_spherical(self: &mut Self) -> i8 {
        assert!(matches!(
            self.rotational_symmetry,
            Some(RotationalSymmetry::Spherical)
        ));

        let start_guard: usize = 30;
        let stable_c2_ratio: f64 = 0.5;
        let c2_termination_counts = HashSet::from([3, 9, 15]);

        let mut count_c2 = 0;
        let mut count_c2_stable = 0;
        let mut n_pairs = 0;

        let sea_groups = self.sea_groups.clone().unwrap();
        let order_2 = ElementOrder::Int(2);
        for sea_group in sea_groups.iter() {
            if sea_group.len() < 2 {
                continue;
            }
            for atom2s in sea_group.iter().combinations(2) {
                n_pairs += 1;
                let atom_i_pos = self.molecule.get_all_atoms()[*atom2s[0]].coordinates;
                let atom_j_pos = self.molecule.get_all_atoms()[*atom2s[1]].coordinates;

                // Case B: C2 might cross through any two atoms
                if self.check_proper(&order_2, &atom_i_pos.coords) {
                    if self.add_proper(order_2.clone(), atom_i_pos.coords, false) {
                        count_c2 += 1;
                        count_c2_stable = 0;
                    }
                }

                // Case A: C2 might cross through the midpoint of two atoms
                let midvec = 0.5 * (&atom_i_pos.coords + &atom_j_pos.coords);
                if midvec.norm() > self.molecule.threshold && self.check_proper(&order_2, &midvec) {
                    if self.add_proper(order_2.clone(), midvec, false) {
                        count_c2 += 1;
                        count_c2_stable = 0;
                    }
                }

                // Check if count_c2 has reached stability.
                if (count_c2_stable as f64) / (n_pairs as f64) > stable_c2_ratio
                    && n_pairs > start_guard
                {
                    if c2_termination_counts.contains(&count_c2) {
                        break;
                    }
                }
                count_c2_stable += 1;
            }

            if c2_termination_counts.contains(&count_c2) {
                break;
            }
        }
        count_c2
    }
    /// Performs point-group detection analysis for a spherical top.
    pub fn analyse_spherical(&mut self) {
        assert!(matches!(
            self.rotational_symmetry.as_ref().unwrap(),
            RotationalSymmetry::Spherical
        ));
        if self.molecule.atoms.len() == 1 {
            self.point_group = Some("O(3)".to_owned());
            log::debug!(
                "Point group determined: {}",
                self.point_group.as_ref().unwrap()
            );
            self.add_proper(ElementOrder::Inf, Vector3::new(0.0, 0.0, 1.0), true);
            self.add_proper(ElementOrder::Inf, Vector3::new(0.0, 1.0, 0.0), true);
            self.add_proper(ElementOrder::Inf, Vector3::new(1.0, 0.0, 0.0), true);
            self.add_improper(
                ElementOrder::Int(2),
                Vector3::new(0.0, 0.0, 1.0),
                true,
                SymmetryElementKind::ImproperMirrorPlane,
                None,
            );
            return;
        }

        // Locating all possible and distinct C2 axes
        let count_c2 = self.search_c2_spherical();
        log::debug!("Located {} C2 axes.", count_c2);
        assert!(HashSet::from([3, 9, 15]).contains(&count_c2));

        // Locating improper elements
        let order_1 = ElementOrder::Int(1);
        let order_2 = ElementOrder::Int(2);
        let sig = SymmetryElementKind::ImproperMirrorPlane;
        match count_c2 {
            3 => {
                // Tetrahedral, so either T, Td, or Th
                log::debug!("Tetrahedral family.");
                if self.check_improper(&order_2, &Vector3::new(0.0, 0.0, 1.0), &sig) {
                    // Inversion centre
                    log::debug!("Located an inversion centre.");
                    self.point_group = Some("Th".to_owned());
                    log::debug!(
                        "Point group determined: {}",
                        self.point_group.as_ref().unwrap()
                    );
                    assert!(self.add_improper(
                        order_2.clone(),
                        Vector3::new(0.0, 0.0, 1.0),
                        false,
                        sig.clone(),
                        None
                    ));
                    assert!(self.add_improper(
                        order_2.clone(),
                        Vector3::new(0.0, 0.0, 1.0),
                        true,
                        sig.clone(),
                        None
                    ));
                } else {
                    let mut c2s = self.proper_elements[&order_2].iter();
                    let normal = c2s.next().unwrap().axis + c2s.next().unwrap().axis;
                    if self.check_improper(&order_1, &normal, &sig) {
                        // σd
                        log::debug!("Located σd.");
                        self.point_group = Some("Td".to_owned());
                        log::debug!(
                            "Point group determined: {}",
                            self.point_group.as_ref().unwrap()
                        );
                        let sigmad_normals = {
                            let mut axes = vec![];
                            for c2s in self.proper_elements[&order_2].iter().combinations(2) {
                                let axis_p = c2s[0].axis + c2s[1].axis;
                                assert!(self.check_improper(&order_1, &axis_p, &sig));
                                axes.push(axis_p);
                                let axis_m = c2s[0].axis - c2s[1].axis;
                                assert!(self.check_improper(&order_1, &axis_m, &sig));
                                axes.push(axis_m);
                            }
                            axes
                        };
                        let sigmad_generator_normal = sigmad_normals[0].clone_owned();
                        for axis in sigmad_normals {
                            assert!(self.add_improper(
                                order_1.clone(),
                                axis,
                                false,
                                sig.clone(),
                                Some("d".to_owned())
                            ));
                        }
                        assert!(self.add_improper(
                            order_1.clone(),
                            sigmad_generator_normal,
                            true,
                            sig.clone(),
                            Some("d".to_owned())
                        ));
                    } else {
                        // No σd => chiral
                        self.point_group = Some("T".to_owned());
                        log::debug!(
                            "Point group determined: {}",
                            self.point_group.as_ref().unwrap()
                        );
                    }
                }
            } // end count_c2 = 3
            9 => {
                // 6 C2 and 3 C4^2; Octahedral, so either O or Oh
                log::debug!("Octahedral family.");
                if self.check_improper(&order_2, &Vector3::new(0.0, 0.0, 1.0), &sig) {
                    // Inversion centre
                    log::debug!("Located an inversion centre.");
                    self.point_group = Some("Oh".to_owned());
                    log::debug!(
                        "Point group determined: {}",
                        self.point_group.as_ref().unwrap()
                    );
                    assert!(self.add_improper(
                        order_2.clone(),
                        Vector3::new(0.0, 0.0, 1.0),
                        false,
                        sig.clone(),
                        None
                    ));
                    assert!(self.add_improper(
                        order_2.clone(),
                        Vector3::new(0.0, 0.0, 1.0),
                        true,
                        sig.clone(),
                        None
                    ));
                } else {
                    // No inversion centre => chiral
                    self.point_group = Some("O".to_owned());
                    log::debug!(
                        "Point group determined: {}",
                        self.point_group.as_ref().unwrap()
                    );
                }
            } // end count_c2 = 9
            15 => {
                // Icosahedral, so either I or Ih
                log::debug!("Icosahedral family.");
                if self.check_improper(&order_2, &Vector3::new(0.0, 0.0, 1.0), &sig) {
                    // Inversion centre
                    log::debug!("Located an inversion centre.");
                    self.point_group = Some("Ih".to_owned());
                    log::debug!(
                        "Point group determined: {}",
                        self.point_group.as_ref().unwrap()
                    );
                    assert!(self.add_improper(
                        order_2.clone(),
                        Vector3::new(0.0, 0.0, 1.0),
                        false,
                        sig.clone(),
                        None
                    ));
                    assert!(self.add_improper(
                        order_2.clone(),
                        Vector3::new(0.0, 0.0, 1.0),
                        true,
                        sig.clone(),
                        None
                    ));
                } else {
                    // No inversion centre => chiral
                    self.point_group = Some("I".to_owned());
                    log::debug!(
                        "Point group determined: {}",
                        self.point_group.as_ref().unwrap()
                    );
                }
            } // end count_c2 = 15
            _ => panic!("Invalid number of C2 axes."),
        } // end match count_c2

        // Locating all possible and distinct C3 axes
        let mut count_c3 = 0;
        let mut found_consistent_c3 = false;
        let sea_groups = self.sea_groups.clone().unwrap();
        let order_3 = ElementOrder::Int(3);
        for sea_group in sea_groups.iter() {
            if sea_group.len() < 3 {
                continue;
            }
            if found_consistent_c3 {
                break;
            };
            for atom3s in sea_group.iter().combinations(3) {
                let all_atoms = self.molecule.get_all_atoms();
                let atom_i = all_atoms[*atom3s[0]];
                let atom_j = all_atoms[*atom3s[1]];
                let atom_k = all_atoms[*atom3s[2]];
                if !geometry::check_regular_polygon(&[&atom_i, &atom_j, &atom_k]) {
                    continue;
                }
                let vec_ij = atom_j.coordinates - atom_i.coordinates;
                let vec_ik = atom_k.coordinates - atom_i.coordinates;
                let vec_normal = vec_ij.cross(&vec_ik);
                assert!(vec_normal.norm() > self.molecule.threshold);
                if self.check_proper(&order_3, &vec_normal) {
                    count_c3 += self.add_proper(order_3.clone(), vec_normal, false) as i32;
                }
                if count_c2 == 3 && count_c3 == 4 {
                    // Tetrahedral, 4 C3 axes
                    found_consistent_c3 = true;
                    break;
                }
                if count_c2 == 9 && count_c3 == 4 {
                    // Octahedral, 4 C3 axes
                    found_consistent_c3 = true;
                    break;
                }
                if count_c2 == 15 && count_c3 == 10 {
                    // Icosahedral, 10 C3 axes
                    found_consistent_c3 = true;
                    break;
                }
            }
        }
        assert!(found_consistent_c3);

        if count_c3 == 4 {
            // Tetrahedral or octahedral, C3 axes are also generators.
            let c3_axes: Vec<Vector3<f64>> = self.proper_elements[&order_3]
                .iter()
                .map(|element| element.axis)
                .collect();
            for c3_axis in c3_axes.iter() {
                self.add_proper(order_3.clone(), c3_axis.clone(), true);
            }
        }

        // Locating all possible and distinct C4 axes for O and Oh point groups
        if count_c2 == 9 {
            let mut count_c4 = 0;
            let mut found_consistent_c4 = false;
            let sea_groups = self.sea_groups.clone().unwrap();
            let order_4 = ElementOrder::Int(4);
            for sea_group in sea_groups.iter() {
                if sea_group.len() < 4 {
                    continue;
                }
                if found_consistent_c4 {
                    break;
                };
                for atom4s in sea_group.iter().combinations(4) {
                    let all_atoms = self.molecule.get_all_atoms();
                    let atom_i = all_atoms[*atom4s[0]];
                    let atom_j = all_atoms[*atom4s[1]];
                    let atom_k = all_atoms[*atom4s[2]];
                    let atom_l = all_atoms[*atom4s[3]];
                    if !geometry::check_regular_polygon(&[&atom_i, &atom_j, &atom_k, &atom_l]) {
                        continue;
                    }
                    let vec_ij = atom_j.coordinates - atom_i.coordinates;
                    let vec_ik = atom_k.coordinates - atom_i.coordinates;
                    let vec_normal = vec_ij.cross(&vec_ik);
                    assert!(vec_normal.norm() > self.molecule.threshold);
                    if self.check_proper(&order_4, &vec_normal) {
                        count_c4 += self.add_proper(order_4.clone(), vec_normal, false) as i32;
                    }
                    if count_c4 == 3 {
                        found_consistent_c4 = true;
                        break;
                    }
                }
            }
            assert!(found_consistent_c4);
            self.add_proper(
                order_4.clone(),
                self.proper_elements[&order_4]
                    .iter()
                    .next()
                    .unwrap()
                    .axis
                    .clone(),
                true,
            );
        } // end locating C4 axes for O and Oh

        // Locating all possible and distinct C5 axes for I and Ih point groups
        if count_c2 == 15 {
            let mut count_c5 = 0;
            let mut found_consistent_c5 = false;
            let sea_groups = self.sea_groups.clone().unwrap();
            let order_5 = ElementOrder::Int(5);
            for sea_group in sea_groups.iter() {
                if sea_group.len() < 5 {
                    continue;
                }
                if found_consistent_c5 {
                    break;
                };
                for atom5s in sea_group.iter().combinations(5) {
                    let all_atoms = self.molecule.get_all_atoms();
                    let atom_i = all_atoms[*atom5s[0]];
                    let atom_j = all_atoms[*atom5s[1]];
                    let atom_k = all_atoms[*atom5s[2]];
                    let atom_l = all_atoms[*atom5s[3]];
                    let atom_m = all_atoms[*atom5s[4]];
                    if !geometry::check_regular_polygon(&[
                        &atom_i, &atom_j, &atom_k, &atom_l, &atom_m,
                    ]) {
                        continue;
                    }
                    let vec_ij = atom_j.coordinates - atom_i.coordinates;
                    let vec_ik = atom_k.coordinates - atom_i.coordinates;
                    let vec_normal = vec_ij.cross(&vec_ik);
                    assert!(vec_normal.norm() > self.molecule.threshold);
                    if self.check_proper(&order_5, &vec_normal) {
                        count_c5 += self.add_proper(order_5.clone(), vec_normal, false) as i32;
                        self.add_proper(order_5.clone(), vec_normal, true);
                    }
                    if count_c5 == 6 {
                        found_consistent_c5 = true;
                        break;
                    }
                }
            }
            assert!(found_consistent_c5);
        } // end locating C5 axes for I and Ih

        // Locating any other improper rotation axes for the non-chinal groups
        if *self.point_group.as_ref().unwrap() == "Td" {
            // Locating S4
            let order_4 = ElementOrder::Int(4);
            let improper_s4_axes: Vec<Vector3<f64>> = {
                self.proper_elements[&order_2]
                    .iter()
                    .filter_map(|c2_ele| {
                        if self.check_improper(&order_4, &c2_ele.axis, &sig) {
                            Some(c2_ele.axis.clone())
                        } else {
                            None
                        }
                    })
                    .collect()
            };
            let mut count_s4 = 0;
            for s4_axis in improper_s4_axes.into_iter() {
                count_s4 +=
                    self.add_improper(order_4.clone(), s4_axis, false, sig.clone(), None) as i32;
            }
            assert_eq!(count_s4, 3);
        }
        // end locating improper axes for Td
        else if *self.point_group.as_ref().unwrap() == "Th" {
            // Locating σh
            let order_1 = ElementOrder::Int(1);
            let sigmah_normals: Vec<Vector3<f64>> = {
                self.proper_elements[&order_2]
                    .iter()
                    .filter_map(|c2_ele| {
                        if self.check_improper(&order_1, &c2_ele.axis, &sig) {
                            Some(c2_ele.axis.clone())
                        } else {
                            None
                        }
                    })
                    .collect()
            };
            let mut count_sigmah = 0;
            for sigmah_normal in sigmah_normals.into_iter() {
                count_sigmah += self.add_improper(
                    order_1.clone(),
                    sigmah_normal,
                    false,
                    sig.clone(),
                    Some("h".to_owned()),
                ) as i32;
            }
            assert_eq!(count_sigmah, 3);

            // Locating S6
            let order_6 = ElementOrder::Int(6);
            let s6_axes: Vec<Vector3<f64>> = {
                self.proper_elements[&order_3]
                    .iter()
                    .filter_map(|c3_ele| {
                        if self.check_improper(&order_6, &c3_ele.axis, &sig) {
                            Some(c3_ele.axis.clone())
                        } else {
                            None
                        }
                    })
                    .collect()
            };
            let mut count_s6 = 0;
            for s6_axis in s6_axes.into_iter() {
                count_s6 +=
                    self.add_improper(order_1.clone(), s6_axis, false, sig.clone(), None) as i32;
            }
            assert_eq!(count_s6, 4);
        }
        // end locating improper axes for Th
        else if *self.point_group.as_ref().unwrap() == "Oh" {
            // Locating S4
            let order_4 = ElementOrder::Int(4);
            let s4_axes: Vec<Vector3<f64>> = {
                self.proper_elements[&order_2]
                    .iter()
                    .filter_map(|c2_ele| {
                        if self.check_improper(&order_4, &c2_ele.axis, &sig) {
                            Some(c2_ele.axis.clone())
                        } else {
                            None
                        }
                    })
                    .collect()
            };
            let mut count_s4 = 0;
            let mut count_sigmah = 0;
            for s4_axis in s4_axes.into_iter() {
                count_s4 +=
                    self.add_improper(order_4.clone(), s4_axis, false, sig.clone(), None) as i32;
                count_sigmah += self.add_improper(
                    order_1.clone(),
                    s4_axis,
                    false,
                    sig.clone(),
                    Some("h".to_owned()),
                ) as i32;
            }
            assert_eq!(count_s4, 3);
            assert_eq!(count_sigmah, 3);

            // Locating σd
            let sigmad_normals: Vec<Vector3<f64>> = {
                self.proper_elements[&order_2]
                    .iter()
                    .filter_map(|c2_ele| {
                        if !self.check_improper(&order_4, &c2_ele.axis, &sig)
                            && self.check_improper(&order_1, &c2_ele.axis, &sig)
                        {
                            Some(c2_ele.axis.clone())
                        } else {
                            None
                        }
                    })
                    .collect()
            };
            let mut count_sigmad = 0;
            for sigmad_normal in sigmad_normals.into_iter() {
                count_sigmad += self.add_improper(
                    order_1.clone(),
                    sigmad_normal,
                    false,
                    sig.clone(),
                    Some("d".to_owned()),
                ) as i32;
            }
            assert_eq!(count_sigmad, 6);

            // Locating S6
            let order_6 = ElementOrder::Int(6);
            let s6_axes: Vec<Vector3<f64>> = {
                self.proper_elements[&order_3]
                    .iter()
                    .filter_map(|c3_ele| {
                        if self.check_improper(&order_6, &c3_ele.axis, &sig) {
                            Some(c3_ele.axis.clone())
                        } else {
                            None
                        }
                    })
                    .collect()
            };
            let mut count_s6 = 0;
            for s6_axis in s6_axes.into_iter() {
                count_s6 +=
                    self.add_improper(order_6.clone(), s6_axis, false, sig.clone(), None) as i32;
            }
            assert_eq!(count_s6, 4);
        } // end locating improper axes for Oh
        else if *self.point_group.as_ref().unwrap() == "Ih" {
            // Locating S10
            let order_5 = ElementOrder::Int(5);
            let order_10 = ElementOrder::Int(10);
            let s10_axes: Vec<Vector3<f64>> = {
                self.proper_elements[&order_5]
                    .iter()
                    .filter_map(|c5_ele| {
                        if self.check_improper(&order_10, &c5_ele.axis, &sig) {
                            Some(c5_ele.axis.clone())
                        } else {
                            None
                        }
                    })
                    .collect()
            };
            let mut count_s10 = 0;
            for s10_axis in s10_axes.into_iter() {
                count_s10 +=
                    self.add_improper(order_10.clone(), s10_axis, false, sig.clone(), None) as i32;
            }
            assert_eq!(count_s10, 6);

            // Locating S6
            let order_6 = ElementOrder::Int(6);
            let s6_axes: Vec<Vector3<f64>> = {
                self.proper_elements[&order_3]
                    .iter()
                    .filter_map(|c3_ele| {
                        if self.check_improper(&order_6, &c3_ele.axis, &sig) {
                            Some(c3_ele.axis.clone())
                        } else {
                            None
                        }
                    })
                    .collect()
            };
            let mut count_s6 = 0;
            for s6_axis in s6_axes.into_iter() {
                count_s6 +=
                    self.add_improper(order_6.clone(), s6_axis, false, sig.clone(), None) as i32;
            }
            assert_eq!(count_s6, 10);

            // Locating σ
            let sigma_normals: Vec<Vector3<f64>> = {
                self.proper_elements[&order_2]
                    .iter()
                    .filter_map(|c2_ele| {
                        if self.check_improper(&order_1, &c2_ele.axis, &sig) {
                            Some(c2_ele.axis.clone())
                        } else {
                            None
                        }
                    })
                    .collect()
            };
            let mut count_sigma = 0;
            for sigma_normal in sigma_normals.into_iter() {
                count_sigma += self.add_improper(
                    order_1.clone(),
                    sigma_normal,
                    false,
                    sig.clone(),
                    Some("d".to_owned()),
                ) as i32;
            }
            assert_eq!(count_sigma, 15);
        } // end locating improper axes for Ih
    }
}
