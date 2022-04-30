use super::Symmetry;
use crate::rotsym::RotationalSymmetry;
use crate::symmetry::symmetry_element::{ElementOrder, SymmetryElementKind};
use itertools;
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
            for (atom_i, atom_j) in itertools::iproduct!(sea_group, sea_group) {
                n_pairs += 1;
                let atom_i_pos = self.molecule.get_all_atoms()[*atom_i].coordinates;
                let atom_j_pos = self.molecule.get_all_atoms()[*atom_j].coordinates;

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
        let inv = SymmetryElementKind::ImproperInversionCentre;
        let sig = SymmetryElementKind::ImproperMirrorPlane;
        match count_c2 {
            3 => {
                // Tetrahedral, so either T, Td, or Th
                log::debug!("Tetrahedral family.");
                if self.check_improper(&order_1, &Vector3::new(0.0, 0.0, 1.0), &inv) {
                    // Inversion centre
                    log::debug!("Located an inversion centre.");
                    self.point_group = Some("Th".to_owned());
                    log::debug!(
                        "Point group determined: {}",
                        self.point_group.as_ref().unwrap()
                    );
                    assert!(self.add_improper(
                        order_1.clone(),
                        Vector3::new(0.0, 0.0, 1.0),
                        false,
                        inv.clone(),
                        None
                    ));
                    assert!(self.add_improper(
                        order_1.clone(),
                        Vector3::new(0.0, 0.0, 1.0),
                        true,
                        inv.clone(),
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
                            let c2s_1 = self.proper_elements[&order_2].iter();
                            let c2s_2 = self.proper_elements[&order_2].iter();
                            let mut axes = vec![];
                            for (c2_i, c2_j) in itertools::iproduct!(c2s_1, c2s_2) {
                                let axis_p = c2_i.axis + c2_j.axis;
                                assert!(self.check_improper(
                                    &order_1,
                                    &axis_p,
                                    &sig
                                ));
                                axes.push(axis_p);
                                let axis_m = c2_i.axis - c2_j.axis;
                                assert!(self.check_improper(
                                    &order_1,
                                    &axis_m,
                                    &sig
                                ));
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
            }, // end count_c2 = 3
            9 => {
                // 6 C2 and 3 C4^2; Octahedral, so either O or Oh
                log::debug!("Octahedral family.");
                if self.check_improper(&order_1, &Vector3::new(0.0, 0.0, 1.0), &inv) {
                    // Inversion centre
                    log::debug!("Located an inversion centre.");
                    self.point_group = Some("Oh".to_owned());
                    log::debug!(
                        "Point group determined: {}",
                        self.point_group.as_ref().unwrap()
                    );
                    assert!(self.add_improper(
                        order_1.clone(),
                        Vector3::new(0.0, 0.0, 1.0),
                        false,
                        inv.clone(),
                        None
                    ));
                    assert!(self.add_improper(
                        order_1.clone(),
                        Vector3::new(0.0, 0.0, 1.0),
                        true,
                        inv.clone(),
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
            }, // end count_c2 = 9
            15 => {
                // Icosahedral, so either I or Ih
                log::debug!("Icosahedral family.");
                if self.check_improper(&order_1, &Vector3::new(0.0, 0.0, 1.0), &inv) {
                    // Inversion centre
                    log::debug!("Located an inversion centre.");
                    self.point_group = Some("Ih".to_owned());
                    log::debug!(
                        "Point group determined: {}",
                        self.point_group.as_ref().unwrap()
                    );
                    assert!(self.add_improper(
                        order_1.clone(),
                        Vector3::new(0.0, 0.0, 1.0),
                        false,
                        inv.clone(),
                        None
                    ));
                    assert!(self.add_improper(
                        order_1.clone(),
                        Vector3::new(0.0, 0.0, 1.0),
                        true,
                        inv.clone(),
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
            }, // end count_c2 = 15
            _ => panic!("Invalid number of C2 axes.")
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
            if found_consistent_c3 { break };
            // for atom3s in itertools::iproduct!(sea_group, sea_group, sea_group) {
            // }
        }
    }

}
