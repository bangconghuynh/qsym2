use std::collections::HashSet;

use itertools::{self, Itertools};
use log;
use nalgebra::Vector3;

use crate::aux::geometry;
use crate::rotsym::RotationalSymmetry;
use crate::symmetry::symmetry_element::SIG;
use crate::symmetry::symmetry_element_order::{ElementOrder, ORDER_1, ORDER_2};

use super::{PreSymmetry, Symmetry};

impl Symmetry {
    /// Locates and adds all possible and distinct $`C_2`$ axes present in the
    /// molecule in `presym`, provided that `presym` is a spherical top.
    ///
    /// # Arguments
    ///
    /// * presym - A pre-symmetry-analysis struct containing information about
    /// the molecular system.
    ///
    /// # Returns
    ///
    /// The number of distinct $`C_2`$ axes located.
    fn search_c2_spherical(&mut self, presym: &PreSymmetry, tr: bool) -> u32 {
        assert!(matches!(
            presym.rotational_symmetry,
            RotationalSymmetry::Spherical
        ));

        let start_guard: u32 = 30;
        let stable_c2_ratio: f64 = 0.5;
        let c2_termination_counts = HashSet::from([3, 9, 15]);

        let mut count_c2: u32 = 0;
        let mut count_c2_stable: u32 = 0;
        let mut n_pairs: u32 = 0;

        let sea_groups = &presym.sea_groups;
        for sea_group in sea_groups.iter() {
            if sea_group.len() < 2 {
                continue;
            }
            for atom2s in sea_group.iter().combinations(2) {
                n_pairs += 1;
                let atom_i_pos = atom2s[0].coordinates;
                let atom_j_pos = atom2s[1].coordinates;

                // Case B: C2 might cross through any two atoms
                if let Some(proper_kind) = presym.check_proper(&ORDER_2, &atom_i_pos.coords, tr) {
                    if self.add_proper(
                        ORDER_2,
                        atom_i_pos.coords,
                        false,
                        presym.molecule.threshold,
                        proper_kind.contains_time_reversal(),
                    ) {
                        count_c2 += 1;
                        count_c2_stable = 0;
                    }
                }

                // Case A: C2 might cross through the midpoint of two atoms
                let midvec = 0.5 * (atom_i_pos.coords + atom_j_pos.coords);
                if let Some(proper_kind) = presym.check_proper(&ORDER_2, &midvec, tr) {
                    if midvec.norm() > presym.molecule.threshold
                        && self.add_proper(
                            ORDER_2,
                            midvec,
                            false,
                            presym.molecule.threshold,
                            proper_kind.contains_time_reversal(),
                        )
                    {
                        count_c2 += 1;
                        count_c2_stable = 0;
                    }
                }

                // Check if count_c2 has reached stability.
                if (f64::from(count_c2_stable)) / (f64::from(n_pairs)) > stable_c2_ratio
                    && n_pairs > start_guard
                    && c2_termination_counts.contains(&count_c2)
                {
                    break;
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
    ///
    /// # Arguments
    ///
    /// * presym - A pre-symmetry-analysis struct containing information about
    /// the molecular system.
    ///
    /// # Panics
    ///
    /// Panics when any inconsistencies are encountered along the point-group detection path.
    #[allow(clippy::too_many_lines)]
    pub fn analyse_spherical(&mut self, presym: &PreSymmetry, tr: bool) {
        assert!(matches!(
            presym.rotational_symmetry,
            RotationalSymmetry::Spherical
        ));
        if presym.molecule.atoms.len() == 1 {
            self.point_group = Some("O(3)".to_owned());
            log::debug!(
                "Point group determined: {}",
                self.point_group.as_ref().expect("No point groups found.")
            );
            self.add_proper(
                ElementOrder::Inf,
                Vector3::new(0.0, 0.0, 1.0),
                true,
                presym.molecule.threshold,
                false,
            );
            self.add_proper(
                ElementOrder::Inf,
                Vector3::new(0.0, 1.0, 0.0),
                true,
                presym.molecule.threshold,
                false,
            );
            self.add_proper(
                ElementOrder::Inf,
                Vector3::new(1.0, 0.0, 0.0),
                true,
                presym.molecule.threshold,
                false,
            );
            self.add_improper(
                ElementOrder::Int(2),
                Vector3::new(0.0, 0.0, 1.0),
                true,
                SIG.clone(),
                None,
                presym.molecule.threshold,
                false,
            );
            return;
        }

        // Locating all possible and distinct C2 axes
        let count_c2 = self.search_c2_spherical(presym, tr);
        log::debug!("Located {} C2 axes.", count_c2);
        assert!(HashSet::from([3, 9, 15]).contains(&count_c2));

        // Locating improper elements
        match count_c2 {
            3 => {
                // Tetrahedral, so either T, Td, or Th
                log::debug!("Tetrahedral family.");
                if let Some(improper_kind) =
                    presym.check_improper(&ORDER_2, &Vector3::new(0.0, 0.0, 1.0), &SIG, tr)
                {
                    // Inversion centre
                    log::debug!("Located an inversion centre.");
                    self.point_group = Some("Th".to_owned());
                    log::debug!(
                        "Point group determined: {}",
                        self.point_group.as_ref().expect("No point groups found.")
                    );
                    assert!(self.add_improper(
                        ORDER_2,
                        Vector3::new(0.0, 0.0, 1.0),
                        false,
                        SIG.clone(),
                        None,
                        presym.molecule.threshold,
                        improper_kind.contains_time_reversal()
                    ));
                    assert!(self.add_improper(
                        ORDER_2,
                        Vector3::new(0.0, 0.0, 1.0),
                        true,
                        SIG.clone(),
                        None,
                        presym.molecule.threshold,
                        improper_kind.contains_time_reversal()
                    ));
                } else {
                    let mut c2s = self
                        .get_proper(&ORDER_2)
                        .expect(" No C2 elements found.")
                        .into_iter()
                        .take(2)
                        .cloned()
                        .collect_vec()
                        .into_iter();
                    let normal = c2s.next().expect("No C2 elements found.").axis
                        + c2s
                            .next()
                            .expect("Two C2 elements expected, but only one found.")
                            .axis;
                    if presym.check_improper(&ORDER_1, &normal, &SIG, tr).is_some() {
                        // σd
                        log::debug!("Located σd.");
                        self.point_group = Some("Td".to_owned());
                        log::debug!(
                            "Point group determined: {}",
                            self.point_group.as_ref().expect("No point groups found.")
                        );
                        let sigmad_normals = {
                            let mut axes = vec![];
                            for c2s in self
                                .get_proper(&ORDER_2)
                                .expect(" No C2 elements found.")
                                .iter()
                                .combinations(2)
                            {
                                let axis_p = c2s[0].axis + c2s[1].axis;
                                let p_improper_check =
                                    presym.check_improper(&ORDER_1, &axis_p, &SIG, tr);
                                assert!(p_improper_check.is_some());
                                axes.push((
                                    axis_p,
                                    p_improper_check.unwrap().contains_time_reversal(),
                                ));

                                let axis_m = c2s[0].axis - c2s[1].axis;
                                let m_improper_check =
                                    presym.check_improper(&ORDER_1, &axis_m, &SIG, tr);
                                assert!(m_improper_check.is_some());
                                axes.push((
                                    axis_m,
                                    m_improper_check.unwrap().contains_time_reversal(),
                                ));
                            }
                            axes
                        };
                        let sigmad_generator_normal = sigmad_normals[0].clone();
                        for (axis, axis_tr) in sigmad_normals {
                            assert!(self.add_improper(
                                ORDER_1,
                                axis,
                                false,
                                SIG.clone(),
                                Some("d".to_owned()),
                                presym.molecule.threshold,
                                axis_tr
                            ));
                        }
                        assert!(self.add_improper(
                            ORDER_1,
                            sigmad_generator_normal.0,
                            true,
                            SIG.clone(),
                            Some("d".to_owned()),
                            presym.molecule.threshold,
                            sigmad_generator_normal.1
                        ));
                    } else {
                        // No σd => chiral
                        self.point_group = Some("T".to_owned());
                        log::debug!(
                            "Point group determined: {}",
                            self.point_group.as_ref().expect("No point groups found.")
                        );
                    }
                }
            } // end count_c2 = 3
            9 => {
                // 6 C2 and 3 C4^2; Octahedral, so either O or Oh
                log::debug!("Octahedral family.");
                if let Some(improper_kind) =
                    presym.check_improper(&ORDER_2, &Vector3::new(0.0, 0.0, 1.0), &SIG, tr)
                {
                    // Inversion centre
                    log::debug!("Located an inversion centre.");
                    self.point_group = Some("Oh".to_owned());
                    log::debug!(
                        "Point group determined: {}",
                        self.point_group.as_ref().expect("No point groups found.")
                    );
                    assert!(self.add_improper(
                        ORDER_2,
                        Vector3::new(0.0, 0.0, 1.0),
                        false,
                        SIG.clone(),
                        None,
                        presym.molecule.threshold,
                        improper_kind.contains_time_reversal()
                    ));
                    assert!(self.add_improper(
                        ORDER_2,
                        Vector3::new(0.0, 0.0, 1.0),
                        true,
                        SIG.clone(),
                        None,
                        presym.molecule.threshold,
                        improper_kind.contains_time_reversal()
                    ));
                } else {
                    // No inversion centre => chiral
                    self.point_group = Some("O".to_owned());
                    log::debug!(
                        "Point group determined: {}",
                        self.point_group.as_ref().expect("No point groups found.")
                    );
                }
            } // end count_c2 = 9
            15 => {
                // Icosahedral, so either I or Ih
                log::debug!("Icosahedral family.");
                if let Some(improper_kind) =
                    presym.check_improper(&ORDER_2, &Vector3::new(0.0, 0.0, 1.0), &SIG, tr)
                {
                    // Inversion centre
                    log::debug!("Located an inversion centre.");
                    self.point_group = Some("Ih".to_owned());
                    log::debug!(
                        "Point group determined: {}",
                        self.point_group.as_ref().expect("No point groups found.")
                    );
                    assert!(self.add_improper(
                        ORDER_2,
                        Vector3::new(0.0, 0.0, 1.0),
                        false,
                        SIG.clone(),
                        None,
                        presym.molecule.threshold,
                        improper_kind.contains_time_reversal()
                    ));
                    assert!(self.add_improper(
                        ORDER_2,
                        Vector3::new(0.0, 0.0, 1.0),
                        true,
                        SIG.clone(),
                        None,
                        presym.molecule.threshold,
                        improper_kind.contains_time_reversal()
                    ));
                } else {
                    // No inversion centre => chiral
                    self.point_group = Some("I".to_owned());
                    log::debug!(
                        "Point group determined: {}",
                        self.point_group.as_ref().expect("No point groups found.")
                    );
                }
            } // end count_c2 = 15
            _ => panic!("Invalid number of C2 axes."),
        } // end match count_c2

        // Locating all possible and distinct C3 axes
        let mut count_c3 = 0;
        let mut found_consistent_c3 = false;
        let sea_groups = &presym.sea_groups;
        let order_3 = ElementOrder::Int(3);
        for sea_group in sea_groups.iter() {
            if sea_group.len() < 3 {
                continue;
            }
            if found_consistent_c3 {
                break;
            };
            for atom3s in sea_group.iter().combinations(3) {
                let atom_i = atom3s[0];
                let atom_j = atom3s[1];
                let atom_k = atom3s[2];
                if !geometry::check_regular_polygon(&[atom_i, atom_j, atom_k]) {
                    continue;
                }
                let vec_ij = atom_j.coordinates - atom_i.coordinates;
                let vec_ik = atom_k.coordinates - atom_i.coordinates;
                let vec_normal = vec_ij.cross(&vec_ik);
                assert!(vec_normal.norm() > presym.molecule.threshold);
                if let Some(proper_kind) = presym.check_proper(&order_3, &vec_normal, tr) {
                    count_c3 += i32::from(self.add_proper(
                        order_3,
                        vec_normal,
                        false,
                        presym.molecule.threshold,
                        proper_kind.contains_time_reversal(),
                    ));
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
            let c3s = self
                .get_proper(&order_3)
                .expect(" No C3 elements found.")
                .into_iter()
                .cloned()
                .collect_vec();
            for c3 in &c3s {
                self.add_proper(
                    order_3,
                    c3.axis,
                    true,
                    presym.molecule.threshold,
                    c3.contains_time_reversal(),
                );
            }
        }

        // Locating all possible and distinct C4 axes for O and Oh point groups
        if count_c2 == 9 {
            let mut count_c4 = 0;
            let mut found_consistent_c4 = false;
            let sea_groups = &presym.sea_groups;
            let order_4 = ElementOrder::Int(4);
            for sea_group in sea_groups.iter() {
                if sea_group.len() < 4 {
                    continue;
                }
                if found_consistent_c4 {
                    break;
                };
                for atom4s in sea_group.iter().combinations(4) {
                    let atom_i = atom4s[0];
                    let atom_j = atom4s[1];
                    let atom_k = atom4s[2];
                    let atom_l = atom4s[3];
                    if !geometry::check_regular_polygon(&[atom_i, atom_j, atom_k, atom_l]) {
                        continue;
                    }
                    let vec_ij = atom_j.coordinates - atom_i.coordinates;
                    let vec_ik = atom_k.coordinates - atom_i.coordinates;
                    let vec_normal = vec_ij.cross(&vec_ik);
                    assert!(vec_normal.norm() > presym.molecule.threshold);
                    if let Some(proper_kind) = presym.check_proper(&order_4, &vec_normal, tr) {
                        count_c4 += i32::from(self.add_proper(
                            order_4,
                            vec_normal,
                            false,
                            presym.molecule.threshold,
                            proper_kind.contains_time_reversal(),
                        ));
                    }
                    if count_c4 == 3 {
                        found_consistent_c4 = true;
                        break;
                    }
                }
            }
            assert!(found_consistent_c4);

            // Add a C4 as a generator
            let c4 = *self
                .get_proper(&order_4)
                .expect(" No C4 elements found.")
                .iter()
                .next()
                .expect("Expected C4 not found.");
            self.add_proper(
                order_4,
                c4.axis,
                true,
                presym.molecule.threshold,
                c4.contains_time_reversal(),
            );
        } // end locating C4 axes for O and Oh

        // Locating all possible and distinct C5 axes for I and Ih point groups
        if count_c2 == 15 {
            let mut count_c5 = 0;
            let mut found_consistent_c5 = false;
            let sea_groups = &presym.sea_groups;
            let order_5 = ElementOrder::Int(5);
            for sea_group in sea_groups.iter() {
                if sea_group.len() < 5 {
                    continue;
                }
                if found_consistent_c5 {
                    break;
                };
                for atom5s in sea_group.iter().combinations(5) {
                    let atom_i = atom5s[0];
                    let atom_j = atom5s[1];
                    let atom_k = atom5s[2];
                    let atom_l = atom5s[3];
                    let atom_m = atom5s[4];
                    if !geometry::check_regular_polygon(&[atom_i, atom_j, atom_k, atom_l, atom_m]) {
                        continue;
                    }
                    let vec_ij = atom_j.coordinates - atom_i.coordinates;
                    let vec_ik = atom_k.coordinates - atom_i.coordinates;
                    let vec_normal = vec_ij.cross(&vec_ik);
                    assert!(vec_normal.norm() > presym.molecule.threshold);
                    if let Some(proper_kind) = presym.check_proper(&order_5, &vec_normal, tr) {
                        count_c5 += i32::from(self.add_proper(
                            order_5,
                            vec_normal,
                            false,
                            presym.molecule.threshold,
                            proper_kind.contains_time_reversal(),
                        ));
                        self.add_proper(
                            order_5,
                            vec_normal,
                            true,
                            presym.molecule.threshold,
                            proper_kind.contains_time_reversal(),
                        );
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
        if *self.point_group.as_ref().expect("No point groups found.") == "Td" {
            // Locating S4
            let order_4 = ElementOrder::Int(4);
            let improper_s4_axes: Vec<(Vector3<f64>, bool)> = {
                self.get_proper(&ORDER_2)
                    .expect("Expected C2 elements not found.")
                    .iter()
                    .filter_map(|c2_ele| {
                        if let Some(improper_kind) =
                            presym.check_improper(&order_4, &c2_ele.axis, &SIG, tr)
                        {
                            Some((c2_ele.axis, improper_kind.contains_time_reversal()))
                        } else {
                            None
                        }
                    })
                    .collect()
            };
            let mut count_s4 = 0;
            for (s4_axis, s4_axis_tr) in improper_s4_axes {
                count_s4 += i32::from(self.add_improper(
                    order_4,
                    s4_axis,
                    false,
                    SIG.clone(),
                    None,
                    presym.molecule.threshold,
                    s4_axis_tr,
                ));
            }
            assert_eq!(count_s4, 3);
        }
        // end locating improper axes for Td
        else if *self.point_group.as_ref().expect("No point groups found.") == "Th" {
            // Locating σh
            let sigmah_normals: Vec<(Vector3<f64>, bool)> = {
                // self.get_elements(&ROT)
                //     .unwrap_or(&HashMap::new())
                //     .get(&ORDER_2)
                //     .unwrap_or(&HashSet::new())
                //     .iter()
                //     .chain(
                //         self.get_elements(&TRROT)
                //             .unwrap_or(&HashMap::new())
                //             .get(&ORDER_2)
                //             .unwrap_or(&HashSet::new()),
                //     )
                self.get_proper(&ORDER_2)
                    .expect("Expected C2 elements not found.")
                    .iter()
                    .filter_map(|c2_ele| {
                        if let Some(improper_kind) =
                            presym.check_improper(&ORDER_1, &c2_ele.axis, &SIG, tr)
                        {
                            Some((c2_ele.axis, improper_kind.contains_time_reversal()))
                        } else {
                            None
                        }
                    })
                    .collect()
            };
            let mut count_sigmah = 0;
            for (sigmah_normal, sigmah_normal_tr) in sigmah_normals {
                count_sigmah += i32::from(self.add_improper(
                    ORDER_1,
                    sigmah_normal,
                    false,
                    SIG.clone(),
                    Some("h".to_owned()),
                    presym.molecule.threshold,
                    sigmah_normal_tr,
                ));
            }
            assert_eq!(count_sigmah, 3);

            // Locating S6
            let order_6 = ElementOrder::Int(6);
            let s6_axes: Vec<(Vector3<f64>, bool)> = {
                self.get_proper(&order_3)
                    .expect("Expected C3 elements not found.")
                    .iter()
                    .filter_map(|c3_ele| {
                        if let Some(improper_kind) =
                            presym.check_improper(&order_6, &c3_ele.axis, &SIG, tr)
                        {
                            Some((c3_ele.axis, improper_kind.contains_time_reversal()))
                        } else {
                            None
                        }
                    })
                    .collect()
            };
            let mut count_s6 = 0;
            for (s6_axis, s6_axis_tr) in s6_axes {
                count_s6 += i32::from(self.add_improper(
                    order_6,
                    s6_axis,
                    false,
                    SIG.clone(),
                    None,
                    presym.molecule.threshold,
                    s6_axis_tr,
                ));
            }
            assert_eq!(count_s6, 4);
        }
        // end locating improper axes for Th
        else if *self.point_group.as_ref().expect("No point groups found.") == "Oh" {
            // Locating S4
            let order_4 = ElementOrder::Int(4);
            let s4_axes: Vec<(Vector3<f64>, bool)> = {
                self.get_proper(&ORDER_2)
                    .expect("Expected C2 elements not found.")
                    .iter()
                    .filter_map(|c2_ele| {
                        if let Some(improper_kind) =
                            presym.check_improper(&order_4, &c2_ele.axis, &SIG, tr)
                        {
                            Some((c2_ele.axis, improper_kind.contains_time_reversal()))
                        } else {
                            None
                        }
                    })
                    .collect()
            };
            let mut count_s4 = 0;
            for (s4_axis, s4_axis_tr) in &s4_axes {
                count_s4 += i32::from(self.add_improper(
                    order_4,
                    s4_axis.clone(),
                    false,
                    SIG.clone(),
                    None,
                    presym.molecule.threshold,
                    *s4_axis_tr,
                ));
            }
            assert_eq!(count_s4, 3);

            let sigmah_axes: Vec<(Vector3<f64>, bool)> = {
                s4_axes
                    .iter()
                    .filter_map(|(sigmah_axis, _)| {
                        if let Some(improper_kind) =
                            presym.check_improper(&ORDER_1, &sigmah_axis, &SIG, tr)
                        {
                            Some((sigmah_axis.clone(), improper_kind.contains_time_reversal()))
                        } else {
                            None
                        }
                    })
                    .collect()
            };
            let mut count_sigmah = 0;
            for (sigmah_axis, sigmah_axis_tr) in sigmah_axes {
                count_sigmah += i32::from(self.add_improper(
                    ORDER_1,
                    sigmah_axis,
                    false,
                    SIG.clone(),
                    Some("h".to_owned()),
                    presym.molecule.threshold,
                    sigmah_axis_tr,
                ));
            }
            assert_eq!(count_sigmah, 3);

            // Locating σd
            let sigmad_normals: Vec<(Vector3<f64>, bool)> = {
                self.get_proper(&ORDER_2)
                    .expect("Expected C2 elements not found.")
                    .iter()
                    .filter_map(|c2_ele| {
                        if presym
                            .check_improper(&order_4, &c2_ele.axis, &SIG, tr)
                            .is_none()
                        {
                            if let Some(improper_kind) =
                                presym.check_improper(&ORDER_1, &c2_ele.axis, &SIG, tr)
                            {
                                Some((c2_ele.axis, improper_kind.contains_time_reversal()))
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    })
                    .collect()
            };
            let mut count_sigmad = 0;
            for (sigmad_normal, sigmad_normal_tr) in sigmad_normals {
                count_sigmad += i32::from(self.add_improper(
                    ORDER_1,
                    sigmad_normal,
                    false,
                    SIG.clone(),
                    Some("d".to_owned()),
                    presym.molecule.threshold,
                    sigmad_normal_tr,
                ));
            }
            assert_eq!(count_sigmad, 6);

            // Locating S6
            let order_6 = ElementOrder::Int(6);
            let s6_axes: Vec<(Vector3<f64>, bool)> = {
                self.get_proper(&order_3)
                    .expect("Expected C3 elements not found.")
                    .iter()
                    .filter_map(|c3_ele| {
                        if let Some(improper_kind) =
                            presym.check_improper(&order_6, &c3_ele.axis, &SIG, tr)
                        {
                            Some((c3_ele.axis, improper_kind.contains_time_reversal()))
                        } else {
                            None
                        }
                    })
                    .collect()
            };
            let mut count_s6 = 0;
            for (s6_axis, s6_axis_tr) in s6_axes {
                count_s6 += i32::from(self.add_improper(
                    order_6,
                    s6_axis,
                    false,
                    SIG.clone(),
                    None,
                    presym.molecule.threshold,
                    s6_axis_tr,
                ));
            }
            assert_eq!(count_s6, 4);
        }
        // end locating improper axes for Oh
        else if *self.point_group.as_ref().expect("No point groups found.") == "Ih" {
            // Locating S10
            let order_5 = ElementOrder::Int(5);
            let order_10 = ElementOrder::Int(10);
            let s10_axes: Vec<(Vector3<f64>, bool)> = {
                // self.get_elements(&ROT)
                //     .unwrap_or(&HashMap::new())
                //     .get(&order_5)
                //     .unwrap_or(&HashSet::new())
                //     .iter()
                //     .chain(
                //         self.get_elements(&TRROT)
                //             .unwrap_or(&HashMap::new())
                //             .get(&order_5)
                //             .unwrap_or(&HashSet::new()),
                //     )
                self.get_proper(&order_5)
                    .expect("Expected C5 elements not found.")
                    .iter()
                    .filter_map(|c5_ele| {
                        if let Some(improper_kind) =
                            presym.check_improper(&order_10, &c5_ele.axis, &SIG, tr)
                        {
                            Some((c5_ele.axis, improper_kind.contains_time_reversal()))
                        } else {
                            None
                        }
                    })
                    .collect()
            };
            let mut count_s10 = 0;
            for (s10_axis, s10_axis_tr) in s10_axes {
                count_s10 += i32::from(self.add_improper(
                    order_10,
                    s10_axis,
                    false,
                    SIG.clone(),
                    None,
                    presym.molecule.threshold,
                    s10_axis_tr,
                ));
            }
            assert_eq!(count_s10, 6);

            // Locating S6
            let order_6 = ElementOrder::Int(6);
            let s6_axes: Vec<(Vector3<f64>, bool)> = {
                self.get_proper(&order_3)
                    .expect("Expected C3 elements not found.")
                    .iter()
                    .filter_map(|c3_ele| {
                        if let Some(improper_kind) =
                            presym.check_improper(&order_6, &c3_ele.axis, &SIG, tr)
                        {
                            Some((c3_ele.axis, improper_kind.contains_time_reversal()))
                        } else {
                            None
                        }
                    })
                    .collect()
            };
            let mut count_s6 = 0;
            for (s6_axis, s6_axis_tr) in s6_axes {
                count_s6 += i32::from(self.add_improper(
                    order_6,
                    s6_axis,
                    false,
                    SIG.clone(),
                    None,
                    presym.molecule.threshold,
                    s6_axis_tr,
                ));
            }
            assert_eq!(count_s6, 10);

            // Locating σ
            let sigma_normals: Vec<(Vector3<f64>, bool)> = {
                self.get_proper(&ORDER_2)
                    .expect("Expected C2 elements not found.")
                    .iter()
                    .filter_map(|c2_ele| {
                        if let Some(improper_kind) =
                            presym.check_improper(&ORDER_1, &c2_ele.axis, &SIG, tr)
                        {
                            Some((c2_ele.axis, improper_kind.contains_time_reversal()))
                        } else {
                            None
                        }
                    })
                    .collect()
            };
            let mut count_sigma = 0;
            for (sigma_normal, sigma_normal_tr) in sigma_normals {
                count_sigma += i32::from(self.add_improper(
                    ORDER_1,
                    sigma_normal,
                    false,
                    SIG.clone(),
                    None,
                    presym.molecule.threshold,
                    sigma_normal_tr,
                ));
            }
            assert_eq!(count_sigma, 15);
        } // end locating improper axes for Ih
    }
}
