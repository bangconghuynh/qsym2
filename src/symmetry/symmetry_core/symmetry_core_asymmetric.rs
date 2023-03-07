use itertools::Itertools;
use log;
use nalgebra::Vector3;

use crate::rotsym::RotationalSymmetry;
use crate::symmetry::symmetry_core::_search_proper_rotations;
use crate::symmetry::symmetry_element::{SymmetryElement, ROT, SIG, TRROT, TRSIG};
use crate::symmetry::symmetry_element_order::{ORDER_1, ORDER_2};

use super::{PreSymmetry, Symmetry};

impl Symmetry {
    /// Performs point-group detection analysis for an asymmetric-top molecule.
    ///
    /// The possible asymmetric top point groups are:
    ///
    /// * $`\mathcal{C}_{1}`$ and $`\mathcal{C}_{2}`$,
    /// * $`\mathcal{C}_{2v}`$,
    /// * $`\mathcal{C}_{2h}`$,
    /// * $`\mathcal{C}_{s}`$,
    /// * $`\mathcal{D}_{2}`$,
    /// * $`\mathcal{D}_{2h}`$, and
    /// * $`\mathcal{C}_{i}`$.
    ///
    /// These are all Abelian groups.
    ///
    /// # Arguments
    ///
    /// * `presym` - A pre-symmetry-analysis structure containing information about the molecular
    /// system.
    /// * `tr` - A flag indicating if time reversal should also be considered. A time-reversed
    /// symmetry element will only be considered if its non-time-reversed version turns out to be
    /// not a symmetry element.
    ///
    /// # Panics
    ///
    /// Panics when any inconsistencies are encountered along the point-group detection path.
    #[allow(clippy::too_many_lines)]
    pub fn analyse_asymmetric(&mut self, presym: &PreSymmetry, tr: bool) {
        let (_mois, principal_axes) = presym.molecule.calc_moi();

        assert!(matches!(
            presym.rotational_symmetry,
            RotationalSymmetry::AsymmetricPlanar | RotationalSymmetry::AsymmetricNonPlanar
        ));

        _search_proper_rotations(presym, self, true, tr);
        log::debug!("Proper elements found: {:?}", self.get_elements(&ROT));
        log::debug!(
            "Time-reversed proper elements found: {:?}",
            self.get_elements(&TRROT)
        );

        // Classify into point groups
        let count_c2 = self
            .get_proper(&ORDER_2)
            .map_or(0, |proper_elements| proper_elements.len());
        assert!(count_c2 == 0 || count_c2 == 1 || count_c2 == 3);

        let max_ord = self.get_max_proper_order();

        if count_c2 == 3 {
            // Dihedral, either D2h or D2.
            log::debug!("Dihedral family (asymmetric top).");
            assert_eq!(max_ord, ORDER_2);

            // Principal axis, which is C2, is also a generator.
            // If the group is a black-white magnetic group, then one C2 axis is non-time-reversed,
            // while the other two are. Hence, one C2 generator is non-time-reversed, and the other
            // must be. We take care of this by sorting `c2s` to put any non-time-reversed elements
            // first.
            let mut c2s = self
                .get_proper(&ORDER_2)
                .expect(" No C2 elements found.")
                .into_iter()
                .cloned()
                .collect_vec();
            c2s.sort_by_key(SymmetryElement::contains_time_reversal);
            let mut c2s = c2s.into_iter();
            let c2 = c2s.next().expect(" No C2 elements found.");
            self.add_proper(
                max_ord,
                c2.raw_axis,
                true,
                presym.dist_threshold,
                c2.contains_time_reversal(),
            );

            // One other C2 axis is also a generator.
            let another_c2 = c2s.next().expect("No more C2s found.");
            self.add_proper(
                max_ord,
                another_c2.raw_axis,
                true,
                presym.dist_threshold,
                another_c2.contains_time_reversal(),
            );

            let z_vec = Vector3::new(0.0, 0.0, 1.0);
            if let Some(improper_kind) = presym.check_improper(&ORDER_2, &z_vec, &SIG, tr) {
                // Inversion centre, D2h
                log::debug!("Located an inversion centre.");
                self.set_group_name("D2h".to_owned());
                self.add_improper(
                    ORDER_2,
                    z_vec,
                    false,
                    SIG.clone(),
                    None,
                    presym.dist_threshold,
                    improper_kind.contains_time_reversal(),
                );

                // Add remaining mirror planes, each of which is
                // perpendicular to a C2 axis.
                let c2s = self
                    .get_proper(&ORDER_2)
                    .expect(" No C2 elements found.")
                    .into_iter()
                    .cloned()
                    .collect_vec();
                for c2 in &c2s {
                    let improper_check = presym.check_improper(&ORDER_1, &c2.raw_axis, &SIG, tr);
                    assert!(improper_check.is_some());
                    self.add_improper(
                        ORDER_1,
                        c2.raw_axis,
                        false,
                        SIG.clone(),
                        None,
                        presym.dist_threshold,
                        improper_check
                            .unwrap_or_else(|| {
                                panic!(
                                    "Expected mirror plane perpendicular to `{}` not found.",
                                    c2.raw_axis
                                )
                            })
                            .contains_time_reversal(),
                    );
                }
                // let sigmas = self.get_sigma_elements("").expect("No σ found.");
                // let sigma = sigmas.iter().next().expect("No σ found.");
                let principal_element = self.get_proper_principal_element();
                let improper_check =
                    presym.check_improper(&ORDER_1, &principal_element.raw_axis, &SIG, tr);
                assert!(improper_check.is_some());
                self.add_improper(
                    ORDER_1,
                    principal_element.raw_axis,
                    true,
                    SIG.clone(),
                    None,
                    presym.dist_threshold,
                    improper_check
                        .expect(
                            "Expected mirror plane perpendicular to the principal axis not found.",
                        )
                        .contains_time_reversal(),
                );
            } else {
                // Chiral, D2
                self.set_group_name("D2".to_owned());
            }
        } else if count_c2 == 1 {
            // Non-dihedral, either C2, C2v, or C2h
            log::debug!("Non-dihedral family (asymmetric top).");
            assert_eq!(max_ord, ORDER_2);

            // Principal axis, which is C2, is also a generator.
            let c2s = self.get_proper(&ORDER_2).expect(" No C2 elements found.");
            let c2 = (*c2s.iter().next().expect(" No C2 elements found.")).clone();
            self.add_proper(
                max_ord,
                c2.raw_axis,
                true,
                presym.dist_threshold,
                c2.contains_time_reversal(),
            );

            let z_vec = Vector3::new(0.0, 0.0, 1.0);
            if let Some(improper_kind) = presym.check_improper(&ORDER_2, &z_vec, &SIG, tr) {
                // Inversion centre, C2h
                log::debug!("Located an inversion centre.");
                self.add_improper(
                    ORDER_2,
                    z_vec,
                    false,
                    SIG.clone(),
                    None,
                    presym.dist_threshold,
                    improper_kind.contains_time_reversal(),
                );
                self.set_group_name("C2h".to_owned());

                // There is one σh.
                let c2 = (*self
                    .get_proper(&ORDER_2)
                    .expect(" No C2 elements found.")
                    .iter()
                    .next()
                    .expect(" No C2 elements found."))
                .clone();

                let improper_check = presym.check_improper(&ORDER_1, &c2.raw_axis, &SIG, tr);
                assert!(improper_check.is_some());
                self.add_improper(
                    ORDER_1,
                    c2.raw_axis,
                    false,
                    SIG.clone(),
                    Some("h".to_owned()),
                    presym.dist_threshold,
                    improper_check
                        .as_ref()
                        .unwrap_or_else(|| {
                            panic!(
                                "Expected mirror plane perpendicular to {} not found.",
                                c2.raw_axis
                            )
                        })
                        .contains_time_reversal(),
                );
                self.add_improper(
                    ORDER_1,
                    c2.raw_axis,
                    true,
                    SIG.clone(),
                    Some("h".to_owned()),
                    presym.dist_threshold,
                    improper_check
                        .unwrap_or_else(|| {
                            panic!(
                                "Expected mirror plane perpendicular to {} not found.",
                                c2.raw_axis
                            )
                        })
                        .contains_time_reversal(),
                );
            } else {
                // No inversion centres.
                // Locate σ planes
                let mut count_sigma = 0;
                if matches!(
                    presym.rotational_symmetry,
                    RotationalSymmetry::AsymmetricPlanar
                ) {
                    // Planar system. The plane of the system (perpendicular to the highest-MoI
                    // principal axis) might be a symmetry element: time-reversed in the presence of
                    // a magnetic field (which must also lie in this plane), or both in the absence
                    // of a magnetic field.
                    if let Some(improper_kind) =
                        presym.check_improper(&ORDER_1, &principal_axes[2], &SIG, tr)
                    {
                        if presym.molecule.magnetic_atoms.is_some() {
                            assert!(improper_kind.contains_time_reversal());
                        }
                        count_sigma += u32::from(self.add_improper(
                            ORDER_1,
                            principal_axes[2],
                            false,
                            SIG.clone(),
                            Some("v".to_owned()),
                            presym.dist_threshold,
                            improper_kind.contains_time_reversal(),
                        ));
                    }
                }

                let sea_groups = &presym.sea_groups;
                for sea_group in sea_groups.iter() {
                    if count_sigma == 2 {
                        break;
                    }
                    if sea_group.len() < 2 {
                        continue;
                    }
                    for atom2s in sea_group.iter().combinations(2) {
                        if count_sigma == 2 {
                            break;
                        }
                        let normal = (atom2s[0].coordinates.coords - atom2s[1].coordinates.coords)
                            .normalize();
                        if let Some(improper_kind) =
                            presym.check_improper(&ORDER_1, &normal, &SIG, tr)
                        {
                            if c2.contains_time_reversal()
                                && !improper_kind.contains_time_reversal()
                            {
                                log::debug!("The C2 axis is actually θ·C2. The non-time-reversed σv will be assigned as σh.");
                                count_sigma += u32::from(self.add_improper(
                                    ORDER_1,
                                    normal,
                                    false,
                                    SIG.clone(),
                                    Some("h".to_owned()),
                                    presym.dist_threshold,
                                    improper_kind.contains_time_reversal(),
                                ));
                            } else {
                                count_sigma += u32::from(self.add_improper(
                                    ORDER_1,
                                    normal,
                                    false,
                                    SIG.clone(),
                                    Some("v".to_owned()),
                                    presym.dist_threshold,
                                    improper_kind.contains_time_reversal(),
                                ));
                            }
                        }
                    }
                }

                log::debug!(
                    "Located {} σ ({} σv and {} σh).",
                    count_sigma,
                    self.get_sigma_elements("v")
                        .map_or(0, |sigmavs| sigmavs.len()),
                    self.get_sigma_elements("h")
                        .map_or(0, |sigmavs| sigmavs.len()),
                );
                if count_sigma == 2 {
                    self.set_group_name("C2v".to_owned());

                    // In C2v, one of the σ's is also a generator. We prioritise the
                    // non-time-reversed one as the generator.
                    let mut sigmas = self
                        .get_sigma_elements("v")
                        .unwrap_or_else(|| {
                            log::debug!("No σv found. Searching for σh instead.");
                            self.get_sigma_elements("h").expect("No σh found either.")
                        })
                        .into_iter()
                        .chain(self.get_sigma_elements("h").unwrap_or_default().into_iter())
                        .cloned()
                        .collect_vec();
                    sigmas.sort_by_key(SymmetryElement::contains_time_reversal);
                    let sigma = sigmas.first().expect("No σv or σh found.");
                    self.add_improper(
                        ORDER_1,
                        sigma.raw_axis,
                        true,
                        SIG.clone(),
                        Some(sigma.additional_subscript.clone()),
                        presym.dist_threshold,
                        sigma.contains_time_reversal(),
                    );
                } else {
                    assert_eq!(count_sigma, 0);
                    self.set_group_name("C2".to_owned());
                }
            }
        } else {
            // No C2 axes, so either C1, Ci, or Cs
            log::debug!("No C2 axes found.");
            let z_vec = Vector3::new(0.0, 0.0, 1.0);
            if let Some(improper_kind) = presym.check_improper(&ORDER_2, &z_vec, &SIG, tr) {
                log::debug!("Located an inversion centre.");
                self.set_group_name("Ci".to_owned());
                self.add_improper(
                    ORDER_2,
                    z_vec,
                    false,
                    SIG.clone(),
                    None,
                    presym.dist_threshold,
                    improper_kind.contains_time_reversal(),
                );
                self.add_improper(
                    ORDER_2,
                    z_vec,
                    true,
                    SIG.clone(),
                    None,
                    presym.dist_threshold,
                    improper_kind.contains_time_reversal(),
                );
            } else {
                log::debug!("No inversion centres found.");
                // Locate mirror planes
                let sea_groups = &presym.sea_groups;
                let mut count_sigma = 0;
                for sea_group in sea_groups.iter() {
                    if count_sigma > 0 {
                        break;
                    }
                    if sea_group.len() < 2 {
                        continue;
                    }
                    for atom2s in sea_group.iter().combinations(2) {
                        let normal = (atom2s[0].coordinates.coords - atom2s[1].coordinates.coords)
                            .normalize();
                        if let Some(improper_kind) =
                            presym.check_improper(&ORDER_1, &normal, &SIG, tr)
                        {
                            count_sigma += u32::from(self.add_improper(
                                ORDER_1,
                                normal,
                                false,
                                SIG.clone(),
                                None,
                                presym.dist_threshold,
                                improper_kind.contains_time_reversal(),
                            ));
                        }
                    }
                }

                if count_sigma == 0
                    && matches!(
                        presym.rotational_symmetry,
                        RotationalSymmetry::AsymmetricPlanar
                    )
                {
                    log::debug!("Planar molecule based on MoIs but no σ found from SEA groups.");
                    log::debug!("Locating the planar mirror plane based on MoIs...");
                    let sigma_check = presym.check_improper(&ORDER_1, &principal_axes[2], &SIG, tr);
                    assert!(
                        sigma_check.is_some(),
                        "Failed to check reflection symmetry perpendicular to the highest-MoI principal axis."
                    );
                    assert!(
                        self.add_improper(
                            ORDER_1,
                            principal_axes[2],
                            false,
                            SIG.clone(),
                            None,
                            presym.dist_threshold,
                            sigma_check
                                .expect(
                                    "Expected mirror plane perpendicular to the highest-MoI principal axis not found.",
                                )
                                .contains_time_reversal(),
                        ),
                        "Failed to add mirror plane perpendicular to the highest-MoI principal axis."
                    );
                    log::debug!("Located one planar mirror plane based on MoIs.");
                    count_sigma += 1;

                    // Old algorithm
                    // for atom3s in presym.molecule.atoms.iter().combinations(3) {
                    //     let normal = (atom3s[1].coordinates.coords - atom3s[0].coordinates.coords)
                    //         .cross(&(atom3s[2].coordinates.coords - atom3s[0].coordinates.coords));
                    //     if normal.norm() < presym.dist_threshold {
                    //         if let Some(e_atoms) = &presym.molecule.electric_atoms {
                    //             let normal = (atom3s[1].coordinates.coords
                    //                 - atom3s[0].coordinates.coords)
                    //                 .cross(
                    //                     &(e_atoms[1].coordinates.coords
                    //                         - e_atoms[0].coordinates.coords),
                    //                 );
                    //         } else {
                    //             continue;
                    //         }
                    //     }
                    //     let normal = normal.normalize();
                    //     if presym.check_improper(&ORDER_1, &normal, &SIG) {
                    //         count_sigma += self.add_improper(
                    //             ORDER_1,
                    //             normal,
                    //             false,
                    //             SIG.clone(),
                    //             None,
                    //             presym.dist_threshold,
                    //         ) as u32;
                    //         break;
                    //     }
                    // }
                    // assert_eq!(count_sigma, 1);
                    // log::debug!("Located one planar mirror plane based on MoIs.");
                }

                log::debug!("Located {} σ.", count_sigma);
                if count_sigma > 0 {
                    assert_eq!(count_sigma, 1);
                    let old_sigmas = self
                        .get_elements_mut(&SIG)
                        .expect("No improper elements found.")
                        .remove(&ORDER_1)
                        .unwrap_or_else(|| {
                            self.get_elements_mut(&TRSIG)
                                .expect("No improper elements found.")
                                .remove(&ORDER_1)
                                .expect("No σ found.")
                        });
                    assert_eq!(old_sigmas.len(), 1);
                    let old_sigma = old_sigmas.into_iter().next().expect("No σ found.");
                    self.add_improper(
                        ORDER_1,
                        old_sigma.raw_axis,
                        false,
                        SIG.clone(),
                        Some("h".to_owned()),
                        presym.dist_threshold,
                        old_sigma.contains_time_reversal(),
                    );
                    self.add_improper(
                        ORDER_1,
                        old_sigma.raw_axis,
                        true,
                        SIG.clone(),
                        Some("h".to_owned()),
                        presym.dist_threshold,
                        old_sigma.contains_time_reversal(),
                    );

                    self.set_group_name("Cs".to_owned());
                } else {
                    let identity = (*self
                        .get_proper(&ORDER_1)
                        .expect("No identity found.")
                        .iter()
                        .next()
                        .expect("No identity found."))
                    .clone();

                    self.add_proper(ORDER_1, identity.raw_axis, true, presym.dist_threshold, false);
                    self.set_group_name("C1".to_owned());
                }
            }
        }
    }
}
