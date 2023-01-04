use std::collections::HashMap;

use itertools::Itertools;
use log;
use nalgebra::Vector3;

use crate::rotsym::RotationalSymmetry;
use crate::symmetry::symmetry_core::_search_proper_rotations;
use crate::symmetry::symmetry_element::{ROT, SIG};
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
    /// * `presym` - A pre-symmetry-analysis struct containing information about
    /// the molecular system.
    ///
    /// # Panics
    ///
    /// Panics when any inconsistencies are encountered along the point-group detection path.
    #[allow(clippy::too_many_lines)]
    pub fn analyse_asymmetric(&mut self, presym: &PreSymmetry) {
        let (_mois, principal_axes) = presym.molecule.calc_moi();

        assert!(matches!(
            presym.rotational_symmetry,
            RotationalSymmetry::AsymmetricPlanar | RotationalSymmetry::AsymmetricNonPlanar
        ));

        _search_proper_rotations(presym, self, true);
        log::debug!("Proper elements found: {:?}", self.get_elements(&ROT));

        // Classify into point groups
        let count_c2 = if self
            .get_elements(&ROT)
            .unwrap_or(&HashMap::new())
            .contains_key(&ORDER_2)
        {
            self.get_elements(&ROT).unwrap_or(&HashMap::new())[&ORDER_2].len()
        } else {
            0
        };
        assert!(count_c2 == 0 || count_c2 == 1 || count_c2 == 3);

        let max_ord = self.get_max_proper_order();

        if count_c2 == 3 {
            // Dihedral, either D2h or D2.
            log::debug!("Dihedral family (asymmetric top).");
            assert_eq!(max_ord, ORDER_2);

            // Principal axis, which is C2, is also a generator.
            #[allow(clippy::needless_collect)]
            let c2_axes: Vec<_> = self.get_elements(&ROT).unwrap_or(&HashMap::new())[&ORDER_2]
                .iter()
                .map(|ele| ele.axis)
                .collect();
            let mut c2_axes_iter = c2_axes.into_iter();
            self.add_proper(
                max_ord,
                c2_axes_iter.next().expect("No C2 axes found."),
                true,
                presym.dist_threshold,
            );

            // One other C2 axis is also a generator.
            self.add_proper(
                max_ord,
                c2_axes_iter.next().expect("No other C2 axes found."),
                true,
                presym.dist_threshold,
            );

            let z_vec = Vector3::new(0.0, 0.0, 1.0);
            if presym.check_improper(&ORDER_2, &z_vec, &SIG) {
                // Inversion centre, D2h
                log::debug!("Located an inversion centre.");
                self.point_group = Some("D2h".to_owned());
                log::debug!(
                    "Point group determined: {}",
                    self.point_group.as_ref().expect("No point groups found.")
                );
                self.add_improper(
                    ORDER_2,
                    z_vec,
                    false,
                    SIG.clone(),
                    None,
                    presym.dist_threshold,
                );

                // Add remaining mirror planes, each of which is
                // perpendicular to a C2 axis.
                let c2_axes: Vec<_> = self.get_elements(&ROT).unwrap_or(&HashMap::new())[&ORDER_2]
                    .iter()
                    .map(|ele| ele.axis)
                    .collect();
                for c2_axis in c2_axes {
                    assert!(presym.check_improper(&ORDER_1, &c2_axis, &SIG));
                    self.add_improper(
                        ORDER_1,
                        c2_axis,
                        false,
                        SIG.clone(),
                        None,
                        presym.dist_threshold,
                    );
                }
                self.add_improper(
                    ORDER_1,
                    self.get_sigma_elements("")
                        .expect("No σ found.")
                        .iter()
                        .next()
                        .expect("No σ found.")
                        .axis,
                    true,
                    SIG.clone(),
                    None,
                    presym.dist_threshold,
                );
            } else {
                // Chiral, D2
                self.point_group = Some("D2".to_owned());
                log::debug!(
                    "Point group determined: {}",
                    self.point_group.as_ref().expect("No point groups found.")
                );
            }
        } else if count_c2 == 1 {
            // Non-dihedral, either C2, C2v, or C2h
            log::debug!("Non-dihedral family (asymmetric top).");
            assert_eq!(max_ord, ORDER_2);

            // Principal axis, which is C2, is also a generator.
            let c2_axis = self.get_elements(&ROT).unwrap_or(&HashMap::new())[&max_ord]
                .iter()
                .next()
                .expect("No C2 axes found.")
                .axis;
            self.add_proper(max_ord, c2_axis, true, presym.dist_threshold);

            let z_vec = Vector3::new(0.0, 0.0, 1.0);
            if presym.check_improper(&ORDER_2, &z_vec, &SIG) {
                // Inversion centre, C2h
                log::debug!("Located an inversion centre.");
                self.add_improper(
                    ORDER_2,
                    z_vec,
                    false,
                    SIG.clone(),
                    None,
                    presym.dist_threshold,
                );
                self.point_group = Some("C2h".to_owned());
                log::debug!(
                    "Point group determined: {}",
                    self.point_group.as_ref().expect("No point groups found.")
                );

                // There is one σh.
                let c2_axis = self.get_elements(&ROT).unwrap_or(&HashMap::new())[&max_ord]
                    .iter()
                    .next()
                    .expect("No C2 axes found.")
                    .axis;
                assert!(presym.check_improper(&ORDER_1, &c2_axis, &SIG));
                self.add_improper(
                    ORDER_1,
                    c2_axis,
                    false,
                    SIG.clone(),
                    Some("h".to_owned()),
                    presym.dist_threshold,
                );
                self.add_improper(
                    ORDER_1,
                    c2_axis,
                    true,
                    SIG.clone(),
                    Some("h".to_owned()),
                    presym.dist_threshold,
                );
            } else {
                // No inversion centres.
                // Locate σv planes
                let mut count_sigmav = 0;
                if (matches!(
                    presym.rotational_symmetry,
                    RotationalSymmetry::AsymmetricPlanar
                ) && matches!(presym.molecule.magnetic_atoms, None))
                {
                    assert!(presym.check_improper(&ORDER_1, &principal_axes[2], &SIG));
                    count_sigmav += u32::from(self.add_improper(
                        ORDER_1,
                        principal_axes[2],
                        false,
                        SIG.clone(),
                        Some("v".to_owned()),
                        presym.dist_threshold,
                    ));
                }

                let sea_groups = &presym.sea_groups;
                for sea_group in sea_groups.iter() {
                    if count_sigmav == 2 {
                        break;
                    }
                    if sea_group.len() < 2 {
                        continue;
                    }
                    for atom2s in sea_group.iter().combinations(2) {
                        if count_sigmav == 2 {
                            break;
                        }
                        let normal = (atom2s[0].coordinates.coords - atom2s[1].coordinates.coords)
                            .normalize();
                        if presym.check_improper(&ORDER_1, &normal, &SIG) {
                            count_sigmav += u32::from(self.add_improper(
                                ORDER_1,
                                normal,
                                false,
                                SIG.clone(),
                                Some("v".to_owned()),
                                presym.dist_threshold,
                            ));
                        }
                    }
                }

                log::debug!("Located {} σv.", count_sigmav);
                if count_sigmav == 2 {
                    self.point_group = Some("C2v".to_owned());
                    log::debug!(
                        "Point group determined: {}",
                        self.point_group.as_ref().expect("No point groups found.")
                    );

                    // In C2v, σv is also a generator.
                    self.add_improper(
                        ORDER_1,
                        self.get_sigma_elements("v")
                            .expect("No σv found.")
                            .iter()
                            .next()
                            .expect("No σv found.")
                            .axis,
                        true,
                        SIG.clone(),
                        Some("v".to_owned()),
                        presym.dist_threshold,
                    );
                } else {
                    assert_eq!(count_sigmav, 0);
                    self.point_group = Some("C2".to_owned());
                    log::debug!(
                        "Point group determined: {}",
                        self.point_group.as_ref().expect("No point groups found.")
                    );
                }
            }
        } else {
            // No C2 axes, so either C1, Ci, or Cs
            log::debug!("No C2 axes found.");
            let z_vec = Vector3::new(0.0, 0.0, 1.0);
            if presym.check_improper(&ORDER_2, &z_vec, &SIG) {
                log::debug!("Located an inversion centre.");
                self.point_group = Some("Ci".to_owned());
                log::debug!(
                    "Point group determined: {}",
                    self.point_group.as_ref().expect("No point groups found.")
                );
                self.add_improper(
                    ORDER_2,
                    z_vec,
                    false,
                    SIG.clone(),
                    None,
                    presym.dist_threshold,
                );
                self.add_improper(
                    ORDER_2,
                    z_vec,
                    true,
                    SIG.clone(),
                    None,
                    presym.dist_threshold,
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
                        if presym.check_improper(&ORDER_1, &normal, &SIG) {
                            count_sigma += u32::from(self.add_improper(
                                ORDER_1,
                                normal,
                                false,
                                SIG.clone(),
                                None,
                                presym.dist_threshold,
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
                    assert!(
                        presym.check_improper(&ORDER_1, &principal_axes[2], &SIG),
                        "Failed to check reflection symmetry from highest-MoI principal axis."
                    );
                    assert!(
                        self.add_improper(
                            ORDER_1,
                            principal_axes[2],
                            false,
                            SIG.clone(),
                            None,
                            presym.dist_threshold,
                        ),
                        "Failed to add mirror plane from highest-MoI principal axis."
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
                        .expect("No σ found.");
                    assert_eq!(old_sigmas.len(), 1);
                    let old_sigma = old_sigmas.into_iter().next().expect("No σ found.");
                    self.add_improper(
                        ORDER_1,
                        old_sigma.axis,
                        false,
                        SIG.clone(),
                        Some("h".to_owned()),
                        presym.dist_threshold,
                    );
                    self.add_improper(
                        ORDER_1,
                        old_sigma.axis,
                        true,
                        SIG.clone(),
                        Some("h".to_owned()),
                        presym.dist_threshold,
                    );

                    self.point_group = Some("Cs".to_owned());
                    log::debug!(
                        "Point group determined: {}",
                        self.point_group.as_ref().expect("No point groups found.")
                    );
                } else {
                    self.add_proper(
                        ORDER_1,
                        self.get_elements(&ROT).unwrap_or(&HashMap::new())[&ORDER_1]
                            .iter()
                            .next()
                            .expect("No identity found.")
                            .axis,
                        true,
                        presym.dist_threshold,
                    );
                    self.point_group = Some("C1".to_owned());
                    log::debug!(
                        "Point group determined: {}",
                        self.point_group.as_ref().expect("No point groups found.")
                    );
                }
            }
        }
    }
}
