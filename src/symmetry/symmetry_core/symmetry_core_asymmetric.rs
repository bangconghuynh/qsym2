use super::{PreSymmetry, Symmetry};
use crate::rotsym::RotationalSymmetry;
use crate::symmetry::symmetry_core::_search_proper_rotations;
use crate::symmetry::symmetry_element::{
    ElementOrder, ORDER_1, ORDER_2, SIG,
};
use approx;
use itertools::Itertools;
use log;
use nalgebra::Vector3;

impl Symmetry {
    /// Performs point-group detection analysis for an asymmetric-top molecule.
    ///
    /// The possible symmetric top point groups are:
    ///
    /// * $`\mathcal{C}_{1}`$ and $`\mathcal{C}_{2}`$,
    /// * $`\mathcal{C}_{2v}`$,
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
    pub fn analyse_asymmetric(&mut self, presym: &PreSymmetry) {
        let (_mois, _principal_axes) = presym.molecule.calc_moi();

        assert!(matches!(
            presym.rotational_symmetry,
            RotationalSymmetry::AsymmetricPlanar | RotationalSymmetry::AsymmetricNonPlanar
        ));

        _search_proper_rotations(presym, self, true);
        log::debug!("Proper elements found: {:?}", self.proper_elements);

        // Classify into point groups
        let count_c2 = if self.proper_elements.contains_key(&ORDER_2) {
            self.proper_elements[&ORDER_2].len()
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
            let c2_axes: Vec<_> = self.proper_elements[&ORDER_2]
                .iter()
                .map(|ele| ele.axis.clone())
                .collect();
            let mut c2_axes_iter = c2_axes.into_iter();
            self.add_proper(
                max_ord.clone(),
                c2_axes_iter.next().unwrap(),
                true,
                presym.dist_threshold,
            );

            // One other C2 axis is also a generator.
            self.add_proper(
                max_ord.clone(),
                c2_axes_iter.next().unwrap(),
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
                    self.point_group.as_ref().unwrap()
                );
                self.add_improper(
                    ORDER_2.clone(),
                    z_vec,
                    false,
                    SIG.clone(),
                    None,
                    presym.dist_threshold,
                );

                // Add remaining mirror planes, each of which is
                // perpendicular to a C2 axis.
                let c2_axes: Vec<_> = self.proper_elements[&ORDER_2]
                    .iter()
                    .map(|ele| ele.axis.clone())
                    .collect();
                for c2_axis in c2_axes.into_iter() {
                    assert!(presym.check_improper(&ORDER_1, &c2_axis, &SIG));
                    self.add_improper(
                        ORDER_1.clone(),
                        c2_axis,
                        false,
                        SIG.clone(),
                        None,
                        presym.dist_threshold,
                    );
                }
                self.add_improper(
                    ORDER_1.clone(),
                    self.get_sigma_elements("")
                        .unwrap()
                        .iter()
                        .next()
                        .unwrap()
                        .axis
                        .clone(),
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
                    self.point_group.as_ref().unwrap()
                );
            }
        } else if count_c2 == 1 {
            // Non-dihedral, either C2, C2v, or C2h
            log::debug!("Non-dihedral family (asymmetric top).");
            assert_eq!(max_ord, ORDER_2);

            // Principal axis, which is C2, is also a generator.
            let c2_axis = self.proper_elements[&max_ord].iter().next().unwrap().axis;
            self.add_proper(max_ord.clone(), c2_axis, true, presym.dist_threshold);

            let z_vec = Vector3::new(0.0, 0.0, 1.0);
            if presym.check_improper(&ORDER_2, &z_vec, &SIG) {
                // Inversion centre, C2h
                log::debug!("Located an inversion centre.");
                self.point_group = Some("C2h".to_owned());
                log::debug!(
                    "Point group determined: {}",
                    self.point_group.as_ref().unwrap()
                );

                // There is one σh.
                let c2_axis = self.proper_elements[&max_ord].iter().next().unwrap().axis;
                assert!(presym.check_improper(&ORDER_1, &c2_axis, &SIG));
                self.add_improper(
                    ORDER_1.clone(),
                    c2_axis,
                    false,
                    SIG.clone(),
                    Some("h".to_owned()),
                    presym.dist_threshold,
                );
                self.add_improper(
                    ORDER_1.clone(),
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
                    assert!(presym.check_improper(&ORDER_1, &_principal_axes[2], &SIG));
                    count_sigmav += self.add_improper(
                        ORDER_1.clone(),
                        _principal_axes[2].clone(),
                        false,
                        SIG.clone(),
                        Some("v".to_owned()),
                        presym.dist_threshold,
                    ) as u32;
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
                            count_sigmav += self.add_improper(
                                ORDER_1.clone(),
                                normal,
                                false,
                                SIG.clone(),
                                Some("v".to_owned()),
                                presym.dist_threshold,
                            ) as u32;
                        }
                    }
                }

                log::debug!("Located {} σv.", count_sigmav);
                if count_sigmav == 2 {
                    self.point_group = Some("C2v".to_owned());
                    log::debug!(
                        "Point group determined: {}",
                        self.point_group.as_ref().unwrap()
                    );

                    // In C2v, σv is also a generator.
                    self.add_improper(
                        ORDER_1.clone(),
                        self.get_sigma_elements("v")
                            .unwrap()
                            .iter()
                            .next()
                            .unwrap()
                            .axis
                            .clone(),
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
                        self.point_group.as_ref().unwrap()
                    );
                }
            }
        } else {
            // No C2 axes, so either C1, Ci, or Cs
            let z_vec = Vector3::new(0.0, 0.0, 1.0);
            if presym.check_improper(&ORDER_2, &z_vec, &SIG) {
                log::debug!("Located an inversion centre.");
                self.point_group = Some("Ci".to_owned());
                log::debug!(
                    "Point group determined: {}",
                    self.point_group.as_ref().unwrap()
                );
                self.add_improper(
                    ORDER_2.clone(),
                    z_vec,
                    false,
                    SIG.clone(),
                    None,
                    presym.dist_threshold,
                );
                self.add_improper(
                    ORDER_2.clone(),
                    z_vec,
                    true,
                    SIG.clone(),
                    None,
                    presym.dist_threshold,
                );
            } else {
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
                            count_sigma += self.add_improper(
                                ORDER_1.clone(),
                                normal,
                                false,
                                SIG.clone(),
                                None,
                                presym.dist_threshold,
                            ) as u32;
                        }
                    }

                    if count_sigma == 0
                        && matches!(
                            presym.rotational_symmetry,
                            RotationalSymmetry::AsymmetricPlanar
                        )
                    {
                        log::debug!(
                            "Planar molecule based on MoIs but no σ found from SEA groups."
                        );
                        log::debug!("Locating the planar mirror plane based on MoIs...");
                        assert!(presym.check_improper(&ORDER_1, &_principal_axes[2], &SIG));
                        assert!(self.add_improper(
                            ORDER_1.clone(),
                            _principal_axes[2],
                            false,
                            SIG.clone(),
                            None,
                            presym.dist_threshold,
                        ));
                        log::debug!("Located one planar mirror plane based on MoIs.");
                        count_sigma += 1;
                    }

                    log::debug!("Located {} σ.", count_sigma);
                    if count_sigma > 0 {
                        assert_eq!(count_sigma, 1);
                        let old_sigmas = self.improper_elements.remove(&ORDER_1).unwrap();
                        assert_eq!(old_sigmas.len(), 1);
                        let old_sigma = old_sigmas.into_iter().next().unwrap();
                        self.add_improper(
                            ORDER_1.clone(),
                            old_sigma.axis,
                            false,
                            SIG.clone(),
                            Some("h".to_owned()),
                            presym.dist_threshold,
                        );
                        self.add_improper(
                            ORDER_1.clone(),
                            old_sigma.axis,
                            true,
                            SIG.clone(),
                            Some("h".to_owned()),
                            presym.dist_threshold,
                        );

                        self.point_group = Some("Cs".to_owned());
                        log::debug!(
                            "Point group determined: {}",
                            self.point_group.as_ref().unwrap()
                        );
                    } else {
                        self.add_proper(
                            ORDER_1.clone(),
                            self.proper_elements[&ORDER_1].iter().next().unwrap().axis,
                            true,
                            presym.dist_threshold
                        );
                        self.point_group = Some("C1".to_owned());
                        log::debug!(
                            "Point group determined: {}",
                            self.point_group.as_ref().unwrap()
                        );
                    }
                }
            }
        }
    }
}
