use super::{PreSymmetry, Symmetry};
use crate::aux::atom::Atom;
use crate::aux::molecule::Molecule;
use crate::rotsym::RotationalSymmetry;
use crate::symmetry::symmetry_element::{
    ElementOrder, SymmetryElement, INV, ORDER_1, ORDER_2, SIG,
};
use approx;
use divisors;
use itertools::Itertools;
use log;
use nalgebra::Vector3;

impl Symmetry {
    /// Performs point-group detection analysis for a symmetric-top molecule.
    ///
    /// The possible symmetric top point groups are:
    ///
    /// * $\mathcal{C}_{n}$ (except $\mathcal{C}_1$ and $\mathcal{C}_2$),
    /// * $\mathcal{C}_{nh}$ (except $\mathcal{C}_{2h}$),
    /// * $\mathcal{C}_{nv}$ (except $\mathcal{C}_{2v}$),
    /// * $\mathcal{D}_{n}$ (except $\mathcal{D}_{2}$),
    /// * $\mathcal{D}_{nh}$ (except $\mathcal{D}_{2h}$),
    /// * $\mathcal{D}_{nd}$, and
    /// * $\mathcal{S}_{2n}$.
    ///
    /// The exceptions are all Abelian groups (but not all Abelian groups are the
    /// exceptions, *e.g.* $\mathcal{S}_{2n}$.
    ///
    /// # Arguments
    ///
    /// * `presym` - A pre-symmetry-analysis struct containing information about
    /// the molecular system.
    pub fn analyse_symmetric(&mut self, presym: &PreSymmetry) {
        let (_mois, _principal_axes) = presym.molecule.calc_moi();

        assert!(matches!(
            presym.rotational_symmetry,
            RotationalSymmetry::ProlateNonLinear
                | RotationalSymmetry::OblateNonPlanar
                | RotationalSymmetry::OblatePlanar
        ));

        _search_proper_rotations(presym, self, false);

        // Classify into point groups
        let max_ord = self.get_max_proper_order();
        let max_ord_u32 = match max_ord {
            ElementOrder::Int(ord_i) => Some(ord_i),
            ElementOrder::Float(ord_f, ord_f_thresh) => {
                if approx::relative_eq!(
                    ord_f,
                    ord_f.round(),
                    epsilon = ord_f_thresh,
                    max_relative = ord_f_thresh
                ) && ord_f > 0.0
                {
                    Some(ord_f as u32)
                } else {
                    None
                }
            }
            _ => None,
        }
        .unwrap();
        let dihedral = {
            if self.proper_elements.contains_key(&ORDER_2) {
                if max_ord > ORDER_2 {
                    assert_eq!(self.proper_elements[&max_ord].len(), 1);
                    let principal_axis = self.proper_elements[&max_ord].iter().next().unwrap().axis;
                    let n_c2_perp = self.proper_elements[&ORDER_2]
                        .iter()
                        .filter(|c2_ele| {
                            c2_ele.axis.dot(&principal_axis).abs() < presym.dist_threshold
                        })
                        .count();
                    ElementOrder::Int(n_c2_perp as u32) == max_ord
                } else {
                    max_ord == ORDER_2 && self.proper_elements[&ORDER_2].len() == 3
                }
            } else {
                false
            }
        };

        if dihedral {
            // Dihedral family
            log::debug!("Dihedral family.");

            // Principal axis is also a generator.
            self.add_proper(
                max_ord.clone(),
                self.proper_elements[&max_ord].iter().next().unwrap().axis,
                true,
                presym.dist_threshold,
            );

            let principal_element = self.proper_elements[&max_ord].iter().next().unwrap();
            let c2_element = self.proper_elements[&ORDER_2]
                .iter()
                .find_map(|c2_ele| {
                    if c2_ele.axis.dot(&principal_element.axis).abs() < presym.dist_threshold {
                        Some(c2_ele)
                    } else {
                        None
                    }
                })
                .unwrap();
            self.add_proper(
                ORDER_2.clone(),
                c2_element.axis,
                true,
                presym.dist_threshold,
            );

            let principal_element = self.proper_elements[&max_ord].iter().next().unwrap();
            if presym.check_improper(&ORDER_1, &principal_element.axis, &SIG) {
                // Dnh (n > 2)
                assert!(max_ord > ORDER_2);
                log::debug!("Located σh.");
                self.point_group = Some(format!("D{max_ord}h"));
                log::debug!(
                    "Point group determined: {}",
                    self.point_group.as_ref().unwrap()
                );
                self.add_improper(
                    ORDER_1.clone(),
                    principal_element.axis,
                    false,
                    SIG.clone(),
                    Some("h".to_owned()),
                    presym.dist_threshold,
                );
                let principal_element = self.proper_elements[&max_ord].iter().next().unwrap();
                self.add_improper(
                    ORDER_1.clone(),
                    principal_element.axis,
                    true,
                    SIG.clone(),
                    Some("h".to_owned()),
                    presym.dist_threshold,
                );

                // Locate all other mirror planes and improper axes
                // We take all the other mirror planes to be σv.
                // It's really not worth trying to classify them into σv and σd,
                // as this classification is more conventional than fundamental.
                let non_id_c_elements =
                    self.proper_elements.values().fold(vec![], |acc, c_eles| {
                        acc.into_iter()
                            .chain(c_eles.iter().filter(|ele| ele.order != ORDER_1).cloned())
                            .collect()
                    });
                if max_ord_u32 % 2 == 0 {
                    // Dnh, n even, an inversion centre is expected.
                    let z_vec = Vector3::new(0.0, 0.0, 1.0);
                    assert!(presym.check_improper(&ORDER_2, &z_vec, &SIG));
                    self.add_improper(
                        ORDER_2.clone(),
                        z_vec,
                        false,
                        SIG.clone(),
                        None,
                        presym.dist_threshold,
                    );

                    for c_element in non_id_c_elements.into_iter() {
                        let principal_element =
                            self.proper_elements[&max_ord].iter().next().unwrap();
                        let sigma_symbol = _deduce_sigma_symbol(
                            &c_element.axis,
                            principal_element,
                            presym.dist_threshold,
                            false,
                        );
                        // iCn
                        assert!(presym.check_improper(&c_element.order, &c_element.axis, &INV));
                        self.add_improper(
                            c_element.order.clone(),
                            c_element.axis.clone(),
                            false,
                            INV.clone(),
                            sigma_symbol,
                            presym.dist_threshold,
                        );
                    }
                } else {
                    // Dnh, n odd, only σh is expected.
                    let sigma_h = self
                        .get_sigma_elements("h")
                        .unwrap()
                        .into_iter()
                        .next()
                        .unwrap()
                        .clone();
                    _add_sigmahcn(self, &sigma_h, non_id_c_elements, &presym);
                }
            }
            // end Dnh
            else {
                // Dnd
                let sigmad_axes = self.proper_elements[&ORDER_2].iter().combinations(2).fold(
                    vec![],
                    |mut acc, c2_elements| {
                        let c2_axis_i = c2_elements[0].axis;
                        let c2_axis_j = c2_elements[1].axis;
                        let axis_p = (c2_axis_i + c2_axis_j).normalize();
                        if presym.check_improper(&ORDER_1, &axis_p, &SIG) {
                            acc.push(axis_p);
                        };
                        let axis_m = (c2_axis_i - c2_axis_j).normalize();
                        if presym.check_improper(&ORDER_1, &axis_m, &SIG) {
                            acc.push(axis_m);
                        };
                        acc
                    },
                );

                let mut count_sigmad = 0u32;
                for sigmad_axis in sigmad_axes.into_iter() {
                    count_sigmad += self.add_improper(
                        ORDER_1.clone(),
                        sigmad_axis,
                        false,
                        SIG.clone(),
                        Some("d".to_owned()),
                        presym.dist_threshold,
                    ) as u32;
                    if count_sigmad == max_ord_u32 {
                        break;
                    }
                }
                log::debug!("Located {} σd.", count_sigmad);

                if count_sigmad == max_ord_u32 {
                    // Dnd
                    let sigmad_axis = self
                        .get_sigma_elements("d")
                        .unwrap()
                        .iter()
                        .next()
                        .unwrap()
                        .axis;
                    self.add_improper(
                        ORDER_1.clone(),
                        sigmad_axis,
                        true,
                        SIG.clone(),
                        Some("d".to_owned()),
                        presym.dist_threshold,
                    );
                    self.point_group = Some(format!("D{max_ord}d"));
                    log::debug!(
                        "Point group determined: {}",
                        self.point_group.as_ref().unwrap()
                    );

                    if max_ord_u32 % 2 == 0 {
                        // Dnd, n even, only σd planes are present.
                        let non_id_c_elements =
                            self.proper_elements.values().fold(vec![], |acc, c_eles| {
                                acc.into_iter()
                                    .chain(
                                        c_eles.iter().filter(|ele| ele.order != ORDER_1).cloned(),
                                    )
                                    .collect()
                            });
                        for c_element in non_id_c_elements.into_iter() {
                            let double_order =
                                ElementOrder::new(2.0 * c_element.order.to_float(), f64::EPSILON);
                            if presym.check_improper(&double_order, &c_element.axis, &SIG) {
                                self.add_improper(
                                    double_order,
                                    c_element.axis,
                                    false,
                                    SIG.clone(),
                                    None,
                                    presym.dist_threshold,
                                );
                            }
                        }
                    } else {
                        // Dnd, n odd, an inversion centre is expected.
                        let vec_z = Vector3::new(0.0, 0.0, 1.0);
                        assert!(presym.check_improper(&ORDER_2, &vec_z, &SIG));
                        self.add_improper(
                            ORDER_2.clone(),
                            vec_z,
                            false,
                            SIG.clone(),
                            None,
                            presym.dist_threshold,
                        );
                        let non_id_c_elements =
                            self.proper_elements.values().fold(vec![], |acc, c_eles| {
                                acc.into_iter()
                                    .chain(
                                        c_eles.iter().filter(|ele| ele.order != ORDER_1).cloned(),
                                    )
                                    .collect()
                            });
                        for c_element in non_id_c_elements.into_iter() {
                            let principal_element =
                                self.proper_elements[&max_ord].iter().next().unwrap();
                            let sigma_symbol = _deduce_sigma_symbol(
                                &c_element.axis,
                                principal_element,
                                presym.dist_threshold,
                                true, // sigma_v forced to become sigma_d
                            );
                            assert!(presym.check_improper(&c_element.order, &c_element.axis, &INV));
                            self.add_improper(
                                c_element.order.clone(),
                                c_element.axis.clone(),
                                false,
                                INV.clone(),
                                sigma_symbol,
                                presym.dist_threshold,
                            );
                        }
                    }
                } else {
                    // Dn (n > 2)
                    self.point_group = Some(format!("D{max_ord}"));
                    log::debug!(
                        "Point group determined: {}",
                        self.point_group.as_ref().unwrap()
                    );
                }
            }
        } else {
            // Non-dihedral family
            log::debug!("Non-dihedral family.");

            // Locate mirror planes
            let mut count_sigma = 0u32;
            let sea_groups = &presym.sea_groups;
            for sea_group in sea_groups.iter() {
                if count_sigma == max_ord_u32 {
                    break;
                }
                if sea_group.len() < 2 {
                    continue;
                }
                for atom2s in sea_group.iter().combinations(2) {
                    if count_sigma == max_ord_u32 {
                        break;
                    }
                    let principal_element = self.proper_elements[&max_ord].iter().next().unwrap();
                    let normal =
                        (atom2s[0].coordinates.coords - atom2s[1].coordinates.coords).normalize();
                    if presym.check_improper(&ORDER_1, &normal, &SIG) {
                        let sigma_symbol = _deduce_sigma_symbol(
                            &normal,
                            principal_element,
                            presym.dist_threshold,
                            false,
                        );
                        count_sigma += self.add_improper(
                            ORDER_1.clone(),
                            normal,
                            false,
                            SIG.clone(),
                            sigma_symbol,
                            presym.dist_threshold,
                        ) as u32;
                    }
                }
            }

            if count_sigma == max_ord_u32 {
                if max_ord_u32 > 1 {
                    // Cnv (n > 2)
                    log::debug!("Found {} σv planes.", count_sigma);
                    self.point_group = Some(format!("C{max_ord}v"));
                    log::debug!(
                        "Point group determined: {}",
                        self.point_group.as_ref().unwrap()
                    );
                    let principal_axis = self.proper_elements[&max_ord].iter().next().unwrap().axis;
                    self.add_proper(max_ord.clone(), principal_axis, true, presym.dist_threshold);
                    let sigma_v_normal = self
                        .get_sigma_elements("v")
                        .unwrap()
                        .iter()
                        .next()
                        .unwrap()
                        .axis;
                    self.add_improper(
                        ORDER_1.clone(),
                        sigma_v_normal,
                        true,
                        SIG.clone(),
                        Some("v".to_owned()),
                        presym.dist_threshold,
                    );
                } else {
                    // Cs
                    log::debug!("Found {} σ planes.", count_sigma);
                    self.point_group = Some("Cs".to_owned());
                    log::debug!(
                        "Point group determined: {}",
                        self.point_group.as_ref().unwrap()
                    );
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
                }
            } else if {
                let principal_axis = self.proper_elements[&max_ord].iter().next().unwrap().axis;
                presym.check_improper(&ORDER_1, &principal_axis, &SIG)
            } {
                // Cnh (n > 2)
                assert_eq!(count_sigma, 1);
                log::debug!("Found no σv planes but one σh plane.");
                self.point_group = Some(format!("C{max_ord}h"));
                let principal_axis = self.proper_elements[&max_ord].iter().next().unwrap().axis;
                self.add_proper(max_ord.clone(), principal_axis, true, presym.dist_threshold);
                self.add_improper(
                    ORDER_1.clone(),
                    principal_axis,
                    true,
                    SIG.clone(),
                    Some("h".to_owned()),
                    presym.dist_threshold,
                );

                // Locate the remaining improper elements
                let non_id_c_elements =
                    self.proper_elements.values().fold(vec![], |acc, c_eles| {
                        acc.into_iter()
                            .chain(c_eles.iter().filter(|ele| ele.order != ORDER_1).cloned())
                            .collect()
                    });
                if max_ord_u32 % 2 == 0 {
                    // Cnh, n even, an inversion centre is expected.
                    let vec_z = Vector3::new(0.0, 0.0, 1.0);
                    assert!(presym.check_improper(&ORDER_2, &vec_z, &SIG));
                    self.add_improper(
                        ORDER_2.clone(),
                        vec_z,
                        false,
                        SIG.clone(),
                        None,
                        presym.dist_threshold,
                    );
                    for c_element in non_id_c_elements.into_iter() {
                        let principal_element =
                            self.proper_elements[&max_ord].iter().next().unwrap();
                        let sigma_symbol = _deduce_sigma_symbol(
                            &c_element.axis,
                            principal_element,
                            presym.dist_threshold,
                            false,
                        );
                        // iCn
                        assert!(presym.check_improper(&c_element.order, &c_element.axis, &INV));
                        self.add_improper(
                            c_element.order.clone(),
                            c_element.axis.clone(),
                            false,
                            INV.clone(),
                            sigma_symbol,
                            presym.dist_threshold,
                        );
                    }
                } else {
                    // Cnh, n odd, only σh is present.
                    let sigma_h = self
                        .get_sigma_elements("h")
                        .unwrap()
                        .into_iter()
                        .next()
                        .unwrap()
                        .clone();
                    _add_sigmahcn(self, &sigma_h, non_id_c_elements, &presym);
                }
            } else if {
                let double_max_ord = ElementOrder::new(2.0 * max_ord.to_float(), f64::EPSILON);
                let principal_axis = self.proper_elements[&max_ord].iter().next().unwrap().axis;
                presym.check_improper(&double_max_ord, &principal_axis, &SIG)
            } {
                // S2n
                let double_max_ord = ElementOrder::new(2.0 * max_ord.to_float(), f64::EPSILON);
                self.point_group = Some(format!("S{double_max_ord}"));
                log::debug!(
                    "Point group determined: {}",
                    self.point_group.as_ref().unwrap()
                );
                let principal_axis = self.proper_elements[&max_ord].iter().next().unwrap().axis;
                self.add_improper(
                    double_max_ord.clone(),
                    principal_axis,
                    false,
                    SIG.clone(),
                    None,
                    presym.dist_threshold,
                );
                self.add_improper(
                    double_max_ord,
                    principal_axis,
                    true,
                    SIG.clone(),
                    None,
                    presym.dist_threshold,
                );

                // Locate the remaining improper symmetry elements
                if max_ord_u32 % 2 != 0 {
                    // Odd rotation sub groups, an inversion centre is expected.
                    let vec_z = Vector3::new(0.0, 0.0, 1.0);
                    assert!(presym.check_improper(&ORDER_2, &vec_z, &SIG));
                    self.add_improper(
                        ORDER_2.clone(),
                        vec_z,
                        false,
                        SIG.clone(),
                        None,
                        presym.dist_threshold,
                    );
                }
            } else {
                // Cn (n > 2)
                self.point_group = Some(format!("C{max_ord}"));
                log::debug!(
                    "Point group determined: {}",
                    self.point_group.as_ref().unwrap()
                );
                let principal_axis = self.proper_elements[&max_ord].iter().next().unwrap().axis;
                self.add_proper(max_ord, principal_axis, true, presym.dist_threshold);
            };
        }
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
/// Abelian point groups for which the highest possible rotation order is $2$
/// and there can be at most three $\mathcal{C}_2$ axes.
///
/// # Returns
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

/// Determines the mirror-plane symbol given a principal axis.
///
/// Arguments:
///
/// * `sigma_axis` - The normalised normal vector of a mirror plane.
/// * `principal_axis` - The normalised principal rotation axis.
/// * `thresh` - Threshold for comparisons.
///
/// Returns:
///
/// The mirror-plane symbol.
fn _deduce_sigma_symbol(
    sigma_axis: &Vector3<f64>,
    principal_element: &SymmetryElement,
    thresh: f64,
    force_d: bool,
) -> Option<String> {
    let sigma_symbol = if approx::relative_eq!(
        principal_element.axis.dot(&sigma_axis).abs(),
        0.0,
        epsilon = thresh,
        max_relative = thresh
    ) && principal_element.order != ORDER_1
    {
        // Vertical plane containing principal axis
        if force_d {
            Some("d".to_owned())
        } else {
            Some("v".to_owned())
        }
    } else if approx::relative_eq!(
        principal_element.axis.cross(&sigma_axis).norm(),
        0.0,
        epsilon = thresh,
        max_relative = thresh
    ) && principal_element.order != ORDER_1
    {
        // Horizontal plane perpendicular to principal axis
        Some("h".to_owned())
    } else {
        None
    };
    sigma_symbol
}

/// Adds improper elements constructed as a product between a $\sigma_h$ and a
/// rotation axis.
///
/// The constructed improper elements will be added to `sym`.
///
/// # Arguments
///
/// * `sym` - A symmetry struct to store the improper rotation elements found.
/// * `sigma_h` - A $\sigma_h$ mirror plane.
/// * `non_id_c_elements` - A vector of non-identity rotation elements to
/// consider.
/// * `presym` - A pre-symmetry-analysis struct containing information about
/// the molecular system.
fn _add_sigmahcn(
    sym: &mut Symmetry,
    sigma_h: &SymmetryElement,
    non_id_c_elements: Vec<SymmetryElement>,
    presym: &PreSymmetry,
) {
    assert!(sigma_h.is_mirror_plane());
    for c_element in non_id_c_elements.into_iter() {
        if approx::relative_eq!(
            c_element.axis.cross(&sigma_h.axis).norm(),
            0.0,
            epsilon = presym.dist_threshold,
            max_relative = presym.dist_threshold
        ) {
            // Cn is orthogonal to σh. The product Cn * σh is Sn.
            log::debug!("Cn is orthogonal to σh.");
            assert!(presym.check_improper(&c_element.order, &c_element.axis, &SIG));
            let sigma_symbol = if c_element.order == ORDER_1 {
                Some("h".to_owned())
            } else {
                None
            };
            sym.add_improper(
                c_element.order,
                c_element.axis,
                false,
                SIG.clone(),
                sigma_symbol,
                presym.dist_threshold,
            );
        } else {
            // Cn is C2 and is contained in σh.
            // The product σh * C2 is a σv plane.
            approx::assert_relative_eq!(
                c_element.axis.dot(&sigma_h.axis).abs(),
                0.0,
                epsilon = presym.dist_threshold,
                max_relative = presym.dist_threshold
            );
            assert_eq!(c_element.order, ORDER_2);
            log::debug!("Cn is C2 and must therefore be contained in σh.");
            let s_axis = c_element.axis.cross(&sigma_h.axis).normalize();
            assert!(presym.check_improper(&ORDER_1, &s_axis, &SIG));
            sym.add_improper(
                ORDER_1.clone(),
                s_axis,
                false,
                SIG.clone(),
                Some("v".to_owned()),
                presym.dist_threshold,
            );
        }
    }
}
