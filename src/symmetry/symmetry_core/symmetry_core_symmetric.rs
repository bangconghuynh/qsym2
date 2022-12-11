use super::{PreSymmetry, Symmetry};
use crate::rotsym::RotationalSymmetry;
use crate::symmetry::symmetry_core::_search_proper_rotations;
use crate::symmetry::symmetry_element::{SymmetryElement, INV, SIG};
use crate::symmetry::symmetry_element_order::{ElementOrder, ORDER_1, ORDER_2};
use approx;
use itertools::Itertools;
use log;
use nalgebra::Vector3;

impl Symmetry {
    /// Performs point-group detection analysis for a symmetric-top molecule.
    ///
    /// The possible symmetric top point groups are:
    ///
    /// * $`\mathcal{C}_{n}`$ (except $`\mathcal{C}_1`$ and $`\mathcal{C}_2`$),
    /// * $`\mathcal{C}_{nv}`$ (except $`\mathcal{C}_{2v}`$),
    /// * $`\mathcal{C}_{nh}`$ (except $`\mathcal{C}_{2h}`$),
    /// * $`\mathcal{D}_{n}`$ (except $`\mathcal{D}_{2}`$),
    /// * $`\mathcal{D}_{nh}`$ (except $`\mathcal{D}_{2h}`$),
    /// * $`\mathcal{D}_{nd}`$, and
    /// * $`\mathcal{S}_{2n}`$.
    ///
    /// The exceptions are all Abelian groups (but not all Abelian groups are the
    /// exceptions, *e.g.* $`\mathcal{S}_{2n}`$).
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
                max_ord,
                self.proper_elements[&max_ord].iter().next().unwrap().axis,
                true,
                presym.dist_threshold,
            );

            let principal_element = self.proper_elements[&max_ord].iter().next().unwrap();
            let c2_element = self.proper_elements[&ORDER_2]
                .iter()
                .find(|c2_ele| {
                    c2_ele.axis.dot(&principal_element.axis).abs() < presym.dist_threshold
                })
                .unwrap();
            self.add_proper(
                ORDER_2,
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
                    ORDER_1,
                    principal_element.axis,
                    false,
                    SIG.clone(),
                    Some("h".to_owned()),
                    presym.dist_threshold,
                );
                let principal_element = self.proper_elements[&max_ord].iter().next().unwrap();
                self.add_improper(
                    ORDER_1,
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
                            .chain(c_eles.iter().filter(|ele| ele.proper_order != ORDER_1).cloned())
                            .collect()
                    });
                if max_ord_u32 % 2 == 0 {
                    // Dnh, n even, an inversion centre is expected.
                    let z_vec = Vector3::new(0.0, 0.0, 1.0);
                    assert!(presym.check_improper(&ORDER_2, &z_vec, &SIG));
                    self.add_improper(
                        ORDER_2,
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
                        assert!(presym.check_improper(&c_element.proper_order, &c_element.axis, &INV));
                        self.add_improper(
                            c_element.proper_order,
                            c_element.axis,
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
                    _add_sigmahcn(self, &sigma_h, non_id_c_elements, presym);
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
                        ORDER_1,
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
                        ORDER_1,
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
                                        c_eles.iter().filter(|ele| ele.proper_order != ORDER_1).cloned(),
                                    )
                                    .collect()
                            });
                        for c_element in non_id_c_elements.into_iter() {
                            let double_order =
                                ElementOrder::new(2.0 * c_element.proper_order.to_float(), f64::EPSILON);
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
                            ORDER_2,
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
                                        c_eles.iter().filter(|ele| ele.proper_order != ORDER_1).cloned(),
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
                            assert!(presym.check_improper(&c_element.proper_order, &c_element.axis, &INV));
                            self.add_improper(
                                c_element.proper_order,
                                c_element.axis,
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
                            ORDER_1,
                            normal,
                            false,
                            SIG.clone(),
                            sigma_symbol,
                            presym.dist_threshold,
                        ) as u32;
                    }
                }
            }

            #[allow(clippy::blocks_in_if_conditions)]
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
                    self.add_proper(max_ord, principal_axis, true, presym.dist_threshold);
                    let sigma_v_normal = self
                        .get_sigma_elements("v")
                        .unwrap()
                        .iter()
                        .next()
                        .unwrap()
                        .axis;
                    self.add_improper(
                        ORDER_1,
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
                self.add_proper(max_ord, principal_axis, true, presym.dist_threshold);
                self.add_improper(
                    ORDER_1,
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
                            .chain(c_eles.iter().filter(|ele| ele.proper_order != ORDER_1).cloned())
                            .collect()
                    });
                if max_ord_u32 % 2 == 0 {
                    // Cnh, n even, an inversion centre is expected.
                    let vec_z = Vector3::new(0.0, 0.0, 1.0);
                    assert!(presym.check_improper(&ORDER_2, &vec_z, &SIG));
                    self.add_improper(
                        ORDER_2,
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
                        assert!(presym.check_improper(&c_element.proper_order, &c_element.axis, &INV));
                        self.add_improper(
                            c_element.proper_order,
                            c_element.axis,
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
                    _add_sigmahcn(self, &sigma_h, non_id_c_elements, presym);
                }
            } else if {
                let double_max_ord = ElementOrder::new(2.0 * max_ord.to_float(), f64::EPSILON);
                let principal_axis = self.proper_elements[&max_ord].iter().next().unwrap().axis;
                presym.check_improper(&double_max_ord, &principal_axis, &SIG)
            } {
                // S2n
                let double_max_ord = ElementOrder::new(2.0 * max_ord.to_float(), f64::EPSILON);
                self.point_group = if double_max_ord == ElementOrder::Int(2) {
                    // S2 is Ci.
                    Some(format!("Ci"))
                } else {
                    Some(format!("S{double_max_ord}"))
                };
                log::debug!(
                    "Point group determined: {}",
                    self.point_group.as_ref().unwrap()
                );
                let principal_axis = self.proper_elements[&max_ord].iter().next().unwrap().axis;
                self.add_improper(
                    double_max_ord,
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
                        ORDER_2,
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
    if approx::relative_eq!(
        principal_element.axis.dot(sigma_axis).abs(),
        0.0,
        epsilon = thresh,
        max_relative = thresh
    ) && principal_element.proper_order != ORDER_1
    {
        // Vertical plane containing principal axis
        if force_d {
            Some("d".to_owned())
        } else {
            Some("v".to_owned())
        }
    } else if approx::relative_eq!(
        principal_element.axis.cross(sigma_axis).norm(),
        0.0,
        epsilon = thresh,
        max_relative = thresh
    ) && principal_element.proper_order != ORDER_1
    {
        // Horizontal plane perpendicular to principal axis
        Some("h".to_owned())
    } else {
        None
    }
}

/// Adds improper elements constructed as a product between a $`\sigma_h`$ and a
/// rotation axis.
///
/// The constructed improper elements will be added to `sym`.
///
/// # Arguments
///
/// * `sym` - A symmetry struct to store the improper rotation elements found.
/// * `sigma_h` - A $`\sigma_h`$ mirror plane.
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
            assert!(presym.check_improper(&c_element.proper_order, &c_element.axis, &SIG));
            let sigma_symbol = if c_element.proper_order == ORDER_1 {
                Some("h".to_owned())
            } else {
                None
            };
            sym.add_improper(
                c_element.proper_order,
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
            assert_eq!(c_element.proper_order, ORDER_2);
            log::debug!("Cn is C2 and must therefore be contained in σh.");
            let s_axis = c_element.axis.cross(&sigma_h.axis).normalize();
            assert!(presym.check_improper(&ORDER_1, &s_axis, &SIG));
            sym.add_improper(
                ORDER_1,
                s_axis,
                false,
                SIG.clone(),
                Some("v".to_owned()),
                presym.dist_threshold,
            );
        }
    }
}
