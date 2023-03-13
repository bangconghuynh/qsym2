use std::collections::HashMap;

use approx;
use indexmap::IndexSet;
use itertools::Itertools;
use log;
use nalgebra::Vector3;

use crate::rotsym::RotationalSymmetry;
use crate::symmetry::symmetry_core::_search_proper_rotations;
use crate::symmetry::symmetry_element::{SymmetryElement, INV, ROT, SIG, TRROT, TRSIG};
use crate::symmetry::symmetry_element_order::{ElementOrder, ORDER_1, ORDER_2};
use crate::symmetry::symmetry_symbols::deduce_sigma_symbol;

use super::{PreSymmetry, Symmetry};

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
    pub fn analyse_symmetric(&mut self, presym: &PreSymmetry, tr: bool) {
        let (_mois, _principal_axes) = presym.molecule.calc_moi();

        assert!(matches!(
            presym.rotational_symmetry,
            RotationalSymmetry::ProlateNonLinear
                | RotationalSymmetry::OblateNonPlanar
                | RotationalSymmetry::OblatePlanar
        ));

        _search_proper_rotations(presym, self, false, tr);

        // Classify into point groups
        let max_ord = self.get_max_proper_order();
        let max_ord_u32 = match max_ord {
            ElementOrder::Int(ord_i) => Some(ord_i),
            ElementOrder::Inf => None,
        }
        .unwrap_or_else(|| panic!("`{max_ord}` has an unexpected order value."));
        let dihedral = {
            if self
                .get_elements(&ROT)
                .unwrap_or(&HashMap::new())
                .contains_key(&ORDER_2)
                || self
                    .get_elements(&TRROT)
                    .unwrap_or(&HashMap::new())
                    .contains_key(&ORDER_2)
            {
                if max_ord > ORDER_2 {
                    assert_eq!(
                        self.get_proper(&max_ord)
                            .unwrap_or_else(|| panic!(
                                "No proper elements of order `{max_ord}` found."
                            ))
                            .len(),
                        1
                    );

                    let principal_axis = self.get_proper_principal_element().raw_axis;
                    let n_c2_perp = self
                        .get_proper(&ORDER_2)
                        .unwrap_or_else(|| panic!("No proper elements of order `{ORDER_2}` found."))
                        .iter()
                        .filter(|c2_ele| {
                            c2_ele.raw_axis.dot(&principal_axis).abs() < presym.dist_threshold
                        })
                        .count();
                    ElementOrder::Int(
                        n_c2_perp.try_into().unwrap_or_else(|_| {
                            panic!("Unable to convert `{n_c2_perp}` to `u32`.")
                        }),
                    ) == max_ord
                } else {
                    max_ord == ORDER_2
                        && self
                            .get_proper(&ORDER_2)
                            .unwrap_or_else(|| {
                                panic!("No proper elements of order `{ORDER_2}` found.")
                            })
                            .len()
                            == 3
                }
            } else {
                false
            }
        };

        if dihedral {
            // Dihedral family
            log::debug!("Dihedral family.");

            // Principal axis is also a generator.
            let principal_element = self.get_proper_principal_element().clone();
            self.add_proper(
                max_ord,
                principal_element.raw_axis,
                true,
                presym.dist_threshold,
                principal_element.contains_time_reversal(),
            );

            // A C2 axis perpendicular to the principal axis is also a generator.
            let perp_c2_element = &(*self
                .get_proper(&ORDER_2)
                .expect("No C2 elements found.")
                .iter()
                .find(|c2_ele| {
                    c2_ele.raw_axis.dot(&principal_element.raw_axis).abs() < presym.dist_threshold
                })
                .expect("No C2 axes perpendicular to the principal axis found."))
            .clone();
            self.add_proper(
                ORDER_2,
                perp_c2_element.raw_axis,
                true,
                presym.dist_threshold,
                perp_c2_element.contains_time_reversal(),
            );

            if let Some(improper_kind) =
                presym.check_improper(&ORDER_1, &principal_element.raw_axis, &SIG, tr)
            {
                // Dnh (n >= 2)
                assert!(max_ord >= ORDER_2);
                log::debug!("Located σh.");
                self.set_group_name(format!("D{max_ord}h"));
                self.add_improper(
                    ORDER_1,
                    principal_element.raw_axis,
                    false,
                    SIG.clone(),
                    Some("h".to_owned()),
                    presym.dist_threshold,
                    improper_kind.contains_time_reversal(),
                );
                self.add_improper(
                    ORDER_1,
                    principal_element.raw_axis,
                    true,
                    SIG.clone(),
                    Some("h".to_owned()),
                    presym.dist_threshold,
                    improper_kind.contains_time_reversal(),
                );

                // Locate all other mirror planes and improper axes
                // We take all the other mirror planes to be σv.
                // It's really not worth trying to classify them into σv and σd,
                // as this classification is more conventional than fundamental.
                let non_id_c_elements = self
                    .get_elements(&ROT)
                    .unwrap_or(&HashMap::new())
                    .values()
                    .chain(
                        self.get_elements(&TRROT)
                            .unwrap_or(&HashMap::new())
                            .values(),
                    )
                    .fold(vec![], |acc, c_eles| {
                        acc.into_iter()
                            .chain(
                                c_eles
                                    .iter()
                                    .filter(|ele| *ele.raw_proper_order() != ORDER_1)
                                    .cloned(),
                            )
                            .collect()
                    });
                if max_ord_u32 % 2 == 0 {
                    // Dnh, n even, an inversion centre is expected.
                    let z_vec = Vector3::new(0.0, 0.0, 1.0);
                    let inversion_check = presym.check_improper(&ORDER_2, &z_vec, &SIG, tr);
                    assert!(inversion_check.is_some());
                    self.add_improper(
                        ORDER_2,
                        z_vec,
                        false,
                        SIG.clone(),
                        None,
                        presym.dist_threshold,
                        inversion_check
                            .expect("Expected inversion centre not found.")
                            .contains_time_reversal(),
                    );

                    for c_element in non_id_c_elements {
                        let principal_element = self.get_proper_principal_element();
                        let sigma_symbol = deduce_sigma_symbol(
                            &c_element.raw_axis,
                            principal_element,
                            presym.dist_threshold,
                            false,
                        );
                        // iCn
                        let icn_check = presym.check_improper(
                            c_element.raw_proper_order(),
                            &c_element.raw_axis,
                            &INV,
                            tr,
                        );
                        assert!(icn_check.is_some());
                        self.add_improper(
                            *c_element.raw_proper_order(),
                            c_element.raw_axis,
                            false,
                            INV.clone(),
                            sigma_symbol,
                            presym.dist_threshold,
                            icn_check
                                .expect("Expected iCn not found.")
                                .contains_time_reversal(),
                        );
                    }
                } else {
                    // Dnh, n odd, only σh is expected.
                    let sigma_h = self
                        .get_sigma_elements("h")
                        .expect("No σh found.")
                        .into_iter()
                        .next()
                        .expect("No σh found.")
                        .clone();
                    _add_sigmahcn(self, &sigma_h, non_id_c_elements, presym, tr);
                }
            }
            // end Dnh
            else {
                // Dnd
                let sigmad_axes = self
                    .get_elements(&ROT)
                    .unwrap_or(&HashMap::new())
                    .get(&ORDER_2)
                    .unwrap_or(&IndexSet::new())
                    .iter()
                    .chain(
                        self.get_elements(&TRROT)
                            .unwrap_or(&HashMap::new())
                            .get(&ORDER_2)
                            .unwrap_or(&IndexSet::new()),
                    )
                    .combinations(2)
                    .fold(vec![], |mut acc, c2_elements| {
                        let c2_axis_i = c2_elements[0].raw_axis;
                        let c2_axis_j = c2_elements[1].raw_axis;
                        let axis_p = (c2_axis_i + c2_axis_j).normalize();
                        if let Some(improper_kind) =
                            presym.check_improper(&ORDER_1, &axis_p, &SIG, tr)
                        {
                            acc.push((axis_p, improper_kind.contains_time_reversal()));
                        };
                        let axis_m = (c2_axis_i - c2_axis_j).normalize();
                        if let Some(improper_kind) =
                            presym.check_improper(&ORDER_1, &axis_m, &SIG, tr)
                        {
                            acc.push((axis_m, improper_kind.contains_time_reversal()));
                        };
                        acc
                    });

                let mut count_sigmad = 0u32;
                for (sigmad_axis, sigmad_axis_tr) in sigmad_axes {
                    count_sigmad += u32::from(self.add_improper(
                        ORDER_1,
                        sigmad_axis,
                        false,
                        SIG.clone(),
                        Some("d".to_owned()),
                        presym.dist_threshold,
                        sigmad_axis_tr,
                    ));
                    if count_sigmad == max_ord_u32 {
                        break;
                    }
                }
                log::debug!("Located {} σd.", count_sigmad);

                if count_sigmad == max_ord_u32 {
                    // Dnd
                    let sigmads = self.get_sigma_elements("d").expect("No σd found.");
                    let sigmad = sigmads.iter().next().expect("No σd found.");
                    self.add_improper(
                        ORDER_1,
                        sigmad.raw_axis,
                        true,
                        SIG.clone(),
                        Some("d".to_owned()),
                        presym.dist_threshold,
                        sigmad.contains_time_reversal(),
                    );
                    self.set_group_name(format!("D{max_ord}d"));

                    if max_ord_u32 % 2 == 0 {
                        // Dnd, n even, only σd planes are present.
                        let non_id_c_elements = self
                            .get_elements(&ROT)
                            .unwrap_or(&HashMap::new())
                            .values()
                            .chain(
                                self.get_elements(&TRROT)
                                    .unwrap_or(&HashMap::new())
                                    .values(),
                            )
                            .fold(vec![], |acc, c_eles| {
                                acc.into_iter()
                                    .chain(
                                        c_eles
                                            .iter()
                                            .filter(|ele| *ele.raw_proper_order() != ORDER_1)
                                            .cloned(),
                                    )
                                    .collect()
                            });
                        for c_element in non_id_c_elements {
                            let double_order = ElementOrder::new(
                                2.0 * c_element.raw_proper_order().to_float(),
                                f64::EPSILON,
                            );
                            if let Some(improper_kind) =
                                presym.check_improper(&double_order, &c_element.raw_axis, &SIG, tr)
                            {
                                self.add_improper(
                                    double_order,
                                    c_element.raw_axis,
                                    false,
                                    SIG.clone(),
                                    None,
                                    presym.dist_threshold,
                                    improper_kind.contains_time_reversal(),
                                );
                            }
                        }
                    } else {
                        // Dnd, n odd, an inversion centre is expected.
                        let vec_z = Vector3::new(0.0, 0.0, 1.0);
                        let inversion_check = presym.check_improper(&ORDER_2, &vec_z, &SIG, tr);
                        assert!(inversion_check.is_some());
                        self.add_improper(
                            ORDER_2,
                            vec_z,
                            false,
                            SIG.clone(),
                            None,
                            presym.dist_threshold,
                            inversion_check
                                .expect("Expected inversion centre not found.")
                                .contains_time_reversal(),
                        );
                        let non_id_c_elements = self
                            .get_elements(&ROT)
                            .unwrap_or(&HashMap::new())
                            .values()
                            .chain(
                                self.get_elements(&TRROT)
                                    .unwrap_or(&HashMap::new())
                                    .values(),
                            )
                            .fold(vec![], |acc, c_eles| {
                                acc.into_iter()
                                    .chain(
                                        c_eles
                                            .iter()
                                            .filter(|ele| *ele.raw_proper_order() != ORDER_1)
                                            .cloned(),
                                    )
                                    .collect()
                            });
                        for c_element in non_id_c_elements {
                            let principal_element = self.get_proper_principal_element();
                            let sigma_symbol = deduce_sigma_symbol(
                                &c_element.raw_axis,
                                principal_element,
                                presym.dist_threshold,
                                true, // sigma_v forced to become sigma_d
                            );
                            let icn_check = presym.check_improper(
                                c_element.raw_proper_order(),
                                &c_element.raw_axis,
                                &INV,
                                tr,
                            );
                            assert!(icn_check.is_some());
                            self.add_improper(
                                *c_element.raw_proper_order(),
                                c_element.raw_axis,
                                false,
                                INV.clone(),
                                sigma_symbol,
                                presym.dist_threshold,
                                icn_check
                                    .expect("Expected iCn not found.")
                                    .contains_time_reversal(),
                            );
                        }
                    }
                } else {
                    // Dn (n > 2)
                    self.set_group_name(format!("D{max_ord}"));
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
                    let principal_element = self.get_proper_principal_element();
                    let normal =
                        (atom2s[0].coordinates.coords - atom2s[1].coordinates.coords).normalize();
                    log::debug!("Checking: {}, {}, {normal}", atom2s[0], atom2s[1]);
                    if let Some(improper_kind) = presym.check_improper(&ORDER_1, &normal, &SIG, tr)
                    {
                        let sigma_symbol = deduce_sigma_symbol(
                            &normal,
                            principal_element,
                            presym.dist_threshold,
                            false,
                        );
                        count_sigma += u32::from(self.add_improper(
                            ORDER_1,
                            normal,
                            false,
                            SIG.clone(),
                            sigma_symbol,
                            presym.dist_threshold,
                            improper_kind.contains_time_reversal(),
                        ));
                    }
                }
            }

            if matches!(presym.rotational_symmetry, RotationalSymmetry::OblatePlanar) {
                // Planar system. The plane of the system (perpendicular to the highest-MoI
                // principal axis) might be a symmetry element: time-reversed in the presence of
                // a magnetic field (which must also lie in this plane), or both in the absence
                // of a magnetic field.
                let (_, principal_axes) = presym.molecule.calc_moi();
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
                        Some("h".to_owned()),
                        presym.dist_threshold,
                        improper_kind.contains_time_reversal(),
                    ));
                }
            }

            if count_sigma == max_ord_u32 {
                if max_ord_u32 > 1 {
                    let principal_element = self.get_proper_principal_element();
                    // Cnv
                    log::debug!("Found {} σv planes.", count_sigma);
                    if max_ord_u32 == 2 && principal_element.contains_time_reversal() {
                        // C2v, but with θ·C2
                        log::debug!("The C2 axis is actually θ·C2. The non-time-reversed σv will be reassigned as σh.");
                        let old_sigmas = self
                            .get_elements_mut(&SIG)
                            .expect("No improper elements found.")
                            .remove(&ORDER_1)
                            .expect("No σv found.");
                        let old_sigma = old_sigmas.iter().next().expect("No σv found.");
                        self.add_improper(
                            ORDER_1,
                            old_sigma.raw_axis,
                            false,
                            SIG.clone(),
                            Some("h".to_owned()),
                            presym.dist_threshold,
                            old_sigma.contains_time_reversal(),
                        );
                    }
                    self.set_group_name(format!("C{max_ord}v"));
                    let principal_element = self.get_proper_principal_element();
                    self.add_proper(
                        max_ord,
                        principal_element.raw_axis,
                        true,
                        presym.dist_threshold,
                        principal_element.contains_time_reversal(),
                    );

                    // One of the σ's is also a generator. We prioritise the non-time-reversed one
                    // as the generator.
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
                    // Cs
                    log::debug!("Found {} σ planes.", count_sigma);
                    self.set_group_name("Cs".to_owned());
                    let old_sigmas = if self.elements.contains_key(&SIG) {
                        self.get_elements_mut(&SIG)
                            .expect("No improper elements found.")
                            .remove(&ORDER_1)
                            .expect("No σ found.")
                    } else {
                        self.get_elements_mut(&TRSIG)
                            .expect("No time-reversed improper elements found.")
                            .remove(&ORDER_1)
                            .expect("No σ found.")
                    };
                    debug_assert_eq!(old_sigmas.len(), 1);
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
                }
            } else {
                let principal_element = self.get_proper_principal_element().clone();
                if let Some(improper_kind) =
                    presym.check_improper(&ORDER_1, &principal_element.raw_axis, &SIG, tr)
                {
                    // Cnh (n > 2)
                    assert_eq!(count_sigma, 1);
                    log::debug!("Found no σv planes but one σh plane.");
                    self.set_group_name(format!("C{max_ord}h"));
                    self.add_proper(
                        max_ord,
                        principal_element.raw_axis,
                        true,
                        presym.dist_threshold,
                        principal_element.contains_time_reversal(),
                    );
                    self.add_improper(
                        ORDER_1,
                        principal_element.raw_axis,
                        true,
                        SIG.clone(),
                        Some("h".to_owned()),
                        presym.dist_threshold,
                        improper_kind.contains_time_reversal(),
                    );

                    // Locate the remaining improper elements
                    let non_id_c_elements = self
                        .get_elements(&ROT)
                        .unwrap_or(&HashMap::new())
                        .values()
                        .chain(
                            self.get_elements(&TRROT)
                                .unwrap_or(&HashMap::new())
                                .values(),
                        )
                        .fold(vec![], |acc, c_eles| {
                            acc.into_iter()
                                .chain(
                                    c_eles
                                        .iter()
                                        .filter(|ele| *ele.raw_proper_order() != ORDER_1)
                                        .cloned(),
                                )
                                .collect()
                        });
                    if max_ord_u32 % 2 == 0 {
                        // Cnh, n even, an inversion centre is expected.
                        let vec_z = Vector3::new(0.0, 0.0, 1.0);
                        let inversion_check = presym.check_improper(&ORDER_2, &vec_z, &SIG, tr);
                        assert!(inversion_check.is_some());
                        self.add_improper(
                            ORDER_2,
                            vec_z,
                            false,
                            SIG.clone(),
                            None,
                            presym.dist_threshold,
                            inversion_check
                                .expect("Expected inversion centre not found.")
                                .contains_time_reversal(),
                        );
                        for c_element in non_id_c_elements {
                            let principal_element = self.get_proper_principal_element();
                            let sigma_symbol = deduce_sigma_symbol(
                                &c_element.raw_axis,
                                principal_element,
                                presym.dist_threshold,
                                false,
                            );
                            // iCn
                            let icn_check = presym.check_improper(
                                &c_element.raw_proper_order(),
                                &c_element.raw_axis,
                                &INV,
                                tr,
                            );
                            assert!(icn_check.is_some());
                            self.add_improper(
                                *c_element.raw_proper_order(),
                                c_element.raw_axis,
                                false,
                                INV.clone(),
                                sigma_symbol,
                                presym.dist_threshold,
                                icn_check
                                    .expect("Expected iCn not found.")
                                    .contains_time_reversal(),
                            );
                        }
                    } else {
                        // Cnh, n odd, only σh is present.
                        let sigma_h = self
                            .get_sigma_elements("h")
                            .expect("No σh found.")
                            .into_iter()
                            .next()
                            .expect("No σh found.")
                            .clone();
                        _add_sigmahcn(self, &sigma_h, non_id_c_elements, presym, tr);
                    }
                } else {
                    let double_max_ord = ElementOrder::new(2.0 * max_ord.to_float(), f64::EPSILON);
                    if let Some(improper_kind) =
                        presym.check_improper(&double_max_ord, &principal_element.raw_axis, &SIG, tr)
                    {
                        // S2n
                        self.set_group_name(if double_max_ord == ElementOrder::Int(2) {
                            // S2 is Ci.
                            "Ci".to_string()
                        } else {
                            format!("S{double_max_ord}")
                        });
                        self.add_improper(
                            double_max_ord,
                            principal_element.raw_axis,
                            false,
                            SIG.clone(),
                            None,
                            presym.dist_threshold,
                            improper_kind.contains_time_reversal(),
                        );
                        self.add_improper(
                            double_max_ord,
                            principal_element.raw_axis,
                            true,
                            SIG.clone(),
                            None,
                            presym.dist_threshold,
                            improper_kind.contains_time_reversal(),
                        );

                        // Locate the remaining improper symmetry elements
                        if max_ord_u32 % 2 != 0 {
                            // Odd rotation sub groups, an inversion centre is expected.
                            let vec_z = Vector3::new(0.0, 0.0, 1.0);
                            let inversion_check = presym.check_improper(&ORDER_2, &vec_z, &SIG, tr);
                            assert!(inversion_check.is_some());
                            self.add_improper(
                                ORDER_2,
                                vec_z,
                                false,
                                SIG.clone(),
                                None,
                                presym.dist_threshold,
                                inversion_check
                                    .expect("Expected inversion centre not found.")
                                    .contains_time_reversal(),
                            );
                        }
                    } else {
                        // Cn (n > 2)
                        self.set_group_name(format!("C{max_ord}"));
                        self.add_proper(
                            max_ord,
                            principal_element.raw_axis,
                            true,
                            presym.dist_threshold,
                            principal_element.contains_time_reversal(),
                        );
                    }
                }
            }
        }
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
    tr: bool,
) {
    assert!(sigma_h.is_o3_mirror_plane(false) || sigma_h.is_o3_mirror_plane(true));
    for c_element in non_id_c_elements {
        if approx::relative_eq!(
            c_element.raw_axis.cross(&sigma_h.raw_axis).norm(),
            0.0,
            epsilon = presym.dist_threshold,
            max_relative = presym.dist_threshold
        ) {
            // Cn is orthogonal to σh. The product Cn * σh is Sn.
            log::debug!("Cn is orthogonal to σh.");
            let sn_check =
                presym.check_improper(c_element.raw_proper_order(), &c_element.raw_axis, &SIG, tr);
            assert!(sn_check.is_some());
            let sigma_symbol = if *c_element.raw_proper_order() == ORDER_1 {
                Some("h".to_owned())
            } else {
                None
            };
            sym.add_improper(
                *c_element.raw_proper_order(),
                c_element.raw_axis,
                false,
                SIG.clone(),
                sigma_symbol,
                presym.dist_threshold,
                sn_check
                    .expect("Expected Sn axis not found.")
                    .contains_time_reversal(),
            );
        } else {
            // Cn is C2 and is contained in σh.
            // The product σh * C2 is a σv plane.
            approx::assert_relative_eq!(
                c_element.raw_axis.dot(&sigma_h.raw_axis).abs(),
                0.0,
                epsilon = presym.dist_threshold,
                max_relative = presym.dist_threshold
            );
            assert_eq!(*c_element.raw_proper_order(), ORDER_2);
            log::debug!("Cn is C2 and must therefore be contained in σh.");
            let s_axis = c_element.raw_axis.cross(&sigma_h.raw_axis).normalize();
            let sigmav_check = presym.check_improper(&ORDER_1, &s_axis, &SIG, tr);
            assert!(sigmav_check.is_some());
            sym.add_improper(
                ORDER_1,
                s_axis,
                false,
                SIG.clone(),
                Some("v".to_owned()),
                presym.dist_threshold,
                sigmav_check
                    .expect("Expected σv not found.")
                    .contains_time_reversal(),
            );
        }
    }
}
