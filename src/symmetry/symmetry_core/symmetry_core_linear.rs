use super::{PreSymmetry, Symmetry};
use crate::aux::geometry;
use crate::aux::molecule::Molecule;
use crate::rotsym::RotationalSymmetry;
use crate::symmetry::symmetry_element::{
    ElementOrder, SymmetryElementKind, ORDER_1, ORDER_2, ORDER_I, SIG,
};
use approx;
use itertools::{self, Itertools};
use log;
use nalgebra::Vector3;
use std::collections::HashSet;

impl Symmetry {
    /// Performs point-group detection analysis for a linear molecule.
    pub fn analyse_linear(&mut self, presym: &PreSymmetry) {
        let (mois, principal_axes) = presym.molecule.calc_moi();

        assert!(matches!(
            presym.rotational_symmetry,
            RotationalSymmetry::ProlateLinear
        ));
        assert!(presym
            .sea_groups
            .iter()
            .all(|sea_group| sea_group.len() <= 2));

        // C∞
        assert!(approx::relative_eq!(
            mois[0],
            0.0,
            epsilon = presym.molecule.threshold,
            max_relative = presym.molecule.threshold
        ));
        assert!(self.add_proper(
            ORDER_I.clone(),
            principal_axes[0].clone(),
            true,
            presym.molecule.threshold
        ));

        if _check_inversion(&presym) {
            // i
            log::debug!("Located an inversion centre.");
            assert!(self.add_improper(
                ORDER_2.clone(),
                Vector3::new(0.0, 0.0, 1.0),
                false,
                SIG.clone(),
                None,
                presym.molecule.threshold
            ));

            // σh must exist if C∞ and i both exist.
            log::debug!("σh implied from C∞ and i.");
            assert!(presym.check_improper(&ORDER_1, &principal_axes[0], &SIG));
            assert!(self.add_improper(
                ORDER_1.clone(),
                principal_axes[0].clone_owned(),
                true,
                SIG.clone(),
                Some("h".to_owned()),
                presym.molecule.threshold
            ));

            if presym.check_proper(&ORDER_2, &principal_axes[1]) {
                // C2
                log::debug!("Located a C2 axis perpendicular to C∞.");
                self.add_proper(
                    ORDER_2.clone(),
                    principal_axes[1].clone(),
                    true,
                    presym.molecule.threshold
                );
                self.point_group = Some("D∞h".to_owned());
            } else {
                // No C2
                log::debug!("No C2 axes perpendicular to C∞ found.");
                self.point_group = Some("C∞h".to_owned());
            }
        } else {
            // No i
            log::debug!("No inversion centres found.");
            if presym.check_improper(&ORDER_1, &principal_axes[1], &SIG) {
                // σv
                log::debug!("Located a σv plane.");
                self.add_improper(
                    ORDER_1.clone(),
                    principal_axes[1].clone_owned(),
                    true,
                    SIG.clone(),
                    Some("v".to_owned()),
                    presym.molecule.threshold
                );
                self.point_group = Some("C∞v".to_owned());
            } else {
                // No σv
                log::debug!("No σv planes found.");
                self.point_group = Some("C∞".to_owned());
            }
        }
        log::debug!(
            "Point group determined: {}",
            self.point_group.as_ref().unwrap()
        );
    }
}

fn _check_inversion(presym: &PreSymmetry) -> bool {
    presym.check_improper(
        &ElementOrder::Int(2),
        &Vector3::new(0.0, 0.0, 1.0),
        &SymmetryElementKind::ImproperMirrorPlane,
    )
}
