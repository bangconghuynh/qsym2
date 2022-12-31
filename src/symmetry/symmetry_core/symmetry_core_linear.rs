use super::{PreSymmetry, Symmetry};
use crate::rotsym::RotationalSymmetry;
use crate::symmetry::symmetry_element::{SymmetryElementKind, SIG};
use crate::symmetry::symmetry_element_order::{ElementOrder, ORDER_1, ORDER_2, ORDER_I};
use approx;
use log;
use nalgebra::Vector3;

impl Symmetry {
    /// Performs point-group detection analysis for a linear molecule.
    ///
    /// # Arguments
    ///
    /// * `presym` - A pre-symmetry-analysis struct containing information about
    /// the molecular system.
    ///
    /// # Panics
    ///
    /// Panics when any inconsistencies are encountered along the point-group detection path.
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
            epsilon = presym.dist_threshold,
            max_relative = presym.dist_threshold
        ));
        assert!(self.add_proper(
            ORDER_I,
            principal_axes[0],
            true,
            presym.dist_threshold
        ));

        if presym.check_improper(
            &ElementOrder::Int(2),
            &Vector3::new(0.0, 0.0, 1.0),
            &SymmetryElementKind::ImproperMirrorPlane,
        ) {
            // i
            log::debug!("Located an inversion centre.");
            assert!(self.add_improper(
                ORDER_2,
                Vector3::new(0.0, 0.0, 1.0),
                false,
                SIG.clone(),
                None,
                presym.dist_threshold
            ));

            // σh must exist if C∞ and i both exist.
            log::debug!("σh implied from C∞ and i.");
            assert!(presym.check_improper(&ORDER_1, &principal_axes[0], &SIG));
            assert!(self.add_improper(
                ORDER_1,
                principal_axes[0].clone_owned(),
                true,
                SIG.clone(),
                Some("h".to_owned()),
                presym.dist_threshold
            ));

            if presym.check_proper(&ORDER_2, &principal_axes[1]) {
                // C2
                log::debug!("Located a C2 axis perpendicular to C∞.");
                self.add_proper(
                    ORDER_2,
                    principal_axes[1],
                    true,
                    presym.dist_threshold
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
                    ORDER_1,
                    principal_axes[1].clone_owned(),
                    true,
                    SIG.clone(),
                    Some("v".to_owned()),
                    presym.dist_threshold
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
            self.point_group.as_ref().expect("No point groups found.")
        );
    }
}
