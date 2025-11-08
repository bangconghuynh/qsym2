//! Molecular symmetry element detection for linear systems.

use anyhow::{self, ensure, format_err};

use super::{PreSymmetry, Symmetry};
use crate::rotsym::RotationalSymmetry;
use crate::symmetry::symmetry_element::SIG;
use crate::symmetry::symmetry_element_order::{ElementOrder, ORDER_1, ORDER_2, ORDER_I};
use approx;
use log;
use nalgebra::Vector3;

impl Symmetry {
    /// Performs point-group detection analysis for a linear molecule.
    ///
    /// # Arguments
    ///
    /// * `presym` - A pre-symmetry-analysis structure containing information about the molecular
    ///   system.
    /// * `tr` - A flag indicating if time reversal should also be considered. A time-reversed
    ///   symmetry element will only be considered if its non-time-reversed version turns out to be
    ///   not a symmetry element.
    pub(super) fn analyse_linear(
        &mut self,
        presym: &PreSymmetry,
        tr: bool,
    ) -> Result<(), anyhow::Error> {
        let (mois, principal_axes) = presym.recentred_molecule.calc_moi();

        ensure!(
            matches!(
                presym.rotational_symmetry,
                RotationalSymmetry::ProlateLinear
            ),
            "Unexpected rotational symmetry -- expected: {}, actual: {}",
            RotationalSymmetry::ProlateLinear,
            presym.rotational_symmetry
        );
        ensure!(
            presym
                .sea_groups
                .iter()
                .all(|sea_group| sea_group.len() <= 2),
            "Unexpected SEA groups of more than two atoms found for a linear molecule."
        );

        // C∞
        ensure!(
            approx::relative_eq!(
                mois[0],
                0.0,
                epsilon = presym.dist_threshold,
                max_relative = presym.dist_threshold
            ),
            "Unexpected non-zero smallest principal moment of inertia."
        );
        ensure!(
            self.add_proper(
                ORDER_I,
                &principal_axes[0],
                true,
                presym.dist_threshold,
                false
            ),
            "Expected C∞ axis not added."
        );
        if let Some(improper_kind) = presym.check_improper(
            &ElementOrder::Int(2),
            &Vector3::new(0.0, 0.0, 1.0),
            &SIG,
            tr,
        ) {
            // i
            log::debug!("Located an inversion centre.");
            ensure!(
                self.add_improper(
                    ORDER_2,
                    &Vector3::new(0.0, 0.0, 1.0),
                    false,
                    SIG,
                    None,
                    presym.dist_threshold,
                    improper_kind.contains_time_reversal()
                ),
                "Expected inversion centre not added."
            );

            // σh must exist if C∞ and i both exist.
            log::debug!("σh implied from C∞ and i.");
            let sigma_check = presym.check_improper(&ORDER_1, &principal_axes[0], &SIG, tr);
            ensure!(
                sigma_check.is_some(),
                "Expected σh implied by C∞ and i not found."
            );
            ensure!(
                self.add_improper(
                    ORDER_1,
                    &principal_axes[0],
                    true,
                    SIG,
                    Some("h".to_owned()),
                    presym.dist_threshold,
                    sigma_check
                        .ok_or_else(|| format_err!(
                            "Expected mirror plane implied by C∞ and i not found."
                        ))?
                        .contains_time_reversal(),
                ),
                "Expected σh implied by C∞ and i not added."
            );

            if let Some(proper_kind) = presym.check_proper(&ORDER_2, &principal_axes[1], tr) {
                // C2
                log::debug!("Located a C2 axis perpendicular to C∞.");
                self.add_proper(
                    ORDER_2,
                    &principal_axes[1],
                    true,
                    presym.dist_threshold,
                    proper_kind.contains_time_reversal(),
                );
                self.set_group_name("D∞h".to_owned());
            } else {
                // No C2
                log::debug!("No C2 axes perpendicular to C∞ found.");
                self.set_group_name("C∞h".to_owned());
            }
        } else {
            // No i
            log::debug!("No inversion centres found.");
            if let Some(improper_kind) =
                presym.check_improper(&ORDER_1, &principal_axes[1], &SIG, tr)
            {
                // σv
                log::debug!("Located a σv plane.");
                self.add_improper(
                    ORDER_1,
                    &principal_axes[1],
                    true,
                    SIG,
                    Some("v".to_owned()),
                    presym.dist_threshold,
                    improper_kind.contains_time_reversal(),
                );
                self.set_group_name("C∞v".to_owned());
            } else {
                // No σv
                log::debug!("No σv planes found.");
                self.set_group_name("C∞".to_owned());
            }
        }

        Ok(())
    }
}
