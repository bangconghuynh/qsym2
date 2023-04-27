use std::fmt;

use anyhow::{bail, format_err};
use derive_builder::Builder;
use itertools::Itertools;
use log;
use nalgebra::{Point3, Vector3};

use crate::aux::atom::{Atom, AtomKind};
use crate::aux::misc::log_title;
use crate::aux::molecule::Molecule;
use crate::drivers::{QSym2Driver, QSym2Output};
use crate::symmetry::symmetry_core::{PreSymmetry, Symmetry};

#[cfg(test)]
#[path = "point_group_detection_tests.rs"]
mod point_group_detection_tests;

// ==================
// Struct definitions
// ==================

// ----------
// Parameters
// ----------

#[derive(Clone, Builder, Debug)]
pub struct PointGroupDetectionParams {
    #[builder(setter(custom), default = "vec![1.0e-4, 1.0e-5, 1.0e-6]")]
    moi_thresholds: Vec<f64>,

    #[builder(setter(custom), default = "vec![1.0e-4, 1.0e-5, 1.0e-6]")]
    distance_thresholds: Vec<f64>,

    time_reversal: bool,

    #[builder(default = "None")]
    fictitious_magnetic_fields: Option<Vec<(Point3<f64>, Vector3<f64>)>>,

    #[builder(default = "None")]
    fictitious_electric_fields: Option<Vec<(Point3<f64>, Vector3<f64>)>>,

    #[builder(default = "false")]
    print_symmetry_elements: bool,

    #[builder(default = "true")]
    print_class_transversal: bool,
}

impl PointGroupDetectionParams {
    pub fn builder() -> PointGroupDetectionParamsBuilder {
        PointGroupDetectionParamsBuilder::default()
    }
}

impl PointGroupDetectionParamsBuilder {
    fn moi_thresholds(&mut self, threshs: &[f64]) -> &mut Self {
        self.moi_thresholds = Some(threshs.to_vec());
        self
    }

    fn distance_thresholds(&mut self, threshs: &[f64]) -> &mut Self {
        self.distance_thresholds = Some(threshs.to_vec());
        self
    }
}

impl QSym2Output for PointGroupDetectionParams {
    fn log_output(&self) {
        let threshs = self
            .moi_thresholds
            .iter()
            .cartesian_product(self.distance_thresholds.iter());
        let nthreshs = threshs.clone().count();
        if nthreshs == 1 {
            log_title("§ Fixed-Threshold Point-Group Detection §");
            log::info!(target: "output", "");
            log::info!(target: "output", "MoI threshold: {:.3e}", self.moi_thresholds[0]);
            log::info!(target: "output", "Geo threshold: {:.3e}", self.distance_thresholds[0]);
        } else {
            log_title("§ Variable-Threshold Point-Group Detection §");
            log::info!(target: "output", "");
            log::info!(
                target: "output",
                "MoI thresholds: {}",
                self
                    .moi_thresholds
                    .iter()
                    .map(|v| format!("{v:.3e}"))
                    .join(", ")
            );
            log::info!(
                target: "output",
                "Geo thresholds: {}",
                self
                    .distance_thresholds
                    .iter()
                    .map(|v| format!("{v:.3e}"))
                    .join(", ")
            );
            log::info!(target: "output", "");
        }

        if let Some(fictitious_magnetic_fields) = self.fictitious_magnetic_fields.as_ref() {
            log::info!(target: "output", "Fictitious magnetic fields:");
            for (origin, field) in fictitious_magnetic_fields.iter() {
                log::info!(
                    target: "output",
                    "  ({}) ± ({})",
                    origin.iter().map(|x| format!("{x:+.3}")).join(", "),
                    field.iter().map(|x| format!("{x:+.3}")).join(", "),
                );
            }
            log::info!(target: "output", "");
        }

        if let Some(fictitious_electric_fields) = self.fictitious_electric_fields.as_ref() {
            log::info!(target: "output", "Fictitious electric fields:");
            for (origin, field) in fictitious_electric_fields.iter() {
                log::info!(target: "output",
                    "  ({}) + ({})",
                    origin.iter().map(|x| format!("{x:+.3}")).join(", "),
                    field.iter().map(|x| format!("{x:+.3}")).join(", "),
                );
            }
            log::info!(target: "output", "");
        }

        log::info!(
            target: "output",
            "Considering time reversal: {}",
            if self.time_reversal { "yes" } else { "no" }
        );
        log::info!(target: "output", "");
    }
}

// ------
// Result
// ------

#[derive(Clone, Builder)]
pub struct PointGroupDetectionResult {
    pre_symmetry: PreSymmetry,

    unitary_symmetry: Symmetry,

    #[builder(default = "None")]
    magnetic_symmetry: Option<Symmetry>,
}

impl PointGroupDetectionResult {
    fn builder() -> PointGroupDetectionResultBuilder {
        PointGroupDetectionResultBuilder::default()
    }
}

// ------
// Driver
// ------

#[derive(Clone, Builder)]
pub struct PointGroupDetectionDriver<'a> {
    parameters: PointGroupDetectionParams,

    #[builder(default = "None")]
    xyz: Option<String>,

    #[builder(default = "None")]
    molecule: Option<&'a Molecule>,

    #[builder(default = "None")]
    result: Option<PointGroupDetectionResult>,
}

impl<'a> PointGroupDetectionDriver<'a> {
    pub fn builder() -> PointGroupDetectionDriverBuilder<'a> {
        PointGroupDetectionDriverBuilder::default()
    }

    fn detect_point_group(&mut self) -> Result<(), anyhow::Error> {
        let params = &self.parameters;
        params.log_output();

        let threshs = params
            .moi_thresholds
            .iter()
            .cartesian_product(params.distance_thresholds.iter());
        let nthreshs = threshs.clone().count();

        let count_length = usize::try_from(nthreshs.ilog10() + 2).map_err(|_| {
            format_err!("Unable to convert `{}` to `usize`.", nthreshs.ilog10() + 2)
        })?;
        log::info!(target: "output", "{}", "┈".repeat(count_length + 75));
        log::info!(
            target: "output",
            "{:>width$} {:>12} {:>12} {:>14} {:>9} {:>12} {:>9}",
            "#",
            "MoI thresh",
            "Geo thresh",
            "Mag. group",
            "Elements",
            "Uni. group",
            "Elements",
            width = count_length
        );
        log::info!(target: "output", "{}", "┈".repeat(count_length + 75));
        let mut i = 0;
        let syms = threshs.map(|(moi_thresh, dist_thresh)| {
            // Create a new molecule with the current distance threshold for symmetry analysis
            let mut mol = match (self.molecule, self.xyz.as_ref()) {
                (Some(molecule), None) => Molecule::from_atoms(
                    &molecule.get_all_atoms().into_iter().cloned().collect_vec(),
                    *dist_thresh
                ),
                (None, Some(xyz)) => Molecule::from_xyz(
                    xyz,
                    *dist_thresh
                ),
                _ => bail!("Both `molecule` and `xyz` are specified.")
            };

            // Add any fictitious magnetic fields
            if let Some(fictitious_magnetic_fields) = params.fictitious_magnetic_fields.as_ref() {
                if let Some(_) = mol.magnetic_atoms {
                    bail!("Cannot set fictitious magnetic fields in the presence of actual magnetic fields.")
                } else {
                    let fictitious_magnetic_atoms = fictitious_magnetic_fields.iter().flat_map(|(origin, vec)| {
                        Ok::<[Atom; 2], anyhow::Error>([
                            Atom::new_special(AtomKind::Magnetic(true), origin + vec, *dist_thresh).ok_or_else(||
                                format_err!("Cannot construct a fictitious magnetic atom.")
                            )?,
                            Atom::new_special(AtomKind::Magnetic(false), origin - vec, *dist_thresh).ok_or_else(||
                                format_err!("Cannot construct a fictitious magnetic atom.".to_string())
                            )?,
                        ])
                    }).flatten().collect_vec();
                    mol.magnetic_atoms = Some(fictitious_magnetic_atoms);
                }
            }

            // Add any fictitious electric fields
            if let Some(fictitious_electric_fields) = params.fictitious_electric_fields.as_ref() {
                if let Some(_) = mol.electric_atoms {
                    bail!("Cannot set fictitious electric fields in the presence of actual electric fields.")
                } else {
                    let fictitious_electric_atoms = fictitious_electric_fields.iter().flat_map(|(origin, vec)| {
                        Ok::<Atom, anyhow::Error>(
                            Atom::new_special(AtomKind::Electric(true), origin + vec, *dist_thresh).ok_or_else(||
                                format_err!("Cannot construct a fictitious electric atom.")
                            )?
                        )
                    }).collect_vec();
                    mol.electric_atoms = Some(fictitious_electric_atoms);
                }
            }

            // Perform point-group detection
            // A recentred copy of the molecule will be used for all point-group detection.
            let presym = PreSymmetry::builder()
                .moi_threshold(*moi_thresh)
                .molecule(&mol, true)
                .build()
                .map_err(|_| format_err!("Cannot construct a pre-symmetry structure."))?;
            let mut uni_sym = Symmetry::new();
            let uni_res = uni_sym.analyse(&presym, false);
            let uni_ok = uni_res.is_ok();
            if !uni_ok {
                log::error!("{}", uni_res.unwrap_err())
            }
            let uni_group_name = uni_sym.group_name.clone().unwrap_or("?".to_string());
            let uni_group_nele = if uni_sym.is_infinite() {
                "∞".to_string()
            } else {
                uni_sym.n_elements().to_string()
            };

            let (mag_sym_opt, mag_ok) = if params.time_reversal {
                let mut mag_sym = Symmetry::new();
                let mag_res = mag_sym.analyse(&presym, true);
                let mag_ok = mag_res.is_ok();
                if !mag_ok {
                    log::error!("{}", mag_res.unwrap_err())
                }
                (Some(mag_sym), mag_ok)
            } else {
                (None, true)
            };
            let mag_group_name = mag_sym_opt
                .as_ref()
                .map(|mag_sym| {
                    mag_sym
                        .group_name
                        .clone()
                        .unwrap_or("?".to_string())
                })
                .unwrap_or_else(|| "--".to_string());
            let mag_group_nele = mag_sym_opt
                .as_ref()
                .map(|mag_sym| if mag_sym.is_infinite() {
                    "∞".to_string()
                } else {
                    mag_sym.n_elements().to_string()
                })
                .unwrap_or_else(|| "--".to_string());

            i += 1;
            if uni_ok && mag_ok {
                log::info!(
                    target: "output",
                    "{:>width$} {:>12.3e} {:>12.3e} {:>14} {:>9} {:>12} {:>9}",
                    i,
                    moi_thresh,
                    dist_thresh,
                    mag_group_name,
                    mag_group_nele,
                    uni_group_name,
                    uni_group_nele,
                    width = count_length
                );
                Ok((presym, uni_sym, mag_sym_opt))
            } else {
                if !uni_ok {
                    log::debug!(
                        "Unitary group detection with MoI threshold {:.3e} and distance threshold {:.3e} has failed.",
                        moi_thresh,
                        dist_thresh
                    );
                }
                if !mag_ok {
                    log::debug!(
                        "Magnetic group detection with MoI threshold {:.3e} and distance threshold {:.3e} has failed.",
                        moi_thresh,
                        dist_thresh
                    );
                }
                log::info!(
                    target: "output",
                    "{:>width$} {:>12.3e} {:>12.3e} {:>14} {:>9} {:>12} {:>9}",
                    i,
                    moi_thresh,
                    dist_thresh,
                    "--",
                    "--",
                    "--",
                    "--",
                    width = count_length
                );
                bail!(
                    "Group determination with MoI threshold {:.3e} and distance threshold {:.3e} has failed.",
                    moi_thresh,
                    dist_thresh
                )
            }
        })
        .filter_map(|res_sym| res_sym.ok())
        .collect_vec();
        log::info!(target: "output", "{}", "┈".repeat(count_length + 75));
        log::info!(target: "output", "(The number of symmetry elements is not the same as the order of the group.)");
        log::info!(target: "output", "");

        let (highest_presym, highest_uni_sym, highest_mag_sym_opt) = syms
            .into_iter()
            .max_by(
                |(presym_a, uni_sym_a, mag_sym_opt_a), (presym_b, uni_sym_b, mag_sym_opt_b)| {
                    (
                        mag_sym_opt_a
                            .as_ref()
                            .map(|mag_sym| mag_sym.is_infinite())
                            .unwrap_or(false),
                        mag_sym_opt_a
                            .as_ref()
                            .map(|mag_sym| mag_sym.n_elements())
                            .unwrap_or(0),
                        uni_sym_a.is_infinite(),
                        uni_sym_a.n_elements(),
                        1.0 / presym_a.moi_threshold,
                        1.0 / presym_a.dist_threshold,
                    )
                        .partial_cmp(&(
                            mag_sym_opt_b
                                .as_ref()
                                .map(|mag_sym| mag_sym.is_infinite())
                                .unwrap_or(false),
                            mag_sym_opt_b
                                .as_ref()
                                .map(|mag_sym| mag_sym.n_elements())
                                .unwrap_or(0),
                            uni_sym_b.is_infinite(),
                            uni_sym_b.n_elements(),
                            1.0 / presym_b.moi_threshold,
                            1.0 / presym_b.dist_threshold,
                        ))
                        .expect("Unable to perform a comparison.")
                },
            )
            .ok_or_else(|| {
                format_err!("Unable to identify the highest-symmetry group.".to_string())
            })?;

        if let Some(highest_mag_sym) = &highest_mag_sym_opt {
            let n_mag_elements = if highest_mag_sym.is_infinite() {
                "∞".to_string()
            } else {
                highest_mag_sym.n_elements().to_string()
            };
            log::info!(
                target: "output",
                "Highest mag. group found: {} ({} {})",
                highest_mag_sym.group_name.as_ref().unwrap_or(&"?".to_string()),
                n_mag_elements,
                if n_mag_elements != "1" {
                    "symmetry elements"
                } else {
                    "symmetry element"
                }
            );
        }

        let n_uni_elements = if highest_uni_sym.is_infinite() {
            "∞".to_string()
        } else {
            highest_uni_sym.n_elements().to_string()
        };
        log::info!(
            target: "output",
            "Highest uni. group found: {} ({} {})",
            highest_uni_sym.group_name.as_ref().unwrap_or(&"?".to_string()),
            n_uni_elements,
            if n_uni_elements != "1" {
                "symmetry elements"
            } else {
                "symmetry element"
            }
        );
        log::info!(
            target: "output",
            "  Associated MoI threshold: {:.3e}",
            highest_presym.moi_threshold
        );
        log::info!(
            target: "output",
            "  Associated geo threshold: {:.3e}",
            highest_presym.dist_threshold
        );

        self.result = PointGroupDetectionResult::builder()
            .pre_symmetry(highest_presym)
            .unitary_symmetry(highest_uni_sym)
            .magnetic_symmetry(highest_mag_sym_opt)
            .build()
            .ok();

        Ok(())
    }
}

impl<'a> QSym2Driver for PointGroupDetectionDriver<'a> {
    type Outcome = PointGroupDetectionResult;

    fn result(&self) -> Result<&Self::Outcome, anyhow::Error> {
        self.result
            .as_ref()
            .ok_or_else(|| format_err!("No point-group detection results found."))
    }

    fn run(&mut self) -> Result<(), anyhow::Error> {
        self.detect_point_group()
    }
}
