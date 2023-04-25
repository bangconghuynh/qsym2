use std::panic::catch_unwind;

use derive_builder::Builder;
use itertools::Itertools;
use log;
use nalgebra::{Point3, Vector3};

use crate::aux::atom::{Atom, AtomKind};
use crate::aux::geometry::Transform;
use crate::aux::molecule::Molecule;
use crate::drivers::{QSym2Driver, QSym2Error};
use crate::symmetry::symmetry_core::{PreSymmetry, Symmetry};

// ==================
// Struct definitions
// ==================

// ----------
// Parameters
// ----------

#[derive(Clone, Builder)]
pub struct PointGroupDetectionParams {
    #[builder(setter(custom), default = "vec![1.0e-4, 1.0e-5, 1.0e-6]")]
    moi_thresholds: Vec<f64>,

    #[builder(setter(custom), default = "vec![1.0e-4, 1.0e-5, 1.0e-6]")]
    distance_thresholds: Vec<f64>,

    recentre_molecule: bool,

    reorientate_molecule: bool,

    time_reversal: bool,

    double_group: bool,

    #[builder(default = "None")]
    infinite_order_to_finite: Option<usize>,

    #[builder(default = "None")]
    fictitious_magnetic_fields: Option<Vec<(Point3<f64>, Vector3<f64>)>>,

    #[builder(default = "None")]
    fictitious_electric_fields: Option<Vec<(Point3<f64>, Vector3<f64>)>>,

    #[builder(default = "false")]
    print_symmetry_elements: bool,

    #[builder(default = "true")]
    print_class_transversal: bool,
}

// ------
// Result
// ------

#[derive(Clone, Builder)]
pub struct PointGroupDetectionResult {
    symmetry: Symmetry,
}

// ------
// Driver
// ------

#[derive(Clone, Builder)]
pub struct PointGroupDetectionDriver<'a> {
    parameters: PointGroupDetectionParams,

    molecule: &'a Molecule,

    result: Result<PointGroupDetectionResult, QSym2Error<'static>>,
}

impl<'a> PointGroupDetectionDriver<'a> {
    fn detect_point_group(&mut self) -> Result<&mut Self, QSym2Error<'static>> {
        let params = &self.parameters;
        let threshs = params
            .moi_thresholds
            .iter()
            .cartesian_product(params.distance_thresholds.iter());
        let nthreshs = threshs.peekable().count();
        if nthreshs == 1 {
            log::info!("§ Fixed-Threshold Point-Group Detection §");
            log::info!("MoI threshold     : {:.3e}", params.moi_thresholds[0]);
            log::info!("Distance threshold: {:.3e}", params.distance_thresholds[0]);
        } else {
            log::info!("────────────────────────────────────────────");
            log::info!("§ Variable-Threshold Point-Group Detection §");
            log::info!("────────────────────────────────────────────");
            log::info!(
                "MoI thresholds     : {}",
                params
                    .moi_thresholds
                    .iter()
                    .map(|v| format!("{v:.3e}"))
                    .join(", ")
            );
            log::info!(
                "Distance thresholds: {}",
                params
                    .distance_thresholds
                    .iter()
                    .map(|v| format!("{v:.3e}"))
                    .join(", ")
            );
        }

        let count_length = usize::try_from(nthreshs.ilog10() + 1).map_err(|_| QSym2Error {
            source: None,
            msg: Some("Unable to convert `nthreshs.ilog10() + 1` to `usize`."),
        })?;
        log::info!(
            "{:>width$} {:>12} {:>12} {:>11} {:>8}",
            "#",
            "MoI Thresh",
            "Dist Thresh",
            "Point Group",
            "Elements",
            width = count_length
        );
        log::info!("{}", "┈".repeat(count_length + 47));
        let mut i = 0;
        let syms = threshs.map(|(moi_thresh, dist_thresh)| {
            // Create a new molecule with the current distance threshold for symmetry analysis
            let mut mol = Molecule::from_atoms(
                &self.molecule.get_all_atoms().into_iter().cloned().collect_vec(),
                *dist_thresh
            );

            // Add any fictitious magnetic fields
            if let Some(fictitious_magnetic_fields) = params.fictitious_magnetic_fields.as_ref() {
                if let Some(_) = mol.magnetic_atoms {
                    return Err(
                        QSym2Error {
                            source: None,
                            msg: Some("Cannot set fictitious magnetic fields in the presence of actual magnetic fields.")
                        }
                    );
                } else {
                    let fictitious_magnetic_atoms = fictitious_magnetic_fields.iter().flat_map(|(origin, vec)| {
                        Ok([
                            Atom::new_special(AtomKind::Magnetic(true), origin + vec, *dist_thresh).ok_or(
                                QSym2Error {
                                    source: None,
                                    msg: Some("Cannot construct a fictitious magnetic atom.")
                                }
                            )?,
                            Atom::new_special(AtomKind::Magnetic(false), origin - vec, *dist_thresh).ok_or(
                                QSym2Error {
                                    source: None,
                                    msg: Some("Cannot construct a fictitious magnetic atom.")
                                }
                            )?,
                        ])
                    }).flatten().collect_vec();
                    mol.magnetic_atoms = Some(fictitious_magnetic_atoms);
                }
            }

            // Add any fictitious electric fields
            if let Some(fictitious_electric_fields) = params.fictitious_electric_fields.as_ref() {
                if let Some(_) = mol.electric_atoms {
                    return Err(
                        QSym2Error {
                            source: None,
                            msg: Some("Cannot set fictitious electric fields in the presence of actual electric fields.")
                        }
                    );
                } else {
                    let fictitious_electric_atoms = fictitious_electric_fields.iter().flat_map(|(origin, vec)| {
                        Ok(
                            Atom::new_special(AtomKind::Electric(true), origin + vec, *dist_thresh).ok_or(
                                QSym2Error {
                                    source: None,
                                    msg: Some("Cannot construct a fictitious electric atom.")
                                }
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
                .map_err(|_| QSym2Error {
                    source: None,
                    msg: Some("Cannot construct a pre-symmetry structure.")
                })?;
            let mut sym = Symmetry::new();
            let result = catch_unwind(|| sym.analyse(&presym, params.time_reversal));
            i += 1;
            if result.is_ok() {
                log::info!(
                    "{:>width$} {:>12} {:>12} {:>11} {:>8}",
                    i,
                    moi_thresh,
                    dist_thresh,
                    sym.group_name.unwrap_or("?".to_string()),
                    sym.n_elements(),
                    width = count_length
                );
                Ok((presym, sym))
            } else {
                log::info!(
                    "{:>width$} {:>12} {:>12} {:>11} {:>8}",
                    i,
                    moi_thresh,
                    dist_thresh,
                    "--",
                    "--",
                    width = count_length
                );
                Err(QSym2Error{
                    source: None,
                    msg: Some(&format!(
                            "Point-group determination with MoI threshold {:.3e} and distance threshold {:.3e} has failed.",
                            moi_thresh,
                            dist_thresh
                        ))
                })
            }
        })
        .filter_map(|res_sym| res_sym.ok())
        .collect_vec();

        let (highest_presym, highest_sym) = syms
            .iter()
            .max_by(|(presym_a, sym_a), (presym_b, sym_b)| {
                (
                    sym_a.n_elements(),
                    1.0 / presym_a.moi_threshold,
                    1.0 / presym_a.dist_threshold,
                )
                    .partial_cmp(&(
                        sym_b.n_elements(),
                        1.0 / presym_b.moi_threshold,
                        1.0 / presym_b.dist_threshold,
                    ))
                    .expect("Unable to perform a comparison.")
            })
            .ok_or(QSym2Error {
                source: None,
                msg: Some("Unable to identify the highest-symmetry group."),
            })?;

        let n_elements = highest_sym.n_elements();
        log::info!(
            "Highest point group found: {} ({} {})",
            highest_sym.group_name.unwrap_or("?".to_string()),
            n_elements,
            if n_elements != 1 {
                "symmetry elements"
            } else {
                "symmetry element"
            }
        );
        log::info!("  Associated MoI threshold     : {:.3e}", highest_presym.moi_threshold);
        log::info!("  Associated distance threshold: {:.3e}", highest_presym.dist_threshold);

        Ok(self)
    }
}

impl<'a> QSym2Driver for PointGroupDetectionDriver<'a> {
    type Outcome = PointGroupDetectionResult;

    fn result(&self) -> Result<&Self::Outcome, &QSym2Error> {
        self.result.as_ref()
    }

    fn run(&mut self) {}
}
