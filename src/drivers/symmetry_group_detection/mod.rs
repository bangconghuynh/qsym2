use std::fmt;

use anyhow::{bail, format_err};
use derive_builder::Builder;
use itertools::Itertools;
use log;
use nalgebra::{Point3, Vector3};
use serde::{Deserialize, Serialize};

use crate::aux::atom::{Atom, AtomKind};
use crate::aux::format::{log_subtitle, log_title, nice_bool, write_subtitle};
use crate::aux::molecule::Molecule;
use crate::drivers::{QSym2Driver, QSym2Output};
use crate::io::{write_qsym2, QSym2FileType};
use crate::symmetry::symmetry_core::{PreSymmetry, Symmetry};
use crate::symmetry::symmetry_element::{AntiunitaryKind, SymmetryElementKind};

#[cfg(test)]
#[path = "symmetry_group_detection_tests.rs"]
mod symmetry_group_detection_tests;

// ==================
// Struct definitions
// ==================

// ----------
// Parameters
// ----------

/// A structure containing control parameters for symmetry-group detection.
#[derive(Clone, Builder, Debug, Serialize, Deserialize)]
pub struct SymmetryGroupDetectionParams {
    /// Thresholds for moment-of-inertia comparisons.
    #[builder(setter(custom), default = "vec![1.0e-4, 1.0e-5, 1.0e-6]")]
    pub moi_thresholds: Vec<f64>,

    /// Thresholds for distance and geometry comparisons.
    #[builder(setter(custom), default = "vec![1.0e-4, 1.0e-5, 1.0e-6]")]
    pub distance_thresholds: Vec<f64>,

    /// Boolean indicating if time reversal is to be taken into account.
    pub time_reversal: bool,

    /// Magnetic fields to be added to the system. Each magnetic field is specified by an origin
    /// $`\mathbf{O}`$ and a vector $`\mathbf{v}`$, for which a `magnetic(+)` special atom will be
    /// added at $`\mathbf{O} + \mathbf{v}`$, and a `magnetic(-)` special atom will be added at
    /// $`\mathbf{O} - \mathbf{v}`$.
    #[builder(default = "None")]
    pub magnetic_fields: Option<Vec<(Point3<f64>, Vector3<f64>)>>,

    /// Electric fields to be added to the system. Each electric field is specified by an origin
    /// $`\mathbf{O}`$ and a vector $`\mathbf{v}`$, for which an `electric(+)` special atom will be
    /// added at $`\mathbf{O} + \mathbf{v}`$.
    #[builder(default = "None")]
    pub electric_fields: Option<Vec<(Point3<f64>, Vector3<f64>)>>,

    /// Boolean indicating if the origins specified in [`Self::magnetic_fields`] and
    /// [`Self::electric_fields`] are to be taken relative to the molecule's centre of
    /// mass rather than to the space-fixed origin.
    #[builder(default = "false")]
    pub field_origin_com: bool,

    /// Boolean indicating if a summary of the located symmetry elements is to be written to the
    /// output file.
    #[builder(default = "false")]
    pub write_symmetry_elements: bool,

    /// Optional name for saving the result as a binary file of type [`QSym2FileType::Sym`]. If
    /// `None`, the result will not be saved.
    #[builder(default = "None")]
    pub result_save_name: Option<String>,
}

impl SymmetryGroupDetectionParams {
    /// Returns a builder to construct a [`SymmetryGroupDetectionParams`] structure.
    pub fn builder() -> SymmetryGroupDetectionParamsBuilder {
        SymmetryGroupDetectionParamsBuilder::default()
    }
}

impl SymmetryGroupDetectionParamsBuilder {
    pub fn moi_thresholds(&mut self, threshs: &[f64]) -> &mut Self {
        self.moi_thresholds = Some(threshs.to_vec());
        self
    }

    pub fn distance_thresholds(&mut self, threshs: &[f64]) -> &mut Self {
        self.distance_thresholds = Some(threshs.to_vec());
        self
    }
}

impl fmt::Display for SymmetryGroupDetectionParams {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let threshs = self
            .moi_thresholds
            .iter()
            .cartesian_product(self.distance_thresholds.iter());
        let nthreshs = threshs.clone().count();
        if nthreshs == 1 {
            writeln!(f, "Fixed thresholds:")?;
            writeln!(f, "  MoI threshold: {:.3e}", self.moi_thresholds[0])?;
            writeln!(f, "  Geo threshold: {:.3e}", self.distance_thresholds[0])?;
        } else {
            writeln!(f, "Variable thresholds:")?;
            writeln!(
                f,
                "  MoI thresholds: {}",
                self.moi_thresholds
                    .iter()
                    .map(|v| format!("{v:.3e}"))
                    .join(", ")
            )?;
            writeln!(
                f,
                "  Geo thresholds: {}",
                self.distance_thresholds
                    .iter()
                    .map(|v| format!("{v:.3e}"))
                    .join(", ")
            )?;
            writeln!(f)?;
        }

        if self.magnetic_fields.is_some() || self.electric_fields.is_some() {
            if self.field_origin_com {
                writeln!(f, "Field origins relative to: molecule's centre of mass")?;
            } else {
                writeln!(f, "Field origins relative to: space-fixed origin")?;
            }
        }

        if let Some(magnetic_fields) = self.magnetic_fields.as_ref() {
            writeln!(f, "Magnetic fields:")?;
            for (origin, field) in magnetic_fields.iter() {
                writeln!(
                    f,
                    "  ({}) ± ({})",
                    origin.iter().map(|x| format!("{x:+.3}")).join(", "),
                    field.iter().map(|x| format!("{x:+.3}")).join(", "),
                )?;
            }
            writeln!(f)?;
        }

        if let Some(electric_fields) = self.electric_fields.as_ref() {
            writeln!(f, "Electric fields:")?;
            for (origin, field) in electric_fields.iter() {
                writeln!(
                    f,
                    "  ({}) + ({})",
                    origin.iter().map(|x| format!("{x:+.3}")).join(", "),
                    field.iter().map(|x| format!("{x:+.3}")).join(", "),
                )?;
            }
            writeln!(f)?;
        }

        writeln!(
            f,
            "Consider time reversal: {}",
            nice_bool(self.time_reversal)
        )?;
        writeln!(
            f,
            "Report symmetry elements/generators: {}",
            nice_bool(self.write_symmetry_elements)
        )?;
        writeln!(
            f,
            "Save symmetry-group detection results to file: {}",
            if let Some(name) = self.result_save_name.as_ref() {
                format!("{name}{}", QSym2FileType::Sym.ext())
            } else {
                nice_bool(false)
            }
        )?;
        writeln!(f)?;

        Ok(())
    }
}

// ------
// Result
// ------

/// A structure to contain symmetry-group detection results.
#[derive(Clone, Builder, Debug, Serialize, Deserialize)]
pub struct SymmetryGroupDetectionResult {
    /// The control parameters used to obtain this set of results.
    pub parameters: SymmetryGroupDetectionParams,

    /// The [`PreSymmetry`] structure containing basic geometrical information of the system prior
    /// to symmetry-group detection.
    pub pre_symmetry: PreSymmetry,

    /// The [`Symmetry`] structure containing unitary symmetry information of the system.
    pub unitary_symmetry: Symmetry,

    /// The [`Symmetry`] structure containing magnetic symmetry information of the system. This is
    /// only present if time-reversal symmetry has been considered.
    #[builder(default = "None")]
    pub magnetic_symmetry: Option<Symmetry>,
}

impl SymmetryGroupDetectionResult {
    /// Returns a builder to construct a [`SymmetryGroupDetectionResult`] structure.
    fn builder() -> SymmetryGroupDetectionResultBuilder {
        SymmetryGroupDetectionResultBuilder::default()
    }

    /// Writes the symmetry elements (unitary and magnetic if available) found in a nicely
    /// formatted table.
    fn write_symmetry_elements(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(magnetic_symmetry) = self.magnetic_symmetry.as_ref() {
            write_subtitle(
                f,
                &format!(
                    "Symmetry element report for magnetic group {}",
                    magnetic_symmetry
                        .group_name
                        .as_ref()
                        .unwrap_or(&"?".to_string()),
                ),
            )?;
            writeln!(f)?;
            write_element_table(f, magnetic_symmetry)?;
            writeln!(f)?;
        }

        write_subtitle(
            f,
            &format!(
                "Symmetry element report for unitary group {}",
                self.unitary_symmetry
                    .group_name
                    .as_ref()
                    .unwrap_or(&"?".to_string())
            ),
        )?;
        writeln!(f)?;
        write_element_table(f, &self.unitary_symmetry)?;
        Ok(())
    }
}

impl fmt::Display for SymmetryGroupDetectionResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(highest_mag_sym) = self.magnetic_symmetry.as_ref() {
            let n_mag_elements = if highest_mag_sym.is_infinite() {
                "∞".to_string()
            } else {
                highest_mag_sym.n_elements().to_string()
            };
            writeln!(
                f,
                "Highest mag. group found: {} ({} {})",
                highest_mag_sym
                    .group_name
                    .as_ref()
                    .unwrap_or(&"?".to_string()),
                n_mag_elements,
                if n_mag_elements != "1" {
                    "symmetry elements"
                } else {
                    "symmetry element"
                }
            )?;
        }

        let n_uni_elements = if self.unitary_symmetry.is_infinite() {
            "∞".to_string()
        } else {
            self.unitary_symmetry.n_elements().to_string()
        };
        writeln!(
            f,
            "Highest uni. group found: {} ({} {})",
            self.unitary_symmetry
                .group_name
                .as_ref()
                .unwrap_or(&"?".to_string()),
            n_uni_elements,
            if n_uni_elements != "1" {
                "symmetry elements"
            } else {
                "symmetry element"
            }
        )?;
        writeln!(
            f,
            "  Associated MoI threshold: {:.3e}",
            self.pre_symmetry.moi_threshold
        )?;
        writeln!(
            f,
            "  Associated geo threshold: {:.3e}",
            self.pre_symmetry.dist_threshold
        )?;
        writeln!(f)?;

        if self.parameters.write_symmetry_elements {
            self.write_symmetry_elements(f)?;
        }

        Ok(())
    }
}

// ------
// Driver
// ------

/// A driver for symmetry-group detection.
#[derive(Clone, Builder)]
pub struct SymmetryGroupDetectionDriver<'a> {
    /// The control parameters for symmetry group detection.
    parameters: &'a SymmetryGroupDetectionParams,

    /// A path to a `.xyz` file specifying the geometry of the molecule for symmetry analysis.
    /// Only one of this or [`Self::molecule`] should be specified.
    #[builder(default = "None")]
    xyz: Option<String>,

    /// A molecule for symmetry analysis. Only one of this or [`Self::xyz`] should be specified.
    #[builder(default = "None")]
    molecule: Option<&'a Molecule>,

    /// The result of the symmetry-group detection.
    #[builder(setter(skip), default = "None")]
    result: Option<SymmetryGroupDetectionResult>,
}

impl<'a> SymmetryGroupDetectionDriver<'a> {
    /// Returns a builder to construct a [`SymmetryGroupDetectionResult`] structure.
    pub fn builder() -> SymmetryGroupDetectionDriverBuilder<'a> {
        SymmetryGroupDetectionDriverBuilder::default()
    }

    /// Executes symmetry-group detection.
    fn detect_symmetry_group(&mut self) -> Result<(), anyhow::Error> {
        log_title("Symmetry-Group Detection");
        log::info!(target: "qsym2-output", "");
        let params = self.parameters;
        params.log_output_display();

        let smallest_dist_thresh = *params
            .distance_thresholds
            .iter()
            .min_by(|x, y| {
                x.partial_cmp(y)
                    .expect("Unable to determine the smallest distance threshold.")
            })
            .ok_or_else(|| format_err!("Unable to determine the smallest distance threshold."))?;
        let target_mol = match (self.molecule, self.xyz.as_ref()) {
            (Some(molecule), None) => Molecule::from_atoms(
                &molecule.get_all_atoms().into_iter().cloned().collect_vec(),
                smallest_dist_thresh,
            ),
            (None, Some(xyz)) => Molecule::from_xyz(xyz, smallest_dist_thresh),
            _ => bail!("Neither or both `molecule` and `xyz` are specified."),
        };
        log::info!(target: "qsym2-output", "Molecule for symmetry-group detection:");
        target_mol.log_output_display();
        log::info!(target: "qsym2-output", "");

        let threshs = params
            .moi_thresholds
            .iter()
            .cartesian_product(params.distance_thresholds.iter());
        let nthreshs = threshs.clone().count();

        log_subtitle("Threshold-scanning symmetry-group detection");
        log::info!(target: "qsym2-output", "");

        let count_length = usize::try_from(nthreshs.ilog10() + 2).map_err(|_| {
            format_err!("Unable to convert `{}` to `usize`.", nthreshs.ilog10() + 2)
        })?;
        log::info!(target: "qsym2-output", "{}", "┈".repeat(count_length + 75));
        log::info!(
            target: "qsym2-output",
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
        log::info!(target: "qsym2-output", "{}", "┈".repeat(count_length + 75));
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
                _ => bail!("Neither or both `molecule` and `xyz` are specified.")
            };

            // Add any magnetic fields
            let global_origin = if params.field_origin_com {
                mol.calc_com() - Point3::origin()
            } else {
                Vector3::zeros()
            };
            if let Some(magnetic_fields) = params.magnetic_fields.as_ref() {
                if mol.magnetic_atoms.is_some() {
                    bail!("Magnetic fields already present. Additional magnetic fields cannot be added.")
                } else {
                    let magnetic_atoms = magnetic_fields.iter().flat_map(|(origin, vec)| {
                        Ok::<[Atom; 2], anyhow::Error>([
                            Atom::new_special(AtomKind::Magnetic(true), origin + global_origin + vec, *dist_thresh).ok_or_else(||
                                format_err!("Cannot construct a fictitious magnetic atom.")
                            )?,
                            Atom::new_special(AtomKind::Magnetic(false), origin + global_origin - vec, *dist_thresh).ok_or_else(||
                                format_err!("Cannot construct a fictitious magnetic atom.")
                            )?,
                        ])
                    }).flatten().collect_vec();
                    mol.magnetic_atoms = Some(magnetic_atoms);
                }
            }

            // Add any electric fields
            if let Some(electric_fields) = params.electric_fields.as_ref() {
                if mol.electric_atoms.is_some() {
                    bail!("Electric fields already present. Additional electric fields cannot be added.")
                } else {
                    let electric_atoms = electric_fields.iter().flat_map(|(origin, vec)| {
                        Atom::new_special(AtomKind::Electric(true), origin + global_origin + vec, *dist_thresh).ok_or_else(||
                                format_err!("Cannot construct a fictitious electric atom.")
                            )
                    }).collect_vec();
                    mol.electric_atoms = Some(electric_atoms);
                }
            }

            // Perform symmetry-group detection
            // A recentred copy of the molecule will be used for all symmetry-group detection.
            let presym = PreSymmetry::builder()
                .moi_threshold(*moi_thresh)
                .molecule(&mol)
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
                    target: "qsym2-output",
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
                    target: "qsym2-output",
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
        log::info!(target: "qsym2-output", "{}", "┈".repeat(count_length + 75));
        log::info!(target: "qsym2-output", "(The number of symmetry elements is not the same as the order of the group.)");
        log::info!(target: "qsym2-output", "");

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

        self.result = SymmetryGroupDetectionResult::builder()
            .parameters(params.clone())
            .pre_symmetry(highest_presym)
            .unitary_symmetry(highest_uni_sym)
            .magnetic_symmetry(highest_mag_sym_opt)
            .build()
            .ok();

        // Save symmetry-group detection result, if requested
        if let Some(pd_res) = self.result.as_ref() {
            pd_res.log_output_display();
            if let Some(name) = params.result_save_name.as_ref() {
                write_qsym2(name, QSym2FileType::Sym, pd_res)?;
                log::info!(
                    target: "qsym2-output",
                    "Symmetry-group detection results saved as {name}{}.",
                    QSym2FileType::Sym.ext()
             );
                log::info!(target: "qsym2-output", "");
            }
        }

        Ok(())
    }
}

impl QSym2Driver for SymmetryGroupDetectionDriver<'_> {
    type Outcome = SymmetryGroupDetectionResult;

    fn result(&self) -> Result<&Self::Outcome, anyhow::Error> {
        self.result
            .as_ref()
            .ok_or_else(|| format_err!("No symmetry-group detection results found."))
    }

    fn run(&mut self) -> Result<(), anyhow::Error> {
        self.detect_symmetry_group()
    }
}

// =========
// Functions
// =========

/// Writes symmetry elements/generators in a [`Symmetry`] structure in a nicely formatted table.
fn write_element_table(f: &mut fmt::Formatter<'_>, sym: &Symmetry) -> fmt::Result {
    let all_elements = sym
        .elements
        .iter()
        .map(|(kind, kind_elements)| (false, kind, kind_elements))
        .chain(
            sym.generators
                .iter()
                .map(|(kind, kind_generators)| (true, kind, kind_generators)),
        );
    all_elements
        .sorted_by_key(|(generator, kind, _)| {
            (
                *generator,
                kind.contains_antiunitary(),
                !matches!(kind, SymmetryElementKind::Proper(_)),
            )
        })
        .try_for_each(|(generator, kind, kind_elements)| {
            if !sym.is_infinite() && generator {
                Ok::<(), fmt::Error>(())
            } else {
                if generator {
                    writeln!(f, "> {kind} generators")?;
                } else {
                    writeln!(f, "> {kind} elements")?;
                }
                writeln!(f, "{}", "┈".repeat(54))?;
                writeln!(
                    f,
                    "{:>7} {:>7} {:>11}  {:>11}  {:>11}",
                    "", "Symbol", "x", "y", "z"
                )?;
                writeln!(f, "{}", "┈".repeat(54))?;
                kind_elements
                    .keys()
                    .sorted()
                    .into_iter()
                    .try_for_each(|order| {
                        let order_elements = kind_elements.get(order).unwrap_or_else(|| {
                            panic!("Elements/generators of order `{order}` cannot be retrieved.")
                        });
                        let any_element = order_elements
                            .get_index(0)
                            .expect("Unable to retrieve an element/generator of order `{order}`.");
                        let kind_str = match any_element.kind() {
                            SymmetryElementKind::Proper(_) => "",
                            SymmetryElementKind::ImproperInversionCentre(_) => {
                                " (inversion-centre)"
                            }
                            SymmetryElementKind::ImproperMirrorPlane(_) => " (mirror-plane)",
                        };
                        let au_str = match any_element.contains_antiunitary() {
                            None => "",
                            Some(AntiunitaryKind::TimeReversal) => " (time-reversed)",
                            Some(AntiunitaryKind::ComplexConjugation) => " (complex-conjugated)",
                        };
                        writeln!(f, " Order: {order}{au_str}{kind_str}")?;
                        order_elements.iter().try_for_each(|element| {
                            let axis = element.raw_axis();
                            writeln!(
                                f,
                                "{:>7} {:>7} {:>+11.7}  {:>+11.7}  {:>+11.7}",
                                element.get_simplified_symbol(),
                                element.get_full_symbol(),
                                axis[0],
                                axis[1],
                                axis[2]
                            )?;
                            Ok::<(), fmt::Error>(())
                        })?;
                        Ok::<(), fmt::Error>(())
                    })?;
                writeln!(f, "{}", "┈".repeat(54))?;
                writeln!(f)?;
                Ok::<(), fmt::Error>(())
            }
        })?;
    Ok(())
}
