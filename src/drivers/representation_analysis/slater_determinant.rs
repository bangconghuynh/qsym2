//! Driver for symmetry analysis of Slater determinants.

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::ops::Mul;

use anyhow::{self, bail, ensure, format_err};
use approx::abs_diff_eq;
use derive_builder::Builder;
use duplicate::duplicate_item;
use indexmap::IndexMap;
use itertools::Itertools;
use ndarray::{Array2, Array4, s};
use ndarray_linalg::types::Lapack;
use num::ToPrimitive;
use num_complex::{Complex, ComplexFloat};
use num_traits::Float;
use num_traits::real::Real;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::analysis::{
    EigenvalueComparisonMode, Orbit, Overlap, ProjectionDecomposition, RepAnalysis,
    log_overlap_eigenvalues,
};
use crate::angmom::spinor_rotation_3d::{SpinConstraint, SpinOrbitCoupled, StructureConstraint};
use crate::chartab::chartab_group::CharacterProperties;
use crate::chartab::chartab_symbols::ReducibleLinearSpaceSymbol;
use crate::chartab::{CharacterTable, SubspaceDecomposable};
use crate::drivers::QSym2Driver;
use crate::drivers::representation_analysis::angular_function::{
    AngularFunctionRepAnalysisParams, find_angular_function_representation,
    find_spinor_function_representation,
};
use crate::drivers::representation_analysis::{
    CharacterTableDisplay, MagneticSymmetryAnalysisKind, fn_construct_magnetic_group,
    fn_construct_unitary_group, log_bao, log_cc_transversal,
};
use crate::drivers::symmetry_group_detection::SymmetryGroupDetectionResult;
use crate::group::{GroupProperties, MagneticRepresentedGroup, UnitaryRepresentedGroup};
use crate::io::format::{
    QSym2Output, log_subtitle, nice_bool, qsym2_output, write_subtitle, write_title,
};
use crate::symmetry::symmetry_element::symmetry_operation::{
    SpecialSymmetryTransformation, SymmetryOperation,
};
use crate::symmetry::symmetry_group::{
    MagneticRepresentedSymmetryGroup, SymmetryGroupProperties, UnitaryRepresentedSymmetryGroup,
};
use crate::symmetry::symmetry_symbols::{
    MirrorParity, MullikenIrcorepSymbol, SymmetryClassSymbol, deduce_mirror_parities,
};
use crate::symmetry::symmetry_transformation::SymmetryTransformationKind;
use crate::target::density::density_analysis::DensitySymmetryOrbit;
use crate::target::determinant::SlaterDeterminant;
use crate::target::determinant::determinant_analysis::SlaterDeterminantSymmetryOrbit;
use crate::target::orbital::orbital_analysis::generate_det_mo_orbits;

#[cfg(test)]
#[path = "slater_determinant_tests.rs"]
mod slater_determinant_tests;

// ==================
// Struct definitions
// ==================

// ----------
// Parameters
// ----------

const fn default_true() -> bool {
    true
}
const fn default_symbolic() -> Option<CharacterTableDisplay> {
    Some(CharacterTableDisplay::Symbolic)
}

/// Structure containing control parameters for Slater determinant representation analysis.
#[derive(Clone, Builder, Debug, Serialize, Deserialize)]
pub struct SlaterDeterminantRepAnalysisParams<T: From<f64>> {
    /// Threshold for checking if subspace multiplicities are integral.
    pub integrality_threshold: T,

    /// Threshold for determining zero eigenvalues in the orbit overlap matrix.
    pub linear_independence_threshold: T,

    /// Boolean indicating if molecular orbital symmetries are to be analysed alongside the overall
    /// determinantal symmetry.
    #[builder(default = "true")]
    #[serde(default = "default_true")]
    pub analyse_mo_symmetries: bool,

    /// Boolean indicating if molecular orbital mirror parities are to be analysed alongside
    /// molecular orbital symmetries.
    #[builder(default = "false")]
    #[serde(default)]
    pub analyse_mo_mirror_parities: bool,

    /// Boolean indicating if symmetry projection should be performed for molecular orbitals.
    #[builder(default = "false")]
    #[serde(default)]
    pub analyse_mo_symmetry_projections: bool,

    /// Boolean indicating if density symmetries are to be analysed alongside wavefunction symmetries
    /// for this determinant.
    #[builder(default = "false")]
    #[serde(default)]
    pub analyse_density_symmetries: bool,

    /// Option indicating if the magnetic group is to be used for symmetry analysis, and if so,
    /// whether unitary representations or unitary-antiunitary corepresentations should be used.
    #[builder(default = "None")]
    #[serde(default)]
    pub use_magnetic_group: Option<MagneticSymmetryAnalysisKind>,

    /// Boolean indicating if the double group is to be used for symmetry analysis.
    #[builder(default = "false")]
    #[serde(default)]
    pub use_double_group: bool,

    /// Boolean indicating if the Cayley table of the group, if available, should be used to speed
    /// up the computation of orbit overlap matrices.
    #[builder(default = "true")]
    #[serde(default = "default_true")]
    pub use_cayley_table: bool,

    /// The kind of symmetry transformation to be applied on the reference determinant to generate
    /// the orbit for symmetry analysis.
    #[builder(default = "SymmetryTransformationKind::Spatial")]
    #[serde(default)]
    pub symmetry_transformation_kind: SymmetryTransformationKind,

    /// Option indicating if the character table of the group used for symmetry analysis is to be
    /// printed out.
    #[builder(default = "Some(CharacterTableDisplay::Symbolic)")]
    #[serde(default = "default_symbolic")]
    pub write_character_table: Option<CharacterTableDisplay>,

    /// Boolean indicating if the eigenvalues of the orbit overlap matrix are to be printed out.
    #[builder(default = "true")]
    #[serde(default = "default_true")]
    pub write_overlap_eigenvalues: bool,

    /// The comparison mode for filtering out orbit overlap eigenvalues.
    #[builder(default = "EigenvalueComparisonMode::Modulus")]
    #[serde(default)]
    pub eigenvalue_comparison_mode: EigenvalueComparisonMode,

    /// The finite order to which any infinite-order symmetry element is reduced, so that a finite
    /// subgroup of an infinite group can be used for the symmetry analysis.
    #[builder(default = "None")]
    #[serde(default)]
    pub infinite_order_to_finite: Option<u32>,
}

impl<T> SlaterDeterminantRepAnalysisParams<T>
where
    T: Float + From<f64>,
{
    /// Returns a builder to construct a [`SlaterDeterminantRepAnalysisParams`] structure.
    pub fn builder() -> SlaterDeterminantRepAnalysisParamsBuilder<T> {
        SlaterDeterminantRepAnalysisParamsBuilder::default()
    }
}

impl Default for SlaterDeterminantRepAnalysisParams<f64> {
    fn default() -> Self {
        Self::builder()
            .integrality_threshold(1e-7)
            .linear_independence_threshold(1e-7)
            .build()
            .expect("Unable to construct a default `SlaterDeterminantRepAnalysisParams<f64>`.")
    }
}

impl<T> fmt::Display for SlaterDeterminantRepAnalysisParams<T>
where
    T: From<f64> + fmt::LowerExp + fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "Integrality threshold: {:.3e}",
            self.integrality_threshold
        )?;
        writeln!(
            f,
            "Linear independence threshold: {:.3e}",
            self.linear_independence_threshold
        )?;
        writeln!(
            f,
            "Orbit eigenvalue comparison mode: {}",
            self.eigenvalue_comparison_mode
        )?;
        writeln!(
            f,
            "Write overlap eigenvalues: {}",
            nice_bool(self.write_overlap_eigenvalues)
        )?;
        writeln!(f)?;
        writeln!(
            f,
            "Analyse molecular orbital symmetry: {}",
            nice_bool(self.analyse_mo_symmetries)
        )?;
        writeln!(
            f,
            "Analyse molecular orbital symmetry projection: {}",
            nice_bool(self.analyse_mo_symmetry_projections)
        )?;
        writeln!(
            f,
            "Analyse molecular orbital mirror parity: {}",
            nice_bool(self.analyse_mo_mirror_parities)
        )?;
        writeln!(
            f,
            "Analyse density symmetry: {}",
            nice_bool(self.analyse_density_symmetries)
        )?;
        writeln!(f)?;
        writeln!(
            f,
            "Use magnetic group for analysis: {}",
            match self.use_magnetic_group {
                None => "no",
                Some(MagneticSymmetryAnalysisKind::Representation) =>
                    "yes, using unitary representations",
                Some(MagneticSymmetryAnalysisKind::Corepresentation) =>
                    "yes, using magnetic corepresentations",
            }
        )?;
        writeln!(
            f,
            "Use double group for analysis: {}",
            nice_bool(self.use_double_group)
        )?;
        writeln!(
            f,
            "Use Cayley table for orbit overlap matrices: {}",
            nice_bool(self.use_cayley_table)
        )?;
        if let Some(finite_order) = self.infinite_order_to_finite {
            writeln!(f, "Infinite order to finite: {finite_order}")?;
        }
        writeln!(
            f,
            "Symmetry transformation kind: {}",
            self.symmetry_transformation_kind
        )?;
        writeln!(f)?;
        writeln!(
            f,
            "Write character table: {}",
            if let Some(chartab_display) = self.write_character_table.as_ref() {
                format!("yes, {}", chartab_display.to_string().to_lowercase())
            } else {
                "no".to_string()
            }
        )?;

        Ok(())
    }
}

// ------
// Result
// ------

/// Structure to contain Slater determinant representation analysis results.
#[derive(Clone, Builder)]
pub struct SlaterDeterminantRepAnalysisResult<'a, G, T, SC>
where
    G: SymmetryGroupProperties + Clone,
    G::CharTab: SubspaceDecomposable<T>,
    T: ComplexFloat + Lapack,
    SC: StructureConstraint + fmt::Display,
    <T as ComplexFloat>::Real: From<f64> + fmt::LowerExp + fmt::Debug,
{
    /// The control parameters used to obtain this set of Slater determinant representation
    /// analysis results.
    parameters: &'a SlaterDeterminantRepAnalysisParams<<T as ComplexFloat>::Real>,

    /// The Slater determinant being analysed.
    determinant: &'a SlaterDeterminant<'a, T, SC>,

    /// The group used for the representation analysis.
    group: G,

    /// The deduced overall symmetry of the determinant.
    determinant_symmetry: Result<<G::CharTab as SubspaceDecomposable<T>>::Decomposition, String>,

    /// The deduced symmetries of the molecular orbitals constituting the determinant, if required.
    mo_symmetries: Option<Vec<Vec<Option<<G::CharTab as SubspaceDecomposable<T>>::Decomposition>>>>,

    /// The deduced symmetry projections of the molecular orbitals constituting the determinant, if required.
    mo_symmetry_projections: Option<
        Vec<
            Vec<
                Option<
                    Vec<(
                        <<G as CharacterProperties>::CharTab as CharacterTable>::RowSymbol,
                        Complex<f64>,
                    )>,
                >,
            >,
        >,
    >,

    /// The deduced mirror parities of the molecular orbitals constituting the determinant, if required.
    mo_mirror_parities:
        Option<Vec<Vec<Option<IndexMap<SymmetryClassSymbol<SymmetryOperation>, MirrorParity>>>>>,

    /// The overlap eigenvalues above and below the linear independence threshold for each
    /// molecular orbital symmetry deduction.
    mo_symmetries_thresholds: Option<Vec<Vec<(Option<T>, Option<T>)>>>,

    /// The deduced symmetries of the various densities constructible from the determinant, if
    /// required. In each tuple, the first element gives a description of the density corresponding
    /// to the symmetry result.
    determinant_density_symmetries: Option<
        Vec<(
            String,
            Result<<G::CharTab as SubspaceDecomposable<T>>::Decomposition, String>,
        )>,
    >,

    /// The deduced symmetries of the total densities of the molecular orbitals constituting the
    /// determinant, if required.
    mo_density_symmetries:
        Option<Vec<Vec<Option<<G::CharTab as SubspaceDecomposable<T>>::Decomposition>>>>,
}

impl<'a, G, T, SC> SlaterDeterminantRepAnalysisResult<'a, G, T, SC>
where
    G: SymmetryGroupProperties + Clone,
    G::CharTab: SubspaceDecomposable<T>,
    T: ComplexFloat + Lapack,
    SC: StructureConstraint + Clone + fmt::Display,
    <T as ComplexFloat>::Real: From<f64> + fmt::LowerExp + fmt::Debug,
{
    /// Returns a builder to construct a new [`SlaterDeterminantRepAnalysisResultBuilder`]
    /// structure.
    fn builder() -> SlaterDeterminantRepAnalysisResultBuilder<'a, G, T, SC> {
        SlaterDeterminantRepAnalysisResultBuilder::default()
    }
}

impl<'a, G, T, SC> SlaterDeterminantRepAnalysisResult<'a, G, T, SC>
where
    G: SymmetryGroupProperties + Clone,
    G::CharTab: SubspaceDecomposable<T>,
    T: ComplexFloat + Lapack,
    SC: StructureConstraint + fmt::Display,
    <T as ComplexFloat>::Real: From<f64> + fmt::LowerExp + fmt::Debug,
{
    /// Returns the group used for the representation analysis.
    pub fn group(&self) -> &G {
        &self.group
    }

    /// Returns the determinant symmetry obtained from the analysis result.
    pub fn determinant_symmetry(
        &self,
    ) -> &Result<<G::CharTab as SubspaceDecomposable<T>>::Decomposition, String> {
        &self.determinant_symmetry
    }

    /// Returns the deduced symmetries of the molecular orbitals constituting the determinant, if required.
    pub fn mo_symmetries(
        &self,
    ) -> &Option<Vec<Vec<Option<<G::CharTab as SubspaceDecomposable<T>>::Decomposition>>>> {
        &self.mo_symmetries
    }

    /// Returns the deduced symmetry projections of the molecular orbitals constituting the determinant, if required.
    pub fn mo_symmetry_projections(
        &self,
    ) -> &Option<
        Vec<
            Vec<
                Option<
                    Vec<(
                        <<G as CharacterProperties>::CharTab as CharacterTable>::RowSymbol,
                        Complex<f64>,
                    )>,
                >,
            >,
        >,
    > {
        &self.mo_symmetry_projections
    }

    /// Returns the deduced symmetries of the various densities constructible from the determinant,
    /// if required. In each tuple, the first element gives a description of the density corresponding
    /// to the symmetry result.
    pub fn determinant_density_symmetries(
        &self,
    ) -> &Option<
        Vec<(
            String,
            Result<<G::CharTab as SubspaceDecomposable<T>>::Decomposition, String>,
        )>,
    > {
        &self.determinant_density_symmetries
    }

    /// Returns the deduced symmetries of the total densities of the molecular orbitals constituting
    /// the determinant, if required.
    pub fn mo_density_symmetries(
        &self,
    ) -> &Option<Vec<Vec<Option<<G::CharTab as SubspaceDecomposable<T>>::Decomposition>>>> {
        &self.mo_density_symmetries
    }
}

impl<'a, G, T, SC> fmt::Display for SlaterDeterminantRepAnalysisResult<'a, G, T, SC>
where
    G: SymmetryGroupProperties + Clone,
    G::CharTab: SubspaceDecomposable<T>,
    T: ComplexFloat + Lapack,
    SC: StructureConstraint + fmt::Display,
    <T as ComplexFloat>::Real: From<f64> + fmt::LowerExp + fmt::Debug + fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write_subtitle(f, "Orbit-based symmetry analysis results")?;
        writeln!(f)?;
        writeln!(
            f,
            "> Group: {} ({})",
            self.group
                .finite_subgroup_name()
                .map(|subgroup_name| format!("{} > {}", self.group.name(), subgroup_name))
                .unwrap_or(self.group.name()),
            self.group.group_type().to_string().to_lowercase()
        )?;
        writeln!(f)?;
        writeln!(f, "> Overall determinantal result")?;
        writeln!(
            f,
            "  Energy  : {}",
            self.determinant
                .energy()
                .map(|e| e.to_string())
                .unwrap_or_else(|err| format!("-- ({err})"))
        )?;
        writeln!(
            f,
            "  Symmetry: {}",
            self.determinant_symmetry
                .as_ref()
                .map(|s| s.to_string())
                .unwrap_or_else(|err| format!("-- ({err})"))
        )?;
        writeln!(f)?;

        if let Some(den_syms) = self.determinant_density_symmetries.as_ref() {
            writeln!(f, "> Overall determinantal density result")?;
            let den_type_width = den_syms
                .iter()
                .map(|(den_type, _)| den_type.chars().count())
                .max()
                .unwrap_or(7)
                .max(7);
            for (den_type, den_sym_res) in den_syms.iter() {
                writeln!(
                    f,
                    "  {den_type:<den_type_width$}: {}",
                    den_sym_res
                        .as_ref()
                        .map(|e| e.to_string())
                        .unwrap_or_else(|err| format!("-- ({err})"))
                )?;
            }
            writeln!(f)?;
        }

        if let Some(mo_symmetries) = self.mo_symmetries.as_ref() {
            let mo_spin_index_length = 4;
            let mo_index_length = mo_symmetries
                .iter()
                .map(|spin_mo_symmetries| spin_mo_symmetries.len())
                .max()
                .and_then(|max_mo_length| usize::try_from(max_mo_length.ilog10() + 2).ok())
                .unwrap_or(4);
            let mo_occ_length = 5;
            let mo_symmetry_length = mo_symmetries
                .iter()
                .flat_map(|spin_mo_symmetries| {
                    spin_mo_symmetries.iter().map(|mo_sym| {
                        mo_sym
                            .as_ref()
                            .map(|sym| sym.to_string())
                            .unwrap_or("--".to_string())
                            .chars()
                            .count()
                    })
                })
                .max()
                .unwrap_or(0)
                .max(8);
            let mo_energies_opt = self.determinant.mo_energies();
            let mo_energy_length = mo_energies_opt
                .map(|mo_energies| {
                    mo_energies
                        .iter()
                        .flat_map(|spin_mo_energies| {
                            spin_mo_energies.map(|v| format!("{v:+.7}").chars().count())
                        })
                        .max()
                        .unwrap_or(0)
                })
                .unwrap_or(0)
                .max(6);

            let mo_eig_above_length: usize = self
                .mo_symmetries_thresholds
                .as_ref()
                .map(|mo_symmetries_thresholds| {
                    mo_symmetries_thresholds
                        .iter()
                        .flat_map(|spin_mo_symmetries_thresholds| {
                            spin_mo_symmetries_thresholds.iter().map(|(above, _)| {
                                above
                                    .as_ref()
                                    .map(|eig| format!("{eig:+.3e}"))
                                    .unwrap_or("--".to_string())
                                    .chars()
                                    .count()
                            })
                        })
                        .max()
                        .unwrap_or(10)
                        .max(10)
                })
                .unwrap_or(10);
            let mo_eig_below_length: usize = self
                .mo_symmetries_thresholds
                .as_ref()
                .map(|mo_symmetries_thresholds| {
                    mo_symmetries_thresholds
                        .iter()
                        .flat_map(|spin_mo_symmetries_thresholds| {
                            spin_mo_symmetries_thresholds.iter().map(|(_, below)| {
                                below
                                    .as_ref()
                                    .map(|eig| format!("{eig:+.3e}"))
                                    .unwrap_or("--".to_string())
                                    .chars()
                                    .count()
                            })
                        })
                        .max()
                        .unwrap_or(10)
                        .max(10)
                })
                .unwrap_or(10);

            let precision = Real::ceil(ComplexFloat::abs(ComplexFloat::log10(
                self.parameters.linear_independence_threshold,
            )))
            .to_usize()
            .expect("Unable to convert the linear independence threshold exponent to `usize`.");
            let mo_sym_projections_str_opt =
                self.mo_symmetry_projections
                    .as_ref()
                    .map(|mo_sym_projectionss| {
                        mo_sym_projectionss
                            .iter()
                            .enumerate()
                            .map(|(ispin, mo_sym_projections)| {
                                mo_sym_projections
                                    .iter()
                                    .enumerate()
                                    .map(|(imo, mo_sym_projection)| {
                                        mo_sym_projection
                                            .as_ref()
                                            .map(|sym_proj| {
                                                if let Some(mo_symmetriess) = self.mo_symmetries() {
                                                    if let Some(sym) = &mo_symmetriess[ispin][imo] {
                                                        let sym_proj_hashmap = sym_proj
                                                            .iter()
                                                            .cloned()
                                                            .collect::<HashMap<_, _>>();
                                                        sym.subspaces()
                                                            .iter()
                                                            .map(|(subspace, _)| {
                                                                format!(
                                                                    "{subspace}: {}",
                                                                    sym_proj_hashmap
                                                                        .get(subspace)
                                                                        .map(|composition| {
                                                                            if abs_diff_eq!(
                                                                                composition.im,
                                                                                0.0,
                                                                                epsilon = self.parameters.linear_independence_threshold.to_f64().expect("Unable to convert the linear independence threshold to `f64`.")
                                                                            ) {
                                                                                format!(
                                                                                    "{:+.precision$}",
                                                                                    composition.re
                                                                                )
                                                                            } else {
                                                                                format!(
                                                                            "{composition:+.precision$}")
                                                                            }
                                                                        })
                                                                        .unwrap_or(
                                                                            "--".to_string()
                                                                        )
                                                                )
                                                            })
                                                            .join(", ")
                                                    } else {
                                                        "--".to_string()
                                                    }
                                                } else {
                                                    "--".to_string()
                                                }
                                            })
                                            .unwrap_or("--".to_string())
                                    })
                                    .collect::<Vec<_>>()
                            })
                            .collect::<Vec<_>>()
                    });
            let mo_sym_projections_length_opt =
                mo_sym_projections_str_opt
                    .as_ref()
                    .map(|mo_sym_projectionss| {
                        mo_sym_projectionss
                            .iter()
                            .flat_map(|mo_sym_projections| {
                                mo_sym_projections
                                    .iter()
                                    .map(|mo_sym_projection| mo_sym_projection.chars().count())
                            })
                            .max()
                            .unwrap_or(18)
                            .max(18)
                    });
            let mo_sym_projections_length = mo_sym_projections_length_opt.unwrap_or(0);
            let mo_sym_projections_gap = mo_sym_projections_length_opt.map(|_| 2).unwrap_or(0);
            let mo_sym_projections_heading = mo_sym_projections_length_opt
                .map(|_| "MO sym. projection")
                .unwrap_or("");

            let mirrors = self
                .group
                .filter_cc_symbols(|cc| cc.is_spatial_reflection());
            let mo_mirror_parities_length_opt = self.mo_mirror_parities.as_ref().map(|_| {
                let mirror_heading = mirrors.iter().map(|sigma| format!("p[{sigma}]")).join("  ");
                let length = mirror_heading.chars().count();
                (mirror_heading, length)
            });
            let mo_mirror_parities_gap = mo_mirror_parities_length_opt
                .as_ref()
                .map(|_| 2)
                .unwrap_or(0);
            let (mo_mirror_parities_heading, mo_mirror_parities_length) =
                mo_mirror_parities_length_opt.unwrap_or((String::new(), 0));

            let mo_den_symss_str_opt = self.mo_density_symmetries.as_ref().map(|mo_den_symss| {
                mo_den_symss
                    .iter()
                    .map(|mo_den_syms| {
                        mo_den_syms
                            .iter()
                            .map(|mo_den_sym| {
                                mo_den_sym
                                    .as_ref()
                                    .map(|sym| sym.to_string())
                                    .unwrap_or("--".to_string())
                            })
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            });
            let mo_density_length_opt = mo_den_symss_str_opt.as_ref().map(|mo_den_symss| {
                mo_den_symss
                    .iter()
                    .flat_map(|mo_den_syms| {
                        mo_den_syms
                            .iter()
                            .map(|mo_den_sym| mo_den_sym.chars().count())
                    })
                    .max()
                    .unwrap_or(12)
                    .max(12)
            });
            let mo_density_length = mo_density_length_opt.unwrap_or(0);
            let mo_density_gap = mo_density_length_opt.map(|_| 2).unwrap_or(0);
            let mo_density_heading = mo_density_length_opt.map(|_| "Density sym.").unwrap_or("");

            let table_width = 14
                + mo_spin_index_length
                + mo_index_length
                + mo_occ_length
                + mo_energy_length
                + mo_symmetry_length
                + mo_mirror_parities_gap
                + mo_mirror_parities_length
                + mo_eig_above_length
                + mo_eig_below_length
                + mo_sym_projections_gap
                + mo_sym_projections_length
                + mo_density_gap
                + mo_density_length;

            writeln!(f, "> Molecular orbital results")?;
            writeln!(
                f,
                "  Structure constraint: {}",
                self.determinant
                    .structure_constraint()
                    .to_string()
                    .to_lowercase()
            )?;
            if self.mo_mirror_parities.as_ref().is_some() {
                writeln!(f, "")?;
                writeln!(
                    f,
                    "Column p[σ] gives the parity under the reflection class σ: {} => even, {} => odd, {} => neither.",
                    MirrorParity::Even,
                    MirrorParity::Odd,
                    MirrorParity::Neither
                )?;
            }
            writeln!(f, "{}", "┈".repeat(table_width))?;
            if mo_density_length > 0 {
                writeln!(
                    f,
                    " {:>mo_spin_index_length$}  {:>mo_index_length$}  {:<mo_occ_length$}  {:<mo_energy_length$}  {:<mo_symmetry_length$}{}{:mo_mirror_parities_length$}  {:<mo_eig_above_length$}  {:<mo_eig_below_length$}{}{:mo_sym_projections_length$}{}{}",
                    "Spin",
                    "MO",
                    "Occ.",
                    "Energy",
                    "Symmetry",
                    " ".repeat(mo_mirror_parities_gap),
                    mo_mirror_parities_heading,
                    "Eig. above",
                    "Eig. below",
                    " ".repeat(mo_sym_projections_gap),
                    mo_sym_projections_heading,
                    " ".repeat(mo_density_gap),
                    mo_density_heading
                )?;
            } else if mo_sym_projections_length > 0 {
                writeln!(
                    f,
                    " {:>mo_spin_index_length$}  {:>mo_index_length$}  {:<mo_occ_length$}  {:<mo_energy_length$}  {:<mo_symmetry_length$}{}{:mo_mirror_parities_length$}  {:<mo_eig_above_length$}  {:<mo_eig_below_length$}{}{}",
                    "Spin",
                    "MO",
                    "Occ.",
                    "Energy",
                    "Symmetry",
                    " ".repeat(mo_mirror_parities_gap),
                    mo_mirror_parities_heading,
                    "Eig. above",
                    "Eig. below",
                    " ".repeat(mo_sym_projections_gap),
                    mo_sym_projections_heading,
                )?;
            } else {
                writeln!(
                    f,
                    " {:>mo_spin_index_length$}  {:>mo_index_length$}  {:<mo_occ_length$}  {:<mo_energy_length$}  {:<mo_symmetry_length$}{}{:mo_mirror_parities_length$}  {:<mo_eig_above_length$}  {}",
                    "Spin",
                    "MO",
                    "Occ.",
                    "Energy",
                    "Symmetry",
                    " ".repeat(mo_mirror_parities_gap),
                    mo_mirror_parities_heading,
                    "Eig. above",
                    "Eig. below",
                )?;
            };
            writeln!(f, "{}", "┈".repeat(table_width))?;

            let empty_string = String::new();
            for (spini, spin_mo_symmetries) in mo_symmetries.iter().enumerate() {
                writeln!(f, " Spin {spini}")?;
                for (moi, mo_sym) in spin_mo_symmetries.iter().enumerate() {
                    let occ_str = self
                        .determinant
                        .occupations()
                        .get(spini)
                        .and_then(|spin_occs| spin_occs.get(moi))
                        .map(|occ| format!("{occ:>.3}"))
                        .unwrap_or("--".to_string());
                    let mo_energy_str = mo_energies_opt
                        .and_then(|mo_energies| mo_energies.get(spini))
                        .and_then(|spin_mo_energies| spin_mo_energies.get(moi))
                        .map(|mo_energy| format!("{mo_energy:>+mo_energy_length$.7}"))
                        .unwrap_or("--".to_string());
                    let mo_sym_str = mo_sym
                        .as_ref()
                        .map(|sym| sym.to_string())
                        .unwrap_or("--".to_string());

                    let mo_mirror_parities_str = self
                        .mo_mirror_parities
                        .as_ref()
                        .and_then(|mo_mirror_paritiess| {
                            mo_mirror_paritiess
                                .get(spini)
                                .and_then(|spin_mo_mirror_parities| {
                                    spin_mo_mirror_parities
                                        .get(moi)
                                        .map(|mo_mirror_parities_opt| {
                                            mo_mirror_parities_opt
                                                .as_ref()
                                                .map(|mo_mirror_parities| {
                                                    mirrors
                                                        .iter()
                                                        .map(|sigma| {
                                                            let sigma_length =
                                                                sigma.to_string().chars().count()
                                                                    + 3;
                                                            mo_mirror_parities
                                                                .get(sigma)
                                                                .map(|parity| {
                                                                    format!(
                                                                        "{:^sigma_length$}",
                                                                        parity.to_string()
                                                                    )
                                                                })
                                                                .unwrap_or_else(|| {
                                                                    format!(
                                                                        "{:^sigma_length$}",
                                                                        "--"
                                                                    )
                                                                })
                                                        })
                                                        .join("  ")
                                                })
                                                .unwrap_or(String::new())
                                        })
                                })
                        })
                        .unwrap_or(String::new());

                    let (eig_above_str, eig_below_str) = self
                        .mo_symmetries_thresholds
                        .as_ref()
                        .map(|mo_symmetries_thresholds| {
                            mo_symmetries_thresholds
                                .get(spini)
                                .and_then(|spin_mo_symmetries_thresholds| {
                                    spin_mo_symmetries_thresholds.get(moi)
                                })
                                .map(|(eig_above_opt, eig_below_opt)| {
                                    (
                                        eig_above_opt
                                            .map(|eig_above| format!("{eig_above:>+.3e}"))
                                            .unwrap_or("--".to_string()),
                                        eig_below_opt
                                            .map(|eig_below| format!("{eig_below:>+.3e}"))
                                            .unwrap_or("--".to_string()),
                                    )
                                })
                                .unwrap_or(("--".to_string(), "--".to_string()))
                        })
                        .unwrap_or(("--".to_string(), "--".to_string()));

                    let mo_symmetry_projections_str = mo_sym_projections_str_opt
                        .as_ref()
                        .and_then(|mo_symmetry_projectionss| {
                            mo_symmetry_projectionss.get(spini).and_then(
                                |spin_mo_symmetry_projections| {
                                    spin_mo_symmetry_projections.get(moi)
                                },
                            )
                        })
                        .unwrap_or(&empty_string);

                    let mo_density_symmetries_str = mo_den_symss_str_opt
                        .as_ref()
                        .and_then(|mo_density_symmetriess| {
                            mo_density_symmetriess.get(spini).and_then(
                                |spin_mo_density_symmetries| spin_mo_density_symmetries.get(moi),
                            )
                        })
                        .unwrap_or(&empty_string);

                    match (mo_density_length, mo_sym_projections_length) {
                        (0, 0) => {
                            writeln!(
                                f,
                                " {spini:>mo_spin_index_length$}  \
                                {moi:>mo_index_length$}  \
                                {occ_str:<mo_occ_length$}  \
                                {mo_energy_str:<mo_energy_length$}  \
                                {mo_sym_str:<mo_symmetry_length$}\
                                {}{:mo_mirror_parities_length$}  \
                                {eig_above_str:<mo_eig_above_length$}  \
                                {eig_below_str}",
                                " ".repeat(mo_mirror_parities_gap),
                                mo_mirror_parities_str,
                            )?;
                        }
                        (_, 0) => {
                            writeln!(
                                f,
                                " {spini:>mo_spin_index_length$}  \
                                {moi:>mo_index_length$}  \
                                {occ_str:<mo_occ_length$}  \
                                {mo_energy_str:<mo_energy_length$}  \
                                {mo_sym_str:<mo_symmetry_length$}\
                                {}{:mo_mirror_parities_length$}  \
                                {eig_above_str:<mo_eig_above_length$}  \
                                {eig_below_str:<mo_eig_below_length$}  \
                                {mo_density_symmetries_str}",
                                " ".repeat(mo_mirror_parities_gap),
                                mo_mirror_parities_str,
                            )?;
                        }
                        (0, _) => {
                            writeln!(
                                f,
                                " {spini:>mo_spin_index_length$}  \
                                {moi:>mo_index_length$}  \
                                {occ_str:<mo_occ_length$}  \
                                {mo_energy_str:<mo_energy_length$}  \
                                {mo_sym_str:<mo_symmetry_length$}\
                                {}{:mo_mirror_parities_length$}  \
                                {eig_above_str:<mo_eig_above_length$}  \
                                {eig_below_str:<mo_eig_below_length$}  \
                                {mo_symmetry_projections_str}",
                                " ".repeat(mo_mirror_parities_gap),
                                mo_mirror_parities_str,
                            )?;
                        }
                        (_, _) => {
                            writeln!(
                                f,
                                " {spini:>mo_spin_index_length$}  \
                                {moi:>mo_index_length$}  \
                                {occ_str:<mo_occ_length$}  \
                                {mo_energy_str:<mo_energy_length$}  \
                                {mo_sym_str:<mo_symmetry_length$}\
                                {}{:mo_mirror_parities_length$}  \
                                {eig_above_str:<mo_eig_above_length$}  \
                                {eig_below_str:<mo_eig_below_length$}  \
                                {mo_symmetry_projections_str:<mo_sym_projections_length$}  \
                                {mo_density_symmetries_str}",
                                " ".repeat(mo_mirror_parities_gap),
                                mo_mirror_parities_str,
                            )?;
                        }
                    }
                }
            }

            writeln!(f, "{}", "┈".repeat(table_width))?;
            writeln!(f)?;
        }

        Ok(())
    }
}

impl<'a, G, T, SC> fmt::Debug for SlaterDeterminantRepAnalysisResult<'a, G, T, SC>
where
    G: SymmetryGroupProperties + Clone,
    G::CharTab: SubspaceDecomposable<T>,
    T: ComplexFloat + Lapack,
    SC: StructureConstraint + fmt::Display,
    <T as ComplexFloat>::Real: From<f64> + fmt::LowerExp + fmt::Debug + fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{self}")
    }
}

// ------
// Driver
// ------

// ~~~~~~~~~~~~~~~~~
// Struct definition
// ~~~~~~~~~~~~~~~~~

/// Driver structure for performing representation analysis on Slater determinants.
#[derive(Clone, Builder)]
#[builder(build_fn(validate = "Self::validate"))]
pub struct SlaterDeterminantRepAnalysisDriver<'a, G, T, SC>
where
    G: SymmetryGroupProperties + Clone,
    G::CharTab: SubspaceDecomposable<T>,
    T: ComplexFloat + Lapack,
    SC: StructureConstraint + fmt::Display,
    <T as ComplexFloat>::Real: From<f64> + fmt::LowerExp + fmt::Debug,
{
    /// The control parameters for Slater determinant representation analysis.
    parameters: &'a SlaterDeterminantRepAnalysisParams<<T as ComplexFloat>::Real>,

    /// The Slater determinant to be analysed.
    determinant: &'a SlaterDeterminant<'a, T, SC>,

    /// The result from symmetry-group detection on the underlying molecular structure of the
    /// Slater determinant.
    symmetry_group: &'a SymmetryGroupDetectionResult,

    /// The atomic-orbital overlap matrix of the underlying basis set used to describe the
    /// determinant. This is either for a single component corresponding to the basis functions
    /// specified by the basis angular order structure in [`Self::determinant`], or for *all*
    /// explicit components specified by the coefficients in [`Self::determinant`].
    sao: &'a Array2<T>,

    /// The complex-symmetric atomic-orbital overlap matrix of the underlying basis set used to
    /// describe the determinant. This is either for a single component corresponding to the basis
    /// functions specified by the basis angular order structure in [`Self::determinant`], or for
    /// *all* explicit components specified by the coefficients in [`Self::determinant`]. This is
    /// required if antiunitary symmetry operations are involved. If none is provided, this will be
    /// assumed to be the same as [`Self::sao`].
    #[builder(default = "None")]
    sao_h: Option<&'a Array2<T>>,

    /// The atomic-orbital four-centre spatial overlap matrix of the underlying basis set used to
    /// describe the determinant. This is only required for density symmetry analysis.
    #[builder(default = "None")]
    sao_spatial_4c: Option<&'a Array4<T>>,

    /// The complex-symmetric atomic-orbital four-centre spatial overlap matrix of the underlying
    /// basis set used to describe the determinant. This is only required for density symmetry
    /// analysis. This is required if antiunitary symmetry operations are involved. If none is
    /// provided, this will be assumed to be the same as [`Self::sao_spatial_4c`], if any.
    #[builder(default = "None")]
    sao_spatial_4c_h: Option<&'a Array4<T>>,

    /// The control parameters for symmetry analysis of angular functions.
    angular_function_parameters: &'a AngularFunctionRepAnalysisParams,

    /// The result of the Slater determinant representation analysis.
    #[builder(setter(skip), default = "None")]
    result: Option<SlaterDeterminantRepAnalysisResult<'a, G, T, SC>>,
}

impl<'a, G, T, SC> SlaterDeterminantRepAnalysisDriverBuilder<'a, G, T, SC>
where
    G: SymmetryGroupProperties + Clone,
    G::CharTab: SubspaceDecomposable<T>,
    T: ComplexFloat + Lapack,
    SC: StructureConstraint + fmt::Display,
    <T as ComplexFloat>::Real: From<f64> + fmt::LowerExp + fmt::Debug,
{
    fn validate(&self) -> Result<(), String> {
        let params = self
            .parameters
            .ok_or("No Slater determinant representation analysis parameters found.".to_string())?;

        let sym_res = self
            .symmetry_group
            .ok_or("No symmetry group information found.".to_string())?;

        let sao = self.sao.ok_or("No SAO matrix found.".to_string())?;

        if let Some(sao_h) = self.sao_h.flatten() {
            if sao_h.shape() != sao.shape() {
                return Err("Mismatched shapes between `sao` and `sao_h`.".to_string());
            }
        }

        match (
            self.sao_spatial_4c.flatten(),
            self.sao_spatial_4c_h.flatten(),
        ) {
            (Some(sao_spatial_4c), Some(sao_spatial_4c_h)) => {
                if sao_spatial_4c_h.shape() != sao_spatial_4c.shape() {
                    return Err(
                        "Mismatched shapes between `sao_spatial_4c` and `sao_spatial_4c_h`."
                            .to_string(),
                    );
                }
            }
            (None, Some(_)) => {
                return Err("`sao_spatial_4c_h` is provided without `sao_spatial_4c`.".to_string());
            }
            _ => {}
        }

        let det = self
            .determinant
            .ok_or("No Slater determinant found.".to_string())?;

        let sym = if params.use_magnetic_group.is_some() {
            sym_res
                .magnetic_symmetry
                .as_ref()
                .ok_or("Magnetic symmetry requested for representation analysis, but no magnetic symmetry found.")?
        } else {
            &sym_res.unitary_symmetry
        };

        if sym.is_infinite() && params.infinite_order_to_finite.is_none() {
            Err(format!(
                "Representation analysis cannot be performed using the entirety of the infinite group `{}`. \
                    Consider setting the parameter `infinite_order_to_finite` to restrict to a finite subgroup instead.",
                sym.group_name
                    .as_ref()
                    .expect("No symmetry group name found.")
            ))
        } else {
            let nfuncs_set = det
                .baos()
                .iter()
                .map(|bao| bao.n_funcs())
                .collect::<HashSet<_>>();
            let uniform_component_check = nfuncs_set.len() == 1
                && {
                    let nfuncs = nfuncs_set.iter().next().ok_or_else(|| "Unable to extract the uniform number of basis functions per explicit component.".to_string())?;
                    sao.nrows() == *nfuncs && sao.ncols() == *nfuncs
                };
            let total_component_check = {
                let nfuncs_tot = det.baos().iter().map(|bao| bao.n_funcs()).sum::<usize>();
                sao.nrows() == nfuncs_tot && sao.ncols() == nfuncs_tot
            };
            if !uniform_component_check && !total_component_check {
                Err("The dimensions of the SAO matrix do not match either the uniform number of AO basis functions per explicit component or the total number of AO basis functions across all explicit components.".to_string())
            } else {
                Ok(())
            }
        }
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~
// Struct implementations
// ~~~~~~~~~~~~~~~~~~~~~~

// Generic for all symmetry groups G, determinant numeric type T, and structure constraint SC
// ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

impl<'a, G, T, SC> SlaterDeterminantRepAnalysisDriver<'a, G, T, SC>
where
    G: SymmetryGroupProperties + Clone,
    G::CharTab: SubspaceDecomposable<T>,
    T: ComplexFloat + Lapack,
    SC: StructureConstraint + Clone + fmt::Display,
    <T as ComplexFloat>::Real: From<f64> + fmt::LowerExp + fmt::Debug,
{
    /// Returns a builder to construct a [`SlaterDeterminantRepAnalysisDriver`] structure.
    pub fn builder() -> SlaterDeterminantRepAnalysisDriverBuilder<'a, G, T, SC> {
        SlaterDeterminantRepAnalysisDriverBuilder::default()
    }

    /// Constructs the appropriate atomic-orbital overlap matrix based on the structure constraint
    /// of the determinant and the specified overlap matrix.
    fn construct_sao(&self) -> Result<(Array2<T>, Option<Array2<T>>), anyhow::Error> {
        let nbas_set = self
            .determinant
            .baos()
            .iter()
            .map(|bao| bao.n_funcs())
            .collect::<HashSet<_>>();
        let uniform_component = nbas_set.len() == 1;
        let ncomps = self
            .determinant
            .structure_constraint()
            .n_explicit_comps_per_coefficient_matrix();
        let provided_dim = self.sao.nrows();

        if uniform_component {
            let nbas = nbas_set.iter().next().ok_or_else(|| format_err!("Unable to extract the uniform number of basis functions per explicit component."))?;
            if provided_dim == *nbas {
                let sao = {
                    let mut sao_mut = Array2::zeros((ncomps * nbas, ncomps * nbas));
                    (0..ncomps).for_each(|icomp| {
                        let start = icomp * nbas;
                        let end = (icomp + 1) * nbas;
                        sao_mut
                            .slice_mut(s![start..end, start..end])
                            .assign(self.sao);
                    });
                    sao_mut
                };

                let sao_h = self.sao_h.map(|sao_h| {
                    let mut sao_h_mut = Array2::zeros((ncomps * nbas, ncomps * nbas));
                    (0..ncomps).for_each(|icomp| {
                        let start = icomp * nbas;
                        let end = (icomp + 1) * nbas;
                        sao_h_mut
                            .slice_mut(s![start..end, start..end])
                            .assign(sao_h);
                    });
                    sao_h_mut
                });

                Ok((sao, sao_h))
            } else {
                ensure!(provided_dim == nbas * ncomps);
                Ok((self.sao.clone(), self.sao_h.cloned()))
            }
        } else {
            let nbas_tot = self
                .determinant
                .baos()
                .iter()
                .map(|bao| bao.n_funcs())
                .sum::<usize>();
            ensure!(provided_dim == nbas_tot);
            Ok((self.sao.clone(), self.sao_h.cloned()))
        }
    }
}

// Specific for unitary-represented symmetry groups, but generic for determinant numeric type T and structure constraint SC
// ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

impl<'a, T, SC> SlaterDeterminantRepAnalysisDriver<'a, UnitaryRepresentedSymmetryGroup, T, SC>
where
    T: ComplexFloat + Lapack + Sync + Send,
    <T as ComplexFloat>::Real: From<f64> + fmt::LowerExp + fmt::Debug + Sync + Send,
    SC: StructureConstraint + fmt::Display,
    for<'b> Complex<f64>: Mul<&'b T, Output = Complex<f64>>,
{
    fn_construct_unitary_group!(
        /// Constructs the unitary-represented group (which itself can be unitary or magnetic) ready
        /// for Slater determinant representation analysis.
        construct_unitary_group
    );
}

// Specific for magnetic-represented symmetry groups, but generic for determinant numeric type T and structure constraint SC
// '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

impl<'a, T, SC> SlaterDeterminantRepAnalysisDriver<'a, MagneticRepresentedSymmetryGroup, T, SC>
where
    T: ComplexFloat + Lapack + Sync + Send,
    <T as ComplexFloat>::Real: From<f64> + Sync + Send + fmt::LowerExp + fmt::Debug,
    SC: StructureConstraint + fmt::Display,
    for<'b> Complex<f64>: Mul<&'b T, Output = Complex<f64>>,
{
    fn_construct_magnetic_group!(
        /// Constructs the magnetic-represented group (which itself can only be magnetic) ready for
        /// Slater determinant corepresentation analysis.
        construct_magnetic_group
    );
}

// Specific for unitary-represented and magnetic-represented symmetry groups and determinant numeric types f64 and C128
// ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

#[duplicate_item(
    duplicate!{
        [
            dtype_nested sctype_nested;
            [ f64 ] [ SpinConstraint ];
            [ Complex<f64> ] [ SpinConstraint ];
            [ Complex<f64> ] [ SpinOrbitCoupled ];
        ]
        [
            gtype_ [ UnitaryRepresentedSymmetryGroup ]
            dtype_ [ dtype_nested ]
            sctype_ [ sctype_nested ]
            doc_sub_ [ "Performs representation analysis using a unitary-represented group and stores the result." ]
            analyse_fn_ [ analyse_representation ]
            construct_group_ [ self.construct_unitary_group()? ]
            calc_projections_ [
                log_subtitle("Slater determinant projection decompositions");
                qsym2_output!("");
                qsym2_output!("  Projections are defined w.r.t. the following inner product:");
                qsym2_output!("    {}", det_orbit.origin().overlap_definition());
                qsym2_output!("");
                det_orbit
                    .projections_to_string(
                        &det_orbit.calc_projection_compositions()?,
                        params.integrality_threshold,
                    )
                    .log_output_display();
                qsym2_output!("");
            ]
            calc_mo_projections_ [
                let mo_projectionss = mo_orbitss.iter().map(|mo_orbits| {
                    mo_orbits
                        .iter()
                        .map(|mo_orbit| mo_orbit.calc_projection_compositions().ok())
                        .collect::<Vec<_>>()
                }).collect::<Vec<_>>();
                Some(mo_projectionss)
            ]
        ]
    }
    duplicate!{
        [
            dtype_nested sctype_nested;
            [ f64 ] [ SpinConstraint ];
            [ Complex<f64> ] [ SpinConstraint ];
            [ Complex<f64> ] [ SpinOrbitCoupled ];
        ]
        [
            gtype_ [ MagneticRepresentedSymmetryGroup ]
            dtype_ [ dtype_nested ]
            sctype_ [ sctype_nested ]
            doc_sub_ [ "Performs corepresentation analysis using a magnetic-represented group and stores the result." ]
            analyse_fn_ [ analyse_corepresentation ]
            construct_group_ [ self.construct_magnetic_group()? ]
            calc_projections_ [ ]
            calc_mo_projections_ [
                None::<Vec<Vec<Option<Vec<(MullikenIrcorepSymbol, Complex<f64>)>>>>>
            ]
        ]
    }
)]
impl<'a> SlaterDeterminantRepAnalysisDriver<'a, gtype_, dtype_, sctype_> {
    #[doc = doc_sub_]
    fn analyse_fn_(&mut self) -> Result<(), anyhow::Error> {
        let params = self.parameters;
        let (sao, sao_h) = self.construct_sao()?;
        let group = construct_group_;
        log_cc_transversal(&group);
        let _ = find_angular_function_representation(&group, self.angular_function_parameters);
        if group.is_double_group() {
            let _ = find_spinor_function_representation(&group, self.angular_function_parameters);
        }
        for (bao_i, bao) in self.determinant.baos().iter().enumerate() {
            log_bao(bao, Some(bao_i));
        }

        // Determinant and orbital symmetries
        let (
            det_symmetry,
            mo_symmetries,
            mo_symmetry_projections,
            mo_mirror_parities,
            mo_symmetries_thresholds,
        ) = if params.analyse_mo_symmetries {
            let mos = self.determinant.to_orbitals();
            let (mut det_orbit, mut mo_orbitss) = generate_det_mo_orbits(
                self.determinant,
                &mos,
                &group,
                &sao,
                sao_h.as_ref(),
                params.integrality_threshold,
                params.linear_independence_threshold,
                params.symmetry_transformation_kind.clone(),
                params.eigenvalue_comparison_mode.clone(),
                params.use_cayley_table,
            )?;
            det_orbit.calc_xmat(false)?;
            if params.write_overlap_eigenvalues {
                if let Some(smat_eigvals) = det_orbit.smat_eigvals.as_ref() {
                    log_overlap_eigenvalues(
                        "Determinant orbit overlap eigenvalues",
                        smat_eigvals,
                        params.linear_independence_threshold,
                        &params.eigenvalue_comparison_mode,
                    );
                    qsym2_output!("");
                }
            }

            let det_symmetry = det_orbit.analyse_rep().map_err(|err| err.to_string());

            {
                calc_projections_
            }

            let mo_symmetries = mo_orbitss
                .iter_mut()
                .map(|mo_orbits| {
                    mo_orbits
                        .par_iter_mut()
                        .map(|mo_orbit| {
                            mo_orbit.calc_xmat(false).ok()?;
                            mo_orbit.analyse_rep().ok()
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();

            let mo_symmetry_projections_opt = if params.analyse_mo_symmetry_projections {
                calc_mo_projections_
            } else {
                None
            };

            let mo_mirror_parities_opt = if params.analyse_mo_mirror_parities {
                Some(
                    mo_symmetries
                        .iter()
                        .map(|spin_mo_symmetries| {
                            spin_mo_symmetries
                                .iter()
                                .map(|mo_sym_opt| {
                                    mo_sym_opt.as_ref().map(|mo_sym| {
                                        deduce_mirror_parities(det_orbit.group(), mo_sym)
                                    })
                                })
                                .collect::<Vec<_>>()
                        })
                        .collect::<Vec<_>>(),
                )
            } else {
                None
            };

            let mo_symmetries_thresholds = mo_orbitss
                .iter_mut()
                .map(|mo_orbits| {
                    mo_orbits
                        .par_iter_mut()
                        .map(|mo_orbit| {
                            mo_orbit
                                .smat_eigvals
                                .as_ref()
                                .map(|eigvals| {
                                    let mut eigvals_vec = eigvals.iter().collect::<Vec<_>>();
                                    match mo_orbit.eigenvalue_comparison_mode {
                                        EigenvalueComparisonMode::Modulus => {
                                            eigvals_vec.sort_by(|a, b| {
                                                a.abs().partial_cmp(&b.abs()).expect("Unable to compare two eigenvalues based on their moduli.")
                                            });
                                        }
                                        EigenvalueComparisonMode::Real => {
                                            eigvals_vec.sort_by(|a, b| {
                                                a.re().partial_cmp(&b.re()).expect("Unable to compare two eigenvalues based on their real parts.")
                                            });
                                        }
                                    }
                                    let eigval_above = match mo_orbit.eigenvalue_comparison_mode {
                                        EigenvalueComparisonMode::Modulus => eigvals_vec
                                            .iter()
                                            .find(|val| {
                                                val.abs() >= mo_orbit.linear_independence_threshold
                                            })
                                            .copied()
                                            .copied(),
                                        EigenvalueComparisonMode::Real => eigvals_vec
                                            .iter()
                                            .find(|val| {
                                                val.re() >= mo_orbit.linear_independence_threshold
                                            })
                                            .copied()
                                            .copied(),
                                    };
                                    eigvals_vec.reverse();
                                    let eigval_below = match mo_orbit.eigenvalue_comparison_mode {
                                        EigenvalueComparisonMode::Modulus => eigvals_vec
                                            .iter()
                                            .find(|val| {
                                                val.abs() < mo_orbit.linear_independence_threshold
                                            })
                                            .copied()
                                            .copied(),
                                        EigenvalueComparisonMode::Real => eigvals_vec
                                            .iter()
                                            .find(|val| {
                                                val.re() < mo_orbit.linear_independence_threshold
                                            })
                                            .copied()
                                            .copied(),
                                    };
                                    (eigval_above, eigval_below)
                                })
                                .unwrap_or((None, None))
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            (
                det_symmetry,
                Some(mo_symmetries),
                mo_symmetry_projections_opt,
                mo_mirror_parities_opt,
                Some(mo_symmetries_thresholds),
            ) // det_symmetry, mo_symmetries, mo_symmetry_projections, mo_mirror_parities, mo_symmetries_thresholds
        } else {
            let mut det_orbit = SlaterDeterminantSymmetryOrbit::builder()
                .group(&group)
                .origin(self.determinant)
                .integrality_threshold(params.integrality_threshold)
                .linear_independence_threshold(params.linear_independence_threshold)
                .symmetry_transformation_kind(params.symmetry_transformation_kind.clone())
                .eigenvalue_comparison_mode(params.eigenvalue_comparison_mode.clone())
                .build()?;
            let det_symmetry = det_orbit
                .calc_smat(Some(&sao), sao_h.as_ref(), params.use_cayley_table)
                .and_then(|det_orb| det_orb.normalise_smat())
                .map_err(|err| err.to_string())
                .and_then(|det_orb| {
                    det_orb.calc_xmat(false).map_err(|err| err.to_string())?;
                    if params.write_overlap_eigenvalues {
                        if let Some(smat_eigvals) = det_orb.smat_eigvals.as_ref() {
                            log_overlap_eigenvalues(
                                "Determinant orbit overlap eigenvalues",
                                smat_eigvals,
                                params.linear_independence_threshold,
                                &params.eigenvalue_comparison_mode,
                            );
                            qsym2_output!("");
                        }
                    }
                    det_orb.analyse_rep().map_err(|err| err.to_string())
                });

            {
                calc_projections_
            }

            (det_symmetry, None, None, None, None) // det_symmetry, mo_symmetries, mo_symmetry_projections, mo_mirror_parities, mo_symmetries_thresholds
        };

        // Density and orbital density symmetries
        let (den_symmetries, mo_den_symmetries) = if params.analyse_density_symmetries {
            let den_syms = self.determinant.to_densities().map(|densities| {
                let mut spin_den_syms = densities
                    .iter()
                    .enumerate()
                    .map(|(ispin, den)| {
                        let den_sym_res = || {
                            let mut den_orbit = DensitySymmetryOrbit::builder()
                                .group(&group)
                                .origin(den)
                                .integrality_threshold(params.integrality_threshold)
                                .linear_independence_threshold(params.linear_independence_threshold)
                                .symmetry_transformation_kind(
                                    params.symmetry_transformation_kind.clone(),
                                )
                                .eigenvalue_comparison_mode(
                                    params.eigenvalue_comparison_mode.clone(),
                                )
                                .build()?;
                            den_orbit
                                .calc_smat(
                                    self.sao_spatial_4c,
                                    self.sao_spatial_4c_h,
                                    params.use_cayley_table,
                                )?
                                .normalise_smat()?
                                .calc_xmat(false)?;
                            den_orbit.analyse_rep().map_err(|err| format_err!(err))
                        };
                        (
                            format!("Spin-{ispin} density"),
                            den_sym_res().map_err(|err| err.to_string()),
                        )
                    })
                    .collect::<Vec<_>>();
                let mut extra_den_syms = densities
                    .calc_extra_densities()
                    .expect("Unable to calculate extra densities.")
                    .iter()
                    .map(|(label, den)| {
                        let den_sym_res = || {
                            let mut den_orbit = DensitySymmetryOrbit::builder()
                                .group(&group)
                                .origin(den)
                                .integrality_threshold(params.integrality_threshold)
                                .linear_independence_threshold(params.linear_independence_threshold)
                                .symmetry_transformation_kind(
                                    params.symmetry_transformation_kind.clone(),
                                )
                                .eigenvalue_comparison_mode(
                                    params.eigenvalue_comparison_mode.clone(),
                                )
                                .build()?;
                            den_orbit
                                .calc_smat(
                                    self.sao_spatial_4c,
                                    self.sao_spatial_4c_h,
                                    params.use_cayley_table,
                                )?
                                .calc_xmat(false)?;
                            den_orbit.analyse_rep().map_err(|err| format_err!(err))
                        };
                        (label.clone(), den_sym_res().map_err(|err| err.to_string()))
                    })
                    .collect_vec();
                // let mut extra_den_syms = match self.determinant.structure_constraint() {
                //     SpinConstraint::Restricted(_) => {
                //         vec![("Total density".to_string(), spin_den_syms[0].1.clone())]
                //     }
                //     SpinConstraint::Unrestricted(nspins, _)
                //     | SpinConstraint::Generalised(nspins, _) => {
                //         let total_den_sym_res = || {
                //             let nspatial = self.determinant.bao().n_funcs();
                //             let zero_den = Density::<dtype_>::builder()
                //                 .density_matrix(Array2::<dtype_>::zeros((nspatial, nspatial)))
                //                 .bao(self.determinant.bao())
                //                 .mol(self.determinant.mol())
                //                 .complex_symmetric(self.determinant.complex_symmetric())
                //                 .threshold(self.determinant.threshold())
                //                 .build()?;
                //             let total_den =
                //                 densities.iter().fold(zero_den, |acc, denmat| acc + denmat);
                //             let mut total_den_orbit = DensitySymmetryOrbit::builder()
                //                 .group(&group)
                //                 .origin(&total_den)
                //                 .integrality_threshold(params.integrality_threshold)
                //                 .linear_independence_threshold(params.linear_independence_threshold)
                //                 .symmetry_transformation_kind(
                //                     params.symmetry_transformation_kind.clone(),
                //                 )
                //                 .eigenvalue_comparison_mode(
                //                     params.eigenvalue_comparison_mode.clone(),
                //                 )
                //                 .build()?;
                //             total_den_orbit
                //                 .calc_smat(
                //                     self.sao_spatial_4c,
                //                     self.sao_spatial_4c_h,
                //                     params.use_cayley_table,
                //                 )?
                //                 .calc_xmat(false)?;
                //             total_den_orbit
                //                 .analyse_rep()
                //                 .map_err(|err| format_err!(err))
                //         };
                //         let mut extra_syms = vec![(
                //             "Total density".to_string(),
                //             total_den_sym_res().map_err(|err| err.to_string()),
                //         )];
                //         extra_syms.extend((0..usize::from(*nspins)).combinations(2).map(
                //             |indices| {
                //                 let i = indices[0];
                //                 let j = indices[1];
                //                 let den_ij = &densities[i] - &densities[j];
                //                 let den_ij_sym_res = || {
                //                     let mut den_ij_orbit = DensitySymmetryOrbit::builder()
                //                         .group(&group)
                //                         .origin(&den_ij)
                //                         .integrality_threshold(params.integrality_threshold)
                //                         .linear_independence_threshold(
                //                             params.linear_independence_threshold,
                //                         )
                //                         .symmetry_transformation_kind(
                //                             params.symmetry_transformation_kind.clone(),
                //                         )
                //                         .eigenvalue_comparison_mode(
                //                             params.eigenvalue_comparison_mode.clone(),
                //                         )
                //                         .build()?;
                //                     den_ij_orbit
                //                         .calc_smat(
                //                             self.sao_spatial_4c,
                //                             self.sao_spatial_4c_h,
                //                             params.use_cayley_table,
                //                         )?
                //                         .calc_xmat(false)?;
                //                     den_ij_orbit.analyse_rep().map_err(|err| format_err!(err))
                //                 };
                //                 (
                //                     format!("Spin-polarised density {i} - {j}"),
                //                     den_ij_sym_res().map_err(|err| err.to_string()),
                //                 )
                //             },
                //         ));
                //         extra_syms
                //     }
                // };
                spin_den_syms.append(&mut extra_den_syms);
                spin_den_syms
            });

            let mo_den_syms = if params.analyse_mo_symmetries {
                let mo_den_symmetries = self
                    .determinant
                    .to_orbitals()
                    .iter()
                    .map(|mos| {
                        mos.par_iter()
                            .map(|mo| {
                                let mo_den = mo.to_total_density().ok()?;
                                let mut mo_den_orbit = DensitySymmetryOrbit::builder()
                                    .group(&group)
                                    .origin(&mo_den)
                                    .integrality_threshold(params.integrality_threshold)
                                    .linear_independence_threshold(
                                        params.linear_independence_threshold,
                                    )
                                    .symmetry_transformation_kind(
                                        params.symmetry_transformation_kind.clone(),
                                    )
                                    .eigenvalue_comparison_mode(
                                        params.eigenvalue_comparison_mode.clone(),
                                    )
                                    .build()
                                    .ok()?;
                                log::debug!("Computing overlap matrix for an MO density orbit...");
                                mo_den_orbit
                                    .calc_smat(
                                        self.sao_spatial_4c,
                                        self.sao_spatial_4c_h,
                                        params.use_cayley_table,
                                    )
                                    .ok()?
                                    .normalise_smat()
                                    .ok()?
                                    .calc_xmat(false)
                                    .ok()?;
                                log::debug!(
                                    "Computing overlap matrix for an MO density orbit... Done."
                                );
                                mo_den_orbit.analyse_rep().ok()
                            })
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>();
                Some(mo_den_symmetries)
            } else {
                None
            };

            (den_syms.ok(), mo_den_syms)
        } else {
            (None, None)
        };

        let result = SlaterDeterminantRepAnalysisResult::builder()
            .parameters(params)
            .determinant(self.determinant)
            .group(group)
            .determinant_symmetry(det_symmetry)
            .determinant_density_symmetries(den_symmetries)
            .mo_symmetries(mo_symmetries)
            .mo_symmetry_projections(mo_symmetry_projections)
            .mo_mirror_parities(mo_mirror_parities)
            .mo_symmetries_thresholds(mo_symmetries_thresholds)
            .mo_density_symmetries(mo_den_symmetries)
            .build()?;
        self.result = Some(result);

        Ok(())
    }
}

// ~~~~~~~~~~~~~~~~~~~~~
// Trait implementations
// ~~~~~~~~~~~~~~~~~~~~~

// Generic for all symmetry groups G and determinant numeric type T
// ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

impl<'a, G, T, SC> fmt::Display for SlaterDeterminantRepAnalysisDriver<'a, G, T, SC>
where
    G: SymmetryGroupProperties + Clone,
    G::CharTab: SubspaceDecomposable<T>,
    T: ComplexFloat + Lapack,
    SC: StructureConstraint + fmt::Display,
    <T as ComplexFloat>::Real: From<f64> + fmt::LowerExp + fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write_title(f, "Slater Determinant Symmetry Analysis")?;
        writeln!(f)?;
        writeln!(f, "{}", self.parameters)?;
        Ok(())
    }
}

impl<'a, G, T, SC> fmt::Debug for SlaterDeterminantRepAnalysisDriver<'a, G, T, SC>
where
    G: SymmetryGroupProperties + Clone,
    G::CharTab: SubspaceDecomposable<T>,
    T: ComplexFloat + Lapack,
    SC: StructureConstraint + fmt::Display,
    <T as ComplexFloat>::Real: From<f64> + fmt::LowerExp + fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{self}")
    }
}

// Specific for unitary/magnetic-represented groups and determinant numeric type f64/Complex<f64>
// ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

#[duplicate_item(
    duplicate!{
        [
            dtype_nested sctype_nested;
            [ f64 ] [ SpinConstraint ];
            [ Complex<f64> ] [ SpinConstraint ];
            [ Complex<f64> ] [ SpinOrbitCoupled ];
        ]
        [
            gtype_ [ UnitaryRepresentedSymmetryGroup ]
            dtype_ [ dtype_nested ]
            sctype_ [ sctype_nested ]
            analyse_fn_ [ analyse_representation ]
        ]
    }
    duplicate!{
        [
            dtype_nested sctype_nested;
            [ f64 ] [ SpinConstraint ];
            [ Complex<f64> ] [ SpinConstraint ];
            [ Complex<f64> ] [ SpinOrbitCoupled ];
        ]
        [
            gtype_ [ MagneticRepresentedSymmetryGroup ]
            dtype_ [ dtype_nested ]
            sctype_ [ sctype_nested ]
            analyse_fn_ [ analyse_corepresentation ]
        ]
    }
)]
impl<'a> QSym2Driver for SlaterDeterminantRepAnalysisDriver<'a, gtype_, dtype_, sctype_> {
    type Params = SlaterDeterminantRepAnalysisParams<f64>;

    type Outcome = SlaterDeterminantRepAnalysisResult<'a, gtype_, dtype_, sctype_>;

    fn result(&self) -> Result<&Self::Outcome, anyhow::Error> {
        self.result.as_ref().ok_or_else(|| {
            format_err!("No Slater determinant representation analysis results found.")
        })
    }

    fn run(&mut self) -> Result<(), anyhow::Error> {
        self.log_output_display();
        self.analyse_fn_()?;
        self.result()?.log_output_display();
        Ok(())
    }
}
