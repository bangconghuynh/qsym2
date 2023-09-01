use std::fmt;
use std::ops::Mul;

use anyhow::{self, bail, format_err};
use derive_builder::Builder;
use duplicate::duplicate_item;
use itertools::Itertools;
use ndarray::{s, Array2, Array4};
use ndarray_linalg::types::Lapack;
use num_complex::{Complex, ComplexFloat};
use num_traits::Float;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::analysis::{log_overlap_eigenvalues, EigenvalueComparisonMode, RepAnalysis};
use crate::angmom::spinor_rotation_3d::SpinConstraint;
use crate::chartab::chartab_group::CharacterProperties;
use crate::chartab::SubspaceDecomposable;
use crate::drivers::representation_analysis::angular_function::{
    find_angular_function_representation, AngularFunctionRepAnalysisParams,
};
use crate::drivers::representation_analysis::{
    log_bao, log_cc_transversal, CharacterTableDisplay, MagneticSymmetryAnalysisKind,
};
use crate::drivers::symmetry_group_detection::SymmetryGroupDetectionResult;
use crate::drivers::QSym2Driver;
use crate::group::{GroupProperties, MagneticRepresentedGroup, UnitaryRepresentedGroup};
use crate::io::format::{
    log_subtitle, nice_bool, qsym2_output, write_subtitle, write_title, QSym2Output,
};
use crate::symmetry::symmetry_group::{
    MagneticRepresentedSymmetryGroup, SymmetryGroupProperties, UnitaryRepresentedSymmetryGroup,
};
use crate::symmetry::symmetry_transformation::SymmetryTransformationKind;
use crate::target::density::density_analysis::DensitySymmetryOrbit;
use crate::target::density::Density;
use crate::target::determinant::determinant_analysis::SlaterDeterminantSymmetryOrbit;
use crate::target::determinant::SlaterDeterminant;
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

/// A structure containing control parameters for Slater determinant representation analysis.
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

/// A structure to contain Slater determinant representation analysis results.
#[derive(Clone, Builder)]
pub struct SlaterDeterminantRepAnalysisResult<'a, G, T>
where
    G: SymmetryGroupProperties + Clone,
    G::CharTab: SubspaceDecomposable<T>,
    T: ComplexFloat + Lapack,
    <T as ComplexFloat>::Real: From<f64> + fmt::LowerExp + fmt::Debug,
{
    /// The control parameters used to obtain this set of Slater determinant representation
    /// analysis results.
    parameters: &'a SlaterDeterminantRepAnalysisParams<<T as ComplexFloat>::Real>,

    /// The Slater determinant being analysed.
    determinant: &'a SlaterDeterminant<'a, T>,

    /// The group used for the representation analysis.
    group: G,

    /// The deduced overall symmetry of the determinant.
    determinant_symmetry: Result<<G::CharTab as SubspaceDecomposable<T>>::Decomposition, String>,

    /// The deduced symmetries of the molecular orbitals constituting the determinant, if required.
    mo_symmetries: Option<Vec<Vec<Option<<G::CharTab as SubspaceDecomposable<T>>::Decomposition>>>>,

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

impl<'a, G, T> SlaterDeterminantRepAnalysisResult<'a, G, T>
where
    G: SymmetryGroupProperties + Clone,
    G::CharTab: SubspaceDecomposable<T>,
    T: ComplexFloat + Lapack,
    <T as ComplexFloat>::Real: From<f64> + fmt::LowerExp + fmt::Debug,
{
    /// Returns a builder to construct a new [`SlaterDeterminantRepAnalysisResultBuilder`]
    /// structure.
    fn builder() -> SlaterDeterminantRepAnalysisResultBuilder<'a, G, T> {
        SlaterDeterminantRepAnalysisResultBuilder::default()
    }
}

impl<'a, G, T> fmt::Display for SlaterDeterminantRepAnalysisResult<'a, G, T>
where
    G: SymmetryGroupProperties + Clone,
    G::CharTab: SubspaceDecomposable<T>,
    T: ComplexFloat + Lapack,
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
            let mo_index_length = mo_symmetries
                .iter()
                .map(|spin_mo_symmetries| spin_mo_symmetries.len())
                .max()
                .and_then(|max_mo_length| usize::try_from(max_mo_length.ilog10() + 2).ok())
                .unwrap_or(4);

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
                    .unwrap_or(13)
                    .max(13)
            });
            writeln!(f, "> Molecular orbital results")?;
            writeln!(
                f,
                "  Spin constraint: {}",
                self.determinant
                    .spin_constraint()
                    .to_string()
                    .to_lowercase()
            )?;

            if let Some(mo_density_length) = mo_density_length_opt {
                // Includes MO density symmetries
                writeln!(
                    f,
                    "{}",
                    "┈".repeat(
                        25 + mo_index_length
                            + mo_energy_length
                            + mo_symmetry_length
                            + mo_eig_above_length
                            + mo_eig_below_length
                            + mo_density_length
                    )
                )?;
                writeln!(
                    f,
                    "{:>5}  {:>mo_index_length$}  {:<5}  {:<mo_energy_length$}  {:<mo_symmetry_length$}  {:<mo_eig_above_length$}  {:<mo_eig_below_length$}  Den. symmetry",
                    "Spin", "MO", "Occ.", "Energy", "Symmetry", "Eig. above", "Eig. below"
                )?;
                writeln!(
                    f,
                    "{}",
                    "┈".repeat(
                        25 + mo_index_length
                            + mo_energy_length
                            + mo_symmetry_length
                            + mo_eig_above_length
                            + mo_eig_below_length
                            + mo_density_length
                    )
                )?;
                for (spini, (spin_mo_symmetries, spin_mo_den_symmetries)) in mo_symmetries
                    .iter()
                    .zip(
                        mo_den_symss_str_opt
                            .expect("No MO density symmetries found.")
                            .iter(),
                    )
                    .enumerate()
                {
                    writeln!(f, " Spin {spini}")?;
                    for (moi, (mo_sym, mo_den_sym_str)) in spin_mo_symmetries
                        .iter()
                        .zip(spin_mo_den_symmetries.iter())
                        .enumerate()
                    {
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
                        writeln!(
                            f,
                            "{spini:>5}  \
                            {moi:>mo_index_length$}  \
                            {occ_str:<5}  \
                            {mo_energy_str:<mo_energy_length$}  \
                            {mo_sym_str:<mo_symmetry_length$}  \
                            {eig_above_str:<mo_eig_above_length$}  \
                            {eig_below_str:<mo_eig_below_length$}  \
                            {mo_den_sym_str}"
                        )?;
                    }
                }
                writeln!(
                    f,
                    "{}",
                    "┈".repeat(
                        25 + mo_index_length
                            + mo_energy_length
                            + mo_symmetry_length
                            + mo_eig_above_length
                            + mo_eig_below_length
                            + mo_density_length
                    )
                )?;
            } else {
                // No MO density symmetries
                writeln!(
                    f,
                    "{}",
                    "┈".repeat(
                        23 + mo_index_length
                            + mo_energy_length
                            + mo_symmetry_length
                            + mo_eig_above_length
                            + mo_eig_below_length
                    )
                )?;
                writeln!(
                    f,
                    "{:>5}  {:>mo_index_length$}  {:<5}  {:<mo_energy_length$}  {:<mo_symmetry_length$}  {:<mo_eig_above_length$}  Eig. below",
                    "Spin", "MO", "Occ.", "Energy", "Symmetry", "Eig. above"
                )?;
                writeln!(
                    f,
                    "{}",
                    "┈".repeat(
                        23 + mo_index_length
                            + mo_energy_length
                            + mo_symmetry_length
                            + mo_eig_above_length
                            + mo_eig_below_length
                    )
                )?;
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
                        writeln!(
                            f,
                            "{spini:>5}  \
                            {moi:>mo_index_length$}  \
                            {occ_str:<5}  \
                            {mo_energy_str:<mo_energy_length$}  \
                            {mo_sym_str:<mo_symmetry_length$}  \
                            {eig_above_str:<mo_eig_above_length$}  \
                            {eig_below_str}"
                        )?;
                    }
                }
                writeln!(
                    f,
                    "{}",
                    "┈".repeat(
                        23 + mo_index_length
                            + mo_energy_length
                            + mo_symmetry_length
                            + mo_eig_above_length
                            + mo_eig_below_length
                    )
                )?;
            }
        }

        Ok(())
    }
}

impl<'a, G, T> fmt::Debug for SlaterDeterminantRepAnalysisResult<'a, G, T>
where
    G: SymmetryGroupProperties + Clone,
    G::CharTab: SubspaceDecomposable<T>,
    T: ComplexFloat + Lapack,
    <T as ComplexFloat>::Real: From<f64> + fmt::LowerExp + fmt::Debug + fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{self}")
    }
}

impl<'a, G, T> SlaterDeterminantRepAnalysisResult<'a, G, T>
where
    G: SymmetryGroupProperties + Clone,
    G::CharTab: SubspaceDecomposable<T>,
    T: ComplexFloat + Lapack,
    <T as ComplexFloat>::Real: From<f64> + fmt::LowerExp + fmt::Debug + fmt::Display,
{
    /// Returns the determinant symmetry obtained from the analysis result.
    pub fn determinant_symmetry(
        &self,
    ) -> &Result<<G::CharTab as SubspaceDecomposable<T>>::Decomposition, String> {
        &self.determinant_symmetry
    }
}

// ------
// Driver
// ------

// ~~~~~~~~~~~~~~~~~
// Struct definition
// ~~~~~~~~~~~~~~~~~

/// A driver structure for performing representation analysis on Slater determinants.
#[derive(Clone, Builder)]
#[builder(build_fn(validate = "Self::validate"))]
pub struct SlaterDeterminantRepAnalysisDriver<'a, G, T>
where
    G: SymmetryGroupProperties + Clone,
    G::CharTab: SubspaceDecomposable<T>,
    T: ComplexFloat + Lapack,
    <T as ComplexFloat>::Real: From<f64> + fmt::LowerExp + fmt::Debug,
{
    /// The control parameters for Slater determinant representation analysis.
    parameters: &'a SlaterDeterminantRepAnalysisParams<<T as ComplexFloat>::Real>,

    /// The Slater determinant to be analysed.
    determinant: &'a SlaterDeterminant<'a, T>,

    /// The result from symmetry-group detection on the underlying molecular structure of the
    /// Slater determinant.
    symmetry_group: &'a SymmetryGroupDetectionResult,

    /// The atomic-orbital spatial overlap matrix of the underlying basis set used to describe the
    /// determinant.
    sao_spatial: &'a Array2<T>,

    /// The atomic-orbital four-centre spatial overlap matrix of the underlying basis set used to
    /// describe the determinant. This is only required for density symmetry analysis.
    #[builder(default = "None")]
    sao_spatial_4c: Option<&'a Array4<T>>,

    /// The control parameters for symmetry analysis of angular functions.
    angular_function_parameters: &'a AngularFunctionRepAnalysisParams,

    /// The result of the Slater determinant representation analysis.
    #[builder(setter(skip), default = "None")]
    result: Option<SlaterDeterminantRepAnalysisResult<'a, G, T>>,
}

impl<'a, G, T> SlaterDeterminantRepAnalysisDriverBuilder<'a, G, T>
where
    G: SymmetryGroupProperties + Clone,
    G::CharTab: SubspaceDecomposable<T>,
    T: ComplexFloat + Lapack,
    <T as ComplexFloat>::Real: From<f64> + fmt::LowerExp + fmt::Debug,
{
    fn validate(&self) -> Result<(), String> {
        let params = self
            .parameters
            .ok_or("No Slater determinant representation analysis parameters found.".to_string())?;

        let sym_res = self
            .symmetry_group
            .ok_or("No symmetry group information found.".to_string())?;

        let sao_spatial = self
            .sao_spatial
            .ok_or("No spatial SAO matrix found.".to_string())?;

        let det = self
            .determinant
            .ok_or("No Slater determinant found.".to_string())?;

        let sym = if params.use_magnetic_group.is_some() {
            sym_res
                .magnetic_symmetry
                .as_ref()
                .ok_or("Magnetic symmetry requested as symmetrisation target, but no magnetic symmetry found.")?
        } else {
            &sym_res.unitary_symmetry
        };

        if sym.is_infinite() && params.infinite_order_to_finite.is_none() {
            Err(
                format!(
                    "Representation analysis cannot be performed using the entirety of the infinite group `{}`. \
                    Consider setting the parameter `infinite_order_to_finite` to restrict to a finite subgroup instead.",
                    sym.group_name.as_ref().expect("No target group name found.")
                )
            )
        } else if det.bao().n_funcs() != sao_spatial.nrows()
            || det.bao().n_funcs() != sao_spatial.ncols()
        {
            Err("The dimensions of the spatial SAO matrix do not match the number of spatial AO basis functions.".to_string())
        } else {
            Ok(())
        }
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~
// Struct implementations
// ~~~~~~~~~~~~~~~~~~~~~~

// Generic for all symmetry groups G and determinant numeric type T
// ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

impl<'a, G, T> SlaterDeterminantRepAnalysisDriver<'a, G, T>
where
    G: SymmetryGroupProperties + Clone,
    G::CharTab: SubspaceDecomposable<T>,
    T: ComplexFloat + Lapack,
    <T as ComplexFloat>::Real: From<f64> + fmt::LowerExp + fmt::Debug,
{
    /// Returns a builder to construct a [`SlaterDeterminantRepAnalysisDriver`] structure.
    pub fn builder() -> SlaterDeterminantRepAnalysisDriverBuilder<'a, G, T> {
        SlaterDeterminantRepAnalysisDriverBuilder::default()
    }

    /// Constructs the appropriate atomic-orbital overlap matrix based on the spin constraint of
    /// the determinant.
    fn construct_sao(&self) -> Result<Array2<T>, anyhow::Error> {
        let sao = match self.determinant.spin_constraint() {
            SpinConstraint::Restricted(_) | SpinConstraint::Unrestricted(_, _) => {
                self.sao_spatial.clone()
            }
            SpinConstraint::Generalised(nspins, _) => {
                let nspins_usize = usize::from(*nspins);
                let nspatial = self.sao_spatial.nrows();
                let mut sao_g = Array2::zeros((nspins_usize * nspatial, nspins_usize * nspatial));
                (0..nspins_usize).for_each(|ispin| {
                    let start = ispin * nspatial;
                    let end = (ispin + 1) * nspatial;
                    sao_g
                        .slice_mut(s![start..end, start..end])
                        .assign(self.sao_spatial);
                });
                sao_g
            }
        };

        Ok(sao)
    }
}

// Specific for unitary-represented symmetry groups, but generic for determinant numeric type T
// ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

impl<'a, T> SlaterDeterminantRepAnalysisDriver<'a, UnitaryRepresentedSymmetryGroup, T>
where
    T: ComplexFloat + Lapack + Sync + Send,
    <T as ComplexFloat>::Real: From<f64> + fmt::LowerExp + fmt::Debug + Sync + Send,
    for<'b> Complex<f64>: Mul<&'b T, Output = Complex<f64>>,
{
    /// Constructs the unitary-represented group (which itself can be unitary or magnetic) ready
    /// for representation analysis.
    fn construct_unitary_group(&self) -> Result<UnitaryRepresentedSymmetryGroup, anyhow::Error> {
        let params = self.parameters;
        let sym = match params.use_magnetic_group {
            Some(MagneticSymmetryAnalysisKind::Representation) => self.symmetry_group
                .magnetic_symmetry
                .as_ref()
                .ok_or_else(|| {
                    format_err!(
                        "Magnetic symmetry requested for analysis, but no magnetic symmetry found."
                    )
                })?,
            Some(MagneticSymmetryAnalysisKind::Corepresentation) => bail!("Magnetic corepresentations requested, but unitary-represented group is being constructed."),
            None => &self.symmetry_group.unitary_symmetry
        };
        let group = if params.use_double_group {
            UnitaryRepresentedGroup::from_molecular_symmetry(sym, params.infinite_order_to_finite)?
                .to_double_group()?
        } else {
            UnitaryRepresentedGroup::from_molecular_symmetry(sym, params.infinite_order_to_finite)?
        };

        qsym2_output!(
            "Unitary-represented group for representation analysis: {}",
            group.name()
        );
        qsym2_output!("");
        if let Some(chartab_display) = params.write_character_table.as_ref() {
            log_subtitle("Character table of irreducible representations");
            qsym2_output!("");
            match chartab_display {
                CharacterTableDisplay::Symbolic => {
                    group.character_table().log_output_debug();
                    "Any `En` in a character value denotes the first primitive n-th root of unity:\n  \
                    En = exp(2πi/n)".log_output_display();
                }
                CharacterTableDisplay::Numerical => group.character_table().log_output_display(),
            }
            qsym2_output!("");
            "Note 1: `FS` contains the classification of the irreps using the Frobenius--Schur indicator:\n  \
            `r` = real: the irrep and its complex-conjugate partner are real and identical,\n  \
            `c` = complex: the irrep and its complex-conjugate partner are complex and inequivalent,\n  \
            `q` = quaternion: the irrep and its complex-conjugate partner are complex and equivalent.\n\n\
            Note 2: The conjugacy classes are sorted according to the following order:\n  \
            E -> C_n (n descending) -> C2 -> i -> S_n (n decending) -> σ\n  \
            Within each order and power, elements with axes close to Cartesian axes are put first.\n  \
            Within each equi-inclination from Cartesian axes, z-inclined axes are put first, then y, then x.\n\n\
            Note 3: The Mulliken labels generated for the irreps in the table above are internally consistent.\n  \
            However, certain labels might differ from those tabulated elsewhere using other conventions.\n  \
            If need be, please check with other literature to ensure external consistency.".log_output_display();
            qsym2_output!("");
        }
        Ok(group)
    }
}

// Specific for magnetic-represented symmetry groups, but generic for determinant numeric type T
// ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

impl<'a, T> SlaterDeterminantRepAnalysisDriver<'a, MagneticRepresentedSymmetryGroup, T>
where
    T: ComplexFloat + Lapack + Sync + Send,
    <T as ComplexFloat>::Real: From<f64> + Sync + Send + fmt::LowerExp + fmt::Debug,
    for<'b> Complex<f64>: Mul<&'b T, Output = Complex<f64>>,
{
    /// Constructs the magnetic-represented group (which itself can only be magnetic) ready for
    /// corepresentation analysis.
    fn construct_magnetic_group(&self) -> Result<MagneticRepresentedSymmetryGroup, anyhow::Error> {
        let params = self.parameters;
        let sym = match params.use_magnetic_group {
            Some(MagneticSymmetryAnalysisKind::Corepresentation) => self.symmetry_group
                .magnetic_symmetry
                .as_ref()
                .ok_or_else(|| {
                    format_err!(
                        "Magnetic symmetry requested for analysis, but no magnetic symmetry found."
                    )
                })?,
            Some(MagneticSymmetryAnalysisKind::Representation) => bail!("Unitary representations requested, but magnetic-represented group is being constructed."),
            None => &self.symmetry_group.unitary_symmetry
        };
        let group = if params.use_double_group {
            MagneticRepresentedGroup::from_molecular_symmetry(sym, params.infinite_order_to_finite)?
                .to_double_group()?
        } else {
            MagneticRepresentedGroup::from_molecular_symmetry(sym, params.infinite_order_to_finite)?
        };

        qsym2_output!(
            "Magnetic-represented group for corepresentation analysis: {}",
            group.name()
        );
        qsym2_output!("");

        if let Some(chartab_display) = params.write_character_table.as_ref() {
            log_subtitle("Character table of irreducible corepresentations");
            qsym2_output!("");
            match chartab_display {
                CharacterTableDisplay::Symbolic => {
                    group.character_table().log_output_debug();
                    "Any `En` in a character value denotes the first primitive n-th root of unity:\n  \
                    En = exp(2πi/n)".log_output_display();
                }
                CharacterTableDisplay::Numerical => group.character_table().log_output_display(),
            }
            qsym2_output!("");
            "Note 1: The ircorep notation `D[Δ]` means that this ircorep is induced by the representation Δ\n  \
            of the unitary halving subgroup. The exact nature of Δ determines the kind of D[Δ].\n\n\
            Note 2: `IN` shows the intertwining numbers of the ircoreps which classify them into three kinds:\n  \
            `1` = 1st kind: the ircorep is induced by a single irrep of the unitary halving subgroup once,\n  \
            `4` = 2nd kind: the ircorep is induced by a single irrep of the unitary halving subgroup twice,\n  \
            `2` = 3rd kind: the ircorep is induced by an irrep of the unitary halving subgroup and its Wigner conjugate.\n\n\
            Note 3: Only unitary-represented elements are shown in the character table, as characters of\n  \
            antiunitary-represented elements are not invariant under a change of basis.\n\n\
            Refs:\n  \
            Newmarch, J. D. & Golding, R. M. J. Math. Phys. 23, 695–704 (1982)\n  \
            Bradley, C. J. & Davies, B. L. Rev. Mod. Phys. 40, 359–379 (1968)\n  \
            Newmarch, J. D. J. Math. Phys. 24, 742–756 (1983)".log_output_display();
            qsym2_output!("");
        }

        Ok(group)
    }
}

// Specific for unitary-represented and magnetic-represented symmetry groups and determinant numeric types f64 and C128
// ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

#[duplicate_item(
    duplicate!{
        [ dtype_nested; [f64]; [Complex<f64>] ]
        [
            gtype_ [ UnitaryRepresentedSymmetryGroup ]
            dtype_ [ dtype_nested ]
            doc_sub_ [ "Performs representation analysis using a unitary-represented group and stores the result." ]
            analyse_fn_ [ analyse_representation ]
            construct_group_ [ self.construct_unitary_group()? ]
        ]
    }
    duplicate!{
        [ dtype_nested; [f64]; [Complex<f64>] ]
        [
            gtype_ [ MagneticRepresentedSymmetryGroup ]
            dtype_ [ dtype_nested ]
            doc_sub_ [ "Performs corepresentation analysis using a magnetic-represented group and stores the result." ]
            analyse_fn_ [ analyse_corepresentation ]
            construct_group_ [ self.construct_magnetic_group()? ]
        ]
    }
)]
impl<'a> SlaterDeterminantRepAnalysisDriver<'a, gtype_, dtype_> {
    #[doc = doc_sub_]
    ///
    /// Linear independence is checked using the moduli of the overlap eigenvalues. Complex
    /// eigenvalues outside the threshold radius centred at the origin on the Argand diagram are
    /// thus allowed.
    fn analyse_fn_(&mut self) -> Result<(), anyhow::Error> {
        let params = self.parameters;
        let sao = self.construct_sao()?;
        let group = construct_group_;
        log_cc_transversal(&group);
        let _ = find_angular_function_representation(&group, self.angular_function_parameters);
        log_bao(self.determinant.bao());

        // Determinant and orbital symmetries
        let (det_symmetry, mo_symmetries, mo_symmetries_thresholds) = if params
            .analyse_mo_symmetries
        {
            let mos = self.determinant.to_orbitals();
            let (mut det_orbit, mut mo_orbitss) = generate_det_mo_orbits(
                self.determinant,
                &mos,
                &group,
                &sao,
                params.integrality_threshold,
                params.linear_independence_threshold,
                params.symmetry_transformation_kind.clone(),
                params.eigenvalue_comparison_mode.clone(),
            )?;
            det_orbit.calc_xmat(false);
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
            let mo_symmetries = mo_orbitss
                .iter_mut()
                .map(|mo_orbits| {
                    mo_orbits
                        .par_iter_mut()
                        .map(|mo_orbit| {
                            mo_orbit.calc_xmat(false);
                            mo_orbit.analyse_rep().ok()
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
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
                Some(mo_symmetries_thresholds),
            )
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
                .calc_smat(Some(&sao))
                .and_then(|det_orb| det_orb.normalise_smat())
                .map_err(|err| err.to_string())
                .and_then(|det_orb| {
                    det_orb.calc_xmat(false);
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
            (det_symmetry, None, None)
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
                                .calc_smat(self.sao_spatial_4c)?
                                .normalise_smat()?
                                .calc_xmat(false);
                            den_orbit.analyse_rep().map_err(|err| format_err!(err))
                        };
                        (
                            format!("Spin-{ispin} density"),
                            den_sym_res().map_err(|err| err.to_string()),
                        )
                    })
                    .collect::<Vec<_>>();
                let mut extra_den_syms = match self.determinant.spin_constraint() {
                    SpinConstraint::Restricted(_) => {
                        vec![("Total density".to_string(), spin_den_syms[0].1.clone())]
                    }
                    SpinConstraint::Unrestricted(nspins, _)
                    | SpinConstraint::Generalised(nspins, _) => {
                        let total_den_sym_res = || {
                            let nspatial = self.determinant.bao().n_funcs();
                            let zero_den = Density::<dtype_>::builder()
                                .density_matrix(Array2::<dtype_>::zeros((nspatial, nspatial)))
                                .bao(self.determinant.bao())
                                .mol(self.determinant.mol())
                                .complex_symmetric(self.determinant.complex_symmetric())
                                .threshold(self.determinant.threshold())
                                .build()?;
                            let total_den =
                                densities.iter().fold(zero_den, |acc, denmat| acc + denmat);
                            let mut total_den_orbit = DensitySymmetryOrbit::builder()
                                .group(&group)
                                .origin(&total_den)
                                .integrality_threshold(params.integrality_threshold)
                                .linear_independence_threshold(params.linear_independence_threshold)
                                .symmetry_transformation_kind(
                                    params.symmetry_transformation_kind.clone(),
                                )
                                .eigenvalue_comparison_mode(
                                    params.eigenvalue_comparison_mode.clone(),
                                )
                                .build()?;
                            total_den_orbit
                                .calc_smat(self.sao_spatial_4c)?
                                .calc_xmat(false);
                            total_den_orbit
                                .analyse_rep()
                                .map_err(|err| format_err!(err))
                        };
                        let mut extra_syms = vec![(
                            "Total density".to_string(),
                            total_den_sym_res().map_err(|err| err.to_string()),
                        )];
                        extra_syms.extend((0..usize::from(*nspins)).combinations(2).map(
                            |indices| {
                                let i = indices[0];
                                let j = indices[1];
                                let den_ij = &densities[i] - &densities[j];
                                let den_ij_sym_res = || {
                                    let mut den_ij_orbit = DensitySymmetryOrbit::builder()
                                        .group(&group)
                                        .origin(&den_ij)
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
                                        .build()?;
                                    den_ij_orbit
                                        .calc_smat(self.sao_spatial_4c)?
                                        .calc_xmat(false);
                                    den_ij_orbit.analyse_rep().map_err(|err| format_err!(err))
                                };
                                (
                                    format!("Spin-polarised density {i} - {j}"),
                                    den_ij_sym_res().map_err(|err| err.to_string()),
                                )
                            },
                        ));
                        extra_syms
                    }
                };
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
                                mo_den_orbit
                                    .calc_smat(self.sao_spatial_4c)
                                    .ok()?
                                    .normalise_smat()
                                    .ok()?
                                    .calc_xmat(false);
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

impl<'a, G, T> fmt::Display for SlaterDeterminantRepAnalysisDriver<'a, G, T>
where
    G: SymmetryGroupProperties + Clone,
    G::CharTab: SubspaceDecomposable<T>,
    T: ComplexFloat + Lapack,
    <T as ComplexFloat>::Real: From<f64> + fmt::LowerExp + fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write_title(f, "Slater Determinant Symmetry Analysis")?;
        writeln!(f)?;
        writeln!(f, "{}", self.parameters)?;
        Ok(())
    }
}

impl<'a, G, T> fmt::Debug for SlaterDeterminantRepAnalysisDriver<'a, G, T>
where
    G: SymmetryGroupProperties + Clone,
    G::CharTab: SubspaceDecomposable<T>,
    T: ComplexFloat + Lapack,
    <T as ComplexFloat>::Real: From<f64> + fmt::LowerExp + fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{self}")
    }
}

// Specific for unitary-represented symmetry groups and determinant numeric type f64
// '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

impl<'a> QSym2Driver
    for SlaterDeterminantRepAnalysisDriver<'a, UnitaryRepresentedSymmetryGroup, f64>
{
    type Outcome = SlaterDeterminantRepAnalysisResult<'a, UnitaryRepresentedSymmetryGroup, f64>;

    fn result(&self) -> Result<&Self::Outcome, anyhow::Error> {
        self.result
            .as_ref()
            .ok_or_else(|| format_err!("No representation analysis results found."))
    }

    fn run(&mut self) -> Result<(), anyhow::Error> {
        self.log_output_display();
        self.analyse_representation()?;
        self.result()?.log_output_display();
        Ok(())
    }
}

// Specific for unitary-represented symmetry groups and determinant numeric type Complex<f64>
// ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

impl<'a> QSym2Driver
    for SlaterDeterminantRepAnalysisDriver<'a, UnitaryRepresentedSymmetryGroup, Complex<f64>>
{
    type Outcome =
        SlaterDeterminantRepAnalysisResult<'a, UnitaryRepresentedSymmetryGroup, Complex<f64>>;

    fn result(&self) -> Result<&Self::Outcome, anyhow::Error> {
        self.result
            .as_ref()
            .ok_or_else(|| format_err!("No representation analysis results found."))
    }

    fn run(&mut self) -> Result<(), anyhow::Error> {
        self.log_output_display();
        self.analyse_representation()?;
        self.result()?.log_output_display();
        Ok(())
    }
}

// Specific for magnetic-represented symmetry groups and determinant numeric type f64
// ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

impl<'a> QSym2Driver
    for SlaterDeterminantRepAnalysisDriver<'a, MagneticRepresentedSymmetryGroup, f64>
{
    type Outcome = SlaterDeterminantRepAnalysisResult<'a, MagneticRepresentedSymmetryGroup, f64>;

    fn result(&self) -> Result<&Self::Outcome, anyhow::Error> {
        self.result
            .as_ref()
            .ok_or_else(|| format_err!("No representation analysis results found."))
    }

    fn run(&mut self) -> Result<(), anyhow::Error> {
        self.log_output_display();
        self.analyse_corepresentation()?;
        self.result()?.log_output_display();
        Ok(())
    }
}

// Specific for magnetic-represented symmetry groups and determinant numeric type Complex<f64>
// '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

impl<'a> QSym2Driver
    for SlaterDeterminantRepAnalysisDriver<'a, MagneticRepresentedSymmetryGroup, Complex<f64>>
{
    type Outcome =
        SlaterDeterminantRepAnalysisResult<'a, MagneticRepresentedSymmetryGroup, Complex<f64>>;

    fn result(&self) -> Result<&Self::Outcome, anyhow::Error> {
        self.result
            .as_ref()
            .ok_or_else(|| format_err!("No representation analysis results found."))
    }

    fn run(&mut self) -> Result<(), anyhow::Error> {
        self.log_output_display();
        self.analyse_corepresentation()?;
        self.result()?.log_output_display();
        Ok(())
    }
}
