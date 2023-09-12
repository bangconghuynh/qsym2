use std::fmt;
use std::ops::Mul;

use anyhow::{self, bail, format_err};
use derive_builder::Builder;
use duplicate::duplicate_item;
use ndarray_linalg::types::Lapack;
use num_complex::{Complex, ComplexFloat};
use num_traits::Float;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::analysis::{EigenvalueComparisonMode, RepAnalysis};
use crate::chartab::chartab_group::CharacterProperties;
use crate::chartab::SubspaceDecomposable;
use crate::drivers::representation_analysis::angular_function::{
    find_angular_function_representation, AngularFunctionRepAnalysisParams,
};
use crate::drivers::representation_analysis::{
    fn_construct_magnetic_group, fn_construct_unitary_group, log_cc_transversal,
    CharacterTableDisplay, MagneticSymmetryAnalysisKind,
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
use crate::target::vibration::vibration_analysis::VibrationalCoordinateSymmetryOrbit;
use crate::target::vibration::VibrationalCoordinateCollection;

#[cfg(test)]
#[path = "vibrational_coordinate_tests.rs"]
mod vibrational_coordinate_tests;

// ==================
// Struct definitions
// ==================

// ----------
// Parameters
// ----------

const fn default_symbolic() -> Option<CharacterTableDisplay> {
    Some(CharacterTableDisplay::Symbolic)
}

/// A structure containing control parameters for vibrational coordinate representation analysis.
#[derive(Clone, Builder, Debug, Serialize, Deserialize)]
pub struct VibrationalCoordinateRepAnalysisParams<T: From<f64>> {
    /// Threshold for checking if subspace multiplicities are integral.
    pub integrality_threshold: T,

    /// Threshold for determining zero eigenvalues in the orbit overlap matrix.
    pub linear_independence_threshold: T,

    /// Option indicating if the magnetic group is to be used for symmetry analysis, and if so,
    /// whether unitary representations or unitary-antiunitary corepresentations should be used.
    #[builder(default = "None")]
    #[serde(default)]
    pub use_magnetic_group: Option<MagneticSymmetryAnalysisKind>,

    /// Boolean indicating if the double group is to be used for symmetry analysis.
    #[builder(default = "false")]
    #[serde(default)]
    pub use_double_group: bool,

    /// The kind of symmetry transformation to be applied on the reference vibrational coordinate to
    /// generate the orbit for symmetry analysis.
    #[builder(default = "SymmetryTransformationKind::Spatial")]
    #[serde(default)]
    pub symmetry_transformation_kind: SymmetryTransformationKind,

    /// Option indicating if the character table of the group used for symmetry analysis is to be
    /// printed out.
    #[builder(default = "Some(CharacterTableDisplay::Symbolic)")]
    #[serde(default = "default_symbolic")]
    pub write_character_table: Option<CharacterTableDisplay>,

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

impl<T> VibrationalCoordinateRepAnalysisParams<T>
where
    T: Float + From<f64>,
{
    /// Returns a builder to construct a [`VibrationalCoordinateRepAnalysisParams`] structure.
    pub fn builder() -> VibrationalCoordinateRepAnalysisParamsBuilder<T> {
        VibrationalCoordinateRepAnalysisParamsBuilder::default()
    }
}

impl Default for VibrationalCoordinateRepAnalysisParams<f64> {
    fn default() -> Self {
        Self::builder()
            .integrality_threshold(1e-7)
            .linear_independence_threshold(1e-7)
            .build()
            .expect("Unable to construct a default `VibrationalCoordinateRepAnalysisParams<f64>`.")
    }
}

impl<T> fmt::Display for VibrationalCoordinateRepAnalysisParams<T>
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

/// A structure to contain vibrational coordinate representation analysis results.
#[derive(Clone, Builder)]
pub struct VibrationalCoordinateRepAnalysisResult<'a, G, T>
where
    G: SymmetryGroupProperties + Clone,
    G::CharTab: SubspaceDecomposable<T>,
    T: ComplexFloat + Lapack,
    <T as ComplexFloat>::Real: From<f64> + fmt::LowerExp + fmt::Debug,
{
    /// The control parameters used to obtain this set of vibrational coordinate representation
    /// analysis results.
    parameters: &'a VibrationalCoordinateRepAnalysisParams<<T as ComplexFloat>::Real>,

    /// The collection of vibrational coordinates being analysed.
    vibrational_coordinate_collection: &'a VibrationalCoordinateCollection<'a, T>,

    /// The group used for the representation analysis.
    group: G,

    /// The deduced symmetries of the vibrational coordinates.
    vib_symmetries: Vec<Result<<G::CharTab as SubspaceDecomposable<T>>::Decomposition, String>>,

    /// The overlap eigenvalues above and below the linear independence threshold for each
    /// vibrational coordinate symmetry deduction.
    vib_symmetries_thresholds: Option<Vec<(Option<T>, Option<T>)>>,
}

impl<'a, G, T> VibrationalCoordinateRepAnalysisResult<'a, G, T>
where
    G: SymmetryGroupProperties + Clone,
    G::CharTab: SubspaceDecomposable<T>,
    T: ComplexFloat + Lapack,
    <T as ComplexFloat>::Real: From<f64> + fmt::LowerExp + fmt::Debug,
{
    /// Returns a builder to construct a new [`VibrationalCoordinateRepAnalysisResultBuilder`]
    /// structure.
    fn builder() -> VibrationalCoordinateRepAnalysisResultBuilder<'a, G, T> {
        VibrationalCoordinateRepAnalysisResultBuilder::default()
    }
}

impl<'a, G, T> fmt::Display for VibrationalCoordinateRepAnalysisResult<'a, G, T>
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

        let vib_index_length =
            (usize::try_from(self.vibrational_coordinate_collection.n_modes().ilog10())
                .unwrap_or(4)
                + 2)
            .max(4);
        let vib_frequency_length = self
            .vibrational_coordinate_collection
            .frequencies()
            .iter()
            .map(|freq| format!("{freq:+.3}").chars().count())
            .max()
            .unwrap_or(9)
            .max(9);
        let vib_symmetry_length = self
            .vib_symmetries
            .iter()
            .map(|vib_sym_res| {
                vib_sym_res
                    .as_ref()
                    .map(|sym| sym.to_string())
                    .unwrap_or("--".to_string())
                    .chars()
                    .count()
            })
            .max()
            .unwrap_or(8)
            .max(8);

        let vib_eig_above_length: usize = self
            .vib_symmetries_thresholds
            .as_ref()
            .map(|vib_symmetries_thresholds| {
                vib_symmetries_thresholds
                    .iter()
                    .map(|(above, _)| {
                        above
                            .as_ref()
                            .map(|eig| format!("{eig:+.3e}"))
                            .unwrap_or("--".to_string())
                            .chars()
                            .count()
                    })
                    .max()
                    .unwrap_or(10)
                    .max(10)
            })
            .unwrap_or(10);
        let vib_eig_below_length: usize = self
            .vib_symmetries_thresholds
            .as_ref()
            .map(|vib_symmetries_thresholds| {
                vib_symmetries_thresholds
                    .iter()
                    .map(|(_, below)| {
                        below
                            .as_ref()
                            .map(|eig| format!("{eig:+.3e}"))
                            .unwrap_or("--".to_string())
                            .chars()
                            .count()
                    })
                    .max()
                    .unwrap_or(10)
                    .max(10)
            })
            .unwrap_or(10);

        writeln!(f, "> Vibrational coordinate results")?;

        writeln!(
            f,
            "{}",
            "┈".repeat(
                10 + vib_index_length
                    + vib_frequency_length
                    + vib_symmetry_length
                    + vib_eig_above_length
                    + vib_eig_below_length
            )
        )?;
        writeln!(
            f,
            " {:>vib_index_length$}  {:<vib_frequency_length$}  {:<vib_symmetry_length$}  {:<vib_eig_above_length$}  Eig. below",
            "Mode", "Frequency", "Symmetry", "Eig. above"
        )?;
        writeln!(
            f,
            "{}",
            "┈".repeat(
                10 + vib_index_length
                    + vib_frequency_length
                    + vib_symmetry_length
                    + vib_eig_above_length
                    + vib_eig_below_length
            )
        )?;

        for (vibi, vib_sym) in self.vib_symmetries.iter().enumerate() {
            let vib_frequency_str = self
                .vibrational_coordinate_collection
                .frequencies()
                .get(vibi)
                .map(|freq| format!("{freq:>+vib_frequency_length$.3}"))
                .unwrap_or("--".to_string());
            let vib_sym_str = vib_sym
                .as_ref()
                .map(|sym| sym.to_string())
                .unwrap_or("--".to_string());
            let (eig_above_str, eig_below_str) = self
                .vib_symmetries_thresholds
                .as_ref()
                .map(|vib_symmetries_thresholds| {
                    vib_symmetries_thresholds
                        .get(vibi)
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
                " {vibi:>vib_index_length$}  \
                {vib_frequency_str:<vib_frequency_length$}  \
                {vib_sym_str:<vib_symmetry_length$}  \
                {eig_above_str:<vib_eig_above_length$}  \
                {eig_below_str}"
            )?;
        }
        writeln!(
            f,
            "{}",
            "┈".repeat(
                10 + vib_index_length
                    + vib_frequency_length
                    + vib_symmetry_length
                    + vib_eig_above_length
                    + vib_eig_below_length
            )
        )?;

        Ok(())
    }
}

impl<'a, G, T> fmt::Debug for VibrationalCoordinateRepAnalysisResult<'a, G, T>
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

impl<'a, G, T> VibrationalCoordinateRepAnalysisResult<'a, G, T>
where
    G: SymmetryGroupProperties + Clone,
    G::CharTab: SubspaceDecomposable<T>,
    T: ComplexFloat + Lapack,
    <T as ComplexFloat>::Real: From<f64> + fmt::LowerExp + fmt::Debug + fmt::Display,
{
    /// Returns the vibrational coordinate symmetries obtained from the analysis result.
    pub fn vibrational_coordinate_symmetries(
        &self,
    ) -> &Vec<Result<<G::CharTab as SubspaceDecomposable<T>>::Decomposition, String>> {
        &self.vib_symmetries
    }
}

// ------
// Driver
// ------

// ~~~~~~~~~~~~~~~~~
// Struct definition
// ~~~~~~~~~~~~~~~~~

/// A driver structure for performing representation analysis on vibrational coordinates.
#[derive(Clone, Builder)]
#[builder(build_fn(validate = "Self::validate"))]
pub struct VibrationalCoordinateRepAnalysisDriver<'a, G, T>
where
    G: SymmetryGroupProperties + Clone,
    G::CharTab: SubspaceDecomposable<T>,
    T: ComplexFloat + Lapack,
    <T as ComplexFloat>::Real: From<f64> + fmt::LowerExp + fmt::Debug,
{
    /// The control parameters for Slater determinant representation analysis.
    parameters: &'a VibrationalCoordinateRepAnalysisParams<<T as ComplexFloat>::Real>,

    /// The collection of vibrational coordinates to be analysed.
    vibrational_coordinate_collection: &'a VibrationalCoordinateCollection<'a, T>,

    /// The result from symmetry-group detection on the underlying molecular structure of the
    /// vibrational coordinates.
    symmetry_group: &'a SymmetryGroupDetectionResult,

    /// The control parameters for symmetry analysis of angular functions.
    angular_function_parameters: &'a AngularFunctionRepAnalysisParams,

    /// The result of the vibrational coordinate representation analysis.
    #[builder(setter(skip), default = "None")]
    result: Option<VibrationalCoordinateRepAnalysisResult<'a, G, T>>,
}

impl<'a, G, T> VibrationalCoordinateRepAnalysisDriverBuilder<'a, G, T>
where
    G: SymmetryGroupProperties + Clone,
    G::CharTab: SubspaceDecomposable<T>,
    T: ComplexFloat + Lapack,
    <T as ComplexFloat>::Real: From<f64> + fmt::LowerExp + fmt::Debug,
{
    fn validate(&self) -> Result<(), String> {
        let params = self.parameters.ok_or(
            "No vibrational coordinate representation analysis parameters found.".to_string(),
        )?;

        let sym_res = self
            .symmetry_group
            .ok_or("No symmetry group information found.".to_string())?;

        let sym = if params.use_magnetic_group.is_some() {
            sym_res
                .magnetic_symmetry
                .as_ref()
                .ok_or("Magnetic symmetry requested for representation analysis, but no magnetic symmetry found.")?
        } else {
            &sym_res.unitary_symmetry
        };

        if sym.is_infinite() && params.infinite_order_to_finite.is_none() {
            Err(
                format!(
                    "Representation analysis cannot be performed using the entirety of the infinite group `{}`. \
                    Consider setting the parameter `infinite_order_to_finite` to restrict to a finite subgroup instead.",
                    sym.group_name.as_ref().expect("No symmetry group name found.")
                )
            )
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

impl<'a, G, T> VibrationalCoordinateRepAnalysisDriver<'a, G, T>
where
    G: SymmetryGroupProperties + Clone,
    G::CharTab: SubspaceDecomposable<T>,
    T: ComplexFloat + Lapack,
    <T as ComplexFloat>::Real: From<f64> + fmt::LowerExp + fmt::Debug,
{
    /// Returns a builder to construct a [`VibrationalCoordinateRepAnalysisDriver`] structure.
    pub fn builder() -> VibrationalCoordinateRepAnalysisDriverBuilder<'a, G, T> {
        VibrationalCoordinateRepAnalysisDriverBuilder::default()
    }
}

// Specific for unitary-represented symmetry groups, but generic for determinant numeric type T
// ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

impl<'a, T> VibrationalCoordinateRepAnalysisDriver<'a, UnitaryRepresentedSymmetryGroup, T>
where
    T: ComplexFloat + Lapack + Sync + Send,
    <T as ComplexFloat>::Real: From<f64> + fmt::LowerExp + fmt::Debug + Sync + Send,
    for<'b> Complex<f64>: Mul<&'b T, Output = Complex<f64>>,
{
    fn_construct_unitary_group!(
        /// Constructs the unitary-represented group (which itself can be unitary or magnetic) ready
        /// for vibrational coordinate representation analysis.
        construct_unitary_group
    );
}

// Specific for magnetic-represented symmetry groups, but generic for determinant numeric type T
// ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

impl<'a, T> VibrationalCoordinateRepAnalysisDriver<'a, MagneticRepresentedSymmetryGroup, T>
where
    T: ComplexFloat + Lapack + Sync + Send,
    <T as ComplexFloat>::Real: From<f64> + Sync + Send + fmt::LowerExp + fmt::Debug,
    for<'b> Complex<f64>: Mul<&'b T, Output = Complex<f64>>,
{
    fn_construct_magnetic_group!(
        /// Constructs the magnetic-represented group (which itself can only be magnetic) ready for
        /// vibrational coordinate corepresentation analysis.
        construct_magnetic_group
    );
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
impl<'a> VibrationalCoordinateRepAnalysisDriver<'a, gtype_, dtype_> {
    #[doc = doc_sub_]
    ///
    /// Linear independence is checked using the moduli of the overlap eigenvalues. Complex
    /// eigenvalues outside the threshold radius centred at the origin on the Argand diagram are
    /// thus allowed.
    fn analyse_fn_(&mut self) -> Result<(), anyhow::Error> {
        let params = self.parameters;
        let group = construct_group_;
        log_cc_transversal(&group);
        let _ = find_angular_function_representation(&group, self.angular_function_parameters);

        let (vib_symmetries, vib_symmetries_thresholds) = {
            let vibs = self
                .vibrational_coordinate_collection
                .to_vibrational_coordinates();
            let vib_orbits = vibs.par_iter().map(|vib| {
                let mut vib_orbit = VibrationalCoordinateSymmetryOrbit::builder()
                    .group(&group)
                    .origin(vib)
                    .integrality_threshold(params.integrality_threshold)
                    .linear_independence_threshold(params.linear_independence_threshold)
                    .symmetry_transformation_kind(params.symmetry_transformation_kind.clone())
                    .eigenvalue_comparison_mode(params.eigenvalue_comparison_mode.clone())
                    .build()?;
                vib_orbit
                    .calc_smat(None)
                    .and_then(|orbit| orbit.calc_xmat(false))?;
                Ok::<_, anyhow::Error>(vib_orbit)
            });
            let (vib_symmetries, vib_symmetries_thresholds): (Vec<_>, Vec<_>) = vib_orbits
                .map(|vib_orbit_res| {
                    vib_orbit_res
                        .as_ref()
                        .map(|vib_orbit| {
                            let sym_res = vib_orbit.analyse_rep().map_err(|err| err.to_string());
                            let eigs = vib_orbit
                                .smat_eigvals
                                .as_ref()
                                .map(|eigvals| {
                                    let mut eigvals_vec = eigvals.iter().collect::<Vec<_>>();
                                    match vib_orbit.eigenvalue_comparison_mode {
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
                                    let eigval_above = match vib_orbit.eigenvalue_comparison_mode {
                                        EigenvalueComparisonMode::Modulus => eigvals_vec
                                            .iter()
                                            .find(|val| {
                                                val.abs() >= vib_orbit.linear_independence_threshold
                                            })
                                            .copied()
                                            .copied(),
                                        EigenvalueComparisonMode::Real => eigvals_vec
                                            .iter()
                                            .find(|val| {
                                                val.re() >= vib_orbit.linear_independence_threshold
                                            })
                                            .copied()
                                            .copied(),
                                    };
                                    eigvals_vec.reverse();
                                    let eigval_below = match vib_orbit.eigenvalue_comparison_mode {
                                        EigenvalueComparisonMode::Modulus => eigvals_vec
                                            .iter()
                                            .find(|val| {
                                                val.abs() < vib_orbit.linear_independence_threshold
                                            })
                                            .copied()
                                            .copied(),
                                        EigenvalueComparisonMode::Real => eigvals_vec
                                            .iter()
                                            .find(|val| {
                                                val.re() < vib_orbit.linear_independence_threshold
                                            })
                                            .copied()
                                            .copied(),
                                    };
                                    (eigval_above, eigval_below)
                                })
                                .unwrap_or_else(|| (None, None));
                            (sym_res, eigs)
                        })
                        .unwrap_or((Err("--".to_string()), (None, None)))
                })
                .unzip();
            (vib_symmetries, vib_symmetries_thresholds)
        };

        let result = VibrationalCoordinateRepAnalysisResult::builder()
            .parameters(params)
            .vibrational_coordinate_collection(self.vibrational_coordinate_collection)
            .group(group)
            .vib_symmetries(vib_symmetries)
            .vib_symmetries_thresholds(Some(vib_symmetries_thresholds))
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

impl<'a, G, T> fmt::Display for VibrationalCoordinateRepAnalysisDriver<'a, G, T>
where
    G: SymmetryGroupProperties + Clone,
    G::CharTab: SubspaceDecomposable<T>,
    T: ComplexFloat + Lapack,
    <T as ComplexFloat>::Real: From<f64> + fmt::LowerExp + fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write_title(f, "Vibrational Coordinate Symmetry Analysis")?;
        writeln!(f)?;
        writeln!(f, "{}", self.parameters)?;
        Ok(())
    }
}

impl<'a, G, T> fmt::Debug for VibrationalCoordinateRepAnalysisDriver<'a, G, T>
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
    for VibrationalCoordinateRepAnalysisDriver<'a, UnitaryRepresentedSymmetryGroup, f64>
{
    type Outcome = VibrationalCoordinateRepAnalysisResult<'a, UnitaryRepresentedSymmetryGroup, f64>;

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
    for VibrationalCoordinateRepAnalysisDriver<'a, UnitaryRepresentedSymmetryGroup, Complex<f64>>
{
    type Outcome =
        VibrationalCoordinateRepAnalysisResult<'a, UnitaryRepresentedSymmetryGroup, Complex<f64>>;

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
    for VibrationalCoordinateRepAnalysisDriver<'a, MagneticRepresentedSymmetryGroup, f64>
{
    type Outcome =
        VibrationalCoordinateRepAnalysisResult<'a, MagneticRepresentedSymmetryGroup, f64>;

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
    for VibrationalCoordinateRepAnalysisDriver<'a, MagneticRepresentedSymmetryGroup, Complex<f64>>
{
    type Outcome =
        VibrationalCoordinateRepAnalysisResult<'a, MagneticRepresentedSymmetryGroup, Complex<f64>>;

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
