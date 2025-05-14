//! Driver for symmetry analysis of electron densities.

use std::collections::HashSet;
use std::fmt;
use std::ops::Mul;

use anyhow::{self, bail, format_err};
use derive_builder::Builder;
use duplicate::duplicate_item;
use ndarray::Array4;
use ndarray_linalg::types::Lapack;
use num_complex::{Complex, ComplexFloat};
use num_traits::Float;
use serde::{Deserialize, Serialize};

use crate::analysis::{EigenvalueComparisonMode, RepAnalysis};
use crate::chartab::chartab_group::CharacterProperties;
use crate::chartab::SubspaceDecomposable;
use crate::drivers::representation_analysis::angular_function::{
    find_angular_function_representation, find_spinor_function_representation,
    AngularFunctionRepAnalysisParams,
};
use crate::drivers::representation_analysis::{
    fn_construct_magnetic_group, fn_construct_unitary_group, log_bao, log_cc_transversal,
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
use crate::target::density::density_analysis::DensitySymmetryOrbit;
use crate::target::density::Density;

#[cfg(test)]
#[path = "density_tests.rs"]
mod density_tests;

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

/// Structure containing control parameters for electron density representation analysis.
#[derive(Clone, Builder, Debug, Serialize, Deserialize)]
pub struct DensityRepAnalysisParams<T: From<f64>> {
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

    /// Boolean indicating if the Cayley table of the group, if available, should be used to speed
    /// up the computation of orbit overlap matrices.
    #[builder(default = "true")]
    #[serde(default = "default_true")]
    pub use_cayley_table: bool,

    /// The kind of symmetry transformation to be applied on the reference density to generate
    /// the orbit for symmetry analysis.
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

impl<T> DensityRepAnalysisParams<T>
where
    T: Float + From<f64>,
{
    /// Returns a builder to construct a [`DensityRepAnalysisParams`] structure.
    pub fn builder() -> DensityRepAnalysisParamsBuilder<T> {
        DensityRepAnalysisParamsBuilder::default()
    }
}

impl Default for DensityRepAnalysisParams<f64> {
    fn default() -> Self {
        Self::builder()
            .integrality_threshold(1e-7)
            .linear_independence_threshold(1e-7)
            .build()
            .expect("Unable to construct a default `DensityRepAnalysisParams<f64>`.")
    }
}

impl<T> fmt::Display for DensityRepAnalysisParams<T>
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

/// Structure to contain electron density representation analysis results.
#[derive(Clone, Builder)]
pub struct DensityRepAnalysisResult<'a, G, T>
where
    G: SymmetryGroupProperties + Clone,
    G::CharTab: SubspaceDecomposable<T>,
    T: ComplexFloat + Lapack,
    <T as ComplexFloat>::Real: From<f64> + fmt::LowerExp + fmt::Debug,
{
    /// The control parameters used to obtain this set of electron density representation analysis
    /// results.
    parameters: &'a DensityRepAnalysisParams<<T as ComplexFloat>::Real>,

    /// The densities being analysed and their associated names or descriptions.
    densities: Vec<(String, &'a Density<'a, T>)>,

    /// The group used for the representation analysis.
    group: G,

    /// The deduced symmetries of the electron densities.
    density_symmetries: Vec<Result<<G::CharTab as SubspaceDecomposable<T>>::Decomposition, String>>,

    /// The overlap eigenvalues above and below the linear independence threshold for each
    /// electron density symmetry deduction.
    density_symmetries_thresholds: Vec<(Option<T>, Option<T>)>,
}

impl<'a, G, T> DensityRepAnalysisResult<'a, G, T>
where
    G: SymmetryGroupProperties + Clone,
    G::CharTab: SubspaceDecomposable<T>,
    T: ComplexFloat + Lapack,
    <T as ComplexFloat>::Real: From<f64> + fmt::LowerExp + fmt::Debug,
{
    /// Returns a builder to construct a new [`DensityRepAnalysisResultBuilder`]
    /// structure.
    fn builder() -> DensityRepAnalysisResultBuilder<'a, G, T> {
        DensityRepAnalysisResultBuilder::default()
    }
}

impl<'a, G, T> fmt::Display for DensityRepAnalysisResult<'a, G, T>
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

        let density_index_length = usize::try_from(self.densities.len().ilog10() + 1)
            .unwrap_or(1)
            .max(1);
        let density_text_length = self
            .densities
            .iter()
            .map(|(desc, _)| desc.chars().count())
            .max()
            .unwrap_or(7)
            .max(7);
        let symmetry_length = self
            .density_symmetries
            .iter()
            .map(|den_sym_res| {
                den_sym_res
                    .as_ref()
                    .map(|sym| sym.to_string())
                    .unwrap_or_else(|err| err.clone())
                    .chars()
                    .count()
            })
            .max()
            .unwrap_or(8)
            .max(8);
        let eig_above_length: usize = self
            .density_symmetries_thresholds
            .iter()
            .map(|(eig_above_opt, _)| {
                eig_above_opt
                    .as_ref()
                    .map(|eig_above| format!("{eig_above:+.3e}").chars().count())
                    .unwrap_or(10)
            })
            .max()
            .unwrap_or(10)
            .max(10);
        let eig_below_length: usize = self
            .density_symmetries_thresholds
            .iter()
            .map(|(_, eig_below_opt)| {
                eig_below_opt
                    .as_ref()
                    .map(|eig_below| format!("{eig_below:+.3e}").chars().count())
                    .unwrap_or(10)
            })
            .max()
            .unwrap_or(10)
            .max(10);

        let table_width = 10
            + density_index_length
            + density_text_length
            + symmetry_length
            + eig_above_length
            + eig_below_length;

        writeln!(f, "> Density results")?;
        writeln!(f, "{}", "┈".repeat(table_width))?;
        writeln!(
            f,
            " {:>density_index_length$}  {:<density_text_length$}  {:<symmetry_length$}  {:<eig_above_length$}  Eig. below",
            "#",
            "Density",
            "Symmetry",
            "Eig. above",
        )?;
        writeln!(f, "{}", "┈".repeat(table_width))?;
        for (deni, (((den, _), den_sym), (eig_above_opt, eig_below_opt))) in self
            .densities
            .iter()
            .zip(self.density_symmetries.iter())
            .zip(self.density_symmetries_thresholds.iter())
            .enumerate()
        {
            writeln!(
                f,
                " {:>density_index_length$}  {:<density_text_length$}  {:<symmetry_length$}  {:<eig_above_length$}  {}",
                deni,
                den,
                den_sym.as_ref().map(|sym| sym.to_string()).unwrap_or_else(|err| err.clone()),
                eig_above_opt
                    .map(|eig_above| format!("{eig_above:>+.3e}"))
                    .unwrap_or("--".to_string()),
                eig_below_opt
                    .map(|eig_below| format!("{eig_below:>+.3e}"))
                    .unwrap_or("--".to_string()),
            )?;
        }
        writeln!(f, "{}", "┈".repeat(table_width))?;
        writeln!(f)?;

        Ok(())
    }
}

impl<'a, G, T> fmt::Debug for DensityRepAnalysisResult<'a, G, T>
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

impl<'a, G, T> DensityRepAnalysisResult<'a, G, T>
where
    G: SymmetryGroupProperties + Clone,
    G::CharTab: SubspaceDecomposable<T>,
    T: ComplexFloat + Lapack,
    <T as ComplexFloat>::Real: From<f64> + fmt::LowerExp + fmt::Debug + fmt::Display,
{
    /// Returns the symmetries of the specified densities obtained from the analysis result.
    pub fn density_symmetries(
        &self,
    ) -> &Vec<Result<<G::CharTab as SubspaceDecomposable<T>>::Decomposition, String>> {
        &self.density_symmetries
    }
}

// ------
// Driver
// ------

// ~~~~~~~~~~~~~~~~~
// Struct definition
// ~~~~~~~~~~~~~~~~~

/// Driver structure for performing representation analysis on electron densities.
#[derive(Clone, Builder)]
#[builder(build_fn(validate = "Self::validate"))]
pub struct DensityRepAnalysisDriver<'a, G, T>
where
    G: SymmetryGroupProperties + Clone,
    G::CharTab: SubspaceDecomposable<T>,
    T: ComplexFloat + Lapack,
    <T as ComplexFloat>::Real: From<f64> + fmt::LowerExp + fmt::Debug,
{
    /// The control parameters for electron density representation analysis.
    parameters: &'a DensityRepAnalysisParams<<T as ComplexFloat>::Real>,

    /// The densities being analysed and their associated names or descriptions.
    densities: Vec<(String, &'a Density<'a, T>)>,

    /// The result from symmetry-group detection on the underlying molecular structure of the
    /// electron densities.
    symmetry_group: &'a SymmetryGroupDetectionResult,

    /// The atomic-orbital four-centre spatial overlap matrix of the underlying basis set used to
    /// describe the densities.
    sao_spatial_4c: &'a Array4<T>,

    /// The complex-symmetric atomic-orbital four-centre spatial overlap matrix of the underlying
    /// basis set used to describe the densities. This is required if antiunitary symmetry
    /// operations are involved. If none is provided, this will be assumed to be the same as
    /// [`Self::sao_spatial_4c`].
    #[builder(default = "None")]
    sao_spatial_4c_h: Option<&'a Array4<T>>,

    /// The control parameters for symmetry analysis of angular functions.
    angular_function_parameters: &'a AngularFunctionRepAnalysisParams,

    /// The result of the electron density representation analysis.
    #[builder(setter(skip), default = "None")]
    result: Option<DensityRepAnalysisResult<'a, G, T>>,
}

impl<'a, G, T> DensityRepAnalysisDriverBuilder<'a, G, T>
where
    G: SymmetryGroupProperties + Clone,
    G::CharTab: SubspaceDecomposable<T>,
    T: ComplexFloat + Lapack,
    <T as ComplexFloat>::Real: From<f64> + fmt::LowerExp + fmt::Debug,
{
    fn validate(&self) -> Result<(), String> {
        let params = self
            .parameters
            .ok_or("No electron density representation analysis parameters found.".to_string())?;

        let sym_res = self
            .symmetry_group
            .ok_or("No symmetry group information found.".to_string())?;

        let sao_spatial_4c = self
            .sao_spatial_4c
            .ok_or("No four-centre spatial SAO matrix found.".to_string())?;

        if let Some(sao_spatial_4c_h) = self.sao_spatial_4c_h.flatten() {
            if sao_spatial_4c_h.shape() != sao_spatial_4c.shape() {
                return Err(
                    "Mismatched shapes between `sao_spatial_4c` and `sao_spatial_4c_h`."
                        .to_string(),
                );
            }
        }

        let dens = self
            .densities
            .as_ref()
            .ok_or("No electron densities found.".to_string())?;

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
            let baos = dens
                .iter()
                .map(|(_, den)| den.bao())
                .collect::<HashSet<_>>();
            if baos.len() != 1 {
                Err("Inconsistent basis angular order information between densities.".to_string())
            } else {
                let naos = sao_spatial_4c.shape().iter().collect::<HashSet<_>>();
                if naos.len() != 1 {
                    Err("The shape of the four-centre spatial SAO tensor is invalid: all four dimensions must have the same length.".to_string())
                } else {
                    let nao = **naos.iter().next().ok_or(
                        "Unable to extract the dimensions of the four-centre spatial SAO tensor.",
                    )?;
                    if !dens.iter().all(|(_, den)| den.bao().n_funcs() == nao) {
                        Err("The dimensions of the four-centre spatial SAO tensor do not match the number of spatial AO basis functions.".to_string())
                    } else {
                        Ok(())
                    }
                }
            }
        }
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~
// Struct implementations
// ~~~~~~~~~~~~~~~~~~~~~~

// Generic for all symmetry groups G and density numeric type T
// ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

impl<'a, G, T> DensityRepAnalysisDriver<'a, G, T>
where
    G: SymmetryGroupProperties + Clone,
    G::CharTab: SubspaceDecomposable<T>,
    T: ComplexFloat + Lapack,
    <T as ComplexFloat>::Real: From<f64> + fmt::LowerExp + fmt::Debug,
{
    /// Returns a builder to construct a [`DensityRepAnalysisDriver`] structure.
    pub fn builder() -> DensityRepAnalysisDriverBuilder<'a, G, T> {
        DensityRepAnalysisDriverBuilder::default()
    }
}

// Specific for unitary-represented symmetry groups, but generic for density numeric type T
// ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

impl<'a, T> DensityRepAnalysisDriver<'a, UnitaryRepresentedSymmetryGroup, T>
where
    T: ComplexFloat + Lapack + Sync + Send,
    <T as ComplexFloat>::Real: From<f64> + fmt::LowerExp + fmt::Debug + Sync + Send,
    for<'b> Complex<f64>: Mul<&'b T, Output = Complex<f64>>,
{
    fn_construct_unitary_group!(
        /// Constructs the unitary-represented group (which itself can be unitary or magnetic) ready
        /// for electron density representation analysis.
        construct_unitary_group
    );
}

// Specific for magnetic-represented symmetry groups, but generic for density numeric type T
// '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

impl<'a, T> DensityRepAnalysisDriver<'a, MagneticRepresentedSymmetryGroup, T>
where
    T: ComplexFloat + Lapack + Sync + Send,
    <T as ComplexFloat>::Real: From<f64> + Sync + Send + fmt::LowerExp + fmt::Debug,
    for<'b> Complex<f64>: Mul<&'b T, Output = Complex<f64>>,
{
    fn_construct_magnetic_group!(
        /// Constructs the magnetic-represented group (which itself can only be magnetic) ready for
        /// electron density corepresentation analysis.
        construct_magnetic_group
    );
}

// Specific for unitary-represented and magnetic-represented symmetry groups and density numeric types f64 and C128
// ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

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
impl<'a> DensityRepAnalysisDriver<'a, gtype_, dtype_> {
    #[doc = doc_sub_]
    fn analyse_fn_(&mut self) -> Result<(), anyhow::Error> {
        let params = self.parameters;
        let sao_spatial_4c = self.sao_spatial_4c;
        let sao_spatial_4c_h = self.sao_spatial_4c_h;
        let group = construct_group_;
        log_cc_transversal(&group);
        let _ = find_angular_function_representation(&group, self.angular_function_parameters);
        if group.is_double_group() {
            let _ = find_spinor_function_representation(&group, self.angular_function_parameters);
        }
        let bao = self
            .densities
            .iter()
            .next()
            .map(|(_, den)| den.bao())
            .ok_or_else(|| {
                format_err!("Basis angular order information could not be extracted.")
            })?;
        log_bao(bao);

        let (den_symmetries, den_symmetries_thresholds): (Vec<_>, Vec<_>) =
            self.densities.iter().map(|(_, den)| {
                DensitySymmetryOrbit::builder()
                    .group(&group)
                    .origin(den)
                    .integrality_threshold(params.integrality_threshold)
                    .linear_independence_threshold(params.linear_independence_threshold)
                    .symmetry_transformation_kind(params.symmetry_transformation_kind.clone())
                    .eigenvalue_comparison_mode(params.eigenvalue_comparison_mode.clone())
                    .build()
                    .map_err(|err| format_err!(err))
                    .and_then(|mut den_orbit| {
                        den_orbit
                            .calc_smat(Some(sao_spatial_4c), sao_spatial_4c_h, params.use_cayley_table)?
                            .normalise_smat()?
                            .calc_xmat(false)?;
                        let density_symmetry_thresholds = den_orbit
                            .smat_eigvals
                            .as_ref()
                            .map(|eigvals| {
                                let mut eigvals_vec = eigvals.iter().collect::<Vec<_>>();
                                match den_orbit.eigenvalue_comparison_mode() {
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
                                let eigval_above = match den_orbit.eigenvalue_comparison_mode() {
                                    EigenvalueComparisonMode::Modulus => eigvals_vec
                                        .iter()
                                        .find(|val| {
                                            val.abs() >= den_orbit.linear_independence_threshold
                                        })
                                        .copied()
                                        .copied(),
                                    EigenvalueComparisonMode::Real => eigvals_vec
                                        .iter()
                                        .find(|val| {
                                            val.re() >= den_orbit.linear_independence_threshold
                                        })
                                        .copied()
                                        .copied(),
                                };
                                eigvals_vec.reverse();
                                let eigval_below = match den_orbit.eigenvalue_comparison_mode() {
                                    EigenvalueComparisonMode::Modulus => eigvals_vec
                                        .iter()
                                        .find(|val| {
                                            val.abs() < den_orbit.linear_independence_threshold
                                        })
                                        .copied()
                                        .copied(),
                                    EigenvalueComparisonMode::Real => eigvals_vec
                                        .iter()
                                        .find(|val| {
                                            val.re() < den_orbit.linear_independence_threshold
                                        })
                                        .copied()
                                        .copied(),
                                };
                                (eigval_above, eigval_below)
                            })
                            .unwrap_or((None, None));
                        let den_sym = den_orbit.analyse_rep().map_err(|err| err.to_string());
                        Ok((den_sym, density_symmetry_thresholds))
                    })
                    .unwrap_or_else(|err| (Err(err.to_string()), (None, None)))
            }).unzip();

        let result = DensityRepAnalysisResult::builder()
            .parameters(params)
            .densities(self.densities.clone())
            .group(group)
            .density_symmetries(den_symmetries)
            .density_symmetries_thresholds(den_symmetries_thresholds)
            .build()?;
        self.result = Some(result);

        Ok(())
    }
}

// ~~~~~~~~~~~~~~~~~~~~~
// Trait implementations
// ~~~~~~~~~~~~~~~~~~~~~

// Generic for all symmetry groups G and density numeric type T
// ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

impl<'a, G, T> fmt::Display for DensityRepAnalysisDriver<'a, G, T>
where
    G: SymmetryGroupProperties + Clone,
    G::CharTab: SubspaceDecomposable<T>,
    T: ComplexFloat + Lapack,
    <T as ComplexFloat>::Real: From<f64> + fmt::LowerExp + fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write_title(f, "Electron Density Symmetry Analysis")?;
        writeln!(f)?;
        writeln!(f, "{}", self.parameters)?;
        Ok(())
    }
}

impl<'a, G, T> fmt::Debug for DensityRepAnalysisDriver<'a, G, T>
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

// Specific for unitary/magnetic-represented groups and density numeric type f64/Complex<f64>
// ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

#[duplicate_item(
    duplicate!{
        [ dtype_nested; [f64]; [Complex<f64>] ]
        [
            gtype_ [ UnitaryRepresentedSymmetryGroup ]
            dtype_ [ dtype_nested ]
            analyse_fn_ [ analyse_representation ]
        ]
    }
    duplicate!{
        [ dtype_nested; [f64]; [Complex<f64>] ]
        [
            gtype_ [ MagneticRepresentedSymmetryGroup ]
            dtype_ [ dtype_nested ]
            analyse_fn_ [ analyse_corepresentation ]
        ]
    }
)]
impl<'a> QSym2Driver for DensityRepAnalysisDriver<'a, gtype_, dtype_> {
    type Params = DensityRepAnalysisParams<f64>;

    type Outcome = DensityRepAnalysisResult<'a, gtype_, dtype_>;

    fn result(&self) -> Result<&Self::Outcome, anyhow::Error> {
        self.result.as_ref().ok_or_else(|| {
            format_err!("No electron density representation analysis results found.")
        })
    }

    fn run(&mut self) -> Result<(), anyhow::Error> {
        self.log_output_display();
        self.analyse_fn_()?;
        self.result()?.log_output_display();
        Ok(())
    }
}
