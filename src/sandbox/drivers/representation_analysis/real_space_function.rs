//! Sandbox driver for symmetry analysis of RealSpaceFunctiones.

use std::fmt;
use std::ops::Mul;

use anyhow::{self, bail, format_err};
use derive_builder::Builder;
use duplicate::duplicate_item;
use nalgebra::Point3;
use ndarray::Array1;
use ndarray_linalg::types::Lapack;
use num_complex::{Complex, ComplexFloat};
use num_traits::Float;
use serde::{Deserialize, Serialize};

use crate::analysis::{
    EigenvalueComparisonMode, Orbit, Overlap, ProjectionDecomposition, RepAnalysis,
    log_overlap_eigenvalues,
};
use crate::chartab::SubspaceDecomposable;
use crate::chartab::chartab_group::CharacterProperties;
use crate::drivers::QSym2Driver;
use crate::drivers::representation_analysis::angular_function::{
    AngularFunctionRepAnalysisParams, find_angular_function_representation,
    find_spinor_function_representation,
};
use crate::drivers::representation_analysis::{
    CharacterTableDisplay, MagneticSymmetryAnalysisKind, fn_construct_magnetic_group,
    fn_construct_unitary_group, log_cc_transversal,
};
use crate::drivers::symmetry_group_detection::SymmetryGroupDetectionResult;
use crate::group::{GroupProperties, MagneticRepresentedGroup, UnitaryRepresentedGroup};
use crate::io::format::{
    QSym2Output, log_subtitle, nice_bool, qsym2_output, write_subtitle, write_title,
};
use crate::sandbox::target::real_space_function::RealSpaceFunction;
use crate::sandbox::target::real_space_function::real_space_function_analysis::RealSpaceFunctionSymmetryOrbit;
use crate::symmetry::symmetry_group::{
    MagneticRepresentedSymmetryGroup, SymmetryGroupProperties, UnitaryRepresentedSymmetryGroup,
};
use crate::symmetry::symmetry_transformation::SymmetryTransformationKind;

#[cfg(test)]
#[path = "real_space_function_tests.rs"]
mod real_space_function_tests;

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

/// Structure containing control parameters for real-space function representation analysis.
#[derive(Clone, Builder, Debug, Serialize, Deserialize)]
pub struct RealSpaceFunctionRepAnalysisParams<T: From<f64>> {
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

    /// The kind of symmetry transformation to be applied on the reference real-space function to
    /// generate the orbit for symmetry analysis.
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

impl<T> RealSpaceFunctionRepAnalysisParams<T>
where
    T: Float + From<f64>,
{
    /// Returns a builder to construct a [`RealSpaceFunctionRepAnalysisParams`] structure.
    pub fn builder() -> RealSpaceFunctionRepAnalysisParamsBuilder<T> {
        RealSpaceFunctionRepAnalysisParamsBuilder::default()
    }
}

impl Default for RealSpaceFunctionRepAnalysisParams<f64> {
    fn default() -> Self {
        Self::builder()
            .integrality_threshold(1e-7)
            .linear_independence_threshold(1e-7)
            .build()
            .expect("Unable to construct a default `RealSpaceFunctionRepAnalysisParams<f64>`.")
    }
}

impl<T> fmt::Display for RealSpaceFunctionRepAnalysisParams<T>
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

/// Structure to contain real-space function representation analysis results.
#[derive(Clone, Builder)]
pub struct RealSpaceFunctionRepAnalysisResult<'a, G, T, F>
where
    G: SymmetryGroupProperties + Clone,
    G::CharTab: SubspaceDecomposable<T>,
    T: ComplexFloat + Lapack,
    <T as ComplexFloat>::Real: From<f64> + fmt::LowerExp + fmt::Debug,
    F: Clone + Fn(&Point3<f64>) -> T,
{
    /// The control parameters used to obtain this set of real-space function representation
    /// analysis results.
    parameters: &'a RealSpaceFunctionRepAnalysisParams<<T as ComplexFloat>::Real>,

    /// The RealSpaceFunction being analysed.
    real_space_function: &'a RealSpaceFunction<T, F>,

    /// The group used for the representation analysis.
    group: G,

    /// The deduced symmetry of the real-space function.
    real_space_function_symmetry:
        Result<<G::CharTab as SubspaceDecomposable<T>>::Decomposition, String>,
}

impl<'a, G, T, F> RealSpaceFunctionRepAnalysisResult<'a, G, T, F>
where
    G: SymmetryGroupProperties + Clone,
    G::CharTab: SubspaceDecomposable<T>,
    T: ComplexFloat + Lapack,
    <T as ComplexFloat>::Real: From<f64> + fmt::LowerExp + fmt::Debug,
    F: Clone + Fn(&Point3<f64>) -> T,
{
    /// Returns a builder to construct a new [`RealSpaceFunctionRepAnalysisResultBuilder`] structure.
    pub fn builder() -> RealSpaceFunctionRepAnalysisResultBuilder<'a, G, T, F> {
        RealSpaceFunctionRepAnalysisResultBuilder::default()
    }
}

impl<'a, G, T, F> fmt::Display for RealSpaceFunctionRepAnalysisResult<'a, G, T, F>
where
    G: SymmetryGroupProperties + Clone,
    G::CharTab: SubspaceDecomposable<T>,
    T: ComplexFloat + Lapack,
    <T as ComplexFloat>::Real: From<f64> + fmt::LowerExp + fmt::Debug + fmt::Display,
    F: Clone + Fn(&Point3<f64>) -> T,
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
        writeln!(f, "> Overall real-space function result")?;
        writeln!(
            f,
            "  Grid size: {} {}",
            self.real_space_function.grid_points().len(),
            if self.real_space_function.grid_points().len() == 1 {
                "point"
            } else {
                "points"
            }
        )?;
        writeln!(
            f,
            "  Symmetry: {}",
            self.real_space_function_symmetry
                .as_ref()
                .map(|s| s.to_string())
                .unwrap_or_else(|err| format!("-- ({err})"))
        )?;
        writeln!(f)?;

        Ok(())
    }
}

impl<'a, G, T, F> fmt::Debug for RealSpaceFunctionRepAnalysisResult<'a, G, T, F>
where
    G: SymmetryGroupProperties + Clone,
    G::CharTab: SubspaceDecomposable<T>,
    T: ComplexFloat + Lapack,
    <T as ComplexFloat>::Real: From<f64> + fmt::LowerExp + fmt::Debug + fmt::Display,
    F: Clone + Fn(&Point3<f64>) -> T,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{self}")
    }
}

impl<'a, G, T, F> RealSpaceFunctionRepAnalysisResult<'a, G, T, F>
where
    G: SymmetryGroupProperties + Clone,
    G::CharTab: SubspaceDecomposable<T>,
    T: ComplexFloat + Lapack,
    <T as ComplexFloat>::Real: From<f64> + fmt::LowerExp + fmt::Debug + fmt::Display,
    F: Clone + Fn(&Point3<f64>) -> T,
{
    /// Returns the real-space function symmetry obtained from the analysis result.
    pub fn real_space_function_symmetry(
        &self,
    ) -> &Result<<G::CharTab as SubspaceDecomposable<T>>::Decomposition, String> {
        &self.real_space_function_symmetry
    }

    /// Returns the parameters used for the representation analysis.
    pub fn parameters(&self) -> &RealSpaceFunctionRepAnalysisParams<<T as ComplexFloat>::Real> {
        self.parameters
    }
}

// ------
// Driver
// ------

// ~~~~~~~~~~~~~~~~~
// Struct definition
// ~~~~~~~~~~~~~~~~~

/// Driver structure for performing representation analysis on real-space functions.
#[derive(Clone, Builder)]
#[builder(build_fn(validate = "Self::validate"))]
pub struct RealSpaceFunctionRepAnalysisDriver<'a, G, T, F>
where
    G: SymmetryGroupProperties + Clone,
    G::CharTab: SubspaceDecomposable<T>,
    T: ComplexFloat + Lapack,
    <T as ComplexFloat>::Real: From<f64> + fmt::LowerExp + fmt::Debug,
    F: Clone + Fn(&Point3<f64>) -> T,
{
    /// The control parameters for real-space function representation analysis.
    parameters: &'a RealSpaceFunctionRepAnalysisParams<<T as ComplexFloat>::Real>,

    /// The real-space function to be analysed.
    real_space_function: &'a RealSpaceFunction<T, F>,

    /// The result from symmetry-group detection that will then be used to construct the full group
    /// for the definition and analysis of the real-space function.
    symmetry_group: &'a SymmetryGroupDetectionResult,

    /// The weight used in the evaluation of the inner products between real-space functions.
    weight: &'a Array1<T>,

    /// The control parameters for symmetry analysis of angular functions.
    angular_function_parameters: &'a AngularFunctionRepAnalysisParams,

    /// The result of the real-space function representation analysis.
    #[builder(setter(skip), default = "None")]
    result: Option<RealSpaceFunctionRepAnalysisResult<'a, G, T, F>>,
}

impl<'a, G, T, F> RealSpaceFunctionRepAnalysisDriverBuilder<'a, G, T, F>
where
    G: SymmetryGroupProperties + Clone,
    G::CharTab: SubspaceDecomposable<T>,
    T: ComplexFloat + Lapack,
    <T as ComplexFloat>::Real: From<f64> + fmt::LowerExp + fmt::Debug,
    F: Clone + Fn(&Point3<f64>) -> T,
{
    fn validate(&self) -> Result<(), String> {
        let _ = self
            .real_space_function
            .ok_or("No real-space function specified.".to_string())?;

        let params = self.parameters.ok_or(
            "No real-space function representation analysis parameters found.".to_string(),
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
            Err(format!(
                "Representation analysis cannot be performed using the entirety of the infinite group `{}`. \
                    Consider setting the parameter `infinite_order_to_finite` to restrict to a finite subgroup instead.",
                sym.group_name
                    .as_ref()
                    .expect("No symmetry group name found.")
            ))
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

impl<'a, G, T, F> RealSpaceFunctionRepAnalysisDriver<'a, G, T, F>
where
    G: SymmetryGroupProperties + Clone,
    G::CharTab: SubspaceDecomposable<T>,
    T: ComplexFloat + Lapack,
    <T as ComplexFloat>::Real: From<f64> + fmt::LowerExp + fmt::Debug,
    F: Clone + Fn(&Point3<f64>) -> T,
{
    /// Returns a builder to construct a [`RealSpaceFunctionRepAnalysisDriver`] structure.
    pub fn builder() -> RealSpaceFunctionRepAnalysisDriverBuilder<'a, G, T, F> {
        RealSpaceFunctionRepAnalysisDriverBuilder::default()
    }
}

// Specific for unitary-represented symmetry groups, but generic for function numeric type T
// '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

impl<'a, T, F> RealSpaceFunctionRepAnalysisDriver<'a, UnitaryRepresentedSymmetryGroup, T, F>
where
    T: ComplexFloat + Lapack + Sync + Send,
    <T as ComplexFloat>::Real: From<f64> + fmt::LowerExp + fmt::Debug + Sync + Send,
    for<'b> Complex<f64>: Mul<&'b T, Output = Complex<f64>>,
    F: Clone + Fn(&Point3<f64>) -> T,
{
    fn_construct_unitary_group!(
        /// Constructs the unitary-represented group (which itself can be unitary or magnetic) ready
        /// for real-space function representation analysis.
        construct_unitary_group
    );
}

// Specific for magnetic-represented symmetry groups, but generic for function numeric type T
// ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

impl<'a, T, F> RealSpaceFunctionRepAnalysisDriver<'a, MagneticRepresentedSymmetryGroup, T, F>
where
    T: ComplexFloat + Lapack + Sync + Send,
    <T as ComplexFloat>::Real: From<f64> + Sync + Send + fmt::LowerExp + fmt::Debug,
    for<'b> Complex<f64>: Mul<&'b T, Output = Complex<f64>>,
    F: Clone + Fn(&Point3<f64>) -> T,
{
    fn_construct_magnetic_group!(
        /// Constructs the magnetic-represented group (which itself can only be magnetic) ready for
        /// real-space function corepresentation analysis.
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
            calc_projections_ [
                log_subtitle("Real-space function projection decompositions");
                qsym2_output!("");
                qsym2_output!("  Projections are defined w.r.t. the following inner product:");
                qsym2_output!("    {}", real_space_function_orbit.origin().overlap_definition());
                qsym2_output!("");
                real_space_function_orbit
                    .projections_to_string(
                        &real_space_function_orbit.calc_projection_compositions()?,
                        params.integrality_threshold,
                    )
                    .log_output_display();
                qsym2_output!("");
            ]
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
            calc_projections_ [ ]
        ]
    }
)]
impl<'a, F> RealSpaceFunctionRepAnalysisDriver<'a, gtype_, dtype_, F>
where
    F: Clone + Sync + Send + Fn(&Point3<f64>) -> dtype_,
{
    #[doc = doc_sub_]
    fn analyse_fn_(&mut self) -> Result<(), anyhow::Error> {
        let params = self.parameters;
        let group = construct_group_;
        log_cc_transversal(&group);
        let _ = find_angular_function_representation(&group, self.angular_function_parameters);
        if group.is_double_group() {
            let _ = find_spinor_function_representation(&group, self.angular_function_parameters);
        }

        let mut real_space_function_orbit = RealSpaceFunctionSymmetryOrbit::builder()
            .origin(self.real_space_function)
            .group(&group)
            .integrality_threshold(params.integrality_threshold)
            .linear_independence_threshold(params.linear_independence_threshold)
            .symmetry_transformation_kind(params.symmetry_transformation_kind.clone())
            .eigenvalue_comparison_mode(params.eigenvalue_comparison_mode.clone())
            .build()?;
        let real_space_function_symmetry = real_space_function_orbit
            .calc_smat(Some(self.weight), None, params.use_cayley_table)
            .and_then(|real_space_function_orb| real_space_function_orb.normalise_smat())
            .map_err(|err| err.to_string())
            .and_then(|real_space_function_orb| {
                real_space_function_orb
                    .calc_xmat(false)
                    .map_err(|err| err.to_string())?;
                if params.write_overlap_eigenvalues
                    && let Some(smat_eigvals) = real_space_function_orb.smat_eigvals.as_ref()
                {
                    log_overlap_eigenvalues(
                        "Real-space function orbit overlap eigenvalues",
                        smat_eigvals,
                        params.linear_independence_threshold,
                        &params.eigenvalue_comparison_mode,
                    );
                    qsym2_output!("");
                }
                real_space_function_orb
                    .analyse_rep()
                    .map_err(|err| err.to_string())
            });

        {
            calc_projections_
        }

        let result = RealSpaceFunctionRepAnalysisResult::builder()
            .parameters(params)
            .real_space_function(self.real_space_function)
            .group(group)
            .real_space_function_symmetry(real_space_function_symmetry)
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

impl<'a, G, T, F> fmt::Display for RealSpaceFunctionRepAnalysisDriver<'a, G, T, F>
where
    G: SymmetryGroupProperties + Clone,
    G::CharTab: SubspaceDecomposable<T>,
    T: ComplexFloat + Lapack,
    <T as ComplexFloat>::Real: From<f64> + fmt::LowerExp + fmt::Debug,
    F: Clone + Fn(&Point3<f64>) -> T,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write_title(f, "Real-Space Function Symmetry Analysis")?;
        writeln!(f)?;
        writeln!(f, "{}", self.parameters)?;
        Ok(())
    }
}

impl<'a, G, T, F> fmt::Debug for RealSpaceFunctionRepAnalysisDriver<'a, G, T, F>
where
    G: SymmetryGroupProperties + Clone,
    G::CharTab: SubspaceDecomposable<T>,
    T: ComplexFloat + Lapack,
    <T as ComplexFloat>::Real: From<f64> + fmt::LowerExp + fmt::Debug,
    F: Clone + Fn(&Point3<f64>) -> T,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{self}")
    }
}

// Specific for unitary/magnetic-represented groups and function numeric type f64/Complex<f64>
// '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

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
impl<'a, F> QSym2Driver for RealSpaceFunctionRepAnalysisDriver<'a, gtype_, dtype_, F>
where
    F: Clone + Sync + Send + Fn(&Point3<f64>) -> dtype_,
{
    type Params = RealSpaceFunctionRepAnalysisParams<f64>;

    type Outcome = RealSpaceFunctionRepAnalysisResult<'a, gtype_, dtype_, F>;

    fn result(&self) -> Result<&Self::Outcome, anyhow::Error> {
        self.result
            .as_ref()
            .ok_or_else(|| format_err!("No real-space function analysis results found."))
    }

    fn run(&mut self) -> Result<(), anyhow::Error> {
        self.log_output_display();
        self.analyse_fn_()?;
        self.result()?.log_output_display();
        Ok(())
    }
}
