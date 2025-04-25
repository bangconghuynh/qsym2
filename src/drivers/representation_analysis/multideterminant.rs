//! Driver for symmetry analysis of multi-determinantal wavefunctions.

use std::collections::HashSet;
use std::fmt;
use std::hash::Hash;
use std::ops::Mul;

use anyhow::{self, bail, format_err};
use derive_builder::Builder;
use duplicate::duplicate_item;
use ndarray::{s, Array2};
use ndarray_linalg::types::Lapack;
use num_complex::{Complex, ComplexFloat};
use num_traits::Float;
use serde::{Deserialize, Serialize};

use crate::analysis::{
    log_overlap_eigenvalues, EigenvalueComparisonMode, Overlap, ProjectionDecomposition,
    RepAnalysis,
};
use crate::angmom::spinor_rotation_3d::{SpinConstraint, StructureConstraint};
use crate::chartab::chartab_group::CharacterProperties;
use crate::chartab::SubspaceDecomposable;
use crate::drivers::representation_analysis::angular_function::{
    find_angular_function_representation, AngularFunctionRepAnalysisParams,
};
use crate::drivers::representation_analysis::{
    fn_construct_magnetic_group, fn_construct_unitary_group, log_bao, log_cc_transversal,
    CharacterTableDisplay, MagneticSymmetryAnalysisKind,
};
use crate::drivers::symmetry_group_detection::SymmetryGroupDetectionResult;
use crate::drivers::QSym2Driver;
use crate::group::{GroupProperties, MagneticRepresentedGroup, UnitaryRepresentedGroup};
use crate::io::format::{
    log_micsec_begin, log_micsec_end, log_subtitle, nice_bool, qsym2_output, write_subtitle,
    write_title, QSym2Output,
};
use crate::symmetry::symmetry_group::{
    MagneticRepresentedSymmetryGroup, SymmetryGroupProperties, UnitaryRepresentedSymmetryGroup,
};
use crate::symmetry::symmetry_transformation::SymmetryTransformationKind;
use crate::target::determinant::SlaterDeterminant;
use crate::target::noci::basis::{Basis, EagerBasis, OrbitBasis};
use crate::target::noci::multideterminant::multideterminant_analysis::MultiDeterminantSymmetryOrbit;
use crate::target::noci::multideterminant::MultiDeterminant;

#[cfg(test)]
#[path = "multideterminant_tests.rs"]
mod multideterminant_tests;

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

/// Structure containing control parameters for multi-determinantal wavefunction representation
/// analysis.
#[derive(Clone, Builder, Debug, Serialize, Deserialize)]
pub struct MultiDeterminantRepAnalysisParams<T: From<f64>> {
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

    /// The kind of symmetry transformation to be applied on the reference multi-determinantal
    /// wavefunction to generate the orbit for symmetry analysis.
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

impl<T> MultiDeterminantRepAnalysisParams<T>
where
    T: Float + From<f64>,
{
    /// Returns a builder to construct a [`MultiDeterminantRepAnalysisParams`] structure.
    pub fn builder() -> MultiDeterminantRepAnalysisParamsBuilder<T> {
        MultiDeterminantRepAnalysisParamsBuilder::default()
    }
}

impl Default for MultiDeterminantRepAnalysisParams<f64> {
    fn default() -> Self {
        Self::builder()
            .integrality_threshold(1e-7)
            .linear_independence_threshold(1e-7)
            .build()
            .expect("Unable to construct a default `MultiDeterminantRepAnalysisParams<f64>`.")
    }
}

impl<T> fmt::Display for MultiDeterminantRepAnalysisParams<T>
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

/// Structure to contain multi-determinantal representation analysis results.
#[derive(Clone, Builder)]
pub struct MultiDeterminantRepAnalysisResult<'a, G, T, B, SC>
where
    G: SymmetryGroupProperties + Clone,
    G::CharTab: SubspaceDecomposable<T>,
    T: ComplexFloat + Lapack,
    <T as ComplexFloat>::Real: From<f64> + fmt::LowerExp + fmt::Debug,
    B: Basis<SlaterDeterminant<'a, T, SC>> + Clone,
    SC: StructureConstraint + Hash + Eq + fmt::Display,
{
    /// The control parameters used to obtain this set of multi-determinantal wavefunction
    /// representation analysis results.
    parameters: &'a MultiDeterminantRepAnalysisParams<<T as ComplexFloat>::Real>,

    /// The multi-determinantal wavefunctions being analysed.
    multidets: Vec<&'a MultiDeterminant<'a, T, B, SC>>,

    /// The group used for the representation analysis.
    group: G,

    /// The deduced symmetries of the multi-determinantal wavefunctions.
    multidet_symmetries:
        Vec<Result<<G::CharTab as SubspaceDecomposable<T>>::Decomposition, String>>,

    /// The overlap eigenvalues above and below the linear independence threshold for each
    /// multi-determinantal wavefunction symmetry deduction.
    multidet_symmetries_thresholds: Vec<(Option<T>, Option<T>)>,
}

impl<'a, G, T, B, SC> MultiDeterminantRepAnalysisResult<'a, G, T, B, SC>
where
    G: SymmetryGroupProperties + Clone,
    G::CharTab: SubspaceDecomposable<T>,
    T: ComplexFloat + Lapack,
    <T as ComplexFloat>::Real: From<f64> + fmt::LowerExp + fmt::Debug,
    B: Basis<SlaterDeterminant<'a, T, SC>> + Clone,
    SC: StructureConstraint + Clone + Hash + Eq + fmt::Display,
{
    /// Returns a builder to construct a new [`MultiDeterminantRepAnalysisResultBuilder`]
    /// structure.
    fn builder() -> MultiDeterminantRepAnalysisResultBuilder<'a, G, T, B, SC> {
        MultiDeterminantRepAnalysisResultBuilder::default()
    }

    /// Returns the multi-determinantal wavefunction symmetries obtained from the analysis result.
    pub fn multidet_symmetries(
        &self,
    ) -> &Vec<Result<<G::CharTab as SubspaceDecomposable<T>>::Decomposition, String>> {
        &self.multidet_symmetries
    }
}

impl<'a, G, T, B, SC> fmt::Display for MultiDeterminantRepAnalysisResult<'a, G, T, B, SC>
where
    G: SymmetryGroupProperties + Clone,
    G::CharTab: SubspaceDecomposable<T>,
    T: ComplexFloat + Lapack,
    <T as ComplexFloat>::Real: From<f64> + fmt::LowerExp + fmt::Debug + fmt::Display,
    B: Basis<SlaterDeterminant<'a, T, SC>> + Clone,
    SC: StructureConstraint + Clone + Hash + Eq + fmt::Display,
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

        let multidet_index_length = usize::try_from(self.multidets.len().ilog10() + 1).unwrap_or(4);
        let multidet_symmetry_length = self
            .multidet_symmetries
            .iter()
            .map(|multidet_sym| {
                multidet_sym
                    .as_ref()
                    .map(|sym| sym.to_string())
                    .unwrap_or_else(|err| err.clone())
                    .chars()
                    .count()
            })
            .max()
            .unwrap_or(0)
            .max(8);
        let multidet_energy_length = self
            .multidets
            .iter()
            .map(|multidet| {
                multidet
                    .energy()
                    .map(|v| format!("{v:+.7}").chars().count())
                    .unwrap_or(2)
            })
            .max()
            .unwrap_or(6)
            .max(6);

        let multidet_eig_above_length: usize = self
            .multidet_symmetries_thresholds
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
            .max(10);

        let multidet_eig_below_length: usize = self
            .multidet_symmetries_thresholds
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
            .max(10);

        let table_width = 10
            + multidet_index_length
            + multidet_energy_length
            + multidet_symmetry_length
            + multidet_eig_above_length
            + multidet_eig_below_length;

        writeln!(f, "> Multi-determinantal results")?;
        writeln!(
            f,
            "  Spin constraint: {}",
            self.multidets
                .get(0)
                .map(|multidet_0| multidet_0.structure_constraint().to_string().to_lowercase())
                .unwrap_or("--".to_string())
        )?;
        writeln!(f, "{}", "┈".repeat(table_width))?;
        writeln!(
            f,
            " {:>multidet_index_length$}  {:<multidet_energy_length$}  {:<multidet_symmetry_length$}  {:<multidet_eig_above_length$}  Eig. below",
            "#",
            "Energy",
            "Symmetry",
            "Eig. above",
        )?;
        writeln!(f, "{}", "┈".repeat(table_width))?;

        for (multidet_i, multidet) in self.multidets.iter().enumerate() {
            let multidet_energy_str = multidet
                .energy()
                .map(|multidet_energy| format!("{multidet_energy:>+multidet_energy_length$.7}"))
                .unwrap_or("--".to_string());
            let multidet_sym_str = self
                .multidet_symmetries
                .get(multidet_i)
                .ok_or_else(|| format!("Unable to retrieve the symmetry of multideterminantal wavefunction index `{multidet_i}`."))
                .and_then(|sym_res| sym_res.as_ref().map(|sym| sym.to_string()).map_err(|err| err.to_string()))
                .unwrap_or_else(|err| err);

            let (eig_above_str, eig_below_str) = self
                .multidet_symmetries_thresholds
                .get(multidet_i)
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
                .unwrap_or(("--".to_string(), "--".to_string()));
            writeln!(
                f,
                " {multidet_i:>multidet_index_length$}  \
                {multidet_energy_str:<multidet_energy_length$}  \
                {multidet_sym_str:<multidet_symmetry_length$}  \
                {eig_above_str:<multidet_eig_above_length$}  \
                {eig_below_str}"
            )?;
        }

        writeln!(f, "{}", "┈".repeat(table_width))?;
        writeln!(f)?;

        Ok(())
    }
}

impl<'a, G, T, B, SC> fmt::Debug for MultiDeterminantRepAnalysisResult<'a, G, T, B, SC>
where
    G: SymmetryGroupProperties + Clone,
    G::CharTab: SubspaceDecomposable<T>,
    T: ComplexFloat + Lapack,
    <T as ComplexFloat>::Real: From<f64> + fmt::LowerExp + fmt::Debug + fmt::Display,
    B: Basis<SlaterDeterminant<'a, T, SC>> + Clone,
    SC: StructureConstraint + Clone + Hash + Eq + fmt::Display,
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

/// Driver structure for performing representation analysis on multi-determinantal wavefunctions.
#[derive(Clone, Builder)]
#[builder(build_fn(validate = "Self::validate"))]
pub struct MultiDeterminantRepAnalysisDriver<'a, G, T, B, SC>
where
    G: SymmetryGroupProperties + Clone,
    G::CharTab: SubspaceDecomposable<T>,
    T: ComplexFloat + Lapack,
    <T as ComplexFloat>::Real: From<f64> + fmt::LowerExp + fmt::Debug,
    B: Basis<SlaterDeterminant<'a, T, SC>> + Clone,
    SC: StructureConstraint + Hash + Eq + fmt::Display,
{
    /// The control parameters for multi-determinantal wavefunction representation analysis.
    parameters: &'a MultiDeterminantRepAnalysisParams<<T as ComplexFloat>::Real>,

    /// The multi-determinantal wavefunctions to be analysed.
    multidets: Vec<&'a MultiDeterminant<'a, T, B, SC>>,

    /// The result from symmetry-group detection on the underlying molecular structure of the
    /// multi-determinantal wavefunctions.
    symmetry_group: &'a SymmetryGroupDetectionResult,

    /// The atomic-orbital spatial overlap matrix of the underlying basis set used to describe the
    /// wavefunctions.
    sao_spatial: &'a Array2<T>,

    /// The complex-symmetric atomic-orbital spatial overlap matrix of the underlying basis set used
    /// to describe the wavefunctions. This is required if antiunitary symmetry operations are
    /// involved. If none is provided, this will be assumed to be the same as [`Self::sao_spatial`].
    #[builder(default = "None")]
    sao_spatial_h: Option<&'a Array2<T>>,

    /// The control parameters for symmetry analysis of angular functions.
    angular_function_parameters: &'a AngularFunctionRepAnalysisParams,

    /// The result of the multi-determinantal wavefunction representation analysis.
    #[builder(setter(skip), default = "None")]
    result: Option<MultiDeterminantRepAnalysisResult<'a, G, T, B, SC>>,
}

impl<'a, G, T, B, SC> MultiDeterminantRepAnalysisDriverBuilder<'a, G, T, B, SC>
where
    G: SymmetryGroupProperties + Clone,
    G::CharTab: SubspaceDecomposable<T>,
    T: ComplexFloat + Lapack,
    <T as ComplexFloat>::Real: From<f64> + fmt::LowerExp + fmt::Debug,
    B: Basis<SlaterDeterminant<'a, T, SC>> + Clone,
    SC: StructureConstraint + Hash + Eq + fmt::Display,
{
    fn validate(&self) -> Result<(), String> {
        let params = self.parameters.ok_or(
            "No multi-determinantal wavefunction representation analysis parameters found."
                .to_string(),
        )?;

        let sym_res = self
            .symmetry_group
            .ok_or("No symmetry group information found.".to_string())?;

        let sao_spatial = self
            .sao_spatial
            .ok_or("No spatial SAO matrix found.".to_string())?;

        if let Some(sao_spatial_h) = self.sao_spatial_h.flatten() {
            if sao_spatial_h.shape() != sao_spatial.shape() {
                return Err(
                    "Mismatched shapes between `sao_spatial` and `sao_spatial_h`.".to_string(),
                );
            }
        }

        let multidets = self
            .multidets
            .as_ref()
            .ok_or("No multi-determinantal wavefunctions found.".to_string())?;
        let mut n_spatial_set = multidets
            .iter()
            .flat_map(|multidet| {
                multidet
                    .basis()
                    .iter()
                    .map(|det_res| det_res.map(|det| det.bao().n_funcs()))
            })
            .collect::<Result<HashSet<usize>, _>>()
            .map_err(|err| err.to_string())?;
        let n_spatial = if n_spatial_set.len() == 1 {
            n_spatial_set
                .drain()
                .next()
                .ok_or("Unable to retrieve the number of spatial AO basis functions.".to_string())
        } else {
            Err("Inconsistent numbers of spatial AO basis functions across multi-determinantal wavefunctions.".to_string())
        }?;

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
        } else if n_spatial != sao_spatial.nrows() || n_spatial != sao_spatial.ncols() {
            Err("The dimensions of the spatial SAO matrix do not match the number of spatial AO basis functions.".to_string())
        } else {
            Ok(())
        }
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~
// Struct implementations
// ~~~~~~~~~~~~~~~~~~~~~~

// Generic for all symmetry groups G and wavefunction numeric type T
// '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

impl<'a, G, T, B, SC> MultiDeterminantRepAnalysisDriver<'a, G, T, B, SC>
where
    G: SymmetryGroupProperties + Clone,
    G::CharTab: SubspaceDecomposable<T>,
    T: ComplexFloat + Lapack,
    <T as ComplexFloat>::Real: From<f64> + fmt::LowerExp + fmt::Debug,
    B: Basis<SlaterDeterminant<'a, T, SC>> + Clone,
    SC: StructureConstraint + Clone + Hash + Eq + fmt::Display,
{
    /// Returns a builder to construct a [`MultiDeterminantRepAnalysisDriver`] structure.
    pub fn builder() -> MultiDeterminantRepAnalysisDriverBuilder<'a, G, T, B, SC> {
        MultiDeterminantRepAnalysisDriverBuilder::default()
    }

    /// Constructs the appropriate atomic-orbital overlap matrix based on the spin constraint of
    /// the multi-determinantal wavefunctions.
    fn construct_sao(&self) -> Result<(Array2<T>, Option<Array2<T>>), anyhow::Error> {
        let mut structure_constraint_set = self
            .multidets
            .iter()
            .map(|multidet| multidet.structure_constraint())
            .collect::<HashSet<_>>();
        let structure_constraint = if structure_constraint_set.len() == 1 {
            structure_constraint_set.drain().next().ok_or(format_err!(
                "Unable to retrieve the structure constraint of the multi-determinantal wavefunctions."
            ))
        } else {
            Err(format_err!(
                "Inconsistent structure constraints across multi-determinantal wavefunctions."
            ))
        }?;

        let nbas = self.sao_spatial.nrows();
        let ncomps = structure_constraint.n_explicit_comps_per_coefficient_matrix();

        let sao = {
            let mut sao_mut = Array2::zeros((ncomps * nbas, ncomps * nbas));
            (0..ncomps).for_each(|icomp| {
                let start = icomp * nbas;
                let end = (icomp + 1) * nbas;
                sao_mut
                    .slice_mut(s![start..end, start..end])
                    .assign(self.sao_spatial);
            });
            sao_mut
        };

        let sao_h = self.sao_spatial_h.map(|sao_spatial_h| {
            let mut sao_h_mut = Array2::zeros((ncomps * nbas, ncomps * nbas));
            (0..ncomps).for_each(|icomp| {
                let start = icomp * nbas;
                let end = (icomp + 1) * nbas;
                sao_h_mut
                    .slice_mut(s![start..end, start..end])
                    .assign(sao_spatial_h);
            });
            sao_h_mut
        });

        Ok((sao, sao_h))
    }
}

// Specific for unitary-represented symmetry groups, but generic for wavefunction numeric type T
// '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

impl<'a, T, B, SC> MultiDeterminantRepAnalysisDriver<'a, UnitaryRepresentedSymmetryGroup, T, B, SC>
where
    T: ComplexFloat + Lapack + Sync + Send,
    <T as ComplexFloat>::Real: From<f64> + fmt::LowerExp + fmt::Debug + Sync + Send,
    for<'b> Complex<f64>: Mul<&'b T, Output = Complex<f64>>,
    B: Basis<SlaterDeterminant<'a, T, SC>> + Clone,
    SC: StructureConstraint + Hash + Eq + fmt::Display,
{
    fn_construct_unitary_group!(
        /// Constructs the unitary-represented group (which itself can be unitary or magnetic) ready
        /// for Slater determinant representation analysis.
        construct_unitary_group
    );
}

// Specific for magnetic-represented symmetry groups, but generic for wavefunction numeric type T
// ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

impl<'a, T, B, SC> MultiDeterminantRepAnalysisDriver<'a, MagneticRepresentedSymmetryGroup, T, B, SC>
where
    T: ComplexFloat + Lapack + Sync + Send,
    <T as ComplexFloat>::Real: From<f64> + Sync + Send + fmt::LowerExp + fmt::Debug,
    for<'b> Complex<f64>: Mul<&'b T, Output = Complex<f64>>,
    B: Basis<SlaterDeterminant<'a, T, SC>> + Clone,
    SC: StructureConstraint + Hash + Eq + fmt::Display,
{
    fn_construct_magnetic_group!(
        /// Constructs the magnetic-represented group (which itself can only be magnetic) ready for
        /// Slater determinant corepresentation analysis.
        construct_magnetic_group
    );
}

// Specific for unitary-represented and magnetic-represented symmetry groups and wavefunction numeric types f64 and C128
// '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

#[duplicate_item(
    duplicate!{
        [ dtype_nested; [f64]; [Complex<f64>] ]
        duplicate!{
            [ sctype_nested; [SpinConstraint] ]
            duplicate!{
                [
                    [
                        btype_nested [OrbitBasis<'a, UnitaryRepresentedSymmetryGroup, SlaterDeterminant<'a, dtype_nested, sctype_nested>>]
                        calc_smat_nested [calc_smat_optimised]
                    ]
                    [
                        btype_nested [EagerBasis<SlaterDeterminant<'a, dtype_nested, sctype_nested>>]
                        calc_smat_nested [calc_smat]
                    ]
                ]
                [
                    gtype_ [ UnitaryRepresentedSymmetryGroup ]
                    dtype_ [ dtype_nested ]
                    btype_ [ btype_nested ]
                    sctype_ [ sctype_nested ]
                    doc_sub_ [ "Performs representation analysis using a unitary-represented group and stores the result." ]
                    analyse_fn_ [ analyse_representation ]
                    construct_group_ [ self.construct_unitary_group()? ]
                    calc_smat_ [ calc_smat_nested ]
                    calc_projections_ [
                        log_subtitle("Multi-determinantal wavefunction projection decompositions");
                        qsym2_output!("");
                        qsym2_output!("  Projections are defined w.r.t. the following inner product:");
                        qsym2_output!("    {}", multidet_orbit.origin().overlap_definition());
                        qsym2_output!("");
                        multidet_orbit
                            .projections_to_string(
                                &multidet_orbit.calc_projection_compositions()?,
                                params.integrality_threshold,
                            )
                            .log_output_display();
                        qsym2_output!("");
                    ]
                ]
            }
        }
    }
    duplicate!{
        [ dtype_nested; [f64]; [Complex<f64>] ]
        duplicate!{
            [ sctype_nested; [SpinConstraint] ]
            duplicate!{
                [
                    [
                        btype_nested [OrbitBasis<'a, MagneticRepresentedSymmetryGroup, SlaterDeterminant<'a, dtype_nested, sctype_nested>>]
                        calc_smat_nested [calc_smat_optimised]
                    ]
                    [
                        btype_nested [EagerBasis<SlaterDeterminant<'a, dtype_nested, sctype_nested>>]
                        calc_smat_nested [calc_smat]
                    ]
                ]
                [
                    gtype_ [ MagneticRepresentedSymmetryGroup ]
                    dtype_ [ dtype_nested ]
                    btype_ [ btype_nested ]
                    sctype_ [ sctype_nested ]
                    doc_sub_ [ "Performs corepresentation analysis using a magnetic-represented group and stores the result." ]
                    analyse_fn_ [ analyse_corepresentation ]
                    construct_group_ [ self.construct_magnetic_group()? ]
                    calc_smat_ [ calc_smat_nested ]
                    calc_projections_ [ ]
                ]
            }
    }
    }
)]
impl<'a> MultiDeterminantRepAnalysisDriver<'a, gtype_, dtype_, btype_, sctype_> {
    #[doc = doc_sub_]
    fn analyse_fn_(&mut self) -> Result<(), anyhow::Error> {
        let params = self.parameters;
        let (sao, sao_h) = self.construct_sao()?;
        let group = construct_group_;
        log_cc_transversal(&group);
        let _ = find_angular_function_representation(&group, self.angular_function_parameters);
        if let Some(det) = self
            .multidets
            .get(0)
            .and_then(|multidet| multidet.basis().first())
        {
            log_bao(det.bao());
        }

        let (multidet_symmetries, multidet_symmetries_thresholds): (Vec<_>, Vec<_>) = self.multidets
            .iter()
            .enumerate()
            .map(|(i, multidet)| {
                log_micsec_begin(&format!("Multi-determinantal wavefunction {i}"));
                qsym2_output!("");
                let res = MultiDeterminantSymmetryOrbit::builder()
                    .group(&group)
                    .origin(multidet)
                    .integrality_threshold(params.integrality_threshold)
                    .linear_independence_threshold(params.linear_independence_threshold)
                    .symmetry_transformation_kind(params.symmetry_transformation_kind.clone())
                    .eigenvalue_comparison_mode(params.eigenvalue_comparison_mode.clone())
                    .build()
                    .map_err(|err| format_err!(err))
                    .and_then(|mut multidet_orbit| {
                        multidet_orbit
                            .calc_smat_(Some(&sao), sao_h.as_ref(), params.use_cayley_table)?
                            .normalise_smat()?
                            .calc_xmat(false)?;
                        log_overlap_eigenvalues(
                            "Overlap eigenvalues",
                            multidet_orbit.smat_eigvals.as_ref().ok_or(format_err!("Orbit overlap eigenvalues not found."))?,
                            params.linear_independence_threshold,
                            &params.eigenvalue_comparison_mode
                        );
                        qsym2_output!("");
                        let multidet_symmetry_thresholds = multidet_orbit
                            .smat_eigvals
                            .as_ref()
                            .map(|eigvals| {
                                let mut eigvals_vec = eigvals.iter().collect::<Vec<_>>();
                                match multidet_orbit.eigenvalue_comparison_mode() {
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
                                let eigval_above = match multidet_orbit.eigenvalue_comparison_mode() {
                                    EigenvalueComparisonMode::Modulus => eigvals_vec
                                        .iter()
                                        .find(|val| {
                                            val.abs() >= multidet_orbit.linear_independence_threshold
                                        })
                                        .copied()
                                        .copied(),
                                    EigenvalueComparisonMode::Real => eigvals_vec
                                        .iter()
                                        .find(|val| {
                                            val.re() >= multidet_orbit.linear_independence_threshold
                                        })
                                        .copied()
                                        .copied(),
                                };
                                eigvals_vec.reverse();
                                let eigval_below = match multidet_orbit.eigenvalue_comparison_mode() {
                                    EigenvalueComparisonMode::Modulus => eigvals_vec
                                        .iter()
                                        .find(|val| {
                                            val.abs() < multidet_orbit.linear_independence_threshold
                                        })
                                        .copied()
                                        .copied(),
                                    EigenvalueComparisonMode::Real => eigvals_vec
                                        .iter()
                                        .find(|val| {
                                            val.re() < multidet_orbit.linear_independence_threshold
                                        })
                                        .copied()
                                        .copied(),
                                };
                                (eigval_above, eigval_below)
                            })
                            .unwrap_or((None, None));
                        let multidet_sym = multidet_orbit.analyse_rep().map_err(|err| err.to_string());
                        { calc_projections_ }
                        Ok((multidet_sym, multidet_symmetry_thresholds))
                    })
                    .unwrap_or_else(|err| (Err(err.to_string()), (None, None)));
                log_micsec_end(&format!("Multi-determinantal wavefunction {i}"));
                qsym2_output!("");
                res
            }).unzip();

        let result = MultiDeterminantRepAnalysisResult::builder()
            .parameters(params)
            .multidets(self.multidets.clone())
            .group(group)
            .multidet_symmetries(multidet_symmetries)
            .multidet_symmetries_thresholds(multidet_symmetries_thresholds)
            .build()?;
        self.result = Some(result);

        Ok(())
    }
}

// ~~~~~~~~~~~~~~~~~~~~~
// Trait implementations
// ~~~~~~~~~~~~~~~~~~~~~

// Generic for all symmetry groups G, basis B, and wavefunction numeric type T
// '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

impl<'a, G, T, B, SC> fmt::Display for MultiDeterminantRepAnalysisDriver<'a, G, T, B, SC>
where
    G: SymmetryGroupProperties + Clone,
    G::CharTab: SubspaceDecomposable<T>,
    T: ComplexFloat + Lapack,
    <T as ComplexFloat>::Real: From<f64> + fmt::LowerExp + fmt::Debug,
    B: Basis<SlaterDeterminant<'a, T, SC>> + Clone,
    SC: StructureConstraint + Hash + Eq + fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write_title(f, "Multi-determinantal Wavefunction Symmetry Analysis")?;
        writeln!(f)?;
        writeln!(f, "{}", self.parameters)?;
        Ok(())
    }
}

impl<'a, G, T, B, SC> fmt::Debug for MultiDeterminantRepAnalysisDriver<'a, G, T, B, SC>
where
    G: SymmetryGroupProperties + Clone,
    G::CharTab: SubspaceDecomposable<T>,
    T: ComplexFloat + Lapack,
    <T as ComplexFloat>::Real: From<f64> + fmt::LowerExp + fmt::Debug,
    B: Basis<SlaterDeterminant<'a, T, SC>> + Clone,
    SC: StructureConstraint + Hash + Eq + fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{self}")
    }
}

// Specific for unitary/magnetic-represented groups and determinant numeric type f64/Complex<f64>
// ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

#[duplicate_item(
    duplicate!{
        [ dtype_nested; [f64]; [Complex<f64>] ]
        duplicate!{
            [ sctype_nested; [SpinConstraint] ]
            duplicate!{
                [
                    btype_nested;
                    [OrbitBasis<'a, UnitaryRepresentedSymmetryGroup, SlaterDeterminant<'a, dtype_nested, sctype_nested>>];
                    [EagerBasis<SlaterDeterminant<'a, dtype_nested, sctype_nested>>]
                ]
                [
                    gtype_ [ UnitaryRepresentedSymmetryGroup ]
                    dtype_ [ dtype_nested ]
                    btype_ [ btype_nested ]
                    sctype_ [ sctype_nested ]
                    analyse_fn_ [ analyse_representation ]
                ]
            }
        }
    }
    duplicate!{
        [ dtype_nested; [f64]; [Complex<f64>] ]
        duplicate!{
            [ sctype_nested; [SpinConstraint] ]
            duplicate!{
                [
                    btype_nested;
                    [OrbitBasis<'a, MagneticRepresentedSymmetryGroup, SlaterDeterminant<'a, dtype_nested, sctype_nested>>];
                    [EagerBasis<SlaterDeterminant<'a, dtype_nested, sctype_nested>>]
                ]
                [
                    gtype_ [ MagneticRepresentedSymmetryGroup ]
                    dtype_ [ dtype_nested ]
                    btype_ [ btype_nested ]
                    sctype_ [ sctype_nested ]
                    analyse_fn_ [ analyse_corepresentation ]
                ]
            }
        }
    }
)]
impl<'a> QSym2Driver for MultiDeterminantRepAnalysisDriver<'a, gtype_, dtype_, btype_, sctype_> {
    type Params = MultiDeterminantRepAnalysisParams<f64>;

    type Outcome = MultiDeterminantRepAnalysisResult<'a, gtype_, dtype_, btype_, sctype_>;

    fn result(&self) -> Result<&Self::Outcome, anyhow::Error> {
        self.result.as_ref().ok_or_else(|| {
            format_err!(
                "No multi-determinantal wavefunction representation analysis results found."
            )
        })
    }

    fn run(&mut self) -> Result<(), anyhow::Error> {
        self.log_output_display();
        self.analyse_fn_()?;
        self.result()?.log_output_display();
        Ok(())
    }
}
