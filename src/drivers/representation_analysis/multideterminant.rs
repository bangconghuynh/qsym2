//! Driver for symmetry analysis of multi-determinantal wavefunctions.

use std::fmt;
use std::ops::Mul;

use anyhow::{self, bail, format_err};
use derive_builder::Builder;
use duplicate::duplicate_item;
use indexmap::IndexMap;
use itertools::Itertools;
use ndarray::{s, Array2, Array4};
use ndarray_linalg::types::Lapack;
use num_complex::{Complex, ComplexFloat};
use num_traits::Float;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::analysis::{
    log_overlap_eigenvalues, EigenvalueComparisonMode, Orbit, Overlap, ProjectionDecomposition,
    RepAnalysis,
};
use crate::angmom::spinor_rotation_3d::SpinConstraint;
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
    log_subtitle, nice_bool, qsym2_output, write_subtitle, write_title, QSym2Output,
};
use crate::symmetry::symmetry_element::symmetry_operation::{
    SpecialSymmetryTransformation, SymmetryOperation,
};
use crate::symmetry::symmetry_group::{
    MagneticRepresentedSymmetryGroup, SymmetryGroupProperties, UnitaryRepresentedSymmetryGroup,
};
use crate::symmetry::symmetry_symbols::{
    deduce_mirror_parities, MirrorParity, SymmetryClassSymbol,
};
use crate::symmetry::symmetry_transformation::SymmetryTransformationKind;
use crate::target::density::density_analysis::DensitySymmetryOrbit;
use crate::target::density::Density;
use crate::target::determinant::determinant_analysis::SlaterDeterminantSymmetryOrbit;
use crate::target::determinant::SlaterDeterminant;
use crate::target::noci::basis::Basis;
use crate::target::noci::multideterminant::MultiDeterminant;
use crate::target::orbital::orbital_analysis::generate_det_mo_orbits;

// #[cfg(test)]
// #[path = "slater_determinant_tests.rs"]
// mod slater_determinant_tests;

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
pub struct MultiDeterminantRepAnalysisResult<'a, G, T, B>
where
    G: SymmetryGroupProperties + Clone,
    G::CharTab: SubspaceDecomposable<T>,
    T: ComplexFloat + Lapack,
    <T as ComplexFloat>::Real: From<f64> + fmt::LowerExp + fmt::Debug,
    B: Basis<SlaterDeterminant<'a, T>> + Clone,
{
    /// The control parameters used to obtain this set of multi-determinantal wavefunction
    /// representation analysis results.
    parameters: &'a MultiDeterminantRepAnalysisParams<<T as ComplexFloat>::Real>,

    /// The multi-determinantal wavefunctions being analysed.
    multidets: Vec<&'a MultiDeterminant<'a, T, B>>,

    /// The group used for the representation analysis.
    group: G,

    /// The deduced symmetries of the multi-determinantal wavefunctions.
    multidet_symmetries:
        Option<Vec<Option<<G::CharTab as SubspaceDecomposable<T>>::Decomposition>>>,

    /// The overlap eigenvalues above and below the linear independence threshold for each
    /// multi-determinantal wavefunction symmetry deduction.
    multidet_symmetries_thresholds: Option<Vec<(Option<T>, Option<T>)>>,
}

impl<'a, G, T, B> MultiDeterminantRepAnalysisResult<'a, G, T, B>
where
    G: SymmetryGroupProperties + Clone,
    G::CharTab: SubspaceDecomposable<T>,
    T: ComplexFloat + Lapack,
    <T as ComplexFloat>::Real: From<f64> + fmt::LowerExp + fmt::Debug,
    B: Basis<SlaterDeterminant<'a, T>> + Clone,
{
    /// Returns a builder to construct a new [`MultiDeterminantRepAnalysisResultBuilder`]
    /// structure.
    fn builder() -> MultiDeterminantRepAnalysisResultBuilder<'a, G, T, B> {
        MultiDeterminantRepAnalysisResultBuilder::default()
    }
}

impl<'a, G, T, B> fmt::Display for MultiDeterminantRepAnalysisResult<'a, G, T, B>
where
    G: SymmetryGroupProperties + Clone,
    G::CharTab: SubspaceDecomposable<T>,
    T: ComplexFloat + Lapack,
    <T as ComplexFloat>::Real: From<f64> + fmt::LowerExp + fmt::Debug + fmt::Display,
    B: Basis<SlaterDeterminant<'a, T>> + Clone,
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

        let multidet_index_length = usize::try_from(self.multidets.len().ilog10() + 2).unwrap_or(4);
        let multidet_symmetry_length = self
            .multidet_symmetries
            .as_ref()
            .map(|multidet_symmetries| {
                multidet_symmetries
                    .iter()
                    .map(|multidet_sym| {
                        multidet_sym
                            .as_ref()
                            .map(|sym| sym.to_string())
                            .unwrap_or("--".to_string())
                            .chars()
                            .count()
                    })
                    .max()
                    .unwrap_or(0)
                    .max(8)
            })
            .unwrap_or(8);
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
            .as_ref()
            .map(|multidet_symmetries_thresholds| {
                multidet_symmetries_thresholds
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

        let multidet_eig_below_length: usize = self
            .multidet_symmetries_thresholds
            .as_ref()
            .map(|multidet_symmetries_thresholds| {
                multidet_symmetries_thresholds
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

        let table_width = 14
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
                .map(|multidet_0| multidet_0.spin_constraint().to_string().to_lowercase())
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
                .as_ref()
                .and_then(|syms| syms.get(multidet_i))
                .and_then(|sym_opt| sym_opt.as_ref().map(|sym| sym.to_string()))
                .unwrap_or("--".to_string());

            let (eig_above_str, eig_below_str) = self
                .multidet_symmetries_thresholds
                .as_ref()
                .and_then(|multidet_symmetries_thresholds| {
                    multidet_symmetries_thresholds.get(multidet_i)
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

        Ok(())
    }
}
//
// impl<'a, G, T> fmt::Debug for SlaterDeterminantRepAnalysisResult<'a, G, T>
// where
//     G: SymmetryGroupProperties + Clone,
//     G::CharTab: SubspaceDecomposable<T>,
//     T: ComplexFloat + Lapack,
//     <T as ComplexFloat>::Real: From<f64> + fmt::LowerExp + fmt::Debug + fmt::Display,
// {
//     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
//         writeln!(f, "{self}")
//     }
// }
//
// impl<'a, G, T> SlaterDeterminantRepAnalysisResult<'a, G, T>
// where
//     G: SymmetryGroupProperties + Clone,
//     G::CharTab: SubspaceDecomposable<T>,
//     T: ComplexFloat + Lapack,
//     <T as ComplexFloat>::Real: From<f64> + fmt::LowerExp + fmt::Debug + fmt::Display,
// {
//     /// Returns the determinant symmetry obtained from the analysis result.
//     pub fn determinant_symmetry(
//         &self,
//     ) -> &Result<<G::CharTab as SubspaceDecomposable<T>>::Decomposition, String> {
//         &self.determinant_symmetry
//     }
// }
//
// // ------
// // Driver
// // ------
//
// // ~~~~~~~~~~~~~~~~~
// // Struct definition
// // ~~~~~~~~~~~~~~~~~
//
// /// Driver structure for performing representation analysis on Slater determinants.
// #[derive(Clone, Builder)]
// #[builder(build_fn(validate = "Self::validate"))]
// pub struct SlaterDeterminantRepAnalysisDriver<'a, G, T>
// where
//     G: SymmetryGroupProperties + Clone,
//     G::CharTab: SubspaceDecomposable<T>,
//     T: ComplexFloat + Lapack,
//     <T as ComplexFloat>::Real: From<f64> + fmt::LowerExp + fmt::Debug,
// {
//     /// The control parameters for Slater determinant representation analysis.
//     parameters: &'a SlaterDeterminantRepAnalysisParams<<T as ComplexFloat>::Real>,
//
//     /// The Slater determinant to be analysed.
//     determinant: &'a SlaterDeterminant<'a, T>,
//
//     /// The result from symmetry-group detection on the underlying molecular structure of the
//     /// Slater determinant.
//     symmetry_group: &'a SymmetryGroupDetectionResult,
//
//     /// The atomic-orbital spatial overlap matrix of the underlying basis set used to describe the
//     /// determinant.
//     sao_spatial: &'a Array2<T>,
//
//     /// The complex-symmetric atomic-orbital spatial overlap matrix of the underlying basis set used
//     /// to describe the determinant. This is required if antiunitary symmetry operations are
//     /// involved. If none is provided, this will be assumed to be the same as [`Self::sao_spatial`].
//     #[builder(default = "None")]
//     sao_spatial_h: Option<&'a Array2<T>>,
//
//     /// The atomic-orbital four-centre spatial overlap matrix of the underlying basis set used to
//     /// describe the determinant. This is only required for density symmetry analysis.
//     #[builder(default = "None")]
//     sao_spatial_4c: Option<&'a Array4<T>>,
//
//     /// The complex-symmetric atomic-orbital four-centre spatial overlap matrix of the underlying
//     /// basis set used to describe the determinant. This is only required for density symmetry
//     /// analysis. This is required if antiunitary symmetry operations are involved. If none is
//     /// provided, this will be assumed to be the same as [`Self::sao_spatial_4c`], if any.
//     #[builder(default = "None")]
//     sao_spatial_4c_h: Option<&'a Array4<T>>,
//
//     /// The control parameters for symmetry analysis of angular functions.
//     angular_function_parameters: &'a AngularFunctionRepAnalysisParams,
//
//     /// The result of the Slater determinant representation analysis.
//     #[builder(setter(skip), default = "None")]
//     result: Option<SlaterDeterminantRepAnalysisResult<'a, G, T>>,
// }
//
// impl<'a, G, T> SlaterDeterminantRepAnalysisDriverBuilder<'a, G, T>
// where
//     G: SymmetryGroupProperties + Clone,
//     G::CharTab: SubspaceDecomposable<T>,
//     T: ComplexFloat + Lapack,
//     <T as ComplexFloat>::Real: From<f64> + fmt::LowerExp + fmt::Debug,
// {
//     fn validate(&self) -> Result<(), String> {
//         let params = self
//             .parameters
//             .ok_or("No Slater determinant representation analysis parameters found.".to_string())?;
//
//         let sym_res = self
//             .symmetry_group
//             .ok_or("No symmetry group information found.".to_string())?;
//
//         let sao_spatial = self
//             .sao_spatial
//             .ok_or("No spatial SAO matrix found.".to_string())?;
//
//         if let Some(sao_spatial_h) = self.sao_spatial_h.flatten() {
//             if sao_spatial_h.shape() != sao_spatial.shape() {
//                 return Err(
//                     "Mismatched shapes between `sao_spatial` and `sao_spatial_h`.".to_string(),
//                 );
//             }
//         }
//
//         match (
//             self.sao_spatial_4c.flatten(),
//             self.sao_spatial_4c_h.flatten(),
//         ) {
//             (Some(sao_spatial_4c), Some(sao_spatial_4c_h)) => {
//                 if sao_spatial_4c_h.shape() != sao_spatial_4c.shape() {
//                     return Err(
//                         "Mismatched shapes between `sao_spatial_4c` and `sao_spatial_4c_h`."
//                             .to_string(),
//                     );
//                 }
//             }
//             (None, Some(_)) => {
//                 return Err("`sao_spatial_4c_h` is provided without `sao_spatial_4c`.".to_string());
//             }
//             _ => {}
//         }
//
//         let det = self
//             .determinant
//             .ok_or("No Slater determinant found.".to_string())?;
//
//         let sym = if params.use_magnetic_group.is_some() {
//             sym_res
//                 .magnetic_symmetry
//                 .as_ref()
//                 .ok_or("Magnetic symmetry requested for representation analysis, but no magnetic symmetry found.")?
//         } else {
//             &sym_res.unitary_symmetry
//         };
//
//         if sym.is_infinite() && params.infinite_order_to_finite.is_none() {
//             Err(
//                 format!(
//                     "Representation analysis cannot be performed using the entirety of the infinite group `{}`. \
//                     Consider setting the parameter `infinite_order_to_finite` to restrict to a finite subgroup instead.",
//                     sym.group_name.as_ref().expect("No symmetry group name found.")
//                 )
//             )
//         } else if det.bao().n_funcs() != sao_spatial.nrows()
//             || det.bao().n_funcs() != sao_spatial.ncols()
//         {
//             Err("The dimensions of the spatial SAO matrix do not match the number of spatial AO basis functions.".to_string())
//         } else {
//             Ok(())
//         }
//     }
// }
//
// // ~~~~~~~~~~~~~~~~~~~~~~
// // Struct implementations
// // ~~~~~~~~~~~~~~~~~~~~~~
//
// // Generic for all symmetry groups G and determinant numeric type T
// // ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
//
// impl<'a, G, T> SlaterDeterminantRepAnalysisDriver<'a, G, T>
// where
//     G: SymmetryGroupProperties + Clone,
//     G::CharTab: SubspaceDecomposable<T>,
//     T: ComplexFloat + Lapack,
//     <T as ComplexFloat>::Real: From<f64> + fmt::LowerExp + fmt::Debug,
// {
//     /// Returns a builder to construct a [`SlaterDeterminantRepAnalysisDriver`] structure.
//     pub fn builder() -> SlaterDeterminantRepAnalysisDriverBuilder<'a, G, T> {
//         SlaterDeterminantRepAnalysisDriverBuilder::default()
//     }
//
//     /// Constructs the appropriate atomic-orbital overlap matrix based on the spin constraint of
//     /// the determinant.
//     fn construct_sao(&self) -> Result<(Array2<T>, Option<Array2<T>>), anyhow::Error> {
//         let sao = match self.determinant.spin_constraint() {
//             SpinConstraint::Restricted(_) | SpinConstraint::Unrestricted(_, _) => {
//                 self.sao_spatial.clone()
//             }
//             SpinConstraint::Generalised(nspins, _) => {
//                 let nspins_usize = usize::from(*nspins);
//                 let nspatial = self.sao_spatial.nrows();
//                 let mut sao_g = Array2::zeros((nspins_usize * nspatial, nspins_usize * nspatial));
//                 (0..nspins_usize).for_each(|ispin| {
//                     let start = ispin * nspatial;
//                     let end = (ispin + 1) * nspatial;
//                     sao_g
//                         .slice_mut(s![start..end, start..end])
//                         .assign(self.sao_spatial);
//                 });
//                 sao_g
//             }
//         };
//
//         let sao_h =
//             self.sao_spatial_h
//                 .map(|sao_spatial_h| match self.determinant.spin_constraint() {
//                     SpinConstraint::Restricted(_) | SpinConstraint::Unrestricted(_, _) => {
//                         sao_spatial_h.clone()
//                     }
//                     SpinConstraint::Generalised(nspins, _) => {
//                         let nspins_usize = usize::from(*nspins);
//                         let nspatial = sao_spatial_h.nrows();
//                         let mut sao_g =
//                             Array2::zeros((nspins_usize * nspatial, nspins_usize * nspatial));
//                         (0..nspins_usize).for_each(|ispin| {
//                             let start = ispin * nspatial;
//                             let end = (ispin + 1) * nspatial;
//                             sao_g
//                                 .slice_mut(s![start..end, start..end])
//                                 .assign(sao_spatial_h);
//                         });
//                         sao_g
//                     }
//                 });
//
//         Ok((sao, sao_h))
//     }
// }
//
// // Specific for unitary-represented symmetry groups, but generic for determinant numeric type T
// // ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
//
// impl<'a, T> SlaterDeterminantRepAnalysisDriver<'a, UnitaryRepresentedSymmetryGroup, T>
// where
//     T: ComplexFloat + Lapack + Sync + Send,
//     <T as ComplexFloat>::Real: From<f64> + fmt::LowerExp + fmt::Debug + Sync + Send,
//     for<'b> Complex<f64>: Mul<&'b T, Output = Complex<f64>>,
// {
//     fn_construct_unitary_group!(
//         /// Constructs the unitary-represented group (which itself can be unitary or magnetic) ready
//         /// for Slater determinant representation analysis.
//         construct_unitary_group
//     );
// }
//
// // Specific for magnetic-represented symmetry groups, but generic for determinant numeric type T
// // ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
//
// impl<'a, T> SlaterDeterminantRepAnalysisDriver<'a, MagneticRepresentedSymmetryGroup, T>
// where
//     T: ComplexFloat + Lapack + Sync + Send,
//     <T as ComplexFloat>::Real: From<f64> + Sync + Send + fmt::LowerExp + fmt::Debug,
//     for<'b> Complex<f64>: Mul<&'b T, Output = Complex<f64>>,
// {
//     fn_construct_magnetic_group!(
//         /// Constructs the magnetic-represented group (which itself can only be magnetic) ready for
//         /// Slater determinant corepresentation analysis.
//         construct_magnetic_group
//     );
// }
//
// // Specific for unitary-represented and magnetic-represented symmetry groups and determinant numeric types f64 and C128
// // ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
//
// #[duplicate_item(
//     duplicate!{
//         [ dtype_nested; [f64]; [Complex<f64>] ]
//         [
//             gtype_ [ UnitaryRepresentedSymmetryGroup ]
//             dtype_ [ dtype_nested ]
//             doc_sub_ [ "Performs representation analysis using a unitary-represented group and stores the result." ]
//             analyse_fn_ [ analyse_representation ]
//             construct_group_ [ self.construct_unitary_group()? ]
//             calc_projections_ [
//                 log_subtitle("Slater determinant projection decompositions");
//                 qsym2_output!("");
//                 qsym2_output!("  Projections are defined w.r.t. the following inner product:");
//                 qsym2_output!("    {}", det_orbit.origin().overlap_definition());
//                 qsym2_output!("");
//                 det_orbit
//                     .projections_to_string(
//                         &det_orbit.calc_projection_compositions()?,
//                         params.integrality_threshold,
//                     )
//                     .log_output_display();
//                 qsym2_output!("");
//             ]
//         ]
//     }
//     duplicate!{
//         [ dtype_nested; [f64]; [Complex<f64>] ]
//         [
//             gtype_ [ MagneticRepresentedSymmetryGroup ]
//             dtype_ [ dtype_nested ]
//             doc_sub_ [ "Performs corepresentation analysis using a magnetic-represented group and stores the result." ]
//             analyse_fn_ [ analyse_corepresentation ]
//             construct_group_ [ self.construct_magnetic_group()? ]
//             calc_projections_ [ ]
//         ]
//     }
// )]
// impl<'a> SlaterDeterminantRepAnalysisDriver<'a, gtype_, dtype_> {
//     #[doc = doc_sub_]
//     fn analyse_fn_(&mut self) -> Result<(), anyhow::Error> {
//         let params = self.parameters;
//         let (sao, sao_h) = self.construct_sao()?;
//         let group = construct_group_;
//         log_cc_transversal(&group);
//         let _ = find_angular_function_representation(&group, self.angular_function_parameters);
//         log_bao(self.determinant.bao());
//
//         // Determinant and orbital symmetries
//         let (det_symmetry, mo_symmetries, mo_mirror_parities, mo_symmetries_thresholds) = if params
//             .analyse_mo_symmetries
//         {
//             let mos = self.determinant.to_orbitals();
//             let (mut det_orbit, mut mo_orbitss) = generate_det_mo_orbits(
//                 self.determinant,
//                 &mos,
//                 &group,
//                 &sao,
//                 sao_h.as_ref(),
//                 params.integrality_threshold,
//                 params.linear_independence_threshold,
//                 params.symmetry_transformation_kind.clone(),
//                 params.eigenvalue_comparison_mode.clone(),
//                 params.use_cayley_table,
//             )?;
//             det_orbit.calc_xmat(false)?;
//             if params.write_overlap_eigenvalues {
//                 if let Some(smat_eigvals) = det_orbit.smat_eigvals.as_ref() {
//                     log_overlap_eigenvalues(
//                         "Determinant orbit overlap eigenvalues",
//                         smat_eigvals,
//                         params.linear_independence_threshold,
//                         &params.eigenvalue_comparison_mode,
//                     );
//                     qsym2_output!("");
//                 }
//             }
//
//             let det_symmetry = det_orbit.analyse_rep().map_err(|err| err.to_string());
//
//             { calc_projections_ }
//
//             let mo_symmetries = mo_orbitss
//                 .iter_mut()
//                 .map(|mo_orbits| {
//                     mo_orbits
//                         .par_iter_mut()
//                         .map(|mo_orbit| {
//                             mo_orbit.calc_xmat(false).ok()?;
//                             mo_orbit.analyse_rep().ok()
//                         })
//                         .collect::<Vec<_>>()
//                 })
//                 .collect::<Vec<_>>();
//
//             let mo_mirror_parities = if params.analyse_mo_mirror_parities {
//                 Some(
//                     mo_symmetries
//                         .iter()
//                         .map(|spin_mo_symmetries| {
//                             spin_mo_symmetries
//                                 .iter()
//                                 .map(|mo_sym_opt| {
//                                     mo_sym_opt.as_ref().map(|mo_sym| {
//                                         deduce_mirror_parities(det_orbit.group(), mo_sym)
//                                     })
//                                 })
//                                 .collect::<Vec<_>>()
//                         })
//                         .collect::<Vec<_>>(),
//                 )
//             } else {
//                 None
//             };
//
//             let mo_symmetries_thresholds = mo_orbitss
//                 .iter_mut()
//                 .map(|mo_orbits| {
//                     mo_orbits
//                         .par_iter_mut()
//                         .map(|mo_orbit| {
//                             mo_orbit
//                                 .smat_eigvals
//                                 .as_ref()
//                                 .map(|eigvals| {
//                                     let mut eigvals_vec = eigvals.iter().collect::<Vec<_>>();
//                                     match mo_orbit.eigenvalue_comparison_mode {
//                                         EigenvalueComparisonMode::Modulus => {
//                                             eigvals_vec.sort_by(|a, b| {
//                                                 a.abs().partial_cmp(&b.abs()).expect("Unable to compare two eigenvalues based on their moduli.")
//                                             });
//                                         }
//                                         EigenvalueComparisonMode::Real => {
//                                             eigvals_vec.sort_by(|a, b| {
//                                                 a.re().partial_cmp(&b.re()).expect("Unable to compare two eigenvalues based on their real parts.")
//                                             });
//                                         }
//                                     }
//                                     let eigval_above = match mo_orbit.eigenvalue_comparison_mode {
//                                         EigenvalueComparisonMode::Modulus => eigvals_vec
//                                             .iter()
//                                             .find(|val| {
//                                                 val.abs() >= mo_orbit.linear_independence_threshold
//                                             })
//                                             .copied()
//                                             .copied(),
//                                         EigenvalueComparisonMode::Real => eigvals_vec
//                                             .iter()
//                                             .find(|val| {
//                                                 val.re() >= mo_orbit.linear_independence_threshold
//                                             })
//                                             .copied()
//                                             .copied(),
//                                     };
//                                     eigvals_vec.reverse();
//                                     let eigval_below = match mo_orbit.eigenvalue_comparison_mode {
//                                         EigenvalueComparisonMode::Modulus => eigvals_vec
//                                             .iter()
//                                             .find(|val| {
//                                                 val.abs() < mo_orbit.linear_independence_threshold
//                                             })
//                                             .copied()
//                                             .copied(),
//                                         EigenvalueComparisonMode::Real => eigvals_vec
//                                             .iter()
//                                             .find(|val| {
//                                                 val.re() < mo_orbit.linear_independence_threshold
//                                             })
//                                             .copied()
//                                             .copied(),
//                                     };
//                                     (eigval_above, eigval_below)
//                                 })
//                                 .unwrap_or((None, None))
//                         })
//                         .collect::<Vec<_>>()
//                 })
//                 .collect::<Vec<_>>();
//             (
//                 det_symmetry,
//                 Some(mo_symmetries),
//                 mo_mirror_parities,
//                 Some(mo_symmetries_thresholds),
//             )
//         } else {
//             let mut det_orbit = SlaterDeterminantSymmetryOrbit::builder()
//                 .group(&group)
//                 .origin(self.determinant)
//                 .integrality_threshold(params.integrality_threshold)
//                 .linear_independence_threshold(params.linear_independence_threshold)
//                 .symmetry_transformation_kind(params.symmetry_transformation_kind.clone())
//                 .eigenvalue_comparison_mode(params.eigenvalue_comparison_mode.clone())
//                 .build()?;
//             let det_symmetry = det_orbit
//                 .calc_smat(Some(&sao), sao_h.as_ref(), params.use_cayley_table)
//                 .and_then(|det_orb| det_orb.normalise_smat())
//                 .map_err(|err| err.to_string())
//                 .and_then(|det_orb| {
//                     det_orb.calc_xmat(false).map_err(|err| err.to_string())?;
//                     if params.write_overlap_eigenvalues {
//                         if let Some(smat_eigvals) = det_orb.smat_eigvals.as_ref() {
//                             log_overlap_eigenvalues(
//                                 "Determinant orbit overlap eigenvalues",
//                                 smat_eigvals,
//                                 params.linear_independence_threshold,
//                                 &params.eigenvalue_comparison_mode,
//                             );
//                             qsym2_output!("");
//                         }
//                     }
//                     det_orb.analyse_rep().map_err(|err| err.to_string())
//                 });
//
//             { calc_projections_ }
//
//             (det_symmetry, None, None, None)
//         };
//
//         // Density and orbital density symmetries
//         let (den_symmetries, mo_den_symmetries) = if params.analyse_density_symmetries {
//             let den_syms = self.determinant.to_densities().map(|densities| {
//                 let mut spin_den_syms = densities
//                     .iter()
//                     .enumerate()
//                     .map(|(ispin, den)| {
//                         let den_sym_res = || {
//                             let mut den_orbit = DensitySymmetryOrbit::builder()
//                                 .group(&group)
//                                 .origin(den)
//                                 .integrality_threshold(params.integrality_threshold)
//                                 .linear_independence_threshold(params.linear_independence_threshold)
//                                 .symmetry_transformation_kind(
//                                     params.symmetry_transformation_kind.clone(),
//                                 )
//                                 .eigenvalue_comparison_mode(
//                                     params.eigenvalue_comparison_mode.clone(),
//                                 )
//                                 .build()?;
//                             den_orbit
//                                 .calc_smat(
//                                     self.sao_spatial_4c,
//                                     self.sao_spatial_4c_h,
//                                     params.use_cayley_table,
//                                 )?
//                                 .normalise_smat()?
//                                 .calc_xmat(false)?;
//                             den_orbit.analyse_rep().map_err(|err| format_err!(err))
//                         };
//                         (
//                             format!("Spin-{ispin} density"),
//                             den_sym_res().map_err(|err| err.to_string()),
//                         )
//                     })
//                     .collect::<Vec<_>>();
//                 let mut extra_den_syms = match self.determinant.spin_constraint() {
//                     SpinConstraint::Restricted(_) => {
//                         vec![("Total density".to_string(), spin_den_syms[0].1.clone())]
//                     }
//                     SpinConstraint::Unrestricted(nspins, _)
//                     | SpinConstraint::Generalised(nspins, _) => {
//                         let total_den_sym_res = || {
//                             let nspatial = self.determinant.bao().n_funcs();
//                             let zero_den = Density::<dtype_>::builder()
//                                 .density_matrix(Array2::<dtype_>::zeros((nspatial, nspatial)))
//                                 .bao(self.determinant.bao())
//                                 .mol(self.determinant.mol())
//                                 .complex_symmetric(self.determinant.complex_symmetric())
//                                 .threshold(self.determinant.threshold())
//                                 .build()?;
//                             let total_den =
//                                 densities.iter().fold(zero_den, |acc, denmat| acc + denmat);
//                             let mut total_den_orbit = DensitySymmetryOrbit::builder()
//                                 .group(&group)
//                                 .origin(&total_den)
//                                 .integrality_threshold(params.integrality_threshold)
//                                 .linear_independence_threshold(params.linear_independence_threshold)
//                                 .symmetry_transformation_kind(
//                                     params.symmetry_transformation_kind.clone(),
//                                 )
//                                 .eigenvalue_comparison_mode(
//                                     params.eigenvalue_comparison_mode.clone(),
//                                 )
//                                 .build()?;
//                             total_den_orbit
//                                 .calc_smat(
//                                     self.sao_spatial_4c,
//                                     self.sao_spatial_4c_h,
//                                     params.use_cayley_table,
//                                 )?
//                                 .calc_xmat(false)?;
//                             total_den_orbit
//                                 .analyse_rep()
//                                 .map_err(|err| format_err!(err))
//                         };
//                         let mut extra_syms = vec![(
//                             "Total density".to_string(),
//                             total_den_sym_res().map_err(|err| err.to_string()),
//                         )];
//                         extra_syms.extend((0..usize::from(*nspins)).combinations(2).map(
//                             |indices| {
//                                 let i = indices[0];
//                                 let j = indices[1];
//                                 let den_ij = &densities[i] - &densities[j];
//                                 let den_ij_sym_res = || {
//                                     let mut den_ij_orbit = DensitySymmetryOrbit::builder()
//                                         .group(&group)
//                                         .origin(&den_ij)
//                                         .integrality_threshold(params.integrality_threshold)
//                                         .linear_independence_threshold(
//                                             params.linear_independence_threshold,
//                                         )
//                                         .symmetry_transformation_kind(
//                                             params.symmetry_transformation_kind.clone(),
//                                         )
//                                         .eigenvalue_comparison_mode(
//                                             params.eigenvalue_comparison_mode.clone(),
//                                         )
//                                         .build()?;
//                                     den_ij_orbit
//                                         .calc_smat(
//                                             self.sao_spatial_4c,
//                                             self.sao_spatial_4c_h,
//                                             params.use_cayley_table,
//                                         )?
//                                         .calc_xmat(false)?;
//                                     den_ij_orbit.analyse_rep().map_err(|err| format_err!(err))
//                                 };
//                                 (
//                                     format!("Spin-polarised density {i} - {j}"),
//                                     den_ij_sym_res().map_err(|err| err.to_string()),
//                                 )
//                             },
//                         ));
//                         extra_syms
//                     }
//                 };
//                 spin_den_syms.append(&mut extra_den_syms);
//                 spin_den_syms
//             });
//
//             let mo_den_syms = if params.analyse_mo_symmetries {
//                 let mo_den_symmetries = self
//                     .determinant
//                     .to_orbitals()
//                     .iter()
//                     .map(|mos| {
//                         mos.par_iter()
//                             .map(|mo| {
//                                 let mo_den = mo.to_total_density().ok()?;
//                                 let mut mo_den_orbit = DensitySymmetryOrbit::builder()
//                                     .group(&group)
//                                     .origin(&mo_den)
//                                     .integrality_threshold(params.integrality_threshold)
//                                     .linear_independence_threshold(
//                                         params.linear_independence_threshold,
//                                     )
//                                     .symmetry_transformation_kind(
//                                         params.symmetry_transformation_kind.clone(),
//                                     )
//                                     .eigenvalue_comparison_mode(
//                                         params.eigenvalue_comparison_mode.clone(),
//                                     )
//                                     .build()
//                                     .ok()?;
//                                 log::debug!("Computing overlap matrix for an MO density orbit...");
//                                 mo_den_orbit
//                                     .calc_smat(
//                                         self.sao_spatial_4c,
//                                         self.sao_spatial_4c_h,
//                                         params.use_cayley_table,
//                                     )
//                                     .ok()?
//                                     .normalise_smat()
//                                     .ok()?
//                                     .calc_xmat(false)
//                                     .ok()?;
//                                 log::debug!(
//                                     "Computing overlap matrix for an MO density orbit... Done."
//                                 );
//                                 mo_den_orbit.analyse_rep().ok()
//                             })
//                             .collect::<Vec<_>>()
//                     })
//                     .collect::<Vec<_>>();
//                 Some(mo_den_symmetries)
//             } else {
//                 None
//             };
//
//             (den_syms.ok(), mo_den_syms)
//         } else {
//             (None, None)
//         };
//
//         let result = SlaterDeterminantRepAnalysisResult::builder()
//             .parameters(params)
//             .determinant(self.determinant)
//             .group(group)
//             .determinant_symmetry(det_symmetry)
//             .determinant_density_symmetries(den_symmetries)
//             .mo_symmetries(mo_symmetries)
//             .mo_mirror_parities(mo_mirror_parities)
//             .mo_symmetries_thresholds(mo_symmetries_thresholds)
//             .mo_density_symmetries(mo_den_symmetries)
//             .build()?;
//         self.result = Some(result);
//
//         Ok(())
//     }
// }
//
// // ~~~~~~~~~~~~~~~~~~~~~
// // Trait implementations
// // ~~~~~~~~~~~~~~~~~~~~~
//
// // Generic for all symmetry groups G and determinant numeric type T
// // ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
//
// impl<'a, G, T> fmt::Display for SlaterDeterminantRepAnalysisDriver<'a, G, T>
// where
//     G: SymmetryGroupProperties + Clone,
//     G::CharTab: SubspaceDecomposable<T>,
//     T: ComplexFloat + Lapack,
//     <T as ComplexFloat>::Real: From<f64> + fmt::LowerExp + fmt::Debug,
// {
//     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
//         write_title(f, "Slater Determinant Symmetry Analysis")?;
//         writeln!(f)?;
//         writeln!(f, "{}", self.parameters)?;
//         Ok(())
//     }
// }
//
// impl<'a, G, T> fmt::Debug for SlaterDeterminantRepAnalysisDriver<'a, G, T>
// where
//     G: SymmetryGroupProperties + Clone,
//     G::CharTab: SubspaceDecomposable<T>,
//     T: ComplexFloat + Lapack,
//     <T as ComplexFloat>::Real: From<f64> + fmt::LowerExp + fmt::Debug,
// {
//     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
//         writeln!(f, "{self}")
//     }
// }
//
// // Specific for unitary/magnetic-represented groups and determinant numeric type f64/Complex<f64>
// // ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
//
// #[duplicate_item(
//     duplicate!{
//         [ dtype_nested; [f64]; [Complex<f64>] ]
//         [
//             gtype_ [ UnitaryRepresentedSymmetryGroup ]
//             dtype_ [ dtype_nested ]
//             analyse_fn_ [ analyse_representation ]
//         ]
//     }
//     duplicate!{
//         [ dtype_nested; [f64]; [Complex<f64>] ]
//         [
//             gtype_ [ MagneticRepresentedSymmetryGroup ]
//             dtype_ [ dtype_nested ]
//             analyse_fn_ [ analyse_corepresentation ]
//         ]
//     }
// )]
// impl<'a> QSym2Driver for SlaterDeterminantRepAnalysisDriver<'a, gtype_, dtype_> {
//     type Params = SlaterDeterminantRepAnalysisParams<f64>;
//
//     type Outcome = SlaterDeterminantRepAnalysisResult<'a, gtype_, dtype_>;
//
//     fn result(&self) -> Result<&Self::Outcome, anyhow::Error> {
//         self.result.as_ref().ok_or_else(|| {
//             format_err!("No Slater determinant representation analysis results found.")
//         })
//     }
//
//     fn run(&mut self) -> Result<(), anyhow::Error> {
//         self.log_output_display();
//         self.analyse_fn_()?;
//         self.result()?.log_output_display();
//         Ok(())
//     }
// }
