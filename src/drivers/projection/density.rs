//! Driver for symmetry projection of electron densities.

use std::collections::HashSet;
use std::fmt;
use std::str::FromStr;

use anyhow::{bail, format_err};
use derive_builder::Builder;
use duplicate::duplicate_item;
use indexmap::IndexMap;
use ndarray_linalg::{Lapack, Norm};
use num::Complex;
use num_complex::ComplexFloat;
use serde::{Deserialize, Serialize, de::DeserializeOwned};

use crate::analysis::EigenvalueComparisonMode;
use crate::chartab::CharacterTable;
use crate::chartab::chartab_group::CharacterProperties;
use crate::drivers::QSym2Driver;
use crate::drivers::representation_analysis::{
    CharacterTableDisplay, MagneticSymmetryAnalysisKind, fn_construct_unitary_group, log_bao,
    log_cc_transversal,
};
use crate::drivers::symmetry_group_detection::SymmetryGroupDetectionResult;
use crate::group::{GroupProperties, UnitaryRepresentedGroup};
use crate::io::format::{
    QSym2Output, log_subtitle, nice_bool, qsym2_output, write_subtitle, write_title,
};
use crate::projection::Projectable;
use crate::symmetry::symmetry_group::{SymmetryGroupProperties, UnitaryRepresentedSymmetryGroup};
use crate::symmetry::symmetry_symbols::MullikenIrrepSymbol;
use crate::symmetry::symmetry_transformation::{SymmetryTransformable, SymmetryTransformationKind};
use crate::target::density::Density;
use crate::target::density::density_analysis::DensitySymmetryOrbit;

#[cfg(test)]
#[path = "density_tests.rs"]
mod density_tests;

// ----------
// Parameters
// ----------

const fn default_symbolic() -> Option<CharacterTableDisplay> {
    Some(CharacterTableDisplay::Symbolic)
}

/// Structure containing control parameters for electron density projection.
#[derive(Clone, Builder, Debug, Serialize, Deserialize)]
pub struct DensityProjectionParams {
    /// Option indicating if the magnetic group is to be used for projection, and if so,
    /// whether unitary representations or unitary-antiunitary corepresentations should be used.
    #[builder(default = "None")]
    #[serde(default)]
    pub use_magnetic_group: Option<MagneticSymmetryAnalysisKind>,

    /// Boolean indicating if the double group is to be used for symmetry projection.
    #[builder(default = "false")]
    #[serde(default)]
    pub use_double_group: bool,

    /// Option indicating if the character table of the group used for symmetry projection is to be
    /// printed out.
    #[builder(default = "Some(CharacterTableDisplay::Symbolic)")]
    #[serde(default = "default_symbolic")]
    pub write_character_table: Option<CharacterTableDisplay>,

    /// The kind of symmetry transformation to be applied on the reference density to generate
    /// the orbit for symmetry projection.
    #[builder(default = "SymmetryTransformationKind::Spatial")]
    #[serde(default)]
    pub symmetry_transformation_kind: SymmetryTransformationKind,

    /// The finite order to which any infinite-order symmetry element is reduced, so that a finite
    /// subgroup of an infinite group can be used for the symmetry projection.
    #[builder(default = "None")]
    #[serde(default)]
    pub infinite_order_to_finite: Option<u32>,

    /// The projection targets supplied symbolically.
    #[builder(default = "None")]
    #[serde(default)]
    pub symbolic_projection_targets: Option<Vec<String>>,

    /// The projection targets supplied numerically, where each value gives the index of the
    /// projection subspace based on the group's character table.
    #[builder(default = "None")]
    #[serde(default)]
    pub numeric_projection_targets: Option<Vec<usize>>,
}

impl DensityProjectionParams {
    /// Returns a builder to construct a [`DensityProjectionParams`] structure.
    pub fn builder() -> DensityProjectionParamsBuilder {
        DensityProjectionParamsBuilder::default()
    }
}

impl fmt::Display for DensityProjectionParams {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(symbolic_projection_targets) = self.symbolic_projection_targets.as_ref() {
            writeln!(f, "Projection subspaces (symbolic):")?;
            for projection_target in symbolic_projection_targets.iter() {
                writeln!(f, "  {projection_target}")?;
            }
            writeln!(f)?;
        }
        if let Some(numeric_projection_targets) = self.numeric_projection_targets.as_ref() {
            writeln!(f, "Projection subspaces (numeric):")?;
            for projection_target in numeric_projection_targets.iter() {
                writeln!(f, "  {projection_target}")?;
            }
            writeln!(f)?;
        }
        writeln!(
            f,
            "Use magnetic group for projection: {}",
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
            "Use double group for projection: {}",
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

/// Structure to contain electron density projection results.
#[derive(Clone, Builder)]
pub struct DensityProjectionResult<'a, G, T>
where
    G: SymmetryGroupProperties + Clone + 'a,
    T: ComplexFloat + Lapack,
    Density<'a, T>: SymmetryTransformable,
    DensitySymmetryOrbit<'a, G, T>: Projectable<G, Density<'a, T>>,
{
    /// The control parameters used to obtain this set of electron density projection results.
    parameters: &'a DensityProjectionParams,

    /// The densities being projected and their associated names or descriptions.
    densities: Vec<(String, &'a Density<'a, T>)>,

    /// The group used for the projection.
    group: G,

    /// The projected densities. Each tuple in the vector contains the name or description of the
    /// density being projected and an indexmap containing the projected densities indexed by the
    /// requested subspace labels.
    projected_densities: Vec<(
        String,
        IndexMap<G::RowSymbol, Result<Density<'a, T>, String>>,
    )>,
}

impl<'a, G, T> DensityProjectionResult<'a, G, T>
where
    G: SymmetryGroupProperties + Clone,
    G::RowSymbol: Serialize + DeserializeOwned,
    T: ComplexFloat + Lapack,
    Density<'a, T>: SymmetryTransformable,
    DensitySymmetryOrbit<'a, G, T>: Projectable<G, Density<'a, T>>,
{
    pub fn builder() -> DensityProjectionResultBuilder<'a, G, T> {
        DensityProjectionResultBuilder::default()
    }

    /// Returns the projected densities.
    pub fn projected_densities(
        &self,
    ) -> &Vec<(
        String,
        IndexMap<G::RowSymbol, Result<Density<'a, T>, String>>,
    )> {
        &self.projected_densities
    }
}

impl<'a, G, T> fmt::Display for DensityProjectionResult<'a, G, T>
where
    G: SymmetryGroupProperties + Clone,
    G::RowSymbol: Serialize + DeserializeOwned,
    T: ComplexFloat + Lapack,
    Density<'a, T>: SymmetryTransformable,
    DensitySymmetryOrbit<'a, G, T>: Projectable<G, Density<'a, T>>,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write_subtitle(f, "Orbit-based symmetry projection summary")?;
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

        for (name, projected_densities) in self.projected_densities.iter() {
            writeln!(f, ">> Density-matrix Frobenius norm for projected {name}")?;

            let (rows, norms): (Vec<String>, Vec<String>) = projected_densities
                .iter()
                .map(|(row, den_res)| {
                    let norm = den_res
                        .as_ref()
                        .map(|den| format!("{:.3e}", den.density_matrix().norm_l2()))
                        .unwrap_or_else(|err| err.to_string());
                    (row.to_string(), norm)
                })
                .unzip();

            let row_length = rows
                .iter()
                .map(|row| row.chars().count())
                .max()
                .unwrap_or(8)
                .max(8);
            let norm_length = norms
                .iter()
                .map(|norm| norm.chars().count())
                .max()
                .unwrap_or(14)
                .max(14);
            let table_width = 4 + row_length + norm_length;
            writeln!(f, "{}", "┈".repeat(table_width))?;
            writeln!(f, " {:<row_length$}  Frobenius norm", "Subspace",)?;
            writeln!(f, "{}", "┈".repeat(table_width))?;
            for (row, norm) in rows.iter().zip(norms) {
                writeln!(f, " {:<row_length$}  {:<}", row, norm)?;
            }
            writeln!(f, "{}", "┈".repeat(table_width))?;

            writeln!(f)?;
        }
        Ok(())
    }
}

impl<'a, G, T> fmt::Debug for DensityProjectionResult<'a, G, T>
where
    G: SymmetryGroupProperties + Clone,
    G::RowSymbol: Serialize + DeserializeOwned,
    T: ComplexFloat + Lapack,
    Density<'a, T>: SymmetryTransformable,
    DensitySymmetryOrbit<'a, G, T>: Projectable<G, Density<'a, T>>,
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
//
/// Driver structure for performing projection on electron densities.
#[derive(Clone, Builder)]
#[builder(build_fn(validate = "Self::validate"))]
pub struct DensityProjectionDriver<'a, G, T>
where
    G: SymmetryGroupProperties + Clone,
    G::RowSymbol: Serialize + DeserializeOwned,
    T: ComplexFloat + Lapack,
    Density<'a, T>: SymmetryTransformable,
    DensitySymmetryOrbit<'a, G, T>: Projectable<G, Density<'a, T>>,
{
    /// The control parameters used to obtain this set of electron density projection results.
    parameters: &'a DensityProjectionParams,

    /// The densities being projected and their associated names or descriptions.
    densities: Vec<(String, &'a Density<'a, T>)>,

    /// The result from symmetry-group detection on the underlying molecular structure of the
    /// electron densities. Only the unitary symmetry group will be used for projection, since
    /// magnetic-group projection is not yet formulated.
    symmetry_group: &'a SymmetryGroupDetectionResult,

    /// The result of the electron density projection.
    #[builder(setter(skip), default = "None")]
    result: Option<DensityProjectionResult<'a, G, T>>,
}

impl<'a, G, T> DensityProjectionDriverBuilder<'a, G, T>
where
    G: SymmetryGroupProperties + Clone,
    G::RowSymbol: Serialize + DeserializeOwned,
    T: ComplexFloat + Lapack,
    Density<'a, T>: SymmetryTransformable,
    DensitySymmetryOrbit<'a, G, T>: Projectable<G, Density<'a, T>>,
{
    fn validate(&self) -> Result<(), String> {
        let params = self
            .parameters
            .ok_or("No electron density projection parameters found.".to_string())?;

        let sym_res = self
            .symmetry_group
            .ok_or("No symmetry group information found.".to_string())?;

        let dens = self
            .densities
            .as_ref()
            .ok_or("No electron densities found.".to_string())?;

        let sym = if params.use_magnetic_group.is_some() {
            sym_res
                .magnetic_symmetry
                .as_ref()
                .ok_or("Magnetic symmetry requested for symmetry projection, but no magnetic symmetry found.")?
        } else {
            &sym_res.unitary_symmetry
        };

        if sym.is_infinite() && params.infinite_order_to_finite.is_none() {
            Err(format!(
                "Projection cannot be performed using the entirety of the infinite group `{}`. \
                    Consider setting the parameter `infinite_order_to_finite` to restrict to a finite subgroup instead.",
                sym.group_name
                    .as_ref()
                    .expect("No symmetry group name found.")
            ))
        } else {
            let baos = dens
                .iter()
                .map(|(_, den)| den.bao())
                .collect::<HashSet<_>>();
            if baos.len() != 1 {
                Err("Inconsistent basis angular order information between densities.".to_string())
            } else {
                Ok(())
            }
        }
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~
// Struct implementations
// ~~~~~~~~~~~~~~~~~~~~~~

// Generic for all symmetry groups G and density numeric type T
// ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

impl<'a, G, T> DensityProjectionDriver<'a, G, T>
where
    G: SymmetryGroupProperties + Clone,
    G::RowSymbol: Serialize + DeserializeOwned,
    T: ComplexFloat + Lapack,
    Density<'a, T>: SymmetryTransformable,
    DensitySymmetryOrbit<'a, G, T>: Projectable<G, Density<'a, T>>,
{
    /// Returns a builder to construct a [`DensityProjectionDriver`] structure.
    pub fn builder() -> DensityProjectionDriverBuilder<'a, G, T> {
        DensityProjectionDriverBuilder::default()
    }
}

// Specific for unitary-represented symmetry groups, but generic for density numeric type T
// ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

impl<'a, T> DensityProjectionDriver<'a, UnitaryRepresentedSymmetryGroup, T>
where
    T: ComplexFloat + Lapack,
    Density<'a, T>: SymmetryTransformable,
    DensitySymmetryOrbit<'a, UnitaryRepresentedSymmetryGroup, T>:
        Projectable<UnitaryRepresentedSymmetryGroup, Density<'a, T>>,
{
    fn_construct_unitary_group!(
        /// Constructs the unitary-represented group (which itself can be unitary or magnetic) ready
        /// for electron density projection.
        construct_unitary_group
    );
}

// Specific for unitary-represented symmetry groups and density numeric types f64 and C128
// '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

#[duplicate_item(
    duplicate!{
        [ dtype_nested; [f64]; [Complex<f64>] ]
        [
            gtype_ [ UnitaryRepresentedSymmetryGroup ]
            dtype_ [ dtype_nested ]
            doc_sub_ [ "Performs projection using a unitary-represented group and stores the result." ]
            projection_fn_ [ projection_representation ]
            construct_group_ [ self.construct_unitary_group()? ]
        ]
    }
)]
impl<'a> DensityProjectionDriver<'a, gtype_, dtype_> {
    #[doc = doc_sub_]
    fn projection_fn_(&mut self) -> Result<(), anyhow::Error> {
        let params = self.parameters;
        let group = construct_group_;
        log_cc_transversal(&group);
        let bao = self
            .densities
            .iter()
            .next()
            .map(|(_, den)| den.bao())
            .ok_or_else(|| {
                format_err!("Basis angular order information could not be extracted.")
            })?;
        log_bao(bao, None);

        let all_rows = group.character_table().get_all_rows();
        let rows = params
            .symbolic_projection_targets
            .as_ref()
            .unwrap_or(&vec![])
            .iter()
            .map(|row_str| MullikenIrrepSymbol::from_str(row_str).map_err(|err| format_err!(err)))
            .chain(
                params
                    .numeric_projection_targets
                    .as_ref()
                    .unwrap_or(&vec![])
                    .iter()
                    .map(|row_index| {
                        all_rows.get_index(*row_index).cloned().ok_or_else(|| {
                            format_err!(
                                "Unable to retrieve the subspace label with index {row_index}."
                            )
                        })
                    }),
            )
            .collect::<Result<Vec<_>, _>>()?;

        let projected_densities = self
            .densities
            .iter()
            .map(|(name, den)| {
                let projections = DensitySymmetryOrbit::builder()
                    .group(&group)
                    .origin(den)
                    .symmetry_transformation_kind(params.symmetry_transformation_kind.clone())
                    .integrality_threshold(1e-14)
                    .linear_independence_threshold(1e-14)
                    .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
                    .build()
                    .map_err(|err| format_err!(err))
                    .map(|den_orbit| {
                        rows.iter()
                            .map(|row| {
                                (
                                    row.clone(),
                                    den_orbit
                                        .project_onto(row)
                                        .map_err(|err| err.to_string())
                                        .and_then(|temp_projected_den| {
                                            Density::builder()
                                                .bao(bao)
                                                .complex_symmetric(den.complex_symmetric())
                                                .complex_conjugated(den.complex_conjugated())
                                                .mol(den.mol())
                                                .density_matrix(
                                                    temp_projected_den.density_matrix().clone(),
                                                )
                                                .threshold(den.threshold())
                                                .build()
                                                .map_err(|err| err.to_string())
                                        }),
                                )
                            })
                            .collect::<IndexMap<_, _>>()
                    })?;
                Ok((name.clone(), projections))
            })
            .collect::<Result<Vec<_>, anyhow::Error>>()?;

        let result = DensityProjectionResult::builder()
            .parameters(params)
            .densities(self.densities.clone())
            .group(group.clone())
            .projected_densities(projected_densities)
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

impl<'a, G, T> fmt::Display for DensityProjectionDriver<'a, G, T>
where
    G: SymmetryGroupProperties + Clone,
    G::RowSymbol: Serialize + DeserializeOwned,
    T: ComplexFloat + Lapack,
    Density<'a, T>: SymmetryTransformable,
    DensitySymmetryOrbit<'a, G, T>: Projectable<G, Density<'a, T>>,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write_title(f, "Electron Density Symmetry Projection")?;
        writeln!(f)?;
        writeln!(f, "{}", self.parameters)?;
        Ok(())
    }
}

impl<'a, G, T> fmt::Debug for DensityProjectionDriver<'a, G, T>
where
    G: SymmetryGroupProperties + Clone,
    G::RowSymbol: Serialize + DeserializeOwned,
    T: ComplexFloat + Lapack,
    Density<'a, T>: SymmetryTransformable,
    DensitySymmetryOrbit<'a, G, T>: Projectable<G, Density<'a, T>>,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{self}")
    }
}

// Specific for unitary-represented symmetry groups and density numeric types f64 and C128
// '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

#[duplicate_item(
    duplicate!{
        [ dtype_nested; [f64]; [Complex<f64>] ]
        [
            gtype_ [ UnitaryRepresentedSymmetryGroup ]
            dtype_ [ dtype_nested ]
            doc_sub_ [ "Performs projection using a unitary-represented group and stores the result." ]
            projection_fn_ [ projection_representation ]
            construct_group_ [ self.construct_unitary_group()? ]
        ]
    }
)]
impl<'a> QSym2Driver for DensityProjectionDriver<'a, gtype_, dtype_> {
    type Params = DensityProjectionParams;

    type Outcome = DensityProjectionResult<'a, gtype_, dtype_>;

    fn result(&self) -> Result<&Self::Outcome, anyhow::Error> {
        self.result
            .as_ref()
            .ok_or_else(|| format_err!("No electron density projection results found."))
    }

    fn run(&mut self) -> Result<(), anyhow::Error> {
        self.log_output_display();
        self.projection_fn_()?;
        self.result()?.log_output_display();
        Ok(())
    }
}
