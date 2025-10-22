//! Driver for symmetry projection of electron densities.

use std::fmt;
use std::hash::Hash;
use std::str::FromStr;

use anyhow::{bail, format_err};
use derive_builder::Builder;
use duplicate::duplicate_item;
use indexmap::IndexMap;
use ndarray_linalg::Lapack;
use num::Complex;
use num_complex::ComplexFloat;
use serde::{Deserialize, Serialize, de::DeserializeOwned};

use crate::analysis::EigenvalueComparisonMode;
use crate::angmom::spinor_rotation_3d::{SpinConstraint, SpinOrbitCoupled, StructureConstraint};
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
use crate::target::determinant::SlaterDeterminant;
use crate::target::determinant::determinant_analysis::SlaterDeterminantSymmetryOrbit;
use crate::target::noci::basis::{Basis, EagerBasis};
use crate::target::noci::multideterminant::MultiDeterminant;

#[cfg(test)]
#[path = "slater_determinant_tests.rs"]
mod slater_determinant_tests;

// ----------
// Parameters
// ----------

const fn default_symbolic() -> Option<CharacterTableDisplay> {
    Some(CharacterTableDisplay::Symbolic)
}

/// Structure containing control parameters for Slater determinant projection.
#[derive(Clone, Builder, Debug, Serialize, Deserialize)]
pub struct SlaterDeterminantProjectionParams {
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

    /// The kind of symmetry transformation to be applied on the reference Slater determinant to
    /// generate the orbit for symmetry projection.
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

impl SlaterDeterminantProjectionParams {
    /// Returns a builder to construct a [`SlaterDeterminantProjectionParams`] structure.
    pub fn builder() -> SlaterDeterminantProjectionParamsBuilder {
        SlaterDeterminantProjectionParamsBuilder::default()
    }
}

impl fmt::Display for SlaterDeterminantProjectionParams {
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

/// Structure to contain Slater determinant projection results.
#[derive(Clone, Builder)]
pub struct SlaterDeterminantProjectionResult<'a, G, T, SC>
where
    G: SymmetryGroupProperties + Clone + 'a,
    T: ComplexFloat + Lapack,
    SC: StructureConstraint + Hash + Eq + fmt::Display + 'a,
    SlaterDeterminant<'a, T, SC>: SymmetryTransformable,
    SlaterDeterminantSymmetryOrbit<'a, G, T, SC>: Projectable<G, SlaterDeterminant<'a, T, SC>>,
{
    /// The control parameters used to obtain this set of electron density projection results.
    parameters: &'a SlaterDeterminantProjectionParams,

    /// The Slater determinant being projected.
    slater_determinant: &'a SlaterDeterminant<'a, T, SC>,

    /// The group used for the projection.
    group: G,

    /// The projected Slater determinants given as an indexmap containing the projected Slater
    /// determinant indexed by the requested subspace labels.
    projected_slater_determinants: IndexMap<
        G::RowSymbol,
        Result<MultiDeterminant<'a, T, EagerBasis<SlaterDeterminant<'a, T, SC>>, SC>, String>,
    >,
}

impl<'a, G, T, SC> SlaterDeterminantProjectionResult<'a, G, T, SC>
where
    G: SymmetryGroupProperties + Clone,
    G::RowSymbol: Serialize + DeserializeOwned,
    T: ComplexFloat + Lapack,
    SC: StructureConstraint + Hash + Eq + Clone + fmt::Display,
    SlaterDeterminant<'a, T, SC>: SymmetryTransformable,
    SlaterDeterminantSymmetryOrbit<'a, G, T, SC>: Projectable<G, SlaterDeterminant<'a, T, SC>>,
{
    pub fn builder() -> SlaterDeterminantProjectionResultBuilder<'a, G, T, SC> {
        SlaterDeterminantProjectionResultBuilder::default()
    }

    /// Returns the projected densities.
    pub fn projected_slater_determinants(
        &self,
    ) -> &IndexMap<
        G::RowSymbol,
        Result<MultiDeterminant<'a, T, EagerBasis<SlaterDeterminant<'a, T, SC>>, SC>, String>,
    > {
        &self.projected_slater_determinants
    }
}

impl<'a, G, T, SC> fmt::Display for SlaterDeterminantProjectionResult<'a, G, T, SC>
where
    G: SymmetryGroupProperties + Clone,
    T: ComplexFloat + Lapack,
    SC: StructureConstraint + Hash + Eq + fmt::Display,
    SlaterDeterminant<'a, T, SC>: SymmetryTransformable,
    SlaterDeterminantSymmetryOrbit<'a, G, T, SC>: Projectable<G, SlaterDeterminant<'a, T, SC>>,
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

        // for (name, projected_densities) in self.projected_densities.iter() {
        //     writeln!(f, ">> Density-matrix Frobenius norm for projected {name}")?;
        //
        //     let (rows, norms): (Vec<String>, Vec<String>) = projected_densities
        //         .iter()
        //         .map(|(row, den_res)| {
        //             let norm = den_res
        //                 .as_ref()
        //                 .map(|den| format!("{:.3e}", den.density_matrix().norm_l2()))
        //                 .unwrap_or_else(|err| err.to_string());
        //             (row.to_string(), norm)
        //         })
        //         .unzip();
        //
        //     let row_length = rows
        //         .iter()
        //         .map(|row| row.chars().count())
        //         .max()
        //         .unwrap_or(7)
        //         .max(7);
        //     let norm_length = norms
        //         .iter()
        //         .map(|norm| norm.chars().count())
        //         .max()
        //         .unwrap_or(14)
        //         .max(14);
        //     let table_width = 4 + row_length + norm_length;
        //     writeln!(f, "{}", "┈".repeat(table_width))?;
        //     writeln!(f, " {:<row_length$}  Frobenius norm", "Subspace",)?;
        //     writeln!(f, "{}", "┈".repeat(table_width))?;
        //     for (row, norm) in rows.iter().zip(norms) {
        //         writeln!(f, " {:<row_length$}  {:<}", row, norm)?;
        //     }
        //     writeln!(f, "{}", "┈".repeat(table_width))?;
        //
        //     writeln!(f)?;
        // }
        Ok(())
    }
}

impl<'a, G, T, SC> fmt::Debug for SlaterDeterminantProjectionResult<'a, G, T, SC>
where
    G: SymmetryGroupProperties + Clone,
    T: ComplexFloat + Lapack,
    SC: StructureConstraint + Hash + Eq + fmt::Display,
    SlaterDeterminant<'a, T, SC>: SymmetryTransformable,
    SlaterDeterminantSymmetryOrbit<'a, G, T, SC>: Projectable<G, SlaterDeterminant<'a, T, SC>>,
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
/// Driver structure for performing projection on Slater determinants.
#[derive(Clone, Builder)]
#[builder(build_fn(validate = "Self::validate"))]
pub struct SlaterDeterminantProjectionDriver<'a, G, T, SC>
where
    G: SymmetryGroupProperties + Clone,
    G::RowSymbol: Serialize + DeserializeOwned,
    T: ComplexFloat + Lapack,
    SC: StructureConstraint + Hash + Eq + fmt::Display,
    SlaterDeterminant<'a, T, SC>: SymmetryTransformable,
    SlaterDeterminantSymmetryOrbit<'a, G, T, SC>: Projectable<G, SlaterDeterminant<'a, T, SC>>,
{
    /// The control parameters used to obtain this set of Slater determinant projection results.
    parameters: &'a SlaterDeterminantProjectionParams,

    /// The Slater determinant being projected.
    slater_determinant: &'a SlaterDeterminant<'a, T, SC>,

    /// The result from symmetry-group detection on the underlying molecular structure of the
    /// Slater determinant. Only the unitary symmetry group will be used for projection, since
    /// magnetic-group projection is not yet formulated.
    symmetry_group: &'a SymmetryGroupDetectionResult,

    /// The result of the Slater determinant projection.
    #[builder(setter(skip), default = "None")]
    result: Option<SlaterDeterminantProjectionResult<'a, G, T, SC>>,
}

impl<'a, G, T, SC> SlaterDeterminantProjectionDriverBuilder<'a, G, T, SC>
where
    G: SymmetryGroupProperties + Clone,
    G::RowSymbol: Serialize + DeserializeOwned,
    T: ComplexFloat + Lapack,
    SC: StructureConstraint + Hash + Eq + fmt::Display,
    SlaterDeterminant<'a, T, SC>: SymmetryTransformable,
    SlaterDeterminantSymmetryOrbit<'a, G, T, SC>: Projectable<G, SlaterDeterminant<'a, T, SC>>,
{
    fn validate(&self) -> Result<(), String> {
        let params = self
            .parameters
            .ok_or("No Slater determinant projection parameters found.".to_string())?;

        let sym_res = self
            .symmetry_group
            .ok_or("No symmetry group information found.".to_string())?;

        let _sd = self
            .slater_determinant
            .as_ref()
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
                "Projection cannot be performed using the entirety of the infinite group `{}`. \
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

// Generic for all symmetry groups G and density numeric type T
// ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

impl<'a, G, T, SC> SlaterDeterminantProjectionDriver<'a, G, T, SC>
where
    G: SymmetryGroupProperties + Clone,
    G::RowSymbol: Serialize + DeserializeOwned,
    T: ComplexFloat + Lapack,
    SC: StructureConstraint + Hash + Eq + Clone + fmt::Display,
    SlaterDeterminant<'a, T, SC>: SymmetryTransformable,
    SlaterDeterminantSymmetryOrbit<'a, G, T, SC>: Projectable<G, SlaterDeterminant<'a, T, SC>>,
{
    /// Returns a builder to construct a [`SlaterDeterminantProjectionDriver`] structure.
    pub fn builder() -> SlaterDeterminantProjectionDriverBuilder<'a, G, T, SC> {
        SlaterDeterminantProjectionDriverBuilder::default()
    }
}

// Specific for unitary-represented symmetry groups, but generic for density numeric type T
// ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

impl<'a, T, SC> SlaterDeterminantProjectionDriver<'a, UnitaryRepresentedSymmetryGroup, T, SC>
where
    T: ComplexFloat + Lapack,
    SC: StructureConstraint + Hash + Eq + Clone + fmt::Display,
    SlaterDeterminant<'a, T, SC>: SymmetryTransformable,
    SlaterDeterminantSymmetryOrbit<'a, UnitaryRepresentedSymmetryGroup, T, SC>:
        Projectable<UnitaryRepresentedSymmetryGroup, SlaterDeterminant<'a, T, SC>>,
{
    fn_construct_unitary_group!(
        /// Constructs the unitary-represented group (which itself can be unitary or magnetic) ready
        /// for Slater determinant projection.
        construct_unitary_group
    );
}

// Specific for unitary-represented symmetry groups and density numeric types f64 and C128
// '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

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
            doc_sub_ [ "Performs projection using a unitary-represented group and stores the result." ]
            projection_fn_ [ projection_representation ]
            construct_group_ [ self.construct_unitary_group()? ]
        ]
    }
)]
impl<'a> SlaterDeterminantProjectionDriver<'a, gtype_, dtype_, sctype_> {
    #[doc = doc_sub_]
    fn projection_fn_(&mut self) -> Result<(), anyhow::Error> {
        let params = self.parameters;
        let group = construct_group_;
        let original_sd = self.slater_determinant;
        log_cc_transversal(&group);
        let baos = original_sd.baos();
        for (bao_i, bao) in baos.iter().enumerate() {
            log_bao(bao, Some(bao_i));
        }

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

        let projected_slater_determinants = SlaterDeterminantSymmetryOrbit::builder()
            .group(&group)
            .origin(&original_sd)
            .symmetry_transformation_kind(params.symmetry_transformation_kind.clone())
            .integrality_threshold(1e-14)
            .linear_independence_threshold(1e-14)
            .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
            .build()
            .map_err(|err| format_err!(err))
            .and_then(|sd_orbit| {
                rows.iter()
                    .map(|row| {
                        sd_orbit.project_onto(row).and_then(|temp_projected_sd| {
                            let basis_dets = temp_projected_sd
                                .basis()
                                .iter()
                                .map(|det_res| {
                                    det_res.and_then(|det| {
                                        SlaterDeterminant::builder()
                                            .structure_constraint(
                                                det.structure_constraint().clone(),
                                            )
                                            .baos(baos.clone())
                                            .complex_symmetric(det.complex_symmetric())
                                            .complex_conjugated(det.complex_conjugated())
                                            .mol(original_sd.mol())
                                            .coefficients(&det.coefficients().clone())
                                            .threshold(original_sd.threshold())
                                            .build()
                                            .map_err(|err| format_err!(err))
                                    })
                                })
                                .collect::<Result<Vec<_>, _>>()?;
                            let basis = EagerBasis::builder()
                                .elements(basis_dets)
                                .build()
                                .map_err(|err| format_err!(err))?;
                            Ok((
                                row.clone(),
                                MultiDeterminant::builder()
                                    .basis(basis)
                                    .complex_conjugated(temp_projected_sd.complex_conjugated())
                                    .coefficients(temp_projected_sd.coefficients().clone())
                                    .threshold(temp_projected_sd.threshold())
                                    .build()
                                    .map_err(|err| err.to_string()),
                            ))
                        })
                    })
                    .collect::<Result<IndexMap<_, _>, _>>()
            })?;

        let result = SlaterDeterminantProjectionResult::builder()
            .parameters(params)
            .slater_determinant(&self.slater_determinant)
            .group(group.clone())
            .projected_slater_determinants(projected_slater_determinants)
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

impl<'a, G, T, SC> fmt::Display for SlaterDeterminantProjectionDriver<'a, G, T, SC>
where
    G: SymmetryGroupProperties + Clone,
    G::RowSymbol: Serialize + DeserializeOwned,
    T: ComplexFloat + Lapack,
    SC: StructureConstraint + Hash + Eq + fmt::Display,
    SlaterDeterminant<'a, T, SC>: SymmetryTransformable,
    SlaterDeterminantSymmetryOrbit<'a, G, T, SC>: Projectable<G, SlaterDeterminant<'a, T, SC>>,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write_title(f, "Slater Determinant Symmetry Projection")?;
        writeln!(f)?;
        writeln!(f, "{}", self.parameters)?;
        Ok(())
    }
}

impl<'a, G, T, SC> fmt::Debug for SlaterDeterminantProjectionDriver<'a, G, T, SC>
where
    G: SymmetryGroupProperties + Clone,
    G::RowSymbol: Serialize + DeserializeOwned,
    T: ComplexFloat + Lapack,
    SC: StructureConstraint + Hash + Eq + fmt::Display,
    SlaterDeterminant<'a, T, SC>: SymmetryTransformable,
    SlaterDeterminantSymmetryOrbit<'a, G, T, SC>: Projectable<G, SlaterDeterminant<'a, T, SC>>,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{self}")
    }
}

// Specific for unitary-represented symmetry groups and density numeric types f64 and C128
// '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

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
            doc_sub_ [ "Performs projection using a unitary-represented group and stores the result." ]
            projection_fn_ [ projection_representation ]
            construct_group_ [ self.construct_unitary_group()? ]
        ]
    }
)]
impl<'a> QSym2Driver for SlaterDeterminantProjectionDriver<'a, gtype_, dtype_, sctype_> {
    type Params = SlaterDeterminantProjectionParams;

    type Outcome = SlaterDeterminantProjectionResult<'a, gtype_, dtype_, sctype_>;

    fn result(&self) -> Result<&Self::Outcome, anyhow::Error> {
        self.result
            .as_ref()
            .ok_or_else(|| format_err!("No Slater determinant projection results found."))
    }

    fn run(&mut self) -> Result<(), anyhow::Error> {
        self.log_output_display();
        self.projection_fn_()?;
        self.result()?.log_output_display();
        Ok(())
    }
}
