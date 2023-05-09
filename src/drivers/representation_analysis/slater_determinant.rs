use std::fmt;

use anyhow::{self, format_err};
use derive_builder::Builder;
use ndarray::{s, Array2};
use ndarray_linalg::types::Lapack;
use num_complex::{Complex, ComplexFloat};

use crate::analysis::RepAnalysis;
use crate::angmom::spinor_rotation_3d::SpinConstraint;
use crate::aux::format::{nice_bool, write_subtitle, write_title};
use crate::chartab::chartab_symbols::{DecomposedSymbol, ReducibleLinearSpaceSymbol};
use crate::drivers::symmetry_group_detection::SymmetryGroupDetectionResult;
use crate::drivers::{QSym2Driver, QSym2Output};
use crate::group::{GroupProperties, MagneticRepresentedGroup, UnitaryRepresentedGroup};
use crate::symmetry::symmetry_group::{
    MagneticRepresentedSymmetryGroup, SymmetryGroupProperties, UnitaryRepresentedSymmetryGroup,
};
use crate::symmetry::symmetry_symbols::{MullikenIrcorepSymbol, MullikenIrrepSymbol};
use crate::symmetry::symmetry_transformation::SymmetryTransformationKind;
use crate::target::determinant::determinant_analysis::SlaterDeterminantSymmetryOrbit;
use crate::target::determinant::SlaterDeterminant;
use crate::target::orbital::orbital_analysis::MolecularOrbitalSymmetryOrbit;

#[cfg(test)]
#[path = "slater_determinant_tests.rs"]
mod slater_determinant_tests;

// ==================
// Struct definitions
// ==================

// ----------
// Parameters
// ----------

/// A structure containing control parameters for Slater determinant representation analysis.
#[derive(Clone, Builder, Debug)]
pub struct SlaterDeterminantRepAnalysisParams<T>
where
    T: ComplexFloat + Lapack,
    <T as ComplexFloat>::Real: fmt::LowerExp + fmt::Debug,
{
    integrality_threshold: <T as ComplexFloat>::Real,

    linear_independence_threshold: <T as ComplexFloat>::Real,

    analyse_mo_symmetries: bool,

    use_magnetic_group: bool,

    use_double_group: bool,

    symmetry_transformation_kind: SymmetryTransformationKind,

    /// The finite order to which any infinite-order symmetry element is reduced, so that a finite
    /// subgroup of an infinite group can be used for the representation analysis.
    #[builder(default = "None")]
    infinite_order_to_finite: Option<u32>,
}

impl<T> SlaterDeterminantRepAnalysisParams<T>
where
    T: ComplexFloat + Lapack,
    <T as ComplexFloat>::Real: fmt::LowerExp + fmt::Debug,
{
    /// Returns a builder to construct a [`SlaterDeterminantRepAnalysisParams`] structure.
    pub fn builder() -> SlaterDeterminantRepAnalysisParamsBuilder<T> {
        SlaterDeterminantRepAnalysisParamsBuilder::default()
    }
}

impl<T> fmt::Display for SlaterDeterminantRepAnalysisParams<T>
where
    T: ComplexFloat + Lapack,
    <T as ComplexFloat>::Real: fmt::LowerExp + fmt::Debug,
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
        writeln!(f, "")?;
        writeln!(
            f,
            "Analyse molecular orbital symmetry: {}",
            nice_bool(self.analyse_mo_symmetries)
        )?;
        writeln!(f, "")?;
        writeln!(
            f,
            "Use magnetic group for analysis: {}",
            nice_bool(self.use_magnetic_group)
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

        Ok(())
    }
}

// ------
// Result
// ------

/// A structure to contain Slater determinant representation analysis results.
#[derive(Clone, Builder)]
pub struct SlaterDeterminantRepAnalysisResult<'a, R, T>
where
    R: ReducibleLinearSpaceSymbol,
    T: ComplexFloat + Lapack,
    <T as ComplexFloat>::Real: fmt::LowerExp + fmt::Debug,
{
    /// The control parameters used to obtain this set of Slater determinant representation
    /// analysis results.
    parameters: &'a SlaterDeterminantRepAnalysisParams<T>,

    determinant: &'a SlaterDeterminant<'a, T>,

    determinant_symmetry: Option<R>,

    mo_symmetries: Option<Vec<Vec<Option<R>>>>,
}

impl<'a, R, T> SlaterDeterminantRepAnalysisResult<'a, R, T>
where
    R: ReducibleLinearSpaceSymbol,
    T: ComplexFloat + Lapack,
    <T as ComplexFloat>::Real: fmt::LowerExp + fmt::Debug,
{
    fn builder() -> SlaterDeterminantRepAnalysisResultBuilder<'a, R, T> {
        SlaterDeterminantRepAnalysisResultBuilder::default()
    }
}

impl<'a, R, T> fmt::Display for SlaterDeterminantRepAnalysisResult<'a, R, T>
where
    R: ReducibleLinearSpaceSymbol,
    T: ComplexFloat + Lapack,
    <T as ComplexFloat>::Real: fmt::LowerExp + fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write_subtitle(f, "Orbit-based symmetry analysis results")?;
        writeln!(f, "")?;
        writeln!(f, "> Overall determinantal result")?;
        writeln!(
            f,
            "  Energy  : {}",
            self.determinant
                .energy()
                .as_ref()
                .map(|e| e.to_string())
                .unwrap_or("--".to_string())
        )?;
        writeln!(
            f,
            "  Symmetry: {}",
            self.determinant_symmetry
                .as_ref()
                .map(|s| s.to_string())
                .unwrap_or("--".to_string())
        )?;
        writeln!(f, "")?;

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
            writeln!(f, "> Molecular orbital results")?;
            writeln!(
                f,
                "{}",
                "┈".repeat(16 + mo_energy_length + mo_symmetry_length)
            )?;
            writeln!(
                f,
                "{:>5}  {:>4}  {:<mo_energy_length$}  {}",
                "Spin", "MO", "Energy", "Symmetry"
            )?;
            writeln!(
                f,
                "{}",
                "┈".repeat(16 + mo_energy_length + mo_symmetry_length)
            )?;
            for (spini, spin_mo_symmetries) in mo_symmetries.iter().enumerate() {
                for (moi, mo_sym) in spin_mo_symmetries.iter().enumerate() {
                    let mo_energy_str = mo_energies_opt
                        .and_then(|mo_energies| mo_energies.get(spini))
                        .and_then(|spin_mo_energies| spin_mo_energies.get(moi))
                        .map(|mo_energy| format!("{:>+mo_energy_length$.7}", mo_energy))
                        .unwrap_or("--".to_string());
                    let mo_sym_str = mo_sym
                        .as_ref()
                        .map(|sym| sym.to_string())
                        .unwrap_or("--".to_string());
                    writeln!(
                        f,
                        "{spini:>5}  {moi:>4}  {mo_energy_str:<mo_energy_length$}  {mo_sym_str}"
                    )?;
                }
            }
            writeln!(
                f,
                "{}",
                "┈".repeat(16 + mo_energy_length + mo_symmetry_length)
            )?;
        }

        Ok(())
    }
}

impl<'a, R, T> fmt::Debug for SlaterDeterminantRepAnalysisResult<'a, R, T>
where
    R: ReducibleLinearSpaceSymbol,
    T: ComplexFloat + Lapack,
    <T as ComplexFloat>::Real: fmt::LowerExp + fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{self}")
    }
}

// ------
// Driver
// ------

#[derive(Clone, Builder)]
#[builder(build_fn(validate = "Self::validate"))]
pub struct SlaterDeterminantRepAnalysisDriver<'a, R, T>
where
    R: ReducibleLinearSpaceSymbol,
    T: ComplexFloat + Lapack,
    <T as ComplexFloat>::Real: fmt::LowerExp + fmt::Debug,
{
    /// The control parameters for Slater determinant representation analysis.
    parameters: &'a SlaterDeterminantRepAnalysisParams<T>,

    determinant: &'a SlaterDeterminant<'a, T>,

    symmetry_group: &'a SymmetryGroupDetectionResult<'a>,

    sao_spatial: &'a Array2<T>,

    #[builder(setter(skip), default = "None")]
    result: Option<SlaterDeterminantRepAnalysisResult<'a, R, T>>,
}

impl<'a, R, T> SlaterDeterminantRepAnalysisDriverBuilder<'a, R, T>
where
    R: ReducibleLinearSpaceSymbol,
    T: ComplexFloat + Lapack,
    <T as ComplexFloat>::Real: fmt::LowerExp + fmt::Debug,
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

        let sym = if params.use_magnetic_group {
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
                    "Molecule symmetrisation cannot be performed using the entirety of the infinite group `{}`. Consider setting the parameter `infinite_order_to_finite` to restrict to a finite subgroup instead.",
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

impl<'a, R, T> SlaterDeterminantRepAnalysisDriver<'a, R, T>
where
    R: ReducibleLinearSpaceSymbol,
    T: ComplexFloat + Lapack,
    <T as ComplexFloat>::Real: fmt::LowerExp + fmt::Debug,
{
    /// Returns a builder to construct a [`SlaterDeterminantRepAnalysisDriver`] structure.
    pub fn builder() -> SlaterDeterminantRepAnalysisDriverBuilder<'a, R, T> {
        SlaterDeterminantRepAnalysisDriverBuilder::default()
    }

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

impl<'a, R, T> fmt::Display for SlaterDeterminantRepAnalysisDriver<'a, R, T>
where
    R: ReducibleLinearSpaceSymbol,
    T: ComplexFloat + Lapack,
    <T as ComplexFloat>::Real: fmt::LowerExp + fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write_title(f, "Slater Determinant Symmetry Analysis")?;
        writeln!(f, "")?;
        writeln!(f, "{}", self.parameters)?;
        Ok(())
    }
}

impl<'a, R, T> fmt::Debug for SlaterDeterminantRepAnalysisDriver<'a, R, T>
where
    R: ReducibleLinearSpaceSymbol,
    T: ComplexFloat + Lapack,
    <T as ComplexFloat>::Real: fmt::LowerExp + fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{self}")
    }
}

impl<'a, T> SlaterDeterminantRepAnalysisDriver<'a, DecomposedSymbol<MullikenIrrepSymbol>, T>
where
    T: ComplexFloat + Lapack,
    <T as ComplexFloat>::Real: fmt::LowerExp + fmt::Debug,
{
    fn construct_unitary_group(&self) -> Result<UnitaryRepresentedSymmetryGroup, anyhow::Error> {
        let params = self.parameters;
        let sym = if params.use_magnetic_group {
            self.symmetry_group
                .magnetic_symmetry
                .as_ref()
                .ok_or_else(|| {
                    format_err!(
                        "Magnetic symmetry requested for analysis, but no magnetic symmetry found."
                    )
                })?
        } else {
            &self.symmetry_group.unitary_symmetry
        };
        let group = if params.use_double_group {
            UnitaryRepresentedGroup::from_molecular_symmetry(sym, params.infinite_order_to_finite)?
                .to_double_group()?
        } else {
            UnitaryRepresentedGroup::from_molecular_symmetry(sym, params.infinite_order_to_finite)?
        };

        log::info!(
            target: "output",
            "Unitary-represented group for representation analysis: {}",
            group.name()
        );
        log::info!(target: "output", "");

        Ok(group)
    }
}

impl<'a> SlaterDeterminantRepAnalysisDriver<'a, DecomposedSymbol<MullikenIrrepSymbol>, f64> {
    fn analyse_representation(&mut self) -> Result<(), anyhow::Error> {
        let params = self.parameters;
        let sao = self.construct_sao()?;
        let group = self.construct_unitary_group()?;

        let mut det_orbit = SlaterDeterminantSymmetryOrbit::builder()
            .group(&group)
            .origin(self.determinant)
            .integrality_threshold(params.integrality_threshold)
            .linear_independence_threshold(params.linear_independence_threshold)
            .symmetry_transformation_kind(params.symmetry_transformation_kind.clone())
            .build()?;
        det_orbit.calc_smat(Some(&sao)).calc_xmat(false);
        let det_symmetry = det_orbit.analyse_rep().ok();

        let mo_symmetries = if params.analyse_mo_symmetries {
            let mos = self.determinant.to_orbitals();
            let mut mos_orbits = MolecularOrbitalSymmetryOrbit::from_orbitals(
                &group,
                &mos,
                params.symmetry_transformation_kind.clone(),
                params.integrality_threshold,
                params.linear_independence_threshold,
            );
            let m = mos_orbits
                .iter_mut()
                .map(|mos_orbit| {
                    mos_orbit
                        .iter_mut()
                        .map(|mo_orbit| {
                            mo_orbit.calc_smat(Some(&sao)).calc_xmat(false);
                            mo_orbit.analyse_rep().ok()
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            Some(m)
        } else {
            None
        };

        let result = SlaterDeterminantRepAnalysisResult::builder()
            .parameters(params)
            .determinant(self.determinant)
            .determinant_symmetry(det_symmetry)
            .mo_symmetries(mo_symmetries)
            .build()?;
        self.result = Some(result);

        Ok(())
    }
}

impl<'a>
    SlaterDeterminantRepAnalysisDriver<'a, DecomposedSymbol<MullikenIrrepSymbol>, Complex<f64>>
{
    fn analyse_representation(&mut self) -> Result<(), anyhow::Error> {
        let params = self.parameters;
        let sao = self.construct_sao()?;
        let group = self.construct_unitary_group()?;

        let mut det_orbit = SlaterDeterminantSymmetryOrbit::builder()
            .group(&group)
            .origin(self.determinant)
            .integrality_threshold(params.integrality_threshold)
            .linear_independence_threshold(params.linear_independence_threshold)
            .symmetry_transformation_kind(params.symmetry_transformation_kind.clone())
            .build()?;
        det_orbit.calc_smat(Some(&sao)).calc_xmat(false);
        let det_symmetry = det_orbit.analyse_rep().ok();

        let mo_symmetries = if params.analyse_mo_symmetries {
            let mos = self.determinant.to_orbitals();
            let mut mos_orbits = MolecularOrbitalSymmetryOrbit::from_orbitals(
                &group,
                &mos,
                params.symmetry_transformation_kind.clone(),
                params.integrality_threshold,
                params.linear_independence_threshold,
            );
            let m = mos_orbits
                .iter_mut()
                .map(|mos_orbit| {
                    mos_orbit
                        .iter_mut()
                        .map(|mo_orbit| {
                            mo_orbit.calc_smat(Some(&sao)).calc_xmat(false);
                            mo_orbit.analyse_rep().ok()
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            Some(m)
        } else {
            None
        };

        let result = SlaterDeterminantRepAnalysisResult::builder()
            .parameters(params)
            .determinant(self.determinant)
            .determinant_symmetry(det_symmetry)
            .mo_symmetries(mo_symmetries)
            .build()?;
        self.result = Some(result);

        Ok(())
    }
}

impl<'a, T> SlaterDeterminantRepAnalysisDriver<'a, DecomposedSymbol<MullikenIrcorepSymbol>, T>
where
    T: ComplexFloat + Lapack,
    <T as ComplexFloat>::Real: fmt::LowerExp + fmt::Debug,
{
    fn construct_magnetic_group(&self) -> Result<MagneticRepresentedSymmetryGroup, anyhow::Error> {
        let params = self.parameters;
        let sym = if params.use_magnetic_group {
            self.symmetry_group
                .magnetic_symmetry
                .as_ref()
                .ok_or_else(|| {
                    format_err!(
                        "Magnetic symmetry requested for analysis, but no magnetic symmetry found."
                    )
                })?
        } else {
            &self.symmetry_group.unitary_symmetry
        };
        let group = if params.use_double_group {
            MagneticRepresentedGroup::from_molecular_symmetry(sym, params.infinite_order_to_finite)?
                .to_double_group()?
        } else {
            MagneticRepresentedGroup::from_molecular_symmetry(sym, params.infinite_order_to_finite)?
        };

        log::info!(
            target: "output",
            "Magnetic-represented group for corepresentation analysis: {}",
            group.name()
        );
        log::info!(target: "output", "");

        Ok(group)
    }
}

impl<'a> SlaterDeterminantRepAnalysisDriver<'a, DecomposedSymbol<MullikenIrcorepSymbol>, f64> {
    fn analyse_corepresentation(&mut self) -> Result<(), anyhow::Error> {
        let params = self.parameters;
        let sao = self.construct_sao()?;
        let group = self.construct_magnetic_group()?;

        let mut det_orbit = SlaterDeterminantSymmetryOrbit::builder()
            .group(&group)
            .origin(self.determinant)
            .integrality_threshold(params.integrality_threshold)
            .linear_independence_threshold(params.linear_independence_threshold)
            .symmetry_transformation_kind(params.symmetry_transformation_kind.clone())
            .build()?;
        det_orbit.calc_smat(Some(&sao)).calc_xmat(false);
        let det_symmetry = det_orbit.analyse_rep().ok();

        let mo_symmetries = if params.analyse_mo_symmetries {
            let mos = self.determinant.to_orbitals();
            let mut mos_orbits = MolecularOrbitalSymmetryOrbit::from_orbitals(
                &group,
                &mos,
                params.symmetry_transformation_kind.clone(),
                params.integrality_threshold,
                params.linear_independence_threshold,
            );
            let m = mos_orbits
                .iter_mut()
                .map(|mos_orbit| {
                    mos_orbit
                        .iter_mut()
                        .map(|mo_orbit| {
                            mo_orbit.calc_smat(Some(&sao)).calc_xmat(false);
                            mo_orbit.analyse_rep().ok()
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            Some(m)
        } else {
            None
        };

        let result = SlaterDeterminantRepAnalysisResult::builder()
            .parameters(params)
            .determinant(self.determinant)
            .determinant_symmetry(det_symmetry)
            .mo_symmetries(mo_symmetries)
            .build()?;
        self.result = Some(result);

        Ok(())
    }
}

impl<'a>
    SlaterDeterminantRepAnalysisDriver<'a, DecomposedSymbol<MullikenIrcorepSymbol>, Complex<f64>>
{
    fn analyse_corepresentation(&mut self) -> Result<(), anyhow::Error> {
        let params = self.parameters;
        let sao = self.construct_sao()?;
        let group = self.construct_magnetic_group()?;

        let mut det_orbit = SlaterDeterminantSymmetryOrbit::builder()
            .group(&group)
            .origin(self.determinant)
            .integrality_threshold(params.integrality_threshold)
            .linear_independence_threshold(params.linear_independence_threshold)
            .symmetry_transformation_kind(params.symmetry_transformation_kind.clone())
            .build()?;
        det_orbit.calc_smat(Some(&sao)).calc_xmat(false);
        let det_symmetry = det_orbit.analyse_rep().ok();

        let mo_symmetries = if params.analyse_mo_symmetries {
            let mos = self.determinant.to_orbitals();
            let mut mos_orbits = MolecularOrbitalSymmetryOrbit::from_orbitals(
                &group,
                &mos,
                params.symmetry_transformation_kind.clone(),
                params.integrality_threshold,
                params.linear_independence_threshold,
            );
            let m = mos_orbits
                .iter_mut()
                .map(|mos_orbit| {
                    mos_orbit
                        .iter_mut()
                        .map(|mo_orbit| {
                            mo_orbit.calc_smat(Some(&sao)).calc_xmat(false);
                            mo_orbit.analyse_rep().ok()
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            Some(m)
        } else {
            None
        };

        let result = SlaterDeterminantRepAnalysisResult::builder()
            .parameters(params)
            .determinant(self.determinant)
            .determinant_symmetry(det_symmetry)
            .mo_symmetries(mo_symmetries)
            .build()?;
        self.result = Some(result);

        Ok(())
    }
}

impl<'a> QSym2Driver
    for SlaterDeterminantRepAnalysisDriver<'a, DecomposedSymbol<MullikenIrrepSymbol>, f64>
{
    type Outcome =
        SlaterDeterminantRepAnalysisResult<'a, DecomposedSymbol<MullikenIrrepSymbol>, f64>;

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

impl<'a> QSym2Driver
    for SlaterDeterminantRepAnalysisDriver<'a, DecomposedSymbol<MullikenIrrepSymbol>, Complex<f64>>
{
    type Outcome =
        SlaterDeterminantRepAnalysisResult<'a, DecomposedSymbol<MullikenIrrepSymbol>, Complex<f64>>;

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

impl<'a> QSym2Driver
    for SlaterDeterminantRepAnalysisDriver<'a, DecomposedSymbol<MullikenIrcorepSymbol>, f64>
{
    type Outcome =
        SlaterDeterminantRepAnalysisResult<'a, DecomposedSymbol<MullikenIrcorepSymbol>, f64>;

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

impl<'a> QSym2Driver
    for SlaterDeterminantRepAnalysisDriver<
        'a,
        DecomposedSymbol<MullikenIrcorepSymbol>,
        Complex<f64>,
    >
{
    type Outcome = SlaterDeterminantRepAnalysisResult<
        'a,
        DecomposedSymbol<MullikenIrcorepSymbol>,
        Complex<f64>,
    >;

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
