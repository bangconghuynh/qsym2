use std::fmt;
use std::path::PathBuf;

use anyhow::{self, bail, format_err, Context};
use derive_builder::Builder;
use hdf5::{self, H5Type};
use lazy_static::lazy_static;
use log;
use nalgebra::Point3;
use ndarray::{Array1, Array2, Axis, Ix3};
use ndarray_linalg::types::Lapack;
use num_complex::ComplexFloat;
use num_traits::{One, Zero};
use numeric_sort;
use periodic_table::periodic_table;
use regex::Regex;

use crate::angmom::spinor_rotation_3d::SpinConstraint;
use crate::auxiliary::atom::{Atom, ElementMap};
use crate::auxiliary::molecule::Molecule;
use crate::basis::ao::BasisAngularOrder;
use crate::basis::ao::*;
use crate::chartab::chartab_group::CharacterProperties;
use crate::chartab::SubspaceDecomposable;
use crate::drivers::representation_analysis::angular_function::AngularFunctionRepAnalysisParams;
use crate::drivers::representation_analysis::slater_determinant::{
    SlaterDeterminantRepAnalysisDriver, SlaterDeterminantRepAnalysisParams,
};
use crate::drivers::representation_analysis::MagneticSymmetryAnalysisKind;
use crate::drivers::symmetry_group_detection::{
    SymmetryGroupDetectionDriver, SymmetryGroupDetectionResult,
};
use crate::drivers::QSym2Driver;
use crate::interfaces::input::SymmetryGroupDetectionInputKind;
use crate::io::format::{log_macsec_begin, log_macsec_end, qsym2_error, qsym2_output};
use crate::io::{read_qsym2_binary, QSym2FileType};
use crate::symmetry::symmetry_core::Symmetry;
use crate::symmetry::symmetry_group::{
    MagneticRepresentedSymmetryGroup, SymmetryGroupProperties, UnitaryRepresentedSymmetryGroup,
};
use crate::target::determinant::SlaterDeterminant;

#[cfg(test)]
#[path = "hdf5_tests.rs"]
mod hdf5_test;

// ---------------------
// Full Q-Chem H5 Driver
// ---------------------

lazy_static! {
    static ref SP_PATH_RE: Regex =
        Regex::new(r"(.*sp)\\energy_function$").expect("Regex pattern invalid.");
}

// ~~~~~~~~~~~~~~~~~
// Struct definition
// ~~~~~~~~~~~~~~~~~

/// A driver to perform symmetry-group detection and representation symmetry analysis for all
/// discoverable single-point calculation data stored in a Q-Chem's `qarchive.h5` file.
#[derive(Clone, Builder)]
pub(crate) struct QChemH5Driver<'a, T>
where
    T: ComplexFloat + Lapack,
    <T as ComplexFloat>::Real: From<f64> + fmt::LowerExp + fmt::Debug,
{
    /// The `qarchive.h5` file name.
    filename: PathBuf,

    /// The input specification controlling symmetry-group detection.
    symmetry_group_detection_input: &'a SymmetryGroupDetectionInputKind,

    /// The parameters controlling representation analysis of standard angular functions.
    angular_function_analysis_parameters: &'a AngularFunctionRepAnalysisParams,

    /// The parameters controlling representation analysis of Slater determinants.
    slater_det_rep_analysis_parameters:
        &'a SlaterDeterminantRepAnalysisParams<<T as ComplexFloat>::Real>,

    /// The simplified result of the analysis. Each element in the vector is a tuple containing the
    /// group name and the representation symmetry of the Slater determinant for one single-point
    /// calculation.
    #[builder(default = "None")]
    result: Option<Vec<(String, String)>>,
}

// ~~~~~~~~~~~~~~~~~~~~~~
// Struct implementations
// ~~~~~~~~~~~~~~~~~~~~~~

// Generic for all determinant numeric type T
// ''''''''''''''''''''''''''''''''''''''''''

impl<'a, T> QChemH5Driver<'a, T>
where
    T: ComplexFloat + Lapack + H5Type,
    <T as ComplexFloat>::Real: From<f64> + fmt::LowerExp + fmt::Debug,
{
    /// Returns a builder to construct a [`QChemH5Driver`].
    pub(crate) fn builder() -> QChemH5DriverBuilder<'a, T> {
        QChemH5DriverBuilder::default()
    }
}

// Specific for determinant numeric type f64
// '''''''''''''''''''''''''''''''''''''''''

impl<'a> QChemH5Driver<'a, f64> {
    /// Performs analysis for all real-valued single-point determinants.
    fn analyse(&mut self) -> Result<(), anyhow::Error> {
        let f = hdf5::File::open(&self.filename)?;
        let mut sp_paths = f
            .group(".counters")?
            .member_names()?
            .iter()
            .filter_map(|path| {
                if SP_PATH_RE.is_match(path) {
                    let path = path.replace("\\", "/");
                    let mut energy_function_indices = f
                        .group(&path)
                        .and_then(|sp_energy_function_group| {
                            sp_energy_function_group.member_names()
                        })
                        .ok()?;
                    numeric_sort::sort(&mut energy_function_indices);
                    Some((
                        path.replace("/energy_function", ""),
                        energy_function_indices,
                    ))
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        sp_paths.sort_by(|(path_a, _), (path_b, _)| numeric_sort::cmp(path_a, path_b));

        let pd_input = self.symmetry_group_detection_input;
        let afa_params = self.angular_function_analysis_parameters;
        let sda_params = self.slater_det_rep_analysis_parameters;
        let result = sp_paths
            .iter()
            .flat_map(|(sp_path, energy_function_indices)| {
                energy_function_indices.iter().map(|energy_function_index| {
                    log_macsec_begin(&format!(
                        "Analysis for {} (energy function {energy_function_index})",
                        sp_path.clone()
                    ));
                    qsym2_output!("");
                    let sp = f.group(sp_path)?;
                    let sp_driver_result = match sda_params.use_magnetic_group {
                        Some(MagneticSymmetryAnalysisKind::Corepresentation) => {
                            let mut sp_driver = QChemH5SinglePointDriver::<
                                MagneticRepresentedSymmetryGroup,
                                f64,
                            >::builder()
                            .sp_group(&sp)
                            .energy_function_index(energy_function_index)
                            .symmetry_group_detection_input(pd_input)
                            .angular_function_analysis_parameters(afa_params)
                            .slater_det_rep_analysis_parameters(sda_params)
                            .build()?;
                            let _ = sp_driver.run();
                            sp_driver.result().map(|(sym, rep)| {
                                (
                                    sym.group_name
                                        .as_ref()
                                        .unwrap_or(&String::new())
                                        .to_string(),
                                    rep.as_ref()
                                        .map(|rep| rep.to_string())
                                        .unwrap_or_else(|err| err.to_string()),
                                )
                            })
                        }
                        Some(MagneticSymmetryAnalysisKind::Representation) | None => {
                            let mut sp_driver = QChemH5SinglePointDriver::<
                                UnitaryRepresentedSymmetryGroup,
                                f64,
                            >::builder()
                            .sp_group(&sp)
                            .energy_function_index(energy_function_index)
                            .symmetry_group_detection_input(pd_input)
                            .angular_function_analysis_parameters(afa_params)
                            .slater_det_rep_analysis_parameters(sda_params)
                            .build()?;
                            let _ = sp_driver.run();
                            sp_driver.result().map(|(sym, rep)| {
                                (
                                    sym.group_name
                                        .as_ref()
                                        .unwrap_or(&String::new())
                                        .to_string(),
                                    rep.as_ref()
                                        .map(|rep| rep.to_string())
                                        .unwrap_or_else(|err| err.to_string()),
                                )
                            })
                        }
                    };
                    qsym2_output!("");
                    log_macsec_end(&format!(
                        "Analysis for {} (energy function {energy_function_index})",
                        sp_path.clone()
                    ));
                    qsym2_output!("");
                    sp_driver_result
                })
            })
            .map(|res| {
                res.unwrap_or((
                    "Unidentified symmetry group".to_string(),
                    "Unidentified (co)representation".to_string(),
                ))
            })
            .collect::<Vec<_>>();

        log_macsec_begin("Q-Chem HDF5 Archive Summary");
        qsym2_output!("");
        let path_length = sp_paths
            .iter()
            .map(|(path, _)| path.chars().count())
            .max()
            .unwrap_or(18)
            .max(18);
        let energy_function_length = sp_paths
            .iter()
            .map(|(_, energy_function_indices)| {
                energy_function_indices
                    .iter()
                    .map(|index| index.chars().count())
                    .max()
                    .unwrap_or(1)
                    .max(1)
            })
            .max()
            .unwrap_or(7)
            .max(7);
        let group_length = result
            .iter()
            .map(|(group, _)| group.chars().count())
            .max()
            .unwrap_or(5)
            .max(5);
        let sym_length = result
            .iter()
            .map(|(_, sym)| sym.chars().count())
            .max()
            .unwrap_or(13)
            .max(13);
        let table_width = path_length + energy_function_length + group_length + sym_length + 8;
        qsym2_output!("{}", "┈".repeat(table_width));
        qsym2_output!(
            " {:<path_length$}  {:<energy_function_length$}  {:<group_length$}  {:<}",
            "Single-point calc.",
            "E func.",
            "Group",
            "Det. symmetry"
        );
        qsym2_output!("{}", "┈".repeat(table_width));
        sp_paths
            .iter()
            .flat_map(|(sp_path, energy_function_indices)| {
                energy_function_indices
                    .iter()
                    .map(|index| (sp_path.clone(), index))
            })
            .zip(result.iter())
            .for_each(|((path, index), (group, sym))| {
                qsym2_output!(
                    " {:<path_length$}  {:<energy_function_length$}  {:<group_length$}  {:<#}",
                    path,
                    index,
                    group,
                    sym
                );
            });
        qsym2_output!("{}", "┈".repeat(table_width));
        qsym2_output!("");
        log_macsec_end("Q-Chem HDF5 Archive Summary");

        self.result = Some(result);
        Ok(())
    }
}

impl<'a> QSym2Driver for QChemH5Driver<'a, f64> {
    type Outcome = Vec<(String, String)>;

    fn result(&self) -> Result<&Self::Outcome, anyhow::Error> {
        self.result
            .as_ref()
            .ok_or(format_err!("No Q-Chem HDF5 analysis results found."))
    }

    fn run(&mut self) -> Result<(), anyhow::Error> {
        self.analyse()
    }
}

// ------------------
// SinglePoint Driver
// ------------------

// ~~~~~~~~~~~~~~~
// Enum definition
// ~~~~~~~~~~~~~~~

/// An enumerated type to distinguish different kinds of molecular orbitals.
enum OrbitalType {
    /// Canonical molecular orbitals as obtained by diagonalising Fock matrices.
    Canonical,

    /// Localised molecular orbitals as obtained by a localisation method.
    Localised,
}

// ~~~~~~~~~~~~~~~~~
// Struct definition
// ~~~~~~~~~~~~~~~~~

/// A driver to perform symmetry-group detection and representation analysis for a single-point
/// calculation result in a Q-Chem's `qarchive.h5` file.
#[derive(Clone, Builder)]
struct QChemH5SinglePointDriver<'a, G, T>
where
    G: SymmetryGroupProperties + Clone,
    G::CharTab: SubspaceDecomposable<T>,
    T: ComplexFloat + Lapack,
    <T as ComplexFloat>::Real: From<f64> + fmt::LowerExp + fmt::Debug,
{
    /// A H5 group containing data from a single-point calculation.
    sp_group: &'a hdf5::Group,

    /// The index of the energy function whose results are to be considered for this single-point
    /// symmetry analysis.
    #[builder(setter(custom))]
    energy_function_index: String,

    /// The parameters controlling symmetry-group detection.
    symmetry_group_detection_input: &'a SymmetryGroupDetectionInputKind,

    /// The parameters controlling representation analysis of standard angular functions.
    angular_function_analysis_parameters: &'a AngularFunctionRepAnalysisParams,

    /// The parameters controlling representation analysis of Slater determinants.
    slater_det_rep_analysis_parameters:
        &'a SlaterDeterminantRepAnalysisParams<<T as ComplexFloat>::Real>,

    /// The symmetry of the system and the representation of the Slater determinant.
    #[builder(default = "None")]
    result: Option<(
        Symmetry,
        Result<<G::CharTab as SubspaceDecomposable<T>>::Decomposition, String>,
    )>,
}

// ~~~~~~~~~~~~~~~~~~~~~~
// Struct implementations
// ~~~~~~~~~~~~~~~~~~~~~~

impl<'a, G, T> QChemH5SinglePointDriverBuilder<'a, G, T>
where
    G: SymmetryGroupProperties + Clone,
    G::CharTab: SubspaceDecomposable<T>,
    T: ComplexFloat + Lapack,
    <T as ComplexFloat>::Real: From<f64> + fmt::LowerExp + fmt::Debug,
{
    fn energy_function_index(&mut self, idx: &str) -> &mut Self {
        self.energy_function_index = Some(idx.to_string());
        self
    }
}

// Generic for all symmetry groups G and determinant numeric type T
// ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

impl<'a, G, T> QChemH5SinglePointDriver<'a, G, T>
where
    G: SymmetryGroupProperties + Clone,
    G::CharTab: SubspaceDecomposable<T>,
    T: ComplexFloat + Lapack + H5Type,
    <T as ComplexFloat>::Real: From<f64> + fmt::LowerExp + fmt::Debug,
{
    /// Returns a builder to construct a [`QChemH5SinglePointDriver`].
    fn builder() -> QChemH5SinglePointDriverBuilder<'a, G, T> {
        QChemH5SinglePointDriverBuilder::default()
    }

    /// Extracts the molecular structure from the single-point H5 group.
    fn extract_molecule(&self) -> Result<Molecule, anyhow::Error> {
        let emap = ElementMap::new();
        let coordss = self
            .sp_group
            .dataset("structure/coordinates")?
            .read_2d::<f64>()?;
        let atomic_numbers = self
            .sp_group
            .dataset("structure/nuclei")?
            .read_1d::<usize>()?;
        let atoms = coordss
            .rows()
            .into_iter()
            .zip(atomic_numbers.iter())
            .map(|(coords, atomic_number)| {
                let element = periodic_table()
                    .get(*atomic_number - 1)
                    .ok_or(hdf5::Error::from(
                        format!(
                            "Element with atomic number {atomic_number} could not be identified."
                        )
                        .as_str(),
                    ))?
                    .symbol;
                let coordinates = Point3::new(coords[0], coords[1], coords[2]);
                Ok::<_, hdf5::Error>(Atom::new_ordinary(element, coordinates, &emap, 1e-8))
            })
            .collect::<Result<Vec<Atom>, _>>()?;
        let mol = Molecule::from_atoms(&atoms, 1e-14);
        Ok(mol)
    }

    /// Extracts the spatial atomic-orbital overlap matrix from the single-point H5 group.
    fn extract_sao(&self) -> Result<Array2<T>, anyhow::Error> {
        self.sp_group
            .dataset("aobasis/overlap_matrix")?
            .read_2d::<T>()
            .map_err(|err| err.into())
    }

    /// Extracts the basis angular order information from the single-point H5 group.
    fn extract_bao(&self, mol: &'a Molecule) -> Result<BasisAngularOrder<'a>, anyhow::Error> {
        let shell_types = self
            .sp_group
            .dataset("aobasis/shell_types")?
            .read_1d::<i32>()?;
        let shell_to_atom_map = self
            .sp_group
            .dataset("aobasis/shell_to_atom_map")?
            .read_1d::<usize>()?
            .iter()
            .zip(shell_types.iter())
            .flat_map(|(&idx, shell_type)| {
                if *shell_type == -1 {
                    vec![idx, idx]
                } else {
                    vec![idx]
                }
            })
            .collect::<Vec<_>>();

        let bss: Vec<BasisShell> = shell_types
            .iter()
            .flat_map(|shell_type| {
                if *shell_type == 0 {
                    // S shell
                    vec![BasisShell::new(0, ShellOrder::Cart(CartOrder::qchem(0)))]
                } else if *shell_type == 1 {
                    // P shell
                    vec![BasisShell::new(1, ShellOrder::Cart(CartOrder::qchem(1)))]
                } else if *shell_type == -1 {
                    // SP shell
                    vec![
                        BasisShell::new(0, ShellOrder::Cart(CartOrder::qchem(0))),
                        BasisShell::new(1, ShellOrder::Cart(CartOrder::qchem(1))),
                    ]
                } else if *shell_type < 0 {
                    // Cartesian D shell or higher
                    let l = shell_type.unsigned_abs();
                    vec![BasisShell::new(l, ShellOrder::Cart(CartOrder::qchem(l)))]
                } else {
                    // Pure D shell or higher
                    let l = shell_type.unsigned_abs();
                    vec![BasisShell::new(
                        l,
                        ShellOrder::Pure(PureOrder::increasingm(l)),
                    )]
                }
            })
            .collect::<Vec<BasisShell>>();

        let batms = mol
            .atoms
            .iter()
            .enumerate()
            .map(|(atom_i, atom)| {
                let shells = bss
                    .iter()
                    .zip(shell_to_atom_map.iter())
                    .filter_map(|(bs, atom_index)| {
                        if *atom_index == atom_i {
                            Some(bs.clone())
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>();
                BasisAtom::new(atom, &shells)
            })
            .collect::<Vec<BasisAtom>>();
        Ok(BasisAngularOrder::new(&batms))
    }

    /// Extracts the Slater determinant from the single-point H5 group.
    ///
    /// # Arguments
    ///
    /// * `mol` - The molecule to be associated with the extracted determinant.
    /// * `bao` - The basis angular order information to be associated with the extracted determinant.
    /// * `threshold` - The comparison threshold to be associated with the extracted determinant.
    ///
    /// # Returns
    ///
    /// The extracted Slater determinant.
    fn extract_determinant(
        &self,
        mol: &'a Molecule,
        bao: &'a BasisAngularOrder,
        threshold: <T as ComplexFloat>::Real,
        orbital_type: OrbitalType,
    ) -> Result<SlaterDeterminant<'a, T>, anyhow::Error> {
        let energy = self
            .sp_group
            .dataset(&format!(
                "energy_function/{}/energy",
                self.energy_function_index
            ))?
            .read_scalar::<T>()
            .map_err(|err| err.to_string());
        let orbital_path = match orbital_type {
            OrbitalType::Canonical => format!(
                "energy_function/{}/method/scf/molecular_orbitals",
                self.energy_function_index
            ),
            OrbitalType::Localised => format!(
                "energy_function/{}/analysis/localized_orbitals/{}/molecular_orbitals",
                self.energy_function_index, self.energy_function_index
            ),
        };
        let nspins = self
            .sp_group
            .dataset(&format!("{orbital_path}/nsets"))?
            .read_scalar::<usize>()?;
        let nmo = self
            .sp_group
            .dataset(&format!("{orbital_path}/norb",))?
            .read_scalar::<usize>()?;
        let (spincons, occs) = match nspins {
            1 => {
                log::warn!(
                    "The number of spin spaces detected is 1. \
                    It will be assumed that this implies an RHF calculation. \
                    However, it must be noted that, if the calculation is GHF instead, then the \
                    following symmetry analysis will be wrong, because Q-Chem does not archive \
                    GHF MO coefficients correctly."
                );
                let nalpha = self
                    .sp_group
                    .dataset("structure/nalpha")?
                    .read_scalar::<usize>()?;
                let occ_a = Array1::from_vec(
                    (0..nmo)
                        .map(|i| {
                            if i < nalpha {
                                <T as ComplexFloat>::Real::one()
                            } else {
                                <T as ComplexFloat>::Real::zero()
                            }
                        })
                        .collect::<Vec<_>>(),
                );
                (SpinConstraint::Restricted(2), vec![occ_a])
            }
            2 => {
                let nalpha = self
                    .sp_group
                    .dataset("structure/nalpha")?
                    .read_scalar::<usize>()?;
                let nbeta = self
                    .sp_group
                    .dataset("structure/nbeta")?
                    .read_scalar::<usize>()?;
                let occ_a = Array1::from_vec(
                    (0..nmo)
                        .map(|i| {
                            if i < nalpha {
                                <T as ComplexFloat>::Real::one()
                            } else {
                                <T as ComplexFloat>::Real::zero()
                            }
                        })
                        .collect::<Vec<_>>(),
                );
                let occ_b = Array1::from_vec(
                    (0..nmo)
                        .map(|i| {
                            if i < nbeta {
                                <T as ComplexFloat>::Real::one()
                            } else {
                                <T as ComplexFloat>::Real::zero()
                            }
                        })
                        .collect::<Vec<_>>(),
                );
                (SpinConstraint::Unrestricted(2, true), vec![occ_a, occ_b])
            }
            _ => {
                bail!("Unexpected number of spin spaces from Q-Chem.")
            }
        };
        let cs = self
            .sp_group
            .dataset(&format!("{orbital_path}/mo_coefficients"))?
            .read::<T, Ix3>()?
            .axis_iter(Axis(0))
            .map(|c| c.to_owned())
            .collect::<Vec<_>>();
        let mo_energies = self
            .sp_group
            .dataset(&format!("{orbital_path}/mo_energies"))
            .and_then(|mo_energies_dataset| {
                mo_energies_dataset.read_2d::<T>().map(|mo_energies_arr| {
                    mo_energies_arr
                        .columns()
                        .into_iter()
                        .map(|c| c.to_owned())
                        .collect::<Vec<_>>()
                })
            })
            .ok();

        SlaterDeterminant::builder()
            .spin_constraint(spincons)
            .bao(bao)
            .complex_symmetric(false)
            .mol(mol)
            .coefficients(&cs)
            .occupations(&occs)
            .mo_energies(mo_energies)
            .energy(energy)
            .threshold(threshold)
            .build()
            .map_err(|err| err.into())
    }
}

// Specific for unitary-represented symmetry groups and determinant numeric type f64
// '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

impl<'a> QChemH5SinglePointDriver<'a, UnitaryRepresentedSymmetryGroup, f64> {
    /// Performs symmetry-group detection and unitary-represented representation analysis.
    fn analyse(&mut self) -> Result<(), anyhow::Error> {
        let mol = self.extract_molecule()
            .with_context(|| "Unable to extract the molecule from the HDF5 file while performing symmetry analysis for a single-point Q-Chem calculation")?;
        log::debug!("Performing symmetry-group detection...");
        let pd_res = match self.symmetry_group_detection_input {
            SymmetryGroupDetectionInputKind::Parameters(pd_params) => {
                let mut pd_driver = SymmetryGroupDetectionDriver::builder()
                    .parameters(pd_params)
                    .molecule(Some(&mol))
                    .build()
                    .unwrap();
                let pd_run = pd_driver.run();
                if let Err(err) = pd_run {
                    qsym2_error!("Symmetry-group detection has failed with error:");
                    qsym2_error!("  {err:#}");
                }
                let pd_res = pd_driver.result()?;
                pd_res.clone()
            }
            SymmetryGroupDetectionInputKind::FromFile(path) => {
                read_qsym2_binary::<SymmetryGroupDetectionResult, _>(path, QSym2FileType::Sym)
                    .with_context(|| "Unable to read the specified .qsym2.sym file while performing symmetry analysis for a single-point Q-Chem calculation")?
            }
        };
        let recentred_mol = &pd_res.pre_symmetry.recentred_molecule;
        let sym = if self
            .slater_det_rep_analysis_parameters
            .use_magnetic_group
            .is_some()
        {
            pd_res.magnetic_symmetry.clone()
        } else {
            Some(pd_res.unitary_symmetry.clone())
        }
        .ok_or(format_err!("Symmetry not found."))?;
        log::debug!("Performing symmetry-group detection... Done.");

        let rep = || {
            log::debug!("Extracting AO basis information for representation analysis...");
            let sao = self.extract_sao()
                .with_context(|| "Unable to extract the SAO matrix from the HDF5 file while performing symmetry analysis for a single-point Q-Chem calculation")
                .map_err(|err| err.to_string())?;
            let bao = self.extract_bao(recentred_mol)
                .with_context(|| "Unable to extract the basis angular order information from the HDF5 file while performing symmetry analysis for a single-point Q-Chem calculation")
                .map_err(|err| err.to_string())?;
            log::debug!("Extracting AO basis information for representation analysis... Done.");
            log::debug!("Extracting canonical determinant information for representation analysis...");
            let det = self.extract_determinant(
                recentred_mol,
                &bao,
                self.slater_det_rep_analysis_parameters
                    .linear_independence_threshold,
                OrbitalType::Canonical,
            )
            .with_context(|| "Unable to extract the determinant from the HDF5 file while performing symmetry analysis for a single-point Q-Chem calculation")
            .map_err(|err| err.to_string())?;
            log::debug!("Extracting canonical determinant information for representation analysis... Done.");

            log::debug!("Running representation analysis on canonical determinant...");
            let mut sda_driver =
                SlaterDeterminantRepAnalysisDriver::<UnitaryRepresentedSymmetryGroup, f64>::builder()
                    .parameters(self.slater_det_rep_analysis_parameters)
                    .angular_function_parameters(self.angular_function_analysis_parameters)
                    .determinant(&det)
                    .sao_spatial(&sao)
                    .symmetry_group(&pd_res)
                    .build()
                    .with_context(|| "Unable to extract a Slater determinant representation analysis driver while performing symmetry analysis for a single-point Q-Chem calculation")
                    .map_err(|err| err.to_string())?;
            let sda_run = sda_driver.run();
            log::debug!("Running representation analysis on canonical determinant... Done.");
            if let Err(err) = sda_run {
                qsym2_error!("Representation analysis has failed with error:");
                qsym2_error!("  {err:#}");
            }

            // let _ = self.extract_determinant(
            //     recentred_mol,
            //     &bao,
            //     self.slater_det_rep_analysis_parameters
            //         .linear_independence_threshold,
            //     OrbitalType::Localised,
            // ).and_then(|loc_det| {
            //     log::debug!("Running representation analysis on localised determinant...");
            //     let mut loc_sda_driver =
            //         SlaterDeterminantRepAnalysisDriver::<UnitaryRepresentedSymmetryGroup, f64>::builder()
            //             .parameters(self.slater_det_rep_analysis_parameters)
            //             .angular_function_parameters(self.angular_function_analysis_parameters)
            //             .determinant(&loc_det)
            //             .sao_spatial(&sao)
            //             .symmetry_group(&pd_res)
            //             .build()?;
            //     let res = loc_sda_driver.run();
            //     log::debug!("Running representation analysis on localised determinant... Done.");
            //     res
            // });

            sda_driver
                .result()
                .map_err(|err| err.to_string())
                .and_then(|sda_res| sda_res.determinant_symmetry().clone())
        };
        self.result = Some((sym, rep()));
        Ok(())
    }
}

// Specific for magnetic-represented symmetry groups and determinant numeric type f64
// ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

impl<'a> QChemH5SinglePointDriver<'a, MagneticRepresentedSymmetryGroup, f64> {
    /// Performs symmetry-group detection and magnetic-represented corepresentation analysis.
    fn analyse(&mut self) -> Result<(), anyhow::Error> {
        let mol = self.extract_molecule()
            .with_context(|| "Unable to extract the molecule from the HDF5 file while performing symmetry analysis for a single-point Q-Chem calculation")?;
        log::debug!("Performing symmetry-group detection...");
        let pd_res = match self.symmetry_group_detection_input {
            SymmetryGroupDetectionInputKind::Parameters(pd_params) => {
                let mut pd_driver = SymmetryGroupDetectionDriver::builder()
                    .parameters(pd_params)
                    .molecule(Some(&mol))
                    .build()
                    .unwrap();
                let pd_run = pd_driver.run();
                if let Err(err) = pd_run {
                    qsym2_error!("Symmetry-group detection has failed with error:");
                    qsym2_error!("  {err:#}");
                }
                let pd_res = pd_driver.result()?;
                pd_res.clone()
            }
            SymmetryGroupDetectionInputKind::FromFile(path) => {
                read_qsym2_binary::<SymmetryGroupDetectionResult, _>(path, QSym2FileType::Sym)
                    .with_context(|| "Unable to read the specified .qsym2.sym file while performing symmetry analysis for a single-point Q-Chem calculation")?
            }
        };
        let recentred_mol = &pd_res.pre_symmetry.recentred_molecule;
        let sym = if self
            .slater_det_rep_analysis_parameters
            .use_magnetic_group
            .is_some()
        {
            pd_res.magnetic_symmetry.clone()
        } else {
            Some(pd_res.unitary_symmetry.clone())
        }
        .ok_or(format_err!("Symmetry not found."))?;
        log::debug!("Performing symmetry-group detection... Done.");

        let rep = || {
            log::debug!("Extracting AO basis information for corepresentation analysis...");
            let sao = self.extract_sao()
                .with_context(|| "Unable to extract the SAO matrix from the HDF5 file while performing symmetry analysis for a single-point Q-Chem calculation")
                .map_err(|err| err.to_string())?;
            let bao = self.extract_bao(recentred_mol)
                .with_context(|| "Unable to extract the basis angular order information from the HDF5 file while performing symmetry analysis for a single-point Q-Chem calculation")
                .map_err(|err| err.to_string())?;
            log::debug!("Extracting AO basis information for corepresentation analysis... Done.");
            log::debug!("Extracting determinant information for corepresentation analysis...");
            let det = self.extract_determinant(
                recentred_mol,
                &bao,
                self.slater_det_rep_analysis_parameters
                    .linear_independence_threshold,
                OrbitalType::Canonical,
            )
            .with_context(|| "Unable to extract the determinant from the HDF5 file while performing symmetry analysis for a single-point Q-Chem calculation")
            .map_err(|err| err.to_string())?;
            log::debug!(
                "Extracting determinant information for corepresentation analysis... Done."
            );

            log::debug!("Running corepresentation analysis...");
            let mut sda_driver =
                SlaterDeterminantRepAnalysisDriver::<MagneticRepresentedSymmetryGroup, f64>::builder()
                    .parameters(self.slater_det_rep_analysis_parameters)
                    .angular_function_parameters(self.angular_function_analysis_parameters)
                    .determinant(&det)
                    .sao_spatial(&sao)
                    .symmetry_group(&pd_res)
                    .build()
                    .with_context(|| "Unable to extract a Slater determinant representation analysis driver while performing symmetry analysis for a single-point Q-Chem calculation")
                    .map_err(|err| err.to_string())?;
            let sda_run = sda_driver.run();
            log::debug!("Running corepresentation analysis... Done.");

            if let Err(err) = sda_run {
                qsym2_error!("Corepresentation analysis has failed with error:");
                qsym2_error!("  {err:#}");
            }
            sda_driver
                .result()
                .map_err(|err| err.to_string())
                .and_then(|sda_res| sda_res.determinant_symmetry().clone())
        };
        self.result = Some((sym, rep()));
        Ok(())
    }
}

// ~~~~~~~~~~~~~~~~~~~~~
// Trait implementations
// ~~~~~~~~~~~~~~~~~~~~~

// Specific for unitary-represented symmetry groups and determinant numeric type f64
// '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

impl<'a> QSym2Driver for QChemH5SinglePointDriver<'a, UnitaryRepresentedSymmetryGroup, f64> {
    type Outcome = (
        Symmetry,
        Result<<<UnitaryRepresentedSymmetryGroup as CharacterProperties>::CharTab as SubspaceDecomposable<f64>>::Decomposition, String>,
    );

    fn result(&self) -> Result<&Self::Outcome, anyhow::Error> {
        self.result
            .as_ref()
            .ok_or(format_err!("No Q-Chem single-point analysis results (unitary-represented group, real determinant) found."))
    }

    fn run(&mut self) -> Result<(), anyhow::Error> {
        self.analyse()
    }
}

// Specific for magnetic-represented symmetry groups and determinant numeric type f64
// ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

impl<'a> QSym2Driver for QChemH5SinglePointDriver<'a, MagneticRepresentedSymmetryGroup, f64> {
    type Outcome = (
        Symmetry,
        Result<<<MagneticRepresentedSymmetryGroup as CharacterProperties>::CharTab as SubspaceDecomposable<f64>>::Decomposition, String>,
    );

    fn result(&self) -> Result<&Self::Outcome, anyhow::Error> {
        self.result
            .as_ref()
            .ok_or(format_err!("No Q-Chem single-point analysis results (magnetic-represented group, real determinant) found."))
    }

    fn run(&mut self) -> Result<(), anyhow::Error> {
        self.analyse()
    }
}
