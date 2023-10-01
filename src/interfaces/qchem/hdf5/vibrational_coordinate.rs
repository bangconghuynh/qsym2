use std::fmt;
use std::marker::PhantomData;
use std::path::PathBuf;

use anyhow::{self, format_err, Context};
use derive_builder::Builder;
use duplicate::duplicate_item;
use hdf5::{self, H5Type};
use lazy_static::lazy_static;
use log;
use nalgebra::Point3;
use ndarray::{Array2, Axis, Ix3, ShapeBuilder};
use ndarray_linalg::types::Lapack;
use num_complex::ComplexFloat;
use numeric_sort;
use periodic_table::periodic_table;
use regex::Regex;

use crate::auxiliary::atom::{Atom, ElementMap};
use crate::auxiliary::molecule::Molecule;
use crate::chartab::SubspaceDecomposable;
use crate::drivers::representation_analysis::angular_function::AngularFunctionRepAnalysisParams;
use crate::drivers::representation_analysis::vibrational_coordinate::{
    VibrationalCoordinateRepAnalysisDriver, VibrationalCoordinateRepAnalysisParams,
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
use crate::target::vibration::VibrationalCoordinateCollection;

#[cfg(test)]
#[path = "vibrational_coordinate_tests.rs"]
mod vibrational_coordinate_tests;

// =====================
// Full Q-Chem H5 Driver
// =====================

lazy_static! {
    static ref VIB_PATH_RE: Regex = Regex::new(
        r"(?<sp_path>.*)energy_function\\(?<energy_function>.*)\\analysis\\vibrational$"
    )
    .expect("Regex pattern invalid.");
}

// -----------------
// Struct definition
// -----------------

/// A driver to perform symmetry-group detection and vibration representation symmetry analysis for
/// all discoverable single-point calculation data stored in a Q-Chem's `qarchive.h5` file.
#[derive(Clone, Builder)]
pub(crate) struct QChemVibrationH5Driver<'a, T>
where
    T: Clone,
{
    /// The `qarchive.h5` file name.
    filename: PathBuf,

    /// The input specification controlling symmetry-group detection.
    symmetry_group_detection_input: &'a SymmetryGroupDetectionInputKind,

    /// The parameters controlling representation analysis of standard angular functions.
    angular_function_analysis_parameters: &'a AngularFunctionRepAnalysisParams,

    /// The parameters controlling representation analysis.
    rep_analysis_parameters: &'a VibrationalCoordinateRepAnalysisParams<f64>,

    /// The simplified result of the analysis. Each element in the vector is a tuple containing the
    /// group name and a message indicating if the analysis has been successful.
    #[builder(default = "None")]
    result: Option<Vec<(String, String)>>,

    /// The numerical type of the vibrational coordinates.
    #[builder(setter(skip), default = "PhantomData")]
    numerical_type: PhantomData<T>,
}

// ----------------------
// Struct implementations
// ----------------------

impl<'a, T> QChemVibrationH5Driver<'a, T>
where
    T: Clone
{
    /// Returns a builder to construct a [`QChemVibrationH5Driver`].
    pub(crate) fn builder() -> QChemVibrationH5DriverBuilder<'a, T> {
        QChemVibrationH5DriverBuilder::default()
    }
}

// Specific for vibrational coordinates numeric type f64
// '''''''''''''''''''''''''''''''''''''''''''''''''''''
impl<'a> QChemVibrationH5Driver<'a, f64> {
    /// Performs analysis for all real-valued single-point vibrational coordinates.
    fn analyse(&mut self) -> Result<(), anyhow::Error> {
        let f = hdf5::File::open(&self.filename)?;
        let mut sp_paths = f
            .group(".counters")?
            .member_names()?
            .iter()
            .filter_map(|path| {
                if let Some(caps) = VIB_PATH_RE.captures(path) {
                    let sp_path = caps["sp_path"].replace("\\", "/");
                    let energy_function_index = caps["energy_function"].to_string();
                    Some((sp_path, energy_function_index))
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        sp_paths.sort_by(|(path_a, _), (path_b, _)| numeric_sort::cmp(path_a, path_b));

        let pd_input = self.symmetry_group_detection_input;
        let afa_params = self.angular_function_analysis_parameters;
        let vca_params = self.rep_analysis_parameters;
        let result = sp_paths
            .iter()
            .map(|(sp_path, energy_function_index)| {
                log_macsec_begin(&format!(
                    "Analysis for {} (energy function {energy_function_index})",
                    sp_path.clone()
                ));
                qsym2_output!("");
                let sp = f.group(sp_path)?;
                let sp_driver_result = match vca_params.use_magnetic_group {
                    Some(MagneticSymmetryAnalysisKind::Corepresentation) => {
                        let mut sp_driver = QChemVibrationH5SinglePointDriver::<
                            MagneticRepresentedSymmetryGroup,
                            f64,
                        >::builder()
                        .sp_group(&sp)
                        .energy_function_index(energy_function_index)
                        .symmetry_group_detection_input(pd_input)
                        .angular_function_analysis_parameters(afa_params)
                        .rep_analysis_parameters(vca_params)
                        .build()?;
                        let _ = sp_driver.run();
                        sp_driver.result().map(|(sym, vca_res)| {
                            (
                                sym.group_name
                                    .as_ref()
                                    .unwrap_or(&String::new())
                                    .to_string(),
                                vca_res
                                    .as_ref()
                                    .map(|_| "Ok".to_string())
                                    .unwrap_or_else(|err| err.to_string()),
                            )
                        })
                    }
                    Some(MagneticSymmetryAnalysisKind::Representation) | None => {
                        let mut sp_driver = QChemVibrationH5SinglePointDriver::<
                            UnitaryRepresentedSymmetryGroup,
                            f64,
                        >::builder()
                        .sp_group(&sp)
                        .energy_function_index(energy_function_index)
                        .symmetry_group_detection_input(pd_input)
                        .angular_function_analysis_parameters(afa_params)
                        .rep_analysis_parameters(vca_params)
                        .build()?;
                        let _ = sp_driver.run();
                        sp_driver.result().map(|(sym, vca_res)| {
                            (
                                sym.group_name
                                    .as_ref()
                                    .unwrap_or(&String::new())
                                    .to_string(),
                                vca_res
                                    .as_ref()
                                    .map(|_| "Ok".to_string())
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
            .map(|res| {
                res.unwrap_or_else(|err| {
                    (
                        "Unidentified symmetry group".to_string(),
                        format!("Unidentified (co)representations: {err}"),
                    )
                })
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
            .map(|(_, energy_function_index)| energy_function_index.chars().count().max(1))
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
            .unwrap_or(20)
            .max(20);
        let table_width = path_length + energy_function_length + group_length + sym_length + 8;
        qsym2_output!("{}", "┈".repeat(table_width));
        qsym2_output!(
            " {:<path_length$}  {:<energy_function_length$}  {:<group_length$}  {:<}",
            "Single-point calc.",
            "E func.",
            "Group",
            "Vib. symmetry status"
        );
        qsym2_output!("{}", "┈".repeat(table_width));
        sp_paths
            .iter()
            .map(|(sp_path, energy_function_index)| (sp_path.clone(), energy_function_index))
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

impl<'a> QSym2Driver for QChemVibrationH5Driver<'a, f64> {
    type Params = VibrationalCoordinateRepAnalysisParams<f64>;

    type Outcome = Vec<(String, String)>;

    fn result(&self) -> Result<&Self::Outcome, anyhow::Error> {
        self.result.as_ref().ok_or(format_err!(
            "No Q-Chem HDF5 analysis results for real vibrational coordinates found."
        ))
    }

    fn run(&mut self) -> Result<(), anyhow::Error> {
        self.analyse()
    }
}

// ==================
// SinglePoint Driver
// ==================

// -----------------
// Struct definition
// -----------------

/// A driver to perform symmetry-group detection and vibration representation analysis for a
/// single-point calculation result in a Q-Chem's `qarchive.h5` file.
#[derive(Clone, Builder)]
struct QChemVibrationH5SinglePointDriver<'a, G, T>
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

    /// The parameters controlling the representation analysis of vibrational coordinates.
    rep_analysis_parameters: &'a VibrationalCoordinateRepAnalysisParams<f64>,

    #[builder(setter(skip), default = "PhantomData")]
    group_type: PhantomData<G>,

    #[builder(setter(skip), default = "PhantomData")]
    numerical_type: PhantomData<T>,

    /// The symmetry of the system.
    #[builder(default = "None")]
    result: Option<(Symmetry, Result<Vec<String>, String>)>,
}

// ----------------------
// Struct implementations
// ----------------------

impl<'a, G, T> QChemVibrationH5SinglePointDriverBuilder<'a, G, T>
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

impl<'a, G, T> QChemVibrationH5SinglePointDriver<'a, G, T>
where
    G: SymmetryGroupProperties + Clone,
    G::CharTab: SubspaceDecomposable<T>,
    T: ComplexFloat + Lapack + H5Type,
    <T as ComplexFloat>::Real: From<f64> + fmt::LowerExp + fmt::Debug,
{
    /// Returns a builder to construct a [`QChemVibrationH5SinglePointDriver`].
    fn builder() -> QChemVibrationH5SinglePointDriverBuilder<'a, G, T> {
        QChemVibrationH5SinglePointDriverBuilder::default()
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
}

// Generic for all symmetry groups G and vibrational coordinates numeric type T
// ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
impl<'a, G, T> QChemVibrationH5SinglePointDriver<'a, G, T>
where
    G: SymmetryGroupProperties + Clone,
    G::CharTab: SubspaceDecomposable<T>,
    T: ComplexFloat + Lapack + H5Type,
    <T as ComplexFloat>::Real: From<f64> + fmt::LowerExp + fmt::Debug,
{
    /// Extracts the vibrational coordinate collection from the single-point H5 group.
    ///
    /// # Arguments
    ///
    /// * `mol` - The molecule to be associated with the extracted vibrational coordinate
    /// collection.
    /// * `threshold` - The comparison threshold to be associated with the extracted vibrational
    /// coordinate collection.
    ///
    /// # Returns
    ///
    /// The extracted vibrational coordinate collection.
    fn extract_vibrational_coordinate_collection(
        &self,
        mol: &'a Molecule,
        threshold: <T as ComplexFloat>::Real,
    ) -> Result<VibrationalCoordinateCollection<'a, T>, anyhow::Error> {
        let frequencies = self
            .sp_group
            .dataset(&format!(
                "energy_function/{}/analysis/vibrational/1/frequencies",
                self.energy_function_index
            ))?
            .read_1d::<T>()
            .map_err(|err| format_err!(err))?;
        let natoms = self
            .sp_group
            .dataset(&format!(
                "energy_function/{}/analysis/vibrational/1/natoms",
                self.energy_function_index
            ))?
            .read_scalar::<usize>()
            .map_err(|err| format_err!(err))?;
        let nmodes = self
            .sp_group
            .dataset(&format!(
                "energy_function/{}/analysis/vibrational/1/nmodes",
                self.energy_function_index
            ))?
            .read_scalar::<usize>()
            .map_err(|err| format_err!(err))?;
        let coefficients = Array2::from_shape_vec(
            (3 * natoms, nmodes).f(),
            self.sp_group
                .dataset(&format!(
                    "energy_function/{}/analysis/vibrational/1/modes",
                    self.energy_function_index
                ))?
                .read::<T, Ix3>()?
                .axis_iter(Axis(0))
                .flatten()
                .cloned()
                .collect::<Vec<_>>(),
        )
        .map_err(|err| format_err!(err))?;

        VibrationalCoordinateCollection::builder()
            .mol(mol)
            .frequencies(frequencies)
            .coefficients(coefficients)
            .threshold(threshold)
            .build()
            .map_err(|err| err.into())
    }
}

// Specific for unitary-represented and magnetic-represented symmetry groups and determinant numeric type f64
// ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#[duplicate_item(
    [
        gtype_ [ UnitaryRepresentedSymmetryGroup ]
        doc_ [ "Performs symmetry-group detection and unitary-represented representation analysis." ]
    ]
    [
        gtype_ [ MagneticRepresentedSymmetryGroup ]
        doc_ [ "Performs symmetry-group detection and magnetic-represented corepresentation analysis." ]
    ]
)]
impl<'a> QChemVibrationH5SinglePointDriver<'a, gtype_, f64> {
    #[doc = doc_]
    fn analyse(&mut self) -> Result<(), anyhow::Error> {
        let mol = self.extract_molecule()
            .with_context(|| "Unable to extract the molecule from the HDF5 file while performing symmetry analysis for a single-point Q-Chem calculation")?;
        log::debug!("Performing symmetry-group detection...");
        let pd_res = match self.symmetry_group_detection_input {
            SymmetryGroupDetectionInputKind::Parameters(pd_params) => {
                let mut pd_driver = SymmetryGroupDetectionDriver::builder()
                    .parameters(pd_params)
                    .molecule(Some(&mol))
                    .build()?;
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
        let sym = if self.rep_analysis_parameters.use_magnetic_group.is_some() {
            pd_res.magnetic_symmetry.clone()
        } else {
            Some(pd_res.unitary_symmetry.clone())
        }
        .ok_or(format_err!("Symmetry not found."))?;
        log::debug!("Performing symmetry-group detection... Done.");

        let rep = || {
            log::debug!(
                "Extracting vibrational coordinate information for representation analysis..."
            );
            let vibs = self.extract_vibrational_coordinate_collection(
                recentred_mol,
                self.rep_analysis_parameters
                    .linear_independence_threshold,
            )
            .with_context(|| "Unable to extract the vibrational coordinate collection from the HDF5 file while performing symmetry analysis for a single-point Q-Chem calculation")
            .map_err(|err| err.to_string())?;
            log::debug!(
                "Extracting vibrational coordinate information for representation analysis... Done."
            );

            log::debug!("Running representation analysis on vibrational coordinate collection...");
            let mut vca_driver =
                VibrationalCoordinateRepAnalysisDriver::<gtype_, f64>::builder()
                    .parameters(self.rep_analysis_parameters)
                    .angular_function_parameters(self.angular_function_analysis_parameters)
                    .vibrational_coordinate_collection(&vibs)
                    .symmetry_group(&pd_res)
                    .build()
                    .with_context(|| "Unable to construct a vibrational coordinate representation analysis driver while performing symmetry analysis for a single-point Q-Chem calculation")
                    .map_err(|err| err.to_string())?;
            let vca_run = vca_driver.run();
            qsym2_output!("");
            log::debug!(
                "Running representation analysis on vibrational coordinate collection... Done."
            );
            if let Err(err) = vca_run {
                qsym2_error!("Representation analysis has failed with error:");
                qsym2_error!("  {err:#}");
            }

            vca_driver
                .result()
                .map(|vca_res| {
                    vca_res
                        .vibrational_coordinate_symmetries()
                        .iter()
                        .map(|vc_res| {
                            vc_res
                                .as_ref()
                                .map(|vc_sym| vc_sym.to_string())
                                .unwrap_or_else(|err| err.clone())
                        })
                        .collect::<Vec<_>>()
                })
                .map_err(|err| err.to_string())
        };
        self.result = Some((sym, rep()));
        Ok(())
    }
}

// ---------------------
// Trait implementations
// ---------------------

// ~~~~~~~~~~~~~~~~~~~
// Slater determinants
// ~~~~~~~~~~~~~~~~~~~

// Specific for unitary-represented and magnetic-represented symmetry groups and determinant numeric type f64
// ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#[duplicate_item(
    [
        gtype_ [ UnitaryRepresentedSymmetryGroup ]
        err_ [ "No Q-Chem single-point analysis results (unitary-represented group, real vibrational coordinates) found." ]
    ]
    [
        gtype_ [ MagneticRepresentedSymmetryGroup ]
        err_ [ "No Q-Chem single-point analysis results (magnetic-represented group, real vibrational coordinates) found." ]
    ]
)]
impl<'a> QSym2Driver for QChemVibrationH5SinglePointDriver<'a, gtype_, f64> {
    type Params = VibrationalCoordinateRepAnalysisParams<f64>;

    type Outcome = (Symmetry, Result<Vec<String>, String>);

    fn result(&self) -> Result<&Self::Outcome, anyhow::Error> {
        self.result.as_ref().ok_or(format_err!(err_))
    }

    fn run(&mut self) -> Result<(), anyhow::Error> {
        self.analyse()
    }
}
