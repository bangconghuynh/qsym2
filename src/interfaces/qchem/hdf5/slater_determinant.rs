//! Slater determinants from Q-Chem HDF5 archives.

use std::fmt;
use std::marker::PhantomData;
use std::path::PathBuf;

use anyhow::{self, Context, bail, format_err};
use derive_builder::Builder;
use duplicate::duplicate_item;
use factorial::DoubleFactorial;
use hdf5::{self, H5Type};
use lazy_static::lazy_static;
use log;
use nalgebra::Point3;
use ndarray::{Array1, Array2, Array4, Axis, Ix3, s};
use ndarray_linalg::types::Lapack;
use num_complex::ComplexFloat;
use num_traits::{One, ToPrimitive, Zero};
use numeric_sort;
use periodic_table::periodic_table;
use regex::Regex;

use crate::angmom::spinor_rotation_3d::{SpinConstraint, StructureConstraint};
use crate::auxiliary::atom::{Atom, ElementMap};
use crate::auxiliary::molecule::Molecule;
use crate::basis::ao::*;
use crate::basis::ao_integrals::*;
use crate::chartab::SubspaceDecomposable;
use crate::chartab::chartab_group::CharacterProperties;
use crate::drivers::QSym2Driver;
use crate::drivers::representation_analysis::MagneticSymmetryAnalysisKind;
use crate::drivers::representation_analysis::angular_function::AngularFunctionRepAnalysisParams;
use crate::drivers::representation_analysis::slater_determinant::{
    SlaterDeterminantRepAnalysisDriver, SlaterDeterminantRepAnalysisParams,
};
use crate::drivers::symmetry_group_detection::{
    SymmetryGroupDetectionDriver, SymmetryGroupDetectionResult,
};
#[cfg(feature = "integrals")]
use crate::integrals::shell_tuple::build_shell_tuple_collection;
use crate::interfaces::input::SymmetryGroupDetectionInputKind;
use crate::io::format::{
    log_macsec_begin, log_macsec_end, log_micsec_begin, log_micsec_end, qsym2_error, qsym2_output,
};
use crate::io::{QSym2FileType, read_qsym2_binary};
use crate::symmetry::symmetry_core::Symmetry;
use crate::symmetry::symmetry_group::{
    MagneticRepresentedSymmetryGroup, SymmetryGroupProperties, UnitaryRepresentedSymmetryGroup,
};
use crate::target::determinant::SlaterDeterminant;

#[cfg(test)]
#[path = "slater_determinant_tests.rs"]
mod slater_determinant_tests;

// =====================
// Full Q-Chem H5 Driver
// =====================

lazy_static! {
    static ref SP_PATH_RE: Regex =
        Regex::new(r"(.*sp)\\energy_function$").expect("Regex pattern invalid.");
}

// -----------------
// Struct definition
// -----------------

/// Driver to perform symmetry-group detection and Slater determinant representation symmetry
/// analysis for all discoverable single-point calculation data stored in a Q-Chem's `qarchive.h5`
/// file.
#[derive(Clone, Builder)]
pub struct QChemSlaterDeterminantH5Driver<'a, T>
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
    rep_analysis_parameters: &'a SlaterDeterminantRepAnalysisParams<f64>,

    /// The simplified result of the analysis. Each element in the vector is a tuple containing the
    /// group name and the representation symmetry of the Slater determinant for one single-point
    /// calculation.
    #[builder(default = "None")]
    result: Option<Vec<(String, String)>>,

    /// The numerical type of the Slater determinant.
    #[builder(setter(skip), default = "PhantomData")]
    numerical_type: PhantomData<T>,
}

// ----------------------
// Struct implementations
// ----------------------

impl<'a, T> QChemSlaterDeterminantH5Driver<'a, T>
where
    T: Clone,
{
    /// Returns a builder to construct a [`QChemSlaterDeterminantH5Driver`].
    pub fn builder() -> QChemSlaterDeterminantH5DriverBuilder<'a, T> {
        QChemSlaterDeterminantH5DriverBuilder::default()
    }
}

// ~~~~~~~~~~~~~~~~~~~
// Slater determinants
// ~~~~~~~~~~~~~~~~~~~

// Specific for Slater determinant numeric type f64
// ''''''''''''''''''''''''''''''''''''''''''''''''
impl<'a> QChemSlaterDeterminantH5Driver<'a, f64> {
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
        let sda_params = self.rep_analysis_parameters;
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
                            let mut sp_driver = QChemSlaterDeterminantH5SinglePointDriver::<
                                MagneticRepresentedSymmetryGroup,
                                f64,
                            >::builder()
                            .sp_group(&sp)
                            .energy_function_index(energy_function_index)
                            .symmetry_group_detection_input(pd_input)
                            .angular_function_analysis_parameters(afa_params)
                            .rep_analysis_parameters(sda_params)
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
                            let mut sp_driver = QChemSlaterDeterminantH5SinglePointDriver::<
                                UnitaryRepresentedSymmetryGroup,
                                f64,
                            >::builder()
                            .sp_group(&sp)
                            .energy_function_index(energy_function_index)
                            .symmetry_group_detection_input(pd_input)
                            .angular_function_analysis_parameters(afa_params)
                            .rep_analysis_parameters(sda_params)
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
                res.unwrap_or_else(|err| {
                    (
                        "Unidentified symmetry group".to_string(),
                        format!("Unidentified (co)representation: {err}"),
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

impl<'a> QSym2Driver for QChemSlaterDeterminantH5Driver<'a, f64> {
    type Params = SlaterDeterminantRepAnalysisParams<f64>;

    type Outcome = Vec<(String, String)>;

    fn result(&self) -> Result<&Self::Outcome, anyhow::Error> {
        self.result.as_ref().ok_or(format_err!(
            "No Q-Chem HDF5 analysis results for a real Slater determinant found."
        ))
    }

    fn run(&mut self) -> Result<(), anyhow::Error> {
        self.analyse()
    }
}

// ==================
// SinglePoint Driver
// ==================

// ---------------
// Enum definition
// ---------------

/// Enumerated type to distinguish different kinds of molecular orbitals.
pub enum OrbitalType {
    /// Canonical molecular orbitals as obtained by diagonalising Fock matrices.
    Canonical,

    /// Localised molecular orbitals as obtained by a localisation method.
    Localised,
}

// -----------------
// Struct definition
// -----------------

/// Driver to perform symmetry-group detection and representation analysis for a single-point
/// calculation result in a Q-Chem's `qarchive.h5` file.
#[derive(Clone, Builder)]
pub struct QChemSlaterDeterminantH5SinglePointDriver<'a, G, T>
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
    rep_analysis_parameters: &'a SlaterDeterminantRepAnalysisParams<f64>,

    /// The symmetry of the system and the representation of the Slater determinant.
    #[allow(clippy::type_complexity)]
    #[builder(default = "None")]
    result: Option<(
        Symmetry,
        Result<<G::CharTab as SubspaceDecomposable<T>>::Decomposition, String>,
    )>,
}

// ----------------------
// Struct implementations
// ----------------------

impl<'a, G, T> QChemSlaterDeterminantH5SinglePointDriverBuilder<'a, G, T>
where
    G: SymmetryGroupProperties + Clone,
    G::CharTab: SubspaceDecomposable<T>,
    T: ComplexFloat + Lapack,
    <T as ComplexFloat>::Real: From<f64> + fmt::LowerExp + fmt::Debug,
{
    pub fn energy_function_index(&mut self, idx: &str) -> &mut Self {
        self.energy_function_index = Some(idx.to_string());
        self
    }
}

impl<'a, G, T> QChemSlaterDeterminantH5SinglePointDriver<'a, G, T>
where
    G: SymmetryGroupProperties + Clone,
    G::CharTab: SubspaceDecomposable<T>,
    T: ComplexFloat + Lapack + H5Type,
    <T as ComplexFloat>::Real: From<f64> + fmt::LowerExp + fmt::Debug,
{
    /// Returns a builder to construct a [`QChemSlaterDeterminantH5SinglePointDriver`].
    pub fn builder() -> QChemSlaterDeterminantH5SinglePointDriverBuilder<'a, G, T> {
        QChemSlaterDeterminantH5SinglePointDriverBuilder::default()
    }

    /// Extracts the molecular structure from the single-point H5 group.
    pub fn extract_molecule(&self) -> Result<Molecule, anyhow::Error> {
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

    /// Extracts the basis angular order information from the single-point H5 group.
    pub fn extract_bao(&self, mol: &'a Molecule) -> Result<BasisAngularOrder<'a>, anyhow::Error> {
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

    /// Extracts the spatial atomic-orbital overlap matrix from the single-point H5 group.
    ///
    /// Note that the overlap matrix in the HDF5 file uses lexicographic order for Cartesian
    /// functions. This is inconsistent with the conventional Q-Chem ordering used for molecular
    /// orbital coefficients. See [`Self::recompute_sao`] for a way to get the atomic-orbital
    /// overlap matrix with the consistent Cartesian ordering.
    pub fn extract_sao(&self) -> Result<Array2<T>, anyhow::Error> {
        self.sp_group
            .dataset("aobasis/overlap_matrix")?
            .read_2d::<T>()
            .map_err(|err| err.into())
    }

    /// Extracts the full basis set information from the single-point H5 group.
    pub fn extract_basis_set(
        &self,
        mol: &'a Molecule,
    ) -> Result<BasisSet<f64, f64>, anyhow::Error> {
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

        let primitives_per_shell = self
            .sp_group
            .dataset("aobasis/primitives_per_shell")?
            .read_1d::<usize>()?;
        let contraction_coefficients = self
            .sp_group
            .dataset("aobasis/contraction_coefficients")?
            .read_1d::<f64>()?;
        let sp_contraction_coefficients = self
            .sp_group
            .dataset("aobasis/sp_contraction_coefficients")?
            .read_1d::<f64>()?;
        let primitive_exponents = self
            .sp_group
            .dataset("aobasis/primitive_exponents")?
            .read_1d::<f64>()?;
        // `shell_coordinates` in Bohr radius.
        let shell_coordinates = self
            .sp_group
            .dataset("aobasis/shell_coordinates")?
            .read_2d::<f64>()?;

        let bscs: Vec<BasisShellContraction<f64, f64>> = primitives_per_shell
            .iter()
            .scan(0, |end, n_prims| {
                let start = *end;
                *end += n_prims;
                Some((start, *end))
            })
            .zip(shell_types.iter())
            .zip(shell_coordinates.rows())
            .flat_map(|(((start, end), shell_type), centre)| {
                if *shell_type == 0 {
                    // S shell
                    let basis_shell = BasisShell::new(0, ShellOrder::Cart(CartOrder::qchem(0)));
                    let primitives = primitive_exponents
                        .slice(s![start..end])
                        .iter()
                        .cloned()
                        .zip(
                            contraction_coefficients
                                .slice(s![start..end])
                                .iter()
                                .cloned(),
                        )
                        .collect::<Vec<_>>();
                    let contraction = GaussianContraction { primitives };
                    let cart_origin = Point3::new(centre[0], centre[1], centre[2]);
                    vec![BasisShellContraction {
                        basis_shell,
                        contraction,
                        cart_origin,
                        k: None,
                    }]
                } else if *shell_type == 1 {
                    // P shell
                    let basis_shell = BasisShell::new(1, ShellOrder::Cart(CartOrder::qchem(1)));
                    let primitives = primitive_exponents
                        .slice(s![start..end])
                        .iter()
                        .cloned()
                        .zip(
                            contraction_coefficients
                                .slice(s![start..end])
                                .iter()
                                .cloned(),
                        )
                        .collect::<Vec<_>>();
                    let contraction = GaussianContraction { primitives };
                    let cart_origin = Point3::new(centre[0], centre[1], centre[2]);
                    vec![BasisShellContraction {
                        basis_shell,
                        contraction,
                        cart_origin,
                        k: None,
                    }]
                } else if *shell_type == -1 {
                    // SP shell
                    let basis_shell_s = BasisShell::new(0, ShellOrder::Cart(CartOrder::qchem(0)));
                    let primitives_s = primitive_exponents
                        .slice(s![start..end])
                        .iter()
                        .cloned()
                        .zip(
                            contraction_coefficients
                                .slice(s![start..end])
                                .iter()
                                .cloned(),
                        )
                        .collect::<Vec<_>>();
                    let contraction_s = GaussianContraction {
                        primitives: primitives_s,
                    };

                    let basis_shell_p = BasisShell::new(1, ShellOrder::Cart(CartOrder::qchem(1)));
                    let primitives_p = primitive_exponents
                        .slice(s![start..end])
                        .iter()
                        .cloned()
                        .zip(
                            sp_contraction_coefficients
                                .slice(s![start..end])
                                .iter()
                                .cloned(),
                        )
                        .collect::<Vec<_>>();
                    let contraction_p = GaussianContraction {
                        primitives: primitives_p,
                    };

                    let cart_origin = Point3::new(centre[0], centre[1], centre[2]);
                    vec![
                        BasisShellContraction {
                            basis_shell: basis_shell_s,
                            contraction: contraction_s,
                            cart_origin,
                            k: None,
                        },
                        BasisShellContraction {
                            basis_shell: basis_shell_p,
                            contraction: contraction_p,
                            cart_origin,
                            k: None,
                        },
                    ]
                } else if *shell_type < 0 {
                    // Cartesian D shell or higher
                    let l = shell_type.unsigned_abs();
                    let basis_shell = BasisShell::new(l, ShellOrder::Cart(CartOrder::qchem(l)));
                    let primitives = primitive_exponents
                        .slice(s![start..end])
                        .iter()
                        .cloned()
                        .zip(
                            contraction_coefficients
                                .slice(s![start..end])
                                .iter()
                                .cloned(),
                        )
                        .collect::<Vec<_>>();
                    let contraction = GaussianContraction { primitives };
                    let cart_origin = Point3::new(centre[0], centre[1], centre[2]);
                    vec![BasisShellContraction {
                        basis_shell,
                        contraction,
                        cart_origin,
                        k: None,
                    }]
                } else {
                    // Pure D shell or higher
                    let l = shell_type.unsigned_abs();
                    let basis_shell =
                        BasisShell::new(l, ShellOrder::Pure(PureOrder::increasingm(l)));
                    let primitives = primitive_exponents
                        .slice(s![start..end])
                        .iter()
                        .cloned()
                        .zip(
                            contraction_coefficients
                                .slice(s![start..end])
                                .iter()
                                .cloned(),
                        )
                        .collect::<Vec<_>>();
                    let contraction = GaussianContraction { primitives };
                    let cart_origin = Point3::new(centre[0], centre[1], centre[2]);
                    vec![BasisShellContraction {
                        basis_shell,
                        contraction,
                        cart_origin,
                        k: None,
                    }]
                }
            })
            .collect::<Vec<BasisShellContraction<f64, f64>>>();

        let basis_atoms = mol
            .atoms
            .iter()
            .enumerate()
            .map(|(atom_i, _)| {
                bscs
                    .iter()
                    .zip(shell_to_atom_map.iter())
                    .filter_map(|(bs, atom_index)| {
                        if *atom_index == atom_i {
                            Some(bs.clone())
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<Vec<BasisShellContraction<f64, f64>>>>();

        let mut basis_set = BasisSet::<f64, f64>::new(basis_atoms);

        // Q-Chem renormalises each Gaussian primitive, but this is not the convention used in
        // QSym². We therefore un-normalise the Gaussian primitives to restore the original
        // contraction coefficients.
        let prefactor = (2.0 / std::f64::consts::PI).powf(0.75);
        basis_set.all_shells_mut().for_each(|bsc| {
            let l = bsc.basis_shell().l;
            let l_i32 = l
                .to_i32()
                .unwrap_or_else(|| panic!("Unable to convert `{l}` to `i32`."));
            let l_f64 = l
                .to_f64()
                .unwrap_or_else(|| panic!("Unable to convert `{l}` to `f64`."));
            let doufac_sqrt = if l == 0 {
                1
            } else {
                ((2 * l) - 1)
                    .checked_double_factorial()
                    .unwrap_or_else(|| panic!("Unable to obtain `{}!!`.", 2 * l - 1))
            }
            .to_f64()
            .unwrap_or_else(|| panic!("Unable to convert `{}!!` to `f64`.", 2 * l - 1))
            .sqrt();
            bsc.contraction.primitives.iter_mut().for_each(|(a, c)| {
                let n = prefactor * 2.0.powi(l_i32) * a.powf(l_f64 / 2.0 + 0.75) / doufac_sqrt;
                *c /= n;
            });
        });
        Ok(basis_set)
    }

    /// Recomputes the spatial atomic-orbital overlap matrix.
    ///
    /// The overlap matrix stored in the H5 group unfortunately uses lexicographic order for
    /// Cartesian functions, which is inconsistent with that used in the coefficients. We thus
    /// recompute the overlap matrix from the basis set information using the conventional Q-Chem
    /// order for Cartesian functions.
    pub fn recompute_sao(&self) -> Result<Array2<f64>, anyhow::Error> {
        log::debug!("Recomputing atomic-orbital overlap matrix...");
        let mol = self.extract_molecule()?;
        let basis_set = self.extract_basis_set(&mol)?;
        let stc = build_shell_tuple_collection![
            <s1, s2>;
            false, false;
            &basis_set, &basis_set;
            f64
        ];
        let sao_res = stc
            .overlap([0, 0])
            .pop()
            .ok_or(format_err!("Unable to compute the AO overlap matrix."));
        log::debug!("Recomputing atomic-orbital overlap matrix... Done.");
        sao_res
    }
}

// ~~~~~~~~~~~~~~~~~~~
// Slater determinants
// ~~~~~~~~~~~~~~~~~~~

// Generic for all symmetry groups G and determinant numeric type T
// ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
impl<'a, G, T> QChemSlaterDeterminantH5SinglePointDriver<'a, G, T>
where
    G: SymmetryGroupProperties + Clone,
    G::CharTab: SubspaceDecomposable<T>,
    T: ComplexFloat + Lapack + H5Type,
    <T as ComplexFloat>::Real: From<f64> + fmt::LowerExp + fmt::Debug,
{
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
    pub fn extract_determinant(
        &self,
        mol: &'a Molecule,
        bao: &'a BasisAngularOrder,
        threshold: <T as ComplexFloat>::Real,
        orbital_type: OrbitalType,
    ) -> Result<SlaterDeterminant<'a, T, SpinConstraint>, anyhow::Error> {
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

        let ncomps = spincons.n_explicit_comps_per_coefficient_matrix();
        SlaterDeterminant::builder()
            .structure_constraint(spincons)
            .baos((0..ncomps).map(|_| bao).collect::<Vec<_>>())
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
impl<'a> QChemSlaterDeterminantH5SinglePointDriver<'a, gtype_, f64> {
    #[doc = doc_]
    pub fn analyse(&mut self) -> Result<(), anyhow::Error> {
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
        let sym = if self.rep_analysis_parameters.use_magnetic_group.is_some() {
            pd_res.magnetic_symmetry.clone()
        } else {
            Some(pd_res.unitary_symmetry.clone())
        }
        .ok_or(format_err!("Symmetry not found."))?;
        log::debug!("Performing symmetry-group detection... Done.");

        let rep = || {
            log::debug!("Extracting AO basis information for representation analysis...");
            let sao = self.recompute_sao()
                .with_context(|| "Unable to extract the SAO matrix from the HDF5 file while performing symmetry analysis for a single-point Q-Chem calculation")
                .map_err(|err| err.to_string())?;
            let bao = self.extract_bao(recentred_mol)
                .with_context(|| "Unable to extract the basis angular order information from the HDF5 file while performing symmetry analysis for a single-point Q-Chem calculation")
                .map_err(|err| err.to_string())?;
            let basis_set_opt = if self.rep_analysis_parameters.analyse_density_symmetries {
                self.extract_basis_set(recentred_mol).ok()
            } else {
                None
            };
            log::debug!("Extracting AO basis information for representation analysis... Done.");

            #[cfg(feature = "integrals")]
            let sao_4c: Option<Array4<f64>> = basis_set_opt.map(|basis_set| {
                log::debug!(
                    "Computing four-centre overlap integrals for density symmetry analysis..."
                );
                let stc = build_shell_tuple_collection![
                    <s1, s2, s3, s4>;
                    false, false, false, false;
                    &basis_set, &basis_set, &basis_set, &basis_set;
                    f64
                ];
                let sao_4c = stc
                    .overlap([0, 0, 0, 0])
                    .pop()
                    .expect("Unable to retrieve the four-centre overlap tensor.");
                log::debug!(
                    "Computing four-centre overlap integrals for density symmetry analysis... Done."
                );
                sao_4c
            });

            #[cfg(not(feature = "integrals"))]
            let sao_4c: Option<Array4<f64>> = None;

            log::debug!(
                "Extracting canonical determinant information for representation analysis..."
            );
            let det = self.extract_determinant(
                recentred_mol,
                &bao,
                self.rep_analysis_parameters
                    .linear_independence_threshold,
                OrbitalType::Canonical,
            )
            .with_context(|| "Unable to extract the determinant from the HDF5 file while performing symmetry analysis for a single-point Q-Chem calculation")
            .map_err(|err| err.to_string())?;
            log::debug!(
                "Extracting canonical determinant information for representation analysis... Done."
            );

            log::debug!("Running representation analysis on canonical determinant...");
            let mut sda_driver =
                SlaterDeterminantRepAnalysisDriver::<gtype_, f64, SpinConstraint>::builder()
                    .parameters(self.rep_analysis_parameters)
                    .angular_function_parameters(self.angular_function_analysis_parameters)
                    .determinant(&det)
                    .sao(&sao)
                    .sao_spatial_4c(sao_4c.as_ref())
                    .symmetry_group(&pd_res)
                    .build()
                    .with_context(|| "Unable to construct a Slater determinant representation analysis driver while performing symmetry analysis for a single-point Q-Chem calculation")
                    .map_err(|err| err.to_string())?;
            log_micsec_begin("Canonical orbital representation analysis");
            let sda_run = sda_driver.run();
            log_micsec_end("Canonical orbital representation analysis");
            qsym2_output!("");
            log::debug!("Running representation analysis on canonical determinant... Done.");
            if let Err(err) = sda_run {
                qsym2_error!("Representation analysis has failed with error:");
                qsym2_error!("  {err:#}");
            }

            let _ = self
                .extract_determinant(
                    recentred_mol,
                    &bao,
                    self.rep_analysis_parameters.linear_independence_threshold,
                    OrbitalType::Localised,
                )
                .and_then(|loc_det| {
                    log::debug!("Running representation analysis on localised determinant...");
                    let mut loc_sda_driver = SlaterDeterminantRepAnalysisDriver::<
                        UnitaryRepresentedSymmetryGroup,
                        f64,
                        SpinConstraint,
                    >::builder()
                    .parameters(self.rep_analysis_parameters)
                    .angular_function_parameters(self.angular_function_analysis_parameters)
                    .determinant(&loc_det)
                    .sao(&sao)
                    .sao_spatial_4c(sao_4c.as_ref())
                    .symmetry_group(&pd_res)
                    .build()?;
                    log_micsec_begin("Localised orbital representation analysis");
                    let res = loc_sda_driver.run();
                    log_micsec_end("Localised orbital representation analysis");
                    qsym2_output!("");
                    log::debug!(
                        "Running representation analysis on localised determinant... Done."
                    );
                    res
                });

            sda_driver
                .result()
                .map_err(|err| err.to_string())
                .and_then(|sda_res| sda_res.determinant_symmetry().clone())
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
        err_ [ "No Q-Chem single-point analysis results (unitary-represented group, real determinant) found." ]
    ]
    [
        gtype_ [ MagneticRepresentedSymmetryGroup ]
        err_ [ "No Q-Chem single-point analysis results (magnetic-represented group, real determinant) found." ]
    ]
)]
impl<'a> QSym2Driver for QChemSlaterDeterminantH5SinglePointDriver<'a, gtype_, f64> {
    type Params = SlaterDeterminantRepAnalysisParams<f64>;

    type Outcome = (
        Symmetry,
        Result<
            <<gtype_ as CharacterProperties>::CharTab as SubspaceDecomposable<f64>>::Decomposition,
            String,
        >,
    );

    fn result(&self) -> Result<&Self::Outcome, anyhow::Error> {
        self.result.as_ref().ok_or(format_err!(err_))
    }

    fn run(&mut self) -> Result<(), anyhow::Error> {
        self.analyse()
    }
}
