use std::fmt;

use anyhow::{self, bail, format_err};
use derive_builder::Builder;
use hdf5::{self, H5Type};
use lazy_static::lazy_static;
use log;
use nalgebra::Point3;
use ndarray::{Array1, Array2, Axis, Ix3};
use ndarray_linalg::types::Lapack;
use num_complex::ComplexFloat;
use num_traits::{One, Zero};
use periodic_table::periodic_table;
use regex::Regex;

use crate::angmom::spinor_rotation_3d::SpinConstraint;
use crate::aux::ao_basis::BasisAngularOrder;
use crate::aux::ao_basis::*;
use crate::aux::atom::{Atom, ElementMap};
use crate::aux::molecule::Molecule;
use crate::chartab::chartab_group::CharacterProperties;
use crate::chartab::SubspaceDecomposable;
use crate::drivers::representation_analysis::angular_function::AngularFunctionRepAnalysisParams;
use crate::drivers::representation_analysis::slater_determinant::{
    SlaterDeterminantRepAnalysisDriver, SlaterDeterminantRepAnalysisParams,
};
use crate::drivers::symmetry_group_detection::{
    SymmetryGroupDetectionDriver, SymmetryGroupDetectionParams,
};
use crate::drivers::QSym2Driver;
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

#[derive(Clone, Builder)]
struct QChemH5Driver<'a, T>
where
    T: ComplexFloat + Lapack,
    <T as ComplexFloat>::Real: From<f64> + fmt::LowerExp + fmt::Debug,
{
    f: &'a hdf5::File,

    symmetry_group_detection_parameters: &'a SymmetryGroupDetectionParams,

    angular_function_analysis_parameters: &'a AngularFunctionRepAnalysisParams,

    slater_det_rep_analysis_parameters:
        &'a SlaterDeterminantRepAnalysisParams<<T as ComplexFloat>::Real>,

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
    fn builder() -> QChemH5DriverBuilder<'a, T> {
        QChemH5DriverBuilder::default()
    }
}

// Specific for determinant numeric type f64
// '''''''''''''''''''''''''''''''''''''''''

impl<'a> QChemH5Driver<'a, f64> {
    fn analyse(&mut self) -> Result<(), anyhow::Error> {
        let sp_paths = self
            .f
            .group(".counters")?
            .member_names()?
            .iter()
            .filter_map(|path| {
                println!("path: {path}");
                if SP_PATH_RE.is_match(path) {
                    Some(path.replace("\\", "/").replace("/energy_function", ""))
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();

        let pd_params = self.symmetry_group_detection_parameters;
        let afa_params = self.angular_function_analysis_parameters;
        let sda_params = self.slater_det_rep_analysis_parameters;
        let result = sp_paths
            .iter()
            .map(|sp_path| {
                log::info!(target: "qsym2-output", ">>>>> Beginning of analysis for {sp_path}.");
                log::info!(target: "qsym2-output", "");
                let sp = self.f.group(sp_path)?;
                let sp_driver_result = if sda_params.use_magnetic_group {
                    let mut sp_driver =
                        QChemH5SinglePointDriver::<MagneticRepresentedSymmetryGroup, f64>::builder(
                        )
                        .sp_group(&sp)
                        .symmetry_group_detection_parameters(pd_params)
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
                            rep.to_string(),
                        )
                    })
                } else {
                    let mut sp_driver =
                        QChemH5SinglePointDriver::<UnitaryRepresentedSymmetryGroup, f64>::builder()
                            .sp_group(&sp)
                            .symmetry_group_detection_parameters(pd_params)
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
                            rep.to_string(),
                        )
                    })
                };
                log::info!(target: "qsym2-output", "");
                log::info!(target: "qsym2-output", "<<<<< End of analysis for {sp_path}.");
                log::info!(target: "qsym2-output", "");
                sp_driver_result
            })
            .map(|res| {
                res.unwrap_or((
                    "Unidentified symmetry group".to_string(),
                    "Unidentified (co)representation".to_string(),
                ))
            })
            .collect::<Vec<_>>();
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

// ~~~~~~~~~~~~~~~~~
// Struct definition
// ~~~~~~~~~~~~~~~~~

#[derive(Clone, Builder)]
struct QChemH5SinglePointDriver<'a, G, T>
where
    G: SymmetryGroupProperties + Clone,
    G::CharTab: SubspaceDecomposable<T>,
    T: ComplexFloat + Lapack,
    <T as ComplexFloat>::Real: From<f64> + fmt::LowerExp + fmt::Debug,
{
    sp_group: &'a hdf5::Group,

    symmetry_group_detection_parameters: &'a SymmetryGroupDetectionParams,

    angular_function_analysis_parameters: &'a AngularFunctionRepAnalysisParams,

    slater_det_rep_analysis_parameters:
        &'a SlaterDeterminantRepAnalysisParams<<T as ComplexFloat>::Real>,

    #[builder(default = "None")]
    result: Option<(
        Symmetry,
        <G::CharTab as SubspaceDecomposable<T>>::Decomposition,
    )>,
}

// ~~~~~~~~~~~~~~~~~~~~~~
// Struct implementations
// ~~~~~~~~~~~~~~~~~~~~~~

// Generic for all symmetry groups G and determinant numeric type T
// ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

impl<'a, G, T> QChemH5SinglePointDriver<'a, G, T>
where
    G: SymmetryGroupProperties + Clone,
    G::CharTab: SubspaceDecomposable<T>,
    T: ComplexFloat + Lapack + H5Type,
    <T as ComplexFloat>::Real: From<f64> + fmt::LowerExp + fmt::Debug,
{
    fn builder() -> QChemH5SinglePointDriverBuilder<'a, G, T> {
        QChemH5SinglePointDriverBuilder::default()
    }

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

    fn extract_sao(&self) -> Result<Array2<T>, anyhow::Error> {
        self.sp_group
            .dataset("aobasis/overlap_matrix")?
            .read_2d::<T>()
            .map_err(|err| err.into())
    }

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

    fn extract_determinant(
        &self,
        mol: &'a Molecule,
        bao: &'a BasisAngularOrder,
        threshold: <T as ComplexFloat>::Real,
    ) -> Result<SlaterDeterminant<'a, T>, anyhow::Error> {
        let n_spatial = bao.n_funcs();

        let energy = self
            .sp_group
            .dataset("energy_function/1/energy")?
            .read_scalar::<T>()
            .map_err(|err| err.to_string());
        let nspins = self
            .sp_group
            .dataset("energy_function/1/method/scf/molecular_orbitals/nsets")?
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
                    (0..n_spatial)
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
                    (0..n_spatial)
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
                    (0..n_spatial)
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
            .dataset("energy_function/1/method/scf/molecular_orbitals/mo_coefficients")?
            .read::<T, Ix3>()?
            .axis_iter(Axis(0))
            .map(|c| c.to_owned())
            .collect::<Vec<_>>();
        let mo_energies = self
            .sp_group
            .dataset("energy_function/1/method/scf/molecular_orbitals/mo_energies")
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
    fn analyse(&mut self) -> Result<(), anyhow::Error> {
        let mol = self.extract_molecule()?;
        let mut pd_driver = SymmetryGroupDetectionDriver::builder()
            .parameters(self.symmetry_group_detection_parameters)
            .molecule(Some(&mol))
            .build()
            .unwrap();
        pd_driver.run()?;
        let pd_res = pd_driver.result()?;
        let recentred_mol = &pd_res.pre_symmetry.recentred_molecule;

        let sao = self.extract_sao()?;
        let bao = self.extract_bao(recentred_mol)?;
        let det = self.extract_determinant(
            recentred_mol,
            &bao,
            self.slater_det_rep_analysis_parameters
                .linear_independence_threshold,
        )?;

        let mut sda_driver =
            SlaterDeterminantRepAnalysisDriver::<UnitaryRepresentedSymmetryGroup, f64>::builder()
                .parameters(self.slater_det_rep_analysis_parameters)
                .angular_function_parameters(self.angular_function_analysis_parameters)
                .determinant(&det)
                .sao_spatial(&sao)
                .symmetry_group(pd_res)
                .build()
                .unwrap();
        sda_driver.run()?;
        let sda_res = sda_driver.result()?;
        let sym = if self.slater_det_rep_analysis_parameters.use_magnetic_group {
            pd_res.magnetic_symmetry.clone()
        } else {
            Some(pd_res.unitary_symmetry.clone())
        }
        .ok_or(format_err!("Symmetry not found."))?;
        self.result = Some((
            sym,
            sda_res.determinant_symmetry().as_ref().unwrap().clone(),
        ));
        Ok(())
    }
}

// Specific for magnetic-represented symmetry groups and determinant numeric type f64
// ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

impl<'a> QChemH5SinglePointDriver<'a, MagneticRepresentedSymmetryGroup, f64> {
    fn analyse(&mut self) -> Result<(), anyhow::Error> {
        let mol = self.extract_molecule()?;
        let mut pd_driver = SymmetryGroupDetectionDriver::builder()
            .parameters(&self.symmetry_group_detection_parameters)
            .molecule(Some(&mol))
            .build()
            .unwrap();
        pd_driver.run()?;
        let pd_res = pd_driver.result()?;
        let recentred_mol = &pd_res.pre_symmetry.recentred_molecule;

        let sao = self.extract_sao()?;
        let bao = self.extract_bao(recentred_mol)?;
        let det = self.extract_determinant(
            recentred_mol,
            &bao,
            self.slater_det_rep_analysis_parameters
                .linear_independence_threshold,
        )?;

        let mut sda_driver =
            SlaterDeterminantRepAnalysisDriver::<MagneticRepresentedSymmetryGroup, f64>::builder()
                .parameters(self.slater_det_rep_analysis_parameters)
                .angular_function_parameters(self.angular_function_analysis_parameters)
                .determinant(&det)
                .sao_spatial(&sao)
                .symmetry_group(pd_res)
                .build()
                .unwrap();
        sda_driver.run()?;
        let sda_res = sda_driver.result()?;
        let sym = if self.slater_det_rep_analysis_parameters.use_magnetic_group {
            pd_res.magnetic_symmetry.clone()
        } else {
            Some(pd_res.unitary_symmetry.clone())
        }
        .ok_or(format_err!("Symmetry not found."))?;
        self.result = Some((
            sym,
            sda_res.determinant_symmetry().as_ref().unwrap().clone(),
        ));
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
        <<UnitaryRepresentedSymmetryGroup as CharacterProperties>::CharTab as SubspaceDecomposable<f64>>::Decomposition,
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
        <<MagneticRepresentedSymmetryGroup as CharacterProperties>::CharTab as SubspaceDecomposable<f64>>::Decomposition,
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
