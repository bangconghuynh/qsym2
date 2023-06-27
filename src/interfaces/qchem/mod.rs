use std::collections::{HashMap, HashSet};
use std::fmt;
use std::fs::File;
use std::io::{BufRead, BufReader, Lines};
use std::str::FromStr;

use anyhow::{self, bail, ensure, format_err};
use derive_builder::Builder;
use lazy_static::lazy_static;
use nalgebra::Point3;
use ndarray::Array2;
use periodic_table::periodic_table;
use regex::Regex;

use crate::aux::atom::{Atom, ElementMap};
use crate::aux::molecule::Molecule;
use crate::interfaces::input::ao_basis::*;

lazy_static! {
    static ref NUMBER_RE: Regex =
        Regex::new(r"[-+]?\d+(\.\d+E?[-+]?\d+)?").expect("Regex pattern invalid.");
}

lazy_static! {
    static ref HEADING_RE: Regex =
        Regex::new(r"(?P<title>.+)\b\s+(?P<type>[IR])\s+N=\s+(?P<nele>\d+)")
            .expect("Regex pattern invalid.");
}

#[derive(Builder, Clone)]
struct QChemCheckPointFile {
    /// The geometries parsed from the checkpoint file.
    mols: Vec<Molecule>,

    /// The input basis angular order information parsed from the checkpoint file.
    inp_bao: InputBasisAngularOrder,

    a_cs: Option<Vec<Array2<f64>>>,

    b_cs: Option<Vec<Array2<f64>>>,
}

impl QChemCheckPointFile {
    fn builder() -> QChemCheckPointFileBuilder {
        QChemCheckPointFileBuilder::default()
    }

    /// Parses a Q-Chem FCHK file.
    fn parse(name: &str) -> Result<Self, anyhow::Error> {
        let reader = BufReader::new(File::open(name).map_err(|err| format_err!(err))?);
        let mut lines = reader.lines();
        let sections = HashSet::from([
            "Atomic numbers",
            "Current cartesian coordinates",
            "Shell types",
            "Shell to atom map",
            "Alpha MO coefficients",
            "Beta MO coefficients",
        ]);
        let mut parsed_i32_sections = HashMap::<String, Vec<Vec<i32>>>::new();
        let mut parsed_f64_sections = HashMap::<String, Vec<Vec<f64>>>::new();
        while let Some(line_res) = lines.next() {
            if let Ok(line) = line_res.as_ref() {
                if let Some(caps) = HEADING_RE.captures(line) {
                    let (title, data_type, nele) = (
                        caps.name("title").expect("No 'title' found.").as_str(),
                        caps.name("type").expect("No 'type' found.").as_str(),
                        str::parse::<usize>(caps.name("nele").expect("No 'nele' found.").as_str())
                            .unwrap(),
                    );
                    if sections.contains(title) {
                        match data_type {
                            "I" => {
                                parsed_i32_sections
                                    .entry(title.to_string())
                                    .and_modify(|section| {
                                        section.push(
                                            parse_qchem_fchk_section::<_, i32>(&mut lines, nele, 6)
                                                .unwrap_or_else(|err| panic!("Parsing of a section with title {title} has failed with error: {err}"))
                                        )
                                    })
                                    .or_insert(vec![parse_qchem_fchk_section::<_, i32>(
                                        &mut lines, nele, 6
                                    )?]);
                            }
                            "R" => {
                                parsed_f64_sections
                                    .entry(title.to_string())
                                    .and_modify(|section| {
                                        section.push(
                                            parse_qchem_fchk_section::<_, f64>(&mut lines, nele, 5)
                                                .unwrap_or_else(|err| panic!("Parsing of a section with title {title} has failed with error: {err}"))
                                        )
                                    })
                                    .or_insert(vec![parse_qchem_fchk_section::<_, f64>(
                                        &mut lines, nele, 5
                                    )?]);
                            }
                            &_ => panic!("Unexpected data type in FCHK file."),
                        }
                    } else {
                        continue;
                    }
                }
            }
        }

        let emap = ElementMap::new();
        let coordss = parsed_f64_sections
            .get("Current cartesian coordinates")
            .ok_or(format_err!("Current cartesian coordinates not found."))?;
        let atomic_numbers = parsed_i32_sections
            .get("Atomic numbers")
            .ok_or(format_err!("Atomic numbers not found."))?
            .get(0)
            .ok_or(format_err!("Atomic numbers not found."))?
            .iter()
            .map(|atomic_number_i32| {
                usize::try_from(atomic_number_i32.unsigned_abs())
                    .expect("Unable to convert an atomic number to `usize`.")
            })
            .collect::<Vec<_>>();
        let mut mols = coordss
            .iter()
            .map(|coords| {
                let atoms = atomic_numbers
                    .iter()
                    .enumerate()
                    .map(|(i, atomic_number)| {
                        let element = periodic_table()
                            .get(*atomic_number - 1)
                            .ok_or(format_err!("Element {atomic_number} not found."))?
                            .symbol;
                        let coordinates =
                            Point3::new(coords[3 * i], coords[3 * i + 1], coords[3 * i + 2]);
                        Ok(Atom::new_ordinary(element, coordinates, &emap, 1e-8))
                    })
                    .collect::<Result<Vec<Atom>, anyhow::Error>>()?;
                Ok::<Molecule, anyhow::Error>(Molecule::from_atoms(&atoms, 1e-8))
            })
            .collect::<Result<Vec<Molecule>, anyhow::Error>>()?;

        let shell_types = parsed_i32_sections
            .get("Shell types")
            .ok_or(format_err!("Shell types not found."))?
            .get(0)
            .ok_or(format_err!("Shell types not found."))?;
        let shell_to_atom_map = parsed_i32_sections
            .get("Shell to atom map")
            .ok_or(format_err!("Shell to atom map not found."))?
            .get(0)
            .ok_or(format_err!("Shell to atom map not found."))?
            .iter()
            .zip(shell_types.iter())
            .flat_map(|(idx, shell_type)| {
                let idx_usize = usize::try_from(idx.unsigned_abs())
                    .expect("Unable to convert an atom index to `usize`.");
                if *shell_type == -1 {
                    vec![idx_usize, idx_usize]
                } else {
                    vec![idx_usize]
                }
            })
            .collect::<Vec<_>>();

        let bss: Vec<InputBasisShell> = shell_types
            .iter()
            .flat_map(|shell_type| {
                if *shell_type == 0 {
                    // S shell
                    vec![InputBasisShell::builder()
                        .l(0)
                        .shell_order(InputShellOrder::CartQChem)
                        .build()
                        .map_err(|err| format_err!(err))]
                } else if *shell_type == 1 {
                    // P shell
                    vec![InputBasisShell::builder()
                        .l(1)
                        .shell_order(InputShellOrder::CartQChem)
                        .build()
                        .map_err(|err| format_err!(err))]
                } else if *shell_type == -1 {
                    // SP shell
                    vec![
                        InputBasisShell::builder()
                            .l(0)
                            .shell_order(InputShellOrder::CartQChem)
                            .build()
                            .map_err(|err| format_err!(err)),
                        InputBasisShell::builder()
                            .l(1)
                            .shell_order(InputShellOrder::CartQChem)
                            .build()
                            .map_err(|err| format_err!(err)),
                    ]
                } else if *shell_type > 0 {
                    vec![InputBasisShell::builder()
                        .l(shell_type.unsigned_abs())
                        .shell_order(InputShellOrder::CartQChem)
                        .build()
                        .map_err(|err| format_err!(err))]
                } else {
                    vec![InputBasisShell::builder()
                        .l(shell_type.unsigned_abs())
                        .shell_order(InputShellOrder::Pure(true))
                        .build()
                        .map_err(|err| format_err!(err))]
                }
            })
            .collect::<Result<Vec<InputBasisShell>, anyhow::Error>>()?;
        ensure!(
            bss.len() == shell_to_atom_map.len(),
            "Unequal lengths between `bss` and `shell_to_atom_map`."
        );

        let batms: Vec<InputBasisAtom> = atomic_numbers
            .iter()
            .enumerate()
            .map(|(atom_i, atomic_number)| {
                let shells = bss
                    .iter()
                    .zip(shell_to_atom_map.iter())
                    .filter_map(|(bs, atom_index)| {
                        // atom_index is 1-based.
                        if *atom_index == atom_i + 1 {
                            Some(bs.clone())
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>();
                let element = periodic_table()
                    .get(*atomic_number - 1)
                    .ok_or(format_err!("Element {atomic_number} not found."))?
                    .symbol;
                InputBasisAtom::builder()
                    .atom((atom_i, element.to_string()))
                    .basis_shells(shells)
                    .build()
                    .map_err(|err| format_err!(err))
            })
            .collect::<Result<Vec<InputBasisAtom>, anyhow::Error>>()?;

        let inp_bao = InputBasisAngularOrder(batms);
        let bao = inp_bao.to_basis_angular_order(&mols[0])?;
        let n_spatial = bao.n_funcs();

        let a_cs_opt = parsed_f64_sections
            .get("Alpha MO coefficients")
            .map(|a_cs| {
                a_cs.iter()
                    .map(|a_c| {
                        Array2::from_shape_vec(
                            (n_spatial, a_c.len().div_euclid(n_spatial)),
                            a_c.clone(),
                        )
                        .map_err(|err| format_err!(err))
                    })
                    .collect::<Result<Vec<_>, _>>()
            })
            .transpose()?;

        let b_cs_opt = parsed_f64_sections
            .get("Beta MO coefficients")
            .map(|b_cs| {
                b_cs.iter()
                    .map(|b_c| {
                        Array2::from_shape_vec(
                            (n_spatial, b_c.len().div_euclid(n_spatial)),
                            b_c.clone(),
                        )
                        .map_err(|err| format_err!(err))
                    })
                    .collect::<Result<Vec<_>, _>>()
            })
            .transpose()?;

        if let Some(a_cs) = a_cs_opt.as_ref() {
            if a_cs.len() == mols.len() + 1 {
                // Geometry optimisation
                mols.remove(0);
            }
        }

        QChemCheckPointFile::builder()
            .mols(mols)
            .inp_bao(inp_bao)
            .a_cs(a_cs_opt)
            .b_cs(b_cs_opt)
            .build()
            .map_err(|err| format_err!(err))
    }
}

fn parse_qchem_fchk_section<B: BufRead, T: FromStr + fmt::Debug>(
    lines: &mut Lines<B>,
    n: usize,
    c: usize,
) -> Result<Vec<T>, anyhow::Error> {
    let n_lines = n.div_euclid(c) + usize::from(n.rem_euclid(c) != 0);
    let numbers = lines
        .take(n_lines)
        .filter_map(|line_res| {
            line_res.ok().map(|line| {
                NUMBER_RE
                    .find_iter(&line)
                    .filter_map(|digits| digits.as_str().parse::<T>().ok())
                    .collect::<Vec<_>>()
            })
        })
        .flatten()
        .collect::<Vec<_>>();
    ensure!(
        numbers.len() == n,
        "Expected {n} elements, got {}.",
        numbers.len()
    );
    Ok(numbers)
}

#[cfg(test)]
mod qchem_test {
    use super::QChemCheckPointFile;

    #[test]
    fn test_parse_qchem_fchk() {
        let qchem_fchk = QChemCheckPointFile::parse("HF.inp.fchk").unwrap();
        for mol in qchem_fchk.mols {
            println!("{mol}");
            println!(
                "{}",
                qchem_fchk.inp_bao.to_basis_angular_order(&mol).unwrap()
            );
        }
    }
}
