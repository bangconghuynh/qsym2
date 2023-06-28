use std::collections::{HashMap, HashSet};
use std::fmt;
use std::fs::File;
use std::io::{BufRead, BufReader, Lines};
use std::str::FromStr;

use anyhow::{self, ensure, format_err};
use derive_builder::{Builder, UninitializedFieldError};
use lazy_static::lazy_static;
use nalgebra::Point3;
use ndarray::{Array2, Axis, ShapeBuilder};
use num_traits::Zero;
use periodic_table::periodic_table;
use regex::Regex;

use crate::aux::ao_basis::{
    BasisAngularOrder, BasisAtom, BasisShell, CartOrder, PureOrder, ShellOrder,
};
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
struct QChemCheckPointParser {
    i32_sections: HashMap<String, Vec<Vec<i32>>>,

    f64_sections: HashMap<String, Vec<Vec<f64>>>,
}

impl QChemCheckPointParser {
    fn builder() -> QChemCheckPointParserBuilder {
        QChemCheckPointParserBuilder::default()
    }

    /// Parses the required sections of a Q-Chem FCHK file.
    fn parse(name: &str) -> Result<Self, anyhow::Error> {
        let reader = BufReader::new(File::open(name).map_err(|err| format_err!(err))?);
        let mut lines = reader.lines();
        let mut parsed_i32_sections = HashMap::<String, Vec<Vec<i32>>>::new();
        let mut parsed_f64_sections = HashMap::<String, Vec<Vec<f64>>>::new();

        let sections = HashSet::from([
            "Atomic numbers",
            "Current cartesian coordinates",
            "Shell types",
            "Shell to atom map",
            "Overlap Matrix",
            "Alpha MO coefficients",
            "Beta MO coefficients",
        ]);
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

        QChemCheckPointParser::builder()
            .i32_sections(parsed_i32_sections)
            .f64_sections(parsed_f64_sections)
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

#[derive(Builder, Clone)]
#[builder(build_fn(skip))]
struct QChemCheckPoint {
    parser: QChemCheckPointParser,

    /// The geometries parsed from the checkpoint file.
    #[builder(setter(skip))]
    molecules: Option<Vec<Molecule>>,

    /// The input basis angular order information parsed from the checkpoint file.
    #[builder(setter(skip))]
    inp_bao: Option<InputBasisAngularOrder>,

    #[builder(setter(skip))]
    saos: Option<Vec<Array2<f64>>>,

    #[builder(setter(skip))]
    cs: Option<Vec<Vec<Array2<f64>>>>,
}

impl QChemCheckPoint {
    fn builder() -> QChemCheckPointBuilder {
        QChemCheckPointBuilder::default()
    }

    fn from_file(name: &str) -> Result<Self, anyhow::Error> {
        let parser = QChemCheckPointParser::parse(name)?;
        Self::builder()
            .parser(parser)
            .build()
            .map_err(|err| format_err!(err))
    }
}

impl QChemCheckPointBuilder {
    fn extract_molecules(&self) -> Option<Vec<Molecule>> {
        let emap = ElementMap::new();
        let coordss = self
            .parser
            .as_ref()?
            .f64_sections
            .get("Current cartesian coordinates")?;
        let atomic_numbers = self
            .parser
            .as_ref()?
            .i32_sections
            .get("Atomic numbers")?
            .get(0)?
            .iter()
            .map(|atomic_number_i32| {
                usize::try_from(atomic_number_i32.unsigned_abs())
                    .expect("Unable to convert an atomic number to `usize`.")
            })
            .collect::<Vec<_>>();
        coordss
            .iter()
            .map(|coords| {
                let atoms = atomic_numbers
                    .iter()
                    .enumerate()
                    .map(|(i, atomic_number)| {
                        let element = periodic_table().get(*atomic_number - 1)?.symbol;
                        let coordinates =
                            Point3::new(coords[3 * i], coords[3 * i + 1], coords[3 * i + 2]);
                        Some(Atom::new_ordinary(element, coordinates, &emap, 1e-8))
                    })
                    .collect::<Option<Vec<Atom>>>()?;
                Some(Molecule::from_atoms(&atoms, 1e-8))
            })
            .collect::<Option<Vec<Molecule>>>()
    }

    fn extract_inp_bao(&self) -> Option<InputBasisAngularOrder> {
        let atomic_numbers = self
            .parser
            .as_ref()?
            .i32_sections
            .get("Atomic numbers")?
            .get(0)?
            .iter()
            .map(|atomic_number_i32| {
                usize::try_from(atomic_number_i32.unsigned_abs())
                    .expect("Unable to convert an atomic number to `usize`.")
            })
            .collect::<Vec<_>>();
        let shell_types = self
            .parser
            .as_ref()?
            .i32_sections
            .get("Shell types")?
            .get(0)?;
        let shell_to_atom_map = self
            .parser
            .as_ref()?
            .i32_sections
            .get("Shell to atom map")?
            .get(0)?
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
                        .ok()]
                } else if *shell_type == 1 {
                    // P shell
                    vec![InputBasisShell::builder()
                        .l(1)
                        .shell_order(InputShellOrder::CartQChem)
                        .build()
                        .ok()]
                } else if *shell_type == -1 {
                    // SP shell
                    vec![
                        InputBasisShell::builder()
                            .l(0)
                            .shell_order(InputShellOrder::CartQChem)
                            .build()
                            .ok(),
                        InputBasisShell::builder()
                            .l(1)
                            .shell_order(InputShellOrder::CartQChem)
                            .build()
                            .ok(),
                    ]
                } else if *shell_type > 0 {
                    // Cartesian D shell or higher
                    vec![InputBasisShell::builder()
                        .l(shell_type.unsigned_abs())
                        .shell_order(InputShellOrder::CartQChem)
                        .build()
                        .ok()]
                } else {
                    // Pure D shell or higher
                    vec![InputBasisShell::builder()
                        .l(shell_type.unsigned_abs())
                        .shell_order(InputShellOrder::Pure(true))
                        .build()
                        .ok()]
                }
            })
            .collect::<Option<Vec<InputBasisShell>>>()?;

        atomic_numbers
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
                let element = periodic_table().get(*atomic_number - 1)?.symbol;
                InputBasisAtom::builder()
                    .atom((atom_i, element.to_string()))
                    .basis_shells(shells)
                    .build()
                    .ok()
            })
            .collect::<Option<Vec<InputBasisAtom>>>()
            .map(|batms| InputBasisAngularOrder(batms))
    }

    fn extract_sao(&self, bao: Option<&BasisAngularOrder>) -> Option<Vec<Array2<f64>>> {
        let qchem_bao = bao?;
        let molden_bao = construct_molden_bao(qchem_bao);
        let perm = qchem_bao
            .get_perm_of_functions_fixed_shells(&molden_bao)
            .ok()?;
        let n_spatial = qchem_bao.n_funcs();
        self.parser
            .as_ref()?
            .f64_sections
            .get("Overlap Matrix")
            .map(|saos| {
                saos.iter()
                    .map(|sao_v| {
                        vector_to_symmetric_matrix(sao_v, n_spatial)
                            .map_err(|err| format_err!(err))
                            .map(|sao| {
                                sao.select(Axis(0), perm.image())
                                    .select(Axis(1), perm.image())
                            })
                    })
                    .collect::<Result<Vec<_>, _>>()
            })?
            .ok()
    }

    fn extract_mo_coefficients(
        &self,
        bao: Option<&BasisAngularOrder>,
    ) -> Option<Vec<Vec<Array2<f64>>>> {
        let qchem_bao = bao?;
        let molden_bao = construct_molden_bao(qchem_bao);
        let perm = qchem_bao
            .get_perm_of_functions_fixed_shells(&molden_bao)
            .ok()?;
        let n_spatial = qchem_bao.n_funcs();

        let a_cs_opt = self
            .parser
            .as_ref()?
            .f64_sections
            .get("Alpha MO coefficients")
            .map(|a_cs| {
                a_cs.iter()
                    .map(|a_c| {
                        Array2::from_shape_vec(
                            (n_spatial, a_c.len().div_euclid(n_spatial)).f(),
                            a_c.clone(),
                        )
                        .map_err(|err| format_err!(err))
                        .map(|a_c| a_c.select(Axis(0), perm.image()))
                    })
                    .collect::<Result<Vec<_>, _>>()
            })?
            .ok();

        let b_cs_opt = self
            .parser
            .as_ref()?
            .f64_sections
            .get("Beta MO coefficients")
            .map(|b_cs| {
                b_cs.iter()
                    .map(|b_c| {
                        Array2::from_shape_vec(
                            (n_spatial, b_c.len().div_euclid(n_spatial)).f(),
                            b_c.clone(),
                        )
                        .map_err(|err| format_err!(err))
                        .map(|b_c| b_c.select(Axis(0), &perm.image()))
                    })
                    .collect::<Result<Vec<_>, _>>()
            })?
            .ok();

        match (a_cs_opt.as_ref(), b_cs_opt.as_ref()) {
            (Some(a_cs), Some(b_cs)) => {
                if a_cs.len() == b_cs.len() {
                    Some(
                        a_cs.into_iter()
                            .zip(b_cs.into_iter())
                            .map(|(a_c, b_c)| vec![a_c.clone(), b_c.clone()])
                            .collect::<Vec<_>>(),
                    )
                } else {
                    None
                }
            }
            (Some(cs), None) | (None, Some(cs)) => {
                Some(cs.into_iter().map(|c| vec![c.clone()]).collect::<Vec<_>>())
            }
            (None, None) => None,
        }
    }

    fn build(&self) -> Result<QChemCheckPoint, QChemCheckPointBuilderError> {
        let inp_bao = self.extract_inp_bao();
        let mols = self.extract_molecules();
        let no_mol_err =
            QChemCheckPointBuilderError::from(UninitializedFieldError::new("molecules"));
        let no_init_mol_err =
            QChemCheckPointBuilderError::from(UninitializedFieldError::new("molecules[0]"));
        let mol0 = mols
            .as_ref()
            .ok_or(no_mol_err)?
            .get(0)
            .ok_or(no_init_mol_err)?;
        let bao = inp_bao
            .as_ref()
            .and_then(|inp_bao| inp_bao.to_basis_angular_order(&mol0).ok());
        let cs = self.extract_mo_coefficients(bao.as_ref());
        let saos = self.extract_sao(bao.as_ref());
        Ok(QChemCheckPoint {
            parser: self
                .parser
                .as_ref()
                .ok_or(QChemCheckPointBuilderError::from(
                    UninitializedFieldError::new("parser"),
                ))?
                .clone(),
            molecules: mols,
            saos,
            inp_bao,
            cs,
        })
    }
}

fn vector_to_symmetric_matrix<T>(vec: &Vec<T>, n: usize) -> Result<Array2<T>, anyhow::Error>
where
    T: Clone + Copy + Zero + fmt::Display,
{
    let expected_length = (n * (n + 1)).div_euclid(2);
    for val in vec {
        println!("VAL: {val}")
    }
    ensure!(
        vec.len() == expected_length,
        "For a square matrix of dimension {n}, a vector of length {expected_length} is required. \
        However, a vector of length {} is supplied.",
        vec.len()
    );
    let mut mat: Array2<T> = Array2::zeros((n, n));
    let mut vec_idx = 0;
    for j in 0..n {
        for i in 0..=j {
            println!("Setting ({i}, {j}) = {}", vec[vec_idx]);
            mat[(i, j)] = *vec.get(vec_idx).ok_or(format_err!(
                "Unable to retrieve element {vec_idx} from the vector."
            ))?;
            if i != j {
                mat[(j, i)] = mat[(i, j)];
            }
            vec_idx += 1;
        }
    }
    Ok(mat)
}

fn construct_molden_bao<'a>(bao: &BasisAngularOrder<'a>) -> BasisAngularOrder<'a> {
    BasisAngularOrder::new(
        &bao.basis_atoms
            .iter()
            .map(|batm| {
                BasisAtom::new(
                    batm.atom,
                    &batm
                        .basis_shells
                        .iter()
                        .map(|bs| {
                            if bs.l <= 1 {
                                bs.clone()
                            } else {
                                let shl_ord = match bs.shell_order {
                                    ShellOrder::Pure(_) => {
                                        ShellOrder::Pure(PureOrder::molden(bs.l))
                                    }
                                    ShellOrder::Cart(_) => {
                                        ShellOrder::Cart(CartOrder::molden(bs.l))
                                    }
                                };
                                BasisShell::new(bs.l, shl_ord)
                            }
                        })
                        .collect::<Vec<_>>(),
                )
            })
            .collect::<Vec<_>>(),
    )
}

#[cfg(test)]
mod qchem_test {
    use super::QChemCheckPoint;

    #[test]
    fn test_parse_qchem_fchk() {
        let qchem_fchk = QChemCheckPoint::from_file("HF.inp.fchk").unwrap();
        for (i, mol) in qchem_fchk.molecules.as_ref().unwrap().iter().enumerate() {
            println!("{mol}");
            println!(
                "{}",
                qchem_fchk
                    .inp_bao
                    .as_ref()
                    .unwrap()
                    .to_basis_angular_order(&mol)
                    .unwrap()
            );
            println!("{:#?}", qchem_fchk.saos.as_ref().unwrap()[i]);
            println!("{:#?}", qchem_fchk.cs.as_ref().unwrap()[i]);
        }
    }
}
