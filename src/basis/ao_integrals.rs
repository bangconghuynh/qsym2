use std::collections::HashMap;
use std::ops::Index;

use anyhow::{self, format_err};
use derive_builder::Builder;
use nalgebra::{Point3, Vector3};
use rayon::prelude::*;
use reqwest;
use serde::{Deserialize, Serialize};

use crate::auxiliary::atom::{ElementMap, ANGSTROM_TO_BOHR};
use crate::auxiliary::molecule::Molecule;
use crate::basis::ao::{BasisShell, CartOrder, PureOrder, ShellOrder};

#[cfg(test)]
#[path = "ao_integrals_tests.rs"]
mod ao_integrals_tests;

// -------------------
// GaussianContraction
// -------------------

/// A structure to handle primitives in a Gaussian contraction.
#[derive(Clone, Builder, Debug)]
pub(crate) struct GaussianContraction<E, C> {
    /// Constituent primitives in the contraction. Each primitive has the form
    /// $`c\exp\left[-\alpha\lvert \mathbf{r} - \mathbf{R} \rvert^2\right]`$ is characterised by a
    /// tuple of its exponent $`\alpha`$ and coefficient $`c`$, respectively.
    pub(crate) primitives: Vec<(E, C)>,
}

impl<E, C> GaussianContraction<E, C> {
    pub(crate) fn contraction_length(&self) -> usize {
        self.primitives.len()
    }
}

// ---------------------
// BasisShellContraction
// ---------------------

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Deserialisable structs for BSE data retrieval
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

const BSE_BASE_API: &str = "https://www.basissetexchange.org/api";

#[derive(Serialize, Deserialize, Debug)]
struct BSEResponse {
    name: String,
    version: String,
    elements: HashMap<u32, BSEElement>,
}

#[derive(Serialize, Deserialize, Debug)]
struct BSEElement {
    electron_shells: Vec<BSEElectronShell>,
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(try_from = "BSEElectronShellRaw")]
struct BSEElectronShell {
    function_type: String,
    region: String,
    angular_momentum: Vec<u32>,
    exponents: Vec<f64>,
    coefficients: Vec<Vec<f64>>,
}

#[derive(Deserialize)]
struct BSEElectronShellRaw {
    function_type: String,
    region: String,
    angular_momentum: Vec<u32>,
    exponents: Vec<String>,
    coefficients: Vec<Vec<String>>,
}

impl TryFrom<BSEElectronShellRaw> for BSEElectronShell {
    type Error = std::num::ParseFloatError;

    fn try_from(other: BSEElectronShellRaw) -> Result<Self, Self::Error> {
        let converted = Self {
            function_type: other.function_type,
            region: other.region,
            angular_momentum: other.angular_momentum,
            exponents: other
                .exponents
                .iter()
                .map(|s| s.parse::<f64>())
                .collect::<Result<Vec<_>, _>>()?,
            coefficients: other
                .coefficients
                .iter()
                .map(|d| {
                    d.iter()
                        .map(|s| s.parse::<f64>())
                        .collect::<Result<Vec<_>, _>>()
                })
                .collect::<Result<Vec<_>, _>>()?,
        };
        Ok(converted)
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// BasisShellContraction definition
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// A structure to handle all shell information for integrals.
#[derive(Clone, Builder, Debug)]
pub(crate) struct BasisShellContraction<E, C> {
    /// Basis function ordering information.
    pub(crate) basis_shell: BasisShell,

    /// The Gaussian primitives in the contraction of this shell.
    pub(crate) contraction: GaussianContraction<E, C>,

    /// The Cartesian origin $`\mathbf{R}`$ of this shell.
    pub(crate) cart_origin: Point3<f64>,

    /// The optional plane-wave $`\mathbf{k}`$ vector in the exponent
    /// $`\exp\left[i\mathbf{k}\cdot(\mathbf{r} - \mathbf{R})\right]`$ associated with this shell.
    /// If this is `None`, then this exponent is set to unity.
    pub(crate) k: Option<Vector3<f64>>,
}

impl<E, C> BasisShellContraction<E, C> {
    pub(crate) fn basis_shell(&self) -> &BasisShell {
        &self.basis_shell
    }

    pub(crate) fn k(&self) -> Option<&Vector3<f64>> {
        self.k.as_ref()
    }

    pub(crate) fn cart_origin(&self) -> &Point3<f64> {
        &self.cart_origin
    }

    pub(crate) fn contraction_length(&self) -> usize {
        self.contraction.contraction_length()
    }
}

impl BasisShellContraction<f64, f64> {
    pub(crate) fn renormalise(&mut self) -> &mut Self {
        let c_self = self.clone();
        let st = crate::integrals::shell_tuple::build_shell_tuple![
            (&c_self, true), (&c_self, false); f64
        ];
        let ovs = st.overlap([0, 0]);
        let norm = ovs[0].iter().next().unwrap();
        let scale = 1.0 / norm.sqrt();
        self.contraction.primitives.iter_mut().for_each(|(_, d)| {
            *d *= scale;
        });
        self
    }

    pub(crate) fn from_bse(
        mol: &Molecule,
        basis_name: &str,
        cart: bool,
        optimised_contraction: bool,
        version: usize,
        mol_bohr: bool,
        force_renormalisation: bool,
    ) -> Result<Vec<Vec<Self>>, anyhow::Error> {
        let emap = ElementMap::new();
        let mut bscs = mol
            .atoms
            .par_iter()
            .map(|atom| {
                let element = &atom.atomic_symbol;
                let api_url = format!(
                    "{BSE_BASE_API}/basis/\
                {basis_name}/format/json/\
                ?elements={element}\
                &optimize_general={optimised_contraction}\
                &version={version}"
                );
                let rjson: BSEResponse = reqwest::blocking::get(&api_url)?.json()?;
                let atomic_number = emap
                    .get(element)
                    .ok_or(format_err!("Element {element} not found."))?
                    .0;
                rjson
                    .elements
                    .get(&atomic_number)
                    .ok_or(format_err!(
                        "Basis information for element {element} not found."
                    ))
                    .map(|element| {
                        element
                            .electron_shells
                            .iter()
                            .flat_map(|shell| {
                                shell
                                    .angular_momentum
                                    .iter()
                                    .cycle()
                                    .zip(shell.coefficients.iter())
                                    .map(|(&l, d)| {
                                        let shell_order = if cart {
                                            ShellOrder::Cart(CartOrder::lex(l))
                                        } else {
                                            ShellOrder::Pure(PureOrder::increasingm(l))
                                        };
                                        let basis_shell = BasisShell::new(l, shell_order);

                                        let contraction = GaussianContraction::<f64, f64> {
                                            primitives: shell
                                                .exponents
                                                .iter()
                                                .copied()
                                                .zip(d.iter().copied())
                                                .filter(|(_, d)| d.abs() > 1e-13)
                                                .collect::<Vec<(f64, f64)>>(),
                                        };

                                        let cart_origin = if mol_bohr {
                                            atom.coordinates.clone()
                                        } else {
                                            atom.coordinates.clone() * ANGSTROM_TO_BOHR
                                        };

                                        if force_renormalisation {
                                            let mut bsc = BasisShellContraction {
                                                basis_shell,
                                                contraction,
                                                cart_origin,
                                                k: None,
                                            };
                                            bsc.renormalise();
                                            bsc
                                        } else {
                                            BasisShellContraction {
                                                basis_shell,
                                                contraction,
                                                cart_origin,
                                                k: None,
                                            }
                                        }
                                    })
                            })
                            .collect::<Vec<_>>()
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(bscs)
    }
}

// --------
// BasisSet
// --------
pub(crate) struct BasisSet<E, C> {
    basis_atoms: Vec<Vec<BasisShellContraction<E, C>>>,
    atom_boundaries: Vec<(usize, usize)>,
    shell_boundaries: Vec<(usize, usize)>,
}

impl<E, C> BasisSet<E, C> {
    pub(crate) fn new(batms: Vec<Vec<BasisShellContraction<E, C>>>) -> Self {
        let atom_boundaries = batms
            .iter()
            .scan(0, |acc, batm| {
                let atom_length = batm
                    .iter()
                    .map(|bs| bs.basis_shell.n_funcs())
                    .sum::<usize>();
                let boundary = (*acc, *acc + atom_length);
                *acc += atom_length;
                Some(boundary)
            })
            .collect::<Vec<_>>();
        let shell_boundaries = batms
            .iter()
            .flatten()
            .scan(0, |acc, bsc| {
                let shell_length = bsc.basis_shell.n_funcs();
                let boundary = (*acc, *acc + shell_length);
                *acc += shell_length;
                Some(boundary)
            })
            .collect::<Vec<_>>();
        Self {
            basis_atoms: batms,
            atom_boundaries,
            shell_boundaries,
        }
    }

    fn update_shell_boundaries(&mut self) -> &mut Self {
        self.shell_boundaries = self
            .basis_atoms
            .iter()
            .flatten()
            .scan(0, |acc, bsc| {
                let shell_length = bsc.basis_shell.n_funcs();
                let boundary = (*acc, *acc + shell_length);
                *acc += shell_length;
                Some(boundary)
            })
            .collect::<Vec<_>>();
        self
    }

    pub(crate) fn n_shells(&self) -> usize {
        self.basis_atoms.iter().map(|batm| batm.len()).sum::<usize>()
    }

    pub(crate) fn sort_by_angular_momentum(&mut self) -> &mut Self {
        self.basis_atoms
            .iter_mut()
            .for_each(|batm| batm.sort_by_key(|bsc| bsc.basis_shell.l));
        self.update_shell_boundaries()
    }

    pub(crate) fn shell_boundaries(&self) -> &Vec<(usize, usize)> {
        &self.shell_boundaries
    }

    pub(crate) fn all_shells(&self) -> impl Iterator<Item = &BasisShellContraction<E, C>> {
        self.basis_atoms.iter().flatten()
    }

    pub(crate) fn all_shells_mut(
        &mut self,
    ) -> impl Iterator<Item = &mut BasisShellContraction<E, C>> {
        self.basis_atoms.iter_mut().flatten()
    }
}

impl BasisSet<f64, f64> {
    pub(crate) fn from_bse(
        mol: &Molecule,
        basis_name: &str,
        cart: bool,
        optimised_contraction: bool,
        version: usize,
        mol_bohr: bool,
        force_renormalisation: bool,
    ) -> Result<Self, anyhow::Error> {
        Ok(Self::new(BasisShellContraction::<f64, f64>::from_bse(
            mol,
            basis_name,
            cart,
            optimised_contraction,
            version,
            mol_bohr,
            force_renormalisation,
        )?))
    }
}

impl<E, C> Index<usize> for BasisSet<E, C> {
    type Output = BasisShellContraction<E, C>;

    fn index(&self, i: usize) -> &Self::Output {
        self.basis_atoms
            .iter()
            .flatten()
            .nth(i)
            .unwrap_or_else(|| panic!("Unable to obtain the basis shell with index {i}."))
    }
}
