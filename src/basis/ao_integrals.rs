use std::collections::HashMap;

use anyhow::{self, format_err};
use derive_builder::Builder;
use factorial::DoubleFactorial;
use nalgebra::{Point3, Vector3};
use num_traits::ToPrimitive;
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

    /// The function starting index of this shell in the basis set.
    pub(crate) start_index: usize,

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


    pub(crate) fn from_bse(
        mol: &Molecule,
        basis_name: &str,
        cart: bool,
        optimised_contraction: bool,
        version: usize,
        mol_bohr: bool,
    ) -> Result<Vec<BasisShellContraction<f64, f64>>, anyhow::Error> {
        let emap = ElementMap::new();
        let mut bscs = mol
            .atoms
            .par_iter()
            .map(|atom| {
                let element = &atom.atomic_symbol;
                println!("=============> Atom: {atom}");
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

                                        BasisShellContraction {
                                            basis_shell,
                                            start_index: 0,
                                            contraction,
                                            cart_origin,
                                            k: None,
                                        }
                                    })
                            })
                            .collect::<Vec<_>>()
                    })
            })
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
        let mut start_index = 0;
        bscs.iter_mut().for_each(|bsc| {
            bsc.start_index = start_index;
            start_index += bsc.basis_shell.n_funcs();
            if optimised_contraction {
                bsc.renormalise();
            }
        });
        Ok(bscs)
    }
}

impl BasisShellContraction<f64, f64> {
    pub(crate) fn renormalise(&mut self) -> &mut Self {
        // let contraction_length = self.contraction.contraction_length();
        // let doufac = if self.basis_shell.l > 0 {
        //     (2 * self.basis_shell.l - 1)
        //         .checked_double_factorial()
        //         .unwrap_or_else(|| panic!("Unable to obtain `{}!!`.", 2 * self.basis_shell.l - 1))
        //         .to_f64()
        //         .unwrap_or_else(|| panic!("Unable to convert `{}!!` to `f64.", 2 * self.basis_shell.l - 1))
        // } else {
        //     1.0
        // };
        // let pi_pow_3_2 = std::f64::consts::PI.powi(3).sqrt();
        // let norm = (0..contraction_length).map(|i| {
        //     let (ei, di) = self.contraction.primitives[i];
        //     (0..=i).map(|j| {
        //         let (ej, dj) = self.contraction.primitives[j];
        //         let gamma = ei + ej;
        //         let fac = if i == j {
        //             1.0f64
        //         } else {
        //             2.0f64
        //         };
        //         let l_i32 = self
        //             .basis_shell
        //             .l
        //             .to_i32()
        //             .unwrap_or_else(|| panic!("Unable to convert `{}` to `i32`.", self.basis_shell.l));
        //         fac * di * dj
        //         / (
        //             2u32
        //                 .pow(self.basis_shell.l)
        //                 .to_f64()
        //                 .unwrap_or_else(|| panic!("Unable to convert `2^{}` to `f64`.", self.basis_shell.l))
        //             * gamma.powi(l_i32 + 1)
        //             * gamma.sqrt()
        //         )
        //     }).sum::<f64>()
        // }).sum::<f64>();
        crate::integrals::shell_tuple::define_shell_tuple![<s1, s2>];
        let c_self = self.clone();
        let st = crate::integrals::shell_tuple::build_shell_tuple![(&c_self, true), (&c_self, false); f64];
        let ovs = st.overlap([0, 0]);
        let norm = ovs[0].iter().next().unwrap();
        let scale = 1.0 / norm.sqrt();
        self.contraction.primitives.iter_mut().for_each(|(_, d)| {
            *d *= scale;
        });
        self
    }
}

