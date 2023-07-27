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
    /// The number of primitive Gaussians in this contraction.
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
const CONTRACTION_COEFF_THRESH: f64 = 1e-16;

/// A structure to represent the REST API result fro, BasisSetExchange.
#[derive(Serialize, Deserialize, Debug)]
struct BSEResponse {
    /// Name of the basis set.
    name: String,

    /// Version of the basis set.
    version: String,

    /// A hashmap between atomic numbers and element basis information.
    elements: HashMap<u32, BSEElement>,
}

/// A structure to handle basis set information for an element.
#[derive(Serialize, Deserialize, Debug)]
struct BSEElement {
    /// A vector of basis set information for the shells in this element.
    electron_shells: Vec<BSEElectronShell>,
}

/// A structure to handle basis set information for a shell.
#[derive(Serialize, Deserialize, Debug)]
#[serde(try_from = "BSEElectronShellRaw")]
struct BSEElectronShell {
    /// The type of basis functions in this shell.
    function_type: String,

    /// The chemical region described by this shell.
    region: String,

    /// the angular momentum of this shell.
    angular_momentum: Vec<u32>,

    /// A vector of primitive exponents.
    exponents: Vec<f64>,

    /// A vector of vectors of primitive coefficients. Each inner vector is to be interpreted as a
    /// separate shell with the same primitive exponents and angular momentum, but different
    /// contraction coefficients.
    coefficients: Vec<Vec<f64>>,
}

/// A structure to handle basis set information for a shell, as obtained raw from BasisSetExchange.
#[derive(Deserialize)]
struct BSEElectronShellRaw {
    /// The type of basis functions in this shell.
    function_type: String,

    /// The chemical region described by this shell.
    region: String,

    /// the angular momentum of this shell.
    angular_momentum: Vec<u32>,

    /// A vector of primitive exponents.
    exponents: Vec<String>,

    /// A vector of vectors of primitive coefficients. Each inner vector is to be interpreted as a
    /// separate shell with the same primitive exponents and angular momentum, but different
    /// contraction coefficients.
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
    #[builder(default = "None")]
    pub(crate) k: Option<Vector3<f64>>,
}

impl<E, C> BasisShellContraction<E, C> {
    /// The basis function ordering information of this shell.
    pub(crate) fn basis_shell(&self) -> &BasisShell {
        &self.basis_shell
    }

    /// The plane-wave $`\mathbf{k}`$ vector in the exponent.
    pub(crate) fn k(&self) -> Option<&Vector3<f64>> {
        self.k.as_ref()
    }

    /// The Cartesian origin $`\mathbf{R}`$ of this shell.
    pub(crate) fn cart_origin(&self) -> &Point3<f64> {
        &self.cart_origin
    }

    /// The number of primitive Gaussians in this shell.
    pub(crate) fn contraction_length(&self) -> usize {
        self.contraction.contraction_length()
    }

    /// Applies a uniform magnetic field to the shell and sets its plane-wave $`k`$ vector
    /// according to
    ///
    /// ```math
    ///     \mathbf{k} = \frac{1}{2} \mathbf{B} \times (\mathbf{R} - \mathbf{G}),
    /// ```
    ///
    /// where $`\mathbf{B}`$ is the uniform magnetic field vector, $`\mathbf{R}`$ is the Cartesian
    /// origin of this shell, and $`\mathbf{G}`$ the gauge origin with respect to which the
    /// magnetic field is defined. Both $`\mathbf{R}`$ and $`\mathbf{G}`$ are points in a
    /// space-fixed coordinate system.
    ///
    /// # Arguments
    ///
    /// * `b` - The magnetic field vector $`\mathbf{B}`$.
    /// * `g` - The gauge origin.
    pub(crate) fn apply_magnetic_field(&mut self, b: &Vector3<f64>, g: &Point3<f64>) -> &mut Self {
        let k = 0.5 * b.cross(&(self.cart_origin.coords - g.coords));
        self.k = Some(k);
        self
    }
}

impl BasisShellContraction<f64, f64> {
    /// Computes the self-overlap ($`\mathcal{l}_2`$-norm) of this shell and divides in-place the
    /// contraction coefficients by ther square root of this, so that the functions in the shell
    /// are always normalised.
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

    /// Retrieves basis information from BasisSetExchange and constructs a vector of vectors of
    /// [`Self`] for a specified molecule. Each inner vector is for one atom in the molecule.
    ///
    /// This method produces basis name and function ordering that are uniform across all atoms and
    /// shells. The result from this method can be mutated for finer control of this.
    ///
    /// # Arguments
    ///
    /// * `mol` - A molecule.
    /// * `basis_name` - The name of the basis set to be retrieved.
    /// * `cart` - A boolean indicating if the shell functions should have lexicographic Cartesian
    /// ordering. If `false`, the shell functions shall have increasing-$`m`$ pure ordering
    /// instead.
    /// * `optimised_contraction` - A boolean indicating if the optimised contraction version of
    /// shells should be requested.
    /// * `version` - The requested version of the basis set information.
    /// * `mol_bohr` - A boolean indicating of the coordinates of the atoms in `mol` are to be
    /// interpreted in units of Bohr. If `false`, they are assumed to be in units of Ångström and
    /// will be converted to Bohr.
    /// * `force_renormalisation` - A boolean indicating if each shell is renormalised by scaling
    /// its primitive contraction coefficients by the inverse square root of its
    /// $\mathcal{l}_2$-norm.
    ///
    /// # Returns
    ///
    /// A vector of vectors of [`Self`].
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
        let bscs = mol
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
                                                .filter(|(_, d)| d.abs() > CONTRACTION_COEFF_THRESH)
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
/// A structure to manage basis information for a molecule.
#[derive(Clone, Debug)]
pub(crate) struct BasisSet<E, C> {
    /// A vector of vectors containing basis information for the atoms in this molecule. Each inner
    /// vector is for one atom.
    basis_atoms: Vec<Vec<BasisShellContraction<E, C>>>,

    /// The function boundaries for the atoms in the molecule.
    atom_boundaries: Vec<(usize, usize)>,

    /// The function boundaries for the shells in the molecule.
    shell_boundaries: Vec<(usize, usize)>,
}

impl<E, C> BasisSet<E, C> {
    /// Creates a new [`BasisSet`] structure from a vector of vectors of basis shells.
    ///
    /// # Arguments
    ///
    /// * `batms` - A vector of vectors of basis shells. Each inner vector is for one atom.
    ///
    /// # Returns
    ///
    /// A new [`BasisSet`] structure.
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

    /// Updates the cached shell boundaries. This is required when the shells or atoms have been
    /// reordered.
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

    /// Applies a uniform magnetic field to all shells and sets their plane-wave $`k`$ vectors.
    /// See the documentation of [`BasisShellContraction::apply_magnetic_field`] for more
    /// information.
    ///
    /// # Arguments
    ///
    /// * `b` - The magnetic field vector $`\mathbf{B}`$.
    /// * `g` - The gauge origin.
    pub(crate) fn apply_magnetic_field(&mut self, b: &Vector3<f64>, g: &Point3<f64>) -> &mut Self {
        self.all_shells_mut().for_each(|shell| {
            shell.apply_magnetic_field(b, g);
        });
        self
    }

    /// The number of shells in the basis set.
    pub(crate) fn n_shells(&self) -> usize {
        self.basis_atoms
            .iter()
            .map(|batm| batm.len())
            .sum::<usize>()
    }

    /// The number of basis functions in the basis set.
    pub(crate) fn n_funcs(&self) -> usize {
        self.all_shells().map(|shell| shell.basis_shell.n_funcs()).sum::<usize>()
    }

    /// Sorts the shells in each atom by their angular momenta.
    pub(crate) fn sort_by_angular_momentum(&mut self) -> &mut Self {
        self.basis_atoms
            .iter_mut()
            .for_each(|batm| batm.sort_by_key(|bsc| bsc.basis_shell.l));
        self.update_shell_boundaries()
    }

    /// Returns the function shell boundaries.
    pub(crate) fn shell_boundaries(&self) -> &Vec<(usize, usize)> {
        &self.shell_boundaries
    }

    /// Returns an iterator over all shells in the basis set.
    pub(crate) fn all_shells(&self) -> impl Iterator<Item = &BasisShellContraction<E, C>> {
        self.basis_atoms.iter().flatten()
    }

    /// Returns a mutable iterator over all shells in the basis set.
    pub(crate) fn all_shells_mut(
        &mut self,
    ) -> impl Iterator<Item = &mut BasisShellContraction<E, C>> {
        self.basis_atoms.iter_mut().flatten()
    }
}

impl BasisSet<f64, f64> {
    /// Retrieves basis information from BasisSetExchange and constructs [`Self`] for a specified
    /// molecule.
    ///
    /// This method produces basis name and function ordering that are uniform across all atoms and
    /// shells. The result from this method can be mutated for finer control of this.
    ///
    /// # Arguments
    ///
    /// * `mol` - A molecule.
    /// * `basis_name` - The name of the basis set to be retrieved.
    /// * `cart` - A boolean indicating if the shell functions should have lexicographic Cartesian
    /// ordering. If `false`, the shell functions shall have increasing-$`m`$ pure ordering
    /// instead.
    /// * `optimised_contraction` - A boolean indicating if the optimised contraction version of
    /// shells should be requested.
    /// * `version` - The requested version of the basis set information.
    /// * `mol_bohr` - A boolean indicating of the coordinates of the atoms in `mol` are to be
    /// interpreted in units of Bohr. If `false`, they are assumed to be in units of Ångström and
    /// will be converted to Bohr.
    /// * `force_renormalisation` - A boolean indicating if each shell is renormalised by scaling
    /// its primitive contraction coefficients by the inverse square root of its
    /// $\mathcal{l}_2$-norm.
    ///
    /// # Returns
    ///
    /// A [`BasisSet`] structure.
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
