use std::collections::{HashMap, HashSet};
use std::fs;
use std::process;

use log;
use nalgebra::{DVector, Matrix3, Point3, Vector3};
use num_traits::ToPrimitive;

use crate::aux::atom::{Atom, AtomKind, ElementMap};
use crate::aux::geometry::{self, ImproperRotationKind, Transform};
use crate::permutation::{PermutableCollection, Permutation};

#[cfg(test)]
#[path = "sea_tests.rs"]
mod sea_tests;

#[cfg(test)]
#[path = "molecule_tests.rs"]
mod molecule_tests;

// ==================
// Struct definitions
// ==================

/// A struct containing the atoms constituting a molecule.
#[derive(Clone, Debug)]
pub struct Molecule {
    /// The atoms constituting this molecule.
    pub atoms: Vec<Atom>,

    /// Optional special atom to represent the electric field applied to this molecule.
    pub electric_atoms: Option<[Atom; 1]>,

    /// Optional special atoms to represent the magnetic field applied to this molecule.
    pub magnetic_atoms: Option<Vec<Atom>>,

    /// A threshold for approximate equality comparisons.
    pub threshold: f64,
}

impl Molecule {
    /// Parses an `xyz` file to construct a molecule.
    ///
    /// # Arguments
    ///
    /// * `filename` - The `xyz` file to be parsed.
    ///
    /// # Returns
    ///
    /// The parsed [`Molecule`] struct.
    ///
    /// # Panics
    ///
    /// Panics when unable to parse the provided `xyz` file.
    #[must_use]
    pub fn from_xyz(filename: &str, thresh: f64) -> Self {
        let contents = fs::read_to_string(filename).unwrap_or_else(|err| {
            log::error!("Unable to read file {}.", filename);
            log::error!("{}", err);
            process::exit(1);
        });

        let mut atoms: Vec<Atom> = vec![];
        let emap = ElementMap::new();
        let mut n_atoms = 0usize;
        for (i, line) in contents.lines().enumerate() {
            if i == 0 {
                n_atoms = line.parse::<usize>().unwrap_or_else(|err| {
                    log::error!("Unable to read number of atoms in {}.", filename);
                    log::error!("{}", err);
                    process::exit(1);
                });
            } else if i == 1 {
                continue;
            } else {
                atoms.push(
                    Atom::from_xyz(line, &emap, thresh)
                        .unwrap_or_else(|| panic!("Unable to parse {line} to give an atom.")),
                );
            }
        }
        assert_eq!(
            atoms.len(),
            n_atoms,
            "Expected {} atoms, got {} instead.",
            n_atoms,
            atoms.len()
        );
        Molecule {
            atoms,
            electric_atoms: None,
            magnetic_atoms: None,
            threshold: thresh,
        }
    }

    /// Construct a molecule from an array of atoms.
    ///
    /// # Arguments
    ///
    /// * `all_atoms` - The atoms (of all types) constituting this molecule.
    /// * `threshold` - A threshold for approximate equality comparisons.
    ///
    /// # Returns
    ///
    /// The constructed [`Molecule`] struct.
    ///
    /// # Panics
    ///
    /// Panics when the numbers of fictitious special atoms, if any, are invalid. It is expected
    /// that, when present, there are two magnetic special atoms and/or one electric special atom.
    #[must_use]
    pub fn from_atoms(all_atoms: &[Atom], threshold: f64) -> Self {
        let atoms: Vec<Atom> = all_atoms
            .iter()
            .filter(|atom| matches!(atom.kind, AtomKind::Ordinary))
            .cloned()
            .collect();
        let magnetic_atoms_vec: Vec<Atom> = all_atoms
            .iter()
            .filter(|atom| matches!(atom.kind, AtomKind::Magnetic(_)))
            .cloned()
            .collect();
        assert_eq!(magnetic_atoms_vec.len() % 2, 0);
        let magnetic_atoms = if magnetic_atoms_vec.is_empty() {
            None
        } else {
            Some(magnetic_atoms_vec)
        };

        let electric_atoms_vec: Vec<Atom> = all_atoms
            .iter()
            .filter(|atom| matches!(atom.kind, AtomKind::Electric(_)))
            .cloned()
            .collect();
        assert!(electric_atoms_vec.len() == 1 || electric_atoms_vec.is_empty());
        let electric_atoms = if electric_atoms_vec.len() == 1 {
            Some([electric_atoms_vec[0].clone()])
        } else {
            None
        };

        Molecule {
            atoms,
            electric_atoms,
            magnetic_atoms,
            threshold,
        }
    }

    /// Retrieves a vector of references to all atoms in this molecule,
    /// including special ones, if any.
    ///
    /// # Returns
    ///
    /// All atoms in this molecule.
    #[must_use]
    pub fn get_all_atoms(&self) -> Vec<&Atom> {
        let mut atoms: Vec<&Atom> = vec![];
        for atom in &self.atoms {
            atoms.push(atom);
        }
        if let Some(magnetic_atoms) = &self.magnetic_atoms {
            for magnetic_atom in magnetic_atoms.iter() {
                atoms.push(magnetic_atom);
            }
        }
        if let Some(electric_atoms) = &self.electric_atoms {
            for electric_atom in electric_atoms.iter() {
                atoms.push(electric_atom);
            }
        }
        atoms
    }

    /// Calculates the centre of mass of the molecule.
    ///
    /// This does not take into account fictitious special atoms.
    ///
    /// # Returns
    ///
    /// The centre of mass.
    #[must_use]
    pub fn calc_com(&self) -> Point3<f64> {
        let atoms = &self.atoms;
        let mut com: Point3<f64> = Point3::origin();
        if atoms.is_empty() {
            return com;
        }
        let mut tot_m: f64 = 0.0;
        for atom in atoms.iter() {
            let m: f64 = atom.atomic_mass;
            com += atom.coordinates * m - Point3::origin();
            tot_m += m;
        }
        com *= 1.0 / tot_m;
        log::debug!("Centre of mass:");
        for component in com.iter() {
            log::debug!("  {component:+.14}");
        }
        com
    }

    /// Calculates the inertia tensor of the molecule.
    ///
    /// This *does* take into account fictitious special atoms.
    ///
    /// # Arguments
    ///
    /// * `origin` - An origin about which the inertia tensor is evaluated.
    ///
    /// # Returns
    ///
    /// The inertia tensor as a $`3 \times 3`$ matrix.
    #[must_use]
    pub fn calc_inertia_tensor(&self, origin: &Point3<f64>) -> Matrix3<f64> {
        let atoms = self.get_all_atoms();
        let mut inertia_tensor = Matrix3::zeros();
        for atom in &atoms {
            let rel_coordinates: Vector3<f64> = atom.coordinates - origin;
            for i in 0..3 {
                for j in 0..=i {
                    if i == j {
                        inertia_tensor[(i, j)] += atom.atomic_mass
                            * (rel_coordinates.norm_squared()
                                - rel_coordinates[i] * rel_coordinates[j]);
                    } else {
                        inertia_tensor[(i, j)] -=
                            atom.atomic_mass * rel_coordinates[i] * rel_coordinates[j];
                        inertia_tensor[(j, i)] -=
                            atom.atomic_mass * rel_coordinates[j] * rel_coordinates[i];
                    }
                }
            }
        }
        log::debug!("Origin for inertia tensor:");
        for component in origin.iter() {
            log::debug!("  {component:+.14}");
        }
        log::debug!("Inertia tensor:\n{}", inertia_tensor);
        inertia_tensor
    }

    /// Calculates the moments of inertia and the corresponding principal axes.
    ///
    /// This *does* take into account fictitious special atoms.
    ///
    /// # Returns
    ///
    /// * The moments of inertia in ascending order.
    /// * The corresponding principal axes.
    ///
    /// # Panics
    ///
    /// Panics when any of the moments of inertia cannot be compared.
    #[must_use]
    pub fn calc_moi(&self) -> ([f64; 3], [Vector3<f64>; 3]) {
        let inertia_eig = self.calc_inertia_tensor(&self.calc_com()).symmetric_eigen();
        let eigenvalues: Vec<f64> = inertia_eig.eigenvalues.iter().copied().collect();
        let eigenvectors: Vec<_> = inertia_eig.eigenvectors.column_iter().collect();
        let mut eigen_tuple: Vec<(f64, _)> = eigenvalues
            .iter()
            .copied()
            .zip(eigenvectors.iter().copied())
            .collect();
        eigen_tuple.sort_by(|(eigval0, _), (eigval1, _)| {
            eigval0
                .partial_cmp(eigval1)
                .unwrap_or_else(|| panic!("{eigval0} and {eigval1} cannot be compared."))
        });
        let (sorted_eigenvalues, sorted_eigenvectors): (Vec<f64>, Vec<_>) =
            eigen_tuple.into_iter().unzip();
        (
            [
                sorted_eigenvalues[0],
                sorted_eigenvalues[1],
                sorted_eigenvalues[2],
            ],
            [
                geometry::get_positive_pole(
                    &Vector3::new(
                        sorted_eigenvectors[0][(0, 0)],
                        sorted_eigenvectors[0][(1, 0)],
                        sorted_eigenvectors[0][(2, 0)],
                    ),
                    self.threshold,
                ),
                geometry::get_positive_pole(
                    &Vector3::new(
                        sorted_eigenvectors[1][(0, 0)],
                        sorted_eigenvectors[1][(1, 0)],
                        sorted_eigenvectors[1][(2, 0)],
                    ),
                    self.threshold,
                ),
                geometry::get_positive_pole(
                    &Vector3::new(
                        sorted_eigenvectors[2][(0, 0)],
                        sorted_eigenvectors[2][(1, 0)],
                        sorted_eigenvectors[2][(2, 0)],
                    ),
                    self.threshold,
                ),
            ],
        )
    }

    /// Determines the sets of symmetry-equivalent atoms.
    ///
    /// This *does* take into account fictitious special atoms.
    ///
    /// # Returns
    ///
    /// * Copies of the atoms in the molecule, grouped into symmetry-equivalent
    /// groups.
    ///
    /// # Panics
    ///
    /// Panics when the any of the mass-weighted interatomic distances cannot be compared.
    #[must_use]
    pub fn calc_sea_groups(&self) -> Vec<Vec<Atom>> {
        let atoms = &self.atoms;
        let all_atoms = &self.get_all_atoms();
        let ord_coords: Vec<_> = atoms.iter().map(|atm| atm.coordinates).collect();
        let all_coords: Vec<_> = all_atoms.iter().map(|atm| atm.coordinates).collect();
        let all_masses: Vec<_> = all_atoms.iter().map(|atm| atm.atomic_mass).collect();
        let mut dist_columns: Vec<DVector<f64>> = vec![];

        // Determine indices of symmetry-equivalent atoms
        let mut equiv_indicess: Vec<Vec<usize>> = vec![vec![0]];
        for (j, coord_j) in ord_coords.iter().enumerate() {
            // column_j is the j-th column in the mass-weighted interatomic
            // distance matrix. This column contains distances from ordinary atom j
            // to all other atoms (both ordinary and fictitious) in the molecule.
            // So this distance matrix is tall and thin when fictitious atoms are present.
            let mut column_j: Vec<f64> = vec![];
            for (i, coord_i) in all_coords.iter().enumerate() {
                let diff = coord_j - coord_i;
                column_j.push(diff.norm() / all_masses[i]);
            }
            column_j.sort_by(|a, b| {
                a.partial_cmp(b)
                    .unwrap_or_else(|| panic!("{a} and {b} cannot be compared."))
            });
            let column_j_vec = DVector::from_vec(column_j);
            if j == 0 {
                dist_columns.push(column_j_vec);
            } else {
                let equiv_set_search = equiv_indicess.iter().position(|equiv_indices| {
                    dist_columns[equiv_indices[0]].relative_eq(
                        &column_j_vec,
                        self.threshold,
                        self.threshold,
                    )
                });
                dist_columns.push(column_j_vec);
                if let Some(index) = equiv_set_search {
                    equiv_indicess[index].push(j);
                } else {
                    equiv_indicess.push(vec![j]);
                };
            }
        }
        let mut sea_groups: Vec<Vec<Atom>> = equiv_indicess
            .iter()
            .map(|equiv_indices| {
                equiv_indices
                    .iter()
                    .map(|index| atoms[*index].clone())
                    .collect()
            })
            .collect();

        if let Some(magnetic_atoms) = &self.magnetic_atoms {
            sea_groups.push(vec![magnetic_atoms[0].clone(), magnetic_atoms[1].clone()]);
        }
        if let Some(electric_atoms) = &self.electric_atoms {
            sea_groups.push(vec![electric_atoms[0].clone()]);
        }
        log::debug!("Number of SEA groups: {}", sea_groups.len());
        sea_groups
    }

    /// Adds two fictitious magnetic atoms to represent the magnetic field.
    ///
    /// # Arguments
    ///
    /// * `magnetic_field` - The magnetic field vector. If zero or `None`, any magnetic
    /// field present will be removed.
    ///
    /// # Panics
    ///
    /// Panics when the number of atoms cannot be represented as an `f64` value.
    pub fn set_magnetic_field(&mut self, magnetic_field: Option<Vector3<f64>>) {
        if let Some(b_vec) = magnetic_field {
            if approx::relative_ne!(b_vec.norm(), 0.0) {
                let com = self.calc_com();
                let ave_mag = {
                    let average_distance = self
                        .atoms
                        .iter()
                        .fold(0.0, |acc, atom| acc + (atom.coordinates - com).magnitude())
                        / self.atoms.len().to_f64().unwrap_or_else(|| {
                            panic!("Unable to convert `{}` to `f64`.", self.atoms.len())
                        });
                    if average_distance > 0.0 {
                        average_distance
                    } else {
                        0.5
                    }
                };
                let b_vec_norm = b_vec.normalize() * ave_mag * 0.5;
                self.magnetic_atoms = Some(vec![
                    Atom::new_special(AtomKind::Magnetic(true), com + b_vec_norm, self.threshold)
                        .expect("Unable to construct a special magnetic atom."),
                    Atom::new_special(AtomKind::Magnetic(false), com - b_vec_norm, self.threshold)
                        .expect("Unable to construct a special magnetic atom."),
                ]);
            } else {
                self.magnetic_atoms = None;
            }
        } else {
            self.magnetic_atoms = None;
        }
    }

    /// Adds one fictitious electric atom to represent the electric field.
    ///
    /// # Arguments
    ///
    /// * `electric_field` - The electric field vector. If zero or `None`, any electric
    /// field present will be removed.
    ///
    /// # Panics
    ///
    /// Panics when the number of atoms cannot be represented as an `f64` value.
    pub fn set_electric_field(&mut self, electric_field: Option<Vector3<f64>>) {
        if let Some(e_vec) = electric_field {
            if approx::relative_ne!(e_vec.norm(), 0.0) {
                let com = self.calc_com();
                let ave_mag = {
                    let average_distance = self
                        .atoms
                        .iter()
                        .fold(0.0, |acc, atom| acc + (atom.coordinates - com).magnitude())
                        / self.atoms.len().to_f64().unwrap_or_else(|| {
                            panic!("Unable to convert `{}` to `f64`.", self.atoms.len())
                        });
                    if average_distance > 0.0 {
                        average_distance
                    } else {
                        0.5
                    }
                };
                let e_vec_norm = e_vec.normalize() * ave_mag * 0.5;
                self.electric_atoms = Some([Atom::new_special(
                    AtomKind::Electric(true),
                    com + e_vec_norm,
                    self.threshold,
                )
                .expect("Unable to construct an electric special atom.")]);
            } else {
                self.electric_atoms = None;
            }
        } else {
            self.electric_atoms = None;
        }
    }
}

// =====================
// Trait implementations
// =====================

impl Transform for Molecule {
    fn transform_mut(&mut self, mat: &Matrix3<f64>) {
        for atom in &mut self.atoms {
            atom.transform_mut(mat);
        }
        if let Some(ref mut mag_atoms) = self.magnetic_atoms {
            for atom in mag_atoms.iter_mut() {
                atom.transform_mut(mat);
            }
        }
        if let Some(ref mut ele_atoms) = self.electric_atoms {
            for atom in ele_atoms.iter_mut() {
                atom.transform_mut(mat);
            }
        }
    }

    fn rotate_mut(&mut self, angle: f64, axis: &Vector3<f64>) {
        for atom in &mut self.atoms {
            atom.rotate_mut(angle, axis);
        }
        if let Some(ref mut mag_atoms) = self.magnetic_atoms {
            for atom in mag_atoms.iter_mut() {
                atom.rotate_mut(angle, axis);
            }
        }
        if let Some(ref mut ele_atoms) = self.electric_atoms {
            for atom in ele_atoms.iter_mut() {
                atom.rotate_mut(angle, axis);
            }
        }
    }

    fn improper_rotate_mut(
        &mut self,
        angle: f64,
        axis: &Vector3<f64>,
        kind: &ImproperRotationKind,
    ) {
        for atom in &mut self.atoms {
            atom.improper_rotate_mut(angle, axis, kind);
        }
        if let Some(ref mut mag_atoms) = self.magnetic_atoms {
            for atom in mag_atoms.iter_mut() {
                atom.improper_rotate_mut(angle, axis, kind);
            }
        }
        if let Some(ref mut ele_atoms) = self.electric_atoms {
            for atom in ele_atoms.iter_mut() {
                atom.improper_rotate_mut(angle, axis, kind);
            }
        }
    }

    fn translate_mut(&mut self, tvec: &Vector3<f64>) {
        for atom in &mut self.atoms {
            atom.translate_mut(tvec);
        }
        if let Some(ref mut mag_atoms) = self.magnetic_atoms {
            for atom in mag_atoms.iter_mut() {
                atom.translate_mut(tvec);
            }
        }
        if let Some(ref mut ele_atoms) = self.electric_atoms {
            for atom in ele_atoms.iter_mut() {
                atom.translate_mut(tvec);
            }
        }
    }

    fn recentre_mut(&mut self) {
        let com = self.calc_com();
        let tvec = -Vector3::new(com[0], com[1], com[2]);
        self.translate_mut(&tvec);
    }

    fn reverse_time_mut(&mut self) {
        if let Some(ref mut mag_atoms) = self.magnetic_atoms {
            for atom in mag_atoms.iter_mut() {
                atom.reverse_time_mut();
            }
        }
    }

    fn transform(&self, mat: &Matrix3<f64>) -> Self {
        let mut transformed_mol = self.clone();
        transformed_mol.transform_mut(mat);
        transformed_mol
    }

    fn rotate(&self, angle: f64, axis: &Vector3<f64>) -> Self {
        let mut rotated_mol = self.clone();
        rotated_mol.rotate_mut(angle, axis);
        rotated_mol
    }

    fn improper_rotate(
        &self,
        angle: f64,
        axis: &Vector3<f64>,
        kind: &ImproperRotationKind,
    ) -> Self {
        let mut improper_rotated_mol = self.clone();
        improper_rotated_mol.improper_rotate_mut(angle, axis, kind);
        improper_rotated_mol
    }

    fn translate(&self, tvec: &Vector3<f64>) -> Self {
        let mut translated_mol = self.clone();
        translated_mol.translate_mut(tvec);
        translated_mol
    }

    fn recentre(&self) -> Self {
        let mut recentred_mol = self.clone();
        recentred_mol.recentre_mut();
        recentred_mol
    }

    fn reverse_time(&self) -> Self {
        let mut time_reversed_mol = self.clone();
        time_reversed_mol.reverse_time_mut();
        time_reversed_mol
    }
}

impl PartialEq for Molecule {
    fn eq(&self, other: &Self) -> bool {
        if self.atoms.len() != other.atoms.len() {
            return false;
        };
        let thresh = self
            .atoms
            .iter()
            .chain(other.atoms.iter())
            .fold(0.0_f64, |acc, atom| acc.max(atom.threshold));

        let mut other_atoms_ref: HashSet<_> = other.atoms.iter().collect();
        for s_atom in &self.atoms {
            let o_atom = other
                .atoms
                .iter()
                .find(|o_atm| (s_atom.coordinates - o_atm.coordinates).norm() < thresh);
            match o_atom {
                Some(atm) => {
                    other_atoms_ref.remove(atm);
                }
                None => {
                    break;
                }
            }
        }
        if !other_atoms_ref.is_empty() {
            return false;
        }

        if let Some(self_mag_atoms) = &self.magnetic_atoms {
            if let Some(other_mag_atoms) = &other.magnetic_atoms {
                let mut other_mag_atoms_ref: HashSet<_> = other_mag_atoms.iter().collect();
                for s_atom in self_mag_atoms.iter() {
                    let o_atom = other_mag_atoms.iter().find(|o_atm| {
                        (s_atom.coordinates - o_atm.coordinates).norm() < thresh
                            && s_atom.kind == o_atm.kind
                    });
                    match o_atom {
                        Some(atm) => {
                            other_mag_atoms_ref.remove(atm);
                        }
                        None => {
                            break;
                        }
                    }
                }
                if !other_mag_atoms_ref.is_empty() {
                    return false;
                }
            } else {
                return false;
            }
        } else if other.magnetic_atoms.is_some() {
            return false;
        };

        if let Some(self_ele_atoms) = &self.electric_atoms {
            if let Some(other_ele_atoms) = &other.electric_atoms {
                let mut other_ele_atoms_ref: HashSet<_> = other_ele_atoms.iter().collect();
                for s_atom in self_ele_atoms.iter() {
                    let o_atom = other_ele_atoms.iter().find(|o_atm| {
                        (s_atom.coordinates - o_atm.coordinates).norm() < thresh
                            && s_atom.kind == o_atm.kind
                    });
                    match o_atom {
                        Some(atm) => {
                            other_ele_atoms_ref.remove(atm);
                        }
                        None => {
                            break;
                        }
                    }
                }
                if !other_ele_atoms_ref.is_empty() {
                    return false;
                }
            } else {
                return false;
            }
        } else if other.electric_atoms.is_some() {
            return false;
        };
        true
    }
}

impl PermutableCollection for Molecule {
    type Item = Atom;

    /// Determines the permutation of *ordinary* atoms to map `self` to `other`. Special fictitious
    /// atoms are not included.
    fn perm(&self, other: &Self) -> Option<Permutation> {
        let o_atoms: HashMap<Atom, usize> = other
            .atoms
            .iter()
            .enumerate()
            .map(|(i, atom)| (atom.clone(), i))
            .collect();
        let image_opt: Option<Vec<usize>> = self
            .atoms
            .iter()
            .map(|s_atom| o_atoms.get(s_atom).copied())
            .collect();
        image_opt.map(|image| Permutation::from_image(&image))
    }
}
