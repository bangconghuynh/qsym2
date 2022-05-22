use crate::aux::atom::{Atom, AtomKind, ElementMap};
use crate::aux::geometry::{self, Transform};
use crate::aux::misc::HashableFloat;
use crate::symmetry::symmetry_element::SymmetryElementKind;
use nalgebra::{DVector, Matrix, Matrix3, Point3, Vector3};
use std::collections::HashSet;
use std::fs;
use std::process;

#[cfg(test)]
#[path = "sea_tests.rs"]
mod sea_tests;

#[cfg(test)]
#[path = "molecule_tests.rs"]
mod molecule_tests;

/// A struct containing the atoms constituting a molecule.
#[derive(Clone, Debug)]
pub struct Molecule {
    /// The atoms constituting this molecule.
    pub atoms: Vec<Atom>,

    /// Optional special atoms to represent the electric field applied to this molecule.
    electric_atoms: Option<[Atom; 2]>,

    /// Optional special atoms to represent the magnetic field applied to this molecule.
    magnetic_atoms: Option<[Atom; 2]>,

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
    pub fn from_xyz(filename: &str, thresh: f64) -> Molecule {
        let contents = fs::read_to_string(filename).unwrap_or_else(|err| {
            println!("Unable to read file {}.", filename);
            println!("{}", err);
            process::exit(1);
        });

        let mut atoms: Vec<Atom> = vec![];
        let emap = ElementMap::new();
        let mut n_atoms = 0usize;
        for (i, line) in contents.lines().enumerate() {
            if i == 0 {
                n_atoms = line.parse::<usize>().unwrap();
            } else if i == 1 {
                continue;
            } else {
                atoms.push(Atom::from_xyz(&line, &emap, thresh).unwrap());
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

    /// Retrieves a vector of references to all atoms in this molecule,
    /// including special ones, if any.
    ///
    /// # Returns
    ///
    /// All atoms in this molecule.
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
    /// # Arguments
    ///
    /// * `verbose` - The print level.
    ///
    /// # Returns
    ///
    /// The centre of mass.
    pub fn calc_com(&self, verbose: u64) -> Point3<f64> {
        let atoms = &self.atoms;
        let mut com: Point3<f64> = Point3::origin();
        let mut tot_m: f64 = 0.0;
        for atom in atoms.iter() {
            let m: f64 = atom.atomic_mass;
            com += &atom.coordinates * m - Point3::origin();
            tot_m += m;
        }
        com *= 1.0 / tot_m;
        if verbose > 0 {
            println!("Centre of mass: {}", com);
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
    /// * `verbose` - The print level.
    ///
    /// # Returns
    ///
    /// The inertia tensor as a $3 \times 3$ matrix.
    pub fn calc_inertia_tensor(&self, origin: &Point3<f64>, verbose: u64) -> Matrix3<f64> {
        let atoms = self.get_all_atoms();
        let mut inertia_tensor = Matrix3::zeros();
        for atom in atoms.iter() {
            let rel_coordinates: Vector3<f64> = &atom.coordinates - origin;
            for i in 0..3 {
                for j in 0..=i {
                    if i != j {
                        inertia_tensor[(i, j)] -=
                            atom.atomic_mass * rel_coordinates[i] * rel_coordinates[j];
                        inertia_tensor[(j, i)] -=
                            atom.atomic_mass * rel_coordinates[j] * rel_coordinates[i];
                    } else {
                        inertia_tensor[(i, j)] += atom.atomic_mass
                            * (rel_coordinates.norm_squared()
                                - rel_coordinates[i] * rel_coordinates[j]);
                    }
                }
            }
        }
        if verbose > 1 {
            println!("Origin for inertia tensor: {}", origin);
            println!("Inertia tensor:\n{}", inertia_tensor);
        }
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
    fn calc_moi(&self) -> ([f64; 3], [Vector3<f64>; 3]) {
        let inertia_eig = self
            .calc_inertia_tensor(&self.calc_com(0), 0)
            .symmetric_eigen();
        let eigenvalues: Vec<f64> = inertia_eig.eigenvalues.iter().cloned().collect();
        let eigenvectors: Vec<_> = inertia_eig.eigenvectors.column_iter().collect();
        let mut eigen_tuple: Vec<(f64, _)> = eigenvalues
            .iter()
            .cloned()
            .zip(eigenvectors.iter().cloned())
            .collect();
        eigen_tuple.sort_by(|(eigval0, _), (eigval1, _)| eigval0.partial_cmp(eigval1).unwrap());
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
                    ), self.threshold
                ),
                geometry::get_positive_pole(
                    &Vector3::new(
                        sorted_eigenvectors[1][(0, 0)],
                        sorted_eigenvectors[1][(1, 0)],
                        sorted_eigenvectors[1][(2, 0)],
                    ), self.threshold
                ),
                geometry::get_positive_pole(
                    &Vector3::new(
                        sorted_eigenvectors[2][(0, 0)],
                        sorted_eigenvectors[2][(1, 0)],
                        sorted_eigenvectors[2][(2, 0)],
                    ), self.threshold
                ),
            ],
        )
    }

    /// Determines the sets of symmetry-equivalent atoms.
    ///
    /// # Arguments
    ///
    /// * `dist_thresh` - The threshold for distance comparison.
    /// * `verbose` - The print level.
    ///
    /// # Returns
    ///
    /// The vector of vectors of symmetry-equivalent atom indices.
    pub fn calc_sea_groups(&self, verbose: u64) -> Vec<Vec<usize>> {
        let atoms = self.get_all_atoms();
        let all_coords: Vec<_> = atoms.iter().map(|atm| atm.coordinates).collect();
        let all_masses: Vec<_> = atoms.iter().map(|atm| atm.atomic_mass).collect();
        let mut dist_columns: Vec<DVector<f64>> = vec![];

        // Determine indices of symmetry-equivalent atoms
        let mut equiv_indicess: Vec<Vec<usize>> = vec![vec![0]];
        for (j, coord_j) in all_coords.iter().enumerate() {
            // column_j is the j-th column in the mass-weighted interatomic
            // distance matrix.
            let mut column_j: Vec<f64> = vec![];
            for (i, coord_i) in all_coords.iter().enumerate() {
                let diff = coord_j - coord_i;
                column_j.push(diff.norm() / all_masses[i]);
            }
            column_j.sort_by(|a, b| a.partial_cmp(b).unwrap());
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
        if verbose > 0 {
            println!("Number of SEA groups: {}", equiv_indicess.len());
        }
        equiv_indicess
    }

    /// Adds two fictitious magnetic atoms to represent the magnetic field.
    ///
    /// # Arguments
    ///
    /// * magnetic_field - The magnetic field vector. If `None`, any magnetic
    /// field present will be removed.
    pub fn set_magnetic_field(&mut self, magnetic_field: Option<Vector3<f64>>) {
        if let Some(b_vec) = magnetic_field {
            approx::assert_relative_ne!(b_vec.norm(), 0.0);
            let com = self.calc_com(0);
            self.magnetic_atoms = Some([
                Atom::new_special(AtomKind::Magnetic(true), com + b_vec, self.threshold).unwrap(),
                Atom::new_special(AtomKind::Magnetic(false), com - b_vec, self.threshold).unwrap(),
            ])
        } else {
            self.magnetic_atoms = None;
        }
    }

    /// Adds two fictitious magnetic atoms to represent the electric field.
    ///
    /// # Arguments
    ///
    /// * electric_field - The electric field vector. If `None`, any magnetic
    /// field present will be removed.
    pub fn set_electric_field(&mut self, electric_field: Option<Vector3<f64>>) {
        if let Some(e_vec) = electric_field {
            approx::assert_relative_ne!(e_vec.norm(), 0.0);
            let com = self.calc_com(0);
            self.electric_atoms = Some([
                Atom::new_special(AtomKind::Electric(true), com + 1.1 * e_vec, self.threshold)
                    .unwrap(),
                Atom::new_special(AtomKind::Electric(false), com - e_vec, self.threshold).unwrap(),
            ])
        } else {
            self.electric_atoms = None;
        }
    }
}

impl Transform for Molecule {
    fn transform_mut(self: &mut Self, mat: &Matrix3<f64>) {
        for atom in self.atoms.iter_mut() {
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

    fn rotate_mut(self: &mut Self, angle: f64, axis: &Vector3<f64>) {
        for atom in self.atoms.iter_mut() {
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
        self: &mut Self,
        angle: f64,
        axis: &Vector3<f64>,
        kind: &SymmetryElementKind,
    ) {
        for atom in self.atoms.iter_mut() {
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

    fn translate_mut(self: &mut Self, tvec: &Vector3<f64>) {
        for atom in self.atoms.iter_mut() {
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

    fn recentre_mut(self: &mut Self) {
        let com = self.calc_com(0);
        let tvec = -Vector3::new(com[0], com[1], com[2]);
        self.translate_mut(&tvec);
    }

    fn transform(self: &Self, mat: &Matrix3<f64>) -> Self {
        let mut transformed_mol = self.clone();
        transformed_mol.transform_mut(mat);
        transformed_mol
    }

    fn rotate(self: &Self, angle: f64, axis: &Vector3<f64>) -> Self {
        let mut rotated_mol = self.clone();
        rotated_mol.rotate_mut(angle, axis);
        rotated_mol
    }

    fn improper_rotate(
        self: &Self,
        angle: f64,
        axis: &Vector3<f64>,
        kind: &SymmetryElementKind,
    ) -> Self {
        let mut improper_rotated_mol = self.clone();
        improper_rotated_mol.improper_rotate_mut(angle, axis, kind);
        improper_rotated_mol
    }

    fn translate(self: &Self, tvec: &Vector3<f64>) -> Self {
        let mut translated_mol = self.clone();
        translated_mol.translate_mut(tvec);
        translated_mol
    }

    fn recentre(self: &Self) -> Self {
        let mut recentred_mol = self.clone();
        recentred_mol.recentre_mut();
        recentred_mol
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
        for s_atom in self.atoms.iter() {
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
        if other_atoms_ref.len() != 0 {
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
                if other_mag_atoms_ref.len() != 0 {
                    return false;
                }
            } else {
                return false;
            }
        } else {
            if let Some(_) = other.magnetic_atoms {
                return false;
            }
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
                if other_ele_atoms_ref.len() != 0 {
                    return false;
                }
            } else {
                return false;
            }
        } else {
            if let Some(_) = other.electric_atoms {
                return false;
            }
        };
        true
    }
}

// impl Eq for Molecule {}
