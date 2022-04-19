use crate::aux::atom::{Atom, ElementMap, AtomKind};
use approx;
use nalgebra::{DVector, Matrix3, Point3, Vector3};
use std::fs;
use std::process;

#[cfg(test)]
#[path = "sea_tests.rs"]
mod sea_tests;

/// A struct containing the atoms constituting a molecule.
#[derive(Clone, Debug)]
pub struct Molecule {
    /// The atoms constituting this molecule.
    atoms: Vec<Atom>,
    electric_atoms: Option<(Atom, Atom)>,
    magnetic_atoms: Option<(Atom, Atom)>,
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
    pub fn from_xyz(filename: &str) -> Molecule {
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
                atoms.push(Atom::from_xyz(&line, &emap).unwrap());
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
        }
    }

    /// Calculates the centre of mass of the molecule.
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
    /// # Arguments
    ///
    /// * `origin` - An origin about which the inertia tensor is evaluated.
    /// * `verbose` - The print level.
    ///
    /// # Returns
    ///
    /// The inertia tensor as a $3 \times 3$ matrix.
    pub fn calc_moi(&self, origin: &Point3<f64>, verbose: u64) -> Matrix3<f64> {
        let atoms = &self.atoms;
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

    /// Determines the sets of symmetry-equivalent atoms.
    ///
    /// # Arguments
    ///
    /// * `dist_thresh` - The threshold for distance comparison.
    /// * `verbose` - The print level.
    ///
    /// # Returns
    ///
    /// The list of sets of symmetry-equivalent atoms.
    pub fn calc_sea_groups(&self, dist_thresh: f64, verbose: u64) -> Vec<Vec<&Atom>> {
        let atoms = &self.atoms;
        let mut all_coords: Vec<&Point3<f64>> = vec![];
        let mut all_masses: Vec<f64> = vec![];
        for atom in atoms {
            all_coords.push(&atom.coordinates);
            all_masses.push(atom.atomic_mass);
        }
        let mut columns: Vec<DVector<f64>> = vec![];
        let decimals = -dist_thresh.log10().round() as i32;
        let rounding_factor = (10 as f64).powi(decimals);

        // Determine indices of symmetry-equivalent atoms
        let mut equiv_indicess: Vec<Vec<usize>> = vec![vec![0]];
        for (j, coord_j) in all_coords.iter().enumerate() {
            let mut column_j: Vec<f64> = vec![];
            for (i, coord_i) in all_coords.iter().enumerate() {
                let diff = *coord_j - *coord_i;
                column_j.push(
                    (diff.norm() / all_masses[i] * rounding_factor).round() / rounding_factor,
                );
            }
            column_j.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let column_j_vec = DVector::from_vec(column_j);
            if j == 0 {
                columns.push(column_j_vec);
            } else {
                let equiv_set_search = equiv_indicess.iter().position(|equiv_indices| {
                    columns[equiv_indices[0]].relative_eq(&column_j_vec, dist_thresh, dist_thresh)
                });
                columns.push(column_j_vec);
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

        // Convert indices to atom references
        let mut sea_groups: Vec<Vec<&Atom>> = vec![];
        for (i, equiv_indices) in equiv_indicess.iter().enumerate() {
            sea_groups.push(vec![]);
            for equiv_index in equiv_indices.into_iter() {
                sea_groups[i].push(&atoms[*equiv_index]);
            }
        }
        sea_groups
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
            self.magnetic_atoms = Some((
                Atom::new_special(AtomKind::Magnetic(true), com + b_vec).unwrap(),
                Atom::new_special(AtomKind::Magnetic(false), com - b_vec).unwrap(),
            ))
        } else { self.magnetic_atoms = None; }
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
            self.electric_atoms = Some((
                Atom::new_special(AtomKind::Electric(true), com + e_vec).unwrap(),
                Atom::new_special(AtomKind::Electric(false), com - 1.1*e_vec).unwrap(),
            ))
        } else { self.electric_atoms = None; }
    }
}
