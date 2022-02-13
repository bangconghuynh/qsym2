use crate::aux::atom::{Atom, ElementMap};
use crate::aux::geometry::Point3D;
use nalgebra as na;
use std::fs;
use std::process;

pub struct Molecule {
    /// The atoms constituting this molecule.
    atoms: Vec<Atom>,
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
        Molecule { atoms }
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
    pub fn calc_com(&self, verbose: u64) -> Point3D<f64> {
        let atoms = &self.atoms;
        let mut com: Point3D<f64> = Point3D {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        };
        let mut tot_m: f64 = 0.0;
        for atom in atoms.iter() {
            let m: f64 = atom.atomic_mass;
            com += &atom.coordinates * m;
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
    pub fn calc_moi(&self, origin: &Point3D<f64>, verbose: u64) -> na::Matrix3<f64> {
        let atoms = &self.atoms;
        let mut inertia_tensor = na::Matrix3::from_element(0.0);
        for atom in atoms.iter() {
            let rel_coordinates: Point3D<f64> = &atom.coordinates - origin;
            for i in 0..3 {
                for j in 0..=i {
                    if i != j {
                        inertia_tensor[(i, j)] -=
                            atom.atomic_mass * rel_coordinates[i] * rel_coordinates[j];
                        inertia_tensor[(j, i)] -=
                            atom.atomic_mass * rel_coordinates[j] * rel_coordinates[i];
                    } else {
                        inertia_tensor[(i, j)] += atom.atomic_mass
                            * (rel_coordinates.sq_norm() - rel_coordinates[i] * rel_coordinates[j]);
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
}
