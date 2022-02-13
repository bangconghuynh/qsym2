use nalgebra as na;
use std::fmt;

use crate::aux::atom::Atom;
use crate::aux::geometry::Point3D;


fn diff<T: na::RealField + Copy>(a: T, b: T, abs_compare: bool) -> T {
    if abs_compare {
        (a - b).abs()
    } else {
        (a - b).abs() / (a + b).abs()
    }
}

pub enum RotationalSymmetry {
    Spherical,
    OblatePlanar,
    OblateNonPlanar,
    ProlateLinear,
    ProlateNonLinear,
    AsymmetricPlanar,
    AsymmetricNonPlanar,
}

impl fmt::Display for RotationalSymmetry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RotationalSymmetry::Spherical => write!(f, "Spherical"),
            RotationalSymmetry::OblatePlanar => write!(f, "Oblate, planar"),
            RotationalSymmetry::OblateNonPlanar => write!(f, "Oblate, non-planar"),
            RotationalSymmetry::ProlateLinear => write!(f, "Prolate, linear"),
            RotationalSymmetry::ProlateNonLinear => write!(f, "Prolate, non-linear"),
            RotationalSymmetry::AsymmetricPlanar => write!(f, "Asymmetric, planar"),
            RotationalSymmetry::AsymmetricNonPlanar => write!(f, "Asymmetric, non-planar"),
        }
    }
}

pub fn calc_com(atoms: &Vec<Atom>, verbose: u64) -> Point3D<f64> {
    let mut num: Point3D<f64> = Point3D {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };
    let mut tot_m: f64 = 0.0;
    for atom in atoms.iter() {
        let m: f64 = atom.atomic_mass;
        num += &atom.coordinates * m;
        tot_m += m;
    }
    num *= 1.0 / tot_m;
    if verbose > 0 {
        println!("Centre of mass: {}", num);
    }
    num
}

pub fn calc_moi(atoms: &mut Vec<Atom>, com: &Point3D<f64>, verbose: u64) -> na::Matrix3<f64> {
    let mut inertia_tensor = na::Matrix3::from_element(0.0);
    for atom in atoms.iter_mut() {
        atom.coordinates -= com;
        for i in 0..3 {
            for j in 0..=i {
                if i != j {
                    inertia_tensor[(i, j)] -=
                        atom.atomic_mass * atom.coordinates[i] * atom.coordinates[j];
                    inertia_tensor[(j, i)] -=
                        atom.atomic_mass * atom.coordinates[j] * atom.coordinates[i];
                } else {
                    inertia_tensor[(i, j)] += atom.atomic_mass
                        * (atom.coordinates.sq_norm() - atom.coordinates[i] * atom.coordinates[j]);
                }
            }
        }
    }
    if verbose > 1 {
        println!("Recentred structure:\n{:#?}", atoms);
        println!("Inertia tensor:\n{}", inertia_tensor);
    }
    inertia_tensor
}

pub fn calc_rotational_symmetry(
    inertia_tensor: &na::Matrix3<f64>,
    thresh: f64,
    verbose: u64,
    abs_compare: bool,
) -> RotationalSymmetry {
    let moi_mat = inertia_tensor.symmetric_eigenvalues();
    let mut moi: Vec<&f64> = moi_mat.iter().collect();
    moi.sort_by(|a, b| a.partial_cmp(b).unwrap());
    if verbose > 0 {
        println!(
            "Moments of inertia:\n {:.6}\n {:.6}\n {:.6}",
            moi[0], moi[1], moi[2]
        );
        if abs_compare {
            println!("Threshold for absolute MoI comparison: {:.3e}", thresh);
        } else {
            println!("Threshold for relative MoI comparison: {:.3e}", thresh);
        }
    }
    if diff(*moi[0], *moi[1], abs_compare) < thresh {
        if diff(*moi[1], *moi[2], abs_compare) < thresh {
            return RotationalSymmetry::Spherical;
        }
        if diff(*moi[2], *moi[0] + *moi[1], abs_compare) < thresh {
            return RotationalSymmetry::OblatePlanar;
        }
        return RotationalSymmetry::OblateNonPlanar;
    }
    if diff(*moi[1], *moi[2], abs_compare) < thresh {
        if moi[0].abs() < thresh {
            return RotationalSymmetry::ProlateLinear;
        }
        return RotationalSymmetry::ProlateNonLinear;
    }
    if diff(*moi[2], *moi[0] + *moi[1], abs_compare) < thresh {
        return RotationalSymmetry::AsymmetricPlanar;
    }
    RotationalSymmetry::AsymmetricNonPlanar
}
