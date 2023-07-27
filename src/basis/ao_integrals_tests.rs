use approx;
use nalgebra::{Vector3, Point3};

use crate::auxiliary::molecule::Molecule;
use crate::basis::ao_integrals::BasisSet;

const ROOT: &str = env!("CARGO_MANIFEST_DIR");

#[test]
fn test_ao_integrals_basis_set_frombse() {
    let mol = Molecule::from_xyz(&format!("{ROOT}/tests/xyz/xef4.xyz"), 1e-7);

    let bs = BasisSet::<f64, f64>::from_bse(
        &mol,
        "def2-TZVP",
        false, // cart
        true,  // optimised_contraction
        0,     // version
        true,  // mol_bohr
        false  // force_renormalisation
    ).unwrap();

    assert_eq!(bs.n_shells(), 60);
    assert_eq!(bs.n_funcs(), 174);
}

#[test]
fn test_ao_integrals_basis_set_magnetic_field() {
    let mol = Molecule::from_xyz(&format!("{ROOT}/tests/xyz/hocl.xyz"), 1e-7);

    let mut bs = BasisSet::<f64, f64>::from_bse(
        &mol,
        "STO-3G",
        false, // cart
        true,  // optimised_contraction
        0,     // version
        true,  // mol_bohr
        false  // force_renormalisation
    ).unwrap();
    bs.apply_magnetic_field(&Vector3::z(), &Point3::origin());

    for i in 0..3 {
        // O shells
        approx::assert_relative_eq!(bs[i].k().unwrap(), &Vector3::new(0.92288785, -2.319123, 0.0));
    }
    for i in 3..8 {
        // Cl shells
        approx::assert_relative_eq!(bs[i].k().unwrap(), &Vector3::new(0.64046835, -1.51670395, 0.0));
    }
    for i in 8..9 {
        // H shell
        approx::assert_relative_eq!(bs[i].k().unwrap(), &Vector3::new(0.86190385, -2.45865765, 0.0));
    }
}
