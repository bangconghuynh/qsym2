use approx::assert_abs_diff_eq;

use crate::angmom::spinor_rotation_3d::SpinConstraint;
use crate::auxiliary::atom::ElementMap;

use crate::target::noci::backend::auxiliary::extract_pyscf_scf_data;

const ROOT: &str = env!("CARGO_MANIFEST_DIR");

#[test]
fn test_hamiltonian_scf_energy_h3_631gs() {
    let emap = ElementMap::new();
    let filename = format!("{ROOT}/tests/noci_backend_hdf5/h3_631gs.hdf5");
    let pyscf_data = extract_pyscf_scf_data::<_, f64>(filename).unwrap();
    let mol = pyscf_data.get_mol(&emap, 1e-7).unwrap();
    let bao = pyscf_data.get_bao(&mol).unwrap();
    let (overlap_ao, hamiltonian_ao) = pyscf_data.get_integrals::<SpinConstraint>().unwrap();
    let det = pyscf_data
        .get_slater_determinant(&mol, &bao, SpinConstraint::Unrestricted(2, true), 1e-14)
        .unwrap();

    // -----------------------------------
    // SCF energy of reference determinant
    // -----------------------------------
    let (zeroe, onee, twoe) = hamiltonian_ao
        .calc_hamiltonian_matrix_element_contributions(&det, &det, overlap_ao.sao(), 1e-10, 1e-7)
        .unwrap();
    // PySCF reference values
    assert_abs_diff_eq!(zeroe, hamiltonian_ao.enuc, epsilon = 1e-7);
    assert_abs_diff_eq!(onee, pyscf_data.scf_e_1, epsilon = 1e-7);
    assert_abs_diff_eq!(twoe, pyscf_data.scf_e_2, epsilon = 1e-5);
}

#[test]
fn test_hamiltonian_scf_energy_h3_sto3g() {
    let emap = ElementMap::new();
    let filename = format!("{ROOT}/tests/noci_backend_hdf5/h3_sto3g.hdf5");
    let pyscf_data = extract_pyscf_scf_data::<_, f64>(filename).unwrap();
    let mol = pyscf_data.get_mol(&emap, 1e-7).unwrap();
    let bao = pyscf_data.get_bao(&mol).unwrap();
    let (overlap_ao, hamiltonian_ao) = pyscf_data.get_integrals::<SpinConstraint>().unwrap();
    let det = pyscf_data
        .get_slater_determinant(&mol, &bao, SpinConstraint::Unrestricted(2, true), 1e-14)
        .unwrap();

    // -----------------------------------
    // SCF energy of reference determinant
    // -----------------------------------
    let (zeroe, onee, twoe) = hamiltonian_ao
        .calc_hamiltonian_matrix_element_contributions(&det, &det, overlap_ao.sao(), 1e-10, 1e-7)
        .unwrap();
    // PySCF reference values
    assert_abs_diff_eq!(zeroe, hamiltonian_ao.enuc, epsilon = 1e-7);
    assert_abs_diff_eq!(onee, pyscf_data.scf_e_1, epsilon = 1e-7);
    assert_abs_diff_eq!(twoe, pyscf_data.scf_e_2, epsilon = 1e-4);
}

#[test]
fn test_hamiltonian_scf_energy_ch4p_sto3g() {
    let emap = ElementMap::new();
    let filename = format!("{ROOT}/tests/noci_backend_hdf5/ch4p_sto3g.hdf5");
    let pyscf_data = extract_pyscf_scf_data::<_, f64>(filename).unwrap();
    let mol = pyscf_data.get_mol(&emap, 1e-7).unwrap();
    let bao = pyscf_data.get_bao(&mol).unwrap();
    let (overlap_ao, hamiltonian_ao) = pyscf_data.get_integrals::<SpinConstraint>().unwrap();
    let det = pyscf_data
        .get_slater_determinant(&mol, &bao, SpinConstraint::Unrestricted(2, true), 1e-14)
        .unwrap();

    // -----------------------------------
    // SCF energy of reference determinant
    // -----------------------------------
    let (zeroe, onee, twoe) = hamiltonian_ao
        .calc_hamiltonian_matrix_element_contributions(&det, &det, overlap_ao.sao(), 1e-10, 1e-7)
        .unwrap();
    // PySCF reference values
    assert_abs_diff_eq!(zeroe, hamiltonian_ao.enuc, epsilon = 1e-7);
    assert_abs_diff_eq!(onee, pyscf_data.scf_e_1, epsilon = 1e-7);
    assert_abs_diff_eq!(twoe, pyscf_data.scf_e_2, epsilon = 1e-3);
}
