use approx::assert_abs_diff_eq;

use crate::angmom::spinor_rotation_3d::SpinConstraint;
use crate::auxiliary::atom::ElementMap;
use crate::target::noci::backend::auxiliary::extract_pyscf_scf_data;

const ROOT: &str = env!("CARGO_MANIFEST_DIR");

fn test_hamiltonian_scf_energy_pyscf(molname: &str) -> () {
    let emap = ElementMap::new();
    let filename = format!("{ROOT}/tests/noci_backend_hdf5/{molname}.hdf5");
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
    assert_abs_diff_eq!(zeroe, hamiltonian_ao.enuc, epsilon = 1e-12);
    assert_abs_diff_eq!(onee, pyscf_data.scf_e_1, epsilon = 1e-12);
    assert_abs_diff_eq!(twoe, pyscf_data.scf_e_2, epsilon = 1e-12);
    assert_abs_diff_eq!(zeroe + onee + twoe, pyscf_data.scf_e_scf, epsilon = 1e-12);
}

#[test]
fn test_hamiltonian_scf_energy_h3_631gs() {
    test_hamiltonian_scf_energy_pyscf("h3_631gs");
}

#[test]
fn test_hamiltonian_scf_energy_h3_sto3g() {
    test_hamiltonian_scf_energy_pyscf("h3_sto3g");
}

#[test]
fn test_hamiltonian_scf_energy_h4_sto3g() {
    test_hamiltonian_scf_energy_pyscf("h4_sto3g");
}

#[test]
fn test_hamiltonian_scf_energy_h4_631gds() {
    test_hamiltonian_scf_energy_pyscf("h4_631gds");
}

#[test]
fn test_hamiltonian_scf_energy_ch4p_sto3g() {
    test_hamiltonian_scf_energy_pyscf("ch4p_sto3g");
}

#[test]
fn test_hamiltonian_scf_energy_ch4p_631gds() {
    test_hamiltonian_scf_energy_pyscf("ch4p_631gds");
}

#[test]
fn test_hamiltonian_scf_energy_c6h6p_sto3g() {
    test_hamiltonian_scf_energy_pyscf("c6h6p_sto3g");
}

// #[test]
// fn test_hamiltonian_scf_energy_e_and_i_h_sto3g() {
//     let emap = ElementMap::new();
//     let atm_h = Atom::from_xyz("H 0.0 0.0 0.0", &emap, 1e-7).unwrap();
//     let mol_h = Molecule::from_atoms(&[atm_h.clone()], 1e-7);
//
//     let bs_1 = BasisShell::new(
//         1,
//         ShellOrder::Spinor(SpinorOrder::increasingm(1, true, None)),
//     );
//     let batm_h = BasisAtom::new(&atm_h, &[bs_1]);
//     let bao = BasisAngularOrder::new(&[batm_h]);
//
//     let bs_1_sp = BasisShell::new(
//         1,
//         ShellOrder::Spinor(SpinorOrder::increasingm(
//             1,
//             true,
//             Some(SpinorBalanceSymmetry::KineticBalance),
//         )),
//     );
//     let batm_h_sp = BasisAtom::new(&atm_h, &[bs_1_sp]);
//     let bao_sp = BasisAngularOrder::new(&[batm_h_sp]);
//
//     // ---------
//     // Integrals
//     // ---------
//     #[rustfmt::skip]
//     let sao = array![
//         [Complex::new(1.0000000000000e+00, 0.0), Complex::new(0.0000000000000e+00, 0.0), Complex::new(0.0000000000000e+00, 0.0), Complex::new(0.0000000000000e+00, 0.0)],
//         [Complex::new(0.0000000000000e+00, 0.0), Complex::new(1.0000000000000e+00, 0.0), Complex::new(0.0000000000000e+00, 0.0), Complex::new(0.0000000000000e+00, 0.0)],
//         [Complex::new(0.0000000000000e+00, 0.0), Complex::new(0.0000000000000e+00, 0.0), Complex::new(2.0236363463312e-05, 0.0), Complex::new(0.0000000000000e+00, 0.0)],
//         [Complex::new(0.0000000000000e+00, 0.0), Complex::new(0.0000000000000e+00, 0.0), Complex::new(0.0000000000000e+00, 0.0), Complex::new(2.0236363463312e-05, 0.0)],
//     ];
//
//     #[rustfmt::skip]
//     let onee = array![
//         [Complex::new(-1.2266137331239, 0.0), Complex::new( 0.0000000000000, 0.0), Complex::new( 0.7600318835666, 0.0), Complex::new( 0.0000000000000, 0.0)],
//         [Complex::new( 0.0000000000000, 0.0), Complex::new(-1.2266137331239, 0.0), Complex::new( 0.0000000000000, 0.0), Complex::new( 0.7600318835666, 0.0)],
//         [Complex::new( 0.7600318835666, 0.0), Complex::new( 0.0000000000000, 0.0), Complex::new(-0.7600545522661, 0.0), Complex::new( 0.0000000000000, 0.0)],
//         [Complex::new( 0.0000000000000, 0.0), Complex::new( 0.7600318835666, 0.0), Complex::new( 0.0000000000000, 0.0), Complex::new(-0.7600545522661, 0.0)],
//     ];
//
//     #[rustfmt::skip]
//     let twoe = array![
//         [
//             [
//                 [Complex::new(7.7460594391990e-01, 0.0), Complex::new(0.0000000000000e+00, 0.0), Complex::new(0.0000000000000e+00, 0.0), Complex::new(0.0000000000000e+00, 0.0)],
//                 [Complex::new(0.0000000000000e+00, 0.0), Complex::new(7.7460594391990e-01, 0.0), Complex::new(0.0000000000000e+00, 0.0), Complex::new(0.0000000000000e+00, 0.0)],
//                 [Complex::new(0.0000000000000e+00, 0.0), Complex::new(0.0000000000000e+00, 0.0), Complex::new(1.5530895551139e-05, 0.0), Complex::new(0.0000000000000e+00, 0.0)],
//                 [Complex::new(0.0000000000000e+00, 0.0), Complex::new(0.0000000000000e+00, 0.0), Complex::new(0.0000000000000e+00, 0.0), Complex::new(1.5530895551139e-05, 0.0)],
//             ],
//             [
//                 [Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)],
//                 [Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)],
//                 [Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)],
//                 [Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)],
//             ],
//             [
//                 [Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)],
//                 [Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)],
//                 [Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)],
//                 [Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)],
//             ],
//             [
//                 [Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)],
//                 [Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)],
//                 [Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)],
//                 [Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)],
//             ],
//         ],
//         [
//             [
//                 [Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)],
//                 [Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)],
//                 [Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)],
//                 [Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)],
//             ],
//             [
//                 [Complex::new(7.7460594391990e-01, 0.0), Complex::new(0.0000000000000e+00, 0.0), Complex::new(0.0000000000000e+00, 0.0), Complex::new(0.0000000000000e+00, 0.0)],
//                 [Complex::new(0.0000000000000e+00, 0.0), Complex::new(7.7460594391990e-01, 0.0), Complex::new(0.0000000000000e+00, 0.0), Complex::new(0.0000000000000e+00, 0.0)],
//                 [Complex::new(0.0000000000000e+00, 0.0), Complex::new(0.0000000000000e+00, 0.0), Complex::new(1.5530895551139e-05, 0.0), Complex::new(0.0000000000000e+00, 0.0)],
//                 [Complex::new(0.0000000000000e+00, 0.0), Complex::new(0.0000000000000e+00, 0.0), Complex::new(0.0000000000000e+00, 0.0), Complex::new(1.5530895551139e-05, 0.0)],
//             ],
//             [
//                 [Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)],
//                 [Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)],
//                 [Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)],
//                 [Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)],
//             ],
//             [
//                 [Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)],
//                 [Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)],
//                 [Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)],
//                 [Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)],
//             ],
//         ],
//         [
//             [
//                 [Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)],
//                 [Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)],
//                 [Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)],
//                 [Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)],
//             ],
//             [
//                 [Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)],
//                 [Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)],
//                 [Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)],
//                 [Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)],
//             ],
//             [
//                 [Complex::new(1.5530895551139e-05, 0.0), Complex::new(0.0000000000000e+00, 0.0), Complex::new(0.0000000000000e+00, 0.0), Complex::new(0.0000000000000e+00, 0.0)],
//                 [Complex::new(0.0000000000000e+00, 0.0), Complex::new(1.5530895551139e-05, 0.0), Complex::new(0.0000000000000e+00, 0.0), Complex::new(0.0000000000000e+00, 0.0)],
//                 [Complex::new(0.0000000000000e+00, 0.0), Complex::new(0.0000000000000e+00, 0.0), Complex::new(3.1219078939283e-10, 0.0), Complex::new(0.0000000000000e+00, 0.0)],
//                 [Complex::new(0.0000000000000e+00, 0.0), Complex::new(0.0000000000000e+00, 0.0), Complex::new(0.0000000000000e+00, 0.0), Complex::new(3.1219078939283e-10, 0.0)],
//             ],
//             [
//                 [Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)],
//                 [Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)],
//                 [Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)],
//                 [Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)],
//             ],
//         ],
//         [
//             [
//                 [Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)],
//                 [Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)],
//                 [Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)],
//                 [Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)],
//             ],
//             [
//                 [Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)],
//                 [Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)],
//                 [Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)],
//                 [Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)],
//             ],
//             [
//                 [Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)],
//                 [Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)],
//                 [Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)],
//                 [Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)],
//             ],
//             [
//                 [Complex::new(1.5530895551139e-05, 0.0), Complex::new(0.0000000000000e+00, 0.0), Complex::new(0.0000000000000e+00, 0.0), Complex::new(0.0000000000000e+00, 0.0)],
//                 [Complex::new(0.0000000000000e+00, 0.0), Complex::new(1.5530895551139e-05, 0.0), Complex::new(0.0000000000000e+00, 0.0), Complex::new(0.0000000000000e+00, 0.0)],
//                 [Complex::new(0.0000000000000e+00, 0.0), Complex::new(0.0000000000000e+00, 0.0), Complex::new(3.1219078939283e-10, 0.0), Complex::new(0.0000000000000e+00, 0.0)],
//                 [Complex::new(0.0000000000000e+00, 0.0), Complex::new(0.0000000000000e+00, 0.0), Complex::new(0.0000000000000e+00, 0.0), Complex::new(3.1219078939283e-10, 0.0)],
//             ],
//         ],
//     ];
//
//     let overlap_ao = OverlapAO::<Complex<f64>, SpinOrbitCoupled>::builder()
//         .sao(sao.view())
//         .build()
//         .unwrap();
//     let hamiltonian_ao = HamiltonianAO::<Complex<f64>, SpinOrbitCoupled>::builder()
//         .onee(onee.view())
//         .twoe(twoe.view())
//         .enuc(Complex::from(0.0))
//         .build()
//         .unwrap();
//
//     // ~~~~~~~~~~~~~~~~
//     // E(Σ) determinant
//     // ~~~~~~~~~~~~~~~~
//     #[rustfmt::skip]
//     // let c_e = array![
//     //     [Complex::new(-0.0044983607768177255, 0.0000000000000000000), Complex::new(0.0000000000000000000, 0.0000000000000000000), Complex::new(0.0000000000000000000, 0.0000000000000000000), Complex::new(0.9999898823239771000, 0.0000000000000000000)],
//     //     [Complex::new(0.0000000000000000000, 0.0000000000000000000), Complex::new(-0.0044983607768177255, 0.0000000000000000000), Complex::new(0.9999898823239771000, 0.0000000000000000000), Complex::new(0.0000000000000000000, 0.0000000000000000000)],
//     //     [Complex::new(222.2948341501447000000, 0.0000000000000000000), Complex::new(0.0000000000000000000, 0.0000000000000000000), Complex::new(0.0000000000000000000, 0.0000000000000000000), Complex::new(0.9999724802277992000, 0.0000000000000000000)],
//     //     [Complex::new(0.0000000000000000000, 0.0000000000000000000), Complex::new(222.2948341501447000000, 0.0000000000000000000), Complex::new(0.9999724802277992000, 0.0000000000000000000), Complex::new(0.0000000000000000000, 0.0000000000000000000)],
//     // ];
//     // let occ = array![0.0, 0.0, 1.0, 0.0];
//     let c_e = array![
//         [Complex::new(0.0000000000000000000, 0.0000000000000000000)],
//         [Complex::new(0.9999898823239771000, 0.0000000000000000000)],
//         [Complex::new(0.0000000000000000000, 0.0000000000000000000)],
//         [Complex::new(0.9999724802277992000, 0.0000000000000000000)],
//     ];
//     let occ = array![1.0];
//     let det_e = SlaterDeterminant::<Complex<f64>, SpinOrbitCoupled>::builder()
//         .coefficients(&[c_e])
//         .occupations(&[occ])
//         .baos(vec![&bao, &bao_sp])
//         .mol(&mol_h)
//         .structure_constraint(SpinOrbitCoupled::JAdapted(2))
//         .complex_symmetric(false)
//         .threshold(1e-14)
//         .build()
//         .unwrap();
//
//     let (zeroe, onee, twoe) = hamiltonian_ao
//         .calc_hamiltonian_matrix_element_contributions(
//             &det_e,
//             &det_e,
//             overlap_ao.sao(),
//             1e-10,
//             1e-7,
//         )
//         .unwrap();
//     println!("E: {zeroe} + {onee} + {twoe} = {}", zeroe + onee + twoe);
//
//     // ~~~~~~~~~~~~~~~~
//     // i(Σ) determinant
//     // ~~~~~~~~~~~~~~~~
//     #[rustfmt::skip]
//     // let c_i = array![
//     //     [Complex::new(-0.0044983607768177255, 0.0000000000000000000), Complex::new(0.0000000000000000000, 0.0000000000000000000), Complex::new(0.0000000000000000000, 0.0000000000000000000), Complex::new(0.9999898823239771000, 0.0000000000000000000)],
//     //     [Complex::new(0.0000000000000000000, 0.0000000000000000000), Complex::new(-0.0044983607768177255, 0.0000000000000000000), Complex::new(0.9999898823239771000, 0.0000000000000000000), Complex::new(0.0000000000000000000, 0.0000000000000000000)],
//     //     [Complex::new(-222.2948341501447000000, 0.0000000000000000000), Complex::new(0.0000000000000000000, 0.0000000000000000000), Complex::new(0.0000000000000000000, 0.0000000000000000000), Complex::new(-0.9999724802277992000, 0.0000000000000000000)],
//     //     [Complex::new(0.0000000000000000000, 0.0000000000000000000), Complex::new(-222.2948341501447000000, 0.0000000000000000000), Complex::new(-0.9999724802277992000, 0.0000000000000000000), Complex::new(0.0000000000000000000, 0.0000000000000000000)],
//     // ];
//     // let occ = array![0.0, 0.0, 1.0, 0.0];
//     let c_i = array![
//         [Complex::new(0.0000000000000000000, 0.0000000000000000000)],
//         [Complex::new(0.9999898823239771000, 0.0000000000000000000)],
//         [Complex::new(0.0000000000000000000, 0.0000000000000000000)],
//         [Complex::new(-0.9999724802277992000, 0.0000000000000000000)],
//     ];
//     let occ = array![1.0];
//     let det_i = SlaterDeterminant::<Complex<f64>, SpinOrbitCoupled>::builder()
//         .coefficients(&[c_i])
//         .occupations(&[occ])
//         .baos(vec![&bao, &bao_sp])
//         .mol(&mol_h)
//         .structure_constraint(SpinOrbitCoupled::JAdapted(2))
//         .complex_symmetric(false)
//         .threshold(1e-14)
//         .build()
//         .unwrap();
//
//     let (zeroe, onee, twoe) = hamiltonian_ao
//         .calc_hamiltonian_matrix_element_contributions(
//             &det_i,
//             &det_i,
//             overlap_ao.sao(),
//             1e-10,
//             1e-7,
//         )
//         .unwrap();
//     println!("E: {zeroe} + {onee} + {twoe} = {}", zeroe + onee + twoe);
//
//     let c_e_1 = array![
//         Complex::new(0.0000000000000000000, 0.0000000000000000000),
//         Complex::new(0.9999898823239771000, 0.0000000000000000000),
//         Complex::new(0.0000000000000000000, 0.0000000000000000000),
//         Complex::new(0.9999724802277992000, 0.0000000000000000000),
//     ];
//     let c_i_1 = array![
//         Complex::new(0.0000000000000000000, 0.0000000000000000000),
//         Complex::new(0.9999898823239771000, 0.0000000000000000000),
//         Complex::new(0.0000000000000000000, 0.0000000000000000000),
//         Complex::new(-0.9999724802277992000, 0.0000000000000000000),
//     ];
//     println!(
//         "e: {}",
//         einsum(
//             "i,ij,j->",
//             &[&c_e_1.view(), &hamiltonian_ao.onee.view(), &c_e_1.view()]
//         )
//         .unwrap()
//         .into_dimensionality::<Ix0>()
//         .unwrap()
//     );
//     println!(
//         "i: {}",
//         einsum(
//             "i,ij,j->",
//             &[&c_i_1.view(), &hamiltonian_ao.onee.view(), &c_i_1.view()]
//         )
//         .unwrap()
//         .into_dimensionality::<Ix0>()
//         .unwrap()
//     );
// }
