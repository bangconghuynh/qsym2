use approx::{assert_abs_diff_eq, assert_abs_diff_ne};
// use log4rs;
use itertools::Itertools;

use crate::angmom::spinor_rotation_3d::SpinConstraint;
use crate::auxiliary::atom::ElementMap;
use crate::chartab::chartab_symbols::DecomposedSymbol;
use crate::drivers::representation_analysis::angular_function::AngularFunctionRepAnalysisParams;
use crate::drivers::representation_analysis::multideterminant::{
    MultiDeterminantRepAnalysisDriver, MultiDeterminantRepAnalysisParams,
};
use crate::drivers::representation_analysis::CharacterTableDisplay;
use crate::drivers::symmetry_group_detection::{
    SymmetryGroupDetectionDriver, SymmetryGroupDetectionParams,
};
use crate::drivers::QSym2Driver;
use crate::group::UnitaryRepresentedGroup;
use crate::symmetry::symmetry_group::{SymmetryGroupProperties, UnitaryRepresentedSymmetryGroup};
use crate::symmetry::symmetry_symbols::MullikenIrrepSymbol;
use crate::symmetry::symmetry_transformation::SymmetryTransformationKind;

use crate::target::noci::backend::auxiliary::extract_pyscf_scf_data;
use crate::target::noci::backend::solver::noci::NOCISolvable;

const ROOT: &str = env!("CARGO_MANIFEST_DIR");

#[test]
fn test_solver_noci_energy_ch4p_sto3g() {
    // log4rs::init_file("log4rs.yml", Default::default()).unwrap();
    // Symmetry: |A|_(1) ⊕ |T|_(2)
    let emap = ElementMap::new();
    let filename = format!("{ROOT}/tests/noci_backend_hdf5/ch4p_sto3g.hdf5");
    let pyscf_data = extract_pyscf_scf_data::<_, f64>(filename).unwrap();
    let mol = pyscf_data.get_mol(&emap, 1e-7).unwrap();
    let bao = pyscf_data.get_bao(&mol).unwrap();
    let (overlap_ao, hamiltonian_ao) = pyscf_data.get_integrals::<SpinConstraint>().unwrap();
    let det = pyscf_data
        .get_slater_determinant(&mol, &bao, SpinConstraint::Unrestricted(2, true), 1e-14)
        .unwrap();

    // ------------------------
    // NOCI from symmetry orbit
    // ------------------------
    let params = SymmetryGroupDetectionParams::builder()
        .distance_thresholds(&[1e-7])
        .moi_thresholds(&[1e-7])
        .time_reversal(false)
        .write_symmetry_elements(true)
        .build()
        .unwrap();
    let mut pd_driver = SymmetryGroupDetectionDriver::builder()
        .parameters(&params)
        .molecule(Some(&mol))
        .build()
        .unwrap();
    pd_driver.run().unwrap();
    let pd_res = pd_driver.result().unwrap();
    let group =
        UnitaryRepresentedGroup::from_molecular_symmetry(&pd_res.unitary_symmetry, None).unwrap();

    let system = (&hamiltonian_ao, &overlap_ao);
    let multidets = system
        .solve_symmetry_noci(
            &[&det],
            &group,
            SymmetryTransformationKind::Spatial,
            true,
            1e-7,
            1e-7,
        )
        .unwrap();
    assert_eq!(multidets.len(), 4);

    // Degeneracies
    let energies = multidets
        .iter()
        .map(|multidet| multidet.energy())
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    assert_abs_diff_eq!(*energies[0], *energies[1], epsilon = 1e-7);
    assert_abs_diff_eq!(*energies[1], *energies[2], epsilon = 1e-7);

    assert_abs_diff_ne!(*energies[2], *energies[3], epsilon = 1e-7);

    // Symmetries
    let afa_params = AngularFunctionRepAnalysisParams::default();
    let mda_params = MultiDeterminantRepAnalysisParams::<f64>::builder()
        .integrality_threshold(1e-6)
        .linear_independence_threshold(1e-6)
        .use_magnetic_group(None)
        .use_double_group(false)
        .use_cayley_table(true)
        .symmetry_transformation_kind(SymmetryTransformationKind::Spatial)
        .write_character_table(Some(CharacterTableDisplay::Symbolic))
        .build()
        .unwrap();

    let sao = overlap_ao.sao().to_owned();
    let mut mda_driver = MultiDeterminantRepAnalysisDriver::<
        UnitaryRepresentedSymmetryGroup,
        f64,
        _,
        SpinConstraint,
    >::builder()
    .parameters(&mda_params)
    .angular_function_parameters(&afa_params)
    .multidets(multidets.iter().collect_vec())
    .sao(&sao)
    .symmetry_group(pd_res)
    .build()
    .unwrap();
    assert!(mda_driver.run().is_ok());
    assert_eq!(
        mda_driver.result().unwrap().multidet_symmetries()[0],
        Ok(DecomposedSymbol::<MullikenIrrepSymbol>::new("||T|_(2)|").unwrap())
    );
    assert_eq!(
        mda_driver.result().unwrap().multidet_symmetries()[1],
        Ok(DecomposedSymbol::<MullikenIrrepSymbol>::new("||T|_(2)|").unwrap())
    );
    assert_eq!(
        mda_driver.result().unwrap().multidet_symmetries()[2],
        Ok(DecomposedSymbol::<MullikenIrrepSymbol>::new("||T|_(2)|").unwrap())
    );
    assert_eq!(
        mda_driver.result().unwrap().multidet_symmetries()[3],
        Ok(DecomposedSymbol::<MullikenIrrepSymbol>::new("||A|_(1)|").unwrap())
    );
}

#[test]
fn test_solver_noci_energy_ch4p_631gdstar() {
    // Symmetry: |A|_(1) ⊕ |T|_(2)
    let emap = ElementMap::new();
    let filename = format!("{ROOT}/tests/noci_backend_hdf5/ch4p_631gds.hdf5");
    let pyscf_data = extract_pyscf_scf_data::<_, f64>(filename).unwrap();
    let mol = pyscf_data.get_mol(&emap, 1e-7).unwrap();
    let bao = pyscf_data.get_bao(&mol).unwrap();
    let (overlap_ao, hamiltonian_ao) = pyscf_data.get_integrals::<SpinConstraint>().unwrap();
    let det = pyscf_data
        .get_slater_determinant(&mol, &bao, SpinConstraint::Unrestricted(2, true), 1e-14)
        .unwrap();

    // ------------------------
    // NOCI from symmetry orbit
    // ------------------------
    let params = SymmetryGroupDetectionParams::builder()
        .distance_thresholds(&[1e-7])
        .moi_thresholds(&[1e-7])
        .time_reversal(false)
        .write_symmetry_elements(true)
        .build()
        .unwrap();
    let mut pd_driver = SymmetryGroupDetectionDriver::builder()
        .parameters(&params)
        .molecule(Some(&mol))
        .build()
        .unwrap();
    pd_driver.run().unwrap();
    let pd_res = pd_driver.result().unwrap();
    let group =
        UnitaryRepresentedGroup::from_molecular_symmetry(&pd_res.unitary_symmetry, None).unwrap();

    let system = (&hamiltonian_ao, &overlap_ao);
    let multidets = system
        .solve_symmetry_noci(
            &[&det],
            &group,
            SymmetryTransformationKind::Spatial,
            true,
            1e-7,
            1e-7,
        )
        .unwrap();
    assert_eq!(multidets.len(), 4);

    // Degeneracies
    let energies = multidets
        .iter()
        .map(|multidet| multidet.energy())
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    assert_abs_diff_eq!(*energies[0], *energies[1], epsilon = 1e-7);
    assert_abs_diff_eq!(*energies[1], *energies[2], epsilon = 1e-7);

    assert_abs_diff_ne!(*energies[2], *energies[3], epsilon = 1e-7);

    // Symmetries
    let afa_params = AngularFunctionRepAnalysisParams::default();
    let mda_params = MultiDeterminantRepAnalysisParams::<f64>::builder()
        .integrality_threshold(1e-6)
        .linear_independence_threshold(1e-6)
        .use_magnetic_group(None)
        .use_double_group(false)
        .use_cayley_table(true)
        .symmetry_transformation_kind(SymmetryTransformationKind::Spatial)
        .write_character_table(Some(CharacterTableDisplay::Symbolic))
        .build()
        .unwrap();

    let sao = overlap_ao.sao().to_owned();
    let mut mda_driver = MultiDeterminantRepAnalysisDriver::<
        UnitaryRepresentedSymmetryGroup,
        f64,
        _,
        SpinConstraint,
    >::builder()
    .parameters(&mda_params)
    .angular_function_parameters(&afa_params)
    .multidets(multidets.iter().collect_vec())
    .sao(&sao)
    .symmetry_group(pd_res)
    .build()
    .unwrap();
    assert!(mda_driver.run().is_ok());
    assert_eq!(
        mda_driver.result().unwrap().multidet_symmetries()[0],
        Ok(DecomposedSymbol::<MullikenIrrepSymbol>::new("||T|_(2)|").unwrap())
    );
    assert_eq!(
        mda_driver.result().unwrap().multidet_symmetries()[1],
        Ok(DecomposedSymbol::<MullikenIrrepSymbol>::new("||T|_(2)|").unwrap())
    );
    assert_eq!(
        mda_driver.result().unwrap().multidet_symmetries()[2],
        Ok(DecomposedSymbol::<MullikenIrrepSymbol>::new("||T|_(2)|").unwrap())
    );
    assert_eq!(
        mda_driver.result().unwrap().multidet_symmetries()[3],
        Ok(DecomposedSymbol::<MullikenIrrepSymbol>::new("||A|_(1)|").unwrap())
    );
}

#[test]
fn test_solver_noci_energy_h6_sto3g() {
    // Symmetry: |A|_(2g) ⊕ |E|_(g) ⊕ |T|_(1g)
    let emap = ElementMap::new();
    let filename = format!("{ROOT}/tests/noci_backend_hdf5/h6_sto3g.hdf5");
    let pyscf_data = extract_pyscf_scf_data::<_, f64>(filename).unwrap();
    let mol = pyscf_data.get_mol(&emap, 1e-7).unwrap();
    let bao = pyscf_data.get_bao(&mol).unwrap();
    let (overlap_ao, hamiltonian_ao) = pyscf_data.get_integrals::<SpinConstraint>().unwrap();
    let det = pyscf_data
        .get_slater_determinant(&mol, &bao, SpinConstraint::Unrestricted(2, true), 1e-14)
        .unwrap();

    // ------------------------
    // NOCI from symmetry orbit
    // ------------------------
    let params = SymmetryGroupDetectionParams::builder()
        .distance_thresholds(&[1e-7])
        .moi_thresholds(&[1e-7])
        .time_reversal(false)
        .write_symmetry_elements(true)
        .build()
        .unwrap();
    let mut pd_driver = SymmetryGroupDetectionDriver::builder()
        .parameters(&params)
        .molecule(Some(&mol))
        .build()
        .unwrap();
    pd_driver.run().unwrap();
    let pd_res = pd_driver.result().unwrap();
    let group =
        UnitaryRepresentedGroup::from_molecular_symmetry(&pd_res.unitary_symmetry, None).unwrap();

    let system = (&hamiltonian_ao, &overlap_ao);
    let multidets = system
        .solve_symmetry_noci(
            &[&det],
            &group,
            SymmetryTransformationKind::Spatial,
            true,
            1e-7,
            1e-7,
        )
        .unwrap();
    assert_eq!(multidets.len(), 6);

    // Degeneracies
    let energies = multidets
        .iter()
        .map(|multidet| multidet.energy())
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    assert_abs_diff_eq!(*energies[0], *energies[1], epsilon = 1e-7);
    assert_abs_diff_eq!(*energies[1], *energies[2], epsilon = 1e-7);

    assert_abs_diff_ne!(*energies[2], *energies[3], epsilon = 1e-7);
    assert_abs_diff_eq!(*energies[3], *energies[4], epsilon = 1e-7);

    assert_abs_diff_ne!(*energies[4], *energies[5], epsilon = 1e-7);

    // Symmetries
    let afa_params = AngularFunctionRepAnalysisParams::default();
    let mda_params = MultiDeterminantRepAnalysisParams::<f64>::builder()
        .integrality_threshold(1e-6)
        .linear_independence_threshold(1e-6)
        .use_magnetic_group(None)
        .use_double_group(false)
        .use_cayley_table(true)
        .symmetry_transformation_kind(SymmetryTransformationKind::Spatial)
        .write_character_table(Some(CharacterTableDisplay::Symbolic))
        .build()
        .unwrap();

    let sao = overlap_ao.sao().to_owned();
    let mut mda_driver = MultiDeterminantRepAnalysisDriver::<
        UnitaryRepresentedSymmetryGroup,
        f64,
        _,
        SpinConstraint,
    >::builder()
    .parameters(&mda_params)
    .angular_function_parameters(&afa_params)
    .multidets(multidets.iter().collect_vec())
    .sao(&sao)
    .symmetry_group(pd_res)
    .build()
    .unwrap();
    assert!(mda_driver.run().is_ok());
    assert_eq!(
        mda_driver.result().unwrap().multidet_symmetries()[0],
        Ok(DecomposedSymbol::<MullikenIrrepSymbol>::new("||T|_(1g)|").unwrap())
    );
    assert_eq!(
        mda_driver.result().unwrap().multidet_symmetries()[1],
        Ok(DecomposedSymbol::<MullikenIrrepSymbol>::new("||T|_(1g)|").unwrap())
    );
    assert_eq!(
        mda_driver.result().unwrap().multidet_symmetries()[2],
        Ok(DecomposedSymbol::<MullikenIrrepSymbol>::new("||T|_(1g)|").unwrap())
    );
    assert_eq!(
        mda_driver.result().unwrap().multidet_symmetries()[3],
        Ok(DecomposedSymbol::<MullikenIrrepSymbol>::new("||E|_(g)|").unwrap())
    );
    assert_eq!(
        mda_driver.result().unwrap().multidet_symmetries()[4],
        Ok(DecomposedSymbol::<MullikenIrrepSymbol>::new("||E|_(g)|").unwrap())
    );
    assert_eq!(
        mda_driver.result().unwrap().multidet_symmetries()[5],
        Ok(DecomposedSymbol::<MullikenIrrepSymbol>::new("||A|_(2g)|").unwrap())
    );
}

#[test]
fn test_solver_noci_energy_h6_631gdstar() {
    // Symmetry: |T|_(1g) ⊕ |T|_(2g)
    let emap = ElementMap::new();
    let filename = format!("{ROOT}/tests/noci_backend_hdf5/h6_631gds.hdf5");
    let pyscf_data = extract_pyscf_scf_data::<_, f64>(filename).unwrap();
    let mol = pyscf_data.get_mol(&emap, 1e-7).unwrap();
    let bao = pyscf_data.get_bao(&mol).unwrap();
    let (overlap_ao, hamiltonian_ao) = pyscf_data.get_integrals::<SpinConstraint>().unwrap();
    let det = pyscf_data
        .get_slater_determinant(&mol, &bao, SpinConstraint::Unrestricted(2, true), 1e-14)
        .unwrap();

    // ------------------------
    // NOCI from symmetry orbit
    // ------------------------
    let params = SymmetryGroupDetectionParams::builder()
        .distance_thresholds(&[1e-7])
        .moi_thresholds(&[1e-7])
        .time_reversal(false)
        .write_symmetry_elements(true)
        .build()
        .unwrap();
    let mut pd_driver = SymmetryGroupDetectionDriver::builder()
        .parameters(&params)
        .molecule(Some(&mol))
        .build()
        .unwrap();
    pd_driver.run().unwrap();
    let pd_res = pd_driver.result().unwrap();
    let group =
        UnitaryRepresentedGroup::from_molecular_symmetry(&pd_res.unitary_symmetry, None).unwrap();

    let system = (&hamiltonian_ao, &overlap_ao);
    let multidets = system
        .solve_symmetry_noci(
            &[&det],
            &group,
            SymmetryTransformationKind::Spatial,
            true,
            1e-7,
            1e-7,
        )
        .unwrap();
    assert_eq!(multidets.len(), 6);

    // Degeneracies
    let energies = multidets
        .iter()
        .map(|multidet| multidet.energy())
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    assert_abs_diff_eq!(*energies[0], *energies[1], epsilon = 1e-7);
    assert_abs_diff_eq!(*energies[1], *energies[2], epsilon = 1e-7);

    assert_abs_diff_ne!(*energies[2], *energies[3], epsilon = 1e-7);
    assert_abs_diff_eq!(*energies[3], *energies[4], epsilon = 1e-7);
    assert_abs_diff_eq!(*energies[4], *energies[5], epsilon = 1e-7);

    // Symmetries
    let afa_params = AngularFunctionRepAnalysisParams::default();
    let mda_params = MultiDeterminantRepAnalysisParams::<f64>::builder()
        .integrality_threshold(1e-6)
        .linear_independence_threshold(1e-6)
        .use_magnetic_group(None)
        .use_double_group(false)
        .use_cayley_table(true)
        .symmetry_transformation_kind(SymmetryTransformationKind::Spatial)
        .write_character_table(Some(CharacterTableDisplay::Symbolic))
        .build()
        .unwrap();

    let sao = overlap_ao.sao().to_owned();
    let mut mda_driver = MultiDeterminantRepAnalysisDriver::<
        UnitaryRepresentedSymmetryGroup,
        f64,
        _,
        SpinConstraint,
    >::builder()
    .parameters(&mda_params)
    .angular_function_parameters(&afa_params)
    .multidets(multidets.iter().collect_vec())
    .sao(&sao)
    .symmetry_group(pd_res)
    .build()
    .unwrap();
    assert!(mda_driver.run().is_ok());
    assert_eq!(
        mda_driver.result().unwrap().multidet_symmetries()[0],
        Ok(DecomposedSymbol::<MullikenIrrepSymbol>::new("||T|_(1g)|").unwrap())
    );
    assert_eq!(
        mda_driver.result().unwrap().multidet_symmetries()[1],
        Ok(DecomposedSymbol::<MullikenIrrepSymbol>::new("||T|_(1g)|").unwrap())
    );
    assert_eq!(
        mda_driver.result().unwrap().multidet_symmetries()[2],
        Ok(DecomposedSymbol::<MullikenIrrepSymbol>::new("||T|_(1g)|").unwrap())
    );
    assert_eq!(
        mda_driver.result().unwrap().multidet_symmetries()[3],
        Ok(DecomposedSymbol::<MullikenIrrepSymbol>::new("||T|_(2g)|").unwrap())
    );
    assert_eq!(
        mda_driver.result().unwrap().multidet_symmetries()[4],
        Ok(DecomposedSymbol::<MullikenIrrepSymbol>::new("||T|_(2g)|").unwrap())
    );
    assert_eq!(
        mda_driver.result().unwrap().multidet_symmetries()[5],
        Ok(DecomposedSymbol::<MullikenIrrepSymbol>::new("||T|_(2g)|").unwrap())
    );
}

#[test]
fn test_solver_noci_energy_h4_sto3g() {
    // Symmetry: |A|_(1g) ⊕ |B|_(1g)
    let emap = ElementMap::new();
    let filename = format!("{ROOT}/tests/noci_backend_hdf5/h4_sto3g.hdf5");
    let pyscf_data = extract_pyscf_scf_data::<_, f64>(filename).unwrap();
    let mol = pyscf_data.get_mol(&emap, 1e-7).unwrap();
    let bao = pyscf_data.get_bao(&mol).unwrap();
    let (overlap_ao, hamiltonian_ao) = pyscf_data.get_integrals::<SpinConstraint>().unwrap();
    let det = pyscf_data
        .get_slater_determinant(&mol, &bao, SpinConstraint::Unrestricted(2, true), 1e-14)
        .unwrap();

    // ------------------------
    // NOCI from symmetry orbit
    // ------------------------
    let params = SymmetryGroupDetectionParams::builder()
        .distance_thresholds(&[1e-7])
        .moi_thresholds(&[1e-7])
        .time_reversal(false)
        .write_symmetry_elements(true)
        .build()
        .unwrap();
    let mut pd_driver = SymmetryGroupDetectionDriver::builder()
        .parameters(&params)
        .molecule(Some(&mol))
        .build()
        .unwrap();
    pd_driver.run().unwrap();
    let pd_res = pd_driver.result().unwrap();
    let group =
        UnitaryRepresentedGroup::from_molecular_symmetry(&pd_res.unitary_symmetry, None).unwrap();

    let system = (&hamiltonian_ao, &overlap_ao);
    let multidets = system
        .solve_symmetry_noci(
            &[&det],
            &group,
            SymmetryTransformationKind::Spatial,
            true,
            1e-7,
            1e-7,
        )
        .unwrap();
    assert_eq!(multidets.len(), 2);

    // Degeneracies
    let energies = multidets
        .iter()
        .map(|multidet| multidet.energy())
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    assert_abs_diff_ne!(*energies[0], *energies[1], epsilon = 1e-7);

    // Symmetries
    let afa_params = AngularFunctionRepAnalysisParams::default();
    let mda_params = MultiDeterminantRepAnalysisParams::<f64>::builder()
        .integrality_threshold(1e-6)
        .linear_independence_threshold(1e-6)
        .use_magnetic_group(None)
        .use_double_group(false)
        .use_cayley_table(true)
        .symmetry_transformation_kind(SymmetryTransformationKind::Spatial)
        .write_character_table(Some(CharacterTableDisplay::Symbolic))
        .build()
        .unwrap();

    let sao = overlap_ao.sao().to_owned();
    let mut mda_driver = MultiDeterminantRepAnalysisDriver::<
        UnitaryRepresentedSymmetryGroup,
        f64,
        _,
        SpinConstraint,
    >::builder()
    .parameters(&mda_params)
    .angular_function_parameters(&afa_params)
    .multidets(multidets.iter().collect_vec())
    .sao(&sao)
    .symmetry_group(pd_res)
    .build()
    .unwrap();
    assert!(mda_driver.run().is_ok());
    assert_eq!(
        mda_driver.result().unwrap().multidet_symmetries()[0],
        Ok(DecomposedSymbol::<MullikenIrrepSymbol>::new("||B|_(1g)|").unwrap())
    );
    assert_eq!(
        mda_driver.result().unwrap().multidet_symmetries()[1],
        Ok(DecomposedSymbol::<MullikenIrrepSymbol>::new("||A|_(1g)|").unwrap())
    );
}

#[test]
fn test_solver_noci_energy_h4_631gdstar() {
    // Symmetry: |A|_(2g) ⊕ |B|_(1g)
    let emap = ElementMap::new();
    let filename = format!("{ROOT}/tests/noci_backend_hdf5/h4_631gds.hdf5");
    let pyscf_data = extract_pyscf_scf_data::<_, f64>(filename).unwrap();
    let mol = pyscf_data.get_mol(&emap, 1e-7).unwrap();
    let bao = pyscf_data.get_bao(&mol).unwrap();
    let (overlap_ao, hamiltonian_ao) = pyscf_data.get_integrals::<SpinConstraint>().unwrap();
    let det = pyscf_data
        .get_slater_determinant(&mol, &bao, SpinConstraint::Unrestricted(2, true), 1e-14)
        .unwrap();

    // ------------------------
    // NOCI from symmetry orbit
    // ------------------------
    let params = SymmetryGroupDetectionParams::builder()
        .distance_thresholds(&[1e-7])
        .moi_thresholds(&[1e-7])
        .time_reversal(false)
        .write_symmetry_elements(true)
        .build()
        .unwrap();
    let mut pd_driver = SymmetryGroupDetectionDriver::builder()
        .parameters(&params)
        .molecule(Some(&mol))
        .build()
        .unwrap();
    pd_driver.run().unwrap();
    let pd_res = pd_driver.result().unwrap();
    let group =
        UnitaryRepresentedGroup::from_molecular_symmetry(&pd_res.unitary_symmetry, None).unwrap();

    let system = (&hamiltonian_ao, &overlap_ao);
    let multidets = system
        .solve_symmetry_noci(
            &[&det],
            &group,
            SymmetryTransformationKind::Spatial,
            true,
            1e-7,
            1e-7,
        )
        .unwrap();
    assert_eq!(multidets.len(), 2);

    // Degeneracies
    let energies = multidets
        .iter()
        .map(|multidet| multidet.energy())
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    assert_abs_diff_ne!(*energies[0], *energies[1], epsilon = 1e-7);

    // Symmetries
    let afa_params = AngularFunctionRepAnalysisParams::default();
    let mda_params = MultiDeterminantRepAnalysisParams::<f64>::builder()
        .integrality_threshold(1e-6)
        .linear_independence_threshold(1e-6)
        .use_magnetic_group(None)
        .use_double_group(false)
        .use_cayley_table(true)
        .symmetry_transformation_kind(SymmetryTransformationKind::Spatial)
        .write_character_table(Some(CharacterTableDisplay::Symbolic))
        .build()
        .unwrap();

    let sao = overlap_ao.sao().to_owned();
    let mut mda_driver = MultiDeterminantRepAnalysisDriver::<
        UnitaryRepresentedSymmetryGroup,
        f64,
        _,
        SpinConstraint,
    >::builder()
    .parameters(&mda_params)
    .angular_function_parameters(&afa_params)
    .multidets(multidets.iter().collect_vec())
    .sao(&sao)
    .symmetry_group(pd_res)
    .build()
    .unwrap();
    assert!(mda_driver.run().is_ok());
    assert_eq!(
        mda_driver.result().unwrap().multidet_symmetries()[0],
        Ok(DecomposedSymbol::<MullikenIrrepSymbol>::new("||A|_(2g)|").unwrap())
    );
    assert_eq!(
        mda_driver.result().unwrap().multidet_symmetries()[1],
        Ok(DecomposedSymbol::<MullikenIrrepSymbol>::new("||B|_(1g)|").unwrap())
    );
}

#[test]
fn test_solver_noci_energy_c6h6p_sto3g() {
    // Symmetry: |B|_(2g) ⊕ |E|_(1g)
    let emap = ElementMap::new();
    let filename = format!("{ROOT}/tests/noci_backend_hdf5/c6h6p_sto3g.hdf5");
    let pyscf_data = extract_pyscf_scf_data::<_, f64>(filename).unwrap();
    let mol = pyscf_data.get_mol(&emap, 1e-6).unwrap();
    let bao = pyscf_data.get_bao(&mol).unwrap();
    let (overlap_ao, hamiltonian_ao) = pyscf_data.get_integrals::<SpinConstraint>().unwrap();
    let det = pyscf_data
        .get_slater_determinant(&mol, &bao, SpinConstraint::Unrestricted(2, true), 1e-14)
        .unwrap();

    // ------------------------
    // NOCI from symmetry orbit
    // ------------------------
    let params = SymmetryGroupDetectionParams::builder()
        .distance_thresholds(&[1e-6])
        .moi_thresholds(&[1e-6])
        .time_reversal(false)
        .write_symmetry_elements(true)
        .build()
        .unwrap();
    let mut pd_driver = SymmetryGroupDetectionDriver::builder()
        .parameters(&params)
        .molecule(Some(&mol))
        .build()
        .unwrap();
    pd_driver.run().unwrap();
    let pd_res = pd_driver.result().unwrap();
    let group =
        UnitaryRepresentedGroup::from_molecular_symmetry(&pd_res.unitary_symmetry, None).unwrap();

    let system = (&hamiltonian_ao, &overlap_ao);
    let multidets = system
        .solve_symmetry_noci(
            &[&det],
            &group,
            SymmetryTransformationKind::Spatial,
            true,
            1e-7,
            1e-7,
        )
        .unwrap();
    assert_eq!(multidets.len(), 3);
    let energies = multidets
        .iter()
        .map(|multidet| multidet.energy())
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    assert_abs_diff_eq!(*energies[0], *energies[1], epsilon = 1e-7);

    assert_abs_diff_ne!(*energies[1], *energies[2], epsilon = 1e-7);

    // Symmetries
    let afa_params = AngularFunctionRepAnalysisParams::default();
    let mda_params = MultiDeterminantRepAnalysisParams::<f64>::builder()
        .integrality_threshold(1e-6)
        .linear_independence_threshold(1e-6)
        .use_magnetic_group(None)
        .use_double_group(false)
        .use_cayley_table(true)
        .symmetry_transformation_kind(SymmetryTransformationKind::Spatial)
        .write_character_table(Some(CharacterTableDisplay::Symbolic))
        .build()
        .unwrap();

    let sao = overlap_ao.sao().to_owned();
    let mut mda_driver = MultiDeterminantRepAnalysisDriver::<
        UnitaryRepresentedSymmetryGroup,
        f64,
        _,
        SpinConstraint,
    >::builder()
    .parameters(&mda_params)
    .angular_function_parameters(&afa_params)
    .multidets(multidets.iter().collect_vec())
    .sao(&sao)
    .symmetry_group(pd_res)
    .build()
    .unwrap();
    assert!(mda_driver.run().is_ok());
    assert_eq!(
        mda_driver.result().unwrap().multidet_symmetries()[0],
        Ok(DecomposedSymbol::<MullikenIrrepSymbol>::new("||E|_(1g)|").unwrap())
    );
    assert_eq!(
        mda_driver.result().unwrap().multidet_symmetries()[1],
        Ok(DecomposedSymbol::<MullikenIrrepSymbol>::new("||E|_(1g)|").unwrap())
    );
    assert_eq!(
        mda_driver.result().unwrap().multidet_symmetries()[2],
        Ok(DecomposedSymbol::<MullikenIrrepSymbol>::new("||B|_(2g)|").unwrap())
    );
}
