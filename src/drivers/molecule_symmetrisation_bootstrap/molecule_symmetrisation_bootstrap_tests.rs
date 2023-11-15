// use log4rs;
use nalgebra::Vector3;
use serial_test::serial;

use crate::auxiliary::molecule::Molecule;
use crate::drivers::molecule_symmetrisation_bootstrap::{
    MoleculeSymmetrisationBootstrapDriver, MoleculeSymmetrisationBootstrapParams,
};
use crate::drivers::symmetry_group_detection::{
    SymmetryGroupDetectionDriver, SymmetryGroupDetectionParams,
};
use crate::drivers::QSym2Driver;
use crate::group::{GroupProperties, UnitaryRepresentedGroup};
use crate::symmetry::symmetry_group::SymmetryGroupProperties;

const ROOT: &str = env!("CARGO_MANIFEST_DIR");

#[test]
fn test_drivers_molecule_symmetrisation_bootstrap_benzene() {
    // log4rs::init_file("log4rs.yml", Default::default()).unwrap();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/benzene_imperfect.xyz");
    let mol = Molecule::from_xyz(&path, 1e-7);

    let msb_params = MoleculeSymmetrisationBootstrapParams::builder()
        .reorientate_molecule(true)
        .use_magnetic_group(false)
        .loose_moi_threshold(1e-1)
        .loose_distance_threshold(4e-1)
        .target_moi_threshold(1e-8)
        .target_distance_threshold(1e-8)
        .infinite_order_to_finite(Some(8))
        .max_iterations(50)
        .verbose(2)
        .build()
        .unwrap();
    let mut msb_driver = MoleculeSymmetrisationBootstrapDriver::builder()
        .parameters(&msb_params)
        .molecule(&mol)
        .build()
        .unwrap();
    assert!(msb_driver.run().is_ok());

    let verifying_pd_params = SymmetryGroupDetectionParams::builder()
        .moi_thresholds(&[msb_params.target_moi_threshold])
        .distance_thresholds(&[msb_params.target_distance_threshold])
        .time_reversal(msb_params.use_magnetic_group)
        .write_symmetry_elements(true)
        .build()
        .unwrap();
    let mut verifying_pd_driver = SymmetryGroupDetectionDriver::builder()
        .parameters(&verifying_pd_params)
        .molecule(
            msb_driver
                .result()
                .ok()
                .map(|res| &res.symmetrised_molecule),
        )
        .build()
        .unwrap();
    assert!(verifying_pd_driver.run().is_ok());
    let verifying_pd_res = verifying_pd_driver.result().unwrap();
    assert_eq!(
        verifying_pd_res
            .unitary_symmetry
            .group_name
            .as_ref()
            .unwrap(),
        "D6h"
    );
}

#[test]
fn test_drivers_molecule_symmetrisation_bootstrap_nh3() {
    // log4rs::init_file("log4rs.yml", Default::default()).unwrap();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/nh3_imperfect.xyz");
    let mol = Molecule::from_xyz(&path, 1e-7);

    let msb_params = MoleculeSymmetrisationBootstrapParams::builder()
        .reorientate_molecule(true)
        .use_magnetic_group(false)
        .loose_moi_threshold(1e-1)
        .loose_distance_threshold(1e-1)
        .target_moi_threshold(1e-8)
        .target_distance_threshold(1e-8)
        .infinite_order_to_finite(Some(8))
        .max_iterations(50)
        .verbose(2)
        .build()
        .unwrap();
    let mut msb_driver = MoleculeSymmetrisationBootstrapDriver::builder()
        .parameters(&msb_params)
        .molecule(&mol)
        .build()
        .unwrap();
    assert!(msb_driver.run().is_ok());

    let verifying_pd_params = SymmetryGroupDetectionParams::builder()
        .moi_thresholds(&[msb_params.target_moi_threshold])
        .distance_thresholds(&[msb_params.target_distance_threshold])
        .time_reversal(msb_params.use_magnetic_group)
        .write_symmetry_elements(true)
        .build()
        .unwrap();
    let mut verifying_pd_driver = SymmetryGroupDetectionDriver::builder()
        .parameters(&verifying_pd_params)
        .molecule(
            msb_driver
                .result()
                .ok()
                .map(|res| &res.symmetrised_molecule),
        )
        .build()
        .unwrap();
    assert!(verifying_pd_driver.run().is_ok());
    let verifying_pd_res = verifying_pd_driver.result().unwrap();
    assert_eq!(
        verifying_pd_res
            .unitary_symmetry
            .group_name
            .as_ref()
            .unwrap(),
        "C3v"
    );
}

#[test]
fn test_drivers_molecule_symmetrisation_bootstrap_8cpp() {
    // log4rs::init_file("log4rs.yml", Default::default()).unwrap();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/8cpp_imperfect.xyz");
    let mol = Molecule::from_xyz(&path, 1e-7);

    let msb_params = MoleculeSymmetrisationBootstrapParams::builder()
        .reorientate_molecule(true)
        .use_magnetic_group(false)
        .loose_moi_threshold(1e-1)
        .loose_distance_threshold(7e-1)
        .target_moi_threshold(1e-8)
        .target_distance_threshold(1e-8)
        .infinite_order_to_finite(Some(8))
        .max_iterations(50)
        .verbose(2)
        .build()
        .unwrap();
    let mut msb_driver = MoleculeSymmetrisationBootstrapDriver::builder()
        .parameters(&msb_params)
        .molecule(&mol)
        .build()
        .unwrap();
    assert!(msb_driver.run().is_ok());

    let verifying_pd_params = SymmetryGroupDetectionParams::builder()
        .moi_thresholds(&[msb_params.target_moi_threshold])
        .distance_thresholds(&[msb_params.target_distance_threshold])
        .time_reversal(msb_params.use_magnetic_group)
        .write_symmetry_elements(true)
        .build()
        .unwrap();
    let mut verifying_pd_driver = SymmetryGroupDetectionDriver::builder()
        .parameters(&verifying_pd_params)
        .molecule(
            msb_driver
                .result()
                .ok()
                .map(|res| &res.symmetrised_molecule),
        )
        .build()
        .unwrap();
    assert!(verifying_pd_driver.run().is_ok());
    let verifying_pd_res = verifying_pd_driver.result().unwrap();
    assert_eq!(
        verifying_pd_res
            .unitary_symmetry
            .group_name
            .as_ref()
            .unwrap(),
        "D4d"
    );
}

#[test]
fn test_drivers_molecule_symmetrisation_bootstrap_vf6_magnetic_field() {
    // log4rs::init_file("log4rs.yml", Default::default()).unwrap();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/vf6_imperfect.xyz");
    let mut mol = Molecule::from_xyz(&path, 1e-7);
    mol.set_magnetic_field(Some(Vector3::new(1.0, 1.0, 1.0)));

    let msb_params = MoleculeSymmetrisationBootstrapParams::builder()
        .reorientate_molecule(true)
        .use_magnetic_group(true)
        .loose_moi_threshold(1e-1)
        .loose_distance_threshold(1e-1)
        .target_moi_threshold(1e-8)
        .target_distance_threshold(1e-8)
        .infinite_order_to_finite(Some(8))
        .max_iterations(50)
        .verbose(2)
        .build()
        .unwrap();
    let mut msb_driver = MoleculeSymmetrisationBootstrapDriver::builder()
        .parameters(&msb_params)
        .molecule(&mol)
        .build()
        .unwrap();
    assert!(msb_driver.run().is_ok());

    let verifying_pd_params = SymmetryGroupDetectionParams::builder()
        .moi_thresholds(&[msb_params.target_moi_threshold])
        .distance_thresholds(&[msb_params.target_distance_threshold])
        .time_reversal(msb_params.use_magnetic_group)
        .write_symmetry_elements(true)
        .build()
        .unwrap();
    let mut verifying_pd_driver = SymmetryGroupDetectionDriver::builder()
        .parameters(&verifying_pd_params)
        .molecule(
            msb_driver
                .result()
                .ok()
                .map(|res| &res.symmetrised_molecule),
        )
        .build()
        .unwrap();
    assert!(verifying_pd_driver.run().is_ok());
    let verifying_pd_res = verifying_pd_driver.result().unwrap();
    assert_eq!(
        verifying_pd_res
            .unitary_symmetry
            .group_name
            .as_ref()
            .unwrap(),
        "S6"
    );
    assert_eq!(
        verifying_pd_res
            .magnetic_symmetry
            .as_ref()
            .unwrap()
            .group_name
            .as_ref()
            .unwrap(),
        "D3d"
    );
}

#[test]
fn test_drivers_molecule_symmetrisation_bootstrap_h4_magnetic_field() {
    // log4rs::init_file("log4rs.yml", Default::default()).unwrap();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/h4_imperfect.xyz");
    let mut mol = Molecule::from_xyz(&path, 1e-7);
    mol.set_magnetic_field(Some(Vector3::z()));

    let msb_params = MoleculeSymmetrisationBootstrapParams::builder()
        .reorientate_molecule(true)
        .use_magnetic_group(true)
        .loose_moi_threshold(1e-2)
        .loose_distance_threshold(1e-2)
        .target_moi_threshold(1e-8)
        .target_distance_threshold(1e-8)
        .infinite_order_to_finite(Some(8))
        .max_iterations(50)
        .verbose(2)
        .build()
        .unwrap();
    let mut msb_driver = MoleculeSymmetrisationBootstrapDriver::builder()
        .parameters(&msb_params)
        .molecule(&mol)
        .build()
        .unwrap();
    assert!(msb_driver.run().is_ok());

    let verifying_pd_params = SymmetryGroupDetectionParams::builder()
        .moi_thresholds(&[msb_params.target_moi_threshold])
        .distance_thresholds(&[msb_params.target_distance_threshold])
        .time_reversal(msb_params.use_magnetic_group)
        .write_symmetry_elements(true)
        .build()
        .unwrap();
    let mut verifying_pd_driver = SymmetryGroupDetectionDriver::builder()
        .parameters(&verifying_pd_params)
        .molecule(
            msb_driver
                .result()
                .ok()
                .map(|res| &res.symmetrised_molecule),
        )
        .build()
        .unwrap();
    assert!(verifying_pd_driver.run().is_ok());
    let verifying_pd_res = verifying_pd_driver.result().unwrap();
    assert_eq!(
        verifying_pd_res
            .unitary_symmetry
            .group_name
            .as_ref()
            .unwrap(),
        "C4h"
    );
    assert_eq!(
        verifying_pd_res
            .magnetic_symmetry
            .as_ref()
            .unwrap()
            .group_name
            .as_ref()
            .unwrap(),
        "D4h"
    );
}

#[test]
fn test_drivers_molecule_symmetrisation_bootstrap_vf6_electric_field() {
    // log4rs::init_file("log4rs.yml", Default::default()).unwrap();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/vf6_imperfect.xyz");
    let mut mol = Molecule::from_xyz(&path, 1e-7);
    mol.set_electric_field(Some(Vector3::y()));

    let msb_params = MoleculeSymmetrisationBootstrapParams::builder()
        .reorientate_molecule(true)
        .use_magnetic_group(true)
        .loose_moi_threshold(1e-2)
        .loose_distance_threshold(1e-2)
        .target_moi_threshold(1e-8)
        .target_distance_threshold(1e-8)
        .infinite_order_to_finite(Some(8))
        .max_iterations(50)
        .verbose(2)
        .build()
        .unwrap();
    let mut msb_driver = MoleculeSymmetrisationBootstrapDriver::builder()
        .parameters(&msb_params)
        .molecule(&mol)
        .build()
        .unwrap();
    assert!(msb_driver.run().is_ok());

    let verifying_pd_params = SymmetryGroupDetectionParams::builder()
        .moi_thresholds(&[msb_params.target_moi_threshold])
        .distance_thresholds(&[msb_params.target_distance_threshold])
        .time_reversal(msb_params.use_magnetic_group)
        .write_symmetry_elements(true)
        .build()
        .unwrap();
    let mut verifying_pd_driver = SymmetryGroupDetectionDriver::builder()
        .parameters(&verifying_pd_params)
        .molecule(
            msb_driver
                .result()
                .ok()
                .map(|res| &res.symmetrised_molecule),
        )
        .build()
        .unwrap();
    assert!(verifying_pd_driver.run().is_ok());
    let verifying_pd_res = verifying_pd_driver.result().unwrap();
    assert_eq!(
        verifying_pd_res
            .unitary_symmetry
            .group_name
            .as_ref()
            .unwrap(),
        "C4v"
    );
}

#[test]
fn test_drivers_molecule_symmetrisation_bootstrap_c2h2() {
    // log4rs::init_file("log4rs.yml", Default::default()).unwrap();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c2h2_imperfect.xyz");
    let mol = Molecule::from_xyz(&path, 1e-7);

    let msb_params = MoleculeSymmetrisationBootstrapParams::builder()
        .reorientate_molecule(true)
        .use_magnetic_group(false)
        .loose_moi_threshold(1e-2)
        .loose_distance_threshold(1e-2)
        .target_moi_threshold(1e-8)
        .target_distance_threshold(1e-8)
        .infinite_order_to_finite(Some(8))
        .max_iterations(50)
        .verbose(2)
        .build()
        .unwrap();
    let mut msb_driver = MoleculeSymmetrisationBootstrapDriver::builder()
        .parameters(&msb_params)
        .molecule(&mol)
        .build()
        .unwrap();
    assert!(msb_driver.run().is_ok());

    let verifying_pd_params = SymmetryGroupDetectionParams::builder()
        .moi_thresholds(&[msb_params.target_moi_threshold])
        .distance_thresholds(&[msb_params.target_distance_threshold])
        .time_reversal(msb_params.use_magnetic_group)
        .write_symmetry_elements(true)
        .build()
        .unwrap();
    let mut verifying_pd_driver = SymmetryGroupDetectionDriver::builder()
        .parameters(&verifying_pd_params)
        .molecule(
            msb_driver
                .result()
                .ok()
                .map(|res| &res.symmetrised_molecule),
        )
        .build()
        .unwrap();
    assert!(verifying_pd_driver.run().is_ok());
    let verifying_pd_res = verifying_pd_driver.result().unwrap();
    assert_eq!(
        verifying_pd_res
            .unitary_symmetry
            .group_name
            .as_ref()
            .unwrap(),
        "D∞h"
    );
}

#[test]
#[ignore]
#[serial]
fn test_drivers_molecule_symmetrisation_bootstrap_cp10() {
    // log4rs::init_file("log4rs.yml", Default::default()).unwrap();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/cp10_flat.xyz");
    let mol = Molecule::from_xyz(&path, 1e-7);

    let msb_params = MoleculeSymmetrisationBootstrapParams::builder()
        .reorientate_molecule(true)
        .use_magnetic_group(false)
        .loose_moi_threshold(1e-1)
        .loose_distance_threshold(2e-1)
        .target_moi_threshold(1e-8)
        .target_distance_threshold(1e-8)
        .infinite_order_to_finite(Some(8))
        .max_iterations(50)
        .verbose(2)
        .build()
        .unwrap();
    let mut msb_driver = MoleculeSymmetrisationBootstrapDriver::builder()
        .parameters(&msb_params)
        .molecule(&mol)
        .build()
        .unwrap();
    assert!(msb_driver.run().is_ok());

    let verifying_pd_params = SymmetryGroupDetectionParams::builder()
        .moi_thresholds(&[msb_params.target_moi_threshold])
        .distance_thresholds(&[msb_params.target_distance_threshold])
        .time_reversal(msb_params.use_magnetic_group)
        .write_symmetry_elements(true)
        .build()
        .unwrap();
    let mut verifying_pd_driver = SymmetryGroupDetectionDriver::builder()
        .parameters(&verifying_pd_params)
        .molecule(
            msb_driver
                .result()
                .ok()
                .map(|res| &res.symmetrised_molecule),
        )
        .build()
        .unwrap();
    assert!(verifying_pd_driver.run().is_ok());
    let verifying_pd_res = verifying_pd_driver.result().unwrap();
    assert_eq!(
        verifying_pd_res
            .unitary_symmetry
            .group_name
            .as_ref()
            .unwrap(),
        "D10h"
    );
    let group =
        UnitaryRepresentedGroup::from_molecular_symmetry(&verifying_pd_res.unitary_symmetry, None)
            .unwrap();
    assert_eq!(group.name(), "D10h");
}

#[test]
#[ignore]
#[serial]
fn test_drivers_molecule_symmetrisation_bootstrap_h100() {
    // log4rs::init_file("log4rs.yml", Default::default()).unwrap();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/h100_imperfect.xyz");
    let mol = Molecule::from_xyz(&path, 1e-7);

    let msb_params = MoleculeSymmetrisationBootstrapParams::builder()
        .reorientate_molecule(true)
        .use_magnetic_group(false)
        .loose_moi_threshold(1e-1)
        .loose_distance_threshold(2e-1)
        .target_moi_threshold(1e-8)
        .target_distance_threshold(1e-8)
        .infinite_order_to_finite(Some(8))
        .max_iterations(50)
        .verbose(2)
        .build()
        .unwrap();
    let mut msb_driver = MoleculeSymmetrisationBootstrapDriver::builder()
        .parameters(&msb_params)
        .molecule(&mol)
        .build()
        .unwrap();
    assert!(msb_driver.run().is_ok());

    let verifying_pd_params = SymmetryGroupDetectionParams::builder()
        .moi_thresholds(&[msb_params.target_moi_threshold])
        .distance_thresholds(&[msb_params.target_distance_threshold])
        .time_reversal(msb_params.use_magnetic_group)
        .write_symmetry_elements(true)
        .build()
        .unwrap();
    let mut verifying_pd_driver = SymmetryGroupDetectionDriver::builder()
        .parameters(&verifying_pd_params)
        .molecule(
            msb_driver
                .result()
                .ok()
                .map(|res| &res.symmetrised_molecule),
        )
        .build()
        .unwrap();
    assert!(verifying_pd_driver.run().is_ok());
    let verifying_pd_res = verifying_pd_driver.result().unwrap();
    assert_eq!(
        verifying_pd_res
            .unitary_symmetry
            .group_name
            .as_ref()
            .unwrap(),
        "D100h"
    );
}
