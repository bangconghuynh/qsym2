// use log4rs;
use nalgebra::{Point3, Vector3};
use serial_test::serial;

use crate::auxiliary::molecule::Molecule;
use crate::drivers::molecule_symmetrisation_by_distance_matrix::{
    MoleculeSymmetrisationDistMatDriver, MoleculeSymmetrisationDistMatParams,
};
use crate::drivers::symmetry_group_detection::{
    SymmetryGroupDetectionDriver, SymmetryGroupDetectionParams,
};
use crate::drivers::QSym2Driver;
use crate::group::{GroupProperties, UnitaryRepresentedGroup};
use crate::symmetry::symmetry_group::SymmetryGroupProperties;

const ROOT: &str = env!("CARGO_MANIFEST_DIR");

#[test]
fn test_drivers_molecule_symmetrisation_distmat_vf6_magnetic_field() {
    // log4rs::init_file("log4rs.yml", Default::default()).unwrap();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/benzene_imperfect.xyz");
    let mol = Molecule::from_xyz(&path, 1e-7);
    // let pd_params = SymmetryGroupDetectionParams::builder()
    //     .moi_thresholds(&[1e-2, 1e-3, 1e-4])
    //     .distance_thresholds(&[1e-2, 1e-3, 1e-4])
    //     .fictitious_magnetic_fields(Some(vec![(
    //         Point3::new(0.0, 0.0, 0.0),
    //         Vector3::new(1.0, 1.0, 1.0),
    //     )]))
    //     .field_origin_com(true)
    //     .time_reversal(true)
    //     .write_symmetry_elements(false)
    //     .build()
    //     .unwrap();
    // let mut pd_driver = SymmetryGroupDetectionDriver::builder()
    //     .parameters(&pd_params)
    //     .xyz(Some(path.into()))
    //     .build()
    //     .unwrap();
    // assert!(pd_driver.run().is_ok());
    // let pd_res = pd_driver.result().unwrap();

    let ms_params = MoleculeSymmetrisationDistMatParams::builder()
        .loose_distance_threshold(1e-1)
        .target_moi_threshold(1e-8)
        .target_distance_threshold(1e-8)
        .reorientate_molecule(true)
        .max_iterations(10)
        .verbose(2)
        .build()
        .unwrap();
    let mut ms_driver = MoleculeSymmetrisationDistMatDriver::builder()
        .parameters(&ms_params)
        .molecule(&mol)
        .build()
        .unwrap();
    ms_driver.run();
    // assert!(ms_driver.run().is_ok());

    // let verifying_pd_params = SymmetryGroupDetectionParams::builder()
    //     .moi_thresholds(&[ms_params.target_moi_threshold])
    //     .distance_thresholds(&[ms_params.target_distance_threshold])
    //     .time_reversal(true)
    //     .write_symmetry_elements(true)
    //     .build()
    //     .unwrap();
    // let mut verifying_pd_driver = SymmetryGroupDetectionDriver::builder()
    //     .parameters(&verifying_pd_params)
    //     .molecule(ms_driver.result().ok().map(|res| &res.symmetrised_molecule))
    //     .build()
    //     .unwrap();
    // assert!(verifying_pd_driver.run().is_ok());
    // let verifying_pd_res = verifying_pd_driver.result().unwrap();
    // assert_eq!(
    //     verifying_pd_res
    //         .unitary_symmetry
    //         .group_name
    //         .as_ref()
    //         .unwrap(),
    //     "S6"
    // );
    // assert_eq!(
    //     verifying_pd_res
    //         .magnetic_symmetry
    //         .as_ref()
    //         .unwrap()
    //         .group_name
    //         .as_ref()
    //         .unwrap(),
    //     "D3d"
    // );
}

// #[test]
// fn test_drivers_molecule_symmetrisation_h4_magnetic_field() {
//     // log4rs::init_file("log4rs.yml", Default::default()).unwrap();
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/h4_imperfect.xyz");
//     let pd_params = SymmetryGroupDetectionParams::builder()
//         .moi_thresholds(&[1e-2, 1e-3, 1e-4])
//         .distance_thresholds(&[1e-2, 1e-3, 1e-4])
//         .fictitious_magnetic_fields(Some(vec![(Point3::new(0.0, 0.0, 0.0), Vector3::z())]))
//         .field_origin_com(true)
//         .time_reversal(true)
//         .write_symmetry_elements(false)
//         .build()
//         .unwrap();
//     let mut pd_driver = SymmetryGroupDetectionDriver::builder()
//         .parameters(&pd_params)
//         .xyz(Some(path.into()))
//         .build()
//         .unwrap();
//     assert!(pd_driver.run().is_ok());
//     let pd_res = pd_driver.result().unwrap();
//
//     let ms_params = MoleculeSymmetrisationParams::builder()
//         .use_magnetic_group(false)
//         .target_moi_threshold(1e-8)
//         .target_distance_threshold(1e-8)
//         .reorientate_molecule(true)
//         .max_iterations(10)
//         .verbose(2)
//         .build()
//         .unwrap();
//     let mut ms_driver = MoleculeSymmetrisationDriver::builder()
//         .parameters(&ms_params)
//         .target_symmetry_result(pd_res)
//         .build()
//         .unwrap();
//     assert!(ms_driver.run().is_ok());
//
//     let verifying_pd_params = SymmetryGroupDetectionParams::builder()
//         .moi_thresholds(&[ms_params.target_moi_threshold])
//         .distance_thresholds(&[ms_params.target_distance_threshold])
//         .time_reversal(true)
//         .write_symmetry_elements(true)
//         .build()
//         .unwrap();
//     let mut verifying_pd_driver = SymmetryGroupDetectionDriver::builder()
//         .parameters(&verifying_pd_params)
//         .molecule(ms_driver.result().ok().map(|res| &res.symmetrised_molecule))
//         .build()
//         .unwrap();
//     assert!(verifying_pd_driver.run().is_ok());
//     let verifying_pd_res = verifying_pd_driver.result().unwrap();
//     assert_eq!(
//         verifying_pd_res
//             .unitary_symmetry
//             .group_name
//             .as_ref()
//             .unwrap(),
//         "C4h"
//     );
//     assert_eq!(
//         verifying_pd_res
//             .magnetic_symmetry
//             .as_ref()
//             .unwrap()
//             .group_name
//             .as_ref()
//             .unwrap(),
//         "D4h"
//     );
// }
//
// #[test]
// fn test_drivers_molecule_symmetrisation_vf6_electric_field() {
//     // log4rs::init_file("log4rs.yml", Default::default()).unwrap();
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/vf6_imperfect.xyz");
//     let pd_params = SymmetryGroupDetectionParams::builder()
//         .moi_thresholds(&[1e-2, 5e-3])
//         .distance_thresholds(&[1e-2, 5e-3])
//         .fictitious_electric_fields(Some(vec![(Point3::new(0.0, 0.0, 0.0), Vector3::y())]))
//         .field_origin_com(true)
//         .time_reversal(true)
//         .write_symmetry_elements(false)
//         .build()
//         .unwrap();
//     let mut pd_driver = SymmetryGroupDetectionDriver::builder()
//         .parameters(&pd_params)
//         .xyz(Some(path.into()))
//         .build()
//         .unwrap();
//     assert!(pd_driver.run().is_ok());
//     let pd_res = pd_driver.result().unwrap();
//
//     let ms_params = MoleculeSymmetrisationParams::builder()
//         .use_magnetic_group(true)
//         .target_moi_threshold(1e-8)
//         .target_distance_threshold(1e-8)
//         .reorientate_molecule(true)
//         .max_iterations(10)
//         .verbose(2)
//         .build()
//         .unwrap();
//     let mut ms_driver = MoleculeSymmetrisationDriver::builder()
//         .parameters(&ms_params)
//         .target_symmetry_result(pd_res)
//         .build()
//         .unwrap();
//     assert!(ms_driver.run().is_ok());
//
//     let verifying_pd_params = SymmetryGroupDetectionParams::builder()
//         .moi_thresholds(&[ms_params.target_moi_threshold])
//         .distance_thresholds(&[ms_params.target_distance_threshold])
//         .time_reversal(true)
//         .write_symmetry_elements(true)
//         .build()
//         .unwrap();
//     let mut verifying_pd_driver = SymmetryGroupDetectionDriver::builder()
//         .parameters(&verifying_pd_params)
//         .molecule(ms_driver.result().ok().map(|res| &res.symmetrised_molecule))
//         .build()
//         .unwrap();
//     assert!(verifying_pd_driver.run().is_ok());
//     let verifying_pd_res = verifying_pd_driver.result().unwrap();
//     assert_eq!(
//         verifying_pd_res
//             .unitary_symmetry
//             .group_name
//             .as_ref()
//             .unwrap(),
//         "C4v"
//     );
// }
//
// #[test]
// fn test_drivers_molecule_symmetrisation_c2h2() {
//     // log4rs::init_file("log4rs.yml", Default::default()).unwrap();
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/c2h2_imperfect.xyz");
//     let pd_params = SymmetryGroupDetectionParams::builder()
//         .moi_thresholds(&[1e-2, 1e-3])
//         .distance_thresholds(&[1e-2, 1e-3])
//         .write_symmetry_elements(false)
//         .time_reversal(true)
//         .build()
//         .unwrap();
//     let mut pd_driver = SymmetryGroupDetectionDriver::builder()
//         .parameters(&pd_params)
//         .xyz(Some(path.into()))
//         .build()
//         .unwrap();
//     assert!(pd_driver.run().is_ok());
//     let pd_res = pd_driver.result().unwrap();
//
//     let ms_params = MoleculeSymmetrisationParams::builder()
//         .use_magnetic_group(true)
//         .target_moi_threshold(1e-8)
//         .target_distance_threshold(1e-8)
//         .reorientate_molecule(true)
//         .max_iterations(10)
//         .infinite_order_to_finite(Some(4))
//         .verbose(2)
//         .build()
//         .unwrap();
//     let mut ms_driver = MoleculeSymmetrisationDriver::builder()
//         .parameters(&ms_params)
//         .target_symmetry_result(pd_res)
//         .build()
//         .unwrap();
//     assert!(ms_driver.run().is_ok());
//
//     let verifying_pd_params = SymmetryGroupDetectionParams::builder()
//         .moi_thresholds(&[ms_params.target_moi_threshold])
//         .distance_thresholds(&[ms_params.target_distance_threshold])
//         .time_reversal(true)
//         .write_symmetry_elements(true)
//         .build()
//         .unwrap();
//     let mut verifying_pd_driver = SymmetryGroupDetectionDriver::builder()
//         .parameters(&verifying_pd_params)
//         .molecule(ms_driver.result().ok().map(|res| &res.symmetrised_molecule))
//         .build()
//         .unwrap();
//     assert!(verifying_pd_driver.run().is_ok());
//     let verifying_pd_res = verifying_pd_driver.result().unwrap();
//     assert_eq!(
//         verifying_pd_res
//             .unitary_symmetry
//             .group_name
//             .as_ref()
//             .unwrap(),
//         "D∞h"
//     );
//     assert_eq!(
//         verifying_pd_res
//             .magnetic_symmetry
//             .as_ref()
//             .unwrap()
//             .group_name
//             .as_ref()
//             .unwrap(),
//         "D∞h + θ·D∞h"
//     );
// }
//
// #[test]
// #[ignore]
// #[serial]
// fn test_drivers_molecule_symmetrisation_cp10() {
//     // log4rs::init_file("log4rs.yml", Default::default()).unwrap();
//     let path: String = format!("{}{}", ROOT, "/tests/xyz/cp10_flat.xyz");
//     let params = SymmetryGroupDetectionParams::builder()
//         .distance_thresholds(&[2e-1])
//         .moi_thresholds(&[1e-1])
//         .time_reversal(false)
//         .write_symmetry_elements(false)
//         .build()
//         .unwrap();
//     let mut pd_driver = SymmetryGroupDetectionDriver::builder()
//         .parameters(&params)
//         .xyz(Some(path.into()))
//         .build()
//         .unwrap();
//     assert!(pd_driver.run().is_ok());
//     let pd_res = pd_driver.result().unwrap();
//     assert_eq!(pd_res.unitary_symmetry.group_name.as_ref().unwrap(), "D10h");
//
//     let ms_params = MoleculeSymmetrisationParams::builder()
//         .use_magnetic_group(false)
//         .target_moi_threshold(1e-8)
//         .target_distance_threshold(1e-8)
//         .reorientate_molecule(true)
//         .max_iterations(10)
//         .verbose(2)
//         .build()
//         .unwrap();
//     let mut ms_driver = MoleculeSymmetrisationDriver::builder()
//         .parameters(&ms_params)
//         .target_symmetry_result(pd_res)
//         .build()
//         .unwrap();
//     assert!(ms_driver.run().is_ok());
//
//     let verifying_pd_params = SymmetryGroupDetectionParams::builder()
//         .moi_thresholds(&[ms_params.target_moi_threshold])
//         .distance_thresholds(&[ms_params.target_distance_threshold])
//         .time_reversal(false)
//         .write_symmetry_elements(true)
//         .build()
//         .unwrap();
//     let mut verifying_pd_driver = SymmetryGroupDetectionDriver::builder()
//         .parameters(&verifying_pd_params)
//         .molecule(ms_driver.result().ok().map(|res| &res.symmetrised_molecule))
//         .build()
//         .unwrap();
//     assert!(verifying_pd_driver.run().is_ok());
//     let verifying_pd_res = verifying_pd_driver.result().unwrap();
//     assert_eq!(
//         verifying_pd_res
//             .unitary_symmetry
//             .group_name
//             .as_ref()
//             .unwrap(),
//         "D10h"
//     );
//     let group =
//         UnitaryRepresentedGroup::from_molecular_symmetry(&verifying_pd_res.unitary_symmetry, None)
//             .unwrap();
//     assert_eq!(group.name(), "D10h");
// }
