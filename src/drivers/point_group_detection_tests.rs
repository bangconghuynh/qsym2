use log4rs;

use nalgebra::{Point3, Vector3};

use crate::drivers::point_group_detection::{PointGroupDetectionDriver, PointGroupDetectionParams};
use crate::drivers::QSym2Driver;

const ROOT: &str = env!("CARGO_MANIFEST_DIR");

#[test]
fn test_drivers_point_group_detection_vf6() {
    log4rs::init_file("log4rs.yml", Default::default()).unwrap();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/vf6.xyz");
    let params = PointGroupDetectionParams::builder()
        .distance_thresholds(&[1e-6, 1e-7])
        .moi_thresholds(&[1e-6, 1e-7])
        .fictitious_magnetic_fields(Some(vec![(Point3::origin(), Vector3::new(1.0, 1.0, 1.0))]))
        .time_reversal(true)
        .build()
        .unwrap();
    let mut pd_driver = PointGroupDetectionDriver::builder()
        .parameters(params)
        .xyz(Some(path.clone()))
        .build()
        .unwrap();
    pd_driver.run().unwrap();
    let pd_res = pd_driver.result().unwrap();
    assert_eq!(pd_res.pre_symmetry.dist_threshold, 1e-7);
    assert_eq!(pd_res.pre_symmetry.moi_threshold, 1e-7);
    assert_eq!(pd_res.unitary_symmetry.group_name.as_ref().unwrap(), "S6");
    assert_eq!(
        pd_res
            .magnetic_symmetry
            .as_ref()
            .unwrap()
            .group_name
            .as_ref()
            .unwrap(),
        "D3d"
    );

    let params = PointGroupDetectionParams::builder()
        .distance_thresholds(&[1e-6, 1e-7, 1e-8, 1e-9])
        .moi_thresholds(&[1e-6, 1e-7, 1e-8, 1e-9])
        .fictitious_magnetic_fields(Some(vec![(Point3::origin(), Vector3::x())]))
        .fictitious_electric_fields(Some(vec![(Point3::origin(), Vector3::y())]))
        .time_reversal(true)
        .build()
        .unwrap();
    let mut pd_driver = PointGroupDetectionDriver::builder()
        .parameters(params)
        .xyz(Some(path.clone()))
        .build()
        .unwrap();
    pd_driver.run().unwrap();
    let pd_res = pd_driver.result().unwrap();
    assert_eq!(pd_res.pre_symmetry.dist_threshold, 1e-9);
    assert_eq!(pd_res.pre_symmetry.moi_threshold, 1e-9);
    assert_eq!(pd_res.unitary_symmetry.group_name.as_ref().unwrap(), "Cs");
    assert_eq!(
        pd_res
            .magnetic_symmetry
            .as_ref()
            .unwrap()
            .group_name
            .as_ref()
            .unwrap(),
        "C2v"
    );
}

#[test]
fn test_drivers_point_group_detection_c2h2() {
    log4rs::init_file("log4rs.yml", Default::default()).unwrap();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c2h2.xyz");
    let params = PointGroupDetectionParams::builder()
        .distance_thresholds(&[1e-6, 1e-7])
        .moi_thresholds(&[1e-6, 1e-7])
        .time_reversal(true)
        .build()
        .unwrap();
    let mut pd_driver = PointGroupDetectionDriver::builder()
        .parameters(params)
        .xyz(Some(path.clone()))
        .build()
        .unwrap();
    pd_driver.run().unwrap();
    let pd_res = pd_driver.result().unwrap();
    assert_eq!(pd_res.unitary_symmetry.group_name.as_ref().unwrap(), "D∞h");
    assert_eq!(
        pd_res
            .magnetic_symmetry
            .as_ref()
            .unwrap()
            .group_name
            .as_ref()
            .unwrap(),
        "D∞h + θ·D∞h"
    );

    let params = PointGroupDetectionParams::builder()
        .distance_thresholds(&[1e-6, 1e-7])
        .moi_thresholds(&[1e-6, 1e-7])
        .fictitious_magnetic_fields(Some(vec![(Point3::new(0.5, 0.5, 0.5), Vector3::new(1.0, 1.0, 1.0))]))
        .time_reversal(true)
        .build()
        .unwrap();
    let mut pd_driver = PointGroupDetectionDriver::builder()
        .parameters(params)
        .xyz(Some(path.clone()))
        .build()
        .unwrap();
    pd_driver.run().unwrap();
    let pd_res = pd_driver.result().unwrap();
    assert_eq!(pd_res.pre_symmetry.dist_threshold, 1e-7);
    assert_eq!(pd_res.pre_symmetry.moi_threshold, 1e-7);
    assert_eq!(pd_res.unitary_symmetry.group_name.as_ref().unwrap(), "C∞h");
    assert_eq!(
        pd_res
            .magnetic_symmetry
            .as_ref()
            .unwrap()
            .group_name
            .as_ref()
            .unwrap(),
        "D∞h"
    );

    let params = PointGroupDetectionParams::builder()
        .distance_thresholds(&[1e-6, 1e-7, 1e-15])
        .moi_thresholds(&[1e-6, 1e-7, 1e-15])
        .fictitious_electric_fields(Some(vec![(Point3::new(0.5, 0.5, 0.5), Vector3::new(1.0, 1.0, 1.0))]))
        .time_reversal(true)
        .build()
        .unwrap();
    let mut pd_driver = PointGroupDetectionDriver::builder()
        .parameters(params)
        .xyz(Some(path.clone()))
        .build()
        .unwrap();
    pd_driver.run().unwrap();
    let pd_res = pd_driver.result().unwrap();
    assert_eq!(pd_res.pre_symmetry.dist_threshold, 1e-7);
    assert_eq!(pd_res.pre_symmetry.moi_threshold, 1e-7);
    assert_eq!(pd_res.unitary_symmetry.group_name.as_ref().unwrap(), "C∞v");
    assert_eq!(
        pd_res
            .magnetic_symmetry
            .as_ref()
            .unwrap()
            .group_name
            .as_ref()
            .unwrap(),
        "C∞v + θ·C∞v"
    );
}
