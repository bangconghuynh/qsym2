use env_logger;

use nalgebra::{Point3, Vector3};

use crate::drivers::QSym2Driver;
use crate::drivers::point_group_detection::{PointGroupDetectionDriver, PointGroupDetectionParams};

const ROOT: &str = env!("CARGO_MANIFEST_DIR");

#[test]
fn test_drivers_point_group_detection_vf6() {
    env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/benzene.xyz");
    let params = PointGroupDetectionParams::builder()
        .distance_thresholds(&[1e-6, 1e-13])
        .moi_thresholds(&[1e-6, 1e-13])
        .fictitious_magnetic_fields(Some(vec![(Point3::origin(), Vector3::new(1.0, 1.0, 1.0))]))
        .fictitious_electric_fields(Some(vec![(Point3::origin(), Vector3::x())]))
        .time_reversal(true)
        .build()
        .unwrap();
    let mut pd_driver = PointGroupDetectionDriver::builder()
        .parameters(params)
        .xyz(Some(path))
        .build()
        .unwrap();
    pd_driver.run().unwrap();
}
