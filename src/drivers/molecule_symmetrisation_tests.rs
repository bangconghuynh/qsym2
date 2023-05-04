use log4rs;

use nalgebra::{Point3, Vector3};

use crate::drivers::QSym2Driver;
use crate::drivers::symmetry_group_detection::{
    SymmetryGroupDetectionDriver, SymmetryGroupDetectionParams,
};
use crate::drivers::molecule_symmetrisation::{
    MoleculeSymmetrisationDriver, MoleculeSymmetrisationParams,
};


const ROOT: &str = env!("CARGO_MANIFEST_DIR");

#[test]
fn test_drivers_molecule_symmetrisation_vf6() {
    log4rs::init_file("log4rs.yml", Default::default()).unwrap();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/vf6_imperfect.xyz");
    let pd_params = SymmetryGroupDetectionParams::builder()
        .moi_thresholds(&[1e-2, 1e-3, 1e-4])
        .distance_thresholds(&[1e-2, 1e-3, 1e-4])
        .fictitious_magnetic_fields(Some(vec![(Point3::new(0.0, 0.0, 0.0), Vector3::new(1.0, 1.0, 1.0))]))
        .fictitious_origin_com(true)
        .time_reversal(true)
        .write_symmetry_elements(true)
        .build()
        .unwrap();
    let mut pd_driver = SymmetryGroupDetectionDriver::builder()
        .parameters(&pd_params)
        .xyz(Some(path.clone()))
        .build()
        .unwrap();
    pd_driver.run().unwrap();
    let pd_res = pd_driver.result().unwrap();

    let ms_params = MoleculeSymmetrisationParams::builder()
        .use_magnetic_symmetry(true)
        .target_moi_threshold(1e-8)
        .target_distance_threshold(1e-8)
        .max_iterations(100)
        .verbose(2)
        .build()
        .unwrap();
    let mut ms_driver = MoleculeSymmetrisationDriver::builder()
        .parameters(&ms_params)
        .target_symmetry_result(&pd_res)
        .build()
        .unwrap();
    let ms_run = ms_driver.run();
    println!("{:?}", ms_run);
}
