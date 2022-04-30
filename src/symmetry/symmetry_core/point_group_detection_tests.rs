use env_logger;
use crate::aux::geometry::Transform;
use crate::aux::molecule::Molecule;
use crate::symmetry::symmetry_core::Symmetry;

const ROOT: &str = env!("CARGO_MANIFEST_DIR");

#[test]
fn test_point_group_detection_atom() {
    env_logger::init();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/th.xyz");
    let mut mol = Molecule::from_xyz(&path, 1e-7);
    mol.recentre_mut();
    let mut sym = Symmetry::builder()
        .moi_threshold(1e-14)
        .molecule(&mol)
        .build()
        .unwrap();
    sym.analyse();
}
