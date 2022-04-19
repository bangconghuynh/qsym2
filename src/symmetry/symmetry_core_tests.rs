use crate::aux::molecule::Molecule;
use crate::symmetry::symmetry_core::Symmetry;

const ROOT: &str = env!("CARGO_MANIFEST_DIR");

#[test]
fn test_symmetry_constructor() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/c3h3.xyz");
    let mol = Molecule::from_xyz(&path);
    let sym = Symmetry::builder()
        .threshold(1e-14)
        .moi_threshold(1e-14)
        .molecule(&mol)
        .build()
        .unwrap();
    // println!("{:?}", &sym);
}
