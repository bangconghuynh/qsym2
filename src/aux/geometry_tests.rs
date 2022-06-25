use crate::aux::atom::{Atom, ElementMap};
use crate::aux::geometry;

#[test]
fn test_check_regular_polygon() {
    let emap = ElementMap::new();
    let atom_0 = Atom::from_xyz("B 0.1 0.1 1.8", &emap, 1e-7).unwrap();
    let atom_1 = Atom::from_xyz("B 0.0 0.1 1.8", &emap, 1e-7).unwrap();
    let atom_2 = Atom::from_xyz("B 0.0 0.0 1.8", &emap, 1e-7).unwrap();
    let atom_3 = Atom::from_xyz("B 0.1 0.0 1.8", &emap, 1e-7).unwrap();
    assert!(!geometry::check_regular_polygon(&[
        &atom_0, &atom_2, &atom_3
    ]));
    assert!(geometry::check_regular_polygon(&[
        &atom_0, &atom_1, &atom_2, &atom_3
    ]));
    assert!(geometry::check_regular_polygon(&[
        &atom_0, &atom_2, &atom_1, &atom_3
    ]));

    let atom_4 = Atom::from_xyz("C -2.1191966  0.7095799 0.0000000", &emap, 1e-6).unwrap();
    let atom_5 = Atom::from_xyz("C -3.4698725  0.3448934 0.0000000", &emap, 1e-6).unwrap();
    let atom_6 = Atom::from_xyz("C -1.1280310 -0.2777965 0.0000000", &emap, 1e-6).unwrap();
    let atom_7 = Atom::from_xyz("C -3.8293826 -1.0071693 0.0000000", &emap, 1e-6).unwrap();
    let atom_8 = Atom::from_xyz("C -2.8382171 -1.9945456 0.0000000", &emap, 1e-6).unwrap();
    let atom_9 = Atom::from_xyz("C -1.4875413 -1.6298593 0.0000000", &emap, 1e-6).unwrap();
    assert!(geometry::check_regular_polygon(&[
        &atom_4, &atom_5, &atom_6, &atom_7, &atom_8, &atom_9
    ]));
    assert!(geometry::check_regular_polygon(&[
        &atom_6, &atom_5, &atom_4, &atom_8, &atom_9, &atom_7
    ]));
}
