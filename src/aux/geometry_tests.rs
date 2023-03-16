use nalgebra::Vector3;

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

#[test]
fn test_standard_positive_hemisphere() {
    let poshem_c = geometry::PositiveHemisphere::new_standard_cartesian(1e-7);
    let poshem_s = geometry::PositiveHemisphere::new_standard_spherical(1e-7);

    let axis_z = Vector3::<f64>::z();
    assert!(poshem_c.check_positive_pole(&axis_z));
    assert!(poshem_s.check_positive_pole(&axis_z));
    assert!(!poshem_c.check_positive_pole(&-axis_z));
    assert!(!poshem_s.check_positive_pole(&-axis_z));

    let axis_x = Vector3::<f64>::x();
    assert!(poshem_c.check_positive_pole(&axis_x));
    assert!(poshem_s.check_positive_pole(&axis_x));
    assert!(!poshem_c.check_positive_pole(&-axis_x));
    assert!(!poshem_s.check_positive_pole(&-axis_x));

    let axis_y = Vector3::<f64>::y();
    assert!(poshem_c.check_positive_pole(&axis_y));
    assert!(poshem_s.check_positive_pole(&axis_y));
    assert!(!poshem_c.check_positive_pole(&-axis_y));
    assert!(!poshem_s.check_positive_pole(&-axis_y));

    let axis_0 = Vector3::new(-1.0, -1.0, 1.0);
    assert!(poshem_c.check_positive_pole(&axis_0));
    assert!(poshem_s.check_positive_pole(&axis_0));
    assert!(!poshem_c.check_positive_pole(&-axis_0));
    assert!(!poshem_s.check_positive_pole(&-axis_0));

    let axis_1 = Vector3::new(-1.0, 1.0, 0.0);
    assert!(!poshem_c.check_positive_pole(&axis_1));
    assert!(!poshem_s.check_positive_pole(&axis_1));
    assert!(poshem_c.check_positive_pole(&-axis_1));
    assert!(poshem_s.check_positive_pole(&-axis_1));

    let axis_2 = Vector3::new(1.0, 0.0, -0.2);
    assert!(!poshem_c.check_positive_pole(&axis_2));
    assert!(!poshem_s.check_positive_pole(&axis_2));
    assert!(poshem_c.check_positive_pole(&-axis_2));
    assert!(poshem_s.check_positive_pole(&-axis_2));

    let axis_3 = Vector3::new(-0.1, 0.8, 0.0);
    assert!(!poshem_c.check_positive_pole(&axis_3));
    assert!(!poshem_s.check_positive_pole(&axis_3));
    assert!(poshem_c.check_positive_pole(&-axis_3));
    assert!(poshem_s.check_positive_pole(&-axis_3));

    let axis_4 = Vector3::new(-0.1, 0.8, 0.2);
    assert!(poshem_c.check_positive_pole(&axis_4));
    assert!(poshem_s.check_positive_pole(&axis_4));
    assert!(!poshem_c.check_positive_pole(&-axis_4));
    assert!(!poshem_s.check_positive_pole(&-axis_4));
}

#[test]
fn test_custom_positive_hemisphere() {
    let poshem_s = geometry::PositiveHemisphere::new_spherical_disjoint_equatorial_arcs(
        Vector3::z(),
        Vector3::x(),
        3,
        1e-7
    );

    let axis_x = Vector3::<f64>::x();
    assert!(poshem_s.check_positive_pole(&axis_x));
    assert!(!poshem_s.check_positive_pole(&-axis_x));

    let axis_y = Vector3::<f64>::y();
    assert!(!poshem_s.check_positive_pole(&axis_y));
    assert!(poshem_s.check_positive_pole(&-axis_y));

    let axis_0 = Vector3::<f64>::new(-0.5, 3.0f64.sqrt() / 2.0, 0.0);
    assert!(poshem_s.check_positive_pole(&axis_0));
    assert!(!poshem_s.check_positive_pole(&-axis_0));

    let axis_1 = Vector3::<f64>::new(-0.5, -3.0f64.sqrt() / 2.0, 0.0);
    assert!(poshem_s.check_positive_pole(&axis_1));
    assert!(!poshem_s.check_positive_pole(&-axis_1));

    let poshem_s2 = geometry::PositiveHemisphere::new_spherical_disjoint_equatorial_arcs(
        Vector3::z(),
        Vector3::new(-0.5, 3.0f64.sqrt() / 2.0, 0.0),
        3,
        1e-7
    );

    assert!(poshem_s2.check_positive_pole(&axis_x));
    assert!(!poshem_s2.check_positive_pole(&-axis_x));
    assert!(!poshem_s2.check_positive_pole(&axis_y));
    assert!(poshem_s2.check_positive_pole(&-axis_y));
    assert!(poshem_s2.check_positive_pole(&axis_0));
    assert!(!poshem_s2.check_positive_pole(&-axis_0));
    assert!(poshem_s2.check_positive_pole(&axis_1));
    assert!(!poshem_s2.check_positive_pole(&-axis_1));

    let poshem_s3 = geometry::PositiveHemisphere::new_spherical_disjoint_equatorial_arcs(
        Vector3::z(),
        Vector3::new(-0.5, -3.0f64.sqrt() / 2.0, 0.0),
        3,
        1e-7
    );

    assert!(poshem_s3.check_positive_pole(&axis_x));
    assert!(!poshem_s3.check_positive_pole(&-axis_x));
    assert!(!poshem_s3.check_positive_pole(&axis_y));
    assert!(poshem_s3.check_positive_pole(&-axis_y));
    assert!(poshem_s3.check_positive_pole(&axis_0));
    assert!(!poshem_s3.check_positive_pole(&-axis_0));
    assert!(poshem_s3.check_positive_pole(&axis_1));
    assert!(!poshem_s3.check_positive_pole(&-axis_1));
}
