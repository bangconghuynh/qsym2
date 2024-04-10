// use env_logger;
use nalgebra::Point3;
use ndarray::{array, Array1};

use crate::analysis::{EigenvalueComparisonMode, RepAnalysis};
use crate::auxiliary::atom::{Atom, ElementMap};
use crate::auxiliary::geometry::Transform;
use crate::auxiliary::molecule::Molecule;
use crate::chartab::chartab_symbols::DecomposedSymbol;
use crate::group::UnitaryRepresentedGroup;
use crate::sandbox::target::pes::pes_analysis::PESSymmetryOrbit;
use crate::sandbox::target::pes::PES;
use crate::symmetry::symmetry_core::{PreSymmetry, Symmetry};
use crate::symmetry::symmetry_group::SymmetryGroupProperties;
use crate::symmetry::symmetry_symbols::MullikenIrrepSymbol;
use crate::symmetry::symmetry_transformation::SymmetryTransformationKind;

#[test]
fn test_pes_orbit_rep_analysis_d4h() {
    // env_logger::init();
    let emap = ElementMap::new();
    let atm_s0 = Atom::from_xyz("S +1.0 +1.0 0.0", &emap, 1e-7).unwrap();
    let atm_s1 = Atom::from_xyz("S -1.0 +1.0 0.0", &emap, 1e-7).unwrap();
    let atm_s2 = Atom::from_xyz("S -1.0 -1.0 0.0", &emap, 1e-7).unwrap();
    let atm_s3 = Atom::from_xyz("S +1.0 -1.0 0.0", &emap, 1e-7).unwrap();

    let mol_s4 = Molecule::from_atoms(
        &[
            atm_s0.clone(),
            atm_s1.clone(),
            atm_s2.clone(),
            atm_s3.clone(),
        ],
        1e-7,
    )
    .recentre();

    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol_s4)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false).unwrap();
    let group_u_d4h = UnitaryRepresentedGroup::from_molecular_symmetry(&sym, None).unwrap();

    // =========
    // 2D domain
    // =========
    let grid_points = vec![
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(0.5, 0.5, 0.0),
        Point3::new(1.5, 0.5, 0.0),
        Point3::new(1.5, 1.5, 0.0),
        Point3::new(0.5, 1.5, 0.0),
    ];

    let weight: Array1<f64> = Array1::from_iter(
        grid_points
            .iter()
            .map(|pt| (-(pt - Point3::<f64>::origin()).magnitude_squared()).exp()),
    );

    // A1g
    let a1g_values = array![
        [0.0, 1.0, 2.0, 2.0, 2.0], // E
        [0.0, 1.0, 2.0, 2.0, 2.0], // C4
        [0.0, 1.0, 2.0, 2.0, 2.0], // C4^3
        [0.0, 1.0, 2.0, 2.0, 2.0], // C2z
        [0.0, 1.0, 2.0, 2.0, 2.0], // C2y
        [0.0, 1.0, 2.0, 2.0, 2.0], // C2x
        [0.0, 1.0, 2.0, 2.0, 2.0], // C2xy
        [0.0, 1.0, 2.0, 2.0, 2.0], // C2xy'
        [0.0, 1.0, 2.0, 2.0, 2.0], // i
        [0.0, 1.0, 2.0, 2.0, 2.0], // S4
        [0.0, 1.0, 2.0, 2.0, 2.0], // S4^3
        [0.0, 1.0, 2.0, 2.0, 2.0], // σh
        [0.0, 1.0, 2.0, 2.0, 2.0], // σvxz
        [0.0, 1.0, 2.0, 2.0, 2.0], // σvyz
        [0.0, 1.0, 2.0, 2.0, 2.0], // σd
        [0.0, 1.0, 2.0, 2.0, 2.0], // σd'
    ];

    let a1g_pes = PES::builder()
        .group(&group_u_d4h)
        .grid_points(grid_points.clone())
        .values(a1g_values)
        .build()
        .unwrap();

    let mut orbit_a1g_pes = PESSymmetryOrbit::builder()
        .origin(&a1g_pes)
        .integrality_threshold(1e-7)
        .linear_independence_threshold(1e-7)
        .symmetry_transformation_kind(SymmetryTransformationKind::Spatial)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
        .build()
        .unwrap();
    let _ = orbit_a1g_pes
        .calc_smat(Some(&weight), None, true)
        .unwrap()
        .calc_xmat(false);
    assert_eq!(
        orbit_a1g_pes.analyse_rep().unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new("||A|_(1g)|").unwrap()
    );

    // B1g (x2 - y2)
    let b1g_values = array![
        [0.0, 0.0, 2.0, 0.0, -2.0], // E
        [0.0, 0.0, -2.0, 0.0, 2.0], // C4
        [0.0, 0.0, -2.0, 0.0, 2.0], // C4^3
        [0.0, 0.0, 2.0, 0.0, -2.0], // C2z
        [0.0, 0.0, 2.0, 0.0, -2.0], // C2y
        [0.0, 0.0, 2.0, 0.0, -2.0], // C2x
        [0.0, 0.0, -2.0, 0.0, 2.0], // C2xy
        [0.0, 0.0, -2.0, 0.0, 2.0], // C2xy'
        [0.0, 0.0, 2.0, 0.0, -2.0], // i
        [0.0, 0.0, -2.0, 0.0, 2.0], // S4
        [0.0, 0.0, -2.0, 0.0, 2.0], // S4^3
        [0.0, 0.0, 2.0, 0.0, -2.0], // σh
        [0.0, 0.0, 2.0, 0.0, -2.0], // σvxz
        [0.0, 0.0, 2.0, 0.0, -2.0], // σvyz
        [0.0, 0.0, -2.0, 0.0, 2.0], // σd
        [0.0, 0.0, -2.0, 0.0, 2.0], // σd'
    ];

    let b1g_pes = PES::builder()
        .group(&group_u_d4h)
        .grid_points(grid_points.clone())
        .values(b1g_values)
        .build()
        .unwrap();

    let mut orbit_b1g_pes = PESSymmetryOrbit::builder()
        .origin(&b1g_pes)
        .integrality_threshold(1e-7)
        .linear_independence_threshold(1e-7)
        .symmetry_transformation_kind(SymmetryTransformationKind::Spatial)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
        .build()
        .unwrap();
    let _ = orbit_b1g_pes
        .calc_smat(Some(&weight), None, true)
        .unwrap()
        .calc_xmat(false);
    assert_eq!(
        orbit_b1g_pes.analyse_rep().unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new("||B|_(1g)|").unwrap()
    );

    // B2g (xy)
    let b2g_values = array![
        [0.0, 1.0, 2.0, 2.0, 2.0],     // E
        [0.0, -1.0, -2.0, -2.0, -2.0], // C4
        [0.0, -1.0, -2.0, -2.0, -2.0], // C4^3
        [0.0, 1.0, 2.0, 2.0, 2.0],     // C2z
        [0.0, -1.0, -2.0, -2.0, -2.0], // C2y
        [0.0, -1.0, -2.0, -2.0, -2.0], // C2x
        [0.0, 1.0, 2.0, 2.0, 2.0],     // C2xy
        [0.0, 1.0, 2.0, 2.0, 2.0],     // C2xy'
        [0.0, 1.0, 2.0, 2.0, 2.0],     // i
        [0.0, -1.0, -2.0, -2.0, -2.0], // S4
        [0.0, -1.0, -2.0, -2.0, -2.0], // S4^3
        [0.0, 1.0, 2.0, 2.0, 2.0],     // σh
        [0.0, -1.0, -2.0, -2.0, -2.0], // σvxz
        [0.0, -1.0, -2.0, -2.0, -2.0], // σvyz
        [0.0, 1.0, 2.0, 2.0, 2.0],     // σd
        [0.0, 1.0, 2.0, 2.0, 2.0],     // σd'
    ];

    let b2g_pes = PES::builder()
        .group(&group_u_d4h)
        .grid_points(grid_points.clone())
        .values(b2g_values)
        .build()
        .unwrap();

    let mut orbit_b2g_pes = PESSymmetryOrbit::builder()
        .origin(&b2g_pes)
        .integrality_threshold(1e-7)
        .linear_independence_threshold(1e-7)
        .symmetry_transformation_kind(SymmetryTransformationKind::Spatial)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
        .build()
        .unwrap();
    let _ = orbit_b2g_pes
        .calc_smat(Some(&weight), None, true)
        .unwrap()
        .calc_xmat(false);
    assert_eq!(
        orbit_b2g_pes.analyse_rep().unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new("||B|_(2g)|").unwrap()
    );

    // Eu (y)
    let eu_values = array![
        [0.0, 1.0, 2.0, 2.0, 2.0],     // E
        [0.0, 1.0, 2.0, 2.0, 2.0],     // C4
        [0.0, -1.0, -2.0, -2.0, -2.0], // C4^3
        [0.0, -1.0, -2.0, -2.0, -2.0], // C2z
        [0.0, 1.0, 2.0, 2.0, 2.0],     // C2y
        [0.0, -1.0, -2.0, -2.0, -2.0], // C2x
        [0.0, 1.0, 2.0, 2.0, 2.0],     // C2xy
        [0.0, -1.0, -2.0, -2.0, -2.0], // C2xy'
        [0.0, -1.0, -2.0, -2.0, -2.0], // i
        [0.0, 1.0, 2.0, 2.0, 2.0],     // S4
        [0.0, -1.0, -2.0, -2.0, -2.0], // S4^3
        [0.0, 1.0, 2.0, 2.0, 2.0],     // σh
        [0.0, -1.0, -2.0, -2.0, -2.0], // σvxz
        [0.0, 1.0, 2.0, 2.0, 2.0],     // σvyz
        [0.0, -1.0, -2.0, -2.0, -2.0], // σd
        [0.0, 1.0, 2.0, 2.0, 2.0],     // σd'
    ];

    let eu_pes = PES::builder()
        .group(&group_u_d4h)
        .grid_points(grid_points)
        .values(eu_values)
        .build()
        .unwrap();

    let mut orbit_eu_pes = PESSymmetryOrbit::builder()
        .origin(&eu_pes)
        .integrality_threshold(1e-7)
        .linear_independence_threshold(1e-7)
        .symmetry_transformation_kind(SymmetryTransformationKind::Spatial)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
        .build()
        .unwrap();
    let _ = orbit_eu_pes
        .calc_smat(Some(&weight), None, true)
        .unwrap()
        .calc_xmat(false);
    assert_eq!(
        orbit_eu_pes.analyse_rep().unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new("||E|_(u)|").unwrap()
    );

    // =========
    // 3D domain
    // =========
    let grid_points = vec![
        Point3::new(0.0, 0.0, 1.0),
        Point3::new(0.5, 0.5, 1.0),
        Point3::new(1.5, 0.5, 1.0),
        Point3::new(1.5, 1.5, 1.0),
        Point3::new(0.5, 1.5, 1.0),
    ];

    let weight: Array1<f64> = Array1::from_iter(
        grid_points
            .iter()
            .map(|pt| (-(pt - Point3::<f64>::origin()).magnitude_squared()).exp()),
    );

    // B1u (xyz)
    let b1u_values = array![
        [0.0, 1.0, 2.0, 3.0, 2.0],     // E
        [0.0, -1.0, -2.0, -3.0, -2.0], // C4
        [0.0, -1.0, -2.0, -3.0, -2.0], // C4^3
        [0.0, 1.0, 2.0, 3.0, 2.0],     // C2z
        [0.0, 1.0, 2.0, 3.0, 2.0],     // C2y
        [0.0, 1.0, 2.0, 3.0, 2.0],     // C2x
        [0.0, -1.0, -2.0, -3.0, -2.0], // C2xy
        [0.0, -1.0, -2.0, -3.0, -2.0], // C2xy'
        [0.0, -1.0, -2.0, -3.0, -2.0], // i
        [0.0, 1.0, 2.0, 3.0, 2.0],     // S4
        [0.0, 1.0, 2.0, 3.0, 2.0],     // S4^3
        [0.0, -1.0, -2.0, -3.0, -2.0], // σh
        [0.0, -1.0, -2.0, -3.0, -2.0], // σvxz
        [0.0, -1.0, -2.0, -3.0, -2.0], // σvyz
        [0.0, 1.0, 2.0, 3.0, 2.0],     // σd
        [0.0, 1.0, 2.0, 3.0, 2.0],     // σd'
    ];

    let b1u_pes = PES::builder()
        .group(&group_u_d4h)
        .grid_points(grid_points.clone())
        .values(b1u_values)
        .build()
        .unwrap();

    let mut orbit_b1u_pes = PESSymmetryOrbit::builder()
        .origin(&b1u_pes)
        .integrality_threshold(1e-7)
        .linear_independence_threshold(1e-7)
        .symmetry_transformation_kind(SymmetryTransformationKind::Spatial)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
        .build()
        .unwrap();
    let _ = orbit_b1u_pes
        .calc_smat(Some(&weight), None, true)
        .unwrap()
        .calc_xmat(false);
    assert_eq!(
        orbit_b1u_pes.analyse_rep().unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new("||B|_(1u)|").unwrap()
    );

    // Eg (xz)
    let f1 = 0.5 * (15.0 / std::f64::consts::PI).sqrt() * (0.5 * 1.0)
        / (0.5f64.powi(2) + 0.5f64.powi(2) + 1.0f64.powi(2));
    let f2 = 0.5 * (15.0 / std::f64::consts::PI).sqrt() * (1.5 * 1.0)
        / (1.5f64.powi(2) + 0.5f64.powi(2) + 1.0f64.powi(2));
    let f3 = 0.5 * (15.0 / std::f64::consts::PI).sqrt() * (1.5 * 1.0)
        / (1.5f64.powi(2) + 1.5f64.powi(2) + 1.0f64.powi(2));
    let f4 = 0.5 * (15.0 / std::f64::consts::PI).sqrt() * (1.5 * 1.0)
        / (1.5f64.powi(2) + 1.5f64.powi(2) + 1.0f64.powi(2));
    let eg_values = array![
        [0.0, f1, f2, f3, f4],     // E
        [0.0, f1, f4, f3, f2],     // C4
        [0.0, -f1, -f4, -f3, -f2], // C4^3
        [0.0, -f1, -f2, -f3, -f4], // C2z
        [0.0, f1, f2, f3, f4],     // C2y
        [0.0, -f1, -f2, -f3, -f4], // C2x
        [0.0, -f1, -f4, -f3, -f2], // C2xy
        [0.0, f1, f4, f3, f2],     // C2xy'
        [0.0, f1, f2, f3, f4],     // i
        [0.0, -f1, -f4, -f3, -f2], // S4
        [0.0, f1, f4, f3, f2],     // S4^3
        [0.0, -f1, -f2, -f3, -f4], // σh
        [0.0, f1, f2, f3, f4],     // σvxz
        [0.0, -f1, -f2, -f3, -f4], // σvyz
        [0.0, -f1, -f4, -f3, -f2], // σd
        [0.0, f1, f4, f3, f2],     // σd'
    ];

    let eg_pes = PES::builder()
        .group(&group_u_d4h)
        .grid_points(grid_points)
        .values(eg_values)
        .build()
        .unwrap();

    let mut orbit_eg_pes = PESSymmetryOrbit::builder()
        .origin(&eg_pes)
        .integrality_threshold(1e-7)
        .linear_independence_threshold(1e-7)
        .symmetry_transformation_kind(SymmetryTransformationKind::Spatial)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
        .build()
        .unwrap();
    let _ = orbit_eg_pes
        .calc_smat(Some(&weight), None, true)
        .unwrap()
        .calc_xmat(false);
    // 2Eg because the grid is too coarse to allow the extra linearly independent components to be
    // removed.
    assert_eq!(
        orbit_eg_pes.analyse_rep().unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new("2||E|_(g)|").unwrap()
    );
}
