use itertools::Itertools;
// use env_logger;
use nalgebra::Point3;
use ndarray::Array1;
use num::ToPrimitive;

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
    let grid_points = (-50..=50).cartesian_product(-50..=50).map(|(i, j)| {
        let x = 0.0 + i.to_f64().unwrap() * 0.02;
        let y = 0.0 + j.to_f64().unwrap() * 0.02;
        Point3::new(x, y, 0.0)
    }).collect_vec();

    let weight: Array1<f64> = Array1::from_iter(
        grid_points
            .iter()
            .map(|pt| (-(pt - Point3::<f64>::origin()).magnitude_squared()).exp()),
    );

    // A1g
    let a1g_pes = PES::builder()
        .function(|pt| (pt - Point3::<f64>::origin()).magnitude_squared())
        .grid_points(grid_points.clone())
        .build()
        .unwrap();

    let mut orbit_a1g_pes = PESSymmetryOrbit::builder()
        .origin(&a1g_pes)
        .group(&group_u_d4h)
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
    let b1g_pes = PES::builder()
        .function(|pt| pt.x.powi(2) - pt.y.powi(2))
        .grid_points(grid_points.clone())
        .build()
        .unwrap();

    let mut orbit_b1g_pes = PESSymmetryOrbit::builder()
        .origin(&b1g_pes)
        .group(&group_u_d4h)
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
    let b2g_pes = PES::builder()
        .function(|pt| pt.x * pt.y)
        .grid_points(grid_points.clone())
        .build()
        .unwrap();

    let mut orbit_b2g_pes = PESSymmetryOrbit::builder()
        .origin(&b2g_pes)
        .group(&group_u_d4h)
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
    let eu_pes = PES::builder()
        .function(|pt| 1.5 * pt.y)
        .grid_points(grid_points)
        .build()
        .unwrap();

    let mut orbit_eu_pes = PESSymmetryOrbit::builder()
        .origin(&eu_pes)
        .group(&group_u_d4h)
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
    //
    // // =========
    // // 3D domain
    // // =========
    let grid_points = (-50..=50).cartesian_product(-50..=50).cartesian_product(-50..=50).map(|((i, j), k)| {
        let x = 0.0 + i.to_f64().unwrap() * 0.02;
        let y = 0.0 + j.to_f64().unwrap() * 0.02;
        let z = 0.0 + k.to_f64().unwrap() * 0.02;
        Point3::new(x, y, z)
    }).collect_vec();

    let weight: Array1<f64> = Array1::from_iter(
        grid_points
            .iter()
            .map(|pt| (-(pt - Point3::<f64>::origin()).magnitude_squared()).exp()),
    );

    // B1u (xyz)
    let b1u_pes = PES::builder()
        .function(|pt| pt.x * pt.y * pt.z)
        .grid_points(grid_points.clone())
        .build()
        .unwrap();

    let mut orbit_b1u_pes = PESSymmetryOrbit::builder()
        .origin(&b1u_pes)
        .group(&group_u_d4h)
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

    let eg_pes = PES::builder()
        .function(|pt| pt.x * pt.z)
        .grid_points(grid_points)
        .build()
        .unwrap();

    let mut orbit_eg_pes = PESSymmetryOrbit::builder()
        .origin(&eg_pes)
        .group(&group_u_d4h)
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
    assert_eq!(
        orbit_eg_pes.analyse_rep().unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new("||E|_(g)|").unwrap()
    );
}
