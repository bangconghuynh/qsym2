// use env_logger;
use itertools::Itertools;
use nalgebra::{Point3, Vector3};
use ndarray::{array, concatenate, s, Array1, Axis};
use ndarray_linalg::assert_close_l2;
use num_complex::Complex;
use serial_test::serial;

use crate::analysis::{EigenvalueComparisonMode, RepAnalysis};
use crate::angmom::spinor_rotation_3d::SpinConstraint;
use crate::auxiliary::atom::{Atom, ElementMap};
use crate::auxiliary::geometry::Transform;
use crate::auxiliary::molecule::Molecule;
use crate::basis::ao::{
    BasisAngularOrder, BasisAtom, BasisShell, CartOrder, PureOrder, ShellOrder,
};
use crate::chartab::chartab_symbols::DecomposedSymbol;
use crate::group::{GroupProperties, MagneticRepresentedGroup, UnitaryRepresentedGroup};
use crate::sandbox::target::pes::pes_analysis::PESSymmetryOrbit;
use crate::sandbox::target::pes::PES;
use crate::symmetry::symmetry_core::{PreSymmetry, Symmetry};
use crate::symmetry::symmetry_group::SymmetryGroupProperties;
use crate::symmetry::symmetry_symbols::{MullikenIrcorepSymbol, MullikenIrrepSymbol};
use crate::symmetry::symmetry_transformation::{
    ComplexConjugationTransformable, SymmetryTransformable, SymmetryTransformationKind,
    TimeReversalTransformable,
};

type C128 = Complex<f64>;

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
    for element in group_u_d4h.elements().iter() {
        println!("{element}");
    }

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
    println!("{weight}");

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

    // Eu
    let eu_values = array![
        [0.0, 1.0, 2.0, 2.0, 2.0], // E
        [0.0, 1.0, 2.0, 2.0, 2.0], // C4
        [0.0, -1.0, -2.0, -2.0, -2.0], // C4^3
        [0.0, -1.0, -2.0, -2.0, -2.0], // C2z
        [0.0, 1.0, 2.0, 2.0, 2.0], // C2y
        [0.0, -1.0, -2.0, -2.0, -2.0], // C2x
        [0.0, 1.0, 2.0, 2.0, 2.0], // C2xy
        [0.0, -1.0, -2.0, -2.0, -2.0], // C2xy'
        [0.0, -1.0, -2.0, -2.0, -2.0], // i
        [0.0, 1.0, 2.0, 2.0, 2.0], // S4
        [0.0, -1.0, -2.0, -2.0, -2.0], // S4^3
        [0.0, 1.0, 2.0, 2.0, 2.0], // σh
        [0.0, -1.0, -2.0, -2.0, -2.0], // σvxz
        [0.0, 1.0, 2.0, 2.0, 2.0], // σvyz
        [0.0, -1.0, -2.0, -2.0, -2.0], // σd
        [0.0, 1.0, 2.0, 2.0, 2.0], // σd'
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
}
//
// #[cfg(feature = "integrals")]
// #[test]
// #[serial]
// fn test_density_orbit_rep_analysis_s4_sqpl_pypz() {
//     use crate::basis::ao_integrals::*;
//     use crate::integrals::shell_tuple::build_shell_tuple_collection;
//
//     // env_logger::init();
//     let emap = ElementMap::new();
//     let atm_s0 = Atom::from_xyz("S +1.0 +1.0 0.0", &emap, 1e-7).unwrap();
//     let atm_s1 = Atom::from_xyz("S -1.0 +1.0 0.0", &emap, 1e-7).unwrap();
//     let atm_s2 = Atom::from_xyz("S -1.0 -1.0 0.0", &emap, 1e-7).unwrap();
//     let atm_s3 = Atom::from_xyz("S +1.0 -1.0 0.0", &emap, 1e-7).unwrap();
//
//     let bsp_p = BasisShell::new(1, ShellOrder::Pure(PureOrder::increasingm(1)));
//
//     let batm_s0 = BasisAtom::new(&atm_s0, &[bsp_p.clone()]);
//     let batm_s1 = BasisAtom::new(&atm_s1, &[bsp_p.clone()]);
//     let batm_s2 = BasisAtom::new(&atm_s2, &[bsp_p.clone()]);
//     let batm_s3 = BasisAtom::new(&atm_s3, &[bsp_p.clone()]);
//
//     let bao_s4 = BasisAngularOrder::new(&[batm_s0, batm_s1, batm_s2, batm_s3]);
//     let mol_s4 = Molecule::from_atoms(
//         &[
//             atm_s0.clone(),
//             atm_s1.clone(),
//             atm_s2.clone(),
//             atm_s3.clone(),
//         ],
//         1e-7,
//     )
//     .recentre();
//
//     let presym = PreSymmetry::builder()
//         .moi_threshold(1e-7)
//         .molecule(&mol_s4)
//         .build()
//         .unwrap();
//     let mut sym = Symmetry::new();
//     sym.analyse(&presym, false).unwrap();
//     let group_u_d4h = UnitaryRepresentedGroup::from_molecular_symmetry(&sym, None).unwrap();
//
//     let mut sym_tr = Symmetry::new();
//     sym_tr.analyse(&presym, true).unwrap();
//     let group_u_grey_d4h = UnitaryRepresentedGroup::from_molecular_symmetry(&sym_tr, None).unwrap();
//     let group_m_grey_d4h =
//         MagneticRepresentedGroup::from_molecular_symmetry(&sym_tr, None).unwrap();
//
//     let mut mol_s4_bz = mol_s4.clone();
//     mol_s4_bz.set_magnetic_field(Some(0.1 * Vector3::z()));
//     let presym_bz = PreSymmetry::builder()
//         .moi_threshold(1e-7)
//         .molecule(&mol_s4_bz)
//         .build()
//         .unwrap();
//
//     let mut sym_bz = Symmetry::new();
//     sym_bz.analyse(&presym_bz, false).unwrap();
//     let group_u_c4h = UnitaryRepresentedGroup::from_molecular_symmetry(&sym_bz, None).unwrap();
//
//     let mut sym_bz_tr = Symmetry::new();
//     sym_bz_tr.analyse(&presym_bz, true).unwrap();
//     let group_u_bw_d4h_c4h =
//         UnitaryRepresentedGroup::from_molecular_symmetry(&sym_bz_tr, None).unwrap();
//     let group_m_bw_d4h_c4h =
//         MagneticRepresentedGroup::from_molecular_symmetry(&sym_bz_tr, None).unwrap();
//
//     // -----------------
//     // Orbital densities
//     // -----------------
//
//     // S0pz
//     #[rustfmt::skip]
//     let calpha = array![
//         [0.0], [1.0], [0.0],
//         [0.0], [0.0], [0.0],
//         [0.0], [0.0], [0.0],
//         [0.0], [0.0], [0.0],
//     ];
//     // S0py
//     #[rustfmt::skip]
//     let cbeta = array![
//         [1.0], [0.0], [0.0],
//         [0.0], [0.0], [0.0],
//         [0.0], [0.0], [0.0],
//         [0.0], [0.0], [0.0],
//     ];
//     let oalpha = array![1.0];
//     let obeta = array![1.0];
//     let det_ru = SlaterDeterminant::<f64>::builder()
//         .coefficients(&[calpha.clone(), cbeta.clone()])
//         .occupations(&[oalpha, obeta])
//         .bao(&bao_s4)
//         .mol(&mol_s4)
//         .spin_constraint(SpinConstraint::Unrestricted(2, false))
//         .complex_symmetric(false)
//         .threshold(1e-14)
//         .build()
//         .unwrap();
//     let dens_ru = det_ru.to_densities().unwrap();
//     let dena_ru = &dens_ru[0];
//     let denb_ru = &dens_ru[1];
//     let dentot_ru = dena_ru + denb_ru;
//     let denspin_ru = dena_ru - denb_ru;
//
//     let det_cu: SlaterDeterminant<C128> = det_ru.clone().into();
//     let dens_cu = det_cu.to_densities().unwrap();
//     let dena_cu = &dens_cu[0];
//     let denb_cu = &dens_cu[1];
//     let dentot_cu = dena_cu + denb_cu;
//     let denspin_cu = dena_cu - denb_cu;
//
//     // ------
//     // Metric
//     // ------
//
//     let gc = GaussianContraction::<f64, f64> {
//         primitives: vec![(3.4252509140, 1.0)],
//     };
//     let bsc0 = BasisShellContraction::<f64, f64> {
//         basis_shell: bsp_p.clone(),
//         contraction: gc.clone(),
//         cart_origin: Point3::new(1.0, 1.0, 0.0),
//         k: None,
//     };
//     let bsc1 = BasisShellContraction::<f64, f64> {
//         basis_shell: bsp_p.clone(),
//         contraction: gc.clone(),
//         cart_origin: Point3::new(-1.0, 1.0, 0.0),
//         k: None,
//     };
//     let bsc2 = BasisShellContraction::<f64, f64> {
//         basis_shell: bsp_p.clone(),
//         contraction: gc.clone(),
//         cart_origin: Point3::new(-1.0, -1.0, 0.0),
//         k: None,
//     };
//     let bsc3 = BasisShellContraction::<f64, f64> {
//         basis_shell: bsp_p.clone(),
//         contraction: gc.clone(),
//         cart_origin: Point3::new(1.0, -1.0, 0.0),
//         k: None,
//     };
//     let bscs = BasisSet::new(vec![vec![bsc0], vec![bsc1], vec![bsc2], vec![bsc3]]);
//     let stc = build_shell_tuple_collection![
//         <s1, s2, s3, s4>;
//         false, false, false, false;
//         &bscs, &bscs, &bscs, &bscs;
//         f64
//     ];
//     let ovs = stc.overlap([0, 0, 0, 0]);
//     let sao_ru = &ovs[0];
//
//     let mut bscs_bz = bscs.clone();
//     bscs_bz.apply_magnetic_field(&(0.1 * Vector3::z()), &Point3::origin());
//     let stc_bz = build_shell_tuple_collection![
//         <s1, s2, s3, s4>;
//         false, false, false, false;
//         &bscs, &bscs, &bscs, &bscs;
//         C128
//     ];
//     let ovs_bz = stc_bz.overlap([0, 0, 0, 0]);
//     let sao_cu_bz = &ovs_bz[0];
//
//     // ~~~~~~~~~~~~~~~~~~~~~~~~~
//     // u D4h (ordinary, unitary)
//     // ~~~~~~~~~~~~~~~~~~~~~~~~~
//
//     let mut orbit_ru_u_d4h_spatial_orbital_density_a = DensitySymmetryOrbit::builder()
//         .group(&group_u_d4h)
//         .origin(&dena_ru)
//         .integrality_threshold(1e-14)
//         .linear_independence_threshold(1e-14)
//         .symmetry_transformation_kind(SymmetryTransformationKind::Spatial)
//         .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
//         .build()
//         .unwrap();
//     let _ = orbit_ru_u_d4h_spatial_orbital_density_a
//         .calc_smat(Some(sao_ru), None, true)
//         .unwrap()
//         .calc_xmat(false);
//     assert_eq!(
//         orbit_ru_u_d4h_spatial_orbital_density_a
//             .analyse_rep()
//             .unwrap(),
//         DecomposedSymbol::<MullikenIrrepSymbol>::new("||A|_(1g)| ⊕ ||B|_(2g)| ⊕ ||E|_(u)|")
//             .unwrap()
//     );
//
//     let mut orbit_ru_u_d4h_spatial_orbital_density_b = DensitySymmetryOrbit::builder()
//         .group(&group_u_d4h)
//         .origin(&denb_ru)
//         .integrality_threshold(1e-14)
//         .linear_independence_threshold(1e-14)
//         .symmetry_transformation_kind(SymmetryTransformationKind::Spatial)
//         .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
//         .build()
//         .unwrap();
//     let _ = orbit_ru_u_d4h_spatial_orbital_density_b
//         .calc_smat(Some(sao_ru), None, true)
//         .unwrap()
//         .calc_xmat(false);
//     assert_eq!(
//         orbit_ru_u_d4h_spatial_orbital_density_b
//             .analyse_rep()
//             .unwrap(),
//         DecomposedSymbol::<MullikenIrrepSymbol>::new(
//             "||A|_(1g)| ⊕ ||A|_(2g)| ⊕ ||B|_(1g)| ⊕ ||B|_(2g)| ⊕ 2||E|_(u)|"
//         )
//         .unwrap()
//     );
//
//     let mut orbit_ru_u_d4h_spatial_orbital_density_total = DensitySymmetryOrbit::builder()
//         .group(&group_u_d4h)
//         .origin(&dentot_ru)
//         .integrality_threshold(1e-14)
//         .linear_independence_threshold(1e-14)
//         .symmetry_transformation_kind(SymmetryTransformationKind::Spatial)
//         .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
//         .build()
//         .unwrap();
//     let _ = orbit_ru_u_d4h_spatial_orbital_density_total
//         .calc_smat(Some(sao_ru), None, true)
//         .unwrap()
//         .calc_xmat(false);
//     assert_eq!(
//         orbit_ru_u_d4h_spatial_orbital_density_total
//             .analyse_rep()
//             .unwrap(),
//         DecomposedSymbol::<MullikenIrrepSymbol>::new(
//             "||A|_(1g)| ⊕ ||A|_(2g)| ⊕ ||B|_(1g)| ⊕ ||B|_(2g)| ⊕ 2||E|_(u)|"
//         )
//         .unwrap()
//     );
//
//     let mut orbit_ru_u_d4h_spatial_orbital_density_spin = DensitySymmetryOrbit::builder()
//         .group(&group_u_d4h)
//         .origin(&denspin_ru)
//         .integrality_threshold(1e-14)
//         .linear_independence_threshold(1e-14)
//         .symmetry_transformation_kind(SymmetryTransformationKind::Spatial)
//         .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
//         .build()
//         .unwrap();
//     let _ = orbit_ru_u_d4h_spatial_orbital_density_spin
//         .calc_smat(Some(sao_ru), None, true)
//         .unwrap()
//         .calc_xmat(false);
//     assert_eq!(
//         orbit_ru_u_d4h_spatial_orbital_density_spin
//             .analyse_rep()
//             .unwrap(),
//         DecomposedSymbol::<MullikenIrrepSymbol>::new(
//             "||A|_(1g)| ⊕ ||A|_(2g)| ⊕ ||B|_(1g)| ⊕ ||B|_(2g)| ⊕ 2||E|_(u)|"
//         )
//         .unwrap()
//     );
//
//     // ~~~~~~~~~~~~~~~~~~~~~~
//     // u D4h' (grey, unitary)
//     // ~~~~~~~~~~~~~~~~~~~~~~
//
//     let mut orbit_ru_u_grey_d4h_spatial_orbital_density_a = DensitySymmetryOrbit::builder()
//         .group(&group_u_grey_d4h)
//         .origin(&dena_ru)
//         .integrality_threshold(1e-14)
//         .linear_independence_threshold(1e-14)
//         .symmetry_transformation_kind(SymmetryTransformationKind::Spatial)
//         .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
//         .build()
//         .unwrap();
//     let _ = orbit_ru_u_grey_d4h_spatial_orbital_density_a
//         .calc_smat(Some(sao_ru), None, true)
//         .unwrap()
//         .calc_xmat(false);
//     assert_eq!(
//         orbit_ru_u_grey_d4h_spatial_orbital_density_a
//             .analyse_rep()
//             .unwrap(),
//         DecomposedSymbol::<MullikenIrrepSymbol>::new(
//             "|^(+)|A|_(1g)| ⊕ |^(+)|B|_(2g)| ⊕ |^(+)|E|_(u)|"
//         )
//         .unwrap()
//     );
//
//     let mut orbit_ru_u_grey_d4h_spatial_orbital_density_b = DensitySymmetryOrbit::builder()
//         .group(&group_u_grey_d4h)
//         .origin(&denb_ru)
//         .integrality_threshold(1e-14)
//         .linear_independence_threshold(1e-14)
//         .symmetry_transformation_kind(SymmetryTransformationKind::Spatial)
//         .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
//         .build()
//         .unwrap();
//     let _ = orbit_ru_u_grey_d4h_spatial_orbital_density_b
//         .calc_smat(Some(sao_ru), None, true)
//         .unwrap()
//         .calc_xmat(false);
//     assert_eq!(
//         orbit_ru_u_grey_d4h_spatial_orbital_density_b
//             .analyse_rep()
//             .unwrap(),
//         DecomposedSymbol::<MullikenIrrepSymbol>::new(
//             "|^(+)|A|_(1g)| ⊕ |^(+)|A|_(2g)| ⊕ |^(+)|B|_(1g)| ⊕ |^(+)|B|_(2g)| ⊕ 2|^(+)|E|_(u)|"
//         )
//         .unwrap()
//     );
//
//     let mut orbit_ru_u_grey_d4h_spatial_orbital_density_total = DensitySymmetryOrbit::builder()
//         .group(&group_u_grey_d4h)
//         .origin(&dentot_ru)
//         .integrality_threshold(1e-14)
//         .linear_independence_threshold(1e-14)
//         .symmetry_transformation_kind(SymmetryTransformationKind::Spatial)
//         .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
//         .build()
//         .unwrap();
//     let _ = orbit_ru_u_grey_d4h_spatial_orbital_density_total
//         .calc_smat(Some(sao_ru), None, true)
//         .unwrap()
//         .calc_xmat(false);
//     assert_eq!(
//         orbit_ru_u_grey_d4h_spatial_orbital_density_total
//             .analyse_rep()
//             .unwrap(),
//         DecomposedSymbol::<MullikenIrrepSymbol>::new(
//             "|^(+)|A|_(1g)| ⊕ |^(+)|A|_(2g)| ⊕ |^(+)|B|_(1g)| ⊕ |^(+)|B|_(2g)| ⊕ 2|^(+)|E|_(u)|"
//         )
//         .unwrap()
//     );
//
//     let mut orbit_ru_u_grey_d4h_spatial_orbital_density_spin = DensitySymmetryOrbit::builder()
//         .group(&group_u_grey_d4h)
//         .origin(&denspin_ru)
//         .integrality_threshold(1e-14)
//         .linear_independence_threshold(1e-14)
//         .symmetry_transformation_kind(SymmetryTransformationKind::Spatial)
//         .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
//         .build()
//         .unwrap();
//     let _ = orbit_ru_u_grey_d4h_spatial_orbital_density_spin
//         .calc_smat(Some(sao_ru), None, true)
//         .unwrap()
//         .calc_xmat(false);
//     assert_eq!(
//         orbit_ru_u_grey_d4h_spatial_orbital_density_spin
//             .analyse_rep()
//             .unwrap(),
//         DecomposedSymbol::<MullikenIrrepSymbol>::new(
//             "|^(+)|A|_(1g)| ⊕ |^(+)|A|_(2g)| ⊕ |^(+)|B|_(1g)| ⊕ |^(+)|B|_(2g)| ⊕ 2|^(+)|E|_(u)|"
//         )
//         .unwrap()
//     );
//
//     // ~~~~~~~~~~~~~~~~~~~~~~~
//     // m D4h' (grey, magnetic)
//     // ~~~~~~~~~~~~~~~~~~~~~~~
//
//     let mut orbit_ru_m_grey_d4h_spatial_orbital_density_a = DensitySymmetryOrbit::builder()
//         .group(&group_m_grey_d4h)
//         .origin(&dena_ru)
//         .integrality_threshold(1e-14)
//         .linear_independence_threshold(1e-14)
//         .symmetry_transformation_kind(SymmetryTransformationKind::Spatial)
//         .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
//         .build()
//         .unwrap();
//     let _ = orbit_ru_m_grey_d4h_spatial_orbital_density_a
//         .calc_smat(Some(sao_ru), None, true)
//         .unwrap()
//         .calc_xmat(false);
//     assert_eq!(
//         orbit_ru_m_grey_d4h_spatial_orbital_density_a
//             .analyse_rep()
//             .unwrap(),
//         DecomposedSymbol::<MullikenIrcorepSymbol>::new("||A|_(1g)| ⊕ ||B|_(2g)| ⊕ ||E|_(u)|")
//             .unwrap()
//     );
//
//     let mut orbit_ru_m_grey_d4h_spatial_orbital_density_b = DensitySymmetryOrbit::builder()
//         .group(&group_m_grey_d4h)
//         .origin(&denb_ru)
//         .integrality_threshold(1e-14)
//         .linear_independence_threshold(1e-14)
//         .symmetry_transformation_kind(SymmetryTransformationKind::Spatial)
//         .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
//         .build()
//         .unwrap();
//     let _ = orbit_ru_m_grey_d4h_spatial_orbital_density_b
//         .calc_smat(Some(sao_ru), None, true)
//         .unwrap()
//         .calc_xmat(false);
//     assert_eq!(
//         orbit_ru_m_grey_d4h_spatial_orbital_density_b
//             .analyse_rep()
//             .unwrap(),
//         DecomposedSymbol::<MullikenIrcorepSymbol>::new(
//             "||A|_(1g)| ⊕ ||A|_(2g)| ⊕ ||B|_(1g)| ⊕ ||B|_(2g)| ⊕ 2||E|_(u)|"
//         )
//         .unwrap()
//     );
//
//     let mut orbit_ru_m_grey_d4h_spatial_orbital_density_total = DensitySymmetryOrbit::builder()
//         .group(&group_m_grey_d4h)
//         .origin(&dentot_ru)
//         .integrality_threshold(1e-14)
//         .linear_independence_threshold(1e-14)
//         .symmetry_transformation_kind(SymmetryTransformationKind::Spatial)
//         .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
//         .build()
//         .unwrap();
//     let _ = orbit_ru_m_grey_d4h_spatial_orbital_density_total
//         .calc_smat(Some(sao_ru), None, true)
//         .unwrap()
//         .calc_xmat(false);
//     assert_eq!(
//         orbit_ru_m_grey_d4h_spatial_orbital_density_total
//             .analyse_rep()
//             .unwrap(),
//         DecomposedSymbol::<MullikenIrcorepSymbol>::new(
//             "||A|_(1g)| ⊕ ||A|_(2g)| ⊕ ||B|_(1g)| ⊕ ||B|_(2g)| ⊕ 2||E|_(u)|"
//         )
//         .unwrap()
//     );
//
//     let mut orbit_ru_m_grey_d4h_spatial_orbital_density_spin = DensitySymmetryOrbit::builder()
//         .group(&group_m_grey_d4h)
//         .origin(&denspin_ru)
//         .integrality_threshold(1e-14)
//         .linear_independence_threshold(1e-14)
//         .symmetry_transformation_kind(SymmetryTransformationKind::Spatial)
//         .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
//         .build()
//         .unwrap();
//     let _ = orbit_ru_m_grey_d4h_spatial_orbital_density_spin
//         .calc_smat(Some(sao_ru), None, true)
//         .unwrap()
//         .calc_xmat(false);
//     assert_eq!(
//         orbit_ru_m_grey_d4h_spatial_orbital_density_spin
//             .analyse_rep()
//             .unwrap(),
//         DecomposedSymbol::<MullikenIrcorepSymbol>::new(
//             "||A|_(1g)| ⊕ ||A|_(2g)| ⊕ ||B|_(1g)| ⊕ ||B|_(2g)| ⊕ 2||E|_(u)|"
//         )
//         .unwrap()
//     );
//
//     // ~~~~~~~~~~~~~~~~~~~~~~~~~
//     // u C4h (ordinary, unitary)
//     // ~~~~~~~~~~~~~~~~~~~~~~~~~
//
//     let mut orbit_cu_u_c4h_spatial_orbital_density_a = DensitySymmetryOrbit::builder()
//         .group(&group_u_c4h)
//         .origin(&dena_cu)
//         .integrality_threshold(1e-14)
//         .linear_independence_threshold(1e-14)
//         .symmetry_transformation_kind(SymmetryTransformationKind::Spatial)
//         .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
//         .build()
//         .unwrap();
//     let _ = orbit_cu_u_c4h_spatial_orbital_density_a
//         .calc_smat(Some(sao_cu_bz), None, true)
//         .unwrap()
//         .calc_xmat(false);
//     assert_eq!(
//         orbit_cu_u_c4h_spatial_orbital_density_a
//             .analyse_rep()
//             .unwrap(),
//         DecomposedSymbol::<MullikenIrrepSymbol>::new(
//             "||A|_(g)| ⊕ ||B|_(g)| ⊕ |_(a)|Γ|_(u)| ⊕ |_(b)|Γ|_(u)|"
//         )
//         .unwrap()
//     );
//
//     let mut orbit_cu_u_c4h_spatial_orbital_density_b = DensitySymmetryOrbit::builder()
//         .group(&group_u_c4h)
//         .origin(&denb_cu)
//         .integrality_threshold(1e-14)
//         .linear_independence_threshold(1e-14)
//         .symmetry_transformation_kind(SymmetryTransformationKind::Spatial)
//         .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
//         .build()
//         .unwrap();
//     let _ = orbit_cu_u_c4h_spatial_orbital_density_b
//         .calc_smat(Some(sao_cu_bz), None, true)
//         .unwrap()
//         .calc_xmat(false);
//     assert_eq!(
//         orbit_cu_u_c4h_spatial_orbital_density_b
//             .analyse_rep()
//             .unwrap(),
//         DecomposedSymbol::<MullikenIrrepSymbol>::new(
//             "||A|_(g)| ⊕ ||B|_(g)| ⊕ |_(a)|Γ|_(u)| ⊕ |_(b)|Γ|_(u)|"
//         )
//         .unwrap()
//     );
//
//     let mut orbit_cu_u_c4h_spatial_orbital_density_total = DensitySymmetryOrbit::builder()
//         .group(&group_u_c4h)
//         .origin(&dentot_cu)
//         .integrality_threshold(1e-14)
//         .linear_independence_threshold(1e-14)
//         .symmetry_transformation_kind(SymmetryTransformationKind::Spatial)
//         .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
//         .build()
//         .unwrap();
//     let _ = orbit_cu_u_c4h_spatial_orbital_density_total
//         .calc_smat(Some(sao_cu_bz), None, true)
//         .unwrap()
//         .calc_xmat(false);
//     assert_eq!(
//         orbit_cu_u_c4h_spatial_orbital_density_total
//             .analyse_rep()
//             .unwrap(),
//         DecomposedSymbol::<MullikenIrrepSymbol>::new(
//             "||A|_(g)| ⊕ ||B|_(g)| ⊕ |_(a)|Γ|_(u)| ⊕ |_(b)|Γ|_(u)|"
//         )
//         .unwrap()
//     );
//
//     let mut orbit_cu_u_c4h_spatial_orbital_density_spin = DensitySymmetryOrbit::builder()
//         .group(&group_u_c4h)
//         .origin(&denspin_cu)
//         .integrality_threshold(1e-14)
//         .linear_independence_threshold(1e-14)
//         .symmetry_transformation_kind(SymmetryTransformationKind::Spatial)
//         .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
//         .build()
//         .unwrap();
//     let _ = orbit_cu_u_c4h_spatial_orbital_density_spin
//         .calc_smat(Some(sao_cu_bz), None, true)
//         .unwrap()
//         .calc_xmat(false);
//     assert_eq!(
//         orbit_cu_u_c4h_spatial_orbital_density_spin
//             .analyse_rep()
//             .unwrap(),
//         DecomposedSymbol::<MullikenIrrepSymbol>::new(
//             "||A|_(g)| ⊕ ||B|_(g)| ⊕ |_(a)|Γ|_(u)| ⊕ |_(b)|Γ|_(u)|"
//         )
//         .unwrap()
//     );
//
//     // ~~~~~~~~~~~~~~~~~~~~~~~~
//     // u D4h(C4h) (bw, unitary)
//     // ~~~~~~~~~~~~~~~~~~~~~~~~
//
//     let mut orbit_cu_u_bw_d4h_c4h_spatial_orbital_density_a = DensitySymmetryOrbit::builder()
//         .group(&group_u_bw_d4h_c4h)
//         .origin(&dena_cu)
//         .integrality_threshold(1e-14)
//         .linear_independence_threshold(1e-14)
//         .symmetry_transformation_kind(SymmetryTransformationKind::Spatial)
//         .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
//         .build()
//         .unwrap();
//     let _ = orbit_cu_u_bw_d4h_c4h_spatial_orbital_density_a
//         .calc_smat(Some(sao_cu_bz), None, true)
//         .unwrap()
//         .calc_xmat(false);
//     assert_eq!(
//         orbit_cu_u_bw_d4h_c4h_spatial_orbital_density_a
//             .analyse_rep()
//             .unwrap(),
//         DecomposedSymbol::<MullikenIrrepSymbol>::new("||A|_(1g)| ⊕ ||B|_(2g)| ⊕ ||E|_(u)|")
//             .unwrap()
//     );
//
//     let mut orbit_cu_u_bw_d4h_c4h_spatial_orbital_density_b = DensitySymmetryOrbit::builder()
//         .group(&group_u_bw_d4h_c4h)
//         .origin(&denb_cu)
//         .integrality_threshold(1e-14)
//         .linear_independence_threshold(1e-14)
//         .symmetry_transformation_kind(SymmetryTransformationKind::Spatial)
//         .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
//         .build()
//         .unwrap();
//     let _ = orbit_cu_u_bw_d4h_c4h_spatial_orbital_density_b
//         .calc_smat(Some(sao_cu_bz), None, true)
//         .unwrap()
//         .calc_xmat(false);
//     assert_eq!(
//         orbit_cu_u_bw_d4h_c4h_spatial_orbital_density_b
//             .analyse_rep()
//             .unwrap(),
//         DecomposedSymbol::<MullikenIrrepSymbol>::new(
//             "||A|_(1g)| ⊕ ||A|_(2g)| ⊕ ||B|_(1g)| ⊕ ||B|_(2g)| ⊕ 2||E|_(u)|"
//         )
//         .unwrap()
//     );
//
//     let mut orbit_cu_u_bw_d4h_c4h_spatial_orbital_density_total = DensitySymmetryOrbit::builder()
//         .group(&group_u_bw_d4h_c4h)
//         .origin(&dentot_cu)
//         .integrality_threshold(1e-14)
//         .linear_independence_threshold(1e-14)
//         .symmetry_transformation_kind(SymmetryTransformationKind::Spatial)
//         .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
//         .build()
//         .unwrap();
//     let _ = orbit_cu_u_bw_d4h_c4h_spatial_orbital_density_total
//         .calc_smat(Some(sao_cu_bz), None, true)
//         .unwrap()
//         .calc_xmat(false);
//     assert_eq!(
//         orbit_cu_u_bw_d4h_c4h_spatial_orbital_density_total
//             .analyse_rep()
//             .unwrap(),
//         DecomposedSymbol::<MullikenIrrepSymbol>::new(
//             "||A|_(1g)| ⊕ ||A|_(2g)| ⊕ ||B|_(1g)| ⊕ ||B|_(2g)| ⊕ 2||E|_(u)|"
//         )
//         .unwrap()
//     );
//
//     let mut orbit_cu_u_bw_d4h_c4h_spatial_orbital_density_spin = DensitySymmetryOrbit::builder()
//         .group(&group_u_bw_d4h_c4h)
//         .origin(&denspin_cu)
//         .integrality_threshold(1e-14)
//         .linear_independence_threshold(1e-14)
//         .symmetry_transformation_kind(SymmetryTransformationKind::Spatial)
//         .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
//         .build()
//         .unwrap();
//     let _ = orbit_cu_u_bw_d4h_c4h_spatial_orbital_density_spin
//         .calc_smat(Some(sao_cu_bz), None, true)
//         .unwrap()
//         .calc_xmat(false);
//     assert_eq!(
//         orbit_cu_u_bw_d4h_c4h_spatial_orbital_density_spin
//             .analyse_rep()
//             .unwrap(),
//         DecomposedSymbol::<MullikenIrrepSymbol>::new(
//             "||A|_(1g)| ⊕ ||A|_(2g)| ⊕ ||B|_(1g)| ⊕ ||B|_(2g)| ⊕ 2||E|_(u)|"
//         )
//         .unwrap()
//     );
//
//     // ~~~~~~~~~~~~~~~~~~~~~~~~~
//     // m D4h(C4h) (bw, magnetic)
//     // ~~~~~~~~~~~~~~~~~~~~~~~~~
//
//     let mut orbit_cu_m_bw_d4h_c4h_spatial_orbital_density_a = DensitySymmetryOrbit::builder()
//         .group(&group_m_bw_d4h_c4h)
//         .origin(&dena_cu)
//         .integrality_threshold(1e-14)
//         .linear_independence_threshold(1e-14)
//         .symmetry_transformation_kind(SymmetryTransformationKind::Spatial)
//         .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
//         .build()
//         .unwrap();
//     let _ = orbit_cu_m_bw_d4h_c4h_spatial_orbital_density_a
//         .calc_smat(Some(sao_cu_bz), None, true)
//         .unwrap()
//         .calc_xmat(false);
//     assert_eq!(
//         orbit_cu_m_bw_d4h_c4h_spatial_orbital_density_a
//             .analyse_rep()
//             .unwrap(),
//         DecomposedSymbol::<MullikenIrcorepSymbol>::new(
//             "||A|_(g)| ⊕ ||B|_(g)| ⊕ |_(a)|Γ|_(u)| ⊕ |_(b)|Γ|_(u)|"
//         )
//         .unwrap()
//     );
//
//     let mut orbit_cu_m_bw_d4h_c4h_spatial_orbital_density_b = DensitySymmetryOrbit::builder()
//         .group(&group_m_bw_d4h_c4h)
//         .origin(&denb_cu)
//         .integrality_threshold(1e-14)
//         .linear_independence_threshold(1e-14)
//         .symmetry_transformation_kind(SymmetryTransformationKind::Spatial)
//         .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
//         .build()
//         .unwrap();
//     let _ = orbit_cu_m_bw_d4h_c4h_spatial_orbital_density_b
//         .calc_smat(Some(sao_cu_bz), None, true)
//         .unwrap()
//         .calc_xmat(false);
//     assert_eq!(
//         orbit_cu_m_bw_d4h_c4h_spatial_orbital_density_b
//             .analyse_rep()
//             .unwrap(),
//         DecomposedSymbol::<MullikenIrcorepSymbol>::new(
//             "2||A|_(g)| ⊕ 2||B|_(g)| ⊕ 2|_(a)|Γ|_(u)| ⊕ 2|_(b)|Γ|_(u)|"
//         )
//         .unwrap()
//     );
//
//     let mut orbit_cu_m_bw_d4h_c4h_spatial_orbital_density_total = DensitySymmetryOrbit::builder()
//         .group(&group_m_bw_d4h_c4h)
//         .origin(&dentot_cu)
//         .integrality_threshold(1e-14)
//         .linear_independence_threshold(1e-14)
//         .symmetry_transformation_kind(SymmetryTransformationKind::Spatial)
//         .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
//         .build()
//         .unwrap();
//     let _ = orbit_cu_m_bw_d4h_c4h_spatial_orbital_density_total
//         .calc_smat(Some(sao_cu_bz), None, true)
//         .unwrap()
//         .calc_xmat(false);
//     assert_eq!(
//         orbit_cu_m_bw_d4h_c4h_spatial_orbital_density_total
//             .analyse_rep()
//             .unwrap(),
//         DecomposedSymbol::<MullikenIrcorepSymbol>::new(
//             "2||A|_(g)| ⊕ 2||B|_(g)| ⊕ 2|_(a)|Γ|_(u)| ⊕ 2|_(b)|Γ|_(u)|"
//         )
//         .unwrap()
//     );
//
//     let mut orbit_cu_m_bw_d4h_c4h_spatial_orbital_density_spin = DensitySymmetryOrbit::builder()
//         .group(&group_m_bw_d4h_c4h)
//         .origin(&denspin_cu)
//         .integrality_threshold(1e-14)
//         .linear_independence_threshold(1e-14)
//         .symmetry_transformation_kind(SymmetryTransformationKind::Spatial)
//         .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
//         .build()
//         .unwrap();
//     let _ = orbit_cu_m_bw_d4h_c4h_spatial_orbital_density_spin
//         .calc_smat(Some(sao_cu_bz), None, true)
//         .unwrap()
//         .calc_xmat(false);
//     assert_eq!(
//         orbit_cu_m_bw_d4h_c4h_spatial_orbital_density_spin
//             .analyse_rep()
//             .unwrap(),
//         DecomposedSymbol::<MullikenIrcorepSymbol>::new(
//             "2||A|_(g)| ⊕ 2||B|_(g)| ⊕ 2|_(a)|Γ|_(u)| ⊕ 2|_(b)|Γ|_(u)|"
//         )
//         .unwrap()
//     );
// }
