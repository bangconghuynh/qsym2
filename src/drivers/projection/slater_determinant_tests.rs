use std::str::FromStr;

use approx::{assert_abs_diff_eq, assert_abs_diff_ne};
use itertools::Itertools;
// use log4rs;
use nalgebra::ComplexField;
use ndarray::array;
use ndarray_linalg::{Norm, Trace};
use num_complex::Complex;

use crate::analysis::{EigenvalueComparisonMode, Overlap, RepAnalysis};
use crate::angmom::spinor_rotation_3d::SpinConstraint;
use crate::basis::ao::{
    BasisAngularOrder, BasisAtom, BasisShell, CartOrder, PureOrder, ShellOrder,
};
use crate::chartab::chartab_symbols::{DecomposedSymbol, ReducibleLinearSpaceSymbol};
use crate::drivers::QSym2Driver;
use crate::drivers::projection::slater_determinant::{
    SlaterDeterminantProjectionDriver, SlaterDeterminantProjectionParams,
};
use crate::drivers::representation_analysis::{
    CharacterTableDisplay, MagneticSymmetryAnalysisKind,
};
use crate::drivers::symmetry_group_detection::{
    SymmetryGroupDetectionDriver, SymmetryGroupDetectionParams,
};
use crate::symmetry::symmetry_group::UnitaryRepresentedSymmetryGroup;
use crate::symmetry::symmetry_symbols::MullikenIrrepSymbol;
use crate::symmetry::symmetry_transformation::SymmetryTransformationKind;
use crate::target::determinant::SlaterDeterminant;
use crate::target::noci::multideterminant::multideterminant_analysis::MultiDeterminantSymmetryOrbit;

type C128 = Complex<f64>;

const ROOT: &str = env!("CARGO_MANIFEST_DIR");

#[test]
fn test_drivers_slater_determinant_projection_vf6() {
    // log4rs::init_file("log4rs.yml", Default::default()).unwrap();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/vf6.xyz");

    let pd_params = SymmetryGroupDetectionParams::builder()
        .moi_thresholds(&[1e-6])
        .distance_thresholds(&[1e-6])
        .field_origin_com(true)
        .time_reversal(true)
        .write_symmetry_elements(true)
        .build()
        .unwrap();
    let mut pd_driver = SymmetryGroupDetectionDriver::builder()
        .parameters(&pd_params)
        .xyz(Some(path.into()))
        .build()
        .unwrap();
    assert!(pd_driver.run().is_ok());
    let pd_res = pd_driver.result().unwrap();
    let mol_vf6 = &pd_res.pre_symmetry.recentred_molecule;

    let bsc_d = BasisShell::new(2, ShellOrder::Cart(CartOrder::lex(2)));
    let bsp_s = BasisShell::new(0, ShellOrder::Pure(PureOrder::increasingm(0)));

    let batm_v = BasisAtom::new(&mol_vf6.atoms[0], &[bsc_d]);
    let batm_f0 = BasisAtom::new(&mol_vf6.atoms[1], &[bsp_s.clone()]);
    let batm_f1 = BasisAtom::new(&mol_vf6.atoms[2], &[bsp_s.clone()]);
    let batm_f2 = BasisAtom::new(&mol_vf6.atoms[3], &[bsp_s.clone()]);
    let batm_f3 = BasisAtom::new(&mol_vf6.atoms[4], &[bsp_s.clone()]);
    let batm_f4 = BasisAtom::new(&mol_vf6.atoms[5], &[bsp_s.clone()]);
    let batm_f5 = BasisAtom::new(&mol_vf6.atoms[6], &[bsp_s]);

    let bao_vf6 =
        BasisAngularOrder::new(&[batm_v, batm_f0, batm_f1, batm_f2, batm_f3, batm_f4, batm_f5]);

    let thr = 1.0 / 3.0;
    let sao_spatial = array![
        [1.0, 0.0, 0.0, thr, 0.0, thr, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [thr, 0.0, 0.0, 1.0, 0.0, thr, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [thr, 0.0, 0.0, thr, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    ]
    .mapv(C128::from);

    // =====================================
    // αdxy αdyy αdzz αdx2-y2 βdxz βdxx βdyz
    // =====================================
    #[rustfmt::skip]
    let calpha = array![
        [0.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, -1.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ];
    #[rustfmt::skip]
    let cbeta = array![
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ];
    let oalpha = array![1.0, 1.0, 0.0, 0.0];
    let obeta = array![1.0, 0.0, 0.0];
    let det_d3_cg: SlaterDeterminant<C128, SpinConstraint> =
        SlaterDeterminant::<f64, SpinConstraint>::builder()
            .coefficients(&[calpha, cbeta])
            .occupations(&[oalpha, obeta])
            .baos(vec![&bao_vf6])
            .mol(mol_vf6)
            .structure_constraint(SpinConstraint::Unrestricted(2, false))
            .complex_symmetric(false)
            .threshold(1e-14)
            .build()
            .unwrap()
            .to_generalised()
            .into();

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // u Oh (ordinary, unitary) - spin spatial
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // ||T|_(1g)| ⊕ ||T|_(2g)|
    let sdp_params = SlaterDeterminantProjectionParams::builder()
        .numeric_projection_targets(Some(vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))
        .use_magnetic_group(None)
        .use_double_group(false)
        .symmetry_transformation_kind(SymmetryTransformationKind::SpinSpatial)
        .write_character_table(Some(CharacterTableDisplay::Symbolic))
        .build()
        .unwrap();

    let mut sdp_driver = SlaterDeterminantProjectionDriver::<
        UnitaryRepresentedSymmetryGroup,
        C128,
        SpinConstraint,
    >::builder()
    .parameters(&sdp_params)
    .determinant(&det_d3_cg)
    .sao(Some(&sao_spatial))
    .symmetry_group(pd_res)
    .build()
    .unwrap();
    assert!(sdp_driver.run().is_ok());

    let group = sdp_driver.construct_unitary_group().unwrap();
    let sao = sdp_driver.construct_sao().unwrap().0.unwrap();
    for sym in ["||T|_(1g)|", "||T|_(2g)|"].iter() {
        let row = MullikenIrrepSymbol::from_str(sym).unwrap();
        let dyy_p = sdp_driver
            .result()
            .as_ref()
            .unwrap()
            .projected_determinants()
            .get(&row)
            .unwrap()
            .as_ref()
            .unwrap();
        let norm_sq = dyy_p.overlap(&dyy_p, Some(&sao), None).unwrap();
        let mixed_denmat = dyy_p
            .density_matrix(&sao.view(), 1e-7, 1e-7)
            .unwrap()
            .dot(&sao)
            / norm_sq;
        assert_abs_diff_ne!(mixed_denmat.norm_l2(), 0.0, epsilon = 1e-7);
        assert_abs_diff_eq!(mixed_denmat.trace().unwrap().abs(), 3.0, epsilon = 1e-7);

        let mut orbit_dyy_p = MultiDeterminantSymmetryOrbit::builder()
            .group(&group)
            .origin(&dyy_p)
            .integrality_threshold(1e-7)
            .linear_independence_threshold(1e-7)
            .symmetry_transformation_kind(SymmetryTransformationKind::SpinSpatial)
            .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
            .build()
            .unwrap();
        let _ = orbit_dyy_p
            .calc_smat(Some(&sao), None, true)
            .unwrap()
            .calc_xmat(false);
        assert_eq!(
            orbit_dyy_p.analyse_rep().unwrap(),
            DecomposedSymbol::<MullikenIrrepSymbol>::new(sym).unwrap()
        );
    }

    for sym in [
        "||A|_(1g)|",
        "||A|_(2g)|",
        "||E|_(g)|",
        "||A|_(1u)|",
        "||A|_(2u)|",
        "||E|_(u)|",
        "||T|_(1u)|",
        "||T|_(2u)|",
    ]
    .iter()
    {
        let row = MullikenIrrepSymbol::from_str(sym).unwrap();
        let dyy_p = sdp_driver
            .result()
            .as_ref()
            .unwrap()
            .projected_determinants()
            .get(&row)
            .unwrap()
            .as_ref()
            .unwrap();
        let norm_sq = dyy_p.overlap(&dyy_p, Some(&sao), None).unwrap();
        assert_abs_diff_eq!(norm_sq.abs(), 0.0, epsilon = 1e-7);
    }

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // u Oh* (ordinary double, unitary) - spin spatial
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // ||E~|_(1g)| ⊕ ||E~|_(2g)| ⊕ 2||F~|_(g)|
    let sdp_params = SlaterDeterminantProjectionParams::builder()
        .numeric_projection_targets(Some((0..=15).collect_vec()))
        .use_magnetic_group(None)
        .use_double_group(true)
        .symmetry_transformation_kind(SymmetryTransformationKind::SpinSpatial)
        .write_character_table(Some(CharacterTableDisplay::Symbolic))
        .build()
        .unwrap();

    let mut sdp_driver = SlaterDeterminantProjectionDriver::<
        UnitaryRepresentedSymmetryGroup,
        C128,
        SpinConstraint,
    >::builder()
    .parameters(&sdp_params)
    .determinant(&det_d3_cg)
    .sao(Some(&sao_spatial))
    .symmetry_group(pd_res)
    .build()
    .unwrap();
    assert!(sdp_driver.run().is_ok());

    let group = sdp_driver.construct_unitary_group().unwrap();
    let sao = sdp_driver.construct_sao().unwrap().0.unwrap();
    for (sym, mult) in [("||E~|_(1g)|", 1), ("||E~|_(2g)|", 1), ("||F~|_(g)|", 2)].iter() {
        let row = MullikenIrrepSymbol::from_str(sym).unwrap();
        let dyy_p = sdp_driver
            .result()
            .as_ref()
            .unwrap()
            .projected_determinants()
            .get(&row)
            .unwrap()
            .as_ref()
            .unwrap();
        let norm_sq = dyy_p.overlap(&dyy_p, Some(&sao), None).unwrap();
        let mixed_denmat = dyy_p
            .density_matrix(&sao.view(), 1e-7, 1e-7)
            .unwrap()
            .dot(&sao)
            / norm_sq;
        assert_abs_diff_ne!(mixed_denmat.norm_l2(), 0.0, epsilon = 1e-7);
        assert_abs_diff_eq!(mixed_denmat.trace().unwrap().abs(), 3.0, epsilon = 1e-7);

        let mut orbit_dyy_p = MultiDeterminantSymmetryOrbit::builder()
            .group(&group)
            .origin(&dyy_p)
            .integrality_threshold(1e-7)
            .linear_independence_threshold(1e-7)
            .symmetry_transformation_kind(SymmetryTransformationKind::SpinSpatial)
            .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
            .build()
            .unwrap();
        let _ = orbit_dyy_p
            .calc_smat(Some(&sao), None, true)
            .unwrap()
            .calc_xmat(false);
        assert_eq!(
            orbit_dyy_p.analyse_rep().unwrap(),
            DecomposedSymbol::<MullikenIrrepSymbol>::from_subspaces(&[(row, *mult)])
        );
    }

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // m Oh + θ·Oh (grey, magnetic) - spin spatial
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // No projections supported for magnetic-represented groups
    let sdp_params = SlaterDeterminantProjectionParams::builder()
        .numeric_projection_targets(Some((0..=9).collect_vec()))
        .use_magnetic_group(Some(MagneticSymmetryAnalysisKind::Corepresentation))
        .use_double_group(false)
        .symmetry_transformation_kind(SymmetryTransformationKind::SpinSpatial)
        .write_character_table(Some(CharacterTableDisplay::Symbolic))
        .build()
        .unwrap();

    let mut sdp_driver = SlaterDeterminantProjectionDriver::<
        UnitaryRepresentedSymmetryGroup,
        C128,
        SpinConstraint,
    >::builder()
    .parameters(&sdp_params)
    .determinant(&det_d3_cg)
    .sao(Some(&sao_spatial))
    .symmetry_group(pd_res)
    .build()
    .unwrap();
    assert!(sdp_driver.run().is_err());
}
