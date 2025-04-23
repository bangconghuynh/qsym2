use std::path::Path;

// use log4rs;
use byteorder::LittleEndian;
use itertools::Itertools;
use nalgebra::Vector3;
use ndarray::{array, Array1, Array2};
use num_complex::Complex;
use serial_test::serial;

use crate::analysis::{EigenvalueComparisonMode, RepAnalysis};
use crate::angmom::spinor_rotation_3d::SpinConstraint;
use crate::auxiliary::molecule::Molecule;
use crate::basis::ao::{
    BasisAngularOrder, BasisAtom, BasisShell, CartOrder, PureOrder, ShellOrder,
};
#[cfg(feature = "integrals")]
use crate::basis::ao_integrals::{BasisSet, BasisShellContraction, GaussianContraction};
use crate::chartab::chartab_symbols::DecomposedSymbol;
use crate::drivers::representation_analysis::angular_function::AngularFunctionRepAnalysisParams;
use crate::drivers::representation_analysis::slater_determinant::{
    SlaterDeterminantRepAnalysisDriver, SlaterDeterminantRepAnalysisParams,
};
use crate::drivers::representation_analysis::{
    CharacterTableDisplay, MagneticSymmetryAnalysisKind,
};
use crate::drivers::symmetry_group_detection::{
    SymmetryGroupDetectionDriver, SymmetryGroupDetectionParams,
};
use crate::drivers::QSym2Driver;
#[cfg(feature = "integrals")]
use crate::integrals::shell_tuple::build_shell_tuple_collection;
use crate::io::numeric::NumericReader;
use crate::symmetry::symmetry_core::{PreSymmetry, Symmetry};
use crate::symmetry::symmetry_group::{
    MagneticRepresentedSymmetryGroup, SymmetryGroupProperties, UnitaryRepresentedSymmetryGroup,
};
use crate::symmetry::symmetry_symbols::{MullikenIrcorepSymbol, MullikenIrrepSymbol};
use crate::symmetry::symmetry_transformation::SymmetryTransformationKind;
use crate::target::determinant::SlaterDeterminant;
use crate::target::orbital::orbital_analysis::MolecularOrbitalSymmetryOrbit;

type C128 = Complex<f64>;

const ROOT: &str = env!("CARGO_MANIFEST_DIR");

#[test]
fn test_drivers_slater_determinant_analysis_vf6() {
    // log4rs::init_file("log4rs.yml", Default::default()).unwrap();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/vf6.xyz");

    let afa_params = AngularFunctionRepAnalysisParams::default();

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
            .bao(&bao_vf6)
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
    let sda_params = SlaterDeterminantRepAnalysisParams::<f64>::builder()
        .integrality_threshold(1e-10)
        .linear_independence_threshold(1e-10)
        .analyse_mo_symmetries(true)
        .use_magnetic_group(None)
        .use_double_group(false)
        .use_cayley_table(true)
        .symmetry_transformation_kind(SymmetryTransformationKind::SpinSpatial)
        .write_overlap_eigenvalues(false)
        .write_character_table(Some(CharacterTableDisplay::Symbolic))
        .build()
        .unwrap();

    let mut sda_driver = SlaterDeterminantRepAnalysisDriver::<
        UnitaryRepresentedSymmetryGroup,
        C128,
        SpinConstraint,
    >::builder()
    .parameters(&sda_params)
    .angular_function_parameters(&afa_params)
    .determinant(&det_d3_cg)
    .sao_spatial(&sao_spatial)
    .symmetry_group(pd_res)
    .build()
    .unwrap();
    assert!(sda_driver.run().is_ok());
    assert_eq!(
        *sda_driver
            .result()
            .unwrap()
            .determinant_symmetry
            .as_ref()
            .unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new("||T|_(1g)| ⊕ ||T|_(2g)|").unwrap()
    );

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // u Oh* (ordinary double, unitary) - spin spatial
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    let sda_params = SlaterDeterminantRepAnalysisParams::<f64>::builder()
        .integrality_threshold(1e-10)
        .linear_independence_threshold(1e-10)
        .analyse_mo_symmetries(true)
        .use_magnetic_group(None)
        .use_double_group(true)
        .use_cayley_table(true)
        .symmetry_transformation_kind(SymmetryTransformationKind::SpinSpatial)
        .write_overlap_eigenvalues(false)
        .write_character_table(None)
        .build()
        .unwrap();

    let mut sda_driver = SlaterDeterminantRepAnalysisDriver::<
        UnitaryRepresentedSymmetryGroup,
        C128,
        SpinConstraint,
    >::builder()
    .parameters(&sda_params)
    .angular_function_parameters(&afa_params)
    .determinant(&det_d3_cg)
    .sao_spatial(&sao_spatial)
    .symmetry_group(pd_res)
    .build()
    .unwrap();
    assert!(sda_driver.run().is_ok());
    assert_eq!(
        *sda_driver
            .result()
            .unwrap()
            .determinant_symmetry
            .as_ref()
            .unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new("||E~|_(1g)| ⊕ ||E~|_(2g)| ⊕ 2||F~|_(g)|")
            .unwrap()
    );

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // u Oh + θ·Oh (grey, unitary) - spin spatial
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    let sda_params = SlaterDeterminantRepAnalysisParams::<f64>::builder()
        .integrality_threshold(1e-10)
        .linear_independence_threshold(1e-10)
        .analyse_mo_symmetries(true)
        .use_magnetic_group(Some(MagneticSymmetryAnalysisKind::Representation))
        .use_double_group(false)
        .use_cayley_table(true)
        .symmetry_transformation_kind(SymmetryTransformationKind::SpinSpatial)
        .write_overlap_eigenvalues(false)
        .write_character_table(None)
        .build()
        .unwrap();

    let mut sda_driver = SlaterDeterminantRepAnalysisDriver::<
        UnitaryRepresentedSymmetryGroup,
        C128,
        SpinConstraint,
    >::builder()
    .parameters(&sda_params)
    .angular_function_parameters(&afa_params)
    .determinant(&det_d3_cg)
    .sao_spatial(&sao_spatial)
    .symmetry_group(pd_res)
    .build()
    .unwrap();
    assert!(sda_driver.run().is_ok());
    assert!(sda_driver.result().unwrap().determinant_symmetry.is_err());

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // u (Oh + θ·Oh)* (grey double, unitary) - spin spatial
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    let sda_params = SlaterDeterminantRepAnalysisParams::<f64>::builder()
        .integrality_threshold(1e-10)
        .linear_independence_threshold(1e-10)
        .analyse_mo_symmetries(true)
        .use_magnetic_group(Some(MagneticSymmetryAnalysisKind::Representation))
        .use_double_group(true)
        .use_cayley_table(true)
        .symmetry_transformation_kind(SymmetryTransformationKind::SpinSpatial)
        .write_overlap_eigenvalues(false)
        .write_character_table(None)
        .build()
        .unwrap();

    let mut sda_driver = SlaterDeterminantRepAnalysisDriver::<
        UnitaryRepresentedSymmetryGroup,
        C128,
        SpinConstraint,
    >::builder()
    .parameters(&sda_params)
    .angular_function_parameters(&afa_params)
    .determinant(&det_d3_cg)
    .sao_spatial(&sao_spatial)
    .symmetry_group(pd_res)
    .build()
    .unwrap();
    assert!(sda_driver.run().is_ok());
    assert!(sda_driver.result().unwrap().determinant_symmetry.is_err());

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // m Oh + θ·Oh (grey, magnetic) - spin spatial
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    let sda_params = SlaterDeterminantRepAnalysisParams::<f64>::builder()
        .integrality_threshold(1e-10)
        .linear_independence_threshold(1e-10)
        .analyse_mo_symmetries(true)
        .use_magnetic_group(Some(MagneticSymmetryAnalysisKind::Corepresentation))
        .use_double_group(false)
        .use_cayley_table(true)
        .symmetry_transformation_kind(SymmetryTransformationKind::SpinSpatial)
        .write_overlap_eigenvalues(false)
        .write_character_table(None)
        .build()
        .unwrap();

    let mut sda_driver = SlaterDeterminantRepAnalysisDriver::<
        MagneticRepresentedSymmetryGroup,
        C128,
        SpinConstraint,
    >::builder()
    .parameters(&sda_params)
    .angular_function_parameters(&afa_params)
    .determinant(&det_d3_cg)
    .sao_spatial(&sao_spatial)
    .symmetry_group(pd_res)
    .build()
    .unwrap();
    assert!(sda_driver.run().is_ok());
    assert_eq!(
        *sda_driver
            .result()
            .unwrap()
            .determinant_symmetry
            .as_ref()
            .unwrap(),
        DecomposedSymbol::<MullikenIrcorepSymbol>::new("2||T|_(1g)| ⊕ 2||T|_(2g)|").unwrap()
    );

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // m (Oh + θ·Oh)* (grey double, magnetic) - spin spatial
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    let sda_params = SlaterDeterminantRepAnalysisParams::<f64>::builder()
        .integrality_threshold(1e-10)
        .linear_independence_threshold(1e-10)
        .analyse_mo_symmetries(true)
        .use_magnetic_group(Some(MagneticSymmetryAnalysisKind::Corepresentation))
        .use_double_group(true)
        .use_cayley_table(true)
        .symmetry_transformation_kind(SymmetryTransformationKind::SpinSpatial)
        .write_overlap_eigenvalues(true)
        .write_character_table(None)
        .build()
        .unwrap();

    let mut sda_driver = SlaterDeterminantRepAnalysisDriver::<
        MagneticRepresentedSymmetryGroup,
        C128,
        SpinConstraint,
    >::builder()
    .parameters(&sda_params)
    .angular_function_parameters(&afa_params)
    .determinant(&det_d3_cg)
    .sao_spatial(&sao_spatial)
    .symmetry_group(pd_res)
    .build()
    .unwrap();
    assert!(sda_driver.run().is_ok());
    assert_eq!(
        *sda_driver
            .result()
            .unwrap()
            .determinant_symmetry
            .as_ref()
            .unwrap(),
        DecomposedSymbol::<MullikenIrcorepSymbol>::new("||E~|_(1g)| ⊕ ||E~|_(2g)| ⊕ 2||F~|_(g)|")
            .unwrap()
    );
}

#[cfg(feature = "integrals")]
#[test]
#[serial]
fn test_drivers_slater_determinant_density_analysis_vf6() {
    // log4rs::init_file("log4rs.yml", Default::default()).unwrap();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/vf6.xyz");

    let afa_params = AngularFunctionRepAnalysisParams::default();

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

    let batm_v = BasisAtom::new(&mol_vf6.atoms[0], &[bsc_d.clone()]);
    let batm_f0 = BasisAtom::new(&mol_vf6.atoms[1], &[bsp_s.clone()]);
    let batm_f1 = BasisAtom::new(&mol_vf6.atoms[2], &[bsp_s.clone()]);
    let batm_f2 = BasisAtom::new(&mol_vf6.atoms[3], &[bsp_s.clone()]);
    let batm_f3 = BasisAtom::new(&mol_vf6.atoms[4], &[bsp_s.clone()]);
    let batm_f4 = BasisAtom::new(&mol_vf6.atoms[5], &[bsp_s.clone()]);
    let batm_f5 = BasisAtom::new(&mol_vf6.atoms[6], &[bsp_s.clone()]);

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

    // -------------------
    // Four-centre overlap
    // -------------------

    let gc = GaussianContraction::<f64, f64> {
        primitives: vec![(3.4252509140, 1.0)],
    };
    let bsc0 = BasisShellContraction::<f64, f64> {
        basis_shell: bsc_d,
        contraction: gc.clone(),
        cart_origin: mol_vf6.atoms[0].coordinates.clone(),
        k: None,
    };
    let bsc1 = BasisShellContraction::<f64, f64> {
        basis_shell: bsp_s.clone(),
        contraction: gc.clone(),
        cart_origin: mol_vf6.atoms[1].coordinates.clone(),
        k: None,
    };
    let bsc2 = BasisShellContraction::<f64, f64> {
        basis_shell: bsp_s.clone(),
        contraction: gc.clone(),
        cart_origin: mol_vf6.atoms[2].coordinates.clone(),
        k: None,
    };
    let bsc3 = BasisShellContraction::<f64, f64> {
        basis_shell: bsp_s.clone(),
        contraction: gc.clone(),
        cart_origin: mol_vf6.atoms[3].coordinates.clone(),
        k: None,
    };
    let bsc4 = BasisShellContraction::<f64, f64> {
        basis_shell: bsp_s.clone(),
        contraction: gc.clone(),
        cart_origin: mol_vf6.atoms[4].coordinates.clone(),
        k: None,
    };
    let bsc5 = BasisShellContraction::<f64, f64> {
        basis_shell: bsp_s.clone(),
        contraction: gc.clone(),
        cart_origin: mol_vf6.atoms[5].coordinates.clone(),
        k: None,
    };
    let bsc6 = BasisShellContraction::<f64, f64> {
        basis_shell: bsp_s,
        contraction: gc,
        cart_origin: mol_vf6.atoms[6].coordinates.clone(),
        k: None,
    };

    let bscs = BasisSet::new(vec![
        vec![bsc0],
        vec![bsc1],
        vec![bsc2],
        vec![bsc3],
        vec![bsc4],
        vec![bsc5],
        vec![bsc6],
    ]);
    let stc = build_shell_tuple_collection![
        <s1, s2, s3, s4>;
        false, false, false, false;
        &bscs, &bscs, &bscs, &bscs;
        f64
    ];
    let ovs = stc.overlap([0, 0, 0, 0]);
    let sao_spatial_4c = ovs[0].mapv(C128::from);

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
            .bao(&bao_vf6)
            .mol(mol_vf6)
            .structure_constraint(SpinConstraint::Unrestricted(2, false))
            .complex_symmetric(false)
            .threshold(1e-14)
            .build()
            .unwrap()
            .to_generalised()
            .into();

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // u Oh (ordinary, unitary) - spin-spatial
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    let sda_params = SlaterDeterminantRepAnalysisParams::<f64>::builder()
        .integrality_threshold(1e-10)
        .linear_independence_threshold(1e-10)
        .analyse_mo_symmetries(true)
        .analyse_density_symmetries(true)
        .use_magnetic_group(None)
        .use_double_group(false)
        .use_cayley_table(true)
        .symmetry_transformation_kind(SymmetryTransformationKind::SpinSpatial)
        .write_overlap_eigenvalues(false)
        .write_character_table(Some(CharacterTableDisplay::Symbolic))
        .build()
        .unwrap();

    let mut sda_driver = SlaterDeterminantRepAnalysisDriver::<
        UnitaryRepresentedSymmetryGroup,
        C128,
        SpinConstraint,
    >::builder()
    .parameters(&sda_params)
    .angular_function_parameters(&afa_params)
    .determinant(&det_d3_cg)
    .sao_spatial(&sao_spatial)
    .sao_spatial_4c(Some(&sao_spatial_4c))
    .symmetry_group(pd_res)
    .build()
    .unwrap();
    assert!(sda_driver.run().is_ok());
    assert_eq!(
        *sda_driver
            .result()
            .unwrap()
            .determinant_symmetry
            .as_ref()
            .unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new("||T|_(1g)| ⊕ ||T|_(2g)|").unwrap()
    );

    // Spin-0 density symmetry
    assert_eq!(
        *sda_driver
            .result()
            .unwrap()
            .determinant_density_symmetries
            .as_ref()
            .unwrap()[0]
            .1
            .as_ref()
            .unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new("||A|_(1g)| ⊕ 2||E|_(g)|").unwrap()
    );
    // Spin-1 density symmetry
    assert_eq!(
        *sda_driver
            .result()
            .unwrap()
            .determinant_density_symmetries
            .as_ref()
            .unwrap()[1]
            .1
            .as_ref()
            .unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new("||A|_(1g)| ⊕ ||E|_(g)|").unwrap()
    );
    // Total density symmetry
    assert_eq!(
        *sda_driver
            .result()
            .unwrap()
            .determinant_density_symmetries
            .as_ref()
            .unwrap()[2]
            .1
            .as_ref()
            .unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new("||A|_(1g)| ⊕ 2||E|_(g)|").unwrap()
    );
    // Spin-polarised density 0 - 1 symmetry
    assert_eq!(
        *sda_driver
            .result()
            .unwrap()
            .determinant_density_symmetries
            .as_ref()
            .unwrap()[3]
            .1
            .as_ref()
            .unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new("||A|_(1g)| ⊕ 2||E|_(g)|").unwrap()
    );

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // u Oh* (ordinary double, unitary) - spin spatial
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    let sda_params = SlaterDeterminantRepAnalysisParams::<f64>::builder()
        .integrality_threshold(1e-10)
        .linear_independence_threshold(1e-10)
        .analyse_mo_symmetries(true)
        .analyse_density_symmetries(true)
        .use_magnetic_group(None)
        .use_double_group(true)
        .use_cayley_table(true)
        .symmetry_transformation_kind(SymmetryTransformationKind::SpinSpatial)
        .write_overlap_eigenvalues(false)
        .write_character_table(None)
        .build()
        .unwrap();

    let mut sda_driver = SlaterDeterminantRepAnalysisDriver::<
        UnitaryRepresentedSymmetryGroup,
        C128,
        SpinConstraint,
    >::builder()
    .parameters(&sda_params)
    .angular_function_parameters(&afa_params)
    .determinant(&det_d3_cg)
    .sao_spatial(&sao_spatial)
    .sao_spatial_4c(Some(&sao_spatial_4c))
    .symmetry_group(pd_res)
    .build()
    .unwrap();
    assert!(sda_driver.run().is_ok());
    assert_eq!(
        *sda_driver
            .result()
            .unwrap()
            .determinant_symmetry
            .as_ref()
            .unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new("||E~|_(1g)| ⊕ ||E~|_(2g)| ⊕ 2||F~|_(g)|")
            .unwrap()
    );

    // Spin-0 density symmetry
    assert_eq!(
        *sda_driver
            .result()
            .unwrap()
            .determinant_density_symmetries
            .as_ref()
            .unwrap()[0]
            .1
            .as_ref()
            .unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new("||A|_(1g)| ⊕ 2||E|_(g)|").unwrap()
    );
    // Spin-1 density symmetry
    assert_eq!(
        *sda_driver
            .result()
            .unwrap()
            .determinant_density_symmetries
            .as_ref()
            .unwrap()[1]
            .1
            .as_ref()
            .unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new("||A|_(1g)| ⊕ ||E|_(g)|").unwrap()
    );
    // Total density symmetry
    assert_eq!(
        *sda_driver
            .result()
            .unwrap()
            .determinant_density_symmetries
            .as_ref()
            .unwrap()[2]
            .1
            .as_ref()
            .unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new("||A|_(1g)| ⊕ 2||E|_(g)|").unwrap()
    );
    // Spin-polarised density 0 - 1 symmetry
    assert_eq!(
        *sda_driver
            .result()
            .unwrap()
            .determinant_density_symmetries
            .as_ref()
            .unwrap()[3]
            .1
            .as_ref()
            .unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new("||A|_(1g)| ⊕ 2||E|_(g)|").unwrap()
    );

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // u Oh + θ·Oh (grey, unitary) - spin spatial
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    let sda_params = SlaterDeterminantRepAnalysisParams::<f64>::builder()
        .integrality_threshold(1e-10)
        .linear_independence_threshold(1e-10)
        .analyse_mo_symmetries(true)
        .analyse_density_symmetries(true)
        .use_magnetic_group(Some(MagneticSymmetryAnalysisKind::Representation))
        .use_double_group(false)
        .use_cayley_table(true)
        .symmetry_transformation_kind(SymmetryTransformationKind::SpinSpatial)
        .write_overlap_eigenvalues(false)
        .write_character_table(None)
        .build()
        .unwrap();

    let mut sda_driver = SlaterDeterminantRepAnalysisDriver::<
        UnitaryRepresentedSymmetryGroup,
        C128,
        SpinConstraint,
    >::builder()
    .parameters(&sda_params)
    .angular_function_parameters(&afa_params)
    .determinant(&det_d3_cg)
    .sao_spatial(&sao_spatial)
    .sao_spatial_4c(Some(&sao_spatial_4c))
    .symmetry_group(pd_res)
    .build()
    .unwrap();
    assert!(sda_driver.run().is_ok());
    assert!(sda_driver.result().unwrap().determinant_symmetry.is_err());

    // Spin-0 density symmetry
    assert_eq!(
        *sda_driver
            .result()
            .unwrap()
            .determinant_density_symmetries
            .as_ref()
            .unwrap()[0]
            .1
            .as_ref()
            .unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new("|^(+)|A|_(1g)| ⊕ 2|^(+)|E|_(g)|").unwrap()
    );
    // Spin-1 density symmetry
    assert_eq!(
        *sda_driver
            .result()
            .unwrap()
            .determinant_density_symmetries
            .as_ref()
            .unwrap()[1]
            .1
            .as_ref()
            .unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new("|^(+)|A|_(1g)| ⊕ |^(+)|E|_(g)|").unwrap()
    );
    // Total density symmetry
    assert_eq!(
        *sda_driver
            .result()
            .unwrap()
            .determinant_density_symmetries
            .as_ref()
            .unwrap()[2]
            .1
            .as_ref()
            .unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new("|^(+)|A|_(1g)| ⊕ 2|^(+)|E|_(g)|").unwrap()
    );
    // Spin-polarised density 0 - 1 symmetry
    assert_eq!(
        *sda_driver
            .result()
            .unwrap()
            .determinant_density_symmetries
            .as_ref()
            .unwrap()[3]
            .1
            .as_ref()
            .unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new("|^(+)|A|_(1g)| ⊕ 2|^(+)|E|_(g)|").unwrap()
    );

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // u (Oh + θ·Oh)* (grey double, unitary) - spin spatial
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    let sda_params = SlaterDeterminantRepAnalysisParams::<f64>::builder()
        .integrality_threshold(1e-10)
        .linear_independence_threshold(1e-10)
        .analyse_mo_symmetries(true)
        .analyse_density_symmetries(true)
        .use_magnetic_group(Some(MagneticSymmetryAnalysisKind::Representation))
        .use_double_group(true)
        .use_cayley_table(true)
        .symmetry_transformation_kind(SymmetryTransformationKind::SpinSpatial)
        .write_overlap_eigenvalues(false)
        .write_character_table(None)
        .build()
        .unwrap();

    let mut sda_driver = SlaterDeterminantRepAnalysisDriver::<
        UnitaryRepresentedSymmetryGroup,
        C128,
        SpinConstraint,
    >::builder()
    .parameters(&sda_params)
    .angular_function_parameters(&afa_params)
    .determinant(&det_d3_cg)
    .sao_spatial(&sao_spatial)
    .sao_spatial_4c(Some(&sao_spatial_4c))
    .symmetry_group(pd_res)
    .build()
    .unwrap();
    assert!(sda_driver.run().is_ok());
    assert!(sda_driver.result().unwrap().determinant_symmetry.is_err());

    // Spin-0 density symmetry
    assert_eq!(
        *sda_driver
            .result()
            .unwrap()
            .determinant_density_symmetries
            .as_ref()
            .unwrap()[0]
            .1
            .as_ref()
            .unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new("|^(+)|A|_(1g)| ⊕ 2|^(+)|E|_(g)|").unwrap()
    );
    // Spin-1 density symmetry
    assert_eq!(
        *sda_driver
            .result()
            .unwrap()
            .determinant_density_symmetries
            .as_ref()
            .unwrap()[1]
            .1
            .as_ref()
            .unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new("|^(+)|A|_(1g)| ⊕ |^(+)|E|_(g)|").unwrap()
    );
    // Total density symmetry
    assert_eq!(
        *sda_driver
            .result()
            .unwrap()
            .determinant_density_symmetries
            .as_ref()
            .unwrap()[2]
            .1
            .as_ref()
            .unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new("|^(+)|A|_(1g)| ⊕ 2|^(+)|E|_(g)|").unwrap()
    );
    // Spin-polarised density 0 - 1 symmetry
    assert_eq!(
        *sda_driver
            .result()
            .unwrap()
            .determinant_density_symmetries
            .as_ref()
            .unwrap()[3]
            .1
            .as_ref()
            .unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new("|^(+)|A|_(1g)| ⊕ 2|^(+)|E|_(g)|").unwrap()
    );

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // m Oh + θ·Oh (grey, magnetic) - spin spatial
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    let sda_params = SlaterDeterminantRepAnalysisParams::<f64>::builder()
        .integrality_threshold(1e-10)
        .linear_independence_threshold(1e-10)
        .analyse_mo_symmetries(true)
        .analyse_density_symmetries(true)
        .use_magnetic_group(Some(MagneticSymmetryAnalysisKind::Corepresentation))
        .use_double_group(false)
        .use_cayley_table(true)
        .symmetry_transformation_kind(SymmetryTransformationKind::SpinSpatial)
        .write_overlap_eigenvalues(false)
        .write_character_table(None)
        .build()
        .unwrap();

    let mut sda_driver = SlaterDeterminantRepAnalysisDriver::<
        MagneticRepresentedSymmetryGroup,
        C128,
        SpinConstraint,
    >::builder()
    .parameters(&sda_params)
    .angular_function_parameters(&afa_params)
    .determinant(&det_d3_cg)
    .sao_spatial(&sao_spatial)
    .sao_spatial_4c(Some(&sao_spatial_4c))
    .symmetry_group(pd_res)
    .build()
    .unwrap();
    assert!(sda_driver.run().is_ok());
    assert_eq!(
        *sda_driver
            .result()
            .unwrap()
            .determinant_symmetry
            .as_ref()
            .unwrap(),
        DecomposedSymbol::<MullikenIrcorepSymbol>::new("2||T|_(1g)| ⊕ 2||T|_(2g)|").unwrap()
    );

    // Spin-0 density symmetry
    assert_eq!(
        *sda_driver
            .result()
            .unwrap()
            .determinant_density_symmetries
            .as_ref()
            .unwrap()[0]
            .1
            .as_ref()
            .unwrap(),
        DecomposedSymbol::<MullikenIrcorepSymbol>::new("||A|_(1g)| ⊕ 2||E|_(g)|").unwrap()
    );
    // Spin-1 density symmetry
    assert_eq!(
        *sda_driver
            .result()
            .unwrap()
            .determinant_density_symmetries
            .as_ref()
            .unwrap()[1]
            .1
            .as_ref()
            .unwrap(),
        DecomposedSymbol::<MullikenIrcorepSymbol>::new("||A|_(1g)| ⊕ ||E|_(g)|").unwrap()
    );
    // Total density symmetry
    assert_eq!(
        *sda_driver
            .result()
            .unwrap()
            .determinant_density_symmetries
            .as_ref()
            .unwrap()[2]
            .1
            .as_ref()
            .unwrap(),
        DecomposedSymbol::<MullikenIrcorepSymbol>::new("||A|_(1g)| ⊕ 2||E|_(g)|").unwrap()
    );
    // Spin-polarised density 0 - 1 symmetry
    assert_eq!(
        *sda_driver
            .result()
            .unwrap()
            .determinant_density_symmetries
            .as_ref()
            .unwrap()[3]
            .1
            .as_ref()
            .unwrap(),
        DecomposedSymbol::<MullikenIrcorepSymbol>::new("||A|_(1g)| ⊕ 2||E|_(g)|").unwrap()
    );

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // m (Oh + θ·Oh)* (grey double, magnetic) - spin spatial
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    let sda_params = SlaterDeterminantRepAnalysisParams::<f64>::builder()
        .integrality_threshold(1e-10)
        .linear_independence_threshold(1e-10)
        .analyse_mo_symmetries(true)
        .analyse_density_symmetries(true)
        .use_magnetic_group(Some(MagneticSymmetryAnalysisKind::Corepresentation))
        .use_double_group(true)
        .use_cayley_table(true)
        .symmetry_transformation_kind(SymmetryTransformationKind::SpinSpatial)
        .write_overlap_eigenvalues(true)
        .write_character_table(None)
        .build()
        .unwrap();

    let mut sda_driver = SlaterDeterminantRepAnalysisDriver::<
        MagneticRepresentedSymmetryGroup,
        C128,
        SpinConstraint,
    >::builder()
    .parameters(&sda_params)
    .angular_function_parameters(&afa_params)
    .determinant(&det_d3_cg)
    .sao_spatial(&sao_spatial)
    .sao_spatial_4c(Some(&sao_spatial_4c))
    .symmetry_group(pd_res)
    .build()
    .unwrap();
    assert!(sda_driver.run().is_ok());
    assert_eq!(
        *sda_driver
            .result()
            .unwrap()
            .determinant_symmetry
            .as_ref()
            .unwrap(),
        DecomposedSymbol::<MullikenIrcorepSymbol>::new("||E~|_(1g)| ⊕ ||E~|_(2g)| ⊕ 2||F~|_(g)|")
            .unwrap()
    );

    // Spin-0 density symmetry
    assert_eq!(
        *sda_driver
            .result()
            .unwrap()
            .determinant_density_symmetries
            .as_ref()
            .unwrap()[0]
            .1
            .as_ref()
            .unwrap(),
        DecomposedSymbol::<MullikenIrcorepSymbol>::new("||A|_(1g)| ⊕ 2||E|_(g)|").unwrap()
    );
    // Spin-1 density symmetry
    assert_eq!(
        *sda_driver
            .result()
            .unwrap()
            .determinant_density_symmetries
            .as_ref()
            .unwrap()[1]
            .1
            .as_ref()
            .unwrap(),
        DecomposedSymbol::<MullikenIrcorepSymbol>::new("||A|_(1g)| ⊕ ||E|_(g)|").unwrap()
    );
    // Total density symmetry
    assert_eq!(
        *sda_driver
            .result()
            .unwrap()
            .determinant_density_symmetries
            .as_ref()
            .unwrap()[2]
            .1
            .as_ref()
            .unwrap(),
        DecomposedSymbol::<MullikenIrcorepSymbol>::new("||A|_(1g)| ⊕ 2||E|_(g)|").unwrap()
    );
    // Spin-polarised density 0 - 1 symmetry
    assert_eq!(
        *sda_driver
            .result()
            .unwrap()
            .determinant_density_symmetries
            .as_ref()
            .unwrap()[3]
            .1
            .as_ref()
            .unwrap(),
        DecomposedSymbol::<MullikenIrcorepSymbol>::new("||A|_(1g)| ⊕ 2||E|_(g)|").unwrap()
    );
}

#[test]
fn test_drivers_slater_determinant_analysis_h2co_by_mag_uni() {
    // log4rs::init_file("log4rs.yml", Default::default()).unwrap();
    let path: String = format!("{}{}", ROOT, "/tests/binaries/h2co/h2co.xyz");

    let mut mol_h2co = Molecule::from_xyz(&path, 1e-6);
    mol_h2co.set_magnetic_field(Some(Vector3::new(0.0, 1.0, 0.0)));

    let afa_params = AngularFunctionRepAnalysisParams::default();

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
        .molecule(Some(&mol_h2co))
        .build()
        .unwrap();
    assert!(pd_driver.run().is_ok());
    let pd_res = pd_driver.result().unwrap();
    let mol_h2co = &pd_res.pre_symmetry.recentred_molecule;

    let bsc_s = BasisShell::new(0, ShellOrder::Cart(CartOrder::lex(0)));
    let bsc_p = BasisShell::new(1, ShellOrder::Cart(CartOrder::lex(1)));
    let bsc_d = BasisShell::new(2, ShellOrder::Cart(CartOrder::lex(2)));
    let bsc_f = BasisShell::new(3, ShellOrder::Cart(CartOrder::lex(3)));

    let batm_c = BasisAtom::new(
        &mol_h2co.atoms[0],
        &[
            bsc_s.clone(),
            bsc_s.clone(),
            bsc_s.clone(),
            bsc_s.clone(),
            bsc_s.clone(),
            bsc_s.clone(),
            bsc_s.clone(),
            bsc_s.clone(),
            bsc_s.clone(),
            bsc_s.clone(),
            bsc_p.clone(),
            bsc_p.clone(),
            bsc_p.clone(),
            bsc_p.clone(),
            bsc_p.clone(),
            bsc_d.clone(),
            bsc_d.clone(),
            bsc_f.clone(),
        ],
    );
    let batm_o = BasisAtom::new(
        &mol_h2co.atoms[0],
        &[
            bsc_s.clone(),
            bsc_s.clone(),
            bsc_s.clone(),
            bsc_s.clone(),
            bsc_s.clone(),
            bsc_s.clone(),
            bsc_s.clone(),
            bsc_s.clone(),
            bsc_s.clone(),
            bsc_s.clone(),
            bsc_p.clone(),
            bsc_p.clone(),
            bsc_p.clone(),
            bsc_p.clone(),
            bsc_p.clone(),
            bsc_d.clone(),
            bsc_d.clone(),
            bsc_f.clone(),
        ],
    );
    let batm_h1 = BasisAtom::new(
        &mol_h2co.atoms[0],
        &[
            bsc_s.clone(),
            bsc_s.clone(),
            bsc_s.clone(),
            bsc_s.clone(),
            bsc_s,
            bsc_p.clone(),
            bsc_p,
            bsc_d,
        ],
    );
    let batm_h2 = batm_h1.clone();

    let bao_h2co = BasisAngularOrder::new(&[batm_c, batm_o, batm_h1, batm_h2]);
    let n_spatial: usize = 128;

    let sao_spatial = read_complex_array(
        &format!("{ROOT}/tests/binaries/h2co/sao_spatial_re"),
        &format!("{ROOT}/tests/binaries/h2co/sao_spatial_im"),
        n_spatial,
        n_spatial,
    );
    let sao_spatial_h = read_complex_array(
        &format!("{ROOT}/tests/binaries/h2co/sao_spatial_h_re"),
        &format!("{ROOT}/tests/binaries/h2co/sao_spatial_h_im"),
        n_spatial,
        n_spatial,
    );
    let ca = read_complex_array(
        &format!("{ROOT}/tests/binaries/h2co/a_re"),
        &format!("{ROOT}/tests/binaries/h2co/a_im"),
        n_spatial,
        n_spatial,
    );
    let cb = read_complex_array(
        &format!("{ROOT}/tests/binaries/h2co/b_re"),
        &format!("{ROOT}/tests/binaries/h2co/b_im"),
        n_spatial,
        n_spatial,
    );

    // ----------------------
    // Determinantal symmetry
    // ----------------------
    let oa = Array1::from_iter((0..n_spatial).map(|i| if i < 8 { 1.0 } else { 0.0 }));
    let ob = oa.clone();
    let det_cu = SlaterDeterminant::<C128, SpinConstraint>::builder()
        .coefficients(&[ca, cb])
        .occupations(&[oa, ob])
        .bao(&bao_h2co)
        .mol(mol_h2co)
        .structure_constraint(SpinConstraint::Unrestricted(2, false))
        .complex_symmetric(false)
        .threshold(1e-14)
        .build()
        .unwrap();

    let sda_params = SlaterDeterminantRepAnalysisParams::<f64>::builder()
        .integrality_threshold(1e-7)
        .linear_independence_threshold(1e-7)
        .analyse_mo_symmetries(true)
        .use_magnetic_group(Some(MagneticSymmetryAnalysisKind::Representation))
        .use_double_group(false)
        .use_cayley_table(true)
        .symmetry_transformation_kind(SymmetryTransformationKind::Spatial)
        .write_overlap_eigenvalues(false)
        .write_character_table(Some(CharacterTableDisplay::Symbolic))
        .build()
        .unwrap();

    let mut sda_driver = SlaterDeterminantRepAnalysisDriver::<
        UnitaryRepresentedSymmetryGroup,
        C128,
        SpinConstraint,
    >::builder()
    .parameters(&sda_params)
    .angular_function_parameters(&afa_params)
    .determinant(&det_cu)
    .sao_spatial(&sao_spatial)
    .sao_spatial_h(Some(&sao_spatial_h))
    .symmetry_group(pd_res)
    .build()
    .unwrap();
    assert!(sda_driver.run().is_ok());
    assert!(sda_driver.result().unwrap().determinant_symmetry.is_err());

    // ------------------
    // Orbital symmetries
    // ------------------
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-6)
        .molecule(&mol_h2co)
        .build()
        .unwrap();
    let mut magsym = Symmetry::new();
    magsym.analyse(&presym, true).unwrap();
    let group_u_c2v_cs =
        UnitaryRepresentedSymmetryGroup::from_molecular_symmetry(&magsym, None).unwrap();

    let orbss = det_cu.to_orbitals();
    let orb_a6 = &orbss[0][6];
    let orb_a7 = &orbss[0][7];

    let mut orb_a6_orbit = MolecularOrbitalSymmetryOrbit::builder()
        .group(&group_u_c2v_cs)
        .origin(&orb_a6)
        .integrality_threshold(1e-7)
        .linear_independence_threshold(1e-7)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
        .symmetry_transformation_kind(SymmetryTransformationKind::Spatial)
        .build()
        .unwrap();
    let mut orb_a7_orbit = MolecularOrbitalSymmetryOrbit::builder()
        .group(&group_u_c2v_cs)
        .origin(&orb_a7)
        .integrality_threshold(1e-7)
        .linear_independence_threshold(1e-7)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
        .symmetry_transformation_kind(SymmetryTransformationKind::Spatial)
        .build()
        .unwrap();

    // See Figure 5 in arXiv:2402.15595 and the discussion therein for an explanation.
    let _ = orb_a6_orbit
        .calc_smat(Some(&sao_spatial), Some(&sao_spatial_h), true)
        .unwrap()
        .calc_xmat(false);
    assert!(orb_a6_orbit.analyse_rep().is_err());

    let _ = orb_a7_orbit
        .calc_smat(Some(&sao_spatial), Some(&sao_spatial_h), true)
        .unwrap()
        .calc_xmat(false);
    assert_eq!(
        orb_a7_orbit.analyse_rep().unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new("||A|^(')_(1)|").unwrap()
    );
}

/// Reads in a two-dimensional complex array from two separate binary files, each containing the
/// real or the imaginary part of the array.
///
/// # Arguments
///
/// * `path_re` - Path to the binary file containing the real part of the array.
/// * `path_im` - Path to the binary file containing the imaginary part of the array.
///
/// # Returns
///
/// The two-dimensional complex array.
fn read_complex_array<P: AsRef<Path>>(
    path_re: P,
    path_im: P,
    nrows: usize,
    ncols: usize,
) -> Array2<Complex<f64>> {
    let re = NumericReader::<_, LittleEndian, f64>::from_file(path_re)
        .unwrap()
        .collect::<Vec<_>>();
    let im = NumericReader::<_, LittleEndian, f64>::from_file(path_im)
        .unwrap()
        .collect::<Vec<_>>();
    Array2::from_shape_vec(
        (nrows, ncols),
        re.into_iter()
            .zip(im.into_iter())
            .map(|(re, im)| Complex::<f64>::new(re, im))
            .collect_vec(),
    )
    .unwrap()
}
