use ndarray::array;
use num_complex::Complex;

use crate::angmom::spinor_rotation_3d::SpinConstraint;
use crate::basis::ao::{
    BasisAngularOrder, BasisAtom, BasisShell, CartOrder, PureOrder, ShellOrder,
};
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
use crate::integrals::shell_tuple::build_shell_tuple_collection;
use crate::symmetry::symmetry_group::{
    MagneticRepresentedSymmetryGroup, UnitaryRepresentedSymmetryGroup,
};
use crate::symmetry::symmetry_symbols::{MullikenIrcorepSymbol, MullikenIrrepSymbol};
use crate::symmetry::symmetry_transformation::SymmetryTransformationKind;
use crate::target::determinant::SlaterDeterminant;

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
    let det_d3_cg: SlaterDeterminant<C128> = SlaterDeterminant::<f64>::builder()
        .coefficients(&[calpha, cbeta])
        .occupations(&[oalpha, obeta])
        .bao(&bao_vf6)
        .mol(mol_vf6)
        .spin_constraint(SpinConstraint::Unrestricted(2, false))
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
        .symmetry_transformation_kind(SymmetryTransformationKind::SpinSpatial)
        .write_overlap_eigenvalues(false)
        .write_character_table(Some(CharacterTableDisplay::Symbolic))
        .build()
        .unwrap();

    let mut sda_driver =
        SlaterDeterminantRepAnalysisDriver::<UnitaryRepresentedSymmetryGroup, C128>::builder()
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
        .symmetry_transformation_kind(SymmetryTransformationKind::SpinSpatial)
        .write_overlap_eigenvalues(false)
        .write_character_table(None)
        .build()
        .unwrap();

    let mut sda_driver =
        SlaterDeterminantRepAnalysisDriver::<UnitaryRepresentedSymmetryGroup, C128>::builder()
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
        DecomposedSymbol::<MullikenIrrepSymbol>::new("||E~|_(1g)| ⊕ ||E~|_(2g)| ⊕ 2||G~|_(g)|")
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
        .symmetry_transformation_kind(SymmetryTransformationKind::SpinSpatial)
        .write_overlap_eigenvalues(false)
        .write_character_table(None)
        .build()
        .unwrap();

    let mut sda_driver =
        SlaterDeterminantRepAnalysisDriver::<UnitaryRepresentedSymmetryGroup, C128>::builder()
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
        .symmetry_transformation_kind(SymmetryTransformationKind::SpinSpatial)
        .write_overlap_eigenvalues(false)
        .write_character_table(None)
        .build()
        .unwrap();

    let mut sda_driver =
        SlaterDeterminantRepAnalysisDriver::<UnitaryRepresentedSymmetryGroup, C128>::builder()
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
        .symmetry_transformation_kind(SymmetryTransformationKind::SpinSpatial)
        .write_overlap_eigenvalues(false)
        .write_character_table(None)
        .build()
        .unwrap();

    let mut sda_driver =
        SlaterDeterminantRepAnalysisDriver::<MagneticRepresentedSymmetryGroup, C128>::builder()
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
        .symmetry_transformation_kind(SymmetryTransformationKind::SpinSpatial)
        .write_overlap_eigenvalues(true)
        .write_character_table(None)
        .build()
        .unwrap();

    let mut sda_driver =
        SlaterDeterminantRepAnalysisDriver::<MagneticRepresentedSymmetryGroup, C128>::builder()
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
        DecomposedSymbol::<MullikenIrcorepSymbol>::new("||E~|_(1g)| ⊕ ||E~|_(2g)| ⊕ 2||G~|_(g)|")
            .unwrap()
    );
}

#[test]
fn test_drivers_slater_determinant_density_analysis_vf6() {
    log4rs::init_file("log4rs.yml", Default::default()).unwrap();
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
    let det_d3_cg: SlaterDeterminant<C128> = SlaterDeterminant::<f64>::builder()
        .coefficients(&[calpha, cbeta])
        .occupations(&[oalpha, obeta])
        .bao(&bao_vf6)
        .mol(mol_vf6)
        .spin_constraint(SpinConstraint::Unrestricted(2, false))
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
        .symmetry_transformation_kind(SymmetryTransformationKind::SpinSpatial)
        .write_overlap_eigenvalues(false)
        .write_character_table(Some(CharacterTableDisplay::Symbolic))
        .build()
        .unwrap();

    let mut sda_driver =
        SlaterDeterminantRepAnalysisDriver::<UnitaryRepresentedSymmetryGroup, C128>::builder()
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
        .symmetry_transformation_kind(SymmetryTransformationKind::SpinSpatial)
        .write_overlap_eigenvalues(false)
        .write_character_table(None)
        .build()
        .unwrap();

    let mut sda_driver =
        SlaterDeterminantRepAnalysisDriver::<UnitaryRepresentedSymmetryGroup, C128>::builder()
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
        DecomposedSymbol::<MullikenIrrepSymbol>::new("||E~|_(1g)| ⊕ ||E~|_(2g)| ⊕ 2||G~|_(g)|")
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
        .symmetry_transformation_kind(SymmetryTransformationKind::SpinSpatial)
        .write_overlap_eigenvalues(false)
        .write_character_table(None)
        .build()
        .unwrap();

    let mut sda_driver =
        SlaterDeterminantRepAnalysisDriver::<UnitaryRepresentedSymmetryGroup, C128>::builder()
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
        .symmetry_transformation_kind(SymmetryTransformationKind::SpinSpatial)
        .write_overlap_eigenvalues(false)
        .write_character_table(None)
        .build()
        .unwrap();

    let mut sda_driver =
        SlaterDeterminantRepAnalysisDriver::<UnitaryRepresentedSymmetryGroup, C128>::builder()
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
        .symmetry_transformation_kind(SymmetryTransformationKind::SpinSpatial)
        .write_overlap_eigenvalues(false)
        .write_character_table(None)
        .build()
        .unwrap();

    let mut sda_driver =
        SlaterDeterminantRepAnalysisDriver::<MagneticRepresentedSymmetryGroup, C128>::builder()
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
        DecomposedSymbol::<MullikenIrcorepSymbol>::new("||A|_(g)| ⊕ 2||E|_(g)|").unwrap()
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
        DecomposedSymbol::<MullikenIrcorepSymbol>::new("||A|_(g)| ⊕ ||E|_(g)|").unwrap()
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
        DecomposedSymbol::<MullikenIrcorepSymbol>::new("||A|_(g)| ⊕ 2||E|_(g)|").unwrap()
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
        DecomposedSymbol::<MullikenIrcorepSymbol>::new("||A|_(g)| ⊕ 2||E|_(g)|").unwrap()
    );
}
