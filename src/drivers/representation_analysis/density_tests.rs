// use env_logger;
use nalgebra::Point3;
use ndarray::array;
use serial_test::serial;

use crate::angmom::spinor_rotation_3d::SpinConstraint;
use crate::auxiliary::atom::{Atom, ElementMap};
use crate::auxiliary::geometry::Transform;
use crate::auxiliary::molecule::Molecule;
use crate::basis::ao::{BasisAngularOrder, BasisAtom, BasisShell, PureOrder, ShellOrder};
use crate::chartab::chartab_symbols::DecomposedSymbol;
use crate::drivers::representation_analysis::angular_function::AngularFunctionRepAnalysisParams;
use crate::drivers::representation_analysis::density::{
    DensityRepAnalysisDriver, DensityRepAnalysisParams,
};
use crate::drivers::representation_analysis::CharacterTableDisplay;
use crate::drivers::symmetry_group_detection::{
    SymmetryGroupDetectionDriver, SymmetryGroupDetectionParams,
};
use crate::drivers::QSym2Driver;
use crate::symmetry::symmetry_group::UnitaryRepresentedSymmetryGroup;
use crate::symmetry::symmetry_symbols::MullikenIrrepSymbol;
use crate::symmetry::symmetry_transformation::SymmetryTransformationKind;
use crate::target::determinant::SlaterDeterminant;

#[test]
#[cfg(feature = "integrals")]
#[serial]
fn test_drivers_density_analysis_s4_sqpl_pxpy() {
    use crate::basis::ao_integrals::*;
    use crate::integrals::shell_tuple::build_shell_tuple_collection;

    // log4rs::init_file("log4rs.yml", Default::default()).unwrap();
    let emap = ElementMap::new();
    let atm_s0 = Atom::from_xyz("S +1.0 +1.0 0.0", &emap, 1e-7).unwrap();
    let atm_s1 = Atom::from_xyz("S -1.0 +1.0 0.0", &emap, 1e-7).unwrap();
    let atm_s2 = Atom::from_xyz("S -1.0 -1.0 0.0", &emap, 1e-7).unwrap();
    let atm_s3 = Atom::from_xyz("S +1.0 -1.0 0.0", &emap, 1e-7).unwrap();

    let bsp_p = BasisShell::new(1, ShellOrder::Pure(PureOrder::increasingm(1)));

    let batm_s0 = BasisAtom::new(&atm_s0, &[bsp_p.clone()]);
    let batm_s1 = BasisAtom::new(&atm_s1, &[bsp_p.clone()]);
    let batm_s2 = BasisAtom::new(&atm_s2, &[bsp_p.clone()]);
    let batm_s3 = BasisAtom::new(&atm_s3, &[bsp_p.clone()]);

    let bao_s4 = BasisAngularOrder::new(&[batm_s0, batm_s1, batm_s2, batm_s3]);
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
        .molecule(Some(&mol_s4))
        .build()
        .unwrap();
    assert!(pd_driver.run().is_ok());
    let pd_res = pd_driver.result().unwrap();

    // -----------------
    // Orbital densities
    // -----------------

    // S0px
    #[rustfmt::skip]
    let calpha = array![
        [0.0], [0.0], [1.0],
        [0.0], [0.0], [0.0],
        [0.0], [0.0], [0.0],
        [0.0], [0.0], [0.0],
    ];
    // S0py
    #[rustfmt::skip]
    let cbeta = array![
        [1.0], [0.0], [0.0],
        [0.0], [0.0], [0.0],
        [0.0], [0.0], [0.0],
        [0.0], [0.0], [0.0],
    ];
    let oalpha = array![1.0];
    let obeta = array![1.0];
    let det_ru = SlaterDeterminant::<f64, SpinConstraint>::builder()
        .coefficients(&[calpha.clone(), cbeta.clone()])
        .occupations(&[oalpha, obeta])
        .baos(vec![&bao_s4])
        .mol(&mol_s4)
        .structure_constraint(SpinConstraint::Unrestricted(2, false))
        .complex_symmetric(false)
        .threshold(1e-14)
        .build()
        .unwrap();
    let dens_ru = det_ru.to_densities().unwrap();
    let dena_ru = &dens_ru[0];
    let denb_ru = &dens_ru[1];
    let dentot_ru = dena_ru + denb_ru;
    let denspin_ru = dena_ru - denb_ru;

    // ------
    // Metric
    // ------

    let gc = GaussianContraction::<f64, f64> {
        primitives: vec![(3.4252509140, 1.0)],
    };
    let bsc0 = BasisShellContraction::<f64, f64> {
        basis_shell: bsp_p.clone(),
        contraction: gc.clone(),
        cart_origin: Point3::new(1.0, 1.0, 0.0),
        k: None,
    };
    let bsc1 = BasisShellContraction::<f64, f64> {
        basis_shell: bsp_p.clone(),
        contraction: gc.clone(),
        cart_origin: Point3::new(-1.0, 1.0, 0.0),
        k: None,
    };
    let bsc2 = BasisShellContraction::<f64, f64> {
        basis_shell: bsp_p.clone(),
        contraction: gc.clone(),
        cart_origin: Point3::new(-1.0, -1.0, 0.0),
        k: None,
    };
    let bsc3 = BasisShellContraction::<f64, f64> {
        basis_shell: bsp_p.clone(),
        contraction: gc.clone(),
        cart_origin: Point3::new(1.0, -1.0, 0.0),
        k: None,
    };
    let bscs = BasisSet::new(vec![vec![bsc0], vec![bsc1], vec![bsc2], vec![bsc3]]);
    let stc = build_shell_tuple_collection![
        <s1, s2, s3, s4>;
        false, false, false, false;
        &bscs, &bscs, &bscs, &bscs;
        f64
    ];
    let ovs = stc.overlap([0, 0, 0, 0]);
    let sao_ru = &ovs[0];

    // ~~~~~~~~~~~~~~~~~~~~~~~~~
    // u D4h (ordinary, unitary)
    // ~~~~~~~~~~~~~~~~~~~~~~~~~
    let da_params = DensityRepAnalysisParams::<f64>::builder()
        .integrality_threshold(1e-10)
        .linear_independence_threshold(1e-10)
        .use_magnetic_group(None)
        .use_double_group(false)
        .use_cayley_table(true)
        .symmetry_transformation_kind(SymmetryTransformationKind::SpinSpatial)
        .write_character_table(Some(CharacterTableDisplay::Symbolic))
        .build()
        .unwrap();

    let mut da_driver = DensityRepAnalysisDriver::<UnitaryRepresentedSymmetryGroup, f64>::builder()
        .parameters(&da_params)
        .angular_function_parameters(&afa_params)
        .densities(vec![
            ("Alpha density".to_string(), dena_ru),
            ("Beta density".to_string(), denb_ru),
            ("Total density".to_string(), &dentot_ru),
            ("Spin density".to_string(), &denspin_ru),
        ])
        .sao_spatial_4c(sao_ru)
        .symmetry_group(pd_res)
        .build()
        .unwrap();
    assert!(da_driver.run().is_ok());
    assert_eq!(
        da_driver.result().unwrap().density_symmetries()[0],
        Ok(DecomposedSymbol::<MullikenIrrepSymbol>::new(
            "||A|_(1g)| ⊕ ||A|_(2g)| ⊕ ||B|_(1g)| ⊕ ||B|_(2g)| ⊕ 2||E|_(u)|"
        )
        .unwrap())
    );
    assert_eq!(
        da_driver.result().unwrap().density_symmetries()[1],
        Ok(DecomposedSymbol::<MullikenIrrepSymbol>::new(
            "||A|_(1g)| ⊕ ||A|_(2g)| ⊕ ||B|_(1g)| ⊕ ||B|_(2g)| ⊕ 2||E|_(u)|"
        )
        .unwrap())
    );
    assert_eq!(
        da_driver.result().unwrap().density_symmetries()[2],
        Ok(
            DecomposedSymbol::<MullikenIrrepSymbol>::new("||A|_(1g)| ⊕ ||B|_(2g)| ⊕ ||E|_(u)|")
                .unwrap()
        )
    );
    assert_eq!(
        da_driver.result().unwrap().density_symmetries()[3],
        Ok(
            DecomposedSymbol::<MullikenIrrepSymbol>::new("||A|_(2g)| ⊕ ||B|_(1g)| ⊕ ||E|_(u)|")
                .unwrap()
        )
    );
}
