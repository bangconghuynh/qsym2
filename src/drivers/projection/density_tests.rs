use approx::{assert_abs_diff_eq, assert_abs_diff_ne};
// use log4rs;
use ndarray::array;
use ndarray_linalg::Norm;

use crate::angmom::spinor_rotation_3d::SpinConstraint;
use crate::auxiliary::atom::{Atom, ElementMap};
use crate::auxiliary::geometry::Transform;
use crate::auxiliary::molecule::Molecule;
use crate::basis::ao::{BasisAngularOrder, BasisAtom, BasisShell, PureOrder, ShellOrder};
use crate::drivers::QSym2Driver;
use crate::drivers::projection::density::{DensityProjectionDriver, DensityProjectionParams};
use crate::drivers::representation_analysis::{
    CharacterTableDisplay, MagneticSymmetryAnalysisKind,
};
use crate::drivers::symmetry_group_detection::{
    SymmetryGroupDetectionDriver, SymmetryGroupDetectionParams,
};
use crate::symmetry::symmetry_group::UnitaryRepresentedSymmetryGroup;
use crate::symmetry::symmetry_transformation::SymmetryTransformationKind;
use crate::target::determinant::SlaterDeterminant;

#[test]
fn test_drivers_density_projection_s4_sqpl_pxpy() {
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

    // ~~~~~~~~~~~~~~~~~~~~~~~~~
    // u D4h (ordinary, unitary)
    // ~~~~~~~~~~~~~~~~~~~~~~~~~

    let dp_params = DensityProjectionParams::builder()
        .symbolic_projection_targets(Some(vec![
            "||A|_(1g)|".to_string(),
            "||A|_(2g)|".to_string(),
            "||B|_(1g)|".to_string(),
            "||B|_(2g)|".to_string(),
            "||E|_(g)|".to_string(),
            // "||A|_(1u)|".to_string(),
            // "||A|_(2u)|".to_string(),
            // "||B|_(1u)|".to_string(),
            // "||B|_(2u)|".to_string(),
            // "||E|_(u)|".to_string(),
        ]))
        .numeric_projection_targets(Some(vec![5, 6, 7, 8, 9]))
        .use_magnetic_group(None)
        .use_double_group(false)
        .symmetry_transformation_kind(SymmetryTransformationKind::Spatial)
        .write_character_table(Some(CharacterTableDisplay::Symbolic))
        .build()
        .unwrap();

    let mut dp_driver = DensityProjectionDriver::<UnitaryRepresentedSymmetryGroup, f64>::builder()
        .parameters(&dp_params)
        .densities(vec![
            ("Alpha density".to_string(), dena_ru),
            ("Beta density".to_string(), denb_ru),
            ("Total density".to_string(), &dentot_ru),
            ("Spin density".to_string(), &denspin_ru),
        ])
        .symmetry_group(pd_res)
        .build()
        .unwrap();
    assert!(dp_driver.run().is_ok());

    let dena_ru_a1g = dp_driver.result().unwrap().projected_densities[0]
        .1
        .get_index(0)
        .unwrap()
        .1
        .as_ref()
        .unwrap();
    assert_abs_diff_ne!(dena_ru_a1g.density_matrix().norm_l2(), 0.0, epsilon = 1e-7);
    let dena_ru_a2g = dp_driver.result().unwrap().projected_densities[0]
        .1
        .get_index(1)
        .unwrap()
        .1
        .as_ref()
        .unwrap();
    assert_abs_diff_ne!(dena_ru_a2g.density_matrix().norm_l2(), 0.0, epsilon = 1e-7);
    let dena_ru_b1g = dp_driver.result().unwrap().projected_densities[0]
        .1
        .get_index(2)
        .unwrap()
        .1
        .as_ref()
        .unwrap();
    assert_abs_diff_ne!(dena_ru_b1g.density_matrix().norm_l2(), 0.0, epsilon = 1e-7);
    let dena_ru_b2g = dp_driver.result().unwrap().projected_densities[0]
        .1
        .get_index(3)
        .unwrap()
        .1
        .as_ref()
        .unwrap();
    assert_abs_diff_ne!(dena_ru_b2g.density_matrix().norm_l2(), 0.0, epsilon = 1e-7);
    let dena_ru_eg = dp_driver.result().unwrap().projected_densities[0]
        .1
        .get_index(4)
        .unwrap()
        .1
        .as_ref()
        .unwrap();
    assert_abs_diff_eq!(dena_ru_eg.density_matrix().norm_l2(), 0.0, epsilon = 1e-7);
    let dena_ru_a1u = dp_driver.result().unwrap().projected_densities[0]
        .1
        .get_index(5)
        .unwrap()
        .1
        .as_ref()
        .unwrap();
    assert_abs_diff_eq!(dena_ru_a1u.density_matrix().norm_l2(), 0.0, epsilon = 1e-7);
    let dena_ru_a2u = dp_driver.result().unwrap().projected_densities[0]
        .1
        .get_index(6)
        .unwrap()
        .1
        .as_ref()
        .unwrap();
    assert_abs_diff_eq!(dena_ru_a2u.density_matrix().norm_l2(), 0.0, epsilon = 1e-7);
    let dena_ru_b1u = dp_driver.result().unwrap().projected_densities[0]
        .1
        .get_index(7)
        .unwrap()
        .1
        .as_ref()
        .unwrap();
    assert_abs_diff_eq!(dena_ru_b1u.density_matrix().norm_l2(), 0.0, epsilon = 1e-7);
    let dena_ru_b2u = dp_driver.result().unwrap().projected_densities[0]
        .1
        .get_index(8)
        .unwrap()
        .1
        .as_ref()
        .unwrap();
    assert_abs_diff_eq!(dena_ru_b2u.density_matrix().norm_l2(), 0.0, epsilon = 1e-7);
    let dena_ru_eu = dp_driver.result().unwrap().projected_densities[0]
        .1
        .get_index(9)
        .unwrap()
        .1
        .as_ref()
        .unwrap();
    assert_abs_diff_ne!(dena_ru_eu.density_matrix().norm_l2(), 0.0, epsilon = 1e-7);

    // ~~~~~~~~~~~~~~~~~~~~~~
    // u D4h' (grey, unitary)
    // ~~~~~~~~~~~~~~~~~~~~~~

    let dp_params = DensityProjectionParams::builder()
        .symbolic_projection_targets(Some(vec![
            "|^(+)|A|_(1g)|".to_string(),
            "|^(+)|A|_(2g)|".to_string(),
            "|^(+)|B|_(1g)|".to_string(),
            "|^(+)|B|_(2g)|".to_string(),
            "|^(+)|E|_(g)|".to_string(),
            "|^(+)|A|_(1u)|".to_string(),
            "|^(+)|A|_(2u)|".to_string(),
            "|^(+)|B|_(1u)|".to_string(),
            "|^(+)|B|_(2u)|".to_string(),
            "|^(+)|E|_(u)|".to_string(),
            "|^(-)|A|_(1g)|".to_string(),
            "|^(-)|A|_(2g)|".to_string(),
            "|^(-)|B|_(1g)|".to_string(),
            "|^(-)|B|_(2g)|".to_string(),
            "|^(-)|E|_(g)|".to_string(),
            "|^(-)|A|_(1u)|".to_string(),
            "|^(-)|A|_(2u)|".to_string(),
            "|^(-)|B|_(1u)|".to_string(),
            "|^(-)|B|_(2u)|".to_string(),
            "|^(-)|E|_(u)|".to_string(),
        ]))
        .use_magnetic_group(Some(MagneticSymmetryAnalysisKind::Representation))
        .use_double_group(false)
        .symmetry_transformation_kind(SymmetryTransformationKind::Spatial)
        .write_character_table(Some(CharacterTableDisplay::Symbolic))
        .build()
        .unwrap();

    let mut dp_driver = DensityProjectionDriver::<UnitaryRepresentedSymmetryGroup, f64>::builder()
        .parameters(&dp_params)
        .densities(vec![
            ("Alpha density".to_string(), dena_ru),
            ("Beta density".to_string(), denb_ru),
            ("Total density".to_string(), &dentot_ru),
            ("Spin density".to_string(), &denspin_ru),
        ])
        .symmetry_group(pd_res)
        .build()
        .unwrap();
    assert!(dp_driver.run().is_ok());
}
