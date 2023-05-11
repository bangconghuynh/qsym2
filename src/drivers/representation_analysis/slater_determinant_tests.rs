use nalgebra::{Point3, Vector3};
use ndarray::array;
use num_complex::Complex;

use crate::angmom::spinor_rotation_3d::SpinConstraint;
use crate::aux::ao_basis::{BasisAngularOrder, BasisAtom, BasisShell, CartOrder, ShellOrder};
use crate::drivers::representation_analysis::CharacterTableDisplay;
use crate::drivers::representation_analysis::slater_determinant::{
    SlaterDeterminantRepAnalysisDriver, SlaterDeterminantRepAnalysisParams,
};
use crate::drivers::symmetry_group_detection::{
    SymmetryGroupDetectionDriver, SymmetryGroupDetectionParams,
};
use crate::drivers::QSym2Driver;
use crate::symmetry::symmetry_group::{
    MagneticRepresentedSymmetryGroup, UnitaryRepresentedSymmetryGroup,
};
use crate::symmetry::symmetry_transformation::SymmetryTransformationKind;
use crate::target::determinant::SlaterDeterminant;

type C128 = Complex<f64>;

const ROOT: &str = env!("CARGO_MANIFEST_DIR");

#[test]
fn test_drivers_slater_determinant_analysis_vf6_magnetic_field() {
    log4rs::init_file("log4rs.yml", Default::default()).unwrap();
    let path: String = format!("{}{}", ROOT, "/tests/xyz/vf6.xyz");
    let pd_params = SymmetryGroupDetectionParams::builder()
        .moi_thresholds(&[1e-6])
        .distance_thresholds(&[1e-6])
        // .fictitious_magnetic_fields(Some(vec![(
        //     Point3::new(0.0, 0.0, 0.0),
        //     Vector3::new(1.0, 1.0, 1.0),
        // )]))
        .fictitious_origin_com(true)
        .time_reversal(true)
        .write_symmetry_elements(true)
        .build()
        .unwrap();
    let mut pd_driver = SymmetryGroupDetectionDriver::builder()
        .parameters(&pd_params)
        .xyz(Some(path.clone()))
        .build()
        .unwrap();
    assert!(pd_driver.run().is_ok());
    let pd_res = pd_driver.result().unwrap();
    let mol_vf6 = &pd_res.pre_symmetry.recentred_molecule;

    let bsc_d = BasisShell::new(2, ShellOrder::Cart(CartOrder::lex(2)));
    let bsp_s = BasisShell::new(0, ShellOrder::Pure(true));

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
    // ];
    ].mapv(|x| C128::from(x));

    // -------------------------------------
    // αdxy αdyy αdzz αdx2-y2 βdxz βdxx βdyz
    // -------------------------------------
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
        .mol(&mol_vf6)
        .spin_constraint(SpinConstraint::Unrestricted(2, false))
        .complex_symmetric(false)
        .threshold(1e-14)
        .build()
        .unwrap()
        .to_generalised()
        .into();

    let sda_params = SlaterDeterminantRepAnalysisParams::<C128>::builder()
        .integrality_threshold(1e-10)
        .linear_independence_threshold(1e-10)
        .analyse_mo_symmetries(true)
        .use_magnetic_group(true)
        .use_double_group(true)
        .symmetry_transformation_kind(SymmetryTransformationKind::SpinSpatial)
        .write_overlap_eigenvalues(true)
        .write_character_table(Some(CharacterTableDisplay::Symbolic))
        .build()
        .unwrap();

    let mut sda_driver =
        SlaterDeterminantRepAnalysisDriver::<MagneticRepresentedSymmetryGroup, C128>::builder()
        .parameters(&sda_params)
        .determinant(&det_d3_cg)
        .sao_spatial(&sao_spatial)
        .symmetry_group(&pd_res)
        .build()
        .unwrap();
    assert!(sda_driver.run().is_ok());
}
