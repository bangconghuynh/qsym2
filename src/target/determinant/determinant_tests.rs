// use env_logger;
use itertools::Itertools;
use nalgebra::{Point3, Vector3};
use ndarray::{array, concatenate, s, Array2, Axis};
use ndarray_linalg::assert_close_l2;
use num_complex::Complex;
use num_traits::Pow;

use crate::analysis::{EigenvalueComparisonMode, Overlap, RepAnalysis};
use crate::angmom::spinor_rotation_3d::SpinConstraint;
use crate::auxiliary::atom::{Atom, ElementMap};
use crate::auxiliary::geometry::Transform;
use crate::auxiliary::molecule::Molecule;
use crate::basis::ao::{
    BasisAngularOrder, BasisAtom, BasisShell, CartOrder, PureOrder, ShellOrder,
};
use crate::chartab::chartab_symbols::DecomposedSymbol;
use crate::group::{GroupProperties, MagneticRepresentedGroup, UnitaryRepresentedGroup};
use crate::symmetry::symmetry_core::{PreSymmetry, Symmetry};
use crate::symmetry::symmetry_group::SymmetryGroupProperties;
use crate::symmetry::symmetry_symbols::{MullikenIrcorepSymbol, MullikenIrrepSymbol};
use crate::symmetry::symmetry_transformation::{
    SymmetryTransformable, SymmetryTransformationKind, TimeReversalTransformable,
};
use crate::target::determinant::determinant_analysis::SlaterDeterminantSymmetryOrbit;
use crate::target::determinant::SlaterDeterminant;

type C128 = Complex<f64>;

#[test]
fn test_determinant_transformation_bf4_sqpl() {
    // env_logger::init();
    let emap = ElementMap::new();
    let atm_b0 = Atom::from_xyz("B 0.0 0.0 0.0", &emap, 1e-7).unwrap();
    let atm_f0 = Atom::from_xyz("F 1.0 0.0 0.0", &emap, 1e-7).unwrap();
    let atm_f1 = Atom::from_xyz("F 0.0 1.0 0.0", &emap, 1e-7).unwrap();
    let atm_f2 = Atom::from_xyz("F -1.0 0.0 0.0", &emap, 1e-7).unwrap();
    let atm_f3 = Atom::from_xyz("F 0.0 -1.0 0.0", &emap, 1e-7).unwrap();

    let bss_p = BasisShell::new(0, ShellOrder::Pure(PureOrder::increasingm(0)));
    let bsp_c = BasisShell::new(1, ShellOrder::Cart(CartOrder::lex(1)));
    let bsd_c = BasisShell::new(2, ShellOrder::Cart(CartOrder::lex(2)));

    let batm_b0 = BasisAtom::new(&atm_b0, &[bss_p.clone(), bsp_c.clone(), bsd_c.clone()]);
    let batm_f0 = BasisAtom::new(&atm_f0, &[bss_p.clone(), bsp_c.clone(), bsd_c.clone()]);
    let batm_f1 = BasisAtom::new(&atm_f1, &[bss_p.clone(), bsp_c.clone(), bsd_c.clone()]);
    let batm_f2 = BasisAtom::new(&atm_f2, &[bss_p.clone(), bsp_c.clone(), bsd_c]);
    let batm_f3 = BasisAtom::new(&atm_f3, &[bss_p, bsp_c]);

    let bao_bf4 = BasisAngularOrder::new(&[batm_b0, batm_f0, batm_f1, batm_f2, batm_f3]);
    let mol_bf4 = Molecule::from_atoms(
        &[
            atm_b0.clone(),
            atm_f0.clone(),
            atm_f1.clone(),
            atm_f2.clone(),
            atm_f3.clone(),
        ],
        1e-7,
    );

    #[rustfmt::skip]
    let calpha = array![
        [ 1.0, 1.0],
        [ 1.0, 0.0], [0.0, 1.0], [0.0, 0.0],
        [-1.0, 0.0], [0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 0.0], [0.0, 0.0],
        [ 0.0, 0.0],
        [ 1.0, 0.0], [0.0, 0.0], [0.0, 0.0],
        [ 0.0, 0.0], [1.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 1.0], [0.0, 0.0],
        [ 0.0, 0.0],
        [ 0.0, 0.0], [0.0, 1.0], [0.0, 0.0],
        [ 0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0],
        [ 0.0, 0.0],
        [ 1.0, 0.0], [0.0, 0.0], [0.0, 0.0],
        [ 0.0, 0.0], [1.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 1.0], [0.0, 0.0],
        [ 0.0, 0.0],
        [ 0.0, 0.0], [0.0, 1.0], [0.0, 0.0]
    ];
    let oalpha = array![1.0, 1.0];

    let det = SlaterDeterminant::<f64>::builder()
        .coefficients(&[calpha])
        .occupations(&[oalpha.clone()])
        .bao(&bao_bf4)
        .mol(&mol_bf4)
        .spin_constraint(SpinConstraint::Restricted(2))
        .complex_symmetric(false)
        .threshold(1e-14)
        .build()
        .unwrap();

    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol_bf4)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false).unwrap();
    let group = UnitaryRepresentedGroup::from_molecular_symmetry(&sym, None).unwrap();

    let c4p1 = group.get_index(1).unwrap();
    let tdet_c4p1 = det.sym_transform_spatial(&c4p1).unwrap();
    #[rustfmt::skip]
    let tcalpha_ref = array![
        [1.0,  1.0],
        [0.0, -1.0], [ 1.0, 0.0], [0.0,  0.0],
        [1.0,  0.0], [ 0.0, 0.0], [0.0,  0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, 0.0],
        [0.0,  0.0],
        [0.0, -1.0], [ 0.0, 0.0], [0.0,  0.0],
        [0.0,  0.0],
        [0.0,  0.0], [ 1.0, 0.0], [0.0,  0.0],
        [0.0,  0.0], [-1.0, 0.0], [0.0, -1.0], [ 0.0, 0.0], [0.0, 0.0], [0.0, 0.0],
        [0.0,  0.0],
        [0.0, -1.0], [ 0.0, 0.0], [0.0,  0.0],
        [0.0,  0.0], [-1.0, 0.0], [0.0,  0.0], [ 0.0, 0.0], [0.0, 1.0], [0.0, 0.0],
        [0.0,  0.0],
        [0.0,  0.0], [ 1.0, 0.0], [0.0,  0.0],
        [0.0,  0.0], [-1.0, 0.0], [0.0, -1.0], [ 0.0, 0.0], [0.0, 0.0], [0.0, 0.0]
    ];
    let tdet_c4p1_ref = SlaterDeterminant::<f64>::builder()
        .coefficients(&[tcalpha_ref])
        .occupations(&[oalpha])
        .bao(&bao_bf4)
        .mol(&mol_bf4)
        .spin_constraint(SpinConstraint::Restricted(2))
        .complex_symmetric(false)
        .threshold(1e-14)
        .build()
        .unwrap();
    assert_eq!(tdet_c4p1, tdet_c4p1_ref);
}

#[test]
fn test_determinant_transformation_s4_sqpl() {
    // env_logger::init();
    let emap = ElementMap::new();
    let atm_s0 = Atom::from_xyz("S +1.0 +1.0 0.0", &emap, 1e-7).unwrap();
    let atm_s1 = Atom::from_xyz("S -1.0 +1.0 0.0", &emap, 1e-7).unwrap();
    let atm_s2 = Atom::from_xyz("S -1.0 -1.0 0.0", &emap, 1e-7).unwrap();
    let atm_s3 = Atom::from_xyz("S +1.0 -1.0 0.0", &emap, 1e-7).unwrap();

    let bsd_p = BasisShell::new(2, ShellOrder::Pure(PureOrder::increasingm(2)));

    let batm_s0 = BasisAtom::new(&atm_s0, &[bsd_p.clone()]);
    let batm_s1 = BasisAtom::new(&atm_s1, &[bsd_p.clone()]);
    let batm_s2 = BasisAtom::new(&atm_s2, &[bsd_p.clone()]);
    let batm_s3 = BasisAtom::new(&atm_s3, &[bsd_p]);

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

    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol_s4)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false).unwrap();
    let group = UnitaryRepresentedGroup::from_molecular_symmetry(&sym, None).unwrap();

    // Unrestricted spin constraint
    #[rustfmt::skip]
    let calpha = array![
        [ 1.0], [ 0.0], [ 0.0], [ 0.0], [ 0.0],
        [-1.0], [ 0.0], [ 0.0], [ 0.0], [ 0.0],
        [ 1.0], [ 0.0], [ 0.0], [ 0.0], [ 0.0],
        [-1.0], [ 0.0], [ 0.0], [ 0.0], [ 0.0]
    ];
    #[rustfmt::skip]
    let cbeta = array![
        [ 0.0], [ 0.0], [ 0.0], [ 0.0], [ 1.0],
        [ 0.0], [ 0.0], [ 0.0], [ 0.0], [ 1.0],
        [ 0.0], [ 0.0], [ 0.0], [ 0.0], [ 1.0],
        [ 0.0], [ 0.0], [ 0.0], [ 0.0], [ 1.0]
    ];
    let oalpha = array![1.0];
    let obeta = array![0.5];
    let detunres = SlaterDeterminant::<f64>::builder()
        .coefficients(&[calpha, cbeta])
        .occupations(&[oalpha.clone(), obeta.clone()])
        .bao(&bao_s4)
        .mol(&mol_s4)
        .spin_constraint(SpinConstraint::Unrestricted(2, false))
        .complex_symmetric(false)
        .threshold(1e-14)
        .build()
        .unwrap();

    let c4p1 = group.get_index(1).unwrap();
    #[rustfmt::skip]
    let tcalpha_ref = array![
        [ 1.0], [0.0], [0.0], [0.0], [0.0],
        [-1.0], [0.0], [0.0], [0.0], [0.0],
        [ 1.0], [0.0], [0.0], [0.0], [0.0],
        [-1.0], [0.0], [0.0], [0.0], [0.0]
    ];
    #[rustfmt::skip]
    let tcbeta_ref = -array![
        [ 0.0], [0.0], [0.0], [0.0], [1.0],
        [ 0.0], [0.0], [0.0], [0.0], [1.0],
        [ 0.0], [0.0], [0.0], [0.0], [1.0],
        [ 0.0], [0.0], [0.0], [0.0], [1.0]
    ];
    let tdetunres_c4p1_ref = SlaterDeterminant::<f64>::builder()
        .coefficients(&[tcalpha_ref, tcbeta_ref])
        .occupations(&[oalpha, obeta])
        .bao(&bao_s4)
        .mol(&mol_s4)
        .spin_constraint(SpinConstraint::Unrestricted(2, false))
        .complex_symmetric(false)
        .threshold(1e-14)
        .build()
        .unwrap();
    let tdetunres_c4p1 = detunres.sym_transform_spatial(&c4p1).unwrap();
    assert_eq!(tdetunres_c4p1, tdetunres_c4p1_ref);

    // Generalised spin constraint
    #[rustfmt::skip]
    let calpha2 = array![
        [0.0], [ 0.0], [0.0], [ 1.0], [0.0],
        [0.0], [ 0.0], [0.0], [-1.0], [0.0],
        [0.0], [ 0.0], [0.0], [-1.0], [0.0],
        [0.0], [ 0.0], [0.0], [ 1.0], [0.0]
    ];
    let calpha2_gen = concatenate!(Axis(0), calpha2, Array2::zeros((20, 1)));
    #[rustfmt::skip]
    let cbeta2 = array![
        [0.0], [ 1.0], [0.0], [ 0.0], [0.0],
        [0.0], [ 1.0], [0.0], [ 0.0], [0.0],
        [0.0], [-1.0], [0.0], [ 0.0], [0.0],
        [0.0], [-1.0], [0.0], [ 0.0], [0.0]
    ];
    let cbeta2_gen = concatenate!(Axis(0), Array2::zeros((20, 1)), cbeta2);
    let cgen = concatenate![Axis(1), calpha2_gen, cbeta2_gen];
    let ogen = array![0.5, 1.0];
    let detgen = SlaterDeterminant::<f64>::builder()
        .coefficients(&[cgen.clone()])
        .occupations(&[ogen.clone()])
        .bao(&bao_s4)
        .mol(&mol_s4)
        .spin_constraint(SpinConstraint::Generalised(2, false))
        .complex_symmetric(false)
        .threshold(1e-14)
        .build()
        .unwrap();

    let tcalpha2_gen = concatenate!(Axis(0), cbeta2, Array2::zeros((20, 1)));
    let tcbeta2_gen = concatenate!(Axis(0), Array2::zeros((20, 1)), calpha2);
    let tcgen_ref = concatenate![Axis(1), tcalpha2_gen, tcbeta2_gen];
    let tdetgen_c4p1_ref = SlaterDeterminant::<f64>::builder()
        .coefficients(&[tcgen_ref])
        .occupations(&[ogen.clone()])
        .bao(&bao_s4)
        .mol(&mol_s4)
        .spin_constraint(SpinConstraint::Generalised(2, false))
        .complex_symmetric(false)
        .threshold(1e-14)
        .build()
        .unwrap();
    let tdetgen_c4p1 = detgen.sym_transform_spatial(&c4p1).unwrap();
    assert_eq!(tdetgen_c4p1, tdetgen_c4p1_ref);

    // S1(+0.000, +0.000, +1.000)
    let s1zp1 = group.get_index(11).unwrap();
    let tdetgen_s1zp1_ref = SlaterDeterminant::<f64>::builder()
        .coefficients(&[-cgen.clone()])
        .occupations(&[ogen.clone()])
        .bao(&bao_s4)
        .mol(&mol_s4)
        .spin_constraint(SpinConstraint::Generalised(2, false))
        .complex_symmetric(false)
        .threshold(1e-14)
        .build()
        .unwrap();
    let tdetgen_s1zp1 = detgen.sym_transform_spatial(&s1zp1).unwrap();
    assert_eq!(tdetgen_s1zp1, tdetgen_s1zp1_ref);

    // S1(+0.000, +1.000, +0.000)
    let s1yp1 = group.get_index(12).unwrap();
    let tdetgen_s1yp1_ref = SlaterDeterminant::<f64>::builder()
        .coefficients(&[cgen.clone()])
        .occupations(&[ogen.clone()])
        .bao(&bao_s4)
        .mol(&mol_s4)
        .spin_constraint(SpinConstraint::Generalised(2, false))
        .complex_symmetric(false)
        .threshold(1e-14)
        .build()
        .unwrap();
    let tdetgen_s1yp1 = detgen.sym_transform_spatial(&s1yp1).unwrap();
    assert_eq!(tdetgen_s1yp1, tdetgen_s1yp1_ref);

    // S1(+1.000, +0.000, +0.000)
    let s1xp1 = group.get_index(13).unwrap();
    let tdetgen_s1xp1_ref = SlaterDeterminant::<f64>::builder()
        .coefficients(&[cgen.clone()])
        .occupations(&[ogen.clone()])
        .bao(&bao_s4)
        .mol(&mol_s4)
        .spin_constraint(SpinConstraint::Generalised(2, false))
        .complex_symmetric(false)
        .threshold(1e-14)
        .build()
        .unwrap();
    let tdetgen_s1xp1 = detgen.sym_transform_spatial(&s1xp1).unwrap();
    assert_eq!(tdetgen_s1xp1, tdetgen_s1xp1_ref);

    // i
    let ip1 = group.get_index(8).unwrap();
    let tdetgen_ip1_ref = SlaterDeterminant::<f64>::builder()
        .coefficients(&[-cgen])
        .occupations(&[ogen.clone()])
        .bao(&bao_s4)
        .mol(&mol_s4)
        .spin_constraint(SpinConstraint::Generalised(2, false))
        .complex_symmetric(false)
        .threshold(1e-14)
        .build()
        .unwrap();
    let tdetgen_ip1 = detgen.sym_transform_spatial(&ip1).unwrap();
    assert_eq!(tdetgen_ip1, tdetgen_ip1_ref);

    // S4(+0.000, +0.000, +1.000)
    let s4p1 = group.get_index(9).unwrap();
    let tcgen_s4p1_ref = concatenate![Axis(1), -tcalpha2_gen, -tcbeta2_gen];
    let tdetgen_s4p1_ref = SlaterDeterminant::<f64>::builder()
        .coefficients(&[tcgen_s4p1_ref])
        .occupations(&[ogen])
        .bao(&bao_s4)
        .mol(&mol_s4)
        .spin_constraint(SpinConstraint::Generalised(2, false))
        .complex_symmetric(false)
        .threshold(1e-14)
        .build()
        .unwrap();
    let tdetgen_s4p1 = detgen.sym_transform_spatial(&s4p1).unwrap();
    assert_eq!(tdetgen_s4p1, tdetgen_s4p1_ref);
}

#[test]
fn test_determinant_transformation_b3_real_timerev() {
    // env_logger::init();
    let emap = ElementMap::new();
    let atm_b0 = Atom::from_xyz("B 0.0 0.0 0.0", &emap, 1e-7).unwrap();
    let atm_b1 = Atom::from_xyz("B 1.0 0.0 0.0", &emap, 1e-7).unwrap();
    let atm_b2 = Atom::new_ordinary("B", Point3::new(0.5, 3.0f64.sqrt() / 2.0, 0.0), &emap, 1e-7);

    let bss_p = BasisShell::new(0, ShellOrder::Pure(PureOrder::increasingm(0)));
    let bsp_c = BasisShell::new(1, ShellOrder::Cart(CartOrder::lex(1)));

    let batm_b0 = BasisAtom::new(&atm_b0, &[bss_p.clone(), bsp_c.clone()]);
    let batm_b1 = BasisAtom::new(&atm_b1, &[bss_p.clone(), bsp_c.clone()]);
    let batm_b2 = BasisAtom::new(&atm_b2, &[bss_p, bsp_c]);

    let bao_b3 = BasisAngularOrder::new(&[batm_b0, batm_b1, batm_b2]);
    let mol_b3 =
        Molecule::from_atoms(&[atm_b0.clone(), atm_b1.clone(), atm_b2.clone()], 1e-7).recentre();

    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol_b3)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false).unwrap();
    let group = UnitaryRepresentedGroup::from_molecular_symmetry(&sym, None).unwrap();
    let c3p1 = group.get_index(1).unwrap();

    let sqr = 3.0f64.sqrt() / 2.0;
    #[rustfmt::skip]
    let calpha = array![
        [ 1.0,  0.0],
        [ sqr,  1.0], [-0.5,  0.0], [0.0, 0.0],
        [ 1.0,  0.0],
        [ 0.0,  1.0], [ 1.0,  0.0], [0.0, 0.0],
        [ 1.0,  0.0],
        [-sqr, -1.0], [-0.5,  0.0], [0.0, 0.0]
    ];
    #[rustfmt::skip]
    let cbeta = array![
        [ 0.0,  0.0],
        [-0.5,  0.0], [-sqr,  1.0], [0.0, 0.0],
        [ 0.0,  0.0],
        [ 1.0,  0.0], [ 0.0,  1.0], [0.0, 0.0],
        [ 0.0,  0.0],
        [-0.5,  0.0], [ sqr, -1.0], [0.0, 0.0]
    ];

    // Reference coefficients for c3p1 with time reversal
    #[rustfmt::skip]
    let tcalpha_ref = array![
        [ 0.0,  0.0],
        [ 0.5, -sqr], [ sqr, -0.5], [0.0, 0.0],
        [ 0.0,  0.0],
        [-1.0,  sqr], [ 0.0,  0.5], [0.0, 0.0],
        [ 0.0,  0.0],
        [ 0.5,  sqr], [-sqr,  0.5], [0.0, 0.0]
    ];
    #[rustfmt::skip]
    let tcbeta_ref = array![
        [ 1.0,  0.0],
        [ sqr,  0.5], [-0.5, -sqr], [0.0, 0.0],
        [ 1.0,  0.0],
        [ 0.0, -0.5], [ 1.0,  sqr], [0.0, 0.0],
        [ 1.0,  0.0],
        [-sqr, -0.5], [-0.5,  sqr], [0.0, 0.0]
    ];

    // Unrestricted spin constraint, spin-projection-pure orbitals
    let oalpha = array![0.5, 0.5];
    let obeta = array![0.75, 0.25];

    let detunres = SlaterDeterminant::<f64>::builder()
        .coefficients(&[calpha.clone(), cbeta.clone()])
        .occupations(&[oalpha.clone(), obeta.clone()])
        .bao(&bao_b3)
        .mol(&mol_b3)
        .spin_constraint(SpinConstraint::Unrestricted(2, false))
        .complex_symmetric(false)
        .threshold(1e-14)
        .build()
        .unwrap();

    let tdetunres_tr = detunres.transform_timerev().unwrap();
    let tdetunres_tr_ref = SlaterDeterminant::<f64>::builder()
        .coefficients(&[-cbeta.clone(), calpha.clone()])
        .occupations(&[obeta.clone(), oalpha.clone()])
        .bao(&bao_b3)
        .mol(&mol_b3)
        .spin_constraint(SpinConstraint::Unrestricted(2, false))
        .complex_symmetric(false)
        .threshold(1e-14)
        .build()
        .unwrap();
    assert_eq!(tdetunres_tr, tdetunres_tr_ref);

    let tdetunres_c3p1_tr = detunres
        .sym_transform_spatial(&c3p1)
        .unwrap()
        .transform_timerev()
        .unwrap();
    let tdetunres_c3p1_tr_ref = SlaterDeterminant::<f64>::builder()
        .coefficients(&[tcalpha_ref.clone(), tcbeta_ref.clone()])
        .occupations(&[obeta, oalpha])
        .bao(&bao_b3)
        .mol(&mol_b3)
        .spin_constraint(SpinConstraint::Unrestricted(2, false))
        .complex_symmetric(false)
        .threshold(1e-14)
        .build()
        .unwrap();
    assert_eq!(tdetunres_c3p1_tr, tdetunres_c3p1_tr_ref);

    // Generalised spin constraint, spin-projection-mixed orbitals
    let cgen = concatenate![Axis(0), calpha, cbeta];
    let ogen = array![1.0, 1.0];

    let detgen = SlaterDeterminant::<f64>::builder()
        .coefficients(&[cgen])
        .occupations(&[ogen.clone()])
        .bao(&bao_b3)
        .mol(&mol_b3)
        .spin_constraint(SpinConstraint::Generalised(2, false))
        .complex_symmetric(false)
        .threshold(1e-14)
        .build()
        .unwrap();

    let tdetgen_tr = detgen.transform_timerev().unwrap();
    let tcgen_ref = concatenate![Axis(0), -cbeta, calpha];
    let tdetgen_tr_ref = SlaterDeterminant::<f64>::builder()
        .coefficients(&[tcgen_ref])
        .occupations(&[ogen.clone()])
        .bao(&bao_b3)
        .mol(&mol_b3)
        .spin_constraint(SpinConstraint::Generalised(2, false))
        .complex_symmetric(false)
        .threshold(1e-14)
        .build()
        .unwrap();
    assert_eq!(tdetgen_tr, tdetgen_tr_ref);

    let tdetgen_c3p1_tr = detgen
        .sym_transform_spatial(&c3p1)
        .unwrap()
        .transform_timerev()
        .unwrap();
    let tcgen_ref = concatenate![Axis(0), tcalpha_ref, tcbeta_ref];
    let tdetgen_c3p1_tr_ref = SlaterDeterminant::<f64>::builder()
        .coefficients(&[tcgen_ref])
        .occupations(&[ogen])
        .bao(&bao_b3)
        .mol(&mol_b3)
        .spin_constraint(SpinConstraint::Generalised(2, false))
        .complex_symmetric(false)
        .threshold(1e-14)
        .build()
        .unwrap();
    assert_eq!(tdetgen_c3p1_tr, tdetgen_c3p1_tr_ref);
}

#[test]
fn test_determinant_transformation_c2_complex_timerev() {
    // env_logger::init();
    let emap = ElementMap::new();
    let atm_c0 = Atom::from_xyz("C 1.0 0.0 0.0", &emap, 1e-7).unwrap();
    let atm_c1 = Atom::from_xyz("C 0.0 0.0 0.0", &emap, 1e-7).unwrap();

    let bsp_c = BasisShell::new(1, ShellOrder::Cart(CartOrder::lex(1)));

    let batm_c0 = BasisAtom::new(&atm_c0, &[bsp_c.clone()]);
    let batm_c1 = BasisAtom::new(&atm_c1, &[bsp_c]);

    let bao_c2 = BasisAngularOrder::new(&[batm_c0, batm_c1]);
    let mol_c2 = Molecule::from_atoms(&[atm_c0.clone(), atm_c1.clone()], 1e-7).recentre();

    #[rustfmt::skip]
    let calpha = array![
        [C128::new( 1.0, 0.0), C128::new(1.0,  2.5)], [C128::from(0.0), C128::new(0.0, -3.4)], [C128::from(0.0), C128::from(0.2)],
        [C128::new(-1.0, 0.0), C128::new(1.0, -2.5)], [C128::from(0.0), C128::new(0.0, -3.4)], [C128::from(0.0), C128::from(0.2)]
    ];
    let calpha_gen = concatenate!(Axis(0), calpha, Array2::zeros((6, 2)));
    #[rustfmt::skip]
    let cbeta = array![
        [C128::new( 1.0, 2.0), C128::new(2.0,  3.9)], [C128::from(1.0), C128::new(0.0,  0.0)], [C128::from(0.0), C128::from(0.0)],
        [C128::new( 2.0, 4.0), C128::new(2.0, -3.9)], [C128::from(1.0), C128::new(0.0,  0.0)], [C128::from(0.0), C128::from(0.0)]
    ];
    let cbeta_gen = concatenate!(Axis(0), Array2::zeros((6, 2)), cbeta);
    let cgen = concatenate![Axis(1), calpha_gen, cbeta_gen];
    let ogen = array![1.0, 1.0, 1.0, 1.0];

    let detgen = SlaterDeterminant::<C128>::builder()
        .coefficients(&[cgen])
        .occupations(&[ogen.clone()])
        .bao(&bao_c2)
        .mol(&mol_c2)
        .spin_constraint(SpinConstraint::Generalised(2, false))
        .complex_symmetric(false)
        .threshold(1e-14)
        .build()
        .unwrap();

    let tcalpha_gen_ref = concatenate!(Axis(0), Array2::zeros((6, 2)), calpha.map(|x| x.conj()));
    let tcbeta_gen_ref = concatenate!(Axis(0), -cbeta.map(|x| x.conj()), Array2::zeros((6, 2)));
    let tcgen_ref = concatenate![Axis(1), tcalpha_gen_ref, tcbeta_gen_ref];
    let tdetgen_tr_ref = SlaterDeterminant::<C128>::builder()
        .coefficients(&[tcgen_ref])
        .occupations(&[ogen])
        .bao(&bao_c2)
        .mol(&mol_c2)
        .spin_constraint(SpinConstraint::Generalised(2, false))
        .complex_symmetric(false)
        .threshold(1e-14)
        .build()
        .unwrap();
    let tdetgen_tr = detgen.transform_timerev().unwrap();
    assert_eq!(tdetgen_tr, tdetgen_tr_ref);
}

#[test]
fn test_determinant_transformation_c3_spin_rotation() {
    // env_logger::init();
    let emap = ElementMap::new();
    let atm_c0 = Atom::from_xyz("C 1.0 0.0 0.0", &emap, 1e-7).unwrap();
    let atm_c1 = Atom::from_xyz("C 0.0 1.0 0.0", &emap, 1e-7).unwrap();
    let atm_c2 = Atom::from_xyz("C 0.0 0.0 0.0", &emap, 1e-7).unwrap();

    let bss_p = BasisShell::new(0, ShellOrder::Pure(PureOrder::increasingm(0)));
    let bsp_c = BasisShell::new(1, ShellOrder::Cart(CartOrder::lex(1)));

    let batm_c0 = BasisAtom::new(&atm_c0, &[bss_p.clone(), bsp_c.clone()]);
    let batm_c1 = BasisAtom::new(&atm_c1, &[bss_p.clone(), bsp_c.clone()]);
    let batm_c2 = BasisAtom::new(&atm_c2, &[bss_p, bsp_c]);

    let bao_c3 = BasisAngularOrder::new(&[batm_c0, batm_c1, batm_c2]);
    let mol_c3 =
        Molecule::from_atoms(&[atm_c0.clone(), atm_c1.clone(), atm_c2.clone()], 1e-7).recentre();
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol_c3)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false).unwrap();
    let group = UnitaryRepresentedGroup::from_molecular_symmetry(&sym, None)
        .unwrap()
        .to_double_group()
        .unwrap();

    #[rustfmt::skip]
    let calpha = array![
        [1.0, 0.0],
        [0.0, 1.0], [0.0, 0.0], [ 0.0, 0.0],
        [0.0, 0.0],
        [1.0, 1.0], [0.0, 0.0], [ 0.0, 0.0],
        [0.0, 0.0],
        [0.0, 1.0], [1.0, 0.0], [ 0.0, 0.0]
    ];
    let calpha_gen = concatenate!(Axis(0), calpha, Array2::zeros((12, 2)));
    #[rustfmt::skip]
    let cbeta = array![
        [0.0, 0.0],
        [0.0, 0.0], [0.0, 0.0], [ 1.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0], [0.0, 0.0], [-1.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0], [1.0, 0.0], [ 1.0, 0.0]
    ];
    let cbeta_gen = concatenate!(Axis(0), Array2::zeros((12, 2)), cbeta);
    let cgen = concatenate![Axis(1), calpha_gen, cbeta_gen];
    let ogen = array![1.0, 1.0, 1.0, 1.0];
    let detgen: SlaterDeterminant<C128> = SlaterDeterminant::<f64>::builder()
        .coefficients(&[cgen.clone()])
        .occupations(&[ogen.clone()])
        .bao(&bao_c3)
        .mol(&mol_c3)
        .spin_constraint(SpinConstraint::Generalised(2, false))
        .complex_symmetric(false)
        .threshold(1e-14)
        .build()
        .unwrap()
        .into();

    // ----------------
    // Proper rotations
    // ----------------

    let sqr = 2.0f64.sqrt() / 2.0;
    let c2_nsr_p1 = group.get_index(2).unwrap();
    let tcalpha_gen = concatenate!(
        Axis(0),
        Array2::zeros((12, 2)),
        C128::new(1.0, -1.0) * (calpha.clone() * sqr).map(C128::from)
    );
    let tcbeta_gen = concatenate!(
        Axis(0),
        -C128::new(1.0, 1.0) * (cbeta.clone() * sqr).map(C128::from),
        Array2::zeros((12, 2)),
    );
    let tcgen_ref = concatenate![Axis(1), tcalpha_gen, tcbeta_gen];
    let tdetgen_c2_nsr_p1_ref = SlaterDeterminant::<C128>::builder()
        .coefficients(&[tcgen_ref.clone()])
        .occupations(&[ogen.clone()])
        .bao(&bao_c3)
        .mol(&mol_c3)
        .spin_constraint(SpinConstraint::Generalised(2, false))
        .complex_symmetric(false)
        .threshold(1e-14)
        .build()
        .unwrap();
    let tdetgen_c2_nsr_p1 = detgen.sym_transform_spin(&c2_nsr_p1).unwrap();
    assert_eq!(tdetgen_c2_nsr_p1, tdetgen_c2_nsr_p1_ref);

    let c2_nsr_p2 = (&c2_nsr_p1).pow(2);
    let tdetgen_c2_nsr_p2_ref: SlaterDeterminant<C128> = SlaterDeterminant::<f64>::builder()
        .coefficients(&[-cgen])
        .occupations(&[ogen.clone()])
        .bao(&bao_c3)
        .mol(&mol_c3)
        .spin_constraint(SpinConstraint::Generalised(2, false))
        .complex_symmetric(false)
        .threshold(1e-14)
        .build()
        .unwrap()
        .into();
    let tdetgen_c2_nsr_p2 = detgen.sym_transform_spin(&c2_nsr_p2).unwrap();
    assert_eq!(tdetgen_c2_nsr_p2, tdetgen_c2_nsr_p2_ref);

    let e_isr = group.get_index(1).unwrap();
    let tdetgen_e_isr = detgen.sym_transform_spin(&e_isr).unwrap();
    assert_eq!(tdetgen_e_isr, tdetgen_c2_nsr_p2_ref);

    let c2_nsr_p3 = (&c2_nsr_p1).pow(3);
    let tdetgen_c2_nsr_p3_ref = SlaterDeterminant::<C128>::builder()
        .coefficients(&[-tcgen_ref])
        .occupations(&[ogen.clone()])
        .bao(&bao_c3)
        .mol(&mol_c3)
        .spin_constraint(SpinConstraint::Generalised(2, false))
        .complex_symmetric(false)
        .threshold(1e-14)
        .build()
        .unwrap();
    let tdetgen_c2_nsr_p3 = detgen.sym_transform_spin(&c2_nsr_p3).unwrap();
    assert_eq!(tdetgen_c2_nsr_p3, tdetgen_c2_nsr_p3_ref);

    let c2_nsr_p4 = (&c2_nsr_p1).pow(4);
    let tdetgen_c2_nsr_p4 = detgen.sym_transform_spin(&c2_nsr_p4).unwrap();
    assert_eq!(tdetgen_c2_nsr_p4, detgen);

    // ------------------
    // Improper rotations
    // ------------------
    let sxy_tcalpha_gen = concatenate!(
        Axis(0),
        C128::new(0.0, -1.0) * calpha.map(C128::from),
        Array2::zeros((12, 2))
    );
    let sxy_tcbeta_gen = concatenate!(
        Axis(0),
        Array2::zeros((12, 2)),
        C128::new(0.0, 1.0) * cbeta.map(C128::from)
    );
    let sxy_tcgen_ref = concatenate![Axis(1), sxy_tcalpha_gen, sxy_tcbeta_gen];

    let sxy_nsr_p1 = group.get_index(4).unwrap();
    let tdetgen_sxy_nsr_ref = SlaterDeterminant::<C128>::builder()
        .coefficients(&[sxy_tcgen_ref.clone()])
        .occupations(&[ogen.clone()])
        .bao(&bao_c3)
        .mol(&mol_c3)
        .spin_constraint(SpinConstraint::Generalised(2, false))
        .complex_symmetric(false)
        .threshold(1e-14)
        .build()
        .unwrap();
    let tdetgen_sxy_nsr = detgen.sym_transform_spin(&sxy_nsr_p1).unwrap();
    assert_eq!(tdetgen_sxy_nsr, tdetgen_sxy_nsr_ref);

    let tdetgen_sxy_nsr_p2 = detgen.sym_transform_spin(&(&sxy_nsr_p1).pow(2)).unwrap();
    assert_eq!(tdetgen_sxy_nsr_p2, tdetgen_e_isr);

    let tdetgen_sxy_nsr_p3_ref = SlaterDeterminant::<C128>::builder()
        .coefficients(&[-sxy_tcgen_ref])
        .occupations(&[ogen.clone()])
        .bao(&bao_c3)
        .mol(&mol_c3)
        .spin_constraint(SpinConstraint::Generalised(2, false))
        .complex_symmetric(false)
        .threshold(1e-14)
        .build()
        .unwrap();
    let tdetgen_sxy_nsr_p3 = detgen.sym_transform_spin(&(&sxy_nsr_p1).pow(3)).unwrap();
    assert_eq!(tdetgen_sxy_nsr_p3, tdetgen_sxy_nsr_p3_ref);

    let tcalpha_sxyz_gen = concatenate!(
        Axis(0),
        Array2::zeros((12, 2)),
        -C128::new(1.0, 1.0) * (calpha * sqr).map(C128::from)
    );
    let tcbeta_sxyz_gen = concatenate!(
        Axis(0),
        C128::new(1.0, -1.0) * (cbeta * sqr).map(C128::from),
        Array2::zeros((12, 2)),
    );
    let tcgen_sxyz_ref = concatenate![Axis(1), tcalpha_sxyz_gen, tcbeta_sxyz_gen];
    let tdetgen_sxyz_nsr_p1_ref = SlaterDeterminant::<C128>::builder()
        .coefficients(&[tcgen_sxyz_ref])
        .occupations(&[ogen])
        .bao(&bao_c3)
        .mol(&mol_c3)
        .spin_constraint(SpinConstraint::Generalised(2, false))
        .complex_symmetric(false)
        .threshold(1e-14)
        .build()
        .unwrap();
    let sxyz_nsr_p1 = group.get_index(6).unwrap();
    let tdetgen_sxyz_nsr_p1 = detgen.sym_transform_spin(&sxyz_nsr_p1).unwrap();
    assert_eq!(tdetgen_sxyz_nsr_p1, tdetgen_sxyz_nsr_p1_ref);
}

#[test]
fn test_determinant_transformation_h4_spin_spatial_rotation_composition() {
    // env_logger::init();
    let emap = ElementMap::new();
    let atm_h0 = Atom::from_xyz("H 0.0 0.0 0.0", &emap, 1e-7).unwrap();
    let atm_h1 = Atom::from_xyz("H 1.0 0.0 0.0", &emap, 1e-7).unwrap();
    let atm_h2 = Atom::from_xyz("H 1.0 1.0 0.0", &emap, 1e-7).unwrap();
    let atm_h3 = Atom::from_xyz("H 0.0 1.0 0.0", &emap, 1e-7).unwrap();

    let bss_p = BasisShell::new(0, ShellOrder::Pure(PureOrder::increasingm(0)));
    let bsp_c = BasisShell::new(1, ShellOrder::Cart(CartOrder::lex(1)));
    let bsd_c = BasisShell::new(2, ShellOrder::Cart(CartOrder::lex(2)));

    let batm_h0 = BasisAtom::new(&atm_h0, &[bss_p.clone(), bsp_c.clone(), bsd_c.clone()]);
    let batm_h1 = BasisAtom::new(&atm_h1, &[bss_p.clone(), bsp_c.clone(), bsd_c.clone()]);
    let batm_h2 = BasisAtom::new(&atm_h2, &[bss_p.clone(), bsp_c.clone(), bsd_c.clone()]);
    let batm_h3 = BasisAtom::new(&atm_h3, &[bss_p, bsp_c, bsd_c]);

    let bao_h4 = BasisAngularOrder::new(&[batm_h0, batm_h1, batm_h2, batm_h3]);

    let mol_h4 = Molecule::from_atoms(
        &[
            atm_h0.clone(),
            atm_h1.clone(),
            atm_h2.clone(),
            atm_h3.clone(),
        ],
        1e-7,
    )
    .recentre();

    #[rustfmt::skip]
    let calpha = array![
        [1.0, 0.0],
        [0.0, 1.0], [0.0, 0.0], [ 0.0, 0.0],
        [1.0, 0.0], [0.0, 1.0], [ 0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [ 0.0, 1.0],
        [0.0, 0.0],
        [1.0, 1.0], [0.0, 0.0], [ 0.0, 0.0],
        [0.0, 1.0], [0.0, 1.0], [ 1.0, 0.0], [1.0, 0.0], [0.0, 0.0], [ 0.0, 1.0],
        [0.0, 0.0],
        [0.0, 1.0], [1.0, 0.0], [ 0.0, 0.0],
        [1.0, 0.0], [0.0, 1.0], [ 0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [ 0.0, 1.0],
        [0.0, 1.0],
        [0.0, 0.0], [0.0, 1.0], [ 0.0, 0.0],
        [1.0, 0.0], [0.0, 0.0], [ 0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [ 0.0, 1.0],
    ];
    let calpha_gen = concatenate!(Axis(0), calpha, Array2::zeros((40, 2)));
    #[rustfmt::skip]
    let cbeta = array![
        [0.0, 0.0],
        [0.0, 0.0], [0.0, 0.0], [ 1.0, 0.0],
        [0.0, 1.0], [0.0, 0.0], [ 0.0, 1.0], [1.0, 0.0], [0.0, 0.0], [ 0.0, 1.0],
        [0.0, 0.0],
        [0.0, 0.0], [0.0, 0.0], [-1.0, 0.0],
        [0.0, -1.0], [0.0, 0.0], [ 1.0, 0.0], [0.0, 0.0], [0.0, 1.0], [ 0.0, 1.0],
        [0.0, 0.0],
        [0.0, 0.0], [1.0, 0.0], [ 1.0, 0.0],
        [1.0, 0.0], [1.0, 0.0], [ -1.0, 0.0], [0.0, 0.0], [1.0, 1.0], [ 1.0, 1.0],
        [1.0, 0.0],
        [0.0, 0.0], [0.0, 0.0], [-1.0, 0.0],
        [0.0, 1.0], [0.0, 0.0], [ -1.0, 0.0], [0.0, 0.0], [0.0, 1.0], [ 0.0, 1.0],
    ];
    let cbeta_gen = concatenate!(Axis(0), Array2::zeros((40, 2)), cbeta);
    let cgen = concatenate![Axis(1), calpha_gen, cbeta_gen];
    let ogen = array![1.0, 1.0, 1.0, 1.0];
    let detgen: SlaterDeterminant<C128> = SlaterDeterminant::<f64>::builder()
        .coefficients(&[cgen])
        .occupations(&[ogen])
        .bao(&bao_h4)
        .mol(&mol_h4)
        .spin_constraint(SpinConstraint::Generalised(2, false))
        .complex_symmetric(false)
        .threshold(1e-14)
        .build()
        .unwrap()
        .into();

    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol_h4)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false).unwrap();
    let group = UnitaryRepresentedGroup::from_molecular_symmetry(&sym, None)
        .unwrap()
        .to_double_group()
        .unwrap();

    let elements_i = group.elements();
    let elements_j = group.elements();
    elements_i
        .into_iter()
        .cartesian_product(elements_j)
        .for_each(|(op_i, op_j)| {
            let op_k = op_i * op_j;

            let spatial_tdetgen_ij = detgen
                .sym_transform_spatial(op_j)
                .unwrap()
                .sym_transform_spatial(op_i)
                .unwrap();
            let spatial_tdetgen_k = detgen.sym_transform_spatial(&op_k).unwrap();
            assert_eq!(spatial_tdetgen_k, spatial_tdetgen_ij);

            let spin_tdetgen_ij = detgen
                .sym_transform_spin(op_j)
                .unwrap()
                .sym_transform_spin(op_i)
                .unwrap();
            let spin_tdetgen_k = detgen.sym_transform_spin(&op_k).unwrap();
            assert_eq!(spin_tdetgen_k, spin_tdetgen_ij);

            let spin_spatial_tdetgen_ij = detgen
                .sym_transform_spin_spatial(op_j)
                .unwrap()
                .sym_transform_spin_spatial(op_i)
                .unwrap();
            let spin_spatial_tdetgen_k = detgen.sym_transform_spin_spatial(&op_k).unwrap();
            assert_eq!(spin_spatial_tdetgen_k, spin_spatial_tdetgen_ij);
        });
}

#[test]
fn test_determinant_analysis_overlap() {
    let emap = ElementMap::new();
    let atm_h0 = Atom::from_xyz("H -0.550 +1.100 +0.000", &emap, 1e-7).unwrap();
    let atm_h1 = Atom::from_xyz("H +0.550 +1.100 +0.000", &emap, 1e-7).unwrap();
    let atm_h2 = Atom::from_xyz("H +0.550 +0.000 +0.000", &emap, 1e-7).unwrap();
    let atm_h3 = Atom::from_xyz("H -0.550 +0.000 +0.000", &emap, 1e-7).unwrap();

    let bss_p = BasisShell::new(0, ShellOrder::Pure(PureOrder::increasingm(0)));

    let batm_h0 = BasisAtom::new(&atm_h0, &[bss_p.clone()]);
    let batm_h1 = BasisAtom::new(&atm_h1, &[bss_p.clone()]);
    let batm_h2 = BasisAtom::new(&atm_h2, &[bss_p.clone()]);
    let batm_h3 = BasisAtom::new(&atm_h3, &[bss_p]);

    let bao_h4 = BasisAngularOrder::new(&[batm_h0, batm_h1, batm_h2, batm_h3]);
    let mol_h4 = Molecule::from_atoms(
        &[
            atm_h0.clone(),
            atm_h1.clone(),
            atm_h2.clone(),
            atm_h3.clone(),
        ],
        1e-7,
    )
    .recentre();

    // SAO from libint2
    let sao = array![
        [1.00000000, 0.43958641, 0.23697772, 0.43958641],
        [0.43958641, 1.00000000, 0.43958641, 0.23697772],
        [0.23697772, 0.43958641, 1.00000000, 0.43958641],
        [0.43958641, 0.23697772, 0.43958641, 1.00000000],
    ];

    // State 0
    #[rustfmt::skip]
    let ca0 = array![
        [0.343_713_607_477_907_8, -0.572_402_330_060_102_3],
        [0.343_713_607_477_907_75, -0.572_402_330_060_099_8],
        [0.343_713_606_198_103_26,  0.572_402_330_828_593_2],
        [0.343_713_606_198_102_9,  0.572_402_330_828_592],
    ];
    let oa0 = array![1.0, 1.0];
    let det0 = SlaterDeterminant::<f64>::builder()
        .coefficients(&[ca0.clone(), ca0])
        .occupations(&[oa0.clone(), oa0])
        .bao(&bao_h4)
        .mol(&mol_h4)
        .spin_constraint(SpinConstraint::Unrestricted(2, false))
        .complex_symmetric(false)
        .threshold(1e-14)
        .build()
        .unwrap();

    // State 1
    #[rustfmt::skip]
    let ca1 = array![
        [-0.572_402_331_217_283_7, -0.835_885_639_699_404_5],
        [ 0.572_402_331_217_297_4,  0.835_885_639_699_397_1],
        [ 0.572_402_329_671_392_3, -0.835_885_640_757_999_9],
        [-0.572_402_329_671_413_7,  0.835_885_640_758_005_6],
    ];
    let oa1 = array![1.0, 1.0];
    let det1 = SlaterDeterminant::<f64>::builder()
        .coefficients(&[ca1.clone(), ca1])
        .occupations(&[oa1.clone(), oa1])
        .bao(&bao_h4)
        .mol(&mol_h4)
        .spin_constraint(SpinConstraint::Unrestricted(2, false))
        .complex_symmetric(false)
        .threshold(1e-14)
        .build()
        .unwrap();

    // State 2
    #[rustfmt::skip]
    let ca2 = array![
        [-0.572_403_072_220_630_5, -0.572_401_589_195_638_3],
        [-0.572_401_588_106_471_9,  0.572_403_073_309_794_1],
        [ 0.572_403_072_781_256_6,  0.572_401_587_577_937_2],
        [ 0.572_401_588_667_104_1, -0.572_403_071_692_093],
    ];
    let oa2 = array![1.0, 1.0];
    let det2 = SlaterDeterminant::<f64>::builder()
        .coefficients(&[ca2.clone(), ca2])
        .occupations(&[oa2.clone(), oa2])
        .bao(&bao_h4)
        .mol(&mol_h4)
        .spin_constraint(SpinConstraint::Unrestricted(2, false))
        .complex_symmetric(false)
        .threshold(1e-14)
        .build()
        .unwrap();

    // State 3
    #[rustfmt::skip]
    let ca3 = array![
        [0.343_713_607_416_821_7,  0.835_885_639_938_028],
        [0.343_713_607_416_821_94, -0.835_885_639_938_027_6],
        [0.343_713_606_259_189_23,  0.835_885_640_519_376_1],
        [0.343_713_606_259_188_73, -0.835_885_640_519_376_2],
    ];
    let oa3 = array![1.0, 1.0];
    let det3 = SlaterDeterminant::<f64>::builder()
        .coefficients(&[ca3.clone(), ca3])
        .occupations(&[oa3.clone(), oa3])
        .bao(&bao_h4)
        .mol(&mol_h4)
        .spin_constraint(SpinConstraint::Unrestricted(2, false))
        .complex_symmetric(false)
        .threshold(1e-14)
        .build()
        .unwrap();

    // State 4
    #[rustfmt::skip]
    let ca4 = array![
        [ 0.572_402_331_415_958_3, -0.835_885_641_713_886_2],
        [ 0.572_402_331_415_968_4,  0.835_885_641_713_886_7],
        [-0.572_402_329_472_725_2, -0.835_885_638_743_517_1],
        [-0.572_402_329_472_735_3,  0.835_885_638_743_518_5],
    ];
    let oa4 = array![1.0, 1.0];
    let det4 = SlaterDeterminant::<f64>::builder()
        .coefficients(&[ca4.clone(), ca4])
        .occupations(&[oa4.clone(), oa4])
        .bao(&bao_h4)
        .mol(&mol_h4)
        .spin_constraint(SpinConstraint::Unrestricted(2, false))
        .complex_symmetric(false)
        .threshold(1e-14)
        .build()
        .unwrap();

    // State 5
    #[rustfmt::skip]
    let ca5 = array![
        [ 0.540_339_823_859_483_1, -0.937_623_077_414_211_1],
        [-0.695_157_956_994_567_2,  0.634_095_514_495_094_8],
        [-0.149_776_107_799_195_78, -0.656_469_860_326_082_8],
        [ 0.685_480_503_996_699_8,  0.983_834_663_329_823_1],
    ];
    let oa5 = array![1.0, 1.0];
    let det5 = SlaterDeterminant::<f64>::builder()
        .coefficients(&[ca5.clone(), ca5])
        .occupations(&[oa5.clone(), oa5])
        .bao(&bao_h4)
        .mol(&mol_h4)
        .spin_constraint(SpinConstraint::Unrestricted(2, false))
        .complex_symmetric(false)
        .threshold(1e-14)
        .build()
        .unwrap();

    // Real, UHF
    let mut smat = Array2::<f64>::zeros((6, 6));
    [&det0, &det1, &det2, &det3, &det4, &det5]
        .iter()
        .enumerate()
        .combinations_with_replacement(2)
        .for_each(|pair| {
            let (i, deti) = pair[0];
            let (j, detj) = pair[1];
            smat[(i, j)] = deti.overlap(detj, Some(&sao), None).unwrap();
            if i != j {
                smat[(j, i)] = detj.overlap(deti, Some(&sao), None).unwrap();
            }
        });
    let smat_ref = array![
        [1.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0050563],
        [0.0000000, 1.0000000, 0.0000000, 0.0000000, 0.0000000, 0.7611240],
        [0.0000000, 0.0000000, 1.0000000, 0.0000000, 0.0000000, 0.0576832],
        [0.0000000, 0.0000000, 0.0000000, 1.0000000, 0.0000000, 0.0719396],
        [0.0000000, 0.0000000, 0.0000000, 0.0000000, 1.0000000, 0.1041424],
        [0.0050563, 0.7611240, 0.0576832, 0.0719396, 0.1041424, 1.0000000],
    ];
    assert_close_l2!(&smat, &smat_ref, 1e-7);

    // Real, GHF
    let nbas = sao.nrows();
    let mut sao_g = Array2::<f64>::zeros((2 * nbas, 2 * nbas));
    sao_g.slice_mut(s![0..nbas, 0..nbas]).assign(&sao);
    sao_g
        .slice_mut(s![nbas..(2 * nbas), nbas..(2 * nbas)])
        .assign(&sao);
    let mut smat_g = Array2::<f64>::zeros((6, 6));
    [&det0, &det1, &det2, &det3, &det4, &det5]
        .iter()
        .map(|det| det.to_generalised())
        .enumerate()
        .combinations_with_replacement(2)
        .for_each(|pair| {
            let i = pair[0].0;
            let deti = &pair[0].1;
            let j = pair[1].0;
            let detj = &pair[1].1;
            smat_g[(i, j)] = deti.overlap(detj, Some(&sao_g), None).unwrap();
            if i != j {
                smat_g[(j, i)] = detj.overlap(deti, Some(&sao_g), None).unwrap();
            }
        });
    assert_close_l2!(&smat_g, &smat_ref, 1e-7);

    // Complex, UHF
    let sao_c = sao.mapv(C128::from);
    let mut smat_c = Array2::<C128>::zeros((6, 6));
    [&det0, &det1, &det2, &det3, &det4, &det5]
        .iter()
        .map(|&det| SlaterDeterminant::<C128>::from(det.clone()))
        .enumerate()
        .combinations_with_replacement(2)
        .for_each(|pair| {
            let i = pair[0].0;
            let deti = &pair[0].1;
            let j = pair[1].0;
            let detj = &pair[1].1;
            smat_c[(i, j)] = deti.overlap(detj, Some(&sao_c), None).unwrap();
            if i != j {
                smat_c[(j, i)] = detj.overlap(deti, Some(&sao_c), None).unwrap();
            }
        });
    let smat_c_ref = smat_ref.mapv(C128::from);
    assert_close_l2!(&smat_c, &smat_c_ref, 1e-7);

    // Complex, GHF
    let sao_cg = sao_g.mapv(C128::from);
    let mut smat_cg = Array2::<C128>::zeros((6, 6));
    [&det0, &det1, &det2, &det3, &det4, &det5]
        .iter()
        .map(|&det| SlaterDeterminant::<C128>::from(det.clone()).to_generalised())
        .enumerate()
        .combinations_with_replacement(2)
        .for_each(|pair| {
            let i = pair[0].0;
            let deti = &pair[0].1;
            let j = pair[1].0;
            let detj = &pair[1].1;
            smat_cg[(i, j)] = deti.overlap(detj, Some(&sao_cg), None).unwrap();
            if i != j {
                smat_cg[(j, i)] = detj.overlap(deti, Some(&sao_cg), None).unwrap();
            }
        });
    assert_close_l2!(&smat_cg, &smat_c_ref, 1e-7);
}

#[test]
fn test_determinant_orbit_mat_s4_sqpl_s() {
    // env_logger::init();
    let emap = ElementMap::new();
    let atm_s0 = Atom::from_xyz("S +1.0 +1.0 0.0", &emap, 1e-7).unwrap();
    let atm_s1 = Atom::from_xyz("S -1.0 +1.0 0.0", &emap, 1e-7).unwrap();
    let atm_s2 = Atom::from_xyz("S -1.0 -1.0 0.0", &emap, 1e-7).unwrap();
    let atm_s3 = Atom::from_xyz("S +1.0 -1.0 0.0", &emap, 1e-7).unwrap();

    let bss_p = BasisShell::new(0, ShellOrder::Pure(PureOrder::increasingm(0)));

    let batm_s0 = BasisAtom::new(&atm_s0, &[bss_p.clone()]);
    let batm_s1 = BasisAtom::new(&atm_s1, &[bss_p.clone()]);
    let batm_s2 = BasisAtom::new(&atm_s2, &[bss_p.clone()]);
    let batm_s3 = BasisAtom::new(&atm_s3, &[bss_p]);

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

    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol_s4)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false).unwrap();
    let group = UnitaryRepresentedGroup::from_molecular_symmetry(&sym, None).unwrap();

    #[rustfmt::skip]
    let calpha = array![
        [1.0],
        [0.0],
        [0.0],
        [0.0],
    ];
    #[rustfmt::skip]
    let cbeta = array![
        [1.0],
        [0.0],
        [0.0],
        [0.0],
    ];
    let oalpha = array![1.0];
    let obeta = array![1.0];
    let det = SlaterDeterminant::<f64>::builder()
        .coefficients(&[calpha, cbeta])
        .occupations(&[oalpha, obeta])
        .bao(&bao_s4)
        .mol(&mol_s4)
        .spin_constraint(SpinConstraint::Unrestricted(2, false))
        .complex_symmetric(false)
        .threshold(1e-14)
        .build()
        .unwrap();

    let mut orbit = SlaterDeterminantSymmetryOrbit::builder()
        .group(&group)
        .origin(&det)
        .integrality_threshold(1e-14)
        .linear_independence_threshold(1e-14)
        .symmetry_transformation_kind(SymmetryTransformationKind::Spatial)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
        .build()
        .unwrap();

    let sao = Array2::<f64>::eye(4);
    let _ = orbit
        .calc_smat(Some(&sao), None, true)
        .unwrap()
        .calc_xmat(false);
    let smat = orbit.smat().unwrap().clone();
    let xmat = orbit.xmat();

    let os = xmat.t().dot(&smat).dot(xmat);
    assert_eq!(os.shape(), &[4, 4]);
    assert_close_l2!(&os, &Array2::<f64>::eye(os.shape()[0]), 1e-7);

    let det_c = SlaterDeterminant::<C128>::from(det.clone());
    let sao_c = sao.mapv(C128::from);
    let mut orbit_c = SlaterDeterminantSymmetryOrbit::builder()
        .group(&group)
        .origin(&det_c)
        .integrality_threshold(1e-14)
        .linear_independence_threshold(1e-14)
        .symmetry_transformation_kind(SymmetryTransformationKind::Spatial)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
        .build()
        .unwrap();
    let _ = orbit_c
        .calc_smat(Some(&sao_c), None, true)
        .unwrap()
        .calc_xmat(false);
    let smat_c = orbit_c.smat().unwrap().clone();
    let xmat_c = orbit_c.xmat();

    let os_c = xmat_c.t().mapv(|x| x.conj()).dot(&smat_c).dot(xmat_c);
    assert_eq!(os_c.shape(), &[4, 4]);
    assert_close_l2!(&os_c, &Array2::<C128>::eye(os.shape()[0]), 1e-7);

    assert_close_l2!(&os.map(C128::from), &os_c, 1e-7);
}

#[test]
fn test_determinant_orbit_rep_analysis_s4_sqpl_pz() {
    // env_logger::init();
    let emap = ElementMap::new();
    let atm_s0 = Atom::from_xyz("S +1.0 +1.0 0.0", &emap, 1e-7).unwrap();
    let atm_s1 = Atom::from_xyz("S -1.0 +1.0 0.0", &emap, 1e-7).unwrap();
    let atm_s2 = Atom::from_xyz("S -1.0 -1.0 0.0", &emap, 1e-7).unwrap();
    let atm_s3 = Atom::from_xyz("S +1.0 -1.0 0.0", &emap, 1e-7).unwrap();

    let bsp_p = BasisShell::new(1, ShellOrder::Pure(PureOrder::increasingm(1)));

    let batm_s0 = BasisAtom::new(&atm_s0, &[bsp_p.clone()]);
    let batm_s1 = BasisAtom::new(&atm_s1, &[bsp_p.clone()]);
    let batm_s2 = BasisAtom::new(&atm_s2, &[bsp_p.clone()]);
    let batm_s3 = BasisAtom::new(&atm_s3, &[bsp_p]);

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

    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol_s4)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false).unwrap();
    let group_u_d4h = UnitaryRepresentedGroup::from_molecular_symmetry(&sym, None).unwrap();
    let group_u_d4h_double = group_u_d4h.to_double_group().unwrap();

    let mut sym_tr = Symmetry::new();
    sym_tr.analyse(&presym, true).unwrap();
    let group_u_grey_d4h = UnitaryRepresentedGroup::from_molecular_symmetry(&sym_tr, None).unwrap();
    let group_u_grey_d4h_double = group_u_grey_d4h.to_double_group().unwrap();

    let group_m_grey_d4h =
        MagneticRepresentedGroup::from_molecular_symmetry(&sym_tr, None).unwrap();
    let group_m_grey_d4h_double = group_m_grey_d4h.to_double_group().unwrap();

    let mut mol_s4_bz = mol_s4.clone();
    mol_s4_bz.set_magnetic_field(Some(0.1 * Vector3::z()));
    let presym_bz = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol_s4_bz)
        .build()
        .unwrap();
    let mut sym_bz = Symmetry::new();
    sym_bz.analyse(&presym_bz, true).unwrap();
    let group_u_bw_d4h_c4h =
        UnitaryRepresentedGroup::from_molecular_symmetry(&sym_bz, None).unwrap();
    let group_u_bw_d4h_c4h_double = group_u_bw_d4h_c4h.to_double_group().unwrap();
    let group_m_bw_d4h_c4h =
        MagneticRepresentedGroup::from_molecular_symmetry(&sym_bz, None).unwrap();
    let group_m_bw_d4h_c4h_double = group_m_bw_d4h_c4h.to_double_group().unwrap();

    // ----------
    // 1-electron
    // ----------

    #[rustfmt::skip]
    let calpha = array![
        [0.0], [1.0], [0.0],
        [0.0], [0.0], [0.0],
        [0.0], [0.0], [0.0],
        [0.0], [0.0], [0.0],
    ];
    #[rustfmt::skip]
    let cbeta = array![
        [1.0], [0.0], [0.0],
        [0.0], [0.0], [0.0],
        [0.0], [0.0], [0.0],
        [0.0], [0.0], [0.0],
    ];
    let oalpha = array![1.0];
    let obeta_empty = array![0.0];
    let det_1e_cg: SlaterDeterminant<C128> = SlaterDeterminant::<f64>::builder()
        .coefficients(&[calpha.clone(), cbeta.clone()])
        .occupations(&[oalpha.clone(), obeta_empty])
        .bao(&bao_s4)
        .mol(&mol_s4)
        .spin_constraint(SpinConstraint::Unrestricted(2, false))
        .complex_symmetric(false)
        .threshold(1e-14)
        .build()
        .unwrap()
        .to_generalised()
        .into();

    // ----------
    // 2-electron
    // ----------

    let obeta_filled = array![1.0];
    let det_2e_cg: SlaterDeterminant<C128> = SlaterDeterminant::<f64>::builder()
        .coefficients(&[calpha, cbeta])
        .occupations(&[oalpha, obeta_filled])
        .bao(&bao_s4)
        .mol(&mol_s4)
        .spin_constraint(SpinConstraint::Unrestricted(2, false))
        .complex_symmetric(false)
        .threshold(1e-14)
        .build()
        .unwrap()
        .to_generalised()
        .into();

    let sao_cg = Array2::<C128>::eye(24);

    // ~~~~~~~~~~~~~~~~~~~~~~~~~
    // u D4h (ordinary, unitary)
    // ~~~~~~~~~~~~~~~~~~~~~~~~~

    let mut orbit_cg_u_d4h_spatial_1e = SlaterDeterminantSymmetryOrbit::builder()
        .group(&group_u_d4h)
        .origin(&det_1e_cg)
        .integrality_threshold(1e-14)
        .linear_independence_threshold(1e-14)
        .symmetry_transformation_kind(SymmetryTransformationKind::Spatial)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
        .build()
        .unwrap();
    let _ = orbit_cg_u_d4h_spatial_1e
        .calc_smat(Some(&sao_cg), None, true)
        .unwrap()
        .calc_xmat(false);
    assert_eq!(
        orbit_cg_u_d4h_spatial_1e.analyse_rep().unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new("||E|_(g)| ⊕ ||A|_(2u)| ⊕ ||B|_(1u)|")
            .unwrap()
    );

    let mut orbit_cg_u_d4h_spatial_2e = SlaterDeterminantSymmetryOrbit::builder()
        .group(&group_u_d4h)
        .origin(&det_2e_cg)
        .integrality_threshold(1e-14)
        .linear_independence_threshold(1e-14)
        .symmetry_transformation_kind(SymmetryTransformationKind::Spatial)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
        .build()
        .unwrap();
    let _ = orbit_cg_u_d4h_spatial_2e
        .calc_smat(Some(&sao_cg), None, true)
        .unwrap()
        .calc_xmat(false);
    assert_eq!(
        orbit_cg_u_d4h_spatial_2e.analyse_rep().unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new(
            "2||E|_(g)| ⊕ ||A|_(1u)| ⊕ ||A|_(2u)| ⊕ ||B|_(1u)| ⊕ ||B|_(2u)|"
        )
        .unwrap(),
    );

    let mut orbit_cg_u_d4h_spin_1e = SlaterDeterminantSymmetryOrbit::builder()
        .group(&group_u_d4h)
        .origin(&det_1e_cg)
        .integrality_threshold(1e-14)
        .linear_independence_threshold(1e-14)
        .symmetry_transformation_kind(SymmetryTransformationKind::Spin)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
        .build()
        .unwrap();
    let _ = orbit_cg_u_d4h_spin_1e
        .calc_smat(Some(&sao_cg), None, true)
        .unwrap()
        .calc_xmat(false);
    assert_eq!(
        orbit_cg_u_d4h_spin_1e.analyse_rep().unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new("||A|_(1g)|").unwrap()
    );

    let mut orbit_cg_u_d4h_spin_2e = SlaterDeterminantSymmetryOrbit::builder()
        .group(&group_u_d4h)
        .origin(&det_2e_cg)
        .integrality_threshold(1e-14)
        .linear_independence_threshold(1e-14)
        .symmetry_transformation_kind(SymmetryTransformationKind::Spin)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
        .build()
        .unwrap();
    let _ = orbit_cg_u_d4h_spin_2e
        .calc_smat(Some(&sao_cg), None, true)
        .unwrap()
        .calc_xmat(false);
    assert_eq!(
        orbit_cg_u_d4h_spin_2e.analyse_rep().unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new("||A|_(1g)|").unwrap()
    );

    let mut orbit_cg_u_d4h_spin_spatial_1e = SlaterDeterminantSymmetryOrbit::builder()
        .group(&group_u_d4h)
        .origin(&det_1e_cg)
        .integrality_threshold(1e-14)
        .linear_independence_threshold(1e-14)
        .symmetry_transformation_kind(SymmetryTransformationKind::SpinSpatial)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
        .build()
        .unwrap();
    let _ = orbit_cg_u_d4h_spin_spatial_1e
        .calc_smat(Some(&sao_cg), None, true)
        .unwrap()
        .calc_xmat(false);
    assert_eq!(
        orbit_cg_u_d4h_spin_spatial_1e.analyse_rep().unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new("||E|_(g)| ⊕ ||A|_(2u)| ⊕ ||B|_(1u)|")
            .unwrap()
    );

    let mut orbit_cg_u_d4h_spin_spatial_2e = SlaterDeterminantSymmetryOrbit::builder()
        .group(&group_u_d4h)
        .origin(&det_2e_cg)
        .integrality_threshold(1e-14)
        .linear_independence_threshold(1e-14)
        .symmetry_transformation_kind(SymmetryTransformationKind::SpinSpatial)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
        .build()
        .unwrap();
    let _ = orbit_cg_u_d4h_spin_spatial_2e
        .calc_smat(Some(&sao_cg), None, true)
        .unwrap()
        .calc_xmat(false);
    assert_eq!(
        orbit_cg_u_d4h_spin_spatial_2e.analyse_rep().unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new(
            "2||E|_(g)| ⊕ ||A|_(1u)| ⊕ ||A|_(2u)| ⊕ ||B|_(1u)| ⊕ ||B|_(2u)|"
        )
        .unwrap(),
    );

    // ~~~~~~~~~~~~~~~~~~~~~~
    // u D4h' (grey, unitary)
    // ~~~~~~~~~~~~~~~~~~~~~~

    let mut orbit_cg_u_grey_d4h_spatial_1e = SlaterDeterminantSymmetryOrbit::builder()
        .group(&group_u_grey_d4h)
        .origin(&det_1e_cg)
        .integrality_threshold(1e-14)
        .linear_independence_threshold(1e-14)
        .symmetry_transformation_kind(SymmetryTransformationKind::Spatial)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
        .build()
        .unwrap();
    let _ = orbit_cg_u_grey_d4h_spatial_1e
        .calc_smat(Some(&sao_cg), None, true)
        .unwrap()
        .calc_xmat(false);
    assert_eq!(
        orbit_cg_u_grey_d4h_spatial_1e.analyse_rep().unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new(
            "|^(+)|E|_(g)| ⊕ |^(+)|A|_(2u)| ⊕ |^(+)|B|_(1u)|"
        )
        .unwrap()
    );

    let mut orbit_cg_u_grey_d4h_spatial_2e = SlaterDeterminantSymmetryOrbit::builder()
        .group(&group_u_grey_d4h)
        .origin(&det_2e_cg)
        .integrality_threshold(1e-14)
        .linear_independence_threshold(1e-14)
        .symmetry_transformation_kind(SymmetryTransformationKind::Spatial)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
        .build()
        .unwrap();
    let _ = orbit_cg_u_grey_d4h_spatial_2e
        .calc_smat(Some(&sao_cg), None, true)
        .unwrap()
        .calc_xmat(false);
    assert_eq!(
        orbit_cg_u_grey_d4h_spatial_2e.analyse_rep().unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new(
            "2|^(+)|E|_(g)| ⊕ |^(+)|A|_(1u)| ⊕ |^(+)|A|_(2u)| ⊕ |^(+)|B|_(1u)| ⊕ |^(+)|B|_(2u)|"
        )
        .unwrap(),
    );

    let mut orbit_cg_u_grey_d4h_spin_1e = SlaterDeterminantSymmetryOrbit::builder()
        .group(&group_u_grey_d4h)
        .origin(&det_1e_cg)
        .integrality_threshold(1e-14)
        .linear_independence_threshold(1e-14)
        .symmetry_transformation_kind(SymmetryTransformationKind::Spin)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
        .build()
        .unwrap();
    let _ = orbit_cg_u_grey_d4h_spin_1e
        .calc_smat(Some(&sao_cg), None, true)
        .unwrap()
        .calc_xmat(false);
    // Character analysis would give ^(+)|A|_(1g) ⊕ ^(-)|A|_(1g), but |α⟩ and |β⟩ cannot be linearly
    // combined to span each of the subspaces separately.
    assert!(orbit_cg_u_grey_d4h_spin_1e.analyse_rep().is_err());

    let mut orbit_cg_u_grey_d4h_spin_2e = SlaterDeterminantSymmetryOrbit::builder()
        .group(&group_u_grey_d4h)
        .origin(&det_2e_cg)
        .integrality_threshold(1e-14)
        .linear_independence_threshold(1e-14)
        .symmetry_transformation_kind(SymmetryTransformationKind::Spin)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
        .build()
        .unwrap();
    let _ = orbit_cg_u_grey_d4h_spin_2e
        .calc_smat(Some(&sao_cg), None, true)
        .unwrap()
        .calc_xmat(false);
    assert_eq!(
        orbit_cg_u_grey_d4h_spin_2e.analyse_rep().unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new("|^(+)|A|_(1g)| ⊕ |^(-)|A|_(1g)|").unwrap(),
    );

    let mut orbit_cg_u_grey_d4h_spin_spatial_1e = SlaterDeterminantSymmetryOrbit::builder()
        .group(&group_u_grey_d4h)
        .origin(&det_1e_cg)
        .integrality_threshold(1e-14)
        .linear_independence_threshold(1e-14)
        .symmetry_transformation_kind(SymmetryTransformationKind::SpinSpatial)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
        .build()
        .unwrap();
    let _ = orbit_cg_u_grey_d4h_spin_spatial_1e
        .calc_smat(Some(&sao_cg), None, true)
        .unwrap()
        .calc_xmat(false);
    assert!(orbit_cg_u_grey_d4h_spin_spatial_1e.analyse_rep().is_err());

    let mut orbit_cg_u_grey_d4h_spin_spatial_2e = SlaterDeterminantSymmetryOrbit::builder()
        .group(&group_u_grey_d4h)
        .origin(&det_2e_cg)
        .integrality_threshold(1e-14)
        .linear_independence_threshold(1e-14)
        .symmetry_transformation_kind(SymmetryTransformationKind::SpinSpatial)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
        .build()
        .unwrap();
    let _ = orbit_cg_u_grey_d4h_spin_spatial_2e
        .calc_smat(Some(&sao_cg), None, true)
        .unwrap()
        .calc_xmat(false);
    assert_eq!(
        orbit_cg_u_grey_d4h_spin_spatial_2e.analyse_rep().unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new(
            "2|^(+)|E|_(g)| ⊕ |^(+)|A|_(1u)| ⊕ |^(+)|A|_(2u)| ⊕ |^(+)|B|_(1u)| ⊕ |^(+)|B|_(2u)|
            ⊕ 2|^(-)|E|_(g)| ⊕ |^(-)|A|_(1u)| ⊕ |^(-)|A|_(2u)| ⊕ |^(-)|B|_(1u)| ⊕ |^(-)|B|_(2u)|"
        )
        .unwrap(),
    );

    // ~~~~~~~~~~~~~~~~~~~~~~~
    // m D4h' (grey, magnetic)
    // ~~~~~~~~~~~~~~~~~~~~~~~

    let mut orbit_cg_m_grey_d4h_spatial_1e = SlaterDeterminantSymmetryOrbit::builder()
        .group(&group_m_grey_d4h)
        .origin(&det_1e_cg)
        .integrality_threshold(1e-14)
        .linear_independence_threshold(1e-14)
        .symmetry_transformation_kind(SymmetryTransformationKind::Spatial)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
        .build()
        .unwrap();
    let _ = orbit_cg_m_grey_d4h_spatial_1e
        .calc_smat(Some(&sao_cg), None, true)
        .unwrap()
        .calc_xmat(false);
    assert_eq!(
        orbit_cg_m_grey_d4h_spatial_1e.analyse_rep().unwrap(),
        DecomposedSymbol::<MullikenIrcorepSymbol>::new("||E|_(g)| ⊕ ||A|_(2u)| ⊕ ||B|_(1u)|")
            .unwrap()
    );

    let mut orbit_cg_m_grey_d4h_spatial_2e = SlaterDeterminantSymmetryOrbit::builder()
        .group(&group_m_grey_d4h)
        .origin(&det_2e_cg)
        .integrality_threshold(1e-14)
        .linear_independence_threshold(1e-14)
        .symmetry_transformation_kind(SymmetryTransformationKind::Spatial)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
        .build()
        .unwrap();
    let _ = orbit_cg_m_grey_d4h_spatial_2e
        .calc_smat(Some(&sao_cg), None, true)
        .unwrap()
        .calc_xmat(false);
    assert_eq!(
        orbit_cg_m_grey_d4h_spatial_2e.analyse_rep().unwrap(),
        DecomposedSymbol::<MullikenIrcorepSymbol>::new(
            "2||E|_(g)| ⊕ ||A|_(1u)| ⊕ ||A|_(2u)| ⊕ ||B|_(1u)| ⊕ ||B|_(2u)|"
        )
        .unwrap(),
    );

    let mut orbit_cg_m_grey_d4h_spin_1e = SlaterDeterminantSymmetryOrbit::builder()
        .group(&group_m_grey_d4h)
        .origin(&det_1e_cg)
        .integrality_threshold(1e-14)
        .linear_independence_threshold(1e-14)
        .symmetry_transformation_kind(SymmetryTransformationKind::Spin)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
        .build()
        .unwrap();
    let _ = orbit_cg_m_grey_d4h_spin_1e
        .calc_smat(Some(&sao_cg), None, true)
        .unwrap()
        .calc_xmat(false);
    assert_eq!(
        orbit_cg_m_grey_d4h_spin_1e.analyse_rep().unwrap(),
        DecomposedSymbol::<MullikenIrcorepSymbol>::new("2||A|_(1g)|").unwrap()
    );

    let mut orbit_cg_m_grey_d4h_spin_2e = SlaterDeterminantSymmetryOrbit::builder()
        .group(&group_m_grey_d4h)
        .origin(&det_2e_cg)
        .integrality_threshold(1e-14)
        .linear_independence_threshold(1e-14)
        .symmetry_transformation_kind(SymmetryTransformationKind::Spin)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
        .build()
        .unwrap();
    let _ = orbit_cg_m_grey_d4h_spin_2e
        .calc_smat(Some(&sao_cg), None, true)
        .unwrap()
        .calc_xmat(false);
    assert_eq!(
        orbit_cg_m_grey_d4h_spin_2e.analyse_rep().unwrap(),
        DecomposedSymbol::<MullikenIrcorepSymbol>::new("2||A|_(1g)|").unwrap(),
    );

    let mut orbit_cg_m_grey_d4h_spin_spatial_1e = SlaterDeterminantSymmetryOrbit::builder()
        .group(&group_m_grey_d4h)
        .origin(&det_1e_cg)
        .integrality_threshold(1e-14)
        .linear_independence_threshold(1e-14)
        .symmetry_transformation_kind(SymmetryTransformationKind::SpinSpatial)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
        .build()
        .unwrap();
    let _ = orbit_cg_m_grey_d4h_spin_spatial_1e
        .calc_smat(Some(&sao_cg), None, true)
        .unwrap()
        .calc_xmat(false);
    assert_eq!(
        orbit_cg_m_grey_d4h_spin_spatial_1e.analyse_rep().unwrap(),
        DecomposedSymbol::<MullikenIrcorepSymbol>::new("2||E|_(g)| ⊕ 2||A|_(2u)| ⊕ 2||B|_(1u)|")
            .unwrap()
    );

    let mut orbit_cg_m_grey_d4h_spin_spatial_2e = SlaterDeterminantSymmetryOrbit::builder()
        .group(&group_m_grey_d4h)
        .origin(&det_2e_cg)
        .integrality_threshold(1e-14)
        .linear_independence_threshold(1e-14)
        .symmetry_transformation_kind(SymmetryTransformationKind::SpinSpatial)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
        .build()
        .unwrap();
    let _ = orbit_cg_m_grey_d4h_spin_spatial_2e
        .calc_smat(Some(&sao_cg), None, true)
        .unwrap()
        .calc_xmat(false);
    assert_eq!(
        orbit_cg_m_grey_d4h_spin_spatial_2e.analyse_rep().unwrap(),
        DecomposedSymbol::<MullikenIrcorepSymbol>::new(
            "4||E|_(g)| ⊕ 2||A|_(1u)| ⊕ 2||A|_(2u)| ⊕ 2||B|_(1u)| ⊕ 2||B|_(2u)|"
        )
        .unwrap(),
    );

    // ~~~~~~~~~~~~~~~~~~~~~~~~
    // u D4h(C4h) (bw, unitary)
    // ~~~~~~~~~~~~~~~~~~~~~~~~

    let mut orbit_cg_u_bw_d4h_c4h_spatial_1e = SlaterDeterminantSymmetryOrbit::builder()
        .group(&group_u_bw_d4h_c4h)
        .origin(&det_1e_cg)
        .integrality_threshold(1e-14)
        .linear_independence_threshold(1e-14)
        .symmetry_transformation_kind(SymmetryTransformationKind::Spatial)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
        .build()
        .unwrap();
    let _ = orbit_cg_u_bw_d4h_c4h_spatial_1e
        .calc_smat(Some(&sao_cg), None, true)
        .unwrap()
        .calc_xmat(false);
    assert_eq!(
        orbit_cg_u_bw_d4h_c4h_spatial_1e.analyse_rep().unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new("||E|_(g)| ⊕ ||A|_(2u)| ⊕ ||B|_(1u)|")
            .unwrap()
    );

    let mut orbit_cg_u_bw_d4h_c4h_spatial_2e = SlaterDeterminantSymmetryOrbit::builder()
        .group(&group_u_bw_d4h_c4h)
        .origin(&det_2e_cg)
        .integrality_threshold(1e-14)
        .linear_independence_threshold(1e-14)
        .symmetry_transformation_kind(SymmetryTransformationKind::Spatial)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
        .build()
        .unwrap();
    let _ = orbit_cg_u_bw_d4h_c4h_spatial_2e
        .calc_smat(Some(&sao_cg), None, true)
        .unwrap()
        .calc_xmat(false);
    assert_eq!(
        orbit_cg_u_bw_d4h_c4h_spatial_2e.analyse_rep().unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new(
            "2||E|_(g)| ⊕ ||A|_(1u)| ⊕ ||A|_(2u)| ⊕ ||B|_(1u)| ⊕ ||B|_(2u)|"
        )
        .unwrap(),
    );

    let mut orbit_cg_u_bw_d4h_c4h_spin_1e = SlaterDeterminantSymmetryOrbit::builder()
        .group(&group_u_bw_d4h_c4h)
        .origin(&det_1e_cg)
        .integrality_threshold(1e-14)
        .linear_independence_threshold(1e-14)
        .symmetry_transformation_kind(SymmetryTransformationKind::Spin)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
        .build()
        .unwrap();
    let _ = orbit_cg_u_bw_d4h_c4h_spin_1e
        .calc_smat(Some(&sao_cg), None, true)
        .unwrap()
        .calc_xmat(false);
    // Character analysis would give |A|_(1g) ⊕ |A|_(2g), but |α⟩ and |β⟩ cannot be linearly
    // combined to span each of the subspaces separately.
    assert!(orbit_cg_u_bw_d4h_c4h_spin_1e.analyse_rep().is_err());

    let mut orbit_cg_u_bw_d4h_c4h_spin_2e = SlaterDeterminantSymmetryOrbit::builder()
        .group(&group_u_bw_d4h_c4h)
        .origin(&det_2e_cg)
        .integrality_threshold(1e-14)
        .linear_independence_threshold(1e-14)
        .symmetry_transformation_kind(SymmetryTransformationKind::Spin)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
        .build()
        .unwrap();
    let _ = orbit_cg_u_bw_d4h_c4h_spin_2e
        .calc_smat(Some(&sao_cg), None, true)
        .unwrap()
        .calc_xmat(false);
    assert_eq!(
        orbit_cg_u_bw_d4h_c4h_spin_2e.analyse_rep().unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new("||A|_(1g)| ⊕ ||A|_(2g)|").unwrap(),
    );

    let mut orbit_cg_u_bw_d4h_c4h_spin_spatial_1e = SlaterDeterminantSymmetryOrbit::builder()
        .group(&group_u_bw_d4h_c4h)
        .origin(&det_1e_cg)
        .integrality_threshold(1e-14)
        .linear_independence_threshold(1e-14)
        .symmetry_transformation_kind(SymmetryTransformationKind::SpinSpatial)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
        .build()
        .unwrap();
    let _ = orbit_cg_u_bw_d4h_c4h_spin_spatial_1e
        .calc_smat(Some(&sao_cg), None, true)
        .unwrap()
        .calc_xmat(false);
    assert!(orbit_cg_u_bw_d4h_c4h_spin_spatial_1e.analyse_rep().is_err());

    let mut orbit_cg_u_bw_d4h_c4h_spin_spatial_2e = SlaterDeterminantSymmetryOrbit::builder()
        .group(&group_u_bw_d4h_c4h)
        .origin(&det_2e_cg)
        .integrality_threshold(1e-14)
        .linear_independence_threshold(1e-14)
        .symmetry_transformation_kind(SymmetryTransformationKind::SpinSpatial)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
        .build()
        .unwrap();
    let _ = orbit_cg_u_bw_d4h_c4h_spin_spatial_2e
        .calc_smat(Some(&sao_cg), None, true)
        .unwrap()
        .calc_xmat(false);
    // Half of the irreps are missing here, but that is because this orbit starts from a
    // spin-collinear origin, but there is no pure spin rotation operations in the unitary ordinary
    // group to rotate spin independently from spatial degrees of freedom.
    assert_eq!(
        orbit_cg_u_bw_d4h_c4h_spin_spatial_2e.analyse_rep().unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new(
            "2||E|_(g)| ⊕ ||A|_(1u)| ⊕ ||A|_(2u)| ⊕ ||B|_(1u)| ⊕ ||B|_(2u)|"
        )
        .unwrap(),
    );

    // ~~~~~~~~~~~~~~~~~~~~~~~~~
    // m D4h(C4h) (bw, magnetic)
    // ~~~~~~~~~~~~~~~~~~~~~~~~~

    let mut orbit_cg_m_bw_d4h_c4h_spatial_1e = SlaterDeterminantSymmetryOrbit::builder()
        .group(&group_m_bw_d4h_c4h)
        .origin(&det_1e_cg)
        .integrality_threshold(1e-14)
        .linear_independence_threshold(1e-14)
        .symmetry_transformation_kind(SymmetryTransformationKind::Spatial)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
        .build()
        .unwrap();
    let _ = orbit_cg_m_bw_d4h_c4h_spatial_1e
        .calc_smat(Some(&sao_cg), None, true)
        .unwrap()
        .calc_xmat(false);
    assert_eq!(
        orbit_cg_m_bw_d4h_c4h_spatial_1e.analyse_rep().unwrap(),
        DecomposedSymbol::<MullikenIrcorepSymbol>::new(
            "|_(a)|Γ|_(g)| ⊕ |_(b)|Γ|_(g)| ⊕ ||A|_(u)| ⊕ ||B|_(u)|"
        )
        .unwrap()
    );

    let mut orbit_cg_m_bw_d4h_c4h_spatial_2e = SlaterDeterminantSymmetryOrbit::builder()
        .group(&group_m_bw_d4h_c4h)
        .origin(&det_2e_cg)
        .integrality_threshold(1e-14)
        .linear_independence_threshold(1e-14)
        .symmetry_transformation_kind(SymmetryTransformationKind::Spatial)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
        .build()
        .unwrap();
    let _ = orbit_cg_m_bw_d4h_c4h_spatial_2e
        .calc_smat(Some(&sao_cg), None, true)
        .unwrap()
        .calc_xmat(false);
    assert_eq!(
        orbit_cg_m_bw_d4h_c4h_spatial_2e.analyse_rep().unwrap(),
        DecomposedSymbol::<MullikenIrcorepSymbol>::new(
            "2|_(a)|Γ|_(g)| ⊕ 2|_(b)|Γ|_(g)| ⊕ 2||A|_(u)| ⊕ 2||B|_(u)|"
        )
        .unwrap(),
    );

    let mut orbit_cg_m_bw_d4h_c4h_spin_1e = SlaterDeterminantSymmetryOrbit::builder()
        .group(&group_m_bw_d4h_c4h)
        .origin(&det_1e_cg)
        .integrality_threshold(1e-14)
        .linear_independence_threshold(1e-14)
        .symmetry_transformation_kind(SymmetryTransformationKind::Spin)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
        .build()
        .unwrap();
    let _ = orbit_cg_m_bw_d4h_c4h_spin_1e
        .calc_smat(Some(&sao_cg), None, true)
        .unwrap()
        .calc_xmat(false);
    assert_eq!(
        orbit_cg_m_bw_d4h_c4h_spin_1e.analyse_rep().unwrap(),
        DecomposedSymbol::<MullikenIrcorepSymbol>::new("2||A|_(g)|").unwrap(),
    );

    let mut orbit_cg_m_bw_d4h_c4h_spin_2e = SlaterDeterminantSymmetryOrbit::builder()
        .group(&group_m_bw_d4h_c4h)
        .origin(&det_2e_cg)
        .integrality_threshold(1e-14)
        .linear_independence_threshold(1e-14)
        .symmetry_transformation_kind(SymmetryTransformationKind::Spin)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
        .build()
        .unwrap();
    let _ = orbit_cg_m_bw_d4h_c4h_spin_2e
        .calc_smat(Some(&sao_cg), None, true)
        .unwrap()
        .calc_xmat(false);
    assert_eq!(
        orbit_cg_m_bw_d4h_c4h_spin_2e.analyse_rep().unwrap(),
        DecomposedSymbol::<MullikenIrcorepSymbol>::new("2||A|_(g)|").unwrap(),
    );

    let mut orbit_cg_m_bw_d4h_c4h_spin_spatial_1e = SlaterDeterminantSymmetryOrbit::builder()
        .group(&group_m_bw_d4h_c4h)
        .origin(&det_1e_cg)
        .integrality_threshold(1e-14)
        .linear_independence_threshold(1e-14)
        .symmetry_transformation_kind(SymmetryTransformationKind::SpinSpatial)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
        .build()
        .unwrap();
    let _ = orbit_cg_m_bw_d4h_c4h_spin_spatial_1e
        .calc_smat(Some(&sao_cg), None, true)
        .unwrap()
        .calc_xmat(false);
    assert_eq!(
        orbit_cg_m_bw_d4h_c4h_spin_spatial_1e.analyse_rep().unwrap(),
        DecomposedSymbol::<MullikenIrcorepSymbol>::new(
            "2|_(a)|Γ|_(g)| ⊕ 2|_(b)|Γ|_(g)| ⊕ 2||A|_(u)| ⊕ 2||B|_(u)|"
        )
        .unwrap()
    );

    let mut orbit_cg_m_bw_d4h_c4h_spin_spatial_2e = SlaterDeterminantSymmetryOrbit::builder()
        .group(&group_m_bw_d4h_c4h)
        .origin(&det_2e_cg)
        .integrality_threshold(1e-14)
        .linear_independence_threshold(1e-14)
        .symmetry_transformation_kind(SymmetryTransformationKind::SpinSpatial)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
        .build()
        .unwrap();
    let _ = orbit_cg_m_bw_d4h_c4h_spin_spatial_2e
        .calc_smat(Some(&sao_cg), None, true)
        .unwrap()
        .calc_xmat(false);
    // Half of the ircoreps are missing here, but that is because this orbit starts from a
    // spin-collinear origin, but there is no pure spin rotation operations in the group to
    // rotate spin independently from spatial degrees of freedom?
    assert_eq!(
        orbit_cg_m_bw_d4h_c4h_spin_spatial_2e.analyse_rep().unwrap(),
        DecomposedSymbol::<MullikenIrcorepSymbol>::new(
            "2|_(a)|Γ|_(g)| ⊕ 2|_(b)|Γ|_(g)| ⊕ 2||A|_(u)| ⊕ 2||B|_(u)|"
        )
        .unwrap(),
    );

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // u D4h* (ordinary double, unitary)
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    let mut orbit_cg_u_d4h_double_spatial_1e = SlaterDeterminantSymmetryOrbit::builder()
        .group(&group_u_d4h_double)
        .origin(&det_1e_cg)
        .integrality_threshold(1e-14)
        .linear_independence_threshold(1e-14)
        .symmetry_transformation_kind(SymmetryTransformationKind::Spatial)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
        .build()
        .unwrap();
    let _ = orbit_cg_u_d4h_double_spatial_1e
        .calc_smat(Some(&sao_cg), None, true)
        .unwrap()
        .calc_xmat(false);
    assert_eq!(
        orbit_cg_u_d4h_double_spatial_1e.analyse_rep().unwrap(),
        orbit_cg_u_d4h_spatial_1e.analyse_rep().unwrap(),
    );

    let mut orbit_cg_u_d4h_double_spatial_2e = SlaterDeterminantSymmetryOrbit::builder()
        .group(&group_u_d4h_double)
        .origin(&det_2e_cg)
        .integrality_threshold(1e-14)
        .linear_independence_threshold(1e-14)
        .symmetry_transformation_kind(SymmetryTransformationKind::Spatial)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
        .build()
        .unwrap();
    let _ = orbit_cg_u_d4h_double_spatial_2e
        .calc_smat(Some(&sao_cg), None, true)
        .unwrap()
        .calc_xmat(false);
    assert_eq!(
        orbit_cg_u_d4h_double_spatial_2e.analyse_rep().unwrap(),
        orbit_cg_u_d4h_spatial_2e.analyse_rep().unwrap(),
    );

    let mut orbit_cg_u_d4h_double_spin_1e = SlaterDeterminantSymmetryOrbit::builder()
        .group(&group_u_d4h_double)
        .origin(&det_1e_cg)
        .integrality_threshold(1e-14)
        .linear_independence_threshold(1e-14)
        .symmetry_transformation_kind(SymmetryTransformationKind::Spin)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
        .build()
        .unwrap();
    let _ = orbit_cg_u_d4h_double_spin_1e
        .calc_smat(Some(&sao_cg), None, true)
        .unwrap()
        .calc_xmat(false);
    assert_eq!(
        orbit_cg_u_d4h_double_spin_1e.analyse_rep().unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new("||E~|_(1g)|").unwrap()
    );

    let mut orbit_cg_u_d4h_double_spin_2e = SlaterDeterminantSymmetryOrbit::builder()
        .group(&group_u_d4h_double)
        .origin(&det_2e_cg)
        .integrality_threshold(1e-14)
        .linear_independence_threshold(1e-14)
        .symmetry_transformation_kind(SymmetryTransformationKind::Spin)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
        .build()
        .unwrap();
    let _ = orbit_cg_u_d4h_double_spin_2e
        .calc_smat(Some(&sao_cg), None, true)
        .unwrap()
        .calc_xmat(false);
    assert_eq!(
        orbit_cg_u_d4h_double_spin_2e.analyse_rep().unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new("||A|_(1g)| ⊕ ||A|_(2g)|").unwrap()
    );

    let mut orbit_cg_u_d4h_double_spin_spatial_1e = SlaterDeterminantSymmetryOrbit::builder()
        .group(&group_u_d4h_double)
        .origin(&det_1e_cg)
        .integrality_threshold(1e-14)
        .linear_independence_threshold(1e-14)
        .symmetry_transformation_kind(SymmetryTransformationKind::SpinSpatial)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
        .build()
        .unwrap();
    let _ = orbit_cg_u_d4h_double_spin_spatial_1e
        .calc_smat(Some(&sao_cg), None, true)
        .unwrap()
        .calc_xmat(false);
    assert_eq!(
        orbit_cg_u_d4h_double_spin_spatial_1e.analyse_rep().unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new(
            "||E~|_(1g)| ⊕ ||E~|_(2g)| ⊕ ||E~|_(1u)| ⊕ ||E~|_(2u)|"
        )
        .unwrap()
    );

    let mut orbit_cg_u_d4h_double_spin_spatial_2e = SlaterDeterminantSymmetryOrbit::builder()
        .group(&group_u_d4h_double)
        .origin(&det_2e_cg)
        .integrality_threshold(1e-14)
        .linear_independence_threshold(1e-14)
        .symmetry_transformation_kind(SymmetryTransformationKind::SpinSpatial)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
        .build()
        .unwrap();
    let _ = orbit_cg_u_d4h_double_spin_spatial_2e
        .calc_smat(Some(&sao_cg), None, true)
        .unwrap()
        .calc_xmat(false);
    // Half of the irreps are missing here, but that is because this orbit starts from a
    // spin-collinear origin, but there is no pure spin rotation operations in the unitary ordinary
    // group to rotate spin independently from spatial degrees of freedom.
    assert_eq!(
        orbit_cg_u_d4h_double_spin_spatial_2e.analyse_rep().unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new(
            "2||E|_(g)| ⊕ ||A|_(1u)| ⊕ ||A|_(2u)| ⊕ ||B|_(1u)| ⊕ ||B|_(2u)|"
        )
        .unwrap(),
    );

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // u D4h'* (grey double, unitary)
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    let mut orbit_cg_u_grey_d4h_double_spatial_1e = SlaterDeterminantSymmetryOrbit::builder()
        .group(&group_u_grey_d4h_double)
        .origin(&det_1e_cg)
        .integrality_threshold(1e-14)
        .linear_independence_threshold(1e-14)
        .symmetry_transformation_kind(SymmetryTransformationKind::Spatial)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
        .build()
        .unwrap();
    let _ = orbit_cg_u_grey_d4h_double_spatial_1e
        .calc_smat(Some(&sao_cg), None, true)
        .unwrap()
        .calc_xmat(false);
    assert_eq!(
        orbit_cg_u_grey_d4h_double_spatial_1e.analyse_rep().unwrap(),
        orbit_cg_u_grey_d4h_spatial_1e.analyse_rep().unwrap(),
    );

    let mut orbit_cg_u_grey_d4h_double_spatial_2e = SlaterDeterminantSymmetryOrbit::builder()
        .group(&group_u_grey_d4h_double)
        .origin(&det_2e_cg)
        .integrality_threshold(1e-14)
        .linear_independence_threshold(1e-14)
        .symmetry_transformation_kind(SymmetryTransformationKind::Spatial)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
        .build()
        .unwrap();
    let _ = orbit_cg_u_grey_d4h_double_spatial_2e
        .calc_smat(Some(&sao_cg), None, true)
        .unwrap()
        .calc_xmat(false);
    assert_eq!(
        orbit_cg_u_grey_d4h_double_spatial_2e.analyse_rep().unwrap(),
        orbit_cg_u_grey_d4h_spatial_2e.analyse_rep().unwrap(),
    );

    let mut orbit_cg_u_grey_d4h_double_spin_1e = SlaterDeterminantSymmetryOrbit::builder()
        .group(&group_u_grey_d4h_double)
        .origin(&det_1e_cg)
        .integrality_threshold(1e-14)
        .linear_independence_threshold(1e-14)
        .symmetry_transformation_kind(SymmetryTransformationKind::Spin)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
        .build()
        .unwrap();
    let _ = orbit_cg_u_grey_d4h_double_spin_1e
        .calc_smat(Some(&sao_cg), None, true)
        .unwrap()
        .calc_xmat(false);
    assert!(orbit_cg_u_grey_d4h_double_spin_1e.analyse_rep().is_err(),);

    let mut orbit_cg_u_grey_d4h_double_spin_2e = SlaterDeterminantSymmetryOrbit::builder()
        .group(&group_u_grey_d4h_double)
        .origin(&det_2e_cg)
        .integrality_threshold(1e-14)
        .linear_independence_threshold(1e-14)
        .symmetry_transformation_kind(SymmetryTransformationKind::Spin)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
        .build()
        .unwrap();
    let _ = orbit_cg_u_grey_d4h_double_spin_2e
        .calc_smat(Some(&sao_cg), None, true)
        .unwrap()
        .calc_xmat(false);
    assert_eq!(
        orbit_cg_u_grey_d4h_double_spin_2e.analyse_rep().unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new("|^(+)|A|_(1g)| ⊕ |^(-)|A|_(2g)|").unwrap()
    );

    let mut orbit_cg_u_grey_d4h_double_spin_spatial_1e = SlaterDeterminantSymmetryOrbit::builder()
        .group(&group_u_grey_d4h_double)
        .origin(&det_1e_cg)
        .integrality_threshold(1e-14)
        .linear_independence_threshold(1e-14)
        .symmetry_transformation_kind(SymmetryTransformationKind::SpinSpatial)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
        .build()
        .unwrap();
    let _ = orbit_cg_u_grey_d4h_double_spin_spatial_1e
        .calc_smat(Some(&sao_cg), None, true)
        .unwrap()
        .calc_xmat(false);
    assert!(orbit_cg_u_grey_d4h_double_spin_spatial_1e
        .analyse_rep()
        .is_err(),);

    let mut orbit_cg_u_grey_d4h_double_spin_spatial_2e = SlaterDeterminantSymmetryOrbit::builder()
        .group(&group_u_grey_d4h_double)
        .origin(&det_2e_cg)
        .integrality_threshold(1e-14)
        .linear_independence_threshold(1e-14)
        .symmetry_transformation_kind(SymmetryTransformationKind::SpinSpatial)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
        .build()
        .unwrap();
    let _ = orbit_cg_u_grey_d4h_double_spin_spatial_2e
        .calc_smat(Some(&sao_cg), None, true)
        .unwrap()
        .calc_xmat(false);
    assert_eq!(
        orbit_cg_u_grey_d4h_double_spin_spatial_2e
            .analyse_rep()
            .unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new(
            "2|^(+)|E|_(g)| ⊕ |^(+)|A|_(1u)| ⊕ |^(+)|A|_(2u)| ⊕ |^(+)|B|_(1u)| ⊕ |^(+)|B|_(2u)|
            ⊕ 2|^(-)|E|_(g)| ⊕ |^(-)|A|_(1u)| ⊕ |^(-)|A|_(2u)| ⊕ |^(-)|B|_(1u)| ⊕ |^(-)|B|_(2u)|"
        )
        .unwrap(),
    );

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // m D4h'* (grey double, magnetic)
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    let mut orbit_cg_m_grey_d4h_double_spatial_1e = SlaterDeterminantSymmetryOrbit::builder()
        .group(&group_m_grey_d4h_double)
        .origin(&det_1e_cg)
        .integrality_threshold(1e-14)
        .linear_independence_threshold(1e-14)
        .symmetry_transformation_kind(SymmetryTransformationKind::Spatial)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
        .build()
        .unwrap();
    let _ = orbit_cg_m_grey_d4h_double_spatial_1e
        .calc_smat(Some(&sao_cg), None, true)
        .unwrap()
        .calc_xmat(false);
    assert_eq!(
        orbit_cg_m_grey_d4h_double_spatial_1e.analyse_rep().unwrap(),
        orbit_cg_m_grey_d4h_spatial_1e.analyse_rep().unwrap(),
    );

    let mut orbit_cg_m_grey_d4h_double_spatial_2e = SlaterDeterminantSymmetryOrbit::builder()
        .group(&group_m_grey_d4h_double)
        .origin(&det_2e_cg)
        .integrality_threshold(1e-14)
        .linear_independence_threshold(1e-14)
        .symmetry_transformation_kind(SymmetryTransformationKind::Spatial)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
        .build()
        .unwrap();
    let _ = orbit_cg_m_grey_d4h_double_spatial_2e
        .calc_smat(Some(&sao_cg), None, true)
        .unwrap()
        .calc_xmat(false);
    assert_eq!(
        orbit_cg_m_grey_d4h_double_spatial_2e.analyse_rep().unwrap(),
        orbit_cg_m_grey_d4h_spatial_2e.analyse_rep().unwrap(),
    );

    let mut orbit_cg_m_grey_d4h_double_spin_1e = SlaterDeterminantSymmetryOrbit::builder()
        .group(&group_m_grey_d4h_double)
        .origin(&det_1e_cg)
        .integrality_threshold(1e-14)
        .linear_independence_threshold(1e-14)
        .symmetry_transformation_kind(SymmetryTransformationKind::Spin)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
        .build()
        .unwrap();
    let _ = orbit_cg_m_grey_d4h_double_spin_1e
        .calc_smat(Some(&sao_cg), None, true)
        .unwrap()
        .calc_xmat(false);
    assert_eq!(
        orbit_cg_m_grey_d4h_double_spin_1e.analyse_rep().unwrap(),
        DecomposedSymbol::<MullikenIrcorepSymbol>::new("||E~|_(1g)|").unwrap()
    );

    let mut orbit_cg_m_grey_d4h_double_spin_2e = SlaterDeterminantSymmetryOrbit::builder()
        .group(&group_m_grey_d4h_double)
        .origin(&det_2e_cg)
        .integrality_threshold(1e-14)
        .linear_independence_threshold(1e-14)
        .symmetry_transformation_kind(SymmetryTransformationKind::Spin)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
        .build()
        .unwrap();
    let _ = orbit_cg_m_grey_d4h_double_spin_2e
        .calc_smat(Some(&sao_cg), None, true)
        .unwrap()
        .calc_xmat(false);
    assert_eq!(
        orbit_cg_m_grey_d4h_double_spin_2e.analyse_rep().unwrap(),
        DecomposedSymbol::<MullikenIrcorepSymbol>::new("||A|_(1g)| ⊕ ||A|_(2g)|").unwrap()
    );

    let mut orbit_cg_m_grey_d4h_double_spin_spatial_1e = SlaterDeterminantSymmetryOrbit::builder()
        .group(&group_m_grey_d4h_double)
        .origin(&det_1e_cg)
        .integrality_threshold(1e-14)
        .linear_independence_threshold(1e-14)
        .symmetry_transformation_kind(SymmetryTransformationKind::SpinSpatial)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
        .build()
        .unwrap();
    let _ = orbit_cg_m_grey_d4h_double_spin_spatial_1e
        .calc_smat(Some(&sao_cg), None, true)
        .unwrap()
        .calc_xmat(false);
    assert_eq!(
        orbit_cg_m_grey_d4h_double_spin_spatial_1e
            .analyse_rep()
            .unwrap(),
        DecomposedSymbol::<MullikenIrcorepSymbol>::new(
            "||E~|_(1g)| ⊕ ||E~|_(2g)| ⊕ ||E~|_(1u)| ⊕ ||E~|_(2u)|"
        )
        .unwrap()
    );

    let mut orbit_cg_m_grey_d4h_double_spin_spatial_2e = SlaterDeterminantSymmetryOrbit::builder()
        .group(&group_m_grey_d4h_double)
        .origin(&det_2e_cg)
        .integrality_threshold(1e-14)
        .linear_independence_threshold(1e-14)
        .symmetry_transformation_kind(SymmetryTransformationKind::SpinSpatial)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
        .build()
        .unwrap();
    let _ = orbit_cg_m_grey_d4h_double_spin_spatial_2e
        .calc_smat(Some(&sao_cg), None, true)
        .unwrap()
        .calc_xmat(false);
    // Compared to u D4h*, all ircoreps are present here because there is time reversal in this
    // group which is essentially a pure spin rotation operation that can rotate spin independently
    // from spatial degrees of freedom.
    assert_eq!(
        orbit_cg_m_grey_d4h_double_spin_spatial_2e
            .analyse_rep()
            .unwrap(),
        DecomposedSymbol::<MullikenIrcorepSymbol>::new(
            "4||E|_(g)| ⊕ 2||A|_(1u)| ⊕ 2||A|_(2u)| ⊕ 2||B|_(1u)| ⊕ 2||B|_(2u)|"
        )
        .unwrap(),
    );

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // u D4h(C4h)* (bw double, unitary)
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    let mut orbit_cg_u_bw_d4h_c4h_double_spatial_1e = SlaterDeterminantSymmetryOrbit::builder()
        .group(&group_u_bw_d4h_c4h_double)
        .origin(&det_1e_cg)
        .integrality_threshold(1e-14)
        .linear_independence_threshold(1e-14)
        .symmetry_transformation_kind(SymmetryTransformationKind::Spatial)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
        .build()
        .unwrap();
    let _ = orbit_cg_u_bw_d4h_c4h_double_spatial_1e
        .calc_smat(Some(&sao_cg), None, true)
        .unwrap()
        .calc_xmat(false);
    assert_eq!(
        orbit_cg_u_bw_d4h_c4h_double_spatial_1e
            .analyse_rep()
            .unwrap(),
        orbit_cg_u_bw_d4h_c4h_spatial_1e.analyse_rep().unwrap(),
    );

    let mut orbit_cg_u_bw_d4h_c4h_double_spatial_2e = SlaterDeterminantSymmetryOrbit::builder()
        .group(&group_u_bw_d4h_c4h_double)
        .origin(&det_2e_cg)
        .integrality_threshold(1e-14)
        .linear_independence_threshold(1e-14)
        .symmetry_transformation_kind(SymmetryTransformationKind::Spatial)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
        .build()
        .unwrap();
    let _ = orbit_cg_u_bw_d4h_c4h_double_spatial_2e
        .calc_smat(Some(&sao_cg), None, true)
        .unwrap()
        .calc_xmat(false);
    assert_eq!(
        orbit_cg_u_bw_d4h_c4h_double_spatial_2e
            .analyse_rep()
            .unwrap(),
        orbit_cg_u_bw_d4h_c4h_spatial_2e.analyse_rep().unwrap(),
    );

    let mut orbit_cg_u_bw_d4h_c4h_double_spin_1e = SlaterDeterminantSymmetryOrbit::builder()
        .group(&group_u_bw_d4h_c4h_double)
        .origin(&det_1e_cg)
        .integrality_threshold(1e-14)
        .linear_independence_threshold(1e-14)
        .symmetry_transformation_kind(SymmetryTransformationKind::Spin)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
        .build()
        .unwrap();
    let _ = orbit_cg_u_bw_d4h_c4h_double_spin_1e
        .calc_smat(Some(&sao_cg), None, true)
        .unwrap()
        .calc_xmat(false);
    assert!(orbit_cg_u_bw_d4h_c4h_double_spin_1e.analyse_rep().is_err());

    let mut orbit_cg_u_bw_d4h_c4h_double_spin_2e = SlaterDeterminantSymmetryOrbit::builder()
        .group(&group_u_bw_d4h_c4h_double)
        .origin(&det_2e_cg)
        .integrality_threshold(1e-14)
        .linear_independence_threshold(1e-14)
        .symmetry_transformation_kind(SymmetryTransformationKind::Spin)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
        .build()
        .unwrap();
    let _ = orbit_cg_u_bw_d4h_c4h_double_spin_2e
        .calc_smat(Some(&sao_cg), None, true)
        .unwrap()
        .calc_xmat(false);
    assert_eq!(
        orbit_cg_u_bw_d4h_c4h_double_spin_2e.analyse_rep().unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new("||A|_(1g)|").unwrap(),
    );

    let mut orbit_cg_u_bw_d4h_c4h_double_spin_spatial_1e =
        SlaterDeterminantSymmetryOrbit::builder()
            .group(&group_u_bw_d4h_c4h_double)
            .origin(&det_1e_cg)
            .integrality_threshold(1e-14)
            .linear_independence_threshold(1e-14)
            .symmetry_transformation_kind(SymmetryTransformationKind::SpinSpatial)
            .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
            .build()
            .unwrap();
    let _ = orbit_cg_u_bw_d4h_c4h_double_spin_spatial_1e
        .calc_smat(Some(&sao_cg), None, true)
        .unwrap()
        .calc_xmat(false);
    assert!(orbit_cg_u_bw_d4h_c4h_double_spin_spatial_1e
        .analyse_rep()
        .is_err());

    let mut orbit_cg_u_bw_d4h_c4h_double_spin_spatial_2e =
        SlaterDeterminantSymmetryOrbit::builder()
            .group(&group_u_bw_d4h_c4h_double)
            .origin(&det_2e_cg)
            .integrality_threshold(1e-14)
            .linear_independence_threshold(1e-14)
            .symmetry_transformation_kind(SymmetryTransformationKind::SpinSpatial)
            .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
            .build()
            .unwrap();
    let _ = orbit_cg_u_bw_d4h_c4h_double_spin_spatial_2e
        .calc_smat(Some(&sao_cg), None, true)
        .unwrap()
        .calc_xmat(false);
    assert_eq!(
        orbit_cg_u_bw_d4h_c4h_double_spin_spatial_2e
            .analyse_rep()
            .unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new(
            "2||E|_(g)| ⊕ ||A|_(1u)| ⊕ ||A|_(2u)| ⊕ ||B|_(1u)| ⊕ ||B|_(2u)|"
        )
        .unwrap(),
    );

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // m D4h(C4h)* (bw double, magnetic)
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    let mut orbit_cg_m_bw_d4h_c4h_double_spatial_1e = SlaterDeterminantSymmetryOrbit::builder()
        .group(&group_m_bw_d4h_c4h_double)
        .origin(&det_1e_cg)
        .integrality_threshold(1e-14)
        .linear_independence_threshold(1e-14)
        .symmetry_transformation_kind(SymmetryTransformationKind::Spatial)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
        .build()
        .unwrap();
    let _ = orbit_cg_m_bw_d4h_c4h_double_spatial_1e
        .calc_smat(Some(&sao_cg), None, true)
        .unwrap()
        .calc_xmat(false);
    assert_eq!(
        orbit_cg_m_bw_d4h_c4h_double_spatial_1e
            .analyse_rep()
            .unwrap(),
        DecomposedSymbol::<MullikenIrcorepSymbol>::new(
            "|_(a)|Γ|_(g)| ⊕ |_(b)|Γ|_(g)| ⊕ ||A|_(u)| ⊕ ||B|_(u)|"
        )
        .unwrap()
    );

    let mut orbit_cg_m_bw_d4h_c4h_double_spatial_2e = SlaterDeterminantSymmetryOrbit::builder()
        .group(&group_m_bw_d4h_c4h_double)
        .origin(&det_2e_cg)
        .integrality_threshold(1e-14)
        .linear_independence_threshold(1e-14)
        .symmetry_transformation_kind(SymmetryTransformationKind::Spatial)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
        .build()
        .unwrap();
    let _ = orbit_cg_m_bw_d4h_c4h_double_spatial_2e
        .calc_smat(Some(&sao_cg), None, true)
        .unwrap()
        .calc_xmat(false);
    assert_eq!(
        orbit_cg_m_bw_d4h_c4h_double_spatial_2e
            .analyse_rep()
            .unwrap(),
        DecomposedSymbol::<MullikenIrcorepSymbol>::new(
            "2|_(a)|Γ|_(g)| ⊕ 2|_(b)|Γ|_(g)| ⊕ 2||A|_(u)| ⊕ 2||B|_(u)|"
        )
        .unwrap(),
    );

    let mut orbit_cg_m_bw_d4h_c4h_double_spin_1e = SlaterDeterminantSymmetryOrbit::builder()
        .group(&group_m_bw_d4h_c4h_double)
        .origin(&det_1e_cg)
        .integrality_threshold(1e-14)
        .linear_independence_threshold(1e-14)
        .symmetry_transformation_kind(SymmetryTransformationKind::Spin)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
        .build()
        .unwrap();
    let _ = orbit_cg_m_bw_d4h_c4h_double_spin_1e
        .calc_smat(Some(&sao_cg), None, true)
        .unwrap()
        .calc_xmat(false);
    assert_eq!(
        orbit_cg_m_bw_d4h_c4h_double_spin_1e.analyse_rep().unwrap(),
        DecomposedSymbol::<MullikenIrcorepSymbol>::new("|_(b)|Γ~|_(1g)|").unwrap(),
    );

    let mut orbit_cg_m_bw_d4h_c4h_double_spin_2e = SlaterDeterminantSymmetryOrbit::builder()
        .group(&group_m_bw_d4h_c4h_double)
        .origin(&det_2e_cg)
        .integrality_threshold(1e-14)
        .linear_independence_threshold(1e-14)
        .symmetry_transformation_kind(SymmetryTransformationKind::Spin)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
        .build()
        .unwrap();
    let _ = orbit_cg_m_bw_d4h_c4h_double_spin_2e
        .calc_smat(Some(&sao_cg), None, true)
        .unwrap()
        .calc_xmat(false);
    assert_eq!(
        orbit_cg_m_bw_d4h_c4h_double_spin_2e.analyse_rep().unwrap(),
        DecomposedSymbol::<MullikenIrcorepSymbol>::new("||A|_(g)|").unwrap(),
    );

    let mut orbit_cg_m_bw_d4h_c4h_double_spin_spatial_1e =
        SlaterDeterminantSymmetryOrbit::builder()
            .group(&group_m_bw_d4h_c4h_double)
            .origin(&det_1e_cg)
            .integrality_threshold(1e-14)
            .linear_independence_threshold(1e-14)
            .symmetry_transformation_kind(SymmetryTransformationKind::SpinSpatial)
            .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
            .build()
            .unwrap();
    let _ = orbit_cg_m_bw_d4h_c4h_double_spin_spatial_1e
        .calc_smat(Some(&sao_cg), None, true)
        .unwrap()
        .calc_xmat(false);
    assert_eq!(
        orbit_cg_m_bw_d4h_c4h_double_spin_spatial_1e
            .analyse_rep()
            .unwrap(),
        DecomposedSymbol::<MullikenIrcorepSymbol>::new(
            "|_(a)|Γ~|_(1g)| ⊕ |_(b)|Γ~|_(2g)| ⊕ |_(a)|Γ~|_(2u)| ⊕ |_(b)|Γ~|_(1u)|"
        )
        .unwrap()
    );

    let mut orbit_cg_m_bw_d4h_c4h_spin_spatial_2e = SlaterDeterminantSymmetryOrbit::builder()
        .group(&group_m_bw_d4h_c4h)
        .origin(&det_2e_cg)
        .integrality_threshold(1e-14)
        .linear_independence_threshold(1e-14)
        .symmetry_transformation_kind(SymmetryTransformationKind::SpinSpatial)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
        .build()
        .unwrap();
    let _ = orbit_cg_m_bw_d4h_c4h_spin_spatial_2e
        .calc_smat(Some(&sao_cg), None, true)
        .unwrap()
        .calc_xmat(false);
    assert_eq!(
        orbit_cg_m_bw_d4h_c4h_spin_spatial_2e.analyse_rep().unwrap(),
        DecomposedSymbol::<MullikenIrcorepSymbol>::new(
            "2|_(a)|Γ|_(g)| ⊕ 2|_(b)|Γ|_(g)| ⊕ 2||A|_(u)| ⊕ 2||B|_(u)|"
        )
        .unwrap(),
    );
}

#[test]
fn test_determinant_orbit_rep_analysis_vf6_oct_qchem_order() {
    // env_logger::init();
    let emap = ElementMap::new();
    let atm_v = Atom::from_xyz("V +0.0 +0.0 +0.0", &emap, 1e-7).unwrap();
    let atm_f0 = Atom::from_xyz("F +0.0 +0.0 +1.0", &emap, 1e-7).unwrap();
    let atm_f1 = Atom::from_xyz("F +0.0 +0.0 -1.0", &emap, 1e-7).unwrap();
    let atm_f2 = Atom::from_xyz("F +1.0 +0.0 +0.0", &emap, 1e-7).unwrap();
    let atm_f3 = Atom::from_xyz("F -1.0 +0.0 +0.0", &emap, 1e-7).unwrap();
    let atm_f4 = Atom::from_xyz("F +0.0 +1.0 +0.0", &emap, 1e-7).unwrap();
    let atm_f5 = Atom::from_xyz("F +0.0 -1.0 +0.0", &emap, 1e-7).unwrap();

    let bsc_d = BasisShell::new(2, ShellOrder::Cart(CartOrder::qchem(2)));
    let bsp_s = BasisShell::new(0, ShellOrder::Pure(PureOrder::increasingm(0)));

    let batm_v = BasisAtom::new(&atm_v, &[bsc_d]);
    let batm_f0 = BasisAtom::new(&atm_f0, &[bsp_s.clone()]);
    let batm_f1 = BasisAtom::new(&atm_f1, &[bsp_s.clone()]);
    let batm_f2 = BasisAtom::new(&atm_f2, &[bsp_s.clone()]);
    let batm_f3 = BasisAtom::new(&atm_f3, &[bsp_s.clone()]);
    let batm_f4 = BasisAtom::new(&atm_f4, &[bsp_s.clone()]);
    let batm_f5 = BasisAtom::new(&atm_f5, &[bsp_s]);

    let bao_vf6 =
        BasisAngularOrder::new(&[batm_v, batm_f0, batm_f1, batm_f2, batm_f3, batm_f4, batm_f5]);
    let mol_vf6 = Molecule::from_atoms(
        &[
            atm_v.clone(),
            atm_f0.clone(),
            atm_f1.clone(),
            atm_f2.clone(),
            atm_f3.clone(),
            atm_f4.clone(),
            atm_f5.clone(),
        ],
        1e-7,
    )
    .recentre();

    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol_vf6)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false).unwrap();
    let group_u_oh = UnitaryRepresentedGroup::from_molecular_symmetry(&sym, None).unwrap();
    let group_u_oh_double = group_u_oh.to_double_group().unwrap();

    let thr = 1.0 / 3.0;
    let sao = array![
        [1.0, 0.0, thr, 0.0, 0.0, thr, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [thr, 0.0, 1.0, 0.0, 0.0, thr, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [thr, 0.0, thr, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    ];
    let mut sao_g = Array2::zeros((24, 24));
    sao_g.slice_mut(s![0..12, 0..12]).assign(&sao);
    sao_g.slice_mut(s![12..24, 12..24]).assign(&sao);
    let sao_cg = sao_g.mapv(C128::from);

    // ---
    // dyy
    // ---
    #[rustfmt::skip]
    let calpha = array![
        [0.0], [0.0], [1.0], [0.0], [0.0], [0.0],
        [0.0], [0.0], [0.0], [0.0], [0.0], [0.0],
    ];
    #[rustfmt::skip]
    let cbeta = array![
        [0.0], [0.0], [1.0], [0.0], [0.0], [0.0],
        [0.0], [0.0], [0.0], [0.0], [0.0], [0.0],
    ];
    let oalpha = array![1.0];
    let obeta_empty = array![0.0];
    let det_dyy_cg: SlaterDeterminant<C128> = SlaterDeterminant::<f64>::builder()
        .coefficients(&[calpha, cbeta])
        .occupations(&[oalpha, obeta_empty])
        .bao(&bao_vf6)
        .mol(&mol_vf6)
        .spin_constraint(SpinConstraint::Unrestricted(2, false))
        .complex_symmetric(false)
        .threshold(1e-14)
        .build()
        .unwrap()
        .to_generalised()
        .into();

    // ---
    // dxz
    // ---
    #[rustfmt::skip]
    let calpha = array![
        [0.0], [0.0], [0.0], [1.0], [0.0], [0.0],
        [0.0], [0.0], [0.0], [0.0], [0.0], [0.0],
    ];
    #[rustfmt::skip]
    let cbeta = array![
        [0.0], [0.0], [1.0], [0.0], [0.0], [0.0],
        [0.0], [0.0], [0.0], [0.0], [0.0], [0.0],
    ];
    let oalpha = array![1.0];
    let obeta_empty = array![0.0];
    let det_dxz_cg: SlaterDeterminant<C128> = SlaterDeterminant::<f64>::builder()
        .coefficients(&[calpha, cbeta])
        .occupations(&[oalpha, obeta_empty])
        .bao(&bao_vf6)
        .mol(&mol_vf6)
        .spin_constraint(SpinConstraint::Unrestricted(2, false))
        .complex_symmetric(false)
        .threshold(1e-14)
        .build()
        .unwrap()
        .to_generalised()
        .into();

    // ~~~~~~~~~~~~~~~~~~~~~~~~~
    // u D4h (ordinary, unitary)
    // ~~~~~~~~~~~~~~~~~~~~~~~~~

    let mut orbit_cg_u_oh_spatial_dyy = SlaterDeterminantSymmetryOrbit::builder()
        .group(&group_u_oh)
        .origin(&det_dyy_cg)
        .integrality_threshold(1e-14)
        .linear_independence_threshold(1e-14)
        .symmetry_transformation_kind(SymmetryTransformationKind::Spatial)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
        .build()
        .unwrap();
    let _ = orbit_cg_u_oh_spatial_dyy
        .calc_smat(Some(&sao_cg), None, true)
        .unwrap()
        .calc_xmat(false);
    assert_eq!(
        orbit_cg_u_oh_spatial_dyy.analyse_rep().unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new("||A|_(1g)| ⊕ ||E|_(g)|").unwrap()
    );

    let mut orbit_cg_u_oh_spatial_dxz = SlaterDeterminantSymmetryOrbit::builder()
        .group(&group_u_oh)
        .origin(&det_dxz_cg)
        .integrality_threshold(1e-14)
        .linear_independence_threshold(1e-14)
        .symmetry_transformation_kind(SymmetryTransformationKind::Spatial)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
        .build()
        .unwrap();
    let _ = orbit_cg_u_oh_spatial_dxz
        .calc_smat(Some(&sao_cg), None, true)
        .unwrap()
        .calc_xmat(false);
    assert_eq!(
        orbit_cg_u_oh_spatial_dxz.analyse_rep().unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new("||T|_(2g)|").unwrap()
    );

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // u D4h* (ordinary double, unitary)
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    let mut orbit_cg_u_oh_double_spin_spatial_dyy = SlaterDeterminantSymmetryOrbit::builder()
        .group(&group_u_oh_double)
        .origin(&det_dyy_cg)
        .integrality_threshold(1e-14)
        .linear_independence_threshold(1e-14)
        .symmetry_transformation_kind(SymmetryTransformationKind::SpinSpatial)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
        .build()
        .unwrap();
    let _ = orbit_cg_u_oh_double_spin_spatial_dyy
        .calc_smat(Some(&sao_cg), None, true)
        .unwrap()
        .calc_xmat(false);
    assert_eq!(
        orbit_cg_u_oh_double_spin_spatial_dyy.analyse_rep().unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new("||E~|_(1g)| ⊕ ||F~|_(g)|").unwrap()
    );

    let mut orbit_cg_u_oh_double_spin_spatial_dyy_nocayley =
        SlaterDeterminantSymmetryOrbit::builder()
            .group(&group_u_oh_double)
            .origin(&det_dyy_cg)
            .integrality_threshold(1e-14)
            .linear_independence_threshold(1e-14)
            .symmetry_transformation_kind(SymmetryTransformationKind::SpinSpatial)
            .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
            .build()
            .unwrap();
    let _ = orbit_cg_u_oh_double_spin_spatial_dyy_nocayley
        .calc_smat(Some(&sao_cg), None, false)
        .unwrap()
        .calc_xmat(false);
    assert_eq!(
        orbit_cg_u_oh_double_spin_spatial_dyy_nocayley
            .analyse_rep()
            .unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new("||E~|_(1g)| ⊕ ||F~|_(g)|").unwrap()
    );

    let mut orbit_cg_u_oh_double_spin_spatial_dxz = SlaterDeterminantSymmetryOrbit::builder()
        .group(&group_u_oh_double)
        .origin(&det_dxz_cg)
        .integrality_threshold(1e-14)
        .linear_independence_threshold(1e-14)
        .symmetry_transformation_kind(SymmetryTransformationKind::SpinSpatial)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
        .build()
        .unwrap();
    let _ = orbit_cg_u_oh_double_spin_spatial_dxz
        .calc_smat(Some(&sao_cg), None, true)
        .unwrap()
        .calc_xmat(false);
    assert_eq!(
        orbit_cg_u_oh_double_spin_spatial_dxz.analyse_rep().unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new("||E~|_(2g)| ⊕ ||F~|_(g)|").unwrap()
    );

    let mut orbit_cg_u_oh_double_spin_spatial_dxz_nocayley =
        SlaterDeterminantSymmetryOrbit::builder()
            .group(&group_u_oh_double)
            .origin(&det_dxz_cg)
            .integrality_threshold(1e-14)
            .linear_independence_threshold(1e-14)
            .symmetry_transformation_kind(SymmetryTransformationKind::SpinSpatial)
            .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
            .build()
            .unwrap();
    let _ = orbit_cg_u_oh_double_spin_spatial_dxz_nocayley
        .calc_smat(Some(&sao_cg), None, false)
        .unwrap()
        .calc_xmat(false);
    assert_eq!(
        orbit_cg_u_oh_double_spin_spatial_dxz_nocayley
            .analyse_rep()
            .unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new("||E~|_(2g)| ⊕ ||F~|_(g)|").unwrap()
    );
}
