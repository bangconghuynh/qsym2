use env_logger;
use nalgebra::Point3;
use ndarray::{array, concatenate, Array2, Axis};
use num_complex::Complex;
use num_traits::Pow;

use crate::angmom::spinor_rotation_3d::SpinConstraint;
use crate::aux::ao_basis::{BasisAngularOrder, BasisAtom, BasisShell, CartOrder, ShellOrder};
use crate::aux::atom::{Atom, ElementMap};
use crate::aux::geometry::Transform;
use crate::aux::molecule::Molecule;
use crate::group::{GroupProperties, UnitaryRepresentedGroup};
use crate::symmetry::symmetry_core::{PreSymmetry, Symmetry};
use crate::symmetry::symmetry_group::SymmetryGroupProperties;
use crate::symmetry::symmetry_transformation::determinant::Determinant;
use crate::symmetry::symmetry_transformation::{SymmetryTransformable, TimeReversalTransformable};

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

    let bss_p = BasisShell::new(0, ShellOrder::Pure(true));
    let bsp_c = BasisShell::new(1, ShellOrder::Cart(CartOrder::lex(1)));
    let bsd_c = BasisShell::new(2, ShellOrder::Cart(CartOrder::lex(2)));

    let batm_b0 = BasisAtom::new(&atm_b0, &[bss_p.clone(), bsp_c.clone(), bsd_c.clone()]);
    let batm_f0 = BasisAtom::new(&atm_f0, &[bss_p.clone(), bsp_c.clone(), bsd_c.clone()]);
    let batm_f1 = BasisAtom::new(&atm_f1, &[bss_p.clone(), bsp_c.clone(), bsd_c.clone()]);
    let batm_f2 = BasisAtom::new(&atm_f2, &[bss_p.clone(), bsp_c.clone(), bsd_c.clone()]);
    let batm_f3 = BasisAtom::new(&atm_f3, &[bss_p.clone(), bsp_c.clone()]);

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

    let det = Determinant::<f64>::new(
        &[calpha],
        &[oalpha.clone()],
        &bao_bf4,
        &mol_bf4,
        SpinConstraint::Restricted(2),
        1e-14,
    );

    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol_bf4, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    let group = UnitaryRepresentedGroup::from_molecular_symmetry(&sym, None);

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
    let tdet_c4p1_ref = Determinant::<f64>::new(
        &[tcalpha_ref],
        &[oalpha],
        &bao_bf4,
        &mol_bf4,
        SpinConstraint::Restricted(2),
        1e-14,
    );
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

    let bsd_p = BasisShell::new(2, ShellOrder::Pure(true));

    let batm_s0 = BasisAtom::new(&atm_s0, &[bsd_p.clone()]);
    let batm_s1 = BasisAtom::new(&atm_s1, &[bsd_p.clone()]);
    let batm_s2 = BasisAtom::new(&atm_s2, &[bsd_p.clone()]);
    let batm_s3 = BasisAtom::new(&atm_s3, &[bsd_p.clone()]);

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
        .molecule(&mol_s4, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    let group = UnitaryRepresentedGroup::from_molecular_symmetry(&sym, None);

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
    let detunres = Determinant::<f64>::new(
        &[calpha.clone(), cbeta.clone()],
        &[oalpha.clone(), obeta.clone()],
        &bao_s4,
        &mol_s4,
        SpinConstraint::Unrestricted(2, false),
        1e-14,
    );

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
    let tdetunres_c4p1_ref = Determinant::<f64>::new(
        &[tcalpha_ref.clone(), tcbeta_ref.clone()],
        &[oalpha.clone(), obeta.clone()],
        &bao_s4,
        &mol_s4,
        SpinConstraint::Unrestricted(2, false),
        1e-14,
    );
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
    let detgen = Determinant::<f64>::new(
        &[cgen.clone()],
        &[ogen.clone()],
        &bao_s4,
        &mol_s4,
        SpinConstraint::Generalised(2, false),
        1e-14,
    );

    let tcalpha2_gen = concatenate!(Axis(0), cbeta2, Array2::zeros((20, 1)));
    let tcbeta2_gen = concatenate!(Axis(0), Array2::zeros((20, 1)), calpha2);
    let tcgen_ref = concatenate![Axis(1), tcalpha2_gen, tcbeta2_gen];
    let tdetgen_c4p1_ref = Determinant::<f64>::new(
        &[tcgen_ref],
        &[ogen.clone()],
        &bao_s4,
        &mol_s4,
        SpinConstraint::Generalised(2, false),
        1e-14,
    );
    let tdetgen_c4p1 = detgen.sym_transform_spatial(&c4p1).unwrap();
    assert_eq!(tdetgen_c4p1, tdetgen_c4p1_ref);

    // S1(+0.000, +0.000, +1.000)
    let s1zp1 = group.get_index(11).unwrap();
    let tdetgen_s1zp1_ref = Determinant::<f64>::new(
        &[-cgen.clone()],
        &[ogen.clone()],
        &bao_s4,
        &mol_s4,
        SpinConstraint::Generalised(2, false),
        1e-14,
    );
    let tdetgen_s1zp1 = detgen.sym_transform_spatial(&s1zp1).unwrap();
    assert_eq!(tdetgen_s1zp1, tdetgen_s1zp1_ref);

    // S1(+0.000, +1.000, +0.000)
    let s1yp1 = group.get_index(12).unwrap();
    let tdetgen_s1yp1_ref = Determinant::<f64>::new(
        &[cgen.clone()],
        &[ogen.clone()],
        &bao_s4,
        &mol_s4,
        SpinConstraint::Generalised(2, false),
        1e-14,
    );
    let tdetgen_s1yp1 = detgen.sym_transform_spatial(&s1yp1).unwrap();
    assert_eq!(tdetgen_s1yp1, tdetgen_s1yp1_ref);

    // S1(+1.000, +0.000, +0.000)
    let s1xp1 = group.get_index(12).unwrap();
    let tdetgen_s1xp1_ref = Determinant::<f64>::new(
        &[cgen.clone()],
        &[ogen.clone()],
        &bao_s4,
        &mol_s4,
        SpinConstraint::Generalised(2, false),
        1e-14,
    );
    let tdetgen_s1xp1 = detgen.sym_transform_spatial(&s1xp1).unwrap();
    assert_eq!(tdetgen_s1xp1, tdetgen_s1xp1_ref);

    // i
    let ip1 = group.get_index(8).unwrap();
    let tdetgen_ip1_ref = Determinant::<f64>::new(
        &[-cgen.clone()],
        &[ogen.clone()],
        &bao_s4,
        &mol_s4,
        SpinConstraint::Generalised(2, false),
        1e-14,
    );
    let tdetgen_ip1 = detgen.sym_transform_spatial(&ip1).unwrap();
    assert_eq!(tdetgen_ip1, tdetgen_ip1_ref);

    // S4(+0.000, +0.000, +1.000)
    let s4p1 = group.get_index(9).unwrap();
    let tcgen_s4p1_ref = concatenate![Axis(1), -tcalpha2_gen, -tcbeta2_gen];
    let tdetgen_s4p1_ref = Determinant::<f64>::new(
        &[tcgen_s4p1_ref],
        &[ogen.clone()],
        &bao_s4,
        &mol_s4,
        SpinConstraint::Generalised(2, false),
        1e-14,
    );
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

    let bss_p = BasisShell::new(0, ShellOrder::Pure(true));
    let bsp_c = BasisShell::new(1, ShellOrder::Cart(CartOrder::lex(1)));

    let batm_b0 = BasisAtom::new(&atm_b0, &[bss_p.clone(), bsp_c.clone()]);
    let batm_b1 = BasisAtom::new(&atm_b1, &[bss_p.clone(), bsp_c.clone()]);
    let batm_b2 = BasisAtom::new(&atm_b2, &[bss_p.clone(), bsp_c.clone()]);

    let bao_b3 = BasisAngularOrder::new(&[batm_b0, batm_b1, batm_b2]);
    let mol_b3 =
        Molecule::from_atoms(&[atm_b0.clone(), atm_b1.clone(), atm_b2.clone()], 1e-7).recentre();

    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol_b3, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    let group = UnitaryRepresentedGroup::from_molecular_symmetry(&sym, None);
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

    let detunres = Determinant::<f64>::new(
        &[calpha.clone(), cbeta.clone()],
        &[oalpha.clone(), obeta.clone()],
        &bao_b3,
        &mol_b3,
        SpinConstraint::Unrestricted(2, false),
        1e-14,
    );

    let tdetunres_tr = detunres.transform_timerev().unwrap();
    let tdetunres_tr_ref = Determinant::<f64>::new(
        &[-cbeta.clone(), calpha.clone()],
        &[obeta.clone(), oalpha.clone()],
        &bao_b3,
        &mol_b3,
        SpinConstraint::Unrestricted(2, false),
        1e-14,
    );
    assert_eq!(tdetunres_tr, tdetunres_tr_ref);

    let tdetunres_c3p1_tr = detunres
        .sym_transform_spatial(&c3p1)
        .unwrap()
        .transform_timerev()
        .unwrap();
    let tdetunres_c3p1_tr_ref = Determinant::<f64>::new(
        &[tcalpha_ref.clone(), tcbeta_ref.clone()],
        &[obeta.clone(), oalpha.clone()],
        &bao_b3,
        &mol_b3,
        SpinConstraint::Unrestricted(2, false),
        1e-14,
    );
    assert_eq!(tdetunres_c3p1_tr, tdetunres_c3p1_tr_ref);

    // Generalised spin constraint, spin-projection-mixed orbitals
    let cgen = concatenate![Axis(0), calpha, cbeta];
    let ogen = array![1.0, 1.0];

    let detgen = Determinant::<f64>::new(
        &[cgen],
        &[ogen.clone()],
        &bao_b3,
        &mol_b3,
        SpinConstraint::Generalised(2, false),
        1e-14,
    );

    let tdetgen_tr = detgen.transform_timerev().unwrap();
    let tcgen_ref = concatenate![Axis(0), -cbeta, calpha];
    let tdetgen_tr_ref = Determinant::<f64>::new(
        &[tcgen_ref],
        &[ogen.clone()],
        &bao_b3,
        &mol_b3,
        SpinConstraint::Generalised(2, false),
        1e-14,
    );
    assert_eq!(tdetgen_tr, tdetgen_tr_ref);

    let tdetgen_c3p1_tr = detgen
        .sym_transform_spatial(&c3p1)
        .unwrap()
        .transform_timerev()
        .unwrap();
    let tcgen_ref = concatenate![Axis(0), tcalpha_ref, tcbeta_ref];
    let tdetgen_c3p1_tr_ref = Determinant::<f64>::new(
        &[tcgen_ref],
        &[ogen],
        &bao_b3,
        &mol_b3,
        SpinConstraint::Generalised(2, false),
        1e-14,
    );
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
    let batm_c1 = BasisAtom::new(&atm_c1, &[bsp_c.clone()]);

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

    let detgen = Determinant::<C128>::new(
        &[cgen],
        &[ogen.clone()],
        &bao_c2,
        &mol_c2,
        SpinConstraint::Generalised(2, false),
        1e-14,
    );

    let tcalpha_gen_ref = concatenate!(Axis(0), Array2::zeros((6, 2)), calpha.map(|x| x.conj()));
    let tcbeta_gen_ref = concatenate!(Axis(0), -cbeta.map(|x| x.conj()), Array2::zeros((6, 2)));
    let tcgen_ref = concatenate![Axis(1), tcalpha_gen_ref, tcbeta_gen_ref];
    let tdetgen_tr_ref = Determinant::<C128>::new(
        &[tcgen_ref],
        &[ogen.clone()],
        &bao_c2,
        &mol_c2,
        SpinConstraint::Generalised(2, false),
        1e-14,
    );
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

    let bss_p = BasisShell::new(0, ShellOrder::Pure(true));
    let bsp_c = BasisShell::new(1, ShellOrder::Cart(CartOrder::lex(1)));

    let batm_c0 = BasisAtom::new(&atm_c0, &[bss_p.clone(), bsp_c.clone()]);
    let batm_c1 = BasisAtom::new(&atm_c1, &[bss_p.clone(), bsp_c.clone()]);
    let batm_c2 = BasisAtom::new(&atm_c2, &[bss_p.clone(), bsp_c.clone()]);

    let bao_c3 = BasisAngularOrder::new(&[batm_c0, batm_c1, batm_c2]);
    let mol_c3 =
        Molecule::from_atoms(&[atm_c0.clone(), atm_c1.clone(), atm_c2.clone()], 1e-7).recentre();
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol_c3, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    let group = UnitaryRepresentedGroup::from_molecular_symmetry(&sym, None).to_double_group();

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
    let detgen: Determinant<C128> = Determinant::new(
        &[cgen.clone()],
        &[ogen.clone()],
        &bao_c3,
        &mol_c3,
        SpinConstraint::Generalised(2, false),
        1e-14,
    )
    .into();

    // ----------------
    // Proper rotations
    // ----------------

    let sqr = 2.0f64.sqrt() / 2.0;
    let c2_nsr_p1 = group.get_index(2).unwrap();
    let tcalpha_gen = concatenate!(
        Axis(0),
        Array2::zeros((12, 2)),
        C128::new(1.0, -1.0) * (calpha.clone() * sqr).map(|x| C128::from(x))
    );
    let tcbeta_gen = concatenate!(
        Axis(0),
        -C128::new(1.0, 1.0) * (cbeta.clone() * sqr).map(|x| C128::from(x)),
        Array2::zeros((12, 2)),
    );
    let tcgen_ref = concatenate![Axis(1), tcalpha_gen, tcbeta_gen];
    let tdetgen_c2_nsr_p1_ref: Determinant<C128> = Determinant::new(
        &[tcgen_ref.clone()],
        &[ogen.clone()],
        &bao_c3,
        &mol_c3,
        SpinConstraint::Generalised(2, false),
        1e-14,
    );
    let tdetgen_c2_nsr_p1 = detgen.sym_transform_spin(&c2_nsr_p1).unwrap();
    assert_eq!(tdetgen_c2_nsr_p1, tdetgen_c2_nsr_p1_ref);

    let c2_nsr_p2 = (&c2_nsr_p1).pow(2);
    let tdetgen_c2_nsr_p2_ref: Determinant<C128> = Determinant::new(
        &[-cgen],
        &[ogen.clone()],
        &bao_c3,
        &mol_c3,
        SpinConstraint::Generalised(2, false),
        1e-14,
    )
    .into();
    let tdetgen_c2_nsr_p2 = detgen.sym_transform_spin(&c2_nsr_p2).unwrap();
    assert_eq!(tdetgen_c2_nsr_p2, tdetgen_c2_nsr_p2_ref);

    let e_isr = group.get_index(1).unwrap();
    let tdetgen_e_isr = detgen.sym_transform_spin(&e_isr).unwrap();
    assert_eq!(tdetgen_e_isr, tdetgen_c2_nsr_p2_ref);

    let c2_nsr_p3 = (&c2_nsr_p1).pow(3);
    let tdetgen_c2_nsr_p3_ref: Determinant<C128> = Determinant::new(
        &[-tcgen_ref],
        &[ogen.clone()],
        &bao_c3,
        &mol_c3,
        SpinConstraint::Generalised(2, false),
        1e-14,
    )
    .into();
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
        C128::new(0.0, -1.0) * calpha.map(|x| C128::from(x)),
        Array2::zeros((12, 2))
    );
    let sxy_tcbeta_gen = concatenate!(
        Axis(0),
        Array2::zeros((12, 2)),
        C128::new(0.0, 1.0) * cbeta.map(|x| C128::from(x))
    );
    let sxy_tcgen_ref = concatenate![Axis(1), sxy_tcalpha_gen, sxy_tcbeta_gen];

    let sxy_nsr_p1 = group.get_index(4).unwrap();
    let tdetgen_sxy_nsr_ref: Determinant<C128> = Determinant::new(
        &[sxy_tcgen_ref],
        &[ogen.clone()],
        &bao_c3,
        &mol_c3,
        SpinConstraint::Generalised(2, false),
        1e-14,
    )
    .into();
    let tdetgen_sxy_nsr = detgen.sym_transform_spin(&sxy_nsr_p1).unwrap();
    assert_eq!(tdetgen_sxy_nsr, tdetgen_sxy_nsr_ref);

    let tdetgen_sxy_nsr_p2 = detgen.sym_transform_spin(&(&sxy_nsr_p1).pow(2)).unwrap();
    assert_eq!(tdetgen_sxy_nsr_p2, tdetgen_e_isr);
}
