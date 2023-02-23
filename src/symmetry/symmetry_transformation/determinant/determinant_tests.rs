use env_logger;
use nalgebra::Point3;
use ndarray::{array, concatenate, Axis};

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
    let tdet_c4p1 = det.transform(&c4p1).unwrap();
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
fn test_determinant_transformation_b3_timerev() {
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

    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol_b3, true)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false);
    let group = UnitaryRepresentedGroup::from_molecular_symmetry(&sym, None);
    let c3p1 = group.get_index(1).unwrap();
    let tcalpha_ref = array![
        [ 0.0,  0.0],
        [ 0.5, -sqr], [ sqr, -0.5], [0.0, 0.0],
        [ 0.0,  0.0],
        [-1.0,  sqr], [ 0.0,  0.5], [0.0, 0.0],
        [ 0.0,  0.0],
        [ 0.5,  sqr], [-sqr,  0.5], [0.0, 0.0]
    ];
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
        SpinConstraint::Unrestricted(2),
        1e-14,
    );

    let tdetunres_tr = detunres.transform_timerev().unwrap();
    let tdetunres_tr_ref = Determinant::<f64>::new(
        &[-cbeta.clone(), calpha.clone()],
        &[obeta.clone(), oalpha.clone()],
        &bao_b3,
        &mol_b3,
        SpinConstraint::Unrestricted(2),
        1e-14,
    );
    assert_eq!(tdetunres_tr, tdetunres_tr_ref);

    let tdetunres_c3p1_tr = detunres.transform(&c3p1).unwrap().transform_timerev().unwrap();
    let tdetunres_c3p1_tr_ref = Determinant::<f64>::new(
        &[tcalpha_ref.clone(), tcbeta_ref.clone()],
        &[obeta.clone(), oalpha.clone()],
        &bao_b3,
        &mol_b3,
        SpinConstraint::Unrestricted(2),
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
        SpinConstraint::Generalised(2),
        1e-14,
    );

    let tdetgen_tr = detgen.transform_timerev().unwrap();
    let tcgen_ref = concatenate![Axis(0), -cbeta, calpha];
    let tdetgen_tr_ref = Determinant::<f64>::new(
        &[tcgen_ref],
        &[ogen.clone()],
        &bao_b3,
        &mol_b3,
        SpinConstraint::Generalised(2),
        1e-14,
    );
    assert_eq!(tdetgen_tr, tdetgen_tr_ref);

    let tdetgen_c3p1_tr = detgen.transform(&c3p1).unwrap().transform_timerev().unwrap();
    let tcgen_ref = concatenate![Axis(0), tcalpha_ref, tcbeta_ref];
    let tdetgen_c3p1_tr_ref = Determinant::<f64>::new(
        &[tcgen_ref],
        &[ogen],
        &bao_b3,
        &mol_b3,
        SpinConstraint::Generalised(2),
        1e-14,
    );
    assert_eq!(tdetgen_c3p1_tr, tdetgen_c3p1_tr_ref);
}
