use ndarray::{array, Axis};
use env_logger;

use crate::angmom::spinor_rotation_3d::SpinConstraint;
use crate::aux::ao_basis::{BasisAngularOrder, BasisAtom, BasisShell, CartOrder, ShellOrder};
use crate::aux::atom::{Atom, ElementMap};
use crate::aux::molecule::Molecule;
use crate::group::{GroupProperties, UnitaryRepresentedGroup};
use crate::symmetry::symmetry_core::{PreSymmetry, Symmetry};
use crate::symmetry::symmetry_group::SymmetryGroupProperties;
use crate::symmetry::symmetry_transformation::determinant::Determinant;
use crate::symmetry::symmetry_transformation::SymmetryTransformable;

#[test]
fn test_determinant_transformation_bf4_sqpl() {
    env_logger::init();
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
        1e-14
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

    let tdet = det.transform(&c4p1).unwrap();

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
    let tdet_ref = Determinant::<f64>::new(
        &[tcalpha_ref],
        &[oalpha],
        &bao_bf4,
        &mol_bf4,
        SpinConstraint::Restricted(2),
        1e-14
    );
    assert_eq!(tdet, tdet_ref);
}
