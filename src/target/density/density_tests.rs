// use env_logger;
use itertools::Itertools;
use nalgebra::{Point3, Vector3};
use ndarray::{array, concatenate, s, Array2, Axis};
use ndarray_linalg::assert::close_l2;
use num_complex::Complex;
use serial_test::serial;

use crate::analysis::{EigenvalueComparisonMode, RepAnalysis};
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
    ComplexConjugationTransformable, SymmetryTransformable, SymmetryTransformationKind,
    TimeReversalTransformable,
};
use crate::target::density::density_analysis::DensitySymmetryOrbit;
use crate::target::density::Density;
use crate::target::determinant::SlaterDeterminant;

type C128 = Complex<f64>;

#[test]
fn test_density_transformation_bf4_sqpl() {
    // env_logger::init();
    let emap = ElementMap::new();
    let atm_b0 = Atom::from_xyz("B 0.0 0.0 0.0", &emap, 1e-7).unwrap();
    let atm_f0 = Atom::from_xyz("F 1.0 0.0 0.0", &emap, 1e-7).unwrap();
    let atm_f1 = Atom::from_xyz("F 0.0 1.0 0.0", &emap, 1e-7).unwrap();
    let atm_f2 = Atom::from_xyz("F -1.0 0.0 0.0", &emap, 1e-7).unwrap();
    let atm_f3 = Atom::from_xyz("F 0.0 -1.0 0.0", &emap, 1e-7).unwrap();

    let bsp_c = BasisShell::new(1, ShellOrder::Cart(CartOrder::lex(1)));

    let batm_b0 = BasisAtom::new(&atm_b0, &[bsp_c.clone()]);
    let batm_f0 = BasisAtom::new(&atm_f0, &[bsp_c.clone()]);
    let batm_f1 = BasisAtom::new(&atm_f1, &[bsp_c.clone()]);
    let batm_f2 = BasisAtom::new(&atm_f2, &[bsp_c.clone()]);
    let batm_f3 = BasisAtom::new(&atm_f3, &[bsp_c]);

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
    let d_bpxpy_f0py_f1pz = array![
        [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ];

    let den = Density::<f64>::builder()
        .density_matrix(d_bpxpy_f0py_f1pz)
        .bao(&bao_bf4)
        .mol(&mol_bf4)
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
    let tden_c4p1 = den.sym_transform_spatial(&c4p1).unwrap();
    #[rustfmt::skip]
    let d_bpypx_f1px_f2pz = array![
        [1.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ];
    let tden_c4p1_ref = Density::<f64>::builder()
        .density_matrix(d_bpypx_f1px_f2pz)
        .bao(&bao_bf4)
        .mol(&mol_bf4)
        .complex_symmetric(false)
        .threshold(1e-14)
        .build()
        .unwrap();
    assert_eq!(tden_c4p1, tden_c4p1_ref);
}

#[test]
fn test_density_transformation_s4_sqpl() {
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

    let mut d_s0dyzdz2 = Array2::<f64>::zeros((20, 20));
    d_s0dyzdz2[(1, 1)] = 1.9;
    d_s0dyzdz2[(2, 2)] = 2.1;
    let den = Density::<f64>::builder()
        .density_matrix(d_s0dyzdz2)
        .bao(&bao_s4)
        .mol(&mol_s4)
        .complex_symmetric(false)
        .threshold(1e-14)
        .build()
        .unwrap();

    let c4p1 = group.get_index(1).unwrap();
    let mut d_s1dxzdz2 = Array2::<f64>::zeros((20, 20));
    d_s1dxzdz2[(7, 7)] = 2.1;
    d_s1dxzdz2[(8, 8)] = 1.9;
    let tden_c4p1_ref = Density::<f64>::builder()
        .density_matrix(d_s1dxzdz2)
        .bao(&bao_s4)
        .mol(&mol_s4)
        .complex_symmetric(false)
        .threshold(1e-14)
        .build()
        .unwrap();
    let tden_c4p1 = den.sym_transform_spatial(&c4p1).unwrap();
    assert_eq!(tden_c4p1, tden_c4p1_ref);

    #[rustfmt::skip]
    let calpha2 = array![
        [0.0], [ 0.0], [0.0], [ 1.0], [0.0],
        [0.0], [ 0.0], [0.0], [-1.0], [0.0],
        [0.0], [ 0.0], [0.0], [-1.0], [0.0],
        [0.0], [ 0.0], [0.0], [ 1.0], [0.0]
    ];
    #[rustfmt::skip]
    let cbeta2 = array![
        [0.0], [ 1.0], [0.0], [ 0.0], [0.0],
        [0.0], [ 1.0], [0.0], [ 0.0], [0.0],
        [0.0], [-1.0], [0.0], [ 0.0], [0.0],
        [0.0], [-1.0], [0.0], [ 0.0], [0.0]
    ];
    let det2 = SlaterDeterminant::<f64, SpinConstraint>::builder()
        .coefficients(&[calpha2, cbeta2])
        .occupations(&[array![1.0], array![0.8]])
        .baos(vec![&bao_s4])
        .mol(&mol_s4)
        .structure_constraint(SpinConstraint::Unrestricted(2, false))
        .complex_symmetric(false)
        .threshold(1e-14)
        .build()
        .unwrap();

    let den2s = det2.to_densities().unwrap();
    let den2_a = &den2s[0];
    let den2_b = &den2s[1];

    // C4(+0.000, +0.000, +1.000)
    let tdet2_c4p1 = det2.sym_transform_spatial(&c4p1).unwrap();
    let tden2_c4p1 = tdet2_c4p1.to_densities().unwrap();
    let tden2_a_c4p1 = den2_a.sym_transform_spatial(&c4p1).unwrap();
    let tden2_b_c4p1 = den2_b.sym_transform_spatial(&c4p1).unwrap();
    assert_eq!(tden2_a_c4p1, tden2_c4p1[0]);
    assert_eq!(tden2_b_c4p1, tden2_c4p1[1]);

    // S1(+0.000, +0.000, +1.000)
    let s1zp1 = group.get_index(11).unwrap();
    let tdet2_s1zp1 = det2.sym_transform_spatial(&s1zp1).unwrap();
    let tden2_s1zp1 = tdet2_s1zp1.to_densities().unwrap();
    let tden2_a_s1zp1 = den2_a.sym_transform_spatial(&s1zp1).unwrap();
    let tden2_b_s1zp1 = den2_b.sym_transform_spatial(&s1zp1).unwrap();
    assert_eq!(tden2_a_s1zp1, tden2_s1zp1[0]);
    assert_eq!(tden2_b_s1zp1, tden2_s1zp1[1]);

    // S1(+0.000, +1.000, +0.000)
    let s1yp1 = group.get_index(12).unwrap();
    let tdet2_s1yp1 = det2.sym_transform_spatial(&s1yp1).unwrap();
    let tden2_s1yp1 = tdet2_s1yp1.to_densities().unwrap();
    let tden2_a_s1yp1 = den2_a.sym_transform_spatial(&s1yp1).unwrap();
    let tden2_b_s1yp1 = den2_b.sym_transform_spatial(&s1yp1).unwrap();
    assert_eq!(tden2_a_s1yp1, tden2_s1yp1[0]);
    assert_eq!(tden2_b_s1yp1, tden2_s1yp1[1]);

    // S1(+1.000, +0.000, +0.000)
    let s1xp1 = group.get_index(14).unwrap();
    let tdet2_s1xp1 = det2.sym_transform_spatial(&s1xp1).unwrap();
    let tden2_s1xp1 = tdet2_s1xp1.to_densities().unwrap();
    let tden2_a_s1xp1 = den2_a.sym_transform_spatial(&s1xp1).unwrap();
    let tden2_b_s1xp1 = den2_b.sym_transform_spatial(&s1xp1).unwrap();
    assert_eq!(tden2_a_s1xp1, tden2_s1xp1[0]);
    assert_eq!(tden2_b_s1xp1, tden2_s1xp1[1]);

    // i
    let ip1 = group.get_index(8).unwrap();
    let tdet2_ip1 = det2.sym_transform_spatial(&ip1).unwrap();
    let tden2_ip1 = tdet2_ip1.to_densities().unwrap();
    let tden2_a_ip1 = den2_a.sym_transform_spatial(&ip1).unwrap();
    let tden2_b_ip1 = den2_b.sym_transform_spatial(&ip1).unwrap();
    assert_eq!(tden2_a_ip1, tden2_ip1[0]);
    assert_eq!(tden2_b_ip1, tden2_ip1[1]);

    // S4(+0.000, +0.000, +1.000)
    let s4p1 = group.get_index(9).unwrap();
    let tdet2_s4p1 = det2.sym_transform_spatial(&s4p1).unwrap();
    let tden2_s4p1 = tdet2_s4p1.to_densities().unwrap();
    let tden2_a_s4p1 = den2_a.sym_transform_spatial(&s4p1).unwrap();
    let tden2_b_s4p1 = den2_b.sym_transform_spatial(&s4p1).unwrap();
    assert_eq!(tden2_a_s4p1, tden2_s4p1[0]);
    assert_eq!(tden2_b_s4p1, tden2_s4p1[1]);
}

#[test]
fn test_density_transformation_b3_real_timerev() {
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

    // Generalised spin constraint, spin-projection-mixed orbitals
    let cgen = concatenate![Axis(0), calpha, cbeta];
    let ogen = array![1.0, 1.0];

    let detgen = SlaterDeterminant::<f64, SpinConstraint>::builder()
        .coefficients(&[cgen])
        .occupations(&[ogen.clone()])
        .baos(vec![&bao_b3, &bao_b3])
        .mol(&mol_b3)
        .structure_constraint(SpinConstraint::Generalised(2, false))
        .complex_symmetric(false)
        .threshold(1e-14)
        .build()
        .unwrap();

    let da = calpha.dot(&calpha.t());
    let db = cbeta.dot(&cbeta.t());
    let den_a = Density::<f64>::builder()
        .density_matrix(da)
        .bao(&bao_b3)
        .mol(&mol_b3)
        .complex_symmetric(false)
        .threshold(1e-14)
        .build()
        .unwrap();
    let den_b = Density::<f64>::builder()
        .density_matrix(db)
        .bao(&bao_b3)
        .mol(&mol_b3)
        .complex_symmetric(false)
        .threshold(1e-14)
        .build()
        .unwrap();

    let tdetgen_tr = detgen.transform_timerev().unwrap();
    let tden_a_tr = den_a.transform_timerev().unwrap();
    let tden_b_tr = den_b.transform_timerev().unwrap();

    let tdgen_tr_ref = tdetgen_tr.coefficients()[0].dot(&tdetgen_tr.coefficients()[0].t());
    let tda_tr_ref = tdgen_tr_ref.slice(s![0..12, 0..12]);
    let tdb_tr_ref = tdgen_tr_ref.slice(s![12..24, 12..24]);

    // `den_a` and `den_b` have no knowledge of spins, because they are spatial densities. Hence,
    // time-reversal simply causes complex conjugation on them (which has no effects in this case
    // since they are all real). However, `detgen` has a spin structure, and time reversal brings
    // about spin flip.
    close_l2(tden_a_tr.density_matrix(), &tdb_tr_ref, 1e-14);
    close_l2(tden_b_tr.density_matrix(), &tda_tr_ref, 1e-14);

    let tdetgen_c3p1_tr = detgen
        .sym_transform_spatial(&c3p1)
        .unwrap()
        .transform_timerev()
        .unwrap();
    let tdengen_c3p1_tr = tdetgen_c3p1_tr.to_densities().unwrap();
    let tden_a_c3p1_tr = den_a
        .sym_transform_spatial(&c3p1)
        .unwrap()
        .transform_timerev()
        .unwrap();
    let tden_b_c3p1_tr = den_b
        .sym_transform_spatial(&c3p1)
        .unwrap()
        .transform_timerev()
        .unwrap();

    // `den_a` and `den_b` are spatial, so not affected by any spin transformations. However,
    // `detgen` has a spin structure and so can be transformed by spin transformations. The result
    // is that the spin blocks from spin-transformed `detgen` are different from `den_a` and
    // `den_b`.
    assert_ne!(tden_a_c3p1_tr, tdengen_c3p1_tr[0]);
    assert_ne!(tden_b_c3p1_tr, tdengen_c3p1_tr[1]);
}

#[test]
fn test_density_transformation_c2_complex_timerev() {
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

    let detgen = SlaterDeterminant::<C128, SpinConstraint>::builder()
        .coefficients(&[cgen])
        .occupations(&[ogen.clone()])
        .baos(vec![&bao_c2, &bao_c2])
        .mol(&mol_c2)
        .structure_constraint(SpinConstraint::Generalised(2, false))
        .complex_symmetric(false)
        .threshold(1e-14)
        .build()
        .unwrap();
    let dengens = detgen.to_densities().unwrap();
    let den_a = &dengens[0];
    let den_b = &dengens[1];

    let tdetgen_tr = detgen.transform_timerev().unwrap();
    let tden_a_tr = den_a.transform_timerev().unwrap();
    let tden_b_tr = den_b.transform_timerev().unwrap();
    let tdgen_tr_ref =
        tdetgen_tr.coefficients()[0].dot(&tdetgen_tr.coefficients()[0].t().map(C128::conj));

    // `den_a` and `den_b` have no knowledge of spins, because they are spatial densities. Hence,
    // time-reversal simply causes complex conjugation on them. However, `detgen` has a spin
    // structure, and time reversal brings about spin flip.
    assert_eq!(tden_a_tr, den_a.transform_cc().unwrap());
    assert_eq!(tden_b_tr, den_b.transform_cc().unwrap());
    close_l2(
        tden_b_tr.density_matrix(),
        &tdgen_tr_ref.slice(s![0..6, 0..6]),
        1e-14,
    );
    close_l2(
        tden_a_tr.density_matrix(),
        &tdgen_tr_ref.slice(s![6..12, 6..12]),
        1e-14,
    );
}

#[test]
fn test_density_transformation_h4_spin_spatial_rotation_composition() {
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
    let detgen: SlaterDeterminant<C128, SpinConstraint> =
        SlaterDeterminant::<f64, SpinConstraint>::builder()
            .coefficients(&[cgen])
            .occupations(&[ogen])
            .baos(vec![&bao_h4, &bao_h4])
            .mol(&mol_h4)
            .structure_constraint(SpinConstraint::Generalised(2, false))
            .complex_symmetric(false)
            .threshold(1e-12)
            .build()
            .unwrap()
            .into();
    let dengens = detgen.to_densities().unwrap();
    let dengen = &dengens[0];

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

            let spatial_tdengen_ij = dengen
                .sym_transform_spatial(op_j)
                .unwrap()
                .sym_transform_spatial(op_i)
                .unwrap();
            let spatial_tdengen_k = dengen.sym_transform_spatial(&op_k).unwrap();
            assert_eq!(spatial_tdengen_k, spatial_tdengen_ij);

            let spin_tdengen_ij = dengen
                .sym_transform_spin(op_j)
                .unwrap()
                .sym_transform_spin(op_i)
                .unwrap();
            let spin_tdengen_k = dengen.sym_transform_spin(&op_k).unwrap();
            assert_eq!(spin_tdengen_k, spin_tdengen_ij);

            let spin_spatial_tdengen_ij = dengen
                .sym_transform_spin_spatial(op_j)
                .unwrap()
                .sym_transform_spin_spatial(op_i)
                .unwrap();
            let spin_spatial_tdengen_k = dengen.sym_transform_spin_spatial(&op_k).unwrap();
            assert_eq!(spin_spatial_tdengen_k, spin_spatial_tdengen_ij);
        });
}

#[cfg(feature = "integrals")]
#[test]
#[serial]
fn test_density_orbit_rep_analysis_s4_sqpl_pxpy() {
    use crate::basis::ao_integrals::*;
    use crate::integrals::shell_tuple::build_shell_tuple_collection;

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

    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol_s4)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false).unwrap();
    let group_u_d4h = UnitaryRepresentedGroup::from_molecular_symmetry(&sym, None).unwrap();

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

    let mut orbit_ru_u_d4h_spatial_orbital_density_a = DensitySymmetryOrbit::builder()
        .group(&group_u_d4h)
        .origin(&dena_ru)
        .integrality_threshold(1e-14)
        .linear_independence_threshold(1e-14)
        .symmetry_transformation_kind(SymmetryTransformationKind::Spatial)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
        .build()
        .unwrap();
    let _ = orbit_ru_u_d4h_spatial_orbital_density_a
        .calc_smat(Some(sao_ru), None, true)
        .unwrap()
        .calc_xmat(false);
    assert_eq!(
        orbit_ru_u_d4h_spatial_orbital_density_a
            .analyse_rep()
            .unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new(
            "||A|_(1g)| ⊕ ||A|_(2g)| ⊕ ||B|_(1g)| ⊕ ||B|_(2g)| ⊕ 2||E|_(u)|"
        )
        .unwrap()
    );

    let mut orbit_ru_u_d4h_spatial_orbital_density_b = DensitySymmetryOrbit::builder()
        .group(&group_u_d4h)
        .origin(&denb_ru)
        .integrality_threshold(1e-14)
        .linear_independence_threshold(1e-14)
        .symmetry_transformation_kind(SymmetryTransformationKind::Spatial)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
        .build()
        .unwrap();
    let _ = orbit_ru_u_d4h_spatial_orbital_density_b
        .calc_smat(Some(sao_ru), None, true)
        .unwrap()
        .calc_xmat(false);
    assert_eq!(
        orbit_ru_u_d4h_spatial_orbital_density_b
            .analyse_rep()
            .unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new(
            "||A|_(1g)| ⊕ ||A|_(2g)| ⊕ ||B|_(1g)| ⊕ ||B|_(2g)| ⊕ 2||E|_(u)|"
        )
        .unwrap()
    );

    let mut orbit_ru_u_d4h_spatial_orbital_density_total = DensitySymmetryOrbit::builder()
        .group(&group_u_d4h)
        .origin(&dentot_ru)
        .integrality_threshold(1e-14)
        .linear_independence_threshold(1e-14)
        .symmetry_transformation_kind(SymmetryTransformationKind::Spatial)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
        .build()
        .unwrap();
    let _ = orbit_ru_u_d4h_spatial_orbital_density_total
        .calc_smat(Some(sao_ru), None, true)
        .unwrap()
        .calc_xmat(false);
    assert_eq!(
        orbit_ru_u_d4h_spatial_orbital_density_total
            .analyse_rep()
            .unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new("||A|_(1g)| ⊕ ||B|_(2g)| ⊕ ||E|_(u)|")
            .unwrap()
    );

    let mut orbit_ru_u_d4h_spatial_orbital_density_spin = DensitySymmetryOrbit::builder()
        .group(&group_u_d4h)
        .origin(&denspin_ru)
        .integrality_threshold(1e-14)
        .linear_independence_threshold(1e-14)
        .symmetry_transformation_kind(SymmetryTransformationKind::Spatial)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
        .build()
        .unwrap();
    let _ = orbit_ru_u_d4h_spatial_orbital_density_spin
        .calc_smat(Some(sao_ru), None, true)
        .unwrap()
        .calc_xmat(false);
    assert_eq!(
        orbit_ru_u_d4h_spatial_orbital_density_spin
            .analyse_rep()
            .unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new("||A|_(2g)| ⊕ ||B|_(1g)| ⊕ ||E|_(u)|")
            .unwrap()
    );
}

#[cfg(feature = "integrals")]
#[test]
#[serial]
fn test_density_orbit_rep_analysis_s4_sqpl_pypz() {
    use crate::basis::ao_integrals::*;
    use crate::integrals::shell_tuple::build_shell_tuple_collection;

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

    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol_s4)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false).unwrap();
    let group_u_d4h = UnitaryRepresentedGroup::from_molecular_symmetry(&sym, None).unwrap();

    let mut sym_tr = Symmetry::new();
    sym_tr.analyse(&presym, true).unwrap();
    let group_u_grey_d4h = UnitaryRepresentedGroup::from_molecular_symmetry(&sym_tr, None).unwrap();
    let group_m_grey_d4h =
        MagneticRepresentedGroup::from_molecular_symmetry(&sym_tr, None).unwrap();

    let mut mol_s4_bz = mol_s4.clone();
    mol_s4_bz.set_magnetic_field(Some(0.1 * Vector3::z()));
    let presym_bz = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol_s4_bz)
        .build()
        .unwrap();

    let mut sym_bz = Symmetry::new();
    sym_bz.analyse(&presym_bz, false).unwrap();
    let group_u_c4h = UnitaryRepresentedGroup::from_molecular_symmetry(&sym_bz, None).unwrap();

    let mut sym_bz_tr = Symmetry::new();
    sym_bz_tr.analyse(&presym_bz, true).unwrap();
    let group_u_bw_d4h_c4h =
        UnitaryRepresentedGroup::from_molecular_symmetry(&sym_bz_tr, None).unwrap();
    let group_m_bw_d4h_c4h =
        MagneticRepresentedGroup::from_molecular_symmetry(&sym_bz_tr, None).unwrap();

    // -----------------
    // Orbital densities
    // -----------------

    // S0pz
    #[rustfmt::skip]
    let calpha = array![
        [0.0], [1.0], [0.0],
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

    let det_cu: SlaterDeterminant<C128, SpinConstraint> = det_ru.clone().into();
    let dens_cu = det_cu.to_densities().unwrap();
    let dena_cu = &dens_cu[0];
    let denb_cu = &dens_cu[1];
    let dentot_cu = dena_cu + denb_cu;
    let denspin_cu = dena_cu - denb_cu;

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

    let mut bscs_bz = bscs.clone();
    bscs_bz.apply_magnetic_field(&(0.1 * Vector3::z()), &Point3::origin());
    let stc_bz = build_shell_tuple_collection![
        <s1, s2, s3, s4>;
        false, false, false, false;
        &bscs, &bscs, &bscs, &bscs;
        C128
    ];
    let ovs_bz = stc_bz.overlap([0, 0, 0, 0]);
    let sao_cu_bz = &ovs_bz[0];

    // ~~~~~~~~~~~~~~~~~~~~~~~~~
    // u D4h (ordinary, unitary)
    // ~~~~~~~~~~~~~~~~~~~~~~~~~

    let mut orbit_ru_u_d4h_spatial_orbital_density_a = DensitySymmetryOrbit::builder()
        .group(&group_u_d4h)
        .origin(&dena_ru)
        .integrality_threshold(1e-14)
        .linear_independence_threshold(1e-14)
        .symmetry_transformation_kind(SymmetryTransformationKind::Spatial)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
        .build()
        .unwrap();
    let _ = orbit_ru_u_d4h_spatial_orbital_density_a
        .calc_smat(Some(sao_ru), None, true)
        .unwrap()
        .calc_xmat(false);
    assert_eq!(
        orbit_ru_u_d4h_spatial_orbital_density_a
            .analyse_rep()
            .unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new("||A|_(1g)| ⊕ ||B|_(2g)| ⊕ ||E|_(u)|")
            .unwrap()
    );

    let mut orbit_ru_u_d4h_spatial_orbital_density_b = DensitySymmetryOrbit::builder()
        .group(&group_u_d4h)
        .origin(&denb_ru)
        .integrality_threshold(1e-14)
        .linear_independence_threshold(1e-14)
        .symmetry_transformation_kind(SymmetryTransformationKind::Spatial)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
        .build()
        .unwrap();
    let _ = orbit_ru_u_d4h_spatial_orbital_density_b
        .calc_smat(Some(sao_ru), None, true)
        .unwrap()
        .calc_xmat(false);
    assert_eq!(
        orbit_ru_u_d4h_spatial_orbital_density_b
            .analyse_rep()
            .unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new(
            "||A|_(1g)| ⊕ ||A|_(2g)| ⊕ ||B|_(1g)| ⊕ ||B|_(2g)| ⊕ 2||E|_(u)|"
        )
        .unwrap()
    );

    let mut orbit_ru_u_d4h_spatial_orbital_density_total = DensitySymmetryOrbit::builder()
        .group(&group_u_d4h)
        .origin(&dentot_ru)
        .integrality_threshold(1e-14)
        .linear_independence_threshold(1e-14)
        .symmetry_transformation_kind(SymmetryTransformationKind::Spatial)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
        .build()
        .unwrap();
    let _ = orbit_ru_u_d4h_spatial_orbital_density_total
        .calc_smat(Some(sao_ru), None, true)
        .unwrap()
        .calc_xmat(false);
    assert_eq!(
        orbit_ru_u_d4h_spatial_orbital_density_total
            .analyse_rep()
            .unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new(
            "||A|_(1g)| ⊕ ||A|_(2g)| ⊕ ||B|_(1g)| ⊕ ||B|_(2g)| ⊕ 2||E|_(u)|"
        )
        .unwrap()
    );

    let mut orbit_ru_u_d4h_spatial_orbital_density_spin = DensitySymmetryOrbit::builder()
        .group(&group_u_d4h)
        .origin(&denspin_ru)
        .integrality_threshold(1e-14)
        .linear_independence_threshold(1e-14)
        .symmetry_transformation_kind(SymmetryTransformationKind::Spatial)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
        .build()
        .unwrap();
    let _ = orbit_ru_u_d4h_spatial_orbital_density_spin
        .calc_smat(Some(sao_ru), None, true)
        .unwrap()
        .calc_xmat(false);
    assert_eq!(
        orbit_ru_u_d4h_spatial_orbital_density_spin
            .analyse_rep()
            .unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new(
            "||A|_(1g)| ⊕ ||A|_(2g)| ⊕ ||B|_(1g)| ⊕ ||B|_(2g)| ⊕ 2||E|_(u)|"
        )
        .unwrap()
    );

    // ~~~~~~~~~~~~~~~~~~~~~~
    // u D4h' (grey, unitary)
    // ~~~~~~~~~~~~~~~~~~~~~~

    let mut orbit_ru_u_grey_d4h_spatial_orbital_density_a = DensitySymmetryOrbit::builder()
        .group(&group_u_grey_d4h)
        .origin(&dena_ru)
        .integrality_threshold(1e-14)
        .linear_independence_threshold(1e-14)
        .symmetry_transformation_kind(SymmetryTransformationKind::Spatial)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
        .build()
        .unwrap();
    let _ = orbit_ru_u_grey_d4h_spatial_orbital_density_a
        .calc_smat(Some(sao_ru), None, true)
        .unwrap()
        .calc_xmat(false);
    assert_eq!(
        orbit_ru_u_grey_d4h_spatial_orbital_density_a
            .analyse_rep()
            .unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new(
            "|^(+)|A|_(1g)| ⊕ |^(+)|B|_(2g)| ⊕ |^(+)|E|_(u)|"
        )
        .unwrap()
    );

    let mut orbit_ru_u_grey_d4h_spatial_orbital_density_b = DensitySymmetryOrbit::builder()
        .group(&group_u_grey_d4h)
        .origin(&denb_ru)
        .integrality_threshold(1e-14)
        .linear_independence_threshold(1e-14)
        .symmetry_transformation_kind(SymmetryTransformationKind::Spatial)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
        .build()
        .unwrap();
    let _ = orbit_ru_u_grey_d4h_spatial_orbital_density_b
        .calc_smat(Some(sao_ru), None, true)
        .unwrap()
        .calc_xmat(false);
    assert_eq!(
        orbit_ru_u_grey_d4h_spatial_orbital_density_b
            .analyse_rep()
            .unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new(
            "|^(+)|A|_(1g)| ⊕ |^(+)|A|_(2g)| ⊕ |^(+)|B|_(1g)| ⊕ |^(+)|B|_(2g)| ⊕ 2|^(+)|E|_(u)|"
        )
        .unwrap()
    );

    let mut orbit_ru_u_grey_d4h_spatial_orbital_density_total = DensitySymmetryOrbit::builder()
        .group(&group_u_grey_d4h)
        .origin(&dentot_ru)
        .integrality_threshold(1e-14)
        .linear_independence_threshold(1e-14)
        .symmetry_transformation_kind(SymmetryTransformationKind::Spatial)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
        .build()
        .unwrap();
    let _ = orbit_ru_u_grey_d4h_spatial_orbital_density_total
        .calc_smat(Some(sao_ru), None, true)
        .unwrap()
        .calc_xmat(false);
    assert_eq!(
        orbit_ru_u_grey_d4h_spatial_orbital_density_total
            .analyse_rep()
            .unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new(
            "|^(+)|A|_(1g)| ⊕ |^(+)|A|_(2g)| ⊕ |^(+)|B|_(1g)| ⊕ |^(+)|B|_(2g)| ⊕ 2|^(+)|E|_(u)|"
        )
        .unwrap()
    );

    let mut orbit_ru_u_grey_d4h_spatial_orbital_density_spin = DensitySymmetryOrbit::builder()
        .group(&group_u_grey_d4h)
        .origin(&denspin_ru)
        .integrality_threshold(1e-14)
        .linear_independence_threshold(1e-14)
        .symmetry_transformation_kind(SymmetryTransformationKind::Spatial)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
        .build()
        .unwrap();
    let _ = orbit_ru_u_grey_d4h_spatial_orbital_density_spin
        .calc_smat(Some(sao_ru), None, true)
        .unwrap()
        .calc_xmat(false);
    assert_eq!(
        orbit_ru_u_grey_d4h_spatial_orbital_density_spin
            .analyse_rep()
            .unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new(
            "|^(+)|A|_(1g)| ⊕ |^(+)|A|_(2g)| ⊕ |^(+)|B|_(1g)| ⊕ |^(+)|B|_(2g)| ⊕ 2|^(+)|E|_(u)|"
        )
        .unwrap()
    );

    // ~~~~~~~~~~~~~~~~~~~~~~~
    // m D4h' (grey, magnetic)
    // ~~~~~~~~~~~~~~~~~~~~~~~

    let mut orbit_ru_m_grey_d4h_spatial_orbital_density_a = DensitySymmetryOrbit::builder()
        .group(&group_m_grey_d4h)
        .origin(&dena_ru)
        .integrality_threshold(1e-14)
        .linear_independence_threshold(1e-14)
        .symmetry_transformation_kind(SymmetryTransformationKind::Spatial)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
        .build()
        .unwrap();
    let _ = orbit_ru_m_grey_d4h_spatial_orbital_density_a
        .calc_smat(Some(sao_ru), None, true)
        .unwrap()
        .calc_xmat(false);
    assert_eq!(
        orbit_ru_m_grey_d4h_spatial_orbital_density_a
            .analyse_rep()
            .unwrap(),
        DecomposedSymbol::<MullikenIrcorepSymbol>::new("||A|_(1g)| ⊕ ||B|_(2g)| ⊕ ||E|_(u)|")
            .unwrap()
    );

    let mut orbit_ru_m_grey_d4h_spatial_orbital_density_b = DensitySymmetryOrbit::builder()
        .group(&group_m_grey_d4h)
        .origin(&denb_ru)
        .integrality_threshold(1e-14)
        .linear_independence_threshold(1e-14)
        .symmetry_transformation_kind(SymmetryTransformationKind::Spatial)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
        .build()
        .unwrap();
    let _ = orbit_ru_m_grey_d4h_spatial_orbital_density_b
        .calc_smat(Some(sao_ru), None, true)
        .unwrap()
        .calc_xmat(false);
    assert_eq!(
        orbit_ru_m_grey_d4h_spatial_orbital_density_b
            .analyse_rep()
            .unwrap(),
        DecomposedSymbol::<MullikenIrcorepSymbol>::new(
            "||A|_(1g)| ⊕ ||A|_(2g)| ⊕ ||B|_(1g)| ⊕ ||B|_(2g)| ⊕ 2||E|_(u)|"
        )
        .unwrap()
    );

    let mut orbit_ru_m_grey_d4h_spatial_orbital_density_total = DensitySymmetryOrbit::builder()
        .group(&group_m_grey_d4h)
        .origin(&dentot_ru)
        .integrality_threshold(1e-14)
        .linear_independence_threshold(1e-14)
        .symmetry_transformation_kind(SymmetryTransformationKind::Spatial)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
        .build()
        .unwrap();
    let _ = orbit_ru_m_grey_d4h_spatial_orbital_density_total
        .calc_smat(Some(sao_ru), None, true)
        .unwrap()
        .calc_xmat(false);
    assert_eq!(
        orbit_ru_m_grey_d4h_spatial_orbital_density_total
            .analyse_rep()
            .unwrap(),
        DecomposedSymbol::<MullikenIrcorepSymbol>::new(
            "||A|_(1g)| ⊕ ||A|_(2g)| ⊕ ||B|_(1g)| ⊕ ||B|_(2g)| ⊕ 2||E|_(u)|"
        )
        .unwrap()
    );

    let mut orbit_ru_m_grey_d4h_spatial_orbital_density_spin = DensitySymmetryOrbit::builder()
        .group(&group_m_grey_d4h)
        .origin(&denspin_ru)
        .integrality_threshold(1e-14)
        .linear_independence_threshold(1e-14)
        .symmetry_transformation_kind(SymmetryTransformationKind::Spatial)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
        .build()
        .unwrap();
    let _ = orbit_ru_m_grey_d4h_spatial_orbital_density_spin
        .calc_smat(Some(sao_ru), None, true)
        .unwrap()
        .calc_xmat(false);
    assert_eq!(
        orbit_ru_m_grey_d4h_spatial_orbital_density_spin
            .analyse_rep()
            .unwrap(),
        DecomposedSymbol::<MullikenIrcorepSymbol>::new(
            "||A|_(1g)| ⊕ ||A|_(2g)| ⊕ ||B|_(1g)| ⊕ ||B|_(2g)| ⊕ 2||E|_(u)|"
        )
        .unwrap()
    );

    // ~~~~~~~~~~~~~~~~~~~~~~~~~
    // u C4h (ordinary, unitary)
    // ~~~~~~~~~~~~~~~~~~~~~~~~~

    let mut orbit_cu_u_c4h_spatial_orbital_density_a = DensitySymmetryOrbit::builder()
        .group(&group_u_c4h)
        .origin(&dena_cu)
        .integrality_threshold(1e-14)
        .linear_independence_threshold(1e-14)
        .symmetry_transformation_kind(SymmetryTransformationKind::Spatial)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
        .build()
        .unwrap();
    let _ = orbit_cu_u_c4h_spatial_orbital_density_a
        .calc_smat(Some(sao_cu_bz), None, true)
        .unwrap()
        .calc_xmat(false);
    assert_eq!(
        orbit_cu_u_c4h_spatial_orbital_density_a
            .analyse_rep()
            .unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new(
            "||A|_(g)| ⊕ ||B|_(g)| ⊕ |_(a)|Γ|_(u)| ⊕ |_(b)|Γ|_(u)|"
        )
        .unwrap()
    );

    let mut orbit_cu_u_c4h_spatial_orbital_density_b = DensitySymmetryOrbit::builder()
        .group(&group_u_c4h)
        .origin(&denb_cu)
        .integrality_threshold(1e-14)
        .linear_independence_threshold(1e-14)
        .symmetry_transformation_kind(SymmetryTransformationKind::Spatial)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
        .build()
        .unwrap();
    let _ = orbit_cu_u_c4h_spatial_orbital_density_b
        .calc_smat(Some(sao_cu_bz), None, true)
        .unwrap()
        .calc_xmat(false);
    assert_eq!(
        orbit_cu_u_c4h_spatial_orbital_density_b
            .analyse_rep()
            .unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new(
            "||A|_(g)| ⊕ ||B|_(g)| ⊕ |_(a)|Γ|_(u)| ⊕ |_(b)|Γ|_(u)|"
        )
        .unwrap()
    );

    let mut orbit_cu_u_c4h_spatial_orbital_density_total = DensitySymmetryOrbit::builder()
        .group(&group_u_c4h)
        .origin(&dentot_cu)
        .integrality_threshold(1e-14)
        .linear_independence_threshold(1e-14)
        .symmetry_transformation_kind(SymmetryTransformationKind::Spatial)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
        .build()
        .unwrap();
    let _ = orbit_cu_u_c4h_spatial_orbital_density_total
        .calc_smat(Some(sao_cu_bz), None, true)
        .unwrap()
        .calc_xmat(false);
    assert_eq!(
        orbit_cu_u_c4h_spatial_orbital_density_total
            .analyse_rep()
            .unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new(
            "||A|_(g)| ⊕ ||B|_(g)| ⊕ |_(a)|Γ|_(u)| ⊕ |_(b)|Γ|_(u)|"
        )
        .unwrap()
    );

    let mut orbit_cu_u_c4h_spatial_orbital_density_spin = DensitySymmetryOrbit::builder()
        .group(&group_u_c4h)
        .origin(&denspin_cu)
        .integrality_threshold(1e-14)
        .linear_independence_threshold(1e-14)
        .symmetry_transformation_kind(SymmetryTransformationKind::Spatial)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
        .build()
        .unwrap();
    let _ = orbit_cu_u_c4h_spatial_orbital_density_spin
        .calc_smat(Some(sao_cu_bz), None, true)
        .unwrap()
        .calc_xmat(false);
    assert_eq!(
        orbit_cu_u_c4h_spatial_orbital_density_spin
            .analyse_rep()
            .unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new(
            "||A|_(g)| ⊕ ||B|_(g)| ⊕ |_(a)|Γ|_(u)| ⊕ |_(b)|Γ|_(u)|"
        )
        .unwrap()
    );

    // ~~~~~~~~~~~~~~~~~~~~~~~~
    // u D4h(C4h) (bw, unitary)
    // ~~~~~~~~~~~~~~~~~~~~~~~~

    let mut orbit_cu_u_bw_d4h_c4h_spatial_orbital_density_a = DensitySymmetryOrbit::builder()
        .group(&group_u_bw_d4h_c4h)
        .origin(&dena_cu)
        .integrality_threshold(1e-14)
        .linear_independence_threshold(1e-14)
        .symmetry_transformation_kind(SymmetryTransformationKind::Spatial)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
        .build()
        .unwrap();
    let _ = orbit_cu_u_bw_d4h_c4h_spatial_orbital_density_a
        .calc_smat(Some(sao_cu_bz), None, true)
        .unwrap()
        .calc_xmat(false);
    assert_eq!(
        orbit_cu_u_bw_d4h_c4h_spatial_orbital_density_a
            .analyse_rep()
            .unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new("||A|_(1g)| ⊕ ||B|_(2g)| ⊕ ||E|_(u)|")
            .unwrap()
    );

    let mut orbit_cu_u_bw_d4h_c4h_spatial_orbital_density_b = DensitySymmetryOrbit::builder()
        .group(&group_u_bw_d4h_c4h)
        .origin(&denb_cu)
        .integrality_threshold(1e-14)
        .linear_independence_threshold(1e-14)
        .symmetry_transformation_kind(SymmetryTransformationKind::Spatial)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
        .build()
        .unwrap();
    let _ = orbit_cu_u_bw_d4h_c4h_spatial_orbital_density_b
        .calc_smat(Some(sao_cu_bz), None, true)
        .unwrap()
        .calc_xmat(false);
    assert_eq!(
        orbit_cu_u_bw_d4h_c4h_spatial_orbital_density_b
            .analyse_rep()
            .unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new(
            "||A|_(1g)| ⊕ ||A|_(2g)| ⊕ ||B|_(1g)| ⊕ ||B|_(2g)| ⊕ 2||E|_(u)|"
        )
        .unwrap()
    );

    let mut orbit_cu_u_bw_d4h_c4h_spatial_orbital_density_total = DensitySymmetryOrbit::builder()
        .group(&group_u_bw_d4h_c4h)
        .origin(&dentot_cu)
        .integrality_threshold(1e-14)
        .linear_independence_threshold(1e-14)
        .symmetry_transformation_kind(SymmetryTransformationKind::Spatial)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
        .build()
        .unwrap();
    let _ = orbit_cu_u_bw_d4h_c4h_spatial_orbital_density_total
        .calc_smat(Some(sao_cu_bz), None, true)
        .unwrap()
        .calc_xmat(false);
    assert_eq!(
        orbit_cu_u_bw_d4h_c4h_spatial_orbital_density_total
            .analyse_rep()
            .unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new(
            "||A|_(1g)| ⊕ ||A|_(2g)| ⊕ ||B|_(1g)| ⊕ ||B|_(2g)| ⊕ 2||E|_(u)|"
        )
        .unwrap()
    );

    let mut orbit_cu_u_bw_d4h_c4h_spatial_orbital_density_spin = DensitySymmetryOrbit::builder()
        .group(&group_u_bw_d4h_c4h)
        .origin(&denspin_cu)
        .integrality_threshold(1e-14)
        .linear_independence_threshold(1e-14)
        .symmetry_transformation_kind(SymmetryTransformationKind::Spatial)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
        .build()
        .unwrap();
    let _ = orbit_cu_u_bw_d4h_c4h_spatial_orbital_density_spin
        .calc_smat(Some(sao_cu_bz), None, true)
        .unwrap()
        .calc_xmat(false);
    assert_eq!(
        orbit_cu_u_bw_d4h_c4h_spatial_orbital_density_spin
            .analyse_rep()
            .unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new(
            "||A|_(1g)| ⊕ ||A|_(2g)| ⊕ ||B|_(1g)| ⊕ ||B|_(2g)| ⊕ 2||E|_(u)|"
        )
        .unwrap()
    );

    // ~~~~~~~~~~~~~~~~~~~~~~~~~
    // m D4h(C4h) (bw, magnetic)
    // ~~~~~~~~~~~~~~~~~~~~~~~~~

    let mut orbit_cu_m_bw_d4h_c4h_spatial_orbital_density_a = DensitySymmetryOrbit::builder()
        .group(&group_m_bw_d4h_c4h)
        .origin(&dena_cu)
        .integrality_threshold(1e-14)
        .linear_independence_threshold(1e-14)
        .symmetry_transformation_kind(SymmetryTransformationKind::Spatial)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
        .build()
        .unwrap();
    let _ = orbit_cu_m_bw_d4h_c4h_spatial_orbital_density_a
        .calc_smat(Some(sao_cu_bz), None, true)
        .unwrap()
        .calc_xmat(false);
    assert_eq!(
        orbit_cu_m_bw_d4h_c4h_spatial_orbital_density_a
            .analyse_rep()
            .unwrap(),
        DecomposedSymbol::<MullikenIrcorepSymbol>::new(
            "||A|_(g)| ⊕ ||B|_(g)| ⊕ |_(a)|Γ|_(u)| ⊕ |_(b)|Γ|_(u)|"
        )
        .unwrap()
    );

    let mut orbit_cu_m_bw_d4h_c4h_spatial_orbital_density_b = DensitySymmetryOrbit::builder()
        .group(&group_m_bw_d4h_c4h)
        .origin(&denb_cu)
        .integrality_threshold(1e-14)
        .linear_independence_threshold(1e-14)
        .symmetry_transformation_kind(SymmetryTransformationKind::Spatial)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
        .build()
        .unwrap();
    let _ = orbit_cu_m_bw_d4h_c4h_spatial_orbital_density_b
        .calc_smat(Some(sao_cu_bz), None, true)
        .unwrap()
        .calc_xmat(false);
    assert_eq!(
        orbit_cu_m_bw_d4h_c4h_spatial_orbital_density_b
            .analyse_rep()
            .unwrap(),
        DecomposedSymbol::<MullikenIrcorepSymbol>::new(
            "2||A|_(g)| ⊕ 2||B|_(g)| ⊕ 2|_(a)|Γ|_(u)| ⊕ 2|_(b)|Γ|_(u)|"
        )
        .unwrap()
    );

    let mut orbit_cu_m_bw_d4h_c4h_spatial_orbital_density_total = DensitySymmetryOrbit::builder()
        .group(&group_m_bw_d4h_c4h)
        .origin(&dentot_cu)
        .integrality_threshold(1e-14)
        .linear_independence_threshold(1e-14)
        .symmetry_transformation_kind(SymmetryTransformationKind::Spatial)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
        .build()
        .unwrap();
    let _ = orbit_cu_m_bw_d4h_c4h_spatial_orbital_density_total
        .calc_smat(Some(sao_cu_bz), None, true)
        .unwrap()
        .calc_xmat(false);
    assert_eq!(
        orbit_cu_m_bw_d4h_c4h_spatial_orbital_density_total
            .analyse_rep()
            .unwrap(),
        DecomposedSymbol::<MullikenIrcorepSymbol>::new(
            "2||A|_(g)| ⊕ 2||B|_(g)| ⊕ 2|_(a)|Γ|_(u)| ⊕ 2|_(b)|Γ|_(u)|"
        )
        .unwrap()
    );

    let mut orbit_cu_m_bw_d4h_c4h_spatial_orbital_density_spin = DensitySymmetryOrbit::builder()
        .group(&group_m_bw_d4h_c4h)
        .origin(&denspin_cu)
        .integrality_threshold(1e-14)
        .linear_independence_threshold(1e-14)
        .symmetry_transformation_kind(SymmetryTransformationKind::Spatial)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
        .build()
        .unwrap();
    let _ = orbit_cu_m_bw_d4h_c4h_spatial_orbital_density_spin
        .calc_smat(Some(sao_cu_bz), None, true)
        .unwrap()
        .calc_xmat(false);
    assert_eq!(
        orbit_cu_m_bw_d4h_c4h_spatial_orbital_density_spin
            .analyse_rep()
            .unwrap(),
        DecomposedSymbol::<MullikenIrcorepSymbol>::new(
            "2||A|_(g)| ⊕ 2||B|_(g)| ⊕ 2|_(a)|Γ|_(u)| ⊕ 2|_(b)|Γ|_(u)|"
        )
        .unwrap()
    );
}
