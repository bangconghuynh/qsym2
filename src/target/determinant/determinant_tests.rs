use itertools::Itertools;
use nalgebra::{Point3, Vector3};
use ndarray::{Array2, Axis, array, concatenate, s};
use ndarray_linalg::assert::close_l2;
use num_complex::Complex;
use num_traits::Pow;

use crate::analysis::{EigenvalueComparisonMode, Overlap, RepAnalysis};
use crate::angmom::spinor_rotation_3d::{SpinConstraint, SpinOrbitCoupled};
use crate::auxiliary::atom::{Atom, ElementMap};
use crate::auxiliary::geometry::Transform;
use crate::auxiliary::molecule::Molecule;
use crate::basis::ao::{
    BasisAngularOrder, BasisAtom, BasisShell, CartOrder, PureOrder, ShellOrder, SpinorBalanceSymmetry, SpinorOrder
};
use crate::chartab::chartab_symbols::DecomposedSymbol;
use crate::group::{GroupProperties, MagneticRepresentedGroup, UnitaryRepresentedGroup};
use crate::symmetry::symmetry_core::{PreSymmetry, Symmetry};
use crate::symmetry::symmetry_element::symmetry_operation::SymmetryOperation;
use crate::symmetry::symmetry_element::{
    ROT, RotationGroup, SpecialSymmetryTransformation, SymmetryElement, TRROT,
};
use crate::symmetry::symmetry_element_order::ElementOrder;
use crate::symmetry::symmetry_group::SymmetryGroupProperties;
use crate::symmetry::symmetry_symbols::{MullikenIrcorepSymbol, MullikenIrrepSymbol};
use crate::symmetry::symmetry_transformation::{
    SymmetryTransformable, SymmetryTransformationKind, TimeReversalTransformable,
};
use crate::target::determinant::SlaterDeterminant;
use crate::target::determinant::determinant_analysis::SlaterDeterminantSymmetryOrbit;

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

    let det = SlaterDeterminant::<f64, SpinConstraint>::builder()
        .coefficients(&[calpha])
        .occupations(&[oalpha.clone()])
        .baos(vec![&bao_bf4])
        .mol(&mol_bf4)
        .structure_constraint(SpinConstraint::Restricted(2))
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
    let tdet_c4p1_ref = SlaterDeterminant::<f64, SpinConstraint>::builder()
        .coefficients(&[tcalpha_ref])
        .occupations(&[oalpha])
        .baos(vec![&bao_bf4])
        .mol(&mol_bf4)
        .structure_constraint(SpinConstraint::Restricted(2))
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
    let detunres = SlaterDeterminant::<f64, SpinConstraint>::builder()
        .coefficients(&[calpha, cbeta])
        .occupations(&[oalpha.clone(), obeta.clone()])
        .baos(vec![&bao_s4])
        .mol(&mol_s4)
        .structure_constraint(SpinConstraint::Unrestricted(2, false))
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
    let tdetunres_c4p1_ref = SlaterDeterminant::<f64, SpinConstraint>::builder()
        .coefficients(&[tcalpha_ref, tcbeta_ref])
        .occupations(&[oalpha, obeta])
        .baos(vec![&bao_s4])
        .mol(&mol_s4)
        .structure_constraint(SpinConstraint::Unrestricted(2, false))
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
    let detgen = SlaterDeterminant::<f64, SpinConstraint>::builder()
        .coefficients(&[cgen.clone()])
        .occupations(&[ogen.clone()])
        .baos(vec![&bao_s4, &bao_s4])
        .mol(&mol_s4)
        .structure_constraint(SpinConstraint::Generalised(2, false))
        .complex_symmetric(false)
        .threshold(1e-14)
        .build()
        .unwrap();

    let tcalpha2_gen = concatenate!(Axis(0), cbeta2, Array2::zeros((20, 1)));
    let tcbeta2_gen = concatenate!(Axis(0), Array2::zeros((20, 1)), calpha2);
    let tcgen_ref = concatenate![Axis(1), tcalpha2_gen, tcbeta2_gen];
    let tdetgen_c4p1_ref = SlaterDeterminant::<f64, SpinConstraint>::builder()
        .coefficients(&[tcgen_ref])
        .occupations(&[ogen.clone()])
        .baos(vec![&bao_s4, &bao_s4])
        .mol(&mol_s4)
        .structure_constraint(SpinConstraint::Generalised(2, false))
        .complex_symmetric(false)
        .threshold(1e-14)
        .build()
        .unwrap();
    let tdetgen_c4p1 = detgen.sym_transform_spatial(&c4p1).unwrap();
    assert_eq!(tdetgen_c4p1, tdetgen_c4p1_ref);

    // S1(+0.000, +0.000, +1.000)
    let s1zp1 = group.get_index(11).unwrap();
    let tdetgen_s1zp1_ref = SlaterDeterminant::<f64, SpinConstraint>::builder()
        .coefficients(&[-cgen.clone()])
        .occupations(&[ogen.clone()])
        .baos(vec![&bao_s4, &bao_s4])
        .mol(&mol_s4)
        .structure_constraint(SpinConstraint::Generalised(2, false))
        .complex_symmetric(false)
        .threshold(1e-14)
        .build()
        .unwrap();
    let tdetgen_s1zp1 = detgen.sym_transform_spatial(&s1zp1).unwrap();
    assert_eq!(tdetgen_s1zp1, tdetgen_s1zp1_ref);

    // S1(+0.000, +1.000, +0.000)
    let s1yp1 = group.get_index(12).unwrap();
    let tdetgen_s1yp1_ref = SlaterDeterminant::<f64, SpinConstraint>::builder()
        .coefficients(&[cgen.clone()])
        .occupations(&[ogen.clone()])
        .baos(vec![&bao_s4, &bao_s4])
        .mol(&mol_s4)
        .structure_constraint(SpinConstraint::Generalised(2, false))
        .complex_symmetric(false)
        .threshold(1e-14)
        .build()
        .unwrap();
    let tdetgen_s1yp1 = detgen.sym_transform_spatial(&s1yp1).unwrap();
    assert_eq!(tdetgen_s1yp1, tdetgen_s1yp1_ref);

    // S1(+1.000, +0.000, +0.000)
    let s1xp1 = group.get_index(13).unwrap();
    let tdetgen_s1xp1_ref = SlaterDeterminant::<f64, SpinConstraint>::builder()
        .coefficients(&[cgen.clone()])
        .occupations(&[ogen.clone()])
        .baos(vec![&bao_s4, &bao_s4])
        .mol(&mol_s4)
        .structure_constraint(SpinConstraint::Generalised(2, false))
        .complex_symmetric(false)
        .threshold(1e-14)
        .build()
        .unwrap();
    let tdetgen_s1xp1 = detgen.sym_transform_spatial(&s1xp1).unwrap();
    assert_eq!(tdetgen_s1xp1, tdetgen_s1xp1_ref);

    // i
    let ip1 = group.get_index(8).unwrap();
    let tdetgen_ip1_ref = SlaterDeterminant::<f64, SpinConstraint>::builder()
        .coefficients(&[-cgen])
        .occupations(&[ogen.clone()])
        .baos(vec![&bao_s4, &bao_s4])
        .mol(&mol_s4)
        .structure_constraint(SpinConstraint::Generalised(2, false))
        .complex_symmetric(false)
        .threshold(1e-14)
        .build()
        .unwrap();
    let tdetgen_ip1 = detgen.sym_transform_spatial(&ip1).unwrap();
    assert_eq!(tdetgen_ip1, tdetgen_ip1_ref);

    // S4(+0.000, +0.000, +1.000)
    let s4p1 = group.get_index(9).unwrap();
    let tcgen_s4p1_ref = concatenate![Axis(1), -tcalpha2_gen, -tcbeta2_gen];
    let tdetgen_s4p1_ref = SlaterDeterminant::<f64, SpinConstraint>::builder()
        .coefficients(&[tcgen_s4p1_ref])
        .occupations(&[ogen])
        .baos(vec![&bao_s4, &bao_s4])
        .mol(&mol_s4)
        .structure_constraint(SpinConstraint::Generalised(2, false))
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

    let detunres = SlaterDeterminant::<f64, SpinConstraint>::builder()
        .coefficients(&[calpha.clone(), cbeta.clone()])
        .occupations(&[oalpha.clone(), obeta.clone()])
        .baos(vec![&bao_b3])
        .mol(&mol_b3)
        .structure_constraint(SpinConstraint::Unrestricted(2, false))
        .complex_symmetric(false)
        .threshold(1e-14)
        .build()
        .unwrap();

    let tdetunres_tr = detunres.transform_timerev().unwrap();
    let tdetunres_tr_ref = SlaterDeterminant::<f64, SpinConstraint>::builder()
        .coefficients(&[-cbeta.clone(), calpha.clone()])
        .occupations(&[obeta.clone(), oalpha.clone()])
        .baos(vec![&bao_b3])
        .mol(&mol_b3)
        .structure_constraint(SpinConstraint::Unrestricted(2, false))
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
    let tdetunres_c3p1_tr_ref = SlaterDeterminant::<f64, SpinConstraint>::builder()
        .coefficients(&[tcalpha_ref.clone(), tcbeta_ref.clone()])
        .occupations(&[obeta, oalpha])
        .baos(vec![&bao_b3])
        .mol(&mol_b3)
        .structure_constraint(SpinConstraint::Unrestricted(2, false))
        .complex_symmetric(false)
        .threshold(1e-14)
        .build()
        .unwrap();
    assert_eq!(tdetunres_c3p1_tr, tdetunres_c3p1_tr_ref);

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

    let tdetgen_tr = detgen.transform_timerev().unwrap();
    let tcgen_ref = concatenate![Axis(0), -cbeta, calpha];
    let tdetgen_tr_ref = SlaterDeterminant::<f64, SpinConstraint>::builder()
        .coefficients(&[tcgen_ref])
        .occupations(&[ogen.clone()])
        .baos(vec![&bao_b3, &bao_b3])
        .mol(&mol_b3)
        .structure_constraint(SpinConstraint::Generalised(2, false))
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
    let tdetgen_c3p1_tr_ref = SlaterDeterminant::<f64, SpinConstraint>::builder()
        .coefficients(&[tcgen_ref])
        .occupations(&[ogen])
        .baos(vec![&bao_b3, &bao_b3])
        .mol(&mol_b3)
        .structure_constraint(SpinConstraint::Generalised(2, false))
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

    let tcalpha_gen_ref = concatenate!(Axis(0), Array2::zeros((6, 2)), calpha.map(|x| x.conj()));
    let tcbeta_gen_ref = concatenate!(Axis(0), -cbeta.map(|x| x.conj()), Array2::zeros((6, 2)));
    let tcgen_ref = concatenate![Axis(1), tcalpha_gen_ref, tcbeta_gen_ref];
    let tdetgen_tr_ref = SlaterDeterminant::<C128, SpinConstraint>::builder()
        .coefficients(&[tcgen_ref])
        .occupations(&[ogen])
        .baos(vec![&bao_c2, &bao_c2])
        .mol(&mol_c2)
        .structure_constraint(SpinConstraint::Generalised(2, false))
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
    let detgen: SlaterDeterminant<C128, SpinConstraint> =
        SlaterDeterminant::<f64, SpinConstraint>::builder()
            .coefficients(&[cgen.clone()])
            .occupations(&[ogen.clone()])
            .baos(vec![&bao_c3, &bao_c3])
            .mol(&mol_c3)
            .structure_constraint(SpinConstraint::Generalised(2, false))
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
    let tdetgen_c2_nsr_p1_ref = SlaterDeterminant::<C128, SpinConstraint>::builder()
        .coefficients(&[tcgen_ref.clone()])
        .occupations(&[ogen.clone()])
        .baos(vec![&bao_c3, &bao_c3])
        .mol(&mol_c3)
        .structure_constraint(SpinConstraint::Generalised(2, false))
        .complex_symmetric(false)
        .threshold(1e-14)
        .build()
        .unwrap();
    let tdetgen_c2_nsr_p1 = detgen.sym_transform_spin(&c2_nsr_p1).unwrap();
    assert_eq!(tdetgen_c2_nsr_p1, tdetgen_c2_nsr_p1_ref);

    let c2_nsr_p2 = (&c2_nsr_p1).pow(2);
    let tdetgen_c2_nsr_p2_ref: SlaterDeterminant<C128, SpinConstraint> =
        SlaterDeterminant::<f64, SpinConstraint>::builder()
            .coefficients(&[-cgen])
            .occupations(&[ogen.clone()])
            .baos(vec![&bao_c3, &bao_c3])
            .mol(&mol_c3)
            .structure_constraint(SpinConstraint::Generalised(2, false))
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
    let tdetgen_c2_nsr_p3_ref = SlaterDeterminant::<C128, SpinConstraint>::builder()
        .coefficients(&[-tcgen_ref])
        .occupations(&[ogen.clone()])
        .baos(vec![&bao_c3, &bao_c3])
        .mol(&mol_c3)
        .structure_constraint(SpinConstraint::Generalised(2, false))
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
    let tdetgen_sxy_nsr_ref = SlaterDeterminant::<C128, SpinConstraint>::builder()
        .coefficients(&[sxy_tcgen_ref.clone()])
        .occupations(&[ogen.clone()])
        .baos(vec![&bao_c3, &bao_c3])
        .mol(&mol_c3)
        .structure_constraint(SpinConstraint::Generalised(2, false))
        .complex_symmetric(false)
        .threshold(1e-14)
        .build()
        .unwrap();
    let tdetgen_sxy_nsr = detgen.sym_transform_spin(&sxy_nsr_p1).unwrap();
    assert_eq!(tdetgen_sxy_nsr, tdetgen_sxy_nsr_ref);

    let tdetgen_sxy_nsr_p2 = detgen.sym_transform_spin(&(&sxy_nsr_p1).pow(2)).unwrap();
    assert_eq!(tdetgen_sxy_nsr_p2, tdetgen_e_isr);

    let tdetgen_sxy_nsr_p3_ref = SlaterDeterminant::<C128, SpinConstraint>::builder()
        .coefficients(&[-sxy_tcgen_ref])
        .occupations(&[ogen.clone()])
        .baos(vec![&bao_c3, &bao_c3])
        .mol(&mol_c3)
        .structure_constraint(SpinConstraint::Generalised(2, false))
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
    let tdetgen_sxyz_nsr_p1_ref = SlaterDeterminant::<C128, SpinConstraint>::builder()
        .coefficients(&[tcgen_sxyz_ref])
        .occupations(&[ogen])
        .baos(vec![&bao_c3, &bao_c3])
        .mol(&mol_c3)
        .structure_constraint(SpinConstraint::Generalised(2, false))
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
    let detgen: SlaterDeterminant<C128, SpinConstraint> =
        SlaterDeterminant::<f64, SpinConstraint>::builder()
            .coefficients(&[cgen])
            .occupations(&[ogen])
            .baos(vec![&bao_h4, &bao_h4])
            .mol(&mol_h4)
            .structure_constraint(SpinConstraint::Generalised(2, false))
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
fn test_determinant_transformation_h_jadapted_twoj_1() {
    let emap = ElementMap::new();
    let atm_h0 = Atom::from_xyz("H 0.0 0.0 0.0", &emap, 1e-7).unwrap();

    let bs_sp1half = BasisShell::new(
        1,
        ShellOrder::Spinor(SpinorOrder::increasingm(1, true, None)),
    );

    let batm_h0 = BasisAtom::new(&atm_h0, &[bs_sp1half.clone()]);
    let bao_h = BasisAngularOrder::new(&[batm_h0]);
    let mol_h = Molecule::from_atoms(&[atm_h0.clone()], 1e-7);

    #[rustfmt::skip]
    let c = array![
        [Complex::new(1.0, 1.0)],
        [Complex::new(2.0, 2.0)],
    ];
    let occ = array![1.0];

    let det = SlaterDeterminant::<Complex<f64>, SpinOrbitCoupled>::builder()
        .coefficients(&[c])
        .occupations(&[occ.clone()])
        .baos(vec![&bao_h])
        .mol(&mol_h)
        .structure_constraint(SpinOrbitCoupled::JAdapted(1))
        .complex_symmetric(false)
        .threshold(1e-14)
        .build()
        .unwrap();

    // -
    // θ
    // -
    let t_element = SymmetryElement::builder()
        .threshold(1e-12)
        .proper_order(ElementOrder::Int(1))
        .proper_power(1)
        .raw_axis(Vector3::new(0.0, 0.0, 1.0))
        .kind(TRROT)
        .rotation_group(RotationGroup::SU2(true))
        .build()
        .unwrap();
    let t_su2 = SymmetryOperation::builder()
        .generating_element(t_element)
        .power(1)
        .build()
        .unwrap();

    let tdet_t = det.sym_transform_spin_spatial(&t_su2).unwrap();
    let tdet_t2 = det.transform_timerev().unwrap();
    assert_eq!(tdet_t, tdet_t2);
    #[rustfmt::skip]
    let tdet_t_c0_ref = array![
        [Complex::new(2.0, -2.0)],
        [Complex::new(-1.0, 1.0)],
    ];
    close_l2(&tdet_t.coefficients()[0], &tdet_t_c0_ref, 1e-14);

    // ---
    // C2y
    // ---
    let c2y_element = SymmetryElement::builder()
        .threshold(1e-12)
        .proper_order(ElementOrder::Int(2))
        .proper_power(1)
        .raw_axis(Vector3::new(0.0, 1.0, 0.0))
        .kind(ROT)
        .rotation_group(RotationGroup::SU2(true))
        .build()
        .unwrap();
    let c2y_su2 = SymmetryOperation::builder()
        .generating_element(c2y_element)
        .power(1)
        .build()
        .unwrap();

    let tdet_c2y = det.sym_transform_spin_spatial(&c2y_su2).unwrap();
    #[rustfmt::skip]
    let tdet_c2y_c0_ref = array![
        [Complex::new(2.0, 2.0)],
        [Complex::new(-1.0, -1.0)],
    ];
    close_l2(&tdet_c2y.coefficients()[0], &tdet_c2y_c0_ref, 1e-14);

    // -------
    // θ ⋅ C2y
    // -------
    let tdet_c2y_t = tdet_c2y.sym_transform_spin_spatial(&t_su2).unwrap();
    let tdet_tc2y = det
        .sym_transform_spin_spatial(&(&t_su2 * &c2y_su2))
        .unwrap();
    assert_eq!(tdet_c2y_t, tdet_tc2y);
    #[rustfmt::skip]
    let tdet_c2y_t_c0_ref = array![
        [Complex::new(-1.0, 1.0)],
        [Complex::new(-2.0, 2.0)],
    ];
    close_l2(&tdet_c2y_t.coefficients()[0], &tdet_c2y_t_c0_ref, 1e-14);

    // -------
    // C2y ⋅ θ
    // -------
    let tdet_t_c2y = tdet_t.sym_transform_spin_spatial(&c2y_su2).unwrap();
    let tdet_c2yt = det
        .sym_transform_spin_spatial(&(&c2y_su2 * &t_su2))
        .unwrap();
    assert_eq!(tdet_t_c2y, tdet_c2yt);
    #[rustfmt::skip]
    let tdet_t_c2y_c0_ref = array![
        [Complex::new(-1.0, 1.0)],
        [Complex::new(-2.0, 2.0)],
    ];
    close_l2(&tdet_t_c2y.coefficients()[0], &tdet_t_c2y_c0_ref, 1e-14);

    // ---
    // C3z
    // ---
    let c3z_element = SymmetryElement::builder()
        .threshold(1e-12)
        .proper_order(ElementOrder::Int(3))
        .proper_power(1)
        .raw_axis(Vector3::new(0.0, 0.0, 1.0))
        .kind(ROT)
        .rotation_group(RotationGroup::SU2(true))
        .build()
        .unwrap();
    let c3z_su2 = SymmetryOperation::builder()
        .generating_element(c3z_element)
        .power(1)
        .build()
        .unwrap();

    let tdet_c3z = det.sym_transform_spin_spatial(&c3z_su2).unwrap();
    #[rustfmt::skip]
    let pi3 = std::f64::consts::FRAC_PI_3;
    let tdet_c3z_c0_ref = array![
        [Complex::new(1.0, 1.0) * Complex::new(pi3.cos(), pi3.sin())],
        [Complex::new(2.0, 2.0) * Complex::new(pi3.cos(), -pi3.sin())],
    ];
    close_l2(&tdet_c3z.coefficients()[0], &tdet_c3z_c0_ref, 1e-14);

    // -------
    // θ ⋅ C3z
    // -------
    let tdet_c3z_t = tdet_c3z.sym_transform_spin_spatial(&t_su2).unwrap();
    let tdet_tc3z = det
        .sym_transform_spin_spatial(&(&t_su2 * &c3z_su2))
        .unwrap();
    assert_eq!(tdet_c3z_t, tdet_tc3z);
    let tdet_c3z_t_c0_ref = array![
        [Complex::new(2.0, -2.0) * Complex::new(pi3.cos(), pi3.sin())],
        [Complex::new(-1.0, 1.0) * Complex::new(pi3.cos(), -pi3.sin())],
    ];
    close_l2(&tdet_c3z_t.coefficients()[0], &tdet_c3z_t_c0_ref, 1e-14);

    // -------
    // C3z ⋅ θ
    // -------
    let tdet_t_c3z = tdet_t.sym_transform_spin_spatial(&c3z_su2).unwrap();
    let tdet_c3zt = det
        .sym_transform_spin_spatial(&(&c3z_su2 * &t_su2))
        .unwrap();
    assert_eq!(tdet_t_c3z, tdet_c3zt);
    let tdet_t_c3z_c0_ref = array![
        [Complex::new(2.0, -2.0) * Complex::new(pi3.cos(), pi3.sin())],
        [Complex::new(-1.0, 1.0) * Complex::new(pi3.cos(), -pi3.sin())],
    ];
    close_l2(&tdet_t_c3z.coefficients()[0], &tdet_t_c3z_c0_ref, 1e-14);

    // -------
    // C3(111)
    // -------
    let c3_111_element = SymmetryElement::builder()
        .threshold(1e-12)
        .proper_order(ElementOrder::Int(3))
        .proper_power(1)
        .raw_axis(Vector3::new(1.0, 1.0, 1.0))
        .kind(ROT)
        .rotation_group(RotationGroup::SU2(true))
        .build()
        .unwrap();
    let c3_111_su2 = SymmetryOperation::builder()
        .generating_element(c3_111_element)
        .power(1)
        .build()
        .unwrap();
    let tdet_c3_111 = det.sym_transform_spin_spatial(&c3_111_su2).unwrap();
    let tdet_c3_111_ref = array![[Complex::new(2.0, 1.0)], [Complex::new(2.0, -1.0)],];
    close_l2(&tdet_c3_111.coefficients()[0], &tdet_c3_111_ref, 1e-14);
}

#[test]
fn test_determinant_transformation_h_jadapted_twoj_2() {
    let emap = ElementMap::new();
    let atm_h0 = Atom::from_xyz("H 0.0 0.0 0.0", &emap, 1e-7).unwrap();

    let bs_p1 = BasisShell::new(1, ShellOrder::Pure(PureOrder::increasingm(1)));

    let batm_h0 = BasisAtom::new(&atm_h0, &[bs_p1.clone()]);
    let bao_h = BasisAngularOrder::new(&[batm_h0]);
    let mol_h = Molecule::from_atoms(&[atm_h0.clone()], 1e-7);

    #[rustfmt::skip]
    let c = array![
        [Complex::new(1.0, 1.0)],
        [Complex::new(2.0, 2.0)],
        [Complex::new(3.0, 3.0)],
    ];
    let occ = array![1.0];

    let det = SlaterDeterminant::<Complex<f64>, SpinOrbitCoupled>::builder()
        .coefficients(&[c])
        .occupations(&[occ.clone()])
        .baos(vec![&bao_h])
        .mol(&mol_h)
        .structure_constraint(SpinOrbitCoupled::JAdapted(1))
        .complex_symmetric(false)
        .threshold(1e-14)
        .build()
        .unwrap();

    // -
    // θ
    // -
    let t_element = SymmetryElement::builder()
        .threshold(1e-12)
        .proper_order(ElementOrder::Int(1))
        .proper_power(1)
        .raw_axis(Vector3::new(0.0, 0.0, 1.0))
        .kind(TRROT)
        .rotation_group(RotationGroup::SU2(true))
        .build()
        .unwrap();
    let t_su2 = SymmetryOperation::builder()
        .generating_element(t_element)
        .power(1)
        .build()
        .unwrap();

    let tdet_t = det.sym_transform_spin_spatial(&t_su2).unwrap();
    let tdet_t2 = det.transform_timerev().unwrap();
    assert_eq!(tdet_t, tdet_t2);
    #[rustfmt::skip]
    let tdet_t_c0_ref = array![
        [Complex::new(3.0, -3.0)],
        [Complex::new(-2.0, 2.0)],
        [Complex::new(1.0, -1.0)],
    ];
    close_l2(&tdet_t.coefficients()[0], &tdet_t_c0_ref, 1e-14);

    // ---
    // C2y
    // ---
    let c2y_element = SymmetryElement::builder()
        .threshold(1e-12)
        .proper_order(ElementOrder::Int(2))
        .proper_power(1)
        .raw_axis(Vector3::new(0.0, 1.0, 0.0))
        .kind(ROT)
        .rotation_group(RotationGroup::SU2(true))
        .build()
        .unwrap();
    let c2y_su2 = SymmetryOperation::builder()
        .generating_element(c2y_element)
        .power(1)
        .build()
        .unwrap();

    let tdet_c2y = det.sym_transform_spin_spatial(&c2y_su2).unwrap();
    #[rustfmt::skip]
    let tdet_c2y_c0_ref = array![
        [Complex::new(3.0, 3.0)],
        [Complex::new(-2.0, -2.0)],
        [Complex::new(1.0, 1.0)],
    ];
    close_l2(&tdet_c2y.coefficients()[0], &tdet_c2y_c0_ref, 1e-14);

    // -------
    // θ ⋅ C2y
    // -------
    let tdet_c2y_t = tdet_c2y.sym_transform_spin_spatial(&t_su2).unwrap();
    let tdet_tc2y = det
        .sym_transform_spin_spatial(&(&t_su2 * &c2y_su2))
        .unwrap();
    assert_eq!(tdet_c2y_t, tdet_tc2y);
    #[rustfmt::skip]
    let tdet_c2y_t_c0_ref = array![
        [Complex::new(1.0, -1.0)],
        [Complex::new(2.0, -2.0)],
        [Complex::new(3.0, -3.0)],
    ];
    close_l2(&tdet_c2y_t.coefficients()[0], &tdet_c2y_t_c0_ref, 1e-14);

    // -------
    // C2y ⋅ θ
    // -------
    let tdet_t_c2y = tdet_t.sym_transform_spin_spatial(&c2y_su2).unwrap();
    let tdet_c2yt = det
        .sym_transform_spin_spatial(&(&c2y_su2 * &t_su2))
        .unwrap();
    assert_eq!(tdet_t_c2y, tdet_c2yt);
    #[rustfmt::skip]
    let tdet_t_c2y_c0_ref = array![
        [Complex::new(1.0, -1.0)],
        [Complex::new(2.0, -2.0)],
        [Complex::new(3.0, -3.0)],
    ];
    close_l2(&tdet_t_c2y.coefficients()[0], &tdet_t_c2y_c0_ref, 1e-14);

    // ---
    // C3z
    // ---
    let c3z_element = SymmetryElement::builder()
        .threshold(1e-12)
        .proper_order(ElementOrder::Int(3))
        .proper_power(1)
        .raw_axis(Vector3::new(0.0, 0.0, 1.0))
        .kind(ROT)
        .rotation_group(RotationGroup::SU2(true))
        .build()
        .unwrap();
    let c3z_su2 = SymmetryOperation::builder()
        .generating_element(c3z_element)
        .power(1)
        .build()
        .unwrap();

    let tdet_c3z = det.sym_transform_spin_spatial(&c3z_su2).unwrap();
    #[rustfmt::skip]
    let pi3 = std::f64::consts::FRAC_PI_3;
    let tdet_c3z_c0_ref = array![
        [Complex::new(1.0, 1.0) * Complex::new(-pi3.cos(), pi3.sin())],
        [Complex::new(2.0, 2.0)],
        [Complex::new(3.0, 3.0) * Complex::new(-pi3.cos(), -pi3.sin())],
    ];
    close_l2(&tdet_c3z.coefficients()[0], &tdet_c3z_c0_ref, 1e-14);

    // -------
    // θ ⋅ C3z
    // -------
    let tdet_c3z_t = tdet_c3z.sym_transform_spin_spatial(&t_su2).unwrap();
    let tdet_tc3z = det
        .sym_transform_spin_spatial(&(&t_su2 * &c3z_su2))
        .unwrap();
    assert_eq!(tdet_c3z_t, tdet_tc3z);
    let tdet_c3z_t_c0_ref = array![
        [Complex::new(3.0, -3.0) * Complex::new(-pi3.cos(), pi3.sin())],
        [Complex::new(-2.0, 2.0)],
        [Complex::new(1.0, -1.0) * Complex::new(-pi3.cos(), -pi3.sin())],
    ];
    close_l2(&tdet_c3z_t.coefficients()[0], &tdet_c3z_t_c0_ref, 1e-14);

    // -------
    // C3z ⋅ θ
    // -------
    let tdet_t_c3z = tdet_t.sym_transform_spin_spatial(&c3z_su2).unwrap();
    let tdet_c3zt = det
        .sym_transform_spin_spatial(&(&c3z_su2 * &t_su2))
        .unwrap();
    assert_eq!(tdet_t_c3z, tdet_c3zt);
    let tdet_t_c3z_c0_ref = array![
        [Complex::new(3.0, -3.0) * Complex::new(-pi3.cos(), pi3.sin())],
        [Complex::new(-2.0, 2.0)],
        [Complex::new(1.0, -1.0) * Complex::new(-pi3.cos(), -pi3.sin())],
    ];
    close_l2(&tdet_t_c3z.coefficients()[0], &tdet_t_c3z_c0_ref, 1e-14);
}

#[test]
fn test_determinant_transformation_h_jadapted_twoj_3() {
    let emap = ElementMap::new();
    let atm_h0 = Atom::from_xyz("H 0.0 0.0 0.0", &emap, 1e-7).unwrap();

    let bs_sp3 = BasisShell::new(
        3,
        ShellOrder::Spinor(SpinorOrder::increasingm(3, true, None)),
    );

    let batm_h0 = BasisAtom::new(&atm_h0, &[bs_sp3.clone()]);
    let bao_h = BasisAngularOrder::new(&[batm_h0]);
    let mol_h = Molecule::from_atoms(&[atm_h0.clone()], 1e-7);

    #[rustfmt::skip]
    let c = array![
        [Complex::new(1.0, 1.0)],
        [Complex::new(2.0, 2.0)],
        [Complex::new(3.0, 3.0)],
        [Complex::new(4.0, 4.0)],
    ];
    let occ = array![1.0];

    let det = SlaterDeterminant::<Complex<f64>, SpinOrbitCoupled>::builder()
        .coefficients(&[c])
        .occupations(&[occ.clone()])
        .baos(vec![&bao_h])
        .mol(&mol_h)
        .structure_constraint(SpinOrbitCoupled::JAdapted(1))
        .complex_symmetric(false)
        .threshold(1e-14)
        .build()
        .unwrap();

    // -
    // θ
    // -
    let t_element = SymmetryElement::builder()
        .threshold(1e-12)
        .proper_order(ElementOrder::Int(1))
        .proper_power(1)
        .raw_axis(Vector3::new(0.0, 0.0, 1.0))
        .kind(TRROT)
        .rotation_group(RotationGroup::SU2(true))
        .build()
        .unwrap();
    let t_su2 = SymmetryOperation::builder()
        .generating_element(t_element)
        .power(1)
        .build()
        .unwrap();

    let tdet_t = det.sym_transform_spin_spatial(&t_su2).unwrap();
    let tdet_t2 = det.transform_timerev().unwrap();
    assert_eq!(tdet_t, tdet_t2);
    #[rustfmt::skip]
    let tdet_t_c0_ref = array![
        [Complex::new(4.0, -4.0)],
        [Complex::new(-3.0, 3.0)],
        [Complex::new(2.0, -2.0)],
        [Complex::new(-1.0, 1.0)],
    ];
    close_l2(&tdet_t.coefficients()[0], &tdet_t_c0_ref, 1e-14);

    // ---
    // C2y
    // ---
    let c2y_element = SymmetryElement::builder()
        .threshold(1e-12)
        .proper_order(ElementOrder::Int(2))
        .proper_power(1)
        .raw_axis(Vector3::new(0.0, 1.0, 0.0))
        .kind(ROT)
        .rotation_group(RotationGroup::SU2(true))
        .build()
        .unwrap();
    let c2y_su2 = SymmetryOperation::builder()
        .generating_element(c2y_element)
        .power(1)
        .build()
        .unwrap();

    let tdet_c2y = det.sym_transform_spin_spatial(&c2y_su2).unwrap();
    #[rustfmt::skip]
    let tdet_c2y_c0_ref = array![
        [Complex::new(4.0, 4.0)],
        [Complex::new(-3.0, -3.0)],
        [Complex::new(2.0, 2.0)],
        [Complex::new(-1.0, -1.0)],
    ];
    close_l2(&tdet_c2y.coefficients()[0], &tdet_c2y_c0_ref, 1e-14);

    // -------
    // θ ⋅ C2y
    // -------
    let tdet_c2y_t = tdet_c2y.sym_transform_spin_spatial(&t_su2).unwrap();
    let tdet_tc2y = det
        .sym_transform_spin_spatial(&(&t_su2 * &c2y_su2))
        .unwrap();
    assert_eq!(tdet_c2y_t, tdet_tc2y);
    #[rustfmt::skip]
    let tdet_c2y_t_c0_ref = array![
        [Complex::new(-1.0, 1.0)],
        [Complex::new(-2.0, 2.0)],
        [Complex::new(-3.0, 3.0)],
        [Complex::new(-4.0, 4.0)],
    ];
    close_l2(&tdet_c2y_t.coefficients()[0], &tdet_c2y_t_c0_ref, 1e-14);

    // -------
    // C2y ⋅ θ
    // -------
    let tdet_t_c2y = tdet_t.sym_transform_spin_spatial(&c2y_su2).unwrap();
    let tdet_c2yt = det
        .sym_transform_spin_spatial(&(&c2y_su2 * &t_su2))
        .unwrap();
    assert_eq!(tdet_t_c2y, tdet_c2yt);
    #[rustfmt::skip]
    let tdet_t_c2y_c0_ref = array![
        [Complex::new(-1.0, 1.0)],
        [Complex::new(-2.0, 2.0)],
        [Complex::new(-3.0, 3.0)],
        [Complex::new(-4.0, 4.0)],
    ];
    close_l2(&tdet_t_c2y.coefficients()[0], &tdet_t_c2y_c0_ref, 1e-14);

    // ---
    // C3z
    // ---
    let c3z_element = SymmetryElement::builder()
        .threshold(1e-12)
        .proper_order(ElementOrder::Int(3))
        .proper_power(1)
        .raw_axis(Vector3::new(0.0, 0.0, 1.0))
        .kind(ROT)
        .rotation_group(RotationGroup::SU2(true))
        .build()
        .unwrap();
    let c3z_su2 = SymmetryOperation::builder()
        .generating_element(c3z_element)
        .power(1)
        .build()
        .unwrap();

    let tdet_c3z = det.sym_transform_spin_spatial(&c3z_su2).unwrap();
    #[rustfmt::skip]
    let pi3 = std::f64::consts::FRAC_PI_3;
    let tdet_c3z_c0_ref = array![
        [-Complex::new(1.0, 1.0)],
        [Complex::new(2.0, 2.0) * Complex::new(pi3.cos(), pi3.sin())],
        [Complex::new(3.0, 3.0) * Complex::new(pi3.cos(), -pi3.sin())],
        [-Complex::new(4.0, 4.0)],
    ];
    close_l2(&tdet_c3z.coefficients()[0], &tdet_c3z_c0_ref, 1e-14);

    // -------
    // θ ⋅ C3z
    // -------
    let tdet_c3z_t = tdet_c3z.sym_transform_spin_spatial(&t_su2).unwrap();
    let tdet_tc3z = det
        .sym_transform_spin_spatial(&(&t_su2 * &c3z_su2))
        .unwrap();
    assert_eq!(tdet_c3z_t, tdet_tc3z);
    let tdet_c3z_t_c0_ref = array![
        [Complex::new(-4.0, 4.0)],
        [Complex::new(-3.0, 3.0) * Complex::new(pi3.cos(), pi3.sin())],
        [Complex::new(2.0, -2.0) * Complex::new(pi3.cos(), -pi3.sin())],
        [Complex::new(1.0, -1.0)],
    ];
    close_l2(&tdet_c3z_t.coefficients()[0], &tdet_c3z_t_c0_ref, 1e-14);

    // -------
    // C3z ⋅ θ
    // -------
    let tdet_t_c3z = tdet_t.sym_transform_spin_spatial(&c3z_su2).unwrap();
    let tdet_c3zt = det
        .sym_transform_spin_spatial(&(&c3z_su2 * &t_su2))
        .unwrap();
    assert_eq!(tdet_t_c3z, tdet_c3zt);
    let tdet_t_c3z_c0_ref = array![
        [Complex::new(-4.0, 4.0)],
        [Complex::new(-3.0, 3.0) * Complex::new(pi3.cos(), pi3.sin())],
        [Complex::new(2.0, -2.0) * Complex::new(pi3.cos(), -pi3.sin())],
        [Complex::new(1.0, -1.0)],
    ];
    close_l2(&tdet_t_c3z.coefficients()[0], &tdet_t_c3z_c0_ref, 1e-14);
}

#[test]
fn test_determinant_transformation_bf4_sqpl_jadapted() {
    // env_logger::init();
    let emap = ElementMap::new();
    let atm_b0 = Atom::from_xyz("B 0.0 0.0 0.0", &emap, 1e-7).unwrap();
    let atm_f0 = Atom::from_xyz("F 1.0 0.0 0.0", &emap, 1e-7).unwrap();
    let atm_f1 = Atom::from_xyz("F 0.0 1.0 0.0", &emap, 1e-7).unwrap();
    let atm_f2 = Atom::from_xyz("F -1.0 0.0 0.0", &emap, 1e-7).unwrap();
    let atm_f3 = Atom::from_xyz("F 0.0 -1.0 0.0", &emap, 1e-7).unwrap();

    let bs_sp1half = BasisShell::new(
        1,
        ShellOrder::Spinor(SpinorOrder::increasingm(1, true, None)),
    );

    let batm_b0 = BasisAtom::new(&atm_b0, &[bs_sp1half.clone()]);
    let batm_f0 = BasisAtom::new(&atm_f0, &[bs_sp1half.clone()]);
    let batm_f1 = BasisAtom::new(&atm_f1, &[bs_sp1half.clone()]);
    let batm_f2 = BasisAtom::new(&atm_f2, &[bs_sp1half.clone()]);
    let batm_f3 = BasisAtom::new(&atm_f3, &[bs_sp1half.clone()]);

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
    let c = array![
        [1.0], [0.0], // B0β
        [0.0], [1.0], // F0α
        [1.0], [0.0], // F1β
        [0.0], [1.0], // F2α
        [1.0], [0.0], // F3β
    ];
    let occ = array![1.0];

    let det: SlaterDeterminant<Complex<f64>, SpinOrbitCoupled> =
        SlaterDeterminant::<f64, SpinOrbitCoupled>::builder()
            .coefficients(&[c])
            .occupations(&[occ.clone()])
            .baos(vec![&bao_bf4])
            .mol(&mol_bf4)
            .structure_constraint(SpinOrbitCoupled::JAdapted(1))
            .complex_symmetric(false)
            .threshold(1e-14)
            .build()
            .unwrap()
            .into();

    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol_bf4)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, true).unwrap();
    let group = UnitaryRepresentedGroup::from_molecular_symmetry(&sym, None)
        .unwrap()
        .to_double_group()
        .unwrap();

    // Group elements:
    //  0 - E(Σ)
    //  1 - E(QΣ)
    //  2 - C4(Σ)(+0.000, +0.000, +1.000)
    //  3 - C4(QΣ)(+0.000, +0.000, +1.000)
    //  4 - C4(Σ)(+0.000, +0.000, -1.000)
    //  5 - C4(QΣ)(+0.000, +0.000, -1.000)
    //  6 - C2(Σ)(+0.000, +0.000, +1.000)
    //  7 - C2(QΣ)(+0.000, +0.000, +1.000)
    //  8 - C2(Σ)(+0.000, +1.000, +0.000)
    //  9 - C2(QΣ)(+0.000, +1.000, +0.000)
    // 10 - C2(Σ)(+1.000, +0.000, +0.000)
    // 11 - C2(QΣ)(+1.000, +0.000, +0.000)
    // 12 - C2(Σ)(+0.707, +0.707, +0.000)
    // 13 - C2(Σ)(+0.707, -0.707, +0.000)
    // 14 - C2(QΣ)(+0.707, +0.707, +0.000)
    // 15 - C2(QΣ)(+0.707, -0.707, +0.000)
    // 16 - i(Σ)
    // 17 - i(QΣ)
    // 18 - S4(Σ)(+0.000, +0.000, +1.000)
    // 19 - S4(QΣ)(+0.000, +0.000, +1.000)
    // 20 - S4(Σ)(+0.000, +0.000, -1.000)
    // 21 - S4(QΣ)(+0.000, +0.000, -1.000)
    // 22 - σh(Σ)(+0.000, +0.000, +1.000)
    // 23 - σh(QΣ)(+0.000, +0.000, +1.000)
    // 24 - σv(Σ)(+0.000, +1.000, +0.000)
    // 25 - σv(QΣ)(+0.000, +1.000, +0.000)
    // 26 - σv(Σ)(+1.000, +0.000, +0.000)
    // 27 - σv(QΣ)(+1.000, +0.000, +0.000)
    // 28 - σv(Σ)(+0.707, +0.707, +0.000)
    // 29 - σv(Σ)(+0.707, -0.707, +0.000)
    // 30 - σv(QΣ)(+0.707, +0.707, +0.000)
    // 31 - σv(QΣ)(+0.707, -0.707, +0.000)
    // 32 - θ(Σ)
    // 33 - θ(QΣ)
    // 34 - θ·C4(Σ)(+0.000, +0.000, +1.000)
    // 35 - θ·C4(QΣ)(+0.000, +0.000, +1.000)
    // 36 - [θ·C4(Σ)(+0.000, +0.000, +1.000)]^3
    // 37 - θ·C4(QΣ)(+0.000, +0.000, -1.000)
    // 38 - θ·C2(Σ)(+0.000, +0.000, +1.000)
    // 39 - θ·C2(QΣ)(+0.000, +0.000, +1.000)
    // 40 - θ·C2(Σ)(+0.000, +1.000, +0.000)
    // 41 - θ·C2(QΣ)(+0.000, +1.000, +0.000)
    // 42 - θ·C2(Σ)(+1.000, +0.000, +0.000)
    // 43 - θ·C2(QΣ)(+1.000, +0.000, +0.000)
    // 44 - θ·C2(Σ)(+0.707, +0.707, +0.000)
    // 45 - θ·C2(Σ)(+0.707, -0.707, +0.000)
    // 46 - θ·C2(QΣ)(+0.707, +0.707, +0.000)
    // 47 - θ·C2(QΣ)(+0.707, -0.707, +0.000)
    // 48 - θ·i(Σ)
    // 49 - θ·i(QΣ)
    // 50 - θ·S4(Σ)(+0.000, +0.000, +1.000)
    // 51 - θ·S4(QΣ)(+0.000, +0.000, +1.000)
    // 52 - [θ·S4(Σ)(+0.000, +0.000, +1.000)]^3
    // 53 - θ·S4(QΣ)(+0.000, +0.000, -1.000)
    // 54 - θ·σh(Σ)(+0.000, +0.000, +1.000)
    // 55 - θ·σh(QΣ)(+0.000, +0.000, +1.000)
    // 56 - θ·σv(Σ)(+0.000, +1.000, +0.000)
    // 57 - θ·σv(QΣ)(+0.000, +1.000, +0.000)
    // 58 - θ·σv(Σ)(+1.000, +0.000, +0.000)
    // 59 - θ·σv(QΣ)(+1.000, +0.000, +0.000)
    // 60 - θ·σv(Σ)(+0.707, +0.707, +0.000)
    // 61 - θ·σv(Σ)(+0.707, -0.707, +0.000)
    // 62 - θ·σv(QΣ)(+0.707, +0.707, +0.000)
    // 63 - θ·σv(QΣ)(+0.707, -0.707, +0.000)

    // -------------------------------
    // Explicit reference coefficients
    // -------------------------------
    let c4p1_nsr = group.get_index(2).unwrap();
    assert!(det.sym_transform_spatial(&c4p1_nsr).is_err());
    let tdet_c4p1_nsr = det.sym_transform_spin_spatial(&c4p1_nsr).unwrap();
    let sqrt2inv = std::f64::consts::FRAC_1_SQRT_2;
    #[rustfmt::skip]
    let tc_ref = array![
        [Complex::new(sqrt2inv, sqrt2inv)], [Complex::from(0.0)], // B0β
        [Complex::new(sqrt2inv, sqrt2inv)], [Complex::from(0.0)], // F3β
        [Complex::from(0.0)], [Complex::new(sqrt2inv, -sqrt2inv)], // F0α
        [Complex::new(sqrt2inv, sqrt2inv)], [Complex::from(0.0)], // F1β
        [Complex::from(0.0)], [Complex::new(sqrt2inv, -sqrt2inv)], // F2α
    ];
    let tdet_c4p1_nsr_ref = SlaterDeterminant::<Complex<f64>, SpinOrbitCoupled>::builder()
        .coefficients(&[tc_ref])
        .occupations(&[occ.clone()])
        .baos(vec![&bao_bf4])
        .mol(&mol_bf4)
        .structure_constraint(SpinOrbitCoupled::JAdapted(1))
        .complex_symmetric(false)
        .threshold(1e-14)
        .build()
        .unwrap();
    assert_eq!(tdet_c4p1_nsr, tdet_c4p1_nsr_ref);

    let c4p1_isr = group.get_index(3).unwrap();
    let tdet_c4p1_isr = det.sym_transform_spin_spatial(&c4p1_isr).unwrap();
    #[rustfmt::skip]
    let tc_ref = -array![
        [Complex::new(sqrt2inv, sqrt2inv)], [Complex::from(0.0)], // B0β
        [Complex::new(sqrt2inv, sqrt2inv)], [Complex::from(0.0)], // F3β
        [Complex::from(0.0)], [Complex::new(sqrt2inv, -sqrt2inv)], // F0α
        [Complex::new(sqrt2inv, sqrt2inv)], [Complex::from(0.0)], // F1β
        [Complex::from(0.0)], [Complex::new(sqrt2inv, -sqrt2inv)], // F2α
    ];
    let tdet_c4p1_isr_ref = SlaterDeterminant::<Complex<f64>, SpinOrbitCoupled>::builder()
        .coefficients(&[tc_ref])
        .occupations(&[occ.clone()])
        .bao(&bao_bf4)
        .mol(&mol_bf4)
        .structure_constraint(SpinOrbitCoupled::JAdapted(1))
        .complex_symmetric(false)
        .threshold(1e-14)
        .build()
        .unwrap();
    assert_eq!(tdet_c4p1_isr, tdet_c4p1_isr_ref);

    let c4pm1_nsr = group.get_index(4).unwrap();
    let tdet_c4pm1_nsr = det.sym_transform_spin_spatial(&c4pm1_nsr).unwrap();
    #[rustfmt::skip]
    let tc_ref = array![
        [Complex::new(sqrt2inv, -sqrt2inv)], [Complex::from(0.0)], // B0β
        [Complex::new(sqrt2inv, -sqrt2inv)], [Complex::from(0.0)], // F1β
        [Complex::from(0.0)], [Complex::new(sqrt2inv, sqrt2inv)], // F2α
        [Complex::new(sqrt2inv, -sqrt2inv)], [Complex::from(0.0)], // F3β
        [Complex::from(0.0)], [Complex::new(sqrt2inv, sqrt2inv)], // F0α
    ];
    let tdet_c4pm1_nsr_ref = SlaterDeterminant::<Complex<f64>, SpinOrbitCoupled>::builder()
        .coefficients(&[tc_ref])
        .occupations(&[occ.clone()])
        .baos(vec![&bao_bf4])
        .mol(&mol_bf4)
        .structure_constraint(SpinOrbitCoupled::JAdapted(1))
        .complex_symmetric(false)
        .threshold(1e-14)
        .build()
        .unwrap();
    assert_eq!(tdet_c4pm1_nsr, tdet_c4pm1_nsr_ref);

    let tdet_c4pm1_nsr_c4p1_nsr = det
        .sym_transform_spin_spatial(&c4pm1_nsr)
        .unwrap()
        .sym_transform_spin_spatial(&c4p1_nsr)
        .unwrap();
    assert!((&c4p1_nsr * &c4pm1_nsr).is_identity());
    assert_eq!(tdet_c4pm1_nsr_c4p1_nsr, det);

    // --------------------------------------------------
    // Implicit via representation/corepresentation rules
    // --------------------------------------------------

    //  4 - C4(Σ)(+0.000, +0.000, -1.000)
    // 15 - C2(QΣ)(+0.707, -0.707, +0.000)
    let tdet_s15_s4 = det
        .sym_transform_spin_spatial(&group.get_index(15).unwrap())
        .unwrap()
        .sym_transform_spin_spatial(&group.get_index(4).unwrap())
        .unwrap();
    let tdet_s4s15 = det
        .sym_transform_spin_spatial(&(group.get_index(4).unwrap() * group.get_index(15).unwrap()))
        .unwrap();
    assert_eq!(tdet_s15_s4, tdet_s4s15);

    // 18 - S4(Σ)(+0.000, +0.000, +1.000)
    // 32 - θ(Σ)
    let tdet_s18_s32 = det
        .sym_transform_spin_spatial(&group.get_index(18).unwrap())
        .unwrap()
        .sym_transform_spin_spatial(&group.get_index(32).unwrap())
        .unwrap();
    let tdet_s32s18 = det
        .sym_transform_spin_spatial(&(group.get_index(32).unwrap() * group.get_index(18).unwrap()))
        .unwrap();
    assert_eq!(tdet_s18_s32, tdet_s32s18);

    let tdet_s32_s18 = det
        .sym_transform_spin_spatial(&group.get_index(32).unwrap())
        .unwrap()
        .sym_transform_spin_spatial(&group.get_index(18).unwrap())
        .unwrap();
    let tdet_s18s32 = det
        .sym_transform_spin_spatial(&(group.get_index(18).unwrap() * group.get_index(32).unwrap()))
        .unwrap();
    assert_eq!(tdet_s32_s18, tdet_s18s32);

    // 21 - S4(QΣ)(+0.000, +0.000, -1.000)
    // 56 - θ·σv(Σ)(+0.000, +1.000, +0.000)
    let tdet_s21_s56 = det
        .sym_transform_spin_spatial(&group.get_index(21).unwrap())
        .unwrap()
        .sym_transform_spin_spatial(&group.get_index(56).unwrap())
        .unwrap();
    let tdet_s56s21 = det
        .sym_transform_spin_spatial(&(group.get_index(56).unwrap() * group.get_index(21).unwrap()))
        .unwrap();
    assert_eq!(tdet_s21_s56, tdet_s56s21);

    let tdet_s56_s21 = det
        .sym_transform_spin_spatial(&group.get_index(56).unwrap())
        .unwrap()
        .sym_transform_spin_spatial(&group.get_index(21).unwrap())
        .unwrap();
    let tdet_s21s56 = det
        .sym_transform_spin_spatial(&(group.get_index(21).unwrap() * group.get_index(56).unwrap()))
        .unwrap();
    assert_eq!(tdet_s56_s21, tdet_s21s56);

    // 51 - θ·S4(QΣ)(+0.000, +0.000, +1.000)
    // 55 - θ·σh(QΣ)(+0.000, +0.000, +1.000)
    let tdet_s51_s55 = det
        .sym_transform_spin_spatial(&group.get_index(51).unwrap())
        .unwrap()
        .sym_transform_spin_spatial(&group.get_index(55).unwrap())
        .unwrap();
    let tdet_s55s51 = det
        .sym_transform_spin_spatial(&(group.get_index(55).unwrap() * group.get_index(51).unwrap()))
        .unwrap();
    assert_eq!(tdet_s51_s55, tdet_s55s51);

    let tdet_s55_s51 = det
        .sym_transform_spin_spatial(&group.get_index(55).unwrap())
        .unwrap()
        .sym_transform_spin_spatial(&group.get_index(51).unwrap())
        .unwrap();
    let tdet_s51s55 = det
        .sym_transform_spin_spatial(&(group.get_index(51).unwrap() * group.get_index(55).unwrap()))
        .unwrap();
    assert_eq!(tdet_s55_s51, tdet_s51s55);
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
    let det0 = SlaterDeterminant::<f64, SpinConstraint>::builder()
        .coefficients(&[ca0.clone(), ca0])
        .occupations(&[oa0.clone(), oa0])
        .baos(vec![&bao_h4])
        .mol(&mol_h4)
        .structure_constraint(SpinConstraint::Unrestricted(2, false))
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
    let det1 = SlaterDeterminant::<f64, SpinConstraint>::builder()
        .coefficients(&[ca1.clone(), ca1])
        .occupations(&[oa1.clone(), oa1])
        .baos(vec![&bao_h4])
        .mol(&mol_h4)
        .structure_constraint(SpinConstraint::Unrestricted(2, false))
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
    let det2 = SlaterDeterminant::<f64, SpinConstraint>::builder()
        .coefficients(&[ca2.clone(), ca2])
        .occupations(&[oa2.clone(), oa2])
        .baos(vec![&bao_h4])
        .mol(&mol_h4)
        .structure_constraint(SpinConstraint::Unrestricted(2, false))
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
    let det3 = SlaterDeterminant::<f64, SpinConstraint>::builder()
        .coefficients(&[ca3.clone(), ca3])
        .occupations(&[oa3.clone(), oa3])
        .baos(vec![&bao_h4])
        .mol(&mol_h4)
        .structure_constraint(SpinConstraint::Unrestricted(2, false))
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
    let det4 = SlaterDeterminant::<f64, SpinConstraint>::builder()
        .coefficients(&[ca4.clone(), ca4])
        .occupations(&[oa4.clone(), oa4])
        .baos(vec![&bao_h4])
        .mol(&mol_h4)
        .structure_constraint(SpinConstraint::Unrestricted(2, false))
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
    let det5 = SlaterDeterminant::<f64, SpinConstraint>::builder()
        .coefficients(&[ca5.clone(), ca5])
        .occupations(&[oa5.clone(), oa5])
        .baos(vec![&bao_h4])
        .mol(&mol_h4)
        .structure_constraint(SpinConstraint::Unrestricted(2, false))
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
        [
            1.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0050563
        ],
        [
            0.0000000, 1.0000000, 0.0000000, 0.0000000, 0.0000000, 0.7611240
        ],
        [
            0.0000000, 0.0000000, 1.0000000, 0.0000000, 0.0000000, 0.0576832
        ],
        [
            0.0000000, 0.0000000, 0.0000000, 1.0000000, 0.0000000, 0.0719396
        ],
        [
            0.0000000, 0.0000000, 0.0000000, 0.0000000, 1.0000000, 0.1041424
        ],
        [
            0.0050563, 0.7611240, 0.0576832, 0.0719396, 0.1041424, 1.0000000
        ],
    ];
    close_l2(&smat, &smat_ref, 1e-7);

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
    close_l2(&smat_g, &smat_ref, 1e-7);

    // Complex, UHF
    let sao_c = sao.mapv(C128::from);
    let mut smat_c = Array2::<C128>::zeros((6, 6));
    [&det0, &det1, &det2, &det3, &det4, &det5]
        .iter()
        .map(|&det| SlaterDeterminant::<C128, SpinConstraint>::from(det.clone()))
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
    close_l2(&smat_c, &smat_c_ref, 1e-7);

    // Complex, GHF
    let sao_cg = sao_g.mapv(C128::from);
    let mut smat_cg = Array2::<C128>::zeros((6, 6));
    [&det0, &det1, &det2, &det3, &det4, &det5]
        .iter()
        .map(|&det| SlaterDeterminant::<C128, SpinConstraint>::from(det.clone()).to_generalised())
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
    close_l2(&smat_cg, &smat_c_ref, 1e-7);
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
    let det = SlaterDeterminant::<f64, SpinConstraint>::builder()
        .coefficients(&[calpha, cbeta])
        .occupations(&[oalpha, obeta])
        .baos(vec![&bao_s4])
        .mol(&mol_s4)
        .structure_constraint(SpinConstraint::Unrestricted(2, false))
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
    close_l2(&os, &Array2::<f64>::eye(os.shape()[0]), 1e-7);

    let det_c = SlaterDeterminant::<C128, SpinConstraint>::from(det.clone());
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
    close_l2(&os_c, &Array2::<C128>::eye(os.shape()[0]), 1e-7);

    close_l2(&os.map(C128::from), &os_c, 1e-7);
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
    let det_1e_cg: SlaterDeterminant<C128, SpinConstraint> =
        SlaterDeterminant::<f64, SpinConstraint>::builder()
            .coefficients(&[calpha.clone(), cbeta.clone()])
            .occupations(&[oalpha.clone(), obeta_empty])
            .baos(vec![&bao_s4])
            .mol(&mol_s4)
            .structure_constraint(SpinConstraint::Unrestricted(2, false))
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
    let det_2e_cg: SlaterDeterminant<C128, SpinConstraint> =
        SlaterDeterminant::<f64, SpinConstraint>::builder()
            .coefficients(&[calpha, cbeta])
            .occupations(&[oalpha, obeta_filled])
            .baos(vec![&bao_s4])
            .mol(&mol_s4)
            .structure_constraint(SpinConstraint::Unrestricted(2, false))
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
    assert!(
        orbit_cg_u_grey_d4h_double_spin_spatial_1e
            .analyse_rep()
            .is_err(),
    );

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
    assert!(
        orbit_cg_u_bw_d4h_c4h_double_spin_spatial_1e
            .analyse_rep()
            .is_err()
    );

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
fn test_determinant_orbit_rep_analysis_bh3_spintimerev_odd() {
    // env_logger::init();
    let emap = ElementMap::new();
    let atm_b0 = Atom::from_xyz("B   -0.0000000    0.0000000   -0.0000000", &emap, 1e-6).unwrap();
    let atm_h0 = Atom::from_xyz("H    0.5905582   -1.0228767    0.0000000", &emap, 1e-6).unwrap();
    let atm_h1 = Atom::from_xyz("H    0.5905582    1.0228767   -0.0000000", &emap, 1e-6).unwrap();
    let atm_h2 = Atom::from_xyz("H   -1.1811163    0.0000000   -0.0000000", &emap, 1e-6).unwrap();

    let bsp_s = BasisShell::new(0, ShellOrder::Pure(PureOrder::increasingm(0)));
    let bsp_p = BasisShell::new(1, ShellOrder::Pure(PureOrder::increasingm(1)));

    let batm_b0 = BasisAtom::new(&atm_b0, &[bsp_s.clone(), bsp_s.clone(), bsp_p]);
    let batm_h0 = BasisAtom::new(&atm_h0, &[bsp_s.clone()]);
    let batm_h1 = BasisAtom::new(&atm_h1, &[bsp_s.clone()]);
    let batm_h2 = BasisAtom::new(&atm_h2, &[bsp_s]);

    let bao_bh3 = BasisAngularOrder::new(&[batm_b0, batm_h0, batm_h1, batm_h2]);
    let mol_bh3 = Molecule::from_atoms(
        &[
            atm_b0.clone(),
            atm_h0.clone(),
            atm_h1.clone(),
            atm_h2.clone(),
        ],
        1e-6,
    )
    .recentre();

    let presym = PreSymmetry::builder()
        .moi_threshold(1e-6)
        .molecule(&mol_bh3)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false).unwrap();
    let group_u_d3h = UnitaryRepresentedGroup::from_molecular_symmetry(&sym, None).unwrap();

    let mut sym_tr = Symmetry::new();
    sym_tr.analyse(&presym, true).unwrap();
    let group_m_grey_d3h =
        MagneticRepresentedGroup::from_molecular_symmetry(&sym_tr, None).unwrap();
    let group_m_grey_d3h_double = group_m_grey_d3h.to_double_group().unwrap();

    // ==========
    // 5-electron
    // ==========
    // 1s² 2s² 2py
    #[rustfmt::skip]
    let calpha = array![
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ];
    #[rustfmt::skip]
    let cbeta = array![
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
    ];
    let oalpha = array![1.0, 1.0, 1.0];
    let obeta = array![1.0, 1.0];
    let det_cg: SlaterDeterminant<C128, SpinConstraint> =
        SlaterDeterminant::<f64, SpinConstraint>::builder()
            .coefficients(&[calpha.clone(), cbeta.clone()])
            .occupations(&[oalpha.clone(), obeta.clone()])
            .baos(vec![&bao_bh3])
            .mol(&mol_bh3)
            .structure_constraint(SpinConstraint::Unrestricted(2, false))
            .complex_symmetric(false)
            .threshold(1e-14)
            .build()
            .unwrap()
            .to_generalised()
            .into();

    let sao_cg = Array2::<C128>::eye(16);

    // Spatial, single unitary group
    let mut orbit_cg_u_d3h_spatial = SlaterDeterminantSymmetryOrbit::builder()
        .group(&group_u_d3h)
        .origin(&det_cg)
        .integrality_threshold(1e-7)
        .linear_independence_threshold(1e-7)
        .symmetry_transformation_kind(SymmetryTransformationKind::Spatial)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
        .build()
        .unwrap();
    let _ = orbit_cg_u_d3h_spatial
        .calc_smat(Some(&sao_cg), None, true)
        .unwrap()
        .calc_xmat(false);
    assert_eq!(
        orbit_cg_u_d3h_spatial.analyse_rep().unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new("||E|^(')|").unwrap()
    );

    // Spatial with spin-including time reversal, single magnetic grey group (corepresentations)
    let mut orbit_cg_m_grey_d3h_spatialwithspintimerev = SlaterDeterminantSymmetryOrbit::builder()
        .group(&group_m_grey_d3h)
        .origin(&det_cg)
        .integrality_threshold(1e-6)
        .linear_independence_threshold(1e-6)
        .symmetry_transformation_kind(SymmetryTransformationKind::SpatialWithSpinTimeReversal)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
        .build()
        .unwrap();
    let _ = orbit_cg_m_grey_d3h_spatialwithspintimerev
        .calc_smat(Some(&sao_cg), None, true)
        .unwrap()
        .calc_xmat(false);
    assert_eq!(
        orbit_cg_m_grey_d3h_spatialwithspintimerev
            .analyse_rep()
            .unwrap(),
        DecomposedSymbol::<MullikenIrcorepSymbol>::new("2||E|^(')|").unwrap()
    );

    // Spatial with spin-including time reversal, double magnetic grey group (corepresentations)
    let mut orbit_cg_m_grey_d3h_spatialwithspintimerev = SlaterDeterminantSymmetryOrbit::builder()
        .group(&group_m_grey_d3h_double)
        .origin(&det_cg)
        .integrality_threshold(1e-6)
        .linear_independence_threshold(1e-6)
        .symmetry_transformation_kind(SymmetryTransformationKind::SpatialWithSpinTimeReversal)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
        .build()
        .unwrap();
    let _ = orbit_cg_m_grey_d3h_spatialwithspintimerev
        .calc_smat(Some(&sao_cg), None, true)
        .unwrap()
        .calc_xmat(false);
    assert_eq!(
        orbit_cg_m_grey_d3h_spatialwithspintimerev
            .analyse_rep()
            .unwrap(),
        DecomposedSymbol::<MullikenIrcorepSymbol>::new("2||E|^(')|").unwrap()
    );

    // Spin-spatial, double magnetic grey group (corepresentations)
    let mut orbit_cg_m_grey_d3h_spinspatial = SlaterDeterminantSymmetryOrbit::builder()
        .group(&group_m_grey_d3h_double)
        .origin(&det_cg)
        .integrality_threshold(1e-6)
        .linear_independence_threshold(1e-6)
        .symmetry_transformation_kind(SymmetryTransformationKind::SpinSpatial)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
        .build()
        .unwrap();
    let _ = orbit_cg_m_grey_d3h_spinspatial
        .calc_smat(Some(&sao_cg), None, true)
        .unwrap()
        .calc_xmat(false);
    assert_eq!(
        orbit_cg_m_grey_d3h_spinspatial.analyse_rep().unwrap(),
        DecomposedSymbol::<MullikenIrcorepSymbol>::new("||E~|_(2)| ⊕ ||E~|_(3)|").unwrap()
    );
}

#[test]
fn test_determinant_orbit_rep_analysis_bh3_spintimerev_even() {
    // env_logger::init();
    let emap = ElementMap::new();
    let atm_b0 = Atom::from_xyz("B   -0.0000000    0.0000000   -0.0000000", &emap, 1e-6).unwrap();
    let atm_h0 = Atom::from_xyz("H    0.5905582   -1.0228767    0.0000000", &emap, 1e-6).unwrap();
    let atm_h1 = Atom::from_xyz("H    0.5905582    1.0228767   -0.0000000", &emap, 1e-6).unwrap();
    let atm_h2 = Atom::from_xyz("H   -1.1811163    0.0000000   -0.0000000", &emap, 1e-6).unwrap();

    let bsp_s = BasisShell::new(0, ShellOrder::Pure(PureOrder::increasingm(0)));
    let bsp_p = BasisShell::new(1, ShellOrder::Pure(PureOrder::increasingm(1)));

    let batm_b0 = BasisAtom::new(&atm_b0, &[bsp_s.clone(), bsp_s.clone(), bsp_p]);
    let batm_h0 = BasisAtom::new(&atm_h0, &[bsp_s.clone()]);
    let batm_h1 = BasisAtom::new(&atm_h1, &[bsp_s.clone()]);
    let batm_h2 = BasisAtom::new(&atm_h2, &[bsp_s]);

    let bao_bh3 = BasisAngularOrder::new(&[batm_b0, batm_h0, batm_h1, batm_h2]);
    let mol_bh3 = Molecule::from_atoms(
        &[
            atm_b0.clone(),
            atm_h0.clone(),
            atm_h1.clone(),
            atm_h2.clone(),
        ],
        1e-6,
    )
    .recentre();

    let presym = PreSymmetry::builder()
        .moi_threshold(1e-6)
        .molecule(&mol_bh3)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false).unwrap();
    let group_u_d3h = UnitaryRepresentedGroup::from_molecular_symmetry(&sym, None).unwrap();

    let mut sym_tr = Symmetry::new();
    sym_tr.analyse(&presym, true).unwrap();
    let group_u_grey_d3h = UnitaryRepresentedGroup::from_molecular_symmetry(&sym_tr, None).unwrap();
    let group_u_grey_d3h_double = group_u_grey_d3h.to_double_group().unwrap();

    let group_m_grey_d3h =
        MagneticRepresentedGroup::from_molecular_symmetry(&sym_tr, None).unwrap();
    let group_m_grey_d3h_double = group_m_grey_d3h.to_double_group().unwrap();

    // ==========
    // 6-electron
    // ==========
    // 1s² 2s² 2pyα 2pzβ
    #[rustfmt::skip]
    let calpha = array![
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ];
    #[rustfmt::skip]
    let cbeta = array![
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ];
    let oalpha = array![1.0, 1.0, 1.0];
    let obeta = array![1.0, 1.0, 1.0];
    let det_cg: SlaterDeterminant<C128, SpinConstraint> =
        SlaterDeterminant::<f64, SpinConstraint>::builder()
            .coefficients(&[calpha.clone(), cbeta.clone()])
            .occupations(&[oalpha.clone(), obeta.clone()])
            .baos(vec![&bao_bh3])
            .mol(&mol_bh3)
            .structure_constraint(SpinConstraint::Unrestricted(2, false))
            .complex_symmetric(false)
            .threshold(1e-14)
            .build()
            .unwrap()
            .to_generalised()
            .into();

    let sao_cg = Array2::<C128>::eye(16);

    // Spatial, single unitary group
    let mut orbit_cg_u_d3h_spatial = SlaterDeterminantSymmetryOrbit::builder()
        .group(&group_u_d3h)
        .origin(&det_cg)
        .integrality_threshold(1e-7)
        .linear_independence_threshold(1e-7)
        .symmetry_transformation_kind(SymmetryTransformationKind::Spatial)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
        .build()
        .unwrap();
    let _ = orbit_cg_u_d3h_spatial
        .calc_smat(Some(&sao_cg), None, true)
        .unwrap()
        .calc_xmat(false);
    assert_eq!(
        orbit_cg_u_d3h_spatial.analyse_rep().unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new("||E|^('')|").unwrap()
    );

    // Spatial with spin-including time reversal, single magnetic grey group (representations)
    let mut orbit_cg_u_grey_d3h_spatialwithspintimerev = SlaterDeterminantSymmetryOrbit::builder()
        .group(&group_u_grey_d3h)
        .origin(&det_cg)
        .integrality_threshold(1e-6)
        .linear_independence_threshold(1e-6)
        .symmetry_transformation_kind(SymmetryTransformationKind::SpatialWithSpinTimeReversal)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
        .build()
        .unwrap();
    let _ = orbit_cg_u_grey_d3h_spatialwithspintimerev
        .calc_smat(Some(&sao_cg), None, true)
        .unwrap()
        .calc_xmat(false);
    assert_eq!(
        orbit_cg_u_grey_d3h_spatialwithspintimerev
            .analyse_rep()
            .unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new("|^(+)|E|^('')| ⊕ |^(-)|E|^('')|").unwrap()
    );

    // Spatial with spin-including time reversal, single magnetic grey group (corepresentations)
    let mut orbit_cg_m_grey_d3h_spatialwithspintimerev = SlaterDeterminantSymmetryOrbit::builder()
        .group(&group_m_grey_d3h)
        .origin(&det_cg)
        .integrality_threshold(1e-6)
        .linear_independence_threshold(1e-6)
        .symmetry_transformation_kind(SymmetryTransformationKind::SpatialWithSpinTimeReversal)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
        .build()
        .unwrap();
    let _ = orbit_cg_m_grey_d3h_spatialwithspintimerev
        .calc_smat(Some(&sao_cg), None, true)
        .unwrap()
        .calc_xmat(false);
    assert_eq!(
        orbit_cg_m_grey_d3h_spatialwithspintimerev
            .analyse_rep()
            .unwrap(),
        DecomposedSymbol::<MullikenIrcorepSymbol>::new("2||E|^('')|").unwrap()
    );

    // Spatial with spin-including time reversal, double magnetic grey group (representations)
    let mut orbit_cg_m_grey_d3h_spatialwithspintimerev_double =
        SlaterDeterminantSymmetryOrbit::builder()
            .group(&group_u_grey_d3h_double)
            .origin(&det_cg)
            .integrality_threshold(1e-6)
            .linear_independence_threshold(1e-6)
            .symmetry_transformation_kind(SymmetryTransformationKind::SpatialWithSpinTimeReversal)
            .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
            .build()
            .unwrap();
    let _ = orbit_cg_m_grey_d3h_spatialwithspintimerev_double
        .calc_smat(Some(&sao_cg), None, true)
        .unwrap()
        .calc_xmat(false);
    assert_eq!(
        orbit_cg_m_grey_d3h_spatialwithspintimerev_double
            .analyse_rep()
            .unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new("|^(+)|E|^('')| ⊕ |^(-)|E|^('')|").unwrap()
    );

    // Spatial with spin-including time reversal, double magnetic grey group (corepresentations)
    let mut orbit_cg_m_grey_d3h_spatialwithspintimerev_double =
        SlaterDeterminantSymmetryOrbit::builder()
            .group(&group_m_grey_d3h_double)
            .origin(&det_cg)
            .integrality_threshold(1e-6)
            .linear_independence_threshold(1e-6)
            .symmetry_transformation_kind(SymmetryTransformationKind::SpatialWithSpinTimeReversal)
            .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
            .build()
            .unwrap();
    let _ = orbit_cg_m_grey_d3h_spatialwithspintimerev_double
        .calc_smat(Some(&sao_cg), None, true)
        .unwrap()
        .calc_xmat(false);
    assert_eq!(
        orbit_cg_m_grey_d3h_spatialwithspintimerev_double
            .analyse_rep()
            .unwrap(),
        DecomposedSymbol::<MullikenIrcorepSymbol>::new("2||E|^('')|").unwrap()
    );

    // Spin-spatial, double magnetic grey group (representations)
    let mut orbit_cg_u_grey_d3h_spinspatial_double = SlaterDeterminantSymmetryOrbit::builder()
        .group(&group_u_grey_d3h_double)
        .origin(&det_cg)
        .integrality_threshold(1e-6)
        .linear_independence_threshold(1e-6)
        .symmetry_transformation_kind(SymmetryTransformationKind::SpinSpatial)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
        .build()
        .unwrap();
    let _ = orbit_cg_u_grey_d3h_spinspatial_double
        .calc_smat(Some(&sao_cg), None, true)
        .unwrap()
        .calc_xmat(false);
    assert_eq!(
        orbit_cg_u_grey_d3h_spinspatial_double
            .analyse_rep()
            .unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new("|^(+)|E|^('')| ⊕ |^(-)|E|^('')|").unwrap()
    );

    // Spin-spatial, double magnetic grey group (corepresentations)
    let mut orbit_cg_m_grey_d3h_spinspatial_double = SlaterDeterminantSymmetryOrbit::builder()
        .group(&group_m_grey_d3h_double)
        .origin(&det_cg)
        .integrality_threshold(1e-6)
        .linear_independence_threshold(1e-6)
        .symmetry_transformation_kind(SymmetryTransformationKind::SpinSpatial)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
        .build()
        .unwrap();
    let _ = orbit_cg_m_grey_d3h_spinspatial_double
        .calc_smat(Some(&sao_cg), None, true)
        .unwrap()
        .calc_xmat(false);
    assert_eq!(
        orbit_cg_m_grey_d3h_spinspatial_double
            .analyse_rep()
            .unwrap(),
        DecomposedSymbol::<MullikenIrcorepSymbol>::new("2||E|^('')|").unwrap()
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
    let det_dyy_cg: SlaterDeterminant<C128, SpinConstraint> =
        SlaterDeterminant::<f64, SpinConstraint>::builder()
            .coefficients(&[calpha, cbeta])
            .occupations(&[oalpha, obeta_empty])
            .baos(vec![&bao_vf6])
            .mol(&mol_vf6)
            .structure_constraint(SpinConstraint::Unrestricted(2, false))
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
    let det_dxz_cg: SlaterDeterminant<C128, SpinConstraint> =
        SlaterDeterminant::<f64, SpinConstraint>::builder()
            .coefficients(&[calpha, cbeta])
            .occupations(&[oalpha, obeta_empty])
            .baos(vec![&bao_vf6])
            .mol(&mol_vf6)
            .structure_constraint(SpinConstraint::Unrestricted(2, false))
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

#[test]
fn test_determinant_orbit_rep_analysis_h_jadapted() {
    // env_logger::init();
    let emap = ElementMap::new();
    let atm_h0 = Atom::from_xyz("H 0.0 0.0 0.0", &emap, 1e-7).unwrap();

    let bs_sp1 = BasisShell::new(
        1,
        ShellOrder::Spinor(SpinorOrder::increasingm(1, true, None)),
    );
    let bs_sp3 = BasisShell::new(
        3,
        ShellOrder::Spinor(SpinorOrder::increasingm(3, true, None)),
    );
    let bs_sp5 = BasisShell::new(
        5,
        ShellOrder::Spinor(SpinorOrder::increasingm(5, true, None)),
    );
    let bs_sp7 = BasisShell::new(
        7,
        ShellOrder::Spinor(SpinorOrder::increasingm(7, true, None)),
    );

    let batm_h0 = BasisAtom::new(
        &atm_h0,
        &[
            bs_sp1.clone(),
            bs_sp3.clone(),
            bs_sp5.clone(),
            bs_sp7.clone(),
        ],
    );
    let bao_h = BasisAngularOrder::new(&[batm_h0]);
    let mol_h = Molecule::from_atoms(&[atm_h0.clone()], 1e-7);

    // |1/2, -1/2⟩
    #[rustfmt::skip]
    let c_12 = array![
        [Complex::new(1.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
    ];
    let occ = array![1.0];

    let det_12 = SlaterDeterminant::<Complex<f64>, SpinOrbitCoupled>::builder()
        .coefficients(&[c_12])
        .occupations(&[occ.clone()])
        .baos(vec![&bao_h])
        .mol(&mol_h)
        .structure_constraint(SpinOrbitCoupled::JAdapted(1))
        .complex_symmetric(false)
        .threshold(1e-14)
        .build()
        .unwrap();

    // |3/2, -1/2⟩
    #[rustfmt::skip]
    let c_32 = array![
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(1.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
    ];

    let det_32 = SlaterDeterminant::<Complex<f64>, SpinOrbitCoupled>::builder()
        .coefficients(&[c_32])
        .occupations(&[occ.clone()])
        .baos(vec![&bao_h])
        .mol(&mol_h)
        .structure_constraint(SpinOrbitCoupled::JAdapted(1))
        .complex_symmetric(false)
        .threshold(1e-14)
        .build()
        .unwrap();

    // |5/2, 5/2⟩
    #[rustfmt::skip]
    let c_52 = array![
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(1.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
    ];

    let det_52 = SlaterDeterminant::<Complex<f64>, SpinOrbitCoupled>::builder()
        .coefficients(&[c_52])
        .occupations(&[occ.clone()])
        .baos(vec![&bao_h])
        .mol(&mol_h)
        .structure_constraint(SpinOrbitCoupled::JAdapted(1))
        .complex_symmetric(false)
        .threshold(1e-14)
        .build()
        .unwrap();

    // |7/2, 1/2⟩ + |7/2, 5/2⟩
    #[rustfmt::skip]
    let c_72 = array![
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.70710678118, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.70710678118, 0.0)],
        [Complex::new(0.0, 0.0)],
    ];

    let det_72 = SlaterDeterminant::<Complex<f64>, SpinOrbitCoupled>::builder()
        .coefficients(&[c_72])
        .occupations(&[occ.clone()])
        .baos(vec![&bao_h])
        .mol(&mol_h)
        .structure_constraint(SpinOrbitCoupled::JAdapted(1))
        .complex_symmetric(false)
        .threshold(1e-14)
        .build()
        .unwrap();

    // |1/2, -1/2⟩ ⊗ |1/2, +1/2⟩
    #[rustfmt::skip]
    let c_1212 = array![
        [Complex::new(1.0, 0.0), Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0), Complex::new(1.0, 0.0)],
        [Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)],
    ];
    let occ = array![1.0, 1.0];

    let det_1212 = SlaterDeterminant::<Complex<f64>, SpinOrbitCoupled>::builder()
        .coefficients(&[c_1212])
        .occupations(&[occ.clone()])
        .baos(vec![&bao_h])
        .mol(&mol_h)
        .structure_constraint(SpinOrbitCoupled::JAdapted(1))
        .complex_symmetric(false)
        .threshold(1e-14)
        .build()
        .unwrap();

    let sao: Array2<f64> = Array2::eye(20);
    let sao_c = sao.mapv(C128::from);

    // ~~~~~~~~~~~~~~~~~~~~~~~
    // u Oh* (double, unitary)
    // ~~~~~~~~~~~~~~~~~~~~~~~
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol_h)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false).unwrap();
    let group_u_oh = UnitaryRepresentedGroup::from_molecular_symmetry(&sym, Some(4)).unwrap();
    let group_u_oh_double = group_u_oh.to_double_group().unwrap();

    let mut orbit_c_u_oh_double_spinspatial_12 = SlaterDeterminantSymmetryOrbit::builder()
        .group(&group_u_oh_double)
        .origin(&det_12)
        .integrality_threshold(1e-14)
        .linear_independence_threshold(1e-12)
        .symmetry_transformation_kind(SymmetryTransformationKind::SpinSpatial)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
        .build()
        .unwrap();
    let _ = orbit_c_u_oh_double_spinspatial_12
        .calc_smat(Some(&sao_c), None, true)
        .unwrap()
        .calc_xmat(false);
    assert_eq!(
        orbit_c_u_oh_double_spinspatial_12.analyse_rep().unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new("||E~|_(1g)|").unwrap()
    );

    let mut orbit_c_u_oh_double_spinspatial_32 = SlaterDeterminantSymmetryOrbit::builder()
        .group(&group_u_oh_double)
        .origin(&det_32)
        .integrality_threshold(1e-14)
        .linear_independence_threshold(1e-12)
        .symmetry_transformation_kind(SymmetryTransformationKind::SpinSpatial)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
        .build()
        .unwrap();
    let _ = orbit_c_u_oh_double_spinspatial_32
        .calc_smat(Some(&sao_c), None, true)
        .unwrap()
        .calc_xmat(false);
    assert_eq!(
        orbit_c_u_oh_double_spinspatial_32.analyse_rep().unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new("||F~|_(g)|").unwrap()
    );

    let mut orbit_c_u_oh_double_spinspatial_52 = SlaterDeterminantSymmetryOrbit::builder()
        .group(&group_u_oh_double)
        .origin(&det_52)
        .integrality_threshold(1e-14)
        .linear_independence_threshold(1e-12)
        .symmetry_transformation_kind(SymmetryTransformationKind::SpinSpatial)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
        .build()
        .unwrap();
    let _ = orbit_c_u_oh_double_spinspatial_52
        .calc_smat(Some(&sao_c), None, true)
        .unwrap()
        .calc_xmat(false);
    assert_eq!(
        orbit_c_u_oh_double_spinspatial_52.analyse_rep().unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new("||E~|_(2g)| ⊕ ||F~|_(g)|").unwrap()
    );

    let mut orbit_c_u_oh_double_spinspatial_72 = SlaterDeterminantSymmetryOrbit::builder()
        .group(&group_u_oh_double)
        .origin(&det_72)
        .integrality_threshold(1e-14)
        .linear_independence_threshold(1e-12)
        .symmetry_transformation_kind(SymmetryTransformationKind::SpinSpatial)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
        .build()
        .unwrap();
    let _ = orbit_c_u_oh_double_spinspatial_72
        .calc_smat(Some(&sao_c), None, true)
        .unwrap()
        .calc_xmat(false);
    assert_eq!(
        orbit_c_u_oh_double_spinspatial_72.analyse_rep().unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new("||E~|_(1g)| ⊕ ||E~|_(2g)| ⊕ ||F~|_(g)|")
            .unwrap()
    );

    let mut orbit_c_u_oh_double_spinspatial_1212 = SlaterDeterminantSymmetryOrbit::builder()
        .group(&group_u_oh_double)
        .origin(&det_1212)
        .integrality_threshold(1e-14)
        .linear_independence_threshold(1e-12)
        .symmetry_transformation_kind(SymmetryTransformationKind::SpinSpatial)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
        .build()
        .unwrap();
    let _ = orbit_c_u_oh_double_spinspatial_1212
        .calc_smat(Some(&sao_c), None, true)
        .unwrap()
        .calc_xmat(false);
    assert_eq!(
        orbit_c_u_oh_double_spinspatial_1212.analyse_rep().unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new("||A|_(1g)|").unwrap()
    );

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // u Oh'* (gray double, unitary)
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol_h)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, true).unwrap();
    let group_u_oh_gray = UnitaryRepresentedGroup::from_molecular_symmetry(&sym, Some(4)).unwrap();
    let group_u_oh_gray_double = group_u_oh_gray.to_double_group().unwrap();

    let mut orbit_c_u_oh_gray_double_spinspatial_12 = SlaterDeterminantSymmetryOrbit::builder()
        .group(&group_u_oh_gray_double)
        .origin(&det_12)
        .integrality_threshold(1e-14)
        .linear_independence_threshold(1e-12)
        .symmetry_transformation_kind(SymmetryTransformationKind::SpinSpatial)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
        .build()
        .unwrap();
    let _ = orbit_c_u_oh_gray_double_spinspatial_12
        .calc_smat(Some(&sao_c), None, true)
        .unwrap()
        .calc_xmat(false);
    // Unitary-represented magnetic groups cannot be used for symmetry analysis of odd-electron
    // systems where spin is treated explicitly.
    assert!(
        orbit_c_u_oh_gray_double_spinspatial_12
            .analyse_rep()
            .is_err()
    );

    let mut orbit_c_u_oh_gray_double_spinspatial_1212 = SlaterDeterminantSymmetryOrbit::builder()
        .group(&group_u_oh_gray_double)
        .origin(&det_1212)
        .integrality_threshold(1e-12)
        .linear_independence_threshold(1e-12)
        .symmetry_transformation_kind(SymmetryTransformationKind::SpinSpatial)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
        .build()
        .unwrap();
    let _ = orbit_c_u_oh_gray_double_spinspatial_1212
        .calc_smat(Some(&sao_c), None, true)
        .unwrap()
        .calc_xmat(false);
    assert_eq!(
        orbit_c_u_oh_gray_double_spinspatial_1212
            .analyse_rep()
            .unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new("|^(+)|A|_(1g)|").unwrap()
    );

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // m Oh'* (gray double, magnetic)
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol_h)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, true).unwrap();
    let group_m_oh_gray = MagneticRepresentedGroup::from_molecular_symmetry(&sym, Some(4)).unwrap();
    let group_m_oh_gray_double = group_m_oh_gray.to_double_group().unwrap();

    let mut orbit_c_m_oh_gray_double_spinspatial_12 = SlaterDeterminantSymmetryOrbit::builder()
        .group(&group_m_oh_gray_double)
        .origin(&det_12)
        .integrality_threshold(1e-10)
        .linear_independence_threshold(1e-8)
        .symmetry_transformation_kind(SymmetryTransformationKind::SpinSpatial)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
        .build()
        .unwrap();
    let _ = orbit_c_m_oh_gray_double_spinspatial_12
        .calc_smat(Some(&sao_c), None, true)
        .unwrap()
        .calc_xmat(false);
    assert_eq!(
        orbit_c_m_oh_gray_double_spinspatial_12
            .analyse_rep()
            .unwrap(),
        DecomposedSymbol::<MullikenIrcorepSymbol>::new("||E~|_(1g)|").unwrap()
    );

    let mut orbit_c_m_oh_gray_double_spinspatial_32 = SlaterDeterminantSymmetryOrbit::builder()
        .group(&group_m_oh_gray_double)
        .origin(&det_32)
        .integrality_threshold(1e-10)
        .linear_independence_threshold(1e-8)
        .symmetry_transformation_kind(SymmetryTransformationKind::SpinSpatial)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
        .build()
        .unwrap();
    let _ = orbit_c_m_oh_gray_double_spinspatial_32
        .calc_smat(Some(&sao_c), None, true)
        .unwrap()
        .calc_xmat(false);
    assert_eq!(
        orbit_c_m_oh_gray_double_spinspatial_32
            .analyse_rep()
            .unwrap(),
        DecomposedSymbol::<MullikenIrcorepSymbol>::new("||F~|_(g)|").unwrap()
    );

    let mut orbit_c_m_oh_gray_double_spinspatial_52 = SlaterDeterminantSymmetryOrbit::builder()
        .group(&group_m_oh_gray_double)
        .origin(&det_52)
        .integrality_threshold(1e-10)
        .linear_independence_threshold(1e-8)
        .symmetry_transformation_kind(SymmetryTransformationKind::SpinSpatial)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
        .build()
        .unwrap();
    let _ = orbit_c_m_oh_gray_double_spinspatial_52
        .calc_smat(Some(&sao_c), None, true)
        .unwrap()
        .calc_xmat(false);
    assert_eq!(
        orbit_c_m_oh_gray_double_spinspatial_52
            .analyse_rep()
            .unwrap(),
        DecomposedSymbol::<MullikenIrcorepSymbol>::new("||E~|_(2g)| ⊕ ||F~|_(g)|").unwrap()
    );

    let mut orbit_c_m_oh_gray_double_spinspatial_72 = SlaterDeterminantSymmetryOrbit::builder()
        .group(&group_m_oh_gray_double)
        .origin(&det_72)
        .integrality_threshold(1e-10)
        .linear_independence_threshold(1e-8)
        .symmetry_transformation_kind(SymmetryTransformationKind::SpinSpatial)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
        .build()
        .unwrap();
    let _ = orbit_c_m_oh_gray_double_spinspatial_72
        .calc_smat(Some(&sao_c), None, true)
        .unwrap()
        .calc_xmat(false);
    assert_eq!(
        orbit_c_m_oh_gray_double_spinspatial_72
            .analyse_rep()
            .unwrap(),
        DecomposedSymbol::<MullikenIrcorepSymbol>::new("||E~|_(1g)| ⊕ ||E~|_(2g)| ⊕ ||F~|_(g)|")
            .unwrap()
    );

    let mut orbit_c_m_oh_gray_double_spinspatial_1212 = SlaterDeterminantSymmetryOrbit::builder()
        .group(&group_m_oh_gray_double)
        .origin(&det_1212)
        .integrality_threshold(1e-10)
        .linear_independence_threshold(1e-8)
        .symmetry_transformation_kind(SymmetryTransformationKind::SpinSpatial)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
        .build()
        .unwrap();
    let _ = orbit_c_m_oh_gray_double_spinspatial_1212
        .calc_smat(Some(&sao_c), None, true)
        .unwrap()
        .calc_xmat(false);
    assert_eq!(
        orbit_c_m_oh_gray_double_spinspatial_1212
            .analyse_rep()
            .unwrap(),
        DecomposedSymbol::<MullikenIrcorepSymbol>::new("||A|_(1g)|").unwrap()
    );
}

#[test]
fn test_determinant_orbit_rep_analysis_bh4_tet_jadapted() {
    // env_logger::init();
    let emap = ElementMap::new();
    let atm_b0 = Atom::from_xyz("B 0.0 0.0 0.0", &emap, 1e-7).unwrap();
    let atm_h0 = Atom::from_xyz("H    0.6405130   -0.6405130    0.6405130", &emap, 1e-7).unwrap();
    let atm_h1 = Atom::from_xyz("H    0.6405130    0.6405130   -0.6405130", &emap, 1e-7).unwrap();
    let atm_h2 = Atom::from_xyz("H   -0.6405130    0.6405130    0.6405130", &emap, 1e-7).unwrap();
    let atm_h3 = Atom::from_xyz("H   -0.6405130   -0.6405130   -0.6405130", &emap, 1e-7).unwrap();

    let bs_sp1 = BasisShell::new(
        1,
        ShellOrder::Spinor(SpinorOrder::increasingm(1, true, None)),
    );
    let bs_sp3 = BasisShell::new(
        3,
        ShellOrder::Spinor(SpinorOrder::increasingm(3, true, None)),
    );

    let batm_b0 = BasisAtom::new(&atm_b0, &[bs_sp1.clone(), bs_sp1.clone(), bs_sp3.clone()]);
    let batm_h0 = BasisAtom::new(&atm_h0, &[bs_sp1.clone()]);
    let batm_h1 = BasisAtom::new(&atm_h1, &[bs_sp1.clone()]);
    let batm_h2 = BasisAtom::new(&atm_h2, &[bs_sp1.clone()]);
    let batm_h3 = BasisAtom::new(&atm_h3, &[bs_sp1.clone()]);
    let bao_bh4 = BasisAngularOrder::new(&[batm_b0, batm_h0, batm_h1, batm_h2, batm_h3]);
    let mol_bh4 = Molecule::from_atoms(
        &[
            atm_b0.clone(),
            atm_h0.clone(),
            atm_h1.clone(),
            atm_h2.clone(),
            atm_h3.clone(),
        ],
        1e-7,
    );

    // a|1/2, 1/2⟩ ⊗ b|1/2, 1/2⟩
    #[rustfmt::skip]
    let c_1212 = array![
        // B
        [Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)],
        [Complex::new(1.0, 0.0), Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0), Complex::new(1.0, 0.0)],
        [Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)],
        // H0
        [Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)],
        // H1
        [Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)],
        // H2
        [Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)],
        // H3
        [Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)],
    ];
    let occ = array![1.0, 1.0];

    let det_1212 = SlaterDeterminant::<Complex<f64>, SpinOrbitCoupled>::builder()
        .coefficients(&[c_1212])
        .occupations(&[occ])
        .baos(vec![&bao_bh4])
        .mol(&mol_bh4)
        .structure_constraint(SpinOrbitCoupled::JAdapted(1))
        .complex_symmetric(false)
        .threshold(1e-14)
        .build()
        .unwrap();

    // a|1/2, 1/2⟩ ⊗ b|1/2, -1/2⟩
    #[rustfmt::skip]
    let c_12m12 = array![
        // B
        [Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)],
        [Complex::new(1.0, 0.0), Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0), Complex::new(1.0, 0.0)],
        [Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)],
        // H0
        [Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)],
        // H1
        [Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)],
        // H2
        [Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)],
        // H3
        [Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)],
    ];
    let occ = array![1.0, 1.0];

    let det_12m12 = SlaterDeterminant::<Complex<f64>, SpinOrbitCoupled>::builder()
        .coefficients(&[c_12m12])
        .occupations(&[occ])
        .baos(vec![&bao_bh4])
        .mol(&mol_bh4)
        .structure_constraint(SpinOrbitCoupled::JAdapted(1))
        .complex_symmetric(false)
        .threshold(1e-14)
        .build()
        .unwrap();

    // |1/2, 1/2⟩ ⊗ (|3/2, -1/2⟩ ⊕ |3/2, 3/2⟩)
    #[rustfmt::skip]
    let c_1232 = array![
        // B
        [Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)],
        [Complex::new(1.0, 0.0), Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0), Complex::new(1.0, 0.0)],
        [Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0), Complex::new(1.0, 0.0)],
        // H0
        [Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)],
        // H1
        [Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)],
        // H2
        [Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)],
        // H3
        [Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)],
    ];
    let occ = array![1.0, 1.0];

    let det_1232 = SlaterDeterminant::<Complex<f64>, SpinOrbitCoupled>::builder()
        .coefficients(&[c_1232])
        .occupations(&[occ])
        .baos(vec![&bao_bh4])
        .mol(&mol_bh4)
        .structure_constraint(SpinOrbitCoupled::JAdapted(1))
        .complex_symmetric(false)
        .threshold(1e-14)
        .build()
        .unwrap();

    let sao: Array2<f64> = Array2::eye(16);
    let sao_c = sao.mapv(C128::from);

    // ~~~~~~~~~~~~~~~~~~~~~~~
    // u Td* (double, unitary)
    // ~~~~~~~~~~~~~~~~~~~~~~~
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol_bh4)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false).unwrap();
    let group_u_td = UnitaryRepresentedGroup::from_molecular_symmetry(&sym, None).unwrap();
    let group_u_td_double = group_u_td.to_double_group().unwrap();

    let mut orbit_c_u_td_double_spinspatial_1212 = SlaterDeterminantSymmetryOrbit::builder()
        .group(&group_u_td_double)
        .origin(&det_1212)
        .integrality_threshold(1e-14)
        .linear_independence_threshold(1e-12)
        .symmetry_transformation_kind(SymmetryTransformationKind::SpinSpatial)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
        .build()
        .unwrap();
    let _ = orbit_c_u_td_double_spinspatial_1212
        .calc_smat(Some(&sao_c), None, true)
        .unwrap()
        .calc_xmat(false);
    assert_eq!(
        orbit_c_u_td_double_spinspatial_1212.analyse_rep().unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new("||T|_(1)|").unwrap()
    );

    let mut orbit_c_u_td_double_spinspatial_12m12 = SlaterDeterminantSymmetryOrbit::builder()
        .group(&group_u_td_double)
        .origin(&det_12m12)
        .integrality_threshold(1e-14)
        .linear_independence_threshold(1e-12)
        .symmetry_transformation_kind(SymmetryTransformationKind::SpinSpatial)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
        .build()
        .unwrap();
    let _ = orbit_c_u_td_double_spinspatial_12m12
        .calc_smat(Some(&sao_c), None, true)
        .unwrap()
        .calc_xmat(false);
    assert_eq!(
        orbit_c_u_td_double_spinspatial_12m12.analyse_rep().unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new("||A|_(1)| ⊕ ||T|_(1)|").unwrap()
    );

    let mut orbit_c_u_td_double_spinspatial_1232 = SlaterDeterminantSymmetryOrbit::builder()
        .group(&group_u_td_double)
        .origin(&det_1232)
        .integrality_threshold(1e-14)
        .linear_independence_threshold(1e-12)
        .symmetry_transformation_kind(SymmetryTransformationKind::SpinSpatial)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
        .build()
        .unwrap();
    let _ = orbit_c_u_td_double_spinspatial_1232
        .calc_smat(Some(&sao_c), None, true)
        .unwrap()
        .calc_xmat(false);
    assert_eq!(
        orbit_c_u_td_double_spinspatial_1232.analyse_rep().unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new("||E|| ⊕ ||T|_(1)| ⊕ ||T|_(2)|").unwrap()
    );
}

#[test]
fn test_determinant_orbit_rep_analysis_bh3_jadapted() {
    // env_logger::init();
    let emap = ElementMap::new();
    let atm_b0 = Atom::from_xyz("B 0.0 0.0 0.0", &emap, 1e-7).unwrap();
    let atm_h0 = Atom::from_xyz("H  0.5905546  1.0228705 0.0000000", &emap, 1e-7).unwrap();
    let atm_h1 = Atom::from_xyz("H  0.5905546 -1.0228705 0.0000000", &emap, 1e-7).unwrap();
    let atm_h2 = Atom::from_xyz("H -1.1811091  0.0000000 0.0000000", &emap, 1e-7).unwrap();

    let bs_p1 = BasisShell::new(1, ShellOrder::Pure(PureOrder::increasingm(1)));
    let bs_sp1 = BasisShell::new(
        1,
        ShellOrder::Spinor(SpinorOrder::increasingm(1, true, None)),
    );

    let batm_b0 = BasisAtom::new(&atm_b0, &[bs_sp1.clone(), bs_p1.clone()]);
    let batm_h0 = BasisAtom::new(&atm_h0, &[bs_sp1.clone()]);
    let batm_h1 = BasisAtom::new(&atm_h1, &[bs_sp1.clone()]);
    let batm_h2 = BasisAtom::new(&atm_h2, &[bs_sp1.clone()]);
    let bao_bh3 = BasisAngularOrder::new(&[batm_b0, batm_h0, batm_h1, batm_h2]);
    let mol_bh3 = Molecule::from_atoms(
        &[
            atm_b0.clone(),
            atm_h0.clone(),
            atm_h1.clone(),
            atm_h2.clone(),
        ],
        1e-7,
    );

    // B |1/2, 1/2⟩
    #[rustfmt::skip]
    let c_b12 = array![
        [Complex::new(0.0, 0.0)],
        [Complex::new(1.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)], // end of component 1
        [Complex::new(0.0, 0.0)],
        [Complex::new(1.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
    ];
    let occ = array![1.0];

    let det_b12 = SlaterDeterminant::<Complex<f64>, SpinOrbitCoupled>::builder()
        .coefficients(&[c_b12])
        .occupations(&[occ.clone()])
        .baos(vec![&bao_bh3, &bao_bh3])
        .mol(&mol_bh3)
        .structure_constraint(SpinOrbitCoupled::JAdapted(2))
        .complex_symmetric(false)
        .threshold(1e-14)
        .build()
        .unwrap();

    // B |1, 0⟩
    #[rustfmt::skip]
    let c_b1z = array![
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(1.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)], // end of component 1
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(1.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
    ];

    let det_b1z = SlaterDeterminant::<Complex<f64>, SpinOrbitCoupled>::builder()
        .coefficients(&[c_b1z])
        .occupations(&[occ.clone()])
        .baos(vec![&bao_bh3, &bao_bh3])
        .mol(&mol_bh3)
        .structure_constraint(SpinOrbitCoupled::JAdapted(2))
        .complex_symmetric(false)
        .threshold(1e-14)
        .build()
        .unwrap();

    // B |1, 1⟩
    #[rustfmt::skip]
    let c_b11 = array![
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(1.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)], // end of component 1
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
    ];

    let det_b11 = SlaterDeterminant::<Complex<f64>, SpinOrbitCoupled>::builder()
        .coefficients(&[c_b11])
        .occupations(&[occ.clone()])
        .baos(vec![&bao_bh3, &bao_bh3])
        .mol(&mol_bh3)
        .structure_constraint(SpinOrbitCoupled::JAdapted(2))
        .complex_symmetric(false)
        .threshold(1e-14)
        .build()
        .unwrap();

    // H |1/2, 1/2⟩
    #[rustfmt::skip]
    let c_h12 = array![
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(1.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)], // end of component 1
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(1.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
    ];

    let det_h12 = SlaterDeterminant::<Complex<f64>, SpinOrbitCoupled>::builder()
        .coefficients(&[c_h12])
        .occupations(&[occ.clone()])
        .baos(vec![&bao_bh3, &bao_bh3])
        .mol(&mol_bh3)
        .structure_constraint(SpinOrbitCoupled::JAdapted(2))
        .complex_symmetric(false)
        .threshold(1e-14)
        .build()
        .unwrap();

    let sao: Array2<f64> = Array2::eye(22);
    let sao_c = sao.mapv(C128::from);

    // ~~~~~~~~~~~~~~~~~~~~~~~~
    // u Oh (ordinary, unitary)
    // ~~~~~~~~~~~~~~~~~~~~~~~~
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol_bh3)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false).unwrap();
    let group_u_oh = UnitaryRepresentedGroup::from_molecular_symmetry(&sym, None).unwrap();
    let group_u_oh_double = group_u_oh.to_double_group().unwrap();

    let mut orbit_c_u_oh_spinspatial_b12 = SlaterDeterminantSymmetryOrbit::builder()
        .group(&group_u_oh_double)
        .origin(&det_b12)
        .integrality_threshold(1e-14)
        .linear_independence_threshold(1e-12)
        .symmetry_transformation_kind(SymmetryTransformationKind::SpinSpatial)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
        .build()
        .unwrap();
    let _ = orbit_c_u_oh_spinspatial_b12
        .calc_smat(Some(&sao_c), None, true)
        .unwrap()
        .calc_xmat(false);
    assert_eq!(
        orbit_c_u_oh_spinspatial_b12.analyse_rep().unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new("||E~|_(1)|").unwrap()
    );

    let mut orbit_c_u_oh_spinspatial_b1z = SlaterDeterminantSymmetryOrbit::builder()
        .group(&group_u_oh_double)
        .origin(&det_b1z)
        .integrality_threshold(1e-14)
        .linear_independence_threshold(1e-12)
        .symmetry_transformation_kind(SymmetryTransformationKind::SpinSpatial)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
        .build()
        .unwrap();
    let _ = orbit_c_u_oh_spinspatial_b1z
        .calc_smat(Some(&sao_c), None, true)
        .unwrap()
        .calc_xmat(false);
    assert_eq!(
        orbit_c_u_oh_spinspatial_b1z.analyse_rep().unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new("||A|_(2)^('')|").unwrap()
    );

    let mut orbit_c_u_oh_spinspatial_b11 = SlaterDeterminantSymmetryOrbit::builder()
        .group(&group_u_oh_double)
        .origin(&det_b11)
        .integrality_threshold(1e-14)
        .linear_independence_threshold(1e-12)
        .symmetry_transformation_kind(SymmetryTransformationKind::SpinSpatial)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
        .build()
        .unwrap();
    let _ = orbit_c_u_oh_spinspatial_b11
        .calc_smat(Some(&sao_c), None, true)
        .unwrap()
        .calc_xmat(false);
    assert_eq!(
        orbit_c_u_oh_spinspatial_b11.analyse_rep().unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new("||E|^(')|").unwrap()
    );

    let mut orbit_c_u_oh_spinspatial_h12 = SlaterDeterminantSymmetryOrbit::builder()
        .group(&group_u_oh_double)
        .origin(&det_h12)
        .integrality_threshold(1e-14)
        .linear_independence_threshold(1e-12)
        .symmetry_transformation_kind(SymmetryTransformationKind::SpinSpatial)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
        .build()
        .unwrap();
    let _ = orbit_c_u_oh_spinspatial_h12
        .calc_smat(Some(&sao_c), None, true)
        .unwrap()
        .calc_xmat(false);
    // (A1' ⊕ E') ⊗ E~1 = E~1 ⊕ E~2 ⊕ E~3
    assert_eq!(
        orbit_c_u_oh_spinspatial_h12.analyse_rep().unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new("||E~|_(1)| ⊕ ||E~|_(2)| ⊕ ||E~|_(3)|")
            .unwrap()
    );
}

#[test]
fn test_determinant_orbit_rep_analysis_c2_d4h_jadapted() {
    // env_logger::init();
    let emap = ElementMap::new();
    let atm_c0 = Atom::from_xyz("C 0.0 0.0 1.0", &emap, 1e-7).unwrap();
    let atm_c1 = Atom::from_xyz("C 0.0 0.0 -1.0", &emap, 1e-7).unwrap();

    let bs_sp1g = BasisShell::new(
        1,
        ShellOrder::Spinor(SpinorOrder::increasingm(1, true, None)),
    );
    let bs_sp1u = BasisShell::new(
        1,
        ShellOrder::Spinor(SpinorOrder::increasingm(1, false, None)),
    );
    let bs_sp3u = BasisShell::new(
        3,
        ShellOrder::Spinor(SpinorOrder::increasingm(3, false, None)),
    );

    let batm_c0 = BasisAtom::new(
        &atm_c0,
        &[
            bs_sp1g.clone(),
            bs_sp1g.clone(),
            bs_sp1u.clone(),
            bs_sp3u.clone(),
        ],
    );
    let batm_c1 = BasisAtom::new(
        &atm_c1,
        &[
            bs_sp1g.clone(),
            bs_sp1g.clone(),
            bs_sp1u.clone(),
            bs_sp3u.clone(),
        ],
    );
    let bao_c2 = BasisAngularOrder::new(&[batm_c0, batm_c1]);

    let bs_sp1g_sp = BasisShell::new(
        1,
        ShellOrder::Spinor(SpinorOrder::increasingm(1, true, Some(SpinorBalanceSymmetry::KineticBalance))),
    );
    let bs_sp1u_sp = BasisShell::new(
        1,
        ShellOrder::Spinor(SpinorOrder::increasingm(1, false, Some(SpinorBalanceSymmetry::KineticBalance))),
    );
    let bs_sp3u_sp = BasisShell::new(
        3,
        ShellOrder::Spinor(SpinorOrder::increasingm(3, false, Some(SpinorBalanceSymmetry::KineticBalance))),
    );

    let batm_c0_sp = BasisAtom::new(
        &atm_c0,
        &[
            bs_sp1g_sp.clone(),
            bs_sp1g_sp.clone(),
            bs_sp1u_sp.clone(),
            bs_sp3u_sp.clone(),
        ],
    );
    let batm_c1_sp = BasisAtom::new(
        &atm_c1,
        &[
            bs_sp1g_sp.clone(),
            bs_sp1g_sp.clone(),
            bs_sp1u_sp.clone(),
            bs_sp3u_sp.clone(),
        ],
    );
    let bao_c2_sp = BasisAngularOrder::new(&[batm_c0_sp, batm_c1_sp]);

    let mol_c2 = Molecule::from_atoms(&[atm_c0.clone(), atm_c1.clone()], 1e-7);

    #[rustfmt::skip]
    let c = array![
        // C0L
        [Complex::new(-0.0003780042106, 0.0),],
        [Complex::new(0.0000000000000, 0.0),],
        [Complex::new(0.0034308766785, 0.0),],
        [Complex::new(-0.0000000000000, 0.0),],
        [Complex::new(-0.0004134451084, 0.0),],
        [Complex::new(-0.0000000000000, 0.0),],
        [Complex::new(0.0000000000000, 0.0),],
        [Complex::new(0.0003433541652, 0.0),],
        [Complex::new(-0.0000000000000, 0.0),],
        [Complex::new(-0.0000000000001, 0.0),],
        // C1L
        [Complex::new(0.0003780042107, 0.0),],
        [Complex::new(-0.0000000000000, 0.0),],
        [Complex::new(-0.0034308766786, 0.0),],
        [Complex::new(0.0000000000000, 0.0),],
        [Complex::new(-0.0004134451084, 0.0),],
        [Complex::new(-0.0000000000000, 0.0),],
        [Complex::new(-0.0000000000000, 0.0),],
        [Complex::new(0.0003433541652, 0.0),],
        [Complex::new(0.0000000000000, 0.0),],
        [Complex::new(-0.0000000000001, 0.0),],
        // C0S
        [Complex::new(-1.9029618881319, 0.0),],
        [Complex::new(-0.0000000000022, 0.0),],
        [Complex::new(-194.29090211500, 0.0),],
        [Complex::new(-0.0000000000189, 0.0),],
        [Complex::new(1.3752024341039, 0.0),],
        [Complex::new(0.0000000000198, 0.0),],
        [Complex::new(-0.0000000001368, 0.0),],
        [Complex::new(-24.534398430316, 0.0),],
        [Complex::new(0.0000000000348, 0.0),],
        [Complex::new(0.0000000027981, 0.0),],
        // C1S
        [Complex::new(1.9029618882276, 0.0),],
        [Complex::new(-0.0000000000029, 0.0),],
        [Complex::new(194.29090212711, 0.0),],
        [Complex::new(-0.0000000002778, 0.0),],
        [Complex::new(1.3752024339343, 0.0),],
        [Complex::new(0.0000000000214, 0.0),],
        [Complex::new(0.0000000000951, 0.0),],
        [Complex::new(-24.534398431459, 0.0),],
        [Complex::new(-0.0000000000158, 0.0),],
        [Complex::new(0.0000000018739, 0.0),],
    ];
    let occ = array![1.0];

    let det = SlaterDeterminant::<Complex<f64>, SpinOrbitCoupled>::builder()
        .coefficients(&[c])
        .occupations(&[occ])
        .baos(vec![&bao_c2, &bao_c2_sp])
        .mol(&mol_c2)
        .structure_constraint(SpinOrbitCoupled::JAdapted(2))
        .complex_symmetric(false)
        .threshold(1e-14)
        .build()
        .unwrap();

    #[rustfmt::skip]
    let sao = array![
        [1.0000000e+00, 0.0, 2.4836239e-01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.2692524e-08, 0.0, 2.3299101e-02, 0.0, 2.2758673e-02, 0.0, 0.0, 3.2185623e-02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0000000e+00, 0.0, 2.4836239e-01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.2692524e-08, 0.0, 2.3299101e-02, 0.0, -2.2758673e-02, 0.0, 0.0, 3.2185623e-02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [2.4836239e-01, 0.0, 1.0000000e+00, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.3299101e-02, 0.0, 2.7604505e-01, 0.0, 1.8316823e-01, 0.0, 0.0, 2.5903900e-01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 2.4836239e-01, 0.0, 1.0000000e+00, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.3299101e-02, 0.0, 2.7604505e-01, 0.0, -1.8316823e-01, 0.0, 0.0, 2.5903900e-01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0000000e+00, 0.0, 0.0, 0.0, 0.0, 0.0, -2.2758673e-02, 0.0, -1.8316823e-01, 0.0, -2.6364115e-03, 0.0, 0.0, -2.1418046e-01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0000000e+00, 0.0, 0.0, 0.0, 0.0, 0.0, 2.2758673e-02, 0.0, 1.8316823e-01, 0.0, -2.6364115e-03, 0.0, 0.0, 2.1418046e-01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0000000e+00, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.4881204e-01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0000000e+00, 0.0, 0.0, -3.2185623e-02, 0.0, -2.5903900e-01, 0.0, -2.1418046e-01, 0.0, 0.0, -1.5408486e-01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0000000e+00, 0.0, 0.0, -3.2185623e-02, 0.0, -2.5903900e-01, 0.0, 2.1418046e-01, 0.0, 0.0, -1.5408486e-01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0000000e+00, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.4881204e-01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [2.2692524e-08, 0.0, 2.3299101e-02, 0.0, -2.2758673e-02, 0.0, 0.0, -3.2185623e-02, 0.0, 0.0, 1.0000000e+00, 0.0, 2.4836239e-01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 2.2692524e-08, 0.0, 2.3299101e-02, 0.0, 2.2758673e-02, 0.0, 0.0, -3.2185623e-02, 0.0, 0.0, 1.0000000e+00, 0.0, 2.4836239e-01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [2.3299101e-02, 0.0, 2.7604505e-01, 0.0, -1.8316823e-01, 0.0, 0.0, -2.5903900e-01, 0.0, 0.0, 2.4836239e-01, 0.0, 1.0000000e+00, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 2.3299101e-02, 0.0, 2.7604505e-01, 0.0, 1.8316823e-01, 0.0, 0.0, -2.5903900e-01, 0.0, 0.0, 2.4836239e-01, 0.0, 1.0000000e+00, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [2.2758673e-02, 0.0, 1.8316823e-01, 0.0, -2.6364115e-03, 0.0, 0.0, -2.1418046e-01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0000000e+00, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, -2.2758673e-02, 0.0, -1.8316823e-01, 0.0, -2.6364115e-03, 0.0, 0.0, 2.1418046e-01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0000000e+00, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.4881204e-01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0000000e+00, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [3.2185623e-02, 0.0, 2.5903900e-01, 0.0, -2.1418046e-01, 0.0, 0.0, -1.5408486e-01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0000000e+00, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 3.2185623e-02, 0.0, 2.5903900e-01, 0.0, 2.1418046e-01, 0.0, 0.0, -1.5408486e-01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0000000e+00, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.4881204e-01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0000000e+00, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.2311187e-04, 0.0, -2.2868798e-06, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -3.0897868e-11, 0.0, -2.0603154e-07, 0.0, -1.5028759e-07, 0.0, 0.0, -2.1253875e-07, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.2311187e-04, 0.0, -2.2868798e-06, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -3.0897868e-11, 0.0, -2.0603154e-07, 0.0, 1.5028759e-07, 0.0, 0.0, -2.1253875e-07, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.2868798e-06, 0.0, 1.2573975e-05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.0603154e-07, 0.0, 3.6787223e-07, 0.0, 1.3313449e-06, 0.0, 0.0, 1.8828060e-06, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.2868798e-06, 0.0, 1.2573975e-05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.0603154e-07, 0.0, 3.6787223e-07, 0.0, -1.3313449e-06, 0.0, 0.0, 1.8828060e-06, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.9345511e-05, 0.0, 0.0, -2.9560440e-21, 0.0, 0.0, 1.5028759e-07, 0.0, -1.3313449e-06, 0.0, -1.1544867e-06, 0.0, 0.0, -2.9314108e-06, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.9345511e-05, 0.0, 0.0, 2.9560440e-21, 0.0, 0.0, -1.5028759e-07, 0.0, 1.3313449e-06, 0.0, -1.1544867e-06, 0.0, 0.0, 2.9314108e-06, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.9345511e-05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 9.1833376e-07, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.9560440e-21, 0.0, 0.0, 3.9345511e-05, 0.0, 0.0, 2.1253875e-07, 0.0, -1.8828060e-06, 0.0, -2.9314108e-06, 0.0, 0.0, -3.2273072e-06, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.9560440e-21, 0.0, 0.0, 3.9345511e-05, 0.0, 0.0, 2.1253875e-07, 0.0, -1.8828060e-06, 0.0, 2.9314108e-06, 0.0, 0.0, -3.2273072e-06, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.9345511e-05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 9.1833376e-07],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -3.0897868e-11, 0.0, -2.0603154e-07, 0.0, 1.5028759e-07, 0.0, 0.0, 2.1253875e-07, 0.0, 0.0, 4.2311187e-04, 0.0, -2.2868798e-06, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -3.0897868e-11, 0.0, -2.0603154e-07, 0.0, -1.5028759e-07, 0.0, 0.0, 2.1253875e-07, 0.0, 0.0, 4.2311187e-04, 0.0, -2.2868798e-06, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.0603154e-07, 0.0, 3.6787223e-07, 0.0, -1.3313449e-06, 0.0, 0.0, -1.8828060e-06, 0.0, 0.0, -2.2868798e-06, 0.0, 1.2573975e-05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.0603154e-07, 0.0, 3.6787223e-07, 0.0, 1.3313449e-06, 0.0, 0.0, -1.8828060e-06, 0.0, 0.0, -2.2868798e-06, 0.0, 1.2573975e-05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.5028759e-07, 0.0, 1.3313449e-06, 0.0, -1.1544867e-06, 0.0, 0.0, -2.9314108e-06, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.9345511e-05, 0.0, 0.0, -2.9560440e-21, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.5028759e-07, 0.0, -1.3313449e-06, 0.0, -1.1544867e-06, 0.0, 0.0, 2.9314108e-06, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.9345511e-05, 0.0, 0.0, 2.9560440e-21, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 9.1833376e-07, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.9345511e-05, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.1253875e-07, 0.0, 1.8828060e-06, 0.0, -2.9314108e-06, 0.0, 0.0, -3.2273072e-06, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.9560440e-21, 0.0, 0.0, 3.9345511e-05, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.1253875e-07, 0.0, 1.8828060e-06, 0.0, 2.9314108e-06, 0.0, 0.0, -3.2273072e-06, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.9560440e-21, 0.0, 0.0, 3.9345511e-05, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 9.1833376e-07, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.9345511e-05],
    ];
    let sao_c = sao.mapv(C128::from);

    // ~~~~~~~~~~~~~~~~~~~~~~~~
    // u D4h* (double, unitary)
    // ~~~~~~~~~~~~~~~~~~~~~~~~
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol_c2)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false).unwrap();
    let group_u_d4h = UnitaryRepresentedGroup::from_molecular_symmetry(&sym, Some(4)).unwrap();
    let group_u_d4h_double = group_u_d4h.to_double_group().unwrap();

    let mut orbit_c_u_d4h_double_spinspatial = SlaterDeterminantSymmetryOrbit::builder()
        .group(&group_u_d4h_double)
        .origin(&det)
        .integrality_threshold(1e-14)
        .linear_independence_threshold(1e-12)
        .symmetry_transformation_kind(SymmetryTransformationKind::SpinSpatial)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
        .build()
        .unwrap();
    let _ = orbit_c_u_d4h_double_spinspatial
        .calc_smat(Some(&sao_c), None, true)
        .unwrap()
        .normalise_smat()
        .unwrap()
        .calc_xmat(false);
    assert_eq!(
        orbit_c_u_d4h_double_spinspatial.analyse_rep().unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new("||E~|_(1u)|").unwrap()
    );
}
