use std::str::FromStr;

use log4rs;
use nalgebra::Vector3;
use ndarray::{Array2, array};
use ndarray_linalg::assert::close_l2;
use num_complex::Complex;

use crate::analysis::{EigenvalueComparisonMode, RepAnalysis};
use crate::angmom::spinor_rotation_3d::SpinOrbitCoupled;
use crate::auxiliary::atom::{Atom, ElementMap};
use crate::auxiliary::molecule::Molecule;
use crate::basis::ao::{
    BasisAngularOrder, BasisAtom, BasisShell, PureOrder, ShellOrder, SpinorBalanceSymmetry,
    SpinorOrder, SpinorParticleType,
};
use crate::chartab::chartab_symbols::DecomposedSymbol;
use crate::group::{GroupProperties, MagneticRepresentedGroup, UnitaryRepresentedGroup};
use crate::projection::Projectable;
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
use crate::target::noci::multideterminant::multideterminant_analysis::MultiDeterminantSymmetryOrbit;

type C128 = Complex<f64>;

#[test]
fn test_determinant_transformation_h_jadapted_twoj_1() {
    let emap = ElementMap::new();
    let atm_h0 = Atom::from_xyz("H 0.0 0.0 0.0", &emap, 1e-7).unwrap();

    let bs_sp1half = BasisShell::new(
        1,
        ShellOrder::Spinor(SpinorOrder::increasingm(
            1,
            true,
            SpinorParticleType::Fermion(None),
        )),
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
        ShellOrder::Spinor(SpinorOrder::increasingm(
            3,
            true,
            SpinorParticleType::Fermion(None),
        )),
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
        ShellOrder::Spinor(SpinorOrder::increasingm(
            1,
            true,
            SpinorParticleType::Fermion(None),
        )),
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
        .baos(vec![&bao_bf4])
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
fn test_determinant_transformation_h_jadapted_4c_sto3g() {
    // log4rs::init_file("log4rs.yml", Default::default()).unwrap();
    // ~~~~~~~~~
    // Integrals
    // ~~~~~~~~~
    #[rustfmt::skip]
    let sao = array![
        [1.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00],
        [0.0000000000000e+00, 1.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00],
        [0.0000000000000e+00, 0.0000000000000e+00, 2.0236363463312e-05, 0.0000000000000e+00],
        [0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 2.0236363463312e-05],
    ];
    let sao_c = sao.mapv(Complex::<f64>::from);

    // ~~~~~~~~
    // Geometry
    // ~~~~~~~~
    let emap = ElementMap::new();
    let atm_h0 = Atom::from_xyz("H 0.0 0.0 0.0", &emap, 1e-7).unwrap();

    let bs_sp1 = BasisShell::new(
        1,
        ShellOrder::Spinor(SpinorOrder::increasingm(
            1,
            true,
            SpinorParticleType::Fermion(None),
        )),
    );

    let batm_h0 = BasisAtom::new(&atm_h0, &[bs_sp1]);
    let bao_h = BasisAngularOrder::new(&[batm_h0]);

    let bs_sp1_sp = BasisShell::new(
        1,
        ShellOrder::Spinor(SpinorOrder::increasingm(
            1,
            true,
            SpinorParticleType::Antifermion(Some(SpinorBalanceSymmetry::KineticBalance)),
        )),
    );

    let batm_h0_sp = BasisAtom::new(&atm_h0, &[bs_sp1_sp]);
    let bao_h_sp = BasisAngularOrder::new(&[batm_h0_sp]);

    let mol_h = Molecule::from_atoms(&[atm_h0.clone()], 1e-7);

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

    // ~~~~~~~~~~
    // L|s1/2,1/2
    // ~~~~~~~~~~
    #[rustfmt::skip]
    let c_as_l = array![
        // CL
        [Complex::new(0.0, 0.0)],
        [Complex::new(1.0, 0.0)],
        // CS
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
    ];
    let occ = array![1.0];

    let det = SlaterDeterminant::<Complex<f64>, SpinOrbitCoupled>::builder()
        .coefficients(&[c_as_l])
        .occupations(&[occ.clone()])
        .baos(vec![&bao_h, &bao_h_sp])
        .mol(&mol_h)
        .structure_constraint(SpinOrbitCoupled::JAdapted(2))
        .complex_symmetric(false)
        .threshold(1e-14)
        .build()
        .unwrap();

    let mut orbit_c_u_oh_double_spinspatial = SlaterDeterminantSymmetryOrbit::builder()
        .group(&group_u_oh_double)
        .origin(&det)
        .integrality_threshold(1e-7)
        .linear_independence_threshold(1e-7)
        .symmetry_transformation_kind(SymmetryTransformationKind::SpinSpatial)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
        .build()
        .unwrap();
    let _ = orbit_c_u_oh_double_spinspatial
        .calc_smat(Some(&sao_c), None, true)
        .unwrap()
        .normalise_smat()
        .unwrap()
        .calc_xmat(false);
    assert_eq!(
        orbit_c_u_oh_double_spinspatial.analyse_rep().unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new("||E~|_(1g)|").unwrap()
    );

    // ~~~~~~~~~~~~~~~
    // S|σ·p(s1/2,1/2)
    // ~~~~~~~~~~~~~~~
    #[rustfmt::skip]
    let c_as_s = array![
        // CL
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        // CS
        [Complex::new(0.0, 0.0)],
        [Complex::new(1.0, 0.0)],
    ];
    let occ = array![1.0];

    let det = SlaterDeterminant::<Complex<f64>, SpinOrbitCoupled>::builder()
        .coefficients(&[c_as_s])
        .occupations(&[occ.clone()])
        .baos(vec![&bao_h, &bao_h_sp])
        .mol(&mol_h)
        .structure_constraint(SpinOrbitCoupled::JAdapted(2))
        .complex_symmetric(false)
        .threshold(1e-14)
        .build()
        .unwrap();

    let mut orbit_c_u_oh_double_spinspatial = SlaterDeterminantSymmetryOrbit::builder()
        .group(&group_u_oh_double)
        .origin(&det)
        .integrality_threshold(1e-7)
        .linear_independence_threshold(1e-7)
        .symmetry_transformation_kind(SymmetryTransformationKind::SpinSpatial)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
        .build()
        .unwrap();
    let _ = orbit_c_u_oh_double_spinspatial
        .calc_smat(Some(&sao_c), None, true)
        .unwrap()
        .normalise_smat()
        .unwrap()
        .calc_xmat(false);
    assert_eq!(
        orbit_c_u_oh_double_spinspatial.analyse_rep().unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new("||E~|_(1g)|").unwrap()
    );
}

#[test]
fn test_determinant_transformation_h_jadapted_4c_sto3g_antifermion() {
    // log4rs::init_file("log4rs.yml", Default::default()).unwrap();
    // ~~~~~~~~~
    // Integrals
    // ~~~~~~~~~
    #[rustfmt::skip]
    let sao = array![
        [1.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00],
        [0.0000000000000e+00, 1.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00],
        [0.0000000000000e+00, 0.0000000000000e+00, 2.0236363463312e-05, 0.0000000000000e+00],
        [0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 2.0236363463312e-05],
    ];
    let sao_c = sao.mapv(Complex::<f64>::from);

    // ~~~~~~~~
    // Geometry
    // ~~~~~~~~
    let emap = ElementMap::new();
    let atm_h0 = Atom::from_xyz("H 0.0 0.0 0.0", &emap, 1e-7).unwrap();

    let bs_sp1 = BasisShell::new(
        1,
        ShellOrder::Spinor(SpinorOrder::increasingm(
            1,
            true,
            SpinorParticleType::Antifermion(None),
        )),
    );

    let batm_h0 = BasisAtom::new(&atm_h0, &[bs_sp1]);
    let bao_h = BasisAngularOrder::new(&[batm_h0]);

    let bs_sp1_sp = BasisShell::new(
        1,
        ShellOrder::Spinor(SpinorOrder::increasingm(
            1,
            true,
            SpinorParticleType::Fermion(Some(SpinorBalanceSymmetry::KineticBalance)),
        )),
    );

    let batm_h0_sp = BasisAtom::new(&atm_h0, &[bs_sp1_sp]);
    let bao_h_sp = BasisAngularOrder::new(&[batm_h0_sp]);

    let mol_h = Molecule::from_atoms(&[atm_h0.clone()], 1e-7);

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

    // ~~~~~~~~~~
    // L|s1/2,1/2
    // ~~~~~~~~~~
    #[rustfmt::skip]
    let c_as_l = array![
        // CL
        [Complex::new(0.0, 0.0)],
        [Complex::new(1.0, 0.0)],
        // CS
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
    ];
    let occ = array![1.0];

    let det = SlaterDeterminant::<Complex<f64>, SpinOrbitCoupled>::builder()
        .coefficients(&[c_as_l])
        .occupations(&[occ.clone()])
        .baos(vec![&bao_h, &bao_h_sp])
        .mol(&mol_h)
        .structure_constraint(SpinOrbitCoupled::JAdapted(2))
        .complex_symmetric(false)
        .threshold(1e-14)
        .build()
        .unwrap();

    let mut orbit_c_u_oh_double_spinspatial = SlaterDeterminantSymmetryOrbit::builder()
        .group(&group_u_oh_double)
        .origin(&det)
        .integrality_threshold(1e-7)
        .linear_independence_threshold(1e-7)
        .symmetry_transformation_kind(SymmetryTransformationKind::SpinSpatial)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
        .build()
        .unwrap();
    let _ = orbit_c_u_oh_double_spinspatial
        .calc_smat(Some(&sao_c), None, true)
        .unwrap()
        .normalise_smat()
        .unwrap()
        .calc_xmat(false);
    assert_eq!(
        orbit_c_u_oh_double_spinspatial.analyse_rep().unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new("||E~|_(1u)|").unwrap()
    );

    // ~~~~~~~~~~~~~~~
    // S|σ·p(s1/2,1/2)
    // ~~~~~~~~~~~~~~~
    #[rustfmt::skip]
    let c_as_s = array![
        // CL
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        // CS
        [Complex::new(0.0, 0.0)],
        [Complex::new(1.0, 0.0)],
    ];
    let occ = array![1.0];

    let det = SlaterDeterminant::<Complex<f64>, SpinOrbitCoupled>::builder()
        .coefficients(&[c_as_s])
        .occupations(&[occ.clone()])
        .baos(vec![&bao_h, &bao_h_sp])
        .mol(&mol_h)
        .structure_constraint(SpinOrbitCoupled::JAdapted(2))
        .complex_symmetric(false)
        .threshold(1e-14)
        .build()
        .unwrap();

    let mut orbit_c_u_oh_double_spinspatial = SlaterDeterminantSymmetryOrbit::builder()
        .group(&group_u_oh_double)
        .origin(&det)
        .integrality_threshold(1e-7)
        .linear_independence_threshold(1e-7)
        .symmetry_transformation_kind(SymmetryTransformationKind::SpinSpatial)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
        .build()
        .unwrap();
    let _ = orbit_c_u_oh_double_spinspatial
        .calc_smat(Some(&sao_c), None, true)
        .unwrap()
        .normalise_smat()
        .unwrap()
        .calc_xmat(false);
    assert_eq!(
        orbit_c_u_oh_double_spinspatial.analyse_rep().unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new("||E~|_(1u)|").unwrap()
    );
}

#[test]
fn test_determinant_transformation_h_jadapted_4c_631gds() {
    // ~~~~~~~~~
    // Integrals
    // ~~~~~~~~~
    #[rustfmt::skip]
    let sao = array![
        [1.0000000000000e+00, 0.0000000000000e+00, 6.5829204933933e-01, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00],
        [0.0000000000000e+00, 1.0000000000000e+00, 0.0000000000000e+00, 6.5829204933933e-01, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00],
        [6.5829204933933e-01, 0.0000000000000e+00, 1.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00],
        [0.0000000000000e+00, 6.5829204933933e-01, 0.0000000000000e+00, 1.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00],
        [0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 1.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00],
        [0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 1.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00],
        [0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 1.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00],
        [0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 1.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00],
        [0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 1.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00],
        [0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 1.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00],
        [0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 3.7160879153747e-05, 0.0000000000000e+00, 6.9156224256527e-06, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00],
        [0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 3.7160879153747e-05, 0.0000000000000e+00, 6.9156224256527e-06, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00],
        [0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 6.9156224256527e-06, 0.0000000000000e+00, 6.4411959220339e-06, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00],
        [0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 6.9156224256527e-06, 0.0000000000000e+00, 6.4411959220339e-06, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00],
        [0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 7.3220611828754e-05, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00],
        [0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 7.3220611828754e-05, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00],
        [0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 7.3220611828754e-05, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00],
        [0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 7.3220611828754e-05, 0.0000000000000e+00, 0.0000000000000e+00],
        [0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 7.3220611828754e-05, 0.0000000000000e+00],
        [0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 7.3220611828754e-05],
    ];
    let sao_c = sao.mapv(Complex::<f64>::from);

    // ~~~~~~~~
    // Geometry
    // ~~~~~~~~
    let emap = ElementMap::new();
    let atm_h0 = Atom::from_xyz("H 0.0 0.0 0.0", &emap, 1e-7).unwrap();

    // s1/2
    let bs_sp_s1 = BasisShell::new(
        1,
        ShellOrder::Spinor(SpinorOrder::increasingm(
            1,
            true,
            SpinorParticleType::Fermion(None),
        )),
    );
    // p1/2
    let bs_sp_p1 = BasisShell::new(
        1,
        ShellOrder::Spinor(SpinorOrder::increasingm(
            1,
            false,
            SpinorParticleType::Fermion(None),
        )),
    );
    // p3/2
    let bs_sp_p3 = BasisShell::new(
        3,
        ShellOrder::Spinor(SpinorOrder::increasingm(
            3,
            false,
            SpinorParticleType::Fermion(None),
        )),
    );

    let batm_h0 = BasisAtom::new(&atm_h0, &[bs_sp_s1.clone(), bs_sp_s1, bs_sp_p1, bs_sp_p3]);
    let bao_h = BasisAngularOrder::new(&[batm_h0]);

    let bs_sp_s1_sp = BasisShell::new(
        1,
        ShellOrder::Spinor(SpinorOrder::increasingm(
            1,
            true,
            SpinorParticleType::Antifermion(Some(SpinorBalanceSymmetry::KineticBalance)),
        )),
    );
    let bs_sp_p1_sp = BasisShell::new(
        1,
        ShellOrder::Spinor(SpinorOrder::increasingm(
            1,
            false,
            SpinorParticleType::Antifermion(Some(SpinorBalanceSymmetry::KineticBalance)),
        )),
    );
    let bs_sp_p3_sp = BasisShell::new(
        3,
        ShellOrder::Spinor(SpinorOrder::increasingm(
            3,
            false,
            SpinorParticleType::Antifermion(Some(SpinorBalanceSymmetry::KineticBalance)),
        )),
    );

    let batm_h0_sp = BasisAtom::new(
        &atm_h0,
        &[bs_sp_s1_sp.clone(), bs_sp_s1_sp, bs_sp_p1_sp, bs_sp_p3_sp],
    );
    let bao_h_sp = BasisAngularOrder::new(&[batm_h0_sp]);

    let mol_h = Molecule::from_atoms(&[atm_h0.clone()], 1e-7);

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

    // ~~~~~~~~~~
    // L|s1/2,1/2
    // ~~~~~~~~~~
    #[rustfmt::skip]
    let c_as_l = array![
        // CL
        // s1/2
        [Complex::new(0.0, 0.0)],
        [Complex::new(1.0, 0.0)],
        // s1/2
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        // p1/2
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        // p3/2
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        // CS
        // σ·p(s1/2)
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        // σ·p(s1/2)
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        // σ·p(p1/2)
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        // σ·p(p3/2)
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
    ];
    let occ = array![1.0];

    let det = SlaterDeterminant::<Complex<f64>, SpinOrbitCoupled>::builder()
        .coefficients(&[c_as_l])
        .occupations(&[occ.clone()])
        .baos(vec![&bao_h, &bao_h_sp])
        .mol(&mol_h)
        .structure_constraint(SpinOrbitCoupled::JAdapted(2))
        .complex_symmetric(false)
        .threshold(1e-14)
        .build()
        .unwrap();

    let mut orbit_c_u_oh_double_spinspatial = SlaterDeterminantSymmetryOrbit::builder()
        .group(&group_u_oh_double)
        .origin(&det)
        .integrality_threshold(1e-7)
        .linear_independence_threshold(1e-7)
        .symmetry_transformation_kind(SymmetryTransformationKind::SpinSpatial)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
        .build()
        .unwrap();
    let _ = orbit_c_u_oh_double_spinspatial
        .calc_smat(Some(&sao_c), None, true)
        .unwrap()
        .normalise_smat()
        .unwrap()
        .calc_xmat(false);
    assert_eq!(
        orbit_c_u_oh_double_spinspatial.analyse_rep().unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new("||E~|_(1g)|").unwrap()
    );

    // ~~~~~~~~~~~~~~~
    // S|σ·p(s1/2,1/2)
    // ~~~~~~~~~~~~~~~
    #[rustfmt::skip]
    let c_as_s = array![
        // CL
        // s1/2
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        // s1/2
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        // p1/2
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        // p3/2
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        // CS
        // σ·p(s1/2)
        [Complex::new(0.0, 0.0)],
        [Complex::new(1.0, 0.0)],
        // σ·p(s1/2)
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        // σ·p(p1/2)
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        // σ·p(p3/2)
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
    ];
    let occ = array![1.0];

    let det = SlaterDeterminant::<Complex<f64>, SpinOrbitCoupled>::builder()
        .coefficients(&[c_as_s])
        .occupations(&[occ.clone()])
        .baos(vec![&bao_h, &bao_h_sp])
        .mol(&mol_h)
        .structure_constraint(SpinOrbitCoupled::JAdapted(2))
        .complex_symmetric(false)
        .threshold(1e-14)
        .build()
        .unwrap();

    let mut orbit_c_u_oh_double_spinspatial = SlaterDeterminantSymmetryOrbit::builder()
        .group(&group_u_oh_double)
        .origin(&det)
        .integrality_threshold(1e-7)
        .linear_independence_threshold(1e-7)
        .symmetry_transformation_kind(SymmetryTransformationKind::SpinSpatial)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
        .build()
        .unwrap();
    let _ = orbit_c_u_oh_double_spinspatial
        .calc_smat(Some(&sao_c), None, true)
        .unwrap()
        .normalise_smat()
        .unwrap()
        .calc_xmat(false);
    assert_eq!(
        orbit_c_u_oh_double_spinspatial.analyse_rep().unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new("||E~|_(1g)|").unwrap()
    );

    // ~~~~~~~~~~
    // L|p1/2,1/2
    // ~~~~~~~~~~
    #[rustfmt::skip]
    let c_as_l = array![
        // CL
        // s1/2
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        // s1/2
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        // p1/2
        [Complex::new(0.0, 0.0)],
        [Complex::new(1.0, 0.0)],
        // p3/2
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        // CS
        // σ·p(s1/2)
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        // σ·p(s1/2)
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        // σ·p(p1/2)
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        // σ·p(p3/2)
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
    ];
    let occ = array![1.0];

    let det = SlaterDeterminant::<Complex<f64>, SpinOrbitCoupled>::builder()
        .coefficients(&[c_as_l])
        .occupations(&[occ.clone()])
        .baos(vec![&bao_h, &bao_h_sp])
        .mol(&mol_h)
        .structure_constraint(SpinOrbitCoupled::JAdapted(2))
        .complex_symmetric(false)
        .threshold(1e-14)
        .build()
        .unwrap();

    let mut orbit_c_u_oh_double_spinspatial = SlaterDeterminantSymmetryOrbit::builder()
        .group(&group_u_oh_double)
        .origin(&det)
        .integrality_threshold(1e-7)
        .linear_independence_threshold(1e-7)
        .symmetry_transformation_kind(SymmetryTransformationKind::SpinSpatial)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
        .build()
        .unwrap();
    let _ = orbit_c_u_oh_double_spinspatial
        .calc_smat(Some(&sao_c), None, true)
        .unwrap()
        .normalise_smat()
        .unwrap()
        .calc_xmat(false);
    assert_eq!(
        orbit_c_u_oh_double_spinspatial.analyse_rep().unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new("||E~|_(1u)|").unwrap()
    );

    // ~~~~~~~~~~~~~~~
    // S|σ·p(p1/2,1/2)
    // ~~~~~~~~~~~~~~~
    #[rustfmt::skip]
    let c_as_s = array![
        // CL
        // s1/2
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        // s1/2
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        // p1/2
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        // p3/2
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        // CS
        // σ·p(s1/2)
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        // σ·p(s1/2)
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        // σ·p(p1/2)
        [Complex::new(0.0, 0.0)],
        [Complex::new(1.0, 0.0)],
        // σ·p(p3/2)
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
    ];
    let occ = array![1.0];

    let det = SlaterDeterminant::<Complex<f64>, SpinOrbitCoupled>::builder()
        .coefficients(&[c_as_s])
        .occupations(&[occ.clone()])
        .baos(vec![&bao_h, &bao_h_sp])
        .mol(&mol_h)
        .structure_constraint(SpinOrbitCoupled::JAdapted(2))
        .complex_symmetric(false)
        .threshold(1e-14)
        .build()
        .unwrap();

    let mut orbit_c_u_oh_double_spinspatial = SlaterDeterminantSymmetryOrbit::builder()
        .group(&group_u_oh_double)
        .origin(&det)
        .integrality_threshold(1e-7)
        .linear_independence_threshold(1e-7)
        .symmetry_transformation_kind(SymmetryTransformationKind::SpinSpatial)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
        .build()
        .unwrap();
    let _ = orbit_c_u_oh_double_spinspatial
        .calc_smat(Some(&sao_c), None, true)
        .unwrap()
        .normalise_smat()
        .unwrap()
        .calc_xmat(false);
    assert_eq!(
        orbit_c_u_oh_double_spinspatial.analyse_rep().unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new("||E~|_(1u)|").unwrap()
    );

    // ~~~~~~~~~~
    // L|p3/2,1/2
    // ~~~~~~~~~~
    #[rustfmt::skip]
    let c_as_l = array![
        // CL
        // s1/2
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        // s1/2
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        // p1/2
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        // p3/2
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(1.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        // CS
        // σ·p(s1/2)
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        // σ·p(s1/2)
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        // σ·p(p1/2)
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        // σ·p(p3/2)
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
    ];
    let occ = array![1.0];

    let det = SlaterDeterminant::<Complex<f64>, SpinOrbitCoupled>::builder()
        .coefficients(&[c_as_l])
        .occupations(&[occ.clone()])
        .baos(vec![&bao_h, &bao_h_sp])
        .mol(&mol_h)
        .structure_constraint(SpinOrbitCoupled::JAdapted(2))
        .complex_symmetric(false)
        .threshold(1e-14)
        .build()
        .unwrap();

    let mut orbit_c_u_oh_double_spinspatial = SlaterDeterminantSymmetryOrbit::builder()
        .group(&group_u_oh_double)
        .origin(&det)
        .integrality_threshold(1e-7)
        .linear_independence_threshold(1e-7)
        .symmetry_transformation_kind(SymmetryTransformationKind::SpinSpatial)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
        .build()
        .unwrap();
    let _ = orbit_c_u_oh_double_spinspatial
        .calc_smat(Some(&sao_c), None, true)
        .unwrap()
        .normalise_smat()
        .unwrap()
        .calc_xmat(false);
    assert_eq!(
        orbit_c_u_oh_double_spinspatial.analyse_rep().unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new("||F~|_(u)|").unwrap()
    );

    // ~~~~~~~~~~~~~~~~
    // S|σ·p(p3/2,-1/2)
    // ~~~~~~~~~~~~~~~~
    #[rustfmt::skip]
    let c_as_s = array![
        // CL
        // s1/2
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        // s1/2
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        // p1/2
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        // p3/2
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        // CS
        // σ·p(s1/2)
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        // σ·p(s1/2)
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        // σ·p(p1/2)
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        // σ·p(p3/2)
        [Complex::new(0.0, 0.0)],
        [Complex::new(1.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
    ];
    let occ = array![1.0];

    let det = SlaterDeterminant::<Complex<f64>, SpinOrbitCoupled>::builder()
        .coefficients(&[c_as_s])
        .occupations(&[occ.clone()])
        .baos(vec![&bao_h, &bao_h_sp])
        .mol(&mol_h)
        .structure_constraint(SpinOrbitCoupled::JAdapted(2))
        .complex_symmetric(false)
        .threshold(1e-14)
        .build()
        .unwrap();

    let mut orbit_c_u_oh_double_spinspatial = SlaterDeterminantSymmetryOrbit::builder()
        .group(&group_u_oh_double)
        .origin(&det)
        .integrality_threshold(1e-7)
        .linear_independence_threshold(1e-7)
        .symmetry_transformation_kind(SymmetryTransformationKind::SpinSpatial)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
        .build()
        .unwrap();
    let _ = orbit_c_u_oh_double_spinspatial
        .calc_smat(Some(&sao_c), None, true)
        .unwrap()
        .normalise_smat()
        .unwrap()
        .calc_xmat(false);
    assert_eq!(
        orbit_c_u_oh_double_spinspatial.analyse_rep().unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new("||F~|_(u)|").unwrap()
    );
}

#[test]
fn test_determinant_orbit_rep_analysis_h_jadapted() {
    // env_logger::init();
    let emap = ElementMap::new();
    let atm_h0 = Atom::from_xyz("H 0.0 0.0 0.0", &emap, 1e-7).unwrap();

    let bs_sp1 = BasisShell::new(
        1,
        ShellOrder::Spinor(SpinorOrder::increasingm(
            1,
            true,
            SpinorParticleType::Fermion(None),
        )),
    );
    let bs_sp3 = BasisShell::new(
        3,
        ShellOrder::Spinor(SpinorOrder::increasingm(
            3,
            true,
            SpinorParticleType::Fermion(None),
        )),
    );
    let bs_sp5 = BasisShell::new(
        5,
        ShellOrder::Spinor(SpinorOrder::increasingm(
            5,
            true,
            SpinorParticleType::Fermion(None),
        )),
    );
    let bs_sp7 = BasisShell::new(
        7,
        ShellOrder::Spinor(SpinorOrder::increasingm(
            7,
            true,
            SpinorParticleType::Fermion(None),
        )),
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
        ShellOrder::Spinor(SpinorOrder::increasingm(
            1,
            true,
            SpinorParticleType::Fermion(None),
        )),
    );
    let bs_sp3 = BasisShell::new(
        3,
        ShellOrder::Spinor(SpinorOrder::increasingm(
            3,
            true,
            SpinorParticleType::Fermion(None),
        )),
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
        ShellOrder::Spinor(SpinorOrder::increasingm(
            1,
            true,
            SpinorParticleType::Fermion(None),
        )),
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

    // ~~~~~~~~~~~~~~~~~~~~~~~~~
    // u D3h (ordinary, unitary)
    // ~~~~~~~~~~~~~~~~~~~~~~~~~
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol_bh3)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false).unwrap();
    let group_u_d3h = UnitaryRepresentedGroup::from_molecular_symmetry(&sym, None).unwrap();
    let group_u_d3h_double = group_u_d3h.to_double_group().unwrap();

    let mut orbit_c_u_oh_spinspatial_b12 = SlaterDeterminantSymmetryOrbit::builder()
        .group(&group_u_d3h_double)
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
        .group(&group_u_d3h_double)
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
        .group(&group_u_d3h_double)
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
        .group(&group_u_d3h_double)
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
    #[rustfmt::skip]
    let sao = array![
        [1.0000000000000e+00, 0.0000000000000e+00, 2.4836239031011e-01, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 2.2136461219795e-12, 0.0000000000000e+00, 7.4470666948293e-03, 0.0000000000000e+00, 8.3223475208978e-03, 0.0000000000000e+00, 0.0000000000000e+00, 1.1769576734836e-02, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00],
        [0.0000000000000e+00, 1.0000000000000e+00, 0.0000000000000e+00, 2.4836239031011e-01, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 2.2136461219795e-12, 0.0000000000000e+00, 7.4470666948293e-03, 0.0000000000000e+00, -8.3223475208978e-03, 0.0000000000000e+00, 0.0000000000000e+00, 1.1769576734836e-02, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00],
        [2.4836239031011e-01, 0.0000000000000e+00, 1.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 7.4470666948293e-03, 0.0000000000000e+00, 1.3939620582153e-01, 0.0000000000000e+00, 1.0575119562202e-01, 0.0000000000000e+00, 0.0000000000000e+00, 1.4955477508583e-01, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00],
        [0.0000000000000e+00, 2.4836239031011e-01, 0.0000000000000e+00, 1.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 7.4470666948293e-03, 0.0000000000000e+00, 1.3939620582153e-01, 0.0000000000000e+00, -1.0575119562202e-01, 0.0000000000000e+00, 0.0000000000000e+00, 1.4955477508583e-01, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00],
        [0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 1.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, -8.3223475208978e-03, 0.0000000000000e+00, -1.0575119562202e-01, 0.0000000000000e+00, -2.7273740342598e-02, 0.0000000000000e+00, 0.0000000000000e+00, -1.3085096018484e-01, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00],
        [0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 1.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 8.3223475208978e-03, 0.0000000000000e+00, 1.0575119562202e-01, 0.0000000000000e+00, -2.7273740342598e-02, 0.0000000000000e+00, 0.0000000000000e+00, 1.3085096018484e-01, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00],
        [0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 1.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 6.5251860928872e-02, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00],
        [0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 1.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, -1.1769576734836e-02, 0.0000000000000e+00, -1.4955477508583e-01, 0.0000000000000e+00, -1.3085096018484e-01, 0.0000000000000e+00, 0.0000000000000e+00, -1.1979934161407e-01, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00],
        [0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 1.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, -1.1769576734836e-02, 0.0000000000000e+00, -1.4955477508583e-01, 0.0000000000000e+00, 1.3085096018484e-01, 0.0000000000000e+00, 0.0000000000000e+00, -1.1979934161407e-01, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00],
        [0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 1.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 6.5251860928872e-02, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00],
        [2.2136461219795e-12, 0.0000000000000e+00, 7.4470666948293e-03, 0.0000000000000e+00, -8.3223475208978e-03, 0.0000000000000e+00, 0.0000000000000e+00, -1.1769576734836e-02, 0.0000000000000e+00, 0.0000000000000e+00, 1.0000000000000e+00, 0.0000000000000e+00, 2.4836239031011e-01, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00],
        [0.0000000000000e+00, 2.2136461219795e-12, 0.0000000000000e+00, 7.4470666948293e-03, 0.0000000000000e+00, 8.3223475208978e-03, 0.0000000000000e+00, 0.0000000000000e+00, -1.1769576734836e-02, 0.0000000000000e+00, 0.0000000000000e+00, 1.0000000000000e+00, 0.0000000000000e+00, 2.4836239031011e-01, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00],
        [7.4470666948293e-03, 0.0000000000000e+00, 1.3939620582153e-01, 0.0000000000000e+00, -1.0575119562202e-01, 0.0000000000000e+00, 0.0000000000000e+00, -1.4955477508583e-01, 0.0000000000000e+00, 0.0000000000000e+00, 2.4836239031011e-01, 0.0000000000000e+00, 1.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00],
        [0.0000000000000e+00, 7.4470666948293e-03, 0.0000000000000e+00, 1.3939620582153e-01, 0.0000000000000e+00, 1.0575119562202e-01, 0.0000000000000e+00, 0.0000000000000e+00, -1.4955477508583e-01, 0.0000000000000e+00, 0.0000000000000e+00, 2.4836239031011e-01, 0.0000000000000e+00, 1.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00],
        [8.3223475208978e-03, 0.0000000000000e+00, 1.0575119562202e-01, 0.0000000000000e+00, -2.7273740342598e-02, 0.0000000000000e+00, 0.0000000000000e+00, -1.3085096018484e-01, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 1.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00],
        [0.0000000000000e+00, -8.3223475208978e-03, 0.0000000000000e+00, -1.0575119562202e-01, 0.0000000000000e+00, -2.7273740342598e-02, 0.0000000000000e+00, 0.0000000000000e+00, 1.3085096018484e-01, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 1.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00],
        [0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 6.5251860928872e-02, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 1.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00],
        [1.1769576734836e-02, 0.0000000000000e+00, 1.4955477508583e-01, 0.0000000000000e+00, -1.3085096018484e-01, 0.0000000000000e+00, 0.0000000000000e+00, -1.1979934161407e-01, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 1.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00],
        [0.0000000000000e+00, 1.1769576734836e-02, 0.0000000000000e+00, 1.4955477508583e-01, 0.0000000000000e+00, 1.3085096018484e-01, 0.0000000000000e+00, 0.0000000000000e+00, -1.1979934161407e-01, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 1.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00],
        [0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 6.5251860928872e-02, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 1.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00],
        [0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 4.2311187369936e-04, 0.0000000000000e+00, -2.2868797648943e-06, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, -4.9348868463316e-15, 0.0000000000000e+00, -1.3627230007748e-07, 0.0000000000000e+00, -8.1660230128646e-08, 0.0000000000000e+00, 0.0000000000000e+00, -1.1548500495444e-07, 0.0000000000000e+00, 0.0000000000000e+00],
        [0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 4.2311187369936e-04, 0.0000000000000e+00, -2.2868797648943e-06, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, -4.9348868463316e-15, 0.0000000000000e+00, -1.3627230007748e-07, 0.0000000000000e+00, 8.1660230128646e-08, 0.0000000000000e+00, 0.0000000000000e+00, -1.1548500495444e-07, 0.0000000000000e+00],
        [0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, -2.2868797648943e-06, 0.0000000000000e+00, 1.2573975267781e-05, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, -1.3627230007748e-07, 0.0000000000000e+00, -4.0555841747839e-07, 0.0000000000000e+00, 2.1480441532004e-07, 0.0000000000000e+00, 0.0000000000000e+00, 3.0377931740323e-07, 0.0000000000000e+00, 0.0000000000000e+00],
        [0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, -2.2868797648943e-06, 0.0000000000000e+00, 1.2573975267781e-05, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, -1.3627230007748e-07, 0.0000000000000e+00, -4.0555841747839e-07, 0.0000000000000e+00, -2.1480441532004e-07, 0.0000000000000e+00, 0.0000000000000e+00, 3.0377931740323e-07, 0.0000000000000e+00],
        [0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 3.9345510535862e-05, 0.0000000000000e+00, 0.0000000000000e+00, -2.9560439683427e-21, 0.0000000000000e+00, 0.0000000000000e+00, 8.1660230128645e-08, 0.0000000000000e+00, -2.1480441532004e-07, 0.0000000000000e+00, -5.4119805424749e-07, 0.0000000000000e+00, 0.0000000000000e+00, -8.5586364015703e-07, 0.0000000000000e+00, 0.0000000000000e+00],
        [0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 3.9345510535862e-05, 0.0000000000000e+00, 0.0000000000000e+00, 2.9560439683427e-21, 0.0000000000000e+00, 0.0000000000000e+00, -8.1660230128645e-08, 0.0000000000000e+00, 2.1480441532004e-07, 0.0000000000000e+00, -5.4119805424749e-07, 0.0000000000000e+00, 0.0000000000000e+00, 8.5586364015703e-07, 0.0000000000000e+00],
        [0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 3.9345510535862e-05, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 6.3988929478544e-08, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00],
        [0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, -2.9560439683427e-21, 0.0000000000000e+00, 0.0000000000000e+00, 3.9345510535862e-05, 0.0000000000000e+00, 0.0000000000000e+00, 1.1548500495444e-07, 0.0000000000000e+00, -3.0377931740323e-07, 0.0000000000000e+00, -8.5586364015703e-07, 0.0000000000000e+00, 0.0000000000000e+00, -1.1463850379735e-06, 0.0000000000000e+00, 0.0000000000000e+00],
        [0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 2.9560439683427e-21, 0.0000000000000e+00, 0.0000000000000e+00, 3.9345510535862e-05, 0.0000000000000e+00, 0.0000000000000e+00, 1.1548500495444e-07, 0.0000000000000e+00, -3.0377931740323e-07, 0.0000000000000e+00, 8.5586364015703e-07, 0.0000000000000e+00, 0.0000000000000e+00, -1.1463850379735e-06, 0.0000000000000e+00],
        [0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 3.9345510535862e-05, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 6.3988929478544e-08],
        [0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, -4.9348868463316e-15, 0.0000000000000e+00, -1.3627230007748e-07, 0.0000000000000e+00, 8.1660230128645e-08, 0.0000000000000e+00, 0.0000000000000e+00, 1.1548500495444e-07, 0.0000000000000e+00, 0.0000000000000e+00, 4.2311187369936e-04, 0.0000000000000e+00, -2.2868797648943e-06, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00],
        [0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, -4.9348868463316e-15, 0.0000000000000e+00, -1.3627230007748e-07, 0.0000000000000e+00, -8.1660230128645e-08, 0.0000000000000e+00, 0.0000000000000e+00, 1.1548500495444e-07, 0.0000000000000e+00, 0.0000000000000e+00, 4.2311187369936e-04, 0.0000000000000e+00, -2.2868797648943e-06, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00],
        [0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, -1.3627230007748e-07, 0.0000000000000e+00, -4.0555841747839e-07, 0.0000000000000e+00, -2.1480441532004e-07, 0.0000000000000e+00, 0.0000000000000e+00, -3.0377931740323e-07, 0.0000000000000e+00, 0.0000000000000e+00, -2.2868797648943e-06, 0.0000000000000e+00, 1.2573975267781e-05, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00],
        [0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, -1.3627230007748e-07, 0.0000000000000e+00, -4.0555841747839e-07, 0.0000000000000e+00, 2.1480441532004e-07, 0.0000000000000e+00, 0.0000000000000e+00, -3.0377931740323e-07, 0.0000000000000e+00, 0.0000000000000e+00, -2.2868797648943e-06, 0.0000000000000e+00, 1.2573975267781e-05, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00],
        [0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, -8.1660230128646e-08, 0.0000000000000e+00, 2.1480441532004e-07, 0.0000000000000e+00, -5.4119805424749e-07, 0.0000000000000e+00, 0.0000000000000e+00, -8.5586364015703e-07, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 3.9345510535862e-05, 0.0000000000000e+00, 0.0000000000000e+00, -2.9560439683427e-21, 0.0000000000000e+00, 0.0000000000000e+00],
        [0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 8.1660230128646e-08, 0.0000000000000e+00, -2.1480441532004e-07, 0.0000000000000e+00, -5.4119805424749e-07, 0.0000000000000e+00, 0.0000000000000e+00, 8.5586364015703e-07, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 3.9345510535862e-05, 0.0000000000000e+00, 0.0000000000000e+00, 2.9560439683427e-21, 0.0000000000000e+00],
        [0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 6.3988929478544e-08, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 3.9345510535862e-05, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00],
        [0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, -1.1548500495444e-07, 0.0000000000000e+00, 3.0377931740323e-07, 0.0000000000000e+00, -8.5586364015703e-07, 0.0000000000000e+00, 0.0000000000000e+00, -1.1463850379735e-06, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, -2.9560439683427e-21, 0.0000000000000e+00, 0.0000000000000e+00, 3.9345510535862e-05, 0.0000000000000e+00, 0.0000000000000e+00],
        [0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, -1.1548500495444e-07, 0.0000000000000e+00, 3.0377931740323e-07, 0.0000000000000e+00, 8.5586364015703e-07, 0.0000000000000e+00, 0.0000000000000e+00, -1.1463850379735e-06, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 2.9560439683427e-21, 0.0000000000000e+00, 0.0000000000000e+00, 3.9345510535862e-05, 0.0000000000000e+00],
        [0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 6.3988929478544e-08, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 0.0000000000000e+00, 3.9345510535862e-05],
    ];
    let sao_c = sao.mapv(C128::from);

    let emap = ElementMap::new();
    let atm_c0 = Atom::from_xyz("C 0.0 0.0 1.0", &emap, 1e-7).unwrap();
    let atm_c1 = Atom::from_xyz("C 0.0 0.0 -1.0", &emap, 1e-7).unwrap();

    let bs_sp1g = BasisShell::new(
        1,
        ShellOrder::Spinor(SpinorOrder::increasingm(
            1,
            true,
            SpinorParticleType::Fermion(None),
        )),
    );
    let bs_sp1u = BasisShell::new(
        1,
        ShellOrder::Spinor(SpinorOrder::increasingm(
            1,
            false,
            SpinorParticleType::Fermion(None),
        )),
    );
    let bs_sp3u = BasisShell::new(
        3,
        ShellOrder::Spinor(SpinorOrder::increasingm(
            3,
            false,
            SpinorParticleType::Fermion(None),
        )),
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
        ShellOrder::Spinor(SpinorOrder::increasingm(
            1,
            true,
            SpinorParticleType::Antifermion(Some(SpinorBalanceSymmetry::KineticBalance)),
        )),
    );
    let bs_sp1u_sp = BasisShell::new(
        1,
        ShellOrder::Spinor(SpinorOrder::increasingm(
            1,
            false,
            SpinorParticleType::Antifermion(Some(SpinorBalanceSymmetry::KineticBalance)),
        )),
    );
    let bs_sp3u_sp = BasisShell::new(
        3,
        ShellOrder::Spinor(SpinorOrder::increasingm(
            3,
            false,
            SpinorParticleType::Antifermion(Some(SpinorBalanceSymmetry::KineticBalance)),
        )),
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

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // C0: L|s1/2,1/2, C1: L|s1/2,1/2
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #[rustfmt::skip]
    let c = array![
        // C0L
        // s1/2
        [Complex::new(0.0, 0.0)],
        [Complex::new(1.0, 0.0)],
        // s1/2
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        // p1/2
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        // p3/2
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        // C1L
        // s1/2
        [Complex::new(0.0, 0.0)],
        [Complex::new(1.0, 0.0)],
        // s1/2
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        // p1/2
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        // p3/2
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        // C0S
        // s1/2
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        // s1/2
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        // p1/2
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        // p3/2
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        // C1S
        // s1/2
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        // s1/2
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        // p1/2
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        // p3/2
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
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
    let mut orbit_c_u_d4h_double_spinspatial = SlaterDeterminantSymmetryOrbit::builder()
        .group(&group_u_d4h_double)
        .origin(&det)
        .integrality_threshold(1e-10)
        .linear_independence_threshold(1e-10)
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
        DecomposedSymbol::<MullikenIrrepSymbol>::new("||E~|_(1g)|").unwrap()
    );

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // C0: S|σ·p(s1/2,1/2), C1: S|σ·p(s1/2,1/2)
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #[rustfmt::skip]
    let c = array![
        // C0L
        // s1/2
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        // s1/2
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        // p1/2
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        // p3/2
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        // C1L
        // s1/2
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        // s1/2
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        // p1/2
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        // p3/2
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        // C0S
        // s1/2
        [Complex::new(0.0, 0.0)],
        [Complex::new(1.0, 0.0)],
        // s1/2
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        // p1/2
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        // p3/2
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        // C1S
        // s1/2
        [Complex::new(0.0, 0.0)],
        [Complex::new(1.0, 0.0)],
        // s1/2
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        // p1/2
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        // p3/2
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
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
    let mut orbit_c_u_d4h_double_spinspatial = SlaterDeterminantSymmetryOrbit::builder()
        .group(&group_u_d4h_double)
        .origin(&det)
        .integrality_threshold(1e-10)
        .linear_independence_threshold(1e-10)
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
        DecomposedSymbol::<MullikenIrrepSymbol>::new("||E~|_(1g)|").unwrap()
    );

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // C0: L|p1/2,1/2, C1: L|p1/2,1/2
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #[rustfmt::skip]
    let c = array![
        // C0L
        // s1/2
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        // s1/2
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        // p1/2
        [Complex::new(0.0, 0.0)],
        [Complex::new(1.0, 0.0)],
        // p3/2
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        // C1L
        // s1/2
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        // s1/2
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        // p1/2
        [Complex::new(0.0, 0.0)],
        [Complex::new(1.0, 0.0)],
        // p3/2
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        // C0S
        // s1/2
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        // s1/2
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        // p1/2
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        // p3/2
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        // C1S
        // s1/2
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        // s1/2
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        // p1/2
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        // p3/2
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
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
    let mut orbit_c_u_d4h_double_spinspatial = SlaterDeterminantSymmetryOrbit::builder()
        .group(&group_u_d4h_double)
        .origin(&det)
        .integrality_threshold(1e-10)
        .linear_independence_threshold(1e-10)
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

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // C0: S|σ·p(p1/2,-1/2), C1: S|σ·p(p1/2,-1/2)
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #[rustfmt::skip]
    let c = array![
        // C0L
        // s1/2
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        // s1/2
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        // p1/2
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        // p3/2
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        // C1L
        // s1/2
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        // s1/2
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        // p1/2
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        // p3/2
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        // C0S
        // s1/2
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        // s1/2
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        // p1/2
        [Complex::new(1.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        // p3/2
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        // C1S
        // s1/2
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        // s1/2
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        // p1/2
        [Complex::new(1.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        // p3/2
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
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
    let mut orbit_c_u_d4h_double_spinspatial = SlaterDeterminantSymmetryOrbit::builder()
        .group(&group_u_d4h_double)
        .origin(&det)
        .integrality_threshold(1e-10)
        .linear_independence_threshold(1e-10)
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

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // C0: L|p3/2,-3/2, C1: L|p3/2,-3/2
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #[rustfmt::skip]
    let c = array![
        // C0L
        // s1/2
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        // s1/2
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        // p1/2
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        // p3/2
        [Complex::new(1.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        // C1L
        // s1/2
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        // s1/2
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        // p1/2
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        // p3/2
        [Complex::new(1.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        // C0S
        // s1/2
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        // s1/2
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        // p1/2
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        // p3/2
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        // C1S
        // s1/2
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        // s1/2
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        // p1/2
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        // p3/2
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
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
    let mut orbit_c_u_d4h_double_spinspatial = SlaterDeterminantSymmetryOrbit::builder()
        .group(&group_u_d4h_double)
        .origin(&det)
        .integrality_threshold(1e-10)
        .linear_independence_threshold(1e-10)
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
        DecomposedSymbol::<MullikenIrrepSymbol>::new("||E~|_(2u)|").unwrap()
    );

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // C0: S|σ·p(p3/2,-3/2), C1: S|σ·p(p3/2,-3/2)
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #[rustfmt::skip]
    let c = array![
        // C0L
        // s1/2
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        // s1/2
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        // p1/2
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        // p3/2
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        // C1L
        // s1/2
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        // s1/2
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        // p1/2
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        // p3/2
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        // C0S
        // s1/2
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        // s1/2
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        // p1/2
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        // p3/2
        [Complex::new(1.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        // C1S
        // s1/2
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        // s1/2
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        // p1/2
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        // p3/2
        [Complex::new(1.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0)],
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
    let mut orbit_c_u_d4h_double_spinspatial = SlaterDeterminantSymmetryOrbit::builder()
        .group(&group_u_d4h_double)
        .origin(&det)
        .integrality_threshold(1e-10)
        .linear_independence_threshold(1e-10)
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
        DecomposedSymbol::<MullikenIrrepSymbol>::new("||E~|_(2u)|").unwrap()
    );

    // ~~~~~~~~~~~~~~~~~~~~
    // From SCF calculation
    // ~~~~~~~~~~~~~~~~~~~~
    #[rustfmt::skip]
    let c = array![
        // C0L
        // s1/2
        [Complex::new( 2.4486944178034e-14, 0.0)],
        [Complex::new(-8.7104436554765e-17, 0.0)],
        // s1/2
        [Complex::new(-3.5442841874863e-13, 0.0)],
        [Complex::new(-2.8602751944162e-15, 0.0)],
        // p1/2
        [Complex::new( 3.0478827284529e-13, 0.0)],
        [Complex::new(-3.3653955755938e-15, 0.0)],
        // p3/2
        [Complex::new(-9.7484514117187e-05, 0.0)],
        [Complex::new(-2.7627905635682e-12, 0.0)],
        [Complex::new(-4.6215691778800e-14, 0.0)],
        [Complex::new(-4.7398849476705e-03, 0.0)],
        // C1L
        // s1/2
        [Complex::new( 1.8622926835857e-14, 0.0)],
        [Complex::new( 9.3794154221104e-16, 0.0)],
        // s1/2
        [Complex::new(-3.0663932167157e-13, 0.0)],
        [Complex::new(-8.9443829511150e-15, 0.0)],
        // p1/2
        [Complex::new(-3.1660207152545e-13, 0.0)],
        [Complex::new( 7.9717316953837e-15, 0.0)],
        // p3/2
        [Complex::new( 9.7484477880200e-05, 0.0)],
        [Complex::new( 2.7637497273557e-12, 0.0)],
        [Complex::new( 4.1496952400697e-14, 0.0)],
        [Complex::new( 4.7398831849033e-03, 0.0)],
        // C0S
        // s1/2
        [Complex::new( 7.9136516860415e-11, 0.0)],
        [Complex::new( 6.4315380625317e-13, 0.0)],
        // s1/2
        [Complex::new( 2.6467071826140e-09, 0.0)],
        [Complex::new(-6.9535036259970e-11, 0.0)],
        // p1/2
        [Complex::new(-1.1877827982822e-09, 0.0)],
        [Complex::new(-1.9821294824477e-12, 0.0)],
        // p3/2
        [Complex::new( 2.3198388476928e+00, 0.0)],
        [Complex::new( 7.4293141703271e-08, 0.0)],
        [Complex::new( 1.2126857147434e-09, 0.0)],
        [Complex::new( 1.1279503554907e+02, 0.0)],
        // C1S
        // s1/2
        [Complex::new( 4.9507879377485e-11, 0.0)],
        [Complex::new( 2.7319702050367e-12, 0.0)],
        // s1/2
        [Complex::new(-4.9375241043960e-10, 0.0)],
        [Complex::new( 2.2485777795550e-10, 0.0)],
        // p1/2
        [Complex::new( 1.6012593515864e-09, 0.0)],
        [Complex::new(-8.5284748583437e-11, 0.0)],
        // p3/2
        [Complex::new(-2.3198378681613e+00, 0.0)],
        [Complex::new(-7.4186041545098e-08, 0.0)],
        [Complex::new(-1.1369675663676e-09, 0.0)],
        [Complex::new(-1.1279498789929e+02, 0.0)],
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
    let mut orbit_c_u_d4h_double_spinspatial = SlaterDeterminantSymmetryOrbit::builder()
        .group(&group_u_d4h_double)
        .origin(&det)
        .integrality_threshold(1e-10)
        .linear_independence_threshold(1e-10)
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
        DecomposedSymbol::<MullikenIrrepSymbol>::new("||E~|_(2g)|").unwrap()
    );
}

#[test]
fn test_determinant_projection_bh3_jadapted() {
    // env_logger::init();
    let emap = ElementMap::new();
    let atm_b0 = Atom::from_xyz("B 0.0 0.0 0.0", &emap, 1e-7).unwrap();
    let atm_h0 = Atom::from_xyz("H  0.5905546  1.0228705 0.0000000", &emap, 1e-7).unwrap();
    let atm_h1 = Atom::from_xyz("H  0.5905546 -1.0228705 0.0000000", &emap, 1e-7).unwrap();
    let atm_h2 = Atom::from_xyz("H -1.1811091  0.0000000 0.0000000", &emap, 1e-7).unwrap();

    let bs_p1 = BasisShell::new(1, ShellOrder::Pure(PureOrder::increasingm(1)));
    let bs_sp1 = BasisShell::new(
        1,
        ShellOrder::Spinor(SpinorOrder::increasingm(
            1,
            true,
            SpinorParticleType::Fermion(None),
        )),
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
    let occ = array![1.0];

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

    // ~~~~~~~~~~~~~~~~~~~~~~~~~
    // u D3h (ordinary, unitary)
    // ~~~~~~~~~~~~~~~~~~~~~~~~~
    let presym = PreSymmetry::builder()
        .moi_threshold(1e-7)
        .molecule(&mol_bh3)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false).unwrap();
    let group_u_d3h = UnitaryRepresentedGroup::from_molecular_symmetry(&sym, None).unwrap();
    let group_u_d3h_double = group_u_d3h.to_double_group().unwrap();

    let mut orbit_c_u_oh_spinspatial_h12 = SlaterDeterminantSymmetryOrbit::builder()
        .group(&group_u_d3h_double)
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

    for sym in ["||E~|_(1)|", "||E~|_(2)|", "||E~|_(3)|"] {
        let row = MullikenIrrepSymbol::from_str(sym).unwrap();
        let h12_p = orbit_c_u_oh_spinspatial_h12.project_onto(&row).unwrap();
        let mut orbit_h12_p = MultiDeterminantSymmetryOrbit::builder()
            .group(&group_u_d3h_double)
            .origin(&h12_p)
            .integrality_threshold(1e-7)
            .linear_independence_threshold(1e-7)
            .symmetry_transformation_kind(SymmetryTransformationKind::SpinSpatial)
            .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
            .build()
            .unwrap();
        let _ = orbit_h12_p
            .calc_smat_optimised(Some(&sao_c), None, true)
            .unwrap()
            .calc_xmat(false);
        assert_eq!(
            orbit_h12_p.analyse_rep().unwrap(),
            DecomposedSymbol::<MullikenIrrepSymbol>::new(sym).unwrap()
        );

        let h12_p_eager = h12_p.to_eager_basis().unwrap();
        let mut orbit_h12_p_eager = MultiDeterminantSymmetryOrbit::builder()
            .group(&group_u_d3h_double)
            .origin(&h12_p_eager)
            .integrality_threshold(1e-7)
            .linear_independence_threshold(1e-7)
            .symmetry_transformation_kind(SymmetryTransformationKind::SpinSpatial)
            .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
            .build()
            .unwrap();
        let _ = orbit_h12_p_eager
            .calc_smat(Some(&sao_c), None, true)
            .unwrap()
            .calc_xmat(false);
        assert_eq!(
            orbit_h12_p_eager.analyse_rep().unwrap(),
            DecomposedSymbol::<MullikenIrrepSymbol>::new(sym).unwrap()
        );
    }

    for sym in [
        "||A|_(1)^(')|",
        "||A|_(2)^(')|",
        "||E|^(')|",
        "||A|_(1)^('')|",
        "||A|_(2)^('')|",
        "||E|^('')|",
    ] {
        let row = MullikenIrrepSymbol::from_str(sym).unwrap();
        let h12_p = orbit_c_u_oh_spinspatial_h12.project_onto(&row).unwrap();
        let mut orbit_h12_p = MultiDeterminantSymmetryOrbit::builder()
            .group(&group_u_d3h_double)
            .origin(&h12_p)
            .integrality_threshold(1e-7)
            .linear_independence_threshold(1e-7)
            .symmetry_transformation_kind(SymmetryTransformationKind::SpinSpatial)
            .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
            .build()
            .unwrap();
        assert!(
            orbit_h12_p
                .calc_smat_optimised(Some(&sao_c), None, true)
                .unwrap()
                .calc_xmat(false)
                .is_err()
        );
    }
}
