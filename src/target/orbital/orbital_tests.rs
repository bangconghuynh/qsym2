// use env_logger;
use ndarray::{array, s, Array2};
use num_complex::Complex;

use crate::analysis::{EigenvalueComparisonMode, RepAnalysis};
use crate::angmom::spinor_rotation_3d::SpinConstraint;
use crate::basis::ao::{
    BasisAngularOrder, BasisAtom, BasisShell, CartOrder, PureOrder, ShellOrder,
};
use crate::auxiliary::atom::{Atom, ElementMap};
use crate::auxiliary::geometry::Transform;
use crate::auxiliary::molecule::Molecule;
use crate::chartab::chartab_symbols::DecomposedSymbol;
use crate::group::UnitaryRepresentedGroup;
use crate::symmetry::symmetry_core::{PreSymmetry, Symmetry};
use crate::symmetry::symmetry_group::SymmetryGroupProperties;
use crate::symmetry::symmetry_symbols::MullikenIrrepSymbol;
use crate::symmetry::symmetry_transformation::SymmetryTransformationKind;
use crate::target::determinant::determinant_analysis::SlaterDeterminantSymmetryOrbit;
use crate::target::determinant::SlaterDeterminant;
use crate::target::orbital::orbital_analysis::MolecularOrbitalSymmetryOrbit;

type C128 = Complex<f64>;

#[test]
fn test_orbital_orbit_rep_analysis_vf6_oct_lex_order() {
    // env_logger::init();
    let emap = ElementMap::new();
    let atm_v = Atom::from_xyz("V +0.0 +0.0 +0.0", &emap, 1e-7).unwrap();
    let atm_f0 = Atom::from_xyz("F +0.0 +0.0 +1.0", &emap, 1e-7).unwrap();
    let atm_f1 = Atom::from_xyz("F +0.0 +0.0 -1.0", &emap, 1e-7).unwrap();
    let atm_f2 = Atom::from_xyz("F +1.0 +0.0 +0.0", &emap, 1e-7).unwrap();
    let atm_f3 = Atom::from_xyz("F -1.0 +0.0 +0.0", &emap, 1e-7).unwrap();
    let atm_f4 = Atom::from_xyz("F +0.0 +1.0 +0.0", &emap, 1e-7).unwrap();
    let atm_f5 = Atom::from_xyz("F +0.0 -1.0 +0.0", &emap, 1e-7).unwrap();

    let bsc_d = BasisShell::new(2, ShellOrder::Cart(CartOrder::lex(2)));
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
    ];
    let mut sao_g = Array2::zeros((24, 24));
    sao_g.slice_mut(s![0..12, 0..12]).assign(&sao);
    sao_g.slice_mut(s![12..24, 12..24]).assign(&sao);
    let sao_cg = sao_g.mapv(C128::from);

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

    let orbs_d3_cg = det_d3_cg.to_orbitals();

    // ~~~~~~~~~~~~~~~~~~~~~~~~
    // u Oh (ordinary, unitary)
    // ~~~~~~~~~~~~~~~~~~~~~~~~

    let mut orbit_cg_u_oh_spatial_d3 = SlaterDeterminantSymmetryOrbit::builder()
        .group(&group_u_oh)
        .origin(&det_d3_cg)
        .integrality_threshold(1e-14)
        .linear_independence_threshold(1e-14)
        .symmetry_transformation_kind(SymmetryTransformationKind::Spatial)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
        .build()
        .unwrap();
    let _ = orbit_cg_u_oh_spatial_d3
        .calc_smat(Some(&sao_cg))
        .unwrap()
        .calc_xmat(false);
    assert_eq!(
        orbit_cg_u_oh_spatial_d3.analyse_rep().unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new("||T|_(1g)| ⊕ ||T|_(2g)|").unwrap()
    );

    let orbit_cg_u_oh_spatial_d3_orbs = MolecularOrbitalSymmetryOrbit::from_orbitals(
        &group_u_oh,
        &orbs_d3_cg,
        SymmetryTransformationKind::Spatial,
        EigenvalueComparisonMode::Modulus,
        1e-14,
        1e-14,
    )
    .into_iter()
    .flatten();
    let orbs_d3_cg_ref = vec![
        DecomposedSymbol::<MullikenIrrepSymbol>::new("||T|_(2g)|").unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new("||A|_(1g)| ⊕ ||E|_(g)|").unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new("||A|_(1g)| ⊕ ||E|_(g)|").unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new("||E|_(g)|").unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new("||T|_(2g)|").unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new("||A|_(1g)| ⊕ ||E|_(g)|").unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new("||T|_(2g)|").unwrap(),
    ];
    orbit_cg_u_oh_spatial_d3_orbs
        .zip(orbs_d3_cg_ref.iter())
        .for_each(|(mut orb_orbit, sym_ref)| {
            let _ = orb_orbit.calc_smat(Some(&sao_cg)).unwrap().calc_xmat(false);
            assert_eq!(orb_orbit.analyse_rep().unwrap(), *sym_ref);
        });

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // u Oh* (ordinary double, unitary)
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    let mut orbit_cg_u_oh_double_spin_spatial_d3 = SlaterDeterminantSymmetryOrbit::builder()
        .group(&group_u_oh_double)
        .origin(&det_d3_cg)
        .integrality_threshold(1e-14)
        .linear_independence_threshold(1e-14)
        .symmetry_transformation_kind(SymmetryTransformationKind::SpinSpatial)
        .eigenvalue_comparison_mode(EigenvalueComparisonMode::Modulus)
        .build()
        .unwrap();
    let _ = orbit_cg_u_oh_double_spin_spatial_d3
        .calc_smat(Some(&sao_cg))
        .unwrap()
        .calc_xmat(false);
    assert_eq!(
        orbit_cg_u_oh_double_spin_spatial_d3.analyse_rep().unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new("||E~|_(1g)| ⊕ ||E~|_(2g)| ⊕ 2||G~|_(g)|")
            .unwrap()
    );

    let orbit_cg_u_oh_double_spin_spatial_d3_orbs = MolecularOrbitalSymmetryOrbit::from_orbitals(
        &group_u_oh_double,
        &orbs_d3_cg,
        SymmetryTransformationKind::SpinSpatial,
        EigenvalueComparisonMode::Modulus,
        1e-13,
        1e-13,
    )
    .into_iter()
    .flatten();
    let orbs_d3_cg_spin_spatial_ref = vec![
        DecomposedSymbol::<MullikenIrrepSymbol>::new("||E~|_(2g)| ⊕ ||G~|_(g)|").unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new("||E~|_(1g)| ⊕ ||G~|_(g)|").unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new("||E~|_(1g)| ⊕ ||G~|_(g)|").unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new("||G~|_(g)|").unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new("||E~|_(2g)| ⊕ ||G~|_(g)|").unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new("||E~|_(1g)| ⊕ ||G~|_(g)|").unwrap(),
        DecomposedSymbol::<MullikenIrrepSymbol>::new("||E~|_(2g)| ⊕ ||G~|_(g)|").unwrap(),
    ];
    orbit_cg_u_oh_double_spin_spatial_d3_orbs
        .zip(orbs_d3_cg_spin_spatial_ref.iter())
        .for_each(|(mut orb_orbit, sym_ref)| {
            let _ = orb_orbit.calc_smat(Some(&sao_cg)).unwrap().calc_xmat(false);
            assert_eq!(orb_orbit.analyse_rep().unwrap(), *sym_ref);
        });
}
