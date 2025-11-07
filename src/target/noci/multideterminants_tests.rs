// use env_logger;
use anyhow::format_err;
use approx::assert_abs_diff_eq;
use ndarray::linalg::Dot;
use ndarray::{Array2, ShapeBuilder, array, s};
use ndarray_linalg::{Trace, assert_close_l2};

use crate::angmom::spinor_rotation_3d::SpinConstraint;
use crate::auxiliary::atom::{Atom, ElementMap};
use crate::auxiliary::geometry::Transform;
use crate::auxiliary::molecule::Molecule;
use crate::basis::ao::{BasisAngularOrder, BasisAtom, BasisShell, CartOrder, ShellOrder};
use crate::group::UnitaryRepresentedGroup;
use crate::symmetry::symmetry_core::{PreSymmetry, Symmetry};
use crate::symmetry::symmetry_group::SymmetryGroupProperties;
use crate::symmetry::symmetry_transformation::SymmetryTransformable;
use crate::target::determinant::SlaterDeterminant;
use crate::target::noci::basis::OrbitBasis;
use crate::target::noci::multideterminant::MultiDeterminant;
use crate::target::noci::multideterminants::MultiDeterminants;

#[test]
fn test_multideterminants_density_matrices_bh3p() {
    // env_logger::init();
    let emap = ElementMap::new();
    let atm_b0 = Atom::from_xyz("B +0.0000000 +0.0000000 +0.0000000", &emap, 1e-6).unwrap();
    let atm_h0 = Atom::from_xyz("H +2.2319728 +0.0000000 +0.0000000", &emap, 1e-6).unwrap();
    let atm_h1 = Atom::from_xyz("H -1.1159864 +1.9329451 +0.0000000", &emap, 1e-6).unwrap();
    let atm_h2 = Atom::from_xyz("H -1.1159864 -1.9329451 +0.0000000", &emap, 1e-6).unwrap();

    let bsc_s = BasisShell::new(0, ShellOrder::Cart(CartOrder::qchem(0)));
    let bsc_p = BasisShell::new(1, ShellOrder::Cart(CartOrder::qchem(1)));

    let batm_b0 = BasisAtom::new(&atm_b0, &[bsc_s.clone(), bsc_s.clone(), bsc_p]);
    let batm_h0 = BasisAtom::new(&atm_h0, &[bsc_s.clone()]);
    let batm_h1 = BasisAtom::new(&atm_h1, &[bsc_s.clone()]);
    let batm_h2 = BasisAtom::new(&atm_h2, &[bsc_s]);

    let bao_bh3 = BasisAngularOrder::new(&[batm_b0, batm_h0, batm_h1, batm_h2]);
    let mol_bh3 = Molecule::from_atoms(
        &[
            atm_b0.clone(),
            atm_h0.clone(),
            atm_h1.clone(),
            atm_h2.clone(),
        ],
        1e-7,
    )
    .recentre();

    #[rustfmt::skip]
    let sao_spatial = array![
        [1.0000000000e+00, 2.6932778931e-01,  9.4782710980e-26,  0.0000000000e+00, 0.0000000000e+00, 6.7859361797e-02,  6.7859362160e-02,  6.7859362160e-02],
        [2.6932778931e-01, 1.0000000000e+00,  1.2858522622e-25,  0.0000000000e+00, 0.0000000000e+00, 4.8046489293e-01,  4.8046489415e-01,  4.8046489415e-01],
        [9.4782710980e-26, 1.2858522622e-25,  1.0000000000e+00,  0.0000000000e+00, 0.0000000000e+00, 4.9114906941e-01, -2.4557453461e-01, -2.4557453461e-01],
        [0.0000000000e+00, 0.0000000000e+00,  0.0000000000e+00,  1.0000000000e+00, 0.0000000000e+00, 1.3585745364e-18,  4.2534757211e-01, -4.2534757211e-01],
        [0.0000000000e+00, 0.0000000000e+00,  0.0000000000e+00,  0.0000000000e+00, 1.0000000000e+00, 1.1007922795e-17,  1.1007922835e-17,  1.1007922835e-17],
        [6.7859361797e-02, 4.8046489293e-01,  4.9114906941e-01,  1.3585745364e-18, 1.1007922795e-17, 1.0000000000e+00,  1.1054269548e-01,  1.1054269548e-01],
        [6.7859362160e-02, 4.8046489415e-01, -2.4557453461e-01,  4.2534757211e-01, 1.1007922835e-17, 1.1054269548e-01,  1.0000000000e+00,  1.1054269548e-01],
        [6.7859362160e-02, 4.8046489415e-01, -2.4557453461e-01, -4.2534757211e-01, 1.1007922835e-17, 1.1054269548e-01,  1.1054269548e-01,  1.0000000000e+00],
    ];

    let presym = PreSymmetry::builder()
        .moi_threshold(1e-6)
        .molecule(&mol_bh3)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false).unwrap();
    let group_u_d3h = UnitaryRepresentedGroup::from_molecular_symmetry(&sym, None).unwrap();

    #[rustfmt::skip]
    let calpha = array![
        [ 9.9094822668e-01,  2.1094092425e-01, -9.1043832870e-02,  1.9275274426e-07, -2.7163580292e-17, -2.0229108812e-01, -1.6766294154e-01, -9.1807542544e-07],
        [ 3.7005073059e-02, -4.9880197973e-01,  2.6849239780e-01, -6.3085246365e-07, -2.9517132323e-17,  1.1236317557e+00,  9.6768336940e-01,  5.3514883500e-06],
        [ 1.2719874371e-03, -1.2743557031e-01, -4.6754080091e-01,  6.5162721643e-07, -2.5355415584e-16,  8.1722327325e-01, -8.8493724789e-01, -4.6336528683e-06],
        [-1.1507233803e-09,  4.6203081457e-08, -6.6725239454e-07, -5.5850670546e-01,  2.4274939577e-16, -2.0838575264e-07, -6.3952238153e-06,  1.1721236793e+00],
        [ 6.2577823900e-18,  6.9240136181e-16,  9.6566637588e-17,  1.0256982876e-15,  1.0000000000e+00,  6.4806787193e-17, -9.4134588447e-17, -1.3672685772e-17],
        [-6.8532773554e-03, -4.8576281461e-01, -4.3931282042e-01,  5.4058842513e-07,  5.8382327450e-16, -1.1894897318e+00,  2.5549377385e-01,  1.2091665427e-06],
        [-6.1294785032e-03, -1.6930146833e-01,  3.0026994481e-01, -4.0976664620e-01,  4.6886144877e-16, -2.4347975289e-01, -8.8507556401e-01, -8.8303945858e-01],
        [-6.1294791615e-03, -1.6930171688e-01,  3.0027116981e-01,  4.0976599981e-01, -4.4302715403e-16, -2.4347996641e-01, -8.8508501652e-01,  8.8302976104e-01],
    ];
    #[rustfmt::skip]
    let cbeta = array![
        [ 9.9148226739e-01,  2.4315897515e-01, -3.9311742896e-08,  1.5986751651e-02, -9.2283765144e-18, -2.3844294722e-01,  2.1556819199e-07,  6.7245804609e-02],
        [ 3.5320941068e-02, -6.5090133680e-01,  1.5327149282e-07, -5.4332750254e-02,  4.9591544464e-18,  1.3974414854e+00, -1.1953103509e-06, -3.7484064882e-01],
        [ 1.6258500806e-03, -1.4810475543e-02,  5.8317665195e-07, -5.1649132323e-01, -1.0647975772e-16,  2.8331408493e-01,  5.9457987903e-07,  1.1569578656e+00],
        [-2.2133034759e-09,  6.1774088317e-08, -6.0502499171e-01, -4.9680052143e-07,  1.3305075310e-16, -6.8113471801e-07, -1.1488030375e+00,  8.4116461662e-07],
        [-6.9481685589e-18,  1.3067088655e-16,  4.5203045483e-16, -3.3363768522e-16,  1.0000000000e+00,  4.2828090708e-17, -7.9734498844e-18, -7.2211448108e-17],
        [-7.8018553518e-03, -1.6946431421e-01,  3.8846271415e-07, -5.0760880794e-01, -6.0886480240e-17, -1.0006315352e+00,  2.1892538411e-08, -7.8832856751e-01],
        [-5.9117106008e-03, -2.5863228203e-01, -3.7405438952e-01,  2.5412423327e-01,  1.1628120886e-16, -6.2277782258e-01,  8.9874556233e-01,  6.6494654768e-01],
        [-5.9117139937e-03, -2.5863209763e-01,  3.7405392439e-01,  2.5412508836e-01, -1.0699388211e-16, -6.2277899616e-01, -8.9874385269e-01,  6.6494776582e-01],
    ];
    let oalpha = array![1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0];
    let obeta = array![1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let det = SlaterDeterminant::<f64, SpinConstraint>::builder()
        .coefficients(&[calpha, cbeta])
        .occupations(&[oalpha, obeta])
        .baos(vec![&bao_bh3])
        .mol(&mol_bh3)
        .structure_constraint(SpinConstraint::Unrestricted(2, false))
        .complex_symmetric(false)
        .threshold(1e-7)
        .build()
        .unwrap();

    // --------------------
    // Determinant symmetry
    // --------------------
    // ||A|^(')_(1)| ⊕ ||E|^(')|

    // -----------
    // NOCI states
    // -----------

    let orbit_basis = OrbitBasis::builder()
        .group(&group_u_d3h)
        .origins(vec![det.clone()])
        .action(|g, det| det.sym_transform_spatial(g).map_err(|err| format_err!(err)))
        .build()
        .unwrap();

    // A'1
    let sqrt12 = 12.0f64.sqrt();
    let sqrt12inv = 1.0 / sqrt12;
    let a1_multidet = MultiDeterminant::builder()
        .basis(orbit_basis.clone())
        .coefficients(array![
            sqrt12inv, sqrt12inv, sqrt12inv, sqrt12inv, sqrt12inv, sqrt12inv, sqrt12inv, sqrt12inv,
            sqrt12inv, sqrt12inv, sqrt12inv, sqrt12inv,
        ])
        .threshold(1e-7)
        .build()
        .unwrap();
    let a1_denmat = a1_multidet
        .density_matrix(&sao_spatial.view(), 1e-7, 1e-7, true)
        .unwrap();

    // E'(x)
    let ex_multidet = MultiDeterminant::builder()
        .basis(orbit_basis.clone())
        .coefficients(array![
            0.0,
            1.0 / 2.0f64.sqrt(),
            -1.0 / 2.0f64.sqrt(),
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ])
        .threshold(1e-7)
        .build()
        .unwrap();
    let ex_denmat = ex_multidet
        .density_matrix(&sao_spatial.view(), 1e-7, 1e-7, true)
        .unwrap();

    // Collection
    let multidets_coeffs = Array2::from_shape_vec(
        (12, 2).f(),
        a1_multidet
            .coefficients()
            .iter()
            .cloned()
            .chain(ex_multidet.coefficients().iter().cloned())
            .collect::<Vec<_>>(),
    )
    .unwrap();
    let multidets = MultiDeterminants::builder()
        .basis(orbit_basis.clone())
        .coefficients(multidets_coeffs)
        .threshold(1e-7)
        .build()
        .unwrap();

    let denmats = multidets
        .density_matrices(&sao_spatial.view(), 1e-7, 1e-7, true)
        .unwrap();

    assert_close_l2!(&denmats.slice(s![0, .., ..]), &a1_denmat, 1e-7);
    assert_close_l2!(&denmats.slice(s![1, .., ..]), &ex_denmat, 1e-7);

    let a1_denmat2: Array2<_> = denmats.slice(s![0, .., ..]).to_owned();
    assert_abs_diff_eq!(a1_denmat2.dot(&sao_spatial).trace().unwrap(), 7.0, epsilon = 1e-7);

    let ex_denmat2: Array2<_> = denmats.slice(s![1, .., ..]).to_owned();
    assert_abs_diff_eq!(ex_denmat2.dot(&sao_spatial).trace().unwrap(), 7.0, epsilon = 1e-7);
}
