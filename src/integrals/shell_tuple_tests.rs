use approx;
use nalgebra::{Point3, Vector3};
use ndarray::{array, Array2};
use ndarray_linalg::assert::close_l2;

use crate::basis::ao::*;
use crate::basis::ao_integrals::*;

#[test]
fn test_integrals_shell_tuple() {
    let bs0 = BasisShell::new(1, ShellOrder::Cart(CartOrder::lex(1)));
    let gc0 = GaussianContraction::<f64, f64> {
        primitives: vec![(0.1, 0.3), (0.2, 0.5)],
    };
    let bsc0 = BasisShellContraction::<f64, f64> {
        basis_shell: bs0,
        contraction: gc0,
        cart_origin: Point3::new(1.0, 0.0, 0.0),
        k: None,
    };

    let bs1 = BasisShell::new(2, ShellOrder::Cart(CartOrder::lex(2)));
    let gc1 = GaussianContraction::<f64, f64> {
        primitives: vec![(0.3, 0.2), (0.4, 0.6), (0.5, 0.4)],
    };
    let bsc1 = BasisShellContraction::<f64, f64> {
        basis_shell: bs1,
        contraction: gc1,
        cart_origin: Point3::new(2.0, 1.0, 1.0),
        k: Some(Vector3::z()),
    };

    let st = build_shell_tuple![(&bsc0, true), (&bsc1, false), (&bsc1, true); f64];
    assert_eq!(st.angular_shell_shape, [3, 6, 6]);
    assert_eq!(st.primitive_shell_shape, [2, 3, 3]);
    assert_eq!(st.ks, [None, Some(-Vector3::z()), Some(Vector3::z())]);
    assert_eq!(st.k.norm(), 0.0);
    assert_eq!(st.ns, [1, 2, 2]);
    assert_eq!(st.zg[(1, 2, 0)], 1.0);
    approx::assert_relative_eq!(st.zd[(0, 1, 2)], 0.02);
    approx::assert_relative_eq!(st.dd[(0, 1, 2)], 0.072);
    approx::assert_relative_eq!(st.rg[(1, 1, 2)], Point3::new(2.0, 0.9, 0.9) / 1.1);
    approx::assert_relative_eq!(
        st.qs[1].as_ref().unwrap()[(1, 1, 2)],
        Vector3::new(2.0, 0.9, 0.9) / 1.1 - Vector3::new(2.0, 1.0, 1.0)
    );

    let st2 = build_shell_tuple![(&bsc0, true), (&bsc1, false); f64];
    assert_eq!(st2.angular_shell_shape, [3, 6]);
    assert_eq!(st2.primitive_shell_shape, [2, 3]);
}

#[test]
fn test_integrals_shell_tuple_overlap_2c_h2() {
    // ~~~~~~~~~~~~~~~~~
    // H2, STO-3G
    // Reference: Q-Chem
    // ~~~~~~~~~~~~~~~~~
    let bs_cs = BasisShell::new(0, ShellOrder::Cart(CartOrder::lex(0)));
    let gc_h_sto3g_1s = GaussianContraction::<f64, f64> {
        primitives: vec![
            (3.4252509140, 0.1543289673),
            (0.6239137298, 0.5353281423),
            (0.1688554040, 0.4446345422),
        ],
    };

    let bsc0 = BasisShellContraction::<f64, f64> {
        basis_shell: bs_cs.clone(),
        contraction: gc_h_sto3g_1s.clone(),
        cart_origin: Point3::new(0.0, 0.0, 0.0),
        k: None,
    };
    let bsc1 = BasisShellContraction::<f64, f64> {
        basis_shell: bs_cs.clone(),
        contraction: gc_h_sto3g_1s,
        cart_origin: Point3::new(0.0, 0.0, 1.0),
        k: None,
    };

    let st_00 = build_shell_tuple![(&bsc0, true), (&bsc0, false); f64];
    let ovs_00 = st_00.overlap([0, 0]);
    close_l2(&ovs_00[0], &array![[1.0]], 1e-7);

    let st_01 = build_shell_tuple![(&bsc0, true), (&bsc1, false); f64];
    let ovs_01 = st_01.overlap([0, 0]);
    close_l2(&ovs_01[0], &array![[0.796588301]], 1e-7);

    // ~~~~~~~~~~~~~~~~~
    // H2, 6-31G**
    // Reference: Q-Chem
    // ~~~~~~~~~~~~~~~~~
    let bs_cp = BasisShell::new(1, ShellOrder::Cart(CartOrder::lex(1)));
    let gc_h_631gss_1s = GaussianContraction::<f64, f64> {
        primitives: vec![
            (18.731136960, 0.03349460434),
            (2.8253943650, 0.23472695350),
            (0.6401216923, 0.81375732610),
        ],
    };
    let gc_h_631gss_2s = GaussianContraction::<f64, f64> {
        primitives: vec![(0.1612777588, 1.00000000000)],
    };
    let gc_h_631gss_2p = GaussianContraction::<f64, f64> {
        primitives: vec![(1.1000000000, 1.00000000000)],
    };

    let bsc0_1s = BasisShellContraction::<f64, f64> {
        basis_shell: bs_cs.clone(),
        contraction: gc_h_631gss_1s.clone(),
        cart_origin: Point3::new(0.0, 0.0, 0.0),
        k: None,
    };
    let bsc0_2s = BasisShellContraction::<f64, f64> {
        basis_shell: bs_cs.clone(),
        contraction: gc_h_631gss_2s.clone(),
        cart_origin: Point3::new(0.0, 0.0, 0.0),
        k: None,
    };
    let bsc0_2p = BasisShellContraction::<f64, f64> {
        basis_shell: bs_cp.clone(),
        contraction: gc_h_631gss_2p.clone(),
        cart_origin: Point3::new(0.0, 0.0, 0.0),
        k: None,
    };
    let bsc1_1s = BasisShellContraction::<f64, f64> {
        basis_shell: bs_cs.clone(),
        contraction: gc_h_631gss_1s,
        cart_origin: Point3::new(1.0, 1.0, 1.0),
        k: None,
    };
    let bsc1_2s = BasisShellContraction::<f64, f64> {
        basis_shell: bs_cs.clone(),
        contraction: gc_h_631gss_2s,
        cart_origin: Point3::new(1.0, 1.0, 1.0),
        k: None,
    };
    let bsc1_2p = BasisShellContraction::<f64, f64> {
        basis_shell: bs_cp.clone(),
        contraction: gc_h_631gss_2p,
        cart_origin: Point3::new(1.0, 1.0, 1.0),
        k: None,
    };

    // <01s|:>
    let st_01s01s = build_shell_tuple![(&bsc0_1s, true), (&bsc0_1s, false); f64];
    let ovs_01s01s = st_01s01s.overlap([0, 0]);
    close_l2(&ovs_01s01s[0], &array![[1.0]], 1e-6);

    let st_01s02s = build_shell_tuple![(&bsc0_1s, true), (&bsc0_2s, false); f64];
    let ovs_01s02s = st_01s02s.overlap([0, 0]);
    close_l2(&ovs_01s02s[0], &array![[0.6582920]], 1e-6);

    let st_01s02p = build_shell_tuple![(&bsc0_1s, true), (&bsc0_2p, false); f64];
    let ovs_01s02p = st_01s02p.overlap([0, 0]);
    close_l2(&ovs_01s02p[0], &array![[0.0, 0.0, 0.0]], 1e-6);

    let st_01s11s = build_shell_tuple![(&bsc0_1s, true), (&bsc1_1s, false); f64];
    let ovs_01s11s = st_01s11s.overlap([0, 0]);
    close_l2(&ovs_01s11s[0], &array![[0.3107063]], 1e-6);

    let st_01s12s = build_shell_tuple![(&bsc0_1s, true), (&bsc1_2s, false); f64];
    let ovs_01s12s = st_01s12s.overlap([0, 0]);
    close_l2(&ovs_01s12s[0], &array![[0.4437869]], 1e-6);

    let st_01s12p = build_shell_tuple![(&bsc0_1s, true), (&bsc1_2p, false); f64];
    let ovs_01s12p = st_01s12p.overlap([0, 0]);
    close_l2(
        &ovs_01s12p[0],
        &array![[-0.2056149, -0.2056149, -0.2056149]],
        1e-6
    );

    // <02s|:>
    let st_02s02s = build_shell_tuple![(&bsc0_2s, true), (&bsc0_2s, false); f64];
    let ovs_02s02s = st_02s02s.overlap([0, 0]);
    close_l2(&ovs_02s02s[0], &array![[1.0]], 1e-6);

    let st_02s02p = build_shell_tuple![(&bsc0_2s, true), (&bsc0_2p, false); f64];
    let ovs_02s02p = st_02s02p.overlap([0, 0]);
    close_l2(&ovs_02s02p[0], &array![[0.0, 0.0, 0.0]], 1e-6);

    let st_02s11s = build_shell_tuple![(&bsc0_2s, true), (&bsc1_1s, false); f64];
    let ovs_02s11s = st_02s11s.overlap([0, 0]);
    close_l2(&ovs_02s11s[0], &array![[0.4437869]], 1e-6);

    let st_02s12s = build_shell_tuple![(&bsc0_2s, true), (&bsc1_2s, false); f64];
    let ovs_02s12s = st_02s12s.overlap([0, 0]);
    close_l2(&ovs_02s12s[0], &array![[0.7851216]], 1e-6);

    let st_02s12p = build_shell_tuple![(&bsc0_2s, true), (&bsc1_2p, false); f64];
    let ovs_02s12p = st_02s12p.overlap([0, 0]);
    close_l2(
        &ovs_02s12p[0],
        &array![[-0.0960035, -0.0960035, -0.0960035]],
        1e-6
    );

    // <02p|:>
    let st_02p02p = build_shell_tuple![(&bsc0_2p, true), (&bsc0_2p, false); f64];
    let ovs_02p02p = st_02p02p.overlap([0, 0]);
    close_l2(
        &ovs_02p02p[0],
        &array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],],
        1e-6
    );

    let st_02p11s = build_shell_tuple![(&bsc0_2p, true), (&bsc1_1s, false); f64];
    let ovs_02p11s = st_02p11s.overlap([0, 0]);
    close_l2(
        &ovs_02p11s[0],
        &array![[0.2056149], [0.2056149], [0.2056149],],
        1e-6
    );

    let st_02p12s = build_shell_tuple![(&bsc0_2p, true), (&bsc1_2s, false); f64];
    let ovs_02p12s = st_02p12s.overlap([0, 0]);
    close_l2(
        &ovs_02p12s[0],
        &array![[0.0960035], [0.0960035], [0.0960035],],
        1e-6
    );

    let st_02p12p = build_shell_tuple![(&bsc0_2p, true), (&bsc1_2p, false); f64];
    let ovs_02p12p = st_02p12p.overlap([0, 0]);
    #[rustfmt::skip]
    close_l2(
        &ovs_02p12p[0],
        &array![
            [-0.0192050, -0.2112549, -0.2112549],
            [-0.2112549, -0.0192050, -0.2112549],
            [-0.2112549, -0.2112549, -0.0192050],
        ],
        1e-6
    );
}

#[test]
fn test_integrals_shell_tuple_overlap_2c_li2() {
    // ~~~~~~~~~~~~~~~~~
    // Li2, 6-31G*
    // Reference: Q-Chem
    // ~~~~~~~~~~~~~~~~~
    let bs_cs = BasisShell::new(0, ShellOrder::Cart(CartOrder::qchem(0)));
    let bs_cp = BasisShell::new(1, ShellOrder::Cart(CartOrder::qchem(1)));
    let bs_cd = BasisShell::new(2, ShellOrder::Cart(CartOrder::qchem(2)));

    let gc_li_631gs_1s = GaussianContraction::<f64, f64> {
        primitives: vec![
            (0.6424189150e+03, 0.2142607810e-02),
            (0.9679851530e+02, 0.1620887150e-01),
            (0.2209112120e+02, 0.7731557250e-01),
            (0.6201070250e+01, 0.2457860520e+00),
            (0.1935117680e+01, 0.4701890040e+00),
            (0.6367357890e+00, 0.3454708450e+00),
        ],
    };
    let gc_li_631gs_2s = GaussianContraction::<f64, f64> {
        primitives: vec![
            (0.2324918408e+01, -0.3509174574e-01),
            (0.6324303556e+00, -0.1912328431e+00),
            (0.7905343475e-01, 0.1083987795e+01),
        ],
    };
    let gc_li_631gs_2p = GaussianContraction::<f64, f64> {
        primitives: vec![
            (0.2324918408e+01, 0.8941508043e-02),
            (0.6324303556e+00, 0.1410094640e+00),
            (0.7905343475e-01, 0.9453636953e+00),
        ],
    };
    let gc_li_631gs_3s = GaussianContraction::<f64, f64> {
        primitives: vec![(0.3596197175e-01, 0.1000000000e+01)],
    };
    let gc_li_631gs_3p = GaussianContraction::<f64, f64> {
        primitives: vec![(0.3596197175e-01, 0.1000000000e+01)],
    };
    let gc_li_631gs_3d = GaussianContraction::<f64, f64> {
        primitives: vec![(0.2000000000e+00, 1.0000000)],
    };

    let bsc0_1s = BasisShellContraction::<f64, f64> {
        basis_shell: bs_cs.clone(),
        contraction: gc_li_631gs_1s.clone(),
        cart_origin: Point3::new(0.0, 0.0, 0.0),
        k: None,
    };
    let bsc0_2s = BasisShellContraction::<f64, f64> {
        basis_shell: bs_cs.clone(),
        contraction: gc_li_631gs_2s.clone(),
        cart_origin: Point3::new(0.0, 0.0, 0.0),
        k: None,
    };
    let bsc0_2p = BasisShellContraction::<f64, f64> {
        basis_shell: bs_cp.clone(),
        contraction: gc_li_631gs_2p.clone(),
        cart_origin: Point3::new(0.0, 0.0, 0.0),
        k: None,
    };
    let bsc0_3s = BasisShellContraction::<f64, f64> {
        basis_shell: bs_cs.clone(),
        contraction: gc_li_631gs_3s.clone(),
        cart_origin: Point3::new(0.0, 0.0, 0.0),
        k: None,
    };
    let bsc0_3p = BasisShellContraction::<f64, f64> {
        basis_shell: bs_cp.clone(),
        contraction: gc_li_631gs_3p.clone(),
        cart_origin: Point3::new(0.0, 0.0, 0.0),
        k: None,
    };
    let bsc0_3d = BasisShellContraction::<f64, f64> {
        basis_shell: bs_cd.clone(),
        contraction: gc_li_631gs_3d.clone(),
        cart_origin: Point3::new(0.0, 0.0, 0.0),
        k: None,
    };
    let bsc1_1s = BasisShellContraction::<f64, f64> {
        basis_shell: bs_cs.clone(),
        contraction: gc_li_631gs_1s.clone(),
        cart_origin: Point3::new(2.0, 1.0, 0.0),
        k: None,
    };
    let bsc1_2s = BasisShellContraction::<f64, f64> {
        basis_shell: bs_cs.clone(),
        contraction: gc_li_631gs_2s.clone(),
        cart_origin: Point3::new(2.0, 1.0, 0.0),
        k: None,
    };
    let bsc1_2p = BasisShellContraction::<f64, f64> {
        basis_shell: bs_cp.clone(),
        contraction: gc_li_631gs_2p.clone(),
        cart_origin: Point3::new(2.0, 1.0, 0.0),
        k: None,
    };
    let bsc1_3s = BasisShellContraction::<f64, f64> {
        basis_shell: bs_cs.clone(),
        contraction: gc_li_631gs_3s.clone(),
        cart_origin: Point3::new(2.0, 1.0, 0.0),
        k: None,
    };
    let bsc1_3p = BasisShellContraction::<f64, f64> {
        basis_shell: bs_cp.clone(),
        contraction: gc_li_631gs_3p.clone(),
        cart_origin: Point3::new(2.0, 1.0, 0.0),
        k: None,
    };
    let bsc1_3d = BasisShellContraction::<f64, f64> {
        basis_shell: bs_cd.clone(),
        contraction: gc_li_631gs_3d.clone(),
        cart_origin: Point3::new(2.0, 1.0, 0.0),
        k: None,
    };

    // <01s|:>
    let st_01s01s = build_shell_tuple![(&bsc0_1s, true), (&bsc0_1s, false); f64];
    let ovs_01s01s = st_01s01s.overlap([0, 0]);
    close_l2(&ovs_01s01s[0], &array![[1.0]], 1e-6);

    let st_01s02s = build_shell_tuple![(&bsc0_1s, true), (&bsc0_2s, false); f64];
    let ovs_01s02s = st_01s02s.overlap([0, 0]);
    close_l2(&ovs_01s02s[0], &array![[0.1452582]], 1e-6);

    let st_01s02p = build_shell_tuple![(&bsc0_1s, true), (&bsc0_2p, false); f64];
    let ovs_01s02p = st_01s02p.overlap([0, 0]);
    close_l2(&ovs_01s02p[0], &array![[0.0, 0.0, 0.0]], 1e-6);

    let st_01s03s = build_shell_tuple![(&bsc0_1s, true), (&bsc0_3s, false); f64];
    let ovs_01s03s = st_01s03s.overlap([0, 0]);
    close_l2(&ovs_01s03s[0], &array![[0.1857425]], 1e-6);

    let st_01s03p = build_shell_tuple![(&bsc0_1s, true), (&bsc0_3p, false); f64];
    let ovs_01s03p = st_01s03p.overlap([0, 0]);
    close_l2(&ovs_01s03p[0], &array![[0.0, 0.0, 0.0]], 1e-6);

    let st_01s03d = build_shell_tuple![(&bsc0_1s, true), (&bsc0_3d, false); f64];
    let ovs_01s03d = st_01s03d.overlap([0, 0]);
    close_l2(
        &ovs_01s03d[0],
        &array![[0.0996256, 0.0, 0.0996256, 0.0, 0.0, 0.0996256]],
        1e-6
    );

    let st_01s11s = build_shell_tuple![(&bsc0_1s, true), (&bsc1_1s, false); f64];
    let ovs_01s11s = st_01s11s.overlap([0, 0]);
    close_l2(&ovs_01s11s[0], &array![[0.0545917]], 1e-6);

    let st_01s12s = build_shell_tuple![(&bsc0_1s, true), (&bsc1_2s, false); f64];
    let ovs_01s12s = st_01s12s.overlap([0, 0]);
    close_l2(&ovs_01s12s[0], &array![[0.2143351]], 1e-6);

    let st_01s12p = build_shell_tuple![(&bsc0_1s, true), (&bsc1_2p, false); f64];
    let ovs_01s12p = st_01s12p.overlap([0, 0]);
    close_l2(&ovs_01s12p[0], &array![[-0.2452074, -0.1226037, 0.0]], 1e-6);

    let st_01s13s = build_shell_tuple![(&bsc0_1s, true), (&bsc1_3s, false); f64];
    let ovs_01s13s = st_01s13s.overlap([0, 0]);
    close_l2(&ovs_01s13s[0], &array![[0.1562076]], 1e-6);

    let st_01s13p = build_shell_tuple![(&bsc0_1s, true), (&bsc1_3p, false); f64];
    let ovs_01s13p = st_01s13p.overlap([0, 0]);
    close_l2(&ovs_01s13p[0], &array![[-0.1141146, -0.0570573, 0.0]], 1e-6);

    let st_01s13d = build_shell_tuple![(&bsc0_1s, true), (&bsc1_3d, false); f64];
    let ovs_01s13d = st_01s13d.overlap([0, 0]);
    close_l2(
        &ovs_01s13d[0],
        &array![[0.3470063, 0.2615882, 0.1204642, 0.0, 0.0, 0.0449502]],
        1e-6
    );

    // <02p|:>
    let st_02p02p = build_shell_tuple![(&bsc0_2p, true), (&bsc0_2p, false); f64];
    let ovs_02p02p = st_02p02p.overlap([0, 0]);
    #[rustfmt::skip]
    close_l2(
        &ovs_02p02p[0],
        &array![
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        1e-6
    );

    let st_02p03s = build_shell_tuple![(&bsc0_2p, true), (&bsc0_3s, false); f64];
    let ovs_02p03s = st_02p03s.overlap([0, 0]);
    #[rustfmt::skip]
    close_l2(
        &ovs_02p03s[0],
        &array![
            [0.0],
            [0.0],
            [0.0],
        ],
        1e-6
    );

    let st_02p03p = build_shell_tuple![(&bsc0_2p, true), (&bsc0_3p, false); f64];
    let ovs_02p03p = st_02p03p.overlap([0, 0]);
    #[rustfmt::skip]
    close_l2(
        &ovs_02p03p[0],
        &array![
            [0.8020639, 0.0, 0.0],
            [0.0, 0.8020639, 0.0],
            [0.0, 0.0, 0.8020639],
        ],
        1e-6
    );

    let st_02p03d = build_shell_tuple![(&bsc0_2p, true), (&bsc0_3d, false); f64];
    let ovs_02p03d = st_02p03d.overlap([0, 0]);
    #[rustfmt::skip]
    close_l2(
        &ovs_02p03d[0],
        &array![
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        1e-6
    );

    let st_02p11s = build_shell_tuple![(&bsc0_2p, true), (&bsc1_1s, false); f64];
    let ovs_02p11s = st_02p11s.overlap([0, 0]);
    #[rustfmt::skip]
    close_l2(
        &ovs_02p11s[0],
        &array![
            [0.2452074],
            [0.1226037],
            [0.0000000],
        ],
        1e-6
    );

    let st_02p12s = build_shell_tuple![(&bsc0_2p, true), (&bsc1_2s, false); f64];
    let ovs_02p12s = st_02p12s.overlap([0, 0]);
    #[rustfmt::skip]
    close_l2(
        &ovs_02p12s[0],
        &array![
            [0.4137897],
            [0.2068949],
            [0.0000000],
        ],
        1e-6
    );

    let st_02p12p = build_shell_tuple![(&bsc0_2p, true), (&bsc1_2p, false); f64];
    let ovs_02p12p = st_02p12p.overlap([0, 0]);
    #[rustfmt::skip]
    close_l2(
        &ovs_02p12p[0],
        &array![
            [ 0.5209479, -0.1381780, 0.0000000],
            [-0.1381780,  0.7282150, 0.0000000],
            [ 0.0000000,  0.0000000, 0.7973040],
        ],
        1e-6
    );

    let st_02p13s = build_shell_tuple![(&bsc0_2p, true), (&bsc1_3s, false); f64];
    let ovs_02p13s = st_02p13s.overlap([0, 0]);
    #[rustfmt::skip]
    close_l2(
        &ovs_02p13s[0],
        &array![
            [0.2685371],
            [0.1342685],
            [0.0000000],
        ],
        1e-6
    );

    let st_02p13p = build_shell_tuple![(&bsc0_2p, true), (&bsc1_3p, false); f64];
    let ovs_02p13p = st_02p13p.overlap([0, 0]);
    #[rustfmt::skip]
    close_l2(
        &ovs_02p13p[0],
        &array![
            [ 0.5667930, -0.0706191, 0.0000000],
            [-0.0706191,  0.6727216, 0.0000000],
            [ 0.0000000,  0.0000000, 0.7080311],
        ],
        1e-6
    );

    let st_02p13d = build_shell_tuple![(&bsc0_2p, true), (&bsc1_3d, false); f64];
    let ovs_02p13d = st_02p13d.overlap([0, 0]);
    #[rustfmt::skip]
    close_l2(
        &ovs_02p13d[0],
        &array![
            [0.1412689, -0.0692422, 0.4466108,  0.0000000,  0.0000000, 0.4175240],
            [0.2669356, -0.2896237, 0.0270043,  0.0000000,  0.0000000, 0.2087620],
            [0.0000000,  0.0000000, 0.0000000, -0.3400035, -0.1700018, 0.0000000],
        ],
        1e-6
    );

    // <03d|:>
    let st_03d03d = build_shell_tuple![(&bsc0_3d, true), (&bsc0_3d, false); f64];
    let ovs_03d03d = st_03d03d.overlap([0, 0]);
    #[rustfmt::skip]
    close_l2(
        &ovs_03d03d[0],
        &array![
            [1.0000000, 0.0000000, 0.3333333, 0.0000000, 0.0000000, 0.3333333],
            [0.0000000, 1.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000],
            [0.3333333, 0.0000000, 1.0000000, 0.0000000, 0.0000000, 0.3333333],
            [0.0000000, 0.0000000, 0.0000000, 1.0000000, 0.0000000, 0.0000000],
            [0.0000000, 0.0000000, 0.0000000, 0.0000000, 1.0000000, 0.0000000],
            [0.3333333, 0.0000000, 0.3333333, 0.0000000, 0.0000000, 1.0000000],
        ],
        1e-6
    );

    let st_03d13d = build_shell_tuple![(&bsc0_3d, true), (&bsc1_3d, false); f64];
    let ovs_03d13d = st_03d13d.overlap([0, 0]);
    #[rustfmt::skip]
    close_l2(
        &ovs_03d13d[0],
        &array![
            [ 0.4124408, -0.0280145,  0.4367021,  0.0000000,  0.0000000, 0.3639184],
            [-0.0280145,  0.0970449, -0.1120578,  0.0000000,  0.0000000, 0.1400723],
            [ 0.4367021, -0.1120578,  0.5337470,  0.0000000,  0.0000000, 0.2426123],
            [ 0.0000000,  0.0000000,  0.0000000,  0.1213061, -0.2426123, 0.0000000],
            [ 0.0000000,  0.0000000,  0.0000000, -0.2426123,  0.4852245, 0.0000000],
            [ 0.3639184,  0.1400723,  0.2426123,  0.0000000,  0.0000000, 0.6065307],
        ],
        1e-6
    );
}

#[test]
fn test_integrals_shell_tuple_overlap_2c_cr2() {
    // ~~~~~~~~~~~~~~~~~
    // Cr2, 6-31G*
    // Reference: Q-Chem
    // ~~~~~~~~~~~~~~~~~
    // let bs_cs = BasisShell::new(0, ShellOrder::Cart(CartOrder::qchem(0)));
    let bs_cp = BasisShell::new(1, ShellOrder::Cart(CartOrder::qchem(1)));
    let bs_cd = BasisShell::new(2, ShellOrder::Cart(CartOrder::qchem(2)));
    let bs_pf = BasisShell::new(3, ShellOrder::Pure(PureOrder::increasingm(3)));

    // let gc_cr_631gs_1s = GaussianContraction::<f64, f64> {
    //     primitives: vec![
    //         (0.5178981000e+05, 0.1776181956e-02),
    //         (0.7776849000e+04, 0.1360475966e-01),
    //         (0.1771385000e+04, 0.6706924832e-01),
    //         (0.4991588000e+03, 0.2323103942e+00),
    //         (0.1597982000e+03, 0.4802409880e+00),
    //         (0.5447021000e+02, 0.3487652913e+00),
    //     ],
    // };
    // let gc_cr_631gs_2s = GaussianContraction::<f64, f64> {
    //     primitives: vec![
    //         (0.1064328000e+04, 0.2399669027e-02),
    //         (0.2532138000e+03, 0.3194886035e-01),
    //         (0.8160924000e+02, 0.1250868014e+00),
    //         (0.3048193000e+02, -0.3221866036e-01),
    //         (0.1229439000e+02, -0.6172284069e+00),
    //         (0.5037722000e+01, -0.4525936050e+00),
    //     ],
    // };
    // let gc_cr_631gs_2p = GaussianContraction::<f64, f64> {
    //     primitives: vec![
    //         (0.1064328000e+04, 0.3986996969e-02),
    //         (0.2532138000e+03, 0.3104661976e-01),
    //         (0.8160924000e+02, 0.1350517989e+00),
    //         (0.3048193000e+02, 0.3448864973e+00),
    //         (0.1229439000e+02, 0.4628570964e+00),
    //         (0.5037722000e+01, 0.2110425984e+00),
    //     ],
    // };
    // let gc_cr_631gs_3s = GaussianContraction::<f64, f64> {
    //     primitives: vec![
    //         (0.4156291000e+02, -0.3454215978e-02),
    //         (0.1367627000e+02, 0.7218427953e-01),
    //         (0.5844390000e+01, 0.2544819984e+00),
    //         (0.2471609000e+01, -0.2934533981e+00),
    //         (0.1028308000e+01, -0.7385454952e+00),
    //         (0.4072500000e+00, -0.1947156987e+00),
    //     ],
    // };
    // let gc_cr_631gs_3p = GaussianContraction::<f64, f64> {
    //     primitives: vec![
    //         (0.4156291000e+02, -0.6722497017e-02),
    //         (0.1367627000e+02, -0.2806471007e-01),
    //         (0.5844390000e+01, 0.5820028015e-01),
    //         (0.2471609000e+01, 0.3916988010e+00),
    //         (0.1028308000e+01, 0.5047823013e+00),
    //         (0.4072500000e+00, 0.1790290005e+00),
    //     ],
    // };
    // let gc_cr_631gs_4s = GaussianContraction::<f64, f64> {
    //     primitives: vec![
    //         (0.1571464000e+01, 0.5892221460e-01),
    //         (0.6055800000e+00, 0.2976056242e+00),
    //         (0.9856100000e-01, -0.1147506479e+01),
    //     ],
    // };
    // let gc_cr_631gs_4p = GaussianContraction::<f64, f64> {
    //     primitives: vec![
    //         (0.1571464000e+01, -0.1930100080e+00),
    //         (0.6055800000e+00, 0.9605620398e-01),
    //         (0.9856100000e-01, 0.9817609407e+00),
    //     ],
    // };
    // let gc_cr_631gs_5s = GaussianContraction::<f64, f64> {
    //     primitives: vec![(0.3645900000e-01, 0.1000000000e+01)],
    // };
    let gc_cr_631gs_5p = GaussianContraction::<f64, f64> {
        primitives: vec![(0.3645900000e-01, 0.1000000000e+01)],
    };
    let gc_cr_631gs_3d = GaussianContraction::<f64, f64> {
        primitives: vec![
            (0.1841930000e+02, 0.8650816335e-01),
            (0.4812661000e+01, 0.3826699148e+00),
            (0.1446447000e+01, 0.7093772274e+00),
        ],
    };
    let gc_cr_631gs_4d = GaussianContraction::<f64, f64> {
        primitives: vec![(0.4004130000e+00, 1.0)],
    };
    let gc_cr_631gs_4f = GaussianContraction::<f64, f64> {
        primitives: vec![(0.8000000000e+00, 1.0)],
    };

    // let bsc0_1s = BasisShellContraction::<f64, f64> {
    //     basis_shell: bs_cs.clone(),
    //     contraction: gc_cr_631gs_1s.clone(),
    //     cart_origin: Point3::new(0.0, 0.0, 0.0),
    //     k: None,
    // };
    // let bsc0_2s = BasisShellContraction::<f64, f64> {
    //     basis_shell: bs_cs.clone(),
    //     contraction: gc_cr_631gs_2s.clone(),
    //     cart_origin: Point3::new(0.0, 0.0, 0.0),
    //     k: None,
    // };
    // let bsc0_2p = BasisShellContraction::<f64, f64> {
    //     basis_shell: bs_cp.clone(),
    //     contraction: gc_cr_631gs_2p.clone(),
    //     cart_origin: Point3::new(0.0, 0.0, 0.0),
    //     k: None,
    // };
    // let bsc0_3s = BasisShellContraction::<f64, f64> {
    //     basis_shell: bs_cs.clone(),
    //     contraction: gc_cr_631gs_3s.clone(),
    //     cart_origin: Point3::new(0.0, 0.0, 0.0),
    //     k: None,
    // };
    // let bsc0_3p = BasisShellContraction::<f64, f64> {
    //     basis_shell: bs_cp.clone(),
    //     contraction: gc_cr_631gs_3p.clone(),
    //     cart_origin: Point3::new(0.0, 0.0, 0.0),
    //     k: None,
    // };
    // let bsc0_4s = BasisShellContraction::<f64, f64> {
    //     basis_shell: bs_cs.clone(),
    //     contraction: gc_cr_631gs_4s.clone(),
    //     cart_origin: Point3::new(0.0, 0.0, 0.0),
    //     k: None,
    // };
    // let bsc0_4p = BasisShellContraction::<f64, f64> {
    //     basis_shell: bs_cp.clone(),
    //     contraction: gc_cr_631gs_4p.clone(),
    //     cart_origin: Point3::new(0.0, 0.0, 0.0),
    //     k: None,
    // };
    // let bsc0_5s = BasisShellContraction::<f64, f64> {
    //     basis_shell: bs_cs.clone(),
    //     contraction: gc_cr_631gs_5s.clone(),
    //     cart_origin: Point3::new(0.0, 0.0, 0.0),
    //     k: None,
    // };
    // let bsc0_5p = BasisShellContraction::<f64, f64> {
    //     basis_shell: bs_cp.clone(),
    //     contraction: gc_cr_631gs_5p.clone(),
    //     cart_origin: Point3::new(0.0, 0.0, 0.0),
    //     k: None,
    // };
    let bsc0_3d = BasisShellContraction::<f64, f64> {
        basis_shell: bs_cd.clone(),
        contraction: gc_cr_631gs_3d.clone(),
        cart_origin: Point3::new(0.0, 0.0, 0.0),
        k: None,
    };
    let bsc0_4d = BasisShellContraction::<f64, f64> {
        basis_shell: bs_cd.clone(),
        contraction: gc_cr_631gs_4d.clone(),
        cart_origin: Point3::new(0.0, 0.0, 0.0),
        k: None,
    };
    let bsc0_4f = BasisShellContraction::<f64, f64> {
        basis_shell: bs_pf.clone(),
        contraction: gc_cr_631gs_4f.clone(),
        cart_origin: Point3::new(0.0, 0.0, 0.0),
        k: None,
    };
    // let bsc1_1s = BasisShellContraction::<f64, f64> {
    //     basis_shell: bs_cs.clone(),
    //     contraction: gc_cr_631gs_1s.clone(),
    //     cart_origin: Point3::new(1.0, 1.0, 0.0),
    //     k: None,
    // };
    // let bsc1_2s = BasisShellContraction::<f64, f64> {
    //     basis_shell: bs_cs.clone(),
    //     contraction: gc_cr_631gs_2s.clone(),
    //     cart_origin: Point3::new(1.0, 1.0, 0.0),
    //     k: None,
    // };
    // let bsc1_2p = BasisShellContraction::<f64, f64> {
    //     basis_shell: bs_cp.clone(),
    //     contraction: gc_cr_631gs_2p.clone(),
    //     cart_origin: Point3::new(1.0, 1.0, 0.0),
    //     k: None,
    // };
    // let bsc1_3s = BasisShellContraction::<f64, f64> {
    //     basis_shell: bs_cs.clone(),
    //     contraction: gc_cr_631gs_3s.clone(),
    //     cart_origin: Point3::new(1.0, 1.0, 0.0),
    //     k: None,
    // };
    // let bsc1_3p = BasisShellContraction::<f64, f64> {
    //     basis_shell: bs_cp.clone(),
    //     contraction: gc_cr_631gs_3p.clone(),
    //     cart_origin: Point3::new(1.0, 1.0, 0.0),
    //     k: None,
    // };
    // let bsc1_4s = BasisShellContraction::<f64, f64> {
    //     basis_shell: bs_cs.clone(),
    //     contraction: gc_cr_631gs_4s.clone(),
    //     cart_origin: Point3::new(1.0, 1.0, 0.0),
    //     k: None,
    // };
    // let bsc1_4p = BasisShellContraction::<f64, f64> {
    //     basis_shell: bs_cp.clone(),
    //     contraction: gc_cr_631gs_4p.clone(),
    //     cart_origin: Point3::new(1.0, 1.0, 0.0),
    //     k: None,
    // };
    // let bsc1_5s = BasisShellContraction::<f64, f64> {
    //     basis_shell: bs_cs.clone(),
    //     contraction: gc_cr_631gs_5s.clone(),
    //     cart_origin: Point3::new(1.0, 1.0, 0.0),
    //     k: None,
    // };
    let bsc1_5p = BasisShellContraction::<f64, f64> {
        basis_shell: bs_cp.clone(),
        contraction: gc_cr_631gs_5p.clone(),
        cart_origin: Point3::new(1.0, 1.0, 0.0),
        k: None,
    };
    let bsc1_3d = BasisShellContraction::<f64, f64> {
        basis_shell: bs_cd.clone(),
        contraction: gc_cr_631gs_3d.clone(),
        cart_origin: Point3::new(1.0, 1.0, 0.0),
        k: None,
    };
    let bsc1_4d = BasisShellContraction::<f64, f64> {
        basis_shell: bs_cd.clone(),
        contraction: gc_cr_631gs_4d.clone(),
        cart_origin: Point3::new(1.0, 1.0, 0.0),
        k: None,
    };
    let bsc1_4f = BasisShellContraction::<f64, f64> {
        basis_shell: bs_pf.clone(),
        contraction: gc_cr_631gs_4f.clone(),
        cart_origin: Point3::new(1.0, 1.0, 0.0),
        k: None,
    };

    // <04f|:>
    let st_04f03d = build_shell_tuple![(&bsc0_4f, true), (&bsc0_3d, false); f64];
    let ovs_04f03d = st_04f03d.overlap([0, 0]);
    #[rustfmt::skip]
    close_l2(
        &ovs_04f03d[0],
        &array![
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        1e-6
    );

    let st_04f04d = build_shell_tuple![(&bsc0_4f, true), (&bsc0_4d, false); f64];
    let ovs_04f04d = st_04f04d.overlap([0, 0]);
    #[rustfmt::skip]
    close_l2(
        &ovs_04f04d[0],
        &array![
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        1e-6
    );

    let st_04f04f = build_shell_tuple![(&bsc0_4f, true), (&bsc0_4f, false); f64];
    let ovs_04f04f = st_04f04f.overlap([0, 0]);
    #[rustfmt::skip]
    close_l2(
        &ovs_04f04f[0],
        &Array2::<f64>::eye(7),
        1e-6
    );

    let st_04f15p = build_shell_tuple![(&bsc0_4f, true), (&bsc1_5p, false); f64];
    let ovs_04f15p = st_04f15p.overlap([0, 0]);
    #[rustfmt::skip]
    close_l2(
        &ovs_04f15p[0],
        &array![
            [ 0.00072276, -0.00001720,  0.00000000],
            [ 0.00000000,  0.00000000,  0.00060418],
            [-0.00017773, -0.00036879,  0.00000000],
            [ 0.00000000,  0.00000000, -0.00046799],
            [-0.00036879, -0.00017773,  0.00000000],
            [ 0.00000000,  0.00000000,  0.00000000],
            [ 0.00001720, -0.00072276,  0.00000000],
        ],
        1e-5
    );

    let st_04f13d = build_shell_tuple![(&bsc0_4f, true), (&bsc1_3d, false); f64];
    let ovs_04f13d = st_04f13d.overlap([0, 0]);
    #[rustfmt::skip]
    close_l2(
        &ovs_04f13d[0],
        &array![
            [ 0.0435394,  0.1130559,  0.0776040,  0.0000000,  0.0000000,  0.2035663],
            [ 0.0000000,  0.0000000,  0.0000000, -0.0240873, -0.0240873,  0.0000000],
            [-0.1588652,  0.0987073, -0.1500698,  0.0000000,  0.0000000,  0.0574159],
            [ 0.0000000,  0.0000000,  0.0000000,  0.0186579,  0.0186579,  0.0000000],
            [-0.1500698,  0.0987073, -0.1588652,  0.0000000,  0.0000000,  0.0574159],
            [ 0.0000000,  0.0000000,  0.0000000,  0.2945346, -0.2945346,  0.0000000],
            [-0.0776040, -0.1130559, -0.0435394,  0.0000000,  0.0000000, -0.2035663],
        ],
        1e-6
    );

    let st_04f14d = build_shell_tuple![(&bsc0_4f, true), (&bsc1_4d, false); f64];
    let ovs_04f14d = st_04f14d.overlap([0, 0]);
    #[rustfmt::skip]
    close_l2(
        &ovs_04f14d[0],
        &array![
            [ 0.0234881,  0.1958447, -0.1643977,  0.0000000,  0.0000000,  0.0179388],
            [ 0.0000000,  0.0000000,  0.0000000,  0.1328553,  0.1328553,  0.0000000],
            [-0.0252059,  0.0284947, -0.0737178,  0.0000000,  0.0000000,  0.1941764],
            [ 0.0000000,  0.0000000,  0.0000000, -0.1029093, -0.1029093,  0.0000000],
            [-0.0737178,  0.0284947, -0.0252059,  0.0000000,  0.0000000,  0.1941764],
            [ 0.0000000,  0.0000000,  0.0000000,  0.2849139, -0.2849139,  0.0000000],
            [ 0.1643977, -0.1958447, -0.0234881,  0.0000000,  0.0000000, -0.0179388],
        ],
        1e-6
    );

    let st_04f14f = build_shell_tuple![(&bsc0_4f, true), (&bsc1_4f, false); f64];
    let ovs_04f14f = st_04f14f.overlap([0, 0]);
    #[rustfmt::skip]
    close_l2(
        &ovs_04f14f[0],
        &array![
            [-0.2360475, 0.0000000, -0.0816754,  0.0000000,  0.0853880,  0.0000000,  0.0383427],
            [ 0.0000000, 0.0179732,  0.0000000,  0.3341268,  0.0000000, -0.0000000,  0.0000000],
            [-0.0816754, 0.0000000,  0.0668601,  0.0000000, -0.3393332,  0.0000000, -0.0853880],
            [ 0.0000000, 0.3341268,  0.0000000,  0.1905155,  0.0000000, -0.0000000,  0.0000000],
            [ 0.0853880, 0.0000000, -0.3393332,  0.0000000,  0.0668601,  0.0000000,  0.0816754],
            [ 0.0000000, 0.0000000,  0.0000000, -0.0000000,  0.0000000, -0.2695974,  0.0000000],
            [ 0.0383427, 0.0000000, -0.0853880,  0.0000000,  0.0816754,  0.0000000, -0.2360475],
        ],
        1e-6
    );
}

#[test]
fn test_integrals_shell_tuple_overlap_2c_optimised_contraction() {
    // ~~~~~~~~~~~~~~~~~
    // C, cc-pVQZ
    // Reference: libint
    // ~~~~~~~~~~~~~~~~~
    let bs_cs = BasisShell::new(0, ShellOrder::Cart(CartOrder::lex(0)));
    let gc_c_ccpvqz_1s = GaussianContraction::<f64, f64> {
        primitives: vec![
            (8.264000e+01, 2.006000e-03),
            (1.241000e+01, 1.534300e-02),
            (2.824000e+00, 7.557900e-02),
        ],
    };

    let mut bsc0 = BasisShellContraction::<f64, f64> {
        basis_shell: bs_cs.clone(),
        contraction: gc_c_ccpvqz_1s.clone(),
        cart_origin: Point3::new(0.0, 0.0, 0.0),
        k: None,
    };
    bsc0.renormalise();

    let st_00 = build_shell_tuple![(&bsc0, true), (&bsc0, false); f64];
    let ovs_00 = st_00.overlap([0, 0]);
    close_l2(&ovs_00[0], &array![[1.0]], 1e-7);
}

#[test]
fn test_integrals_shell_tuple_overlap_2c_b_qchem() {
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // B, cc-pVTZ
    // Reference: Q-Chem
    // Note: Q-Chem's definitions of these shells are not normalised. Without normalisation, the
    // overlap elements are incorrect.
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    let bs_cs = BasisShell::new(0, ShellOrder::Cart(CartOrder::lex(0)));
    let gc_b_ccpvtz_1s = GaussianContraction::<f64, f64> {
        primitives: vec![
            (5.47300000e+03, 5.60470000e-04),
            (8.20900000e+02, 4.33323000e-03),
            (1.86800000e+02, 2.21642700e-02),
            (5.28300000e+01, 8.52621600e-02),
            (1.70800000e+01, 2.40817150e-01),
            (5.99900000e+00, 4.38850320e-01),
            (2.20800000e+00, 3.43751710e-01),
        ],
    };
    let gc_b_ccpvtz_2s = GaussianContraction::<f64, f64> {
        primitives: vec![
            (5.47300000e+03,  1.83000000e-06),
            (8.20900000e+02,  1.14100000e-05),
            (5.28300000e+01, -5.72750000e-04),
            (1.70800000e+01, -6.49550000e-03),
            (5.99900000e+00, -3.99012300e-02),
            (2.20800000e+00, -1.27115170e-01),
            (2.41500000e-01,  7.86849720e-01),
        ],
    };

    let mut bsc0 = BasisShellContraction::<f64, f64> {
        basis_shell: bs_cs.clone(),
        contraction: gc_b_ccpvtz_1s.clone(),
        cart_origin: Point3::new(0.0, 0.0, 0.0),
        k: None,
    };
    bsc0.renormalise();
    let mut bsc1 = BasisShellContraction::<f64, f64> {
        basis_shell: bs_cs.clone(),
        contraction: gc_b_ccpvtz_2s.clone(),
        cart_origin: Point3::new(0.0, 0.0, 0.0),
        k: None,
    };
    bsc1.renormalise();

    let st_00 = build_shell_tuple![(&bsc0, true), (&bsc0, false); f64];
    let ovs_00 = st_00.overlap([0, 0]);
    close_l2(&ovs_00[0], &array![[1.0]], 1e-7);

    let st_01 = build_shell_tuple![(&bsc0, true), (&bsc1, false); f64];
    let ovs_01 = st_01.overlap([0, 0]);
    close_l2(&ovs_01[0], &array![[0.1087534]], 1e-6);
}
