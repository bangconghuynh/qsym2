use approx;
use nalgebra::{Point3, Vector3};
use ndarray::array;
use ndarray_linalg::assert_close_l2;

use crate::basis::ao::*;
use crate::basis::ao_integrals::*;
use crate::integrals::shell_tuple::*;

#[test]
fn test_integrals_shell_tuple() {
    define_shell_tuple![<s1, s2, s3>];

    let bs0 = BasisShell::new(1, ShellOrder::Cart(CartOrder::lex(1)));
    let gc0 = GaussianContraction::<f64, f64> {
        primitives: vec![(0.1, 0.3), (0.2, 0.5)],
    };
    let bsc0 = BasisShellContraction::<f64, f64> {
        basis_shell: bs0,
        start_index: 0,
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
        start_index: 3,
        contraction: gc1,
        cart_origin: Point3::new(2.0, 1.0, 1.0),
        k: Some(Vector3::z()),
    };

    let st = build_shell_tuple![(&bsc0, true), (&bsc1, false), (&bsc1, true); C128];
    assert_eq!(st.function_shell_shape, [3, 6, 6]);
    assert_eq!(st.shell_boundaries, [(0, 3), (3, 9), (3, 9)]);
    assert_eq!(st.ks, [None, Some(Vector3::z()), Some(-Vector3::z())]);
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
}

#[test]
fn test_integrals_shell_tuple_collection() {
    let bs0 = BasisShell::new(1, ShellOrder::Cart(CartOrder::lex(1)));
    let gc0 = GaussianContraction::<f64, f64> {
        primitives: vec![(0.1, 0.1), (0.2, 0.2)],
    };
    let bsc0 = BasisShellContraction::<f64, f64> {
        basis_shell: bs0,
        start_index: 0,
        contraction: gc0,
        cart_origin: Point3::new(1.0, 0.0, 0.0),
        k: None,
    };
    let bs1 = BasisShell::new(2, ShellOrder::Cart(CartOrder::lex(2)));
    let gc1 = GaussianContraction::<f64, f64> {
        primitives: vec![(0.3, 0.3), (0.4, 0.4), (0.5, 0.5)],
    };
    let bsc1 = BasisShellContraction::<f64, f64> {
        basis_shell: bs1,
        start_index: 3,
        contraction: gc1,
        cart_origin: Point3::new(2.0, 1.0, 1.0),
        k: Some(Vector3::z()),
    };
    let bscs_1 = [&bsc0];
    let bscs_2 = [&bsc0, &bsc1];
    let stc = build_shell_tuple_collection![
        <s1, s2, s3, s4, s5>;
        true, true, false, true, false;
        bscs_2, bscs_1, bscs_2, bscs_2, bscs_2;
        C128
    ];
    assert_eq!(stc.lmax(), 2);
    assert_eq!(stc.ccs, [true, true, false, true, false]);
    assert_eq!(stc.unique_shell_tuples_iter([1, 1, 0, 2, 0]).count(), 12);
    assert_eq!(
        stc.unique_shell_tuples_iter([1, 1, 0, 2, 0])
            .flat_map(|(_, equiv_terms)| equiv_terms)
            .count(),
        16
    );
}

#[test]
fn test_integrals_shell_tuple_overlap_2c_h2() {
    define_shell_tuple![<s1, s2>];

    // H2, STO-3G
    // Reference: Q-Chem
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
        start_index: 0,
        contraction: gc_h_sto3g_1s.clone(),
        cart_origin: Point3::new(0.0, 0.0, 0.0),
        k: None,
    };
    let bsc1 = BasisShellContraction::<f64, f64> {
        basis_shell: bs_cs.clone(),
        start_index: 1,
        contraction: gc_h_sto3g_1s,
        cart_origin: Point3::new(0.0, 0.0, 1.0),
        k: None,
    };

    let st_00 = build_shell_tuple![(&bsc0, true), (&bsc0, false); f64];
    let ovs_00 = st_00.overlap([0, 0]);
    assert_close_l2!(&ovs_00[0], &array![[1.0]], 1e-7);

    let st_01 = build_shell_tuple![(&bsc0, true), (&bsc1, false); f64];
    let ovs_01 = st_01.overlap([0, 0]);
    assert_close_l2!(&ovs_01[0], &array![[0.796588301]], 1e-7);

    // H2, 6-31G**
    // Reference: Q-Chem
    let bs_cp = BasisShell::new(1, ShellOrder::Cart(CartOrder::lex(1)));
    let gc_h_631gss_1s = GaussianContraction::<f64, f64> {
        primitives: vec![
            (18.731136960, 0.03349460434),
            (2.8253943650, 0.23472695350),
            (0.6401216923, 0.81375732610),
        ],
    };
    let gc_h_631gss_2s = GaussianContraction::<f64, f64> {
        primitives: vec![
            (0.1612777588, 1.00000000000),
        ],
    };
    let gc_h_631gss_2p = GaussianContraction::<f64, f64> {
        primitives: vec![
            (1.1000000000, 1.00000000000),
        ],
    };

    let bsc0_1s = BasisShellContraction::<f64, f64> {
        basis_shell: bs_cs.clone(),
        start_index: 0,
        contraction: gc_h_631gss_1s.clone(),
        cart_origin: Point3::new(0.0, 0.0, 0.0),
        k: None,
    };
    let bsc0_2s = BasisShellContraction::<f64, f64> {
        basis_shell: bs_cs.clone(),
        start_index: 1,
        contraction: gc_h_631gss_2s.clone(),
        cart_origin: Point3::new(0.0, 0.0, 0.0),
        k: None,
    };
    let bsc0_2p = BasisShellContraction::<f64, f64> {
        basis_shell: bs_cp.clone(),
        start_index: 2,
        contraction: gc_h_631gss_2p.clone(),
        cart_origin: Point3::new(0.0, 0.0, 0.0),
        k: None,
    };
    let bsc1_1s = BasisShellContraction::<f64, f64> {
        basis_shell: bs_cs.clone(),
        start_index: 5,
        contraction: gc_h_631gss_1s,
        cart_origin: Point3::new(1.0, 1.0, 1.0),
        k: None,
    };
    let bsc1_2s = BasisShellContraction::<f64, f64> {
        basis_shell: bs_cs.clone(),
        start_index: 6,
        contraction: gc_h_631gss_2s,
        cart_origin: Point3::new(1.0, 1.0, 1.0),
        k: None,
    };
    let bsc1_2p = BasisShellContraction::<f64, f64> {
        basis_shell: bs_cp.clone(),
        start_index: 7,
        contraction: gc_h_631gss_2p,
        cart_origin: Point3::new(1.0, 1.0, 1.0),
        k: None,
    };

    // <01s|:>
    let st_01s01s = build_shell_tuple![(&bsc0_1s, true), (&bsc0_1s, false); f64];
    let ovs_01s01s = st_01s01s.overlap([0, 0]);
    assert_close_l2!(&ovs_01s01s[0], &array![[1.0]], 1e-6);

    let st_01s02s = build_shell_tuple![(&bsc0_1s, true), (&bsc0_2s, false); f64];
    let ovs_01s02s = st_01s02s.overlap([0, 0]);
    assert_close_l2!(&ovs_01s02s[0], &array![[0.6582920]], 1e-6);

    let st_01s02p = build_shell_tuple![(&bsc0_1s, true), (&bsc0_2p, false); f64];
    let ovs_01s02p = st_01s02p.overlap([0, 0]);
    assert_close_l2!(&ovs_01s02p[0], &array![[0.0, 0.0, 0.0]], 1e-6);

    let st_01s11s = build_shell_tuple![(&bsc0_1s, true), (&bsc1_1s, false); f64];
    let ovs_01s11s = st_01s11s.overlap([0, 0]);
    assert_close_l2!(&ovs_01s11s[0], &array![[0.3107063]], 1e-6);

    let st_01s12s = build_shell_tuple![(&bsc0_1s, true), (&bsc1_2s, false); f64];
    let ovs_01s12s = st_01s12s.overlap([0, 0]);
    assert_close_l2!(&ovs_01s12s[0], &array![[0.4437869]], 1e-6);

    let st_01s12p = build_shell_tuple![(&bsc0_1s, true), (&bsc1_2p, false); f64];
    let ovs_01s12p = st_01s12p.overlap([0, 0]);
    assert_close_l2!(&ovs_01s12p[0], &array![[-0.2056149, -0.2056149, -0.2056149]], 1e-6);

    // <02s|:>
    let st_02s02s = build_shell_tuple![(&bsc0_2s, true), (&bsc0_2s, false); f64];
    let ovs_02s02s = st_02s02s.overlap([0, 0]);
    assert_close_l2!(&ovs_02s02s[0], &array![[1.0]], 1e-6);

    let st_02s02p = build_shell_tuple![(&bsc0_2s, true), (&bsc0_2p, false); f64];
    let ovs_02s02p = st_02s02p.overlap([0, 0]);
    assert_close_l2!(&ovs_02s02p[0], &array![[0.0, 0.0, 0.0]], 1e-6);

    let st_02s11s = build_shell_tuple![(&bsc0_2s, true), (&bsc1_1s, false); f64];
    let ovs_02s11s = st_02s11s.overlap([0, 0]);
    assert_close_l2!(&ovs_02s11s[0], &array![[0.4437869]], 1e-6);

    let st_02s12s = build_shell_tuple![(&bsc0_2s, true), (&bsc1_2s, false); f64];
    let ovs_02s12s = st_02s12s.overlap([0, 0]);
    assert_close_l2!(&ovs_02s12s[0], &array![[0.7851216]], 1e-6);

    let st_02s12p = build_shell_tuple![(&bsc0_2s, true), (&bsc1_2p, false); f64];
    let ovs_02s12p = st_02s12p.overlap([0, 0]);
    assert_close_l2!(&ovs_02s12p[0], &array![[-0.0960035, -0.0960035, -0.0960035]], 1e-6);

    // <02p|:>
    let st_02p02p = build_shell_tuple![(&bsc0_2p, true), (&bsc0_2p, false); f64];
    let ovs_02p02p = st_02p02p.overlap([0, 0]);
    assert_close_l2!(
        &ovs_02p02p[0],
        &array![
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        1e-6
    );

    let st_02p11s = build_shell_tuple![(&bsc0_2p, true), (&bsc1_1s, false); f64];
    let ovs_02p11s = st_02p11s.overlap([0, 0]);
    assert_close_l2!(
        &ovs_02p11s[0],
        &array![
            [0.2056149],
            [0.2056149],
            [0.2056149],
        ],
        1e-6
    );

    let st_02p12s = build_shell_tuple![(&bsc0_2p, true), (&bsc1_2s, false); f64];
    let ovs_02p12s = st_02p12s.overlap([0, 0]);
    assert_close_l2!(
        &ovs_02p12s[0],
        &array![
            [0.0960035],
            [0.0960035],
            [0.0960035],
        ],
        1e-6
    );

    let st_02p12p = build_shell_tuple![(&bsc0_2p, true), (&bsc1_2p, false); f64];
    let ovs_02p12p = st_02p12p.overlap([0, 0]);
    assert_close_l2!(
        &ovs_02p12p[0],
        &array![
            [-0.0192050, -0.2112549, -0.2112549],
            [-0.2112549, -0.0192050, -0.2112549],
            [-0.2112549, -0.2112549, -0.0192050],
        ],
        1e-6
    );
}
