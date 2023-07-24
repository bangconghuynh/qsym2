use byteorder::LittleEndian;
use nalgebra::{Point3, Vector3};
use ndarray::{array, Array2};
use ndarray_linalg::assert_close_l2;
use num_complex::Complex;

use crate::auxiliary::molecule::Molecule;
use crate::basis::ao::*;
use crate::basis::ao_integrals::*;
use crate::io::numeric::NumericReader;

const ROOT: &str = env!("CARGO_MANIFEST_DIR");

type C128 = Complex<f64>;

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
            .flat_map(|(_, _, equiv_perms)| equiv_perms)
            .count(),
        16
    );
    assert_eq!(stc.function_all_shell_shape, [9, 3, 9, 9, 9]);
}

#[test]
fn test_integrals_shell_tuple_collection_overlap_2c_h2() {
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

    let bscs = [&bsc0, &bsc1];
    let stc = build_shell_tuple_collection![
        <s1, s2>;
        true, false;
        bscs, bscs;
        f64
    ];
    let ovs = stc.overlap([0, 0]);

    #[rustfmt::skip]
    assert_close_l2!(
        &ovs[0],
        &array![
            [1.0000000, 0.7965883],
            [0.7965883, 1.0000000],
        ],
        1e-7
    );
}

#[test]
fn test_integrals_shell_tuple_collection_overlap_2c_bf3() {
    // ~~~~~~~~~~~~~~~~~
    // BF3, cc-pVTZ
    // Reference: libint
    // ~~~~~~~~~~~~~~~~~
    let bs_ps = BasisShell::new(0, ShellOrder::Pure(PureOrder::increasingm(0)));
    let bs_pp = BasisShell::new(1, ShellOrder::Pure(PureOrder::increasingm(1)));
    let bs_pd = BasisShell::new(2, ShellOrder::Pure(PureOrder::increasingm(2)));
    let bs_pf = BasisShell::new(3, ShellOrder::Pure(PureOrder::increasingm(3)));

    let gc_b_ccpvtz_1s = GaussianContraction::<f64, f64> {
        primitives: vec![
            (5.473000e+03, 5.550000e-04),
            (8.209000e+02, 4.291000e-03),
            (1.868000e+02, 2.194900e-02),
            (5.283000e+01, 8.444100e-02),
            (1.708000e+01, 2.385570e-01),
            (5.999000e+00, 4.350720e-01),
            (2.208000e+00, 3.419550e-01),
            (5.879000e-01, 3.685600e-02),
            (2.415000e-01, -9.545000e-03),
            (8.610000e-02, 2.368000e-03),
        ],
    };
    let gc_b_ccpvtz_2s = GaussianContraction::<f64, f64> {
        primitives: vec![
            (5.473000e+03, -1.120000e-04),
            (8.209000e+02, -8.680000e-04),
            (1.868000e+02, -4.484000e-03),
            (5.283000e+01, -1.768300e-02),
            (1.708000e+01, -5.363900e-02),
            (5.999000e+00, -1.190050e-01),
            (2.208000e+00, -1.658240e-01),
            (5.879000e-01, 1.201070e-01),
            (2.415000e-01, 5.959810e-01),
            (8.610000e-02, 4.110210e-01),
        ],
    };
    let gc_b_ccpvtz_3s = GaussianContraction::<f64, f64> {
        primitives: vec![(5.879000e-01, 1.000000e+00)],
    };
    let gc_b_ccpvtz_4s = GaussianContraction::<f64, f64> {
        primitives: vec![(8.610000e-02, 1.000000e+00)],
    };
    let gc_b_ccpvtz_2p = GaussianContraction::<f64, f64> {
        primitives: vec![
            (1.205000e+01, 1.311800e-02),
            (2.613000e+00, 7.989600e-02),
            (7.475000e-01, 2.772750e-01),
            (2.385000e-01, 5.042700e-01),
            (7.698000e-02, 3.536800e-01),
        ],
    };
    let gc_b_ccpvtz_3p = GaussianContraction::<f64, f64> {
        primitives: vec![(2.385000e-01, 1.000000e+00)],
    };
    let gc_b_ccpvtz_4p = GaussianContraction::<f64, f64> {
        primitives: vec![(7.698000e-02, 1.000000e+00)],
    };
    let gc_b_ccpvtz_3d = GaussianContraction::<f64, f64> {
        primitives: vec![(6.610000e-01, 1.000000e+00)],
    };
    let gc_b_ccpvtz_4d = GaussianContraction::<f64, f64> {
        primitives: vec![(1.990000e-01, 1.000000e+00)],
    };
    let gc_b_ccpvtz_4f = GaussianContraction::<f64, f64> {
        primitives: vec![(4.900000e-01, 1.000000e+00)],
    };

    let gc_f_ccpvtz_1s = GaussianContraction::<f64, f64> {
        primitives: vec![
            (1.950000e+04, 5.070000e-04),
            (2.923000e+03, 3.923000e-03),
            (6.645000e+02, 2.020000e-02),
            (1.875000e+02, 7.901000e-02),
            (6.062000e+01, 2.304390e-01),
            (2.142000e+01, 4.328720e-01),
            (7.950000e+00, 3.499640e-01),
            (2.257000e+00, 4.323300e-02),
            (8.815000e-01, -7.892000e-03),
            (3.041000e-01, 2.384000e-03),
        ],
    };
    let gc_f_ccpvtz_2s = GaussianContraction::<f64, f64> {
        primitives: vec![
            (1.950000e+04, -1.170000e-04),
            (2.923000e+03, -9.120000e-04),
            (6.645000e+02, -4.717000e-03),
            (1.875000e+02, -1.908600e-02),
            (6.062000e+01, -5.965500e-02),
            (2.142000e+01, -1.400100e-01),
            (7.950000e+00, -1.767820e-01),
            (2.257000e+00, 1.716250e-01),
            (8.815000e-01, 6.050430e-01),
            (3.041000e-01, 3.695120e-01),
        ],
    };
    let gc_f_ccpvtz_3s = GaussianContraction::<f64, f64> {
        primitives: vec![(2.257000e+00, 1.000000e+00)],
    };
    let gc_f_ccpvtz_4s = GaussianContraction::<f64, f64> {
        primitives: vec![(3.041000e-01, 1.000000e+00)],
    };
    let gc_f_ccpvtz_2p = GaussianContraction::<f64, f64> {
        primitives: vec![
            (4.388000e+01, 1.666500e-02),
            (9.926000e+00, 1.044720e-01),
            (2.930000e+00, 3.172600e-01),
            (9.132000e-01, 4.873430e-01),
            (2.672000e-01, 3.346040e-01),
        ],
    };
    let gc_f_ccpvtz_3p = GaussianContraction::<f64, f64> {
        primitives: vec![(9.132000e-01, 1.000000e+00)],
    };
    let gc_f_ccpvtz_4p = GaussianContraction::<f64, f64> {
        primitives: vec![(2.672000e-01, 1.000000e+00)],
    };
    let gc_f_ccpvtz_3d = GaussianContraction::<f64, f64> {
        primitives: vec![(3.107000e+00, 1.000000e+00)],
    };
    let gc_f_ccpvtz_4d = GaussianContraction::<f64, f64> {
        primitives: vec![(8.550000e-01, 1.000000e+00)],
    };
    let gc_f_ccpvtz_4f = GaussianContraction::<f64, f64> {
        primitives: vec![(1.917000e+00, 1.000000e+00)],
    };

    let bscs = vec![
        // B0
        BasisShellContraction::<f64, f64> {
            basis_shell: bs_ps.clone(),
            start_index: 0,
            contraction: gc_b_ccpvtz_1s.clone(),
            cart_origin: Point3::new(0.0, 0.0, 0.0),
            k: None,
        },
        BasisShellContraction::<f64, f64> {
            basis_shell: bs_ps.clone(),
            start_index: 1,
            contraction: gc_b_ccpvtz_2s.clone(),
            cart_origin: Point3::new(0.0, 0.0, 0.0),
            k: None,
        },
        BasisShellContraction::<f64, f64> {
            basis_shell: bs_ps.clone(),
            start_index: 2,
            contraction: gc_b_ccpvtz_3s.clone(),
            cart_origin: Point3::new(0.0, 0.0, 0.0),
            k: None,
        },
        BasisShellContraction::<f64, f64> {
            basis_shell: bs_ps.clone(),
            start_index: 3,
            contraction: gc_b_ccpvtz_4s.clone(),
            cart_origin: Point3::new(0.0, 0.0, 0.0),
            k: None,
        },
        BasisShellContraction::<f64, f64> {
            basis_shell: bs_pp.clone(),
            start_index: 4,
            contraction: gc_b_ccpvtz_2p.clone(),
            cart_origin: Point3::new(0.0, 0.0, 0.0),
            k: None,
        },
        BasisShellContraction::<f64, f64> {
            basis_shell: bs_pp.clone(),
            start_index: 7,
            contraction: gc_b_ccpvtz_3p.clone(),
            cart_origin: Point3::new(0.0, 0.0, 0.0),
            k: None,
        },
        BasisShellContraction::<f64, f64> {
            basis_shell: bs_pp.clone(),
            start_index: 10,
            contraction: gc_b_ccpvtz_4p.clone(),
            cart_origin: Point3::new(0.0, 0.0, 0.0),
            k: None,
        },
        BasisShellContraction::<f64, f64> {
            basis_shell: bs_pd.clone(),
            start_index: 13,
            contraction: gc_b_ccpvtz_3d.clone(),
            cart_origin: Point3::new(0.0, 0.0, 0.0),
            k: None,
        },
        BasisShellContraction::<f64, f64> {
            basis_shell: bs_pd.clone(),
            start_index: 18,
            contraction: gc_b_ccpvtz_4d.clone(),
            cart_origin: Point3::new(0.0, 0.0, 0.0),
            k: None,
        },
        BasisShellContraction::<f64, f64> {
            basis_shell: bs_pf.clone(),
            start_index: 23,
            contraction: gc_b_ccpvtz_4f.clone(),
            cart_origin: Point3::new(0.0, 0.0, 0.0),
            k: None,
        },
        // F1
        BasisShellContraction::<f64, f64> {
            basis_shell: bs_ps.clone(),
            start_index: 30,
            contraction: gc_f_ccpvtz_1s.clone(),
            cart_origin: Point3::new(0.7221259, -1.2507587, 0.0) * 1.8897259886,
            k: None,
        },
        BasisShellContraction::<f64, f64> {
            basis_shell: bs_ps.clone(),
            start_index: 31,
            contraction: gc_f_ccpvtz_2s.clone(),
            cart_origin: Point3::new(0.7221259, -1.2507587, 0.0) * 1.8897259886,
            k: None,
        },
        BasisShellContraction::<f64, f64> {
            basis_shell: bs_ps.clone(),
            start_index: 32,
            contraction: gc_f_ccpvtz_3s.clone(),
            cart_origin: Point3::new(0.7221259, -1.2507587, 0.0) * 1.8897259886,
            k: None,
        },
        BasisShellContraction::<f64, f64> {
            basis_shell: bs_ps.clone(),
            start_index: 33,
            contraction: gc_f_ccpvtz_4s.clone(),
            cart_origin: Point3::new(0.7221259, -1.2507587, 0.0) * 1.8897259886,
            k: None,
        },
        BasisShellContraction::<f64, f64> {
            basis_shell: bs_pp.clone(),
            start_index: 34,
            contraction: gc_f_ccpvtz_2p.clone(),
            cart_origin: Point3::new(0.7221259, -1.2507587, 0.0) * 1.8897259886,
            k: None,
        },
        BasisShellContraction::<f64, f64> {
            basis_shell: bs_pp.clone(),
            start_index: 37,
            contraction: gc_f_ccpvtz_3p.clone(),
            cart_origin: Point3::new(0.7221259, -1.2507587, 0.0) * 1.8897259886,
            k: None,
        },
        BasisShellContraction::<f64, f64> {
            basis_shell: bs_pp.clone(),
            start_index: 40,
            contraction: gc_f_ccpvtz_4p.clone(),
            cart_origin: Point3::new(0.7221259, -1.2507587, 0.0) * 1.8897259886,
            k: None,
        },
        BasisShellContraction::<f64, f64> {
            basis_shell: bs_pd.clone(),
            start_index: 43,
            contraction: gc_f_ccpvtz_3d.clone(),
            cart_origin: Point3::new(0.7221259, -1.2507587, 0.0) * 1.8897259886,
            k: None,
        },
        BasisShellContraction::<f64, f64> {
            basis_shell: bs_pd.clone(),
            start_index: 48,
            contraction: gc_f_ccpvtz_4d.clone(),
            cart_origin: Point3::new(0.7221259, -1.2507587, 0.0) * 1.8897259886,
            k: None,
        },
        BasisShellContraction::<f64, f64> {
            basis_shell: bs_pf.clone(),
            start_index: 53,
            contraction: gc_f_ccpvtz_4f.clone(),
            cart_origin: Point3::new(0.7221259, -1.2507587, 0.0) * 1.8897259886,
            k: None,
        },
        // F2
        BasisShellContraction::<f64, f64> {
            basis_shell: bs_ps.clone(),
            start_index: 60,
            contraction: gc_f_ccpvtz_1s.clone(),
            cart_origin: Point3::new(0.7221259, 1.2507587, 0.0) * 1.8897259886,
            k: None,
        },
        BasisShellContraction::<f64, f64> {
            basis_shell: bs_ps.clone(),
            start_index: 61,
            contraction: gc_f_ccpvtz_2s.clone(),
            cart_origin: Point3::new(0.7221259, 1.2507587, 0.0) * 1.8897259886,
            k: None,
        },
        BasisShellContraction::<f64, f64> {
            basis_shell: bs_ps.clone(),
            start_index: 62,
            contraction: gc_f_ccpvtz_3s.clone(),
            cart_origin: Point3::new(0.7221259, 1.2507587, 0.0) * 1.8897259886,
            k: None,
        },
        BasisShellContraction::<f64, f64> {
            basis_shell: bs_ps.clone(),
            start_index: 63,
            contraction: gc_f_ccpvtz_4s.clone(),
            cart_origin: Point3::new(0.7221259, 1.2507587, 0.0) * 1.8897259886,
            k: None,
        },
        BasisShellContraction::<f64, f64> {
            basis_shell: bs_pp.clone(),
            start_index: 64,
            contraction: gc_f_ccpvtz_2p.clone(),
            cart_origin: Point3::new(0.7221259, 1.2507587, 0.0) * 1.8897259886,
            k: None,
        },
        BasisShellContraction::<f64, f64> {
            basis_shell: bs_pp.clone(),
            start_index: 67,
            contraction: gc_f_ccpvtz_3p.clone(),
            cart_origin: Point3::new(0.7221259, 1.2507587, 0.0) * 1.8897259886,
            k: None,
        },
        BasisShellContraction::<f64, f64> {
            basis_shell: bs_pp.clone(),
            start_index: 70,
            contraction: gc_f_ccpvtz_4p.clone(),
            cart_origin: Point3::new(0.7221259, 1.2507587, 0.0) * 1.8897259886,
            k: None,
        },
        BasisShellContraction::<f64, f64> {
            basis_shell: bs_pd.clone(),
            start_index: 73,
            contraction: gc_f_ccpvtz_3d.clone(),
            cart_origin: Point3::new(0.7221259, 1.2507587, 0.0) * 1.8897259886,
            k: None,
        },
        BasisShellContraction::<f64, f64> {
            basis_shell: bs_pd.clone(),
            start_index: 78,
            contraction: gc_f_ccpvtz_4d.clone(),
            cart_origin: Point3::new(0.7221259, 1.2507587, 0.0) * 1.8897259886,
            k: None,
        },
        BasisShellContraction::<f64, f64> {
            basis_shell: bs_pf.clone(),
            start_index: 83,
            contraction: gc_f_ccpvtz_4f.clone(),
            cart_origin: Point3::new(0.7221259, 1.2507587, 0.0) * 1.8897259886,
            k: None,
        },
        // F3
        BasisShellContraction::<f64, f64> {
            basis_shell: bs_ps.clone(),
            start_index: 90,
            contraction: gc_f_ccpvtz_1s.clone(),
            cart_origin: Point3::new(-1.4442518, 0.0, 0.0) * 1.8897259886,
            k: None,
        },
        BasisShellContraction::<f64, f64> {
            basis_shell: bs_ps.clone(),
            start_index: 91,
            contraction: gc_f_ccpvtz_2s.clone(),
            cart_origin: Point3::new(-1.4442518, 0.0, 0.0) * 1.8897259886,
            k: None,
        },
        BasisShellContraction::<f64, f64> {
            basis_shell: bs_ps.clone(),
            start_index: 92,
            contraction: gc_f_ccpvtz_3s.clone(),
            cart_origin: Point3::new(-1.4442518, 0.0, 0.0) * 1.8897259886,
            k: None,
        },
        BasisShellContraction::<f64, f64> {
            basis_shell: bs_ps.clone(),
            start_index: 93,
            contraction: gc_f_ccpvtz_4s.clone(),
            cart_origin: Point3::new(-1.4442518, 0.0, 0.0) * 1.8897259886,
            k: None,
        },
        BasisShellContraction::<f64, f64> {
            basis_shell: bs_pp.clone(),
            start_index: 94,
            contraction: gc_f_ccpvtz_2p.clone(),
            cart_origin: Point3::new(-1.4442518, 0.0, 0.0) * 1.8897259886,
            k: None,
        },
        BasisShellContraction::<f64, f64> {
            basis_shell: bs_pp.clone(),
            start_index: 97,
            contraction: gc_f_ccpvtz_3p.clone(),
            cart_origin: Point3::new(-1.4442518, 0.0, 0.0) * 1.8897259886,
            k: None,
        },
        BasisShellContraction::<f64, f64> {
            basis_shell: bs_pp.clone(),
            start_index: 100,
            contraction: gc_f_ccpvtz_4p.clone(),
            cart_origin: Point3::new(-1.4442518, 0.0, 0.0) * 1.8897259886,
            k: None,
        },
        BasisShellContraction::<f64, f64> {
            basis_shell: bs_pd.clone(),
            start_index: 103,
            contraction: gc_f_ccpvtz_3d.clone(),
            cart_origin: Point3::new(-1.4442518, 0.0, 0.0) * 1.8897259886,
            k: None,
        },
        BasisShellContraction::<f64, f64> {
            basis_shell: bs_pd.clone(),
            start_index: 108,
            contraction: gc_f_ccpvtz_4d.clone(),
            cart_origin: Point3::new(-1.4442518, 0.0, 0.0) * 1.8897259886,
            k: None,
        },
        BasisShellContraction::<f64, f64> {
            basis_shell: bs_pf.clone(),
            start_index: 113,
            contraction: gc_f_ccpvtz_4f.clone(),
            cart_origin: Point3::new(-1.4442518, 0.0, 0.0) * 1.8897259886,
            k: None,
        },
    ];

    let bscs_ref = bscs.iter().collect::<Vec<_>>();
    let stc = build_shell_tuple_collection![
        <s1, s2>;
        true, false;
        bscs_ref, bscs_ref;
        f64
    ];
    let ovs = stc.overlap([0, 0]);

    let sao_v = NumericReader::<_, LittleEndian, f64>::from_file(format!(
        "{ROOT}/tests/binaries/bf3_sao/sao_libint"
    ))
    .unwrap()
    .collect::<Vec<_>>();
    let sao = Array2::from_shape_vec((120, 120), sao_v).unwrap();
    #[rustfmt::skip]
    assert_close_l2!(
        &ovs[0],
        &sao,
        1e-5
    );
}

#[test]
fn test_integrals_shell_tuple_collection_overlap_2c_benzene_rest_api() {
    // ~~~~~~~~~~~~~~~~~
    // Benzene, cc-pVQZ
    // Reference: libint
    // ~~~~~~~~~~~~~~~~~
    let mol = Molecule::from_xyz(&format!("{ROOT}/tests/xyz/benzene.xyz"), 1e-7);

    let bscs =
        BasisShellContraction::<f64, f64>::from_bse(&mol, "cc-pVQZ", true, true, 0, false).unwrap();
    let bscs_ref = bscs.iter().collect::<Vec<_>>();
    use std::time::Instant;
    let now = Instant::now();
    let stc = build_shell_tuple_collection![
        <s1, s2>;
        true, false;
        bscs_ref, bscs_ref;
        f64
    ];
    let ovs = stc.overlap([0, 0]);
    let elapsed_time = now.elapsed();
    println!("Took: {}", elapsed_time.as_nanos());
    let sao_v = NumericReader::<_, LittleEndian, f64>::from_file(format!(
        "{ROOT}/tests/binaries/benzene_sao/sao_libint_opt"
    ))
    .unwrap()
    .collect::<Vec<_>>();
    let sao = Array2::from_shape_vec((630, 630), sao_v).unwrap();
    #[rustfmt::skip]
    assert_close_l2!(
        &ovs[0],
        &sao,
        1e-5
    );
}
