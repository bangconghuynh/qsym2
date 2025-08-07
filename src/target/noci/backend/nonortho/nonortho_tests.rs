use indexmap::IndexSet;
use ndarray::{array, Array1, Array2};
use ndarray_linalg::assert::close_l2;
use num_complex::{Complex, ComplexFloat};

use crate::target::noci::backend::nonortho::{calc_lowdin_pairing, CanonicalOrthogonalisable};

#[test]
fn test_calc_lowdin_pairing_trivial() {
    #[rustfmt::skip]
    let cw = array![
        [Complex::new(1.0, 0.5), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0), Complex::new(0.5, 0.5), Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(1.0, 0.0)],
        [Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)],
    ];
    #[rustfmt::skip]
    let cx = array![
        [Complex::new(1.0, 0.5), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0), Complex::new(0.0, 1.0), Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 1.0)],
        [Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)],
        [Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)],
    ];
    let sao = Array2::<Complex<f64>>::eye(6);

    let lowdinpairedcs_conv =
        calc_lowdin_pairing(&cw.view(), &cx.view(), &sao.view(), false, 1e-12, 1e-7).unwrap();
    let lowdin_ovs_conv = lowdinpairedcs_conv.lowdin_overlaps;
    close_l2(
        &Array1::from_vec(lowdin_ovs_conv),
        &array![
            Complex::new(1.0, 0.5).conj() * Complex::new(1.0, 0.5),
            Complex::new(0.5, 0.5).conj() * Complex::new(0.0, 1.0),
            Complex::ZERO,
        ],
        1e-7,
    );
    assert_eq!(lowdinpairedcs_conv.zero_indices, IndexSet::from([2]));

    let lowdinpairedcs_holo =
        calc_lowdin_pairing(&cw.view(), &cx.view(), &sao.view(), true, 1e-12, 1e-7).unwrap();
    let lowdin_ovs_holo = lowdinpairedcs_holo.lowdin_overlaps;
    close_l2(
        &Array1::from_vec(lowdin_ovs_holo),
        &array![
            Complex::new(1.0, 0.5) * Complex::new(1.0, 0.5),
            Complex::new(0.5, 0.5) * Complex::new(0.0, 1.0),
            Complex::ZERO,
        ],
        1e-7,
    );
    assert_eq!(lowdinpairedcs_holo.zero_indices, IndexSet::from([2]));
}

#[test]
fn test_calc_lowdin_pairing_nontrivial() {
    #[rustfmt::skip]
    let cw = array![
        [Complex::new(1.0,  0.5), Complex::new(0.0,  0.0), Complex::new(0.0, 1.0)],
        [Complex::new(0.5,  0.0), Complex::new(0.5,  0.5), Complex::new(3.0, 0.0)],
        [Complex::new(0.2,  0.2), Complex::new(0.1,  0.0), Complex::new(0.0, 2.0)],
        [Complex::new(0.4,  0.3), Complex::new(0.0, -1.0), Complex::new(1.0, 0.0)],
        [Complex::new(0.5, -0.1), Complex::new(0.7,  0.0), Complex::new(0.0, 4.0)],
        [Complex::new(1.0,  0.1), Complex::new(0.7,  0.2), Complex::new(0.5, 0.0)],
    ];
    #[rustfmt::skip]
    let cx = array![
        [Complex::new(2.0, 0.5), Complex::new(0.5,  0.0), Complex::new(0.0, 0.0)],
        [Complex::new(0.2, 0.0), Complex::new(0.2,  1.0), Complex::new(0.1, 0.0)],
        [Complex::new(0.4, 0.3), Complex::new(0.1,  0.0), Complex::new(0.0, 1.0)],
        [Complex::new(0.6, 0.9), Complex::new(0.8, -0.2), Complex::new(2.0, 0.0)],
        [Complex::new(0.0, 0.8), Complex::new(0.6,  0.2), Complex::new(0.0, 4.0)],
        [Complex::new(0.2, 0.7), Complex::new(0.4,  0.0), Complex::new(0.0, 0.8)],
    ];
    let sao = Array2::<Complex<f64>>::eye(6);

    let lowdinpairedcs_conv =
        calc_lowdin_pairing(&cw.view(), &cx.view(), &sao.view(), false, 1e-12, 1e-7).unwrap();
    let cwt = &lowdinpairedcs_conv.paired_cw;
    let cxt = &lowdinpairedcs_conv.paired_cx;
    let ovt = cwt.t().map(|x| x.conj()).dot(&sao).dot(cxt);
    let ovt_from_lowdinovs =
        Array2::from_diag(&Array1::from_vec(lowdinpairedcs_conv.lowdin_overlaps));
    close_l2(&ovt, &ovt_from_lowdinovs, 1e-7);
    assert_eq!(lowdinpairedcs_conv.zero_indices, IndexSet::new());

    let lowdinpairedcs_holo =
        calc_lowdin_pairing(&cw.view(), &cx.view(), &sao.view(), true, 1e-12, 1e-7).unwrap();
    let cwt = &lowdinpairedcs_holo.paired_cw;
    let cxt = &lowdinpairedcs_holo.paired_cx;
    let ovt = cwt.t().dot(&sao).dot(cxt);
    let ovt_from_lowdinovs =
        Array2::from_diag(&Array1::from_vec(lowdinpairedcs_holo.lowdin_overlaps));
    close_l2(&ovt, &ovt_from_lowdinovs, 1e-7);
    assert_eq!(lowdinpairedcs_conv.zero_indices, IndexSet::new());
}

#[test]
fn test_canonical_orthogonalisation_real_symmetric() {
    // Full-rank
    #[rustfmt::skip]
    let smat = array![
        [6.0, 2.0, 1.0, 0.5, 0.0],
        [2.0, 5.0, 2.0, 1.0, 0.5],
        [1.0, 2.0, 4.0, 1.5, 1.0],
        [0.5, 1.0, 1.5, 3.0, 1.0],
        [0.0, 0.5, 1.0, 1.0, 2.0],
    ];
    let xmat_res = smat
        .view()
        .calc_canonical_orthogonal_matrix(true, false, 1e-7, 1e-7)
        .unwrap();
    let xmat = xmat_res.xmat();
    let xmat_d = xmat_res.xmat_d();
    let smat_t = xmat_d.dot(&smat).dot(&xmat);
    close_l2(&smat_t, &Array2::eye(5), 1e-7);

    // Full-rank, with negative eigenvalues
    #[rustfmt::skip]
    let smat = array![
        [ 0.0,  2.0, -1.0,  0.0,  1.0],
        [ 2.0, -3.0,  4.0, -1.0,  0.0],
        [-1.0,  4.0,  5.0,  2.0, -2.0],
        [ 0.0, -1.0,  2.0,  3.0,  1.0],
        [ 1.0,  0.0, -2.0,  1.0,  2.0],
    ];
    let xmat_res_res = smat
        .view()
        .calc_canonical_orthogonal_matrix(true, false, 1e-7, 1e-7);
    assert!(xmat_res_res.is_err());

    // Nullity = 1
    #[rustfmt::skip]
    let smat = array![
        [ 2.0, -1.0,  0.0,  0.0, -1.0],
        [-1.0,  2.0, -1.0,  0.0,  0.0],
        [ 0.0, -1.0,  2.0, -1.0,  0.0],
        [ 0.0,  0.0, -1.0,  2.0, -1.0],
        [-1.0,  0.0,  0.0, -1.0,  2.0],
    ];
    let xmat_res = smat
        .view()
        .calc_canonical_orthogonal_matrix(true, false, 1e-7, 1e-7)
        .unwrap();
    let xmat = xmat_res.xmat();
    let xmat_d = xmat_res.xmat_d();
    let smat_t = xmat_d.dot(&smat).dot(&xmat);
    close_l2(&smat_t, &Array2::eye(4), 1e-7);

    // Nullity = 2
    #[rustfmt::skip]
    let smat = array![
        [ 2.0, -1.0,  0.0,  0.0, -1.0,  0.0],
        [-1.0,  2.0, -1.0,  0.0,  0.0,  0.0],
        [ 0.0, -1.0,  2.0, -1.0,  0.0,  0.0],
        [ 0.0,  0.0, -1.0,  2.0, -1.0,  0.0],
        [-1.0,  0.0,  0.0, -1.0,  2.0,  0.0],
        [ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
    ];
    let xmat_res = smat
        .view()
        .calc_canonical_orthogonal_matrix(true, false, 1e-7, 1e-7)
        .unwrap();
    let xmat = xmat_res.xmat();
    let xmat_d = xmat_res.xmat_d();
    let smat_t = xmat_d.dot(&smat).dot(&xmat);
    close_l2(&smat_t, &Array2::eye(4), 1e-7);

    // Nullity = 3
    #[rustfmt::skip]
    let smat = array![
        [ 1.0, -1.0,  0.0,  0.0,  0.0,  0.0],
        [-1.0,  1.0,  0.0,  0.0,  0.0,  0.0],
        [ 0.0,  0.0,  2.0, -1.0, -1.0,  0.0],
        [ 0.0,  0.0, -1.0,  1.0,  0.0,  0.0],
        [ 0.0,  0.0, -1.0,  0.0,  1.0,  0.0],
        [ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
    ];
    let xmat_res = smat
        .view()
        .calc_canonical_orthogonal_matrix(true, false, 1e-7, 1e-7)
        .unwrap();
    let xmat = xmat_res.xmat();
    let xmat_d = xmat_res.xmat_d();
    let smat_t = xmat_d.dot(&smat).dot(&xmat);
    close_l2(&smat_t, &Array2::eye(3), 1e-7);

    // Nullity = 3
    #[rustfmt::skip]
    let smat = array![
        [ 1.0, -1.0,  0.0,  0.0,  0.0,  0.0,  0.0],
        [-1.0,  1.0,  0.0,  0.0,  0.0,  0.0,  0.0],
        [ 0.0,  0.0,  1.0, -1.0,  0.0,  0.0,  0.0],
        [ 0.0,  0.0, -1.0,  1.0,  0.0,  0.0,  0.0],
        [ 0.0,  0.0,  0.0,  0.0,  2.0, -1.0, -1.0],
        [ 0.0,  0.0,  0.0,  0.0, -1.0,  2.0, -1.0],
        [ 0.0,  0.0,  0.0,  0.0, -1.0, -1.0,  2.0],
    ];
    let xmat_res = smat
        .view()
        .calc_canonical_orthogonal_matrix(true, false, 1e-7, 1e-7)
        .unwrap();
    let xmat = xmat_res.xmat();
    let xmat_d = xmat_res.xmat_d();
    let smat_t = xmat_d.dot(&smat).dot(&xmat);
    close_l2(&smat_t, &Array2::eye(4), 1e-7);
}

#[test]
fn test_canonical_orthogonalisation_complex_hermitian() {
    // Full-rank
    #[rustfmt::skip]
    let smat = array![
        [Complex::new(4.0,  0.0), Complex::new(1.0,  2.0), Complex::new(0.0, -1.0), Complex::new(0.0,  0.0), Complex::new(3.0, 1.0)],
        [Complex::new(1.0, -2.0), Complex::new(5.0,  0.0), Complex::new(2.0,  0.0), Complex::new(1.0,  1.0), Complex::new(0.0, 0.0)],
        [Complex::new(0.0,  1.0), Complex::new(2.0,  0.0), Complex::new(3.0,  0.0), Complex::new(1.0, -1.0), Complex::new(0.0, 2.0)],
        [Complex::new(0.0,  0.0), Complex::new(1.0, -1.0), Complex::new(1.0,  1.0), Complex::new(4.0,  0.0), Complex::new(1.0, 0.0)],
        [Complex::new(3.0, -1.0), Complex::new(0.0,  0.0), Complex::new(0.0, -2.0), Complex::new(1.0,  0.0), Complex::new(6.0, 0.0)],
    ];
    let xmat_res = smat
        .view()
        .calc_canonical_orthogonal_matrix(false, false, 1e-7, 1e-7)
        .unwrap();
    let xmat = xmat_res.xmat();
    let xmat_d = xmat_res.xmat_d();
    let smat_t = xmat_d.dot(&smat).dot(&xmat);
    // close_l2(&smat_t, &Array2::eye(5), 1e-7);
    close_l2(
        &smat_t.map(|v| v.conj()).t().dot(&smat_t),
        &Array2::eye(5),
        1e-7,
    );

    // Full-rank, with negative eigenvalues
    #[rustfmt::skip]
    let smat = array![
        [Complex::new(2.0,  0.0), Complex::new( 1.0,  1.0), Complex::new( 0.0, -1.0), Complex::new( 0.0,  0.0), Complex::new(1.0,  0.0)],
        [Complex::new(1.0, -1.0), Complex::new( 0.0,  0.0), Complex::new(-1.0,  2.0), Complex::new( 0.0,  0.0), Complex::new(0.0,  0.0)],
        [Complex::new(0.0,  1.0), Complex::new(-1.0, -2.0), Complex::new( 1.0,  0.0), Complex::new( 2.0,  0.0), Complex::new(0.0, -1.0)],
        [Complex::new(0.0,  0.0), Complex::new( 0.0,  0.0), Complex::new( 2.0,  0.0), Complex::new(-1.0,  0.0), Complex::new(1.0,  1.0)],
        [Complex::new(1.0,  0.0), Complex::new( 0.0,  0.0), Complex::new( 0.0,  1.0), Complex::new( 1.0, -1.0), Complex::new(3.0,  0.0)],
    ];
    let xmat_res = smat
        .view()
        .calc_canonical_orthogonal_matrix(false, false, 1e-7, 1e-7)
        .unwrap();
    let xmat = xmat_res.xmat();
    let xmat_d = xmat_res.xmat_d();
    let smat_t = xmat_d.dot(&smat).dot(&xmat);
    close_l2(
        &smat_t.map(|v| v.conj()).t().dot(&smat_t),
        &Array2::eye(5),
        1e-7,
    );

    // Nullity = 1, with negative eigenvalues
    #[rustfmt::skip]
    let smat = array![
        [Complex::new(21.387112,   0.0), Complex::new(14.000000,  2.0), Complex::new(24.000000,   2.0), Complex::new( 33.000000,  6.0), Complex::new(38.000000, 12.0)],
        [Complex::new(14.000000,  -2.0), Complex::new(99.387112,  0.0), Complex::new(90.000000, -10.0), Complex::new( 75.000000, -5.0), Complex::new(64.000000, -6.0)],
        [Complex::new(24.000000,  -2.0), Complex::new(90.000000, 10.0), Complex::new(94.387112,   0.0), Complex::new( 88.000000, 16.0), Complex::new(75.000000,  7.0)],
        [Complex::new(33.000000,  -6.0), Complex::new(75.000000,  5.0), Complex::new(88.000000, -16.0), Complex::new(110.387012,  0.0), Complex::new(86.000000,  0.0)],
        [Complex::new(38.000000, -12.0), Complex::new(64.000000,  6.0), Complex::new(75.000000,  -7.0), Complex::new( 86.000000,  0.0), Complex::new(81.387112,  0.0)],
    ];
    let xmat_res = smat
        .view()
        .calc_canonical_orthogonal_matrix(false, false, 1e-7, 1e-7)
        .unwrap();
    let xmat = xmat_res.xmat();
    let xmat_d = xmat_res.xmat_d();
    let smat_t = xmat_d.dot(&smat).dot(&xmat);
    close_l2(
        &smat_t.map(|v| v.conj()).t().dot(&smat_t),
        &Array2::eye(4),
        1e-7,
    );

    // Nullity = 2
    #[rustfmt::skip]
    let smat = array![
        [ Complex::new(24.0,   0.0), Complex::new( 14.0,  2.0), Complex::new(26.0,   2.0), Complex::new(36.0, 10.0), Complex::new(38.0, 12.0)],
        [ Complex::new(14.0,  -2.0), Complex::new(103.0,  0.0), Complex::new(90.0, -10.0), Complex::new(75.0, -5.0), Complex::new(64.0, -6.0)],
        [ Complex::new(26.0,  -2.0), Complex::new( 90.0, 10.0), Complex::new(94.0,   0.0), Complex::new(82.0,  8.0), Complex::new(75.0,  7.0)],
        [ Complex::new(36.0, -10.0), Complex::new( 75.0,  5.0), Complex::new(82.0,  -8.0), Complex::new(89.0,  0.0), Complex::new(86.0,  0.0)],
        [ Complex::new(38.0, -12.0), Complex::new( 64.0,  6.0), Complex::new(75.0,  -7.0), Complex::new(86.0,  0.0), Complex::new(85.0,  0.0)],
    ];
    let xmat_res = smat
        .view()
        .calc_canonical_orthogonal_matrix(false, false, 1e-7, 1e-7)
        .unwrap();
    let xmat = xmat_res.xmat();
    let xmat_d = xmat_res.xmat_d();
    let smat_t = xmat_d.dot(&smat).dot(&xmat);
    close_l2(
        &smat_t.map(|v| v.conj()).t().dot(&smat_t),
        &Array2::eye(3),
        1e-7,
    );
}

#[test]
fn test_canonical_orthogonalisation_complex_symmetric() {
    // Full-rank
    #[rustfmt::skip]
    let smat = array![
        [Complex::new(55.0, 20.0), Complex::new( 62.0,  2.0), Complex::new( 26.0,  10.0), Complex::new(21.0,  17.0), Complex::new( 38.0,   5.0)],
        [Complex::new(62.0,  2.0), Complex::new(159.0,  2.0), Complex::new( 46.0,  -4.0), Complex::new(58.0,  15.0), Complex::new( 75.0,  -6.0)],
        [Complex::new(26.0, 10.0), Complex::new( 46.0, -4.0), Complex::new(-41.0, -60.0), Complex::new(30.0,  24.0), Complex::new( 48.0, -43.0)],
        [Complex::new(21.0, 17.0), Complex::new( 58.0, 15.0), Complex::new( 30.0,  24.0), Complex::new(86.0, -12.0), Complex::new( 82.0,  10.0)],
        [Complex::new(38.0,  5.0), Complex::new( 75.0, -6.0), Complex::new( 48.0, -43.0), Complex::new(82.0,  10.0), Complex::new(122.0,   0.0)],
    ];
    let xmat_res = smat
        .view()
        .calc_canonical_orthogonal_matrix(true, false, 1e-7, 1e-7)
        .unwrap();
    let xmat = xmat_res.xmat();
    let xmat_d = xmat_res.xmat_d();
    let smat_t = xmat_d.dot(&smat).dot(&xmat);
    close_l2(&smat_t, &Array2::eye(5), 1e-7);

    // Nullity = 1
    #[rustfmt::skip]
    let smat = array![
        [Complex::new(40.0, 12.0), Complex::new( 65.0,  7.0), Complex::new( 18.0,   8.0), Complex::new( 1.0,  12.0), Complex::new(14.0,  -1.0)],
        [Complex::new(65.0,  7.0), Complex::new(159.0,  0.0), Complex::new( 48.0,  -2.0), Complex::new(63.0,  20.0), Complex::new(81.0,   0.0)],
        [Complex::new(18.0,  8.0), Complex::new( 48.0, -2.0), Complex::new(-45.0, -60.0), Complex::new(20.0,  24.0), Complex::new(36.0, -43.0)],
        [Complex::new( 1.0, 12.0), Complex::new( 63.0, 20.0), Complex::new( 20.0,  24.0), Complex::new(61.0, -12.0), Complex::new(52.0,  10.0)],
        [Complex::new(14.0, -1.0), Complex::new( 81.0,  0.0), Complex::new( 36.0, -43.0), Complex::new(52.0,  10.0), Complex::new(86.0,   0.0)],
    ];
    let xmat_res = smat
        .view()
        .calc_canonical_orthogonal_matrix(true, false, 1e-7, 1e-7)
        .unwrap();
    let xmat = xmat_res.xmat();
    let xmat_d = xmat_res.xmat_d();
    let smat_t = xmat_d.dot(&smat).dot(&xmat);
    close_l2(&smat_t, &Array2::eye(4), 1e-7);
}
