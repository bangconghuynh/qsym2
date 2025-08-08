use crate::analysis::EigenvalueComparisonMode;
use ndarray::{array, Array2};
use ndarray_linalg::close_l2;
use num::Complex;
use num_complex::ComplexFloat;

use crate::target::noci::backend::solver::GeneralisedEigenvalueSolvable;

#[test]
fn test_solver_generalised_eigenvalue_problem_real_hermitian_3x3_full_rank_posdef() {
    #[rustfmt::skip]
    let hmat = array![
        [-0.75,  0.10, -0.25],
        [ 0.10, -0.85,  0.14],
        [-0.25,  0.14, -0.90],
    ];
    #[rustfmt::skip]
    let smat = array![
        [1.00, 0.10, 0.40],
        [0.10, 1.00, 0.23],
        [0.40, 0.23, 1.00]
    ];
    let res_canortho = (&hmat.view(), &smat.view())
        .solve_generalised_eigenvalue_problem_with_canonical_orthogonalisation(
            true,
            1e-7,
            1e-7,
            EigenvalueComparisonMode::Real,
        )
        .unwrap();
    close_l2(
        &res_canortho.eigenvalues,
        &array![-1.32487907, -0.86450811, -0.55307039],
        1e-7,
    );
    #[rustfmt::skip]
    close_l2(
        &res_canortho.eigenvectors,
        &array![
            [ 0.11878677,  1.00554779,  0.40661445],
            [ 0.76607214, -0.34074007,  0.59411223],
            [-0.88011825, -0.53873036,  0.42386529],
        ],
        1e-7,
    );
    close_l2(
        &hmat.dot(&res_canortho.eigenvectors),
        &smat
            .dot(&res_canortho.eigenvectors)
            .dot(&Array2::from_diag(&res_canortho.eigenvalues)),
        1e-7,
    );

    let res_ggev = (&hmat.view(), &smat.view())
        .solve_generalised_eigenvalue_problem_with_ggev(
            true,
            1e-7,
            1e-7,
            EigenvalueComparisonMode::Real,
        )
        .unwrap();
    close_l2(
        &res_ggev.eigenvalues,
        &array![-1.32487907, -0.86450811, -0.55307039],
        1e-7,
    );
    #[rustfmt::skip]
    close_l2(
        &res_ggev.eigenvectors,
        &array![
            [ 0.11878677,  1.00554779,  0.40661445],
            [ 0.76607214, -0.34074007,  0.59411223],
            [-0.88011825, -0.53873036,  0.42386529],
        ],
        1e-7,
    );
    close_l2(
        &hmat.dot(&res_ggev.eigenvectors),
        &smat
            .dot(&res_ggev.eigenvectors)
            .dot(&Array2::from_diag(&res_ggev.eigenvalues)),
        1e-7,
    );

    let hmat_c = hmat.map(Complex::from);
    let smat_c = smat.map(Complex::from);
    let res_c_canortho = (&hmat_c.view(), &smat_c.view())
        .solve_generalised_eigenvalue_problem_with_canonical_orthogonalisation(
            false,
            1e-7,
            1e-7,
            EigenvalueComparisonMode::Real,
        )
        .unwrap();
    close_l2(
        &res_c_canortho.eigenvalues,
        &array![-1.32487907, -0.86450811, -0.55307039].map(Complex::from),
        1e-7,
    );
    #[rustfmt::skip]
    close_l2(
        &res_c_canortho.eigenvectors,
        &array![
            [ 0.11878677,  1.00554779,  0.40661445],
            [ 0.76607214, -0.34074007,  0.59411223],
            [-0.88011825, -0.53873036,  0.42386529],
        ].map(Complex::from),
        1e-7,
    );
    close_l2(
        &hmat_c.dot(&res_c_canortho.eigenvectors),
        &smat_c
            .dot(&res_c_canortho.eigenvectors)
            .dot(&Array2::from_diag(&res_c_canortho.eigenvalues)),
        1e-7,
    );

    let res_c_ggev = (&hmat_c.view(), &smat_c.view())
        .solve_generalised_eigenvalue_problem_with_ggev(
            true,
            1e-7,
            1e-7,
            EigenvalueComparisonMode::Real,
        )
        .unwrap();
    close_l2(
        &res_c_ggev.eigenvalues,
        &array![-1.32487907, -0.86450811, -0.55307039].map(Complex::from),
        1e-7,
    );
    #[rustfmt::skip]
    close_l2(
        &res_c_ggev.eigenvectors,
        &array![
            [ 0.11878677,  1.00554779,  0.40661445],
            [ 0.76607214, -0.34074007,  0.59411223],
            [-0.88011825, -0.53873036,  0.42386529],
        ].map(Complex::from),
        1e-7,
    );
    close_l2(
        &hmat_c.dot(&res_c_ggev.eigenvectors),
        &smat_c
            .dot(&res_c_ggev.eigenvectors)
            .dot(&Array2::from_diag(&res_c_ggev.eigenvalues)),
        1e-7,
    );
}

#[test]
fn test_solver_generalised_eigenvalue_problem_real_hermitian_3x3_full_rank_nonposdef() {
    #[rustfmt::skip]
    let hmat = array![
        [-0.75,  0.10, -0.25],
        [ 0.10, -0.85,  0.14],
        [-0.25,  0.14, -0.90],
    ];
    #[rustfmt::skip]
    let smat = array![
        [0.00, 0.10, 0.40],
        [0.10, 0.00, 0.23],
        [0.40, 0.23, 0.00]
    ];

    // Reals
    let res_canortho = (&hmat.view(), &smat.view())
        .solve_generalised_eigenvalue_problem_with_canonical_orthogonalisation(
            true,
            1e-7,
            1e-7,
            EigenvalueComparisonMode::Real,
        );
    // smat not positive-definite and not canonical-orthogonalisable over the reals
    assert!(res_canortho.is_err());

    let res_ggev = (&hmat.view(), &smat.view()).solve_generalised_eigenvalue_problem_with_ggev(
        true,
        1e-7,
        1e-7,
        EigenvalueComparisonMode::Real,
    );
    // Negative squared norms, not normalisable over the reals
    assert!(res_ggev.is_err());

    let hmat_c = hmat.map(Complex::from);
    let smat_c = smat.map(Complex::from);

    // Complex-symmetric interpretation
    let res_cs_canortho = (&hmat_c.view(), &smat_c.view())
        .solve_generalised_eigenvalue_problem_with_canonical_orthogonalisation(
            true,
            1e-7,
            1e-7,
            EigenvalueComparisonMode::Real,
        )
        .unwrap();
    close_l2(
        &res_cs_canortho.eigenvalues,
        &array![-1.721769774, 1.402759736, 11.33939047].map(Complex::from),
        1e-7,
    );
    #[rustfmt::skip]
    close_l2(
        &hmat_c.dot(&res_cs_canortho.eigenvectors),
        &smat_c
            .dot(&res_cs_canortho.eigenvectors)
            .dot(&Array2::from_diag(&res_cs_canortho.eigenvalues)),
        1e-7,
    );

    let res_cs_ggev = (&hmat_c.view(), &smat_c.view())
        .solve_generalised_eigenvalue_problem_with_ggev(
            true,
            1e-7,
            1e-7,
            EigenvalueComparisonMode::Real,
        )
        .unwrap();
    close_l2(
        &res_cs_ggev.eigenvalues,
        &array![-1.721769774, 1.402759736, 11.33939047].map(Complex::from),
        1e-7,
    );
    #[rustfmt::skip]
    close_l2(
        &hmat_c.dot(&res_cs_ggev.eigenvectors),
        &smat_c
            .dot(&res_cs_ggev.eigenvectors)
            .dot(&Array2::from_diag(&res_cs_ggev.eigenvalues)),
        1e-7,
    );

    // Complex-Hermitian interpretation
    let res_ch_canortho = (&hmat_c.view(), &smat_c.view())
        .solve_generalised_eigenvalue_problem_with_canonical_orthogonalisation(
            false,
            1e-7,
            1e-7,
            EigenvalueComparisonMode::Real,
        )
        .unwrap();
    close_l2(
        &res_ch_canortho.eigenvalues,
        &array![-1.721769774, 1.402759736, 11.33939047].map(Complex::from),
        1e-7,
    );
    #[rustfmt::skip]
    close_l2(
        &hmat_c.dot(&res_ch_canortho.eigenvectors),
        &smat_c
            .dot(&res_ch_canortho.eigenvectors)
            .dot(&Array2::from_diag(&res_ch_canortho.eigenvalues)),
        1e-7,
    );

    let res_ch_ggev = (&hmat_c.view(), &smat_c.view())
        .solve_generalised_eigenvalue_problem_with_ggev(
            false,
            1e-7,
            1e-7,
            EigenvalueComparisonMode::Real,
        )
        .unwrap();
    close_l2(
        &res_ch_ggev.eigenvalues,
        &array![-1.721769774, 1.402759736, 11.33939047].map(Complex::from),
        1e-7,
    );
    #[rustfmt::skip]
    close_l2(
        &hmat_c.dot(&res_ch_ggev.eigenvectors),
        &smat_c
            .dot(&res_ch_ggev.eigenvectors)
            .dot(&Array2::from_diag(&res_ch_ggev.eigenvalues)),
        1e-7,
    );
}

#[test]
fn test_solver_generalised_eigenvalue_problem_complex_symmetric_3x3_full_rank() {
    #[rustfmt::skip]
    let hmat_cs = array![
        [Complex::new(-0.75,  0.00), Complex::new( 0.10, 0.25), Complex::new(-0.25, -0.10)],
        [Complex::new( 0.10,  0.25), Complex::new(-0.85, 0.00), Complex::new( 0.14,  0.35)],
        [Complex::new(-0.25, -0.10), Complex::new( 0.14, 0.35), Complex::new(-0.90,  0.00)],
    ];
    #[rustfmt::skip]
    let smat_cs = array![
        [Complex::new( 1.00,  0.00), Complex::new(-0.20, 0.25), Complex::new( 0.66, -0.10)],
        [Complex::new(-0.20,  0.25), Complex::new( 1.00, 0.00), Complex::new( 0.28,  0.15)],
        [Complex::new( 0.66, -0.10), Complex::new( 0.28, 0.15), Complex::new( 1.00,  0.00)],
    ];

    let res_cs_canortho = (&hmat_cs.view(), &smat_cs.view())
        .solve_generalised_eigenvalue_problem_with_canonical_orthogonalisation(
            true,
            1e-7,
            1e-7,
            EigenvalueComparisonMode::Real,
        )
        .unwrap();
    #[rustfmt::skip]
    close_l2(
        &res_cs_canortho.eigenvalues,
        &array![
            Complex::new(-1.9886530570,  1.3732637400),
            Complex::new(-0.7543753691, -0.4092157793),
            Complex::new(-0.5150389758,  0.3509710598)
        ],
        1e-7,
    );
    #[rustfmt::skip]
    close_l2(
        &hmat_cs.dot(&res_cs_canortho.eigenvectors),
        &smat_cs
            .dot(&res_cs_canortho.eigenvectors)
            .dot(&Array2::from_diag(&res_cs_canortho.eigenvalues)),
        1e-7,
    );

    let res_cs_ggev = (&hmat_cs.view(), &smat_cs.view())
        .solve_generalised_eigenvalue_problem_with_ggev(
            true,
            1e-7,
            1e-7,
            EigenvalueComparisonMode::Real,
        )
        .unwrap();
    #[rustfmt::skip]
    close_l2(
        &res_cs_ggev.eigenvalues,
        &array![
            Complex::new(-1.9886530570,  1.3732637400),
            Complex::new(-0.7543753691, -0.4092157793),
            Complex::new(-0.5150389758,  0.3509710598)
        ],
        1e-7,
    );
    #[rustfmt::skip]
    close_l2(
        &hmat_cs.dot(&res_cs_ggev.eigenvectors),
        &smat_cs
            .dot(&res_cs_ggev.eigenvectors)
            .dot(&Array2::from_diag(&res_cs_ggev.eigenvalues)),
        1e-7,
    );

    let res_ch = (&hmat_cs.view(), &smat_cs.view())
        .solve_generalised_eigenvalue_problem_with_canonical_orthogonalisation(
            false,
            1e-7,
            1e-7,
            EigenvalueComparisonMode::Real,
        );
    // hmat_cs and smat_cs are complex-symmetric
    assert!(res_ch.is_err());
}

#[test]
fn test_solver_generalised_eigenvalue_problem_complex_hermitian_3x3_full_rank_posdef() {
    #[rustfmt::skip]
    let hmat_ch = array![
        [Complex::new(-0.75,  0.00), Complex::new( 0.10, 0.25), Complex::new(-0.25, -0.10)],
        [Complex::new( 0.10, -0.25), Complex::new(-0.85, 0.00), Complex::new( 0.14, -0.35)],
        [Complex::new(-0.25,  0.10), Complex::new( 0.14, 0.35), Complex::new(-0.90,  0.00)],
    ];
    #[rustfmt::skip]
    let smat_ch = array![
        [Complex::new( 1.00, 0.00), Complex::new(-0.20, -0.25), Complex::new( 0.66, -0.10)],
        [Complex::new(-0.20, 0.25), Complex::new( 1.00,  0.00), Complex::new( 0.28, -0.15)],
        [Complex::new( 0.66, 0.10), Complex::new( 0.28,  0.15), Complex::new( 1.00,  0.00)],
    ];

    let res_ch_canortho = (&hmat_ch.view(), &smat_ch.view())
        .solve_generalised_eigenvalue_problem_with_canonical_orthogonalisation(
            false,
            1e-7,
            1e-7,
            EigenvalueComparisonMode::Real,
        )
        .unwrap();
    #[rustfmt::skip]
    close_l2(
        &res_ch_canortho.eigenvalues,
        &array![
            Complex::from(-6.0629788080),
            Complex::from(-0.7497197140),
            Complex::from(-0.3651816039)
        ],
        1e-7,
    );
    close_l2(
        &hmat_ch.dot(&res_ch_canortho.eigenvectors),
        &smat_ch
            .dot(&res_ch_canortho.eigenvectors)
            .dot(&Array2::from_diag(&res_ch_canortho.eigenvalues)),
        1e-7,
    );

    let res_ch_ggev = (&hmat_ch.view(), &smat_ch.view())
        .solve_generalised_eigenvalue_problem_with_ggev(
            false,
            1e-7,
            1e-7,
            EigenvalueComparisonMode::Real,
        )
        .unwrap();
    #[rustfmt::skip]
    close_l2(
        &res_ch_ggev.eigenvalues,
        &array![
            Complex::from(-6.0629788080),
            Complex::from(-0.7497197140),
            Complex::from(-0.3651816039)
        ],
        1e-7,
    );
    close_l2(
        &hmat_ch.dot(&res_ch_ggev.eigenvectors),
        &smat_ch
            .dot(&res_ch_ggev.eigenvectors)
            .dot(&Array2::from_diag(&res_ch_ggev.eigenvalues)),
        1e-7,
    );
}

#[test]
fn test_solver_generalised_eigenvalue_problem_real_hermitian_3x3_nullity_1_one_negative_ggev() {
    #[rustfmt::skip]
    let hmat = array![
        [ 4.9192343463, -0.1467367703,  0.7385391666],
        [-0.1467367703,  6.2350102567, -2.9683155770],
        [ 0.7385391666, -2.9683155770,  4.4066218188],
    ];
    // Eigenvalues of smat_ch: -1.640466013, 0.6230519529, 0.0
    #[rustfmt::skip]
    let smat = array![
        [-0.0322296196, -0.1291472238, -0.1886007953],
        [-0.1291472238,  0.0076030298, -0.9970206693],
        [-0.1886007953, -0.9970206693, -0.9927874707]
    ];
    let res_ggev = (&hmat.view(), &smat.view()).solve_generalised_eigenvalue_problem_with_ggev(
        false,
        1e-7,
        1e-7,
        EigenvalueComparisonMode::Modulus,
    );
    assert!(res_ggev.is_err());

    let hmat_ch = hmat.map(Complex::from);
    let smat_ch = smat.map(Complex::from);

    let res_ch_ggev = (&hmat_ch.view(), &smat_ch.view())
        .solve_generalised_eigenvalue_problem_with_ggev(
            false,
            1e-7,
            1e-7,
            EigenvalueComparisonMode::Real,
        )
        .unwrap();
    close_l2(
        &res_ch_ggev.eigenvalues,
        &array![Complex::from(-1.380754093), Complex::from(13.41449558)],
        1e-7,
    );
}

#[test]
fn test_solver_generalised_eigenvalue_problem_complex_hermitian_3x3_nullity_1_one_negative_ggev() {
    #[rustfmt::skip]
    let hmat_ch = array![
        [Complex::new( 4.9192343463,  0.0000000000), Complex::new(-0.1467367703, -1.7329799301), Complex::new( 0.7385391666,  2.3989906789)],
        [Complex::new(-0.1467367703,  1.7329799301), Complex::new( 6.2350102567,  0.0000000000), Complex::new(-2.9683155770, -3.3180237204)],
        [Complex::new( 0.7385391666, -2.3989906789), Complex::new(-2.9683155770,  3.3180237204), Complex::new( 4.4066218188,  0.0000000000)],
    ];
    // Eigenvalues of smat_ch: -2.073861814, 0.4752478556, 0.0
    #[rustfmt::skip]
    let smat_ch = array![
        [Complex::new(-0.2259629188,  0.0000000000), Complex::new(-0.1291472238,  0.4188976270), Complex::new(-0.1886007953, 0.5059039423)],
        [Complex::new(-0.1291472238, -0.4188976270), Complex::new(-0.1861302694,  0.0000000000), Complex::new(-0.9970206693, 0.1972322095)],
        [Complex::new(-0.1886007953, -0.5059039423), Complex::new(-0.9970206693, -0.1972322095), Complex::new(-1.1865207699, 0.0000000000)],
    ];

    let res_ch_ggev = (&hmat_ch.view(), &smat_ch.view())
        .solve_generalised_eigenvalue_problem_with_ggev(
            false,
            1e-7,
            1e-7,
            EigenvalueComparisonMode::Real,
        )
        .unwrap();
    close_l2(
        &res_ch_ggev.eigenvalues,
        &array![Complex::from(-0.08018188065), Complex::from(7.444862113)],
        1e-7,
    );
    close_l2(
        &hmat_ch.dot(&res_ch_ggev.eigenvectors),
        &smat_ch
            .dot(&res_ch_ggev.eigenvectors)
            .dot(&Array2::from_diag(&res_ch_ggev.eigenvalues)),
        1e-7,
    );
}

#[test]
fn test_solver_generalised_eigenvalue_problem_complex_hermitian_3x3_nullity_1_one_negative_canortho(
) {
    #[rustfmt::skip]
    let hmat_ch = array![
        [Complex::new( 4.9192343463,  0.0000000000), Complex::new(-0.1467367703, -1.7329799301), Complex::new( 0.7385391666,  2.3989906789)],
        [Complex::new(-0.1467367703,  1.7329799301), Complex::new( 6.2350102567,  0.0000000000), Complex::new(-2.9683155770, -3.3180237204)],
        [Complex::new( 0.7385391666, -2.3989906789), Complex::new(-2.9683155770,  3.3180237204), Complex::new( 4.4066218188,  0.0000000000)],
    ];
    // Eigenvalues of smat_ch: -2.073861814, 0.4752478556, 0.0
    #[rustfmt::skip]
    let smat_ch = array![
        [Complex::new(-0.2259629188,  0.0000000000), Complex::new(-0.1291472238,  0.4188976270), Complex::new(-0.1886007953, 0.5059039423)],
        [Complex::new(-0.1291472238, -0.4188976270), Complex::new(-0.1861302694,  0.0000000000), Complex::new(-0.9970206693, 0.1972322095)],
        [Complex::new(-0.1886007953, -0.5059039423), Complex::new(-0.9970206693, -0.1972322095), Complex::new(-1.1865207699, 0.0000000000)],
    ];

    let res_ch_canortho = (&hmat_ch.view(), &smat_ch.view())
        .solve_generalised_eigenvalue_problem_with_canonical_orthogonalisation(
            false,
            1e-7,
            1e-7,
            EigenvalueComparisonMode::Real,
        )
        .unwrap();
    // The finite eigenvalues for (H, S) is -0.08018188065 and 7.444862113 (from Mathematica). But
    // these differ from the eigenvalues by canonical orthogonalisation because of the nullity of S.
    close_l2(
        &res_ch_canortho.eigenvalues,
        &array![
            Complex::from(-0.39830030365645097),
            Complex::from(11.84041072912712)
        ],
        1e-7,
    );
    // The eigenvectors from thecanonical-orthogonalised eigenvalue problem will not satisfy the
    // original generalised eigenvalue problem.
    assert_ne!(
        (hmat_ch.dot(&res_ch_canortho.eigenvectors)
            - smat_ch
                .dot(&res_ch_canortho.eigenvectors)
                .dot(&Array2::from_diag(&res_ch_canortho.eigenvalues)))
        .map(|v| v.abs().powi(2))
        .into_iter()
        .sum::<f64>()
        .sqrt(),
        1e-7,
    );
}

#[test]
fn test_solver_generalised_eigenvalue_problem_complex_symmetric_3x3_nullity_1_ggev() {
    #[rustfmt::skip]
    let hmat_ch = array![
        [Complex::new( 4.9192343463, 0.0000000000), Complex::new(-0.1467367703, 1.7329799301), Complex::new( 0.7385391666, 2.3989906789)],
        [Complex::new(-0.1467367703, 1.7329799301), Complex::new( 6.2350102567, 0.0000000000), Complex::new(-2.9683155770, 3.3180237204)],
        [Complex::new( 0.7385391666, 2.3989906789), Complex::new(-2.9683155770, 3.3180237204), Complex::new( 4.4066218188, 0.0000000000)],
    ];
    // Eigenvalues of smat_ch: -2.034903493 - 0.630285052i, -0.8230265762 + 0.0785661840i, 0.0
    #[rustfmt::skip]
    let smat_ch = array![
        [Complex::new(-0.6457349560, -0.1839062892), Complex::new(-0.1291472238, -0.4188976270), Complex::new(-0.1886007953, -0.5059039423)],
        [Complex::new(-0.1291472238, -0.4188976270), Complex::new(-0.6059023066, -0.1839062892), Complex::new(-0.9970206693, -0.1972322095)],
        [Complex::new(-0.1886007953, -0.5059039423), Complex::new(-0.9970206693, -0.1972322095), Complex::new(-1.6062928070, -0.1839062890)],
    ];

    let res_ch_ggev = (&hmat_ch.view(), &smat_ch.view())
        .solve_generalised_eigenvalue_problem_with_ggev(
            true,
            1e-7,
            1e-7,
            EigenvalueComparisonMode::Real,
        )
        .unwrap();
    close_l2(
        &res_ch_ggev.eigenvalues,
        &array![
            Complex::new(-7.278289282, 1.494719941),
            Complex::new(-1.563889764, -0.934926052)
        ],
        1e-7,
    );
    close_l2(
        &hmat_ch.dot(&res_ch_ggev.eigenvectors),
        &smat_ch
            .dot(&res_ch_ggev.eigenvectors)
            .dot(&Array2::from_diag(&res_ch_ggev.eigenvalues)),
        1e-7,
    );
}
