use crate::aux::ao_basis::{BasisAngularOrder, BasisAtom, BasisShell, CartOrder, ShellOrder};
use crate::aux::atom::{Atom, ElementMap};

#[test]
fn test_ao_basis_cartorder() {
    // =========
    // lcart = 0
    // =========
    let co_0_lex = CartOrder::lex(0);
    assert_eq!(co_0_lex.cart_tuples, vec![(0, 0, 0)]);

    let co_0_qchem = CartOrder::qchem(0);
    assert_eq!(co_0_qchem.cart_tuples, vec![(0, 0, 0)]);

    // =========
    // lcart = 1
    // =========
    let co_1_lex = CartOrder::lex(1);
    assert_eq!(co_1_lex.cart_tuples, vec![(1, 0, 0), (0, 1, 0), (0, 0, 1)]);

    let co_1_qchem = CartOrder::qchem(1);
    assert_eq!(
        co_1_qchem.cart_tuples,
        vec![(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    );

    // =========
    // lcart = 2
    // =========
    let co_2_lex = CartOrder::lex(2);
    assert_eq!(
        co_2_lex.cart_tuples,
        vec![
            (2, 0, 0),
            (1, 1, 0),
            (1, 0, 1),
            (0, 2, 0),
            (0, 1, 1),
            (0, 0, 2),
        ]
    );

    let co_2_qchem = CartOrder::qchem(2);
    assert_eq!(
        co_2_qchem.cart_tuples,
        vec![
            (2, 0, 0),
            (1, 1, 0),
            (0, 2, 0),
            (1, 0, 1),
            (0, 1, 1),
            (0, 0, 2),
        ]
    );

    // =========
    // lcart = 3
    // =========
    let co_3_lex = CartOrder::lex(3);
    assert_eq!(
        co_3_lex.cart_tuples,
        vec![
            (3, 0, 0),
            (2, 1, 0),
            (2, 0, 1),
            (1, 2, 0),
            (1, 1, 1),
            (1, 0, 2),
            (0, 3, 0),
            (0, 2, 1),
            (0, 1, 2),
            (0, 0, 3),
        ]
    );

    let co_3_qchem = CartOrder::qchem(3);
    assert_eq!(
        co_3_qchem.cart_tuples,
        vec![
            (3, 0, 0),
            (2, 1, 0),
            (1, 2, 0),
            (0, 3, 0),
            (2, 0, 1),
            (1, 1, 1),
            (0, 2, 1),
            (1, 0, 2),
            (0, 1, 2),
            (0, 0, 3),
        ]
    );

    // =========
    // lcart = 4
    // =========
    let co_4_lex = CartOrder::lex(4);
    assert_eq!(
        co_4_lex.cart_tuples,
        vec![
            (4, 0, 0),
            (3, 1, 0),
            (3, 0, 1),
            (2, 2, 0),
            (2, 1, 1),
            (2, 0, 2),
            (1, 3, 0),
            (1, 2, 1),
            (1, 1, 2),
            (1, 0, 3),
            (0, 4, 0),
            (0, 3, 1),
            (0, 2, 2),
            (0, 1, 3),
            (0, 0, 4),
        ]
    );

    let co_4_qchem = CartOrder::qchem(4);
    assert_eq!(
        co_4_qchem.cart_tuples,
        vec![
            (4, 0, 0),
            (3, 1, 0),
            (2, 2, 0),
            (1, 3, 0),
            (0, 4, 0),
            (3, 0, 1),
            (2, 1, 1),
            (1, 2, 1),
            (0, 3, 1),
            (2, 0, 2),
            (1, 1, 2),
            (0, 2, 2),
            (1, 0, 3),
            (0, 1, 3),
            (0, 0, 4),
        ]
    );
}

#[test]
fn test_ao_basis_basisshell() {
    let bs0_p = BasisShell::new(0, ShellOrder::Pure(true));
    let bs0_c = BasisShell::new(0, ShellOrder::Cart(CartOrder::lex(0)));
    assert_eq!(bs0_p.n_funcs(), 1);
    assert_eq!(bs0_c.n_funcs(), 1);

    let bs1_p = BasisShell::new(1, ShellOrder::Pure(true));
    let bs1_c = BasisShell::new(1, ShellOrder::Cart(CartOrder::lex(1)));
    assert_eq!(bs1_p.n_funcs(), 3);
    assert_eq!(bs1_c.n_funcs(), 3);

    let bs2_p = BasisShell::new(2, ShellOrder::Pure(true));
    let bs2_c = BasisShell::new(2, ShellOrder::Cart(CartOrder::lex(2)));
    assert_eq!(bs2_p.n_funcs(), 5);
    assert_eq!(bs2_c.n_funcs(), 6);

    let bs3_p = BasisShell::new(3, ShellOrder::Pure(true));
    let bs3_c = BasisShell::new(3, ShellOrder::Cart(CartOrder::lex(3)));
    assert_eq!(bs3_p.n_funcs(), 7);
    assert_eq!(bs3_c.n_funcs(), 10);
}

#[test]
fn test_ao_basis_basisatom() {
    let emap = ElementMap::new();
    let atm = Atom::from_xyz("C 0.0 0.0 0.0", &emap, 1e-7).unwrap();

    let bs1s_p = BasisShell::new(0, ShellOrder::Pure(true));
    let bs2s_p = BasisShell::new(0, ShellOrder::Pure(true));
    let bs2p_p = BasisShell::new(1, ShellOrder::Pure(true));
    let bs3s_p = BasisShell::new(0, ShellOrder::Pure(true));
    let bs3p_p = BasisShell::new(1, ShellOrder::Pure(true));
    let bs3d_c = BasisShell::new(2, ShellOrder::Cart(CartOrder::lex(2)));

    let batm = BasisAtom::new(&atm, &[bs1s_p, bs2s_p, bs2p_p, bs3s_p, bs3p_p, bs3d_c]);

    assert_eq!(batm.n_funcs(), 15);
    assert_eq!(
        batm.shell_boundary_indices(),
        &[(0, 1), (1, 2), (2, 5), (5, 6), (6, 9), (9, 15),]
    );
}

#[test]
fn test_ao_basis_basisangularorder() {
    let emap = ElementMap::new();
    let atm_c = Atom::from_xyz("C 0.0 0.0 0.0", &emap, 1e-7).unwrap();

    let bs1s_p = BasisShell::new(0, ShellOrder::Pure(true));
    let bs2s_p = BasisShell::new(0, ShellOrder::Pure(true));
    let bs2p_p = BasisShell::new(1, ShellOrder::Pure(true));
    let bs3s_p = BasisShell::new(0, ShellOrder::Pure(true));
    let bs3p_p = BasisShell::new(1, ShellOrder::Pure(true));
    let bs3d_c = BasisShell::new(2, ShellOrder::Cart(CartOrder::lex(2)));
    let bs3d_p = BasisShell::new(2, ShellOrder::Pure(true));
    let bs4s_p = BasisShell::new(0, ShellOrder::Pure(true));
    let bs4p_p = BasisShell::new(1, ShellOrder::Pure(true));
    let bs4d_p = BasisShell::new(2, ShellOrder::Pure(true));

    let batm_c = BasisAtom::new(
        &atm_c,
        &[
            bs1s_p.clone(),
            bs2s_p.clone(),
            bs2p_p.clone(),
            bs3s_p.clone(),
            bs3p_p.clone(),
            bs3d_c,
        ],
    );

    let atm_h1 = Atom::from_xyz("H 0.0 0.0 1.0", &emap, 1e-7).unwrap();
    let batm_h1 = BasisAtom::new(&atm_h1, &[bs1s_p.clone(), bs2s_p.clone(), bs3s_p.clone()]);

    let atm_h2 = Atom::from_xyz("H 0.0 0.0 -1.0", &emap, 1e-7).unwrap();
    let batm_h2 = BasisAtom::new(&atm_h2, &[bs1s_p.clone(), bs2s_p.clone(), bs3s_p.clone()]);

    let atm_f = Atom::from_xyz("F 0.0 1.0 0.0", &emap, 1e-7).unwrap();
    let batm_f = BasisAtom::new(
        &atm_f,
        &[
            bs1s_p.clone(),
            bs2s_p.clone(),
            bs2p_p.clone(),
            bs3s_p.clone(),
            bs3p_p.clone(),
            bs3d_p.clone(),
        ],
    );

    let atm_cl = Atom::from_xyz("Cl 0.0 -1.0 0.0", &emap, 1e-7).unwrap();
    let batm_cl = BasisAtom::builder()
        .atom(&atm_cl)
        .basis_shells(&[
            bs1s_p, bs2s_p, bs2p_p, bs3s_p, bs3p_p, bs3d_p, bs4s_p, bs4p_p, bs4d_p,
        ])
        .build()
        .unwrap();

    let bao = BasisAngularOrder::new(&[batm_c, batm_h1, batm_h2, batm_f, batm_cl]);
    println!("{bao}");

    assert_eq!(bao.n_funcs(), 58);
    assert_eq!(bao.basis_shells().collect::<Vec<_>>().len(), 27);
    assert_eq!(
        bao.atom_boundary_indices(),
        &[(0, 15), (15, 18), (18, 21), (21, 35), (35, 58)]
    );
    assert_eq!(
        bao.shell_boundary_indices(),
        &[
            (0, 1),
            (1, 2),
            (2, 5),
            (5, 6),
            (6, 9),
            (9, 15),
            (15, 16),
            (16, 17),
            (17, 18),
            (18, 19),
            (19, 20),
            (20, 21),
            (21, 22),
            (22, 23),
            (23, 26),
            (26, 27),
            (27, 30),
            (30, 35),
            (35, 36),
            (36, 37),
            (37, 40),
            (40, 41),
            (41, 44),
            (44, 49),
            (49, 50),
            (50, 53),
            (53, 58),
        ]
    );
}
