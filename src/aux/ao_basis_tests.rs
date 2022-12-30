use crate::aux::ao_basis::{BasisAtom, BasisShell, CartOrder, ShellOrder};
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
    let bs0_p = BasisShell::builder()
        .l(0)
        .shell_order(ShellOrder::Pure(true))
        .build()
        .unwrap();
    let bs0_c = BasisShell::builder()
        .l(0)
        .shell_order(ShellOrder::Cart(CartOrder::lex(0)))
        .build()
        .unwrap();
    assert_eq!(bs0_p.n_funcs(), 1);
    assert_eq!(bs0_c.n_funcs(), 1);

    let bs1_p = BasisShell::builder()
        .l(1)
        .shell_order(ShellOrder::Pure(true))
        .build()
        .unwrap();
    let bs1_c = BasisShell::builder()
        .l(1)
        .shell_order(ShellOrder::Cart(CartOrder::lex(1)))
        .build()
        .unwrap();
    assert_eq!(bs1_p.n_funcs(), 3);
    assert_eq!(bs1_c.n_funcs(), 3);

    let bs2_p = BasisShell::builder()
        .l(2)
        .shell_order(ShellOrder::Pure(true))
        .build()
        .unwrap();
    let bs2_c = BasisShell::builder()
        .l(2)
        .shell_order(ShellOrder::Cart(CartOrder::lex(2)))
        .build()
        .unwrap();
    assert_eq!(bs2_p.n_funcs(), 5);
    assert_eq!(bs2_c.n_funcs(), 6);

    let bs3_p = BasisShell::builder()
        .l(3)
        .shell_order(ShellOrder::Pure(true))
        .build()
        .unwrap();
    let bs3_c = BasisShell::builder()
        .l(3)
        .shell_order(ShellOrder::Cart(CartOrder::lex(3)))
        .build()
        .unwrap();
    assert_eq!(bs3_p.n_funcs(), 7);
    assert_eq!(bs3_c.n_funcs(), 10);
}

#[test]
fn test_ao_basis_basisatom() {
    let emap = ElementMap::new();
    let atm = Atom::from_xyz("C 0.0 0.0 0.0", &emap, 1e-7).unwrap();

    let bs1s_p = BasisShell::builder()
        .l(0)
        .shell_order(ShellOrder::Pure(true))
        .build()
        .unwrap();
    let bs2s_p = BasisShell::builder()
        .l(0)
        .shell_order(ShellOrder::Pure(true))
        .build()
        .unwrap();
    let bs2p_p = BasisShell::builder()
        .l(1)
        .shell_order(ShellOrder::Pure(true))
        .build()
        .unwrap();
    let bs3s_p = BasisShell::builder()
        .l(0)
        .shell_order(ShellOrder::Pure(true))
        .build()
        .unwrap();
    let bs3p_p = BasisShell::builder()
        .l(1)
        .shell_order(ShellOrder::Pure(true))
        .build()
        .unwrap();
    let bs3d_c = BasisShell::builder()
        .l(2)
        .shell_order(ShellOrder::Cart(CartOrder::lex(2)))
        .build()
        .unwrap();

    let batm = BasisAtom::builder()
        .atom(&atm)
        .basis_shells(&[bs1s_p, bs2s_p, bs2p_p, bs3s_p, bs3p_p, bs3d_c])
        .build()
        .unwrap();

    assert_eq!(batm.n_funcs(), 15);
    assert_eq!(
        batm.shell_boundary_indices(),
        &[
            (0, 1),
            (1, 2),
            (2, 5),
            (5, 6),
            (6, 9),
            (9, 15),
        ]
    );
}
