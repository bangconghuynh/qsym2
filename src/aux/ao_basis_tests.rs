use crate::aux::ao_basis::CartOrder;


#[test]
fn test_sh_conversion_cartorder() {
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
