use approx;
use nalgebra::{Point3, Vector3};

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

    let st = build_shell_tuple![(&bsc0, true), (&bsc1, false), (&bsc1, true)];
    assert_eq!(st.shell_shape, [3, 6, 6]);
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
    let bscs = [(&bsc0, false), (&bsc1, false)];
    let stc = build_shell_tuple_collection![<s1, s2>; bscs, bscs];
    println!("{:#?}", stc.lmax);

    // let st2 = build_shell_tuple![(&bsc, true), (&bsc, false), (&bsc, false), (&bsc, true)];
    // println!("{}", st2.rank());
}
