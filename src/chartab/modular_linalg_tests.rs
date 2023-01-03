

use ndarray::{array, Array1};
use num_modular::{ModularInteger, MontgomeryInt};

use crate::chartab::modular_linalg::{
    modular_determinant, modular_eig, modular_kernel, modular_rref, split_space,
};
use crate::chartab::reducedint::IntoLinAlgReducedInt;

#[test]
fn test_modular_linalg_deteterminant() {
    let i0_5 = MontgomeryInt::<u32>::new(0, &5).linalg();
    let i1_5 = i0_5.convert(1);
    let i2_5 = i0_5.convert(2);
    let i3_5 = i0_5.convert(3);

    let arr = array![[i0_5, i1_5], [i2_5, i3_5]];
    assert_eq!(modular_determinant(&arr), i3_5);

    let arr_2 = arr.clone() * 2;
    assert_eq!(modular_determinant(&arr_2), i2_5);

    let arr_3 = arr.clone() * 3;
    assert_eq!(modular_determinant(&arr_3), i2_5);

    let arr_4 = arr.clone() * 4;
    assert_eq!(modular_determinant(&arr_4), i3_5);

    let arr_5 = arr * 5;
    assert_eq!(modular_determinant(&arr_5), i0_5);

    let i0_13 = MontgomeryInt::<u32>::new(0, &13).linalg();
    let i1_13 = i0_13.convert(1);
    let i2_13 = i0_13.convert(2);
    let i3_13 = i0_13.convert(3);
    let i4_13 = i0_13.convert(4);
    let i5_13 = i0_13.convert(5);
    let i6_13 = i0_13.convert(6);
    let i7_13 = i0_13.convert(7);
    let i8_13 = i0_13.convert(8);
    let i9_13 = i0_13.convert(9);
    let i10_13 = i0_13.convert(10);
    let i11_13 = i0_13.convert(11);

    let arr_6 = array![
        [i1_13, i5_13, i6_13],
        [i8_13, i4_13, i7_13],
        [i10_13, i9_13, i11_13]
    ];
    assert_eq!(modular_determinant(&arr_6), i5_13);

    let arr_7 = array![
        [i0_13, i5_13, i6_13],
        [i0_13, i0_13, i7_13],
        [i1_13, i9_13, i0_13]
    ];
    assert_eq!(modular_determinant(&arr_7), i9_13);

    let arr_8 = array![
        [i0_13, i0_13, i6_13],
        [i0_13, i0_13, i7_13],
        [i1_13, i9_13, i0_13]
    ];
    assert_eq!(modular_determinant(&arr_8), i0_13);

    let arr_9 = array![
        [i1_13, i1_13, i6_13],
        [i1_13, i1_13, i7_13],
        [i1_13, i9_13, i0_13]
    ];
    assert_eq!(modular_determinant(&arr_9), i5_13);

    let arr_10 = array![
        [i1_13, i2_13, i6_13],
        [i2_13, i4_13, i7_13],
        [i4_13, i8_13, i0_13]
    ];
    assert_eq!(modular_determinant(&arr_10), i0_13);

    let arr_11 = array![
        [i1_13, i2_13, i6_13, i8_13],
        [i2_13, i4_13, i7_13, i7_13],
        [i4_13, i8_13, i0_13, i1_13],
        [i5_13, i1_13, i9_13, i7_13]
    ];
    assert_eq!(modular_determinant(&arr_11), i3_13);
}

#[test]
fn test_modular_linalg_rref_kernel() {
    let i0_13 = MontgomeryInt::<u32>::new(0, &13).linalg();
    let i1_13 = i0_13.convert(1);
    let i2_13 = i0_13.convert(2);
    let i3_13 = i0_13.convert(3);
    let i4_13 = i0_13.convert(4);
    let i5_13 = i0_13.convert(5);
    let i6_13 = i0_13.convert(6);
    let i7_13 = i0_13.convert(7);
    let i8_13 = i0_13.convert(8);
    let i9_13 = i0_13.convert(9);
    let i10_13 = i0_13.convert(10);
    let i11_13 = i0_13.convert(11);
    let i12_13 = i0_13.convert(12);

    let arr_1 = array![
        [i0_13, i2_13, i6_13, i4_13],
        [i9_13, i4_13, i10_13, i1_13],
        [i4_13, i8_13, i0_13, i12_13]
    ];

    let (arr_1_rref, arr_1_nulldim) = modular_rref(&arr_1);
    assert_eq!(
        arr_1_rref,
        array![
            [i1_13, i0_13, i7_13, i0_13],
            [i0_13, i1_13, i3_13, i0_13],
            [i0_13, i0_13, i0_13, i1_13]
        ]
    );
    assert_eq!(arr_1_nulldim, 1);

    let arr_1_kernel_vecs = modular_kernel(&arr_1);
    assert_eq!(arr_1_kernel_vecs.len(), arr_1_nulldim);
    assert!(arr_1_kernel_vecs
        .iter()
        .all(|vec| { arr_1.dot(vec) == Array1::from_elem((3,), i0_13) }));

    let arr_2 = array![
        [i0_13, i0_13, i8_13, i1_13, i9_13, i8_13],
        [i0_13, i6_13, i12_13, i3_13, i5_13, i4_13],
        [i0_13, i7_13, i8_13, i2_13, i8_13, i9_13],
        [i8_13, i9_13, i2_13, i1_13, i6_13, i6_13],
        [i2_13, i2_13, i7_13, i10_13, i11_13, i12_13]
    ];

    let (arr_2_rref, arr_2_nulldim) = modular_rref(&arr_2);
    assert_eq!(
        arr_2_rref,
        array![
            [i1_13, i0_13, i0_13, i0_13, i0_13, i12_13],
            [i0_13, i1_13, i0_13, i0_13, i0_13, i6_13],
            [i0_13, i0_13, i1_13, i0_13, i0_13, i6_13],
            [i0_13, i0_13, i0_13, i1_13, i0_13, i2_13],
            [i0_13, i0_13, i0_13, i0_13, i1_13, i4_13]
        ]
    );
    assert_eq!(arr_2_nulldim, 1);

    let arr_2_kernel_vecs = modular_kernel(&arr_2);
    assert_eq!(arr_2_kernel_vecs.len(), arr_2_nulldim);
    assert!(arr_2_kernel_vecs
        .iter()
        .all(|vec| { arr_2.dot(vec) == Array1::from_elem((5,), i0_13) }));

    let arr_3 = array![
        [i0_13, i0_13, i2_13, i1_13, i9_13, i8_13],
        [i0_13, i6_13, i6_13, i3_13, i5_13, i4_13],
        [i0_13, i7_13, i4_13, i2_13, i8_13, i9_13],
        [i8_13, i9_13, i2_13, i1_13, i6_13, i6_13],
        [i2_13, i2_13, i8_13, i4_13, i11_13, i12_13]
    ];

    let (arr_3_rref, arr_3_nulldim) = modular_rref(&arr_3);
    assert_eq!(
        arr_3_rref,
        array![
            [i1_13, i0_13, i0_13, i0_13, i0_13, i0_13],
            [i0_13, i1_13, i0_13, i0_13, i0_13, i0_13],
            [i0_13, i0_13, i1_13, i7_13, i0_13, i0_13],
            [i0_13, i0_13, i0_13, i0_13, i1_13, i0_13],
            [i0_13, i0_13, i0_13, i0_13, i0_13, i1_13]
        ]
    );
    assert_eq!(arr_3_nulldim, 1);

    let arr_3_kernel_vecs = modular_kernel(&arr_3);
    assert_eq!(arr_3_kernel_vecs.len(), arr_3_nulldim);
    assert!(arr_3_kernel_vecs
        .iter()
        .all(|vec| { arr_3.dot(vec) == Array1::from_elem((5,), i0_13) }));

    let arr_4 = array![
        [i0_13, i0_13, i2_13, i1_13, i9_13, i8_13],
        [i0_13, i6_13, i6_13, i3_13, i5_13, i4_13],
        [i0_13, i7_13, i4_13, i2_13, i8_13, i9_13],
        [i0_13, i9_13, i2_13, i1_13, i6_13, i6_13],
        [i0_13, i2_13, i8_13, i4_13, i11_13, i12_13]
    ];

    let (arr_4_rref, arr_4_nulldim) = modular_rref(&arr_4);
    assert_eq!(
        arr_4_rref,
        array![
            [i0_13, i1_13, i0_13, i0_13, i0_13, i0_13],
            [i0_13, i0_13, i1_13, i7_13, i0_13, i0_13],
            [i0_13, i0_13, i0_13, i0_13, i1_13, i0_13],
            [i0_13, i0_13, i0_13, i0_13, i0_13, i1_13],
            [i0_13, i0_13, i0_13, i0_13, i0_13, i0_13],
        ]
    );
    assert_eq!(arr_4_nulldim, 2);

    let arr_4_kernel_vecs = modular_kernel(&arr_4);
    assert_eq!(arr_4_kernel_vecs.len(), arr_4_nulldim);
    assert!(arr_4_kernel_vecs
        .iter()
        .all(|vec| { arr_4.dot(vec) == Array1::from_elem((5,), i0_13) }));

    let arr_5 = array![
        [i2_13, i0_13, i0_13, i2_13, i1_13, i9_13, i8_13],
        [i4_13, i0_13, i6_13, i6_13, i3_13, i5_13, i4_13],
        [i12_13, i0_13, i7_13, i4_13, i2_13, i8_13, i9_13],
        [i1_13, i0_13, i9_13, i2_13, i1_13, i6_13, i6_13],
        [i9_13, i0_13, i2_13, i8_13, i4_13, i11_13, i12_13]
    ];

    let (arr_5_rref, arr_5_nulldim) = modular_rref(&arr_5);
    assert_eq!(
        arr_5_rref,
        array![
            [i1_13, i0_13, i0_13, i0_13, i0_13, i0_13, i0_13],
            [i0_13, i0_13, i1_13, i0_13, i0_13, i0_13, i0_13],
            [i0_13, i0_13, i0_13, i1_13, i7_13, i0_13, i0_13],
            [i0_13, i0_13, i0_13, i0_13, i0_13, i1_13, i0_13],
            [i0_13, i0_13, i0_13, i0_13, i0_13, i0_13, i1_13]
        ]
    );
    assert_eq!(arr_5_nulldim, 2);

    let arr_5_kernel_vecs = modular_kernel(&arr_5);
    assert_eq!(arr_5_kernel_vecs.len(), arr_5_nulldim);
    assert!(arr_5_kernel_vecs
        .iter()
        .all(|vec| { arr_5.dot(vec) == Array1::from_elem((5,), i0_13) }));

    let arr_6 = array![
        [i2_13, i0_13, i0_13, i2_13, i1_13, i9_13, i8_13],
        [i4_13, i0_13, i6_13, i6_13, i3_13, i5_13, i4_13],
        [i9_13, i0_13, i2_13, i8_13, i4_13, i11_13, i12_13]
    ];

    let (arr_6_rref, arr_6_nulldim) = modular_rref(&arr_6);
    assert_eq!(
        arr_6_rref,
        array![
            [i1_13, i0_13, i0_13, i0_13, i0_13, i5_13, i5_13],
            [i0_13, i0_13, i1_13, i0_13, i0_13, i11_13, i7_13],
            [i0_13, i0_13, i0_13, i1_13, i7_13, i6_13, i12_13]
        ]
    );
    assert_eq!(arr_6_nulldim, 4);

    let arr_6_kernel_vecs = modular_kernel(&arr_6);
    assert_eq!(arr_6_kernel_vecs.len(), arr_6_nulldim);
    assert!(arr_6_kernel_vecs
        .iter()
        .all(|vec| { arr_6.dot(vec) == Array1::from_elem((3,), i0_13) }));
}

#[test]
fn test_modular_linalg_eig() {
    let m19 = MontgomeryInt::<u32>::new(0, &19).linalg();
    let i_19s: Vec<_> = (0..19).map(|x| m19.convert(x)).collect();

    let arr_1 = array![
        [i_19s[12], i_19s[14], i_19s[0], i_19s[0]],
        [i_19s[14], i_19s[12], i_19s[0], i_19s[0]],
        [i_19s[0], i_19s[0], i_19s[3], i_19s[0]],
        [i_19s[0], i_19s[0], i_19s[0], i_19s[3]]
    ];
    let eigs = modular_eig(&arr_1);
    eigs.iter().for_each(|(val, vecs)| {
        vecs.iter().for_each(|vec| {
            assert_eq!(arr_1.dot(vec), vec.map(|x| { x * val }));
        })
    });

    let m5 = MontgomeryInt::<u32>::new(0, &5).linalg();
    let i_5s: Vec<_> = (0..5).map(|x| m5.convert(x)).collect();

    let arr_2 = array![[i_5s[2], i_5s[2]], [i_5s[1], i_5s[1]]];
    let eigs = modular_eig(&arr_2);
    eigs.iter().for_each(|(val, vecs)| {
        vecs.iter().for_each(|vec| {
            assert_eq!(arr_2.dot(vec), vec.map(|x| { x * val }));
        })
    });

    let arr_3 = array![[i_19s[7], i_19s[2]], [i_19s[15], i_19s[1]]];
    let eigs = modular_eig(&arr_3);
    eigs.iter().for_each(|(val, vecs)| {
        vecs.iter().for_each(|vec| {
            assert_eq!(arr_3.dot(vec), vec.map(|x| { x * val }));
        })
    });

    // Ref: L. C. Grove, Groups and Characters, John Wiley & Sons, Inc., 1997,
    // p. 157
    let m41 = MontgomeryInt::<u32>::new(0, &41).linalg();
    let i_41s: Vec<_> = (0..41).map(|x| m41.convert(x)).collect();
    let arr_4 = array![
        [i_41s[0], i_41s[1], i_41s[0], i_41s[0], i_41s[0]],
        [i_41s[4], i_41s[3], i_41s[0], i_41s[0], i_41s[0]],
        [i_41s[0], i_41s[0], i_41s[4], i_41s[0], i_41s[0]],
        [i_41s[0], i_41s[0], i_41s[0], i_41s[4], i_41s[0]],
        [i_41s[0], i_41s[0], i_41s[0], i_41s[0], i_41s[4]],
    ];
    let eigs = modular_eig(&arr_4);
    eigs.iter().for_each(|(val, vecs)| {
        vecs.iter().for_each(|vec| {
            assert_eq!(arr_4.dot(vec), vec.map(|x| { x * val }));
        })
    });
    assert_eq!(eigs.len(), 2);
    assert_eq!(eigs.get(&i_41s[4]).unwrap().len(), 4);
    assert_eq!(eigs.get(&i_41s[40]).unwrap().len(), 1);

    // Ref: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.598.7381&rep=rep1&type=pdf,
    // p. 11
    let m5 = MontgomeryInt::<u32>::new(0, &5).linalg();
    let i_5s: Vec<_> = (0..5).map(|x| m5.convert(x)).collect();
    let arr_5 = array![
        [i_5s[0], i_5s[1], i_5s[0], i_5s[0]],
        [i_5s[1], i_5s[0], i_5s[0], i_5s[0]],
        [i_5s[0], i_5s[0], i_5s[0], i_5s[1]],
        [i_5s[0], i_5s[0], i_5s[1], i_5s[0]],
    ];
    let eigs = modular_eig(&arr_5);
    eigs.iter().for_each(|(val, vecs)| {
        vecs.iter().for_each(|vec| {
            assert_eq!(arr_5.dot(vec), vec.map(|x| { x * val }));
        })
    });
    assert_eq!(eigs.len(), 2);
    assert_eq!(eigs.get(&i_5s[1]).unwrap().len(), 2);
    assert_eq!(eigs.get(&i_5s[4]).unwrap().len(), 2);
}

#[test]
fn test_modular_linalg_split_space() {
    // Ref: L. C. Grove, Groups and Characters, John Wiley & Sons, Inc., 1997,
    // p. 157
    let class_sizes: Vec<usize> = vec![1, 4, 5, 5, 5];
    let perm_for_conj = vec![0, 1, 2, 3, 4];
    let m41 = MontgomeryInt::<u32>::new(0, &41).linalg();
    let i_41s: Vec<_> = (0..41).map(|x| m41.convert(x)).collect();
    let arr_1 = array![
        [i_41s[0], i_41s[1], i_41s[0], i_41s[0], i_41s[0]],
        [i_41s[4], i_41s[3], i_41s[0], i_41s[0], i_41s[0]],
        [i_41s[0], i_41s[0], i_41s[4], i_41s[0], i_41s[0]],
        [i_41s[0], i_41s[0], i_41s[0], i_41s[4], i_41s[0]],
        [i_41s[0], i_41s[0], i_41s[0], i_41s[0], i_41s[4]],
    ];
    let arr_2 = array![
        [i_41s[0], i_41s[0], i_41s[1], i_41s[0], i_41s[0]],
        [i_41s[0], i_41s[0], i_41s[4], i_41s[0], i_41s[0]],
        [i_41s[0], i_41s[0], i_41s[0], i_41s[5], i_41s[0]],
        [i_41s[0], i_41s[0], i_41s[0], i_41s[0], i_41s[5]],
        [i_41s[5], i_41s[5], i_41s[0], i_41s[0], i_41s[0]],
    ];

    let eigs = modular_eig(&arr_1);
    let ev4_eigvecs = eigs.get(&i_41s[4]).unwrap();
    let ev4_eigvecs_subspaces =
        split_space(&arr_2, ev4_eigvecs, &class_sizes, Some(&perm_for_conj)).unwrap();
    assert_eq!(ev4_eigvecs_subspaces.len(), 4);

    let mat2_on_mat1_ev4_subspaces_ref = vec![
        array![i_41s[1], i_41s[4], i_41s[5], i_41s[5], i_41s[5]],
        array![i_41s[1], i_41s[4], i_41s[36], i_41s[5], i_41s[36]],
        array![i_41s[1], i_41s[4], i_41s[4], i_41s[36], i_41s[37]],
        array![i_41s[1], i_41s[4], i_41s[37], i_41s[36], i_41s[4]],
    ];

    assert!(ev4_eigvecs_subspaces.iter().all(|x| {
        x.len() == 1
            && mat2_on_mat1_ev4_subspaces_ref
                .iter()
                .any(|ref_x| x[0] == ref_x)
    }));
}
