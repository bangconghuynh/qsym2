use ndarray::{array, Array1};
use num_modular::{ModularInteger, MontgomeryInt};

use crate::chartab::modular_linalg::{modular_determinant, modular_kernel, modular_rref};
use crate::chartab::reducedint::IntoLinAlgReducedInt;

#[test]
fn test_modular_linalg_deteterminant() {
    let i0_5 = MontgomeryInt::<u64>::new(0, &5).linalg();
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

    let arr_5 = arr.clone() * 5;
    assert_eq!(modular_determinant(&arr_5), i0_5);

    let i0_13 = MontgomeryInt::<u64>::new(0, &13).linalg();
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
    let i0_13 = MontgomeryInt::<u64>::new(0, &13).linalg();
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

    let kernel_vecs = modular_kernel(&arr_1);
    assert!(kernel_vecs
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
}
