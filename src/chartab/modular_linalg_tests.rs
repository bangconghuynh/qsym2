use ndarray::{Array1, Array2};
use num_modular::{ModularInteger, MontgomeryInt, Montgomery};
use num_traits::{Inv, One, Pow, Zero};

use crate::chartab::modular_linalg::{modular_determinant, modular_rref, modular_kernel};
use crate::chartab::reducedint::{LinAlgReducedInt, ReducedIntToLinAlgReducedInt};

type LinAlgMontgomeryInt<T> = LinAlgReducedInt<T, Montgomery<T, T>>;

#[test]
fn test_modular_linalg_deteterminant() {
    let i0_5 = MontgomeryInt::<u64>::new(0, &5).linalg();
    let i1_5 = i0_5.convert(1);
    let i2_5 = i0_5.convert(2);
    let i3_5 = i0_5.convert(3);

    let arr = Array2::<LinAlgMontgomeryInt<u64>>::from_shape_vec(
        (2, 2), vec![i0_5, i1_5, i2_5, i3_5]
    ).unwrap();
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

    let arr_6 = Array2::<LinAlgMontgomeryInt<u64>>::from_shape_vec(
        (3, 3), vec![
            i1_13, i5_13, i6_13,
            i8_13, i4_13, i7_13,
            i10_13, i9_13, i11_13
        ]
    ).unwrap();
    assert_eq!(modular_determinant(&arr_6), i5_13);

    let arr_7 = Array2::<LinAlgMontgomeryInt<u64>>::from_shape_vec(
        (3, 3), vec![
            i0_13, i5_13, i6_13,
            i0_13, i0_13, i7_13,
            i1_13, i9_13, i0_13
        ]
    ).unwrap();
    assert_eq!(modular_determinant(&arr_7), i9_13);

    let arr_8 = Array2::<LinAlgMontgomeryInt<u64>>::from_shape_vec(
        (3, 3), vec![
            i0_13, i0_13, i6_13,
            i0_13, i0_13, i7_13,
            i1_13, i9_13, i0_13
        ]
    ).unwrap();
    assert_eq!(modular_determinant(&arr_8), i0_13);

    let arr_9 = Array2::<LinAlgMontgomeryInt<u64>>::from_shape_vec(
        (3, 3), vec![
            i1_13, i1_13, i6_13,
            i1_13, i1_13, i7_13,
            i1_13, i9_13, i0_13
        ]
    ).unwrap();
    assert_eq!(modular_determinant(&arr_9), i5_13);

    let arr_10 = Array2::<LinAlgMontgomeryInt<u64>>::from_shape_vec(
        (3, 3), vec![
            i1_13, i2_13, i6_13,
            i2_13, i4_13, i7_13,
            i4_13, i8_13, i0_13
        ]
    ).unwrap();
    assert_eq!(modular_determinant(&arr_10), i0_13);

    let arr_11 = Array2::<LinAlgMontgomeryInt<u64>>::from_shape_vec(
        (4, 4), vec![
            i1_13, i2_13, i6_13, i8_13,
            i2_13, i4_13, i7_13, i7_13,
            i4_13, i8_13, i0_13, i1_13,
            i5_13, i1_13, i9_13, i7_13,
        ]
    ).unwrap();
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

    let arr_1 = Array2::<LinAlgMontgomeryInt<u64>>::from_shape_vec(
        (3, 4), vec![
            i0_13, i2_13, i6_13, i4_13,
            i9_13, i4_13, i10_13, i1_13,
            i4_13, i8_13, i0_13, i12_13
        ]
    ).unwrap();

    let (arr_1_rref, arr_1_nulldim) = modular_rref(&arr_1);
    assert_eq!(
        arr_1_rref,
        Array2::<LinAlgMontgomeryInt<u64>>::from_shape_vec(
            (3, 4), vec![
                i1_13, i0_13, i7_13, i0_13,
                i0_13, i1_13, i3_13, i0_13,
                i0_13, i0_13, i0_13, i1_13
            ]
        ).unwrap()
    );
    assert_eq!(arr_1_nulldim, 1);

    let arr_1_test = Array2::<u64>::from_shape_vec(
        (3, 4), vec![
            0, 2, 6, 4,
            9, 4, 10, 1,
            4, 8, 0, 12
        ]
    ).unwrap();
    let kernel_vecs = modular_kernel(&arr_1);
    assert!(
        kernel_vecs.iter().all(|vec| { arr_1.dot(vec) == Array1::from_elem((3,), i0_13) })
    );

    let arr_2 = Array2::<LinAlgMontgomeryInt<u64>>::from_shape_vec(
        (5, 6), vec![
            i0_13, i0_13, i8_13, i1_13, i9_13, i8_13,
            i0_13, i6_13, i12_13, i3_13, i5_13, i4_13,
            i0_13, i7_13, i8_13, i2_13, i8_13, i9_13,
            i8_13, i9_13, i2_13, i1_13, i6_13, i6_13,
            i2_13, i2_13, i7_13, i10_13, i11_13, i12_13,
        ]
    ).unwrap();

    let (arr_2_rref, arr_2_nulldim) = modular_rref(&arr_2);
    assert_eq!(
        arr_2_rref,
        Array2::<LinAlgMontgomeryInt<u64>>::from_shape_vec(
            (5, 6), vec![
                i1_13, i0_13, i0_13, i0_13, i0_13, i12_13,
                i0_13, i1_13, i0_13, i0_13, i0_13, i6_13,
                i0_13, i0_13, i1_13, i0_13, i0_13, i6_13,
                i0_13, i0_13, i0_13, i1_13, i0_13, i2_13,
                i0_13, i0_13, i0_13, i0_13, i1_13, i4_13,
            ]
        ).unwrap()
    );
    assert_eq!(arr_2_nulldim, 1);

    let arr_3 = Array2::<LinAlgMontgomeryInt<u64>>::from_shape_vec(
        (5, 6), vec![
            i0_13, i0_13, i2_13, i1_13, i9_13, i8_13,
            i0_13, i6_13, i6_13, i3_13, i5_13, i4_13,
            i0_13, i7_13, i4_13, i2_13, i8_13, i9_13,
            i8_13, i9_13, i2_13, i1_13, i6_13, i6_13,
            i2_13, i2_13, i8_13, i4_13, i11_13, i12_13,
        ]
    ).unwrap();

    let (arr_3_rref, arr_3_nulldim) = modular_rref(&arr_3);
    assert_eq!(
        arr_3_rref,
        Array2::<LinAlgMontgomeryInt<u64>>::from_shape_vec(
            (5, 6), vec![
                i1_13, i0_13, i0_13, i0_13, i0_13, i0_13,
                i0_13, i1_13, i0_13, i0_13, i0_13, i0_13,
                i0_13, i0_13, i1_13, i7_13, i0_13, i0_13,
                i0_13, i0_13, i0_13, i0_13, i1_13, i0_13,
                i0_13, i0_13, i0_13, i0_13, i0_13, i1_13,
            ]
        ).unwrap()
    );
    assert_eq!(arr_3_nulldim, 1);

    let arr_4 = Array2::<LinAlgMontgomeryInt<u64>>::from_shape_vec(
        (5, 6), vec![
            i0_13, i0_13, i2_13, i1_13, i9_13, i8_13,
            i0_13, i6_13, i6_13, i3_13, i5_13, i4_13,
            i0_13, i7_13, i4_13, i2_13, i8_13, i9_13,
            i0_13, i9_13, i2_13, i1_13, i6_13, i6_13,
            i0_13, i2_13, i8_13, i4_13, i11_13, i12_13,
        ]
    ).unwrap();

    let (arr_4_rref, arr_4_nulldim) = modular_rref(&arr_4);
    assert_eq!(
        arr_4_rref,
        Array2::<LinAlgMontgomeryInt<u64>>::from_shape_vec(
            (5, 6), vec![
                i0_13, i1_13, i0_13, i0_13, i0_13, i0_13,
                i0_13, i0_13, i1_13, i7_13, i0_13, i0_13,
                i0_13, i0_13, i0_13, i0_13, i1_13, i0_13,
                i0_13, i0_13, i0_13, i0_13, i0_13, i1_13,
                i0_13, i0_13, i0_13, i0_13, i0_13, i0_13,
            ]
        ).unwrap()
    );
    assert_eq!(arr_4_nulldim, 2);

    let arr_5 = Array2::<LinAlgMontgomeryInt<u64>>::from_shape_vec(
        (5, 7), vec![
            i2_13, i0_13, i0_13, i2_13, i1_13, i9_13, i8_13,
            i4_13, i0_13, i6_13, i6_13, i3_13, i5_13, i4_13,
            i12_13, i0_13, i7_13, i4_13, i2_13, i8_13, i9_13,
            i1_13, i0_13, i9_13, i2_13, i1_13, i6_13, i6_13,
            i9_13, i0_13, i2_13, i8_13, i4_13, i11_13, i12_13,
        ]
    ).unwrap();

    let (arr_5_rref, arr_5_nulldim) = modular_rref(&arr_5);
    assert_eq!(
        arr_5_rref,
        Array2::<LinAlgMontgomeryInt<u64>>::from_shape_vec(
            (5, 7), vec![
                i1_13, i0_13, i0_13, i0_13, i0_13, i0_13, i0_13,
                i0_13, i0_13, i1_13, i0_13, i0_13, i0_13, i0_13,
                i0_13, i0_13, i0_13, i1_13, i7_13, i0_13, i0_13,
                i0_13, i0_13, i0_13, i0_13, i0_13, i1_13, i0_13,
                i0_13, i0_13, i0_13, i0_13, i0_13, i0_13, i1_13,
            ]
        ).unwrap()
    );
    assert_eq!(arr_5_nulldim, 2);
}
