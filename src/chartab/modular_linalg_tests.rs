use num_modular::{ModularInteger, MontgomeryInt};
use ndarray::Array2;

use crate::chartab::modular_linalg::modular_determinant;

#[test]
fn test_modular_linalg_deteterminant() {
    let i0_5 = MontgomeryInt::<u64>::new(0, &5);
    let i1_5 = i0_5.convert(1);
    let i2_5 = i0_5.convert(2);
    let i3_5 = i0_5.convert(3);

    let arr = Array2::<MontgomeryInt<u64>>::from_shape_vec(
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

    let i0_13 = MontgomeryInt::<u64>::new(0, &13);
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

    let arr_6 = Array2::<MontgomeryInt<u64>>::from_shape_vec(
        (3, 3), vec![
            i1_13, i5_13, i6_13,
            i8_13, i4_13, i7_13,
            i10_13, i9_13, i11_13
        ]
    ).unwrap();
    assert_eq!(modular_determinant(&arr_6), i5_13);

    let arr_7 = Array2::<MontgomeryInt<u64>>::from_shape_vec(
        (3, 3), vec![
            i0_13, i5_13, i6_13,
            i0_13, i0_13, i7_13,
            i1_13, i9_13, i0_13
        ]
    ).unwrap();
    assert_eq!(modular_determinant(&arr_7), i9_13);

    let arr_8 = Array2::<MontgomeryInt<u64>>::from_shape_vec(
        (3, 3), vec![
            i0_13, i0_13, i6_13,
            i0_13, i0_13, i7_13,
            i1_13, i9_13, i0_13
        ]
    ).unwrap();
    assert_eq!(modular_determinant(&arr_8), i0_13);

    let arr_9 = Array2::<MontgomeryInt<u64>>::from_shape_vec(
        (3, 3), vec![
            i1_13, i1_13, i6_13,
            i1_13, i1_13, i7_13,
            i1_13, i9_13, i0_13
        ]
    ).unwrap();
    assert_eq!(modular_determinant(&arr_9), i5_13);

    let arr_10 = Array2::<MontgomeryInt<u64>>::from_shape_vec(
        (3, 3), vec![
            i1_13, i2_13, i6_13,
            i2_13, i4_13, i7_13,
            i4_13, i8_13, i0_13
        ]
    ).unwrap();
    assert_eq!(modular_determinant(&arr_10), i0_13);

    let arr_11 = Array2::<MontgomeryInt<u64>>::from_shape_vec(
        (4, 4), vec![
            i1_13, i2_13, i6_13, i8_13,
            i2_13, i4_13, i7_13, i7_13,
            i4_13, i8_13, i0_13, i1_13,
            i5_13, i1_13, i9_13, i7_13,
        ]
    ).unwrap();
    assert_eq!(modular_determinant(&arr_11), i3_13);
}
