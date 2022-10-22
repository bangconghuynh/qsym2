use std::panic;

use num_traits::Inv;
use num_modular::{ModularInteger, MontgomeryInt, VanillaInt};
use ndarray::Array2;

#[test]
fn test_reducedint_arithmetic() {
    // Z/2Z is a field, but 2 is even, so we cannot use MontgomeryInt.
    let i0_2 = VanillaInt::<u64>::new(0, &2);
    let i1_2 = i0_2.convert(1);
    let i2_2 = i0_2.convert(2);
    assert_eq!(i0_2, i2_2);
    assert_eq!(i1_2 + i2_2, i1_2);
    assert_eq!(i1_2 / i1_2, i1_2);

    // Z/4Z is not a field.
    let i0_4 = VanillaInt::<u64>::new(0, &4);
    let i1_4 = i0_4.convert(1);
    let i2_4 = i0_4.convert(2);
    let i3_4 = i0_4.convert(3);
    assert_eq!(i3_4, i1_4 / i3_4);
    assert!(panic::catch_unwind(|| {
        println!("{:?}", i1_4 / i2_4);
    }).is_err());
    assert!(panic::catch_unwind(|| {
        i2_4.inv();
    }).is_err());

    // Z/5Z is a field, and 5 is odd, so we can use MontgomeryInt.
    let i0_5 = MontgomeryInt::<u64>::new(0, &5);
    let i1_5 = i0_5.convert(1);
    let i2_5 = i0_5.convert(2);
    let i3_5 = i0_5.convert(3);
    let i4_5 = i0_5.convert(4);
    assert_eq!(i1_5 + i3_5, i4_5);
    assert_eq!(i2_5 * i3_5, i1_5);
    assert_eq!(i4_5 * i3_5, i2_5);
    assert_eq!(i4_5 / i3_5, i3_5);
    assert_eq!(i2_5 / i3_5, i4_5);
    assert_eq!(i2_5 / i2_5, i1_5);
}

#[test]
fn test_reducedint_array() {
    let i0_5 = MontgomeryInt::<u64>::new(0, &5);
    let i1_5 = i0_5.convert(1);
    let i2_5 = i0_5.convert(2);
    let i3_5 = i0_5.convert(3);
    let i4_5 = i0_5.convert(4);

    let arr = Array2::<MontgomeryInt<u64>>::from_shape_vec(
        (2, 2), vec![i0_5, i1_5, i2_5, i3_5]
    ).unwrap();

    let arr_2 = arr.clone() * 4;
    let arr_2_ref = Array2::<MontgomeryInt<u64>>::from_shape_vec(
        (2, 2), vec![i0_5, i4_5, i3_5, i2_5]
    ).unwrap();
    assert_eq!(arr_2, arr_2_ref);

    let arr_3 = arr.clone() + arr_2;
    let arr_3_ref = Array2::<MontgomeryInt<u64>>::from_shape_vec(
        (2, 2), vec![i0_5, i0_5, i0_5, i0_5]
    ).unwrap();
    assert_eq!(arr_3, arr_3_ref);

    let arr_4 = -arr.clone();
    assert_eq!(arr_4, arr_2_ref);

    let arr_5 = arr.clone() * 12;
    let arr_5_ref = Array2::<MontgomeryInt<u64>>::from_shape_vec(
        (2, 2), vec![i0_5, i2_5, i4_5, i1_5]
    ).unwrap();
    assert_eq!(arr_5, arr_5_ref);

    let arr_6 = arr.clone() * i2_5.inv().residue();
    let arr_6_ref = Array2::<MontgomeryInt<u64>>::from_shape_vec(
        (2, 2), vec![i0_5, i3_5, i1_5, i4_5]
    ).unwrap();
    assert_eq!(arr_6, arr_6_ref);
}
