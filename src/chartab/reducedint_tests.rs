use std::panic;

use ndarray::array;
use num_modular::{ModularInteger, MontgomeryInt, VanillaInt};
use num_traits::{Inv, One, Pow, Zero};

use crate::chartab::reducedint::{IntoLinAlgReducedInt, LinAlgMontgomeryInt};

#[test]
fn test_linalgreducedint_identities() {
    let zero = LinAlgMontgomeryInt::<u64>::zero();
    let one = LinAlgMontgomeryInt::<u64>::one();
    let i0_7 = MontgomeryInt::<u64>::new(0, &7).linalg();
    let i1_7 = i0_7.convert(1);
    let i2_7 = i0_7.convert(2);
    let i3_7 = i0_7.convert(3);
    let i4_7 = i0_7.convert(4);
    let i5_7 = i0_7.convert(5);
    let i6_7 = i0_7.convert(6);

    // Zero
    assert!(Zero::is_zero(&zero));
    assert!(Zero::is_zero(&i0_7));
    assert_eq!(zero, i0_7);

    // One
    assert!(one.is_one());
    assert!(i1_7.is_one());
    assert_eq!(one, i1_7);

    // Addition
    assert_eq!(zero + zero, i0_7);
    assert_eq!(i0_7 + zero, i0_7);
    assert_eq!(zero + one, i1_7);
    assert_eq!(i0_7 + one, i1_7);
    assert_eq!(zero + i1_7, i1_7);
    assert_eq!(i0_7 + i1_7, i1_7);
    assert_eq!(zero + i6_7, i6_7);
    assert_eq!(one + i6_7, zero);
    assert_eq!(one + i1_7, i2_7);
    assert!(panic::catch_unwind(|| { one + one }).is_err());

    // Subtraction
    assert_eq!(zero - zero, i0_7);
    assert_eq!(one - zero, i1_7);
    assert_eq!(i3_7 - zero, i3_7);
    assert_eq!(i4_7 - i4_7, zero);
    assert_eq!(i0_7 - one, i6_7);
    assert_eq!(i1_7 - one, zero);
    assert_eq!(i5_7 - one, i4_7);
    assert_eq!(one - i1_7, zero);
    assert_eq!(one - one, zero);
    assert!(panic::catch_unwind(|| { zero - one }).is_err());

    // Multiplication
    assert_eq!(zero * zero, i0_7);
    assert_eq!(zero * i1_7, i0_7);
    assert_eq!(i0_7 * i1_7, zero);
    assert_eq!(i3_7 * one, i3_7);
    assert_eq!(one * i6_7, i6_7);
    assert_eq!(zero * one, i0_7);
    assert_eq!(i3_7 * i6_7, i4_7);

    // Division
    assert_eq!(zero / i4_7, zero);
    assert_eq!(zero / one, zero);
    assert_eq!(i5_7 / one, i5_7);
    assert_eq!(one / i5_7, i3_7);
    assert_eq!(i3_7 / i4_7, i6_7);
    assert!(panic::catch_unwind(|| { i5_7 / zero }).is_err());
    assert!(panic::catch_unwind(|| { one / zero }).is_err());
    assert!(panic::catch_unwind(|| { one / i0_7 }).is_err());

    // Power
    assert_eq!(zero.pow(0), one);
    assert_eq!(i0_7.pow(0), one);
    assert_eq!(zero.pow(4), zero);
    assert_eq!(i0_7.pow(3), zero);
    assert_eq!(one.pow(0), one);
    assert_eq!(one.pow(4), one);
    assert_eq!(i1_7.pow(0), one);
    assert_eq!(i1_7.pow(5), one);
    assert_eq!(i4_7.pow(0), one);

    // Inverse
    assert_eq!(one / i5_7, i5_7.inv());
    assert_eq!(i1_7 / i3_7, i3_7.inv());
    assert_eq!(i4_7.inv(), i2_7);
    assert_eq!(one.inv(), one);
    assert!(panic::catch_unwind(|| { zero.inv() }).is_err());
    assert!(panic::catch_unwind(|| { i0_7.inv() }).is_err());
}

#[test]
fn test_linalgreducedint_multiplicative_order() {
    let zero = LinAlgMontgomeryInt::<u64>::zero();
    let one = LinAlgMontgomeryInt::<u64>::one();
    let i0_7 = MontgomeryInt::<u64>::new(0, &7).linalg();
    let i1_7 = i0_7.convert(1);
    let i2_7 = i0_7.convert(2);
    let i3_7 = i0_7.convert(3);
    let i4_7 = i0_7.convert(4);
    let i5_7 = i0_7.convert(5);
    let i6_7 = i0_7.convert(6);

    assert!(i0_7.multiplicative_order().is_none());
    assert!(zero.multiplicative_order().is_none());

    assert_eq!(i1_7.multiplicative_order(), Some(1));
    assert_eq!(one.multiplicative_order(), Some(1));

    assert_eq!(i2_7.multiplicative_order(), Some(3));
    assert_eq!(i3_7.multiplicative_order(), Some(6));
    assert_eq!(i4_7.multiplicative_order(), Some(3));
    assert_eq!(i5_7.multiplicative_order(), Some(6));
    assert_eq!(i6_7.multiplicative_order(), Some(2));

    let i0_9 = MontgomeryInt::<u64>::new(0, &9).linalg();
    let i1_9 = i0_9.convert(1);
    let i2_9 = i0_9.convert(2);
    let i3_9 = i0_9.convert(3);
    assert_eq!(i1_9.multiplicative_order(), Some(1));
    assert_eq!(i2_9.multiplicative_order(), Some(6));
    assert!(i3_9.multiplicative_order().is_none());
}

#[test]
fn test_linalgreducedint_arithmetic() {
    // Z/2Z is a field, but 2 is even, so we cannot use MontgomeryInt.
    let i0_2 = VanillaInt::<u64>::new(0, &2).linalg();
    let i1_2 = i0_2.convert(1);
    let i2_2 = i0_2.convert(2);
    assert_eq!(i0_2, i2_2);
    assert_eq!(i1_2 + i2_2, i1_2);
    assert_eq!(i1_2 / i1_2, i1_2);

    // Z/4Z is not a field.
    let i0_4 = VanillaInt::<u64>::new(0, &4).linalg();
    let i1_4 = i0_4.convert(1);
    let i2_4 = i0_4.convert(2);
    let i3_4 = i0_4.convert(3);
    assert_eq!(i3_4, i1_4 / i3_4);
    assert!(panic::catch_unwind(|| {
        println!("{:?}", i1_4 / i2_4);
    })
    .is_err());
    assert!(panic::catch_unwind(|| {
        i2_4.inv();
    })
    .is_err());

    // Z/5Z is a field, and 5 is odd, so we can use MontgomeryInt.
    let i0_5 = MontgomeryInt::<u64>::new(0, &5).linalg();
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

    // Z/13Z is a field, and 13 is odd, so we can use MontgomeryInt.
    let i0_13 = MontgomeryInt::<u64>::new(0, &13).linalg();
    let i1_13 = i0_13.convert(1);
    let i2_13 = i0_13.convert(2);
    let i3_13 = i0_13.convert(3);
    let i4_13 = i0_13.convert(4);
    let i5_13 = i0_13.convert(5);
    let i6_13 = i0_13.convert(6);
    let i10_13 = i0_13.convert(10);
    let i12_13 = i0_13.convert(12);
    assert_eq!(i1_13 + i3_13, i4_13);
    assert_eq!(i2_13 * i3_13, i6_13);
    assert_eq!(i4_13 * i3_13, i12_13);
    assert_eq!(i4_13 / i3_13, i10_13);
    assert_eq!(i2_13 / i3_13, i5_13);
    assert_eq!(i2_13 / i2_13, i1_13);
}

#[test]
fn test_linalgreducedint_array() {
    let i0_5 = MontgomeryInt::<u64>::new(0, &5).linalg();
    let i1_5 = i0_5.convert(1);
    let i2_5 = i0_5.convert(2);
    let i3_5 = i0_5.convert(3);
    let i4_5 = i0_5.convert(4);

    let arr = array![[i0_5, i1_5], [i2_5, i3_5]];

    let arr_2 = arr.clone() * 4;
    let arr_2_ref = array![[i0_5, i4_5], [i3_5, i2_5]];
    assert_eq!(arr_2, arr_2_ref);

    let arr_3 = arr.clone() + arr_2;
    let arr_3_ref = array![[i0_5, i0_5], [i0_5, i0_5]];
    assert_eq!(arr_3, arr_3_ref);

    let arr_4 = -arr.clone();
    assert_eq!(arr_4, arr_2_ref);

    let arr_5 = arr.clone() * 12;
    let arr_5_ref = array![[i0_5, i2_5], [i4_5, i1_5]];
    assert_eq!(arr_5, arr_5_ref);

    let arr_6 = arr.clone() * i2_5.inv().residue();
    let arr_6_ref = array![[i0_5, i3_5], [i1_5, i4_5]];
    assert_eq!(arr_6, arr_6_ref);
}
