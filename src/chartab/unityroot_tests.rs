use num_traits::Pow;
use std::collections::HashSet;

use crate::chartab::unityroot::UnityRoot;

#[test]
fn test_unityroot_equality() {
    let e3 = UnityRoot::new(1u32, 3u32);
    let e6p2 = UnityRoot::new(2u32, 6u32);
    assert_eq!(e3, e6p2);

    let e2 = UnityRoot::new(1u32, 2u32);
    let e8p4 = UnityRoot::new(4u32, 8u32);
    let e4p2 = UnityRoot::new(2u32, 4u32);
    assert_eq!(e2, e8p4);
    assert_eq!(e4p2, e8p4);

    let e8 = UnityRoot::new(1u32, 8u32);
    assert_eq!(e8.pow(4), e8p4);
    assert_eq!(e8.pow(-4), e8p4);
    assert_eq!(e8.pow(6), e8.pow(-2));
    assert_eq!(e8.pow(5), e8.pow(-3));
    assert_eq!(e8.pow(3), e8.pow(-5));
    assert_eq!(e8.pow(3), e8.pow(11));

    assert_eq!(e8.pow(0), e2.pow(0));
    assert_eq!(e8.pow(8), e2.pow(4));
    assert_eq!(e8.pow(0), e2.pow(-8));

    assert_eq!(e8.pow(4), e8.pow(4).complex_conjugate());
    assert_ne!(e8.pow(3), e8.pow(3).complex_conjugate());
}

#[test]
fn test_unityroot_hashing() {
    let mut s1: HashSet<UnityRoot> = HashSet::new();
    let e3 = UnityRoot::new(1u32, 3u32);
    let e6p2 = UnityRoot::new(2u32, 6u32);
    s1.insert(e3);
    assert_eq!(s1.len(), 1);
    s1.insert(e6p2);
    assert_eq!(s1.len(), 1);

    let e2 = UnityRoot::new(1u32, 2u32);
    let e2p1 = e2.pow(1);
    let e8p4 = UnityRoot::new(4u32, 8u32);
    let e4p2 = UnityRoot::new(2u32, 4u32);
    s1.insert(e2p1);
    s1.insert(e8p4);
    s1.insert(e4p2);
    assert_eq!(s1.len(), 2);

    let e8 = UnityRoot::new(1u32, 8u32);
    let e8p1 = e8.pow(1);
    s1.insert(e8p1);
    assert_eq!(s1.len(), 3);

    let e8p4 = e8.pow(4);
    s1.insert(e8p4);
    let e8pm4 = e8.pow(-4);
    s1.insert(e8pm4);
    assert_eq!(s1.len(), 3);

    let e8p6 = e8.pow(6);
    s1.insert(e8p6);
    assert_eq!(s1.len(), 4);

    let e4 = UnityRoot::new(1u32, 4u32);
    s1.insert(e4.pow(3));
    assert_eq!(s1.len(), 4);

    let e8p0 = e8.pow(0);
    let e2p2 = e2.pow(2);
    s1.insert(e8p0);
    s1.insert(e2p2);
    assert_eq!(s1.len(), 5);
}

#[test]
fn test_unityroot_partial_ord() {
    let e3 = UnityRoot::new(1u32, 3u32);
    let e3p0 = e3.pow(0);
    let e3p1 = e3.pow(1);
    assert!(e3p0 < e3p1);
    assert!(e3p0 <= e3p1);

    let e7 = UnityRoot::new(1u32, 7u32);
    let e7p1 = e7.pow(1);
    let e7p6 = e7.pow(6);
    let e7pm2 = e7.pow(-2);
    assert!(e7p1 < e3p1);
    assert!(e7p6 > e3p1);
    assert!(e7pm2 > e3p1);
}

#[test]
fn test_unityroot_mul() {
    let e3 = UnityRoot::new(1u32, 3u32);
    let e7 = UnityRoot::new(1u32, 7u32);
    let e21 = UnityRoot::new(1u32, 21u32);
    assert_eq!(&e3 * &e7, e21.pow(10));

    let e3pm1 = e3.pow(-1);
    assert_eq!(&e3pm1 * &e3pm1, e3pm1.pow(2));
    assert_eq!(&e3pm1 * &e3pm1, e3);
    assert_eq!(&e3pm1 * &e3, e3.pow(0));

    assert_eq!(&e3 * &e21.pow(15), e21);
    assert_eq!(&e3 * &e3.complex_conjugate(), e21.pow(0));
}

#[test]
fn test_unityroot_fmt() {
    let e4 = UnityRoot::new(1u32, 4u32);
    assert_eq!(format!("{}", e4.pow(0)), "1".to_string());
    assert_eq!(format!("{}", e4.pow(1)), "i".to_string());
    assert_eq!(format!("{}", e4.pow(2)), "-1".to_string());
    assert_eq!(format!("{}", e4.pow(3)), "-i".to_string());
    assert_eq!(format!("{:?}", e4.pow(0)), "1".to_string());
    assert_eq!(format!("{:?}", e4.pow(1)), "E4".to_string());
    assert_eq!(format!("{:?}", e4.pow(2)), "E2".to_string());
    assert_eq!(format!("{:?}", e4.pow(3)), "(E4)^3".to_string());

    let e6 = UnityRoot::new(1u32, 6u32);
    assert_eq!(format!("{}", e6.pow(0)), "1".to_string());
    assert_eq!(format!("{}", e6.pow(1)), "E6".to_string());
    assert_eq!(format!("{}", e6.pow(2)), "E3".to_string());
    assert_eq!(format!("{}", e6.pow(3)), "-1".to_string());
    assert_eq!(format!("{}", e6.pow(4)), "(E3)^2".to_string());
    assert_eq!(format!("{}", e6.pow(5)), "(E6)^5".to_string());
    assert_eq!(format!("{}", e6.pow(6)), "1".to_string());
    assert_eq!(format!("{:?}", e6.pow(0)), "1".to_string());
    assert_eq!(format!("{:?}", e6.pow(1)), "E6".to_string());
    assert_eq!(format!("{:?}", e6.pow(2)), "E3".to_string());
    assert_eq!(format!("{:?}", e6.pow(3)), "E2".to_string());
    assert_eq!(format!("{:?}", e6.pow(4)), "(E3)^2".to_string());
    assert_eq!(format!("{:?}", e6.pow(5)), "(E6)^5".to_string());
    assert_eq!(format!("{:?}", e6.pow(6)), "1".to_string());

    let e8 = UnityRoot::new(1u32, 8u32);
    assert_eq!(format!("{}", e8.pow(-2)), "-i".to_string());
    assert_eq!(format!("{}", e8.pow(-4)), "-1".to_string());
    assert_eq!(format!("{}", e8.pow(-6)), "i".to_string());
    assert_eq!(format!("{}", e8.pow(-8)), "1".to_string());
    assert_eq!(format!("{:?}", e8.pow(-2)), "(E4)^3".to_string());
    assert_eq!(format!("{:?}", e8.pow(-4)), "E2".to_string());
    assert_eq!(format!("{:?}", e8.pow(-6)), "E4".to_string());
    assert_eq!(format!("{:?}", e8.pow(-8)), "1".to_string());
    assert_eq!(format!("{:?}", e8.pow(10)), "E4".to_string());
}
