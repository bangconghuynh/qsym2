use std::collections::HashSet;
use num_traits::Pow;

use crate::chartab::unityroot::UnityRoot;


#[test]
fn test_unityroot_equality() {
    let e3 = UnityRoot::new(1u64, 3u64);
    let e6p2 = UnityRoot::new(2u64, 6u64);
    assert_eq!(e3, e6p2);

    let e2 = UnityRoot::new(1u64, 2u64);
    let e8p4 = UnityRoot::new(4u64, 8u64);
    let e4p2 = UnityRoot::new(2u64, 4u64);
    assert_eq!(e2, e8p4);
    assert_eq!(e4p2, e8p4);

    let e8 = UnityRoot::new(1u64, 8u64);
    assert_eq!(e8.pow(4), e8p4);
    assert_eq!(e8.pow(-4), e8p4);
    assert_eq!(e8.pow(6), e8.pow(-2));
    assert_eq!(e8.pow(5), e8.pow(-3));
    assert_eq!(e8.pow(3), e8.pow(-5));
    assert_eq!(e8.pow(3), e8.pow(11));
}


#[test]
fn test_unityroot_hashing() {
    let mut s1: HashSet<UnityRoot> = HashSet::new();
    let e3 = UnityRoot::new(1u64, 3u64);
    let e6p2 = UnityRoot::new(2u64, 6u64);
    s1.insert(e3);
    assert_eq!(s1.len(), 1);
    s1.insert(e6p2);
    assert_eq!(s1.len(), 1);

    let e2 = UnityRoot::new(1u64, 2u64);
    let e8p4 = UnityRoot::new(4u64, 8u64);
    let e4p2 = UnityRoot::new(2u64, 4u64);
    s1.insert(e2);
    s1.insert(e8p4);
    s1.insert(e4p2);
    assert_eq!(s1.len(), 2);

    let e8 = UnityRoot::new(1u64, 8u64);
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

    let e4 = UnityRoot::new(1u64, 4u64);
    s1.insert(e4.pow(3));
    assert_eq!(s1.len(), 4);
}
