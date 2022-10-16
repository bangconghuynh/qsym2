use crate::chartab::unityroot::UnityRoot;


#[test]
fn test_unityroot_comparison() {
    let e3 = UnityRoot::new(1u64, 3u64);
    let e6p2 = UnityRoot::new(2u64, 6u64);
    assert_eq!(e3, e6p2);
}
