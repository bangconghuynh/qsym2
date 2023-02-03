use std::collections::HashSet;

use crate::symmetry::symmetry_symbols::MullikenIrcorepSymbol;

#[test]
fn test_symmetry_symbols_mulliken_ircorep_hashability() {
    let d1 = MullikenIrcorepSymbol::new("||E|_(g)| + ||T|_(2g)|").unwrap();
    let d2 = MullikenIrcorepSymbol::new("||T|_(2g)| + ||E|_(g)|").unwrap();
    let d3 = MullikenIrcorepSymbol::new("3||A|_(2g)| + 4||A|_(1g)|").unwrap();



    assert_eq!(format!("{d1}").as_str(), "D[|E|_(g) ⊕ |T|_(2g)]");
    assert_eq!(format!("{d2}").as_str(), "D[|E|_(g) ⊕ |T|_(2g)]");
    assert_eq!(format!("{d3}").as_str(), "D[4|A|_(1g) ⊕ 3|A|_(2g)]");

    assert_eq!(d1, d2);
    assert_ne!(d1, d3);

    let mut ds = HashSet::<MullikenIrcorepSymbol>::new();
    ds.insert(d1);
    assert_eq!(ds.len(), 1);
    ds.insert(d2);
    assert_eq!(ds.len(), 1);
    ds.insert(d3);
    assert_eq!(ds.len(), 2);

    let d4 = MullikenIrcorepSymbol::new("||T|_(2g)| + ||A|_(2g)| + ||A|_(1g)|").unwrap();
    let d5 = MullikenIrcorepSymbol::new("||A|_(1g)| + ||A|_(2g)| + ||T|_(2g)|").unwrap();
    assert_eq!(d4, d5);
    ds.insert(d4);
    assert_eq!(ds.len(), 3);
    ds.insert(d5);
    assert_eq!(ds.len(), 3);
}
