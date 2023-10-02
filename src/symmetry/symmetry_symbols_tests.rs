use std::collections::HashSet;

use nalgebra::Vector3;

use crate::auxiliary::molecule::Molecule;
use crate::chartab::chartab_group::CharacterProperties;
use crate::chartab::chartab_symbols::{CollectionSymbol, DecomposedSymbol, MathematicalSymbol};
use crate::group::class::ClassProperties;
use crate::group::UnitaryRepresentedGroup;
use crate::symmetry::symmetry_core::{PreSymmetry, Symmetry};
use crate::symmetry::symmetry_element::symmetry_operation::{
    SpecialSymmetryTransformation, SymmetryOperation,
};
use crate::symmetry::symmetry_element::{RotationGroup, SymmetryElement, ROT, SIG};
use crate::symmetry::symmetry_element_order::ElementOrder;
use crate::symmetry::symmetry_group::SymmetryGroupProperties;
use crate::symmetry::symmetry_symbols::{
    deduce_mirror_parities, MirrorParity, MullikenIrcorepSymbol, MullikenIrrepSymbol,
    SymmetryClassSymbol,
};

const ROOT: &str = env!("CARGO_MANIFEST_DIR");

#[test]
fn test_symmetry_symbols_mulliken() {
    let a = MullikenIrrepSymbol::new("A").unwrap();
    assert_eq!(a.to_string(), "|A|");

    let a2 = MullikenIrrepSymbol::new("||A||").unwrap();
    assert_eq!(a2.to_string(), "|A|");
    assert_eq!(a, a2);

    let b1dash = MullikenIrrepSymbol::new("||B|^(')_(1)|").unwrap();
    assert_eq!(b1dash.to_string(), "|B|^(')_(1)");
    assert_ne!(a, b1dash);

    let t1g = MullikenIrrepSymbol::new("||T|_(1g)|").unwrap();
    assert_eq!(t1g.to_string(), "|T|_(1g)");
}

#[test]
fn test_symmetry_symbols_class() {
    let c3_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::new(3.0, 1e-14))
        .proper_power(1)
        .raw_axis(Vector3::new(1.0, 1.0, 1.0))
        .kind(ROT)
        .rotation_group(RotationGroup::SO3)
        .build()
        .unwrap();
    let c3 = SymmetryOperation::builder()
        .generating_element(c3_element.clone())
        .power(1)
        .build()
        .unwrap();
    let c3pm1 = SymmetryOperation::builder()
        .generating_element(c3_element)
        .power(-1)
        .build()
        .unwrap();

    let c3_cls = SymmetryClassSymbol::new("2||C3||", Some(vec![c3.clone()])).unwrap();
    assert_eq!(c3_cls.to_string(), "2|C3|");
    assert!(c3_cls.is_proper());
    assert_eq!(c3_cls.multiplicity(), Some(2));
    assert_eq!(c3_cls.size(), 2);

    let c3_cls2 = SymmetryClassSymbol::new("1||C3, C3^(-1)||", Some(vec![c3, c3pm1])).unwrap();
    assert_eq!(c3_cls2.to_string(), "|C3, C3^(-1)|");
    assert!(c3_cls2.is_proper());
    assert_eq!(c3_cls2.multiplicity(), Some(1));
    assert_eq!(c3_cls2.size(), 2);

    let i_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::new(2.0, 1e-14))
        .proper_power(1)
        .raw_axis(Vector3::new(1.0, 1.0, 1.0))
        .kind(SIG)
        .rotation_group(RotationGroup::SO3)
        .build()
        .unwrap();
    let i = SymmetryOperation::builder()
        .generating_element(i_element)
        .power(1)
        .build()
        .unwrap();

    let i_cls = SymmetryClassSymbol::new("1||i||", Some(vec![i])).unwrap();
    assert_eq!(i_cls.to_string(), "|i|");
    assert!(!i_cls.is_proper());
    assert!(i_cls.is_inversion());
    assert!(!i_cls.is_time_reversal());

    let s_element = SymmetryElement::builder()
        .threshold(1e-14)
        .proper_order(ElementOrder::new(1.0, 1e-14))
        .proper_power(1)
        .raw_axis(Vector3::new(1.0, 1.0, 1.0))
        .kind(SIG)
        .rotation_group(RotationGroup::SO3)
        .build()
        .unwrap();
    let s = SymmetryOperation::builder()
        .generating_element(s_element)
        .power(1)
        .build()
        .unwrap();

    let s_cls = SymmetryClassSymbol::new("1||σ|_(h)|", Some(vec![s])).unwrap();
    assert_eq!(s_cls.to_string(), "|σ|_(h)");
    assert!(!s_cls.is_proper());
    assert!(s_cls.is_reflection());
    assert!(!s_cls.is_time_reversal());
}

#[test]
fn test_symmetry_symbols_mulliken_ircorep_hashability() {
    let d1 = MullikenIrcorepSymbol::new("||E|_(g)| ⊕ ||T|_(2g)|").unwrap();
    let d2 = MullikenIrcorepSymbol::new("||T|_(2g)| ⊕ ||E|_(g)|").unwrap();
    let d3 = MullikenIrcorepSymbol::new("3||A|_(2g)| ⊕ 4||A|_(1g)|").unwrap();

    assert_eq!(format!("{d1}").as_str(), "D[|E|_(g) ⊕ |T|_(2g)]");
    assert_eq!(format!("{d2}").as_str(), "D[|T|_(2g) ⊕ |E|_(g)]");
    assert_eq!(format!("{d3}").as_str(), "D[3|A|_(2g) ⊕ 4|A|_(1g)]");

    assert_eq!(d1, d2);
    assert_ne!(d1, d3);

    let mut ds = HashSet::<MullikenIrcorepSymbol>::new();
    ds.insert(d1);
    assert_eq!(ds.len(), 1);
    ds.insert(d2);
    assert_eq!(ds.len(), 1);
    ds.insert(d3);
    assert_eq!(ds.len(), 2);

    let d4 = MullikenIrcorepSymbol::new("||T|_(2g)| ⊕ ||A|_(2g)| ⊕ ||A|_(1g)|").unwrap();
    let d5 = MullikenIrcorepSymbol::new("||A|_(1g)| ⊕ ||A|_(2g)| ⊕ ||T|_(2g)|").unwrap();
    assert_eq!(d4, d5);
    ds.insert(d4);
    assert_eq!(ds.len(), 3);
    ds.insert(d5);
    assert_eq!(ds.len(), 3);
}

#[test]
fn test_deduce_mirror_parities_d6h() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/benzene.xyz");
    let thresh = 1e-7;
    let mol = Molecule::from_xyz(&path, thresh);
    let presym = PreSymmetry::builder()
        .moi_threshold(thresh)
        .molecule(&mol)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false).unwrap();
    let group = UnitaryRepresentedGroup::from_molecular_symmetry(&sym, None).unwrap();

    let sh = group.get_cc_symbol_of_index(11).unwrap();
    let sv = group.get_cc_symbol_of_index(9).unwrap();
    let svd = group.get_cc_symbol_of_index(10).unwrap();

    let rep_a1g = DecomposedSymbol::<MullikenIrrepSymbol>::new("||A|_(1g)|").unwrap();
    let rep_a1g_ps = deduce_mirror_parities(&group, &rep_a1g);
    assert!(rep_a1g_ps.values().all(|p| *p == MirrorParity::Even));

    let rep_a2g = DecomposedSymbol::<MullikenIrrepSymbol>::new("||A|_(2g)|").unwrap();
    let rep_a2g_ps = deduce_mirror_parities(&group, &rep_a2g);
    assert_eq!(*rep_a2g_ps.get(&sh).unwrap(), MirrorParity::Even);
    assert_eq!(*rep_a2g_ps.get(&sv).unwrap(), MirrorParity::Odd);
    assert_eq!(*rep_a2g_ps.get(&svd).unwrap(), MirrorParity::Odd);

    let rep_b1g = DecomposedSymbol::<MullikenIrrepSymbol>::new("||B|_(1g)|").unwrap();
    let rep_b1g_ps = deduce_mirror_parities(&group, &rep_b1g);
    assert_eq!(*rep_b1g_ps.get(&sh).unwrap(), MirrorParity::Odd);
    assert_eq!(*rep_b1g_ps.get(&sv).unwrap(), MirrorParity::Even);
    assert_eq!(*rep_b1g_ps.get(&svd).unwrap(), MirrorParity::Odd);

    let rep_b2g = DecomposedSymbol::<MullikenIrrepSymbol>::new("||B|_(2g)|").unwrap();
    let rep_b2g_ps = deduce_mirror_parities(&group, &rep_b2g);
    assert_eq!(*rep_b2g_ps.get(&sh).unwrap(), MirrorParity::Odd);
    assert_eq!(*rep_b2g_ps.get(&sv).unwrap(), MirrorParity::Odd);
    assert_eq!(*rep_b2g_ps.get(&svd).unwrap(), MirrorParity::Even);

    let rep_e1g = DecomposedSymbol::<MullikenIrrepSymbol>::new("||E|_(1g)|").unwrap();
    let rep_e1g_ps = deduce_mirror_parities(&group, &rep_e1g);
    assert_eq!(*rep_e1g_ps.get(&sh).unwrap(), MirrorParity::Odd);
    assert_eq!(*rep_e1g_ps.get(&sv).unwrap(), MirrorParity::Neither);
    assert_eq!(*rep_e1g_ps.get(&svd).unwrap(), MirrorParity::Neither);

    let rep_e2g = DecomposedSymbol::<MullikenIrrepSymbol>::new("||E|_(2g)|").unwrap();
    let rep_e2g_ps = deduce_mirror_parities(&group, &rep_e2g);
    assert_eq!(*rep_e2g_ps.get(&sh).unwrap(), MirrorParity::Even);
    assert_eq!(*rep_e2g_ps.get(&sv).unwrap(), MirrorParity::Neither);
    assert_eq!(*rep_e2g_ps.get(&svd).unwrap(), MirrorParity::Neither);

    let rep_e1ge2g =
        DecomposedSymbol::<MullikenIrrepSymbol>::new("||E|_(1g)| ⊕ ||E|_(2g)|").unwrap();
    let rep_e1ge2g_ps = deduce_mirror_parities(&group, &rep_e1ge2g);
    assert_eq!(*rep_e1ge2g_ps.get(&sh).unwrap(), MirrorParity::Neither);
    assert_eq!(*rep_e1ge2g_ps.get(&sv).unwrap(), MirrorParity::Neither);
    assert_eq!(*rep_e1ge2g_ps.get(&svd).unwrap(), MirrorParity::Neither);
}

#[test]
fn test_deduce_mirror_parities_d5h() {
    let path: String = format!("{}{}", ROOT, "/tests/xyz/eclipsed_ferrocene.xyz");
    let thresh = 1e-6;
    let mol = Molecule::from_xyz(&path, thresh);
    let presym = PreSymmetry::builder()
        .moi_threshold(thresh)
        .molecule(&mol)
        .build()
        .unwrap();
    let mut sym = Symmetry::new();
    sym.analyse(&presym, false).unwrap();
    let group = UnitaryRepresentedGroup::from_molecular_symmetry(&sym, None).unwrap();
    println!("{}", group.character_table());

    let sh = group.get_cc_symbol_of_index(6).unwrap();
    let sv = group.get_cc_symbol_of_index(7).unwrap();

    let rep_a1d = DecomposedSymbol::<MullikenIrrepSymbol>::new("||A|^(')_(1)|").unwrap();
    let rep_a1d_ps = deduce_mirror_parities(&group, &rep_a1d);
    assert!(rep_a1d_ps.values().all(|p| *p == MirrorParity::Even));

    let rep_a2d = DecomposedSymbol::<MullikenIrrepSymbol>::new("||A|^(')_(2)|").unwrap();
    let rep_a2d_ps = deduce_mirror_parities(&group, &rep_a2d);
    assert_eq!(*rep_a2d_ps.get(&sh).unwrap(), MirrorParity::Even);
    assert_eq!(*rep_a2d_ps.get(&sv).unwrap(), MirrorParity::Odd);

    let rep_e1d = DecomposedSymbol::<MullikenIrrepSymbol>::new("||E|^(')_(1)|").unwrap();
    let rep_e1d_ps = deduce_mirror_parities(&group, &rep_e1d);
    assert_eq!(*rep_e1d_ps.get(&sh).unwrap(), MirrorParity::Even);
    assert_eq!(*rep_e1d_ps.get(&sv).unwrap(), MirrorParity::Neither);
}
