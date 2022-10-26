use num_traits::Pow;

use crate::chartab::character::Character;
use crate::chartab::unityroot::UnityRoot;

#[test]
fn test_character_equality() {
    let e3p1 = UnityRoot::new(1u64, 3u64);
    let e5p1 = UnityRoot::new(1u64, 5u64);
    let c1 = Character::new(&vec![(e3p1.clone(), 2usize), (e5p1.clone(), 1usize)]);
    let c2 = Character::new(&vec![(e5p1.clone(), 1usize), (e3p1.clone(), 2usize)]);
    assert_eq!(c1, c2);

    let e6p2 = UnityRoot::new(2u64, 6u64);
    let c3 = Character::new(&vec![(e5p1.clone(), 1usize), (e6p2.clone(), 2usize)]);
    assert_eq!(c1, c3);

    let c4 = Character::new(&vec![
        (e5p1.clone(), 1usize),
        (e6p2.clone(), 2usize),
        (e3p1.clone(), 2usize),
    ]);
    assert_eq!(c1, c4);

    let c5 = Character::new(&vec![
        (e5p1.clone(), 1usize),
        (e6p2.clone(), 2usize),
        (e3p1.clone(), 1usize),
    ]);
    assert_ne!(c1, c5);

    let e4 = UnityRoot::new(1u64, 4u64);
    let e4p0 = e4.pow(0);
    let e4p2 = e4.pow(2);
    let c6 = Character::new(&vec![(e4p0.clone(), 1usize)]);
    let c7 = Character::new(&vec![(e4p0.clone(), 2usize), (e4p2.clone(), 1usize)]);
    assert_eq!(c6, c7);

    let e4p1 = e4.pow(1);
    let e4p3 = e4.pow(3);
    let c8 = Character::new(&vec![(e4p3.clone(), 1usize)]);
    let c9 = Character::new(&vec![(e4p3.clone(), 2usize), (e4p1.clone(), 1usize)]);
    let c10 = Character::new(&vec![(e4p3.clone(), 2usize)]);
    assert_eq!(c8, c9);
    assert_ne!(c8, c10);

    let c11 = Character::new(&vec![(e4p3.clone(), 0usize)]);
    let c12 = Character::new(&vec![(e5p1.clone(), 0usize), (e6p2.clone(), 0usize)]);
    assert_eq!(c11, c12);
}

#[test]
fn test_character_partial_ord() {
    let e3 = UnityRoot::new(1u64, 3u64);
    let e3p0 = e3.pow(0);
    let e3p1 = e3.pow(1);
    let e3p2 = e3.pow(2);

    // Characters on the unit circle
    let c1 = Character::new(&vec![(e3p0.clone(), 1usize)]);
    let c2 = Character::new(&vec![(e3p1.clone(), 1usize)]);
    let c3 = Character::new(&vec![(e3p2.clone(), 1usize)]);
    assert!(c1 < c2);
    assert!(c1 < c3);
    assert!(c2 < c3);

    // Real characters
    let e4 = UnityRoot::new(1u64, 4u64);
    let e4p0 = e4.pow(0);
    let e4p2 = e4.pow(2);
    let c4 = Character::new(&vec![(e4p0.clone(), 1usize)]);
    let c5 = Character::new(&vec![(e4p0.clone(), 2usize)]);
    assert!(c4 < c5);

    // -1 < -2 since |-1| < |-2|.
    let c6 = Character::new(&vec![(e4p2.clone(), 1usize)]);
    let c7 = Character::new(&vec![(e4p2.clone(), 2usize)]);
    assert!(c6 < c7);

    let e7 = UnityRoot::new(1u64, 7u64);
    let e7p1 = e7.pow(1);
    let c8 = Character::new(&vec![(e7p1.clone(), 1usize)]);
    assert!(c8 < c7);

    let e7pm1 = e7.pow(-1);
    let c9 = Character::new(&vec![(e7pm1.clone(), 1usize)]);
    assert!(c8 < c9);

    let c10 = Character::new(&vec![(e7pm1.clone(), 0usize)]);
    assert!(c10 < c9);
}

#[test]
fn test_character_debug() {
    let e3 = UnityRoot::new(1u64, 3u64);
    let e3p0 = e3.pow(0);
    let e3p1 = e3.pow(1);
    let e3p2 = e3.pow(2);

    let c1 = Character::new(&vec![(e3p0.clone(), 1usize)]);
    assert_eq!(format!("{:?}", c1), "1".to_string());

    let c2 = Character::new(&vec![(e3p1.clone(), 1usize)]);
    assert_eq!(format!("{:?}", c2), "E3".to_string());

    let c3 = Character::new(&vec![(e3p2.clone(), 2usize)]);
    assert_eq!(format!("{:?}", c3), "2*E3^2".to_string());

    let c4 = Character::new(&vec![(e3p2.clone(), 2usize), (e3p0.clone(), 1usize)]);
    assert_eq!(format!("{:?}", c4), "1 + 2*E3^2".to_string());

    let c5 = Character::new(&vec![(e3p2.clone(), 0usize), (e3p1.clone(), 3usize)]);
    assert_eq!(format!("{:?}", c5), "3*E3".to_string());

    let c6 = Character::new(&vec![(e3p2.clone(), 0usize), (e3p1.clone(), 0usize)]);
    assert_eq!(format!("{:?}", c6), "0".to_string());
}

#[test]
fn test_character_fmt() {
    let e4 = UnityRoot::new(1u64, 4u64);
    let e4p0 = e4.pow(0);
    let e4p1 = e4.pow(1);
    let e4p2 = e4.pow(2);
    let e4p3 = e4.pow(3);

    let c0 = Character::new(&vec![(e4p0.clone(), 0usize)]);
    assert_eq!(format!("{}", c0), "0".to_string());
    let c0b = Character::new(&vec![(e4p0.clone(), 1usize), (e4p2.clone(), 1usize)]);
    assert_eq!(format!("{}", c0b), "0".to_string());
    let c0c = Character::new(&vec![(e4p1.clone(), 1usize), (e4p3.clone(), 1usize)]);
    assert_eq!(format!("{}", c0c), "0".to_string());
    let c1 = Character::new(&vec![(e4p0.clone(), 1usize)]);
    assert_eq!(format!("{}", c1), "+1".to_string());
    let c2 = Character::new(&vec![(e4p1.clone(), 2usize)]);
    assert_eq!(format!("{}", c2), "+2i".to_string());
    let c2b = Character::new(&vec![(e4p1.clone(), 1usize)]);
    assert_eq!(format!("{}", c2b), "+i".to_string());
    let c3 = Character::new(&vec![(e4p2.clone(), 3usize)]);
    assert_eq!(format!("{}", c3), "-3".to_string());
    let c4 = Character::new(&vec![(e4p3.clone(), 4usize)]);
    assert_eq!(format!("{}", c4), "-4i".to_string());
    let c4b = Character::new(&vec![(e4p3.clone(), 1usize)]);
    assert_eq!(format!("{}", c4b), "-i".to_string());

    let e3 = UnityRoot::new(1u64, 3u64);
    let e3p1 = e3.pow(1);
    let e3p2 = e3.pow(2);
    let c5 = Character::new(&vec![(e3p1.clone(), 1usize)]);
    assert_eq!(format!("{}", c5), "-0.500 + 0.866i".to_string());
    let c6 = Character::new(&vec![(e3p2.clone(), 1usize)]);
    assert_eq!(format!("{}", c6), "-0.500 - 0.866i".to_string());
    let c7 = Character::new(&vec![(e3p1.clone(), 1usize), (e3p2.clone(), 1usize)]);
    assert_eq!(format!("{}", c7), "-1".to_string());

    let e5 = UnityRoot::new(1u64, 5u64);
    let e5p1 = e5.pow(1);
    let c8 = Character::new(&vec![(e5p1.clone(), 1usize), (e5p1.complex_conjugate(), 1usize)]);
    assert_eq!(format!("{}", c8), "+0.618".to_string());
}
