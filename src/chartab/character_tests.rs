use num_traits::Pow;

use crate::chartab::character::Character;
use crate::chartab::unityroot::UnityRoot;

#[test]
fn test_character_equality() {
    let e3p1 = UnityRoot::new(1u32, 3u32);
    let e5p1 = UnityRoot::new(1u32, 5u32);
    let c1 = Character::new(&[(e3p1.clone(), 2usize), (e5p1.clone(), 1usize)]);
    let c2 = Character::new(&[(e5p1.clone(), 1usize), (e3p1.clone(), 2usize)]);
    assert_eq!(c1, c2);

    let e6p2 = UnityRoot::new(2u32, 6u32);
    let c3 = Character::new(&[(e5p1.clone(), 1usize), (e6p2.clone(), 2usize)]);
    assert_eq!(c1, c3);

    let c4 = Character::new(&[
        (e5p1.clone(), 1usize),
        (e6p2.clone(), 1usize),
        (e3p1.clone(), 1usize),
    ]);
    assert_eq!(c1, c4);

    let c5 = Character::new(&[
        (e5p1.clone(), 1usize),
        (e6p2.clone(), 2usize),
        (e3p1, 1usize),
    ]);
    assert_ne!(c1, c5);

    let e4 = UnityRoot::new(1u32, 4u32);
    let e4p0 = e4.pow(0);
    let e4p2 = e4.pow(2);
    let c6 = Character::new(&[(e4p0.clone(), 1usize)]);
    let c7 = Character::new(&[(e4p0, 2usize), (e4p2, 1usize)]);
    assert_eq!(c6, c7);

    let e4p1 = e4.pow(1);
    let e4p3 = e4.pow(3);
    let c8 = Character::new(&[(e4p3.clone(), 1usize)]);
    let c9 = Character::new(&[(e4p3.clone(), 2usize), (e4p1, 1usize)]);
    let c10 = Character::new(&[(e4p3.clone(), 2usize)]);
    assert_eq!(c8, c9);
    assert_ne!(c8, c10);

    let c11 = Character::new(&[(e4p3, 0usize)]);
    let c12 = Character::new(&[(e5p1, 0usize), (e6p2, 0usize)]);
    assert_eq!(c11, c12);
}

#[test]
fn test_character_partial_ord() {
    let e3 = UnityRoot::new(1u32, 3u32);
    let e3p0 = e3.pow(0);
    let e3p1 = e3.pow(1);
    let e3p2 = e3.pow(2);

    // Characters on the unit circle
    let c1 = Character::new(&[(e3p0, 1usize)]);
    let c2 = Character::new(&[(e3p1, 1usize)]);
    let c3 = Character::new(&[(e3p2, 1usize)]);
    assert!(c1 < c2);
    assert!(c1 < c3);
    assert!(c2 < c3);

    // Real characters
    let e4 = UnityRoot::new(1u32, 4u32);
    let e4p0 = e4.pow(0);
    let e4p2 = e4.pow(2);
    let c4 = Character::new(&[(e4p0.clone(), 1usize)]);
    let c5 = Character::new(&[(e4p0, 2usize)]);
    assert!(c4 < c5);

    // -1 < -2 since |-1| < |-2|.
    let c6 = Character::new(&[(e4p2.clone(), 1usize)]);
    let c7 = Character::new(&[(e4p2, 2usize)]);
    assert!(c6 < c7);

    let e7 = UnityRoot::new(1u32, 7u32);
    let e7p1 = e7.pow(1);
    let c8 = Character::new(&[(e7p1, 1usize)]);
    assert!(c8 < c7);

    let e7pm1 = e7.pow(-1);
    let c9 = Character::new(&[(e7pm1.clone(), 1usize)]);
    assert!(c8 < c9);

    let c10 = Character::new(&[(e7pm1, 0usize)]);
    assert!(c10 < c9);
}

#[test]
fn test_character_partial_ord_advanced() {
    let e8 = UnityRoot::new(1u32, 8u32);
    let e8pi: Vec<_> = (0..8).map(|i| e8.pow(i)).collect();

    let c1 = Character::new(&[(e8pi[1].clone(), 1usize), (e8pi[7].clone(), 1usize)]);
    let c2 = Character::new(&[(e8pi[3].clone(), 1usize), (e8pi[5].clone(), 1usize)]);
    assert!(c1 < c2);

    println!("Trivial");
    let c3 = Character::new(&[(e8pi[0].clone(), 1usize)]);
    let c4 = Character::new(&[(e8pi[4].clone(), 1usize)]);
    println!("End trivial");
    assert!(c3 < c4);
}

#[test]
fn test_character_debug() {
    let e3 = UnityRoot::new(1u32, 3u32);
    let e3p0 = e3.pow(0);
    let e3p1 = e3.pow(1);
    let e3p2 = e3.pow(2);

    let c1 = Character::new(&[(e3p0.clone(), 1usize)]);
    assert_eq!(format!("{:?}", c1), "1".to_string());

    let c2 = Character::new(&[(e3p1.clone(), 1usize)]);
    assert_eq!(format!("{:?}", c2), "E3".to_string());

    let c3 = Character::new(&[(e3p2.clone(), 2usize)]);
    assert_eq!(format!("{:?}", c3), "2*(E3)^2".to_string());

    let c4 = Character::new(&[(e3p2.clone(), 2usize), (e3p0, 1usize)]);
    assert_eq!(format!("{:?}", c4), "1 + 2*(E3)^2".to_string());

    let c5 = Character::new(&[(e3p2.clone(), 0usize), (e3p1.clone(), 3usize)]);
    assert_eq!(format!("{:?}", c5), "3*E3".to_string());

    let c6 = Character::new(&[(e3p2, 0usize), (e3p1, 0usize)]);
    assert_eq!(format!("{:?}", c6), "0".to_string());

    let e7 = UnityRoot::new(1u32, 7u32);
    let c7 = Character::new(
        &(0..=6)
            .into_iter()
            .map(|x| (e7.pow(x), 1))
            .collect::<Vec<_>>(),
    );
    assert_eq!(
        format!("{:?}", c7),
        "1 + E7 + (E7)^2 + (E7)^3 + (E7)^4 + (E7)^5 + (E7)^6".to_string()
    );
}

#[test]
fn test_character_fmt() {
    let e4 = UnityRoot::new(1u32, 4u32);
    let e4p0 = e4.pow(0);
    let e4p1 = e4.pow(1);
    let e4p2 = e4.pow(2);
    let e4p3 = e4.pow(3);

    let c0 = Character::new(&[(e4p0.clone(), 0usize)]);
    assert_eq!(format!("{}", c0), "0".to_string());
    let c0b = Character::new(&[(e4p0.clone(), 1usize), (e4p2.clone(), 1usize)]);
    assert_eq!(format!("{}", c0b), "0".to_string());
    let c0c = Character::new(&[(e4p1.clone(), 1usize), (e4p3.clone(), 1usize)]);
    assert_eq!(format!("{}", c0c), "0".to_string());
    let c1 = Character::new(&[(e4p0, 1usize)]);
    assert_eq!(format!("{}", c1), "+1".to_string());
    let c2 = Character::new(&[(e4p1.clone(), 2usize)]);
    assert_eq!(format!("{}", c2), "+2i".to_string());
    let c2b = Character::new(&[(e4p1, 1usize)]);
    assert_eq!(format!("{}", c2b), "+i".to_string());
    let c3 = Character::new(&[(e4p2, 3usize)]);
    assert_eq!(format!("{}", c3), "-3".to_string());
    let c4 = Character::new(&[(e4p3.clone(), 4usize)]);
    assert_eq!(format!("{}", c4), "-4i".to_string());
    let c4b = Character::new(&[(e4p3, 1usize)]);
    assert_eq!(format!("{}", c4b), "-i".to_string());

    let e3 = UnityRoot::new(1u32, 3u32);
    let e3p1 = e3.pow(1);
    let e3p2 = e3.pow(2);
    let c5 = Character::new(&[(e3p1.clone(), 1usize)]);
    assert_eq!(format!("{}", c5), "E3".to_string());
    assert_eq!(
        format!("{}", c5.get_concise(true)),
        "-0.500 + 0.866i".to_string()
    );
    assert_eq!(
        c5.get_numerical(false, 5),
        "-0.50000 + 0.86603i".to_string()
    );
    let c6 = Character::new(&[(e3p2.clone(), 1usize)]);
    assert_eq!(format!("{}", c6), "(E3)^2".to_string());
    assert_eq!(
        format!("{}", c6.get_concise(true)),
        "-0.500 - 0.866i".to_string()
    );
    assert_eq!(
        c6.get_numerical(false, 5),
        "-0.50000 - 0.86603i".to_string()
    );
    let c7 = Character::new(&[(e3p1, 1usize), (e3p2, 1usize)]);
    assert_eq!(format!("{}", c7), "-1".to_string());
    assert_eq!(format!("{}", c7.get_concise(true)), "-1".to_string());
    assert_eq!(c7.get_numerical(false, 4), "-1.0000 + 0.0000i".to_string());

    let e5 = UnityRoot::new(1u32, 5u32);
    let e5p1 = e5.pow(1);
    let c8 = Character::new(&[(e5p1.clone(), 1usize), (e5p1.complex_conjugate(), 1usize)]);
    assert_eq!(format!("{}", c8), "E5 + (E5)^4".to_string());
    assert_eq!(format!("{}", c8.get_concise(true)), "+0.618".to_string());
    assert_eq!(
        c8.get_numerical(false, 6),
        "+0.618034 + 0.000000i".to_string()
    );

    let e7 = UnityRoot::new(1u32, 7u32);
    let c9 = Character::new(
        &(0..=6)
            .into_iter()
            .map(|x| (e7.pow(x), 1))
            .collect::<Vec<_>>(),
    );
    assert_eq!(format!("{}", c9), "0".to_string());
    assert_eq!(format!("{}", c9.get_concise(true)), "0".to_string());
    assert_eq!(
        c9.get_numerical(false, 7),
        "+0.0000000 + 0.0000000i".to_string()
    );
}

#[test]
fn test_character_simplify() {
    let e3 = UnityRoot::new(1u32, 3u32);
    let e3p0 = e3.pow(0);
    let e3p1 = e3.pow(1);
    let e3p2 = e3.pow(2);

    let e6 = UnityRoot::new(1u32, 6u32);
    let e6p0 = e6.pow(0);
    let e6p2 = e6.pow(2);
    let e6p3 = e6.pow(3);
    let e6p4 = e6.pow(4);
    let e6p5 = e6.pow(5);
    let e6p7 = e6.pow(7);

    let e12 = UnityRoot::new(1u32, 12u32);
    let e12p4 = e12.pow(4);

    let c1 = Character::new(&[(e3p0.clone(), 1usize), (e6p3.clone(), 2usize)]);
    let c1s = c1.simplify();
    assert_eq!(c1, c1s);
    assert_eq!(c1.terms.len(), 2);
    assert_eq!(c1s.terms.len(), 1);
    assert!(c1s.terms.contains_key(&e6p3));
    assert!(!c1s.terms.contains_key(&e3p0));

    let c2 = Character::new(&[(e6p0.clone(), 3usize), (e6p3.clone(), 3usize)]);
    let c2s = c2.simplify();
    assert_eq!(c2, c2s);
    assert_eq!(c2.terms.len(), 2);
    assert!(c2s.terms.is_empty());

    let c3 = Character::new(&[
        (e3p1.clone(), 3usize),
        (e3p2.clone(), 3usize),
        (e6p5.clone(), 2usize),
    ]);
    let c3s = c3.simplify();
    assert_eq!(c3, c3s);
    assert_eq!(c3.terms.len(), 3);
    assert_eq!(c3s.terms.len(), 2);
    assert!(c3s.terms.contains_key(&e3p1));
    assert!(c3s.terms.contains_key(&e3p2));
    assert!(!c3s.terms.contains_key(&e6p5));

    let c4 = Character::new(&[
        (e3p1.clone(), 3usize),
        (e3p2.clone(), 4usize),
        (e6p7.clone(), 4usize),
        (e6p5.clone(), 4usize),
    ]);
    let c4s = c4.simplify();
    assert_eq!(c4, c4s);
    assert_eq!(c4.terms.len(), 4);
    assert_eq!(c4s.terms.len(), 1);
    assert!(!c4s.terms.contains_key(&e3p1));
    assert!(!c4s.terms.contains_key(&e3p2));
    assert!(!c4s.terms.contains_key(&e6p7));
    assert!(c4s.terms.contains_key(&e6p5));

    let c5 = Character::new(&[
        (e3p1.clone(), 1usize),
        (e6p2.clone(), 1usize),
        (e12p4.clone(), 3usize),
        (e3p2.clone(), 2usize),
        (e6p4.clone(), 1usize),
        (e6p5.clone(), 5usize),
    ]);
    let c5s = c5.simplify();
    assert_eq!(c5, c5s);
    assert_eq!(c5.terms.len(), 3);
    assert_eq!(c5s.terms.len(), 1);
    assert!(!c5s.terms.contains_key(&e3p1));
    assert!(!c5s.terms.contains_key(&e6p2));
    assert!(!c5s.terms.contains_key(&e12p4));
    assert!(!c5s.terms.contains_key(&e6p5));
    assert!(c5s.terms.contains_key(&e3p2));
    assert!(c5s.terms.contains_key(&e6p4));
}
