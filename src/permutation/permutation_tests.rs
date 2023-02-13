use itertools::Itertools;
use num_traits::{Inv, Pow};

use crate::permutation::Permutation;

#[test]
fn test_permutation_cycles() {
    let p_01234 = Permutation::from_image(&[0, 1, 2, 3, 4]);
    assert!(p_01234.is_identity());
    assert_eq!(p_01234.cycles(), &[[0], [1], [2], [3], [4]]);
    assert_eq!(p_01234.cycle_pattern(), &[1, 1, 1, 1, 1]);

    let p_32104 = Permutation::from_image(&[3, 2, 1, 0, 4]);
    assert!(!p_32104.is_identity());
    assert_eq!(p_32104.cycles(), &[vec![0, 3], vec![1, 2], vec![4]]);
    assert_eq!(p_32104.cycle_pattern(), &[2, 2, 1]);

    let p_32104_2 = Permutation::from_cycles(p_32104.cycles());
    assert_eq!(p_32104, p_32104_2);

    let p_12340 = Permutation::from_image(&[1, 2, 3, 4, 0]);
    assert_eq!(p_12340.cycles(), &[vec![0, 1, 2, 3, 4]]);
    assert_eq!(p_12340.cycle_pattern(), &[5]);

    let rank = 9;
    let perms = (0..rank)
        .permutations(rank)
        .map(|image| Permutation::from_image(&image));
    for perm in perms.into_iter() {
        assert_eq!(perm, Permutation::from_cycles(&perm.cycles()));
    }
}

#[test]
fn test_permutation_composition() {
    let p_01234 = Permutation::from_image(&[0, 1, 2, 3, 4]);
    let p_32104 = Permutation::from_image(&[3, 2, 1, 0, 4]);
    assert_eq!(p_32104, &p_01234 * &p_32104);
    assert_eq!(p_32104, &p_32104 * &p_01234);

    let p_04213 = Permutation::from_image(&[0, 4, 2, 1, 3]);
    let p_34120 = Permutation::from_image(&[3, 4, 1, 2, 0]);
    let p_12403 = Permutation::from_image(&[1, 2, 4, 0, 3]);
    assert_eq!(p_34120, &p_32104 * &p_04213);
    assert_eq!(p_12403, &p_04213 * &p_32104);

    let p_12340 = Permutation::from_image(&[1, 2, 3, 4, 0]);
    let p_23401 = Permutation::from_image(&[2, 3, 4, 0, 1]);
    assert_eq!(p_23401, &p_12340 * &p_12340);
    assert_eq!(p_23401, (&p_12340).pow(2));
    assert_eq!(&p_23401 * &p_12340, (&p_12340).pow(3));
    assert_eq!((&p_23401).inv(), (&p_12340).pow(-2));
    assert_eq!((&p_23401).inv() * &p_12340, (&p_12340).pow(-1));

    assert_eq!(p_01234, &p_12340 * (&p_12340).inv());
    assert_eq!(p_01234, (&p_12340).pow(0));

    let rank = 8;
    let perms = (0..rank)
        .permutations(rank)
        .map(|image| Permutation::from_image(&image));
    for perm in perms.into_iter() {
        assert!((&perm * (&perm).inv()).is_identity());
    }
}
