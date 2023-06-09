use itertools::Itertools;
use num_traits::{Inv, Pow};

use crate::permutation::Permutation;

#[test]
fn test_permutation_cycles() {
    let p_01234 = Permutation::<u8>::from_image(vec![0, 1, 2, 3, 4]);
    assert!(p_01234.is_identity());
    assert_eq!(p_01234.cycles(), &[[0], [1], [2], [3], [4]]);
    assert_eq!(p_01234.cycle_pattern(), &[1, 1, 1, 1, 1]);

    let p_32104 = Permutation::<u8>::from_image(vec![3, 2, 1, 0, 4]);
    assert!(!p_32104.is_identity());
    assert_eq!(p_32104.cycles(), &[vec![0, 3], vec![1, 2], vec![4]]);
    assert_eq!(p_32104.cycle_pattern(), &[2, 2, 1]);

    let p_32104_2 = Permutation::<u8>::from_cycles(&p_32104.cycles());
    assert_eq!(p_32104, p_32104_2);

    let p_12340 = Permutation::<u8>::from_image(vec![1, 2, 3, 4, 0]);
    assert_eq!(p_12340.cycles(), &[vec![0, 1, 2, 3, 4]]);
    assert_eq!(p_12340.cycle_pattern(), &[5]);

    let rank = 9u8;
    let perms = (0..rank)
        .permutations(usize::from(rank))
        .map(Permutation::<u8>::from_image);
    for perm in perms {
        assert_eq!(perm, Permutation::<u8>::from_cycles(&perm.cycles()));
    }
}

#[test]
fn test_permutation_composition() {
    let p_01234 = Permutation::<u8>::from_image(vec![0, 1, 2, 3, 4]);
    let p_32104 = Permutation::<u8>::from_image(vec![3, 2, 1, 0, 4]);
    assert_eq!(p_32104, &p_01234 * &p_32104);
    assert_eq!(p_32104, &p_32104 * &p_01234);

    let p_04213 = Permutation::<u8>::from_image(vec![0, 4, 2, 1, 3]);
    let p_34120 = Permutation::<u8>::from_image(vec![3, 4, 1, 2, 0]);
    let p_12403 = Permutation::<u8>::from_image(vec![1, 2, 4, 0, 3]);
    assert_eq!(p_34120, &p_32104 * &p_04213);
    assert_eq!(p_12403, &p_04213 * &p_32104);

    let p_12340 = Permutation::<u8>::from_image(vec![1, 2, 3, 4, 0]);
    let p_23401 = Permutation::<u8>::from_image(vec![2, 3, 4, 0, 1]);
    assert_eq!(p_23401, &p_12340 * &p_12340);
    assert_eq!(p_23401, (&p_12340).pow(2));
    assert_eq!(&p_23401 * &p_12340, (&p_12340).pow(3));
    assert_eq!((&p_23401).inv(), (&p_12340).pow(-2));
    assert_eq!((&p_23401).inv() * &p_12340, (&p_12340).pow(-1));

    assert_eq!(p_01234, &p_12340 * (&p_12340).inv());
    assert_eq!(p_01234, (&p_12340).pow(0));

    let rank = 8u8;
    let perms = (0..rank)
        .permutations(usize::from(rank))
        .map(Permutation::<u8>::from_image);
    for perm in perms {
        assert!((&perm * (&perm).inv()).is_identity());
    }
}

#[test]
fn test_permutation_lehmer() {
    let p201 = Permutation::<u8>::from_image(vec![2, 0, 1]);
    assert_eq!(&p201.lehmer(None), &[2, 0, 0]);

    let p1506432 = Permutation::<u8>::from_image(vec![1, 5, 0, 6, 3, 4, 2]);
    assert_eq!(&p1506432.lehmer(None), &[1, 4, 0, 3, 1, 1, 0]);

    let p2154603 = Permutation::<u8>::from_image(vec![2, 1, 5, 4, 6, 0, 3]);
    assert_eq!(&p2154603.lehmer(None), &[2, 1, 3, 2, 2, 0, 0]);
    assert_eq!(p2154603.lehmer_index(None), 1648);
    assert_eq!(Permutation::<u8>::from_lehmer_index(1648, 7).unwrap(), p2154603);

    let p012345678 = Permutation::<u8>::from_image(vec![0, 1, 2, 3, 4, 5, 6, 7, 8]);
    assert_eq!(p012345678.lehmer_index(None), 0);
    assert_eq!(Permutation::<u8>::from_lehmer_index(0, 9), Some(p012345678));

    let p2154603d = Permutation::<u8>::from_lehmer(p2154603.lehmer(None));
    assert_eq!(p2154603, p2154603d);

    let p24107635 = Permutation::<u8>::from_image(vec![2, 4, 1, 0, 7, 6, 3, 5]);
    assert_eq!(&p24107635.lehmer(None), &[2, 3, 1, 0, 3, 2, 0, 0]);

    let p24107635d = Permutation::<u8>::from_lehmer(p24107635.lehmer(None));
    assert_eq!(p24107635, p24107635d);
}
