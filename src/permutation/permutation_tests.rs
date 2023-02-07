use itertools::Itertools;

use crate::permutation::Permutation;

#[test]
fn test_permutation_cycles() {
    let perms = (0..4).permutations(4).map(|image| Permutation::new(&image));
    for perm in perms.into_iter() {
        println!("{:?}: {:?}", perm.image, perm.cycle_pattern());
    }
}
