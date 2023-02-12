use env_logger;

use crate::chartab::chartab_group::CharacterProperties;
use crate::permutation::permutation_group::{PermutationGroupProperties, PermutationGroup};

#[test]
fn test_permutation_group_construction() {
    env_logger::init();
    let sym3 = PermutationGroup::from_rank(10);
    println!("{:?}", sym3.character_table());
}
