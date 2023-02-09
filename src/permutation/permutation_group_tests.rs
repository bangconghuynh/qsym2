use env_logger;

use crate::group::UnitaryRepresentedGroup;
use crate::chartab::chartab_group::CharacterProperties;
use crate::permutation::permutation_group::PermutationGroupProperties;

#[test]
fn test_permutation_group_construction() {
    env_logger::init();
    let sym3 = UnitaryRepresentedGroup::from_rank(7);
    println!("{}", sym3.character_table());
}
