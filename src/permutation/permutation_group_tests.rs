use env_logger;
use itertools::Itertools;
use num_traits::ToPrimitive;

use crate::chartab::chartab_group::CharacterProperties;
use crate::chartab::chartab_symbols::{CollectionSymbol, LinearSpaceSymbol};
use crate::chartab::{CharacterTable, RepCharacterTable};
use crate::permutation::permutation_group::{
    PermutationClassSymbol, PermutationGroup, PermutationGroupProperties, PermutationIrrepSymbol,
};

fn test_irrep_character_table_algebraic_validity(
    chartab: &RepCharacterTable<PermutationIrrepSymbol, PermutationClassSymbol<u8>>,
) {
    let order: usize = chartab.classes.keys().map(|cc| cc.size()).sum();
    let order_i32 = order
        .to_i32()
        .expect("Unable to convert the group order to `i32`.");

    // Sum of squared dimensions
    assert_eq!(
        order,
        chartab
            .irreps
            .keys()
            .map(|irrep| irrep.dimensionality().pow(2))
            .sum()
    );

    // Square character table
    assert_eq!(chartab.array().nrows(), chartab.array().ncols());

    // Reality and integrality of characters
    let thresh = 1e-13 * (chartab.classes.len() as f64);
    let chartab_i = chartab.array().map(|chr| {
        let chr_c = chr.simplify().complex_value();
        let res = approx::relative_eq!(
            chr_c.re,
            chr_c.re.round(),
            epsilon = thresh,
            max_relative = thresh
        ) && approx::relative_eq!(chr_c.im, 0.0, epsilon = thresh, max_relative = thresh);
        if !res {
            panic!("{chr_c} is a non-integer.")
        } else {
            chr_c
                .re
                .round()
                .to_i32()
                .unwrap_or_else(|| panic!("Unable to convert {chr_c} to `i32`."))
        }
    });

    // First orthogonality theorem (row-orthogonality)
    assert!(chartab
        .irreps
        .keys()
        .combinations_with_replacement(2)
        .all(|irreps_pair| {
            let irrep_i = irreps_pair[0];
            let irrep_j = irreps_pair[1];
            let i = *chartab.irreps.get(irrep_i).unwrap();
            let j = *chartab.irreps.get(irrep_j).unwrap();
            let inprod_unnormed: i32 = chartab.classes.iter().fold(0i32, |acc, (cc, &k)| {
                let chr_ik = chartab_i[[i, k]];
                let chr_jk = chartab_i[[j, k]];
                acc + (cc.size() as i32) * chr_ik * chr_jk
            });
            assert_eq!(inprod_unnormed.rem_euclid(order_i32), 0);
            let inprod = inprod_unnormed.div_euclid(order_i32);

            if i == j {
                inprod == 1
            } else {
                inprod == 0
            }
        }));

    // Second orthogonality theorem (column-orthogonality)
    assert!(chartab
        .classes
        .keys()
        .combinations_with_replacement(2)
        .all(|ccs_pair| {
            let cc_i = ccs_pair[0];
            let cc_j = ccs_pair[1];
            let i = *chartab.classes.get(cc_i).unwrap();
            let j = *chartab.classes.get(cc_j).unwrap();
            let inprod_unnormed: i32 = chartab.irreps.iter().fold(0i32, |acc, (_, &k)| {
                let chr_ki = chartab_i[[k, i]];
                let chr_kj = chartab_i[[k, j]];
                acc + (cc_i.size() as i32) * chr_ki * chr_kj
            });
            assert_eq!(inprod_unnormed.rem_euclid(order_i32), 0);
            let inprod = inprod_unnormed.div_euclid(order_i32);

            if i == j {
                inprod == 1
            } else {
                inprod == 0
            }
        }));
}

#[test]
fn test_permutation_group_chartab() {
    // env_logger::init();
    for p in 1..=10 {
        let sym = PermutationGroup::from_rank(p);
        test_irrep_character_table_algebraic_validity(sym.character_table());
    }
}
