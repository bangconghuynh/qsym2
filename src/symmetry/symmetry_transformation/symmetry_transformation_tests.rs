use ndarray::{array, Axis};

use crate::aux::ao_basis::{BasisAngularOrder, BasisAtom, BasisShell, ShellOrder};
use crate::aux::atom::{Atom, ElementMap};
use crate::permutation::Permutation;
use crate::symmetry::symmetry_transformation::permute_array_by_atoms;

#[test]
fn test_symmetry_transformation_permute_array_by_atoms() {
    let emap = ElementMap::new();
    let atm_c0 = Atom::from_xyz("C 1.0 0.0 0.0", &emap, 1e-7).unwrap();
    let atm_c1 = Atom::from_xyz("C 0.0 1.0 0.0", &emap, 1e-7).unwrap();
    let atm_c2 = Atom::from_xyz("C -1.0 0.0 0.0", &emap, 1e-7).unwrap();
    let atm_c3 = Atom::from_xyz("C 0.0 -1.0 0.0", &emap, 1e-7).unwrap();

    let bs1s_p = BasisShell::new(0, ShellOrder::Pure(true));
    let bs2s_p = BasisShell::new(0, ShellOrder::Pure(true));
    let bs2p_p = BasisShell::new(1, ShellOrder::Pure(true));

    let batm_c0 = BasisAtom::new(&atm_c0, &[bs1s_p.clone(), bs2s_p.clone(), bs2p_p.clone()]);
    let batm_c1 = BasisAtom::new(&atm_c1, &[bs1s_p.clone(), bs2s_p.clone(), bs2p_p.clone()]);
    let batm_c2 = BasisAtom::new(&atm_c2, &[bs1s_p.clone(), bs2s_p.clone(), bs2p_p.clone()]);
    let batm_c3 = BasisAtom::new(&atm_c3, &[bs1s_p.clone(), bs2s_p.clone(), bs2p_p]);

    let bao_c4 = BasisAngularOrder::new(&[batm_c0, batm_c1, batm_c2, batm_c3]);
    let arr = array![
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0],
        [3.0, 3.0, 3.0],
        [4.0, 4.0, 4.0],
        [5.0, 5.0, 5.0],
        [6.0, 6.0, 6.0],
        [7.0, 7.0, 7.0],
        [8.0, 8.0, 8.0],
        [9.0, 9.0, 9.0],
        [10.0, 10.0, 10.0],
        [11.0, 11.0, 11.0],
        [12.0, 12.0, 12.0],
        [13.0, 13.0, 13.0],
        [14.0, 14.0, 14.0],
        [15.0, 15.0, 15.0],
        [16.0, 16.0, 16.0],
        [17.0, 17.0, 17.0],
        [18.0, 18.0, 18.0],
        [19.0, 19.0, 19.0],
    ];

    let perm0 = Permutation::<usize>::from_image(vec![1, 0, 3, 2]);
    let perm_arr_0 = permute_array_by_atoms(&arr, &perm0, &[Axis(0)], &bao_c4);
    assert_eq!(
        perm_arr_0,
        array![
            [5.0, 5.0, 5.0],
            [6.0, 6.0, 6.0],
            [7.0, 7.0, 7.0],
            [8.0, 8.0, 8.0],
            [9.0, 9.0, 9.0],
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
            [3.0, 3.0, 3.0],
            [4.0, 4.0, 4.0],
            [15.0, 15.0, 15.0],
            [16.0, 16.0, 16.0],
            [17.0, 17.0, 17.0],
            [18.0, 18.0, 18.0],
            [19.0, 19.0, 19.0],
            [10.0, 10.0, 10.0],
            [11.0, 11.0, 11.0],
            [12.0, 12.0, 12.0],
            [13.0, 13.0, 13.0],
            [14.0, 14.0, 14.0],
        ]
    );

    let perm1 = Permutation::<usize>::from_image(vec![1, 3, 2, 0]);
    let perm_arr_1 = permute_array_by_atoms(&arr, &perm1, &[Axis(0)], &bao_c4);
    assert_eq!(
        perm_arr_1,
        array![
            [5.0, 5.0, 5.0],
            [6.0, 6.0, 6.0],
            [7.0, 7.0, 7.0],
            [8.0, 8.0, 8.0],
            [9.0, 9.0, 9.0],
            [15.0, 15.0, 15.0],
            [16.0, 16.0, 16.0],
            [17.0, 17.0, 17.0],
            [18.0, 18.0, 18.0],
            [19.0, 19.0, 19.0],
            [10.0, 10.0, 10.0],
            [11.0, 11.0, 11.0],
            [12.0, 12.0, 12.0],
            [13.0, 13.0, 13.0],
            [14.0, 14.0, 14.0],
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
            [3.0, 3.0, 3.0],
            [4.0, 4.0, 4.0],
        ]
    );

    let atm_h0 = Atom::from_xyz("H 1.0 0.0 0.0", &emap, 1e-7).unwrap();
    let atm_h1 = Atom::from_xyz("H 0.0 1.0 0.0", &emap, 1e-7).unwrap();
    let atm_h2 = Atom::from_xyz("H 0.0 0.0 0.0", &emap, 1e-7).unwrap();

    let batm_h0 = BasisAtom::new(&atm_h0, &[bs1s_p.clone(), bs2s_p.clone()]);
    let batm_h1 = BasisAtom::new(&atm_h1, &[bs1s_p.clone(), bs2s_p.clone()]);
    let batm_h2 = BasisAtom::new(&atm_h2, &[bs1s_p, bs2s_p]);

    let bao_h3 = BasisAngularOrder::new(&[batm_h0, batm_h1, batm_h2]);
    let arr2 = array![
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        [6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
        [12.0, 13.0, 14.0, 15.0, 16.0, 17.0],
        [18.0, 19.0, 20.0, 21.0, 22.0, 23.0],
        [24.0, 25.0, 26.0, 27.0, 28.0, 29.0],
        [30.0, 31.0, 32.0, 33.0, 34.0, 35.0],
    ];

    let perm2 = Permutation::<usize>::from_image(vec![1, 0, 2]);
    let perm_arr2_0 = permute_array_by_atoms(&arr2, &perm2, &[Axis(0), Axis(1)], &bao_h3);
    assert_eq!(
        perm_arr2_0,
        array![
            [14.0, 15.0, 12.0, 13.0, 16.0, 17.0],
            [20.0, 21.0, 18.0, 19.0, 22.0, 23.0],
            [2.0, 3.0, 0.0, 1.0, 4.0, 5.0],
            [8.0, 9.0, 6.0, 7.0, 10.0, 11.0],
            [26.0, 27.0, 24.0, 25.0, 28.0, 29.0],
            [32.0, 33.0, 30.0, 31.0, 34.0, 35.0],
        ]
    );
}
