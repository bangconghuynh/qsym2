use std::collections::HashSet;
use std::fmt;
use std::hash::Hash;
use std::ops::Mul;

use indexmap::IndexMap;
use itertools::Itertools;
use ndarray::{Array2, Array3, Axis, Zip};

use crate::symmetry::symmetry_symbols::ClassSymbol;

pub trait GroupStructure<T>
where
    T: Hash + Eq + Clone + fmt::Debug + Sync + Send,
    for<'a, 'b> &'b T: Mul<&'a T, Output = T>,
{
    fn elements(&self) -> &IndexMap<T, usize>;

    fn cayley_table(&self) -> Option<&Array2<usize>>;

    fn conjugacy_classes(&self) -> Option<&Vec<HashSet<usize>>>;

    fn conjugacy_class_transversal(&self) -> Option<&Vec<usize>>;

    fn conjugacy_class_symbols(&self) -> Option<&IndexMap<ClassSymbol<T>, usize>>;

    fn inverse_conjugacy_classes(&self) -> Option<&Vec<usize>>;

    fn element_to_conjugacy_classes(&self) -> Option<&Vec<usize>>;

    fn class_number(&self) -> Option<usize>;

    fn class_matrix(&self) -> Option<&Array3<usize>>;

    /// Find the conjugacy classes and their inverses for the group.
    /////
    ///// This method sets the [`Self::conjugacy_classes`], [`Self::inverse_conjugacy_classes`],
    ///// [`Self::conjugacy_class_transversal`], [`Self::element_to_conjugacy_classes`], and
    ///// [`Self::class_number`] fields.
    fn compute_conjugacy_classes(&mut self) -> Vec<HashSet<usize>>;

    /// Determines the order of the group.
    fn compute_order(&self) -> usize {
        self.elements().len()
    }

    /// Constructs the Cayley table for the group.
    fn compute_cayley_table(&self) -> Array2<usize> {
        log::debug!("Constructing Cayley table in parallel...");
        let mut ctb = Array2::<usize>::zeros((self.compute_order(), self.compute_order()));
        let elements = self.elements().clone();
        Zip::indexed(&mut ctb).par_for_each(|(i, j), k| {
            let (op_i_ref, _) = elements
                .get_index(i)
                .unwrap_or_else(|| panic!("Element with index {i} cannot be retrieved."));
            let (op_j_ref, _) = elements
                .get_index(j)
                .unwrap_or_else(|| panic!("Element with index {j} cannot be retrieved."));
            let op_k = op_i_ref * op_j_ref;
            *k = *elements
                .get(&op_k)
                .unwrap_or_else(|| panic!("Group closure not fulfilled. The composition {:?} * {:?} = {:?} is not contained in the group. Try changing thresholds.",
                        op_i_ref,
                        op_j_ref,
                        &op_k));
        });
        log::debug!("Constructing Cayley table in parallel... Done.");
        ctb
    }

    /// Assigns generic class symbols to the conjugacy classes.
    fn compute_class_symbols(&mut self) -> IndexMap<ClassSymbol<T>, usize> {
        log::debug!("Assigning generic class symbols...");
        let class_sizes: Vec<_> = self
            .conjugacy_classes()
            .expect("Conjugacy classes not found.")
            .iter()
            .map(HashSet::len)
            .collect();
        let class_symbols_iter = self
            .conjugacy_class_transversal()
            .expect("Conjugacy class transversals not found.")
            .iter()
            .enumerate()
            .map(|(i, &rep_ele_index)| {
                let (rep_ele, _) = self.elements().get_index(rep_ele_index).unwrap_or_else(|| {
                    panic!("Element with index {rep_ele_index} cannot be retrieved.")
                });
                (
                    ClassSymbol::new(
                        format!("{}||K{i}||", class_sizes[i]).as_str(),
                        Some(rep_ele.clone()),
                    )
                    .unwrap_or_else(|_| {
                        panic!(
                            "Unable to construct a class symbol from `{}||K{i}||`.",
                            class_sizes[i]
                        )
                    }),
                    i,
                )
            });
        log::debug!("Assigning generic class symbols... Done.");
        class_symbols_iter.collect::<IndexMap<_, _>>()
    }

    /// Calculates the class matrix $`\mathbf{N}`$ for the conjugacy classes in
    /// the group.
    ///
    /// Let $`K_i`$ be the $`i^{\textrm{th}}`conjugacy class of the group. The
    /// elements of the class matrix $`\mathbf{N}`$ are given by
    ///
    /// ```math
    ///     N_{rst} = \lvert \{ (x, y) \in K_r \times K_s : xy = z \in K_t \} \rvert,
    /// ```
    ///
    /// independent of any $`z \in K_t`$.
    ///
    /// This method sets the [`Self::class_matrix`] field.
    fn compute_class_matrix(&mut self) -> Array3<usize> {
        let mut nmat = Array3::<usize>::zeros((
            self.class_number().expect("Class number not found."),
            self.class_number().expect("Class number not found."),
            self.class_number().expect("Class number not found."),
        ));
        for (r, class_r) in self
            .conjugacy_classes()
            .expect("Conjugacy classes not found.")
            .iter()
            .enumerate()
        {
            let idx_r = class_r.iter().copied().collect::<Vec<_>>();
            for (s, class_s) in self
                .conjugacy_classes()
                .expect("Conjugacy classes not found.")
                .iter()
                .enumerate()
            {
                let idx_s = class_s.iter().copied().collect::<Vec<_>>();
                let cayley_block_rs = self
                    .cayley_table()
                    .expect("Cayley table not found.")
                    .select(Axis(0), &idx_r)
                    .select(Axis(1), &idx_s)
                    .iter()
                    .copied()
                    .counts();

                for (t, class_t) in self
                    .conjugacy_classes()
                    .expect("Conjugacy classes not found.")
                    .iter()
                    .enumerate()
                {
                    nmat[[r, s, t]] = *cayley_block_rs
                        .get(
                            class_t
                                .iter()
                                .next()
                                .expect("No conjugacy classes can be empty."),
                        )
                        .unwrap_or(&0);
                }
            }
        }
        nmat
    }
}
