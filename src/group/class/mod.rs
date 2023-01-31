use std::collections::HashSet;

use derive_builder::Builder;
use indexmap::IndexMap;
use itertools::Itertools;
use ndarray::{s, Array2, Array3, Axis};

use crate::symmetry::symmetry_symbols::ClassSymbol;

#[derive(Builder)]
struct ClassStructure<'a, T: Clone> {
    elements: &'a IndexMap<T, usize>,

    cayley_table: &'a Array2<usize>,

    /// A vector of conjugacy classes for this group.
    ///
    /// Each element in the vector is a hashset containing the indices of the
    /// elements in [`Self::elements`] for a particular conjugacy class. This
    /// thus defines a multi-valued map from each conjugacy class index to one
    /// or more element indices.
    conjugacy_classes: Vec<HashSet<usize>>,

    /// The conjugacy class index of the elements in [`Self::elements`].
    ///
    /// This is the so-called inverse of [`Self::conjugacy_classes`]. This maps
    /// each element index to its corresponding conjugacy class index.
    element_to_conjugacy_classes: Vec<usize>,

    /// The conjugacy class representatives of the group.
    ///
    /// Each element in the vector is an index for a representative element of the corresponding
    /// conjugacy class.
    #[builder(setter(custom))]
    conjugacy_class_transversal: Vec<usize>,

    /// An index map of symbols for the conjugacy classes in this group.
    ///
    /// Each key in the index map is a class symbol, and the associated value is the index of
    /// the corresponding conjugacy class in [`Self::conjugacy_classes`].
    #[builder(setter(custom))]
    conjugacy_class_symbols: IndexMap<ClassSymbol<T>, usize>,

    /// A vector containing the indices of inverse conjugacy classes.
    ///
    /// Each index gives the inverse conjugacy class for the corresponding
    /// conjugacy class.
    #[builder(setter(custom))]
    inverse_conjugacy_classes: Vec<usize>,

    /// The class matrices $`\mathbf{N}`$ for the conjugacy classes in the group.
    ///
    /// Let $`K_i`$ be the $`i^{\textrm{th}}`conjugacy class of the group. The
    /// elements of the class matrix $`\mathbf{N}`$ are given by
    ///
    /// ```math
    ///     N_{rst} = \lvert \{ (x, y) \in K_r \times K_s : xy = z \in K_t \} \rvert,
    /// ```
    ///
    /// independent of any $`z \in K_t`$.
    #[builder(setter(custom))]
    class_matrix: Array3<usize>,
}

impl<'a, T> ClassStructureBuilder<'a, T>
where
    T: Clone,
{
    fn conjugacy_class_transversal(&mut self) -> &mut Self {
        self.conjugacy_class_transversal = Some(
            self.conjugacy_classes
                .as_ref()
                .expect("Conjugacy classes have not been found.")
                .iter()
                .map(|cc| {
                    *cc.iter()
                        .next()
                        .expect("No conjugacy classes can be empty.")
                })
                .collect::<Vec<usize>>(),
        );
        self
    }

    fn conjugacy_class_symbols(&mut self) -> &mut Self {
        log::debug!("Assigning generic class symbols...");
        let class_sizes: Vec<_> = self
            .conjugacy_classes
            .as_ref()
            .expect("Conjugacy classes have not been found.")
            .iter()
            .map(HashSet::len)
            .collect();
        let class_symbols_iter = self
            .conjugacy_class_transversal
            .as_ref()
            .expect("A conjugacy class transversal has not been found.")
            .iter()
            .enumerate()
            .map(|(i, &rep_ele_index)| {
                let (rep_ele, _) = self
                    .elements
                    .expect("No elements found for this group.")
                    .get_index(rep_ele_index)
                    .unwrap_or_else(|| {
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
        self.conjugacy_class_symbols = Some(class_symbols_iter.collect::<IndexMap<_, _>>());
        log::debug!("Assigning generic class symbols... Done.");
        self
    }

    fn inverse_conjugacy_classes(&mut self) -> &mut Self {
        log::debug!("Finding inverse conjugacy classes...");
        let mut iccs: Vec<_> = self
            .conjugacy_classes
            .as_ref()
            .expect("Conjugacy classes have not been found.")
            .iter()
            .map(|_| 0usize)
            .collect();
        let class_number = self
            .conjugacy_classes
            .as_ref()
            .expect("Conjugacy classes have not been found.")
            .len();
        let mut remaining_classes: HashSet<_> = (1..class_number).collect();
        let ctb = self
            .cayley_table
            .expect("No Cayley table found for this group.");
        while !remaining_classes.is_empty() {
            let class_index = *remaining_classes
                .iter()
                .next()
                .expect("Unexpected empty `remaining_classes`.");
            remaining_classes.remove(&class_index);
            let g = *self
                .conjugacy_classes
                .as_ref()
                .expect("Conjugacy classes have not been found.")[class_index]
                .iter()
                .next()
                .expect("No conjugacy classes can be empty.");
            let g_inv = ctb
                .slice(s![g, ..])
                .iter()
                .position(|&x| x == 0)
                .unwrap_or_else(|| {
                    panic!("No identity element can be found in row `{g}` of Cayley table.")
                });
            let inv_class_index = self
                .element_to_conjugacy_classes
                .as_ref()
                .expect("Map from element to conjugacy class has not been found.")[g_inv];
            iccs[class_index] = inv_class_index;
            if remaining_classes.contains(&inv_class_index) {
                remaining_classes.remove(&inv_class_index);
                iccs[inv_class_index] = class_index;
            }
        }
        assert!(iccs.iter().skip(1).all(|&x| x > 0));
        self.inverse_conjugacy_classes = Some(iccs);
        log::debug!("Finding inverse conjugacy classes... Done.");
        self
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
    fn class_matrix(&mut self) -> &mut Self {
        let class_number = self
            .conjugacy_classes
            .as_ref()
            .expect("Conjugacy classes have not been found.")
            .len();
        let mut nmat = Array3::<usize>::zeros((class_number, class_number, class_number));
        for (r, class_r) in self
            .conjugacy_classes
            .as_ref()
            .expect("Conjugacy classes have not been found.")
            .iter()
            .enumerate()
        {
            let idx_r = class_r.iter().copied().collect::<Vec<_>>();
            for (s, class_s) in self
                .conjugacy_classes
                .as_ref()
                .expect("Conjugacy classes have not been found.")
                .iter()
                .enumerate()
            {
                let idx_s = class_s.iter().copied().collect::<Vec<_>>();
                let cayley_block_rs = self
                    .cayley_table
                    .expect("No Cayley table found for this group.")
                    .select(Axis(0), &idx_r)
                    .select(Axis(1), &idx_s)
                    .iter()
                    .copied()
                    .counts();

                for (t, class_t) in self
                    .conjugacy_classes
                    .as_ref()
                    .expect("Conjugacy classes have not been found.")
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
        self.class_matrix = Some(nmat);
        self
    }
}

impl<'a, T> ClassStructure<'a, T>
where
    T: Clone,
{
    #[must_use]
    fn class_number(&self) -> usize {
        self.conjugacy_classes.len()
    }
}

//pub trait ClassStructure
//where
//    Self::Element: Hash + Eq + Clone + fmt::Debug + Sync + Send,
//    for<'a, 'b> &'b Self::Element: Mul<&'a Self::Element, Output = Self::Element>,
//{
//    type Element;

//    #[must_use]
//    fn name(&self) -> &str;

//    #[must_use]
//    fn elements(&self) -> &IndexMap<Self::Element, usize>;

//    #[must_use]
//    fn cayley_table(&self) -> &Array2<usize>;

//    #[must_use]
//    fn conjugacy_classes(&self) -> &Vec<HashSet<usize>>;

//    #[must_use]
//    fn conjugacy_class_transversal(&self) -> &Vec<usize>;

//    #[must_use]
//    fn conjugacy_class_symbols(&self) -> &IndexMap<ClassSymbol<Self::Element>, usize>;

//    #[must_use]
//    fn inverse_conjugacy_classes(&self) -> &Vec<usize>;

//    #[must_use]
//    fn element_to_conjugacy_classes(&self) -> &Vec<usize>;

//    #[must_use]
//    fn class_matrix(&self) -> &Array3<usize>;

//    /// Find the conjugacy classes and their inverses for the group.
//    #[must_use]
//    fn compute_conjugacy_classes(&self) -> (Vec<HashSet<usize>>, Vec<usize>);

//    #[must_use]
//    fn class_number(&self) -> usize {
//        self.conjugacy_classes().len()
//    }

//    /// Assigns generic class symbols to the conjugacy classes.
//    #[must_use]
//    fn compute_class_symbols(&mut self) -> IndexMap<ClassSymbol<Self::Element>, usize> {
//        log::debug!("Assigning generic class symbols...");
//        let class_sizes: Vec<_> = self
//            .conjugacy_classes()
//            .iter()
//            .map(HashSet::len)
//            .collect();
//        let class_symbols_iter = self
//            .conjugacy_class_transversal()
//            .iter()
//            .enumerate()
//            .map(|(i, &rep_ele_index)| {
//                let (rep_ele, _) = self.elements().get_index(rep_ele_index).unwrap_or_else(|| {
//                    panic!("Element with index {rep_ele_index} cannot be retrieved.")
//                });
//                (
//                    ClassSymbol::new(
//                        format!("{}||K{i}||", class_sizes[i]).as_str(),
//                        Some(rep_ele.clone()),
//                    )
//                    .unwrap_or_else(|_| {
//                        panic!(
//                            "Unable to construct a class symbol from `{}||K{i}||`.",
//                            class_sizes[i]
//                        )
//                    }),
//                    i,
//                )
//            });
//        log::debug!("Assigning generic class symbols... Done.");
//        class_symbols_iter.collect::<IndexMap<_, _>>()
//    }

//    #[must_use]
//    fn compute_conjugacy_class_transversal(&self) -> Vec<usize> {
//        self.conjugacy_classes()
//            .iter()
//            .map(|cc| {
//                *cc.iter()
//                    .next()
//                    .expect("No conjugacy classes can be empty.")
//            })
//            .collect()
//    }

//    #[must_use]
//    fn compute_inverse_conjugacy_classes(&self) -> Vec<usize> {
//        log::debug!("Finding inverse conjugacy classes...");
//        let mut iccs: Vec<_> = self
//            .conjugacy_classes()
//            .iter()
//            .map(|_| 0usize)
//            .collect();
//        let mut remaining_classes: HashSet<_> =
//            (1..self.class_number()).collect();
//        let ctb = self.cayley_table();
//        while !remaining_classes.is_empty() {
//            let class_index = *remaining_classes
//                .iter()
//                .next()
//                .expect("Unexpected empty `remaining_classes`.");
//            remaining_classes.remove(&class_index);
//            let g = *self
//                .conjugacy_classes()[class_index]
//                .iter()
//                .next()
//                .expect("No conjugacy classes can be empty.");
//            let g_inv = ctb
//                .slice(s![g, ..])
//                .iter()
//                .position(|&x| x == 0)
//                .unwrap_or_else(|| {
//                    panic!("No identity element can be found in row `{g}` of Cayley table.")
//                });
//            let inv_class_index = self
//                .element_to_conjugacy_classes()[g_inv];
//            iccs[class_index] = inv_class_index;
//            if remaining_classes.contains(&inv_class_index) {
//                remaining_classes.remove(&inv_class_index);
//                iccs[inv_class_index] = class_index;
//            }
//        }
//        assert!(iccs.iter().skip(1).all(|&x| x > 0));
//        log::debug!("Finding inverse conjugacy classes... Done.");
//        iccs
//    }

//    /// Calculates the class matrix $`\mathbf{N}`$ for the conjugacy classes in
//    /// the group.
//    ///
//    /// Let $`K_i`$ be the $`i^{\textrm{th}}`conjugacy class of the group. The
//    /// elements of the class matrix $`\mathbf{N}`$ are given by
//    ///
//    /// ```math
//    ///     N_{rst} = \lvert \{ (x, y) \in K_r \times K_s : xy = z \in K_t \} \rvert,
//    /// ```
//    ///
//    /// independent of any $`z \in K_t`$.
//    ///
//    /// This method sets the [`Self::class_matrix`] field.
//    #[must_use]
//    fn compute_class_matrix(&mut self) -> Array3<usize> {
//        let mut nmat = Array3::<usize>::zeros((
//            self.class_number(),
//            self.class_number(),
//            self.class_number(),
//        ));
//        for (r, class_r) in self
//            .conjugacy_classes()
//            .iter()
//            .enumerate()
//        {
//            let idx_r = class_r.iter().copied().collect::<Vec<_>>();
//            for (s, class_s) in self
//                .conjugacy_classes()
//                .iter()
//                .enumerate()
//            {
//                let idx_s = class_s.iter().copied().collect::<Vec<_>>();
//                let cayley_block_rs = self
//                    .cayley_table()
//                    .select(Axis(0), &idx_r)
//                    .select(Axis(1), &idx_s)
//                    .iter()
//                    .copied()
//                    .counts();

//                for (t, class_t) in self
//                    .conjugacy_classes()
//                    .iter()
//                    .enumerate()
//                {
//                    nmat[[r, s, t]] = *cayley_block_rs
//                        .get(
//                            class_t
//                                .iter()
//                                .next()
//                                .expect("No conjugacy classes can be empty."),
//                        )
//                        .unwrap_or(&0);
//                }
//            }
//        }
//        nmat
//    }
//}
