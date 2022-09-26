use std::collections::HashSet;
use std::fmt::Debug;
use std::hash::Hash;
use std::ops::Mul;

use log;
use indexmap::IndexMap;
use derive_builder::Builder;
use ndarray::{Array2, Zip, s};

#[cfg(test)]
#[path = "group_tests.rs"]
mod group_tests;

/// A struct for managing abstract groups.
#[derive(Builder)]
struct Group<T: Hash + Eq + Clone + Sync + Debug> {
    /// A name for the group.
    name: String,

    /// An ordered hash table containing the elements of the group.
    #[builder(setter(custom))]
    elements: IndexMap<T, usize>,

    /// The order of the group.
    #[builder(setter(skip), default = "self.elements.as_ref().unwrap().len()")]
    order: usize,

    /// The Cayley table for this group w.r.t. the elements in [`Self::elements`].
    ///
    /// Elements are multiplied left-to-right, with functions written on the right.
    /// Row elements are on the left, column elements on the right.  So column
    /// elements act on the function first, followed by row elements.
    ///
    /// Each element in this array contains the index of the resultant operation
    /// from the composition, w.r.t. the array [`Self::elements`].
    #[builder(setter(skip), default = "None")]
    cayley_table: Option<Array2<usize>>,

    /// A vector of conjugacy classes for this group.
    ///
    /// Each element in the vector is a hashset containing the indices of the
    /// elements in [`Self::elements`] for a particular conjugacy class. This
    /// thus defines a multi-valued map from each conjugacy class index to one
    /// or more element indices.
    #[builder(setter(skip), default = "None")]
    conjugacy_classes: Option<Vec<HashSet<usize>>>,

    /// The conjugacy class index of the elements in [`Self::elements`].
    ///
    /// This is the so-called inverse of [`Self::conjugacy_classes`]. This maps
    /// each element index to its corresponding conjugacy class index.
    #[builder(setter(skip), default = "None")]
    element_to_conjugacy_classes: Option<Vec<usize>>,
}

impl<T: Hash + Eq + Clone + Sync + Debug> GroupBuilder<T> {
    fn elements(&mut self, elems: Vec<T>) -> &mut Self {
        self.elements = Some(
            elems
                .into_iter()
                .enumerate()
                .map(|(i, element)| (element, i))
                .collect(),
        );
        self
    }
}

impl<T: Hash + Eq + Clone + Sync + Debug> Group<T>
where
    for<'a, 'b> &'b T: Mul<&'a T, Output = T>,
{
    /// Returns a builder to construct a new group.
    ///
    /// # Returns
    ///
    /// A builder to construct a new group.
    pub fn builder() -> GroupBuilder<T> {
        GroupBuilder::default()
    }

    fn new(name: &str, elements: Vec<T>) -> Self {
        let mut grp = Self::builder()
            .name(name.to_string())
            .elements(elements)
            .build()
            .unwrap();
        grp.construct_cayley_table();
        grp.find_conjugacy_classes();
        grp
    }

    fn construct_cayley_table(&mut self) {
        log::debug!("Constructing Cayley table in parallel...");
        let mut ctb = Array2::<usize>::zeros((self.order, self.order));
        Zip::indexed(&mut ctb).par_for_each(|(i, j), k| {
            let (op_i_ref, _) = self.elements.get_index(i).unwrap();
            let (op_j_ref, _) = self.elements.get_index(j).unwrap();
            let op_k = op_i_ref * op_j_ref;
            *k = *self
                .elements
                .get(&op_k)
                .expect(
                    format!(
                        "Group closure not fulfilled. The composition {:?} * {:?} = {:?} is not contained in the group.",
                        op_i_ref,
                        op_j_ref,
                        &op_k
                    ).as_str()
                );
        });
        self.cayley_table = Some(ctb);
        log::debug!("Constructing Cayley table in parallel... Done.");
    }

    fn is_abelian(&self) -> bool {
        let ctb = self.cayley_table.as_ref().unwrap();
        ctb == ctb.t()
    }

    fn find_conjugacy_classes(&mut self) {
        // Find conjugacy classes
        log::debug!("Finding conjugacy classes...");
        if self.is_abelian() {
            // Abelian group; each element is in its own conjugacy class.
            self.conjugacy_classes = Some(
                (0usize..self.order)
                    .map(|i| HashSet::from([i]))
                    .collect()
            );
            self.element_to_conjugacy_classes = Some(
                (0usize..self.order).collect()
            );
        } else {
            let mut ccs: Vec<HashSet<usize>> = vec![HashSet::from([0usize])];
            let mut e2ccs = vec![0usize; self.order];
            let mut remaining_elements: HashSet<usize> = (1usize..self.order).collect();
            let ctb = self.cayley_table.as_ref().unwrap();

            while remaining_elements.len() > 0 {
                // For a fixed g, find all h such that sg = hs for all s in the group.
                let g = *remaining_elements.iter().next().unwrap();
                let mut cur_cc = HashSet::from([g]);
                e2ccs[g] = ccs.len();
                for s in 0usize..self.order {
                    let sg = ctb[[s, g]];
                    let ctb_xs = ctb.slice(s![.., s]);
                    let h = ctb_xs.iter().position(|&x| x == sg).unwrap();
                    if remaining_elements.contains(&h) {
                        remaining_elements.remove(&h);
                        cur_cc.insert(h);
                        e2ccs[h] = ccs.len();
                    }
                }
                ccs.push(cur_cc);
            }
            assert!(e2ccs.iter().skip(1).all(|&x| x > 0));
            self.element_to_conjugacy_classes = Some(e2ccs);
        }
        log::debug!("Finding conjugacy classes... Done.");
    }
}
