use std::collections::HashSet;
use std::fmt::Debug;
use std::hash::Hash;
use std::ops::Mul;

use log;
use derive_builder::Builder;
use indexmap::IndexMap;
use ndarray::{s, Array2, Zip};

use crate::symmetry::symmetry_core::Symmetry;
use crate::symmetry::symmetry_element::SymmetryOperation;
use crate::symmetry::symmetry_element_order::{ElementOrder, ORDER_1};

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

    /// The number of conjugacy classes of this group.
    ///
    /// This is also the number of distinct irreducible representations of the
    /// group.
    #[builder(setter(skip), default = "None")]
    class_number: Option<usize>,
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

    /// Constructs a group from its elements.
    ///
    /// # Arguments
    ///
    /// * name - A name to be given to the group.
    /// * elements - A vector of *all* group elements.
    ///
    /// # Returns
    ///
    /// A group with its Cayley table constructed and conjugacy classes determined.
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

    /// Checks if this group is Abelian.
    ///
    /// This method requires the Cayley table to have been constructed.
    ///
    /// # Returns
    ///
    /// A flag indicating if this group is Abelian.
    fn is_abelian(&self) -> bool {
        let ctb = self.cayley_table.as_ref().unwrap();
        ctb == ctb.t()
    }

    /// Constructs the Cayley table for the group.
    ///
    /// This method sets the [`Self::cayley_table`] field.
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

    /// Find the conjugacy classes for the group.
    ///
    /// This method sets the [`Self::conjugacy_classes`],
    /// [`Self::element_to_conjugacy_classes`], and [`Self::class_number`] fields.
    fn find_conjugacy_classes(&mut self) {
        // Find conjugacy classes
        log::debug!("Finding conjugacy classes...");
        if self.is_abelian() {
            log::debug!("Abelian group found.");
            // Abelian group; each element is in its own conjugacy class.
            self.conjugacy_classes =
                Some((0usize..self.order).map(|i| HashSet::from([i])).collect());
            self.element_to_conjugacy_classes = Some((0usize..self.order).collect());
        } else {
            // Non-Abelian group.
            log::debug!("Non-Abelian group found.");
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
            self.conjugacy_classes = Some(ccs);
            assert!(e2ccs.iter().skip(1).all(|&x| x > 0));
            self.element_to_conjugacy_classes = Some(e2ccs);
        }
        self.class_number = Some(self.conjugacy_classes.as_ref().unwrap().len());
        log::debug!("Finding conjugacy classes... Done.");
    }
}

/// Constructs a group from molecular symmetry *elements* (not operations).
///
/// # Arguments
///
/// * sym - A molecular symmetry struct.
/// * name - An optional symbolic name to be given to the group. If no name
/// is given, the point-group name from `sym` will be used instead.
/// * infinite_order_to_finite - Interpret infinite-order generating
/// elements as finite-order generating elements to create a finite subgroup
/// of an otherwise infinite group.
///
/// # Returns
///
/// A finite abstract group struct.
fn group_from_molecular_symmetry(
    sym: Symmetry,
    name: Option<&str>,
    infinite_order_to_finite: Option<u64>,
) -> Group<SymmetryOperation> {
    let group_name = if let Some(nam) = name {
        nam.to_string()
    } else {
        sym.point_group.as_ref().unwrap().clone()
    };

    let handles_infinite_group = if sym.is_infinite() {
        assert_ne!(infinite_order_to_finite, None);
        true
    } else {
        false
    };

    let id_element = sym
        .proper_elements
        .get(&ORDER_1)
        .unwrap()
        .iter()
        .next()
        .unwrap()
        .clone();

    let id_operation = SymmetryOperation::builder()
        .generating_element(id_element)
        .power(1)
        .build()
        .unwrap();

    // Finite proper operations
    let mut proper_orders = sym.proper_elements.keys().collect::<Vec<_>>();
    proper_orders.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let proper_operations =
        proper_orders
            .iter()
            .fold(vec![id_operation], |mut acc, proper_order| {
                sym.proper_elements
                    .get(&proper_order)
                    .unwrap()
                    .iter()
                    .for_each(|proper_element| {
                        if let ElementOrder::Int(io) = proper_order {
                            acc.extend((1..*io).map(|power| {
                                SymmetryOperation::builder()
                                    .generating_element(proper_element.clone())
                                    .power(power as i32)
                                    .build()
                                    .unwrap()
                            }))
                        }
                    });
                acc
            });

    // Finite improper operations
    let mut improper_orders = sym.improper_elements.keys().collect::<Vec<_>>();
    improper_orders.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let improper_operations =
        improper_orders
            .iter()
            .fold(vec![], |mut acc, improper_order| {
                sym.improper_elements
                    .get(&improper_order)
                    .unwrap()
                    .iter()
                    .for_each(|improper_element| {
                        if let ElementOrder::Int(io) = improper_order {
                            acc.extend((1..(2 * *io)).step_by(2).map(|power| {
                                SymmetryOperation::builder()
                                    .generating_element(improper_element.clone())
                                    .power(power as i32)
                                    .build()
                                    .unwrap()
                            }))
                        }
                    });
                acc
            });

    let operations: HashSet<_> = proper_operations
        .into_iter()
        .chain(improper_operations)
        .collect();

    let mut sorted_operations: Vec<SymmetryOperation> = operations.into_iter().collect();
    sorted_operations.sort_by_key(
        |op| (
            !op.is_proper(),
            !(op.is_identity() || op.is_inversion()),
            op.is_binary_rotation() || op.is_reflection(),
            -(*op.total_proper_fraction.unwrap().denom().unwrap() as i64),
            op.power
        )
    );
    Group::<SymmetryOperation>::new(group_name.as_str(), sorted_operations)
}
