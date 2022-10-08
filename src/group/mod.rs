use std::collections::HashSet;
use std::fmt::Debug;
use std::hash::Hash;
use std::ops::Mul;

use derive_builder::Builder;
use indexmap::IndexMap;
use itertools::Itertools;
use log;
use ndarray::{s, Array2, Zip};
use rayon::iter::ParallelBridge;
use rayon::prelude::*;

use crate::symmetry::symmetry_core::Symmetry;
use crate::symmetry::symmetry_element::{SymmetryElement, SymmetryOperation, SIG};
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

    /// An optional name if this group is actually a finite subgroup of [`Self::name`].
    #[builder(default = "None")]
    finite_subgroup_name: Option<String>,

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
/// * infinite_order_to_finite - Interpret infinite-order generating
/// elements as finite-order generating elements to create a finite subgroup
/// of an otherwise infinite group.
///
/// # Returns
///
/// A finite abstract group struct.
fn group_from_molecular_symmetry(
    sym: Symmetry,
    infinite_order_to_finite: Option<u32>,
) -> Group<SymmetryOperation> {
    let group_name = sym.point_group.as_ref().unwrap().clone();

    let handles_infinite_group = if sym.is_infinite() {
        assert_ne!(infinite_order_to_finite, None);
        infinite_order_to_finite
    } else {
        None
    };

    if let Some(finite_order) = handles_infinite_group {
        if group_name == "O(3)" {
            assert!(
                matches!(finite_order, 2 | 4),
                "Finite order of {} is not yet supported for O(3).",
                finite_order
            );
        }
    }

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

    // Finite proper operations from generators
    let proper_operations_from_generators = if let Some(fin_ord) = handles_infinite_group {
        sym.proper_generators
            .par_iter()
            .fold(|| vec![], |mut acc, (order, proper_generators)| {
                proper_generators
                    .iter()
                    .for_each(|proper_generator| {
                        let finite_order = match order {
                            ElementOrder::Int(io) => *io,
                            ElementOrder::Inf => fin_ord
                        };
                        let finite_proper_element = SymmetryElement::builder()
                            .threshold(proper_generator.threshold)
                            .proper_order(ElementOrder::Int(finite_order))
                            .proper_power(1)
                            .axis(proper_generator.axis)
                            .kind(proper_generator.kind.clone())
                            .additional_superscript(
                                proper_generator.additional_superscript.clone(),
                            )
                            .additional_subscript(
                                proper_generator.additional_subscript.clone(),
                            )
                            .build()
                            .unwrap();
                        acc.extend((1..finite_order).map(|power| {
                            SymmetryOperation::builder()
                                .generating_element(finite_proper_element.clone())
                                .power(power as i32)
                                .build()
                                .unwrap()
                        }));
                    });
                acc
            })
            .reduce(|| vec![], |mut acc, vec| { acc.extend(vec); acc })
    } else {
        vec![]
    };

    // Finite improper operations
    let mut improper_orders = sym.improper_elements.keys().collect::<Vec<_>>();
    improper_orders.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let improper_operations = improper_orders
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

    // Finite improper operations from generators
    let improper_operations_from_generators = if let Some(fin_ord) = handles_infinite_group {
        sym.improper_generators
            .par_iter()
            .fold(|| vec![], |mut acc, (order, improper_generators)| {
                improper_generators
                    .iter()
                    .for_each(|improper_generator| {
                        let finite_order = match order {
                            ElementOrder::Int(io) => *io,
                            ElementOrder::Inf => fin_ord
                        };
                        let finite_improper_element = SymmetryElement::builder()
                            .threshold(improper_generator.threshold)
                            .proper_order(ElementOrder::Int(finite_order))
                            .proper_power(1)
                            .axis(improper_generator.axis)
                            .kind(improper_generator.kind.clone())
                            .additional_superscript(
                                improper_generator.additional_superscript.clone(),
                            )
                            .additional_subscript(
                                improper_generator.additional_subscript.clone(),
                            )
                            .build()
                            .unwrap();
                        acc.extend((1..(2 * finite_order)).step_by(2).map(|power| {
                            SymmetryOperation::builder()
                                .generating_element(finite_improper_element.clone())
                                .power(power as i32)
                                .build()
                                .unwrap()
                        }));
                    });
                acc
            })
            .reduce(|| vec![], |mut acc, vec| { acc.extend(vec); acc })
    } else {
        vec![]
    };

    let operations: HashSet<_> = if let None = handles_infinite_group {
        proper_operations
            .into_iter()
            .chain(proper_operations_from_generators)
            .chain(improper_operations)
            .chain(improper_operations_from_generators)
            .collect()
    } else {
        // Fulfil group closure
        log::debug!("Fulfilling closure for a finite subgroup of an infinite group...");
        let mut existing_operations: HashSet<_> = proper_operations
            .into_iter()
            .chain(proper_operations_from_generators)
            .chain(improper_operations)
            .chain(improper_operations_from_generators)
            .collect();
        let mut extra_operations = HashSet::<SymmetryOperation>::new();
        let mut npasses = 0;
        let mut nstable = 0;

        while nstable < 2 || npasses == 0 {
            let n_extra_operations = extra_operations.len();
            existing_operations.extend(extra_operations);

            npasses += 1;
            log::debug!(
                "Generating all group elements: {} pass{}, {} element{} (of which {} {} new)",
                npasses,
                {
                    if npasses > 1 {
                        "es"
                    } else {
                        ""
                    }
                }
                .to_string(),
                existing_operations.len(),
                {
                    if existing_operations.len() > 1 {
                        "s"
                    } else {
                        ""
                    }
                }
                .to_string(),
                n_extra_operations,
                {
                    if n_extra_operations > 1 {
                        "are"
                    } else {
                        "is"
                    }
                }
                .to_string(),
            );

            extra_operations = existing_operations
                .iter()
                .combinations_with_replacement(2)
                .par_bridge()
                .filter_map(|op_pairs| {
                    let op_i_ref = op_pairs[0];
                    let op_j_ref = op_pairs[1];
                    let op_k = op_i_ref * op_j_ref;
                    if !existing_operations.contains(&op_k) {
                        if !op_k.is_proper() {
                            Some(op_k.convert_to_improper_kind(&SIG))
                        } else {
                            Some(op_k)
                        }
                    } else {
                        None
                    }
                })
                .collect();
            if extra_operations.len() == 0 {
                nstable += 1;
            } else {
                nstable = 0;
            }
        }

        assert_eq!(extra_operations.len(), 0);
        log::debug!(
            "Group closure reached with {} elements.",
            existing_operations.len()
        );
        existing_operations
    };

    let mut sorted_operations: Vec<SymmetryOperation> = operations.into_iter().collect();
    sorted_operations.sort_by_key(|op| {
        (
            !op.is_proper(),
            !(op.is_identity() || op.is_inversion()),
            op.is_binary_rotation() || op.is_reflection(),
            -(*op.total_proper_fraction.unwrap().denom().unwrap() as i64),
            op.power,
        )
    });

    let mut group = Group::<SymmetryOperation>::new(group_name.as_str(), sorted_operations);
    if let Some(_) = handles_infinite_group {
        let finite_group = if group.name.contains("∞") {
            // # C∞, C∞h, C∞v, S∞, D∞, D∞h, D∞d
            if group.name.as_bytes()[0] == b'D' {
                if matches!(group.name.as_bytes().iter().last().unwrap(), b'h' | b'd') {
                    assert_eq!(group.order % 4, 0);
                    group.name.replace("∞", format!("{}", group.order / 4).as_str())
                } else {
                    assert_eq!(group.order % 2, 0);
                    group.name.replace("∞", format!("{}", group.order / 2).as_str())
                }
            } else {
                assert!(matches!(group.name.as_bytes()[0], b'C' | b'S'));
                if matches!(group.name.as_bytes().iter().last().unwrap(), b'h' | b'v') {
                    assert_eq!(group.order % 2, 0);
                    if group.order > 2 {
                        group.name.replace("∞", format!("{}", group.order / 2).as_str())
                    } else {
                        assert_eq!(group.name.as_bytes()[0], b'C');
                        "Cs".to_string()
                    }
                } else {
                    group.name.replace("∞", format!("{}", group.order).as_str())
                }
            }
        } else {
            // O(3)
            match group.order {
                8 => "D2h".to_string(),
                48 => "Oh".to_string(),
                _ => panic!("Unsupported number of group elements.")
            }
        };
        group.finite_subgroup_name = Some(finite_group);
    }
    group
}
