use std::collections::{HashMap, HashSet};
use std::fmt;
use std::hash::Hash;

use indexmap::{IndexMap, IndexSet};
use itertools::Itertools;
use ordered_float::OrderedFloat;
use rayon::prelude::*;

use super::{ClassAnalysed, Group, GroupType, UnitaryGroup};
use crate::symmetry::symmetry_core::Symmetry;
use crate::symmetry::symmetry_element::symmetry_operation::FiniteOrder;
use crate::symmetry::symmetry_element::symmetry_operation::SpecialSymmetryTransformation;
use crate::symmetry::symmetry_element::{
    SymmetryElement, SymmetryOperation, ROT, SIG, TRROT, TRSIG,
};
use crate::symmetry::symmetry_element_order::{ElementOrder, ORDER_1};
use crate::symmetry::symmetry_symbols::{deduce_sigma_symbol, ClassSymbol};

impl<T> Group<T>
where
    T: Hash + Eq + Clone + Sync + fmt::Debug + FiniteOrder + SpecialSymmetryTransformation,
{
    /// Returns `true` if all elements in this group are unitary.
    fn all_unitary(&self) -> bool {
        self.elements.keys().all(|op| !op.is_antiunitary())
    }

    fn group_type(&self) -> GroupType {
        if self.all_unitary() {
            GroupType::Ordinary(false)
        } else if self
            .elements
            .keys()
            .any(SpecialSymmetryTransformation::is_time_reversal)
        {
            GroupType::MagneticGrey(false)
        } else {
            GroupType::MagneticBlackWhite(false)
        }
    }
}

trait SymmetryGroup: ClassAnalysed<Element = SymmetryOperation> {
    #[must_use]
    fn compute_class_symbols_from_symmetry(
        &self,
        elements: &IndexMap<Self::Element, usize>,
    ) -> IndexMap<ClassSymbol<Self::Element>, usize> {
        log::debug!("Assigning class symbols from symmetry operations...");
        let mut proper_class_orders: HashMap<(ElementOrder, Option<u32>, i32, String), usize> =
            HashMap::new();
        let mut improper_class_orders: HashMap<(ElementOrder, Option<u32>, i32, String), usize> =
            HashMap::new();
        let mut tr_proper_class_orders: HashMap<(ElementOrder, Option<u32>, i32, String), usize> =
            HashMap::new();
        let mut tr_improper_class_orders: HashMap<(ElementOrder, Option<u32>, i32, String), usize> =
            HashMap::new();
        let class_symbols_iter =
            self.conjugacy_classes()
                .iter()
                .enumerate()
                .map(|(i, class_element_indices)| {
                    let rep_ele_index = *class_element_indices
                        .iter()
                        .min_by_key(|&&j| {
                            let op = elements
                                .get_index(j)
                                .unwrap_or_else(|| {
                                    panic!("Element with index {j} cannot be retrieved.")
                                })
                                .0;
                            (op.power, op.generating_element.proper_power)
                        })
                        .expect("Unable to obtain a representative element index.");
                    let (rep_ele, _) = elements.get_index(rep_ele_index).unwrap_or_else(|| {
                        panic!("Unable to retrieve group element with index `{rep_ele_index}`.")
                    });
                    if rep_ele.is_identity() {
                        (
                            ClassSymbol::new("1||E||", Some(rep_ele.clone()))
                                .expect("Unable to construct a class symbol from `1||E||`."),
                            i,
                        )
                    } else if rep_ele.is_inversion() {
                        (
                            ClassSymbol::new("1||i||", Some(rep_ele.clone()))
                                .expect("Unable to construct a class symbol from `1||i||`."),
                            i,
                        )
                    } else if rep_ele.is_time_reversal() {
                        (
                            ClassSymbol::new("1||θ||", Some(rep_ele.clone()))
                                .expect("Unable to construct a class symbol from `1||θ||`."),
                            i,
                        )
                    } else {
                        let rep_proper_order = rep_ele.generating_element.proper_order;
                        let rep_proper_power = rep_ele.generating_element.proper_power;
                        let rep_power = rep_ele.power;
                        let rep_sub = rep_ele.generating_element.additional_subscript.clone();
                        let class_orders = match (rep_ele.is_antiunitary(), rep_ele.is_proper()) {
                            (false, true) => &mut proper_class_orders,
                            (false, false) => &mut improper_class_orders,
                            (true, true) => &mut tr_proper_class_orders,
                            (true, false) => &mut tr_improper_class_orders,
                        };
                        let dash = if let Some(v) = class_orders.get_mut(&(
                            rep_proper_order,
                            rep_proper_power,
                            rep_power,
                            rep_sub.clone(),
                        )) {
                            *v += 1;
                            "'".repeat(*v)
                        } else {
                            class_orders.insert(
                                (rep_proper_order, rep_proper_power, rep_power, rep_sub),
                                0,
                            );
                            String::new()
                        };
                        let size = class_element_indices.len();
                        (
                            ClassSymbol::new(
                                format!(
                                    "{}||{}|^({})|",
                                    size,
                                    rep_ele.get_abbreviated_symbol(),
                                    dash
                                )
                                .as_str(),
                                Some(rep_ele.clone()),
                            )
                            .unwrap_or_else(|_| {
                                panic!(
                                "Unable to construct a class symbol from `{size}||{}|^({dash})|`",
                                rep_ele.get_abbreviated_symbol()
                            )
                            }),
                            i,
                        )
                    }
                });
        log::debug!("Assigning class symbols from symmetry operations... Done.");
        class_symbols_iter.collect::<IndexMap<_, _>>()
    }


}

impl SymmetryGroup for UnitaryGroup<SymmetryOperation> {}

/// Constructs a group from molecular symmetry *elements* (not operations).
///
/// # Arguments
///
/// * `sym` - A molecular symmetry struct.
/// * `infinite_order_to_finite` - Interpret infinite-order generating
/// elements as finite-order generating elements to create a finite subgroup
/// of an otherwise infinite group.
///
/// # Returns
///
/// A finite abstract group struct.
#[allow(clippy::too_many_lines)]
pub fn group_from_molecular_symmetry(
    sym: &Symmetry,
    infinite_order_to_finite: Option<u32>,
) -> impl ClassAnalysed<Element = SymmetryOperation> {
    let group_name = sym
        .group_name
        .as_ref()
        .expect("No point groups found.")
        .clone();

    let handles_infinite_group = if sym.is_infinite() {
        assert_ne!(infinite_order_to_finite, None);
        infinite_order_to_finite
    } else {
        None
    };

    if let Some(finite_order) = handles_infinite_group {
        if group_name.contains("O(3)") {
            if !matches!(finite_order, 2 | 4) {
                log::error!(
                    "Finite order of {} is not yet supported for {}.",
                    finite_order,
                    group_name
                );
            }
            assert!(
                matches!(finite_order, 2 | 4),
                "Finite order of {} is not yet supported for {}.",
                finite_order,
                group_name
            );
        }
    }

    let id_element = sym
        .get_elements(&ROT)
        .unwrap_or(&HashMap::new())
        .get(&ORDER_1)
        .expect("No identity elements found.")
        .iter()
        .next()
        .expect("No identity elements found.")
        .clone();

    let id_operation = SymmetryOperation::builder()
        .generating_element(id_element)
        .power(1)
        .build()
        .expect("Unable to construct an identity operation.");

    let empty_elements: HashMap<ElementOrder, IndexSet<SymmetryElement>> = HashMap::new();

    // Finite proper operations
    let mut proper_orders = sym
        .get_elements(&ROT)
        .unwrap_or(&empty_elements)
        .keys()
        .collect::<Vec<_>>();
    proper_orders.sort_by(|a, b| {
        a.partial_cmp(b)
            .unwrap_or_else(|| panic!("`{a}` and `{b}` cannot be compared."))
    });
    let proper_operations =
        proper_orders
            .iter()
            .fold(vec![id_operation], |mut acc, proper_order| {
                sym.get_elements(&ROT)
                    .unwrap_or(&empty_elements)
                    .get(proper_order)
                    .unwrap_or_else(|| panic!("Proper elements C{proper_order} not found."))
                    .iter()
                    .for_each(|proper_element| {
                        if let ElementOrder::Int(io) = proper_order {
                            acc.extend((1..*io).map(|power| {
                                SymmetryOperation::builder()
                                    .generating_element(proper_element.clone())
                                    .power(power.try_into().unwrap_or_else(|_| {
                                        panic!("Unable to convert `{power}` to `i32`.")
                                    }))
                                    .build()
                                    .expect("Unable to construct a symmetry operation.")
                            }));
                        }
                    });
                acc
            });

    // Finite proper operations from generators
    let proper_operations_from_generators = if let Some(fin_ord) = handles_infinite_group {
        sym.get_generators(&ROT)
            .unwrap_or(&empty_elements)
            .par_iter()
            .fold(std::vec::Vec::new, |mut acc, (order, proper_generators)| {
                for proper_generator in proper_generators.iter() {
                    let finite_order = match order {
                        ElementOrder::Int(io) => *io,
                        ElementOrder::Inf => fin_ord,
                    };
                    let finite_proper_element = SymmetryElement::builder()
                        .threshold(proper_generator.threshold)
                        .proper_order(ElementOrder::Int(finite_order))
                        .proper_power(1)
                        .axis(proper_generator.axis)
                        .kind(proper_generator.kind.clone())
                        .additional_superscript(proper_generator.additional_superscript.clone())
                        .additional_subscript(proper_generator.additional_subscript.clone())
                        .build()
                        .expect("Unable to construct a symmetry element.");
                    acc.extend((1..finite_order).map(|power| {
                        SymmetryOperation::builder()
                            .generating_element(finite_proper_element.clone())
                            .power(power.try_into().unwrap_or_else(|_| {
                                panic!("Unable to convert `{power}` to `i32`.")
                            }))
                            .build()
                            .expect("Unable to construct a symmetry operation.")
                    }));
                }
                acc
            })
            .reduce(std::vec::Vec::new, |mut acc, vec| {
                acc.extend(vec);
                acc
            })
    } else {
        vec![]
    };

    // Finite time-reversed proper operations
    let mut tr_proper_orders = sym
        .get_elements(&TRROT)
        .unwrap_or(&empty_elements)
        .keys()
        .collect::<Vec<_>>();
    tr_proper_orders.sort_by(|a, b| {
        a.partial_cmp(b)
            .unwrap_or_else(|| panic!("`{a}` and `{b}` cannot be compared."))
    });
    let tr_proper_operations = tr_proper_orders
        .iter()
        .fold(vec![], |mut acc, tr_proper_order| {
            sym.get_elements(&TRROT)
                .unwrap_or(&empty_elements)
                .get(tr_proper_order)
                .unwrap_or_else(|| panic!("Proper elements θ·C{tr_proper_order} not found."))
                .iter()
                .for_each(|tr_proper_element| {
                    if let ElementOrder::Int(io) = tr_proper_order {
                        acc.extend((1..(2 * *io)).step_by(2).map(|power| {
                            SymmetryOperation::builder()
                                .generating_element(tr_proper_element.clone())
                                .power(power.try_into().unwrap_or_else(|_| {
                                    panic!("Unable to convert `{power}` to `i32`.")
                                }))
                                .build()
                                .expect("Unable to construct a symmetry operation.")
                        }));
                    }
                });
            acc
        });

    // Finite time-reversed proper operations from generators
    let tr_proper_operations_from_generators = if let Some(fin_ord) = handles_infinite_group {
        sym.get_generators(&TRROT)
            .unwrap_or(&empty_elements)
            .par_iter()
            .fold(
                std::vec::Vec::new,
                |mut acc, (order, tr_proper_generators)| {
                    for tr_proper_generator in tr_proper_generators.iter() {
                        let finite_order = match order {
                            ElementOrder::Int(io) => *io,
                            ElementOrder::Inf => fin_ord,
                        };
                        let finite_tr_proper_element = SymmetryElement::builder()
                            .threshold(tr_proper_generator.threshold)
                            .proper_order(ElementOrder::Int(finite_order))
                            .proper_power(1)
                            .axis(tr_proper_generator.axis)
                            .kind(tr_proper_generator.kind.clone())
                            .additional_superscript(
                                tr_proper_generator.additional_superscript.clone(),
                            )
                            .additional_subscript(tr_proper_generator.additional_subscript.clone())
                            .build()
                            .expect("Unable to construct a symmetry element.");
                        acc.extend((1..finite_order).map(|power| {
                            SymmetryOperation::builder()
                                .generating_element(finite_tr_proper_element.clone())
                                .power(power.try_into().unwrap_or_else(|_| {
                                    panic!("Unable to convert `{power}` to `i32`.")
                                }))
                                .build()
                                .expect("Unable to construct a symmetry operation.")
                        }));
                    }
                    acc
                },
            )
            .reduce(std::vec::Vec::new, |mut acc, vec| {
                acc.extend(vec);
                acc
            })
    } else {
        vec![]
    };

    // Finite improper operations
    let mut improper_orders = sym
        .get_elements(&SIG)
        .unwrap_or(&empty_elements)
        .keys()
        .collect::<Vec<_>>();
    improper_orders.sort_by(|a, b| {
        a.partial_cmp(b)
            .unwrap_or_else(|| panic!("`{a}` and `{b}` cannot be compared."))
    });
    let improper_operations = improper_orders
        .iter()
        .fold(vec![], |mut acc, improper_order| {
            sym.get_elements(&SIG)
                .unwrap_or(&empty_elements)
                .get(improper_order)
                .unwrap_or_else(|| panic!("Improper elements S{improper_order} not found."))
                .iter()
                .for_each(|improper_element| {
                    if let ElementOrder::Int(io) = improper_order {
                        acc.extend((1..(2 * *io)).step_by(2).map(|power| {
                            SymmetryOperation::builder()
                                .generating_element(improper_element.clone())
                                .power(power.try_into().unwrap_or_else(|_| {
                                    panic!("Unable to convert `{power}` to `i32`.")
                                }))
                                .build()
                                .expect("Unable to construct a symmetry operation.")
                        }));
                    }
                });
            acc
        });

    // Finite improper operations from generators
    let improper_operations_from_generators = if let Some(fin_ord) = handles_infinite_group {
        sym.get_generators(&SIG)
            .unwrap_or(&empty_elements)
            .par_iter()
            .fold(
                std::vec::Vec::new,
                |mut acc, (order, improper_generators)| {
                    for improper_generator in improper_generators.iter() {
                        let finite_order = match order {
                            ElementOrder::Int(io) => *io,
                            ElementOrder::Inf => fin_ord,
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
                            .additional_subscript(improper_generator.additional_subscript.clone())
                            .build()
                            .expect("Unable to construct a symmetry element.");
                        acc.extend((1..(2 * finite_order)).step_by(2).map(|power| {
                            SymmetryOperation::builder()
                                .generating_element(finite_improper_element.clone())
                                .power(power.try_into().unwrap_or_else(|_| {
                                    panic!("Unable to convert `{power}` to `i32`.")
                                }))
                                .build()
                                .expect("Unable to construct a symmetry operation.")
                        }));
                    }
                    acc
                },
            )
            .reduce(std::vec::Vec::new, |mut acc, vec| {
                acc.extend(vec);
                acc
            })
    } else {
        vec![]
    };

    // Finite time-reversed improper operations
    let mut tr_improper_orders = sym
        .get_elements(&TRSIG)
        .unwrap_or(&empty_elements)
        .keys()
        .collect::<Vec<_>>();
    tr_improper_orders.sort_by(|a, b| {
        a.partial_cmp(b)
            .unwrap_or_else(|| panic!("`{a}` and `{b}` cannot be compared."))
    });
    let tr_improper_operations =
        tr_improper_orders
            .iter()
            .fold(vec![], |mut acc, tr_improper_order| {
                sym.get_elements(&TRSIG)
                    .unwrap_or(&empty_elements)
                    .get(tr_improper_order)
                    .unwrap_or_else(|| {
                        panic!("Improper elements θ·S{tr_improper_order} not found.")
                    })
                    .iter()
                    .for_each(|tr_improper_element| {
                        if let ElementOrder::Int(io) = tr_improper_order {
                            acc.extend((1..(2 * *io)).step_by(2).map(|power| {
                                SymmetryOperation::builder()
                                    .generating_element(tr_improper_element.clone())
                                    .power(power.try_into().unwrap_or_else(|_| {
                                        panic!("Unable to convert `{power}` to `i32`.")
                                    }))
                                    .build()
                                    .expect("Unable to construct a symmetry operation.")
                            }));
                        }
                    });
                acc
            });

    // Finite time-reversed improper operations from generators
    let tr_improper_operations_from_generators = if let Some(fin_ord) = handles_infinite_group {
        sym.get_generators(&TRSIG)
            .unwrap_or(&empty_elements)
            .par_iter()
            .fold(
                std::vec::Vec::new,
                |mut acc, (order, tr_improper_generators)| {
                    for tr_improper_generator in tr_improper_generators.iter() {
                        let finite_order = match order {
                            ElementOrder::Int(io) => *io,
                            ElementOrder::Inf => fin_ord,
                        };
                        let finite_tr_improper_element = SymmetryElement::builder()
                            .threshold(tr_improper_generator.threshold)
                            .proper_order(ElementOrder::Int(finite_order))
                            .proper_power(1)
                            .axis(tr_improper_generator.axis)
                            .kind(tr_improper_generator.kind.clone())
                            .additional_superscript(
                                tr_improper_generator.additional_superscript.clone(),
                            )
                            .additional_subscript(
                                tr_improper_generator.additional_subscript.clone(),
                            )
                            .build()
                            .expect("Unable to construct a symmetry element.");
                        acc.extend((1..(2 * finite_order)).step_by(2).map(|power| {
                            SymmetryOperation::builder()
                                .generating_element(finite_tr_improper_element.clone())
                                .power(power.try_into().unwrap_or_else(|_| {
                                    panic!("Unable to convert `{power}` to `i32`.")
                                }))
                                .build()
                                .expect("Unable to construct a symmetry operation.")
                        }));
                    }
                    acc
                },
            )
            .reduce(std::vec::Vec::new, |mut acc, vec| {
                acc.extend(vec);
                acc
            })
    } else {
        vec![]
    };

    let operations: IndexSet<_> = if handles_infinite_group.is_none() {
        proper_operations
            .into_iter()
            .chain(proper_operations_from_generators)
            .chain(improper_operations)
            .chain(improper_operations_from_generators)
            .chain(tr_proper_operations)
            .chain(tr_proper_operations_from_generators)
            .chain(tr_improper_operations)
            .chain(tr_improper_operations_from_generators)
            .collect()
    } else {
        // Fulfil group closure
        log::debug!("Fulfilling closure for a finite subgroup of an infinite group...");
        let mut existing_operations: IndexSet<_> = proper_operations
            .into_iter()
            .chain(proper_operations_from_generators)
            .chain(improper_operations)
            .chain(improper_operations_from_generators)
            .chain(tr_proper_operations)
            .chain(tr_proper_operations_from_generators)
            .chain(tr_improper_operations)
            .chain(tr_improper_operations_from_generators)
            .collect();
        let mut extra_operations = HashSet::<SymmetryOperation>::new();
        let mut npasses = 0;
        let mut nstable = 0;

        let principal_element = sym.get_proper_principal_element();
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
                    if n_extra_operations == 1 {
                        "is"
                    } else {
                        "are"
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
                    if existing_operations.contains(&op_k) {
                        None
                    } else if op_k.is_proper() {
                        Some(op_k)
                    } else if (op_k.is_reflection() || op_k.is_tr_reflection())
                        && op_k.generating_element.additional_subscript.is_empty()
                    {
                        if let Some(sigma_symbol) = deduce_sigma_symbol(
                            &op_k.generating_element.axis,
                            principal_element,
                            op_k.generating_element.threshold,
                            false,
                        ) {
                            let mut op_k_sym = op_k.convert_to_improper_kind(&SIG);
                            op_k_sym.generating_element.additional_subscript = sigma_symbol;
                            Some(op_k_sym)
                        } else {
                            Some(op_k.convert_to_improper_kind(&SIG))
                        }
                    } else {
                        Some(op_k.convert_to_improper_kind(&SIG))
                    }
                })
                .collect();
            if extra_operations.is_empty() {
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
        let (axis_closeness, closest_axis) = op.generating_element.closeness_to_cartesian_axes();
        (
            op.is_antiunitary(),
            !op.is_proper(),
            !(op.is_identity()
                || op.is_inversion()
                || op.is_time_reversal()
                || op.is_tr_inversion()),
            op.is_binary_rotation()
                || op.is_tr_binary_rotation()
                || op.is_reflection()
                || op.is_tr_reflection(),
            -(i64::try_from(
                *op.total_proper_fraction
                    .expect("No total proper fractions found.")
                    .denom()
                    .expect("The denominator of the total proper fraction cannot be extracted."),
            )
            .unwrap_or_else(|_| {
                panic!(
                    "Unable to convert the denominator of `{:?}` to `i64`.",
                    op.total_proper_fraction
                )
            })),
            op.power,
            OrderedFloat(axis_closeness),
            closest_axis,
        )
    });

    let mut group = UnitaryGroup::<SymmetryOperation>::new(group_name.as_str(), sorted_operations);
    if handles_infinite_group.is_some() {
        let finite_group = if group.name.contains('∞') {
            // C∞, C∞h, C∞v, S∞, D∞, D∞h, D∞d, or the corresponding grey groups
            if group.name.as_bytes()[0] == b'D' {
                if matches!(
                    group
                        .name
                        .as_bytes()
                        .iter()
                        .last()
                        .expect("The last character in the group name cannot be retrieved."),
                    b'h' | b'd'
                ) {
                    if group.name.contains('θ') {
                        assert_eq!(group.abstract_group.order() % 8, 0);
                        group.name.replace(
                            '∞',
                            format!("{}", group.abstract_group.order() / 8).as_str(),
                        )
                    } else {
                        assert_eq!(group.abstract_group.order() % 4, 0);
                        group.name.replace(
                            '∞',
                            format!("{}", group.abstract_group.order() / 4).as_str(),
                        )
                    }
                } else {
                    if group.name.contains('θ') {
                        assert_eq!(group.abstract_group.order() % 4, 0);
                        group.name.replace(
                            '∞',
                            format!("{}", group.abstract_group.order() / 4).as_str(),
                        )
                    } else {
                        assert_eq!(group.abstract_group.order() % 2, 0);
                        group.name.replace(
                            '∞',
                            format!("{}", group.abstract_group.order() / 2).as_str(),
                        )
                    }
                }
            } else {
                assert!(matches!(group.name.as_bytes()[0], b'C' | b'S'));
                if matches!(
                    group
                        .name
                        .as_bytes()
                        .iter()
                        .last()
                        .expect("The last character in the group name cannot be retrieved."),
                    b'h' | b'v'
                ) {
                    if group.name.contains('θ') {
                        assert_eq!(group.abstract_group.order() % 4, 0);
                    } else {
                        assert_eq!(group.abstract_group.order() % 2, 0);
                    }
                    if group.abstract_group.order() > 2 {
                        if group.name.contains('θ') {
                            group.name.replace(
                                '∞',
                                format!("{}", group.abstract_group.order() / 4).as_str(),
                            )
                        } else {
                            group.name.replace(
                                '∞',
                                format!("{}", group.abstract_group.order() / 2).as_str(),
                            )
                        }
                    } else {
                        assert_eq!(group.name.as_bytes()[0], b'C');
                        "Cs".to_string()
                    }
                } else {
                    group
                        .name
                        .replace('∞', format!("{}", group.abstract_group.order()).as_str())
                }
            }
        } else {
            // O(3) or the corresponding grey group
            match group.abstract_group.order() {
                8 => "D2h".to_string(),
                16 => "D2h + θ·D2h".to_string(),
                48 => "Oh".to_string(),
                96 => "Oh + θ·Oh".to_string(),
                _ => panic!("Unsupported number of group elements."),
            }
        };
        group.finite_subgroup_name = Some(finite_group);
    }
    group
        .class_structure
        .as_mut()
        .unwrap()
        .conjugacy_class_symbols =
        group.compute_class_symbols_from_symmetry(&group.abstract_group.elements);

    // group.construct_irrep_character_table();
    // if !group.is_unitary() {
    //     let unitary_elements = group
    //         .elements
    //         .iter()
    //         .filter_map(|(op, _)| {
    //             if !op.is_antiunitary() {
    //                 Some(op.clone())
    //             } else {
    //                 None
    //             }
    //         })
    //         .collect_vec();
    //     // let mut group = Group::<SymmetryOperation>::new(group_name.as_str(), sorted_operations);
    //     let mut unitary_subgroup = UnitaryGroup::<SymmetryOperation>::new(
    //         format!("U({})", group.name).as_str(),
    //         unitary_elements,
    //     );
    //     unitary_subgroup.finite_subgroup_name = group
    //         .finite_subgroup_name
    //         .as_ref()
    //         .map(|finite_group| format!("U({finite_group})"));
    //     unitary_subgroup.assign_class_symbols_from_symmetry();
    //     unitary_subgroup.construct_irrep_character_table();
    //     group.construct_ircorep_character_table(unitary_subgroup);
    // }
    group
}
