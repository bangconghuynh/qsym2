use std::collections::HashMap;

use indexmap::IndexMap;

use super::{GroupProperties, GroupType, MagneticRepresentedGroup, UnitaryRepresentedGroup};
use crate::chartab::CharacterTable;
use crate::group::class::ClassProperties;
use crate::group::construct_chartab::CharacterProperties;
use crate::symmetry::symmetry_core::Symmetry;
use crate::symmetry::symmetry_element::symmetry_operation::SpecialSymmetryTransformation;
use crate::symmetry::symmetry_element::SymmetryOperation;
use crate::symmetry::symmetry_element_order::ElementOrder;
use crate::symmetry::symmetry_symbols::{ClassSymbol, CollectionSymbol, MullikenIrrepSymbol};

pub trait SymmetryGroupProperties: ClassProperties<ClassElement = SymmetryOperation> {
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
    fn from_molecular_symmetry(sym: &Symmetry, infinite_order_to_finite: Option<u32>) -> Self;

    fn finite_group_name(&mut self) -> String {
        let finite_group = if self.name().contains('∞') {
            // C∞, C∞h, C∞v, S∞, D∞, D∞h, D∞d, or the corresponding grey groups
            if self.name().as_bytes()[0] == b'D' {
                if matches!(
                    self.name()
                        .as_bytes()
                        .iter()
                        .last()
                        .expect("The last character in the group name cannot be retrieved."),
                    b'h' | b'd'
                ) {
                    if self.name().contains('θ') {
                        assert_eq!(self.abstract_group().order() % 8, 0);
                        self.name().replace(
                            '∞',
                            format!("{}", self.abstract_group().order() / 8).as_str(),
                        )
                    } else {
                        assert_eq!(self.abstract_group().order() % 4, 0);
                        self.name().replace(
                            '∞',
                            format!("{}", self.abstract_group().order() / 4).as_str(),
                        )
                    }
                } else {
                    if self.name().contains('θ') {
                        assert_eq!(self.abstract_group().order() % 4, 0);
                        self.name().replace(
                            '∞',
                            format!("{}", self.abstract_group().order() / 4).as_str(),
                        )
                    } else {
                        assert_eq!(self.abstract_group().order() % 2, 0);
                        self.name().replace(
                            '∞',
                            format!("{}", self.abstract_group().order() / 2).as_str(),
                        )
                    }
                }
            } else {
                assert!(matches!(self.name().as_bytes()[0], b'C' | b'S'));
                if matches!(
                    self.name()
                        .as_bytes()
                        .iter()
                        .last()
                        .expect("The last character in the group name cannot be retrieved."),
                    b'h' | b'v'
                ) {
                    if self.name().contains('θ') {
                        assert_eq!(self.abstract_group().order() % 4, 0);
                    } else {
                        assert_eq!(self.abstract_group().order() % 2, 0);
                    }
                    if self.abstract_group().order() > 2 {
                        if self.name().contains('θ') {
                            self.name().replace(
                                '∞',
                                format!("{}", self.abstract_group().order() / 4).as_str(),
                            )
                        } else {
                            self.name().replace(
                                '∞',
                                format!("{}", self.abstract_group().order() / 2).as_str(),
                            )
                        }
                    } else {
                        assert_eq!(self.name().as_bytes()[0], b'C');
                        "Cs".to_string()
                    }
                } else {
                    self.name()
                        .replace('∞', format!("{}", self.abstract_group().order()).as_str())
                }
            }
        } else if self.name().contains("O(3)") {
            // O(3) or the corresponding grey group
            match self.abstract_group().order() {
                8 => "D2h".to_string(),
                16 => "D2h + θ·D2h".to_string(),
                48 => "Oh".to_string(),
                96 => "Oh + θ·Oh".to_string(),
                _ => panic!("Unsupported number of group elements for a finite group of O(3)."),
            }
        } else {
            // This is already a finite group.
            self.name().to_string()
        };
        finite_group
    }

    /// Returns `true` if all elements in this group are unitary.
    fn all_unitary(&self) -> bool {
        self.elements().keys().all(|op| !op.is_antiunitary())
    }

    fn group_type(&self) -> GroupType {
        if self.all_unitary() {
            GroupType::Ordinary(false)
        } else if self
            .elements()
            .keys()
            .any(SpecialSymmetryTransformation::is_time_reversal)
        {
            GroupType::MagneticGrey(false)
        } else {
            GroupType::MagneticBlackWhite(false)
        }
    }

    fn set_class_symbols_from_symmetry(&mut self) {
        log::debug!("Assigning class symbols from symmetry operations...");
        let mut proper_class_orders: HashMap<(ElementOrder, Option<u32>, i32, String), usize> =
            HashMap::new();
        let mut improper_class_orders: HashMap<(ElementOrder, Option<u32>, i32, String), usize> =
            HashMap::new();
        let mut tr_proper_class_orders: HashMap<(ElementOrder, Option<u32>, i32, String), usize> =
            HashMap::new();
        let mut tr_improper_class_orders: HashMap<(ElementOrder, Option<u32>, i32, String), usize> =
            HashMap::new();
        let symmetry_class_symbols = self
            .conjugacy_class_symbols()
            .iter()
            .map(|(old_symbol, &i)| {
                let rep_ele_index = *self.conjugacy_classes()[i]
                    .iter()
                    .min_by_key(|&&j| {
                        let op = self
                            .elements()
                            .get_index(j)
                            .unwrap_or_else(|| {
                                panic!("Element with index {j} cannot be retrieved.")
                            })
                            .0;
                        (op.power, op.generating_element.proper_power)
                    })
                    .expect("Unable to obtain a representative element index.");
                let (rep_ele, _) = self.elements().get_index(rep_ele_index).unwrap_or_else(|| {
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
                        class_orders
                            .insert((rep_proper_order, rep_proper_power, rep_power, rep_sub), 0);
                        String::new()
                    };
                    let size = old_symbol.size();
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
            })
            .collect::<IndexMap<_, _>>();
        self.class_structure_mut()
            .set_class_symbols(symmetry_class_symbols);
        log::debug!("Assigning class symbols from symmetry operations... Done.");
    }
}

impl SymmetryGroupProperties for UnitaryRepresentedGroup<SymmetryOperation> {
    /// Constructs a unitary group from molecular symmetry *elements* (not operations).
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
    fn from_molecular_symmetry(sym: &Symmetry, infinite_order_to_finite: Option<u32>) -> Self {
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

        let sorted_operations = sym.generate_all_operations(infinite_order_to_finite);

        let mut group = Self::new(group_name.as_str(), sorted_operations);
        if handles_infinite_group.is_some() {
            group.finite_subgroup_name = Some(group.finite_group_name());
        }
        group.set_class_symbols_from_symmetry();
        group.construct_character_table();
        group
    }
}

impl SymmetryGroupProperties
    for MagneticRepresentedGroup<
        SymmetryOperation,
        UnitaryRepresentedGroup<SymmetryOperation>,
        <UnitaryRepresentedGroup<SymmetryOperation> as CharacterProperties<
            MullikenIrrepSymbol,
            ClassSymbol<SymmetryOperation>,
        >>::CharTab,
    >
{
    /// Constructs a magnetic group from molecular symmetry *elements* (not operations).
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
    fn from_molecular_symmetry(sym: &Symmetry, infinite_order_to_finite: Option<u32>) -> Self {
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

        let sorted_operations = sym.generate_all_operations(infinite_order_to_finite);

        log::debug!("Constructing the unitary subgroup for the magnetic group...");
        let unitary_operations = sorted_operations
            .iter()
            .filter_map(|op| {
                if !op.is_antiunitary() {
                    Some(op.clone())
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        let mut unitary_subgroup = UnitaryRepresentedGroup::<SymmetryOperation>::new(
            group_name.as_str(),
            unitary_operations,
        );
        unitary_subgroup.set_class_symbols_from_symmetry();
        unitary_subgroup.construct_character_table();
        log::debug!("Constructing the unitary subgroup for the magnetic group... Done.");

        let mut group = Self::new(group_name.as_str(), sorted_operations, unitary_subgroup);
        if handles_infinite_group.is_some() {
            group.finite_subgroup_name = Some(group.finite_group_name());
        }
        group.set_class_symbols_from_symmetry();
        group.construct_character_table();
        group
    }
}
