use counter::Counter;
use indexmap::IndexMap;
use itertools::Itertools;
use nalgebra::Vector3;

use crate::chartab::chartab_group::{
    CharacterProperties, IrcorepCharTabConstruction, IrrepCharTabConstruction,
};
use crate::chartab::chartab_symbols::CollectionSymbol;
use crate::chartab::{CharacterTable, RepCharacterTable};
use crate::group::class::ClassProperties;
use crate::group::{GroupProperties, GroupType, MagneticRepresentedGroup, UnitaryRepresentedGroup};
use crate::symmetry::symmetry_core::Symmetry;
use crate::symmetry::symmetry_element::symmetry_operation::{
    sort_operations, SpecialSymmetryTransformation,
};
use crate::symmetry::symmetry_element::SymmetryOperation;
use crate::symmetry::symmetry_symbols::{
    deduce_mulliken_irrep_symbols, deduce_principal_classes, sort_irreps, MullikenIrcorepSymbol,
    MullikenIrrepSymbol, SymmetryClassSymbol, FORCED_PRINCIPAL_GROUPS,
};

#[cfg(test)]
#[path = "symmetry_group_tests.rs"]
mod symmetry_group_tests;

#[cfg(test)]
#[path = "symmetry_chartab_tests.rs"]
mod symmetry_chartab_tests;

// =================
// Trait definitions
// =================

pub trait SymmetryGroupProperties:
    ClassProperties<
        GroupElement = SymmetryOperation,
        ClassSymbol = SymmetryClassSymbol<SymmetryOperation>,
    > + CharacterProperties
{
    // ----------------
    // Required methods
    // ----------------

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
    /// A finite group of symmetry operations.
    fn from_molecular_symmetry(sym: &Symmetry, infinite_order_to_finite: Option<u32>) -> Self;

    /// Converts a symmetry group to its equivalent double group.
    ///
    /// # Returns
    ///
    /// The double group.
    fn to_double_group(&self) -> Self;

    /// Reorders and relabels the rows and columns of the constructed character table using
    /// symmetry-specific rules and conventions.
    fn canonicalise_character_table(&mut self) {}

    // ----------------
    // Provided methods
    // ----------------

    /// Deduces the group name in Schönflies notation of a finite subgroup of an infinite molecular
    /// symmetry group.
    fn deduce_finite_group_name(&mut self) -> String {
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
                        assert_eq!(self.order() % 8, 0);
                        self.name()
                            .replace('∞', format!("{}", self.order() / 8).as_str())
                    } else {
                        assert_eq!(self.order() % 4, 0);
                        self.name()
                            .replace('∞', format!("{}", self.order() / 4).as_str())
                    }
                } else if self.name().contains('θ') {
                    assert_eq!(self.order() % 4, 0);
                    self.name()
                        .replace('∞', format!("{}", self.order() / 4).as_str())
                } else {
                    assert_eq!(self.order() % 2, 0);
                    self.name()
                        .replace('∞', format!("{}", self.order() / 2).as_str())
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
                        assert_eq!(self.order() % 4, 0);
                    } else {
                        assert_eq!(self.order() % 2, 0);
                    }
                    if self.order() > 2 {
                        if self.name().contains('θ') {
                            self.name()
                                .replace('∞', format!("{}", self.order() / 4).as_str())
                        } else {
                            self.name()
                                .replace('∞', format!("{}", self.order() / 2).as_str())
                        }
                    } else {
                        assert_eq!(self.name().as_bytes()[0], b'C');
                        "Cs".to_string()
                    }
                } else {
                    self.name()
                        .replace('∞', format!("{}", self.order()).as_str())
                }
            }
        } else if self.name().contains("O(3)") {
            // O(3) or the corresponding grey group
            match self.order() {
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
        self.elements()
            .clone()
            .into_iter()
            .all(|op| !op.is_antiunitary())
    }

    fn is_double_group(&self) -> bool {
        let double = if self.elements().clone().into_iter().all(|op| op.is_su2()) {
            true
        } else if self.elements().clone().into_iter().all(|op| !op.is_su2()) {
            false
        } else {
            panic!("Mixed SU(2) and SO(3) proper rotations are not allowed.");
        };
        double
    }

    /// Determines whether this group is an ordinary (double) group, a magnetic grey (double)
    /// group, or a magnetic black-and-white (double) group.
    fn group_type(&self) -> GroupType {
        let double = self.is_double_group();
        if self.all_unitary() {
            GroupType::Ordinary(double)
        } else if self
            .elements()
            .clone()
            .into_iter()
            .any(|op| op.is_time_reversal())
        {
            GroupType::MagneticGrey(double)
        } else {
            GroupType::MagneticBlackWhite(double)
        }
    }

    /// Sets the conjugacy class symbols in this group based on molecular symmetry.
    fn class_symbols_from_symmetry(&mut self) -> Vec<SymmetryClassSymbol<SymmetryOperation>> {
        log::debug!("Assigning class symbols from symmetry operations...");
        let mut undashed_class_symbols: Counter<SymmetryClassSymbol<SymmetryOperation>, usize> =
            Counter::new();

        let symmetry_class_symbols = (0..self.class_number())
            .map(|i| {
                let old_symbol = self.get_cc_symbol_of_index(i).unwrap_or_else(|| {
                    panic!("No symmetry symbol for class index `{i}` can be found.")
                });
                let rep_ele_index = *self
                    .get_cc_index(i)
                    .unwrap_or_else(|| panic!("No conjugacy class index `{i}` can be found."))
                    .iter()
                    .min_by_key(|&&j| {
                        let op = self.get_index(j).unwrap_or_else(|| {
                            panic!("Element with index {j} cannot be retrieved.")
                        });
                        (
                            op.is_su2_class_1(),
                            op.power < 0,
                            op.power,
                            op.generating_element
                                .proper_fraction()
                                .map(|frac| frac.is_sign_negative())
                                .or_else(|| {
                                    op.generating_element.raw_proper_power().map(|&pp| pp < 0)
                                })
                                .unwrap(),
                            op.generating_element
                                .proper_fraction()
                                .map(|frac| *frac.numer().unwrap())
                                .or_else(|| {
                                    op.generating_element
                                        .raw_proper_power()
                                        .map(|pp| pp.unsigned_abs())
                                })
                                .unwrap(),
                        )
                    })
                    .expect("Unable to obtain a representative element index.");
                let rep_ele = self.get_index(rep_ele_index).unwrap_or_else(|| {
                    panic!("Unable to retrieve group element with index `{rep_ele_index}`.")
                });

                let su2 = if rep_ele.is_su2_class_1() {
                    "(QΣ)"
                } else if rep_ele.is_su2() {
                    "(Σ)"
                } else {
                    ""
                };
                if rep_ele.is_identity() {
                    // E(Σ) and E(QΣ) cannot be in the same conjugacy class.
                    let id_sym = SymmetryClassSymbol::new(
                        format!("1||E{su2}||").as_str(),
                        Some(vec![rep_ele.clone()]),
                    )
                    .unwrap_or_else(|_| {
                        panic!("Unable to construct a class symbol from `1||E{su2}||`.")
                    });
                    assert!(undashed_class_symbols.insert(id_sym.clone(), 1).is_none());
                    id_sym
                } else if rep_ele.is_inversion() {
                    // i(Σ) and i(QΣ) cannot be in the same conjugacy class.
                    let inv_sym = SymmetryClassSymbol::new(
                        format!("1||i{su2}||").as_str(),
                        Some(vec![rep_ele.clone()]),
                    )
                    .unwrap_or_else(|_| {
                        panic!("Unable to construct a class symbol from `1||i{su2}||`.")
                    });
                    assert!(undashed_class_symbols.insert(inv_sym.clone(), 1).is_none());
                    inv_sym
                } else if rep_ele.is_time_reversal() {
                    // θ(Σ) and θ(QΣ) cannot be in the same conjugacy class.
                    let trev_sym = SymmetryClassSymbol::new(
                        format!("1||θ{su2}||").as_str(),
                        Some(vec![rep_ele.clone()]),
                    )
                    .unwrap_or_else(|_| {
                        panic!("Unable to construct a class symbol from `1||θ{su2}||`.")
                    });
                    assert!(undashed_class_symbols.insert(trev_sym.clone(), 1).is_none());
                    trev_sym
                } else {
                    let (mult, main_symbol, reps) = if rep_ele.is_su2() && !rep_ele.is_su2_class_1()
                    {
                        // This class might contain both class-0 and class-1 elements, in which
                        // case we show one of each in the main symbol, and set the multiplicity to
                        // be half the class size.
                        let alt_rep_ele_option = self
                            .get_cc_index(i)
                            .unwrap_or_else(|| {
                                panic!("No conjugacy class index `{i}` can be found.")
                            })
                            .iter()
                            .find_map(|&j| {
                                let op = self.get_index(j).unwrap_or_else(|| {
                                    panic!("Element with index {j} cannot be retrieved.")
                                });
                                if op.is_su2_class_1() {
                                    Some(op)
                                } else {
                                    None
                                }
                            });
                        if let Some(alt_rep_ele) = alt_rep_ele_option {
                            assert_eq!(old_symbol.size().rem_euclid(2), 0);
                            (
                                old_symbol.size().div_euclid(2),
                                format!(
                                    "{}, {}",
                                    rep_ele.get_abbreviated_symbol(),
                                    alt_rep_ele.get_abbreviated_symbol()
                                ),
                                vec![rep_ele, alt_rep_ele],
                            )
                        } else {
                            (
                                old_symbol.size(),
                                rep_ele.get_abbreviated_symbol(),
                                vec![rep_ele],
                            )
                        }
                    } else {
                        (
                            old_symbol.size(),
                            rep_ele.get_abbreviated_symbol(),
                            vec![rep_ele],
                        )
                    };
                    let undashed_sym =
                        SymmetryClassSymbol::new(format!("1||{}||", main_symbol).as_str(), None)
                            .unwrap_or_else(|_| {
                                panic!(
                                    "Unable to construct a coarse class symbol from `1||{}||`",
                                    main_symbol
                                )
                            });
                    undashed_class_symbols
                        .entry(undashed_sym.clone())
                        .and_modify(|counter| *counter += 1)
                        .or_insert(1);
                    let dash = undashed_class_symbols
                        .get(&undashed_sym)
                        .map(|&counter| "'".repeat(counter - 1))
                        .unwrap();

                    SymmetryClassSymbol::new(
                        format!("{mult}||{}|^({dash})|", main_symbol,).as_str(),
                        Some(reps),
                    )
                    .unwrap_or_else(|_| {
                        panic!(
                            "Unable to construct a class symbol from `{mult}||{}|^({dash})|`",
                            main_symbol
                        )
                    })
                }
            })
            .collect::<Vec<_>>();
        log::debug!("Assigning class symbols from symmetry operations... Done.");
        symmetry_class_symbols
    }
}

// =====================
// Trait implementations
// =====================

// -----------------------
// UnitaryRepresentedGroup
// -----------------------

impl SymmetryGroupProperties
    for UnitaryRepresentedGroup<
        SymmetryOperation,
        MullikenIrrepSymbol,
        SymmetryClassSymbol<SymmetryOperation>,
    >
{
    /// Constructs a unitary-represented group from molecular symmetry *elements* (not operations).
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
    /// A unitary-represented group of the symmetry operations generated by `sym`.
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
            let finite_subgroup_name = group.deduce_finite_group_name();
            group.set_finite_subgroup_name(Some(finite_subgroup_name));
        }
        let symbols = group.class_symbols_from_symmetry();
        group.set_class_symbols(&symbols);
        group.construct_irrep_character_table();
        group.canonicalise_character_table();
        group
    }

    fn to_double_group(&self) -> Self {
        let mut su2_operations = self
            .elements()
            .clone()
            .into_iter()
            .map(|op| op.to_su2_class_0())
            .collect_vec();
        let q_identity = SymmetryOperation::from_quaternion(
            (-1.0, -Vector3::z()),
            true,
            su2_operations[0].generating_element.threshold(),
            1,
            false,
            true,
        );
        let su2_1_operations = su2_operations
            .iter()
            .map(|op| {
                let mut q_op = op * &q_identity;

                // Multiplying by q_identity does not change subscript/superscript information
                // such as inversion parity or mirror plane type.
                q_op.generating_element.additional_subscript =
                    op.generating_element.additional_subscript.clone();
                q_op.generating_element.additional_superscript =
                    op.generating_element.additional_superscript.clone();
                q_op
            })
            .collect_vec();
        su2_operations.extend(su2_1_operations.into_iter());
        sort_operations(&mut su2_operations);

        let group_name = self.name().clone() + "*";
        let finite_group_name = self.finite_subgroup_name().map(|name| name.clone() + "*");
        let mut group = Self::new(group_name.as_str(), su2_operations);
        group.set_finite_subgroup_name(finite_group_name);
        let symbols = group.class_symbols_from_symmetry();
        group.set_class_symbols(&symbols);
        group.construct_irrep_character_table();
        group.canonicalise_character_table();
        group
    }

    /// Reorders and relabels the rows and columns of the constructed character table using
    /// Mulliken conventions for the irreducible representations.
    fn canonicalise_character_table(&mut self) {
        let old_chartab = self.character_table();
        let class_symbols: IndexMap<_, _> = (0..self.class_number())
            .map(|i| (self.get_cc_symbol_of_index(i).unwrap(), i))
            .collect();

        let su2_0 = if self.is_double_group() { "(Σ)" } else { "" };
        let i_cc = SymmetryClassSymbol::new(format!("1||i{su2_0}||").as_str(), None)
            .unwrap_or_else(|_| panic!("Unable to construct a class symbol from `1||i{su2_0}||`."));
        let s_cc = SymmetryClassSymbol::new(format!("1||σh{su2_0}||").as_str(), None)
            .unwrap_or_else(|_| {
                panic!("Unable to construct a class symbol from `1||σh{su2_0}||`.")
            });
        let ts_cc = SymmetryClassSymbol::new(format!("1||θ·σh{su2_0}||").as_str(), None)
            .unwrap_or_else(|_| panic!("Unable to construct class symbol `1||θ·σh{su2_0}||`."));

        let force_principal = if FORCED_PRINCIPAL_GROUPS.contains(&self.name())
            || FORCED_PRINCIPAL_GROUPS.contains(
                self.finite_subgroup_name()
                    .unwrap_or(&String::new())
                    .as_str(),
            ) {
            let c3_cc: SymmetryClassSymbol<SymmetryOperation> =
                SymmetryClassSymbol::new(format!("8||C3{su2_0}||").as_str(), None).unwrap_or_else(
                    |_| panic!("Unable to construct a class symbol from `8||C3{su2_0}||`."),
                );
            log::debug!(
                "Group is {}. Principal-axis classes will be forced to be {}. This is to obtain non-standard Mulliken symbols that are in line with conventions in the literature.",
                self.name(),
                c3_cc
            );
            Some(c3_cc)
        } else {
            None
        };

        let principal_classes = if force_principal.is_some() {
            deduce_principal_classes(
                &class_symbols,
                None::<fn(&SymmetryClassSymbol<SymmetryOperation>) -> bool>,
                force_principal,
            )
        } else if class_symbols.contains_key(&i_cc) {
            log::debug!(
                "Inversion centre exists. Principal-axis classes will be forced to be proper."
            );
            deduce_principal_classes(
                &class_symbols,
                Some(|cc: &SymmetryClassSymbol<SymmetryOperation>| {
                    cc.is_proper() && !cc.is_antiunitary()
                }),
                None,
            )
        } else if class_symbols.contains_key(&s_cc) {
            log::debug!(
                "Horizontal mirror plane exists. Principal-axis classes will be forced to be proper."
            );
            deduce_principal_classes(
                &class_symbols,
                Some(|cc: &SymmetryClassSymbol<SymmetryOperation>| {
                    cc.is_proper() && !cc.is_antiunitary()
                }),
                None,
            )
        } else if class_symbols.contains_key(&ts_cc) {
            log::debug!(
                "Time-reversed horizontal mirror plane exists. Principal-axis classes will be forced to be proper."
            );
            deduce_principal_classes(
                &class_symbols,
                Some(|cc: &SymmetryClassSymbol<SymmetryOperation>| {
                    cc.is_proper() && !cc.is_antiunitary()
                }),
                None,
            )
        } else if !self.elements().iter().all(|op| !op.is_antiunitary()) {
            log::debug!(
                "Antiunitary elements exist without any inversion centres or horizonal mirror planes. Principal-axis classes will be forced to be unitary."
            );
            deduce_principal_classes(
                &class_symbols,
                Some(|cc: &SymmetryClassSymbol<SymmetryOperation>| !cc.is_antiunitary()),
                None,
            )
        } else {
            deduce_principal_classes(
                &class_symbols,
                None::<fn(&SymmetryClassSymbol<SymmetryOperation>) -> bool>,
                None,
            )
        };

        let (char_arr, sorted_fs) = sort_irreps(
            &old_chartab.array().view(),
            &old_chartab.frobenius_schurs.values().copied().collect_vec(),
            &class_symbols,
            &principal_classes,
        );

        let ordered_irreps =
            deduce_mulliken_irrep_symbols(&char_arr.view(), &class_symbols, &principal_classes);

        self.irrep_character_table = Some(RepCharacterTable::new(
            &old_chartab.name,
            &ordered_irreps,
            &class_symbols.keys().cloned().collect::<Vec<_>>(),
            &principal_classes,
            char_arr,
            &sorted_fs,
        ));
    }
}

// ------------------------
// MagneticRepresentedGroup
// ------------------------

impl SymmetryGroupProperties
    for MagneticRepresentedGroup<
        SymmetryOperation,
        UnitaryRepresentedGroup<
            SymmetryOperation,
            MullikenIrrepSymbol,
            SymmetryClassSymbol<SymmetryOperation>,
        >,
        MullikenIrcorepSymbol,
    >
{
    /// Constructs a magnetic-represented group from molecular symmetry *elements* (not operations).
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
    /// A magnetic-represented group of the symmetry operations generated by `sym`.
    ///
    /// # Panics
    ///
    /// Panics if `sym` generates no antiunitary operations.
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

        assert!(
            sorted_operations
                .iter()
                .any(SpecialSymmetryTransformation::is_antiunitary),
            "No antiunitary operations found from the `Symmetry` structure."
        );

        log::debug!("Constructing the unitary subgroup for the magnetic group...");
        let unitary_operations = sorted_operations
            .iter()
            .filter_map(|op| {
                if op.is_antiunitary() {
                    None
                } else {
                    Some(op.clone())
                }
            })
            .collect::<Vec<_>>();
        let mut unitary_subgroup = UnitaryRepresentedGroup::<
            SymmetryOperation,
            MullikenIrrepSymbol,
            SymmetryClassSymbol<SymmetryOperation>,
        >::new(group_name.as_str(), unitary_operations);
        let uni_symbols = unitary_subgroup.class_symbols_from_symmetry();
        unitary_subgroup.set_class_symbols(&uni_symbols);
        unitary_subgroup.construct_irrep_character_table();
        unitary_subgroup.canonicalise_character_table();
        log::debug!("Constructing the unitary subgroup for the magnetic group... Done.");

        log::debug!("Constructing the magnetic group...");
        let mut group = Self::new(group_name.as_str(), sorted_operations, unitary_subgroup);
        if handles_infinite_group.is_some() {
            let finite_subgroup_name = group.deduce_finite_group_name();
            group.set_finite_subgroup_name(Some(finite_subgroup_name));
        }
        let symbols = group.class_symbols_from_symmetry();
        group.set_class_symbols(&symbols);
        group.construct_ircorep_character_table();
        log::debug!("Constructing the magnetic group... Done.");
        group
    }

    fn to_double_group(&self) -> Self {
        let mut su2_operations = self
            .elements()
            .clone()
            .into_iter()
            .map(|op| op.to_su2_class_0())
            .collect_vec();
        let q_identity = SymmetryOperation::from_quaternion(
            (-1.0, -Vector3::z()),
            true,
            su2_operations[0].generating_element.threshold(),
            1,
            false,
            true,
        );
        let su2_1_operations = su2_operations
            .iter()
            .map(|op| {
                let mut q_op = op * &q_identity;

                // Multiplying by q_identity does not change subscript/superscript information
                // such as inversion parity or mirror plane type.
                q_op.generating_element.additional_subscript =
                    op.generating_element.additional_subscript.clone();
                q_op.generating_element.additional_superscript =
                    op.generating_element.additional_superscript.clone();
                q_op
            })
            .collect_vec();
        su2_operations.extend(su2_1_operations.into_iter());
        sort_operations(&mut su2_operations);

        let double_unitary_subgroup = self.unitary_subgroup().to_double_group();

        let group_name = self.name().clone() + "*";
        let finite_group_name = self.finite_subgroup_name().map(|name| name.clone() + "*");
        let mut group = Self::new(group_name.as_str(), su2_operations, double_unitary_subgroup);
        group.set_finite_subgroup_name(finite_group_name);
        let symbols = group.class_symbols_from_symmetry();
        group.set_class_symbols(&symbols);
        group.construct_ircorep_character_table();
        group.canonicalise_character_table();
        group
    }
}
