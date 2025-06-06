//! Abstract groups of symmetry operations.

use anyhow::{self, ensure};
use counter::Counter;
use indexmap::IndexMap;
use itertools::Itertools;
use lazy_static::lazy_static;
use nalgebra::Vector3;
use ordered_float::OrderedFloat;
use regex::Regex;

use crate::auxiliary::geometry::{self, PositiveHemisphere};
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
use crate::symmetry::symmetry_element::{SymmetryOperation, SIG};
use crate::symmetry::symmetry_symbols::{
    deduce_mulliken_irrep_symbols, deduce_principal_classes, sort_irreps, MullikenIrcorepSymbol,
    MullikenIrrepSymbol, SymmetryClassSymbol, FORCED_C3_PRINCIPAL_GROUPS,
};

#[cfg(test)]
#[path = "symmetry_group_tests.rs"]
mod symmetry_group_tests;

#[cfg(test)]
#[path = "symmetry_chartab_tests.rs"]
mod symmetry_chartab_tests;

lazy_static! {
    static ref SQUARE_BRACKETED_GROUP_NAME_RE: Regex =
        Regex::new(r"u\[(.+)\]").expect("Regex pattern invalid.");
}

// ======================
// Type alias definitions
// ======================

/// Type for symmetry groups in which all elements are represented as mathematical unitary
/// operators. These groups admit unitary irreducible representations.
pub type UnitaryRepresentedSymmetryGroup = UnitaryRepresentedGroup<
    SymmetryOperation,
    MullikenIrrepSymbol,
    SymmetryClassSymbol<SymmetryOperation>,
>;

/// Type for symmetry groups in which half of the elements are represented as mathematical unitary
/// operators, and the other half as mathematical antiunitary operators. These groups admit
/// irreducible corepresentations.
pub type MagneticRepresentedSymmetryGroup = MagneticRepresentedGroup<
    SymmetryOperation,
    UnitaryRepresentedSymmetryGroup,
    MullikenIrcorepSymbol,
>;

// =================
// Trait definitions
// =================

/// Trait defining behaviours of symmetry groups.
pub trait SymmetryGroupProperties:
    ClassProperties<
        GroupElement = SymmetryOperation,
        ClassSymbol = SymmetryClassSymbol<SymmetryOperation>,
    > + CharacterProperties
    + Sized
{
    // ----------------
    // Required methods
    // ----------------

    /// Constructs a group from molecular symmetry *elements* (not operations).
    ///
    /// # Arguments
    ///
    /// * `sym` - A molecular symmetry structure containing the symmetry *elements*.
    /// * `infinite_order_to_finite` - Interpret infinite-order generating
    /// elements as finite-order generating elements to create a finite subgroup
    /// of an otherwise infinite group.
    ///
    /// # Returns
    ///
    /// A finite group of symmetry operations.
    fn from_molecular_symmetry(
        sym: &Symmetry,
        infinite_order_to_finite: Option<u32>,
    ) -> Result<Self, anyhow::Error>;

    /// Converts a symmetry group to its corresponding double group.
    ///
    /// # Returns
    ///
    /// The corresponding double group.
    fn to_double_group(&self) -> Result<Self, anyhow::Error>;

    /// Reorders and relabels the rows and columns of the constructed character table using
    /// symmetry-specific rules and conventions.
    ///
    /// The default implementation of this method is to do nothing. Specific trait implementations
    /// can override this to provide specific ways to canonicalise character tables.
    fn canonicalise_character_table(&mut self) {}

    // ----------------
    // Provided methods
    // ----------------

    /// Deduces the group name in Schönflies notation of a finite subgroup of an infinite molecular
    /// symmetry group.
    fn deduce_finite_group_name(&mut self) -> String {
        let (full_name, full_order, grey, double) = if let Some((_, [grp_name])) =
            SQUARE_BRACKETED_GROUP_NAME_RE
                .captures(&self.name())
                .map(|caps| caps.extract())
        {
            // u[group] means taking the unitary halving subgroup of 'group', i.e. 'group' is a grey
            // group.
            // So, we shall deduce the finite subgroup of 'group' first, and then take the unitary
            // halving subgroup of that.
            (
                grp_name.to_string(),
                self.order() * 2,
                true,
                grp_name.contains('*'),
            )
        } else {
            (self.name(), self.order(), false, self.name().contains('*'))
        };
        let finite_group = if full_name.contains('∞') {
            // C∞, C∞h, C∞v, S∞, D∞, D∞h, D∞d, or the corresponding grey groups
            if full_name.as_bytes()[0] == b'D' {
                if matches!(
                    full_name
                        .as_bytes()
                        .iter()
                        .last()
                        .expect("The last character in the group name cannot be retrieved."),
                    b'h' | b'd'
                ) {
                    if full_name.contains('θ') {
                        assert_eq!(full_order % 8, 0);
                        full_name.replace('∞', format!("{}", full_order / 8).as_str())
                    } else {
                        assert_eq!(full_order % 4, 0);
                        full_name.replace('∞', format!("{}", full_order / 4).as_str())
                    }
                } else if full_name.contains('θ') {
                    assert_eq!(full_order % 4, 0);
                    full_name.replace('∞', format!("{}", full_order / 4).as_str())
                } else {
                    assert_eq!(full_order % 2, 0);
                    full_name.replace('∞', format!("{}", full_order / 2).as_str())
                }
            } else {
                assert!(matches!(full_name.as_bytes()[0], b'C' | b'S'));
                if matches!(
                    full_name
                        .as_bytes()
                        .iter()
                        .last()
                        .expect("The last character in the group name cannot be retrieved."),
                    b'h' | b'v'
                ) {
                    if full_name.contains('θ') {
                        assert_eq!(
                            full_order % 4,
                            0,
                            "Unexpected order {} for group {full_name}.",
                            full_order
                        );
                    } else {
                        assert_eq!(
                            full_order % 2,
                            0,
                            "Unexpected order {} for group {full_name}.",
                            full_order
                        );
                    }
                    if full_order > 2 {
                        if full_name.contains('θ') {
                            full_name.replace('∞', format!("{}", full_order / 4).as_str())
                        } else {
                            full_name.replace('∞', format!("{}", full_order / 2).as_str())
                        }
                    } else {
                        assert_eq!(full_name.as_bytes()[0], b'C');
                        "Cs".to_string()
                    }
                } else {
                    full_name.replace('∞', format!("{}", full_order).as_str())
                }
            }
        } else if full_name.contains("O(3)") {
            // O(3) or the corresponding grey group
            match (full_order, double) {
                (8, false) => "D2h".to_string(),
                (16, true) => "D2h*".to_string(),
                (16, false) => "D2h + θ·D2h".to_string(),
                (32, true) => "(D2h + θ·D2h)*".to_string(),
                (48, false) => "Oh".to_string(),
                (96, true) => "Oh*".to_string(),
                (96, false) => "Oh + θ·Oh".to_string(),
                (192, true) => "(Oh + θ·Oh)*".to_string(),
                _ => panic!("Unsupported number of group elements ({full_order}) for a finite group of {full_name}."),
            }
        } else {
            // This is already a finite group.
            full_name
        };
        if grey {
            format!("u[{finite_group}]")
        } else {
            finite_group
        }
    }

    /// Returns `true` if all elements in this group are unitary.
    fn all_unitary(&self) -> bool {
        self.elements()
            .clone()
            .into_iter()
            .all(|op| !op.contains_time_reversal())
    }

    /// Returns `true` if all elements in this group are in $`\mathsf{SU}'(2)`$ or `false` if they
    /// are all in $`\mathsf{O}(3)`$.
    ///
    /// # Panics
    ///
    /// Panics if mixed $`\mathsf{SU}'(2)`$ and $`\mathsf{O}(3)`$ elements are found.
    fn is_double_group(&self) -> bool {
        if self.elements().clone().into_iter().all(|op| op.is_su2()) {
            true
        } else if self.elements().clone().into_iter().all(|op| !op.is_su2()) {
            false
        } else {
            panic!("Mixed SU(2) and SO(3) proper rotations are not allowed.");
        }
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

    /// Returns the conjugacy class symbols in this group based on molecular symmetry.
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
                        let op = self
                            .get_index(j)
                            .unwrap_or_else(|| {
                                panic!("Element with index {j} cannot be retrieved.")
                            })
                            .to_symmetry_element();
                        (
                            op.is_su2_class_1(), // prioritise class 0
                            !geometry::check_standard_positive_pole(
                                &op.proper_rotation_pole(),
                                op.threshold(),
                            ), // prioritise positive rotation
                            op.proper_fraction()
                                .map(|frac| {
                                    *frac.numer().expect(
                                        "The numerator of the proper fraction cannot be retrieved.",
                                    )
                                })
                                .or_else(|| op.raw_proper_power().map(|pp| pp.unsigned_abs()))
                                .expect(
                                    "No angle information for the proper rotation can be found.",
                                ), // prioritise small rotation angle
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
                        Some(vec![rep_ele]),
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
                        Some(vec![rep_ele]),
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
                        Some(vec![rep_ele]),
                    )
                    .unwrap_or_else(|_| {
                        panic!("Unable to construct a class symbol from `1||θ{su2}||`.")
                    });
                    assert!(undashed_class_symbols.insert(trev_sym.clone(), 1).is_none());
                    trev_sym
                } else {
                    let (mult, main_symbol, reps) = if rep_ele.is_su2() && !rep_ele.is_su2_class_1()
                    {
                        // This class might contain both class-0 and class-1 elements.
                        let class_1_count = self
                            .get_cc_index(i)
                            .unwrap_or_else(|| {
                                panic!("No conjugacy class index `{i}` can be found.")
                            })
                            .iter()
                            .filter(|&j| {
                                let op = self.get_index(*j).unwrap_or_else(|| {
                                    panic!("Element with index {j} cannot be retrieved.")
                                });
                                op.is_su2_class_1()
                            })
                            .count();
                        if old_symbol.size().rem_euclid(2) == 0
                            && class_1_count == old_symbol.size().div_euclid(2)
                        {
                            // Both class-0 and class-1 elements occur in equal number. We show one
                            // of each and halve the multiplicity.
                            let class_1_rep_ele = self
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
                                })
                                .expect("Unable to find a class-1 element in this class.");
                            (
                                old_symbol.size().div_euclid(2),
                                format!(
                                    "{}, {}",
                                    rep_ele.get_abbreviated_symbol(),
                                    class_1_rep_ele.get_abbreviated_symbol()
                                ),
                                vec![rep_ele, class_1_rep_ele],
                            )
                        } else if class_1_count > 0 {
                            // Both class-0 and class-1 elements occur, but not in equal numbers.
                            // We show all of them and set the multiplicity to 1.
                            let ops = self
                                .get_cc_index(i)
                                .unwrap_or_else(|| {
                                    panic!("No conjugacy class index `{i}` can be found.")
                                })
                                .iter()
                                .map(|&j| {
                                    self.get_index(j).unwrap_or_else(|| {
                                        panic!("Element with index {j} cannot be retrieved.")
                                    })
                                })
                                .collect_vec();
                            (
                                1,
                                ops.iter().map(|op| op.get_abbreviated_symbol()).join(", "),
                                ops,
                            )
                        } else {
                            // Only class-0 elements occur.
                            (
                                old_symbol.size(),
                                rep_ele.get_abbreviated_symbol(),
                                vec![rep_ele],
                            )
                        }
                    } else {
                        // Only class-1 elements occur, or no SU2 elements at all.
                        (
                            old_symbol.size(),
                            rep_ele.get_abbreviated_symbol(),
                            vec![rep_ele],
                        )
                    };
                    let undashed_sym =
                        SymmetryClassSymbol::new(format!("1||{main_symbol}||").as_str(), None)
                            .unwrap_or_else(|_| {
                                panic!(
                                    "Unable to construct a coarse class symbol from `1||{main_symbol}||`"
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
                        format!("{mult}||{main_symbol}|^({dash})|",).as_str(),
                        Some(reps),
                    )
                    .unwrap_or_else(|_| {
                        panic!(
                            "Unable to construct a class symbol from `{mult}||{main_symbol}|^({dash})|`"
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
    fn from_molecular_symmetry(
        sym: &Symmetry,
        infinite_order_to_finite: Option<u32>,
    ) -> Result<Self, anyhow::Error> {
        let group_name = sym
            .group_name
            .as_ref()
            .expect("No point groups found.")
            .clone();

        let handles_infinite_group = if sym.is_infinite() {
            assert!(
                infinite_order_to_finite.is_some(),
                "No finite orders specified for an infinite-order group."
            );
            infinite_order_to_finite
        } else {
            None
        };

        let sorted_operations = sym.generate_all_operations(infinite_order_to_finite);

        let mut group = Self::new(group_name.as_str(), sorted_operations)?;
        if handles_infinite_group.is_some() {
            let finite_subgroup_name = group.deduce_finite_group_name();
            group.set_finite_subgroup_name(Some(finite_subgroup_name));
        }
        let symbols = group.class_symbols_from_symmetry();
        group.set_class_symbols(&symbols);
        group.construct_irrep_character_table();
        group.canonicalise_character_table();
        Ok(group)
    }

    /// Constructs the double group of this unitary-represented group.
    ///
    /// # Returns
    ///
    /// The unitary-represented double group.
    fn to_double_group(&self) -> Result<Self, anyhow::Error> {
        log::debug!(
            "Constructing the double group for unitary-represented {}...",
            self.name()
        );

        // Check for classes of multiple C2 axes.
        let poshem = find_positive_hemisphere(self);

        if let Some(pos_hem) = poshem.as_ref() {
            log::debug!("New positive hemisphere:");
            log::debug!("{pos_hem}");
        }

        let mut su2_operations = self
            .elements()
            .clone()
            .into_iter()
            .map(|op| {
                let mut su2_op = op.to_su2_class_0();
                su2_op.set_positive_hemisphere(poshem.as_ref());
                su2_op
            })
            .collect_vec();
        let q_identity = SymmetryOperation::from_quaternion(
            (-1.0, -Vector3::z()),
            true,
            su2_operations[0].generating_element.threshold(),
            1,
            false,
            true,
            poshem,
        );
        let su2_1_operations = su2_operations
            .iter()
            .map(|op| {
                let mut q_op = op * &q_identity;
                if !q_op.is_proper() {
                    q_op = q_op.convert_to_improper_kind(&SIG);
                }

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

        let group_name = if self.name().contains('+') {
            format!("({})*", self.name())
        } else {
            self.name() + "*"
        };
        let finite_group_name = self.finite_subgroup_name().map(|name| {
            if name.contains('+') {
                format!("({name})*")
            } else {
                name.clone() + "*"
            }
        });
        let mut group = Self::new(group_name.as_str(), su2_operations)?;
        group.set_finite_subgroup_name(finite_group_name);
        let symbols = group.class_symbols_from_symmetry();
        group.set_class_symbols(&symbols);
        group.construct_irrep_character_table();
        group.canonicalise_character_table();
        Ok(group)
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
        let s2_cc = SymmetryClassSymbol::new("1||σh(Σ), σh(QΣ)||".to_string().as_str(), None)
            .unwrap_or_else(|_| {
                panic!("Unable to construct a class symbol from `1||σh(Σ), σh(QΣ)||`.")
            });
        let ts_cc = SymmetryClassSymbol::new(format!("1||θ·σh{su2_0}||").as_str(), None)
            .unwrap_or_else(|_| panic!("Unable to construct class symbol `1||θ·σh{su2_0}||`."));

        let force_principal = if FORCED_C3_PRINCIPAL_GROUPS.contains(&self.name())
            || FORCED_C3_PRINCIPAL_GROUPS.contains(
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
                    cc.is_proper() && !cc.contains_time_reversal()
                }),
                None,
            )
        } else if class_symbols.contains_key(&s_cc) || class_symbols.contains_key(&s2_cc) {
            log::debug!(
                "Horizontal mirror plane exists. Principal-axis classes will be forced to be proper."
            );
            deduce_principal_classes(
                &class_symbols,
                Some(|cc: &SymmetryClassSymbol<SymmetryOperation>| {
                    cc.is_proper() && !cc.contains_time_reversal()
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
                    cc.is_proper() && !cc.contains_time_reversal()
                }),
                None,
            )
        } else if !self
            .elements()
            .iter()
            .all(|op| !op.contains_time_reversal())
        {
            log::debug!(
                "Antiunitary elements exist without any inversion centres or horizonal mirror planes. Principal-axis classes will be forced to be unitary."
            );
            deduce_principal_classes(
                &class_symbols,
                Some(|cc: &SymmetryClassSymbol<SymmetryOperation>| !cc.contains_time_reversal()),
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
    fn from_molecular_symmetry(
        sym: &Symmetry,
        infinite_order_to_finite: Option<u32>,
    ) -> Result<Self, anyhow::Error> {
        let group_name = sym
            .group_name
            .as_ref()
            .expect("No point groups found.")
            .clone();

        let handles_infinite_group = if sym.is_infinite() {
            assert!(
                infinite_order_to_finite.is_some(),
                "No finite orders specified for an infinite-order group."
            );
            infinite_order_to_finite
        } else {
            None
        };

        let sorted_operations = sym.generate_all_operations(infinite_order_to_finite);

        ensure!(
            sorted_operations
                .iter()
                .any(SpecialSymmetryTransformation::contains_time_reversal),
            "A magnetic-represented group is requested, but no antiunitary operations can be found. \
            Ensure that time reversal is considered during symmetry-group detection."
        );

        log::debug!("Constructing the unitary subgroup for the magnetic group...");
        let unitary_operations = sorted_operations
            .iter()
            .filter_map(|op| {
                if op.contains_time_reversal() {
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
        >::new(&format!("u[{group_name}]"), unitary_operations)?;
        let uni_symbols = unitary_subgroup.class_symbols_from_symmetry();
        unitary_subgroup.set_class_symbols(&uni_symbols);
        if handles_infinite_group.is_some() {
            let finite_subgroup_name = unitary_subgroup.deduce_finite_group_name();
            unitary_subgroup.set_finite_subgroup_name(Some(finite_subgroup_name));
        }
        unitary_subgroup.construct_irrep_character_table();
        unitary_subgroup.canonicalise_character_table();
        log::debug!("Constructing the unitary subgroup for the magnetic group... Done.");

        log::debug!("Constructing the magnetic group...");
        let mut group = Self::new(group_name.as_str(), sorted_operations, unitary_subgroup)?;
        if handles_infinite_group.is_some() {
            let finite_subgroup_name = group.deduce_finite_group_name();
            group.set_finite_subgroup_name(Some(finite_subgroup_name));
        }
        let symbols = group.class_symbols_from_symmetry();
        group.set_class_symbols(&symbols);
        group.construct_ircorep_character_table();
        log::debug!("Constructing the magnetic group... Done.");
        Ok(group)
    }

    /// Constructs the double group of this magnetic-represented group.
    ///
    /// Note that the unitary subgroup of the magnetic-represented double group is not necessarily
    /// the same as the double group of the unitary subgroup. This difference can manifest when
    /// there are binary rotations or reflections and the positive hemisphere of the
    /// magnetic-represented group might be different from the positive hemisphere of the unitary
    /// subgroup.
    ///
    /// # Returns
    ///
    /// The magnetic-represented double group.
    fn to_double_group(&self) -> Result<Self, anyhow::Error> {
        log::debug!(
            "Constructing the double group for magnetic-represented {}...",
            self.name()
        );

        // Check for classes of multiple C2 axes.
        let poshem = find_positive_hemisphere(self);

        if let Some(pos_hem) = poshem.as_ref() {
            log::debug!("New positive hemisphere:");
            log::debug!("{pos_hem}");
        }

        let mut su2_operations = self
            .elements()
            .clone()
            .into_iter()
            .map(|op| {
                let mut su2_op = op.to_su2_class_0();
                su2_op.set_positive_hemisphere(poshem.as_ref());
                su2_op
            })
            .collect_vec();
        let q_identity = SymmetryOperation::from_quaternion(
            (-1.0, -Vector3::z()),
            true,
            su2_operations[0].generating_element.threshold(),
            1,
            false,
            true,
            poshem,
        );
        let su2_1_operations = su2_operations
            .iter()
            .map(|op| {
                let mut q_op = op * &q_identity;
                if !q_op.is_proper() {
                    q_op = q_op.convert_to_improper_kind(&SIG);
                }

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

        let group_name = if self.name().contains('+') {
            format!("({})*", self.name())
        } else {
            self.name() + "*"
        };
        let finite_group_name = self.finite_subgroup_name().map(|name| {
            if name.contains('+') {
                format!("({name})*")
            } else {
                name.clone() + "*"
            }
        });

        log::debug!("Constructing the double unitary subgroup for the magnetic double group...");
        let unitary_su2_operations = su2_operations
            .iter()
            .filter_map(|op| {
                if op.contains_time_reversal() {
                    None
                } else {
                    Some(op.clone())
                }
            })
            .collect::<Vec<_>>();
        let mut double_unitary_subgroup =
            UnitaryRepresentedGroup::<
                SymmetryOperation,
                MullikenIrrepSymbol,
                SymmetryClassSymbol<SymmetryOperation>,
            >::new(&format!("u[{group_name}]"), unitary_su2_operations)?;
        let uni_symbols = double_unitary_subgroup.class_symbols_from_symmetry();
        double_unitary_subgroup.set_class_symbols(&uni_symbols);
        if double_unitary_subgroup.name().contains('∞')
            || double_unitary_subgroup.name().contains("O(3)")
        {
            let finite_subgroup_name = double_unitary_subgroup.deduce_finite_group_name();
            double_unitary_subgroup.set_finite_subgroup_name(Some(finite_subgroup_name));
        }
        double_unitary_subgroup.construct_irrep_character_table();
        double_unitary_subgroup.canonicalise_character_table();
        log::debug!(
            "Constructing the double unitary subgroup for the magnetic double group... Done."
        );

        let mut group = Self::new(group_name.as_str(), su2_operations, double_unitary_subgroup)?;
        group.set_finite_subgroup_name(finite_group_name);
        let symbols = group.class_symbols_from_symmetry();
        group.set_class_symbols(&symbols);
        group.construct_ircorep_character_table();
        group.canonicalise_character_table();
        Ok(group)
    }
}

// =================
// Utility functions
// =================

/// Finds the custom positive hemisphere for a group such that any classes of odd non-coaxial binary
/// rotations or reflections have all of their elements have the poles in the positive hemisphere.
fn find_positive_hemisphere<G>(group: &G) -> Option<PositiveHemisphere>
where
    G: GroupProperties<GroupElement = SymmetryOperation>
        + ClassProperties<ClassSymbol = SymmetryClassSymbol<SymmetryOperation>>,
{
    log::debug!("Checking for classes of odd non-coaxial binary rotations or reflections...");
    let poshem = (0..group.class_number()).find_map(|cc_i| {
        let cc_symbol = group
            .get_cc_symbol_of_index(cc_i)
            .expect("Unable to retrive a conjugacy class symbol.");
        if cc_symbol.is_spatial_binary_rotation() || cc_symbol.is_spatial_reflection() {
            let cc = group
                .get_cc_index(cc_i)
                .expect("Unable to retrieve a conjugacy class.");
            if cc.len() > 1 && cc.len().rem_euclid(2) == 1 {
                let mut all_c2s = cc
                    .iter()
                    .map(|&op_i| {
                        group.get_index(op_i)
                            .expect("Unable to retrieve a group operation.")
                    })
                    .collect_vec();
                all_c2s.sort_by_key(|c2| {
                    let (axis_closeness, closest_axis) = c2.generating_element.closeness_to_cartesian_axes();
                    (OrderedFloat(axis_closeness), closest_axis)
                });
                let c2x = all_c2s.first().expect("Unable to retrieve the last C2 operation.");
                let c20 = all_c2s.last().expect("Unable to retrieve the first C2 operation.");
                let z_basis = geometry::get_standard_positive_pole(
                    &c2x
                        .generating_element
                        .raw_axis()
                        .cross(c20.generating_element.raw_axis()),
                    c2x.generating_element.threshold(),
                );
                let x_basis = *c2x.generating_element.raw_axis();
                log::debug!("Found a class of odd non-coaxial binary rotations or reflections:");
                for c2 in all_c2s.iter() {
                    log::debug!("  {c2}");
                }
                log::debug!("Adjusting the positive hemisphere to encompass all class-0 binary-rotation or reflection poles...");
                Some(PositiveHemisphere::new_spherical_disjoint_equatorial_arcs(
                    z_basis,
                    x_basis,
                    cc.len(),
                ))
            } else {
                None
            }
        } else {
            None
        }
    });
    if poshem.is_none() {
        log::debug!("No classes of odd non-coaxial binary rotations or reflections found.");
    }
    log::debug!("Checking for classes of odd non-coaxial binary rotations or reflections... Done.");
    poshem
}
