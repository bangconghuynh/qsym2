use std::collections::HashSet;
use std::fmt;
use std::hash::Hash;
use std::ops::Mul;

use derive_builder::Builder;
use indexmap::IndexMap;
use itertools::Itertools;
use ndarray::{s, Array2, Array3, Axis};

use crate::symmetry::symmetry_element::symmetry_operation::FiniteOrder;
use crate::symmetry::symmetry_symbols::{ClassSymbol, MullikenIrrepSymbol};
use crate::chartab::CharacterTable;

use super::{Group, GroupProperties, MagneticRepresentedGroup, UnitaryRepresentedGroup};

#[derive(Builder, Clone)]
pub struct ClassStructure<T>
where
    T: Mul<Output = T> + Hash + Eq + Clone + Sync + fmt::Debug + FiniteOrder,
{
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
    element_to_conjugacy_classes: Vec<Option<usize>>,

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

impl<T> ClassStructureBuilder<T>
where
    T: Mul<Output = T> + Hash + Eq + Clone + Sync + fmt::Debug + FiniteOrder,
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

    fn conjugacy_class_symbols(&mut self, elements: &IndexMap<T, usize>) -> &mut Self {
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
                let (rep_ele, _) = elements.get_index(rep_ele_index).unwrap_or_else(|| {
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

    fn inverse_conjugacy_classes(&mut self, ctb: &Array2<usize>) -> &mut Self {
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
                .expect("Map from element to conjugacy class has not been found.")[g_inv]
                .unwrap_or_else(|| {
                    panic!("Element index `{g_inv}` does not have a conjugacy class.",)
                });
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
    fn class_matrix(&mut self, ctb: &Array2<usize>) -> &mut Self {
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
                let cayley_block_rs = ctb
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

impl<T> ClassStructure<T>
where
    T: Mul<Output = T> + Hash + Eq + Clone + Sync + fmt::Debug + FiniteOrder,
{
    /// Returns a builder to construct a new class structure.
    ///
    /// # Returns
    ///
    /// A builder to construct a new class structure.
    fn builder() -> ClassStructureBuilder<T> {
        ClassStructureBuilder::default()
    }

    #[must_use]
    fn class_number(&self) -> usize {
        self.conjugacy_classes.len()
    }

    fn new(
        group: &Group<T>,
        conjugacy_classes: Vec<HashSet<usize>>,
        element_to_conjugacy_classes: Vec<Option<usize>>,
    ) -> Self {
        let ctb = group
            .cayley_table
            .as_ref()
            .expect("Cayley table not found for this group.");
        Self::builder()
            .conjugacy_classes(conjugacy_classes)
            .element_to_conjugacy_classes(element_to_conjugacy_classes)
            .conjugacy_class_transversal()
            .conjugacy_class_symbols(&group.elements)
            .inverse_conjugacy_classes(ctb)
            .class_matrix(ctb)
            .build()
            .expect("Unable to construct a `ClassStructure`.")
    }

    pub fn set_class_symbols(&mut self, csyms: IndexMap<ClassSymbol<T>, usize>) {
        self.conjugacy_class_symbols = csyms;
    }
}

pub trait ClassProperties: GroupProperties<GroupElement = Self::ClassElement>
where
    Self::ClassElement: Mul<Output = Self::ClassElement> + Hash + Eq + Clone + Sync + fmt::Debug + FiniteOrder,
    for<'a, 'b> &'b Self::ClassElement: Mul<&'a Self::ClassElement, Output = Self::ClassElement>,
{
    type ClassElement;

    #[must_use]
    fn class_structure(&self) -> &ClassStructure<Self::ClassElement>;

    fn class_structure_mut(&mut self) -> &mut ClassStructure<Self::ClassElement>;

    fn compute_class_structure(&mut self);

    #[must_use]
    fn conjugacy_classes(&self) -> &Vec<HashSet<usize>> {
        &self.class_structure().conjugacy_classes
    }

    #[must_use]
    fn element_to_conjugacy_classes(&self) -> &Vec<Option<usize>> {
        &self.class_structure().element_to_conjugacy_classes
    }

    #[must_use]
    fn conjugacy_class_transversal(&self) -> &Vec<usize> {
        &self.class_structure().conjugacy_class_transversal
    }

    #[must_use]
    fn conjugacy_class_symbols(&self) -> &IndexMap<ClassSymbol<Self::ClassElement>, usize> {
        &self.class_structure().conjugacy_class_symbols
    }

    #[must_use]
    fn inverse_conjugacy_classes(&self) -> &Vec<usize> {
        &self.class_structure().inverse_conjugacy_classes
    }

    #[must_use]
    fn class_matrix(&self) -> &Array3<usize> {
        &self.class_structure().class_matrix
    }

    #[must_use]
    fn class_number(&self) -> usize {
        self.class_structure().class_number()
    }
}

impl<T> ClassProperties for UnitaryRepresentedGroup<T>
where
    T: Mul<Output = T> + Hash + Eq + Clone + Sync + fmt::Debug + FiniteOrder,
    for<'a, 'b> &'b T: Mul<&'a T, Output = T>,
{
    type ClassElement = T;

    fn class_structure(&self) -> &ClassStructure<Self::ClassElement> {
        self.class_structure
            .as_ref()
            .expect("Class structure not found for this group.")
    }

    fn class_structure_mut(&mut self) -> &mut ClassStructure<Self::ClassElement> {
        self.class_structure
            .as_mut()
            .expect("Class structure not found for this group.")
    }

    fn compute_class_structure(&mut self) {
        log::debug!("Finding unitary conjugacy classes...");
        let order = self.abstract_group.order();
        let (ccs, e2ccs) = if self.abstract_group.is_abelian() {
            log::debug!("Abelian group found.");
            // Abelian group; each element is in its own conjugacy class.
            (
                (0usize..order)
                    .map(|i| HashSet::from([i]))
                    .collect::<Vec<_>>(),
                (0usize..order).map(|i| Some(i)).collect::<Vec<_>>(),
            )
        } else {
            // Non-Abelian group.
            log::debug!("Non-Abelian group found.");
            let mut ccs: Vec<HashSet<usize>> = vec![HashSet::from([0usize])];
            let mut e2ccs = vec![0usize; order];
            let mut remaining_elements: HashSet<usize> = (1usize..order).collect();
            let ctb = self
                .abstract_group
                .cayley_table
                .as_ref()
                .expect("Cayley table not found.");

            while !remaining_elements.is_empty() {
                // For a fixed g, find all h such that sg = hs for all s in the group.
                let g = *remaining_elements
                    .iter()
                    .next()
                    .expect("Unexpected empty `remaining_elements`.");
                let mut cur_cc = HashSet::from([g]);
                for s in 0usize..order {
                    let sg = ctb[[s, g]];
                    let ctb_xs = ctb.slice(s![.., s]);
                    let h = ctb_xs.iter().position(|&x| x == sg).unwrap_or_else(|| {
                        panic!("No element `{sg}` can be found in column `{s}` of Cayley table.")
                    });
                    if remaining_elements.contains(&h) {
                        remaining_elements.remove(&h);
                        cur_cc.insert(h);
                    }
                }
                ccs.push(cur_cc);
            }
            ccs.sort_by_key(|cc| {
                *cc.iter()
                    .min()
                    .expect("Unable to find the minimum element index in one conjugacy class.")
            });
            ccs.iter().enumerate().for_each(|(i, cc)| {
                cc.iter().for_each(|&j| e2ccs[j] = i);
            });
            assert!(e2ccs.iter().skip(1).all(|&x| x > 0));
            (ccs, e2ccs.iter().map(|&i| Some(i)).collect::<Vec<_>>())
        };
        log::debug!("Finding unitary conjugacy classes... Done.");

        let class_structure = ClassStructure::<T>::new(&self.abstract_group, ccs, e2ccs);
        self.class_structure = Some(class_structure);
    }
}

impl<T, U, UC> ClassProperties for MagneticRepresentedGroup<T, U, UC>
where
    T: Mul<Output = T> + Hash + Eq + Clone + Sync + fmt::Debug + FiniteOrder,
    for<'a, 'b> &'b T: Mul<&'a T, Output = T>,
    U: Clone + GroupProperties<GroupElement = T>,
    UC: CharacterTable<MullikenIrrepSymbol, ClassSymbol<T>>
{
    type ClassElement = T;

    fn class_structure(&self) -> &ClassStructure<Self::ClassElement> {
        self.class_structure
            .as_ref()
            .expect("Class structure not found for this group.")
    }

    fn class_structure_mut(&mut self) -> &mut ClassStructure<Self::ClassElement> {
        self.class_structure
            .as_mut()
            .expect("Class structure not found for this group.")
    }

    fn compute_class_structure(&mut self) {
        log::debug!("Finding magnetic conjugacy classes...");
        let order = self.abstract_group.order();
        let mut ccs: Vec<HashSet<usize>> = vec![HashSet::from([0usize])];
        let mut e2ccs = vec![None; order];
        let mut remaining_unitary_elements = self
            .elements()
            .iter()
            .skip(1)
            .filter_map(|(op, &i)| if self.check_elem_antiunitary(op) { None } else { Some(i) })
            .collect::<HashSet<usize>>();
        let ctb = self
            .abstract_group
            .cayley_table
            .as_ref()
            .expect("Cayley table not found.");

        while !remaining_unitary_elements.is_empty() {
            // For a fixed unitary g, find all unitary h such that ug = hu for all unitary u
            // in the group, and all unitary h such that ag^(-1) = ha for all antiunitary a in
            // the group.
            let g = *remaining_unitary_elements
                .iter()
                .next()
                .expect("Unexpected empty `remaining_elements`.");
            let ctb_xg = ctb.slice(s![.., g]);
            let ginv = ctb_xg
                .iter()
                .position(|&x| x == 0)
                .unwrap_or_else(|| panic!("The inverse of `{g}` cannot be found."));
            let mut cur_cc = HashSet::from([g]);
            for (op, &s) in self.elements().iter() {
                let h = if self.check_elem_antiunitary(op) {
                    // s denotes a.
                    let sginv = ctb[[s, ginv]];
                    let ctb_xs = ctb.slice(s![.., s]);
                    ctb_xs.iter().position(|&x| x == sginv).unwrap_or_else(|| {
                        panic!("No element `{sginv}` can be found in column `{s}` of Cayley table.")
                    })
                } else {
                    // s denotes u.
                    let sg = ctb[[s, g]];
                    let ctb_xs = ctb.slice(s![.., s]);
                    ctb_xs.iter().position(|&x| x == sg).unwrap_or_else(|| {
                        panic!(
                            "No element `{sg}` can be found in column `{s}` of Cayley table."
                        )
                    })
                };
                if remaining_unitary_elements.contains(&h) {
                    remaining_unitary_elements.remove(&h);
                    cur_cc.insert(h);
                }
            }
            ccs.push(cur_cc);
        }
        ccs.sort_by_key(|cc| {
            *cc.iter()
                .min()
                .expect("Unable to find the minimum element index in one conjugacy class.")
        });
        ccs.iter().enumerate().for_each(|(i, cc)| {
            cc.iter().for_each(|&j| e2ccs[j] = Some(i));
        });
        assert!(e2ccs.iter().skip(1).all(|x_opt| if let Some(x) = x_opt { *x > 0 } else { true }));

        let class_structure = ClassStructure::<T>::new(&self.abstract_group, ccs, e2ccs);
        self.class_structure = Some(class_structure);
        log::debug!("Finding magnetic conjugacy classes... Done.");
    }
}
