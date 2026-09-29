//! Basis for non-orthogonal configuration interaction of Slater determinants.

use std::collections::VecDeque;
use std::fmt;

use anyhow::{self, ensure, format_err};
use derive_builder::Builder;
use itertools::structs::Product;
use itertools::{Itertools, izip};
use ndarray::{Array1, Array2, Axis};
use ndarray_linalg::solve::Determinant;
use ndarray_linalg::types::Lapack;
use num_complex::ComplexFloat;
use num_traits::Float;
use rayon::prelude::*;

use crate::angmom::spinor_rotation_3d::StructureConstraint;
use crate::group::GroupProperties;
use crate::target::determinant::SlaterDeterminant;

#[path = "basis_transformation.rs"]
mod basis_transformation;

#[cfg(test)]
#[path = "basis_tests.rs"]
mod basis_tests;

// =====
// Basis
// =====

// -----------------
// Trait definitions
// -----------------

/// Trait defining behaviours of a basis consisting of linear-space items.
pub trait Basis<I> {
    /// Type of the iterator over items in the basis.
    type BasisIter<'b>: Iterator<Item = Result<I, anyhow::Error>>
    where
        Self: 'b;

    /// Returns the number of items in the basis.
    fn n_items(&self) -> usize;

    /// An iterator over items in the basis.
    fn iter(&self) -> Self::BasisIter<'_>;

    /// Shared reference to the first item in the basis.
    fn first(&self) -> Option<I>;
}

// --------------------------------------
// Struct definitions and implementations
// --------------------------------------

// ~~~~~~~~~~~~~~~~~~~~~~
// Lazy basis from orbits
// ~~~~~~~~~~~~~~~~~~~~~~

#[derive(Builder, Clone)]
pub struct OrbitBasis<'g, G, I>
where
    G: GroupProperties,
{
    /// The origins from which orbits are generated.
    origins: Vec<I>,

    /// The group acting on the origins to generate orbits, the concatenation of which forms the
    /// basis.
    group: &'g G,

    /// Additional operators acting on the entire orbit basis (right-most operator acts first). Each
    /// operator has an associated action that defines how it operates on the elements in the
    /// orbit basis.
    #[builder(default = "None")]
    #[allow(clippy::type_complexity)]
    prefactors: Option<
        VecDeque<(
            G::GroupElement,
            fn(&G::GroupElement, &I) -> Result<I, anyhow::Error>,
        )>,
    >,

    /// A function defining the action of each group element on the origin.
    action: fn(&G::GroupElement, &I) -> Result<I, anyhow::Error>,
}

impl<'g, G, I> OrbitBasis<'g, G, I>
where
    G: GroupProperties + Clone,
    I: Clone,
{
    pub fn builder() -> OrbitBasisBuilder<'g, G, I> {
        OrbitBasisBuilder::<'g, G, I>::default()
    }

    /// The origins from which orbits are generated.
    pub fn origins(&self) -> &Vec<I> {
        &self.origins
    }

    /// The group acting on the origins to generate orbits, the concatenation of which forms the
    /// basis.
    pub fn group(&self) -> &G {
        self.group
    }

    /// Additional operators acting on the entire orbit basis (right-most operator acts first). Each
    /// operator has an associated action that defines how it operatres on the elements in the
    /// orbit basis.
    #[allow(clippy::type_complexity)]
    pub fn prefactors(
        &self,
    ) -> Option<
        &VecDeque<(
            G::GroupElement,
            fn(&G::GroupElement, &I) -> Result<I, anyhow::Error>,
        )>,
    > {
        self.prefactors.as_ref()
    }
}

impl<'g, G, I> OrbitBasis<'g, G, I>
where
    G: GroupProperties,
    I: Clone,
{
    /// Converts this orbit basis into the equivalent eager basis.
    pub fn to_eager(&self) -> Result<EagerBasis<I>, anyhow::Error> {
        let elements = self.iter().collect::<Result<Vec<_>, _>>()?;
        EagerBasis::builder()
            .elements(elements)
            .build()
            .map_err(|err| format_err!(err))
    }
}

impl<'g, G, I> Basis<I> for OrbitBasis<'g, G, I>
where
    G: GroupProperties,
    I: Clone,
{
    type BasisIter<'b>
        = OrbitBasisIterator<G, I>
    where
        Self: 'b;

    fn n_items(&self) -> usize {
        self.origins.len() * self.group.order()
    }

    /// Iterates over the elements of the [`OrbitBasis`]. Each element is indexed by `iI` where `i`
    /// enumerates the group elements and `I` enumerates the origins. `I` is the fast index and `i`
    /// the slow one.
    fn iter(&self) -> Self::BasisIter<'_> {
        OrbitBasisIterator::new(
            self.prefactors.clone(),
            self.group,
            self.origins.clone(),
            self.action,
        )
    }

    fn first(&self) -> Option<I> {
        if let Some(prefactors) = self.prefactors.as_ref() {
            prefactors
                .iter()
                .rev()
                .try_fold(self.origins.first()?.clone(), |acc, (symop, action)| {
                    (action)(symop, &acc).ok()
                })
        } else {
            self.origins.first().cloned()
        }
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Iterator for lazy basis from orbits
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Lazy iterator for basis constructed from the concatenation of orbits generated from multiple
/// origins.
pub struct OrbitBasisIterator<G, I>
where
    G: GroupProperties,
{
    #[allow(clippy::type_complexity)]
    prefactors: Option<
        VecDeque<(
            G::GroupElement,
            fn(&G::GroupElement, &I) -> Result<I, anyhow::Error>,
        )>,
    >,

    /// A mutable iterator over the Cartesian product between the group elements and the origins.
    group_origin_iter: Product<
        <<G as GroupProperties>::ElementCollection as IntoIterator>::IntoIter,
        std::vec::IntoIter<I>,
    >,

    /// A function defining the action of each group element on the origin.
    action: fn(&G::GroupElement, &I) -> Result<I, anyhow::Error>,
}

impl<G, I> OrbitBasisIterator<G, I>
where
    G: GroupProperties,
    I: Clone,
{
    /// Creates and returns a new orbit basis iterator.
    ///
    /// Each element returned by the iterator is indexed by `iI` where `i` enumerates the group
    /// elements and `I` enumerates the origins. `I` is the fast index and `i` the slow one.
    ///
    /// # Arguments
    ///
    /// * `group` - A group.
    /// * `origins` - A slice of origins.
    /// * `action` - A function or closure defining the action of each group element on the origins.
    ///
    /// # Returns
    ///
    /// An orbit basis iterator.
    #[allow(clippy::type_complexity)]
    fn new(
        prefactors: Option<
            VecDeque<(
                G::GroupElement,
                fn(&G::GroupElement, &I) -> Result<I, anyhow::Error>,
            )>,
        >,
        group: &G,
        origins: Vec<I>,
        action: fn(&G::GroupElement, &I) -> Result<I, anyhow::Error>,
    ) -> Self {
        Self {
            prefactors,
            group_origin_iter: group
                .elements()
                .clone()
                .into_iter()
                .cartesian_product(origins),
            action,
        }
    }
}

impl<G, I> Iterator for OrbitBasisIterator<G, I>
where
    G: GroupProperties,
    I: Clone,
{
    type Item = Result<I, anyhow::Error>;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(prefactors) = self.prefactors.as_ref() {
            self.group_origin_iter.next().and_then(|(op, origin)| {
                prefactors
                    .iter()
                    .rev()
                    .try_fold((self.action)(&op, &origin), |acc_res, (symop, action)| {
                        acc_res.map(|acc| (action)(symop, &acc))
                    })
                    .ok()
            })
        } else {
            self.group_origin_iter
                .next()
                .map(|(op, origin)| (self.action)(&op, &origin))
        }
    }
}

// ~~~~~~~~~~~
// Eager basis
// ~~~~~~~~~~~

#[derive(Builder, Clone)]
pub struct EagerBasis<I: Clone> {
    /// The elements in the basis.
    elements: Vec<I>,
}

impl<I: Clone> EagerBasis<I> {
    pub fn builder() -> EagerBasisBuilder<I> {
        EagerBasisBuilder::default()
    }
}

impl<I: Clone> Basis<I> for EagerBasis<I> {
    type BasisIter<'b>
        = std::vec::IntoIter<Result<I, anyhow::Error>>
    where
        Self: 'b;

    fn n_items(&self) -> usize {
        self.elements.len()
    }

    fn iter(&self) -> Self::BasisIter<'_> {
        self.elements
            .iter()
            .cloned()
            .map(Ok)
            .collect_vec()
            .into_iter()
    }

    fn first(&self) -> Option<I> {
        self.elements.first().cloned()
    }
}

// ~~~~~~~~~
// FCI basis
// ~~~~~~~~~

#[derive(Builder, Clone)]
#[builder(build_fn(validate = "Self::validate"))]
pub struct FCIBasis<'a, T, SC>
where
    T: ComplexFloat + Lapack,
    SC: StructureConstraint + fmt::Display,
{
    /// The reference determinant of the full-configuration-interaction basis. This determinant
    /// carries both occupied and virtual molecular orbitals.
    reference: SlaterDeterminant<'a, T, SC>,

    /// Vector of full-configuration-interaction occupation patterns.
    occupation_patterns: Vec<Vec<Array1<<T as ComplexFloat>::Real>>>,
}

impl<'a, T, SC> FCIBasisBuilder<'a, T, SC>
where
    T: ComplexFloat + Lapack,
    SC: StructureConstraint + fmt::Display,
{
    fn validate(&self) -> Result<(), String> {
        if let Some(ref reference) = self.reference
            && let Some(ref occupation_patterns) = self.occupation_patterns
        {
            let reference_occs = reference.occupations();
            if occupation_patterns.iter().all(|occs| {
                occs.len() == reference_occs.len()
                    && occs
                        .iter()
                        .zip(reference_occs)
                        .all(|(occ, reference_occ)| occ.shape() == reference_occ.shape())
            }) {
                Ok(())
            } else {
                Err("Inconsistent occupation patterns.".to_string())
            }
        } else {
            Err("Missing `reference` or `occupation_patterns`.".to_string())
        }
    }
}

impl<'a, T, SC> FCIBasis<'a, T, SC>
where
    T: ComplexFloat + Lapack,
    SC: StructureConstraint + fmt::Display + Clone,
{
    pub fn builder() -> FCIBasisBuilder<'a, T, SC> {
        FCIBasisBuilder::<T, SC>::default()
    }

    /// The reference determinant of the full-configuration-interaction basis.
    pub fn reference(&self) -> &SlaterDeterminant<'a, T, SC> {
        &self.reference
    }

    /// The full-configuration-interaction occupation patterns.
    pub fn occupation_patterns(&self) -> &Vec<Vec<Array1<<T as ComplexFloat>::Real>>> {
        &self.occupation_patterns
    }
}

impl<'a, T, SC> FCIBasis<'a, T, SC>
where
    T: ComplexFloat + Lapack,
    SC: StructureConstraint + fmt::Display + Clone + PartialEq,
{
    /// Computes the FCI metric (*i.e.* the overlap matrix between the basis elements) between this
    /// FCI basis and another.
    ///
    /// # Arguments
    ///
    /// * `other` - Another FCI basis.
    /// * `metric` - The atomic-orbital overlap matrix with respect to the conventional sesquilinear
    ///   inner product.
    /// * `metric_h` - The atomic-orbital overlap matrix with respect to the bilinear inner product.
    ///
    /// # Returns
    ///
    /// The overmap matrix between the basis elements.
    pub fn fci_metric(
        &self,
        other: &FCIBasis<'a, T, SC>,
        metric: Option<&Array2<T>>,
        metric_h: Option<&Array2<T>>,
    ) -> Result<Array2<T>, anyhow::Error> {
        let sao = metric.ok_or_else(|| format_err!("No atomic-orbital metric found."))?;
        let sao_h = metric_h.unwrap_or(sao);

        let s_ref = self.reference();
        let o_ref = other.reference();
        let thresh = Float::sqrt(s_ref.threshold() * o_ref.threshold());

        ensure!(
            s_ref.structure_constraint() == o_ref.structure_constraint(),
            "Inconsistent structure constraints between the two FCI bases."
        );
        ensure!(
            s_ref.coefficients().len() == o_ref.coefficients().len(),
            "Inconsistent numbers of coefficient matrices between the references of the two FCI bases."
        );
        ensure!(
            s_ref.baos() == o_ref.baos(),
            "Inconsistent basis angular order between the two FCI bases."
        );
        ensure!(
            s_ref.complex_symmetric() == s_ref.complex_symmetric(),
            "Inconsistent `complex_symmetric` between the two FCI bases."
        );

        let mo_ov_mats = s_ref
            .coefficients()
            .iter()
            .zip(o_ref.coefficients().iter())
            .map(|(cw, cx)| {
                if s_ref.complex_symmetric() {
                    match (s_ref.complex_conjugated(), o_ref.complex_conjugated()) {
                        (false, false) => cw.t().dot(sao_h).dot(cx),
                        (true, false) => cw.t().dot(sao).dot(cx),
                        (false, true) => cx.t().dot(sao).dot(cw),
                        (true, true) => cw.t().dot(&sao_h.t()).dot(cx),
                    }
                } else {
                    match (s_ref.complex_conjugated(), o_ref.complex_conjugated()) {
                        (false, false) => cw.t().mapv(|x| x.conj()).dot(sao).dot(cx),
                        (true, false) => cw.t().mapv(|x| x.conj()).dot(sao_h).dot(cx),
                        (false, true) => cx
                            .t()
                            .mapv(|x| x.conj())
                            .dot(sao_h)
                            .dot(cw)
                            .mapv(|x| x.conj()),
                        (true, true) => cw.t().mapv(|x| x.conj()).dot(&sao.t()).dot(cx),
                    }
                }
            })
            .collect_vec();

        let ovs_vec = self
            .occupation_patterns
            .iter()
            .cartesian_product(other.occupation_patterns.iter())
            .map(|(s_occs, o_occs)| {
                let ov = izip!(s_occs, o_occs, &mo_ov_mats)
                    .map(|(s_occ, o_occ, mo_ov_mat)| {
                        let nonzero_s_occ =
                            s_occ.iter().positions(|&occ| occ > thresh).collect_vec();
                        let nonzero_o_occ =
                            o_occ.iter().positions(|&occ| occ > thresh).collect_vec();
                        let mo_ov_mat_occ = mo_ov_mat
                            .select(Axis(0), &nonzero_s_occ)
                            .select(Axis(1), &nonzero_o_occ);
                        mo_ov_mat_occ
                            .det()
                            .expect("The determinant of the MO overlap matrix could not be found.")
                    })
                    .fold(T::one(), |acc, x| acc * x);

                let implicit_factor = s_ref.structure_constraint().implicit_factor()?;
                if implicit_factor > 1 {
                    let p_i32 = i32::try_from(implicit_factor)?;
                    Ok(ComplexFloat::powi(ov, p_i32))
                } else {
                    Ok(ov)
                }
            })
            .collect::<Result<Vec<_>, anyhow::Error>>()?;

        Ok(Array2::from_shape_vec(
            (self.n_items(), other.n_items()),
            ovs_vec,
        )?)
    }
}

impl<'a, T, SC> Basis<SlaterDeterminant<'a, T, SC>> for FCIBasis<'a, T, SC>
where
    T: ComplexFloat + Lapack,
    SC: 'a + StructureConstraint + fmt::Display + Clone,
{
    type BasisIter<'b>
        = FCIBasisIterator<'b, 'a, T, SC>
    where
        Self: 'b;

    fn n_items(&self) -> usize {
        self.occupation_patterns.len()
    }

    /// Iterates over the elements of the [`FCIBasis`]. Each element is a Slater determinant
    /// obtained by replacing the occupation pattern in the reference with the corresponding one
    /// from the [`Self::occupation_patterns`] list.
    fn iter(&self) -> Self::BasisIter<'_> {
        FCIBasisIterator::new(&self)
    }

    fn first(&self) -> Option<SlaterDeterminant<'a, T, SC>> {
        Some(self.reference.clone())
    }
}

// ~~~~~~~~~~~~~~~~~~
// FCIBasis interator
// ~~~~~~~~~~~~~~~~~~

/// Lazy iterator for full-configuration-interaction basis.
#[derive(Builder, Clone)]
pub struct FCIBasisIterator<'b, 'a, T, SC>
where
    T: ComplexFloat + Lapack,
    SC: StructureConstraint + fmt::Display,
{
    /// The associated full-configuration-interaction basis.
    fci_basis: &'b FCIBasis<'a, T, SC>,

    /// The current index of the occupation patterns.
    current_occupation_index: usize,
}

impl<'b, 'a, T, SC> FCIBasisIterator<'b, 'a, T, SC>
where
    T: ComplexFloat + Lapack,
    SC: StructureConstraint + fmt::Display,
{
    /// Creates and returns a new full-configuration-interaction basis iterator.
    ///
    /// # Arguments
    ///
    /// * `fci_basis` - The associated full-configuration-interaction basis.
    ///
    /// # Returns
    ///
    /// An full-configuration-interaction basis iterator.
    fn new(fci_basis: &'b FCIBasis<'a, T, SC>) -> Self {
        Self {
            fci_basis,
            current_occupation_index: 0,
        }
    }
}

impl<'b, 'a, T, SC> Iterator for FCIBasisIterator<'b, 'a, T, SC>
where
    T: ComplexFloat + Lapack,
    SC: StructureConstraint + fmt::Display + Clone,
{
    type Item = Result<SlaterDeterminant<'a, T, SC>, anyhow::Error>;

    fn next(&mut self) -> Option<Self::Item> {
        let reference = &self.fci_basis.reference;
        let index = self.current_occupation_index;
        self.current_occupation_index += 1;
        if index >= self.fci_basis.occupation_patterns.len() {
            None
        } else {
            let det = SlaterDeterminant::builder()
                .structure_constraint(reference.structure_constraint().clone())
                .baos(reference.baos().clone())
                .complex_symmetric(reference.complex_symmetric())
                .complex_conjugated(reference.complex_conjugated())
                .mol(reference.mol())
                .coefficients(reference.coefficients())
                .occupations(&self.fci_basis.occupation_patterns.get(index)?)
                .mo_energies(reference.mo_energies().cloned())
                .threshold(reference.threshold())
                .build()
                .map_err(|err| format_err!(err));
            Some(det)
        }
    }
}
