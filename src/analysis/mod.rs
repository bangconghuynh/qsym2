//! Symmetry analysis via representation and corepresentation theories.

use std::cmp::Ordering;
use std::fmt;

use anyhow::{self, format_err, Context};
use itertools::Itertools;
use log;
use ndarray::{s, Array, Array1, Array2, Axis, Dimension, Ix0, Ix2};
use ndarray_einsum_beta::*;
use ndarray_linalg::{solve::Inverse, types::Lapack};
use num_complex::ComplexFloat;
use num_traits::ToPrimitive;
#[cfg(feature = "python")]
use pyo3::prelude::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::chartab::chartab_group::CharacterProperties;
use crate::chartab::{DecompositionError, SubspaceDecomposable};
use crate::group::{class::ClassProperties, GroupProperties};
use crate::io::format::{log_subtitle, qsym2_output};

// =======
// Overlap
// =======

// ----------------
// Trait definition
// ----------------

/// Trait for computing the inner product
/// $`\langle \hat{\iota} \mathbf{v}_i, \mathbf{v}_j \rangle`$ between two linear-space quantities
/// $`\mathbf{v}_i`$ and $`\mathbf{v}_j`$. The involutory operator $`\hat{\iota}`$ determines
/// whether the inner product is a sesquilinear form or a bilinear form.
pub trait Overlap<T, D>
where
    T: ComplexFloat + fmt::Debug + Lapack,
    D: Dimension,
{
    /// If `true`, the inner product is bilinear and $`\hat{\iota} = \hat{\kappa}`$. If `false`,
    /// the inner product is sesquilinear and $`\hat{\iota} = \mathrm{id}`$.
    fn complex_symmetric(&self) -> bool;

    /// Returns the overlap between `self` and `other`, with respect to a metric `metric` of the
    /// underlying basis in which `self` and `other` are expressed.
    fn overlap(&self, other: &Self, metric: Option<&Array<T, D>>) -> Result<T, anyhow::Error>;
}

// =====
// Orbit
// =====

// --------------------------------------
// Struct definitions and implementations
// --------------------------------------

/// Lazy iterator for orbits generated by the action of a group on an origin.
pub struct OrbitIterator<'a, G, I>
where
    G: GroupProperties,
{
    /// A mutable iterator over the elements of the group. Each element will be applied on the
    /// origin to yield a corresponding item in the orbit.
    group_iter: <<G as GroupProperties>::ElementCollection as IntoIterator>::IntoIter,

    /// The origin of the orbit.
    origin: &'a I,

    /// A function defining the action of each group element on the origin.
    action: fn(&G::GroupElement, &I) -> Result<I, anyhow::Error>,
}

impl<'a, G, I> OrbitIterator<'a, G, I>
where
    G: GroupProperties,
{
    /// Creates and returns a new orbit iterator.
    ///
    /// # Arguments
    ///
    /// * `group` - A group.
    /// * `origin` - An origin.
    /// * `action` - A function or closure defining the action of each group element on the origin.
    ///
    /// # Returns
    ///
    /// An orbit iterator.
    pub fn new(
        group: &G,
        origin: &'a I,
        action: fn(&G::GroupElement, &I) -> Result<I, anyhow::Error>,
    ) -> Self {
        Self {
            group_iter: group.elements().clone().into_iter(),
            origin,
            action,
        }
    }
}

impl<'a, G, I> Iterator for OrbitIterator<'a, G, I>
where
    G: GroupProperties,
{
    type Item = Result<I, anyhow::Error>;

    fn next(&mut self) -> Option<Self::Item> {
        self.group_iter
            .next()
            .map(|op| (self.action)(&op, self.origin))
    }
}

// ----------------
// Trait definition
// ----------------

/// Trait for orbits arising from group actions.
pub trait Orbit<G, I>
where
    G: GroupProperties,
{
    /// Type of the iterator over items in the orbit.
    type OrbitIter: Iterator<Item = Result<I, anyhow::Error>>;

    /// The group generating the orbit.
    fn group(&self) -> &G;

    /// The origin of the orbit.
    fn origin(&self) -> &I;

    /// An iterator over items in the orbit arising from the action of the group on the origin.
    fn iter(&self) -> Self::OrbitIter;
}

// ========
// Analysis
// ========

// ---------------
// Enum definition
// ---------------

/// Enumerated type specifying the comparison mode for filtering out orbit overlap
/// eigenvalues.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "python", pyclass)]
pub enum EigenvalueComparisonMode {
    /// Compares the eigenvalues using only their real parts.
    Real,

    /// Compares the eigenvalues using their moduli.
    Modulus,
}

impl Default for EigenvalueComparisonMode {
    fn default() -> Self {
        Self::Modulus
    }
}

impl fmt::Display for EigenvalueComparisonMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EigenvalueComparisonMode::Real => write!(f, "Real part"),
            EigenvalueComparisonMode::Modulus => write!(f, "Modulus"),
        }
    }
}

// ----------------
// Trait definition
// ----------------

/// Trait for representation or corepresentation analysis on an orbit of items spanning a
/// linear space.
pub trait RepAnalysis<G, I, T, D>: Orbit<G, I>
where
    T: ComplexFloat + Lapack + fmt::Debug + Send + Sync,
    <T as ComplexFloat>::Real: ToPrimitive,
    G: GroupProperties + ClassProperties + CharacterProperties,
    G::GroupElement: fmt::Display,
    G::CharTab: SubspaceDecomposable<T>,
    D: Dimension,
    I: Overlap<T, D> + Clone + Send + Sync,
    Self::OrbitIter: Iterator<Item = Result<I, anyhow::Error>> + Send,
{
    // ----------------
    // Required methods
    // ----------------

    /// Sets the overlap matrix between the items in the orbit.
    ///
    /// # Arguments
    ///
    /// * `smat` - The overlap matrix between the items in the orbit.
    fn set_smat(&mut self, smat: Array2<T>);

    /// Returns the overlap matrix between the items in the orbit.
    #[must_use]
    fn smat(&self) -> Option<&Array2<T>>;

    /// Returns the transformation matrix $`\mathbf{X}`$ for the overlap matrix $`\mathbf{S}`$
    /// between the items in the orbit.
    ///
    /// The matrix $`\mathbf{X}`$ serves to bring $`\mathbf{S}`$ to full rank, *i.e.*, the matrix
    /// $`\tilde{\mathbf{S}}`$ defined by
    ///
    /// ```math
    ///     \tilde{\mathbf{S}} = \mathbf{X}^{\dagger\lozenge} \mathbf{S} \mathbf{X}
    /// ```
    ///
    /// is a full-rank matrix.
    ///
    /// If the overlap between items is complex-symmetric (see [`Overlap::complex_symmetric`]), then
    /// $`\lozenge = *`$ is the complex-conjugation operation, otherwise, $`\lozenge`$ is the
    /// identity.
    ///
    /// Depending on how $`\mathbf{X}`$ has been computed, $`\tilde{\mathbf{S}}`$ might also be
    /// orthogonal. Either way, $`\tilde{\mathbf{S}}`$ is always guaranteed to be of full-rank.
    #[must_use]
    fn xmat(&self) -> &Array2<T>;

    /// Returns the norm-preserving scalar map $`f`$ for every element of the generating group
    /// defined by
    ///
    /// ```math
    ///     \langle \hat{\iota} \mathbf{v}_w, \hat{g}_i \mathbf{v}_x \rangle
    ///     = f \left( \langle \hat{\iota} \hat{g}_i^{-1} \mathbf{v}_w, \mathbf{v}_x \rangle \right).
    /// ```
    ///
    /// Typically, if $`\hat{g}_i`$ is unitary, then $`f`$ is the identity, and if $`\hat{g}_i`$ is
    /// antiunitary, then $`f`$ is the complex-conjugation operation. Either way, the norm of the
    /// inner product is preserved.
    #[must_use]
    fn norm_preserving_scalar_map(&self, i: usize) -> fn(T) -> T;

    /// Returns the threshold for integrality checks of irreducible representation or
    /// corepresentation multiplicities.
    #[must_use]
    fn integrality_threshold(&self) -> <T as ComplexFloat>::Real;

    /// Returns the enumerated type specifying the comparison mode for filtering out orbit overlap
    /// eigenvalues.
    #[must_use]
    fn eigenvalue_comparison_mode(&self) -> &EigenvalueComparisonMode;

    // ----------------
    // Provided methods
    // ----------------

    /// Calculates and stores the overlap matrix between items in the orbit, with respect to a
    /// metric of the basis in which these items are expressed.
    ///
    /// # Arguments
    ///
    /// * `metric` - The metric of the basis in which the orbit items are expressed.
    fn calc_smat(&mut self, metric: Option<&Array<T, D>>) -> Result<&mut Self, anyhow::Error> {
        let order = self.group().order();
        let mut smat = Array2::<T>::zeros((order, order));
        let item_0 = self.origin();
        if let Some(ctb) = self.group().cayley_table() {
            log::debug!("Cayley table available. Group closure will be used to speed up overlap matrix computation.");
            let ovs = self
                .iter()
                .par_bridge()
                .map(|item_res| {
                    let item = item_res?;
                    item.overlap(item_0, metric)
                })
                .collect::<Result<Vec<_>, _>>()?;
            for (i, j) in (0..order).cartesian_product(0..order) {
                let jinv = ctb
                    .slice(s![.., j])
                    .iter()
                    .position(|&x| x == 0)
                    .ok_or(format_err!(
                        "Unable to find the inverse of group element `{j}`."
                    ))?;
                let jinv_i = ctb[(jinv, i)];
                smat[(i, j)] = self.norm_preserving_scalar_map(jinv)(ovs[jinv_i]);
            }
        } else {
            log::debug!("Cayley table not available. Overlap matrix will be constructed without group-closure speed-up.");
            for pair in self
                .iter()
                .map(|item_res| item_res.map_err(|err| err.to_string()))
                .enumerate()
                .combinations_with_replacement(2)
            {
                let (w, item_w_res) = &pair[0];
                let (x, item_x_res) = &pair[1];
                let item_w = item_w_res
                    .as_ref()
                    .map_err(|err| format_err!(err.clone()))
                    .with_context(|| "One of the items in the orbit is not available")?;
                let item_x = item_x_res
                    .as_ref()
                    .map_err(|err| format_err!(err.clone()))
                    .with_context(|| "One of the items in the orbit is not available")?;
                smat[(*w, *x)] = item_w.overlap(item_x, metric).map_err(|err| {
                    log::error!("{err}");
                    log::error!(
                        "Unable to calculate the overlap between items `{w}` and `{x}` in the orbit."
                    );
                    err
                })?;
                if *w != *x {
                    smat[(*x, *w)] = item_x.overlap(item_w, metric).map_err(|err| {
                            log::error!("{err}");
                            log::error!(
                                "Unable to calculate the overlap between items `{x}` and `{w}` in the orbit."
                            );
                            err
                        })?;
                }
            }
        }
        if self.origin().complex_symmetric() {
            self.set_smat((smat.clone() + smat.t().to_owned()).mapv(|x| x / (T::one() + T::one())))
        } else {
            self.set_smat(
                (smat.clone() + smat.t().to_owned().mapv(|x| x.conj()))
                    .mapv(|x| x / (T::one() + T::one())),
            )
        }
        Ok(self)
    }

    /// Normalises overlap matrix between items in the orbit such that its diagonal entries are
    /// unity.
    ///
    /// # Errors
    ///
    /// Errors if no orbit overlap matrix can be found, of if linear-algebraic errors are
    /// encountered.
    fn normalise_smat(&mut self) -> Result<&mut Self, anyhow::Error> {
        let smat = self
            .smat()
            .ok_or(format_err!("No orbit overlap matrix to normalise."))?;
        let norm = smat.diag().mapv(|x| <T as ComplexFloat>::sqrt(x));
        let nspatial = norm.len();
        let norm_col = norm
            .clone()
            .into_shape([nspatial, 1])
            .map_err(|err| format_err!(err))?;
        let norm_row = norm
            .into_shape([1, nspatial])
            .map_err(|err| format_err!(err))?;
        let norm_mat = norm_col.dot(&norm_row);
        let normalised_smat = smat / norm_mat;
        self.set_smat(normalised_smat);
        Ok(self)
    }

    /// Computes the $`\mathbf{T}(g)`$ matrix for a particular element $`g`$ of the generating
    /// group.
    ///
    /// The elements of this matrix are given by
    ///
    /// ```math
    ///     T_{wx}(g)
    ///         = \langle \hat{\iota} \hat{g}_w \mathbf{v}_0, \hat{g} \hat{g}_x \mathbf{v}_0 \rangle.
    /// ```
    ///
    /// This means that $`\mathbf{T}(g)`$ is just the orbit overlap matrix $`\mathbf{S}`$ with its
    /// columns permuted according to the way $`g`$ composites on the elements in the group from
    /// the left.
    ///
    /// # Arguments
    ///
    /// * `op` - The element $`g`$ in the generating group.
    ///
    /// # Returns
    ///
    /// The matrix $`\mathbf{T}(g)`$.
    #[must_use]
    fn calc_tmat(&self, op: &G::GroupElement) -> Result<Array2<T>, anyhow::Error> {
        let ctb = self
            .group()
            .cayley_table()
            .expect("The Cayley table for the group cannot be found.");
        let i = self.group().get_index_of(op).unwrap_or_else(|| {
            panic!("Unable to retrieve the index of element `{op}` in the group.")
        });
        let ix = ctb.slice(s![i, ..]).iter().cloned().collect::<Vec<_>>();
        let twx = self
            .smat()
            .ok_or(format_err!("No orbit overlap matrix found."))?
            .select(Axis(1), &ix);
        Ok(twx)
    }

    /// Computes the representation or corepresentation matrix $`\mathbf{D}(g)`$ for a particular
    /// element $`g`$ in the generating group in the basis of the orbit.
    ///
    /// The matrix $`\mathbf{D}(g)`$ is defined by
    ///
    /// ```math
    ///     \hat{g} \mathcal{G} \cdot \mathbf{v}_0 = \mathcal{G} \cdot \mathbf{v}_0 \mathbf{D}(g),
    /// ```
    ///
    /// where $`\mathcal{G} \cdot \mathbf{v}_0`$ is the orbit generated by the action of the group
    /// $`\mathcal{G}`$ on the origin $`\mathbf{v}_0`$.
    ///
    /// # Arguments
    ///
    /// * `op` - The element $`g`$ of the generating group.
    ///
    /// # Returns
    ///
    /// The matrix $`\mathbf{D}(g)`$.
    #[must_use]
    fn calc_dmat(&self, op: &G::GroupElement) -> Result<Array2<T>, anyhow::Error> {
        let complex_symmetric = self.origin().complex_symmetric();
        let xmath = if complex_symmetric {
            self.xmat().t().to_owned()
        } else {
            self.xmat().t().mapv(|x| x.conj())
        };
        let smattilde = xmath
            .dot(
                self.smat()
                    .ok_or(format_err!("No orbit overlap matrix found."))?,
            )
            .dot(self.xmat());
        let smattilde_inv = smattilde
            .inv()
            .expect("The inverse of S~ could not be found.");
        let dmat = einsum(
            "ij,jk,kl,lm->im",
            &[&smattilde_inv, &xmath, &self.calc_tmat(op)?, self.xmat()],
        )
        .map_err(|err| format_err!(err))
        .with_context(|| "Unable to compute the matrix product [(S~)^(-1) X† T X].")?
        .into_dimensionality::<Ix2>()
        .map_err(|err| format_err!(err))
        .with_context(|| {
            "Unable to convert the matrix product [(S~)^(-1) X† T X] to two dimensions."
        });
        dmat
    }

    /// Computes the character of a particular element $`g`$ in the generating group in the basis
    /// of the orbit.
    ///
    /// See [`Self::calc_dmat`] for more information.
    ///
    /// # Arguments
    ///
    /// * `op` - The element $`g`$ of the generating group.
    ///
    /// # Returns
    ///
    /// The character $`\chi(g)`$.
    #[must_use]
    fn calc_character(&self, op: &G::GroupElement) -> Result<T, anyhow::Error> {
        let complex_symmetric = self.origin().complex_symmetric();
        let xmath = if complex_symmetric {
            self.xmat().t().to_owned()
        } else {
            self.xmat().t().mapv(|x| x.conj())
        };
        let smattilde = xmath
            .dot(
                self.smat()
                    .ok_or(format_err!("No orbit overlap matrix found."))?,
            )
            .dot(self.xmat());
        let smattilde_inv = smattilde
            .inv()
            .expect("The inverse of S~ could not be found.");
        let chi = einsum(
            "ij,jk,kl,li",
            &[&smattilde_inv, &xmath, &self.calc_tmat(op)?, self.xmat()],
        )
        .map_err(|err| format_err!(err))
        .with_context(|| "Unable to compute the trace of the matrix product [(S~)^(-1) X† T X].")?
        .into_dimensionality::<Ix0>()
        .map_err(|err| format_err!(err))
        .with_context(|| "Unable to convert the trace of the matrix product [(S~)^(-1) X† T X] to zero dimensions.")?;
        chi.into_iter().next().ok_or(format_err!(
            "Unable to extract the character from the representation matrix."
        ))
    }

    /// Computes the characters of the elements in a conjugacy-class transversal of the generating
    /// group in the basis of the orbit.
    ///
    /// See [`Self::calc_dmat`]  and [`Self::calc_character`] for more information.
    ///
    /// # Returns
    ///
    /// The conjugacy class symbols and the corresponding characters.
    #[must_use]
    fn calc_characters(
        &self,
    ) -> Result<Vec<(<G as ClassProperties>::ClassSymbol, T)>, anyhow::Error> {
        let complex_symmetric = self.origin().complex_symmetric();
        let xmath = if complex_symmetric {
            self.xmat().t().to_owned()
        } else {
            self.xmat().t().mapv(|x| x.conj())
        };
        let smattilde = xmath
            .dot(
                self.smat()
                    .ok_or(format_err!("No orbit overlap matrix found."))?,
            )
            .dot(self.xmat());
        let smattilde_inv = smattilde
            .inv()
            .expect("The inverse of S~ could not be found.");
        let chis = (0..self.group().class_number()).map(|cc_i| {
            let cc = self.group().get_cc_symbol_of_index(cc_i).unwrap();
            let op = self.group().get_cc_transversal(cc_i).unwrap();
            let chi = einsum(
                "ij,jk,kl,li",
                &[&smattilde_inv, &xmath, &self.calc_tmat(&op)?, self.xmat()],
            )
            .map_err(|err| format_err!(err))
            .with_context(|| "Unable to compute the trace of the matrix product [(S~)^(-1) X† T X].")?
            .into_dimensionality::<Ix0>()
            .map_err(|err| format_err!(err))
            .with_context(|| "Unable to convert the trace of the matrix product [(S~)^(-1) X† T X] to zero dimensions.")?;
            let chi_val = chi.into_iter().next().ok_or(format_err!(
                "Unable to extract the character from the representation matrix."
            ))?;
            Ok((cc, chi_val))
        }).collect::<Result<Vec<_>, _>>();
        chis
    }

    /// Reduces the representation or corepresentation spanned by the items in the orbit to a
    /// direct sum of the irreducible representations or corepresentations of the generating group.
    ///
    /// # Returns
    ///
    /// The decomposed result.
    ///
    /// # Errors
    ///
    /// Errors if the decomposition fails, *e.g.* because one or more calculated multiplicities
    /// are non-integral.
    fn analyse_rep(
        &self,
    ) -> Result<
        <<G as CharacterProperties>::CharTab as SubspaceDecomposable<T>>::Decomposition,
        DecompositionError,
    > {
        let chis = self
            .calc_characters()
            .map_err(|err| DecompositionError(err.to_string()))?;
        let res = self.group().character_table().reduce_characters(
            &chis.iter().map(|(cc, chi)| (cc, *chi)).collect::<Vec<_>>(),
            self.integrality_threshold(),
        );
        res
    }
}

// =================
// Macro definitions
// =================

macro_rules! fn_calc_xmat_real {
    ( $(#[$meta:meta])* $vis:vis $func:ident ) => {
        $(#[$meta])*
        $vis fn $func(&mut self, preserves_full_rank: bool) -> Result<&mut Self, anyhow::Error> {
            // Real, symmetric S
            let thresh = self.linear_independence_threshold;
            let smat = self
                .smat
                .as_ref()
                .ok_or(format_err!("No overlap matrix found for this orbit."))?;
            use ndarray_linalg::norm::Norm;
            if (smat.to_owned() - smat.t()).norm_l2() > thresh {
                Err(format_err!("Overlap matrix is not symmetric."))
            } else {
                let (s_eig, umat) = smat.eigh(UPLO::Lower).map_err(|err| format_err!(err))?;
                let nonzero_s_indices = match self.eigenvalue_comparison_mode {
                    EigenvalueComparisonMode::Modulus => {
                        s_eig.iter().positions(|x| x.abs() > thresh).collect_vec()
                    }
                    EigenvalueComparisonMode::Real => {
                        s_eig.iter().positions(|x| *x > thresh).collect_vec()
                    }
                };
                let nonzero_s_eig = s_eig.select(Axis(0), &nonzero_s_indices);
                let nonzero_umat = umat.select(Axis(1), &nonzero_s_indices);
                let nullity = smat.shape()[0] - nonzero_s_indices.len();
                let xmat = if nullity == 0 && preserves_full_rank {
                    Array2::eye(smat.shape()[0])
                } else {
                    let s_s = Array2::<f64>::from_diag(&nonzero_s_eig.mapv(|x| 1.0 / x.sqrt()));
                    nonzero_umat.dot(&s_s)
                };
                self.smat_eigvals = Some(s_eig);
                self.xmat = Some(xmat);
                Ok(self)
            }
        }
    }
}

macro_rules! fn_calc_xmat_complex {
    ( $(#[$meta:meta])* $vis:vis $func:ident ) => {
        $(#[$meta])*
        $vis fn $func(&mut self, preserves_full_rank: bool) -> Result<&mut Self, anyhow::Error> {
            // Complex S, symmetric or Hermitian
            let thresh = self.linear_independence_threshold;
            let smat = self
                .smat
                .as_ref()
                .ok_or(format_err!("No overlap matrix found for this orbit."))?;
            let (s_eig, umat_nonortho) = smat.eig().map_err(|err| format_err!(err))?;

            let nonzero_s_indices = match self.eigenvalue_comparison_mode {
                EigenvalueComparisonMode::Modulus => s_eig
                    .iter()
                    .positions(|x| ComplexFloat::abs(*x) > thresh)
                    .collect_vec(),
                EigenvalueComparisonMode::Real => {
                    if s_eig
                        .iter()
                        .any(|x| Float::abs(ComplexFloat::im(*x)) > thresh)
                    {
                        log::warn!("Comparing eigenvalues using the real parts, but not all eigenvalues are real.");
                    }
                    s_eig
                        .iter()
                        .positions(|x| ComplexFloat::re(*x) > thresh)
                        .collect_vec()
                }
            };
            let nonzero_s_eig = s_eig.select(Axis(0), &nonzero_s_indices);
            let nonzero_umat_nonortho = umat_nonortho.select(Axis(1), &nonzero_s_indices);

            // `eig` does not guarantee orthogonality of `nonzero_umat_nonortho`.
            // Gram--Schmidt is therefore required.
            let nonzero_umat = complex_modified_gram_schmidt(
                &nonzero_umat_nonortho,
                self.origin.complex_symmetric(),
                thresh,
            )
            .map_err(
                |_| format_err!("Unable to orthonormalise the linearly-independent eigenvectors of the overlap matrix.")
            )?;

            let nullity = smat.shape()[0] - nonzero_s_indices.len();
            let xmat = if nullity == 0 && preserves_full_rank {
                Array2::<Complex<T>>::eye(smat.shape()[0])
            } else {
                let s_s = Array2::<Complex<T>>::from_diag(
                    &nonzero_s_eig.mapv(|x| Complex::<T>::from(T::one()) / x.sqrt()),
                );
                nonzero_umat.dot(&s_s)
            };
            self.smat_eigvals = Some(s_eig);
            self.xmat = Some(xmat);
            Ok(self)
        }
    }
}

pub(crate) use fn_calc_xmat_complex;
pub(crate) use fn_calc_xmat_real;

// =================
// Utility functions
// =================

/// Logs overlap eigenvalues nicely and indicates where the threshold has been crossed.
///
/// # Arguments
///
/// * `eigvals` - The eigenvalues.
/// * `thresh` - The cut-off threshold to be marked out.
/// * `thresh_cmp` - The function for comparing with threshold. The threshold is marked out when
/// the function first evaluates to [`Ordering::Less`].
pub(crate) fn log_overlap_eigenvalues<T>(
    title: &str,
    eigvals: &Array1<T>,
    thresh: <T as ComplexFloat>::Real,
    eigenvalue_comparison_mode: &EigenvalueComparisonMode,
    // thresh_cmp: fn(&T, &<T as ComplexFloat>::Real) -> Ordering,
) where
    T: std::fmt::LowerExp + ComplexFloat,
    <T as ComplexFloat>::Real: std::fmt::LowerExp,
{
    let mut eigvals_sorted = eigvals.iter().collect::<Vec<_>>();
    match eigenvalue_comparison_mode {
        EigenvalueComparisonMode::Modulus => {
            eigvals_sorted.sort_by(|a, b| a.abs().partial_cmp(&b.abs()).unwrap());
        }
        EigenvalueComparisonMode::Real => {
            eigvals_sorted.sort_by(|a, b| a.re().partial_cmp(&b.re()).unwrap());
        }
    }
    eigvals_sorted.reverse();
    let eigvals_str = eigvals_sorted
        .iter()
        .map(|v| format!("{v:+.3e}"))
        .collect::<Vec<_>>();
    log_subtitle(title);
    qsym2_output!("");

    match eigenvalue_comparison_mode {
        EigenvalueComparisonMode::Modulus => {
            qsym2_output!("Eigenvalues are sorted in decreasing order of their moduli.");
        }
        EigenvalueComparisonMode::Real => {
            qsym2_output!("Eigenvalues are sorted in decreasing order of their real parts.");
        }
    }
    let count_length = usize::try_from(eigvals.len().ilog10() + 2).unwrap_or(2);
    let eigval_length = eigvals_str
        .iter()
        .map(|v| v.chars().count())
        .max()
        .unwrap_or(20);
    qsym2_output!("{}", "┈".repeat(count_length + 3 + eigval_length));
    qsym2_output!("{:>count_length$}  Eigenvalue", "#");
    qsym2_output!("{}", "┈".repeat(count_length + 3 + eigval_length));
    let mut write_thresh = false;
    for (i, eigval) in eigvals_str.iter().enumerate() {
        let cmp = match eigenvalue_comparison_mode {
            EigenvalueComparisonMode::Modulus => {
                eigvals_sorted[i].abs().partial_cmp(&thresh).expect(
                    "Unable to compare the modulus of an eigenvalue with the specified threshold.",
                )
            }
            EigenvalueComparisonMode::Real => eigvals_sorted[i].re().partial_cmp(&thresh).expect(
                "Unable to compare the real part of an eigenvalue with the specified threshold.",
            ),
        };
        if cmp == Ordering::Less && !write_thresh {
            let comparison_mode_str = match eigenvalue_comparison_mode {
                EigenvalueComparisonMode::Modulus => "modulus-based",
                EigenvalueComparisonMode::Real => "real-part-based",
            };
            qsym2_output!(
                "{} <-- linear independence threshold ({comparison_mode_str}): {:+.3e}",
                "-".repeat(count_length + 3 + eigval_length),
                thresh
            );
            write_thresh = true;
        }
        qsym2_output!("{i:>count_length$}  {eigval}",);
    }
    qsym2_output!("{}", "┈".repeat(count_length + 3 + eigval_length));
}
