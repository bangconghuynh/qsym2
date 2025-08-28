use anyhow::{self, format_err};
use ndarray::{Array2, Ix2, ScalarOperand};
use ndarray_einsum::einsum;
use num_complex::ComplexFloat;

use super::nonortho::LowdinPairedCoefficients;

/// Calculates the unweighted codensity matrix between two Löwdin-paired spin-orbitals $`i`$ in
/// determinants $`^{w}\Psi`$ and $`^{x}\Psi`$.
///
/// The unweighted codensity matrix between the above two spin-orbitals is given by
/// ```math
///     ^{wx}\mathbf{P}_i
///       = \ ^{x}\mathbf{C}_i \otimes
///         (^{w}\mathbf{G}_i^{\star})^{\lozenge}
///       = \ ^{x}\mathbf{C}_i
///         \ ^{w}\mathbf{C}_i^{\dagger\lozenge},
/// ```
/// where $`\mathbf{C}_i`$ is the coefficient column vector for spin-orbital $`\chi_i`$ such that
///
/// ```math
///     \chi_i(\mathbf{x}) = \sum_{\mu} \psi_{\mu}(\mathbf{x}) C_{\mu i}.
/// ```
///
/// # Arguments
///
/// * `lowdin_paired_coefficients` - Structure containing the Löwdin-paired coefficient matrices.
/// * `i` - Index of the corresponding Löwdin-paired spin-orbitals for which the unweighted
/// codensity matrix is to be computed.
///
/// # Returns
///
/// The unweighted codensity matrix $`^{wx}\mathbf{P}_i`$.
pub fn calc_unweighted_codensity_matrix<T>(
    lowdin_paired_coefficients: &LowdinPairedCoefficients<T>,
    i: usize,
) -> Result<Array2<T>, anyhow::Error>
where
    T: ComplexFloat + 'static,
{
    let (cwt, cxt) = lowdin_paired_coefficients.paired_coefficients();
    let cwi = &cwt.column(i);
    let cxi = &cxt.column(i);
    if cwi.shape() != cxi.shape() {
        Err(format_err!(
            "Coefficient dimensions mismatched: cwi ({:?}) !~ cxi ({:?}).",
            cwi.shape(),
            cxi.shape()
        ))
    } else if lowdin_paired_coefficients.complex_symmetric() {
        einsum("m,n->mn", &[cxi, cwi])
            .map_err(|err| format_err!(err))
            .and_then(|p| {
                p.into_dimensionality::<Ix2>()
                    .map_err(|err| format_err!(err))
            })
    } else {
        einsum("m,n->mn", &[cxi, &cwi.map(|x| x.conj())])
            .map_err(|err| format_err!(err))
            .and_then(|p| {
                p.into_dimensionality::<Ix2>()
                    .map_err(|err| format_err!(err))
            })
    }
}

/// Calculates the weighted codensity matrix between a set of Löwdin-paired spin-orbitals in
/// determinants $`^{w}\Psi`$ and $`^{x}\Psi`$.
///
/// The weighted codensity matrix described above is given by
/// ```math
///     ^{wx}\mathbf{W} =
///         \sum_{\substack{i = 1\\ ^{wx}\lambda_i \neq 0}}^{N_{\mathrm{e}}}
///             \frac{^{wx}\mathbf{P}_i}{^{wx}\lambda_i}
/// ```
/// where $`^{wx}\mathbf{P}_i`$ is the unweighted codensity matrix between Löwdin-paired spin-orbitals
/// $`i`$ in determinants $`^{w}\Psi`$ and $`^{x}\Psi`$, and the sum runs over up to
/// :math:`N_{\mathrm{e}}` pairs of Löwdin-paired spin-orbitals of consideration, excluding those
/// that have zero Löwdin overlaps.
///
/// # Arguments
///
/// * `lowdin_paired_coefficients` - Structure containing the Löwdin-paired coefficient matrices.
///
/// # Returns
///
/// The weighted codensity matrix $`^{wx}\mathbf{W}`$.
pub fn calc_weighted_codensity_matrix<T>(
    lowdin_paired_coefficients: &LowdinPairedCoefficients<T>,
) -> Result<Array2<T>, anyhow::Error>
where
    T: ComplexFloat + ScalarOperand,
{
    let nbasis = lowdin_paired_coefficients.nbasis();
    lowdin_paired_coefficients.nonzero_indices().iter().fold(
        Ok::<_, anyhow::Error>(Array2::<T>::zeros((nbasis, nbasis))),
        |acc_res, &i| {
            acc_res.and_then(|acc| {
                Ok(acc
                    + calc_unweighted_codensity_matrix(lowdin_paired_coefficients, i)?
                        / *lowdin_paired_coefficients
                            .lowdin_overlaps()
                            .get(i)
                            .ok_or_else(|| {
                                format_err!("Unable to retrieve the Löwdin overlap with index {i}.")
                            })?)
            })
        },
    )
}
