//! Implementation of symmetry projection for electron densities.

use anyhow::{self, format_err};
use ndarray::Array2;
use num::{Complex, ToPrimitive};

use super::density_analysis::DensitySymmetryOrbit;
use crate::analysis::Orbit;
use crate::chartab::chartab_symbols::LinearSpaceSymbol;
use crate::projection::Projectable;
use crate::symmetry::symmetry_group::SymmetryGroupProperties;
use crate::target::density::Density;

impl<'a, G> Projectable<G, Density<'a, f64>> for DensitySymmetryOrbit<'a, G, f64>
where
    G: SymmetryGroupProperties,
{
    type Projected<'p>
        = Result<Density<'p, f64>, anyhow::Error>
    where
        Self: 'p;

    fn project_onto(&self, row: &G::RowSymbol) -> Self::Projected<'_> {
        let group_order = self
            .group()
            .order()
            .to_f64()
            .ok_or_else(|| format_err!("Unable to convert the group order to `f64`."))?;
        let dim_f64 = row
            .dimensionality()
            .to_f64()
            .ok_or_else(|| format_err!("Unable to convert the degeneracy to `f64`."))?;
        let denmat_res = self.generate_orbit_algebra_terms(row).fold(
            Ok(Array2::<f64>::zeros(self.origin().density_matrix().raw_dim())),
            |acc_res, item_res| acc_res.and_then(|acc| item_res.and_then(|(chr, den)| {
                let chr_complex = chr.complex_conjugate().complex_value();
                if chr_complex.im > self.origin().threshold() {
                    Err(format_err!("Complex characters encountered. Density projection fails over the reals."))
                } else {
                    Ok(acc + dim_f64 * chr_complex.re / group_order * den.density_matrix())
                }
            })),
        );
        denmat_res.and_then(|denmat| {
            Density::builder()
                .bao(self.origin().bao)
                .complex_symmetric(self.origin().complex_symmetric)
                .complex_conjugated(self.origin().complex_conjugated)
                .mol(self.origin().mol)
                .density_matrix(denmat)
                .threshold(self.origin().threshold)
                .build()
                .map_err(|err| format_err!(err))
        })
    }
}

impl<'a, G> Projectable<G, Density<'a, Complex<f64>>> for DensitySymmetryOrbit<'a, G, Complex<f64>>
where
    G: SymmetryGroupProperties,
{
    type Projected<'p>
        = Result<Density<'p, Complex<f64>>, anyhow::Error>
    where
        Self: 'p;

    fn project_onto(&self, row: &G::RowSymbol) -> Self::Projected<'_> {
        let group_order = Complex::from(
            self.group()
                .order()
                .to_f64()
                .ok_or_else(|| format_err!("Unable to convert the group order to `f64`."))?,
        );
        let dim_f64 = row
            .dimensionality()
            .to_f64()
            .ok_or_else(|| format_err!("Unable to convert the degeneracy to `f64`."))?;
        let denmat_res = self.generate_orbit_algebra_terms(row).fold(
            Ok(Array2::<Complex<f64>>::zeros(
                self.origin().density_matrix().raw_dim(),
            )),
            |acc_res, item_res| {
                acc_res.and_then(|acc| {
                    item_res.and_then(|(chr, den)| {
                        let chr_complex = chr.complex_conjugate().complex_value();
                        Ok(acc + dim_f64 * chr_complex / group_order * den.density_matrix())
                    })
                })
            },
        );
        denmat_res.and_then(|denmat| {
            Density::builder()
                .bao(self.origin().bao)
                .complex_symmetric(self.origin().complex_symmetric)
                .complex_conjugated(self.origin().complex_conjugated)
                .mol(self.origin().mol)
                .density_matrix(denmat)
                .threshold(self.origin().threshold)
                .build()
                .map_err(|err| format_err!(err))
        })
    }
}
