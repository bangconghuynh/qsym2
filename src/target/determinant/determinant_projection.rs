//! Implementation of symmetry projection for Slater determinants.

use anyhow::{self, format_err};
use duplicate::duplicate_item;
use ndarray::Array1;
use num::{Complex, ToPrimitive};

use crate::analysis::Orbit;
use crate::angmom::spinor_rotation_3d::{SpinConstraint, SpinOrbitCoupled};
use crate::chartab::chartab_symbols::LinearSpaceSymbol;
use crate::projection::Projectable;
use crate::symmetry::symmetry_group::SymmetryGroupProperties;
use crate::target::noci::basis::OrbitBasis;
use crate::target::noci::multideterminant::MultiDeterminant;

use super::SlaterDeterminant;
use super::determinant_analysis::SlaterDeterminantSymmetryOrbit;

#[duplicate_item(
    [
        dtype_  [ f64 ]
        sctype_ [ SpinConstraint ]
        coeffs_ [
            let coeffs = self
                .generate_orbit_algebra_terms(row)
                .map(|item_res| item_res.and_then(|(chr, _)| {
                    let chr_complex = chr.complex_conjugate().complex_value();
                    if chr_complex.im > self.origin().threshold() {
                        Err(format_err!("Complex characters encountered. Slater determinant projection fails over the reals."))
                    } else {
                        Ok(dim_f64 / group_order * chr_complex.re)
                    }
                }))
                .collect::<Result<Vec<_>, _>>()?
        ]
    ]
    [
        dtype_  [ Complex<f64> ]
        sctype_ [ SpinConstraint ]
        coeffs_ [
            let coeffs = self
                .generate_orbit_algebra_terms(row)
                .map(|item_res| {
                    item_res
                        .map(|(chr, _)| dim_f64 / group_order * chr.complex_conjugate().complex_value())
                })
                .collect::<Result<Vec<_>, _>>()?
        ]
    ]
    [
        dtype_  [ Complex<f64> ]
        sctype_ [ SpinOrbitCoupled ]
        coeffs_ [
            let coeffs = self
                .generate_orbit_algebra_terms(row)
                .map(|item_res| {
                    item_res
                        .map(|(chr, _)| dim_f64 / group_order * chr.complex_conjugate().complex_value())
                })
                .collect::<Result<Vec<_>, _>>()?
        ]
    ]
)]
impl<'a, G> Projectable<G, SlaterDeterminant<'a, dtype_, sctype_>>
    for SlaterDeterminantSymmetryOrbit<'a, G, dtype_, sctype_>
where
    G: SymmetryGroupProperties + Clone,
{
    type Projected<'p>
        = Result<
        MultiDeterminant<
            'p,
            dtype_,
            OrbitBasis<'p, G, SlaterDeterminant<'p, dtype_, sctype_>>,
            sctype_,
        >,
        anyhow::Error,
    >
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

        let orbit_basis = OrbitBasis::builder()
            .origins(vec![self.origin().clone()])
            .group(self.group())
            .action(self.action())
            .prefactors(None)
            .build()
            .map_err(|err| format_err!(err))?;

        coeffs_;

        MultiDeterminant::builder()
            .basis(orbit_basis)
            .coefficients(Array1::from_vec(coeffs))
            .threshold(self.origin().threshold())
            .build()
            .map_err(|err| format_err!(err))
    }
}
