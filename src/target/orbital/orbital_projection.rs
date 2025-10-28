//! Implementation of symmetry projection for molecular orbitals.

use anyhow::{self, format_err};
use duplicate::duplicate_item;
use ndarray::Array1;
use num::{Complex, ToPrimitive};

use super::orbital_analysis::MolecularOrbitalSymmetryOrbit;
use crate::analysis::Orbit;
use crate::angmom::spinor_rotation_3d::{SpinConstraint, SpinOrbitCoupled};
use crate::chartab::chartab_symbols::LinearSpaceSymbol;
use crate::projection::Projectable;
use crate::symmetry::symmetry_group::SymmetryGroupProperties;
use crate::target::orbital::MolecularOrbital;

#[duplicate_item(
    [
        dtype_  [ f64 ]
        sctype_ [ SpinConstraint ]
        coeffs_ [
            let c_res = self.generate_orbit_algebra_terms(row).fold(
                Ok(Array1::zeros(self.origin().coefficients().raw_dim())),
                |acc_res, item_res| acc_res.and_then(|acc| item_res.and_then(|(chr, mo)| {
                    let chr_complex = chr.complex_conjugate().complex_value();
                    if chr_complex.im > self.origin().threshold() {
                        Err(format_err!("Complex characters encountered. Density projection fails over the reals."))
                    } else {
                        Ok(acc + dim_f64 * chr_complex.re / group_order * mo.coefficients())
                    }
                })),
            )
        ]
    ]
    [
        dtype_  [ Complex<f64> ]
        sctype_ [ SpinConstraint ]
        coeffs_ [
            let c_res = self.generate_orbit_algebra_terms(row).fold(
                Ok(Array1::zeros(self.origin().coefficients().raw_dim())),
                |acc_res, item_res| acc_res.and_then(|acc| item_res.map(|(chr, mo)| {
                    let chr_complex = chr.complex_conjugate().complex_value();
                    acc + dim_f64 * chr_complex / group_order * mo.coefficients()
                })),
            )
        ]
    ]
    [
        dtype_  [ Complex<f64> ]
        sctype_ [ SpinOrbitCoupled ]
        coeffs_ [
            let c_res = self.generate_orbit_algebra_terms(row).fold(
                Ok(Array1::zeros(self.origin().coefficients().raw_dim())),
                |acc_res, item_res| acc_res.and_then(|acc| item_res.map(|(chr, mo)| {
                    let chr_complex = chr.complex_conjugate().complex_value();
                    acc + dim_f64 * chr_complex / group_order * mo.coefficients()
                })),
            )
        ]
    ]
)]
impl<'a, G> Projectable<G, MolecularOrbital<'a, dtype_, sctype_>>
    for MolecularOrbitalSymmetryOrbit<'a, G, dtype_, sctype_>
where
    G: SymmetryGroupProperties,
{
    type Projected<'p>
        = Result<MolecularOrbital<'p, dtype_, sctype_>, anyhow::Error>
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
        coeffs_;
        c_res.and_then(|c| {
            MolecularOrbital::builder()
                .structure_constraint(self.origin().structure_constraint.clone())
                .baos(self.origin().baos.clone())
                .complex_symmetric(self.origin().complex_symmetric)
                .complex_conjugated(self.origin().complex_conjugated)
                .component_index(self.origin().component_index)
                .mol(self.origin().mol)
                .coefficients(c)
                .threshold(self.origin().threshold)
                .build()
                .map_err(|err| format_err!(err))
        })
    }
}
