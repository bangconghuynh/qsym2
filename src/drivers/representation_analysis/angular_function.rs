use anyhow::{self, format_err};
use derive_builder::Builder;
use nalgebra::Point3;
use ndarray::{Array1, Array2};
use num_complex::ComplexFloat;
use rayon::prelude::*;

use crate::analysis::{EigenvalueComparisonMode, RepAnalysis};
use crate::angmom::sh_conversion::sh_cart2rl_mat;
use crate::angmom::spinor_rotation_3d::SpinConstraint;
use crate::auxiliary::atom::{Atom, ElementMap};
use crate::auxiliary::molecule::Molecule;
use crate::basis::ao::{
    cart_tuple_to_str, BasisAngularOrder, BasisAtom, BasisShell, CartOrder, PureOrder, ShellOrder,
};
use crate::chartab::chartab_group::CharacterProperties;
use crate::chartab::SubspaceDecomposable;
use crate::io::format::{log_subtitle, qsym2_output};
use crate::symmetry::symmetry_group::SymmetryGroupProperties;
use crate::symmetry::symmetry_transformation::SymmetryTransformationKind;
use crate::target::determinant::SlaterDeterminant;
use crate::target::orbital::orbital_analysis::generate_det_mo_orbits;

// ==================
// Struct definitions
// ==================

// ----------
// Parameters
// ----------

/// A structure containing control parameters for angular function representation analysis.
#[derive(Clone, Builder, Debug)]
pub struct AngularFunctionRepAnalysisParams {
    /// Threshold for checking if subspace multiplicities are integral.
    #[builder(default = "1e-7")]
    pub integrality_threshold: f64,

    /// Threshold for determining zero eigenvalues in the orbit overlap matrix.
    #[builder(default = "1e-7")]
    pub linear_independence_threshold: f64,

    /// The maximum angular momentum degree to be analysed.
    #[builder(default = "2")]
    pub max_angular_momentum: u32,
}

impl AngularFunctionRepAnalysisParams {
    /// Returns a builder to construct a [`AngularFunctionRepAnalysisParams`] structure.
    pub fn builder() -> AngularFunctionRepAnalysisParamsBuilder {
        AngularFunctionRepAnalysisParamsBuilder::default()
    }
}

impl Default for AngularFunctionRepAnalysisParams {
    fn default() -> Self {
        Self::builder()
            .build()
            .expect("Unable to build as default `AngularFunctionRepAnalysisParams`.")
    }
}

// =========
// Functions
// =========

/// Determines the (co)representations of a group spanned by the angular functions (spherical/solid
/// harmonics or Cartesian functions).
///
/// # Arguments
///
/// * `group` - A symmetry group.
/// * `params` - A parameter structure controlling the determination of angular function
/// symmetries.
pub(crate) fn find_angular_function_representation<G>(
    group: &G,
    params: &AngularFunctionRepAnalysisParams,
) -> Result<(), anyhow::Error>
where
    G: SymmetryGroupProperties + Clone + Send + Sync,
    G::CharTab: SubspaceDecomposable<f64>,
    <<G as CharacterProperties>::CharTab as SubspaceDecomposable<f64>>::Decomposition: Send + Sync,
{
    let emap = ElementMap::new();
    let thresh = group
        .elements()
        .clone()
        .into_iter()
        .next()
        .expect("No symmetry operation found.")
        .generating_element
        .threshold();
    let mol = Molecule::from_atoms(
        &[Atom::new_ordinary("H", Point3::origin(), &emap, thresh)],
        thresh,
    );
    let lmax = params.max_angular_momentum;

    let (pure_symss, cart_symss) = (0..=lmax).fold(
        (
            Vec::with_capacity(usize::try_from(lmax)?),
            Vec::with_capacity(usize::try_from(lmax)?),
        ),
        |mut acc, l| {
            [
                ShellOrder::Pure(PureOrder::increasingm(l)),
                ShellOrder::Cart(CartOrder::lex(l)),
            ]
            .iter()
            .for_each(|shell_order| {
                let bao = BasisAngularOrder::new(&[BasisAtom::new(
                    &mol.atoms[0],
                    &[BasisShell::new(l, shell_order.clone())],
                )]);
                let n_spatial = bao.n_funcs();
                let cs = vec![Array2::<f64>::eye(n_spatial)];
                let occs = vec![Array1::<f64>::ones(n_spatial)];
                let sao = match shell_order {
                    ShellOrder::Pure(_) => Array2::<f64>::eye(bao.n_funcs()),
                    ShellOrder::Cart(cartorder) => {
                        let cart2rl =
                            sh_cart2rl_mat(l, l, cartorder, true, &PureOrder::increasingm(l));
                        cart2rl.mapv(ComplexFloat::conj).t().dot(&cart2rl)
                    }
                };

                let mo_symmetries = SlaterDeterminant::<f64>::builder()
                    .spin_constraint(SpinConstraint::Restricted(1))
                    .bao(&bao)
                    .complex_symmetric(false)
                    .mol(&mol)
                    .coefficients(&cs)
                    .occupations(&occs)
                    .threshold(params.linear_independence_threshold)
                    .build()
                    .map_err(|err| format_err!(err))
                    .and_then(|det| {
                        let mos = det.to_orbitals();
                        generate_det_mo_orbits(
                            &det,
                            &mos,
                            group,
                            &sao,
                            params.integrality_threshold,
                            params.linear_independence_threshold,
                            SymmetryTransformationKind::Spatial,
                            EigenvalueComparisonMode::Real,
                        )
                        .map(|(_, mut mo_orbitss)| {
                            mo_orbitss[0]
                                .par_iter_mut()
                                .map(|mo_orbit| {
                                    mo_orbit.calc_xmat(false)?;
                                    mo_orbit.analyse_rep().map_err(|err| format_err!(err))
                                })
                                .collect::<Vec<_>>()
                        })
                    });
                match shell_order {
                    ShellOrder::Pure(_) => acc.0.push(mo_symmetries),
                    ShellOrder::Cart(_) => acc.1.push(mo_symmetries),
                }
            });
            acc
        },
    );

    let pure_sym_strss = pure_symss
        .into_iter()
        .map(|l_pure_syms_opt| {
            l_pure_syms_opt
                .map(|l_pure_syms| {
                    l_pure_syms
                        .into_iter()
                        .map(|sym_opt| {
                            sym_opt
                                .map(|sym| sym.to_string())
                                .unwrap_or_else(|err| err.to_string())
                        })
                        .collect::<Vec<_>>()
                })
                .unwrap_or_else(|err| vec![err.to_string()])
        })
        .collect::<Vec<_>>();
    let cart_sym_strss = cart_symss
        .into_iter()
        .map(|l_cart_syms_opt| {
            l_cart_syms_opt
                .map(|l_cart_syms| {
                    l_cart_syms
                        .into_iter()
                        .map(|sym_opt| {
                            sym_opt
                                .map(|sym| sym.to_string())
                                .unwrap_or_else(|err| err.to_string())
                        })
                        .collect::<Vec<_>>()
                })
                .unwrap_or_else(|err| vec![err.to_string()])
        })
        .collect::<Vec<_>>();

    let pure_sym_width = pure_sym_strss
        .iter()
        .flat_map(|syms| syms.iter().map(|sym| sym.chars().count()))
        .max()
        .unwrap_or(9)
        .max(9);
    let cart_sym_width = cart_sym_strss
        .iter()
        .flat_map(|syms| syms.iter().map(|sym| sym.chars().count()))
        .max()
        .unwrap_or(9)
        .max(9);

    let l_width = lmax.to_string().chars().count();
    let pure_width = (l_width + 1).max(4);
    let _pure_width_m1 = pure_width - 1;
    let cart_width = usize::try_from(lmax)?.max(4);

    log_subtitle(&format!(
        "Space-fixed spatial angular function symmetries in {}",
        group.finite_subgroup_name().unwrap_or(&group.name())
    ));
    qsym2_output!("");
    qsym2_output!(
        "{}",
        "┈".repeat(l_width + pure_width + pure_sym_width + cart_width + cart_sym_width + 11)
    );
    qsym2_output!(
        " {:>l_width$}  {:>pure_width$}  {:<pure_sym_width$}   {:>cart_width$}  {:<}",
        "l",
        "Pure",
        "Pure sym.",
        "Cart",
        "Cart sym."
    );
    qsym2_output!(
        "{}",
        "┈".repeat(l_width + pure_width + pure_sym_width + cart_width + cart_sym_width + 11)
    );

    let empty_str = String::new();
    (0..=usize::try_from(lmax)?).for_each(|l| {
        if l > 0 {
            qsym2_output!("");
        }
        let n_pure = 2 * l + 1;
        let mut i_pure = 0;

        let l_u32 = u32::try_from(l).unwrap_or_else(|err| panic!("{err}"));
        let cartorder = CartOrder::lex(l_u32);
        cartorder
            .iter()
            .enumerate()
            .for_each(|(i_cart, cart_tuple)| {
                let l_str = if i_cart == 0 {
                    format!("{l:>l_width$}")
                } else {
                    " ".repeat(l_width)
                };

                let pure_str = if i_pure < n_pure {
                    let pure_str = format!(
                        "{:>pure_width$}  {:<pure_sym_width$}",
                        if i_pure < l {
                            format!("-{}", i_pure.abs_diff(l))
                        } else {
                            format!("+{}", i_pure.abs_diff(l))
                        },
                        pure_sym_strss
                            .get(l)
                            .and_then(|l_pure_sym_strs| l_pure_sym_strs.get(i_pure))
                            .unwrap_or(&empty_str)
                    );
                    i_pure += 1;
                    pure_str
                } else {
                    " ".repeat(pure_width + pure_sym_width + 2)
                };

                let cart_symbol = cart_tuple_to_str(cart_tuple, true);
                let cart_str = format!(
                    "{cart_symbol:>cart_width$}  {:<}",
                    cart_sym_strss
                        .get(l)
                        .and_then(|l_cart_sym_strs| l_cart_sym_strs.get(i_cart))
                        .unwrap_or(&empty_str)
                );

                qsym2_output!(" {l_str}  {pure_str}   {cart_str}");
            });
    });
    qsym2_output!(
        "{}",
        "┈".repeat(l_width + pure_width + pure_sym_width + cart_width + cart_sym_width + 11)
    );
    qsym2_output!("");

    Ok(())
}
