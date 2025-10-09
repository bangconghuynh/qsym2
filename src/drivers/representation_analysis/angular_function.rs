//! Driver for symmetry analysis of angular functions.

use anyhow::{self, ensure, format_err};
use derive_builder::Builder;
use nalgebra::{Point3, Vector3};
use ndarray::{Array1, Array2};
use num_complex::{Complex, ComplexFloat};
use rayon::prelude::*;

use crate::analysis::{EigenvalueComparisonMode, RepAnalysis};
use crate::angmom::sh_conversion::sh_cart2rl_mat;
use crate::angmom::spinor_rotation_3d::{SpinConstraint, SpinOrbitCoupled};
use crate::auxiliary::atom::{Atom, ElementMap};
use crate::auxiliary::molecule::Molecule;
use crate::basis::ao::{
    BasisAngularOrder, BasisAtom, BasisShell, CartOrder, PureOrder, ShellOrder, SpinorOrder,
    SpinorParticleType, cart_tuple_to_str,
};
use crate::chartab::SubspaceDecomposable;
use crate::chartab::chartab_group::CharacterProperties;
use crate::io::format::{log_subtitle, qsym2_output};
use crate::symmetry::symmetry_group::SymmetryGroupProperties;
use crate::symmetry::symmetry_transformation::SymmetryTransformationKind;
use crate::target::determinant::SlaterDeterminant;
use crate::target::orbital::orbital_analysis::generate_det_mo_orbits;
use crate::target::tensor::axialvector::axialvector_analysis::AxialVector3SymmetryOrbit;
use crate::target::tensor::axialvector::{AxialVector3, TimeParity};

// ==================
// Struct definitions
// ==================

// ----------
// Parameters
// ----------

/// Structure containing control parameters for angular function representation analysis.
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
        .ok_or_else(|| format_err!("No symmetry operation found."))?
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
                let nbas = bao.n_funcs();
                let cs = vec![Array2::<f64>::eye(nbas)];
                let occs = vec![Array1::<f64>::ones(nbas)];
                let sao = match shell_order {
                    ShellOrder::Pure(_) | ShellOrder::Spinor(_) => {
                        Array2::<f64>::eye(bao.n_funcs())
                    }
                    ShellOrder::Cart(cartorder) => {
                        let cart2rl =
                            sh_cart2rl_mat(l, l, cartorder, true, &PureOrder::increasingm(l));
                        cart2rl.mapv(ComplexFloat::conj).t().dot(&cart2rl)
                    }
                };

                let mo_symmetries = SlaterDeterminant::<f64, SpinConstraint>::builder()
                    .structure_constraint(SpinConstraint::Restricted(1))
                    .baos(vec![&bao])
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
                            None, // Is this right for complex spherical harmonics?
                            params.integrality_threshold,
                            params.linear_independence_threshold,
                            SymmetryTransformationKind::Spatial,
                            EigenvalueComparisonMode::Real,
                            true,
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
                    ShellOrder::Spinor(_) => {
                        panic!("Unexpected spinor shell in `find_angular_function_representation`.")
                    }
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
    let rot_sym_strs = [Vector3::x(), Vector3::y(), Vector3::z()]
        .into_iter()
        .map(|v| {
            let rv = AxialVector3::<f64>::builder()
                .components(v)
                .time_parity(TimeParity::Even)
                .threshold(params.linear_independence_threshold)
                .build()
                .map_err(|err| format_err!(err))?;
            let mut orbit_rv = AxialVector3SymmetryOrbit::builder()
                .group(group)
                .origin(&rv)
                .integrality_threshold(params.integrality_threshold)
                .linear_independence_threshold(params.linear_independence_threshold)
                .symmetry_transformation_kind(SymmetryTransformationKind::Spatial)
                .eigenvalue_comparison_mode(EigenvalueComparisonMode::Real)
                .build()
                .map_err(|err| format_err!(err))?;
            let _ = orbit_rv
                .calc_smat(None, None, true)
                .unwrap()
                .calc_xmat(false);
            orbit_rv
                .analyse_rep()
                .map(|sym| sym.to_string())
                .map_err(|err| format_err!(err))
        })
        .map(|sym| sym.unwrap_or_else(|err| err.to_string()))
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
    let rot_sym_width = rot_sym_strs
        .iter()
        .map(|sym| sym.chars().count())
        .max()
        .unwrap_or(8)
        .max(8);

    let l_width = lmax.to_string().chars().count();
    let pure_width = (l_width + 1).max(4);
    let cart_width = usize::try_from(lmax)?.max(4);
    let rot_width = 3;

    log_subtitle(&format!(
        "Space-fixed spatial angular function symmetries in {}",
        group.finite_subgroup_name().unwrap_or(&group.name())
    ));
    qsym2_output!("");
    if lmax < 1 {
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
    } else {
        qsym2_output!(
            "{}",
            "┈".repeat(
                l_width
                    + pure_width
                    + pure_sym_width
                    + cart_width
                    + cart_sym_width
                    + rot_width
                    + rot_sym_width
                    + 15
            )
        );
        qsym2_output!(
            " {:>l_width$}  {:>pure_width$}  {:<pure_sym_width$}   {:>cart_width$}  {:<cart_sym_width$}  {:>rot_width$}  {:<}",
            "l",
            "Pure",
            "Pure sym.",
            "Cart",
            "Cart sym.",
            "Rot",
            "Rot sym."
        );
        qsym2_output!(
            "{}",
            "┈".repeat(
                l_width
                    + pure_width
                    + pure_sym_width
                    + cart_width
                    + cart_sym_width
                    + rot_width
                    + rot_sym_width
                    + 15
            )
        );
    }

    let empty_str = String::new();
    (0..=usize::try_from(lmax)?).for_each(|l| {
        if l > 0 {
            qsym2_output!("");
        }
        let n_pure = 2 * l + 1;
        let mut i_pure = 0;
        let mut i_rot = 0;

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
                let cart_str = if l == 1 {
                    // Rot sym to follow.
                    format!(
                        "{cart_symbol:>cart_width$}  {:<cart_sym_width$}",
                        cart_sym_strss
                            .get(l)
                            .and_then(|l_cart_sym_strs| l_cart_sym_strs.get(i_cart))
                            .unwrap_or(&empty_str)
                    )
                } else {
                    // No rot sym to follow.
                    format!(
                        "{cart_symbol:>cart_width$}  {:<}",
                        cart_sym_strss
                            .get(l)
                            .and_then(|l_cart_sym_strs| l_cart_sym_strs.get(i_cart))
                            .unwrap_or(&empty_str)
                    )
                };

                if l == 1 && i_rot < 3 {
                    let rot_str = format!(
                        "{:>rot_width$}  {:<}",
                        match i_rot {
                            0 => "Rx",
                            1 => "Ry",
                            2 => "Rz",
                            _ => "--",
                        },
                        rot_sym_strs.get(i_rot).unwrap_or(&"--".to_string())
                    );
                    i_rot += 1;
                    qsym2_output!(" {l_str}  {pure_str}   {cart_str}  {rot_str}");
                } else {
                    qsym2_output!(" {l_str}  {pure_str}   {cart_str}");
                }
            });
    });
    if lmax < 1 {
        qsym2_output!(
            "{}",
            "┈".repeat(l_width + pure_width + pure_sym_width + cart_width + cart_sym_width + 11)
        );
    } else {
        qsym2_output!(
            "{}",
            "┈".repeat(
                l_width
                    + pure_width
                    + pure_sym_width
                    + cart_width
                    + cart_sym_width
                    + rot_width
                    + rot_sym_width
                    + 15
            )
        );
    }
    qsym2_output!("");

    Ok(())
}

/// Determines the (co)representations of a group spanned by the spinor functions.
///
/// # Arguments
///
/// * `group` - A symmetry group.
/// * `params` - A parameter structure controlling the determination of spinor function
/// symmetries.
pub(crate) fn find_spinor_function_representation<G>(
    group: &G,
    params: &AngularFunctionRepAnalysisParams,
) -> Result<(), anyhow::Error>
where
    G: SymmetryGroupProperties + Clone + Send + Sync,
    G::CharTab: SubspaceDecomposable<Complex<f64>>,
    <<G as CharacterProperties>::CharTab as SubspaceDecomposable<Complex<f64>>>::Decomposition:
        Send + Sync,
{
    ensure!(
        group.is_double_group(),
        "The specified group is not a double group."
    );

    let emap = ElementMap::new();
    let thresh = group
        .elements()
        .clone()
        .into_iter()
        .next()
        .ok_or_else(|| format_err!("No symmetry operation found."))?
        .generating_element
        .threshold();
    let mol = Molecule::from_atoms(
        &[Atom::new_ordinary("H", Point3::origin(), &emap, thresh)],
        thresh,
    );
    let lmax = params.max_angular_momentum;

    let spinor_symss = (1..2 * lmax).step_by(2).fold(
        Vec::with_capacity(2 * usize::try_from(lmax)?),
        |mut acc, two_j| {
            let even_first = two_j.rem_euclid(4) == 1;
            let shell_order_g = ShellOrder::Spinor(SpinorOrder::increasingm(
                two_j,
                even_first,
                SpinorParticleType::Fermion(None),
            ));
            let shell_order_u = ShellOrder::Spinor(SpinorOrder::increasingm(
                two_j,
                !even_first,
                SpinorParticleType::Fermion(None),
            ));
            let bao_g = BasisAngularOrder::new(&[BasisAtom::new(
                &mol.atoms[0],
                &[BasisShell::new(two_j, shell_order_g)],
            )]);
            let bao_u = BasisAngularOrder::new(&[BasisAtom::new(
                &mol.atoms[0],
                &[BasisShell::new(two_j, shell_order_u)],
            )]);
            let nbas = bao_g.n_funcs();
            let cs = vec![Array2::<Complex<f64>>::eye(nbas)];
            let occs = vec![Array1::<f64>::ones(nbas)];
            let sao = Array2::<Complex<f64>>::eye(bao_g.n_funcs());

            for bao in [bao_g, bao_u] {
                let mo_symmetries = SlaterDeterminant::<Complex<f64>, SpinOrbitCoupled>::builder()
                    .structure_constraint(SpinOrbitCoupled::JAdapted(1))
                    .baos(vec![&bao])
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
                            None, // Is this right for complex spherical harmonics?
                            params.integrality_threshold,
                            params.linear_independence_threshold,
                            SymmetryTransformationKind::SpinSpatial,
                            EigenvalueComparisonMode::Modulus,
                            true,
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
                acc.push(mo_symmetries);
            }
            acc
        },
    );

    let spinor_sym_strss = spinor_symss
        .into_iter()
        .map(|l_spinor_syms_opt| {
            l_spinor_syms_opt
                .map(|l_spinor_syms| {
                    l_spinor_syms
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

    let spinor_sym_width = spinor_sym_strss
        .iter()
        .flat_map(|syms| syms.iter().map(|sym| sym.chars().count()))
        .max()
        .unwrap_or(11)
        .max(11);

    let j_width = format!("{}/2", 2 * lmax - 1).chars().count();
    let l_width = format!("(+; l = {lmax})").chars().count();
    let jl_width = j_width + l_width + 1;
    let mj_width = (j_width + 1).max(6);

    log_subtitle(&format!(
        "Space-fixed spinor function symmetries in {}",
        group.finite_subgroup_name().unwrap_or(&group.name())
    ));
    qsym2_output!("");
    qsym2_output!("{}", "┈".repeat(jl_width + mj_width + spinor_sym_width + 6));
    qsym2_output!(
        " {:>jl_width$}  {:>mj_width$}  {:<}",
        "j",
        "Spinor",
        "Spinor sym.",
    );
    qsym2_output!("{}", "┈".repeat(jl_width + mj_width + spinor_sym_width + 6));

    let empty_str = String::new();
    (1..usize::try_from(2 * lmax)?)
        .step_by(2)
        .enumerate()
        .for_each(|(i_two_j, two_j)| {
            if two_j > 1 {
                qsym2_output!("");
            }
            let n_spinor = two_j + 1;

            let even_first = two_j.rem_euclid(4) == 1;
            let two_j_u32 = u32::try_from(two_j).unwrap_or_else(|err| panic!("{err}"));
            let spinororder_g =
                SpinorOrder::increasingm(two_j_u32, even_first, SpinorParticleType::Fermion(None));
            spinororder_g
                .iter()
                .enumerate()
                .for_each(|(i_spinor, two_mj)| {
                    let j_str = if i_spinor == 0 {
                        let j_str_temp = format!(
                            "{two_j}/2 ({}; l = {})",
                            if spinororder_g.a() == 1 { "+" } else { "-" },
                            spinororder_g.l()
                        );
                        format!("{j_str_temp:>jl_width$}")
                    } else {
                        " ".repeat(jl_width)
                    };

                    let spinor_str = if i_spinor < n_spinor {
                        let spinor_str_temp = format!("{two_mj:+}/2");
                        let spinor_str = format!(
                            "{spinor_str_temp:>mj_width$}  {:<}",
                            spinor_sym_strss
                                .get(i_two_j * 2)
                                .and_then(|l_spinor_sym_strs| l_spinor_sym_strs.get(i_spinor))
                                .unwrap_or(&empty_str)
                        );
                        spinor_str
                    } else {
                        " ".repeat(mj_width + spinor_sym_width + 2)
                    };

                    qsym2_output!(" {j_str}  {spinor_str}");
                });

            qsym2_output!("");

            let spinororder_u =
                SpinorOrder::increasingm(two_j_u32, !even_first, SpinorParticleType::Fermion(None));
            spinororder_u
                .iter()
                .enumerate()
                .for_each(|(i_spinor, two_mj)| {
                    let j_str = if i_spinor == 0 {
                        let j_str_temp = format!(
                            "{two_j}/2 ({}; l = {})",
                            if spinororder_u.a() == 1 { "+" } else { "-" },
                            spinororder_u.l()
                        );
                        format!("{j_str_temp:>jl_width$}")
                    } else {
                        " ".repeat(jl_width)
                    };

                    let spinor_str = if i_spinor < n_spinor {
                        let spinor_str_temp = format!("{two_mj:+}/2");
                        let spinor_str = format!(
                            "{spinor_str_temp:>mj_width$}  {:<}",
                            spinor_sym_strss
                                .get(i_two_j * 2 + 1)
                                .and_then(|l_spinor_sym_strs| l_spinor_sym_strs.get(i_spinor))
                                .unwrap_or(&empty_str)
                        );
                        spinor_str
                    } else {
                        " ".repeat(mj_width + spinor_sym_width + 2)
                    };

                    qsym2_output!(" {j_str}  {spinor_str}");
                });
        });
    qsym2_output!("{}", "┈".repeat(jl_width + mj_width + spinor_sym_width + 6));
    qsym2_output!("");

    Ok(())
}
