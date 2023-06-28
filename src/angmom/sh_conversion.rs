use std::cmp::Ordering;

use factorial::Factorial;
use ndarray::{Array2, Axis};
use num::{BigUint, Complex};
use num_traits::{cast::ToPrimitive, Zero};

use crate::aux::ao_basis::{CartOrder, PureOrder};
use crate::permutation::PermutableCollection;

#[cfg(test)]
#[path = "sh_conversion_tests.rs"]
mod sh_conversion_tests;

/// Calculates the number of combinations of `n` things taken `r` at a time (signed arguments).
///
/// If $`n < 0`$ or $`r < 0`$ or $`r > n`$, `0` is returned.
///
/// # Arguments
///
/// * `n` - Number of things.
/// * `r` - Number of elements taken.
///
/// # Returns
///
/// The number of combinations.
fn comb(n: i32, r: i32) -> BigUint {
    if n < 0 || r < 0 || r > n {
        BigUint::zero()
    } else {
        let nu = u32::try_from(n).expect("Unable to convert `n` to `u32`.");
        let ru = u32::try_from(r).expect("Unable to convert `r` to `u32`.");
        (nu - ru + 1..=nu).product::<BigUint>()
            / BigUint::from(ru)
                .checked_factorial()
                .unwrap_or_else(|| panic!("Unable to compute the factorial of {ru}."))
    }
}

/// Calculates the number of combinations of `n` things taken `r` at a time (unsigned arguments).
///
/// If $`r > n`$, `0` is returned.
///
/// # Arguments
///
/// * `n` - Number of things.
/// * `r` - Number of elements taken.
///
/// # Returns
///
/// The number of combinations.
fn combu(nu: u32, ru: u32) -> BigUint {
    if ru > nu {
        BigUint::zero()
    } else {
        (nu - ru + 1..=nu).product::<BigUint>()
            / BigUint::from(ru)
                .checked_factorial()
                .unwrap_or_else(|| panic!("Unable to compute the factorial of {ru}."))
    }
}

/// Calculates the number of permutations of `n` things taken `r` at a time (signed arguments).
///
/// If $`n < 0`$ or $`r < 0`$ or $`r > n`$, `0` is returned.
///
/// # Arguments
///
/// * `n` - Number of things.
/// * `r` - Number of elements taken.
///
/// # Returns
///
/// The number of permutations.
fn perm(n: i32, r: i32) -> BigUint {
    if n < 0 || r < 0 || r > n {
        BigUint::zero()
    } else {
        let nu = u32::try_from(n).expect("Unable to convert `n` to `u32`.");
        let ru = u32::try_from(r).expect("Unable to convert `r` to `u32`.");
        (nu - ru + 1..=nu).product::<BigUint>()
    }
}

/// Calculates the number of permutations of `n` things taken `r` at a time (unsigned arguments).
///
/// If $`r > n`$, `0` is returned.
///
/// # Arguments
///
/// * `n` - Number of things.
/// * `r` - Number of elements taken.
///
/// # Returns
///
/// The number of permutations.
fn permu(nu: u32, ru: u32) -> BigUint {
    if ru > nu {
        BigUint::zero()
    } else {
        (nu - ru + 1..=nu).product::<BigUint>()
    }
}

/// Obtains the normalisation constant for a solid harmonic Gaussian, as given in Equation 8 of
/// Schlegel, H. B. & Frisch, M. J. Transformation between Cartesian and pure spherical harmonic
/// Gaussians. *International Journal of Quantum Chemistry* **54**, 83–87 (1995),
/// [DOI](https://doi.org/10.1002/qua.560540202).
///
/// The complex solid harmonic Gaussian is defined in Equation 1 of the above reference as
///
/// ```math
/// \tilde{g}(\alpha, l, m, n, \mathbf{r})
///     = \tilde{N}(n, \alpha) Y_l^m r^n e^{-\alpha r^2},
/// ```
///
/// where $`Y_l^m`$ is a complex spherical harmonic of degree $`l`$ and order $`m`$.
///
/// # Arguments
///
/// * `n` - The non-negative exponent of the radial part of the solid harmonic Gaussian.
/// * `alpha` - The coefficient on the exponent of the Gaussian term.
///
/// # Returns
///
/// The normalisation constant $`\tilde{N}(n, \alpha)`$.
fn norm_sph_gaussian(n: u32, alpha: f64) -> f64 {
    let num = (BigUint::from(2u64).pow(2 * n + 3)
        * BigUint::from(u64::from(n) + 1)
            .checked_factorial()
            .unwrap_or_else(|| panic!("Unable to compute the factorial of {}.", u64::from(n) + 1)))
    .to_f64()
    .expect("Unable to convert a `BigUint` value to `f64`.")
        * alpha.powf(f64::from(n) + 1.5);
    let den = BigUint::from(2 * u64::from(n) + 2)
        .checked_factorial()
        .unwrap_or_else(|| {
            panic!(
                "Unable to compute the factorial of {}.",
                2 * u64::from(n) + 2
            )
        })
        .to_f64()
        .expect("Unable to convert a `BigUint` value to `f64`.")
        * std::f64::consts::PI.sqrt();
    (num / den).sqrt()
}

/// Obtains the normalisation constant for a Cartesian Gaussian, as given in Equation 9 of
/// Schlegel, H. B. & Frisch, M. J. Transformation between Cartesian and pure spherical harmonic
/// Gaussians. *International Journal of Quantum Chemistry* **54**, 83–87 (1995),
/// [DOI](https://doi.org/10.1002/qua.560540202).
///
/// The Cartesian Gaussian is defined in Equation 2 of the above reference as
///
/// ```math
/// g(\alpha, l_x, l_y, l_z, \mathbf{r})
///     = N(l_x, l_y, l_z, \alpha) x^{l_x} y^{l_y} z^{l_z} e^{-\alpha r^2}.
/// ```
///
/// # Arguments
///
/// * `lcartqns` - A tuple of $`(l_x, l_y, l_z)`$ specifying the non-negative exponents of
/// the Cartesian components of the Cartesian Gaussian.
/// * `alpha` - The coefficient on the exponent of the Gaussian term.
///
/// # Returns
///
/// The normalisation constant $`N(l_x, l_y, l_z, \alpha)`$.
fn norm_cart_gaussian(lcartqns: (u32, u32, u32), alpha: f64) -> f64 {
    let (lx, ly, lz) = lcartqns;
    let lcart = lx + ly + lz;
    let num = (BigUint::from(2u32).pow(2 * lcart)
        * BigUint::from(lx)
            .checked_factorial()
            .unwrap_or_else(|| panic!("Unable to compute the factorial of {lx}."))
        * BigUint::from(ly)
            .checked_factorial()
            .unwrap_or_else(|| panic!("Unable to compute the factorial of {ly}."))
        * BigUint::from(lz)
            .checked_factorial()
            .unwrap_or_else(|| panic!("Unable to compute the factorial of {lz}.")))
    .to_f64()
    .expect("Unable to convert a `BigUint` value to `f64`.")
        * alpha.powf(f64::from(lcart) + 1.5);
    let den = (BigUint::from(2 * lx)
        .checked_factorial()
        .unwrap_or_else(|| panic!("Unable to compute the factorial of {}.", 2 * lx))
        * BigUint::from(2 * ly)
            .checked_factorial()
            .unwrap_or_else(|| panic!("Unable to compute the factorial of {}.", 2 * ly))
        * BigUint::from(2 * lz)
            .checked_factorial()
            .unwrap_or_else(|| panic!("Unable to compute the factorial of {}.", 2 * lz)))
    .to_f64()
    .expect("Unable to convert a `BigUint` value to `f64`.")
        * std::f64::consts::PI.powi(3).sqrt();
    (num / den).sqrt()
}

/// Obtain the complex coefficients $`c(l, m_l, n, l_x, l_y, l_z)`$ based on Equation 15 of
/// Schlegel, H. B. & Frisch, M. J. Transformation between Cartesian and pure spherical harmonic
/// Gaussians. *International Journal of Quantum Chemistry* **54**, 83–87 (1995),
/// [DOI](https://doi.org/10.1002/qua.560540202), but more generalised
/// for $`l \leq l_{\mathrm{cart}} = l_x + l_y + l_z`$.
///
/// Let $`\tilde{g}(\alpha, l, m_l, l_{\mathrm{cart}}, \mathbf{r})`$ be a complex solid
/// harmonic Gaussian as defined in Equation 1 of
/// the above reference with $`n = l_{\mathrm{cart}}`$, and let
/// $`g(\alpha, l_x, l_y, l_z, \mathbf{r})`$ be a Cartesian Gaussian as defined in
/// Equation 2 of the above reference.
/// The complex coefficients $`c(l, m_l, n, l_x, l_y, l_z)`$ effect the transformation
///
/// ```math
/// \tilde{g}(\alpha, l, m_l, l_{\mathrm{cart}}, \mathbf{r})
/// = \sum_{l_x+l_y+l_z=l_{\mathrm{cart}}}
///     c(l, m_l, l_{\mathrm{cart}}, l_x, l_y, l_z)
///     g(\alpha, l_x, l_y, l_z, \mathbf{r})
/// ```
///
/// and are given by
///
/// ```math
/// c(l, m_l, l_{\mathrm{cart}}, l_x, l_y, l_z)
/// = \frac{\tilde{N}(l_{\mathrm{cart}}, \alpha)}{N(l_x, l_y, l_z, \alpha)}
///     \tilde{c}(l, m_l, l_{\mathrm{cart}}, l_x, l_y, l_z).
/// ```
///
/// The normalisation constants $`\tilde{N}(l_{\mathrm{cart}}, \alpha)`$
/// and $`N(l_x, l_y, l_z, \alpha)`$ are given in Equations 8 and 9 of
/// the above reference, and for $`n = l_{\mathrm{cart}}`$, this ratio turns out to be
/// independent of $`\alpha`$.
/// The more general form of $`\tilde{c}(l, m_l, l_{\mathrm{cart}}, l_x, l_y, l_z)`$ has been
/// derived to be
///
/// ```math
/// \tilde{c}(l, m_l, l_{\mathrm{cart}}, l_x, l_y, l_z)
/// = \frac{\lambda_{\mathrm{cs}}}{2^l l!}
///     \sqrt{\frac{(2l+1)(l-\lvert m_l \rvert)!}{4\pi(l+\lvert m_l \rvert)!}}
///     \sum_{i=0}^{(l-\lvert m_l \rvert)/2}
///         {l\choose i} \frac{(-1)^i(2l-2i)!}{(l-\lvert m_l \rvert -2i)!}\\
///     \sum_{p=0}^{\lvert m_l \rvert} {{\lvert m_l \rvert} \choose p}
///         (\pm \mathbb{i})^{\lvert m_l \rvert-p}
///     \sum_{q=0}^{\Delta l/2} {{\Delta l/2} \choose q} {i \choose j_q}
///     \sum_{k=0}^{j_q} {q \choose t_{pk}} {j_q \choose k}
/// ```
///
/// where $`+\mathbb{i}`$ applies for $`m_l > 0`$, $`-\mathbb{i}`$
/// for $`m_l \le 0`$, $`\lambda_{\mathrm{cs}}`$ is the Condon--Shortley
/// phase given by
///
/// ```math
/// \lambda_{\mathrm{cs}} =
///     \begin{cases}
///         (-1)^{m_l} & m_l > 0 \\
///         1          & m_l \leq 0
///     \end{cases}
/// ```
///
/// and
///
/// ```math
/// t_{pk} = \frac{l_x-p-2k}{2} \quad \textrm{and} \quad
/// j_q = \frac{l_x+l_y-\lvert m_l \rvert-2q}{2}.
/// ```
///
/// If $`\Delta l`$ is odd, $`\tilde{c}(l, m_l, l_{\mathrm{cart}}, l_x, l_y, l_z)`$ must vanish.
/// When  $`t_{pk}`$ or $`j_q`$ is a half-integer, the inner sum in which it is involved
/// evaluates to zero.
///
/// # Arguments
///
/// * `lpureqns` - A tuple of $`(l, m_l)`$ specifying the quantum numbers for the spherical
/// harmonic component of the solid harmonic Gaussian.
/// * `lcartqns` - A tuple of $`(l_x, l_y, l_z)`$ specifying the exponents of the Cartesian
/// components of the Cartesian Gaussian.
/// * `csphase` - If `true`, the Condon--Shortley phase will be used as defined above.
/// If `false`, this phase will be set to unity.
///
/// # Returns
///
/// The complex factor $`c(l, m_l, l_{\mathrm{cart}}, l_x, l_y, l_z)`$.
///
/// # Panics
///
/// Panics when any required factorials cannot be computed.
#[allow(clippy::too_many_lines)]
pub fn complexc(lpureqns: (u32, i32), lcartqns: (u32, u32, u32), csphase: bool) -> Complex<f64> {
    let (l, m) = lpureqns;
    let li32 = i32::try_from(l).unwrap_or_else(|_| panic!("Cannot convert `{l}` to `i32`."));
    assert!(
        m.unsigned_abs() <= l,
        "m must be between -l and l (inclusive)."
    );
    let (lx, ly, lz) = lcartqns;
    let lxi32 = i32::try_from(lx).unwrap_or_else(|_| panic!("Cannot convert `{lx}` to `i32`."));
    let lyi32 = i32::try_from(ly).unwrap_or_else(|_| panic!("Cannot convert `{ly}` to `i32`."));
    let lzi32 = i32::try_from(lz).unwrap_or_else(|_| panic!("Cannot convert `{lz}` to `i32`."));
    let lcart = lx + ly + lz;
    let lcarti32 = lxi32 + lyi32 + lzi32;
    let dl = lcarti32 - li32;
    if dl % 2 != 0 {
        return Complex::<f64>::zero();
    }

    let num = f64::from(
        (2 * l + 1)
            * (l - m.unsigned_abs())
                .checked_factorial()
                .unwrap_or_else(|| {
                    panic!(
                        "Unable to compute the factorial of {}.",
                        l - m.unsigned_abs()
                    )
                }),
    );
    let den = 4.0
        * std::f64::consts::PI
        * f64::from(
            (l + m.unsigned_abs())
                .checked_factorial()
                .unwrap_or_else(|| {
                    panic!(
                        "Unable to compute the factorial of {}.",
                        l + m.unsigned_abs()
                    )
                }),
        );
    let mut prefactor =
        1.0 / f64::from(
            2u32.pow(l)
                * l.checked_factorial()
                    .unwrap_or_else(|| panic!("Unable to compute the factorial of {l}.")),
        ) * (num / den).sqrt();
    if csphase && m > 0 {
        prefactor *=
            f64::from((-1i32).pow(u32::try_from(m).expect("Unable to convert `m` to `u32`.")));
    }
    let ntilde = norm_sph_gaussian(lcart, 1.0);
    let n = norm_cart_gaussian(lcartqns, 1.0);

    let si =
        (0..=((l - m.unsigned_abs()).div_euclid(2))).fold(Complex::<f64>::zero(), |acc_si, i| {
            // i <= (l - |m|) / 2
            let ii32 =
                i32::try_from(i).unwrap_or_else(|_| panic!("Cannot convert `{i}` to `i32`."));
            let mut ifactor = combu(l, i)
                .to_f64()
                .expect("Unable to convert a `BigUint` value to `f64`.")
                * BigUint::from(2 * l - 2 * i)
                    .checked_factorial()
                    .unwrap_or_else(|| {
                        panic!("Unable to compute the factorial of {}.", 2 * l - 2 * i)
                    })
                    .to_f64()
                    .unwrap_or_else(|| {
                        panic!(
                            "Unable to convert the factorial of {} to `f64`.",
                            2 * l - 2 * i
                        )
                    })
                / BigUint::from(l - m.unsigned_abs() - 2 * i)
                    .checked_factorial()
                    .unwrap_or_else(|| {
                        panic!(
                            "Unable to compute the factorial of {}.",
                            l - m.unsigned_abs() - 2 * i
                        )
                    })
                    .to_f64()
                    .unwrap_or_else(|| {
                        panic!(
                            "Unable to convert the factorial of {} to `f64`.",
                            l - m.unsigned_abs() - 2 * i
                        )
                    });
            if i % 2 == 1 {
                ifactor *= -1.0;
            };
            let sp = (0..=(m.unsigned_abs())).fold(Complex::<f64>::zero(), |acc_sp, p| {
                let pi32 =
                    i32::try_from(p).unwrap_or_else(|_| panic!("Cannot convert `{p}` to `i32`."));
                let pfactor = if m > 0 {
                    combu(m.unsigned_abs(), p)
                        .to_f64()
                        .expect("Unable to convert a `BigUint` value to `f64`.")
                        * Complex::<f64>::i().powu(m.unsigned_abs() - p)
                } else {
                    combu(m.unsigned_abs(), p)
                        .to_f64()
                        .expect("Unable to convert a `BigUint` value to `f64`.")
                        * (-1.0 * Complex::<f64>::i()).powu(m.unsigned_abs() - p)
                };
                let sq = (0..=(dl.div_euclid(2))).fold(Complex::<f64>::zero(), |acc_sq, q| {
                    let jq_num = lxi32 + lyi32 - 2 * q - m.abs();
                    if jq_num.rem_euclid(2) == 0 {
                        let jq = jq_num.div_euclid(2);
                        let qfactor = (comb(dl.div_euclid(2), q) * comb(ii32, jq))
                            .to_f64()
                            .expect("Unable to convert a `BigUint` value to `f64`.");
                        let sk = (0..=jq).fold(Complex::<f64>::zero(), |acc_sk, k| {
                            let tpk_num = lxi32 - pi32 - 2 * k;
                            if tpk_num.rem_euclid(2) == 0 {
                                let tpk = tpk_num.div_euclid(2);
                                let kfactor = (comb(q, tpk) * comb(jq, k))
                                    .to_f64()
                                    .expect("Unable to convert a `BigUint` value to `f64`.");
                                acc_sk + kfactor
                            } else {
                                acc_sk
                            }
                        });
                        acc_sq + qfactor * sk
                    } else {
                        acc_sq
                    }
                });
                acc_sp + pfactor * sq
            });
            acc_si + ifactor * sp
        });
    (ntilde / n) * prefactor * si
}

/// Calculates the overlap between two normalised Cartesian Gaussians of the same order and radial
/// width, as given in Equation 19 of Schlegel, H. B. & Frisch, M. J. Transformation between
/// Cartesian and pure spherical harmonic Gaussians. *International Journal of Quantum Chemistry*
/// **54**, 83–87 (1995), [DOI](https://doi.org/10.1002/qua.560540202).
///
/// # Arguments
///
/// * `lcartqns1` - A tuple of $`(l_x, l_y, l_z)`$ specifying the exponents of the Cartesian
/// components of the first Cartesian Gaussian.
/// * `lcartqns2` - A tuple of $`(l_x, l_y, l_z)`$ specifying the exponents of the Cartesian
/// components of the first Cartesian Gaussian.
///
/// # Returns
///
/// The overlap between the two specified normalised Cartesian Gaussians.
fn cartov(lcartqns1: (u32, u32, u32), lcartqns2: (u32, u32, u32)) -> f64 {
    let (lx1, ly1, lz1) = lcartqns1;
    let (lx2, ly2, lz2) = lcartqns2;
    let lcart1 = lx1 + ly1 + lz1;
    let lcart2 = lx2 + ly2 + lz2;
    assert_eq!(
        lcart1, lcart2,
        "Only Cartesian Gaussians of the same order are supported."
    );

    if (lx1 + lx2).rem_euclid(2) == 0
        && (ly1 + ly2).rem_euclid(2) == 0
        && (lz1 + lz2).rem_euclid(2) == 0
    {
        let num1 = (BigUint::from(lx1 + lx2)
            .checked_factorial()
            .unwrap_or_else(|| panic!("Unable to compute the factorial of {}.", lx1 + lx2))
            * BigUint::from(ly1 + ly2)
                .checked_factorial()
                .unwrap_or_else(|| panic!("Unable to compute the factorial of {}.", ly1 + ly2))
            * BigUint::from(lz1 + lz2)
                .checked_factorial()
                .unwrap_or_else(|| panic!("Unable to compute the factorial of {}.", lz1 + lz2)))
        .to_f64()
        .expect("Unable to convert a `BigUint` value to `f64`.");

        let den1 = (BigUint::from((lx1 + lx2).div_euclid(2))
            .checked_factorial()
            .unwrap_or_else(|| {
                panic!(
                    "Unable to compute the factorial of {}.",
                    (lx1 + lx2).div_euclid(2)
                )
            })
            * BigUint::from((ly1 + ly2).div_euclid(2))
                .checked_factorial()
                .unwrap_or_else(|| {
                    panic!(
                        "Unable to compute the factorial of {}.",
                        (ly1 + ly2).div_euclid(2)
                    )
                })
            * BigUint::from((lz1 + lz2).div_euclid(2))
                .checked_factorial()
                .unwrap_or_else(|| {
                    panic!(
                        "Unable to compute the factorial of {}.",
                        (lz1 + lz2).div_euclid(2)
                    )
                }))
        .to_f64()
        .expect("Unable to convert a `BigUint` value to `f64`.");

        let num2 = (BigUint::from(lx1)
            .checked_factorial()
            .unwrap_or_else(|| panic!("Unable to compute the factorial of {lx1}."))
            * BigUint::from(ly1)
                .checked_factorial()
                .unwrap_or_else(|| panic!("Unable to compute the factorial of {ly1}."))
            * BigUint::from(lz1)
                .checked_factorial()
                .unwrap_or_else(|| panic!("Unable to compute the factorial of {lz1}."))
            * BigUint::from(lx2)
                .checked_factorial()
                .unwrap_or_else(|| panic!("Unable to compute the factorial of {lx2}."))
            * BigUint::from(ly2)
                .checked_factorial()
                .unwrap_or_else(|| panic!("Unable to compute the factorial of {ly2}."))
            * BigUint::from(lz2)
                .checked_factorial()
                .unwrap_or_else(|| panic!("Unable to compute the factorial of {lz2}.")))
        .to_f64()
        .expect("Unable to convert a `BigUint` value to `f64`.");

        let den2 = (BigUint::from(2 * lx1)
            .checked_factorial()
            .unwrap_or_else(|| panic!("Unable to compute the factorial of {}.", 2 * lx1))
            * BigUint::from(2 * ly1)
                .checked_factorial()
                .unwrap_or_else(|| panic!("Unable to compute the factorial of {}.", 2 * ly1))
            * BigUint::from(2 * lz1)
                .checked_factorial()
                .unwrap_or_else(|| panic!("Unable to compute the factorial of {}.", 2 * lz1))
            * BigUint::from(2 * lx2)
                .checked_factorial()
                .unwrap_or_else(|| panic!("Unable to compute the factorial of {}.", 2 * lx2))
            * BigUint::from(2 * ly2)
                .checked_factorial()
                .unwrap_or_else(|| panic!("Unable to compute the factorial of {}.", 2 * ly2))
            * BigUint::from(2 * lz2)
                .checked_factorial()
                .unwrap_or_else(|| panic!("Unable to compute the factorial of {}.", 2 * lz2)))
        .to_f64()
        .expect("Unable to convert a `BigUint` value to `f64`.");

        (num1 / den1) * (num2 / den2).sqrt()
    } else {
        0.0
    }
}

/// Computes the inverse complex coefficients $`c^{-1}(l_x, l_y, l_z, l, m_l, l_{\mathrm{cart}})`$
/// based on Equation 18 of Schlegel, H. B. & Frisch, M. J. Transformation between
/// Cartesian and pure spherical harmonic Gaussians. *International Journal of Quantum Chemistry*
/// **54**, 83–87 (1995), [DOI](https://doi.org/10.1002/qua.560540202), but more generalised for
/// $`l \leq l_{\mathrm{cart}} = l_x + l_y + l_z`$.
///
/// Let $`\tilde{g}(\alpha, l, m_l, l_{\mathrm{cart}}, \mathbf{r})`$ be a complex solid
/// harmonic Gaussian as defined in Equation 1 of the above reference with
/// $`n = l_{\mathrm{cart}}`$, and let $`g(\alpha, l_x, l_y, l_z, \mathbf{r})`$ be a Cartesian
/// Gaussian as defined in Equation 2 of the above reference. The inverse complex coefficients
/// $`c^{-1}(l_x, l_y, l_z, l, m_l, l_{\mathrm{cart}})`$ effect the inverse transformation
///
/// ```math
/// g(\alpha, l_x, l_y, l_z, \mathbf{r})
/// = \sum_{l \le l_{\mathrm{cart}} = l_x+l_y+l_z} \sum_{m_l = -l}^{l}
///     c^{-1}(l_x, l_y, l_z, l, m_l, l_{\mathrm{cart}})
///     \tilde{g}(\alpha, l, m_l, l_{\mathrm{cart}}, \mathbf{r}).
/// ```
///
/// # Arguments
///
/// * `lcartqns` - A tuple of $`(l_x, l_y, l_z)`$ specifying the exponents of the Cartesian
/// components of the Cartesian Gaussian.
/// * `lpureqns` - A tuple of $`(l, m_l)`$ specifying the quantum numbers for the spherical
/// harmonic component of the solid harmonic Gaussian.
/// * `csphase` - If `true`, the Condon--Shortley phase will be used as defined in
/// [`complexc`]. If `false`, this phase will be set to unity.
///
/// # Returns
///
/// $`c^{-1}(l_x, l_y, l_z, l, m_l, l_{\mathrm{cart}})`$.
pub fn complexcinv(lcartqns: (u32, u32, u32), lpureqns: (u32, i32), csphase: bool) -> Complex<f64> {
    let (lx, ly, lz) = lcartqns;
    let lcart = lx + ly + lz;
    let mut cinv = Complex::<f64>::zero();
    for lx2 in 0..=lcart {
        for ly2 in 0..=(lcart - lx2) {
            let lz2 = lcart - lx2 - ly2;
            cinv += cartov(lcartqns, (lx2, ly2, lz2))
                * complexc(lpureqns, (lx2, ly2, lz2), csphase).conj();
        }
    }
    cinv
}

/// Obtains the transformation matrix $`\boldsymbol{\Upsilon}^{(l)}`$ allowing complex spherical
/// harmonics to be expressed as linear combinations of real spherical harmonics.
///
/// Let $`Y_{lm}`$ be a real spherical harmonic of degree $`l`$. Then, a complex spherical
/// harmonic of degree $`l`$ and order $`m`$ is given by
///
/// ```math
/// Y_l^m =
///     \begin{cases}
///         \frac{\lambda_{\mathrm{cs}}}{\sqrt{2}}
///         \left(Y_{l\lvert m \rvert}
///               - \mathbb{i} Y_{l,-\lvert m \rvert}\right)
///         & \mathrm{if}\ m < 0 \\
///         Y_{l0} & \mathrm{if}\ m = 0 \\
///         \frac{\lambda_{\mathrm{cs}}}{\sqrt{2}}
///         \left(Y_{l\lvert m \rvert}
///               + \mathbb{i} Y_{l,-\lvert m \rvert}\right)
///         & \mathrm{if}\ m > 0 \\
///     \end{cases}
/// ```
///
/// where $`\lambda_{\mathrm{cs}}`$ is the Condon--Shortley phase as defined in [`complexc`].
/// The linear combination coefficients can then be gathered into a square matrix
/// $`\boldsymbol{\Upsilon}^{(l)}`$ of dimensions $`(2l+1)\times(2l+1)`$ such that
///
/// ```math
///     Y_l^m = \sum_{m'} Y_{lm'} \Upsilon^{(l)}_{m'm}.
/// ```
///
/// # Arguments
///
/// * `l` - The spherical harmonic degree.
/// * `csphase` - If `true`, $`\lambda_{\mathrm{cs}}`$ is as defined in [`complexc`]. If `false`,
/// $`\lambda_{\mathrm{cs}} = 1`$.
/// * `pureorder` - A [`PureOrder`] struct giving the ordering of the components of the pure
/// Gaussians.
///
/// # Returns
///
/// The $`\boldsymbol{\Upsilon}^{(l)}`$ matrix.
pub fn sh_c2r_mat(l: u32, csphase: bool, pureorder: &PureOrder) -> Array2<Complex<f64>> {
    assert_eq!(pureorder.lpure, l, "Mismatched pure ranks.");
    let lusize = l as usize;
    let mut upmat = Array2::<Complex<f64>>::zeros((2 * lusize + 1, 2 * lusize + 1));
    let po_il = PureOrder::increasingm(l);
    for &mcomplex in po_il.iter() {
        let absmreal = mcomplex.unsigned_abs() as usize;
        match mcomplex.cmp(&0) {
            Ordering::Less => {
                // Python-equivalent:
                // upmat[-absmreal + l, mcomplex + l] = -1.0j / np.sqrt(2)
                // upmat[+absmreal + l, mcomplex + l] = 1.0 / np.sqrt(2)
                // mcomplex = -absmreal
                upmat[(lusize - absmreal, lusize - absmreal)] =
                    Complex::<f64>::new(0.0, -1.0 / 2.0f64.sqrt());
                upmat[(lusize + absmreal, lusize - absmreal)] =
                    Complex::<f64>::new(1.0 / 2.0f64.sqrt(), 0.0);
            }
            Ordering::Equal => {
                upmat[(lusize, lusize)] = Complex::<f64>::from(1.0);
            }
            Ordering::Greater => {
                let lcs = if csphase {
                    f64::from((-1i32).pow(
                        u32::try_from(mcomplex).expect("Unable to convert `mcomplex` to `u32`."),
                    ))
                } else {
                    1.0
                };
                // Python-equivalent:
                // upmat[-absmreal + l, mcomplex + l] = lcs * 1.0j / np.sqrt(2)
                // upmat[+absmreal + l, mcomplex + l] = lcs * 1.0 / np.sqrt(2)
                // mcomplex = absmreal
                upmat[(lusize - absmreal, lusize + absmreal)] =
                    lcs * Complex::<f64>::new(0.0, 1.0 / 2.0f64.sqrt());
                upmat[(lusize + absmreal, lusize + absmreal)] =
                    lcs * Complex::<f64>::new(1.0 / 2.0f64.sqrt(), 0.0);
            }
        }
    }

    // upmat is always in increasing-m order. We now permute, if required.
    if *pureorder != po_il {
        let perm = pureorder.get_perm_of(&po_il).expect(
            "Permutation to obtain `pureorder` from the increasing-m order could not be found.",
        );
        let image = perm.image();
        upmat.select(Axis(0), &image).select(Axis(1), &image)
    } else {
        upmat
    }
}

/// Obtains the matrix $`\boldsymbol{\Upsilon}^{(l)\dagger}`$ allowing real spherical harmonics
/// to be expressed as linear combinations of complex spherical harmonics.
///
/// Let $`Y_l^m`$ be a complex spherical harmonic of degree $`l`$ and order $`m`$.
/// Then, a real degree-$`l`$ spherical harmonic $`Y_{lm}`$ can be defined as
///
/// ```math
/// Y_{lm} =
///     \begin{cases}
///         \frac{\mathbb{i}}{\sqrt{2}}
///         \left(Y_l^{-\lvert m \rvert}
///               - \lambda'_{\mathrm{cs}} Y_l^{\lvert m \rvert}\right)
///         & \mathrm{if}\ m < 0 \\
///         Y_l^0 & \mathrm{if}\ m = 0 \\
///         \frac{1}{\sqrt{2}}
///         \left(Y_l^{-\lvert m \rvert}
///               + \lambda'_{\mathrm{cs}} Y_l^{\lvert m \rvert}\right)
///         & \mathrm{if}\ m > 0 \\
///     \end{cases}
/// ```
///
/// where $`\lambda'_{\mathrm{cs}} = (-1)^{\lvert m \rvert}`$ if the Condon--Shortley phase as
/// defined in [`complexc`] is employed for the complex spherical harmonics, and
/// $`\lambda'_{\mathrm{cs}} = 1`$ otherwise. The linear combination coefficients turn out to be
/// given by the elements of matrix $`\boldsymbol{\Upsilon}^{(l)\dagger}`$ of dimensions
/// $`(2l+1)\times(2l+1)`$ such that
///
/// ```math
///     Y_{lm} = \sum_{m'} Y_l^{m'} [\Upsilon^{(l)\dagger}]_{m'm}.
/// ```
///
/// It is obvious from the orthonormality of $`Y_{lm}`$ and $`Y_l^m`$ that
/// $`\boldsymbol{\Upsilon}^{(l)\dagger} = [\boldsymbol{\Upsilon}^{(l)}]^{-1}`$ where
/// $`\boldsymbol{\Upsilon}^{(l)}`$ is defined in [`sh_c2r_mat`].
///
/// # Arguments
///
/// * `l` - The spherical harmonic degree.
/// * `csphase` - If `true`, $`\lambda_{\mathrm{cs}}`$ is as defined in [`complexc`]. If `false`,
/// $`\lambda_{\mathrm{cs}} = 1`$.
/// * `pureorder` - A [`PureOrder`] struct giving the ordering of the components of the pure
/// Gaussians.
///
/// # Returns
///
/// The $`\boldsymbol{\Upsilon}^{(l)\dagger}`$ matrix.
pub fn sh_r2c_mat(l: u32, csphase: bool, pureorder: &PureOrder) -> Array2<Complex<f64>> {
    let mut mat = sh_c2r_mat(l, csphase, pureorder).t().to_owned();
    mat.par_mapv_inplace(|x| x.conj());
    mat
}

/// Obtains the matrix $`\mathbf{U}^{(l_{\mathrm{cart}}, l)}`$ containing linear combination
/// coefficients of Cartesian Gaussians in the expansion of a complex solid harmonic Gaussian,
/// *i.e.*, briefly,
///
/// ```math
/// \tilde{\mathbf{g}}^{\mathsf{T}}(l)
///     = \mathbf{g}^{\mathsf{T}}(l_{\mathrm{cart}})
///     \ \mathbf{U}^{(l_{\mathrm{cart}}, l)}.
/// ```
///
/// Let $`\tilde{g}(\alpha, \lambda, l_{\mathrm{cart}}, \mathbf{r})`$ be a complex solid harmonic
/// Gaussian as defined in Equation 1 of Schlegel, H. B. & Frisch, M. J. Transformation between
/// Cartesian and pure spherical harmonic Gaussians. *International Journal of Quantum Chemistry*
/// **54**, 83–87 (1995), [DOI](https://doi.org/10.1002/qua.560540202) with
/// $`n = l_{\mathrm{cart}}`$, and let $`g(\alpha, \lambda_{\mathrm{cart}}, \mathbf{r})`$ be a
/// Cartesian Gaussian as defined in Equation 2 of the above reference.
/// Here, $`\lambda`$ is a single index labelling a complex solid harmonic Gaussian of spherical
/// harmonic degree $`l`$ and order $`m_l`$, and $`\lambda_{\mathrm{cart}}`$ a single index
/// labelling a Cartesian Gaussian of degrees $`(l_x, l_y, l_z)`$ such that
/// $`l_x + l_y + l_z = l_{\mathrm{cart}}`$. We can then write
///
/// ```math
/// \tilde{g}(\alpha, \lambda, l_{\mathrm{cart}}, \mathbf{r})
/// = \sum_{\lambda_{\mathrm{cart}}}
///     g(\alpha, \lambda_{\mathrm{cart}}, \mathbf{r})
///     U^{(l_{\mathrm{cart}}, l)}_{\lambda_{\mathrm{cart}}\lambda}
/// ```
///
/// where $`U^{(l_{\mathrm{cart}}, l)}_{\lambda_{\mathrm{cart}}\lambda}`$
/// is given by the complex coefficients
///
/// ```math
/// U^{(l_{\mathrm{cart}}, l)}_{\lambda_{\mathrm{cart}}\lambda} =
///     c(l, m_l, l_{\mathrm{cart}}, l_x, l_y, l_z)
/// ```
///
/// defined in [`complexc`].
///
/// $`\mathbf{U}^{(l_{\mathrm{cart}}, l)}`$ has dimensions
/// $`\frac{1}{2}(l_{\mathrm{cart}}+1)(l_{\mathrm{cart}}+2) \times (2l+1)`$ and contains only
/// zero elements if $`l`$ and $`l_{\mathrm{cart}}`$ have different parities.
/// It can be verified that
/// $`\mathbf{V}^{(l,l_{\mathrm{cart}})}
/// \ \mathbf{U}^{(l_{\mathrm{cart}}, l)} = \boldsymbol{I}_{2l+1}`$, where
/// $`\mathbf{V}^{(l,l_{\mathrm{cart}})}`$ is given in [`sh_cart2cl_mat`].
///
/// # Arguments
///
/// * `lcart` - The total Cartesian degree for the Cartesian Gaussians and
///  also for the radial part of the solid harmonic Gaussian.
/// * `l` - The degree of the complex spherical harmonic factor in the solid
///  harmonic Gaussian.
/// * `cartorder` - A [`CartOrder`] struct giving the ordering of the components of the Cartesian
/// Gaussians.
/// * `csphase` - Set to `true` to use the Condon--Shortley phase in the calculations of the $`c`$
/// coefficients. See [`complexc`] for more details.
/// * `pureorder` - A [`PureOrder`] struct giving the ordering of the components of the pure
/// Gaussians.
///
/// # Returns
///
/// The $`\mathbf{U}^{(l_{\mathrm{cart}}, l)}`$ matrix.
pub fn sh_cl2cart_mat(
    lcart: u32,
    l: u32,
    cartorder: &CartOrder,
    csphase: bool,
    pureorder: &PureOrder,
) -> Array2<Complex<f64>> {
    assert_eq!(cartorder.lcart, lcart, "Mismatched Cartesian ranks.");
    assert_eq!(pureorder.lpure, l, "Mismatched pure ranks.");
    let mut umat = Array2::<Complex<f64>>::zeros((
        ((lcart + 1) * (lcart + 2)).div_euclid(2) as usize,
        2 * l as usize + 1,
    ));
    for (i, &m) in pureorder.iter().enumerate() {
        for (icart, &lcartqns) in cartorder.iter().enumerate() {
            umat[(icart, i)] = complexc((l, m), lcartqns, csphase);
        }
    }
    umat
}

/// Obtains the matrix $`\mathbf{V}^{(l, l_{\mathrm{cart}})}`$ containing linear combination
/// coefficients of complex solid harmonic Gaussians of a specific degree in the expansion of
/// Cartesian Gaussians, *i.e.*, briefly,
///
/// ```math
/// \mathbf{g}^{\mathsf{T}}(l_{\mathrm{cart}})
///     = \tilde{\mathbf{g}}^{\mathsf{T}}(l)
///     \ \mathbf{V}^{(l, l_{\mathrm{cart}})}.
/// ```
///
/// Let $`\tilde{g}(\alpha, \lambda, l_{\mathrm{cart}}, \mathbf{r})`$ be a complex solid harmonic
/// Gaussian as defined in Equation 1 of Schlegel, H. B. & Frisch, M. J. Transformation between
/// Cartesian and pure spherical harmonic Gaussians. *International Journal of Quantum Chemistry*
/// **54**, 83–87 (1995), [DOI](https://doi.org/10.1002/qua.560540202) with
/// $`n = l_{\mathrm{cart}}`$, and let $`g(\alpha, \lambda_{\mathrm{cart}}, \mathbf{r})`$ be a
/// Cartesian Gaussian as defined in Equation 2 of the above reference.  Here, $`\lambda`$ is a
/// single index labelling a complex solid harmonic Gaussian of spherical harmonic degree $`l`$
/// and order $`m_l`$, and $`\lambda_{\mathrm{cart}}`$ a single index labelling a Cartesian
/// Gaussian of degrees $`(l_x, l_y, l_z)`$ such that $`l_x + l_y + l_z = l_{\mathrm{cart}}`$.
/// We can then write
///
/// ```math
/// g(\alpha, \lambda_{\mathrm{cart}}, \mathbf{r})
/// = \sum_{\substack{\lambda\\ l \leq l_{\mathrm{cart}}}}
///     \tilde{g}(\alpha, \lambda, l_{\mathrm{cart}}, \mathbf{r})
///     V^{(l_{\mathrm{cart}})}_{\lambda\lambda_{\mathrm{cart}}}
/// ```
///
/// where $`V^{(l_{\mathrm{cart}})}_{\lambda\lambda_{\mathrm{cart}}}`$ is given by the inverse
/// complex coefficients
///
/// ```math
/// V^{(l_{\mathrm{cart}})}_{\lambda\lambda_{\mathrm{cart}}} =
///     c^{-1}(l_x, l_y, l_z, l, m_l, l_{\mathrm{cart}})
/// ```
///
/// defined in [`complexcinv`].
///
/// We can order the rows $`\lambda`$ of $`\mathbf{V}^{(l_{\mathrm{cart}})}`$ that have the same
/// $`l`$ into rectangular blocks of dimensions
/// $`(2l+1) \times \frac{1}{2}(l_{\mathrm{cart}}+1)(l_{\mathrm{cart}}+2)`$
/// which give contributions from complex solid harmonic Gaussians of a particular degree $`l`$.
/// We denote these blocks $`\mathbf{V}^{(l, l_{\mathrm{cart}})}`$.
/// They contain only zero elements if $`l`$ and $`l_{\mathrm{cart}}`$ have different parities.
///
/// # Arguments
///
/// * `l` - The degree of the complex spherical harmonic factor in the solid
///  harmonic Gaussian.
/// * `lcart` - The total Cartesian degree for the Cartesian Gaussians and
///  also for the radial part of the solid harmonic Gaussian.
/// * `cartorder` - A [`CartOrder`] struct giving the ordering of the components of the Cartesian
/// Gaussians.
/// * `csphase` - Set to `true` to use the Condon--Shortley phase in the calculations of the
/// $`c^{-1}`$ coefficients. See [`complexc`] and [`complexcinv`] for more details.
/// * `pureorder` - A [`PureOrder`] struct giving the ordering of the components of the pure
/// Gaussians.
///
/// # Returns
///
/// The $`\mathbf{V}^{(l, l_{\mathrm{cart}})}`$ block.
pub fn sh_cart2cl_mat(
    l: u32,
    lcart: u32,
    cartorder: &CartOrder,
    csphase: bool,
    pureorder: &PureOrder,
) -> Array2<Complex<f64>> {
    assert_eq!(pureorder.lpure, l, "Mismatched pure ranks.");
    assert_eq!(cartorder.lcart, lcart, "Mismatched Cartesian ranks.");
    let mut vmat = Array2::<Complex<f64>>::zeros((
        2 * l as usize + 1,
        ((lcart + 1) * (lcart + 2)).div_euclid(2) as usize,
    ));
    for (icart, &lcartqns) in cartorder.iter().enumerate() {
        for (i, &m) in pureorder.iter().enumerate() {
            vmat[(i, icart)] = complexcinv(lcartqns, (l, m), csphase);
        }
    }
    vmat
}

/// Obtain the matrix $`\mathbf{W}^{(l_{\mathrm{cart}}, l)}`$ containing linear combination
/// coefficients of Cartesian Gaussians in the expansion of a real solid harmonic Gaussian, *i.e.*,
/// briefly,
///
/// ```math
/// \bar{\mathbf{g}}^{\mathsf{T}}(l)
///     = \mathbf{g}^{\mathsf{T}}(l_{\mathrm{cart}})
///     \ \mathbf{W}^{(l_{\mathrm{cart}}, l)}.
/// ```
///
/// Let $`\bar{g}(\alpha, \lambda, l_{\mathrm{cart}}, \mathbf{r})`$ be
/// a real solid harmonic Gaussian defined in a similar manner to Equation 1 of Schlegel, H. B.
/// & Frisch, M. J. Transformation between Cartesian and pure spherical harmonic Gaussians.
/// *International Journal of Quantum Chemistry* **54**, 83–87 (1995),
/// [DOI](https://doi.org/10.1002/qua.560540202) with $`n = l_{\mathrm{cart}}`$ but with real
/// rather than complex spherical harmonic factors, and let
/// $`g(\alpha, \lambda_{\mathrm{cart}}, \mathbf{r})`$ be a Cartesian Gaussian as defined in
/// Equation 2 of the above reference. Here, $`\lambda`$ is a single index labelling a complex
/// solid harmonic Gaussian of spherical harmonic degree $`l`$ and order $`m_l`$, and
/// $`\lambda_{\mathrm{cart}}`$ a single index labelling a Cartesian Gaussian of degrees
/// $`(l_x, l_y, l_z)`$ such that $`l_x + l_y + l_z = l_{\mathrm{cart}}`$. We can then write
///
/// ```math
/// \bar{g}(\alpha, \lambda, l_{\mathrm{cart}}, \mathbf{r})
/// = \sum_{\lambda_{\mathrm{cart}}}
///     g(\alpha, \lambda_{\mathrm{cart}}, \mathbf{r})
///     W^{(l_{\mathrm{cart}}, l)}_{\lambda_{\mathrm{cart}}\lambda}.
/// ```
///
/// $`\mathbf{W}^{(l_{\mathrm{cart}}, l)}`$ is given by
///
/// ```math
/// \mathbf{W}^{(l_{\mathrm{cart}}, l)}
/// = \mathbf{U}^{(l_{\mathrm{cart}}, l)}
///   \boldsymbol{\Upsilon}^{(l)\dagger},
/// ```
///
/// where $`\boldsymbol{\Upsilon}^{(l)\dagger}`$ is defined in [`sh_r2c_mat`] and
/// $`\mathbf{U}^{(l_{\mathrm{cart}}, l)}`$ in [`sh_cl2cart_mat`].
/// $`\mathbf{W}^{(l_{\mathrm{cart}}, l)}`$ must be real.
/// $`\mathbf{W}^{(l_{\mathrm{cart}}, l)}`$ has dimensions
/// $`\frac{1}{2}(l_{\mathrm{cart}}+1)(l_{\mathrm{cart}}+2) \times (2l+1)`$ and contains only zero
/// elements if $`l`$ and $`l_{\mathrm{cart}}`$ have different parities. It can be verified that
/// $`\mathbf{X}^{(l,l_{\mathrm{cart}})}
/// \ \mathbf{W}^{(l_{\mathrm{cart}}, l)} = \boldsymbol{I}_{2l+1}`$, where
/// $`\mathbf{X}^{(l,l_{\mathrm{cart}})}`$ is given in
/// [`sh_cart2rl_mat`].
///
/// # Arguments
///
/// * lcart - The total Cartesian degree for the Cartesian Gaussians and
///  also for the radial part of the solid harmonic Gaussian.
/// * l - The degree of the complex spherical harmonic factor in the solid
///  harmonic Gaussian.
/// * cartorder - A [`CartOrder`] struct giving the ordering of the components of the Cartesian
/// Gaussians.
/// * `csphase` - Set to `true` to use the Condon--Shortley phase in the calculations of the $`c`$
/// coefficients. See [`complexc`] for more details.
/// * `pureorder` - A [`PureOrder`] struct giving the ordering of the components of the pure
/// Gaussians.
///
/// # Returns
///
/// The $`\mathbf{W}^{(l_{\mathrm{cart}}, l)}`$ matrix.
pub fn sh_rl2cart_mat(
    lcart: u32,
    l: u32,
    cartorder: &CartOrder,
    csphase: bool,
    pureorder: &PureOrder,
) -> Array2<f64> {
    assert_eq!(cartorder.lcart, lcart, "Mismatched Cartesian ranks.");
    assert_eq!(pureorder.lpure, l, "Mismatched pure ranks.");
    let upmatdagger = sh_r2c_mat(l, csphase, pureorder);
    let umat = sh_cl2cart_mat(lcart, l, cartorder, csphase, pureorder);
    let wmat = umat.dot(&upmatdagger);
    assert!(
        wmat.iter()
            .all(|x| approx::relative_eq!(x.im, 0.0, max_relative = 1e-7, epsilon = 1e-7)),
        "wmat is not entirely real."
    );
    wmat.map(|x| x.re)
}

/// Obtains the real matrix $`\mathbf{X}^{(l, l_{\mathrm{cart}})}`$ containing linear combination
/// coefficients of real solid harmonic Gaussians of a specific degree in the expansion of
/// Cartesian Gaussians, *i.e.*, briefly,
///
/// ```math
/// \mathbf{g}^{\mathsf{T}}(l_{\mathrm{cart}})
///     = \bar{\mathbf{g}}^{\mathsf{T}}(l)
///     \ \mathbf{X}^{(l, l_{\mathrm{cart}})}.
/// ```
///
/// Let $`\bar{g}(\alpha, \lambda, l_{\mathrm{cart}}, \mathbf{r})`$ be a real solid harmonic
/// Gaussian defined in a similar manner to Equation 1 of Schlegel, H. B. & Frisch, M. J.
/// Transformation between Cartesian and pure spherical harmonic Gaussians. *International
/// Journal of Quantum Chemistry* **54**, 83–87 (1995),
/// [DOI](https://doi.org/10.1002/qua.560540202)
/// with $`n = l_{\mathrm{cart}}`$, but with real rather than complex spherical harmonic factors,
/// and let $`g(\alpha, \lambda_{\mathrm{cart}}, \mathbf{r})`$ be a Cartesian Gaussian as defined
/// in Equation 2 of the above reference.  Here, $`\lambda`$ is a single index labelling a real
/// solid harmonic Gaussian of spherical harmonic degree $`l`$ and real order $`m_l`$, and
/// $`\lambda_{\mathrm{cart}}`$ a single index labelling a Cartesian Gaussian of degrees
/// $`(l_x, l_y, l_z)`$ such that $`l_x + l_y + l_z = l_{\mathrm{cart}}`$.
/// We can then write
///
/// ```math
/// g(\alpha, \lambda_{\mathrm{cart}}, \mathbf{r})
/// = \sum_{\substack{\lambda\\ l \leq l_{\mathrm{cart}}}}
///     \bar{g}(\alpha, \lambda, l_{\mathrm{cart}}, \mathbf{r})
///     X^{(l_{\mathrm{cart}})}_{\lambda\lambda_{\mathrm{cart}}}.
/// ```
///
/// We can order the rows $`\lambda`$ of $`\mathbf{X}^{(l_{\mathrm{cart}})}`$ that have the same
/// $`l`$ into rectangular blocks of dimensions
/// $`(2l+1) \times \frac{1}{2}(l_{\mathrm{cart}}+1)(l_{\mathrm{cart}}+2)`$.
/// We denote these blocks $`\mathbf{X}^{(l, l_{\mathrm{cart}})}`$ which are given by
///
/// ```math
/// \mathbf{X}^{(l, l_{\mathrm{cart}})}
/// = \boldsymbol{\Upsilon}^{(l)} \mathbf{V}^{(l, l_{\mathrm{cart}})},
/// ```
///
/// where $`\boldsymbol{\Upsilon}^{(l)}`$ is defined in
/// [`sh_c2r_mat`] and $`\boldsymbol{V}^{(l, l_{\mathrm{cart}})}`$ in [`sh_cart2cl_mat`].
/// $`\mathbf{X}^{(l, l_{\mathrm{cart}})}`$ must be real.
///
/// # Arguments
///
/// * `l` - The degree of the complex spherical harmonic factor in the solid
///  harmonic Gaussian.
/// * `lcart` - The total Cartesian degree for the Cartesian Gaussians and
///  also for the radial part of the solid harmonic Gaussian.
/// * `cartorder` - A [`CartOrder`] struct giving the ordering of the components of the Cartesian
/// Gaussians.
/// * `csphase` - Set to `true` to use the Condon--Shortley phase in the calculations of the
/// $`c^{-1}`$ coefficients. See [`complexc`] and [`complexcinv`] for more details.
/// * `pureorder` - A [`PureOrder`] struct giving the ordering of the components of the pure
/// Gaussians.
///
/// # Returns
///
/// The $`\mathbf{X}^{(l, l_{\mathrm{cart}})}`$ block.
pub fn sh_cart2rl_mat(
    l: u32,
    lcart: u32,
    cartorder: &CartOrder,
    csphase: bool,
    pureorder: &PureOrder,
) -> Array2<f64> {
    assert_eq!(cartorder.lcart, lcart, "Mismatched Cartesian ranks.");
    assert_eq!(pureorder.lpure, l, "Mismatched pure ranks.");
    let upmat = sh_c2r_mat(l, csphase, pureorder);
    let vmat = sh_cart2cl_mat(l, lcart, cartorder, csphase, pureorder);
    let xmat = upmat.dot(&vmat);
    assert!(
        xmat.iter()
            .all(|x| approx::relative_eq!(x.im, 0.0, max_relative = 1e-7, epsilon = 1e-7)),
        "xmat is not entirely real."
    );
    xmat.map(|x| x.re)
}

/// Returns a list of $`\mathbf{W}^{(l_{\mathrm{cart}}, l)}`$ for
/// $`l_{\mathrm{cart}} \ge l \ge 0`$ and $`l \equiv l_{\mathrm{cart}} \mod 2`$.
///
/// $`\mathbf{W}^{(l_{\mathrm{cart}}, l)}`$ is defined in [`sh_rl2cart_mat`].
///
/// # Arguments
///
/// * `lcart` - The total Cartesian degree for the Cartesian Gaussians and
///  also for the radial part of the solid harmonic Gaussian.
/// * `cartorder` - A [`CartOrder`] struct giving the ordering of the components of the Cartesian
/// Gaussians.
/// * `csphase` - Set to `true` to use the Condon--Shortley phase in the calculations of the
/// $`c`$ coefficients. See [`complexc`] for more details.
/// * `pureorder` - A closure to generate a [`PureOrder`] struct giving the ordering of the
/// components of the pure Gaussians for a particular value of `l`.
///
/// # Returns
///
/// A vector of $`\mathbf{W}^{(l_{\mathrm{cart}}, l)}`$ matrices with
/// $`l_{\mathrm{cart}} \ge l \ge 0`$ and $`l \equiv l_{\mathrm{cart}} \mod 2`$ in decreasing
/// $`l`$ order.
pub fn sh_r2cart(
    lcart: u32,
    cartorder: &CartOrder,
    csphase: bool,
    pureorder: fn(u32) -> PureOrder,
) -> Vec<Array2<f64>> {
    assert_eq!(cartorder.lcart, lcart, "Mismatched Cartesian ranks.");
    let lrange = if lcart.rem_euclid(2) == 0 {
        #[allow(clippy::range_plus_one)]
        (0..lcart + 1).step_by(2).rev()
    } else {
        #[allow(clippy::range_plus_one)]
        (1..lcart + 1).step_by(2).rev()
    };
    lrange
        .map(|l| sh_rl2cart_mat(lcart, l, cartorder, csphase, &pureorder(l)))
        .collect()
}

/// Returns a list of $`\mathbf{X}^{(l, l_{\mathrm{cart}})}`$ for
/// $`l_{\mathrm{cart}} \ge l \ge 0`$ and $`l \equiv l_{\mathrm{cart}} \mod 2`$.
///
/// $`\mathbf{X}^{(l, l_{\mathrm{cart}})}`$ is defined in [`sh_cart2rl_mat`].
///
/// # Arguments
///
/// * `lcart` - The total Cartesian degree for the Cartesian Gaussians and
///  also for the radial part of the solid harmonic Gaussian.
/// * `cartorder` - A [`CartOrder`] struct giving the ordering of the components of the Cartesian
/// Gaussians.
/// * `csphase` - Set to `true` to use the Condon--Shortley phase in the calculations of the
/// $`c^{-1}`$ coefficients. See [`complexc`] and [`complexcinv`] for more details.
/// * `pureorder` - A closure to generate a [`PureOrder`] struct giving the ordering of the
/// components of the pure Gaussians for a particular value of `l`.
///
/// # Returns
///
/// A vector of $`\mathbf{X}^{(l, l_{\mathrm{cart}})}`$ matrices with
/// $`l_{\mathrm{cart}} \ge l \ge 0`$ and $`l \equiv l_{\mathrm{cart}} \mod 2`$ in decreasing
/// $`l`$ order.
pub fn sh_cart2r(
    lcart: u32,
    cartorder: &CartOrder,
    csphase: bool,
    pureorder: fn(u32) -> PureOrder,
) -> Vec<Array2<f64>> {
    assert_eq!(cartorder.lcart, lcart, "Mismatched Cartesian ranks.");
    let lrange = if lcart.rem_euclid(2) == 0 {
        #[allow(clippy::range_plus_one)]
        (0..lcart + 1).step_by(2).rev()
    } else {
        #[allow(clippy::range_plus_one)]
        (1..lcart + 1).step_by(2).rev()
    };
    lrange
        .map(|l| sh_cart2rl_mat(l, lcart, cartorder, csphase, &pureorder(l)))
        .collect()
}
