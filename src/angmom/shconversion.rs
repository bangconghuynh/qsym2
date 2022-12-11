use factorial::Factorial;
use num::{BigUint, Complex};
use num_traits::{cast::ToPrimitive, Zero};

#[cfg(test)]
#[path = "shconversion_tests.rs"]
mod shconversion_tests;

/// Calculates the number of combinations of `n` things taken `r` at a time (signed arguments).
///
/// If $`n < 0`$ or $`r < 0`$ or $`r > n`$, `0` is returned.
///
/// # Arguments
///
/// n - Number of things.
/// r - Number of elements taken.
///
/// # Returns
///
/// The number of combinations.
fn comb(n: i32, r: i32) -> BigUint {
    if n < 0 || r < 0 || r > n {
        BigUint::zero()
    } else {
        let nu = n as u32;
        let ru = r as u32;
        (nu - ru + 1..=nu).product::<BigUint>() / BigUint::from(ru).checked_factorial().unwrap()
    }
}

/// Calculates the number of combinations of `n` things taken `r` at a time (unsigned arguments).
///
/// If $`r > n`$, `0` is returned.
///
/// # Arguments
///
/// n - Number of things.
/// r - Number of elements taken.
///
/// # Returns
///
/// The number of combinations.
fn combu(nu: u32, ru: u32) -> BigUint {
    if ru > nu {
        BigUint::zero()
    } else {
        (nu - ru + 1..=nu).product::<BigUint>() / BigUint::from(ru).checked_factorial().unwrap()
    }
}

/// Calculates the number of permutations of `n` things taken `r` at a time (signed arguments).
///
/// If $`n < 0`$ or $`r < 0`$ or $`r > n`$, `0` is returned.
///
/// # Arguments
///
/// n - Number of things.
/// r - Number of elements taken.
///
/// # Returns
///
/// The number of permutations.
fn perm(n: i32, r: i32) -> BigUint {
    if n < 0 || r < 0 || r > n {
        BigUint::zero()
    } else {
        let nu = n as u32;
        let ru = r as u32;
        (nu - ru + 1..=nu).product::<BigUint>()
    }
}

/// Calculates the number of permutations of `n` things taken `r` at a time (unsigned arguments).
///
/// If $`r > n`$, `0` is returned.
///
/// # Arguments
///
/// n - Number of things.
/// r - Number of elements taken.
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
/// * n - The non-negative exponent of the radial part of the solid harmonic Gaussian.
/// * alpha - The coefficient on the exponent of the Gaussian term.
///
/// # Returns
///
/// The normalisation constant $`\tilde{N}(n, \alpha)`$.
fn norm_sph_gaussian(n: u32, alpha: f64) -> f64 {
    let num = (BigUint::from(2u64).pow(2 * n + 3)
        * BigUint::from(n as u64 + 1).checked_factorial().unwrap())
    .to_f64()
    .unwrap()
        * alpha.powf(n as f64 + 1.5);
    let den = BigUint::from(2 * n as u64 + 2)
        .checked_factorial()
        .unwrap()
        .to_f64()
        .unwrap()
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
/// * lcartqns - A tuple of $`(l_x, l_y, l_z)`$ specifying the non-negative exponents of
/// the Cartesian components of the Cartesian Gaussian.
/// * alpha - The coefficient on the exponent of the Gaussian term.
///
/// # Returns
///
/// The normalisation constant $`N(l_x, l_y, l_z, \alpha)`$.
fn norm_cart_gaussian(lcartqns: (u32, u32, u32), alpha: f64) -> f64 {
    let (lx, ly, lz) = lcartqns;
    let lcart = lx + ly + lz;
    let num = (BigUint::from(2u32).pow(2 * lcart)
        * BigUint::from(lx).checked_factorial().unwrap()
        * BigUint::from(ly).checked_factorial().unwrap()
        * BigUint::from(lz).checked_factorial().unwrap())
    .to_f64()
    .unwrap()
        * alpha.powf(lcart as f64 + 1.5);
    let den = (BigUint::from(2 * lx).checked_factorial().unwrap()
        * BigUint::from(2 * ly).checked_factorial().unwrap()
        * BigUint::from(2 * lz).checked_factorial().unwrap())
    .to_f64()
    .unwrap()
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
/// * lpureqns - A tuple of $`(l, m_l)`$ specifying the quantum numbers for the spherical
/// harmonic component of the solid harmonic Gaussian.
/// * lcartqns - A tuple of $`(l_x, l_y, l_z)`$ specifying the exponents of the Cartesian
/// components of the Cartesian Gaussian.
/// * csphase - If `true`, the Condon--Shortley phase will be used as defined above.
/// If `false`, this phase will be set to unity.
///
/// # Returns
///
/// The complex factor $`c(l, m_l, l_{\mathrm{cart}}, l_x, l_y, l_z)`$.
fn complexc(lpureqns: (u32, i32), lcartqns: (u32, u32, u32), csphase: bool) -> Complex<f64> {
    let (l, m) = lpureqns;
    assert!(m.abs() as u32 <= l, "m must be between -l and l (inclusive).");
    let (lx, ly, lz) = lcartqns;
    let lcart = lx + ly + lz;
    let dl = lcart as i32 - l as i32;
    if dl % 2 != 0 {
        return Complex::<f64>::zero();
    }

    let num = ((2 * l + 1) * (l - m.abs() as u32).checked_factorial().unwrap()) as f64;
    let den = 4.0 * std::f64::consts::PI * (l + m.abs() as u32).checked_factorial().unwrap() as f64;
    let mut prefactor =
        1.0 / ((2u32.pow(l) * l.checked_factorial().unwrap()) as f64) * (num / den).sqrt();
    if csphase && m > 0 {
        prefactor *= (-1i32).pow(m as u32) as f64;
    }
    let ntilde = norm_sph_gaussian(lcart, 1.0);
    let n = norm_cart_gaussian(lcartqns, 1.0);

    let si =
        (0..=((l - m.abs() as u32).div_euclid(2))).fold(Complex::<f64>::zero(), |acc_si, i| {
            let ifactor = combu(l, i).to_f64().unwrap()
                * ((-1i32).pow(i) * (2 * l - 2 * i).checked_factorial().unwrap() as i32) as f64
                / (l - m.abs() as u32 - 2 * i).checked_factorial().unwrap() as f64;
            let sp = (0..=(m.abs() as u32)).fold(Complex::<f64>::zero(), |acc_sp, p| {
                let pfactor = if m > 0 {
                    combu(m.abs() as u32, p).to_f64().unwrap()
                        * Complex::<f64>::i().powu(m.abs() as u32 - p)
                } else {
                    combu(m.abs() as u32, p).to_f64().unwrap()
                        * (-1.0 * Complex::<f64>::i()).powu(m.abs() as u32 - p)
                };
                let sq = (0..=(dl.div_euclid(2))).fold(Complex::<f64>::zero(), |acc_sq, q| {
                    let jq_num = (lx + ly) as i32 - 2 * q as i32 - m.abs();
                    if jq_num.rem_euclid(2) == 0 {
                        let jq = jq_num.div_euclid(2);
                        let qfactor = (comb(dl.div_euclid(2), q as i32) * comb(i as i32, jq))
                            .to_f64()
                            .unwrap();
                        let sk = (0..=jq).fold(Complex::<f64>::zero(), |acc_sk, k| {
                            let tpk_num = lx as i32 - p as i32 - 2 * k;
                            if tpk_num.rem_euclid(2) == 0 {
                                let tpk = tpk_num.div_euclid(2);
                                let kfactor = (comb(q, tpk) * comb(jq, k)).to_f64().unwrap();
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
