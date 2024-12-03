//! Three-dimensional rotations of spinors.

use std::cmp;
use std::fmt;

use approx;
use factorial::Factorial;
use nalgebra::Vector3;
use ndarray::{array, Array2, Axis};
use num::{BigUint, Complex, Zero};
use num_traits::ToPrimitive;
use serde::{Deserialize, Serialize};

use crate::auxiliary::geometry::normalise_rotation_angle;

#[cfg(test)]
#[path = "spinor_rotation_3d_tests.rs"]
mod spinor_rotation_3d_tests;

// ================
// Enum definitions
// ================

/// Enumerated type to manage spin constraints and spin space information.
#[derive(Clone, Debug, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub enum SpinConstraint {
    /// Variant for restricted spin constraint: the spatial parts of all spin spaces are identical.
    /// The associated value is the number of spin spaces.
    Restricted(u16),

    /// Variant for unrestricted spin constraint: the spatial parts of different spin spaces are
    /// different, but spin collinearity is maintained. The associated values are the number of spin
    /// spaces (*i.e.* the number of different spatial parts that are handled separately) and a
    /// boolean indicating if the spin spaces are arranged in increasing $`m`$ order.
    Unrestricted(u16, bool),

    /// Variant for generalised spin constraint: the spatial parts of different spin spaces are
    /// different, and no spin collinearity is imposed. The associated values are the number of spin
    /// spaces and a boolean indicating if the spin spaces are arranged in increasing $`m`$ order.
    Generalised(u16, bool),
}

impl SpinConstraint {
    /// Returns the total number of units of consideration.
    ///
    /// A 'unit' of consideration is commonly known as a 'spin channel' or 'spin space'.
    pub fn nunits(&self) -> u16 {
        match self {
            Self::Restricted(nspins) => *nspins,
            Self::Unrestricted(nspins, _) => *nspins,
            Self::Generalised(_, _) => 1,
        }
    }

    /// Returns the number of spin spaces per 'unit' of consideration.
    ///
    /// A 'unit' of consideration is commonly known as a 'spin channel' or 'spin space'.
    pub fn nspins_per_unit(&self) -> u16 {
        match self {
            Self::Restricted(_) => 1,
            Self::Unrestricted(_, _) => 1,
            Self::Generalised(nspins, _) => *nspins,
        }
    }

    /// Returns the total number of spin spaces.
    pub fn nspins(&self) -> u16 {
        match self {
            Self::Restricted(nspins) => *nspins,
            Self::Unrestricted(nspins, _) => *nspins,
            Self::Generalised(nspins, _) => *nspins,
        }
    }
}

impl fmt::Display for SpinConstraint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Restricted(nspins) => write!(
                f,
                "Restricted ({} spin {})",
                nspins,
                if *nspins == 1 { "space" } else { "spaces" }
            ),
            Self::Unrestricted(nspins, increasingm) => write!(
                f,
                "Unrestricted ({} spin {}, {} m)",
                nspins,
                if *nspins == 1 { "space" } else { "spaces" },
                if *increasingm {
                    "increasing"
                } else {
                    "decreasing"
                }
            ),
            Self::Generalised(nspins, increasingm) => write!(
                f,
                "Generalised ({} spin {}, {} m)",
                nspins,
                if *nspins == 1 { "space" } else { "spaces" },
                if *increasingm {
                    "increasing"
                } else {
                    "decreasing"
                }
            ),
        }
    }
}

// =========
// Functions
// =========

/// Returns an element in the Wigner rotation matrix for $`j = 1/2`$ defined by
///
/// ```math
///     \hat{R}(\alpha, \beta, \gamma) \ket{\tfrac{1}{2}m}
///     = \sum_{m'} \ket{\tfrac{1}{2}m'} D^{(1/2)}_{m'm}(\alpha, \beta, \gamma).
/// ```
///
/// # Arguments
///
/// * `mdashi` - Index for $`m'`$ given by $`m'+\tfrac{1}{2}`$.
/// * `mi` - Index for $`m`$ given by $`m+\tfrac{1}{2}`$.
/// * `euler_angles` - A triplet of Euler angles $`(\alpha, \beta, \gamma)`$ in radians, following
/// the Whitaker convention, *i.e.* $`z_2-y-z_1`$ (extrinsic rotations).
///
/// # Returns
///
/// The element $`D^{(1/2)}_{m'm}(\alpha, \beta, \gamma)`$.
fn dmat_euler_element(mdashi: usize, mi: usize, euler_angles: (f64, f64, f64)) -> Complex<f64> {
    assert!(mdashi == 0 || mdashi == 1, "mdashi can only be 0 or 1.");
    assert!(mi == 0 || mi == 1, "mi can only be 0 or 1.");
    let (alpha, beta, gamma) = euler_angles;
    let d = if (mi, mdashi) == (1, 1) {
        // m = 1/2, mdash = 1/2
        (beta / 2.0).cos()
    } else if (mi, mdashi) == (1, 0) {
        // m = 1/2, mdash = -1/2
        (beta / 2.0).sin()
    } else if (mi, mdashi) == (0, 1) {
        // m = -1/2, mdash = 1/2
        -(beta / 2.0).sin()
    } else if (mi, mdashi) == (0, 0) {
        // m = -1/2, mdash = -1/2
        (beta / 2.0).cos()
    } else {
        panic!("Invalid mi and/or mdashi.");
    };

    let alpha_basic = alpha.rem_euclid(2.0 * std::f64::consts::PI);
    let gamma_basic = gamma.rem_euclid(2.0 * std::f64::consts::PI);
    let i = Complex::<f64>::i();
    let mut prefactor = (-i
        * (alpha_basic
            * (mdashi
                .to_f64()
                .unwrap_or_else(|| panic!("Unable to convert `{mdashi}` to `f64`."))
                - 0.5)
            + gamma_basic
                * (mi
                    .to_f64()
                    .unwrap_or_else(|| panic!("Unable to convert `{mi}` to `f64`."))
                    - 0.5)))
        .exp();

    // Half-integer j = 1/2; double-group behaviours possible.
    let alpha_double = approx::relative_eq!(
        alpha.div_euclid(2.0 * std::f64::consts::PI).rem_euclid(2.0),
        1.0,
        epsilon = 1e-14,
        max_relative = 1e-14
    );
    let gamma_double = approx::relative_eq!(
        gamma.div_euclid(2.0 * std::f64::consts::PI).rem_euclid(2.0),
        1.0,
        epsilon = 1e-14,
        max_relative = 1e-14
    );
    if alpha_double != gamma_double {
        prefactor *= -1.0;
    }
    prefactor * d
}

/// Returns the Wigner rotation matrix for $`j = 1/2`$ whose elements are defined by
///
/// ```math
/// \hat{R}(\alpha, \beta, \gamma) \ket{\tfrac{1}{2}m}
/// = \sum_{m'} \ket{\tfrac{1}{2}m'} D^{(1/2)}_{m'm}(\alpha, \beta, \gamma).
/// ```
///
/// # Arguments
///
/// * `euler_angles` - A triplet of Euler angles $`(\alpha, \beta, \gamma)`$ in radians, following
/// the Whitaker convention, *i.e.* $`z_2-y-z_1`$ (extrinsic rotations).
/// * `increasingm` - If `true`, the rows and columns of $`\mathbf{D}^{(1/2)}`$ are
/// arranged in increasing order of $`m_l = -l, \ldots, l`$. If `false`, the order is reversed:
/// $`m_l = l, \ldots, -l`$. The recommended default is `false`, in accordance with convention.
///
/// # Returns
///
/// The matrix $`\mathbf{D}^{(1/2)}(\alpha, \beta, \gamma)`$.
#[must_use]
pub fn dmat_euler(euler_angles: (f64, f64, f64), increasingm: bool) -> Array2<Complex<f64>> {
    let mut dmat = Array2::<Complex<f64>>::zeros((2, 2));
    for mdashi in 0..2 {
        for mi in 0..2 {
            dmat[(mdashi, mi)] = dmat_euler_element(mdashi, mi, euler_angles);
        }
    }
    if !increasingm {
        dmat.invert_axis(Axis(0));
        dmat.invert_axis(Axis(1));
    }
    dmat
}

/// Returns the Wigner rotation matrix for $`j = 1/2`$ whose elements are defined by
///
/// ```math
/// \hat{R}(\phi\hat{\mathbf{n}}) \ket{\tfrac{1}{2}m}
/// = \sum_{m'} \ket{\tfrac{1}{2}m'} D^{(1/2)}_{m'm}(\phi\hat{\mathbf{n}}).
/// ```
///
/// The parametrisation of $`\mathbf{D}^{(1/2)}`$ by $`\phi`$ and $`\hat{\mathbf{n}}`$ is given
/// in (**4**-9.12) of Altmann, S. L. Rotations, Quaternions, and Double Groups. (Dover
/// Publications, Inc., 2005).
///
/// # Arguments
///
/// * `angle` - The angle $`\phi`$ of the rotation in radians. A positive rotation is an
/// anticlockwise rotation when looking down `axis`.
/// * `axis` - A space-fixed vector defining the axis of rotation. The supplied vector will be
/// normalised.
/// * `increasingm` - If `true`, the rows and columns of $`\mathbf{D}^{(1/2)}`$ are
/// arranged in increasing order of $`m_l = -l, \ldots, l`$. If `false`, the order is reversed:
/// $`m_l = l, \ldots, -l`$. The recommended default is `false`, in accordance with convention.
///
/// # Returns
///
/// The matrix $`\mathbf{D}^{(1/2)}(\phi\hat{\mathbf{n}})`$.
#[must_use]
pub fn dmat_angleaxis(angle: f64, axis: Vector3<f64>, increasingm: bool) -> Array2<Complex<f64>> {
    let normalised_axis = axis.normalize();
    let nx = normalised_axis.x;
    let ny = normalised_axis.y;
    let nz = normalised_axis.z;

    let i = Complex::<f64>::i();
    let double_angle = angle.rem_euclid(4.0 * std::f64::consts::PI);
    let mut dmat = array![
        [
            (double_angle / 2.0).cos() + i * nz * (double_angle / 2.0).sin(),
            (ny - i * nx) * (double_angle / 2.0).sin()
        ],
        [
            -(ny + i * nx) * (double_angle / 2.0).sin(),
            (double_angle / 2.0).cos() - i * nz * (double_angle / 2.0).sin(),
        ]
    ];
    if !increasingm {
        dmat.invert_axis(Axis(0));
        dmat.invert_axis(Axis(1));
    }
    dmat
}

/// Returns an element in the Wigner rotation matrix for an integral or half-integral
/// $`j`$, defined by
/// ```math
/// \hat{R}(\alpha, \beta, \gamma) \ket{jm}
/// = \sum_{m'} \ket{jm'} D^{(j)}_{m'm}(\alpha, \beta, \gamma)
/// ```
/// where $`-\pi \le \alpha \le \pi`$, $`0 \le \beta \le \pi`$, $`-\pi \le \gamma \le \pi`$.
///
/// The explicit expression for the elements of $`\mathbf{D}^{(j)}(\alpha, \beta, \gamma)`$
/// is given in Professor Anthony Stone's graduate lecture notes on Angular Momentum at the
/// University of Cambridge in 2006.
///
/// # Arguments
///
/// * `twoj` - Two times the angular momentum $`2j`$. If this is even, $`j`$ is integral; otherwise,
/// $`j`$ is half-integral.
/// * `mdashi` - Index for $`m'`$ given by $`m'+\tfrac{1}{2}`$.
/// * `mi` - Index for $`m`$ given by $`m+\tfrac{1}{2}`$.
/// * `euler_angles` - A triplet of Euler angles $`(\alpha, \beta, \gamma)`$ in radians, following
/// the Whitaker convention, *i.e.* $`z_2-y-z_1`$ (extrinsic rotations).
///
/// # Returns
///
/// The element $`D^{(j)}_{m'm}(\alpha, \beta, \gamma)`$.
#[allow(clippy::too_many_lines)]
pub fn dmat_euler_gen_element(
    twoj: u32,
    mdashi: usize,
    mi: usize,
    euler_angles: (f64, f64, f64),
) -> Complex<f64> {
    assert!(
        mdashi <= twoj as usize,
        "`mdashi` must be between 0 and {twoj} (inclusive).",
    );
    assert!(
        mi <= twoj as usize,
        "`mi` must be between 0 and {twoj} (inclusive).",
    );
    let (alpha, beta, gamma) = euler_angles;
    let j = f64::from(twoj) / 2.0;
    let mdash = mdashi
        .to_f64()
        .unwrap_or_else(|| panic!("Unable to convert `{mdashi}` to `f64`."))
        - j;
    let m = mi
        .to_f64()
        .unwrap_or_else(|| panic!("Unable to convert `{mi}` to `f64`."))
        - j;

    let i = Complex::<f64>::i();
    let alpha_basic = alpha.rem_euclid(2.0 * std::f64::consts::PI);
    let gamma_basic = gamma.rem_euclid(2.0 * std::f64::consts::PI);
    let mut prefactor = (-i * (alpha_basic * mdash + gamma_basic * m)).exp();

    if twoj % 2 != 0 {
        // Half-integer j; double-group behaviours possible.
        let alpha_double = approx::relative_eq!(
            alpha.div_euclid(2.0 * std::f64::consts::PI).rem_euclid(2.0),
            1.0,
            epsilon = 1e-14,
            max_relative = 1e-14
        );
        let gamma_double = approx::relative_eq!(
            gamma.div_euclid(2.0 * std::f64::consts::PI).rem_euclid(2.0),
            1.0,
            epsilon = 1e-14,
            max_relative = 1e-14
        );
        if alpha_double != gamma_double {
            prefactor *= -1.0;
        }
    }

    // tmax = min(int(j + mdash), int(j - m))
    // j + mdash = mdashi
    // j - m = twoj - mi
    let tmax = cmp::min(mdashi, twoj as usize - mi);

    // tmin = max(0, int(mdash - m))
    // mdash - m = mdashi - mi
    let tmin = if mdashi > mi { mdashi - mi } else { 0 };

    let d = (tmin..=tmax).fold(Complex::<f64>::zero(), |acc, t| {
        // j - m = twoj - mi
        // j - mdash = twoj - mdashi
        let num = (BigUint::from(mdashi)
            .checked_factorial()
            .unwrap_or_else(|| panic!("Unable to compute the factorial of {mdashi}."))
            * BigUint::from(twoj as usize - mdashi)
                .checked_factorial()
                .unwrap_or_else(|| {
                    panic!(
                        "Unable to compute the factorial of {}.",
                        twoj as usize - mdashi
                    )
                })
            * BigUint::from(mi)
                .checked_factorial()
                .unwrap_or_else(|| panic!("Unable to compute the factorial of {mi}."))
            * BigUint::from(twoj as usize - mi)
                .checked_factorial()
                .unwrap_or_else(|| {
                    panic!("Unable to compute the factorial of {}.", twoj as usize - mi)
                }))
        .to_f64()
        .expect("Unable to convert a `BigUint` value to `f64`.")
        .sqrt();

        // t <= j + mdash ==> j + mdash - t = mdashi - t >= 0
        // t <= j - m ==> j - m - t = twoj - mi - t >= 0
        // t >= 0
        // t >= mdash - m ==> t - (mdash - m) = t + mi - mdashi >= 0
        let den = (BigUint::from(mdashi - t)
            .checked_factorial()
            .unwrap_or_else(|| panic!("Unable to compute the factorial of {}.", mdashi - t))
            * BigUint::from(twoj as usize - mi - t)
                .checked_factorial()
                .unwrap_or_else(|| {
                    panic!(
                        "Unable to compute the factorial of {}.",
                        twoj as usize - mi - t
                    )
                })
            * BigUint::from(t)
                .checked_factorial()
                .unwrap_or_else(|| panic!("Unable to compute the factorial of {t}."))
            * BigUint::from(t + mi - mdashi)
                .checked_factorial()
                .unwrap_or_else(|| {
                    panic!("Unable to compute the factorial of {}.", t + mi - mdashi)
                }))
        .to_f64()
        .expect("Unable to convert a `BigUint` value to `f64`.");

        let trigfactor = (beta / 2.0).cos().powi(
            i32::try_from(twoj as usize + mdashi - mi - 2 * t).unwrap_or_else(|_| {
                panic!(
                    "Unable to convert `{}` to `i32`.",
                    twoj as usize + mdashi - mi - 2 * t
                )
            }),
        ) * (beta / 2.0).sin().powi(
            i32::try_from(2 * t + mi - mdashi).unwrap_or_else(|_| {
                panic!("Unable to convert `{}` to `i32`.", 2 * t + mi - mdashi)
            }),
        );

        if t % 2 == 0 {
            acc + (num / den) * trigfactor
        } else {
            acc - (num / den) * trigfactor
        }
    });

    prefactor * d
}

/// Returns the Wigner rotation matrix in the Euler-angle parametrisation for any integral or
/// half-integral $`j`$ whose elements are defined by
/// ```math
/// \hat{R}(\alpha, \beta, \gamma) \ket{jm}
/// = \sum_{m'} \ket{jm'} D^{(j)}_{m'm}(\alpha, \beta, \gamma)
/// ```
/// and given in [`dmat_euler_gen_element`], where $`-\pi \le \alpha \le \pi`$,
/// $`0 \le \beta \le \pi`$, $`-\pi \le \gamma \le \pi`$.
///
/// # Arguments
///
/// * `twoj` - Two times the angular momentum $`2j`$. If this is even, $`j`$ is integral; otherwise,
/// $`j`$ is half-integral.
/// * `euler_angles` - A triplet of Euler angles $`(\alpha, \beta, \gamma)`$ in radians, following
/// the Whitaker convention, *i.e.* $`z_2-y-z_1`$ (extrinsic rotations).
/// * `increasingm` - If `true`, the rows and columns of $`\mathbf{D}^{(j)}`$ are
/// arranged in increasing order of $`m_l = -l, \ldots, l`$. If `false`, the order is reversed:
/// $`m_l = l, \ldots, -l`$. The recommended default is `false`, in accordance with convention.
///
/// # Returns
///
/// The matrix $`\mathbf{D}^{(j)}(\alpha, \beta, \gamma)`$.
#[must_use]
pub fn dmat_euler_gen(
    twoj: u32,
    euler_angles: (f64, f64, f64),
    increasingm: bool,
) -> Array2<Complex<f64>> {
    let dim = twoj as usize + 1;
    let mut dmat = Array2::<Complex<f64>>::zeros((dim, dim));
    for mdashi in 0..dim {
        for mi in 0..dim {
            dmat[(mdashi, mi)] = dmat_euler_gen_element(twoj, mdashi, mi, euler_angles);
        }
    }
    if !increasingm {
        dmat.invert_axis(Axis(0));
        dmat.invert_axis(Axis(1));
    }
    dmat
}

/// Returns the Wigner rotation matrix in the angle-axis parametrisation for any integral or
/// half-integral $`j`$  whose elements are defined by
///
/// ```math
/// \hat{R}(\phi\hat{\mathbf{n}}) \ket{jm}
/// = \sum_{m'} \ket{jm'} D^{(j)}_{m'm}(\phi\hat{\mathbf{n}}).
/// ```
///
/// # Arguments
///
/// * `twoj` - Two times the angular momentum $`2j`$. If this is even, $`j`$ is integral; otherwise,
/// $`j`$ is half-integral.
/// * `angle` - The angle $`\phi`$ of the rotation in radians. A positive rotation is an
/// anticlockwise rotation when looking down `axis`.
/// * `axis` - A space-fixed vector defining the axis of rotation. The supplied vector will be
/// normalised.
/// * `increasingm` - If `true`, the rows and columns of $`\mathbf{D}^{(1/2)}`$ are
/// arranged in increasing order of $`m_l = -l, \ldots, l`$. If `false`, the order is reversed:
/// $`m_l = l, \ldots, -l`$. The recommended default is `false`, in accordance with convention.
///
/// # Returns
///
/// The matrix $`\mathbf{D}^{(j)}(\phi\hat{\mathbf{n}})`$.
#[must_use]
pub fn dmat_angleaxis_gen_double(
    twoj: u32,
    angle: f64,
    axis: Vector3<f64>,
    increasingm: bool,
) -> Array2<Complex<f64>> {
    let euler_angles = angleaxis_to_euler_double(angle, axis);
    dmat_euler_gen(twoj, euler_angles, increasingm)
}

/// Converts an angle and axis of rotation to Euler angles using the equations in Section
/// (**3**-5.4) in Altmann, S. L. Rotations, Quaternions, and Double Groups. (Dover
/// Publications, Inc., 2005), but with an extended range,
///
/// ```math
/// 0 \le \alpha \le 2\pi, \quad
/// 0 \le \beta \le \pi, \quad
/// 0 \le \gamma \le 4\pi,
/// ```
///
/// such that all angle-axis parametrisations of $`\phi\hat{\mathbf{n}}`$ for
/// $`0 \le \phi \le 4 \pi`$ are mapped to unique triplets of $`(\alpha, \beta, \gamma)`$,
/// as explained in Fan, P.-D., Chen, J.-Q., Mcaven, L. & Butler, P. Unique Euler angles and
/// self-consistent multiplication tables for double point groups. *International Journal of
/// Quantum Chemistry* **75**, 1–9 (1999),
/// [DOI](https://doi.org/10.1002/(SICI)1097-461X(1999)75:1<1::AID-QUA1>3.0.CO;2-V).
///
/// When $`\beta = 0`$, only the sum $`\alpha+\gamma`$ is determined. Likewise, when
/// $`\beta = \pi`$, only the difference $`\alpha-\gamma`$ is determined. We thus set
/// $`\alpha = 0`$ in these cases and solve for $`\gamma`$ without changing the nature of the
/// results.
///
/// # Arguments
///
/// * `angle` - The angle $`\phi`$ of the rotation in radians. A positive rotation is an
/// anticlockwise rotation when looking down `axis`.
/// * `axis` - A space-fixed vector defining the axis of rotation $`\hat{\mathbf{n}}`$. The supplied
/// vector will be normalised.
///
/// # Returns
///
/// The tuple containing the Euler angles $`(\alpha, \beta, \gamma)`$ in radians, following the
/// Whitaker convention.
fn angleaxis_to_euler_double(angle: f64, axis: Vector3<f64>) -> (f64, f64, f64) {
    let normalised_axis = axis.normalize();
    let nx = normalised_axis.x;
    let ny = normalised_axis.y;
    let nz = normalised_axis.z;

    let double_angle = angle.rem_euclid(4.0 * std::f64::consts::PI);
    let basic_angle = angle.rem_euclid(2.0 * std::f64::consts::PI);
    let double = approx::relative_eq!(
        angle.div_euclid(2.0 * std::f64::consts::PI).rem_euclid(2.0),
        1.0,
        epsilon = 1e-14,
        max_relative = 1e-14
    );

    let cosbeta = 1.0 - 2.0 * (nx.powi(2) + ny.powi(2)) * (basic_angle / 2.0).sin().powi(2);
    let cosbeta = if cosbeta.abs() > 1.0 {
        // Numerical errors can cause cosbeta to be outside [-1, 1].
        approx::assert_relative_eq!(cosbeta.abs(), 1.0, epsilon = 1e-14, max_relative = 1e-14);
        cosbeta.round()
    } else {
        cosbeta
    };

    // acos gives 0 <= beta <= pi.
    let beta = cosbeta.acos();

    let (alpha, gamma) =
        if approx::relative_ne!(cosbeta.abs(), 1.0, epsilon = 1e-14, max_relative = 1e-14) {
            // cosbeta != 1 or -1, beta != 0 or pi
            // alpha and gamma are given by Equations (**3**-5.4) to (**3**-5.10)
            // in Altmann, S. L. Rotations, Quaternions, and Double Groups. (Dover Publications,
            // Inc., 2005).
            // These equations yield the same alpha and gamma for phi and phi+2pi.
            // We therefore account for double-group behaviours separately.
            let num_alpha =
                -nx * basic_angle.sin() + 2.0 * ny * nz * (basic_angle / 2.0).sin().powi(2);
            let den_alpha =
                ny * basic_angle.sin() + 2.0 * nx * nz * (basic_angle / 2.0).sin().powi(2);
            let alpha = num_alpha
                .atan2(den_alpha)
                .rem_euclid(2.0 * std::f64::consts::PI);

            let num_gamma =
                nx * basic_angle.sin() + 2.0 * ny * nz * (basic_angle / 2.0).sin().powi(2);
            let den_gamma =
                ny * basic_angle.sin() - 2.0 * nx * nz * (basic_angle / 2.0).sin().powi(2);
            let gamma_raw = num_gamma.atan2(den_gamma);
            let gamma = if double {
                (gamma_raw + 2.0 * std::f64::consts::PI).rem_euclid(4.0 * std::f64::consts::PI)
            } else {
                gamma_raw.rem_euclid(4.0 * std::f64::consts::PI)
            };

            (alpha, gamma)
        } else if approx::relative_eq!(cosbeta, 1.0, epsilon = 1e-14, max_relative = 1e-14) {
            // cosbeta == 1, beta == 0
            // cos(0.5(alpha+gamma)) = cos(0.5phi)
            // We set alpha == 0 by convention.
            // We then set gamma = phi mod (4*pi).
            (0.0, double_angle)
        } else {
            // cosbeta == -1, beta == pi
            // sin(0.5phi) must be non-zero, otherwise cosbeta == 1, a
            // contradiction.
            // sin(0.5(alpha-gamma)) = -nx*sin(0.5phi)
            // cos(0.5(alpha-gamma)) = +ny*sin(0.5phi)
            // We set alpha == 0 by convention.
            // gamma then lies in [-2pi, 2pi].
            // We obtain the same gamma for phi and phi+2pi.
            // We therefore account for double-group behaviours separately.
            let gamma_raw = 2.0 * nx.atan2(ny);
            let gamma = if double {
                (gamma_raw + 2.0 * std::f64::consts::PI).rem_euclid(4.0 * std::f64::consts::PI)
            } else {
                gamma_raw.rem_euclid(4.0 * std::f64::consts::PI)
            };

            (0.0, gamma)
        };

    (alpha, beta, gamma)
}

/// Returns the Wigner rotation matrix in the angle-axis parametrisation for any integral or
/// half-integral $`j`$  whose elements are defined by
///
/// ```math
/// \hat{R}(\phi\hat{\mathbf{n}}) \ket{jm}
/// = \sum_{m'} \ket{jm'} D^{(j)}_{m'm}(\phi\hat{\mathbf{n}}),
/// ```
///
/// where the angle of rotation is ensured to be in the range $`[-\pi, \pi]`$. In other words, for
/// half-odd-integer $`j`$, this function only returns Wigner rotation matrices corresponding to
/// three-dimensional rotations connected to the identity via a homotopy path of class 0.
///
/// # Arguments
///
/// * `twoj` - Two times the angular momentum $`2j`$. If this is even, $`j`$ is integral; otherwise,
/// $`j`$ is half-integral.
/// * `angle` - The angle $`\phi`$ of the rotation in radians. A positive rotation is an
/// anticlockwise rotation when looking down `axis`.
/// * `axis` - A space-fixed vector defining the axis of rotation. The supplied vector will be
/// normalised.
/// * `increasingm` - If `true`, the rows and columns of $`\mathbf{D}^{(1/2)}`$ are
/// arranged in increasing order of $`m_l = -l, \ldots, l`$. If `false`, the order is reversed:
/// $`m_l = l, \ldots, -l`$. The recommended default is `false`, in accordance with convention.
///
/// # Returns
///
/// The matrix $`\mathbf{D}^{(j)}(\phi\hat{\mathbf{n}})`$.
#[must_use]
pub fn dmat_angleaxis_gen_single(
    twoj: u32,
    angle: f64,
    axis: Vector3<f64>,
    increasingm: bool,
) -> Array2<Complex<f64>> {
    let euler_angles = angleaxis_to_euler_single(angle, axis, 1e-14);
    dmat_euler_gen(twoj, euler_angles, increasingm)
}

/// Converts an angle and axis of rotation to Euler angles using the equations in Section
/// (**3**-5.4) in Altmann, S. L. Rotations, Quaternions, and Double Groups. (Dover
/// Publications, Inc., 2005) such that
///
/// ```math
/// -\pi \le \alpha \le \pi, \quad
/// 0 \le \beta \le \pi, \quad
/// -\pi \le \gamma \le \pi.
/// ```
///
/// When $`\beta = 0`$, only the sum $`\alpha+\gamma`$ is determined. Likewise, when
/// $`\beta = \pi`$, only the difference $`\alpha-\gamma`$ is determined. We thus set
/// $`\alpha = 0`$ in these cases and solve for $`\gamma`$ without changing the nature of the
/// results.
///
/// # Arguments
///
/// * `angle` - The angle $`\phi`$ of the rotation in radians. A positive rotation is an
/// anticlockwise rotation when looking down `axis`.
/// * `axis` - A space-fixed vector defining the axis of rotation $`\hat{\mathbf{n}}`$. The supplied
/// vector will be normalised.
///
/// # Returns
///
/// The tuple containing the Euler angles $`(\alpha, \beta, \gamma)`$ in radians, following the
/// Whitaker convention.
fn angleaxis_to_euler_single(angle: f64, axis: Vector3<f64>, thresh: f64) -> (f64, f64, f64) {
    let normalised_axis = axis.normalize();
    let nx = normalised_axis.x;
    let ny = normalised_axis.y;
    let nz = normalised_axis.z;

    let (normalised_angle, _) = normalise_rotation_angle(angle, thresh);

    let cosbeta = 1.0 - 2.0 * (nx.powi(2) + ny.powi(2)) * (normalised_angle / 2.0).sin().powi(2);
    let cosbeta = if cosbeta.abs() > 1.0 {
        // Numerical errors can cause cosbeta to be outside [-1, 1].
        approx::assert_relative_eq!(cosbeta.abs(), 1.0, epsilon = thresh, max_relative = thresh);
        cosbeta.round()
    } else {
        cosbeta
    };

    // 0 <= beta <= pi.
    let beta = cosbeta.acos();

    let (alpha, gamma) =
        if approx::relative_ne!(cosbeta.abs(), 1.0, epsilon = thresh, max_relative = thresh) {
            // cosbeta != 1 or -1, beta != 0 or pi
            // alpha and gamma are given by Equations (**3**-5.4) to (**3**-5.10)
            // in Altmann, S. L. Rotations, Quaternions, and Double Groups. (Dover Publications,
            // Inc., 2005).
            let num_alpha = -nx * normalised_angle.sin()
                + 2.0 * ny * nz * (normalised_angle / 2.0).sin().powi(2);
            let den_alpha = ny * normalised_angle.sin()
                + 2.0 * nx * nz * (normalised_angle / 2.0).sin().powi(2);
            // -pi <= alpha <= pi
            let alpha = num_alpha.atan2(den_alpha);

            let num_gamma = nx * normalised_angle.sin()
                + 2.0 * ny * nz * (normalised_angle / 2.0).sin().powi(2);
            let den_gamma = ny * normalised_angle.sin()
                - 2.0 * nx * nz * (normalised_angle / 2.0).sin().powi(2);
            // -pi <= gamma <= pi
            let gamma = num_gamma.atan2(den_gamma);

            (alpha, gamma)
        } else if approx::relative_eq!(cosbeta, 1.0, epsilon = thresh, max_relative = thresh) {
            // cosbeta == 1, beta == 0
            // cos(0.5(alpha+gamma)) = cos(0.5phi)
            // We set alpha == 0 by convention.
            // We then set gamma = phi.
            (0.0, normalised_angle)
        } else {
            // cosbeta == -1, beta == pi
            // sin(0.5phi) must be non-zero, otherwise cosbeta == 1, a contradiction.
            // sin(0.5(alpha-gamma)) = -nx*sin(0.5phi)
            // cos(0.5(alpha-gamma)) = +ny*sin(0.5phi)
            // We set alpha == 0 by convention.
            // gamma then lies in [-pi, pi].
            let gamma = 2.0 * nx.atan2(ny);

            (0.0, gamma)
        };

    (alpha, beta, gamma)
}
