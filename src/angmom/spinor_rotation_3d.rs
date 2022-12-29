use std::cmp;

use approx;
use factorial::Factorial;
use nalgebra::Vector3;
use ndarray::{array, Array2, Axis};
use num::{BigUint, Complex, Zero};
use num_traits::ToPrimitive;

#[cfg(test)]
#[path = "spinor_rotation_3d_tests.rs"]
mod spinor_rotation_3d_tests;

/// Returns an element in the Wigner rotation matrix for $`j = 1/2`$ defined by
///
/// ```math
///     \hat{R}(\alpha, \beta, \gamma) \ket{\tfrac{1}{2}m}
///     = \sum_{m'} \ket{\tfrac{1}{2}m'} D^{(1/2)}_{m'm}(\alpha, \beta, \gamma).
/// ```
///
/// # Arguments
///
/// * mdashi - Index for $`m'`$ given by $`m'+\tfrac{1}{2}`$.
/// * mi - Index for $`m`$ given by $`m+\tfrac{1}{2}`$.
/// * euler_angles - A triplet of Euler angles $`(\alpha, \beta, \gamma)`$ in radians, following
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
    let mut prefactor =
        (-i * (alpha_basic * (mdashi as f64 - 0.5) + gamma_basic * (mi as f64 - 0.5))).exp();

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
/// * euler_angles - A triplet of Euler angles $`(\alpha, \beta, \gamma)`$ in radians, following
/// the Whitaker convention, *i.e.* $`z_2-y-z_1`$ (extrinsic rotations).
/// * increasingm - If `true`, the rows and columns of $`\mathbf{D}^{(1/2)}`$ are
/// arranged in increasing order of $`m_l = -l, \ldots, l`$. If `false`, the order is reversed:
/// $`m_l = l, \ldots, -l`$. The recommended default is `false`, in accordance with convention.
///
/// # Returns
///
/// The matrix $`\mathbf{D}^{(1/2)}(\alpha, \beta, \gamma)`$.
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
/// * angle - The angle $`\phi`$ of the rotation in radians. A positive rotation is an
/// anticlockwise rotation when looking down `axis`.
/// * axis - A space-fixed vector defining the axis of rotation. The supplied vector will be
/// normalised.
/// * increasingm - If `true`, the rows and columns of $`\mathbf{D}^{(1/2)}`$ are
/// arranged in increasing order of $`m_l = -l, \ldots, l`$. If `false`, the order is reversed:
/// $`m_l = l, \ldots, -l`$. The recommended default is `false`, in accordance with convention.
///
/// # Returns
///
/// The matrix $`\mathbf{D}^{(1/2)}(\phi\hat{\mathbf{n}})`$.
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
///
/// ```math
/// \hat{R}(\alpha, \beta, \gamma) \ket{jm}
/// = \sum_{m'} \ket{jm'} D^{(j)}_{m'm}(\alpha, \beta, \gamma).
/// ```
///
/// The explicit expression for the elements of $`\mathbf{D}^{(1/2)}(\alpha, \beta, \gamma)`$
/// is given in Professor Anthony Stone's graduate lecture notes on Angular Momentum at the
/// University of Cambridge in 2006.
///
/// # Arguments
///
/// * twoj - Two times the angular momentum $`2j`$. If this is even, $`j`$ is integral; otherwise,
/// $`j`$ is half-integral.
/// * mdashi - Index for $`m'`$ given by $`m'+\tfrac{1}{2}`$.
/// * mi - Index for $`m`$ given by $`m+\tfrac{1}{2}`$.
/// * euler_angles - A triplet of Euler angles $`(\alpha, \beta, \gamma)`$ in radians, following
/// the Whitaker convention, *i.e.* $`z_2-y-z_1`$ (extrinsic rotations).
///
/// # Returns
///
/// The element $`D^{(j)}_{m'm}(\alpha, \beta, \gamma)`$.
fn dmat_euler_gen_element(
    twoj: u32,
    mdashi: usize,
    mi: usize,
    euler_angles: (f64, f64, f64),
) -> Complex<f64> {
    assert!(
        mdashi <= twoj as usize,
        "mdashi must be between 0 and {} (inclusive).",
        twoj
    );
    assert!(
        mi <= twoj as usize,
        "mi must be between 0 and {} (inclusive).",
        twoj
    );
    let (alpha, beta, gamma) = euler_angles;
    let j = twoj as f64 / 2.0;
    let mdash = mdashi as f64 - j;
    let m = mi as f64 - j;

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
        let num = (BigUint::from(mdashi).checked_factorial().unwrap()
            * BigUint::from(twoj as usize - mdashi)
                .checked_factorial()
                .unwrap()
            * BigUint::from(mi).checked_factorial().unwrap()
            * BigUint::from(twoj as usize - mi)
                .checked_factorial()
                .unwrap())
        .to_f64()
        .unwrap()
        .sqrt();

        // t <= j + mdash ==> j + mdash - t = mdashi - t >= 0
        // t <= j - m ==> j - m - t = twoj - mi - t >= 0
        // t >= 0
        // t >= mdash - m ==> t - (mdash - m) = t + mi - mdashi >= 0
        let den = (BigUint::from(mdashi - t).checked_factorial().unwrap()
            * BigUint::from(twoj as usize - mi - t)
                .checked_factorial()
                .unwrap()
            * BigUint::from(t).checked_factorial().unwrap()
            * BigUint::from(t + mi - mdashi).checked_factorial().unwrap())
        .to_f64()
        .unwrap();

        let trigfactor = (beta / 2.0)
            .cos()
            .powi(<usize as TryInto<i32>>::try_into(twoj as usize + mdashi - mi - 2 * t).unwrap())
            * (beta / 2.0)
                .sin()
                .powi(<usize as TryInto<i32>>::try_into(2 * t + mi - mdashi).unwrap());

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
///
/// ```math
/// \hat{R}(\alpha, \beta, \gamma) \ket{jm}
/// = \sum_{m'} \ket{jm'} D^{(j)}_{m'm}(\alpha, \beta, \gamma).
/// ```
///
/// and given in [`dmat_euler_gen_element`].
///
/// # Arguments
///
/// * twoj - Two times the angular momentum $`2j`$. If this is even, $`j`$ is integral; otherwise,
/// $`j`$ is half-integral.
/// * euler_angles - A triplet of Euler angles $`(\alpha, \beta, \gamma)`$ in radians, following
/// the Whitaker convention, *i.e.* $`z_2-y-z_1`$ (extrinsic rotations).
/// * increasingm - If `true`, the rows and columns of $`\mathbf{D}^{(j)}`$ are
/// arranged in increasing order of $`m_l = -l, \ldots, l`$. If `false`, the order is reversed:
/// $`m_l = l, \ldots, -l`$. The recommended default is `false`, in accordance with convention.
///
/// # Returns
///
/// The matrix $`\mathbf{D}^{(j)}(\alpha, \beta, \gamma)`$.
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
/// \hat{R}(phi\hat{\mathbf{n}}) \ket{jm}
/// = \sum_{m'} \ket{jm'} D^{(j)}_{m'm}(phi\hat{\mathbf{n}}).
/// ```
///
/// # Arguments
///
/// * twoj - Two times the angular momentum $`2j`$. If this is even, $`j`$ is integral; otherwise,
/// $`j`$ is half-integral.
/// * angle - The angle $`\phi`$ of the rotation in radians. A positive rotation is an
/// anticlockwise rotation when looking down `axis`.
/// * axis - A space-fixed vector defining the axis of rotation. The supplied vector will be
/// normalised.
/// * increasingm - If `true`, the rows and columns of $`\mathbf{D}^{(1/2)}`$ are
/// arranged in increasing order of $`m_l = -l, \ldots, l`$. If `false`, the order is reversed:
/// $`m_l = l, \ldots, -l`$. The recommended default is `false`, in accordance with convention.
///
/// # Returns
///
/// The matrix $`\mathbf{D}^{(j)}(\phi\hat{\mathbf{n}})`$.
pub fn dmat_angleaxis_gen(
    twoj: u32,
    angle: f64,
    axis: Vector3<f64>,
    increasingm: bool,
) -> Array2<Complex<f64>> {
    let euler_angles = angleaxis_to_euler(angle, axis);
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
/// $`0 \le \phi \le 4\pi` are mapped to unique triplets of $`(\alpha, \beta, \gamma)`$,
/// as explained in Fan, P.-D., Chen, J.-Q., Mcaven, L. & Butler, P. Unique Euler angles and
/// self-consistent multiplication tables for double point groups. *International Journal of
/// Quantum Chemistry* **75**, 1–9 (1999),
///[DOI](https://doi.org/10.1002/(SICI)1097-461X(1999)75:1<1::AID-QUA1>3.0.CO;2-V).
///
/// When $`\beta = 0`$, only the sum $`\alpha+\gamma`$ is determined. Likewise, when
/// $`\beta = \pi`$, only the difference $`\alpha-\gamma`$ is determined. We thus set
/// $`\alpha = 0`$ in these cases and solve for $`\gamma` without changing the nature of the
/// results.
///
/// # Arguments
///
/// * angle - The angle $`\phi`$ of the rotation in radians. A positive rotation is an
/// anticlockwise rotation when looking down `axis`.
/// * axis - A space-fixed vector defining the axis of rotation $`\hat{\mathbf{n}}`$. The supplied
/// vector will be normalised.
///
/// # Returns
///
/// The tuple containing the Euler angles $`(\alpha, \beta, \gamma)`$ in radians, following the
/// Whitaker convention.
fn angleaxis_to_euler(angle: f64, axis: Vector3<f64>) -> (f64, f64, f64) {
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
            let gamma = if !double {
                gamma_raw.rem_euclid(4.0 * std::f64::consts::PI)
            } else {
                (gamma_raw + 2.0 * std::f64::consts::PI).rem_euclid(4.0 * std::f64::consts::PI)
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
            let gamma = if !double {
                gamma_raw.rem_euclid(4.0 * std::f64::consts::PI)
            } else {
                (gamma_raw + 2.0 * std::f64::consts::PI).rem_euclid(4.0 * std::f64::consts::PI)
            };

            (0.0, gamma)
        };

    (alpha, beta, gamma)
}
