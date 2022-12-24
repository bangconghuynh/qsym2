use std::cmp;

use factorial::Factorial;
use nalgebra::Vector3;
use ndarray::{array, Array2, Axis};
use num::{BigUint, Complex, Zero};
use num_traits::ToPrimitive;

// #[cfg(test)]
// #[path = "sh_rotation_3d_tests.rs"]
// mod sh_rotation_3d_tests;

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

    let i = Complex::<f64>::i();
    (-1.0 * i * (alpha * (mdashi as f64 - 0.5) + gamma * (mi as f64 - 0.5))).exp() * d
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
fn dmat_euler(euler_angles: (f64, f64, f64), increasingm: bool) -> Array2<Complex<f64>> {
    let mut dmat = Array2::<Complex<f64>>::zeros((2, 2));
    for mdashi in (0..2) {
        for mi in (0..2) {
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
/// \hat{R}(\alpha, \beta, \gamma) \ket{\tfrac{1}{2}m}
/// = \sum_{m'} \ket{\tfrac{1}{2}m'} D^{(1/2)}_{m'm}(\alpha, \beta, \gamma).
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
/// The matrix $`\mathbf{D}^{(1/2)}(\phi, \hat{\mathbf{n}})`$.
fn dmat_angleaxis(angle: f64, axis: Vector3<f64>, increasingm: bool) -> Array2<Complex<f64>> {
    let normalised_axis = axis.normalize();
    let nx = normalised_axis.x;
    let ny = normalised_axis.y;
    let nz = normalised_axis.z;

    let i = Complex::<f64>::i();
    let mut dmat = array![
        [
            (angle / 2.0).cos() + i * nz * (angle / 2.0).sin(),
            (ny - i * nx) * (angle / 2.0).sin()
        ],
        [
            -(ny + i * nx) * (angle / 2.0).sin(),
            (angle / 2.0).cos() - i * nz * (angle / 2.0).sin(),
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
    let prefactor = (-i * (alpha * mdash + gamma * m)).exp();

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
        .unwrap();

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
                .powi(<usize as TryInto<i32>>::try_into(2 * t - mdashi + mi).unwrap());

        acc + (-1.0f64).powi(<usize as TryInto<i32>>::try_into(t).unwrap())
            * (num / den)
            * trigfactor
    });

    prefactor * d
}
