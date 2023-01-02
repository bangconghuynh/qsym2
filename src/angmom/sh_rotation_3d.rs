use std::cmp::Ordering;

use approx;
use nalgebra::{Rotation3, Unit, Vector3};
use ndarray::{Array2, Axis, ShapeBuilder};
use num_traits::ToPrimitive;

#[cfg(test)]
#[path = "sh_rotation_3d_tests.rs"]
mod sh_rotation_3d_tests;

/// Returns the generalised Kronecker delta $`\delta_{ij}`$ for any $`i`$ and $`j`$ that have a
/// partial equivalence relation.
///
/// # Returns
///
/// `0` if $`i \ne j`$, `1` if $`i = j`$.
fn kdelta<T: PartialEq>(i: &T, j: &T) -> u8 {
    u8::from(i == j)
}

/// Returns the function $`_iP^l_{\mu m'}`$ as defined in Table 2 of Ivanic, J. & Ruedenberg, K.
/// Rotation Matrices for Real Spherical Harmonics. Direct Determination by Recursion.
/// *The Journal of Physical Chemistry* **100**, 9099–9100 (1996),
/// [DOI](https://doi.org/10.1021/jp953350u).
///
/// # Arguments
///
/// * `i` - The index $`i`$ satisfying $`-1 \le i \le 1`$.
/// * `l` - The spherical harmonic order $`l \ge 2`$.
/// * `mu` - The index $`\mu`$ satisfying $`-l+1 \le \mu \le l-1`$.
/// * `mdash` - The index $`m'`$ satisfying $`-l \le m' \le l`$.
/// * `rmat` - The representation matrix of the transformation of interest in the basis of
/// coordinate *functions* $`(y, z, x)`$, which are isosymmetric to the real spherical harmonics
/// $`(Y_{1, -1}, Y_{1, 0}, Y_{1, 1})`$.
/// * `rlm1` - The representation matrix of the transformation of interest in the basis of the real
/// spherical harmonics $`Y_{l-1, m}`$ ordered by increasing $`m`$.
///
/// # Returns
///
/// The value of $`_iP^l_{\mu m'}`$.
fn func_p(i: i8, l: u32, mu: i64, mdash: i64, rmat: &Array2<f64>, rlm1: &Array2<f64>) -> f64 {
    assert!(i.abs() <= 1, "`i` must be between -1 and 1 (inclusive).");
    assert!(l >= 2, "`l` must be at least 2.");
    let li64 = i64::from(l);
    assert!(
        mu.abs() < li64,
        "Index `mu` = {} lies outside [{}, {}].",
        mu,
        -(li64) + 1,
        l - 1
    );
    assert!(
        mdash.abs() <= li64,
        "Index `mdash` = {} lies outside [-{}, {}].",
        mdash,
        l,
        l
    );
    assert_eq!(rmat.shape(), &[3, 3], "`rmat` must be a 3 × 3 matrix.");
    assert_eq!(
        rlm1.shape(),
        &[2 * l as usize - 1, 2 * l as usize - 1],
        "`rlm1` must be a {} × {} matrix.",
        2 * l as usize - 1,
        2 * l as usize - 1
    );

    let ii = usize::try_from(i + 1).expect("Unable to convert `i + 1` to `usize`.");
    let mui =
        usize::try_from(mu + (li64 - 1)).expect("Unable to convert `mu + (l - 1)` to `usize`.");
    let mdashi = usize::try_from(mdash + li64).expect("Unable to convert `mdash + l` to `usize`.");
    let lusize = usize::try_from(l).expect("Unable to convert `l` to `usize`.");
    if mdash == li64 {
        // Easier-to-read expression:
        // R[i + 1, 1 + 1] * Rlm1[mu + (l - 1), l - 1 + (l - 1)]
        //  - R[i + 1, -1 + 1] * Rlm1[mu + (l - 1), -l + 1 + (l - 1)]
        rmat[(ii, 2)] * rlm1[(mui, 2 * (lusize - 1))] - rmat[(ii, 0)] * rlm1[(mui, 0)]
    } else if mdash == -li64 {
        // Easier-to-read expression:
        // R[i + 1, 1 + 1] * Rlm1[mu + (l - 1), -l + 1 + (l - 1)]
        //  + R[i + 1, -1 + 1] * Rlm1[mu + (l - 1), l - 1 + (l - 1)]
        rmat[(ii, 2)] * rlm1[(mui, 0)] + rmat[(ii, 0)] * rlm1[(mui, 2 * (lusize - 1))]
    } else {
        // Easier-to-read expression:
        // R[i + 1, 0 + 1] * Rlm1[mu + (l - 1), mdash + (l - 1)]
        rmat[(ii, 1)] * rlm1[(mui, mdashi - 1)]
    }
}

/// Returns the function $`U^l_{mm'}`$ as defined in Table 2 of Ivanic, J. & Ruedenberg, K.
/// Rotation Matrices for Real Spherical Harmonics. Direct Determination by Recursion.
/// *The Journal of Physical Chemistry* **100**, 9099–9100 (1996),
/// [DOI](https://doi.org/10.1021/jp953350u).
///
/// # Arguments
///
/// * `l` - The spherical harmonic order $`l \ge 2`$.
/// * `m` - The index $`m`$ satisfying $`-l+1 \le \mu \le l-1`$.
/// * `mdash` - The index $`m'`$ satisfying $`-l \le m' \le l`$.
/// * `rmat` - The representation matrix of the transformation of interest in the basis of
/// coordinate *functions* $`(y, z, x)`$, which are isosymmetric to the real spherical harmonics
/// $`(Y_{1, -1}, Y_{1, 0}, Y_{1, 1})`$.
/// *  rlm1 - The representation matrix of the transformation of interest in the basis of the real
/// spherical harmonics $`Y_{l-1, m}`$ ordered by increasing $`m`$.
///
/// # Returns
///
/// The value of $`U^l_{mm'}`$.
fn func_u(l: u32, m: i64, mdash: i64, rmat: &Array2<f64>, rlm1: &Array2<f64>) -> f64 {
    func_p(0, l, m, mdash, rmat, rlm1)
}

/// Returns the function $`V^l_{mm'}`$ as defined in Table 2 of Ivanic, J. & Ruedenberg, K.
/// Rotation Matrices for Real Spherical Harmonics. Direct Determination by Recursion.
/// *The Journal of Physical Chemistry* **100**, 9099–9100 (1996),
/// [DOI](https://doi.org/10.1021/jp953350u).
///
/// # Arguments
///
/// * `l` - The spherical harmonic order $`l \ge 2`$.
/// * `m` - The index $`m`$ satisfying $`-l \le m \le l`$.
/// * `mdash` - The index $`m'`$ satisfying $`-l \le m' \le l`$.
/// * `rmat` - The representation matrix of the transformation of interest in the basis of
/// coordinate *functions* $`(y, z, x)`$, which are isosymmetric to the real spherical harmonics
/// $`(Y_{1, -1}, Y_{1, 0}, Y_{1, 1})`$.
/// * `rlm1` - The representation matrix of the transformation of interest in the basis of the real
/// spherical harmonics $`Y_{l-1, m}`$ ordered by increasing $`m`$.
///
/// # Returns
///
/// The value of $`V^l_{mm'}`$.
fn func_v(l: u32, m: i64, mdash: i64, rmat: &Array2<f64>, rlm1: &Array2<f64>) -> f64 {
    assert!(l >= 2, "`l` must be at least 2.");
    let li64 = i64::from(l);
    assert!(
        m.abs() <= li64,
        "Index `m` = {} lies outside [-{}, {}].",
        m,
        l,
        l
    );
    assert!(
        mdash.abs() <= li64,
        "Index `mdash` = {} lies outside [-{}, {}].",
        mdash,
        l,
        l
    );
    assert_eq!(rmat.shape(), &[3, 3], "`rmat` must be a 3 × 3 matrix.");
    assert_eq!(
        rlm1.shape(),
        &[2 * l as usize - 1, 2 * l as usize - 1],
        "`rlm1` must be a {} × {} matrix.",
        2 * l as usize - 1,
        2 * l as usize - 1
    );

    match m.cmp(&0) {
        Ordering::Greater => {
            func_p(1, l, m - 1, mdash, rmat, rlm1) * (f64::from(1 + kdelta(&m, &1))).sqrt()
                - func_p(-1, l, -m + 1, mdash, rmat, rlm1) * (f64::from(1 - kdelta(&m, &1)))
        }
        Ordering::Less => {
            func_p(1, l, m + 1, mdash, rmat, rlm1) * (f64::from(1 - kdelta(&m, &(-1))))
                + func_p(-1, l, -m - 1, mdash, rmat, rlm1)
                    * (f64::from(1 + kdelta(&m, &(-1)))).sqrt()
        }
        Ordering::Equal => {
            func_p(1, l, 1, mdash, rmat, rlm1) + func_p(-1, l, -1, mdash, rmat, rlm1)
        }
    }
}

/// Returns the function $`W^l_{mm'}`$ as defined in Table 2 of Ivanic, J. & Ruedenberg, K.
/// Rotation Matrices for Real Spherical Harmonics. Direct Determination by Recursion.
/// *The Journal of Physical Chemistry* **100**, 9099–9100 (1996),
/// [DOI](https://doi.org/10.1021/jp953350u).
///
/// # Arguments
///
/// * `l` - The spherical harmonic order $`l \ge 2`$.
/// * `m` - The index $`m`$ satisfying $`-l+2 \le m \le l-2`$ and $`m \ne 0`$.
/// * `mdash` - The index $`m'`$ satisfying $`-l \le m' \le l`$.
/// * `rmat` - The representation matrix of the transformation of interest in the basis of
/// coordinate *functions* $`(y, z, x)`$, which are isosymmetric to the real spherical harmonics
/// $`(Y_{1, -1}, Y_{1, 0}, Y_{1, 1})`$.
/// * `rlm1` - The representation matrix of the transformation of interest in the basis of the real
/// spherical harmonics $`Y_{l-1, m}`$ ordered by increasing $`m`$.
///
/// # Returns
///
/// The value of $`W^l_{mm'}`$.
fn func_w(l: u32, m: i64, mdash: i64, rmat: &Array2<f64>, rlm1: &Array2<f64>) -> f64 {
    assert!(l >= 2, "`l` must be at least 2.");
    let li64 = i64::from(l);
    assert!(
        m.abs() <= li64 - 2,
        "Index `m` = {} lies outside [{}, {}].",
        m,
        -li64 + 2,
        li64 - 2
    );
    assert_ne!(m, 0, "`m` cannot be zero.");
    assert!(
        mdash.abs() <= li64,
        "Index `mdash` = {} lies outside [-{}, {}].",
        mdash,
        l,
        l
    );
    assert_eq!(rmat.shape(), &[3, 3], "`rmat` must be a 3 × 3 matrix.");
    assert_eq!(
        rlm1.shape(),
        &[2 * l as usize - 1, 2 * l as usize - 1],
        "`rlm1` must be a {} × {} matrix.",
        2 * l as usize - 1,
        2 * l as usize - 1
    );
    if m > 0 {
        func_p(1, l, m + 1, mdash, rmat, rlm1) + func_p(-1, l, -m - 1, mdash, rmat, rlm1)
    } else {
        func_p(1, l, m - 1, mdash, rmat, rlm1) - func_p(-1, l, -m + 1, mdash, rmat, rlm1)
    }
}

/// Returns the coefficient $`u^l_{mm'}`$ as defined in Table 1 of Ivanic, J. & Ruedenberg, K.
/// Rotation Matrices for Real Spherical Harmonics. Direct Determination by Recursion.
/// *The Journal of Physical Chemistry* **100**, 9099–9100 (1996),
/// [DOI](https://doi.org/10.1021/jp953350u).
///
/// # Arguments
///
/// * `l` - The spherical harmonic order $`l`$.
/// * `m` - The index $`m`$ satisfying $`-l \le m \le l`$.
/// * `mdash` - The index $`m'`$ satisfying $`-l \le m' \le l`$.
///
/// # Returns
///
/// The value of $`u^l_{mm'}`$.
fn coeff_u(l: u32, m: i64, mdash: i64) -> f64 {
    let li64 = i64::from(l);
    assert!(
        m.abs() <= li64,
        "Index `m` = {} lies outside [-{}, {}].",
        m,
        l,
        l
    );
    assert!(
        mdash.abs() <= li64,
        "Index `mdash` = {} lies outside [-{}, {}].",
        mdash,
        l,
        l
    );

    let num = (li64 + m) * (li64 - m);
    let den = if mdash.abs() < li64 {
        (li64 + mdash) * (li64 - mdash)
    } else {
        (2 * li64) * (2 * li64 - 1)
    };
    (num.to_f64()
        .unwrap_or_else(|| panic!("Unable to convert `{num}` to `f64`."))
        / den
            .to_f64()
            .unwrap_or_else(|| panic!("Unable to convert `{den}` to `f64`.")))
    .sqrt()
}

/// Returns the coefficient $`v^l_{mm'}`$ as defined in Table 1 of Ivanic, J. & Ruedenberg, K.
/// Rotation Matrices for Real Spherical Harmonics. Direct Determination by Recursion.
/// *The Journal of Physical Chemistry* **100**, 9099–9100 (1996),
/// [DOI](https://doi.org/10.1021/jp953350u).
///
/// # Arguments
///
/// * `l` - The spherical harmonic order $`l`$.
/// * `m` - The index $`m`$ satisfying $`-l \le m \le l`$.
/// * `mdash` - The index $`m'`$ satisfying $`-l \le m' \le l`$.
///
/// # Returns
///
/// The value of $`v^l_{mm'}`$.
fn coeff_v(l: u32, m: i64, mdash: i64) -> f64 {
    let li64 = i64::from(l);
    assert!(
        m.abs() <= li64,
        "Index `m` = {} lies outside [-{}, {}].",
        m,
        l,
        l
    );
    assert!(
        mdash.abs() <= li64,
        "Index `mdash` = {} lies outside [-{}, {}].",
        mdash,
        l,
        l
    );

    let num = (1 + i64::from(kdelta(&m, &0))) * (li64 + m.abs() - 1) * (li64 + m.abs());
    let den = if mdash.abs() < li64 {
        (li64 + mdash) * (li64 - mdash)
    } else {
        (2 * li64) * (2 * li64 - 1)
    };
    0.5 * (num
        .to_f64()
        .unwrap_or_else(|| panic!("Unable to convert `{num}` to `f64`."))
        / den
            .to_f64()
            .unwrap_or_else(|| panic!("Unable to convert `{den}` to `f64`.")))
    .sqrt()
        * f64::from(1 - 2 * i16::from(kdelta(&m, &0)))
}

/// Returns the coefficient $`w^l_{mm'}`$ as defined in Table 1 of Ivanic, J. & Ruedenberg, K.
/// Rotation Matrices for Real Spherical Harmonics. Direct Determination by Recursion.
/// *The Journal of Physical Chemistry* **100**, 9099–9100 (1996),
/// [DOI](https://doi.org/10.1021/jp953350u).
///
/// # Arguments
///
/// * `l` - The spherical harmonic order $`l`$.
/// * `m` - The index $`m`$ satisfying $`-l \le m \le l`$.
/// * `mdash` - The index $`m'`$ satisfying $`-l \le m' \le l`$.
///
/// # Returns
///
/// The value of $`w^l_{mm'}`$.
fn coeff_w(l: u32, m: i64, mdash: i64) -> f64 {
    let li64 = i64::from(l);
    assert!(
        m.abs() <= li64,
        "Index `m` = {} lies outside [-{}, {}].",
        m,
        l,
        l
    );
    assert!(
        mdash.abs() <= li64,
        "Index `mdash` = {} lies outside [-{}, {}].",
        mdash,
        l,
        l
    );

    let num = (li64 - m.abs() - 1) * (li64 - m.abs());
    let den = if mdash.abs() < li64 {
        (li64 + mdash) * (li64 - mdash)
    } else {
        (2 * li64) * (2 * li64 - 1)
    };
    -0.5 * ((num
        .to_f64()
        .unwrap_or_else(|| panic!("Unable to convert `{num}` to `f64`.")))
        / (den
            .to_f64()
            .unwrap_or_else(|| panic!("Unable to convert `{den}` to `f64`."))))
    .sqrt()
        * f64::from(1 - kdelta(&m, &0))
}

/// Returns the representation matrix $`\mathbf{R}`$ for a rotation in the basis of the coordinate
/// *functions* $`(y, z, x)`$.
///
/// Let $`\hat{R}(\phi, \hat{\mathbf{n}})`$ be a rotation parametrised by the angle $`\phi`$ and
/// axis $`\hat{\mathbf{n}}`$. The corresponding representation matrix
/// $`\mathbf{R}(\phi, \hat{\mathbf{n}})`$ is defined as
///
/// ```math
/// \hat{R}(\phi, \hat{\mathbf{n}})\ (y, z, x)
/// = (y, z, x) \mathbf{R}(\phi, \hat{\mathbf{n}})
/// ```
///
/// See Section **2**-4 of Altmann, S. L. Rotations, Quaternions, and Double Groups. (Dover
/// Publications, Inc., 2005) for a detailed discussion on how $`(y, z, x)`$ should be considered
/// as coordinate *functions*.
///
/// # Arguments
///
/// * `angle` - The angle $`\phi`$ of the rotation in radians. A positive rotation is an
/// anticlockwise rotation when looking down `axis`.
/// * `axis` - A space-fixed vector defining the axis of rotation. The supplied vector will be
/// normalised.

/// # Returns
///
/// The representation matrix $`\mathbf{R}(\phi, \hat{\mathbf{n}})`$.
///
/// # Panics
///
/// Panics when a three-dimensional rotation matrix cannot be constructed for `angle` and `axis`.
#[must_use]
pub fn rmat(angle: f64, axis: Vector3<f64>) -> Array2<f64> {
    let normalised_axis = Unit::new_normalize(axis);
    let rot = Rotation3::from_axis_angle(&normalised_axis, angle);
    // nalgebra matrix iter is column-major.
    let rot_array = Array2::<f64>::from_shape_vec(
        (3, 3).f(),
        rot.into_inner().iter().copied().collect::<Vec<_>>(),
    )
    .unwrap_or_else(
        |_| panic!(
            "Unable to construct a three-dimensional rotation matrix for angle {angle} and axis {axis}."
        )
    );
    rot_array
        .select(Axis(0), &[1, 2, 0])
        .select(Axis(1), &[1, 2, 0])
}

/// Computes the representation matrix $`\mathbf{R}^l`$ for a transformation of interest in the
/// basis of real spherical harmonics $`Y_{lm}`$ ordered by increasing $`m`$, as defined in
/// Equation 5.8 and given in Equation 8.1 of Ivanic, J. & Ruedenberg, K. Rotation Matrices for
/// Real Spherical Harmonics. Direct Determination by Recursion. *The Journal of Physical
/// Chemistry* **100**, 9099–9100 (1996), [DOI](https://doi.org/10.1021/jp953350u).
///
/// # Arguments
///
/// * `l` - The spherical harmonic order $`l \ge 2`$.
/// * `rmat` - The representation matrix of the transformation of interest in the basis of coordinate
/// *functions* $`(y, z, x)`$, which are isosymmetric to the real spherical harmonics
/// $`(Y_{1, -1}, Y_{1, 0}, Y_{1, 1})`$.
/// * `rlm1` - The representation matrix of the transformation of interest in the basis of the real
/// spherical harmonics $`Y_{l-1, m}`$ ordered by increasing $`m`$.
///
/// # Returns
///
/// The required representation matrix $`\mathbf{R}^l`$.
///
/// # Panics
///
/// Panics when `l` is less than `2`.
#[must_use]
pub fn rlmat(l: u32, rmat: &Array2<f64>, rlm1: &Array2<f64>) -> Array2<f64> {
    assert!(l >= 2, "`l` must be at least 2.");
    let li64 = i64::from(l);
    let mut rl = Array2::<f64>::zeros((2 * l as usize + 1, 2 * l as usize + 1));
    for (mi, m) in (-li64..=li64).enumerate() {
        for (mdashi, mdash) in (-li64..=li64).enumerate() {
            let cu = coeff_u(l, m, mdash);
            let f_u = if approx::relative_ne!(cu.abs(), 0.0, epsilon = 1e-14, max_relative = 1e-14)
            {
                func_u(l, m, mdash, rmat, rlm1)
            } else {
                0.0
            };

            let cv = coeff_v(l, m, mdash);
            let f_v = if approx::relative_ne!(cv.abs(), 0.0, epsilon = 1e-14, max_relative = 1e-14)
            {
                func_v(l, m, mdash, rmat, rlm1)
            } else {
                0.0
            };

            let cw = coeff_w(l, m, mdash);
            let f_w = if approx::relative_ne!(cw.abs(), 0.0, epsilon = 1e-14, max_relative = 1e-14)
            {
                func_w(l, m, mdash, rmat, rlm1)
            } else {
                0.0
            };

            rl[(mi, mdashi)] = cu * f_u + cv * f_v + cw * f_w;
        }
    }
    rl
}
