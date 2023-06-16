use std::collections::hash_map::DefaultHasher;
use std::error::Error;
use std::fmt;
use std::hash::{Hash, Hasher};

use itertools::{Itertools, MultiProduct};
use log;
use ndarray::{stack, Array1, Array2, Axis};
use num_complex::ComplexFloat;

/// A trait to enable floating point numbers to be hashed.
pub trait HashableFloat {
    /// Returns a float rounded after being multiplied by a factor.
    ///
    /// Let $`x`$ be a float, $`k`$ a factor, and $`[\cdot]`$ denote the
    /// rounding-to-integer operation. This function yields $`[x \times k] / k`$.
    ///
    /// Arguments
    ///
    /// * `threshold` - The inverse $`k^{-1}`$ of the factor $`k`$ used in the
    /// rounding of the float.
    ///
    /// Returns
    ///
    /// The rounded float.
    #[must_use]
    fn round_factor(self, threshold: Self) -> Self;

    /// Returns the mantissa-exponent-sign triplet for a float.
    ///
    /// Reference: <https://stackoverflow.com/questions/39638363/how-can-i-use-a-hashmap-with-f64-as-key-in-rust>
    ///
    /// # Arguments
    ///
    /// * `val` - A floating point number.
    ///
    /// # Returns
    ///
    /// The corresponding mantissa-exponent-sign triplet.
    fn integer_decode(self) -> (u64, i16, i8);
}

impl HashableFloat for f64 {
    fn round_factor(self, factor: f64) -> Self {
        (self / factor).round() * factor + 0.0
    }

    fn integer_decode(self) -> (u64, i16, i8) {
        let bits: u64 = self.to_bits();
        let sign: i8 = if bits >> 63 == 0 { 1 } else { -1 };
        let mut exponent: i16 = ((bits >> 52) & 0x7ff) as i16;
        let mantissa = if exponent == 0 {
            (bits & 0x000f_ffff_ffff_ffff) << 1
        } else {
            (bits & 0x000f_ffff_ffff_ffff) | 0x0010_0000_0000_0000
        };

        exponent -= 1023 + 52;
        (mantissa, exponent, sign)
    }
}

/// Returns the hash value of a hashable struct.
///
/// Arguments
///
/// * `t` - A struct of a hashable type.
///
/// Returns
///
/// The hash value.
pub fn calculate_hash<T: Hash>(t: &T) -> u64 {
    let mut s = DefaultHasher::new();
    t.hash(&mut s);
    s.finish()
}

pub trait ProductRepeat: Iterator + Clone
where
    Self::Item: Clone,
{
    /// Rust implementation of Python's `itertools.product()` with repetition.
    ///
    /// From <https://stackoverflow.com/a/68231315>.
    ///
    /// # Arguments
    ///
    /// * `repeat` - Number of repetitions of the given iterator.
    ///
    /// # Returns
    ///
    /// A [`MultiProduct`] iterator.
    fn product_repeat(self, repeat: usize) -> MultiProduct<Self> {
        std::iter::repeat(self)
            .take(repeat)
            .multi_cartesian_product()
    }
}

impl<T: Iterator + Clone> ProductRepeat for T where T::Item: Clone {}

// =============
// Gram--Schmidt
// =============

#[derive(Debug, Clone)]
pub struct GramSchmidtError<'a, T> {
    pub mat: Option<&'a Array2<T>>,
    pub vecs: Option<&'a [Array1<T>]>,
}

impl<'a, T: fmt::Display + fmt::Debug> fmt::Display for GramSchmidtError<'a, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "Unable to perform Gram--Schmidt orthogonalisation on:",)?;
        if let Some(mat) = self.mat {
            writeln!(f, "{mat}")?;
        } else if let Some(vecs) = self.vecs {
            for vec in vecs {
                writeln!(f, "{vec}")?;
            }
        } else {
            writeln!(f, "Unspecified basis vectors for Gram--Schmidt.")?;
        }
        Ok(())
    }
}

impl<'a, T: fmt::Display + fmt::Debug> Error for GramSchmidtError<'a, T> {}

/// Performs modified Gram--Schmidt orthonormalisation on a set of column vectors in a matrix with
/// respect to the complex-symmetric or Hermitian dot product.
///
/// # Arguments
///
/// * `vmat` - Matrix containing column vectors forming a basis for a subspace.
/// * `complex_symmetric` - A boolean indicating if the vector dot product is complex-symmetric. If
/// `false`, the conventional Hermitian dot product is used.
/// * `thresh` - A threshold for determining self-orthogonal vectors.
///
/// # Returns
///
/// The orthonormal vectors forming a basis for the same subspace collected as column vectors in a
/// matrix.
///
/// # Errors
///
/// Errors when the orthonormalisation procedure fails, which occurs when there is linear dependency
/// between the basis vectors and/or when self-orthogonal vectors are encountered.
pub fn complex_modified_gram_schmidt<T>(
    vmat: &Array2<T>,
    complex_symmetric: bool,
    thresh: T::Real,
) -> Result<Array2<T>, GramSchmidtError<T>>
where
    T: ComplexFloat + fmt::Display + 'static,
{
    let mut us: Vec<Array1<T>> = Vec::with_capacity(vmat.shape()[1]);
    let mut us_sq_norm: Vec<T> = Vec::with_capacity(vmat.shape()[1]);
    for (i, vi) in vmat.columns().into_iter().enumerate() {
        // u[i] now initialised with v[i]
        us.push(vi.to_owned());

        // Project ui onto all uj (0 <= j < i)
        // This is the 'modified' part of Gram--Schmidt. We project the current (and being updated)
        // ui onto uj, rather than projecting vi onto uj. This enhances numerical stability.
        for j in 0..i {
            let p_uj_ui = if complex_symmetric {
                us[j].t().dot(&us[i]) / us_sq_norm[j]
            } else {
                us[j].t().map(|x| x.conj()).dot(&us[i]) / us_sq_norm[j]
            };
            us[i] = &us[i] - us[j].map(|&x| x * p_uj_ui);
        }

        // Evaluate the squared norm of ui which will no longer be changed after this iteration.
        // us_sq_norm[i] now available.
        let us_sq_norm_i = if complex_symmetric {
            us[i].t().dot(&us[i])
        } else {
            us[i].t().map(|x| x.conj()).dot(&us[i])
        };
        if us_sq_norm_i.abs() < thresh {
            log::error!("A zero-norm vector found: {}", us[i]);
            return Err(GramSchmidtError {
                mat: Some(vmat),
                vecs: None,
            });
        }
        us_sq_norm.push(us_sq_norm_i);
    }

    // Normalise ui
    for i in 0..us.len() {
        us[i].mapv_inplace(|x| x / us_sq_norm[i].sqrt());
    }

    let ortho_check = us.iter().enumerate().all(|(i, ui)| {
        us.iter().enumerate().all(|(j, uj)| {
            let ov_ij = if complex_symmetric {
                ui.dot(uj)
            } else {
                ui.map(|x| x.conj()).dot(uj)
            };
            i == j || ov_ij.abs() < thresh
        })
    });

    if ortho_check {
        let umat = stack(Axis(1), &us.iter().map(|u| u.view()).collect_vec())
            .expect("Unable to concatenate the orthogonal vectors into a matrix.");
        Ok(umat)
    } else {
        log::error!("Post-Gram--Schmidt orthogonality check failed.");
        Err(GramSchmidtError {
            mat: Some(vmat),
            vecs: None,
        })
    }
}
