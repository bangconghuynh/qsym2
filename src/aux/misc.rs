use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use itertools::{Itertools, MultiProduct};

pub trait HashableFloat {
    /// Returns a float rounded after being multiplied by a factor.
    ///
    /// Let $`x`$ be a float, $k$ a factor, and $[\cdot]$ denote the
    /// rounding-to-integer operation. This function yields $`[x \times k] / k`$.
    ///
    /// Arguments
    ///
    /// * threshold - The inverse $`k^{-1}`$ of the factor $k$ used in the
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
    /// * val - A floating point number.
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
/// * t - A struct of a hashable type.
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
    /// Rust implementation of Python's itertools.product() with repetition.
    ///
    /// From <https://stackoverflow.com/a/68231315>.
    ///
    /// # Arguments
    ///
    /// * repeat - Number of repetitions of the given iterator.
    fn product_repeat(self, repeat: usize) -> MultiProduct<Self> {
        std::iter::repeat(self)
            .take(repeat)
            .multi_cartesian_product()
    }
}

impl<T: Iterator + Clone> ProductRepeat for T where T::Item: Clone {}
