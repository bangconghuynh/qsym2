use std::mem;
use num;
use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;


pub trait HashableFloat {
    /// Returns a float rounded after being multiplied by a factor.
    ///
    /// Let $x$ be a float, $k$ a factor, and $[\cdot]$ denote the
    /// rounding-to-integer operation. This function yields $[x \times k] / k$.
    ///
    /// Arguments
    ///
    /// * factor - The factor $k$ used in the rounding of the float.
    ///
    /// Returns
    ///
    /// The rounded float.
    fn round_factor(self, factor: Self) -> Self;

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
        (self * factor).round() / factor
    }

    fn integer_decode(self) -> (u64, i16, i8) {
        let bits: u64 = unsafe { mem::transmute(self) };
        let sign: i8 = if bits >> 63 == 0 { 1 } else { -1 };
        let mut exponent: i16 = ((bits >> 52) & 0x7ff) as i16;
        let mantissa = if exponent == 0 {
            (bits & 0xfffffffffffff) << 1
        } else {
            (bits & 0xfffffffffffff) | 0x10000000000000
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
