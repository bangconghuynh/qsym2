//! Angular momentum conversion and transformation.

use phf::phf_map;

pub mod sh_conversion;
pub mod sh_rotation_3d;
pub mod spinor_rotation_3d;

/// Alphabetical labels of angular momenta.
pub static ANGMOM_LABELS: [&str; 7] = ["S", "P", "D", "F", "G", "H", "I"];

/// Indices of alphabetical labels of angular momenta.
pub static ANGMOM_INDICES: phf::Map<&'static str, u32> = phf_map! {
    "S" => 0,
    "P" => 1,
    "D" => 2,
    "F" => 3,
    "G" => 4,
    "H" => 5,
    "I" => 6,
};
