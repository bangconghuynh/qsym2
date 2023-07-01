use serde::{Serialize, Deserialize};

use crate::interfaces::input::ao_basis::InputBasisAngularOrder;

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Input target: Slater determinant; source: custom
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// A serialisable/deserialisable structure containing control parameters for acquiring Slater
/// determinant(s) from a custom specification.
#[derive(Clone, Serialize, Deserialize, Default)]
pub struct CustomSlaterDeterminantSource {
    /// A specification for the basis angular order information.
    pub bao: InputBasisAngularOrder,
}
