use std::fmt;

use crate::aux::ao_basis::BasisAngularOrder;
use crate::aux::format::{log_subtitle, QSym2Output};
use crate::group::class::ClassPropertiesSummary;

pub mod slater_determinant;
pub mod angular_function;

/// An enumerated type indicating the format of character table print-out.
#[derive(Clone, Debug)]
pub enum CharacterTableDisplay {
    /// Prints the character table symbolically showing explicitly the roots of unity.
    Symbolic,

    /// Prints the character table numerically where each character is a complex number.
    Numerical
}

impl fmt::Display for CharacterTableDisplay {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            CharacterTableDisplay::Symbolic => write!(f, "Symbolic"),
            CharacterTableDisplay::Numerical => write!(f, "Numerical"),
        }
    }
}

/// Logs basis angular order information nicely.
///
/// # Arguments
///
/// * `bao` - The basis angular order information structure.
fn log_bao(bao: &BasisAngularOrder) {
    log_subtitle("Basis angular order");
    log::info!(target: "qsym2-output", "");
    log::info!(
        target: "qsym2-output",
        "The basis angular order information dictates how basis functions in each basis shell are transformed.\n\
        It is important to check that this is consistent with the basis set being used, otherwise incorrect\n\
        symmetry results will be obtained."
    );
    bao.log_output_display();
    log::info!(target: "qsym2-output", "");
}

/// Logs a conjugacy class transversal of a group nicely.
///
/// # Arguments
///
/// * `group` - A group for which a conjugacy class transversal should be logged nicely.
fn log_cc_transversal<G>(group: &G)
where
    G: ClassPropertiesSummary,
    G::GroupElement: fmt::Display
{
    log_subtitle("Conjugacy class transversal");
    log::info!(target: "qsym2-output", "");
    group.class_transversal_to_string().log_output_display();
    log::info!(target: "qsym2-output", "");

}
