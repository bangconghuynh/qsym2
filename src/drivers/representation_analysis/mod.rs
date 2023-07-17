use std::fmt;

use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

use crate::basis::ao::BasisAngularOrder;
use crate::group::class::ClassPropertiesSummary;
use crate::io::format::{log_subtitle, qsym2_output, QSym2Output};

pub mod angular_function;
pub mod slater_determinant;

/// An enumerated type indicating the format of character table print-out.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub enum CharacterTableDisplay {
    /// Prints the character table symbolically showing explicitly the roots of unity.
    Symbolic,

    /// Prints the character table numerically where each character is a complex number.
    Numerical,
}

impl fmt::Display for CharacterTableDisplay {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            CharacterTableDisplay::Symbolic => write!(f, "Symbolic"),
            CharacterTableDisplay::Numerical => write!(f, "Numerical"),
        }
    }
}

/// An enumerated type indicating the type of magnetic symmetry to be used for representation
/// analysis.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[pyclass]
pub enum MagneticSymmetryAnalysisKind {
    /// Variant indicating that unitary representations should be used for magnetic symmetry
    /// analysis.
    Representation,

    /// Variant indicating that magnetic corepresentations should be used for magnetic symmetry
    /// analysis.
    Corepresentation,
}

impl fmt::Display for MagneticSymmetryAnalysisKind {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            MagneticSymmetryAnalysisKind::Representation => write!(f, "Unitary representations"),
            MagneticSymmetryAnalysisKind::Corepresentation => {
                write!(f, "Magnetic corepresentations")
            }
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
    qsym2_output!("");
    "The basis angular order information dictates how basis functions in each basis shell are transformed.\n\
    It is important to check that this is consistent with the basis set being used, otherwise incorrect\n\
    symmetry results will be obtained.".log_output_display();
    bao.log_output_display();
    qsym2_output!("");
}

/// Logs a conjugacy class transversal of a group nicely.
///
/// # Arguments
///
/// * `group` - A group for which a conjugacy class transversal should be logged nicely.
fn log_cc_transversal<G>(group: &G)
where
    G: ClassPropertiesSummary,
    G::GroupElement: fmt::Display,
{
    log_subtitle("Conjugacy class transversal");
    qsym2_output!("");
    group.class_transversal_to_string().log_output_display();
    qsym2_output!("");
}
