//! Drivers for symmetry analysis via representation and corepresentation theories.

use std::fmt;

#[cfg(feature = "python")]
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

use crate::basis::ao::BasisAngularOrder;
use crate::group::class::ClassPropertiesSummary;
use crate::io::format::{QSym2Output, log_subtitle, qsym2_output};

pub mod angular_function;
pub mod density;
pub mod multideterminant;
pub mod slater_determinant;
pub mod vibrational_coordinate;

// ================
// Enum definitions
// ================

/// Enumerated type indicating the format of character table print-out.
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

/// Enumerated type indicating the type of magnetic symmetry to be used for representation
/// analysis.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[cfg_attr(feature = "python", pyclass(eq, eq_int))]
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

// =================
// Utility functions
// =================

/// Logs basis angular order information nicely.
///
/// # Arguments
///
/// * `bao` - The basis angular order information structure.
/// * `index` - The optional index for the explicit component to which the basis angular
///   information corresponds. If `None`, then it is assumed that the basis angular order information
///   is uniform across all explicit components.
pub(crate) fn log_bao(bao: &BasisAngularOrder, index: Option<usize>) {
    if let Some(i) = index {
        log_subtitle(&format!("Basis angular order (explicit component {i})"));
    } else {
        log_subtitle("Basis angular order (uniform across all explicit components)");
    }
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
pub(crate) fn log_cc_transversal<G>(group: &G)
where
    G: ClassPropertiesSummary,
    G::GroupElement: fmt::Display,
{
    log_subtitle("Conjugacy class transversal");
    qsym2_output!("");
    group.class_transversal_to_string().log_output_display();
    qsym2_output!("");
}

// =================
// Macro definitions
// =================

macro_rules! fn_construct_unitary_group {
    ( $(#[$meta:meta])* $vis:vis $func:ident ) => {
        $(#[$meta])*
        $vis fn $func(&self) -> Result<UnitaryRepresentedSymmetryGroup, anyhow::Error> {
            let params = self.parameters;
            let sym = match params.use_magnetic_group {
                Some(MagneticSymmetryAnalysisKind::Representation) => self.symmetry_group
                    .magnetic_symmetry
                    .as_ref()
                    .ok_or_else(|| {
                        format_err!(
                            "Magnetic symmetry requested for analysis, but no magnetic symmetry found."
                        )
                    })?,
                Some(MagneticSymmetryAnalysisKind::Corepresentation) => bail!("Magnetic corepresentations requested, but unitary-represented group is being constructed."),
                None => &self.symmetry_group.unitary_symmetry
            };
            let group = if params.use_double_group {
                UnitaryRepresentedGroup::from_molecular_symmetry(sym, params.infinite_order_to_finite)?
                    .to_double_group()?
            } else {
                UnitaryRepresentedGroup::from_molecular_symmetry(sym, params.infinite_order_to_finite)?
            };

            qsym2_output!(
                "Unitary-represented group for representation analysis: {}",
                group.name()
            );
            qsym2_output!("");
            if let Some(chartab_display) = params.write_character_table.as_ref() {
                log_subtitle("Character table of irreducible representations");
                qsym2_output!("");
                match chartab_display {
                    CharacterTableDisplay::Symbolic => {
                        group.character_table().log_output_debug();
                        "Any `En` in a character value denotes the first primitive n-th root of unity:\n  \
                        En = exp(2πi/n)".log_output_display();
                    }
                    CharacterTableDisplay::Numerical => group.character_table().log_output_display(),
                }
                qsym2_output!("");
                qsym2_output!("The symbol `◈` indicates the principal class of the group.");
                qsym2_output!("");
                "Note 1: `FS` contains the classification of the irreps using the Frobenius--Schur indicator:\n  \
                `r` = real: the irrep and its complex-conjugate partner are real and identical,\n  \
                `c` = complex: the irrep and its complex-conjugate partner are complex and inequivalent,\n  \
                `q` = quaternion: the irrep and its complex-conjugate partner are complex and equivalent.\n\n\
                Note 2: The conjugacy classes are sorted according to the following order:\n  \
                E -> C_n (n descending) -> C2 -> i -> S_n (n decending) -> σ\n  \
                Within each order and power, elements with axes close to Cartesian axes are put first.\n  \
                Within each equi-inclination from Cartesian axes, z-inclined axes are put first, then y, then x.\n\n\
                Note 3: The Mulliken labels generated for the irreps in the table above are internally consistent.\n  \
                However, certain labels might differ from those tabulated elsewhere using other conventions.\n  \
                If need be, please check with other literature to ensure external consistency.".log_output_display();
                qsym2_output!("");
            }
            Ok(group)
        }
    }
}

macro_rules! fn_construct_magnetic_group {
    ( $(#[$meta:meta])* $vis:vis $func:ident ) => {
        $(#[$meta])*
        $vis fn $func(&self) -> Result<MagneticRepresentedSymmetryGroup, anyhow::Error> {
            let params = self.parameters;
            let sym = match params.use_magnetic_group {
                Some(MagneticSymmetryAnalysisKind::Corepresentation) => self.symmetry_group
                    .magnetic_symmetry
                    .as_ref()
                    .ok_or_else(|| {
                        format_err!(
                            "Magnetic symmetry requested for analysis, but no magnetic symmetry found."
                        )
                    })?,
                Some(MagneticSymmetryAnalysisKind::Representation) => bail!("Unitary representations requested, but magnetic-represented group is being constructed."),
                None => &self.symmetry_group.unitary_symmetry
            };
            let group = if params.use_double_group {
                MagneticRepresentedGroup::from_molecular_symmetry(sym, params.infinite_order_to_finite)?
                    .to_double_group()?
            } else {
                MagneticRepresentedGroup::from_molecular_symmetry(sym, params.infinite_order_to_finite)?
            };

            qsym2_output!(
                "Magnetic-represented group for corepresentation analysis: {}",
                group.name()
            );
            qsym2_output!("");

            if let Some(chartab_display) = params.write_character_table.as_ref() {
                log_subtitle("Character table of irreducible corepresentations");
                qsym2_output!("");
                match chartab_display {
                    CharacterTableDisplay::Symbolic => {
                        group.character_table().log_output_debug();
                        "Any `En` in a character value denotes the first primitive n-th root of unity:\n  \
                        En = exp(2πi/n)".log_output_display();
                    }
                    CharacterTableDisplay::Numerical => group.character_table().log_output_display(),
                }
                qsym2_output!("");
                qsym2_output!("The symbol `◈` indicates the principal class of the group.");
                qsym2_output!("");
                "Note 1: The ircorep notation `D[Δ]` means that this ircorep is induced by the representation Δ\n  \
                of the unitary halving subgroup. The exact nature of Δ determines the kind of D[Δ].\n\n\
                Note 2: `IN` shows the intertwining numbers of the ircoreps which classify them into three kinds:\n  \
                `1` = 1st kind: the ircorep is induced by a single irrep of the unitary halving subgroup once,\n  \
                `4` = 2nd kind: the ircorep is induced by a single irrep of the unitary halving subgroup twice,\n  \
                `2` = 3rd kind: the ircorep is induced by an irrep of the unitary halving subgroup and its Wigner conjugate.\n\n\
                Note 3: Only unitary-represented elements are shown in the character table, as characters of\n  \
                antiunitary-represented elements are not invariant under a change of basis.\n\n\
                Refs:\n  \
                Newmarch, J. D. & Golding, R. M. J. Math. Phys. 23, 695–704 (1982)\n  \
                Bradley, C. J. & Davies, B. L. Rev. Mod. Phys. 40, 359–379 (1968)\n  \
                Newmarch, J. D. J. Math. Phys. 24, 742–756 (1983)".log_output_display();
                qsym2_output!("");
            }

            Ok(group)
        }
    }
}

pub(crate) use fn_construct_magnetic_group;
pub(crate) use fn_construct_unitary_group;
