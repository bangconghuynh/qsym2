use std::fmt;

pub mod slater_determinant;

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
