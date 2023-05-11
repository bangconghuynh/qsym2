use std::fmt;

pub mod slater_determinant;

#[derive(Clone, Debug)]
pub enum CharacterTableDisplay {
    Symbolic,
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
