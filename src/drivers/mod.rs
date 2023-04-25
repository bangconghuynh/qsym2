use std::error::Error;
use std::fmt;

use anyhow;

pub mod point_group_detection;

// =================
// Trait definitions
// =================

pub trait QSym2Driver {
    type Outcome;

    fn run(&mut self) -> Result<(), anyhow::Error>;

    fn result(&self) -> Result<&Self::Outcome, anyhow::Error>;
}

// ==================
// Struct definitions
// ==================

// ----------
// QSym2Error
// ----------

#[derive(Clone, Debug)]
pub struct QSym2Error<'a>
{
    source: Option<&'a dyn Error>,

    msg: Option<String>,
}

impl<'a> fmt::Display for QSym2Error<'a>
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if let Some(msg) = &self.msg {
            writeln!(f, "QSym2 has encountered an error with the following message:")?;
            writeln!(f, "{}", msg)?;
        } else {
            writeln!(f, "QSym2 has encountered an error.")?;
        }
        if let Some(source) = self.source {
            writeln!(f, "This error originates from another error:")?;
            writeln!(f, "{}", source)?;
        }
        write!(f, "Please report this error at https://gitlab.com/bangconghuynh/qsym2/-/issues/ for further support.")
    }
}

impl Error for QSym2Error<'static>
{
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        self.source
    }
}
