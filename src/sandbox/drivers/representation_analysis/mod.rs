//! Sandbox drivers for symmetry analysis via representation and corepresentation theories.

use std::fmt;

#[cfg(feature = "python")]
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

use crate::basis::ao::BasisAngularOrder;
use crate::group::class::ClassPropertiesSummary;
use crate::io::format::{log_subtitle, qsym2_output, QSym2Output};

pub mod pes;
