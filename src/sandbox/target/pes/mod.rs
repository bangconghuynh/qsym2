//! Potential energy surfaces.

use std::fmt;

use derive_builder::Builder;
use itertools::Itertools;
use log;
use nalgebra::Point3;
use ndarray_linalg::types::Lapack;
use num_complex::{Complex, ComplexFloat};
use num_traits::float::{Float, FloatConst};

#[cfg(test)]
mod pes_tests;

pub mod pes_analysis;
mod pes_transformation;

// ==================
// Struct definitions
// ==================

/// Structure to manage potential energy surfaces.
#[derive(Builder, Clone)]
#[builder(build_fn(validate = "Self::validate"))]
pub struct PES<T>
where
    T: ComplexFloat + Lapack,
{
    /// The grid points $`\mathbf{r}_j`$ at which the PES is to be evaluated.
    grid_points: Vec<Point3<f64>>,

    /// The function $`\mathbb{R}^3 \to T`$ defining the PES.
    function: fn(&Point3<f64>) -> T,

    /// A boolean indicating if action of [`Self::function`] needs to be complex-conjugated.
    #[builder(default = "false")]
    complex_conjugated: bool,
}

impl<T> PESBuilder<T>
where
    T: ComplexFloat + Lapack,
{
    fn validate(&self) -> Result<(), String> {
        let grid_points = self
            .grid_points
            .as_ref()
            .ok_or("No grid points found.".to_string())?;
        let function = self
            .function
            .as_ref()
            .ok_or("No PES function found.".to_string())?;
        Ok(())
    }
}

impl<T> PES<T>
where
    T: ComplexFloat + Clone + Lapack,
{
    /// Returns a builder to construct a new [`PES`].
    pub fn builder() -> PESBuilder<T> {
        PESBuilder::default()
    }

    /// Returns a vector of shared references to the grid points at which the PES is evaluated.
    pub fn grid_points(&self) -> Vec<&Point3<f64>> {
        self.grid_points.iter().collect_vec()
    }

    /// Returns a shared reference to the function defining the PES.
    pub fn function(&self) -> &fn(&Point3<f64>) -> T {
        &self.function
    }
}

// =====================
// Trait implementations
// =====================

// // ----
// // From
// // ----
// impl<T> From<PES<T>> for PES<Complex<T>>
// where
//     T: Float + FloatConst + Lapack,
//     Complex<T>: Lapack,
// {
//     fn from(value: PES<T>) -> Self {
//         PES::<Complex<T>>::builder()
//             .grid_points(value.grid_points.clone())
//             .function(|pt| Complex::from(value.function()(pt)))
//             .build()
//             .expect("Unable to complexify a `PES`.")
//     }
// }

// -------
// Display
// -------
impl<T> fmt::Display for PES<T>
where
    T: ComplexFloat + Lapack,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "PES[grid of {} {}]",
            self.grid_points.len(),
            if self.grid_points.len() == 1 {
                "point"
            } else {
                "points"
            }
        )?;
        Ok(())
    }
}
