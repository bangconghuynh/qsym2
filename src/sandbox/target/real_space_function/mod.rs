//! Real-space functions.

use std::fmt;

use derive_builder::Builder;
use itertools::Itertools;
use nalgebra::Point3;
use ndarray_linalg::types::Lapack;
use num_complex::ComplexFloat;

#[cfg(test)]
mod real_space_function_tests;

pub mod real_space_function_analysis;
mod real_space_function_transformation;

// ==================
// Struct definitions
// ==================

/// Structure to manage real-space functions.
#[derive(Builder, Clone)]
#[builder(build_fn(validate = "Self::validate"))]
pub struct RealSpaceFunction<T, F>
where
    T: ComplexFloat + Lapack,
    F: Fn(&Point3<f64>) -> T,
{
    /// The grid points $`\mathbf{r}_j`$ at which the real-space function is to be evaluated.
    grid_points: Vec<Point3<f64>>,

    /// The function $`\mathbb{R}^3 \to T`$.
    function: F,

    /// A boolean indicating if action of [`Self::function`] needs to be complex-conjugated.
    #[builder(default = "false")]
    complex_conjugated: bool,
}

impl<T, F> RealSpaceFunctionBuilder<T, F>
where
    T: ComplexFloat + Lapack,
    F: Fn(&Point3<f64>) -> T,
{
    fn validate(&self) -> Result<(), String> {
        let _ = self
            .grid_points
            .as_ref()
            .ok_or("No grid points found.".to_string())?;
        let _ = self
            .function
            .as_ref()
            .ok_or("No real-space function found.".to_string())?;
        Ok(())
    }
}

impl<T, F> RealSpaceFunction<T, F>
where
    T: ComplexFloat + Clone + Lapack,
    F: Clone + Fn(&Point3<f64>) -> T,
{
    /// Returns a builder to construct a new [`RealSpaceFunction`].
    pub fn builder() -> RealSpaceFunctionBuilder<T, F> {
        RealSpaceFunctionBuilder::default()
    }

    /// Returns a vector of shared references to the grid points at which the real-space function is
    /// evaluated.
    pub fn grid_points(&self) -> Vec<&Point3<f64>> {
        self.grid_points.iter().collect_vec()
    }

    /// Returns a shared reference to the function defining the [`RealSpaceFunction`].
    pub fn function(&self) -> &F {
        &self.function
    }
}

// =====================
// Trait implementations
// =====================

// -------
// Display
// -------
impl<T, F> fmt::Display for RealSpaceFunction<T, F>
where
    T: ComplexFloat + Lapack,
    F: Fn(&Point3<f64>) -> T,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "RealSpaceFunction{}[grid of {} {}]",
            if self.complex_conjugated { "*" } else { "" },
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
