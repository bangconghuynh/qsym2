//! Potential energy surfaces.

use std::fmt;
use std::iter::Sum;

use derive_builder::Builder;
use itertools::Itertools;
use log;
use nalgebra::Point3;
use ndarray::Array2;
use ndarray_linalg::types::Lapack;
use num_complex::{Complex, ComplexFloat};
use num_traits::float::{Float, FloatConst};

use crate::group::GroupProperties;

#[cfg(test)]
mod pes_tests;

pub mod pes_analysis;
mod pes_transformation;

// ==================
// Struct definitions
// ==================

/// Structure to manage potential energy surfaces evaluated at symmetry-equivalent points.
#[derive(Builder, Clone)]
#[builder(build_fn(validate = "Self::validate"))]
pub struct PES<'a, G, T>
where
    T: ComplexFloat + Lapack,
    G: GroupProperties + Clone,
{
    /// The group governing the symmetry-equivalent points at which this PES has been evaluated.
    group: &'a G,

    /// The symmetry-unique grid points $`\mathbf{r}_j`$ at which the PES is evaluated.
    grid_points: Vec<Point3<f64>>,

    /// The values of this PES evaluated at symmetry-equivalent points. The elements `$V_{ij}$` is
    /// given by $`V(\hat{g}_i \mathbf{r}_j)`$, where $`\mathbf{r}_j`$ is a symmetry-unique grid
    /// point at which the PES is evaluated.
    values: Array2<T>,
}

impl<'a, G, T> PESBuilder<'a, G, T>
where
    T: ComplexFloat + Lapack,
    G: GroupProperties + Clone,
{
    fn validate(&self) -> Result<(), String> {
        let group = self.group.ok_or("No symmetry group found.".to_string())?;
        let grid_points = self
            .grid_points
            .as_ref()
            .ok_or("No symmetry-unique grid points found.".to_string())?;
        let values = self.values.as_ref().ok_or("No PES found.".to_string())?;

        let pes_shape = group.order() == values.shape()[0];
        if !pes_shape {
            log::error!(
                "The number of rows in the PES value matrix ({:?}) does not match the group order ({}).",
                values.shape()[0],
                group.order()
            );
        }

        let grid_count = grid_points.len() == values.shape()[1];
        if !grid_count {
            log::error!(
                "The number of columns in the PES value matrix ({:?}) does not match the number of grid points ({}).",
                values.shape()[1],
                grid_points.len()
            );
        }

        if pes_shape && grid_count {
            Ok(())
        } else {
            Err("PES validation failed.".to_string())
        }
    }
}

impl<'a, G, T> PES<'a, G, T>
where
    T: ComplexFloat + Clone + Lapack,
    G: GroupProperties + Clone,
{
    /// Returns a builder to construct a new [`PES`].
    pub fn builder() -> PESBuilder<'a, G, T> {
        PESBuilder::default()
    }

    /// Returns a vector of shared references to the symmetry-unique grid points at which the PES
    /// is evaluated.
    pub fn grid_points(&self) -> Vec<&Point3<f64>> {
        self.grid_points.iter().collect_vec()
    }

    /// Returns a shared reference to the value matrix of the PES.
    pub fn values(&self) -> &Array2<T> {
        &self.values
    }

    /// Returns a shared reference to the group governing the symmetry-equivalent points at which
    /// the PES is evaluated.
    pub fn group(&self) -> &G {
        self.group
    }
}

// =====================
// Trait implementations
// =====================

// ----
// From
// ----
impl<'a, G, T> From<PES<'a, G, T>> for PES<'a, G, Complex<T>>
where
    T: Float + FloatConst + Lapack,
    Complex<T>: Lapack,
    G: GroupProperties + Clone,
{
    fn from(value: PES<'a, G, T>) -> Self {
        PES::<'a, G, Complex<T>>::builder()
            .grid_points(value.grid_points.clone())
            .values(value.values.map(Complex::from))
            .group(value.group)
            .build()
            .expect("Unable to complexify a `PES`.")
    }
}

// -------
// Display
// -------
impl<'a, G, T> fmt::Display for PES<'a, G, T>
where
    T: fmt::Display + ComplexFloat + Lapack,
    <T as ComplexFloat>::Real: Sum + From<u16> + fmt::Display,
    G: GroupProperties + Clone,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "PES[value matrix of dimensions {}]",
            self.values
                .shape()
                .iter()
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join("×")
        )?;
        Ok(())
    }
}
