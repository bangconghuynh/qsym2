use derive_builder::Builder;
use nalgebra::{Point3, Vector3};

use crate::basis::ao::BasisShell;

// -------------------
// GaussianContraction
// -------------------

/// A structure to handle primitives in a Gaussian contraction.
#[derive(Clone, Builder, Debug)]
pub(crate) struct GaussianContraction<E, C> {
    /// Constituent primitives in the contraction. Each primitive has the form
    /// $`c\exp\left[-\alpha\lvert \mathbf{r} - \mathbf{R} \rvert^2\right]`$ is characterised by a
    /// tuple of its exponent $`\alpha`$ and coefficient $`c`$, respectively.
    pub(crate) primitives: Vec<(E, C)>,
}

// ---------------------
// BasisShellContraction
// ---------------------

/// A structure to handle all shell information for integrals.
#[derive(Clone, Builder, Debug)]
pub(crate) struct BasisShellContraction<E, C> {
    /// Basis function ordering information.
    pub(crate) basis_shell: BasisShell,

    /// The function starting index of this shell in the basis set.
    pub(crate) start_index: usize,

    /// The Gaussian primitives in the contraction of this shell.
    pub(crate) contraction: GaussianContraction<E, C>,

    /// The Cartesian origin $`\mathbf{R}`$ of this shell.
    pub(crate) cart_origin: Point3<f64>,

    /// The optional plane-wave $`\mathbf{k}`$ vector in the exponent
    /// $`\exp\left[i\mathbf{k}\cdot(\mathbf{r} - \mathbf{R})\right]`$ associated with this shell.
    /// If this is `None`, then this exponent is set to unity.
    pub(crate) k: Option<Vector3<f64>>,
}

impl<E, C> BasisShellContraction<E, C> {
    pub(crate) fn basis_shell(&self) -> &BasisShell {
        &self.basis_shell
    }

    pub(crate) fn k(&self) -> Option<&Vector3<f64>> {
        self.k.as_ref()
    }

    pub(crate) fn cart_origin(&self) -> &Point3<f64> {
        &self.cart_origin
    }
}
