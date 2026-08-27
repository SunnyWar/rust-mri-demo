/// Eigenvalues of a fitted diffusion tensor, clipped to be non-negative and
/// sorted descending (l1 >= l2 >= l3). This is the common input every scalar
/// DTI metric (FA, MD, RD, AD, ...) is computed from, so new metrics only
/// need this, not the raw tensor or the fitting machinery.
#[derive(Debug, Clone, Copy)]
pub struct TensorEigenDecomp {
    pub eigenvalues: [f32; 3],
}

use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Error)]
pub enum DtiError {
    #[error("expected {expected} gradient directions (one per DWI volume), found {found}")]
    GradientCountMismatch { expected: usize, found: usize },

    #[error("mask spatial shape does not match DWI volume shape")]
    MaskShapeMismatch,
}
