/// Eigenvalues of a fitted diffusion tensor, clipped to be non-negative and
/// sorted descending (l1 >= l2 >= l3). This is the common input every scalar
/// DTI metric (FA, MD, RD, AD, ...) is computed from, so new metrics only
/// need this, not the raw tensor or the fitting machinery.
#[derive(Debug, Clone, Copy)]
pub struct TensorEigenDecomp {
    pub eigenvalues: [f32; 3],
}

#[derive(Debug, Clone, PartialEq)]
pub enum DtiError {
    /// Number of gradient directions didn't match the number of DWI volumes.
    GradientCountMismatch { expected: usize, found: usize },
    /// Mask shape didn't match the spatial dimensions of the DWI volume.
    MaskShapeMismatch,
}

impl std::fmt::Display for DtiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DtiError::GradientCountMismatch { expected, found } => write!(
                f,
                "expected {expected} gradient directions (one per DWI volume), found {found}"
            ),
            DtiError::MaskShapeMismatch => {
                write!(f, "mask spatial shape does not match DWI volume shape")
            }
        }
    }
}

impl std::error::Error for DtiError {}
