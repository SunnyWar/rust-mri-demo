use nalgebra::{DMatrix, DVector, SMatrix};
use ndarray::{Array3, Array4};

use super::design::build_design_matrix;
use super::types::{DtiError, TensorEigenDecomp};

/// The 6 unique components of a symmetric 3x3 diffusion tensor, in the order
/// they're stored in a `TensorField`: [Dxx, Dyy, Dzz, Dxy, Dxz, Dyz].
pub const TENSOR_COMPONENTS: usize = 6;

/// Per-voxel diffusion tensors for a whole volume, shape (nx, ny, nz, 6).
/// Voxels that were skipped (masked out, or non-positive signal) are left as
/// all-zero components. This is the expensive product of DTI fitting — the
/// WLLS solve is the costly step, not any individual scalar metric — so it's
/// kept around and reused rather than recomputed per metric.
pub type TensorField = Array4<f32>;

/// Fits a 3x3 diffusion tensor at a single voxel using Weighted Linear Least
/// Squares (WLLS), with W = diag(S^2) to correct for log-transformation
/// noise heteroscedasticity. Returns `None` if any signal is non-positive
/// (can't take its log) or the weighted normal equations are singular.
pub fn fit_voxel_tensor(signal: &[f32], design: &DMatrix<f32>) -> Option<SMatrix<f32, 3, 3>> {
    let n_dirs = signal.len();
    let mut y_vec = DVector::<f32>::zeros(n_dirs);
    let mut weights = DVector::<f32>::zeros(n_dirs);

    for (i, &s) in signal.iter().enumerate() {
        if s <= 0.0 {
            return None;
        }
        y_vec[i] = s.ln();
        weights[i] = s * s; // W_ii = S_i^2
    }

    // Weighted system: (X^T W X) beta = X^T W y
    let mut wx = design.clone();
    for i in 0..n_dirs {
        let w = weights[i];
        for j in 0..7 {
            wx[(i, j)] *= w;
        }
    }

    let xt_w_x = design.transpose() * &wx;
    let xt_w_y = wx.transpose() * &y_vec;

    let beta = xt_w_x.lu().solve(&xt_w_y)?;

    let (dxx, dyy, dzz) = (beta[0], beta[1], beta[2]);
    let (dxy, dxz, dyz) = (beta[3], beta[4], beta[5]);
    Some(SMatrix::<f32, 3, 3>::new(
        dxx, dxy, dxz, dxy, dyy, dyz, dxz, dyz, dzz,
    ))
}

/// Eigendecomposes a fitted tensor into the sorted, non-negative eigenvalues
/// every scalar DTI metric is built from.
pub fn eigen_decompose(tensor: &SMatrix<f32, 3, 3>) -> TensorEigenDecomp {
    let eigen = tensor.symmetric_eigen();
    let mut eigenvalues = [
        eigen.eigenvalues[0].max(0.0),
        eigen.eigenvalues[1].max(0.0),
        eigen.eigenvalues[2].max(0.0),
    ];
    eigenvalues.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    TensorEigenDecomp { eigenvalues }
}

/// Fits a diffusion tensor at every (masked-in) voxel in the volume.
pub fn fit_tensor_field(
    dwi: &Array4<f32>,
    gradients: &[[f32; 3]],
    bvalue: f32,
    mask: Option<&Array3<f32>>,
) -> Result<TensorField, DtiError> {
    let shape = dwi.shape();
    let (nx, ny, nz, n_dirs) = (shape[0], shape[1], shape[2], shape[3]);

    if gradients.len() != n_dirs {
        return Err(DtiError::GradientCountMismatch {
            expected: n_dirs,
            found: gradients.len(),
        });
    }
    if let Some(m) = mask {
        if m.shape() != [nx, ny, nz] {
            return Err(DtiError::MaskShapeMismatch);
        }
    }

    let design = build_design_matrix(gradients, bvalue);
    let mut field = TensorField::zeros((nx, ny, nz, TENSOR_COMPONENTS));
    let mut signal = vec![0.0f32; n_dirs];

    for x in 0..nx {
        for y in 0..ny {
            for z in 0..nz {
                if let Some(m) = mask {
                    if m[[x, y, z]] <= 0.0 {
                        continue;
                    }
                }

                for i in 0..n_dirs {
                    signal[i] = dwi[[x, y, z, i]];
                }

                if let Some(tensor) = fit_voxel_tensor(&signal, &design) {
                    field[[x, y, z, 0]] = tensor[(0, 0)]; // Dxx
                    field[[x, y, z, 1]] = tensor[(1, 1)]; // Dyy
                    field[[x, y, z, 2]] = tensor[(2, 2)]; // Dzz
                    field[[x, y, z, 3]] = tensor[(0, 1)]; // Dxy
                    field[[x, y, z, 4]] = tensor[(0, 2)]; // Dxz
                    field[[x, y, z, 5]] = tensor[(1, 2)]; // Dyz
                }
            }
        }
    }

    Ok(field)
}

/// Reconstructs the symmetric tensor matrix stored at a voxel in a
/// `TensorField`.
pub fn tensor_at(field: &TensorField, x: usize, y: usize, z: usize) -> SMatrix<f32, 3, 3> {
    let (dxx, dyy, dzz) = (
        field[[x, y, z, 0]],
        field[[x, y, z, 1]],
        field[[x, y, z, 2]],
    );
    let (dxy, dxz, dyz) = (
        field[[x, y, z, 3]],
        field[[x, y, z, 4]],
        field[[x, y, z, 5]],
    );
    SMatrix::<f32, 3, 3>::new(dxx, dxy, dxz, dxy, dyy, dyz, dxz, dyz, dzz)
}
