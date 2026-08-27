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
    if let Some(m) = mask
        && m.shape() != [nx, ny, nz]
    {
        return Err(DtiError::MaskShapeMismatch);
    }

    let design = build_design_matrix(gradients, bvalue);
    let mut field = TensorField::zeros((nx, ny, nz, TENSOR_COMPONENTS));
    let mut signal = vec![0.0f32; n_dirs];

    for x in 0..nx {
        for y in 0..ny {
            for z in 0..nz {
                if let Some(m) = mask
                    && m[[x, y, z]] <= 0.0
                {
                    continue;
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

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::SMatrix;
    use ndarray::{Array3, Array4};

    const EPSILON: f32 = 1e-4;

    #[inline]
    fn assert_near(actual: f32, expected: f32) {
        let abs_err = (actual - expected).abs();
        let max_val = actual.abs().max(expected.abs());
        let tol = EPSILON * max_val.max(1.0);
        assert!(
            abs_err <= tol,
            "expected {expected}, got {actual} (diff: {abs_err}, tol: {tol})"
        );
    }

    /// Helper to generate a standard 6-direction gradient set plus a b0 baseline.
    fn standard_acquisition_scheme() -> (Vec<[f32; 3]>, f32) {
        let bvalue = 1000.0;
        let gradients = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0 / 2.0_f32.sqrt(), 1.0 / 2.0_f32.sqrt(), 0.0],
            [1.0 / 2.0_f32.sqrt(), 0.0, 1.0 / 2.0_f32.sqrt()],
            [0.0, 1.0 / 2.0_f32.sqrt(), 1.0 / 2.0_f32.sqrt()],
        ];
        (gradients, bvalue)
    }

    /// Forward model: S_i = S0 * exp(-b * g^T D g)
    fn synthesize_signal(
        s0: f32,
        tensor: &SMatrix<f32, 3, 3>,
        gradients: &[[f32; 3]],
        bvalue: f32,
    ) -> Vec<f32> {
        gradients
            .iter()
            .map(|g| {
                let norm_sq = g[0] * g[0] + g[1] * g[1] + g[2] * g[2];
                if norm_sq < 1e-4 {
                    s0
                } else {
                    let g_vec = nalgebra::Vector3::new(g[0], g[1], g[2]);
                    let quad = g_vec.dot(&(tensor * g_vec));
                    s0 * (-bvalue * quad).exp()
                }
            })
            .collect()
    }

    // --- fit_voxel_tensor tests ---

    #[test]
    fn test_fit_voxel_tensor_recovers_known_isotropic_tensor() {
        let (gradients, bvalue) = standard_acquisition_scheme();
        let design = build_design_matrix(&gradients, bvalue);

        // Isotropic diffusion tensor with D = 0.001 mm^2/s
        let expected_d = 0.001_f32;
        let true_tensor = SMatrix::<f32, 3, 3>::new(
            expected_d, 0.0, 0.0, 0.0, expected_d, 0.0, 0.0, 0.0, expected_d,
        );

        let signal = synthesize_signal(1000.0, &true_tensor, &gradients, bvalue);
        let fitted = fit_voxel_tensor(&signal, &design).expect("Fit failed");

        assert_near(fitted[(0, 0)], expected_d);
        assert_near(fitted[(1, 1)], expected_d);
        assert_near(fitted[(2, 2)], expected_d);
        assert_near(fitted[(0, 1)], 0.0);
        assert_near(fitted[(0, 2)], 0.0);
        assert_near(fitted[(1, 2)], 0.0);
    }

    #[test]
    fn test_fit_voxel_tensor_recovers_anisotropic_off_axis_tensor() {
        let (gradients, bvalue) = standard_acquisition_scheme();
        let design = build_design_matrix(&gradients, bvalue);

        let true_tensor = SMatrix::<f32, 3, 3>::new(
            0.0015, 0.0003, 0.0001, 0.0003, 0.0008, 0.0002, 0.0001, 0.0002, 0.0005,
        );

        let signal = synthesize_signal(800.0, &true_tensor, &gradients, bvalue);
        let fitted = fit_voxel_tensor(&signal, &design).expect("Fit failed");

        for i in 0..3 {
            for j in 0..3 {
                assert_near(fitted[(i, j)], true_tensor[(i, j)]);
            }
        }
    }

    #[test]
    fn test_fit_voxel_tensor_non_positive_signal_returns_none() {
        let (gradients, bvalue) = standard_acquisition_scheme();
        let design = build_design_matrix(&gradients, bvalue);

        let mut signal = vec![1000.0; gradients.len()];

        // Zero signal check
        signal[2] = 0.0;
        assert!(fit_voxel_tensor(&signal, &design).is_none());

        // Negative signal check
        signal[2] = -10.0;
        assert!(fit_voxel_tensor(&signal, &design).is_none());
    }

    // --- eigen_decompose tests ---

    #[test]
    fn test_eigen_decompose_sorting_and_clamping() {
        // Diagonal matrix with out-of-order and negative entries
        let tensor =
            SMatrix::<f32, 3, 3>::new(0.0005, 0.0, 0.0, 0.0, -0.0002, 0.0, 0.0, 0.0, 0.0020);

        let decomp = eigen_decompose(&tensor);

        // Eigenvalues should be sorted descending and clamped to >= 0.0
        assert_near(decomp.eigenvalues[0], 0.0020);
        assert_near(decomp.eigenvalues[1], 0.0005);
        assert_near(decomp.eigenvalues[2], 0.0);
    }

    // --- fit_tensor_field & tensor_at tests ---

    #[test]
    fn test_fit_tensor_field_and_tensor_at_roundtrip() {
        let (gradients, bvalue) = standard_acquisition_scheme();
        let n_dirs = gradients.len();
        let (nx, ny, nz) = (2, 2, 2);

        let true_tensor =
            SMatrix::<f32, 3, 3>::new(0.0010, 0.0002, 0.0, 0.0002, 0.0010, 0.0, 0.0, 0.0, 0.0003);

        let mut dwi = Array4::<f32>::zeros((nx, ny, nz, n_dirs));
        let signal = synthesize_signal(1000.0, &true_tensor, &gradients, bvalue);

        for x in 0..nx {
            for y in 0..ny {
                for z in 0..nz {
                    for i in 0..n_dirs {
                        dwi[[x, y, z, i]] = signal[i];
                    }
                }
            }
        }

        let field = fit_tensor_field(&dwi, &gradients, bvalue, None).unwrap();
        let reconstructed = tensor_at(&field, 1, 1, 1);

        for i in 0..3 {
            for j in 0..3 {
                assert_near(reconstructed[(i, j)], true_tensor[(i, j)]);
            }
        }
    }

    #[test]
    fn test_fit_tensor_field_respects_mask() {
        let (gradients, bvalue) = standard_acquisition_scheme();
        let n_dirs = gradients.len();
        let (nx, ny, nz) = (2, 1, 1);

        let dwi = Array4::<f32>::from_elem((nx, ny, nz, n_dirs), 500.0);
        let mut mask = Array3::<f32>::zeros((nx, ny, nz));
        mask[[0, 0, 0]] = 1.0; // Voxel (0,0,0) included, (1,0,0) excluded

        let field = fit_tensor_field(&dwi, &gradients, bvalue, Some(&mask)).unwrap();

        // Included voxel should have non-zero components
        let tensor_in = tensor_at(&field, 0, 0, 0);
        assert!(tensor_in.norm() > 0.0);

        // Excluded voxel must stay all zeroes
        let tensor_out = tensor_at(&field, 1, 0, 0);
        assert_eq!(tensor_out.norm(), 0.0);
    }

    #[test]
    fn test_fit_tensor_field_error_mismatched_gradients() {
        let (gradients, bvalue) = standard_acquisition_scheme();
        let dwi = Array4::<f32>::zeros((2, 2, 2, 5)); // 5 directions in DWI vs 7 in gradients

        let res = fit_tensor_field(&dwi, &gradients, bvalue, None);
        assert!(matches!(
            res,
            Err(DtiError::GradientCountMismatch {
                expected: 5,
                found: 7
            })
        ));
    }

    #[test]
    fn test_fit_tensor_field_error_mismatched_mask() {
        let (gradients, bvalue) = standard_acquisition_scheme();
        let dwi = Array4::<f32>::zeros((2, 2, 2, gradients.len()));
        let bad_mask = Array3::<f32>::zeros((3, 2, 2)); // Shape mismatch on nx

        let res = fit_tensor_field(&dwi, &gradients, bvalue, Some(&bad_mask));
        assert!(matches!(res, Err(DtiError::MaskShapeMismatch)));
    }
}
