use nalgebra::DMatrix;

/// Builds the DTI design matrix X (N x 7), one row per gradient direction:
/// [ -b*gx^2, -b*gy^2, -b*gz^2, -2b*gx*gy, -2b*gx*gz, -2b*gy*gz, 1 ]
///
/// This only depends on the acquisition scheme (gradients + b-value), never
/// on the image data, so it's built once per acquisition and reused across
/// every voxel — pulling it out also makes it independently testable against
/// a known gradient table.
pub fn build_design_matrix(gradients: &[[f32; 3]], bvalue: f32) -> DMatrix<f32> {
    let n_dirs = gradients.len();
    let mut x_mat = DMatrix::<f32>::zeros(n_dirs, 7);

    for (i, g) in gradients.iter().enumerate() {
        let norm_sq = g[0] * g[0] + g[1] * g[1] + g[2] * g[2];
        let b = if norm_sq > 1e-4 { bvalue } else { 0.0 };

        x_mat[(i, 0)] = -b * g[0] * g[0];
        x_mat[(i, 1)] = -b * g[1] * g[1];
        x_mat[(i, 2)] = -b * g[2] * g[2];
        x_mat[(i, 3)] = -2.0 * b * g[0] * g[1];
        x_mat[(i, 4)] = -2.0 * b * g[0] * g[2];
        x_mat[(i, 5)] = -2.0 * b * g[1] * g[2];
        x_mat[(i, 6)] = 1.0;
    }

    x_mat
}

#[cfg(test)]
mod tests {
    use super::*;

    #[inline]
    fn assert_near(actual: f32, expected: f32) {
        let abs_err = (actual - expected).abs();
        // Use relative tolerance scaled to magnitude, with an absolute floor for zero
        let max_val = actual.abs().max(expected.abs());
        let tol = 1e-4 * max_val.max(1.0);

        assert!(
            abs_err <= tol,
            "expected {expected}, got {actual} (diff: {abs_err}, tol: {tol})"
        );
    }

    #[test]
    fn test_dimensions_and_structure() {
        let gradients = vec![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let x = build_design_matrix(&gradients, 1000.0);

        assert_eq!(x.nrows(), 3);
        assert_eq!(x.ncols(), 7);
    }

    #[test]
    fn test_zero_gradient_row_has_zero_b_columns() {
        let x = build_design_matrix(&[[0.0, 0.0, 0.0]], 1000.0);
        for j in 0..6 {
            assert_eq!(x[(0, j)], 0.0);
        }
        assert_eq!(x[(0, 6)], 1.0);
    }

    #[test]
    fn test_sub_threshold_gradient_zeroed() {
        // Gradient norm squared < 1e-4 (e.g., |g| = 0.005, norm_sq = 0.00005)
        let x = build_design_matrix(&[[0.005, 0.005, 0.0]], 1000.0);
        for j in 0..6 {
            assert_eq!(x[(0, j)], 0.0);
        }
        assert_eq!(x[(0, 6)], 1.0);
    }

    #[test]
    fn test_single_axis_gradient() {
        let gradients = vec![[1.0, 0.0, 0.0]];
        let x = build_design_matrix(&gradients, 1000.0);

        // Expected row: [-1000, 0, 0, 0, 0, 0, 1]
        assert_near(x[(0, 0)], -1000.0);
        assert_near(x[(0, 1)], 0.0);
        assert_near(x[(0, 2)], 0.0);
        assert_near(x[(0, 3)], 0.0);
        assert_near(x[(0, 4)], 0.0);
        assert_near(x[(0, 5)], 0.0);
        assert_near(x[(0, 6)], 1.0);
    }

    #[test]
    fn test_off_axis_gradient_cross_terms() {
        // g = [1/sqrt(2), 1/sqrt(2), 0]
        let val = 1.0 / 2.0_f32.sqrt();
        let gradients = vec![[val, val, 0.0]];
        let bvalue = 1000.0;
        let x = build_design_matrix(&gradients, bvalue);

        // -b * gx^2 = -1000 * 0.5 = -500
        // -b * gy^2 = -1000 * 0.5 = -500
        // -2b * gx * gy = -2 * 1000 * 0.5 = -1000
        assert_near(x[(0, 0)], -500.0);
        assert_near(x[(0, 1)], -500.0);
        assert_near(x[(0, 2)], 0.0);
        assert_near(x[(0, 3)], -1000.0);
        assert_near(x[(0, 4)], 0.0);
        assert_near(x[(0, 5)], 0.0);
        assert_near(x[(0, 6)], 1.0);
    }

    #[test]
    fn test_empty_gradients() {
        let x = build_design_matrix(&[], 1000.0);
        assert_eq!(x.nrows(), 0);
        assert_eq!(x.ncols(), 7);
    }
}
