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

    #[test]
    fn zero_gradient_row_has_zero_b_columns() {
        let x = build_design_matrix(&[[0.0, 0.0, 0.0]], 1000.0);
        for j in 0..6 {
            assert_eq!(x[(0, j)], 0.0);
        }
        assert_eq!(x[(0, 6)], 1.0);
    }
}
