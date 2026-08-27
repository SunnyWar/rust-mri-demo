use ndarray::{Array3, Axis};

/// Convolves a single axis in-place using a 1D Gaussian kernel.
fn convolve_axis(vol: &mut Array3<f32>, kernel: &[f32], axis: Axis) {
    let mut temp = vol.clone();
    let radius = (kernel.len() / 2) as isize;

    // Iterate over 1D lanes along the target axis
    for (src_lane, mut dst_lane) in vol.lanes(axis).into_iter().zip(temp.lanes_mut(axis)) {
        let len = src_lane.len() as isize;

        for x in 0..len {
            let mut sum = 0.0;
            let mut weight = 0.0;

            for (i, &w) in kernel.iter().enumerate() {
                let ix = x + i as isize - radius;
                if ix >= 0 && ix < len {
                    sum += src_lane[ix as usize] * w;
                    weight += w;
                }
            }

            dst_lane[x as usize] = if weight > 0.0 { sum / weight } else { 0.0 };
        }
    }

    *vol = temp;
}

/// Applies 3D separable Gaussian convolution accounting for physical voxel
/// dimensions (zooms: dx, dy, dz in mm) extracted from NIfTI affine metadata.
pub fn gaussian_smooth_3d_anisotropic(vol: &mut Array3<f32>, sigma_mm: f32, zooms: [f32; 3]) {
    let sigmas_vox = [
        sigma_mm / zooms[0],
        sigma_mm / zooms[1],
        sigma_mm / zooms[2],
    ];

    let axes = [Axis(0), Axis(1), Axis(2)];

    for (&sigma, axis) in sigmas_vox.iter().zip(axes) {
        if sigma > 0.0 {
            let kernel = build_1d_gaussian_kernel(sigma);
            convolve_axis(vol, &kernel, axis);
        }
    }
}

/// Builds a normalized 1D Gaussian kernel truncated at `truncation * sigma` (default 4.0).
pub fn build_1d_gaussian_kernel(sigma_vox: f32) -> Vec<f32> {
    if sigma_vox <= 0.0 {
        return vec![1.0];
    }

    // Truncate kernel at 4.0 standard deviations
    let radius = (4.0 * sigma_vox).ceil() as usize;
    let size = 2 * radius + 1;
    let mut kernel = Vec::with_capacity(size);

    let two_sigma_sq = 2.0 * sigma_vox * sigma_vox;
    let norm_factor = 1.0 / ((2.0 * std::f32::consts::PI).sqrt() * sigma_vox);

    let mut sum = 0.0;
    for i in 0..size {
        let x = i as f32 - radius as f32;
        let val = norm_factor * (-x * x / two_sigma_sq).exp();
        kernel.push(val);
        sum += val;
    }

    // Normalize so kernel sums to 1.0 (preserves signal energy)
    if sum > 0.0 {
        for val in kernel.iter_mut() {
            *val /= sum;
        }
    }

    kernel
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

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

    // --- build_1d_gaussian_kernel tests ---

    #[test]
    fn test_build_1d_gaussian_kernel_properties() {
        let sigma = 1.5;
        let kernel = build_1d_gaussian_kernel(sigma);

        // Kernel size must be odd: 2 * ceil(4 * 1.5) + 1 = 2 * 6 + 1 = 13
        assert_eq!(kernel.len() % 2, 1);
        assert_eq!(kernel.len(), 13);

        // Sum must be normalized to 1.0
        let sum: f32 = kernel.iter().sum();
        assert_near(sum, 1.0);

        // Kernel must be symmetric around center
        let radius = kernel.len() / 2;
        for i in 1..=radius {
            assert_near(kernel[radius - i], kernel[radius + i]);
        }

        // Peak must be strictly at the center
        for i in 0..radius {
            assert!(kernel[i] <= kernel[i + 1]);
        }
    }

    #[test]
    fn test_build_1d_gaussian_kernel_zero_or_negative_sigma() {
        assert_eq!(build_1d_gaussian_kernel(0.0), vec![1.0]);
        assert_eq!(build_1d_gaussian_kernel(-1.0), vec![1.0]);
    }

    // --- convolve_axis tests ---

    #[test]
    fn test_convolve_axis_preserves_constant_volume() {
        let mut vol = Array3::from_elem((5, 5, 5), 42.0);
        let kernel = build_1d_gaussian_kernel(1.0);

        // Normalized re-weighting on boundaries must preserve flat signals across all axes
        convolve_axis(&mut vol, &kernel, Axis(0));
        convolve_axis(&mut vol, &kernel, Axis(1));
        convolve_axis(&mut vol, &kernel, Axis(2));

        for &val in vol.iter() {
            assert_near(val, 42.0);
        }
    }

    #[test]
    fn test_convolve_axis_impulse_response() {
        let mut vol = Array3::zeros((7, 1, 1));
        vol[[3, 0, 0]] = 1.0;

        let kernel = vec![0.25, 0.5, 0.25];
        convolve_axis(&mut vol, &kernel, Axis(0));

        assert_near(vol[[2, 0, 0]], 0.25);
        assert_near(vol[[3, 0, 0]], 0.50);
        assert_near(vol[[4, 0, 0]], 0.25);
    }

    // --- gaussian_smooth_3d_anisotropic tests ---

    #[test]
    fn test_gaussian_smooth_3d_anisotropic_handles_voxel_spacing() {
        // Anisotropic resolution: thick 4.0mm slices in Z vs high-res 1.0mm in X/Y
        let zooms = [1.0, 1.0, 4.0];
        let sigma_mm = 2.0;

        // Effective voxel sigmas: X=2.0, Y=2.0, Z=0.5
        let k_x = build_1d_gaussian_kernel(sigma_mm / zooms[0]);
        let k_z = build_1d_gaussian_kernel(sigma_mm / zooms[2]);

        // Z kernel radius must be significantly smaller due to thick slice spacing
        assert!(k_z.len() < k_x.len());

        let mut vol = Array3::zeros((9, 9, 9));
        vol[[4, 4, 4]] = 100.0;

        gaussian_smooth_3d_anisotropic(&mut vol, sigma_mm, zooms);

        // High-res X axis (sigma=2.0 voxels) spreads wider than thick-slice Z axis (sigma=0.5 voxels)
        let val_x_neighbor = vol[[5, 4, 4]];
        let val_z_neighbor = vol[[4, 4, 5]];

        assert!(
            val_x_neighbor > val_z_neighbor,
            "Smoothing across higher voxel density (X) should retain more energy at distance 1 than thick Z"
        );
    }

    #[test]
    fn test_gaussian_smooth_3d_zero_sigma_noop() {
        let mut vol = Array3::from_elem((3, 3, 3), 10.0);
        let vol_orig = vol.clone();

        gaussian_smooth_3d_anisotropic(&mut vol, 0.0, [1.0, 1.0, 1.0]);
        assert_eq!(vol, vol_orig);
    }
}
