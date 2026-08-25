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
