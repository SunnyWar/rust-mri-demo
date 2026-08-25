use nalgebra::{DMatrix, DVector, SMatrix};
use ndarray::Array3;
use ndarray::Array4;

/// Computes a binary brain mask using Otsu's optimal thresholding algorithm,
/// separating foreground tissue from background noise/skull space.
pub fn generate_otsu_mask(vol: &Array3<f32>, n_bins: usize) -> Array3<f32> {
    let mut min_val = f32::MAX;
    let mut max_val = f32::MIN;

    for &val in vol.iter() {
        if val > 0.0 {
            if val < min_val {
                min_val = val;
            }
            if val > max_val {
                max_val = val;
            }
        }
    }

    if min_val >= max_val {
        return Array3::zeros(vol.raw_dim());
    }

    // Build intensity histogram
    let mut histogram = vec![0u64; n_bins];
    let range = max_val - min_val;
    let mut total_pixels = 0u64;

    for &val in vol.iter() {
        if val > 0.0 {
            let bin = (((val - min_val) / range) * (n_bins - 1) as f32) as usize;
            let bin = bin.min(n_bins - 1);
            histogram[bin] += 1;
            total_pixels += 1;
        }
    }

    if total_pixels == 0 {
        return Array3::zeros(vol.raw_dim());
    }

    // Calculate optimal Otsu threshold maximizing between-class variance
    let mut sum_b = 0.0f64;
    let mut w_b = 0u64;
    let mut max_variance = 0.0f64;
    let mut threshold_bin = 0usize;

    let sum_total: f64 = histogram
        .iter()
        .enumerate()
        .map(|(i, &count)| i as f64 * count as f64)
        .sum();

    for i in 0..n_bins {
        w_b += histogram[i];
        if w_b == 0 {
            continue;
        }
        let w_f = total_pixels - w_b;
        if w_f == 0 {
            break;
        }

        sum_b += i as f64 * histogram[i] as f64;
        let mean_b = sum_b / w_b as f64;
        let mean_f = (sum_total - sum_b) / w_f as f64;

        let variance_between = (w_b as f64) * (w_f as f64) * (mean_b - mean_f) * (mean_b - mean_f);

        if variance_between > max_variance {
            max_variance = variance_between;
            threshold_bin = i;
        }
    }

    let threshold_val = min_val + (threshold_bin as f32 / n_bins as f32) * range;

    // Apply binary mask
    vol.mapv(|val| if val >= threshold_val { 1.0 } else { 0.0 })
}

/// Applies 3D separable Gaussian convolution accounting for physical voxel
/// dimensions (zooms: dx, dy, dz in mm) extracted from NIfTI affine metadata.
pub fn gaussian_smooth_3d_anisotropic(vol: &mut Array3<f32>, sigma_mm: f32, zooms: [f32; 3]) {
    let shape = vol.shape();
    let (nx, ny, nz) = (shape[0], shape[1], shape[2]);

    // Compute axis-specific kernel sigmas in voxel space: sigma_vox = sigma_mm / voxel_spacing
    let sigmas_vox = [
        sigma_mm / zooms[0],
        sigma_mm / zooms[1],
        sigma_mm / zooms[2],
    ];

    // Convolve Axis 0 (X)
    let kernel_x = build_1d_gaussian_kernel(sigmas_vox[0]);
    let mut temp = vol.clone();
    for y in 0..ny {
        for z in 0..nz {
            for x in 0..nx {
                let mut sum = 0.0;
                let mut weight = 0.0;
                let k_len = kernel_x.len() as isize / 2;
                for (i, &w) in kernel_x.iter().enumerate() {
                    let ix = x as isize + i as isize - k_len;
                    if ix >= 0 && ix < nx as isize {
                        sum += vol[[ix as usize, y, z]] * w;
                        weight += w;
                    }
                }
                temp[[x, y, z]] = if weight > 0.0 { sum / weight } else { 0.0 };
            }
        }
    }
    *vol = temp.clone();

    // Convolve Axis 1 (Y)
    let kernel_y = build_1d_gaussian_kernel(sigmas_vox[1]);
    for x in 0..nx {
        for z in 0..nz {
            for y in 0..ny {
                let mut sum = 0.0;
                let mut weight = 0.0;
                let k_len = kernel_y.len() as isize / 2;
                for (i, &w) in kernel_y.iter().enumerate() {
                    let iy = y as isize + i as isize - k_len;
                    if iy >= 0 && iy < ny as isize {
                        sum += vol[[x, iy as usize, z]] * w;
                        weight += w;
                    }
                }
                temp[[x, y, z]] = if weight > 0.0 { sum / weight } else { 0.0 };
            }
        }
    }
    *vol = temp.clone();

    // Convolve Axis 2 (Z)
    let kernel_z = build_1d_gaussian_kernel(sigmas_vox[2]);
    for x in 0..nx {
        for y in 0..ny {
            for z in 0..nz {
                let mut sum = 0.0;
                let mut weight = 0.0;
                let k_len = kernel_z.len() as isize / 2;
                for (i, &w) in kernel_z.iter().enumerate() {
                    let iz = z as isize + i as isize - k_len;
                    if iz >= 0 && iz < nz as isize {
                        sum += vol[[x, y, iz as usize]] * w;
                        weight += w;
                    }
                }
                temp[[x, y, z]] = if weight > 0.0 { sum / weight } else { 0.0 };
            }
        }
    }
    *vol = temp;
}

fn build_1d_gaussian_kernel(sigma_vox: f32) -> Vec<f32> {
    if sigma_vox <= 0.0 {
        return vec![1.0];
    }
    let radius = (3.0 * sigma_vox).ceil() as usize;
    let size = 2 * radius + 1;
    let mut kernel = vec![0.0; size];
    let two_sigma_sq = 2.0 * sigma_vox * sigma_vox;

    for i in 0..size {
        let x = i as f32 - radius as f32;
        kernel[i] = (-x * x / two_sigma_sq).exp();
    }
    kernel
}

/// Fits a 3x3 diffusion tensor using Weighted Linear Least Squares (WLLS)
/// variance weighting (W = diag(S^2)) to correct for log-transformation noise heteroscedasticity.
pub fn compute_dti_fa_wlls(
    dwi: &Array4<f32>,
    gradients: &[[f32; 3]],
    bvalue: f32,
    mask: Option<&Array3<f32>>,
) -> Array3<f32> {
    let shape = dwi.shape();
    let (nx, ny, nz, n_dirs) = (shape[0], shape[1], shape[2], shape[3]);

    // Build design matrix X (N x 7): [ -b*gx^2, -b*gy^2, -b*gz^2, -2b*gx*gy, -2b*gx*gz, -2b*gy*gz, 1 ]
    let mut x_mat = DMatrix::<f32>::zeros(n_dirs, 7);
    for i in 0..n_dirs {
        let g = gradients[i];
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

    let mut fa_map = Array3::<f32>::zeros((nx, ny, nz));

    for x in 0..nx {
        for y in 0..ny {
            for z in 0..nz {
                if let Some(m) = mask
                    && m[[x, y, z]] <= 0.0
                {
                    continue;
                }

                let mut y_vec = DVector::<f32>::zeros(n_dirs);
                let mut weights = DVector::<f32>::zeros(n_dirs);
                let mut valid = true;

                for i in 0..n_dirs {
                    let s = dwi[[x, y, z, i]];
                    if s <= 0.0 {
                        valid = false;
                        break;
                    }
                    y_vec[i] = s.ln();
                    // Weight W_ii = S_i^2 (heteroscedasticity variance scaling)
                    weights[i] = s * s;
                }

                if !valid {
                    continue;
                }

                // Weighted system: (X^T W X) beta = X^T W y
                let mut wx = x_mat.clone();
                for i in 0..n_dirs {
                    let w = weights[i];
                    for j in 0..7 {
                        wx[(i, j)] *= w;
                    }
                }

                let xt_w_x = x_mat.transpose() * &wx;
                let xt_w_y = wx.transpose() * &y_vec;

                if let Some(beta) = xt_w_x.lu().solve(&xt_w_y) {
                    // Extract tensor components
                    let (dxx, dyy, dzz) = (beta[0], beta[1], beta[2]);
                    let (dxy, dxz, dyz) = (beta[3], beta[4], beta[5]);

                    let tensor =
                        SMatrix::<f32, 3, 3>::new(dxx, dxy, dxz, dxy, dyy, dyz, dxz, dyz, dzz);

                    let eigen = tensor.symmetric_eigen();
                    let l1 = eigen.eigenvalues[0].max(0.0);
                    let l2 = eigen.eigenvalues[1].max(0.0);
                    let l3 = eigen.eigenvalues[2].max(0.0);

                    let mean_diff = (l1 + l2 + l3) / 3.0;
                    let num = ((l1 - mean_diff).powi(2)
                        + (l2 - mean_diff).powi(2)
                        + (l3 - mean_diff).powi(2))
                    .sqrt();
                    let denom = (l1 * l1 + l2 * l2 + l3 * l3).sqrt();

                    let fa = if denom > 1e-6 {
                        (1.5f32).sqrt() * (num / denom)
                    } else {
                        0.0
                    };

                    fa_map[[x, y, z]] = fa.min(1.0);
                }
            }
        }
    }

    fa_map
}
