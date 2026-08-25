use crate::imaging::Volume;
use nalgebra::Matrix3;
use ndarray::s;
use ndarray::{Array3, Axis};
use rayon::prelude::*;
use std::f32::consts::PI;

pub fn perfusion_transform(vol: &mut Volume, alpha: f32) {
    vol.data.par_iter_mut().for_each(|v| {
        *v = *v * (-alpha * *v).exp();
    });
}

/// Generates a normalized 1D Gaussian kernel for a given standard deviation (sigma in voxels).
fn gaussian_kernel_1d(sigma: f32) -> Vec<f32> {
    let radius = (3.0 * sigma).ceil() as usize;
    let size = 2 * radius + 1;
    let mut kernel = vec![0.0; size];
    let two_sigma_sq = 2.0 * sigma * sigma;
    let norm = 1.0 / ((2.0 * PI).sqrt() * sigma);

    let mut sum = 0.0;
    for i in 0..size {
        let x = i as f32 - radius as f32;
        kernel[i] = norm * (-x * x / two_sigma_sq).exp();
        sum += kernel[i];
    }
    // Normalize kernel so unit area is preserved
    for k in kernel.iter_mut() {
        *k /= sum;
    }
    kernel
}

/// Applies a 1D convolution along a specific axis of a 3D array buffer.
fn blur_axis(src: &Array3<f32>, axis: usize, kernel: &[f32]) -> Array3<f32> {
    let shape = src.shape();
    let (nx, ny, nz) = (shape[0], shape[1], shape[2]);
    let radius = kernel.len() / 2;
    let mut dst = Array3::zeros((nx, ny, nz));

    match axis {
        0 => {
            // Axis 0 (X-blur)
            dst.axis_iter_mut(Axis(1))
                .zip(src.axis_iter(Axis(1)))
                .par_bridge()
                .for_each(|(mut dst_slice, src_slice)| {
                    for z in 0..nz {
                        for x in 0..nx {
                            let mut val = 0.0;
                            for (k_idx, &k_val) in kernel.iter().enumerate() {
                                let ix = (x as isize + k_idx as isize - radius as isize)
                                    .clamp(0, nx as isize - 1)
                                    as usize;
                                val += src_slice[[ix, z]] * k_val;
                            }
                            dst_slice[[x, z]] = val;
                        }
                    }
                });
        }
        1 => {
            // Axis 1 (Y-blur)
            dst.axis_iter_mut(Axis(0))
                .zip(src.axis_iter(Axis(0)))
                .par_bridge()
                .for_each(|(mut dst_slice, src_slice)| {
                    for z in 0..nz {
                        for y in 0..ny {
                            let mut val = 0.0;
                            for (k_idx, &k_val) in kernel.iter().enumerate() {
                                let iy = (y as isize + k_idx as isize - radius as isize)
                                    .clamp(0, ny as isize - 1)
                                    as usize;
                                val += src_slice[[iy, z]] * k_val;
                            }
                            dst_slice[[y, z]] = val;
                        }
                    }
                });
        }
        2 => {
            // Axis 2 (Z-blur) - Corrected indexing
            dst.axis_iter_mut(Axis(0))
                .zip(src.axis_iter(Axis(0)))
                .par_bridge()
                .for_each(|(mut dst_slice, src_slice)| {
                    for y in 0..ny {
                        for z in 0..nz {
                            let mut val = 0.0;
                            for (k_idx, &k_val) in kernel.iter().enumerate() {
                                let iz = (z as isize + k_idx as isize - radius as isize)
                                    .clamp(0, nz as isize - 1)
                                    as usize;
                                val += src_slice[[y, iz]] * k_val;
                            }
                            dst_slice[[y, z]] = val;
                        }
                    }
                });
        }
        _ => unreachable!(),
    }
    dst
}

/// Computes a full 3D separable Gaussian blur over dynamic volumes (3D or 4D).
/// Applies 3D separable spatial blur in-place using double-buffered memory.
pub fn gaussian_smooth_3d(vol: &mut Volume, sigma: f32) {
    let kernel = gaussian_kernel_1d(sigma);
    let shape = vol.data.shape();

    if shape.len() == 3 {
        let mut buf_a = vol
            .data
            .clone()
            .into_dimensionality::<ndarray::Ix3>()
            .unwrap();

        // Ping-pong between buf_a and buf_b across the 3 spatial axes
        let buf_b = blur_axis(&buf_a, 0, &kernel);
        buf_a = blur_axis(&buf_b, 1, &kernel);
        let final_buf = blur_axis(&buf_a, 2, &kernel);

        vol.data = final_buf.into_dyn();
    } else if shape.len() == 4 {
        let n_volumes = shape[3];
        let mut smoothed_4d = ndarray::Array4::zeros((shape[0], shape[1], shape[2], n_volumes));

        // Parallelize across 4D gradient/time volumes directly with Rayon
        smoothed_4d
            .axis_iter_mut(Axis(3))
            .zip(vol.data.axis_iter(Axis(3)))
            .par_bridge()
            .for_each(|(mut dst_3d, src_3d)| {
                let slice_3d = src_3d
                    .to_owned()
                    .into_dimensionality::<ndarray::Ix3>()
                    .unwrap();
                let b1 = blur_axis(&slice_3d, 0, &kernel);
                let b2 = blur_axis(&b1, 1, &kernel);
                let b3 = blur_axis(&b2, 2, &kernel);
                dst_3d.assign(&b3);
            });

        vol.data = smoothed_4d.into_dyn();
    }
}

/// Computes a 3D Fractional Anisotropy map from a 4D DWI Volume.
/// Expects gradient directions `gradients` (N x 3 matrix) and b-value `b_val`.
pub fn compute_dti_fa(vol: &Volume, gradients: &[[f32; 3]], b_val: f32) -> Volume {
    let shape = vol.data.shape();
    assert_eq!(shape.len(), 4, "FA computation requires a 4D DWI volume");

    let (nx, ny, nz, n_dirs) = (shape[0], shape[1], shape[2], shape[3]);
    assert_eq!(n_dirs, gradients.len());

    // Pre-compute Pseudoinverse Matrix once outside the voxel loops
    let w_pinv = {
        let mut w_mat = nalgebra::DMatrix::<f32>::zeros(n_dirs - 1, 6);
        for (r, g) in gradients[1..].iter().enumerate() {
            let (gx, gy, gz) = (g[0], g[1], g[2]);
            w_mat[(r, 0)] = gx * gx;
            w_mat[(r, 1)] = 2.0 * gx * gy;
            w_mat[(r, 2)] = 2.0 * gx * gz;
            w_mat[(r, 3)] = gy * gy;
            w_mat[(r, 4)] = 2.0 * gy * gz;
            w_mat[(r, 5)] = gz * gz;
        }
        w_mat
            .pseudo_inverse(1e-6)
            .expect("singular gradient matrix")
    };

    let mut fa_map = Array3::<f32>::zeros((nx, ny, nz));

    // Parallelize tensor fitting across 3D slice planes
    fa_map
        .axis_iter_mut(Axis(2))
        .zip(vol.data.axis_iter(Axis(2)))
        .par_bridge()
        .for_each(|(mut fa_slice, dwi_slice)| {
            // Allocate a stack-based buffer per thread to reuse across voxels
            let mut y_arr = vec![0.0f32; n_dirs - 1];

            for x in 0..nx {
                for y in 0..ny {
                    let s0 = dwi_slice[[x, y, 0]];
                    if s0 <= 10.0 {
                        // Skip low-signal/background voxels
                        fa_slice[[x, y]] = 0.0;
                        continue;
                    }

                    // Populate reusable array without heap allocations inside the loop
                    for i in 1..n_dirs {
                        let si = dwi_slice[[x, y, i]].max(1.0);
                        y_arr[i - 1] = (s0 / si).ln() / b_val;
                    }

                    // Wrap contiguous slice into a Matrix view (Zero allocations)
                    let y_vec = nalgebra::DVectorSlice::from_slice(&y_arr, n_dirs - 1);

                    // Fit Tensor elements d = [Dxx, Dxy, Dxz, Dyy, Dyz, Dzz]
                    let d = &w_pinv * y_vec;

                    // Reconstruct 3x3 symmetric Tensor Matrix
                    let tensor = Matrix3::new(d[0], d[1], d[2], d[1], d[3], d[4], d[2], d[4], d[5]);

                    // Eigendecomposition
                    let eigen = tensor.symmetric_eigen();
                    let l1 = eigen.eigenvalues[0].max(0.0);
                    let l2 = eigen.eigenvalues[1].max(0.0);
                    let l3 = eigen.eigenvalues[2].max(0.0);

                    let mean_diffusivity = (l1 + l2 + l3) / 3.0;
                    let num = (l1 - mean_diffusivity).powi(2)
                        + (l2 - mean_diffusivity).powi(2)
                        + (l3 - mean_diffusivity).powi(2);
                    let denom = l1.powi(2) + l2.powi(2) + l3.powi(2);

                    let fa = if denom > 1e-8 {
                        (1.5 * num / denom).sqrt().clamp(0.0, 1.0)
                    } else {
                        0.0
                    };

                    fa_slice[[x, y]] = fa;
                }
            }
        });

    Volume {
        data: fa_map.into_dyn(),
        header: vol.header.clone(),
    }
}
