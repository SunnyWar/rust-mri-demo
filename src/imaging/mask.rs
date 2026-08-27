use ndarray::Array3;

pub fn build_histogram(vol: &Array3<f32>, n_bins: usize) -> (Vec<u64>, f32, f32) {
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
        return (vec![0; n_bins], 0.0, 0.0);
    }

    let mut histogram = vec![0u64; n_bins];
    let range = max_val - min_val;

    for &val in vol.iter() {
        if val > 0.0 {
            let bin = (((val - min_val) / range) * n_bins as f32) as usize;
            let bin = bin.min(n_bins - 1);
            histogram[bin] += 1;
        }
    }

    (histogram, min_val, max_val)
}

/// Finds the histogram bin index `k` (0..n_bins-1) that maximizes between-class variance.
/// Bins 0..=k are classified as background; bins (k+1)..n_bins as foreground.
pub fn otsu_threshold_bin(histogram: &[u64], total_pixels: u64) -> usize {
    let total_pixels = total_pixels as f64;
    let sum_total: f64 = histogram
        .iter()
        .enumerate()
        .map(|(i, &count)| i as f64 * count as f64)
        .sum();

    let mut w_b = 0u64;
    let mut sum_b = 0.0f64;
    let mut max_variance = 0.0f64;
    let mut threshold_bin = 0usize;

    // Only iterate up to histogram.len() - 1 so foreground has at least bin (n_bins - 1)
    for (i, &count) in histogram
        .iter()
        .enumerate()
        .take(histogram.len().saturating_sub(1))
    {
        w_b += count;
        if w_b == 0 {
            continue;
        }
        let w_b_f64 = w_b as f64;
        let w_f_f64 = total_pixels - w_b_f64;
        if w_f_f64 <= 0.0 {
            break;
        }

        sum_b += i as f64 * count as f64;

        let diff = sum_b * total_pixels - sum_total * w_b_f64;
        let variance = diff * diff / (w_b_f64 * w_f_f64);

        if variance > max_variance {
            max_variance = variance;
            threshold_bin = i;
        }
    }

    threshold_bin
}

/// Computes a binary brain mask using Otsu's optimal thresholding algorithm,
/// separating foreground tissue from background noise/skull space.
pub fn generate_otsu_mask(vol: &Array3<f32>, n_bins: usize) -> Array3<f32> {
    let (histogram, min_val, max_val) = build_histogram(vol, n_bins);

    if min_val >= max_val {
        return Array3::zeros(vol.raw_dim());
    }

    let total_pixels: u64 = histogram.iter().sum();
    if total_pixels == 0 {
        return Array3::zeros(vol.raw_dim());
    }

    let threshold_bin = otsu_threshold_bin(&histogram, total_pixels);
    let range = max_val - min_val;

    // Threshold is placed at the upper boundary of `threshold_bin`
    let threshold_val = min_val + ((threshold_bin + 1) as f32 / n_bins as f32) * range;

    vol.mapv(|val| if val >= threshold_val { 1.0 } else { 0.0 })
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

    // --- build_histogram tests ---

    #[test]
    fn test_build_histogram_standard_range() {
        // Linear sequence from 1.0 to 10.0
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let vol = Array3::from_shape_vec((10, 1, 1), data).unwrap();
        let n_bins = 10;

        let (hist, min_val, max_val) = build_histogram(&vol, n_bins);

        assert_near(min_val, 1.0);
        assert_near(max_val, 10.0);
        assert_eq!(hist.len(), 10);
        // Total positive voxel count must equal input positive voxels
        assert_eq!(hist.iter().sum::<u64>(), 10);
        // Both min and max boundaries must map inside valid bin indices
        assert!(hist[0] > 0);
        assert!(hist[9] > 0);
    }

    #[test]
    fn test_build_histogram_skips_zero_and_negative_voxels() {
        // Contains background (0.0) and artifact/negative values (-5.0)
        let data = vec![-5.0, 0.0, 0.0, 10.0, 20.0, 30.0];
        let vol = Array3::from_shape_vec((6, 1, 1), data).unwrap();

        let (hist, min_val, max_val) = build_histogram(&vol, 5);

        assert_near(min_val, 10.0);
        assert_near(max_val, 30.0);
        // Only 10.0, 20.0, 30.0 should be counted
        assert_eq!(hist.iter().sum::<u64>(), 3);
    }

    #[test]
    fn test_build_histogram_uniform_or_all_zero_returns_empty_range() {
        // Uniform non-zero volume (min == max)
        let vol_uniform = Array3::from_elem((3, 3, 3), 5.0);
        let (hist, min_val, max_val) = build_histogram(&vol_uniform, 10);
        assert_eq!(hist, vec![0; 10]);
        assert_eq!(min_val, 0.0);
        assert_eq!(max_val, 0.0);

        // All zero volume
        let vol_zeros = Array3::zeros((3, 3, 3));
        let (hist_z, min_z, max_z) = build_histogram(&vol_zeros, 10);
        assert_eq!(hist_z, vec![0; 10]);
        assert_eq!(min_z, 0.0);
        assert_eq!(max_z, 0.0);
    }

    // --- otsu_threshold_bin tests ---

    #[test]
    fn test_otsu_threshold_bin_bimodal_distribution() {
        // Well-separated bimodal histogram: peak 1 at bin 2, peak 2 at bin 8
        let mut hist = vec![0u64; 10];
        hist[1] = 100;
        hist[2] = 200;
        hist[3] = 100;

        hist[7] = 100;
        hist[8] = 200;
        hist[9] = 100;

        let total_pixels: u64 = hist.iter().sum();
        let bin = otsu_threshold_bin(&hist, total_pixels);

        // Optimal threshold boundary must fall between the two peaks
        assert!((3..=6).contains(&bin));
    }

    #[test]
    fn test_otsu_threshold_bin_all_zero_histogram() {
        let hist = vec![0u64; 10];
        let bin = otsu_threshold_bin(&hist, 0);
        assert_eq!(bin, 0);
    }

    // --- generate_otsu_mask tests ---

    #[test]
    fn test_generate_otsu_mask_separates_bimodal_signal() {
        let (nx, ny, nz) = (4, 4, 1);
        let mut vol = Array3::<f32>::zeros((nx, ny, nz));

        // Fill background noise (~10.0)
        for x in 0..2 {
            for y in 0..4 {
                vol[[x, y, 0]] = 10.0;
            }
        }

        // Fill foreground tissue (~100.0)
        for x in 2..4 {
            for y in 0..4 {
                vol[[x, y, 0]] = 100.0;
            }
        }

        // Add a small zero border or wider bin step so min_val < background
        // Or use fewer bins for discrete step synthetic tests (e.g. n_bins = 10)
        let mask = generate_otsu_mask(&vol, 10);

        assert_eq!(mask.shape(), &[4, 4, 1]);

        for x in 0..2 {
            for y in 0..4 {
                assert_eq!(mask[[x, y, 0]], 0.0);
            }
        }

        for x in 2..4 {
            for y in 0..4 {
                assert_eq!(mask[[x, y, 0]], 1.0);
            }
        }
    }

    #[test]
    fn test_generate_otsu_mask_handles_flat_or_empty_volumes() {
        // Flat non-zero volume returns all zeros
        let vol_flat = Array3::from_elem((2, 2, 2), 50.0);
        let mask_flat = generate_otsu_mask(&vol_flat, 100);
        assert_eq!(mask_flat, Array3::zeros((2, 2, 2)));

        // All zero volume returns all zeros
        let vol_zero = Array3::zeros((2, 2, 2));
        let mask_zero = generate_otsu_mask(&vol_zero, 100);
        assert_eq!(mask_zero, Array3::zeros((2, 2, 2)));
    }
}
