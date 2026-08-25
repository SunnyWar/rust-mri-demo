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
            let bin = (((val - min_val) / range) * (n_bins - 1) as f32) as usize;
            let bin = bin.min(n_bins - 1);
            histogram[bin] += 1;
        }
    }

    (histogram, min_val, max_val)
}

pub fn otsu_threshold_bin(histogram: &[u64], total_pixels: u64) -> usize {
    let mut sum_b = 0.0f64;
    let mut w_b = 0u64;
    let mut max_variance = 0.0f64;
    let mut threshold_bin = 0usize;

    let sum_total: f64 = histogram
        .iter()
        .enumerate()
        .map(|(i, &count)| i as f64 * count as f64)
        .sum();

    for i in 0..histogram.len() {
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

    threshold_bin
}

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
