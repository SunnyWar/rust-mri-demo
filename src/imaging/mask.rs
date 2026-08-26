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

/// Finds the histogram bin that maximizes between-class variance (Otsu's method).
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

    for (i, &count) in histogram.iter().enumerate() {
        w_b += count;
        if w_b == 0 {
            continue;
        }
        let w_b = w_b as f64;
        let w_f = total_pixels - w_b;
        if w_f <= 0.0 {
            break;
        }

        sum_b += i as f64 * count as f64;

        // Between-class variance, reformulated to need one division instead
        // of two (no separate mean_b / mean_f):
        //   variance = w_b * w_f * (mean_b - mean_f)^2
        //            = (sum_b * total_pixels - sum_total * w_b)^2 / (w_b * w_f)
        let diff = sum_b * total_pixels - sum_total * w_b;
        let variance = diff * diff / (w_b * w_f);

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
    let threshold_val = min_val + (threshold_bin as f32 / n_bins as f32) * range;

    vol.mapv(|val| if val >= threshold_val { 1.0 } else { 0.0 })
}
