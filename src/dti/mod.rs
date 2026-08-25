mod design;
mod fit;
mod metrics;
mod types;

pub use fit::fit_tensor_field;
pub use metrics::fa_map;

use ndarray::{Array3, Array4};

/// Backward-compatible shim matching the original `compute_dti_fa_wlls`
/// signature.
pub fn compute_dti_fa_wlls(
    dwi: &Array4<f32>,
    gradients: &[[f32; 3]],
    bvalue: f32,
    mask: Option<&Array3<f32>>,
) -> Array3<f32> {
    match fit_tensor_field(dwi, gradients, bvalue, mask) {
        Ok(field) => fa_map(&field),
        Err(_) => Array3::zeros((dwi.shape()[0], dwi.shape()[1], dwi.shape()[2])),
    }
}
