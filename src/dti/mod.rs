mod design;
mod fit;
mod metrics;
mod types;

pub use fit::fit_tensor_field;
pub use metrics::{
    axial_diffusivity, compute_scalar_map, fractional_anisotropy, mean_diffusivity,
    radial_diffusivity,
};
