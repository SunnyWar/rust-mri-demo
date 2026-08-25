use ndarray::Array3;

use super::fit::{TensorField, tensor_at};
use super::types::TensorEigenDecomp;

/// Fractional anisotropy — how directionally biased the diffusion is (0 =
/// isotropic, 1 = maximally anisotropic).
pub fn fractional_anisotropy(eig: &TensorEigenDecomp) -> f32 {
    let [l1, l2, l3] = eig.eigenvalues;
    let mean_diff = (l1 + l2 + l3) / 3.0;
    let num =
        ((l1 - mean_diff).powi(2) + (l2 - mean_diff).powi(2) + (l3 - mean_diff).powi(2)).sqrt();
    let denom = (l1 * l1 + l2 * l2 + l3 * l3).sqrt();

    if denom > 1e-6 {
        ((1.5f32).sqrt() * (num / denom)).min(1.0)
    } else {
        0.0
    }
}

/// Mean diffusivity — average diffusion rate, direction-independent.
pub fn mean_diffusivity(eig: &TensorEigenDecomp) -> f32 {
    let [l1, l2, l3] = eig.eigenvalues;
    (l1 + l2 + l3) / 3.0
}

/// Axial diffusivity — diffusion along the principal (fastest) direction.
pub fn axial_diffusivity(eig: &TensorEigenDecomp) -> f32 {
    eig.eigenvalues[0]
}

/// Radial diffusivity — average diffusion perpendicular to the principal
/// direction. Commonly used alongside FA/MD to characterize white-matter
/// integrity (e.g. in demyelination).
pub fn radial_diffusivity(eig: &TensorEigenDecomp) -> f32 {
    (eig.eigenvalues[1] + eig.eigenvalues[2]) / 2.0
}

/// Applies a per-voxel scalar metric function across a whole tensor field,
/// skipping voxels whose tensor is all-zero (unfit/masked-out). Adding a new
/// whole-volume map (e.g. a color-FA RGB volume, or a principal eigenvector
/// map for tractography seeding) is a matter of writing one small function
/// like this one, not re-deriving or re-running the tensor fit.
pub fn compute_scalar_map(
    field: &TensorField,
    metric: impl Fn(&TensorEigenDecomp) -> f32,
) -> Array3<f32> {
    let shape = field.shape();
    let (nx, ny, nz) = (shape[0], shape[1], shape[2]);
    let mut out = Array3::<f32>::zeros((nx, ny, nz));

    for x in 0..nx {
        for y in 0..ny {
            for z in 0..nz {
                let tensor = tensor_at(field, x, y, z);
                if tensor.iter().all(|&v| v == 0.0) {
                    continue;
                }
                let eigen = super::fit::eigen_decompose(&tensor);
                out[[x, y, z]] = metric(&eigen);
            }
        }
    }

    out
}

/// Convenience wrapper matching the original `compute_dti_fa_wlls` output
/// shape, now built on top of the reusable tensor field / metric machinery.
pub fn fa_map(field: &TensorField) -> Array3<f32> {
    compute_scalar_map(field, fractional_anisotropy)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn isotropic_tensor_has_zero_fa() {
        let eig = TensorEigenDecomp {
            eigenvalues: [1.0, 1.0, 1.0],
        };
        assert!(fractional_anisotropy(&eig).abs() < 1e-6);
    }

    #[test]
    fn fully_anisotropic_tensor_has_fa_near_one() {
        let eig = TensorEigenDecomp {
            eigenvalues: [1.0, 0.0, 0.0],
        };
        assert!((fractional_anisotropy(&eig) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn mean_diffusivity_is_average_of_eigenvalues() {
        let eig = TensorEigenDecomp {
            eigenvalues: [3.0, 2.0, 1.0],
        };
        assert_eq!(mean_diffusivity(&eig), 2.0);
    }
}
