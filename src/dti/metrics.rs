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

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array4;

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

    // --- Fractional Anisotropy (FA) ---

    #[test]
    fn isotropic_tensor_has_zero_fa() {
        let eig = TensorEigenDecomp {
            eigenvalues: [1.0, 1.0, 1.0],
        };
        assert_near(fractional_anisotropy(&eig), 0.0);
    }

    #[test]
    fn fully_anisotropic_tensor_has_fa_near_one() {
        let eig = TensorEigenDecomp {
            eigenvalues: [1.0, 0.0, 0.0],
        };
        assert_near(fractional_anisotropy(&eig), 1.0);
    }

    #[test]
    fn zero_eigenvalues_fa_is_zero() {
        // Tests denom <= 1e-6 guard
        let eig = TensorEigenDecomp {
            eigenvalues: [0.0, 0.0, 0.0],
        };
        assert_eq!(fractional_anisotropy(&eig), 0.0);
    }

    #[test]
    fn planar_anisotropy_fa() {
        // Oblate / planar tensor (l1 = l2 > l3)
        // FA = sqrt(0.5) ≈ 0.7071068
        let eig = TensorEigenDecomp {
            eigenvalues: [1.0, 1.0, 0.0],
        };
        assert_near(fractional_anisotropy(&eig), (0.5f32).sqrt());
    }

    #[test]
    fn fa_is_clamped_to_one() {
        let eig = TensorEigenDecomp {
            eigenvalues: [100.0, 0.0, 0.0],
        };
        assert!(fractional_anisotropy(&eig) <= 1.0);
    }

    // --- Mean Diffusivity (MD) ---

    #[test]
    fn mean_diffusivity_is_average_of_eigenvalues() {
        let eig = TensorEigenDecomp {
            eigenvalues: [3.0, 2.0, 1.0],
        };
        assert_near(mean_diffusivity(&eig), 2.0);
    }

    #[test]
    fn mean_diffusivity_zero_eigenvalues() {
        let eig = TensorEigenDecomp {
            eigenvalues: [0.0, 0.0, 0.0],
        };
        assert_near(mean_diffusivity(&eig), 0.0);
    }

    // --- Axial Diffusivity (AD) ---

    #[test]
    fn axial_diffusivity_returns_principal_eigenvalue() {
        let eig = TensorEigenDecomp {
            eigenvalues: [0.0020, 0.0005, 0.0002],
        };
        assert_near(axial_diffusivity(&eig), 0.0020);
    }

    // --- Radial Diffusivity (RD) ---

    #[test]
    fn radial_diffusivity_averages_secondary_eigenvalues() {
        let eig = TensorEigenDecomp {
            eigenvalues: [0.0020, 0.0006, 0.0004],
        };
        // (0.0006 + 0.0004) / 2 = 0.0005
        assert_near(radial_diffusivity(&eig), 0.0005);
    }

    // --- Volume Mapping (compute_scalar_map) ---

    #[test]
    fn compute_scalar_map_evaluates_fitted_voxels_and_skips_zero_tensors() {
        let (nx, ny, nz) = (2, 2, 1);
        let mut field = TensorField::zeros((nx, ny, nz, 6));

        // Voxel (0,0,0): Prolate anisotropic tensor
        // Dxx=0.002, Dyy=0.0005, Dzz=0.0005 -> Eigs: [0.002, 0.0005, 0.0005]
        field[[0, 0, 0, 0]] = 0.002;
        field[[0, 0, 0, 1]] = 0.0005;
        field[[0, 0, 0, 2]] = 0.0005;

        // Voxel (1,0,0): Isotropic tensor
        // Dxx=0.001, Dyy=0.001, Dzz=0.001 -> Eigs: [0.001, 0.001, 0.001]
        field[[1, 0, 0, 0]] = 0.001;
        field[[1, 0, 0, 1]] = 0.001;
        field[[1, 0, 0, 2]] = 0.001;

        // Voxels (0,1,0) and (1,1,0) remain all zeros (unfit/masked)

        let fa_map = compute_scalar_map(&field, fractional_anisotropy);
        let md_map = compute_scalar_map(&field, mean_diffusivity);
        let ad_map = compute_scalar_map(&field, axial_diffusivity);
        let rd_map = compute_scalar_map(&field, radial_diffusivity);

        // Verify shape matches volume space (nx, ny, nz)
        assert_eq!(fa_map.shape(), &[2, 2, 1]);

        // Masked/unfit voxels stay 0.0
        assert_eq!(fa_map[[0, 1, 0]], 0.0);
        assert_eq!(fa_map[[1, 1, 0]], 0.0);
        assert_eq!(md_map[[0, 1, 0]], 0.0);

        // Voxel (0,0,0) metrics
        let expected_fa_000 = fractional_anisotropy(&TensorEigenDecomp {
            eigenvalues: [0.002, 0.0005, 0.0005],
        });
        assert_near(fa_map[[0, 0, 0]], expected_fa_000);
        assert_near(md_map[[0, 0, 0]], 0.001);
        assert_near(ad_map[[0, 0, 0]], 0.002);
        assert_near(rd_map[[0, 0, 0]], 0.0005);

        // Voxel (1,0,0) metrics (Isotropic)
        assert_near(fa_map[[1, 0, 0]], 0.0);
        assert_near(md_map[[1, 0, 0]], 0.001);
        assert_near(ad_map[[1, 0, 0]], 0.001);
        assert_near(rd_map[[1, 0, 0]], 0.001);
    }
}
