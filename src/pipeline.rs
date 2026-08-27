use crate::cli::Cli;
use crate::dti::types::DtiError;
use crate::dti::{
    axial_diffusivity, compute_scalar_map, fit_tensor_field, fractional_anisotropy,
    mean_diffusivity, radial_diffusivity,
};
use crate::imaging::io::{load_gradients, save_scalar_map};
use crate::imaging::mask::generate_otsu_mask;
use crate::imaging::nifti::load_nifti;
use crate::imaging::smooth::gaussian_smooth_3d_anisotropic;
use nifti::NiftiHeader;
use std::path::{Path, PathBuf};
use std::time::Instant;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ProcessError {
    #[error("Unexpected shape for file {path}: {ndim} dimensions")]
    UnexpectedShape { path: PathBuf, ndim: usize },

    #[error("Tensor fit failed for file {path}: {reason}")]
    TensorFitFailed { path: PathBuf, reason: DtiError },

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Invalid path: {path}")]
    InvalidPath { path: PathBuf },
}

/// Configuration for processing 4D DWI volumes.
struct DwiProcessingConfig<'a> {
    dwi: ndarray::Array4<f32>,
    gradients: Vec<[f32; 3]>,
    bvalue: f32,
    mask: ndarray::Array3<f32>,
    zooms: [f32; 3],
    sigma_fa: f32,
    header: &'a NiftiHeader,
    file: &'a Path,
    emit_md: bool,
    emit_ad: bool,
    emit_rd: bool,
}

/// Configuration for processing 3D structural volumes.
struct StructProcessingConfig<'a> {
    struct_vol: ndarray::Array3<f32>,
    zooms: [f32; 3],
    sigma_3d: f32,
    header: &'a NiftiHeader,
    file: &'a Path,
}

/// Extracts non-zero pixdim values, defaulting to 1.0 if zero or negative.
fn extract_zooms(pixdim: &[f32; 8]) -> [f32; 3] {
    [
        if pixdim[1] > 0.0 { pixdim[1] } else { 1.0 },
        if pixdim[2] > 0.0 { pixdim[2] } else { 1.0 },
        if pixdim[3] > 0.0 { pixdim[3] } else { 1.0 },
    ]
}

/// Applies a mask to a volume, zeroing out masked voxels.
fn apply_mask(vol: &mut ndarray::Array3<f32>, mask: &ndarray::Array3<f32>) {
    vol.zip_mut_with(mask, |val, &m| {
        if m == 0.0 {
            *val = 0.0;
        }
    });
}

/// Processes a 4D DWI volume.
fn process_4d(config: DwiProcessingConfig) -> Result<(), ProcessError> {
    let DwiProcessingConfig {
        dwi,
        gradients,
        bvalue,
        mask,
        zooms,
        sigma_fa,
        header,
        file,
        emit_md,
        emit_ad,
        emit_rd,
    } = config;

    let field = fit_tensor_field(&dwi, &gradients, bvalue, Some(&mask)).map_err(|reason| {
        ProcessError::TensorFitFailed {
            path: file.to_owned(),
            reason,
        }
    })?;

    let mut fa_map = compute_scalar_map(&field, fractional_anisotropy);
    gaussian_smooth_3d_anisotropic(&mut fa_map, sigma_fa, zooms);
    save_scalar_map(&fa_map, file, "fa_processed", header)?;

    if emit_md {
        let mut md_map = compute_scalar_map(&field, mean_diffusivity);
        gaussian_smooth_3d_anisotropic(&mut md_map, sigma_fa, zooms);
        save_scalar_map(&md_map, file, "md_processed", header)?;
    }

    if emit_ad {
        let mut ad_map = compute_scalar_map(&field, axial_diffusivity);
        gaussian_smooth_3d_anisotropic(&mut ad_map, sigma_fa, zooms);
        save_scalar_map(&ad_map, file, "ad_processed", header)?;
    }

    if emit_rd {
        let mut rd_map = compute_scalar_map(&field, radial_diffusivity);
        gaussian_smooth_3d_anisotropic(&mut rd_map, sigma_fa, zooms);
        save_scalar_map(&rd_map, file, "rd_processed", header)?;
    }

    Ok(())
}

/// Processes a 3D structural volume.
fn process_3d(config: StructProcessingConfig) -> Result<(), ProcessError> {
    let StructProcessingConfig {
        mut struct_vol,
        zooms,
        sigma_3d,
        header,
        file,
    } = config;

    let mask = generate_otsu_mask(&struct_vol, 256);
    gaussian_smooth_3d_anisotropic(&mut struct_vol, sigma_3d, zooms);
    apply_mask(&mut struct_vol, &mask);
    save_scalar_map(&struct_vol, file, "smoothed_processed", header)?;
    Ok(())
}

pub fn process_file(file: &Path, args: &Cli) -> Result<(), ProcessError> {
    let start = Instant::now();

    let file_str = file.to_str().ok_or(ProcessError::InvalidPath {
        path: file.to_owned(),
    })?;

    let vol = load_nifti(file_str)?;
    let shape = vol.data.shape().to_vec();
    let zooms = extract_zooms(&vol.header.pixdim);

    match shape.len() {
        4 => {
            let dwi = vol
                .data
                .into_dimensionality::<ndarray::Ix4>()
                .map_err(|_| ProcessError::UnexpectedShape {
                    path: file.to_owned(),
                    ndim: shape.len(),
                })?;
            let b0 = dwi.index_axis(ndarray::Axis(3), 0).to_owned();
            let mask = generate_otsu_mask(&b0, 256);
            let gradients = load_gradients(file, shape[3])?;

            let config = DwiProcessingConfig {
                dwi,
                gradients,
                bvalue: args.bvalue,
                mask,
                zooms,
                sigma_fa: args.sigma_fa,
                header: &vol.header,
                file,
                emit_md: args.emit_md,
                emit_ad: args.emit_ad,
                emit_rd: args.emit_rd,
            };
            process_4d(config)?;
        }
        3 => {
            let struct_vol = vol
                .data
                .into_dimensionality::<ndarray::Ix3>()
                .map_err(|_| ProcessError::UnexpectedShape {
                    path: file.to_owned(),
                    ndim: shape.len(),
                })?;

            let config = StructProcessingConfig {
                struct_vol,
                zooms,
                sigma_3d: args.sigma_3d,
                header: &vol.header,
                file,
            };
            process_3d(config)?;
        }
        _ => {
            return Err(ProcessError::UnexpectedShape {
                path: file.to_owned(),
                ndim: shape.len(),
            });
        }
    }

    let elapsed = start.elapsed();
    println!(
        "Processed: {} ({:.3}s)",
        file.file_name()
            .and_then(|f| f.to_str())
            .unwrap_or("<invalid filename>"),
        elapsed.as_secs_f32()
    );

    Ok(())
}
