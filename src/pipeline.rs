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

pub fn process_file(file: &Path, args: &Cli) -> Result<(), ProcessError> {
    let start = Instant::now();

    let file_str = file.to_str().ok_or_else(|| ProcessError::InvalidPath {
        path: file.to_owned(),
    })?;

    let vol = load_nifti(file_str)?;
    let ndim = vol.data.ndim(); // Read rank directly without holding a slice borrow
    let zooms = extract_zooms(&vol.header.pixdim);

    match ndim {
        4 => {
            let dwi = vol
                .data
                .into_dimensionality::<ndarray::Ix4>()
                .map_err(|_| ProcessError::UnexpectedShape {
                    path: file.to_owned(),
                    ndim,
                })?;

            run_dwi_pipeline(file, &vol.header, dwi, zooms, args)?;
        }
        3 => {
            let struct_vol = vol
                .data
                .into_dimensionality::<ndarray::Ix3>()
                .map_err(|_| ProcessError::UnexpectedShape {
                    path: file.to_owned(),
                    ndim,
                })?;

            run_structural_pipeline(file, &vol.header, struct_vol, zooms, args.sigma_3d)?;
        }
        _ => {
            return Err(ProcessError::UnexpectedShape {
                path: file.to_owned(),
                ndim,
            });
        }
    }

    println!(
        "Processed: {} ({:.3}s)",
        file.file_name()
            .and_then(|f| f.to_str())
            .unwrap_or("<invalid filename>"),
        start.elapsed().as_secs_f32()
    );

    Ok(())
}

fn run_dwi_pipeline(
    file: &Path,
    header: &NiftiHeader,
    dwi: ndarray::Array4<f32>,
    zooms: [f32; 3],
    args: &Cli,
) -> Result<(), ProcessError> {
    let b0 = dwi.index_axis(ndarray::Axis(3), 0).to_owned();
    let mask = generate_otsu_mask(&b0, 256);
    let gradients = load_gradients(file, dwi.shape()[3])?;

    let field = fit_tensor_field(&dwi, &gradients, args.bvalue, Some(&mask)).map_err(|reason| {
        ProcessError::TensorFitFailed {
            path: file.to_owned(),
            reason,
        }
    })?;

    // Map output flags to scalar functions and suffix names
    let outputs = [
        (true, "fa_processed", fractional_anisotropy as fn(&_) -> _),
        (args.emit_md, "md_processed", mean_diffusivity),
        (args.emit_ad, "ad_processed", axial_diffusivity),
        (args.emit_rd, "rd_processed", radial_diffusivity),
    ];

    for (enabled, suffix, scalar_fn) in outputs {
        if enabled {
            let mut scalar_map = compute_scalar_map(&field, scalar_fn);
            gaussian_smooth_3d_anisotropic(&mut scalar_map, args.sigma_fa, zooms);
            save_scalar_map(&scalar_map, file, suffix, header)?;
        }
    }

    Ok(())
}

fn run_structural_pipeline(
    file: &Path,
    header: &NiftiHeader,
    mut struct_vol: ndarray::Array3<f32>,
    zooms: [f32; 3],
    sigma_3d: f32,
) -> Result<(), ProcessError> {
    let mask = generate_otsu_mask(&struct_vol, 256);

    gaussian_smooth_3d_anisotropic(&mut struct_vol, sigma_3d, zooms);
    apply_mask(&mut struct_vol, &mask);

    save_scalar_map(&struct_vol, file, "smoothed_processed", header)?;

    Ok(())
}
