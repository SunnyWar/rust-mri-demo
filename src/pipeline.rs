use crate::Cli;
use crate::dti::{
    axial_diffusivity, compute_scalar_map, fit_tensor_field, fractional_anisotropy,
    mean_diffusivity, radial_diffusivity,
};
use crate::imaging::io::{load_gradients, save_scalar_map};
use crate::imaging::mask::generate_otsu_mask;
use crate::imaging::smooth::gaussian_smooth_3d_anisotropic;
use crate::{ProcessError, imaging::nifti::load_nifti};
use std::path::Path;
use std::time::Instant;

pub fn process_file(file: &Path, args: &Cli) -> Result<(), ProcessError> {
    let start = Instant::now();

    let file_str = file.to_str().ok_or_else(|| ProcessError::UnexpectedShape {
        path: file.to_owned(),
        ndim: 0,
    })?;

    let vol = load_nifti(file_str)?;
    let shape = vol.data.shape().to_vec();

    let pixdim = vol.header.pixdim;
    let zooms = [
        if pixdim[1] > 0.0 { pixdim[1] } else { 1.0 },
        if pixdim[2] > 0.0 { pixdim[2] } else { 1.0 },
        if pixdim[3] > 0.0 { pixdim[3] } else { 1.0 },
    ];

    if shape.len() == 4 {
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

        let field =
            fit_tensor_field(&dwi, &gradients, args.bvalue, Some(&mask)).map_err(|reason| {
                ProcessError::TensorFitFailed {
                    path: file.to_owned(),
                    reason,
                }
            })?;

        let mut fa_map = compute_scalar_map(&field, fractional_anisotropy);
        gaussian_smooth_3d_anisotropic(&mut fa_map, args.sigma_fa, zooms);
        save_scalar_map(&fa_map, file, "fa_processed", &vol.header)?;

        if args.emit_md {
            let mut md_map = compute_scalar_map(&field, mean_diffusivity);
            gaussian_smooth_3d_anisotropic(&mut md_map, args.sigma_fa, zooms);
            save_scalar_map(&md_map, file, "md_processed", &vol.header)?;
        }

        if args.emit_ad {
            let mut ad_map = compute_scalar_map(&field, axial_diffusivity);
            gaussian_smooth_3d_anisotropic(&mut ad_map, args.sigma_fa, zooms);
            save_scalar_map(&ad_map, file, "ad_processed", &vol.header)?;
        }

        if args.emit_rd {
            let mut rd_map = compute_scalar_map(&field, radial_diffusivity);
            gaussian_smooth_3d_anisotropic(&mut rd_map, args.sigma_fa, zooms);
            save_scalar_map(&rd_map, file, "rd_processed", &vol.header)?;
        }
    } else if shape.len() == 3 {
        let mut struct_vol = vol
            .data
            .into_dimensionality::<ndarray::Ix3>()
            .map_err(|_| ProcessError::UnexpectedShape {
                path: file.to_owned(),
                ndim: shape.len(),
            })?;

        let mask = generate_otsu_mask(&struct_vol, 256);
        gaussian_smooth_3d_anisotropic(&mut struct_vol, args.sigma_3d, zooms);

        struct_vol.zip_mut_with(&mask, |val, &m| {
            if m == 0.0 {
                *val = 0.0;
            }
        });

        save_scalar_map(&struct_vol, file, "smoothed_processed", &vol.header)?;
    } else {
        return Err(ProcessError::UnexpectedShape {
            path: file.to_owned(),
            ndim: shape.len(),
        });
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
