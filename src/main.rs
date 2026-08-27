mod dti;
mod imaging;

use clap::Parser;
use imaging::io::load_gradients;
use ndarray::Array3;
use rayon::prelude::*;
use std::{
    fs,
    path::{Path, PathBuf},
    time::Instant,
};

use crate::{
    dti::{
        axial_diffusivity, compute_scalar_map, fit_tensor_field, fractional_anisotropy,
        mean_diffusivity, radial_diffusivity, types::DtiError,
    },
    imaging::{
        io::find_nii_gz_files, mask::generate_otsu_mask, nifti::load_nifti,
        smooth::gaussian_smooth_3d_anisotropic,
    },
};

use nifti::{NiftiHeader, writer::WriterOptions};

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    root: String,

    #[arg(short = 's', long = "sigma-3d", default_value_t = 1.5)]
    sigma_3d: f32,

    #[arg(long = "sigma-fa", default_value_t = 1.0)]
    sigma_fa: f32,

    #[arg(short, long, default_value_t = 1000.0)]
    bvalue: f32,

    /// Also output a mean diffusivity map
    #[arg(long)]
    emit_md: bool,

    /// Also output an axial diffusivity map
    #[arg(long)]
    emit_ad: bool,

    /// Also output a radial diffusivity map
    #[arg(long)]
    emit_rd: bool,
}

#[derive(Debug)]
enum ProcessError {
    UnexpectedShape { path: PathBuf, ndim: usize },
    TensorFitFailed { path: PathBuf, reason: DtiError },
    Io(std::io::Error),
}

fn get_processed_output_path(file: &Path, suffix: &str) -> PathBuf {
    let filename = file.file_name().unwrap().to_string_lossy();

    // Strip trailing .nii.gz or .nii
    let stem = if let Some(stripped) = filename.strip_suffix(".nii.gz") {
        stripped
    } else if let Some(stripped) = filename.strip_suffix(".nii") {
        stripped
    } else {
        &filename
    };

    file.with_file_name(format!("{}_{}.nii.gz", stem, suffix))
}

fn process_file(file: &Path, args: &Cli) -> Result<(), ProcessError> {
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

fn save_scalar_map(
    vol: &Array3<f32>,
    file: &Path,
    suffix: &str,
    header: &NiftiHeader,
) -> Result<(), ProcessError> {
    let out = get_processed_output_path(file, suffix);

    WriterOptions::new(&out)
        .reference_header(header)
        .write_nifti(vol)
        .map_err(|e| {
            ProcessError::Io(std::io::Error::other(format!(
                "Failed to write NIfTI file {}: {}",
                out.display(),
                e
            )))
        })?;

    Ok(())
}

fn main() {
    let args = Cli::parse();
    let root = Path::new(&args.root);
    let all_start = Instant::now();
    let files = find_nii_gz_files(root);

    println!("Found {} NIfTI files", files.len());
    println!("Processing dataset in parallel with Rayon...");

    // Parallelize pipeline across files concurrently
    let results: Vec<_> = files
        .par_iter()
        .map(|file| (file, process_file(file, &args)))
        .collect();

    for (file, result) in &results {
        if let Err(e) = result {
            eprintln!("FAILED: {} — {:?}", file.display(), e);
        }
    }
    let failed = results.iter().filter(|(_, r)| r.is_err()).count();
    println!(
        "Files processed: {} ({} failed)",
        results.len() - failed,
        failed
    );

    let all_elapsed = all_start.elapsed();
    println!("----------------------------------------");
    println!(
        "All processing completed in {:.3} seconds",
        all_elapsed.as_secs_f32()
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32;
    use std::fs::File;
    use std::io::Write;
    use tempfile::tempdir;

    // --- get_processed_output_path tests ---

    #[test]
    fn test_get_processed_output_path_strips_extensions() {
        let p_gz = Path::new("/data/subject01_dwi.nii.gz");
        let out_fa = get_processed_output_path(p_gz, "fa_processed");
        assert_eq!(
            out_fa,
            PathBuf::from("/data/subject01_dwi_fa_processed.nii.gz")
        );

        let p_nii = Path::new("/data/subject01_t1.nii");
        let out_smooth = get_processed_output_path(p_nii, "smoothed_processed");
        assert_eq!(
            out_smooth,
            PathBuf::from("/data/subject01_t1_smoothed_processed.nii.gz")
        );
    }

    // --- CLI argument parsing tests ---

    #[test]
    fn test_cli_defaults() {
        let args = Cli::try_parse_from(["dti_pipeline", "/path/to/data"]).unwrap();
        assert_eq!(args.root, "/path/to/data");
        assert_eq!(args.sigma_3d, 1.5);
        assert_eq!(args.sigma_fa, 1.0);
        assert_eq!(args.bvalue, 1000.0);
        assert!(!args.emit_md);
        assert!(!args.emit_ad);
        assert!(!args.emit_rd);
    }

    #[test]
    fn test_cli_custom_flags() {
        let args = Cli::try_parse_from([
            "dti_pipeline",
            "/path/to/data",
            "-s",
            "2.5",
            "--sigma-fa",
            "0.8",
            "-b",
            "1500.0",
            "--emit-md",
            "--emit-rd",
        ])
        .unwrap();

        assert_eq!(args.sigma_3d, 2.5);
        assert_eq!(args.sigma_fa, 0.8);
        assert_eq!(args.bvalue, 1500.0);
        assert!(args.emit_md);
        assert!(!args.emit_ad);
        assert!(args.emit_rd);
    }
}
