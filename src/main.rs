mod dti;
mod imaging;

use crate::dti::axial_diffusivity;
use crate::dti::fractional_anisotropy;
use crate::dti::mean_diffusivity;
use crate::dti::radial_diffusivity;
use crate::dti::{compute_scalar_map, fit_tensor_field};
use crate::imaging::mask::generate_otsu_mask;
use crate::imaging::nifti::Volume;
use crate::imaging::nifti::{load_nifti, save_nifti};
use crate::imaging::smooth::gaussian_smooth_3d_anisotropic;
use clap::Parser;
use nifti::NiftiHeader;
use rayon::prelude::*;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

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

fn find_nii_gz_files(root: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();

    let entries = match fs::read_dir(root) {
        Ok(e) => e,
        Err(err) => {
            eprintln!("Warning: Skipping directory {}: {}", root.display(), err);
            return files;
        }
    };

    for entry in entries.flatten() {
        let path = entry.path();

        if path.is_dir() {
            files.extend(find_nii_gz_files(&path));
        } else {
            let filename = path.file_name().unwrap_or_default().to_string_lossy();

            // Match .nii.gz files, but ignore previously generated outputs
            if filename.ends_with(".nii.gz") && !filename.contains("_processed") {
                files.push(path);
            }
        }
    }

    files
}

// Helper to attempt reading associated .bvec gradient files for 4D DWI scans
fn load_gradients(nifti_path: &Path, n_dirs: usize) -> Vec<[f32; 3]> {
    let bvec_path = nifti_path.with_extension("").with_extension("bvec");
    if bvec_path.exists()
        && let Ok(content) = fs::read_to_string(&bvec_path)
    {
        let lines: Vec<&str> = content.lines().collect();
        if lines.len() >= 3 {
            let xs: Vec<f32> = lines[0]
                .split_whitespace()
                .filter_map(|s| s.parse().ok())
                .collect();
            let ys: Vec<f32> = lines[1]
                .split_whitespace()
                .filter_map(|s| s.parse().ok())
                .collect();
            let zs: Vec<f32> = lines[2]
                .split_whitespace()
                .filter_map(|s| s.parse().ok())
                .collect();

            if xs.len() == n_dirs && ys.len() == n_dirs && zs.len() == n_dirs {
                return (0..n_dirs).map(|i| [xs[i], ys[i], zs[i]]).collect();
            }
        }
    }

    // Fallback synthetic gradient directions if .bvec file is missing
    let mut gradients = vec![[0.0, 0.0, 0.0]]; // Index 0: baseline S0 (b=0)
    for i in 1..n_dirs {
        let theta = i as f32 * 0.5;
        let phi = i as f32 * 0.3;
        gradients.push([theta.cos() * phi.sin(), theta.sin() * phi.sin(), phi.cos()]);
    }
    gradients
}

fn get_processed_output_path(file: &Path, suffix: &str) -> PathBuf {
    let filename = file.file_name().unwrap().to_string_lossy();

    // Strip trailing .nii.gz or .nii
    let stem = if filename.ends_with(".nii.gz") {
        &filename[..filename.len() - 7]
    } else if filename.ends_with(".nii") {
        &filename[..filename.len() - 4]
    } else {
        &filename
    };

    file.with_file_name(format!("{}_{}.nii.gz", stem, suffix))
}

fn process_file(file: &Path, args: &Cli) {
    let start = Instant::now();
    let file_str = file.to_str().unwrap();

    let vol = load_nifti(file_str);
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
            .expect("Expected 4D array shape for DWI dataset");

        let b0 = dwi.index_axis(ndarray::Axis(3), 0).to_owned();
        let mask = generate_otsu_mask(&b0, 256);
        let gradients = load_gradients(file, shape[3]);

        let field = fit_tensor_field(&dwi, &gradients, args.bvalue, Some(&mask))
            .expect("DTI tensor fit failed");

        let mut fa_map = compute_scalar_map(&field, fractional_anisotropy);
        gaussian_smooth_3d_anisotropic(&mut fa_map, args.sigma_fa, zooms);
        save_scalar_map(&fa_map, file, "fa_processed", &vol.header);

        if args.emit_md {
            let mut md_map = compute_scalar_map(&field, mean_diffusivity);
            gaussian_smooth_3d_anisotropic(&mut md_map, args.sigma_fa, zooms);
            save_scalar_map(&md_map, file, "md_processed", &vol.header);
        }
        if args.emit_ad {
            let mut ad_map = compute_scalar_map(&field, axial_diffusivity);
            gaussian_smooth_3d_anisotropic(&mut ad_map, args.sigma_fa, zooms);
            save_scalar_map(&ad_map, file, "ad_processed", &vol.header);
        }
        if args.emit_rd {
            let mut rd_map = compute_scalar_map(&field, radial_diffusivity);
            gaussian_smooth_3d_anisotropic(&mut rd_map, args.sigma_fa, zooms);
            save_scalar_map(&rd_map, file, "rd_processed", &vol.header);
        }
    } else {
        let mut struct_vol = vol
            .data
            .into_dimensionality::<ndarray::Ix3>()
            .expect("Expected 3D array shape for structural dataset");

        let mask = generate_otsu_mask(&struct_vol, 256);
        gaussian_smooth_3d_anisotropic(&mut struct_vol, args.sigma_3d, zooms);

        struct_vol.zip_mut_with(&mask, |val, &m| {
            if m == 0.0 {
                *val = 0.0;
            }
        });

        save_scalar_map(&struct_vol, file, "smoothed_processed", &vol.header);
    }

    let elapsed = start.elapsed();
    println!(
        "Processed: {} ({:.3}s)",
        file.file_name().unwrap().to_string_lossy(),
        elapsed.as_secs_f32()
    );
}

fn save_scalar_map(
    map: &ndarray::Array3<f32>,
    source_file: &Path,
    suffix: &str,
    header: &NiftiHeader,
) {
    let output_vol = Volume {
        data: map.clone().into_dyn(),
        header: header.clone(),
    };
    let output_path = get_processed_output_path(source_file, suffix);
    save_nifti(&output_vol, output_path.to_str().unwrap());
}

fn main() {
    let args = Cli::parse();
    let root = Path::new(&args.root);
    let all_start = Instant::now();

    let files = find_nii_gz_files(root);
    let files_processed = AtomicUsize::new(0);

    println!("Found {} NIfTI files", files.len());
    println!("Processing dataset in parallel with Rayon...");

    // Parallelize pipeline across files concurrently
    files.par_iter().for_each(|file| {
        process_file(file, &args);
        files_processed.fetch_add(1, Ordering::Relaxed);
    });

    let all_elapsed = all_start.elapsed();
    println!("----------------------------------------");
    println!(
        "Files processed: {}",
        files_processed.load(Ordering::Relaxed)
    );
    println!(
        "All processing completed in {:.3} seconds",
        all_elapsed.as_secs_f32()
    );
}
