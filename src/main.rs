mod imaging;
mod ops;

use clap::Parser;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// Root folder containing BIDS dataset
    root: String,

    /// Perfusion alpha scaling factor
    #[arg(short, long, default_value_t = 0.01)]
    alpha: f32,

    /// Gaussian blur sigma (mm) for 3D structural volumes
    #[arg(short = 's', long = "sigma-3d", default_value_t = 1.5)]
    sigma_3d: f32,

    /// Gaussian blur sigma (mm) for 4D FA maps
    #[arg(long = "sigma-fa", default_value_t = 1.0)]
    sigma_fa: f32,

    /// Diffusion b-value (s/mm^2) for DTI tensor fitting
    #[arg(short, long, default_value_t = 1000.0)]
    bvalue: f32,
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

fn main() {
    let args = Cli::parse();
    let root = Path::new(&args.root);
    let all_start = Instant::now();

    let files = find_nii_gz_files(root);
    let mut files_processed = 0;

    println!("Found {} NIfTI files", files.len());
    println!("Processing with alpha = {}", args.alpha);

    for file in files {
        println!("----------------------------------------");
        println!("Processing: {}", file.display());

        let start = Instant::now();
        let file_str = file.to_str().unwrap();

        let mut vol = imaging::load_nifti(file_str);
        let shape = vol.data.shape().to_vec();

        let output_path = if shape.len() == 4 {
            println!("Detected 4D DWI volume ({} gradient steps)", shape[3]);

            // Stage 1: Compute FA map using user-specified b-value
            let gradients = load_gradients(&file, shape[3]);
            let mut fa_vol = ops::compute_dti_fa(&vol, &gradients, args.bvalue);

            // Stage 2: Apply spatial smoothing using user-specified FA sigma
            ops::gaussian_smooth_3d(&mut fa_vol, args.sigma_fa);

            // Stage 3: Apply non-linear perfusion transform
            ops::perfusion_transform(&mut fa_vol, args.alpha);
            vol = fa_vol;

            get_processed_output_path(&file, "fa_processed")
        } else {
            println!(
                "Detected 3D volume (spatial shape: {}x{}x{})",
                shape[0], shape[1], shape[2]
            );

            // Stage 1: Spatial Gaussian Blur using user-specified 3D sigma
            ops::gaussian_smooth_3d(&mut vol, args.sigma_3d);

            // Stage 2: Intensity Transform
            ops::perfusion_transform(&mut vol, args.alpha);

            get_processed_output_path(&file, "smoothed_processed")
        };

        imaging::save_nifti(&vol, output_path.to_str().unwrap());
        files_processed += 1;

        let elapsed = start.elapsed();
        println!("Completed in {:.3} seconds", elapsed.as_secs_f32());
    }

    let all_elapsed = all_start.elapsed();
    println!("Files processed: {}", files_processed);
    println!(
        "All processing completed in {:.3} seconds",
        all_elapsed.as_secs_f32()
    );
}
