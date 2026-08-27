mod dti;
mod imaging;

use clap::Parser;
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
        mask::generate_otsu_mask, nifti::load_nifti, smooth::gaussian_smooth_3d_anisotropic,
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
fn load_gradients(nifti_path: &Path, n_dirs: usize) -> Result<Vec<[f32; 3]>, ProcessError> {
    let bvec_path = nifti_path.with_extension("").with_extension("bvec");

    // Try reading .bvec file
    if bvec_path.exists() {
        let content = fs::read_to_string(&bvec_path).map_err(ProcessError::Io)?;
        let lines: Vec<&str> = content.lines().collect();

        if lines.len() >= 3 {
            let parse_line = |line: &str| -> Result<Vec<f32>, ProcessError> {
                line.split_whitespace()
                    .map(|s| {
                        s.parse::<f32>().map_err(|e| {
                            ProcessError::Io(std::io::Error::new(
                                std::io::ErrorKind::InvalidData,
                                format!("Invalid float in {}: {}", bvec_path.display(), e),
                            ))
                        })
                    })
                    .collect()
            };

            let xs = parse_line(lines[0])?;
            let ys = parse_line(lines[1])?;
            let zs = parse_line(lines[2])?;

            if xs.len() == n_dirs && ys.len() == n_dirs && zs.len() == n_dirs {
                let grads = (0..n_dirs)
                    .map(|i| [xs[i], ys[i], zs[i]])
                    .collect::<Vec<_>>();
                return Ok(grads);
            }
        }
    }

    // Fallback synthetic gradient directions
    let mut gradients = Vec::with_capacity(n_dirs);
    gradients.push([0.0, 0.0, 0.0]); // baseline S0

    for i in 1..n_dirs {
        let theta = i as f32 * 0.5;
        let phi = i as f32 * 0.3;
        gradients.push([theta.cos() * phi.sin(), theta.sin() * phi.sin(), phi.cos()]);
    }

    Ok(gradients)
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

    // --- find_nii_gz_files tests ---

    #[test]
    fn test_find_nii_gz_files_recurses_and_filters_outputs() -> std::io::Result<()> {
        let dir = tempdir()?;
        let sub_dir = dir.path().join("sub-01");
        fs::create_dir(&sub_dir)?;

        // Valid raw scans
        let scan1 = dir.path().join("raw1.nii.gz");
        let scan2 = sub_dir.join("dwi.nii.gz");
        // Output file to skip
        let processed = sub_dir.join("dwi_fa_processed.nii.gz");
        // Non-NIfTI file
        let txt_file = dir.path().join("notes.txt");

        File::create(&scan1)?;
        File::create(&scan2)?;
        File::create(&processed)?;
        File::create(&txt_file)?;

        let mut found = find_nii_gz_files(dir.path());
        found.sort();

        let mut expected = vec![scan1, scan2];
        expected.sort();

        assert_eq!(found, expected);
        Ok(())
    }

    // --- load_gradients tests ---

    #[test]
    fn test_load_gradients_from_valid_bvec_file() -> std::io::Result<()> {
        let dir = tempdir()?;
        let nifti_path = dir.path().join("dwi.nii.gz");
        let bvec_path = dir.path().join("dwi.bvec");

        File::create(&nifti_path)?;
        let mut f = File::create(&bvec_path)?;
        // Write 3x3 matrix (3 diffusion directions)
        writeln!(f, "1.0  0.0  0.70710677")?;
        writeln!(f, "0.0  1.0  0.0")?;
        writeln!(f, "0.0  0.0  0.70710677")?;

        let grads = load_gradients(&nifti_path, 3).unwrap();
        assert_eq!(grads.len(), 3);
        assert_eq!(grads[0], [1.0, 0.0, 0.0]);
        assert_eq!(grads[1], [0.0, 1.0, 0.0]);
        assert_eq!(
            grads[2],
            [f32::consts::FRAC_1_SQRT_2, 0.0, f32::consts::FRAC_1_SQRT_2]
        );

        Ok(())
    }

    #[test]
    fn test_load_gradients_fallback_when_bvec_missing() {
        let dir = tempdir().unwrap();
        let nifti_path = dir.path().join("dwi_nobvec.nii.gz");

        let grads = load_gradients(&nifti_path, 4).unwrap();
        assert_eq!(grads.len(), 4);

        // First gradient direction must be baseline b=0 [0, 0, 0]
        assert_eq!(grads[0], [0.0, 0.0, 0.0]);

        // Synthetic directions must be normalized / distinct non-zero vectors
        for g in &grads[1..] {
            let norm = (g[0] * g[0] + g[1] * g[1] + g[2] * g[2]).sqrt();
            assert!(norm > 0.0);
        }
    }

    #[test]
    fn test_load_gradients_fallback_on_dimension_mismatch() -> std::io::Result<()> {
        let dir = tempdir()?;
        let nifti_path = dir.path().join("dwi.nii.gz");
        let bvec_path = dir.path().join("dwi.bvec");

        File::create(&nifti_path)?;
        let mut f = File::create(&bvec_path)?;
        // 2 directions in bvec file, but 3 expected
        writeln!(f, "1.0 0.0")?;
        writeln!(f, "0.0 1.0")?;
        writeln!(f, "0.0 0.0")?;

        let grads = load_gradients(&nifti_path, 3).unwrap();
        assert_eq!(grads.len(), 3);
        // Fallback synthetic baseline triggers due to count mismatch
        assert_eq!(grads[0], [0.0, 0.0, 0.0]);

        Ok(())
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
